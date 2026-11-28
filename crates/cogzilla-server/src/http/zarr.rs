//! `/zarr/*` HTTP routes, unified failure-to-HTTP-status table (#388), and `cogzilla-io`-backed
//! Zarr store adapter.

use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant, UNIX_EPOCH};

use bytes::Bytes;
use cogzilla_kernel::{
    ColormapName, OutputFormat, ResampleMode, SingleBandStyle, encode_npy_typed,
    render_single_band, sample_general,
};
use cogzilla_proj::{CoordPlan, CrsModel, TileCoord, normalize_geographic_lon};
use cogzilla_raster::SampleType;
use cogzilla_zarr::hierarchy::BoxFuture;
use cogzilla_zarr::{
    CfTimeUnits, FsStore, MultiscalePyramid,
    ZarrCaps, ZarrError, ZarrGeo, ZarrHierarchy, ZarrStore, parse_sel_params,
    read_chunk_window_for_band_with_caps, read_coord_1d, resolve_geo, resolve_sel_indices,
    select_variable,
};
use dashmap::DashMap;
use http_body_util::Full;
use hyper::header::{ACCESS_CONTROL_ALLOW_ORIGIN, CACHE_CONTROL, CONTENT_TYPE, ETAG, HeaderValue};
use hyper::{Method, Request, Response, StatusCode};
use serde_json::{Value, json};

use crate::engine::TileEngine;
use crate::geotiff::encode_geotiff_f32;
use crate::stats::encode_rgba_output;
use super::html::append_vary_accept;
use super::{parse_u8_like, parse_u32_like, parse_y_scale_ext, percent_decode};

mod metadata;
mod stats;
use self::metadata::{
    handle_bounds, handle_dataset_coordinate, handle_dataset_dict, handle_dataset_dimensions,
    handle_dataset_keys, handle_info, handle_map_html, handle_tilejson, handle_tileset,
    handle_tileset_list, handle_validate, handle_wmts, require_tms,
};
use self::stats::{handle_get_statistics, handle_post_feature, handle_post_statistics};

/// Global observability counters for `/zarr/*` exposed on `/metrics` (#386).
struct ZarrMetrics {
    stores_opened: AtomicU64,
    chunks_fetched: AtomicU64,
    chunk_holes: AtomicU64,
    bytes_decoded: AtomicU64,
    cap_refusals: AtomicU64,
}

static ZARR_METRICS: ZarrMetrics = ZarrMetrics {
    stores_opened: AtomicU64::new(0),
    chunks_fetched: AtomicU64::new(0),
    chunk_holes: AtomicU64::new(0),
    bytes_decoded: AtomicU64::new(0),
    cap_refusals: AtomicU64::new(0),
};

/// Append Prometheus `/metrics` lines for Zarr reads, holes, decoded bytes, and cap refusals (#386).
pub(crate) fn append_zarr_metrics(out: &mut String) {
    let _ = writeln!(out, "# HELP cogzilla_zarr_stores_opened_total Total Zarr hierarchies opened.");
    let _ = writeln!(out, "# TYPE cogzilla_zarr_stores_opened_total counter");
    let _ = writeln!(
        out,
        "cogzilla_zarr_stores_opened_total {}",
        ZARR_METRICS.stores_opened.load(Ordering::Relaxed)
    );
    let _ = writeln!(out, "# HELP cogzilla_zarr_chunks_fetched_total Total Zarr chunks decoded.");
    let _ = writeln!(out, "# TYPE cogzilla_zarr_chunks_fetched_total counter");
    let _ = writeln!(
        out,
        "cogzilla_zarr_chunks_fetched_total {}",
        ZARR_METRICS.chunks_fetched.load(Ordering::Relaxed)
    );
    let _ = writeln!(out, "# HELP cogzilla_zarr_chunk_holes_total Unwritten Zarr chunk holes short-circuited.");
    let _ = writeln!(out, "# TYPE cogzilla_zarr_chunk_holes_total counter");
    let _ = writeln!(
        out,
        "cogzilla_zarr_chunk_holes_total {}",
        ZARR_METRICS.chunk_holes.load(Ordering::Relaxed)
    );
    let _ = writeln!(out, "# HELP cogzilla_zarr_bytes_decoded_total Resident bytes decoded from Zarr chunks.");
    let _ = writeln!(out, "# TYPE cogzilla_zarr_bytes_decoded_total counter");
    let _ = writeln!(
        out,
        "cogzilla_zarr_bytes_decoded_total {}",
        ZARR_METRICS.bytes_decoded.load(Ordering::Relaxed)
    );
    let _ = writeln!(out, "# HELP cogzilla_zarr_cap_refusals_total Zarr requests refused by fan-out or decode caps.");
    let _ = writeln!(out, "# TYPE cogzilla_zarr_cap_refusals_total counter");
    let _ = writeln!(
        out,
        "cogzilla_zarr_cap_refusals_total {}",
        ZARR_METRICS.cap_refusals.load(Ordering::Relaxed)
    );
    let _ = writeln!(out, "# HELP cogzilla_zarr_hierarchy_cache_entries Cached Zarr store hierarchies.");
    let _ = writeln!(out, "# TYPE cogzilla_zarr_hierarchy_cache_entries gauge");
    let _ = writeln!(
        out,
        "cogzilla_zarr_hierarchy_cache_entries {}",
        zarr_hierarchy_cache().len()
    );
    let _ = writeln!(out, "# HELP cogzilla_zarr_remote_chunk_cache_bytes Resident bytes in the Zarr remote chunk cache.");
    let _ = writeln!(out, "# TYPE cogzilla_zarr_remote_chunk_cache_bytes gauge");
    let _ = writeln!(
        out,
        "cogzilla_zarr_remote_chunk_cache_bytes {}",
        ZARR_REMOTE_CHUNK_CACHE_BYTES.load(Ordering::Relaxed)
    );
}

/// Snapshot of Zarr telemetry and cache occupancy for `GET /stats`.
pub(crate) fn zarr_stats_json() -> Value {
    json!({
        "stores_opened": ZARR_METRICS.stores_opened.load(Ordering::Relaxed),
        "chunks_fetched": ZARR_METRICS.chunks_fetched.load(Ordering::Relaxed),
        "chunk_holes": ZARR_METRICS.chunk_holes.load(Ordering::Relaxed),
        "bytes_decoded": ZARR_METRICS.bytes_decoded.load(Ordering::Relaxed),
        "cap_refusals": ZARR_METRICS.cap_refusals.load(Ordering::Relaxed),
        "hierarchy_cache_entries": zarr_hierarchy_cache().len(),
        "remote_chunk_cache_bytes": ZARR_REMOTE_CHUNK_CACHE_BYTES.load(Ordering::Relaxed),
    })
}

struct CachedZarrStore {
    hierarchy: Arc<ZarrHierarchy>,
    /// Store freshness validator (local `mtime_ns:size` or remote `ETag`/`Last-Modified`).
    validator: String,
    fetched_at: Instant,
    last_used: AtomicU64,
}

struct ZarrHierarchyCache {
    map: DashMap<String, CachedZarrStore>,
    budget: cogzilla_io::ByteBudget,
}

impl ZarrHierarchyCache {
    fn new() -> Self {
        Self {
            map: DashMap::new(),
            budget: cogzilla_io::ByteBudget::new(1024),
        }
    }

    #[inline(always)]
    fn len(&self) -> usize {
        self.map.len()
    }

    #[inline(always)]
    fn clear(&self) {
        self.map.clear();
    }

    #[inline(always)]
    fn touch(&self, entry: &CachedZarrStore) {
        entry.last_used.store(self.budget.touch(), Ordering::Relaxed);
    }

    fn insert_and_evict(&self, key: String, entry: CachedZarrStore, max_entries: usize) {
        self.map.insert(key, entry);
        let limit = max_entries.max(1) as u64;
        if self.map.len() as u64 > limit {
            self.budget.evict_to_limit(self, limit);
        }
    }
}

impl cogzilla_io::EvictTarget for ZarrHierarchyCache {
    type Key = String;

    fn current_usage(&self) -> Option<u64> {
        Some(self.map.len() as u64)
    }

    fn candidates(&self) -> Vec<(u64, Self::Key)> {
        self.map
            .iter()
            .map(|item| (item.value().last_used.load(Ordering::Relaxed), item.key().clone()))
            .collect()
    }

    fn remove_if_cold(&self, key: &Self::Key, expected_stamp: u64) -> Option<u64> {
        self.map
            .remove_if(key, |_, v| v.last_used.load(Ordering::Relaxed) == expected_stamp)
            .map(|_| 1)
    }

    fn remove_now(&self, key: &Self::Key) -> Option<u64> {
        self.map.remove(key).map(|_| 1)
    }
}

fn zarr_hierarchy_cache() -> &'static ZarrHierarchyCache {
    static CACHE: OnceLock<ZarrHierarchyCache> = OnceLock::new();
    CACHE.get_or_init(ZarrHierarchyCache::new)
}

const MAX_ZARR_REMOTE_CHUNK_CACHE_BYTES: usize = 128 * 1024 * 1024;
static ZARR_REMOTE_CHUNK_CACHE_BYTES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

fn zarr_remote_chunk_cache() -> &'static DashMap<String, Vec<u8>> {
    static CACHE: OnceLock<DashMap<String, Vec<u8>>> = OnceLock::new();
    CACHE.get_or_init(DashMap::new)
}

/// Flush all cached Zarr store hierarchies and remote chunks (called by `POST /admin/clear-cache`, #387).
pub(crate) fn clear_zarr_cache() {
    zarr_hierarchy_cache().clear();
    zarr_remote_chunk_cache().clear();
    ZARR_REMOTE_CHUNK_CACHE_BYTES.store(0, Ordering::Relaxed);
}

fn zarr_cache_control_header() -> &'static HeaderValue {
    crate::config::Config::global().zarr_cache_control_header()
}

fn local_store_validator(root: &std::path::Path) -> String {
    // Hash mtime + len of whichever root metadata file exists (plus root dir mtime if present).
    for name in ["zarr.json", ".zmetadata", ".zgroup", ".zarray"] {
        let p = root.join(name);
        if let Ok(meta) = std::fs::metadata(&p) {
            let mtime_ns = meta
                .modified()
                .ok()
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            return format!("fs:{name}:{}:{mtime_ns}", meta.len());
        }
    }
    "fs:none".into()
}

fn record_window_metrics(wb: &cogzilla_zarr::ZarrWindowBlocks) {
    let mut fetched = 0u64;
    let mut holes = 0u64;
    let mut bytes = 0u64;
    for blk in &wb.blocks {
        match blk {
            Some(b) => {
                fetched += 1;
                bytes += b.data.byte_len() as u64;
            }
            None => holes += 1,
        }
    }
    ZARR_METRICS.chunks_fetched.fetch_add(fetched, Ordering::Relaxed);
    ZARR_METRICS.chunk_holes.fetch_add(holes, Ordering::Relaxed);
    ZARR_METRICS.bytes_decoded.fetch_add(bytes, Ordering::Relaxed);
}

/// Unified failure -> HTTP status table for every `/zarr/*` route (#388), delegating to the
/// server-wide [`super::HttpErrorClass`] rule.
///
/// - Every error message names the offending thing (keys looked for, available variables,
///   available range, rank, dimension + index, CRS, codec + supported list, chunk key, or cap).
/// - No `/zarr/*` route constructs an error status inline; all errors pass through this mapping.
pub(crate) fn classify_zarr_error(err: &ZarrError) -> super::HttpErrorClass {
    match err {
        ZarrError::NotAZarrStore { .. }
        | ZarrError::GroupNotFound { .. }
        | ZarrError::VariableNotFound { .. }
        | ZarrError::AmbiguousVariable { .. }
        | ZarrError::InvalidSel { .. }
        | ZarrError::SelOutOfRange { .. }
        | ZarrError::UnsupportedCalendar { .. }
        | ZarrError::UnsupportedRank(_)
        | ZarrError::UnsupportedSampleType(_)
        | ZarrError::UnsupportedVersion(_)
        | ZarrError::IrregularGrid { .. }
        | ZarrError::UnsupportedCrs { .. }
        | ZarrError::InvalidSelection { .. }
        | ZarrError::CapExceeded { .. }
        | ZarrError::ChunkTooLarge { .. }
        | ZarrError::MissingField { .. }
        | ZarrError::BadField { .. }
        | ZarrError::Json(_) => super::HttpErrorClass::ClientFault,

        ZarrError::NotFound { .. } => super::HttpErrorClass::NotFound,
        ZarrError::UpstreamRefused { .. } => super::HttpErrorClass::Forbidden,
        ZarrError::UpstreamTimeout { .. } => super::HttpErrorClass::UpstreamTimeout,

        ZarrError::UnsupportedCodec { .. } => super::HttpErrorClass::NotImplemented,

        ZarrError::MissingChunk { .. }
        | ZarrError::Codec { .. }
        | ZarrError::UpstreamError { .. }
        | ZarrError::Io { .. } => super::HttpErrorClass::UpstreamFault,
    }
}

pub fn zarr_error_status(err: &ZarrError) -> StatusCode {
    classify_zarr_error(err).status()
}

/// Convert any [`ZarrError`] into a JSON HTTP error response using [`zarr_error_status`].
pub fn zarr_error_to_response(err: &ZarrError) -> Response<Full<Bytes>> {
    if matches!(err, ZarrError::CapExceeded { .. } | ZarrError::ChunkTooLarge { .. }) {
        ZARR_METRICS.cap_refusals.fetch_add(1, Ordering::Relaxed);
    }
    let status = zarr_error_status(err);
    let body = serde_json::to_vec(&json!({ "detail": err.to_string() })).unwrap_or_default();
    let mut resp = super::plain_resp(status, Bytes::from(body), "application/json");
    append_vary_accept(resp.headers_mut());
    resp
}

fn json_ok(val: &Value, content_type: &'static str) -> Response<Full<Bytes>> {
    let body = serde_json::to_vec(val).unwrap_or_default();
    let mut resp = super::plain_resp(StatusCode::OK, Bytes::from(body), content_type);
    append_vary_accept(resp.headers_mut());
    resp
}

/// Store adapter bridging local filesystem paths (`file://` or bare path) and any remote URI
/// supported by `cogzilla-io` (`http://`, `https://`, `s3://`, `gs://`, `az://`, `abfs://`,
/// `abfss://`, `/vsi*`) into [`ZarrStore`] using the shared [`cogzilla_io::CogStore`] client pool,
/// cloud credential chain, allowlist, AIMD concurrency limiter, and `IoMetrics`.
pub enum ServerZarrStore {
    Local(FsStore),
    Remote {
        base_uri: String,
        io: Arc<cogzilla_io::CogStore>,
    },
}

fn is_remote_zarr_uri(uri: &str) -> bool {
    uri.starts_with("http://")
        || uri.starts_with("https://")
        || uri.starts_with("s3://")
        || uri.starts_with("gs://")
        || uri.starts_with("az://")
        || uri.starts_with("abfs://")
        || uri.starts_with("abfss://")
        || uri.starts_with("/vsicurl/")
        || uri.starts_with("/vsis3/")
        || uri.starts_with("/vsigs/")
        || uri.starts_with("/vsiaz/")
}

fn map_io_err<T>(key: String, err: cogzilla_io::IoError) -> Result<Option<T>, ZarrError> {
    match err.root() {
        cogzilla_io::IoError::HttpStatus { status: 404, .. } => Ok(None),
        cogzilla_io::IoError::HttpStatus {
            status: status @ (401 | 403),
            ..
        } => Err(ZarrError::UpstreamRefused {
            key,
            status: *status,
        }),
        cogzilla_io::IoError::Forbidden(_) => Err(ZarrError::UpstreamRefused { key, status: 403 }),
        cogzilla_io::IoError::Http(e) if e.is_timeout() => Err(ZarrError::UpstreamTimeout {
            key,
            detail: e.to_string(),
        }),
        cogzilla_io::IoError::Fs(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        other => Err(ZarrError::UpstreamError {
            key,
            detail: other.to_string(),
        }),
    }
}

impl ServerZarrStore {
    pub fn from_url(raw_url: &str, io: Arc<cogzilla_io::CogStore>) -> Result<Self, ZarrError> {
        let trimmed = raw_url.trim();
        if trimmed.is_empty() {
            return Err(ZarrError::InvalidSelection {
                detail: "missing required query parameter 'url'".into(),
            });
        }
        // Honour COGZILLA_ALLOWED_URL_PREFIXES allowlist consistently with /cog/*.
        if cogzilla_io::check_uri_allowed(trimmed).is_err() {
            return Err(ZarrError::UpstreamRefused {
                key: trimmed.to_string(),
                status: 403,
            });
        }
        if is_remote_zarr_uri(trimmed) {
            return Ok(Self::Remote {
                base_uri: trimmed.trim_end_matches('/').to_string(),
                io,
            });
        }
        if let Some(scheme_pos) = trimmed.find("://") {
            let scheme = &trimmed[..scheme_pos];
            if scheme != "file" {
                return Err(ZarrError::InvalidSelection {
                    detail: format!("unsupported Zarr URL scheme '{scheme}://'"),
                });
            }
        }
        let local_path = match cogzilla_io::strip_file_uri_prefix(trimmed) {
            Ok(Some(p)) => p,
            Ok(None) => trimmed,
            Err(()) => {
                return Err(ZarrError::InvalidSelection {
                    detail: format!("invalid file:// URI {trimmed:?}"),
                });
            }
        };
        let p = PathBuf::from(local_path);
        if !p.exists() {
            return Err(ZarrError::NotFound {
                key: trimmed.to_string(),
            });
        }
        Ok(Self::Local(FsStore::new(p)))
    }
}

impl ZarrStore for ServerZarrStore {
    fn get<'a>(&'a self, key: &'a str) -> BoxFuture<'a, Result<Option<Vec<u8>>, ZarrError>> {
        match self {
            Self::Local(fs) => fs.get(key),
            Self::Remote { base_uri, io } => {
                let uri = format!("{base_uri}/{}", key.trim_start_matches('/'));
                let io = Arc::clone(io);
                let key_owned = key.to_string();
                Box::pin(async move {
                    if let Some(cached) = zarr_remote_chunk_cache().get(&uri) {
                        return Ok(Some(cached.value().clone()));
                    }
                    match io
                        .fetch_uri_bytes(&uri, cogzilla_zarr::MAX_DECODED_CHUNK_BYTES)
                        .await
                    {
                        Ok(bytes) => {
                            let vec = bytes.to_vec();
                            let len = vec.len();
                            if len <= 16 * 1024 * 1024 {
                                let cache = zarr_remote_chunk_cache();
                                let prev = ZARR_REMOTE_CHUNK_CACHE_BYTES.load(Ordering::Relaxed);
                                if prev.saturating_add(len) > MAX_ZARR_REMOTE_CHUNK_CACHE_BYTES {
                                    cache.clear();
                                    ZARR_REMOTE_CHUNK_CACHE_BYTES.store(0, Ordering::Relaxed);
                                }
                                if cache.insert(uri, vec.clone()).is_none() {
                                    ZARR_REMOTE_CHUNK_CACHE_BYTES.fetch_add(len, Ordering::Relaxed);
                                }
                            }
                            Ok(Some(vec))
                        }
                        Err(err) => map_io_err(key_owned, err),
                    }
                })
            }
        }
    }

    fn get_range<'a>(
        &'a self,
        key: &'a str,
        start: u64,
        end: u64,
    ) -> BoxFuture<'a, Result<Option<Vec<u8>>, ZarrError>> {
        match self {
            Self::Local(fs) => fs.get_range(key, start, end),
            Self::Remote { base_uri, io } => {
                let uri = format!("{base_uri}/{}", key.trim_start_matches('/'));
                let io = Arc::clone(io);
                let key_owned = key.to_string();
                Box::pin(async move {
                    if start >= end {
                        return Ok(Some(Vec::new()));
                    }
                    match io.fetch_uri_range(&uri, start, end).await {
                        Ok(bytes) => Ok(Some(bytes.to_vec())),
                        Err(err) => map_io_err(key_owned, err),
                    }
                })
            }
        }
    }

    fn size<'a>(&'a self, key: &'a str) -> BoxFuture<'a, Result<Option<u64>, ZarrError>> {
        match self {
            Self::Local(fs) => fs.size(key),
            Self::Remote { base_uri, io } => {
                let uri = format!("{base_uri}/{}", key.trim_start_matches('/'));
                let io = Arc::clone(io);
                let key_owned = key.to_string();
                Box::pin(async move {
                    match io.fetch_uri_size(&uri).await {
                        Ok(size) => Ok(size),
                        Err(err) => map_io_err(key_owned, err),
                    }
                })
            }
        }
    }

    fn list_dir<'a>(
        &'a self,
        prefix: &'a str,
    ) -> BoxFuture<'a, Result<Option<Vec<String>>, ZarrError>> {
        match self {
            Self::Local(fs) => fs.list_dir(prefix),
            Self::Remote { .. } => Box::pin(async { Ok(None) }),
        }
    }
}

/// Parsed `/zarr/*` query parameters, projected from the shared [`super::query::parse_zarr_request`]
/// parser so all 12 common parameters follow the exact same rules as `/cog`, `/stac`,
/// `/mosaicjson`, and `/searches`.
#[derive(Debug)]
struct ZarrQuery {
    url: Option<String>,
    variable: Option<String>,
    group: Option<String>,
    sel: Vec<String>,
    bidx: Option<u64>,
    decode_times: bool,
    show_times: bool,
    rescale: Option<(f32, f32)>,
    colormap: ColormapName,
    resample: ResampleMode,
    nodata: Option<f32>,
    unscale: bool,
    return_mask: bool,
    max_size: Option<u32>,
    tilesize: Option<u32>,
    format: Option<String>,
    coord_crs: Option<u16>,
}

impl ZarrQuery {
    fn from_parts(common: super::query::RequestParams, zp: super::query::ZarrParams) -> Self {
        let has_key = |names: &[&str]| common.keys.iter().any(|k| names.contains(&k.as_str()));
        Self {
            url: common.urls.last().cloned(),
            variable: zp.variable,
            group: zp.group,
            sel: zp.sel,
            bidx: zp.bidx,
            decode_times: zp.decode_times,
            show_times: zp.show_times,
            rescale: common.rescale.last().copied(),
            colormap: common.colormap,
            resample: zp.resample,
            nodata: common.nodata,
            unscale: common.unscale,
            return_mask: common.return_mask,
            max_size: has_key(&["max_size"])
                .then_some(if common.max_size_zero { 0 } else { common.max_size }),
            tilesize: zp.tilesize,
            format: common.format_param,
            coord_crs: zp.coord_crs_raw_epsg,
        }
    }

    fn resample_mode(&self) -> ResampleMode {
        self.resample
    }

    fn colormap(&self) -> ColormapName {
        self.colormap
    }
}

fn parse_output_format(ext: Option<&str>, q_fmt: Option<&str>) -> (OutputFormat, &'static str) {
    let raw = ext.or(q_fmt).unwrap_or("png").to_ascii_lowercase();
    match raw.as_str() {
        "tif" | "tiff" => (OutputFormat::RawF32, "image/tiff; application=geotiff"),
        // Zarr encoders treat `OutputFormat::RawF32` as GeoTIFF, so do not route raw binary
        // extensions through `parse_ext_format`'s `RawF32` sentinel.
        "f32" | "bin" | "raw" => (OutputFormat::Png, "image/png"),
        other => super::parse_ext_format(other, Some(80.0), 85)
            .unwrap_or((OutputFormat::Png, "image/png")),
    }
}

fn binary_ok_with_etag_hdr(
    body: Bytes,
    content_type: &'static str,
    etag_hdr: HeaderValue,
    if_none_match: Option<&str>,
) -> Response<Full<Bytes>> {
    let etag = etag_hdr.to_str().unwrap_or("");
    let cc = zarr_cache_control_header().clone();
    if let Some(inm) = if_none_match
        && super::etag_matches(inm, etag)
    {
        return Response::builder()
            .status(StatusCode::NOT_MODIFIED)
            .header(ETAG, etag_hdr)
            .header(CACHE_CONTROL, cc)
            .header(ACCESS_CONTROL_ALLOW_ORIGIN, HeaderValue::from_static("*"))
            .body(Full::new(Bytes::new()))
            .unwrap();
    }
    Response::builder()
        .status(StatusCode::OK)
        .header(CONTENT_TYPE, HeaderValue::from_static(content_type))
        .header(ETAG, etag_hdr)
        .header(CACHE_CONTROL, cc)
        .header(ACCESS_CONTROL_ALLOW_ORIGIN, HeaderValue::from_static("*"))
        .body(Full::new(body))
        .unwrap()
}

fn binary_ok_with_etag(
    body: Bytes,
    content_type: &'static str,
    etag_key: &str,
    if_none_match: Option<&str>,
) -> Response<Full<Bytes>> {
    let etag_hdr = crate::tile_cache::compute_tile_etag(Some(etag_key), &body);
    binary_ok_with_etag_hdr(body, content_type, etag_hdr, if_none_match)
}

fn binary_ok_from_cached_tile(
    tile: crate::tile_cache::CachedTile,
    if_none_match: Option<&str>,
) -> Response<Full<Bytes>> {
    let etag_hdr = tile
        .etag
        .unwrap_or_else(|| crate::tile_cache::compute_tile_etag(None, &tile.bytes));
    binary_ok_with_etag_hdr(tile.bytes, tile.content_type, etag_hdr, if_none_match)
}

fn zarr_err_to_engine_err(err: ZarrError) -> crate::engine::EngineError {
    use crate::engine::EngineError;
    if matches!(err, ZarrError::CapExceeded { .. } | ZarrError::ChunkTooLarge { .. }) {
        ZARR_METRICS.cap_refusals.fetch_add(1, Ordering::Relaxed);
    }
    match err {
        ZarrError::NotFound { key } => EngineError::NotFound(key),
        ZarrError::UpstreamRefused { .. } => {
            EngineError::Io(cogzilla_io::IoError::Forbidden(err.to_string()))
        }
        other => EngineError::BadRequest(other.to_string()),
    }
}

/// Main dispatcher for all `/zarr/*` routes.
pub(super) async fn handle<B>(
    ctx: super::RequestCtx<'_>,
    method: Method,
    sub: &[&str],
    pq: super::query::RequestParams,
    zp: super::query::ZarrParams,
    req: Request<B>,
) -> super::Resp
where
    B: hyper::body::Body,
    B::Error: std::fmt::Display,
{
    let super::RequestCtx {
        engine,
        path,
        raw_query,
        host,
        forwarded_prefix: _,
        if_none_match,
    } = ctx;
    let post_body = if method == Method::POST {
        match super::read_limited_body(req.into_body()).await {
            Ok(b) => Some(b),
            Err(e) => return super::error_to_http_response(&e),
        }
    } else {
        None
    };
    let io = &engine.store;
    let is_head = method == Method::HEAD;
    let dispatch_method = if is_head { Method::GET } else { method };
    let q = ZarrQuery::from_parts(pq, zp);
    let result = match (dispatch_method, sub) {
        // Metadata & introspection family (#372)
        (Method::GET, ["info"]) => handle_info(io, &q, false).await,
        (Method::GET, ["info.geojson"]) => handle_info(io, &q, true).await,
        (Method::GET, ["bounds"]) => handle_bounds(io, &q).await,
        (Method::GET, ["validate"]) => handle_validate(io, &q).await,
        (Method::GET, ["dataset"]) | (Method::GET, ["dataset", "keys"]) => {
            handle_dataset_keys(io, &q).await
        }
        (Method::GET, ["dataset", "dict"]) => handle_dataset_dict(io, &q).await,
        (Method::GET, ["dataset", "dimensions"]) => handle_dataset_dimensions(io, &q).await,
        (Method::GET, ["dataset", "coordinates", name]) => {
            handle_dataset_coordinate(io, &q, &percent_decode(name)).await
        }

        // Raster family (#373): OGC Tilesets, TileJSON, Map HTML, WMTS, Tiles, Point, Preview, BBox, GET Statistics
        (Method::GET, ["tiles"]) => handle_tileset_list(io, &q, host, raw_query).await,
        (Method::GET, ["tiles", tms]) => handle_tileset(io, &q, tms, host, raw_query).await,
        (Method::GET, ["tilejson.json"]) => {
            handle_tilejson(io, &q, "WebMercatorQuad", host, raw_query).await
        }
        (Method::GET, [tms, "tilejson.json"]) => handle_tilejson(io, &q, tms, host, raw_query).await,
        (Method::GET, ["map.html"]) => handle_map_html("WebMercatorQuad", host, raw_query),
        (Method::GET, [tms, "map.html"]) => handle_map_html(tms, host, raw_query),
        (Method::GET, ["WMTSCapabilities.xml"]) => {
            handle_wmts(io, &q, "WebMercatorQuad", host, raw_query).await
        }
        (Method::GET, [tms, "WMTSCapabilities.xml"]) => handle_wmts(io, &q, tms, host, raw_query).await,
        (Method::GET, ["tiles", tms, zs, xs, ys]) => {
            handle_tile(engine, &q, path, tms, zs, xs, ys, raw_query, if_none_match).await
        }
        (Method::GET, ["tiles", zs, xs, ys]) => {
            handle_tile(
                engine,
                &q,
                path,
                "WebMercatorQuad",
                zs,
                xs,
                ys,
                raw_query,
                if_none_match,
            )
            .await
        }
        (Method::GET, ["point", coords]) => handle_point(io, &q, &percent_decode(coords)).await,
        (Method::GET, ["preview"]) => handle_preview(io, &q, None, None, raw_query, if_none_match).await,
        (Method::GET, [seg]) if seg.starts_with("preview.") => {
            let ext = seg.strip_prefix("preview.");
            handle_preview(io, &q, None, ext, raw_query, if_none_match).await
        }
        (Method::GET, ["preview", dims_seg]) => match parse_dims_ext(dims_seg) {
            Ok((dims, ext)) => handle_preview(io, &q, Some(dims), ext, raw_query, if_none_match).await,
            Err(e) => Err(e),
        },
        (Method::GET, ["bbox", bbox_seg]) => {
            let (bbox_str, ext) = split_bbox_ext(&percent_decode(bbox_seg));
            handle_bbox(io, &q, &bbox_str, None, ext.as_deref(), raw_query, if_none_match).await
        }
        (Method::GET, ["bbox", bbox_seg, dims_seg]) => match parse_dims_ext(dims_seg) {
            Ok((dims, ext)) => {
                handle_bbox(
                    io,
                    &q,
                    &percent_decode(bbox_seg),
                    Some(dims),
                    ext,
                    raw_query,
                    if_none_match,
                )
                .await
            }
            Err(e) => Err(e),
        },
        (Method::GET, ["statistics"]) => handle_get_statistics(io, &q).await,

        // POST zonal statistics & feature routes (#374)
        (Method::POST, ["statistics"]) => {
            handle_post_statistics(io, &q, post_body.unwrap_or_default()).await
        }
        (Method::POST, ["feature"]) => {
            handle_post_feature(io, &q, None, None, post_body.unwrap_or_default()).await
        }
        (Method::POST, [seg]) if seg.starts_with("feature.") => {
            let ext = seg.strip_prefix("feature.");
            handle_post_feature(io, &q, None, ext, post_body.unwrap_or_default()).await
        }
        (Method::POST, ["feature", dims_seg]) => match parse_dims_ext(dims_seg) {
            Ok((dims, ext)) => {
                handle_post_feature(io, &q, Some(dims), ext, post_body.unwrap_or_default()).await
            }
            Err(e) => Err(e),
        },
        _ => {
            return Response::builder()
                .status(StatusCode::NOT_FOUND)
                .header(CONTENT_TYPE, HeaderValue::from_static("application/json"))
                .header(ACCESS_CONTROL_ALLOW_ORIGIN, HeaderValue::from_static("*"))
                .body(Full::new(Bytes::from(
                    r#"{"detail":"Zarr route not found"}"#,
                )))
                .unwrap();
        }
    };

    let mut resp = match result {
        Ok(resp) => resp,
        Err(err) => zarr_error_to_response(&err),
    };
    if resp.status() == StatusCode::NOT_MODIFIED {
        engine
            .stats
            .http_304_not_modified
            .fetch_add(1, Ordering::Relaxed);
    }
    if is_head {
        use hyper::body::Body as _;
        if let Some(len) = resp.body().size_hint().exact()
            && !resp.headers().contains_key(hyper::header::CONTENT_LENGTH)
        {
            resp.headers_mut().insert(
                hyper::header::CONTENT_LENGTH,
                HeaderValue::from(len),
            );
        }
        *resp.body_mut() = Full::new(Bytes::new());
    }
    resp
}

fn parse_dims_ext(seg: &str) -> Result<((u32, u32), Option<&str>), ZarrError> {
    let (dims_part, ext) = match seg.rfind('.') {
        Some(pos) => (&seg[..pos], Some(&seg[pos + 1..])),
        None => (seg, None),
    };
    let Some((ws, hs)) = dims_part.split_once('x').or_else(|| dims_part.split_once('X')) else {
        return Err(ZarrError::InvalidSelection {
            detail: format!("invalid dimensions segment {seg:?} (expected WxH)"),
        });
    };
    let (Ok(w), Ok(h)) = (ws.parse::<u32>(), hs.parse::<u32>()) else {
        return Err(ZarrError::InvalidSelection {
            detail: format!("invalid width/height in {seg:?}"),
        });
    };
    if w == 0 || h == 0 || w > 2048 || h > 2048 {
        return Err(ZarrError::InvalidSelection {
            detail: format!("dimensions {w}x{h} out of allowed range 1..=2048"),
        });
    }
    Ok(((w, h), ext))
}

fn split_bbox_ext(seg: &str) -> (String, Option<String>) {
    // e.g. "-123,37,-121.5,38.5.png" -> ("-123,37,-121.5,38.5", Some("png"))
    if let Some(pos) = seg.rfind('.') {
        let after = &seg[pos + 1..];
        if !after.is_empty() && after.bytes().all(|b| b.is_ascii_alphabetic()) {
            return (seg[..pos].to_string(), Some(after.to_string()));
        }
    }
    (seg.to_string(), None)
}

async fn open_store_and_hierarchy(
    io: &Arc<cogzilla_io::CogStore>,
    q: &ZarrQuery,
) -> Result<(ServerZarrStore, Arc<ZarrHierarchy>, String), ZarrError> {
    let url = q.url.as_deref().unwrap_or("");
    let store = ServerZarrStore::from_url(url, Arc::clone(io))?;
    let cache = zarr_hierarchy_cache();

    match &store {
        ServerZarrStore::Local(fs) => {
            let validator = local_store_validator(fs.root());
            if let Some(entry) = cache.map.get(url)
                && entry.validator == validator
            {
                cache.touch(&entry);
                return Ok((store, Arc::clone(&entry.hierarchy), validator));
            }
            let hierarchy = Arc::new(ZarrHierarchy::open(&store).await?);
            ZARR_METRICS.stores_opened.fetch_add(1, Ordering::Relaxed);
            let stamp = cache.budget.touch();
            cache.insert_and_evict(
                url.to_string(),
                CachedZarrStore {
                    hierarchy: Arc::clone(&hierarchy),
                    validator: validator.clone(),
                    fetched_at: Instant::now(),
                    last_used: AtomicU64::new(stamp),
                },
                io.max_open_uris(),
            );
            Ok((store, hierarchy, validator))
        }
        ServerZarrStore::Remote { base_uri, io } => {
            let ttl = io.remote_cache_ttl();
            if ttl > Duration::ZERO
                && let Some(entry) = cache.map.get(url)
                && entry.fetched_at.elapsed() < ttl
            {
                cache.touch(&entry);
                return Ok((store, Arc::clone(&entry.hierarchy), entry.validator.clone()));
            }

            // Probe root metadata validator cheaply via shared CogStore HEAD (ETag / Last-Modified)
            let mut remote_validator = String::new();
            for meta_name in ["zarr.json", ".zmetadata"] {
                if let Some(v) = io.head_uri_validator(&format!("{base_uri}/{meta_name}")).await {
                    remote_validator = format!("{meta_name}:{v}");
                    break;
                }
            }

            if !remote_validator.is_empty()
                && let Some(mut entry) = cache.map.get_mut(url)
                && entry.validator == remote_validator
            {
                entry.fetched_at = Instant::now();
                cache.touch(&entry);
                return Ok((store, Arc::clone(&entry.hierarchy), remote_validator));
            }

            let hierarchy = Arc::new(ZarrHierarchy::open(&store).await?);
            ZARR_METRICS.stores_opened.fetch_add(1, Ordering::Relaxed);
            let validator = if remote_validator.is_empty() {
                format!("http:gen:{}", ZARR_METRICS.stores_opened.load(Ordering::Relaxed))
            } else {
                remote_validator
            };
            let stamp = cache.budget.touch();
            cache.insert_and_evict(
                url.to_string(),
                CachedZarrStore {
                    hierarchy: Arc::clone(&hierarchy),
                    validator: validator.clone(),
                    fetched_at: Instant::now(),
                    last_used: AtomicU64::new(stamp),
                },
                io.max_open_uris(),
            );
            Ok((store, hierarchy, validator))
        }
    }
}

// ============================================================================
// Raster route family (#373) — all pixels flow through BandSource + cogzilla-kernel
// ============================================================================

async fn resolve_band_index(
    store: &ServerZarrStore,
    group: &cogzilla_zarr::GroupView<'_>,
    sel_var: &cogzilla_zarr::SelectedVariable<'_>,
    q: &ZarrQuery,
) -> Result<Option<u64>, ZarrError> {
    if sel_var.meta.rank() == 2 {
        return Ok(None);
    }
    if !q.sel.is_empty() {
        let specs = parse_sel_params(&q.sel)?;
        for spec in &specs {
            if sel_var.dims.extra_name.as_deref() == Some(spec.dim.as_str())
                && let Some(coord_meta) = group.array(&spec.dim)
            {
                let vals = read_coord_1d(store, &group.child_prefix(&spec.dim), coord_meta).await?;
                let cf = if q.decode_times {
                    CfTimeUnits::from_meta(coord_meta)?
                } else {
                    None
                };
                let idxs = resolve_sel_indices(spec, &vals, cf.as_ref())?;
                return Ok(idxs.first().map(|&i| i as u64));
            }
        }
    }
    Ok(Some(q.bidx.unwrap_or(0)))
}

/// Build an exact `CoordPlan<N>` from any TMS bounds `[minx, miny, maxx, maxy]` in `dst_crs` into
/// the Zarr array's pixel grid (`geo.crs.model` + `geo.affine`).
fn build_general_tile_plan<const N: usize>(
    dst_crs: &CrsModel,
    dst_bounds: [f64; 4],
    geo: &ZarrGeo,
    ifd_w: u32,
    ifd_h: u32,
) -> Option<CoordPlan<N>> {
    let crs_to_pixel = geo.affine.invert()?;
    let [min_x, min_y, max_x, max_y] = dst_bounds;
    let step_x = (max_x - min_x) / (N as f64);
    let step_y = (max_y - min_y) / (N as f64);

    let mut u = vec![f32::NAN; N * N].into_boxed_slice();
    let mut v = vec![f32::NAN; N * N].into_boxed_slice();
    let mut min_u = f32::INFINITY;
    let mut max_u = f32::NEG_INFINITY;
    let mut min_v = f32::INFINITY;
    let mut max_v = f32::NEG_INFINITY;
    let mut any_in = false;
    let fw = ifd_w as f32;
    let fh = ifd_h as f32;

    let same_3857 = matches!(dst_crs, CrsModel::Epsg3857) && matches!(geo.crs.model, CrsModel::Epsg3857);
    let same_geo = dst_crs.is_wgs84_degrees() && geo.crs.model.is_wgs84_degrees();

    for py in 0..N {
        let dy = max_y - (py as f64 + 0.5) * step_y;
        let row_off = py * N;
        for px in 0..N {
            let dx = min_x + (px as f64 + 0.5) * step_x;
            let (sx, sy) = if same_3857 || same_geo {
                (dx, dy)
            } else {
                let (lon, lat) = dst_crs.to_lonlat_deg(dx, dy);
                if !lon.is_finite() || !lat.is_finite() {
                    continue;
                }
                geo.crs.model.from_lonlat_deg(lon, lat)
            };
            if sx.is_finite() && sy.is_finite() {
                let (col, row) = crs_to_pixel.forward(sx, sy);
                let uu = (col - 0.5) as f32;
                let vv = (row - 0.5) as f32;
                u[row_off + px] = uu;
                v[row_off + px] = vv;
                if uu >= -1.0 && uu <= fw && vv >= -1.0 && vv <= fh {
                    any_in = true;
                    min_u = min_u.min(uu);
                    max_u = max_u.max(uu);
                    min_v = min_v.min(vv);
                    max_v = max_v.max(vv);
                }
            }
        }
    }
    if !any_in {
        return None;
    }
    let src_min_col = (min_u.floor() as i32).max(0);
    let src_max_col = (max_u.floor() as i32)
        .saturating_add(1)
        .min(ifd_w as i32 - 1);
    let src_min_row = (min_v.floor() as i32).max(0);
    let src_max_row = (max_v.floor() as i32)
        .saturating_add(1)
        .min(ifd_h as i32 - 1);
    if src_min_col > src_max_col || src_min_row > src_max_row {
        return None;
    }
    Some(CoordPlan::Exact2D {
        u,
        v,
        src_min_col,
        src_max_col,
        src_min_row,
        src_max_row,
        all_in_bounds: false,
    })
}

/// Compute the ground-resolution downsampling ratio (`tile_res_in_src_units / base_res`)
/// for multiscale Zarr overview selection, properly projecting a 1-pixel step from `dst_crs`
/// into `base_crs` so cross-CRS requests (e.g. `WebMercatorQuad` meters over `EPSG:4326`
/// degrees) do not compare meters against degrees.
fn multiscale_downsample_factor(
    dst_crs: &CrsModel,
    tile_bounds: [f64; 4],
    tile_size: u32,
    base_crs: &CrsModel,
    base_res: f64,
) -> f64 {
    if !base_res.is_finite() || base_res <= 0.0 {
        return 1.0;
    }
    let denom = tile_size.max(1) as f64;
    let step_x = (tile_bounds[2] - tile_bounds[0]).abs() / denom;
    let step_y = (tile_bounds[3] - tile_bounds[1]).abs() / denom;
    let same_crs = (matches!(dst_crs, CrsModel::Epsg3857)
        && matches!(base_crs, CrsModel::Epsg3857))
        || (dst_crs.is_wgs84_degrees() && base_crs.is_wgs84_degrees());
    if same_crs {
        return step_x.min(step_y) / base_res;
    }
    let cx = 0.5 * (tile_bounds[0] + tile_bounds[2]);
    let cy = 0.5 * (tile_bounds[1] + tile_bounds[3]);
    let project = |dx: f64, dy: f64| -> Option<(f64, f64)> {
        let (lon, lat) = dst_crs.to_lonlat_deg(dx, dy);
        if !lon.is_finite() || !lat.is_finite() {
            return None;
        }
        let (sx, sy) = base_crs.from_lonlat_deg(lon, lat);
        (sx.is_finite() && sy.is_finite()).then_some((sx, sy))
    };
    if let (Some((x0, y0)), Some((x1, y1)), Some((x2, y2))) = (
        project(cx, cy),
        project(cx + step_x, cy),
        project(cx, cy + step_y),
    ) {
        let is_geo = base_crs.is_geographic();
        let wrap_dx = |d: f64| {
            if is_geo && d.abs() > 180.0 {
                d - 360.0 * d.signum()
            } else {
                d
            }
        };
        let rx = wrap_dx(x1 - x0).hypot(y1 - y0);
        let ry = wrap_dx(x2 - x0).hypot(y2 - y0);
        let tile_res_src = rx.min(ry);
        if tile_res_src.is_finite() && tile_res_src > 0.0 {
            return tile_res_src / base_res;
        }
    }
    // Degenerate projection (e.g. pole): stay on the finest level rather than over-downsampling.
    1.0
}

#[allow(clippy::too_many_arguments)]
async fn handle_tile(
    engine: &Arc<TileEngine>,
    q: &ZarrQuery,
    path: &str,
    tms_id: &str,
    zs: &str,
    xs: &str,
    ys: &str,
    raw_query: &str,
    if_none_match: Option<&str>,
) -> Result<Response<Full<Bytes>>, ZarrError> {
    let tms = require_tms(tms_id)?;
    let (Some(z), Some(x), Some((y, scale_opt, ext_opt))) =
        (parse_u8_like(zs), parse_u32_like(xs), parse_y_scale_ext(ys))
    else {
        return Err(ZarrError::InvalidSelection {
            detail: format!("invalid tile coordinate z={zs} x={xs} y={ys}"),
        });
    };

    let tile_size = q
        .tilesize
        .unwrap_or_else(|| 256 * scale_opt.unwrap_or(1).clamp(1, 4));
    let max_ts = crate::config::Config::global().max_tilesize;
    if tile_size == 0 || tile_size > max_ts {
        return Err(ZarrError::InvalidSelection {
            detail: format!("tilesize {tile_size} out of allowed range 1..={max_ts}"),
        });
    }

    let Some(tile_bounds) = tms.tile_bounds(z, x, y, 0.0, tile_size) else {
        return Err(ZarrError::InvalidSelection {
            detail: format!("tile {z}/{x}/{y} is out of matrix bounds for {tms_id}"),
        });
    };
    let dst_crs = tms.crs_model().map_err(|e| ZarrError::UnsupportedCrs {
        crs: tms_id.to_string(),
        detail: e.to_string(),
    })?;

    let (store, hierarchy, validator) = open_store_and_hierarchy(&engine.store, q).await?;
    // Honour GeoZarr / OME-Zarr `multiscales` when `?group=` is not explicitly pinned (#375).
    let mut effective_group = q.group.clone();
    if effective_group.is_none()
        && let Ok(root_g) = hierarchy.group(None)
        && let Ok(Some(pyr)) = MultiscalePyramid::discover(&hierarchy, &root_g, q.variable.as_deref())
    {
        let base_lvl = &pyr.levels[0];
        if let Ok(base_g) = hierarchy.group(Some(&base_lvl.group_path))
            && let Some(base_meta) = base_g.array(&base_lvl.variable)
            && let Ok(base_geo) = resolve_geo(&store, &base_g, base_meta).await
        {
            let base_res = base_geo.affine.a.abs().min(base_geo.affine.e.abs()).max(1e-9);
            let req_downsample = multiscale_downsample_factor(
                &dst_crs,
                tile_bounds,
                tile_size,
                &base_geo.crs.model,
                base_res,
            );
            let chosen = pyr.select_for_downsample(req_downsample);
            effective_group = Some(chosen.group_path.clone());
        }
    }
    let sel_var = select_variable(&hierarchy, effective_group.as_deref(), q.variable.as_deref())?;
    let group = hierarchy.group(effective_group.as_deref())?;
    let geo = resolve_geo(&store, &group, sel_var.meta).await?;
    let band_idx = resolve_band_index(&store, &group, &sel_var, q).await?;

    let ifd_w = sel_var.meta.shape[sel_var.dims.x_axis] as u32;
    let ifd_h = sel_var.meta.shape[sel_var.dims.y_axis] as u32;
    let crs_to_pixel = geo.affine.invert().ok_or_else(|| ZarrError::IrregularGrid {
        dim: geo.x_dim.clone(),
        index: 0,
        detail: "singular affine transform".into(),
    })?;

    let (fmt, content_type) = parse_output_format(ext_opt, q.format.as_deref());
    let (rmin, rmax) = q.rescale.unwrap_or((0.0, 255.0));
    let style = SingleBandStyle {
        rescale_min: rmin,
        rescale_max: rmax,
        colormap: q.colormap(),
        resample: q.resample_mode(),
        format: fmt,
    };

    let caps = ZarrCaps::from_env();
    // Key on the full request path + query (which carries extension, format, tilesize, TMS, etc.)
    // plus the store freshness validator — the same seam used by the other four route families.
    let cache_key = format!("{path}?{raw_query}#{validator}");

    let rendered = engine
        .tile_cache
        .get_or_render(cache_key, || async {
            let body = if tile_size == 256 {
                let plan = if tms_id == "WebMercatorQuad" {
                    CoordPlan::<256>::build(
                        &geo.crs.model,
                        &crs_to_pixel,
                        None,
                        ifd_w,
                        ifd_h,
                        TileCoord { z, x, y },
                    )
                } else {
                    build_general_tile_plan::<256>(&dst_crs, tile_bounds, &geo, ifd_w, ifd_h)
                };
                let Some(plan) = plan else {
                    return Err(crate::engine::EngineError::NotFound(format!(
                        "tile {tms_id}/{z}/{x}/{y} outside Zarr dataset bounds"
                    )));
                };
                let (c0, c1, r0, r1) = plan.src_pixel_bounds();
                let pad = 2u32;
                let wx0 = (c0.max(0) as u32).saturating_sub(pad);
                let wy0 = (r0.max(0) as u32).saturating_sub(pad);
                let wx1 = ((c1.max(0) as u32) + 1 + pad).min(ifd_w);
                let wy1 = ((r1.max(0) as u32) + 1 + pad).min(ifd_h);

                let wb = read_chunk_window_for_band_with_caps(
                    &store,
                    &sel_var.store_prefix,
                    sel_var.meta,
                    band_idx,
                    wx0,
                    wy0,
                    wx1,
                    wy1,
                    16,
                    &caps,
                )
                .await
                .map_err(zarr_err_to_engine_err)?;
                let refs = wb.block_refs();
                let mut src = wb.as_band_source(&refs);
                if !q.unscale {
                    src.scale_offset = None;
                }
                if let Some(nd) = q.nodata {
                    src = src.with_nodata_override(Some(nd));
                }
                match fmt {
                    OutputFormat::RawF32 => {
                        let raw = render_single_band::<256>(
                            &plan,
                            &src,
                            &SingleBandStyle {
                                format: OutputFormat::RawF32,
                                ..style
                            },
                        );
                        encode_geotiff_f32(256, 256, &raw, tile_bounds, tms.epsg, q.return_mask)
                    }
                    OutputFormat::Npy => {
                        let raw = render_single_band::<256>(
                            &plan,
                            &src,
                            &SingleBandStyle {
                                format: OutputFormat::RawF32,
                                ..style
                            },
                        );
                        let plane: Vec<f32> = raw
                            .chunks_exact(4)
                            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                            .collect();
                        encode_zarr_npy(&plane, 256, 256, q.return_mask)
                    }
                    _ => render_single_band::<256>(&plan, &src, &style),
                }
            } else {
                // General tile size (e.g. 512x512, @2x, custom tilesize): sample through `sample_general`
                render_zarr_rect_bytes(
                    &store,
                    &sel_var,
                    &geo,
                    band_idx,
                    &dst_crs,
                    tile_bounds,
                    tms.epsg,
                    tile_size,
                    tile_size,
                    q,
                    fmt,
                )
                .await
                .map_err(zarr_err_to_engine_err)?
            };
            Ok((body, content_type))
        })
        .await;

    engine.stats.tiles_served.fetch_add(1, Ordering::Relaxed);
    match rendered {
        Ok(cached) => Ok(binary_ok_from_cached_tile(cached, if_none_match)),
        Err(e) => Ok(super::error_to_http_response(&e)),
    }
}

#[allow(clippy::too_many_arguments)]
async fn render_zarr_rect_bytes(
    store: &ServerZarrStore,
    sel_var: &cogzilla_zarr::SelectedVariable<'_>,
    geo: &ZarrGeo,
    band_idx: Option<u64>,
    dst_crs: &CrsModel,
    dst_bounds: [f64; 4],
    dst_epsg: u16,
    out_w: u32,
    out_h: u32,
    q: &ZarrQuery,
    fmt: OutputFormat,
) -> Result<Bytes, ZarrError> {
    let ifd_w = sel_var.meta.shape[sel_var.dims.x_axis] as u32;
    let ifd_h = sel_var.meta.shape[sel_var.dims.y_axis] as u32;
    let crs_to_pixel = geo.affine.invert().ok_or_else(|| ZarrError::IrregularGrid {
        dim: geo.x_dim.clone(),
        index: 0,
        detail: "singular affine transform".into(),
    })?;

    let [min_x, min_y, max_x, max_y] = dst_bounds;
    let step_x = (max_x - min_x) / (out_w as f64);
    let step_y = (max_y - min_y) / (out_h as f64);
    let same_3857 = matches!(dst_crs, CrsModel::Epsg3857) && matches!(geo.crs.model, CrsModel::Epsg3857);
    let same_geo = dst_crs.is_wgs84_degrees() && geo.crs.model.is_wgs84_degrees();

    let n_px = (out_w as usize) * (out_h as usize);
    let mut u_coords = vec![f32::NAN; n_px];
    let mut v_coords = vec![f32::NAN; n_px];
    let mut min_u = f32::INFINITY;
    let mut max_u = f32::NEG_INFINITY;
    let mut min_v = f32::INFINITY;
    let mut max_v = f32::NEG_INFINITY;
    let mut any_in = false;
    let fw = ifd_w as f32;
    let fh = ifd_h as f32;

    for py in 0..out_h as usize {
        let dy = max_y - (py as f64 + 0.5) * step_y;
        let row_off = py * (out_w as usize);
        for px in 0..out_w as usize {
            let dx = min_x + (px as f64 + 0.5) * step_x;
            let (sx, sy) = if same_3857 || same_geo {
                (dx, dy)
            } else {
                let (lon, lat) = dst_crs.to_lonlat_deg(dx, dy);
                if !lon.is_finite() || !lat.is_finite() {
                    continue;
                }
                geo.crs.model.from_lonlat_deg(lon, lat)
            };
            if sx.is_finite() && sy.is_finite() {
                let (col, row) = crs_to_pixel.forward(sx, sy);
                let uu = (col - 0.5) as f32;
                let vv = (row - 0.5) as f32;
                u_coords[row_off + px] = uu;
                v_coords[row_off + px] = vv;
                if uu >= -1.0 && uu <= fw && vv >= -1.0 && vv <= fh {
                    any_in = true;
                    min_u = min_u.min(uu);
                    max_u = max_u.max(uu);
                    min_v = min_v.min(vv);
                    max_v = max_v.max(vv);
                }
            }
        }
    }

    if !any_in {
        return Err(ZarrError::NotFound {
            key: "requested bbox/tile lies outside Zarr dataset bounds".into(),
        });
    }

    let pad = 2u32;
    let wx0 = (min_u.floor().max(0.0) as u32).saturating_sub(pad);
    let wy0 = (min_v.floor().max(0.0) as u32).saturating_sub(pad);
    let wx1 = ((max_u.ceil().max(0.0) as u32) + 1 + pad).min(ifd_w);
    let wy1 = ((max_v.ceil().max(0.0) as u32) + 1 + pad).min(ifd_h);

    let caps = ZarrCaps::from_env();
    let wb = read_chunk_window_for_band_with_caps(
        store,
        &sel_var.store_prefix,
        sel_var.meta,
        band_idx,
        wx0,
        wy0,
        wx1,
        wy1,
        16,
        &caps,
    )
    .await?;
    let refs = wb.block_refs();
    let mut src = wb.as_band_source(&refs);
    if !q.unscale {
        src.scale_offset = None;
    }
    if let Some(nd) = q.nodata {
        src = src.with_nodata_override(Some(nd));
    }

    let mode = q.resample_mode();
    let mut plane = vec![f32::NAN; n_px];
    for i in 0..n_px {
        let u = u_coords[i];
        let v = v_coords[i];
        if u.is_finite() && v.is_finite() {
            plane[i] = sample_general(&src, u, v, mode);
        }
    }

    Ok(encode_zarr_plane(&plane, out_w, out_h, dst_bounds, dst_epsg, q, fmt))
}

#[allow(clippy::too_many_arguments)]
async fn render_zarr_rect(
    store: &ServerZarrStore,
    sel_var: &cogzilla_zarr::SelectedVariable<'_>,
    geo: &ZarrGeo,
    band_idx: Option<u64>,
    dst_crs: &CrsModel,
    dst_bounds: [f64; 4],
    dst_epsg: u16,
    out_w: u32,
    out_h: u32,
    q: &ZarrQuery,
    fmt: OutputFormat,
    content_type: &'static str,
    etag_key: &str,
    if_none_match: Option<&str>,
) -> Result<Response<Full<Bytes>>, ZarrError> {
    let body = render_zarr_rect_bytes(
        store,
        sel_var,
        geo,
        band_idx,
        dst_crs,
        dst_bounds,
        dst_epsg,
        out_w,
        out_h,
        q,
        fmt,
    )
    .await?;
    Ok(binary_ok_with_etag(body, content_type, etag_key, if_none_match))
}

fn f32_plane_to_le_bytes(plane: &[f32]) -> Vec<u8> {
    let mut raw = Vec::with_capacity(plane.len() * 4);
    for &v in plane {
        raw.extend_from_slice(&v.to_le_bytes());
    }
    raw
}

fn encode_zarr_npy(plane: &[f32], out_w: u32, out_h: u32, return_mask: bool) -> Bytes {
    encode_npy_typed(
        plane,
        1,
        out_h as usize,
        out_w as usize,
        SampleType::Float32,
        None,
        return_mask,
    )
}

fn encode_zarr_plane(
    plane: &[f32],
    out_w: u32,
    out_h: u32,
    dst_bounds: [f64; 4],
    dst_epsg: u16,
    q: &ZarrQuery,
    fmt: OutputFormat,
) -> Bytes {
    match fmt {
        OutputFormat::Npy => encode_zarr_npy(plane, out_w, out_h, q.return_mask),
        OutputFormat::RawF32 => {
            let raw = f32_plane_to_le_bytes(plane);
            encode_geotiff_f32(out_w, out_h, &raw, dst_bounds, dst_epsg, q.return_mask)
        }
        _ => {
            let (rmin, rmax) = q.rescale.unwrap_or((0.0, 255.0));
            let denom = if (rmax - rmin).abs() < 1e-12 {
                1.0
            } else {
                rmax - rmin
            };
            let lut = q.colormap().lut();
            let mut rgba = vec![0u8; plane.len() * 4];
            for (i, &val) in plane.iter().enumerate() {
                if val.is_finite() {
                    let norm = ((val - rmin) / denom).clamp(0.0, 1.0);
                    let idx = (norm * 255.0).round() as usize;
                    let c = lut.rgb[idx];
                    rgba[i * 4] = c[0];
                    rgba[i * 4 + 1] = c[1];
                    rgba[i * 4 + 2] = c[2];
                    rgba[i * 4 + 3] = 255;
                }
            }
            encode_rgba_output(out_w as usize, out_h as usize, &rgba, fmt, q.return_mask)
        }
    }
}

async fn handle_point(io: &Arc<cogzilla_io::CogStore>, q: &ZarrQuery, coords: &str) -> Result<Response<Full<Bytes>>, ZarrError> {
    let Some((xs, ys)) = coords.split_once(',') else {
        return Err(ZarrError::InvalidSelection {
            detail: format!("expected /zarr/point/{{lon}},{{lat}}, got {coords:?}"),
        });
    };
    let (Ok(px), Ok(py)) = (xs.trim().parse::<f64>(), ys.trim().parse::<f64>()) else {
        return Err(ZarrError::InvalidSelection {
            detail: format!("invalid coordinates {coords:?}"),
        });
    };
    let (store, hierarchy, _validator) = open_store_and_hierarchy(io, q).await?;
    let sel_var = select_variable(&hierarchy, q.group.as_deref(), q.variable.as_deref())?;
    let group = hierarchy.group(q.group.as_deref())?;
    let geo = resolve_geo(&store, &group, sel_var.meta).await?;
    let band_idx = resolve_band_index(&store, &group, &sel_var, q).await?;

    let (mut sx, sy) = if let Some(epsg) = q.coord_crs {
        let src_crs = CrsModel::from_epsg(epsg).map_err(|e| ZarrError::UnsupportedCrs {
            crs: format!("EPSG:{epsg}"),
            detail: e.to_string(),
        })?;
        let (lon, lat) = src_crs.to_lonlat_deg(px, py);
        geo.crs.model.from_lonlat_deg(lon, lat)
    } else {
        let lon = normalize_geographic_lon(px).ok_or_else(|| ZarrError::NotFound {
            key: format!("longitude {px} out of range"),
        })?;
        geo.crs.model.from_lonlat_deg(lon, py)
    };

    let ifd_w = sel_var.meta.shape[sel_var.dims.x_axis] as i32;
    let ifd_h = sel_var.meta.shape[sel_var.dims.y_axis] as i32;
    if let Some(min_lon) = geo.crs.model.geographic_0_360_min_lon(&geo.affine, ifd_w as u32)
        && sx < min_lon
    {
        sx += 360.0;
    }

    let crs_to_pixel = geo.affine.invert().ok_or_else(|| ZarrError::IrregularGrid {
        dim: geo.x_dim.clone(),
        index: 0,
        detail: "singular affine transform".into(),
    })?;
    let (col_f, row_f) = crs_to_pixel.forward(sx, sy);
    let col = col_f.floor() as i32;
    let row = row_f.floor() as i32;
    if col < 0 || row < 0 || col >= ifd_w || row >= ifd_h {
        return Err(ZarrError::NotFound {
            key: format!("point ({px}, {py}) outside Zarr dataset bounds"),
        });
    }

    let caps = ZarrCaps::from_env();
    let wb = read_chunk_window_for_band_with_caps(
        &store,
        &sel_var.store_prefix,
        sel_var.meta,
        band_idx,
        col as u32,
        row as u32,
        (col as u32) + 1,
        (row as u32) + 1,
        1,
        &caps,
    )
    .await?;
    let refs = wb.block_refs();
    let mut src = wb.as_band_source(&refs);
    if !q.unscale {
        src.scale_offset = None;
    }
    if let Some(nd) = q.nodata {
        src = src.with_nodata_override(Some(nd));
    }

    let val_json = if sel_var.meta.dtype.needs_f64() {
        let v = src.sample_f64_checked(col, row);
        if v.is_nan() { Value::Null } else { json!(v) }
    } else {
        let v = src.sample_checked(col, row);
        if v.is_nan() { Value::Null } else { json!(v) }
    };

    let body = json!({
        "coordinates": [px, py],
        "values": [val_json],
        "band_names": ["b1"],
    });
    Ok(json_ok(&body, "application/json"))
}

async fn handle_preview(
    io: &Arc<cogzilla_io::CogStore>,
    q: &ZarrQuery,
    dims: Option<(u32, u32)>,
    ext: Option<&str>,
    raw_query: &str,
    if_none_match: Option<&str>,
) -> Result<Response<Full<Bytes>>, ZarrError> {
    let (store, hierarchy, _validator) = open_store_and_hierarchy(io, q).await?;
    let sel_var = select_variable(&hierarchy, q.group.as_deref(), q.variable.as_deref())?;
    let group = hierarchy.group(q.group.as_deref())?;
    let geo = resolve_geo(&store, &group, sel_var.meta).await?;
    let band_idx = resolve_band_index(&store, &group, &sel_var, q).await?;

    let ifd_w = sel_var.meta.shape[sel_var.dims.x_axis] as u32;
    let ifd_h = sel_var.meta.shape[sel_var.dims.y_axis] as u32;
    let (out_w, out_h) = match dims {
        Some((w, h)) => (w, h),
        None => {
            let max_s = q.max_size.unwrap_or(1024);
            if max_s == 0 || (ifd_w <= max_s && ifd_h <= max_s) {
                (ifd_w.min(2048), ifd_h.min(2048))
            } else {
                let scale = (max_s as f64) / (ifd_w.max(ifd_h) as f64);
                (
                    ((ifd_w as f64 * scale).round() as u32).max(1),
                    ((ifd_h as f64 * scale).round() as u32).max(1),
                )
            }
        }
    };

    let (fmt, content_type) = parse_output_format(ext, q.format.as_deref());
    let etag_key = format!("zarr:preview:{out_w}x{out_h}?{raw_query}");
    render_zarr_rect(
        &store,
        &sel_var,
        &geo,
        band_idx,
        &geo.crs.model,
        geo.bounds,
        geo.crs.epsg.unwrap_or(4326),
        out_w,
        out_h,
        q,
        fmt,
        content_type,
        &etag_key,
        if_none_match,
    )
    .await
}

async fn handle_bbox(
    io: &Arc<cogzilla_io::CogStore>,
    q: &ZarrQuery,
    bbox_str: &str,
    dims: Option<(u32, u32)>,
    ext: Option<&str>,
    raw_query: &str,
    if_none_match: Option<&str>,
) -> Result<Response<Full<Bytes>>, ZarrError> {
    let parts: Vec<f64> = bbox_str
        .split(',')
        .filter_map(|s| s.trim().parse::<f64>().ok())
        .collect();
    if parts.len() != 4 || parts.iter().any(|v| !v.is_finite()) {
        return Err(ZarrError::InvalidSelection {
            detail: format!("invalid bbox {bbox_str:?} (expected minx,miny,maxx,maxy)"),
        });
    }
    let (mut minx, miny, mut maxx, maxy) = (parts[0], parts[1], parts[2], parts[3]);
    let dst_epsg = q.coord_crs.unwrap_or(4326);
    let dst_crs = CrsModel::from_epsg(dst_epsg).map_err(|e| ZarrError::UnsupportedCrs {
        crs: format!("EPSG:{dst_epsg}"),
        detail: e.to_string(),
    })?;
    if dst_crs.is_geographic() && minx > maxx {
        maxx += 360.0;
    }
    if minx >= maxx || miny >= maxy {
        return Err(ZarrError::InvalidSelection {
            detail: format!("invalid bbox ordering ({minx},{miny},{maxx},{maxy})"),
        });
    }
    let _ = &mut minx;

    let (store, hierarchy, _validator) = open_store_and_hierarchy(io, q).await?;
    let sel_var = select_variable(&hierarchy, q.group.as_deref(), q.variable.as_deref())?;
    let group = hierarchy.group(q.group.as_deref())?;
    let geo = resolve_geo(&store, &group, sel_var.meta).await?;
    let band_idx = resolve_band_index(&store, &group, &sel_var, q).await?;

    let (out_w, out_h) = dims.unwrap_or((256, 256));
    let (fmt, content_type) = parse_output_format(ext, q.format.as_deref());
    let etag_key = format!("zarr:bbox:{bbox_str}:{out_w}x{out_h}?{raw_query}");
    render_zarr_rect(
        &store,
        &sel_var,
        &geo,
        band_idx,
        &dst_crs,
        [minx, miny, maxx, maxy],
        dst_epsg,
        out_w,
        out_h,
        q,
        fmt,
        content_type,
        &etag_key,
        if_none_match,
    )
    .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use cogzilla_zarr::SUPPORTED_CODECS;

    /// Asserts every single row of #388's failure-to-HTTP-status table: both the status code AND
    /// that the JSON `detail` message names the offending thing.
    #[test]
    fn status_table_covers_every_row_of_issue_388() {
        let cases: Vec<(ZarrError, StatusCode, &str)> = vec![
            (
                ZarrError::NotAZarrStore {
                    searched: "zarr.json, .zmetadata, .zgroup, .zarray",
                },
                StatusCode::BAD_REQUEST,
                "zarr.json",
            ),
            (
                ZarrError::NotFound {
                    key: "s3://bucket/missing.zarr".into(),
                },
                StatusCode::NOT_FOUND,
                "missing.zarr",
            ),
            (
                ZarrError::UpstreamRefused {
                    key: "s3://private/store".into(),
                    status: 403,
                },
                StatusCode::FORBIDDEN,
                "403",
            ),
            (
                ZarrError::UpstreamTimeout {
                    key: "https://example.com/zarr.json".into(),
                    detail: "timed out".into(),
                },
                StatusCode::GATEWAY_TIMEOUT,
                "zarr.json",
            ),
            (
                ZarrError::UpstreamError {
                    key: "https://example.com/c/0/0".into(),
                    detail: "truncated".into(),
                },
                StatusCode::BAD_GATEWAY,
                "c/0/0",
            ),
            (
                ZarrError::AmbiguousVariable {
                    available: vec!["temp".into(), "precip".into()],
                },
                StatusCode::BAD_REQUEST,
                "precip",
            ),
            (
                ZarrError::SelOutOfRange {
                    dim: "time".into(),
                    requested: "2025-01-01".into(),
                    min: "2000-01-01".into(),
                    max: "2000-06-01".into(),
                },
                StatusCode::BAD_REQUEST,
                "2000-06-01",
            ),
            (
                ZarrError::InvalidSel {
                    detail: "conflicting sel methods for dimension \"time\"".into(),
                },
                StatusCode::BAD_REQUEST,
                "conflicting",
            ),
            (
                ZarrError::UnsupportedRank(4),
                StatusCode::BAD_REQUEST,
                "4",
            ),
            (
                ZarrError::IrregularGrid {
                    dim: "x".into(),
                    index: 3,
                    detail: "non-uniform spacing".into(),
                },
                StatusCode::BAD_REQUEST,
                "index 3",
            ),
            (
                ZarrError::UnsupportedCrs {
                    crs: "geostationary".into(),
                    detail: "not modelled".into(),
                },
                StatusCode::BAD_REQUEST,
                "geostationary",
            ),
            (
                ZarrError::UnsupportedCodec {
                    codec: "mystery".into(),
                    chunk_key: "c/0/0".into(),
                    supported: SUPPORTED_CODECS,
                },
                StatusCode::NOT_IMPLEMENTED,
                "zstd",
            ),
            (
                ZarrError::MissingChunk {
                    chunk_key: "temp/c/1/2".into(),
                },
                StatusCode::BAD_GATEWAY,
                "temp/c/1/2",
            ),
            (
                ZarrError::Codec {
                    codec: "sharding_indexed".into(),
                    chunk_key: "data/c/0/0".into(),
                    detail: "crc32c mismatch".into(),
                },
                StatusCode::BAD_GATEWAY,
                "data/c/0/0",
            ),
            (
                ZarrError::CapExceeded {
                    kind: "covering chunk count",
                    computed: 512,
                    cap: 256,
                },
                StatusCode::BAD_REQUEST,
                "512",
            ),
        ];

        for (err, expected_status, expected_substr) in cases {
            let resp = zarr_error_to_response(&err);
            assert_eq!(resp.status(), expected_status, "wrong status for {err}");
            assert!(
                err.to_string().contains(expected_substr),
                "message {:?} missing {:?}",
                err.to_string(),
                expected_substr
            );
        }
    }

    #[test]
    fn encode_zarr_npy_writes_valid_header_and_rio_tiler_f32_mask() {
        let plane = [12.5f32, f32::NAN, -3.25, 0.0];
        let with_mask = encode_zarr_npy(&plane, 2, 2, true);
        let (shape, start) = cogzilla_kernel::f32_npy_shape(&with_mask)
            .expect("encode_zarr_npy must produce a valid <f4 .npy header");
        assert_eq!(shape, vec![2, 2, 2]);
        let vals: Vec<f32> = with_mask[start..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        // Data plane: NaN replaced by 0.0, valid values preserved
        assert_eq!(&vals[0..4], &[12.5, 0.0, -3.25, 0.0]);
        // Mask plane: f32::MAX where valid, f32::MIN where masked
        assert_eq!(&vals[4..8], &[f32::MAX, f32::MIN, f32::MAX, f32::MAX]);

        let no_mask = encode_zarr_npy(&plane, 2, 2, false);
        let (shape_nm, start_nm) = cogzilla_kernel::f32_npy_shape(&no_mask)
            .expect("return_mask=false must produce a valid <f4 .npy header");
        assert_eq!(shape_nm, vec![1, 2, 2]);
        let vals_nm: Vec<f32> = no_mask[start_nm..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(vals_nm, vec![12.5, 0.0, -3.25, 0.0]);
    }

    #[test]
    fn multiscale_downsample_factor_handles_cross_crs_mercator_over_epsg4326() {
        // 128x128 EPSG:4326 base level over 1 deg x 1 deg -> 0.0078125 deg/px
        let base_res = 1.0 / 128.0;
        let tms = require_tms("WebMercatorQuad").unwrap();
        let b_z10 = tms.tile_bounds(10, 163, 396, 0.0, 256).unwrap();
        let b_z5 = tms.tile_bounds(5, 5, 12, 0.0, 256).unwrap();
        let f_z10 = multiscale_downsample_factor(
            &CrsModel::Epsg3857,
            b_z10,
            256,
            &CrsModel::Geographic,
            base_res,
        );
        let f_z5 = multiscale_downsample_factor(
            &CrsModel::Epsg3857,
            b_z5,
            256,
            &CrsModel::Geographic,
            base_res,
        );
        // At z=10 (~152 m/px ~= 0.0011 deg/px), factor must be < 1.0 (selects finest level 0).
        assert!(f_z10 < 1.0, "expected z10 downsample < 1.0, got {f_z10}");
        // At z=5 (~4892 m/px ~= 0.035 deg/px), factor must be > 4.0 (selects 4x coarser overview).
        assert!(f_z5 > 4.0 && f_z5 < 8.0, "expected z5 downsample in (4, 8), got {f_z5}");
    }

    #[test]
    fn hierarchy_cache_evicts_coldest_entry_instead_of_clearing_all() {
        let cache = ZarrHierarchyCache::new();
        let max_open = 3usize;
        let dummy = Arc::new(ZarrHierarchy {
            version: cogzilla_zarr::ZarrVersion::V3,
            open_mode: cogzilla_zarr::OpenMode::V3Consolidated,
            nodes: std::collections::BTreeMap::new(),
        });
        let mk = |stamp: u64| CachedZarrStore {
            hierarchy: Arc::clone(&dummy),
            validator: "v1".into(),
            fetched_at: Instant::now(),
            last_used: AtomicU64::new(stamp),
        };

        for name in ["store_a", "store_b", "store_c"] {
            let stamp = cache.budget.touch();
            cache.insert_and_evict(name.into(), mk(stamp), max_open);
        }
        assert_eq!(cache.len(), max_open);

        // Touch store_a so store_b becomes the coldest entry.
        {
            let a = cache.map.get("store_a").unwrap();
            cache.touch(&a);
        }

        // Inserting the (max_open + 1)-th store evicts only the coldest entry (store_b),
        // leaving max_open (3) warm entries cached instead of flushing to 0.
        let stamp_d = cache.budget.touch();
        cache.insert_and_evict("store_d".into(), mk(stamp_d), max_open);
        assert_eq!(
            cache.len(),
            max_open,
            "opening max_open_uris + 1 stores must leave max_open_uris entries cached"
        );
        assert!(!cache.map.contains_key("store_b"), "coldest entry must be evicted");
        assert!(cache.map.contains_key("store_a"), "recently touched entry must survive");
        assert!(cache.map.contains_key("store_c"));
        assert!(cache.map.contains_key("store_d"));
    }
}
