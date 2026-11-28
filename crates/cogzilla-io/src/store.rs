use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use cogzilla_codec::{
    BandData, DecodedTile, apply_alpha_band_to_tile, apply_mask_to_tile, decode_mask_tile_rows,
    decode_tile_rows_exact as decode_tile_rows,
};
use cogzilla_tiff::{CogIndex, IfdMeta, PlanarConfig, TiffError, parse_cog, parse_cog_sparse};
use dashmap::DashMap;

/// Return the 0-based alpha plane index if this IFD is a `PlanarConfig::Planar` COG with an
/// alpha channel (`ExtraSamples` tag 338 containing 1 or 2, e.g. Gray+Alpha or RGBA).
fn planar_alpha_plane(ifd: &IfdMeta) -> Option<usize> {
    if ifd.planar_config != PlanarConfig::Planar {
        return None;
    }
    cogzilla_codec::alpha_channel_index(ifd.samples_per_pixel as usize, &ifd.extra_samples)
}

/// Lookup result for a paired planar alpha-plane tile or internal mask tile.
enum SidecarTileRange {
    /// Non-empty tile at `(offset, byte_len)`.
    Present(u64, u32),
    /// Sparse (`byte_count == 0`) mask/alpha tile — 100% transparent, zero I/O needed!
    SparseTransparent,
    /// Not applicable (e.g. querying the alpha plane itself) or out of bounds.
    NotApplicable,
}

fn mark_tile_all_nodata(tile: &mut DecodedTile) {
    for b in &mut tile.bands {
        if let BandData::F64(ref mut s) = b.data {
            s.fill(f64::NAN);
        } else {
            b.data.make_f32_mut().fill(f32::NAN);
        }
        b.stats = cogzilla_codec::BlockStats::ALL_NODATA;
    }
}

/// Given a planar color-plane `tile_idx`, return its paired alpha-plane tile range status.
fn planar_alpha_tile_range(ifd: &IfdMeta, tile_idx: u32, alpha_plane: usize) -> SidecarTileRange {
    let per_plane = ((ifd.tiles_across as usize) * (ifd.tiles_down as usize)).max(1);
    let plane = (tile_idx as usize) / per_plane;
    if plane == alpha_plane {
        return SidecarTileRange::NotApplicable;
    }
    let alpha_idx = alpha_plane * per_plane + (tile_idx as usize) % per_plane;
    let Some(&offset) = ifd.tile_offsets.get(alpha_idx) else {
        return SidecarTileRange::NotApplicable;
    };
    let Some(&len) = ifd.tile_byte_counts.get(alpha_idx) else {
        return SidecarTileRange::NotApplicable;
    };
    if len == 0 {
        SidecarTileRange::SparseTransparent
    } else {
        SidecarTileRange::Present(offset, len)
    }
}

fn mask_tile_range(mask_ifd: &cogzilla_tiff::MaskIfdMeta, mask_idx: usize) -> SidecarTileRange {
    let Some(&len) = mask_ifd.tile_byte_counts.get(mask_idx) else {
        return SidecarTileRange::NotApplicable;
    };
    if len == 0 {
        return SidecarTileRange::SparseTransparent;
    }
    match mask_ifd.tile_offsets.get(mask_idx) {
        Some(&off) => SidecarTileRange::Present(off, len),
        None => SidecarTileRange::NotApplicable,
    }
}

/// Lightweight `'static` clone of `IfdMeta` carrying only the fields `decode_tile` reads
/// (omitting large `tile_offsets`/`tile_byte_counts` vectors) so multi-block decodes can fan out
/// across Tokio worker threads with negligible allocation.
fn clone_for_decode(ifd: &IfdMeta) -> IfdMeta {
    IfdMeta {
        width: ifd.width,
        height: ifd.height,
        tile_width: ifd.tile_width,
        tile_height: ifd.tile_height,
        tiles_across: ifd.tiles_across,
        tiles_down: ifd.tiles_down,
        sample_type: ifd.sample_type,
        samples_per_pixel: ifd.samples_per_pixel,
        planar_config: ifd.planar_config,
        compression: ifd.compression,
        predictor: ifd.predictor,
        photometric: ifd.photometric,
        extra_samples: ifd.extra_samples.clone(),
        jpeg_tables: ifd.jpeg_tables.clone(),
        mask_ifd: None,
        is_stripped: ifd.is_stripped,
        pixel_to_crs: ifd.pixel_to_crs,
        crs_to_pixel: ifd.crs_to_pixel,
        gcp_poly: None,
        tile_offsets: Box::default(),
        tile_byte_counts: Box::default(),
    }
}

use crate::backend::{CogSource, IoError, IoMetrics, TileByteReq, coalesce_ranges};
use crate::cache::{BlockKey, InFlightGuard, LookupResult, SingleflightBlockCache};
use crate::disk_cache::DiskBlockCache;
use crate::limiter::AimdLimiter;
use crate::s3::S3Config;

const INITIAL_HEADER_FETCH_BYTES: u64 = 128 * 1024;
const RANGE_GAP_THRESHOLD: u64 = 32 * 1024;
const MAX_COALESCED_SPAN: u64 = 4 * 1024 * 1024;
const DEFAULT_MAX_OPEN_URIS: usize = 4096;
/// Default remote cache revalidation TTL in seconds (`COGZILLA_CACHE_TTL_SECS`).
const DEFAULT_REMOTE_CACHE_TTL_SECS: u64 = 5;

#[cfg(unix)]
fn file_inode(meta: &std::fs::Metadata) -> u64 {
    use std::os::unix::fs::MetadataExt;
    meta.ino()
}

#[cfg(not(unix))]
fn file_inode(_meta: &std::fs::Metadata) -> u64 {
    0
}

/// Source freshness metadata used to detect overwritten/deleted local files and replaced remote
/// objects (`ETag` / `Last-Modified` / bounded TTL).
pub enum SourceFreshness {
    Local {
        path: std::path::PathBuf,
        len: u64,
        mtime: Option<std::time::SystemTime>,
        inode: u64,
    },
    Remote {
        validator: crate::backend::RemoteValidator,
        opened_at: std::time::Instant,
        last_checked_ms: AtomicU64,
        /// Serializes TTL-expired conditional revalidation so N concurrent requests don't fire N
        /// simultaneous `Range: bytes=0-0` GETs when the TTL ticks over.
        revalidate_lock: tokio::sync::Mutex<()>,
    },
    /// Cloud object store (`s3://`, `gs://`, `az://`, `abfs(s)://`): cached without 5s TTL
    /// thrashing (clearable via `/admin/clear-cache` or LRU eviction).
    Cloud,
}

impl SourceFreshness {
    /// Independent copy of these validators (fresh atomics and lock) for a derived view.
    fn snapshot(&self) -> Self {
        match self {
            SourceFreshness::Local { path, len, mtime, inode } => SourceFreshness::Local {
                path: path.clone(),
                len: *len,
                mtime: *mtime,
                inode: *inode,
            },
            SourceFreshness::Remote { validator, opened_at, last_checked_ms, .. } => SourceFreshness::Remote {
                validator: validator.clone(),
                opened_at: *opened_at,
                last_checked_ms: AtomicU64::new(last_checked_ms.load(Ordering::Relaxed)),
                revalidate_lock: tokio::sync::Mutex::new(()),
            },
            SourceFreshness::Cloud => SourceFreshness::Cloud,
        }
    }
}

pub struct OpenedCog {
    /// Identity of these decoded blocks in the block and GPU caches.
    pub id: u64,
    /// `id` of the opened source the URI cache holds: `id` itself, but a `?nodata=` view's
    /// parent's, so a source found changed through a view evicts the parent.
    pub source_id: u64,
    /// Value for tiles the file leaves unwritten (`byte_count == 0`), when GDAL treats them as
    /// valid data rather than no data: `Some(0.0)` when the file declares no `GDAL_NODATA` and no
    /// mask/alpha band, or `Some(file_nodata)` on a `?nodata=` view whose override differs from it.
    pub sparse_fill: Option<f64>,
    pub uri: String,
    pub index: CogIndex,
    pub source: CogSource,
    pub freshness: SourceFreshness,
}

/// Per GDAL `SPARSE_OK` semantics, an unwritten sparse tile (`TileByteCounts == 0`) is filled
/// with the file's `GDAL_NODATA` (masked -> `None`), or with `0.0` as valid data when the file
/// declares no `GDAL_NODATA` and no mask/alpha band.
fn default_sparse_fill(index: &CogIndex) -> Option<f64> {
    if index.nodata_f64.is_some() || index.nodata.is_some() {
        return None;
    }
    let masked = index.ifds.iter().any(|ifd| {
        ifd.mask_ifd.is_some()
            || cogzilla_codec::alpha_channel_index(
                ifd.samples_per_pixel as usize,
                &ifd.extra_samples,
            )
            .is_some()
    });
    (!masked).then_some(0.0)
}

impl OpenedCog {
    pub fn remote_validator(&self) -> Option<&crate::backend::RemoteValidator> {
        match &self.freshness {
            SourceFreshness::Remote { validator, .. } => Some(validator),
            SourceFreshness::Local { .. } | SourceFreshness::Cloud => None,
        }
    }

    /// Key for tier-2 `DiskBlockCache`: incorporates the HTTP `ETag`/`Last-Modified` validator
    /// when present so replaced remote objects never serve stale disk-cached blocks; returns
    /// `None` for local files.
    pub fn disk_cache_uri(&self) -> Option<std::borrow::Cow<'_, str>> {
        match &self.freshness {
            SourceFreshness::Local { .. } => None,
            SourceFreshness::Cloud => Some(std::borrow::Cow::Borrowed(self.uri.as_str())),
            SourceFreshness::Remote { validator, .. } => {
                match validator
                    .etag
                    .as_deref()
                    .or(validator.last_modified.as_deref())
                {
                    Some(tag) => Some(std::borrow::Cow::Owned(format!("{}#v={tag}", self.uri))),
                    // Without an ETag or Last-Modified validator, scope the disk cache key to this
                    // process instance and `OpenedCog::id` so RAM-eviction spills still hit disk
                    // within the TTL window, while TTL re-opens and restarts never serve stale data.
                    None => {
                        static PROCESS_NONCE: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
                        let nonce = *PROCESS_NONCE.get_or_init(|| {
                            let nanos = std::time::SystemTime::now()
                                .duration_since(std::time::UNIX_EPOCH)
                                .map(|d| d.as_nanos() as u64)
                                .unwrap_or(0);
                            nanos ^ ((std::process::id() as u64).rotate_left(32))
                        });
                        Some(std::borrow::Cow::Owned(format!(
                            "{}#g={nonce:x}-{}",
                            self.uri, self.id
                        )))
                    }
                }
            }
        }
    }
}

struct OpenFlight {
    notify: tokio::sync::Notify,
    outcome: std::sync::Mutex<Option<Result<Arc<OpenedCog>, Arc<IoError>>>>,
}

impl OpenFlight {
    fn new() -> Self {
        Self {
            notify: tokio::sync::Notify::new(),
            outcome: std::sync::Mutex::new(None),
        }
    }

    fn take_outcome(&self) -> Option<Result<Arc<OpenedCog>, Arc<IoError>>> {
        self.outcome
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    fn set_outcome(&self, res: Result<Arc<OpenedCog>, Arc<IoError>>) {
        *self.outcome.lock().unwrap_or_else(|e| e.into_inner()) = Some(res);
    }
}

enum OpenSlot {
    Ready {
        cog: Arc<OpenedCog>,
        last_used: AtomicU64,
    },
    InFlight(Arc<OpenFlight>),
}

impl crate::budget::EvictTarget for CogStore {
    type Key = String;

    fn current_usage(&self) -> Option<u64> {
        Some(self.datasets_by_uri.len() as u64)
    }

    fn candidates(&self) -> Vec<(u64, Self::Key)> {
        self.datasets_by_uri
            .iter()
            .filter_map(|item| match item.value() {
                OpenSlot::Ready { last_used, .. } => {
                    Some((last_used.load(Ordering::Relaxed), item.key().clone()))
                }
                OpenSlot::InFlight(_) => None,
            })
            .collect()
    }

    fn remove_if_cold(&self, key: &Self::Key, expected_stamp: u64) -> Option<u64> {
        self.datasets_by_uri
            .remove_if(key, |_, v| {
                matches!(v, OpenSlot::Ready { last_used, .. }
                    if last_used.load(Ordering::Relaxed) == expected_stamp)
            })
            .map(|_| 1)
    }

    fn remove_now(&self, key: &Self::Key) -> Option<u64> {
        self.datasets_by_uri
            .remove_if(key, |_, v| matches!(v, OpenSlot::Ready { .. }))
            .map(|_| 1)
    }
}

/// Manages opened COGs (local mmap + remote HTTP/S3) and the shared singleflight block cache.
pub struct CogStore {
    next_id: AtomicU64,
    datasets_by_name: DashMap<String, Arc<OpenedCog>, ahash::RandomState>,
    datasets_by_uri: DashMap<String, OpenSlot, ahash::RandomState>,
    /// `?nodata=` views keyed by `(parent OpenedCog::id, override bits)`; see [`CogStore::with_nodata`].
    nodata_views: DashMap<(u64, u32), Arc<OpenedCog>, ahash::RandomState>,
    uri_budget: crate::budget::ByteBudget,
    pub cache: Arc<SingleflightBlockCache>,
    pub disk_cache: Option<Arc<DiskBlockCache>>,
    pub upstream_limiter: Arc<AimdLimiter>,
    /// Bounds concurrent heavy CPU offloads (`spawn_blocking` for >2 MiB blocks, large stats/bbox)
    /// to `num_cpus` so pathological workloads never oversubscribe OS threads or starve Tokio I/O.
    pub heavy_cpu_sem: Arc<tokio::sync::Semaphore>,
    pub client: reqwest::Client,
    /// Separate client with redirects disabled for S3/GCS/Azure so 307 region redirects are
    /// handled explicitly (with SigV4 re-signing) instead of failing the HTTP allowlist redirect hook.
    pub cloud_client: reqwest::Client,
    pub s3_cfg: Option<S3Config>,
    pub metrics: Arc<IoMetrics>,
    /// Bounded TTL for remote source cache revalidation (`COGZILLA_CACHE_TTL_SECS`, default 5s).
    pub remote_cache_ttl: Duration,
    /// Monotonic counter incremented whenever any cached source is invalidated or deleted,
    /// allowing downstream caches (e.g. rendered tile cache) to flush stale entries in O(1).
    pub invalidation_epoch: AtomicU64,
}

/// Build default outbound HTTP headers from standard GDAL environment variables:
/// `GDAL_HTTP_BEARER`, `GDAL_HTTP_USERPWD`, `GDAL_HTTP_HEADERS`, and `GDAL_HTTP_HEADER_FILE`.
/// Only attached when `COGZILLA_ALLOWED_URL_PREFIXES` is explicitly set so credentials can
/// never be exfiltrated to arbitrary attacker-controlled origins via SSRF.
fn build_gdal_http_headers() -> reqwest::header::HeaderMap {
    use reqwest::header::{AUTHORIZATION, HeaderMap, HeaderName, HeaderValue};

    let mut headers = HeaderMap::new();
    if !crate::allowlist::allowlist_enabled() {
        return headers;
    }
    let mut push_line = |line: &str| {
        if let Some((k, v)) = line.split_once(':')
            && let (Ok(name), Ok(mut val)) = (
                HeaderName::from_bytes(k.trim().as_bytes()),
                HeaderValue::from_str(v.trim()),
            ) {
                val.set_sensitive(true);
                headers.insert(name, val);
            }
    };

    if let Ok(path) = std::env::var("GDAL_HTTP_HEADER_FILE")
        && !path.trim().is_empty()
            && let Ok(contents) = std::fs::read_to_string(path.trim())
        {
            for line in contents.lines() {
                push_line(line);
            }
        }
    if let Ok(raw) = std::env::var("GDAL_HTTP_HEADERS") {
        for part in raw.split(['\n', '\r']) {
            push_line(part);
        }
    }
    if let Some(bearer) = std::env::var("GDAL_HTTP_BEARER")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
    {
        if let Ok(mut val) = HeaderValue::from_str(&format!("Bearer {bearer}")) {
            val.set_sensitive(true);
            headers.insert(AUTHORIZATION, val);
        }
    } else if let Some(userpwd) = std::env::var("GDAL_HTTP_USERPWD")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
    {
        let b64 = crate::cloud::base64_encode(userpwd.as_bytes());
        if let Ok(mut val) = HeaderValue::from_str(&format!("Basic {b64}")) {
            val.set_sensitive(true);
            headers.insert(AUTHORIZATION, val);
        }
    }
    headers
}

impl CogStore {
    pub fn new(cache_max_bytes: usize) -> Self {
        let max_open_uris = match std::env::var("COGZILLA_MAX_OPEN_URIS") {
            Ok(raw) if !raw.trim().is_empty() => match raw.trim().parse::<usize>() {
                Ok(n) if n > 0 => n,
                _ => {
                    eprintln!(
                        "cogzilla-io: warning: COGZILLA_MAX_OPEN_URIS={raw:?} ignored (expected positive integer; using default {DEFAULT_MAX_OPEN_URIS})"
                    );
                    DEFAULT_MAX_OPEN_URIS
                }
            },
            _ => DEFAULT_MAX_OPEN_URIS,
        };
        Self::with_limits(cache_max_bytes, max_open_uris, Arc::new(AimdLimiter::from_env()))
    }

    /// Configured maximum number of cached open URIs (`COGZILLA_MAX_OPEN_URIS`, default 4096).
    pub fn max_open_uris(&self) -> usize {
        self.uri_budget.max_bytes() as usize
    }

    /// Configured remote cache revalidation TTL (`COGZILLA_CACHE_TTL_SECS`, default 5s).
    pub fn remote_cache_ttl(&self) -> Duration {
        self.remote_cache_ttl
    }

    pub fn with_limits(
        cache_max_bytes: usize,
        max_open_uris: usize,
        upstream_limiter: Arc<AimdLimiter>,
    ) -> Self {
        // Without these a stalled S3 response would hang the tile request forever. The read timeout
        // is per read, not per response, so large range bodies are fine as long as bytes keep flowing.
        let env_secs = |key: &str, default: u64| {
            std::env::var(key)
                .ok()
                .and_then(|v| v.trim().parse::<u64>().ok())
                .filter(|&s| s > 0)
                .unwrap_or(default)
        };
        let connect_timeout = Duration::from_secs(env_secs("GDAL_HTTP_CONNECTTIMEOUT", 10));
        let read_timeout = Duration::from_secs(env_secs("GDAL_HTTP_TIMEOUT", 30));
        let gdal_headers = build_gdal_http_headers();
        let mut client_builder = reqwest::Client::builder()
            .pool_max_idle_per_host(64)
            .tcp_nodelay(true)
            .connect_timeout(connect_timeout)
            .read_timeout(read_timeout)
            .redirect(reqwest::redirect::Policy::custom(|attempt| {
                if attempt.previous().len() >= 10 {
                    return attempt.error("too many HTTP redirects");
                }
                if let Err(e) = crate::allowlist::check_uri_allowed(attempt.url().as_str()) {
                    return attempt.error(e);
                }
                attempt.follow()
            }));
        if !gdal_headers.is_empty() {
            client_builder = client_builder.default_headers(gdal_headers);
        }
        let client = client_builder.build().expect("reqwest client");
        let cloud_client = reqwest::Client::builder()
            .pool_max_idle_per_host(64)
            .tcp_nodelay(true)
            .connect_timeout(connect_timeout)
            .read_timeout(read_timeout)
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .expect("reqwest cloud client");
        // Two-lane fairness: cap slow-lane heavy offloads (>512^2 crops, large stats, >2 MiB
        // block decodes) to half of available cores so fast interactive map tiles always retain
        // at least 50% CPU capacity under mixed-load bursts.
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        let heavy_permits = (cores / 2).max(2).min(cores.max(1));
        let remote_cache_ttl = match std::env::var("COGZILLA_CACHE_TTL_SECS") {
            Ok(raw) if !raw.trim().is_empty() => match raw.trim().parse::<u64>() {
                Ok(secs) => Duration::from_secs(secs),
                Err(_) => {
                    eprintln!(
                        "cogzilla-io: warning: COGZILLA_CACHE_TTL_SECS={raw:?} ignored (expected non-negative integer; using default {DEFAULT_REMOTE_CACHE_TTL_SECS})"
                    );
                    Duration::from_secs(DEFAULT_REMOTE_CACHE_TTL_SECS)
                }
            },
            _ => Duration::from_secs(DEFAULT_REMOTE_CACHE_TTL_SECS),
        };
        Self {
            next_id: AtomicU64::new(1),
            datasets_by_name: DashMap::with_hasher(ahash::RandomState::new()),
            datasets_by_uri: DashMap::with_hasher(ahash::RandomState::new()),
            nodata_views: DashMap::with_hasher(ahash::RandomState::new()),
            uri_budget: crate::budget::ByteBudget::new(max_open_uris.max(1) as u64),
            cache: Arc::new(SingleflightBlockCache::new(cache_max_bytes)),
            disk_cache: None,
            upstream_limiter,
            heavy_cpu_sem: Arc::new(tokio::sync::Semaphore::new(heavy_permits)),
            client,
            cloud_client,
            s3_cfg: Some(S3Config::from_env_full()),
            metrics: Arc::new(IoMetrics::default()),
            remote_cache_ttl,
            invalidation_epoch: AtomicU64::new(0),
        }
    }

    /// Run a heavy synchronous CPU closure on Tokio's blocking pool gated by `heavy_cpu_sem`
    /// (`num_cpus` permits) so large blocks, `/cog/statistics`, and giant bbox crops never stall
    /// Tokio's async I/O reactor threads or oversubscribe the blocking pool.
    pub async fn run_heavy_cpu<F, R>(&self, f: F) -> Result<R, IoError>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        let _permit = self
            .heavy_cpu_sem
            .clone()
            .acquire_owned()
            .await
            .map_err(|e| IoError::Fs(std::io::Error::other(format!("heavy CPU semaphore closed: {e}"))))?;
        tokio::task::spawn_blocking(f)
            .await
            .map_err(|e| IoError::Fs(std::io::Error::other(format!("heavy CPU task failed: {e}"))))
    }

    /// Attach a persistent tier-2 compressed-block disk cache for remote S3/HTTP sources.
    pub fn with_disk_cache(mut self, disk_cache: Arc<DiskBlockCache>) -> Self {
        self.disk_cache = Some(disk_cache);
        self
    }

    #[inline(always)]
    fn uri_tick(&self) -> u64 {
        self.uri_budget.touch()
    }

    /// Number of currently cached opened-COG URI entries.
    pub fn open_uri_count(&self) -> usize {
        self.datasets_by_uri.len()
    }

    pub fn register_local(
        &self,
        name: impl Into<String>,
        path: impl AsRef<Path>,
    ) -> Result<Arc<OpenedCog>, IoError> {
        let uri = path.as_ref().to_string_lossy().to_string();
        let meta = std::fs::metadata(&path)?;
        let source = CogSource::open_local(&path)?;
        let index = match &source {
            CogSource::LocalMmap { mmap } => parse_cog(mmap)?,
            _ => unreachable!(),
        };
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let sparse_fill = default_sparse_fill(&index);
        let cog = Arc::new(OpenedCog {
            id,
            source_id: id,
            sparse_fill,
            uri: uri.clone(),
            index,
            source,
            freshness: SourceFreshness::Local {
                path: path.as_ref().to_path_buf(),
                len: meta.len(),
                mtime: meta.modified().ok(),
                inode: file_inode(&meta),
            },
        });
        self.datasets_by_name.insert(name.into(), cog.clone());
        self.datasets_by_uri.insert(
            uri,
            OpenSlot::Ready {
                cog: cog.clone(),
                last_used: AtomicU64::new(self.uri_tick()),
            },
        );
        Ok(cog)
    }

    /// The COG as read with `nodata` in place of its own `GDAL_NODATA` (GDAL's nodata override,
    /// which TiTiler's `?nodata=` is): pixels equal to `nodata` are masked and pixels equal to
    /// the file's own nodata become valid data. Returns `cog` itself when there is no override or
    /// it matches the file's nodata. The view has its own id, so its decoded blocks are cached
    /// apart from the file's; views are kept per `(cog, nodata)` and dropped in bulk past a bound.
    pub fn with_nodata(&self, cog: &Arc<OpenedCog>, nodata: Option<f32>) -> Arc<OpenedCog> {
        const MAX_NODATA_VIEWS: usize = 256;
        let Some(nd) = nodata else { return cog.clone() };
        // NaN is no data whatever the file says, so a NaN override over no nodata is no change.
        // The file's value is compared at full precision (`nodata` is its f32 rounding).
        let same = match cog.index.nodata_f64.or(cog.index.nodata.map(f64::from)) {
            Some(own) => own == f64::from(nd) || (own.is_nan() && nd.is_nan()),
            None => nd.is_nan(),
        };
        if same {
            return cog.clone();
        }
        let key = (cog.id, if nd.is_nan() { f32::NAN.to_bits() } else { nd.to_bits() });
        if let Some(v) = self.nodata_views.get(&key) {
            return v.clone();
        }
        if self.nodata_views.len() >= MAX_NODATA_VIEWS {
            self.nodata_views.clear();
        }
        // GDAL fills an unwritten tile with the file's nodata (0 with none); under the override
        // that is data unless it is NaN or the override itself. A mask or alpha band still
        // leaves such tiles transparent, so they are left as no data there.
        let fill = cog.index.nodata_f64.or(cog.index.nodata.map(f64::from)).unwrap_or(0.0);
        let masked = cog.index.ifds.iter().any(|ifd| {
            ifd.mask_ifd.is_some()
                || cogzilla_codec::alpha_channel_index(ifd.samples_per_pixel as usize, &ifd.extra_samples).is_some()
        });
        let sparse_fill = (!fill.is_nan() && fill != f64::from(nd) && !masked).then_some(fill);
        let mut index = cog.index.clone();
        index.nodata = Some(nd);
        index.nodata_f64 = Some(f64::from(nd));
        let view = Arc::new(OpenedCog {
            id: self.next_id.fetch_add(1, Ordering::Relaxed),
            source_id: cog.source_id,
            sparse_fill,
            uri: cog.uri.clone(),
            index,
            source: cog.source.clone(),
            freshness: cog.freshness.snapshot(),
        });
        self.nodata_views.entry(key).or_insert(view).clone()
    }

    /// The block of an unwritten tile of `cog`'s IFD `ifd_idx`: `None` (no data), or for a
    /// `?nodata=` view with a [`OpenedCog::sparse_fill`], a tile of that value. The fill is the
    /// parent's, so one tile per parent and IFD serves every view of it; it is kept in the block
    /// cache, under its byte budget, at a tile index no tile has.
    fn sparse_tile(&self, cog: &OpenedCog, ifd_idx: usize) -> Option<Arc<DecodedTile>> {
        let fill = cog.sparse_fill?;
        let key = BlockKey {
            dataset_id: cog.source_id,
            ifd_idx: ifd_idx as u16,
            tile_idx: u32::MAX,
        };
        if let Some(t) = self.cache.try_get_fast(&key) {
            return Some(t);
        }
        let ifd = &cog.index.ifds[ifd_idx];
        let bands = match ifd.planar_config {
            PlanarConfig::Chunky => ifd.samples_per_pixel.max(1) as usize,
            _ => 1,
        };
        let px = (ifd.tile_width as usize) * (ifd.tile_height as usize);
        let wide = matches!(
            ifd.sample_type,
            cogzilla_raster::SampleType::Uint32
                | cogzilla_raster::SampleType::Int32
                | cogzilla_raster::SampleType::Uint64
                | cogzilla_raster::SampleType::Int64
                | cogzilla_raster::SampleType::Float64
        );
        let band = || cogzilla_codec::DecodedBand {
            data: if wide {
                BandData::F64(vec![fill; px].into_boxed_slice())
            } else {
                BandData::F32(cogzilla_codec::AlignedF32Slab::filled(px, fill as f32))
            },
            width: ifd.tile_width,
            height: ifd.tile_height,
            stats: cogzilla_codec::BlockStats {
                all_nodata: false,
                no_nodata: true,
                min: fill as f32,
                max: fill as f32,
            },
        };
        let tile = Arc::new(DecodedTile { bands: (0..bands).map(|_| band()).collect() });
        self.cache.complete(key, tile.clone());
        Some(tile)
    }

    pub fn get_registered(&self, name: &str) -> Option<Arc<OpenedCog>> {
        self.datasets_by_name.get(name).map(|v| v.clone())
    }

    /// Evict a cached URI (optionally only if its `OpenedCog::id` matches `expected_id`) and
    /// advance `invalidation_epoch` so downstream caches flush stale rendered tiles.
    /// Never evicts `OpenSlot::InFlight` entries so concurrent waiters are always woken by the owner.
    pub fn invalidate_uri(&self, uri: &str, expected_id: Option<u64>) {
        let removed = match expected_id {
            Some(id) => self
                .datasets_by_uri
                .remove_if(uri, |_, slot| matches!(slot, OpenSlot::Ready { cog, .. } if cog.id == id))
                .is_some(),
            None => self
                .datasets_by_uri
                .remove_if(uri, |_, slot| matches!(slot, OpenSlot::Ready { .. }))
                .is_some(),
        };
        if removed {
            self.invalidation_epoch.fetch_add(1, Ordering::AcqRel);
        }
    }

    /// Evict all cached `Ready` `OpenedCog` entries (leaving any `InFlight` opens intact) and
    /// advance `invalidation_epoch`. Called by `POST /admin/clear-cache` so overwritten Cloud COGs
    /// (`s3://`, `gs://`, `az://`) re-read fresh IFD headers on the next request.
    pub fn clear_open_uris(&self) {
        self.datasets_by_uri
            .retain(|_, slot| matches!(slot, OpenSlot::InFlight(_)));
        self.nodata_views.clear();
        self.invalidation_epoch.fetch_add(1, Ordering::AcqRel);
        if let Some(ref dc) = self.disk_cache {
            dc.clear();
        }
    }

    /// Validate a cached `OpenedCog` against its backing source.
    /// - Local files: one `stat()` checking `(len, mtime, inode)` (~1–2 µs); returns `Err(NotFound)`
    ///   if deleted or `Ok(false)` if modified.
    /// - Remote HTTP files: zero network calls within `remote_cache_ttl`; once TTL elapses, sends
    ///   conditional `If-None-Match` / `If-Modified-Since` (or expires if the origin sent no validator).
    async fn is_cached_cog_fresh(&self, uri: &str, cog: &Arc<OpenedCog>) -> Result<bool, IoError> {
        match &cog.freshness {
            SourceFreshness::Local {
                path,
                len,
                mtime,
                inode,
            } => match std::fs::metadata(path) {
                Ok(meta) => {
                    let same = meta.len() == *len
                        && meta.modified().ok() == *mtime
                        && file_inode(&meta) == *inode;
                    if !same {
                        self.invalidate_uri(uri, Some(cog.id));
                    }
                    Ok(same)
                }
                Err(e) => {
                    self.invalidate_uri(uri, Some(cog.id));
                    Err(IoError::Fs(e))
                }
            },
            SourceFreshness::Remote {
                validator,
                opened_at,
                last_checked_ms,
                revalidate_lock,
            } => {
                let ttl_ms = self.remote_cache_ttl.as_millis() as u64;
                let now_ms = opened_at.elapsed().as_millis() as u64;
                let prev_ms = last_checked_ms.load(Ordering::Acquire);
                if ttl_ms > 0 && now_ms.saturating_sub(prev_ms) < ttl_ms {
                    return Ok(true);
                }
                // Singleflight TTL revalidation: serialize concurrent waiters and re-check TTL
                // (and whether this cog was already invalidated) after acquiring the lock.
                let _guard = revalidate_lock.lock().await;
                let now_ms = opened_at.elapsed().as_millis() as u64;
                let prev_ms = last_checked_ms.load(Ordering::Acquire);
                if ttl_ms > 0 && now_ms.saturating_sub(prev_ms) < ttl_ms {
                    return Ok(true);
                }
                let still_cached = self
                    .datasets_by_uri
                    .get(uri)
                    .is_some_and(|slot| matches!(&*slot, OpenSlot::Ready { cog: c, .. } if c.id == cog.id));
                if !still_cached {
                    return Ok(false);
                }
                if validator.etag.is_some() || validator.last_modified.is_some() {
                    match cog.source.revalidate_remote(validator).await {
                        Ok(true) => {
                            last_checked_ms.store(
                                opened_at.elapsed().as_millis() as u64,
                                Ordering::Release,
                            );
                            Ok(true)
                        }
                        Ok(false) => {
                            self.invalidate_uri(uri, Some(cog.id));
                            Ok(false)
                        }
                        Err(e) => {
                            self.invalidate_uri(uri, Some(cog.id));
                            Err(e)
                        }
                    }
                } else {
                    self.invalidate_uri(uri, Some(cog.id));
                    Ok(false)
                }
            }
            SourceFreshness::Cloud => Ok(true),
        }
    }

    /// Validate freshness of any already-cached URIs in `uris` (without opening uncached ones).
    /// Used before encoded tile-cache lookups so warm `--tile-cache-mb > 0` hits still detect
    /// overwritten/deleted local files and TTL-expired remote changes.
    pub async fn ensure_uris_fresh(&self, uris: &[String]) -> Result<(), IoError> {
        for uri in uris {
            let cached_hit = self.datasets_by_uri.get(uri).and_then(|existing| {
                if let OpenSlot::Ready { cog, .. } = existing.value() {
                    Some(cog.clone())
                } else {
                    None
                }
            });
            if let Some(cog) = cached_hit {
                let _ = self.is_cached_cog_fresh(uri, &cog).await?;
            }
        }
        Ok(())
    }

    /// Open (or fetch from metadata cache) any local path, `file://`, `http(s)://`, or `s3://` COG.
    /// Singleflight-deduplicated so a 32-tile cold burst against a new S3 COG issues only 1 header GET.
    pub async fn open_uri(&self, uri: &str) -> Result<Arc<OpenedCog>, IoError> {
        crate::allowlist::check_uri_allowed(uri)?;
        loop {
            let cached_hit = self.datasets_by_uri.get(uri).and_then(|existing| {
                if let OpenSlot::Ready { cog, last_used } = existing.value() {
                    last_used.store(self.uri_tick(), Ordering::Relaxed);
                    Some(cog.clone())
                } else {
                    None
                }
            });
            if let Some(cog) = cached_hit {
                if self.is_cached_cog_fresh(uri, &cog).await? {
                    return Ok(cog);
                }
                continue;
            }

            let (wait_flight, ready_hit, own_flight) = {
                use dashmap::mapref::entry::Entry;
                match self.datasets_by_uri.entry(uri.to_string()) {
                    Entry::Occupied(occ) => match occ.get() {
                        OpenSlot::Ready { cog, last_used } => {
                            last_used.store(self.uri_tick(), Ordering::Relaxed);
                            (None, Some(cog.clone()), None)
                        }
                        OpenSlot::InFlight(f) => (Some(f.clone()), None, None),
                    },
                    Entry::Vacant(vac) => {
                        let f = Arc::new(OpenFlight::new());
                        vac.insert(OpenSlot::InFlight(f.clone()));
                        (None, None, Some(f))
                    }
                }
            };
            if let Some(cog) = ready_hit {
                if self.is_cached_cog_fresh(uri, &cog).await? {
                    return Ok(cog);
                }
                continue;
            }

            if let Some(flight) = wait_flight {
                let fut = flight.notify.notified();
                tokio::pin!(fut);
                fut.as_mut().enable();
                if let Some(outcome) = flight.take_outcome() {
                    return outcome.map_err(IoError::Shared);
                }
                match self.datasets_by_uri.get(uri) {
                    Some(existing) => match existing.value() {
                        OpenSlot::Ready { cog, last_used } => {
                            last_used.store(self.uri_tick(), Ordering::Relaxed);
                            return Ok(cog.clone());
                        }
                        OpenSlot::InFlight(cur_f) if !Arc::ptr_eq(cur_f, &flight) => continue,
                        OpenSlot::InFlight(_) => {}
                    },
                    None => {
                        if let Some(outcome) = flight.take_outcome() {
                            return outcome.map_err(IoError::Shared);
                        }
                        continue;
                    }
                }
                fut.await;
                if let Some(outcome) = flight.take_outcome() {
                    return outcome.map_err(IoError::Shared);
                }
                continue;
            }

            // We own the cold open; ensure abort on cancellation (leaving outcome == None so a
            // waiter takes over), while explicit open errors broadcast Err(Arc<IoError>) in 1 RTT.
            struct OpenAbortGuard<'a> {
                map: &'a DashMap<String, OpenSlot, ahash::RandomState>,
                uri: String,
                flight: Arc<OpenFlight>,
                done: bool,
            }
            impl Drop for OpenAbortGuard<'_> {
                fn drop(&mut self) {
                    if !self.done
                        && let Some((_, OpenSlot::InFlight(f))) = self.map.remove_if(
                            &self.uri,
                            |_, v| matches!(v, OpenSlot::InFlight(cur) if Arc::ptr_eq(cur, &self.flight)),
                        ) {
                            f.notify.notify_waiters();
                        }
                }
            }
            let my_flight = own_flight.expect("owner branch always allocates OpenFlight");
            let mut open_guard = OpenAbortGuard {
                map: &self.datasets_by_uri,
                uri: uri.to_string(),
                flight: my_flight.clone(),
                done: false,
            };

            let open_res: Result<Arc<OpenedCog>, IoError> = async {
                let source = CogSource::open_remote_limited(
                    uri,
                    if is_cloud_scheme(uri) {
                        self.cloud_client.clone()
                    } else {
                        self.client.clone()
                    },
                    self.s3_cfg.clone(),
                    self.metrics.clone(),
                    self.upstream_limiter.clone(),
                )?;

                let (index, freshness) = match &source {
                    CogSource::LocalMmap { mmap } => {
                        let local_path = crate::backend::resolve_local_or_reject_scheme(uri)?;
                        let meta = std::fs::metadata(local_path)?;
                        let idx = parse_cog(mmap)?;
                        let fresh = SourceFreshness::Local {
                            path: std::path::PathBuf::from(local_path),
                            len: meta.len(),
                            mtime: meta.modified().ok(),
                            inode: file_inode(&meta),
                        };
                        (idx, fresh)
                    }
                    CogSource::RemoteS3 { .. }
                    | CogSource::RemoteHttp { .. }
                    | CogSource::RemoteGcs { .. }
                    | CogSource::RemoteAzure { .. } => {
                        // Speculative 128 KiB single-RTT header read for COGs; if any IFD or tag
                        // array lies further in the file (e.g. non-COG GeoTIFFs with trailing IFDs),
                        // jump directly to its offset with a bounded range request instead of
                        // downloading the intervening pixel data.
                        const MAX_HEADER_FETCH_BYTES: u64 = 32 * 1024 * 1024;
                        const MAX_HEADER_FETCH_ITERS: u32 = 64;
                        const MIN_SEEK_WINDOW: u64 = 32 * 1024;
                        let first_resp = source
                            .fetch_range_with_meta(0, INITIAL_HEADER_FETCH_BYTES, None)
                            .await?;
                        let validator = crate::backend::RemoteValidator {
                            etag: first_resp.etag,
                            last_modified: first_resp.last_modified,
                        };
                        let first_chunk = first_resp.bytes.to_vec();
                        let mut total_fetched = first_chunk.len() as u64;
                        let mut chunks: Vec<(u64, Vec<u8>)> = vec![(0, first_chunk)];
                        let mut header_fetches = 0u32;
                        let idx = loop {
                            match parse_cog_sparse(&chunks) {
                                Ok(idx) => break idx,
                                Err(TiffError::NeedRange { offset, end }) => {
                                    let needed_len = end.saturating_sub(offset);
                                    if needed_len > MAX_HEADER_FETCH_BYTES {
                                        return Err(IoError::Tiff(TiffError::Corrupt(
                                            "TIFF header/IFD exceeds 32 MiB limit",
                                        )));
                                    }
                                    header_fetches += 1;
                                    if header_fetches > MAX_HEADER_FETCH_ITERS {
                                        return Err(IoError::Tiff(TiffError::Corrupt(
                                            "TIFF header required too many incremental range fetches",
                                        )));
                                    }
                                    // If `offset` falls inside or right at the end of an existing
                                    // chunk, extend that chunk from its end so we don't re-fetch
                                    // the overlapping prefix.
                                    let extend_idx = chunks.iter().position(|(base, data)| {
                                        let cend = base.saturating_add(data.len() as u64);
                                        *base <= offset && offset <= cend && end > cend
                                    });
                                    let (fetch_start, fetch_end) = if let Some(idx) = extend_idx {
                                        let cend = chunks[idx]
                                            .0
                                            .saturating_add(chunks[idx].1.len() as u64);
                                        let want_end = end
                                            .max(cend.saturating_add(MIN_SEEK_WINDOW));
                                        (cend, want_end)
                                    } else {
                                        let want_end = end
                                            .max(offset.saturating_add(MIN_SEEK_WINDOW));
                                        (offset, want_end)
                                    };
                                    let fetch_len = fetch_end.saturating_sub(fetch_start);
                                    if total_fetched.saturating_add(fetch_len)
                                        > MAX_HEADER_FETCH_BYTES
                                    {
                                        return Err(IoError::Tiff(TiffError::Corrupt(
                                            "TIFF header/IFD exceeds 32 MiB limit",
                                        )));
                                    }
                                    let more = source
                                        .fetch_range_with_meta(
                                            fetch_start,
                                            fetch_end,
                                            Some(&validator),
                                        )
                                        .await?
                                        .bytes;
                                    if more.is_empty()
                                        || fetch_start.saturating_add(more.len() as u64) < end
                                    {
                                        return Err(IoError::Tiff(TiffError::NeedMoreBytes {
                                            required: end,
                                            available: fetch_start
                                                .saturating_add(more.len() as u64),
                                        }));
                                    }
                                    total_fetched =
                                        total_fetched.saturating_add(more.len() as u64);
                                    if let Some(idx) = extend_idx {
                                        chunks[idx].1.extend_from_slice(&more);
                                    } else {
                                        chunks.push((fetch_start, more.to_vec()));
                                    }
                                }
                                Err(e) => return Err(IoError::Tiff(e)),
                            }
                        };
                        let fresh = if matches!(source, CogSource::RemoteHttp { .. }) {
                            SourceFreshness::Remote {
                                validator,
                                opened_at: std::time::Instant::now(),
                                last_checked_ms: AtomicU64::new(0),
                                revalidate_lock: tokio::sync::Mutex::new(()),
                            }
                        } else {
                            SourceFreshness::Cloud
                        };
                        (idx, fresh)
                    }
                };

                let id = self.next_id.fetch_add(1, Ordering::Relaxed);
                let sparse_fill = default_sparse_fill(&index);
                Ok(Arc::new(OpenedCog {
                    id,
                    source_id: id,
                    sparse_fill,
                    uri: uri.to_string(),
                    index,
                    source,
                    freshness,
                }))
            }
            .await;

            open_guard.done = true;
            match open_res {
                Ok(cog) => {
                    my_flight.set_outcome(Ok(cog.clone()));
                    let prev = self.datasets_by_uri.insert(
                        uri.to_string(),
                        OpenSlot::Ready {
                            cog: cog.clone(),
                            last_used: AtomicU64::new(self.uri_tick()),
                        },
                    );
                    my_flight.notify.notify_waiters();
                    if let Some(OpenSlot::InFlight(f)) = prev
                        && !Arc::ptr_eq(&f, &my_flight)
                    {
                        f.set_outcome(Ok(cog.clone()));
                        f.notify.notify_waiters();
                    }
                    if self.datasets_by_uri.len() > self.max_open_uris() {
                        self.uri_budget.evict_to_budget(self);
                    }
                    return Ok(cog);
                }
                Err(err) => {
                    let shared = Arc::new(err);
                    // Set outcome before removing the InFlight entry so no waiter can observe
                    // `None` in both the map and `take_outcome()`.
                    my_flight.set_outcome(Err(shared.clone()));
                    let removed = self.datasets_by_uri.remove_if(uri, |_, v| {
                        matches!(v, OpenSlot::InFlight(cur) if Arc::ptr_eq(cur, &my_flight))
                    });
                    my_flight.notify.notify_waiters();
                    if let Some((_, OpenSlot::InFlight(f))) = removed
                        && !Arc::ptr_eq(&f, &my_flight)
                    {
                        f.set_outcome(Err(shared.clone()));
                        f.notify.notify_waiters();
                    }
                    return Err(IoError::Shared(shared));
                }
            }
        }
    }

    /// Synchronous zero-allocation fast path for fully cached blocks or a single local `mmap` miss.
    /// Returns `Ok(Some(blocks))` if satisfied immediately without awaiting; `Ok(None)` if async I/O
    /// or multi-block parallel decoding is needed.
    pub fn try_get_blocks_sync(
        &self,
        cog: &OpenedCog,
        ifd_idx: usize,
        tile_indices: &[Option<u32>],
    ) -> Result<Option<Vec<Option<Arc<DecodedTile>>>>, IoError> {
        let ifd = &cog.index.ifds[ifd_idx];
        let mut out = Vec::with_capacity(tile_indices.len());
        let mut single_local_miss: Option<(usize, u32, BlockKey, &[u8])> = None;

        for (slot, &opt_idx) in tile_indices.iter().enumerate() {
            let Some(tile_idx) = opt_idx else {
                out.push(None);
                continue;
            };
            let count = ifd
                .tile_byte_counts
                .get(tile_idx as usize)
                .copied()
                .ok_or_else(|| IoError::Tiff(TiffError::Corrupt("tile index out of bounds")))?;
            if count == 0 {
                out.push(self.sparse_tile(cog, ifd_idx));
                continue;
            }
            let key = BlockKey {
                dataset_id: cog.id,
                ifd_idx: ifd_idx as u16,
                tile_idx,
            };
            if let Some(hit) = self.cache.try_get_fast(&key) {
                out.push(Some(hit));
                continue;
            }

            let offset = ifd
                .tile_offsets
                .get(tile_idx as usize)
                .copied()
                .ok_or_else(|| IoError::Tiff(TiffError::Corrupt("tile index out of bounds")))?;
            if let Some(slice) = cog.source.local_slice(offset, count) {
                // If >= 2 local blocks missed, hand off to `get_blocks` so they decode in parallel!
                if single_local_miss.is_some() {
                    return Ok(None);
                }
                single_local_miss = Some((slot, tile_idx, key, slice));
                out.push(None);
            } else {
                return Ok(None);
            }
        }

        if let Some((slot, tile_idx, key, slice)) = single_local_miss {
            match self.cache.acquire(key) {
                LookupResult::Hit(hit) => out[slot] = Some(hit),
                LookupResult::Owner => {
                    let guard = self.cache.guard(key);
                    let rows = ifd.block_height(tile_idx);
                    let decode_res: Result<DecodedTile, IoError> = (|| {
                        let mut decoded = decode_tile_rows(
                            slice,
                            ifd,
                            cog.index.byte_order,
                            cog.index.nodata_f64,
                            rows,
                        )?;
                        if let Some(mask_ifd) = ifd.mask_ifd.as_deref()
                            && mask_ifd.tile_width == ifd.tile_width
                                && mask_ifd.tile_height == ifd.tile_height
                            {
                                let per_plane =
                                    (ifd.tiles_across as usize) * (ifd.tiles_down as usize);
                                let mask_idx = (tile_idx as usize) % per_plane.max(1);
                                match mask_tile_range(mask_ifd, mask_idx) {
                                    SidecarTileRange::Present(moff, mlen) => {
                                        let Some(mslice) = cog.source.local_slice(moff, mlen) else {
                                            return Err(IoError::Tiff(TiffError::NeedMoreBytes {
                                                required: moff.saturating_add(mlen as u64),
                                                available: 0,
                                            }));
                                        };
                                        let mask = decode_mask_tile_rows(
                                            mslice,
                                            mask_ifd,
                                            mask_ifd.block_height(mask_idx),
                                        )?;
                                        apply_mask_to_tile(&mut decoded, &mask);
                                    }
                                    SidecarTileRange::SparseTransparent => {
                                        mark_tile_all_nodata(&mut decoded);
                                    }
                                    SidecarTileRange::NotApplicable => {}
                                }
                            }
                        if let Some(alpha_plane) = planar_alpha_plane(ifd) {
                            match planar_alpha_tile_range(ifd, tile_idx, alpha_plane) {
                                SidecarTileRange::Present(aoff, alen) => {
                                    let Some(aslice) = cog.source.local_slice(aoff, alen) else {
                                        return Err(IoError::Tiff(TiffError::NeedMoreBytes {
                                            required: aoff.saturating_add(alen as u64),
                                            available: 0,
                                        }));
                                    };
                                    let alpha_tile = decode_tile_rows(
                                        aslice,
                                        ifd,
                                        cog.index.byte_order,
                                        None,
                                        rows,
                                    )?;
                                    if let Some(ab) = alpha_tile.bands.first() {
                                        apply_alpha_band_to_tile(&mut decoded, ab);
                                    }
                                }
                                SidecarTileRange::SparseTransparent => {
                                    mark_tile_all_nodata(&mut decoded);
                                }
                                SidecarTileRange::NotApplicable => {}
                            }
                        }
                        Ok(decoded)
                    })();
                    let decoded = match decode_res {
                        Ok(d) => d,
                        Err(e) => {
                            let shared = Arc::new(e);
                            guard.fail(shared.clone());
                            return Err(IoError::Shared(shared));
                        }
                    };
                    let arc = Arc::new(decoded);
                    guard.complete(arc.clone());
                    out[slot] = Some(arc);
                }
                LookupResult::Wait(_) => {
                    return Ok(None);
                }
            }
        }

        Ok(Some(out))
    }

    /// Async path: satisfies any mix of cache hits, singleflight waiters, local mmap,
    /// and coalesced concurrent remote HTTP/S3 Range requests!
    pub async fn get_blocks(
        &self,
        cog: &OpenedCog,
        ifd_idx: usize,
        tile_indices: &[Option<u32>],
    ) -> Result<Vec<Option<Arc<DecodedTile>>>, IoError> {
        // Bound per-request block fan-out and cumulative decoded bytes so a hostile overview-less
        // TIFF cannot force tens of thousands of simultaneous block decodes into RAM.
        const MAX_BLOCKS_PER_REQUEST: usize = 512;
        const MAX_DECODED_BYTES_PER_REQUEST: usize = 256 * 1024 * 1024;
        let ifd = &cog.index.ifds[ifd_idx];
        let active_blocks = tile_indices.iter().filter(|o| o.is_some()).count();
        if active_blocks > MAX_BLOCKS_PER_REQUEST {
            return Err(IoError::Tiff(TiffError::Corrupt(
                "request exceeds maximum 512 source blocks per operation",
            )));
        }
        let bands_per_block = match ifd.planar_config {
            cogzilla_tiff::PlanarConfig::Chunky => ifd.samples_per_pixel as usize,
            cogzilla_tiff::PlanarConfig::Planar => 1,
        };
        let bytes_per_sample = cogzilla_codec::decoded_bytes_per_sample(ifd.sample_type);
        let per_block_bytes = (ifd.tile_width as usize)
            .saturating_mul(ifd.tile_height as usize)
            .saturating_mul(bands_per_block)
            .saturating_mul(bytes_per_sample);
        if active_blocks.saturating_mul(per_block_bytes) > MAX_DECODED_BYTES_PER_REQUEST {
            return Err(IoError::Tiff(TiffError::Corrupt(
                "request exceeds maximum 256 MiB decoded blocks per operation",
            )));
        }
        // Cap compressed byte counts as well so a hostile TIFF with tiny decoded dimensions
        // and huge/duplicate compressed tile_byte_counts cannot trigger multi-GiB transfers.
        const MAX_COMPRESSED_TILE_BYTES: u32 = 64 * 1024 * 1024;
        const MAX_COMPRESSED_BYTES_PER_REQUEST: u64 = 256 * 1024 * 1024;
        let alpha_plane_cap = planar_alpha_plane(ifd);
        let mut total_compressed: u64 = 0;
        for tidx in tile_indices.iter().flatten() {
            let bc = ifd.tile_byte_counts.get(*tidx as usize).copied().unwrap_or(0);
            if bc > MAX_COMPRESSED_TILE_BYTES {
                return Err(IoError::Tiff(TiffError::Corrupt(
                    "compressed tile exceeds 64 MiB limit",
                )));
            }
            total_compressed = total_compressed.saturating_add(bc as u64);
            if let Some(ref m) = ifd.mask_ifd {
                let per_plane = (ifd.tiles_across as usize).saturating_mul(ifd.tiles_down as usize).max(1);
                let mbc = m.tile_byte_counts.get((*tidx as usize) % per_plane).copied().unwrap_or(0);
                if mbc > MAX_COMPRESSED_TILE_BYTES {
                    return Err(IoError::Tiff(TiffError::Corrupt(
                        "compressed mask tile exceeds 64 MiB limit",
                    )));
                }
                total_compressed = total_compressed.saturating_add(mbc as u64);
            }
            if let Some(ap) = alpha_plane_cap
                && let SidecarTileRange::Present(_, abc) = planar_alpha_tile_range(ifd, *tidx, ap)
            {
                if abc > MAX_COMPRESSED_TILE_BYTES {
                    return Err(IoError::Tiff(TiffError::Corrupt(
                        "compressed alpha tile exceeds 64 MiB limit",
                    )));
                }
                total_compressed = total_compressed.saturating_add(abc as u64);
            }
            if total_compressed > MAX_COMPRESSED_BYTES_PER_REQUEST {
                return Err(IoError::Tiff(TiffError::Corrupt(
                    "request exceeds maximum 256 MiB compressed tiles per operation",
                )));
            }
        }

        // Fast path: if local or already warm in cache, zero async overhead!
        if let Some(fast) = self.try_get_blocks_sync(cog, ifd_idx, tile_indices)? {
            return Ok(fast);
        }

        let mut results: Vec<Option<Arc<DecodedTile>>> = vec![None; tile_indices.len()];
        // Each owned slot holds an Option<InFlightGuard> so cancellation/panic auto-aborts!
        let mut owned: Vec<(usize, Option<InFlightGuard>, u64, u32)> = Vec::new();
        let mut waiters: Vec<(usize, BlockKey, Arc<crate::cache::InFlightBlock>)> = Vec::new();

        // Deduplicate identical tile_indices within the same call (e.g. chunky multi-band requests)
        let mut seen_owned: HashMap<u32, usize> = HashMap::new();

        for (slot, &opt_idx) in tile_indices.iter().enumerate() {
            let Some(tile_idx) = opt_idx else { continue };
            let count = ifd
                .tile_byte_counts
                .get(tile_idx as usize)
                .copied()
                .ok_or_else(|| IoError::Tiff(TiffError::Corrupt("tile index out of bounds")))?;
            if count == 0 {
                results[slot] = self.sparse_tile(cog, ifd_idx);
                continue;
            }
            if seen_owned.contains_key(&tile_idx) {
                continue;
            }
            let key = BlockKey {
                dataset_id: cog.id,
                ifd_idx: ifd_idx as u16,
                tile_idx,
            };
            match self.cache.acquire(key) {
                LookupResult::Hit(tile) => {
                    results[slot] = Some(tile);
                }
                LookupResult::Owner => {
                    let offset = ifd
                        .tile_offsets
                        .get(tile_idx as usize)
                        .copied()
                        .ok_or_else(|| {
                            IoError::Tiff(TiffError::Corrupt("tile index out of bounds"))
                        })?;
                    seen_owned.insert(tile_idx, slot);
                    owned.push((slot, Some(self.cache.guard(key)), offset, count));
                }
                LookupResult::Wait(notify) => {
                    waiters.push((slot, key, notify));
                }
            }
        }

        // Fetch & decode all owned misses (bounded multi-core decode when beneficial)
        if !owned.is_empty() {
            let is_local = matches!(cog.source, CogSource::LocalMmap { .. });
            let num_workers = tokio::runtime::Handle::current()
                .metrics()
                .num_workers()
                .max(1);
            let expected_bytes = (ifd.tile_width as usize)
                .saturating_mul(ifd.tile_height as usize)
                .saturating_mul(bands_per_block)
                .saturating_mul(ifd.sample_type.bytes_per_sample());
            const HEAVY_DECODE_BYTES: usize = 2 * 1024 * 1024;
            let heavy_block = expected_bytes > HEAVY_DECODE_BYTES;
            let trivial_block = ifd.compression == cogzilla_tiff::Compression::None
                && ifd.predictor == cogzilla_tiff::Predictor::None
                && expected_bytes <= 256 * 1024;

            let alpha_plane = planar_alpha_plane(ifd);
            let no_sidecar_mask = ifd.mask_ifd.is_none() && alpha_plane.is_none();

            if is_local {
                if heavy_block && no_sidecar_mask {
                    let decode_meta = Arc::new(clone_for_decode(ifd));
                    let byte_order = cog.index.byte_order;
                    let nodata = cog.index.nodata_f64;
                    let mut decode_set = tokio::task::JoinSet::new();
                    let mut decode_err: Option<IoError> = None;

                    for (orig_slot, guard_opt, offset, count) in owned.drain(..) {
                        let rows = tile_indices[orig_slot]
                            .map(|tidx| ifd.block_height(tidx))
                            .unwrap_or(ifd.tile_height);
                        let sem = self.heavy_cpu_sem.clone();
                        let src = cog.source.clone();
                        let meta = decode_meta.clone();
                        decode_set.spawn(async move {
                            let res: Result<DecodedTile, IoError> = async {
                                let _permit = sem.acquire_owned().await.map_err(|e| {
                                    IoError::Fs(std::io::Error::other(format!(
                                        "heavy CPU semaphore closed: {e}"
                                    )))
                                })?;
                                tokio::task::spawn_blocking(move || {
                                    let Some(slice) = src.local_slice(offset, count) else {
                                        return Err(IoError::Tiff(TiffError::NeedMoreBytes {
                                            required: offset.saturating_add(count as u64),
                                            available: 0,
                                        }));
                                    };
                                    Ok::<_, IoError>(decode_tile_rows(
                                        slice, &meta, byte_order, nodata, rows,
                                    )?)
                                })
                                .await
                                .map_err(|e| {
                                    IoError::Fs(std::io::Error::other(format!(
                                        "heavy local decode task failed: {e}"
                                    )))
                                })?
                            }
                            .await;
                            match res {
                                Ok(decoded) => {
                                    let arc = Arc::new(decoded);
                                    if let Some(guard) = guard_opt {
                                        guard.complete(arc.clone());
                                    }
                                    Ok::<_, IoError>((orig_slot, arc))
                                }
                                Err(e) => {
                                    let shared = Arc::new(e);
                                    if let Some(guard) = guard_opt {
                                        guard.fail(shared.clone());
                                    }
                                    Err(IoError::Shared(shared))
                                }
                            }
                        });
                    }

                    while let Some(res) = decode_set.join_next().await {
                        match res {
                            Ok(Ok((orig_slot, arc))) => results[orig_slot] = Some(arc),
                            Ok(Err(e)) => decode_err = Some(e),
                            Err(e) => {
                                decode_err = Some(IoError::Fs(std::io::Error::other(format!(
                                    "heavy local decode task failed: {e}"
                                ))));
                            }
                        }
                    }
                    if let Some(e) = decode_err {
                        return Err(e);
                    }
                } else if num_workers >= 2
                    && owned.len() >= 2
                    && !trivial_block
                    && no_sidecar_mask
                {
                    let decode_meta = Arc::new(clone_for_decode(ifd));
                    let byte_order = cog.index.byte_order;
                    let nodata = cog.index.nodata_f64;
                    let mut decode_set = tokio::task::JoinSet::new();
                    let mut decode_err: Option<IoError> = None;

                    for (orig_slot, guard_opt, offset, count) in owned.drain(..) {
                        if decode_set.len() >= num_workers
                            && let Some(res) = decode_set.join_next().await {
                                match res {
                                    Ok(Ok((s, arc))) => results[s] = Some(arc),
                                    Ok(Err(e)) => decode_err = Some(e),
                                    Err(e) => {
                                        decode_err = Some(IoError::Fs(std::io::Error::other(
                                            format!("local decode task failed: {e}"),
                                        )));
                                    }
                                }
                            }
                        let rows = tile_indices[orig_slot]
                            .map(|tidx| ifd.block_height(tidx))
                            .unwrap_or(ifd.tile_height);
                        let src = cog.source.clone();
                        let meta = decode_meta.clone();
                        decode_set.spawn(async move {
                            let res: Result<DecodedTile, IoError> = (|| {
                                let Some(slice) = src.local_slice(offset, count) else {
                                    return Err(IoError::Tiff(TiffError::NeedMoreBytes {
                                        required: offset.saturating_add(count as u64),
                                        available: 0,
                                    }));
                                };
                                Ok(decode_tile_rows(slice, &meta, byte_order, nodata, rows)?)
                            })();
                            match res {
                                Ok(decoded) => {
                                    let arc = Arc::new(decoded);
                                    if let Some(guard) = guard_opt {
                                        guard.complete(arc.clone());
                                    }
                                    Ok::<_, IoError>((orig_slot, arc))
                                }
                                Err(e) => {
                                    let shared = Arc::new(e);
                                    if let Some(guard) = guard_opt {
                                        guard.fail(shared.clone());
                                    }
                                    Err(IoError::Shared(shared))
                                }
                            }
                        });
                    }

                    while let Some(res) = decode_set.join_next().await {
                        match res {
                            Ok(Ok((orig_slot, arc))) => {
                                results[orig_slot] = Some(arc);
                            }
                            Ok(Err(e)) => decode_err = Some(e),
                            Err(e) => {
                                decode_err = Some(IoError::Fs(std::io::Error::other(format!(
                                    "local decode task failed: {e}"
                                ))));
                            }
                        }
                    }
                    if let Some(e) = decode_err {
                        return Err(e);
                    }
                } else {
                    // Zero-copy inline mmap decode (covers single-block, 1-vCPU, trivial
                    // uncompressed blocks, and local COGs with an internal mask_ifd).
                    let byte_order = cog.index.byte_order;
                    let nodata = cog.index.nodata_f64;
                    for (orig_slot, guard_opt, offset, count) in owned.drain(..) {
                        let decode_res: Result<DecodedTile, IoError> = (|| {
                            let Some(slice) = cog.source.local_slice(offset, count) else {
                                return Err(IoError::Tiff(TiffError::NeedMoreBytes {
                                    required: offset.saturating_add(count as u64),
                                    available: 0,
                                }));
                            };
                            let rows = tile_indices[orig_slot]
                                .map(|tidx| ifd.block_height(tidx))
                                .unwrap_or(ifd.tile_height);
                            let mut decoded =
                                decode_tile_rows(slice, ifd, byte_order, nodata, rows)?;
                            if let Some(mask_ifd) = ifd.mask_ifd.as_deref()
                                && mask_ifd.tile_width == ifd.tile_width
                                    && mask_ifd.tile_height == ifd.tile_height
                                    && let Some(Some(tidx)) = tile_indices.get(orig_slot) {
                                        let per_plane = (ifd.tiles_across as usize)
                                            * (ifd.tiles_down as usize);
                                        let mask_idx = (*tidx as usize) % per_plane.max(1);
                                        match mask_tile_range(mask_ifd, mask_idx) {
                                            SidecarTileRange::Present(moff, mlen) => {
                                                let Some(mslice) =
                                                    cog.source.local_slice(moff, mlen)
                                                else {
                                                    return Err(IoError::Tiff(
                                                        TiffError::NeedMoreBytes {
                                                            required: moff
                                                                .saturating_add(mlen as u64),
                                                            available: 0,
                                                        },
                                                    ));
                                                };
                                                let mask = decode_mask_tile_rows(
                                                    mslice,
                                                    mask_ifd,
                                                    mask_ifd.block_height(mask_idx),
                                                )?;
                                                apply_mask_to_tile(&mut decoded, &mask);
                                            }
                                            SidecarTileRange::SparseTransparent => {
                                                mark_tile_all_nodata(&mut decoded);
                                            }
                                            SidecarTileRange::NotApplicable => {}
                                        }
                                    }
                            if let Some(ap) = alpha_plane
                                && let Some(Some(tidx)) = tile_indices.get(orig_slot) {
                                    match planar_alpha_tile_range(ifd, *tidx, ap) {
                                        SidecarTileRange::Present(aoff, alen) => {
                                            let Some(aslice) = cog.source.local_slice(aoff, alen)
                                            else {
                                                return Err(IoError::Tiff(
                                                    TiffError::NeedMoreBytes {
                                                        required: aoff
                                                            .saturating_add(alen as u64),
                                                        available: 0,
                                                    },
                                                ));
                                            };
                                            let alpha_tile = decode_tile_rows(
                                                aslice, ifd, byte_order, None, rows,
                                            )?;
                                            if let Some(ab) = alpha_tile.bands.first() {
                                                apply_alpha_band_to_tile(&mut decoded, ab);
                                            }
                                        }
                                        SidecarTileRange::SparseTransparent => {
                                            mark_tile_all_nodata(&mut decoded);
                                        }
                                        SidecarTileRange::NotApplicable => {}
                                    }
                                }
                            Ok(decoded)
                        })();
                        let decoded = match decode_res {
                            Ok(d) => d,
                            Err(e) => {
                                let shared = Arc::new(e);
                                if let Some(guard) = guard_opt {
                                    guard.fail(shared.clone());
                                }
                                return Err(IoError::Shared(shared));
                            }
                        };
                        let arc = Arc::new(decoded);
                        if let Some(guard) = guard_opt {
                            guard.complete(arc.clone());
                        }
                        results[orig_slot] = Some(arc);
                    }
                }
            } else {
                // Tier-2 compressed-block disk cache lookup for remote sources (RAM miss -> disk hit).
                // Synchronously probe the in-memory index first so 100% cold misses do 0 JoinSet/spawn_blocking hops!
                if no_sidecar_mask
                    && let Some(dc) = &self.disk_cache
                    && let Some(dc_uri) = cog.disk_cache_uri()
                {
                    let prev_owned = std::mem::take(&mut owned);
                    let mut disk_candidates = Vec::new();
                    for item in prev_owned {
                        let tidx = tile_indices[item.0].unwrap();
                        if let Some((key, path)) =
                            dc.lookup_path(&dc_uri, ifd_idx as u16, tidx)
                        {
                            disk_candidates.push((item, key, path));
                        } else {
                            owned.push(item);
                        }
                    }

                        if !disk_candidates.is_empty() {
                            let meta = Arc::new(clone_for_decode(ifd));
                            let byte_order = cog.index.byte_order;
                            let nodata = cog.index.nodata_f64;
                            let max_disk_conc = std::thread::available_parallelism()
                                .map(|n| n.get())
                                .unwrap_or(4)
                                .clamp(1, 16);
                            let mut disk_set = tokio::task::JoinSet::new();
                            let mut iter = disk_candidates.into_iter();
                            let spawn_one = |disk_set: &mut tokio::task::JoinSet<_>,
                                             (item, key, path): (
                                                 (usize, Option<InFlightGuard>, u64, u32),
                                                 [u8; 16],
                                                 std::path::PathBuf,
                                             )| {
                                let (orig_slot, guard_opt, offset, count) = item;
                                let rows = tile_indices[orig_slot]
                                    .map(|tidx| ifd.block_height(tidx))
                                    .unwrap_or(ifd.tile_height);
                                let dc_c = dc.clone();
                                let m = meta.clone();
                                let sem = if heavy_block {
                                    Some(self.heavy_cpu_sem.clone())
                                } else {
                                    None
                                };
                                disk_set.spawn(async move {
                                    let _permit = match &sem {
                                        Some(s) => s.acquire().await.ok(),
                                        None => None,
                                    };
                                    let decoded_opt = tokio::task::spawn_blocking(move || {
                                        let compressed = dc_c.read_verified_sync(key, &path)?;
                                        if compressed.len() == count as usize
                                            && let Ok(decoded) = decode_tile_rows(
                                                &compressed,
                                                &m,
                                                byte_order,
                                                nodata,
                                                rows,
                                            ) {
                                                return Some(Arc::new(decoded));
                                            }
                                        // Stale length or undecodable payload: evict from disk cache
                                        // so the remote refetch can persist the fresh block.
                                        dc_c.invalidate(key);
                                        None
                                    })
                                    .await
                                    .ok()
                                    .flatten();
                                    if let Some(arc) = decoded_opt {
                                        if let Some(guard) = guard_opt {
                                            guard.complete(arc.clone());
                                        }
                                        return Ok::<
                                            _,
                                            (usize, Option<InFlightGuard>, u64, u32),
                                        >((orig_slot, arc));
                                    }
                                    // Corrupt/evicted disk entry: fail-open and refetch from remote!
                                    Err((orig_slot, guard_opt, offset, count))
                                });
                            };
                            for cand in iter.by_ref().take(max_disk_conc) {
                                spawn_one(&mut disk_set, cand);
                            }
                            while let Some(res) = disk_set.join_next().await {
                                match res {
                                    Ok(Ok((orig_slot, arc))) => {
                                        results[orig_slot] = Some(arc);
                                    }
                                    Ok(Err(miss)) => {
                                        owned.push(miss);
                                    }
                                    Err(e) => {
                                        return Err(IoError::Fs(std::io::Error::other(format!(
                                            "disk cache decode task failed: {e}"
                                        ))));
                                    }
                                }
                                if let Some(next_cand) = iter.next() {
                                    spawn_one(&mut disk_set, next_cand);
                                }
                            }
                        }
                }

                if !owned.is_empty() {
                    let reqs: Vec<TileByteReq> = owned
                        .iter()
                        .enumerate()
                        .map(|(i, &(_, _, offset, len))| TileByteReq {
                            slot: i,
                            offset,
                            len,
                        })
                        .collect();
                    let mut spans =
                        coalesce_ranges(reqs, RANGE_GAP_THRESHOLD, MAX_COALESCED_SPAN);

                    // Parallel decode set capped at `num_workers` in-flight so wide requests
                    // overlap download+decode without flooding Tokio's async workers.
                    let mut decode_set: tokio::task::JoinSet<
                        Result<(usize, Arc<DecodedTile>), IoError>,
                    > = tokio::task::JoinSet::new();
                    let mut decode_meta_opt: Option<Arc<IfdMeta>> = None;
                    let byte_order = cog.index.byte_order;
                    let nodata = cog.index.nodata_f64;
                    let mut fetch_err: Option<IoError> = None;

                    let remote_val = cog.remote_validator().cloned();
                    // Fast path: 1 coalesced span -> await fetch_range_with_meta directly (no JoinSet spawn).
                    // Capture Err into `fetch_err` (instead of early `?`) so `guard.fail(...)`
                    // broadcasts the error to all parked singleflight waiters in 1 RTT.
                    let mut single_span = if spans.len() == 1 {
                        let span = spans.pop().unwrap();
                        match cog
                            .source
                            .fetch_range_with_meta(span.start, span.end, remote_val.as_ref())
                            .await
                        {
                            Ok(resp) => Some((span, resp.bytes)),
                            Err(e) => {
                                if matches!(e.root(), IoError::SourceChanged { .. }) {
                                    self.invalidate_uri(&cog.uri, Some(cog.source_id));
                                }
                                fetch_err = Some(e);
                                None
                            }
                        }
                    } else {
                        None
                    };

                    let mut join_set = tokio::task::JoinSet::new();
                    if single_span.is_none() && fetch_err.is_none() {
                        for span in spans {
                            let src = cog.source.clone();
                            let rv = remote_val.clone();
                            join_set.spawn(async move {
                                let resp = src
                                    .fetch_range_with_meta(span.start, span.end, rv.as_ref())
                                    .await?;
                                Ok::<_, IoError>((span, resp.bytes))
                            });
                        }
                    }

                    loop {
                        let next_span = if let Some(pair) = single_span.take() {
                            Some(Ok(pair))
                        } else {
                            match join_set.join_next().await {
                                Some(Ok(r)) => Some(r),
                                Some(Err(e)) => {
                                    fetch_err = Some(IoError::Fs(std::io::Error::other(format!(
                                        "remote fetch task failed: {e}"
                                    ))));
                                    continue;
                                }
                                None => None,
                            }
                        };
                        let Some(span_res) = next_span else {
                            break;
                        };
                        match span_res {
                            Ok((span, data)) => {
                                let has_more_fetches = !join_set.is_empty();
                                let can_parallel = no_sidecar_mask
                                    && (heavy_block
                                        || (!trivial_block
                                            && ((num_workers >= 2 && span.items.len() >= 2)
                                                || has_more_fetches)));
                                if can_parallel {
                                    let meta = decode_meta_opt
                                        .get_or_insert_with(|| Arc::new(clone_for_decode(ifd)))
                                        .clone();
                                    for item in span.items {
                                        if decode_set.len() >= num_workers
                                            && let Some(res) = decode_set.join_next().await {
                                                match res {
                                                    Ok(Ok((s, arc))) => results[s] = Some(arc),
                                                    Ok(Err(e)) => fetch_err = Some(e),
                                                    Err(e) => {
                                                        fetch_err = Some(IoError::Fs(
                                                            std::io::Error::other(format!(
                                                                "remote decode task failed: {e}"
                                                            )),
                                                        ));
                                                    }
                                                }
                                            }
                                        let (orig_slot, guard_opt, _, _) = &mut owned[item.slot];
                                        let orig_slot = *orig_slot;
                                        let rel_start = (item.offset - span.start) as usize;
                                        let rel_end = rel_start + item.len as usize;
                                        if rel_end > data.len() {
                                            fetch_err =
                                                Some(IoError::Tiff(TiffError::NeedMoreBytes {
                                                    required: item
                                                        .offset
                                                        .saturating_add(item.len as u64),
                                                    available: span
                                                        .start
                                                        .saturating_add(data.len() as u64),
                                                }));
                                            continue;
                                        }
                                        let slice = data.slice(rel_start..rel_end);
                                        let disk_put = self.disk_cache.as_ref().and_then(|dc| {
                                            let dc_uri = cog.disk_cache_uri()?.into_owned();
                                            tile_indices
                                                .get(orig_slot)
                                                .and_then(|o| *o)
                                                .map(|tidx| {
                                                    (dc.clone(), dc_uri, ifd_idx as u16, tidx)
                                                })
                                        });
                                        let rows = tile_indices[orig_slot]
                                            .map(|tidx| ifd.block_height(tidx))
                                            .unwrap_or(ifd.tile_height);
                                        let guard = guard_opt.take();
                                        let m = meta.clone();
                                        let sem = if heavy_block {
                                            Some(self.heavy_cpu_sem.clone())
                                        } else {
                                            None
                                        };
                                        decode_set.spawn(async move {
                                            let slice_for_disk = disk_put
                                                .as_ref()
                                                .map(|_| slice.clone());
                                            let decode_res: Result<DecodedTile, IoError> = async {
                                                if let Some(sem) = sem {
                                                    let _permit =
                                                        sem.acquire_owned().await.map_err(|e| {
                                                            IoError::Fs(std::io::Error::other(
                                                                format!(
                                                                    "heavy CPU semaphore closed: {e}"
                                                                ),
                                                            ))
                                                        })?;
                                                    let tile = tokio::task::spawn_blocking(
                                                        move || {
                                                            decode_tile_rows(
                                                                &slice, &m, byte_order, nodata,
                                                                rows,
                                                            )
                                                        },
                                                    )
                                                    .await
                                                    .map_err(|e| {
                                                        IoError::Fs(std::io::Error::other(format!(
                                                            "heavy remote decode task failed: {e}"
                                                        )))
                                                    })??;
                                                    Ok(tile)
                                                } else {
                                                    Ok(decode_tile_rows(
                                                        &slice, &m, byte_order, nodata, rows,
                                                    )?)
                                                }
                                            }
                                            .await;
                                            let decoded = match decode_res {
                                                Ok(d) => d,
                                                Err(e) => {
                                                    let shared = Arc::new(e);
                                                    if let Some(g) = guard {
                                                        g.fail(shared.clone());
                                                    }
                                                    return Err(IoError::Shared(shared));
                                                }
                                            };
                                            // Persist to tier-2 disk cache only AFTER decode succeeds
                                            // so corrupt upstream payloads never poison disk.
                                            if let (Some((dc, uri, ifd_u16, tidx)), Some(bytes)) =
                                                (disk_put, slice_for_disk)
                                            {
                                                dc.put_async(&uri, ifd_u16, tidx, bytes);
                                            }
                                            let arc = Arc::new(decoded);
                                            if let Some(g) = guard {
                                                g.complete(arc.clone());
                                            }
                                            Ok((orig_slot, arc))
                                        });
                                    }
                                } else {
                                    // If this IFD has a separate 1-bit transparency mask_ifd, coalesce
                                    // and fetch all needed mask ranges for this span concurrently first.
                                    let mut mask_bytes_by_slot: Vec<Option<bytes::Bytes>> =
                                        vec![None; span.items.len()];
                                    let mut sparse_transparent_by_slot: Vec<bool> =
                                        vec![false; span.items.len()];
                                    if let Some(mask_ifd) = ifd.mask_ifd.as_deref()
                                        && mask_ifd.tile_width == ifd.tile_width
                                            && mask_ifd.tile_height == ifd.tile_height
                                        {
                                            let per_plane = (ifd.tiles_across as usize)
                                                * (ifd.tiles_down as usize);
                                            let mut mreqs = Vec::new();
                                            for (idx_in_span, item) in span.items.iter().enumerate()
                                            {
                                                let orig_slot = owned[item.slot].0;
                                                if let Some(Some(tidx)) =
                                                    tile_indices.get(orig_slot)
                                                {
                                                    let mask_idx =
                                                        (*tidx as usize) % per_plane.max(1);
                                                    match mask_tile_range(mask_ifd, mask_idx) {
                                                        SidecarTileRange::Present(moff, mlen) => {
                                                            mreqs.push(TileByteReq {
                                                                slot: idx_in_span,
                                                                offset: moff,
                                                                len: mlen,
                                                            });
                                                        }
                                                        SidecarTileRange::SparseTransparent => {
                                                            sparse_transparent_by_slot[idx_in_span] =
                                                                true;
                                                        }
                                                        SidecarTileRange::NotApplicable => {}
                                                    }
                                                }
                                            }
                                            if !mreqs.is_empty() {
                                                let mspans = coalesce_ranges(
                                                    mreqs,
                                                    RANGE_GAP_THRESHOLD,
                                                    MAX_COALESCED_SPAN,
                                                );
                                                let mut mset = tokio::task::JoinSet::new();
                                                for ms in mspans {
                                                    let src = cog.source.clone();
                                                    let rv = remote_val.clone();
                                                    mset.spawn(async move {
                                                        let b = src
                                                            .fetch_range_with_meta(
                                                                ms.start,
                                                                ms.end,
                                                                rv.as_ref(),
                                                            )
                                                            .await?
                                                            .bytes;
                                                        Ok::<_, IoError>((ms, b))
                                                    });
                                                }
                                                while let Some(mres) = mset.join_next().await {
                                                    match mres {
                                                        Ok(Ok((ms, mb))) => {
                                                            for mi in ms.items {
                                                                let rs =
                                                                    (mi.offset - ms.start) as usize;
                                                                let re = rs + mi.len as usize;
                                                                if re <= mb.len() {
                                                                    mask_bytes_by_slot[mi.slot] =
                                                                        Some(mb.slice(rs..re));
                                                                } else {
                                                                    fetch_err = Some(IoError::Tiff(
                                                                        TiffError::NeedMoreBytes {
                                                                            required: mi
                                                                                .offset
                                                                                .saturating_add(
                                                                                    mi.len as u64,
                                                                                ),
                                                                            available: ms
                                                                                .start
                                                                                .saturating_add(
                                                                                    mb.len() as u64,
                                                                                ),
                                                                        },
                                                                    ));
                                                                }
                                                            }
                                                        }
                                                        Ok(Err(e)) => fetch_err = Some(e),
                                                        Err(e) => {
                                                            fetch_err = Some(IoError::Fs(
                                                                std::io::Error::other(format!(
                                                                    "mask fetch task failed: {e}"
                                                                )),
                                                            ));
                                                        }
                                                    }
                                                }
                                            }
                                        }

                                    // If this IFD is a planar RGBA COG, coalesce and fetch paired
                                    // alpha-plane tiles for this span concurrently too.
                                    let mut alpha_bytes_by_slot: Vec<Option<bytes::Bytes>> =
                                        vec![None; span.items.len()];
                                    if let Some(ap) = alpha_plane {
                                        let mut areqs = Vec::new();
                                        for (idx_in_span, item) in span.items.iter().enumerate() {
                                            let orig_slot = owned[item.slot].0;
                                            if let Some(Some(tidx)) = tile_indices.get(orig_slot) {
                                                match planar_alpha_tile_range(ifd, *tidx, ap) {
                                                    SidecarTileRange::Present(aoff, alen) => {
                                                        areqs.push(TileByteReq {
                                                            slot: idx_in_span,
                                                            offset: aoff,
                                                            len: alen,
                                                        });
                                                    }
                                                    SidecarTileRange::SparseTransparent => {
                                                        sparse_transparent_by_slot[idx_in_span] =
                                                            true;
                                                    }
                                                    SidecarTileRange::NotApplicable => {}
                                                }
                                            }
                                        }
                                        if !areqs.is_empty() {
                                            let aspans = coalesce_ranges(
                                                areqs,
                                                RANGE_GAP_THRESHOLD,
                                                MAX_COALESCED_SPAN,
                                            );
                                            let mut aset = tokio::task::JoinSet::new();
                                            for aspan in aspans {
                                                let src = cog.source.clone();
                                                let rv = remote_val.clone();
                                                aset.spawn(async move {
                                                    let b = src
                                                        .fetch_range_with_meta(
                                                            aspan.start,
                                                            aspan.end,
                                                            rv.as_ref(),
                                                        )
                                                        .await?
                                                        .bytes;
                                                    Ok::<_, IoError>((aspan, b))
                                                });
                                            }
                                            while let Some(ares) = aset.join_next().await {
                                                match ares {
                                                    Ok(Ok((aspan, ab))) => {
                                                        for ai in aspan.items {
                                                            let rs = (ai.offset - aspan.start)
                                                                as usize;
                                                            let re = rs + ai.len as usize;
                                                            if re <= ab.len() {
                                                                alpha_bytes_by_slot[ai.slot] =
                                                                    Some(ab.slice(rs..re));
                                                            } else {
                                                                fetch_err = Some(IoError::Tiff(
                                                                    TiffError::NeedMoreBytes {
                                                                        required: ai
                                                                            .offset
                                                                            .saturating_add(
                                                                                ai.len as u64,
                                                                            ),
                                                                        available: aspan
                                                                            .start
                                                                            .saturating_add(
                                                                                ab.len() as u64,
                                                                            ),
                                                                    },
                                                                ));
                                                            }
                                                        }
                                                    }
                                                    Ok(Err(e)) => fetch_err = Some(e),
                                                    Err(e) => {
                                                        fetch_err = Some(IoError::Fs(
                                                            std::io::Error::other(format!(
                                                                "planar alpha fetch task failed: {e}"
                                                            )),
                                                        ));
                                                    }
                                                }
                                            }
                                        }
                                    }

                                    if fetch_err.is_some() {
                                        continue;
                                    }

                                    for (idx_in_span, item) in span.items.into_iter().enumerate() {
                                        let (orig_slot, guard_opt, _, _) = &mut owned[item.slot];
                                        let rel_start = (item.offset - span.start) as usize;
                                        let rel_end = rel_start + item.len as usize;
                                        if rel_end > data.len() {
                                            fetch_err =
                                                Some(IoError::Tiff(TiffError::NeedMoreBytes {
                                                    required: item
                                                        .offset
                                                        .saturating_add(item.len as u64),
                                                    available: span
                                                        .start
                                                        .saturating_add(data.len() as u64),
                                                }));
                                            continue;
                                        }
                                        let rows = tile_indices
                                            .get(*orig_slot)
                                            .and_then(|o| *o)
                                            .map(|tidx| ifd.block_height(tidx))
                                            .unwrap_or(ifd.tile_height);
                                        match decode_tile_rows(
                                            &data[rel_start..rel_end],
                                            ifd,
                                            byte_order,
                                            nodata,
                                            rows,
                                        ) {
                                            Ok(mut decoded) => {
                                                if no_sidecar_mask
                                                    && let Some(dc) = &self.disk_cache
                                                    && let Some(dc_uri) = cog.disk_cache_uri()
                                                    && let Some(Some(tidx)) =
                                                        tile_indices.get(*orig_slot)
                                                {
                                                    dc.put_async(
                                                        &dc_uri,
                                                        ifd_idx as u16,
                                                        *tidx,
                                                        data.slice(rel_start..rel_end),
                                                    );
                                                }
                                                if sparse_transparent_by_slot[idx_in_span] {
                                                    mark_tile_all_nodata(&mut decoded);
                                                } else if let (Some(mask_ifd), Some(mbytes)) = (
                                                    ifd.mask_ifd.as_deref(),
                                                    mask_bytes_by_slot[idx_in_span].as_ref(),
                                                ) {
                                                    let mask_rows = tile_indices
                                                        .get(*orig_slot)
                                                        .and_then(|o| *o)
                                                        .map(|tidx| {
                                                            let per_plane = (ifd.tiles_across
                                                                as usize)
                                                                * (ifd.tiles_down as usize);
                                                            mask_ifd.block_height(
                                                                (tidx as usize) % per_plane.max(1),
                                                            )
                                                        })
                                                        .unwrap_or(mask_ifd.tile_height);
                                                    match decode_mask_tile_rows(
                                                        mbytes, mask_ifd, mask_rows,
                                                    ) {
                                                        Ok(mask) => {
                                                            apply_mask_to_tile(&mut decoded, &mask);
                                                        }
                                                        Err(e) => {
                                                            fetch_err = Some(IoError::Codec(e));
                                                            continue;
                                                        }
                                                    }
                                                }
                                                if let Some(abytes) =
                                                    alpha_bytes_by_slot[idx_in_span].as_ref()
                                                {
                                                    match decode_tile_rows(
                                                        abytes, ifd, byte_order, None, rows,
                                                    ) {
                                                        Ok(alpha_tile) => {
                                                            if let Some(ab) =
                                                                alpha_tile.bands.first()
                                                            {
                                                                apply_alpha_band_to_tile(
                                                                    &mut decoded,
                                                                    ab,
                                                                );
                                                            }
                                                        }
                                                        Err(e) => {
                                                            fetch_err = Some(IoError::Codec(e));
                                                            continue;
                                                        }
                                                    }
                                                }
                                                let arc = Arc::new(decoded);
                                                if let Some(guard) = guard_opt.take() {
                                                    guard.complete(arc.clone());
                                                }
                                                results[*orig_slot] = Some(arc);
                                            }
                                            Err(e) => {
                                                fetch_err = Some(IoError::Codec(e));
                                            }
                                        }
                                    }
                                }
                            }
                            Err(e) => fetch_err = Some(e),
                        }
                    }

                    while let Some(res) = decode_set.join_next().await {
                        match res {
                            Ok(Ok((orig_slot, arc))) => {
                                results[orig_slot] = Some(arc);
                            }
                            Ok(Err(e)) => fetch_err = Some(e),
                            Err(e) => {
                                fetch_err = Some(IoError::Fs(std::io::Error::other(format!(
                                    "remote decode task failed: {e}"
                                ))));
                            }
                        }
                    }

                    if let Some(e) = fetch_err {
                        if matches!(e.root(), IoError::SourceChanged { .. }) {
                            self.invalidate_uri(&cog.uri, Some(cog.source_id));
                        }
                        let shared = Arc::new(e);
                        // Broadcast the explicit upstream/decode error to all parked waiters for
                        // any still-owned guards so they fail in 1 RTT instead of retrying serially.
                        for (_, guard_opt, _, _) in owned.drain(..) {
                            if let Some(guard) = guard_opt {
                                guard.fail(shared.clone());
                            }
                        }
                        return Err(IoError::Shared(shared));
                    }
                }
            }
        }

        // Resolve any singleflight waiters; read the leader's shared outcome directly so:
        // - `Some(Ok(tile))` succeeds even if the cache evicted the block before wakeup,
        // - `Some(Err(err))` fails immediately in 1 RTT (preventing K× retry storms & AIMD collapse),
        // - `None` (true leader cancellation) seamlessly takes over via `retry_slots`.
        let mut retry_slots: Vec<(usize, u32)> = Vec::new();
        for (slot, key, mut flight) in waiters {
            loop {
                if let Some(outcome) = flight.take_outcome() {
                    match outcome {
                        Ok(hit) => {
                            results[slot] = Some(hit);
                            break;
                        }
                        Err(err) => return Err(IoError::Shared(err)),
                    }
                }
                let switch_to = {
                    let fut = flight.notify.notified();
                    tokio::pin!(fut);
                    fut.as_mut().enable();
                    if let Some(outcome) = flight.take_outcome() {
                        match outcome {
                            Ok(hit) => {
                                results[slot] = Some(hit);
                                break;
                            }
                            Err(err) => return Err(IoError::Shared(err)),
                        }
                    }
                    match self.cache.peek_state(&key) {
                        Some(Ok(hit)) => {
                            results[slot] = Some(hit);
                            break;
                        }
                        Some(Err(live_flight)) => {
                            if !Arc::ptr_eq(&flight, &live_flight) {
                                Some(live_flight)
                            } else {
                                fut.await;
                                None
                            }
                        }
                        None => {
                            if let Some(outcome) = flight.take_outcome() {
                                match outcome {
                                    Ok(hit) => {
                                        results[slot] = Some(hit);
                                        break;
                                    }
                                    Err(err) => return Err(IoError::Shared(err)),
                                }
                            }
                            retry_slots.push((slot, key.tile_idx));
                            break;
                        }
                    }
                };
                if let Some(new_flight) = switch_to {
                    flight = new_flight;
                }
            }
        }

        if !retry_slots.is_empty() {
            let retry_indices: Vec<Option<u32>> =
                retry_slots.iter().map(|&(_, tidx)| Some(tidx)).collect();
            let recovered = Box::pin(self.get_blocks(cog, ifd_idx, &retry_indices)).await?;
            for ((slot, _), tile) in retry_slots.into_iter().zip(recovered) {
                results[slot] = tile;
            }
        }

        // Fill any duplicate slots that shared a tile_idx with an owned slot
        for (slot, &opt_idx) in tile_indices.iter().enumerate() {
            if results[slot].is_none()
                && let Some(tile_idx) = opt_idx
                    && let Some(&first_slot) = seen_owned.get(&tile_idx) {
                        results[slot] = results[first_slot].clone();
                    }
        }

        Ok(results)
    }

    /// Fetch raw compressed bytes for a single COG internal tile, bypassing decoding.
    ///
    /// Used by zero-decode pass-through for WebMercator-aligned JPEG/WebP COGs.
    /// Respects the tier-2 persistent disk cache for remote sources and zero-copy mmap
    /// slicing for local files.
    pub async fn get_raw_block(
        &self,
        cog: &OpenedCog,
        ifd_idx: usize,
        tile_idx: u32,
        offset: u64,
        byte_len: u32,
    ) -> Result<bytes::Bytes, IoError> {
        const MAX_RAW_PASSTHROUGH_BYTES: u32 = 16 * 1024 * 1024;
        if byte_len > MAX_RAW_PASSTHROUGH_BYTES {
            return Err(IoError::Tiff(TiffError::Corrupt(
                "raw passthrough tile exceeds 16 MiB limit",
            )));
        }
        let want = byte_len as usize;
        let dc_uri = cog.disk_cache_uri();
        if let (Some(dc), Some(dcu)) = (&self.disk_cache, &dc_uri)
            && let Some(cached) = dc.get(dcu, ifd_idx as u16, tile_idx).await {
                if cached.len() == want {
                    return Ok(cached);
                }
                let key = DiskBlockCache::key_bytes(dcu, ifd_idx as u16, tile_idx);
                dc.invalidate(key);
            }
        let end = offset.checked_add(byte_len as u64).ok_or_else(|| {
            IoError::Tiff(TiffError::NeedMoreBytes {
                required: u64::MAX,
                available: 0,
            })
        })?;
        let raw = match cog
            .source
            .fetch_range_with_meta(offset, end, cog.remote_validator())
            .await
        {
            Ok(resp) => resp.bytes,
            Err(e @ IoError::SourceChanged { .. }) => {
                self.invalidate_uri(&cog.uri, Some(cog.source_id));
                return Err(e);
            }
            Err(e) => return Err(e),
        };
        if raw.len() < want {
            return Err(IoError::Tiff(TiffError::NeedMoreBytes {
                required: end,
                available: offset.saturating_add(raw.len() as u64),
            }));
        }
        let raw = if raw.len() > want {
            raw.slice(..want)
        } else {
            raw
        };
        if let (Some(dc), Some(dcu)) = (&self.disk_cache, &dc_uri) {
            let valid_passthrough_magic = (raw.len() >= 2 && raw[0] == 0xFF && raw[1] == 0xD8)
                || (raw.len() >= 12 && &raw[0..4] == b"RIFF" && &raw[8..12] == b"WEBP");
            if valid_passthrough_magic {
                dc.put_async(dcu, ifd_idx as u16, tile_idx, raw.clone());
            }
        }
        Ok(raw)
    }

    /// Fetch a full metadata object (STAC Item JSON, MosaicJSON, etc.) from any supported URI
    /// (`s3://`, `/vsis3/`, `gs://`, `/vsigs/`, `az://`, `abfs(s)://`, `/vsiaz/`, `http(s)://`,
    /// `file://`, or local path) with allowlist checks, dynamic cloud auth (IAM/IRSA/ECS/IMDSv2,
    /// GCS ADC/GCE metadata, Azure SharedKey/SAS/Bearer), AIMD concurrency limiting, transient
    /// retries, and a hard streaming byte cap (`max_bytes`) to prevent OOM.
    pub async fn fetch_uri_bytes(
        &self,
        uri: &str,
        max_bytes: usize,
    ) -> Result<bytes::Bytes, IoError> {
        crate::allowlist::check_uri_allowed(uri)?;

        if let Some((bucket, key)) = crate::s3::parse_s3_uri(uri) {
            let cfg = self
                .s3_cfg
                .clone()
                .unwrap_or_else(crate::s3::S3Config::from_env_full);
            let dyn_creds = crate::backend::shared_dynamic_creds();
            return crate::s3::fetch_s3_object_capped(
                &self.cloud_client,
                bucket,
                key,
                &cfg,
                Some(&dyn_creds),
                None,
                max_bytes,
                Some(&self.upstream_limiter),
                Some(&self.metrics),
            )
            .await;
        }
        if uri.starts_with("s3://") || uri.starts_with("/vsis3/") {
            return Err(IoError::InvalidUri(uri.to_string()));
        }

        if let Some((bucket, key)) = crate::cloud::parse_gcs_uri(uri) {
            let cfg = crate::cloud::GcsConfig::from_env();
            let tc = crate::backend::shared_gcs_token_cache();
            return crate::cloud::fetch_gcs_object_capped(
                &self.cloud_client,
                bucket,
                key,
                &cfg,
                &tc,
                max_bytes,
                &self.metrics,
                &self.upstream_limiter,
            )
            .await;
        }
        if uri.starts_with("gs://") || uri.starts_with("/vsigs/") {
            return Err(IoError::InvalidUri(uri.to_string()));
        }

        if let Some(target) = crate::cloud::parse_azure_uri(uri, None) {
            let az_cfg = crate::cloud::AzureConfig::from_env();
            return crate::cloud::fetch_azure_object_capped(
                &self.cloud_client,
                &target,
                &az_cfg,
                max_bytes,
                &self.metrics,
                &self.upstream_limiter,
            )
            .await;
        }
        if uri.starts_with("az://")
            || uri.starts_with("abfs://")
            || uri.starts_with("abfss://")
            || uri.starts_with("/vsiaz/")
        {
            return Err(IoError::InvalidUri(uri.to_string()));
        }

        let http_uri = if let Some(rest) = uri.strip_prefix("/vsicurl/") {
            if rest.starts_with("http://") || rest.starts_with("https://") {
                Some(rest)
            } else {
                return Err(IoError::InvalidUri(uri.to_string()));
            }
        } else if uri.starts_with("http://") || uri.starts_with("https://") {
            Some(uri)
        } else {
            None
        };
        if let Some(uri) = http_uri {
            let max_retries = crate::retry::RetryPolicy::from_env_with_default(4).max_retries;
            let redacted = crate::backend::redact_http_url(uri);
            return crate::retry::execute_get_bytes_with_retry(
                || Ok((self.client.get(uri), redacted.clone())),
                crate::retry::FetchSpec {
                    range: None,
                    max_bytes: Some(max_bytes),
                    validator: None,
                    max_retries,
                },
                &self.upstream_limiter,
                &self.metrics,
            )
            .await;
        }

        let local = crate::backend::resolve_local_or_reject_scheme(uri)?;
        let meta = std::fs::metadata(local)?;
        if !meta.is_file() {
            return Err(IoError::Fs(std::io::Error::other(format!(
                "local path '{local}' is not a regular file"
            ))));
        }
        if meta.len() > max_bytes as u64 {
            return Err(IoError::Fs(std::io::Error::other(format!(
                "local file '{local}' size {} exceeds limit of {max_bytes} bytes",
                meta.len()
            ))));
        }
        use std::io::Read;
        let file = std::fs::File::open(local)?;
        let mut bytes = Vec::with_capacity((meta.len() as usize).min(max_bytes));
        file.take((max_bytes as u64).saturating_add(1))
            .read_to_end(&mut bytes)?;
        if bytes.len() > max_bytes {
            return Err(IoError::Fs(std::io::Error::other(format!(
                "local file '{local}' exceeds limit of {max_bytes} bytes"
            ))));
        }
        Ok(bytes::Bytes::from(bytes))
    }

    /// Fetch an exclusive byte range `[start, end)` from any URI supported by `cogzilla-io`
    /// (`s3://`, `gs://`, `az://`, `abfs://`, `abfss://`, `http://`, `https://`, `/vsi*`,
    /// `file://`, or local path), reusing the shared client pool, cloud credential chain,
    /// allowlist, AIMD concurrency limiter, and `IoMetrics`.
    pub async fn fetch_uri_range(
        &self,
        uri: &str,
        start: u64,
        end: u64,
    ) -> Result<bytes::Bytes, IoError> {
        crate::allowlist::check_uri_allowed(uri)?;
        let source = CogSource::open_remote_limited(
            uri,
            if is_cloud_scheme(uri) {
                self.cloud_client.clone()
            } else {
                self.client.clone()
            },
            self.s3_cfg.clone(),
            self.metrics.clone(),
            self.upstream_limiter.clone(),
        )?;
        source.fetch_range(start, end).await
    }

    /// Cheaply probe an HTTP(S) URI via `HEAD` using the shared pooled client to extract a
    /// freshness validator (`ETag` / `Last-Modified` / `Content-Length`). Returns `None` for
    /// non-HTTP schemes or if the origin does not support `HEAD`.
    pub async fn head_uri_validator(&self, uri: &str) -> Option<String> {
        let http_uri = uri.strip_prefix("/vsicurl/").unwrap_or(uri);
        if !http_uri.starts_with("http://") && !http_uri.starts_with("https://") {
            return None;
        }
        if crate::allowlist::check_uri_allowed(uri).is_err() {
            return None;
        }
        let resp = self.client.head(http_uri).send().await.ok()?;
        if !resp.status().is_success() {
            return None;
        }
        let et = resp
            .headers()
            .get(reqwest::header::ETAG)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        let lm = resp
            .headers()
            .get(reqwest::header::LAST_MODIFIED)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        let cl = resp
            .headers()
            .get(reqwest::header::CONTENT_LENGTH)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        Some(format!("http:{et}:{lm}:{cl}"))
    }

    /// Return the total byte size of a URI using the shared pooled client (`HEAD` for HTTP(S),
    /// `metadata` for local files), or `Ok(None)` on `404`.
    pub async fn fetch_uri_size(&self, uri: &str) -> Result<Option<u64>, IoError> {
        crate::allowlist::check_uri_allowed(uri)?;
        let http_uri = uri.strip_prefix("/vsicurl/").unwrap_or(uri);
        if http_uri.starts_with("http://") || http_uri.starts_with("https://") {
            let permit = self.upstream_limiter.acquire().await;
            let resp = self
                .client
                .head(http_uri)
                .send()
                .await
                .map_err(crate::backend::io_error_from_reqwest)?;
            let status = resp.status().as_u16();
            if status == 404 {
                permit.on_success();
                return Ok(None);
            }
            if !resp.status().is_success() {
                return Err(IoError::HttpStatus {
                    status,
                    url: crate::backend::redact_http_url(http_uri),
                });
            }
            permit.on_success();
            let len = resp
                .headers()
                .get(reqwest::header::CONTENT_LENGTH)
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<u64>().ok())
                .ok_or_else(|| IoError::HttpStatus {
                    status: 502,
                    url: crate::backend::redact_http_url(http_uri),
                })?;
            return Ok(Some(len));
        }
        let local = crate::backend::resolve_local_or_reject_scheme(uri)?;
        match std::fs::metadata(local) {
            Ok(meta) => Ok(Some(meta.len())),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(IoError::Fs(e)),
        }
    }
}

fn is_cloud_scheme(uri: &str) -> bool {
    uri.starts_with("s3://")
        || uri.starts_with("/vsis3/")
        || uri.starts_with("gs://")
        || uri.starts_with("/vsigs/")
        || uri.starts_with("az://")
        || uri.starts_with("abfs://")
        || uri.starts_with("abfss://")
        || uri.starts_with("/vsiaz/")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal sparse 1-tile tiled TIFF on disk with the given tile size, bit depth,
    /// and TIFF SampleFormat (1=uint, 2=int, 3=float). `TileByteCounts` is 0 so the file is
    /// ~200 bytes regardless of tile dimensions.
    fn write_sparse_tiled_tiff(
        path: &std::path::Path,
        tile_dim: u32,
        bits_per_sample: u16,
        sample_format: u16,
    ) {
        let entries: [(u16, u16, u32, u32); 12] = [
            (256, 4, 1, tile_dim),              // ImageWidth
            (257, 4, 1, tile_dim),              // ImageLength
            (258, 3, 1, bits_per_sample as u32), // BitsPerSample
            (259, 3, 1, 1),                     // Compression: none
            (262, 3, 1, 1),                     // MinIsBlack
            (277, 3, 1, 1),                     // SamplesPerPixel
            (284, 3, 1, 1),                     // PlanarConfiguration: chunky
            (322, 4, 1, tile_dim),              // TileWidth
            (323, 4, 1, tile_dim),              // TileLength
            (324, 4, 1, 0),                     // TileOffsets: [0] (inline)
            (325, 4, 1, 0),                     // TileByteCounts: [0] (inline sparse)
            (339, 3, 1, sample_format as u32),  // SampleFormat
        ];
        let mut buf = Vec::new();
        buf.extend_from_slice(b"II");
        buf.extend_from_slice(&42u16.to_le_bytes());
        buf.extend_from_slice(&8u32.to_le_bytes());
        buf.extend_from_slice(&(entries.len() as u16).to_le_bytes());
        for (tag, ty, count, val) in entries {
            buf.extend_from_slice(&tag.to_le_bytes());
            buf.extend_from_slice(&ty.to_le_bytes());
            buf.extend_from_slice(&count.to_le_bytes());
            buf.extend_from_slice(&val.to_le_bytes());
        }
        buf.extend_from_slice(&0u32.to_le_bytes());
        std::fs::write(path, &buf).unwrap();
    }

    #[tokio::test]
    async fn decoded_bytes_cap_uses_exact_bytes_per_sample_for_f64_and_i8() {
        let dir = std::env::temp_dir().join(format!(
            "cogzilla-io-cap-475-{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();

        // Float64 (8 bytes/sample), 512x512 tiles => 2 MiB per block.
        // 129 blocks = 258 MiB > 256 MiB cap => must be refused (previously under-counted as 129 MiB).
        let f64_path = dir.join("sparse_f64.tif");
        write_sparse_tiled_tiff(&f64_path, 512, 64, 3);
        let store = CogStore::new(4 * 1024 * 1024);
        let f64_cog = store
            .open_uri(&f64_path.to_string_lossy())
            .await
            .unwrap();
        assert_eq!(
            f64_cog.index.ifds[0].sample_type,
            cogzilla_raster::SampleType::Float64
        );
        let over_cap = vec![Some(0u32); 129];
        let err = store
            .get_blocks(&f64_cog, 0, &over_cap)
            .await
            .err()
            .expect("expected 256 MiB decoded-bytes cap error for 129 Float64 blocks");
        assert!(
            err.to_string().contains("256 MiB"),
            "expected 256 MiB decoded-bytes cap error for 129 Float64 blocks, got: {err}"
        );
        // 128 blocks = exactly 256 MiB => must succeed.
        let at_cap = vec![Some(0u32); 128];
        assert!(store.get_blocks(&f64_cog, 0, &at_cap).await.is_ok());

        // Int8 decodes into BandData::F32 (4 bytes/sample), 512x512 tiles => 1 MiB per block.
        // 257 blocks = 257 MiB > 256 MiB cap => must be refused; 256 blocks = 256 MiB => ok.
        let i8_path = dir.join("sparse_i8.tif");
        write_sparse_tiled_tiff(&i8_path, 512, 8, 2);
        let i8_cog = store
            .open_uri(&i8_path.to_string_lossy())
            .await
            .unwrap();
        assert_eq!(
            i8_cog.index.ifds[0].sample_type,
            cogzilla_raster::SampleType::Int8
        );
        let i8_over = vec![Some(0u32); 257];
        assert!(
            store.get_blocks(&i8_cog, 0, &i8_over).await.is_err(),
            "257 Int8 blocks decode to 257 MiB of BandData::F32 and must exceed the 256 MiB cap"
        );
        let i8_at = vec![Some(0u32); 256];
        assert!(store.get_blocks(&i8_cog, 0, &i8_at).await.is_ok());

        // Uint8 decodes into compact BandData::U8 (1 byte/sample), 512x512 tiles => 256 KiB/block.
        // 300 blocks = 75 MiB <= 256 MiB cap => must succeed.
        let u8_path = dir.join("sparse_u8.tif");
        write_sparse_tiled_tiff(&u8_path, 512, 8, 1);
        let u8_cog = store
            .open_uri(&u8_path.to_string_lossy())
            .await
            .unwrap();
        assert_eq!(
            u8_cog.index.ifds[0].sample_type,
            cogzilla_raster::SampleType::Uint8
        );
        let u8_blocks = vec![Some(0u32); 300];
        assert!(store.get_blocks(&u8_cog, 0, &u8_blocks).await.is_ok());

        // Int32 decodes into BandData::F64 (8 bytes/sample), 512x512 tiles => 2 MiB/block.
        // 129 blocks = 258 MiB > 256 MiB cap => must be refused.
        let i32_path = dir.join("sparse_i32.tif");
        write_sparse_tiled_tiff(&i32_path, 512, 32, 2);
        let i32_cog = store
            .open_uri(&i32_path.to_string_lossy())
            .await
            .unwrap();
        assert_eq!(
            i32_cog.index.ifds[0].sample_type,
            cogzilla_raster::SampleType::Int32
        );
        assert!(
            store.get_blocks(&i32_cog, 0, &over_cap).await.is_err(),
            "129 Int32 blocks decode to 258 MiB of BandData::F64 and must exceed the 256 MiB cap"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }
}
