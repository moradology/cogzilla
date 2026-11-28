use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use bytes::Bytes;
use cogzilla_io::{CogStore, IoError, OpenedCog};
use cogzilla_kernel::{
    BandSource, ColormapName, DemAlgorithm, ExprProgram, OutputFormat, ResampleMode, RgbStyle,
    SingleBandStyle, SpectralExpr, empty_tile_bytes,
    render_bandmath1, render_bandmath2, render_bandmath3, render_dem_algorithm, render_expr_vm,
    render_rgb_bands, render_sar_rgb, render_single_band, render_terrain_tile,
};
use cogzilla_proj::{
    CoordPlan, CrsModel, ProjError, TileCoord, tile_bounds_3857,
};
use cogzilla_raster::SampleType;
use cogzilla_tiff::{Compression, IfdMeta, PlanarConfig};
use dashmap::DashMap;
use thiserror::Error;

use crate::catalog::{MosaicJsonCatalog, SceneCatalog};
use crate::tile_cache::TileCache;

/// The most scenes of a mosaic that are read at once.
pub const MOSAIC_SCENES_AHEAD: usize = 8;

mod mosaic;
mod multi_cog;

pub(crate) use self::multi_cog::{cogs_share_base_grid, validate_multi_cog_grid};

#[derive(Debug, Error)]
pub enum EngineError {
    #[error("I/O error: {0}")]
    Io(#[from] IoError),
    #[error("projection error: {0}")]
    Proj(#[from] ProjError),
    #[error("invalid request: {0}")]
    BadRequest(String),
    #[error("payload too large: {0}")]
    PayloadTooLarge(String),
    #[error("request timeout: {0}")]
    RequestTimeout(String),
    #[error("dataset not found: {0}")]
    NotFound(String),
    #[error("upstream error: {0}")]
    BadGateway(String),
    #[error("no content")]
    NoContent,
    #[error("method not allowed")]
    MethodNotAllowed(&'static str),
    #[error("{0}")]
    Shared(Arc<EngineError>),
}

#[derive(Debug, Clone)]
pub enum BandRequest {
    Single {
        band_1based: u16,
        rescale: (f32, f32),
        colormap: ColormapName,
    },
    Rgb {
        bands_1based: [u16; 3],
        rescale_min: [f32; 3],
        rescale_max: [f32; 3],
    },
    Spectral1 {
        expr: SpectralExpr,
        band_1based: u16,
        rescale: (f32, f32),
        colormap: ColormapName,
    },
    SarRgb {
        bands_1based: [u16; 2],
        rescale_min: [f32; 3],
        rescale_max: [f32; 3],
    },
    Spectral2 {
        expr: SpectralExpr,
        bands_1based: [u16; 2],
        rescale: (f32, f32),
        colormap: ColormapName,
    },
    Spectral3 {
        expr: SpectralExpr,
        bands_1based: [u16; 3],
        rescale: (f32, f32),
        colormap: ColormapName,
    },
    CustomExpr {
        prog: ExprProgram,
        rescale: (f32, f32),
        colormap: ColormapName,
    },
    CustomExprRgb {
        progs: [ExprProgram; 3],
        rescale_min: [f32; 3],
        rescale_max: [f32; 3],
    },
    Terrain {
        band_1based: u16,
        algorithm: DemAlgorithm,
        /// Band math in place of `band_1based`, as for [`BandRequest::Dem`].
        math: Option<ExprProgram>,
    },
    Dem {
        band_1based: u16,
        algorithm: DemAlgorithm,
        rescale: (f32, f32),
        colormap: ColormapName,
        /// Band math the algorithm reads in place of `band_1based`: an expression's first
        /// output, as rio-tiler gives a terrain algorithm the expression's results.
        math: Option<ExprProgram>,
    },
    /// Arbitrary N-band semicolon-separated custom expressions (e.g. 2-band or 4..=15-band)
    /// for scientific `.tif`/`.npy`/`.npz`/`.f32` output.
    CustomExprMulti {
        progs: Vec<ExprProgram>,
    },
    /// Arbitrary N-band selection (e.g. 2-band or 4..=15-band) for scientific `.tif`/`.npy`/`.f32` output.
    MultiBand {
        bands_1based: Vec<u16>,
    },
}

/// The blocks a terrain algorithm reads over a tile's window: one band's, or band math's
/// over the bands it reads, block by block ([`cogzilla_kernel::band_math_blocks`]).
struct TerrainBlocks {
    tiles: Vec<Option<Arc<cogzilla_codec::DecodedTile>>>,
    planar: bool,
    planes: Vec<u16>,
    per_plane: usize,
    math: Option<Vec<Option<cogzilla_codec::DecodedBand>>>,
}

impl TerrainBlocks {
    fn band(&self, b_1based: u16) -> Vec<Option<&cogzilla_codec::DecodedBand>> {
        let (blocks, in_decoded) = if self.planar {
            let at = self.planes.iter().position(|&p| p + 1 == b_1based).unwrap_or(0);
            (&self.tiles[at * self.per_plane..(at + 1) * self.per_plane], 0)
        } else {
            (&self.tiles[..], usize::from(b_1based - 1))
        };
        blocks.iter().map(|opt| opt.as_ref().and_then(|t| t.bands.get(in_decoded))).collect()
    }

    /// The blocks read as the algorithm's one band.
    fn refs(&self, band_1based: u16) -> Vec<Option<&cogzilla_codec::DecodedBand>> {
        match &self.math {
            Some(blocks) => blocks.iter().map(Option::as_ref).collect(),
            None => self.band(band_1based),
        }
    }
}

fn band_scale_offset(cog: &OpenedCog, b_1based: u16) -> (f32, f32) {
    let b = usize::from(b_1based.saturating_sub(1));
    (cog.index.scales.get(b).copied().unwrap_or(1.0), cog.index.offsets.get(b).copied().unwrap_or(0.0))
}

impl TileEngine {
    async fn read_terrain_blocks(
        &self,
        cog: &OpenedCog,
        ifd_idx: usize,
        (tx0, tx1, ty0, ty1): (u32, u32, u32, u32),
        band_1based: u16,
        math: Option<&ExprProgram>,
        unscale: bool,
    ) -> Result<TerrainBlocks, EngineError> {
        let ifd = &cog.index.ifds[ifd_idx];
        // The bands read: the one asked for, or those of the band math over them.
        let wanted: Vec<u16> = math.map_or_else(|| vec![band_1based], |prog| prog.used_bands.clone());
        for &b in &wanted {
            if b == 0 || b > ifd.samples_per_pixel {
                return Err(EngineError::BadRequest(format!(
                    "band {} out of range (dataset has {} bands)",
                    b, ifd.samples_per_pixel
                )));
            }
        }
        // A chunky raster's blocks hold every band; a planar one's, one band each.
        let planar = matches!(ifd.planar_config, PlanarConfig::Planar);
        let planes: Vec<u16> = if planar { wanted.iter().map(|b| b - 1).collect() } else { vec![wanted[0] - 1] };
        let per_plane = ((tx1 - tx0 + 1) * (ty1 - ty0 + 1)) as usize;
        let mut indices = Vec::with_capacity(per_plane * planes.len());
        for &plane in &planes {
            for ty in ty0..=ty1 {
                for tx in tx0..=tx1 {
                    indices.push(Some(ifd.tile_index(tx, ty, plane) as u32));
                }
            }
        }
        let tiles = self.store.get_blocks(cog, ifd_idx, &indices).await?;
        let mut read = TerrainBlocks { tiles, planar, planes, per_plane, math: None };
        if let Some(prog) = math {
            let refs: Vec<_> = prog.used_bands.iter().map(|&b| read.band(b)).collect();
            let inputs: Vec<_> = prog
                .used_bands
                .iter()
                .zip(&refs)
                .map(|(&b, r)| (&r[..], None, unscale.then(|| band_scale_offset(cog, b))))
                .collect();
            // Every pixel of every block read: kept off the async I/O workers.
            let blocks = block_in_place_if_multithread(true, || cogzilla_kernel::band_math_blocks(prog, &inputs));
            read.math = Some(blocks);
        }
        Ok(read)
    }
}

#[derive(Debug, Clone)]
pub struct TileRequest {
    pub tile: TileCoord,
    pub tile_size: u32,
    pub bands: BandRequest,
    pub resample: ResampleMode,
    pub format: OutputFormat,
    pub unscale: bool,
}

/// Returns `true` when `?unscale=true` must be applied per-sample in `BandSource` (rather than
/// being zero-cost folded into `Single`/`Rgb` rescale bounds): raw float output (`.npy`), DEM
/// algorithms, terrain encodings, spectral indices, SAR dB, and custom band-math expressions.
#[inline]
pub(crate) fn needs_source_unscale(bands: &BandRequest, format: OutputFormat) -> bool {
    matches!(
        format,
        OutputFormat::RawF32 | OutputFormat::RawF64 | OutputFormat::Npy | OutputFormat::NpyF64
    ) || !matches!(bands, BandRequest::Single { .. } | BandRequest::Rgb { .. })
}

/// Whether `bands` draws one value per pixel under the grayscale map: a grey picture.
fn draws_one_grey_band(bands: &BandRequest) -> bool {
    matches!(
        bands,
        BandRequest::Single { colormap: ColormapName::Grayscale, .. }
            | BandRequest::Spectral1 { colormap: ColormapName::Grayscale, .. }
            | BandRequest::Spectral2 { colormap: ColormapName::Grayscale, .. }
            | BandRequest::Spectral3 { colormap: ColormapName::Grayscale, .. }
            | BandRequest::CustomExpr { colormap: ColormapName::Grayscale, .. }
            | BandRequest::Dem { colormap: ColormapName::Grayscale, .. }
    )
}

/// Wrap raw band-major little-endian `f32` tile bytes into the requested float output format
/// (`RawF32`, `Npy`, `RawF64`, or `NpyF64`) so float64 requests on multi-COG/mosaic paths never
/// fall through to RGBA8.
fn format_float_tile(raw_f32: Bytes, bands: usize, tile_size: u32, fmt: OutputFormat) -> Bytes {
    match fmt {
        OutputFormat::Npy => cogzilla_kernel::wrap_raw_f32_as_npy_with_mask(
            raw_f32,
            bands,
            tile_size,
            tile_size,
            true,
        ),
        OutputFormat::RawF64 => {
            let mut out = bytes::BytesMut::with_capacity(raw_f32.len() * 2);
            for chunk in raw_f32.chunks_exact(4) {
                let v = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f64;
                out.extend_from_slice(&v.to_le_bytes());
            }
            out.freeze()
        }
        OutputFormat::NpyF64 => {
            let vals: Vec<f64> = raw_f32
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
                .collect();
            cogzilla_kernel::encode_npy_typed_f64(
                &vals,
                bands,
                tile_size as usize,
                tile_size as usize,
                cogzilla_raster::SampleType::Float64,
                None,
                true,
            )
        }
        _ => raw_f32,
    }
}

#[derive(Default)]
pub struct EngineStats {
    pub tiles_served: AtomicU64,
    pub empty_short_circuits: AtomicU64,
    pub passthrough_hits: AtomicU64,
    pub http_304_not_modified: AtomicU64,
    pub gpu_dispatches: AtomicU64,
}

/// Map a fetch/I/O error on an asset referenced by a STAC item or MosaicJSON document to
/// `502 Bad Gateway` naming the asset (while preserving `403 Forbidden` for allowlist rejections,
/// `501 Not Implemented` for unsupported TIFF features, and `400 Bad Request` for invalid URIs /
/// malformed TIFFs).
pub fn asset_fetch_error(asset_uri: &str, err: IoError) -> EngineError {
    if err.is_fatal() {
        return EngineError::Io(err);
    }
    match err.root() {
        IoError::InvalidUri(_)
        | IoError::Tiff(_)
        | IoError::SourceChanged { .. } => EngineError::Io(err),
        _ => {
            let safe = cogzilla_io::redact_http_url(asset_uri);
            EngineError::BadGateway(format!(
                "referenced asset '{safe}' could not be fetched: {err}"
            ))
        }
    }
}

impl EngineError {
    /// True when this error must never be swallowed by a per-asset fallback
    /// (e.g. an allowlist / SSRF refusal).
    pub fn is_fatal(&self) -> bool {
        match self {
            Self::Io(io) => io.is_fatal(),
            Self::Shared(inner) => inner.is_fatal(),
            _ => false,
        }
    }
}

/// Absorb non-fatal per-asset errors as `Ok(None)` while always propagating fatal errors
/// (such as [`IoError::Forbidden`] allowlist refusals).
pub fn tolerate_asset_failure<T>(result: Result<T, EngineError>) -> Result<Option<T>, EngineError> {
    match result {
        Ok(v) => Ok(Some(v)),
        Err(e) if e.is_fatal() => Err(e),
        Err(_) => Ok(None),
    }
}

pub struct TileEngine {
    pub store: Arc<CogStore>,
    crs_cache: DashMap<u64, CrsModel, ahash::RandomState>,
    scenes: DashMap<String, Vec<Arc<OpenedCog>>, ahash::RandomState>,
    catalogs: DashMap<String, Arc<dyn SceneCatalog>, ahash::RandomState>,
    pub stac_cache: DashMap<String, Arc<crate::stac::StacItem>, ahash::RandomState>,
    pub mosaicjson_cache: DashMap<String, Arc<MosaicJsonCatalog>, ahash::RandomState>,
    pub tile_cache: Arc<TileCache>,
    tile_cache_epoch: AtomicU64,
    pub stats: Arc<EngineStats>,
    pgstac: std::sync::OnceLock<Arc<cogzilla_pgstac::PgStac>>,
    pub aimd: crate::aimd::AimdLimiter,
    pub gpu: Option<Arc<crate::gpu_backend::GpuBackend>>,
}

/// Run a synchronous CPU-heavy closure via `tokio::task::block_in_place` when `should_offload` is
/// true and we are on a multi-threaded Tokio runtime, so `/healthz`, `/metrics`, and AIMD
/// `429 Too Many Requests` shedding stay responsive under CPU-only saturation.
#[inline]
pub(crate) fn block_in_place_if_multithread<R>(should_offload: bool, f: impl FnOnce() -> R) -> R {
    if should_offload
        && tokio::runtime::Handle::try_current()
            .is_ok_and(|h| h.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread)
    {
        tokio::task::block_in_place(f)
    } else {
        f()
    }
}

/// Run a synchronous CPU tile render, moving heavy non-Nearest renders (`N >= 256`) off Tokio's
/// async I/O worker pool via [`block_in_place_if_multithread`].
#[inline]
fn run_cpu_render<R>(n: usize, resample: ResampleMode, f: impl FnOnce() -> R) -> R {
    block_in_place_if_multithread(n >= 256 && !matches!(resample, ResampleMode::Nearest), f)
}

/// Resampling kernel support margin `(lo, hi)` in source pixels relative to
/// `[floor(min), floor(max)]`.
pub(crate) fn resample_fetch_margin(resample: ResampleMode, is_dem: bool) -> (i32, i32) {
    let (lo, hi) = match resample {
        ResampleMode::Nearest => (0, 0),
        ResampleMode::Bilinear => (0, 1),
        ResampleMode::Cubic | ResampleMode::CubicSpline | ResampleMode::Gauss => (1, 2),
        ResampleMode::Lanczos => (2, 3),
        // Footprint modes (Average/Mode/Rms/Min/Max/Med/Q1/Q3/Sum) use exact footprint bounds in
        // `render_bbox_impl` and `tile_fetch_range_for_plan`.
        _ => (1, 2),
    };
    if is_dem {
        (lo + 2, hi + 2)
    } else {
        (lo, hi)
    }
}

/// Compute the inclusive source tile range `(tx0, tx1, ty0, ty1)` required to sample `plan`
/// under `resample` (using the exact `coord_plan_scale` half-footprint for footprint modes so
/// overview-less COGs or coarse overview jumps never under-fetch adjacent internal tiles).
fn tile_fetch_range_for_plan<const N: usize>(
    plan: &CoordPlan<N>,
    resample: ResampleMode,
    is_dem: bool,
    ifd: &cogzilla_tiff::IfdMeta,
) -> (u32, u32, u32, u32) {
    let is_footprint = matches!(
        resample,
        ResampleMode::Average
            | ResampleMode::Mode
            | ResampleMode::Rms
            | ResampleMode::Min
            | ResampleMode::Max
            | ResampleMode::Med
            | ResampleMode::Q1
            | ResampleMode::Q3
            | ResampleMode::Sum
    );
    let widened = if is_footprint || resample == ResampleMode::Nearest {
        None
    } else {
        let (sx, sy) = cogzilla_kernel::coord_plan_scale(plan);
        cogzilla_kernel::kernel_widens(resample, sx, sy).then_some((sx, sy))
    };
    let (pad_lo_x, pad_hi_x, pad_lo_y, pad_hi_y) = if is_footprint {
        let (sx, sy) = cogzilla_kernel::coord_plan_scale(plan);
        let hx = (0.5 * sx).ceil().max(1.0) as i32;
        let hy = (0.5 * sy).ceil().max(1.0) as i32;
        if is_dem {
            (hx + 2, hx + 3, hy + 2, hy + 3)
        } else {
            (hx, hx + 1, hy, hy + 1)
        }
    } else if let Some((sx, sy)) = widened {
        // GDAL-style widened kernel: the support grows to `radius * scale` source pixels.
        let hx = cogzilla_kernel::widened_kernel_radius(resample, sx).ceil().max(1.0) as i32;
        let hy = cogzilla_kernel::widened_kernel_radius(resample, sy).ceil().max(1.0) as i32;
        if is_dem {
            (hx + 2, hx + 3, hy + 2, hy + 3)
        } else {
            (hx, hx + 1, hy, hy + 1)
        }
    } else {
        let (lo, hi) = resample_fetch_margin(resample, is_dem);
        (lo, hi, lo, hi)
    };
    if pad_lo_x > 0 || pad_hi_x > 1 || pad_lo_y > 0 || pad_hi_y > 1 {
        // CoordPlan::src_pixel_bounds() is the exact bilinear footprint [floor(min), floor(max)+1].
        let extra_lo_x = pad_lo_x;
        let extra_hi_x = (pad_hi_x - 1).max(0);
        let extra_lo_y = pad_lo_y;
        let extra_hi_y = (pad_hi_y - 1).max(0);
        let (c0, c1, r0, r1) = plan.src_pixel_bounds();
        let d_tx0 = (c0.saturating_sub(extra_lo_x).max(0) as u32) / ifd.tile_width;
        let d_tx1 = ((c1.saturating_add(extra_hi_x).max(0) as u32) / ifd.tile_width)
            .min(ifd.tiles_across.saturating_sub(1));
        let d_ty0 = (r0.saturating_sub(extra_lo_y).max(0) as u32) / ifd.tile_height;
        let d_ty1 = ((r1.saturating_add(extra_hi_y).max(0) as u32) / ifd.tile_height)
            .min(ifd.tiles_down.saturating_sub(1));
        (d_tx0, d_tx1, d_ty0, d_ty1)
    } else {
        plan.src_tile_range(ifd.tile_width, ifd.tile_height, ifd.tiles_across, ifd.tiles_down)
    }
}

/// Shared startup configuration for `cogzilla-server` and `cogzilla-lambda`.
#[derive(Debug, Clone)]
pub struct EngineBootConfig {
    pub cache_mb: usize,
    pub tile_cache_mb: usize,
    pub disk_cache_mb: u64,
    pub disk_cache_dir: Option<std::path::PathBuf>,
}

impl TileEngine {
    pub fn new(cache_max_bytes: usize) -> Self {
        Self::with_store(Arc::new(CogStore::new(cache_max_bytes)))
    }

    pub fn with_store(store: Arc<CogStore>) -> Self {
        Self::with_store_and_tile_cache(store, 0)
    }

    pub fn with_store_and_tile_cache(store: Arc<CogStore>, tile_cache_max_bytes: usize) -> Self {
        let gpu = crate::gpu_backend::GpuBackend::try_init(256 * 1024 * 1024);
        let has_gpu = gpu.is_some();

        Self {
            store,
            crs_cache: DashMap::with_hasher(ahash::RandomState::new()),
            scenes: DashMap::with_hasher(ahash::RandomState::new()),
            catalogs: DashMap::with_hasher(ahash::RandomState::new()),
            stac_cache: DashMap::with_hasher(ahash::RandomState::new()),
            mosaicjson_cache: DashMap::with_hasher(ahash::RandomState::new()),
            tile_cache: Arc::new(TileCache::new(tile_cache_max_bytes)),
            tile_cache_epoch: AtomicU64::new(0),
            stats: Arc::new(EngineStats::default()),
            pgstac: std::sync::OnceLock::new(),
            aimd: crate::aimd::AimdLimiter::from_env(has_gpu),
            gpu,
        }
    }

    /// Returns the active GPU backend when the `gpu` feature is enabled and hardware initialized,
    /// or `None` on CPU-only builds / hosts without a usable GPU.
    #[inline]
    pub fn accelerated(&self) -> Option<&crate::gpu_backend::GpuBackend> {
        self.gpu.as_deref()
    }

    /// Build a `TileEngine` from `EngineBootConfig`, opening the optional tier-2
    /// compressed-block disk cache if `disk_cache_mb > 0`.
    pub fn boot(cfg: EngineBootConfig) -> Arc<Self> {
        let mut store = CogStore::new(cfg.cache_mb.saturating_mul(1024 * 1024));
        if cfg.disk_cache_mb > 0 {
            let dir = cfg
                .disk_cache_dir
                .unwrap_or_else(|| std::env::temp_dir().join("cogzilla-disk-cache"));
            match cogzilla_io::DiskBlockCache::open(
                &dir,
                cfg.disk_cache_mb.saturating_mul(1024 * 1024),
            ) {
                Ok(dc) => {
                    tracing::info!(
                        dir = %dir.display(),
                        disk_cache_mb = cfg.disk_cache_mb,
                        "enabled tier-2 compressed-block disk cache"
                    );
                    store = store.with_disk_cache(dc);
                }
                Err(e) => {
                    tracing::warn!(
                        dir = %dir.display(),
                        error = %e,
                        "failed to open disk cache directory; continuing with tier 2 disabled"
                    );
                }
            }
        }
        Arc::new(Self::with_store_and_tile_cache(
            Arc::new(store),
            cfg.tile_cache_mb.saturating_mul(1024 * 1024),
        ))
    }

    /// Attach the pgSTAC database behind the `/searches` routes. The first call wins.
    pub fn set_pgstac(&self, pgstac: Arc<cogzilla_pgstac::PgStac>) -> bool {
        self.pgstac.set(pgstac).is_ok()
    }

    /// The pgSTAC database, when one is configured.
    pub fn pgstac(&self) -> Option<&Arc<cogzilla_pgstac::PgStac>> {
        self.pgstac.get()
    }

    /// Attach the pgSTAC database named by `DATABASE_URL` or the `PG*` variables, if any.
    /// `Ok(false)` means none is configured. No connection is opened here.
    pub fn attach_pgstac_from_env(&self) -> Result<bool, String> {
        let Some(config) = cogzilla_pgstac::PgStacConfig::from_env().map_err(|e| e.to_string())?
        else {
            return Ok(false);
        };
        let pgstac = cogzilla_pgstac::PgStac::new(config).map_err(|e| e.to_string())?;
        Ok(self.set_pgstac(Arc::new(pgstac)))
    }

    /// Replace the engine's AIMD concurrency limiter (useful for tests and custom deployments).
    pub fn with_aimd(mut self, aimd: crate::aimd::AimdLimiter) -> Self {
        self.aimd = aimd;
        self
    }

    /// Flush the rendered tile cache if any underlying source was invalidated or replaced.
    /// Costs a single relaxed/acquire atomic load on the warm path.
    #[inline(always)]
    pub fn sync_tile_cache_epoch(&self) {
        let cur = self.store.invalidation_epoch.load(Ordering::Acquire);
        let seen = self.tile_cache_epoch.load(Ordering::Relaxed);
        if cur != seen && self.tile_cache_epoch.swap(cur, Ordering::AcqRel) != cur {
            self.tile_cache.clear();
            self.clear_gpu_caches();
        }
    }

    /// Synchronously clear all GPU VRAM slab caches (used by `/admin/clear-cache` and epoch sync).
    pub fn clear_gpu_caches(&self) {
        if let Some(gpu) = &self.gpu {
            gpu.slab_cache.clear();
            gpu.multi_slab_cache.clear();
        }
    }

    /// Register a pluggable spatial catalog (`SceneCatalog`) for `/mosaic/{name}/tiles`.
    pub fn register_catalog(&self, name: impl Into<String>, catalog: Arc<dyn SceneCatalog>) {
        self.catalogs.insert(name.into(), catalog);
    }

    pub fn get_catalog(&self, name: &str) -> Option<Arc<dyn SceneCatalog>> {
        self.catalogs.get(name).map(|v| v.clone())
    }

    pub fn catalog_count(&self) -> usize {
        self.catalogs.len()
    }

    /// Pre-register a multi-COG satellite scene (e.g., Landsat `[B4, B3, B2]` or SAR `[VV, VH]`).
    pub fn register_local_scene<P: AsRef<Path>>(
        &self,
        name: impl Into<String>,
        paths: &[P],
    ) -> Result<Vec<Arc<OpenedCog>>, EngineError> {
        let name_str = name.into();
        let mut cogs = Vec::with_capacity(paths.len());
        for (i, p) in paths.iter().enumerate() {
            let key = format!("{name_str}__band{i}");
            let cog = self.store.register_local(key, p)?;
            let _ = self.get_or_build_crs(&cog)?;
            cogs.push(cog);
        }
        self.scenes.insert(name_str, cogs.clone());
        Ok(cogs)
    }

    pub fn get_scene(&self, name: &str) -> Option<Vec<Arc<OpenedCog>>> {
        self.scenes.get(name).map(|v| v.clone())
    }

    pub fn get_or_build_crs(&self, cog: &OpenedCog) -> Result<CrsModel, EngineError> {
        if !cog.index.is_georeferenced {
            let msg = if cog.index.has_rpc {
                "dataset has no supported georeferencing (missing GeoTransform/GCPs; RPC georeferencing is not supported); geographic operations require a georeferenced raster"
            } else {
                "dataset has no georeferencing (missing GeoTransform/GCPs; RPCs are not supported); geographic operations require a georeferenced raster"
            };
            return Err(EngineError::BadRequest(msg.into()));
        }
        if let Some(c) = self.crs_cache.get(&cog.id) {
            return Ok(c.clone());
        }
        let crs = CrsModel::from_geo_keys(&cog.index.geo_keys)?;
        if self.crs_cache.len() >= 8192 {
            self.crs_cache.clear();
        }
        self.crs_cache.insert(cog.id, crs.clone());
        Ok(crs)
    }

    /// Render a single XYZ tile for an already-opened COG.
    pub async fn render_tile(
        &self,
        cog: &OpenedCog,
        req: &TileRequest,
    ) -> Result<Bytes, EngineError> {
        self.stats.tiles_served.fetch_add(1, Ordering::Relaxed);
        if matches!(req.format, OutputFormat::RawF64 | OutputFormat::NpyF64)
            && !req.unscale
            && matches!(req.resample, ResampleMode::Nearest | ResampleMode::Bilinear)
            && cog.index.full_res().sample_type.needs_f64()
        {
            let bands_1based: Option<Vec<u16>> = match &req.bands {
                BandRequest::Single { band_1based, .. } => Some(vec![*band_1based]),
                BandRequest::Rgb { bands_1based, .. } => Some(bands_1based.to_vec()),
                BandRequest::MultiBand { bands_1based } => Some(bands_1based.clone()),
                _ => None,
            };
            if let Some(b1) = bands_1based {
                let kind = cog.index.full_res().sample_type;
                let (minx, miny, maxx, maxy) = cogzilla_proj::tile_bounds_3857(req.tile);
                return Box::pin(crate::stats::render_exact_f64_array(
                    self,
                    cog,
                    [minx, miny, maxx, maxy],
                    &CrsModel::Epsg3857,
                    false,
                    req.tile_size,
                    req.tile_size,
                    None,
                    &b1,
                    req.resample,
                    None,
                    req.unscale,
                    kind,
                    true,
                    matches!(req.format, OutputFormat::RawF64),
                    true,
                ))
                .await;
            }
        }
        let has_alpha_channel = {
            let full = cog.index.full_res();
            cogzilla_codec::alpha_channel_index(full.samples_per_pixel as usize, &full.extra_samples)
                .is_some()
        };
        if req.format == OutputFormat::Npy {
            let mut raw_req = req.clone();
            raw_req.format = OutputFormat::RawF32;
            let raw = match raw_req.tile_size {
                256 => Box::pin(self.render_tile_sized::<256>(cog, &raw_req)).await?,
                512 => Box::pin(self.render_tile_sized::<512>(cog, &raw_req)).await?,
                other => {
                    return Err(EngineError::BadRequest(format!(
                        "unsupported tile_size={other}, expected 256 or 512"
                    )));
                }
            };
            let continuous_alpha = if has_alpha_channel {
                match req.tile_size {
                    256 => Box::pin(self.sample_continuous_alpha_tile::<256>(cog, req)).await?,
                    512 => Box::pin(self.sample_continuous_alpha_tile::<512>(cog, req)).await?,
                    _ => None,
                }
            } else {
                None
            };
            let nb = match req.bands {
                BandRequest::Rgb { .. }
                | BandRequest::SarRgb { .. }
                | BandRequest::Terrain { .. }
                | BandRequest::CustomExprRgb { .. } => 3,
                BandRequest::MultiBand { ref bands_1based } => bands_1based.len().max(1),
                BandRequest::CustomExprMulti { ref progs } => progs.len().max(1),
                _ => 1,
            };
            return Ok(cogzilla_kernel::wrap_raw_f32_as_npy_with_alpha_mask(
                raw,
                nb,
                req.tile_size,
                req.tile_size,
                continuous_alpha.as_deref(),
                true,
            ));
        }
        if has_alpha_channel
            && matches!(
                req.format,
                OutputFormat::Png | OutputFormat::Webp { .. } | OutputFormat::RawRgba
            )
        {
            let continuous_alpha = match req.tile_size {
                256 => Box::pin(self.sample_continuous_alpha_tile::<256>(cog, req)).await?,
                512 => Box::pin(self.sample_continuous_alpha_tile::<512>(cog, req)).await?,
                other => {
                    return Err(EngineError::BadRequest(format!(
                        "unsupported tile_size={other}, expected 256 or 512"
                    )));
                }
            };
            if let Some(alpha) = continuous_alpha {
                let mut rgba_req = req.clone();
                rgba_req.format = OutputFormat::RawRgba;
                let raw = match rgba_req.tile_size {
                    256 => Box::pin(self.render_tile_sized::<256>(cog, &rgba_req)).await?,
                    512 => Box::pin(self.render_tile_sized::<512>(cog, &rgba_req)).await?,
                    _ => unreachable!(),
                };
                let mut rgba = raw.to_vec();
                cogzilla_kernel::apply_continuous_alpha_to_rgba(&mut rgba, &alpha);
                let dim = req.tile_size as usize;
                return Ok(crate::stats::encode_rgba_output(
                    dim,
                    dim,
                    &rgba,
                    req.format,
                    true,
                ));
            }
        }
        match req.tile_size {
            256 => Box::pin(self.render_tile_sized::<256>(cog, req)).await,
            512 => Box::pin(self.render_tile_sized::<512>(cog, req)).await,
            other => Err(EngineError::BadRequest(format!(
                "unsupported tile_size={other}, expected 256 or 512"
            ))),
        }
    }

    /// Resample an RGBA / Gray+Alpha COG's `ExtraSamples` alpha band onto the tile's
    /// `CoordPlan` when any intersecting block contains intermediate translucency values
    /// (`0 < alpha < full_opaque`). Returns `Ok(None)` in O(blocks) for opaque or binary
    /// `0/255` alpha tiles so they stay on the fast path.
    async fn sample_continuous_alpha_tile<const N: usize>(
        &self,
        cog: &OpenedCog,
        req: &TileRequest,
    ) -> Result<Option<Vec<f32>>, EngineError> {
        let crs = self.get_or_build_crs(cog)?;
        let (target_rx, target_ry) = crs.target_resolution_for_tile(req.tile, N as u32);
        let ifd_idx = cog.index.select_ifd(target_rx, target_ry);
        let ifd = &cog.index.ifds[ifd_idx];
        let Some(alpha_idx) =
            cogzilla_codec::alpha_channel_index(ifd.samples_per_pixel as usize, &ifd.extra_samples)
        else {
            return Ok(None);
        };
        let Some(plan) = CoordPlan::<N>::build(
            &crs,
            &ifd.crs_to_pixel,
            ifd.gcp_poly.as_deref(),
            ifd.width,
            ifd.height,
            req.tile,
        ) else {
            return Ok(None);
        };
        let (tx0, tx1, ty0, ty1) = tile_fetch_range_for_plan(&plan, req.resample, false, ifd);
        let grid_w = tx1 - tx0 + 1;
        let grid_h = ty1 - ty0 + 1;
        let fetch_band = match ifd.planar_config {
            PlanarConfig::Chunky => 0u16,
            PlanarConfig::Planar => alpha_idx as u16,
        };
        let mut indices = Vec::with_capacity((grid_w * grid_h) as usize);
        for ty in ty0..=ty1 {
            for tx in tx0..=tx1 {
                indices.push(Some(ifd.tile_index(tx, ty, fetch_band) as u32));
            }
        }
        let tiles = self.store.get_blocks(cog, ifd_idx, &indices).await?;
        let band_in_decoded = match ifd.planar_config {
            PlanarConfig::Chunky => alpha_idx,
            PlanarConfig::Planar => 0,
        };
        let block_refs: Vec<Option<&cogzilla_codec::DecodedBand>> = tiles
            .iter()
            .map(|opt| opt.as_ref().and_then(|t| t.bands.get(band_in_decoded)))
            .collect();
        let full_opaque = cogzilla_kernel::alpha_full_opaque(ifd.sample_type);
        let alpha_src = BandSource::new(
            ifd.width,
            ifd.height,
            ifd.tile_width,
            ifd.tile_height,
            tx0,
            ty0,
            grid_w,
            grid_h,
            &block_refs,
        )
        .with_scale_offset(255.0 / full_opaque, 0.0);
        if !cogzilla_kernel::has_intermediate_alpha(&alpha_src, full_opaque) {
            return Ok(None);
        }
        let style = SingleBandStyle {
            rescale_min: 0.0,
            rescale_max: 255.0,
            colormap: ColormapName::Grayscale,
            resample: req.resample,
            format: OutputFormat::RawF32,
        };
        let raw = render_single_band::<N>(&plan, &alpha_src, &style);
        let mut alpha = Vec::with_capacity(N * N);
        for chunk in raw.chunks_exact(4) {
            alpha.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }
        Ok(Some(alpha))
    }

    /// Render a batch of XYZ tiles for a single COG (e.g. a map viewport burst or pyramid
    /// cache-warming pass).
    ///
    /// When built with `--features gpu` and a hardware GPU is active, eligible single-band
    /// separable tiles that share the same `(ifd_idx, band, tile-block neighborhood, style)` are
    /// fused into a single 3D GPU micro-batch (`GpuTileBatcher`), amortizing the Metal/Vulkan
    /// dispatch floor across all `K` tiles. Any remaining tiles fall back transparently to
    /// `render_tile`.
    pub async fn render_tiles_batch(
        &self,
        cog: &OpenedCog,
        reqs: &[TileRequest],
    ) -> Vec<Result<Bytes, EngineError>> {
        if reqs.is_empty() {
            return Vec::new();
        }
        #[allow(unused_mut)]
        let mut results: Vec<Option<Result<Bytes, EngineError>>> =
            (0..reqs.len()).map(|_| None).collect();

        if reqs.len() >= 2
            && let Some(ref gpu) = self.gpu
            && let Ok(crs) = self.get_or_build_crs(cog)
        {
            #[derive(Hash, PartialEq, Eq)]
            struct BatchKey {
                ifd_idx: usize,
                band0: u16,
                tx0: u32,
                ty0: u32,
                grid_w: u32,
                grid_h: u32,
                tile_size: u32,
                resample: u8,
                rmin_bits: u32,
                rmax_bits: u32,
                colormap: String,
                format_tag: u8,
                format_q: u32,
            }
            struct BatchItem {
                req_idx: usize,
                u: Vec<f32>,
                v: Vec<f32>,
            }
            let mut groups: std::collections::HashMap<BatchKey, (SingleBandStyle, Vec<BatchItem>)> =
                std::collections::HashMap::new();

            for (idx, req) in reqs.iter().enumerate() {
                if req.unscale {
                    continue;
                }
                let BandRequest::Single {
                    band_1based,
                    rescale,
                    colormap,
                } = req.bands
                else {
                    continue;
                };
                if req.tile_size != 256 && req.tile_size != 512 {
                    continue;
                }
                let (format_tag, format_q) = match req.format {
                    OutputFormat::Png => (0u8, 0u32),
                    OutputFormat::RawRgba => (1u8, 0u32),
                    OutputFormat::Webp { quality } => {
                        (2u8, quality.map(f32::to_bits).unwrap_or(0))
                    }
                    OutputFormat::Jpeg { quality } => (3u8, quality as u32),
                    _ => continue,
                };
                let (target_rx, target_ry) = crs.target_resolution_for_tile(req.tile, req.tile_size);
                let ifd_idx = cog.index.select_ifd(target_rx, target_ry);
                let ifd = &cog.index.ifds[ifd_idx];
                if cogzilla_codec::alpha_channel_index(
                    ifd.samples_per_pixel as usize,
                    &ifd.extra_samples,
                )
                .is_some()
                {
                    continue;
                }
                let band0 = band_1based.saturating_sub(1);
                if band0 >= ifd.samples_per_pixel {
                    continue;
                }
                let pad = match req.resample {
                    ResampleMode::Nearest => 0i64,
                    ResampleMode::Bilinear => 1i64,
                    ResampleMode::Cubic => 2i64,
                    _ => 3i64,
                };
                let uv_bounds: Option<(Vec<f32>, Vec<f32>, f32, f32, f32, f32)> =
                    if req.tile_size == 256 {
                        CoordPlan::<256>::build(
                            &crs,
                            &ifd.crs_to_pixel,
                            ifd.gcp_poly.as_deref(),
                            ifd.width,
                            ifd.height,
                            req.tile,
                        )
                        .and_then(|p| match p {
                            CoordPlan::Separable1D { u, v, .. } => {
                                let (mut u0, mut u1) = (f32::INFINITY, f32::NEG_INFINITY);
                                let (mut v0, mut v1) = (f32::INFINITY, f32::NEG_INFINITY);
                                for &x in u.iter() {
                                    u0 = u0.min(x);
                                    u1 = u1.max(x);
                                }
                                for &y in v.iter() {
                                    v0 = v0.min(y);
                                    v1 = v1.max(y);
                                }
                                Some((u.to_vec(), v.to_vec(), u0, u1, v0, v1))
                            }
                            _ => None,
                        })
                    } else {
                        CoordPlan::<512>::build(
                            &crs,
                            &ifd.crs_to_pixel,
                            ifd.gcp_poly.as_deref(),
                            ifd.width,
                            ifd.height,
                            req.tile,
                        )
                        .and_then(|p| match p {
                            CoordPlan::Separable1D { u, v, .. } => {
                                let (mut u0, mut u1) = (f32::INFINITY, f32::NEG_INFINITY);
                                let (mut v0, mut v1) = (f32::INFINITY, f32::NEG_INFINITY);
                                for &x in u.iter() {
                                    u0 = u0.min(x);
                                    u1 = u1.max(x);
                                }
                                for &y in v.iter() {
                                    v0 = v0.min(y);
                                    v1 = v1.max(y);
                                }
                                Some((u.to_vec(), v.to_vec(), u0, u1, v0, v1))
                            }
                            _ => None,
                        })
                    };
                let Some((u_vec, v_vec, min_u, max_u, min_v, max_v)) = uv_bounds else {
                    continue;
                };
                if !min_u.is_finite() || !min_v.is_finite() {
                    continue;
                }
                let px0 = (min_u.floor() as i64 - pad).clamp(0, ifd.width.saturating_sub(1) as i64) as u32;
                let py0 = (min_v.floor() as i64 - pad).clamp(0, ifd.height.saturating_sub(1) as i64) as u32;
                let px1 = (max_u.ceil() as i64 + pad).clamp(0, ifd.width.saturating_sub(1) as i64) as u32;
                let py1 = (max_v.ceil() as i64 + pad).clamp(0, ifd.height.saturating_sub(1) as i64) as u32;
                let tx0 = px0 / ifd.tile_width;
                let ty0 = py0 / ifd.tile_height;
                let tx1 = (px1 / ifd.tile_width).min(ifd.tiles_across.saturating_sub(1));
                let ty1 = (py1 / ifd.tile_height).min(ifd.tiles_down.saturating_sub(1));
                let grid_w = tx1 - tx0 + 1;
                let grid_h = ty1 - ty0 + 1;

                let key = BatchKey {
                    ifd_idx,
                    band0,
                    tx0,
                    ty0,
                    grid_w,
                    grid_h,
                    tile_size: req.tile_size,
                    resample: req.resample as u8,
                    rmin_bits: rescale.0.to_bits(),
                    rmax_bits: rescale.1.to_bits(),
                    colormap: format!("{colormap:?}"),
                    format_tag,
                    format_q,
                };
                let style = SingleBandStyle {
                    rescale_min: rescale.0,
                    rescale_max: rescale.1,
                    colormap,
                    resample: req.resample,
                    format: req.format,
                };
                groups
                    .entry(key)
                    .or_insert_with(|| (style, Vec::new()))
                    .1
                    .push(BatchItem {
                        req_idx: idx,
                        u: u_vec,
                        v: v_vec,
                    });
            }

            for (key, (style, items)) in groups {
                if items.len() < 2 {
                    continue;
                }
                let ifd = &cog.index.ifds[key.ifd_idx];
                let mut indices = Vec::with_capacity((key.grid_w * key.grid_h) as usize);
                for ty in key.ty0..key.ty0 + key.grid_h {
                    for tx in key.tx0..key.tx0 + key.grid_w {
                        indices.push(Some(ifd.tile_index(tx, ty, key.band0) as u32));
                    }
                }
                let Ok(tiles) = self.store.get_blocks(cog, key.ifd_idx, &indices).await else {
                    continue;
                };
                let band_in_decoded = match ifd.planar_config {
                    PlanarConfig::Chunky => key.band0 as usize,
                    PlanarConfig::Planar => 0,
                };
                let block_refs: Vec<Option<&cogzilla_codec::DecodedBand>> = tiles
                    .iter()
                    .map(|opt| opt.as_ref().and_then(|t| t.bands.get(band_in_decoded)))
                    .collect();
                let src = BandSource::new(
                    ifd.width,
                    ifd.height,
                    ifd.tile_width,
                    ifd.tile_height,
                    key.tx0,
                    key.ty0,
                    key.grid_w,
                    key.grid_h,
                    &block_refs,
                );
                let uvs: Vec<(&[f32], &[f32])> = items
                    .iter()
                    .map(|it| (it.u.as_slice(), it.v.as_slice()))
                    .collect();
                if let Some(batch_out) = gpu.try_render_single_band_batch(
                    cog.id,
                    key.ifd_idx,
                    key.band0,
                    key.tile_size as usize,
                    &uvs,
                    &src,
                    &style,
                ) {
                    self.stats
                        .tiles_served
                        .fetch_add(items.len() as u64, Ordering::Relaxed);
                    self.stats.gpu_dispatches.fetch_add(1, Ordering::Relaxed);
                    for (item, bytes) in items.into_iter().zip(batch_out) {
                        results[item.req_idx] = Some(Ok(bytes));
                    }
                }
            }
        }

        let mut final_out = Vec::with_capacity(reqs.len());
        for (slot, req) in results.into_iter().zip(reqs.iter()) {
            match slot {
                Some(res) => final_out.push(res),
                None => final_out.push(self.render_tile(cog, req).await),
            }
        }
        final_out
    }

    async fn render_tile_sized<const N: usize>(
        &self,
        cog: &OpenedCog,
        req: &TileRequest,
    ) -> Result<Bytes, EngineError> {
        let crs = self.get_or_build_crs(cog)?;
        let (target_rx, target_ry) = crs.target_resolution_for_tile(req.tile, N as u32);
        let ifd_idx = cog.index.select_ifd(target_rx, target_ry);
        let ifd = &cog.index.ifds[ifd_idx];

        // Zero-decode pass-through for WebMercator-aligned JPEG/WebP COGs:
        // when the requested XYZ tile matches an internal IFD tile 1:1 with identity RGB
        // styling and matching compression, stream raw compressed bytes straight through
        // without decoding, resampling, or re-encoding.
        if let Some(bytes) = self
            .try_zero_decode_passthrough::<N>(cog, &crs, ifd_idx, ifd, req)
            .await?
        {
            return Ok(bytes);
        }

        let Some(plan) = CoordPlan::<N>::build(
            &crs,
            &ifd.crs_to_pixel,
            ifd.gcp_poly.as_deref(),
            ifd.width,
            ifd.height,
            req.tile,
        ) else {
            self.stats.empty_short_circuits.fetch_add(1, Ordering::Relaxed);
            let nb = match req.bands {
                BandRequest::Rgb { .. }
                | BandRequest::SarRgb { .. }
                | BandRequest::Terrain { .. }
                | BandRequest::CustomExprRgb { .. } => 3,
                BandRequest::MultiBand { ref bands_1based } => bands_1based.len().max(1),
                BandRequest::CustomExprMulti { ref progs } => progs.len().max(1),
                _ => 1,
            };
            return Ok(empty_tile_bytes(N as u32, req.format, nb));
        };

        let (tx0, tx1, ty0, ty1) = tile_fetch_range_for_plan(
            &plan,
            req.resample,
            matches!(req.bands, BandRequest::Dem { .. }),
            ifd,
        );
        let grid_w = tx1 - tx0 + 1;
        let grid_h = ty1 - ty0 + 1;
        let grid_len = (grid_w * grid_h) as usize;

        match &req.bands {
            BandRequest::Single {
                band_1based,
                rescale,
                colormap,
            } => {
                let band0 = band_1based.saturating_sub(1);
                if band0 >= ifd.samples_per_pixel {
                    return Err(EngineError::BadRequest(format!(
                        "band {} out of range (dataset has {} bands)",
                        band_1based, ifd.samples_per_pixel
                    )));
                }
                let mut indices = Vec::with_capacity(grid_len);
                for ty in ty0..=ty1 {
                    for tx in tx0..=tx1 {
                        let idx = ifd.tile_index(tx, ty, band0) as u32;
                        indices.push(Some(idx));
                    }
                }
                let tiles = self.store.get_blocks(cog, ifd_idx, &indices).await?;
                let band_in_decoded = match ifd.planar_config {
                    PlanarConfig::Chunky => band0 as usize,
                    PlanarConfig::Planar => 0,
                };
                let block_refs: Vec<Option<&cogzilla_codec::DecodedBand>> = tiles
                    .iter()
                    .map(|opt| opt.as_ref().and_then(|t| t.bands.get(band_in_decoded)))
                    .collect();

                let apply_unscale = req.unscale && needs_source_unscale(&req.bands, req.format);
                let (sc, off) = if apply_unscale {
                    (
                        cog.index.scales.get(band0 as usize).copied().unwrap_or(1.0),
                        cog.index.offsets.get(band0 as usize).copied().unwrap_or(0.0),
                    )
                } else {
                    (1.0, 0.0)
                };
                let src = BandSource::new(
                    ifd.width,
                    ifd.height,
                    ifd.tile_width,
                    ifd.tile_height,
                    tx0,
                    ty0,
                    grid_w,
                    grid_h,
                    &block_refs,
                )
                .with_scale_offset(sc, off);
                if src.all_nodata {
                    self.stats.empty_short_circuits.fetch_add(1, Ordering::Relaxed);
                }
                let style = SingleBandStyle {
                    rescale_min: rescale.0,
                    rescale_max: rescale.1,
                    colormap: *colormap,
                    resample: req.resample,
                    format: req.format,
                };
                if let Some(gpu) = &self.gpu
                    && let Some(bytes) = gpu
                        .try_render_single_band::<N>(
                            cog.id, ifd_idx, band0, tx0, ty0, &plan, &src, &style,
                        )
                        .await
                    {
                        self.stats.gpu_dispatches.fetch_add(1, Ordering::Relaxed);
                        return Ok(bytes);
                    }
                Ok(run_cpu_render(N, style.resample, || {
                    render_single_band::<N>(&plan, &src, &style)
                }))
            }
            BandRequest::Terrain {
                band_1based,
                algorithm,
                math,
            } => {
                let band0 = band_1based.saturating_sub(1);
                // Band math reads its bands through the shared (boxed) reader; one band keeps
                // the inline read, which the tile path is timed on.
                let math_read = match math {
                    Some(prog) => Some(
                        Box::pin(self.read_terrain_blocks(cog, ifd_idx, (tx0, tx1, ty0, ty1), *band_1based, Some(prog), req.unscale))
                            .await?,
                    ),
                    None => None,
                };
                let tiles;
                let (block_refs, (sc, off)) = if let Some(read) = &math_read {
                    (read.refs(*band_1based), (1.0, 0.0))
                } else {
                    if band0 >= ifd.samples_per_pixel {
                        return Err(EngineError::BadRequest(format!(
                            "band {} out of range (dataset has {} bands)",
                            band_1based, ifd.samples_per_pixel
                        )));
                    }
                    let mut indices = Vec::with_capacity((grid_w * grid_h) as usize);
                    for ty in ty0..=ty1 {
                        for tx in tx0..=tx1 {
                            let idx = ifd.tile_index(tx, ty, band0) as u32;
                            indices.push(Some(idx));
                        }
                    }
                    tiles = self.store.get_blocks(cog, ifd_idx, &indices).await?;
                    let band_in_decoded = match ifd.planar_config {
                        PlanarConfig::Chunky => band0 as usize,
                        PlanarConfig::Planar => 0,
                    };
                    let block_refs: Vec<Option<&cogzilla_codec::DecodedBand>> = tiles
                        .iter()
                        .map(|opt| opt.as_ref().and_then(|t| t.bands.get(band_in_decoded)))
                        .collect();
                    let so = if req.unscale {
                        (
                            cog.index.scales.get(band0 as usize).copied().unwrap_or(1.0),
                            cog.index.offsets.get(band0 as usize).copied().unwrap_or(0.0),
                        )
                    } else {
                        (1.0, 0.0)
                    };
                    (block_refs, so)
                };
                let src = BandSource::new(
                    ifd.width,
                    ifd.height,
                    ifd.tile_width,
                    ifd.tile_height,
                    tx0,
                    ty0,
                    grid_w,
                    grid_h,
                    &block_refs,
                )
                .with_scale_offset(sc, off);
                if src.all_nodata {
                    self.stats.empty_short_circuits.fetch_add(1, Ordering::Relaxed);
                }
                if let Some(gpu) = self.gpu.as_ref()
                    && let Some(bytes) = gpu
                        .try_render_terrain::<N>(
                            math.is_none().then_some((cog.id, ifd_idx, band0)),
                            tx0,
                            ty0,
                            &plan,
                            &src,
                            *algorithm,
                            req.resample,
                            req.format,
                        )
                        .await
                    {
                        self.stats.gpu_dispatches.fetch_add(1, Ordering::Relaxed);
                        return Ok(bytes);
                    }
                Ok(run_cpu_render(N, req.resample, || {
                    render_terrain_tile::<N>(&plan, &src, *algorithm, req.resample, req.format)
                }))
            }
            BandRequest::Dem {
                band_1based,
                algorithm,
                rescale,
                colormap,
                math,
            } => {
                let band0 = band_1based.saturating_sub(1);
                // Band math reads its bands through the shared (boxed) reader; one band keeps
                // the inline read, which the tile path is timed on.
                let math_read = match math {
                    Some(prog) => Some(
                        Box::pin(self.read_terrain_blocks(cog, ifd_idx, (tx0, tx1, ty0, ty1), *band_1based, Some(prog), req.unscale))
                            .await?,
                    ),
                    None => None,
                };
                let tiles;
                let (block_refs, (sc, off)) = if let Some(read) = &math_read {
                    (read.refs(*band_1based), (1.0, 0.0))
                } else {
                    if band0 >= ifd.samples_per_pixel {
                        return Err(EngineError::BadRequest(format!(
                            "band {} out of range (dataset has {} bands)",
                            band_1based, ifd.samples_per_pixel
                        )));
                    }
                    let mut indices = Vec::with_capacity((grid_w * grid_h) as usize);
                    for ty in ty0..=ty1 {
                        for tx in tx0..=tx1 {
                            let idx = ifd.tile_index(tx, ty, band0) as u32;
                            indices.push(Some(idx));
                        }
                    }
                    tiles = self.store.get_blocks(cog, ifd_idx, &indices).await?;
                    let band_in_decoded = match ifd.planar_config {
                        PlanarConfig::Chunky => band0 as usize,
                        PlanarConfig::Planar => 0,
                    };
                    let block_refs: Vec<Option<&cogzilla_codec::DecodedBand>> = tiles
                        .iter()
                        .map(|opt| opt.as_ref().and_then(|t| t.bands.get(band_in_decoded)))
                        .collect();
                    let so = if req.unscale {
                        (
                            cog.index.scales.get(band0 as usize).copied().unwrap_or(1.0),
                            cog.index.offsets.get(band0 as usize).copied().unwrap_or(0.0),
                        )
                    } else {
                        (1.0, 0.0)
                    };
                    (block_refs, so)
                };
                let src = BandSource::new(
                    ifd.width,
                    ifd.height,
                    ifd.tile_width,
                    ifd.tile_height,
                    tx0,
                    ty0,
                    grid_w,
                    grid_h,
                    &block_refs,
                )
                .with_scale_offset(sc, off);
                if src.all_nodata {
                    self.stats.empty_short_circuits.fetch_add(1, Ordering::Relaxed);
                }
                let (bx0, _, bx1, _) = tile_bounds_3857(req.tile);
                // The tile's pixel size in Web Mercator metres, as TiTiler's slope divides by the
                // transform's pixel size.
                let pixel_size_m = ((bx1 - bx0) / N as f64) as f32;
                let style = SingleBandStyle {
                    rescale_min: rescale.0,
                    rescale_max: rescale.1,
                    colormap: *colormap,
                    resample: req.resample,
                    format: req.format,
                };
                if let Some(gpu) = self.gpu.as_ref()
                    && let Some(bytes) = gpu
                        .try_render_dem::<N>(
                            math.is_none().then_some((cog.id, ifd_idx, band0)),
                            tx0,
                            ty0,
                            &plan,
                            &src,
                            *algorithm,
                            pixel_size_m,
                            &style,
                        )
                        .await
                    {
                        self.stats.gpu_dispatches.fetch_add(1, Ordering::Relaxed);
                        return Ok(bytes);
                    }
                Ok(run_cpu_render(N, style.resample, || {
                    render_dem_algorithm::<N>(&plan, &src, *algorithm, pixel_size_m, &style)
                }))
            }
            BandRequest::Spectral1 {
                expr,
                band_1based,
                rescale,
                colormap,
            } => {
                let band0 = band_1based.saturating_sub(1);
                if band0 >= ifd.samples_per_pixel {
                    return Err(EngineError::BadRequest(format!(
                        "band {} out of range (dataset has {} bands)",
                        band_1based, ifd.samples_per_pixel
                    )));
                }
                let mut indices = Vec::with_capacity(grid_len);
                for ty in ty0..=ty1 {
                    for tx in tx0..=tx1 {
                        indices.push(Some(ifd.tile_index(tx, ty, band0) as u32));
                    }
                }
                let tiles = self.store.get_blocks(cog, ifd_idx, &indices).await?;
                let band_in_decoded = match ifd.planar_config {
                    PlanarConfig::Chunky => band0 as usize,
                    PlanarConfig::Planar => 0,
                };
                let block_refs: Vec<Option<&cogzilla_codec::DecodedBand>> = tiles
                    .iter()
                    .map(|opt| opt.as_ref().and_then(|t| t.bands.get(band_in_decoded)))
                    .collect();
                let (sc, off) = if req.unscale {
                    (
                        cog.index.scales.get(band0 as usize).copied().unwrap_or(1.0),
                        cog.index.offsets.get(band0 as usize).copied().unwrap_or(0.0),
                    )
                } else {
                    (1.0, 0.0)
                };
                let src = BandSource::new(
                    ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0, grid_w,
                    grid_h, &block_refs,
                )
                .with_scale_offset(sc, off);
                let style = SingleBandStyle {
                    rescale_min: rescale.0,
                    rescale_max: rescale.1,
                    colormap: *colormap,
                    resample: req.resample,
                    format: req.format,
                };
                if let Some(gpu) = &self.gpu
                    && let Some(bytes) = gpu
                        .try_render_spectral::<N>(
                            cog.id,
                            ifd_idx,
                            &[*band_1based],
                            &plan,
                            std::slice::from_ref(&src),
                            *expr,
                            &style,
                        )
                        .await
                    {
                        self.stats.gpu_dispatches.fetch_add(1, Ordering::Relaxed);
                        return Ok(bytes);
                    }
                Ok(run_cpu_render(N, style.resample, || {
                    render_bandmath1::<N>(&plan, &src, *expr, &style)
                }))
            }
            BandRequest::SarRgb {
                bands_1based,
                rescale_min,
                rescale_max,
            } => {
                let b0 = [
                    bands_1based[0].saturating_sub(1),
                    bands_1based[1].saturating_sub(1),
                ];
                for &b in &b0 {
                    if b >= ifd.samples_per_pixel {
                        return Err(EngineError::BadRequest(format!(
                            "band {} out of range (dataset has {} bands)",
                            b + 1,
                            ifd.samples_per_pixel
                        )));
                    }
                }
                let style = RgbStyle {
                    rescale_min: *rescale_min,
                    rescale_max: *rescale_max,
                    resample: req.resample,
                    format: req.format,
                };
                let so = |b: u16| -> (f32, f32) {
                    if req.unscale {
                        (
                            cog.index.scales.get(b as usize).copied().unwrap_or(1.0),
                            cog.index.offsets.get(b as usize).copied().unwrap_or(0.0),
                        )
                    } else {
                        (1.0, 0.0)
                    }
                };
                match ifd.planar_config {
                    PlanarConfig::Chunky => {
                        let mut indices = Vec::with_capacity(grid_len);
                        for ty in ty0..=ty1 {
                            for tx in tx0..=tx1 {
                                indices.push(Some(ifd.tile_index(tx, ty, 0) as u32));
                            }
                        }
                        let tiles = self.store.get_blocks(cog, ifd_idx, &indices).await?;
                        let a_refs: Vec<_> = tiles
                            .iter()
                            .map(|o| o.as_ref().and_then(|t| t.bands.get(b0[0] as usize)))
                            .collect();
                        let b_refs: Vec<_> = tiles
                            .iter()
                            .map(|o| o.as_ref().and_then(|t| t.bands.get(b0[1] as usize)))
                            .collect();
                        let (s0, o0) = so(b0[0]);
                        let (s1, o1) = so(b0[1]);
                        let srcs = [
                            BandSource::new(
                                ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0,
                                grid_w, grid_h, &a_refs,
                            )
                            .with_scale_offset(s0, o0),
                            BandSource::new(
                                ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0,
                                grid_w, grid_h, &b_refs,
                            )
                            .with_scale_offset(s1, o1),
                        ];
                        if let Some(gpu) = &self.gpu
                            && let Some(bytes) = gpu
                                .try_render_sar_rgb::<N>(
                                    cog.id,
                                    ifd_idx,
                                    *bands_1based,
                                    &plan,
                                    &srcs,
                                    &style,
                                )
                                .await
                            {
                                self.stats.gpu_dispatches.fetch_add(1, Ordering::Relaxed);
                                return Ok(bytes);
                            }
                        Ok(run_cpu_render(N, style.resample, || {
                    render_sar_rgb::<N>(&plan, &srcs, &style)
                }))
                    }
                    PlanarConfig::Planar => {
                        let mut indices = Vec::with_capacity(grid_len * 2);
                        for &band in &b0 {
                            for ty in ty0..=ty1 {
                                for tx in tx0..=tx1 {
                                    indices.push(Some(ifd.tile_index(tx, ty, band) as u32));
                                }
                            }
                        }
                        let all_tiles = self.store.get_blocks(cog, ifd_idx, &indices).await?;
                        let a_refs: Vec<_> = all_tiles[0..grid_len]
                            .iter()
                            .map(|o| o.as_ref().and_then(|t| t.bands.first()))
                            .collect();
                        let b_refs: Vec<_> = all_tiles[grid_len..grid_len * 2]
                            .iter()
                            .map(|o| o.as_ref().and_then(|t| t.bands.first()))
                            .collect();
                        let (s0, o0) = so(b0[0]);
                        let (s1, o1) = so(b0[1]);
                        let srcs = [
                            BandSource::new(
                                ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0,
                                grid_w, grid_h, &a_refs,
                            )
                            .with_scale_offset(s0, o0),
                            BandSource::new(
                                ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0,
                                grid_w, grid_h, &b_refs,
                            )
                            .with_scale_offset(s1, o1),
                        ];
                        if let Some(gpu) = &self.gpu
                            && let Some(bytes) = gpu
                                .try_render_sar_rgb::<N>(
                                    cog.id,
                                    ifd_idx,
                                    *bands_1based,
                                    &plan,
                                    &srcs,
                                    &style,
                                )
                                .await
                            {
                                self.stats.gpu_dispatches.fetch_add(1, Ordering::Relaxed);
                                return Ok(bytes);
                            }
                        Ok(run_cpu_render(N, style.resample, || {
                    render_sar_rgb::<N>(&plan, &srcs, &style)
                }))
                    }
                }
            }
            BandRequest::Rgb {
                bands_1based,
                rescale_min,
                rescale_max,
            } => {
                let b0 = [
                    bands_1based[0].saturating_sub(1),
                    bands_1based[1].saturating_sub(1),
                    bands_1based[2].saturating_sub(1),
                ];
                for &b in &b0 {
                    if b >= ifd.samples_per_pixel {
                        return Err(EngineError::BadRequest(format!(
                            "band {} out of range (dataset has {} bands)",
                            b + 1,
                            ifd.samples_per_pixel
                        )));
                    }
                }

                let style = RgbStyle {
                    rescale_min: *rescale_min,
                    rescale_max: *rescale_max,
                    resample: req.resample,
                    format: req.format,
                };

                match ifd.planar_config {
                    PlanarConfig::Chunky => {
                        // All 3 bands live inside the exact same decoded tiles!
                        let mut indices = Vec::with_capacity(grid_len);
                        for ty in ty0..=ty1 {
                            for tx in tx0..=tx1 {
                                indices.push(Some(ifd.tile_index(tx, ty, 0) as u32));
                            }
                        }
                        let tiles = self.store.get_blocks(cog, ifd_idx, &indices).await?;
                        let r_refs: Vec<_> = tiles
                            .iter()
                            .map(|o| o.as_ref().and_then(|t| t.bands.get(b0[0] as usize)))
                            .collect();
                        let g_refs: Vec<_> = tiles
                            .iter()
                            .map(|o| o.as_ref().and_then(|t| t.bands.get(b0[1] as usize)))
                            .collect();
                        let b_refs: Vec<_> = tiles
                            .iter()
                            .map(|o| o.as_ref().and_then(|t| t.bands.get(b0[2] as usize)))
                            .collect();

                        let apply_unscale =
                            req.unscale && needs_source_unscale(&req.bands, req.format);
                        let so = |b: u16| -> (f32, f32) {
                            if apply_unscale {
                                (
                                    cog.index.scales.get(b as usize).copied().unwrap_or(1.0),
                                    cog.index.offsets.get(b as usize).copied().unwrap_or(0.0),
                                )
                            } else {
                                (1.0, 0.0)
                            }
                        };
                        let (s0, o0) = so(b0[0]);
                        let (s1, o1) = so(b0[1]);
                        let (s2, o2) = so(b0[2]);
                        let srcs = [
                            BandSource::new(
                                ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0,
                                grid_w, grid_h, &r_refs,
                            )
                            .with_scale_offset(s0, o0),
                            BandSource::new(
                                ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0,
                                grid_w, grid_h, &g_refs,
                            )
                            .with_scale_offset(s1, o1),
                            BandSource::new(
                                ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0,
                                grid_w, grid_h, &b_refs,
                            )
                            .with_scale_offset(s2, o2),
                        ];
                        if srcs[0].all_nodata && srcs[1].all_nodata && srcs[2].all_nodata {
                            self.stats.empty_short_circuits.fetch_add(1, Ordering::Relaxed);
                        }
                        if let Some(gpu) = &self.gpu
                            && let Some(bytes) = gpu
                                .try_render_rgb::<N>(cog.id, ifd_idx, *bands_1based, &plan, &srcs, &style)
                                .await
                            {
                                self.stats.gpu_dispatches.fetch_add(1, Ordering::Relaxed);
                                return Ok(bytes);
                            }
                        Ok(run_cpu_render(N, style.resample, || {
                            render_rgb_bands::<N>(&plan, &srcs, &style)
                        }))
                    }
                    PlanarConfig::Planar => {
                        // Fetch all 3 planes in one coalesced call
                        let mut indices = Vec::with_capacity(grid_len * 3);
                        for &band in &b0 {
                            for ty in ty0..=ty1 {
                                for tx in tx0..=tx1 {
                                    indices.push(Some(ifd.tile_index(tx, ty, band) as u32));
                                }
                            }
                        }
                        let tiles = self.store.get_blocks(cog, ifd_idx, &indices).await?;
                        let r_refs: Vec<_> = tiles[0..grid_len]
                            .iter()
                            .map(|o| o.as_ref().and_then(|t| t.bands.first()))
                            .collect();
                        let g_refs: Vec<_> = tiles[grid_len..2 * grid_len]
                            .iter()
                            .map(|o| o.as_ref().and_then(|t| t.bands.first()))
                            .collect();
                        let b_refs: Vec<_> = tiles[2 * grid_len..3 * grid_len]
                            .iter()
                            .map(|o| o.as_ref().and_then(|t| t.bands.first()))
                            .collect();

                        let apply_unscale =
                            req.unscale && needs_source_unscale(&req.bands, req.format);
                        let so = |b: u16| -> (f32, f32) {
                            if apply_unscale {
                                (
                                    cog.index.scales.get(b as usize).copied().unwrap_or(1.0),
                                    cog.index.offsets.get(b as usize).copied().unwrap_or(0.0),
                                )
                            } else {
                                (1.0, 0.0)
                            }
                        };
                        let (s0, o0) = so(b0[0]);
                        let (s1, o1) = so(b0[1]);
                        let (s2, o2) = so(b0[2]);
                        let srcs = [
                            BandSource::new(
                                ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0,
                                grid_w, grid_h, &r_refs,
                            )
                            .with_scale_offset(s0, o0),
                            BandSource::new(
                                ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0,
                                grid_w, grid_h, &g_refs,
                            )
                            .with_scale_offset(s1, o1),
                            BandSource::new(
                                ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0,
                                grid_w, grid_h, &b_refs,
                            )
                            .with_scale_offset(s2, o2),
                        ];
                        if srcs[0].all_nodata && srcs[1].all_nodata && srcs[2].all_nodata {
                            self.stats.empty_short_circuits.fetch_add(1, Ordering::Relaxed);
                        }
                        if let Some(gpu) = &self.gpu
                            && let Some(bytes) = gpu
                                .try_render_rgb::<N>(cog.id, ifd_idx, *bands_1based, &plan, &srcs, &style)
                                .await
                            {
                                self.stats.gpu_dispatches.fetch_add(1, Ordering::Relaxed);
                                return Ok(bytes);
                            }
                        Ok(run_cpu_render(N, style.resample, || {
                            render_rgb_bands::<N>(&plan, &srcs, &style)
                        }))
                    }
                }
            }
            BandRequest::Spectral2 {
                expr,
                bands_1based,
                rescale,
                colormap,
            } => {
                let b0 = [
                    bands_1based[0].saturating_sub(1),
                    bands_1based[1].saturating_sub(1),
                ];
                for &b in &b0 {
                    if b >= ifd.samples_per_pixel {
                        return Err(EngineError::BadRequest(format!(
                            "band {} out of range (dataset has {} bands)",
                            b + 1,
                            ifd.samples_per_pixel
                        )));
                    }
                }
                let style = SingleBandStyle {
                    rescale_min: rescale.0,
                    rescale_max: rescale.1,
                    colormap: *colormap,
                    resample: req.resample,
                    format: req.format,
                };
                let so = |b: u16| -> (f32, f32) {
                    if req.unscale {
                        (
                            cog.index.scales.get(b as usize).copied().unwrap_or(1.0),
                            cog.index.offsets.get(b as usize).copied().unwrap_or(0.0),
                        )
                    } else {
                        (1.0, 0.0)
                    }
                };
                match ifd.planar_config {
                    PlanarConfig::Chunky => {
                        let mut indices = Vec::with_capacity(grid_len);
                        for ty in ty0..=ty1 {
                            for tx in tx0..=tx1 {
                                indices.push(Some(ifd.tile_index(tx, ty, 0) as u32));
                            }
                        }
                        let tiles = self.store.get_blocks(cog, ifd_idx, &indices).await?;
                        let a_refs: Vec<_> = tiles
                            .iter()
                            .map(|o| o.as_ref().and_then(|t| t.bands.get(b0[0] as usize)))
                            .collect();
                        let b_refs: Vec<_> = tiles
                            .iter()
                            .map(|o| o.as_ref().and_then(|t| t.bands.get(b0[1] as usize)))
                            .collect();
                        let (s0, o0) = so(b0[0]);
                        let (s1, o1) = so(b0[1]);
                        let srcs = [
                            BandSource::new(
                                ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0,
                                grid_w, grid_h, &a_refs,
                            )
                            .with_scale_offset(s0, o0),
                            BandSource::new(
                                ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0,
                                grid_w, grid_h, &b_refs,
                            )
                            .with_scale_offset(s1, o1),
                        ];
                        if let Some(gpu) = &self.gpu
                            && let Some(bytes) = gpu
                                .try_render_spectral::<N>(
                                    cog.id,
                                    ifd_idx,
                                    bands_1based,
                                    &plan,
                                    &srcs,
                                    *expr,
                                    &style,
                                )
                                .await
                            {
                                self.stats.gpu_dispatches.fetch_add(1, Ordering::Relaxed);
                                return Ok(bytes);
                            }
                        Ok(run_cpu_render(N, style.resample, || {
                    render_bandmath2::<N>(&plan, &srcs, *expr, &style)
                }))
                    }
                    PlanarConfig::Planar => {
                        let mut indices = Vec::with_capacity(grid_len * 2);
                        for &band in &b0 {
                            for ty in ty0..=ty1 {
                                for tx in tx0..=tx1 {
                                    indices.push(Some(ifd.tile_index(tx, ty, band) as u32));
                                }
                            }
                        }
                        let tiles = self.store.get_blocks(cog, ifd_idx, &indices).await?;
                        let a_refs: Vec<_> = tiles[0..grid_len]
                            .iter()
                            .map(|o| o.as_ref().and_then(|t| t.bands.first()))
                            .collect();
                        let b_refs: Vec<_> = tiles[grid_len..2 * grid_len]
                            .iter()
                            .map(|o| o.as_ref().and_then(|t| t.bands.first()))
                            .collect();
                        let (s0, o0) = so(b0[0]);
                        let (s1, o1) = so(b0[1]);
                        let srcs = [
                            BandSource::new(
                                ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0,
                                grid_w, grid_h, &a_refs,
                            )
                            .with_scale_offset(s0, o0),
                            BandSource::new(
                                ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0,
                                grid_w, grid_h, &b_refs,
                            )
                            .with_scale_offset(s1, o1),
                        ];
                        if let Some(gpu) = &self.gpu
                            && let Some(bytes) = gpu
                                .try_render_spectral::<N>(
                                    cog.id,
                                    ifd_idx,
                                    bands_1based,
                                    &plan,
                                    &srcs,
                                    *expr,
                                    &style,
                                )
                                .await
                            {
                                self.stats.gpu_dispatches.fetch_add(1, Ordering::Relaxed);
                                return Ok(bytes);
                            }
                        Ok(run_cpu_render(N, style.resample, || {
                    render_bandmath2::<N>(&plan, &srcs, *expr, &style)
                }))
                    }
                }
            }
            BandRequest::Spectral3 {
                expr,
                bands_1based,
                rescale,
                colormap,
            } => {
                let b0 = [
                    bands_1based[0].saturating_sub(1),
                    bands_1based[1].saturating_sub(1),
                    bands_1based[2].saturating_sub(1),
                ];
                for &b in &b0 {
                    if b >= ifd.samples_per_pixel {
                        return Err(EngineError::BadRequest(format!(
                            "band {} out of range (dataset has {} bands)",
                            b + 1,
                            ifd.samples_per_pixel
                        )));
                    }
                }
                let style = SingleBandStyle {
                    rescale_min: rescale.0,
                    rescale_max: rescale.1,
                    colormap: *colormap,
                    resample: req.resample,
                    format: req.format,
                };
                match ifd.planar_config {
                    PlanarConfig::Chunky => {
                        let mut indices = Vec::with_capacity(grid_len);
                        for ty in ty0..=ty1 {
                            for tx in tx0..=tx1 {
                                indices.push(Some(ifd.tile_index(tx, ty, 0) as u32));
                            }
                        }
                        let tiles = self.store.get_blocks(cog, ifd_idx, &indices).await?;
                        let r_refs: Vec<_> = tiles
                            .iter()
                            .map(|o| o.as_ref().and_then(|t| t.bands.get(b0[0] as usize)))
                            .collect();
                        let g_refs: Vec<_> = tiles
                            .iter()
                            .map(|o| o.as_ref().and_then(|t| t.bands.get(b0[1] as usize)))
                            .collect();
                        let b_refs: Vec<_> = tiles
                            .iter()
                            .map(|o| o.as_ref().and_then(|t| t.bands.get(b0[2] as usize)))
                            .collect();
                        let so = |b: u16| -> (f32, f32) {
                            if req.unscale {
                                (
                                    cog.index.scales.get(b as usize).copied().unwrap_or(1.0),
                                    cog.index.offsets.get(b as usize).copied().unwrap_or(0.0),
                                )
                            } else {
                                (1.0, 0.0)
                            }
                        };
                        let (s0, o0) = so(b0[0]);
                        let (s1, o1) = so(b0[1]);
                        let (s2, o2) = so(b0[2]);
                        let srcs = [
                            BandSource::new(
                                ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0,
                                grid_w, grid_h, &r_refs,
                            )
                            .with_scale_offset(s0, o0),
                            BandSource::new(
                                ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0,
                                grid_w, grid_h, &g_refs,
                            )
                            .with_scale_offset(s1, o1),
                            BandSource::new(
                                ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0,
                                grid_w, grid_h, &b_refs,
                            )
                            .with_scale_offset(s2, o2),
                        ];
                        if let Some(gpu) = &self.gpu
                            && let Some(bytes) = gpu
                                .try_render_spectral::<N>(
                                    cog.id,
                                    ifd_idx,
                                    bands_1based,
                                    &plan,
                                    &srcs,
                                    *expr,
                                    &style,
                                )
                                .await
                            {
                                self.stats.gpu_dispatches.fetch_add(1, Ordering::Relaxed);
                                return Ok(bytes);
                            }
                        Ok(run_cpu_render(N, style.resample, || {
                    render_bandmath3::<N>(&plan, &srcs, *expr, &style)
                }))
                    }
                    PlanarConfig::Planar => {
                        let mut indices = Vec::with_capacity(grid_len * 3);
                        for &band in &b0 {
                            for ty in ty0..=ty1 {
                                for tx in tx0..=tx1 {
                                    indices.push(Some(ifd.tile_index(tx, ty, band) as u32));
                                }
                            }
                        }
                        let tiles = self.store.get_blocks(cog, ifd_idx, &indices).await?;
                        let r_refs: Vec<_> = tiles[0..grid_len]
                            .iter()
                            .map(|o| o.as_ref().and_then(|t| t.bands.first()))
                            .collect();
                        let g_refs: Vec<_> = tiles[grid_len..2 * grid_len]
                            .iter()
                            .map(|o| o.as_ref().and_then(|t| t.bands.first()))
                            .collect();
                        let b_refs: Vec<_> = tiles[2 * grid_len..3 * grid_len]
                            .iter()
                            .map(|o| o.as_ref().and_then(|t| t.bands.first()))
                            .collect();
                        let so = |b: u16| -> (f32, f32) {
                            if req.unscale {
                                (
                                    cog.index.scales.get(b as usize).copied().unwrap_or(1.0),
                                    cog.index.offsets.get(b as usize).copied().unwrap_or(0.0),
                                )
                            } else {
                                (1.0, 0.0)
                            }
                        };
                        let (s0, o0) = so(b0[0]);
                        let (s1, o1) = so(b0[1]);
                        let (s2, o2) = so(b0[2]);
                        let srcs = [
                            BandSource::new(
                                ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0,
                                grid_w, grid_h, &r_refs,
                            )
                            .with_scale_offset(s0, o0),
                            BandSource::new(
                                ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0,
                                grid_w, grid_h, &g_refs,
                            )
                            .with_scale_offset(s1, o1),
                            BandSource::new(
                                ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0,
                                grid_w, grid_h, &b_refs,
                            )
                            .with_scale_offset(s2, o2),
                        ];
                        if let Some(gpu) = &self.gpu
                            && let Some(bytes) = gpu
                                .try_render_spectral::<N>(
                                    cog.id,
                                    ifd_idx,
                                    bands_1based,
                                    &plan,
                                    &srcs,
                                    *expr,
                                    &style,
                                )
                                .await
                            {
                                self.stats.gpu_dispatches.fetch_add(1, Ordering::Relaxed);
                                return Ok(bytes);
                            }
                        Ok(run_cpu_render(N, style.resample, || {
                    render_bandmath3::<N>(&plan, &srcs, *expr, &style)
                }))
                    }
                }
            }
            BandRequest::CustomExprRgb { .. }
            | BandRequest::CustomExprMulti { .. }
            | BandRequest::MultiBand { .. } => Err(EngineError::BadRequest(
                "multi-band custom expressions are rendered via bbox pipeline".into(),
            )),
            BandRequest::CustomExpr {
                prog,
                rescale,
                colormap,
            } => {
                let style = SingleBandStyle {
                    rescale_min: rescale.0,
                    rescale_max: rescale.1,
                    colormap: *colormap,
                    resample: req.resample,
                    format: req.format,
                };
                let nb = prog.used_bands.len();
                for &b in &prog.used_bands {
                    if b == 0 || b > ifd.samples_per_pixel {
                        return Err(EngineError::BadRequest(format!(
                            "expression band b{b} out of range (dataset has {} bands)",
                            ifd.samples_per_pixel
                        )));
                    }
                }
                if nb == 0 {
                    return Ok(render_expr_vm::<N>(
                        &plan,
                        &[],
                        prog,
                        &style,
                        (ifd.width, ifd.height),
                    ));
                }
                match ifd.planar_config {
                    PlanarConfig::Chunky => {
                        let mut indices = Vec::with_capacity(grid_len);
                        for ty in ty0..=ty1 {
                            for tx in tx0..=tx1 {
                                indices.push(Some(ifd.tile_index(tx, ty, 0) as u32));
                            }
                        }
                        let tiles = self.store.get_blocks(cog, ifd_idx, &indices).await?;
                        let mut band_vecs: Vec<Vec<Option<&cogzilla_codec::DecodedBand>>> =
                            Vec::with_capacity(nb);
                        for &b in &prog.used_bands {
                            let b0 = (b - 1) as usize;
                            band_vecs.push(
                                tiles
                                    .iter()
                                    .map(|o| o.as_ref().and_then(|t| t.bands.get(b0)))
                                    .collect(),
                            );
                        }
                        let srcs: Vec<BandSource<'_>> = band_vecs
                            .iter()
                            .enumerate()
                            .map(|(k, refs)| {
                                let b0 = prog.used_bands[k].saturating_sub(1) as usize;
                                let (sc, off) = if req.unscale {
                                    (
                                        cog.index.scales.get(b0).copied().unwrap_or(1.0),
                                        cog.index.offsets.get(b0).copied().unwrap_or(0.0),
                                    )
                                } else {
                                    (1.0, 0.0)
                                };
                                BandSource::new(
                                    ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0,
                                    ty0, grid_w, grid_h, refs,
                                )
                                .with_scale_offset(sc, off)
                            })
                            .collect();
                        if let Some(gpu) = &self.gpu
                            && let Some(bytes) = gpu
                                .try_render_expr::<N>(cog.id, ifd_idx, &plan, &srcs, prog, &style)
                                .await
                            {
                                self.stats.gpu_dispatches.fetch_add(1, Ordering::Relaxed);
                                return Ok(bytes);
                            }
                        Ok(run_cpu_render(N, style.resample, || {
                    render_expr_vm::<N>(&plan, &srcs, prog, &style, (ifd.width, ifd.height))
                }))
                    }
                    PlanarConfig::Planar => {
                        let mut indices = Vec::with_capacity(grid_len * nb);
                        for &b in &prog.used_bands {
                            let b0 = b - 1;
                            for ty in ty0..=ty1 {
                                for tx in tx0..=tx1 {
                                    indices.push(Some(ifd.tile_index(tx, ty, b0) as u32));
                                }
                            }
                        }
                        let tiles = self.store.get_blocks(cog, ifd_idx, &indices).await?;
                        let mut band_vecs: Vec<Vec<Option<&cogzilla_codec::DecodedBand>>> =
                            Vec::with_capacity(nb);
                        for i in 0..nb {
                            band_vecs.push(
                                tiles[i * grid_len..(i + 1) * grid_len]
                                    .iter()
                                    .map(|o| o.as_ref().and_then(|t| t.bands.first()))
                                    .collect(),
                            );
                        }
                        let srcs: Vec<BandSource<'_>> = band_vecs
                            .iter()
                            .enumerate()
                            .map(|(k, refs)| {
                                let b0 = prog.used_bands[k].saturating_sub(1) as usize;
                                let (sc, off) = if req.unscale {
                                    (
                                        cog.index.scales.get(b0).copied().unwrap_or(1.0),
                                        cog.index.offsets.get(b0).copied().unwrap_or(0.0),
                                    )
                                } else {
                                    (1.0, 0.0)
                                };
                                BandSource::new(
                                    ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0,
                                    ty0, grid_w, grid_h, refs,
                                )
                                .with_scale_offset(sc, off)
                            })
                            .collect();
                        if let Some(gpu) = &self.gpu
                            && let Some(bytes) = gpu
                                .try_render_expr::<N>(cog.id, ifd_idx, &plan, &srcs, prog, &style)
                                .await
                            {
                                self.stats.gpu_dispatches.fetch_add(1, Ordering::Relaxed);
                                return Ok(bytes);
                            }
                        Ok(run_cpu_render(N, style.resample, || {
                    render_expr_vm::<N>(&plan, &srcs, prog, &style, (ifd.width, ifd.height))
                }))
                    }
                }
            }
        }
    }

    async fn try_zero_decode_passthrough<const N: usize>(
        &self,
        cog: &OpenedCog,
        crs: &CrsModel,
        ifd_idx: usize,
        ifd: &IfdMeta,
        req: &TileRequest,
    ) -> Result<Option<Bytes>, EngineError> {
        let Some((tx, ty)) =
            aligned_passthrough_tile_coord::<N>(crs, ifd, cog.index.nodata, req)
        else {
            return Ok(None);
        };

        let Some((offset, byte_len)) = ifd.tile_byte_range(tx, ty, 0) else {
            // When sparse tiles are valid fill data (e.g., no nodata / no mask), fall back to
            // standard rendering rather than returning a transparent empty tile.
            if cog.sparse_fill.is_some() {
                return Ok(None);
            }
            // Sparse empty internal tile -> short-circuit without decoding.
            self.stats
                .empty_short_circuits
                .fetch_add(1, Ordering::Relaxed);
            return Ok(Some(empty_tile_bytes(N as u32, req.format, 3)));
        };

        let tile_idx = ifd.tile_index(tx, ty, 0) as u32;
        let raw = self
            .store
            .get_raw_block(cog, ifd_idx, tile_idx, offset, byte_len)
            .await?;

        let out = if ifd.compression == Compression::Jpeg {
            if let Some(tables) = ifd.jpeg_tables.as_deref() {
                splice_jpeg_tables(tables, &raw)
            } else {
                raw
            }
        } else {
            raw
        };

        self.stats.passthrough_hits.fetch_add(1, Ordering::Relaxed);
        Ok(Some(out))
    }
}

/// Pure eligibility check for zero-decode pass-through of WebMercator-aligned JPEG/WebP COGs.
/// Returns `Some((tx, ty))` when the requested XYZ tile maps 1:1 onto a full, in-bounds
/// internal COG tile with identical codec and identity RGB styling.
fn aligned_passthrough_tile_coord<const N: usize>(
    crs: &CrsModel,
    ifd: &IfdMeta,
    nodata: Option<f32>,
    req: &TileRequest,
) -> Option<(u32, u32)> {
    if !matches!(crs, CrsModel::Epsg3857) || req.unscale {
        return None;
    }
    if ifd.is_stripped
        || ifd.planar_config != PlanarConfig::Chunky
        || ifd.sample_type != SampleType::Uint8
        || ifd.samples_per_pixel != 3
        || ifd.mask_ifd.is_some()
        || !ifd.extra_samples.is_empty()
        || nodata.is_some()
        || !ifd.crs_to_pixel.is_axis_aligned()
        || ifd.tile_width != N as u32
        || ifd.tile_height != N as u32
    {
        return None;
    }

    // Only pass through browser-safe encodings: YCbCr JPEG (photometric=6) or WebP (self-describing).
    // TIFF RGB JPEG (photometric=2) may omit Adobe APP14 transform=0, causing browsers to decode
    // raw RGB as YCbCr (magenta/green corruption).
    let codec_matches = matches!(
        (ifd.compression, ifd.photometric, req.format),
        (Compression::Jpeg, 6, OutputFormat::Jpeg { .. })
            | (Compression::Webp, 2 | 6, OutputFormat::Webp { .. })
    );
    if !codec_matches {
        return None;
    }

    match &req.bands {
        BandRequest::Rgb {
            bands_1based: [1, 2, 3],
            rescale_min,
            rescale_max,
        } if *rescale_min == [0.0, 0.0, 0.0] && *rescale_max == [255.0, 255.0, 255.0] => {}
        _ => return None,
    }

    let (min_x, min_y, max_x, max_y) = tile_bounds_3857(req.tile);
    let (col0, row0) = ifd.crs_to_pixel.forward(min_x, max_y);
    let (col1, row1) = ifd.crs_to_pixel.forward(max_x, min_y);
    let nf = N as f64;
    const TOL: f64 = 1e-2;

    if (col1 - col0 - nf).abs() > TOL || (row1 - row0 - nf).abs() > TOL {
        return None;
    }
    if col0 < -TOL
        || row0 < -TOL
        || col1 > ifd.width as f64 + TOL
        || row1 > ifd.height as f64 + TOL
    {
        return None;
    }

    let tx_f = col0 / nf;
    let ty_f = row0 / nf;
    let tx_round = tx_f.round();
    let ty_round = ty_f.round();
    if (tx_f - tx_round).abs() * nf > TOL || (ty_f - ty_round).abs() * nf > TOL {
        return None;
    }
    let tx = tx_round as i64;
    let ty = ty_round as i64;
    if tx < 0 || ty < 0 || tx >= ifd.tiles_across as i64 || ty >= ifd.tiles_down as i64 {
        return None;
    }
    Some((tx as u32, ty as u32))
}

/// Merge TIFF tag 347 `JPEGTables` with an abbreviated JPEG tile bitstream into a
/// standalone browser-decodable JPEG stream.
fn splice_jpeg_tables(tables: &[u8], raw: &[u8]) -> Bytes {
    if tables.len() >= 4 && raw.len() >= 2 {
        let t_end = if tables.ends_with(&[0xFF, 0xD9]) {
            tables.len() - 2
        } else {
            tables.len()
        };
        let c_start = if raw.starts_with(&[0xFF, 0xD8]) { 2 } else { 0 };
        let mut out = Vec::with_capacity(t_end + (raw.len() - c_start));
        out.extend_from_slice(&tables[..t_end]);
        out.extend_from_slice(&raw[c_start..]);
        Bytes::from(out)
    } else {
        Bytes::copy_from_slice(raw)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cogzilla_raster::Affine2D;

    fn make_aligned_ifd(
        tile: TileCoord,
        compression: Compression,
        tile_size: u32,
        tiles_across: u32,
        tiles_down: u32,
    ) -> IfdMeta {
        let (min_x, min_y, max_x, max_y) = tile_bounds_3857(tile);
        let res_x = (max_x - min_x) / tile_size as f64;
        let res_y = (max_y - min_y) / tile_size as f64;
        // Anchor COG origin 1 tile west and 1 tile north so `tile` sits at internal `(tx=1, ty=1)`.
        let origin_x = min_x - (max_x - min_x);
        let origin_y = max_y + (max_y - min_y);
        let pixel_to_crs = Affine2D {
            a: res_x,
            b: 0.0,
            c: origin_x,
            d: 0.0,
            e: -res_y,
            f: origin_y,
        };
        let crs_to_pixel = pixel_to_crs.invert().unwrap();
        let n_tiles = (tiles_across * tiles_down) as usize;
        IfdMeta {
            width: tile_size * tiles_across,
            height: tile_size * tiles_down,
            tile_width: tile_size,
            tile_height: tile_size,
            tiles_across,
            tiles_down,
            sample_type: SampleType::Uint8,
            samples_per_pixel: 3,
            planar_config: PlanarConfig::Chunky,
            compression,
            predictor: cogzilla_tiff::Predictor::None,
            photometric: 6,
            extra_samples: Vec::new(),
            jpeg_tables: None,
            mask_ifd: None,
            is_stripped: false,
            pixel_to_crs,
            crs_to_pixel,
            gcp_poly: None,
            tile_offsets: vec![0u64; n_tiles].into_boxed_slice(),
            tile_byte_counts: vec![128u32; n_tiles].into_boxed_slice(),
        }
    }

    #[test]
    fn zero_decode_passthrough_matches_aligned_jpeg_and_webp_and_rejects_transforms() {
        let tile = TileCoord { z: 12, x: 655, y: 1583 };
        let ifd_jpeg = make_aligned_ifd(tile, Compression::Jpeg, 256, 4, 4);
        let ifd_webp = make_aligned_ifd(tile, Compression::Webp, 256, 4, 4);

        let rgb_id = BandRequest::Rgb {
            bands_1based: [1, 2, 3],
            rescale_min: [0.0, 0.0, 0.0],
            rescale_max: [255.0, 255.0, 255.0],
        };

        let req_jpeg = TileRequest {
            tile,
            tile_size: 256,
            bands: rgb_id.clone(),
            resample: ResampleMode::Bilinear,
            format: OutputFormat::Jpeg { quality: 85 },
            unscale: false,
        };
        assert_eq!(
            aligned_passthrough_tile_coord::<256>(&CrsModel::Epsg3857, &ifd_jpeg, None, &req_jpeg),
            Some((1, 1))
        );

        let req_webp = TileRequest {
            tile,
            tile_size: 256,
            bands: rgb_id.clone(),
            resample: ResampleMode::Bilinear,
            format: OutputFormat::Webp { quality: Some(85.0) },
            unscale: false,
        };
        assert_eq!(
            aligned_passthrough_tile_coord::<256>(&CrsModel::Epsg3857, &ifd_webp, None, &req_webp),
            Some((1, 1))
        );

        // Reject mismatched format (PNG requested on JPEG COG)
        let req_png = TileRequest {
            format: OutputFormat::Png,
            ..req_jpeg.clone()
        };
        assert_eq!(
            aligned_passthrough_tile_coord::<256>(&CrsModel::Epsg3857, &ifd_jpeg, None, &req_png),
            None
        );

        // Reject rescaled request
        let req_rescaled = TileRequest {
            bands: BandRequest::Rgb {
                bands_1based: [1, 2, 3],
                rescale_min: [10.0, 10.0, 10.0],
                rescale_max: [200.0, 200.0, 200.0],
            },
            ..req_jpeg.clone()
        };
        assert_eq!(
            aligned_passthrough_tile_coord::<256>(&CrsModel::Epsg3857, &ifd_jpeg, None, &req_rescaled),
            None
        );

        // Reject band-swapped request
        let req_swap = TileRequest {
            bands: BandRequest::Rgb {
                bands_1based: [3, 2, 1],
                rescale_min: [0.0, 0.0, 0.0],
                rescale_max: [255.0, 255.0, 255.0],
            },
            ..req_jpeg.clone()
        };
        assert_eq!(
            aligned_passthrough_tile_coord::<256>(&CrsModel::Epsg3857, &ifd_jpeg, None, &req_swap),
            None
        );

        // Reject non-aligned (half-tile shifted) IFD
        let mut ifd_shifted = ifd_jpeg.clone();
        ifd_shifted.pixel_to_crs.c += ifd_shifted.pixel_to_crs.a * 128.0;
        ifd_shifted.crs_to_pixel = ifd_shifted.pixel_to_crs.invert().unwrap();
        assert_eq!(
            aligned_passthrough_tile_coord::<256>(&CrsModel::Epsg3857, &ifd_shifted, None, &req_jpeg),
            None
        );

        // Verify JPEGTables splicing strips trailing EOI (FF D9) from tables and leading SOI (FF D8) from tile
        let spliced = splice_jpeg_tables(
            &[0xFF, 0xD8, 0xFF, 0xDB, 0x01, 0x02, 0xFF, 0xD9],
            &[0xFF, 0xD8, 0xFF, 0xC0, 0xAA, 0xBB, 0xFF, 0xD9],
        );
        assert_eq!(
            spliced.as_ref(),
            &[0xFF, 0xD8, 0xFF, 0xDB, 0x01, 0x02, 0xFF, 0xC0, 0xAA, 0xBB, 0xFF, 0xD9]
        );
    }

    #[tokio::test]
    async fn zero_decode_passthrough_streams_exact_bytes_without_decoding() {
        let tile = TileCoord { z: 12, x: 655, y: 1583 };
        let mut ifd = make_aligned_ifd(tile, Compression::Jpeg, 256, 2, 2);
        // Write a real payload to a temp file and point tile (1, 1) (index 3) to it.
        let payload = b"\xFF\xD8\xFF\xE0\x00\x10JFIF_ZERO_DECODE_TEST_PAYLOAD\xFF\xD9";
        let tmp_path = std::env::temp_dir().join(format!(
            "cogzilla_passthrough_test_{}.bin",
            std::process::id()
        ));
        std::fs::write(&tmp_path, payload).unwrap();
        ifd.tile_offsets = vec![0, 0, 0, 0].into_boxed_slice();
        ifd.tile_byte_counts = vec![0, 0, 0, payload.len() as u32].into_boxed_slice();

        let geo_keys = cogzilla_tiff::GeoKeys {
            model_type: Some(1),
            projected_cs_type: Some(3857),
            ..cogzilla_tiff::GeoKeys::default()
        };
        let index = cogzilla_tiff::CogIndex {
            byte_order: cogzilla_tiff::ByteOrder::Little,
            is_bigtiff: false,
            is_georeferenced: true,
            has_rpc: false,
            main_ifd_offset: 8,
            ifd_offsets: vec![8],
            gcp_residual_px: None,
            geo_keys,
            nodata: None,
            nodata_f64: None,
            scales: vec![1.0; 3],
            offsets: vec![0.0; 3],
            scales_f64: vec![1.0; 3],
            offsets_f64: vec![0.0; 3],
            band_metadata: vec![Vec::new(); 3],
            band_descriptions: vec![String::new(); 3],
            color_map: None,
            ifds: vec![ifd].into(),
        };
        let source = cogzilla_io::CogSource::open_local(&tmp_path).unwrap();
        let cog = OpenedCog {
            id: 999_999,
            source_id: 999_999,
            sparse_fill: None,
            uri: tmp_path.to_string_lossy().into_owned(),
            index,
            source,
            freshness: cogzilla_io::store::SourceFreshness::Local {
                path: tmp_path.clone(),
                len: 0,
                mtime: None,
                inode: 0,
            },
        };

        let engine = TileEngine::new(16 * 1024 * 1024);
        let req = TileRequest {
            tile,
            tile_size: 256,
            bands: BandRequest::Rgb {
                bands_1based: [1, 2, 3],
                rescale_min: [0.0, 0.0, 0.0],
                rescale_max: [255.0, 255.0, 255.0],
            },
            resample: ResampleMode::Bilinear,
            format: OutputFormat::Jpeg { quality: 85 },
            unscale: false,
        };
        let out = engine.render_tile(&cog, &req).await.unwrap();
        let _ = std::fs::remove_file(&tmp_path);

        assert_eq!(out.as_ref(), payload);
        assert_eq!(engine.stats.passthrough_hits.load(Ordering::Relaxed), 1);
        assert_eq!(engine.store.cache.hits.load(Ordering::Relaxed), 0);
        assert_eq!(engine.store.cache.misses.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn catalog_mosaic_empty_npy_includes_mask_plane() {
        struct EmptyCatalog;
        impl SceneCatalog for EmptyCatalog {
            fn query_bbox_3857(&self, _min_x: f64, _min_y: f64, _max_x: f64, _max_y: f64) -> Vec<String> {
                Vec::new()
            }
            fn len(&self) -> usize {
                0
            }
        }
        let engine = TileEngine::new(1024 * 1024);
        let req = TileRequest {
            tile: TileCoord { z: 2, x: 1, y: 1 },
            tile_size: 64,
            bands: BandRequest::Single {
                band_1based: 1,
                rescale: (0.0, 1.0),
                colormap: ColormapName::Grayscale,
            },
            resample: ResampleMode::Bilinear,
            format: OutputFormat::Npy,
            unscale: false,
        };
        let out = engine
            .render_catalog_mosaic_tile(&EmptyCatalog, &req)
            .await
            .unwrap();
        // Parse NPY header to confirm shape is (2, 64, 64) (1 data band + 1 mask plane).
        assert_eq!(&out[..6], b"\x93NUMPY");
        let hdr_len = u16::from_le_bytes([out[8], out[9]]) as usize;
        let hdr = std::str::from_utf8(&out[10..10 + hdr_len]).unwrap();
        assert!(
            hdr.contains("(2, 64, 64)"),
            "expected empty catalog mosaic .npy to include mask plane (2, 64, 64), got: {hdr}"
        );
    }

    #[test]
    fn tile_fetch_range_for_plan_pads_large_footprint() {
        // Construct a Separable1D plan spanning u in [260..=4340] across 256 samples (sx = 16.0).
        // Tile width = 256: u=260 is 4 px inside tile 1, so half-footprint = 8 px reaches into tile 0!
        let mut u = [0.0f32; 256];
        let mut v = [0.0f32; 256];
        for i in 0..256 {
            u[i] = 260.0 + (i as f32) * 16.0;
            v[i] = 260.0 + (i as f32) * 16.0;
        }
        let plan = CoordPlan::Separable1D {
            u: Box::new(u),
            v: Box::new(v),
            src_min_col: 260,
            src_max_col: 4341,
            src_min_row: 260,
            src_max_row: 4341,
            all_in_bounds: true,
        };
        let ifd = make_aligned_ifd(TileCoord { z: 8, x: 0, y: 0 }, Compression::None, 256, 32, 32);
        let (tx0, _tx1, ty0, _ty1) =
            tile_fetch_range_for_plan(&plan, ResampleMode::Average, false, &ifd);
        assert_eq!(tx0, 0, "footprint sx=16 at u=260 must fetch left neighbor tile 0");
        assert_eq!(ty0, 0, "footprint sy=16 at v=260 must fetch top neighbor tile 0");
    }

    #[test]
    fn value_bands_covers_custom_expr_and_terrain() {
        let p = cogzilla_kernel::ExprProgram::compile("b1").unwrap();
        assert_eq!(
            TileEngine::value_bands(&BandRequest::CustomExprRgb {
                progs: [p.clone(), p.clone(), p.clone()],
                rescale_min: [0.0; 3],
                rescale_max: [1.0; 3],
            }),
            3
        );
        assert_eq!(
            TileEngine::value_bands(&BandRequest::CustomExprMulti {
                progs: vec![p.clone(), p.clone(), p.clone(), p.clone()],
            }),
            4
        );
        assert_eq!(
            TileEngine::value_bands(&BandRequest::Terrain {
                band_1based: 1,
                algorithm: cogzilla_kernel::DemAlgorithm::Terrarium,
                math: None,
            }),
            3
        );
        assert_eq!(
            TileEngine::value_bands(&BandRequest::MultiBand {
                bands_1based: vec![1, 2],
            }),
            2
        );
        assert_eq!(
            TileEngine::value_bands(&BandRequest::Single {
                band_1based: 1,
                rescale: (0.0, 255.0),
                colormap: ColormapName::Grayscale,
            }),
            1
        );
    }

    #[test]
    fn format_float_tile_handles_raw_f64_and_npy_f64() {
        let raw_f32: Vec<u8> = [1.5f32, -2.0, 0.25, f32::NAN]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let raw64 = format_float_tile(Bytes::from(raw_f32.clone()), 1, 2, OutputFormat::RawF64);
        assert_eq!(raw64.len(), 4 * 8);
        let vals: Vec<f64> = raw64
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(vals[0], 1.5);
        assert_eq!(vals[1], -2.0);
        assert_eq!(vals[2], 0.25);
        assert!(vals[3].is_nan());

        let npy64 = format_float_tile(Bytes::from(raw_f32), 1, 2, OutputFormat::NpyF64);
        assert_eq!(&npy64[..6], b"\x93NUMPY");
        let hdr_len = u16::from_le_bytes([npy64[8], npy64[9]]) as usize;
        let hdr = std::str::from_utf8(&npy64[10..10 + hdr_len]).unwrap();
        assert!(hdr.contains("<f8"), "expected float64 dtype, got {hdr}");
        assert!(hdr.contains("(2, 2, 2)"), "expected (band+mask, 2, 2), got {hdr}");
    }

    #[test]
    fn tolerate_asset_failure_surfaces_fatal_and_absorbs_non_fatal() {
        let forbidden = EngineError::Io(IoError::Forbidden("https://blocked.example/a.tif".into()));
        assert!(forbidden.is_fatal());
        assert!(tolerate_asset_failure::<u32>(Err(forbidden)).is_err());

        let shared_forbidden = EngineError::Shared(Arc::new(EngineError::Io(IoError::Shared(
            Arc::new(IoError::Forbidden("https://blocked.example/b.tif".into())),
        ))));
        assert!(shared_forbidden.is_fatal());
        assert!(tolerate_asset_failure::<u32>(Err(shared_forbidden)).is_err());

        let transient = EngineError::Io(IoError::HttpStatus {
            status: 503,
            url: "https://ok.example/a.tif".into(),
        });
        assert!(!transient.is_fatal());
        assert_eq!(tolerate_asset_failure::<u32>(Err(transient)).unwrap(), None);
        assert_eq!(tolerate_asset_failure(Ok(42u32)).unwrap(), Some(42));
    }
}
