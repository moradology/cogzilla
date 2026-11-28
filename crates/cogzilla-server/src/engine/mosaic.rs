//! Multi-scene mosaic & catalog tile rendering with early-exit pixel coverage.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use bytes::Bytes;
use cogzilla_io::{IoError, OpenedCog};
use cogzilla_kernel::{ColormapName, OutputFormat, ResampleMode, empty_tile_bytes, encode_rgba_tile};
use cogzilla_proj::{CrsModel, tile_bounds_3857};
use futures_util::stream::{FuturesOrdered, StreamExt};

use crate::catalog::SceneCatalog;

use super::{
    BandRequest, EngineError, MOSAIC_SCENES_AHEAD, TileEngine, TileRequest, asset_fetch_error,
    draws_one_grey_band, format_float_tile, validate_multi_cog_grid,
};

/// A mosaic's scenes, rendered in order with some read ahead of the one being waited for.
/// Dropping it drops the scenes still being read.
struct ScenesInOrder<'a> {
    engine: &'a TileEngine,
    scenes: &'a [Vec<String>],
    base: &'a TileRequest,
    unscale: bool,
    next: usize,
    window: usize,
    reading: FuturesOrdered<SceneRead<'a>>,
}

/// One scene of a mosaic on its way.
type SceneRead<'a> = std::pin::Pin<Box<dyn std::future::Future<Output = Result<Bytes, SceneFailure>> + Send + 'a>>;

/// Why a scene of a mosaic gave no pixels: it could not be opened, or it was open and
/// could not be read. A mosaic of a list of URLs passes over the first and goes on to
/// the next scene; one whose scenes were chosen for the tile fails on either.
enum SceneFailure {
    Unopened(EngineError),
    Unread(EngineError),
}

/// Whether opening a scene failed because of a fatal error (e.g. allowlist refusal).
fn refused(e: &EngineError) -> bool {
    e.is_fatal()
}

impl SceneFailure {
    fn into_error(self) -> EngineError {
        match self {
            Self::Unopened(e) | Self::Unread(e) => e,
        }
    }
}

impl<'a> ScenesInOrder<'a> {
    /// The next scene in order, or `None` after the last.
    async fn next(&mut self) -> Option<Result<Bytes, SceneFailure>> {
        while self.reading.len() < self.window && self.next < self.scenes.len() {
            let scene = &self.scenes[self.next];
            self.reading.push_back(Box::pin(self.engine.render_mosaic_scene(scene, self.base, self.unscale)));
            self.next += 1;
        }
        let scene = self.reading.next().await;
        // Being asked again means the tile is not full yet: read twice as many at once.
        self.window = (self.window * 2).min(MOSAIC_SCENES_AHEAD);
        scene
    }
}

impl TileEngine {
    /// Render a dynamic front-to-back mosaic across `urls` with **early-exit pixel coverage**:
    /// walks scenes in priority order, composites only into unfilled (`alpha == 0`) pixels, and
    /// stops once all `tile_size * tile_size` pixels are opaque. The first scene is read
    /// alone, so a tile it fills opens nothing else; after it, scenes are read a few at a
    /// time (see [`Self::render_mosaic_scenes_ahead`]), so at most that few are opened beyond
    /// the one that fills the tile.
    pub async fn render_mosaic_tile(
        &self,
        urls: &[String],
        req: &TileRequest,
    ) -> Result<Bytes, EngineError> {
        Ok(self.render_mosaic_tile_with_report(urls, req).await?.0)
    }

    /// Render a dynamic front-to-back mosaic across `urls`, returning `(tile_bytes, failed_assets)`
    /// where `failed_assets` lists any redacted asset URLs that could not be fetched and were
    /// skipped before rendering completed.
    pub async fn render_mosaic_tile_with_report(
        &self,
        urls: &[String],
        req: &TileRequest,
    ) -> Result<(Bytes, Vec<String>), EngineError> {
        if urls.is_empty() {
            return Err(EngineError::BadRequest(
                "mosaic requires at least one ?url= parameter".into(),
            ));
        }
        // Bound per-tile scene fan-out so a huge MosaicJSON/catalog of transparent or OOB COGs
        // cannot trigger thousands of sequential open_uri + render_tile calls on a single GET.
        const MAX_MOSAIC_SCENES_PER_TILE: usize = 64;
        let scenes: Vec<Vec<String>> = urls
            .iter()
            .take(MAX_MOSAIC_SCENES_PER_TILE)
            .map(|u| vec![u.clone()])
            .collect();
        self.render_mosaic(&scenes, req, 1, true).await
    }

    /// Open every asset of one mosaic scene, concurrently when there is more than one.
    async fn open_scene(&self, scene: &[String]) -> Result<Vec<Arc<OpenedCog>>, EngineError> {
        if let [only] = scene {
            return Ok(vec![self.store.open_uri(only).await.map_err(|e| asset_fetch_error(only, e))?]);
        }
        let mut js = tokio::task::JoinSet::new();
        for (idx, uri) in scene.iter().enumerate() {
            let store = self.store.clone();
            let uri = uri.clone();
            js.spawn(async move { Ok::<_, IoError>((idx, store.open_uri(&uri).await?)) });
        }
        let mut slots: Vec<Option<Arc<OpenedCog>>> = vec![None; scene.len()];
        while let Some(res) = js.join_next().await {
            let (idx, cog) = res
                .map_err(|e| EngineError::BadRequest(format!("open task failed: {e}")))??;
            slots[idx] = Some(cog);
        }
        Ok(slots.into_iter().flatten().collect())
    }

    /// One scene of a mosaic, as the tile `base` asks for.
    async fn render_mosaic_scene(
        &self,
        scene: &[String],
        base: &TileRequest,
        unscale: bool,
    ) -> Result<Bytes, SceneFailure> {
        let cogs = self.open_scene(scene).await.map_err(SceneFailure::Unopened)?;
        let cog = &cogs[0];
        // Fold each scene's own GDAL scale/offset into Single/Rgb rescale bounds when
        // `?unscale=true` so heterogeneous mosaic scenes unscale accurately at zero per-pixel cost.
        // Band indices address one file, so this applies to single-asset scenes only.
        let scene_req = if unscale
            && cogs.len() == 1
            && matches!(
                base.bands,
                BandRequest::Single { .. } | BandRequest::Rgb { .. }
            ) {
            let mut sr = base.clone();
            let adj = |b1: u16, mn: f32, mx: f32| -> (f32, f32) {
                let i = b1.saturating_sub(1) as usize;
                let sc = cog.index.scales.get(i).copied().unwrap_or(1.0);
                let off = cog.index.offsets.get(i).copied().unwrap_or(0.0);
                if ((sc - 1.0).abs() < 1e-9 && off.abs() < 1e-9) || sc.abs() < 1e-12 {
                    (mn, mx)
                } else {
                    ((mn - off) / sc, (mx - off) / sc)
                }
            };
            match &mut sr.bands {
                BandRequest::Single {
                    band_1based,
                    rescale,
                    ..
                } => {
                    *rescale = adj(*band_1based, rescale.0, rescale.1);
                }
                BandRequest::Rgb {
                    bands_1based,
                    rescale_min,
                    rescale_max,
                } => {
                    for k in 0..3 {
                        let (mn, mx) = adj(bands_1based[k], rescale_min[k], rescale_max[k]);
                        rescale_min[k] = mn;
                        rescale_max[k] = mx;
                    }
                }
                _ => {}
            }
            sr
        } else {
            base.clone()
        };
        self.render_multi_cog_tile(&cogs, &scene_req).await.map_err(SceneFailure::Unread)
    }

    /// The scenes of a mosaic rendered in order, up to a growing number of them at once.
    fn mosaic_scenes_in_order<'a>(
        &'a self,
        scenes: &'a [Vec<String>],
        base: &'a TileRequest,
        unscale: bool,
        ahead: usize,
    ) -> ScenesInOrder<'a> {
        ScenesInOrder {
            engine: self,
            scenes,
            base,
            unscale,
            next: 0,
            window: ahead.clamp(1, MOSAIC_SCENES_AHEAD),
            reading: FuturesOrdered::new(),
        }
    }

    /// Whether rasters lie on one grid, pixel for pixel at every overview and in the same
    /// place, so that they can be read as the bands of one image.
    pub fn on_one_grid(cogs: &[Arc<OpenedCog>]) -> bool {
        let Some(first) = cogs.first() else {
            return true;
        };
        let placed = |c: &OpenedCog| {
            let t = &c.index.full_res().crs_to_pixel;
            [t.a, t.b, t.c, t.d, t.e, t.f]
        };
        let origin = placed(first);
        cogs.iter().all(|c| {
            c.index.ifds.len() == first.index.ifds.len()
                && placed(c).iter().zip(&origin).all(|(a, b)| (a - b).abs() <= 1e-9 * b.abs().max(1.0))
        }) && (0..first.index.ifds.len()).all(|ifd| validate_multi_cog_grid(cogs, ifd).is_ok())
    }

    /// The bands to read of a raster that has `count`, `extra` of them extra samples: those
    /// asked for, counted from one, or all of its own when none is. Refuses a band the raster has not, and more than
    /// [`Self::MAX_PLANES_PER_RASTER`].
    pub fn wanted_bands(count: u16, extra: usize, bands_1based: &[u16]) -> Result<Vec<u16>, EngineError> {
        // With none asked for: every band that is not an extra sample (alpha), as rio-tiler
        // leaves those out. They come last.
        let own = count.saturating_sub(extra.min(usize::from(u16::MAX)) as u16).max(1).min(count);
        let wanted: Vec<u16> =
            if bands_1based.is_empty() { (1..=own).collect() } else { bands_1based.to_vec() };
        if let Some(bad) = wanted.iter().find(|b| **b == 0 || **b > count) {
            let known: Vec<String> = (1..=count).map(|b| b.to_string()).collect();
            let tail = if count == 1 { "," } else { "" };
            return Err(EngineError::BadRequest(format!(
                "band index {bad} out of range (not in ({}{tail}))",
                known.join(", ")
            )));
        }
        if wanted.len() > Self::MAX_PLANES_PER_RASTER {
            return Err(EngineError::BadRequest(format!(
                "{} bands of one asset are too many to read at once (at most {}); choose bands with `bidx`",
                wanted.len(),
                Self::MAX_PLANES_PER_RASTER
            )));
        }
        Ok(wanted)
    }

    /// The most bands of one raster [`Self::raster_planes`] reads for one tile.
    pub const MAX_PLANES_PER_RASTER: usize = 32;

    /// Bands of one raster over a tile, each a plane of `f32` on the tile's grid whatever
    /// the raster's own, NaN where it has no data. An empty `bands_1based` asks for every
    /// band. `req` gives the tile, its size and `unscale`; `resampling` is used for a raster
    /// already in the tile's projection and `reproject` for one that has to be warped into
    /// it. `nodata` (a request's `?nodata=`) stands in place of the raster's own. The raster's
    /// sample type comes back with the planes.
    #[allow(clippy::too_many_arguments)]
    pub async fn raster_planes(
        &self,
        uri: &str,
        bands_1based: &[u16],
        req: &TileRequest,
        resampling: ResampleMode,
        reproject: ResampleMode,
        nodata: Option<f32>,
    ) -> Result<(Vec<Vec<f32>>, cogzilla_raster::SampleType), EngineError> {
        let cog = self.store.with_nodata(&self.store.open_uri(uri).await?, nodata);
        let full = cog.index.full_res();
        let sample_type = full.sample_type;
        let wanted = Self::wanted_bands(full.samples_per_pixel, full.extra_samples.len(), bands_1based)?;
        let mut one = req.clone();
        one.format = OutputFormat::RawF32;
        one.resample = match self.get_or_build_crs(&cog)? {
            CrsModel::Epsg3857 => resampling,
            _ => reproject,
        };
        let pixels = (req.tile_size as usize).pow(2);
        let mut planes = Vec::with_capacity(wanted.len());
        // One band at a time: read three together and the renderer gives none of them
        // where it finds one of them empty.
        for band in &wanted {
            one.bands = BandRequest::Single {
                band_1based: *band,
                rescale: (0.0, 255.0),
                colormap: ColormapName::Grayscale,
            };
            let values = self.render_tile(&cog, &one).await?;
            if values.len() != pixels * 4 {
                return Err(EngineError::BadRequest("unexpected plane size from the renderer".into()));
            }
            planes.push(values.chunks_exact(4).map(|v| f32::from_le_bytes([v[0], v[1], v[2], v[3]])).collect());
        }
        if planes.len() != wanted.len() {
            return Err(EngineError::BadRequest("unexpected number of planes from the renderer".into()));
        }
        Ok((planes, sample_type))
    }

    /// How many bands of values a request's scenes hold: three for a colour composite or terrain
    /// encoding, N for multi-band / multi-expression requests, else one.
    pub fn value_bands(bands: &BandRequest) -> usize {
        match bands {
            BandRequest::Rgb { .. }
            | BandRequest::SarRgb { .. }
            | BandRequest::Terrain { .. }
            | BandRequest::CustomExprRgb { .. } => 3,
            BandRequest::MultiBand { bands_1based } => bands_1based.len().max(1),
            BandRequest::CustomExprMulti { progs } => progs.len().max(1),
            _ => 1,
        }
    }

    /// Every scene of a mosaic as the values its bands hold over the tile, before any
    /// rescaling or colour, handed to `each` in the mosaic's order: band after band
    /// ([`Self::value_bands`] of them) of little-endian `f32`, NaN where the scene has no
    /// data. A scene that does not reach the tile is not handed over. This is what a mosaic
    /// that has to look at every scene is made from; one that takes the first value it finds
    /// should use [`Self::render_mosaic_scenes_ahead`], which stops when the tile is full.
    pub async fn mosaic_scene_values(
        &self,
        scenes: &[Vec<String>],
        req: &TileRequest,
        mut each: impl FnMut(&[u8]),
    ) -> Result<(), EngineError> {
        if scenes.is_empty() || scenes.iter().any(|s| s.is_empty()) {
            return Err(EngineError::BadRequest(
                "mosaic requires at least one asset per scene".into(),
            ));
        }
        let len = Self::value_bands(&req.bands) * (req.tile_size as usize).pow(2) * 4;
        let mut scene_req = req.clone();
        scene_req.format = OutputFormat::RawF32;
        let mut reading = self.mosaic_scenes_in_order(scenes, &scene_req, false, MOSAIC_SCENES_AHEAD);
        // A scene that misses the tile comes back as nothing but no data. It is not a scene
        // of this tile: counted, it would make a tile of zeros out of nothing.
        let nothing = |scene: &[u8]| {
            scene.chunks_exact(4).all(|v| f32::from_le_bytes([v[0], v[1], v[2], v[3]]).is_nan())
        };
        while let Some(scene) = reading.next().await {
            let scene = scene.map_err(SceneFailure::into_error)?;
            if scene.len() == len && !nothing(&scene) {
                each(&scene);
            }
        }
        Ok(())
    }

    /// The mosaic behind [`Self::render_mosaic_tile`], over scenes that are each one asset or
    /// two to three co-registered single-band assets (rendered as one multi-asset composite).
    /// The caller bounds how many scenes it passes.
    pub async fn render_mosaic_scenes(
        &self,
        scenes: &[Vec<String>],
        req: &TileRequest,
    ) -> Result<Bytes, EngineError> {
        Ok(self.render_mosaic(scenes, req, 1, true).await?.0)
    }

    /// [`Self::render_mosaic_scenes`], reading up to `ahead` scenes at once from the start.
    ///
    /// Scenes are always composited in order, and a tile's answer does not depend on
    /// `ahead`. What it changes is the wait: a scene is mostly time spent on its storage,
    /// and scenes read one after another add those waits up. With `ahead` of one, the first
    /// scene is read alone, so a tile it fills opens nothing else, and each scene that
    /// does not fill the tile doubles how many are read at once, up to
    /// [`MOSAIC_SCENES_AHEAD`]. A caller whose list is already only the scenes that add to
    /// the tile (pgSTAC's, chosen by footprint) passes the limit and reads them together.
    pub async fn render_mosaic_scenes_ahead(
        &self,
        scenes: &[Vec<String>],
        req: &TileRequest,
        ahead: usize,
    ) -> Result<Bytes, EngineError> {
        Ok(self.render_mosaic(scenes, req, ahead, false).await?.0)
    }

    /// The mosaic itself. With `skip_unopened`, a scene that cannot be opened is passed
    /// over, and its error is the answer only when no scene could be opened at all: a
    /// list of URLs may hold a dead one. Without, it fails the tile, as any scene that
    /// cannot be read does.
    async fn render_mosaic(
        &self,
        scenes: &[Vec<String>],
        req: &TileRequest,
        ahead: usize,
        skip_unopened: bool,
    ) -> Result<(Bytes, Vec<String>), EngineError> {
        if scenes.is_empty() || scenes.iter().any(|s| s.is_empty()) {
            return Err(EngineError::BadRequest(
                "mosaic requires at least one asset per scene".into(),
            ));
        }
        let n = req.tile_size as usize;
        let total_pixels = n * n;
        let f32_bands: usize = match req.bands {
            BandRequest::Rgb { .. }
            | BandRequest::SarRgb { .. }
            | BandRequest::Terrain { .. }
            | BandRequest::CustomExprRgb { .. } => 3,
            BandRequest::MultiBand { ref bands_1based } => bands_1based.len().max(1),
            BandRequest::CustomExprMulti { ref progs } => progs.len().max(1),
            _ => 1,
        };

        // Helper to fold a scene's GDAL scale/offset into Single/Rgb rescale bounds when `?unscale=true`.
        let fold_scene_unscale = |cog: &OpenedCog, base: &TileRequest| -> TileRequest {
            if base.unscale
                && matches!(
                    base.bands,
                    BandRequest::Single { .. } | BandRequest::Rgb { .. }
                )
            {
                let mut sr = base.clone();
                let adj = |b1: u16, mn: f32, mx: f32| -> (f32, f32) {
                    let i = b1.saturating_sub(1) as usize;
                    let sc = cog.index.scales.get(i).copied().unwrap_or(1.0);
                    let off = cog.index.offsets.get(i).copied().unwrap_or(0.0);
                    if ((sc - 1.0).abs() < 1e-9 && off.abs() < 1e-9) || sc.abs() < 1e-12 {
                        (mn, mx)
                    } else {
                        ((mn - off) / sc, (mx - off) / sc)
                    }
                };
                match &mut sr.bands {
                    BandRequest::Single {
                        band_1based,
                        rescale,
                        ..
                    } => {
                        *rescale = adj(*band_1based, rescale.0, rescale.1);
                    }
                    BandRequest::Rgb {
                        bands_1based,
                        rescale_min,
                        rescale_max,
                    } => {
                        for k in 0..3 {
                            let (mn, mx) = adj(bands_1based[k], rescale_min[k], rescale_max[k]);
                            rescale_min[k] = mn;
                            rescale_max[k] = mx;
                        }
                    }
                    _ => {}
                }
                sr
            } else {
                base.clone()
            }
        };

        // 1-scene fast path: skip canvas allocation, pixel compositing, and re-encoding so single-scene
        // mosaic tiles keep zero-decode JPEG/WebP passthrough and 8-bit indexed PLTE PNG encoding!
        if scenes.len() == 1 && scenes[0].len() == 1 {
            let cog = self
                .store
                .open_uri(&scenes[0][0])
                .await
                .map_err(|e| asset_fetch_error(&scenes[0][0], e))?;
            let scene_req = fold_scene_unscale(&cog, req);
            let bytes = self.render_tile(&cog, &scene_req).await?;
            return Ok((bytes, Vec::new()));
        }

        // Exact f64 mosaic path (`RawF64` / `NpyF64`): composite real `f64` band-major planes
        // (CHW) directly without `f32` truncation, preserving full 53-bit precision (#619).
        if matches!(req.format, OutputFormat::RawF64 | OutputFormat::NpyF64) {
            let mut canvas = vec![f64::NAN; f32_bands * total_pixels];
            let mut filled = 0usize;
            let mut scene_req = req.clone();
            scene_req.format = OutputFormat::RawF64;

            let mut opened_any = false;
            let mut last_open_err: Option<EngineError> = None;
            let mut failed_assets: Vec<String> = Vec::new();
            let mut scene_i = 0usize;
            let mut reading = self.mosaic_scenes_in_order(scenes, &scene_req, false, ahead);
            while filled < total_pixels {
                let scene_urls = &scenes[scene_i.min(scenes.len() - 1)];
                scene_i += 1;
                let Some(src) = reading.next().await else {
                    break;
                };
                let src = match src {
                    Ok(src) => {
                        opened_any = true;
                        src
                    }
                    Err(SceneFailure::Unopened(e)) if skip_unopened && !refused(&e) => {
                        failed_assets.extend(scene_urls.iter().map(|u| cogzilla_io::redact_http_url(u)));
                        last_open_err = Some(e);
                        continue;
                    }
                    Err(SceneFailure::Unread(EngineError::Io(e))) if skip_unopened && !e.is_fatal() => {
                        failed_assets.extend(scene_urls.iter().map(|u| cogzilla_io::redact_http_url(u)));
                        last_open_err = Some(asset_fetch_error(&scene_urls[0], e));
                        continue;
                    }
                    Err(failure) => return Err(failure.into_error()),
                };
                if src.len() != f32_bands * total_pixels * 8 {
                    continue;
                }
                for i in 0..total_pixels {
                    if !canvas[i].is_nan() {
                        continue;
                    }
                    let mut any_valid = false;
                    for b in 0..f32_bands {
                        let byte_off = (b * total_pixels + i) * 8;
                        let v = f64::from_le_bytes(src[byte_off..byte_off + 8].try_into().unwrap());
                        if !v.is_nan() {
                            any_valid = true;
                            break;
                        }
                    }
                    if any_valid {
                        for b in 0..f32_bands {
                            let byte_off = (b * total_pixels + i) * 8;
                            canvas[b * total_pixels + i] =
                                f64::from_le_bytes(src[byte_off..byte_off + 8].try_into().unwrap());
                        }
                        filled += 1;
                    }
                }
            }

            if filled == 0
                && !opened_any
                && let Some(err) = last_open_err
            {
                return Err(err);
            }
            if req.format == OutputFormat::NpyF64 {
                let ts = req.tile_size as usize;
                return Ok((
                    cogzilla_kernel::encode_npy_typed_f64(
                        &canvas,
                        f32_bands,
                        ts,
                        ts,
                        cogzilla_raster::SampleType::Float64,
                        None,
                        true,
                    ),
                    failed_assets,
                ));
            }
            let mut out = bytes::BytesMut::with_capacity(f32_bands * total_pixels * 8);
            for v in canvas {
                out.extend_from_slice(&v.to_le_bytes());
            }
            return Ok((out.freeze(), failed_assets));
        }

        // RawF32 / Npy mosaic path: composite real float band-major planes (CHW)
        // with NaN transparency and early exit once every pixel has valid data.
        if matches!(req.format, OutputFormat::RawF32 | OutputFormat::Npy) {
            let mut canvas = vec![f32::NAN; f32_bands * total_pixels];
            let mut filled = 0usize;
            let mut scene_req = req.clone();
            scene_req.format = OutputFormat::RawF32;
            let empty_f32_sentinel = empty_tile_bytes(req.tile_size, OutputFormat::RawF32, f32_bands);

            let mut opened_any = false;
            let mut last_open_err: Option<EngineError> = None;
            let mut failed_assets: Vec<String> = Vec::new();
            let mut scene_i = 0usize;
            let mut reading = self.mosaic_scenes_in_order(scenes, &scene_req, false, ahead);
            while filled < total_pixels {
                let scene_urls = &scenes[scene_i.min(scenes.len() - 1)];
                scene_i += 1;
                let Some(src) = reading.next().await else {
                    break;
                };
                let src = match src {
                    Ok(src) => {
                        opened_any = true;
                        src
                    }
                    // An asset the allowlist refuses fails the tile, as on `main`, rather than being
                    // passed over as one that could not be read.
                    Err(SceneFailure::Unopened(e)) if skip_unopened && !refused(&e) => {
                        failed_assets.extend(scene_urls.iter().map(|u| cogzilla_io::redact_http_url(u)));
                        last_open_err = Some(e);
                        continue;
                    }
                    // As on `main`: a scene that opened but could not be read is passed over
                    // too, unless the allowlist refused what it reads.
                    Err(SceneFailure::Unread(EngineError::Io(e))) if skip_unopened && !e.is_fatal() => {
                        failed_assets.extend(scene_urls.iter().map(|u| cogzilla_io::redact_http_url(u)));
                        last_open_err = Some(asset_fetch_error(&scene_urls[0], e));
                        continue;
                    }
                    Err(failure) => return Err(failure.into_error()),
                };
                if src.len() != f32_bands * total_pixels * 4
                    || src.as_ptr() == empty_f32_sentinel.as_ptr()
                {
                    continue;
                }
                for i in 0..total_pixels {
                    if !canvas[i].is_nan() {
                        continue;
                    }
                    let mut all_valid = true;
                    for b in 0..f32_bands {
                        let byte_off = (b * total_pixels + i) * 4;
                        let v = f32::from_le_bytes([
                            src[byte_off],
                            src[byte_off + 1],
                            src[byte_off + 2],
                            src[byte_off + 3],
                        ]);
                        if v.is_nan() {
                            all_valid = false;
                            break;
                        }
                    }
                    if all_valid {
                        for b in 0..f32_bands {
                            let byte_off = (b * total_pixels + i) * 4;
                            canvas[b * total_pixels + i] = f32::from_le_bytes([
                                src[byte_off],
                                src[byte_off + 1],
                                src[byte_off + 2],
                                src[byte_off + 3],
                            ]);
                        }
                        filled += 1;
                    }
                }
            }

            if filled == 0 {
                if !opened_any
                    && let Some(err) = last_open_err
                {
                    return Err(err);
                }
                let raw = empty_tile_bytes(req.tile_size, OutputFormat::RawF32, f32_bands);
                return Ok((
                    format_float_tile(raw, f32_bands, req.tile_size, req.format),
                    failed_assets,
                ));
            }
            let mut out = bytes::BytesMut::with_capacity(f32_bands * total_pixels * 4);
            for v in canvas {
                out.extend_from_slice(&v.to_le_bytes());
            }
            let raw = out.freeze();
            return Ok((
                format_float_tile(raw, f32_bands, req.tile_size, req.format),
                failed_assets,
            ));
        }

        let mut canvas = vec![0u8; total_pixels * 4];
        let mut filled = 0usize;

        let mut base_scene_req = req.clone();
        base_scene_req.format = OutputFormat::RawRgba;
        let empty_rgba_sentinel = empty_tile_bytes(req.tile_size, OutputFormat::RawRgba, 1);

        let mut opened_any = false;
        let mut last_open_err: Option<EngineError> = None;
        let mut failed_assets: Vec<String> = Vec::new();
        let mut scene_i = 0usize;
        let mut reading = self.mosaic_scenes_in_order(scenes, &base_scene_req, req.unscale, ahead);
        while filled < total_pixels {
            let scene_urls = &scenes[scene_i.min(scenes.len() - 1)];
            scene_i += 1;
            let Some(src) = reading.next().await else {
                break;
            };
            let src = match src {
                Ok(src) => {
                    opened_any = true;
                    src
                }
                // An asset the allowlist refuses fails the tile, as on `main`, rather than being
                // passed over as one that could not be read.
                Err(SceneFailure::Unopened(e)) if skip_unopened && !refused(&e) => {
                    failed_assets.extend(scene_urls.iter().map(|u| cogzilla_io::redact_http_url(u)));
                    last_open_err = Some(e);
                    continue;
                }
                Err(SceneFailure::Unread(EngineError::Io(e))) if skip_unopened && !e.is_fatal() => {
                    failed_assets.extend(scene_urls.iter().map(|u| cogzilla_io::redact_http_url(u)));
                    last_open_err = Some(asset_fetch_error(&scene_urls[0], e));
                    continue;
                }
                Err(failure) => return Err(failure.into_error()),
            };
            if src.len() != total_pixels * 4 || src.as_ptr() == empty_rgba_sentinel.as_ptr() {
                continue;
            }
            for i in 0..total_pixels {
                let off = i * 4;
                if canvas[off + 3] == 0 && src[off + 3] > 0 {
                    canvas[off] = src[off];
                    canvas[off + 1] = src[off + 1];
                    canvas[off + 2] = src[off + 2];
                    canvas[off + 3] = src[off + 3];
                    if src[off + 3] == 255 {
                        filled += 1;
                    }
                }
            }
        }

        if filled == 0 && canvas.chunks_exact(4).all(|p| p[3] == 0) {
            if !opened_any
                && let Some(err) = last_open_err
            {
                return Err(err);
            }
            return Ok((
                empty_tile_bytes(req.tile_size, req.format, f32_bands),
                failed_assets,
            ));
        }
        // One band drawn in grey is a grey picture, as TiTiler writes it (a palette of a
        // raster's own colours it, and is not).
        let opaque = filled == total_pixels;
        let bytes = if draws_one_grey_band(&req.bands)
            && canvas.chunks_exact(4).all(|p| p[3] == 0 || (p[0] == p[1] && p[1] == p[2]))
        {
            cogzilla_kernel::encode_grey_rgba_tile(&canvas, req.tile_size, opaque, req.format)
        } else {
            encode_rgba_tile(&canvas, req.tile_size, opaque, req.format)
        };
        Ok((bytes, failed_assets))
    }

    /// Query a pluggable `SceneCatalog` for the requested tile's EPSG:3857 bounding box
    /// and render an early-exit mosaic across the intersecting scenes.
    pub async fn render_catalog_mosaic_tile(
        &self,
        catalog: &dyn SceneCatalog,
        req: &TileRequest,
    ) -> Result<Bytes, EngineError> {
        let (min_x, min_y, max_x, max_y) = tile_bounds_3857(req.tile);
        let urls = catalog.query_bbox_3857(min_x, min_y, max_x, max_y);
        if urls.is_empty() {
            self.stats
                .empty_short_circuits
                .fetch_add(1, Ordering::Relaxed);
            let f32_bands = Self::value_bands(&req.bands);
            if matches!(
                req.format,
                OutputFormat::Npy | OutputFormat::RawF64 | OutputFormat::NpyF64
            ) {
                let raw = empty_tile_bytes(req.tile_size, OutputFormat::RawF32, f32_bands);
                return Ok(format_float_tile(raw, f32_bands, req.tile_size, req.format));
            }
            return Ok(empty_tile_bytes(req.tile_size, req.format, f32_bands));
        }
        self.render_mosaic_tile(&urls, req).await
    }
}
