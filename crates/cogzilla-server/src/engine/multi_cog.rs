//! Multi-COG / STAC grid validation and co-registered multi-asset tile rendering.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use bytes::Bytes;
use cogzilla_io::OpenedCog;
use cogzilla_kernel::{
    BandSource, OutputFormat, RgbStyle, SingleBandStyle, empty_tile_bytes, render_bandmath1,
    render_bandmath2, render_bandmath3, render_dem_algorithm, render_expr_vm, render_rgb_bands,
    render_sar_rgb, render_single_band, render_terrain_tile,
};
use cogzilla_proj::{CoordPlan, tile_bounds_3857};

use super::{
    BandRequest, EngineError, TileEngine, TileRequest, format_float_tile, needs_source_unscale,
    run_cpu_render, tile_fetch_range_for_plan,
};

fn affines_match(a: &cogzilla_raster::Affine2D, b: &cogzilla_raster::Affine2D) -> bool {
    let close = |x: f64, y: f64| (x - y).abs() <= 1e-6 * (1.0 + x.abs().max(y.abs()));
    close(a.a, b.a)
        && close(a.b, b.b)
        && close(a.c, b.c)
        && close(a.d, b.d)
        && close(a.e, b.e)
        && close(a.f, b.f)
}

/// Check whether two `GeoKeys` describe the same CRS (falling back to full equality for
/// user-defined GeoTIFF codes `32767`).
fn geo_keys_match(a: &cogzilla_tiff::GeoKeys, b: &cogzilla_tiff::GeoKeys) -> bool {
    if a.model_type != b.model_type
        || a.projected_cs_type != b.projected_cs_type
        || a.geographic_type != b.geographic_type
    {
        return false;
    }
    if a.projected_cs_type.is_none()
        || a.projected_cs_type == Some(32767)
        || a.geographic_type == Some(32767)
    {
        return a == b;
    }
    true
}

/// Fast check whether all COGs in a multi-COG / STAC composite share the exact same base
/// dimensions, tile grid, overview count, CRS, and geotransform across all IFDs (allowing the
/// single-grid fast path).
pub(crate) fn cogs_share_base_grid(cogs: &[Arc<OpenedCog>]) -> bool {
    if cogs.len() <= 1 {
        return true;
    }
    let p_gk = &cogs[0].index.geo_keys;
    let n_ifds = cogs[0].index.ifds.len();
    cogs.iter().skip(1).all(|c| {
        if c.index.ifds.len() != n_ifds || !geo_keys_match(&c.index.geo_keys, p_gk) {
            return false;
        }
        (0..n_ifds).all(|idx| {
            let p = &cogs[0].index.ifds[idx];
            let o = &c.index.ifds[idx];
            o.width == p.width
                && o.height == p.height
                && o.tile_width == p.tile_width
                && o.tile_height == p.tile_height
                && o.tiles_across == p.tiles_across
                && o.tiles_down == p.tiles_down
                && affines_match(&o.pixel_to_crs, &p.pixel_to_crs)
        })
    })
}

/// Verify that all secondary COGs in a multi-COG / STAC composite share the exact same overview
/// index, image dimensions, internal tile grid, CRS, and geotransform as `cogs[0]` so `BandSource`
/// never reads OOB or misregisters pixels.
pub(crate) fn validate_multi_cog_grid(
    cogs: &[Arc<OpenedCog>],
    ifd_idx: usize,
) -> Result<(), EngineError> {
    if cogs.len() <= 1 {
        return Ok(());
    }
    let primary_ifd = &cogs[0].index.ifds[ifd_idx];
    let primary_gk = &cogs[0].index.geo_keys;
    for (k, c) in cogs.iter().enumerate().skip(1) {
        if !geo_keys_match(&c.index.geo_keys, primary_gk) {
            return Err(EngineError::BadRequest(format!(
                "multi-COG asset #{k} CRS does not match primary asset"
            )));
        }
        let Some(other) = c.index.ifds.get(ifd_idx) else {
            return Err(EngineError::BadRequest(format!(
                "multi-COG asset #{k} has {} overviews, fewer than primary asset (need IFD {ifd_idx})",
                c.index.ifds.len()
            )));
        };
        if other.width != primary_ifd.width
            || other.height != primary_ifd.height
            || other.tile_width != primary_ifd.tile_width
            || other.tile_height != primary_ifd.tile_height
            || other.tiles_across != primary_ifd.tiles_across
            || other.tiles_down != primary_ifd.tiles_down
            || !affines_match(&other.pixel_to_crs, &primary_ifd.pixel_to_crs)
        {
            return Err(EngineError::BadRequest(format!(
                "multi-COG asset #{k} grid ({}x{}, tile {}x{}) does not match primary ({}x{}, tile {}x{})",
                other.width,
                other.height,
                other.tile_width,
                other.tile_height,
                primary_ifd.width,
                primary_ifd.height,
                primary_ifd.tile_width,
                primary_ifd.tile_height
            )));
        }
    }
    Ok(())
}

impl TileEngine {
    /// Render a single XYZ tile by compositing 2 or 3 separate co-registered single-band COGs
    /// in parallel (e.g., Landsat C2 `[SR_B4, SR_B3, SR_B2]` RGB or OPERA RTC-S1 `[VV, VH]` SAR-RGB).
    pub async fn render_multi_cog_tile(
        &self,
        cogs: &[Arc<OpenedCog>],
        req: &TileRequest,
    ) -> Result<Bytes, EngineError> {
        if cogs.len() == 1 {
            return self.render_tile(&cogs[0], req).await;
        }
        self.stats.tiles_served.fetch_add(1, Ordering::Relaxed);
        if matches!(req.format, OutputFormat::RawF64 | OutputFormat::NpyF64)
            && !req.unscale
            && matches!(
                req.resample,
                cogzilla_kernel::ResampleMode::Nearest | cogzilla_kernel::ResampleMode::Bilinear
            )
        {
            let bands_1based: Option<Vec<u16>> = match &req.bands {
                BandRequest::Single { band_1based, .. } => Some(vec![*band_1based]),
                BandRequest::Rgb { bands_1based, .. } => Some(bands_1based.to_vec()),
                BandRequest::MultiBand { bands_1based } => Some(bands_1based.clone()),
                _ => None,
            };
            if let Some(b1) = bands_1based {
                let (minx, miny, maxx, maxy) = cogzilla_proj::tile_bounds_3857(req.tile);
                let ts = req.tile_size as usize;
                let n_px = ts * ts;
                let mut vals = Vec::with_capacity(b1.len() * n_px);
                let mut kind = cogzilla_raster::SampleType::Float64;
                let mut first_kind = true;
                for &band_1 in &b1 {
                    let slot = (band_1.saturating_sub(1) as usize).min(cogs.len() - 1);
                    let cog = &cogs[slot];
                    let st = cog.index.full_res().sample_type;
                    if first_kind {
                        kind = st;
                        first_kind = false;
                    } else if kind != st {
                        kind = cogzilla_raster::SampleType::Float64;
                    }
                    let plane_bytes = Box::pin(crate::stats::render_exact_f64_array(
                        self,
                        cog,
                        [minx, miny, maxx, maxy],
                        &cogzilla_proj::CrsModel::Epsg3857,
                        false,
                        req.tile_size,
                        req.tile_size,
                        None,
                        &[1],
                        req.resample,
                        None,
                        false,
                        st,
                        true,
                        true,
                        true,
                    ))
                    .await?;
                    if plane_bytes.len() == n_px * 8 {
                        for chunk in plane_bytes.chunks_exact(8) {
                            vals.push(f64::from_le_bytes(chunk.try_into().unwrap()));
                        }
                    } else {
                        vals.resize(vals.len() + n_px, f64::NAN);
                    }
                }
                if req.format == OutputFormat::RawF64 {
                    let mut out = bytes::BytesMut::with_capacity(vals.len() * 8);
                    for v in vals {
                        out.extend_from_slice(&v.to_le_bytes());
                    }
                    return Ok(out.freeze());
                }
                return Ok(cogzilla_kernel::encode_npy_typed_f64(
                    &vals,
                    b1.len(),
                    ts,
                    ts,
                    kind,
                    None,
                    true,
                ));
            }
        }
        if matches!(
            req.format,
            OutputFormat::Npy | OutputFormat::RawF64 | OutputFormat::NpyF64
        ) {
            let mut raw_req = req.clone();
            raw_req.format = OutputFormat::RawF32;
            let raw = match raw_req.tile_size {
                256 => self.render_multi_cog_sized::<256>(cogs, &raw_req).await?,
                512 => self.render_multi_cog_sized::<512>(cogs, &raw_req).await?,
                other => {
                    return Err(EngineError::BadRequest(format!(
                        "unsupported tile_size={other}, expected 256 or 512"
                    )));
                }
            };
            let nb = Self::value_bands(&req.bands);
            return Ok(format_float_tile(raw, nb, req.tile_size, req.format));
        }
        match req.tile_size {
            256 => self.render_multi_cog_sized::<256>(cogs, req).await,
            512 => self.render_multi_cog_sized::<512>(cogs, req).await,
            other => Err(EngineError::BadRequest(format!(
                "unsupported tile_size={other}, expected 256 or 512"
            ))),
        }
    }

    async fn render_multi_cog_sized<const N: usize>(
        &self,
        cogs: &[Arc<OpenedCog>],
        req: &TileRequest,
    ) -> Result<Bytes, EngineError> {
        if cogs.is_empty() || cogs.len() > 16 {
            return Err(EngineError::BadRequest(format!(
                "multi-COG compositing supports 1..=16 assets, got {}",
                cogs.len()
            )));
        }
        let primary = &cogs[0];
        let crs = self.get_or_build_crs(primary)?;
        let (target_rx, target_ry) = crs.target_resolution_for_tile(req.tile, N as u32);
        let ifd_idx = primary.index.select_ifd(target_rx, target_ry);
        validate_multi_cog_grid(cogs, ifd_idx)?;
        let ifd = &primary.index.ifds[ifd_idx];

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
        let mut indices = Vec::with_capacity((grid_w * grid_h) as usize);
        for ty in ty0..=ty1 {
            for tx in tx0..=tx1 {
                indices.push(Some(ifd.tile_index(tx, ty, 0) as u32));
            }
        }

        // Fetch all co-registered COGs concurrently via JoinSet (supports 2..=16 assets).
        let indices_arc = Arc::new(indices);
        let mut js = tokio::task::JoinSet::new();
        for (slot, c) in cogs.iter().enumerate() {
            let store = self.store.clone();
            let cog = c.clone();
            let idx_ref = indices_arc.clone();
            js.spawn(async move {
                let blocks = store.get_blocks(&cog, ifd_idx, &idx_ref).await?;
                Ok::<_, cogzilla_io::IoError>((slot, blocks))
            });
        }
        let mut fetched: Vec<Option<Vec<Option<Arc<cogzilla_codec::DecodedTile>>>>> =
            vec![None; cogs.len()];
        while let Some(res) = js.join_next().await {
            let (slot, blocks) = res.map_err(|e| {
                EngineError::Io(cogzilla_io::IoError::Fs(std::io::Error::other(format!(
                    "multi-cog tile fetch task failed: {e}"
                ))))
            })??;
            fetched[slot] = Some(blocks);
        }
        let all_tiles: Vec<Vec<Option<Arc<cogzilla_codec::DecodedTile>>>> =
            fetched.into_iter().map(|o| o.unwrap()).collect();
        let all_refs: Vec<Vec<Option<&cogzilla_codec::DecodedBand>>> = all_tiles
            .iter()
            .map(|ts| {
                ts.iter()
                    .map(|o| o.as_ref().and_then(|t| t.bands.first()))
                    .collect()
            })
            .collect();

        let apply_unscale = req.unscale && needs_source_unscale(&req.bands, req.format);
        let make_src = |b_1based: u16| -> Result<BandSource<'_>, EngineError> {
            let idx = b_1based.saturating_sub(1) as usize;
            if b_1based == 0 || idx >= all_refs.len() {
                return Err(EngineError::BadRequest(format!(
                    "band b{b_1based} out of range (only {} assets were provided)",
                    all_refs.len()
                )));
            }
            let (sc, off) = if apply_unscale {
                (
                    cogs[idx].index.scales.first().copied().unwrap_or(1.0),
                    cogs[idx].index.offsets.first().copied().unwrap_or(0.0),
                )
            } else {
                (1.0, 0.0)
            };
            Ok(BandSource::new(
                ifd.width,
                ifd.height,
                ifd.tile_width,
                ifd.tile_height,
                tx0,
                ty0,
                grid_w,
                grid_h,
                &all_refs[idx],
            )
            .with_scale_offset(sc, off))
        };

        let scene_cog_id = |bands_1based: &[u16]| -> u64 {
            // Deterministic FNV-1a hash with high-bit tag so multi-COG scene keys never collide
            // with single-COG `OpenedCog::id` values in `GpuMultiBandCache`.
            let mut h: u64 = 0xcbf2_9ce4_8422_2325;
            for &b in bands_1based {
                let cid = cogs
                    .get(b.saturating_sub(1) as usize)
                    .map(|c| c.id)
                    .unwrap_or(0);
                for byte in cid.to_le_bytes() {
                    h ^= byte as u64;
                    h = h.wrapping_mul(0x0000_0100_0000_01b3);
                }
            }
            h | (1u64 << 63)
        };

        // Band math over the assets (`bN` the Nth), block by block, read as one band.
        // A terrain algorithm over band math reads it over the assets (`bN` the Nth), block
        // by block, as one band.
        let math_blocks = match &req.bands {
            BandRequest::Dem { math: Some(prog), .. } | BandRequest::Terrain { math: Some(prog), .. } => {
                let mut inputs = Vec::with_capacity(prog.used_bands.len());
                for &b in &prog.used_bands {
                    let idx = b.saturating_sub(1) as usize;
                    if b == 0 || idx >= all_refs.len() {
                        return Err(EngineError::BadRequest(format!(
                            "band b{b} out of range (only {} assets were provided)",
                            all_refs.len()
                        )));
                    }
                    let scale_offset = req.unscale.then(|| {
                        (
                            cogs[idx].index.scales.first().copied().unwrap_or(1.0),
                            cogs[idx].index.offsets.first().copied().unwrap_or(0.0),
                        )
                    });
                    inputs.push((&all_refs[idx][..], None, scale_offset));
                }
                // Every pixel of every block read: kept off the async I/O workers.
                Some(super::block_in_place_if_multithread(true, || cogzilla_kernel::band_math_blocks(prog, &inputs)))
            }
            _ => None,
        };
        let math_refs: Option<Vec<Option<&cogzilla_codec::DecodedBand>>> =
            math_blocks.as_ref().map(|blocks| blocks.iter().map(Option::as_ref).collect());
        let terrain_src = |b_1based: u16| -> Result<BandSource<'_>, EngineError> {
            match &math_refs {
                Some(refs) => Ok(BandSource::new(ifd.width, ifd.height, ifd.tile_width, ifd.tile_height, tx0, ty0, grid_w, grid_h, refs)),
                None => make_src(b_1based),
            }
        };
        match &req.bands {
            BandRequest::Single {
                band_1based,
                rescale,
                colormap,
            } => {
                let src = make_src(*band_1based)?;
                let style = SingleBandStyle {
                    rescale_min: rescale.0,
                    rescale_max: rescale.1,
                    colormap: *colormap,
                    resample: req.resample,
                    format: req.format,
                };
                if let Some(gpu) = &self.gpu {
                    let cid = cogs[band_1based.saturating_sub(1) as usize].id;
                    if let Some(bytes) = gpu
                        .try_render_single_band::<N>(cid, ifd_idx, 0, tx0, ty0, &plan, &src, &style)
                        .await
                    {
                        self.stats.gpu_dispatches.fetch_add(1, Ordering::Relaxed);
                        return Ok(bytes);
                    }
                }
                Ok(run_cpu_render(N, style.resample, || {
                    render_single_band::<N>(&plan, &src, &style)
                }).await)
            }
            BandRequest::Terrain {
                band_1based,
                algorithm,
                math,
            } => {
                let src = terrain_src(*band_1based)?;
                if let Some(gpu) = self.gpu.as_ref() {
                    let source_id = math
                        .is_none()
                        .then(|| (cogs[band_1based.saturating_sub(1) as usize].id, ifd_idx, 0));
                    if let Some(bytes) = gpu
                        .try_render_terrain::<N>(
                            source_id,
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
                }
                Ok(run_cpu_render(N, req.resample, || {
                    render_terrain_tile::<N>(&plan, &src, *algorithm, req.resample, req.format)
                }).await)
            }
            BandRequest::Dem {
                band_1based,
                algorithm,
                rescale,
                colormap,
                math,
            } => {
                let src = terrain_src(*band_1based)?;
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
                if let Some(gpu) = self.gpu.as_ref() {
                    let source_id = math
                        .is_none()
                        .then(|| (cogs[band_1based.saturating_sub(1) as usize].id, ifd_idx, 0));
                    if let Some(bytes) = gpu
                        .try_render_dem::<N>(
                            source_id,
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
                }
                Ok(run_cpu_render(N, style.resample, || {
                    render_dem_algorithm::<N>(&plan, &src, *algorithm, pixel_size_m, &style)
                }).await)
            }
            BandRequest::Spectral1 {
                expr,
                band_1based,
                rescale,
                colormap,
            } => {
                let src = make_src(*band_1based)?;
                let style = SingleBandStyle {
                    rescale_min: rescale.0,
                    rescale_max: rescale.1,
                    colormap: *colormap,
                    resample: req.resample,
                    format: req.format,
                };
                if let Some(gpu) = &self.gpu {
                    let cid = cogs[band_1based.saturating_sub(1) as usize].id;
                    if let Some(bytes) = gpu
                        .try_render_spectral::<N>(
                            cid,
                            ifd_idx,
                            &[1],
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
                }
                Ok(run_cpu_render(N, style.resample, || {
                    render_bandmath1::<N>(&plan, &src, *expr, &style)
                }).await)
            }
            BandRequest::SarRgb {
                bands_1based,
                rescale_min,
                rescale_max,
            } => {
                let srcs = [make_src(bands_1based[0])?, make_src(bands_1based[1])?];
                let style = RgbStyle {
                    rescale_min: *rescale_min,
                    rescale_max: *rescale_max,
                    resample: req.resample,
                    format: req.format,
                };
                if let Some(gpu) = &self.gpu {
                    let sid = scene_cog_id(bands_1based);
                    if let Some(bytes) = gpu
                        .try_render_sar_rgb::<N>(sid, ifd_idx, [1, 2], &plan, &srcs, &style)
                        .await
                    {
                        self.stats.gpu_dispatches.fetch_add(1, Ordering::Relaxed);
                        return Ok(bytes);
                    }
                }
                Ok(run_cpu_render(N, style.resample, || {
                    render_sar_rgb::<N>(&plan, &srcs, &style)
                }).await)
            }
            BandRequest::Spectral2 {
                expr,
                bands_1based,
                rescale,
                colormap,
            } => {
                let srcs = [make_src(bands_1based[0])?, make_src(bands_1based[1])?];
                let style = SingleBandStyle {
                    rescale_min: rescale.0,
                    rescale_max: rescale.1,
                    colormap: *colormap,
                    resample: req.resample,
                    format: req.format,
                };
                if let Some(gpu) = &self.gpu {
                    let sid = scene_cog_id(bands_1based);
                    if let Some(bytes) = gpu
                        .try_render_spectral::<N>(sid, ifd_idx, &[1, 2], &plan, &srcs, *expr, &style)
                        .await
                    {
                        self.stats.gpu_dispatches.fetch_add(1, Ordering::Relaxed);
                        return Ok(bytes);
                    }
                }
                Ok(run_cpu_render(N, style.resample, || {
                    render_bandmath2::<N>(&plan, &srcs, *expr, &style)
                }).await)
            }
            BandRequest::Spectral3 {
                expr,
                bands_1based,
                rescale,
                colormap,
            } => {
                let srcs = [
                    make_src(bands_1based[0])?,
                    make_src(bands_1based[1])?,
                    make_src(bands_1based[2])?,
                ];
                let style = SingleBandStyle {
                    rescale_min: rescale.0,
                    rescale_max: rescale.1,
                    colormap: *colormap,
                    resample: req.resample,
                    format: req.format,
                };
                if let Some(gpu) = &self.gpu {
                    let sid = scene_cog_id(bands_1based);
                    if let Some(bytes) = gpu
                        .try_render_spectral::<N>(
                            sid,
                            ifd_idx,
                            &[1, 2, 3],
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
                }
                Ok(run_cpu_render(N, style.resample, || {
                    render_bandmath3::<N>(&plan, &srcs, *expr, &style)
                }).await)
            }
            BandRequest::Rgb {
                bands_1based,
                rescale_min,
                rescale_max,
            } => {
                let srcs = [
                    make_src(bands_1based[0])?,
                    make_src(bands_1based[1])?,
                    make_src(bands_1based[2])?,
                ];
                let style = RgbStyle {
                    rescale_min: *rescale_min,
                    rescale_max: *rescale_max,
                    resample: req.resample,
                    format: req.format,
                };
                if let Some(gpu) = &self.gpu {
                    let sid = scene_cog_id(bands_1based);
                    if let Some(bytes) = gpu
                        .try_render_rgb::<N>(sid, ifd_idx, [1, 2, 3], &plan, &srcs, &style)
                        .await
                    {
                        self.stats.gpu_dispatches.fetch_add(1, Ordering::Relaxed);
                        return Ok(bytes);
                    }
                }
                Ok(run_cpu_render(N, style.resample, || {
                    render_rgb_bands::<N>(&plan, &srcs, &style)
                }).await)
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
                let mut ordered = Vec::with_capacity(prog.used_bands.len());
                for &b in &prog.used_bands {
                    ordered.push(make_src(b)?);
                }
                if let Some(gpu) = &self.gpu {
                    let sid = scene_cog_id(&prog.used_bands);
                    if let Some(bytes) = gpu
                        .try_render_expr::<N>(sid, ifd_idx, &plan, &ordered, prog, &style)
                        .await
                    {
                        self.stats.gpu_dispatches.fetch_add(1, Ordering::Relaxed);
                        return Ok(bytes);
                    }
                }
                Ok(run_cpu_render(N, style.resample, || {
                    render_expr_vm::<N>(&plan, &ordered, prog, &style, (ifd.width, ifd.height))
                }).await)
            }
        }
    }
}
