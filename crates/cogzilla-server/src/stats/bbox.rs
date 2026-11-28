//! Bbox and native-preview renderers (extracted from `stats.rs` for #419).

use super::*;
use crate::multi_asset::render_mixed_grid_bbox;

/// Full-precision `f64` data-array renderer for wide TIFF sample types (`Uint32`, `Int32`,
/// `Uint64`, `Int64`, `Float64`) on unstyled `.npy` and `.tif` routes (`/cog/preview`,
/// `/cog/bbox`, `/cog/feature`, `/cog/tiles`), bypassing `f32` quantization while leaving the
/// hot 8-bit/f32 render path untouched.
#[allow(clippy::too_many_arguments)]
pub async fn render_exact_f64_array(
    engine: &TileEngine,
    cog: &OpenedCog,
    mut bbox: [f64; 4],
    dst_crs: &cogzilla_proj::CrsModel,
    native_preview: bool,
    width: u32,
    height: u32,
    poly_mask: Option<&PolygonMask>,
    bands_1based: &[u16],
    resample: ResampleMode,
    nodata_override: Option<f32>,
    unscale: bool,
    out_kind: cogzilla_raster::SampleType,
    return_mask: bool,
    as_raw_f64: bool,
    allow_empty: bool,
) -> Result<Bytes, EngineError> {
    if !native_preview {
        if !bbox.iter().all(|v| v.is_finite()) || bbox[1] >= bbox[3] {
            return Err(EngineError::BadRequest(format!(
                "invalid bbox: [{}, {}, {}, {}]",
                bbox[0], bbox[1], bbox[2], bbox[3]
            )));
        }
        if dst_crs.is_geographic() && bbox[0] > bbox[2] {
            bbox[2] += 360.0;
        }
        if bbox[0] >= bbox[2] {
            return Err(EngineError::BadRequest(format!(
                "invalid bbox: [{}, {}, {}, {}]",
                bbox[0], bbox[1], bbox[2], bbox[3]
            )));
        }
    }
    let w = width.clamp(1, 4096) as usize;
    let h = height.clamp(1, 4096) as usize;
    let crs = if native_preview && !cog.index.is_georeferenced {
        cogzilla_proj::CrsModel::Geographic
    } else {
        engine.get_or_build_crs(cog)?
    };
    let dlx = (bbox[2] - bbox[0]) / (w as f64);
    let dly = (bbox[3] - bbox[1]) / (h as f64);
    let mut target_res: Option<(f64, f64)> = None;
    let ifd_idx = if native_preview {
        let full = cog.index.full_res();
        let rx0 = full.pixel_to_crs.a.hypot(full.pixel_to_crs.d);
        let ry0 = full.pixel_to_crs.b.hypot(full.pixel_to_crs.e);
        cog.index
            .select_ifd(rx0 * (full.width as f64) / (w as f64), ry0 * (full.height as f64) / (h as f64))
    } else {
        let cx = 0.5 * (bbox[0] + bbox[2]);
        let cy = 0.5 * (bbox[1] + bbox[3]);
        let to_src = |x: f64, y: f64| -> (f64, f64) {
            match dst_crs {
                cogzilla_proj::CrsModel::Epsg3857 => crs.from_3857(x, y),
                cogzilla_proj::CrsModel::Geographic => {
                    let lon = if x > 180.0 { x - 360.0 } else { x };
                    crs.from_lonlat_deg(lon, y)
                }
                other => {
                    let (lon, lat) = other.to_lonlat_deg(x, y);
                    if lon.is_finite() && lat.is_finite() {
                        crs.from_lonlat_deg(lon, lat)
                    } else {
                        (f64::NAN, f64::NAN)
                    }
                }
            }
        };
        let (s0x, s0y) = to_src(cx, cy);
        let (s1x, s1y) = to_src(cx + dlx, cy);
        let (s2x, s2y) = to_src(cx, cy + dly);
        let rx = (s1x - s0x).hypot(s1y - s0y);
        let ry = (s2x - s0x).hypot(s2y - s0y);
        if rx.is_finite() && ry.is_finite() && rx > 0.0 && ry > 0.0 {
            target_res = Some((rx, ry));
            cog.index.select_ifd(rx, ry)
        } else {
            0
        }
    };
    let ifd = &cog.index.ifds[ifd_idx];
    // Source-pixel footprint of one output pixel, for GDAL's widened bilinear when shrinking.
    let (scale_x, scale_y) = if native_preview {
        ((ifd.width as f64 / w as f64) as f32, (ifd.height as f64 / h as f64) as f32)
    } else if let Some((rx, ry)) = target_res {
        let ifd_rx = ifd.pixel_to_crs.a.hypot(ifd.pixel_to_crs.d);
        let ifd_ry = ifd.pixel_to_crs.b.hypot(ifd.pixel_to_crs.e);
        (
            if ifd_rx > 0.0 { (rx / ifd_rx) as f32 } else { 1.0 },
            if ifd_ry > 0.0 { (ry / ifd_ry) as f32 } else { 1.0 },
        )
    } else {
        (1.0, 1.0)
    };
    let widen_kernel = resample != ResampleMode::Nearest
        && cogzilla_kernel::kernel_widens(resample, scale_x, scale_y);
    let n_bands = bands_1based.len();
    for &b in bands_1based {
        if b == 0 || b > ifd.samples_per_pixel {
            return Err(EngineError::BadRequest(format!(
                "bidx {b} out of range (dataset has {} bands)",
                ifd.samples_per_pixel
            )));
        }
    }

    let mut us = vec![f64::NAN; w * h];
    let mut vs = vec![f64::NAN; w * h];
    let mut min_u = f64::INFINITY;
    let mut max_u = f64::NEG_INFINITY;
    let mut min_v = f64::INFINITY;
    let mut max_v = f64::NEG_INFINITY;

    if native_preview {
        let sx = (ifd.width as f64) / (w as f64);
        let sy = (ifd.height as f64) / (h as f64);
        for py in 0..h {
            let v = (py as f64 + 0.5) * sy - 0.5;
            min_v = min_v.min(v);
            max_v = max_v.max(v);
            for px in 0..w {
                let u = (px as f64 + 0.5) * sx - 0.5;
                min_u = min_u.min(u);
                max_u = max_u.max(u);
                us[py * w + px] = u;
                vs[py * w + px] = v;
            }
        }
    } else {
        if let Some(pm) = poly_mask {
            pm.check_work_budget(w * h)?;
        }
        let touched_mask: Option<Vec<bool>> =
            poly_mask.map(|pm| pm.rasterize_all_touched(w, h, bbox, dst_crs));
        let wrap_lo = crs.geographic_0_360_min_lon(&ifd.pixel_to_crs, ifd.width);
        let wrap_hi = crs.geographic_neg_180_max_lon(&ifd.pixel_to_crs, ifd.width);
        for py in 0..h {
            let y = bbox[3] - (py as f64 + 0.5) * dly;
            for px in 0..w {
                if let Some(ref mask) = touched_mask
                    && !mask[py * w + px]
                {
                    continue;
                }
                let x = bbox[0] + (px as f64 + 0.5) * dlx;
                let (sx, sy) = match dst_crs {
                    cogzilla_proj::CrsModel::Epsg3857 => {
                        if wrap_lo.is_some() || wrap_hi.is_some() {
                            let (mut lon, lat) = cogzilla_proj::mercator_to_lonlat_deg(x, y);
                            if let Some(lo) = wrap_lo && lon < lo {
                                lon += 360.0;
                            } else if let Some(hi) = wrap_hi && lon > hi {
                                lon -= 360.0;
                            }
                            crs.from_lonlat_deg(lon, lat)
                        } else {
                            crs.from_3857(x, y)
                        }
                    }
                    cogzilla_proj::CrsModel::Geographic => {
                        let mut lon = if x > 180.0 { x - 360.0 } else { x };
                        if let Some(lo) = wrap_lo && lon < lo {
                            lon += 360.0;
                        } else if let Some(hi) = wrap_hi && lon > hi {
                            lon -= 360.0;
                        }
                        crs.from_lonlat_deg(lon, y)
                    }
                    other => {
                        let (mut lon, lat) = other.to_lonlat_deg(x, y);
                        if !lon.is_finite() || !lat.is_finite() {
                            continue;
                        }
                        if let Some(lo) = wrap_lo && lon < lo {
                            lon += 360.0;
                        } else if let Some(hi) = wrap_hi && lon > hi {
                            lon -= 360.0;
                        }
                        crs.from_lonlat_deg(lon, lat)
                    }
                };
                if !sx.is_finite() || !sy.is_finite() {
                    continue;
                }
                let (col, row) = ifd.crs_to_pixel_pt(sx, sy);
                let u = col - 0.5;
                let v = row - 0.5;
                if u.is_finite() && v.is_finite() {
                    us[py * w + px] = u;
                    vs[py * w + px] = v;
                    if u >= -1.0 && v >= -1.0 && u <= ifd.width as f64 && v <= ifd.height as f64 {
                        min_u = min_u.min(u);
                        max_u = max_u.max(u);
                        min_v = min_v.min(v);
                        max_v = max_v.max(v);
                    }
                }
            }
        }
    }

    let pixels = w * h;
    if !native_preview && !allow_empty && (!min_u.is_finite() || !min_v.is_finite()) {
        let full = cog.index.full_res();
        if !crate::http::tile_box_intersects_cog(&crs, full, dst_crs, bbox) {
            return Err(EngineError::NotFound(format!(
                "bbox [{}, {}, {}, {}] is outside dataset bounds",
                bbox[0], bbox[1], bbox[2], bbox[3]
            )));
        }
    }
    let mut out_f64 = vec![f64::NAN; n_bands * pixels];
    if min_u.is_finite() && min_v.is_finite() {
        let (mx, my) = if widen_kernel {
            (
                cogzilla_kernel::widened_kernel_radius(resample, scale_x).ceil() as i32 + 1,
                cogzilla_kernel::widened_kernel_radius(resample, scale_y).ceil() as i32 + 1,
            )
        } else {
            (1, 1)
        };
        let c0 = (min_u.floor() as i32).saturating_sub(mx).clamp(0, ifd.width as i32 - 1) as u32;
        let c1 = (max_u.ceil() as i32).saturating_add(mx).clamp(0, ifd.width as i32 - 1) as u32;
        let r0 = (min_v.floor() as i32).saturating_sub(my).clamp(0, ifd.height as i32 - 1) as u32;
        let r1 = (max_v.ceil() as i32).saturating_add(my).clamp(0, ifd.height as i32 - 1) as u32;
        let tx0 = c0 / ifd.tile_width;
        let tx1 = c1 / ifd.tile_width;
        let ty0 = r0 / ifd.tile_height;
        let ty1 = r1 / ifd.tile_height;
        let gw = tx1 - tx0 + 1;
        let gh = ty1 - ty0 + 1;

        match ifd.planar_config {
            PlanarConfig::Chunky => {
                let mut indices = Vec::with_capacity((gw * gh) as usize);
                for ty in ty0..=ty1 {
                    for tx in tx0..=tx1 {
                        indices.push(Some(ifd.tile_index(tx, ty, 0) as u32));
                    }
                }
                let tiles = engine.store.get_blocks(cog, ifd_idx, &indices).await?;
                for (bi, &b1) in bands_1based.iter().enumerate() {
                    let b0 = (b1 - 1) as usize;
                    let band_refs: Vec<Option<&cogzilla_codec::DecodedBand>> = tiles
                        .iter()
                        .map(|t| t.as_ref().and_then(|dt| dt.bands.get(b0)))
                        .collect();
                    let mut src = cogzilla_kernel::BandSource::new(
                        ifd.width, ifd.height, ifd.tile_width, ifd.tile_height,
                        tx0, ty0, gw, gh, &band_refs,
                    )
                    .with_nodata_override(nodata_override);
                    if unscale {
                        let sc = cog.index.scales.get(b0).copied().unwrap_or(1.0);
                        let off = cog.index.offsets.get(b0).copied().unwrap_or(0.0);
                        src = src.with_scale_offset(sc, off);
                    }
                    let dst = &mut out_f64[bi * pixels..(bi + 1) * pixels];
                    for i in 0..pixels {
                        dst[i] = if widen_kernel {
                            cogzilla_kernel::sample_f64_kernel_widened(&src, us[i], vs[i], scale_x, scale_y, resample)
                        } else {
                            cogzilla_kernel::sample_f64_general(&src, us[i], vs[i], resample)
                        };
                    }
                }
            }
            PlanarConfig::Planar => {
                for (bi, &b1) in bands_1based.iter().enumerate() {
                    let b0 = (b1 - 1) as usize;
                    let mut indices = Vec::with_capacity((gw * gh) as usize);
                    for ty in ty0..=ty1 {
                        for tx in tx0..=tx1 {
                            indices.push(Some(ifd.tile_index(tx, ty, b0 as u16) as u32));
                        }
                    }
                    let tiles = engine.store.get_blocks(cog, ifd_idx, &indices).await?;
                    let band_refs: Vec<Option<&cogzilla_codec::DecodedBand>> = tiles
                        .iter()
                        .map(|t| t.as_ref().and_then(|dt| dt.bands.first()))
                        .collect();
                    let mut src = cogzilla_kernel::BandSource::new(
                        ifd.width, ifd.height, ifd.tile_width, ifd.tile_height,
                        tx0, ty0, gw, gh, &band_refs,
                    )
                    .with_nodata_override(nodata_override);
                    if unscale {
                        let sc = cog.index.scales.get(b0).copied().unwrap_or(1.0);
                        let off = cog.index.offsets.get(b0).copied().unwrap_or(0.0);
                        src = src.with_scale_offset(sc, off);
                    }
                    let dst = &mut out_f64[bi * pixels..(bi + 1) * pixels];
                    for i in 0..pixels {
                        dst[i] = if widen_kernel {
                            cogzilla_kernel::sample_f64_kernel_widened(&src, us[i], vs[i], scale_x, scale_y, resample)
                        } else {
                            cogzilla_kernel::sample_f64_general(&src, us[i], vs[i], resample)
                        };
                    }
                }
            }
        }
    }

    let bytes = if as_raw_f64 {
        let mut raw = Vec::with_capacity(out_f64.len() * 8);
        for &v in &out_f64 {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        Bytes::from(raw)
    } else {
        cogzilla_kernel::encode_npy_typed_f64(
            &out_f64, n_bands, h, w, out_kind, None, return_mask,
        )
    };
    Ok(bytes)
}

/// Native pixel-space preview renderer backing `/cog/preview*` and `/stac/preview*`,
/// matching TiTiler / `rio-tiler`'s `Reader.preview()` (reads directly in source pixel space
/// without reprojecting through an axis-aligned WGS84 bounding box).
#[allow(clippy::too_many_arguments)]
pub async fn render_native_preview(
    engine: &TileEngine,
    cogs: &[Arc<OpenedCog>],
    bbox_wgs84: [f64; 4],
    width: u32,
    height: u32,
    bands: &BandRequest,
    resample: ResampleMode,
    format: OutputFormat,
    nodata_override: Option<f32>,
    custom_lut: Option<&CustomColormap>,
    color_formula: Option<&ColorFormula>,
    return_mask: bool,
    unscale: bool,
) -> Result<Bytes, EngineError> {
    render_bbox_impl(
        engine,
        cogs,
        bbox_wgs84,
        &cogzilla_proj::CrsModel::Geographic,
        true,
        width,
        height,
        None,
        bands,
        resample,
        format,
        nodata_override,
        custom_lut,
        color_formula,
        return_mask,
        unscale,
    )
    .await
}

/// Arbitrary-dimensions WGS84 bounding-box renderer backing `/cog/bbox`,
/// `POST /cog/feature`, and `WorldCRS84Quad` tiles.
#[allow(clippy::too_many_arguments)]
pub async fn render_wgs84_bbox(
    engine: &TileEngine,
    cogs: &[Arc<OpenedCog>],
    bbox: [f64; 4], // [min_lon, min_lat, max_lon, max_lat]
    width: u32,
    height: u32,
    poly_mask: Option<&PolygonMask>,
    bands: &BandRequest,
    resample: ResampleMode,
    format: OutputFormat,
    nodata_override: Option<f32>,
    custom_lut: Option<&CustomColormap>,
    color_formula: Option<&ColorFormula>,
    return_mask: bool,
    unscale: bool,
) -> Result<Bytes, EngineError> {
    render_bbox_impl(
        engine,
        cogs,
        bbox,
        &cogzilla_proj::CrsModel::Geographic,
        false,
        width,
        height,
        poly_mask,
        bands,
        resample,
        format,
        nodata_override,
        custom_lut,
        color_formula,
        return_mask,
        unscale,
    )
    .await
}

/// Arbitrary-CRS bounding-box renderer backing `?dst_crs=` and all 13 OGC TileMatrixSets.
#[allow(clippy::too_many_arguments)]
pub async fn render_crs_bbox(
    engine: &TileEngine,
    cogs: &[Arc<OpenedCog>],
    bbox: [f64; 4],
    dst_crs: &cogzilla_proj::CrsModel,
    width: u32,
    height: u32,
    poly_mask: Option<&PolygonMask>,
    bands: &BandRequest,
    resample: ResampleMode,
    format: OutputFormat,
    nodata_override: Option<f32>,
    custom_lut: Option<&CustomColormap>,
    color_formula: Option<&ColorFormula>,
    return_mask: bool,
    unscale: bool,
) -> Result<Bytes, EngineError> {
    render_bbox_impl(
        engine,
        cogs,
        bbox,
        dst_crs,
        false,
        width,
        height,
        poly_mask,
        bands,
        resample,
        format,
        nodata_override,
        custom_lut,
        color_formula,
        return_mask,
        unscale,
    )
    .await
}

/// Arbitrary-dimensions EPSG:3857 Web Mercator bounding-box renderer backing `WebMercatorQuad`
/// tiles with `?nodata=`, `?buffer=`, or custom JSON `?colormap=` in true Web Mercator meters.
#[allow(clippy::too_many_arguments)]
pub async fn render_mercator_bbox(
    engine: &TileEngine,
    cogs: &[Arc<OpenedCog>],
    bbox_3857: [f64; 4],
    width: u32,
    height: u32,
    bands: &BandRequest,
    resample: ResampleMode,
    format: OutputFormat,
    nodata_override: Option<f32>,
    custom_lut: Option<&CustomColormap>,
    color_formula: Option<&ColorFormula>,
    return_mask: bool,
    unscale: bool,
) -> Result<Bytes, EngineError> {
    render_bbox_impl(
        engine,
        cogs,
        bbox_3857,
        &cogzilla_proj::CrsModel::Epsg3857,
        false,
        width,
        height,
        None,
        bands,
        resample,
        format,
        nodata_override,
        custom_lut,
        color_formula,
        return_mask,
        unscale,
    )
    .await
}

enum BboxRenderOutput {
    Bytes(Bytes),
    Stats(serde_json::Value),
}

/// Source pixel coordinate representation for `render_bbox_core`.
///
/// Keeping separable 1D axes compact (`w + h` floats = 32 KB at 4096²) instead of eagerly
/// broadcasting to 2D (`2 × w × h` floats = 128 MB) avoids 128 MB CPU allocations, 16.7M-pixel
/// broadcast loops, and 128 MB PCIe `Exact2D` uploads on `/cog/preview`, `/cog/bbox`, `/cog/crop`,
/// `WorldCRS84Quad`, and `WebMercatorQuad` renders — on both GPU and CPU-only deployments.
enum BboxCoordMap {
    Separable { u_cols: Vec<f32>, v_rows: Vec<f32> },
    Grid2D { grid_u: [f32; 289], grid_v: [f32; 289] },
    SeparableMasked { u_cols: Vec<f32>, v_rows: Vec<f32>, mask_bits: Vec<u32> },
    Grid2DMasked { grid_u: [f32; 289], grid_v: [f32; 289], mask_bits: Vec<u32> },
    Exact2D { u_map: Vec<f32>, v_map: Vec<f32> },
}

impl BboxCoordMap {
    #[inline]
    fn uv(&self, px: usize, py: usize, w: usize, h: usize) -> (f32, f32) {
        match self {
            Self::Exact2D { u_map, v_map } => {
                let idx = py * w + px;
                (u_map[idx], v_map[idx])
            }
            Self::Separable { u_cols, v_rows } => {
                let uu = u_cols[px];
                let vv = v_rows[py];
                if uu.is_finite() && vv.is_finite() {
                    (uu, vv)
                } else {
                    (f32::NAN, f32::NAN)
                }
            }
            Self::SeparableMasked { u_cols, v_rows, mask_bits } => {
                let idx = py * w + px;
                if (mask_bits[idx >> 5] & (1u32 << (idx & 31))) == 0 {
                    return (f32::NAN, f32::NAN);
                }
                let uu = u_cols[px];
                let vv = v_rows[py];
                if uu.is_finite() && vv.is_finite() {
                    (uu, vv)
                } else {
                    (f32::NAN, f32::NAN)
                }
            }
            Self::Grid2D { grid_u, grid_v } => Self::eval_grid2d(grid_u, grid_v, px, py, w, h),
            Self::Grid2DMasked { grid_u, grid_v, mask_bits } => {
                let idx = py * w + px;
                if (mask_bits[idx >> 5] & (1u32 << (idx & 31))) == 0 {
                    return (f32::NAN, f32::NAN);
                }
                Self::eval_grid2d(grid_u, grid_v, px, py, w, h)
            }
        }
    }

    #[inline]
    fn eval_grid2d(
        grid_u: &[f32; 289],
        grid_v: &[f32; 289],
        px: usize,
        py: usize,
        w: usize,
        h: usize,
    ) -> (f32, f32) {
        let gy_f = (py as f32) * (16.0f32 / (h as f32));
        let gy = (gy_f as usize).min(15);
        let ty = gy_f - (gy as f32);
        let r0 = gy * 17;
        let r1 = (gy + 1) * 17;
        let gx_f = (px as f32) * (16.0f32 / (w as f32));
        let gx = (gx_f as usize).min(15);
        let tx = gx_f - (gx as f32);
        let u_l = grid_u[r0 + gx] + (grid_u[r1 + gx] - grid_u[r0 + gx]) * ty;
        let u_r = grid_u[r0 + gx + 1] + (grid_u[r1 + gx + 1] - grid_u[r0 + gx + 1]) * ty;
        let v_l = grid_v[r0 + gx] + (grid_v[r1 + gx] - grid_v[r0 + gx]) * ty;
        let v_r = grid_v[r0 + gx + 1] + (grid_v[r1 + gx + 1] - grid_v[r0 + gx + 1]) * ty;
        (u_l + (u_r - u_l) * tx, v_l + (v_r - v_l) * tx)
    }

    /// Visit every pixel `(idx, u, v)` in row-major order without materializing a dense 2D map.
    /// Hoists the variant dispatch and per-row state out of the inner pixel loop.
    #[inline]
    fn for_each_uv(&self, w: usize, h: usize, mut f: impl FnMut(usize, f32, f32)) {
        match self {
            Self::Exact2D { u_map, v_map } => {
                for i in 0..(w * h) {
                    f(i, u_map[i], v_map[i]);
                }
            }
            Self::Separable { u_cols, v_rows } => {
                for py in 0..h {
                    let vv = v_rows[py];
                    let row_off = py * w;
                    if !vv.is_finite() {
                        for px in 0..w {
                            f(row_off + px, f32::NAN, f32::NAN);
                        }
                        continue;
                    }
                    for px in 0..w {
                        let uu = u_cols[px];
                        if uu.is_finite() {
                            f(row_off + px, uu, vv);
                        } else {
                            f(row_off + px, f32::NAN, f32::NAN);
                        }
                    }
                }
            }
            Self::SeparableMasked { u_cols, v_rows, mask_bits } => {
                for py in 0..h {
                    let vv = v_rows[py];
                    let row_off = py * w;
                    if !vv.is_finite() {
                        for px in 0..w {
                            f(row_off + px, f32::NAN, f32::NAN);
                        }
                        continue;
                    }
                    for px in 0..w {
                        let idx = row_off + px;
                        if (mask_bits[idx >> 5] & (1u32 << (idx & 31))) != 0 {
                            let uu = u_cols[px];
                            if uu.is_finite() {
                                f(idx, uu, vv);
                                continue;
                            }
                        }
                        f(idx, f32::NAN, f32::NAN);
                    }
                }
            }
            Self::Grid2D { grid_u, grid_v } => {
                let inv_w = 16.0f32 / (w as f32);
                let inv_h = 16.0f32 / (h as f32);
                for py in 0..h {
                    let gy_f = (py as f32) * inv_h;
                    let gy = (gy_f as usize).min(15);
                    let ty = gy_f - (gy as f32);
                    let r0 = gy * 17;
                    let r1 = (gy + 1) * 17;
                    let row_off = py * w;
                    for px in 0..w {
                        let gx_f = (px as f32) * inv_w;
                        let gx = (gx_f as usize).min(15);
                        let tx = gx_f - (gx as f32);
                        let u_l = grid_u[r0 + gx] + (grid_u[r1 + gx] - grid_u[r0 + gx]) * ty;
                        let u_r = grid_u[r0 + gx + 1] + (grid_u[r1 + gx + 1] - grid_u[r0 + gx + 1]) * ty;
                        let v_l = grid_v[r0 + gx] + (grid_v[r1 + gx] - grid_v[r0 + gx]) * ty;
                        let v_r = grid_v[r0 + gx + 1] + (grid_v[r1 + gx + 1] - grid_v[r0 + gx + 1]) * ty;
                        f(row_off + px, u_l + (u_r - u_l) * tx, v_l + (v_r - v_l) * tx);
                    }
                }
            }
            Self::Grid2DMasked { grid_u, grid_v, mask_bits } => {
                let inv_w = 16.0f32 / (w as f32);
                let inv_h = 16.0f32 / (h as f32);
                for py in 0..h {
                    let gy_f = (py as f32) * inv_h;
                    let gy = (gy_f as usize).min(15);
                    let ty = gy_f - (gy as f32);
                    let r0 = gy * 17;
                    let r1 = (gy + 1) * 17;
                    let row_off = py * w;
                    for px in 0..w {
                        let idx = row_off + px;
                        if (mask_bits[idx >> 5] & (1u32 << (idx & 31))) == 0 {
                            f(idx, f32::NAN, f32::NAN);
                            continue;
                        }
                        let gx_f = (px as f32) * inv_w;
                        let gx = (gx_f as usize).min(15);
                        let tx = gx_f - (gx as f32);
                        let u_l = grid_u[r0 + gx] + (grid_u[r1 + gx] - grid_u[r0 + gx]) * ty;
                        let u_r = grid_u[r0 + gx + 1] + (grid_u[r1 + gx + 1] - grid_u[r0 + gx + 1]) * ty;
                        let v_l = grid_v[r0 + gx] + (grid_v[r1 + gx] - grid_v[r0 + gx]) * ty;
                        let v_r = grid_v[r0 + gx + 1] + (grid_v[r1 + gx + 1] - grid_v[r0 + gx + 1]) * ty;
                        f(idx, u_l + (u_r - u_l) * tx, v_l + (v_r - v_l) * tx);
                    }
                }
            }
        }
    }

    fn as_gpu_source(&self, w: usize, h: usize) -> crate::gpu_backend::BboxCoordSource<'_> {
        match self {
            Self::Separable { u_cols, v_rows } => {
                crate::gpu_backend::BboxCoordSource::Separable {
                    u: u_cols,
                    v: v_rows,
                }
            }
            Self::Grid2D { grid_u, grid_v } => crate::gpu_backend::BboxCoordSource::Grid2D {
                dst_w: w,
                dst_h: h,
                grid_u,
                grid_v,
            },
            Self::SeparableMasked {
                u_cols,
                v_rows,
                mask_bits,
            } => crate::gpu_backend::BboxCoordSource::SeparableMasked {
                u: u_cols,
                v: v_rows,
                mask_bits,
            },
            Self::Grid2DMasked {
                grid_u,
                grid_v,
                mask_bits,
            } => crate::gpu_backend::BboxCoordSource::Grid2DMasked {
                dst_w: w,
                dst_h: h,
                grid_u,
                grid_v,
                mask_bits,
            },
            Self::Exact2D { u_map, v_map } => crate::gpu_backend::BboxCoordSource::Exact2D {
                dst_w: w,
                dst_h: h,
                u_map,
                v_map,
            },
        }
    }
}

/// 1-pass fused GPU (with seamless CPU/2-pass fallback) native-preview band/expression statistics.
#[allow(clippy::too_many_arguments)]
pub async fn render_native_preview_stats(
    engine: &TileEngine,
    cogs: &[Arc<OpenedCog>],
    bbox: [f64; 4],
    width: u32,
    height: u32,
    bands: &BandRequest,
    resample: ResampleMode,
    nodata_override: Option<f32>,
    unscale: bool,
    description: &str,
    stats_opts: &StatsOptions,
) -> Result<serde_json::Value, EngineError> {
    let n_px = (width.clamp(1, 4096) as usize) * (height.clamp(1, 4096) as usize);
    match Box::pin(render_bbox_core(
        engine,
        cogs,
        bbox,
        &cogzilla_proj::CrsModel::Geographic,
        true,
        width,
        height,
        None,
        bands,
        resample,
        OutputFormat::RawF32,
        nodata_override,
        None,
        None,
        true,
        unscale,
        Some((stats_opts, description)),
    ))
    .await?
    {
        BboxRenderOutput::Stats(val) => Ok(val),
        BboxRenderOutput::Bytes(raw) => {
            let pixels: Vec<f32> = raw
                .chunks_exact(4)
                .take(n_px)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            Ok(compute_f32_band_stats_with_engine(
                engine,
                &pixels,
                description,
                stats_opts,
            ))
        }
    }
}

/// 1-pass fused GPU (with seamless CPU/2-pass fallback) WGS84 bbox / polygon-masked statistics.
#[allow(clippy::too_many_arguments)]
pub async fn render_wgs84_bbox_stats(
    engine: &TileEngine,
    cogs: &[Arc<OpenedCog>],
    bbox: [f64; 4],
    width: u32,
    height: u32,
    poly_mask: Option<&PolygonMask>,
    bands: &BandRequest,
    resample: ResampleMode,
    nodata_override: Option<f32>,
    unscale: bool,
    description: &str,
    stats_opts: &StatsOptions,
) -> Result<serde_json::Value, EngineError> {
    let n_px = (width.clamp(1, 4096) as usize) * (height.clamp(1, 4096) as usize);
    match Box::pin(render_bbox_core(
        engine,
        cogs,
        bbox,
        &cogzilla_proj::CrsModel::Geographic,
        false,
        width,
        height,
        poly_mask,
        bands,
        resample,
        OutputFormat::RawF32,
        nodata_override,
        None,
        None,
        true,
        unscale,
        Some((stats_opts, description)),
    ))
    .await?
    {
        BboxRenderOutput::Stats(val) => Ok(val),
        BboxRenderOutput::Bytes(raw) => {
            let pixels: Vec<f32> = raw
                .chunks_exact(4)
                .take(n_px)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            Ok(compute_f32_band_stats_with_engine(
                engine,
                &pixels,
                description,
                stats_opts,
            ))
        }
    }
}

/// Three bands of band math painted as RGB: each channel's program evaluated a run of pixels
/// at a time on the row VM (which works each value as the one-pixel VM does), each sampled at
/// `(u_map, v_map)`, and rescaled by `rescale_min` and `inv`; a pixel with no place or a
/// channel without a value is left as it is. Kept out of line, away from the other renders.
#[inline(never)]
#[allow(clippy::too_many_arguments)]
fn paint_rgb_band_math(
    progs: &[cogzilla_kernel::ExprProgram; 3],
    all_srcs: &[Vec<BandSource<'_>>],
    sample: &impl Fn(&BandSource<'_>, f32, f32) -> f32,
    coord_map: &BboxCoordMap,
    w: usize,
    h: usize,
    rescale_min: [f32; 3],
    inv: [f32; 3],
    rgba: &mut [u8],
) {
    const RUN: usize = 64;
    let npix = w * h;
    let mut rows = vec![[0.0f32; RUN]; 16];
    let mut scratch: Vec<[f32; RUN]> = Vec::new();
    let mut rgb = [[0.0f32; RUN]; 3];
    let mut u_run = [0.0f32; RUN];
    let mut v_run = [0.0f32; RUN];
    for start in (0..npix).step_by(RUN) {
        let n = RUN.min(npix - start);
        for t in 0..n {
            let i = start + t;
            let (u, v) = coord_map.uv(i % w, i / w, w, h);
            u_run[t] = u;
            v_run[t] = v;
        }
        for c in 0..3 {
            let srcs = &all_srcs[c];
            let nb = srcs.len().min(16);
            for t in 0..n {
                let (u, v) = (u_run[t], v_run[t]);
                for k in 0..nb {
                    rows[k][t] = if u.is_nan() { f32::NAN } else { sample(&srcs[k], u, v) };
                }
            }
            progs[c].eval_row_with_scratch::<RUN>(&rows[..nb], &mut scratch, &mut rgb[c]);
        }
        for t in 0..n {
            let i = start + t;
            let px = [rgb[0][t], rgb[1][t], rgb[2][t]];
            if u_run[t].is_nan() || px.iter().any(|v| v.is_nan()) {
                continue;
            }
            for c in 0..3 {
                rgba[i * 4 + c] = ((px[c] - rescale_min[c]) * inv[c]).clamp(0.0, 255.0) as u8;
            }
            rgba[i * 4 + 3] = 255;
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn render_bbox_impl(
    engine: &TileEngine,
    cogs: &[Arc<OpenedCog>],
    bbox: [f64; 4],
    dst_crs: &cogzilla_proj::CrsModel,
    native_preview: bool,
    width: u32,
    height: u32,
    poly_mask: Option<&PolygonMask>,
    bands: &BandRequest,
    resample: ResampleMode,
    format: OutputFormat,
    nodata_override: Option<f32>,
    custom_lut: Option<&CustomColormap>,
    color_formula: Option<&ColorFormula>,
    return_mask: bool,
    unscale: bool,
) -> Result<Bytes, EngineError> {
    match Box::pin(render_bbox_core(
        engine,
        cogs,
        bbox,
        dst_crs,
        native_preview,
        width,
        height,
        poly_mask,
        bands,
        resample,
        format,
        nodata_override,
        custom_lut,
        color_formula,
        return_mask,
        unscale,
        None,
    ))
    .await?
    {
        BboxRenderOutput::Bytes(b) => Ok(b),
        BboxRenderOutput::Stats(_) => unreachable!("fused_stats is None"),
    }
}

#[allow(clippy::too_many_arguments)]
async fn render_bbox_core(
    engine: &TileEngine,
    cogs: &[Arc<OpenedCog>],
    mut bbox: [f64; 4],
    dst_crs: &cogzilla_proj::CrsModel,
    native_preview: bool,
    width: u32,
    height: u32,
    poly_mask: Option<&PolygonMask>,
    bands: &BandRequest,
    resample: ResampleMode,
    format: OutputFormat,
    nodata_override: Option<f32>,
    custom_lut: Option<&CustomColormap>,
    color_formula: Option<&ColorFormula>,
    return_mask: bool,
    unscale: bool,
    fused_stats: Option<(&StatsOptions, &str)>,
) -> Result<BboxRenderOutput, EngineError> {
    if cogs.is_empty() || cogs.len() > 16 {
        return Err(EngineError::BadRequest(format!(
            "render_bbox supports 1..=16 assets, got {}",
            cogs.len()
        )));
    }
    if !native_preview {
        if !bbox.iter().all(|v| v.is_finite()) || bbox[1] >= bbox[3] {
            return Err(EngineError::BadRequest(format!(
                "invalid bbox: [{}, {}, {}, {}]",
                bbox[0], bbox[1], bbox[2], bbox[3]
            )));
        }
        // RFC 7946 antimeridian-crossing geographic bboxes use w > e (e.g. [177, -20, -178, -16]);
        // unwrap e into (180, 540] so dlx > 0 and clamp() never sees min > max.
        if dst_crs.is_geographic() && bbox[0] > bbox[2] {
            bbox[2] += 360.0;
        }
        if bbox[0] >= bbox[2] {
            return Err(EngineError::BadRequest(format!(
                "invalid bbox: [{}, {}, {}, {}]",
                bbox[0], bbox[1], bbox[2], bbox[3]
            )));
        }
    }
    if !matches!(format, OutputFormat::RawF32 | OutputFormat::RawF64 | OutputFormat::Npy | OutputFormat::NpyF64) {
        match bands {
            BandRequest::MultiBand { bands_1based } => {
                return Err(EngineError::BadRequest(format!(
                    "unsupported bidx count {} for visual image output (expected 1 or 3 bands; use .tif, .npy, or .f32)",
                    bands_1based.len()
                )));
            }
            BandRequest::CustomExprMulti { progs } => {
                return Err(EngineError::BadRequest(format!(
                    "unsupported expression count {} for visual image output (expected 1 or 3; use .tif, .npy, or .f32)",
                    progs.len()
                )));
            }
            _ => {}
        }
    }
    // Exact f64 data path for wide sample types (`Uint32`, `Int32`, `Uint64`, `Int64`, `Float64`,
    // including promoted multi-asset integer stacks like `Uint16 + Int16 -> Int32`) on unstyled
    // `.npy` / `.tif` requests (single-COG and multi-COG stacks): bypasses `f32` quantization
    // while leaving the hot 8-bit / f32 render path completely untouched (#619).
    if !cogs.is_empty()
        && !unscale
        && color_formula.is_none()
        && matches!(resample, ResampleMode::Nearest | ResampleMode::Bilinear)
        && matches!(format, OutputFormat::RawF64 | OutputFormat::NpyF64)
    {
        let bands_1based: Option<Vec<u16>> = match bands {
            BandRequest::Single { band_1based, .. } => Some(vec![*band_1based]),
            BandRequest::Rgb { bands_1based, .. } => Some(bands_1based.to_vec()),
            BandRequest::MultiBand { bands_1based } => Some(bands_1based.clone()),
            _ => None,
        };
        if let Some(b1) = bands_1based {
            if cogs.len() == 1 {
                let kind = cogs[0].index.full_res().sample_type;
                return render_exact_f64_array(
                    engine,
                    &cogs[0],
                    bbox,
                    dst_crs,
                    native_preview,
                    width,
                    height,
                    poly_mask,
                    &b1,
                    resample,
                    nodata_override,
                    unscale,
                    kind,
                    return_mask,
                    matches!(format, OutputFormat::RawF64),
                    false,
                )
                .await
                .map(BboxRenderOutput::Bytes);
            }
            let w_us = width.clamp(1, 4096) as usize;
            let h_us = height.clamp(1, 4096) as usize;
            let n_px = w_us * h_us;
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
                } else {
                    kind = crate::http::array_type::wider_type(kind, st);
                }
                let plane_bytes = render_exact_f64_array(
                    engine,
                    cog,
                    bbox,
                    dst_crs,
                    native_preview,
                    width,
                    height,
                    poly_mask,
                    &[1],
                    resample,
                    nodata_override,
                    unscale,
                    st,
                    return_mask,
                    true,
                    false,
                )
                .await?;
                if plane_bytes.len() == n_px * 8 {
                    for chunk in plane_bytes.chunks_exact(8) {
                        vals.push(f64::from_le_bytes(chunk.try_into().unwrap()));
                    }
                } else {
                    vals.resize(vals.len() + n_px, f64::NAN);
                }
            }
            if format == OutputFormat::RawF64 {
                let mut out = BytesMut::with_capacity(vals.len() * 8);
                for v in vals {
                    out.put_slice(&v.to_le_bytes());
                }
                return Ok(BboxRenderOutput::Bytes(out.freeze()));
            }
            return Ok(BboxRenderOutput::Bytes(
                cogzilla_kernel::encode_npy_typed_f64(
                    &vals,
                    b1.len(),
                    h_us,
                    w_us,
                    kind,
                    None,
                    return_mask,
                ),
            ));
        }
    }
    // Never fall through to the RGBA path with `RawF64` / `NpyF64`.
    let format = match format {
        OutputFormat::RawF64 => OutputFormat::RawF32,
        OutputFormat::NpyF64 => OutputFormat::Npy,
        other => other,
    };
    let w = width.clamp(1, 4096) as usize;
    let h = height.clamp(1, 4096) as usize;
    let primary = &cogs[0];
    let crs = if native_preview && !primary.index.is_georeferenced {
        cogzilla_proj::CrsModel::Geographic
    } else {
        engine.get_or_build_crs(primary)?
    };

    let cx = 0.5 * (bbox[0] + bbox[2]);
    let cy = 0.5 * (bbox[1] + bbox[3]);
    let dlx = (bbox[2] - bbox[0]) / (w as f64);
    let dly = (bbox[3] - bbox[1]) / (h as f64);

    let (ifd_idx, target_rx, target_ry) = if native_preview {
        let full = primary.index.full_res();
        let rx0 = full.pixel_to_crs.a.hypot(full.pixel_to_crs.d);
        let ry0 = full.pixel_to_crs.b.hypot(full.pixel_to_crs.e);
        let target_rx = rx0 * (full.width as f64) / (w as f64);
        let target_ry = ry0 * (full.height as f64) / (h as f64);
        (primary.index.select_ifd(target_rx, target_ry), target_rx, target_ry)
    } else {
        let to_src_crs = |x: f64, y: f64| -> (f64, f64) {
            match dst_crs {
                cogzilla_proj::CrsModel::Epsg3857 => crs.from_3857(x, y),
                cogzilla_proj::CrsModel::Geographic => {
                    let lon = if x > 180.0 { x - 360.0 } else { x };
                    crs.from_lonlat_deg(lon, y)
                }
                other => {
                    let (lon, lat) = other.to_lonlat_deg(x, y);
                    if lon.is_finite() && lat.is_finite() {
                        crs.from_lonlat_deg(lon, lat)
                    } else {
                        (f64::NAN, f64::NAN)
                    }
                }
            }
        };

        // Estimate target CRS resolution to pick optimal overview IFD.
        // Probe bbox center first; if out of CRS domain (e.g. WorldCRS84Quad z=0/1 on a UTM COG >75°
        // from center), fall back to the raster's own center clamped into bbox, then 4 quadrant probes.
        let probe_res = |px: f64, py: f64| -> (f64, f64) {
            let (s0x, s0y) = to_src_crs(px, py);
            let (s1x, s1y) = to_src_crs(px + dlx, py);
            let (s2x, s2y) = to_src_crs(px, py + dly);
            let rx = ((s1x - s0x).powi(2) + (s1y - s0y).powi(2)).sqrt();
            let ry = ((s2x - s0x).powi(2) + (s2y - s0y).powi(2)).sqrt();
            (rx, ry)
        };
        let (mut target_rx, mut target_ry) = probe_res(cx, cy);
        if !target_rx.is_finite() || !target_ry.is_finite() {
            let full = primary.index.full_res();
            let rb = crs.wgs84_bounds(&full.pixel_to_crs, full.gcp_poly.as_deref(), full.width, full.height);
            let (mut rcx_raw, rcy_raw) =
                dst_crs.from_lonlat_deg(0.5 * (rb[0] + rb[2]), 0.5 * (rb[1] + rb[3]));
            if dst_crs.is_geographic()
                && rcx_raw < bbox[0]
                && rcx_raw + 360.0 <= bbox[2]
            {
                rcx_raw += 360.0;
            }
            let rcx = rcx_raw.clamp(bbox[0], bbox[2]);
            let rcy = rcy_raw.clamp(bbox[1], bbox[3]);
            let probes = [
                (rcx, rcy),
                (bbox[0] + 0.25 * (bbox[2] - bbox[0]), bbox[1] + 0.25 * (bbox[3] - bbox[1])),
                (bbox[0] + 0.75 * (bbox[2] - bbox[0]), bbox[1] + 0.25 * (bbox[3] - bbox[1])),
                (bbox[0] + 0.25 * (bbox[2] - bbox[0]), bbox[1] + 0.75 * (bbox[3] - bbox[1])),
                (bbox[0] + 0.75 * (bbox[2] - bbox[0]), bbox[1] + 0.75 * (bbox[3] - bbox[1])),
            ];
            for (px, py) in probes {
                let (rx, ry) = probe_res(px, py);
                if rx.is_finite() && ry.is_finite() && rx > 0.0 && ry > 0.0 {
                    target_rx = rx;
                    target_ry = ry;
                    break;
                }
            }
        }
        (primary.index.select_ifd(target_rx, target_ry), target_rx, target_ry)
    };
    if let Some(pm) = poly_mask {
        pm.check_work_budget(w.saturating_mul(h))?;
    }
    if cogs.len() > 1 && crate::engine::validate_multi_cog_grid(cogs, ifd_idx).is_err() {
        return render_mixed_grid_bbox(
            engine,
            cogs,
            bbox,
            dst_crs,
            native_preview,
            w,
            h,
            poly_mask,
            bands,
            resample,
            format,
            nodata_override,
            custom_lut,
            color_formula,
            return_mask,
            unscale,
            None,
        )
        .await
        .map(BboxRenderOutput::Bytes);
    }
    let ifd = &primary.index.ifds[ifd_idx];

    // Precompute source pixel coordinates `(u, v)`.
    // For native preview (`native_preview == true`), map directly in source pixel space without
    // reprojecting through an axis-aligned WGS84 bbox (matching `rio-tiler`'s `Reader.preview()`).
    // Otherwise use a coarse 16-px projection grid + bilinear interpolation.
    let (coord_map, min_u, max_u, min_v, max_v) = if native_preview {
        let sx = (ifd.width as f64) / (w as f64);
        let sy = (ifd.height as f64) / (h as f64);
        let u_cols: Vec<f32> = (0..w).map(|px| ((px as f64 + 0.5) * sx - 0.5) as f32).collect();
        let v_rows: Vec<f32> = (0..h).map(|py| ((py as f64 + 0.5) * sy - 0.5) as f32).collect();
        let min_u = u_cols.first().copied().unwrap_or(0.0);
        let max_u = u_cols.last().copied().unwrap_or(0.0);
        let min_v = v_rows.first().copied().unwrap_or(0.0);
        let max_v = v_rows.last().copied().unwrap_or(0.0);
        (BboxCoordMap::Separable { u_cols, v_rows }, min_u, max_u, min_v, max_v)
    } else {
        let is_separable_fast = ifd.gcp_poly.is_none()
            && ifd.crs_to_pixel.is_axis_aligned()
            && matches!(
                dst_crs,
                cogzilla_proj::CrsModel::Geographic
                    | cogzilla_proj::CrsModel::Epsg3857
                    | cogzilla_proj::CrsModel::WorldMercator
                    | cogzilla_proj::CrsModel::CylindricalEqualArea(_)
            )
            && matches!(
                crs,
                cogzilla_proj::CrsModel::Geographic
                    | cogzilla_proj::CrsModel::Epsg3857
                    | cogzilla_proj::CrsModel::WorldMercator
                    | cogzilla_proj::CrsModel::CylindricalEqualArea(_)
            );
        if is_separable_fast {
            // O(w + h) fast path: zero 2D allocation, zero spawn_blocking overhead, and
            // enables 32 KB `BboxCoordSource::Separable` GPU uploads instead of 128 MB `Exact2D`.
            let wrap_lo = crs.geographic_0_360_min_lon(&ifd.pixel_to_crs, ifd.width);
            let crs_to_pixel = ifd.crs_to_pixel;
            let ifd_w_f = ifd.width as f32;
            let ifd_h_f = ifd.height as f32;
            let to_src_crs = |x: f64, y: f64| -> (f64, f64) {
                let (mut sx, sy) = match dst_crs {
                    cogzilla_proj::CrsModel::Epsg3857 => crs.from_3857(x, y),
                    cogzilla_proj::CrsModel::Geographic => {
                        let lon = if x > 180.0 { x - 360.0 } else { x };
                        crs.from_lonlat_deg(lon, y)
                    }
                    other => {
                        let (lon, lat) = other.to_lonlat_deg(x, y);
                        if lon.is_finite() && lat.is_finite() {
                            crs.from_lonlat_deg(lon, lat)
                        } else {
                            (f64::NAN, f64::NAN)
                        }
                    }
                };
                if let Some(lo) = wrap_lo
                    && sx.is_finite()
                    && sx < lo
                {
                    sx += 360.0;
                }
                (sx, sy)
            };
            let (origin_x, origin_y) = match dst_crs {
                cogzilla_proj::CrsModel::CylindricalEqualArea(cea) => {
                    (cea.false_easting, cea.false_northing)
                }
                _ => (0.0, 0.0),
            };
            let mut u_cols = vec![f32::NAN; w];
            let mut min_u = f32::INFINITY;
            let mut max_u = f32::NEG_INFINITY;
            for (px, slot) in u_cols.iter_mut().enumerate() {
                let x_coord = bbox[0] + (px as f64 + 0.5) * dlx;
                let (sx, _) = to_src_crs(x_coord, origin_y);
                if sx.is_finite() {
                    let uu = (crs_to_pixel.a * sx + crs_to_pixel.c - 0.5) as f32;
                    if uu >= -1.0 && uu <= ifd_w_f {
                        *slot = uu;
                        min_u = min_u.min(uu);
                        max_u = max_u.max(uu);
                    }
                }
            }
            let mut v_rows = vec![f32::NAN; h];
            let mut min_v = f32::INFINITY;
            let mut max_v = f32::NEG_INFINITY;
            for (py, slot) in v_rows.iter_mut().enumerate() {
                let y_coord = bbox[3] - (py as f64 + 0.5) * dly;
                let (_, sy) = to_src_crs(origin_x, y_coord);
                if sy.is_finite() {
                    let vv = (crs_to_pixel.e * sy + crs_to_pixel.f - 0.5) as f32;
                    if vv >= -1.0 && vv <= ifd_h_f {
                        *slot = vv;
                        min_v = min_v.min(vv);
                        max_v = max_v.max(vv);
                    }
                }
            }
            if !min_u.is_finite() || !min_v.is_finite() {
                min_u = f32::INFINITY;
                max_u = f32::NEG_INFINITY;
                min_v = f32::INFINITY;
                max_v = f32::NEG_INFINITY;
            }
            if let Some(pm) = poly_mask {
                let (mask_bits, covered) = pm.rasterize_all_touched_bits(w, h, bbox, dst_crs);
                if covered == 0 {
                    min_u = f32::INFINITY;
                    max_u = f32::NEG_INFINITY;
                    min_v = f32::INFINITY;
                    max_v = f32::NEG_INFINITY;
                }
                (
                    BboxCoordMap::SeparableMasked {
                        u_cols,
                        v_rows,
                        mask_bits,
                    },
                    min_u,
                    max_u,
                    min_v,
                    max_v,
                )
            } else {
                (BboxCoordMap::Separable { u_cols, v_rows }, min_u, max_u, min_v, max_v)
            }
        } else {
            // Try 17x17 coarse Grid2D fast path (2.3 KB vs 128 MB Exact2D!) for curved
            // projections (UTM, Albers, LCC, Polar Stereo) when all 289 control knots are finite
            // and 4 interior midpoint probes verify bilinear error < 0.05 source pixels.
            let grid2d_fast = if ifd.gcp_poly.is_none() {
                let wrap_lo = crs.geographic_0_360_min_lon(&ifd.pixel_to_crs, ifd.width);
                let crs_to_pixel = ifd.crs_to_pixel;
                let ifd_w_f = ifd.width as f32;
                let ifd_h_f = ifd.height as f32;
                let project_px = |px_f: f64, py_f: f64| -> (f32, f32) {
                    let x = bbox[0] + (px_f + 0.5) * dlx;
                    let y = bbox[3] - (py_f + 0.5) * dly;
                    let (mut sx, sy) = match dst_crs {
                        cogzilla_proj::CrsModel::Epsg3857 => crs.from_3857(x, y),
                        cogzilla_proj::CrsModel::Geographic => {
                            let lon = if x > 180.0 { x - 360.0 } else { x };
                            crs.from_lonlat_deg(lon, y)
                        }
                        other => {
                            let (lon, lat) = other.to_lonlat_deg(x, y);
                            if lon.is_finite() && lat.is_finite() {
                                crs.from_lonlat_deg(lon, lat)
                            } else {
                                (f64::NAN, f64::NAN)
                            }
                        }
                    };
                    if let Some(lo) = wrap_lo
                        && sx.is_finite()
                        && sx < lo
                    {
                        sx += 360.0;
                    }
                    if sx.is_finite() && sy.is_finite() {
                        let (col, row) = crs_to_pixel.forward(sx, sy);
                        ((col - 0.5) as f32, (row - 0.5) as f32)
                    } else {
                        (f32::NAN, f32::NAN)
                    }
                };
                let mut grid_u = [0.0f32; 289];
                let mut grid_v = [0.0f32; 289];
                let mut all_finite = true;
                let cell_x = (w as f64) / 16.0;
                let cell_y = (h as f64) / 16.0;
                'grid: for gy in 0..17usize {
                    let py_f = (gy as f64) * cell_y;
                    for gx in 0..17usize {
                        let px_f = (gx as f64) * cell_x;
                        let (uu, vv) = project_px(px_f, py_f);
                        if !uu.is_finite() || !vv.is_finite() {
                            all_finite = false;
                            break 'grid;
                        }
                        grid_u[gy * 17 + gx] = uu;
                        grid_v[gy * 17 + gx] = vv;
                    }
                }
                if all_finite {
                    // Verify curvature accuracy at 4 interior cell midpoints (< 0.05 px).
                    let mut accurate = true;
                    for (gx_i, gy_i) in [(3usize, 3usize), (11, 3), (3, 11), (11, 11)] {
                        let (ex_u, ex_v) =
                            project_px((gx_i as f64 + 0.5) * cell_x, (gy_i as f64 + 0.5) * cell_y);
                        let i00 = gy_i * 17 + gx_i;
                        let bi_u = 0.25
                            * (grid_u[i00]
                                + grid_u[i00 + 1]
                                + grid_u[i00 + 17]
                                + grid_u[i00 + 18]);
                        let bi_v = 0.25
                            * (grid_v[i00]
                                + grid_v[i00 + 1]
                                + grid_v[i00 + 17]
                                + grid_v[i00 + 18]);
                        if !ex_u.is_finite()
                            || !ex_v.is_finite()
                            || (ex_u - bi_u).abs() > 0.05
                            || (ex_v - bi_v).abs() > 0.05
                        {
                            accurate = false;
                            break;
                        }
                    }
                    if accurate {
                        let mut min_u = f32::INFINITY;
                        let mut max_u = f32::NEG_INFINITY;
                        let mut min_v = f32::INFINITY;
                        let mut max_v = f32::NEG_INFINITY;
                        let mut track = |uu: f32, vv: f32| {
                            min_u = min_u.min(uu);
                            max_u = max_u.max(uu);
                            min_v = min_v.min(vv);
                            max_v = max_v.max(vv);
                        };
                        for gy in 0..16usize {
                            for gx in 0..16usize {
                                track(grid_u[gy * 17 + gx], grid_v[gy * 17 + gx]);
                            }
                        }
                        let tx_end = ((w.saturating_sub(1) as f32) * 16.0 / (w as f32) - 15.0)
                            .clamp(0.0, 1.0);
                        let ty_end = ((h.saturating_sub(1) as f32) * 16.0 / (h as f32) - 15.0)
                            .clamp(0.0, 1.0);
                        for gy in 0..16usize {
                            let i0 = gy * 17 + 15;
                            let i1 = gy * 17 + 16;
                            track(
                                grid_u[i0] + tx_end * (grid_u[i1] - grid_u[i0]),
                                grid_v[i0] + tx_end * (grid_v[i1] - grid_v[i0]),
                            );
                        }
                        for gx in 0..16usize {
                            let i0 = 15 * 17 + gx;
                            let i1 = 16 * 17 + gx;
                            track(
                                grid_u[i0] + ty_end * (grid_u[i1] - grid_u[i0]),
                                grid_v[i0] + ty_end * (grid_v[i1] - grid_v[i0]),
                            );
                        }
                        let c00 = 15 * 17 + 15;
                        let c10 = 15 * 17 + 16;
                        let c01 = 16 * 17 + 15;
                        let c11 = 16 * 17 + 16;
                        let u_t = grid_u[c00] + tx_end * (grid_u[c10] - grid_u[c00]);
                        let u_b = grid_u[c01] + tx_end * (grid_u[c11] - grid_u[c01]);
                        let v_t = grid_v[c00] + tx_end * (grid_v[c10] - grid_v[c00]);
                        let v_b = grid_v[c01] + tx_end * (grid_v[c11] - grid_v[c01]);
                        track(u_t + ty_end * (u_b - u_t), v_t + ty_end * (v_b - v_t));

                        if max_u < -1.0 || min_u > ifd_w_f || max_v < -1.0 || min_v > ifd_h_f {
                            min_u = f32::INFINITY;
                            max_u = f32::NEG_INFINITY;
                            min_v = f32::INFINITY;
                            max_v = f32::NEG_INFINITY;
                        }
                        if let Some(pm) = poly_mask {
                            let (mask_bits, covered) =
                                pm.rasterize_all_touched_bits(w, h, bbox, dst_crs);
                            if covered == 0 {
                                min_u = f32::INFINITY;
                                max_u = f32::NEG_INFINITY;
                                min_v = f32::INFINITY;
                                max_v = f32::NEG_INFINITY;
                            }
                            Some((
                                BboxCoordMap::Grid2DMasked {
                                    grid_u,
                                    grid_v,
                                    mask_bits,
                                },
                                min_u,
                                max_u,
                                min_v,
                                max_v,
                            ))
                        } else {
                            Some((
                                BboxCoordMap::Grid2D { grid_u, grid_v },
                                min_u,
                                max_u,
                                min_v,
                                max_v,
                            ))
                        }
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            };
            if let Some(res) = grid2d_fast {
                res
            } else {
            let build_uv_maps = {
        let crs = crs.clone();
        let dst_crs = dst_crs.clone();
        let wrap_lo = crs.geographic_0_360_min_lon(&ifd.pixel_to_crs, ifd.width);
        let wrap_hi = crs.geographic_neg_180_max_lon(&ifd.pixel_to_crs, ifd.width);
        let crs_to_pixel = ifd.crs_to_pixel;
        let gcp_poly = ifd.gcp_poly.clone();
        let ifd_width = ifd.width;
        let ifd_height = ifd.height;
        let poly_mask = poly_mask.cloned();
        move || {
            let map_crs_to_px = |sx: f64, sy: f64| -> (f64, f64) {
                if let Some(ref poly) = gcp_poly {
                    poly.crs_to_pixel(sx, sy)
                } else {
                    crs_to_pixel.forward(sx, sy)
                }
            };
            let touched_mask: Option<Vec<bool>> = poly_mask
                .as_ref()
                .map(|pm| pm.rasterize_all_touched(w, h, bbox, &dst_crs));
            let to_src_crs = |x: f64, y: f64| -> (f64, f64) {
                let (mut sx, sy) = match &dst_crs {
                    cogzilla_proj::CrsModel::Epsg3857 => crs.from_3857(x, y),
                    cogzilla_proj::CrsModel::Geographic => {
                        let lon = if x > 180.0 {
                            x - 360.0
                        } else if x < -180.0 {
                            x + 360.0
                        } else {
                            x
                        };
                        crs.from_lonlat_deg(lon, y)
                    }
                    other => {
                        let (lon, lat) = other.to_lonlat_deg(x, y);
                        if lon.is_finite() && lat.is_finite() {
                            crs.from_lonlat_deg(lon, lat)
                        } else {
                            (f64::NAN, f64::NAN)
                        }
                    }
                };
                if sx.is_finite() {
                    if let Some(lo) = wrap_lo
                        && sx < lo
                    {
                        sx += 360.0;
                    } else if let Some(hi) = wrap_hi
                        && sx > hi
                    {
                        sx -= 360.0;
                    }
                }
                (sx, sy)
            };
            let mut u_map = vec![f32::NAN; w * h];
            let mut v_map = vec![f32::NAN; w * h];
            let mut min_u = f32::INFINITY;
            let mut max_u = f32::NEG_INFINITY;
            let mut min_v = f32::INFINITY;
            let mut max_v = f32::NEG_INFINITY;
            let ifd_w_f = ifd_width as f32;
            let ifd_h_f = ifd_height as f32;

            // Separable 1D fast path: when both dst_crs and src_crs are separable cylindrical/geographic
            // projections and the raster is axis-aligned, u depends only on px and v depends only on py.
            // Uses w + h exact evaluations (faster than a 16-px bilinear grid and 100% exact at every row).
            let is_separable = gcp_poly.is_none()
                && crs_to_pixel.is_axis_aligned()
                && matches!(
                    dst_crs,
                    cogzilla_proj::CrsModel::Geographic
                        | cogzilla_proj::CrsModel::Epsg3857
                        | cogzilla_proj::CrsModel::WorldMercator
                        | cogzilla_proj::CrsModel::CylindricalEqualArea(_)
                )
                && matches!(
                    crs,
                    cogzilla_proj::CrsModel::Geographic
                        | cogzilla_proj::CrsModel::Epsg3857
                        | cogzilla_proj::CrsModel::WorldMercator
                        | cogzilla_proj::CrsModel::CylindricalEqualArea(_)
                );
            if is_separable {
                // Probe separable axes at the destination CRS's origin (0, 0) / (false_easting, false_northing)
                // so a partially overlapping CylindricalEqualArea bbox whose midpoint falls beyond ±y_pole
                // or ±x_max never produces NaN for every column/row.
                let (origin_x, origin_y) = match &dst_crs {
                    cogzilla_proj::CrsModel::CylindricalEqualArea(cea) => {
                        (cea.false_easting, cea.false_northing)
                    }
                    _ => (0.0, 0.0),
                };
                let mut u_cols = vec![f32::NAN; w];
                for px in 0..w {
                    let x_coord = bbox[0] + (px as f64 + 0.5) * dlx;
                    let (sx, _) = to_src_crs(x_coord, origin_y);
                    if sx.is_finite() {
                        let uu = (crs_to_pixel.a * sx + crs_to_pixel.c - 0.5) as f32;
                        if uu >= -1.0 && uu <= ifd_w_f {
                            u_cols[px] = uu;
                        }
                    }
                }
                let mut v_rows = vec![f32::NAN; h];
                for py in 0..h {
                    let y_coord = bbox[3] - (py as f64 + 0.5) * dly;
                    let (_, sy) = to_src_crs(origin_x, y_coord);
                    if sy.is_finite() {
                        let vv = (crs_to_pixel.e * sy + crs_to_pixel.f - 0.5) as f32;
                        if vv >= -1.0 && vv <= ifd_h_f {
                            v_rows[py] = vv;
                        }
                    }
                }
                for py in 0..h {
                    let vv = v_rows[py];
                    if !vv.is_finite() {
                        continue;
                    }
                    let row_off = py * w;
                    if let Some(ref tm) = touched_mask {
                        for px in 0..w {
                            let uu = u_cols[px];
                            if tm[row_off + px] && uu.is_finite() {
                                u_map[row_off + px] = uu;
                                v_map[row_off + px] = vv;
                                min_u = min_u.min(uu);
                                max_u = max_u.max(uu);
                                min_v = min_v.min(vv);
                                max_v = max_v.max(vv);
                            }
                        }
                    } else {
                        for px in 0..w {
                            let uu = u_cols[px];
                            if uu.is_finite() {
                                u_map[row_off + px] = uu;
                                v_map[row_off + px] = vv;
                                min_u = min_u.min(uu);
                                max_u = max_u.max(uu);
                                min_v = min_v.min(vv);
                                max_v = max_v.max(vv);
                            }
                        }
                    }
                }
                return (u_map, v_map, min_u, max_u, min_v, max_v);
            }

            const STEP: usize = 16;
            let gw = w.div_ceil(STEP) + 1;
            let gh = h.div_ceil(STEP) + 1;
            let mut grid_u = vec![f64::NAN; gw * gh];
            let mut grid_v = vec![f64::NAN; gw * gh];
            for gy in 0..gh {
                let py_f = (gy * STEP).min(h) as f64;
                let y_coord = bbox[3] - py_f * dly;
                for gx in 0..gw {
                    let px_f = (gx * STEP).min(w) as f64;
                    let x_coord = bbox[0] + px_f * dlx;
                    let (sx, sy) = to_src_crs(x_coord, y_coord);
                    if sx.is_finite() && sy.is_finite() {
                        let (col, row) = map_crs_to_px(sx, sy);
                        grid_u[gy * gw + gx] = col - 0.5;
                        grid_v[gy * gw + gx] = row - 0.5;
                    }
                }
            }

            for gy in 0..(gh - 1) {
                let y0 = gy * STEP;
                let y1 = ((gy + 1) * STEP).min(h);
                if y0 >= y1 {
                    continue;
                }
                let inv_dy = 1.0 / ((y1 - y0) as f64);
                for gx in 0..(gw - 1) {
                    let x0 = gx * STEP;
                    let x1 = ((gx + 1) * STEP).min(w);
                    if x0 >= x1 {
                        continue;
                    }
                    let inv_dx = 1.0 / ((x1 - x0) as f64);
                    let i00 = gy * gw + gx;
                    let u00 = grid_u[i00];
                    let u10 = grid_u[i00 + 1];
                    let u01 = grid_u[i00 + gw];
                    let u11 = grid_u[i00 + gw + 1];
                    let v00 = grid_v[i00];
                    let v10 = grid_v[i00 + 1];
                    let v01 = grid_v[i00 + gw];
                    let v11 = grid_v[i00 + gw + 1];
                    let all_finite = u00.is_finite()
                        && u10.is_finite()
                        && u01.is_finite()
                        && u11.is_finite()
                        && v00.is_finite()
                        && v10.is_finite()
                        && v01.is_finite()
                        && v11.is_finite();

                    // Skip entire cell if all 4 corners + midpoint are strictly outside the source image
                    // bounds, and fall back to exact per-pixel projection if the cell has high curvature
                    // (> 0.05 px midpoint error) or wraps across the antimeridian / pole in X or Y.
                    let mut use_bilinear = all_finite;
                    if all_finite {
                        let x_mid = bbox[0] + (0.5 * (x0 + x1) as f64) * dlx;
                        let y_mid = bbox[3] - (0.5 * (y0 + y1) as f64) * dly;
                        let (msx, msy) = to_src_crs(x_mid, y_mid);
                        let mid_ok = if msx.is_finite() && msy.is_finite() {
                            let (mcol, mrow) = map_crs_to_px(msx, msy);
                            let mu = mcol - 0.5;
                            let mv = mrow - 0.5;
                            let u_avg = 0.25 * (u00 + u10 + u01 + u11);
                            let v_avg = 0.25 * (v00 + v10 + v01 + v11);
                            mu.is_finite()
                                && mv.is_finite()
                                && (mu - u_avg).abs() <= 1e-3
                                && (mv - v_avg).abs() <= 1e-3
                        } else {
                            false
                        };
                        if !mid_ok {
                            use_bilinear = false;
                        } else {
                            let cu_min = u00.min(u10).min(u01).min(u11) as f32;
                            let cu_max = u00.max(u10).max(u01).max(u11) as f32;
                            let cv_min = v00.min(v10).min(v01).min(v11) as f32;
                            let cv_max = v00.max(v10).max(v01).max(v11) as f32;
                            if cu_max < -1.0 || cu_min > ifd_w_f || cv_max < -1.0 || cv_min > ifd_h_f
                            {
                                continue;
                            }
                            if (ifd_width >= 64 && w > STEP * 2 && (cu_max - cu_min) > 0.5 * ifd_w_f)
                                || (ifd_height >= 64
                                    && h > STEP * 2
                                    && (cv_max - cv_min) > 0.5 * ifd_h_f)
                            {
                                use_bilinear = false;
                            }
                        }
                    }

                    for py in y0..y1 {
                        let row_off = py * w;
                        let ty = (py as f64 + 0.5 - y0 as f64) * inv_dy;
                        let lat = bbox[3] - (py as f64 + 0.5) * dly;
                        if use_bilinear {
                            let ul = u00 + (u01 - u00) * ty;
                            let ur = u10 + (u11 - u10) * ty;
                            let vl = v00 + (v01 - v00) * ty;
                            let vr = v10 + (v11 - v10) * ty;
                            let du = (ur - ul) * inv_dx;
                            let dv = (vr - vl) * inv_dx;
                            let mut u_cur = ul + 0.5 * du;
                            let mut v_cur = vl + 0.5 * dv;
                            if let Some(ref tm) = touched_mask {
                                for px in x0..x1 {
                                    if tm[row_off + px] {
                                        let uu = u_cur as f32;
                                        let vv = v_cur as f32;
                                        if uu >= -1.0
                                            && uu <= ifd_w_f
                                            && vv >= -1.0
                                            && vv <= ifd_h_f
                                        {
                                            u_map[row_off + px] = uu;
                                            v_map[row_off + px] = vv;
                                            min_u = min_u.min(uu);
                                            max_u = max_u.max(uu);
                                            min_v = min_v.min(vv);
                                            max_v = max_v.max(vv);
                                        }
                                    }
                                    u_cur += du;
                                    v_cur += dv;
                                }
                            } else {
                                for px in x0..x1 {
                                    let uu = u_cur as f32;
                                    let vv = v_cur as f32;
                                    if uu >= -1.0
                                        && uu <= ifd_w_f
                                        && vv >= -1.0
                                        && vv <= ifd_h_f
                                    {
                                        u_map[row_off + px] = uu;
                                        v_map[row_off + px] = vv;
                                        min_u = min_u.min(uu);
                                        max_u = max_u.max(uu);
                                        min_v = min_v.min(vv);
                                        max_v = max_v.max(vv);
                                    }
                                    u_cur += du;
                                    v_cur += dv;
                                }
                            }
                        } else {
                            for px in x0..x1 {
                                let lon = bbox[0] + (px as f64 + 0.5) * dlx;
                                if let Some(ref tm) = touched_mask
                                    && !tm[row_off + px] {
                                        continue;
                                    }
                                let (sx, sy) = to_src_crs(lon, lat);
                                if !sx.is_finite() || !sy.is_finite() {
                                    continue;
                                }
                                let (col, row) = map_crs_to_px(sx, sy);
                                let uu = (col - 0.5) as f32;
                                let vv = (row - 0.5) as f32;
                                if uu >= -1.0 && uu <= ifd_w_f && vv >= -1.0 && vv <= ifd_h_f {
                                    u_map[row_off + px] = uu;
                                    v_map[row_off + px] = vv;
                                    min_u = min_u.min(uu);
                                    max_u = max_u.max(uu);
                                    min_v = min_v.min(vv);
                                    max_v = max_v.max(vv);
                                }
                            }
                        }
                    }
                }
            }
            (u_map, v_map, min_u, max_u, min_v, max_v)
        }
    };
    let (u_map, v_map, min_u, max_u, min_v, max_v) = if w * h >= 65_536 || poly_mask.is_some() {
        engine
            .store
            .run_heavy_cpu(build_uv_maps)
            .await?
    } else {
        build_uv_maps()
    };
    (BboxCoordMap::Exact2D { u_map, v_map }, min_u, max_u, min_v, max_v)
    }
    }
    };

    // Empty check: zero source pixel centers mapped into the requested tile/bbox/feature.
    // Only return 404 when the requested bbox genuinely does not intersect the dataset footprint;
    // coarse grids over wide/dateline-spanning polygons whose pixel centers miss narrow edge
    // overlaps still overlap the dataset and should render an empty (all-nodata) 200 image.
    let empty_but_overlapping = if !min_u.is_finite() {
        let full = primary.index.full_res();
        if !crate::http::tile_box_intersects_cog(&crs, full, dst_crs, bbox) {
            let rb = crs.wgs84_bounds(&full.pixel_to_crs, full.gcp_poly.as_deref(), full.width, full.height);
            return Err(EngineError::NotFound(format!(
                "request is outside dataset bounds [{:.6}, {:.6}, {:.6}, {:.6}]",
                rb[0], rb[1], rb[2], rb[3]
            )));
        }
        true
    } else {
        false
    };

    let (footprint_sx, footprint_sy) = if native_preview {
        (
            (ifd.width as f64 / w as f64) as f32,
            (ifd.height as f64 / h as f64) as f32,
        )
    } else {
        let ifd_rx = ifd.pixel_to_crs.a.hypot(ifd.pixel_to_crs.d);
        let ifd_ry = ifd.pixel_to_crs.b.hypot(ifd.pixel_to_crs.e);
        let sx = if target_rx.is_finite() && ifd_rx > 0.0 {
            (target_rx / ifd_rx).max(1e-3) as f32
        } else {
            1.0
        };
        let sy = if target_ry.is_finite() && ifd_ry > 0.0 {
            (target_ry / ifd_ry).max(1e-3) as f32
        } else {
            1.0
        };
        (sx, sy)
    };
    let is_footprint_mode = matches!(
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
    let max_c = (ifd.width as i32 - 1).max(0);
    let max_r = (ifd.height as i32 - 1).max(0);
    let is_dem = matches!(bands, BandRequest::Dem { .. });
    // GDAL-style widened kernel when the request shrinks the overview (bilinear/cubic/
    // cubicspline/lanczos): the support grows to `radius * scale` source pixels.
    let widen_kernel = resample != ResampleMode::Nearest
        && cogzilla_kernel::kernel_widens(resample, footprint_sx, footprint_sy);
    let (c0, c1, r0, r1) = if empty_but_overlapping {
        (0u32, 0u32, 0u32, 0u32)
    } else if widen_kernel {
        let extra = if is_dem { 2 } else { 0 };
        let hx = cogzilla_kernel::widened_kernel_radius(resample, footprint_sx).ceil() as i32 + extra;
        let hy = cogzilla_kernel::widened_kernel_radius(resample, footprint_sy).ceil() as i32 + extra;
        (
            (min_u.floor() as i32).saturating_sub(hx).clamp(0, max_c) as u32,
            (max_u.ceil() as i32).saturating_add(hx).clamp(0, max_c) as u32,
            (min_v.floor() as i32).saturating_sub(hy).clamp(0, max_r) as u32,
            (max_v.ceil() as i32).saturating_add(hy).clamp(0, max_r) as u32,
        )
    } else if is_dem {
        let (lo, hi) = crate::engine::resample_fetch_margin(resample, true);
        (
            (min_u.floor() as i32).saturating_sub(lo).clamp(0, max_c) as u32,
            (max_u.floor() as i32).saturating_add(hi).clamp(0, max_c) as u32,
            (min_v.floor() as i32).saturating_sub(lo).clamp(0, max_r) as u32,
            (max_v.floor() as i32).saturating_add(hi).clamp(0, max_r) as u32,
        )
    } else if resample == ResampleMode::Nearest {
        (
            ((min_u + 0.5).floor() as i32).clamp(0, max_c) as u32,
            ((max_u + 0.5).floor() as i32).clamp(0, max_c) as u32,
            ((min_v + 0.5).floor() as i32).clamp(0, max_r) as u32,
            ((max_v + 0.5).floor() as i32).clamp(0, max_r) as u32,
        )
    } else if is_footprint_mode {
        let hx = 0.5 * footprint_sx;
        let hy = 0.5 * footprint_sy;
        (
            ((min_u + 0.5 - hx).floor() as i32).clamp(0, max_c) as u32,
            (((max_u + 0.5 + hx).ceil() as i32) - 1).clamp(0, max_c) as u32,
            ((min_v + 0.5 - hy).floor() as i32).clamp(0, max_r) as u32,
            (((max_v + 0.5 + hy).ceil() as i32) - 1).clamp(0, max_r) as u32,
        )
    } else {
        let (lo, hi) = crate::engine::resample_fetch_margin(resample, false);
        (
            (min_u.floor() as i32).saturating_sub(lo).clamp(0, max_c) as u32,
            (max_u.floor() as i32).saturating_add(hi).clamp(0, max_c) as u32,
            (min_v.floor() as i32).saturating_sub(lo).clamp(0, max_r) as u32,
            (max_v.floor() as i32).saturating_add(hi).clamp(0, max_r) as u32,
        )
    };
    let tx1 = (c1 / ifd.tile_width).min(ifd.tiles_across.saturating_sub(1));
    let tx0 = (c0 / ifd.tile_width).min(tx1);
    let ty1 = (r1 / ifd.tile_height).min(ifd.tiles_down.saturating_sub(1));
    let ty0 = (r0 / ifd.tile_height).min(ty1);
    let grid_w = tx1 - tx0 + 1;
    let grid_h = ty1 - ty0 + 1;

    let grid_len = (grid_w * grid_h) as usize;
    let is_single_planar =
        cogs.len() == 1 && ifd.planar_config == cogzilla_tiff::PlanarConfig::Planar;

    // Validate 1-based band indices on both single- and multi-COG requests and collect unique
    // 0-based planes needed when a single COG is `PlanarConfig::Planar`.
    let mut planar_bands: Vec<u16> = Vec::new();
    let max_bands = if cogs.len() == 1 {
        ifd.samples_per_pixel
    } else {
        cogs.len() as u16
    };
    {
        let mut check_band = |b_1based: u16| -> Result<(), EngineError> {
            let b0 = b_1based.saturating_sub(1);
            if b_1based == 0 || b0 >= max_bands {
                return Err(EngineError::BadRequest(format!(
                    "band {} out of range (dataset has {} bands)",
                    b_1based, max_bands
                )));
            }
            if is_single_planar && !planar_bands.contains(&b0) {
                planar_bands.push(b0);
            }
            Ok(())
        };
        match bands {
            // A terrain algorithm over band math reads the bands of the band math.
            BandRequest::Dem { math: Some(prog), .. } | BandRequest::Terrain { math: Some(prog), .. } => {
                for &b in &prog.used_bands {
                    check_band(b)?;
                }
            }
            BandRequest::Single { band_1based, .. }
            | BandRequest::Terrain { band_1based, .. }
            | BandRequest::Dem { band_1based, .. }
            | BandRequest::Spectral1 { band_1based, .. } => check_band(*band_1based)?,
            BandRequest::SarRgb { bands_1based, .. }
            | BandRequest::Spectral2 { bands_1based, .. } => {
                check_band(bands_1based[0])?;
                check_band(bands_1based[1])?;
            }
            BandRequest::Rgb { bands_1based, .. }
            | BandRequest::Spectral3 { bands_1based, .. } => {
                check_band(bands_1based[0])?;
                check_band(bands_1based[1])?;
                check_band(bands_1based[2])?;
            }
            BandRequest::CustomExpr { prog, .. } => {
                for &b in &prog.used_bands {
                    check_band(b)?;
                }
            }
            BandRequest::CustomExprRgb { progs, .. } => {
                for prog in progs {
                    for &b in &prog.used_bands {
                        check_band(b)?;
                    }
                }
            }
            BandRequest::CustomExprMulti { progs } => {
                for prog in progs {
                    for &b in &prog.used_bands {
                        check_band(b)?;
                    }
                }
            }
            BandRequest::MultiBand { bands_1based } => {
                for &b in bands_1based {
                    check_band(b)?;
                }
            }
        }
    }

    // For single-COG RGBA / Gray+Alpha sources, capture the 1-based alpha band index and
    // include it in `planar_bands` when planar so intermediate translucency (1..254) can be
    // preserved on visual and `.npy` outputs.
    let alpha_band_info: Option<(u16, f32)> = if cogs.len() == 1
        && fused_stats.is_none()
        && matches!(
            format,
            OutputFormat::Png | OutputFormat::Webp { .. } | OutputFormat::RawRgba | OutputFormat::Npy
        )
        && let Some(alpha_idx) =
            cogzilla_codec::alpha_channel_index(ifd.samples_per_pixel as usize, &ifd.extra_samples)
    {
        let alpha_b0 = alpha_idx as u16;
        if is_single_planar && !planar_bands.contains(&alpha_b0) {
            planar_bands.push(alpha_b0);
        }
        Some((alpha_b0 + 1, cogzilla_kernel::alpha_full_opaque(ifd.sample_type)))
    } else {
        None
    };

    // Skip block I/O completely for 0-band constant expressions (u_map/v_map already enforce
    // raster footprint + polygon bounds).
    let zero_band_expr = match bands {
        BandRequest::CustomExpr { prog, .. } => prog.used_bands.is_empty(),
        BandRequest::CustomExprRgb { progs, .. } => {
            progs.iter().all(|p| p.used_bands.is_empty())
        }
        BandRequest::CustomExprMulti { progs } => {
            progs.iter().all(|p| p.used_bands.is_empty())
        }
        _ => false,
    };

    // Fetch needed bands (single COG chunky/planar or multi-COG)
    let cog_tiles = if zero_band_expr {
        vec![Vec::new(); cogs.len()]
    } else if cogs.len() == 1 {
        if is_single_planar {
            let mut indices = Vec::with_capacity(grid_len * planar_bands.len().max(1));
            for &band in &planar_bands {
                for ty in ty0..=ty1 {
                    for tx in tx0..=tx1 {
                        indices.push(Some(ifd.tile_index(tx, ty, band) as u32));
                    }
                }
            }
            vec![engine.store.get_blocks(&cogs[0], ifd_idx, &indices).await?]
        } else {
            let mut indices = Vec::with_capacity(grid_len);
            for ty in ty0..=ty1 {
                for tx in tx0..=tx1 {
                    indices.push(Some(ifd.tile_index(tx, ty, 0) as u32));
                }
            }
            vec![engine.store.get_blocks(&cogs[0], ifd_idx, &indices).await?]
        }
    } else {
        let mut indices = Vec::with_capacity(grid_len);
        for ty in ty0..=ty1 {
            for tx in tx0..=tx1 {
                indices.push(Some(ifd.tile_index(tx, ty, 0) as u32));
            }
        }
        let indices_arc = Arc::new(indices);
        let mut js = tokio::task::JoinSet::new();
        for (slot, c) in cogs.iter().enumerate() {
            let store = engine.store.clone();
            let cog = c.clone();
            let idx_ref = indices_arc.clone();
            js.spawn(async move {
                let blocks = store.get_blocks(&cog, ifd_idx, &idx_ref).await?;
                Ok::<_, cogzilla_io::IoError>((slot, blocks))
            });
        }
        let mut out: Vec<Option<Vec<Option<Arc<cogzilla_codec::DecodedTile>>>>> =
            vec![None; cogs.len()];
        while let Some(res) = js.join_next().await {
            let (slot, blocks) = res.map_err(|e| {
                EngineError::Io(cogzilla_io::IoError::Fs(std::io::Error::other(format!(
                    "multi-cog bbox fetch task failed: {e}"
                ))))
            })??;
            out[slot] = Some(blocks);
        }
        out.into_iter().map(|o| o.unwrap()).collect()
    };

    let ifd_width = ifd.width;
    let ifd_height = ifd.height;
    let ifd_tile_width = ifd.tile_width;
    let ifd_tile_height = ifd.tile_height;
    let num_cogs = cogs.len();
    let band_scale_offsets: Vec<(f32, f32)> = if num_cogs > 1 {
        cogs.iter()
            .map(|c| {
                (
                    c.index.scales.first().copied().unwrap_or(1.0),
                    c.index.offsets.first().copied().unwrap_or(0.0),
                )
            })
            .collect()
    } else {
        let n = primary.index.scales.len().max(primary.index.offsets.len());
        (0..n)
            .map(|i| {
                (
                    primary.index.scales.get(i).copied().unwrap_or(1.0),
                    primary.index.offsets.get(i).copied().unwrap_or(0.0),
                )
            })
            .collect()
    };

    let native_dem_dxy: Option<(f32, f32)> = if native_preview {
        let full = primary.index.full_res();
        let rx = full.pixel_to_crs.a.hypot(full.pixel_to_crs.d) * (full.width as f64) / (w as f64);
        let ry = full.pixel_to_crs.b.hypot(full.pixel_to_crs.e) * (full.height as f64) / (h as f64);
        // In the raster's own units, as TiTiler divides by the transform's pixel size.
        Some((rx.max(1e-9) as f32, ry.max(1e-9) as f32))
    } else {
        None
    };
    // Shrinking convolution requests stay on the CPU, which widens the kernel like GDAL (the
    // shaders keep the fixed taps).
    let gpu_ctx = engine.gpu.as_ref().filter(|_| !widen_kernel).map(|g| {
        let asset_ids: Vec<u64> = cogs.iter().map(|c| c.id).collect();
        (Arc::clone(g), Arc::clone(&engine.stats), primary.id, asset_ids)
    });
    let render_cpu = move |coord_map: BboxCoordMap,
                           cog_tiles: Vec<Vec<Option<Arc<cogzilla_codec::DecodedTile>>>>,
                           bands: &BandRequest,
                           custom_lut: Option<&CustomColormap>,
                           color_formula: Option<&ColorFormula>,
                           fused_stats: Option<(&StatsOptions, &str)>|
          -> BboxRenderOutput {
        let ifd_w_i = ifd_width as i32;
        let ifd_h_i = ifd_height as i32;
        // Same-CRS boxes and previews have separable maps: lazily initialize the 2-pass separable
        // convolution sampler only when CPU rendering actually runs (zero allocation on GPU hits).
        // For `CubicSpline` and `Lanczos` at `scale <= 1.0` (`!widen_kernel`), `with_grid` turns
        // per-pixel 2D 4x4/6x6 sampling into two 1D passes (~8x faster on 1024x1024 Lanczos).
        // If `with_grid` declines (`> WIDE_GRID_MAX_PIXELS`) when `!widen_kernel`, return `None`
        // rather than falling back to `new_uncached`.
        let use_separable_conv = matches!(coord_map, BboxCoordMap::Separable { .. })
            && matches!(resample, ResampleMode::CubicSpline | ResampleMode::Lanczos);
        let wide_cell: std::cell::OnceCell<Option<cogzilla_kernel::WidenedSampler>> =
            std::cell::OnceCell::new();
        let get_wide = || -> Option<&cogzilla_kernel::WidenedSampler> {
            wide_cell
                .get_or_init(|| {
                    if !widen_kernel && !use_separable_conv {
                        return None;
                    }
                    let grid = match &coord_map {
                        BboxCoordMap::Separable { u_cols, v_rows } => {
                            cogzilla_kernel::WidenedSampler::with_grid(
                                resample,
                                footprint_sx,
                                footprint_sy,
                                u_cols.clone(),
                                v_rows.clone(),
                            )
                        }
                        _ => None,
                    };
                    if grid.is_some() {
                        grid
                    } else if widen_kernel {
                        Some(cogzilla_kernel::WidenedSampler::new_uncached(
                            resample,
                            footprint_sx,
                            footprint_sy,
                        ))
                    } else {
                        None
                    }
                })
                .as_ref()
        };
        let sample = |src: &BandSource<'_>, u: f32, v: f32| -> f32 {
            if !u.is_finite() || !v.is_finite() {
                return f32::NAN;
            }
            match resample {
                ResampleMode::Nearest => {
                    let xi = (u + 0.5).floor() as i32;
                    let yi = (v + 0.5).floor() as i32;
                    if xi >= 0
                        && yi >= 0
                        && xi < ifd_w_i
                        && yi < ifd_h_i
                        && src.nodata_override.is_none()
                    {
                        src.sample_inbounds(xi as u32, yi as u32)
                    } else {
                        src.sample_checked(xi, yi)
                    }
                }
                ResampleMode::Bilinear
                | ResampleMode::Cubic
                | ResampleMode::CubicSpline
                | ResampleMode::Lanczos
                    if get_wide().is_some() =>
                {
                    get_wide().map_or(f32::NAN, |w| w.sample(src, u, v))
                }
                ResampleMode::Cubic => cogzilla_kernel::sample_cubic(src, u, v),
                ResampleMode::Bilinear => {
                    let x0_f = u.floor();
                    let y0_f = v.floor();
                    let fx = u - x0_f;
                    let fy = v - y0_f;
                    let x0 = x0_f as i32;
                    let y0 = y0_f as i32;
                    let (v00, v10, v01, v11) = if x0 >= 0
                        && y0 >= 0
                        && (x0 + 1) < ifd_w_i
                        && (y0 + 1) < ifd_h_i
                        && src.nodata_override.is_none()
                    {
                        src.sample_quad_inbounds(x0 as u32, y0 as u32)
                    } else {
                        (
                            src.sample_checked(x0, y0),
                            src.sample_checked(x0 + 1, y0),
                            src.sample_checked(x0, y0 + 1),
                            src.sample_checked(x0 + 1, y0 + 1),
                        )
                    };
                    let omx = 1.0 - fx;
                    let omy = 1.0 - fy;
                    // Fast branchless path when all 4 corners are valid (99.9% of interior pixels)
                    if !(v00.is_nan() || v10.is_nan() || v01.is_nan() || v11.is_nan()) {
                        let top = v00 * omx + v10 * fx;
                        let bot = v01 * omx + v11 * fx;
                        return top * omy + bot * fy;
                    }
                    let w_lim = src.ifd_w;
                    let h_lim = src.ifd_h;
                    let (mut sum, mut wsum, mut wtot) = (0.0f32, 0.0f32, 0.0f32);
                    for (rx, ry, val, wt) in [
                        (x0, y0, v00, omx * omy),
                        (x0 + 1, y0, v10, fx * omy),
                        (x0, y0 + 1, v01, omx * fy),
                        (x0 + 1, y0 + 1, v11, fx * fy),
                    ] {
                        if rx >= 0 && rx < w_lim && ry >= 0 && ry < h_lim {
                            wtot += wt;
                            if !val.is_nan() {
                                sum += val * wt;
                                wsum += wt;
                            }
                        }
                    }
                    if wsum >= 0.5 * wtot && wsum > 1e-6 { sum / wsum } else { f32::NAN }
                }
                other => cogzilla_kernel::sample_general_unwidened(
                    src,
                    u,
                    v,
                    footprint_sx,
                    footprint_sy,
                    other,
                ),
            }
        };
        // The widened plane of a band on separable maps (`None` otherwise: sample per pixel).
        let plane_of = |src: &BandSource<'_>| get_wide().and_then(|wd| wd.grid_plane(src));

        let apply_source_unscale = unscale
            && (crate::engine::needs_source_unscale(bands, format)
                || custom_lut.is_some_and(CustomColormap::uses_direct_lookup));

        // Helper to build BandSource for 1-based band index across single (chunky/planar) or multi-COG
        let get_band_refs = |b_1based: u16,
                             _asset_slot: usize|
         -> (Vec<Option<&cogzilla_codec::DecodedBand>>, (f32, f32)) {
            let slot = (b_1based.saturating_sub(1) as usize).min(num_cogs.saturating_sub(1));
            let so = if !apply_source_unscale {
                (1.0, 0.0)
            } else if num_cogs > 1 {
                band_scale_offsets
                    .get(slot)
                    .copied()
                    .unwrap_or((1.0, 0.0))
            } else {
                band_scale_offsets
                    .get(b_1based.saturating_sub(1) as usize)
                    .copied()
                    .unwrap_or((1.0, 0.0))
            };
            let refs = if num_cogs > 1 {
                cog_tiles[slot]
                    .iter()
                    .map(|o| o.as_ref().and_then(|t| t.bands.first()))
                    .collect()
            } else if is_single_planar {
                let b0 = b_1based.saturating_sub(1);
                if let Some(pos) = planar_bands.iter().position(|&b| b == b0) {
                    let start = pos * grid_len;
                    let end = (start + grid_len).min(cog_tiles[0].len());
                    cog_tiles[0][start..end]
                        .iter()
                        .map(|o| o.as_ref().and_then(|t| t.bands.first()))
                        .collect()
                } else {
                    vec![None; grid_len]
                }
            } else {
                let b0 = b_1based.saturating_sub(1) as usize;
                cog_tiles[0]
                    .iter()
                    .map(|o| o.as_ref().and_then(|t| t.bands.get(b0)))
                    .collect()
            };
            (refs, so)
        };

        macro_rules! make_src {
            ($refs:expr) => {{
                let (ref blocks, (sc, off)) = *$refs;
                BandSource::new(
                    ifd_width,
                    ifd_height,
                    ifd_tile_width,
                    ifd_tile_height,
                    tx0,
                    ty0,
                    grid_w,
                    grid_h,
                    blocks,
                )
                .with_nodata_override(nodata_override)
                .with_scale_offset(sc, off)
            }};
        }

        // Preserve continuous 0..255 alpha from RGBA / Gray+Alpha COGs when any intersecting
        // block has intermediate translucency (1..254). Returns `None` in O(blocks) for opaque
        // or binary 0/255 alpha tiles.
        let continuous_alpha: Option<Vec<f32>> =
            alpha_band_info.and_then(|(alpha_b1, full_opaque)| {
                let (refs, _) = get_band_refs(alpha_b1, 0);
                let alpha_src = BandSource::new(
                    ifd_width,
                    ifd_height,
                    ifd_tile_width,
                    ifd_tile_height,
                    tx0,
                    ty0,
                    grid_w,
                    grid_h,
                    &refs,
                )
                .with_scale_offset(255.0 / full_opaque, 0.0);
                if !cogzilla_kernel::has_intermediate_alpha(&alpha_src, full_opaque) {
                    return None;
                }
                if let Some(p) = plane_of(&alpha_src) {
                    Some(p[..w * h].to_vec())
                } else {
                    let mut buf = vec![0.0f32; w * h];
                    coord_map.for_each_uv(w, h, |i, u, v| {
                        let val = sample(&alpha_src, u, v);
                        buf[i] = if val.is_finite() { val } else { 0.0 };
                    });
                    Some(buf)
                }
            });

        // A terrain algorithm over band math reads it block by block as one band (on the CPU):
        // each band with the request's nodata and its own scale and offset first.
        let dem_math_blocks = match bands {
            BandRequest::Dem { math: Some(prog), .. } | BandRequest::Terrain { math: Some(prog), .. } => {
                let reads: Vec<_> = prog.used_bands.iter().map(|&b| get_band_refs(b, 0)).collect();
                let inputs: Vec<_> = reads
                    .iter()
                    .map(|(refs, (sc, off))| (&refs[..], nodata_override, (*sc != 1.0 || *off != 0.0).then_some((*sc, *off))))
                    .collect();
                Some(cogzilla_kernel::band_math_blocks(prog, &inputs))
            }
            _ => None,
        };
        let dem_math_refs: Option<Vec<Option<&cogzilla_codec::DecodedBand>>> =
            dem_math_blocks.as_ref().map(|blocks| blocks.iter().map(Option::as_ref).collect());
        macro_rules! dem_src {
            ($refs:expr) => {{
                match &dem_math_refs {
                    Some(blocks) => BandSource::new(
                        ifd_width,
                        ifd_height,
                        ifd_tile_width,
                        ifd_tile_height,
                        tx0,
                        ty0,
                        grid_w,
                        grid_h,
                        blocks,
                    ),
                    None => make_src!($refs),
                }
            }};
        }

        // Sample a (w + 2) x (h + 2) elevation ring by linearly extrapolating (u, v) 1 px past
        // each edge so Horn 3x3 gradients stay seamless across adjacent tiles (WorldCRS84Quad /
        // render_mercator_bbox) and non-square bboxes.
        let sample_dem_ring = |src: &BandSource<'_>, coord_map: &BboxCoordMap| -> Vec<f32> {
            let rw = w + 2;
            let rh = h + 2;
            let mut ring = vec![f32::NAN; rw * rh];
            coord_map.for_each_uv(w, h, |i, u, v| {
                let py = i / w;
                let px = i % w;
                ring[(py + 1) * rw + px + 1] = sample(src, u, v);
            });
            for py in 0..h {
                let rrow = (py + 1) * rw;
                let (u0, v0) = coord_map.uv(0, py, w, h);
                let (u_l, v_l) = if w >= 2 {
                    let (u1, v1) = coord_map.uv(1, py, w, h);
                    (2.0 * u0 - u1, 2.0 * v0 - v1)
                } else {
                    (u0, v0)
                };
                let (ur0, vr0) = coord_map.uv(w - 1, py, w, h);
                let (u_r, v_r) = if w >= 2 {
                    let (ur1, vr1) = coord_map.uv(w - 2, py, w, h);
                    (2.0 * ur0 - ur1, 2.0 * vr0 - vr1)
                } else {
                    (ur0, vr0)
                };
                ring[rrow] = sample(src, u_l, v_l);
                ring[rrow + w + 1] = sample(src, u_r, v_r);
            }
            let extrap_row = |src_y0: usize, src_y1: usize, dst_ry: usize, ring: &mut [f32]| {
                let dr = dst_ry * rw;
                for rx in 0..rw {
                    let px_signed = rx as isize - 1;
                    let (u0, v0, u1, v1) = if px_signed < 0 {
                        let (a0, b0) = coord_map.uv(0, src_y0, w, h);
                        let (a1, b1) = coord_map.uv(0, src_y1, w, h);
                        if w >= 2 {
                            let (c0, d0) = coord_map.uv(1, src_y0, w, h);
                            let (c1, d1) = coord_map.uv(1, src_y1, w, h);
                            (2.0 * a0 - c0, 2.0 * b0 - d0, 2.0 * a1 - c1, 2.0 * b1 - d1)
                        } else {
                            (a0, b0, a1, b1)
                        }
                    } else if (px_signed as usize) >= w {
                        let (a0, b0) = coord_map.uv(w - 1, src_y0, w, h);
                        let (a1, b1) = coord_map.uv(w - 1, src_y1, w, h);
                        if w >= 2 {
                            let (c0, d0) = coord_map.uv(w - 2, src_y0, w, h);
                            let (c1, d1) = coord_map.uv(w - 2, src_y1, w, h);
                            (2.0 * a0 - c0, 2.0 * b0 - d0, 2.0 * a1 - c1, 2.0 * b1 - d1)
                        } else {
                            (a0, b0, a1, b1)
                        }
                    } else {
                        let p = px_signed as usize;
                        let (a0, b0) = coord_map.uv(p, src_y0, w, h);
                        let (a1, b1) = coord_map.uv(p, src_y1, w, h);
                        (a0, b0, a1, b1)
                    };
                    let (u, v) = if h >= 2 {
                        (2.0 * u0 - u1, 2.0 * v0 - v1)
                    } else {
                        (u0, v0)
                    };
                    ring[dr + rx] = sample(src, u, v);
                }
            };
            extrap_row(0, 1.min(h - 1), 0, &mut ring);
            extrap_row(h - 1, h.saturating_sub(2), h + 1, &mut ring);
            ring
        };

        // The DEM grid's pixel size in its own system's units (degrees on a grid in degrees),
        // as TiTiler's slope divides by the transform's pixel size.
        let dem_dxy = || -> (f32, f32) {
            native_dem_dxy.unwrap_or(((dlx.abs()).max(1e-9) as f32, (dly.abs()).max(1e-9) as f32))
        };

        // Real RawF32 / Npy output for single-band / expression / DEM / Terrain / RGB requests
        if matches!(format, OutputFormat::RawF32 | OutputFormat::Npy) {
            if let Some((ref gpu, ref stats, primary_id, ref asset_ids)) = gpu_ctx {
                let single_band_key = |b1: u16| -> (u64, u16) {
                    if num_cogs > 1 {
                        let slot = (b1.saturating_sub(1) as usize).min(asset_ids.len().saturating_sub(1));
                        (asset_ids[slot], 0)
                    } else {
                        (primary_id, b1.saturating_sub(1))
                    }
                };
                let mb_key = |b1s: &[u16], srcs: &[BandSource<'_>]| {
                    (num_cogs == 1)
                        .then(|| {
                            crate::gpu_backend::GpuBackend::multiband_key(
                                primary_id, ifd_idx, b1s, srcs,
                            )
                        })
                        .flatten()
                };
                let coords = coord_map.as_gpu_source(w, h);
                if format == OutputFormat::RawF32
                    && let Some((sopts, desc)) = fused_stats
                    && !sopts.categorical
                    && sopts.histogram_edges.is_none()
                {
                    use crate::gpu_backend::{GpuBboxBands, GpuBboxOutput, GpuBboxResult};
                    let hist_bins = sopts.histogram_bins.clamp(1, 1000);
                    let stats_out = GpuBboxOutput::Stats {
                        resample,
                        num_bins: hist_bins,
                        hist_range: sopts.histogram_range,
                        percentiles: &sopts.percentiles,
                    };
                    let dispatch_spectral_stats = |expr, b1s: &[u16]| {
                        let ref_vecs: Vec<_> = b1s
                            .iter()
                            .enumerate()
                            .map(|(slot, &b)| get_band_refs(b, slot))
                            .collect();
                        let srcs: Vec<_> = ref_vecs.iter().map(|r| make_src!(r)).collect();
                        gpu.try_render_bbox(
                            coords,
                            &GpuBboxBands::Spectral {
                                cache_key: mb_key(b1s, &srcs),
                                srcs: &srcs,
                                expr,
                            },
                            &stats_out,
                        )
                        .and_then(GpuBboxResult::into_stats)
                    };
                    let gpu_stats: Option<crate::gpu_backend::GpuBandStats> = match bands {
                        BandRequest::Single { band_1based, .. } => {
                            let refs = get_band_refs(*band_1based, 0);
                            let src = make_src!(&refs);
                            let (cid, b0) = single_band_key(*band_1based);
                            gpu.try_render_bbox(
                                coords,
                                &GpuBboxBands::Single {
                                    cog_id: cid,
                                    ifd_idx,
                                    band0: b0,
                                    src: &src,
                                },
                                &stats_out,
                            )
                            .and_then(GpuBboxResult::into_stats)
                        }
                        BandRequest::CustomExpr { prog, .. } => {
                            let ref_vecs: Vec<_> = prog
                                .used_bands
                                .iter()
                                .map(|&b| get_band_refs(b, b.saturating_sub(1) as usize))
                                .collect();
                            let srcs: Vec<_> = ref_vecs.iter().map(|r| make_src!(r)).collect();
                            gpu.try_render_bbox(
                                coords,
                                &GpuBboxBands::Expr {
                                    cache_key: mb_key(&prog.used_bands, &srcs),
                                    srcs: &srcs,
                                    prog,
                                },
                                &stats_out,
                            )
                            .and_then(GpuBboxResult::into_stats)
                        }
                        BandRequest::Dem {
                            band_1based,
                            algorithm,
                            math: None,
                            ..
                        } => {
                            let refs = get_band_refs(*band_1based, 0);
                            let src = make_src!(&refs);
                            let (dx_m, dy_m) = dem_dxy();
                            let (cid, b0) = single_band_key(*band_1based);
                            gpu.try_render_bbox(
                                coords,
                                &GpuBboxBands::Dem {
                                    cog_id: cid,
                                    ifd_idx,
                                    band0: b0,
                                    src: &src,
                                    algorithm: *algorithm,
                                    dx_m,
                                    dy_m,
                                },
                                &stats_out,
                            )
                            .and_then(GpuBboxResult::into_stats)
                        }
                        BandRequest::Spectral1 {
                            expr, band_1based, ..
                        } => dispatch_spectral_stats(*expr, std::slice::from_ref(band_1based)),
                        BandRequest::Spectral2 {
                            expr, bands_1based, ..
                        } => dispatch_spectral_stats(*expr, bands_1based),
                        BandRequest::Spectral3 {
                            expr, bands_1based, ..
                        } => dispatch_spectral_stats(*expr, bands_1based),
                        _ => None,
                    };
                    if let Some(gs) = gpu_stats {
                        stats
                            .gpu_dispatches
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        return BboxRenderOutput::Stats(gpu_band_stats_to_json(
                            &gs,
                            w * h,
                            desc,
                            sopts,
                            sopts.hist_dtype.unwrap_or(HistDtype::Float32),
                            None,
                        ));
                    }
                }
                let f32_out = crate::gpu_backend::GpuBboxOutput::F32 { resample };
                let dispatch_spectral_f32 = |expr, b1s: &[u16]| {
                    let ref_vecs: Vec<_> = b1s
                        .iter()
                        .enumerate()
                        .map(|(slot, &b)| get_band_refs(b, slot))
                        .collect();
                    let srcs: Vec<_> = ref_vecs.iter().map(|r| make_src!(r)).collect();
                    gpu.try_render_bbox(
                        coords,
                        &crate::gpu_backend::GpuBboxBands::Spectral {
                            cache_key: mb_key(b1s, &srcs),
                            srcs: &srcs,
                            expr,
                        },
                        &f32_out,
                    )
                    .and_then(crate::gpu_backend::GpuBboxResult::into_bytes)
                };
                let gpu_raw: Option<Bytes> = match bands {
                    BandRequest::Single { band_1based, .. } => {
                        let refs = get_band_refs(*band_1based, 0);
                        let src = make_src!(&refs);
                        let (cid, b0) = single_band_key(*band_1based);
                        gpu.try_render_bbox(
                            coords,
                            &crate::gpu_backend::GpuBboxBands::Single {
                                cog_id: cid,
                                ifd_idx,
                                band0: b0,
                                src: &src,
                            },
                            &f32_out,
                        )
                        .and_then(crate::gpu_backend::GpuBboxResult::into_bytes)
                    }
                    BandRequest::Dem {
                        band_1based,
                        algorithm,
                        math: None,
                        ..
                    } => {
                        let refs = get_band_refs(*band_1based, 0);
                        let src = make_src!(&refs);
                        let (dx_m, dy_m) = dem_dxy();
                        let (cid, b0) = single_band_key(*band_1based);
                        gpu.try_render_bbox(
                            coords,
                            &crate::gpu_backend::GpuBboxBands::Dem {
                                cog_id: cid,
                                ifd_idx,
                                band0: b0,
                                src: &src,
                                algorithm: *algorithm,
                                dx_m,
                                dy_m,
                            },
                            &f32_out,
                        )
                        .and_then(crate::gpu_backend::GpuBboxResult::into_bytes)
                    }
                    BandRequest::CustomExpr { prog, .. } => {
                        let ref_vecs: Vec<_> = prog
                            .used_bands
                            .iter()
                            .map(|&b| get_band_refs(b, b.saturating_sub(1) as usize))
                            .collect();
                        let srcs: Vec<_> = ref_vecs.iter().map(|r| make_src!(r)).collect();
                        gpu.try_render_bbox(
                            coords,
                            &crate::gpu_backend::GpuBboxBands::Expr {
                                cache_key: mb_key(&prog.used_bands, &srcs),
                                srcs: &srcs,
                                prog,
                            },
                            &f32_out,
                        )
                        .and_then(crate::gpu_backend::GpuBboxResult::into_bytes)
                    }
                    BandRequest::Spectral1 {
                        expr, band_1based, ..
                    } => dispatch_spectral_f32(*expr, std::slice::from_ref(band_1based)),
                    BandRequest::Spectral2 {
                        expr, bands_1based, ..
                    } => dispatch_spectral_f32(*expr, bands_1based),
                    BandRequest::Spectral3 {
                        expr, bands_1based, ..
                    } => dispatch_spectral_f32(*expr, bands_1based),
                    BandRequest::Rgb { bands_1based, .. } => {
                        let r0 = get_band_refs(bands_1based[0], 0);
                        let r1 = get_band_refs(bands_1based[1], 1);
                        let r2 = get_band_refs(bands_1based[2], 2);
                        let srcs = [make_src!(&r0), make_src!(&r1), make_src!(&r2)];
                        gpu.try_render_bbox(
                            coords,
                            &crate::gpu_backend::GpuBboxBands::MultiBand {
                                cache_key: mb_key(bands_1based, &srcs),
                                srcs: &srcs,
                            },
                            &f32_out,
                        )
                        .and_then(crate::gpu_backend::GpuBboxResult::into_bytes)
                    }
                    BandRequest::MultiBand { bands_1based } => {
                        if bands_1based.is_empty() || bands_1based.len() > 8 {
                            None
                        } else {
                            let ref_vecs: Vec<_> = bands_1based
                                .iter()
                                .enumerate()
                                .map(|(slot, &b)| get_band_refs(b, slot))
                                .collect();
                            let srcs: Vec<_> = ref_vecs.iter().map(|r| make_src!(r)).collect();
                            gpu.try_render_bbox(
                                coords,
                                &crate::gpu_backend::GpuBboxBands::MultiBand {
                                    cache_key: mb_key(bands_1based, &srcs),
                                    srcs: &srcs,
                                },
                                &f32_out,
                            )
                            .and_then(crate::gpu_backend::GpuBboxResult::into_bytes)
                        }
                    }
                    BandRequest::CustomExprRgb { progs, .. } => {
                        let mut unique_bands: Vec<u16> = Vec::new();
                        for prog in progs {
                            for &b in &prog.used_bands {
                                if !unique_bands.contains(&b) {
                                    unique_bands.push(b);
                                }
                            }
                        }
                        if unique_bands.is_empty() || unique_bands.len() > 8 {
                            None
                        } else {
                            let ref_vecs: Vec<_> = unique_bands
                                .iter()
                                .map(|&b| get_band_refs(b, b.saturating_sub(1) as usize))
                                .collect();
                            let srcs: Vec<_> = ref_vecs.iter().map(|r| make_src!(r)).collect();
                            gpu.try_render_bbox(
                                coords,
                                &crate::gpu_backend::GpuBboxBands::ExprRgb {
                                    cache_key: mb_key(&unique_bands, &srcs),
                                    unique_bands: &unique_bands,
                                    srcs: &srcs,
                                    progs,
                                },
                                &f32_out,
                            )
                            .and_then(crate::gpu_backend::GpuBboxResult::into_bytes)
                        }
                    }
                    _ => None,
                };
                if let Some(raw) = gpu_raw {
                    stats
                        .gpu_dispatches
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if format == OutputFormat::Npy {
                        let nb = match bands {
                            BandRequest::Rgb { .. }
                            | BandRequest::SarRgb { .. }
                            | BandRequest::Terrain { .. }
                            | BandRequest::CustomExprRgb { .. } => 3,
                            BandRequest::MultiBand { bands_1based } => bands_1based.len().max(1),
                            BandRequest::CustomExprMulti { progs } => progs.len().max(1),
                            _ => 1,
                        };
                        return BboxRenderOutput::Bytes(
                            cogzilla_kernel::wrap_raw_f32_as_npy_with_alpha_mask(
                                raw,
                                nb,
                                h as u32,
                                w as u32,
                                continuous_alpha.as_deref(),
                                return_mask,
                            ),
                        );
                    }
                    return BboxRenderOutput::Bytes(raw);
                }
            }

            let mut out = BytesMut::with_capacity(w * h * 4);
            match bands {
                BandRequest::Dem {
                    band_1based,
                    algorithm,
                    ..
                } => {
                    let refs = get_band_refs(*band_1based, 0);
                    let src = dem_src!(&refs);
                    let ring = sample_dem_ring(&src, &coord_map);
                    let rw = w + 2;
                    let (dx_m, dy_m) = dem_dxy();
                    let (inv_dx, inv_dy) = (1.0 / dx_m, 1.0 / dy_m);
                    for py in 0..h {
                        let r0 = py * rw;
                        let r1 = (py + 1) * rw;
                        let r2 = (py + 2) * rw;
                        for px in 0..w {
                            let val = cogzilla_kernel::eval_dem_stencil(ring[r0 + px + 1], ring[r1 + px], ring[r1 + px + 1], ring[r1 + px + 2], ring[r2 + px + 1], inv_dx, inv_dy, *algorithm);
                            out.put_slice(&val.to_le_bytes());
                        }
                    }
                }
                BandRequest::Single { band_1based, .. } => {
                    let refs = get_band_refs(*band_1based, 0);
                    let src = make_src!(&refs);
                    if let Some(p) = plane_of(&src) {
                        for v in &p[..w * h] {
                            out.put_slice(&v.to_le_bytes());
                        }
                    } else {
                        coord_map.for_each_uv(w, h, |_, u, v| {
                            out.put_slice(&sample(&src, u, v).to_le_bytes());
                        });
                    }
                }
                BandRequest::Terrain {
                    band_1based,
                    algorithm,
                    ..
                } => {
                    let refs = get_band_refs(*band_1based, 0);
                    let src = dem_src!(&refs);
                    let mut g_plane = Vec::with_capacity(w * h * 4);
                    let mut b_plane = Vec::with_capacity(w * h * 4);
                    coord_map.for_each_uv(w, h, |_, u, v| {
                        let elev = sample(&src, u, v);
                        if elev.is_nan() {
                            out.put_slice(&f32::NAN.to_le_bytes());
                            g_plane.extend_from_slice(&f32::NAN.to_le_bytes());
                            b_plane.extend_from_slice(&f32::NAN.to_le_bytes());
                        } else {
                            let [r, g, b] = cogzilla_kernel::encode_terrain_pixel(*algorithm, elev);
                            out.put_slice(&(r as f32).to_le_bytes());
                            g_plane.extend_from_slice(&(g as f32).to_le_bytes());
                            b_plane.extend_from_slice(&(b as f32).to_le_bytes());
                        }
                    });
                    out.put_slice(&g_plane);
                    out.put_slice(&b_plane);
                }
                BandRequest::CustomExpr { prog, .. } => {
                    let ref_vecs: Vec<_> = prog
                        .used_bands
                        .iter()
                        .map(|&b| get_band_refs(b, b.saturating_sub(1) as usize))
                        .collect();
                    let srcs: Vec<_> = ref_vecs.iter().map(|r| make_src!(r)).collect();
                    let mut vals = [0.0f32; 16];
                    let nb = srcs.len().min(16);
                    coord_map.for_each_uv(w, h, |_, u, v| {
                        if u.is_nan() {
                            out.put_slice(&f32::NAN.to_le_bytes());
                            return;
                        }
                        for k in 0..nb {
                            vals[k] = sample(&srcs[k], u, v);
                        }
                        out.put_slice(&prog.eval(&vals[..nb]).to_le_bytes());
                    });
                }
                BandRequest::CustomExprRgb { progs, .. } => {
                    for prog in progs {
                        let ref_vecs: Vec<_> = prog
                            .used_bands
                            .iter()
                            .map(|&b| get_band_refs(b, b.saturating_sub(1) as usize))
                            .collect();
                        let srcs: Vec<_> = ref_vecs.iter().map(|r| make_src!(r)).collect();
                        let mut vals = [0.0f32; 16];
                        let nb = srcs.len().min(16);
                        coord_map.for_each_uv(w, h, |_, u, v| {
                            if u.is_nan() {
                                out.put_slice(&f32::NAN.to_le_bytes());
                                return;
                            }
                            for k in 0..nb {
                                vals[k] = sample(&srcs[k], u, v);
                            }
                            out.put_slice(&prog.eval(&vals[..nb]).to_le_bytes());
                        });
                    }
                }
                BandRequest::Spectral1 {
                    expr, band_1based, ..
                } => {
                    let refs = get_band_refs(*band_1based, 0);
                    let src = make_src!(&refs);
                    coord_map.for_each_uv(w, h, |_, u, v| {
                        out.put_slice(&expr.eval1(sample(&src, u, v)).to_le_bytes());
                    });
                }
                BandRequest::Spectral2 {
                    expr, bands_1based, ..
                } => {
                    let r0 = get_band_refs(bands_1based[0], 0);
                    let r1 = get_band_refs(bands_1based[1], 1);
                    let s0 = make_src!(&r0);
                    let s1 = make_src!(&r1);
                    coord_map.for_each_uv(w, h, |_, u, v| {
                        let a = sample(&s0, u, v);
                        let b = sample(&s1, u, v);
                        let val = if a.is_nan() || b.is_nan() {
                            f32::NAN
                        } else {
                            expr.eval2(a, b)
                        };
                        out.put_slice(&val.to_le_bytes());
                    });
                }
                BandRequest::Spectral3 {
                    expr, bands_1based, ..
                } => {
                    let r0 = get_band_refs(bands_1based[0], 0);
                    let r1 = get_band_refs(bands_1based[1], 1);
                    let r2 = get_band_refs(bands_1based[2], 2);
                    let s0 = make_src!(&r0);
                    let s1 = make_src!(&r1);
                    let s2 = make_src!(&r2);
                    coord_map.for_each_uv(w, h, |_, u, v| {
                        let a = sample(&s0, u, v);
                        let b = sample(&s1, u, v);
                        let c = sample(&s2, u, v);
                        let val = if a.is_nan() || b.is_nan() || c.is_nan() {
                            f32::NAN
                        } else {
                            expr.eval3(a, b, c)
                        };
                        out.put_slice(&val.to_le_bytes());
                    });
                }
                BandRequest::Rgb { bands_1based, .. } => {
                    let r0 = get_band_refs(bands_1based[0], 0);
                    let r1 = get_band_refs(bands_1based[1], 1);
                    let r2 = get_band_refs(bands_1based[2], 2);
                    let srcs = [make_src!(&r0), make_src!(&r1), make_src!(&r2)];
                    out.reserve(w * h * 8);
                    for s in &srcs {
                        if let Some(p) = plane_of(s) {
                            for v in &p[..w * h] {
                                out.put_slice(&v.to_le_bytes());
                            }
                            continue;
                        }
                        coord_map.for_each_uv(w, h, |_, u, v| {
                            out.put_slice(&sample(s, u, v).to_le_bytes());
                        });
                    }
                }
                BandRequest::SarRgb { bands_1based, .. } => {
                    let r0 = get_band_refs(bands_1based[0], 0);
                    let r1 = get_band_refs(bands_1based[1], 1);
                    let s0 = make_src!(&r0);
                    let s1 = make_src!(&r1);
                    let mut vh_plane = Vec::with_capacity(w * h * 4);
                    let mut diff_plane = Vec::with_capacity(w * h * 4);
                    out.reserve(w * h * 8);
                    coord_map.for_each_uv(w, h, |_, u, v| {
                        let vv = fast_db_from_power(sample(&s0, u, v));
                        let vh = fast_db_from_power(sample(&s1, u, v));
                        out.put_slice(&vv.to_le_bytes());
                        vh_plane.extend_from_slice(&vh.to_le_bytes());
                        diff_plane.extend_from_slice(&(vv - vh).to_le_bytes());
                    });
                    out.put_slice(&vh_plane);
                    out.put_slice(&diff_plane);
                }
                BandRequest::CustomExprMulti { progs } => {
                    out.reserve(w * h * 4 * progs.len().saturating_sub(1));
                    for prog in progs {
                        let ref_vecs: Vec<_> = prog
                            .used_bands
                            .iter()
                            .map(|&b| get_band_refs(b, b.saturating_sub(1) as usize))
                            .collect();
                        let srcs: Vec<_> = ref_vecs.iter().map(|r| make_src!(r)).collect();
                        let mut vals = [0.0f32; 16];
                        let nb = srcs.len().min(16);
                        coord_map.for_each_uv(w, h, |_, u, v| {
                            if u.is_nan() {
                                out.put_slice(&f32::NAN.to_le_bytes());
                                return;
                            }
                            for k in 0..nb {
                                vals[k] = sample(&srcs[k], u, v);
                            }
                            out.put_slice(&prog.eval(&vals[..nb]).to_le_bytes());
                        });
                    }
                }
                BandRequest::MultiBand { bands_1based } => {
                    out.reserve(w * h * 4 * bands_1based.len().saturating_sub(1));
                    for (slot, &b1) in bands_1based.iter().enumerate() {
                        let r = get_band_refs(b1, slot);
                        let s = make_src!(&r);
                        if let Some(p) = plane_of(&s) {
                            for v in &p[..w * h] {
                                out.put_slice(&v.to_le_bytes());
                            }
                        } else {
                            coord_map.for_each_uv(w, h, |_, u, v| {
                                out.put_slice(&sample(&s, u, v).to_le_bytes());
                            });
                        }
                    }
                }
            }
            let raw = out.freeze();
            if format == OutputFormat::Npy {
                let nb = match bands {
                    BandRequest::Rgb { .. }
                    | BandRequest::SarRgb { .. }
                    | BandRequest::CustomExprRgb { .. }
                    | BandRequest::Terrain { .. } => 3,
                    BandRequest::MultiBand { bands_1based } => bands_1based.len().max(1),
                    BandRequest::CustomExprMulti { progs } => progs.len().max(1),
                    _ => 1,
                };
                return BboxRenderOutput::Bytes(
                    cogzilla_kernel::wrap_raw_f32_as_npy_with_alpha_mask(
                        raw,
                        nb,
                        h as u32,
                        w as u32,
                        continuous_alpha.as_deref(),
                        return_mask,
                    ),
                );
            }
            return BboxRenderOutput::Bytes(raw);
        }

        let cf_single_lut: Option<[u8; 256]> = if custom_lut.is_none()
            && matches!(
                bands,
                BandRequest::Single { .. }
                    | BandRequest::Dem { .. }
                    | BandRequest::CustomExpr { .. }
                    | BandRequest::Spectral1 { .. }
                    | BandRequest::Spectral2 { .. }
                    | BandRequest::Spectral3 { .. }
            )
        {
            color_formula.map(|cf| {
                let mut t = [0u8; 256];
                for i in 0..256 {
                    let mut px = [i as u8, i as u8, i as u8, 255];
                    cf.apply_buffer_in_place(&mut px, 4);
                    t[i] = px[0];
                }
                t
            })
        } else {
            None
        };
        let map_color = |v: f32, rmin: f32, inv: f32, default_cmap: ColormapName| -> [u8; 4] {
            if let Some(cmap) = custom_lut {
                cmap.lookup(v, rmin, inv)
            } else {
                let mut idx = ((v - rmin) * inv).clamp(0.0, 255.0) as usize;
                if let Some(ref cf_lut) = cf_single_lut {
                    idx = cf_lut[idx] as usize;
                }
                default_cmap.lut().rgba_u32[idx].to_le_bytes()
            }
        };

        if continuous_alpha.is_none()
            && matches!(
                format,
                OutputFormat::Png
                    | OutputFormat::Webp { .. }
                    | OutputFormat::Jpeg { .. }
                    | OutputFormat::RawRgba
            )
            && let Some((ref gpu, ref stats, primary_id, ref asset_ids)) = gpu_ctx
        {
            let single_band_key = |b1: u16| -> (u64, u16) {
                if num_cogs > 1 {
                    let slot = (b1.saturating_sub(1) as usize).min(asset_ids.len().saturating_sub(1));
                    (asset_ids[slot], 0)
                } else {
                    (primary_id, b1.saturating_sub(1))
                }
            };
            let mb_key = |b1s: &[u16], srcs: &[BandSource<'_>]| {
                (num_cogs == 1)
                    .then(|| {
                        crate::gpu_backend::GpuBackend::multiband_key(
                            primary_id, ifd_idx, b1s, srcs,
                        )
                    })
                    .flatten()
            };
            let coords = coord_map.as_gpu_source(w, h);
            let dispatch_spectral_rgba = |expr, b1s: &[u16], rescale: (f32, f32), colormap| {
                crate::gpu_backend::GpuBackend::resolve_styled_rgb_lut(
                    colormap,
                    custom_lut,
                    color_formula,
                )
                .and_then(|rgb_lut| {
                    let ref_vecs: Vec<_> = b1s
                        .iter()
                        .enumerate()
                        .map(|(slot, &b)| get_band_refs(b, slot))
                        .collect();
                    let srcs: Vec<_> = ref_vecs.iter().map(|r| make_src!(r)).collect();
                    let style = cogzilla_kernel::SingleBandStyle {
                        rescale_min: rescale.0,
                        rescale_max: rescale.1,
                        colormap,
                        resample,
                        format,
                    };
                    gpu.try_render_bbox(
                        coords,
                        &crate::gpu_backend::GpuBboxBands::Spectral {
                            cache_key: mb_key(b1s, &srcs),
                            srcs: &srcs,
                            expr,
                        },
                        &crate::gpu_backend::GpuBboxOutput::StyledLut {
                            style: &style,
                            rgb_lut: &rgb_lut,
                            return_mask,
                        },
                    )
                    .and_then(crate::gpu_backend::GpuBboxResult::into_bytes)
                })
            };
            let gpu_out: Option<Bytes> = match bands {
                BandRequest::Single {
                    band_1based,
                    rescale,
                    colormap,
                } => crate::gpu_backend::GpuBackend::resolve_styled_rgb_lut(
                    *colormap,
                    custom_lut,
                    color_formula,
                )
                .and_then(|rgb_lut| {
                    let refs = get_band_refs(*band_1based, 0);
                    let src = make_src!(&refs);
                    let style = cogzilla_kernel::SingleBandStyle {
                        rescale_min: rescale.0,
                        rescale_max: rescale.1,
                        colormap: *colormap,
                        resample,
                        format,
                    };
                    let (cid, b0) = single_band_key(*band_1based);
                    gpu.try_render_bbox(
                        coords,
                        &crate::gpu_backend::GpuBboxBands::Single {
                            cog_id: cid,
                            ifd_idx,
                            band0: b0,
                            src: &src,
                        },
                        &crate::gpu_backend::GpuBboxOutput::StyledLut {
                            style: &style,
                            rgb_lut: &rgb_lut,
                            return_mask,
                        },
                    )
                    .and_then(crate::gpu_backend::GpuBboxResult::into_bytes)
                }),
                BandRequest::Dem {
                    band_1based,
                    algorithm,
                    rescale,
                    colormap,
                    math: None,
                } => crate::gpu_backend::GpuBackend::resolve_styled_rgb_lut(
                    *colormap,
                    custom_lut,
                    color_formula,
                )
                .and_then(|rgb_lut| {
                    let refs = get_band_refs(*band_1based, 0);
                    let src = make_src!(&refs);
                    let (dx_m, dy_m) = dem_dxy();
                    let style = cogzilla_kernel::SingleBandStyle {
                        rescale_min: rescale.0,
                        rescale_max: rescale.1,
                        colormap: *colormap,
                        resample,
                        format,
                    };
                    let (cid, b0) = single_band_key(*band_1based);
                    gpu.try_render_bbox(
                        coords,
                        &crate::gpu_backend::GpuBboxBands::Dem {
                            cog_id: cid,
                            ifd_idx,
                            band0: b0,
                            src: &src,
                            algorithm: *algorithm,
                            dx_m,
                            dy_m,
                        },
                        &crate::gpu_backend::GpuBboxOutput::StyledLut {
                            style: &style,
                            rgb_lut: &rgb_lut,
                            return_mask,
                        },
                    )
                    .and_then(crate::gpu_backend::GpuBboxResult::into_bytes)
                }),
                BandRequest::Terrain {
                    band_1based,
                    algorithm,
                    math: None,
                } if custom_lut.is_none() && color_formula.is_none() => {
                    let refs = get_band_refs(*band_1based, 0);
                    let src = make_src!(&refs);
                    let (cid, b0) = single_band_key(*band_1based);
                    gpu.try_render_bbox(
                        coords,
                        &crate::gpu_backend::GpuBboxBands::Terrain {
                            cog_id: cid,
                            ifd_idx,
                            band0: b0,
                            src: &src,
                            algorithm: *algorithm,
                        },
                        &crate::gpu_backend::GpuBboxOutput::StyledTerrain {
                            resample,
                            format,
                            return_mask,
                        },
                    )
                    .and_then(crate::gpu_backend::GpuBboxResult::into_bytes)
                }
                BandRequest::CustomExpr {
                    prog,
                    rescale,
                    colormap,
                } => crate::gpu_backend::GpuBackend::resolve_styled_rgb_lut(
                    *colormap,
                    custom_lut,
                    color_formula,
                )
                .and_then(|rgb_lut| {
                    let ref_vecs: Vec<_> = prog
                        .used_bands
                        .iter()
                        .map(|&b| get_band_refs(b, b.saturating_sub(1) as usize))
                        .collect();
                    let srcs: Vec<_> = ref_vecs.iter().map(|r| make_src!(r)).collect();
                    let style = cogzilla_kernel::SingleBandStyle {
                        rescale_min: rescale.0,
                        rescale_max: rescale.1,
                        colormap: *colormap,
                        resample,
                        format,
                    };
                    gpu.try_render_bbox(
                        coords,
                        &crate::gpu_backend::GpuBboxBands::Expr {
                            cache_key: mb_key(&prog.used_bands, &srcs),
                            srcs: &srcs,
                            prog,
                        },
                        &crate::gpu_backend::GpuBboxOutput::StyledLut {
                            style: &style,
                            rgb_lut: &rgb_lut,
                            return_mask,
                        },
                    )
                    .and_then(crate::gpu_backend::GpuBboxResult::into_bytes)
                }),
                BandRequest::CustomExprRgb {
                    progs,
                    rescale_min,
                    rescale_max,
                } if custom_lut.is_none() => {
                    let mut unique_bands: Vec<u16> = Vec::new();
                    for prog in progs {
                        for &b in &prog.used_bands {
                            if !unique_bands.contains(&b) {
                                unique_bands.push(b);
                            }
                        }
                    }
                    if unique_bands.is_empty() || unique_bands.len() > 8 {
                        None
                    } else {
                        let ref_vecs: Vec<_> = unique_bands
                            .iter()
                            .map(|&b| get_band_refs(b, b.saturating_sub(1) as usize))
                            .collect();
                        let srcs: Vec<_> = ref_vecs.iter().map(|r| make_src!(r)).collect();
                        let style = cogzilla_kernel::RgbStyle {
                            rescale_min: *rescale_min,
                            rescale_max: *rescale_max,
                            resample,
                            format,
                        };
                        gpu.try_render_bbox(
                            coords,
                            &crate::gpu_backend::GpuBboxBands::ExprRgb {
                                cache_key: mb_key(&unique_bands, &srcs),
                                unique_bands: &unique_bands,
                                srcs: &srcs,
                                progs,
                            },
                            &crate::gpu_backend::GpuBboxOutput::StyledRgb {
                                style: &style,
                                color_formula,
                                return_mask,
                            },
                        )
                        .and_then(crate::gpu_backend::GpuBboxResult::into_bytes)
                    }
                }
                BandRequest::Rgb {
                    bands_1based,
                    rescale_min,
                    rescale_max,
                } if custom_lut.is_none() => {
                    let r0 = get_band_refs(bands_1based[0], 0);
                    let r1 = get_band_refs(bands_1based[1], 1);
                    let r2 = get_band_refs(bands_1based[2], 2);
                    let srcs = [make_src!(&r0), make_src!(&r1), make_src!(&r2)];
                    let style = cogzilla_kernel::RgbStyle {
                        rescale_min: *rescale_min,
                        rescale_max: *rescale_max,
                        resample,
                        format,
                    };
                    gpu.try_render_bbox(
                        coords,
                        &crate::gpu_backend::GpuBboxBands::Rgb {
                            cache_key: mb_key(bands_1based, &srcs),
                            srcs: &srcs,
                        },
                        &crate::gpu_backend::GpuBboxOutput::StyledRgb {
                            style: &style,
                            color_formula,
                            return_mask,
                        },
                    )
                    .and_then(crate::gpu_backend::GpuBboxResult::into_bytes)
                }
                BandRequest::SarRgb {
                    bands_1based,
                    rescale_min,
                    rescale_max,
                } if custom_lut.is_none() => {
                    let r0 = get_band_refs(bands_1based[0], 0);
                    let r1 = get_band_refs(bands_1based[1], 1);
                    let srcs = [make_src!(&r0), make_src!(&r1)];
                    let style = cogzilla_kernel::RgbStyle {
                        rescale_min: *rescale_min,
                        rescale_max: *rescale_max,
                        resample,
                        format,
                    };
                    gpu.try_render_bbox(
                        coords,
                        &crate::gpu_backend::GpuBboxBands::SarRgb {
                            cache_key: mb_key(bands_1based, &srcs),
                            srcs: &srcs,
                        },
                        &crate::gpu_backend::GpuBboxOutput::StyledRgb {
                            style: &style,
                            color_formula,
                            return_mask,
                        },
                    )
                    .and_then(crate::gpu_backend::GpuBboxResult::into_bytes)
                }
                BandRequest::Spectral1 {
                    expr,
                    band_1based,
                    rescale,
                    colormap,
                } => dispatch_spectral_rgba(
                    *expr,
                    std::slice::from_ref(band_1based),
                    *rescale,
                    *colormap,
                ),
                BandRequest::Spectral2 {
                    expr,
                    bands_1based,
                    rescale,
                    colormap,
                } => dispatch_spectral_rgba(*expr, bands_1based, *rescale, *colormap),
                BandRequest::Spectral3 {
                    expr,
                    bands_1based,
                    rescale,
                    colormap,
                } => dispatch_spectral_rgba(*expr, bands_1based, *rescale, *colormap),
                _ => None,
            };
            if let Some(bytes) = gpu_out {
                stats
                    .gpu_dispatches
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return BboxRenderOutput::Bytes(bytes);
            }
        }

        let mut rgba = vec![0u8; w * h * 4];
        match bands {
            BandRequest::Single {
                band_1based,
                rescale,
                colormap,
            } => {
                let refs = get_band_refs(*band_1based, 0);
                let src = make_src!(&refs);
                let inv = 255.0 / cogzilla_kernel::safe_rescale_denom(rescale.0, rescale.1);
                if let Some(p) = plane_of(&src) {
                    for (i, &v) in p[..w * h].iter().enumerate() {
                        if !v.is_nan() {
                            let [r, g, b, a] = map_color(v, rescale.0, inv, *colormap);
                            rgba[i * 4] = r;
                            rgba[i * 4 + 1] = g;
                            rgba[i * 4 + 2] = b;
                            rgba[i * 4 + 3] = a;
                        }
                    }
                } else {
                    coord_map.for_each_uv(w, h, |i, u, v| {
                        let val = sample(&src, u, v);
                        if !val.is_nan() {
                            let [r, g, b, a] = map_color(val, rescale.0, inv, *colormap);
                            rgba[i * 4] = r;
                            rgba[i * 4 + 1] = g;
                            rgba[i * 4 + 2] = b;
                            rgba[i * 4 + 3] = a;
                        }
                    });
                }
            }
            BandRequest::Terrain {
                band_1based,
                algorithm,
                ..
            } => {
                let refs = get_band_refs(*band_1based, 0);
                let src = dem_src!(&refs);
                coord_map.for_each_uv(w, h, |i, u, v| {
                    let val = sample(&src, u, v);
                    if !val.is_nan() {
                        let [r, g, b] = cogzilla_kernel::encode_terrain_pixel(*algorithm, val);
                        rgba[i * 4] = r;
                        rgba[i * 4 + 1] = g;
                        rgba[i * 4 + 2] = b;
                        rgba[i * 4 + 3] = 255;
                    }
                });
            }
            BandRequest::Dem {
                band_1based,
                algorithm,
                rescale,
                colormap,
                ..
            } => {
                let refs = get_band_refs(*band_1based, 0);
                let src = dem_src!(&refs);
                let inv = 255.0 / cogzilla_kernel::safe_rescale_denom(rescale.0, rescale.1);
                let ring = sample_dem_ring(&src, &coord_map);
                let rw = w + 2;
                let (dx_m, dy_m) = dem_dxy();
                let (inv_dx, inv_dy) = (1.0 / dx_m, 1.0 / dy_m);
                for py in 0..h {
                    let r0 = py * rw;
                    let r1 = (py + 1) * rw;
                    let r2 = (py + 2) * rw;
                    for px in 0..w {
                        let v = cogzilla_kernel::eval_dem_stencil(ring[r0 + px + 1], ring[r1 + px], ring[r1 + px + 1], ring[r1 + px + 2], ring[r2 + px + 1], inv_dx, inv_dy, *algorithm);
                        if !v.is_nan() {
                            let i = py * w + px;
                            let [r, g, b, a] = map_color(v, rescale.0, inv, *colormap);
                            rgba[i * 4] = r;
                            rgba[i * 4 + 1] = g;
                            rgba[i * 4 + 2] = b;
                            rgba[i * 4 + 3] = a;
                        }
                    }
                }
            }
            BandRequest::Rgb {
                bands_1based,
                rescale_min,
                rescale_max,
            } => {
                let r0 = get_band_refs(bands_1based[0], 0);
                let r1 = get_band_refs(bands_1based[1], 1);
                let r2 = get_band_refs(bands_1based[2], 2);
                let s0 = make_src!(&r0);
                let s1 = make_src!(&r1);
                let s2 = make_src!(&r2);
                let inv = [
                    255.0 / cogzilla_kernel::safe_rescale_denom(rescale_min[0], rescale_max[0]),
                    255.0 / cogzilla_kernel::safe_rescale_denom(rescale_min[1], rescale_max[1]),
                    255.0 / cogzilla_kernel::safe_rescale_denom(rescale_min[2], rescale_max[2]),
                ];
                let fill = [s0.nodata_fill(), s1.nodata_fill(), s2.nodata_fill()];
                if let (Some(p0), Some(p1), Some(p2)) = (plane_of(&s0), plane_of(&s1), plane_of(&s2)) {
                    let n = w * h;
                    let px = rgba[..n * 4].chunks_exact_mut(4);
                    for (((o, &r), &g), &b) in px.zip(&p0[..n]).zip(&p1[..n]).zip(&p2[..n]) {
                        if !(r.is_nan() && g.is_nan() && b.is_nan()) {
                            let r = if r.is_nan() { fill[0] } else { r };
                            let g = if g.is_nan() { fill[1] } else { g };
                            let b = if b.is_nan() { fill[2] } else { b };
                            o[0] = ((r - rescale_min[0]) * inv[0]).clamp(0.0, 255.0) as u8;
                            o[1] = ((g - rescale_min[1]) * inv[1]).clamp(0.0, 255.0) as u8;
                            o[2] = ((b - rescale_min[2]) * inv[2]).clamp(0.0, 255.0) as u8;
                            o[3] = 255;
                        }
                    }
                } else {
                    coord_map.for_each_uv(w, h, |i, u, v| {
                        let r = sample(&s0, u, v);
                        let g = sample(&s1, u, v);
                        let b = sample(&s2, u, v);
                        if !(r.is_nan() && g.is_nan() && b.is_nan()) {
                            let r = if r.is_nan() { fill[0] } else { r };
                            let g = if g.is_nan() { fill[1] } else { g };
                            let b = if b.is_nan() { fill[2] } else { b };
                            rgba[i * 4] = ((r - rescale_min[0]) * inv[0]).clamp(0.0, 255.0) as u8;
                            rgba[i * 4 + 1] = ((g - rescale_min[1]) * inv[1]).clamp(0.0, 255.0) as u8;
                            rgba[i * 4 + 2] = ((b - rescale_min[2]) * inv[2]).clamp(0.0, 255.0) as u8;
                            rgba[i * 4 + 3] = 255;
                        }
                    });
                }
            }
            BandRequest::SarRgb {
                bands_1based,
                rescale_min,
                rescale_max,
            } => {
                let r0 = get_band_refs(bands_1based[0], 0);
                let r1 = get_band_refs(bands_1based[1], 1);
                let s0 = make_src!(&r0);
                let s1 = make_src!(&r1);
                let inv = [
                    255.0 / cogzilla_kernel::safe_rescale_denom(rescale_min[0], rescale_max[0]),
                    255.0 / cogzilla_kernel::safe_rescale_denom(rescale_min[1], rescale_max[1]),
                    255.0 / cogzilla_kernel::safe_rescale_denom(rescale_min[2], rescale_max[2]),
                ];
                coord_map.for_each_uv(w, h, |i, u, v| {
                    let vv = fast_db_from_power(sample(&s0, u, v));
                    let vh = fast_db_from_power(sample(&s1, u, v));
                    if !vv.is_nan() && !vh.is_nan() {
                        rgba[i * 4] = ((vv - rescale_min[0]) * inv[0]).clamp(0.0, 255.0) as u8;
                        rgba[i * 4 + 1] = ((vh - rescale_min[1]) * inv[1]).clamp(0.0, 255.0) as u8;
                        rgba[i * 4 + 2] =
                            (((vv - vh) - rescale_min[2]) * inv[2]).clamp(0.0, 255.0) as u8;
                        rgba[i * 4 + 3] = 255;
                    }
                });
            }
            BandRequest::CustomExpr {
                prog,
                rescale,
                colormap,
            } => {
                let ref_vecs: Vec<_> = prog
                    .used_bands
                    .iter()
                    .map(|&b| get_band_refs(b, b.saturating_sub(1) as usize))
                    .collect();
                let srcs: Vec<_> = ref_vecs.iter().map(|r| make_src!(r)).collect();
                let inv = 255.0 / cogzilla_kernel::safe_rescale_denom(rescale.0, rescale.1);
                let mut vals = [0.0f32; 16];
                let nb = srcs.len().min(16);
                coord_map.for_each_uv(w, h, |i, u, v| {
                    if u.is_nan() {
                        return;
                    }
                    for k in 0..nb {
                        vals[k] = sample(&srcs[k], u, v);
                    }
                    let val = prog.eval(&vals[..nb]);
                    if !val.is_nan() {
                        let [r, g, b, a] = map_color(val, rescale.0, inv, *colormap);
                        rgba[i * 4] = r;
                        rgba[i * 4 + 1] = g;
                        rgba[i * 4 + 2] = b;
                        rgba[i * 4 + 3] = a;
                    }
                });
            }
            BandRequest::CustomExprRgb {
                progs,
                rescale_min,
                rescale_max,
            } => {
                let all_refs: Vec<Vec<_>> = progs
                    .iter()
                    .map(|prog| {
                        prog.used_bands
                            .iter()
                            .map(|&b| get_band_refs(b, b.saturating_sub(1) as usize))
                            .collect()
                    })
                    .collect();
                let all_srcs: Vec<Vec<_>> = all_refs
                    .iter()
                    .map(|rvec| rvec.iter().map(|r| make_src!(r)).collect())
                    .collect();
                let inv = [
                    255.0 / cogzilla_kernel::safe_rescale_denom(rescale_min[0], rescale_max[0]),
                    255.0 / cogzilla_kernel::safe_rescale_denom(rescale_min[1], rescale_max[1]),
                    255.0 / cogzilla_kernel::safe_rescale_denom(rescale_min[2], rescale_max[2]),
                ];
                paint_rgb_band_math(
                    progs,
                    &all_srcs,
                    &sample,
                    &coord_map,
                    w,
                    h,
                    *rescale_min,
                    inv,
                    &mut rgba,
                );
            }
            BandRequest::Spectral1 {
                expr,
                band_1based,
                rescale,
                colormap,
            } => {
                let refs = get_band_refs(*band_1based, 0);
                let src = make_src!(&refs);
                let inv = 255.0 / cogzilla_kernel::safe_rescale_denom(rescale.0, rescale.1);
                coord_map.for_each_uv(w, h, |i, u, v| {
                    let val = expr.eval1(sample(&src, u, v));
                    if !val.is_nan() {
                        let [r, g, b, a] = map_color(val, rescale.0, inv, *colormap);
                        rgba[i * 4] = r;
                        rgba[i * 4 + 1] = g;
                        rgba[i * 4 + 2] = b;
                        rgba[i * 4 + 3] = a;
                    }
                });
            }
            BandRequest::Spectral2 {
                expr,
                bands_1based,
                rescale,
                colormap,
            } => {
                let r0 = get_band_refs(bands_1based[0], 0);
                let r1 = get_band_refs(bands_1based[1], 1);
                let s0 = make_src!(&r0);
                let s1 = make_src!(&r1);
                let inv = 255.0 / cogzilla_kernel::safe_rescale_denom(rescale.0, rescale.1);
                coord_map.for_each_uv(w, h, |i, u, v| {
                    let a = sample(&s0, u, v);
                    let b = sample(&s1, u, v);
                    let val = if a.is_nan() || b.is_nan() {
                        f32::NAN
                    } else {
                        expr.eval2(a, b)
                    };
                    if !val.is_nan() {
                        let [r, g, b, al] = map_color(val, rescale.0, inv, *colormap);
                        rgba[i * 4] = r;
                        rgba[i * 4 + 1] = g;
                        rgba[i * 4 + 2] = b;
                        rgba[i * 4 + 3] = al;
                    }
                });
            }
            BandRequest::Spectral3 {
                expr,
                bands_1based,
                rescale,
                colormap,
            } => {
                let r0 = get_band_refs(bands_1based[0], 0);
                let r1 = get_band_refs(bands_1based[1], 1);
                let r2 = get_band_refs(bands_1based[2], 2);
                let s0 = make_src!(&r0);
                let s1 = make_src!(&r1);
                let s2 = make_src!(&r2);
                let inv = 255.0 / cogzilla_kernel::safe_rescale_denom(rescale.0, rescale.1);
                coord_map.for_each_uv(w, h, |i, u, v| {
                    let a = sample(&s0, u, v);
                    let b = sample(&s1, u, v);
                    let c = sample(&s2, u, v);
                    let val = if a.is_nan() || b.is_nan() || c.is_nan() {
                        f32::NAN
                    } else {
                        expr.eval3(a, b, c)
                    };
                    if !val.is_nan() {
                        let [r, g, b, al] = map_color(val, rescale.0, inv, *colormap);
                        rgba[i * 4] = r;
                        rgba[i * 4 + 1] = g;
                        rgba[i * 4 + 2] = b;
                        rgba[i * 4 + 3] = al;
                    }
                });
            }
            BandRequest::MultiBand { .. } | BandRequest::CustomExprMulti { .. } => {}
        }

        if cf_single_lut.is_none()
            && let Some(cf) = color_formula
        {
            cf.apply_buffer_in_place(&mut rgba, 4);
        }

        if let Some(ref alpha) = continuous_alpha {
            cogzilla_kernel::apply_continuous_alpha_to_rgba(&mut rgba, alpha);
        }

        BboxRenderOutput::Bytes(encode_rgba_output(w, h, &rgba, format, return_mask))
    };

    // Offload large (>512x512) bbox renders to bounded spawn_blocking so giant previews/crops
    // never stall Tokio's async I/O workers.
    if w * h > 512 * 512 {
        let bands_owned = bands.clone();
        let lut_owned = custom_lut.cloned();
        let cf_owned = color_formula.cloned();
        let stats_owned = fused_stats.map(|(o, d)| (o.clone(), d.to_string()));
        let out = engine
            .store
            .run_heavy_cpu(move || {
                render_cpu(
                    coord_map,
                    cog_tiles,
                    &bands_owned,
                    lut_owned.as_ref(),
                    cf_owned.as_ref(),
                    stats_owned.as_ref().map(|(o, d)| (o, d.as_str())),
                )
            })
            .await?;
        Ok(out)
    } else {
        Ok(render_cpu(
            coord_map,
            cog_tiles,
            bands,
            custom_lut,
            color_formula,
            fused_stats,
        ))
    }
}
