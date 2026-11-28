use std::convert::Infallible;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use bytes::Bytes;
use cogzilla_kernel::{
    ColorFormula, ColormapName, CustomColormap, DemAlgorithm, ExprProgram, OutputFormat,
    ResampleMode, SpectralExpr,
};
use cogzilla_proj::mercator::lonlat_deg_to_mercator;
use cogzilla_proj::{TileCoord, mercator_to_lonlat_deg, tile_bounds_3857};
use http_body_util::{BodyExt, Full};
use hyper::header::{
    ACCESS_CONTROL_ALLOW_ORIGIN, CACHE_CONTROL, CONTENT_TYPE, ETAG, HeaderName, HeaderValue,
    IF_NONE_MATCH,
};
use hyper::{Method, Request, Response, StatusCode};

mod algorithm_math;
mod cog;
mod algorithm_params;
mod pgstac;
mod array_type;
#[cfg(test)]
mod asset_query_pins;
mod metrics;
mod mosaicjson;
mod paint;
mod html;
pub(crate) mod pixel_selection;
mod cql2_text;
pub(crate) mod geo;
mod point_expr;
mod query;
mod server;
mod stac;
mod wms;
mod zarr;
mod zonal;

use query::{ParsedQuery, parse_bool_encoded as parse_bool_param};
#[cfg(test)]
use query::parse_query;

/// Shared HTTP response type returned by every route-family handler.
pub(super) type Resp = Response<Full<Bytes>>;

/// Request context shared by all route-family handlers (`/cog`, `/stac`,
/// `/mosaicjson`/`/mosaic`, `/zarr`, `/searches`/`/collections`).
#[derive(Clone, Copy)]
pub(super) struct RequestCtx<'a> {
    pub(super) engine: &'a Arc<TileEngine>,
    pub(super) path: &'a str,
    pub(super) raw_query: &'a str,
    pub(super) host: &'a str,
    pub(super) forwarded_prefix: &'a str,
    pub(super) if_none_match: Option<&'a str>,
}

pub use self::server::{serve, serve_listener, serve_listener_with_shutdown};
pub(crate) use self::geo::{parse_coord_crs_with_epsg, tile_box_intersects_cog};
use self::geo::{
    bbox_to_wgs84, cog_native_epsg_and_bounds, coord_to_lonlat, dataset_native_crs_and_bounds,
    native_bbox_dims_in_crs, native_bbox_dims_in_crs_f64, normalize_render_bbox, query_mosaicjson_wgs84_bbox,
    reproject_wgs84_bounds, tile_intersects_cog, wgs84_bboxes_intersect,
};
#[cfg(test)]
use self::geo::cog_standard_epsg;
use self::wms::handle_cog_wms;
#[cfg(test)]
use self::wms::{build_wms_capabilities, parse_wms_format};

use crate::catalog::{MosaicJsonCatalog, SceneCatalog, load_catalog_json_str};
use crate::engine::{BandRequest, EngineError, TileEngine, TileRequest};
use self::paint::Paint;
use self::pixel_selection::{Fold, Selected, Selection};
use crate::stac::{
    StacItem, StacSource, get_colormap_image, get_colormap_json, get_tms_json,
    list_colormaps_json, list_tms_json, resolve_stac_assets,
};
use crate::geotiff::{encode_geotiff_f32, encode_geotiff_rgba};
use crate::multi_asset::render_multi_asset_bbox;
use crate::stats::{
    PolygonMask, StatsOptions, build_cog_info, build_cog_info_geojson, build_cog_validate,
    build_map_html, build_tilejson, build_wmts_capabilities, build_wmts_capabilities_multi,
    compute_f32_band_stats, compute_f32_band_stats_with_engine,
    compute_statistics_with_options,
    encode_auto_image, encode_rgba_output,
    get_algorithm_json,
    list_algorithms_json, query_point,
    render_crs_bbox, render_mercator_bbox, render_native_preview,
    render_native_preview_stats, render_wgs84_bbox, render_wgs84_bbox_stats,
};

fn percent_decode(input: &str) -> String {
    percent_decode_opts(input, true)
}

/// A query key as Starlette reads it: percent-decoded (`dst%5Fcrs` is `dst_crs`), borrowed
/// when there is nothing to decode.
pub(crate) fn query_key(k: &str) -> std::borrow::Cow<'_, str> {
    if k.contains('%') {
        std::borrow::Cow::Owned(percent_decode(k))
    } else {
        std::borrow::Cow::Borrowed(k)
    }
}

fn percent_decode_opts(input: &str, plus_as_space: bool) -> String {
    let bytes = input.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            let h1 = (bytes[i + 1] as char).to_digit(16);
            let h2 = (bytes[i + 2] as char).to_digit(16);
            if let (Some(a), Some(b)) = (h1, h2) {
                out.push(((a << 4) | b) as u8);
                i += 3;
                continue;
            }
        } else if plus_as_space && bytes[i] == b'+' {
            out.push(b' ');
            i += 1;
            continue;
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Parse an unsigned integer string, also accepting integer-valued decimal strings such as
/// `"256.0"` or `"12.0"` (matching TiTiler / Pydantic coercion).
fn parse_u32_like(s: &str) -> Option<u32> {
    let t = s.trim();
    if let Ok(v) = t.parse::<u32>() {
        return Some(v);
    }
    if t.contains('.')
        && let Ok(f) = t.parse::<f64>()
        && f.is_finite()
        && f >= 0.0
        && f <= u32::MAX as f64
        && f.fract() == 0.0
    {
        return Some(f as u32);
    }
    None
}

fn parse_u16_like(s: &str) -> Option<u16> {
    parse_u32_like(s).and_then(|v| u16::try_from(v).ok())
}

fn parse_u8_like(s: &str) -> Option<u8> {
    parse_u32_like(s).and_then(|v| u8::try_from(v).ok())
}

/// Parse TiTiler's `{y}[@{scale}x][.{ext}]` segment into `(y, scale_opt, ext_opt)`.
fn parse_y_scale_ext(seg: &str) -> Option<(u32, Option<u32>, Option<&str>)> {
    // Handle `1583.0.png` or `1583.0` as well as `1583.png` / `1583@2x.png`.
    let (left, ext) = if let Some(dot) = seg.rfind('.') {
        let after = &seg[dot + 1..];
        if !after.is_empty() && after.bytes().all(|b| b.is_ascii_alphabetic() || b == b'3' || b == b'2') && !after.bytes().all(|b| b.is_ascii_digit()) {
            (&seg[..dot], Some(after))
        } else {
            (seg, None)
        }
    } else {
        (seg, None)
    };
    let (y_str, scale) = match left.split_once('@') {
        Some((ys, sc)) => {
            let s_num = parse_u32_like(sc.trim_end_matches('x'))?;
            (ys, Some(s_num))
        }
        None => (left, None),
    };
    let y = parse_u32_like(y_str)?;
    Some((y, scale, ext))
}

fn parse_wxh_ext(seg: &str) -> Result<(u32, u32, Option<&str>), String> {
    let (dims, ext) = match seg.split_once('.') {
        Some((d, e)) => (d, Some(e)),
        None => (seg, None),
    };
    let Some((ws, hs)) = dims.split_once('x') else {
        return Err(format!("invalid dimensions '{seg}', expected {{w}}x{{h}}[.{{ext}}]"));
    };
    let (Some(w), Some(h)) = (parse_u32_like(ws), parse_u32_like(hs)) else {
        return Err(format!("invalid dimensions '{seg}', expected {{w}}x{{h}}[.{{ext}}]"));
    };
    if w == 0 || h == 0 {
        return Err(format!("requested dimensions {w}x{h} must be >= 1x1"));
    }
    if w > 4096 || h > 4096 {
        return Err(format!(
            "requested dimensions {w}x{h} exceed maximum allowed size 4096x4096"
        ));
    }
    Ok((w, h, ext))
}

fn parse_point_coord(coord: &str) -> Option<(f64, f64)> {
    let dec = percent_decode(coord);
    let (lons, lats) = dec.split_once(',')?;
    let x = lons.trim().parse::<f64>().ok()?;
    let y = lats.trim().parse::<f64>().ok()?;
    (x.is_finite() && y.is_finite()).then_some((x, y))
}

fn parse_bbox4(seg: &str) -> Option<([f64; 4], Option<&str>)> {
    // Handle `minx,miny,maxx,maxy.png` or `minx,miny,maxx,maxy` (allowing %20 spaces around commas)
    let (coords_str, ext) = if let Some(dot) = seg.rfind('.') {
        let after = &seg[dot + 1..];
        if matches!(
            after,
            "png" | "pngraw" | "webp" | "f32" | "bin" | "raw" | "npy" | "jpg" | "jpeg" | "tif" | "tiff"
        ) {
            (&seg[..dot], Some(after))
        } else {
            (seg, None)
        }
    } else {
        (seg, None)
    };
    let dec = percent_decode(coords_str);
    let parts: Vec<&str> = dec.split(',').collect();
    if parts.len() != 4 {
        return None;
    }
    let b = [
        parts[0].trim().parse::<f64>().ok()?,
        parts[1].trim().parse::<f64>().ok()?,
        parts[2].trim().parse::<f64>().ok()?,
        parts[3].trim().parse::<f64>().ok()?,
    ];
    if !b.iter().all(|v| v.is_finite()) {
        return None;
    }
    Some((b, ext))
}

/// Validate and normalize a parsed bbox `[minx, miny, maxx, maxy]` against its coordinate CRS.
/// - For geographic `coord_crs` (`None` or `CrsModel::Geographic`), `minx > maxx` with longitudes
///   in `[-180, 180]` is unwrapped across the antimeridian per RFC 7946 §5.2 (`maxx += 360.0`),
///   producing the exact same bbox as the `[minx, miny, maxx + 360, maxy]` form.
/// - For projected `coord_crs`, `minx >= maxx` is rejected with `400 Bad Request` stating that
///   `minx` must be less than `maxx`.
fn resolve_input_bbox(
    mut bbox: [f64; 4],
    coord_crs: Option<&cogzilla_proj::CrsModel>,
) -> Result<[f64; 4], EngineError> {
    if !bbox.iter().all(|v| v.is_finite()) {
        return Err(EngineError::BadRequest(
            "invalid bbox: coordinates must be finite".into(),
        ));
    }
    if bbox[1] >= bbox[3] {
        return Err(EngineError::BadRequest(format!(
            "invalid bbox: miny ({}) must be less than maxy ({})",
            bbox[1], bbox[3]
        )));
    }
    let is_geo = coord_crs.is_none_or(cogzilla_proj::CrsModel::is_geographic);
    if is_geo {
        // Symmetric ±2-turn longitude normalisation (matching point routes) before RFC 7946 §5.2
        // antimeridian unwrap: shift both bounds by the same whole multiple of 360° if needed.
        if (bbox[0] < -180.0 || bbox[0] > 180.0) && (bbox[2] < -180.0 || bbox[2] > 180.0)
            && let (Some(n0), Some(n2)) = (
                cogzilla_proj::normalize_geographic_lon(bbox[0]),
                cogzilla_proj::normalize_geographic_lon(bbox[2]),
            ) {
                let shift = n0 - bbox[0];
                if (bbox[2] + shift - n2).abs() < 1e-9 || (bbox[2] - bbox[0]) <= 360.0 {
                    bbox[0] += shift;
                    bbox[2] += shift;
                }
            }
    }
    if bbox[0] > bbox[2] {
        if is_geo && bbox[0] >= -180.0 && bbox[0] <= 180.0 && bbox[2] >= -180.0 && bbox[2] <= 180.0
        {
            bbox[2] += 360.0;
        } else {
            return Err(EngineError::BadRequest(format!(
                "invalid bbox: minx ({}) must be less than maxx ({})",
                bbox[0], bbox[2]
            )));
        }
    } else if bbox[0] == bbox[2] {
        return Err(EngineError::BadRequest(format!(
            "invalid bbox: minx ({}) must be less than maxx ({})",
            bbox[0], bbox[2]
        )));
    }
    Ok(bbox)
}

/// A `pixel_selection` by name: TiTiler's names in any case, and `std` for `stdev`.
fn parse_pixel_selection(name: &str) -> Option<Selection> {
    let name = name.to_ascii_lowercase();
    Selection::parse(if name == "std" { "stdev" } else { &name })
}

/// Whether a request's values are the raster's own whole numbers, which `mean` and `median`
/// are cast back to; only then does the cast lose anything.
fn reads_whole_numbers(unscale: bool, bands: &BandRequest, sample_type: cogzilla_raster::SampleType) -> bool {
    !unscale
        && matches!(
            bands,
            BandRequest::Single { .. } | BandRequest::Rgb { .. } | BandRequest::MultiBand { .. }
        )
        && !matches!(
            sample_type,
            cogzilla_raster::SampleType::Float16
                | cogzilla_raster::SampleType::Float32
                | cogzilla_raster::SampleType::Float64
        )
}

/// Reject requests that ask for something this server would otherwise ignore.
///
/// TiTiler answers these with a 422. Returning a 200 with the option silently
/// dropped gives a client the wrong picture and no way to notice.
/// Routes that honour `pixel_selection` (either by compositing scenes through `pixel_selection::Fold`
/// or by forwarding the query parameter into generated mosaic tile URL templates). Every other route
/// refuses `?pixel_selection=` with `400 Bad Request` instead of silently ignoring it.
fn route_supports_pixel_selection(segs: &[&str]) -> bool {
    match segs {
        ["mosaic", "tiles", ..] => true,
        ["mosaic", _, "tiles", ..] => true,
        ["mosaicjson", "tiles", .., "assets"] => false,
        ["mosaicjson", "tiles", ..] => true,
        ["mosaicjson", "bbox", _] => true,
        ["mosaicjson", "bbox", _, seg] if *seg != "assets" => true,
        ["mosaicjson", s] if *s == "feature" || s.starts_with("feature.") => true,
        ["mosaicjson", "feature", ..] => true,
        ["mosaicjson", "statistics"] => true,
        ["mosaicjson", "viewer"] => true,
        ["mosaicjson", _, "tilejson.json" | "map.html" | "WMTSCapabilities.xml"]
        | ["mosaicjson", "tilejson.json" | "map.html" | "WMTSCapabilities.xml"] => true,
        _ => false,
    }
}

/// Whether a route applies `algorithm`, as TiTiler's do that take its post-processing:
/// tiles and what points at them, pictures of a box, a feature or the whole, statistics.
fn route_applies_algorithm(segs: &[&str]) -> bool {
    if !matches!(segs.first(), Some(&("cog" | "stac" | "mosaicjson"))) {
        return false;
    }
    let named = |route: &str| segs.get(1).is_some_and(|s| *s == route || s.strip_prefix(route).is_some_and(|ext| ext.starts_with('.')));
    matches!(segs.last(), Some(&("tilejson.json" | "map.html" | "WMTSCapabilities.xml")))
        || (segs.get(1) == Some(&"tiles") && segs.len() > 3)
        || ["preview", "bbox", "feature", "statistics"].iter().any(|r| named(r))
}

fn validate_query(segs: &[&str], pq: &ParsedQuery) -> Result<(), String> {
    let family = segs.first().copied().unwrap_or("");
    if !matches!(family, "cog" | "stac" | "mosaicjson" | "mosaic" | "tiles" | "colorMaps") {
        return Ok(());
    }
    if let Some(e) = pq.errors.first() {
        return Err(e.clone());
    }
    if let Some(alg) = pq.algorithm.as_deref()
        && !crate::algorithms::is_supported_algorithm(alg)
    {
        return Err(format!(
            "algorithm '{alg}' is not supported (supported: {})",
            crate::algorithms::supported_algorithms_display()
        ));
    }

    // {family}/tiles/{tms}[/{z}/{x}/{y}], {family}/{tms}/tilejson.json, {family}/{tms}/map.html,
    // {family}/{tms}/WMTSCapabilities.xml
    if family != "tiles" && family != "colorMaps" {
        let last = segs.last().copied().unwrap_or("");
        let tms = if matches!(last, "tilejson.json" | "map.html" | "WMTSCapabilities.xml")
            && segs.len() >= 3
        {
            Some(segs[segs.len() - 2])
        } else {
            match segs.iter().position(|s| *s == "tiles") {
                Some(i) if last == "assets" => {
                    // `/mosaicjson/tiles/{tms}/{z}/{x}/{y}/assets` has len - i == 6;
                    // `/mosaicjson/tiles/{z}/{x}/{y}/assets` (len - i == 5) has no TMS segment.
                    if segs.len() - i == 6 {
                        Some(segs[i + 1])
                    } else {
                        None
                    }
                }
                Some(i) if matches!(segs.len() - i, 2 | 5) => Some(segs[i + 1]),
                _ => None,
            }
        };
        if let Some(tms) = tms
            && crate::tms::lookup_tms(tms).is_none() {
                return Err(format!(
                    "tile matrix set '{tms}' is not supported (see /tileMatrixSets)"
                ));
            }
    }

    let has = |names: &[&str]| pq.keys.iter().find(|k| names.contains(&k.as_str())).cloned();
    // Only match route sub-endpoints on `/cog/...`, `/stac/...`, and `/mosaicjson/...` so
    // dataset/catalog names like `/tiles/statistics/0/0/0.png` are never misclassified.
    let route = |name: &str| {
        matches!(segs[0], "cog" | "stac" | "mosaicjson")
            && segs
                .get(1)
                .is_some_and(|s| *s == name || s.starts_with(&format!("{name}.")))
    };
    let is_ogc_map =
        matches!(segs[0], "cog" | "stac" | "mosaicjson") && segs.get(1) == Some(&"map");
    let is_tileset_meta = matches!(segs[0], "cog" | "stac" | "mosaicjson")
        && segs.get(1) == Some(&"tiles")
        && segs.len() <= 3;
    if family != "colorMaps"
        && let Some(k) = has(&["orientation"]) {
            return Err(format!("parameter '{k}' is only supported on /colorMaps routes"));
        }
    if !(route("preview") || route("bbox") || route("feature") || is_ogc_map || route("wms") || route("statistics") || family == "colorMaps")
        && let Some(k) = has(&["width", "height"]) {
            return Err(format!(
                "parameter '{k}' is only supported on preview, bbox, feature, map, wms, statistics, and colorMaps routes"
            ));
        }
    if !(route("preview") || route("bbox") || route("feature") || is_ogc_map || route("statistics"))
        && let Some(k) = has(&["dst_crs"]) {
            return Err(format!(
                "parameter '{k}' is only supported on preview, bbox, feature, map, and statistics routes"
            ));
        }
    if !(route("bounds") || route("info") || is_ogc_map || route("wms") || is_tileset_meta)
        && let Some(k) = has(&["crs"]) {
            return Err(format!(
                "parameter '{k}' is only supported on bounds, info, tiles, map, and wms routes"
            ));
        }
    if route("statistics") || route("asset_statistics") {
        if let Some(k) = has(&["algorithm"])
            && !(segs[0] == "cog" || (segs[0] == "stac" && route("statistics"))) {
                return Err(format!(
                    "parameter '{k}' is only supported on /cog/statistics and /stac/statistics routes"
                ));
            }
        if !(segs[0] == "cog" || (segs[0] == "stac" && route("statistics")))
            && let Some(k) = has(&["expression", "expr", "index"]) {
                return Err(format!(
                    "parameter '{k}' is only supported on /cog/statistics and /stac/statistics routes"
                ));
            }
    } else if let Some(k) = has(&["p", "histogram_bins", "histogram_range", "categorical", "c"]) {
        return Err(format!("parameter '{k}' is only supported on statistics routes"));
    }
    if route("point") {
        if let Some(k) = has(&["algorithm"]) {
            return Err(format!("parameter '{k}' is not supported on point routes"));
        }
    } else if !(route("bbox") || route("feature") || route("statistics") || (segs[0] == "mosaicjson" && route("assets")))
        && let Some(k) = has(&["coord_crs"]) {
            return Err(format!("parameter '{k}' is not supported on this route"));
        }
    if route("bbox") && segs.len() >= 4
        && let Some(k) = has(&["width", "height"]) {
            return Err(format!("parameter '{k}' is not supported when explicit {{w}}x{{h}} dimensions are given"));
        }
    if (route("preview") || route("feature")) && segs.len() >= 3
        && let Some(k) = has(&["width", "height"]) {
            return Err(format!("parameter '{k}' is not supported when explicit {{w}}x{{h}} dimensions are given"));
        }
    let last_seg = segs.last().copied().unwrap_or("");
    if !matches!(last_seg, "tilejson.json" | "map.html" | "WMTSCapabilities.xml") && !is_tileset_meta {
        if let Some(k) = has(&["minzoom", "maxzoom", "tile_format", "tile_scale"]) {
            return Err(format!(
                "parameter '{k}' is only supported on tilejson.json, map.html, tiles, and WMTSCapabilities.xml routes"
            ));
        }
    } else if let (Some(minz), Some(maxz)) = (pq.minzoom, pq.maxzoom)
        && minz > maxz
    {
        return Err(format!("minzoom ({minz}) must be <= maxzoom ({maxz})"));
    }
    Ok(())
}

/// Apply optional `?width=` / `?height=` overrides onto aspect-preserving float dimensions,
/// preserving exact unrounded aspect ratio when only one axis is specified (matching `rio-tiler`).
fn resolve_query_dims_f64(default_w: f64, default_h: f64, q_w: Option<u32>, q_h: Option<u32>) -> (u32, u32) {
    match (q_w, q_h) {
        (Some(w), Some(h)) => (w.clamp(1, 4096), h.clamp(1, 4096)),
        (Some(w), None) => {
            let w = w.clamp(1, 4096);
            let h = ((w as f64 * (default_h / default_w.max(1e-9))).round() as u32)
                .clamp(1, 4096);
            (w, h)
        }
        (None, Some(h)) => {
            let h = h.clamp(1, 4096);
            let w = ((h as f64 * (default_w / default_h.max(1e-9))).round() as u32)
                .clamp(1, 4096);
            (w, h)
        }
        (None, None) => (
            default_w.round().clamp(1.0, 4096.0) as u32,
            default_h.round().clamp(1.0, 4096.0) as u32,
        ),
    }
}

/// Apply optional `?width=` / `?height=` overrides onto aspect-preserving default dimensions.
fn resolve_query_dims(default_w: u32, default_h: u32, q_w: Option<u32>, q_h: Option<u32>) -> (u32, u32) {
    resolve_query_dims_f64(default_w as f64, default_h as f64, q_w, q_h)
}

/// Strip TileJSON/WMTS-only query parameters so they are not forwarded into tile URL templates.
fn filter_tile_template_query(raw_query: &str) -> String {
    raw_query
        .split('&')
        .filter(|pair| {
            let k = query_key(pair.split('=').next().unwrap_or(""));
            !pair.is_empty()
                && !matches!(
                    k.as_ref(),
                    "minzoom" | "maxzoom" | "tile_format" | "tile_scale" | "use_epsg" | "zooms"
                )
        })
        .collect::<Vec<_>>()
        .join("&")
}

/// Apply TiTiler `minzoom`, `maxzoom`, `tile_format`, and `tile_scale` query overrides to a
/// TileJSON document, stripping those TileJSON-only params from the forwarded tile URL template.
fn apply_tilejson_overrides(
    tj: &mut serde_json::Value,
    pq: &ParsedQuery,
    host: &str,
    route_prefix: &str,
    tms: &str,
    raw_query: &str,
) {
    let origin = if host.starts_with("http://") || host.starts_with("https://") {
        host.to_string()
    } else {
        format!("http://{host}")
    };
    let filtered_query: String = raw_query
        .split('&')
        .filter(|pair| {
            let k = query_key(pair.split('=').next().unwrap_or(""));
            !pair.is_empty() && !matches!(k.as_ref(), "minzoom" | "maxzoom" | "tile_format" | "tile_scale")
        })
        .collect::<Vec<_>>()
        .join("&");
    let scale_suffix = match pq.tile_scale {
        Some(sc) if sc > 1 => format!("@{sc}x"),
        _ => String::new(),
    };
    let ext_suffix = pq
        .tile_format
        .as_deref()
        .map(|f| format!(".{f}"))
        .unwrap_or_default();
    let tile_url = if filtered_query.is_empty() {
        format!("{origin}/{route_prefix}/tiles/{tms}/{{z}}/{{x}}/{{y}}{scale_suffix}{ext_suffix}")
    } else {
        format!(
            "{origin}/{route_prefix}/tiles/{tms}/{{z}}/{{x}}/{{y}}{scale_suffix}{ext_suffix}?{filtered_query}"
        )
    };
    tj["tiles"] = serde_json::json!([tile_url]);
    let cur_min = tj.get("minzoom").and_then(|v| v.as_u64()).unwrap_or(0) as u8;
    let cur_max = tj.get("maxzoom").and_then(|v| v.as_u64()).unwrap_or(22) as u8;
    let (new_min, new_max) = resolve_zoom_range(cur_min, cur_max, pq.minzoom, pq.maxzoom);
    tj["minzoom"] = serde_json::json!(new_min);
    tj["maxzoom"] = serde_json::json!(new_max);
    if let Some(center) = tj.get_mut("center").and_then(|v| v.as_array_mut())
        && center.len() >= 3
    {
        center[2] = serde_json::json!(new_min);
    }
}

fn resolve_zoom_range(
    def_min: u8,
    def_max: u8,
    req_min: Option<u8>,
    req_max: Option<u8>,
) -> (u8, u8) {
    match (req_min, req_max) {
        (Some(mn), Some(mx)) => (mn, mx.max(mn)),
        (Some(mn), None) => (mn, def_max.max(mn)),
        (None, Some(mx)) => (def_min.min(mx), mx),
        (None, None) => (def_min, def_max),
    }
}

fn apply_info_bounds_crs(
    info: &mut serde_json::Value,
    wgs84_bounds: [f64; 4],
    bounds_crs: Option<&(cogzilla_proj::CrsModel, u16)>,
) -> Result<(), EngineError> {
    if let Some((dst_crs, dst_epsg)) = bounds_crs {
        let proj_bounds = reproject_wgs84_bounds(wgs84_bounds, dst_crs)?;
        info["bounds"] = serde_json::json!(proj_bounds);
        info["crs"] = serde_json::json!(format!(
            "http://www.opengis.net/def/crs/EPSG/0/{dst_epsg}"
        ));
    }
    Ok(())
}

const GEOTIFF_CONTENT_TYPE: &str = "image/tiff; application=geotiff";

/// Return the widest `SampleType` across one or more stacked COGs (matching numpy band stacking).
fn widest_sample_type(cogs: &[Arc<cogzilla_io::OpenedCog>]) -> cogzilla_raster::SampleType {
    cogs.iter()
        .map(|c| c.index.full_res().sample_type)
        .reduce(array_type::wider_type)
        .unwrap_or(cogzilla_raster::SampleType::Uint8)
}

/// Return true when a `.tif`/`.tiff` request should emit a native IEEE Float32 GeoTIFF
/// (`SampleFormat=3`, `GDAL_NODATA="nan"`) instead of an 8-bit visual RGB/RGBA GeoTIFF.
fn wants_f32_geotiff(
    ct: &str,
    pq: &ParsedQuery,
    source_cog: Option<&cogzilla_io::OpenedCog>,
    bands: &BandRequest,
    sample_type: cogzilla_raster::SampleType,
) -> bool {
    if ct != GEOTIFF_CONTENT_TYPE {
        return false;
    }
    if !pq.rescale.is_empty()
        || pq.colormap_explicit
        || pq.custom_colormap_json.is_some()
        || pq.color_formula.is_some()
    {
        return false;
    }
    match bands {
        BandRequest::Terrain { .. } => false,
        BandRequest::Dem { algorithm, .. } => !matches!(
            algorithm,
            cogzilla_kernel::DemAlgorithm::Hillshade { .. }
                | cogzilla_kernel::DemAlgorithm::Contours { .. }
        ),
        BandRequest::Single { .. } => {
            pq.unscale
                || (source_cog.is_none_or(|c| c.index.color_map.is_none())
                    && !matches!(sample_type, cogzilla_raster::SampleType::Uint8))
        }
        BandRequest::Rgb { .. } => {
            pq.unscale || !matches!(sample_type, cogzilla_raster::SampleType::Uint8)
        }
        BandRequest::CustomExpr { .. }
        | BandRequest::CustomExprRgb { .. }
        | BandRequest::CustomExprMulti { .. }
        | BandRequest::MultiBand { .. }
        | BandRequest::Spectral1 { .. }
        | BandRequest::Spectral2 { .. }
        | BandRequest::Spectral3 { .. }
        | BandRequest::SarRgb { .. } => true,
    }
}

fn geotiff_render_format(
    fmt: OutputFormat,
    ct: &str,
    pq: &ParsedQuery,
    source_cog: Option<&cogzilla_io::OpenedCog>,
    bands: &BandRequest,
    sample_type: cogzilla_raster::SampleType,
    allow_f64: bool,
) -> OutputFormat {
    let can_exact_f64 = allow_f64
        && sample_type.needs_f64()
        && !pq.unscale
        && !array_type::shown(pq)
        && pq.algorithm.is_none()
        && pq.expr_raw.is_none()
        && pq.buffer == 0.0
        && matches!(pq.resample, ResampleMode::Nearest | ResampleMode::Bilinear)
        && matches!(pq.reproject, ResampleMode::Nearest | ResampleMode::Bilinear)
        && matches!(
            bands,
            BandRequest::Single { .. } | BandRequest::Rgb { .. } | BandRequest::MultiBand { .. }
        );
    if wants_f32_geotiff(ct, pq, source_cog, bands, sample_type) {
        if can_exact_f64 {
            OutputFormat::RawF64
        } else {
            OutputFormat::RawF32
        }
    } else if fmt == OutputFormat::Npy && can_exact_f64 {
        OutputFormat::NpyF64
    } else {
        fmt
    }
}

/// Return true when an unrescaled `Uint16` `.png` request should emit a 16-bit PNG
/// matching TiTiler / GDAL's PNG driver instead of downscaling to 8-bit.
fn wants_u16_png(
    fmt: OutputFormat,
    ct: &str,
    pq: &ParsedQuery,
    source_cog: Option<&cogzilla_io::OpenedCog>,
    bands: &BandRequest,
    sample_type: cogzilla_raster::SampleType,
) -> bool {
    fmt == OutputFormat::Png
        && ct == "image/png"
        && matches!(
            sample_type,
            cogzilla_raster::SampleType::Uint12 | cogzilla_raster::SampleType::Uint16
        )
        && !pq.unscale
        && pq.rescale.is_empty()
        && !pq.colormap_explicit
        && pq.custom_colormap_json.is_none()
        && pq.color_formula.is_none()
        && pq.algorithm.is_none()
        && pq.expr_raw.is_none()
        && !source_cog.is_some_and(|c| has_embedded_palette(pq, c, bands))
        && matches!(bands, BandRequest::Single { .. } | BandRequest::Rgb { .. })
        && (TileEngine::value_bands(bands) + usize::from(pq.return_mask)) <= 4
}

fn render_output_format(
    fmt: OutputFormat,
    ct: &str,
    pq: &ParsedQuery,
    source_cog: Option<&cogzilla_io::OpenedCog>,
    bands: &BandRequest,
    sample_type: cogzilla_raster::SampleType,
    allow_f64: bool,
) -> OutputFormat {
    if wants_u16_png(fmt, ct, pq, source_cog, bands, sample_type) {
        OutputFormat::RawF32
    } else {
        geotiff_render_format(fmt, ct, pq, source_cog, bands, sample_type, allow_f64)
    }
}

fn wrap_geotiff_fmt(
    data: Bytes,
    ct: &'static str,
    fmt: OutputFormat,
    w: u32,
    h: u32,
    bbox: [f64; 4],
    epsg: u16,
    return_mask: bool,
) -> Bytes {
    if ct == GEOTIFF_CONTENT_TYPE {
        if matches!(fmt, OutputFormat::RawF64) {
            let n_px = (w as usize).saturating_mul(h as usize).max(1);
            let n_bands = (data.len() / (n_px * 8)).max(1);
            let vals: Vec<f64> = data
                .chunks_exact(8)
                .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
                .collect();
            crate::geotiff::write_geotiff_typed_f64(
                &vals,
                n_bands,
                w,
                h,
                cogzilla_raster::SampleType::Float64,
                bbox,
                epsg,
                return_mask,
            )
        } else if matches!(fmt, OutputFormat::RawF32) {
            encode_geotiff_f32(w, h, &data, bbox, epsg, return_mask)
        } else {
            encode_geotiff_rgba(w, h, &data, bbox, epsg, return_mask)
        }
    } else if ct == "image/png" && matches!(fmt, OutputFormat::RawF32) {
        if let Some(n_px) = (w as usize).checked_mul(h as usize).filter(|&n| n > 0)
            && let Some(row_bytes) = n_px.checked_mul(4)
        {
            let n_bands = data.len() / row_bytes;
            if n_bands >= 1
                && data.len() == n_bands * row_bytes
                && (n_bands + usize::from(return_mask)) <= 4
            {
                let vals: Vec<f32> = data
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                    .collect();
                return pgstac::png_of_16_bits(&vals, n_bands, w as usize, h as usize, return_mask);
            }
        }
        debug_assert!(
            false,
            "wrap_geotiff_fmt: unexpected RawF32 buffer length {} for {}x{} 16-bit PNG",
            data.len(),
            w,
            h
        );
        pgstac::png_of_16_bits(&[], 1, w.max(1) as usize, h.max(1) as usize, return_mask)
    } else {
        data
    }
}

/// Whether `data` may be an array the renderers wrote in float32: a float32 `.npy`, or a
/// GeoTIFF (whose samples [`array_in_its_type`] reads before it changes anything).
fn may_be_f32_array(data: &[u8]) -> bool {
    let npy = data.starts_with(b"\x93NUMPY") && data.get(10..80).is_some_and(|dict| dict.windows(4).any(|w| w == b"<f4'"));
    npy || data.starts_with(b"II*\0")
}

/// An array the renderers wrote in float32, as `.npy` or as a GeoTIFF, rewritten in `kind`,
/// the type TiTiler writes it in ([`array_type`]); with `mask`, its last band is the mask.
/// Left as it is without a type, or when it is not a float32 array of the bands it says.
fn array_in_its_type(data: Bytes, kind: Option<cogzilla_raster::SampleType>, mask: bool) -> Bytes {
    array_in_its_type_rounded(data, kind, mask, None, None)
}

/// [`array_in_its_type`], each value first rounded by `rounding` and put in `kind` as
/// numpy's `astype` puts it ([`array_type::astype`]): what `cast`, `ceil` and `floor` do.
/// `nodata_fill` is substituted for per-band `NaN`s on partially-valid multi-band pixels (#636).
fn array_in_its_type_rounded(
    data: Bytes,
    kind: Option<cogzilla_raster::SampleType>,
    mask: bool,
    nodata_fill: Option<f64>,
    rounding: Option<fn(f64) -> f64>,
) -> Bytes {
    let Some(kind) = kind else { return data };
    let tif = data.starts_with(b"II*\0");
    let typed = match rounding {
        None if tif => crate::geotiff::retype_f32_geotiff_with(&data, kind, mask, nodata_fill, |v| v),
        None => cogzilla_kernel::retype_f32_npy_with(&data, kind, mask, nodata_fill, |v| v),
        Some(round) => {
            let prepare = |v: f64| array_type::astype(kind, round(v));
            if tif {
                crate::geotiff::retype_f32_geotiff_with(&data, kind, mask, nodata_fill, prepare)
            } else {
                cogzilla_kernel::retype_f32_npy_with(&data, kind, mask, nodata_fill, prepare)
            }
        }
    };
    typed.unwrap_or(data)
}

/// What a `/cog` or `/stac` array is decided by.
#[derive(Clone)]
enum ArrayAsked {
    /// The values in their type: the rule, the assets asked for (none on `/cog`), the band
    /// math, whether a mask was asked for, the assets' own band math and the request
    /// expression over it (`ParsedQuery::asset_math_stages`), and any explicit `?nodata=`
    /// override.
    Typed(
        array_type::ArrayRule,
        Vec<String>,
        Option<String>,
        bool,
        Option<(Vec<Option<String>>, String)>,
        Option<f64>,
    ),
    /// The picture's bytes, for a `.npy` asked with a rescale, a colour map or a colour
    /// formula: the request, which says how they are painted.
    Shown(Box<ParsedQuery>),
}

/// `pq` with its assets' own band math folded into the request expression as
/// [`expand_stac_asset_bidx`] folds it, but with each asset's bands counted from one: `None`
/// when no asset has band math, or the request expression is not over its results.
fn fold_asset_math_for_typing(pq: &ParsedQuery) -> Option<ParsedQuery> {
    if pq.asset_exprs.is_empty() {
        return None;
    }
    let names: Vec<String> = pq.assets.iter().map(|a| a.split('|').next().unwrap_or(a).to_string()).collect();
    let own: Vec<Option<String>> = stac_entry_options(pq, &names).into_iter().map(|(own, _)| own).collect();
    let mut folded = pq.clone();
    // The array is typed as numpy types it: each asset's band math over its own raster, the
    // results stacked, then the request expression, if any, over them.
    folded.expr_raw = Some(pq.expr_raw.clone().unwrap_or_else(|| {
        own.iter().map(|e| e.as_deref().unwrap_or("b1")).collect::<Vec<_>>().join(";")
    }));
    folded.asset_math_stages = Some((own, pq.expr_raw.clone().unwrap_or_default()));
    folded.asset_exprs.clear();
    Some(folded)
}

/// What decides the array of a request whose path ends in `last`.
fn array_asked(pq: &ParsedQuery, last: Option<&&str>) -> Option<ArrayAsked> {
    // The assets' own band math is typed as the one expression it becomes over the stack
    // (`expand_stac_asset_bidx`): which band each part reads does not change its type.
    let folded;
    let pq = match fold_asset_math_for_typing(pq) {
        Some(f) => {
            folded = f;
            &folded
        }
        None => pq,
    };
    if let Some(rule) = array_type::ArrayRule::of(pq) {
        return Some(ArrayAsked::Typed(
            rule,
            pq.assets.clone(),
            pq.expr_raw.clone(),
            pq.return_mask,
            pq.asset_math_stages.clone(),
            pq.nodata.map(f64::from),
        ));
    }
    let npy = last.is_some_and(|last| last.ends_with(".npy")) || pq.format_param.as_deref() == Some("npy");
    let shown = npy && array_type::shown(pq);
    shown.then(|| ArrayAsked::Shown(Box::new(pq.clone())))
}

/// A float32 `.npy` of raw values painted as the request paints them ([`paint_selected`]),
/// as TiTiler writes an array asked with a rescale, a colour map or a colour formula: in
/// bytes, preserving all bands for multi-band rescaled arrays (or 1 grey / 3 RGB bands when
/// painted through a colour map or colour formula), then the mask (`255` where a pixel is
/// there) when it was asked for. `first` is the raster read first, for a colour map of its
/// own. `None` when `data` is not such an array.
fn shown_array(pq: &ParsedQuery, first: Option<&cogzilla_io::OpenedCog>, data: &[u8]) -> Option<Bytes> {
    let (shape, start) = cogzilla_kernel::f32_npy_shape(data)?;
    let [planes, height, width] = shape[..] else { return None };
    let pixels = height.checked_mul(width)?;
    let mask = pq.return_mask;
    let bands = planes.checked_sub(usize::from(mask)).filter(|b| *b > 0)?;
    let byte_len = planes.checked_mul(pixels)?.checked_mul(4)?;
    let end = start.checked_add(byte_len)?;
    let body = data.get(start..end)?;
    let floats: Vec<f32> = body.chunks_exact(4).map(|v| f32::from_le_bytes([v[0], v[1], v[2], v[3]])).collect();
    let (values, there) = floats.split_at(bands * pixels);
    // A pixel that is not there is no value, whatever the renderer filled it with.
    let values: Vec<f32> = if mask {
        values
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let m = there[i % pixels];
                if !m.is_finite() || m == 0.0 { f32::NAN } else { *v }
            })
            .collect()
    } else {
        values.to_vec()
    };
    let request = if bands == 1 && pq.expr_raw.is_none() && pq.asset_exprs.is_empty() {
        let rescale = pq.rescale.first().copied().unwrap_or((0.0, 255.0));
        BandRequest::Single { band_1based: 1, rescale, colormap: pq.colormap }
    } else {
        BandRequest::MultiBand { bands_1based: (1..=bands as u16).collect() }
    };
    let custom = match first {
        Some(first) => resolve_custom_lut(pq, first, &request).ok()?,
        None => pq
            .custom_colormap_json
            .as_deref()
            .map(|json| CustomColormap::from_json_str_with_type(json, !pq.rescale.is_empty(), pq.colormap_type_linear))
            .transpose()
            .ok()?,
    };
    // Preserve all N bands for 2-band and 4+-band rescaled arrays when no colormap/formula is active.
    if bands != 1
        && bands != 3
        && pq.colormap == ColormapName::Grayscale
        && custom.is_none()
        && pq.color_formula.is_none()
    {
        let out_planes = bands + usize::from(mask);
        let hdr = cogzilla_kernel::encode_npy_header("|u1", &[out_planes, height, width]);
        let mut out = Vec::with_capacity(hdr.len() + out_planes * pixels);
        out.extend_from_slice(&hdr);
        let mut valid = if mask { vec![0u8; pixels] } else { Vec::new() };
        for b in 0..bands {
            let (min, max) = pq.rescale.get(b).or(pq.rescale.first()).copied().unwrap_or((0.0, 255.0));
            let plane = &values[b * pixels..(b + 1) * pixels];
            for (p, &v) in plane.iter().enumerate() {
                if v.is_finite() {
                    out.push(paint::rescale_byte(v, min, max));
                    if mask {
                        valid[p] = 255;
                    }
                } else {
                    out.push(0);
                }
            }
        }
        if mask {
            out.extend_from_slice(&valid);
        }
        return Some(Bytes::from(out));
    }
    let selected = pixel_selection::Selected { bands, values };
    let grey = bands < 3 && pq.colormap == ColormapName::Grayscale && custom.is_none();
    let rgba = paint_selected(selected, pixels, &request, pq, custom.as_ref()).ok()?;
    let shown = if grey { 1 } else { 3 };
    let planes = shown + usize::from(mask);
    let hdr = cogzilla_kernel::encode_npy_header("|u1", &[planes, height, width]);
    let mut out = Vec::with_capacity(hdr.len() + planes * pixels);
    out.extend_from_slice(&hdr);
    for band in 0..shown {
        out.extend(rgba.chunks_exact(4).map(|px| px[band]));
    }
    if mask {
        out.extend(rgba.chunks_exact(4).map(|px| if px[3] == 0 { 0 } else { 255 }));
    }
    Some(Bytes::from(out))
}

/// Apply `ArrayAsked` (typed `.npy`/`.tif` or painted `|u1` `.npy`) over already-opened `cogs`.
/// Single shared entry point used by `/cog`, `/stac`, and `/mosaicjson` so all three route
/// families produce identical dtypes, mask sentinels, and GeoTIFF nodata metadata.
fn apply_array_asked(
    asked: Option<ArrayAsked>,
    cogs: &[Arc<cogzilla_io::OpenedCog>],
    expression_override: Option<&str>,
    data: Bytes,
) -> Bytes {
    let Some(asked) = asked.filter(|_| may_be_f32_array(&data)) else {
        return data;
    };
    if cogs.is_empty() {
        return data;
    }
    match asked {
        ArrayAsked::Typed(rule, _, expression, mask, stages, query_nodata) => {
            let reads: Vec<_> = cogs.iter().map(|cog| cog.index.full_res().sample_type).collect();
            let kind = match &stages {
                Some((own, request)) if own.len() == reads.len() => {
                    rule.staged_kind(&reads, own, request)
                }
                _ => rule.kind(&reads, expression_override.or(expression.as_deref())),
            };
            let rounding = kind.and_then(|kind| rule.rounding(kind));
            let nodata_fill = if matches!(rule, array_type::ArrayRule::Data { .. })
                && expression_override.or(expression.as_deref()).is_none()
            {
                query_nodata.or_else(|| cogs[0].index.nodata.map(f64::from))
            } else {
                None
            };
            array_in_its_type_rounded(data, kind, mask, nodata_fill, rounding)
        }
        ArrayAsked::Shown(pq) => {
            shown_array(&pq, cogs.first().map(|cog| &**cog), &data).unwrap_or(data)
        }
    }
}

/// A `/cog` answer in the type TiTiler writes its array in: the rasters at `urls` are
/// opened again (they are cached) for their types. Anything but a float32 array, and an
/// array whose rasters cannot be opened, is left as it is.
async fn cog_array_in_its_type(
    engine: &TileEngine,
    asked: Option<ArrayAsked>,
    urls: &[String],
    data: Bytes,
) -> Bytes {
    let Some(asked) = asked.filter(|_| may_be_f32_array(&data)) else {
        return data;
    };
    let Ok(cogs) = open_query_urls(engine, urls, None).await else {
        return data;
    };
    apply_array_asked(Some(asked), &cogs, None, data)
}

/// A `/stac` answer in the type TiTiler writes its array in, from the types of the item's
/// assets that were read and the band math as it was rewritten over them. As
/// [`cog_array_in_its_type`].
async fn stac_array_in_its_type(
    engine: &TileEngine,
    asked: Option<ArrayAsked>,
    item: &crate::stac::StacItem,
    data: Bytes,
) -> Bytes {
    let Some(asked) = asked.filter(|_| may_be_f32_array(&data)) else {
        return data;
    };
    let (assets, expression) = match &asked {
        ArrayAsked::Typed(_, assets, expression, _, _, _) => (assets.clone(), expression.clone()),
        ArrayAsked::Shown(pq) => (pq.assets.clone(), pq.expr_raw.clone()),
    };
    let Ok((names, cogs, expression)) =
        resolve_stac_assets(engine, item, &assets, expression.as_deref(), None).await
    else {
        return data;
    };
    let asked = match asked {
        // The assets read are not the entries asked (`:all:`): each read asset's band math is
        // found by its name and how many times it came before, as the stack finds it.
        ArrayAsked::Typed(rule, entries, expr, mask, Some((own, request)), query_nodata) if entries != names => {
            let of_name = |name: &String| -> Vec<&String> {
                entries.iter().zip(&own).filter(|(n, _)| *n == name).filter_map(|(_, e)| e.as_ref()).collect()
            };
            let own = names
                .iter()
                .zip(occurrences(&names))
                .map(|(name, nth)| {
                    let exprs = of_name(name);
                    exprs.get(nth).or(exprs.last()).map(|e| (*e).clone())
                })
                .collect();
            ArrayAsked::Typed(rule, entries, expr, mask, Some((own, request)), query_nodata)
        }
        asked => asked,
    };
    apply_array_asked(Some(asked), &cogs, expression.as_deref(), data)
}

/// A `/mosaicjson` (or `/mosaic`) answer in the type TiTiler writes its array in, using the
/// exact same [`apply_array_asked`] writer as `/cog` and `/stac`. Zero overhead for non-array
/// image formats (short-circuits immediately on `!may_be_f32_array(&data)`).
async fn mosaic_array_in_its_type(
    engine: &TileEngine,
    asked: Option<ArrayAsked>,
    urls: &[String],
    selection: pixel_selection::Selection,
    data: Bytes,
) -> Bytes {
    let Some(asked) = asked.filter(|_| may_be_f32_array(&data)) else {
        return data;
    };
    if let ArrayAsked::Typed(_, _, _, mask, _, _) = &asked {
        match selection {
            pixel_selection::Selection::Stdev => {
                return array_in_its_type(data, Some(cogzilla_raster::SampleType::Float32), *mask);
            }
            pixel_selection::Selection::Count => {
                let kind = if urls.len().min(64) <= 255 {
                    cogzilla_raster::SampleType::Uint8
                } else {
                    cogzilla_raster::SampleType::Uint16
                };
                return array_in_its_type(data, Some(kind), *mask);
            }
            _ => {}
        }
    }
    let cap = urls.len().min(64);
    let mut cogs = Vec::with_capacity(cap);
    for u in urls.iter().take(cap) {
        if let Ok(cog) = engine.store.open_uri(u).await {
            cogs.push(cog);
        }
    }
    apply_array_asked(Some(asked), &cogs, None, data)
}

/// A rendered `/cog` tile, its array in its type: what the tile cache keeps.
async fn cog_tile_in_its_type(
    engine: &TileEngine,
    asked: Option<ArrayAsked>,
    urls: &[String],
    rendered: impl std::future::Future<Output = Result<(Bytes, &'static str), EngineError>>,
) -> Result<(Bytes, &'static str), EngineError> {
    let (data, ct) = rendered.await?;
    Ok((cog_array_in_its_type(engine, asked, urls, data).await, ct))
}

/// A rendered `/stac` tile, its array in its type: what the tile cache keeps. The item is
/// the one the tile was drawn from, named by `url` or read from pgSTAC.
async fn stac_tile_in_its_type(
    engine: &TileEngine,
    asked: Option<ArrayAsked>,
    source: StacSource,
    rendered: impl std::future::Future<Output = Result<(Bytes, &'static str), EngineError>>,
) -> Result<(Bytes, &'static str), EngineError> {
    let (data, ct) = rendered.await?;
    if asked.is_none() || !may_be_f32_array(&data) {
        return Ok((data, ct));
    }
    match source.load(engine).await {
        Ok(item) => Ok((stac_array_in_its_type(engine, asked, &item, data).await, ct)),
        Err(_) => Ok((data, ct)),
    }
}

#[allow(clippy::too_many_arguments)]
fn finalize_image_output(
    data: Bytes,
    ct: &'static str,
    fmt: OutputFormat,
    auto_fmt: bool,
    out_w: u32,
    out_h: u32,
    bbox: [f64; 4],
    epsg: u16,
    jpeg_quality: u8,
    return_mask: bool,
) -> (Bytes, &'static str) {
    if auto_fmt {
        encode_auto_image(out_w as usize, out_h as usize, &data, jpeg_quality, return_mask)
    } else {
        (wrap_geotiff_fmt(data, ct, fmt, out_w, out_h, bbox, epsg, return_mask), ct)
    }
}

fn attach_x_assets(resp: &mut Response<Full<Bytes>>, assets: &[String]) {
    if assets.is_empty() || !resp.status().is_success() {
        return;
    }
    let joined = assets
        .iter()
        .map(|u| cogzilla_io::redact_http_url(u))
        .collect::<Vec<_>>()
        .join(",");
    if let Ok(hv) = hyper::header::HeaderValue::from_str(&joined) {
        resp.headers_mut().insert("x-assets", hv);
    }
}

fn attach_x_failed_assets(resp: &mut Response<Full<Bytes>>, failed: &[String]) {
    if failed.is_empty() || !resp.status().is_success() {
        return;
    }
    let joined = failed
        .iter()
        .map(|u| cogzilla_io::redact_http_url(u))
        .collect::<Vec<_>>()
        .join(",");
    if let Ok(hv) = hyper::header::HeaderValue::from_str(&joined) {
        resp.headers_mut().insert("x-failed-assets", hv);
    }
}

fn parse_ext_format(
    ext: &str,
    webp_quality: Option<f32>,
    jpeg_quality: u8,
) -> Result<(OutputFormat, &'static str), EngineError> {
    match ext {
        "png" | "pngraw" | "" => Ok((OutputFormat::Png, "image/png")),
        "tif" | "tiff" => Ok((OutputFormat::RawRgba, GEOTIFF_CONTENT_TYPE)),
        "webp" => Ok((
            OutputFormat::Webp {
                quality: webp_quality,
            },
            "image/webp",
        )),
        "jpg" | "jpeg" => Ok((OutputFormat::Jpeg { quality: jpeg_quality }, "image/jpeg")),
        "f32" | "bin" | "raw" => Ok((OutputFormat::RawF32, "application/octet-stream")),
        "npy" => Ok((OutputFormat::Npy, "application/x-binary")),
        other => Err(EngineError::BadRequest(format!(
            "unsupported format '.{other}' (use .png, .webp, .jpg, or .f32)"
        ))),
    }
}

/// Fold GDAL `scale`/`offset` into rescale bounds when `?unscale=true` (zero per-pixel cost!).
/// Supports both single multi-band COGs and multi-asset STAC/multi-COG composites.
/// `slot_bands`: the band each of several `cogs` is read for, when not its first
/// ([`expand_stac_asset_bidx`]), whose scale and offset are then that band's.
fn apply_unscale_cogs(bands: &mut BandRequest, cogs: &[Arc<cogzilla_io::OpenedCog>], slot_bands: Option<&[u16]>) {
    if cogs.is_empty() {
        return;
    }
    let adj = |b1: u16, mn: f32, mx: f32| -> (f32, f32) {
        let i = b1.saturating_sub(1) as usize;
        let (sc, off) = if cogs.len() > 1 {
            let slot = i.min(cogs.len() - 1);
            let c = &cogs[slot];
            let band = slot_bands.and_then(|s| s.get(slot)).map_or(0, |b| usize::from(b.saturating_sub(1)));
            (
                c.index.scales.get(band).copied().unwrap_or(1.0),
                c.index.offsets.get(band).copied().unwrap_or(0.0),
            )
        } else {
            (
                cogs[0].index.scales.get(i).copied().unwrap_or(1.0),
                cogs[0].index.offsets.get(i).copied().unwrap_or(0.0),
            )
        };
        if (sc - 1.0).abs() < 1e-9 && off.abs() < 1e-9 || sc.abs() < 1e-12 {
            (mn, mx)
        } else {
            ((mn - off) / sc, (mx - off) / sc)
        }
    };
    match bands {
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
}

/// Compute aspect-preserving `(width, height)` inside `max_dim` for format-only `/bbox` and
/// `POST /feature` routes (matching TiTiler instead of stretching into a fixed square).
fn aspect_preserving_dims(bbox: [f64; 4], max_dim: u32) -> (u32, u32) {
    let max_d = max_dim.clamp(1, 4096) as f64;
    let dx = (bbox[2] - bbox[0]).abs();
    let dy = (bbox[3] - bbox[1]).abs();
    if !dx.is_finite() || !dy.is_finite() || dx <= 0.0 || dy <= 0.0 {
        let d = max_dim.clamp(1, 4096);
        return (d, d);
    }
    if dx >= dy {
        let w = max_d.round() as u32;
        let h = ((max_d * (dy / dx)).round() as u32).clamp(1, max_dim.clamp(1, 4096));
        (w.max(1), h)
    } else {
        let h = max_d.round() as u32;
        let w = ((max_d * (dx / dy)).round() as u32).clamp(1, max_dim.clamp(1, 4096));
        (w, h.max(1))
    }
}

/// Compute native-resolution `(width, height)` for `dst_bbox` capped at `max_dim`,
/// matching TiTiler / GDAL `WarpedVRT` without upsampling past native pixel resolution.
fn capped_native_bbox_dims(
    engine: &TileEngine,
    cog: &cogzilla_io::OpenedCog,
    dst_crs: &cogzilla_proj::CrsModel,
    wgs_bbox: [f64; 4],
    dst_bbox: [f64; 4],
    max_dim: u32,
) -> (u32, u32) {
    let cap = max_dim.clamp(1, 4096);
    let Ok(crs) = engine.get_or_build_crs(cog) else {
        return aspect_preserving_dims(dst_bbox, cap);
    };
    let (nat_w, nat_h) = native_bbox_dims_in_crs(cog.index.full_res(), &crs, dst_crs, wgs_bbox, dst_bbox);
    if nat_w.max(nat_h) <= cap {
        (nat_w.max(1), nat_h.max(1))
    } else if nat_w >= nat_h {
        let h = ((nat_h as f64 / nat_w as f64) * cap as f64)
            .ceil()
            .clamp(1.0, cap as f64) as u32;
        (cap, h)
    } else {
        let w = ((nat_w as f64 / nat_h as f64) * cap as f64)
            .ceil()
            .clamp(1.0, cap as f64) as u32;
        (w, cap)
    }
}

/// Resolve output `(width, height)` for a `/bbox`, `/feature`, or `/preview?dst_crs=` request:
/// when a single axis (`width` or `height`) is given, derives the other from the unrounded float
/// native aspect ratio (matching `rio-tiler::reader.part` / `output_size` without integer-rounding
/// distortion); otherwise applies `capped_native_bbox_dims`.
#[allow(clippy::too_many_arguments)]
fn resolve_bbox_query_dims(
    engine: &TileEngine,
    cog: &cogzilla_io::OpenedCog,
    dst_crs: &cogzilla_proj::CrsModel,
    wgs_bbox: [f64; 4],
    dst_bbox: [f64; 4],
    max_dim: u32,
    q_w: Option<u32>,
    q_h: Option<u32>,
) -> (u32, u32) {
    match (q_w, q_h) {
        (Some(w), Some(h)) => (w.clamp(1, 4096), h.clamp(1, 4096)),
        (None, None) => capped_native_bbox_dims(engine, cog, dst_crs, wgs_bbox, dst_bbox, max_dim),
        (q_w, q_h) => {
            if let Ok(crs) = engine.get_or_build_crs(cog) {
                let (nat_w, nat_h) = native_bbox_dims_in_crs_f64(
                    cog.index.full_res(),
                    &crs,
                    dst_crs,
                    wgs_bbox,
                    dst_bbox,
                );
                if nat_w.is_finite() && nat_h.is_finite() && nat_w > 0.0 && nat_h > 0.0 {
                    return resolve_query_dims_f64(nat_w, nat_h, q_w, q_h);
                }
            }
            let (def_w, def_h) = capped_native_bbox_dims(engine, cog, dst_crs, wgs_bbox, dst_bbox, max_dim);
            resolve_query_dims(def_w, def_h, q_w, q_h)
        }
    }
}

/// Default rescale range matching TiTiler's `dtype_ranges` when `?rescale=` is omitted.
fn default_rescale_for_dtype(st: cogzilla_raster::SampleType, single: bool) -> (f32, f32) {
    st.reported_range().unwrap_or(if single { (0.0, 1000.0) } else { (0.0, 1.0) })
}

/// The band math of a request whose `algorithm` works over its `expression`'s outputs
/// ([`algorithm_math::over_results`]), with the algorithm's own default range for a picture
/// when none was asked; `None` when there is no such pair.
fn algorithm_over_expression(pq: &ParsedQuery, read: Option<cogzilla_raster::SampleType>) -> Result<Option<ParsedQuery>, EngineError> {
    let (Some(alg), Some(expr)) = (pq.algorithm.as_deref(), pq.expr_raw.as_deref()) else {
        return Ok(None);
    };
    if SpectralExpr::from_str_opt(expr).is_some() {
        return Ok(None);
    }
    let parts: Vec<(String, String)> = expression_parts(expr).map(|p| (p.to_string(), p.to_string())).collect();
    let typed = band_math_output_type(pq, &parts, read);
    let Some(composed_math) = algorithm_math::over_results(alg, typed, pq.cast_dtype.as_deref(), pq.threshold, &parts)? else {
        return Ok(None);
    };
    // The range a picture of the algorithm spans when none is asked, as without band math,
    // from the type of the band math's outputs.
    let lower = alg.to_ascii_lowercase();
    let source = default_rescale_for_dtype(typed.unwrap_or(cogzilla_raster::SampleType::Float32), true);
    let range = match lower.as_str() {
        // A cast to a float has no integer range: the values keep the band math's own.
        "cast" | "ceil" | "floor" => algorithm_math::integer_band(&lower, 1, None, pq.cast_dtype.as_deref()).1.or(Some(source)),
        "normalizedindex" => Some((-1.0, 1.0)),
        "bitonal" => Some((0.0, 255.0)),
        "sum" => Some((source.0, source.1 * parts.len() as f32)),
        _ => Some(source),
    };
    let mut composed = pq.clone();
    composed.expr_raw = Some(composed_math.outputs.into_iter().map(|(_, text)| text).collect::<Vec<_>>().join(";"));
    composed.expr_op_budget = composed_math.max_ops;
    composed.algorithm = None;
    if composed.rescale.is_empty()
        && let Some(range) = range
    {
        composed.rescale = vec![range];
    }
    Ok(Some(composed))
}

/// The type of band math's outputs as numpy types them: as folded from the assets' own band
/// math (`ParsedQuery::band_math_type`), or the expression over bands read as `read`.
fn band_math_output_type(
    pq: &ParsedQuery,
    parts: &[(String, String)],
    read: Option<cogzilla_raster::SampleType>,
) -> Option<cogzilla_raster::SampleType> {
    pq.band_math_type.or_else(|| {
        array_type::ArrayRule::Math { unscale: pq.unscale }
            .kind_with_exprs(&[read?], Some(parts.iter().map(|(_, text)| text)))
    })
}

fn build_band_request(
    pq: &ParsedQuery,
    default_bands: u16,
    sample_type: cogzilla_raster::SampleType,
) -> Result<BandRequest, EngineError> {
    if let Some(composed) = algorithm_over_expression(pq, Some(sample_type))? {
        return build_band_request(&composed, default_bands, sample_type);
    }
    if let Some(def) = pq.algorithm.as_deref().and_then(crate::algorithms::lookup_algorithm) {
        use crate::algorithms::AlgorithmId;
        let band_1based = pq.bidx.first().copied().unwrap_or(1);
        // A terrain algorithm over band math reads the expression's first output, as rio-tiler
        // gives the algorithm the expression's results.
        let dem_math = match pq
            .expr_raw
            .as_deref()
            .filter(|_| def.kind == crate::algorithms::AlgorithmKind::Dem)
            // A spectral preset (`ndvi`) is left to the algorithm's own band, as the band
            // math algorithms leave it (`algorithm_over_expression`).
            .filter(|expr| SpectralExpr::from_str_opt(expr).is_none())
            .and_then(|expr| expression_parts(expr).next())
        {
            Some(first) => {
                let prog = ExprProgram::compile(first).map_err(EngineError::BadRequest)?;
                // The elevations are read where the bands are: an expression that reads
                // none has no raster to read them over.
                if prog.used_bands.is_empty() {
                    return Err(EngineError::BadRequest(format!(
                        "a terrain algorithm's expression must read a band: '{first}'"
                    )));
                }
                Some(prog)
            }
            None => None,
        };
        match def.id {
            AlgorithmId::Terrarium => {
                return Ok(BandRequest::Terrain {
                    band_1based,
                    algorithm: DemAlgorithm::Terrarium,
                    math: dem_math,
                });
            }
            AlgorithmId::TerrainRgb => {
                return Ok(BandRequest::Terrain {
                    band_1based,
                    algorithm: DemAlgorithm::TerrainRgb,
                    math: dem_math,
                });
            }
            AlgorithmId::Hillshade => {
                let rescale = pq.rescale.first().copied().unwrap_or((0.0, 255.0));
                let colormap = if pq.colormap_explicit {
                    pq.colormap
                } else {
                    ColormapName::from_str_opt("gray").unwrap_or(ColormapName::Viridis)
                };
                return Ok(BandRequest::Dem {
                    band_1based,
                    algorithm: DemAlgorithm::Hillshade {
                        azimuth_deg: pq.azimuth,
                        altitude_deg: pq.altitude,
                        z_factor: pq.z_factor,
                    },
                    rescale,
                    colormap,
                    math: dem_math,
                });
            }
            AlgorithmId::Slope => {
                let rescale = pq.rescale.first().copied().unwrap_or((0.0, 90.0));
                return Ok(BandRequest::Dem {
                    band_1based,
                    algorithm: DemAlgorithm::Slope {
                        z_factor: pq.z_factor,
                    },
                    rescale,
                    colormap: pq.colormap,
                    math: dem_math,
                });
            }
            AlgorithmId::Aspect => {
                let rescale = pq.rescale.first().copied().unwrap_or((0.0, 360.0));
                return Ok(BandRequest::Dem {
                    band_1based,
                    algorithm: DemAlgorithm::Aspect,
                    rescale,
                    colormap: pq.colormap,
                    math: dem_math,
                });
            }
            AlgorithmId::Contours => {
                return Ok(BandRequest::Terrain {
                    band_1based,
                    algorithm: DemAlgorithm::Contours {
                        increment: pq.contour_increment.max(1e-6),
                        thickness: pq.contour_thickness,
                        minz: pq.contour_minz,
                        maxz: pq.contour_maxz,
                    },
                    math: dem_math,
                });
            }
            AlgorithmId::NormalizedIndex
            | AlgorithmId::Cast
            | AlgorithmId::Ceil
            | AlgorithmId::Floor
            | AlgorithmId::Min
            | AlgorithmId::Max
            | AlgorithmId::Median
            | AlgorithmId::Mean
            | AlgorithmId::Std
            | AlgorithmId::Var
            | AlgorithmId::Sum
            | AlgorithmId::Grayscale
            | AlgorithmId::Bitonal => {
                let alg_lower = def.name.to_ascii_lowercase();
                let alg = alg_lower.as_str();
                let bands: Vec<u16> = if !pq.bidx.is_empty() {
                    pq.bidx.clone()
                } else {
                    (1..=default_bands.clamp(1, 16)).collect()
                };
                let whole = algorithm_math::whole_numbers(sample_type, pq.unscale);
                let source_range = default_rescale_for_dtype(sample_type, true);
                if matches!(def.id, AlgorithmId::Cast | AlgorithmId::Ceil | AlgorithmId::Floor) && bands.len() >= 3 {
                    // Each of the first three bands, as an RGB picture.
                    let one = |b: u16| -> Result<(ExprProgram, (f32, f32)), EngineError> {
                        let (expr, range) = algorithm_math::integer_band(alg, b, whole, pq.cast_dtype.as_deref());
                        Ok((ExprProgram::compile(&expr).map_err(EngineError::BadRequest)?, range.unwrap_or(source_range)))
                    };
                    let ((p0, dr), (p1, _), (p2, _)) = (one(bands[0])?, one(bands[1])?, one(bands[2])?);
                    let r0 = pq.rescale.first().copied().unwrap_or(dr);
                    let r1 = pq.rescale.get(1).copied().unwrap_or(r0);
                    let r2 = pq.rescale.get(2).copied().unwrap_or(r0);
                    return Ok(BandRequest::CustomExprRgb {
                        progs: [p0, p1, p2],
                        rescale_min: [r0.0, r1.0, r2.0],
                        rescale_max: [r0.1, r1.1, r2.1],
                    });
                }
                let expr_str = algorithm_math::band_math(alg, &bands, whole, pq.cast_dtype.as_deref(), pq.threshold)
                    .map_err(EngineError::BadRequest)?;
                let default_rescale = match def.id {
                    AlgorithmId::NormalizedIndex => (-1.0, 1.0),
                    AlgorithmId::Cast | AlgorithmId::Ceil | AlgorithmId::Floor => {
                        algorithm_math::integer_band(alg, bands[0], whole, pq.cast_dtype.as_deref()).1.unwrap_or(source_range)
                    }
                    AlgorithmId::Bitonal => (0.0, 255.0),
                    AlgorithmId::Sum => (source_range.0, source_range.1 * bands.len() as f32),
                    _ => source_range,
                };
                let prog = ExprProgram::compile(&expr_str).map_err(EngineError::BadRequest)?;
                let rescale = pq.rescale.first().copied().unwrap_or(default_rescale);
                return Ok(BandRequest::CustomExpr {
                    prog,
                    rescale,
                    colormap: pq.colormap,
                });
            }
        }
    }
    if let Some(raw_expr) = &pq.expr_raw {
        if let Some(preset) = SpectralExpr::from_str_opt(raw_expr) {
            return Ok(match preset {
                SpectralExpr::Db => {
                    let rescale = pq.rescale.first().copied().unwrap_or((-25.0, 0.0));
                    let b = pq.bidx.first().copied().unwrap_or(1);
                    BandRequest::Spectral1 {
                        expr: preset,
                        band_1based: b,
                        rescale,
                        colormap: pq.colormap,
                    }
                }
                SpectralExpr::SarRgb => {
                    let bs = if pq.bidx.len() >= 2 {
                        [pq.bidx[0], pq.bidx[1]]
                    } else {
                        [1, 2]
                    };
                    let (rmin, rmax) = match pq.rescale.len() {
                        0 => ([-20.0, -25.0, 0.0], [0.0, -5.0, 15.0]),
                        1 => ([pq.rescale[0].0; 3], [pq.rescale[0].1; 3]),
                        3 => (
                            [pq.rescale[0].0, pq.rescale[1].0, pq.rescale[2].0],
                            [pq.rescale[0].1, pq.rescale[1].1, pq.rescale[2].1],
                        ),
                        n => {
                            return Err(EngineError::BadRequest(format!(
                                "expected 1 or 3 rescale pairs for 3-band SAR RGB output, got {n}"
                            )));
                        }
                    };
                    BandRequest::SarRgb {
                        bands_1based: bs,
                        rescale_min: rmin,
                        rescale_max: rmax,
                    }
                }
                SpectralExpr::Evi => {
                    let rescale = pq.rescale.first().copied().unwrap_or((-0.2, 0.8));
                    let bs = if pq.bidx.len() >= 3 {
                        [pq.bidx[0], pq.bidx[1], pq.bidx[2]]
                    } else {
                        [1, 2, 3]
                    };
                    BandRequest::Spectral3 {
                        expr: preset,
                        bands_1based: bs,
                        rescale,
                        colormap: pq.colormap,
                    }
                }
                _ => {
                    let rescale = pq.rescale.first().copied().unwrap_or((-1.0, 1.0));
                    let bs = if pq.bidx.len() >= 2 {
                        [pq.bidx[0], pq.bidx[1]]
                    } else {
                        [1, 2]
                    };
                    BandRequest::Spectral2 {
                        expr: preset,
                        bands_1based: bs,
                        rescale,
                        colormap: pq.colormap,
                    }
                }
            });
        }
        // Arbitrary TiTiler expression -> compile to bytecode VM!
        // Supports single expressions (`(b1-b2)/(b1+b2)`) and `;`-separated 3-band RGB
        // expressions (`b3*2;b2*2;b1*2`).
        let parts: Vec<&str> = raw_expr
            .split(';')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();
        if parts.is_empty() {
            return Err(EngineError::BadRequest("empty expression".into()));
        }
        if parts.len() == 3 {
            if pq.colormap_explicit || pq.custom_colormap_json.is_some() {
                return Err(EngineError::BadRequest(
                    "colormap cannot be applied to 3-band RGB output".into(),
                ));
            }
            let compile = |p: &str| ExprProgram::compile_with_limit(p, pq.expr_op_budget).map_err(EngineError::BadRequest);
            let (p0, p1, p2) = (compile(parts[0])?, compile(parts[1])?, compile(parts[2])?);
            let (rmin, rmax) = match pq.rescale.len() {
                0 => {
                    let (lo, hi) = default_rescale_for_dtype(sample_type, false);
                    ([lo; 3], [hi; 3])
                }
                1 => ([pq.rescale[0].0; 3], [pq.rescale[0].1; 3]),
                3 => (
                    [pq.rescale[0].0, pq.rescale[1].0, pq.rescale[2].0],
                    [pq.rescale[0].1, pq.rescale[1].1, pq.rescale[2].1],
                ),
                n => {
                    return Err(EngineError::BadRequest(format!(
                        "rescale count ({n}) does not match band count (3); expected 1 or 3 rescale ranges"
                    )));
                }
            };
            return Ok(BandRequest::CustomExprRgb {
                progs: [p0, p1, p2],
                rescale_min: rmin,
                rescale_max: rmax,
            });
        } else if parts.len() != 1 {
            if parts.len() > 15 {
                return Err(EngineError::BadRequest(format!(
                    "multi-band tile expression count ({}) exceeds maximum of 15",
                    parts.len()
                )));
            }
            if pq.colormap_explicit || pq.custom_colormap_json.is_some() {
                return Err(EngineError::BadRequest(
                    "colormap cannot be applied to multi-band expression output".into(),
                ));
            }
            let mut progs = Vec::with_capacity(parts.len());
            for p in &parts {
                progs.push(ExprProgram::compile_with_limit(p, pq.expr_op_budget).map_err(EngineError::BadRequest)?);
            }
            return Ok(BandRequest::CustomExprMulti { progs });
        }
        if pq.rescale.len() > 1 {
            return Err(EngineError::BadRequest(format!(
                "rescale count ({}) does not match band count (1); expected 1 rescale range",
                pq.rescale.len()
            )));
        }
        let prog = ExprProgram::compile_with_limit(parts[0], pq.expr_op_budget).map_err(EngineError::BadRequest)?;
        let rescale = pq.rescale.first().copied().unwrap_or((-1.0, 1.0));
        return Ok(BandRequest::CustomExpr {
            prog,
            rescale,
            colormap: pq.colormap,
        });
    }

    if !pq.bidx.is_empty() && pq.bidx.len() != 1 && pq.bidx.len() != 3 {
        if pq.bidx.len() > 15 {
            return Err(EngineError::BadRequest(format!(
                "bidx count {} exceeds maximum of 15 bands",
                pq.bidx.len()
            )));
        }
        return Ok(BandRequest::MultiBand {
            bands_1based: pq.bidx.clone(),
        });
    }

    if pq.bidx.len() == 3 || (pq.bidx.is_empty() && default_bands >= 3) {
        if pq.colormap_explicit || pq.custom_colormap_json.is_some() {
            return Err(EngineError::BadRequest(
                "colormap cannot be applied to 3-band RGB output".into(),
            ));
        }
        let bs = if pq.bidx.len() == 3 {
            [pq.bidx[0], pq.bidx[1], pq.bidx[2]]
        } else {
            [1, 2, 3]
        };
        let (rmin, rmax) = match pq.rescale.len() {
            0 => {
                let (lo, hi) = default_rescale_for_dtype(sample_type, false);
                ([lo; 3], [hi; 3])
            }
            1 => ([pq.rescale[0].0; 3], [pq.rescale[0].1; 3]),
            3 => (
                [pq.rescale[0].0, pq.rescale[1].0, pq.rescale[2].0],
                [pq.rescale[0].1, pq.rescale[1].1, pq.rescale[2].1],
            ),
            n => {
                return Err(EngineError::BadRequest(format!(
                    "rescale count ({n}) does not match band count (3); expected 1 or 3 rescale ranges"
                )));
            }
        };
        Ok(BandRequest::Rgb {
            bands_1based: bs,
            rescale_min: rmin,
            rescale_max: rmax,
        })
    } else {
        if pq.rescale.len() > 1 {
            return Err(EngineError::BadRequest(format!(
                "expected 1 rescale pair for single-band output, got {}",
                pq.rescale.len()
            )));
        }
        let b = pq.bidx.first().copied().unwrap_or(1);
        let default_range = default_rescale_for_dtype(sample_type, true);
        let rescale = pq.rescale.first().copied().unwrap_or(default_range);
        Ok(BandRequest::Single {
            band_1based: b,
            rescale,
            colormap: pq.colormap,
        })
    }
}

/// Match TiTiler / rio-tiler resampling semantics: when the source CRS matches the target CRS,
/// `resampling` applies; when reprojecting across CRSes, `reproject` applies (both default to
/// `Nearest`). Uses `engine.get_or_build_crs` (cached) to avoid rebuilding `CrsModel` per tile.
fn resolve_resample(
    engine: Option<&TileEngine>,
    pq: &ParsedQuery,
    source_cog: Option<&cogzilla_io::OpenedCog>,
    target_epsg: u16,
) -> ResampleMode {
    let Some(cog) = source_cog else {
        return pq.resample;
    };
    let gk = &cog.index.geo_keys;
    let is_projected = gk.model_type == Some(1)
        || gk.projected_cs_type.is_some()
        || gk.proj_coord_trans.is_some();
    if (target_epsg != 32767 && gk.projected_cs_type == Some(target_epsg))
        || (!is_projected && gk.geographic_type == Some(target_epsg))
    {
        return pq.resample;
    }
    let crs_res = if let Some(eng) = engine {
        eng.get_or_build_crs(cog).ok()
    } else {
        cogzilla_proj::CrsModel::from_geo_keys(gk).ok()
    };
    let same_crs = match crs_res {
        Some(cogzilla_proj::CrsModel::Epsg3857) => target_epsg == 3857,
        Some(ref c) if c.is_wgs84_degrees() => cogzilla_proj::CrsModel::from_epsg(target_epsg)
            .is_ok_and(|t| t.is_wgs84_degrees()),
        _ => false,
    };
    if same_crs { pq.resample } else { pq.reproject }
}

fn build_ogc_tileset_json(
    host: &str,
    forwarded_prefix: &str,
    route_prefix: &str,
    tms_def: &crate::tms::TmsDef,
    raw_query: &str,
    bounds: Option<[f64; 4]>,
    minzoom: Option<u8>,
    maxzoom: Option<u8>,
) -> serde_json::Value {
    let suffix = if !forwarded_prefix.is_empty() && !host.ends_with(forwarded_prefix) {
        forwarded_prefix
    } else {
        ""
    };
    let origin = if host.starts_with("http://") || host.starts_with("https://") {
        format!("{host}{suffix}")
    } else {
        format!("http://{host}{suffix}")
    };
    let q_suffix = if raw_query.is_empty() {
        String::new()
    } else {
        format!("?{raw_query}")
    };
    let tile_q: String = filter_tile_template_query(raw_query)
        .split('&')
        .filter(|pair| !pair.is_empty() && query_key(pair.split('=').next().unwrap_or("")) != "crs")
        .collect::<Vec<_>>()
        .join("&");
    let tile_q_suffix = if tile_q.is_empty() {
        String::new()
    } else {
        format!("?{tile_q}")
    };
    let [w, s, e, n] = bounds.unwrap_or([-180.0, -90.0, 180.0, 90.0]);
    let limits: Vec<serde_json::Value> = tms_def
        .matrices
        .iter()
        .filter(|m| {
            let Ok(z) = m.id.parse::<u8>() else {
                return false;
            };
            minzoom.is_none_or(|mn| z >= mn) && maxzoom.is_none_or(|mx| z <= mx)
        })
        .map(|m| {
            let (min_col, max_col, min_row, max_row) =
                tms_def.tile_limits_for_wgs84_bounds(m, [w, s, e, n]);
            serde_json::json!({
                "tileMatrix": m.id,
                "minTileRow": min_row,
                "maxTileRow": max_row,
                "minTileCol": min_col,
                "maxTileCol": max_col,
            })
        })
        .collect();
    serde_json::json!({
        "title": tms_def.id,
        "id": tms_def.id,
        "tileMatrixSetId": tms_def.id,
        "tileMatrixSetURI": format!("http://www.opengis.net/def/tilematrixset/OGC/1.0/{}", tms_def.id),
        "dataType": "map",
        "crs": tms_def.crs_uri,
        "accessConstraints": "unclassified",
        "boundingBox": {
            "lowerLeft": [w, s],
            "upperRight": [e, n],
            "crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84",
        },
        "tileMatrixSetLimits": limits,
        "links": [
            {
                "href": format!("{origin}/{route_prefix}/tiles/{}{q_suffix}", tms_def.id),
                "rel": "self",
                "type": "application/json",
                "title": tms_def.id,
            },
            {
                "href": format!("{origin}/tileMatrixSets/{}", tms_def.id),
                "rel": "http://www.opengis.net/def/rel/ogc/1.0/tiling-scheme",
                "type": "application/json",
                "title": format!("Definition of '{}' tile matrix set", tms_def.id),
            },
            {
                "href": format!("{origin}/{route_prefix}/tiles/{}/{{tileMatrix}}/{{tileCol}}/{{tileRow}}{tile_q_suffix}", tms_def.id),
                "rel": "item",
                "type": "image/png",
                "title": "Tiles",
                "templated": true,
            },
        ],
    })
}

fn build_ogc_tilesets_list(
    host: &str,
    forwarded_prefix: &str,
    route_prefix: &str,
    raw_query: &str,
    bounds: Option<[f64; 4]>,
) -> serde_json::Value {
    let tilesets: Vec<serde_json::Value> = crate::tms::ALL_TMS
        .iter()
        .map(|t| {
            build_ogc_tileset_json(
                host,
                forwarded_prefix,
                route_prefix,
                t,
                raw_query,
                bounds,
                None,
                None,
            )
        })
        .collect();
    serde_json::json!({ "tilesets": tilesets })
}

/// A 405 response with an optional `Allow` header, shared across all route families.
fn method_not_allowed_opt(allow: Option<&str>) -> Response<Full<Bytes>> {
    let body = serde_json::json!({ "detail": "Method Not Allowed" }).to_string();
    let mut resp = plain_resp(
        StatusCode::METHOD_NOT_ALLOWED,
        Bytes::from(body),
        "application/json",
    );
    if let Some(Ok(hv)) = allow.map(HeaderValue::from_str) {
        resp.headers_mut().insert(hyper::header::ALLOW, hv);
    }
    resp
}

fn method_not_allowed_resp(allow: &'static str) -> Response<Full<Bytes>> {
    method_not_allowed_opt(Some(allow))
}

fn strip_null_object_fields(mut v: serde_json::Value) -> serde_json::Value {
    if let Some(obj) = v.as_object_mut() {
        obj.retain(|_, val| !val.is_null());
    }
    v
}

fn utc_now_rfc3339() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    // Howard Hinnant's civil-from-days algorithm
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    let hh = rem / 3600;
    let mm = (rem % 3600) / 60;
    let ss = rem % 60;
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = (yoe as i64) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}Z")
}

fn percent_encode_component(s: &str) -> String {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    let mut out = String::with_capacity(s.len());
    for &b in s.as_bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
            out.push(b as char);
        } else {
            out.push('%');
            out.push(HEX[(b >> 4) as usize] as char);
            out.push(HEX[(b & 0x0F) as usize] as char);
        }
    }
    out
}

fn build_stac_render_entry(
    host: &str,
    forwarded_prefix: &str,
    stac_url: &str,
    render_id: &str,
    render: &serde_json::Value,
) -> serde_json::Value {
    let suffix = if !forwarded_prefix.is_empty() && !host.ends_with(forwarded_prefix) {
        forwarded_prefix
    } else {
        ""
    };
    let origin = if host.starts_with("http://") || host.starts_with("https://") {
        format!("{host}{suffix}")
    } else {
        format!("http://{host}{suffix}")
    };
    let enc_url = percent_encode_component(stac_url);
    let enc_rid = percent_encode_component(render_id);
    let mut qs_parts = vec![format!("url={enc_url}")];
    if let Some(obj) = render.as_object() {
        for (k, v) in obj {
            let ek = percent_encode_component(k);
            match k.as_str() {
                "title" | "minmax_zoom" | "tilematrixsets" => continue,
                "assets" => {
                    if let Some(arr) = v.as_array() {
                        for a in arr {
                            if let Some(s) = a.as_str() {
                                qs_parts.push(format!("assets={}", percent_encode_component(s)));
                            }
                        }
                    }
                }
                "rescale" => {
                    if let Some(arr) = v.as_array() {
                        for pair in arr {
                            if let Some(p) = pair.as_array()
                                && p.len() == 2
                            {
                                qs_parts.push(format!("rescale={},{}", p[0], p[1]));
                            }
                        }
                    }
                }
                "bidx" => {
                    if let Some(arr) = v.as_array() {
                        for b in arr {
                            qs_parts.push(format!("bidx={b}"));
                        }
                    }
                }
                "colormap" => match v {
                    serde_json::Value::String(s) => {
                        qs_parts.push(format!("colormap={}", percent_encode_component(s)));
                    }
                    serde_json::Value::Object(_) | serde_json::Value::Array(_) => {
                        qs_parts.push(format!(
                            "colormap={}",
                            percent_encode_component(&v.to_string())
                        ));
                    }
                    _ => {}
                },
                _ => match v {
                    serde_json::Value::String(s) => {
                        qs_parts.push(format!("{ek}={}", percent_encode_component(s)));
                    }
                    serde_json::Value::Number(n) => qs_parts.push(format!("{ek}={n}")),
                    serde_json::Value::Bool(b) => qs_parts.push(format!("{ek}={b}")),
                    _ => {}
                },
            }
        }
    }
    let render_qs = qs_parts.join("&");
    serde_json::json!({
        "valid": true,
        "params": render,
        "links": [
            {
                "rel": "self",
                "type": "application/json",
                "href": format!("{origin}/stac/renders/{enc_rid}?url={enc_url}"),
            },
            {
                "rel": "tilesets-map",
                "type": "application/json",
                "title": "TileJSON",
                "href": format!("{origin}/stac/WebMercatorQuad/tilejson.json?{render_qs}"),
            },
            {
                "rel": "tilesets-map",
                "type": "application/xml",
                "title": "WMTS",
                "href": format!("{origin}/stac/WebMercatorQuad/WMTSCapabilities.xml?{render_qs}"),
            },
        ],
    })
}

fn data_band_count(ifd: &cogzilla_tiff::IfdMeta) -> u16 {
    if cogzilla_codec::alpha_channel_index(ifd.samples_per_pixel as usize, &ifd.extra_samples)
        .is_some()
    {
        ifd.samples_per_pixel.saturating_sub(1).max(1)
    } else {
        ifd.samples_per_pixel
    }
}

fn effective_bbox_max_size(pq: &ParsedQuery) -> u32 {
    if pq.keys.iter().any(|k| k == "max_size") {
        pq.max_size
    } else {
        // TiTiler defaults max_size=None on /bbox and /feature (full native resolution, capped at 4096)
        4096
    }
}

fn maybe_upgrade_multiband_default(
    bands: BandRequest,
    fmt: OutputFormat,
    ct: &str,
    pq: &ParsedQuery,
    default_bands: u16,
) -> BandRequest {
    if pq.bidx.is_empty()
        && pq.expr_raw.is_none()
        && pq.algorithm.is_none()
        && (default_bands == 2 || (4..=15).contains(&default_bands))
        && (matches!(fmt, OutputFormat::Npy | OutputFormat::RawF32)
            || (ct == GEOTIFF_CONTENT_TYPE
                && pq.rescale.is_empty()
                && !pq.colormap_explicit
                && pq.custom_colormap_json.is_none()
                && pq.color_formula.is_none()))
    {
        return BandRequest::MultiBand {
            bands_1based: (1..=default_bands).collect(),
        };
    }
    bands
}

/// TiTiler's `cast`, `ceil` and `floor` keep every band. An array of other than one or
/// three bands (which are band math), on a route that writes arrays in their type
/// (`apply_array_asked`), reads them as they are, and each value is rounded and wrapped as
/// it is typed (`ArrayRule::rounding`).
fn keep_algorithm_bands(
    bands: BandRequest,
    fmt: OutputFormat,
    ct: &str,
    pq: &ParsedQuery,
    default_bands: u16,
) -> BandRequest {
    let array = fmt == OutputFormat::Npy || ct == GEOTIFF_CONTENT_TYPE;
    let asked = if pq.bidx.is_empty() { usize::from(default_bands) } else { pq.bidx.len() };
    if matches!(pq.algorithm.as_deref(), Some("cast" | "ceil" | "floor"))
        && pq.expr_raw.is_none()
        && array
        && !array_type::shown(pq)
        && pq.asset_exprs.is_empty()
        && (asked == 2 || (4..=15).contains(&asked))
    {
        let bands_1based = if pq.bidx.is_empty() { (1..=default_bands).collect() } else { pq.bidx.clone() };
        return BandRequest::MultiBand { bands_1based };
    }
    bands
}

/// The bands of a request on a route that writes arrays in their type (`/cog`, `/stac`,
/// `/mosaicjson`).
fn build_band_request_for_output(
    pq: &ParsedQuery,
    default_bands: u16,
    sample_type: cogzilla_raster::SampleType,
    fmt: OutputFormat,
    ct: &str,
) -> Result<BandRequest, EngineError> {
    let bands = build_band_request(pq, default_bands, sample_type)?;
    let bands = maybe_upgrade_multiband_default(bands, fmt, ct, pq, default_bands);
    Ok(keep_algorithm_bands(bands, fmt, ct, pq, default_bands))
}

fn build_tile_request(
    engine: Option<&TileEngine>,
    tile: TileCoord,
    scale_opt: Option<u32>,
    ext_opt: Option<&str>,
    pq: &ParsedQuery,
    default_bands: u16,
    sample_type: cogzilla_raster::SampleType,
    is_crs84: bool,
    source_cog: Option<&cogzilla_io::OpenedCog>,
) -> Result<(TileRequest, &'static str), EngineError> {
    let tms_id = if is_crs84 {
        "WorldCRS84Quad"
    } else {
        "WebMercatorQuad"
    };
    build_tile_request_for_tms(
        engine,
        tile,
        scale_opt,
        ext_opt,
        pq,
        default_bands,
        sample_type,
        tms_id,
        source_cog,
        false,
    )
}

/// A tile request; `typed_arrays` when the route writes its arrays in their type
/// (`apply_array_asked`), which some algorithms' bands rely on ([`keep_algorithm_bands`]).
#[allow(clippy::too_many_arguments)]
fn build_tile_request_for_tms(
    engine: Option<&TileEngine>,
    tile: TileCoord,
    scale_opt: Option<u32>,
    ext_opt: Option<&str>,
    pq: &ParsedQuery,
    default_bands: u16,
    sample_type: cogzilla_raster::SampleType,
    tms_id: &str,
    source_cog: Option<&cogzilla_io::OpenedCog>,
    typed_arrays: bool,
) -> Result<(TileRequest, &'static str), EngineError> {
    let (cols, rows, target_epsg, base_tile_size) =
        if let Some(tms_def) = crate::tms::lookup_tms(tms_id) {
            let Some(m) = tms_def.matrix(tile.z) else {
                return Err(EngineError::BadRequest(format!(
                    "zoom level z={} is not defined for tile matrix set '{}'",
                    tile.z, tms_id
                )));
            };
            (m.matrix_width, m.matrix_height, tms_def.epsg, m.tile_width)
        } else {
            if tile.z > 30 {
                return Err(EngineError::BadRequest(format!(
                    "zoom level z={} exceeds maximum supported zoom (30)",
                    tile.z
                )));
            }
            let rows = 1u64 << (tile.z as u32);
            (rows, rows, 3857u16, 256u32)
        };
    if (tile.x as u64) >= cols || (tile.y as u64) >= rows {
        return Err(EngineError::BadRequest(format!(
            "tile coordinate ({}, {}) out of range for {tms_id} z={} (expected x < {cols}, y < {rows})",
            tile.x, tile.y, tile.z
        )));
    }
    let eff_ext = ext_opt.or(pq.format_param.as_deref()).unwrap_or("png");
    let (format, content_type) = parse_ext_format(eff_ext, pq.webp_quality, pq.jpeg_quality)?;
    let eff_default_bands = match source_cog {
        Some(c) if default_bands == c.index.full_res().samples_per_pixel => {
            data_band_count(c.index.full_res())
        }
        _ => default_bands,
    };
    let bands = if typed_arrays {
        build_band_request_for_output(pq, eff_default_bands, sample_type, format, content_type)?
    } else {
        let bands = build_band_request(pq, eff_default_bands, sample_type)?;
        maybe_upgrade_multiband_default(bands, format, content_type, pq, eff_default_bands)
    };
    let format = geotiff_render_format(format, content_type, pq, source_cog, &bands, sample_type, false);
    if !matches!(format, OutputFormat::RawF32 | OutputFormat::RawF64 | OutputFormat::Npy | OutputFormat::NpyF64) {
        match bands {
            BandRequest::MultiBand { ref bands_1based }
                if !(pq.pixel_selection.by_last_band()
                    && matches!(bands_1based.len(), 2 | 4)) =>
            {
                return Err(EngineError::BadRequest(
                    "multi-band bidx (2 or 4..=15 bands) is only supported for .tif, .npy, or .f32 output"
                        .into(),
                ));
            }
            BandRequest::CustomExprMulti { ref progs }
                if !(pq.pixel_selection.by_last_band() && matches!(progs.len(), 2 | 4)) =>
            {
                return Err(EngineError::BadRequest(format!(
                    "unsupported expression count {} for visual image output (expected 1 or 3; use .tif, .npy, or .f32)",
                    progs.len()
                )));
            }
            _ => {}
        }
    }
    let has_explicit_tile_size = pq
        .keys
        .iter()
        .any(|k| matches!(k.as_str(), "tile_size" | "tilesize" | "size" | "scale"));
    let tile_size = match scale_opt {
        Some(sc) if (1..=4).contains(&sc) => {
            let sz = sc.saturating_mul(base_tile_size);
            if sz > 4096 {
                return Err(EngineError::BadRequest(format!(
                    "scaled tile size {sz} exceeds maximum (4096) for {tms_id}"
                )));
            }
            sz
        }
        Some(other) => {
            return Err(EngineError::BadRequest(format!(
                "invalid tile scale @{other}x (expected 1..=4)"
            )));
        }
        None => {
            if has_explicit_tile_size {
                pq.tile_size
            } else {
                base_tile_size
            }
        }
    };
    let buffered_dim = (tile_size as f32 + 2.0 * pq.buffer).round() as u32;
    if buffered_dim > 4096 {
        return Err(EngineError::BadRequest(format!(
            "buffered tile size {buffered_dim} (tile_size={tile_size}, buffer={}) exceeds maximum (4096) for {tms_id}",
            pq.buffer
        )));
    }
    Ok((
        TileRequest {
            tile,
            tile_size,
            bands,
            resample: resolve_resample(engine, pq, source_cog, target_epsg),
            format,
            unscale: pq.unscale,
        },
        content_type,
    ))
}

/// The range of an integer sample type (`None` for floats), for the statistics read.
pub(crate) fn array_type_range(kind: cogzilla_raster::SampleType) -> Option<(f64, f64)> {
    kind.storable_range()
}

/// The type that holds every band of `cogs` as read.
fn reads_type(cogs: &[Arc<cogzilla_io::OpenedCog>]) -> cogzilla_raster::SampleType {
    cogs.iter()
        .map(|cog| cog.index.full_res().sample_type)
        .reduce(array_type::wider_type)
        .unwrap_or(cogzilla_raster::SampleType::Float32)
}

/// The statistics' band math, each output's description and expression compiled, with the
/// request's band-math `algorithm` worked over the outputs ([`algorithm_math::over_results`]).
///
/// Also gives the histogram's edge type when an algorithm was worked over the band math: that
/// of the algorithm's output (float32 for `normalizedIndex`), not one read from the text.
fn compile_stats_parts(
    pq: &ParsedQuery,
    parts: Vec<(String, String)>,
    read: Option<cogzilla_raster::SampleType>,
) -> Result<(Vec<(String, ExprProgram)>, Option<crate::stats::histogram::HistDtype>), EngineError> {
    if parts.len() > 16 {
        return Err(EngineError::BadRequest("expression exceeds max 16 semicolon-separated outputs".into()));
    }
    let typed = band_math_output_type(pq, &parts, read);
    let composed = match pq.algorithm.as_deref() {
        Some(alg) => algorithm_math::over_results(alg, typed, pq.cast_dtype.as_deref(), pq.threshold, &parts)?,
        None => None,
    };
    let (parts, max_ops, hist) = match composed {
        Some(c) => (c.outputs, c.max_ops, c.kind.map(|kind| crate::stats::histogram::HistDtype::for_sample_type(kind, false))),
        None => (parts, ExprProgram::MAX_OPS, None),
    };
    let compiled = parts
        .into_iter()
        .map(|(desc, text)| Ok((desc, ExprProgram::compile_with_limit(&text, max_ops).map_err(EngineError::BadRequest)?)))
        .collect::<Result<Vec<_>, EngineError>>()?;
    Ok((compiled, hist))
}

fn build_stats_options(pq: &ParsedQuery, allow_expr: bool) -> Result<StatsOptions, EngineError> {
    build_stats_options_with_bands(pq, allow_expr, 0, None)
}

/// The statistics' options; `read` is the type of the bands, when it is known.
fn build_stats_options_with_bands(
    pq: &ParsedQuery,
    allow_expr: bool,
    default_bands: u16,
    read: Option<cogzilla_raster::SampleType>,
) -> Result<StatsOptions, EngineError> {
    let mut opts = StatsOptions { resample: pq.resample, ..StatsOptions::default() };
    if !pq.percentiles.is_empty() {
        opts.percentiles = pq.percentiles.clone();
    }
    if let Some(bins) = pq.histogram_bins {
        opts.histogram_bins = bins;
    }
    opts.histogram_edges = pq.histogram_edges.clone();
    opts.histogram_range = pq.histogram_range;
    opts.categorical = pq.categorical;
    if !pq.categories.is_empty() {
        opts.categories = Some(pq.categories.clone());
    }
    if !pq.bidx.is_empty() {
        opts.bidx = Some(pq.bidx.clone());
    }
    opts.width = pq.width;
    opts.height = pq.height;
    opts.max_size_explicit = !pq.max_size_zero && pq.keys.iter().any(|k| k == "max_size");
    opts.area_weighted = pq.area_weighted;
    if allow_expr
        && let Some(raw) = pq.expr_raw.as_deref()
    {
        let parts: Vec<(String, String)> = expression_parts(raw).map(|p| (p.to_string(), p.to_string())).collect();
        let (compiled, hist) = compile_stats_parts(pq, parts, read)?;
        if !compiled.is_empty() {
            opts.expressions = Some(compiled);
        }
        if hist.is_some() {
            opts.hist_dtype = hist;
        }
    } else if allow_expr
        && let Some(alg_raw) = pq.algorithm.as_deref()
    {
        let bands: Vec<u16> = if !pq.bidx.is_empty() {
            pq.bidx.clone()
        } else {
            (1..=default_bands.clamp(1, 16)).collect()
        };
        let alg = alg_raw.to_ascii_lowercase();
        // The bands' type is not known here: they are taken as possibly fractional.
        let expr_str = algorithm_math::band_math(&alg, &bands, None, pq.cast_dtype.as_deref(), pq.threshold)
            .map_err(|e| match alg.as_str() {
                "normalizedindex" => EngineError::BadRequest(e),
                _ => EngineError::BadRequest(format!("algorithm '{alg_raw}' is not supported on statistics routes")),
            })?;
        let prog = ExprProgram::compile(&expr_str).map_err(EngineError::BadRequest)?;
        opts.expressions = Some(vec![(alg_raw.to_string(), prog)]);
    }
    Ok(opts)
}

/// Apply optional `?expression=` (supporting `;`-separated multi-band outputs) or `?bidx=`
/// filtering to a TiTiler `/point` JSON payload.
fn apply_point_filters(
    pt: &mut serde_json::Value,
    expr_raw: Option<&str>,
    bidx: &[u16],
) -> Result<(), EngineError> {
    let Some(vals) = pt.get("values").and_then(|v| v.as_array()).cloned() else {
        return Ok(());
    };
    let names: Vec<serde_json::Value> = pt
        .get("band_names")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let descs: Vec<serde_json::Value> = pt
        .get("band_descriptions")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_else(|| names.clone());

    if let Some(raw_expr) = expr_raw {
        let parts: Vec<&str> = raw_expr
            .split(';')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();
        if parts.is_empty() {
            return Err(EngineError::BadRequest("empty expression".into()));
        }
        if parts.len() > 16 {
            return Err(EngineError::BadRequest(
                "expression exceeds max 16 semicolon-separated outputs".into(),
            ));
        }
        let n_bands = vals.len();
        let mut out_vals = Vec::with_capacity(parts.len());
        let mut out_names = Vec::with_capacity(parts.len());
        for expr in parts {
            let prog = ExprProgram::compile(expr).map_err(EngineError::BadRequest)?;
            let mut slot_vals = Vec::with_capacity(prog.used_bands.len());
            for &b in &prog.used_bands {
                if b == 0 || (b as usize) > n_bands {
                    return Err(EngineError::BadRequest(format!(
                        "expression references b{b} but point query only has {n_bands} bands"
                    )));
                }
                let f = vals[(b - 1) as usize]
                    .as_f64()
                    .map(|x| x as f32)
                    .unwrap_or(f32::NAN);
                slot_vals.push(f);
            }
            let res = prog.eval(&slot_vals);
            out_vals.push(if res.is_finite() {
                serde_json::json!(res)
            } else {
                serde_json::Value::Null
            });
            out_names.push(serde_json::json!(expr));
        }
        let band_names: Vec<serde_json::Value> = (1..=out_vals.len())
            .map(|i| serde_json::json!(format!("b{i}")))
            .collect();
        pt["values"] = serde_json::Value::Array(out_vals);
        pt["band_names"] = serde_json::Value::Array(band_names);
        pt["band_descriptions"] = serde_json::Value::Array(out_names);
        return Ok(());
    }

    if !bidx.is_empty() {
        let n_bands = vals.len();
        let mut f_vals = Vec::with_capacity(bidx.len());
        let mut f_names = Vec::with_capacity(bidx.len());
        let mut f_descs = Vec::with_capacity(bidx.len());
        for &b in bidx {
            if b == 0 || (b as usize) > n_bands {
                return Err(EngineError::BadRequest(format!(
                    "band {b} out of range (dataset has {n_bands} bands)"
                )));
            }
            let idx = (b - 1) as usize;
            f_vals.push(vals[idx].clone());
            f_names.push(names.get(idx).cloned().unwrap_or_else(|| serde_json::json!(format!("b{b}"))));
            f_descs.push(descs.get(idx).cloned().unwrap_or_else(|| serde_json::json!(format!("b{b}"))));
        }
        pt["values"] = serde_json::Value::Array(f_vals);
        pt["band_names"] = serde_json::Value::Array(f_names);
        pt["band_descriptions"] = serde_json::Value::Array(f_descs);
    }
    Ok(())
}

fn validate_mosaic_tile_params(_pq: &ParsedQuery, tile_size: u32) -> Result<(), EngineError> {
    if !(64..=2048).contains(&tile_size) {
        return Err(EngineError::BadRequest(format!(
            "mosaic tile_size must be in 64..=2048 (got {tile_size})"
        )));
    }
    Ok(())
}

fn default_raw_band_count(bands: &BandRequest, pixel_selection: Selection) -> usize {
    if pixel_selection == Selection::Count {
        return 1;
    }
    let raw_nb = match bands {
        BandRequest::Rgb { .. }
        | BandRequest::SarRgb { .. }
        | BandRequest::Terrain { .. }
        | BandRequest::CustomExprRgb { .. } => 3,
        BandRequest::MultiBand { bands_1based } => bands_1based.len().max(1),
        BandRequest::CustomExprMulti { progs } => progs.len().max(1),
        _ => 1,
    };
    if pixel_selection.by_last_band() {
        raw_nb.saturating_sub(1).max(1)
    } else {
        raw_nb
    }
}

/// Hard memory budget (64 MiB) for `pixel_selection=median`, which must buffer
/// per-scene planes in order to compute exact per-pixel medians.
const MAX_MEDIAN_BUFFER_BYTES: usize = 64 * 1024 * 1024;

async fn render_mosaic_tile_pixel_selection(
    engine: &TileEngine,
    urls: &[String],
    treq: TileRequest,
    pq: &ParsedQuery,
    failed_out: &mut Vec<String>,
) -> Result<Bytes, EngineError> {
    let urls = &urls[..urls.len().min(64)];
    let out_dim = (treq.tile_size as f32 + 2.0 * pq.buffer).round() as u32;
    let bbox_3857 = web_mercator_buffered_3857_bbox(
        treq.tile.z,
        treq.tile.x,
        treq.tile.y,
        pq.buffer,
        treq.tile_size,
    );
    let n_px = (out_dim as usize) * (out_dim as usize);
    let mut first_lut: Option<cogzilla_kernel::CustomColormap> = None;

    // Only `median` holds its scenes, within MAX_MEDIAN_BUFFER_BYTES; the other methods
    // keep a running figure for each value.
    let method = pq.pixel_selection;
    let mut fold: Option<Fold> = None;
    let mut taken = 0usize;
    let mut any_ok = false;
    let mut last_err: Option<EngineError> = None;

    for (scene_i, url) in urls.iter().enumerate() {
        if scene_i > 0 {
            tokio::task::consume_budget().await;
        }
        let cog = match engine.store.open_uri(url).await.map(|c| engine.store.with_nodata(&c, pq.nodata)) {
            Ok(c) => c,
            Err(e) if e.is_fatal() => {
                return Err(EngineError::Io(e));
            }
            Err(e) => {
                failed_out.push(url.clone());
                last_err = Some(crate::engine::asset_fetch_error(url, e));
                continue;
            }
        };
        let cogs = vec![cog];
        let mut scene_bands = treq.bands.clone();
        let data_bands = data_band_count(cogs[0].index.full_res());
        if method.by_last_band()
            && pq.bidx.is_empty()
            && pq.expr_raw.is_none()
            && pq.algorithm.is_none()
            && data_bands >= 2
            && matches!(scene_bands, BandRequest::Single { .. } | BandRequest::Rgb { .. })
        {
            scene_bands = BandRequest::MultiBand {
                bands_1based: (1..=data_bands).collect(),
            };
        }
        if pq.unscale {
            apply_unscale_cogs(&mut scene_bands, &cogs, None);
        }
        if first_lut.is_none()
            && !(matches!(method, Selection::Count | Selection::Stdev)
                && pq.custom_colormap_json.is_none())
        {
            first_lut = resolve_custom_lut(pq, &cogs[0], &scene_bands)?;
        }
        let resample = resolve_resample(Some(engine), pq, Some(&cogs[0]), 3857);
        let raw = match render_mercator_bbox(
            engine,
            &cogs,
            bbox_3857,
            out_dim,
            out_dim,
            &scene_bands,
            resample,
            OutputFormat::RawF32,
            pq.nodata,
            None,
            None,
            pq.return_mask,
            pq.unscale,
        )
        .await
        {
            Ok(b) => b,
            Err(EngineError::NotFound(_)) => {
                any_ok = true;
                continue;
            }
            Err(EngineError::Io(e))
                if e.is_fatal() =>
            {
                return Err(EngineError::Io(e));
            }
            Err(e) => {
                failed_out.push(url.clone());
                last_err = Some(e);
                continue;
            }
        };
        any_ok = true;
        if n_px == 0 || raw.len() % (n_px * 4) != 0 {
            continue;
        }
        let nb = raw.len() / (n_px * 4);
        if nb == 0 {
            continue;
        }
        let fold = match &mut fold {
            Some(fold) => fold,
            None => {
                if method.by_last_band() && nb < 2 {
                    return Err(EngineError::BadRequest(
                        "pixel_selection by the last band needs at least two bands".into(),
                    ));
                }
                if Fold::<f32>::working_bytes(method, nb, n_px) > MAX_MEDIAN_BUFFER_BYTES {
                    return Err(EngineError::BadRequest(format!(
                        "pixel_selection working buffer exceeds memory budget ({} MiB limit); reduce tile_size, buffer, or band count",
                        MAX_MEDIAN_BUFFER_BYTES / (1024 * 1024)
                    )));
                }
                let integer =
                    reads_whole_numbers(pq.unscale, &scene_bands, cogs[0].index.full_res().sample_type);
                let absent = pq.nodata.or(cogs[0].index.nodata).unwrap_or(0.0);
                fold.insert(Fold::new(method, nb, n_px, integer).absent_as(absent))
            }
        };
        if fold.bands() != nb {
            continue;
        }
        taken += 1;
        if Fold::<f32>::held_bytes(method, taken, nb, n_px) > MAX_MEDIAN_BUFFER_BYTES {
            return Err(EngineError::BadRequest(format!(
                "pixel_selection=median exceeds memory budget ({} MiB limit); reduce tile_size, buffer, or band count",
                MAX_MEDIAN_BUFFER_BYTES / (1024 * 1024)
            )));
        }
        fold.add_le_bytes(&raw);
    }

    if !any_ok
        && let Some(err) = last_err
    {
        return Err(err);
    }
    let gpu_selected: Option<Selected<f32>> = if let Some(gpu) = engine.accelerated()
        && let Some(gpu_method) = method.gpu()
        && let Some(ref f) = fold
        && let Some((nb, scenes)) = f.held_scenes_for_gpu()
        && scenes.len() <= 32
    {
        let band_planes: Vec<Vec<&[f32]>> = (0..nb)
            .map(|b| {
                let b_off = b * n_px;
                scenes.iter().map(|s| &s[b_off..b_off + n_px]).collect()
            })
            .collect();
        gpu.try_composite_mosaic(&crate::gpu_backend::MosaicCompositeRequest {
            width: out_dim,
            height: out_dim,
            input: crate::gpu_backend::MosaicCompositeInput::Bands(&band_planes),
            method: gpu_method,
            output: crate::gpu_backend::MosaicCompositeOutput::F32,
        })
        .and_then(crate::gpu_backend::MosaicCompositeResult::into_f32_bands)
        .map(|gpu_bands| {
            engine
                .stats
                .gpu_dispatches
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let mut values = Vec::with_capacity(nb * n_px);
            for band_out in gpu_bands {
                values.extend_from_slice(&band_out);
            }
            Selected { bands: nb, values }
        })
    } else {
        None
    };

    let offload_cpu = gpu_selected.is_none()
        && fold.is_some()
        && n_px >= 256 * 256
        && (matches!(method, Selection::Median) || n_px >= 512 * 512);
    crate::engine::block_in_place_if_multithread(offload_cpu, || {
        let selected = gpu_selected
            .or_else(|| fold.and_then(Fold::finish))
            .unwrap_or_else(|| {
                let bands = default_raw_band_count(&treq.bands, method);
                Selected { bands, values: vec![f32::NAN; bands * n_px] }
            });

        if matches!(treq.format, OutputFormat::RawF32 | OutputFormat::Npy) {
            let mut out = bytes::BytesMut::with_capacity(selected.values.len() * 4);
            for v in &selected.values {
                out.extend_from_slice(&v.to_le_bytes());
            }
            let raw = out.freeze();
            if matches!(treq.format, OutputFormat::Npy) {
                return Ok(cogzilla_kernel::wrap_raw_f32_as_npy_with_mask(
                    raw,
                    selected.bands,
                    out_dim,
                    out_dim,
                    pq.return_mask,
                ));
            }
            return Ok(raw);
        }

        let rgba = paint_selected(selected, n_px, &treq.bands, pq, first_lut.as_ref())?;
        if let Some(gpu) = engine.accelerated() {
            return Ok(gpu.encode_rgba_output(
                out_dim as usize,
                out_dim as usize,
                &rgba,
                treq.format,
                pq.return_mask,
            ));
        }
        Ok(encode_rgba_output(
            out_dim as usize,
            out_dim as usize,
            &rgba,
            treq.format,
            pq.return_mask,
        ))
    })
}

/// The picture of what a pixel selection left: `pixels` RGBA pixels, by the rules in
/// [`paint`]. Three bands or more are red, green and blue (the first three); fewer are the
/// first band under the request's colour map. Kept from the single-scene renderer, where
/// rio-tiler has no equivalent or does otherwise: a raster's own palette and a colour map
/// given as JSON (`custom`, which the kernel looks up from the value). A range given
/// upside down is the ramp inverted, which is `rescale_byte`'s doing.
fn paint_selected(
    mut selected: Selected,
    pixels: usize,
    bands: &BandRequest,
    pq: &ParsedQuery,
    custom: Option<&CustomColormap>,
) -> Result<Vec<u8>, EngineError> {
    let formula = pq.parse_color_formula()?;
    let three = selected.bands >= 3;
    // The request's own ranges and colour map where it has them for this many bands;
    // otherwise bytes as they are.
    let paint = match (Paint::of(bands), bands) {
        (Some(paint @ Paint::Rgb { .. }), _) if three => paint,
        (Some(paint @ Paint::Gray { .. }), _) if !three => paint,
        (_, BandRequest::CustomExprRgb { rescale_min, rescale_max, .. }) if three => {
            Paint::Rgb { min: *rescale_min, max: *rescale_max }
        }
        (_, BandRequest::MultiBand { .. }) if three && !pq.rescale.is_empty() => {
            let r0 = pq.rescale[0];
            let r1 = pq.rescale.get(1).copied().unwrap_or(r0);
            let r2 = pq.rescale.get(2).copied().unwrap_or(r0);
            Paint::Rgb { min: [r0.0, r1.0, r2.0], max: [r0.1, r1.1, r2.1] }
        }
        (_, BandRequest::MultiBand { .. }) if !three && !pq.rescale.is_empty() => {
            let (min, max) = pq.rescale[0];
            Paint::Gray { min, max, colormap: pq.colormap }
        }
        _ if three => Paint::Rgb { min: [0.0; 3], max: [255.0; 3] },
        _ => Paint::Gray { min: 0.0, max: 255.0, colormap: pq.colormap },
    };
    if three {
        selected.bands = 3;
        let mut rgba = paint.rgba(&selected, pixels);
        if let Some(formula) = &formula {
            formula.apply_buffer_in_place(&mut rgba, 4);
        }
        return Ok(rgba);
    }
    if let (Some(custom), Paint::Gray { min, max, .. }) = (custom, paint) {
        let inv = 255.0 / cogzilla_kernel::safe_rescale_denom(min, max);
        let mut rgba = vec![0u8; pixels * 4];
        for (px, v) in rgba.chunks_exact_mut(4).zip(&selected.values) {
            if v.is_finite() {
                px.copy_from_slice(&custom.lookup(*v, min, inv));
            }
        }
        // Under a colour map of the file's or the request's own, the formula is for the
        // finished picture.
        if let Some(formula) = &formula {
            formula.apply_buffer_in_place(&mut rgba, 4);
        }
        return Ok(rgba);
    }
    Ok(match &formula {
        Some(formula) => paint.rgba_through(&selected, pixels, &paint::formula_levels(formula)),
        None => paint.rgba(&selected, pixels),
    })
}

async fn render_mosaic_tile_styled(
    engine: &TileEngine,
    urls: &[String],
    mut treq: TileRequest,
    pq: &ParsedQuery,
    first_has_colormap: bool,
    failed_out: &mut Vec<String>,
) -> Result<Bytes, EngineError> {
    if pq.pixel_selection != Selection::First {
        return render_mosaic_tile_pixel_selection(engine, urls, treq, pq, failed_out).await;
    }
    let mut needs_bbox_mosaic = pq.nodata.is_some()
        || pq.custom_colormap_json.is_some()
        || pq.buffer > 0.0
        || matches!(treq.format, OutputFormat::Npy)
        || (pq.color_formula.is_some()
            && !matches!(
                treq.bands,
                BandRequest::Rgb { .. }
                    | BandRequest::SarRgb { .. }
                    | BandRequest::CustomExprRgb { .. }
                    | BandRequest::Terrain { .. }
            ))
        || (matches!(
            treq.bands,
            BandRequest::CustomExprRgb { .. }
                | BandRequest::CustomExprMulti { .. }
                | BandRequest::MultiBand { .. }
        ) && !matches!(treq.format, OutputFormat::RawF64 | OutputFormat::NpyF64))
        || (treq.tile_size != 256 && treq.tile_size != 512);
    if !needs_bbox_mosaic
        && !pq.colormap_explicit
        && pq.custom_colormap_json.is_none()
        && first_has_colormap
    {
        needs_bbox_mosaic = true;
    }

    if needs_bbox_mosaic {
        let urls = &urls[..urls.len().min(64)];
        let out_dim = (treq.tile_size as f32 + 2.0 * pq.buffer).round() as u32;
        let bbox_3857 = web_mercator_buffered_3857_bbox(
            treq.tile.z,
            treq.tile.x,
            treq.tile.y,
            pq.buffer,
            treq.tile_size,
        );
        let n_px = (out_dim as usize) * (out_dim as usize);
        if matches!(treq.format, OutputFormat::RawF32 | OutputFormat::Npy) {
            let mut acc_f32: Option<(usize, Vec<f32>)> = None;
            let mut filled = 0usize;
            let mut any_ok = false;
            let mut last_err: Option<EngineError> = None;
            for (scene_i, url) in urls.iter().enumerate() {
                if scene_i > 0 {
                    tokio::task::consume_budget().await;
                }
                let cog = match engine.store.open_uri(url).await.map(|c| engine.store.with_nodata(&c, pq.nodata)) {
                    Ok(c) => c,
                    Err(e) if e.is_fatal() => {
                        return Err(EngineError::Io(e));
                    }
                    Err(e) => {
                        failed_out.push(url.clone());
                        last_err = Some(crate::engine::asset_fetch_error(url, e));
                        continue;
                    }
                };
                let cogs = vec![cog];
                let mut scene_bands = treq.bands.clone();
                if pq.unscale {
                    apply_unscale_cogs(&mut scene_bands, &cogs, None);
                }
                let resample = resolve_resample(Some(engine), pq, Some(&cogs[0]), 3857);
                let raw = match render_mercator_bbox(
                    engine,
                    &cogs,
                    bbox_3857,
                    out_dim,
                    out_dim,
                    &scene_bands,
                    resample,
                    OutputFormat::RawF32,
                    pq.nodata,
                    None,
                    None,
                    pq.return_mask,
                    pq.unscale,
                )
                .await
                {
                    Ok(b) => b,
                    Err(EngineError::NotFound(_)) => {
                        any_ok = true;
                        continue;
                    }
                    Err(EngineError::Io(e))
                        if e.is_fatal() =>
                    {
                        return Err(EngineError::Io(e));
                    }
                    Err(e) => {
                        failed_out.push(url.clone());
                        last_err = Some(e);
                        continue;
                    }
                };
                any_ok = true;
                if n_px == 0 || raw.len() % (n_px * 4) != 0 {
                    continue;
                }
                let nb = raw.len() / (n_px * 4);
                if nb == 0 {
                    continue;
                }
                let (acc_nb, acc) = acc_f32.get_or_insert_with(|| (nb, vec![f32::NAN; nb * n_px]));
                if *acc_nb != nb {
                    continue;
                }
                for i in 0..n_px {
                    if acc[i].is_nan() {
                        let mut all_valid = true;
                        for b in 0..nb {
                            let boff = (b * n_px + i) * 4;
                            let v = f32::from_le_bytes([
                                raw[boff],
                                raw[boff + 1],
                                raw[boff + 2],
                                raw[boff + 3],
                            ]);
                            if v.is_nan() {
                                all_valid = false;
                                break;
                            }
                        }
                        if all_valid {
                            for b in 0..nb {
                                let boff = (b * n_px + i) * 4;
                                acc[b * n_px + i] = f32::from_le_bytes([
                                    raw[boff],
                                    raw[boff + 1],
                                    raw[boff + 2],
                                    raw[boff + 3],
                                ]);
                            }
                            filled += 1;
                        }
                    }
                }
                if filled >= n_px {
                    break;
                }
            }
            if !any_ok
                && let Some(err) = last_err
            {
                return Err(err);
            }
            let default_nb = default_raw_band_count(&treq.bands, pq.pixel_selection);
            let (nb, acc) = acc_f32.unwrap_or_else(|| (default_nb, vec![f32::NAN; default_nb * n_px]));
            let mut out = bytes::BytesMut::with_capacity(acc.len() * 4);
            for v in acc {
                out.extend_from_slice(&v.to_le_bytes());
            }
            let raw = out.freeze();
            if matches!(treq.format, OutputFormat::Npy) {
                return Ok(cogzilla_kernel::wrap_raw_f32_as_npy_with_mask(
                    raw,
                    nb,
                    out_dim,
                    out_dim,
                    pq.return_mask,
                ));
            }
            return Ok(raw);
        }
        let cf = pq.parse_color_formula()?;
        let mut comp = vec![0u8; n_px * 4];
        let mut filled = 0usize;
        let mut any_ok = false;
        let mut last_err: Option<EngineError> = None;

        for (scene_i, url) in urls.iter().enumerate() {
            if scene_i > 0 {
                tokio::task::consume_budget().await;
            }
            let cog = match engine.store.open_uri(url).await.map(|c| engine.store.with_nodata(&c, pq.nodata)) {
                Ok(c) => c,
                Err(e) if e.is_fatal() => {
                    return Err(EngineError::Io(e));
                }
                Err(e) => {
                    failed_out.push(url.clone());
                    last_err = Some(crate::engine::asset_fetch_error(url, e));
                    continue;
                }
            };
            let cogs = vec![cog];
            let mut scene_bands = treq.bands.clone();
            if pq.unscale {
                apply_unscale_cogs(&mut scene_bands, &cogs, None);
            }
            let scene_lut = resolve_custom_lut(pq, &cogs[0], &scene_bands)?;
            let resample = resolve_resample(Some(engine), pq, Some(&cogs[0]), 3857);
            let scene_rgba = match render_mercator_bbox(
                engine,
                &cogs,
                bbox_3857,
                out_dim,
                out_dim,
                &scene_bands,
                resample,
                OutputFormat::RawRgba,
                pq.nodata,
                scene_lut.as_ref(),
                cf.as_ref(),
                true,
                pq.unscale,
            )
            .await
            {
                Ok(b) => b,
                Err(EngineError::NotFound(_)) => {
                    any_ok = true;
                    continue;
                }
                Err(EngineError::Io(e))
                    if e.is_fatal() =>
                {
                    return Err(EngineError::Io(e));
                }
                Err(e) => {
                    failed_out.push(url.clone());
                    last_err = Some(e);
                    continue;
                }
            };
            any_ok = true;
            if scene_rgba.len() != n_px * 4 {
                continue;
            }
            for i in 0..n_px {
                let o = i * 4;
                if comp[o + 3] == 0 && scene_rgba[o + 3] > 0 {
                    comp[o..o + 4].copy_from_slice(&scene_rgba[o..o + 4]);
                    filled += 1;
                }
            }
            if filled >= n_px {
                break;
            }
        }
        if !any_ok
            && let Some(err) = last_err
        {
            return Err(err);
        }
        return Ok(encode_rgba_output(
            out_dim as usize,
            out_dim as usize,
            &comp,
            treq.format,
            pq.return_mask,
        ));
    }

    if (pq.color_formula.is_some() || !pq.return_mask)
        && !matches!(treq.format, OutputFormat::RawF32 | OutputFormat::RawF64 | OutputFormat::Npy | OutputFormat::NpyF64)
    {
        let cf = pq.parse_color_formula()?;
        let orig_format = treq.format;
        treq.format = OutputFormat::RawRgba;
        let (raw, failed) = engine.render_mosaic_tile_with_report(urls, &treq).await?;
        failed_out.extend(failed);
        let dim = treq.tile_size as usize;
        let data = if let Some(cf) = cf {
            let mut buf = raw.to_vec();
            cf.apply_buffer_in_place(&mut buf, 4);
            encode_rgba_output(dim, dim, &buf, orig_format, pq.return_mask)
        } else {
            encode_rgba_output(dim, dim, &raw, orig_format, pq.return_mask)
        };
        Ok(data)
    } else {
        let (raw, failed) = engine.render_mosaic_tile_with_report(urls, &treq).await?;
        failed_out.extend(failed);
        Ok(raw)
    }
}

/// The band math of the `nth` time (from 0) `asset_name` is asked for, or of the last time.
fn stac_asset_expr_nth<'a>(pq: &'a ParsedQuery, asset_name: &str, nth: usize) -> Option<&'a str> {
    let mut of_it = pq.asset_exprs.iter().filter(|(n, _)| n == asset_name).map(|(_, e)| e.as_str());
    let last = of_it.clone().next_back();
    of_it.nth(nth).or(last)
}

/// How many times each of `names` came before it in `names`.
fn occurrences(names: &[String]) -> Vec<usize> {
    names.iter().enumerate().map(|(i, n)| names[..i].iter().filter(|m| *m == n).count()).collect()
}

/// `expr` with each band token (`b3`, or `B3` as rio-tiler takes it) replaced by what `to`
/// makes of its number, counted from one; a token `to` has nothing for is left as it is.
fn substitute_band_tokens(expr: &str, mut to: impl FnMut(usize) -> Option<String>) -> String {
    let bytes = expr.as_bytes();
    let mut out = String::with_capacity(expr.len() + 8);
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_ascii_alphabetic() || bytes[i] == b'_' {
            let start = i;
            while i < bytes.len() && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            let ident = &expr[start..i];
            let replaced = ident.strip_prefix(['b', 'B']).filter(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit())).and_then(|n| n.parse().ok()).and_then(&mut to);
            out.push_str(replaced.as_deref().unwrap_or(ident));
        } else {
            let ch = expr[i..].chars().next().unwrap_or(' ');
            out.push(ch);
            i += ch.len_utf8();
        }
    }
    out
}

/// The parts of an expression, `;` apart.
fn expression_parts(expr: &str) -> impl Iterator<Item = &str> {
    expr.split(';').map(str::trim).filter(|p| !p.is_empty())
}

/// A request expression over the results of the assets' own band math: each `bK` of it is
/// the Kth result, worked out in brackets.
fn compose_over_results(request: &str, results: &[String]) -> Result<String, EngineError> {
    let mut missing = None;
    let composed = substitute_band_tokens(request, |k| match results.get(k.wrapping_sub(1)) {
        Some(part) => Some(format!("({part})")),
        None => {
            missing.get_or_insert(k);
            None
        }
    });
    match missing {
        Some(k) => Err(EngineError::BadRequest(format!("band {k} out of range ({} bands)", results.len()))),
        None => Ok(composed),
    }
}

/// Whether `expr` names any of `names` (as `{asset}` or `{asset}_b{k}`), even where the name
/// is spelled as its band would be (an asset named `b2`).
fn names_in_expression(expr: &str, names: &[String]) -> bool {
    let found = std::cell::Cell::new(false);
    let _ = crate::stac::rewrite_stac_expr_to(expr, names, |asset, _, _| {
        found.set(true);
        Ok(asset)
    });
    found.get()
}

/// Each of `names`' own band math and own bands: by entry when the query's entries are
/// `names` (an asset asked for twice keeps what each time asked), else by name. An entry's
/// own `|bidx=` wins over what `asset_bidx` gives its asset.
fn stac_entry_options(pq: &ParsedQuery, names: &[String]) -> Vec<(Option<String>, Option<Vec<u16>>)> {
    let by_entry = pq.asset_entries.len() == names.len() && pq.asset_entries.iter().zip(names).all(|(e, n)| &e.name == n);
    if by_entry {
        return pq
            .asset_entries
            .iter()
            .map(|e| {
                let named = || pq.asset_bidx_by_name.iter().find(|(n, _)| n == &e.name).map(|(_, b)| b.clone());
                (e.expression.clone(), e.bands.clone().or_else(named))
            })
            .collect();
    }
    let nth = occurrences(names);
    names
        .iter()
        .zip(nth)
        .map(|(name, nth)| {
            (stac_asset_expr_nth(pq, name, nth).map(str::to_string), stac_asset_selected_bands(pq, name).map(<[u16]>::to_vec))
        })
        .collect()
}

/// The bands of a `/stac` request's assets as the renderer reads them, and its expression over
/// them, as rio-tiler stacks a STAC item: of several assets, every band of each (or those its
/// `|bidx=` picks), one after another, so that `bN` is the Nth band of the stack; an asset's
/// own band math (`|expression=`) is over all of its own bands, at its place in the stack, and
/// a request `expression` is then over the results. Asset names in the request expression
/// stand for the asset's first band in the stack (`{asset}_b{k}` for its band `k`), and `bN`
/// for the Nth band of the stack. Under `asset_as_band` each asset must give one band.
///
/// Returns the rasters to read, one per band read when that is not one band per asset,
/// with the band each is read for.
fn expand_stac_asset_bidx(
    pq: &mut ParsedQuery,
    names: &[String],
    cogs: Vec<Arc<cogzilla_io::OpenedCog>>,
    rewritten: Option<String>,
    read_names: bool,
) -> Result<(Vec<Arc<cogzilla_io::OpenedCog>>, Option<Vec<u16>>), EngineError> {
    let original = std::mem::replace(&mut pq.expr_raw, rewritten);
    let named = read_names && original.as_deref().is_some_and(|expr| names_in_expression(expr, names));
    let options = stac_entry_options(pq, names);
    let mut picked: Vec<Vec<u16>> = Vec::with_capacity(names.len());
    for ((name, cog), (own, sel)) in names.iter().zip(&cogs).zip(&options) {
        let nb = data_band_count(cog.index.full_res());
        // An asset's band math is over all of its bands, and wins over its `bidx`.
        let bands: Vec<u16> = match (own, sel) {
            (Some(_), _) | (None, None) => (1..=nb).collect(),
            (None, Some(sel)) => {
                if let Some(&b) = sel.iter().find(|&&b| b == 0 || b > nb) {
                    return Err(EngineError::BadRequest(format!(
                        "band {b} out of range for asset '{name}' (dataset has {nb} bands)"
                    )));
                }
                sel.clone()
            }
        };
        let gives = own.as_deref().map_or(bands.len(), |own| expression_parts(own).count());
        if pq.asset_as_band && gives != 1 {
            return Err(EngineError::BadRequest(format!(
                "Can't use `asset_as_band` for multibands asset ('{name}' gives {gives} bands)"
            )));
        }
        picked.push(bands);
    }
    // The type of the folded band math's results, as numpy types them stage by stage: each
    // asset's own over its raster, stacked, then the request's (what an algorithm is given).
    if options.iter().any(|(own, _)| own.is_some()) {
        let reads: Vec<cogzilla_raster::SampleType> = cogs.iter().map(|c| c.index.full_res().sample_type).collect();
        let own: Vec<Option<String>> = options.iter().map(|(own, _)| own.clone()).collect();
        let request = pq.expr_raw.clone().unwrap_or_default();
        pq.band_math_type = array_type::ArrayRule::Math { unscale: pq.unscale }.staged_kind(&reads, &own, &request);
    }
    if names.len() == 1 {
        match &options[0] {
            (Some(own), _) => {
                let results: Vec<String> = expression_parts(own).map(str::to_string).collect();
                pq.expr_raw = Some(match pq.expr_raw.take() {
                    None => own.clone(),
                    Some(request) => compose_over_results(&request, &results)?,
                });
                pq.asset_exprs.clear();
                pq.asset_bidx.clear();
                pq.bidx.clear();
            }
            // Band math over the bands chosen reads them in their order: `bN` is the Nth
            // chosen, whatever top-level `bidx` came with it. On `/stac` the asset's name, read
            // first and whole, is the first band chosen and `{asset}_b{k}` its band `k`.
            (None, Some(sel)) => match original.as_deref() {
                Some(expr) if pq.expr_raw.is_some() => {
                    let read = if read_names { names } else { &[] };
                    let remapped = crate::stac::rewrite_stac_expr_with(
                        expr,
                        read,
                        |_, name, band| match band {
                            Some(k) => {
                                if sel.iter().any(|&b| usize::from(b) == k) {
                                    Ok(k)
                                } else {
                                    Err(EngineError::BadRequest(format!(
                                        "band {k} of asset '{name}' is not read"
                                    )))
                                }
                            }
                            None => Ok(sel.first().map_or(1, |&b| usize::from(b))),
                        },
                        |k| match sel.get(k.wrapping_sub(1)) {
                            Some(&b) => Ok(Some(usize::from(b))),
                            None => Err(EngineError::BadRequest(format!(
                                "band {k} out of range for asset '{}' ({} bands chosen)",
                                names[0],
                                sel.len()
                            ))),
                        },
                    )?;
                    pq.expr_raw = Some(remapped);
                }
                _ if pq.bidx.is_empty() => pq.bidx = sel.clone(),
                _ => {}
            },
            _ => {}
        }
        return Ok((cogs, None));
    }
    let offsets: Vec<usize> = picked
        .iter()
        .scan(0usize, |at, bands| {
            let first = *at;
            *at += bands.len();
            Some(first)
        })
        .collect();
    if options.iter().any(|(own, _)| own.is_some()) {
        if named {
            return Err(EngineError::BadRequest(
                "an expression over the results of the assets' own band math takes b1..bn, not asset names".into(),
            ));
        }
        let mut results: Vec<String> = Vec::new();
        for (i, name) in names.iter().enumerate() {
            let count = picked[i].len();
            match &options[i].0 {
                Some(own) => {
                    for part in expression_parts(own) {
                        let mut beyond = None;
                        let shifted = substitute_band_tokens(part, |j| {
                            if (1..=count).contains(&j) {
                                Some(format!("b{}", offsets[i] + j))
                            } else {
                                beyond.get_or_insert(j);
                                None
                            }
                        });
                        if let Some(j) = beyond {
                            return Err(EngineError::BadRequest(format!(
                                "band {j} out of range for asset '{name}' (dataset has {count} bands)"
                            )));
                        }
                        results.push(shifted);
                    }
                }
                None => results.extend((1..=count).map(|k| format!("b{}", offsets[i] + k))),
            }
        }
        pq.expr_raw = Some(match pq.expr_raw.take() {
            None => results.join(";"),
            Some(request) => compose_over_results(&request, &results)?,
        });
        pq.asset_exprs.clear();
        pq.bidx.clear();
    } else if named && let Some(expr) = original {
        // An asset's name is its first band in the stack, `{asset}_b{k}` its band `k`; `bN`
        // stays the Nth band of the stack.
        pq.expr_raw = Some(crate::stac::rewrite_stac_expr_to(&expr, names, |asset, name, band| {
            let at = asset - 1;
            let place = match band {
                None => Some(0),
                Some(k) => picked[at].iter().position(|&b| usize::from(b) == k),
            };
            place.map(|p| offsets[at] + p + 1).ok_or_else(|| {
                EngineError::BadRequest(format!("band {} of asset '{name}' is not read", band.unwrap_or(0)))
            })
        })?);
    }
    if picked.iter().all(|bands| bands.as_slice() == [1]) {
        return Ok((cogs, None));
    }
    let slots: usize = picked.iter().map(Vec::len).sum();
    if slots > 16 {
        return Err(EngineError::BadRequest(format!(
            "{slots} bands over the assets are more than the 16 read at once; choose bands with `|bidx=`"
        )));
    }
    // Without band math every band read is an output, and an answer holds at most 15.
    if pq.expr_raw.is_none() && pq.bidx.is_empty() && slots > 15 {
        return Err(EngineError::BadRequest(format!(
            "{slots} bands over the assets are more than the 15 an answer holds; choose bands with `|bidx=` or `expression`"
        )));
    }
    let mut expanded_cogs = Vec::with_capacity(slots);
    let mut slot_bands = Vec::with_capacity(slots);
    for (cog, bands) in cogs.into_iter().zip(picked) {
        for b in bands {
            expanded_cogs.push(cog.clone());
            slot_bands.push(b);
        }
    }
    Ok((expanded_cogs, Some(slot_bands)))
}

/// A STAC item's band math for its statistics, read as rio-tiler stacks the assets: every band
/// of each (or those its `|bidx=` picks), an asset's own band math in its place, and the
/// request's expression over the stack ([`expand_stac_asset_bidx`]). Gives the slots to read
/// (the band of each, when not one band of each asset in order) and each result compiled and
/// described as TiTiler and titiler-pgstac describe it, in the names of the stacked bands;
/// no band math when the request and its assets have none. `on_stac`: the `/stac` routes,
/// where assets are named in band math.
#[allow(clippy::type_complexity)]
fn stac_stats_band_math(
    pq: &ParsedQuery,
    names: &[String],
    cogs: Vec<Arc<cogzilla_io::OpenedCog>>,
    rewritten_expr: Option<String>,
    on_stac: bool,
) -> Result<
    (
        Vec<Arc<cogzilla_io::OpenedCog>>,
        Option<Vec<u16>>,
        Option<(Vec<(String, ExprProgram)>, Option<crate::stats::histogram::HistDtype>)>,
    ),
    EngineError,
> {
    let options = stac_entry_options(pq, names);
    if rewritten_expr.is_none() && options.iter().all(|(own, _)| own.is_none()) {
        return Ok((cogs, None, None));
    }
    let counts: Vec<u16> = cogs.iter().map(|c| data_band_count(c.index.full_res())).collect();
    let stack_names = pgstac::item_stack_names(names, &counts, &options, pq.asset_as_band);
    // One band of each asset, as read: the stack is the assets in order, and the rewritten
    // expression reads it as it is.
    let one_band_each = counts.iter().all(|&c| c == 1) && options.iter().all(|(own, sel)| own.is_none() && sel.is_none());
    let mut slot_bands = None;
    let mut expanded = None;
    let (cogs, expr) = if one_band_each {
        (cogs, rewritten_expr.unwrap_or_default())
    } else {
        let mut stacked = pq.clone();
        let (cogs, slots) = expand_stac_asset_bidx(&mut stacked, names, cogs, rewritten_expr, on_stac)?;
        slot_bands = slots;
        let expr = stacked.expr_raw.clone().unwrap_or_default();
        expanded = Some(stacked);
        (cogs, expr)
    };
    let stacked = expanded.as_ref().unwrap_or(pq);
    let parts: Vec<&str> = expression_parts(&expr).collect();
    if parts.len() > 16 {
        return Err(EngineError::BadRequest("expression exceeds max 16 semicolon-separated outputs".into()));
    }
    let descriptions: Vec<String> = match pq.expr_raw.as_deref() {
        Some(request) => {
            let mut sorted_names: Vec<&str> = names.iter().map(String::as_str).collect();
            sorted_names.sort_by_key(|b| std::cmp::Reverse(b.len()));
            expression_parts(request)
                .map(|part| {
                    if on_stac && !sorted_names.is_empty() {
                        // Preserve verbatim asset names (including uppercase `B02` and
                        // hyphenated `SR-B1`) before substituting positional `bN`.
                        let bytes = part.as_bytes();
                        let is_ident = |b: u8| b.is_ascii_alphanumeric() || b == b'_' || b == b'-';
                        let mut out = String::with_capacity(part.len());
                        let mut seg_start = 0usize;
                        let mut i = 0usize;
                        while i < bytes.len() {
                            let at_start = i == 0 || !is_ident(bytes[i - 1]);
                            let matched = if at_start {
                                sorted_names.iter().copied().find(|name| {
                                    part[i..].starts_with(name)
                                        && !bytes
                                            .get(i + name.len())
                                            .is_some_and(|&c| is_ident(c))
                                })
                            } else {
                                None
                            };
                            if let Some(name) = matched {
                                out.push_str(&substitute_band_tokens(&part[seg_start..i], |k| {
                                    stack_names.get(k.wrapping_sub(1)).cloned()
                                }));
                                out.push_str(name);
                                i += name.len();
                                seg_start = i;
                            } else {
                                i += part[i..].chars().next().map_or(1, char::len_utf8);
                            }
                        }
                        out.push_str(&substitute_band_tokens(&part[seg_start..], |k| {
                            stack_names.get(k.wrapping_sub(1)).cloned()
                        }));
                        out
                    } else {
                        substitute_band_tokens(part, |k| {
                            stack_names.get(k.wrapping_sub(1)).cloned()
                        })
                    }
                })
                .collect()
        }
        None => stack_names,
    };
    let described = parts
        .iter()
        .enumerate()
        .map(|(idx, part)| (descriptions.get(idx).cloned().unwrap_or_else(|| part.to_string()), part.to_string()))
        .collect();
    let compiled = compile_stats_parts(stacked, described, Some(reads_type(&cogs)))?;
    Ok((cogs, slot_bands, Some(compiled)))
}

#[allow(clippy::too_many_arguments)]
async fn render_stac_bbox_opt(
    engine: &TileEngine,
    cogs: &[Arc<cogzilla_io::OpenedCog>],
    slot_bands: Option<&[u16]>,
    bbox: [f64; 4],
    dst_crs: &cogzilla_proj::CrsModel,
    native_preview: bool,
    out_w: u32,
    out_h: u32,
    poly_mask: Option<&PolygonMask>,
    bands: &BandRequest,
    resample: ResampleMode,
    fmt: OutputFormat,
    nodata: Option<f32>,
    custom_lut: Option<&CustomColormap>,
    cf: Option<&ColorFormula>,
    return_mask: bool,
    unscale: bool,
) -> Result<Bytes, EngineError> {
    if let Some(ab) = slot_bands {
        render_multi_asset_bbox(
            engine,
            cogs,
            ab,
            bbox,
            dst_crs,
            native_preview,
            out_w,
            out_h,
            poly_mask,
            bands,
            resample,
            fmt,
            nodata,
            custom_lut,
            cf,
            return_mask,
            unscale,
        )
        .await
    } else if native_preview {
        render_native_preview(
            engine,
            cogs,
            bbox,
            out_w,
            out_h,
            bands,
            resample,
            fmt,
            nodata,
            custom_lut,
            cf,
            return_mask,
            unscale,
        )
        .await
    } else {
        render_crs_bbox(
            engine,
            cogs,
            bbox,
            dst_crs,
            out_w,
            out_h,
            poly_mask,
            bands,
            resample,
            fmt,
            nodata,
            custom_lut,
            cf,
            return_mask,
            unscale,
        )
        .await
    }
}

/// Resolve the effective output `(CrsModel, epsg)` for `/{cog,stac,mosaicjson}/bbox` routes:
/// explicit `?dst_crs=` wins, otherwise default to `?coord_crs=` (matching TiTiler 2.4 / `rio-tiler`),
/// falling back to `EPSG:4326`.
fn resolve_bbox_dst_crs(pq: &ParsedQuery) -> Option<(cogzilla_proj::CrsModel, u16)> {
    pq.dst_crs
        .clone()
        .or_else(|| pq.coord_crs.clone().zip(pq.coord_crs_epsg))
}

fn stac_asset_selected_bands<'a>(pq: &'a ParsedQuery, asset_name: &str) -> Option<&'a [u16]> {
    pq.asset_bidx
        .iter()
        .find(|(n, _)| n == asset_name)
        .map(|(_, bands)| bands.as_slice())
}

/// Composite up to 16 intersecting MosaicJSON scenes for `/mosaicjson/bbox` and `/mosaicjson/feature`
/// using first-non-transparent alpha-over (or first-non-NaN for f32/npy) blending in target CRS.
#[allow(clippy::too_many_arguments)]
async fn render_mosaic_bbox_composite(
    engine: &TileEngine,
    urls: &[String],
    bbox: [f64; 4],
    dst_crs: &cogzilla_proj::CrsModel,
    out_w: u32,
    out_h: u32,
    poly_mask: Option<&PolygonMask>,
    pq: &ParsedQuery,
    fmt: OutputFormat,
    ct: &'static str,
    epsg: u16,
) -> Result<(Bytes, Vec<String>), EngineError> {
    if urls.is_empty() {
        return Err(EngineError::NotFound("no mosaic scenes intersect requested bounds".into()));
    }
    let out_w = out_w.clamp(1, 4096);
    let out_h = out_h.clamp(1, 4096);
    let urls = &urls[..urls.len().min(64)];
    let method = pq.pixel_selection;
    let total_px = (out_w as usize).saturating_mul(out_h as usize);
    let mut eff_fmt: Option<OutputFormat> = None;
    let mut first_bands: Option<BandRequest> = None;
    let mut first_lut: Option<cogzilla_kernel::CustomColormap> = None;
    let mut acc_rgba: Option<Vec<u8>> = None;
    let mut fold: Option<Fold> = None;
    let mut taken = 0usize;
    let mut opaque_count = 0usize;
    let mut any_ok = false;
    let mut last_err: Option<EngineError> = None;
    let mut used_assets: Vec<String> = Vec::new();

    for (scene_i, u) in urls.iter().enumerate() {
        if scene_i > 0 {
            tokio::task::consume_budget().await;
        }
        let cog = match engine.store.open_uri(u).await.map(|c| engine.store.with_nodata(&c, pq.nodata)) {
            Ok(c) => c,
            Err(e) if e.is_fatal() => {
                return Err(EngineError::Io(e));
            }
            Err(e) => {
                last_err = Some(crate::engine::asset_fetch_error(u, e));
                continue;
            }
        };
        let cogs = [cog];
        let full = cogs[0].index.full_res();
        let data_bands = data_band_count(full);
        let mut bands =
            build_band_request_for_output(pq, data_bands, full.sample_type, fmt, ct)?;
        let scene_fmt = *eff_fmt.get_or_insert_with(|| {
            geotiff_render_format(fmt, ct, pq, Some(&cogs[0]), &bands, full.sample_type, false)
        });
        let is_f32_out = matches!(scene_fmt, OutputFormat::RawF32 | OutputFormat::Npy);
        if !is_f32_out {
            match bands {
                BandRequest::MultiBand { ref bands_1based }
                    if !(method.by_last_band() && matches!(bands_1based.len(), 2 | 4)) =>
                {
                    return Err(EngineError::BadRequest(
                        "multi-band bidx (2 or 4..=15 bands) is only supported for .tif, .npy, or .f32 output"
                            .into(),
                    ));
                }
                BandRequest::CustomExprMulti { ref progs }
                    if !(method.by_last_band() && matches!(progs.len(), 2 | 4)) =>
                {
                    return Err(EngineError::BadRequest(format!(
                        "unsupported expression count {} for visual image output (expected 1 or 3; use .tif, .npy, or .f32)",
                        progs.len()
                    )));
                }
                _ => {}
            }
        }
        if first_bands.is_none() {
            first_bands = Some(bands.clone());
        }
        if method.by_last_band()
            && pq.bidx.is_empty()
            && pq.expr_raw.is_none()
            && pq.algorithm.is_none()
            && data_bands >= 2
            && matches!(bands, BandRequest::Single { .. } | BandRequest::Rgb { .. })
        {
            bands = BandRequest::MultiBand {
                bands_1based: (1..=data_bands).collect(),
            };
        }
        if pq.unscale {
            apply_unscale_cogs(&mut bands, &cogs, None);
        }
        let custom_lut = if matches!(method, Selection::Count | Selection::Stdev)
            && pq.custom_colormap_json.is_none()
        {
            None
        } else {
            resolve_custom_lut(pq, &cogs[0], &bands)?
        };
        if first_lut.is_none() && custom_lut.is_some() {
            first_lut = custom_lut.clone();
        }
        let cf = pq.parse_color_formula()?;
        let resample = resolve_resample(Some(engine), pq, Some(&cogs[0]), epsg);
        if urls.len() == 1 && method == Selection::First {
            let d = render_crs_bbox(
                engine,
                &cogs,
                bbox,
                dst_crs,
                out_w,
                out_h,
                poly_mask,
                &bands,
                resample,
                scene_fmt,
                pq.nodata,
                custom_lut.as_ref(),
                cf.as_ref(),
                pq.return_mask,
                pq.unscale,
            )
            .await?;
            return Ok((
                wrap_geotiff_fmt(
                    d,
                    ct,
                    scene_fmt,
                    out_w,
                    out_h,
                    bbox,
                    epsg,
                    pq.return_mask,
                ),
                vec![u.clone()],
            ));
        }
        if is_f32_out || method != Selection::First {
            let raw = match render_crs_bbox(
                engine,
                &cogs,
                bbox,
                dst_crs,
                out_w,
                out_h,
                poly_mask,
                &bands,
                resample,
                OutputFormat::RawF32,
                pq.nodata,
                None,
                None,
                pq.return_mask,
                pq.unscale,
            )
            .await
            {
                Ok(b) => b,
                Err(EngineError::NotFound(_)) => continue,
                Err(e) => return Err(e),
            };
            any_ok = true;
            if total_px == 0 || raw.len() % (total_px * 4) != 0 {
                continue;
            }
            let nb = raw.len() / (total_px * 4);
            if nb == 0 {
                continue;
            }
            let fold = match &mut fold {
                Some(f) => f,
                None => {
                    if method.by_last_band() && nb < 2 {
                        return Err(EngineError::BadRequest(
                            "pixel_selection by the last band needs at least two bands".into(),
                        ));
                    }
                    if Fold::<f32>::working_bytes(method, nb, total_px) > MAX_MEDIAN_BUFFER_BYTES {
                        return Err(EngineError::BadRequest(format!(
                            "pixel_selection working buffer exceeds memory budget ({} MiB limit); reduce width, height, or band count",
                            MAX_MEDIAN_BUFFER_BYTES / (1024 * 1024)
                        )));
                    }
                    let integer = reads_whole_numbers(pq.unscale, &bands, full.sample_type);
                    let absent = pq.nodata.or(cogs[0].index.nodata).unwrap_or(0.0);
                    fold.insert(Fold::new(method, nb, total_px, integer).absent_as(absent))
                }
            };
            if fold.bands() != nb {
                continue;
            }
            taken += 1;
            if Fold::<f32>::held_bytes(method, taken, nb, total_px) > MAX_MEDIAN_BUFFER_BYTES {
                return Err(EngineError::BadRequest(format!(
                    "pixel_selection=median exceeds memory budget ({} MiB limit); reduce width, height, or band count",
                    MAX_MEDIAN_BUFFER_BYTES / (1024 * 1024)
                )));
            }
            fold.add_le_bytes(&raw);
            used_assets.push(u.clone());
            if fold.is_done() {
                break;
            }
            continue;
        }
        let raw = match render_crs_bbox(
            engine,
            &cogs,
            bbox,
            dst_crs,
            out_w,
            out_h,
            poly_mask,
            &bands,
            resample,
            OutputFormat::RawRgba,
            pq.nodata,
            custom_lut.as_ref(),
            cf.as_ref(),
            true,
            pq.unscale,
        )
        .await
        {
            Ok(b) => b,
            Err(EngineError::NotFound(_)) => continue,
            Err(e) => return Err(e),
        };
        any_ok = true;
        used_assets.push(u.clone());
        if raw.len() != total_px * 4 {
            continue;
        }
        let acc = acc_rgba.get_or_insert_with(|| vec![0u8; total_px * 4]);
        for i in 0..total_px {
            let off = i * 4;
            if acc[off + 3] == 0 && raw[off + 3] > 0 {
                acc[off..off + 4].copy_from_slice(&raw[off..off + 4]);
                if raw[off + 3] == 255 {
                    opaque_count += 1;
                }
            }
        }
        if opaque_count >= total_px {
            break;
        }
    }
    if !any_ok {
        return Err(last_err.unwrap_or_else(|| {
            EngineError::NotFound("no readable mosaic scenes intersect requested bounds".into())
        }));
    }
    let final_fmt = eff_fmt.unwrap_or(fmt);
    if matches!(final_fmt, OutputFormat::RawF32 | OutputFormat::Npy) || method != Selection::First {
        let (out_nb, mut combined): (usize, Vec<f32>) = match fold.and_then(Fold::finish) {
            Some(Selected { bands, values }) => (bands, values),
            None => {
                let nb = first_bands
                    .as_ref()
                    .map(|b| default_raw_band_count(b, method))
                    .unwrap_or(1);
                (nb, vec![f32::NAN; nb * total_px])
            }
        };
        if method == Selection::Count
            && let Some(mask) = poly_mask
        {
            let kept = mask.rasterize_all_touched(out_w as usize, out_h as usize, bbox, dst_crs);
            for (v, &inside) in combined.iter_mut().zip(&kept) {
                if !inside {
                    *v = f32::NAN;
                }
            }
        }
        if matches!(final_fmt, OutputFormat::RawF32 | OutputFormat::Npy) {
            let mut raw_buf = bytes::BytesMut::with_capacity(combined.len() * 4);
            for v in combined {
                raw_buf.extend_from_slice(&v.to_le_bytes());
            }
            let raw = raw_buf.freeze();
            if matches!(final_fmt, OutputFormat::Npy) {
                return Ok((
                    cogzilla_kernel::wrap_raw_f32_as_npy_with_mask(
                        raw,
                        out_nb,
                        out_h,
                        out_w,
                        pq.return_mask,
                    ),
                    used_assets,
                ));
            }
            return Ok((
                wrap_geotiff_fmt(
                    raw,
                    ct,
                    final_fmt,
                    out_w,
                    out_h,
                    bbox,
                    epsg,
                    pq.return_mask,
                ),
                used_assets,
            ));
        }
        let fallback_bands = BandRequest::Single {
            band_1based: 1,
            rescale: (0.0, 255.0),
            colormap: pq.colormap,
        };
        let paint_bands = first_bands.as_ref().unwrap_or(&fallback_bands);
        let rgba = paint_selected(
            Selected {
                bands: out_nb,
                values: combined,
            },
            total_px,
            paint_bands,
            pq,
            first_lut.as_ref(),
        )?;
        let data = encode_rgba_output(out_w as usize, out_h as usize, &rgba, final_fmt, pq.return_mask);
        return Ok((
            wrap_geotiff_fmt(
                data,
                ct,
                final_fmt,
                out_w,
                out_h,
                bbox,
                epsg,
                pq.return_mask,
            ),
            used_assets,
        ));
    }
    let acc = acc_rgba.unwrap_or_else(|| vec![0u8; total_px * 4]);
    let data = if let Some(gpu) = engine.accelerated() {
        gpu.encode_rgba_output(out_w as usize, out_h as usize, &acc, final_fmt, pq.return_mask)
    } else {
        encode_rgba_output(out_w as usize, out_h as usize, &acc, final_fmt, pq.return_mask)
    };
    Ok((
        wrap_geotiff_fmt(
            data,
            ct,
            final_fmt,
            out_w,
            out_h,
            bbox,
            epsg,
            pq.return_mask,
        ),
        used_assets,
    ))
}

/// Compute statistics over a mosaicked set of assets folded through `pixel_selection::Fold`
/// (honouring all 9 `pixel_selection` methods, `bidx`, `unscale`, `nodata`, `width`/`height`,
/// `max_size`, and optional `poly_mask`), returning `(stats_json, used_assets)`.
async fn compute_mosaic_statistics_with_fold(
    engine: &TileEngine,
    urls: &[String],
    wgs_bbox: [f64; 4],
    poly_mask: Option<&PolygonMask>,
    pq: &ParsedQuery,
    stats_opts: &crate::stats::StatsOptions,
) -> Result<(serde_json::Value, Vec<String>), EngineError> {
    if urls.is_empty() {
        return Err(EngineError::NotFound(
            "no mosaic scenes intersect requested geometry".into(),
        ));
    }
    let urls = &urls[..urls.len().min(64)];
    let method = pq.pixel_selection;
    let dst_crs = cogzilla_proj::CrsModel::Geographic;
    let resample = pq.resample;

    // Exact analytical native-grid path for POST /mosaicjson/statistics when no explicit
    // output width/height resampling was requested and all contributing assets share one grid
    // (always true for a single-asset mosaic, guaranteeing exact parity with /cog/statistics).
    if let Some(mask) = poly_mask
        && pq.width.is_none()
        && pq.height.is_none()
    {
        let mut opened: Vec<(String, Arc<cogzilla_io::OpenedCog>)> =
            Vec::with_capacity(urls.len());
        let mut open_err: Option<EngineError> = None;
        for u in urls {
            match engine.store.open_uri(u).await.map(|c| engine.store.with_nodata(&c, pq.nodata)) {
                Ok(c) => opened.push((u.clone(), c)),
                Err(e) if e.is_fatal() => {
                    return Err(EngineError::Io(e));
                }
                Err(e) => open_err = Some(crate::engine::asset_fetch_error(u, e)),
            }
        }
        if opened.is_empty() {
            return Err(open_err.unwrap_or_else(|| {
                EngineError::NotFound("no readable mosaic scenes intersect requested bounds".into())
            }));
        }
        let mut exact_opts = stats_opts.clone();
        if exact_opts.hist_dtype.is_none() {
            let band_dtype = crate::stats::histogram::HistDtype::for_sample_type(
                opened[0].1.index.full_res().sample_type,
                pq.unscale,
            );
            exact_opts.hist_dtype = Some(if method == Selection::Count {
                crate::stats::histogram::HistDtype::Float64
            } else if let Some(expr) = pq.expr_raw.as_deref() {
                crate::stats::histogram::HistDtype::for_expression(expr, band_dtype)
            } else {
                band_dtype
            });
        }
        if let Some(exact) = crate::zonal::compute_native_exact_mosaic_fold_stats(
            engine,
            &opened,
            mask,
            pq.max_size,
            pq.unscale,
            pq.nodata,
            method,
            MAX_MEDIAN_BUFFER_BYTES,
            &exact_opts,
        )
        .await?
        {
            return Ok(exact);
        }
    }

    let mut eff_bbox = wgs_bbox;
    let mut out_dims: Option<(u32, u32)> = None;
    let mut first_sample_type: Option<cogzilla_raster::SampleType> = None;
    let mut selected_bidx: Vec<u16> = Vec::new();
    let mut band_descs: Vec<String> = Vec::new();
    let mut fold: Option<Fold> = None;
    let mut taken = 0usize;
    let mut any_ok = false;
    let mut last_err: Option<EngineError> = None;
    let mut used_assets: Vec<String> = Vec::new();

    for (scene_i, u) in urls.iter().enumerate() {
        if scene_i > 0 {
            tokio::task::consume_budget().await;
        }
        let cog = match engine.store.open_uri(u).await.map(|c| engine.store.with_nodata(&c, pq.nodata)) {
            Ok(c) => c,
            Err(e) if e.is_fatal() => {
                return Err(EngineError::Io(e));
            }
            Err(e) => {
                last_err = Some(crate::engine::asset_fetch_error(u, e));
                continue;
            }
        };
        let cogs = [cog];
        let full = cogs[0].index.full_res();
        first_sample_type.get_or_insert(full.sample_type);
        let (out_w, out_h) = match out_dims {
            Some(d) => d,
            None => {
                // Expand degenerate Point / axis-aligned LineString bboxes by one native
                // pixel along the zero-extent axis so render_crs_bbox can sample them.
                if eff_bbox[2] <= eff_bbox[0] || eff_bbox[3] <= eff_bbox[1] {
                    let (px_dx, px_dy) = if matches!(
                        engine.get_or_build_crs(&cogs[0]),
                        Ok(cogzilla_proj::CrsModel::Geographic)
                    ) && full.pixel_to_crs.a.abs() > 1e-12
                        && full.pixel_to_crs.e.abs() > 1e-12
                    {
                        (full.pixel_to_crs.a.abs(), full.pixel_to_crs.e.abs())
                    } else if let Ok(crs) = engine.get_or_build_crs(&cogs[0]) {
                        let b = crs.wgs84_bounds(&full.pixel_to_crs, full.gcp_poly.as_deref(), full.width, full.height);
                        (
                            ((b[2] - b[0]).abs() / full.width.max(1) as f64).max(1e-6),
                            ((b[3] - b[1]).abs() / full.height.max(1) as f64).max(1e-6),
                        )
                    } else {
                        (1e-5, 1e-5)
                    };
                    if eff_bbox[2] <= eff_bbox[0] {
                        eff_bbox[0] -= 0.5 * px_dx;
                        eff_bbox[2] += 0.5 * px_dx;
                    }
                    if eff_bbox[3] <= eff_bbox[1] {
                        eff_bbox[1] -= 0.5 * px_dy;
                        eff_bbox[3] += 0.5 * px_dy;
                    }
                }
                let cap = if pq.max_size == 0 {
                    4096
                } else {
                    pq.max_size.clamp(1, 4096)
                };
                let (w, h) = if matches!(
                    engine.get_or_build_crs(&cogs[0]),
                    Ok(cogzilla_proj::CrsModel::Geographic)
                ) && full.pixel_to_crs.b.abs() < 1e-12
                    && full.pixel_to_crs.d.abs() < 1e-12
                    && full.pixel_to_crs.a.abs() > 1e-12
                    && full.pixel_to_crs.e.abs() > 1e-12
                {
                    let nat_w_f = (eff_bbox[2] - eff_bbox[0]).abs() / full.pixel_to_crs.a.abs();
                    let nat_h_f = (eff_bbox[3] - eff_bbox[1]).abs() / full.pixel_to_crs.e.abs();
                    if pq.width.is_some() || pq.height.is_some() {
                        resolve_query_dims_f64(nat_w_f, nat_h_f, pq.width, pq.height)
                    } else {
                        let nat_w = nat_w_f.round().max(1.0) as u32;
                        let nat_h = nat_h_f.round().max(1.0) as u32;
                        if nat_w.max(nat_h) <= cap {
                            (nat_w, nat_h)
                        } else if nat_w >= nat_h {
                            let h = ((nat_h as f64 / nat_w as f64) * cap as f64)
                                .ceil()
                                .clamp(1.0, cap as f64) as u32;
                            (cap, h)
                        } else {
                            let w = ((nat_w as f64 / nat_h as f64) * cap as f64)
                                .ceil()
                                .clamp(1.0, cap as f64) as u32;
                            (w, cap)
                        }
                    }
                } else {
                    resolve_bbox_query_dims(engine, &cogs[0], &cogzilla_proj::CrsModel::Geographic, eff_bbox, eff_bbox, cap, pq.width, pq.height)
                };
                let dims = (w.clamp(1, 4096), h.clamp(1, 4096));
                out_dims = Some(dims);
                selected_bidx = if pq.bidx.is_empty() {
                    (1..=data_band_count(full).max(1)).collect()
                } else {
                    for &b in &pq.bidx {
                        if b == 0 || b > full.samples_per_pixel {
                            return Err(EngineError::BadRequest(format!(
                                "band index b{b} out of range (dataset has {} bands)",
                                full.samples_per_pixel
                            )));
                        }
                    }
                    pq.bidx.clone()
                };
                band_descs = selected_bidx
                    .iter()
                    .map(|&b| {
                        cogs[0]
                            .index
                            .band_descriptions
                            .get((b - 1) as usize)
                            .filter(|s| !s.is_empty())
                            .cloned()
                            .unwrap_or_else(|| format!("b{b}"))
                    })
                    .collect();
                dims
            }
        };
        let total_px = (out_w as usize).saturating_mul(out_h as usize);
        let bands = BandRequest::MultiBand {
            bands_1based: selected_bidx.clone(),
        };
        let raw = match render_crs_bbox(
            engine,
            &cogs,
            eff_bbox,
            &dst_crs,
            out_w,
            out_h,
            None,
            &bands,
            resample,
            OutputFormat::RawF32,
            pq.nodata,
            None,
            None,
            true,
            pq.unscale,
        )
        .await
        {
            Ok(b) => b,
            Err(EngineError::NotFound(_)) => continue,
            Err(e) => return Err(e),
        };
        any_ok = true;
        if total_px == 0 || raw.len() % (total_px * 4) != 0 {
            continue;
        }
        let nb = raw.len() / (total_px * 4);
        if nb == 0 {
            continue;
        }
        let fold = match &mut fold {
            Some(f) => f,
            None => {
                if method.by_last_band() && nb < 2 {
                    return Err(EngineError::BadRequest(
                        "pixel_selection by the last band needs at least two bands".into(),
                    ));
                }
                if Fold::<f32>::working_bytes(method, nb, total_px) > MAX_MEDIAN_BUFFER_BYTES {
                    return Err(EngineError::BadRequest(format!(
                        "pixel_selection working buffer exceeds memory budget ({} MiB limit); reduce max_size, width, height, or band count",
                        MAX_MEDIAN_BUFFER_BYTES / (1024 * 1024)
                    )));
                }
                let integer = reads_whole_numbers(pq.unscale, &bands, full.sample_type);
                let absent = pq.nodata.or(cogs[0].index.nodata).unwrap_or(0.0);
                fold.insert(Fold::new(method, nb, total_px, integer).absent_as(absent))
            }
        };
        if fold.bands() != nb {
            continue;
        }
        taken += 1;
        if Fold::<f32>::held_bytes(method, taken, nb, total_px) > MAX_MEDIAN_BUFFER_BYTES {
            return Err(EngineError::BadRequest(format!(
                "pixel_selection=median exceeds memory budget ({} MiB limit); reduce max_size, width, height, or band count",
                MAX_MEDIAN_BUFFER_BYTES / (1024 * 1024)
            )));
        }
        fold.add_le_bytes(&raw);
        used_assets.push(u.clone());
        if fold.is_done() {
            break;
        }
    }

    if !any_ok {
        return Err(last_err.unwrap_or_else(|| {
            EngineError::NotFound("no readable mosaic scenes intersect requested bounds".into())
        }));
    }

    let (out_w, out_h) = out_dims.unwrap_or((1, 1));
    let total_px = (out_w as usize).saturating_mul(out_h as usize);
    let (out_nb, combined): (usize, Vec<f32>) = match fold.and_then(Fold::finish) {
        Some(Selected { bands, values }) => (bands, values),
        None => {
            let nb = if method == Selection::Count {
                1
            } else if method.by_last_band() {
                selected_bidx.len().saturating_sub(1).max(1)
            } else {
                selected_bidx.len().max(1)
            };
            (nb, vec![f32::NAN; nb * total_px])
        }
    };
    let mut eff_opts = stats_opts.clone();
    if eff_opts.hist_dtype.is_none()
        && let Some(st) = first_sample_type
    {
        let band_dtype = crate::stats::histogram::HistDtype::for_sample_type(st, pq.unscale);
        eff_opts.hist_dtype = Some(if method == Selection::Count {
            crate::stats::histogram::HistDtype::Float64
        } else if let Some(expr) = pq.expr_raw.as_deref() {
            crate::stats::histogram::HistDtype::for_expression(expr, band_dtype)
        } else {
            band_dtype
        });
    }
    if let Some(mask) = poly_mask {
        let mut keys_descs: Vec<(String, String)> = Vec::with_capacity(out_nb);
        for bi in 0..out_nb {
            let key = if method == Selection::Count {
                "b1".to_string()
            } else {
                let b_num = selected_bidx.get(bi).copied().unwrap_or((bi + 1) as u16);
                format!("b{b_num}")
            };
            let desc = if method == Selection::Count {
                "count".to_string()
            } else {
                band_descs.get(bi).cloned().unwrap_or_else(|| key.clone())
            };
            keys_descs.push((key, desc));
        }
        let plane_refs: Vec<(&str, &str, &[f32])> = keys_descs
            .iter()
            .enumerate()
            .map(|(bi, (k, d))| {
                (
                    k.as_str(),
                    d.as_str(),
                    &combined[bi * total_px..(bi + 1) * total_px],
                )
            })
            .collect();
        let json = crate::zonal::compute_exact_grid_polygon_stats(
            &plane_refs,
            eff_bbox,
            out_w as usize,
            out_h as usize,
            mask,
            &eff_opts,
        );
        return Ok((json, used_assets));
    }

    let mut out = serde_json::Map::new();
    for bi in 0..out_nb {
        let plane = &combined[bi * total_px..(bi + 1) * total_px];
        let key = if method == Selection::Count {
            "b1".to_string()
        } else {
            let b_num = selected_bidx.get(bi).copied().unwrap_or((bi + 1) as u16);
            format!("b{b_num}")
        };
        let desc = if method == Selection::Count {
            "count".to_string()
        } else {
            band_descs.get(bi).cloned().unwrap_or_else(|| key.clone())
        };
        out.insert(key, compute_f32_band_stats(plane, &desc, &eff_opts));
    }
    Ok((serde_json::Value::Object(out), used_assets))
}

fn web_mercator_buffered_3857_bbox(z: u8, x: u32, y: u32, buffer: f32, tile_size: u32) -> [f64; 4] {
    let (min_x, min_y, max_x, max_y) = tile_bounds_3857(TileCoord { z, x, y });
    let pad_x = (max_x - min_x) * (buffer as f64 / tile_size as f64);
    let pad_y = (max_y - min_y) * (buffer as f64 / tile_size as f64);
    [min_x - pad_x, min_y - pad_y, max_x + pad_x, max_y + pad_y]
}

fn resolve_custom_lut(
    pq: &ParsedQuery,
    cog: &cogzilla_io::OpenedCog,
    bands: &BandRequest,
) -> Result<Option<CustomColormap>, EngineError> {
    if let Some(ref j) = pq.custom_colormap_json {
        return CustomColormap::from_json_str_with_type(
            j,
            !pq.rescale.is_empty(),
            pq.colormap_type_linear,
        )
        .map(Some)
        .map_err(EngineError::BadRequest);
    }
    if !pq.colormap_explicit
        && pq.rescale.is_empty()
        && matches!(bands, BandRequest::Single { .. })
        && let Some(ref pal) = cog.index.color_map
    {
        return Ok(Some(CustomColormap::from_rgba_palette(pal)));
    }
    Ok(None)
}

fn has_embedded_palette(
    pq: &ParsedQuery,
    cog: &cogzilla_io::OpenedCog,
    bands: &BandRequest,
) -> bool {
    pq.custom_colormap_json.is_none()
        && !pq.colormap_explicit
        && pq.rescale.is_empty()
        && matches!(bands, BandRequest::Single { .. })
        && cog.index.color_map.is_some()
}

fn plain_resp(
    status: StatusCode,
    body: impl Into<Bytes>,
    content_type: &'static str,
) -> Response<Full<Bytes>> {
    let mut resp = Response::new(Full::new(body.into()));
    *resp.status_mut() = status;
    resp.headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static(content_type));
    resp.headers_mut()
        .insert(ACCESS_CONTROL_ALLOW_ORIGIN, HeaderValue::from_static("*"));
    resp
}

async fn load_mosaicjson(
    engine: &TileEngine,
    uri: &str,
) -> Result<Arc<MosaicJsonCatalog>, EngineError> {
    cogzilla_io::check_uri_allowed(uri)?;
    if let Some(cached) = engine.mosaicjson_cache.get(uri) {
        return Ok(cached.clone());
    }
    let bytes = engine
        .store
        .fetch_uri_bytes(uri, MAX_POST_BODY_BYTES)
        .await?;
    let decoded_buf: Vec<u8>;
    let json_bytes: &[u8] = if bytes.len() >= 2 && bytes[0] == 0x1f && bytes[1] == 0x8b {
        decoded_buf = cogzilla_codec::decompress_gzip_limited(&bytes, MAX_POST_BODY_BYTES)
            .map_err(|e| EngineError::BadRequest(format!("invalid gzipped MosaicJSON: {e}")))?;
        &decoded_buf
    } else {
        &bytes
    };
    let text = String::from_utf8_lossy(json_bytes);
    let cat = Arc::new(MosaicJsonCatalog::from_json_str(&text).map_err(EngineError::BadRequest)?);
    if engine.mosaicjson_cache.len() >= 256 {
        engine.mosaicjson_cache.clear();
    }
    engine.mosaicjson_cache.insert(uri.to_string(), cat.clone());
    Ok(cat)
}

const MAX_POST_BODY_BYTES: usize = 16 * 1024 * 1024;

/// Maximum features accepted in a `POST /*/statistics` `FeatureCollection` (`0` = unlimited,
/// still bounded by `MAX_POST_BODY_BYTES` and per-feature window/block caps).
fn max_statistics_features() -> usize {
    crate::config::Config::global().max_statistics_features
}

/// Timeout in seconds for reading a POST request body (`0` disables the timeout).
fn body_timeout_secs() -> u64 {
    crate::config::Config::global().body_timeout_secs
}

fn check_feature_collection_len(len: usize) -> Result<(), EngineError> {
    let cap = max_statistics_features();
    if cap > 0 && len > cap {
        return Err(EngineError::BadRequest(format!(
            "FeatureCollection exceeds max {cap} features (got {len})"
        )));
    }
    Ok(())
}

fn is_null_geometry_feature(feat: &serde_json::Value) -> bool {
    feat.get("type").and_then(|v| v.as_str()) == Some("Feature")
        && matches!(feat.get("geometry"), Some(serde_json::Value::Null))
}

fn parse_feature_mask_or_null(
    feat: &serde_json::Value,
    coord_crs: Option<&cogzilla_proj::CrsModel>,
) -> Result<(PolygonMask, [f64; 4]), EngineError> {
    if is_null_geometry_feature(feat) {
        return Ok((PolygonMask::empty(), [0.0; 4]));
    }
    PolygonMask::from_geojson_in_crs(feat, coord_crs)
}

fn wrap_feature_error(idx: usize, feat: &serde_json::Value, err: EngineError) -> EngineError {
    let prefix = match feat.get("id") {
        Some(serde_json::Value::String(s)) => format!("features[{idx}] (id={s:?})"),
        Some(v) if !v.is_null() => format!("features[{idx}] (id={v})"),
        _ => format!("features[{idx}]"),
    };
    match err {
        EngineError::BadRequest(msg) => EngineError::BadRequest(format!("{prefix}: {msg}")),
        other => other,
    }
}

fn insert_feature_stats(feat: &mut serde_json::Value, stats: serde_json::Value) {
    if let Some(props) = feat.get_mut("properties").and_then(|v| v.as_object_mut()) {
        props.insert("statistics".into(), stats);
    } else if let Some(obj) = feat.as_object_mut() {
        obj.insert(
            "properties".into(),
            serde_json::json!({"statistics": stats}),
        );
    }
}

fn fc_worker_count(n: usize) -> usize {
    std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4)
        .clamp(2, 16)
        .min(n)
}

/// Shared worker pool for `POST /*/statistics` on GeoJSON `FeatureCollection` payloads.
/// Parses every feature's polygon mask against `fc_crs`, runs `compute` across bounded
/// tokio workers, inserts each feature's `"statistics"` property, and returns per-feature
/// sidecar outputs `T` (e.g. used asset lists for `X-Assets`).
async fn run_feature_collection_stats<T, F, Fut>(
    geojson: &mut serde_json::Value,
    fc_crs: Option<&cogzilla_proj::CrsModel>,
    compute: F,
) -> Result<Vec<T>, EngineError>
where
    T: Send + 'static,
    F: Fn(Arc<PolygonMask>, [f64; 4]) -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = Result<(serde_json::Value, T), EngineError>> + Send + 'static,
{
    let features = geojson
        .get_mut("features")
        .and_then(|v| v.as_array_mut())
        .ok_or_else(|| EngineError::BadRequest("FeatureCollection missing 'features' array".into()))?;
    check_feature_collection_len(features.len())?;
    let n_feats = features.len();
    let mut parsed = Vec::with_capacity(n_feats);
    for (idx, feat) in features.iter().enumerate() {
        let (mask, bbox) = parse_feature_mask_or_null(feat, fc_crs)
            .map_err(|e| wrap_feature_error(idx, feat, e))?;
        parsed.push((Arc::new(mask), bbox));
    }
    let mut extras = Vec::with_capacity(n_feats);
    if n_feats <= 1 {
        for (idx, (mask, bbox)) in parsed.into_iter().enumerate() {
            let (stats, extra) = compute(mask, bbox)
                .await
                .map_err(|e| wrap_feature_error(idx, &features[idx], e))?;
            insert_feature_stats(&mut features[idx], stats);
            extras.push(extra);
        }
    } else {
        let parsed = Arc::new(parsed);
        let next_idx = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let workers = fc_worker_count(n_feats);
        let compute = Arc::new(compute);
        let mut set = tokio::task::JoinSet::new();
        for _ in 0..workers {
            let parsed_c = parsed.clone();
            let next_c = next_idx.clone();
            let compute_c = compute.clone();
            set.spawn(async move {
                let mut local = Vec::new();
                loop {
                    let idx = next_c.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if idx >= parsed_c.len() {
                        break;
                    }
                    let (ref mask, bbox) = parsed_c[idx];
                    let (stats, extra) = compute_c(mask.clone(), bbox)
                        .await
                        .map_err(|e| (idx, e))?;
                    local.push((idx, stats, extra));
                }
                Ok::<_, (usize, EngineError)>(local)
            });
        }
        while let Some(joined) = set.join_next().await {
            let batch = joined
                .map_err(|e| EngineError::BadRequest(format!("statistics worker failed: {e}")))?
                .map_err(|(idx, e)| wrap_feature_error(idx, &features[idx], e))?;
            for (idx, stats, extra) in batch {
                insert_feature_stats(&mut features[idx], stats);
                extras.push(extra);
            }
        }
    }
    Ok(extras)
}

async fn read_limited_body_with_timeout<B>(body: B, timeout_secs: u64) -> Result<Bytes, EngineError>
where
    B: hyper::body::Body,
    B::Error: std::fmt::Display,
{
    use bytes::Buf;
    let read_fut = async move {
        let mut out = bytes::BytesMut::new();
        tokio::pin!(body);
        while let Some(frame_res) = body.frame().await {
            let frame =
                frame_res.map_err(|e| EngineError::BadRequest(format!("body read error: {e}")))?;
            if let Ok(mut data) = frame.into_data() {
                if out.len().saturating_add(data.remaining()) > MAX_POST_BODY_BYTES {
                    return Err(EngineError::PayloadTooLarge(format!(
                        "request body exceeds {} MiB limit",
                        MAX_POST_BODY_BYTES / (1024 * 1024)
                    )));
                }
                while data.has_remaining() {
                    let chunk = data.chunk();
                    out.extend_from_slice(chunk);
                    let n = chunk.len();
                    data.advance(n);
                }
            }
        }
        Ok(out.freeze())
    };
    if timeout_secs == 0 {
        read_fut.await
    } else {
        tokio::time::timeout(std::time::Duration::from_secs(timeout_secs), read_fut)
            .await
            .map_err(|_| {
                EngineError::RequestTimeout(format!(
                    "request body read timed out after {timeout_secs}s"
                ))
            })?
    }
}

async fn read_limited_body<B>(body: B) -> Result<Bytes, EngineError>
where
    B: hyper::body::Body,
    B::Error: std::fmt::Display,
{
    read_limited_body_with_timeout(body, body_timeout_secs()).await
}

/// Open the first readable scene in a mosaic candidate list so a single missing/corrupt scene at
/// index 0 does not fail the entire mosaic tile when subsequent scenes are valid.
async fn open_first_mosaic_cog(
    engine: &TileEngine,
    urls: &[String],
) -> Result<Arc<cogzilla_io::OpenedCog>, EngineError> {
    let mut last_err: Option<EngineError> = None;
    for u in urls.iter().take(16) {
        match engine.store.open_uri(u).await {
            Ok(c) => return Ok(c),
            Err(e) if e.is_fatal() => {
                return Err(EngineError::Io(e));
            }
            Err(e) => last_err = Some(crate::engine::asset_fetch_error(u, e)),
        }
    }
    Err(last_err.unwrap_or_else(|| {
        EngineError::BadGateway("mosaic contains no readable scenes".into())
    }))
}

/// Open the `?url=` rasters; `nodata` (the request's `?nodata=`) replaces each file's own
/// nodata, as GDAL's override does (see [`cogzilla_io::CogStore::with_nodata`]).
async fn open_query_urls(
    engine: &TileEngine,
    urls: &[String],
    nodata: Option<f32>,
) -> Result<Vec<Arc<cogzilla_io::OpenedCog>>, EngineError> {
    if urls.is_empty() {
        return Err(EngineError::BadRequest(
            "missing ?url=... query parameter".to_string(),
        ));
    }
    if urls.len() > 16 {
        return Err(EngineError::BadRequest(format!(
            "too many ?url= parameters ({}, maximum 16)",
            urls.len()
        )));
    }
    if urls.len() == 1 {
        let cog = engine.store.open_uri(&urls[0]).await?;
        return Ok(vec![engine.store.with_nodata(&cog, nodata)]);
    }
    let mut js = tokio::task::JoinSet::new();
    for (idx, u) in urls.iter().enumerate() {
        let store = engine.store.clone();
        let url = u.clone();
        js.spawn(async move { Ok::<_, cogzilla_io::IoError>((idx, store.open_uri(&url).await?)) });
    }
    let mut slots: Vec<Option<Arc<cogzilla_io::OpenedCog>>> = vec![None; urls.len()];
    while let Some(res) = js.join_next().await {
        let (idx, cog) = res.map_err(|e| {
            EngineError::Io(cogzilla_io::IoError::Fs(std::io::Error::other(format!(
                "open_uri task failed: {e}"
            ))))
        })??;
        slots[idx] = Some(cog);
    }
    Ok(slots.into_iter().map(|s| engine.store.with_nodata(&s.unwrap(), nodata)).collect())
}

/// Convenience entry point for serverless / embedded runtimes (such as AWS Lambda)
/// that already hold the request body in memory as `Bytes`.
pub async fn handle_bytes_request(
    engine: Arc<TileEngine>,
    req: Request<Bytes>,
) -> Response<Bytes> {
    let (parts, body) = req.into_parts();
    let req_full = Request::from_parts(parts, Full::new(body));
    let Ok(resp) = handle_request(engine, req_full).await;
    let (rparts, rbody) = resp.into_parts();
    let bytes = rbody
        .collect()
        .await
        .map(|c| c.to_bytes())
        .unwrap_or_default();
    Response::from_parts(rparts, bytes)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PreferredEncoding {
    None,
    Gzip,
    Brotli,
}

/// Negotiate `Accept-Encoding` across all header lines per RFC 9110 §5.3 & §12.5.3,
/// supporting `br`, `gzip`/`x-gzip`, `identity`, and `*`. Prefers `br` over `gzip` when both
/// have equal explicit positive q-values, and falls back to `gzip` for bare `*`.
fn negotiate_encoding<'a>(vals: impl IntoIterator<Item = &'a str>) -> PreferredEncoding {
    let mut br_q: Option<f32> = None;
    let mut gzip_q: Option<f32> = None;
    let mut identity_q: Option<f32> = None;
    let mut star_q: Option<f32> = None;
    for val in vals {
        for part in val.split(',') {
            let mut iter = part.split(';');
            let enc = iter.next().unwrap_or("").trim();
            let is_br = enc.eq_ignore_ascii_case("br");
            let is_gzip = enc.eq_ignore_ascii_case("gzip") || enc.eq_ignore_ascii_case("x-gzip");
            let is_identity = enc.eq_ignore_ascii_case("identity");
            let is_star = enc == "*";
            if !is_br && !is_gzip && !is_identity && !is_star {
                continue;
            }
            let mut q_val = 1.0f32;
            let mut valid = true;
            for param in iter {
                let p = param.trim();
                if let Some(qv) = p.strip_prefix("q=").or_else(|| p.strip_prefix("Q=")) {
                    match qv.trim().parse::<f32>() {
                        Ok(q) if q.is_finite() && (0.0..=1.0).contains(&q) => q_val = q,
                        _ => {
                            valid = false;
                            break;
                        }
                    }
                }
            }
            if !valid {
                continue;
            }
            if is_br {
                br_q = Some(br_q.map_or(q_val, |prev| prev.max(q_val)));
            } else if is_gzip {
                gzip_q = Some(gzip_q.map_or(q_val, |prev| prev.max(q_val)));
            } else if is_identity {
                identity_q = Some(identity_q.map_or(q_val, |prev| prev.max(q_val)));
            } else {
                star_q = Some(star_q.map_or(q_val, |prev| prev.max(q_val)));
            }
        }
    }
    let eff_br = br_q.or(star_q).unwrap_or(0.0);
    let eff_gzip = gzip_q.or(star_q).unwrap_or(0.0);
    let best_compressed = eff_br.max(eff_gzip);
    if best_compressed <= 0.0 || identity_q.is_some_and(|iq| iq > best_compressed) {
        return PreferredEncoding::None;
    }
    if eff_br > eff_gzip {
        PreferredEncoding::Brotli
    } else if eff_gzip > eff_br {
        PreferredEncoding::Gzip
    } else if br_q.is_some() {
        PreferredEncoding::Brotli
    } else {
        PreferredEncoding::Gzip
    }
}

fn is_compressible_content_type(ct: &str) -> bool {
    let media = ct.split(';').next().unwrap_or("").trim();
    media.eq_ignore_ascii_case("application/json")
        || media.eq_ignore_ascii_case("application/geo+json")
        || media.eq_ignore_ascii_case("application/xml")
        || media.eq_ignore_ascii_case("application/vnd.oai.openapi+json")
        || media.eq_ignore_ascii_case("application/x-binary")
        || media.eq_ignore_ascii_case("application/octet-stream")
        || media.eq_ignore_ascii_case("image/tiff")
        || media.eq_ignore_ascii_case("image/svg+xml")
        || media.starts_with("text/")
}

const X_REQUEST_ID: HeaderName = HeaderName::from_static("x-request-id");

/// Extract a valid caller-supplied `X-Request-Id` header (1..=128 printable non-whitespace ASCII
/// bytes) or generate a compact lock-free hex ID (`<8-hex-process-seed>-<12-hex-seq>`).
fn resolve_request_id(headers: &hyper::HeaderMap) -> HeaderValue {
    if let Some(hv) = headers.get(&X_REQUEST_ID) {
        let bytes = hv.as_bytes();
        if !bytes.is_empty()
            && bytes.len() <= 128
            && bytes.iter().all(|&b| (0x21..=0x7E).contains(&b))
        {
            return hv.clone();
        }
    }
    static SEED: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    let seed = *SEED.get_or_init(|| {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        let pid = std::process::id() as u64;
        (nanos ^ (nanos >> 32) ^ (pid.wrapping_mul(0x9E37_79B9))) as u32
    });
    let seq = SEQ.fetch_add(1, Ordering::Relaxed) & 0x0000_FFFF_FFFF_FFFF;
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut buf = [0u8; 21];
    for i in 0..8 {
        buf[7 - i] = HEX[((seed >> (i * 4)) & 0xF) as usize];
    }
    buf[8] = b'-';
    for i in 0..12 {
        buf[20 - i] = HEX[((seq >> (i * 4)) & 0xF) as usize];
    }
    HeaderValue::from_bytes(&buf).expect("ASCII hex request id")
}

pub async fn handle_request<B>(
    engine: Arc<TileEngine>,
    mut req: Request<B>,
) -> Result<Response<Full<Bytes>>, Infallible>
where
    B: hyper::body::Body,
    B::Error: std::fmt::Display,
{
    let t0 = std::time::Instant::now();
    let req_id = resolve_request_id(req.headers());
    let req_method = req.method().clone();
    // O(1) `Bytes`-backed URI clone (zero heap allocation) for structured access logging.
    let req_uri = req.uri().clone();
    let is_head = req_method == Method::HEAD;
    let is_post = req_method == Method::POST;
    let is_get_or_head = matches!(req_method, Method::GET | Method::HEAD);
    let is_health_or_metrics =
        matches!(req_uri.path(), "/healthz" | "/readyz" | "/metrics" | "/stats");
    let is_control_plane = is_health_or_metrics
        || matches!(
            req_uri.path(),
            "/admin/clear-cache" | "/searches" | "/searches/list" | "/searches/register"
        );
    let bypass_aimd = is_control_plane || req_method == Method::OPTIONS;
    let mut aimd_permit = if bypass_aimd {
        None
    } else {
        match engine.aimd.try_acquire() {
            Some(permit) => Some(permit),
            None => {
                let body = if is_head {
                    Bytes::new()
                } else {
                    Bytes::from_static(br#"{"detail":"server overloaded; retry shortly"}"#)
                };
                let mut resp = plain_resp(
                    StatusCode::TOO_MANY_REQUESTS,
                    body,
                    "application/json",
                );
                resp.headers_mut()
                    .insert(hyper::header::RETRY_AFTER, HeaderValue::from_static("1"));
                if let Ok(hv) = HeaderValue::from_str(&engine.aimd.current_limit().to_string()) {
                    resp.headers_mut()
                        .insert(HeaderName::from_static("x-cogzilla-aimd-limit"), hv);
                }
                if tracing::enabled!(tracing::Level::DEBUG) {
                    let dur_ms = t0.elapsed().as_secs_f64() * 1000.0;
                    tracing::debug!(
                        request_id = req_id.to_str().unwrap_or(""),
                        method = %req_method,
                        path = %req_uri.path(),
                        status = 429u16,
                        dur_ms = format_args!("{dur_ms:.2}"),
                        "http request"
                    );
                }
                resp.headers_mut().insert(X_REQUEST_ID, req_id);
                resp.headers_mut().insert(
                    hyper::header::ACCESS_CONTROL_EXPOSE_HEADERS,
                    HeaderValue::from_static("X-Request-Id"),
                );
                return Ok(resp);
            }
        }
    };
    let preferred_enc = negotiate_encoding(
        req.headers()
            .get_all(hyper::header::ACCEPT_ENCODING)
            .iter()
            .filter_map(|v| v.to_str().ok()),
    );
    engine.sync_tile_cache_epoch();
    let retry_head = if is_get_or_head {
        Some((req.method().clone(), req.uri().clone(), req.headers().clone()))
    } else {
        None
    };
    // `f=html` or `Accept: text/html` on a route that describes something: its JSON, written out as a page.
    let offers_page = is_get_or_head && html::offers_html(req.uri().path());
    let accept_hdr = req
        .headers()
        .get(hyper::header::ACCEPT)
        .and_then(|v| v.to_str().ok());
    let raw_q = req.uri().query().unwrap_or("");
    let colormap_prefers_image = {
        let mut parts = req.uri().path().trim_matches('/').split('/').filter(|s| !s.is_empty());
        matches!((parts.next(), parts.next(), parts.next()), (Some("colorMaps"), Some(_), None))
            && html::explicit_format(raw_q).is_none()
            && accept_hdr.and_then(html::negotiate_colormap_accept).is_some()
    };
    let as_page = (offers_page
        && !colormap_prefers_image
        && html::wants_html(raw_q, accept_hdr))
    .then(|| (req.uri().path().to_string(), raw_q.to_string()));
    // Prevent a cached JSON ETag in `If-None-Match` from short-circuiting an HTML page request to 304.
    if as_page.is_some() {
        req.headers_mut().remove(IF_NONE_MATCH);
    }
    let mut resp = Box::pin(handle_request_inner(engine.clone(), req, is_head)).await?;
    if resp.headers_mut().remove("x-cogzilla-retry").is_some()
        && let Some((m, u, h)) = retry_head
    {
        engine.sync_tile_cache_epoch();
        let mut retry_req = Request::new(Full::new(Bytes::new()));
        *retry_req.method_mut() = m;
        *retry_req.uri_mut() = u;
        *retry_req.headers_mut() = h;
        resp = Box::pin(handle_request_inner(engine.clone(), retry_req, is_head)).await?;
        resp.headers_mut().remove("x-cogzilla-retry");
    }
    if let Some(permit) = aimd_permit.as_mut()
        && (resp.status().is_client_error()
            || resp.status().is_server_error()
            || resp.status() == StatusCode::NOT_MODIFIED
            || resp.status() == StatusCode::NO_CONTENT)
    {
        permit.ignore_rtt();
    }
    drop(aimd_permit);
    if let Some((path, query)) = as_page
        && resp.status() == StatusCode::OK
        && resp
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .is_some_and(|ct| ct.starts_with("application/json"))
    {
        let (mut parts, body) = resp.into_parts();
        let raw = body.collect().await.map(|c| c.to_bytes()).unwrap_or_default();
        resp = match serde_json::from_slice::<serde_json::Value>(&raw) {
            Ok(doc) => {
                let page = html::page(&path, &query, &doc);
                parts.headers.insert(CONTENT_TYPE, HeaderValue::from_static("text/html; charset=utf-8"));
                parts.headers.remove(hyper::header::CONTENT_LENGTH);
                parts.headers.remove(ETAG);
                Response::from_parts(parts, Full::new(Bytes::from(page)))
            }
            Err(_) => Response::from_parts(parts, Full::new(raw)),
        };
    }
    if offers_page && (resp.status().is_success() || resp.status() == StatusCode::NOT_MODIFIED) {
        html::append_vary_accept(resp.headers_mut());
    }
    let dur_ms = t0.elapsed().as_secs_f64() * 1000.0;
    if let Ok(hv) = HeaderValue::from_str(&format!("total;dur={dur_ms:.2}")) {
        resp.headers_mut().insert(
            hyper::header::HeaderName::from_static("server-timing"),
            hv,
        );
    }
    let status = resp.status();
    if status.is_server_error() {
        tracing::warn!(
            request_id = req_id.to_str().unwrap_or(""),
            method = %req_method,
            path = %req_uri.path(),
            status = status.as_u16(),
            dur_ms = format_args!("{dur_ms:.2}"),
            "http request failed"
        );
    } else if tracing::enabled!(tracing::Level::DEBUG) {
        tracing::debug!(
            request_id = req_id.to_str().unwrap_or(""),
            method = %req_method,
            path = %req_uri.path(),
            status = status.as_u16(),
            dur_ms = format_args!("{dur_ms:.2}"),
            "http request"
        );
    }
    resp.headers_mut().insert(X_REQUEST_ID, req_id);
    if !resp
        .headers()
        .contains_key(hyper::header::ACCESS_CONTROL_EXPOSE_HEADERS)
    {
        resp.headers_mut().insert(
            hyper::header::ACCESS_CONTROL_EXPOSE_HEADERS,
            HeaderValue::from_static("X-Request-Id"),
        );
    }
    if is_post {
        // RFC 9110: POST responses depend on the request body and must not carry public Cache-Control/ETag.
        resp.headers_mut().remove(CACHE_CONTROL);
        resp.headers_mut().remove(ETAG);
    } else if is_get_or_head
        && resp.status() == StatusCode::OK
        && !is_health_or_metrics
        && !resp.headers().contains_key(CACHE_CONTROL)
    {
        resp.headers_mut()
            .insert(CACHE_CONTROL, cache_control_header().clone());
    }

    let compressible = resp.status() == StatusCode::OK
        && resp
            .headers()
            .get(CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .is_some_and(is_compressible_content_type);
    if compressible {
        if let Some(existing) = resp.headers().get(hyper::header::VARY).and_then(|v| v.to_str().ok()) {
            if !existing.to_ascii_lowercase().contains("accept-encoding")
                && let Ok(hv) = HeaderValue::from_str(&format!("{existing}, Accept-Encoding")) {
                    resp.headers_mut().insert(hyper::header::VARY, hv);
                }
        } else {
            resp.headers_mut()
                .insert(hyper::header::VARY, HeaderValue::from_static("Accept-Encoding"));
        }
        if preferred_enc != PreferredEncoding::None
            && !is_head
            && !resp.headers().contains_key(hyper::header::CONTENT_ENCODING)
        {
            let (mut parts, body) = resp.into_parts();
            let raw = body
                .collect()
                .await
                .map(|c| c.to_bytes())
                .unwrap_or_default();
            let compressed = if raw.len() >= 500 {
                match preferred_enc {
                    PreferredEncoding::Brotli => {
                        cogzilla_codec::compress_brotli(&raw).ok().map(|b| (b, "br"))
                    }
                    PreferredEncoding::Gzip => {
                        cogzilla_codec::compress_gzip(&raw).ok().map(|b| (b, "gzip"))
                    }
                    PreferredEncoding::None => None,
                }
            } else {
                None
            };
            if let Some((enc_bytes, enc_name)) = compressed
                && enc_bytes.len() < raw.len()
            {
                parts.headers.insert(
                    hyper::header::CONTENT_ENCODING,
                    HeaderValue::from_static(enc_name),
                );
                if let Ok(hv) = HeaderValue::from_str(&enc_bytes.len().to_string()) {
                    parts.headers.insert(hyper::header::CONTENT_LENGTH, hv);
                }
                resp = Response::from_parts(parts, Full::new(Bytes::from(enc_bytes)));
            } else {
                resp = Response::from_parts(parts, Full::new(raw));
            }
        }
    }

    if is_head {
        let (mut parts, body) = resp.into_parts();
        if parts.status != StatusCode::NOT_MODIFIED && parts.status != StatusCode::NO_CONTENT
            && let Some(len) = hyper::body::Body::size_hint(&body).exact()
                && let Ok(hv) = HeaderValue::from_str(&len.to_string()) {
                    parts.headers.insert(hyper::header::CONTENT_LENGTH, hv);
                }
        return Ok(Response::from_parts(parts, Full::new(Bytes::new())));
    }
    Ok(resp)
}

async fn handle_request_inner<B>(
    engine: Arc<TileEngine>,
    req: Request<B>,
    is_head: bool,
) -> Result<Response<Full<Bytes>>, Infallible>
where
    B: hyper::body::Body,
    B::Error: std::fmt::Display,
{
    let proto = req
        .headers()
        .get("x-forwarded-proto")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.split(',').next())
        .map(|s| s.trim())
        .filter(|&s| s.eq_ignore_ascii_case("https") || s.eq_ignore_ascii_case("http"))
        .map(|s| s.to_ascii_lowercase())
        .or_else(|| {
            req.uri()
                .scheme_str()
                .filter(|&s| s.eq_ignore_ascii_case("https") || s.eq_ignore_ascii_case("http"))
                .map(|s| s.to_ascii_lowercase())
        })
        .unwrap_or_else(|| "http".to_string());
    let valid_host = |s: &&str| {
        !s.is_empty()
            && !s.contains('@')
            && s.parse::<hyper::http::uri::Authority>().is_ok()
    };
    let host_raw = req
        .headers()
        .get("x-forwarded-host")
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.split(',').next())
        .map(|s| s.trim())
        .filter(valid_host)
        .or_else(|| {
            req.headers()
                .get("host")
                .and_then(|v| v.to_str().ok())
                .map(|s| s.trim())
                .filter(valid_host)
        })
        .or_else(|| req.uri().authority().map(|a| a.as_str()))
        .unwrap_or("127.0.0.1:8080");
    let forwarded_prefix = req
        .headers()
        .get("x-forwarded-prefix")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.trim().trim_end_matches('/'))
        .filter(|s| {
            !s.is_empty()
                && s.starts_with('/')
                && !s.contains("//")
                && !s.contains("..")
                && s.bytes()
                    .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'/' | b'-' | b'_' | b'.' | b'~'))
        })
        .map(|s| s.to_string())
        .unwrap_or_default();
    let host = format!("{proto}://{host_raw}{forwarded_prefix}");
    let if_none_match = req
        .headers()
        .get(IF_NONE_MATCH)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());
    let method = if is_head {
        Method::GET
    } else {
        req.method().clone()
    };
    let is_post = method == Method::POST;
    let if_none_match = if is_post { None } else { if_none_match };
    let path = req.uri().path().to_string();
    let raw_query = req.uri().query().unwrap_or("").to_string();
    let raw_query = if path.starts_with("/searches/") || path.starts_with("/collections/") {
        pgstac::plain_numbers(&raw_query)
    } else {
        raw_query
    };

    if method == Method::OPTIONS {
        let mut resp = plain_resp(StatusCode::NO_CONTENT, Bytes::new(), "text/plain");
        resp.headers_mut().insert(
            hyper::header::ACCESS_CONTROL_ALLOW_METHODS,
            HeaderValue::from_static("GET, POST, HEAD, OPTIONS"),
        );
        resp.headers_mut().insert(
            hyper::header::ACCESS_CONTROL_ALLOW_HEADERS,
            HeaderValue::from_static("Content-Type, If-None-Match, Authorization, X-Request-Id"),
        );
        return Ok(resp);
    }

    if method == Method::GET && path == "/healthz" {
        return Ok(plain_resp(StatusCode::OK, Bytes::from_static(b"ok"), "text/plain"));
    }

    if method == Method::GET && path == "/readyz" {
        if let Some(pg) = engine.pgstac() {
            match tokio::time::timeout(std::time::Duration::from_secs(3), pg.version()).await {
                Ok(Ok(_)) => {}
                Ok(Err(e)) => {
                    tracing::warn!(error = %e, "readiness check failed: pgSTAC unreachable");
                    let mut resp = plain_resp(
                        StatusCode::SERVICE_UNAVAILABLE,
                        Bytes::from_static(b"not ready: pgSTAC unreachable"),
                        "text/plain",
                    );
                    resp.headers_mut()
                        .insert(CACHE_CONTROL, HeaderValue::from_static("no-store"));
                    return Ok(resp);
                }
                Err(_) => {
                    tracing::warn!("readiness check failed: pgSTAC timed out");
                    let mut resp = plain_resp(
                        StatusCode::SERVICE_UNAVAILABLE,
                        Bytes::from_static(b"not ready: pgSTAC timed out"),
                        "text/plain",
                    );
                    resp.headers_mut()
                        .insert(CACHE_CONTROL, HeaderValue::from_static("no-store"));
                    return Ok(resp);
                }
            }
        }
        let mut resp = plain_resp(StatusCode::OK, Bytes::from_static(b"ok"), "text/plain");
        resp.headers_mut()
            .insert(CACHE_CONTROL, HeaderValue::from_static("no-store"));
        return Ok(resp);
    }

    if method == Method::GET && (path == "/conformance" || path == "/conformance/") {
        let conf = serde_json::json!({
            "conformsTo": [
                "http://www.opengis.net/spec/ogcapi-common-1/1.0/conf/core",
                "http://www.opengis.net/spec/ogcapi-common-1/1.0/conf/landing-page",
                "http://www.opengis.net/spec/ogcapi-common-1/1.0/conf/oas30",
                "http://www.opengis.net/spec/ogcapi-common-1/1.0/conf/json",
                "http://www.opengis.net/spec/ogcapi-common-1/1.0/conf/html",
                "http://www.opengis.net/spec/ogcapi-tiles-1/1.0/conf/core",
                "http://www.opengis.net/spec/ogcapi-tiles-1/1.0/conf/tileset",
                "http://www.opengis.net/spec/ogcapi-tiles-1/1.0/conf/tilesets-list",
                "http://www.opengis.net/spec/ogcapi-tiles-1/1.0/conf/dataset-tilesets",
                "http://www.opengis.net/spec/ogcapi-tiles-1/1.0/conf/geodata-tilesets",
                "http://www.opengis.net/spec/ogcapi-tiles-1/1.0/conf/png",
                "http://www.opengis.net/spec/ogcapi-tiles-1/1.0/conf/jpeg",
                "http://www.opengis.net/spec/ogcapi-tiles-1/1.0/conf/tiff",
                "http://www.opengis.net/spec/ogcapi-maps-1/1.0/conf/core",
                "http://www.opengis.net/spec/ogcapi-maps-1/1.0/conf/crs",
                "http://www.opengis.net/spec/ogcapi-maps-1/1.0/conf/scaling",
                "http://www.opengis.net/spec/ogcapi-maps-1/1.0/conf/spatial-subsetting",
                "http://www.opengis.net/spec/ogcapi-maps-1/1.0/conf/png",
                "http://www.opengis.net/spec/ogcapi-maps-1/1.0/conf/jpeg",
                "http://www.opengis.net/spec/ogcapi-maps-1/1.0/conf/tiff",
                "http://www.opengis.net/spec/tms/2.0/conf/tilematrixset",
                "http://www.opengis.net/spec/tms/2.0/conf/json-tilematrixset",
                "http://www.opengis.net/spec/tms/2.0/conf/tilematrixsetlimits"
            ]
        });
        return Ok(plain_resp(
            StatusCode::OK,
            Bytes::from(conf.to_string()),
            "application/json",
        ));
    }

    if method == Method::GET && (path == "/" || path == "/index.html") {
        let wants_json = raw_query.split('&').any(|kv| kv == "f=json")
            || (path == "/"
                && req
                    .headers()
                    .get("accept")
                    .and_then(|v| v.to_str().ok())
                    .is_some_and(|a| a.contains("application/json") && !a.contains("text/html")));
        if wants_json {
            let landing = serde_json::json!({
                "title": "cogzilla",
                "description": "Zero-GDAL Rust + SIMD drop-in TiTiler COG, STAC & MosaicJSON tile server",
                "links": [
                    { "href": format!("{host}/"), "rel": "self", "type": "application/json", "title": "Landing Page" },
                    { "href": format!("{host}/api"), "rel": "service-desc", "type": "application/vnd.oai.openapi+json;version=3.0", "title": "OpenAPI definition" },
                    { "href": format!("{host}/api.html"), "rel": "service-doc", "type": "text/html", "title": "Swagger UI" },
                    { "href": format!("{host}/conformance"), "rel": "conformance", "type": "application/json", "title": "OGC API conformance classes" },
                    { "href": format!("{host}/tileMatrixSets"), "rel": "http://www.opengis.net/def/rel/ogc/1.0/tiling-schemes", "type": "application/json", "title": "TileMatrixSets" },
                    { "href": format!("{host}/colorMaps"), "rel": "data", "type": "application/json", "title": "ColorMaps" },
                    { "href": format!("{host}/algorithms"), "rel": "data", "type": "application/json", "title": "Algorithms" }
                ]
            });
            let mut resp = plain_resp(
                StatusCode::OK,
                Bytes::from(landing.to_string()),
                "application/json",
            );
            resp.headers_mut()
                .insert(hyper::header::VARY, HeaderValue::from_static("Accept"));
            return Ok(resp);
        }
        let base_path_js = serde_json::to_string(&forwarded_prefix)
            .unwrap_or_else(|_| "\"\"".to_string())
            .replace("</", "<\\/");
        let html = include_str!("landing.html")
            .replace("{{VERSION}}", env!("CARGO_PKG_VERSION"))
            .replace("{{BASE_PATH}}", &forwarded_prefix)
            .replace("{{BASE_PATH_JS}}", &base_path_js);
        let mut resp = plain_resp(StatusCode::OK, Bytes::from(html), "text/html; charset=utf-8");
        if path == "/" {
            resp.headers_mut()
                .insert(hyper::header::VARY, HeaderValue::from_static("Accept"));
        }
        return Ok(resp);
    }

    if method == Method::GET && (path == "/openapi.json" || path == "/api") {
        let ct = if path == "/api" {
            "application/vnd.oai.openapi+json;version=3.1"
        } else {
            "application/json"
        };
        // The shared document, with the pgSTAC routes' paths beside its own.
        let mut body = crate::openapi::openapi_json_bytes();
        if let Ok(mut spec) = serde_json::from_slice::<serde_json::Value>(&body)
            && let (Some(paths), serde_json::Value::Object(extra)) =
                (spec["paths"].as_object_mut(), pgstac::openapi_paths())
        {
            paths.extend(extra);
            body = Bytes::from(spec.to_string());
        }
        return Ok(plain_resp(StatusCode::OK, body, ct));
    }

    if method == Method::GET && (path == "/docs" || path == "/api.html") {
        let spec_url_js = serde_json::to_string(&format!("{forwarded_prefix}/openapi.json"))
            .unwrap_or_else(|_| "\"/openapi.json\"".to_string())
            .replace("</", "<\\/");
        let html = format!(
            r#"<!DOCTYPE html>
<html><head><title>cogzilla API Docs</title>
<link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css"/>
</head><body><div id="swagger-ui"></div>
<script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
<script>SwaggerUIBundle({{url:{spec_url_js},dom_id:'#swagger-ui'}});</script>
</body></html>"#
        );
        return Ok(plain_resp(
            StatusCode::OK,
            Bytes::from(html),
            "text/html; charset=utf-8",
        ));
    }

    if method == Method::GET && path == "/stats" {
        let body = metrics::render_stats_json(&engine);
        return Ok(plain_resp(StatusCode::OK, Bytes::from(body), "application/json"));
    }

    if method == Method::GET && path == "/metrics" {
        let prom = metrics::render_prometheus_metrics(&engine);
        return Ok(plain_resp(
            StatusCode::OK,
            Bytes::from(prom),
            "text/plain; version=0.0.4; charset=utf-8",
        ));
    }

    if method == Method::POST && path == "/admin/clear-cache" {
        engine.store.clear_open_uris();
        engine.store.cache.clear();
        engine.tile_cache.clear();
        engine.clear_gpu_caches();
        engine.stac_cache.clear();
        engine.mosaicjson_cache.clear();
        zarr::clear_zarr_cache();
        if let Some(pg) = engine.pgstac() {
            pg.clear_caches();
        }
        return Ok(plain_resp(StatusCode::OK, Bytes::from_static(b"cleared"), "text/plain"));
    }

    let segs: Vec<&str> = path.trim_matches('/').split('/').collect();
    let req_params = query::parse_family_request(segs.first().copied(), &raw_query);
    let query::FamilyRequestParams {
        common: mut pq,
        family: req_family,
    } = req_params;
    let req_ctx = RequestCtx {
        engine: &engine,
        path: &path,
        raw_query: &raw_query,
        host: &host,
        forwarded_prefix: &forwarded_prefix,
        if_none_match: if_none_match.as_deref(),
    };

    // titiler-pgstac's item routes, /collections/{c}/items/{i}/..., are the /stac router over
    // an item read from pgSTAC instead of from `url`.
    let mut pgstac_item: Option<Arc<StacItem>> = None;
    let mut pgstac_item_extent: Option<pgstac::ItemExtent> = None;
    let mut stac_prefix = String::from("stac");
    let item_preview: Option<String>;
    let item_rest: Option<Vec<&str>> = match segs.as_slice() {
        ["collections", c, "items", i, rest @ ..] if !rest.is_empty() => {
            let ctx = pgstac::Ctx::from_request(req_ctx);
            // A feature posted for an item's statistics is read here: those are worked out by
            // the search routes, from the item alone, and never reach `/stac`.
            if method == Method::POST && rest == ["statistics"] {
                let body = match read_limited_body(req.into_body()).await {
                    Ok(body) => body,
                    Err(e) => return Ok(pgstac::engine_error(&e)),
                };
                return Ok(match Box::pin(pgstac::item_route(&ctx, &method, c, i, rest, &pq, Some(body))).await {
                    pgstac::ItemRoute::Done(resp) => resp,
                    pgstac::ItemRoute::Stac(..) => pgstac::bad_request("an item's statistics of a posted feature are worked out from the item alone".into()),
                });
            }
            match Box::pin(pgstac::item_route(&ctx, &method, c, i, rest, &pq, None)).await {
                pgstac::ItemRoute::Done(resp) => return Ok(pgstac_cacheable(resp, &method)),
                pgstac::ItemRoute::Stac(item, extent) => {
                    pgstac_item = Some(item);
                    pgstac_item_extent = extent;
                    // An item is one image: titiler-pgstac's item routes have no
                    // `pixel_selection`, and pass over any value of it.
                    pq.keys.retain(|k| k != "pixel_selection");
                    pq.errors.retain(|e| !e.starts_with("unsupported pixel_selection"));
                    // What the route does not read, titiler-pgstac passes over: not checked,
                    // and not seen by the routes' own checks of what each takes.
                    pq.errors.retain(|e| !pgstac::item_passes_over_error(&method, rest, e));
                    pq.keys.retain(|k| !pgstac::item_passes_over(&method, rest, k));
                    if pgstac::item_passes_over(&method, rest, "algorithm") {
                        (pq.algorithm, pq.algorithm_error, pq.algorithm_params_unread) = (None, None, None);
                    }
                    pq.pixel_selection = Selection::First;
                    // Band math on an item's point is applied to the `/stac` answer by
                    // `pgstac::item_fixup`, with TiTiler's names: `/stac` must not apply it first.
                    // An expression is over every band read, so the request's `bidx` does
                    // not narrow them first; each asset's own selection still applies.
                    match pgstac::item_asset_options(rest, &raw_query) {
                        Err(resp) => return Ok(resp),
                        Ok(Some((per_asset, own))) => {
                            pq.bidx = own;
                            pq.asset_bidx = per_asset;
                            pq.forget_asset_entries();
                        }
                        Ok(None) => {}
                    }
                    let pairs = pgstac::query_pairs(&raw_query);
                    let has_asset_math = pgstac::asset_math_asked(&pairs);
                    // Each asset's band math on a point is applied by `pgstac::item_fixup` to
                    // every band of the asset, named as such; `asset_as_band` names them there.
                    // A preview's stack is planned over the `assets=` entries, each with its own
                    // band math and bands (`expand_stac_asset_bidx`).
                    // titiler-pgstac reads the listed assets only, not one `asset_bidx` adds.
                    if (pgstac::item_preview_route(rest) || matches!(rest, ["statistics"]))
                        && pq.asset_entries.iter().any(|e| e.listed)
                    {
                        pq.asset_entries.retain(|e| e.listed);
                        pq.assets = pq.asset_entries.iter().map(|e| e.name.clone()).collect();
                    }
                    if matches!(rest, ["point", _]) && has_asset_math {
                        pgstac::point_asset_math_reads(&mut pq, &raw_query);
                    }
                    if matches!(rest, ["point", _]) && pq.expr_raw.take().is_some() {
                        pq.bidx.clear();
                    }
                    stac_prefix = format!("collections/{c}/items/{i}");
                    let mut as_stac = vec!["stac"];
                    // A preview with a side of zero in its path is the preview sized from
                    // the query, which is what works such a side out.
                    item_preview = pgstac::item_preview_by_query(rest, &mut pq);
                    match &item_preview {
                        Some(preview) => as_stac.push(preview),
                        None => as_stac.extend_from_slice(rest),
                    }
                    Some(as_stac)
                }
            }
        }
        _ => None,
    };
    let segs = item_rest.unwrap_or(segs);

    if let Err(msg) = validate_query(&segs, &pq) {
        if pgstac_item.is_none() {
            let body = serde_json::json!({
                "detail": [
                    {
                        "type": "value_error",
                        "loc": ["query"],
                        "msg": msg,
                    }
                ]
            })
            .to_string();
            return Ok(plain_resp(
                StatusCode::UNPROCESSABLE_ENTITY,
                Bytes::from(body),
                "application/json",
            ));
        }
        // Item routes answer in JSON, and do band math on point values themselves.
        let point_math = segs.get(1) == Some(&"point") && msg.contains("'expression'");
        if !point_math {
            return Ok(pgstac::bad_request(msg));
        }
    }

    // titiler-pgstac's routes: /searches/... and /collections/...
    if matches!(segs.first(), Some(&("searches" | "collections"))) {
        let list = matches!(segs[..], ["searches"] | ["searches", "" | "list"]);
        let is_get = method == Method::GET;
        let cacheable = is_get && !list;
        let mut resp = Box::pin(pgstac::handle(req_ctx, method, &segs, pq, req)).await;
        if list && is_get && resp.status() == StatusCode::OK {
            // Prevent handle_request's default GET Cache-Control from caching live search listings.
            resp.headers_mut()
                .insert(CACHE_CONTROL, HeaderValue::from_static("no-store"));
        }
        return Ok(if cacheable { pgstac_cacheable(resp, &Method::GET) } else { resp });
    }
    if pq.keys.iter().any(|k| k == "pixel_selection") && !route_supports_pixel_selection(&segs) {
        return Ok(json_error_resp(
            StatusCode::BAD_REQUEST,
            "parameter 'pixel_selection' is only supported on mosaic tiles, bbox, feature, and statistics routes",
        ));
    }
    // Parameters the algorithm's model refuses, as TiTiler answers them, on the routes that
    // apply an algorithm.
    if route_applies_algorithm(&segs) {
        if let Some(input) = &pq.algorithm_params_unread {
            let body = serde_json::json!({"detail": [{
                "type": "value_error",
                "loc": ["query", "algorithm_params"],
                "msg": "Value error, invalid JSON content",
                "input": input,
            }]});
            return Ok(plain_resp(StatusCode::UNPROCESSABLE_ENTITY, Bytes::from(body.to_string()), "application/json"));
        }
        if let Some(msg) = pq.algorithm_error.as_deref() {
            return Ok(json_error_resp(StatusCode::BAD_REQUEST, msg));
        }
    }

    // ------------------------------------------------------------------------
    // MosaicJSON & Dynamic Spatial Catalog Router (`/mosaicjson/*`, `/mosaic/*`)
    // ------------------------------------------------------------------------
    if matches!(segs.first(), Some(&("mosaicjson" | "mosaic"))) {
        return Ok(Box::pin(mosaicjson::handle(req_ctx, method, &segs, pq, req)).await);
    }


    // ------------------------------------------------------------------------
    // Pre-registered fast path: GET /tiles/{dataset_or_scene}/{z}/{x}/{y}[@scale][.{ext}]
    // ------------------------------------------------------------------------
    if method == Method::GET && segs.first() == Some(&"tiles") && segs.len() == 5 {
        let ds = segs[1];
        let parsed = (
            parse_u8_like(segs[2]),
            parse_u32_like(segs[3]),
            parse_y_scale_ext(segs[4]),
        );
        let cache_key = format!("{path}?{raw_query}");
        let res = match parsed {
            (Some(z), Some(x), Some((y, sc, ext))) => {
                if z > 30 {
                    Err(EngineError::BadRequest(format!(
                        "invalid zoom level z={z} (expected 0..=30)"
                    )))
                } else {
                    let cached = engine
                        .tile_cache
                        .get_or_render(cache_key, || async {
                            let cogs: Vec<Arc<cogzilla_io::OpenedCog>> =
                                if let Some(scene) = engine.get_scene(ds) {
                                    scene
                                } else if let Some(cog) = engine.store.get_registered(ds) {
                                    vec![cog]
                                } else {
                                    return Err(EngineError::NotFound(ds.to_string()));
                                };
                            let db = if cogs.len() > 1 {
                                cogs.len() as u16
                            } else {
                                cogs[0].index.full_res().samples_per_pixel
                            };
                            let st = widest_sample_type(&cogs);
                            let (mut treq, ct) = build_tile_request(
                                Some(&engine),
                                TileCoord { z, x, y },
                                sc,
                                ext,
                                &pq,
                                db,
                                st,
                                false,
                                Some(&cogs[0]),
                            )?;
                            if pq.unscale {
                                apply_unscale_cogs(&mut treq.bands, &cogs, None);
                            }
                            let needs_general_path = pq.nodata.is_some()
                                || pq.custom_colormap_json.is_some()
                                || pq.buffer > 0.0
                                || (!pq.return_mask
                                && matches!(treq.format, OutputFormat::Npy | OutputFormat::NpyF64))
                                || (pq.color_formula.is_some()
                                    && !matches!(
                                        treq.bands,
                                        BandRequest::Rgb { .. }
                                            | BandRequest::SarRgb { .. }
                                            | BandRequest::CustomExprRgb { .. }
                                            | BandRequest::Terrain { .. }
                                    ))
                                || (treq.tile_size != 256 && treq.tile_size != 512)
                                || (matches!(
            treq.bands,
            BandRequest::CustomExprRgb { .. }
                | BandRequest::CustomExprMulti { .. }
                | BandRequest::MultiBand { .. }
        ) && !matches!(treq.format, OutputFormat::RawF64 | OutputFormat::NpyF64))
                                || has_embedded_palette(&pq, &cogs[0], &treq.bands)
                                || !crate::engine::cogs_share_base_grid(&cogs);
                            if needs_general_path {
                                let custom_lut = resolve_custom_lut(&pq, &cogs[0], &treq.bands)?;
                                let cf = pq.parse_color_formula()?;
                                let out_dim = (treq.tile_size as f32 + 2.0 * pq.buffer).round() as u32;
                                let bbox_3857 = web_mercator_buffered_3857_bbox(
                                    z,
                                    x,
                                    y,
                                    pq.buffer,
                                    treq.tile_size,
                                );
                                let data = render_mercator_bbox(
                                    &engine,
                                    &cogs,
                                    bbox_3857,
                                    out_dim,
                                    out_dim,
                                    &treq.bands,
                                    treq.resample,
                                    treq.format,
                                    pq.nodata,
                                    custom_lut.as_ref(),
                                    cf.as_ref(),
                                    pq.return_mask,
                                    pq.unscale,
                                )
                                .await?;
                                let data = wrap_geotiff_fmt(
                                    data,
                                    ct,
                                    treq.format,
                                    out_dim,
                                    out_dim,
                                    bbox_3857,
                                    3857,
                                    pq.return_mask,
                                );
                                Ok((data, ct))
                            } else if (pq.color_formula.is_some() || !pq.return_mask)
                                && !matches!(treq.format, OutputFormat::RawF32 | OutputFormat::RawF64 | OutputFormat::Npy | OutputFormat::NpyF64)
                            {
                                let cf = pq.parse_color_formula()?;
                                let orig_format = treq.format;
                                treq.format = OutputFormat::RawRgba;
                                let raw = engine.render_multi_cog_tile(&cogs, &treq).await?;
                                let dim = treq.tile_size as usize;
                                let data = if let Some(cf) = cf {
                                    let mut buf = raw.to_vec();
                                    cf.apply_buffer_in_place(&mut buf, 4);
                                    encode_rgba_output(dim, dim, &buf, orig_format, pq.return_mask)
                                } else {
                                    encode_rgba_output(dim, dim, &raw, orig_format, pq.return_mask)
                                };
                                let (min_x, min_y, max_x, max_y) = tile_bounds_3857(treq.tile);
                                let data = wrap_geotiff_fmt(
                                    data,
                                    ct,
                                    orig_format,
                                    treq.tile_size,
                                    treq.tile_size,
                                    [min_x, min_y, max_x, max_y],
                                    3857,
                                    pq.return_mask,
                                );
                                Ok((data, ct))
                            } else {
                                let (min_x, min_y, max_x, max_y) = tile_bounds_3857(treq.tile);
                                let b = engine.render_multi_cog_tile(&cogs, &treq).await?;
                                let b = wrap_geotiff_fmt(
                                    b,
                                    ct,
                                    treq.format,
                                    treq.tile_size,
                                    treq.tile_size,
                                    [min_x, min_y, max_x, max_y],
                                    3857,
                                    pq.return_mask,
                                );
                                Ok((b, ct))
                            }
                        })
                        .await;
                    let mut resp = to_http_response_cached(&engine, cached, if_none_match.as_deref());
                    attach_tile_geo_headers(&mut resp, "WebMercatorQuad", z, x, y, sc, &pq);
                    return Ok(resp);
                }
            }
            _ => Err(EngineError::BadRequest(
                "expected /tiles/{dataset}/{z}/{x}/{y}.{png|webp|f32}".to_string(),
            )),
        };
        return Ok(to_http_response(&engine, res, if_none_match.as_deref()));
    }

    // ------------------------------------------------------------------------
    // TiTiler ColorMaps Router: GET /colorMaps, GET /colorMaps/{colorMapId}
    // ------------------------------------------------------------------------
    if method == Method::GET && segs.first() == Some(&"colorMaps") {
        let res = match segs.as_slice() {
            ["colorMaps"] => Ok((
                Bytes::from(list_colormaps_json(&host).to_string()),
                "application/json",
            )),
            ["colorMaps", id] => {
                let negotiated_fmt: Option<&str> = pq.format_param.as_deref().or_else(|| {
                    req.headers()
                        .get(hyper::header::ACCEPT)
                        .and_then(|v| v.to_str().ok())
                        .and_then(html::negotiate_colormap_accept)
                });
                match negotiated_fmt {
                    None | Some("json") | Some("html") => {
                        get_colormap_json(id).map(|v| (Bytes::from(v.to_string()), "application/json"))
                    }
                    Some("npy") => get_colormap_image(
                        id,
                        OutputFormat::Npy,
                        pq.orientation_vertical,
                        pq.width,
                        pq.height,
                    )
                    .map(|b| (b, "application/x-binary")),
                    Some("png") => get_colormap_image(
                        id,
                        OutputFormat::Png,
                        pq.orientation_vertical,
                        pq.width,
                        pq.height,
                    )
                    .map(|b| (b, "image/png")),
                    Some("jpg" | "jpeg") => get_colormap_image(
                        id,
                        OutputFormat::Jpeg { quality: pq.jpeg_quality },
                        pq.orientation_vertical,
                        pq.width,
                        pq.height,
                    )
                    .map(|b| (b, "image/jpeg")),
                    Some("webp") => get_colormap_image(
                        id,
                        OutputFormat::Webp { quality: pq.webp_quality },
                        pq.orientation_vertical,
                        pq.width,
                        pq.height,
                    )
                    .map(|b| (b, "image/webp")),
                    Some(other) => Err(EngineError::BadRequest(format!(
                        "unsupported colormap format '{other}' (use json, html, npy, png, jpg, or webp)"
                    ))),
                }
            }
            _ => Err(EngineError::NotFound(path.clone())),
        };
        return Ok(to_http_response(&engine, res, if_none_match.as_deref()));
    }

    // ------------------------------------------------------------------------
    // TiTiler OGC TileMatrixSets Router: /tileMatrixSets, /tileMatrixSets/{id}
    // ------------------------------------------------------------------------
    if method == Method::GET && segs.first() == Some(&"tileMatrixSets") {
        let res = match segs.as_slice() {
            ["tileMatrixSets"] => Ok((
                Bytes::from(list_tms_json(&host).to_string()),
                "application/json",
            )),
            ["tileMatrixSets", id] => {
                get_tms_json(id).map(|v| (Bytes::from(v.to_string()), "application/json"))
            }
            _ => Err(EngineError::NotFound(path.clone())),
        };
        return Ok(to_http_response(&engine, res, if_none_match.as_deref()));
    }

    // ------------------------------------------------------------------------
    // TiTiler AlgorithmFactory Router: /algorithms, /algorithms/{algorithmId}
    // ------------------------------------------------------------------------
    if method == Method::GET && segs.first() == Some(&"algorithms") {
        let res = match segs.as_slice() {
            ["algorithms"] => Ok((
                Bytes::from(list_algorithms_json(&host).to_string()),
                "application/json",
            )),
            ["algorithms", id] => {
                get_algorithm_json(id).map(|v| (Bytes::from(v.to_string()), "application/json"))
            }
            _ => Err(EngineError::NotFound(path.clone())),
        };
        return Ok(to_http_response(&engine, res, if_none_match.as_deref()));
    }

    // ------------------------------------------------------------------------
    // Full TiTiler STAC Item Router: /stac/* (MultiBaseTilerFactory compatible!)
    // ------------------------------------------------------------------------
    if segs.first() == Some(&"stac") {
        let sub = &segs[1..];
        let stac_ctx = stac::StacContext {
            prefix: &stac_prefix,
            pgstac_item,
            pgstac_item_extent,
        };
        return Ok(Box::pin(stac::handle(req_ctx, method, sub, pq, stac_ctx, req)).await);
    }

    // ------------------------------------------------------------------------
    // Zarr API: /zarr/* (metadata, introspection, raster, and POST zonal routes)
    // ------------------------------------------------------------------------
    if let query::FamilyParams::Zarr(zp) = req_family {
        let sub = &segs[1..];
        return Ok(Box::pin(zarr::handle(req_ctx, method, sub, pq, zp, req)).await);
    }

    // ------------------------------------------------------------------------
    // Full TiTiler Drop-In API: /cog/* (all 19 TiTiler TilerFactory routes!)
    // ------------------------------------------------------------------------
    if segs.first() == Some(&"cog") {
        let sub = &segs[1..];
        return Ok(Box::pin(cog::handle(req_ctx, method, sub, pq, req)).await);
    }

    // 405 Method Not Allowed when a known route is requested with an unsupported HTTP verb
    if matches!(
        path.as_str(),
        "/" | "/index.html"
            | "/healthz"
            | "/readyz"
            | "/metrics"
            | "/conformance"
            | "/conformance/"
            | "/openapi.json"
            | "/api"
            | "/docs"
            | "/api.html"
    ) || matches!(
        segs.first().copied(),
        Some("algorithms" | "colorMaps" | "tileMatrixSets" | "mosaicjson")
    ) {
        let allow = if matches!(segs.as_slice(), ["mosaicjson", "validate" | "statistics" | "feature", ..]) {
            "POST"
        } else {
            "GET, HEAD"
        };
        return Ok(method_not_allowed_resp(allow));
    }

    Ok(json_error_resp(StatusCode::NOT_FOUND, &path))
}

/// A pgSTAC route's answer with the `Cache-Control` the tiles carry, as titiler-pgstac
/// puts one on what it serves: on what a `GET` found, never on an error, and not where
/// one is already set.
fn pgstac_cacheable(mut resp: Response<Full<Bytes>>, method: &Method) -> Response<Full<Bytes>> {
    if *method == Method::GET && resp.status().is_success() && !resp.headers().contains_key(CACHE_CONTROL) {
        resp.headers_mut().insert(CACHE_CONTROL, cache_control_header().clone());
    }
    resp
}

fn cache_control_header() -> &'static HeaderValue {
    crate::config::Config::global().cache_control_header()
}

fn etag_matches(if_none_match: &str, etag: &str) -> bool {
    let trimmed = if_none_match.trim();
    if trimmed == "*" {
        return true;
    }
    let target = etag.strip_prefix("W/").unwrap_or(etag);
    for part in trimmed.split(',') {
        let cand = part.trim().strip_prefix("W/").unwrap_or(part.trim());
        if cand == target {
            return true;
        }
    }
    false
}

fn attach_geo_headers(
    resp: &mut Response<Full<Bytes>>,
    bbox: [f64; 4],
    epsg: Option<u16>,
    is_crs84: bool,
) {
    if resp.status() != StatusCode::OK {
        return;
    }
    if let Ok(hv) = HeaderValue::from_str(&format!(
        "{},{},{},{}",
        bbox[0], bbox[1], bbox[2], bbox[3]
    )) {
        resp.headers_mut()
            .insert(HeaderName::from_static("content-bbox"), hv);
    }
    let crs_hv = if is_crs84 {
        Some(HeaderValue::from_static(
            "<http://www.opengis.net/def/crs/OGC/0/CRS84>",
        ))
    } else if let Some(code) = epsg {
        match code {
            3857 => Some(HeaderValue::from_static(
                "<http://www.opengis.net/def/crs/EPSG/0/3857>",
            )),
            4326 => Some(HeaderValue::from_static(
                "<http://www.opengis.net/def/crs/EPSG/0/4326>",
            )),
            other => HeaderValue::from_str(&format!(
                "<http://www.opengis.net/def/crs/EPSG/0/{other}>"
            ))
            .ok(),
        }
    } else {
        None
    };
    if let Some(hv) = crs_hv {
        resp.headers_mut()
            .insert(HeaderName::from_static("content-crs"), hv);
    }
}

#[derive(Clone, Copy)]
struct TileGeoOpts {
    buffer: f32,
    tile_size: u32,
    has_explicit_ts: bool,
}

impl TileGeoOpts {
    fn from_pq(pq: &ParsedQuery) -> Self {
        let has_explicit_ts = pq
            .keys
            .iter()
            .any(|k| matches!(k.as_str(), "tile_size" | "tilesize" | "size" | "scale"));
        Self {
            buffer: pq.buffer,
            tile_size: pq.tile_size,
            has_explicit_ts,
        }
    }
}

fn attach_tile_geo_headers(
    resp: &mut Response<Full<Bytes>>,
    tms: &str,
    z: u8,
    x: u32,
    y: u32,
    scale_opt: Option<u32>,
    pq: &ParsedQuery,
) {
    attach_tile_geo_headers_opts(resp, tms, z, x, y, scale_opt, TileGeoOpts::from_pq(pq));
}

fn attach_tile_geo_headers_opts(
    resp: &mut Response<Full<Bytes>>,
    tms: &str,
    z: u8,
    x: u32,
    y: u32,
    scale_opt: Option<u32>,
    opts: TileGeoOpts,
) {
    if resp.status() != StatusCode::OK {
        return;
    }
    if tms == "WebMercatorQuad" {
        let ts = scale_opt
            .map(|sc| sc.saturating_mul(256))
            .unwrap_or(if opts.has_explicit_ts { opts.tile_size } else { 256 });
        let bbox = web_mercator_buffered_3857_bbox(z, x, y, opts.buffer, ts);
        attach_geo_headers(resp, bbox, Some(3857), false);
    } else if let Some(tms_def) = crate::tms::lookup_tms(tms)
        && let Some(m) = tms_def.matrix(z)
    {
        let base_ts = m.tile_width.clamp(64, 4096);
        let ts = scale_opt
            .map(|sc| sc.saturating_mul(base_ts))
            .unwrap_or(if opts.has_explicit_ts { opts.tile_size } else { base_ts });
        if let Some(bbox) = tms_def.tile_bounds(z, x, y, opts.buffer, ts) {
            let is_crs84 = tms == "WorldCRS84Quad";
            attach_geo_headers(resp, bbox, Some(tms_def.epsg), is_crs84);
        }
    }
}

fn to_http_response(
    engine: &TileEngine,
    res: Result<(Bytes, &'static str), EngineError>,
    if_none_match: Option<&str>,
) -> Response<Full<Bytes>> {
    match res {
        Ok((bytes, content_type)) => {
            let is_binary_tile = content_type.starts_with("image/")
                || content_type == "application/octet-stream"
                || content_type == "application/x-binary";
            let etag = is_binary_tile.then(|| crate::tile_cache::compute_tile_etag(None, &bytes));
            binary_or_plain_response(engine, bytes, content_type, etag, if_none_match)
        }
        Err(EngineError::Shared(inner)) => error_to_http_response(&inner),
        Err(e) => error_to_http_response(&e),
    }
}

fn to_http_response_cached(
    engine: &TileEngine,
    res: Result<crate::tile_cache::CachedTile, EngineError>,
    if_none_match: Option<&str>,
) -> Response<Full<Bytes>> {
    match res {
        Ok(tile) => {
            binary_or_plain_response(engine, tile.bytes, tile.content_type, tile.etag, if_none_match)
        }
        Err(EngineError::Shared(inner)) => error_to_http_response(&inner),
        Err(e) => error_to_http_response(&e),
    }
}

fn binary_or_plain_response(
    engine: &TileEngine,
    bytes: Bytes,
    content_type: &'static str,
    etag: Option<HeaderValue>,
    if_none_match: Option<&str>,
) -> Response<Full<Bytes>> {
    if let Some(etag_hv) = etag {
        if let Some(inm) = if_none_match
            && let Ok(etag_str) = etag_hv.to_str()
                && etag_matches(inm, etag_str) {
                    engine
                        .stats
                        .http_304_not_modified
                        .fetch_add(1, Ordering::Relaxed);
                    let mut resp = Response::new(Full::new(Bytes::new()));
                    *resp.status_mut() = StatusCode::NOT_MODIFIED;
                    resp.headers_mut().insert(ETAG, etag_hv);
                    resp.headers_mut()
                        .insert(CACHE_CONTROL, cache_control_header().clone());
                    resp.headers_mut()
                        .insert(ACCESS_CONTROL_ALLOW_ORIGIN, HeaderValue::from_static("*"));
                    return resp;
                }
        let mut resp = plain_resp(StatusCode::OK, bytes, content_type);
        resp.headers_mut().insert(ETAG, etag_hv);
        resp.headers_mut()
            .insert(CACHE_CONTROL, cache_control_header().clone());
        resp
    } else {
        plain_resp(StatusCode::OK, bytes, content_type)
    }
}

fn json_error_resp(status: StatusCode, msg: &str) -> Response<Full<Bytes>> {
    let body = serde_json::json!({ "detail": msg }).to_string();
    plain_resp(status, Bytes::from(body), "application/json")
}

/// Semantic failure class shared across all five route families (`/cog`, `/stac`,
/// `/mosaicjson`, `/searches`, `/zarr`).
///
/// Server-wide HTTP status rule:
/// - Client / request / malformed-source fault -> `400 Bad Request` ([`Self::ClientFault`])
/// - Origin auth / access refusal -> `403 Forbidden` ([`Self::Forbidden`])
/// - Missing resource or out-of-footprint target -> `404 Not Found` ([`Self::NotFound`])
/// - Feature not implemented by this build (unsupported codec, predictor, sample format,
///   or GCP count) -> `501 Not Implemented` ([`Self::NotImplemented`])
/// - Origin / store / decode fault -> `502 Bad Gateway` ([`Self::UpstreamFault`])
/// - Origin timeout -> `504 Gateway Timeout` ([`Self::UpstreamTimeout`])
/// - Unexpected internal error -> `500 Internal Server Error` ([`Self::Internal`])
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HttpErrorClass {
    ClientFault,
    Forbidden,
    NotFound,
    NotImplemented,
    UpstreamFault,
    UpstreamTimeout,
    Internal,
}

impl HttpErrorClass {
    pub(crate) const fn status(self) -> StatusCode {
        match self {
            Self::ClientFault => StatusCode::BAD_REQUEST,
            Self::Forbidden => StatusCode::FORBIDDEN,
            Self::NotFound => StatusCode::NOT_FOUND,
            Self::NotImplemented => StatusCode::NOT_IMPLEMENTED,
            Self::UpstreamFault => StatusCode::BAD_GATEWAY,
            Self::UpstreamTimeout => StatusCode::GATEWAY_TIMEOUT,
            Self::Internal => StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

pub(crate) fn classify_tiff_error(err: &cogzilla_tiff::TiffError) -> HttpErrorClass {
    match err {
        cogzilla_tiff::TiffError::UnsupportedCompression(_)
        | cogzilla_tiff::TiffError::UnsupportedPredictor(_)
        | cogzilla_tiff::TiffError::UnsupportedSampleFormat { .. }
        | cogzilla_tiff::TiffError::UnsupportedGcpGeoreferencing(_) => {
            HttpErrorClass::NotImplemented
        }
        _ => HttpErrorClass::ClientFault,
    }
}

pub(crate) fn classify_io_error(io_e: &cogzilla_io::IoError) -> HttpErrorClass {
    match io_e.root() {
        cogzilla_io::IoError::InvalidUri(_) => HttpErrorClass::ClientFault,
        cogzilla_io::IoError::Tiff(e) => classify_tiff_error(e),
        cogzilla_io::IoError::Codec(_) => HttpErrorClass::UpstreamFault,
        cogzilla_io::IoError::Forbidden(_)
        | cogzilla_io::IoError::HttpStatus {
            status: 401 | 403, ..
        } => HttpErrorClass::Forbidden,
        cogzilla_io::IoError::HttpStatus { status: 404, .. } => HttpErrorClass::NotFound,
        cogzilla_io::IoError::HttpStatus {
            status: 408 | 504, ..
        } => HttpErrorClass::UpstreamTimeout,
        cogzilla_io::IoError::HttpStatus { .. }
        | cogzilla_io::IoError::RangeNotSupported { .. }
        | cogzilla_io::IoError::SourceChanged { .. }
        | cogzilla_io::IoError::ResponseTooLarge { .. } => HttpErrorClass::UpstreamFault,
        cogzilla_io::IoError::Http(req_err) => {
            let mut src: Option<&(dyn std::error::Error + 'static)> =
                std::error::Error::source(req_err);
            while let Some(s) = src {
                if matches!(
                    s.downcast_ref::<cogzilla_io::IoError>(),
                    Some(cogzilla_io::IoError::Forbidden(_))
                ) {
                    return HttpErrorClass::Forbidden;
                }
                src = s.source();
            }
            if req_err.is_timeout() {
                HttpErrorClass::UpstreamTimeout
            } else {
                HttpErrorClass::UpstreamFault
            }
        }
        cogzilla_io::IoError::Fs(io_err) if io_err.kind() == std::io::ErrorKind::NotFound => {
            HttpErrorClass::NotFound
        }
        _ => HttpErrorClass::Internal,
    }
}

/// Convert an [`EngineError`] into an HTTP JSON response using [`HttpErrorClass`].
fn error_to_http_response(err: &EngineError) -> Response<Full<Bytes>> {
    match err {
        EngineError::Shared(inner) => error_to_http_response(inner),
        EngineError::NoContent => Response::builder()
            .status(StatusCode::NO_CONTENT)
            .header("access-control-allow-origin", "*")
            .body(Full::new(Bytes::new()))
            .unwrap(),
        EngineError::MethodNotAllowed(allow) => method_not_allowed_resp(allow),
        EngineError::NotFound(msg) => json_error_resp(HttpErrorClass::NotFound.status(), msg),
        EngineError::BadGateway(msg) => json_error_resp(HttpErrorClass::UpstreamFault.status(), msg),
        EngineError::BadRequest(msg) => json_error_resp(HttpErrorClass::ClientFault.status(), msg),
        EngineError::PayloadTooLarge(msg) => json_error_resp(StatusCode::PAYLOAD_TOO_LARGE, msg),
        EngineError::RequestTimeout(msg) => json_error_resp(StatusCode::REQUEST_TIMEOUT, msg),
        // The file is in a coordinate system this server cannot place. That is a
        // property of the requested source, not a server fault.
        EngineError::Proj(e) => json_error_resp(HttpErrorClass::ClientFault.status(), &e.to_string()),
        EngineError::Io(io_e) => {
            let class = classify_io_error(io_e);
            let status = class.status();
            match io_e.root() {
                cogzilla_io::IoError::HttpStatus {
                    status: code @ (401 | 403),
                    url,
                } => json_error_resp(
                    status,
                    &format!("origin refused access (HTTP {code}) fetching {url}"),
                ),
                e @ cogzilla_io::IoError::SourceChanged { .. } => {
                    let mut r = json_error_resp(status, &e.to_string());
                    r.headers_mut()
                        .insert("x-cogzilla-retry", HeaderValue::from_static("1"));
                    r
                }
                e @ cogzilla_io::IoError::Http(req_err) => {
                    let mut src: Option<&(dyn std::error::Error + 'static)> =
                        std::error::Error::source(req_err);
                    while let Some(s) = src {
                        if let Some(cogzilla_io::IoError::Forbidden(u)) =
                            s.downcast_ref::<cogzilla_io::IoError>()
                        {
                            return json_error_resp(
                                status,
                                &cogzilla_io::IoError::Forbidden(u.clone()).to_string(),
                            );
                        }
                        src = s.source();
                    }
                    if req_err.is_timeout() {
                        json_error_resp(status, &format!("upstream timeout: {e}"))
                    } else {
                        json_error_resp(status, &e.to_string())
                    }
                }
                e => {
                    if class == HttpErrorClass::Internal {
                        tracing::warn!(error = %e, "request failed with 500");
                    }
                    json_error_resp(status, &e.to_string())
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;
