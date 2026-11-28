//! Shared query-string parser and request-parameter primitives for all HTTP route families
//! (`/cog`, `/stac`, `/mosaicjson`, `/searches`, and `/zarr`).
//!
//! Having one parser and one set of primitive decoders (`parse_bool`, `parse_resample_mode`,
//! `parse_histogram_bins_spec`, `parse_histogram_range_spec`, `parse_nodata_value`,
//! `looks_like_rfc3339_datetime`) prevents parameter drift between route stacks.

use cogzilla_kernel::{ColorFormula, ColormapName, ResampleMode};

use crate::engine::EngineError;
use super::pixel_selection::Selection;

use super::{
    algorithm_params, parse_bbox4, parse_coord_crs_with_epsg, parse_ext_format,
    parse_pixel_selection, parse_u8_like, parse_u16_like, parse_u32_like, percent_decode,
    percent_decode_opts, query_key,
};

/// Alias for [`ParsedQuery`] as the shared request-parameter carrier across route families.
pub(super) type RequestParams = ParsedQuery;

/// The 5 genuinely Zarr-specific query parameters (`variable`, `group`, `sel`, `decode_times`,
/// `show_times`). All 12 shared parameters (`url`, `bidx`, `rescale`, `colormap_name`,
/// `resampling`/`reproject`, `nodata`, `unscale`, `return_mask`, `max_size`, `tilesize`,
/// `format`/`f`, `coord_crs`) are parsed once by [`parse_query`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ZarrParams {
    pub(super) variable: Option<String>,
    pub(super) group: Option<String>,
    pub(super) sel: Vec<String>,
    pub(super) decode_times: bool,
    pub(super) show_times: bool,
    /// Last-wins 0-based `bidx` (`u64`, supporting >65535 time steps such as hourly ERA5).
    pub(super) bidx: Option<u64>,
    /// Last-wins resampling mode across `resample`/`resampling`/`reproject` (preserving explicit `nearest`).
    pub(super) resample: ResampleMode,
    /// Raw parsed `tilesize`/`tile_size`/`size` (`u32`) so out-of-range values (0, >2048) still trigger 400.
    pub(super) tilesize: Option<u32>,
    /// Raw EPSG code from `coord_crs` when supplied, so unsupported codes (e.g. EPSG:9999) trigger 400.
    pub(super) coord_crs_raw_epsg: Option<u16>,
}

impl Default for ZarrParams {
    fn default() -> Self {
        Self {
            variable: None,
            group: None,
            sel: Vec::new(),
            decode_times: true,
            show_times: false,
            bidx: None,
            resample: ResampleMode::Nearest,
            tilesize: None,
            coord_crs_raw_epsg: None,
        }
    }
}

/// Per-route-family query extension paired with [`RequestParams`].
#[derive(Clone)]
pub(super) enum FamilyParams {
    Cog,
    Stac,
    MosaicJson,
    Searches,
    Zarr(ZarrParams),
}

/// Unified request parameters: shared [`RequestParams`] + per-family extension.
#[derive(Clone)]
pub(super) struct FamilyRequestParams {
    pub(super) common: RequestParams,
    pub(super) family: FamilyParams,
}

/// Parse a `/cog/*` (or root-level) query string into [`FamilyRequestParams`].
pub(super) fn parse_cog_request(raw_query: &str) -> FamilyRequestParams {
    FamilyRequestParams {
        common: parse_query((!raw_query.is_empty()).then_some(raw_query)),
        family: FamilyParams::Cog,
    }
}

/// Parse a `/stac/*` query string into [`FamilyRequestParams`].
pub(super) fn parse_stac_request(raw_query: &str) -> FamilyRequestParams {
    FamilyRequestParams {
        common: parse_query((!raw_query.is_empty()).then_some(raw_query)),
        family: FamilyParams::Stac,
    }
}

/// Parse a `/mosaicjson/*` (or `/mosaic/*`) query string into [`FamilyRequestParams`].
pub(super) fn parse_mosaic_request(raw_query: &str) -> FamilyRequestParams {
    FamilyRequestParams {
        common: parse_query((!raw_query.is_empty()).then_some(raw_query)),
        family: FamilyParams::MosaicJson,
    }
}

/// Parse a `/searches/*` (or `/collections/*`) query string into [`FamilyRequestParams`].
pub(super) fn parse_searches_request(raw_query: &str) -> FamilyRequestParams {
    FamilyRequestParams {
        common: parse_query((!raw_query.is_empty()).then_some(raw_query)),
        family: FamilyParams::Searches,
    }
}

/// Parse a request query string for the route family named by its leading path segment.
pub(super) fn parse_family_request(family_segment: Option<&str>, raw_query: &str) -> FamilyRequestParams {
    match family_segment {
        Some("stac") => parse_stac_request(raw_query),
        Some("mosaicjson" | "mosaic") => parse_mosaic_request(raw_query),
        Some("searches" | "collections") => parse_searches_request(raw_query),
        Some("zarr") => parse_zarr_request(raw_query),
        _ => parse_cog_request(raw_query),
    }
}

/// Parse a `/zarr/*` query string using [`parse_query`] for shared parameters and
/// extracting the 5 Zarr-specific fields plus Zarr's exact `bidx` (`u64`), `resample`
/// (last-wins), `tilesize` (raw `u32`), and `coord_crs` (raw EPSG) semantics.
pub(super) fn parse_zarr_request(raw_query: &str) -> FamilyRequestParams {
    let common = parse_query(Some(raw_query));
    let mut zp = ZarrParams::default();
    for pair in raw_query.split('&').filter(|s| !s.is_empty()) {
        let (k, v) = pair.split_once('=').unwrap_or((pair, ""));
        let key = query_key(k);
        match key.as_ref() {
            "variable" => zp.variable = Some(percent_decode(v)),
            "group" => zp.group = Some(percent_decode(v)),
            "sel" => zp.sel.push(percent_decode(v)),
            "decode_times" => {
                if let Some(b) = parse_bool_encoded(v) {
                    zp.decode_times = b;
                }
            }
            "show_times" => {
                if let Some(b) = parse_bool_encoded(v) {
                    zp.show_times = b;
                }
            }
            "bidx" | "indexes" => {
                for part in percent_decode(v).split(',') {
                    if let Ok(b) = part.trim().parse::<u64>()
                        && b >= 1
                    {
                        zp.bidx = Some(b - 1);
                    }
                }
            }
            "resample" | "resampling" | "reproject" => {
                let dec = percent_decode(v).to_ascii_lowercase();
                if let Some(m) = parse_resample_mode(&dec) {
                    zp.resample = m;
                }
            }
            "tilesize" | "tile_size" | "size" => {
                if let Ok(sz) = percent_decode(v).trim().parse::<u32>() {
                    zp.tilesize = Some(sz);
                }
            }
            "coord_crs" => {
                let dec = percent_decode(v);
                if let Ok((_, epsg)) = parse_coord_crs_with_epsg(&dec) {
                    zp.coord_crs_raw_epsg = Some(epsg);
                } else {
                    let trimmed = dec.trim();
                    let digits = trimmed
                        .strip_prefix("EPSG:")
                        .or_else(|| trimmed.strip_prefix("epsg:"))
                        .unwrap_or(trimmed);
                    if let Ok(code) = digits.trim().parse::<u16>() {
                        zp.coord_crs_raw_epsg = Some(code);
                    }
                }
            }
            _ => {}
        }
    }
    FamilyRequestParams {
        common,
        family: FamilyParams::Zarr(zp),
    }
}

/// Parsed histogram bin specification shared across `/cog`, `/stac`, and `/searches` statistics.
#[derive(Clone, Debug, PartialEq)]
pub(super) enum HistogramBinsSpec {
    Count(usize),
    Edges(Vec<f64>),
}

/// One asset as the query names it, in `ParsedQuery::assets`' order: its own band math (the
/// last `|expression=` of its `assets=` entry) and its own bands (the last `|bidx=` or
/// `|indexes=` of it). `listed` when it came from `assets=`, not from `asset_bidx` alone.
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct AssetEntry {
    pub(super) name: String,
    pub(super) expression: Option<String>,
    pub(super) bands: Option<Vec<u16>>,
    pub(super) listed: bool,
}

#[derive(Clone)]
pub(super) struct ParsedQuery {
    pub(super) urls: Vec<String>,
    pub(super) assets: Vec<String>,
    pub(super) asset_bidx: Vec<(String, Vec<u16>)>,
    pub(super) asset_exprs: Vec<(String, String)>,
    /// Each of `assets`, entry by entry, while it is the list as parsed.
    pub(super) asset_entries: Vec<AssetEntry>,
    /// What `asset_bidx=name|...` gives each asset, by name.
    pub(super) asset_bidx_by_name: Vec<(String, Vec<u16>)>,
    /// The assets' own band math as it was folded into `expr_raw`, for typing an array as
    /// numpy types it: each `assets` entry's band math (`None` for a plain asset), one per
    /// raster read, and the request expression over their results. `None` when nothing was
    /// folded.
    pub(super) asset_math_stages: Option<(Vec<Option<String>>, String)>,
    /// The type of the band math's results where the assets' own band math was folded into
    /// it (`expand_stac_asset_bidx`), as numpy types them stage by stage: what an algorithm
    /// over them is given. `None` when the expression is the request's own.
    pub(super) band_math_type: Option<cogzilla_raster::SampleType>,
    /// The ops the band math may compile to: more than an expression as written only where
    /// an algorithm was worked over its outputs (`algorithm_math::over_results`).
    pub(super) expr_op_budget: usize,
    pub(super) asset_as_band: bool,
    pub(super) bidx: Vec<u16>,
    pub(super) rescale: Vec<(f32, f32)>,
    pub(super) colormap: ColormapName,
    pub(super) custom_colormap_json: Option<String>,
    pub(super) color_formula: Option<String>,
    pub(super) resample: ResampleMode,
    pub(super) reproject: ResampleMode,
    pub(super) tile_size: u32,
    pub(super) max_size: u32,
    pub(super) width: Option<u32>,
    pub(super) height: Option<u32>,
    pub(super) orientation_vertical: bool,
    pub(super) pixel_selection: Selection,
    pub(super) use_epsg: bool,
    pub(super) percentiles: Vec<f64>,
    pub(super) histogram_bins: Option<usize>,
    pub(super) histogram_edges: Option<Vec<f64>>,
    pub(super) histogram_range: Option<(f64, f64)>,
    pub(super) categorical: bool,
    pub(super) categories: Vec<f64>,
    pub(super) area_weighted: bool,
    pub(super) webp_quality: Option<f32>,
    pub(super) jpeg_quality: u8,
    pub(super) expr_raw: Option<String>,
    pub(super) nodata: Option<f32>,
    pub(super) unscale: bool,
    pub(super) strict: bool,
    pub(super) return_mask: bool,
    pub(super) shape_mask: bool,
    pub(super) buffer: f32,
    pub(super) alg_buffer: Option<f32>,
    pub(super) coord_crs_3857: bool,
    pub(super) coord_crs: Option<cogzilla_proj::CrsModel>,
    pub(super) coord_crs_epsg: Option<u16>,
    pub(super) dst_crs: Option<(cogzilla_proj::CrsModel, u16)>,
    pub(super) bounds_crs: Option<(cogzilla_proj::CrsModel, u16)>,
    pub(super) bbox_crs: Option<(cogzilla_proj::CrsModel, u16)>,
    pub(super) bbox_param: Option<[f64; 4]>,
    pub(super) format_param: Option<String>,
    pub(super) minzoom: Option<u8>,
    pub(super) maxzoom: Option<u8>,
    pub(super) tile_format: Option<String>,
    pub(super) tile_scale: Option<u32>,
    pub(super) algorithm: Option<String>,
    pub(super) azimuth: f32,
    pub(super) altitude: f32,
    pub(super) z_factor: f32,
    pub(super) threshold: f32,
    pub(super) contour_increment: f32,
    pub(super) contour_thickness: u32,
    pub(super) contour_minz: f32,
    pub(super) contour_maxz: f32,
    pub(super) cast_dtype: Option<String>,
    /// `algorithm_params` as given, when it is a JSON object.
    pub(super) algorithm_params: Option<serde_json::Map<String, serde_json::Value>>,
    /// What TiTiler's model of the algorithm says of its parameters, which it answers `400`.
    pub(super) algorithm_error: Option<String>,
    /// `algorithm_params` as given when it is not JSON, which FastAPI answers `422`.
    pub(super) algorithm_params_unread: Option<String>,
    pub(super) colormap_explicit: bool,
    pub(super) colormap_name_set: bool,
    pub(super) colormap_type_linear: bool,
    /// True when `?max_size=0` was explicitly requested (full native resolution, no overview step-down).
    pub(super) max_size_zero: bool,
    /// Parameter names present in the query, for per-route validation.
    pub(super) keys: Vec<String>,
    /// Values that could not be parsed or name something unsupported.
    pub(super) errors: Vec<String>,
}

impl ParsedQuery {
    /// Drop the entry-by-entry record of the assets, for a caller that has replaced the
    /// assets' options by name: their readers then read those (`stac_entry_options`).
    pub(super) fn forget_asset_entries(&mut self) {
        self.asset_entries.clear();
        self.asset_bidx_by_name.clear();
    }
}

impl ParsedQuery {
    pub(super) fn parse_color_formula(&self) -> Result<Option<ColorFormula>, EngineError> {
        self.color_formula
            .as_deref()
            .map(ColorFormula::parse)
            .transpose()
            .map_err(EngineError::BadRequest)
    }
}

/// Shared boolean query-parameter decoder across all route stacks (exact match, no whitespace trimming).
pub(super) fn parse_bool(v: &str) -> Option<bool> {
    match v.to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" | "on" | "t" | "y" => Some(true),
        "false" | "0" | "no" | "off" | "f" | "n" => Some(false),
        _ => None,
    }
}

/// Percent-decode, trim, then parse a boolean query parameter.
pub(super) fn parse_bool_encoded(v: &str) -> Option<bool> {
    parse_bool(percent_decode(v).trim())
}

/// Parse a `nodata` query value (`nan`, `inf`, `-inf`, or a finite `f32`).
pub(super) fn parse_nodata_value(raw: &str) -> Option<f32> {
    let trimmed = raw.trim();
    if trimmed.eq_ignore_ascii_case("nan") {
        Some(f32::NAN)
    } else if trimmed.eq_ignore_ascii_case("inf") || trimmed.eq_ignore_ascii_case("+inf") {
        Some(f32::INFINITY)
    } else if trimmed.eq_ignore_ascii_case("-inf") {
        Some(f32::NEG_INFINITY)
    } else if let Ok(nd) = trimmed.parse::<f32>()
        && nd.is_finite()
    {
        Some(nd)
    } else {
        None
    }
}

/// Parse `histogram_bins` (either an integer bin count in `1..=max_count` or a comma-separated
/// edge list of length `2..=max_edges`).
pub(super) fn parse_histogram_bins_spec(
    raw: &str,
    max_count: usize,
    max_edges: usize,
    strict_monotonic: bool,
) -> Result<HistogramBinsSpec, &'static str> {
    if raw.contains(',') {
        let mut edges = Vec::new();
        for tok in raw
            .trim_matches(|c| c == '[' || c == ']')
            .split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            match tok.parse::<f64>() {
                Ok(e) if e.is_finite() => edges.push(e),
                _ => return Err("expected numeric histogram_bins edge list"),
            }
        }
        let monotonic = if strict_monotonic {
            edges.windows(2).all(|w| w[0] < w[1])
        } else {
            edges.windows(2).all(|w| w[0] <= w[1])
        };
        if edges.len() >= 2 && edges.len() <= max_edges && monotonic {
            Ok(HistogramBinsSpec::Edges(edges))
        } else {
            Err("expected strictly increasing histogram_bins edge list (2..=1001 edges)")
        }
    } else {
        match raw.trim().parse::<usize>() {
            Ok(n) if (1..=max_count).contains(&n) => Ok(HistogramBinsSpec::Count(n)),
            _ => Err("expected histogram_bins in 1..=1000 or comma-separated edge list"),
        }
    }
}

/// Parse `histogram_range=min,max` (optionally bracketed `[min,max]`).
pub(super) fn parse_histogram_range_spec(
    raw: &str,
    allow_equal: bool,
) -> Result<(f64, f64), &'static str> {
    let parts: Vec<&str> = raw
        .trim_matches(|c| c == '[' || c == ']')
        .split(',')
        .map(str::trim)
        .collect();
    if parts.len() == 2
        && let (Ok(lo), Ok(hi)) = (parts[0].parse::<f64>(), parts[1].parse::<f64>())
        && lo.is_finite()
        && hi.is_finite()
        && if allow_equal { lo <= hi } else { lo < hi }
    {
        Ok((lo, hi))
    } else {
        Err("expected histogram_range=min,max with min < max")
    }
}

/// Strict `histogram_range=min,max` parser for pgSTAC routes (rejects brackets and whitespace).
pub(super) fn parse_histogram_range_strict(
    raw: &str,
    allow_equal: bool,
) -> Result<(f64, f64), &'static str> {
    if let Some((a, b)) = raw.split_once(',')
        && let (Ok(lo), Ok(hi)) = (a.parse::<f64>(), b.parse::<f64>())
        && lo.is_finite()
        && hi.is_finite()
        && if allow_equal { lo <= hi } else { lo < hi }
    {
        Ok((lo, hi))
    } else {
        Err("expected histogram_range=min,max")
    }
}

/// Whether `s` is a date, or a date and a time, as RFC 3339 writes them.
pub(super) fn looks_like_rfc3339_datetime(s: &str) -> bool {
    let b = s.as_bytes();
    let digits = |from: usize, n: usize| {
        b.get(from..from + n)
            .is_some_and(|d| d.iter().all(u8::is_ascii_digit))
    };
    if !(digits(0, 4)
        && b.get(4) == Some(&b'-')
        && digits(5, 2)
        && b.get(7) == Some(&b'-')
        && digits(8, 2))
    {
        return false;
    }
    if b.len() == 10 {
        return true;
    }
    if !(matches!(b[10], b'T' | b't' | b' ' | b'_')
        && digits(11, 2)
        && b.get(13) == Some(&b':')
        && digits(14, 2))
    {
        return false;
    }
    let mut at = 16;
    if b.get(at) == Some(&b':') && digits(at + 1, 2) {
        at += 3;
        if b.get(at) == Some(&b'.') {
            at += 1;
            let start = at;
            while b.get(at).is_some_and(u8::is_ascii_digit) {
                at += 1;
            }
            if at == start {
                return false;
            }
        }
    }
    match &b[at..] {
        [] | [b'Z' | b'z'] => true,
        [b'+' | b'-', rest @ ..] => {
            matches!(rest, [h1, h2, b':', m1, m2] | [h1, h2, m1, m2] if [h1, h2, m1, m2].iter().all(|c| c.is_ascii_digit()))
        }
        _ => false,
    }
}

/// Parse any supported rasterio / TiTiler resampling method name (lowercase).
pub(super) fn parse_resample_mode(raw: &str) -> Option<ResampleMode> {
    match raw {
        "nearest" => Some(ResampleMode::Nearest),
        "bilinear" => Some(ResampleMode::Bilinear),
        "cubic" => Some(ResampleMode::Cubic),
        "cubic_spline" | "cubicspline" => Some(ResampleMode::CubicSpline),
        "lanczos" => Some(ResampleMode::Lanczos),
        "average" => Some(ResampleMode::Average),
        "mode" => Some(ResampleMode::Mode),
        "gauss" => Some(ResampleMode::Gauss),
        "rms" => Some(ResampleMode::Rms),
        "min" => Some(ResampleMode::Min),
        "max" => Some(ResampleMode::Max),
        "med" => Some(ResampleMode::Med),
        "q1" => Some(ResampleMode::Q1),
        "q3" => Some(ResampleMode::Q3),
        "sum" => Some(ResampleMode::Sum),
        _ => None,
    }
}

/// Strict rasterio per-parameter resampling validation (`resampling` vs `reproject`):
/// `gauss` is read-only; `max`, `min`, `med`, `q1`, `q3`, `sum` are warp-only;
/// `cubicspline` (without underscore) is not rasterio's spelling.
pub(super) fn parse_resample_mode_strict(parameter: &str, method: &str) -> Option<ResampleMode> {
    let of_the_warp = parameter == "reproject";
    match method {
        "gauss" if of_the_warp => None,
        "max" | "min" | "med" | "q1" | "q3" | "sum" if !of_the_warp => None,
        "cubicspline" => None,
        _ => parse_resample_mode(method),
    }
}

pub(super) fn parse_query(q: Option<&str>) -> ParsedQuery {
    let mut pq = ParsedQuery {
        urls: Vec::new(),
        assets: Vec::new(),
        asset_bidx: Vec::new(),
        asset_exprs: Vec::new(),
        asset_entries: Vec::new(),
        asset_bidx_by_name: Vec::new(),
        asset_math_stages: None,
        band_math_type: None,
        expr_op_budget: cogzilla_kernel::ExprProgram::MAX_OPS,
        asset_as_band: false,
        bidx: Vec::new(),
        rescale: Vec::new(),
        colormap: ColormapName::Grayscale,
        custom_colormap_json: None,
        color_formula: None,
        resample: ResampleMode::Nearest,
        reproject: ResampleMode::Nearest,
        tile_size: 256,
        max_size: 1024,
        width: None,
        height: None,
        orientation_vertical: false,
        pixel_selection: Selection::First,
        use_epsg: false,
        percentiles: Vec::new(),
        histogram_bins: None,
        histogram_edges: None,
        histogram_range: None,
        categorical: false,
        categories: Vec::new(),
        area_weighted: true,
        webp_quality: None,
        jpeg_quality: 85,
        expr_raw: None,
        nodata: None,
        unscale: false,
        strict: false,
        return_mask: true,
        shape_mask: true,
        buffer: 0.0,
        alg_buffer: None,
        coord_crs_3857: false,
        coord_crs: None,
        coord_crs_epsg: None,
        dst_crs: None,
        bounds_crs: None,
        bbox_crs: None,
        bbox_param: None,
        format_param: None,
        minzoom: None,
        maxzoom: None,
        tile_format: None,
        tile_scale: None,
        algorithm: None,
        azimuth: 315.0,
        altitude: 45.0,
        z_factor: 1.0,
        threshold: 127.0,
        contour_increment: 35.0,
        contour_thickness: 1,
        contour_minz: -12000.0,
        contour_maxz: 8000.0,
        cast_dtype: None,
        algorithm_params: None,
        algorithm_error: None,
        algorithm_params_unread: None,
        colormap_explicit: false,
        colormap_name_set: false,
        colormap_type_linear: false,
        max_size_zero: false,
        keys: Vec::new(),
        errors: Vec::new(),
    };
    let Some(q) = q else { return pq };
    let mut pq_errors: Vec<String> = Vec::new();
    for pair in q.split('&') {
        if pair.is_empty() {
            continue;
        }
        let Some((k, v)) = pair.split_once('=') else {
            continue;
        };
        let k = query_key(k);
        let k = k.as_ref();
        pq.keys.push(k.to_string());
        let mut bad = |what: &str| pq_errors.push(format!("invalid {k}={}: {what}", percent_decode(v)));
        match k {
            // Decode `+` as space per `application/x-www-form-urlencoded` so standard HTTP clients
            // (`httpx`, `requests`, `urllib.parse.urlencode`) work on paths with spaces; literal `+`
            // is sent as `%2B`.
            "url" => pq.urls.push(percent_decode(v)),
            "assets" => {
                // Preserve literal `+` in per-asset expressions (`assets=red|expression=b1+1`).
                let dec = percent_decode_opts(v, false);
                if dec.contains('|') {
                    // TiTiler `assets=name|bidx=1,2,3`, `assets=name|indexes=1,2,3`, or `assets=name|expression=b1*2`
                    let mut parts = dec.split('|');
                    let asset_name = parts.next().map(str::trim).unwrap_or("");
                    let mut parsed = Vec::new();
                    let mut entry_bands: Option<Vec<u16>> = None;
                    let mut entry_expr: Option<&str> = None;
                    for opt in parts {
                        if let Some(bidx_str) = opt
                            .strip_prefix("bidx=")
                            .or_else(|| opt.strip_prefix("indexes="))
                        {
                            let mut these = Vec::new();
                            for tok in bidx_str.split(',') {
                                match tok.trim().parse::<u16>() {
                                    Ok(b) if b >= 1 => these.push(b),
                                    _ => bad("expected positive integer band indices in |bidx="),
                                }
                            }
                            parsed.extend_from_slice(&these);
                            entry_bands = Some(these);
                        } else if let Some(expr_str) = opt.strip_prefix("expression=") {
                            let expr_str = expr_str.trim();
                            if !expr_str.is_empty() {
                                entry_expr = Some(expr_str);
                            }
                        }
                    }
                    if !asset_name.is_empty() {
                        pq.assets.push(asset_name.to_string());
                        pq.asset_entries.push(AssetEntry {
                            name: asset_name.to_string(),
                            expression: entry_expr.map(str::to_string),
                            bands: entry_bands.filter(|b| !b.is_empty()),
                            listed: true,
                        });
                    }
                    // One per `assets=` entry, the last `|expression=` of it: an asset asked
                    // for twice keeps both.
                    if let Some(expr_str) = entry_expr.filter(|_| !asset_name.is_empty()) {
                        pq.asset_exprs.push((asset_name.to_string(), expr_str.to_string()));
                    }
                    if !asset_name.is_empty() && !parsed.is_empty() {
                        if let Some((_, existing)) =
                            pq.asset_bidx.iter_mut().find(|(n, _)| n == asset_name)
                        {
                            existing.extend(parsed);
                        } else {
                            pq.asset_bidx.push((asset_name.to_string(), parsed));
                        }
                    }
                } else {
                    for a in dec.split(',') {
                        if !a.is_empty() {
                            pq.assets.push(a.to_string());
                            pq.asset_entries.push(AssetEntry { name: a.to_string(), listed: true, ..AssetEntry::default() });
                        }
                    }
                }
            }
            "asset_bidx" => {
                let dec = percent_decode(v);
                if let Some((name, bidx_str)) = dec.split_once('|') {
                    let name = name.trim();
                    let mut parsed = Vec::new();
                    for tok in bidx_str.split(',') {
                        match tok.trim().parse::<u16>() {
                            Ok(b) if b >= 1 => parsed.push(b),
                            _ => bad("expected positive integer band indices in asset_bidx"),
                        }
                    }
                    if !name.is_empty() {
                        if !pq.assets.iter().any(|a| a == name) {
                            pq.assets.push(name.to_string());
                            pq.asset_entries.push(AssetEntry { name: name.to_string(), ..AssetEntry::default() });
                        }
                        if !parsed.is_empty() {
                            match pq.asset_bidx_by_name.iter_mut().find(|(n, _)| n == name) {
                                Some((_, existing)) => existing.extend_from_slice(&parsed),
                                None => pq.asset_bidx_by_name.push((name.to_string(), parsed.clone())),
                            }
                        }
                        if !parsed.is_empty() {
                            if let Some((_, existing)) =
                                pq.asset_bidx.iter_mut().find(|(n, _)| n == name)
                            {
                                existing.extend(parsed);
                            } else {
                                pq.asset_bidx.push((name.to_string(), parsed));
                            }
                        }
                    }
                } else {
                    bad("expected asset_bidx=name|1,2,3");
                }
            }
            "asset_as_band" => match parse_bool_encoded(v) {
                Some(b) => pq.asset_as_band = b,
                None => bad("expected boolean"),
            },
            "format" | "f" => pq.format_param = Some(percent_decode(v).to_ascii_lowercase()),
            "bbox" => {
                let dec = percent_decode(v);
                if let Some((b, None)) = parse_bbox4(&dec) {
                    pq.bbox_param = Some(b);
                } else {
                    bad("expected bbox=minx,miny,maxx,maxy");
                }
            }
            "algorithm" => pq.algorithm = Some(percent_decode(v).to_ascii_lowercase()),
            "algorithm_params" => {
                let dec = percent_decode(v);
                match serde_json::from_str::<serde_json::Value>(&dec) {
                    Ok(serde_json::Value::Object(obj)) => {
                        // Read as pydantic's lax mode reads them (`"45"`, `true`); what the
                        // algorithm's model refuses is answered once the algorithm is known
                        // (`algorithm_error`), and a value refused here is not taken.
                        let number = |key: &str| algorithm_params::number(&obj, key).filter(|x| x.is_finite());
                        if let Some(a) = number("azimuth") {
                            // TiTiler's HillShade uses a 90°-rotated azimuth convention
                            // (default 45° == compass 315°); convert to compass degrees
                            // while preserving cogzilla's native `?azimuth=` compass param.
                            pq.azimuth = (a as f32 - 90.0).rem_euclid(360.0);
                        }
                        if let Some(a) = number("angle_altitude").or_else(|| number("altitude")).or_else(|| number("angle")) {
                            pq.altitude = (a as f32).clamp(-90.0, 90.0);
                        }
                        if let Some(z) = number("z_exaggeration").or_else(|| number("z_factor")).or_else(|| number("zfactor")) {
                            pq.z_factor = (z as f32).max(1e-4);
                        }
                        if let Some(t) = number("threshold") {
                            pq.threshold = t as f32;
                        }
                        if let Some(val) = obj.get("output_dtype").or_else(|| obj.get("dtype")) {
                            match val.as_str() {
                                Some(dt) => pq.cast_dtype = Some(dt.to_ascii_lowercase()),
                                None => bad("output_dtype must be a string"),
                            }
                        }
                        if let Some(x) = number("increment").filter(|x| *x > 0.0) {
                            pq.contour_increment = x as f32;
                        }
                        if let Some(t) = number("thickness").filter(|t| (0.0..=10.0).contains(t)) {
                            pq.contour_thickness = t as u32;
                        }
                        if let Some(x) = number("minz") {
                            pq.contour_minz = x as f32;
                        }
                        if let Some(b) = number("buffer").filter(|b| (0.0..=99.0).contains(b)) {
                            pq.alg_buffer = Some(b as f32);
                        }
                        if let Some(x) = number("maxz") {
                            pq.contour_maxz = x as f32;
                        }
                        pq.algorithm_params = Some(obj);
                    }
                    // Not a mapping: TiTiler fails with a 500 there, and a 400 says why here.
                    Ok(_) => pq.algorithm_error = Some("algorithm_params should be a JSON object".into()),
                    // Answered only where an algorithm is applied, as FastAPI's dependency is.
                    Err(_) => pq.algorithm_params_unread = Some(dec),
                }
            }
            "azimuth" => {
                if let Ok(a) = v.parse::<f32>()
                    && a.is_finite()
                {
                    pq.azimuth = a;
                } else {
                    bad("expected a finite number");
                }
            }
            "altitude" | "angle" | "angle_altitude" => {
                if let Ok(a) = v.parse::<f32>()
                    && a.is_finite()
                {
                    pq.altitude = a.clamp(-90.0, 90.0);
                } else {
                    bad("expected a finite number");
                }
            }
            "z_factor" | "zfactor" => {
                if let Ok(z) = v.parse::<f32>()
                    && z.is_finite()
                {
                    pq.z_factor = z.max(1e-4);
                } else {
                    bad("expected a finite number");
                }
            }
            "bidx" | "indexes" => {
                for part in percent_decode(v).split(',') {
                    match parse_u16_like(part) {
                        Some(0) => bad("band indices are 1-based (>= 1)"),
                        Some(b) => pq.bidx.push(b),
                        None => bad("expected band numbers"),
                    }
                }
            }
            "rescale" => {
                let decoded = percent_decode(v);
                for group in decoded.split(';') {
                    let cleaned = group.trim().trim_matches(|c| c == '[' || c == ']');
                    match cleaned
                        .split_once(',')
                        .map(|(a, b)| (a.trim().parse::<f32>(), b.trim().parse::<f32>()))
                    {
                        Some((Ok(mn), Ok(mx))) if mn.is_finite() && mx.is_finite() => {
                            pq.rescale.push((mn, mx));
                        }
                        _ => bad("expected finite min,max"),
                    }
                }
            }
            "colormap" => {
                let dec = percent_decode(v);
                let trimmed = dec.trim_start();
                if trimmed.starts_with('{') || trimmed.starts_with('[') {
                    pq.custom_colormap_json = Some(dec);
                } else if let Some(cm) = ColormapName::from_str_opt(&dec) {
                    pq.colormap = cm;
                    pq.colormap_explicit = true;
                } else {
                    bad("expected a JSON colormap or a colormap name");
                }
            }
            "colormap_name" => {
                if let Some(cm) = ColormapName::from_str_opt(&percent_decode(v)) {
                    pq.colormap = cm;
                    pq.colormap_explicit = true;
                    pq.colormap_name_set = true;
                } else {
                    bad("unknown colormap (see /colorMaps)");
                }
            }
            "color_formula" => {
                pq.color_formula = Some(percent_decode(v));
            }
            "nodata" => {
                let dec = percent_decode(v);
                match parse_nodata_value(&dec) {
                    Some(nd) => pq.nodata = Some(nd),
                    None => bad("expected a number, nan, inf, or -inf"),
                }
            }
            "unscale" => match parse_bool_encoded(v) {
                Some(b) => pq.unscale = b,
                None => bad("expected boolean"),
            },
            "strict" => match parse_bool_encoded(v) {
                Some(b) => pq.strict = b,
                None => bad("expected boolean"),
            },
            "return_mask" => match parse_bool_encoded(v) {
                Some(b) => pq.return_mask = b,
                None => bad("expected boolean"),
            },
            "shape_mask" => match parse_bool_encoded(v) {
                Some(b) => pq.shape_mask = b,
                None => bad("expected boolean"),
            },
            "area_weighted" => match parse_bool_encoded(v) {
                Some(b) => pq.area_weighted = b,
                None => bad("expected boolean"),
            },
            "cover_scale" | "all_touched" => {
                // Exact analytical coverage weighting needs no supersampling or binary all_touched toggle;
                // accepted and ignored for TiTiler/rasterio client compatibility.
            }
            "buffer" => {
                if let Ok(buf) = v.parse::<f32>()
                    && buf.is_finite()
                {
                    pq.buffer = buf.clamp(0.0, 256.0);
                } else {
                    bad("expected a finite number");
                }
            }
            "coord_crs" => {
                let dec = percent_decode(v);
                match parse_coord_crs_with_epsg(&dec) {
                    Ok((crs, epsg)) => {
                        pq.coord_crs_3857 = matches!(crs, cogzilla_proj::CrsModel::Epsg3857);
                        pq.coord_crs = Some(crs);
                        pq.coord_crs_epsg = Some(epsg);
                    }
                    Err(msg) => bad(&msg),
                }
            }
            "resample" | "resampling" => {
                let dec = percent_decode(v).to_ascii_lowercase();
                match parse_resample_mode(&dec) {
                    Some(m) => pq.resample = m,
                    None => {
                        bad("unsupported resampling method");
                        pq.resample = ResampleMode::Nearest;
                    }
                }
            }
            "reproject" => {
                let dec = percent_decode(v).to_ascii_lowercase();
                match parse_resample_mode(&dec) {
                    Some(m) => pq.reproject = m,
                    None => {
                        bad("unsupported reproject method");
                        pq.reproject = ResampleMode::Nearest;
                    }
                }
            }
            "tile_size" | "tilesize" | "size" => {
                match parse_u32_like(v) {
                    Some(sz) if (1..=2048).contains(&sz) => pq.tile_size = sz,
                    _ => bad("expected a tile size from 1 to 2048"),
                }
            }
            "scale" => match parse_u32_like(v) {
                Some(sc) if (1..=4).contains(&sc) => pq.tile_size = sc * 256,
                _ => bad("expected scale in 1..=4"),
            },
            "tile_scale" => match parse_u32_like(v) {
                Some(sc) if (1..=4).contains(&sc) => pq.tile_scale = Some(sc),
                _ => bad("expected tile_scale in 1..=4"),
            },
            "max_size" => {
                if let Some(ms) = parse_u32_like(v) {
                    // TiTiler treats max_size=0 as full native resolution (capped at 4096 for image routes).
                    pq.max_size_zero = ms == 0;
                    pq.max_size = if ms == 0 { 4096 } else { ms.clamp(1, 4096) };
                } else {
                    bad("expected a number");
                }
            }
            "quality" if v != "lossless" => {
                if let Ok(qual) = v.parse::<f32>()
                    && qual.is_finite()
                {
                    pq.webp_quality = Some(qual.clamp(1.0, 100.0));
                    pq.jpeg_quality = qual.round().clamp(1.0, 100.0) as u8;
                } else {
                    bad("expected a finite quality in 1..=100 or 'lossless'");
                }
            }
            "expr" | "expression" | "index" => {
                // Standard HTTP clients (httpx, requests, URLSearchParams) form-encode spaces as `+`
                // and literal plus as `%2B`, while raw CLI/browser URLs often write `(b1+b2)` directly.
                // Decode `+` as space when form-encoding markers are present so both work cleanly.
                let lower = v.to_ascii_lowercase();
                let plus_is_space = lower.contains("%2b")
                    || lower.contains("+-")
                    || lower.contains("-+")
                    || lower.contains("+*")
                    || lower.contains("*+")
                    || lower.contains("+/")
                    || lower.contains("/+")
                    || lower.contains("+^")
                    || lower.contains("^+")
                    || lower.contains("++")
                    || lower.contains("+(")
                    || lower.contains(")+")
                    || lower.contains(",+")
                    || lower.contains("+%2f")
                    || lower.contains("%2f+")
                    || lower.contains("+%2a")
                    || lower.contains("%2a+")
                    || lower.contains("+%5e")
                    || lower.contains("%5e+")
                    || lower.contains("+%28")
                    || lower.contains("%28+")
                    || lower.contains("+%29")
                    || lower.contains("%29+")
                    || lower.contains("%2c+")
                    || lower.contains("+%3c")
                    || lower.contains("%3c+")
                    || lower.contains("+%3e")
                    || lower.contains("%3e+")
                    || lower.contains("+%3d")
                    || lower.contains("%3d+")
                    || lower.contains("+%21")
                    || lower.contains("%21+")
                    || lower.contains("+%3b")
                    || lower.contains("%3b+")
                    || lower.contains("+<")
                    || lower.contains("<+")
                    || lower.contains("+>")
                    || lower.contains(">+")
                    || lower.contains("+=")
                    || lower.contains("=+")
                    || lower.contains("+!")
                    || lower.contains("+;")
                    || lower.contains(";+");
                let decoded = percent_decode_opts(v, plus_is_space);
                if !decoded.trim().is_empty() {
                    pq.expr_raw = Some(decoded);
                }
            }
            "padding" => match parse_u32_like(v) {
                Some(p) if p <= 64 => {}
                _ => bad("expected padding in 0..=64"),
            },
            "dst_crs" => {
                let dec = percent_decode(v);
                match parse_coord_crs_with_epsg(&dec) {
                    Ok(pair) => pq.dst_crs = Some(pair),
                    Err(msg) => bad(&msg),
                }
            }
            "crs" => {
                let dec = percent_decode(v);
                match parse_coord_crs_with_epsg(&dec) {
                    Ok(pair) => pq.bounds_crs = Some(pair),
                    Err(msg) => bad(&msg),
                }
            }
            "bbox-crs" | "bbox_crs" => {
                let dec = percent_decode(v);
                match parse_coord_crs_with_epsg(&dec) {
                    Ok(pair) => pq.bbox_crs = Some(pair),
                    Err(msg) => bad(&msg),
                }
            }
            "minzoom" => match parse_u8_like(v) {
                Some(z) if z <= 30 => pq.minzoom = Some(z),
                _ => bad("expected minzoom in 0..=30"),
            },
            "maxzoom" => match parse_u8_like(v) {
                Some(z) if z <= 30 => pq.maxzoom = Some(z),
                _ => bad("expected maxzoom in 0..=30"),
            },
            "tile_format" => {
                let fmt = percent_decode(v).to_ascii_lowercase();
                if parse_ext_format(&fmt, None, 85).is_ok() {
                    pq.tile_format = Some(fmt);
                } else {
                    bad("unsupported tile_format (use png, webp, jpg, f32, or npy)");
                }
            }
            "width" => match parse_u32_like(v) {
                Some(n) if (1..=4096).contains(&n) => pq.width = Some(n),
                _ => bad("expected width in 1..=4096"),
            },
            "height" => match parse_u32_like(v) {
                Some(n) if (1..=4096).contains(&n) => pq.height = Some(n),
                _ => bad("expected height in 1..=4096"),
            },
            "orientation" => match v.to_ascii_lowercase().as_str() {
                "horizontal" => pq.orientation_vertical = false,
                "vertical" => pq.orientation_vertical = true,
                _ => bad("expected horizontal or vertical"),
            },
            "use_epsg" => match parse_bool_encoded(v) {
                Some(b) => pq.use_epsg = b,
                None => bad("expected boolean"),
            },
            "p" => {
                let dec = percent_decode(v);
                for tok in dec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                    match tok.parse::<f64>() {
                        Ok(p) if p.is_finite() && (0.0..=100.0).contains(&p) => pq.percentiles.push(p),
                        _ => bad("expected percentile p in 0..=100"),
                    }
                }
            }
            "histogram_bins" => {
                let dec = percent_decode(v);
                match parse_histogram_bins_spec(&dec, 1000, 1001, true) {
                    Ok(HistogramBinsSpec::Count(n)) => pq.histogram_bins = Some(n),
                    Ok(HistogramBinsSpec::Edges(edges)) => pq.histogram_edges = Some(edges),
                    Err(msg) => bad(msg),
                }
            }
            "histogram_range" => {
                let dec = percent_decode(v);
                match parse_histogram_range_spec(&dec, false) {
                    Ok(range) => pq.histogram_range = Some(range),
                    Err(msg) => bad(msg),
                }
            }
            "categorical" => match parse_bool_encoded(v) {
                Some(b) => pq.categorical = b,
                None => bad("expected boolean"),
            },
            "c" => {
                let dec = percent_decode(v);
                for tok in dec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                    match tok.parse::<f64>() {
                        Ok(c) if c.is_finite() => pq.categories.push(c),
                        _ => bad("expected numeric category value"),
                    }
                }
            }
            "pixel_selection" => match parse_pixel_selection(v) {
                Some(ps) => pq.pixel_selection = ps,
                None => pq_errors.push(format!(
                    "unsupported pixel_selection '{v}' (expected first, highest, lowest, mean, median, stdev, count, lastbandlow, or lastbandhigh)"
                )),
            },
            "colormap_type" => {
                let dec = percent_decode(v);
                match dec.to_ascii_lowercase().as_str() {
                    "linear" => pq.colormap_type_linear = true,
                    "explicit" | "discrete" | "" => pq.colormap_type_linear = false,
                    _ => bad("expected colormap_type=explicit or linear"),
                }
            }
            _ => {}
        }
    }
    // TiTiler precedence: colormap_name wins over inline JSON colormap when both are supplied.
    if pq.colormap_name_set {
        pq.custom_colormap_json = None;
    }
    // DEM stencil algorithms (`hillshade`, `slope`, `aspect`, default buffer 3) crop
    // `algorithm_params.buffer` back out of the buffered input tile; since cogzilla's stencil
    // already fetches its own neighbour halo seamlessly, subtract that crop from `pq.buffer`
    // so `?algorithm={hillshade,slope,aspect}&buffer=3` all return 256×256 consistently (#623).
    if pq.buffer > 0.0
        && let Some(default_crop) = pq
            .algorithm
            .as_deref()
            .and_then(crate::algorithms::lookup_algorithm)
            .and_then(|def| def.dem_margin)
    {
        let crop = pq.alg_buffer.unwrap_or(default_crop).min(pq.buffer);
        pq.buffer = (pq.buffer - crop).max(0.0);
    }
    match (pq.algorithm.as_deref(), pq.algorithm_params.as_ref()) {
        (Some(alg), Some(params)) => pq.algorithm_error = algorithm_params::check(alg, params).err(),
        // Without an algorithm its parameters are not read, whatever they are.
        (None, _) => pq.algorithm_error = None,
        _ => {}
    }
    pq.errors = pq_errors;
    pq
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_25_parameters_parse_consistently() {
        // Exercises every one of the 25 shared query parameters listed in #417.
        let qs = concat!(
            "algorithm=hillshade",
            "&algorithm_params=%7B%22azimuth%22%3A180%2C%22z_exaggeration%22%3A2.0%7D",
            "&asset_as_band=true",
            "&assets=red&assets=nir",
            "&bbox=-122.5,46.5,-121.5,47.5",
            "&c=1&c=2",
            "&categorical=true",
            "&colormap=%7B%220%22%3A%5B0%2C0%2C0%2C255%5D%7D",
            "&colormap_name=viridis",
            "&coord_crs=epsg:3857",
            "&cover_scale=10",
            "&dst_crs=epsg:4326",
            "&expression=(b2-b1)/(b2%2Bb1)",
            "&f=png",
            "&format=webp",
            "&histogram_bins=16",
            "&histogram_range=0,100",
            "&max_size=512",
            "&maxzoom=14",
            "&minzoom=4",
            "&p=5&p=95",
            "&pixel_selection=median",
            "&tile_format=png",
            "&tilesize=512",
            "&use_epsg=true",
        );
        let pq = parse_query(Some(qs));
        assert!(pq.errors.is_empty(), "unexpected errors: {:?}", pq.errors);
        assert_eq!(pq.algorithm.as_deref(), Some("hillshade"));
        assert!(pq.algorithm_params.is_some());
        assert_eq!(pq.algorithm_error, None);
        assert!(pq.asset_as_band);
        assert_eq!(pq.assets, vec!["red", "nir"]);
        assert_eq!(pq.bbox_param, Some([-122.5, 46.5, -121.5, 47.5]));
        assert_eq!(pq.categories, vec![1.0, 2.0]);
        assert!(pq.categorical);
        assert_eq!(pq.colormap, ColormapName::from_str_opt("viridis").unwrap());
        // colormap_name wins over inline colormap JSON
        assert!(pq.custom_colormap_json.is_none());
        assert_eq!(pq.coord_crs_epsg, Some(3857));
        assert_eq!(pq.dst_crs.as_ref().map(|(_, e)| *e), Some(4326));
        assert_eq!(pq.expr_raw.as_deref(), Some("(b2-b1)/(b2+b1)"));
        assert_eq!(pq.format_param.as_deref(), Some("webp"));
        assert_eq!(pq.histogram_bins, Some(16));
        assert_eq!(pq.histogram_range, Some((0.0, 100.0)));
        assert_eq!(pq.max_size, 512);
        assert_eq!(pq.maxzoom, Some(14));
        assert_eq!(pq.minzoom, Some(4));
        assert_eq!(pq.percentiles, vec![5.0, 95.0]);
        assert_eq!(pq.pixel_selection, Selection::Median);
        assert_eq!(pq.tile_format.as_deref(), Some("png"));
        assert_eq!(pq.tile_size, 512);
        assert!(pq.use_epsg);
    }

    #[test]
    fn shared_primitives_match_expected_semantics() {
        for t in ["true", "TRUE", "1", "yes", "on", "t", "y"] {
            assert_eq!(parse_bool(t), Some(true), "{t}");
        }
        for f in ["false", "FALSE", "0", "no", "off", "f", "n"] {
            assert_eq!(parse_bool(f), Some(false), "{f}");
        }
        assert_eq!(parse_bool(" true "), None);
        assert_eq!(parse_bool(" 0 "), None);
        assert_eq!(parse_bool_encoded("%20true%20"), Some(true));
        assert_eq!(parse_bool_encoded("%200%20"), Some(false));
        assert_eq!(parse_bool("maybe"), None);

        assert!(parse_nodata_value("nan").unwrap().is_nan());
        assert_eq!(parse_nodata_value("inf"), Some(f32::INFINITY));
        assert_eq!(parse_nodata_value("-inf"), Some(f32::NEG_INFINITY));
        assert_eq!(parse_nodata_value("-9999"), Some(-9999.0));
        assert_eq!(parse_nodata_value("bad"), None);

        assert_eq!(
            parse_histogram_bins_spec("10", 1000, 1001, true),
            Ok(HistogramBinsSpec::Count(10))
        );
        assert_eq!(
            parse_histogram_bins_spec("0,50,100", 1000, 1001, true),
            Ok(HistogramBinsSpec::Edges(vec![0.0, 50.0, 100.0]))
        );
        assert!(parse_histogram_bins_spec("0,50,50", 1000, 1001, true).is_err());
        assert!(parse_histogram_bins_spec("0,50,50", 1000, 1001, false).is_ok());

        assert_eq!(parse_histogram_range_spec("0,100", false), Ok((0.0, 100.0)));
        assert!(parse_histogram_range_spec("5,5", false).is_err());
        assert_eq!(parse_histogram_range_spec("5,5", true), Ok((5.0, 5.0)));
        assert_eq!(parse_histogram_range_strict("0,100", true), Ok((0.0, 100.0)));
        assert!(parse_histogram_range_strict("[0,100]", true).is_err());
        assert!(parse_histogram_range_strict(" 0 , 100 ", true).is_err());

        assert!(looks_like_rfc3339_datetime("2024-06-15"));
        assert!(looks_like_rfc3339_datetime("2024-06-15T12:30:45Z"));
        assert!(looks_like_rfc3339_datetime("2024-06-15T12:30:45.123+02:00"));
        assert!(!looks_like_rfc3339_datetime("not-a-date"));
    }

    #[test]
    fn zarr_request_shares_common_parser_for_all_12_fields_and_parses_5_zarr_fields() {
        let qs = concat!(
            "url=https://example.com/era5.zarr",
            "&variable=temperature",
            "&group=surface",
            "&sel=time=2024-01-01",
            "&sel=level=500",
            "&bidx=2",
            "&decode_times=false",
            "&show_times=true",
            "&rescale=250,310",
            "&colormap_name=viridis",
            "&resampling=bilinear",
            "&nodata=-9999",
            "&unscale=true",
            "&return_mask=false",
            "&max_size=512",
            "&tilesize=512",
            "&format=webp",
            "&coord_crs=EPSG:3857",
        );
        let base = parse_query(Some(qs));
        let zarr = parse_zarr_request(qs);
        // All 12 common fields must match `parse_query` identically.
        assert_eq!(zarr.common.urls, base.urls);
        assert_eq!(zarr.common.bidx, vec![2]);
        assert_eq!(zarr.common.rescale, vec![(250.0, 310.0)]);
        assert_eq!(zarr.common.colormap, base.colormap);
        assert_eq!(zarr.common.resample, ResampleMode::Bilinear);
        assert_eq!(zarr.common.nodata, Some(-9999.0));
        assert!(zarr.common.unscale);
        assert!(!zarr.common.return_mask);
        assert_eq!(zarr.common.max_size, 512);
        assert_eq!(zarr.common.tile_size, 512);
        assert_eq!(zarr.common.format_param.as_deref(), Some("webp"));
        assert_eq!(zarr.common.coord_crs_epsg, Some(3857));
        // The 5 Zarr-specific fields.
        let FamilyParams::Zarr(zp) = zarr.family else {
            panic!("expected FamilyParams::Zarr");
        };
        assert_eq!(zp.variable.as_deref(), Some("temperature"));
        assert_eq!(zp.group.as_deref(), Some("surface"));
        assert_eq!(zp.sel, vec!["time=2024-01-01", "level=500"]);
        assert!(!zp.decode_times);
        assert!(zp.show_times);
    }

    #[test]
    fn zarr_request_preserves_error_inputs_precedence_and_u64_bidx() {
        // Out-of-range tilesize (0 / 9999) is preserved for 400 validation; non-numeric stays None.
        let FamilyParams::Zarr(z0) = parse_zarr_request("tilesize=0").family else { unreachable!() };
        assert_eq!(z0.tilesize, Some(0));
        let FamilyParams::Zarr(z9) = parse_zarr_request("tilesize=9999").family else { unreachable!() };
        assert_eq!(z9.tilesize, Some(9999));
        let FamilyParams::Zarr(zabc) = parse_zarr_request("tilesize=abc").family else { unreachable!() };
        assert_eq!(zabc.tilesize, None);

        // Unsupported coord_crs is preserved as raw EPSG code so handlers return 400 instead of 4326.
        let FamilyParams::Zarr(zcrs) = parse_zarr_request("coord_crs=EPSG:9999").family else { unreachable!() };
        assert_eq!(zcrs.coord_crs_raw_epsg, Some(9999));

        // resampling/reproject is strictly last-write-wins, including overriding back to nearest.
        let FamilyParams::Zarr(zr1) = parse_zarr_request("resampling=bilinear&reproject=cubic").family else { unreachable!() };
        assert_eq!(zr1.resample, ResampleMode::Cubic);
        let FamilyParams::Zarr(zr2) = parse_zarr_request("reproject=bilinear&resampling=nearest").family else { unreachable!() };
        assert_eq!(zr2.resample, ResampleMode::Nearest);

        // bidx supports > u16::MAX (e.g. hourly ERA5) and is last-wins; rescale is last-wins.
        let req = parse_zarr_request("bidx=1&bidx=70001&rescale=0,100&rescale=250,310");
        let FamilyParams::Zarr(zb) = req.family else { unreachable!() };
        assert_eq!(zb.bidx, Some(70_000));
        assert_eq!(req.common.rescale.last().copied(), Some((250.0, 310.0)));
    }

    #[test]
    fn all_five_families_share_identical_common_parameters() {
        let qs = concat!(
            "url=https://example.com/a.tif",
            "&bidx=2",
            "&rescale=10,250",
            "&colormap_name=viridis",
            "&resampling=bilinear",
            "&reproject=cubic",
            "&nodata=-9999",
            "&unscale=true",
            "&return_mask=false",
            "&max_size=512",
            "&tilesize=512",
            "&format=webp",
            "&coord_crs=EPSG:3857",
            "&algorithm=hillshade",
            "&pixel_selection=median",
        );
        let families = [
            ("cog", parse_cog_request(qs)),
            ("stac", parse_stac_request(qs)),
            ("mosaicjson", parse_mosaic_request(qs)),
            ("searches", parse_searches_request(qs)),
            ("zarr", parse_zarr_request(qs)),
        ];
        assert!(matches!(families[0].1.family, FamilyParams::Cog));
        assert!(matches!(families[1].1.family, FamilyParams::Stac));
        assert!(matches!(families[2].1.family, FamilyParams::MosaicJson));
        assert!(matches!(families[3].1.family, FamilyParams::Searches));
        assert!(matches!(families[4].1.family, FamilyParams::Zarr(_)));

        let base = &families[0].1.common;
        for (name, req) in &families[1..] {
            let c = &req.common;
            assert_eq!(c.urls, base.urls, "{name}: urls");
            assert_eq!(c.bidx, base.bidx, "{name}: bidx");
            assert_eq!(c.rescale, base.rescale, "{name}: rescale");
            assert_eq!(c.colormap, base.colormap, "{name}: colormap");
            assert_eq!(c.resample, base.resample, "{name}: resample");
            assert_eq!(c.reproject, base.reproject, "{name}: reproject");
            assert_eq!(c.nodata, base.nodata, "{name}: nodata");
            assert_eq!(c.unscale, base.unscale, "{name}: unscale");
            assert_eq!(c.return_mask, base.return_mask, "{name}: return_mask");
            assert_eq!(c.max_size, base.max_size, "{name}: max_size");
            assert_eq!(c.tile_size, base.tile_size, "{name}: tile_size");
            assert_eq!(c.format_param, base.format_param, "{name}: format_param");
            assert_eq!(c.coord_crs_epsg, base.coord_crs_epsg, "{name}: coord_crs_epsg");
            assert_eq!(c.algorithm, base.algorithm, "{name}: algorithm");
            assert_eq!(c.pixel_selection, base.pixel_selection, "{name}: pixel_selection");
            assert_eq!(c.keys, base.keys, "{name}: keys");
            assert_eq!(c.errors, base.errors, "{name}: errors");
        }
    }

    #[test]
    fn cast_algorithm_params_accepts_output_dtype_and_dtype_alias() {
        let pq_tt = parse_query(Some("algorithm=cast&algorithm_params=%7B%22output_dtype%22%3A%22int16%22%7D"));
        assert_eq!(pq_tt.cast_dtype.as_deref(), Some("int16"));
        assert!(pq_tt.errors.is_empty());

        let pq_alias = parse_query(Some("algorithm=cast&algorithm_params=%7B%22dtype%22%3A%22UINT16%22%7D"));
        assert_eq!(pq_alias.cast_dtype.as_deref(), Some("uint16"));
        assert!(pq_alias.errors.is_empty());

        // TiTiler's canonical `output_dtype` takes precedence if both are present
        let pq_both = parse_query(Some(
            "algorithm=cast&algorithm_params=%7B%22output_dtype%22%3A%22int16%22%2C%22dtype%22%3A%22uint8%22%7D",
        ));
        assert_eq!(pq_both.cast_dtype.as_deref(), Some("int16"));
    }

    #[test]
    fn terrain_stencil_algorithms_share_consistent_buffer_crop() {
        for alg in ["hillshade", "slope", "aspect"] {
            let pq3 = parse_query(Some(&format!("algorithm={alg}&buffer=3")));
            assert_eq!(pq3.buffer, 0.0, "{alg} with buffer=3 should crop to 0.0 (256x256)");
            let pq6 = parse_query(Some(&format!("algorithm={alg}&buffer=6")));
            assert_eq!(pq6.buffer, 3.0, "{alg} with buffer=6 should crop default 3 to 3.0");
            let pq_zero_crop = parse_query(Some(&format!(
                "algorithm={alg}&buffer=3&algorithm_params=%7B%22buffer%22%3A0%7D"
            )));
            assert_eq!(
                pq_zero_crop.buffer, 3.0,
                "{alg} with algorithm_params.buffer=0 should preserve full buffer=3"
            );
        }
    }
}
