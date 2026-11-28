use std::sync::OnceLock;

use bytes::Bytes;
use serde_json::{Map, Value, json};

fn path_param(name: &str, ty: &str, desc: &str) -> Value {
    json!({
        "name": name,
        "in": "path",
        "required": true,
        "description": desc,
        "schema": { "type": ty }
    })
}

fn q_str(name: &str, required: bool, desc: &str) -> Value {
    json!({
        "name": name,
        "in": "query",
        "required": required,
        "description": desc,
        "schema": { "type": "string" }
    })
}

fn q_int(name: &str, desc: &str) -> Value {
    json!({
        "name": name,
        "in": "query",
        "required": false,
        "description": desc,
        "schema": { "type": "integer" }
    })
}

fn q_num(name: &str, desc: &str) -> Value {
    json!({
        "name": name,
        "in": "query",
        "required": false,
        "description": desc,
        "schema": { "type": "number" }
    })
}

fn q_bool(name: &str, required: bool, desc: &str) -> Value {
    json!({
        "name": name,
        "in": "query",
        "required": required,
        "description": desc,
        "schema": { "type": "boolean" }
    })
}

fn q_arr(name: &str, item_ty: &str, required: bool, desc: &str) -> Value {
    json!({
        "name": name,
        "in": "query",
        "required": required,
        "description": desc,
        "schema": { "type": "array", "items": { "type": item_ty } }
    })
}

fn std_responses(content_type: &str) -> Value {
    json!({
        "200": {
            "description": "Successful Response",
            "content": { content_type: {} }
        },
        "400": { "description": "Bad Request" },
        "404": { "description": "Not Found" },
        "422": {
            "description": "Validation Error",
            "content": {
                "application/json": {
                    "schema": { "$ref": "#/components/schemas/HTTPValidationError" }
                }
            }
        }
    })
}

fn op(tag: &str, summary: &str, params: Vec<Value>, ct: &str) -> Value {
    json!({
        "tags": [tag],
        "summary": summary,
        "parameters": params,
        "responses": std_responses(ct)
    })
}

fn post_op(tag: &str, summary: &str, params: Vec<Value>, body_schema_ref: &str, ct: &str) -> Value {
    json!({
        "tags": [tag],
        "summary": summary,
        "parameters": params,
        "requestBody": {
            "required": true,
            "content": {
                "application/json": {
                    "schema": { "$ref": body_schema_ref }
                }
            }
        },
        "responses": std_responses(ct)
    })
}

fn common_band_render_params() -> Vec<Value> {
    vec![
        q_arr("bidx", "integer", false, "1-based band indexes"),
        q_str("expression", false, "Band math expression (e.g. (b1-b2)/(b1+b2))"),
        q_str("nodata", false, "Override internal nodata value (number or 'nan')"),
        q_bool("unscale", false, "Apply dataset internal scale/offset"),
        q_str("resampling", false, "Resampling method (nearest, bilinear, cubic, lanczos)"),
        q_str("reproject", false, "Warp resampling method"),
        q_arr("rescale", "string", false, "Comma-separated min,max rescale range(s)"),
        q_str("color_formula", false, "rio-color formula (e.g. Gamma RGB 1.5)"),
        q_str("colormap_name", false, "Named rio-tiler colormap"),
        q_str("colormap", false, "JSON-encoded custom discrete or interval colormap"),
        q_str("colormap_type", false, "Colormap type (explicit or linear)"),
        q_bool("return_mask", false, "Include alpha/validity mask in output"),
        q_str("algorithm", false, "Raster algorithm (hillshade, slope, aspect, contours, terrarium, terrainrgb, normalizedIndex)"),
        q_str("algorithm_params", false, "JSON-encoded algorithm parameters"),
    ]
}

fn stac_asset_params() -> Vec<Value> {
    vec![
        q_arr("assets", "string", false, "STAC asset names (optionally with |bidx= or |expression=)"),
        q_bool("asset_as_band", false, "Treat each asset as a single band in expressions"),
        q_arr("asset_bidx", "string", false, "Per-asset band index (asset|bidx)"),
        q_arr("asset_expression", "string", false, "Per-asset band expression (asset|expr)"),
    ]
}

fn stats_params() -> Vec<Value> {
    vec![
        q_bool("categorical", false, "Return categorical pixel counts"),
        q_arr("c", "number", false, "Category values to count"),
        q_arr("p", "number", false, "Percentiles to compute (0..100)"),
        q_str("histogram_bins", false, "Histogram bin count or comma-separated edges"),
        q_str("histogram_range", false, "Comma-separated min,max histogram range"),
        q_int("max_size", "Maximum image dimension for statistics sampling"),
        q_int("width", "Explicit output width"),
        q_int("height", "Explicit output height"),
        q_str("dst_crs", false, "Target CRS for statistics resampling"),
    ]
}

fn tms_tilejson_params() -> Vec<Value> {
    vec![
        q_int("minzoom", "Override minimum zoom level"),
        q_int("maxzoom", "Override maximum zoom level"),
        q_str("tile_format", false, "Tile image format (png, jpg, webp, tif, npy)"),
        q_int("tile_scale", "Tile scale factor (1..4)"),
        q_int("tilesize", "Tile pixel dimension"),
        q_num("buffer", "Tile buffer in pixels"),
        q_int("padding", "Source read padding in pixels"),
    ]
}

fn wmts_params() -> Vec<Value> {
    vec![
        q_int("minzoom", "Override minimum zoom level"),
        q_int("maxzoom", "Override maximum zoom level"),
        q_arr("zooms", "string", false, "Per-TMS zoom overrides"),
        q_str("tile_format", false, "Tile image format (png, jpg, webp, tif)"),
        q_int("tile_scale", "Tile scale factor (1..4)"),
        q_num("buffer", "Tile buffer in pixels"),
        q_int("padding", "Source read padding in pixels"),
        q_bool("use_epsg", false, "Emit EPSG:<code> instead of OGC URI"),
    ]
}

fn map_params() -> Vec<Value> {
    vec![
        q_str("bbox", false, "Bounding box minx,miny,maxx,maxy"),
        q_str("bbox-crs", false, "CRS of bbox parameter"),
        q_str("crs", false, "Output map CRS"),
        q_int("width", "Output image width in pixels"),
        q_int("height", "Output image height in pixels"),
        q_int("max_size", "Maximum output dimension"),
        q_str("f", false, "Output image format (png, jpeg, webp, tif, npy)"),
    ]
}

#[derive(Clone, Copy)]
struct FamilyCaps {
    assets: bool,
    mosaic: bool,
    viewer: bool,
    ogc_maps: bool,
}

fn add_family_routes(
    paths: &mut Map<String, Value>,
    prefix: &str,
    tag: &str,
    url_desc: &str,
    caps: FamilyCaps,
) {
    let url_p = q_str("url", true, url_desc);
    let extra_family = || -> Vec<Value> {
        let mut v = vec![url_p.clone()];
        if caps.assets {
            v.extend(stac_asset_params());
        }
        if caps.mosaic {
            v.push(q_str("pixel_selection", false, "Mosaic pixel selection method (first, highest, lowest, mean, median, stdev, lastbandlow, lastbandhigh, count)"));
        }
        v
    };

    // Info / Bounds / Viewer
    {
        let mut p = extra_family();
        p.push(q_str("crs", false, "Target CRS for bounds"));
        paths.insert(
            format!("{prefix}/info"),
            json!({ "get": op(tag, &format!("{tag} metadata"), extra_family(), "application/json") }),
        );
        paths.insert(
            format!("{prefix}/info.geojson"),
            json!({ "get": op(tag, &format!("{tag} GeoJSON metadata"), p, "application/geo+json") }),
        );
        let mut bp = vec![url_p.clone()];
        bp.push(q_str("crs", false, "Target CRS for bounds"));
        paths.insert(
            format!("{prefix}/bounds"),
            json!({ "get": op(tag, &format!("{tag} WGS84 bounds"), bp, "application/json") }),
        );
        if caps.viewer {
            paths.insert(
                format!("{prefix}/viewer"),
                json!({ "get": op(tag, &format!("{tag} interactive viewer"), vec![], "text/html") }),
            );
        }
    }

    // OGC Tilesets
    {
        let mut tp = extra_family();
        tp.push(q_str("crs", false, "Filter or target CRS"));
        tp.push(q_str("f", false, "Response format (json or html)"));
        paths.insert(
            format!("{prefix}/tiles"),
            json!({ "get": op(tag, &format!("List {tag} tilesets"), tp.clone(), "application/json") }),
        );
        let mut t1 = vec![path_param("tileMatrixSetId", "string", "TileMatrixSet identifier")];
        t1.extend(tp);
        t1.push(q_int("minzoom", "Override minimum zoom"));
        t1.push(q_int("maxzoom", "Override maximum zoom"));
        paths.insert(
            format!("{prefix}/tiles/{{tileMatrixSetId}}"),
            json!({ "get": op(tag, &format!("{tag} tileset metadata"), t1, "application/json") }),
        );
    }

    // Tile endpoints (4 TiTiler variants: bare, .{format}, @{scale}x, @{scale}x.{format})
    let tile_query = || -> Vec<Value> {
        let mut v = extra_family();
        v.extend(common_band_render_params());
        v.push(q_int("tilesize", "Tile pixel dimension"));
        v.push(q_num("buffer", "Tile buffer in pixels"));
        v.push(q_int("padding", "Source read padding in pixels"));
        v
    };
    let tile_path_base = || -> Vec<Value> {
        vec![
            path_param("tileMatrixSetId", "string", "TileMatrixSet identifier"),
            path_param("z", "integer", "Tile zoom level"),
            path_param("x", "integer", "Tile column"),
            path_param("y", "integer", "Tile row"),
        ]
    };
    {
        let mut p = tile_path_base();
        p.extend(tile_query());
        p.push(q_str("format", false, "Output image format override (png, jpg, webp, tif, npy)"));
        paths.insert(
            format!("{prefix}/tiles/{{tileMatrixSetId}}/{{z}}/{{x}}/{{y}}"),
            json!({ "get": op(tag, &format!("Render {tag} tile (auto format)"), p, "image/png") }),
        );
    }
    {
        let mut p = tile_path_base();
        p.push(path_param("format", "string", "Output image format (png, jpg, webp, tif, npy)"));
        p.extend(tile_query());
        paths.insert(
            format!("{prefix}/tiles/{{tileMatrixSetId}}/{{z}}/{{x}}/{{y}}.{{format}}"),
            json!({ "get": op(tag, &format!("Render {tag} tile"), p, "image/png") }),
        );
    }
    {
        let mut p = tile_path_base();
        p.push(path_param("scale", "integer", "Tile scale factor (1..4)"));
        p.extend(tile_query());
        p.push(q_str("format", false, "Output image format override (png, jpg, webp, tif, npy)"));
        paths.insert(
            format!("{prefix}/tiles/{{tileMatrixSetId}}/{{z}}/{{x}}/{{y}}@{{scale}}x"),
            json!({ "get": op(tag, &format!("Render scaled {tag} tile (auto format)"), p, "image/png") }),
        );
    }
    {
        let mut p = tile_path_base();
        p.push(path_param("scale", "integer", "Tile scale factor (1..4)"));
        p.push(path_param("format", "string", "Output image format (png, jpg, webp, tif, npy)"));
        p.extend(tile_query());
        paths.insert(
            format!("{prefix}/tiles/{{tileMatrixSetId}}/{{z}}/{{x}}/{{y}}@{{scale}}x.{{format}}"),
            json!({ "get": op(tag, &format!("Render scaled {tag} tile"), p, "image/png") }),
        );
    }

    // TileJSON, map.html, WMTSCapabilities.xml (both /{tms}/... and root)
    let tj_query = || -> Vec<Value> {
        let mut v = extra_family();
        v.extend(common_band_render_params());
        v.extend(tms_tilejson_params());
        v
    };
    {
        let mut p = vec![path_param("tileMatrixSetId", "string", "TileMatrixSet identifier")];
        p.extend(tj_query());
        paths.insert(
            format!("{prefix}/{{tileMatrixSetId}}/tilejson.json"),
            json!({ "get": op(tag, &format!("{tag} TileJSON 3.0"), p, "application/json") }),
        );
        paths.insert(
            format!("{prefix}/tilejson.json"),
            json!({ "get": op(tag, &format!("{tag} TileJSON 3.0 (WebMercatorQuad)"), tj_query(), "application/json") }),
        );
    }
    {
        let mut p = vec![path_param("tileMatrixSetId", "string", "TileMatrixSet identifier")];
        p.extend(tj_query());
        paths.insert(
            format!("{prefix}/{{tileMatrixSetId}}/map.html"),
            json!({ "get": op(tag, &format!("{tag} interactive map viewer"), p, "text/html") }),
        );
        paths.insert(
            format!("{prefix}/map.html"),
            json!({ "get": op(tag, &format!("{tag} interactive map viewer (WebMercatorQuad)"), tj_query(), "text/html") }),
        );
    }
    {
        let mut wq = extra_family();
        wq.extend(common_band_render_params());
        wq.extend(wmts_params());
        paths.insert(
            format!("{prefix}/WMTSCapabilities.xml"),
            json!({ "get": op(tag, &format!("{tag} OGC WMTS Capabilities (all TMS)"), wq.clone(), "application/xml") }),
        );
        let mut wp = vec![path_param("tileMatrixSetId", "string", "TileMatrixSet identifier")];
        wp.extend(wq);
        paths.insert(
            format!("{prefix}/{{tileMatrixSetId}}/WMTSCapabilities.xml"),
            json!({ "get": op(tag, &format!("{tag} OGC WMTS Capabilities"), wp, "application/xml") }),
        );
        if prefix == "/cog" {
            let mut wms_q = extra_family();
            wms_q.extend(common_band_render_params());
            wms_q.push(q_str("REQUEST", true, "WMS request type (GetCapabilities or GetMap)"));
            wms_q.push(q_str("VERSION", false, "WMS version (1.3.0 or 1.1.1)"));
            wms_q.push(q_str("CRS", false, "CRS for GetMap (e.g. EPSG:3857, CRS:84, EPSG:4326)"));
            wms_q.push(q_str("BBOX", false, "Bounding box minx,miny,maxx,maxy"));
            wms_q.push(q_int("WIDTH", "Output image width (1..4096)"));
            wms_q.push(q_int("HEIGHT", "Output image height (1..4096)"));
            wms_q.push(q_str("FORMAT", false, "Output format (image/png, image/jpeg, image/webp, image/tiff)"));
            paths.insert(
                "/cog/wms".to_string(),
                json!({ "get": op(tag, "COG OGC WMS 1.1.1 / 1.3.0 (GetCapabilities & GetMap)", wms_q, "image/png") }),
            );
        }
    }

    // Point
    {
        let mut p = vec![
            path_param("lon", "number", "Longitude or X coordinate"),
            path_param("lat", "number", "Latitude or Y coordinate"),
        ];
        p.extend(extra_family());
        p.push(q_arr("bidx", "integer", false, "1-based band indexes"));
        p.push(q_str("expression", false, "Band math expression"));
        p.push(q_str("nodata", false, "Override nodata value"));
        p.push(q_bool("unscale", false, "Apply dataset scale/offset"));
        p.push(q_str("resampling", false, "Resampling method"));
        p.push(q_str("reproject", false, "Warp resampling method"));
        p.push(q_str("coord_crs", false, "CRS of lon,lat coordinates (default EPSG:4326)"));
        paths.insert(
            format!("{prefix}/point/{{lon}},{{lat}}"),
            json!({ "get": op(tag, &format!("Sample {tag} pixel values at point"), p, "application/json") }),
        );
    }

    // BBox crops
    let bbox_query = || -> Vec<Value> {
        let mut v = extra_family();
        v.extend(common_band_render_params());
        v.push(q_str("coord_crs", false, "CRS of bbox coordinates (default EPSG:4326)"));
        v.push(q_str("dst_crs", false, "Output image CRS"));
        v.push(q_int("max_size", "Maximum output dimension"));
        v.push(q_int("width", "Explicit output width"));
        v.push(q_int("height", "Explicit output height"));
        v
    };
    let bbox_path = || -> Vec<Value> {
        vec![
            path_param("minx", "number", "Minimum X"),
            path_param("miny", "number", "Minimum Y"),
            path_param("maxx", "number", "Maximum X"),
            path_param("maxy", "number", "Maximum Y"),
        ]
    };
    {
        let mut p = bbox_path();
        p.push(path_param("format", "string", "Output image format"));
        p.extend(bbox_query());
        paths.insert(
            format!("{prefix}/bbox/{{minx}},{{miny}},{{maxx}},{{maxy}}.{{format}}"),
            json!({ "get": op(tag, &format!("Render {tag} bounding-box crop"), p, "image/png") }),
        );
    }
    {
        let mut p = bbox_path();
        p.push(path_param("width", "integer", "Output width"));
        p.push(path_param("height", "integer", "Output height"));
        p.push(path_param("format", "string", "Output image format"));
        p.extend(bbox_query());
        paths.insert(
            format!("{prefix}/bbox/{{minx}},{{miny}},{{maxx}},{{maxy}}/{{width}}x{{height}}.{{format}}"),
            json!({ "get": op(tag, &format!("Render sized {tag} bounding-box crop"), p, "image/png") }),
        );
    }

    // Feature crops (POST)
    {
        let fq = || -> Vec<Value> {
            let mut v = extra_family();
            v.extend(common_band_render_params());
            v.push(q_str("coord_crs", false, "CRS of incoming GeoJSON geometry"));
            v.push(q_str("dst_crs", false, "Output image CRS"));
            v.push(q_int("max_size", "Maximum output dimension"));
            v.push(q_int("width", "Explicit output width"));
            v.push(q_int("height", "Explicit output height"));
            v
        };
        let mut f_auto = fq();
        f_auto.push(q_str("format", false, "Output image format override"));
        paths.insert(
            format!("{prefix}/feature"),
            json!({ "post": post_op(tag, &format!("Render {tag} GeoJSON feature crop (auto format)"), f_auto, "#/components/schemas/GeoJSONFeature", "image/png") }),
        );
        let mut pf = vec![path_param("format", "string", "Output image format")];
        pf.extend(fq());
        paths.insert(
            format!("{prefix}/feature.{{format}}"),
            json!({ "post": post_op(tag, &format!("Render {tag} GeoJSON feature crop"), pf, "#/components/schemas/GeoJSONFeature", "image/png") }),
        );
        let mut pwh = vec![
            path_param("width", "integer", "Output width"),
            path_param("height", "integer", "Output height"),
            path_param("format", "string", "Output image format"),
        ];
        pwh.extend(fq());
        paths.insert(
            format!("{prefix}/feature/{{width}}x{{height}}.{{format}}"),
            json!({ "post": post_op(tag, &format!("Render sized {tag} GeoJSON feature crop"), pwh, "#/components/schemas/GeoJSONFeature", "image/png") }),
        );
    }

    // Preview + OGC Maps + Statistics (COG, STAC & Zarr)
    if !caps.mosaic {
        let prev_q = || -> Vec<Value> {
            let mut v = extra_family();
            v.extend(common_band_render_params());
            v.push(q_int("max_size", "Maximum preview dimension"));
            v.push(q_int("width", "Explicit preview width"));
            v.push(q_int("height", "Explicit preview height"));
            v.push(q_str("dst_crs", false, "Target CRS for preview"));
            v
        };
        let mut p_auto = prev_q();
        p_auto.push(q_str("format", false, "Output image format override"));
        paths.insert(
            format!("{prefix}/preview"),
            json!({ "get": op(tag, &format!("Render {tag} preview (auto format)"), p_auto, "image/png") }),
        );
        let mut pf = vec![path_param("format", "string", "Output image format")];
        pf.extend(prev_q());
        paths.insert(
            format!("{prefix}/preview.{{format}}"),
            json!({ "get": op(tag, &format!("Render {tag} preview"), pf, "image/png") }),
        );
        let mut pwh = vec![
            path_param("width", "integer", "Preview width"),
            path_param("height", "integer", "Preview height"),
            path_param("format", "string", "Output image format"),
        ];
        pwh.extend(prev_q());
        paths.insert(
            format!("{prefix}/preview/{{width}}x{{height}}.{{format}}"),
            json!({ "get": op(tag, &format!("Render sized {tag} preview"), pwh, "image/png") }),
        );

        if caps.ogc_maps {
            let mut mp = extra_family();
            mp.extend(common_band_render_params());
            mp.extend(map_params());
            paths.insert(
                format!("{prefix}/map"),
                json!({ "get": op(tag, &format!("{tag} OGC API Maps"), mp, "image/png") }),
            );
        }

        let mut sp = extra_family();
        sp.extend(common_band_render_params());
        sp.extend(stats_params());
        let mut post_sp = sp.clone();
        post_sp.push(q_str("coord_crs", false, "CRS of incoming GeoJSON geometry"));
        post_sp.push(q_bool("area_weighted", false, "Weight statistics by exact WGS84 ellipsoidal pixel ground area (default true; pass area_weighted=false for unweighted CRS-plane TiTiler parity)"));
        post_sp.push(q_int("cover_scale", "Accepted for TiTiler compatibility (no-op; exact analytical coverage needs no supersampling)"));
        post_sp.push(q_bool("all_touched", false, "Accepted for TiTiler compatibility (no-op; exact analytical sub-pixel coverage already includes every touched pixel without binary overcounting)"));
        paths.insert(
            format!("{prefix}/statistics"),
            json!({
                "get": op(tag, &format!("{tag} raster statistics"), sp, "application/json"),
                "post": post_op(tag, &format!("Polygon-clipped {tag} statistics"), post_sp, "#/components/schemas/GeoJSONFeature", "application/geo+json")
            }),
        );
    }
}

pub fn build_openapi_spec() -> Value {
    let mut paths = Map::new();

    // Core / Meta / Discovery
    paths.insert(
        "/".into(),
        json!({ "get": op("Meta", "Landing page", vec![q_str("f", false, "Response format (json or html)")], "application/json") }),
    );
    paths.insert(
        "/conformance".into(),
        json!({ "get": op("Meta", "OGC API conformance classes", vec![q_str("f", false, "Response format (json or html)")], "application/json") }),
    );
    paths.insert(
        "/healthz".into(),
        json!({ "get": op("Meta", "Health check", vec![], "application/json") }),
    );
    paths.insert(
        "/api".into(),
        json!({ "get": op("Meta", "OpenAPI 3.1 specification", vec![], "application/vnd.oai.openapi+json") }),
    );
    paths.insert(
        "/openapi.json".into(),
        json!({ "get": op("Meta", "OpenAPI 3.1 specification", vec![], "application/json") }),
    );
    paths.insert(
        "/api.html".into(),
        json!({ "get": op("Meta", "Swagger UI documentation", vec![], "text/html") }),
    );
    paths.insert(
        "/docs".into(),
        json!({ "get": op("Meta", "Swagger UI documentation", vec![], "text/html") }),
    );

    // Algorithms, ColorMaps, TileMatrixSets
    paths.insert(
        "/algorithms".into(),
        json!({ "get": op("Algorithms", "List available raster algorithms", vec![q_str("f", false, "Response format (json or html)")], "application/json") }),
    );
    paths.insert(
        "/algorithms/{algorithmId}".into(),
        json!({ "get": op("Algorithms", "Get algorithm metadata and parameters", vec![
            path_param("algorithmId", "string", "Algorithm identifier"),
            q_str("f", false, "Response format (json or html)"),
        ], "application/json") }),
    );
    paths.insert(
        "/colorMaps".into(),
        json!({ "get": op("ColorMaps", "List all 211 rio-tiler colormaps", vec![q_str("f", false, "Response format (json or html)")], "application/json") }),
    );
    paths.insert(
        "/colorMaps/{colorMapId}".into(),
        json!({ "get": op("ColorMaps", "Get colormap RGBA LUT or colorbar image", vec![
            path_param("colorMapId", "string", "Colormap identifier"),
            q_str("f", false, "Response format (json, png, jpg, webp)"),
            q_str("format", false, "Alias for f"),
            q_str("orientation", false, "Colorbar orientation (horizontal or vertical)"),
            q_int("width", "Colorbar width"),
            q_int("height", "Colorbar height"),
        ], "application/json") }),
    );
    paths.insert(
        "/tileMatrixSets".into(),
        json!({ "get": op("OGC Tiles", "List supported OGC TileMatrixSets", vec![q_str("f", false, "Response format (json or html)")], "application/json") }),
    );
    paths.insert(
        "/tileMatrixSets/{tileMatrixSetId}".into(),
        json!({ "get": op("OGC Tiles", "Get OGC TileMatrixSet 2.0 definition", vec![
            path_param("tileMatrixSetId", "string", "TileMatrixSet identifier"),
            q_str("f", false, "Response format (json or html)"),
        ], "application/json") }),
    );

    // COG family + COG-specific routes
    add_family_routes(
        &mut paths,
        "/cog",
        "Cloud Optimized GeoTIFF",
        "Dataset URL",
        FamilyCaps {
            assets: false,
            mosaic: false,
            viewer: true,
            ogc_maps: true,
        },
    );
    paths.insert(
        "/cog/validate".into(),
        json!({ "get": op("Cloud Optimized GeoTIFF", "Validate COG structure and layout", vec![
            q_str("url", true, "Dataset URL"),
            q_bool("strict", false, "Treat warnings as validation errors"),
        ], "application/json") }),
    );
    paths.insert(
        "/cog/stac".into(),
        json!({ "get": op("Cloud Optimized GeoTIFF", "Generate STAC 1.1.0 Item for COG", vec![
            q_str("url", true, "Dataset URL"),
            q_str("id", false, "STAC Item ID"),
            q_str("collection", false, "STAC Collection ID"),
            q_str("collection_url", false, "STAC Collection link URL"),
            q_str("datetime", false, "RFC3339 datetime"),
            q_str("asset_name", false, "Asset key name (default 'data')"),
            q_str("asset_href", false, "Override asset href"),
            q_str("asset_media_type", false, "Asset media type or 'auto'"),
            q_arr("asset_roles", "string", false, "Asset roles"),
            q_bool("with_proj", false, "Include proj extension properties"),
            q_bool("with_raster", false, "Include raster extension band stats"),
            q_bool("with_eo", false, "Include eo extension properties"),
            q_arr("extensions", "string", false, "Additional STAC extension URIs"),
            q_int("geometry_densify", "Boundary densification points per edge"),
            q_int("geometry_precision", "Coordinate decimal precision"),
            q_int("max_size", "Maximum size for raster statistics"),
        ], "application/geo+json") }),
    );

    // STAC family + STAC-specific routes
    add_family_routes(
        &mut paths,
        "/stac",
        "SpatioTemporal Asset Catalog",
        "STAC Item URL",
        FamilyCaps {
            assets: true,
            mosaic: false,
            viewer: true,
            ogc_maps: true,
        },
    );
    paths.insert(
        "/stac/assets".into(),
        json!({ "get": op("SpatioTemporal Asset Catalog", "List STAC Item raster assets", vec![q_str("url", true, "STAC Item URL")], "application/json") }),
    );
    {
        let mut p = vec![q_str("url", true, "STAC Item URL")];
        p.extend(stac_asset_params());
        p.extend(common_band_render_params());
        p.extend(stats_params());
        paths.insert(
            "/stac/asset_statistics".into(),
            json!({ "get": op("SpatioTemporal Asset Catalog", "Per-asset STAC statistics", p, "application/json") }),
        );
    }
    paths.insert(
        "/stac/renders".into(),
        json!({ "get": op("SpatioTemporal Asset Catalog", "List STAC Item render definitions", vec![q_str("url", true, "STAC Item URL")], "application/json") }),
    );
    paths.insert(
        "/stac/renders/{render_id}".into(),
        json!({ "get": op("SpatioTemporal Asset Catalog", "Get a STAC Item render definition", vec![
            path_param("render_id", "string", "Render identifier"),
            q_str("url", true, "STAC Item URL"),
        ], "application/json") }),
    );

    // MosaicJSON family + MosaicJSON-specific routes
    add_family_routes(
        &mut paths,
        "/mosaicjson",
        "MosaicJSON",
        "MosaicJSON document URL",
        FamilyCaps {
            assets: false,
            mosaic: true,
            viewer: true,
            ogc_maps: false,
        },
    );
    paths.insert(
        "/mosaicjson/".into(),
        json!({ "get": op("MosaicJSON", "Read MosaicJSON document", vec![q_str("url", true, "MosaicJSON document URL")], "application/json") }),
    );
    paths.insert(
        "/mosaicjson/validate".into(),
        json!({ "post": post_op("MosaicJSON", "Validate a MosaicJSON document", vec![], "#/components/schemas/MosaicJSON", "application/json") }),
    );
    paths.insert(
        "/mosaicjson/tiles/{tileMatrixSetId}/{z}/{x}/{y}/assets".into(),
        json!({ "get": op("MosaicJSON", "List mosaic assets intersecting a tile", vec![
            path_param("tileMatrixSetId", "string", "TileMatrixSet identifier"),
            path_param("z", "integer", "Tile zoom level"),
            path_param("x", "integer", "Tile column"),
            path_param("y", "integer", "Tile row"),
            q_str("url", true, "MosaicJSON document URL"),
        ], "application/json") }),
    );
    paths.insert(
        "/mosaicjson/point/{lon},{lat}/assets".into(),
        json!({ "get": op("MosaicJSON", "List mosaic assets intersecting a point", vec![
            path_param("lon", "number", "Longitude or X coordinate"),
            path_param("lat", "number", "Latitude or Y coordinate"),
            q_str("url", true, "MosaicJSON document URL"),
            q_str("coord_crs", false, "CRS of lon,lat coordinates"),
        ], "application/json") }),
    );
    paths.insert(
        "/mosaicjson/bbox/{minx},{miny},{maxx},{maxy}/assets".into(),
        json!({ "get": op("MosaicJSON", "List mosaic assets intersecting a bounding box", vec![
            path_param("minx", "number", "Minimum X"),
            path_param("miny", "number", "Minimum Y"),
            path_param("maxx", "number", "Maximum X"),
            path_param("maxy", "number", "Maximum Y"),
            q_str("url", true, "MosaicJSON document URL"),
            q_str("coord_crs", false, "CRS of bbox coordinates"),
        ], "application/json") }),
    );

    // Zarr family + Zarr-specific introspection & validation routes
    add_family_routes(
        &mut paths,
        "/zarr",
        "Zarr",
        "Zarr store URL",
        FamilyCaps {
            assets: false,
            mosaic: false,
            viewer: false,
            ogc_maps: false,
        },
    );
    paths.insert(
        "/zarr/validate".into(),
        json!({ "get": op("Zarr", "Validate Zarr v2/v3 structure, codecs, sharding, regularity and CRS", vec![
            q_str("url", true, "Zarr store URL"),
            q_str("variable", false, "Variable name"),
            q_str("group", false, "Group subtree path"),
        ], "application/json") }),
    );
    paths.insert(
        "/zarr/dataset/".into(),
        json!({ "get": op("Zarr", "List data variables in a Zarr store or group", vec![
            q_str("url", true, "Zarr store URL"),
            q_str("group", false, "Group subtree path"),
        ], "application/json") }),
    );
    paths.insert(
        "/zarr/dataset/keys".into(),
        json!({ "get": op("Zarr", "List data variable keys in a Zarr store or group", vec![
            q_str("url", true, "Zarr store URL"),
            q_str("group", false, "Group subtree path"),
        ], "application/json") }),
    );
    paths.insert(
        "/zarr/dataset/dict".into(),
        json!({ "get": op("Zarr", "Return xarray-style dataset dictionary (dims, attrs, coords, data_vars)", vec![
            q_str("url", true, "Zarr store URL"),
            q_str("group", false, "Group subtree path"),
        ], "application/json") }),
    );
    paths.insert(
        "/zarr/dataset/dimensions".into(),
        json!({ "get": op("Zarr", "Return dimension names and sizes for a Zarr store or group", vec![
            q_str("url", true, "Zarr store URL"),
            q_str("group", false, "Group subtree path"),
        ], "application/json") }),
    );
    paths.insert(
        "/zarr/dataset/coordinates/{name}".into(),
        json!({ "get": op("Zarr", "Return 1-D coordinate array values (with optional CF time decoding)", vec![
            path_param("name", "string", "Coordinate array name"),
            q_str("url", true, "Zarr store URL"),
            q_str("group", false, "Group subtree path"),
            q_bool("decode_times", false, "Decode CF time coordinates to ISO-8601 (default true)"),
        ], "application/json") }),
    );

    json!({
        "openapi": "3.1.0",
        "info": {
            "title": "cogzilla",
            "version": env!("CARGO_PKG_VERSION"),
            "description": "Zero-GDAL Rust + SIMD drop-in TiTiler COG, STAC, MosaicJSON & Zarr tile server"
        },
        "paths": Value::Object(paths),
        "components": {
            "schemas": {
                "HTTPValidationError": {
                    "type": "object",
                    "properties": {
                        "detail": {
                            "type": "array",
                            "items": { "$ref": "#/components/schemas/ValidationError" }
                        }
                    }
                },
                "ValidationError": {
                    "type": "object",
                    "required": ["loc", "msg", "type"],
                    "properties": {
                        "loc": { "type": "array", "items": { "anyOf": [{ "type": "string" }, { "type": "integer" }] } },
                        "msg": { "type": "string" },
                        "type": { "type": "string" }
                    }
                },
                "GeoJSONFeature": {
                    "type": "object",
                    "description": "GeoJSON Feature, FeatureCollection, or Geometry"
                },
                "MosaicJSON": {
                    "type": "object",
                    "required": ["mosaicjson", "minzoom", "maxzoom", "tiles"],
                    "properties": {
                        "mosaicjson": { "type": "string" },
                        "minzoom": { "type": "integer" },
                        "maxzoom": { "type": "integer" },
                        "quadkey_zoom": { "type": "integer" },
                        "bounds": { "type": "array", "items": { "type": "number" } },
                        "tiles": { "type": "object" }
                    }
                }
            }
        }
    })
}

pub fn openapi_json_bytes() -> Bytes {
    static CACHED: OnceLock<Bytes> = OnceLock::new();
    CACHED
        .get_or_init(|| Bytes::from(build_openapi_spec().to_string()))
        .clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openapi_spec_covers_all_families_with_parameters_and_responses() {
        let spec = build_openapi_spec();
        let paths = spec["paths"].as_object().unwrap();
        assert!(
            paths.len() >= 80,
            "expected >= 80 documented TiTiler paths, got {}",
            paths.len()
        );
        for required in [
            "/cog/info",
            "/cog/preview",
            "/cog/feature",
            "/cog/map",
            "/cog/stac",
            "/cog/WMTSCapabilities.xml",
            "/stac/renders",
            "/stac/renders/{render_id}",
            "/mosaicjson/",
            "/mosaicjson/validate",
            "/mosaicjson/WMTSCapabilities.xml",
            "/tileMatrixSets/{tileMatrixSetId}",
        ] {
            assert!(paths.contains_key(required), "missing path {required}");
        }
        for (path, item) in paths {
            for (method, op_val) in item.as_object().unwrap() {
                assert!(
                    op_val.get("responses").is_some(),
                    "{method} {path} missing responses"
                );
                assert!(
                    op_val.get("parameters").is_some(),
                    "{method} {path} missing parameters"
                );
            }
        }
        let tile_params = paths["/cog/tiles/{tileMatrixSetId}/{z}/{x}/{y}.{format}"]["get"]
            ["parameters"]
            .as_array()
            .unwrap();
        assert!(tile_params.iter().any(|p| p["name"] == "url"));
        assert!(tile_params.iter().any(|p| p["name"] == "rescale"));
        assert!(tile_params.iter().any(|p| p["name"] == "colormap_name"));
    }

    fn fill_path_params(path: &str) -> String {
        let mut out = path.to_string();
        for (pat, val) in [
            ("{tileMatrixSetId}", "WebMercatorQuad"),
            ("{z}", "0"),
            ("{x}", "0"),
            ("{y}", "0"),
            ("{scale}", "1"),
            ("{format}", "png"),
            ("{lon}", "0"),
            ("{lat}", "0"),
            ("{minx}", "0"),
            ("{miny}", "0"),
            ("{maxx}", "1"),
            ("{maxy}", "1"),
            ("{width}", "64"),
            ("{height}", "64"),
            ("{algorithmId}", "hillshade"),
            ("{colorMapId}", "viridis"),
            ("{render_id}", "default"),
            ("{name}", "x"),
        ] {
            out = out.replace(pat, val);
        }
        assert!(
            !out.contains('{') && !out.contains('}'),
            "unhandled path placeholder in {path} -> {out}"
        );
        out
    }

    #[tokio::test]
    async fn every_documented_path_is_routed() {
        use std::sync::Arc;

        use http_body_util::Full;
        use hyper::{Method, Request, StatusCode};

        let engine = Arc::new(crate::engine::TileEngine::new(8 * 1024 * 1024));
        let spec = build_openapi_spec();
        let paths = spec["paths"].as_object().unwrap();

        let mut unrouted = Vec::new();
        for (path, item) in paths {
            let concrete = fill_path_params(path);
            for method_str in item.as_object().unwrap().keys() {
                let method = match method_str.as_str() {
                    "get" => Method::GET,
                    "post" => Method::POST,
                    "put" => Method::PUT,
                    "delete" => Method::DELETE,
                    other => panic!("unexpected HTTP method {other} for {path}"),
                };
                let req = Request::builder()
                    .method(method)
                    .uri(&concrete)
                    .body(Full::new(Bytes::new()))
                    .unwrap();
                let resp = crate::http::handle_request(Arc::clone(&engine), req)
                    .await
                    .unwrap();
                let status = resp.status();
                if matches!(status, StatusCode::NOT_FOUND | StatusCode::METHOD_NOT_ALLOWED)
                    || status.is_server_error()
                {
                    unrouted.push(format!(
                        "{} {} (via {}) -> {}",
                        method_str.to_uppercase(),
                        path,
                        concrete,
                        status.as_u16()
                    ));
                }
            }
        }
        assert!(
            unrouted.is_empty(),
            "OpenAPI advertises unrouted/disallowed/5xx paths:\n{}",
            unrouted.join("\n")
        );
    }
}
