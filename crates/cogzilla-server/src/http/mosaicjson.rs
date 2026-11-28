//! `/mosaicjson/*` and `/mosaic/*` route family handlers (extracted from `http.rs` for #419).

use super::*;

pub(super) async fn handle<B>(
    ctx: RequestCtx<'_>,
    method: Method,
    segs: &[&str],
    pq: ParsedQuery,
    req: Request<B>,
) -> Resp
where
    B: hyper::body::Body,
    B::Error: std::fmt::Display,
{
    let RequestCtx {
        engine,
        path,
        raw_query,
        host,
        forwarded_prefix,
        if_none_match,
    } = ctx;
    let engine = Arc::clone(engine);
// ------------------------------------------------------------------------
// Dynamic registration of spatial catalogs (MosaicJSON, STAC FeatureCollection,
// JSON array, or live STAC-API POST /search):
//   POST /mosaic/register
// ------------------------------------------------------------------------
if method == Method::POST && path == "/mosaic/register" {
    let res: Result<(Bytes, &'static str), EngineError> = async {
        let body_bytes = read_limited_body(req.into_body()).await?;
        let spec: serde_json::Value = serde_json::from_slice(&body_bytes)
            .map_err(|e| EngineError::BadRequest(format!("invalid JSON body: {e}")))?;
        let name = spec
            .get("name")
            .and_then(|v| v.as_str())
            .filter(|s| {
                !s.is_empty()
                    && *s != "tiles"
                    && *s != "register"
                    && !s.contains('/')
                    && !s.contains('?')
                    && !s.contains('#')
                    && !s.chars().any(|c| c.is_whitespace())
            })
            .ok_or_else(|| {
                EngineError::BadRequest("missing valid 'name' for mosaic catalog".into())
            })?
            .to_string();

        let catalog: Arc<dyn SceneCatalog> = if let Some(cat_val) = spec.get("catalog") {
            load_catalog_json_str(&cat_val.to_string()).map_err(EngineError::BadRequest)?
        } else if let Some(search_url) = spec.get("stac_api_url").and_then(|v| v.as_str()) {
            cogzilla_io::check_uri_allowed(search_url)?;
            let mut search_body = serde_json::Map::new();
            for k in ["collections", "bbox", "datetime", "limit", "query", "filter"] {
                if let Some(v) = spec.get(k) {
                    search_body.insert(k.to_string(), v.clone());
                }
            }
            let resp = engine
                .store
                .client
                .post(search_url)
                .header("content-type", "application/json")
                .body(serde_json::Value::Object(search_body).to_string())
                .send()
                .await
                .map_err(|e| match cogzilla_io::io_error_from_reqwest(e) {
                    io_err if io_err.is_fatal() => EngineError::Io(io_err),
                    other => EngineError::BadRequest(format!("STAC API search failed: {other}")),
                })?;
            if !resp.status().is_success() {
                return Err(EngineError::BadRequest(format!(
                    "STAC API search returned HTTP {}",
                    resp.status()
                )));
            }
            let body_bytes = cogzilla_io::read_response_capped(resp, MAX_POST_BODY_BYTES)
                .await
                .map_err(|e| {
                    EngineError::BadRequest(format!("STAC API response read failed: {e}"))
                })?;
            let text = String::from_utf8_lossy(&body_bytes);
            load_catalog_json_str(&text).map_err(EngineError::BadRequest)?
        } else {
            return Err(EngineError::BadRequest(
                "expected 'catalog' or 'stac_api_url' in /mosaic/register body".into(),
            ));
        };

        let count = catalog.len();
        if engine.catalog_count() >= 256 && engine.get_catalog(&name).is_none() {
            return Err(EngineError::BadRequest(
                "maximum registered mosaic catalogs (256) reached".into(),
            ));
        }
        engine.register_catalog(name.clone(), catalog);
        engine.tile_cache.clear();
        let body = serde_json::json!({
            "name": name,
            "scenes": count,
            "tiles_url": format!("{forwarded_prefix}/mosaic/{name}/tiles/WebMercatorQuad/{{z}}/{{x}}/{{y}}.png"),
        })
        .to_string();
        Ok((Bytes::from(body), "application/json"))
    }
    .await;
    return to_http_response(&engine, res, if_none_match);
}

// ------------------------------------------------------------------------
// TiTiler MosaicJSON Router:
//   GET  /mosaicjson/info?url=...
//   GET  /mosaicjson/bounds?url=...
//   GET  /mosaicjson/tiles?url=...
//   GET  /mosaicjson/tiles/{tms}?url=...
//   GET  /mosaicjson/point/{lon},{lat}?url=...
//   GET  /mosaicjson/{tms}/tilejson.json?url=...
//   GET  /mosaicjson/{tms}/WMTSCapabilities.xml?url=...
//   GET  /mosaicjson/{tms}/map.html?url=...
//   GET  /mosaicjson/tiles/{tms}/{z}/{x}/{y}[@scale][.{ext}]?url=...
//   POST /mosaicjson/validate
// ------------------------------------------------------------------------
if method == Method::POST && segs.first() == Some(&"mosaicjson") {
    let mj_url = pq.urls.first().cloned();
    let mut geo_meta: Option<([f64; 4], Option<u16>, bool)> = None;
    let mut x_assets: Vec<String> = Vec::new();
    let res: Result<(Bytes, &'static str), EngineError> = match segs {
        ["mosaicjson", "validate"] => async {
            let body_bytes = read_limited_body(req.into_body()).await?;
            let text = std::str::from_utf8(&body_bytes)
                .map_err(|e| EngineError::BadRequest(format!("invalid UTF-8 body: {e}")))?;
            let mj = MosaicJsonCatalog::from_json_str(text).map_err(EngineError::BadRequest)?;
            let clean = strip_null_object_fields(mj.raw_json.clone());
            Ok((Bytes::from(clean.to_string()), "application/json"))
        }
        .await,
        ["mosaicjson", "statistics"] => async {
            let u = mj_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
            let mj = load_mosaicjson(&engine, &u).await?;
            let body_bytes = read_limited_body(req.into_body()).await?;
            let stats_opts = build_stats_options(&pq, false)?;
            let mut geojson: serde_json::Value = serde_json::from_slice(&body_bytes)
                .map_err(|e| EngineError::BadRequest(format!("invalid GeoJSON: {e}")))?;
            let fc_crs = match pq.coord_crs.as_ref() {
                Some(c) => Some(c.clone()),
                None => crate::stats::extract_geojson_crs(&geojson)?,
            };
            let pq_shared = Arc::new(pq.clone());
            let compute_for_mask = {
                let eng_m = engine.clone();
                let mj_m = mj.clone();
                let opts_m = stats_opts.clone();
                let pq_m = pq_shared.clone();
                Arc::new(move |mask: PolygonMask, bbox: [f64; 4]| {
                    let eng = eng_m.clone();
                    let mj_c = mj_m.clone();
                    let opts = opts_m.clone();
                    let pq_c = pq_m.clone();
                    async move {
                        if mask.is_empty() {
                            if let Some(first_url) = mj_c.first_asset() {
                                match eng.store.open_uri(first_url).await.map(|c| eng.store.with_nodata(&c, pq_c.nodata)) {
                                    Ok(cog) => {
                                        let stats = compute_statistics_with_options(
                                            &eng,
                                            &cog,
                                            pq_c.max_size,
                                            Some(&mask),
                                            pq_c.unscale,
                                            pq_c.nodata,
                                            &opts,
                                        )
                                        .await?;
                                        return Ok((stats, Vec::new()));
                                    }
                                    Err(e)
                                        if e.is_fatal() =>
                                    {
                                        return Err(EngineError::Io(e));
                                    }
                                    Err(_) => {}
                                }
                            }
                            let mut empty = compute_f32_band_stats(&[], "b1", &opts);
                            if let Some(o) = empty.as_object_mut() {
                                o.insert("area".into(), serde_json::Value::Null);
                            }
                            return Ok((serde_json::json!({"b1": empty}), Vec::new()));
                        }
                        let urls = query_mosaicjson_wgs84_bbox(&mj_c, bbox);
                        compute_mosaic_statistics_with_fold(
                            &eng,
                            &urls,
                            bbox,
                            Some(&mask),
                            &pq_c,
                            &opts,
                        )
                        .await
                    }
                })
            };
            if geojson.get("type").and_then(|v| v.as_str()) == Some("FeatureCollection") {
                let used_batches = run_feature_collection_stats(
                    &mut geojson,
                    fc_crs.as_ref(),
                    move |mask, bbox| compute_for_mask((*mask).clone(), bbox),
                )
                .await?;
                let mut seen_assets = std::collections::HashSet::new();
                for used in used_batches {
                    for a in used {
                        if seen_assets.insert(a.clone()) {
                            x_assets.push(a);
                        }
                    }
                }
                Ok((Bytes::from(geojson.to_string()), "application/geo+json"))
            } else {
                let (mask, bbox) = parse_feature_mask_or_null(&geojson, fc_crs.as_ref())?;
                let (stats, used) = compute_for_mask(mask, bbox).await?;
                x_assets = used;
                if geojson.get("type").and_then(|v| v.as_str()) == Some("Feature") {
                    if let Some(props) =
                        geojson.get_mut("properties").and_then(|v| v.as_object_mut())
                    {
                        props.insert("statistics".into(), stats);
                    } else if let Some(obj) = geojson.as_object_mut() {
                        obj.insert(
                            "properties".into(),
                            serde_json::json!({"statistics": stats}),
                        );
                    }
                    Ok((Bytes::from(geojson.to_string()), "application/geo+json"))
                } else {
                    let feat = serde_json::json!({
                        "type": "Feature",
                        "geometry": geojson,
                        "properties": { "statistics": stats },
                    });
                    Ok((Bytes::from(feat.to_string()), "application/geo+json"))
                }
            }
        }
        .await,
        ["mosaicjson", feat_ext]
            if *feat_ext == "feature" || feat_ext.starts_with("feature.") =>
        async {
            let u = mj_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
            let mj = load_mosaicjson(&engine, &u).await?;
            let body_bytes = read_limited_body(req.into_body()).await?;
            let geojson: serde_json::Value = serde_json::from_slice(&body_bytes)
                .map_err(|e| EngineError::BadRequest(format!("invalid GeoJSON: {e}")))?;
            let (mask, bbox) = PolygonMask::from_geojson_in_crs(&geojson, pq.coord_crs.as_ref())?;
            let urls = query_mosaicjson_wgs84_bbox(&mj, bbox);
            let ext = feat_ext
                .strip_prefix("feature.")
                .or(pq.format_param.as_deref())
                .unwrap_or("png");
            let (fmt, ct) = parse_ext_format(ext, pq.webp_quality, pq.jpeg_quality)?;
            let (dst_crs, dst_epsg) = resolve_bbox_dst_crs(&pq)
                .unwrap_or((cogzilla_proj::CrsModel::Geographic, 4326));
            let out_bbox = reproject_wgs84_bounds(bbox, &dst_crs)?;
            let (def_w, def_h) = aspect_preserving_dims(out_bbox, pq.max_size);
            let (out_w, out_h) = resolve_query_dims(def_w, def_h, pq.width, pq.height);
            let mask_opt = pq.shape_mask.then_some(&mask);
            let (data, used) = render_mosaic_bbox_composite(
                &engine, &urls, out_bbox, &dst_crs, out_w, out_h, mask_opt, &pq, fmt, ct, dst_epsg,
            )
            .await?;
            let data = mosaic_array_in_its_type(
                &engine,
                array_asked(&pq, segs.last()),
                &used,
                pq.pixel_selection,
                data,
            )
            .await;
            x_assets = used;
            geo_meta = Some((out_bbox, Some(dst_epsg), false));
            Ok((data, ct))
        }
        .await,
        ["mosaicjson", "feature", dims_ext] => async {
            let (out_w, out_h, ext_opt) = parse_wxh_ext(dims_ext).map_err(EngineError::BadRequest)?;
            let u = mj_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
            let mj = load_mosaicjson(&engine, &u).await?;
            let body_bytes = read_limited_body(req.into_body()).await?;
            let geojson: serde_json::Value = serde_json::from_slice(&body_bytes)
                .map_err(|e| EngineError::BadRequest(format!("invalid GeoJSON: {e}")))?;
            let (mask, bbox) = PolygonMask::from_geojson_in_crs(&geojson, pq.coord_crs.as_ref())?;
            let urls = query_mosaicjson_wgs84_bbox(&mj, bbox);
            let ext = ext_opt.or(pq.format_param.as_deref()).unwrap_or("png");
            let (fmt, ct) = parse_ext_format(ext, pq.webp_quality, pq.jpeg_quality)?;
            let (dst_crs, dst_epsg) = resolve_bbox_dst_crs(&pq)
                .unwrap_or((cogzilla_proj::CrsModel::Geographic, 4326));
            let out_bbox = reproject_wgs84_bounds(bbox, &dst_crs)?;
            let mask_opt = pq.shape_mask.then_some(&mask);
            let (data, used) = render_mosaic_bbox_composite(
                &engine, &urls, out_bbox, &dst_crs, out_w, out_h, mask_opt, &pq, fmt, ct, dst_epsg,
            )
            .await?;
            let data = mosaic_array_in_its_type(
                &engine,
                array_asked(&pq, segs.last()),
                &used,
                pq.pixel_selection,
                data,
            )
            .await;
            x_assets = used;
            geo_meta = Some((out_bbox, Some(dst_epsg), false));
            Ok((data, ct))
        }
        .await,
        _ => Err(EngineError::NotFound(path.to_string())),
    };
    let mut resp = to_http_response(&engine, res, if_none_match);
    if let Some((bbox, epsg, is_crs84)) = geo_meta {
        attach_geo_headers(&mut resp, bbox, epsg, is_crs84);
    }
    attach_x_assets(&mut resp, &x_assets);
    return resp;
}

if method == Method::GET && segs.first() == Some(&"mosaicjson") {
    let mj_url = pq.urls.first().cloned();
    let mut geo_meta: Option<([f64; 4], Option<u16>, bool)> = None;
    let mut x_assets: Vec<String> = Vec::new();
    let res: Result<(Bytes, &'static str), EngineError> = match segs {
        ["mosaicjson"] => async {
            let u = mj_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
            let mj = load_mosaicjson(&engine, &u).await?;
            let clean = strip_null_object_fields(mj.raw_json.clone());
            Ok((Bytes::from(clean.to_string()), "application/json"))
        }
        .await,
        ["mosaicjson", "viewer"] => {
            let tj_url = format!("{forwarded_prefix}/mosaicjson/WebMercatorQuad/tilejson.json?{raw_query}");
            Ok((Bytes::from(build_map_html(&tj_url)), "text/html; charset=utf-8"))
        }
        ["mosaicjson", "info"] => async {
            let u = mj_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
            let mj = load_mosaicjson(&engine, &u).await?;
            let mut info = mj.info_json();
            apply_info_bounds_crs(&mut info, mj.bounds, pq.bounds_crs.as_ref())?;
            Ok((Bytes::from(info.to_string()), "application/json"))
        }
        .await,
        ["mosaicjson", "info.geojson"] => async {
            let u = mj_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
            let mj = load_mosaicjson(&engine, &u).await?;
            let mut props = mj.info_json();
            apply_info_bounds_crs(&mut props, mj.bounds, pq.bounds_crs.as_ref())?;
            let [w, s, e, n] = mj.bounds;
            let feat = serde_json::json!({
                "type": "Feature",
                "bbox": mj.bounds,
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]
                },
                "properties": props
            });
            Ok((Bytes::from(feat.to_string()), "application/geo+json"))
        }
        .await,
        ["mosaicjson", "assets"] => async {
            let u = mj_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
            let mj = load_mosaicjson(&engine, &u).await?;
            let assets: Vec<String> = if let Some(b) = pq.bbox_param {
                let b = resolve_input_bbox(b, pq.coord_crs.as_ref())?;
                let wgs_bbox = bbox_to_wgs84(b, pq.coord_crs.as_ref())?;
                query_mosaicjson_wgs84_bbox(&mj, wgs_bbox)
            } else {
                let [mx0, my0, mx1, my1] = crate::catalog::wgs84_bbox_to_3857(mj.bounds);
                mj.query_bbox_3857(mx0, my0, mx1, my1)
            };
            Ok((Bytes::from(serde_json::json!(assets).to_string()), "application/json"))
        }
        .await,
        ["mosaicjson", "bounds"] => async {
            let u = mj_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
            let mj = load_mosaicjson(&engine, &u).await?;
            let bounds = if let Some((ref dst, _)) = pq.bounds_crs {
                reproject_wgs84_bounds(mj.bounds, dst)?
            } else {
                mj.bounds
            };
            let body = serde_json::json!({ "bounds": bounds }).to_string();
            Ok((Bytes::from(body), "application/json"))
        }
        .await,
        ["mosaicjson", "statistics"] => async {
            let u = mj_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
            let mj = load_mosaicjson(&engine, &u).await?;
            let [mx0, my0, mx1, my1] = crate::catalog::wgs84_bbox_to_3857(mj.bounds);
            let urls = mj.query_bbox_3857(mx0, my0, mx1, my1);
            let stats_opts = build_stats_options(&pq, false)?;
            let (json, used) = compute_mosaic_statistics_with_fold(
                &engine, &urls, mj.bounds, None, &pq, &stats_opts,
            )
            .await?;
            x_assets = used;
            Ok((Bytes::from(json.to_string()), "application/json"))
        }
        .await,
        ["mosaicjson", "bbox", bbox_seg] => async {
            let Some((raw_bbox, ext_opt)) = parse_bbox4(bbox_seg) else {
                return Err(EngineError::BadRequest("invalid bbox".into()));
            };
            let bbox = resolve_input_bbox(raw_bbox, pq.coord_crs.as_ref())?;
            let u = mj_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
            let mj = load_mosaicjson(&engine, &u).await?;
            let wgs_bbox = bbox_to_wgs84(bbox, pq.coord_crs.as_ref())?;
            let urls = query_mosaicjson_wgs84_bbox(&mj, wgs_bbox);
            let ext = ext_opt.or(pq.format_param.as_deref()).unwrap_or("png");
            let (fmt, ct) = parse_ext_format(ext, pq.webp_quality, pq.jpeg_quality)?;
            let (dst_crs, dst_epsg) = resolve_bbox_dst_crs(&pq).unwrap_or({
                if pq.coord_crs_3857 {
                    (cogzilla_proj::CrsModel::Epsg3857, 3857)
                } else {
                    (cogzilla_proj::CrsModel::Geographic, 4326)
                }
            });
            let out_bbox = if pq.coord_crs_epsg == Some(dst_epsg) {
                bbox
            } else {
                reproject_wgs84_bounds(wgs_bbox, &dst_crs)?
            };
            let (def_w, def_h) = aspect_preserving_dims(out_bbox, pq.max_size);
            let (out_w, out_h) = resolve_query_dims(def_w, def_h, pq.width, pq.height);
            let (data, used) = render_mosaic_bbox_composite(
                &engine, &urls, out_bbox, &dst_crs, out_w, out_h, None, &pq, fmt, ct, dst_epsg,
            )
            .await?;
            let data = mosaic_array_in_its_type(
                &engine,
                array_asked(&pq, segs.last()),
                &used,
                pq.pixel_selection,
                data,
            )
            .await;
            x_assets = used;
            geo_meta = Some((out_bbox, Some(dst_epsg), false));
            Ok((data, ct))
        }
        .await,
        ["mosaicjson", "bbox", bbox_seg, "assets"] => async {
            let Some((raw_bbox, _)) = parse_bbox4(bbox_seg) else {
                return Err(EngineError::BadRequest("invalid bbox".into()));
            };
            let bbox = resolve_input_bbox(raw_bbox, pq.coord_crs.as_ref())?;
            let u = mj_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
            let mj = load_mosaicjson(&engine, &u).await?;
            let wgs_bbox = bbox_to_wgs84(bbox, pq.coord_crs.as_ref())?;
            let urls = query_mosaicjson_wgs84_bbox(&mj, wgs_bbox);
            Ok((Bytes::from(serde_json::json!(urls).to_string()), "application/json"))
        }
        .await,
        ["mosaicjson", "bbox", bbox_seg, dims_ext] => async {
            let Some((raw_bbox, _)) = parse_bbox4(bbox_seg) else {
                return Err(EngineError::BadRequest("invalid bbox".into()));
            };
            let bbox = resolve_input_bbox(raw_bbox, pq.coord_crs.as_ref())?;
            let (out_w, out_h, ext_opt) = parse_wxh_ext(dims_ext).map_err(EngineError::BadRequest)?;
            let u = mj_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
            let mj = load_mosaicjson(&engine, &u).await?;
            let wgs_bbox = bbox_to_wgs84(bbox, pq.coord_crs.as_ref())?;
            let urls = query_mosaicjson_wgs84_bbox(&mj, wgs_bbox);
            let ext = ext_opt.or(pq.format_param.as_deref()).unwrap_or("png");
            let (fmt, ct) = parse_ext_format(ext, pq.webp_quality, pq.jpeg_quality)?;
            let (dst_crs, dst_epsg) = resolve_bbox_dst_crs(&pq).unwrap_or({
                if pq.coord_crs_3857 {
                    (cogzilla_proj::CrsModel::Epsg3857, 3857)
                } else {
                    (cogzilla_proj::CrsModel::Geographic, 4326)
                }
            });
            let out_bbox = if pq.coord_crs_epsg == Some(dst_epsg) {
                bbox
            } else {
                reproject_wgs84_bounds(wgs_bbox, &dst_crs)?
            };
            let (data, used) = render_mosaic_bbox_composite(
                &engine, &urls, out_bbox, &dst_crs, out_w, out_h, None, &pq, fmt, ct, dst_epsg,
            )
            .await?;
            let data = mosaic_array_in_its_type(
                &engine,
                array_asked(&pq, segs.last()),
                &used,
                pq.pixel_selection,
                data,
            )
            .await;
            x_assets = used;
            geo_meta = Some((out_bbox, Some(dst_epsg), false));
            Ok((data, ct))
        }
        .await,
        ["mosaicjson", "tiles"] => async {
            let bounds = if let Some(ref u) = mj_url {
                crate::engine::tolerate_asset_failure(load_mosaicjson(&engine, u).await)?
                    .map(|m| m.bounds)
            } else {
                None
            };
            let body = build_ogc_tilesets_list(
                host,
                forwarded_prefix,
                "mosaicjson",
                raw_query,
                bounds,
            )
            .to_string();
            Ok((Bytes::from(body), "application/json"))
        }
        .await,
        ["mosaicjson", "tiles", tms] => async {
            if let Some(tms_def) = crate::tms::lookup_tms(tms) {
                let (bounds, minz, maxz) = if let Some(ref u) = mj_url {
                    match crate::engine::tolerate_asset_failure(
                        load_mosaicjson(&engine, u).await,
                    )? {
                        Some(m) => (
                            Some(m.bounds),
                            pq.minzoom.or(Some(m.minzoom)),
                            pq.maxzoom.or(Some(m.maxzoom)),
                        ),
                        None => (None, pq.minzoom, pq.maxzoom),
                    }
                } else {
                    (None, pq.minzoom, pq.maxzoom)
                };
                let body = build_ogc_tileset_json(
                    host,
                    forwarded_prefix,
                    "mosaicjson",
                    tms_def,
                    raw_query,
                    bounds,
                    minz,
                    maxz,
                )
                .to_string();
                Ok((Bytes::from(body), "application/json"))
            } else {
                Err(EngineError::NotFound(format!(
                    "unknown TileMatrixSet '{tms}'"
                )))
            }
        }
        .await,
        ["mosaicjson", "point", coord] => async {
            let u = mj_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
            let mj = load_mosaicjson(&engine, &u).await?;
            let Some((x, y)) = parse_point_coord(coord) else {
                return Err(EngineError::BadRequest(
                    "expected /mosaicjson/point/{lon},{lat}".into(),
                ));
            };
            let (lon, lat) = coord_to_lonlat(x, y, pq.coord_crs.as_ref())?;
            let (mx, my) = if pq.coord_crs_3857 {
                (x, y)
            } else {
                lonlat_deg_to_mercator(lon, lat)
            };
            let urls = mj.query_bbox_3857(mx, my, mx, my);
            let mut assets_out = Vec::new();
            let mut opened_any = false;
            let mut last_open_err: Option<EngineError> = None;
            for asset_url in urls.into_iter().take(16) {
                let cog = match engine.store.open_uri(&asset_url).await.map(|c| engine.store.with_nodata(&c, pq.nodata)) {
                    Ok(c) => {
                        opened_any = true;
                        c
                    }
                    Err(e) if e.is_fatal() => {
                        return Err(EngineError::Io(e));
                    }
                    Err(e) => {
                        last_open_err = Some(crate::engine::asset_fetch_error(&asset_url, e));
                        continue;
                    }
                };
                match query_point(&engine, &cog, lon, lat, pq.unscale, pq.nodata).await {
                    Ok(mut pt) => {
                        apply_point_filters(&mut pt, pq.expr_raw.as_deref(), &pq.bidx)?;
                        let vals = pt["values"].clone();
                        let names = pt["band_names"].clone();
                        let descs = pt
                            .get("band_descriptions")
                            .cloned()
                            .unwrap_or_else(|| names.clone());
                        assets_out.push(serde_json::json!({
                            "name": asset_url,
                            "values": vals,
                            "band_names": names,
                            "band_descriptions": descs,
                        }));
                    }
                    Err(EngineError::NotFound(_)) => continue,
                    Err(e) => return Err(e),
                }
            }
            if assets_out.is_empty() {
                if !opened_any
                    && let Some(err) = last_open_err
                {
                    return Err(err);
                }
                return Err(EngineError::NoContent);
            }
            let body = serde_json::json!({
                "coordinates": [x, y],
                "assets": assets_out,
            })
            .to_string();
            Ok((Bytes::from(body), "application/json"))
        }
        .await,
        ["mosaicjson", _, "WMTSCapabilities.xml"] | ["mosaicjson", "WMTSCapabilities.xml"] => {
            async {
                let explicit_tms = (segs.len() == 3).then_some(segs[1]);
                if let Some(tms) = explicit_tms
                    && crate::tms::lookup_tms(tms).is_none()
                {
                    return Err(EngineError::NotFound(format!("unknown TileMatrixSet '{tms}'")));
                }
                let u = mj_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
                let mj = load_mosaicjson(&engine, &u).await?;
                let (minz, maxz) =
                    resolve_zoom_range(mj.minzoom, mj.maxzoom, pq.minzoom, pq.maxzoom);
                let fmt = pq.tile_format.as_deref().unwrap_or("png");
                let scale = pq.tile_scale.unwrap_or(1);
                let tile_q = filter_tile_template_query(raw_query);
                let xml = if let Some(tms) = explicit_tms {
                    build_wmts_capabilities(
                        host,
                        "mosaicjson",
                        tms,
                        "MosaicJSON",
                        mj.bounds,
                        minz,
                        maxz,
                        fmt,
                        scale,
                        &tile_q,
                        raw_query,
                        pq.use_epsg,
                    )
                } else {
                    let entries: Vec<(&str, u8, u8)> = crate::tms::ALL_TMS
                        .iter()
                        .map(|t| (t.id, minz, maxz))
                        .collect();
                    build_wmts_capabilities_multi(
                        host,
                        "mosaicjson",
                        &entries,
                        None,
                        "MosaicJSON",
                        mj.bounds,
                        fmt,
                        scale,
                        &tile_q,
                        raw_query,
                        pq.use_epsg,
                    )
                };
                Ok((Bytes::from(xml), "application/xml"))
            }
            .await
        }
        ["mosaicjson", _, "tilejson.json"] | ["mosaicjson", "tilejson.json"] => async {
            let tms = if segs.len() == 3 { segs[1] } else { "WebMercatorQuad" };
            let u = mj_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
            let mj = load_mosaicjson(&engine, &u).await?;
            let center = serde_json::json!([
                0.5 * (mj.bounds[0] + mj.bounds[2]),
                0.5 * (mj.bounds[1] + mj.bounds[3]),
                mj.minzoom,
            ]);
            let mut tj = serde_json::json!({
                "tilejson": "3.0.0",
                "version": mj.doc_version,
                "scheme": "xyz",
                "tiles": [],
                "minzoom": mj.minzoom,
                "maxzoom": mj.maxzoom,
                "bounds": mj.bounds,
                "center": center,
                "raster_layers": {},
            });
            apply_tilejson_overrides(&mut tj, &pq, host, "mosaicjson", tms, raw_query);
            Ok((Bytes::from(tj.to_string()), "application/json"))
        }
        .await,
        ["mosaicjson", _, "map.html"] | ["mosaicjson", "map.html"] => {
            let tms = if segs.len() == 3 { segs[1] } else { "WebMercatorQuad" };
            let tj_url =
                format!("{forwarded_prefix}/mosaicjson/{tms}/tilejson.json?{raw_query}");
            Ok((Bytes::from(build_map_html(&tj_url)), "text/html; charset=utf-8"))
        }
        ["mosaicjson", "point", coord, "assets"] => async {
            let u = mj_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
            let mj = load_mosaicjson(&engine, &u).await?;
            let Some((x, y)) = parse_point_coord(coord) else {
                return Err(EngineError::BadRequest("expected /mosaicjson/point/{lon},{lat}/assets".into()));
            };
            let (lon, lat) = coord_to_lonlat(x, y, pq.coord_crs.as_ref())?;
            let (mx, my) = lonlat_deg_to_mercator(lon, lat);
            let eps = 1e-6;
            let urls = mj.query_bbox_3857(mx - eps, my - eps, mx + eps, my + eps);
            Ok((Bytes::from(serde_json::json!(urls).to_string()), "application/json"))
        }
        .await,
        ["mosaicjson", "tiles", tms, zs, xs, ys, "assets"] => async {
            let u = mj_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
            let mj = load_mosaicjson(&engine, &u).await?;
            let (Some(z), Some(x), Some(y)) = (parse_u8_like(zs), parse_u32_like(xs), parse_u32_like(ys)) else {
                return Err(EngineError::BadRequest("invalid z/x/y tile coord".into()));
            };
            if z > 30 {
                return Err(EngineError::BadRequest(format!("invalid zoom level z={z} (expected 0..=30)")));
            }
            let urls = if *tms == "WebMercatorQuad" {
                let max_xy = 1u64 << (z as u32);
                if (x as u64) >= max_xy || (y as u64) >= max_xy {
                    return Err(EngineError::BadRequest(format!(
                        "tile coordinate ({x}, {y}) out of range for WebMercatorQuad z={z} (expected x, y < {max_xy})"
                    )));
                }
                let b = web_mercator_buffered_3857_bbox(z, x, y, pq.buffer, 256);
                mj.query_bbox_3857(b[0], b[1], b[2], b[3])
            } else {
                let Some(tms_def) = crate::tms::lookup_tms(tms) else {
                    return Err(EngineError::NotFound(format!("unknown TileMatrixSet '{tms}'")));
                };
                let Some(tb) = tms_def.tile_bounds(z, x, y, pq.buffer, 256) else {
                    return Err(EngineError::BadRequest(format!(
                        "tile coordinate ({z}/{x}/{y}) out of range for TileMatrixSet '{tms}'"
                    )));
                };
                let dst_crs = tms_def.crs_model()?;
                let wgs = bbox_to_wgs84(tb, Some(&dst_crs))?;
                query_mosaicjson_wgs84_bbox(&mj, wgs)
            };
            Ok((Bytes::from(serde_json::json!(urls).to_string()), "application/json"))
        }
        .await,
        ["mosaicjson", "tiles", zs, xs, ys, "assets"]
        | ["mosaicjson", zs, xs, ys, "assets"] => async {
            let u = mj_url.ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
            let mj = load_mosaicjson(&engine, &u).await?;
            let (Some(z), Some(x), Some(y)) = (parse_u8_like(zs), parse_u32_like(xs), parse_u32_like(ys)) else {
                return Err(EngineError::BadRequest("invalid z/x/y tile coord".into()));
            };
            if z > 30 {
                return Err(EngineError::BadRequest(format!("invalid zoom level z={z} (expected 0..=30)")));
            }
            let max_xy = 1u64 << (z as u32);
            if (x as u64) >= max_xy || (y as u64) >= max_xy {
                return Err(EngineError::BadRequest(format!(
                    "tile coordinate ({x}, {y}) out of range for WebMercatorQuad z={z} (expected x, y < {max_xy})"
                )));
            }
            let b = web_mercator_buffered_3857_bbox(z, x, y, pq.buffer, 256);
            let urls = mj.query_bbox_3857(b[0], b[1], b[2], b[3]);
            Ok((Bytes::from(serde_json::json!(urls).to_string()), "application/json"))
        }
        .await,
        ["mosaicjson", "tiles", _, zs, xs, ys]
        | ["mosaicjson", "tiles", zs, xs, ys] => {
            let tms = if segs.len() == 6 { segs[2] } else { "WebMercatorQuad" };
            let cache_key = format!("{path}?{raw_query}");
            if let (Some(u), Some(z), Some(x), Some((y, _, _))) = (
                mj_url.as_deref(),
                parse_u8_like(zs),
                parse_u32_like(xs),
                parse_y_scale_ext(ys),
            ) && let Some(mj) = engine.mosaicjson_cache.get(u)
                && let Some(tms_def) = crate::tms::lookup_tms(tms)
                && let Some(b) = tms_def.tile_bounds(z, x, y, pq.buffer, 256)
            {
                let urls = if tms_def.id == "WebMercatorQuad" {
                    mj.query_bbox_3857(b[0], b[1], b[2], b[3])
                } else {
                    mj.query_bbox_3857(-20_037_508.34, -20_037_508.34, 20_037_508.34, 20_037_508.34)
                };
                let _ = engine.store.ensure_uris_fresh(&urls).await;
                engine.sync_tile_cache_epoch();
            }
            let failed_assets = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
            let failed_assets_inner = failed_assets.clone();
            let cached = engine
                .tile_cache
                .get_or_render(cache_key, || async {
                    let u = mj_url
                        .ok_or_else(|| EngineError::BadRequest("missing ?url=".into()))?;
                    let mj = load_mosaicjson(&engine, &u).await?;
                    let (Some(z), Some(x), Some((y, sc, ext))) = (
                        parse_u8_like(zs),
                        parse_u32_like(xs),
                        parse_y_scale_ext(ys),
                    ) else {
                        return Err(EngineError::BadRequest(
                            "invalid /mosaicjson/tiles coordinate path".into(),
                        ));
                    };
                    if z > 30 {
                        return Err(EngineError::BadRequest(format!(
                            "invalid zoom level z={z} (expected 0..=30)"
                        )));
                    }
                    let tile = TileCoord { z, x, y };
                    let auto_fmt = ext.is_none() && pq.format_param.is_none();
                    if tms != "WebMercatorQuad" {
                        let tms_def = crate::tms::lookup_tms(tms).ok_or_else(|| {
                            EngineError::NotFound(format!("unknown TileMatrixSet '{tms}'"))
                        })?;
                        let query_ts = match sc {
                            Some(s) if (1..=4).contains(&s) => s * 256,
                            _ => pq.tile_size,
                        };
                        let tb = tms_def
                            .tile_bounds(z, x, y, pq.buffer, query_ts)
                            .ok_or_else(|| {
                                EngineError::BadRequest(format!(
                                    "tile coordinate ({z}/{x}/{y}) out of range for TileMatrixSet '{tms}'"
                                ))
                            })?;
                        let dst_crs = tms_def.crs_model()?;
                        let wgs = bbox_to_wgs84(tb, Some(&dst_crs))?;
                        let urls = query_mosaicjson_wgs84_bbox(&mj, wgs);
                        let empty_default_bands = if pq.bidx.is_empty()
                            && (pq.colormap_explicit || pq.custom_colormap_json.is_some())
                        {
                            1
                        } else {
                            3
                        };
                        let (mut treq, ct) = build_tile_request_for_tms(
                            Some(&engine),
                            tile,
                            sc,
                            ext,
                            &pq,
                            empty_default_bands,
                            cogzilla_raster::SampleType::Uint8,
                            tms,
                            None,
                            true,
                        )?;
                        validate_mosaic_tile_params(&pq, treq.tile_size)?;
                        let out_dim = (treq.tile_size as f32 + 2.0 * pq.buffer).round() as u32;
                        if auto_fmt {
                            treq.format = OutputFormat::RawRgba;
                        }
                        if urls.is_empty() {
                            if !wgs84_bboxes_intersect(wgs, mj.bounds) {
                                return Err(EngineError::NotFound(format!(
                                    "Tile {tms}/{z}/{x}/{y} is outside mosaic bounds [{:.6}, {:.6}, {:.6}, {:.6}]",
                                    mj.bounds[0], mj.bounds[1], mj.bounds[2], mj.bounds[3]
                                )));
                            }
                            return Err(EngineError::NoContent);
                        }
                        let (data, used) = render_mosaic_bbox_composite(
                            &engine,
                            &urls,
                            tb,
                            &dst_crs,
                            out_dim,
                            out_dim,
                            None,
                            &pq,
                            treq.format,
                            if auto_fmt { "application/octet-stream" } else { ct },
                            tms_def.epsg,
                        )
                        .await?;
                        if auto_fmt {
                            return Ok(encode_auto_image(
                                out_dim as usize,
                                out_dim as usize,
                                &data,
                                pq.jpeg_quality,
                                pq.return_mask,
                            ));
                        }
                        let data = mosaic_array_in_its_type(
                            &engine,
                            array_asked(&pq, segs.last()),
                            &used,
                            pq.pixel_selection,
                            data,
                        )
                        .await;
                        return Ok((data, ct));
                    }
                    let query_ts = match sc {
                        Some(s) if (1..=4).contains(&s) => s * 256,
                        _ => pq.tile_size,
                    };
                    let buf_bbox =
                        web_mercator_buffered_3857_bbox(z, x, y, pq.buffer, query_ts);
                    let urls = mj.query_bbox_3857(
                        buf_bbox[0],
                        buf_bbox[1],
                        buf_bbox[2],
                        buf_bbox[3],
                    );
                    if urls.is_empty() {
                        let empty_default_bands = if pq.bidx.is_empty()
                            && (pq.colormap_explicit || pq.custom_colormap_json.is_some())
                        {
                            1
                        } else {
                            3
                        };
                        let (treq, _) = build_tile_request(
                            Some(&engine),
                            tile,
                            sc,
                            ext,
                            &pq,
                            empty_default_bands,
                            cogzilla_raster::SampleType::Uint8,
                            false,
                            None,
                        )?;
                        validate_mosaic_tile_params(&pq, treq.tile_size)?;
                        let (lo0, la0) = mercator_to_lonlat_deg(buf_bbox[0], buf_bbox[1]);
                        let (lo1, la1) = mercator_to_lonlat_deg(buf_bbox[2], buf_bbox[3]);
                        let tile_wgs = [lo0.min(lo1), la0.min(la1), lo0.max(lo1), la0.max(la1)];
                        if !wgs84_bboxes_intersect(tile_wgs, mj.bounds) {
                            Err(EngineError::NotFound(format!(
                                "Tile WebMercatorQuad/{z}/{x}/{y} is outside mosaic bounds [{:.6}, {:.6}, {:.6}, {:.6}]",
                                mj.bounds[0], mj.bounds[1], mj.bounds[2], mj.bounds[3]
                            )))
                        } else {
                            Err(EngineError::NoContent)
                        }
                    } else {
                        let first_cog = open_first_mosaic_cog(&engine, &urls).await?;
                        let db = first_cog.index.full_res().samples_per_pixel;
                        let st = first_cog.index.full_res().sample_type;
                        let (mut treq, ct) = build_tile_request(
                            Some(&engine),
                            tile,
                            sc,
                            ext,
                            &pq,
                            db,
                            st,
                            false,
                            Some(&first_cog),
                        )?;
                        validate_mosaic_tile_params(&pq, treq.tile_size)?;
                        let ts = treq.tile_size;
                        if auto_fmt {
                            treq.format = OutputFormat::RawRgba;
                        }
                        let fmt = treq.format;
                        let out_dim = (ts as f32 + 2.0 * pq.buffer).round() as u32;
                        let buf_bbox = web_mercator_buffered_3857_bbox(z, x, y, pq.buffer, ts);
                        let mut failed = Vec::new();
                        let b = render_mosaic_tile_styled(
                            &engine,
                            &urls,
                            treq,
                            &pq,
                            first_cog.index.color_map.is_some(),
                            &mut failed,
                        )
                        .await?;
                        if !failed.is_empty() {
                            *failed_assets_inner.lock().unwrap_or_else(|e| e.into_inner()) = failed;
                        }
                        let (data, ct) = finalize_image_output(
                            b,
                            ct,
                            fmt,
                            auto_fmt,
                            out_dim,
                            out_dim,
                            buf_bbox,
                            3857,
                            pq.jpeg_quality,
                            pq.return_mask,
                        );
                        let data = mosaic_array_in_its_type(
                            &engine,
                            array_asked(&pq, segs.last()),
                            &urls,
                            pq.pixel_selection,
                            data,
                        )
                        .await;
                        Ok((data, ct))
                    }
                })
                .await;
            let mut resp = to_http_response_cached(&engine, cached, if_none_match);
            let failed = failed_assets.lock().unwrap_or_else(|e| e.into_inner()).clone();
            attach_x_failed_assets(&mut resp, &failed);
            if let (Some(z), Some(x), Some((y, sc, _))) =
                (parse_u8_like(zs), parse_u32_like(xs), parse_y_scale_ext(ys))
            {
                attach_tile_geo_headers(&mut resp, tms, z, x, y, sc, &pq);
            }
            return resp;
        }
        _ => Err(EngineError::NotFound(path.to_string())),
    };
    let mut resp = to_http_response(&engine, res, if_none_match);
    if let Some((bbox, epsg, is_crs84)) = geo_meta {
        attach_geo_headers(&mut resp, bbox, epsg, is_crs84);
    }
    attach_x_assets(&mut resp, &x_assets);
    return resp;
}

// ------------------------------------------------------------------------
// Dynamic early-exit COG mosaic:
//   GET /mosaic/tiles/{tileMatrixSetId}/{z}/{x}/{y}[@scale][.{ext}]?url=a&url=b&...
//   GET /mosaic/tiles/{z}/{x}/{y}[@scale][.{ext}]?url=a&url=b&...
// Indexed spatial-catalog mosaic:
//   GET /mosaic/{name}/tiles/{tileMatrixSetId}/{z}/{x}/{y}[@scale][.{ext}]
//   GET /mosaic/{name}/tiles/{z}/{x}/{y}[@scale][.{ext}]
// ------------------------------------------------------------------------
if method == Method::GET && segs.first() == Some(&"mosaic") && segs.get(1) == Some(&"tiles") {
    let coords = match segs {
        ["mosaic", "tiles", "WebMercatorQuad", zs, xs, ys]
        | ["mosaic", "tiles", zs, xs, ys] => (
            parse_u8_like(zs),
            parse_u32_like(xs),
            parse_y_scale_ext(ys),
        ),
        _ => (None, None, None),
    };
    let cache_key = format!("{path}?{raw_query}");
    let res = match coords {
        (Some(z), Some(x), Some((y, sc, ext))) => {
            if z > 30 {
                Err(EngineError::BadRequest(format!(
                    "invalid zoom level z={z} (expected 0..=30)"
                )))
            } else if pq.urls.is_empty() {
                Err(EngineError::BadRequest(
                    "missing required ?url= parameter(s) for /mosaic/tiles".into(),
                ))
            } else if pq.urls.len() > 16 {
                Err(EngineError::BadRequest(format!(
                    "too many ?url= parameters ({}; max 16)",
                    pq.urls.len()
                )))
            } else {
                if let Err(e) = engine.store.ensure_uris_fresh(&pq.urls).await {
                    engine.sync_tile_cache_epoch();
                    if !matches!(e.root(), cogzilla_io::IoError::Fs(io) if io.kind() == std::io::ErrorKind::NotFound) {
                        return to_http_response(
                            &engine,
                            Err(EngineError::Io(e)),
                            if_none_match,
                        );
                    }
                } else {
                    engine.sync_tile_cache_epoch();
                }
                let failed_assets = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
                let failed_assets_inner = failed_assets.clone();
                let cached = engine
                    .tile_cache
                    .get_or_render(cache_key, || async {
                        let first_cog = open_first_mosaic_cog(&engine, &pq.urls).await?;
                        let db = first_cog.index.full_res().samples_per_pixel;
                        let st = first_cog.index.full_res().sample_type;
                        let (treq, ct) = build_tile_request(
                            Some(&engine),
                            TileCoord { z, x, y },
                            sc,
                            ext,
                            &pq,
                            db,
                            st,
                            false,
                            Some(&first_cog),
                        )?;
                        validate_mosaic_tile_params(&pq, treq.tile_size)?;
                        let ts = treq.tile_size;
                        let fmt = treq.format;
                        let out_dim = (ts as f32 + 2.0 * pq.buffer).round() as u32;
                        let buf_bbox = web_mercator_buffered_3857_bbox(z, x, y, pq.buffer, ts);
                        let mut failed = Vec::new();
                        let b = render_mosaic_tile_styled(
                            &engine,
                            &pq.urls,
                            treq,
                            &pq,
                            first_cog.index.color_map.is_some(),
                            &mut failed,
                        )
                        .await?;
                        if !failed.is_empty() {
                            *failed_assets_inner.lock().unwrap_or_else(|e| e.into_inner()) = failed;
                        }
                        let b = wrap_geotiff_fmt(
                            b,
                            ct,
                            fmt,
                            out_dim,
                            out_dim,
                            buf_bbox,
                            3857,
                            pq.return_mask,
                        );
                        Ok((b, ct))
                    })
                    .await;
                let mut resp = to_http_response_cached(&engine, cached, if_none_match);
                let failed = failed_assets.lock().unwrap_or_else(|e| e.into_inner()).clone();
                attach_x_failed_assets(&mut resp, &failed);
                attach_tile_geo_headers(&mut resp, "WebMercatorQuad", z, x, y, sc, &pq);
                return resp;
            }
        }
        _ => Err(EngineError::BadRequest(
            "invalid /mosaic/tiles coordinate path".into(),
        )),
    };
    return to_http_response(&engine, res, if_none_match);
}

if method == Method::GET && segs.first() == Some(&"mosaic") && segs.get(2) == Some(&"tiles") {
    let cat_name = segs[1];
    let coords = match segs {
        ["mosaic", _, "tiles", "WebMercatorQuad", zs, xs, ys]
        | ["mosaic", _, "tiles", zs, xs, ys] => (
            parse_u8_like(zs),
            parse_u32_like(xs),
            parse_y_scale_ext(ys),
        ),
        _ => (None, None, None),
    };
    let res = match (engine.get_catalog(cat_name), coords) {
        (None, _) => Err(EngineError::NotFound(format!("mosaic catalog '{cat_name}'"))),
        (Some(catalog), (Some(z), Some(x), Some((y, sc, ext)))) => {
            let cache_key = format!("{path}?{raw_query}");
            if z <= 30 {
                let (min_x, min_y, max_x, max_y) =
                    cogzilla_proj::tile_bounds_3857(TileCoord { z, x, y });
                let urls = catalog.query_bbox_3857(min_x, min_y, max_x, max_y);
                let _ = engine.store.ensure_uris_fresh(&urls).await;
                engine.sync_tile_cache_epoch();
            }
            let failed_assets = std::sync::Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
            let failed_assets_c = failed_assets.clone();
            let cached = engine
                .tile_cache
                .get_or_render(cache_key, || async {
                    if z > 30 {
                        return Err(EngineError::BadRequest(format!(
                            "invalid zoom level z={z} (expected 0..=30)"
                        )));
                    }
                    let tile = TileCoord { z, x, y };
                    let query_ts = match sc {
                        Some(s) if (1..=4).contains(&s) => s * 256,
                        _ => pq.tile_size,
                    };
                    let buf_bbox =
                        web_mercator_buffered_3857_bbox(z, x, y, pq.buffer, query_ts);
                    let urls = catalog.query_bbox_3857(
                        buf_bbox[0],
                        buf_bbox[1],
                        buf_bbox[2],
                        buf_bbox[3],
                    );
                    if urls.is_empty() {
                        let empty_default_bands = if pq.bidx.is_empty()
                            && (pq.colormap_explicit || pq.custom_colormap_json.is_some())
                        {
                            1
                        } else {
                            3
                        };
                        let (treq, ct) = build_tile_request(
                            Some(&engine),
                            tile,
                            sc,
                            ext,
                            &pq,
                            empty_default_bands,
                            cogzilla_raster::SampleType::Uint8,
                            false,
                            None,
                        )?;
                        validate_mosaic_tile_params(&pq, treq.tile_size)?;
                        let out_dim = (treq.tile_size as f32 + 2.0 * pq.buffer).round() as u32;
                        let nb = default_raw_band_count(&treq.bands, pq.pixel_selection);
                        let b = if treq.format == OutputFormat::Npy {
                            let raw = cogzilla_kernel::empty_tile_bytes(
                                out_dim,
                                OutputFormat::RawF32,
                                nb,
                            );
                            cogzilla_kernel::wrap_raw_f32_as_npy_with_mask(
                                raw,
                                nb,
                                out_dim,
                                out_dim,
                                pq.return_mask,
                            )
                        } else {
                            cogzilla_kernel::empty_tile_bytes(out_dim, treq.format, nb)
                        };
                        let b = wrap_geotiff_fmt(
                            b,
                            ct,
                            treq.format,
                            out_dim,
                            out_dim,
                            buf_bbox,
                            3857,
                            pq.return_mask,
                        );
                        Ok((b, ct))
                    } else {
                        let first_cog = open_first_mosaic_cog(&engine, &urls).await?;
                        let db = first_cog.index.full_res().samples_per_pixel;
                        let st = first_cog.index.full_res().sample_type;
                        let (treq, ct) = build_tile_request(
                            Some(&engine),
                            tile,
                            sc,
                            ext,
                            &pq,
                            db,
                            st,
                            false,
                            Some(&first_cog),
                        )?;
                        validate_mosaic_tile_params(&pq, treq.tile_size)?;
                        let ts = treq.tile_size;
                        let fmt = treq.format;
                        let out_dim = (ts as f32 + 2.0 * pq.buffer).round() as u32;
                        let buf_bbox = web_mercator_buffered_3857_bbox(z, x, y, pq.buffer, ts);
                        let mut failed = Vec::new();
                        let b = render_mosaic_tile_styled(
                            &engine,
                            &urls,
                            treq,
                            &pq,
                            first_cog.index.color_map.is_some(),
                            &mut failed,
                        )
                        .await?;
                        if !failed.is_empty()
                            && let Ok(mut guard) = failed_assets_c.lock()
                        {
                            *guard = failed;
                        }
                        let b = wrap_geotiff_fmt(
                            b,
                            ct,
                            fmt,
                            out_dim,
                            out_dim,
                            buf_bbox,
                            3857,
                            pq.return_mask,
                        );
                        Ok((b, ct))
                    }
                })
                .await;
            let mut resp = to_http_response_cached(&engine, cached, if_none_match);
            attach_tile_geo_headers(&mut resp, "WebMercatorQuad", z, x, y, sc, &pq);
            if let Ok(guard) = failed_assets.lock() {
                attach_x_failed_assets(&mut resp, &guard);
            }
            return resp;
        }
        _ => Err(EngineError::BadRequest(
            "invalid /mosaic/{name}/tiles coordinate path".into(),
        )),
    };
    return to_http_response(&engine, res, if_none_match);
}
    let allow = if matches!(segs, ["mosaicjson", "validate" | "statistics" | "feature", ..]) {
        "POST"
    } else {
        "GET, HEAD"
    };
    if segs.first() == Some(&"mosaicjson") {
        method_not_allowed_resp(allow)
    } else {
        json_error_resp(StatusCode::NOT_FOUND, path)
    }
}
