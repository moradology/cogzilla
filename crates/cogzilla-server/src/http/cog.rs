//! `/cog/*` route family handler (extracted from `http.rs` for #419).

use super::*;

pub(super) async fn handle<B>(
    ctx: RequestCtx<'_>,
    method: Method,
    sub: &[&str],
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
    let mut geo_meta: Option<([f64; 4], Option<u16>, bool)> = None;
    let cog_array = array_asked(&pq, sub.last());
    let cog_urls = pq.urls.clone();
    let res: Result<(Bytes, &'static str), EngineError> = match (method.clone(), sub) {
        // 1. GET /cog/info
        (Method::GET, ["info"]) => async {
            let cogs = open_query_urls(&engine, &pq.urls, None).await?;
            let mut json = build_cog_info(&engine, &cogs[0])?;
            if let Some((ref dst, dst_epsg)) = pq.bounds_crs {
                let crs = engine.get_or_build_crs(&cogs[0])?;
                let full = cogs[0].index.full_res();
                let wgs = crs.wgs84_bounds(
                    &full.pixel_to_crs,
                    full.gcp_poly.as_deref(),
                    full.width,
                    full.height,
                );
                let rep = reproject_wgs84_bounds(wgs, dst)?;
                json["bounds"] = serde_json::json!(rep);
                json["crs"] = serde_json::json!(format!("http://www.opengis.net/def/crs/EPSG/0/{dst_epsg}"));
            }
            Ok((Bytes::from(json.to_string()), "application/json"))
        }
        .await,

        // 2. GET /cog/info.geojson
        (Method::GET, ["info.geojson"]) => async {
            let cogs = open_query_urls(&engine, &pq.urls, None).await?;
            let mut json = build_cog_info_geojson(&engine, &cogs[0])?;
            if let Some((ref dst, dst_epsg)) = pq.bounds_crs {
                let crs = engine.get_or_build_crs(&cogs[0])?;
                let full = cogs[0].index.full_res();
                let wgs = crs.wgs84_bounds(
                    &full.pixel_to_crs,
                    full.gcp_poly.as_deref(),
                    full.width,
                    full.height,
                );
                let rep = reproject_wgs84_bounds(wgs, dst)?;
                json["properties"]["bounds"] = serde_json::json!(rep);
                json["properties"]["crs"] = serde_json::json!(format!("http://www.opengis.net/def/crs/EPSG/0/{dst_epsg}"));
            }
            Ok((Bytes::from(json.to_string()), "application/geo+json"))
        }
        .await,

        // Alias: GET /cog/bounds
        (Method::GET, ["bounds"]) => async {
            let cogs = open_query_urls(&engine, &pq.urls, None).await?;
            let crs = engine.get_or_build_crs(&cogs[0])?;
            let full = cogs[0].index.full_res();
            let bounds = crs.wgs84_bounds(
                &full.pixel_to_crs,
                full.gcp_poly.as_deref(),
                full.width,
                full.height,
            );
            let bounds = if let Some((ref dst, _)) = pq.bounds_crs {
                reproject_wgs84_bounds(bounds, dst)?
            } else {
                bounds
            };
            let body = serde_json::json!({"bounds": bounds}).to_string();
            Ok((Bytes::from(body), "application/json"))
        }
        .await,

        // GET /cog/validate?url=...&strict=false
        (Method::GET, ["validate"]) => async {
            let cogs = open_query_urls(&engine, &pq.urls, None).await?;
            let url = pq.urls.first().map(String::as_str).unwrap_or("");
            let json = build_cog_validate(&engine, &cogs[0], url, pq.strict)?;
            Ok((Bytes::from(json.to_string()), "application/json"))
        }
        .await,

        // 3. GET / POST /cog/statistics (POST supports GeoJSON Polygon clipping!)
        (Method::GET, ["statistics"]) => async {
            let cogs = open_query_urls(&engine, &pq.urls, pq.nodata).await?;
            let full = cogs[0].index.full_res();
            let db = data_band_count(full);
            let is_dem_alg = pq.algorithm.as_deref().is_some_and(|a| {
                crate::algorithms::lookup_algorithm(a)
                    .is_some_and(|d| d.kind == crate::algorithms::AlgorithmKind::Dem)
            });
            if is_dem_alg {
                let crs = engine.get_or_build_crs(&cogs[0])?;
                let bbox = crs.wgs84_bounds(&full.pixel_to_crs, full.gcp_poly.as_deref(), full.width, full.height);
                let scale = (pq.max_size as f64) / (full.width.max(full.height) as f64);
                let def_w = ((full.width as f64 * scale.min(1.0)).ceil() as u32).max(1);
                let def_h = ((full.height as f64 * scale.min(1.0)).ceil() as u32).max(1);
                let (out_w, out_h) = resolve_query_dims(def_w, def_h, pq.width, pq.height);
                let mut bands = build_band_request(&pq, db, full.sample_type)?;
                if pq.unscale {
                    apply_unscale_cogs(&mut bands, &cogs, None);
                }
                let stats_opts = build_stats_options(&pq, false)?;
                let alg_name = pq.algorithm.as_deref().unwrap_or("algorithm");
                if matches!(
                    bands,
                    BandRequest::Single { .. }
                        | BandRequest::Dem { .. }
                        | BandRequest::CustomExpr { .. }
                        | BandRequest::Spectral1 { .. }
                        | BandRequest::Spectral2 { .. }
                        | BandRequest::Spectral3 { .. }
                ) {
                    let stat = render_native_preview_stats(
                        &engine,
                        &cogs,
                        bbox,
                        out_w,
                        out_h,
                        &bands,
                        pq.resample,
                        pq.nodata,
                        pq.unscale,
                        alg_name,
                        &stats_opts,
                    )
                    .await?;
                    let mut map = serde_json::Map::new();
                    map.insert("b1".into(), stat);
                    return Ok((
                        Bytes::from(serde_json::Value::Object(map).to_string()),
                        "application/json",
                    ));
                }
                let raw = render_native_preview(
                    &engine,
                    &cogs,
                    bbox,
                    out_w,
                    out_h,
                    &bands,
                    pq.resample,
                    OutputFormat::RawF32,
                    pq.nodata,
                    None,
                    None,
                    true,
                    pq.unscale,
                )
                .await?;
                let n_px = (out_w as usize) * (out_h as usize);
                let nb = if n_px > 0 && raw.len() >= n_px * 4 {
                    (raw.len() / (n_px * 4)).clamp(1, 16)
                } else {
                    1
                };
                let alg_name = pq.algorithm.as_deref().unwrap_or("algorithm");
                let mut map = serde_json::Map::new();
                for b in 0..nb {
                    let start = b * n_px * 4;
                    let end = (start + n_px * 4).min(raw.len());
                    let plane: Vec<f32> = raw[start..end]
                        .chunks_exact(4)
                        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect();
                    let desc = if nb == 1 {
                        alg_name.to_string()
                    } else {
                        format!("{alg_name}_b{}", b + 1)
                    };
                    map.insert(
                        format!("b{}", b + 1),
                        compute_f32_band_stats_with_engine(&engine, &plane, &desc, &stats_opts),
                    );
                }
                return Ok((Bytes::from(serde_json::Value::Object(map).to_string()), "application/json"));
            }
            let stats_opts = build_stats_options_with_bands(&pq, true, db, Some(reads_type(&cogs)))?;
            let json = compute_statistics_with_options(
                &engine,
                &cogs[0],
                pq.max_size,
                None,
                pq.unscale,
                pq.nodata,
                &stats_opts,
            )
            .await?;
            Ok((Bytes::from(json.to_string()), "application/json"))
        }
        .await,
        (Method::POST, ["statistics"]) => async {
            let body_bytes = read_limited_body(req.into_body()).await?;
            let cogs = open_query_urls(&engine, &pq.urls, pq.nodata).await?;
            let db = data_band_count(cogs[0].index.full_res());
            let stats_opts = build_stats_options_with_bands(&pq, true, db, Some(reads_type(&cogs)))?;
            if body_bytes.is_empty() {
                let json = compute_statistics_with_options(
                    &engine,
                    &cogs[0],
                    pq.max_size,
                    None,
                    pq.unscale,
                    pq.nodata,
                    &stats_opts,
                )
                .await?;
                Ok((Bytes::from(json.to_string()), "application/json"))
            } else {
                let mut geojson: serde_json::Value = serde_json::from_slice(&body_bytes)
                    .map_err(|e| EngineError::BadRequest(format!("invalid GeoJSON: {e}")))?;
                if geojson.get("type").and_then(|v| v.as_str()) == Some("FeatureCollection") {
                    let fc_crs = match pq.coord_crs.as_ref() {
                        Some(c) => Some(c.clone()),
                        None => crate::stats::extract_geojson_crs(&geojson)?,
                    };
                    let eng_c = engine.clone();
                    let cog_c = cogs[0].clone();
                    let opts_c = stats_opts.clone();
                    let max_size = pq.max_size;
                    let unscale = pq.unscale;
                    let nodata = pq.nodata;
                    run_feature_collection_stats(
                        &mut geojson,
                        fc_crs.as_ref(),
                        move |mask, _| {
                            let eng_c = eng_c.clone();
                            let cog_c = cog_c.clone();
                            let opts_c = opts_c.clone();
                            async move {
                                let stats = compute_statistics_with_options(
                                    &eng_c,
                                    &cog_c,
                                    max_size,
                                    Some(&mask),
                                    unscale,
                                    nodata,
                                    &opts_c,
                                )
                                .await?;
                                Ok((stats, ()))
                            }
                        },
                    )
                    .await?;
                    Ok((Bytes::from(geojson.to_string()), "application/geo+json"))
                } else {
                    let (mask, _) =
                        parse_feature_mask_or_null(&geojson, pq.coord_crs.as_ref())?;
                    let stats = compute_statistics_with_options(
                        &engine,
                        &cogs[0],
                        pq.max_size,
                        Some(&mask),
                        pq.unscale,
                        pq.nodata,
                        &stats_opts,
                    )
                    .await?;
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
        }
        .await,

        // 4. GET /cog/tiles (OGC Tiles list)
        (Method::GET, ["tiles"]) => async {
            let bounds = if !pq.urls.is_empty() {
                crate::engine::tolerate_asset_failure(open_query_urls(&engine, &pq.urls, None).await)?
                    .and_then(|cogs| {
                        engine.get_or_build_crs(&cogs[0]).ok().map(|crs| {
                            let full = cogs[0].index.full_res();
                            crs.wgs84_bounds(&full.pixel_to_crs, full.gcp_poly.as_deref(), full.width, full.height)
                        })
                    })
            } else {
                None
            };
            let body = build_ogc_tilesets_list(
                host,
                forwarded_prefix,
                "cog",
                raw_query,
                bounds,
            )
            .to_string();
            Ok((Bytes::from(body), "application/json"))
        }
        .await,

        // 5. GET /cog/tiles/{tileMatrixSetId}
        (Method::GET, ["tiles", tms]) => async {
            let tms_def = crate::tms::lookup_tms(tms)
                .ok_or_else(|| EngineError::NotFound(format!("unknown TileMatrixSet '{tms}'")))?;
            let (bounds, minz, maxz) = if !pq.urls.is_empty() {
                if let Some(cogs) =
                    crate::engine::tolerate_asset_failure(open_query_urls(&engine, &pq.urls, None).await)?
                {
                    if let Ok(crs) = engine.get_or_build_crs(&cogs[0]) {
                        let full = cogs[0].index.full_res();
                        let coarsest = cogs[0].index.ifds.last().unwrap();
                        let ov = (full.width as f64 / coarsest.width.max(1) as f64)
                            .max(full.height as f64 / coarsest.height.max(1) as f64);
                        let (dmin, dmax) = tms_def.zoom_range_for_cog(
                            &crs,
                            &full.pixel_to_crs,
                            full.width,
                            full.height,
                            ov,
                        );
                        let (mn, mx) = resolve_zoom_range(dmin, dmax, pq.minzoom, pq.maxzoom);
                        (
                            Some(crs.wgs84_bounds(&full.pixel_to_crs, full.gcp_poly.as_deref(), full.width, full.height)),
                            Some(mn),
                            Some(mx),
                        )
                    } else {
                        (None, pq.minzoom, pq.maxzoom)
                    }
                } else {
                    (None, pq.minzoom, pq.maxzoom)
                }
            } else {
                (None, pq.minzoom, pq.maxzoom)
            };
            let body = build_ogc_tileset_json(
                host,
                forwarded_prefix,
                "cog",
                tms_def,
                raw_query,
                bounds,
                minz,
                maxz,
            )
            .to_string();
            Ok((Bytes::from(body), "application/json"))
        }
        .await,

        // 6 & 7. GET /cog/tiles/{tileMatrixSetId}/{z}/{x}/{y}[@scale][.{format}]
        //        and alias GET /cog/tiles/{z}/{x}/{y}[@scale][.{format}] (implicit WebMercatorQuad)
        (Method::GET, ["tiles", _, zs, xs, ys]) | (Method::GET, ["tiles", zs, xs, ys]) => {
            let tms = if sub.len() == 5 { sub[1] } else { "WebMercatorQuad" };
            let cache_key = format!("{path}?{raw_query}");
            if let Err(e) = engine.store.ensure_uris_fresh(&pq.urls).await {
                engine.sync_tile_cache_epoch();
                return error_to_http_response(&EngineError::Io(e));
            }
            engine.sync_tile_cache_epoch();
            let cached = engine
                .tile_cache
                .get_or_render(cache_key, || cog_tile_in_its_type(&engine, cog_array.clone(), &cog_urls, async {
                    let cogs = open_query_urls(&engine, &pq.urls, pq.nodata).await?;
                    let (Some(z), Some(x), Some((y, sc, ext))) = (
                        parse_u8_like(zs),
                        parse_u32_like(xs),
                        parse_y_scale_ext(ys),
                    ) else {
                        return Err(EngineError::BadRequest("invalid z/x/y tile coord".into()));
                    };
                    let db = if cogs.len() > 1 {
                        cogs.len() as u16
                    } else {
                        cogs[0].index.full_res().samples_per_pixel
                    };
                    let st = widest_sample_type(&cogs);
                    let (mut treq, ct) = build_tile_request_for_tms(
                        Some(&engine),
                        TileCoord { z, x, y },
                        sc,
                        ext,
                        &pq,
                        db,
                        st,
                        tms,
                        Some(&cogs[0]),
                        true,
                    )?;
                    if pq.unscale {
                        apply_unscale_cogs(&mut treq.bands, &cogs, None);
                    }
                    if matches!(treq.format, OutputFormat::RawF32 | OutputFormat::Npy) {
                        treq.format = geotiff_render_format(
                            treq.format,
                            ct,
                            &pq,
                            Some(&cogs[0]),
                            &treq.bands,
                            st,
                            true,
                        );
                    }

                    let mut any_intersects = false;
                    for c in &cogs {
                        if tile_intersects_cog(
                            &engine,
                            c,
                            tms,
                            TileCoord { z, x, y },
                            pq.buffer,
                            treq.tile_size,
                        )? {
                            any_intersects = true;
                            break;
                        }
                    }
                    if !any_intersects {
                        let full = cogs[0].index.full_res();
                        let b = engine
                            .get_or_build_crs(&cogs[0])
                            .map(|c| c.wgs84_bounds(&full.pixel_to_crs, full.gcp_poly.as_deref(), full.width, full.height))
                            .unwrap_or([-180.0, -90.0, 180.0, 90.0]);
                        return Err(EngineError::NotFound(format!(
                            "Tile {tms}/{z}/{x}/{y} is outside dataset bounds [{:.6}, {:.6}, {:.6}, {:.6}]",
                            b[0], b[1], b[2], b[3]
                        )));
                    }
                    let needs_general_path = tms != "WebMercatorQuad"
                        || pq.buffer > 0.0
                        || pq.nodata.is_some()
                        || pq.custom_colormap_json.is_some()
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
                        || matches!(
        treq.bands,
        BandRequest::CustomExprRgb { .. }
            | BandRequest::CustomExprMulti { .. }
            | BandRequest::MultiBand { .. }
    )
                        || (treq.tile_size != 256 && treq.tile_size != 512)
                        || has_embedded_palette(&pq, &cogs[0], &treq.bands)
                        || !crate::engine::cogs_share_base_grid(&cogs);

                    let auto_fmt = ext.is_none() && pq.format_param.is_none();
                    if auto_fmt {
                        treq.format = OutputFormat::RawRgba;
                    } else if wants_u16_png(treq.format, ct, &pq, Some(&cogs[0]), &treq.bands, st) {
                        treq.format = OutputFormat::RawF32;
                    }
                    if let Some(ref cf_str) = pq.color_formula {
                        ColorFormula::parse(cf_str).map_err(EngineError::BadRequest)?;
                    }

                    if needs_general_path {
                        let custom_lut = resolve_custom_lut(&pq, &cogs[0], &treq.bands)?;
                        let cf = pq.parse_color_formula()?;
                        let out_dim = (treq.tile_size as f32 + 2.0 * pq.buffer).round() as u32;
                        let (data, bbox, epsg) = if tms != "WebMercatorQuad" {
                            let tms_def = crate::tms::lookup_tms(tms).ok_or_else(|| {
                                EngineError::BadRequest(format!("unsupported TMS '{tms}'"))
                            })?;
                            let bbox = tms_def
                                .tile_bounds(z, x, y, pq.buffer, treq.tile_size)
                                .ok_or_else(|| {
                                    EngineError::NotFound(format!(
                                        "Tile {z}/{x}/{y} is outside {tms} matrix"
                                    ))
                                })?;
                            let dst_crs = tms_def.crs_model()?;
                            let d = render_crs_bbox(
                                &engine,
                                &cogs,
                                bbox,
                                &dst_crs,
                                out_dim,
                                out_dim,
                                None,
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
                            (d, bbox, tms_def.epsg)
                        } else {
                            let bbox_3857 = web_mercator_buffered_3857_bbox(
                                z,
                                x,
                                y,
                                pq.buffer,
                                treq.tile_size,
                            );
                            let d = render_mercator_bbox(
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
                            (d, bbox_3857, 3857u16)
                        };
                        Ok(finalize_image_output(data, ct, treq.format, auto_fmt, out_dim, out_dim, bbox, epsg, pq.jpeg_quality, pq.return_mask))
                    } else if (pq.color_formula.is_some() || !pq.return_mask)
                        && !matches!(treq.format, OutputFormat::RawF32 | OutputFormat::RawF64 | OutputFormat::Npy | OutputFormat::NpyF64)
                    {
                        let cf = pq.parse_color_formula()?;
                        let orig_format = treq.format;
                        treq.format = OutputFormat::RawRgba;
                        let raw = engine.render_multi_cog_tile(&cogs, &treq).await?;
                        let dim = treq.tile_size as usize;
                        let bbox = web_mercator_buffered_3857_bbox(z, x, y, 0.0, treq.tile_size);
                        if auto_fmt {
                            let buf = if let Some(cf) = cf {
                                let mut b = raw.to_vec();
                                cf.apply_buffer_in_place(&mut b, 4);
                                Bytes::from(b)
                            } else {
                                raw
                            };
                            Ok(encode_auto_image(dim, dim, &buf, pq.jpeg_quality, pq.return_mask))
                        } else {
                            let data = if let Some(cf) = cf {
                                let mut buf = raw.to_vec();
                                cf.apply_buffer_in_place(&mut buf, 4);
                                encode_rgba_output(dim, dim, &buf, orig_format, pq.return_mask)
                            } else {
                                encode_rgba_output(dim, dim, &raw, orig_format, pq.return_mask)
                            };
                            let data = wrap_geotiff_fmt(data, ct, orig_format, treq.tile_size, treq.tile_size, bbox, 3857, pq.return_mask);
                            Ok((data, ct))
                        }
                    } else {
                        let data = engine.render_multi_cog_tile(&cogs, &treq).await?;
                        let bbox = web_mercator_buffered_3857_bbox(z, x, y, 0.0, treq.tile_size);
                        Ok(finalize_image_output(data, ct, treq.format, auto_fmt, treq.tile_size, treq.tile_size, bbox, 3857, pq.jpeg_quality, pq.return_mask))
                    }
                }))
                .await;
            let mut resp = to_http_response_cached(&engine, cached, if_none_match);
            if let (Some(z), Some(x), Some((y, sc, _))) =
                (parse_u8_like(zs), parse_u32_like(xs), parse_y_scale_ext(ys))
            {
                attach_tile_geo_headers(&mut resp, tms, z, x, y, sc, &pq);
            }
            return resp;
        }

        // 8. GET /cog/{tileMatrixSetId}/tilejson.json & /cog/tilejson.json
        (Method::GET, [tms, "tilejson.json"]) => async {
            let cogs = open_query_urls(&engine, &pq.urls, None).await?;
            let mut json = build_tilejson(&engine, &cogs[0], host, tms, raw_query)?;
            apply_tilejson_overrides(&mut json, &pq, host, "cog", tms, raw_query);
            Ok((Bytes::from(json.to_string()), "application/json"))
        }
        .await,
        (Method::GET, ["tilejson.json"]) => async {
            let cogs = open_query_urls(&engine, &pq.urls, None).await?;
            let mut json =
                build_tilejson(&engine, &cogs[0], host, "WebMercatorQuad", raw_query)?;
            apply_tilejson_overrides(&mut json, &pq, host, "cog", "WebMercatorQuad", raw_query);
            Ok((Bytes::from(json.to_string()), "application/json"))
        }
        .await,

        // 9. GET /cog/{tileMatrixSetId}/map.html & /cog/map.html
        (Method::GET, [_, "map.html"]) | (Method::GET, ["map.html"]) | (Method::GET, ["viewer"]) => {
            let tms = if sub.len() == 2 { sub[0] } else { "WebMercatorQuad" };
            let tj_url = format!("{forwarded_prefix}/cog/{tms}/tilejson.json?{raw_query}");
            Ok((Bytes::from(build_map_html(&tj_url)), "text/html; charset=utf-8"))
        }
        // GET /cog/stac — build a STAC 1.1.0 Item JSON from a COG (matching TiTiler / rio-stac)
        (Method::GET, ["stac"]) => async {
            let cogs = open_query_urls(&engine, &pq.urls, None).await?;
            let cog = &cogs[0];
            let (crs, native_epsg, native_bounds) =
                dataset_native_crs_and_bounds(&engine, cog)?;
            let full = cog.index.full_res();
            let wgs_bbox = crs.wgs84_bounds(
                &full.pixel_to_crs,
                full.gcp_poly.as_deref(),
                full.width,
                full.height,
            );
            let [w, s, e, n] = wgs_bbox;
            let [nw, ns, ne, nn] = native_bounds;
            let mut q_id: Option<String> = None;
            let mut q_collection: Option<String> = None;
            let mut q_collection_url: Option<String> = None;
            let mut q_asset_name: Option<String> = None;
            let mut q_asset_href: Option<String> = None;
            let mut q_asset_media_type: Option<String> = None;
            let mut q_asset_roles: Vec<String> = Vec::new();
            let mut q_extensions: Vec<String> = Vec::new();
            let mut q_datetime: Option<String> = None;
            let mut with_proj = true;
            let mut with_raster = true;
            let mut with_eo = true;
            for pair in raw_query.split('&').filter(|p| !p.is_empty()) {
                if let Some((k, v)) = pair.split_once('=') {
                    let dec = percent_decode(v);
                    match query_key(k).as_ref() {
                        "id" if !dec.is_empty() => q_id = Some(dec),
                        "collection" if !dec.is_empty() => q_collection = Some(dec),
                        "collection_url" if !dec.is_empty() => q_collection_url = Some(dec),
                        "asset_name" if !dec.is_empty() => q_asset_name = Some(dec),
                        "asset_href" if !dec.is_empty() => q_asset_href = Some(dec),
                        "asset_media_type" if !dec.is_empty() => q_asset_media_type = Some(dec),
                        "asset_roles" => {
                            for r in dec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                                q_asset_roles.push(r.to_string());
                            }
                        }
                        "extensions" => {
                            for ext in dec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
                                q_extensions.push(ext.to_string());
                            }
                        }
                        "datetime" if !dec.is_empty() => q_datetime = Some(dec),
                        "with_proj" => match parse_bool_param(v) {
                            Some(b) => with_proj = b,
                            None => return Err(EngineError::BadRequest(format!("invalid with_proj={dec}: expected boolean"))),
                        },
                        "with_raster" => match parse_bool_param(v) {
                            Some(b) => with_raster = b,
                            None => return Err(EngineError::BadRequest(format!("invalid with_raster={dec}: expected boolean"))),
                        },
                        "with_eo" => match parse_bool_param(v) {
                            Some(b) => with_eo = b,
                            None => return Err(EngineError::BadRequest(format!("invalid with_eo={dec}: expected boolean"))),
                        },
                        _ => {}
                    }
                }
            }
            let item_id = q_id.unwrap_or_else(|| {
                std::path::Path::new(&cog.uri)
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("cog-item")
                    .to_string()
            });
            let asset_name = q_asset_name.unwrap_or_else(|| "data".to_string());
            let asset_href = q_asset_href.unwrap_or_else(|| cog.uri.clone());
            let media_type = match q_asset_media_type.as_deref() {
                None | Some("auto") => "image/tiff; application=geotiff".to_string(),
                Some("cog") => "image/tiff; application=geotiff; profile=cloud-optimized".to_string(),
                Some(other) => other.to_string(),
            };
            let mut extensions: Vec<String> = Vec::new();
            let mut push_ext = |uri: &str| {
                if !extensions.iter().any(|e| e == uri) {
                    extensions.push(uri.to_string());
                }
            };
            let mut props = serde_json::Map::new();
            props.insert(
                "datetime".into(),
                serde_json::json!(q_datetime.unwrap_or_else(utc_now_rfc3339)),
            );
            if with_proj {
                push_ext("https://stac-extensions.github.io/projection/v2.0.0/schema.json");
                props.insert("proj:epsg".into(), serde_json::json!(native_epsg));
                props.insert("proj:shape".into(), serde_json::json!([full.height, full.width]));
                props.insert(
                    "proj:transform".into(),
                    serde_json::json!([
                        full.pixel_to_crs.a,
                        full.pixel_to_crs.b,
                        full.pixel_to_crs.c,
                        full.pixel_to_crs.d,
                        full.pixel_to_crs.e,
                        full.pixel_to_crs.f,
                        0.0,
                        0.0,
                        1.0
                    ]),
                );
                props.insert("proj:bbox".into(), serde_json::json!(native_bounds));
                props.insert(
                    "proj:geometry".into(),
                    serde_json::json!({
                        "type": "Polygon",
                        "coordinates": [[[nw, ns], [ne, ns], [ne, nn], [nw, nn], [nw, ns]]]
                    }),
                );
            }
            let count = full.samples_per_pixel as usize;
            let mut asset_obj = serde_json::json!({
                "href": asset_href,
                "type": media_type,
                "roles": q_asset_roles,
            });
            if with_raster {
                push_ext("https://stac-extensions.github.io/raster/v1.1.0/schema.json");
                let sampling = if cog.index.geo_keys.raster_type == Some(2) {
                    "point"
                } else {
                    "area"
                };
                let stats_json = compute_statistics_with_options(
                    &engine,
                    cog,
                    128,
                    None,
                    false,
                    None,
                    &StatsOptions::default(),
                )
                .await
                .ok();
                let rb: Vec<serde_json::Value> = (0..count)
                    .map(|i| {
                        let mut b = serde_json::Map::new();
                        b.insert("data_type".into(), serde_json::json!(full.sample_type.gdal_name()));
                        b.insert("scale".into(), serde_json::json!(cog.index.scales_f64.get(i).copied().unwrap_or(1.0)));
                        b.insert("offset".into(), serde_json::json!(cog.index.offsets_f64.get(i).copied().unwrap_or(0.0)));
                        b.insert("sampling".into(), serde_json::json!(sampling));
                        if let Some(nd) = cog.index.nodata_f64.or_else(|| cog.index.nodata.map(|v| v as f64)) {
                            if nd.is_nan() {
                                b.insert("nodata".into(), serde_json::json!("nan"));
                            } else {
                                b.insert("nodata".into(), serde_json::json!(nd));
                            }
                        }
                        if let Some(bs) = stats_json.as_ref().and_then(|s| s.get(format!("b{}", i + 1))) {
                            b.insert(
                                "statistics".into(),
                                serde_json::json!({
                                    "mean": bs["mean"],
                                    "minimum": bs["min"],
                                    "maximum": bs["max"],
                                    "stddev": bs["std"],
                                    "valid_percent": bs["valid_percent"],
                                }),
                            );
                            if let Some(h) = bs.get("histogram").and_then(|h| h.as_array())
                                && h.len() == 2
                                && let (Some(counts), Some(edges)) = (h[0].as_array(), h[1].as_array())
                                && edges.len() >= 2
                            {
                                b.insert(
                                    "histogram".into(),
                                    serde_json::json!({
                                        "count": counts.len() + 1,
                                        "min": edges.first(),
                                        "max": edges.last(),
                                        "buckets": counts,
                                    }),
                                );
                            }
                        }
                        serde_json::Value::Object(b)
                    })
                    .collect();
                asset_obj["raster:bands"] = serde_json::Value::Array(rb);
            }
            if with_eo {
                push_ext("https://stac-extensions.github.io/eo/v1.1.0/schema.json");
                let eb: Vec<serde_json::Value> = (0..count)
                    .map(|i| {
                        let name = format!("b{}", i + 1);
                        let desc = cog
                            .index
                            .band_descriptions
                            .get(i)
                            .filter(|s| !s.is_empty())
                            .cloned()
                            .unwrap_or_else(|| name.clone());
                        serde_json::json!({ "name": name, "description": desc })
                    })
                    .collect();
                asset_obj["eo:bands"] = serde_json::Value::Array(eb);
            }
            for ext in &q_extensions {
                push_ext(ext);
            }
            let mut links: Vec<serde_json::Value> = Vec::new();
            if let Some(col_url) = q_collection_url {
                links.push(serde_json::json!({
                    "rel": "collection",
                    "type": "application/json",
                    "href": col_url,
                }));
            }
            let mut assets_map = serde_json::Map::new();
            assets_map.insert(asset_name, asset_obj);
            let mut item = serde_json::json!({
                "type": "Feature",
                "stac_version": "1.1.0",
                "stac_extensions": extensions,
                "id": item_id,
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]
                },
                "bbox": wgs_bbox,
                "properties": serde_json::Value::Object(props),
                "links": links,
                "assets": serde_json::Value::Object(assets_map),
            });
            if let Some(col) = q_collection {
                item["collection"] = serde_json::json!(col);
            }
            Ok((Bytes::from(item.to_string()), "application/geo+json"))
        }
        .await,
        // GET /cog/map (OGC API - Maps)
        (Method::GET, ["map"]) => async {
            let cogs = open_query_urls(&engine, &pq.urls, pq.nodata).await?;
            let (native_crs, native_epsg, native_bounds) =
                dataset_native_crs_and_bounds(&engine, &cogs[0])?;
            let full = cogs[0].index.full_res();
            let ext = pq.format_param.as_deref().unwrap_or("png");
            let (fmt, ct) = parse_ext_format(ext, pq.webp_quality, pq.jpeg_quality)?;
            let explicit_crs = pq.dst_crs.clone().or_else(|| pq.bounds_crs.clone());
            let (dst_crs, dst_epsg, used_native_crs) = match explicit_crs {
                Some((c, e)) => (c, Some(e), native_epsg == Some(e)),
                None if ct == GEOTIFF_CONTENT_TYPE && native_epsg.is_none() => {
                    // User-defined projected COGs have no standard EPSG code for GeoTIFF output;
                    // reproject to WGS84 rather than tagging metre coordinates as EPSG:4326.
                    (cogzilla_proj::CrsModel::Geographic, Some(4326), false)
                }
                None => (native_crs.clone(), native_epsg, true),
            };
            let (wgs_bbox, out_bbox) = if let Some(raw_bbox) = pq.bbox_param {
                let raw_bbox =
                    resolve_input_bbox(raw_bbox, pq.bbox_crs.as_ref().map(|(c, _)| c))?;
                let wb = bbox_to_wgs84(
                    raw_bbox,
                    pq.bbox_crs.as_ref().map(|(c, _)| c),
                )?;
                let ob = if (dst_epsg.is_some()
                    && pq.bbox_crs.as_ref().map(|(_, e)| *e) == dst_epsg)
                    || (pq.bbox_crs.is_none() && dst_epsg == Some(4326))
                {
                    normalize_render_bbox(raw_bbox, &dst_crs)?
                } else {
                    normalize_render_bbox(reproject_wgs84_bounds(wb, &dst_crs)?, &dst_crs)?
                };
                (wb, ob)
            } else {
                let wb = native_crs.wgs84_bounds(
                    &full.pixel_to_crs,
                    full.gcp_poly.as_deref(),
                    full.width,
                    full.height,
                );
                let ob = if !used_native_crs {
                    normalize_render_bbox(reproject_wgs84_bounds(wb, &dst_crs)?, &dst_crs)?
                } else {
                    native_bounds
                };
                (wb, ob)
            };
            let (out_w, out_h) = if used_native_crs {
                let px_x = full.pixel_to_crs.a.hypot(full.pixel_to_crs.d).max(1e-12);
                let px_y = full.pixel_to_crs.b.hypot(full.pixel_to_crs.e).max(1e-12);
                let nat_w_f = (out_bbox[2] - out_bbox[0]).abs() / px_x;
                let nat_h_f = (out_bbox[3] - out_bbox[1]).abs() / px_y;
                if pq.width.is_some() || pq.height.is_some() {
                    resolve_query_dims_f64(nat_w_f, nat_h_f, pq.width, pq.height)
                } else {
                    let nat_w = nat_w_f.round().clamp(1.0, 4096.0) as u32;
                    let nat_h = nat_h_f.round().clamp(1.0, 4096.0) as u32;
                    let cap = pq.max_size.clamp(1, 4096);
                    if nat_w.max(nat_h) <= cap {
                        (nat_w, nat_h)
                    } else if nat_w >= nat_h {
                        (
                            cap,
                            ((nat_h as f64 / nat_w as f64) * cap as f64)
                                .ceil()
                                .clamp(1.0, cap as f64) as u32,
                        )
                    } else {
                        (
                            ((nat_w as f64 / nat_h as f64) * cap as f64)
                                .ceil()
                                .clamp(1.0, cap as f64) as u32,
                            cap,
                        )
                    }
                }
            } else {
                resolve_bbox_query_dims(
                    &engine,
                    &cogs[0],
                    &dst_crs,
                    wgs_bbox,
                    out_bbox,
                    pq.max_size,
                    pq.width,
                    pq.height,
                )
            };
            let db = if cogs.len() > 1 { cogs.len() as u16 } else { data_band_count(full) };
            let st = widest_sample_type(&cogs);
            let mut bands = build_band_request_for_output(&pq, db, st, fmt, ct)?;
            let fmt = render_output_format(fmt, ct, &pq, Some(&cogs[0]), &bands, st, cogs.len() == 1);
            if pq.unscale {
                apply_unscale_cogs(&mut bands, &cogs, None);
            }
            let custom_lut = resolve_custom_lut(&pq, &cogs[0], &bands)?;
            let cf = pq.parse_color_formula()?;
            let resample = resolve_resample(Some(&engine), &pq, Some(&cogs[0]), dst_epsg.unwrap_or(0));
            let data = render_crs_bbox(
                &engine, &cogs, out_bbox, &dst_crs, out_w, out_h, None, &bands, resample, fmt,
                pq.nodata, custom_lut.as_ref(), cf.as_ref(), pq.return_mask, pq.unscale,
            )
            .await?;
            let data = wrap_geotiff_fmt(
                data,
                ct,
                fmt,
                out_w,
                out_h,
                out_bbox,
                dst_epsg.unwrap_or(4326),
                pq.return_mask,
            );
            geo_meta = Some((out_bbox, dst_epsg, false));
            Ok((data, ct))
        }
        .await,
        (Method::GET, ["wms"]) => {
            handle_cog_wms(&engine, host, raw_query, &pq, &mut geo_meta).await
        }
        (Method::GET, [_, "WMTSCapabilities.xml"]) | (Method::GET, ["WMTSCapabilities.xml"]) => {
            async {
                let explicit_tms = (sub.len() == 2).then_some(sub[0]);
                if let Some(tms) = explicit_tms
                    && crate::tms::lookup_tms(tms).is_none()
                {
                    return Err(EngineError::NotFound(format!("unknown TileMatrixSet '{tms}'")));
                }
                let cogs = open_query_urls(&engine, &pq.urls, None).await?;
                let crs = engine.get_or_build_crs(&cogs[0])?;
                let full = cogs[0].index.full_res();
                let bounds = crs.wgs84_bounds(
                &full.pixel_to_crs,
                full.gcp_poly.as_deref(),
                full.width,
                full.height,
            );
                let coarsest = cogs[0].index.ifds.last().unwrap();
                let ov_factor = (full.width as f64 / coarsest.width.max(1) as f64)
                    .max(full.height as f64 / coarsest.height.max(1) as f64);
                let fmt = pq.tile_format.as_deref().unwrap_or("png");
                let scale = pq.tile_scale.unwrap_or(1);
                let tile_q = filter_tile_template_query(raw_query);
                let xml = if let Some(tms) = explicit_tms {
                    let (def_min, def_max) = if let Some(tms_def) = crate::tms::lookup_tms(tms) {
                        tms_def.zoom_range_for_cog(&crs, &full.pixel_to_crs, full.width, full.height, ov_factor)
                    } else {
                        crs.web_mercator_zoom_range(&full.pixel_to_crs, full.width, full.height, ov_factor)
                    };
                    let (minz, maxz) =
                        resolve_zoom_range(def_min, def_max, pq.minzoom, pq.maxzoom);
                    build_wmts_capabilities(
                        host,
                        "cog",
                        tms,
                        "Cloud Optimized GeoTIFF",
                        bounds,
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
                        .map(|t| {
                            let (dmin, dmax) = t.zoom_range_for_cog(
                                &crs,
                                &full.pixel_to_crs,
                                full.width,
                                full.height,
                                ov_factor,
                            );
                            let (z0, z1) = resolve_zoom_range(dmin, dmax, pq.minzoom, pq.maxzoom);
                            (t.id, z0, z1)
                        })
                        .collect();
                    build_wmts_capabilities_multi(
                        host,
                        "cog",
                        &entries,
                        None,
                        "Cloud Optimized GeoTIFF",
                        bounds,
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

        // 10. GET /cog/point/{lon},{lat} (supports coord_crs=EPSG:3857 and multi-URL stacking)
        (Method::GET, ["point", coord]) => async {
            let cogs = open_query_urls(&engine, &pq.urls, pq.nodata).await?;
            let Some((x, y)) = parse_point_coord(coord) else {
                return Err(EngineError::BadRequest("expected /cog/point/{lon},{lat}".into()));
            };
            let (lon, lat) = coord_to_lonlat(x, y, pq.coord_crs.as_ref())?;
            let mut json = if cogs.len() == 1 {
                query_point(&engine, &cogs[0], lon, lat, pq.unscale, pq.nodata).await?
            } else {
                let n = cogs.len();
                let mut set = tokio::task::JoinSet::new();
                for (idx, cog) in cogs.into_iter().enumerate() {
                    let eng = engine.clone();
                    let unscale = pq.unscale;
                    let nodata = pq.nodata;
                    set.spawn(async move {
                        let res = query_point(&eng, &cog, lon, lat, unscale, nodata).await;
                        (idx, res)
                    });
                }
                let mut vals = vec![serde_json::Value::Null; n];
                let mut any_ok = false;
                let mut first_oob: Option<EngineError> = None;
                while let Some(joined) = set.join_next().await {
                    let (idx, res) = joined
                        .map_err(|e| EngineError::BadRequest(format!("join error: {e}")))?;
                    match res {
                        Ok(pt) => {
                            any_ok = true;
                            if let Some(v) = pt["values"].as_array().and_then(|a| a.first()) {
                                vals[idx] = v.clone();
                            }
                        }
                        Err(e @ EngineError::NotFound(_)) => {
                            if first_oob.is_none() {
                                first_oob = Some(e);
                            }
                        }
                        Err(e) => return Err(e),
                    }
                }
                if !any_ok {
                    return Err(first_oob.unwrap_or_else(|| {
                        EngineError::NotFound(format!(
                            "point ({lon}, {lat}) is outside dataset bounds"
                        ))
                    }));
                }
                let names: Vec<String> = (1..=n).map(|i| format!("b{i}")).collect();
                serde_json::json!({
                    "coordinates": [lon, lat],
                    "values": vals,
                    "band_names": names,
                    "band_descriptions": names,
                })
            };
            if pq.coord_crs.is_some() {
                json["coordinates"] = serde_json::json!([x, y]);
            }
            apply_point_filters(&mut json, pq.expr_raw.as_deref(), &pq.bidx)?;
            Ok((Bytes::from(json.to_string()), "application/json"))
        }
        .await,

        // 11. GET /cog/preview, /cog/preview.{format}, /cog/preview/{w}x{h}.{format}
        (Method::GET, [p]) if *p == "preview" || p.starts_with("preview.") => async {
            let ext_opt = p.strip_prefix("preview.");
            let auto_fmt = ext_opt.is_none() && pq.format_param.is_none();
            let ext = ext_opt
                .or(pq.format_param.as_deref())
                .unwrap_or("png");
            let cogs = open_query_urls(&engine, &pq.urls, pq.nodata).await?;
            let full = cogs[0].index.full_res();
            let is_georef = cogs[0].index.is_georeferenced;
            if !is_georef && pq.dst_crs.is_some() {
                let _ = engine.get_or_build_crs(&cogs[0])?;
            }
            let (crs_opt, bbox) = if is_georef {
                let crs = engine.get_or_build_crs(&cogs[0])?;
                let b = crs.wgs84_bounds(
                    &full.pixel_to_crs,
                    full.gcp_poly.as_deref(),
                    full.width,
                    full.height,
                );
                (Some(crs), b)
            } else {
                (None, [0.0, 0.0, full.width as f64, full.height as f64])
            };
            let db = if cogs.len() > 1 { cogs.len() as u16 } else { data_band_count(full) };
            let (fmt, ct) = parse_ext_format(ext, pq.webp_quality, pq.jpeg_quality)?;
            let st = widest_sample_type(&cogs);
            let mut bands = build_band_request_for_output(&pq, db, st, fmt, ct)?;
            let fmt = if auto_fmt {
                OutputFormat::RawRgba
            } else {
                render_output_format(fmt, ct, &pq, Some(&cogs[0]), &bands, st, cogs.len() == 1)
            };
            if pq.unscale {
                apply_unscale_cogs(&mut bands, &cogs, None);
            }
            let custom_lut = resolve_custom_lut(&pq, &cogs[0], &bands)?;
            let cf = pq.parse_color_formula()?;
            let (data, out_bbox, out_w, out_h, out_epsg) = if let Some((ref dst_crs, dst_epsg)) = pq.dst_crs {
                let dst_bbox = reproject_wgs84_bounds(bbox, dst_crs)?;
                let (out_w, out_h) = resolve_bbox_query_dims(&engine, &cogs[0], dst_crs, bbox, dst_bbox, pq.max_size, pq.width, pq.height);
                let rs = resolve_resample(Some(&engine), &pq, Some(&cogs[0]), dst_epsg);
                let d = render_crs_bbox(
                    &engine, &cogs, dst_bbox, dst_crs, out_w, out_h, None, &bands,
                    rs, fmt, pq.nodata, custom_lut.as_ref(), cf.as_ref(),
                    pq.return_mask, pq.unscale,
                )
                .await?;
                (d, dst_bbox, out_w, out_h, dst_epsg)
            } else {
                let scale = (pq.max_size as f64) / (full.width.max(full.height) as f64);
                let def_w = ((full.width as f64 * scale.min(1.0)).ceil() as u32).max(1);
                let def_h = ((full.height as f64 * scale.min(1.0)).ceil() as u32).max(1);
                let (out_w, out_h) = resolve_query_dims(def_w, def_h, pq.width, pq.height);
                let (nat_epsg, nat_bbox) = cog_native_epsg_and_bounds(&cogs[0]);
                let north_up_aligned = full.pixel_to_crs.is_axis_aligned()
                    && full.pixel_to_crs.a > 0.0
                    && full.pixel_to_crs.e < 0.0;
                if !is_georef {
                    if ct == GEOTIFF_CONTENT_TYPE {
                        return Err(EngineError::BadRequest(
                            "GeoTIFF output requires a georeferenced dataset; use .png, .jpg, .webp, or .npy".into(),
                        ));
                    }
                    let d = render_native_preview(
                        &engine, &cogs, bbox, out_w, out_h, &bands, pq.resample, fmt,
                        pq.nodata, custom_lut.as_ref(), cf.as_ref(), pq.return_mask, pq.unscale,
                    )
                    .await?;
                    (d, bbox, out_w, out_h, 4326)
                } else if ct == GEOTIFF_CONTENT_TYPE && nat_epsg.is_none() {
                    let rs = resolve_resample(Some(&engine), &pq, Some(&cogs[0]), 4326);
                    let d = render_wgs84_bbox(
                        &engine, &cogs, bbox, out_w, out_h, None, &bands,
                        rs, fmt, pq.nodata, custom_lut.as_ref(), cf.as_ref(),
                        pq.return_mask, pq.unscale,
                    )
                    .await?;
                    (d, bbox, out_w, out_h, 4326)
                } else if ct == GEOTIFF_CONTENT_TYPE && !north_up_aligned {
                    let epsg = nat_epsg.unwrap_or(4326);
                    let d = render_crs_bbox(
                        &engine, &cogs, nat_bbox, crs_opt.as_ref().unwrap(), out_w, out_h, None, &bands,
                        pq.resample, fmt, pq.nodata, custom_lut.as_ref(), cf.as_ref(),
                        pq.return_mask, pq.unscale,
                    )
                    .await?;
                    (d, nat_bbox, out_w, out_h, epsg)
                } else {
                    let d = render_native_preview(
                        &engine, &cogs, bbox, out_w, out_h, &bands, pq.resample, fmt,
                        pq.nodata, custom_lut.as_ref(), cf.as_ref(), pq.return_mask, pq.unscale,
                    )
                    .await?;
                    (d, nat_bbox, out_w, out_h, nat_epsg.unwrap_or(4326))
                }
            };
            let (data, ct) = finalize_image_output(data, ct, fmt, auto_fmt, out_w, out_h, out_bbox, out_epsg, pq.jpeg_quality, pq.return_mask);
            if is_georef {
                geo_meta = Some((out_bbox, Some(out_epsg), false));
            }
            Ok((data, ct))
        }
        .await,
        (Method::GET, ["preview", dims_ext]) => async {
            let (out_w, out_h, ext_opt) = parse_wxh_ext(dims_ext).map_err(EngineError::BadRequest)?;
            let cogs = open_query_urls(&engine, &pq.urls, pq.nodata).await?;
            let full = cogs[0].index.full_res();
            let is_georef = cogs[0].index.is_georeferenced;
            if !is_georef && pq.dst_crs.is_some() {
                let _ = engine.get_or_build_crs(&cogs[0])?;
            }
            let (crs_opt, bbox) = if is_georef {
                let crs = engine.get_or_build_crs(&cogs[0])?;
                let b = crs.wgs84_bounds(
                    &full.pixel_to_crs,
                    full.gcp_poly.as_deref(),
                    full.width,
                    full.height,
                );
                (Some(crs), b)
            } else {
                (None, [0.0, 0.0, full.width as f64, full.height as f64])
            };
            let db = if cogs.len() > 1 { cogs.len() as u16 } else { data_band_count(full) };
            let ext = ext_opt.or(pq.format_param.as_deref()).unwrap_or("png");
            let (fmt, ct) = parse_ext_format(ext, pq.webp_quality, pq.jpeg_quality)?;
            let st = widest_sample_type(&cogs);
            let mut bands = build_band_request_for_output(&pq, db, st, fmt, ct)?;
            let fmt = render_output_format(fmt, ct, &pq, Some(&cogs[0]), &bands, st, cogs.len() == 1);
            if pq.unscale {
                apply_unscale_cogs(&mut bands, &cogs, None);
            }
            let custom_lut = resolve_custom_lut(&pq, &cogs[0], &bands)?;
            let cf = pq.parse_color_formula()?;
            let (data, out_bbox, out_epsg) = if let Some((ref dst_crs, dst_epsg)) = pq.dst_crs {
                let dst_bbox = reproject_wgs84_bounds(bbox, dst_crs)?;
                let rs = resolve_resample(Some(&engine), &pq, Some(&cogs[0]), dst_epsg);
                let d = render_crs_bbox(
                    &engine, &cogs, dst_bbox, dst_crs, out_w, out_h, None, &bands,
                    rs, fmt, pq.nodata, custom_lut.as_ref(), cf.as_ref(),
                    pq.return_mask, pq.unscale,
                )
                .await?;
                (d, dst_bbox, dst_epsg)
            } else {
                let (nat_epsg, nat_bbox) = cog_native_epsg_and_bounds(&cogs[0]);
                let north_up_aligned = full.pixel_to_crs.is_axis_aligned()
                    && full.pixel_to_crs.a > 0.0
                    && full.pixel_to_crs.e < 0.0;
                if !is_georef {
                    if ct == GEOTIFF_CONTENT_TYPE {
                        return Err(EngineError::BadRequest(
                            "GeoTIFF output requires a georeferenced dataset; use .png, .jpg, .webp, or .npy".into(),
                        ));
                    }
                    let d = render_native_preview(
                        &engine, &cogs, bbox, out_w, out_h, &bands, pq.resample, fmt,
                        pq.nodata, custom_lut.as_ref(), cf.as_ref(), pq.return_mask, pq.unscale,
                    )
                    .await?;
                    (d, bbox, 4326)
                } else if ct == GEOTIFF_CONTENT_TYPE && nat_epsg.is_none() {
                    let rs = resolve_resample(Some(&engine), &pq, Some(&cogs[0]), 4326);
                    let d = render_wgs84_bbox(
                        &engine, &cogs, bbox, out_w, out_h, None, &bands,
                        rs, fmt, pq.nodata, custom_lut.as_ref(), cf.as_ref(),
                        pq.return_mask, pq.unscale,
                    )
                    .await?;
                    (d, bbox, 4326)
                } else if ct == GEOTIFF_CONTENT_TYPE && !north_up_aligned {
                    let epsg = nat_epsg.unwrap_or(4326);
                    let d = render_crs_bbox(
                        &engine, &cogs, nat_bbox, crs_opt.as_ref().unwrap(), out_w, out_h, None, &bands,
                        pq.resample, fmt, pq.nodata, custom_lut.as_ref(), cf.as_ref(),
                        pq.return_mask, pq.unscale,
                    )
                    .await?;
                    (d, nat_bbox, epsg)
                } else {
                    let d = render_native_preview(
                        &engine, &cogs, bbox, out_w, out_h, &bands, pq.resample, fmt,
                        pq.nodata, custom_lut.as_ref(), cf.as_ref(), pq.return_mask, pq.unscale,
                    )
                    .await?;
                    (d, nat_bbox, nat_epsg.unwrap_or(4326))
                }
            };
            let data = wrap_geotiff_fmt(data, ct, fmt, out_w, out_h, out_bbox, out_epsg, pq.return_mask);
            if is_georef {
                geo_meta = Some((out_bbox, Some(out_epsg), false));
            }
            Ok((data, ct))
        }
        .await,

        // 12. GET /cog/bbox/{minx},{miny},{maxx},{maxy}[.{format}] & /cog/bbox/{...}/{w}x{h}.{format}
        (Method::GET, ["bbox", bbox_seg]) => async {
            let Some((raw_bbox, ext_opt)) = parse_bbox4(bbox_seg) else {
                return Err(EngineError::BadRequest("expected /cog/bbox/{minx},{miny},{maxx},{maxy}.{ext}".into()));
            };
            let bbox = resolve_input_bbox(raw_bbox, pq.coord_crs.as_ref())?;
            let auto_fmt = ext_opt.is_none() && pq.format_param.is_none();
            let cogs = open_query_urls(&engine, &pq.urls, pq.nodata).await?;
            let full = cogs[0].index.full_res();
            let db = if cogs.len() > 1 { cogs.len() as u16 } else { data_band_count(full) };
            let (fmt, ct) = parse_ext_format(
                ext_opt.or(pq.format_param.as_deref()).unwrap_or("png"),
                pq.webp_quality,
                pq.jpeg_quality,
            )?;
            let st = widest_sample_type(&cogs);
            let mut bands = build_band_request_for_output(&pq, db, st, fmt, ct)?;
            let fmt = if auto_fmt {
                OutputFormat::RawRgba
            } else {
                render_output_format(fmt, ct, &pq, Some(&cogs[0]), &bands, st, cogs.len() == 1)
            };
            if pq.unscale {
                apply_unscale_cogs(&mut bands, &cogs, None);
            }
            let custom_lut = resolve_custom_lut(&pq, &cogs[0], &bands)?;
            let cf = pq.parse_color_formula()?;
            let wgs_bbox_src = bbox_to_wgs84(bbox, pq.coord_crs.as_ref())?;
            let bbox_max = effective_bbox_max_size(&pq);
            let (data, out_bbox, out_w, out_h, epsg) = if let Some((dst_crs, dst_epsg)) = resolve_bbox_dst_crs(&pq).filter(|(_, e)| *e != 4326 && *e != 3857) {
                let dst_bbox = if pq.coord_crs_epsg == Some(dst_epsg) {
                    bbox
                } else {
                    reproject_wgs84_bounds(wgs_bbox_src, &dst_crs)?
                };
                let (out_w, out_h) = resolve_bbox_query_dims(&engine, &cogs[0], &dst_crs, wgs_bbox_src, dst_bbox, bbox_max, pq.width, pq.height);
                let resample = resolve_resample(Some(&engine), &pq, Some(&cogs[0]), dst_epsg);
                let d = render_crs_bbox(
                    &engine, &cogs, dst_bbox, &dst_crs, out_w, out_h, None, &bands, resample, fmt,
                    pq.nodata, custom_lut.as_ref(), cf.as_ref(), pq.return_mask, pq.unscale,
                )
                .await?;
                (d, dst_bbox, out_w, out_h, dst_epsg)
            } else if let Some((ref dst_crs, dst_epsg)) = pq.dst_crs {
                let dst_bbox = if pq.coord_crs_epsg == Some(dst_epsg) {
                    bbox
                } else {
                    reproject_wgs84_bounds(wgs_bbox_src, dst_crs)?
                };
                let (out_w, out_h) = resolve_bbox_query_dims(&engine, &cogs[0], dst_crs, wgs_bbox_src, dst_bbox, bbox_max, pq.width, pq.height);
                let resample = resolve_resample(Some(&engine), &pq, Some(&cogs[0]), dst_epsg);
                let d = render_crs_bbox(
                    &engine, &cogs, dst_bbox, dst_crs, out_w, out_h, None, &bands, resample, fmt,
                    pq.nodata, custom_lut.as_ref(), cf.as_ref(), pq.return_mask, pq.unscale,
                )
                .await?;
                (d, dst_bbox, out_w, out_h, dst_epsg)
            } else if pq.coord_crs_3857 {
                let (out_w, out_h) = resolve_bbox_query_dims(&engine, &cogs[0], &cogzilla_proj::CrsModel::Epsg3857, wgs_bbox_src, bbox, bbox_max, pq.width, pq.height);
                let resample = resolve_resample(Some(&engine), &pq, Some(&cogs[0]), 3857);
                let d = render_mercator_bbox(
                    &engine, &cogs, bbox, out_w, out_h, &bands, resample, fmt,
                    pq.nodata, custom_lut.as_ref(), cf.as_ref(), pq.return_mask, pq.unscale,
                )
                .await?;
                (d, bbox, out_w, out_h, 3857u16)
            } else {
                let (out_w, out_h) = resolve_bbox_query_dims(&engine, &cogs[0], &cogzilla_proj::CrsModel::Geographic, wgs_bbox_src, wgs_bbox_src, bbox_max, pq.width, pq.height);
                let resample = resolve_resample(Some(&engine), &pq, Some(&cogs[0]), 4326);
                let d = render_wgs84_bbox(
                    &engine, &cogs, wgs_bbox_src, out_w, out_h, None, &bands, resample, fmt,
                    pq.nodata, custom_lut.as_ref(), cf.as_ref(), pq.return_mask, pq.unscale,
                )
                .await?;
                (d, wgs_bbox_src, out_w, out_h, 4326u16)
            };
            let (data, ct) = finalize_image_output(data, ct, fmt, auto_fmt, out_w, out_h, out_bbox, epsg, pq.jpeg_quality, pq.return_mask);
            geo_meta = Some((out_bbox, Some(epsg), false));
            Ok((data, ct))
        }
        .await,
        (Method::GET, ["bbox", bbox_seg, dims_ext]) => async {
            let Some((raw_bbox, _)) = parse_bbox4(bbox_seg) else {
                return Err(EngineError::BadRequest("invalid bbox coordinates".into()));
            };
            let bbox = resolve_input_bbox(raw_bbox, pq.coord_crs.as_ref())?;
            let (out_w, out_h, ext_opt) = parse_wxh_ext(dims_ext).map_err(EngineError::BadRequest)?;
            let cogs = open_query_urls(&engine, &pq.urls, pq.nodata).await?;
            let full = cogs[0].index.full_res();
            let db = if cogs.len() > 1 { cogs.len() as u16 } else { data_band_count(full) };
            let ext = ext_opt.or(pq.format_param.as_deref()).unwrap_or("png");
            let (fmt, ct) = parse_ext_format(ext, pq.webp_quality, pq.jpeg_quality)?;
            let st = widest_sample_type(&cogs);
            let mut bands = build_band_request_for_output(&pq, db, st, fmt, ct)?;
            let fmt = render_output_format(fmt, ct, &pq, Some(&cogs[0]), &bands, st, cogs.len() == 1);
            if pq.unscale {
                apply_unscale_cogs(&mut bands, &cogs, None);
            }
            let custom_lut = resolve_custom_lut(&pq, &cogs[0], &bands)?;
            let cf = pq.parse_color_formula()?;
            let (data, out_bbox, epsg) = if let Some((dst_crs, dst_epsg)) = resolve_bbox_dst_crs(&pq).filter(|(_, e)| *e != 4326 && *e != 3857) {
                let dst_bbox = if pq.coord_crs_epsg == Some(dst_epsg) {
                    bbox
                } else {
                    let wgs_bbox = bbox_to_wgs84(bbox, pq.coord_crs.as_ref())?;
                    reproject_wgs84_bounds(wgs_bbox, &dst_crs)?
                };
                let resample = resolve_resample(Some(&engine), &pq, Some(&cogs[0]), dst_epsg);
                let d = render_crs_bbox(
                    &engine, &cogs, dst_bbox, &dst_crs, out_w, out_h, None, &bands, resample, fmt,
                    pq.nodata, custom_lut.as_ref(), cf.as_ref(), pq.return_mask, pq.unscale,
                )
                .await?;
                (d, dst_bbox, dst_epsg)
            } else if let Some((ref dst_crs, dst_epsg)) = pq.dst_crs {
                let dst_bbox = if pq.coord_crs_epsg == Some(dst_epsg) {
                    bbox
                } else {
                    let wgs_bbox = bbox_to_wgs84(bbox, pq.coord_crs.as_ref())?;
                    reproject_wgs84_bounds(wgs_bbox, dst_crs)?
                };
                let resample = resolve_resample(Some(&engine), &pq, Some(&cogs[0]), dst_epsg);
                let d = render_crs_bbox(
                    &engine, &cogs, dst_bbox, dst_crs, out_w, out_h, None, &bands, resample, fmt,
                    pq.nodata, custom_lut.as_ref(), cf.as_ref(), pq.return_mask, pq.unscale,
                )
                .await?;
                (d, dst_bbox, dst_epsg)
            } else if pq.coord_crs_3857 {
                let resample = resolve_resample(Some(&engine), &pq, Some(&cogs[0]), 3857);
                let d = render_mercator_bbox(
                    &engine, &cogs, bbox, out_w, out_h, &bands, resample, fmt,
                    pq.nodata, custom_lut.as_ref(), cf.as_ref(), pq.return_mask, pq.unscale,
                )
                .await?;
                (d, bbox, 3857u16)
            } else {
                let wgs_bbox = bbox_to_wgs84(bbox, pq.coord_crs.as_ref())?;
                let resample = resolve_resample(Some(&engine), &pq, Some(&cogs[0]), 4326);
                let d = render_wgs84_bbox(
                    &engine, &cogs, wgs_bbox, out_w, out_h, None, &bands, resample, fmt,
                    pq.nodata, custom_lut.as_ref(), cf.as_ref(), pq.return_mask, pq.unscale,
                )
                .await?;
                (d, wgs_bbox, 4326u16)
            };
            let data = wrap_geotiff_fmt(data, ct, fmt, out_w, out_h, out_bbox, epsg, pq.return_mask);
            geo_meta = Some((out_bbox, Some(epsg), false));
            Ok((data, ct))
        }
        .await,

        // 13. POST /cog/feature, /cog/feature.{format}, /cog/feature/{w}x{h}.{format}
        (Method::POST, [f]) if *f == "feature" || f.starts_with("feature.") => async {
            let ext_opt = f.strip_prefix("feature.");
            let auto_fmt = ext_opt.is_none() && pq.format_param.is_none();
            let ext = ext_opt
                .or(pq.format_param.as_deref())
                .unwrap_or("png");
            let body_bytes = read_limited_body(req.into_body()).await?;
            let geojson: serde_json::Value = serde_json::from_slice(&body_bytes)
                .map_err(|e| EngineError::BadRequest(format!("invalid GeoJSON: {e}")))?;
            let (mask, bbox) = PolygonMask::from_geojson_in_crs(&geojson, pq.coord_crs.as_ref())?;
            let cogs = open_query_urls(&engine, &pq.urls, pq.nodata).await?;
            let full = cogs[0].index.full_res();
            let db = if cogs.len() > 1 { cogs.len() as u16 } else { data_band_count(full) };
            let (fmt, ct) = parse_ext_format(ext, pq.webp_quality, pq.jpeg_quality)?;
            let st = widest_sample_type(&cogs);
            let mut bands = build_band_request_for_output(&pq, db, st, fmt, ct)?;
            let fmt = if auto_fmt {
                OutputFormat::RawRgba
            } else {
                render_output_format(fmt, ct, &pq, Some(&cogs[0]), &bands, st, cogs.len() == 1)
            };
            if pq.unscale {
                apply_unscale_cogs(&mut bands, &cogs, None);
            }
            let custom_lut = resolve_custom_lut(&pq, &cogs[0], &bands)?;
            let cf = pq.parse_color_formula()?;
            let eff_dst = resolve_bbox_dst_crs(&pq);
            let target_epsg = eff_dst.as_ref().map(|(_, e)| *e).unwrap_or(4326);
            let resample = resolve_resample(Some(&engine), &pq, Some(&cogs[0]), target_epsg);
            let mask_opt = pq.shape_mask.then_some(&mask);
            let feat_max = effective_bbox_max_size(&pq);
            let (data, out_bbox, out_w, out_h, out_epsg) = if let Some((ref dst_crs, dst_epsg)) = eff_dst {
                let dst_bbox = if pq.coord_crs_epsg == Some(dst_epsg) {
                    mask.raw_bbox
                } else if let Some(b) = mask.bbox_in_crs(dst_crs) {
                    b
                } else {
                    reproject_wgs84_bounds(bbox, dst_crs)?
                };
                let (out_w, out_h) = resolve_bbox_query_dims(&engine, &cogs[0], dst_crs, bbox, dst_bbox, feat_max, pq.width, pq.height);
                let d = render_crs_bbox(
                    &engine, &cogs, dst_bbox, dst_crs, out_w, out_h, mask_opt, &bands, resample, fmt,
                    pq.nodata, custom_lut.as_ref(), cf.as_ref(), pq.return_mask, pq.unscale,
                )
                .await?;
                (d, dst_bbox, out_w, out_h, dst_epsg)
            } else {
                let (out_w, out_h) = resolve_bbox_query_dims(&engine, &cogs[0], &cogzilla_proj::CrsModel::Geographic, bbox, bbox, feat_max, pq.width, pq.height);
                let d = render_wgs84_bbox(
                    &engine, &cogs, bbox, out_w, out_h, mask_opt, &bands, resample, fmt,
                    pq.nodata, custom_lut.as_ref(), cf.as_ref(), pq.return_mask, pq.unscale,
                )
                .await?;
                (d, bbox, out_w, out_h, 4326u16)
            };
            let (data, ct) = finalize_image_output(data, ct, fmt, auto_fmt, out_w, out_h, out_bbox, out_epsg, pq.jpeg_quality, pq.return_mask);
            geo_meta = Some((out_bbox, Some(out_epsg), false));
            Ok((data, ct))
        }
        .await,
        (Method::POST, ["feature", dims_ext]) => async {
            let (out_w, out_h, ext_opt) = parse_wxh_ext(dims_ext).map_err(EngineError::BadRequest)?;
            let body_bytes = read_limited_body(req.into_body()).await?;
            let geojson: serde_json::Value = serde_json::from_slice(&body_bytes)
                .map_err(|e| EngineError::BadRequest(format!("invalid GeoJSON: {e}")))?;
            let (mask, bbox) = PolygonMask::from_geojson_in_crs(&geojson, pq.coord_crs.as_ref())?;
            let cogs = open_query_urls(&engine, &pq.urls, pq.nodata).await?;
            let full = cogs[0].index.full_res();
            let db = if cogs.len() > 1 { cogs.len() as u16 } else { data_band_count(full) };
            let ext = ext_opt.or(pq.format_param.as_deref()).unwrap_or("png");
            let (fmt, ct) = parse_ext_format(ext, pq.webp_quality, pq.jpeg_quality)?;
            let st = widest_sample_type(&cogs);
            let mut bands = build_band_request_for_output(&pq, db, st, fmt, ct)?;
            let fmt = render_output_format(fmt, ct, &pq, Some(&cogs[0]), &bands, st, cogs.len() == 1);
            if pq.unscale {
                apply_unscale_cogs(&mut bands, &cogs, None);
            }
            let custom_lut = resolve_custom_lut(&pq, &cogs[0], &bands)?;
            let cf = pq.parse_color_formula()?;
            let eff_dst = resolve_bbox_dst_crs(&pq);
            let target_epsg = eff_dst.as_ref().map(|(_, e)| *e).unwrap_or(4326);
            let resample = resolve_resample(Some(&engine), &pq, Some(&cogs[0]), target_epsg);
            let mask_opt = pq.shape_mask.then_some(&mask);
            let (data, out_bbox, out_epsg) = if let Some((ref dst_crs, dst_epsg)) = eff_dst {
                let dst_bbox = if pq.coord_crs_epsg == Some(dst_epsg) {
                    mask.raw_bbox
                } else if let Some(b) = mask.bbox_in_crs(dst_crs) {
                    b
                } else {
                    reproject_wgs84_bounds(bbox, dst_crs)?
                };
                let d = render_crs_bbox(
                    &engine, &cogs, dst_bbox, dst_crs, out_w, out_h, mask_opt, &bands, resample, fmt,
                    pq.nodata, custom_lut.as_ref(), cf.as_ref(), pq.return_mask, pq.unscale,
                )
                .await?;
                (d, dst_bbox, dst_epsg)
            } else {
                let d = render_wgs84_bbox(
                    &engine, &cogs, bbox, out_w, out_h, mask_opt, &bands, resample, fmt,
                    pq.nodata, custom_lut.as_ref(), cf.as_ref(), pq.return_mask, pq.unscale,
                )
                .await?;
                (d, bbox, 4326u16)
            };
            let data = wrap_geotiff_fmt(data, ct, fmt, out_w, out_h, out_bbox, out_epsg, pq.return_mask);
            geo_meta = Some((out_bbox, Some(out_epsg), false));
            Ok((data, ct))
        }
        .await,

        (_, ["statistics"]) => Err(EngineError::MethodNotAllowed("GET, POST, HEAD")),
        (_, [f]) if *f == "feature" || f.starts_with("feature.") => {
            Err(EngineError::MethodNotAllowed("POST"))
        }
        (_, ["feature", _]) => Err(EngineError::MethodNotAllowed("POST")),
        (_, ["info"] | ["info.geojson"] | ["bounds"] | ["validate"] | ["stac"] | ["tiles"] | ["tiles", ..] | ["point", _] | ["preview", ..] | ["bbox", ..] | ["map"] | ["wms"] | ["viewer"] | ["tilejson.json"] | [_, "tilejson.json"] | ["map.html"] | [_, "map.html"] | ["WMTSCapabilities.xml"] | [_, "WMTSCapabilities.xml"]) => {
            Err(EngineError::MethodNotAllowed("GET, HEAD"))
        }

        _ => Err(EngineError::NotFound(path.to_string())),
    };
    let res = match res {
        Ok((data, ct)) => Ok((cog_array_in_its_type(&engine, cog_array, &cog_urls, data).await, ct)),
        Err(e) => Err(e),
    };
    let mut resp = to_http_response(&engine, res, if_none_match);
    if let Some((bbox, epsg, is_crs84)) = geo_meta {
        attach_geo_headers(&mut resp, bbox, epsg, is_crs84);
    }
    resp
}
