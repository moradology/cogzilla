//! `/stac/*` route family handler (extracted from `http.rs` for #419).

use super::*;

/// Extra context for `/stac/*` (and pgSTAC item routes forwarded through it).
pub(super) struct StacContext<'a> {
    pub(super) prefix: &'a str,
    pub(super) pgstac_item: Option<Arc<StacItem>>,
    pub(super) pgstac_item_extent: Option<pgstac::ItemExtent>,
}

pub(super) async fn handle<B>(
    ctx: RequestCtx<'_>,
    method: Method,
    sub: &[&str],
    pq: ParsedQuery,
    stac: StacContext<'_>,
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
    let StacContext {
        prefix: stac_prefix,
        mut pgstac_item,
        pgstac_item_extent,
    } = stac;
    let stac_x_assets = pq.assets.clone();
    let stac_url_hdr = pq.urls.first().cloned();
    // An item read from pgSTAC is held here; one named by `url` is in the cache by then.
    let stac_item_hdr = pgstac_item.clone();
    let stac_req_assets = pq.assets.clone();
    let stac_req_expr = pq.expr_raw.clone();
    let stac_array = array_asked(&pq, sub.last());
    let attach_resolved_stac_assets = |resp: &mut Response<Full<Bytes>>| {
        if !resp.status().is_success() {
            return;
        }
        let item = stac_item_hdr
            .clone()
            .or_else(|| stac_url_hdr.as_ref().and_then(|u| engine.stac_cache.get(u).map(|item| item.value().clone())));
        if let Some(item) = item
            && let Ok(names) = crate::stac::select_stac_asset_names(
                &item,
                &stac_req_assets,
                stac_req_expr.as_deref(),
            )
        {
            attach_x_assets(resp, &names);
        }
    };
    let mut geo_meta: Option<([f64; 4], Option<u16>, bool)> = None;
    let stac_source = match pgstac_item.take() {
        Some(item) => StacSource::Item(item),
        None => StacSource::Url(pq.urls.first().cloned()),
    };
    let res: Result<(Bytes, &'static str), EngineError> = match (method.clone(), sub) {
        // GET /stac/assets?url=item.json
        (Method::GET, ["assets"]) => async {
            let item = stac_source.load(&engine).await?;
            let keys: Vec<&String> = item.assets.keys().collect();
            Ok((Bytes::from(serde_json::json!(keys).to_string()), "application/json"))
        }
        .await,

        // GET /stac/bounds?url=item.json
        (Method::GET, ["bounds"]) => async {
            let item = stac_source.load(&engine).await?;
            let bounds = if let Some(b) = item.bbox {
                b
            } else {
                let (_, cogs, _) =
                    resolve_stac_assets(&engine, &item, &pq.assets, None, None).await?;
                let mut acc = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];
                for cog in &cogs {
                    let crs = engine.get_or_build_crs(cog)?;
                    let full = cog.index.full_res();
                    let b = crs.wgs84_bounds(&full.pixel_to_crs, full.gcp_poly.as_deref(), full.width, full.height);
                    acc[0] = acc[0].min(b[0]);
                    acc[1] = acc[1].min(b[1]);
                    acc[2] = acc[2].max(b[2]);
                    acc[3] = acc[3].max(b[3]);
                }
                if !acc.iter().all(|v| v.is_finite()) {
                    return Err(EngineError::BadRequest("STAC item has no valid bounds".into()));
                }
                acc
            };
            let bounds = if let Some((ref dst, _)) = pq.bounds_crs {
                reproject_wgs84_bounds(bounds, dst)?
            } else {
                bounds
            };
            let body = serde_json::json!({"bounds": bounds}).to_string();
            Ok((Bytes::from(body), "application/json"))
        }
        .await,

        // GET /stac/info?url=item.json&assets=...
        (Method::GET, ["info"]) => async {
            let item = stac_source.load(&engine).await?;
            let (names, cogs, _) =
                resolve_stac_assets(&engine, &item, &pq.assets, None, None).await?;
            let mut map = serde_json::Map::new();
            for (name, cog) in names.into_iter().zip(cogs) {
                let mut info = build_cog_info(&engine, &cog)?;
                if pq.bounds_crs.is_some() {
                    let crs = engine.get_or_build_crs(&cog)?;
                    let full = cog.index.full_res();
                    let wgs84 = crs.wgs84_bounds(&full.pixel_to_crs, full.gcp_poly.as_deref(), full.width, full.height);
                    apply_info_bounds_crs(&mut info, wgs84, pq.bounds_crs.as_ref())?;
                }
                map.insert(name, info);
            }
            Ok((Bytes::from(serde_json::Value::Object(map).to_string()), "application/json"))
        }
        .await,

        // GET /stac/info.geojson?url=item.json&assets=...
        (Method::GET, ["info.geojson"]) => async {
            let item = stac_source.load(&engine).await?;
            let (names, cogs, _) =
                resolve_stac_assets(&engine, &item, &pq.assets, None, None).await?;
            let mut props = serde_json::Map::new();
            for (name, cog) in names.into_iter().zip(cogs) {
                let mut info = build_cog_info(&engine, &cog)?;
                if pq.bounds_crs.is_some() {
                    let crs = engine.get_or_build_crs(&cog)?;
                    let full = cog.index.full_res();
                    let wgs84 = crs.wgs84_bounds(&full.pixel_to_crs, full.gcp_poly.as_deref(), full.width, full.height);
                    apply_info_bounds_crs(&mut info, wgs84, pq.bounds_crs.as_ref())?;
                }
                props.insert(name, info);
            }
            let geom = item.geometry.clone().unwrap_or_else(|| {
                let b = item.bbox.unwrap_or([-180.0, -90.0, 180.0, 90.0]);
                serde_json::json!({
                    "type": "Polygon",
                    "coordinates": [[[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]], [b[0], b[1]]]]
                })
            });
            let feat = serde_json::json!({
                "type": "Feature",
                "geometry": geom,
                "properties": props,
            });
            Ok((Bytes::from(feat.to_string()), "application/geo+json"))
        }
        .await,

        // GET /stac/asset_statistics (per-asset dict: {asset: {b1: ...}})
        (Method::GET, ["asset_statistics"]) => async {
            let item = stac_source.load(&engine).await?;
            let (names, cogs, _) =
                resolve_stac_assets(&engine, &item, &pq.assets, None, pq.nodata).await?;
            let n = names.len();
            let stats_opts = build_stats_options(&pq, false)?;
            let entry_opts = stac_entry_options(&pq, &names);
            let mut set = tokio::task::JoinSet::new();
            for (idx, cog) in cogs.into_iter().enumerate() {
                let eng = engine.clone();
                let max_size = pq.max_size;
                let unscale = pq.unscale;
                let nodata = pq.nodata;
                let mut opts_c = stats_opts.clone();
                if let Some(expr_str) = entry_opts[idx].0.as_deref() {
                    let prog = ExprProgram::compile(expr_str).map_err(EngineError::BadRequest)?;
                    opts_c.expressions = Some(vec![(expr_str.to_string(), prog)]);
                } else if let Some(sel) = entry_opts[idx].1.as_deref() {
                    opts_c.bidx = Some(sel.to_vec());
                }
                set.spawn(async move {
                    let st = compute_statistics_with_options(
                        &eng, &cog, max_size, None, unscale, nodata, &opts_c,
                    )
                    .await?;
                    Ok::<_, EngineError>((idx, st))
                });
            }
            let mut ordered: Vec<Option<serde_json::Value>> = vec![None; n];
            while let Some(res) = set.join_next().await {
                let (idx, st) = res.map_err(|e| EngineError::BadRequest(format!("join error: {e}")))??;
                ordered[idx] = Some(st);
            }
            let mut map = serde_json::Map::new();
            for (name, st) in names.into_iter().zip(ordered) {
                if let Some(mut st) = st {
                    if let Some(sel) = stac_asset_selected_bands(&pq, &name)
                        && let Some(obj) = st.as_object() {
                            let mut filtered = serde_json::Map::new();
                            for &b in sel {
                                let k = format!("b{b}");
                                if let Some(v) = obj.get(&k) {
                                    filtered.insert(k, v.clone());
                                }
                            }
                            st = serde_json::Value::Object(filtered);
                        }
                    map.insert(name, st);
                }
            }
            Ok((Bytes::from(serde_json::Value::Object(map).to_string()), "application/json"))
        }
        .await,
        // GET /stac/statistics (merged band dict: {b1: ..., b2: ...})
        (Method::GET, ["statistics"]) => async {
            let item = stac_source.load(&engine).await?;
            let (names, cogs, rewritten_expr) =
                resolve_stac_assets(&engine, &item, &pq.assets, pq.expr_raw.as_deref(), pq.nodata).await?;
            // An item's band math is over `b1..bn` of the stack only: an asset named `b2` is
            // not read for `b2`, and a name is refused as titiler-pgstac refuses it.
            let rewritten_expr = if stac_prefix != "stac" { pq.expr_raw.clone() } else { rewritten_expr };
            if stac_prefix != "stac"
                && let Some(expression) = pq.expr_raw.as_deref()
            {
                pgstac::check_item_band_math_names(expression)?;
            }
            let is_dem_alg = pq.algorithm.as_deref().is_some_and(|a| {
                crate::algorithms::lookup_algorithm(a)
                    .is_some_and(|d| d.kind == crate::algorithms::AlgorithmKind::Dem)
            });
            if is_dem_alg {
                // Band math reads the stack the image routes read: asset names rewritten and
                // each asset's bands in its own slots.
                let mut pq_dem = pq.clone();
                let (cogs, dem_slots) = if pq.expr_raw.is_some() || !pq.asset_exprs.is_empty() {
                    expand_stac_asset_bidx(&mut pq_dem, &names, cogs.clone(), rewritten_expr.clone(), stac_prefix == "stac")?
                } else {
                    (cogs.clone(), None)
                };
                let pq = &pq_dem;
                let crs = engine.get_or_build_crs(&cogs[0])?;
                let full = cogs[0].index.full_res();
                let bbox = crs.wgs84_bounds(&full.pixel_to_crs, full.gcp_poly.as_deref(), full.width, full.height);
                let scale = (pq.max_size as f64) / (full.width.max(full.height) as f64);
                let def_w = ((full.width as f64 * scale.min(1.0)).ceil() as u32).max(1);
                let def_h = ((full.height as f64 * scale.min(1.0)).ceil() as u32).max(1);
                let (out_w, out_h) = resolve_query_dims(def_w, def_h, pq.width, pq.height);
                let db = if cogs.len() > 1 { cogs.len() as u16 } else { data_band_count(full) };
                let mut bands = build_band_request(pq, db, full.sample_type)?;
                if pq.unscale {
                    apply_unscale_cogs(&mut bands, &cogs, dem_slots.as_deref());
                }
                let stats_opts = build_stats_options(pq, false)?;
                let alg_name = pq.algorithm.as_deref().unwrap_or("algorithm");
                if dem_slots.is_none()
                    && matches!(
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
                let raw = render_stac_bbox_opt(
                    &engine,
                    &cogs,
                    dem_slots.as_deref(),
                    bbox,
                    &cogzilla_proj::CrsModel::Geographic,
                    true,
                    out_w,
                    out_h,
                    None,
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
            let (cogs, slot_bands, math) =
                stac_stats_band_math(&pq, &names, cogs, rewritten_expr.clone(), stac_prefix == "stac")?;
            let (mut expr_progs, expr_hist) = match math {
                Some((compiled, hist)) => (Some(compiled), hist),
                None => (None, None),
            };
            if expr_progs.is_none() && pq.algorithm.is_some() {
                let db = if cogs.len() > 1 {
                    cogs.len() as u16
                } else {
                    data_band_count(cogs[0].index.full_res())
                };
                expr_progs = build_stats_options_with_bands(&pq, true, db, Some(reads_type(&cogs)))?.expressions;
            }
            if let Some(compiled) = expr_progs {
                let crs = engine.get_or_build_crs(&cogs[0])?;
                let full = cogs[0].index.full_res();
                let bbox = crs.wgs84_bounds(&full.pixel_to_crs, full.gcp_poly.as_deref(), full.width, full.height);
                let scale = (pq.max_size as f64) / (full.width.max(full.height) as f64);
                let def_w = ((full.width as f64 * scale.min(1.0)).ceil() as u32).max(1);
                let def_h = ((full.height as f64 * scale.min(1.0)).ceil() as u32).max(1);
                let (out_w, out_h) = resolve_query_dims(def_w, def_h, pq.width, pq.height);
                let mut stats_opts = build_stats_options(&pq, false)?;
                if expr_hist.is_some() {
                    stats_opts.hist_dtype = expr_hist;
                }
                let mut map = serde_json::Map::new();
                for (idx, (desc, prog)) in compiled.into_iter().enumerate() {
                    let mut bands = BandRequest::CustomExpr {
                        prog,
                        rescale: (0.0, 1.0),
                        colormap: ColormapName::Grayscale,
                    };
                    if pq.unscale {
                        apply_unscale_cogs(&mut bands, &cogs, slot_bands.as_deref());
                    }
                    let stat = match slot_bands.as_deref() {
                        // Several bands of one asset: read slot by slot, then counted.
                        Some(slots) => {
                            let raw = render_stac_bbox_opt(
                                &engine, &cogs, Some(slots), bbox, &cogzilla_proj::CrsModel::Geographic, true, out_w, out_h,
                                None, &bands, pq.resample, OutputFormat::RawF32, pq.nodata, None, None, true, pq.unscale,
                            )
                            .await?;
                            let n_px = (out_w as usize) * (out_h as usize);
                            let pixels: Vec<f32> =
                                raw.chunks_exact(4).take(n_px).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
                            compute_f32_band_stats_with_engine(&engine, &pixels, &desc, &stats_opts)
                        }
                        None => {
                            render_native_preview_stats(
                                &engine,
                                &cogs,
                                bbox,
                                out_w,
                                out_h,
                                &bands,
                                pq.resample,
                                pq.nodata,
                                pq.unscale,
                                &desc,
                                &stats_opts,
                            )
                            .await?
                        }
                    };
                    map.insert(format!("b{}", idx + 1), stat);
                }
                return Ok((Bytes::from(serde_json::Value::Object(map).to_string()), "application/json"));
            }
            let n = cogs.len();
            let stats_opts = build_stats_options(&pq, false)?;
            let entry_opts = stac_entry_options(&pq, &names);
            let mut set = tokio::task::JoinSet::new();
            for (idx, cog) in cogs.into_iter().enumerate() {
                let eng = engine.clone();
                let max_size = pq.max_size;
                let unscale = pq.unscale;
                let nodata = pq.nodata;
                let mut opts_c = stats_opts.clone();
                if let Some(expr_str) = entry_opts[idx].0.as_deref() {
                    let prog = ExprProgram::compile(expr_str).map_err(EngineError::BadRequest)?;
                    opts_c.expressions = Some(vec![(expr_str.to_string(), prog)]);
                } else if let Some(sel) = entry_opts[idx].1.as_deref() {
                    opts_c.bidx = Some(sel.to_vec());
                }
                set.spawn(async move {
                    let nb = cog.index.full_res().samples_per_pixel as usize;
                    let st = compute_statistics_with_options(
                        &eng, &cog, max_size, None, unscale, nodata, &opts_c,
                    )
                    .await?;
                    Ok::<_, EngineError>((idx, nb, st))
                });
            }
            let mut ordered: Vec<Option<(usize, serde_json::Value)>> = vec![None; n];
            while let Some(res) = set.join_next().await {
                let (idx, nb, st) = res.map_err(|e| EngineError::BadRequest(format!("join error: {e}")))??;
                ordered[idx] = Some((nb, st));
            }
            let mut map = serde_json::Map::new();
            let mut band_idx = 1usize;
            for ((name, slot), (own, sel)) in names.iter().zip(ordered).zip(&entry_opts) {
                let Some((nb, st)) = slot else { continue };
                if let Some(obj) = st.as_object() {
                    let bands_iter: Vec<usize> = if own.is_some() {
                        vec![1]
                    } else {
                        match sel.as_deref() {
                            Some(sel) => sel.iter().map(|&b| b as usize).collect(),
                            None if pq.asset_as_band => vec![1],
                            None if !pq.bidx.is_empty() => {
                                pq.bidx.iter().map(|&b| b as usize).collect()
                            }
                            None => (1..=nb).collect(),
                        }
                    };
                    for i in bands_iter {
                        if let Some(v) = obj.get(&format!("b{i}")) {
                            let mut band_val = v.clone();
                            if let Some(bobj) = band_val.as_object_mut() {
                                let desc = if pq.asset_as_band {
                                    name.clone()
                                } else {
                                    format!("{name}_b{i}")
                                };
                                bobj.insert(
                                    "description".to_string(),
                                    serde_json::json!(desc),
                                );
                            }
                            map.insert(format!("b{band_idx}"), band_val);
                            band_idx += 1;
                        }
                    }
                }
            }
            Ok((Bytes::from(serde_json::Value::Object(map).to_string()), "application/json"))
        }
        .await,
        (Method::POST, ["statistics"]) => async {
            let item = stac_source.load(&engine).await?;
            let body_bytes = read_limited_body(req.into_body()).await?;
            // (An item's posted statistics are the search routes', and never come here.)
            let (names, cogs, rewritten_expr) =
                resolve_stac_assets(&engine, &item, &pq.assets, pq.expr_raw.as_deref(), pq.nodata).await?;
            let mut geojson: serde_json::Value = serde_json::from_slice(&body_bytes)
                .map_err(|e| EngineError::BadRequest(format!("invalid GeoJSON: {e}")))?;

            let is_dem_alg = pq.algorithm.as_deref().is_some_and(|a| {
                crate::algorithms::lookup_algorithm(a)
                    .is_some_and(|d| d.kind == crate::algorithms::AlgorithmKind::Dem)
            });
            let mut post_hist = None;
            // Band math reads the assets stacked band by band, as the GET statistics do.
            let (cogs, post_slots, math) = if is_dem_alg {
                (cogs, None, None)
            } else {
                stac_stats_band_math(&pq, &names, cogs, rewritten_expr.clone(), stac_prefix == "stac")?
            };
            let post_expr_progs: Option<Vec<(String, ExprProgram)>> = if let Some((compiled, hist)) = math {
                post_hist = hist;
                Some(compiled)
            } else if rewritten_expr.is_some() && is_dem_alg {
                let parts = expression_parts(rewritten_expr.as_deref().unwrap_or_default())
                    .map(|p| (p.to_string(), p.to_string()))
                    .collect();
                let (compiled, hist) = compile_stats_parts(&pq, parts, Some(reads_type(&cogs)))?;
                post_hist = hist;
                Some(compiled)
            } else if pq.algorithm.is_some() && !is_dem_alg {
                let db = if cogs.len() > 1 {
                    cogs.len() as u16
                } else {
                    data_band_count(cogs[0].index.full_res())
                };
                build_stats_options_with_bands(&pq, true, db, Some(reads_type(&cogs)))?.expressions
            } else {
                None
            };
            let mut stats_opts_post = build_stats_options(&pq, false)?;
            if post_hist.is_some() {
                stats_opts_post.hist_dtype = post_hist;
            }
            // A terrain algorithm's band math reads the stack the image routes read.
            let (cogs, pq, dem_slots) = if is_dem_alg && (pq.expr_raw.is_some() || !pq.asset_exprs.is_empty()) {
                let mut pq_dem = pq.clone();
                let (cogs, slots) =
                    expand_stac_asset_bidx(&mut pq_dem, &names, cogs, rewritten_expr.clone(), stac_prefix == "stac")?;
                (cogs, pq_dem, slots)
            } else {
                (cogs, pq, None)
            };
            let eng_stac = engine.clone();
            let pq_stac = pq.clone();
            let compute_stac_merged_stats = move |mask: Arc<PolygonMask>, poly_bbox: [f64; 4]| {
                let cogs_c = cogs.clone();
                let names_c = names.clone();
                let eng = eng_stac.clone();
                let max_size = pq_stac.max_size;
                let width_opt = pq_stac.width;
                let height_opt = pq_stac.height;
                let unscale = pq_stac.unscale;
                let nodata = pq_stac.nodata;
                let resample = pq_stac.resample;
                let stats_opts = stats_opts_post.clone();
                let pq_bidx_c = pq_stac.bidx.clone();
                let asset_as_band = pq_stac.asset_as_band;
                let expr_progs_c = post_expr_progs.clone();
                let slots_c = post_slots.clone();
                let alg_c = pq_stac.algorithm.clone();
                let pq_clone = pq_stac.clone();
                let dem_slots_c = dem_slots.clone();
                let n = names_c.len();
                async move {
                    if is_dem_alg || expr_progs_c.is_some() {
                        if is_dem_alg {
                            let full = cogs_c[0].index.full_res();
                            let db = if cogs_c.len() > 1 { cogs_c.len() as u16 } else { data_band_count(full) };
                            let mut bands = build_band_request(&pq_clone, db, full.sample_type)?;
                            if mask.is_empty() {
                                let alg_name = alg_c.as_deref().unwrap_or("algorithm");
                                let nb = crate::algorithms::lookup_algorithm(alg_name)
                                    .and_then(|d| d.out_bands)
                                    .unwrap_or(1) as usize;
                                let mut map = serde_json::Map::new();
                                for b in 0..nb {
                                    let desc = if nb == 1 {
                                        alg_name.to_string()
                                    } else {
                                        format!("{alg_name}_b{}", b + 1)
                                    };
                                    let mut empty = compute_f32_band_stats(&[], &desc, &stats_opts);
                                    if let Some(o) = empty.as_object_mut() {
                                        o.insert("area".into(), serde_json::Value::Null);
                                    }
                                    map.insert(format!("b{}", b + 1), empty);
                                }
                                return Ok::<_, EngineError>(serde_json::Value::Object(map));
                            }
                            let (def_w, def_h) = aspect_preserving_dims(poly_bbox, max_size);
                            let (out_w, out_h) = resolve_query_dims(def_w, def_h, width_opt, height_opt);
                            let n_px = (out_w as usize) * (out_h as usize);
                            if unscale {
                                apply_unscale_cogs(&mut bands, &cogs_c, dem_slots_c.as_deref());
                            }
                            let alg_name = alg_c.as_deref().unwrap_or("algorithm");
                            if dem_slots_c.is_none()
                                && matches!(
                                bands,
                                BandRequest::Single { .. }
                                    | BandRequest::Dem { .. }
                                    | BandRequest::CustomExpr { .. }
                                    | BandRequest::Spectral1 { .. }
                                    | BandRequest::Spectral2 { .. }
                                    | BandRequest::Spectral3 { .. }
                            ) {
                                let stat = render_wgs84_bbox_stats(
                                    &eng, &cogs_c, poly_bbox, out_w, out_h, Some(&mask), &bands,
                                    resample, nodata, unscale, alg_name, &stats_opts,
                                )
                                .await?;
                                let mut map = serde_json::Map::new();
                                map.insert("b1".into(), stat);
                                return Ok::<_, EngineError>(serde_json::Value::Object(map));
                            }
                            let raw = if let Some(slots) = dem_slots_c.as_deref() {
                                render_stac_bbox_opt(
                                    &eng, &cogs_c, Some(slots), poly_bbox, &cogzilla_proj::CrsModel::Geographic, false,
                                    out_w, out_h, Some(&mask), &bands, resample, OutputFormat::RawF32, nodata, None, None,
                                    true, unscale,
                                )
                                .await?
                            } else {
                                render_wgs84_bbox(
                                    &eng, &cogs_c, poly_bbox, out_w, out_h, Some(&mask), &bands,
                                    resample, OutputFormat::RawF32, nodata, None, None, true, unscale,
                                )
                                .await?
                            };
                            let nb = if n_px > 0 && raw.len() >= n_px * 4 {
                                (raw.len() / (n_px * 4)).clamp(1, 16)
                            } else {
                                1
                            };
                            let alg_name = alg_c.as_deref().unwrap_or("algorithm");
                            let mut map = serde_json::Map::new();
                            for b in 0..nb {
                                let start = b * n_px * 4;
                                let end = (start + n_px * 4).min(raw.len());
                                let plane: Vec<f32> = raw[start..end]
                                    .chunks_exact(4)
                                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                                    .collect();
                                let desc = if nb == 1 { alg_name.to_string() } else { format!("{alg_name}_b{}", b + 1) };
                                map.insert(format!("b{}", b + 1), compute_f32_band_stats_with_engine(&eng, &plane, &desc, &stats_opts));
                            }
                            return Ok::<_, EngineError>(serde_json::Value::Object(map));
                        }
                        let compiled = expr_progs_c.unwrap();
                        if mask.is_empty() {
                            let mut map = serde_json::Map::new();
                            for (idx, (desc, _)) in compiled.into_iter().enumerate() {
                                let mut empty = compute_f32_band_stats(&[], &desc, &stats_opts);
                                if let Some(o) = empty.as_object_mut() {
                                    o.insert("area".into(), serde_json::Value::Null);
                                }
                                map.insert(format!("b{}", idx + 1), empty);
                            }
                            return Ok::<_, EngineError>(serde_json::Value::Object(map));
                        }
                        let (def_w, def_h) = aspect_preserving_dims(poly_bbox, max_size);
                        let (out_w, out_h) = resolve_query_dims(def_w, def_h, width_opt, height_opt);
                        // Fast exact analytical coverage + ellipsoidal area-weighted native-grid path
                        // (shared with POST /cog/statistics) whenever assets share the same pixel grid
                        // and no explicit output width/height resampling was requested.
                        if width_opt.is_none()
                            && height_opt.is_none()
                            && let Some(exact) = crate::zonal::compute_native_exact_multi_cog_expr_stats(
                                &eng,
                                &cogs_c,
                                slots_c.as_deref(),
                                &mask,
                                max_size,
                                unscale,
                                nodata,
                                &compiled,
                                &stats_opts,
                            )
                            .await?
                        {
                            return Ok::<_, EngineError>(exact);
                        }
                        let mut eff_bbox = poly_bbox;
                        if eff_bbox[2] <= eff_bbox[0] || eff_bbox[3] <= eff_bbox[1] {
                            let (px_dx, px_dy) = if let Some(first) = cogs_c.first()
                                && let Ok(crs) = eng.get_or_build_crs(first)
                            {
                                let f = first.index.full_res();
                                let b = crs.wgs84_bounds(&f.pixel_to_crs, f.gcp_poly.as_deref(), f.width, f.height);
                                (
                                    ((b[2] - b[0]).abs() / f.width.max(1) as f64).max(1e-6),
                                    ((b[3] - b[1]).abs() / f.height.max(1) as f64).max(1e-6),
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
                        let mut rendered: Vec<(String, String, Vec<f32>)> =
                            Vec::with_capacity(compiled.len());
                        for (idx, (desc, prog)) in compiled.into_iter().enumerate() {
                            let mut bands = BandRequest::CustomExpr {
                                prog,
                                rescale: (0.0, 1.0),
                                colormap: ColormapName::Grayscale,
                            };
                            if unscale {
                                apply_unscale_cogs(&mut bands, &cogs_c, slots_c.as_deref());
                            }
                            let raw = match slots_c.as_deref() {
                                Some(slots) => {
                                    render_stac_bbox_opt(
                                        &eng, &cogs_c, Some(slots), eff_bbox, &cogzilla_proj::CrsModel::Geographic, false,
                                        out_w, out_h, None, &bands, resample, OutputFormat::RawF32, nodata, None, None,
                                        true, unscale,
                                    )
                                    .await?
                                }
                                None => {
                                    render_wgs84_bbox(
                                        &eng, &cogs_c, eff_bbox, out_w, out_h, None, &bands,
                                        resample, OutputFormat::RawF32, nodata, None, None, true, unscale,
                                    )
                                    .await?
                                }
                            };
                            let pixels: Vec<f32> = raw
                                .chunks_exact(4)
                                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                                .collect();
                            rendered.push((format!("b{}", idx + 1), desc, pixels));
                        }
                        let plane_refs: Vec<(&str, &str, &[f32])> = rendered
                            .iter()
                            .map(|(k, d, p)| (k.as_str(), d.as_str(), p.as_slice()))
                            .collect();
                        return Ok::<_, EngineError>(
                            crate::zonal::compute_exact_grid_polygon_stats(
                                &plane_refs,
                                eff_bbox,
                                out_w as usize,
                                out_h as usize,
                                &mask,
                                &stats_opts,
                            ),
                        );
                    }
                    let entry_opts = stac_entry_options(&pq_clone, &names_c);
                    let mut set = tokio::task::JoinSet::new();
                    for (idx, cog) in cogs_c.into_iter().enumerate() {
                        let eng_c = eng.clone();
                        let mask_c = mask.clone();
                        let mut opts_c = stats_opts.clone();
                        if let Some(expr_str) = entry_opts[idx].0.as_deref() {
                            let prog = ExprProgram::compile(expr_str)
                                .map_err(EngineError::BadRequest)?;
                            opts_c.expressions = Some(vec![(expr_str.to_string(), prog)]);
                        } else if let Some(sel) = entry_opts[idx].1.as_deref() {
                            opts_c.bidx = Some(sel.to_vec());
                        }
                        let nb = cog.index.full_res().samples_per_pixel as usize;
                        set.spawn(async move {
                            let st = compute_statistics_with_options(
                                &eng_c,
                                &cog,
                                max_size,
                                Some(&mask_c),
                                unscale,
                                nodata,
                                &opts_c,
                            )
                            .await?;
                            Ok::<_, EngineError>((idx, nb, st))
                        });
                    }
                    let mut ordered: Vec<Option<(usize, serde_json::Value)>> = vec![None; n];
                    while let Some(res) = set.join_next().await {
                        let (idx, nb, st) = res
                            .map_err(|e| EngineError::BadRequest(format!("join error: {e}")))??;
                        ordered[idx] = Some((nb, st));
                    }
                    // Merge bands in numeric order 1..=nb across assets to match TiTiler's POST /stac/statistics.
                    let mut merged = serde_json::Map::new();
                    let mut b_counter = 1usize;
                    for ((name, slot), (own, sel)) in names_c.iter().zip(ordered).zip(&entry_opts) {
                        let Some((nb, st)) = slot else { continue };
                        if let Some(obj) = st.as_object() {
                            let bands_iter: Vec<usize> =
                                if own.is_some() {
                                    vec![1]
                                } else {
                                    match sel.as_deref() {
                                        Some(sel) => {
                                            sel.iter().map(|&b| b as usize).collect()
                                        }
                                        None if asset_as_band => vec![1],
                                        None if !pq_bidx_c.is_empty() => {
                                            pq_bidx_c.iter().map(|&b| b as usize).collect()
                                        }
                                        None => (1..=nb).collect(),
                                    }
                                };
                            for b in bands_iter {
                                if let Some(v) = obj.get(&format!("b{b}")) {
                                    let mut band_val = v.clone();
                                    if let Some(bobj) = band_val.as_object_mut() {
                                        let desc = if asset_as_band {
                                            name.clone()
                                        } else {
                                            format!("{name}_b{b}")
                                        };
                                        bobj.insert(
                                            "description".to_string(),
                                            serde_json::json!(desc),
                                        );
                                    }
                                    merged.insert(format!("b{b_counter}"), band_val);
                                    b_counter += 1;
                                }
                            }
                        }
                    }
                    Ok::<_, EngineError>(serde_json::Value::Object(merged))
                }
            };

            if geojson.get("type").and_then(|v| v.as_str()) == Some("FeatureCollection") {
                let fc_crs = match pq.coord_crs.as_ref() {
                    Some(c) => Some(c.clone()),
                    None => crate::stats::extract_geojson_crs(&geojson)?,
                };
                let compute_shared = Arc::new(compute_stac_merged_stats);
                run_feature_collection_stats(
                    &mut geojson,
                    fc_crs.as_ref(),
                    move |mask, poly_bbox| {
                        let compute_c = compute_shared.clone();
                        async move { Ok((compute_c(mask, poly_bbox).await?, ())) }
                    },
                )
                .await?;
                Ok((Bytes::from(geojson.to_string()), "application/geo+json"))
            } else {
                let (mask, poly_bbox) =
                    parse_feature_mask_or_null(&geojson, pq.coord_crs.as_ref())?;
                let stats = compute_stac_merged_stats(Arc::new(mask), poly_bbox).await?;
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

        // GET /stac/tiles & /stac/tiles/{tms} (OGC TilesetList & Tileset, matching /cog/tiles*)
        (Method::GET, ["tiles"]) => async {
            let bounds = crate::engine::tolerate_asset_failure(stac_source.load(&engine).await)?
                .and_then(|item| item.bbox);
            let body = build_ogc_tilesets_list(
                host,
                forwarded_prefix,
                "stac",
                raw_query,
                bounds,
            )
            .to_string();
            Ok((Bytes::from(body), "application/json"))
        }
        .await,
        (Method::GET, ["tiles", tms]) => async {
            let tms_def = crate::tms::lookup_tms(tms)
                .ok_or_else(|| EngineError::NotFound(format!("unknown TileMatrixSet '{tms}'")))?;
            let bounds = crate::engine::tolerate_asset_failure(stac_source.load(&engine).await)?
                .and_then(|item| item.bbox);
            let body = build_ogc_tileset_json(
                host,
                forwarded_prefix,
                "stac",
                tms_def,
                raw_query,
                bounds,
                pq.minzoom,
                pq.maxzoom,
            )
            .to_string();
            Ok((Bytes::from(body), "application/json"))
        }
        .await,

        // GET /stac/{tms}/tilejson.json, /stac/tilejson.json, /stac/{tms}/map.html, /stac/map.html
        (Method::GET, [_, "tilejson.json"]) | (Method::GET, ["tilejson.json"]) => async {
            let tms = if sub.len() == 2 { sub[0] } else { "WebMercatorQuad" };
            let item = stac_source.load(&engine).await?;
            let (_, cogs, _) =
                resolve_stac_assets(&engine, &item, &pq.assets, pq.expr_raw.as_deref(), None).await?;
            let mut tj = build_tilejson(&engine, &cogs[0], host, tms, raw_query)?;
            apply_tilejson_overrides(&mut tj, &pq, host, stac_prefix, tms, raw_query);
            tj["name"] = serde_json::json!(item.id);
            if !item.renders.is_empty() {
                tj["raster_layers"] = serde_json::json!(&item.renders);
            }
            Ok((Bytes::from(tj.to_string()), "application/json"))
        }
        .await,
        // titiler-pgstac has no viewer on an item.
        (Method::GET, ["viewer"]) if stac_prefix != "stac" => {
            Err(EngineError::NotFound("Not Found".into()))
        }
        (Method::GET, [_, "map.html"]) | (Method::GET, ["map.html"]) | (Method::GET, ["viewer"]) => {
            let tms = if sub.len() == 2 { sub[0] } else { "WebMercatorQuad" };
            let tj_url =
                format!("{forwarded_prefix}/{stac_prefix}/{tms}/tilejson.json?{raw_query}");
            Ok((Bytes::from(build_map_html(&tj_url)), "text/html; charset=utf-8"))
        }
        (Method::GET, ["renders"]) => async {
            // The links name the item's address, which only an item given by `url` has.
            let u = match &stac_source {
                StacSource::Url(Some(u)) => u.clone(),
                _ => return Err(EngineError::BadRequest("missing ?url=".into())),
            };
            let item = stac_source.load(&engine).await?;
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
            let mut renders_map = serde_json::Map::new();
            for (rid, rval) in &item.renders {
                renders_map.insert(
                    rid.clone(),
                    build_stac_render_entry(host, forwarded_prefix, &u, rid, rval),
                );
            }
            let out = serde_json::json!({
                "renders": serde_json::Value::Object(renders_map),
                "links": [
                    {
                        "rel": "self",
                        "type": "application/json",
                        "href": format!("{origin}/stac/renders?url={}", percent_encode_component(&u)),
                    }
                ],
            });
            Ok((Bytes::from(out.to_string()), "application/json"))
        }
        .await,
        (Method::GET, ["renders", render_id]) => async {
            let u = match &stac_source {
                StacSource::Url(Some(u)) => u.clone(),
                _ => return Err(EngineError::BadRequest("missing ?url=".into())),
            };
            let item = stac_source.load(&engine).await?;
            let dec_id = percent_decode_opts(render_id, false);
            let render = item
                .renders
                .get(&dec_id)
                .or_else(|| item.renders.get(*render_id))
                .ok_or_else(|| EngineError::NotFound(format!("render '{dec_id}' not found")))?;
            let out = build_stac_render_entry(host, forwarded_prefix, &u, &dec_id, render);
            Ok((Bytes::from(out.to_string()), "application/json"))
        }
        .await,
        // GET /stac/map (OGC API - Maps)
        (Method::GET, ["map"]) => async {
            let item = stac_source.load(&engine).await?;
            let (names, cogs, rewritten_expr) =
                resolve_stac_assets(&engine, &item, &pq.assets, pq.expr_raw.as_deref(), pq.nodata).await?;
            let mut pq_stac = pq;
            let (cogs, slot_bands) = expand_stac_asset_bidx(&mut pq_stac, &names, cogs, rewritten_expr, true)?;
            let crs = engine.get_or_build_crs(&cogs[0])?;
            let full = cogs[0].index.full_res();
            let (dst_crs, dst_epsg) = pq_stac
                .dst_crs
                .clone()
                .or_else(|| pq_stac.bounds_crs.clone())
                .unwrap_or((cogzilla_proj::CrsModel::Geographic, 4326));
            let out_bbox = if let Some(raw_bbox) = pq_stac.bbox_param {
                let raw_bbox =
                    resolve_input_bbox(raw_bbox, pq_stac.bbox_crs.as_ref().map(|(c, _)| c))?;
                if pq_stac.bbox_crs.as_ref().map(|(_, e)| *e) == Some(dst_epsg)
                    || (pq_stac.bbox_crs.is_none() && dst_epsg == 4326)
                {
                    normalize_render_bbox(raw_bbox, &dst_crs)?
                } else {
                    let wb = bbox_to_wgs84(raw_bbox, pq_stac.bbox_crs.as_ref().map(|(c, _)| c))?;
                    normalize_render_bbox(reproject_wgs84_bounds(wb, &dst_crs)?, &dst_crs)?
                }
            } else {
                let wb = item.bbox.unwrap_or_else(|| {
                    crs.wgs84_bounds(&full.pixel_to_crs, full.gcp_poly.as_deref(), full.width, full.height)
                });
                normalize_render_bbox(reproject_wgs84_bounds(wb, &dst_crs)?, &dst_crs)?
            };
            let (def_w, def_h) = aspect_preserving_dims(out_bbox, pq_stac.max_size);
            let (out_w, out_h) = resolve_query_dims(def_w, def_h, pq_stac.width, pq_stac.height);
            let db = if cogs.len() > 1 { cogs.len() as u16 } else { data_band_count(full) };
            let ext = pq_stac.format_param.as_deref().unwrap_or("png");
            let (fmt, ct) = parse_ext_format(ext, pq_stac.webp_quality, pq_stac.jpeg_quality)?;
            let st = widest_sample_type(&cogs);
            let mut bands =
                build_band_request_for_output(&pq_stac, db, st, fmt, ct)?;
            let fmt = render_output_format(
                fmt,
                ct,
                &pq_stac,
                Some(&cogs[0]),
                &bands,
                st,
                cogs.len() == 1 && slot_bands.is_none(),
            );
            if pq_stac.unscale {
                apply_unscale_cogs(&mut bands, &cogs, slot_bands.as_deref());
            }
            let custom_lut = resolve_custom_lut(&pq_stac, &cogs[0], &bands)?;
            let cf = pq_stac.parse_color_formula()?;
            let resample = resolve_resample(Some(&engine), &pq_stac, Some(&cogs[0]), dst_epsg);
            let data = render_stac_bbox_opt(
                &engine, &cogs, slot_bands.as_deref(), out_bbox, &dst_crs, false, out_w, out_h, None, &bands, resample, fmt,
                pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(), pq_stac.return_mask, pq_stac.unscale,
            )
            .await?;
            let data = wrap_geotiff_fmt(data, ct, fmt, out_w, out_h, out_bbox, dst_epsg, pq_stac.return_mask);
            geo_meta = Some((out_bbox, Some(dst_epsg), false));
            Ok((data, ct))
        }
        .await,
        (Method::GET, [_, "WMTSCapabilities.xml"]) | (Method::GET, ["WMTSCapabilities.xml"]) => {
            async {
                let explicit_tms = (sub.len() == 2).then_some(sub[0]);
                if let Some(tms) = explicit_tms
                    && crate::tms::lookup_tms(tms).is_none()
                {
                    return Err(EngineError::NotFound(format!("unknown TileMatrixSet '{tms}'")));
                }
                let item = stac_source.load(&engine).await?;
                let (_, cogs, _) =
                    resolve_stac_assets(&engine, &item, &pq.assets, pq.expr_raw.as_deref(), None)
                        .await?;
                let crs = engine.get_or_build_crs(&cogs[0])?;
                let full = cogs[0].index.full_res();
                let bounds = item
                    .bbox
                    .unwrap_or_else(|| crs.wgs84_bounds(&full.pixel_to_crs, full.gcp_poly.as_deref(), full.width, full.height));
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
                        "stac",
                        tms,
                        &item.id,
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
                        "stac",
                        &entries,
                        None,
                        &item.id,
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

        // GET /stac/tiles/{tms}/{z}/{x}/{y}[@scale][.{format}]
        // and alias GET /stac/tiles/{z}/{x}/{y}[@scale][.{format}] (implicit WebMercatorQuad)
        (Method::GET, ["tiles", _, zs, xs, ys]) | (Method::GET, ["tiles", zs, xs, ys]) => {
            let tms = if sub.len() == 5 { sub[1] } else { "WebMercatorQuad" };
            // An item read from the database can change under the same path: its tile
            // is of the assets it has now.
            let cache_key = match &stac_source {
                StacSource::Item(item) => {
                    format!("{path}?{raw_query}#{:x}", pgstac::source_identity(&item.assets))
                }
                StacSource::Url(_) => format!("{path}?{raw_query}"),
            };
            if let StacSource::Url(Some(u)) = &stac_source
                && let Some(item) = engine.stac_cache.get(u)
            {
                let asset_urls: Vec<String> = item.assets.values().cloned().collect();
                let _ = engine.store.ensure_uris_fresh(&asset_urls).await;
                engine.sync_tile_cache_epoch();
            }
            let stac_tile_geo = TileGeoOpts::from_pq(&pq);
            let stac_tile_source = stac_source.clone();
            let cached = engine
                .tile_cache
                .get_or_render(cache_key, || stac_tile_in_its_type(&engine, stac_array.clone(), stac_tile_source, async {
                    let item = stac_source.load(&engine).await?;
                    let (names, cogs, rewritten_expr) = resolve_stac_assets(
                        &engine,
                        &item,
                        &pq.assets,
                        pq.expr_raw.as_deref(),
                        pq.nodata,
                    )
                    .await?;
                    let (Some(z), Some(x), Some((y, sc, ext))) = (
                        parse_u8_like(zs),
                        parse_u32_like(xs),
                        parse_y_scale_ext(ys),
                    ) else {
                        return Err(EngineError::BadRequest("invalid z/x/y tile coord".into()));
                    };
                    let mut pq_stac = pq;
                    let (cogs, slot_bands) = expand_stac_asset_bidx(&mut pq_stac, &names, cogs, rewritten_expr, true)?;
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
                        &pq_stac,
                        db,
                        st,
                        tms,
                        Some(&cogs[0]),
                        true,
                    )?;
                    // titiler-pgstac's item routes name the media type after the extension
                    // and, without one, choose the format from the pixels.
                    let item_route = stac_prefix != "stac";
                    let ct = if item_route && ext == Some("jpg") { "image/jpg" } else { ct };
                    let item_auto_format =
                        item_route && ext.is_none() && pq_stac.format_param.is_none();
                    if pq_stac.unscale {
                        apply_unscale_cogs(&mut treq.bands, &cogs, slot_bands.as_deref());
                    }
                    if cogs.len() == 1
                        && slot_bands.is_none()
                        && matches!(treq.format, OutputFormat::RawF32 | OutputFormat::Npy)
                    {
                        treq.format = geotiff_render_format(
                            treq.format,
                            ct,
                            &pq_stac,
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
                            pq_stac.buffer,
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
                    let needs_general = tms != "WebMercatorQuad"
                        || slot_bands.is_some()
                        || pq_stac.buffer > 0.0
                        || pq_stac.nodata.is_some()
                        || pq_stac.custom_colormap_json.is_some()
                        || (!pq_stac.return_mask
                            && matches!(treq.format, OutputFormat::Npy | OutputFormat::NpyF64))
                        || (pq_stac.color_formula.is_some()
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
                        || has_embedded_palette(&pq_stac, &cogs[0], &treq.bands)
                        || !crate::engine::cogs_share_base_grid(&cogs);
                    let auto_fmt = ext.is_none() && pq_stac.format_param.is_none();
                    if auto_fmt {
                        treq.format = OutputFormat::RawRgba;
                    } else if wants_u16_png(treq.format, ct, &pq_stac, Some(&cogs[0]), &treq.bands, st) {
                        treq.format = OutputFormat::RawF32;
                    }
                    if let Some(ref cf_str) = pq_stac.color_formula {
                        ColorFormula::parse(cf_str).map_err(EngineError::BadRequest)?;
                    }
                    if needs_general {
                        let custom_lut = resolve_custom_lut(&pq_stac, &cogs[0], &treq.bands)?;
                        let cf = pq_stac.parse_color_formula()?;
                        let out_dim = (treq.tile_size as f32 + 2.0 * pq_stac.buffer).round() as u32;
                        let (data, bbox, epsg) = if tms != "WebMercatorQuad" {
                            let tms_def = crate::tms::lookup_tms(tms).ok_or_else(|| {
                                EngineError::BadRequest(format!("unsupported TMS '{tms}'"))
                            })?;
                            let bbox = tms_def
                                .tile_bounds(z, x, y, pq_stac.buffer, treq.tile_size)
                                .ok_or_else(|| {
                                    EngineError::NotFound(format!(
                                        "Tile {z}/{x}/{y} is outside {tms} matrix"
                                    ))
                                })?;
                            let dst_crs = tms_def.crs_model()?;
                            let d = render_stac_bbox_opt(
                                &engine,
                                &cogs,
                                slot_bands.as_deref(),
                                bbox,
                                &dst_crs,
                                false,
                                out_dim,
                                out_dim,
                                None,
                                &treq.bands,
                                treq.resample,
                                treq.format,
                                pq_stac.nodata,
                                custom_lut.as_ref(),
                                cf.as_ref(),
                                pq_stac.return_mask,
                                pq_stac.unscale,
                            )
                            .await?;
                            (d, bbox, tms_def.epsg)
                        } else {
                            let bbox_3857 = web_mercator_buffered_3857_bbox(
                                z,
                                x,
                                y,
                                pq_stac.buffer,
                                treq.tile_size,
                            );
                            let d = render_stac_bbox_opt(
                                &engine,
                                &cogs,
                                slot_bands.as_deref(),
                                bbox_3857,
                                &cogzilla_proj::CrsModel::Epsg3857,
                                false,
                                out_dim,
                                out_dim,
                                None,
                                &treq.bands,
                                treq.resample,
                                treq.format,
                                pq_stac.nodata,
                                custom_lut.as_ref(),
                                cf.as_ref(),
                                pq_stac.return_mask,
                                pq_stac.unscale,
                            )
                            .await?;
                            (d, bbox_3857, 3857u16)
                        };
                        // An item's picture with pixels missing is a PNG whether or not
                        // the mask was asked for: the format is chosen from the pixels.
                        if item_auto_format
                            && !pq_stac.return_mask
                            && data.len() == (out_dim as usize).pow(2) * 4
                            && data.chunks_exact(4).any(|p| p[3] != 255)
                        {
                            let (w, h) = (out_dim as usize, out_dim as usize);
                            return Ok((encode_rgba_output(w, h, &data, OutputFormat::Png, false), "image/png"));
                        }
                        Ok(finalize_image_output(data, ct, treq.format, auto_fmt, out_dim, out_dim, bbox, epsg, pq_stac.jpeg_quality, pq_stac.return_mask))
                    } else if item_auto_format {
                        pgstac::item_tile_auto_format(&engine, &cogs, treq, &pq_stac).await
                    } else if (pq_stac.color_formula.is_some() || !pq_stac.return_mask)
                        && !matches!(treq.format, OutputFormat::RawF32 | OutputFormat::RawF64 | OutputFormat::Npy | OutputFormat::NpyF64)
                    {
                        let cf = pq_stac.parse_color_formula()?;
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
                            Ok(encode_auto_image(dim, dim, &buf, pq_stac.jpeg_quality, pq_stac.return_mask))
                        } else {
                            let data = if let Some(cf) = cf {
                                let mut buf = raw.to_vec();
                                cf.apply_buffer_in_place(&mut buf, 4);
                                encode_rgba_output(dim, dim, &buf, orig_format, pq_stac.return_mask)
                            } else {
                                encode_rgba_output(dim, dim, &raw, orig_format, pq_stac.return_mask)
                            };
                            let data = wrap_geotiff_fmt(data, ct, orig_format, treq.tile_size, treq.tile_size, bbox, 3857, pq_stac.return_mask);
                            Ok((data, ct))
                        }
                    } else {
                        let data = engine.render_multi_cog_tile(&cogs, &treq).await?;
                        let bbox = web_mercator_buffered_3857_bbox(z, x, y, 0.0, treq.tile_size);
                        Ok(finalize_image_output(data, ct, treq.format, auto_fmt, treq.tile_size, treq.tile_size, bbox, 3857, pq_stac.jpeg_quality, pq_stac.return_mask))
                    }
                }))
                .await;
            if stac_prefix != "stac"
                && let Err(e) = &cached
            {
                return pgstac::engine_error(e);
            }
            let mut resp = to_http_response_cached(&engine, cached, if_none_match);
            attach_resolved_stac_assets(&mut resp);
            if let (Some(z), Some(x), Some((y, sc, _))) =
                (parse_u8_like(zs), parse_u32_like(xs), parse_y_scale_ext(ys))
            {
                attach_tile_geo_headers_opts(&mut resp, tms, z, x, y, sc, stac_tile_geo);
            }
            // An item's tile says where it lies as titiler-pgstac writes it.
            if stac_prefix != "stac" {
                resp = pgstac::item_placed(resp, sub, raw_query);
            }
            return resp;
        }

        // GET /stac/point/{lon},{lat}
        (Method::GET, ["point", coord]) => async {
            let item = stac_source.load(&engine).await?;
            let (names, cogs, rewritten_expr) =
                resolve_stac_assets(&engine, &item, &pq.assets, pq.expr_raw.as_deref(), pq.nodata).await?;
            let Some((x, y)) = parse_point_coord(coord) else {
                return Err(EngineError::BadRequest("expected /stac/point/{lon},{lat}".into()));
            };
            let (lon, lat) = coord_to_lonlat(x, y, pq.coord_crs.as_ref())?;
            let n = names.len();
            let mut set = tokio::task::JoinSet::new();
            for (idx, cog) in cogs.into_iter().enumerate() {
                let eng = engine.clone();
                let unscale = pq.unscale;
                let nodata = pq.nodata;
                set.spawn(async move {
                    let pt = query_point(&eng, &cog, lon, lat, unscale, nodata).await?;
                    Ok::<_, EngineError>((idx, pt))
                });
            }
            let mut ordered: Vec<Option<serde_json::Value>> = vec![None; n];
            while let Some(res) = set.join_next().await {
                let (idx, pt) = res.map_err(|e| EngineError::BadRequest(format!("join error: {e}")))??;
                ordered[idx] = Some(pt);
            }
            let mut all_vals = Vec::new();
            let mut all_names = Vec::new();
            let entry_opts = stac_entry_options(&pq, &names);
            // One plain asset's band math reads its bands as they are; with its own band
            // math or bands chosen, the request's is over those, as over several assets.
            // One asset's band pick under band math: every band is read, and the expression
            // is read over the pick as the statistics read it (`bN` the Nth chosen).
            let single_pick: Option<Vec<u16>> = match entry_opts.as_slice() {
                [(None, Some(sel))] if rewritten_expr.is_some() => Some(sel.clone()),
                _ => None,
            };
            let single_asset_expr = names.len() == 1
                && rewritten_expr.is_some()
                && (single_pick.is_some() || entry_opts.iter().all(|(own, sel)| own.is_none() && sel.is_none()));
            // Band math over several assets reads every band of each, stacked as rio-tiler
            // stacks them (or the bands each `|bidx=` picks); `asset_as_band` reads one.
            let single_band_per_asset = pq.asset_as_band;
            let mut used_asset_bidx = false;
            // Asset names in `/stac` band math read the asset's place in the stack.
            let named = stac_prefix == "stac"
                && names.len() > 1
                && pq.expr_raw.as_deref().is_some_and(|expr| names_in_expression(expr, &names));
            if named && entry_opts.iter().any(|(own, _)| own.is_some()) {
                return Err(EngineError::BadRequest(
                    "an expression over the results of the assets' own band math takes b1..bn, not asset names".into(),
                ));
            }
            let asset_names = names.clone();
            let mut stacked: Vec<Vec<u16>> = Vec::with_capacity(names.len());
            for ((name, pt), (own, sel)) in names.into_iter().zip(ordered).zip(entry_opts) {
                // The bands this asset puts on the stack, by number (0 for band math).
                let mut on_stack: Vec<u16> = Vec::new();
                if let Some(vs) = pt.as_ref().and_then(|p| p["values"].as_array()) {
                    if pq.asset_as_band {
                        let gives = match (own.as_deref(), sel.as_deref()) {
                            (Some(own), _) => expression_parts(own).count(),
                            (None, Some(sel)) => sel.len(),
                            (None, None) => vs.len(),
                        };
                        if gives > 1 {
                            return Err(EngineError::BadRequest(format!(
                                "Can't use `asset_as_band` for multibands asset ('{name}' gives {gives} bands)"
                            )));
                        }
                    }
                    if !single_asset_expr
                        && let Some(expr_str) = own.as_deref()
                    {
                        let parts: Vec<&str> = expr_str
                            .split(';')
                            .map(str::trim)
                            .filter(|s| !s.is_empty())
                            .collect();
                        for expr in parts {
                            let prog =
                                ExprProgram::compile(expr).map_err(EngineError::BadRequest)?;
                            let mut slot_vals = Vec::with_capacity(prog.used_bands.len());
                            for &b in &prog.used_bands {
                                if b == 0 || (b as usize) > vs.len() {
                                    return Err(EngineError::BadRequest(format!(
                                        "expression references b{b} out of range for asset '{name}' (dataset has {} bands)",
                                        vs.len()
                                    )));
                                }
                                let f = vs[(b - 1) as usize]
                                    .as_f64()
                                    .map(|x| x as f32)
                                    .unwrap_or(f32::NAN);
                                slot_vals.push(f);
                            }
                            let res = prog.eval(&slot_vals);
                            all_vals.push(if res.is_finite() {
                                serde_json::json!(res)
                            } else {
                                serde_json::Value::Null
                            });
                            all_names.push(format!("{name}_{expr}"));
                            on_stack.push(0);
                        }
                    } else if !single_asset_expr
                        && let Some(sel) = sel.as_deref()
                    {
                        used_asset_bidx = true;
                        let iter_sel = if single_band_per_asset {
                            &sel[..sel.len().min(1)]
                        } else {
                            sel
                        };
                        for &b in iter_sel {
                            if b == 0 || (b as usize) > vs.len() {
                                return Err(EngineError::BadRequest(format!(
                                    "band {b} out of range for asset '{name}' (dataset has {} bands)",
                                    vs.len()
                                )));
                            }
                            all_vals.push(vs[(b as usize) - 1].clone());
                            all_names.push(if pq.asset_as_band {
                                name.clone()
                            } else {
                                format!("{name}_b{b}")
                            });
                            on_stack.push(b);
                        }
                    } else if single_band_per_asset && single_pick.is_none() {
                        if let Some(v) = vs.first() {
                            all_vals.push(v.clone());
                            all_names.push(if pq.asset_as_band {
                                name.clone()
                            } else {
                                format!("{name}_b1")
                            });
                            on_stack.push(1);
                        }
                    } else {
                        for (i, v) in vs.iter().enumerate() {
                            all_vals.push(v.clone());
                            // (Every band of one asset read for its pick: the asset names
                            // the band it gives under `asset_as_band`.)
                            all_names.push(if pq.asset_as_band { name.clone() } else { format!("{name}_b{}", i + 1) });
                            on_stack.push((i + 1) as u16);
                        }
                    }
                }
                stacked.push(on_stack);
            }
            let stack_names = all_names.clone();
            let rewritten_expr = match pq.expr_raw.as_deref() {
                Some(expr) if named => {
                    let mut at = 0usize;
                    let offsets: Vec<usize> = stacked
                        .iter()
                        .map(|bands| {
                            let first = at;
                            at += bands.len();
                            first
                        })
                        .collect();
                    Some(crate::stac::rewrite_stac_expr_to(expr, &asset_names, |asset, name, band| {
                        let place = match band {
                            None => Some(0),
                            Some(k) => stacked[asset - 1].iter().position(|&b| usize::from(b) == k),
                        };
                        place.map(|p| offsets[asset - 1] + p + 1).ok_or_else(|| {
                            EngineError::BadRequest(format!("band {} of asset '{name}' is not read", band.unwrap_or(0)))
                        })
                    })?)
                }
                Some(expr) if single_pick.is_some() => {
                    let sel = single_pick.as_deref().unwrap_or_default();
                    let read: &[String] = if stac_prefix == "stac" { &asset_names } else { &[] };
                    Some(crate::stac::rewrite_stac_expr_with(
                        expr,
                        read,
                        |_, _, band| Ok(band.unwrap_or_else(|| sel.first().map_or(1, |&b| usize::from(b)))),
                        |k| match sel.get(k.wrapping_sub(1)) {
                            Some(&b) => Ok(Some(usize::from(b))),
                            None => Err(EngineError::BadRequest(format!(
                                "band {k} out of range for asset '{}' ({} bands chosen)",
                                asset_names[0],
                                sel.len()
                            ))),
                        },
                    )?)
                }
                _ => rewritten_expr,
            };
            let band_names: Vec<String> =
                (1..=all_vals.len()).map(|i| format!("b{i}")).collect();
            let mut pt_json = serde_json::json!({
                "coordinates": [x, y],
                "values": all_vals,
                "band_names": band_names,
                "band_descriptions": all_names,
            });
            let point_bidx: &[u16] = if used_asset_bidx && rewritten_expr.is_none() {
                &[]
            } else {
                &pq.bidx
            };
            apply_point_filters(&mut pt_json, rewritten_expr.as_deref(), point_bidx)?;
            // Band math is described, as TiTiler describes it, in the names of the stacked
            // bands (an asset named in it, on `/stac`, stays as written); over one asset's
            // pick, from the bands it reads.
            let describe = if single_pick.is_some() { rewritten_expr.as_deref() } else { pq.expr_raw.as_deref() };
            if let Some(request) = describe {
                let described: Vec<serde_json::Value> = expression_parts(request)
                    .map(|part| {
                        serde_json::json!(substitute_band_tokens(part, |k| {
                            let token = format!("b{k}");
                            if single_pick.is_none() && stac_prefix == "stac" && asset_names.contains(&token) {
                                return None;
                            }
                            stack_names.get(k.wrapping_sub(1)).cloned()
                        }))
                    })
                    .collect();
                pt_json["band_descriptions"] = serde_json::Value::Array(described);
            }
            Ok((Bytes::from(pt_json.to_string()), "application/json"))
        }
        .await,

        // GET /stac/preview[/{w}x{h}][.{ext}]
        (Method::GET, [p]) if *p == "preview" || p.starts_with("preview.") => async {
            let ext_opt = p.strip_prefix("preview.");
            let auto_fmt = ext_opt.is_none() && pq.format_param.is_none();
            let item = stac_source.load(&engine).await?;
            let (names, cogs, rewritten_expr) =
                resolve_stac_assets(&engine, &item, &pq.assets, pq.expr_raw.as_deref(), pq.nodata).await?;
            let mut pq_stac = pq;
            // An item's band math is over `b1..bn`: its asset names are not rewritten.
            let rewritten_expr = if stac_prefix != "stac" { pq_stac.expr_raw.clone() } else { rewritten_expr };
            let (cogs, slot_bands) = expand_stac_asset_bidx(&mut pq_stac, &names, cogs, rewritten_expr, stac_prefix == "stac")?;
            let crs = engine.get_or_build_crs(&cogs[0])?;
            let full = cogs[0].index.full_res();
            let bbox = crs.wgs84_bounds(&full.pixel_to_crs, full.gcp_poly.as_deref(), full.width, full.height);
            let db = if cogs.len() > 1 { cogs.len() as u16 } else { data_band_count(full) };
            let ext = ext_opt.or(pq_stac.format_param.as_deref()).unwrap_or("png");
            let (fmt, ct) = parse_ext_format(ext, pq_stac.webp_quality, pq_stac.jpeg_quality)?;
            let st = widest_sample_type(&cogs);
            let mut bands = build_band_request_for_output(&pq_stac, db, st, fmt, ct)?;
            let fmt = if auto_fmt {
                OutputFormat::RawRgba
            } else {
                render_output_format(
                    fmt,
                    ct,
                    &pq_stac,
                    Some(&cogs[0]),
                    &bands,
                    st,
                    cogs.len() == 1 && slot_bands.is_none(),
                )
            };
            if pq_stac.unscale {
                apply_unscale_cogs(&mut bands, &cogs, slot_bands.as_deref());
            }
            let custom_lut = resolve_custom_lut(&pq_stac, &cogs[0], &bands)?;
            let cf = pq_stac.parse_color_formula()?;
            let (data, out_bbox, out_w, out_h, out_epsg) = if let Some((ref dst_crs, dst_epsg)) = pq_stac.dst_crs {
                let dst_bbox = reproject_wgs84_bounds(bbox, dst_crs)?;
                let (out_w, out_h) = resolve_bbox_query_dims(&engine, &cogs[0], dst_crs, bbox, dst_bbox, pq_stac.max_size, pq_stac.width, pq_stac.height);
                let rs = resolve_resample(Some(&engine), &pq_stac, Some(&cogs[0]), dst_epsg);
                let d = render_stac_bbox_opt(
                    &engine, &cogs, slot_bands.as_deref(), dst_bbox, dst_crs, false, out_w, out_h, None, &bands,
                    rs, fmt, pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(),
                    pq_stac.return_mask, pq_stac.unscale,
                )
                .await?;
                (d, dst_bbox, out_w, out_h, dst_epsg)
            } else {
                let scale = (pq_stac.max_size as f64) / (full.width.max(full.height) as f64);
                let def_w = ((full.width as f64 * scale.min(1.0)).ceil() as u32).max(1);
                let def_h = ((full.height as f64 * scale.min(1.0)).ceil() as u32).max(1);
                let (out_w, out_h) = resolve_query_dims(def_w, def_h, pq_stac.width, pq_stac.height);
                let (nat_epsg, nat_bbox) = cog_native_epsg_and_bounds(&cogs[0]);
                let north_up_aligned = full.pixel_to_crs.is_axis_aligned()
                    && full.pixel_to_crs.a > 0.0
                    && full.pixel_to_crs.e < 0.0;
                if ct == GEOTIFF_CONTENT_TYPE && nat_epsg.is_none() {
                    let rs = resolve_resample(Some(&engine), &pq_stac, Some(&cogs[0]), 4326);
                    let d = render_stac_bbox_opt(
                        &engine, &cogs, slot_bands.as_deref(), bbox, &cogzilla_proj::CrsModel::Geographic, false, out_w, out_h, None, &bands,
                        rs, fmt, pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(),
                        pq_stac.return_mask, pq_stac.unscale,
                    )
                    .await?;
                    (d, bbox, out_w, out_h, 4326)
                } else if ct == GEOTIFF_CONTENT_TYPE && !north_up_aligned {
                    let epsg = nat_epsg.unwrap_or(4326);
                    let d = render_stac_bbox_opt(
                        &engine, &cogs, slot_bands.as_deref(), nat_bbox, &crs, false, out_w, out_h, None, &bands,
                        pq_stac.resample, fmt, pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(),
                        pq_stac.return_mask, pq_stac.unscale,
                    )
                    .await?;
                    (d, nat_bbox, out_w, out_h, epsg)
                } else {
                    let d = render_stac_bbox_opt(
                        &engine, &cogs, slot_bands.as_deref(), bbox, &cogzilla_proj::CrsModel::Geographic, true, out_w, out_h, None, &bands, pq_stac.resample, fmt,
                        pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(), pq_stac.return_mask, pq_stac.unscale,
                    )
                    .await?;
                    (d, nat_bbox, out_w, out_h, nat_epsg.unwrap_or(4326))
                }
            };
            let (data, ct) = finalize_image_output(data, ct, fmt, auto_fmt, out_w, out_h, out_bbox, out_epsg, pq_stac.jpeg_quality, pq_stac.return_mask);
            geo_meta = Some((out_bbox, Some(out_epsg), false));
            Ok((data, ct))
        }
        .await,
        (Method::GET, ["preview", dims_ext]) => async {
            let (out_w, out_h, ext_opt) = parse_wxh_ext(dims_ext).map_err(EngineError::BadRequest)?;
            let item = stac_source.load(&engine).await?;
            let (names, cogs, rewritten_expr) =
                resolve_stac_assets(&engine, &item, &pq.assets, pq.expr_raw.as_deref(), pq.nodata).await?;
            let mut pq_stac = pq;
            // An item's band math is over `b1..bn`: its asset names are not rewritten.
            let rewritten_expr = if stac_prefix != "stac" { pq_stac.expr_raw.clone() } else { rewritten_expr };
            let (cogs, slot_bands) = expand_stac_asset_bidx(&mut pq_stac, &names, cogs, rewritten_expr, stac_prefix == "stac")?;
            let crs = engine.get_or_build_crs(&cogs[0])?;
            let full = cogs[0].index.full_res();
            let bbox = crs.wgs84_bounds(&full.pixel_to_crs, full.gcp_poly.as_deref(), full.width, full.height);
            let db = if cogs.len() > 1 { cogs.len() as u16 } else { data_band_count(full) };
            let ext = ext_opt.or(pq_stac.format_param.as_deref()).unwrap_or("png");
            let (fmt, ct) = parse_ext_format(ext, pq_stac.webp_quality, pq_stac.jpeg_quality)?;
            let st = widest_sample_type(&cogs);
            let mut bands = build_band_request_for_output(&pq_stac, db, st, fmt, ct)?;
            let fmt = render_output_format(
                fmt,
                ct,
                &pq_stac,
                Some(&cogs[0]),
                &bands,
                st,
                cogs.len() == 1 && slot_bands.is_none(),
            );
            if pq_stac.unscale {
                apply_unscale_cogs(&mut bands, &cogs, slot_bands.as_deref());
            }
            let custom_lut = resolve_custom_lut(&pq_stac, &cogs[0], &bands)?;
            let cf = pq_stac.parse_color_formula()?;
            let (data, out_bbox, out_epsg) = if let Some((ref dst_crs, dst_epsg)) = pq_stac.dst_crs {
                let dst_bbox = reproject_wgs84_bounds(bbox, dst_crs)?;
                let rs = resolve_resample(Some(&engine), &pq_stac, Some(&cogs[0]), dst_epsg);
                let d = render_stac_bbox_opt(
                    &engine, &cogs, slot_bands.as_deref(), dst_bbox, dst_crs, false, out_w, out_h, None, &bands,
                    rs, fmt, pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(),
                    pq_stac.return_mask, pq_stac.unscale,
                )
                .await?;
                (d, dst_bbox, dst_epsg)
            } else {
                let (nat_epsg, nat_bbox) = cog_native_epsg_and_bounds(&cogs[0]);
                let north_up_aligned = full.pixel_to_crs.is_axis_aligned()
                    && full.pixel_to_crs.a > 0.0
                    && full.pixel_to_crs.e < 0.0;
                if ct == GEOTIFF_CONTENT_TYPE && nat_epsg.is_none() {
                    let rs = resolve_resample(Some(&engine), &pq_stac, Some(&cogs[0]), 4326);
                    let d = render_stac_bbox_opt(
                        &engine, &cogs, slot_bands.as_deref(), bbox, &cogzilla_proj::CrsModel::Geographic, false, out_w, out_h, None, &bands,
                        rs, fmt, pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(),
                        pq_stac.return_mask, pq_stac.unscale,
                    )
                    .await?;
                    (d, bbox, 4326)
                } else if ct == GEOTIFF_CONTENT_TYPE && !north_up_aligned {
                    let epsg = nat_epsg.unwrap_or(4326);
                    let d = render_stac_bbox_opt(
                        &engine, &cogs, slot_bands.as_deref(), nat_bbox, &crs, false, out_w, out_h, None, &bands,
                        pq_stac.resample, fmt, pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(),
                        pq_stac.return_mask, pq_stac.unscale,
                    )
                    .await?;
                    (d, nat_bbox, epsg)
                } else {
                    let d = render_stac_bbox_opt(
                        &engine, &cogs, slot_bands.as_deref(), bbox, &cogzilla_proj::CrsModel::Geographic, true, out_w, out_h, None, &bands, pq_stac.resample, fmt,
                        pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(), pq_stac.return_mask, pq_stac.unscale,
                    )
                    .await?;
                    (d, nat_bbox, nat_epsg.unwrap_or(4326))
                }
            };
            let data = wrap_geotiff_fmt(data, ct, fmt, out_w, out_h, out_bbox, out_epsg, pq_stac.return_mask);
            geo_meta = Some((out_bbox, Some(out_epsg), false));
            Ok((data, ct))
        }
        .await,

        // GET /stac/bbox/{minx},{miny},{maxx},{maxy}[/{w}x{h}][.{ext}]
        (Method::GET, ["bbox", bbox_seg]) => async {
            let Some((raw_bbox, ext_opt)) = parse_bbox4(bbox_seg) else {
                return Err(EngineError::BadRequest("invalid bbox".into()));
            };
            let bbox = resolve_input_bbox(raw_bbox, pq.coord_crs.as_ref())?;
            let auto_fmt = ext_opt.is_none() && pq.format_param.is_none();
            let item = stac_source.load(&engine).await?;
            let (names, cogs, rewritten_expr) =
                resolve_stac_assets(&engine, &item, &pq.assets, pq.expr_raw.as_deref(), pq.nodata).await?;
            let mut pq_stac = pq;
            let (cogs, slot_bands) = expand_stac_asset_bidx(&mut pq_stac, &names, cogs, rewritten_expr, true)?;
            let db = if cogs.len() > 1 { cogs.len() as u16 } else { data_band_count(cogs[0].index.full_res()) };
            let (fmt, ct) = parse_ext_format(
                ext_opt.or(pq_stac.format_param.as_deref()).unwrap_or("png"),
                pq_stac.webp_quality,
                pq_stac.jpeg_quality,
            )?;
            let st = widest_sample_type(&cogs);
            let mut bands = build_band_request_for_output(&pq_stac, db, st, fmt, ct)?;
            let fmt = if auto_fmt {
                OutputFormat::RawRgba
            } else {
                render_output_format(
                    fmt,
                    ct,
                    &pq_stac,
                    Some(&cogs[0]),
                    &bands,
                    st,
                    cogs.len() == 1 && slot_bands.is_none(),
                )
            };
            if pq_stac.unscale {
                apply_unscale_cogs(&mut bands, &cogs, slot_bands.as_deref());
            }
            let custom_lut = resolve_custom_lut(&pq_stac, &cogs[0], &bands)?;
            let cf = pq_stac.parse_color_formula()?;
            let wgs_bbox_src = bbox_to_wgs84(bbox, pq_stac.coord_crs.as_ref())?;
            let bbox_max = effective_bbox_max_size(&pq_stac);
            let (data, out_bbox, out_w, out_h, epsg) = if let Some((dst_crs, dst_epsg)) = resolve_bbox_dst_crs(&pq_stac).filter(|(_, e)| *e != 4326 && *e != 3857) {
                let dst_bbox = if pq_stac.coord_crs_epsg == Some(dst_epsg) {
                    bbox
                } else {
                    reproject_wgs84_bounds(wgs_bbox_src, &dst_crs)?
                };
                let (out_w, out_h) = resolve_bbox_query_dims(&engine, &cogs[0], &dst_crs, wgs_bbox_src, dst_bbox, bbox_max, pq_stac.width, pq_stac.height);
                let resample = resolve_resample(Some(&engine), &pq_stac, Some(&cogs[0]), dst_epsg);
                let d = render_stac_bbox_opt(
                    &engine, &cogs, slot_bands.as_deref(), dst_bbox, &dst_crs, false, out_w, out_h, None, &bands, resample, fmt,
                    pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(), pq_stac.return_mask, pq_stac.unscale,
                )
                .await?;
                (d, dst_bbox, out_w, out_h, dst_epsg)
            } else if let Some((ref dst_crs, dst_epsg)) = pq_stac.dst_crs {
                let dst_bbox = if pq_stac.coord_crs_epsg == Some(dst_epsg) {
                    bbox
                } else {
                    reproject_wgs84_bounds(wgs_bbox_src, dst_crs)?
                };
                let (out_w, out_h) = resolve_bbox_query_dims(&engine, &cogs[0], dst_crs, wgs_bbox_src, dst_bbox, bbox_max, pq_stac.width, pq_stac.height);
                let resample = resolve_resample(Some(&engine), &pq_stac, Some(&cogs[0]), dst_epsg);
                let d = render_stac_bbox_opt(
                    &engine, &cogs, slot_bands.as_deref(), dst_bbox, dst_crs, false, out_w, out_h, None, &bands, resample, fmt,
                    pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(), pq_stac.return_mask, pq_stac.unscale,
                )
                .await?;
                (d, dst_bbox, out_w, out_h, dst_epsg)
            } else if pq_stac.coord_crs_3857 {
                let (out_w, out_h) = resolve_bbox_query_dims(&engine, &cogs[0], &cogzilla_proj::CrsModel::Epsg3857, wgs_bbox_src, bbox, bbox_max, pq_stac.width, pq_stac.height);
                let resample = resolve_resample(Some(&engine), &pq_stac, Some(&cogs[0]), 3857);
                let d = render_stac_bbox_opt(
                    &engine, &cogs, slot_bands.as_deref(), bbox, &cogzilla_proj::CrsModel::Epsg3857, false, out_w, out_h, None, &bands, resample, fmt,
                    pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(), pq_stac.return_mask, pq_stac.unscale,
                )
                .await?;
                (d, bbox, out_w, out_h, 3857u16)
            } else {
                let (out_w, out_h) = resolve_bbox_query_dims(&engine, &cogs[0], &cogzilla_proj::CrsModel::Geographic, wgs_bbox_src, wgs_bbox_src, bbox_max, pq_stac.width, pq_stac.height);
                let resample = resolve_resample(Some(&engine), &pq_stac, Some(&cogs[0]), 4326);
                let d = render_stac_bbox_opt(
                    &engine, &cogs, slot_bands.as_deref(), wgs_bbox_src, &cogzilla_proj::CrsModel::Geographic, false, out_w, out_h, None, &bands, resample, fmt,
                    pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(), pq_stac.return_mask, pq_stac.unscale,
                )
                .await?;
                (d, wgs_bbox_src, out_w, out_h, 4326u16)
            };
            let (data, ct) = finalize_image_output(data, ct, fmt, auto_fmt, out_w, out_h, out_bbox, epsg, pq_stac.jpeg_quality, pq_stac.return_mask);
            geo_meta = Some((out_bbox, Some(epsg), false));
            Ok((data, ct))
        }
        .await,
        (Method::GET, ["bbox", bbox_seg, dims_ext]) => async {
            let Some((raw_bbox, _)) = parse_bbox4(bbox_seg) else {
                return Err(EngineError::BadRequest("invalid bbox".into()));
            };
            let bbox = resolve_input_bbox(raw_bbox, pq.coord_crs.as_ref())?;
            let (out_w, out_h, ext_opt) = parse_wxh_ext(dims_ext).map_err(EngineError::BadRequest)?;
            let item = stac_source.load(&engine).await?;
            let (names, cogs, rewritten_expr) =
                resolve_stac_assets(&engine, &item, &pq.assets, pq.expr_raw.as_deref(), pq.nodata).await?;
            let mut pq_stac = pq;
            let (cogs, slot_bands) = expand_stac_asset_bidx(&mut pq_stac, &names, cogs, rewritten_expr, true)?;
            let db = if cogs.len() > 1 { cogs.len() as u16 } else { data_band_count(cogs[0].index.full_res()) };
            let ext = ext_opt.or(pq_stac.format_param.as_deref()).unwrap_or("png");
            let (fmt, ct) = parse_ext_format(ext, pq_stac.webp_quality, pq_stac.jpeg_quality)?;
            let st = widest_sample_type(&cogs);
            let mut bands = build_band_request_for_output(&pq_stac, db, st, fmt, ct)?;
            let fmt = render_output_format(
                fmt,
                ct,
                &pq_stac,
                Some(&cogs[0]),
                &bands,
                st,
                cogs.len() == 1 && slot_bands.is_none(),
            );
            if pq_stac.unscale {
                apply_unscale_cogs(&mut bands, &cogs, slot_bands.as_deref());
            }
            let custom_lut = resolve_custom_lut(&pq_stac, &cogs[0], &bands)?;
            let cf = pq_stac.parse_color_formula()?;
            let (data, out_bbox, epsg) = if let Some((dst_crs, dst_epsg)) = resolve_bbox_dst_crs(&pq_stac).filter(|(_, e)| *e != 4326 && *e != 3857) {
                let dst_bbox = if pq_stac.coord_crs_epsg == Some(dst_epsg) {
                    bbox
                } else {
                    let wgs_bbox = bbox_to_wgs84(bbox, pq_stac.coord_crs.as_ref())?;
                    reproject_wgs84_bounds(wgs_bbox, &dst_crs)?
                };
                let resample = resolve_resample(Some(&engine), &pq_stac, Some(&cogs[0]), dst_epsg);
                let d = render_stac_bbox_opt(
                    &engine, &cogs, slot_bands.as_deref(), dst_bbox, &dst_crs, false, out_w, out_h, None, &bands, resample, fmt,
                    pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(), pq_stac.return_mask, pq_stac.unscale,
                )
                .await?;
                (d, dst_bbox, dst_epsg)
            } else if let Some((ref dst_crs, dst_epsg)) = pq_stac.dst_crs {
                let dst_bbox = if pq_stac.coord_crs_epsg == Some(dst_epsg) {
                    bbox
                } else {
                    let wgs_bbox = bbox_to_wgs84(bbox, pq_stac.coord_crs.as_ref())?;
                    reproject_wgs84_bounds(wgs_bbox, dst_crs)?
                };
                let resample = resolve_resample(Some(&engine), &pq_stac, Some(&cogs[0]), dst_epsg);
                let d = render_stac_bbox_opt(
                    &engine, &cogs, slot_bands.as_deref(), dst_bbox, dst_crs, false, out_w, out_h, None, &bands, resample, fmt,
                    pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(), pq_stac.return_mask, pq_stac.unscale,
                )
                .await?;
                (d, dst_bbox, dst_epsg)
            } else if pq_stac.coord_crs_3857 {
                let resample = resolve_resample(Some(&engine), &pq_stac, Some(&cogs[0]), 3857);
                let d = render_stac_bbox_opt(
                    &engine, &cogs, slot_bands.as_deref(), bbox, &cogzilla_proj::CrsModel::Epsg3857, false, out_w, out_h, None, &bands, resample, fmt,
                    pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(), pq_stac.return_mask, pq_stac.unscale,
                )
                .await?;
                (d, bbox, 3857u16)
            } else {
                let wgs_bbox = bbox_to_wgs84(bbox, pq_stac.coord_crs.as_ref())?;
                let resample = resolve_resample(Some(&engine), &pq_stac, Some(&cogs[0]), 4326);
                let d = render_stac_bbox_opt(
                    &engine, &cogs, slot_bands.as_deref(), wgs_bbox, &cogzilla_proj::CrsModel::Geographic, false, out_w, out_h, None, &bands, resample, fmt,
                    pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(), pq_stac.return_mask, pq_stac.unscale,
                )
                .await?;
                (d, wgs_bbox, 4326u16)
            };
            let data = wrap_geotiff_fmt(data, ct, fmt, out_w, out_h, out_bbox, epsg, pq_stac.return_mask);
            geo_meta = Some((out_bbox, Some(epsg), false));
            Ok((data, ct))
        }
        .await,

        // POST /stac/feature/{w}x{h}[.{ext}] and POST /stac/feature[.{ext}]
        (Method::POST, ["feature", dims_ext]) => async {
            let (out_w, out_h, ext_opt) = parse_wxh_ext(dims_ext).map_err(EngineError::BadRequest)?;
            let item = stac_source.load(&engine).await?;
            let body_bytes = read_limited_body(req.into_body()).await?;
            let geojson: serde_json::Value = serde_json::from_slice(&body_bytes)
                .map_err(|e| EngineError::BadRequest(format!("invalid GeoJSON: {e}")))?;
            let (mask, bbox) = PolygonMask::from_geojson_in_crs(&geojson, pq.coord_crs.as_ref())?;
            let (names, cogs, rewritten_expr) =
                resolve_stac_assets(&engine, &item, &pq.assets, pq.expr_raw.as_deref(), pq.nodata).await?;
            let mut pq_stac = pq;
            let (cogs, slot_bands) = expand_stac_asset_bidx(&mut pq_stac, &names, cogs, rewritten_expr, true)?;
            let db = if cogs.len() > 1 { cogs.len() as u16 } else { data_band_count(cogs[0].index.full_res()) };
            let ext = ext_opt.or(pq_stac.format_param.as_deref()).unwrap_or("png");
            let (fmt, ct) = parse_ext_format(ext, pq_stac.webp_quality, pq_stac.jpeg_quality)?;
            let st = widest_sample_type(&cogs);
            let mut bands = build_band_request_for_output(&pq_stac, db, st, fmt, ct)?;
            let fmt = render_output_format(
                fmt,
                ct,
                &pq_stac,
                Some(&cogs[0]),
                &bands,
                st,
                cogs.len() == 1 && slot_bands.is_none(),
            );
            if pq_stac.unscale {
                apply_unscale_cogs(&mut bands, &cogs, slot_bands.as_deref());
            }
            let custom_lut = resolve_custom_lut(&pq_stac, &cogs[0], &bands)?;
            let cf = pq_stac.parse_color_formula()?;
            let eff_dst = resolve_bbox_dst_crs(&pq_stac);
            let target_epsg = eff_dst.as_ref().map(|(_, e)| *e).unwrap_or(4326);
            let resample = resolve_resample(Some(&engine), &pq_stac, Some(&cogs[0]), target_epsg);
            let mask_opt = pq_stac.shape_mask.then_some(&mask);
            let (data, out_bbox, out_epsg) = if let Some((ref dst_crs, dst_epsg)) = eff_dst {
                let dst_bbox = reproject_wgs84_bounds(bbox, dst_crs)?;
                let d = render_stac_bbox_opt(
                    &engine, &cogs, slot_bands.as_deref(), dst_bbox, dst_crs, false, out_w, out_h, mask_opt, &bands, resample, fmt,
                    pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(), pq_stac.return_mask, pq_stac.unscale,
                )
                .await?;
                (d, dst_bbox, dst_epsg)
            } else {
                let d = render_stac_bbox_opt(
                    &engine, &cogs, slot_bands.as_deref(), bbox, &cogzilla_proj::CrsModel::Geographic, false, out_w, out_h, mask_opt, &bands, resample, fmt,
                    pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(), pq_stac.return_mask, pq_stac.unscale,
                )
                .await?;
                (d, bbox, 4326u16)
            };
            let data = wrap_geotiff_fmt(data, ct, fmt, out_w, out_h, out_bbox, out_epsg, pq_stac.return_mask);
            geo_meta = Some((out_bbox, Some(out_epsg), false));
            Ok((data, ct))
        }
        .await,
        (Method::POST, [feat_ext])
            if *feat_ext == "feature" || feat_ext.starts_with("feature.") =>
        async {
            let ext_opt = feat_ext.strip_prefix("feature.");
            let auto_fmt = ext_opt.is_none() && pq.format_param.is_none();
            let s = effective_bbox_max_size(&pq);
            let item = stac_source.load(&engine).await?;
            let body_bytes = read_limited_body(req.into_body()).await?;
            let geojson: serde_json::Value = serde_json::from_slice(&body_bytes)
                .map_err(|e| EngineError::BadRequest(format!("invalid GeoJSON: {e}")))?;
            let (mask, bbox) = PolygonMask::from_geojson_in_crs(&geojson, pq.coord_crs.as_ref())?;
            let (names, cogs, rewritten_expr) =
                resolve_stac_assets(&engine, &item, &pq.assets, pq.expr_raw.as_deref(), pq.nodata).await?;
            let mut pq_stac = pq;
            let (cogs, slot_bands) = expand_stac_asset_bidx(&mut pq_stac, &names, cogs, rewritten_expr, true)?;
            let db = if cogs.len() > 1 { cogs.len() as u16 } else { data_band_count(cogs[0].index.full_res()) };
            let ext = ext_opt.or(pq_stac.format_param.as_deref()).unwrap_or("png");
            let (fmt, ct) = parse_ext_format(ext, pq_stac.webp_quality, pq_stac.jpeg_quality)?;
            let st = widest_sample_type(&cogs);
            let mut bands = build_band_request_for_output(&pq_stac, db, st, fmt, ct)?;
            let fmt = if auto_fmt {
                OutputFormat::RawRgba
            } else {
                render_output_format(
                    fmt,
                    ct,
                    &pq_stac,
                    Some(&cogs[0]),
                    &bands,
                    st,
                    cogs.len() == 1 && slot_bands.is_none(),
                )
            };
            if pq_stac.unscale {
                apply_unscale_cogs(&mut bands, &cogs, slot_bands.as_deref());
            }
            let custom_lut = resolve_custom_lut(&pq_stac, &cogs[0], &bands)?;
            let cf = pq_stac.parse_color_formula()?;
            let eff_dst = resolve_bbox_dst_crs(&pq_stac);
            let target_epsg = eff_dst.as_ref().map(|(_, e)| *e).unwrap_or(4326);
            let resample = resolve_resample(Some(&engine), &pq_stac, Some(&cogs[0]), target_epsg);
            let mask_opt = pq_stac.shape_mask.then_some(&mask);
            let (data, out_bbox, out_w, out_h, out_epsg) = if let Some((ref dst_crs, dst_epsg)) = eff_dst {
                let dst_bbox = reproject_wgs84_bounds(bbox, dst_crs)?;
                let (out_w, out_h) = resolve_bbox_query_dims(&engine, &cogs[0], dst_crs, bbox, dst_bbox, s, pq_stac.width, pq_stac.height);
                let d = render_stac_bbox_opt(
                    &engine, &cogs, slot_bands.as_deref(), dst_bbox, dst_crs, false, out_w, out_h, mask_opt, &bands, resample, fmt,
                    pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(), pq_stac.return_mask, pq_stac.unscale,
                )
                .await?;
                (d, dst_bbox, out_w, out_h, dst_epsg)
            } else {
                let (out_w, out_h) = resolve_bbox_query_dims(&engine, &cogs[0], &cogzilla_proj::CrsModel::Geographic, bbox, bbox, s, pq_stac.width, pq_stac.height);
                let d = render_stac_bbox_opt(
                    &engine, &cogs, slot_bands.as_deref(), bbox, &cogzilla_proj::CrsModel::Geographic, false, out_w, out_h, mask_opt, &bands, resample, fmt,
                    pq_stac.nodata, custom_lut.as_ref(), cf.as_ref(), pq_stac.return_mask, pq_stac.unscale,
                )
                .await?;
                (d, bbox, out_w, out_h, 4326u16)
            };
            let (data, ct) = finalize_image_output(data, ct, fmt, auto_fmt, out_w, out_h, out_bbox, out_epsg, pq_stac.jpeg_quality, pq_stac.return_mask);
            geo_meta = Some((out_bbox, Some(out_epsg), false));
            Ok((data, ct))
        }
        .await,

        (_, ["statistics"]) => Err(EngineError::MethodNotAllowed("GET, POST, HEAD")),
        (_, [f]) if *f == "feature" || f.starts_with("feature.") => {
            Err(EngineError::MethodNotAllowed("POST"))
        }
        (_, ["feature", _]) => Err(EngineError::MethodNotAllowed("POST")),
        (_, ["info"] | ["info.geojson"] | ["bounds"] | ["assets"] | ["asset_statistics"] | ["tiles"] | ["tiles", ..] | ["point", _] | ["preview", ..] | ["bbox", ..] | ["map"] | ["viewer"] | ["renders"] | ["renders", _] | ["tilejson.json"] | [_, "tilejson.json"] | ["map.html"] | [_, "map.html"] | ["WMTSCapabilities.xml"] | [_, "WMTSCapabilities.xml"]) => {
            Err(EngineError::MethodNotAllowed("GET, HEAD"))
        }

        _ => Err(EngineError::NotFound(path.to_string())),
    };
    let res = match res {
        Ok((data, ct)) if stac_array.is_some() && may_be_f32_array(&data) => {
            let source = match &stac_item_hdr {
                Some(item) => StacSource::Item(item.clone()),
                None => StacSource::Url(stac_url_hdr.clone()),
            };
            match source.load(&engine).await {
                Ok(item) => Ok((stac_array_in_its_type(&engine, stac_array, &item, data).await, ct)),
                Err(_) => Ok((data, ct)),
            }
        }
        res => res,
    };
    // titiler-pgstac answers errors as JSON `{"detail": ...}`.
    if stac_prefix != "stac" {
        return match res {
            Err(e) => pgstac::engine_error(&e),
            Ok((body, ct)) => {
                match pgstac::item_fixup(
                    sub,
                    pgstac_item_extent.as_ref(),
                    raw_query,
                    body,
                    ct,
                ) {
                    Ok(fixed) => {
                        let mut resp = to_http_response(&engine, Ok(fixed), if_none_match);
                        attach_x_assets(&mut resp, &stac_x_assets);
                        let resp = if method == Method::GET {
                            pgstac::item_placed(resp, sub, raw_query)
                        } else {
                            resp
                        };
                        let resp = pgstac::item_drawn_placed(resp, sub, geo_meta);
                        pgstac_cacheable(resp, &method)
                    }
                    Err(resp) => resp,
                }
            }
        };
    }
    let mut resp = to_http_response(&engine, res, if_none_match);
    if !matches!(
        sub,
        ["assets"]
            | ["tiles"]
            | ["tiles", _]
            | [_, "map.html"]
            | ["map.html"]
            | ["viewer"]
            | ["renders"]
            | ["renders", _]
    ) {
        attach_resolved_stac_assets(&mut resp);
    }
    if let Some((bbox, epsg, is_crs84)) = geo_meta {
        attach_geo_headers(&mut resp, bbox, epsg, is_crs84);
    }
    resp
}
