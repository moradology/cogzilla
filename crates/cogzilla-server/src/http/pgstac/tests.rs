use super::*;

/// An item's stacked bands are named as titiler-pgstac names them: each result of an
/// asset's own band math after the asset, each band read of a plain asset as `{asset}_bK`
/// (those its `|bidx=` picks), and the asset itself under `asset_as_band`.
#[test]
fn stacked_bands_have_titiler_pgstacs_names() {
    let names = |assets: &[&str]| assets.iter().map(|a| a.to_string()).collect::<Vec<_>>();
    let own = |e: &str| (Some(e.to_string()), None);
    let plain = (None, None);
    assert_eq!(
        item_stack_names(&names(&["visual", "red"]), &[3, 1], &[own("b1+b2;b3"), own("b1*2")], false),
        ["visual_b1+b2", "visual_b3", "red_b1*2"]
    );
    assert_eq!(
        item_stack_names(&names(&["visual", "red"]), &[3, 1], &[plain.clone(), plain.clone()], false),
        ["visual_b1", "visual_b2", "visual_b3", "red_b1"]
    );
    assert_eq!(
        item_stack_names(&names(&["visual", "red"]), &[3, 1], &[(None, Some(vec![3, 1])), plain.clone()], false),
        ["visual_b3", "visual_b1", "red_b1"]
    );
    assert_eq!(item_stack_names(&names(&["nir", "red"]), &[1, 1], &[plain.clone(), plain], true), ["nir", "red"]);
}


/// An asset's band math is over all of its bands: the bands it chooses are dropped, and the
/// options given back so that the shared parser's choice of them is not read either.
#[test]
fn asset_math_drops_the_assets_own_band_choice() {
    let options = |q: &str| item_asset_options(&["point", "1,2"], q).ok().flatten();
    assert_eq!(options("assets=visual%7Cexpression%3Db3%7Cbidx%3D1"), Some((vec![], vec![])));
    assert_eq!(
        options("assets=visual%7Cexpression%3Db3%7Cbidx%3D1&assets=red%7Cbidx%3D1"),
        Some((vec![("red".to_string(), vec![1])], vec![]))
    );
    assert_eq!(options("assets=visual%7Cexpression%3Db3"), None);
}


/// Band math is an `|expression=` option of an asset, not an option whose name ends so.
#[test]
fn asset_math_is_an_expression_option() {
    assert!(asset_math_asked(&query_pairs("assets=red%7Cexpression%3Db1*2")));
    assert!(asset_math_asked(&query_pairs("assets=visual%7Cbidx%3D1%7Cexpression%3Db1")));
    assert!(!asset_math_asked(&query_pairs("assets=visual%7Ccustom_expression%3Db1*2%7Cbidx%3D3")));
    assert!(!asset_math_asked(&query_pairs("assets=red&expression=b1*2")));
}

/// An item's point is read band by band and each asset's band math applied afterwards.
#[test]
fn point_asset_math_applies_each_assets_band_math() {
    let mut doc = json!({
        "values": [77.0, 81.0, 83.0, 750.0, 1234.0],
        "band_descriptions": ["visual_b1", "visual_b2", "visual_b3", "red_b1", "nir_b1"],
    });
    let q = "assets=visual%7Cexpression%3Db1%2Bb2%3Bb3&assets=red%7Cexpression%3Db1*2&assets=nir";
    point_asset_math(&mut doc, &query_pairs(q)).unwrap();
    assert_eq!(doc["values"], json!([158.0, 83.0, 1500.0, 1234.0]));
    assert_eq!(doc["band_names"], json!(["b1", "b2", "b3", "b4"]));
    assert_eq!(doc["band_descriptions"], json!(["visual_b1+b2", "visual_b3", "red_b1*2", "nir_b1"]));

    let mut doc = json!({ "values": [77.0, 81.0, 83.0, 750.0], "band_descriptions": ["visual_b1", "visual_b2", "visual_b3", "red_b1"] });
    let as_band = "assets=visual%7Cexpression%3Db3&assets=red&asset_as_band=true";
    point_asset_math(&mut doc, &query_pairs(as_band)).unwrap();
    assert_eq!((&doc["values"], &doc["band_descriptions"]), (&json!([83.0, 750.0]), &json!(["visual", "red"])));

    let mut doc = json!({ "values": [77.0, 81.0, 83.0, 750.0], "band_descriptions": ["visual_b1", "visual_b2", "visual_b3", "red_b1"] });
    let plain_many = "assets=visual&assets=red%7Cexpression%3Db1&asset_as_band=true";
    assert!(point_asset_math(&mut doc, &query_pairs(plain_many)).is_err(), "a plain asset of three bands is not one band");

    // An asset asked for twice is read twice; a name that ends like a band is its own asset.
    let mut doc = json!({ "values": [750.0, 750.0], "band_descriptions": ["red_b1", "red_b1"] });
    point_asset_math(&mut doc, &query_pairs("assets=red%7Cexpression%3Db1*2&assets=red%7Cexpression%3Db1*3")).unwrap();
    assert_eq!((&doc["values"], &doc["band_descriptions"]), (&json!([1500.0, 2250.0]), &json!(["red_b1*2", "red_b1*3"])));
    let mut doc = json!({ "values": [10.0, 750.0], "band_descriptions": ["red_b1_b1", "red_b1"] });
    point_asset_math(&mut doc, &query_pairs("assets=red_b1%7Cexpression%3Db1*2&assets=red")).unwrap();
    assert_eq!((&doc["values"], &doc["band_descriptions"]), (&json!([20.0, 750.0]), &json!(["red_b1_b1*2", "red_b1"])));
    // Every asset is read whole and each entry's choice of bands taken from it, in the
    // order chosen, whichever way it is spelled; an asset with band math takes none.
    let whole = || json!({ "values": [77.0, 81.0, 83.0, 750.0], "band_descriptions": ["visual_b1", "visual_b2", "visual_b3", "red_b1"] });
    for (q, want) in [
        ("assets=visual%7Cbidx%3D3,1&assets=red%7Cexpression%3Db1*2", json!([83.0, 77.0, 1500.0])),
        ("assets=visual%7Cindexes%3D3&assets=red%7Cexpression%3Db1*2", json!([83.0, 1500.0])),
        ("assets=visual&assets=red%7Cexpression%3Db1*2&asset_bidx=visual%7C3", json!([83.0, 1500.0])),
        ("assets=visual&assets=red%7Cexpression%3Db1*2&asset_bidx=red%7C1", json!([77.0, 81.0, 83.0, 1500.0])),
    ] {
        let mut doc = whole();
        point_asset_math(&mut doc, &query_pairs(q)).unwrap();
        assert_eq!(doc["values"], want, "{q}");
    }
    // The same asset with band math, then plain with a band chosen.
    let mut doc = json!({ "values": [77.0, 81.0, 83.0, 77.0, 81.0, 83.0], "band_descriptions": ["visual_b1", "visual_b2", "visual_b3", "visual_b1", "visual_b2", "visual_b3"] });
    point_asset_math(&mut doc, &query_pairs("assets=visual%7Cexpression%3Db3&assets=visual%7Cbidx%3D1")).unwrap();
    assert_eq!((&doc["values"], &doc["band_descriptions"]), (&json!([83.0, 77.0]), &json!(["visual_b3", "visual_b1"])));
    // Null (nodata) propagates as null; 0/0 maps via nan_to_num to 0.0; empty expression errors.
    let mut doc = json!({ "values": [null, 0.0], "band_descriptions": ["red_b1", "nir_b1"] });
    point_asset_math(&mut doc, &query_pairs("assets=red%7Cexpression%3Db1*2&assets=nir%7Cexpression%3Db1/b1")).unwrap();
    assert_eq!(doc["values"], json!([null, 0.0]));
    let mut doc = json!({ "values": [10.0], "band_descriptions": ["red_b1"] });
    assert!(point_asset_math(&mut doc, &query_pairs("assets=red%7Cexpression%3D")).is_err());
    assert!(point_asset_math(&mut doc, &query_pairs("assets=:all:%7Cexpression%3Db1*2")).is_err());
    // point_asset_math_reads clears bidx/asset_bidx/asset_exprs/asset_as_band and preserves asset order/duplicates.
    let raw = "assets=red%7Cexpression%3Db1*2&assets=red&bidx=1&asset_bidx=red%7C1&asset_as_band=true";
    let mut pq = super::super::parse_query(Some(raw));
    point_asset_math_reads(&mut pq, raw);
    assert_eq!(pq.assets, vec!["red".to_string(), "red".to_string()]);
    assert!(pq.bidx.is_empty() && pq.asset_bidx.is_empty() && pq.asset_exprs.is_empty() && !pq.asset_as_band);
}

#[test]
fn every_resampling_method_is_its_own_kernel() {
    // `cubic_spline` and `lanczos` are drawn with their own kernels, as rasterio's are.
    for parameter in ["resampling", "reproject"] {
        assert_eq!(resample_method(parameter, "cubic_spline"), Some(ResampleMode::CubicSpline));
        assert_eq!(resample_method(parameter, "lanczos"), Some(ResampleMode::Lanczos));
        assert_eq!(resample_method(parameter, "cubic"), Some(ResampleMode::Cubic));
        assert_eq!(resample_method(parameter, "cubicspline"), None, "not rasterio's spelling");
    }
    assert_eq!(resample_method("reproject", "gauss"), None);
    assert_eq!(resample_method("resampling", "sum"), None);
}

#[test]
fn complaints_are_matched_by_the_parameter_they_name() {
    assert_eq!(error_parameter("invalid width=wide: expected width in 1..=4096"), Some("width"));
    assert_eq!(error_parameter("parameter 'cover_scale' is not supported by cogzilla"), Some("cover_scale"));
    assert!(own_error("invalid asset_as_band=perhaps: expected a boolean"));
    assert!(own_error("invalid assets=visual|bidx=x: expected positive integer band indices in |bidx="));
    // A value that spells a parameter's name is not that parameter.
    assert!(!own_error("invalid nodata=histogram_bins: expected a number"));
    assert!(!own_size_error("invalid nodata=width=3: expected a number"));
    let pairs = |q: &str| query_pairs(q);
    assert_eq!(request_expression(&pairs("assets=red&expr=b1*2")), Some("b1*2"));
    assert_eq!(request_expression(&pairs("expression=b1&index=b2")), Some("b2"));
    assert_eq!(request_expression(&pairs("assets=red")), None);
}

#[test]
fn register_body_is_split_like_titiler_pgstac() {
    let (search, metadata) = split_register_body(json!({
        "collections": ["sentinel-2-l2a"],
        "datetime": null,
        "metadata": { "name": "alps", "minzoom": null }
    }))
    .unwrap();
    assert_eq!(search, json!({ "collections": ["sentinel-2-l2a"], "filter-lang": "cql2-json" }));
    assert_eq!(metadata, json!({ "name": "alps", "type": "mosaic" }));

    let (_, metadata) = split_register_body(json!({ "metadata": { "type": "search" } })).unwrap();
    assert_eq!(metadata, json!({ "type": "search" }));
}

#[test]
fn register_body_rejects_what_titiler_pgstac_rejects() {
    let both = json!({ "bbox": [0, 0, 1, 1], "intersects": { "type": "Point", "coordinates": [0, 0] } });
    assert_eq!(split_register_body(both).unwrap_err().status(), StatusCode::UNPROCESSABLE_ENTITY);
    let lang = json!({ "filter-lang": "cql2-text" });
    assert_eq!(split_register_body(lang).unwrap_err().status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert!(split_register_body(json!([1, 2])).is_err());
}

#[test]
fn limits_default_and_parse() {
    assert_eq!(parse_limits(&[]).unwrap(), SearchLimits::default());
    let pairs = query_pairs("scan_limit=50&items_limit=3&time_limit=1&exitwhenfull=false&skipcovered=0");
    let l = parse_limits(&pairs).unwrap();
    assert_eq!((l.scan_limit, l.items_limit, l.time_limit_secs), (50, 3, 1));
    assert!(!l.exitwhenfull && !l.skipcovered);
    assert!(parse_limits(&query_pairs("items_limit=many")).is_err());
    assert!(parse_limits(&query_pairs("scan_limit=-1")).is_err());
}

#[test]
fn tile_polygon_is_a_closed_wgs84_ring() {
    let poly = tile_polygon(TileCoord { z: 1, x: 0, y: 0 });
    let ring = poly["coordinates"][0].as_array().unwrap();
    assert_eq!(ring.len(), 5);
    assert_eq!(ring[0], ring[4]);
    let west = ring[0][0].as_f64().unwrap();
    let north = ring[1][1].as_f64().unwrap();
    assert!((west + 180.0).abs() < 1e-9);
    assert!((north - 85.0511287798066).abs() < 1e-6);
    assert!(ring[2][0].as_f64().unwrap().abs() < 1e-9);
}

#[test]
fn collection_search_is_built_like_titiler_pgstac() {
    let collection = json!({
        "extent": { "spatial": { "bbox": [[-180.0, -90.0, 180.0, 90.0], [1.0, 2.0, 3.0, 4.0]] } },
        "item_assets": { "visual": {}, "aot": {}, "red-jp2": {}, "nir": {}, "blue": {} }
    });
    let (search, metadata) = collection_search("c", &collection, &[]).unwrap();
    assert_eq!(search, json!({ "collections": ["c"], "filter-lang": "cql2-json" }));
    assert_eq!(
        metadata,
        json!({
            "type": "mosaic",
            "name": "Mosaic for 'c' Collection",
            "bounds": [-180.0, -90.0, 180.0, 90.0],
            "assets": ["aot", "nir", "blue", "visual", "red-jp2"]
        })
    );

    let pairs = query_pairs(
        "ids=a,b&bbox=6,44,8,46&datetime=2024-01-01T00:00:00Z/..&sortby=%2Bproperties.eo:cloud_cover,-datetime,id\
         &query=%7B%22eo%3Acloud_cover%22%3A%7B%22lt%22%3A20%7D%7D\
         &filter=%7B%22op%22%3A%22%3D%22%2C%22args%22%3A%5B1%2C1%5D%7D&filter-lang=cql2-json",
    );
    let (search, metadata) = collection_search("c", &json!({}), &pairs).unwrap();
    assert_eq!(
        search,
        json!({
            "collections": ["c"],
            "filter-lang": "cql2-json",
            "ids": ["a", "b"],
            "bbox": [6.0, 44.0, 8.0, 46.0],
            "datetime": "2024-01-01T00:00:00Z/..",
            "sortby": [
                { "field": "properties.eo:cloud_cover", "direction": "asc" },
                { "field": "datetime", "direction": "desc" },
                { "field": "id", "direction": "asc" }
            ],
            "query": { "eo:cloud_cover": { "lt": 20 } },
            "filter": { "op": "=", "args": [1, 1] }
        })
    );
    // The bbox parameter, not the collection's extent, bounds the mosaic.
    assert_eq!(metadata["bounds"], json!([6.0, 44.0, 8.0, 46.0]));
    assert!(metadata.get("assets").is_none());

    // Renders become defaults, without the keys titiler-pgstac drops.
    let with_renders = json!({ "renders": {
        "rgb": { "title": "True colour", "assets": ["visual"], "minmax_zoom": [8, 14], "tilematrixsets": {} },
        "ndvi": { "expression": "(nir-red)/(nir+red)", "rescale": [[-1, 1]], "minmax_zoom": [8] },
        "broken": "not an object"
    }});
    let (_, metadata) = collection_search("c", &with_renders, &[]).unwrap();
    assert_eq!(
        metadata["defaults"],
        json!({
            "rgb": { "assets": ["visual"], "minzoom": 8, "maxzoom": 14 },
            "ndvi": { "expression": "(nir-red)/(nir+red)", "rescale": [[-1, 1]] }
        })
    );

    let status = |q: &str| collection_search("c", &json!({}), &query_pairs(q)).unwrap_err().status();
    assert_eq!(status("bbox=1,2,3"), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(status("bbox=1,2,x,4"), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(status("query=nope"), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(status("filter=eo:cloud_cover%3C"), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(status("filter=eo:cloud_cover"), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(status("filter=a%3D1&filter-lang=cql2-nope"), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(status("filter=a%3D1&filter-lang=cql2-json"), StatusCode::UNPROCESSABLE_ENTITY);

    // A filter is CQL2 text unless `filter-lang` says otherwise, and is stored as the
    // JSON titiler-pgstac's converter gives: numbers as floats, `filter-lang` JSON.
    for query in [
        "filter=eo:cloud_cover%20%3C%2020%20AND%20platform%20%3D%20%27sentinel-2b%27",
        "filter=eo:cloud_cover%3C20+and+platform%3D%27sentinel-2b%27&filter-lang=cql2-text",
    ] {
        let (search, _) = collection_search("c", &json!({}), &query_pairs(query)).unwrap();
        assert_eq!(
            search,
            json!({
                "collections": ["c"],
                "filter-lang": "cql2-json",
                "filter": { "op": "and", "args": [
                    { "op": "<", "args": [{ "property": "eo:cloud_cover" }, 20.0] },
                    { "op": "=", "args": [{ "property": "platform" }, "sentinel-2b"] }
                ]}
            }),
            "{query}"
        );
        assert!(search.to_string().contains("20.0"), "{query}");
    }
}

#[test]
fn default_layers_become_link_query_strings() {
    let metadata = json!({ "defaults": {
        "rgb": { "assets": ["visual"], "minzoom": 8, "maxzoom": 14 },
        "ndvi": { "expression": "(nir-red)/(nir+red)", "rescale": [[-1, 1]], "colormap_name": "viridis" },
        "bands": {
            "assets": "red", "asset_bidx": ["red|1"], "rescale": [[0, 3000], "0,1"],
            "colormap": { "1": [0, 0, 0, 255] }, "unscale": true
        }
    }});
    // jsonb order of the names; `ndvi` has no assets, so it cannot be a tile request.
    assert_eq!(
        layer_queries(&metadata, true),
        vec![
            ("rgb".to_string(), "maxzoom=14&minzoom=8&assets=visual".to_string()),
            (
                "bands".to_string(),
                "unscale=True&rescale=0%2C3000&rescale=0%2C1&colormap=%7B%221%22%3A+%5B0%2C+0%2C+0%2C+255%5D%7D&assets=red%7Cbidx%3D1".to_string()
            ),
        ]
    );
    // Registration links keep asset_bidx as its own parameter.
    assert!(layer_queries(&metadata, false)[1].1.ends_with("&assets=red&asset_bidx=red%7C1"));
    assert!(layer_queries(&json!({ "name": "x" }), true).is_empty());
}

fn utm(epsg: u16) -> CrsModel {
    let keys = cogzilla_tiff::GeoKeys {
        model_type: Some(1),
        projected_cs_type: Some(epsg),
        ..Default::default()
    };
    CrsModel::from_geo_keys(&keys).unwrap()
}

#[test]
fn item_zooms_follow_rio_tiler() {
    // Expected values are rio-tiler 9.4's for the same grids (rasterio's
    // calculate_default_transform, then zoom_for_res).
    let zooms = |crs: &CrsModel, width: f64, height: f64, bounds: [f64; 4]| {
        let e = grid_extent(crs, ProjGrid { bounds, width, height });
        (e.minzoom, e.maxzoom)
    };
    let s2 = [699_960.0, 4_790_220.0, 809_760.0, 4_900_020.0];
    assert_eq!(zooms(&utm(32631), 10980.0, 10980.0, s2), (7, 13));
    assert_eq!(zooms(&utm(32631), 5490.0, 5490.0, s2), (7, 12));
    assert_eq!(zooms(&utm(32631), 1830.0, 1830.0, s2), (8, 11));
    assert_eq!(
        zooms(&utm(32632), 10980.0, 10980.0, [399_960.0, 5_090_220.0, 509_760.0, 5_200_020.0]),
        (7, 13)
    );
    assert_eq!(
        zooms(&utm(32633), 10980.0, 10980.0, [499_980.0, 6_790_200.0, 609_780.0, 6_900_000.0]),
        (7, 13)
    );
    assert_eq!(
        zooms(&utm(32619), 7621.0, 7791.0, [300_000.0, 100_000.0, 528_630.0, 333_730.0]),
        (7, 12)
    );
    assert_eq!(
        zooms(&utm(32755), 10980.0, 10980.0, [300_000.0, 5_690_220.0, 409_800.0, 5_800_020.0]),
        (8, 14)
    );
    assert_eq!(zooms(&CrsModel::Geographic, 3600.0, 1800.0, [-10.0, 30.0, 26.0, 48.0]), (3, 7));
    assert_eq!(
        zooms(&CrsModel::Epsg3857, 4096.0, 2048.0, [500_000.0, 5_300_000.0, 622_304.0, 5_361_152.0]),
        (9, 12)
    );

    // The extent of the Sentinel-2 tile in WGS84, as rasterio's transform_bounds gives it.
    let e = grid_extent(&utm(32631), ProjGrid { bounds: s2, width: 10980.0, height: 10980.0 });
    let want = [5.462735477681616, 43.201211354944526, 6.875738946301351, 44.22597512756446];
    for (got, want) in e.wgs84.iter().zip(want) {
        assert!((got - want).abs() < 1e-9, "{got} vs {want}");
    }
    assert!(e.mercator.is_some());
}

#[test]
fn item_grid_comes_from_the_projection_extension() {
    let tiff = "image/tiff; application=geotiff; profile=cloud-optimized";
    let item = json!({
        "properties": { "proj:epsg": 32631 },
        "assets": {
            "visual": { "href": "v.tif", "type": tiff, "proj:shape": [10980, 10980],
                        "proj:transform": [10, 0, 699960, 0, -10, 4900020] },
            "aot": { "href": "a.tif", "type": tiff, "proj:shape": [5490, 5490],
                     "proj:transform": [20, 0, 699960, 0, -20, 4900020] },
            "aot-jp2": { "href": "a.jp2", "type": "image/jp2" },
            "untyped": { "href": "u.bin" },
            "thumbnail": { "href": "t.jpg", "type": "image/jpeg" },
            "granule_metadata": { "href": "m.xml", "type": "application/xml" }
        }
    });
    // jsonb key order, with the media types rio-tiler reads and untyped assets.
    let readable = readable_assets(&item);
    assert_eq!(readable, ["aot", "visual", "aot-jp2", "untyped"]);
    // The union of the assets' grids at the finest asset's resolution.
    let grid = item_proj_grid(&item, &readable).unwrap();
    assert_eq!(grid.bounds, [699_960.0, 4_790_220.0, 809_760.0, 4_900_020.0]);
    assert_eq!((grid.width, grid.height), (10980.0, 10980.0));

    // Item-level fields are used when no asset has its own.
    let item_level = json!({
        "properties": { "proj:code": "EPSG:32631", "proj:shape": [100, 200],
                        "proj:transform": [30, 0, 500000, 0, -30, 4000000] },
        "assets": { "data": { "href": "d.tif" } }
    });
    let grid = item_proj_grid(&item_level, &["data".to_string()]).unwrap();
    assert_eq!(grid.bounds, [500_000.0, 3_997_000.0, 506_000.0, 4_000_000.0]);
    assert_eq!((grid.width, grid.height), (200.0, 100.0));

    // Assets in different coordinate systems, or no projection fields: nothing to go on.
    let mixed = json!({
        "properties": { "proj:epsg": 32631 },
        "assets": { "a": { "href": "a.tif", "proj:epsg": 32632, "proj:shape": [10, 10],
                           "proj:transform": [10, 0, 0, 0, -10, 0] } }
    });
    assert!(item_proj_grid(&mixed, &["a".to_string()]).is_none());
    let bare = json!({ "properties": {}, "assets": { "a": { "href": "a.tif" } } });
    assert!(item_proj_grid(&bare, &["a".to_string()]).is_none());
}

#[test]
fn item_answers_are_reshaped_like_titiler_pgstac() {
    let extent = ItemExtent { wgs84: [5.0, 43.0, 7.0, 44.0], mercator: None, minzoom: 7, maxzoom: 13 };
    let fix = |sub: &[&str], query: &str, body: Value| {
        let (bytes, _) =
            item_fixup(sub, Some(&extent), query, Bytes::from(body.to_string()), "application/json").unwrap();
        serde_json::from_slice::<Value>(&bytes).unwrap()
    };
    let info = json!({ "visual": { "dtype": "uint8", "minzoom": 9, "maxzoom": 14, "wgs84_bounds": [1, 2, 3, 4] } });
    assert_eq!(fix(&["info"], "", info.clone()), json!({ "visual": { "dtype": "uint8" } }));

    let feature = fix(&["info.geojson"], "", json!({ "type": "Feature", "geometry": null, "properties": info }));
    assert_eq!(feature["bbox"], json!([5.0, 43.0, 7.0, 44.0]));
    assert_eq!(feature["geometry"]["coordinates"][0][2], json!([7.0, 44.0]));
    assert_eq!(feature["properties"], json!({ "visual": { "dtype": "uint8" } }));

    let point = fix(&["point", "6,43"], "", json!({ "band_names": ["red_b1", "nir_b1"], "values": [1.0, 2.0] }));
    assert_eq!(point["band_names"], json!(["b1", "b2"]));

    let tj = json!({ "minzoom": 9, "maxzoom": 14, "bounds": [0, 0, 1, 1], "center": [0.5, 0.5, 9] });
    let fixed = fix(&["WebMercatorQuad", "tilejson.json"], "assets=visual", tj.clone());
    assert_eq!((&fixed["minzoom"], &fixed["maxzoom"]), (&json!(7), &json!(13)));
    assert_eq!(fixed["center"], json!([6.0, 43.5, 7]));
    // Zooms given in the query string stand.
    let fixed = fix(&["WebMercatorQuad", "tilejson.json"], "assets=visual&minzoom=10", tj);
    assert_eq!((&fixed["minzoom"], &fixed["maxzoom"]), (&json!(9), &json!(13)));

    // Tiles are passed through; only the media type of `.jpg` changes.
    let jpg = item_fixup(&["tiles", "WebMercatorQuad", "1", "2", "3.jpg"], None, "", Bytes::from_static(b"x"), "image/jpeg").unwrap();
    assert_eq!(jpg.1, "image/jpg");
    let png = item_fixup(&["tiles", "WebMercatorQuad", "1", "2", "3.png"], None, "", Bytes::from_static(b"x"), "image/png").unwrap();
    assert_eq!((png.0.as_ref(), png.1), (b"x".as_ref(), "image/png"));

    assert!(item_route_needs_assets(&["info"]));
    assert!(item_route_needs_assets(&["preview.png"]));
    assert!(item_route_needs_assets(&["tiles", "WebMercatorQuad", "1", "2", "3"]));
    assert!(!item_route_needs_assets(&["assets"]));
    assert!(!item_route_needs_assets(&["tiles", "WebMercatorQuad"]));
    assert!(!item_route_needs_assets(&["nope"]));
}

#[test]
fn wmts_lists_default_layers_and_the_query_layer() {
    let engine = Arc::new(TileEngine::new(1 << 20));
    let raw_query = "assets=visual&rescale=0,255&tilesize=512&tile_format=jpg&zooms=WebMercatorQuad::8,9";
    let ctx = Ctx {
        engine: &engine,
        host: "http://example.test",
        forwarded_prefix: "",
        path: "/searches/abc/WMTSCapabilities.xml",
        raw_query,
        if_none_match: None,
        only: None,
    };
    let source = || WmtsSource {
        title: "abc",
        bounds: [5.0, 43.5, 12.0, 48.0],
        minzoom: 7,
        maxzoom: 8,
        default_layers: vec![("rgb".into(), "tilesize=256&assets=visual".into())],
    };
    // The body is already in memory, so collecting it is ready on the first poll.
    fn ready<F: std::future::Future>(f: F) -> F::Output {
        let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
        match std::pin::pin!(f).as_mut().poll(&mut cx) {
            std::task::Poll::Ready(v) => v,
            std::task::Poll::Pending => panic!("body was not ready"),
        }
    }
    let body = |resp: Resp| {
        use http_body_util::BodyExt;
        let bytes = ready(resp.into_body().collect()).unwrap().to_bytes();
        String::from_utf8(bytes.to_vec()).unwrap()
    };

    let resp = wmts(&ctx, "/searches/abc", source(), &query_pairs(raw_query));
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(resp.headers()[CONTENT_TYPE], "application/xml");
    let xml = body(resp);
    // The layer from `defaults` keeps the mosaic's zooms and loses its tile size.
    assert!(xml.contains("<ows:Identifier>abc_WebMercatorQuad_rgb</ows:Identifier>"));
    assert!(xml.contains(r#"template="http://example.test/searches/abc/tiles/WebMercatorQuad/{TileMatrix}/{TileCol}/{TileRow}.jpg?assets=visual" />"#));
    assert!(xml.contains("<TileMatrix>7</TileMatrix>"));
    // The query-string layer drops the WMTS-only parameters and takes the zoom override.
    assert!(xml.contains("<ows:Identifier>abc_WebMercatorQuad_default</ows:Identifier>"));
    assert!(xml.contains(r#"{TileRow}.jpg?assets=visual&amp;rescale=0%2C255" />"#));
    assert!(xml.contains("<TileMatrix>9</TileMatrix>"));
    assert!(xml.contains("<MinTileRow>177</MinTileRow>"));
    assert_eq!(xml.matches("<Layer>").count(), 2);
    assert_eq!(xml.matches("<Format>image/jpg</Format>").count(), 2);
    assert!(xml.contains("<ows:LowerCorner>5.0 43.5</ows:LowerCorner>"));
    assert!(xml.contains("<ows:SupportedCRS>http://www.opengis.net/def/crs/EPSG/0/3857</ows:SupportedCRS>"));
    assert!(xml.contains("<ScaleDenominator>559082264.0287178</ScaleDenominator>"));
    assert!(xml.contains("<ScaleDenominator>33.323899747652874</ScaleDenominator>"));
    assert!(xml.contains("WMTSCapabilities.xml?assets=visual&amp;rescale=0,255&amp;tilesize=512"));

    // Nothing that can make a tile, an unknown format, and a format these routes cannot write.
    let bare = WmtsSource { default_layers: Vec::new(), ..source() };
    assert_eq!(wmts(&ctx, "/searches/abc", bare, &[]).status(), StatusCode::BAD_REQUEST);
    let status = |q: &str| wmts(&ctx, "/searches/abc", source(), &query_pairs(q)).status();
    assert_eq!(status("tile_format=nope"), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(status("tile_format=jp2"), StatusCode::BAD_REQUEST);
    assert_eq!(status("tile_format=tif"), StatusCode::OK);
    assert_eq!(status("use_epsg=maybe"), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(status("zooms=WebMercatorQuad"), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(status(""), StatusCode::OK);
}

#[test]
fn an_item_preview_with_a_side_of_zero_is_sized_from_the_query() {
    let sized = |dims: &str| {
        let mut pq = crate::http::parse_query(Some("max_size=32"));
        let path = item_preview_by_query(&["preview", dims], &mut pq);
        (path, pq.width, pq.height, pq.max_size)
    };
    assert_eq!(sized("0x64.png"), (Some("preview.png".into()), None, Some(64), 32));
    assert_eq!(sized("100x0.jpg"), (Some("preview.jpg".into()), Some(100), None, 32));
    assert_eq!(sized("0x0.tif"), (Some("preview.tif".into()), None, None, 32));
    // Both sides given, no format, a side too large or not a number: as it was.
    for dims in ["64x64.png", "0x64", "0x5000.png", "0x-1.png", "ax0.png", "64.png"] {
        assert_eq!(sized(dims), (None, None, None, 32), "{dims}");
    }
    let mut pq = crate::http::parse_query(None);
    assert_eq!(item_preview_by_query(&["bbox", "0x64.png"], &mut pq), None);
    assert_eq!(item_preview_by_query(&["preview", "0x64.png", "more"], &mut pq), None);
}

#[test]
fn selected_values_are_painted_with_the_requests_own_rescale() {
    let nan = f32::NAN;
    // One band through a colour map: 0 to 100 onto 0 to 255, the fraction dropped.
    let gray = Paint::of(&BandRequest::Single { band_1based: 1, rescale: (0.0, 100.0), colormap: ColormapName::Grayscale }).unwrap();
    let rgba = gray.rgba(&Selected { bands: 1, values: vec![0.0, 50.0, 100.0, 250.0, -5.0, nan] }, 6);
    assert_eq!(rgba, [0, 0, 0, 255, 127, 127, 127, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 255, 0, 0, 0, 0]);
    let viridis = Paint::of(&BandRequest::Single { band_1based: 1, rescale: (0.0, 255.0), colormap: ColormapName::Viridis }).unwrap();
    let rgba = viridis.rgba(&Selected { bands: 1, values: vec![0.0, 255.0] }, 2);
    assert_eq!(rgba[..4], [ColormapName::Viridis.lut().rgb[0][0], ColormapName::Viridis.lut().rgb[0][1], ColormapName::Viridis.lut().rgb[0][2], 255]);
    assert_eq!(rgba[4..7], ColormapName::Viridis.lut().rgb[255]);

    // Three bands, each with its own range. A pixel is there when any band of it is.
    let rgb = Paint::of(&BandRequest::Rgb { bands_1based: [1, 2, 3], rescale_min: [0.0, 0.0, 0.0], rescale_max: [255.0, 100.0, 10.0] }).unwrap();
    let selected = Selected { bands: 3, values: vec![255.0, nan, nan, 50.0, 50.0, nan, 5.0, nan, nan] };
    assert_eq!(rgb.rgba(&selected, 3), [255, 127, 127, 255, 0, 127, 0, 255, 0, 0, 0, 0]);
    // A count is one band whatever was asked for: grey on the first band's range.
    let count = Selected { bands: 1, values: vec![0.0, 255.0] };
    assert_eq!(rgb.rgba(&count, 2), [0, 0, 0, 255, 255, 255, 255, 255]);

    let terrain = BandRequest::Terrain { band_1based: 1, algorithm: cogzilla_kernel::DemAlgorithm::Terrarium, math: None };
    assert!(Paint::of(&terrain).is_none());
}

#[test]
fn bands_are_painted_as_titiler_encodes_them() {
    let nan = f32::NAN;
    let ranges = [(0.0, 100.0), (0.0, 10.0), (0.0, 1.0)];
    // One range is for every band, several are one each, none is the default.
    let one = Selected { bands: 1, values: vec![50.0, nan] };
    assert_eq!(paint_bands(&one, 2, &ranges[..1], (0.0, 1.0), ColormapName::Grayscale, None, None).unwrap(), [127, 127, 127, 255, 0, 0, 0, 0]);
    assert_eq!(paint_bands(&one, 2, &[], (0.0, 50.0), ColormapName::Grayscale, None, None).unwrap()[..4], [255, 255, 255, 255]);
    // A colour map of the request's own: classes over the values themselves, or over
    // the byte `rescale` makes of them, or intervals; nothing where there is no value
    // or no class for it.
    let map = |json: &str| MosaicColormap::parse(json).unwrap().unwrap();
    let classes = map(r#"{"50": [1, 2, 3, 200], "127": [9, 9, 9, 255], "700": [7, 7, 7, 255]}"#);
    assert_eq!(paint_bands(&one, 2, &[], (0.0, 1.0), ColormapName::Grayscale, Some(&classes), None).unwrap(), [1, 2, 3, 200, 0, 0, 0, 0]);
    assert_eq!(paint_bands(&one, 2, &ranges[..1], (0.0, 1.0), ColormapName::Grayscale, Some(&classes), None).unwrap()[..4], [9, 9, 9, 255]);
    // A class is for the value that equals it: none for one beside it, or with no class.
    let near = Selected { bands: 1, values: vec![50.4, 70.0, 700.0] };
    assert_eq!(paint_bands(&near, 3, &[], (0.0, 1.0), ColormapName::Grayscale, Some(&classes), None).unwrap(), [0, 0, 0, 0, 0, 0, 0, 0, 7, 7, 7, 255]);
    // Intervals: by the value, or under `rescale` by the byte, and the last given wins.
    let spans = map("[[[0, 40], [1, 1, 1, 255]], [[40, 60], [2, 2, 2, 255]], [[45, 55], [3, 3, 3, 255]], [[120, 130], [4, 4, 4, 255]]]");
    assert_eq!(paint_bands(&one, 2, &[], (0.0, 1.0), ColormapName::Grayscale, Some(&spans), None).unwrap()[..4], [3, 3, 3, 255]);
    assert_eq!(paint_bands(&one, 2, &ranges[..1], (0.0, 1.0), ColormapName::Grayscale, Some(&spans), None).unwrap()[..4], [4, 4, 4, 255]);
    // Classes for every byte take the value as a byte: cut towards zero, round the clock.
    let bytes: Vec<String> = (0..256).map(|k| format!(r#""{k}": [{k}, 0, 0, 255]"#)).collect();
    let bytes = map(&format!("{{{}}}", bytes.join(",")));
    let wide = Selected { bands: 1, values: vec![300.9, -1.0, 7.9] };
    assert_eq!(paint_bands(&wide, 3, &[], (0.0, 1.0), ColormapName::Grayscale, Some(&bytes), None).unwrap(), [44, 0, 0, 255, 255, 0, 0, 255, 7, 0, 0, 255]);
    // A map with nothing in it is no map, and one with too many intervals is refused.
    assert!(MosaicColormap::parse("{}").unwrap().is_none() && MosaicColormap::parse("[]").unwrap().is_none());
    let many: Vec<String> = (0..257).map(|k| format!("[[{k}, {}], [1, 1, 1, 255]]", k + 1)).collect();
    assert!(MosaicColormap::parse(&format!("[{}]", many.join(","))).is_err() && MosaicColormap::parse("{nope").is_err());
    let three = Selected { bands: 3, values: vec![50.0, 5.0, 0.5] };
    assert_eq!(paint_bands(&three, 1, &ranges, (0.0, 1.0), ColormapName::Grayscale, None, None).unwrap(), [127, 127, 127, 255]);
    assert_eq!(paint_bands(&three, 1, &ranges[..1], (0.0, 1.0), ColormapName::Grayscale, None, None).unwrap(), [127, 12, 1, 255]);
    // Two bands: red and green, and the mask for blue; never transparent.
    let two = Selected { bands: 2, values: vec![50.0, nan, nan, 5.0, 10.0, nan] };
    assert_eq!(
        paint_bands(&two, 3, &ranges[..2], (0.0, 1.0), ColormapName::Grayscale, None, None).unwrap(),
        [127, 127, 255, 255, 0, 255, 255, 255, 0, 0, 0, 255]
    );
    // Four bands and more are no picture.
    let four = Selected { bands: 4, values: vec![1.0; 4] };
    assert_eq!(paint_bands(&four, 1, &[], (0.0, 1.0), ColormapName::Grayscale, None, None).unwrap_err(), 4);
}

#[test]
fn band_math_is_held_to_what_a_tile_needs() {
    let bands = [vec![1.0f32; 4]];
    // Results, length and nesting all come from a URL.
    let many = vec!["b1"; super::super::point_expr::MAX_RESULTS + 1].join(";");
    assert!(band_math(&many, &bands).unwrap_err().to_string().contains("results"));
    let long = format!("b1{}", "+b1".repeat(super::super::point_expr::MAX_EXPRESSION_BYTES / 3));
    assert!(band_math(&long, &bands).unwrap_err().to_string().contains("longer"));
    let deep = format!("{}b1{}", "(".repeat(1000), ")".repeat(1000));
    assert!(band_math(&deep, &bands).unwrap_err().to_string().contains("nests"));
    let minus = format!("{}b1", "-".repeat(1000));
    assert!(band_math(&minus, &bands).unwrap_err().to_string().contains("nests"));
    // What band math is in practice is far inside the limits.
    let chain = format!("b1{}", "+b1".repeat(500));
    assert_eq!(band_math(&chain, &bands).unwrap()[0][0], 501.0);
    assert_eq!(band_math("((((b1+1)*2)-3)/(b1+1))**2", &bands).unwrap()[0][0], 0.25);
}

#[test]
fn bytes_are_rio_tilers_to_the_level() {
    // Values on a boundary are where another order of arithmetic lands a level off:
    // 1600 of 4000 is 102 exactly, 51 of 255 is 51.
    assert_eq!(rescale_byte(1600.0, 0.0, 4000.0), 102);
    assert_eq!(rescale_byte(51.0, 0.0, 255.0), 51);
    assert_eq!(rescale_byte(1168.0, 0.0, 4000.0), 74);
    assert_eq!(rescale_byte(-5.0, 0.0, 100.0), 0);
    assert_eq!(rescale_byte(500.0, 0.0, 100.0), 255);
    // A range of nothing is not a panic; one that runs backwards is the ramp inverted.
    assert_eq!(rescale_byte(3.0, 3.0, 3.0), 0);
    assert_eq!(rescale_byte(3.0, 10.0, 0.0), 178);
    // Grey is the byte itself, for every level.
    let grey = Paint::Gray { min: 0.0, max: 255.0, colormap: ColormapName::Grayscale };
    let all: Vec<f32> = (0..=255).map(|v| v as f32).collect();
    let rgba = grey.rgba(&Selected { bands: 1, values: all }, 256);
    for (level, px) in rgba.chunks_exact(4).enumerate() {
        assert_eq!(px, [level as u8, level as u8, level as u8, 255]);
    }
}

#[test]
fn band_math_is_over_whole_planes() {
    let bands = [vec![1.0f32, 2.0, f32::NAN, 4.0, -2.0, 0.0], vec![3.0, 0.0, 1.0, 4.0, 0.0, 0.0]];
    let out = band_math("(b2-b1)/(b2+b1);b1/b2", &bands).unwrap();
    assert_eq!(out.len(), 2);
    assert_eq!(out[0][0], 0.5);
    assert_eq!(out[0][1], -1.0);
    assert!(out[0][2].is_nan(), "no data in, no data out");
    assert_eq!(out[0][3], 0.0);
    assert_eq!(out[1][1], f32::MAX, "positive division by zero saturates to f32::MAX, matching /cog and numpy nan_to_num");
    assert_eq!(out[1][4], f32::MIN, "negative division by zero saturates to f32::MIN, matching /cog and numpy nan_to_num");
    assert_eq!(out[1][5], 0.0, "0/0 maps to 0.0, matching /cog and numpy nan_to_num");
    // Both f64->f32 narrowing and direct f32 nan_to_num agree on IEEE division-by-zero / 0/0 / missing.
    for px in 0..bands[0].len() {
        let missing = bands[0][px].is_nan() || bands[1][px].is_nan();
        let expected = if missing {
            f32::NAN
        } else {
            cogzilla_kernel::band_math_nan_to_num(bands[0][px] / bands[1][px])
        };
        assert_eq!(
            out[1][px].to_bits(),
            expected.to_bits(),
            "px {px}: /searches band_math diverged from cogzilla_kernel::band_math_nan_to_num"
        );
    }
    // Single-precision narrowing of an f64 nan_to_num sentinel must remain finite (not become inf/NaN).
    let f64_pos = cogzilla_kernel::band_math_nan_to_num_f64(2.0f64 / 0.0f64);
    assert_eq!(f64_pos, f64::MAX);
    assert_eq!(cogzilla_kernel::band_math_nan_to_num(f64_pos as f32), f32::MAX);
    let f64_neg = cogzilla_kernel::band_math_nan_to_num_f64(-2.0f64 / 0.0f64);
    assert_eq!(f64_neg, f64::MIN);
    assert_eq!(cogzilla_kernel::band_math_nan_to_num(f64_neg as f32), f32::MIN);
    assert!(band_math("b3", &bands).is_err());
}

#[test]
fn part_sizes_follow_rio_tiler() {
    // A 0.2 by 0.1 degree box over data whose resolution makes it 1802 by 901 pixels.
    let bbox = [8.4, 45.7, 8.6, 45.8];
    let res = 0.2 / 1802.0;
    let query = |max_size, width, height| PartSize::Query { max_size, width, height };
    assert_eq!(part_dims(query(None, None, None), bbox, res), (1802, 901));
    assert_eq!(part_dims(query(Some(256), None, None), bbox, res), (256, 128));
    assert_eq!(part_dims(query(None, Some(300), None), bbox, res), (300, 150));
    assert_eq!(part_dims(query(None, None, Some(90)), bbox, res), (180, 90));
    assert_eq!(part_dims(query(None, Some(300), Some(200)), bbox, res), (300, 200));
    assert_eq!(part_dims(PartSize::Exact(64, 48), bbox, res), (64, 48));
    // `max_size` only shrinks: a box smaller than it keeps the data's resolution.
    assert_eq!(part_dims(query(Some(4096), None, None), bbox, res), (1802, 901));
    // A tall box is limited by its height.
    assert_eq!(part_dims(query(Some(100), None, None), [0.0, 0.0, 0.1, 0.2], res), (50, 100));

    // The lean lookup samples the box as the picture will: the same sizes, wherever
    // they do not depend on the data.
    let grid = |w, h| SampleGrid::for_image(w, h, false);
    assert_eq!(part_grid(query(Some(256), None, None), bbox), grid(256, 128));
    assert_eq!(part_grid(query(None, Some(300), None), bbox), grid(300, 150));
    assert_eq!(part_grid(query(None, None, Some(90)), bbox), grid(180, 90));
    assert_eq!(part_grid(query(None, Some(300), Some(200)), bbox), grid(300, 200));
    assert_eq!(part_grid(PartSize::Exact(64, 48), bbox), grid(64, 48));
    assert_eq!(part_grid(query(Some(100), None, None), [0.0, 0.0, 0.1, 0.2]), grid(50, 100));
    assert!(grid(64, 48).is_some());
    // The data's own resolution is not known yet, and a picture this large is not sampled.
    assert_eq!(part_grid(query(None, None, None), bbox), None);
    assert_eq!(part_grid(PartSize::Exact(5000, 10), bbox), None);
    assert_eq!(part_grid(PartSize::Exact(64, 48), [0.0, 0.0, 0.0, 1.0]), None);

    // `max_size` is dropped once a width or a height is given, as in rio-tiler.
    let pairs = query_pairs("max_size=64&width=300");
    assert_eq!(part_size_from_query(&pairs).unwrap(), query(None, Some(300), None));
    assert_eq!(part_size_from_query(&query_pairs("max_size=64")).unwrap(), query(Some(64), None, None));
    assert!(part_size_from_query(&query_pairs("width=wide")).is_err());
    assert!(part_size_from_query(&query_pairs("height=0")).is_err());
}

#[test]
fn item_renders_are_presented_like_titiler() {
    let item = json!({ "properties": { "renders": {
        "ndvi": { "title": "NDVI", "assets": ["red", "nir"], "expression": "(nir-red)/(nir+red)",
                  "rescale": [[-1, 1]], "colormap_name": "viridis" },
        "one_band": { "assets": ["red"], "asset_bidx": ["red|1"], "minmax_zoom": [8, 12] },
        "no_assets": { "expression": "b1*2" },
        "rgb": { "assets": "visual" }
    }}});
    let renders = item_renders(&item);
    // jsonb key order; a single asset name is a list of one.
    let names: Vec<&str> = renders.iter().map(|r| r.name.as_str()).collect();
    assert_eq!(names, ["rgb", "ndvi", "one_band", "no_assets"]);
    assert_eq!(renders[0].params["assets"], json!(["visual"]));
    assert_eq!(renders[2].params["assets"], json!(["red|bidx=1"]));
    assert!(renders[2].params.get("asset_bidx").is_none());
    assert!(!renders[3].valid && renders[1].valid);

    // The query string is Python's urlencode of the render, lists repeated.
    assert_eq!(
        urlencode_doseq(&renders[1].params),
        "title=NDVI&assets=red&assets=nir&rescale=%5B-1%2C+1%5D&expression=%28nir-red%29%2F%28nir%2Bred%29&colormap_name=viridis"
    );
    assert_eq!(urlencode_doseq(&renders[2].params), "assets=red%7Cbidx%3D1&minmax_zoom=8&minmax_zoom=12");

    let doc = item_render_json("http://h", "/collections/c/items/i", "i", &renders[1]);
    assert_eq!(doc["valid"], true);
    assert_eq!(doc["links"][0]["href"], "http://h/collections/c/items/i/renders/ndvi?url=%3CItem+id%3Di%3E");
    assert_eq!(doc["links"][1]["rel"], "tilesets-map");
    assert!(doc["links"][1]["href"].as_str().unwrap().starts_with(
        "http://h/collections/c/items/i/%7BtileMatrixSetId%7D/tilejson.json?url=%3CItem+id%3Di%3E&title=NDVI&"
    ));
    assert!(doc["links"][2]["href"].as_str().unwrap().contains("/WMTSCapabilities.xml?url="));
    // A render that cannot make a tile has only its own link.
    let doc = item_render_json("http://h", "/collections/c/items/i", "i", &renders[3]);
    assert_eq!((doc["valid"].clone(), doc["links"].as_array().unwrap().len()), (json!(false), 1));
    assert!(item_renders(&json!({ "properties": {} })).is_empty());
}

#[test]
fn point_values_take_band_math() {
    let values = [json!(1060.0), json!(2386.0)];
    let names = ["red_b1".to_string(), "nir_b1".to_string()];
    let (v, band_names, described) = point_expression("(b2-b1)/(b2+b1);b1*2", &values, &names).unwrap();
    assert_eq!(v, [json!(0.3847939640162507), json!(2120.0)]);
    assert_eq!(band_names, ["b1", "b2"]);
    assert_eq!(described, ["(nir_b1-red_b1)/(nir_b1+red_b1)", "red_b1*2"]);
    // A band without a value gives an empty result, and asset names are not band names.
    let (v, _, _) = point_expression("b1+b2", &[Value::Null, json!(1.0)], &names).unwrap();
    assert_eq!(v, [Value::Null]);
    assert_eq!(point_expression("nir-red", &values, &names).unwrap_err(), "Invalid band/asset name 'nir'");

    // On item routes the expression is applied to the `/stac` handler's answer.
    let body = json!({ "coordinates": [1.0, 2.0], "values": [1060.0, 2386.0],
                       "band_names": ["red_b1", "nir_b1"], "band_descriptions": ["red_b1", "nir_b1"] });
    let fix = |query: &str| item_fixup(&["point", "1,2"], None, query, Bytes::from(body.to_string()), "application/json");
    let (bytes, _) = fix("assets=red&assets=nir&expression=b2-b1").unwrap();
    let doc: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(doc["values"], json!([1326.0]));
    assert_eq!(doc["band_names"], json!(["b1"]));
    assert_eq!(doc["band_descriptions"], json!(["nir_b1-red_b1"]));
    assert_eq!(fix("assets=red&expression=nope").unwrap_err().status(), StatusCode::BAD_REQUEST);
}

#[test]
fn statistics_grid_follows_gdal() {
    // A Sentinel-2 10 m raster: full resolution and four overviews.
    let levels = [(10980, 10980), (5490, 5490), (2745, 2745), (1373, 1373), (687, 687)];
    // Read at its own size, or a little reduced: the full resolution.
    assert_eq!(gdal_overview_level(&levels, (541, 542), (541, 542)), 0);
    assert_eq!(gdal_overview_level(&levels, (541, 542), (512, 512)), 0);
    // Reduced 1.8 times: the half-size level is within 1.2 times of that.
    assert_eq!(gdal_overview_level(&levels, (541, 542), (300, 300)), 1);
    // Reduced 8.45 times: the eighth, not the sixteenth.
    assert_eq!(gdal_overview_level(&levels, (541, 542), (64, 64)), 3);
    // The less reduced axis decides.
    assert_eq!(gdal_overview_level(&levels, (1000, 4000), (500, 500)), 1);
    assert_eq!(gdal_overview_level(&levels, (10000, 10000), (100, 100)), 4);
    assert_eq!(gdal_overview_level(&levels[..1], (1000, 1000), (10, 10)), 0);

    // The box grows outwards to whole pixels of the grid.
    let grid = Wgs84Grid { west: 0.0, north: 10.0, res: 0.5 };
    assert_eq!(grid.align([0.6, 8.1, 1.9, 9.4]), [0.5, 8.0, 2.0, 9.5]);
    assert_eq!(grid.align([0.5, 8.0, 2.0, 9.5]), [0.5, 8.0, 2.0, 9.5]);

    // Nearest-neighbour resize, both ways.
    let plane = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    assert_eq!(resize_nearest(plane.clone(), (4, 2), (2, 1)), [6.0, 8.0]);
    assert_eq!(resize_nearest(plane.clone(), (4, 2), (2, 2)), [2.0, 4.0, 6.0, 8.0]);
    assert_eq!(resize_nearest(vec![1.0, 2.0], (2, 1), (4, 1)), [1.0, 1.0, 2.0, 2.0]);
    assert_eq!(resize_nearest(plane.clone(), (4, 2), (4, 2)), plane);
}

#[test]
fn scanlines_are_approximated_like_gdal() {
    // A straight row is its two ends joined, whatever lies between.
    let mut out = vec![(0.0, 0.0); 100];
    approx_scanline(&|i| (i as f64 * 2.0 + 1.0, 7.0 - i as f64 * 0.01), &mut out, 0.125);
    assert!((out[50].0 - 101.0).abs() < 1e-9 && (out[50].1 - 6.5).abs() < 1e-9);

    // A gently curved row is still one straight line: the middle is 0.1 off, under
    // the eighth of a pixel GDAL allows.
    let bend = |i: usize| (i as f64, 0.4 * (i as f64 / 99.0) * (1.0 - i as f64 / 99.0));
    approx_scanline(&bend, &mut out, 0.125);
    assert_eq!(out[49].1, 0.0);
    assert!((bend(49).1 - 0.1).abs() < 0.001);

    // A sharper curve is split until every piece is within the allowance.
    let curve = |i: usize| (i as f64, 40.0 * (i as f64 / 99.0) * (1.0 - i as f64 / 99.0));
    approx_scanline(&curve, &mut out, 0.125);
    for (i, got) in out.iter().enumerate() {
        assert!((got.1 - curve(i).1).abs() <= 0.125, "column {i}: {} vs {}", got.1, curve(i).1);
    }
    // Five points or fewer are worked out one by one, and so is a row that leaves
    // the projection.
    let mut short = vec![(0.0, 0.0); 5];
    approx_scanline(&curve, &mut short, 0.125);
    assert_eq!(short[2], curve(2));
    let broken = |i: usize| if i < 30 { (f64::NAN, f64::NAN) } else { (i as f64, 1.0) };
    approx_scanline(&broken, &mut out, 0.125);
    assert!(out[10].0.is_nan());
    assert_eq!(out[60], (60.0, 1.0));

    // A picture's rows are held to a tolerance at which, with the positions near a
    // source pixel's edge worked out again, every pixel is the projection's own.
    let wave = |i: usize| (i as f64 * 0.37, 300.0 * ((i as f64) / 255.0).powi(2) + (i as f64 * 0.05).sin());
    let mut row = vec![(0.0, 0.0); 256];
    curved_scanline(&wave, &mut row, PICTURE_TOLERANCE);
    for (i, drawn) in row.iter().enumerate() {
        let exact = wave(i);
        let off = (drawn.0 - exact.0).abs().max((drawn.1 - exact.1).abs());
        assert!(off <= 2.0 * PICTURE_TOLERANCE, "column {i} is {off} off");
        // So a drawn position names another pixel than the exact one only near an edge.
        if (drawn.0.floor(), drawn.1.floor()) != (exact.0.floor(), exact.1.floor()) {
            assert!(near_pixel_edge(drawn.0) || near_pixel_edge(drawn.1), "column {i}");
        }
    }
    assert!(near_pixel_edge(7.001) && near_pixel_edge(7.999) && !near_pixel_edge(7.5) && !near_pixel_edge(7.01));

    // A row that bends evenly is one curve: three positions and the two it is checked at.
    let asked = std::cell::Cell::new(0);
    let bow = |i: usize| {
        asked.set(asked.get() + 1);
        (i as f64 * 0.9, 50.0 * (i as f64 / 255.0).powi(2))
    };
    curved_scanline(&bow, &mut row, PICTURE_TOLERANCE);
    assert_eq!(asked.get(), 5);
    assert!((row[100].1 - 50.0 * (100.0f64 / 255.0).powi(2)).abs() < 1e-9);
    // One that leaves the projection is worked out column by column.
    let broken = |i: usize| if i < 30 { (f64::NAN, f64::NAN) } else { (i as f64, 1.0) };
    curved_scanline(&broken, &mut row, PICTURE_TOLERANCE);
    assert!(row[10].0.is_nan());
    assert_eq!(row[60], (60.0, 1.0));
}

/// Every way a picture's row can be asked of a projection that could trip the curve:
/// each family of projection, at home and far outside its area, under tiles that span
/// the world, sit on the antimeridian or reach the poles, with rasters turned, tiny,
/// coarse and far finer than the picture. The pixel taken must be the projection's own.
#[test]
fn pictures_take_the_projections_pixel_in_hostile_cases() {
    use cogzilla_proj::{Ellipsoid, Sinusoidal};
    use cogzilla_raster::Affine2D;

    let mut seed = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    let mut pick = |n: usize| (next() % n as u64) as usize;

    let mut systems: Vec<(String, CrsModel, (f64, f64))> = Vec::new();
    for (code, home) in [
        (3857u16, (10.0, 45.0)), (3395, (10.0, 45.0)), (4326, (10.0, 45.0)),
        (32601, (-177.0, 10.0)), (32631, (3.0, 50.0)), (32660, (177.0, 70.0)), (32733, (15.0, -30.0)), (32760, (177.0, -79.0)),
        (3031, (0.0, -80.0)), (3413, (-45.0, 75.0)), (3995, (0.0, 85.0)), (3035, (10.0, 52.0)), (3575, (10.0, 80.0)),
        (5070, (-96.0, 37.0)), (3338, (-154.0, 60.0)), (2154, (3.0, 46.5)), (3978, (-95.0, 60.0)), (2193, (173.0, -41.0)), (3006, (15.0, 62.0)),
    ] {
        match CrsModel::from_epsg(code) {
            Ok(crs) => systems.push((format!("EPSG:{code}"), crs, home)),
            Err(e) => panic!("EPSG:{code}: {e:?}"),
        }
    }
    systems.push(("sinusoidal".into(), CrsModel::Sinusoidal(Sinusoidal::new(Ellipsoid::sphere(6_371_007.181), 0.0, 0.0, 0.0)), (20.0, 40.0)));

    let world = cogzilla_proj::ORIGIN_SHIFT;
    // `HOSTILE_CASES` runs more of them: twenty thousand is half a billion pixels.
    let cases: usize = std::env::var("HOSTILE_CASES").ok().and_then(|v| v.parse().ok()).unwrap_or(300);
    let (mut pixels, mut worst_off) = (0u64, 0f64);
    for case in 0..cases {
        let (name, crs, natural) = &systems[pick(systems.len())];
        // Where the raster is: at home, or somewhere the projection is strained.
        let place = pick(6);
        let home = match place {
            0 | 1 => *natural,
            2 => (natural.0 + 60.0, natural.1),
            3 => (179.9, natural.1),
            4 => (natural.0, 84.0f64.copysign(natural.1)),
            _ => (natural.0 - 90.0, -natural.1),
        };
        // What is asked: a map tile or a box in degrees.
        let mercator = pick(3) != 0;
        let view = pick(7);
        let (bounds, view_name): ([f64; 4], String) = if mercator {
            let z = [0u8, 1, 2, 3, 5, 8, 12, 16, 20][pick(9)];
            let n = 1u32 << z;
            let (mx, my) = lonlat_deg_to_mercator(home.0.clamp(-179.99, 179.99), home.1);
            let hx = (((mx + world) / (2.0 * world) * f64::from(n)) as u32).min(n - 1);
            let hy = (((world - my) / (2.0 * world) * f64::from(n)) as u32).min(n - 1);
            let (x, y, what) = match view {
                0..=2 => (hx, hy, "home"),
                3 => ((hx + 1) % n, hy, "beside"),
                4 => (0, hy, "west edge"),
                5 => (n - 1, hy, "east edge"),
                _ => (hx, if home.1 > 0.0 { 0 } else { n - 1 }, "pole row"),
            };
            let (a, b, c, d) = tile_bounds_3857(TileCoord { z, x, y });
            ([a, b, c, d], format!("tile z{z} {what}"))
        } else {
            match view {
                0 | 1 => ([home.0 - 0.5, home.1 - 0.5, home.0 + 0.5, home.1 + 0.5], "box at home".into()),
                2 => ([home.0 - 0.001, home.1 - 0.001, home.0 + 0.001, home.1 + 0.001], "tiny box".into()),
                3 => ([-180.0, -90.0, 180.0, 90.0], "whole world".into()),
                4 => ([-200.0, -95.0, 200.0, 95.0], "past the world".into()),
                5 => ([170.0, home.1 - 10.0, 190.0, home.1 + 10.0], "over the antimeridian".into()),
                _ => ([-180.0, 80.0f64.copysign(home.1).min(90.0f64.copysign(home.1)), 180.0, 80.0f64.copysign(home.1).max(90.0f64.copysign(home.1))], "polar cap".into()),
            }
        };
        let (grid_w, grid_h) = [(256, 256), (512, 512), (1024, 1024), (4096, 64), (37, 1000), (10, 10), (2048, 2048), (1531, 311)][pick(8)];
        let block = [grid_w, 512.min(grid_w), 100.min(grid_w)][pick(3)];
        let (dx, dy) = ((bounds[2] - bounds[0]) / grid_w as f64, (bounds[3] - bounds[1]) / grid_h as f64);
        let project = |across: f64, up: f64| {
            if mercator && matches!(crs, CrsModel::Epsg3857) {
                return (across, up);
            }
            let (lon, lat) = if mercator { mercator_to_lonlat_deg(across, up) } else { (across, up) };
            crs.from_lonlat_deg(lon, lat)
        };
        // The raster: its pixels a chosen number of times the picture's, turned, and
        // either endless or small enough that its edges cross the picture.
        let centre = {
            let h = if mercator && !matches!(crs, CrsModel::Epsg3857) { crs.from_lonlat_deg(home.0, home.1) } else if mercator { lonlat_deg_to_mercator(home.0, home.1) } else { crs.from_lonlat_deg(home.0, home.1) };
            if h.0.is_finite() && h.1.is_finite() { h } else { (0.0, 0.0) }
        };
        let (mid_x, mid_y) = ((bounds[0] + bounds[2]) / 2.0, (bounds[1] + bounds[3]) / 2.0);
        let (p, q) = (project(mid_x, mid_y), project(mid_x + dx, mid_y));
        let step = (q.0 - p.0).hypot(q.1 - p.1);
        let step = if step.is_finite() && step > 0.0 { step } else { 10.0 };
        let ratio = [0.05, 0.7, 1.0, 1.7, 30.0][pick(5)];
        let res = step * ratio;
        let turn = [0.0f64, 0.0, 30.0, 90.0, -137.0][pick(5)].to_radians();
        let side = [1u32 << 20, 1000, 257][pick(3)];
        let half = f64::from(side) / 2.0;
        let (cos, sin) = (turn.cos() * res, turn.sin() * res);
        // pixel -> crs: turned, north up when not turned, the home at the middle.
        let to_crs = Affine2D { a: cos, b: sin, c: centre.0 - cos * half - sin * half, d: sin, e: -cos, f: centre.1 - sin * half + cos * half };
        let Some(to_pixel) = to_crs.invert() else { continue };

        let rows: Vec<usize> = if grid_h <= 24 { (0..grid_h).collect() } else { (0..24).map(|k| k * (grid_h - 1) / 23).collect() };
        let mut row_positions = vec![(0f64, 0f64); grid_w];
        for &j in &rows {
            let up = bounds[3] - (j as f64 + 0.5) * dy;
            let exact = |i: usize| {
                let (x, y) = project(bounds[0] + (i as f64 + 0.5) * dx, up);
                to_pixel.forward(x, y)
            };
            row_source_positions(&exact, block, true, &mut row_positions);
            for (i, &drawn) in row_positions.iter().enumerate() {
                let real = exact(i);
                let got = source_cell(drawn, (side, side), true, || exact(i));
                let want = source_cell(real, (side, side), true, || real);
                assert_eq!(
                    got, want,
                    "case {case}: {name} raster at {home:?}, {view_name}, {grid_w}x{grid_h} in stretches of {block}, \
                     pixels {ratio} times the picture's, turned {:.0}, {side} a side: row {j} column {i} drawn {drawn:?}, exactly {real:?}",
                    turn.to_degrees()
                );
                pixels += 1;
                // How far a drawn position strays matters where it could name a pixel.
                let reach = f64::from(side) + 2.0;
                if drawn.0.is_finite() && drawn.1.is_finite() && real.0 > -2.0 && real.1 > -2.0 && real.0 < reach && real.1 < reach {
                    worst_off = worst_off.max((drawn.0 - real.0).abs().max((drawn.1 - real.1).abs()));
                }
            }
        }
    }
    assert!(pixels > 1_000_000, "{pixels} pixels");
    // The check at a source pixel's edge allows for twice the tolerance.
    assert!(worst_off <= 2.0 * PICTURE_TOLERANCE, "a position was drawn {} tolerances out", worst_off / PICTURE_TOLERANCE);
}

/// What a row may do that no map projection does smoothly: jump (a wrapped longitude),
/// turn a corner (a clamped latitude), or leave the projection at a single column.
/// Wherever the curve is drawn all the same, it must be within the allowance the edge
/// check is built on, or the pixel next door could be taken.
#[test]
fn rows_that_jump_turn_or_break_are_not_drawn_across() {
    let allowed = 2.0 * PICTURE_TOLERANCE;
    let (mut worst_jump, mut worst_kink) = (0f64, 0f64);
    for &points in &[10usize, 11, 64, 255, 256, 257, 512, 1000] {
        let n = (points - 1) as f64;
        let mut row = vec![(0.0, 0.0); points];
        // A jump of every size around the tolerance, after every column, on a row that
        // is otherwise straight, bowed, or bent so as to hide the jump from the check.
        for at in 0..points - 1 {
            for &jump in &[0.2, 0.5, 0.9, 1.0, 1.5, 2.0, 2.6, 3.0, 5.0, 7.9, 12.0, 100.0] {
                for &bend in &[0.0, 1.0, -2.0 / 3.0, 2.0 / 3.0, -2.0, 6.0] {
                    let jump = jump * PICTURE_TOLERANCE;
                    let real = |i: usize| {
                        let x = i as f64 / n;
                        // A cubic through the three fitted columns, sized against the jump.
                        let hidden = bend * jump * 8.0 * x * (x - 0.5) * (x - 1.0);
                        let step = if i > at { jump } else { 0.0 };
                        (i as f64 * 0.83 + 0.4 * x * x + step + hidden, 100.0 + 3.0 * x - step)
                    };
                    curved_scanline(&real, &mut row, PICTURE_TOLERANCE);
                    for (i, drawn) in row.iter().enumerate() {
                        let want = real(i);
                        worst_jump = worst_jump.max((drawn.0 - want.0).abs().max((drawn.1 - want.1).abs()));
                    }
                }
            }
            // A corner at every column.
            for &turn in &[0.5, 1.0, 3.0, 10.0, 40.0, 1000.0] {
                let turn = turn * PICTURE_TOLERANCE / n;
                let real = |i: usize| (i as f64 * 1.3, 7.0 + turn * (i as f64 - at as f64 - 0.5).abs() * 4.0);
                curved_scanline(&real, &mut row, PICTURE_TOLERANCE);
                for (i, drawn) in row.iter().enumerate() {
                    worst_kink = worst_kink.max((drawn.1 - real(i).1).abs());
                }
            }
            // A stretch of columns outside the projection, long enough that it cannot
            // lie between two of the columns a curve is fitted and checked at, in one
            // coordinate or both: nothing may be drawn across it.
            for hole in [(f64::NAN, 1.0), (1.0, f64::NAN), (f64::INFINITY, 1.0), (f64::NAN, f64::NAN)] {
                let end = (at + points / 4 + 2).min(points);
                let real = |i: usize| if (at..end).contains(&i) { hole } else { (i as f64 * 0.5, 9.0) };
                curved_scanline(&real, &mut row, PICTURE_TOLERANCE);
                let same = |a: f64, b: f64| a == b || (a.is_nan() && b.is_nan());
                for (i, drawn) in row.iter().enumerate() {
                    let want = real(i);
                    assert!(same(drawn.0, want.0) && same(drawn.1, want.1), "{points} points, hole {hole:?} from {at}: column {i} drew {drawn:?}");
                }
            }
        }
    }
    assert!(
        worst_jump <= allowed && worst_kink <= allowed,
        "drawn {} tolerances out after a jump, {} after a corner",
        worst_jump / PICTURE_TOLERANCE,
        worst_kink / PICTURE_TOLERANCE
    );
}

#[test]
fn item_pictures_with_options_on_assets_go_to_the_search_routes() {
    let by_asset = |rest: &[&str], q: &str| item_picture_by_asset(rest, &query_pairs(q));
    let tile = ["tiles", "WebMercatorQuad", "11", "1059", "746.png"];
    // An array, as `.npy` or a GeoTIFF, whatever its assets: of a tile or a box.
    for array in ["746.npy", "746.tif", "746.tiff"] {
        assert!(by_asset(&["tiles", "WebMercatorQuad", "11", "1059", array], "assets=red"), "{array}");
    }
    assert!(by_asset(&["tiles", "WebMercatorQuad", "11", "1059", "746"], "assets=red&format=npy"));
    assert!(by_asset(&["bbox", "6.3,43.6,6.5,43.8.npy"], "assets=red"));
    assert!(by_asset(&["bbox", "6.3,43.6,6.5,43.8", "64x64.tif"], "assets=red"));
    assert!(!by_asset(&["bbox", "6.3,43.6,6.5,43.8", "64x64.png"], "assets=red"));
    assert!(!by_asset(&["bbox", "6.3,43.6,6.5,43.8", "64x64"], "assets=red&format=npy"));
    assert!(!by_asset(&["preview.npy"], "assets=red"));
    assert!(!by_asset(&tile, "assets=red"));
    // Band math on an asset, and bands chosen on more than one.
    assert!(by_asset(&tile, "assets=red%7Cexpression=b1*2"));
    assert!(by_asset(&tile, "assets=visual%7Cbidx=1&assets=red%7Cbidx=1"));
    assert!(by_asset(&tile, "assets=visual%7Cbidx=3,1&assets=red"));
    assert!(by_asset(&["bbox", "6,43,7,44.png"], "assets=red%7Cexpression=b1*2"));
    assert!(by_asset(&["bbox", "6,43,7,44", "256x256.png"], "assets=red%7Cexpression=b1*2"));
    // What `/stac` draws stays there: plain assets, and bands of a single one.
    assert!(!by_asset(&tile, "assets=visual"));
    assert!(!by_asset(&tile, "assets=red&assets=green&expression=b1/b2"));
    assert!(!by_asset(&tile, "assets=visual%7Cbidx=1"));
    // So does every other route, and a tile in another tile matrix set.
    for rest in [&["preview.png"][..], &["info"], &["statistics"], &["point", "6,43"], &["bbox", "6,43,7,44", "assets"]] {
        assert!(!by_asset(rest, "assets=red%7Cexpression=b1*2"), "{rest:?}");
    }
    assert!(!by_asset(&["tiles", "WGS1984Quad", "11", "1059", "746.png"], "assets=red%7Cexpression=b1*2"));
    // Options that cannot be read are answered for by the route they were sent to.
    assert!(!by_asset(&tile, "assets=red%7Cbidx=x&assets=green"));
    // A box without a format in its path is not one of an item's routes.
    assert!(!by_asset(&["bbox", "6,43,7,44"], "assets=red%7Cexpression=b1*2"));
    assert!(!by_asset(&["bbox", "6,43,7,44", "64x64"], "assets=red%7Cexpression=b1*2"));
    // What only a search reads is not shown to the search routes.
    assert_eq!(
        without_search_options("assets=red%7Cexpression=b1*2&pixel_selection=count&items_limit=many&rescale=0,9&scan_limit=-1&algorithm_params={}"),
        "assets=red%7Cexpression=b1*2&rescale=0,9"
    );
    assert_eq!(without_search_options("algorithm=hillshade&algorithm_params={}"), "algorithm=hillshade&algorithm_params={}");
}

#[test]
fn an_array_asked_for_as_a_geotiff_is_one() {
    let npy = npy_array(&[1.0, f32::NAN, 3.0, 4.0], 2, 2, 1, SampleType::Uint16, true, None);
    let (tiff, kind) = as_geotiff((npy.clone(), "application/x-binary"), true, [8.0, 45.0, 9.0, 46.0], 4326, true);
    assert_eq!(kind, super::super::GEOTIFF_CONTENT_TYPE);
    assert_eq!(&tiff[..4], b"II*\0");
    // Two bands and the mask, in strips of the array's own bytes.
    assert_eq!(&tiff[tiff.len() - 12..], &npy[npy.len() - 12..]);
    let entries = u16::from_le_bytes([tiff[8], tiff[9]]);
    assert_eq!(entries, 15);
    // Anything else is left as it was made.
    let (same, kind) = as_geotiff((npy.clone(), "application/x-binary"), false, [8.0, 45.0, 9.0, 46.0], 4326, true);
    assert_eq!((same, kind), (npy, "application/x-binary"));
    let png = Bytes::from_static(b"\x89PNG....");
    assert_eq!(as_geotiff((png.clone(), "image/png"), true, [0.0, 0.0, 1.0, 1.0], 3857, true).0, png);
}

#[test]
fn assets_take_titilers_options() {
    let parse = |q: &str| parse_assets(&query_pairs(q));
    let spec = |name: &str, indexes: Option<Vec<u16>>, expression: Option<&str>| AssetSpec {
        name: name.into(),
        indexes,
        expression: expression.map(str::to_string),
    };
    assert_eq!(parse("assets=red&assets=nir").unwrap(), [spec("red", None, None), spec("nir", None, None)]);
    // Options keep their commas; a plain entry may still list names with them.
    assert_eq!(
        parse("assets=visual%7Cbidx%3D3,2,1&assets=red,nir").unwrap(),
        [spec("visual", Some(vec![3, 2, 1]), None), spec("red", None, None), spec("nir", None, None)]
    );
    assert_eq!(
        parse("assets=red%7Cexpression%3Db1*2%7Cnope%3D1%7Cbands%3Dx").unwrap(),
        [spec("red", None, Some("b1*2"))]
    );
    for bad in ["assets=visual%7Cbidx", "assets=visual%7Cbidx%3Dx", "assets=visual%7Cbidx%3D1,"] {
        assert_eq!(parse(bad).unwrap_err().status(), StatusCode::UNPROCESSABLE_ENTITY, "{bad}");
    }
    assert_eq!(parse("assets=%7Cbidx%3D1").unwrap_err().status(), StatusCode::BAD_REQUEST);

    // What an asset gives, and how it is described.
    let pixel = |spec: &AssetSpec, as_band: bool| {
        let (plan, described) = asset_bands(spec, 3, as_band)?;
        let mut out = Vec::new();
        plan.pixel(&[109.0, 79.0, 46.0], &mut out);
        Ok::<_, String>((out, described))
    };
    let names = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    assert_eq!(pixel(&spec("visual", None, None), false).unwrap(), (vec![109.0, 79.0, 46.0], names(&["visual_b1", "visual_b2", "visual_b3"])));
    assert_eq!(pixel(&spec("visual", Some(vec![3, 1]), None), false).unwrap(), (vec![46.0, 109.0], names(&["visual_b3", "visual_b1"])));
    assert_eq!(pixel(&spec("visual", None, Some("b1+b2;b3")), false).unwrap(), (vec![188.0, 46.0], names(&["visual_b1+b2", "visual_b3"])));
    // Band math wins over band indexes, and sees the asset's own bands.
    assert_eq!(pixel(&spec("visual", Some(vec![3]), Some("b1*2")), false).unwrap().0, [218.0]);
    assert_eq!(pixel(&spec("visual", Some(vec![2]), None), true).unwrap(), (vec![79.0], names(&["visual"])));
    assert_eq!(pixel(&spec("visual", None, None), true).unwrap_err(), "Can't use `asset_as_band` for multibands asset");
    assert_eq!(pixel(&spec("visual", Some(vec![9]), None), false).unwrap_err(), "band index 9 out of range (not in (1, 2, 3))");
    assert_eq!(asset_bands(&spec("red", Some(vec![0]), None), 1, false).err().unwrap(), "band index 0 out of range (not in (1,))");
    // Whole planes go the same way.
    let (plan, _) = asset_bands(&spec("v", None, Some("b2/b1")), 2, false).unwrap();
    let planes = plan.planes(&[vec![2.0, 0.0], vec![1.0, 1.0]]);
    assert_eq!(planes[0][0], 0.5);
    assert_eq!(planes[0][1], f64::MAX);
}

#[test]
fn the_renderer_gets_what_it_can_draw() {
    let run = |q: &str| {
        let mut pq = super::super::parse_query(Some(q));
        let pairs = query_pairs(q);
        let query = check_mosaic_query(&pq, &pairs).map_err(|r| r.status())?;
        let names = renderer_assets(&mut pq, &query).map_err(|r| r.status())?;
        Ok::<_, StatusCode>((names, pq.bidx, pq.expr_raw, query.asset_as_band))
    };
    let names = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    assert_eq!(run("assets=visual").unwrap(), (names(&["visual"]), vec![], None, false));
    // A bare `bidx` is passed over, as in TiTiler 2; an asset's own is used.
    assert_eq!(run("assets=visual&bidx=2").unwrap().1, Vec::<u16>::new());
    assert_eq!(run("assets=visual%7Cbidx%3D3,2,1").unwrap(), (names(&["visual"]), vec![3, 2, 1], None, false));
    assert_eq!(run("assets=red%7Cexpression%3Db1*2").unwrap().2.as_deref(), Some("b1*2"));
    assert_eq!(
        run("assets=red%7Cbidx%3D1&assets=nir&asset_as_band=true&expression=b2-b1").unwrap(),
        (names(&["red", "nir"]), vec![], Some("b2-b1".to_string()), true)
    );
    // Band math is over b1..bn; asset names are refused as TiTiler refuses them.
    assert_eq!(run("assets=red&assets=nir&expression=nir-red").unwrap_err(), StatusCode::BAD_REQUEST);
    // What the renderer cannot draw is refused, not drawn differently. On tiles all of
    // it is made from planes instead, and nothing else is.
    let planes = |q: &str| {
        let pq = super::super::parse_query(Some(q));
        wants_planes(&pq, &check_mosaic_query(&pq, &query_pairs(q)).unwrap())
    };
    for q in [
        "assets=visual",
        "assets=visual%7Cbidx%3D3,2,1",
        "assets=red%7Cexpression%3Db1*2",
        "assets=red&assets=green&assets=blue",
        "assets=red%7Cbidx%3D1&assets=nir&asset_as_band=true&expression=b2-b1",
        "assets=visual&pixel_selection=median",
    ] {
        assert!(run(q).is_ok() && !planes(q), "{q}");
    }
    for q in [
        "assets=red&assets=nir",
        "assets=red&assets=green&assets=blue&assets=nir",
        "assets=visual&pixel_selection=lastbandlow",
        "assets=red&assets=nir&expression=b1;b2",
    ] {
        assert!(planes(q), "{q}");
    }
    for q in [
        "assets=visual%7Cbidx%3D2&assets=nir",
        "assets=visual%7Cbidx%3D1,2",
        "assets=visual%7Cbidx%3D1&expression=b1*2",
        "assets=visual%7Cexpression%3Db1;b2",
        "assets=red%7Cexpression%3Db1*2&assets=nir",
        "assets=red%7Cexpression%3Db1*2&expression=b1%2B1",
    ] {
        assert_eq!(run(q).unwrap_err(), StatusCode::BAD_REQUEST, "{q}");
        assert!(planes(q), "{q}");
    }
    assert_eq!(run("assets=red&asset_as_band=perhaps").unwrap_err(), StatusCode::UNPROCESSABLE_ENTITY);
    // An `assets` that names nothing is a missing parameter.
    assert_eq!(run("assets=,").unwrap_err(), StatusCode::UNPROCESSABLE_ENTITY);
}

#[test]
fn statistics_options_are_titilers() {
    let pairs = |q: &str| query_pairs(q);
    let (opts, cover) = stats_options(&pairs("assets=red")).unwrap();
    assert_eq!((opts.categorical, opts.percentiles.clone(), opts.bins.clone(), opts.range, cover),
               (false, vec![2, 98], Bins::Count(10), None, 10));
    let (opts, cover) = stats_options(&pairs(
        "categorical=true&c=1&c=2.5&p=5&p=95&histogram_bins=4&histogram_range=0,2000.5&cover_scale=4",
    ))
    .unwrap();
    assert!(opts.categorical);
    assert_eq!(opts.categories, [1.0, 2.5]);
    assert_eq!(opts.percentiles, [5, 95]);
    assert_eq!(opts.bins, Bins::Count(4));
    assert_eq!(opts.range, Some((0.0, 2000.5)));
    assert_eq!(cover, 4);
    let (opts, _) = stats_options(&pairs("histogram_bins=0,100,255")).unwrap();
    assert_eq!(opts.bins, Bins::Edges(vec![0.0, 100.0, 255.0]));
    for bad in [
        "categorical=maybe", "c=x", "p=101", "p=1.5", "histogram_bins=ten", "histogram_bins=0",
        "histogram_bins=3,2,1", "histogram_range=5", "histogram_range=9,1", "cover_scale=0",
        "cover_scale=1001",
    ] {
        let resp = stats_options(&pairs(bad)).unwrap_err();
        assert_eq!(resp.status(), StatusCode::UNPROCESSABLE_ENTITY, "{bad}");
    }
}

#[test]
fn tile_limits_cover_the_bounds() {
    let limits = |z, b| tile_limits(z, b);
    assert_eq!(
        limits(8, [5.0, 43.5, 12.0, 48.0]),
        json!({ "tileMatrix": "8", "minTileRow": 88, "maxTileRow": 93, "minTileCol": 131, "maxTileCol": 136 })
    );
    assert_eq!(
        limits(7, [6.0, 44.0, 8.0, 46.0]),
        json!({ "tileMatrix": "7", "minTileRow": 45, "maxTileRow": 46, "minTileCol": 66, "maxTileCol": 66 })
    );
    // The whole world, including latitudes past the Web Mercator limit, stays on the grid.
    assert_eq!(
        limits(0, [-180.0, -90.0, 180.0, 90.0]),
        json!({ "tileMatrix": "0", "minTileRow": 0, "maxTileRow": 0, "minTileCol": 0, "maxTileCol": 0 })
    );
    assert_eq!(
        limits(3, [-180.0, -90.0, 180.0, 90.0]),
        json!({ "tileMatrix": "3", "minTileRow": 0, "maxTileRow": 7, "minTileCol": 0, "maxTileCol": 7 })
    );
}

#[test]
fn query_strings_are_written_like_python() {
    assert_eq!(quote_plus("(b1-b2)/(b1+b2)"), "%28b1-b2%29%2F%28b1%2Bb2%29");
    assert_eq!(quote_plus("a b~c_d.e"), "a+b~c_d.e");
}

#[test]
fn tile_coordinates_are_parsed() {
    assert!(parse_zxy("3", "7", 7).unwrap().is_some());
    assert!(parse_zxy("3", "8", 0).unwrap().is_none());
    assert!(parse_zxy("31", "0", 0).unwrap().is_none());
    assert_eq!(parse_zxy("x", "0", 0).unwrap_err().status(), StatusCode::UNPROCESSABLE_ENTITY);
}
#[test]
fn bounds_are_written_as_python_writes_floats() {
    for (given, written) in [
        (6.0, "6.0"), (43.580390855607845, "43.580390855607845"), (-0.0, "0.0"), (0.00001, "1e-05"),
        (-0.00001234, "-1.234e-05"), (0.0001, "0.0001"), (1e16, "1e+16"), (1234567890123456.0, "1234567890123456.0"),
        (-20037508.342789244, "-20037508.342789244"),
    ] {
        assert_eq!(python_float(given), written);
    }
}

#[test]
fn a_nodata_the_samples_cannot_hold_is_refused() {
    use cogzilla_raster::SampleType as T;
    assert!(nodata_fits(1000.0, T::Uint16) && nodata_fits(1000.5, T::Uint16) && nodata_fits(-1.0, T::Int16));
    for bad in [f32::NAN, f32::INFINITY, -1.0, 70000.0] {
        assert!(!nodata_fits(bad, T::Uint16));
    }
    assert!(!nodata_fits(256.0, T::Uint8) && nodata_fits(f32::NAN, T::Float32));
}

#[test]
fn item_pictures_say_where_they_lie() {
    let ok = || Response::builder().status(StatusCode::OK).body(Full::new(Bytes::new())).unwrap();
    let said = |sub: &[&str], query: &str| {
        let resp = item_placed(ok(), sub, query);
        let header = |name: &str| resp.headers().get(name).map(|v| v.to_str().unwrap().to_string());
        header("content-bbox").zip(header("content-crs"))
    };
    let crs = |epsg: u16| format!("<http://www.opengis.net/def/crs/EPSG/0/{epsg}>");
    let (bbox, system) = said(&["tiles", "WebMercatorQuad", "0", "0", "0.png"], "assets=visual").unwrap();
    assert_eq!(system, crs(3857));
    assert!(bbox.starts_with("-20037508.34278") && bbox.split(',').count() == 4);
    assert_eq!(said(&["bbox", "6.3,43.6,6.5,43.8.png"], "assets=visual").unwrap(), ("6.3,43.6,6.5,43.8".into(), crs(4326)));
    assert_eq!(
        said(&["bbox", "701000,5404000,720000,5420000", "64x64.png"], "coord_crs=epsg:3857").unwrap(),
        ("701000.0,5404000.0,720000.0,5420000.0".into(), crs(3857))
    );
    assert_eq!(said(&["bbox", "6.3,43.6,6.5,43.8.png"], "dst_crs=EPSG%3A3857").unwrap().1, crs(3857));
    assert_eq!(said(&["bbox", "701000,5404000,720000,5420000.png"], "coord_crs=epsg:3857&dst_crs=epsg:4326").unwrap().1, crs(4326));
    assert_eq!(said(&["bbox", "6.3,43.6,6.5,43.8.png"], "dst_crs=epsg:32631").unwrap().1, crs(32631));
    assert_eq!(said(&["bbox", "765000,4832000,782000,4855000.png"], "coord_crs=epsg:32631").unwrap().1, crs(32631));
    // Unsupported systems, tiles that are not tiles, and other routes go without.
    for (sub, query) in [
        (&["bbox", "6.3,43.6,6.5,43.8.png"][..], "dst_crs=epsg:4275"),
        (&["tiles", "WebMercatorQuad", "1", "2", "0.png"][..], ""),
        (&["tiles", "WGS1984Quad", "0", "0", "0.png"][..], ""),
        (&["preview.png"][..], ""),
        (&["info"][..], ""),
    ] {
        assert_eq!(said(sub, query), None, "{sub:?} {query}");
    }
    // A percent-encoded key is the key it spells, as the handlers read it.
    assert_eq!(said(&["bbox", "6.3,43.6,6.5,43.8.png"], "dst%5Fcrs=EPSG:3857").unwrap().1, crs(3857));
    assert_eq!(said(&["bbox", "6.3,43.6,6.5,43.8.png"], "dst_crs=epsg:3857&dst_crs=epsg:4326").unwrap().1, crs(4326));
    // Degrees asked for as CRS84 are named so; a buffered tile is not placed.
    let ogc = "<http://www.opengis.net/def/crs/OGC/0/CRS84>";
    assert_eq!(said(&["bbox", "6.3,43.6,6.5,43.8.png"], "coord_crs=OGC:CRS84").unwrap().1, ogc);
    assert_eq!(said(&["bbox", "701000,5404000,720000,5420000.png"], "coord_crs=epsg:3857&dst_crs=OGC%3ACRS84").unwrap().1, ogc);
    assert_eq!(said(&["tiles", "WebMercatorQuad", "0", "0", "0.png"], "buffer=10"), None);
    assert!(said(&["tiles", "WebMercatorQuad", "0", "0", "0.png"], "buffer=0").is_some());
    let missing = Response::builder().status(StatusCode::NOT_FOUND).body(Full::new(Bytes::new())).unwrap();
    assert!(item_placed(missing, &["bbox", "6.3,43.6,6.5,43.8.png"], "").headers().get("content-bbox").is_none());
    // `respelled` preserves arbitrary EPSG codes from upstream Content-Crs headers.
    let utm_tile = Response::builder()
        .status(StatusCode::OK)
        .header("content-bbox", "500000,4800000,510000,4810000")
        .header("content-crs", "<http://www.opengis.net/def/crs/EPSG/0/32631>")
        .body(Full::new(Bytes::new()))
        .unwrap();
    let respelled_utm = item_placed(utm_tile, &["tiles", "UTM31N", "0", "0", "0.png"], "");
    assert_eq!(
        respelled_utm.headers().get("content-crs").unwrap().to_str().unwrap(),
        crs(32631)
    );
    assert_eq!(
        respelled_utm.headers().get("content-bbox").unwrap().to_str().unwrap(),
        "500000.0,4800000.0,510000.0,4810000.0"
    );
}

#[test]
fn rgba_to_npy_u8_writes_chw_planes_and_optional_mask() {
    // 2x1 RGBA image: pixel 0 opaque (10,20,30,255), pixel 1 transparent (40,50,60,0).
    let rgba = [10u8, 20, 30, 255, 40, 50, 60, 0];
    let with_mask = rgba_to_npy_u8(&rgba, 2, 1, false, true);
    assert!(with_mask.starts_with(b"\x93NUMPY"));
    let hdr_len = 10 + u16::from_le_bytes([with_mask[8], with_mask[9]]) as usize;
    let hdr = std::str::from_utf8(&with_mask[10..hdr_len]).unwrap();
    assert!(hdr.contains("'|u1'"));
    assert!(hdr.contains("(4, 1, 2)"));
    assert_eq!(&with_mask[hdr_len..], &[10, 40, 20, 50, 30, 60, 255, 0]);

    let no_mask = rgba_to_npy_u8(&rgba, 2, 1, false, false);
    let hdr_len2 = 10 + u16::from_le_bytes([no_mask[8], no_mask[9]]) as usize;
    let hdr2 = std::str::from_utf8(&no_mask[10..hdr_len2]).unwrap();
    assert!(hdr2.contains("(3, 1, 2)"));
    assert_eq!(&no_mask[hdr_len2..], &[10, 40, 20, 50, 30, 60]);

    // Grey 1-band + mask and short-buffer zero-padding safety.
    let grey_short = rgba_to_npy_u8(&[77, 77, 77, 255], 2, 1, true, true);
    let hdr_len3 = 10 + u16::from_le_bytes([grey_short[8], grey_short[9]]) as usize;
    let hdr3 = std::str::from_utf8(&grey_short[10..hdr_len3]).unwrap();
    assert!(hdr3.contains("(2, 1, 2)"));
    assert_eq!(&grey_short[hdr_len3..], &[77, 0, 255, 0]);
}

#[test]
fn a_normalized_index_over_assets_is_band_math() {
    let asked = |query: &str, ext: Option<&str>| {
        let mut pq = super::super::parse_query(Some(query));
        let index = normalized_index_as_band_math(&mut pq, ext);
        assert_eq!(index, pq.algorithm.is_none(), "{query}");
        (pq.algorithm, pq.expr_raw, pq.rescale)
    };
    let math = (None, Some("(b2-b1)/(b2+b1)".to_string()));
    let (algorithm, expression, rescale) = asked("assets=red&assets=nir&algorithm=normalizedIndex", Some("png"));
    assert_eq!((algorithm, expression), math.clone());
    assert_eq!(rescale, vec![(-1.0, 1.0)]);
    // A range asked for is kept; without an extension the picture is still a picture.
    let (algorithm, expression, rescale) = asked("assets=red&assets=nir&algorithm=normalizedIndex&rescale=0,1", None);
    assert_eq!((algorithm, expression), math.clone());
    assert_eq!(rescale, vec![(0.0, 1.0)]);
    // An array is the values as they are, with no range put in.
    for (query, ext) in [
        ("assets=red&assets=nir&algorithm=normalizedIndex", Some("npy")),
        ("assets=red&assets=nir&algorithm=normalizedIndex&format=tif", None),
    ] {
        let (algorithm, expression, rescale) = asked(query, ext);
        assert_eq!((algorithm, expression), math.clone(), "{query}");
        assert!(rescale.is_empty(), "{query}");
    }
    // One asset is the renderer's; a bare array is the algorithm's; band math asked for wins.
    for (query, ext) in [
        ("assets=visual&algorithm=normalizedIndex", Some("png")),
        ("assets=red&assets=nir&algorithm=normalizedIndex", Some("bin")),
        ("assets=red&assets=nir&algorithm=normalizedIndex&expression=b1", Some("png")),
        ("assets=red&assets=nir&algorithm=hillshade", Some("png")),
    ] {
        let (algorithm, _, _) = asked(query, ext);
        assert!(algorithm.is_some(), "{query}");
    }
}

#[test]
fn algorithms_on_a_mosaic_s_bands() {
    let nan = f64::NAN;
    let bands = vec![vec![1.0, 10.0, nan], vec![2.0, nan, nan], vec![4.0, 20.0, nan]];
    let run = |a: BandAlgorithm, whole| a.apply(bands.clone(), whole, 3, (1.0, 1.0)).unwrap();
    let first = |a: BandAlgorithm| run(a, true).0.remove(0);
    assert_eq!(first(BandAlgorithm::Min)[..2], [1.0, 10.0]);
    assert_eq!(first(BandAlgorithm::Sum)[..2], [7.0, 30.0]);
    assert_eq!(first(BandAlgorithm::Mean)[..2], [7.0 / 3.0, 15.0]);
    // numpy's median of an even count is the mean of the two in the middle.
    assert_eq!(first(BandAlgorithm::Median)[..2], [2.0, 15.0]);
    // One degree of freedom less: 1, 2, 4 about 7/3.
    let var = first(BandAlgorithm::Var);
    assert!((var[0] - 7.0 / 3.0).abs() < 1e-12 && (var[1] - 50.0).abs() < 1e-12);
    assert!(first(BandAlgorithm::Std)[2].is_nan());
    // Luma, in the bands' own type: 0.299 + 1.174 + 0.456.
    assert_eq!(first(BandAlgorithm::Grayscale)[0], 1.0);
    assert!((run(BandAlgorithm::Grayscale, false).0[0][0] - 1.929).abs() < 1e-12);
    assert!(BandAlgorithm::Grayscale.apply(vec![vec![1.0]], true, 1, (1.0, 1.0)).is_err());
    assert_eq!(BandAlgorithm::Bitonal.apply(vec![vec![127.0, 128.0, nan]], true, 1, (1.0, 1.0)).unwrap().0[0][..2], [0.0, 255.0]);
    // To bytes as `astype("uint8")` wraps, keeping the bands' descriptions.
    let (cast, described) = BandAlgorithm::Cast.apply(vec![vec![300.0, -1.0, 255.9]], true, 1, (1.0, 1.0)).unwrap();
    assert_eq!((cast[0].as_slice(), described.len()), (&[44.0, 255.0, 255.0][..], 0));
    assert_eq!(BandAlgorithm::Ceil.apply(vec![vec![255.5]], false, 1, (1.0, 1.0)).unwrap().0[0], [0.0]);
    // Terrain-RGB of 100 m: (100 + 10000) / 0.1 = 101000 = 0x018A88.
    let (rgb, names) = BandAlgorithm::TerrainRgb { interval: 0.1, baseval: -10000.0 }.apply(vec![vec![100.0]], true, 1, (1.0, 1.0)).unwrap();
    assert_eq!(rgb, [vec![1.0], vec![138.0], vec![136.0]]);
    assert_eq!(names, ["terrain_r", "terrain_g", "terrain_b"]);
    // Terrarium of 100 m: 32868 = 128 * 256 + 100.
    assert_eq!(BandAlgorithm::Terrarium.apply(vec![vec![100.0]], true, 1, (1.0, 1.0)).unwrap().0, [vec![128.0], vec![100.0], vec![0.0]]);
    // `aspect` is parsed and applied (matching `cogzilla-kernel::DemAlgorithm::Aspect`).
    assert_eq!(BandAlgorithm::parse("aspect", None), Ok(Some(BandAlgorithm::Aspect { buffer: 3, z: 1.0 })));
    assert!(BandAlgorithm::parse("aspect", Some(r#"{"buffer":100}"#)).is_err());
    // Elevation decreasing Eastward (left=20, mid=10, right=0) -> aspect = 90° (East).
    let (east, east_names) = BandAlgorithm::Aspect { buffer: 0, z: 1.0 }
        .apply(vec![vec![20.0, 10.0, 0.0, 20.0, 10.0, 0.0, 20.0, 10.0, 0.0]], false, 3, (1.0, 1.0))
        .unwrap();
    assert_eq!(east_names, ["aspect"]);
    assert!((east[0][4] - 90.0).abs() < 1e-4, "expected 90° East, got {}", east[0][4]);
    // Elevation increasing Southward (row 0 = 0, row 1 = 10, row 2 = 20) -> downhill is North (0°).
    let (north, _) = BandAlgorithm::Aspect { buffer: 0, z: 1.0 }
        .apply(vec![vec![0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 20.0, 20.0, 20.0]], false, 3, (1.0, 1.0))
        .unwrap();
    assert!(north[0][4].abs() < 1e-4, "expected 0° North, got {}", north[0][4]);
    // Flat plane -> NaN aspect.
    let (flat, _) = BandAlgorithm::Aspect { buffer: 0, z: 1.0 }
        .apply(vec![vec![5.0; 9]], false, 3, (1.0, 1.0))
        .unwrap();
    assert!(flat[0][4].is_nan());
    assert!(BandAlgorithm::parse("hillshade", Some(r#"{"azimuth":400}"#)).is_err());
    // A fractional increment is taken, as on the shared routes; one past TiTiler's bound is not.
    assert!(matches!(BandAlgorithm::parse("contours", Some(r#"{"increment":2.5}"#)), Ok(Some(BandAlgorithm::Contours { increment, .. })) if increment == 2.5));
    assert!(BandAlgorithm::parse("contours", Some(r#"{"increment":1000}"#)).is_err());
    assert_eq!(BandAlgorithm::parse("terrainrgb", Some(r#"{"interval":1}"#)), Ok(Some(BandAlgorithm::TerrainRgb { interval: 1.0, baseval: -10000.0 })));
    assert_eq!(BandAlgorithm::parse("terrain-rgb", None), Ok(Some(BandAlgorithm::TerrainRgb { interval: 0.1, baseval: -10000.0 })));
    assert_eq!(BandAlgorithm::parse("mapbox", None), Ok(Some(BandAlgorithm::TerrainRgb { interval: 0.1, baseval: -10000.0 })));
    // numpy's gradient: one-sided on the edges, centred inside.
    let (down, across) = gradient(&[0.0, 1.0, 4.0, 2.0, 4.0, 8.0], 3, 2).unwrap();
    assert_eq!(down, [2.0, 3.0, 4.0, 2.0, 3.0, 4.0]);
    assert_eq!(across, [1.0, 2.0, 3.0, 2.0, 3.0, 4.0]);
    assert_eq!(crop((0..16).map(f64::from).collect(), 4, 4, 1), [5.0, 6.0, 9.0, 10.0]);
}

#[test]
fn every_advertised_algorithm_is_applied_on_both_route_stacks() {
    use crate::algorithms::{ALGORITHMS, AlgorithmId};
    use crate::engine::BandRequest;
    use cogzilla_raster::SampleType;

    // 8x8 sloped plane so buffered DEM algorithms (default buffer=3 -> 2x2 inner crop) succeed.
    let plane = |offset: f64| -> Vec<f64> {
        (0..64).map(|i| offset + (i % 8) as f64 * 10.0 + (i / 8) as f64 * 5.0).collect()
    };
    let three_bands = vec![plane(10.0), plane(50.0), plane(100.0)];
    let one_band = vec![plane(10.0)];

    for def in ALGORITHMS {
        // 1. COG/STAC route stack (`http.rs::build_band_request`): never silently falls back to raw bands.
        let pq = super::super::parse_query(Some(&format!("algorithm={}", def.name)));
        let req = super::super::build_band_request(&pq, 3, SampleType::Float32)
            .unwrap_or_else(|e| panic!("build_band_request failed for '{}': {e}", def.name));
        assert!(
            matches!(
                req,
                BandRequest::Dem { .. }
                    | BandRequest::Terrain { .. }
                    | BandRequest::CustomExpr { .. }
                    | BandRequest::CustomExprRgb { .. }
            ),
            "algorithm '{}' fell through to raw BandRequest: {req:?}",
            def.name
        );

        // 2. pgSTAC route stack (`BandAlgorithm::parse` + `apply`, or `normalizedIndex` band-math path).
        if def.id == AlgorithmId::NormalizedIndex {
            assert_eq!(BandAlgorithm::parse(def.name, None), Ok(None));
            let mut pq_ni = super::super::parse_query(Some("assets=red&assets=nir&algorithm=normalizedIndex"));
            assert!(normalized_index_as_band_math(&mut pq_ni, Some("png")));
            assert_eq!(pq_ni.expr_raw.as_deref(), Some("(b2-b1)/(b2+b1)"));
            continue;
        }
        let parsed = BandAlgorithm::parse(def.name, None)
            .unwrap_or_else(|e| panic!("BandAlgorithm::parse failed for '{}': {e}", def.name))
            .unwrap_or_else(|| panic!("BandAlgorithm::parse silently returned Ok(None) for '{}'", def.name));
        let input = if def.input_bands == Some(1) { one_band.clone() } else { three_bands.clone() };
        let (out_bands, _) = parsed
            .apply(input, false, 8, (1.0, 1.0))
            .unwrap_or_else(|e| panic!("BandAlgorithm::apply failed for '{}': {e}", def.name));
        assert!(!out_bands.is_empty(), "algorithm '{}' produced 0 bands", def.name);
        if let Some(expected_nb) = def.out_bands {
            assert_eq!(
                out_bands.len(),
                usize::from(expected_nb),
                "algorithm '{}' output band count mismatch",
                def.name
            );
        }
    }
}

#[test]
fn a_point_s_numbers_and_a_tileset_s_format() {
    assert_eq!(point_coords("6.73%20,%2043.402").ok(), Some((6.73, 43.402)));
    assert_eq!(point_coords(" 1 , 2 ").ok(), Some((1.0, 2.0)));
    assert!(point_coords("1,abc").is_err());
    let pairs = |f: &str| vec![("f".to_string(), f.to_string())];
    assert!(tileset_format_error(&pairs("html")).is_none());
    assert!(tileset_format_error(&pairs("json")).is_none());
    assert!(tileset_format_error(&[]).is_none());
    for f in ["png", "HTML", ""] {
        assert_eq!(tileset_format_error(&pairs(f)).map(|r| r.status()), Some(StatusCode::UNPROCESSABLE_ENTITY), "{f:?}");
    }
}

#[test]
fn a_float32_band_s_bins_are_numpy_s() {
    let edges = |opts: &StatsOptions, band: &[f64]| match float32_bins(opts, band).bins {
        zonal::Bins::Edges(e) => e,
        zonal::Bins::Count(_) => vec![],
    };
    let band = [f64::from(-0.759_312_33_f32), 0.1, f64::NAN, f64::from(0.759_312_33_f32)];
    // From the band's own ends: float32 arithmetic, the last edge the greatest value.
    let opts = StatsOptions { bins: zonal::Bins::Count(10), ..StatsOptions::default() };
    let from_values = edges(&opts, &band);
    assert_eq!(from_values.len(), 11);
    let (lo, hi) = (-0.759_312_33_f32, 0.759_312_33_f32);
    let step = (hi - lo) / 10.0;
    assert_eq!(from_values[3], f64::from(3.0 * step + lo));
    assert_eq!(from_values[10], f64::from(hi));
    // From a range given: double precision, each edge cast to float32 (-5/7 is one).
    let opts = StatsOptions { bins: zonal::Bins::Count(7), range: Some((-1.0, 1.0)), ..StatsOptions::default() };
    assert_eq!(edges(&opts, &band)[1], f64::from((-1.0 + 2.0 / 7.0) as f32));
    // Edges given, or one value only: left as they are.
    let given = StatsOptions { bins: zonal::Bins::Edges(vec![0.0, 1.0]), ..StatsOptions::default() };
    assert!(matches!(float32_bins(&given, &band).bins, zonal::Bins::Edges(ref e) if e == &vec![0.0, 1.0]));
    let flat = StatsOptions { bins: zonal::Bins::Count(10), ..StatsOptions::default() };
    assert!(matches!(float32_bins(&flat, &[2.0, 2.0]).bins, zonal::Bins::Count(10)));
}

#[test]
fn band_math_on_an_item_s_statistics() {
    // b1..bn over the stacked bands; an asset's name is not one, as in titiler-pgstac.
    assert!(check_item_band_math_names("(b1-b2)/(b1+b2)").is_ok());
    assert!(check_item_band_math_names("b1-b3;b4*2").is_ok());
    let named = check_item_band_math_names("(nir-red)/(nir+red)").unwrap_err();
    assert!(named.to_string().contains("Invalid band/asset name 'nir'"));
}

#[test]
fn an_item_preview_and_feature_say_where_they_were_drawn() {
    let ok = || Response::builder().status(StatusCode::OK).body(Full::new(Bytes::new())).unwrap();
    let header = |resp: &Resp, name: &str| resp.headers().get(name).map(|v| v.to_str().unwrap().to_string());
    let drawn = Some(([699960.0, 4790220.0, 809760.0, 4900020.0], Some(32631), false));
    for sub in [&["preview.png"][..], &["preview"], &["preview", "100x80.png"], &["feature", "64x64.png"], &["feature.png"]] {
        let resp = item_drawn_placed(ok(), sub, drawn);
        assert_eq!(header(&resp, "content-bbox").as_deref(), Some("699960.0,4790220.0,809760.0,4900020.0"), "{sub:?}");
        assert_eq!(header(&resp, "content-crs").as_deref(), Some("<http://www.opengis.net/def/crs/EPSG/0/32631>"));
    }
    // Degrees asked for as CRS84 are named so.
    let resp = item_drawn_placed(ok(), &["feature.png"], Some(([6.5, 43.3, 6.6, 43.4], Some(4326), true)));
    assert_eq!(header(&resp, "content-crs").as_deref(), Some("<http://www.opengis.net/def/crs/OGC/0/CRS84>"));
    // Not another route's, nor without a system, nor on an error.
    assert!(item_drawn_placed(ok(), &["info"], drawn).headers().get("content-bbox").is_none());
    assert!(item_drawn_placed(ok(), &["preview.png"], Some(([0.0, 0.0, 1.0, 1.0], None, false))).headers().get("content-crs").is_none());
    let missing = Response::builder().status(StatusCode::NOT_FOUND).body(Full::new(Bytes::new())).unwrap();
    assert!(item_drawn_placed(missing, &["preview.png"], drawn).headers().get("content-bbox").is_none());
}

#[test]
fn arrays_are_written_in_the_datas_type_with_a_mask() {
    let values = [1.0f32, f32::NAN, 300.0, 4.0];
    let made = npy_array(&values, 2, 2, 1, SampleType::Uint16, true, None);
    assert_eq!(&made[..8], b"\x93NUMPY\x01\x00");
    let header = usize::from(u16::from_le_bytes([made[8], made[9]]));
    assert_eq!((10 + header) % 64, 0);
    let dict = std::str::from_utf8(&made[10..10 + header]).unwrap();
    assert!(dict.starts_with("{'descr': '<u2', 'fortran_order': False, 'shape': (3, 1, 2), }") && dict.ends_with('\n'));
    let words: Vec<u16> = made[10 + header..].chunks_exact(2).map(|w| u16::from_le_bytes([w[0], w[1]])).collect();
    // Two bands, then the mask: a pixel is there when any band of it is.
    assert_eq!(words, [1, 0, 300, 4, u16::MAX, u16::MAX]);
    // Without the mask, and a missing value as the number the request named.
    let bare = npy_array(&values, 2, 2, 1, SampleType::Uint8, false, Some(9.0));
    assert_eq!(&bare[bare.len() - 4..], [1, 9, 255, 4]);
    // Where there is nothing, the mask holds the type's smallest number, as rio-tiler's.
    let doubles = npy_array(&[f32::NAN, 0.5], 1, 2, 1, SampleType::Float64, true, None);
    assert_eq!(&doubles[doubles.len() - 16..doubles.len() - 8], f64::MIN.to_le_bytes());
    assert_eq!(&doubles[doubles.len() - 8..], f64::MAX.to_le_bytes());
}

#[test]
fn a_colour_formula_comes_before_the_colour_map() {
    let one = Selected { bands: 1, values: vec![64.0, f32::NAN] };
    let formula = cogzilla_kernel::ColorFormula::parse("gamma r 2").unwrap();
    let mut level = [64u8, 64, 64, 255];
    formula.apply_buffer_in_place(&mut level, 4);
    assert_ne!(level[0], 64);
    let viridis = ColormapName::Viridis.lut();
    let got = paint_bands(&one, 2, &[(0.0, 255.0)], (0.0, 255.0), ColormapName::Viridis, None, Some(&formula)).unwrap();
    assert_eq!(got[..3], viridis.rgb[usize::from(level[0])]);
    assert_eq!(got[3..], [255, 0, 0, 0, 0]);
}

#[test]
fn the_renderers_grey_takes_the_formula_and_then_the_map() {
    let formula = cogzilla_kernel::ColorFormula::parse("gamma r 2").unwrap();
    let mut bands = BandRequest::Single { band_1based: 1, rescale: (0.0, 255.0), colormap: ColormapName::Viridis };
    let after = MapAfterFormula::take(&formula, &mut bands).unwrap();
    assert!(matches!(bands, BandRequest::Single { colormap: ColormapName::Grayscale, .. }));
    let mut rgba = [64u8, 64, 64, 255, 9, 9, 9, 0];
    after.colour(&mut rgba);
    let level = formula_levels(&formula)[64];
    assert_ne!(level, 64);
    assert_eq!(rgba[..3], ColormapName::Viridis.lut().rgb[usize::from(level)]);
    // A pixel that is not there stays as it was.
    assert_eq!(rgba[3..], [255, 9, 9, 9, 0]);
    // Grey already, or three bands: the formula is for the finished picture.
    assert!(MapAfterFormula::take(&formula, &mut bands).is_none());
    let mut rgb = BandRequest::Rgb { bands_1based: [1, 2, 3], rescale_min: [0.0; 3], rescale_max: [255.0; 3] };
    assert!(MapAfterFormula::take(&formula, &mut rgb).is_none());
    // DEM algorithms (hillshade / slope) also apply color_formula before colormap and emit grey when unmapped.
    let mut dem = BandRequest::Dem {
        band_1based: 1,
        algorithm: cogzilla_kernel::DemAlgorithm::Hillshade { azimuth_deg: 315.0, altitude_deg: 45.0, z_factor: 1.0 },
        rescale: (0.0, 255.0),
        colormap: ColormapName::Viridis,
        math: None,
    };
    assert!(!one_grey_band(&dem));
    let dem_after = MapAfterFormula::take(&formula, &mut dem).unwrap();
    assert!(one_grey_band(&dem));
    let mut dem_px = [64u8, 64, 64, 255];
    dem_after.colour(&mut dem_px);
    assert_eq!(dem_px[..3], ColormapName::Viridis.lut().rgb[usize::from(level)]);
}

#[test]
fn encoded_query_strips_doc_negotiation_params() {
    let pairs = vec![
        ("assets".to_string(), "data".to_string()),
        ("f".to_string(), "html".to_string()),
        ("format".to_string(), "json".to_string()),
        ("rescale".to_string(), "0,255".to_string()),
    ];
    assert_eq!(encoded_query(&pairs), "assets=data&rescale=0%2C255");

    // Image tile format=png is preserved.
    let img = vec![
        ("assets".to_string(), "data".to_string()),
        ("format".to_string(), "png".to_string()),
    ];
    assert_eq!(encoded_query(&img), "assets=data&format=png");
}

#[test]
fn in_raster_type_clamps_packed_overshoot_to_storable_range() {
    use cogzilla_raster::SampleType as T;
    // Cubic/Lanczos overshoots past packed bit widths must clamp to each type's own
    // storable_range (0..1, 0..15, 0..4095), not the wider GDAL reported_range (0..255, 0..65535).
    let clamped_u1 = in_raster_type(vec![vec![-0.4, 0.4, 0.6, 5.2, f32::NAN]], T::Uint1, false);
    assert_eq!(&clamped_u1[0][..4], &[0.0, 0.0, 1.0, 1.0]);
    assert!(clamped_u1[0][4].is_nan());

    let clamped_u4 = in_raster_type(vec![vec![-3.0, 7.6, 42.0, f32::NAN]], T::Uint4, false);
    assert_eq!(&clamped_u4[0][..3], &[0.0, 8.0, 15.0]);
    assert!(clamped_u4[0][3].is_nan());

    let clamped_u12 = in_raster_type(vec![vec![-12.3, 2048.4, 5120.7, 65_000.0, f32::NAN]], T::Uint12, false);
    assert_eq!(&clamped_u12[0][..4], &[0.0, 2048.0, 4095.0, 4095.0]);
    assert!(clamped_u12[0][4].is_nan());

    // Unscaled rasters are floats and bypass integer clamping.
    let unscaled = in_raster_type(vec![vec![-12.3, 5120.7]], T::Uint12, true);
    assert_eq!(unscaled[0], vec![-12.3, 5120.7]);
}

