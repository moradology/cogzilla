use std::path::PathBuf;
use std::sync::Arc;

use cogzilla_server::{TileEngine, serve_listener};
use tokio::net::TcpListener;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../benches/fixtures")
        .join(name)
}

#[tokio::test]
async fn full_titiler_cog_api_surface_works_end_to_end() {
    let sat_path = fixture("satellite_utm10n_rgb_zstd_p3.tif");
    if !sat_path.exists() {
        eprintln!("fixtures not generated; skipping HTTP integration test");
        return;
    }
    let sat_url = sat_path.canonicalize().unwrap().to_string_lossy().to_string();
    let lsat_b4 = fixture("landsat_c2_B4.tif").canonicalize().unwrap().to_string_lossy().to_string();
    let lsat_b3 = fixture("landsat_c2_B3.tif").canonicalize().unwrap().to_string_lossy().to_string();
    let lsat_b2 = fixture("landsat_c2_B2.tif").canonicalize().unwrap().to_string_lossy().to_string();

    let engine = Arc::new(TileEngine::new(128 * 1024 * 1024));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let engine_srv = engine.clone();
    tokio::spawn(async move {
        let _ = serve_listener(engine_srv, listener).await;
    });

    let base = format!("http://{addr}");
    let client = reqwest::Client::new();

    async fn get_json(client: &reqwest::Client, url: String) -> serde_json::Value {
        let resp = client.get(url).send().await.unwrap();
        assert_eq!(resp.status().as_u16(), 200);
        serde_json::from_slice(&resp.bytes().await.unwrap()).unwrap()
    }

    // 0. GET / serves the landing page with the crate version filled in
    let landing = client.get(format!("{base}/")).send().await.unwrap();
    assert_eq!(landing.status().as_u16(), 200);
    assert!(landing.headers()["content-type"].to_str().unwrap().starts_with("text/html"));
    let landing = landing.text().await.unwrap();
    assert!(landing.contains(concat!("v", env!("CARGO_PKG_VERSION"))));
    assert!(!landing.contains("{{VERSION}}"));

    // 0b. Options this server does not implement are rejected, not ignored.
    let tile = format!("{base}/cog/tiles/WebMercatorQuad/10/163/395.png?url={sat_url}");
    for (what, url) in [
        ("unknown algorithm", format!("{tile}&algorithm=does_not_exist")),
        ("unsupported resampling", format!("{tile}&resampling=bogus")),
        ("excessive padding", format!("{tile}&padding=999")),
        ("unsupported reproject", format!("{tile}&reproject=bogus")),
        ("bad rescale", format!("{tile}&rescale=abc")),
        ("unknown colormap", format!("{tile}&bidx=1&colormap_name=nope")),
        ("bad tilesize", format!("{tile}&tilesize=0")),
        ("unknown tms tile", format!("{base}/cog/tiles/FakeQuad/10/163/395.png?url={sat_url}")),
        ("unknown tms tilejson", format!("{base}/cog/FakeQuad/tilejson.json?url={sat_url}")),
        ("non-stats percentiles", format!("{base}/cog/info?url={sat_url}&p=5")),
        ("point algorithm", format!("{base}/cog/point/-122.6,37.7?url={sat_url}&algorithm=hillshade")),
        ("point unsupported coord_crs", format!("{base}/cog/point/1,1?url={sat_url}&coord_crs=epsg:4275")),
        ("preview explicit dims width", format!("{base}/cog/preview/256x256.png?url={sat_url}&width=300")),
        ("bbox explicit dims width", format!("{base}/cog/bbox/-122.8,37.6,-122.5,37.8/256x256.png?url={sat_url}&width=300")),
        ("non-colormap orientation", format!("{base}/cog/info?url={sat_url}&orientation=vertical")),
    ] {
        let resp = client.get(&url).send().await.unwrap();
        assert_eq!(resp.status().as_u16(), 422, "{what} should be rejected with 422: {url}");
        let body: serde_json::Value =
            serde_json::from_slice(&resp.bytes().await.unwrap()).unwrap();
        assert!(body["detail"][0]["msg"].is_string(), "422 response should match FastAPI JSON schema");
    }
    // Supported spellings and parameters this server does not know are still accepted.
    for (what, url) in [
        ("tilesize alias", format!("{tile}&tilesize=512&rescale=0,1")),
        ("cubic_spline", format!("{tile}&resampling=cubic_spline&rescale=0,1")),
        ("lanczos", format!("{tile}&resampling=lanczos&rescale=0,1")),
        ("average", format!("{tile}&resampling=average&rescale=0,1")),
        ("mode", format!("{tile}&resampling=mode&rescale=0,1")),
        ("gauss", format!("{tile}&resampling=gauss&rescale=0,1")),
        ("rms", format!("{tile}&resampling=rms&rescale=0,1")),
        ("med", format!("{tile}&resampling=med&rescale=0,1")),
        ("q1", format!("{tile}&resampling=q1&rescale=0,1")),
        ("q3", format!("{tile}&resampling=q3&rescale=0,1")),
        ("sum", format!("{tile}&resampling=sum&rescale=0,1")),
        ("reproject=average", format!("{tile}&reproject=average&rescale=0,1")),
        ("algorithm contours", format!("{tile}&bidx=1&algorithm=contours&algorithm_params=%7B%22increment%22:0.1,%22thickness%22:2%7D")),
        ("algorithm contours meta", format!("{base}/algorithms/contours")),
        ("reproject=bilinear", format!("{tile}&reproject=bilinear&rescale=0,1")),
        ("padding=0", format!("{tile}&padding=0&rescale=0,1")),
        ("padding=4", format!("{tile}&padding=4&rescale=0,1")),
        ("unrelated parameter", format!("{tile}&rescale=0,1&_cachebust=1&access_token=x")),
        ("hillshade", format!("{tile}&bidx=1&algorithm=hillshade")),
        ("WorldCRS84Quad", format!("{base}/cog/WorldCRS84Quad/tilejson.json?url={sat_url}")),
        ("UTM31WGS84Quad tilejson", format!("{base}/cog/UTM31WGS84Quad/tilejson.json?url={sat_url}")),
        ("WorldMercatorWGS84Quad tilejson", format!("{base}/cog/WorldMercatorWGS84Quad/tilejson.json?url={sat_url}")),
        ("WGS1984Quad tile", format!("{base}/cog/tiles/WGS1984Quad/2/1/1.png?url={sat_url}&rescale=0,1")),
        ("point coord_crs 4326", format!("{base}/cog/point/-122.6,37.7?url={sat_url}&coord_crs=epsg:4326")),
        ("point coord_crs 32610 UTM", format!("{base}/cog/point/535280,4172640?url={sat_url}&coord_crs=epsg:32610")),
        ("preview width query", format!("{base}/cog/preview.png?url={sat_url}&width=300")),
        ("preview dst_crs 4326", format!("{base}/cog/preview/64x64.png?url={sat_url}&dst_crs=epsg:4326")),
        ("preview dst_crs 3857", format!("{base}/cog/preview/64x64.tif?url={sat_url}&dst_crs=epsg:3857")),
        ("info.geojson crs 32610", format!("{base}/cog/info.geojson?url={sat_url}&crs=epsg:32610")),
        ("bounds crs 3857", format!("{base}/cog/bounds?url={sat_url}&crs=epsg:3857")),
        ("bbox geotiff without dims", format!("{base}/cog/bbox/-122.8,37.6,-122.5,37.8.tif?url={sat_url}&max_size=64&rescale=0,1")),
        ("preview max_size=0", format!("{base}/cog/preview.png?url={sat_url}&max_size=0&rescale=0,1")),
        ("colormap vertical orientation", format!("{base}/colorMaps/viridis?f=png&orientation=vertical")),
        ("algorithms list", format!("{base}/algorithms")),
        ("algorithm hillshade", format!("{base}/algorithms/hillshade")),
        ("cog validate", format!("{base}/cog/validate?url={sat_url}")),
        ("cog wmts", format!("{base}/cog/WebMercatorQuad/WMTSCapabilities.xml?url={sat_url}")),
        ("statistics custom percentiles & bins & bidx", format!("{base}/cog/statistics?url={sat_url}&bidx=1&p=5&p=95&histogram_bins=20&categorical=true")),
        ("point expression", format!("{base}/cog/point/-122.6,37.7?url={sat_url}&expression=b1-b2")),
        ("point bidx", format!("{base}/cog/point/-122.6,37.7?url={sat_url}&bidx=2")),
        ("mosaic tile buffer+nodata", format!("{base}/mosaic/tiles/WebMercatorQuad/10/163/395.png?url={sat_url}&rescale=0,1&buffer=4&nodata=0")),
        ("algorithm grayscale", format!("{tile}&algorithm=grayscale")),
        ("algorithm normalizedIndex", format!("{tile}&algorithm=normalizedIndex&bidx=1,2")),
        ("algorithm median 4-band", format!("{tile}&algorithm=median&bidx=1&bidx=2&bidx=3&bidx=1")),
        ("algorithm std 1-band", format!("{tile}&algorithm=std&bidx=1")),
        ("algorithm cast uint8 3-band", format!("{tile}&algorithm=cast&algorithm_params=%7B%22dtype%22:%22uint8%22%7D")),
        ("algorithm ceil 3-band", format!("{tile}&algorithm=ceil")),
        ("algorithm floor 3-band", format!("{tile}&algorithm=floor")),
        ("fractional buffer=0.5", format!("{tile}&buffer=0.5&rescale=0,1")),
        ("algorithm bitonal", format!("{tile}&algorithm=bitonal&algorithm_params=%7B%22threshold%22:0.5%7D")),
        ("statistics expression", format!("{base}/cog/statistics?url={sat_url}&expression=b1-b2;b2-b3")),
        ("point multi-expression", format!("{base}/cog/point/-122.6,37.7?url={sat_url}&expression=b1-b2;b2-b3")),
        ("cog tile geotiff", format!("{base}/cog/tiles/WebMercatorQuad/10/163/395.tif?url={sat_url}&rescale=0,1")),
        ("cog preview geotiff", format!("{base}/cog/preview/64x64.tif?url={sat_url}&rescale=0,1")),
        ("cog tile 3-band rgb expr", format!("{tile}&expression=b3*1.0;b2*1.0;b1*1.0&rescale=0,1")),
        ("ogc conformance", format!("{base}/conformance")),
        ("cog map", format!("{base}/cog/map?url={sat_url}&width=64&height=64&rescale=0,1")),
        ("cog map with bbox and crs", format!("{base}/cog/map?url={sat_url}&bbox=-122.8,37.6,-122.5,37.8&width=64&height=64&crs=epsg:3857&rescale=0,1")),
        ("color_formula numeric channels 123", format!("{tile}&rescale=0,1&color_formula=Gamma%20123%201.5")),
        ("trailing slash normalized", format!("{base}/cog/info/?url={sat_url}")),
    ] {
        let resp = client.get(&url).send().await.unwrap();
        assert_eq!(resp.status().as_u16(), 200, "{what} should be accepted: {url}");
        assert!(
            resp.headers().get("server-timing").is_some(),
            "{what} should include Server-Timing header"
        );
    }
    let st_custom = get_json(
        &client,
        format!("{base}/cog/statistics?url={sat_url}&bidx=1&p=5&p=95&histogram_bins=20"),
    )
    .await;
    assert_eq!(st_custom.as_object().unwrap().len(), 1);
    assert!(st_custom["b1"]["percentile_5"].is_number());
    assert!(st_custom["b1"]["percentile_95"].is_number());
    assert_eq!(st_custom["b1"]["histogram"][0].as_array().unwrap().len(), 20);
    let st_cat = get_json(
        &client,
        format!("{base}/cog/statistics?url={sat_url}&bidx=1&max_size=64&categorical=true&c=0&c=1"),
    )
    .await;
    assert_eq!(st_cat["b1"]["histogram"][0].as_array().unwrap().len(), 2);
    assert_eq!(st_cat["b1"]["histogram"][1], serde_json::json!([0, 1]));
    assert!(st_cat["b1"]["categories"].is_object());

    let pt_expr = get_json(
        &client,
        format!("{base}/cog/point/-122.6,37.7?url={sat_url}&expression=b1-b2"),
    )
    .await;
    assert_eq!(pt_expr["values"].as_array().unwrap().len(), 1);
    assert_eq!(pt_expr["band_names"][0], "b1");
    assert_eq!(pt_expr["band_descriptions"][0], "b1-b2");

    let pt_b2 = get_json(
        &client,
        format!("{base}/cog/point/-122.6,37.7?url={sat_url}&bidx=2"),
    )
    .await;
    assert_eq!(pt_b2["values"].as_array().unwrap().len(), 1);
    assert_eq!(pt_b2["band_names"][0], "b2");

    // Multi-expression /cog/statistics and /cog/point + GeoTIFF .tif output validation
    let st_expr = get_json(
        &client,
        format!("{base}/cog/statistics?url={sat_url}&expression=b1-b2;b2-b3"),
    )
    .await;
    assert!(st_expr["b1"]["mean"].is_number());
    assert_eq!(st_expr["b1"]["description"], "b1-b2");
    assert!(st_expr["b2"]["mean"].is_number());
    assert_eq!(st_expr["b2"]["description"], "b2-b3");

    let pt_multi = get_json(
        &client,
        format!("{base}/cog/point/-122.6,37.7?url={sat_url}&expression=b1-b2;b2-b3"),
    )
    .await;
    assert_eq!(pt_multi["values"].as_array().unwrap().len(), 2);
    assert_eq!(pt_multi["band_names"], serde_json::json!(["b1", "b2"]));
    assert_eq!(pt_multi["band_descriptions"], serde_json::json!(["b1-b2", "b2-b3"]));

    let tif_resp = client
        .get(format!("{base}/cog/tiles/WebMercatorQuad/10/163/395.tif?url={sat_url}&rescale=0,1"))
        .send()
        .await
        .unwrap();
    assert_eq!(tif_resp.headers().get("content-type").unwrap(), "image/tiff; application=geotiff");
    let tif_bytes = tif_resp.bytes().await.unwrap();
    assert_eq!(&tif_bytes[0..4], b"II\x2a\x00", "valid little-endian Classic TIFF magic");
    let parsed_tif = cogzilla_tiff::parse_cog(&tif_bytes).unwrap();
    assert_eq!(parsed_tif.full_res().width, 256);
    assert_eq!(parsed_tif.full_res().height, 256);
    assert_eq!(parsed_tif.geo_keys.projected_cs_type, Some(3857));

    // Out-of-bounds /cog/tiles returns 404 Not Found (matching TiTiler TileOutsideBounds)
    let oob_resp = client
        .get(format!("{base}/cog/tiles/WebMercatorQuad/10/0/0.png?url={sat_url}"))
        .send()
        .await
        .unwrap();
    assert_eq!(oob_resp.status().as_u16(), 404);
    let prev_300 = client
        .get(format!("{base}/cog/preview.png?url={sat_url}&width=300"))
        .send()
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap();
    assert_eq!(u32::from_be_bytes(prev_300[16..20].try_into().unwrap()), 300, "width=300 yields 300 px PNG");
    let vcm = client
        .get(format!("{base}/colorMaps/viridis?f=png&orientation=vertical"))
        .send()
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap();
    assert_eq!(
        (
            u32::from_be_bytes(vcm[16..20].try_into().unwrap()),
            u32::from_be_bytes(vcm[20..24].try_into().unwrap()),
        ),
        (20, 256),
        "orientation=vertical yields 20x256 PNG"
    );
    let big = client.get(format!("{tile}&tilesize=512&rescale=0,1")).send().await.unwrap().bytes().await.unwrap();
    assert_eq!(u32::from_be_bytes(big[16..20].try_into().unwrap()), 512, "tilesize=512 yields a 512 px PNG");

    // 1. GET /cog/info
    let info = get_json(&client, format!("{base}/cog/info?url={sat_url}")).await;
    assert_eq!(info["dtype"], "float32");
    assert_eq!(info["count"], 3);
    assert_eq!(info["width"], 2048);
    let bounds = info["wgs84_bounds"].as_array().unwrap();
    assert!(bounds[0].as_f64().unwrap() < -122.5 && bounds[2].as_f64().unwrap() > -122.5);

    // 2. GET /cog/info.geojson
    let gj = get_json(&client, format!("{base}/cog/info.geojson?url={sat_url}")).await;
    assert_eq!(gj["type"], "Feature");
    assert_eq!(gj["geometry"]["type"], "Polygon");

    // 3. GET /cog/WebMercatorQuad/tilejson.json
    let tj = get_json(&client, format!("{base}/cog/WebMercatorQuad/tilejson.json?url={sat_url}")).await;
    assert_eq!(tj["tilejson"], "3.0.0");
    assert!(tj["tiles"][0].as_str().unwrap().contains("/cog/tiles/WebMercatorQuad/{z}/{x}/{y}"));

    // 4. GET /cog/tiles & /cog/tiles/WebMercatorQuad
    assert_eq!(
        client.get(format!("{base}/cog/tiles")).send().await.unwrap().status().as_u16(),
        200
    );
    assert_eq!(
        client.get(format!("{base}/cog/tiles/WebMercatorQuad")).send().await.unwrap().status().as_u16(),
        200
    );

    // 5. GET /cog/WebMercatorQuad/map.html
    let html = client
        .get(format!("{base}/cog/WebMercatorQuad/map.html?url={sat_url}"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    assert!(html.contains("maplibregl"));

    // 6. GET /cog/tiles/WebMercatorQuad/{z}/{x}/{y}@1x.png & @2x.png & arbitrary ?expression=
    let png_1x = client
        .get(format!(
            "{base}/cog/tiles/WebMercatorQuad/12/652/1583@1x.png?url={sat_url}&bidx=1&bidx=2&bidx=3&rescale=0,0.35"
        ))
        .send()
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap();
    assert_eq!(&png_1x[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);

    let png_2x = client
        .get(format!(
            "{base}/cog/tiles/WebMercatorQuad/12/652/1583@2x.png?url={sat_url}&expression=(b1-b2)/(b1+b2)&rescale=-0.5,0.5&colormap_name=ndvi"
        ))
        .send()
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap();
    assert_eq!(&png_2x[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);

    // 7. Multi-COG Landsat C2 uint16 3-file RGB via TiTiler route
    let lsat_png = client
        .get(format!(
            "{base}/cog/tiles/WebMercatorQuad/12/652/1583.png?url={lsat_b4}&url={lsat_b3}&url={lsat_b2}&rescale=7500,22500"
        ))
        .send()
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap();
    assert_eq!(&lsat_png[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);

    // 8. GET /cog/point/{lon},{lat}
    let pt = get_json(&client, format!("{base}/cog/point/-122.65,37.773?url={sat_url}")).await;
    assert_eq!(pt["band_names"].as_array().unwrap().len(), 3);
    assert!(pt["values"][0].as_f64().unwrap() > 0.0);

    // 9. GET /cog/statistics
    let stats = get_json(&client, format!("{base}/cog/statistics?url={sat_url}&max_size=512")).await;
    assert!(stats["b1"]["valid_pixels"].as_u64().unwrap() > 1000);
    assert!(stats["b1"]["mean"].as_f64().unwrap() > 0.0);

    // 10. GET /cog/preview/256x256.png
    let prev = client
        .get(format!("{base}/cog/preview/256x256.png?url={sat_url}&rescale=0,0.35"))
        .send()
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap();
    assert_eq!(&prev[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);

    // 11. GET /cog/bbox/{minx},{miny},{maxx},{maxy}/256x256.png
    let bbox_img = client
        .get(format!(
            "{base}/cog/bbox/-122.8,37.6,-122.5,37.9/256x256.png?url={sat_url}&rescale=0,0.35"
        ))
        .send()
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap();
    assert_eq!(&bbox_img[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);

    // 12. POST /cog/feature/256x256.png
    let feat = serde_json::json!({
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [-122.8, 37.6],
                [-122.5, 37.6],
                [-122.5, 37.9],
                [-122.8, 37.9],
                [-122.8, 37.6]
            ]]
        },
        "properties": {}
    });
    let feat_img = client
        .post(format!("{base}/cog/feature/256x256.png?url={sat_url}&rescale=0,0.35"))
        .body(feat.to_string())
        .send()
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap();
    assert_eq!(&feat_img[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);

    // ------------------------------------------------------------------------
    // 13. /colorMaps and /colorMaps/{colorMapId}
    // ------------------------------------------------------------------------
    let cmaps: serde_json::Value = serde_json::from_str(
        &client
            .get(format!("{base}/colorMaps"))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap(),
    )
    .unwrap();
    let cmap_list = cmaps["colormaps"].as_array().unwrap();
    assert!(cmap_list.iter().any(|v| v["id"] == "viridis"));
    assert!(cmap_list.iter().any(|v| v["id"] == "rdylgn"));

    let viridis_json: serde_json::Value = serde_json::from_str(
        &client
            .get(format!("{base}/colorMaps/viridis"))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(viridis_json.as_object().unwrap().len(), 256);
    assert_eq!(viridis_json["0"], serde_json::json!([68, 1, 84, 255]));

    let viridis_png = client
        .get(format!("{base}/colorMaps/viridis?format=png"))
        .send()
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap();
    assert_eq!(&viridis_png[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);

    // ------------------------------------------------------------------------
    // 14. /tileMatrixSets and /tileMatrixSets/{tileMatrixSetId}
    // ------------------------------------------------------------------------
    let tms_list: serde_json::Value = serde_json::from_str(
        &client
            .get(format!("{base}/tileMatrixSets"))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap(),
    )
    .unwrap();
    let tms_arr = tms_list["tileMatrixSets"].as_array().unwrap();
    assert_eq!(tms_arr.len(), 13, "expected all 13 morecantile TileMatrixSets");
    assert!(tms_arr.iter().any(|t| t["id"] == "WebMercatorQuad"));
    assert!(tms_arr.iter().any(|t| t["id"] == "WorldCRS84Quad"));
    assert!(tms_arr.iter().any(|t| t["id"] == "UTM31WGS84Quad"));
    assert!(tms_arr.iter().any(|t| t["id"] == "WorldMercatorWGS84Quad"));

    // Verify GeoTIFF wrapping on /mosaic/tiles and native EPSG on /cog/preview.tif
    let mosaic_tif = client
        .get(format!("{base}/mosaic/tiles/WebMercatorQuad/10/163/395.tif?url={sat_url}&rescale=0,1"))
        .send()
        .await
        .unwrap();
    assert_eq!(mosaic_tif.status().as_u16(), 200);
    let mosaic_tif_bytes = mosaic_tif.bytes().await.unwrap();
    assert_eq!(&mosaic_tif_bytes[0..4], b"II\x2a\x00", "/mosaic/tiles .tif must be a valid GeoTIFF");

    let preview_tif = client
        .get(format!("{base}/cog/preview/64x64.tif?url={sat_url}&rescale=0,1"))
        .send()
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap();
    let parsed_preview = cogzilla_tiff::parse_cog(&preview_tif).expect("valid preview GeoTIFF");
    assert_eq!(
        parsed_preview.geo_keys.projected_cs_type,
        Some(32610),
        "native preview GeoTIFF should preserve source UTM EPSG:32610"
    );

    // Verify .npy Content-Type matches TiTiler (application/x-binary)
    let npy_resp = client
        .get(format!("{base}/cog/tiles/WebMercatorQuad/10/163/395.npy?url={sat_url}"))
        .send()
        .await
        .unwrap();
    assert_eq!(npy_resp.status().as_u16(), 200);
    assert_eq!(
        npy_resp.headers().get("content-type").unwrap().to_str().unwrap(),
        "application/x-binary"
    );

    // Verify WMTSCapabilities ScaleDenominator scales inversely with tile_scale
    let wmts_1x = client
        .get(format!("{base}/cog/WebMercatorQuad/WMTSCapabilities.xml?url={sat_url}&tile_scale=1"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    let wmts_2x = client
        .get(format!("{base}/cog/WebMercatorQuad/WMTSCapabilities.xml?url={sat_url}&tile_scale=2"))
        .send()
        .await
        .unwrap()
        .text()
        .await
        .unwrap();
    let first_sd = |xml: &str| -> f64 {
        let s = xml.split("<ScaleDenominator>").nth(1).unwrap();
        s.split("</ScaleDenominator>").next().unwrap().parse::<f64>().unwrap()
    };
    let sd1 = first_sd(&wmts_1x);
    let sd2 = first_sd(&wmts_2x);
    assert!(
        ((sd1 / sd2) - 2.0).abs() < 1e-6,
        "WMTS ScaleDenominator should halve for tile_scale=2 (got {sd1} vs {sd2})"
    );

    let tms_wm: serde_json::Value = serde_json::from_str(
        &client
            .get(format!("{base}/tileMatrixSets/WebMercatorQuad"))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(tms_wm["id"], "WebMercatorQuad");
    assert_eq!(tms_wm["tileMatrices"].as_array().unwrap().len(), 25);

    // ------------------------------------------------------------------------
    // 15. /stac/* (MultiBaseTilerFactory(reader=STACReader))
    // ------------------------------------------------------------------------
    let stac_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../benches/fixtures/landsat_c2_stac.json");
    if stac_path.exists() {
        let stac_url = stac_path.to_string_lossy();

        // /stac/assets
        let assets: serde_json::Value = serde_json::from_str(
            &client
                .get(format!("{base}/stac/assets?url={stac_url}"))
                .send()
                .await
                .unwrap()
                .text()
                .await
                .unwrap(),
        )
        .unwrap();
        let arr = assets.as_array().unwrap();
        assert!(arr.iter().any(|v| v == "red"));
        assert!(arr.iter().any(|v| v == "green"));
        assert!(arr.iter().any(|v| v == "blue"));
        assert!(!arr.iter().any(|v| v == "thumbnail"));

        // /stac/info
        let stac_info: serde_json::Value = serde_json::from_str(
            &client
                .get(format!("{base}/stac/info?url={stac_url}&assets=red&assets=green"))
                .send()
                .await
                .unwrap()
                .text()
                .await
                .unwrap(),
        )
        .unwrap();
        assert!(stac_info.get("red").is_some());
        assert!(stac_info.get("green").is_some());

        // /stac/tiles/WebMercatorQuad/12/652/1583.png?assets=red&assets=green&assets=blue
        let stac_tile = client
            .get(format!(
                "{base}/stac/tiles/WebMercatorQuad/12/652/1583.png?url={stac_url}&assets=red&assets=green&assets=blue&rescale=7000,20000"
            ))
            .send()
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        assert_eq!(&stac_tile[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);

        // /stac/tiles with per-asset asset_bidx=red|1&asset_bidx=green|1&asset_bidx=blue|1 must match assets=red,green,blue
        let stac_ab_resp = client
            .get(format!(
                "{base}/stac/tiles/WebMercatorQuad/12/652/1583.png?url={stac_url}&asset_bidx=red|1&asset_bidx=green|1&asset_bidx=blue|1&rescale=7000,20000"
            ))
            .send()
            .await
            .unwrap();
        assert_eq!(
            stac_ab_resp.headers().get("x-assets").and_then(|v| v.to_str().ok()),
            Some("red,green,blue")
        );
        let stac_ab_tile = stac_ab_resp.bytes().await.unwrap();
        assert_eq!(stac_ab_tile, stac_tile, "per-asset asset_bidx should render identical 3-asset RGB");

        // /stac/tiles/... with asset-named expression
        let stac_expr_tile = client
            .get(format!(
                "{base}/stac/tiles/WebMercatorQuad/12/652/1583.png?url={stac_url}&expression=(red-green)/(red+green)&rescale=-0.5,0.5&colormap_name=rdylgn"
            ))
            .send()
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        assert_eq!(&stac_expr_tile[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);

        // /stac/WebMercatorQuad/tilejson.json
        let stac_tj: serde_json::Value = serde_json::from_str(
            &client
                .get(format!(
                    "{base}/stac/WebMercatorQuad/tilejson.json?url={stac_url}&assets=red&assets=green&assets=blue"
                ))
                .send()
                .await
                .unwrap()
                .text()
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(stac_tj["tilejson"], "3.0.0");

        // /stac/point/{lon},{lat}
        let stac_pt: serde_json::Value = serde_json::from_str(
            &client
                .get(format!(
                    "{base}/stac/point/-122.65,37.77?url={stac_url}&assets=red&assets=green&assets=blue"
                ))
                .send()
                .await
                .unwrap()
                .text()
                .await
                .unwrap(),
        )
        .unwrap();
        assert_eq!(stac_pt["values"].as_array().unwrap().len(), 3);
        assert_eq!(stac_pt["band_names"].as_array().unwrap().len(), 3);

        // /stac/asset_statistics & /stac/statistics
        let stac_astats: serde_json::Value = serde_json::from_str(
            &client
                .get(format!(
                    "{base}/stac/asset_statistics?url={stac_url}&assets=red&assets=green&max_size=256"
                ))
                .send()
                .await
                .unwrap()
                .text()
                .await
                .unwrap(),
        )
        .unwrap();
        assert!(stac_astats.get("red").is_some());
        assert!(stac_astats.get("green").is_some());

        let stac_stats: serde_json::Value = serde_json::from_str(
            &client
                .get(format!(
                    "{base}/stac/statistics?url={stac_url}&assets=red&assets=green&max_size=256"
                ))
                .send()
                .await
                .unwrap()
                .text()
                .await
                .unwrap(),
        )
        .unwrap();
        assert!(stac_stats.get("b1").is_some());
        assert!(stac_stats.get("b2").is_some());

        // /stac/preview/256x256.png
        let stac_prev = client
            .get(format!(
                "{base}/stac/preview/256x256.png?url={stac_url}&assets=red&assets=green&assets=blue&rescale=7000,20000"
            ))
            .send()
            .await
            .unwrap()
            .bytes()
            .await
            .unwrap();
        assert_eq!(&stac_prev[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);
    }

    // ------------------------------------------------------------------------
    // 16. Dynamic early-exit COG mosaic (/mosaic/tiles/...)
    // ------------------------------------------------------------------------
    let merc_path = fixture("mercator_4k_zstd_p3.tif");
    if merc_path.exists() {
        let merc_url = merc_path.canonicalize().unwrap().to_string_lossy().to_string();
        // Proof 1: When the front scene covers 100% of the tile, background URLs are NEVER opened!
        {
            use std::sync::atomic::{AtomicU64, Ordering};
            let hits = Arc::new(AtomicU64::new(0));
            let hits_srv = hits.clone();
            let bg_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
            let bg_addr = bg_listener.local_addr().unwrap();
            tokio::spawn(async move {
                loop {
                    let Ok((_stream, _)) = bg_listener.accept().await else {
                        break;
                    };
                    hits_srv.fetch_add(1, Ordering::Relaxed);
                }
            });
            let bg_url = format!("http://{bg_addr}/never_touched.tif");
            let mosaic_png = client
                .get(format!(
                    "{base}/mosaic/tiles/WebMercatorQuad/12/655/1583.png?url={merc_url}&url={bg_url}&rescale=0,2500&colormap_name=viridis"
                ))
                .send()
                .await
                .unwrap();
            assert_eq!(mosaic_png.status(), 200);
            let bytes = mosaic_png.bytes().await.unwrap();
            assert_eq!(&bytes[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);
            assert_eq!(
                hits.load(Ordering::Relaxed),
                0,
                "background scene should never be opened when front scene covers 100% of pixels"
            );
        }

        // Proof 2: When the front scene has partial nodata holes (coastal_4k_nodata.tif),
        // the mosaic fills those holes from the background scene!
        let coastal_path = fixture("coastal_4k_nodata.tif");
        if coastal_path.exists() {
            let coastal_url = coastal_path.canonicalize().unwrap().to_string_lossy().to_string();
            let resp = client
                .get(format!(
                    "{base}/mosaic/tiles/WebMercatorQuad/12/655/1583.png?url={coastal_url}&url={merc_url}&rescale=0,2500&colormap_name=viridis"
                ))
                .send()
                .await
                .unwrap();
            assert_eq!(resp.status(), 200);
            let bytes = resp.bytes().await.unwrap();
            assert_eq!(&bytes[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);
            // PNG IHDR color_type is at byte offset 25: 2 = RGB (100% opaque, all holes filled!)
            assert_eq!(
                bytes[25], 2,
                "expected opaque RGB PNG (color_type=2) after filling coastal nodata holes from background scene"
            );
        }

        // --------------------------------------------------------------------
        // 17. Indexed spatial catalog mosaic (/mosaic/{name}/tiles/...)
        // --------------------------------------------------------------------
        {
            use cogzilla_proj::{TileCoord, tile_bounds_3857};
            use cogzilla_server::{InMemoryCatalog, SceneEntry};
            let (tx0, ty0, tx1, ty1) = tile_bounds_3857(TileCoord {
                z: 12,
                x: 655,
                y: 1583,
            });
            let cat = InMemoryCatalog::new(vec![
                SceneEntry {
                    uri: merc_url.clone(),
                    bbox_3857: [tx0 - 500.0, ty0 - 500.0, tx1 + 500.0, ty1 + 500.0],
                    priority: 100,
                },
                SceneEntry {
                    uri: "/nonexistent/low_priority_scene.tif".into(),
                    bbox_3857: [tx0 - 500.0, ty0 - 500.0, tx1 + 500.0, ty1 + 500.0],
                    priority: 10,
                },
            ]);
            engine.register_catalog("demo", Arc::new(cat));

            // In-bounds: hits high-priority merc_url and early-exits before touching nonexistent scene!
            let resp = client
                .get(format!(
                    "{base}/mosaic/demo/tiles/WebMercatorQuad/12/655/1583.png?rescale=0,2500&colormap_name=viridis"
                ))
                .send()
                .await
                .unwrap();
            assert_eq!(resp.status(), 200);
            let bytes = resp.bytes().await.unwrap();
            assert_eq!(&bytes[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);

            // Out-of-bounds: spatial index returns 0 scenes -> instant transparent PNG!
            let empty_resp = client
                .get(format!(
                    "{base}/mosaic/demo/tiles/WebMercatorQuad/12/0/0.png"
                ))
                .send()
                .await
                .unwrap();
            assert_eq!(empty_resp.status(), 200);
            let empty_bytes = empty_resp.bytes().await.unwrap();
            assert_eq!(&empty_bytes[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);

            // Unknown catalog -> 404
            let missing_resp = client
                .get(format!(
                    "{base}/mosaic/nope/tiles/WebMercatorQuad/12/655/1583.png"
                ))
                .send()
                .await
                .unwrap();
            assert_eq!(missing_resp.status(), 404);
        }

        // --------------------------------------------------------------------
        // 18. HTTP Caching (ETag, Cache-Control, If-None-Match -> 304) & /metrics
        // --------------------------------------------------------------------
        {
            let tile_url = format!(
                "{base}/cog/tiles/WebMercatorQuad/12/655/1583.png?url={merc_url}&rescale=0,2500&colormap_name=viridis"
            );
            let r1 = client.get(&tile_url).send().await.unwrap();
            assert_eq!(r1.status(), 200);
            let etag = r1
                .headers()
                .get("etag")
                .expect("ETag header missing")
                .to_str()
                .unwrap()
                .to_string();
            let cc = r1
                .headers()
                .get("cache-control")
                .expect("Cache-Control header missing")
                .to_str()
                .unwrap();
            assert!(cc.contains("public"), "unexpected Cache-Control: {cc}");
            let _ = r1.bytes().await.unwrap();

            // Conditional GET with If-None-Match -> 304 Not Modified with empty body
            let r2 = client
                .get(&tile_url)
                .header("If-None-Match", &etag)
                .send()
                .await
                .unwrap();
            assert_eq!(r2.status(), 304);
            assert!(r2.bytes().await.unwrap().is_empty());

            // Prometheus /metrics exposition
            let m = client.get(format!("{base}/metrics")).send().await.unwrap();
            assert_eq!(m.status(), 200);
            let ct = m.headers().get("content-type").unwrap().to_str().unwrap();
            assert!(ct.starts_with("text/plain"));
            let body = m.text().await.unwrap();
            assert!(body.contains("cogzilla_tiles_served_total "));
            assert!(body.contains("cogzilla_http_304_not_modified_total 1"));
            assert!(body.contains("cogzilla_tile_cache_coalesced_total "));
        }

        // --------------------------------------------------------------------
        // 19. MosaicJSON (/mosaicjson/*) & Dynamic Registration (POST /mosaic/register)
        // --------------------------------------------------------------------
        {
            let qk = cogzilla_server::tile_to_quadkey(12, 655, 1583);
            let mj_doc = serde_json::json!({
                "mosaicjson": "0.0.3",
                "minzoom": 10,
                "maxzoom": 16,
                "quadkey_zoom": 12,
                "bounds": [-123.0, 37.0, -122.0, 38.0],
                "tiles": {
                    qk: [merc_url.clone(), "/nonexistent_bg.tif"]
                }
            });
            let mj_path = std::env::temp_dir().join(format!(
                "cogzilla_test_mosaic_{}.json",
                std::process::id()
            ));
            std::fs::write(&mj_path, mj_doc.to_string()).unwrap();
            let mj_url = mj_path.to_string_lossy().to_string();

            let mj_info = get_json(&client, format!("{base}/mosaicjson/info?url={mj_url}")).await;
            assert_eq!(mj_info["quadkey_zoom"], 12);
            assert_eq!(mj_info["tiles_count"], 1);
            assert_eq!(mj_info["mosaic_tilematrixset"], "WebMercatorQuad");
            assert!(mj_info["quadkeys"].as_array().is_some_and(|a| !a.is_empty()));
            assert!(mj_info["crs"].as_str().is_some());

            // Non-WebMercatorQuad TMS on /mosaicjson/tiles (WorldCRS84Quad overlapping 12/655/1583)
            let mj_crs84 = client
                .get(format!(
                    "{base}/mosaicjson/tiles/WorldCRS84Quad/10/327/297.png?url={mj_url}&rescale=0,2500&colormap_name=viridis"
                ))
                .send()
                .await
                .unwrap();
            assert_eq!(mj_crs84.status(), 200);
            let crs84_bytes = mj_crs84.bytes().await.unwrap();
            assert_eq!(&crs84_bytes[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);

            let mj_tile = client
                .get(format!(
                    "{base}/mosaicjson/tiles/WebMercatorQuad/12/655/1583.png?url={mj_url}&rescale=0,2500&colormap_name=viridis"
                ))
                .send()
                .await
                .unwrap();
            assert_eq!(mj_tile.status(), 200);
            let bytes = mj_tile.bytes().await.unwrap();
            assert_eq!(&bytes[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);

            // /mosaicjson/info.geojson, /mosaicjson/assets, /mosaicjson/point, /mosaicjson/bbox, /mosaicjson/statistics
            let mj_geo = get_json(&client, format!("{base}/mosaicjson/info.geojson?url={mj_url}")).await;
            assert_eq!(mj_geo["type"], "Feature");
            let mj_assets = get_json(&client, format!("{base}/mosaicjson/assets?url={mj_url}")).await;
            assert_eq!(mj_assets.as_array().unwrap().len(), 2);
            let mj_pt = get_json(&client, format!("{base}/mosaicjson/point/-122.4,37.75?url={mj_url}")).await;
            assert!(!mj_pt["assets"].as_array().unwrap().is_empty());
            let mj_pt_expr = get_json(
                &client,
                format!("{base}/mosaicjson/point/-122.4,37.75?url={mj_url}&expression=b1*2"),
            )
            .await;
            assert_eq!(mj_pt_expr["assets"][0]["band_names"][0], "b1");
            assert_eq!(mj_pt_expr["assets"][0]["band_descriptions"][0], "b1*2");
            let mj_stats = get_json(&client, format!("{base}/mosaicjson/statistics?url={mj_url}&max_size=128")).await;
            assert!(mj_stats["b1"]["mean"].is_number());
            let mj_bbox = client
                .get(format!("{base}/mosaicjson/bbox/-122.5,37.7,-122.3,37.8/64x64.png?url={mj_url}&rescale=0,2500&colormap_name=viridis"))
                .send()
                .await
                .unwrap();
            assert_eq!(mj_bbox.status(), 200);
            let mj_bbox_bytes = mj_bbox.bytes().await.unwrap();
            assert_eq!(&mj_bbox_bytes[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);

            // /mosaicjson/tiles outside mosaic bounds returns 404 Not Found with JSON detail
            let mj_outside = client
                .get(format!(
                    "{base}/mosaicjson/tiles/WebMercatorQuad/12/0/0.png?url={mj_url}&buffer=8"
                ))
                .send()
                .await
                .unwrap();
            assert_eq!(mj_outside.status(), 404);

            // Dynamic registration via POST /mosaic/register
            let reg_resp = client
                .post(format!("{base}/mosaic/register"))
                .header("content-type", "application/json")
                .body(
                    serde_json::json!({
                        "name": "live_mj",
                        "catalog": mj_doc,
                    })
                    .to_string(),
                )
                .send()
                .await
                .unwrap();
            assert_eq!(reg_resp.status(), 200);
            let reg_tile = client
                .get(format!(
                    "{base}/mosaic/live_mj/tiles/WebMercatorQuad/12/655/1583.png?rescale=0,2500&colormap_name=viridis"
                ))
                .send()
                .await
                .unwrap();
            assert_eq!(reg_tile.status(), 200);
            let _ = std::fs::remove_file(&mj_path);
        }

        // --------------------------------------------------------------------
        // 20. Terrain Tiles: Terrarium & Mapbox Terrain-RGB (?algorithm=...)
        // --------------------------------------------------------------------
        for alg in ["terrarium", "terrainrgb"] {
            let resp = client
                .get(format!(
                    "{base}/cog/tiles/WebMercatorQuad/12/655/1583.png?url={merc_url}&algorithm={alg}"
                ))
                .send()
                .await
                .unwrap();
            assert_eq!(resp.status(), 200, "algorithm={alg} failed");
            let bytes = resp.bytes().await.unwrap();
            assert_eq!(&bytes[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);
            // Opaque RGB PNG (color_type=2) for clean DEM tile
            assert_eq!(bytes[25], 2, "expected RGB PNG for algorithm={alg}");

            // Terrain NPY must emit 3-band RGB + mask (4, 256, 256), not raw 1-band elevation
            let npy_resp = client
                .get(format!(
                    "{base}/cog/tiles/WebMercatorQuad/12/655/1583.npy?url={merc_url}&algorithm={alg}"
                ))
                .send()
                .await
                .unwrap();
            assert_eq!(npy_resp.status(), 200);
            let npy_bytes = npy_resp.bytes().await.unwrap();
            let hdr_str = std::str::from_utf8(&npy_bytes[10..128]).unwrap_or("");
            assert!(
                hdr_str.contains("(4, 256, 256)"),
                "expected 3-band RGB + mask NPY for algorithm={alg}, got header: {hdr_str}"
            );
        }

        // --------------------------------------------------------------------
        // 21. On-the-Fly DEM Hillshade, Slope, Aspect (?algorithm=...)
        // --------------------------------------------------------------------
        for alg in ["hillshade", "slope", "aspect"] {
            let resp = client
                .get(format!(
                    "{base}/cog/tiles/WebMercatorQuad/12/655/1583.png?url={merc_url}&algorithm={alg}&azimuth=315&altitude=45&z_factor=2.0"
                ))
                .send()
                .await
                .unwrap();
            assert_eq!(resp.status(), 200, "algorithm={alg} failed");
            let bytes = resp.bytes().await.unwrap();
            assert_eq!(&bytes[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);
            // Grayscale (0), Gray+Alpha (4), or indexed-palette PNG (3)
            assert!(
                matches!(bytes[25], 0 | 3 | 4),
                "expected grayscale, gray+alpha, or indexed PNG for algorithm={alg}, got color_type={}",
                bytes[25]
            );
        }

        // Verify TiTiler-compatible JSON ?algorithm_params={"azimuth":180,"angle_altitude":30}
        {
            let r_default = client
                .get(format!(
                    "{base}/cog/tiles/WebMercatorQuad/12/655/1583.png?url={merc_url}&algorithm=hillshade"
                ))
                .send()
                .await
                .unwrap();
            let b_default = r_default.bytes().await.unwrap();

            let r_json = client
                .get(format!(
                    "{base}/cog/tiles/WebMercatorQuad/12/655/1583.png?url={merc_url}&algorithm=hillshade&algorithm_params=%7B%22azimuth%22%3A180%2C%22angle_altitude%22%3A30%2C%22z_factor%22%3A2.5%7D"
                ))
                .send()
                .await
                .unwrap();
            assert_eq!(r_json.status(), 200);
            let b_json = r_json.bytes().await.unwrap();
            assert_eq!(&b_json[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);
            assert_ne!(
                b_default, b_json,
                "algorithm_params JSON should alter hillshade lighting"
            );

            // Mixed-resolution / mixed-CRS multi-COG composite (UTM10N RGB + WebMercator f32 DEM)
            let mixed_resp = client
                .get(format!(
                    "{base}/cog/tiles/WebMercatorQuad/12/655/1583.png?url={sat_url}&url={merc_url}&expression=b1+b2&rescale=0,500"
                ))
                .send()
                .await
                .unwrap();
            assert_eq!(
                mixed_resp.status(),
                200,
                "mixed-resolution/mixed-CRS multi-COG composite should succeed"
            );
            let mixed_bytes = mixed_resp.bytes().await.unwrap();
            assert_eq!(&mixed_bytes[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);

            // Audit c40 regression checks:
            // 1. Path {w}x{h} > 4096 must be rejected with 400 (no OOM in GeoTIFF/bbox)
            let oob_dim = client
                .get(format!("{base}/cog/preview/5000x5000.tif?url={merc_url}"))
                .send()
                .await
                .unwrap();
            assert_eq!(oob_dim.status(), 400);

            // 2. Non-4326 geographic GeoTIFF (NAD83 EPSG:4269) must write ModelTypeGeographic (2)
            //    and unstyled float32 DEM .tif must emit native IEEE Float32 GeoTIFF (SampleType::Float32)
            let nad83_resp = client
                .get(format!(
                    "{base}/cog/bbox/-122.5,37.7,-122.4,37.8/16x16.tif?url={merc_url}&dst_crs=EPSG:4269"
                ))
                .send()
                .await
                .unwrap();
            assert_eq!(nad83_resp.status(), 200);
            let nad83_tif = nad83_resp.bytes().await.unwrap();
            let parsed_nad83 = cogzilla_tiff::parse_cog(&nad83_tif).unwrap();
            assert_eq!(parsed_nad83.geo_keys.model_type, Some(2));
            assert_eq!(parsed_nad83.geo_keys.geographic_type, Some(4269));
            assert_eq!(parsed_nad83.geo_keys.projected_cs_type, None);
            assert_eq!(parsed_nad83.full_res().sample_type, cogzilla_raster::SampleType::Float32);

            // Styled .tif with rescale+colormap must still emit 8-bit visual GeoTIFF
            let styled_tif_resp = client
                .get(format!(
                    "{base}/cog/bbox/-122.5,37.7,-122.4,37.8/16x16.tif?url={merc_url}&rescale=0,2500&colormap_name=viridis"
                ))
                .send()
                .await
                .unwrap();
            assert_eq!(styled_tif_resp.status(), 200);
            let parsed_styled = cogzilla_tiff::parse_cog(&styled_tif_resp.bytes().await.unwrap()).unwrap();
            assert_eq!(parsed_styled.full_res().sample_type, cogzilla_raster::SampleType::Uint8);

            // Audit c42 regression checks:
            // Unstyled float32 DEM .tif on /mosaic/tiles must emit Float32 GeoTIFF (not corrupt RGBA8)
            let mosaic_f32_tif = client
                .get(format!(
                    "{base}/mosaic/tiles/WebMercatorQuad/12/655/1583.tif?url={merc_url}"
                ))
                .send()
                .await
                .unwrap();
            assert_eq!(mosaic_f32_tif.status(), 200);
            let parsed_mosaic_f32 = cogzilla_tiff::parse_cog(&mosaic_f32_tif.bytes().await.unwrap()).unwrap();
            assert_eq!(parsed_mosaic_f32.full_res().sample_type, cogzilla_raster::SampleType::Float32);

            // ?algorithm=contours .tif must emit 8-bit visual GeoTIFF (not Float32)
            let contours_tif = client
                .get(format!(
                    "{base}/cog/tiles/WebMercatorQuad/12/655/1583.tif?url={merc_url}&algorithm=contours"
                ))
                .send()
                .await
                .unwrap();
            assert_eq!(contours_tif.status(), 200);
            let parsed_contours = cogzilla_tiff::parse_cog(&contours_tif.bytes().await.unwrap()).unwrap();
            assert_eq!(parsed_contours.full_res().sample_type, cogzilla_raster::SampleType::Uint8);

            // Audit c45 regression check: unknown/malformed TMS on WMTSCapabilities.xml must return 404/422 (never 200 XML)
            let bad_wmts = client
                .get(format!("{base}/cog/a&b/WMTSCapabilities.xml?url={merc_url}"))
                .send()
                .await
                .unwrap();
            assert!(
                bad_wmts.status() == 404 || bad_wmts.status() == 422,
                "unknown WMTS TMS should be rejected, got {}",
                bad_wmts.status()
            );

            // Audit c47 regression check: buffered WebMercatorQuad x=0 tile must not false-404
            let buf_x0 = client
                .get(format!(
                    "{base}/cog/tiles/WebMercatorQuad/1/0/0.png?url={sat_url}&buffer=2&rescale=0,1"
                ))
                .send()
                .await
                .unwrap();
            assert_eq!(buf_x0.status(), 200, "buffered x=0 WebMercatorQuad tile must not false-404");

            // Audit c48 regression check: constant & where() expression statistics
            let const_stats = get_json(
                &client,
                format!("{base}/cog/statistics?url={merc_url}&expression=42&max_size=64"),
            )
            .await;
            assert!((const_stats["b1"]["mean"].as_f64().unwrap() - 42.0).abs() < 1e-4);
            assert_eq!(const_stats["b1"]["description"], "42");

            // 3. 16-bit Tag 320 palette entries >= 256 must be preserved in CustomColormap
            let mut pal = vec![[10u8, 20, 30, 255]; 300];
            pal[299] = [200, 150, 100, 255];
            let cmap = cogzilla_kernel::CustomColormap::from_rgba_palette(&pal);
            assert_eq!(cmap.lookup(299.0, 0.0, 1.0), [200, 150, 100, 255]);

            // Audit c69 regression check: antimeridian (w > e) and inverted (s > n) STAC Item
            // bboxes on /stac/map must not panic in render_bbox_impl and must preserve 5:4 aspect.
            let am_stac_path = std::env::temp_dir().join(format!(
                "cogzilla_stac_antimeridian_{}.json",
                std::process::id()
            ));
            let am_stac = serde_json::json!({
                "type": "Feature",
                "stac_version": "1.0.0",
                "id": "fiji-antimeridian",
                "bbox": [177.0, -20.0, -178.0, -16.0],
                "geometry": null,
                "properties": {},
                "assets": {
                    "b1": { "href": sat_path.to_str().unwrap(), "type": "image/tiff; application=geotiff" }
                }
            });
            std::fs::write(&am_stac_path, am_stac.to_string()).unwrap();
            let am_stac_url = am_stac_path.to_str().unwrap();
            let am_resp = client
                .get(format!(
                    "{base}/stac/map?url={am_stac_url}&assets=b1&max_size=128&rescale=0,1"
                ))
                .send()
                .await
                .unwrap();
            assert_eq!(
                am_resp.status(),
                404,
                "antimeridian /stac/map outside asset bounds must return 404 without panicking"
            );

            let inv_stac = serde_json::json!({
                "type": "Feature",
                "stac_version": "1.0.0",
                "id": "inverted-bbox",
                "bbox": [0.0, 10.0, 1.0, -10.0],
                "geometry": null,
                "properties": {},
                "assets": {
                    "b1": { "href": sat_path.to_str().unwrap(), "type": "image/tiff; application=geotiff" }
                }
            });
            let inv_stac_path = std::env::temp_dir().join(format!(
                "cogzilla_stac_inverted_{}.json",
                std::process::id()
            ));
            std::fs::write(&inv_stac_path, inv_stac.to_string()).unwrap();
            let inv_stac_url = inv_stac_path.to_str().unwrap();
            let inv_resp = client
                .get(format!(
                    "{base}/stac/map?url={inv_stac_url}&assets=b1&rescale=0,1"
                ))
                .send()
                .await
                .unwrap();
            assert_eq!(inv_resp.status(), 400, "inverted s > n STAC bbox must return 400");
            let _ = std::fs::remove_file(&am_stac_path);
            let _ = std::fs::remove_file(&inv_stac_path);

            // Audit c75 regression checks:
            // 1. GeometryCollection accepted in PolygonMask::from_geojson + constant expression
            //    respects shape_mask (outside-polygon pixels must be NaN, not 1.0).
            let gc_poly = serde_json::json!({
                "type": "GeometryCollection",
                "geometries": [{
                    "type": "Polygon",
                    "coordinates": [[
                        [-122.65, 37.76],
                        [-122.60, 37.76],
                        [-122.65, 37.80],
                        [-122.65, 37.76]
                    ]]
                }]
            });
            let gc_resp = client
                .post(format!("{base}/cog/feature/32x32.f32?url={sat_url}&expression=1"))
                .header("content-type", "application/json")
                .body(gc_poly.to_string())
                .send()
                .await
                .unwrap();
            assert_eq!(gc_resp.status(), 200, "GeometryCollection /cog/feature must succeed");
            let gc_bytes = gc_resp.bytes().await.unwrap();
            assert_eq!(gc_bytes.len(), 32 * 32 * 4);
            // Top-right pixel (row 0, col 31) is outside the lower-left triangle -> must be NaN
            let tr_val = f32::from_le_bytes([
                gc_bytes[31 * 4],
                gc_bytes[31 * 4 + 1],
                gc_bytes[31 * 4 + 2],
                gc_bytes[31 * 4 + 3],
            ]);
            assert!(tr_val.is_nan(), "outside-polygon pixel for constant expression must be NaN, got {tr_val}");
            // Bottom-left pixel (row 30, col 1) is inside the triangle -> must be 1.0
            let bl_off = (30 * 32 + 1) * 4;
            let bl_val = f32::from_le_bytes([
                gc_bytes[bl_off],
                gc_bytes[bl_off + 1],
                gc_bytes[bl_off + 2],
                gc_bytes[bl_off + 3],
            ]);
            assert_eq!(bl_val, 1.0, "inside-polygon pixel for expression=1 must be 1.0");

            // 2. 2-vertex degenerate ring must be rejected with 400
            let two_vtx = serde_json::json!({
                "type": "Polygon",
                "coordinates": [[[-122.65, 37.76], [-122.60, 37.80]]]
            });
            let two_resp = client
                .post(format!("{base}/cog/feature/32x32.png?url={merc_url}&rescale=0,2500"))
                .header("content-type", "application/json")
                .body(two_vtx.to_string())
                .send()
                .await
                .unwrap();
            assert_eq!(two_resp.status(), 400, "2-vertex GeoJSON ring must return 400");

            // Audit c77 regression check: mixed-grid multi-COG /cog/feature with constant expr
            // must still mask outside-polygon pixels to NaN.
            let mg_resp = client
                .post(format!(
                    "{base}/cog/feature/32x32.f32?url={sat_url}&url={merc_url}&expression=1"
                ))
                .header("content-type", "application/json")
                .body(gc_poly.to_string())
                .send()
                .await
                .unwrap();
            assert_eq!(mg_resp.status(), 200);
            let mg_bytes = mg_resp.bytes().await.unwrap();
            assert_eq!(mg_bytes.len(), 32 * 32 * 4);
            let mg_tr = f32::from_le_bytes(mg_bytes[31 * 4..32 * 4].try_into().unwrap());
            let mg_bl = f32::from_le_bytes(mg_bytes[bl_off..bl_off + 4].try_into().unwrap());
            assert!(mg_tr.is_nan(), "mixed-grid outside-polygon pixel must be NaN, got {mg_tr}");
            assert_eq!(mg_bl, 1.0, "mixed-grid inside-polygon pixel must be 1.0");

            // Audit c78 regression checks:
            // 1. Mixed-grid where(b2>=0,b2,0) must NOT clip b2 (sat_url) to cogs[0] (merc_url)'s footprint
            let mg_b2 = client
                .post(format!(
                    "{base}/cog/feature/32x32.f32?url={merc_url}&url={sat_url}&expression=where(b2%3E=0,b2,0)"
                ))
                .header("content-type", "application/json")
                .body(gc_poly.to_string())
                .send()
                .await
                .unwrap();
            assert_eq!(mg_b2.status(), 200);
            let mg_b2_bytes = mg_b2.bytes().await.unwrap();
            let mg_b2_tr = f32::from_le_bytes(mg_b2_bytes[31 * 4..32 * 4].try_into().unwrap());
            let mg_b2_bl = f32::from_le_bytes(mg_b2_bytes[bl_off..bl_off + 4].try_into().unwrap());
            assert!(mg_b2_tr.is_nan(), "outside-polygon pixel must remain NaN, got {mg_b2_tr}");
            assert!(mg_b2_bl.is_finite(), "b2 pixel inside sat_url must not be clipped to cogs[0], got {mg_b2_bl}");

            // 2. /mosaic/tiles must skip an unreadable urls[0] and render urls[1]
            let resil_resp = client
                .get(format!(
                    "{base}/mosaic/tiles/WebMercatorQuad/12/655/1583.png?url=/nonexistent_cogzilla_scene_0.tif&url={merc_url}&rescale=0,2500"
                ))
                .send()
                .await
                .unwrap();
            assert_eq!(resil_resp.status(), 200, "mosaic tile must skip unreadable urls[0]");
            let resil_bytes = resil_resp.bytes().await.unwrap();
            assert_eq!(&resil_bytes[0..8], &[137, 80, 78, 71, 13, 10, 26, 10]);

            // 3. GET /stac/statistics must preserve ?bidx=3,1 order
            let bidx_stac_path = std::env::temp_dir().join(format!(
                "cogzilla_stac_bidx_order_{}.json",
                std::process::id()
            ));
            let bidx_stac = serde_json::json!({
                "type": "Feature",
                "stac_version": "1.0.0",
                "id": "bidx-order",
                "bbox": [-122.8, 37.6, -122.5, 37.9],
                "geometry": null,
                "properties": {},
                "assets": {
                    "visual": { "href": sat_path.to_str().unwrap(), "type": "image/tiff; application=geotiff" }
                }
            });
            std::fs::write(&bidx_stac_path, bidx_stac.to_string()).unwrap();
            let bidx_stats = get_json(
                &client,
                format!(
                    "{base}/stac/statistics?url={}&assets=visual&bidx=3&bidx=1&max_size=64",
                    bidx_stac_path.to_str().unwrap()
                ),
            )
            .await;
            assert_eq!(bidx_stats["b1"]["description"].as_str(), Some("visual_b3"));
            assert_eq!(bidx_stats["b2"]["description"].as_str(), Some("visual_b1"));
            let _ = std::fs::remove_file(&bidx_stac_path);

            // 4. MosaicJSON tile assets & pixel_selection (highest, lowest, mean, count)
            let mj_ps_path = std::env::temp_dir().join(format!(
                "cogzilla_mj_ps_{}.json",
                std::process::id()
            ));
            let mj_ps = serde_json::json!({
                "mosaicjson": "0.0.2",
                "minzoom": 8,
                "maxzoom": 14,
                "bounds": [-123.0, 37.0, -121.5, 38.5],
                "quadkey_zoom": 8,
                "tiles": {
                    "02301020": [sat_path.to_str().unwrap(), merc_path.to_str().unwrap()],
                    "02301021": [sat_path.to_str().unwrap(), merc_path.to_str().unwrap()],
                    "02301022": [sat_path.to_str().unwrap(), merc_path.to_str().unwrap()],
                    "02301023": [sat_path.to_str().unwrap(), merc_path.to_str().unwrap()]
                }
            });
            std::fs::write(&mj_ps_path, mj_ps.to_string()).unwrap();
            let tile_assets = get_json(
                &client,
                format!(
                    "{base}/mosaicjson/tiles/WebMercatorQuad/12/655/1583/assets?url={}",
                    mj_ps_path.to_str().unwrap()
                ),
            )
            .await;
            assert!(tile_assets.is_array() && !tile_assets.as_array().unwrap().is_empty());
            let pt_assets = get_json(
                &client,
                format!(
                    "{base}/mosaicjson/point/-122.4,37.75/assets?url={}",
                    mj_ps_path.to_str().unwrap()
                ),
            )
            .await;
            assert!(pt_assets.is_array() && !pt_assets.as_array().unwrap().is_empty());
            let _ = std::fs::remove_file(&mj_ps_path);

            // Compare pixel_selection=highest vs lowest vs mean vs count on two scenes (merc vs sat b1)
            let read_f32_first = |b: &[u8]| -> f32 {
                b.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .find(|v| v.is_finite())
                    .unwrap_or(f32::NAN)
            };
            let hi_bytes = client
                .get(format!("{base}/mosaic/tiles/WebMercatorQuad/12/657/1582.f32?url={merc_url}&url={sat_url}&bidx=1&pixel_selection=highest"))
                .send().await.unwrap().bytes().await.unwrap();
            let lo_bytes = client
                .get(format!("{base}/mosaic/tiles/WebMercatorQuad/12/657/1582.f32?url={merc_url}&url={sat_url}&bidx=1&pixel_selection=lowest"))
                .send().await.unwrap().bytes().await.unwrap();
            let cnt_bytes = client
                .get(format!("{base}/mosaic/tiles/WebMercatorQuad/12/657/1582.f32?url={merc_url}&url={merc_url}&bidx=1&pixel_selection=count"))
                .send().await.unwrap().bytes().await.unwrap();
            let hi_v = read_f32_first(&hi_bytes);
            let lo_v = read_f32_first(&lo_bytes);
            let cnt_v = read_f32_first(&cnt_bytes);
            assert!(hi_v >= lo_v, "highest ({hi_v}) must be >= lowest ({lo_v})");
            assert!((cnt_v - 2.0).abs() < 1e-4, "count on 2 identical scenes must be 2.0, got {cnt_v}");

            let ps_tile = format!("{base}/mosaic/tiles/WebMercatorQuad/12/657/1582");
            let f32s = |b: &[u8]| -> Vec<f32> {
                b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
            };
            let sat_tile = format!("{base}/mosaic/tiles/WebMercatorQuad/12/652/1583");
            let fetch = |tile: &str, query: String| {
                let client = client.clone();
                let url = format!("{tile}.f32?{query}");
                async move {
                    let resp = client.get(url).send().await.unwrap();
                    (resp.status(), resp.bytes().await.unwrap())
                }
            };

            // count is a number everywhere, zero where no scene has a value.
            let counts = f32s(&cnt_bytes);
            assert_eq!(counts.len(), 256 * 256);
            assert!(counts.iter().all(|v| *v == 0.0 || *v == 2.0), "count is 0 or 2 over two identical scenes");

            // Names: any case, and `std` for `stdev`; anything else is refused.
            let (status, stdev) = fetch(&ps_tile, format!("url={merc_url}&url={sat_url}&bidx=1&pixel_selection=stdev")).await;
            assert_eq!(status, 200);
            for name in ["std", "STD", "Stdev"] {
                let (status, body) = fetch(&ps_tile, format!("url={merc_url}&url={sat_url}&bidx=1&pixel_selection={name}")).await;
                assert_eq!(status, 200, "{name}");
                assert_eq!(body, stdev, "{name}");
            }
            let (status, _) = fetch(&ps_tile, format!("url={merc_url}&url={sat_url}&bidx=1&pixel_selection=sum")).await;
            assert_eq!(status, 422);

            // The last-band methods leave the deciding band out, and need a band beside it.
            for name in ["lastbandhigh", "lastbandlow"] {
                let three = format!("url={sat_url}&url={sat_url}&bidx=1&bidx=2&bidx=3&pixel_selection={name}");
                let (status, body) = fetch(&sat_tile, three).await;
                assert_eq!(status, 200, "{name}");
                assert_eq!(body.len(), 2 * 256 * 256 * 4, "{name} of three bands leaves two");
                let (status, _) = fetch(&sat_tile, format!("url={sat_url}&url={sat_url}&bidx=1&pixel_selection={name}")).await;
                assert_eq!(status, 400, "{name} of one band");
            }
            // Over one scene the bands left are that scene's own.
            let (_, kept) = fetch(&sat_tile, format!("url={sat_url}&bidx=1&bidx=2&bidx=3&pixel_selection=lastbandhigh")).await;
            let (_, own) = fetch(&sat_tile, format!("url={sat_url}&url={sat_url}&bidx=1&bidx=2&pixel_selection=highest")).await;
            assert_eq!(kept, own);

            // median holds its scenes, within 64 MiB: two three-band scenes at 2048 px are over it.
            let over = format!("url={sat_url}&url={sat_url}&bidx=1&bidx=2&bidx=3&pixel_selection=median&tilesize=2048");
            let (status, body) = fetch(&sat_tile, over).await;
            assert_eq!(status, 400);
            assert!(
                String::from_utf8_lossy(&body).contains("pixel_selection=median exceeds memory budget (64 MiB limit)"),
                "{}",
                String::from_utf8_lossy(&body)
            );
            let (status, body) = fetch(&sat_tile, format!("url={sat_url}&url={sat_url}&bidx=1&pixel_selection=median")).await;
            assert_eq!(status, 200);
            let (_, one) = fetch(&sat_tile, format!("url={sat_url}&url={sat_url}&bidx=1&pixel_selection=highest")).await;
            assert_eq!(body, one, "the median of a scene taken twice is the scene");

            // mean and median of whole numbers are whole numbers, the fraction dropped;
            // of floats they keep it.
            let (b2, b3) = (fixture("landsat_c2_B2.tif"), fixture("landsat_c2_B3.tif"));
            if b2.exists() && b3.exists() {
                let pair = format!(
                    "url={}&url={}&bidx=1",
                    b2.canonicalize().unwrap().to_string_lossy(),
                    b3.canonicalize().unwrap().to_string_lossy()
                );
                let (_, hi) = fetch(&sat_tile, format!("{pair}&pixel_selection=highest")).await;
                let (_, lo) = fetch(&sat_tile, format!("{pair}&pixel_selection=lowest")).await;
                let (hi, lo) = (f32s(&hi), f32s(&lo));
                let halves = hi.iter().zip(&lo).filter(|(h, l)| h.is_finite() && (**h + **l) % 2.0 == 1.0).count();
                assert!(halves > 0, "some pixel's mean of two scenes has a half to drop");
                for name in ["mean", "median"] {
                    let (status, got) = fetch(&sat_tile, format!("{pair}&pixel_selection={name}")).await;
                    assert_eq!(status, 200, "{name}");
                    let got = f32s(&got);
                    assert_eq!(got.len(), hi.len());
                    for (i, v) in got.iter().enumerate() {
                        if hi[i].is_finite() {
                            let exact = (f64::from(hi[i]) + f64::from(lo[i])) / 2.0;
                            assert_eq!(f64::from(*v), exact.trunc(), "{name} at {i}: of {} and {}", lo[i], hi[i]);
                        } else {
                            assert!(v.is_nan(), "{name} at {i}");
                        }
                    }
                }
            }
            let (_, mean) = fetch(&ps_tile, format!("url={merc_url}&url={sat_url}&bidx=1&pixel_selection=mean")).await;
            assert!(f32s(&mean).iter().any(|v| v.is_finite() && v.fract() != 0.0), "a float mean keeps its fraction");

            // What a pixel selection leaves is painted and encoded like any other picture:
            // every format, with and without the mask, under a colour map and a formula.
            let formats: [(&str, &[u8]); 3] = [("png", &[0x89, b'P', b'N', b'G']), ("jpg", &[0xff, 0xd8]), ("webp", b"RIFF")];
            for (ext, magic) in formats {
                for look in ["", "&return_mask=false", "&rescale=0,100&colormap_name=viridis", "&color_formula=gamma%20rgb%202"] {
                    for bands in ["bidx=1", "bidx=1&bidx=2&bidx=3"] {
                        // A colour map is for one band; it is refused over three.
                        if look.contains("colormap_name") && bands != "bidx=1" {
                            continue;
                        }
                        let url = format!("{sat_tile}.{ext}?url={sat_url}&url={sat_url}&{bands}&pixel_selection=highest{look}");
                        let resp = client.get(&url).send().await.unwrap();
                        assert_eq!(resp.status(), 200, "{url}");
                        let body = resp.bytes().await.unwrap();
                        assert!(body.starts_with(magic), "{url}");
                    }
                }
            }

            // Sweep #1 & #7 C1-C4, D2(validate), G1-G3 regression checks
            let cog_stac = get_json(
                &client,
                format!("{base}/cog/stac?url={sat_url}&id=my-sat&collection=c1&collection_url=https://example.com/c1&asset_name=rgb&asset_href=https://example.com/sat.tif&asset_roles=visual&extensions=https://example.com/ext.json"),
            )
            .await;
            assert_eq!(cog_stac["type"], "Feature");
            assert_eq!(cog_stac["stac_version"], "1.1.0");
            assert_eq!(cog_stac["id"], "my-sat");
            assert_eq!(cog_stac["collection"], "c1");
            assert_eq!(cog_stac["properties"]["proj:epsg"], 32610);
            assert!(cog_stac["properties"]["proj:geometry"].is_object());
            assert_eq!(cog_stac["assets"]["rgb"]["href"], "https://example.com/sat.tif");
            assert_eq!(cog_stac["assets"]["rgb"]["roles"], serde_json::json!(["visual"]));
            assert_eq!(cog_stac["assets"]["rgb"]["type"], "image/tiff; application=geotiff");
            assert!(cog_stac["assets"]["rgb"]["raster:bands"][0]["statistics"].is_object());
            assert!(cog_stac["assets"]["rgb"]["raster:bands"][0]["histogram"].is_object());
            assert_eq!(cog_stac["assets"]["rgb"]["raster:bands"][0]["sampling"], "area");
            assert!(cog_stac["assets"]["rgb"]["eo:bands"].is_array());
            assert_eq!(cog_stac["links"][0]["rel"], "collection");
            assert!(cog_stac["stac_extensions"].as_array().unwrap().iter().any(|v| v == "https://example.com/ext.json"));

            let viewer_cog = client
                .get(format!("{base}/cog/viewer?url={sat_url}"))
                .send().await.unwrap();
            assert_eq!(viewer_cog.status(), 200);
            assert!(viewer_cog.text().await.unwrap().contains("tilejson.json"));

            let stac_renders_path = std::env::temp_dir().join(format!(
                "cogzilla_test_stac_renders_{}.json",
                std::process::id()
            ));
            let stac_renders_doc = serde_json::json!({
                "type": "Feature",
                "stac_version": "1.0.0",
                "id": "renders-test",
                "bbox": [-122.9, 37.5, -122.4, 38.0],
                "geometry": null,
                "properties": {},
                "assets": {
                    "red": { "href": sat_path.to_str().unwrap(), "type": "image/tiff; application=geotiff" }
                },
                "renders": {
                    "visual": { "assets": ["red"], "rescale": [[0, 1]] }
                }
            });
            std::fs::write(&stac_renders_path, stac_renders_doc.to_string()).unwrap();
            let stac_item_url = stac_renders_path.to_str().unwrap();
            let viewer_stac = client
                .get(format!("{base}/stac/viewer?url={stac_item_url}&assets=red"))
                .send().await.unwrap();
            assert_eq!(viewer_stac.status(), 200);

            let renders_json = get_json(&client, format!("{base}/stac/renders?url={stac_item_url}")).await;
            assert!(renders_json["renders"]["visual"]["valid"] == true);
            assert!(renders_json["renders"]["visual"]["params"]["assets"][0] == "red");
            assert!(renders_json["links"].is_array());
            let visual_render = get_json(&client, format!("{base}/stac/renders/visual?url={stac_item_url}")).await;
            assert_eq!(visual_render["valid"], true);
            assert_eq!(visual_render["params"]["assets"][0], "red");
            assert_eq!(visual_render["links"].as_array().unwrap().len(), 3);
            let missing_render = client
                .get(format!("{base}/stac/renders/nonexistent?url={stac_item_url}"))
                .send().await.unwrap();
            assert_eq!(missing_render.status(), 404);
            let _ = std::fs::remove_file(&stac_renders_path);

            let mj_doc = serde_json::json!({
                "mosaicjson": "0.0.3",
                "minzoom": 10,
                "maxzoom": 12,
                "quadkey_zoom": 10,
                "bounds": [-123.0, 37.0, -122.0, 38.0],
                "name": null,
                "attribution": null,
                "tiles": { "0230102033": [merc_url.clone()] }
            });
            let mj_val = client
                .post(format!("{base}/mosaicjson/validate"))
                .header("content-type", "application/json")
                .body(mj_doc.to_string())
                .send().await.unwrap();
            assert_eq!(mj_val.status(), 200);
            let mj_val_json: serde_json::Value =
                serde_json::from_slice(&mj_val.bytes().await.unwrap()).unwrap();
            assert_eq!(mj_val_json["mosaicjson"], "0.0.3");
            assert!(mj_val_json.get("name").is_none(), "null optional fields must be omitted");
            assert!(mj_val_json["tiles"].is_object());

            let mj_root_path = std::env::temp_dir().join(format!(
                "cogzilla_test_mj_root_{}.json",
                std::process::id()
            ));
            std::fs::write(&mj_root_path, mj_doc.to_string()).unwrap();
            let mj_root = get_json(
                &client,
                format!("{base}/mosaicjson/?url={}", mj_root_path.to_str().unwrap()),
            )
            .await;
            assert_eq!(mj_root["mosaicjson"], "0.0.3");
            assert!(mj_root.get("name").is_none());
            assert!(mj_root.get("attribution").is_none());
            assert!(mj_root["tiles"].is_object());

            // Sweep #8 H3: color_formula on single-band mosaic stays grayscale PNG (no red tint)
            // Quadkey "0230102033" corresponds to WebMercatorQuad tile 10/163/395.
            let mj_cf_resp = client
                .get(format!(
                    "{base}/mosaicjson/tiles/WebMercatorQuad/10/163/395.png?url={}&rescale=0,1000&color_formula=gamma%20r%201.6",
                    mj_root_path.to_str().unwrap()
                ))
                .send()
                .await
                .unwrap();
            assert_eq!(mj_cf_resp.status(), 200);
            let mj_cf_bytes = mj_cf_resp.bytes().await.unwrap();
            assert!(mj_cf_bytes.len() > 26 && &mj_cf_bytes[..8] == b"\x89PNG\r\n\x1a\n");
            // PNG IHDR color_type byte at offset 25: 0 = Grayscale (L), 4 = Grayscale+Alpha (LA)
            let color_type = mj_cf_bytes[25];
            assert!(
                color_type == 0 || color_type == 4,
                "single-band mosaic with color_formula must encode grayscale PNG (color_type 0 or 4), got {color_type}"
            );
            // Mosaic point outside coverage returns 204 No Content with empty body (matching TiTiler, #628)
            let mj_pt_oob = client
                .get(format!(
                    "{base}/mosaicjson/point/0,0?url={}",
                    mj_root_path.to_str().unwrap()
                ))
                .send()
                .await
                .unwrap();
            assert_eq!(mj_pt_oob.status(), 204);
            assert!(mj_pt_oob.bytes().await.unwrap().is_empty());
            let _ = std::fs::remove_file(&mj_root_path);

            // Sweep #1 D4, D6, E2 regression checks
            let ts_one = get_json(&client, format!("{base}/cog/tiles/WebMercatorQuad?url={sat_url}")).await;
            assert!(ts_one["links"].as_array().is_some_and(|l| l.len() >= 3));
            assert!(ts_one["boundingBox"]["lowerLeft"].is_array());
            assert!(ts_one["tileMatrixSetLimits"].as_array().is_some_and(|l| !l.is_empty()));
            assert_eq!(ts_one["accessConstraints"], "unclassified");

            let tms_one = get_json(&client, format!("{base}/tileMatrixSets/WebMercatorQuad")).await;
            assert!(tms_one["uri"].as_str().is_some_and(|s| s.contains("WebMercatorQuad")));
            assert!(tms_one["wellKnownScaleSet"].as_str().is_some_and(|s| s.contains("GoogleMapsCompatible")));

            let conf = get_json(&client, format!("{base}/conformance")).await;
            assert_eq!(conf["conformsTo"].as_array().unwrap().len(), 23);

            let mna1 = client.post(format!("{base}/cog/info?url={sat_url}")).send().await.unwrap();
            assert_eq!(mna1.status(), 405);
            assert!(mna1.headers().get("allow").is_some());
            let mna2 = client.get(format!("{base}/cog/feature?url={sat_url}")).send().await.unwrap();
            assert_eq!(mna2.status(), 405);
            let mna3 = client.delete(format!("{base}/healthz")).send().await.unwrap();
            assert_eq!(mna3.status(), 405);

            // Sweep #1 E1 & D5 regression checks: gzip + Vary + Cache-Control on metadata, and 13-TMS WMTS
            let gz_resp = client
                .get(format!("{base}/tileMatrixSets/WebMercatorQuad"))
                .header("accept-encoding", "gzip")
                .send()
                .await
                .unwrap();
            assert_eq!(gz_resp.status(), 200);
            assert_eq!(
                gz_resp.headers().get("content-encoding").and_then(|v| v.to_str().ok()),
                Some("gzip")
            );
            assert!(
                gz_resp
                    .headers()
                    .get("vary")
                    .and_then(|v| v.to_str().ok())
                    .is_some_and(|v| v.to_ascii_lowercase().contains("accept-encoding"))
            );
            assert!(gz_resp.headers().get("cache-control").is_some());
            let gz_bytes = gz_resp.bytes().await.unwrap();
            let decomp = cogzilla_codec::decompress_gzip_limited(&gz_bytes, 1024 * 1024).unwrap();
            let tms_decomp: serde_json::Value = serde_json::from_slice(&decomp).unwrap();
            assert_eq!(tms_decomp["id"], "WebMercatorQuad");

            let wmts_all = client
                .get(format!("{base}/cog/WMTSCapabilities.xml?url={sat_url}"))
                .send()
                .await
                .unwrap();
            assert_eq!(wmts_all.status(), 200);
            assert!(wmts_all.headers().get("cache-control").is_some());
            let wmts_body = wmts_all.text().await.unwrap();
            assert_eq!(wmts_body.matches("<Layer>").count(), 13);
            assert_eq!(wmts_body.matches("<TileMatrixSet>").count(), 26); // 13 links + 13 definitions
            // Sweep #10 K1 & K3: TileMatrixSetLimits, ServiceMetadataURL, EPSG:xxxx on use_epsg=true, zooms stripped
            assert_eq!(wmts_body.matches("<TileMatrixSetLimits>").count(), 13);
            assert!(wmts_body.contains("<MinTileRow>"));
            assert!(wmts_body.contains("<ServiceMetadataURL"));
            let wmts_epsg = client
                .get(format!("{base}/cog/WMTSCapabilities.xml?url={sat_url}&use_epsg=true&zooms=WebMercatorQuad::9,10"))
                .send()
                .await
                .unwrap()
                .text()
                .await
                .unwrap();
            assert!(wmts_epsg.contains("<ows:SupportedCRS>EPSG:3857</ows:SupportedCRS>"));
            assert!(wmts_epsg.contains("<ows:SupportedCRS>EPSG:4326</ows:SupportedCRS>"));
            assert!(
                wmts_epsg
                    .lines()
                    .filter(|l| l.contains("<ResourceURL"))
                    .all(|l| !l.contains("zooms=")),
                "ResourceURL templates must not include zooms="
            );

            // Sweep #10 K2: binary .npy tile gzip-compressed when Accept-Encoding: gzip or *
            let npy_gz = client
                .get(format!("{base}/cog/tiles/WebMercatorQuad/12/652/1583.npy?url={sat_url}"))
                .header("accept-encoding", "gzip")
                .send()
                .await
                .unwrap();
            assert_eq!(npy_gz.status(), 200);
            assert_eq!(
                npy_gz.headers().get("content-encoding").and_then(|v| v.to_str().ok()),
                Some("gzip")
            );
            let npy_gz_bytes = npy_gz.bytes().await.unwrap();
            let npy_raw = cogzilla_codec::decompress_gzip_limited(&npy_gz_bytes, 8 * 1024 * 1024).unwrap();
            assert!(npy_raw.starts_with(b"\x93NUMPY"));

            // Sweep #9 J1-J6 regression checks
            // J1: empty expression= is treated as absent
            let j1 = client
                .get(format!("{base}/cog/tiles/WebMercatorQuad/12/652/1583.png?url={sat_url}&expression="))
                .send().await.unwrap();
            assert_eq!(j1.status(), 200);
            // J2 & J3: float-int params (bidx=1.0, tilesize=16.0, z=12.0) and tilesize < 64
            let j23 = client
                .get(format!("{base}/cog/tiles/WebMercatorQuad/12.0/652/1583.png?url={sat_url}&bidx=1.0&tilesize=16.0"))
                .send().await.unwrap();
            assert_eq!(j23.status(), 200);
            // J4: space after comma in /point and /bbox path coordinates
            let j4p = client
                .get(format!("{base}/cog/point/-122.65,%2037.773?url={sat_url}"))
                .send().await.unwrap();
            assert_eq!(j4p.status(), 200);
            let j4b = client
                .get(format!("{base}/cog/bbox/-122.8,%2037.6,-122.5,37.9/32x32.png?url={sat_url}"))
                .send().await.unwrap();
            assert_eq!(j4b.status(), 200);
            // J5: EPSG:4979, ESRI:102100, and PROJ string CRS forms
            for crs in [
                "EPSG:4979",
                "ESRI:102100",
                "+proj=longlat%20+datum=WGS84",
            ] {
                let r = client
                    .get(format!("{base}/cog/preview/32x32.png?url={sat_url}&dst_crs={crs}"))
                    .send().await.unwrap();
                assert_eq!(r.status(), 200, "dst_crs={crs} should succeed");
            }
            // J6: /cog/preview/5000x5000.png error message states size exceeds limit
            let j6 = client
                .get(format!("{base}/cog/preview/5000x5000.png?url={sat_url}"))
                .send().await.unwrap();
            assert_eq!(j6.status(), 400);
            let j6_body = j6.text().await.unwrap();
            assert!(
                j6_body.contains("4096") && j6_body.to_ascii_lowercase().contains("exceed"),
                "expected size-limit message, got: {j6_body}"
            );

            // N4: mismatched rescale count (2 ranges for 3 bands) returns 400 naming the mismatch
            let n4_bad = client
                .get(format!("{base}/cog/tiles/WebMercatorQuad/12/652/1583.png?url={sat_url}&rescale=0,0.3&rescale=0,0.3"))
                .send().await.unwrap();
            assert_eq!(n4_bad.status(), 400);
            let n4_body = n4_bad.text().await.unwrap();
            assert!(
                n4_body.contains('2') && n4_body.contains('3'),
                "expected rescale count mismatch message naming 2 and 3, got: {n4_body}"
            );
            // 1 range (broadcast) and 3 ranges (per-band) remain valid
            let n4_one = client
                .get(format!("{base}/cog/tiles/WebMercatorQuad/12/652/1583.png?url={sat_url}&rescale=0,0.3"))
                .send().await.unwrap();
            assert_eq!(n4_one.status(), 200);
            let n4_three = client
                .get(format!("{base}/cog/tiles/WebMercatorQuad/12/652/1583.png?url={sat_url}&rescale=0,0.3&rescale=0,0.3&rescale=0,0.3"))
                .send().await.unwrap();
            assert_eq!(n4_three.status(), 200);
        }
    }
}

/// Write a minimal uncompressed single-band float32 GeoTIFF with `ProjectedCSTypeGeoKey = epsg`
/// so CI tests never depend on gitignored `benches/fixtures/`.
#[allow(clippy::too_many_arguments)]
fn write_synthetic_f32_projected_geotiff(
    path: &std::path::Path,
    width: u32,
    height: u32,
    epsg: u16,
    origin_x: f64,
    origin_y: f64,
    pixel_size: f64,
    pixels: &[f32],
) {
    assert_eq!(pixels.len(), (width * height) as usize);
    let mut buf = Vec::<u8>::new();
    // Little-endian classic TIFF header
    buf.extend_from_slice(b"II");
    buf.extend_from_slice(&42u16.to_le_bytes());
    buf.extend_from_slice(&8u32.to_le_bytes()); // IFD at offset 8

    let num_entries: u16 = 13;
    let ifd_size = 2 + (num_entries as usize) * 12 + 4;
    let data_start = 8 + ifd_size;

    let scale_off = data_start as u32;
    let tiepoint_off = scale_off + 24;
    let geokey_off = tiepoint_off + 48;
    let pixel_off = geokey_off + 24;
    let pixel_bytes = (pixels.len() * 4) as u32;

    let push_entry = |b: &mut Vec<u8>, tag: u16, ty: u16, count: u32, val: u32| {
        b.extend_from_slice(&tag.to_le_bytes());
        b.extend_from_slice(&ty.to_le_bytes());
        b.extend_from_slice(&count.to_le_bytes());
        b.extend_from_slice(&val.to_le_bytes());
    };

    buf.extend_from_slice(&num_entries.to_le_bytes());
    push_entry(&mut buf, 256, 4, 1, width); // ImageWidth
    push_entry(&mut buf, 257, 4, 1, height); // ImageLength
    push_entry(&mut buf, 258, 3, 1, 32); // BitsPerSample = 32
    push_entry(&mut buf, 259, 3, 1, 1); // Compression = None
    push_entry(&mut buf, 262, 3, 1, 1); // Photometric = MinIsBlack
    push_entry(&mut buf, 273, 4, 1, pixel_off); // StripOffsets
    push_entry(&mut buf, 277, 3, 1, 1); // SamplesPerPixel = 1
    push_entry(&mut buf, 278, 4, 1, height); // RowsPerStrip
    push_entry(&mut buf, 279, 4, 1, pixel_bytes); // StripByteCounts
    push_entry(&mut buf, 339, 3, 1, 3); // SampleFormat = IEEEFP (3)
    push_entry(&mut buf, 33550, 12, 3, scale_off); // ModelPixelScaleTag
    push_entry(&mut buf, 33922, 12, 6, tiepoint_off); // ModelTiepointTag
    push_entry(&mut buf, 34735, 3, 12, geokey_off); // GeoKeyDirectoryTag
    buf.extend_from_slice(&0u32.to_le_bytes()); // Next IFD = 0

    assert_eq!(buf.len(), scale_off as usize);
    for v in [pixel_size, pixel_size, 0.0f64] {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    assert_eq!(buf.len(), tiepoint_off as usize);
    for v in [0.0f64, 0.0, 0.0, origin_x, origin_y, 0.0] {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    assert_eq!(buf.len(), geokey_off as usize);
    // GeoKeyDirectory: 1,1,0,2 keys: GTModelTypeGeoKey(1024)=1 (Projected), ProjectedCSTypeGeoKey(3072)=epsg
    for v in [1u16, 1, 0, 2, 1024, 0, 1, 1, 3072, 0, 1, epsg] {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    assert_eq!(buf.len(), pixel_off as usize);
    for &p in pixels {
        buf.extend_from_slice(&p.to_le_bytes());
    }
    std::fs::write(path, &buf).unwrap();
}

fn decode_npy_f32(bytes: &[u8]) -> Vec<f32> {
    assert!(bytes.starts_with(b"\x93NUMPY"), "not an npy payload");
    let hlen = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let hdr = std::str::from_utf8(&bytes[10..10 + hlen]).unwrap();
    let payload = &bytes[10 + hlen..];
    if hdr.contains("'|u1'") {
        payload.iter().map(|&b| b as f32).collect()
    } else if hdr.contains("'|i1'") {
        payload.iter().map(|&b| (b as i8) as f32).collect()
    } else if hdr.contains("'<u2'") {
        payload
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]) as f32)
            .collect()
    } else if hdr.contains("'<i2'") {
        payload
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32)
            .collect()
    } else if hdr.contains("'<u4'") {
        payload
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
            .collect()
    } else if hdr.contains("'<i4'") {
        payload
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
            .collect()
    } else if hdr.contains("'<f8'") {
        payload
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32)
            .collect()
    } else {
        payload
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
}

#[tokio::test]
async fn sweep11_l1_l2_and_sweep12_m1_self_contained() {
    let tmp = std::env::temp_dir().join(format!("cogzilla_l1_l2_m1_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();

    // 1. Synthetic 64x64 float32 UTM 10N GeoTIFF (no overviews) for L1 average + L2 CRS tests
    let w = 64u32;
    let h = 64u32;
    let mut grad = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        for x in 0..w {
            grad.push(((y * 7 + x * 3) % 97) as f32 / 100.0 + (x as f32) * 0.01);
        }
    }
    let grad_path = tmp.join("grad_64_utm10n.tif");
    write_synthetic_f32_projected_geotiff(
        &grad_path,
        w,
        h,
        32610,
        500_000.0,
        4_200_000.0,
        30.0,
        &grad,
    );

    // 2. Synthetic 64x64 categorical float32 GeoTIFF for L1 mode test:
    // Each 4x4 block has 11 pixels of class `c` and 5 pixels of other classes.
    let mut cat = vec![0.0f32; (w * h) as usize];
    for by in 0..16u32 {
        for bx in 0..16u32 {
            let dominant = ((by * 16 + bx) % 10 + 1) as f32;
            for dy in 0..4u32 {
                for dx in 0..4u32 {
                    let v = if dy == 0 && dx < 2 {
                        99.0
                    } else if dy == 3 && dx >= 1 {
                        88.0
                    } else {
                        dominant
                    };
                    cat[((by * 4 + dy) * w + (bx * 4 + dx)) as usize] = v;
                }
            }
        }
    }
    let cat_path = tmp.join("cat_64_utm10n.tif");
    write_synthetic_f32_projected_geotiff(
        &cat_path,
        w,
        h,
        32610,
        500_000.0,
        4_200_000.0,
        30.0,
        &cat,
    );

    // 3. Synthetic 16x16 GeoTIFF in unsupported CRS EPSG:54030 (Robinson) for M1 test
    let unsup_path = tmp.join("robinson_54030.tif");
    write_synthetic_f32_projected_geotiff(
        &unsup_path,
        16,
        16,
        54030,
        0.0,
        100_000.0,
        1000.0,
        &vec![42.0f32; 256],
    );

    // 4. Synthetic 32x32 GeoTIFFs in EPSG:6933 (EASE-2 Global), EPSG:3410, and EPSG:8857 (Equal Earth)
    let ease6933_path = tmp.join("ease_6933.tif");
    write_synthetic_f32_projected_geotiff(
        &ease6933_path,
        32,
        32,
        6933,
        -1_000_000.0,
        4_000_000.0,
        25_000.0,
        &(0..1024).map(|i| (i % 50) as f32 + 10.0).collect::<Vec<_>>(),
    );
    let ease3410_path = tmp.join("ease_3410.tif");
    write_synthetic_f32_projected_geotiff(
        &ease3410_path,
        32,
        32,
        3410,
        -1_000_000.0,
        4_000_000.0,
        25_000.0,
        &vec![7.0f32; 1024],
    );
    let ee8857_path = tmp.join("equal_earth_8857.tif");
    write_synthetic_f32_projected_geotiff(
        &ee8857_path,
        32,
        32,
        8857,
        -1_000_000.0,
        4_000_000.0,
        25_000.0,
        &vec![13.0f32; 1024],
    );

    let engine = Arc::new(TileEngine::new(64 * 1024 * 1024));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let engine_srv = engine.clone();
    tokio::spawn(async move {
        let _ = serve_listener(engine_srv, listener).await;
    });
    let base = format!("http://{addr}");
    let client = reqwest::Client::new();

    let grad_url = grad_path.to_string_lossy().to_string();
    let cat_url = cat_path.to_string_lossy().to_string();
    let unsup_url = unsup_path.to_string_lossy().to_string();
    let ease6933_url = ease6933_path.to_string_lossy().to_string();
    let ease3410_url = ease3410_path.to_string_lossy().to_string();
    let ee8857_url = ee8857_path.to_string_lossy().to_string();

    // --- L1: resampling=average at 1x, 2x, 4x, 8x equals exact full-res block mean ---
    for factor in [1u32, 2, 4, 8] {
        let out_dim = w / factor;
        let resp = client
            .get(format!(
                "{base}/cog/preview.npy?url={grad_url}&max_size={out_dim}&resampling=average&return_mask=false"
            ))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200, "average preview factor={factor}");
        let vals = decode_npy_f32(&resp.bytes().await.unwrap());
        assert_eq!(vals.len(), (out_dim * out_dim) as usize);
        let mut max_err = 0.0f64;
        for by in 0..out_dim {
            for bx in 0..out_dim {
                let mut block_sum = 0.0f64;
                for dy in 0..factor {
                    for dx in 0..factor {
                        block_sum += grad[((by * factor + dy) * w + (bx * factor + dx)) as usize] as f64;
                    }
                }
                let expected = block_sum / ((factor * factor) as f64);
                let got = vals[(by * out_dim + bx) as usize] as f64;
                max_err = max_err.max((got - expected).abs());
            }
        }
        assert!(
            max_err < 1e-6,
            "L1 average at {factor}x reduction max_err={max_err} (expected < 1e-6)"
        );
    }

    // --- L1: resampling=mode at 4x matches exact 4x4 block mode ---
    let mode_resp = client
        .get(format!(
            "{base}/cog/preview.npy?url={cat_url}&max_size=16&resampling=mode&return_mask=false"
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(mode_resp.status(), 200);
    let mode_vals = decode_npy_f32(&mode_resp.bytes().await.unwrap());
    assert_eq!(mode_vals.len(), 256);
    for by in 0..16u32 {
        for bx in 0..16u32 {
            let expected = ((by * 16 + bx) % 10 + 1) as f32;
            let got = mode_vals[(by * 16 + bx) as usize];
            assert_eq!(got, expected, "L1 mode mismatch at ({bx},{by})");
        }
    }

    // --- L2: legacy GeoJSON `crs` member honoured on /cog/statistics and /cog/feature ---
    let utm_feature_with_crs = serde_json::json!({
        "type": "Feature",
        "crs": { "type": "name", "properties": { "name": "EPSG:32610" } },
        "properties": {},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [500_000.0, 4_200_000.0],
                [500_300.0, 4_200_000.0],
                [500_300.0, 4_199_700.0],
                [500_000.0, 4_199_700.0],
                [500_000.0, 4_200_000.0]
            ]]
        }
    });
    let st_crs = client
        .post(format!("{base}/cog/statistics?url={grad_url}&max_size=0"))
        .header("content-type", "application/json")
        .body(serde_json::to_string(&utm_feature_with_crs).unwrap())
        .send()
        .await
        .unwrap();
    assert_eq!(st_crs.status(), 200, "legacy GeoJSON crs should be honoured");
    let st_crs_json: serde_json::Value =
        serde_json::from_slice(&st_crs.bytes().await.unwrap()).unwrap();
    let count = st_crs_json["properties"]["statistics"]["b1"]["count"]
        .as_f64()
        .unwrap();
    assert!(
        (count - 100.0).abs() < 1e-4,
        "10x10 UTM pixel box with legacy GeoJSON crs should have count=100, got {count}"
    );

    // FeatureCollection with top-level `crs`
    let utm_fc_with_crs = serde_json::json!({
        "type": "FeatureCollection",
        "crs": { "type": "name", "properties": { "name": "EPSG:32610" } },
        "features": [{
            "type": "Feature",
            "properties": {},
            "geometry": utm_feature_with_crs["geometry"].clone()
        }]
    });
    let fc_crs = client
        .post(format!("{base}/cog/statistics?url={grad_url}&max_size=0"))
        .header("content-type", "application/json")
        .body(serde_json::to_string(&utm_fc_with_crs).unwrap())
        .send()
        .await
        .unwrap();
    assert_eq!(fc_crs.status(), 200);
    let fc_crs_json: serde_json::Value =
        serde_json::from_slice(&fc_crs.bytes().await.unwrap()).unwrap();
    let fc_count = fc_crs_json["features"][0]["properties"]["statistics"]["b1"]["count"]
        .as_f64()
        .unwrap();
    assert!((fc_count - 100.0).abs() < 1e-4, "FC legacy crs count={fc_count}");

    // /cog/feature with legacy `crs`
    let feat_crs = client
        .post(format!("{base}/cog/feature.npy?url={grad_url}&max_size=64"))
        .header("content-type", "application/json")
        .body(serde_json::to_string(&utm_feature_with_crs).unwrap())
        .send()
        .await
        .unwrap();
    assert_eq!(feat_crs.status(), 200, "/cog/feature should honour legacy GeoJSON crs");

    // --- L2: projected coordinates without coord_crs or `crs` return 400 with hint ---
    let utm_feature_no_crs = serde_json::json!({
        "type": "Feature",
        "properties": {},
        "geometry": utm_feature_with_crs["geometry"].clone()
    });
    for ep in [
        format!("{base}/cog/statistics?url={grad_url}&max_size=0"),
        format!("{base}/cog/feature.npy?url={grad_url}&max_size=64"),
    ] {
        let bad = client
            .post(&ep)
            .header("content-type", "application/json")
            .body(serde_json::to_string(&utm_feature_no_crs).unwrap())
            .send()
            .await
            .unwrap();
        assert_eq!(bad.status(), 400, "projected coords without coord_crs must 400 on {ep}");
        let body = bad.text().await.unwrap();
        assert!(
            body.contains("coord_crs"),
            "400 message should hint at coord_crs, got: {body}"
        );
    }

    // --- M1: unsupported CRS (EPSG:54030) returns 400 on /cog/tiles (matching /cog/info) ---
    let info_unsup = client
        .get(format!("{base}/cog/info?url={unsup_url}"))
        .send()
        .await
        .unwrap();
    assert_eq!(info_unsup.status(), 400);
    let info_body = info_unsup.text().await.unwrap();
    assert!(info_body.to_ascii_lowercase().contains("unsupported crs"), "got: {info_body}");

    let tile_unsup = client
        .get(format!("{base}/cog/tiles/WebMercatorQuad/0/0/0.png?url={unsup_url}"))
        .send()
        .await
        .unwrap();
    assert_eq!(
        tile_unsup.status(),
        400,
        "unsupported CRS tile must return 400 (not 404 outside image bounds)"
    );
    let tile_body = tile_unsup.text().await.unwrap();
    assert!(
        tile_body.to_ascii_lowercase().contains("unsupported crs"),
        "expected unsupported CRS message on tile route, got: {tile_body}"
    );

    // --- EPSG:6933, EPSG:3410 (EASE-Grid Global) and EPSG:8857 (Equal Earth) ---
    for (label, u) in [
        ("EPSG:6933", &ease6933_url),
        ("EPSG:3410", &ease3410_url),
        ("EPSG:8857", &ee8857_url),
    ] {
        let info = client.get(format!("{base}/cog/info?url={u}")).send().await.unwrap();
        assert_eq!(info.status(), 200, "{label} /cog/info should return 200");
        let tile = client
            .get(format!("{base}/cog/tiles/WebMercatorQuad/2/1/1.png?url={u}"))
            .send()
            .await
            .unwrap();
        assert_eq!(tile.status(), 200, "{label} /cog/tiles should return 200");
    }

    // Equal-area zonal statistics on EPSG:6933: every 25km x 25km pixel covers exactly 625,000,000 m^2,
    // so area == count * pixel_area and area_weighted=true/false produce identical mean.
    let ease_poly = serde_json::json!({
        "type": "Feature",
        "properties": {},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [-1_000_000.0, 4_000_000.0],
                [-750_000.0,   4_000_000.0],
                [-750_000.0,   3_750_000.0],
                [-1_000_000.0, 3_750_000.0],
                [-1_000_000.0, 4_000_000.0]
            ]]
        }
    });
    let st_aw = client
        .post(format!("{base}/cog/statistics?url={ease6933_url}&max_size=0&coord_crs=EPSG:6933&area_weighted=true"))
        .header("content-type", "application/json")
        .body(serde_json::to_string(&ease_poly).unwrap())
        .send()
        .await
        .unwrap();
    assert_eq!(st_aw.status(), 200);
    let st_aw_j: serde_json::Value = serde_json::from_slice(&st_aw.bytes().await.unwrap()).unwrap();
    let st_un = client
        .post(format!("{base}/cog/statistics?url={ease6933_url}&max_size=0&coord_crs=EPSG:6933&area_weighted=false"))
        .header("content-type", "application/json")
        .body(serde_json::to_string(&ease_poly).unwrap())
        .send()
        .await
        .unwrap();
    assert_eq!(st_un.status(), 200);
    let st_un_j: serde_json::Value = serde_json::from_slice(&st_un.bytes().await.unwrap()).unwrap();
    let b1_aw = &st_aw_j["properties"]["statistics"]["b1"];
    let b1_un = &st_un_j["properties"]["statistics"]["b1"];
    let count_aw = b1_aw["count"].as_f64().unwrap();
    let area_aw = b1_aw["area"].as_f64().unwrap();
    let mean_aw = b1_aw["mean"].as_f64().unwrap();
    let mean_un = b1_un["mean"].as_f64().unwrap();
    assert!((count_aw - 100.0).abs() < 1e-4, "expected 100 px, got {count_aw}");
    let expected_area = 100.0 * 25_000.0 * 25_000.0;
    let rel_area_err = (area_aw - expected_area).abs() / expected_area;
    assert!(
        rel_area_err < 1e-5,
        "EPSG:6933 equal-area should match 100 * 25km^2 = {expected_area}, got {area_aw} (rel_err={rel_area_err})"
    );
    assert!(
        (mean_aw - mean_un).abs() < 1e-12,
        "equal-area EPSG:6933 area_weighted=true ({mean_aw}) and false ({mean_un}) must match"
    );

    let _ = std::fs::remove_dir_all(&tmp);
}

#[tokio::test]
async fn uniform_semantic_http_statuses_across_all_tms() {
    let tmp = std::env::temp_dir().join(format!("cogzilla_statuses_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();

    // Small UTM 10N raster in Northern California (~ -122.99..-122.97, 37.94..37.95)
    let utm_path = tmp.join("small_utm10n.tif");
    write_synthetic_f32_projected_geotiff(
        &utm_path,
        32,
        32,
        32610,
        500_000.0,
        4_200_000.0,
        30.0,
        &vec![123.0f32; 1024],
    );

    // All-NaN (nodata) raster at the exact same location
    let nodata_path = tmp.join("all_nodata_utm10n.tif");
    write_synthetic_f32_projected_geotiff(
        &nodata_path,
        32,
        32,
        32610,
        500_000.0,
        4_200_000.0,
        30.0,
        &vec![f32::NAN; 1024],
    );

    // MosaicJSON with wide advertised bounds [-125, 35, -120, 40] but only one tile covered
    let mj_path = tmp.join("gap_mosaic.json");
    let utm_url = utm_path.to_string_lossy().to_string();
    let nodata_url = nodata_path.to_string_lossy().to_string();
    let mj_json = serde_json::json!({
        "mosaicjson": "0.0.3",
        "minzoom": 8,
        "maxzoom": 12,
        "bounds": [-125.0, 35.0, -120.0, 40.0],
        "quadkey_zoom": 8,
        "tiles": {
            "02301020": [utm_url]
        }
    });
    std::fs::write(&mj_path, mj_json.to_string()).unwrap();
    let mj_url = mj_path.to_string_lossy().to_string();

    let engine = Arc::new(TileEngine::new(32 * 1024 * 1024));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let engine_srv = engine.clone();
    tokio::spawn(async move {
        let _ = serve_listener(engine_srv, listener).await;
    });
    let base = format!("http://{addr}");
    let client = reqwest::Client::new();

    // 1. Out-of-matrix x returns 400 Bad Request
    let bad_x = client
        .get(format!("{base}/cog/tiles/WebMercatorQuad/4/99/0.png?url={utm_url}"))
        .send()
        .await
        .unwrap();
    assert_eq!(bad_x.status(), 400, "out-of-matrix x must return 400");

    // 2. Every single one of the 13 TileMatrixSets returns 404 for a valid tile outside the UTM file
    let outside_tiles = [
        ("WebMercatorQuad", "4/15/15"),
        ("WorldCRS84Quad", "4/31/15"),
        ("WorldMercatorWGS84Quad", "4/15/15"),
        ("WGS1984Quad", "4/31/15"),
        ("UPSArcticWGS84Quad", "4/0/0"),
        ("UPSAntarcticWGS84Quad", "4/0/0"),
        ("CanadianNAD83_LCC", "4/0/0"),
        ("EuropeanETRS89_LAEAQuad", "4/0/0"),
        ("NZTM2000Quad", "4/0/0"),
        ("LINZAntarticaMapTilegrid", "4/0/0"),
        ("UTM31WGS84Quad", "4/0/0"),
        ("CDB1GlobalGrid", "4/0/0"),
        ("GNOSISGlobalGrid", "4/0/0"),
    ];
    for (tms, zxy) in outside_tiles {
        let r = client
            .get(format!("{base}/cog/tiles/{tms}/{zxy}.png?url={utm_url}"))
            .send()
            .await
            .unwrap();
        let status = r.status();
        let body = r.text().await.unwrap();
        assert_eq!(
            status, 404,
            "TMS {tms}/{zxy} outside UTM file must return 404, got {status}: {body}"
        );
        assert!(
            body.contains("outside") && body.contains(tms),
            "404 body for {tms} should name tile & bounds, got: {body}"
        );
    }

    // 3. All-nodata tile INSIDE dataset bounds returns 200 OK
    // UTM 10N (500_000, 4_200_000) is (-123.000, 37.947) -> WebMercator tile 10/162/395
    let nd_resp = client
        .get(format!("{base}/cog/tiles/WebMercatorQuad/10/162/395.png?url={nodata_url}"))
        .send()
        .await
        .unwrap();
    assert_eq!(
        nd_resp.status(),
        200,
        "all-nodata tile inside footprint must return 200"
    );

    // 4. Mosaic gap INSIDE advertised mj.bounds returns 204; tile OUTSIDE mj.bounds returns 404
    // (-121.0, 36.0) is inside [-125, 35, -120, 40] -> WebMercator z=8 x=41, y=99 (uncovered gap)
    let gap_resp = client
        .get(format!("{base}/mosaicjson/tiles/WebMercatorQuad/8/41/99.png?url={mj_url}"))
        .send()
        .await
        .unwrap();
    assert_eq!(gap_resp.status(), 204, "mosaic gap inside bounds must return 204");
    let out_mj = client
        .get(format!("{base}/mosaicjson/tiles/WebMercatorQuad/8/0/0.png?url={mj_url}"))
        .send()
        .await
        .unwrap();
    assert_eq!(out_mj.status(), 404, "mosaic tile outside bounds must return 404");

    // 5. /cog/point, /cog/bbox, and /cog/feature outside footprint all return 404 with detail
    let pt_out = client
        .get(format!("{base}/cog/point/0,0?url={utm_url}"))
        .send()
        .await
        .unwrap();
    assert_eq!(pt_out.status(), 404);
    let bbox_out = client
        .get(format!("{base}/cog/bbox/0,0,1,1/32x32.png?url={utm_url}"))
        .send()
        .await
        .unwrap();
    assert_eq!(bbox_out.status(), 404);
    let feat_out = client
        .post(format!("{base}/cog/feature.png?url={utm_url}&max_size=32"))
        .header("content-type", "application/json")
        .body(r#"{"type":"Feature","properties":{},"geometry":{"type":"Polygon","coordinates":[[[0,0],[1,0],[1,1],[0,1],[0,0]]]}}"#)
        .send()
        .await
        .unwrap();
    assert_eq!(feat_out.status(), 404);

    let _ = std::fs::remove_dir_all(&tmp);
}

#[tokio::test]
async fn s1_cylindrical_equal_area_exact_in_worldcrs84_and_worldmercator() {
    let tmp = std::env::temp_dir().join(format!("cogzilla_s1_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();

    // Global 512x256 ramp rasters in EPSG:6933 and EPSG:3410 (~67.8 km pixels):
    // pixel value = row * 512 + col so any 1-row shift shows up immediately as delta=512.
    let w = 512u32;
    let h = 256u32;
    let ramp: Vec<f32> = (0..(w * h)).map(|i| i as f32).collect();

    let g6933_path = tmp.join("g6933_ramp.tif");
    write_synthetic_f32_projected_geotiff(
        &g6933_path,
        w,
        h,
        6933,
        -17_367_520.0,
        7_314_540.0,
        34_735_040.0 / (w as f64),
        &ramp,
    );
    let g3410_path = tmp.join("g3410_ramp.tif");
    write_synthetic_f32_projected_geotiff(
        &g3410_path,
        w,
        h,
        3410,
        -17_334_200.0,
        7_344_784.0,
        34_668_400.0 / (w as f64),
        &ramp,
    );

    let engine = Arc::new(TileEngine::new(32 * 1024 * 1024));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let engine_srv = engine.clone();
    tokio::spawn(async move {
        let _ = serve_listener(engine_srv, listener).await;
    });
    let base = format!("http://{addr}");
    let client = reqwest::Client::new();

    for (epsg, path, ox, oy, px_sz) in [
        (6933u16, &g6933_path, -17_367_520.0, 7_314_540.0, 34_735_040.0 / (w as f64)),
        (3410u16, &g3410_path, -17_334_200.0, 7_344_784.0, 34_668_400.0 / (w as f64)),
    ] {
        let crs = cogzilla_proj::CrsModel::from_epsg(epsg).unwrap();
        let u = path.to_string_lossy().to_string();

        // Check WorldCRS84Quad z2/2/1 (lon 0..45, lat 0..45) and WorldMercatorWGS84Quad z2/2/1
        for (tms, z, x, y, dst_crs, tb) in [
            (
                "WorldCRS84Quad",
                2u8,
                2u32,
                1u32,
                cogzilla_proj::CrsModel::Geographic,
                [-90.0, 0.0, -45.0, 45.0],
            ),
            (
                "WorldMercatorWGS84Quad",
                2u8,
                2u32,
                1u32,
                cogzilla_proj::CrsModel::WorldMercator,
                [0.0, 0.0, 10_018_754.171_394_622, 10_018_754.171_394_622],
            ),
        ] {
            let resp = client
                .get(format!(
                    "{base}/cog/tiles/{tms}/{z}/{x}/{y}.npy?url={u}&bidx=1&resampling=nearest&reproject=nearest&return_mask=false"
                ))
                .send()
                .await
                .unwrap();
            assert_eq!(resp.status(), 200, "EPSG:{epsg} {tms} {z}/{x}/{y}");
            let vals = decode_npy_f32(&resp.bytes().await.unwrap());
            assert_eq!(vals.len(), 256 * 256);

            let dlx = (tb[2] - tb[0]) / 256.0;
            let dly = (tb[3] - tb[1]) / 256.0;
            let mut mismatches = 0usize;
            for py in 0..256usize {
                let dy = tb[3] - (py as f64 + 0.5) * dly;
                for px in 0..256usize {
                    let dx = tb[0] + (px as f64 + 0.5) * dlx;
                    let (lon, lat) = dst_crs.to_lonlat_deg(dx, dy);
                    let (sx, sy) = crs.from_lonlat_deg(lon, lat);
                    let col = ((sx - ox) / px_sz).floor() as i64;
                    let row = ((oy - sy) / px_sz).floor() as i64;
                    if col >= 0 && col < w as i64 && row >= 0 && row < h as i64 {
                        let expected = (row as u32 * w + col as u32) as f32;
                        let got = vals[py * 256 + px];
                        if (got - expected).abs() > 0.5 {
                            mismatches += 1;
                        }
                    }
                }
            }
            assert_eq!(
                mismatches, 0,
                "S1 EPSG:{epsg} in {tms} {z}/{x}/{y} had {mismatches} mismatched pixels (expected 0)"
            );
        }
    }

    let _ = std::fs::remove_dir_all(&tmp);
}

#[tokio::test]
async fn sweep15_t1_t2_t3_t4_self_contained() {
    let tmp = std::env::temp_dir().join(format!("cogzilla_sweep15_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();

    // 64x64 float32 UTM 10N raster with extreme outliers outside [10..13, 10..13] and
    // 9 distinct float values ~0.17..0.25 inside [10..13, 10..13].
    let w = 64u32;
    let h = 64u32;
    let ox = 500_000.0_f64;
    let oy = 4_200_000.0_f64;
    let px = 30.0_f64;
    let mut data = vec![999.0f32; (w * h) as usize];
    for r in 0..h {
        for c in 0..w {
            if r == 9 || r == 13 || c == 9 || c == 13 {
                data[(r * w + c) as usize] = -999.0;
            }
        }
    }
    for r in 10..13u32 {
        for c in 10..13u32 {
            let v = 0.17_f32 + ((r - 10) * 3 + (c - 10)) as f32 * 0.01;
            data[(r * w + c) as usize] = v;
        }
    }
    let utm_path = tmp.join("t1_t2_t3_utm.tif");
    write_synthetic_f32_projected_geotiff(&utm_path, w, h, 32610, ox, oy, px, &data);

    // 16x16 Web Mercator categorical raster spanning ~20N..50N (strong cos^2(lat) area variation)
    let mut merc_cat = vec![1.0f32; 256];
    for r in 8..16 {
        for c in 0..16 {
            merc_cat[r * 16 + c] = 2.0;
        }
    }
    let merc_path = tmp.join("t4_merc_cat.tif");
    write_synthetic_f32_projected_geotiff(
        &merc_path,
        16,
        16,
        3857,
        0.0,
        6_446_275.841_017_158,
        180_000.0,
        &merc_cat,
    );

    let engine = Arc::new(TileEngine::new(32 * 1024 * 1024));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let engine_srv = engine.clone();
    tokio::spawn(async move {
        let _ = serve_listener(engine_srv, listener).await;
    });
    let base = format!("http://{addr}");
    let client = reqwest::Client::new();
    let utm_url = utm_path.to_string_lossy().to_string();
    let merc_url = merc_path.to_string_lossy().to_string();

    // --- T1 & T2: aligned 3x3 UTM polygon through real coord_crs=EPSG:32610 round-trip ---
    let sq_3x3 = serde_json::json!({
        "type": "Feature",
        "properties": {},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [ox + 10.0 * px, oy - 10.0 * px],
                [ox + 13.0 * px, oy - 10.0 * px],
                [ox + 13.0 * px, oy - 13.0 * px],
                [ox + 10.0 * px, oy - 13.0 * px],
                [ox + 10.0 * px, oy - 10.0 * px]
            ]]
        }
    });
    let r_t12 = client
        .post(format!("{base}/cog/statistics?url={utm_url}&bidx=1&max_size=0&coord_crs=EPSG:32610&categorical=true"))
        .header("content-type", "application/json")
        .body(serde_json::to_string(&sq_3x3).unwrap())
        .send()
        .await
        .unwrap();
    assert_eq!(r_t12.status(), 200);
    let j_t12: serde_json::Value = serde_json::from_slice(&r_t12.bytes().await.unwrap()).unwrap();
    let b1 = &j_t12["properties"]["statistics"]["b1"];
    assert_eq!(b1["valid_pixels"].as_f64().unwrap(), 9.0, "T1 valid_pixels must be 9, got {b1}");
    assert_eq!(b1["count"].as_f64().unwrap(), 9.0);
    assert!(
        (b1["min"].as_f64().unwrap() - 0.17).abs() < 1e-5,
        "T1 min polluted by outlier neighbor: {}",
        b1["min"]
    );
    assert!(
        (b1["max"].as_f64().unwrap() - 0.25).abs() < 1e-5,
        "T1 max polluted by outlier neighbor: {}",
        b1["max"]
    );
    // T2: categorical=true on float data preserves all 9 distinct float values (not truncated to 0)
    let hist_counts = b1["histogram"][0].as_array().unwrap();
    let hist_vals = b1["histogram"][1].as_array().unwrap();
    assert_eq!(hist_vals.len(), 9, "T2 float categorical must preserve 9 distinct values, got {hist_vals:?}");
    assert_eq!(hist_counts.len(), 9);
    assert!((hist_vals[0].as_f64().unwrap() - 0.17).abs() < 1e-5);

    // --- T3: POST /cog/feature/32x32.npy matches GET /cog/bbox/.../32x32.npy identically ---
    let full_rect = serde_json::json!({
        "type": "Feature",
        "properties": {},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [ox, oy],
                [ox + 64.0 * px, oy],
                [ox + 64.0 * px, oy - 64.0 * px],
                [ox, oy - 64.0 * px],
                [ox, oy]
            ]]
        }
    });
    let bbox_str = format!("{},{},{},{}", ox, oy - 64.0 * px, ox + 64.0 * px, oy);
    let r_feat = client
        .post(format!("{base}/cog/feature/32x32.npy?url={utm_url}&bidx=1&coord_crs=EPSG:32610&dst_crs=EPSG:32610&resampling=average&return_mask=false"))
        .header("content-type", "application/json")
        .body(serde_json::to_string(&full_rect).unwrap())
        .send()
        .await
        .unwrap();
    assert_eq!(r_feat.status(), 200);
    let r_bbox = client
        .get(format!("{base}/cog/bbox/{bbox_str}/32x32.npy?url={utm_url}&bidx=1&coord_crs=EPSG:32610&dst_crs=EPSG:32610&resampling=average&return_mask=false"))
        .send()
        .await
        .unwrap();
    assert_eq!(r_bbox.status(), 200);
    let v_feat = decode_npy_f32(&r_feat.bytes().await.unwrap());
    let v_bbox = decode_npy_f32(&r_bbox.bytes().await.unwrap());
    assert_eq!(v_feat.len(), v_bbox.len());
    let max_diff = v_feat
        .iter()
        .zip(v_bbox.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_diff < 1e-6, "T3 /feature vs /bbox max_diff={max_diff}");

    // --- T4: histogram & categories in coverage units even with area_weighted=true + category_areas ---
    let merc_poly = serde_json::json!({
        "type": "Feature",
        "properties": {},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [0.0, 6_446_275.841_017_158],
                [16.0 * 180_000.0, 6_446_275.841_017_158],
                [16.0 * 180_000.0, 6_446_275.841_017_158 - 16.0 * 180_000.0],
                [0.0, 6_446_275.841_017_158 - 16.0 * 180_000.0],
                [0.0, 6_446_275.841_017_158]
            ]]
        }
    });
    let r_t4 = client
        .post(format!("{base}/cog/statistics?url={merc_url}&bidx=1&max_size=0&coord_crs=EPSG:3857&categorical=true&area_weighted=true"))
        .header("content-type", "application/json")
        .body(serde_json::to_string(&merc_poly).unwrap())
        .send()
        .await
        .unwrap();
    assert_eq!(r_t4.status(), 200);
    let j_t4: serde_json::Value = serde_json::from_slice(&r_t4.bytes().await.unwrap()).unwrap();
    let b1_t4 = &j_t4["properties"]["statistics"]["b1"];
    assert_eq!(b1_t4["categories"]["1"].as_f64().unwrap(), 128.0);
    assert_eq!(b1_t4["categories"]["2"].as_f64().unwrap(), 128.0);
    assert_eq!(b1_t4["histogram"], serde_json::json!([[128.0, 128.0], [1.0, 2.0]]));
    let a1 = b1_t4["category_areas"]["1"].as_f64().unwrap();
    let a2 = b1_t4["category_areas"]["2"].as_f64().unwrap();
    let total_area = b1_t4["area"].as_f64().unwrap();
    assert!(a2 > a1, "southern class 2 covers more ground area than northern class 1 in EPSG:3857");
    assert!(
        ((a1 + a2) - total_area).abs() / total_area < 1e-9,
        "category_areas must sum to total area"
    );
    assert!(b1_t4["mean"].as_f64().unwrap() > 1.55);

    let _ = std::fs::remove_dir_all(&tmp);
}

#[tokio::test]
async fn sweep16_u1_stac_post_expr_and_u2a_gcp_message() {
    let tmp = std::env::temp_dir().join(format!("cogzilla_sweep16_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();

    // Two aligned 64x64 UTM 10N single-band assets (`red` = 30.0, `green` = 10.0) in a STAC item
    let w = 64u32;
    let h = 64u32;
    let ox = 500_000.0_f64;
    let oy = 4_200_000.0_f64;
    let px = 30.0_f64;
    let red_path = tmp.join("red.tif");
    let green_path = tmp.join("green.tif");
    write_synthetic_f32_projected_geotiff(&red_path, w, h, 32610, ox, oy, px, &vec![30.0f32; 4096]);
    write_synthetic_f32_projected_geotiff(&green_path, w, h, 32610, ox, oy, px, &vec![10.0f32; 4096]);

    let stac_path = tmp.join("item.json");
    let stac = serde_json::json!({
        "type": "Feature",
        "stac_version": "1.0.0",
        "id": "u1-test",
        "bbox": [-123.0, 37.9, -122.9, 38.0],
        "geometry": null,
        "properties": {},
        "assets": {
            "red": { "href": red_path.to_string_lossy(), "type": "image/tiff; application=geotiff" },
            "green": { "href": green_path.to_string_lossy(), "type": "image/tiff; application=geotiff" }
        }
    });
    std::fs::write(&stac_path, stac.to_string()).unwrap();

    // Synthetic GCP-only TIFF (ModelTiepointTag with 4 GCPs = 24 f64s, no ModelPixelScaleTag)
    let gcp_path = tmp.join("gcp_only.tif");
    {
        let mut buf = Vec::<u8>::new();
        buf.extend_from_slice(b"II");
        buf.extend_from_slice(&42u16.to_le_bytes());
        buf.extend_from_slice(&8u32.to_le_bytes());
        let num_entries: u16 = 12;
        let data_start = (8 + 2 + (num_entries as usize) * 12 + 4) as u32;
        let tp_off = data_start;
        let gkd_off = tp_off + 24 * 8;
        let pix_off = gkd_off + 12 * 2;
        let push = |b: &mut Vec<u8>, tag: u16, ty: u16, count: u32, val: u32| {
            b.extend_from_slice(&tag.to_le_bytes());
            b.extend_from_slice(&ty.to_le_bytes());
            b.extend_from_slice(&count.to_le_bytes());
            b.extend_from_slice(&val.to_le_bytes());
        };
        buf.extend_from_slice(&num_entries.to_le_bytes());
        push(&mut buf, 256, 4, 1, 8);
        push(&mut buf, 257, 4, 1, 8);
        push(&mut buf, 258, 3, 1, 32);
        push(&mut buf, 259, 3, 1, 1);
        push(&mut buf, 262, 3, 1, 1);
        push(&mut buf, 273, 4, 1, pix_off);
        push(&mut buf, 277, 3, 1, 1);
        push(&mut buf, 278, 4, 1, 8);
        push(&mut buf, 279, 4, 1, 8 * 8 * 4);
        push(&mut buf, 339, 3, 1, 3);
        push(&mut buf, 33922, 12, 24, tp_off); // 4 GCPs
        push(&mut buf, 34735, 3, 12, gkd_off); // GeoKeyDirectory: EPSG:4326
        buf.extend_from_slice(&0u32.to_le_bytes());
        for gcp in [
            [0.0f64, 0.0, 0.0, -122.5, 37.9, 0.0],
            [8.0, 0.0, 0.0, -122.4, 37.9, 0.0],
            [0.0, 8.0, 0.0, -122.5, 37.8, 0.0],
            [8.0, 8.0, 0.0, -122.4, 37.8, 0.0],
        ] {
            for v in gcp {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
        for k in [1u16, 1, 0, 2, 1024, 0, 1, 2, 2048, 0, 1, 4326] {
            buf.extend_from_slice(&k.to_le_bytes());
        }
        for _ in 0..64 {
            buf.extend_from_slice(&1.0f32.to_le_bytes());
        }
        std::fs::write(&gcp_path, &buf).unwrap();
    }

    let engine = Arc::new(TileEngine::new(32 * 1024 * 1024));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let engine_srv = engine.clone();
    tokio::spawn(async move {
        let _ = serve_listener(engine_srv, listener).await;
    });
    let base = format!("http://{addr}");
    let client = reqwest::Client::new();
    let stac_url = stac_path.to_string_lossy().to_string();
    let gcp_url = gcp_path.to_string_lossy().to_string();

    // U1: Tilted quad in pixel space -> exact shoelace area = 305.5 px (definitely not 4096 or 16M!)
    let pts = [(10.0, 10.0), (30.0, 12.0), (28.0, 27.0), (9.0, 25.0)];
    let mut ring: Vec<[f64; 2]> = pts.iter().map(|&(c, r)| [ox + c * px, oy - r * px]).collect();
    ring.push(ring[0]);
    let poly = serde_json::json!({
        "type": "Feature",
        "properties": {},
        "geometry": { "type": "Polygon", "coordinates": [ring] }
    });
    // Exact shoelace area in pixel space:
    let mut area2 = 0.0_f64;
    for i in 0..pts.len() {
        let (x0, y0) = pts[i];
        let (x1, y1) = pts[(i + 1) % pts.len()];
        area2 += x0 * y1 - x1 * y0;
    }
    let true_px = 0.5 * area2.abs();

    for (label, query, expected_mean) in [
        ("single-asset red/2", "assets=red&asset_as_band=true&expression=red/2", 15.0),
        ("two-asset red_b1-green_b1", "expression=(red_b1-green_b1)/(red_b1%2Bgreen_b1)", 0.5),
        ("two-asset asset_as_band", "assets=red&assets=green&asset_as_band=true&expression=(red-green)/(red%2Bgreen)", 0.5),
    ] {
        let r = client
            .post(format!("{base}/stac/statistics?url={stac_url}&{query}&max_size=0&coord_crs=EPSG:32610"))
            .header("content-type", "application/json")
            .body(serde_json::to_string(&poly).unwrap())
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 200, "U1 {label}");
        let j: serde_json::Value = serde_json::from_slice(&r.bytes().await.unwrap()).unwrap();
        let b1 = &j["properties"]["statistics"]["b1"];
        let cnt = b1["count"].as_f64().unwrap();
        let mean = b1["mean"].as_f64().unwrap();
        assert!(
            (cnt - true_px).abs() < 1e-4,
            "U1 {label}: expected count={true_px}, got {cnt}"
        );
        assert!(
            (mean - expected_mean).abs() < 1e-5,
            "U1 {label}: expected mean={expected_mean}, got {mean}"
        );
    }

    // U2(b): 4-GCP affine TIFF now succeeds with 200, gcp_residual_px ~ 0, and exact /cog/point!
    let r_gcp = client
        .get(format!("{base}/cog/info?url={gcp_url}"))
        .send()
        .await
        .unwrap();
    assert_eq!(r_gcp.status(), 200);
    let gcp_info: serde_json::Value = serde_json::from_slice(&r_gcp.bytes().await.unwrap()).unwrap();
    let res_px = gcp_info["gcp_residual_px"].as_f64().expect("gcp_residual_px present");
    assert!(res_px < 1e-9, "expected exact affine residual < 1e-9, got {res_px}");
    let r_pt = client
        .get(format!("{base}/cog/point/-122.45,37.85?url={gcp_url}"))
        .send()
        .await
        .unwrap();
    assert_eq!(r_pt.status(), 200);
    let pt_json: serde_json::Value = serde_json::from_slice(&r_pt.bytes().await.unwrap()).unwrap();
    assert_eq!(pt_json["values"][0].as_f64().unwrap(), 1.0);

    let _ = std::fs::remove_dir_all(&tmp);
}


#[tokio::test]
async fn upstream_http_faults_and_uri_schemes_return_semantic_statuses() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    // Spawn a minimal raw TCP mock origin supporting fault routes:
    // /e401, /e403, /e404, /e500, /e503, /norange, /trunc, /redirloop
    let mock_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let mock_addr = mock_listener.local_addr().unwrap();
    let mock_task = tokio::spawn(async move {
        loop {
            let Ok((mut sock, _)) = mock_listener.accept().await else {
                break;
            };
            tokio::spawn(async move {
                let mut buf = [0u8; 2048];
                let n = sock.read(&mut buf).await.unwrap_or(0);
                if n == 0 {
                    return;
                }
                let req = String::from_utf8_lossy(&buf[..n]);
                let path = req
                    .lines()
                    .next()
                    .and_then(|l| l.split_whitespace().nth(1))
                    .unwrap_or("/");
                let resp: Vec<u8> = if path.starts_with("/e401") {
                    b"HTTP/1.1 401 Unauthorized\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_vec()
                } else if path.starts_with("/e403") {
                    b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_vec()
                } else if path.starts_with("/e404") {
                    b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_vec()
                } else if path.starts_with("/e500") {
                    b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_vec()
                } else if path.starts_with("/e503") {
                    b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_vec()
                } else if path.starts_with("/norange") {
                    // Origin ignores Range and advertises a 500 KiB 200 OK body.
                    b"HTTP/1.1 200 OK\r\nContent-Length: 512000\r\nConnection: close\r\n\r\n".to_vec()
                } else if path.starts_with("/trunc") {
                    // Claims 4096 bytes in Content-Length, sends only 16 bytes, then closes.
                    b"HTTP/1.1 206 Partial Content\r\nContent-Length: 4096\r\nConnection: close\r\n\r\n0123456789abcdef".to_vec()
                } else if path.starts_with("/redirloop") {
                    format!(
                        "HTTP/1.1 302 Found\r\nLocation: http://{mock_addr}/redirloop\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                    )
                    .into_bytes()
                } else {
                    b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_vec()
                };
                let _ = sock.write_all(&resp).await;
                let _ = sock.shutdown().await;
            });
        }
    });

    // Reserve a closed port for connection-refused testing.
    let closed_port = {
        let l = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        l.local_addr().unwrap().port()
    };

    let engine = Arc::new(TileEngine::new(16 * 1024 * 1024));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let srv_task = tokio::spawn(async move {
        let _ = serve_listener(engine, listener).await;
    });
    let base = format!("http://{addr}");
    let client = reqwest::Client::new();

    // V1: COG upstream statuses
    let r = client
        .get(format!("{base}/cog/info?url=http://{mock_addr}/e401/a.tif"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 403);
    let body = r.text().await.unwrap();
    assert!(body.contains("origin refused access") && body.contains("401"), "body={body}");

    let r = client
        .get(format!("{base}/cog/info?url=http://{mock_addr}/e403/a.tif"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 403);
    let body = r.text().await.unwrap();
    assert!(body.contains("origin refused access") && body.contains("403"), "body={body}");

    let r = client
        .get(format!("{base}/cog/info?url=http://{mock_addr}/e404/a.tif"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 404);

    let r = client
        .get(format!("{base}/cog/info?url=http://{mock_addr}/e500/a.tif"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 502);
    let body = r.text().await.unwrap();
    assert!(body.contains("500") && body.contains("/e500/a.tif"), "body={body}");

    let r = client
        .get(format!("{base}/cog/info?url=http://{mock_addr}/e503/a.tif"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 502);
    let body = r.text().await.unwrap();
    assert!(body.contains("503") && body.contains("/e503/a.tif"), "body={body}");

    let r = client
        .get(format!("{base}/cog/info?url=http://{mock_addr}/trunc/a.tif"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 502);

    let r = client
        .get(format!("{base}/cog/info?url=http://{mock_addr}/redirloop/a.tif"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 502);

    let r = client
        .get(format!("{base}/cog/info?url=http://127.0.0.1:{closed_port}/a.tif"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 502);

    // V2: Origin ignores Range header (200 OK with full file > 128 KiB) -> 502 with clear message
    let r = client
        .get(format!("{base}/cog/info?url=http://{mock_addr}/norange/a.tif"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 502);
    let body = r.text().await.unwrap();
    assert!(
        body.contains("origin does not support HTTP range requests"),
        "body={body}"
    );

    // V3: Unsupported URI schemes -> 400 naming the scheme and supported ones
    for route in ["/cog/info", "/stac/info", "/mosaicjson/info"] {
        let r = client
            .get(format!("{base}{route}?url=ftp://127.0.0.1/x.tif"))
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 400, "route={route}");
        let body = r.text().await.unwrap();
        assert!(body.contains("ftp://") && body.contains("supported"), "route={route} body={body}");
    }

    // V1 on STAC & MosaicJSON routes too
    for route in ["/stac/info", "/mosaicjson/info"] {
        let r = client
            .get(format!("{base}{route}?url=http://{mock_addr}/e403/item.json"))
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 403, "route={route}");
        let r = client
            .get(format!("{base}{route}?url=http://{mock_addr}/e500/item.json"))
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 502, "route={route}");
        let r = client
            .get(format!("{base}{route}?url=http://{mock_addr}/e404/item.json"))
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 404, "route={route}");
    }

    mock_task.abort();
    srv_task.abort();
}


#[tokio::test]
async fn sweep17_v4_non_cog_ifd_seeking_and_v5_tight_block_fetch() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    // Build a 512x512 float32 tiled (256x256 blocks -> 2x2 = 4 blocks) GeoTIFF in non-COG layout:
    // header (8 B) -> 4 blocks of 256 KiB (1 MiB) -> 1 MiB padding -> IFD at offset > 2 MiB!
    let mut tif: Vec<u8> = Vec::new();
    tif.extend_from_slice(b"II");
    tif.extend_from_slice(&42u16.to_le_bytes());
    let first_ifd_patch_pos = tif.len();
    tif.extend_from_slice(&0u32.to_le_bytes()); // patched later

    let block_bytes = 256usize * 256 * 4; // 262,144 bytes per block
    let mut tile_offsets = [0u32; 4];
    let tile_counts = [block_bytes as u32; 4];
    for (b, off_slot) in tile_offsets.iter_mut().enumerate() {
        *off_slot = tif.len() as u32;
        let val = (b as f32 + 1.0) * 10.0;
        let vb = val.to_le_bytes();
        for _ in 0..(256 * 256) {
            tif.extend_from_slice(&vb);
        }
    }
    // Pad to > 2 MiB before writing IFD so a contiguous header read would have to fetch > 2 MiB.
    tif.resize(2 * 1024 * 1024 + 4096, 0u8);

    // Out-of-line tag payloads placed right next to the IFD.
    let offsets_pos = tif.len() as u32;
    for o in tile_offsets {
        tif.extend_from_slice(&o.to_le_bytes());
    }
    let counts_pos = tif.len() as u32;
    for c in tile_counts {
        tif.extend_from_slice(&c.to_le_bytes());
    }
    let scale_pos = tif.len() as u32;
    for v in [0.001f64, 0.001, 0.0] {
        tif.extend_from_slice(&v.to_le_bytes());
    }
    let tie_pos = tif.len() as u32;
    for v in [0.0f64, 0.0, 0.0, 10.0, 45.0, 0.0] {
        tif.extend_from_slice(&v.to_le_bytes());
    }
    let gkd_pos = tif.len() as u32;
    // GeoKeyDirectory: WGS84 Geographic (EPSG:4326)
    for k in [1u16, 1, 0, 2, 1024, 0, 1, 2, 2048, 0, 1, 4326] {
        tif.extend_from_slice(&k.to_le_bytes());
    }

    let ifd_off = tif.len() as u32;
    tif[first_ifd_patch_pos..first_ifd_patch_pos + 4].copy_from_slice(&ifd_off.to_le_bytes());

    let entries: Vec<(u16, u16, u32, u32)> = vec![
        (256, 4, 1, 512),         // ImageWidth
        (257, 4, 1, 512),         // ImageLength
        (258, 3, 1, 32),          // BitsPerSample
        (259, 3, 1, 1),           // Compression = None
        (262, 3, 1, 1),           // Photometric = MinIsBlack
        (277, 3, 1, 1),           // SamplesPerPixel
        (284, 3, 1, 1),           // PlanarConfiguration = Chunky
        (322, 4, 1, 256),         // TileWidth
        (323, 4, 1, 256),         // TileLength
        (324, 4, 4, offsets_pos), // TileOffsets
        (325, 4, 4, counts_pos),  // TileByteCounts
        (339, 3, 1, 3),           // SampleFormat = IEEEFP
        (33550, 12, 3, scale_pos),
        (33922, 12, 6, tie_pos),
        (34735, 3, 12, gkd_pos),
    ];
    tif.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    for (tag, ty, count, val) in entries {
        tif.extend_from_slice(&tag.to_le_bytes());
        tif.extend_from_slice(&ty.to_le_bytes());
        tif.extend_from_slice(&count.to_le_bytes());
        tif.extend_from_slice(&val.to_le_bytes());
    }
    tif.extend_from_slice(&0u32.to_le_bytes()); // next_ifd = 0

    let tif_arc = Arc::new(tif);
    let req_count = Arc::new(AtomicUsize::new(0));
    let bytes_sent = Arc::new(AtomicUsize::new(0));

    let mock_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let mock_addr = mock_listener.local_addr().unwrap();
    let tif_srv = tif_arc.clone();
    let rc_srv = req_count.clone();
    let bs_srv = bytes_sent.clone();
    let mock_task = tokio::spawn(async move {
        loop {
            let Ok((mut sock, _)) = mock_listener.accept().await else {
                break;
            };
            let data = tif_srv.clone();
            let rc = rc_srv.clone();
            let bs = bs_srv.clone();
            tokio::spawn(async move {
                let mut buf = [0u8; 2048];
                let n = sock.read(&mut buf).await.unwrap_or(0);
                if n == 0 {
                    return;
                }
                let req = String::from_utf8_lossy(&buf[..n]);
                let range_line = req
                    .lines()
                    .find(|l| l.to_ascii_lowercase().starts_with("range:"))
                    .unwrap_or("");
                let total = data.len();
                let (s, e) = if let Some(spec) = range_line.split("bytes=").nth(1) {
                    let mut parts = spec.trim().split('-');
                    let s: usize = parts.next().unwrap_or("0").parse().unwrap_or(0);
                    let e: usize = parts
                        .next()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(total - 1)
                        .min(total - 1);
                    (s, e)
                } else {
                    (0, total - 1)
                };
                let slice = &data[s..=e];
                rc.fetch_add(1, Ordering::Relaxed);
                bs.fetch_add(slice.len(), Ordering::Relaxed);
                let hdr = format!(
                    "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes {s}-{e}/{total}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    slice.len()
                );
                let _ = sock.write_all(hdr.as_bytes()).await;
                let _ = sock.write_all(slice).await;
                let _ = sock.shutdown().await;
            });
        }
    });

    let engine = Arc::new(TileEngine::new(16 * 1024 * 1024));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let srv_task = tokio::spawn(async move {
        let _ = serve_listener(engine, listener).await;
    });
    let base = format!("http://{addr}");
    let client = reqwest::Client::new();
    let url = format!("http://{mock_addr}/noncog.tif");

    // V4: Opening non-COG with IFD > 2 MiB should jump directly to IFD (~160 KiB total, not > 2 MiB)
    let r = client
        .get(format!("{base}/cog/info?url={url}"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let open_reqs = req_count.load(Ordering::Relaxed);
    let open_bytes = bytes_sent.load(Ordering::Relaxed);
    assert!(
        open_bytes < 256 * 1024,
        "V4 failed: non-COG open fetched {open_bytes} bytes in {open_reqs} reqs (expected < 256 KiB)"
    );

    // V5: A 1-block 256x256 native window (top-left block: lon 10.0..10.256, lat 44.744..45.0)
    // must fetch ONLY 1 block (262,144 bytes in 1 range request), not 2x2 blocks!
    req_count.store(0, Ordering::Relaxed);
    bytes_sent.store(0, Ordering::Relaxed);
    let r = client
        .get(format!(
            "{base}/cog/bbox/10.0,44.744,10.256,45.0/256x256.npy?url={url}&coord_crs=EPSG:4326&dst_crs=EPSG:4326"
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let win_reqs = req_count.load(Ordering::Relaxed);
    let win_bytes = bytes_sent.load(Ordering::Relaxed);
    assert_eq!(win_reqs, 1, "V5 failed: expected 1 range request, got {win_reqs}");
    assert_eq!(
        win_bytes, block_bytes,
        "V5 failed: expected exactly 1 block ({block_bytes} bytes), got {win_bytes} bytes"
    );

    mock_task.abort();
    srv_task.abort();
}


#[tokio::test]
async fn sweep16_u3_ungeoreferenced_and_u2b_nonlinear_gcp_self_contained() {
    let tmp = std::env::temp_dir().join(format!("cogzilla_u3_u2b_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();

    // Helper: write a tiny stripped float32 TIFF with optional ModelTiepointTag GCPs (no scale).
    let write_tiff = |path: &std::path::Path, w: u32, h: u32, gcps: Option<&[[f64; 6]]>, pixels: &[f32]| {
        let mut buf = Vec::<u8>::new();
        buf.extend_from_slice(b"II");
        buf.extend_from_slice(&42u16.to_le_bytes());
        buf.extend_from_slice(&8u32.to_le_bytes());
        let has_gcp = gcps.is_some();
        let num_entries: u16 = if has_gcp { 12 } else { 10 };
        let data_start = (8 + 2 + (num_entries as usize) * 12 + 4) as u32;
        let tp_off = data_start;
        let tp_bytes = gcps.map_or(0, |g| (g.len() * 6 * 8) as u32);
        let gkd_off = tp_off + tp_bytes;
        let gkd_bytes = if has_gcp { 12 * 2 } else { 0 };
        let pix_off = gkd_off + gkd_bytes;
        let push = |b: &mut Vec<u8>, tag: u16, ty: u16, count: u32, val: u32| {
            b.extend_from_slice(&tag.to_le_bytes());
            b.extend_from_slice(&ty.to_le_bytes());
            b.extend_from_slice(&count.to_le_bytes());
            b.extend_from_slice(&val.to_le_bytes());
        };
        buf.extend_from_slice(&num_entries.to_le_bytes());
        push(&mut buf, 256, 4, 1, w);
        push(&mut buf, 257, 4, 1, h);
        push(&mut buf, 258, 3, 1, 32);
        push(&mut buf, 259, 3, 1, 1);
        push(&mut buf, 262, 3, 1, 1);
        push(&mut buf, 273, 4, 1, pix_off);
        push(&mut buf, 277, 3, 1, 1);
        push(&mut buf, 278, 4, 1, h);
        push(&mut buf, 279, 4, 1, w * h * 4);
        push(&mut buf, 339, 3, 1, 3);
        if let Some(g) = gcps {
            push(&mut buf, 33922, 12, (g.len() * 6) as u32, tp_off);
            push(&mut buf, 34735, 3, 12, gkd_off);
        }
        buf.extend_from_slice(&0u32.to_le_bytes());
        if let Some(g) = gcps {
            for row in g {
                for v in row {
                    buf.extend_from_slice(&v.to_le_bytes());
                }
            }
            for k in [1u16, 1, 0, 2, 1024, 0, 1, 2, 2048, 0, 1, 4326] {
                buf.extend_from_slice(&k.to_le_bytes());
            }
        }
        for &p in pixels {
            buf.extend_from_slice(&p.to_le_bytes());
        }
        std::fs::write(path, &buf).unwrap();
    };

    // 1. Ungeoreferenced 32x32 TIFF (U3)
    let nogeo_path = tmp.join("nogeo.tif");
    let nogeo_px: Vec<f32> = (0..(32 * 32)).map(|i| (i % 32) as f32 * 10.0).collect();
    write_tiff(&nogeo_path, 32, 32, None, &nogeo_px);

    // 2. Nonlinear quadratic 9-GCP 64x64 TIFF (U2b order-2 polynomial)
    let gcp_nl_path = tmp.join("gcp_nonlinear.tif");
    let mut nl_gcps = Vec::new();
    for &row in &[0.0f64, 32.0, 64.0] {
        for &col in &[0.0f64, 32.0, 64.0] {
            let u = col / 64.0;
            let v = row / 64.0;
            let x = -122.5 + 0.25 * u + 0.01 * u * v;
            let y = 37.9 - 0.25 * v - 0.01 * u * u;
            nl_gcps.push([col, row, 0.0, x, y, 0.0]);
        }
    }
    let mut nl_px = vec![0.0f32; 64 * 64];
    for r in 0..64 {
        for c in 0..64 {
            nl_px[r * 64 + c] = (r * 100 + c) as f32;
        }
    }
    write_tiff(&gcp_nl_path, 64, 64, Some(&nl_gcps), &nl_px);

    let engine = Arc::new(TileEngine::new(16 * 1024 * 1024));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let srv_task = tokio::spawn(async move {
        let _ = serve_listener(engine, listener).await;
    });
    let base = format!("http://{addr}");
    let client = reqwest::Client::new();
    let nogeo_url = nogeo_path.to_string_lossy().to_string();
    let nl_url = gcp_nl_path.to_string_lossy().to_string();

    // U3: /cog/info returns 200 with crs=null, bounds=null, width=32, height=32
    let r = client.get(format!("{base}/cog/info?url={nogeo_url}")).send().await.unwrap();
    assert_eq!(r.status(), 200);
    let info: serde_json::Value = serde_json::from_slice(&r.bytes().await.unwrap()).unwrap();
    assert!(info["crs"].is_null(), "expected crs: null, got {info}");
    assert!(info["bounds"].is_null(), "expected bounds: null, got {info}");
    assert_eq!(info["width"], 32);
    assert_eq!(info["height"], 32);

    // U3: /cog/preview and GET /cog/statistics succeed in pixel space
    let r = client.get(format!("{base}/cog/preview.png?url={nogeo_url}&rescale=0,310")).send().await.unwrap();
    assert_eq!(r.status(), 200);
    let r = client.get(format!("{base}/cog/preview/16x16.png?url={nogeo_url}&rescale=0,310")).send().await.unwrap();
    assert_eq!(r.status(), 200);
    let r = client.get(format!("{base}/cog/statistics?url={nogeo_url}")).send().await.unwrap();
    assert_eq!(r.status(), 200);
    let stats: serde_json::Value = serde_json::from_slice(&r.bytes().await.unwrap()).unwrap();
    assert!((stats["b1"]["mean"].as_f64().unwrap() - 155.0).abs() < 1e-4);

    // U3: Geographic routes return 400 Bad Request
    for path in [
        format!("/cog/bounds?url={nogeo_url}"),
        format!("/cog/point/0,0?url={nogeo_url}"),
        format!("/cog/tiles/WebMercatorQuad/0/0/0.png?url={nogeo_url}"),
        format!("/cog/bbox/0,0,1,1/16x16.png?url={nogeo_url}"),
        format!("/cog/WebMercatorQuad/tilejson.json?url={nogeo_url}"),
    ] {
        let r = client.get(format!("{base}{path}")).send().await.unwrap();
        assert_eq!(r.status(), 400, "U3 expected 400 for {path}");
    }

    // U2(b): Nonlinear 9-GCP TIFF succeeds on /cog/info (with gcp_residual_px) and /cog/point
    let r = client.get(format!("{base}/cog/info?url={nl_url}")).send().await.unwrap();
    assert_eq!(r.status(), 200);
    let nl_info: serde_json::Value = serde_json::from_slice(&r.bytes().await.unwrap()).unwrap();
    let res = nl_info["gcp_residual_px"].as_f64().expect("gcp_residual_px present");
    assert!(res < 0.05, "expected order-2 poly residual < 0.05 px, got {res}");

    // Query center pixel (col=32.5, row=32.5) mapped through the exact quadratic formula:
    let (u, v) = (32.5 / 64.0, 32.5 / 64.0);
    let qx = -122.5 + 0.25 * u + 0.01 * u * v;
    let qy = 37.9 - 0.25 * v - 0.01 * u * u;
    let r = client.get(format!("{base}/cog/point/{qx},{qy}?url={nl_url}")).send().await.unwrap();
    assert_eq!(r.status(), 200);
    let pt: serde_json::Value = serde_json::from_slice(&r.bytes().await.unwrap()).unwrap();
    assert_eq!(pt["values"][0].as_f64().unwrap(), (32 * 100 + 32) as f64);

    // #550: TileJSON bounds on a nonlinear 9-GCP TIFF must match /cog/info's polynomial wgs84_bounds.
    let r = client
        .get(format!("{base}/cog/WebMercatorQuad/tilejson.json?url={nl_url}"))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let nl_tj: serde_json::Value = serde_json::from_slice(&r.bytes().await.unwrap()).unwrap();
    assert_eq!(
        nl_tj["bounds"], nl_info["wgs84_bounds"],
        "TileJSON bounds must match /cog/info polynomial wgs84_bounds on nonlinear 9-GCP TIFF"
    );

    srv_task.abort();
    let _ = std::fs::remove_dir_all(&tmp);
}


#[tokio::test]
async fn sweep18_w1_source_freshness_and_w2_referenced_asset_errors() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let tmp = std::env::temp_dir().join(format!("cogzilla_w1_w2_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();

    // Helper: build a stripped EPSG:4326 float32 GeoTIFF covering [-122.5..-122.244, 37.644..37.9]
    let build_geo_tiff = |w: u32, h: u32, fill_val: f32| -> Vec<u8> {
        let mut buf = Vec::<u8>::new();
        buf.extend_from_slice(b"II");
        buf.extend_from_slice(&42u16.to_le_bytes());
        buf.extend_from_slice(&8u32.to_le_bytes());
        let num_entries: u16 = 13;
        let data_start = (8 + 2 + (num_entries as usize) * 12 + 4) as u32;
        let scale_off = data_start;
        let tp_off = scale_off + 24;
        let gkd_off = tp_off + 48;
        let pix_off = gkd_off + 24;
        let push = |b: &mut Vec<u8>, tag: u16, ty: u16, count: u32, val: u32| {
            b.extend_from_slice(&tag.to_le_bytes());
            b.extend_from_slice(&ty.to_le_bytes());
            b.extend_from_slice(&count.to_le_bytes());
            b.extend_from_slice(&val.to_le_bytes());
        };
        buf.extend_from_slice(&num_entries.to_le_bytes());
        push(&mut buf, 256, 4, 1, w);
        push(&mut buf, 257, 4, 1, h);
        push(&mut buf, 258, 3, 1, 32);
        push(&mut buf, 259, 3, 1, 1);
        push(&mut buf, 262, 3, 1, 1);
        push(&mut buf, 273, 4, 1, pix_off);
        push(&mut buf, 277, 3, 1, 1);
        push(&mut buf, 278, 4, 1, h);
        push(&mut buf, 279, 4, 1, w * h * 4);
        push(&mut buf, 339, 3, 1, 3);
        push(&mut buf, 33550, 12, 3, scale_off);
        push(&mut buf, 33922, 12, 6, tp_off);
        push(&mut buf, 34735, 3, 12, gkd_off);
        buf.extend_from_slice(&0u32.to_le_bytes());
        let px_deg = 0.256 / (w as f64);
        for v in [px_deg, px_deg, 0.0f64] {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        for v in [0.0f64, 0.0, 0.0, -122.5, 37.9, 0.0] {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        for k in [1u16, 1, 0, 2, 1024, 0, 1, 2, 2048, 0, 1, 4326] {
            buf.extend_from_slice(&k.to_le_bytes());
        }
        let vb = fill_val.to_le_bytes();
        for _ in 0..(w * h) {
            buf.extend_from_slice(&vb);
        }
        buf
    };

    // Mutable remote HTTP origin supporting Range, ETag, If-Match (412), and If-None-Match (304)
    let remote_state = Arc::new(std::sync::Mutex::new((
        "\"v1\"".to_string(),
        build_geo_tiff(64, 64, 111.0),
    )));
    let mock_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let mock_addr = mock_listener.local_addr().unwrap();
    let rs_srv = remote_state.clone();
    let mock_task = tokio::spawn(async move {
        loop {
            let Ok((mut sock, _)) = mock_listener.accept().await else {
                break;
            };
            let rs = rs_srv.clone();
            tokio::spawn(async move {
                let mut buf = [0u8; 4096];
                let n = sock.read(&mut buf).await.unwrap_or(0);
                if n == 0 {
                    return;
                }
                let req = String::from_utf8_lossy(&buf[..n]);
                let mut if_match = None;
                let mut if_none_match = None;
                let mut range_spec = None;
                for line in req.lines().skip(1) {
                    if let Some((k, v)) = line.split_once(':') {
                        let kl = k.trim().to_ascii_lowercase();
                        let vt = v.trim().to_string();
                        if kl == "if-match" {
                            if_match = Some(vt);
                        } else if kl == "if-none-match" {
                            if_none_match = Some(vt);
                        } else if kl == "range" {
                            range_spec = Some(vt);
                        }
                    }
                }
                let (etag, data) = rs.lock().unwrap().clone();
                if let Some(im) = if_match
                    && im != etag
                {
                    let hdr = format!(
                        "HTTP/1.1 412 Precondition Failed\r\nETag: {etag}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                    );
                    let _ = sock.write_all(hdr.as_bytes()).await;
                    let _ = sock.shutdown().await;
                    return;
                }
                if let Some(inm) = if_none_match
                    && inm == etag
                {
                    let hdr = format!(
                        "HTTP/1.1 304 Not Modified\r\nETag: {etag}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                    );
                    let _ = sock.write_all(hdr.as_bytes()).await;
                    let _ = sock.shutdown().await;
                    return;
                }
                let total = data.len();
                let (s, e) = if let Some(rs) = range_spec
                    && let Some(spec) = rs.strip_prefix("bytes=")
                {
                    let mut parts = spec.split('-');
                    let s: usize = parts.next().unwrap_or("0").parse().unwrap_or(0);
                    let e: usize = parts
                        .next()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(total - 1)
                        .min(total - 1);
                    (s, e)
                } else {
                    (0, total - 1)
                };
                let slice = &data[s..=e];
                let hdr = format!(
                    "HTTP/1.1 206 Partial Content\r\nETag: {etag}\r\nContent-Range: bytes {s}-{e}/{total}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    slice.len()
                );
                let _ = sock.write_all(hdr.as_bytes()).await;
                let _ = sock.write_all(slice).await;
                let _ = sock.shutdown().await;
            });
        }
    });

    // Use a store with remote_cache_ttl = 0 so we also test immediate conditional revalidation
    let mut store = cogzilla_io::CogStore::new(16 * 1024 * 1024);
    store.remote_cache_ttl = std::time::Duration::ZERO;
    let engine = Arc::new(TileEngine::with_store_and_tile_cache(Arc::new(store), 4 * 1024 * 1024));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let srv_task = tokio::spawn(async move {
        let _ = serve_listener(engine, listener).await;
    });
    let base = format!("http://{addr}");
    let client = reqwest::Client::new();

    // --- W1 Local: overwrite (same size), overwrite (larger size), delete ---
    let local_path = tmp.join("mutable.tif");
    let local_url = local_path.to_string_lossy().to_string();
    std::fs::write(&local_path, build_geo_tiff(64, 64, 111.0)).unwrap();

    let r = client.get(format!("{base}/cog/point/-122.4,37.8?url={local_url}")).send().await.unwrap();
    assert_eq!(r.status(), 200);
    let j: serde_json::Value = serde_json::from_slice(&r.bytes().await.unwrap()).unwrap();
    assert_eq!(j["values"][0].as_f64().unwrap(), 111.0);

    // Overwrite same size via atomic rename (changes inode + mtime even within same second)
    // Also exercise encoded TileCache hit invalidation on /cog/tiles!
    let tile_url = format!("{base}/cog/tiles/WebMercatorQuad/12/655/1583.npy?url={local_url}");
    let t1 = client.get(&tile_url).send().await.unwrap();
    assert_eq!(t1.status(), 200);
    let t1_bytes = t1.bytes().await.unwrap();
    let tmp_swap = tmp.join("mutable_v2.tif");
    std::fs::write(&tmp_swap, build_geo_tiff(64, 64, 222.0)).unwrap();
    std::fs::rename(&tmp_swap, &local_path).unwrap();
    let t2 = client.get(&tile_url).send().await.unwrap();
    assert_eq!(t2.status(), 200);
    assert_ne!(t2.bytes().await.unwrap(), t1_bytes, "W1 TileCache hit must invalidate after overwrite");
    let r = client.get(format!("{base}/cog/point/-122.4,37.8?url={local_url}")).send().await.unwrap();
    assert_eq!(r.status(), 200);
    let j: serde_json::Value = serde_json::from_slice(&r.bytes().await.unwrap()).unwrap();
    assert_eq!(j["values"][0].as_f64().unwrap(), 222.0, "W1 local overwrite failed");

    // Overwrite with larger 128x128 file (changes size)
    std::fs::write(&local_path, build_geo_tiff(128, 128, 333.0)).unwrap();
    let r = client.get(format!("{base}/cog/point/-122.4,37.8?url={local_url}")).send().await.unwrap();
    assert_eq!(r.status(), 200);
    let j: serde_json::Value = serde_json::from_slice(&r.bytes().await.unwrap()).unwrap();
    assert_eq!(j["values"][0].as_f64().unwrap(), 333.0, "W1 local resize failed");

    // Delete file -> 404
    std::fs::remove_file(&local_path).unwrap();
    let r = client.get(format!("{base}/cog/point/-122.4,37.8?url={local_url}")).send().await.unwrap();
    assert_eq!(r.status(), 404, "W1 deleted local file must return 404");

    // --- W1 Remote: ETag replacement detected and re-read ---
    let remote_url = format!("http://{mock_addr}/live.tif");
    let r = client.get(format!("{base}/cog/point/-122.4,37.8?url={remote_url}")).send().await.unwrap();
    assert_eq!(r.status(), 200);
    let j: serde_json::Value = serde_json::from_slice(&r.bytes().await.unwrap()).unwrap();
    assert_eq!(j["values"][0].as_f64().unwrap(), 111.0);

    // Replace remote object with new ETag "v2" and value 222.0
    *remote_state.lock().unwrap() = ("\"v2\"".to_string(), build_geo_tiff(64, 64, 222.0));
    let r = client.get(format!("{base}/cog/point/-122.4,37.8?url={remote_url}")).send().await.unwrap();
    assert_eq!(r.status(), 200);
    let j: serde_json::Value = serde_json::from_slice(&r.bytes().await.unwrap()).unwrap();
    assert_eq!(j["values"][0].as_f64().unwrap(), 222.0, "W1 remote ETag replacement failed");

    // --- W2: Referenced STAC & MosaicJSON asset failures -> 502; partial mosaic -> X-Failed-Assets ---
    let gone_path = tmp.join("gone_asset.tif");
    let valid_path = tmp.join("valid_asset.tif");
    std::fs::write(&valid_path, build_geo_tiff(64, 64, 42.0)).unwrap();

    // STAC item referencing missing asset -> 502 naming gone_asset.tif
    let stac_path = tmp.join("w2_item.json");
    std::fs::write(
        &stac_path,
        serde_json::json!({
            "type": "Feature",
            "stac_version": "1.0.0",
            "id": "w2-stac",
            "bbox": [-122.5, 37.644, -122.244, 37.9],
            "geometry": null,
            "properties": {},
            "assets": {
                "red": { "href": gone_path.to_string_lossy(), "type": "image/tiff; application=geotiff" }
            }
        })
        .to_string(),
    )
    .unwrap();
    let r = client
        .get(format!("{base}/stac/info?url={}&assets=red", stac_path.to_string_lossy()))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 502, "W2 missing STAC asset must return 502");
    let body = r.text().await.unwrap();
    assert!(body.contains("gone_asset.tif"), "W2 502 body should name asset: {body}");

    // MosaicJSON where every asset is missing -> 502 naming gone_asset.tif
    let mj_all_gone = tmp.join("mj_all_gone.json");
    // Quadkey for WebMercatorQuad z=12, x=655, y=1583 -> "023010203333"
    let qk = "023010203333";
    std::fs::write(
        &mj_all_gone,
        serde_json::json!({
            "mosaicjson": "0.0.2",
            "minzoom": 10,
            "maxzoom": 14,
            "quadkey_zoom": 12,
            "bounds": [-122.5, 37.644, -122.244, 37.9],
            "tiles": { qk: [gone_path.to_string_lossy()] }
        })
        .to_string(),
    )
    .unwrap();
    let r = client
        .get(format!("{base}/mosaicjson/tiles/WebMercatorQuad/12/655/1583.png?url={}", mj_all_gone.to_string_lossy()))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 502, "W2 all-missing mosaic assets must return 502");
    let body = r.text().await.unwrap();
    assert!(body.contains("gone_asset.tif"), "W2 mosaic 502 body should name asset: {body}");

    // MosaicJSON /point where every covering asset is missing -> 502 (not 404)
    // (-122.4, 37.75) lies inside WebMercatorQuad z=12, x=655, y=1583 (quadkey 023010203333)
    let r = client
        .get(format!("{base}/mosaicjson/point/-122.4,37.75?url={}", mj_all_gone.to_string_lossy()))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 502, "W2 all-missing mosaicjson/point must return 502");

    // MosaicJSON with [gone_asset.tif, valid_asset.tif] -> 200 with X-Failed-Assets header!
    let mj_partial = tmp.join("mj_partial.json");
    std::fs::write(
        &mj_partial,
        serde_json::json!({
            "mosaicjson": "0.0.2",
            "minzoom": 10,
            "maxzoom": 14,
            "quadkey_zoom": 12,
            "bounds": [-122.5, 37.644, -122.244, 37.9],
            "tiles": { qk: [gone_path.to_string_lossy(), valid_path.to_string_lossy()] }
        })
        .to_string(),
    )
    .unwrap();
    let r = client
        .get(format!("{base}/mosaicjson/tiles/WebMercatorQuad/12/655/1583.png?url={}", mj_partial.to_string_lossy()))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200, "W2 partial mosaic should render 200 from remaining asset");
    let failed_hdr = r
        .headers()
        .get("x-failed-assets")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(
        failed_hdr.contains("gone_asset.tif"),
        "W2 partial mosaic must report failed asset in X-Failed-Assets, got: '{failed_hdr}'"
    );

    mock_task.abort();
    srv_task.abort();
    let _ = std::fs::remove_dir_all(&tmp);
}

#[tokio::test]
async fn sweep23_cc1_null_geometry_and_indexed_fc_errors() {
    let tmp = std::env::temp_dir().join(format!(
        "cogzilla_sweep23_cc1_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();

    // Synthetic 64x64 EPSG:32610 GeoTIFF at (500_000, 4_200_000), 30m px, filled with 42.0
    let cog_path = tmp.join("scene.tif");
    write_synthetic_f32_projected_geotiff(
        &cog_path,
        64,
        64,
        32610,
        500_000.0,
        4_200_000.0,
        30.0,
        &vec![42.0f32; 64 * 64],
    );
    let cog_url = cog_path.to_string_lossy().to_string();

    let stac_path = tmp.join("item.json");
    std::fs::write(
        &stac_path,
        serde_json::json!({
            "type": "Feature",
            "stac_version": "1.0.0",
            "id": "cc1-item",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[-123.5, 37.5], [-122.5, 37.5], [-122.5, 38.5], [-123.5, 38.5], [-123.5, 37.5]]]
            },
            "bbox": [-123.5, 37.5, -122.5, 38.5],
            "properties": {"datetime": "2026-01-01T00:00:00Z"},
            "assets": {
                "data": {"href": cog_url, "type": "image/tiff; application=geotiff"}
            }
        })
        .to_string(),
    )
    .unwrap();
    let stac_url = stac_path.to_string_lossy().to_string();

    let mj_path = tmp.join("mosaic.json");
    std::fs::write(
        &mj_path,
        serde_json::json!({
            "mosaicjson": "0.0.3",
            "minzoom": 8,
            "maxzoom": 12,
            "quadkey_zoom": 8,
            "bounds": [-124.0, 37.0, -122.0, 39.0],
            "tiles": { "02301020": [cog_url] }
        })
        .to_string(),
    )
    .unwrap();
    let mj_url = mj_path.to_string_lossy().to_string();

    let engine = Arc::new(TileEngine::new(32 * 1024 * 1024));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let srv_task = tokio::spawn(serve_listener(engine.clone(), listener));
    let base = format!("http://{addr}");
    let client = reqwest::Client::new();

    let poly_a = serde_json::json!({
        "type": "Polygon",
        "coordinates": [[[500_300.0, 4_199_100.0], [500_900.0, 4_199_100.0], [500_900.0, 4_199_700.0], [500_300.0, 4_199_700.0], [500_300.0, 4_199_100.0]]]
    });
    let poly_c = serde_json::json!({
        "type": "Polygon",
        "coordinates": [[[501_000.0, 4_198_500.0], [501_600.0, 4_198_500.0], [501_600.0, 4_199_000.0], [501_000.0, 4_199_000.0], [501_000.0, 4_198_500.0]]]
    });
    let fc_with_null = serde_json::json!({
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "id": "f0", "properties": {"name": "first"}, "geometry": poly_a},
            {"type": "Feature", "id": "f1-null", "properties": {"name": "middle"}, "geometry": null},
            {"type": "Feature", "id": "f2", "properties": {"name": "last"}, "geometry": poly_c}
        ]
    });
    let fc_malformed = serde_json::json!({
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "id": "ok-0", "properties": {}, "geometry": poly_a},
            {
                "type": "Feature",
                "id": "bad-parcel-42",
                "properties": {},
                "geometry": {"type": "Polygon", "coordinates": [[[500_300.0, 4_199_100.0], [500_900.0, 4_199_100.0]]]}
            },
            {"type": "Feature", "id": "ok-2", "properties": {}, "geometry": poly_c}
        ]
    });

    let endpoints = [
        ("cog", format!("{base}/cog/statistics?url={cog_url}&coord_crs=EPSG:32610"), "b1"),
        (
            "stac",
            format!("{base}/stac/statistics?url={stac_url}&assets=data&coord_crs=EPSG:32610"),
            "b1",
        ),
        (
            "mosaicjson",
            format!("{base}/mosaicjson/statistics?url={mj_url}&coord_crs=EPSG:32610"),
            "b1",
        ),
    ];

    for (label, url, band_key) in &endpoints {
        // 1. 3-feature collection with null geometry in middle returns 3 features in order
        let resp = client
            .post(url)
            .header("content-type", "application/json")
            .body(fc_with_null.to_string())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200, "{label}: null-geom FC must return 200");
        let body: serde_json::Value = serde_json::from_str(&resp.text().await.unwrap()).unwrap();
        let feats = body["features"].as_array().unwrap();
        assert_eq!(feats.len(), 3, "{label}: expected 3 features");
        assert_eq!(feats[0]["id"], "f0");
        assert_eq!(feats[1]["id"], "f1-null");
        assert_eq!(feats[2]["id"], "f2");
        assert!(
            feats[0]["properties"]["statistics"][band_key]["count"]
                .as_f64()
                .unwrap()
                > 0.0,
            "{label}: feature 0 must have positive count"
        );
        let mid_stats = &feats[1]["properties"]["statistics"][band_key];
        assert_eq!(
            mid_stats["count"].as_f64().unwrap(),
            0.0,
            "{label}: null-geometry feature must have count 0, got {mid_stats}"
        );
        assert!(
            mid_stats["mean"].is_null(),
            "{label}: null-geometry feature must have null mean, got {mid_stats}"
        );
        assert!(
            feats[2]["properties"]["statistics"][band_key]["count"]
                .as_f64()
                .unwrap()
                > 0.0,
            "{label}: feature 2 must have positive count"
        );

        // 2. Malformed second feature returns 400 naming features[1] and its id
        let bad = client
            .post(url)
            .header("content-type", "application/json")
            .body(fc_malformed.to_string())
            .send()
            .await
            .unwrap();
        assert_eq!(bad.status(), 400, "{label}: malformed feature must return 400");
        let bad_text = bad.text().await.unwrap();
        assert!(
            bad_text.contains("features[1]") && bad_text.contains("bad-parcel-42"),
            "{label}: 400 detail must name features[1] and id, got: {bad_text}"
        );
    }

    // 3. STAC DEM algorithm + resized expression with null-geometry FC must succeed (200)
    for extra in ["&algorithm=hillshade", "&expression=data*2&width=32"] {
        let r = client
            .post(format!("{base}/stac/statistics?url={stac_url}&assets=data&coord_crs=EPSG:32610{extra}"))
            .header("content-type", "application/json")
            .body(fc_with_null.to_string())
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 200, "STAC {extra} with null-geom FC must return 200");
        let b: serde_json::Value = serde_json::from_str(&r.text().await.unwrap()).unwrap();
        assert_eq!(b["features"][1]["properties"]["statistics"]["b1"]["count"].as_f64().unwrap(), 0.0);
    }

    // 4. MosaicJSON null-geometry with invalid bidx=99 must return 400 (not swallowed 200)
    let null_feat = serde_json::json!({"type": "Feature", "properties": {}, "geometry": null});
    let r_bad_bidx = client
        .post(format!("{base}/mosaicjson/statistics?url={mj_url}&bidx=99"))
        .header("content-type", "application/json")
        .body(null_feat.to_string())
        .send()
        .await
        .unwrap();
    assert_eq!(r_bad_bidx.status(), 400, "mosaicjson null-geom with bidx=99 must return 400");

    srv_task.abort();
    let _ = std::fs::remove_dir_all(&tmp);
}

#[tokio::test]
async fn audit53_color_formula_with_custom_colormap_and_where_nan_zonal() {
    let tmp = std::env::temp_dir().join(format!("cogzilla_audit53_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&tmp);

    // Two 64x64 f32 GeoTIFFs in EPSG:32610 (10 m pixels, origin (500000, 4200640)):
    // a.tif = 10 + col, b.tif = NaN everywhere.
    let w = 64usize;
    let h = 64usize;
    let mut b1 = vec![0.0f32; w * h];
    for (i, px) in b1.iter_mut().enumerate() {
        *px = 10.0 + (i % 64) as f32;
    }
    let b2 = vec![f32::NAN; w * h];
    let a_path = tmp.join("a.tif");
    let b_path = tmp.join("b.tif");
    write_synthetic_f32_projected_geotiff(
        &a_path,
        w as u32,
        h as u32,
        32610,
        500_000.0,
        4_200_640.0,
        10.0,
        &b1,
    );
    write_synthetic_f32_projected_geotiff(
        &b_path,
        w as u32,
        h as u32,
        32610,
        500_000.0,
        4_200_640.0,
        10.0,
        &b2,
    );
    let stac_path = tmp.join("item.json");
    let stac_json = serde_json::json!({
        "type": "Feature",
        "stac_version": "1.0.0",
        "id": "audit53",
        "geometry": null,
        "bbox": [-123.0, 37.9, -122.9, 38.0],
        "properties": {},
        "assets": {
            "a": { "href": a_path.to_string_lossy() },
            "b": { "href": b_path.to_string_lossy() }
        }
    });
    std::fs::write(&stac_path, stac_json.to_string()).unwrap();

    let engine = Arc::new(TileEngine::new(16 * 1024 * 1024));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let srv_task = tokio::spawn(async move {
        let _ = serve_listener(engine, listener).await;
    });
    let base = format!("http://{addr}");
    let client = reqwest::Client::new();
    let a_url = a_path.to_string_lossy().to_string();
    let b_url = b_path.to_string_lossy().to_string();
    let stac_url = stac_path.to_string_lossy().to_string();

    let poly = serde_json::json!({
        "type": "Feature",
        "properties": {},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [500_100.0, 4_200_100.0],
                [500_200.0, 4_200_100.0],
                [500_200.0, 4_200_200.0],
                [500_100.0, 4_200_200.0],
                [500_100.0, 4_200_100.0]
            ]]
        }
    });

    // 1a. Single-COG POST /cog/statistics with where(1, 42, b1) when untaken b1 is NaN must
    // evaluate the taken branch (42) instead of aborting on NaN.
    let r_cog = client
        .post(format!(
            "{base}/cog/statistics?url={b_url}&coord_crs=EPSG:32610&max_size=0&expression=where(1,42,b1)"
        ))
        .header("content-type", "application/json")
        .body(poly.to_string())
        .send()
        .await
        .unwrap();
    assert_eq!(r_cog.status(), 200);
    let cog_stats: serde_json::Value = serde_json::from_str(&r_cog.text().await.unwrap()).unwrap();
    let cog_b1 = &cog_stats["properties"]["statistics"]["b1"];
    assert_eq!(
        cog_b1["valid_pixels"].as_f64().unwrap(),
        100.0,
        "where(1,42,b1) on NaN band must keep all 100 valid pixels: {cog_b1}"
    );
    assert!((cog_b1["mean"].as_f64().unwrap() - 42.0).abs() < 1e-4);

    // 1b. Multi-COG POST /stac/statistics with where(a>0, a, b) when untaken b is NaN must keep
    // all 100 valid pixels from a.
    let r_stac = client
        .post(format!(
            "{base}/stac/statistics?url={stac_url}&assets=a&assets=b&coord_crs=EPSG:32610&max_size=0&expression=where(a%3E0,a,b)"
        ))
        .header("content-type", "application/json")
        .body(poly.to_string())
        .send()
        .await
        .unwrap();
    assert_eq!(r_stac.status(), 200);
    let stac_stats: serde_json::Value = serde_json::from_str(&r_stac.text().await.unwrap()).unwrap();
    let stac_b1 = &stac_stats["properties"]["statistics"]["b1"];
    assert_eq!(
        stac_b1["valid_pixels"].as_f64().unwrap(),
        100.0,
        "STAC where(a>0,a,b) must keep all 100 valid pixels even when untaken b is NaN: {stac_b1}"
    );
    // cols 10..20 have values 20..29 -> mean = 24.5
    assert!((stac_b1["mean"].as_f64().unwrap() - 24.5).abs() < 1e-4);

    // 2. ?color_formula= with custom JSON ?colormap= must actually transform the output pixels.
    let cmap = "%5B%5B%5B0%2C100%5D%2C%5B120%2C120%2C120%2C255%5D%5D%5D";
    let r_plain = client
        .get(format!(
            "{base}/cog/bbox/500000,4200000,500640,4200640/32x32.png?url={a_url}&bidx=1&coord_crs=EPSG:32610&dst_crs=EPSG:32610&colormap={cmap}"
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(r_plain.status(), 200);
    let png_plain = r_plain.bytes().await.unwrap();

    let r_cf = client
        .get(format!(
            "{base}/cog/bbox/500000,4200000,500640,4200640/32x32.png?url={a_url}&bidx=1&coord_crs=EPSG:32610&dst_crs=EPSG:32610&colormap={cmap}&color_formula=Gamma+RGB+2.0"
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(r_cf.status(), 200);
    let png_cf = r_cf.bytes().await.unwrap();
    assert_ne!(
        png_plain, png_cf,
        "color_formula must not be silently dropped when custom JSON colormap is active"
    );

    srv_task.abort();
    let _ = std::fs::remove_dir_all(&tmp);
}

#[tokio::test]
async fn audit55_admin_clear_cache_and_mosaic_tile_cache_freshness() {
    let tmp = std::env::temp_dir().join(format!("cogzilla_audit55_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&tmp);

    // 64x64 f32 GeoTIFF in EPSG:3857 covering WebMercator tile 2/1/1 (x in [-10018754..0], y in [0..10018754]).
    let w = 64u32;
    let h = 64u32;
    let tile_span = 20_037_508.342789244 / 2.0; // z=2 tile width in EPSG:3857
    let px = tile_span / (w as f64);
    let tif_path = tmp.join("fresh_mosaic.tif");
    let v1 = vec![10.0f32; (w * h) as usize];
    write_synthetic_f32_projected_geotiff(
        &tif_path,
        w,
        h,
        3857,
        -tile_span,
        tile_span,
        px,
        &v1,
    );

    // Enable encoded tile_cache (16 MiB) to exercise warm tile_cache freshness on /mosaic/tiles!
    let engine = Arc::new(TileEngine::with_store_and_tile_cache(
        Arc::new(cogzilla_io::CogStore::new(16 * 1024 * 1024)),
        16 * 1024 * 1024,
    ));
    let engine_check = engine.clone();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let srv_task = tokio::spawn(async move {
        let _ = serve_listener(engine, listener).await;
    });
    let base = format!("http://{addr}");
    let client = reqwest::Client::new();
    let tif_url = tif_path.to_string_lossy().to_string();

    // Warm /mosaic/tiles cache
    let r1 = client
        .get(format!("{base}/mosaic/tiles/2/1/1.png?url={tif_url}&rescale=0,100"))
        .send()
        .await
        .unwrap();
    assert_eq!(r1.status(), 200);
    let b1 = r1.bytes().await.unwrap();

    // Overwrite local file with different values -> warm tile_cache MUST detect mtime/inode change!
    let tmp_swap = tmp.join("fresh_mosaic_v2.tif");
    let v2 = vec![90.0f32; (w * h) as usize];
    write_synthetic_f32_projected_geotiff(
        &tmp_swap,
        w,
        h,
        3857,
        -tile_span,
        tile_span,
        px,
        &v2,
    );
    std::fs::rename(&tmp_swap, &tif_path).unwrap();

    let r2 = client
        .get(format!("{base}/mosaic/tiles/2/1/1.png?url={tif_url}&rescale=0,100"))
        .send()
        .await
        .unwrap();
    assert_eq!(r2.status(), 200);
    let b2 = r2.bytes().await.unwrap();
    assert_ne!(
        b1, b2,
        "warm tile_cache on /mosaic/tiles must invalidate when backing file is replaced"
    );

    // POST /admin/clear-cache must bump invalidation_epoch (clearing datasets_by_uri)
    let ep_before = engine_check
        .store
        .invalidation_epoch
        .load(std::sync::atomic::Ordering::Acquire);
    let rc = client
        .post(format!("{base}/admin/clear-cache"))
        .send()
        .await
        .unwrap();
    assert_eq!(rc.status(), 200);
    assert!(
        engine_check
            .store
            .invalidation_epoch
            .load(std::sync::atomic::Ordering::Acquire)
            > ep_before,
        "POST /admin/clear-cache must clear CogStore open URIs and bump invalidation_epoch"
    );

    srv_task.abort();
    let _ = std::fs::remove_dir_all(&tmp);
}

#[tokio::test]
async fn sweep26_dd1_antimeridian_bbox_and_dd2_partial_polygon_valid_percent() {
    let tmp = std::env::temp_dir().join(format!("cogzilla_sweep26_{}", std::process::id()));
    let _ = std::fs::create_dir_all(&tmp);

    // --- DD1 fixture: EPSG:32701 (UTM 1S, central meridian 177W) spanning ~179E..176W, 16S..18S ---
    // At 179E, 16S -> UTM 1S easting ~ 72_000, northing ~ 8_230_000.
    // 500 m pixels x 1024 cols = 512 km (~4.8 deg) -> crosses 180 deg into 176W.
    let am_w = 1024u32;
    let am_h = 512u32;
    let mut am_px = Vec::with_capacity((am_w * am_h) as usize);
    for _r in 0..am_h {
        for c in 0..am_w {
            am_px.push((c + 1) as f32);
        }
    }
    let am_path = tmp.join("am32701.tif");
    write_synthetic_f32_projected_geotiff(
        &am_path,
        am_w,
        am_h,
        32701,
        70_000.0,
        8_230_000.0,
        500.0,
        &am_px,
    );

    // --- DD2 fixture: 64x64 EPSG:32610 (10 m pixels, x in 500000..500640, y in 4200000..4200640) ---
    let z_path = tmp.join("dd2_utm10n.tif");
    write_synthetic_f32_projected_geotiff(
        &z_path,
        64,
        64,
        32610,
        500_000.0,
        4_200_640.0,
        10.0,
        &vec![42.0f32; 64 * 64],
    );

    let engine = Arc::new(TileEngine::new(16 * 1024 * 1024));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let srv_task = tokio::spawn(async move {
        let _ = serve_listener(engine, listener).await;
    });
    let base = format!("http://{addr}");
    let client = reqwest::Client::new();
    let am_url = am_path.to_string_lossy().to_string();
    let z_url = z_path.to_string_lossy().to_string();

    // DD1a: /cog/bbox/{west>east}.npy must match {west,south,east+360,north}.npy byte-for-byte
    let r_wrap = client
        .get(format!("{base}/cog/bbox/179.5,-17.5,-178.5,-16.5.npy?url={am_url}"))
        .send()
        .await
        .unwrap();
    assert_eq!(r_wrap.status(), 200, "DD1: geographic west>east bbox must succeed");
    let b_wrap = r_wrap.bytes().await.unwrap();
    let r_unwrap = client
        .get(format!("{base}/cog/bbox/179.5,-17.5,181.5,-16.5.npy?url={am_url}"))
        .send()
        .await
        .unwrap();
    assert_eq!(r_unwrap.status(), 200);
    let b_unwrap = r_unwrap.bytes().await.unwrap();
    assert_eq!(b_wrap, b_unwrap, "DD1: wrapped and +360 unwrapped /cog/bbox must match identically");

    // DD1b: /cog/bbox/{west>east}/200x100.npy must match {west,south,east+360,north}/200x100.npy
    let r_wrap_wh = client
        .get(format!("{base}/cog/bbox/179.5,-17.5,-178.5,-16.5/200x100.npy?url={am_url}"))
        .send()
        .await
        .unwrap();
    assert_eq!(r_wrap_wh.status(), 200);
    let b_wrap_wh = r_wrap_wh.bytes().await.unwrap();
    let r_unwrap_wh = client
        .get(format!("{base}/cog/bbox/179.5,-17.5,181.5,-16.5/200x100.npy?url={am_url}"))
        .send()
        .await
        .unwrap();
    assert_eq!(r_unwrap_wh.status(), 200);
    let b_unwrap_wh = r_unwrap_wh.bytes().await.unwrap();
    assert_eq!(b_wrap_wh, b_unwrap_wh, "DD1: wrapped and +360 /cog/bbox/200x100 must match identically");

    // DD1c: /cog/map?bbox=179.5,-17.5,-178.5,-16.5 must match 181.5 form
    let r_map_wrap = client
        .get(format!(
            "{base}/cog/map?url={am_url}&bbox=179.5,-17.5,-178.5,-16.5&width=64&height=32&format=npy"
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(r_map_wrap.status(), 200);
    let b_map_wrap = r_map_wrap.bytes().await.unwrap();
    let r_map_unwrap = client
        .get(format!(
            "{base}/cog/map?url={am_url}&bbox=179.5,-17.5,181.5,-16.5&width=64&height=32&format=npy"
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(r_map_unwrap.status(), 200);
    assert_eq!(b_map_wrap, r_map_unwrap.bytes().await.unwrap());

    // DD1d: projected coord_crs with minx > maxx must return 400 with one clear message
    for path in [
        format!("{base}/cog/bbox/179.5,-17.5,-178.5,-16.5.npy?url={am_url}&coord_crs=EPSG:3857"),
        format!("{base}/cog/bbox/179.5,-17.5,-178.5,-16.5/200x100.npy?url={am_url}&coord_crs=EPSG:3857"),
    ] {
        let r_proj = client.get(&path).send().await.unwrap();
        assert_eq!(r_proj.status(), 400, "projected minx>maxx must return 400 for {path}");
        let msg = r_proj.text().await.unwrap();
        assert!(
            msg.contains("minx") && msg.contains("maxx"),
            "projected minx>maxx 400 must clearly name minx and maxx, got: {msg}"
        );
    }

    // DD2a: fully inside polygon stays at valid_percent == 100.0, masked_pixels == 0.0
    let poly_inside = serde_json::json!({
        "type": "Feature",
        "properties": {},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [500_100.0, 4_200_100.0],
                [500_300.0, 4_200_100.0],
                [500_300.0, 4_200_300.0],
                [500_100.0, 4_200_300.0],
                [500_100.0, 4_200_100.0]
            ]]
        }
    });
    let r_in = client
        .post(format!("{base}/cog/statistics?url={z_url}&coord_crs=EPSG:32610&max_size=0"))
        .header("content-type", "application/json")
        .body(poly_inside.to_string())
        .send()
        .await
        .unwrap();
    assert_eq!(r_in.status(), 200);
    let j_in: serde_json::Value = serde_json::from_str(&r_in.text().await.unwrap()).unwrap();
    let b_in = &j_in["properties"]["statistics"]["b1"];
    assert_eq!(b_in["valid_percent"].as_f64().unwrap(), 100.0);
    assert_eq!(b_in["masked_pixels"].as_f64().unwrap(), 0.0);

    // DD2b: half-inside, half-off-file polygon (40x20 = 800 px, 400 px inside file):
    // native path must report valid_percent == 50.0 and masked_pixels == 400.0 (matching width/height).
    let poly_half = serde_json::json!({
        "type": "Feature",
        "properties": {},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [500_440.0, 4_200_100.0],
                [500_840.0, 4_200_100.0],
                [500_840.0, 4_200_300.0],
                [500_440.0, 4_200_300.0],
                [500_440.0, 4_200_100.0]
            ]]
        }
    });
    let r_half = client
        .post(format!("{base}/cog/statistics?url={z_url}&coord_crs=EPSG:32610&max_size=0"))
        .header("content-type", "application/json")
        .body(poly_half.to_string())
        .send()
        .await
        .unwrap();
    assert_eq!(r_half.status(), 200);
    let j_half: serde_json::Value = serde_json::from_str(&r_half.text().await.unwrap()).unwrap();
    let b_half = &j_half["properties"]["statistics"]["b1"];
    assert!(
        (b_half["count"].as_f64().unwrap() - 400.0).abs() < 1e-4,
        "expected count 400, got {b_half}"
    );
    assert!(
        (b_half["valid_percent"].as_f64().unwrap() - 50.0).abs() < 1e-4,
        "DD2: expected valid_percent 50.0 for half-off-file polygon, got {b_half}"
    );
    assert!(
        (b_half["masked_pixels"].as_f64().unwrap() - 400.0).abs() < 1e-4,
        "DD2: expected masked_pixels 400.0 for half-off-file polygon, got {b_half}"
    );

    srv_task.abort();
    let _ = std::fs::remove_dir_all(&tmp);
}

fn write_synthetic_u16_2band_4326_geotiff(
    path: &std::path::Path,
    width: u32,
    height: u32,
    bounds: [f64; 4], // [west, south, east, north]
    band1_val: u16,
    band2_val: u16,
    hole: Option<&str>, // "left" | "top"
) {
    let mut pixels = Vec::<u16>::with_capacity((width * height * 2) as usize);
    for row in 0..height {
        for col in 0..width {
            let masked = match hole {
                Some("left") => col < width / 2,
                Some("top") => row < height / 2,
                _ => false,
            };
            if masked {
                pixels.push(0);
                pixels.push(0);
            } else {
                pixels.push(band1_val);
                pixels.push(band2_val);
            }
        }
    }
    let mut buf = Vec::<u8>::new();
    buf.extend_from_slice(b"II");
    buf.extend_from_slice(&42u16.to_le_bytes());
    buf.extend_from_slice(&8u32.to_le_bytes());

    let num_entries: u16 = 14;
    let ifd_size = 2 + (num_entries as usize) * 12 + 4;
    let data_start = 8 + ifd_size;

    let scale_off = data_start as u32;
    let tiepoint_off = scale_off + 24;
    let geokey_off = tiepoint_off + 48;
    let pixel_off = geokey_off + 24;
    let pixel_bytes = (pixels.len() * 2) as u32;

    let push_entry = |b: &mut Vec<u8>, tag: u16, ty: u16, count: u32, val: u32| {
        b.extend_from_slice(&tag.to_le_bytes());
        b.extend_from_slice(&ty.to_le_bytes());
        b.extend_from_slice(&count.to_le_bytes());
        b.extend_from_slice(&val.to_le_bytes());
    };

    buf.extend_from_slice(&num_entries.to_le_bytes());
    push_entry(&mut buf, 256, 4, 1, width); // ImageWidth
    push_entry(&mut buf, 257, 4, 1, height); // ImageLength
    // BitsPerSample = [16, 16] packed inline in 4 bytes
    push_entry(&mut buf, 258, 3, 2, 16u32 | (16u32 << 16));
    push_entry(&mut buf, 259, 3, 1, 1); // Compression = None
    push_entry(&mut buf, 262, 3, 1, 1); // Photometric = MinIsBlack
    push_entry(&mut buf, 273, 4, 1, pixel_off); // StripOffsets
    push_entry(&mut buf, 277, 3, 1, 2); // SamplesPerPixel = 2
    push_entry(&mut buf, 278, 4, 1, height); // RowsPerStrip
    push_entry(&mut buf, 279, 4, 1, pixel_bytes); // StripByteCounts
    push_entry(&mut buf, 284, 3, 1, 1); // PlanarConfiguration = Chunky
    push_entry(&mut buf, 33550, 12, 3, scale_off); // ModelPixelScaleTag
    push_entry(&mut buf, 33922, 12, 6, tiepoint_off); // ModelTiepointTag
    push_entry(&mut buf, 34735, 3, 12, geokey_off); // GeoKeyDirectoryTag
    // GDAL_NODATA = "0\0" (2 bytes inline in 4-byte value field)
    push_entry(&mut buf, 42113, 2, 2, u32::from_le_bytes([b'0', 0, 0, 0]));
    buf.extend_from_slice(&0u32.to_le_bytes()); // Next IFD = 0

    let sx = (bounds[2] - bounds[0]) / width as f64;
    let sy = (bounds[3] - bounds[1]) / height as f64;
    assert_eq!(buf.len(), scale_off as usize);
    for v in [sx, sy, 0.0f64] {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    assert_eq!(buf.len(), tiepoint_off as usize);
    for v in [0.0f64, 0.0, 0.0, bounds[0], bounds[3], 0.0] {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    assert_eq!(buf.len(), geokey_off as usize);
    // GTModelTypeGeoKey(1024)=2 (Geographic), GeographicTypeGeoKey(2048)=4326
    for v in [1u16, 1, 0, 2, 1024, 0, 1, 2, 2048, 0, 1, 4326] {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    assert_eq!(buf.len(), pixel_off as usize);
    for &p in &pixels {
        buf.extend_from_slice(&p.to_le_bytes());
    }
    std::fs::write(path, &buf).unwrap();
}

/// Parity Sweep #30 FF1:
/// - /mosaicjson/bbox honours all 9 pixel_selection methods and returns X-Assets
/// - GET & POST /mosaicjson/statistics mosaic all intersecting assets through Fold,
///   honour all 9 pixel_selection methods, and return X-Assets
/// - Unsupported routes refuse ?pixel_selection= with 400 (while bogus values stay 422)
#[tokio::test]
async fn sweep30_ff1_mosaic_bbox_statistics_pixel_selection_and_400_refusal() {
    let tmp = std::env::temp_dir().join(format!(
        "cogzilla_sweep30_ff1_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();

    // WebMercator tile 10/163/395 WGS84 bounds (quadkey "0230102033")
    let bounds = [-122.6953125, 37.71859032558816, -122.34375, 37.996162679728116];
    let a_path = tmp.join("a.tif");
    let b_path = tmp.join("b.tif");
    let c_path = tmp.join("c.tif");
    write_synthetic_u16_2band_4326_geotiff(&a_path, 64, 64, bounds, 10, 300, Some("left"));
    write_synthetic_u16_2band_4326_geotiff(&b_path, 64, 64, bounds, 20, 100, None);
    write_synthetic_u16_2band_4326_geotiff(&c_path, 64, 64, bounds, 30, 200, Some("top"));

    let mj_path = tmp.join("m.json");
    let mj = serde_json::json!({
        "mosaicjson": "0.0.3",
        "minzoom": 8,
        "maxzoom": 12,
        "quadkey_zoom": 10,
        "bounds": bounds,
        "center": [(bounds[0] + bounds[2]) / 2.0, (bounds[1] + bounds[3]) / 2.0, 10],
        "tiles": {
            "0230102033": [
                a_path.to_str().unwrap(),
                b_path.to_str().unwrap(),
                c_path.to_str().unwrap()
            ]
        }
    });
    std::fs::write(&mj_path, mj.to_string()).unwrap();

    let engine = Arc::new(TileEngine::new(16 * 1024 * 1024));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let srv_task = tokio::spawn(async move {
        let _ = serve_listener(engine, listener).await;
    });
    let base = format!("http://{addr}");
    let client = reqwest::Client::new();
    let mp = mj_path.to_string_lossy().to_string();
    let ap = a_path.to_string_lossy().to_string();
    let bbox_str = format!("{},{},{},{}", bounds[0], bounds[1], bounds[2], bounds[3]);

    // Expected [TL, TR, BL, BR] for band 1, plus (expected_bands, min, max, mean) for stats
    // Over 64x64 = 4096 pixels with equal 32x32 quadrants.
    let cases = [
        ("first",        [20.0f32, 10.0, 20.0, 10.0], 2usize, 10.0f64, 20.0f64, 15.0f64),
        ("highest",      [20.0, 20.0, 30.0, 30.0], 2, 20.0, 30.0, 25.0),
        ("lowest",       [20.0, 10.0, 20.0, 10.0], 2, 10.0, 20.0, 15.0),
        ("mean",         [20.0, 15.0, 25.0, 20.0], 2, 15.0, 25.0, 20.0),
        ("median",       [20.0, 15.0, 25.0, 20.0], 2, 15.0, 25.0, 20.0),
        ("stdev",        [0.0,  5.0,  5.0,  8.164966], 2, 0.0, 8.164966, 4.541241),
        ("count",        [1.0,  2.0,  2.0,  3.0],  1, 1.0,  3.0,  2.0),
        ("lastbandhigh", [20.0, 10.0, 30.0, 10.0], 1, 10.0, 30.0, 17.5),
        ("lastbandlow",  [20.0, 20.0, 20.0, 20.0], 1, 20.0, 20.0, 20.0),
    ];

    // 1. FF1(a): /mosaicjson/bbox/{bbox}/64x64.npy across all 9 methods + X-Assets
    for (ps, expected_q, expected_nb, _, _, _) in cases {
        let r = client
            .get(format!(
                "{base}/mosaicjson/bbox/{bbox_str}/64x64.npy?url={mp}&pixel_selection={ps}"
            ))
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 200, "bbox {ps} failed");
        let x_assets = r
            .headers()
            .get("x-assets")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        assert!(
            x_assets.contains("a.tif") && x_assets.contains("b.tif"),
            "bbox {ps} missing X-Assets: {x_assets:?}"
        );
        let body = r.bytes().await.unwrap();
        let vals = decode_npy_f32(&body);
        assert_eq!(
            vals.len(),
            (expected_nb + 1) * 64 * 64,
            "bbox {ps} unexpected npy plane count (data+mask)"
        );
        let b1 = &vals[..64 * 64];
        let got_q = [
            b1[16 * 64 + 16],
            b1[16 * 64 + 48],
            b1[48 * 64 + 16],
            b1[48 * 64 + 48],
        ];
        for (i, (&g, &e)) in got_q.iter().zip(expected_q.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-3,
                "bbox {ps} quadrant {i}: got {got_q:?}, expected {expected_q:?}"
            );
        }
    }

    // 2. FF1(b): GET /mosaicjson/statistics across all 9 methods + X-Assets
    for (ps, _, expected_nb, exp_min, exp_max, exp_mean) in cases {
        let r = client
            .get(format!(
                "{base}/mosaicjson/statistics?url={mp}&pixel_selection={ps}"
            ))
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 200, "GET statistics {ps} failed");
        let x_assets = r
            .headers()
            .get("x-assets")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        assert!(
            x_assets.contains("a.tif") && x_assets.contains("b.tif"),
            "GET statistics {ps} missing X-Assets: {x_assets:?}"
        );
        if ps != "first" {
            assert!(
                x_assets.contains("c.tif"),
                "GET statistics {ps} should include c.tif in X-Assets: {x_assets:?}"
            );
        }
        let j: serde_json::Value = serde_json::from_str(&r.text().await.unwrap()).unwrap();
        let obj = j.as_object().unwrap();
        assert_eq!(
            obj.len(),
            expected_nb,
            "GET statistics {ps}: expected {expected_nb} bands, got {j}"
        );
        let b1 = &obj["b1"];
        assert!(
            (b1["count"].as_f64().unwrap() - 4096.0).abs() < 1e-4,
            "GET statistics {ps}: expected full mosaic count 4096, got {b1}"
        );
        assert!(
            (b1["min"].as_f64().unwrap() - exp_min).abs() < 1e-3,
            "GET statistics {ps}: min got {}, expected {exp_min}",
            b1["min"]
        );
        assert!(
            (b1["max"].as_f64().unwrap() - exp_max).abs() < 1e-3,
            "GET statistics {ps}: max got {}, expected {exp_max}",
            b1["max"]
        );
        assert!(
            (b1["mean"].as_f64().unwrap() - exp_mean).abs() < 1e-3,
            "GET statistics {ps}: mean got {}, expected {exp_mean}",
            b1["mean"]
        );
    }

    // 3. FF1(b): POST /mosaicjson/statistics with polygon covering the mosaic
    let poly = serde_json::json!({
        "type": "Feature",
        "properties": {},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [bounds[0], bounds[1]],
                [bounds[2], bounds[1]],
                [bounds[2], bounds[3]],
                [bounds[0], bounds[3]],
                [bounds[0], bounds[1]]
            ]]
        }
    });
    for (ps, _, expected_nb, exp_min, exp_max, exp_mean) in cases {
        let r = client
            .post(format!(
                "{base}/mosaicjson/statistics?url={mp}&pixel_selection={ps}&area_weighted=false"
            ))
            .header("content-type", "application/json")
            .body(poly.to_string())
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 200, "POST statistics {ps} failed");
        let x_assets = r
            .headers()
            .get("x-assets")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        assert!(
            x_assets.contains("a.tif") && x_assets.contains("b.tif"),
            "POST statistics {ps} missing X-Assets: {x_assets:?}"
        );
        let j: serde_json::Value = serde_json::from_str(&r.text().await.unwrap()).unwrap();
        let stats = j["properties"]["statistics"].as_object().unwrap();
        assert_eq!(
            stats.len(),
            expected_nb,
            "POST statistics {ps}: expected {expected_nb} bands, got {j}"
        );
        let b1 = &stats["b1"];
        assert!(
            (b1["count"].as_f64().unwrap() - 4096.0).abs() < 1e-4,
            "POST statistics {ps}: expected full mosaic count 4096, got {b1}"
        );
        assert!(
            (b1["min"].as_f64().unwrap() - exp_min).abs() < 1e-3,
            "POST statistics {ps}: min got {}, expected {exp_min}",
            b1["min"]
        );
        assert!(
            (b1["max"].as_f64().unwrap() - exp_max).abs() < 1e-3,
            "POST statistics {ps}: max got {}, expected {exp_max}",
            b1["max"]
        );
        assert!(
            (b1["mean"].as_f64().unwrap() - exp_mean).abs() < 1e-3,
            "POST statistics {ps}: mean got {}, expected {exp_mean}",
            b1["mean"]
        );
    }

    // 4. Unsupported routes refuse ?pixel_selection= with 400 (bogus value stays 422)
    let cx = (bounds[0] + bounds[2]) / 2.0 + 0.01;
    let cy = (bounds[1] + bounds[3]) / 2.0 - 0.01;
    for unsupported in [
        format!("{base}/mosaicjson/point/{cx},{cy}?url={mp}&pixel_selection=highest"),
        format!("{base}/mosaicjson/info?url={mp}&pixel_selection=highest"),
        format!("{base}/cog/bbox/{bbox_str}/32x32.npy?url={ap}&pixel_selection=highest"),
        format!("{base}/cog/statistics?url={ap}&pixel_selection=highest"),
    ] {
        let r = client.get(&unsupported).send().await.unwrap();
        assert_eq!(
            r.status(),
            400,
            "expected 400 for unsupported pixel_selection on {unsupported}, got {}",
            r.status()
        );
    }
    let r_bogus = client
        .get(format!(
            "{base}/mosaicjson/bbox/{bbox_str}/32x32.npy?url={mp}&pixel_selection=bogus"
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(r_bogus.status(), 422, "bogus pixel_selection must stay 422");

    // 5. Audit #64 regressions:
    // (a) POST /mosaicjson/feature.png?pixel_selection=count must succeed (200) and mask outside-polygon corners
    let tri = serde_json::json!({
        "type": "Feature",
        "properties": {},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [bounds[0], bounds[1]],
                [bounds[2], bounds[1]],
                [bounds[0], bounds[3]],
                [bounds[0], bounds[1]]
            ]]
        }
    });
    let r_feat = client
        .post(format!(
            "{base}/mosaicjson/feature.npy?url={mp}&width=32&height=32&pixel_selection=count"
        ))
        .header("content-type", "application/json")
        .body(tri.to_string())
        .send()
        .await
        .unwrap();
    assert_eq!(r_feat.status(), 200, "POST /mosaicjson/feature.npy?pixel_selection=count must return 200");
    let feat_vals = decode_npy_f32(&r_feat.bytes().await.unwrap());
    // Plane 1 is the mask plane (0 = masked, 255 = valid); top-right corner (row 0, col 31) is outside the triangle
    let mask_plane = &feat_vals[32 * 32..2 * 32 * 32];
    assert_eq!(
        mask_plane[31], 0.0,
        "Selection::Count on POST /mosaicjson/feature must mask outside-polygon pixels"
    );
    assert_eq!(
        mask_plane[31 * 32], 255.0,
        "Selection::Count on POST /mosaicjson/feature must keep inside-polygon pixels"
    );

    // (b) GET /mosaicjson/viewer?pixel_selection=highest -> 200; /mosaicjson/tiles/.../assets?pixel_selection=highest -> 400
    let r_viewer = client
        .get(format!("{base}/mosaicjson/viewer?url={mp}&pixel_selection=highest"))
        .send()
        .await
        .unwrap();
    assert_eq!(r_viewer.status(), 200, "/mosaicjson/viewer forwards query and must accept pixel_selection");
    let r_tile_assets = client
        .get(format!("{base}/mosaicjson/tiles/WebMercatorQuad/10/163/395/assets?url={mp}&pixel_selection=highest"))
        .send()
        .await
        .unwrap();
    assert_eq!(r_tile_assets.status(), 400, "/mosaicjson/tiles/.../assets must refuse pixel_selection with 400");
    // Audit #72 #1 & #11: TMS-less /mosaicjson/tiles/{z}/{x}/{y}/assets -> 200; bogus TMS -> 422;
    // mixed-case / percent-encoded resampling accepted
    let r_assets_notms = client
        .get(format!("{base}/mosaicjson/tiles/10/163/395/assets?url={mp}"))
        .send()
        .await
        .unwrap();
    assert_eq!(r_assets_notms.status(), 200, "/mosaicjson/tiles/z/x/y/assets must return 200");
    let r_assets_badtms = client
        .get(format!("{base}/mosaicjson/tiles/BogusTMS/10/163/395/assets?url={mp}"))
        .send()
        .await
        .unwrap();
    assert_eq!(r_assets_badtms.status(), 422, "/mosaicjson/tiles/BogusTMS/z/x/y/assets must return 422");
    let r_resamp_case = client
        .get(format!("{base}/mosaicjson/tiles/WebMercatorQuad/10/163/395.npy?url={mp}&resampling=Bilinear"))
        .send()
        .await
        .unwrap();
    assert_eq!(r_resamp_case.status(), 200, "mixed-case resampling=Bilinear must return 200");

    // Audit #73: lastbandhigh without explicit bidx on 2-band WebMercator mosaic tile -> 200;
    // GET /mosaicjson/assets?coord_crs=... -> 200; GET / has Vary: Accept; GET /stats not public cached
    let r_lbh_nobidx = client
        .get(format!("{base}/mosaicjson/tiles/WebMercatorQuad/10/163/395.png?url={mp}&pixel_selection=lastbandhigh"))
        .send()
        .await
        .unwrap();
    assert_eq!(r_lbh_nobidx.status(), 200, "2-band visual lastbandhigh without bidx must return 200");
    let r_mj_assets_crs = client
        .get(format!("{base}/mosaicjson/assets?url={mp}&bbox={bbox_str}&coord_crs=EPSG:4326"))
        .send()
        .await
        .unwrap();
    assert_eq!(r_mj_assets_crs.status(), 200, "GET /mosaicjson/assets?coord_crs=... must return 200");
    let r_root = client.get(format!("{base}/")).send().await.unwrap();
    assert_eq!(r_root.status(), 200);
    let vary = r_root
        .headers()
        .get("vary")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(vary.to_ascii_lowercase().contains("accept"), "GET / must include Vary: Accept, got {vary:?}");
    let r_stats_ep = client.get(format!("{base}/stats")).send().await.unwrap();
    assert_eq!(r_stats_ep.status(), 200);
    assert!(
        r_stats_ep.headers().get("cache-control").is_none(),
        "GET /stats must not carry public Cache-Control"
    );

    // (c) POST /mosaicjson/statistics on Point and horizontal LineString with pixel_selection=count
    let pt = serde_json::json!({
        "type": "Feature",
        "properties": {},
        "geometry": { "type": "Point", "coordinates": [cx, cy] }
    });
    let r_pt = client
        .post(format!("{base}/mosaicjson/statistics?url={mp}&pixel_selection=count"))
        .header("content-type", "application/json")
        .body(pt.to_string())
        .send()
        .await
        .unwrap();
    assert_eq!(r_pt.status(), 200, "POST /mosaicjson/statistics on Point must succeed");
    let j_pt: serde_json::Value = serde_json::from_str(&r_pt.text().await.unwrap()).unwrap();
    assert!(
        j_pt["properties"]["statistics"]["b1"]["count"].as_f64().unwrap() >= 1.0,
        "Point stats under count must have valid_pixels >= 1, got {j_pt}"
    );
    let hline = serde_json::json!({
        "type": "Feature",
        "properties": {},
        "geometry": {
            "type": "LineString",
            "coordinates": [[bounds[0] + 0.05, cy], [bounds[2] - 0.05, cy]]
        }
    });
    let r_line = client
        .post(format!("{base}/mosaicjson/statistics?url={mp}&pixel_selection=count"))
        .header("content-type", "application/json")
        .body(hline.to_string())
        .send()
        .await
        .unwrap();
    assert_eq!(r_line.status(), 200, "POST /mosaicjson/statistics on horizontal LineString must succeed");
    let j_line: serde_json::Value = serde_json::from_str(&r_line.text().await.unwrap()).unwrap();
    assert!(
        j_line["properties"]["statistics"]["b1"]["count"].as_f64().unwrap() >= 1.0,
        "LineString stats under count must have valid_pixels >= 1, got {j_line}"
    );

    // (d) 2-band visual PNG under lastbandhigh succeeds (200), while 2-band visual PNG under highest returns 400
    let r_lbh_png = client
        .get(format!(
            "{base}/mosaicjson/tiles/WebMercatorQuad/10/163/395.png?url={mp}&bidx=1&bidx=2&pixel_selection=lastbandhigh&rescale=0,100"
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(r_lbh_png.status(), 200, "2-band visual PNG under lastbandhigh must return 200");
    let r_2b_bad = client
        .get(format!(
            "{base}/mosaicjson/bbox/{bbox_str}/32x32.png?url={mp}&bidx=1&bidx=2&pixel_selection=highest"
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(r_2b_bad.status(), 400, "2-band visual PNG under highest must return 400");

    srv_task.abort();
    let _ = std::fs::remove_dir_all(&tmp);
}

/// Parity Sweep #31 GG1:
/// - One-file mosaic POST /mosaicjson/statistics equals POST /cog/statistics to 1e-9
///   (count, mean, std, sum, valid_pixels, valid_percent, masked_pixels, area, category_areas).
/// - Same-grid and heterogeneous-grid multi-file mosaics keep exact coverage fractions,
///   ellipsoidal area, and polygon-relative valid_percent == 100.0.
/// - Heterogeneous-grid POST /stac/statistics with an expression also keeps exact coverage
///   fractions, ellipsoidal area, and polygon-relative valid_percent == 100.0.
#[tokio::test]
async fn sweep31_gg1_mosaic_and_stac_exact_zonal_statistics() {
    let tmp = std::env::temp_dir().join(format!(
        "cogzilla_sweep31_gg1_{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();

    // 1. One-file projected UTM 10N (EPSG:32610) COG with spatially varying values
    let utm_path = tmp.join("utm10n.tif");
    let mut utm_px = vec![0.0f32; 64 * 64];
    for r in 0..64 {
        for c in 0..64 {
            utm_px[r * 64 + c] = ((r * 3 + c * 5) % 17 + 1) as f32;
        }
    }
    write_synthetic_f32_projected_geotiff(
        &utm_path,
        64,
        64,
        32610,
        500_000.0,
        4_200_640.0,
        10.0,
        &utm_px,
    );
    // Tile covering ~(500_000, 4_200_000) in EPSG:32610 -> ~(-123.0, 37.95)
    // Querying quadkey at zoom 0 ("") matches any bbox in MosaicJSON.
    let one_mj_path = tmp.join("one.json");
    let one_mj = serde_json::json!({
        "mosaicjson": "0.0.3",
        "minzoom": 0,
        "maxzoom": 14,
        "quadkey_zoom": 0,
        "bounds": [-124.0, 37.0, -122.0, 39.0],
        "center": [-123.0, 38.0, 10],
        "tiles": { "": [utm_path.to_str().unwrap()] }
    });
    std::fs::write(&one_mj_path, one_mj.to_string()).unwrap();

    // 2. Heterogeneous-grid mosaic & STAC item: 64x64 (val=10) + 32x32 (val=20) over same 4326 bounds
    let g_bounds = [-122.6953125, 37.71859032558816, -122.34375, 37.996162679728116];
    let g64_path = tmp.join("g64.tif");
    let g32_path = tmp.join("g32.tif");
    write_synthetic_u16_2band_4326_geotiff(&g64_path, 64, 64, g_bounds, 10, 100, None);
    write_synthetic_u16_2band_4326_geotiff(&g32_path, 32, 32, g_bounds, 20, 200, None);
    let het_mj_path = tmp.join("het.json");
    let het_mj = serde_json::json!({
        "mosaicjson": "0.0.3",
        "minzoom": 0,
        "maxzoom": 14,
        "quadkey_zoom": 0,
        "bounds": g_bounds,
        "center": [(g_bounds[0] + g_bounds[2]) / 2.0, (g_bounds[1] + g_bounds[3]) / 2.0, 10],
        "tiles": { "": [g64_path.to_str().unwrap(), g32_path.to_str().unwrap()] }
    });
    std::fs::write(&het_mj_path, het_mj.to_string()).unwrap();

    let stac_path = tmp.join("het_stac.json");
    let stac = serde_json::json!({
        "type": "Feature",
        "stac_version": "1.0.0",
        "id": "het-stac",
        "bbox": g_bounds,
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [g_bounds[0], g_bounds[1]],
                [g_bounds[2], g_bounds[1]],
                [g_bounds[2], g_bounds[3]],
                [g_bounds[0], g_bounds[3]],
                [g_bounds[0], g_bounds[1]]
            ]]
        },
        "properties": { "datetime": "2025-01-01T00:00:00Z" },
        "assets": {
            "hi": { "href": g64_path.to_str().unwrap(), "type": "image/tiff; application=geotiff" },
            "lo": { "href": g32_path.to_str().unwrap(), "type": "image/tiff; application=geotiff" }
        }
    });
    std::fs::write(&stac_path, stac.to_string()).unwrap();

    let engine = Arc::new(TileEngine::new(16 * 1024 * 1024));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let srv_task = tokio::spawn(async move {
        let _ = serve_listener(engine, listener).await;
    });
    let base = format!("http://{addr}");
    let client = reqwest::Client::new();

    // Small tilted quadrilateral in UTM 10N (fractional pixel coverage!)
    let utm_quad = serde_json::json!({
        "type": "Feature",
        "properties": {},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [500_103.3, 4_200_121.6],
                [500_412.7, 4_200_154.2],
                [500_389.4, 4_200_487.8],
                [500_124.1, 4_200_441.5],
                [500_103.3, 4_200_121.6]
            ]]
        }
    });

    // A. One-file mosaic POST /mosaicjson/statistics == POST /cog/statistics to 1e-9
    let utm_u = utm_path.to_string_lossy();
    let one_u = one_mj_path.to_string_lossy();
    for extra in ["", "&categorical=true"] {
        let r_cog = client
            .post(format!(
                "{base}/cog/statistics?url={utm_u}&bidx=1&coord_crs=EPSG:32610{extra}"
            ))
            .header("content-type", "application/json")
            .body(utm_quad.to_string())
            .send()
            .await
            .unwrap();
        assert_eq!(r_cog.status(), 200);
        let j_cog: serde_json::Value = serde_json::from_str(&r_cog.text().await.unwrap()).unwrap();

        let r_mos = client
            .post(format!(
                "{base}/mosaicjson/statistics?url={one_u}&bidx=1&coord_crs=EPSG:32610{extra}"
            ))
            .header("content-type", "application/json")
            .body(utm_quad.to_string())
            .send()
            .await
            .unwrap();
        assert_eq!(r_mos.status(), 200);
        let j_mos: serde_json::Value = serde_json::from_str(&r_mos.text().await.unwrap()).unwrap();

        let bc = &j_cog["properties"]["statistics"]["b1"];
        let bm = &j_mos["properties"]["statistics"]["b1"];
        for field in [
            "count",
            "mean",
            "std",
            "sum",
            "min",
            "max",
            "valid_pixels",
            "masked_pixels",
            "valid_percent",
            "area",
        ] {
            let vc = bc[field].as_f64().unwrap_or_else(|| panic!("cog missing {field}: {bc}"));
            let vm = bm[field].as_f64().unwrap_or_else(|| panic!("mosaic missing {field}: {bm}"));
            assert!(
                (vc - vm).abs() <= 1e-9,
                "GG1 one-file mosaic vs /cog/statistics mismatch on {field} ({extra}): cog={vc} mos={vm}"
            );
        }
        assert_eq!(bm["valid_percent"].as_f64().unwrap(), 100.0);
        assert!(bm["count"].as_f64().unwrap().fract().abs() > 1e-3, "count must be fractional");
        if !extra.is_empty() {
            let cc = bc["category_areas"].as_object().expect("cog category_areas");
            let cm = bm["category_areas"].as_object().expect("mosaic category_areas");
            assert_eq!(cc.len(), cm.len());
            for (k, v) in cc {
                let vc = v.as_f64().unwrap();
                let vm = cm[k].as_f64().unwrap();
                assert!(
                    (vc - vm).abs() <= 1e-9,
                    "GG1 category_areas[{k}] mismatch: cog={vc} mos={vm}"
                );
            }
        }
    }

    // B. Heterogeneous-grid mosaic POST /mosaicjson/statistics keeps exact coverage & polygon valid_percent
    let g_quad = serde_json::json!({
        "type": "Feature",
        "properties": {},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [-122.62, 37.76],
                [-122.41, 37.78],
                [-122.43, 37.94],
                [-122.60, 37.91],
                [-122.62, 37.76]
            ]]
        }
    });
    let het_u = het_mj_path.to_string_lossy();
    let r_het = client
        .post(format!(
            "{base}/mosaicjson/statistics?url={het_u}&bidx=1&pixel_selection=highest"
        ))
        .header("content-type", "application/json")
        .body(g_quad.to_string())
        .send()
        .await
        .unwrap();
    assert_eq!(r_het.status(), 200);
    let j_het: serde_json::Value = serde_json::from_str(&r_het.text().await.unwrap()).unwrap();
    let b_het = &j_het["properties"]["statistics"]["b1"];
    assert!(
        (b_het["mean"].as_f64().unwrap() - 20.0).abs() < 1e-6,
        "heterogeneous mosaic highest mean expected 20.0, got {b_het}"
    );
    assert_eq!(
        b_het["valid_percent"].as_f64().unwrap(),
        100.0,
        "heterogeneous mosaic valid_percent must be 100.0 against polygon, got {b_het}"
    );
    assert_eq!(b_het["masked_pixels"].as_f64().unwrap(), 0.0);
    assert!(
        b_het["area"].as_f64().unwrap() > 1e6,
        "heterogeneous mosaic must include positive ellipsoidal area, got {b_het}"
    );
    assert!(
        b_het["count"].as_f64().unwrap().fract().abs() > 1e-3,
        "heterogeneous mosaic count must be fractional coverage sum, got {b_het}"
    );

    // C. Heterogeneous-grid POST /stac/statistics with expression keeps exact coverage & polygon valid_percent
    // (the two-band assets are stacked band by band: `b3` is `lo`'s first band).
    let stac_u = stac_path.to_string_lossy();
    let r_stac = client
        .post(format!(
            "{base}/stac/statistics?url={stac_u}&assets=hi&assets=lo&expression=b1%2Bb3"
        ))
        .header("content-type", "application/json")
        .body(g_quad.to_string())
        .send()
        .await
        .unwrap();
    assert_eq!(r_stac.status(), 200);
    let j_stac: serde_json::Value = serde_json::from_str(&r_stac.text().await.unwrap()).unwrap();
    let b_stac = &j_stac["properties"]["statistics"]["b1"];
    assert!(
        (b_stac["mean"].as_f64().unwrap() - 30.0).abs() < 1e-6,
        "heterogeneous STAC expr mean expected 30.0, got {b_stac}"
    );
    assert_eq!(
        b_stac["valid_percent"].as_f64().unwrap(),
        100.0,
        "heterogeneous STAC expr valid_percent must be 100.0 against polygon, got {b_stac}"
    );
    assert!(
        b_stac["area"].as_f64().unwrap() > 1e6,
        "heterogeneous STAC expr must include positive ellipsoidal area, got {b_stac}"
    );
    assert!(
        b_stac["count"].as_f64().unwrap().fract().abs() > 1e-3,
        "heterogeneous STAC expr count must be fractional coverage sum, got {b_stac}"
    );

    srv_task.abort();
    let _ = std::fs::remove_dir_all(&tmp);
}

#[tokio::test]
async fn audit75_power_expr_npy_and_unscale_heterogeneous_geotiff_dtypes() {
    let tmp = std::env::temp_dir().join(format!(
        "cogzilla_audit75_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&tmp).unwrap();

    // Helper: write a minimal 1-band EPSG:4326 uncompressed GeoTIFF with optional GDAL_METADATA.
    let write_1band_4326 = |path: &std::path::Path,
                            bits: u16,
                            sample_format: u16,
                            pixels: &[u8],
                            gdal_meta: Option<&str>| {
        let w = 16u32;
        let h = 16u32;
        let bounds = [-122.5f64, 37.7, -122.3, 37.9];
        let mut buf = Vec::<u8>::new();
        buf.extend_from_slice(b"II");
        buf.extend_from_slice(&42u16.to_le_bytes());
        buf.extend_from_slice(&8u32.to_le_bytes());

        let meta_bytes = gdal_meta.map(|s| {
            let mut v = s.as_bytes().to_vec();
            v.push(0);
            v
        });
        let num_entries: u16 = 14 + u16::from(meta_bytes.is_some());
        let ifd_size = 2 + (num_entries as usize) * 12 + 4;
        let data_start = 8 + ifd_size;

        let scale_off = data_start as u32;
        let tiepoint_off = scale_off + 24;
        let geokey_off = tiepoint_off + 48;
        let meta_off = geokey_off + 24;
        let meta_len = meta_bytes.as_ref().map_or(0, Vec::len) as u32;
        let pixel_off = meta_off + meta_len;

        let push_entry = |b: &mut Vec<u8>, tag: u16, ty: u16, count: u32, val: u32| {
            b.extend_from_slice(&tag.to_le_bytes());
            b.extend_from_slice(&ty.to_le_bytes());
            b.extend_from_slice(&count.to_le_bytes());
            b.extend_from_slice(&val.to_le_bytes());
        };

        buf.extend_from_slice(&num_entries.to_le_bytes());
        push_entry(&mut buf, 256, 4, 1, w); // ImageWidth
        push_entry(&mut buf, 257, 4, 1, h); // ImageLength
        push_entry(&mut buf, 258, 3, 1, bits as u32); // BitsPerSample
        push_entry(&mut buf, 259, 3, 1, 1); // Compression = None
        push_entry(&mut buf, 262, 3, 1, 1); // Photometric = MinIsBlack
        push_entry(&mut buf, 273, 4, 1, pixel_off); // StripOffsets
        push_entry(&mut buf, 277, 3, 1, 1); // SamplesPerPixel = 1
        push_entry(&mut buf, 278, 4, 1, h); // RowsPerStrip
        push_entry(&mut buf, 279, 4, 1, pixels.len() as u32); // StripByteCounts
        push_entry(&mut buf, 284, 3, 1, 1); // PlanarConfiguration = Chunky
        push_entry(&mut buf, 339, 3, 1, sample_format as u32); // SampleFormat
        push_entry(&mut buf, 33550, 12, 3, scale_off); // ModelPixelScaleTag
        push_entry(&mut buf, 33922, 12, 6, tiepoint_off); // ModelTiepointTag
        push_entry(&mut buf, 34735, 3, 12, geokey_off); // GeoKeyDirectoryTag
        if meta_len > 0 {
            push_entry(&mut buf, 42112, 2, meta_len, meta_off); // GDAL_METADATA
        }
        buf.extend_from_slice(&0u32.to_le_bytes()); // Next IFD = 0

        let sx = (bounds[2] - bounds[0]) / w as f64;
        let sy = (bounds[3] - bounds[1]) / h as f64;
        for v in [sx, sy, 0.0f64] {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        for v in [0.0f64, 0.0, 0.0, bounds[0], bounds[3], 0.0] {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        for v in [1u16, 1, 0, 2, 1024, 0, 1, 2, 2048, 0, 1, 4326] {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        if let Some(m) = meta_bytes {
            buf.extend_from_slice(&m);
        }
        buf.extend_from_slice(pixels);
        std::fs::write(path, buf).unwrap();
    };

    let u8_path = tmp.join("a_u8.tif");
    let u8_scaled_path = tmp.join("scaled_u8.tif");
    let u16_path = tmp.join("b_u16.tif");
    let u16_two_path = tmp.join("two_u16.tif");
    let f32_path = tmp.join("c_f32.tif");

    let u8_px = vec![50u8; 16 * 16];
    let u16_px: Vec<u8> = [500u16; 16 * 16].iter().flat_map(|v| v.to_le_bytes()).collect();
    let u16_two_px: Vec<u8> = [2u16; 16 * 16].iter().flat_map(|v| v.to_le_bytes()).collect();
    let f32_px: Vec<u8> = [1234.5f32; 16 * 16].iter().flat_map(|v| v.to_le_bytes()).collect();

    write_1band_4326(&u8_path, 8, 1, &u8_px, None);
    write_1band_4326(
        &u8_scaled_path,
        8,
        1,
        &u8_px,
        Some(r#"<GDALMetadata><Item name="scale" sample="0" role="scale">0.01</Item><Item name="offset" sample="0" role="offset">10.0</Item></GDALMetadata>"#),
    );
    write_1band_4326(&u16_path, 16, 1, &u16_px, None);
    write_1band_4326(&u16_two_path, 16, 1, &u16_two_px, None);
    write_1band_4326(&f32_path, 32, 3, &f32_px, None);

    let stac_path = tmp.join("item.json");
    let stac_json = serde_json::json!({
        "type": "Feature",
        "stac_version": "1.0.0",
        "id": "audit75-het",
        "bbox": [-122.5, 37.7, -122.3, 37.9],
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[-122.5, 37.7], [-122.3, 37.7], [-122.3, 37.9], [-122.5, 37.9], [-122.5, 37.7]]]
        },
        "properties": {},
        "assets": {
            "a_u8": { "href": u8_path.to_string_lossy() },
            "b_u16": { "href": u16_path.to_string_lossy() },
            "c_f32": { "href": f32_path.to_string_lossy() }
        }
    });
    std::fs::write(&stac_path, serde_json::to_string(&stac_json).unwrap()).unwrap();

    let engine = Arc::new(TileEngine::new(16 * 1024 * 1024));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let base = format!("http://{addr}");
    let srv_task = tokio::spawn(async move {
        let _ = serve_listener(engine, listener).await;
    });
    let client = reqwest::Client::new();

    // A. Power expressions (`**` and `^`) on integer COGs must promote to Float64 (<f8), not Int32.
    for expr in ["b1**-1", "b1%5E-1"] {
        let r = client
            .get(format!(
                "{base}/cog/preview.npy?url={}&expression={expr}&return_mask=false",
                u16_two_path.to_string_lossy()
            ))
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 200, "expr {expr} failed");
        let bytes = r.bytes().await.unwrap();
        let hdr_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        let hdr = std::str::from_utf8(&bytes[10..10 + hdr_len]).unwrap();
        assert!(hdr.contains("'<f8'"), "expected Float64 <f8 for {expr}, got header: {hdr}");
        let body = &bytes[10 + hdr_len..];
        let v0 = f64::from_le_bytes(body[..8].try_into().unwrap());
        assert!((v0 - 0.5).abs() < 1e-6, "expected 0.5 for 2^-1 ({expr}), got {v0}");
    }

    // B. `unscale=true` on a Uint8 COG requesting `.tif` must emit Float32 GeoTIFF, not 8-bit RGBA.
    let r_unscale = client
        .get(format!(
            "{base}/cog/preview.tif?url={}&unscale=true",
            u8_scaled_path.to_string_lossy()
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(r_unscale.status(), 200);
    let tif_unscale = r_unscale.bytes().await.unwrap();
    let idx_unscale = cogzilla_tiff::parse_cog(&tif_unscale).unwrap();
    assert_eq!(
        idx_unscale.full_res().sample_type,
        cogzilla_raster::SampleType::Float32,
        "unscale=true on Uint8 COG must emit Float32 GeoTIFF"
    );

    // C. Heterogeneous STAC assets starting with Uint8 requesting `.tif` must emit Float32 GeoTIFF.
    let r_stac = client
        .get(format!(
            "{base}/stac/preview.tif?url={}&assets=a_u8&assets=b_u16&assets=c_f32&return_mask=false",
            stac_path.to_string_lossy()
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(r_stac.status(), 200);
    let tif_stac = r_stac.bytes().await.unwrap();
    let idx_stac = cogzilla_tiff::parse_cog(&tif_stac).unwrap();
    assert_eq!(
        idx_stac.full_res().sample_type,
        cogzilla_raster::SampleType::Float32,
        "heterogeneous STAC (u8, u16, f32) must emit Float32 GeoTIFF"
    );
    assert_eq!(idx_stac.full_res().samples_per_pixel, 3);

    srv_task.abort();
    let _ = std::fs::remove_dir_all(&tmp);
}

#[tokio::test]
async fn audit77_shown_npy_multiband_palette_and_invalid_formula() {
    let tmp = std::env::temp_dir().join(format!(
        "cogzilla_audit77_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&tmp).unwrap();

    // 1. 2-band uint16 EPSG:4326 GeoTIFF (band1=50, band2=100)
    let two_band_path = tmp.join("two_band.tif");
    write_synthetic_u16_2band_4326_geotiff(
        &two_band_path,
        16,
        16,
        [-122.5, 37.7, -122.3, 37.9],
        50,
        100,
        None,
    );

    // 2. 3-band RGB uint8 EPSG:4326 GeoTIFF for WebMercator tile invalid color_formula test
    let rgb_path = tmp.join("rgb3.tif");
    {
        let w = 16u32;
        let h = 16u32;
        let bounds = [-122.5f64, 37.7, -122.3, 37.9];
        let rgba = vec![100u8; (w * h * 4) as usize];
        let tif = cogzilla_server::geotiff::encode_geotiff_rgba(w, h, &rgba, bounds, 4326, false);
        std::fs::write(&rgb_path, tif).unwrap();
    }

    // 3. 1-band uint8 paletted GeoTIFF with TIFF ColorMap tag (index 1 -> RGB(10, 20, 30))
    let pal_path = tmp.join("paletted.tif");
    {
        let w = 8u32;
        let h = 8u32;
        let bounds = [-122.5f64, 37.7, -122.3, 37.9];
        let pixels = vec![1u8; (w * h) as usize];
        let mut cmap = vec![0u16; 256 * 3];
        cmap[1] = 10u16 << 8;
        cmap[256 + 1] = 20u16 << 8;
        cmap[512 + 1] = 30u16 << 8;
        let mut buf = Vec::<u8>::new();
        buf.extend_from_slice(b"II");
        buf.extend_from_slice(&42u16.to_le_bytes());
        buf.extend_from_slice(&8u32.to_le_bytes());
        let num_entries: u16 = 14;
        let data_start = 8 + 2 + (num_entries as usize) * 12 + 4;
        let scale_off = data_start as u32;
        let tiepoint_off = scale_off + 24;
        let geokey_off = tiepoint_off + 48;
        let cmap_off = geokey_off + 24;
        let pixel_off = cmap_off + (cmap.len() * 2) as u32;
        let push_entry = |b: &mut Vec<u8>, tag: u16, ty: u16, count: u32, val: u32| {
            b.extend_from_slice(&tag.to_le_bytes());
            b.extend_from_slice(&ty.to_le_bytes());
            b.extend_from_slice(&count.to_le_bytes());
            b.extend_from_slice(&val.to_le_bytes());
        };
        buf.extend_from_slice(&num_entries.to_le_bytes());
        push_entry(&mut buf, 256, 4, 1, w);
        push_entry(&mut buf, 257, 4, 1, h);
        push_entry(&mut buf, 258, 3, 1, 8);
        push_entry(&mut buf, 259, 3, 1, 1);
        push_entry(&mut buf, 262, 3, 1, 3); // Photometric = Palette
        push_entry(&mut buf, 273, 4, 1, pixel_off);
        push_entry(&mut buf, 277, 3, 1, 1);
        push_entry(&mut buf, 278, 4, 1, h);
        push_entry(&mut buf, 279, 4, 1, pixels.len() as u32);
        push_entry(&mut buf, 284, 3, 1, 1);
        push_entry(&mut buf, 320, 3, (cmap.len()) as u32, cmap_off); // ColorMap
        push_entry(&mut buf, 33550, 12, 3, scale_off);
        push_entry(&mut buf, 33922, 12, 6, tiepoint_off);
        push_entry(&mut buf, 34735, 3, 12, geokey_off);
        buf.extend_from_slice(&0u32.to_le_bytes());
        let sx = (bounds[2] - bounds[0]) / w as f64;
        let sy = (bounds[3] - bounds[1]) / h as f64;
        for v in [sx, sy, 0.0f64] {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        for v in [0.0f64, 0.0, 0.0, bounds[0], bounds[3], 0.0] {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        for v in [1u16, 1, 0, 2, 1024, 0, 1, 2, 2048, 0, 1, 4326] {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        for v in cmap {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf.extend_from_slice(&pixels);
        std::fs::write(&pal_path, buf).unwrap();
    }

    let engine = Arc::new(TileEngine::new(16 * 1024 * 1024));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let base = format!("http://{addr}");
    let srv_task = tokio::spawn(async move {
        let _ = serve_listener(engine, listener).await;
    });
    let client = reqwest::Client::new();

    // A. 2-band rescaled `.npy` preserves both bands + mask -> shape (3, 16, 16) |u1
    let r2 = client
        .get(format!(
            "{base}/cog/preview.npy?url={}&bidx=1&bidx=2&rescale=0,100&rescale=0,200",
            two_band_path.to_string_lossy()
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(r2.status(), 200);
    let b2 = r2.bytes().await.unwrap();
    let hdr_len = u16::from_le_bytes([b2[8], b2[9]]) as usize;
    let hdr = std::str::from_utf8(&b2[10..10 + hdr_len]).unwrap();
    assert!(hdr.contains("'|u1'"), "expected |u1, got {hdr}");
    assert!(hdr.contains("(3, 16, 16)"), "expected 2 bands + mask (3, 16, 16), got {hdr}");
    let body = &b2[10 + hdr_len..];
    assert_eq!(body[0], 127);
    assert_eq!(body[256], 127);
    assert_eq!(body[512], 255);

    // B. Paletted COG with color_formula on `.npy` resolves embedded TIFF ColorMap -> 3-band RGB + mask
    let r_pal = client
        .get(format!(
            "{base}/cog/preview.npy?url={}&color_formula=gamma%20rgb%201.0",
            pal_path.to_string_lossy()
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(r_pal.status(), 200);
    let b_pal = r_pal.bytes().await.unwrap();
    let hdr_len = u16::from_le_bytes([b_pal[8], b_pal[9]]) as usize;
    let hdr = std::str::from_utf8(&b_pal[10..10 + hdr_len]).unwrap();
    assert!(hdr.contains("'|u1'"), "expected |u1, got {hdr}");
    assert!(hdr.contains("(4, 8, 8)"), "expected RGB + mask (4, 8, 8) from embedded palette, got {hdr}");
    let body = &b_pal[10 + hdr_len..];
    assert_eq!((body[0], body[64], body[128], body[192]), (10, 20, 30, 255));

    // C. Invalid color_formula on 3-band RGB WebMercator `.npy` tile returns 400 (not 200 raw <f4)
    let r_bad = client
        .get(format!(
            "{base}/cog/tiles/WebMercatorQuad/12/655/1583.npy?url={}&color_formula=bogus_op",
            rgb_path.to_string_lossy()
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(r_bad.status(), 400, "invalid color_formula on RGB .npy tile must return 400");

    srv_task.abort();
    let _ = std::fs::remove_dir_all(&tmp);
}

#[tokio::test]
async fn sweeps_34_to_37_kk1_ii1_jj1_hh1_self_contained() {
    let tmp = std::env::temp_dir().join(format!(
        "cogzilla_sweeps34_37_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&tmp).unwrap();

    // 1. Float32 EPSG:3857 GeoTIFF covering WebMercator tile 10/164/395 (San Francisco) for KK1.
    let tile_b = cogzilla_proj::tile_bounds_3857(cogzilla_proj::TileCoord { z: 10, x: 164, y: 395 });
    let f32_path = tmp.join("kk1_f32.tif");
    {
        let w = 32u32;
        let h = 32u32;
        let raw: Vec<u8> = (0..(w * h))
            .flat_map(|i| (100.0f32 + (i % 32) as f32).to_le_bytes())
            .collect();
        let tif = cogzilla_server::geotiff::encode_geotiff_f32(w, h, &raw, tile_b.into(), 3857, false);
        std::fs::write(&f32_path, tif).unwrap();
    }
    let f32_stac_path = tmp.join("kk1_stac.json");
    let f32_wgs = [-122.4, 37.7, -122.0, 38.0];
    std::fs::write(
        &f32_stac_path,
        serde_json::json!({
            "type": "Feature",
            "stac_version": "1.0.0",
            "id": "kk1-f32",
            "bbox": f32_wgs,
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[-122.4, 37.7], [-122.0, 37.7], [-122.0, 38.0], [-122.4, 38.0], [-122.4, 37.7]]]
            },
            "properties": {},
            "assets": { "data": { "href": f32_path.to_string_lossy() } }
        })
        .to_string(),
    )
    .unwrap();
    let f32_mj_path = tmp.join("kk1_mosaic.json");
    // Quadkey for z=10, x=164, y=395
    let mut qk_sf = String::new();
    for i in (0..10).rev() {
        let mut d = 0u8;
        if (164u32 >> i) & 1 != 0 { d |= 1; }
        if (395u32 >> i) & 1 != 0 { d |= 2; }
        qk_sf.push((b'0' + d) as char);
    }
    std::fs::write(
        &f32_mj_path,
        serde_json::json!({
            "mosaicjson": "0.0.3",
            "minzoom": 10,
            "maxzoom": 10,
            "bounds": [-122.7, 37.4, -121.9, 38.1],
            "quadkey_zoom": 10,
            "tiles": { qk_sf: [f32_path.to_string_lossy()] }
        })
        .to_string(),
    )
    .unwrap();

    // 2. EPSG:32701 antimeridian GeoTIFF spanning ~179E..176W (UTM Zone 1S, ~70000..600000 E, ~8000000..8230000 N)
    //    for II1 (dateline feature at 64x64 / 512x64 / unsized) and JJ1 (symmetric lon norm).
    let am_path = tmp.join("am32701.tif");
    {
        let w = 64u32;
        let h = 64u32;
        // Column-coded pixel values so different longitudes return distinct known values.
        let mut raw = Vec::with_capacity((w * h * 4) as usize);
        for _r in 0..h {
            for c in 0..w {
                raw.extend_from_slice(&((c as f32) * 10.0 + 1.0).to_le_bytes());
            }
        }
        // In EPSG:32701 (central meridian -177°):
        // lon=179°E (-181°) at lat=-17° -> x ≈ 74_300 m; lon=-176°W at lat=-17° -> x ≈ 606_500 m
        // lat=-18° -> y ≈ 8_009_000 m; lat=-16° -> y ≈ 8_231_000 m
        let utm_bbox = [74_300.0, 8_009_000.0, 606_500.0, 8_231_000.0];
        let tif = cogzilla_server::geotiff::encode_geotiff_f32(w, h, &raw, utm_bbox, 32701, false);
        std::fs::write(&am_path, tif).unwrap();
    }
    let am_stac_path = tmp.join("am_stac.json");
    std::fs::write(
        &am_stac_path,
        serde_json::json!({
            "type": "Feature",
            "stac_version": "1.0.0",
            "id": "am32701",
            "bbox": [-180.0, -18.0, 180.0, -16.0],
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[-180.0, -18.0], [180.0, -18.0], [180.0, -16.0], [-180.0, -16.0], [-180.0, -18.0]]]
            },
            "properties": {},
            "assets": { "data": { "href": am_path.to_string_lossy() } }
        })
        .to_string(),
    )
    .unwrap();
    let am_mj_path = tmp.join("am_mosaic.json");
    // Quadkeys at z=0 ("") cover the whole world so both sides of antimeridian resolve am32701.tif.
    std::fs::write(
        &am_mj_path,
        serde_json::json!({
            "mosaicjson": "0.0.3",
            "minzoom": 0,
            "maxzoom": 12,
            "bounds": [-180.0, -18.0, 180.0, -16.0],
            "quadkey_zoom": 0,
            "tiles": { "": [am_path.to_string_lossy()] }
        })
        .to_string(),
    )
    .unwrap();

    // 3. File with space in its name ("my scene.tif", 2-band uint16 with b1=10, b2=20) for HH1.
    let space_path = tmp.join("my scene.tif");
    write_synthetic_u16_2band_4326_geotiff(
        &space_path,
        16,
        16,
        [-122.5, 37.7, -122.3, 37.9],
        10,
        20,
        None,
    );

    let engine = Arc::new(TileEngine::new(16 * 1024 * 1024));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let base = format!("http://{addr}");
    let srv_task = tokio::spawn(async move {
        let _ = serve_listener(engine, listener).await;
    });
    let client = reqwest::Client::new();

    // --- KK1: /cog, /stac, and /mosaicjson share identical .npy mask (f32::MAX) and .tif nodata (None) ---
    let parse_npy_hdr_and_mask_first = |bytes: &[u8]| -> (String, f32) {
        let hlen = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        let hdr = std::str::from_utf8(&bytes[10..10 + hlen]).unwrap().to_string();
        let vals = decode_npy_f32(bytes);
        let half = vals.len() / 2;
        (hdr, vals[half])
    };
    let mut npy_masks = Vec::new();
    for url in [
        format!("{base}/cog/tiles/WebMercatorQuad/10/164/395.npy?url={}", f32_path.to_string_lossy()),
        format!("{base}/stac/tiles/WebMercatorQuad/10/164/395.npy?url={}&assets=data", f32_stac_path.to_string_lossy()),
        format!("{base}/mosaicjson/tiles/WebMercatorQuad/10/164/395.npy?url={}", f32_mj_path.to_string_lossy()),
        format!("{base}/mosaicjson/bbox/-122.3,37.75,-122.1,37.9/32x32.npy?url={}", f32_mj_path.to_string_lossy()),
    ] {
        let r = client.get(&url).send().await.unwrap();
        assert_eq!(r.status(), 200, "KK1 npy {url} failed");
        let b = r.bytes().await.unwrap();
        let (hdr, m0) = parse_npy_hdr_and_mask_first(&b);
        assert!(hdr.contains("'<f4'"), "KK1 expected <f4 for {url}, got {hdr}");
        assert_eq!(m0, f32::MAX, "KK1 expected f32::MAX mask for {url}, got {m0}");
        npy_masks.push(m0);
    }
    assert!(npy_masks.iter().all(|&m| m == f32::MAX));

    for url in [
        format!("{base}/cog/tiles/WebMercatorQuad/10/164/395.tif?url={}", f32_path.to_string_lossy()),
        format!("{base}/stac/tiles/WebMercatorQuad/10/164/395.tif?url={}&assets=data", f32_stac_path.to_string_lossy()),
        format!("{base}/mosaicjson/tiles/WebMercatorQuad/10/164/395.tif?url={}", f32_mj_path.to_string_lossy()),
    ] {
        let r = client.get(&url).send().await.unwrap();
        assert_eq!(r.status(), 200, "KK1 tif {url} failed");
        let tif = r.bytes().await.unwrap();
        let idx = cogzilla_tiff::parse_cog(&tif).unwrap();
        assert_eq!(idx.full_res().sample_type, cogzilla_raster::SampleType::Float32);
        assert_eq!(idx.nodata, None, "KK1 expected nodata=None for {url}, got {:?}", idx.nodata);
    }
    // Styled mosaic .npy with rescale returns |u1 matching /cog
    let r_mj_styled = client
        .get(format!(
            "{base}/mosaicjson/tiles/WebMercatorQuad/10/164/395.npy?url={}&rescale=100,131",
            f32_mj_path.to_string_lossy()
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(r_mj_styled.status(), 200);
    let b_mj_styled = r_mj_styled.bytes().await.unwrap();
    let hlen = u16::from_le_bytes([b_mj_styled[8], b_mj_styled[9]]) as usize;
    assert!(std::str::from_utf8(&b_mj_styled[10..10 + hlen]).unwrap().contains("'|u1'"));

    // --- II1: literal 358-degree dateline ring renders 200 at unsized, 512x64, and 64x64 across /cog, /stac, /mosaicjson ---
    let dateline_ring = serde_json::json!({
        "type": "Feature",
        "properties": {},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [179.5, -17.5],
                [-178.5, -17.5],
                [-178.5, -16.5],
                [179.5, -16.5],
                [179.5, -17.5]
            ]]
        }
    });
    for url in [
        format!("{base}/cog/feature.npy?url={}", am_path.to_string_lossy()),
        format!("{base}/cog/feature/512x64.npy?url={}", am_path.to_string_lossy()),
        format!("{base}/cog/feature/64x64.npy?url={}", am_path.to_string_lossy()),
        format!("{base}/cog/feature.npy?url={}&max_size=64", am_path.to_string_lossy()),
        format!("{base}/cog/feature/64x64.png?url={}", am_path.to_string_lossy()),
        format!("{base}/stac/feature/64x64.npy?url={}&assets=data", am_stac_path.to_string_lossy()),
        format!("{base}/mosaicjson/feature/64x64.npy?url={}", am_mj_path.to_string_lossy()),
    ] {
        let r = client
            .post(&url)
            .header("content-type", "application/json")
            .body(dateline_ring.to_string())
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 200, "II1 expected 200 for {url}, got {}", r.status());
    }
    // Genuinely non-overlapping polygon still returns 404
    let disjoint_poly = serde_json::json!({
        "type": "Feature",
        "properties": {},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[10.0, 10.0], [11.0, 10.0], [11.0, 11.0], [10.0, 11.0], [10.0, 10.0]]]
        }
    });
    let r_disjoint = client
        .post(format!("{base}/cog/feature/64x64.npy?url={}", am_path.to_string_lossy()))
        .header("content-type", "application/json")
        .body(disjoint_poly.to_string())
        .send()
        .await
        .unwrap();
    assert_eq!(r_disjoint.status(), 404, "disjoint polygon must still 404");

    // --- JJ1: symmetric ±2-turn longitude normalisation across /cog/point and /mosaicjson/point ---
    let mut vals_by_lon = std::collections::HashMap::new();
    for lon in ["179.5", "180.5", "-180.5", "181.5", "-540.5", "540.5"] {
        let r_cog = client
            .get(format!("{base}/cog/point/{lon},-17?url={}", am_path.to_string_lossy()))
            .send()
            .await
            .unwrap();
        assert_eq!(r_cog.status(), 200, "JJ1 /cog/point/{lon},-17 failed");
        let j_cog: serde_json::Value = serde_json::from_str(&r_cog.text().await.unwrap()).unwrap();
        let v_cog = j_cog["values"][0].as_f64().unwrap();

        let r_mj = client
            .get(format!("{base}/mosaicjson/point/{lon},-17?url={}", am_mj_path.to_string_lossy()))
            .send()
            .await
            .unwrap();
        assert_eq!(r_mj.status(), 200, "JJ1 /mosaicjson/point/{lon},-17 failed");
        let j_mj: serde_json::Value = serde_json::from_str(&r_mj.text().await.unwrap()).unwrap();
        let v_mj = j_mj["assets"][0]["values"][0].as_f64().unwrap();

        assert_eq!(v_cog, v_mj, "JJ1 /cog vs /mosaicjson mismatch at lon={lon}");
        vals_by_lon.insert(lon, v_cog);
    }
    // Periodic equivalence: 179.5 == -180.5 == -540.5, and 180.5 == 540.5
    assert_eq!(vals_by_lon["179.5"], vals_by_lon["-180.5"]);
    assert_eq!(vals_by_lon["179.5"], vals_by_lon["-540.5"]);
    assert_eq!(vals_by_lon["180.5"], vals_by_lon["540.5"]);
    // Beyond ±720° returns 404 symmetrically on both routes
    for far_lon in ["900.5", "-900.5"] {
        let rc = client
            .get(format!("{base}/cog/point/{far_lon},-17?url={}", am_path.to_string_lossy()))
            .send()
            .await
            .unwrap();
        assert_eq!(rc.status(), 404, "JJ1 /cog/point/{far_lon} must 404");
        let rm = client
            .get(format!("{base}/mosaicjson/point/{far_lon},-17?url={}", am_mj_path.to_string_lossy()))
            .send()
            .await
            .unwrap();
        assert_eq!(rm.status(), 404, "JJ1 /mosaicjson/point/{far_lon} must 404");
    }

    // --- HH1: `+` in url= decoded as space (quote_plus / httpx params), while raw + and %2B in expression= both work ---
    let plus_url = space_path.to_string_lossy().replace(' ', "+");
    let pct_url = space_path.to_string_lossy().replace(' ', "%20");
    let r_plus = client
        .get(format!("{base}/cog/info?url={plus_url}"))
        .send()
        .await
        .unwrap();
    assert_eq!(r_plus.status(), 200, "HH1 url with '+' space encoding must return 200");
    let r_pct = client
        .get(format!("{base}/cog/info?url={pct_url}"))
        .send()
        .await
        .unwrap();
    assert_eq!(r_pct.status(), 200);
    for expr in ["b1+b2", "b1%2Bb2"] {
        let r_expr = client
            .get(format!("{base}/cog/statistics?url={plus_url}&expression={expr}"))
            .send()
            .await
            .unwrap();
        assert_eq!(r_expr.status(), 200, "HH1 expression={expr} failed");
        let j: serde_json::Value = serde_json::from_str(&r_expr.text().await.unwrap()).unwrap();
        assert!((j["b1"]["mean"].as_f64().unwrap() - 30.0).abs() < 1e-6, "HH1 {expr} got {j}");
    }

    srv_task.abort();
    let _ = std::fs::remove_dir_all(&tmp);
}

#[tokio::test]
async fn sweep40_mm1_validate_ungeoreferenced_and_rpc_tiffs() {
    let tmp = std::env::temp_dir().join(format!(
        "cogzilla_sweep40_mm1_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&tmp).unwrap();

    // Helper: write a minimal 16x16 uint8 TIFF with no GeoTransform/GCPs and optional RPCCoefficientTag (50844).
    let write_plain_or_rpc_tiff = |path: &std::path::Path, include_rpc: bool| {
        let w = 16u32;
        let h = 16u32;
        let pixels = vec![42u8; (w * h) as usize];
        let mut buf = Vec::<u8>::new();
        buf.extend_from_slice(b"II");
        buf.extend_from_slice(&42u16.to_le_bytes());
        buf.extend_from_slice(&8u32.to_le_bytes());

        let num_entries: u16 = 10 + u16::from(include_rpc);
        let data_start = (8 + 2 + (num_entries as usize) * 12 + 4) as u32;
        let rpc_bytes = if include_rpc { 92 * 8u32 } else { 0 };
        let rpc_off = data_start;
        let pixel_off = data_start + rpc_bytes;

        let push_entry = |b: &mut Vec<u8>, tag: u16, ty: u16, count: u32, val: u32| {
            b.extend_from_slice(&tag.to_le_bytes());
            b.extend_from_slice(&ty.to_le_bytes());
            b.extend_from_slice(&count.to_le_bytes());
            b.extend_from_slice(&val.to_le_bytes());
        };
        buf.extend_from_slice(&num_entries.to_le_bytes());
        push_entry(&mut buf, 256, 4, 1, w); // ImageWidth
        push_entry(&mut buf, 257, 4, 1, h); // ImageLength
        push_entry(&mut buf, 258, 3, 1, 8); // BitsPerSample
        push_entry(&mut buf, 259, 3, 1, 1); // Compression = None
        push_entry(&mut buf, 262, 3, 1, 1); // Photometric = MinIsBlack
        push_entry(&mut buf, 273, 4, 1, pixel_off); // StripOffsets
        push_entry(&mut buf, 277, 3, 1, 1); // SamplesPerPixel
        push_entry(&mut buf, 278, 4, 1, h); // RowsPerStrip
        push_entry(&mut buf, 279, 4, 1, pixels.len() as u32); // StripByteCounts
        push_entry(&mut buf, 284, 3, 1, 1); // PlanarConfiguration = Chunky
        if include_rpc {
            push_entry(&mut buf, 50844, 12, 92, rpc_off); // RPCCoefficientTag
        }
        buf.extend_from_slice(&0u32.to_le_bytes()); // Next IFD = 0
        if include_rpc {
            for _ in 0..92 {
                buf.extend_from_slice(&0.0f64.to_le_bytes());
            }
        }
        buf.extend_from_slice(&pixels);
        std::fs::write(path, buf).unwrap();
    };

    let plain_path = tmp.join("ungeoref.tif");
    let rpc_path = tmp.join("rpc_only.tif");
    write_plain_or_rpc_tiff(&plain_path, false);
    write_plain_or_rpc_tiff(&rpc_path, true);

    let engine = Arc::new(TileEngine::new(16 * 1024 * 1024));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let base = format!("http://{addr}");
    let srv_task = tokio::spawn(async move {
        let _ = serve_listener(engine, listener).await;
    });
    let client = reqwest::Client::new();

    // 1. Plain ungeoreferenced TIFF: /cog/validate returns 200 with structural report & warning,
    //    and /cog/point returns 400 naming GeoTransform/GCPs/RPCs.
    let r_val = client
        .get(format!("{base}/cog/validate?url={}", plain_path.to_string_lossy()))
        .send()
        .await
        .unwrap();
    assert_eq!(r_val.status(), 200, "MM1 /cog/validate must return 200 on ungeoreferenced TIFF");
    let j_val: serde_json::Value = serde_json::from_str(&r_val.text().await.unwrap()).unwrap();
    assert_eq!(j_val["COG"], serde_json::json!(true));
    assert_eq!(j_val["Profile"]["Width"], serde_json::json!(16));
    assert!(j_val["GEO"]["CRS"].is_null());
    assert!(j_val["GEO"]["BoundingBox"].is_null());
    let warns = j_val["COG_warnings"].as_array().unwrap();
    assert!(
        warns.iter().any(|w| w.as_str().unwrap_or("").contains("georeferencing")),
        "expected missing-georeferencing warning, got {warns:?}"
    );
    let r_pt = client
        .get(format!("{base}/cog/point/0,0?url={}", plain_path.to_string_lossy()))
        .send()
        .await
        .unwrap();
    assert_eq!(r_pt.status(), 400);
    let pt_body = r_pt.text().await.unwrap();
    assert!(pt_body.contains("GeoTransform/GCPs") && pt_body.contains("RPC"), "got {pt_body}");

    // 2. RPC-only TIFF (tag 50844): /cog/validate returns 200 with RPC warning,
    //    and /cog/point returns 400 naming RPCs.
    let r_rpc_val = client
        .get(format!("{base}/cog/validate?url={}", rpc_path.to_string_lossy()))
        .send()
        .await
        .unwrap();
    assert_eq!(r_rpc_val.status(), 200, "MM1 /cog/validate must return 200 on RPC-only TIFF");
    let j_rpc: serde_json::Value = serde_json::from_str(&r_rpc_val.text().await.unwrap()).unwrap();
    assert_eq!(j_rpc["COG"], serde_json::json!(true));
    let rpc_warns = j_rpc["COG_warnings"].as_array().unwrap();
    assert!(
        rpc_warns.iter().any(|w| w.as_str().unwrap_or("").contains("RPC")),
        "expected RPC warning in COG_warnings, got {rpc_warns:?}"
    );
    let r_rpc_pt = client
        .get(format!("{base}/cog/point/0,0?url={}", rpc_path.to_string_lossy()))
        .send()
        .await
        .unwrap();
    assert_eq!(r_rpc_pt.status(), 400);
    let rpc_pt_body = r_rpc_pt.text().await.unwrap();
    assert!(rpc_pt_body.contains("RPC"), "400 on RPC TIFF must name RPCs, got {rpc_pt_body}");

    srv_task.abort();
    let _ = std::fs::remove_dir_all(&tmp);
}

#[tokio::test]
async fn parity_nn1_r2_raster_layers_and_f2_sample_types() {
    let tmp = std::env::temp_dir().join(format!("cogzilla_parity_all4_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();

    // Helper: write a minimal 16x16 tiled classic GeoTIFF (EPSG:4326, bounds [0,0,16,16])
    // with arbitrary (bps, sample_format), raw tile payload, and optional tail-IFD layout.
    let write_test_geotiff = |path: &std::path::Path, bps: u16, sample_fmt: u16, payload: &[u8], ifd_at_tail: bool| {
        let write_geo_arrays = |b: &mut Vec<u8>| -> (u32, u32, u32) {
            let scale_off = b.len() as u32;
            for v in [1.0f64, 1.0, 0.0] {
                b.extend_from_slice(&v.to_le_bytes());
            }
            let tie_off = b.len() as u32;
            for v in [0.0f64, 0.0, 0.0, 0.0, 16.0, 0.0] {
                b.extend_from_slice(&v.to_le_bytes());
            }
            let gk_off = b.len() as u32;
            let gks: [u16; 16] = [
                1, 1, 0, 3,
                1024, 0, 1, 2,    // GTModelTypeGeoKey = Geographic
                1025, 0, 1, 1,    // GTRasterTypeGeoKey = PixelIsArea
                2048, 0, 1, 4326, // GeographicTypeGeoKey = WGS84
            ];
            for k in gks {
                b.extend_from_slice(&k.to_le_bytes());
            }
            (scale_off, tie_off, gk_off)
        };
        let push_entry = |b: &mut Vec<u8>, tag: u16, ty: u16, count: u32, val: u32| {
            b.extend_from_slice(&tag.to_le_bytes());
            b.extend_from_slice(&ty.to_le_bytes());
            b.extend_from_slice(&count.to_le_bytes());
            b.extend_from_slice(&val.to_le_bytes());
        };
        let build_ifd = |tile_off: u32, scale_off: u32, tie_off: u32, gk_off: u32| -> Vec<u8> {
            let mut ifd_bytes: Vec<u8> = Vec::new();
            ifd_bytes.extend_from_slice(&15u16.to_le_bytes());
            push_entry(&mut ifd_bytes, 256, 4, 1, 16);                   // ImageWidth
            push_entry(&mut ifd_bytes, 257, 4, 1, 16);                   // ImageLength
            push_entry(&mut ifd_bytes, 258, 3, 1, bps as u32);           // BitsPerSample
            push_entry(&mut ifd_bytes, 259, 3, 1, 1);                    // Compression = None
            push_entry(&mut ifd_bytes, 262, 3, 1, 1);                    // Photometric = MinIsBlack
            push_entry(&mut ifd_bytes, 277, 3, 1, 1);                    // SamplesPerPixel
            push_entry(&mut ifd_bytes, 284, 3, 1, 1);                    // PlanarConfig = Chunky
            push_entry(&mut ifd_bytes, 322, 4, 1, 16);                   // TileWidth
            push_entry(&mut ifd_bytes, 323, 4, 1, 16);                   // TileLength
            push_entry(&mut ifd_bytes, 324, 4, 1, tile_off);             // TileOffsets
            push_entry(&mut ifd_bytes, 325, 4, 1, payload.len() as u32); // TileByteCounts
            push_entry(&mut ifd_bytes, 339, 3, 1, sample_fmt as u32);    // SampleFormat
            push_entry(&mut ifd_bytes, 33550, 12, 3, scale_off);         // ModelPixelScale
            push_entry(&mut ifd_bytes, 33922, 12, 6, tie_off);           // ModelTiepoint
            push_entry(&mut ifd_bytes, 34735, 3, 16, gk_off);            // GeoKeyDirectory
            ifd_bytes.extend_from_slice(&0u32.to_le_bytes());            // next IFD = 0
            ifd_bytes
        };

        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(b"II");
        buf.extend_from_slice(&42u16.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes());
        if ifd_at_tail {
            let t_off = buf.len() as u32;
            buf.extend_from_slice(payload);
            while buf.len() < 400 {
                buf.push(0);
            }
            let (s_off, ti_off, g_off) = write_geo_arrays(&mut buf);
            let i_off = buf.len() as u32;
            buf[4..8].copy_from_slice(&i_off.to_le_bytes());
            buf.extend_from_slice(&build_ifd(t_off, s_off, ti_off, g_off));
        } else {
            buf[4..8].copy_from_slice(&8u32.to_le_bytes());
            let after_ifd = 8 + 2 + 15 * 12 + 4;
            buf.resize(after_ifd, 0);
            let (s_off, ti_off, g_off) = write_geo_arrays(&mut buf);
            let t_off = buf.len() as u32;
            buf.extend_from_slice(payload);
            let hdr_ifd = build_ifd(t_off, s_off, ti_off, g_off);
            buf[8..8 + hdr_ifd.len()].copy_from_slice(&hdr_ifd);
        }
        std::fs::write(path, &buf).unwrap();
    };

    // NN1: valid front-IFD COG vs tail-IFD GTiff
    let cog_ok_path = tmp.join("cog_front.tif");
    let gtiff_tail_path = tmp.join("gtiff_tail.tif");
    write_test_geotiff(&cog_ok_path, 8, 1, &[42u8; 256], false);
    write_test_geotiff(&gtiff_tail_path, 8, 1, &[42u8; 256], true);

    // F2: 1-bit (alternating 1,0 -> 0b1010_1010 = 0xAA), 4-bit (nibbles 3, 12 -> 0x3C), int64 (-500), uint64 (900)
    let u1_path = tmp.join("u1.tif");
    let u4_path = tmp.join("u4.tif");
    let i64_path = tmp.join("i64.tif");
    let u64_path = tmp.join("u64.tif");
    write_test_geotiff(&u1_path, 1, 1, &[0xAAu8; 32], false);
    write_test_geotiff(&u4_path, 4, 1, &[0x3Cu8; 128], false);
    let i64_bytes: Vec<u8> = std::iter::repeat_n((-500i64).to_le_bytes(), 256).flatten().collect();
    let u64_bytes: Vec<u8> = std::iter::repeat_n(900u64.to_le_bytes(), 256).flatten().collect();
    write_test_geotiff(&i64_path, 64, 2, &i64_bytes, false);
    write_test_geotiff(&u64_path, 64, 1, &u64_bytes, false);

    let engine = Arc::new(TileEngine::new(64 * 1024 * 1024));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let base = format!("http://{addr}");
    let srv_engine = engine.clone();
    let srv_task = tokio::spawn(async move {
        let _ = cogzilla_server::http::serve_listener(srv_engine, listener).await;
    });
    let client = reqwest::Client::new();

    // 1. NN1: front-IFD COG is valid; tail-IFD GTiff reports COG: false with 2 layout errors.
    let v_ok: serde_json::Value = serde_json::from_str(
        &client
            .get(format!("{base}/cog/validate?url={}", cog_ok_path.to_string_lossy()))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(v_ok["COG"], serde_json::json!(true), "front-IFD COG must pass: {v_ok}");
    assert!(v_ok["COG_errors"].as_array().unwrap().is_empty());

    let v_tail: serde_json::Value = serde_json::from_str(
        &client
            .get(format!("{base}/cog/validate?url={}", gtiff_tail_path.to_string_lossy()))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap(),
    )
    .unwrap();
    assert_eq!(v_tail["COG"], serde_json::json!(false), "tail-IFD GTiff must fail COG check: {v_tail}");
    let errs = v_tail["COG_errors"].as_array().unwrap();
    assert_eq!(errs.len(), 2, "expected 2 layout errors, got {errs:?}");
    assert!(errs[0].as_str().unwrap().contains("< 300"), "{errs:?}");
    assert!(errs[1].as_str().unwrap().contains("after its IFD"), "{errs:?}");

    // 2. TileJSON raster_layers field
    let tj: serde_json::Value = serde_json::from_str(
        &client
            .get(format!("{base}/cog/WebMercatorQuad/tilejson.json?url={}", cog_ok_path.to_string_lossy()))
            .send()
            .await
            .unwrap()
            .text()
            .await
            .unwrap(),
    )
    .unwrap();
    assert!(tj.get("raster_layers").is_some_and(|v| v.is_object()), "tilejson missing raster_layers: {tj}");

    // 3. F2: 1-bit, 4-bit, int64, uint64 info + point + statistics + .npy dtype
    for (path, exp_dtype, exp_val, exp_npy_descr) in [
        (&u1_path, "uint8", 1.0f64, "'descr': '|u1'"),
        (&u4_path, "uint8", 3.0f64, "'descr': '|u1'"),
        (&i64_path, "int64", -500.0f64, "'descr': '<i8'"),
        (&u64_path, "uint64", 900.0f64, "'descr': '<u8'"),
    ] {
        let p = path.to_string_lossy();
        let info_r = client.get(format!("{base}/cog/info?url={p}")).send().await.unwrap();
        assert_eq!(info_r.status(), 200, "info failed for {p}");
        let info: serde_json::Value = serde_json::from_str(&info_r.text().await.unwrap()).unwrap();
        assert_eq!(info["dtype"], serde_json::json!(exp_dtype), "dtype mismatch for {p}");

        let pt_r = client.get(format!("{base}/cog/point/0.5,15.5?url={p}")).send().await.unwrap();
        assert_eq!(pt_r.status(), 200, "point failed for {p}");
        let pt: serde_json::Value = serde_json::from_str(&pt_r.text().await.unwrap()).unwrap();
        assert!((pt["values"][0].as_f64().unwrap() - exp_val).abs() < 1e-4, "point mismatch for {p}: {pt}");

        let st_r = client.get(format!("{base}/cog/statistics?url={p}")).send().await.unwrap();
        assert_eq!(st_r.status(), 200, "stats failed for {p}");

        let npy_r = client.get(format!("{base}/cog/preview.npy?url={p}")).send().await.unwrap();
        assert_eq!(npy_r.status(), 200, "preview.npy failed for {p}");
        let npy_bytes = npy_r.bytes().await.unwrap();
        let hdr = std::str::from_utf8(&npy_bytes[10..128.min(npy_bytes.len())]).unwrap_or("");
        assert!(hdr.contains(exp_npy_descr), "expected {exp_npy_descr} in npy header for {p}, got {hdr:?}");
    }

    // 4. R2: unsized /cog/bbox + dst_crs matches GDAL/TiTiler WarpedVRT dimensions exactly
    let sat_fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../benches/fixtures/satellite_utm10n_rgb_zstd_p3.tif");
    if sat_fixture.exists() {
        let sat = sat_fixture.to_string_lossy();
        for (q, exp_w, exp_h) in [
            ("", 976u32, 976u32),
            ("&dst_crs=EPSG:3857", 879, 1111),
            ("&dst_crs=EPSG:32610", 885, 1113),
        ] {
            let r = client
                .get(format!("{base}/cog/bbox/-122.8,37.6,-122.5,37.9.png?url={sat}&max_size=0{q}"))
                .send()
                .await
                .unwrap();
            assert_eq!(r.status(), 200);
            let bytes = r.bytes().await.unwrap();
            assert_eq!(&bytes[0..8], b"\x89PNG\r\n\x1a\n");
            let w = u32::from_be_bytes(bytes[16..20].try_into().unwrap());
            let h = u32::from_be_bytes(bytes[20..24].try_into().unwrap());
            assert_eq!(
                (w, h),
                (exp_w, exp_h),
                "R2 bbox size mismatch for query '{q}'"
            );
        }
    }

    srv_task.abort();
    let _ = std::fs::remove_dir_all(&tmp);
}

#[tokio::test]
async fn sweep43_oo1_exact_wide_dtype_roundtrip() {
    let tmp = std::env::temp_dir().join(format!("cogzilla_sweep43_oo1_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();

    // Write a minimal 8x8 tiled EPSG:4326 GeoTIFF (bounds [0, 0, 8, 8], 1 deg/px).
    let write_8x8 = |path: &std::path::Path, bps: u16, sample_fmt: u16, payload: &[u8]| {
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(b"II");
        buf.extend_from_slice(&42u16.to_le_bytes());
        buf.extend_from_slice(&8u32.to_le_bytes());
        let after_ifd = 8 + 2 + 15 * 12 + 4;
        buf.resize(after_ifd, 0);
        let scale_off = buf.len() as u32;
        for v in [1.0f64, 1.0, 0.0] {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        let tie_off = buf.len() as u32;
        for v in [0.0f64, 0.0, 0.0, 0.0, 8.0, 0.0] {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        let gk_off = buf.len() as u32;
        let gks: [u16; 16] = [
            1, 1, 0, 3,
            1024, 0, 1, 2,
            1025, 0, 1, 1,
            2048, 0, 1, 4326,
        ];
        for k in gks {
            buf.extend_from_slice(&k.to_le_bytes());
        }
        let tile_off = buf.len() as u32;
        buf.extend_from_slice(payload);

        let mut ifd: Vec<u8> = Vec::new();
        let push = |b: &mut Vec<u8>, tag: u16, ty: u16, count: u32, val: u32| {
            b.extend_from_slice(&tag.to_le_bytes());
            b.extend_from_slice(&ty.to_le_bytes());
            b.extend_from_slice(&count.to_le_bytes());
            b.extend_from_slice(&val.to_le_bytes());
        };
        ifd.extend_from_slice(&15u16.to_le_bytes());
        push(&mut ifd, 256, 4, 1, 8);
        push(&mut ifd, 257, 4, 1, 8);
        push(&mut ifd, 258, 3, 1, bps as u32);
        push(&mut ifd, 259, 3, 1, 1);
        push(&mut ifd, 262, 3, 1, 1);
        push(&mut ifd, 277, 3, 1, 1);
        push(&mut ifd, 284, 3, 1, 1);
        push(&mut ifd, 322, 4, 1, 8);
        push(&mut ifd, 323, 4, 1, 8);
        push(&mut ifd, 324, 4, 1, tile_off);
        push(&mut ifd, 325, 4, 1, payload.len() as u32);
        push(&mut ifd, 339, 3, 1, sample_fmt as u32);
        push(&mut ifd, 33550, 12, 3, scale_off);
        push(&mut ifd, 33922, 12, 6, tie_off);
        push(&mut ifd, 34735, 3, 16, gk_off);
        ifd.extend_from_slice(&0u32.to_le_bytes());
        buf[8..8 + ifd.len()].copy_from_slice(&ifd);
        std::fs::write(path, &buf).unwrap();
    };

    // 1. uint32: [16777217, 2147483647, 4294967295, ...]
    let mut u32_vals = [16_777_217u32; 64];
    u32_vals[1] = 2_147_483_647;
    u32_vals[2] = u32::MAX;
    let u32_bytes: Vec<u8> = u32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    let u32_path = tmp.join("u32_big.tif");
    write_8x8(&u32_path, 32, 1, &u32_bytes);

    // 2. int32: [16777217, 16777219, -2147483648, ...]
    let mut i32_vals = [16_777_217i32; 64];
    i32_vals[1] = 16_777_219;
    i32_vals[2] = i32::MIN;
    let i32_bytes: Vec<u8> = i32_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    let i32_path = tmp.join("i32_big.tif");
    write_8x8(&i32_path, 32, 2, &i32_bytes);

    // 3. int64: [1000000000001, 2^53 - 1, ...]
    let mut i64_vals = [1_000_000_000_001i64; 64];
    i64_vals[1] = (1i64 << 53) - 1;
    let i64_bytes: Vec<u8> = i64_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    let i64_path = tmp.join("i64_big.tif");
    write_8x8(&i64_path, 64, 2, &i64_bytes);

    // 4. float64: [12345678901.234568, -1.7976931348623157e308, ...]
    let mut f64_vals = [12_345_678_901.234_568_f64; 64];
    f64_vals[1] = f64::MIN;
    let f64_bytes: Vec<u8> = f64_vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    let f64_path = tmp.join("f64_big.tif");
    write_8x8(&f64_path, 64, 3, &f64_bytes);

    let engine = Arc::new(TileEngine::new(64 * 1024 * 1024));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let base = format!("http://{addr}");
    let srv_engine = engine.clone();
    let srv_task = tokio::spawn(async move {
        let _ = cogzilla_server::http::serve_listener(srv_engine, listener).await;
    });
    let client = reqwest::Client::new();

    fn npy_body(b: &[u8]) -> &[u8] {
        let hlen = u16::from_le_bytes([b[8], b[9]]) as usize;
        &b[10 + hlen..]
    }

    // uint32: point, statistics max == 4294967295, bbox .npy and .tif exact
    let p_u32 = u32_path.to_string_lossy();
    let pt: serde_json::Value = serde_json::from_str(
        &client.get(format!("{base}/cog/point/0.5,7.5?url={p_u32}")).send().await.unwrap().text().await.unwrap()
    ).unwrap();
    assert_eq!(pt["values"][0].as_f64().unwrap(), 16_777_217.0);

    let st: serde_json::Value = serde_json::from_str(
        &client.get(format!("{base}/cog/statistics?url={p_u32}")).send().await.unwrap().text().await.unwrap()
    ).unwrap();
    assert_eq!(st["b1"]["max"].as_f64().unwrap(), 4_294_967_295.0);

    let npy = client.get(format!("{base}/cog/bbox/0,0,8,8/8x8.npy?url={p_u32}&resampling=nearest")).send().await.unwrap().bytes().await.unwrap();
    let body = npy_body(&npy);
    assert_eq!(u32::from_le_bytes(body[0..4].try_into().unwrap()), 16_777_217);
    assert_eq!(u32::from_le_bytes(body[4..8].try_into().unwrap()), 2_147_483_647);
    assert_eq!(u32::from_le_bytes(body[8..12].try_into().unwrap()), u32::MAX);

    let tif = client.get(format!("{base}/cog/bbox/0,0,8,8/8x8.tif?url={p_u32}&resampling=nearest&return_mask=false")).send().await.unwrap().bytes().await.unwrap();
    let parsed = cogzilla_tiff::parse_cog(&tif).unwrap();
    assert_eq!(parsed.full_res().sample_type, cogzilla_raster::SampleType::Uint32);
    let dec = cogzilla_codec::decode_tile(
        &tif[parsed.full_res().tile_offsets[0] as usize..parsed.full_res().tile_offsets[0] as usize + parsed.full_res().tile_byte_counts[0] as usize],
        parsed.full_res(),
        parsed.byte_order,
        None,
    ).unwrap();
    assert_eq!(dec.bands[0].data.get_f64(0), 16_777_217.0);
    assert_eq!(dec.bands[0].data.get_f64(1), 2_147_483_647.0);
    assert_eq!(dec.bands[0].data.get_f64(2), 4_294_967_295.0);

    // int32: exact 16777217, 16777219, i32::MIN
    let p_i32 = i32_path.to_string_lossy();
    let npy = client.get(format!("{base}/cog/bbox/0,0,8,8/8x8.npy?url={p_i32}&resampling=nearest")).send().await.unwrap().bytes().await.unwrap();
    let body = npy_body(&npy);
    assert_eq!(i32::from_le_bytes(body[0..4].try_into().unwrap()), 16_777_217);
    assert_eq!(i32::from_le_bytes(body[4..8].try_into().unwrap()), 16_777_219);
    assert_eq!(i32::from_le_bytes(body[8..12].try_into().unwrap()), i32::MIN);

    // int64: exact 1000000000001 and 2^53 - 1
    let p_i64 = i64_path.to_string_lossy();
    let npy = client.get(format!("{base}/cog/bbox/0,0,8,8/8x8.npy?url={p_i64}&resampling=nearest")).send().await.unwrap().bytes().await.unwrap();
    let body = npy_body(&npy);
    assert_eq!(i64::from_le_bytes(body[0..8].try_into().unwrap()), 1_000_000_000_001);
    assert_eq!(i64::from_le_bytes(body[8..16].try_into().unwrap()), (1i64 << 53) - 1);

    // float64: exact 17-sig-digit value and f64::MIN (-1.7976931348623157e308)
    let p_f64 = f64_path.to_string_lossy();
    let st_f64: serde_json::Value = serde_json::from_str(
        &client.get(format!("{base}/cog/statistics?url={p_f64}")).send().await.unwrap().text().await.unwrap()
    ).unwrap();
    assert_eq!(st_f64["b1"]["min"].as_f64().unwrap(), f64::MIN);

    let npy = client.get(format!("{base}/cog/bbox/0,0,8,8/8x8.npy?url={p_f64}&resampling=nearest")).send().await.unwrap().bytes().await.unwrap();
    let body = npy_body(&npy);
    assert_eq!(f64::from_le_bytes(body[0..8].try_into().unwrap()), 12_345_678_901.234_568_f64);
    assert_eq!(f64::from_le_bytes(body[8..16].try_into().unwrap()), f64::MIN);

    // Audit #94 regression checks:
    // 1. Default return_mask=true on wide-dtype GeoTIFF must NOT invert masked pixels to u32::MAX.
    let tif_masked = client
        .get(format!("{base}/cog/bbox/0,0,16,8/16x8.tif?url={p_u32}&resampling=nearest"))
        .send()
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap();
    let idx_m = cogzilla_tiff::parse_cog(&tif_masked).unwrap();
    assert_eq!(idx_m.full_res().samples_per_pixel, 2);
    assert_eq!(idx_m.full_res().sample_type, cogzilla_raster::SampleType::Uint32);
    let mask_off = idx_m.full_res().tile_offsets[1] as usize;
    let mask_len = idx_m.full_res().tile_byte_counts[1] as usize;
    let mask_slice = &tif_masked[mask_off..mask_off + mask_len];
    // Pixel (0, 0) is inside [0,8]x[0,8] -> u32::MAX; pixel (12, 0) is outside -> 0 (u32::MIN).
    assert_eq!(u32::from_le_bytes(mask_slice[0..4].try_into().unwrap()), u32::MAX);
    assert_eq!(u32::from_le_bytes(mask_slice[12 * 4..12 * 4 + 4].try_into().unwrap()), 0);

    // 2. Styled .npy (rescale / colormap_name) on wide-dtype COG must return painted |u1 bytes, not raw <u4.
    let styled_npy = client
        .get(format!("{base}/cog/preview.npy?url={p_u32}&rescale=0,4294967295"))
        .send()
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap();
    let hlen = u16::from_le_bytes([styled_npy[8], styled_npy[9]]) as usize;
    let hdr = std::str::from_utf8(&styled_npy[10..10 + hlen]).unwrap();
    assert!(hdr.contains("'|u1'"), "expected painted |u1 npy header, got: {hdr}");

    // 3. Multi-COG wide-dtype .tif must produce a valid 2-band uint32 GeoTIFF (not bit-garbled f64).
    let multi_tif = client
        .get(format!("{base}/cog/bbox/0,0,8,8/8x8.tif?url={p_u32}&url={p_u32}&return_mask=false&resampling=nearest"))
        .send()
        .await
        .unwrap();
    assert_eq!(multi_tif.status(), 200);
    let multi_bytes = multi_tif.bytes().await.unwrap();
    let idx_multi = cogzilla_tiff::parse_cog(&multi_bytes).unwrap();
    assert_eq!(idx_multi.full_res().samples_per_pixel, 2);
    assert_eq!(idx_multi.full_res().sample_type, cogzilla_raster::SampleType::Uint32);

    // 4. Extreme f64 statistics (f64::MIN + f64::MAX) must produce finite mean, std, and histogram edges (no JSON null).
    let mut f64_ext_raw = Vec::with_capacity(64 * 8);
    for i in 0..64 {
        let v = if i % 2 == 0 { f64::MIN } else { f64::MAX };
        f64_ext_raw.extend_from_slice(&v.to_le_bytes());
    }
    let f64_ext_path = tmp.join("f64_ext.tif");
    write_8x8(&f64_ext_path, 64, 3, &f64_ext_raw);
    let p_f64_ext = f64_ext_path.to_string_lossy();
    let st_ext: serde_json::Value = serde_json::from_str(
        &client.get(format!("{base}/cog/statistics?url={p_f64_ext}")).send().await.unwrap().text().await.unwrap()
    ).unwrap();
    assert!(st_ext["b1"]["mean"].as_f64().is_some_and(f64::is_finite), "mean must be finite: {st_ext}");
    assert!(st_ext["b1"]["std"].as_f64().is_some_and(f64::is_finite), "std must be finite: {st_ext}");
    let edges = st_ext["b1"]["histogram"][1].as_array().unwrap();
    assert!(edges.iter().all(|e| e.as_f64().is_some_and(f64::is_finite)), "all histogram edges must be finite: {edges:?}");

    srv_task.abort();
    let _ = std::fs::remove_dir_all(&tmp);
}

/// Statistics at a size between a raster's levels are those of its preview read with the
/// resampling asked for, as rio-tiler takes them; a uint16 band's resampled values are whole
/// numbers, as GDAL rounds them into the band's type, in the statistics and the array alike.
#[tokio::test]
async fn statistics_follow_the_resampling_of_the_preview() {
    let path = fixture("landsat_c2_B4.tif");
    if !path.exists() {
        eprintln!("fixtures not generated; skipping");
        return;
    }
    let url = path.canonicalize().unwrap().to_string_lossy().to_string();
    let engine = Arc::new(TileEngine::new(128 * 1024 * 1024));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = serve_listener(engine, listener).await;
    });
    let base = format!("http://{addr}");
    let client = reqwest::Client::new();
    let get = |q: String| {
        let client = client.clone();
        async move {
            let resp = client.get(q).send().await.unwrap();
            assert_eq!(resp.status().as_u16(), 200);
            resp.bytes().await.unwrap()
        }
    };
    let mut means = Vec::new();
    for resampling in ["nearest", "bilinear", "average"] {
        let q = format!("url={url}&max_size=300&resampling={resampling}");
        let stats: serde_json::Value = serde_json::from_slice(&get(format!("{base}/cog/statistics?{q}")).await).unwrap();
        let npy = get(format!("{base}/cog/preview.npy?{q}")).await;
        let values = decode_npy_f32(&npy);
        let (data, mask) = values.split_at(values.len() / 2);
        let valid: Vec<f64> = data.iter().zip(mask).filter(|(_, m)| **m > 0.0).map(|(v, _)| f64::from(*v)).collect();
        assert!(valid.iter().all(|v| v.fract() == 0.0), "{resampling}: a uint16 read is whole");
        let mean = valid.iter().sum::<f64>() / valid.len() as f64;
        let b1 = &stats["b1"];
        assert_eq!(b1["count"].as_f64().unwrap(), valid.len() as f64, "{resampling}");
        assert!((b1["mean"].as_f64().unwrap() - mean).abs() < 1e-6 * mean, "{resampling}: {} vs {mean}", b1["mean"]);
        means.push(mean);
    }
    assert!(means[0] != means[1] && means[1] != means[2], "the resampling changes the numbers: {means:?}");
}

/// `?nodata=` replaces the raster's own nodata, as GDAL's override (and so TiTiler's) does:
/// pixels equal to the file's nodata become data and pixels equal to the request's are masked.
#[tokio::test]
async fn nodata_query_replaces_the_rasters_own() {
    let tmp = std::env::temp_dir().join(format!("cogzilla_nodata_override_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    // GDAL_NODATA=0, the left half 0 and the right half 7 in both bands.
    let path = tmp.join("u16_nodata0.tif");
    write_synthetic_u16_2band_4326_geotiff(&path, 64, 64, [10.0, 40.0, 10.64, 40.64], 7, 7, Some("left"));
    let url = path.to_string_lossy().to_string();
    let engine = Arc::new(TileEngine::new(64 * 1024 * 1024));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = serve_listener(engine, listener).await;
    });
    let base = format!("http://{addr}");
    let client = reqwest::Client::new();
    let stats = |q: &'static str| {
        let (client, base, url) = (client.clone(), base.clone(), url.clone());
        async move {
            let resp = client.get(format!("{base}/cog/statistics?url={url}{q}")).send().await.unwrap();
            assert_eq!(resp.status().as_u16(), 200, "{q}");
            let v: serde_json::Value = serde_json::from_slice(&resp.bytes().await.unwrap()).unwrap();
            (v["b1"]["count"].as_f64().unwrap(), v["b1"]["min"].as_f64().unwrap(), v["b1"]["max"].as_f64().unwrap())
        }
    };
    let half = (64 * 32) as f64;
    assert_eq!(stats("").await, (half, 7.0, 7.0));
    assert_eq!(stats("&nodata=0").await, (half, 7.0, 7.0), "the file's own value changes nothing");
    assert_eq!(stats("&nodata=7").await, (half, 0.0, 0.0), "the file's 0 is data, the request's 7 is not");
    assert_eq!(stats("&nodata=5").await, (2.0 * half, 0.0, 7.0), "nothing equals 5: every pixel is data");
    // The preview's mask says the same, and the file's own read is not changed by the override's.
    for (q, valid) in [("&nodata=7", half), ("", half), ("&nodata=5", 2.0 * half)] {
        let npy = client.get(format!("{base}/cog/preview.npy?url={url}&bidx=1{q}")).send().await.unwrap().bytes().await.unwrap();
        let values = decode_npy_f32(&npy);
        let (data, mask) = values.split_at(values.len() / 2);
        assert_eq!(mask.iter().filter(|m| **m > 0.0).count() as f64, valid, "{q}");
        let want = if q == "&nodata=7" { 0.0 } else { 7.0 };
        assert!(data.iter().zip(mask).any(|(v, m)| *m > 0.0 && *v == want), "{q}");
    }
    let _ = std::fs::remove_dir_all(&tmp);
}

/// A percent-encoded query key is read as the key it spells (`max%5Fsize` is `max_size`), as
/// Starlette reads it; and one band of a mosaic is a grey picture, as TiTiler writes it.
#[tokio::test]
async fn encoded_query_keys_and_grey_mosaic_tiles() {
    let (b4, b3) = (fixture("landsat_c2_B4.tif"), fixture("landsat_c2_B3.tif"));
    if !b4.exists() || !b3.exists() {
        eprintln!("fixtures not generated; skipping");
        return;
    }
    let (b4, b3) = (b4.canonicalize().unwrap(), b3.canonicalize().unwrap());
    let (b4, b3) = (b4.to_string_lossy(), b3.to_string_lossy());
    let engine = Arc::new(TileEngine::new(128 * 1024 * 1024));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = serve_listener(engine, listener).await;
    });
    let base = format!("http://{addr}");
    let client = reqwest::Client::new();
    let get = |u: String| {
        let client = client.clone();
        async move {
            let resp = client.get(&u).send().await.unwrap();
            assert_eq!(resp.status().as_u16(), 200, "{u}");
            resp.bytes().await.unwrap()
        }
    };
    let png_size = |b: &[u8]| (u32::from_be_bytes(b[16..20].try_into().unwrap()), u32::from_be_bytes(b[20..24].try_into().unwrap()));
    for key in ["max_size", "max%5Fsize", "max%5fsize"] {
        let png = get(format!("{base}/cog/preview.png?url={b4}&{key}=64&rescale=0,30000")).await;
        assert_eq!(png_size(&png).0.max(png_size(&png).1), 64, "{key}");
    }
    let tile = |q: &str| format!("{base}/mosaic/tiles/WebMercatorQuad/12/653/1583.png?url={b4}&url={b3}&rescale=0,30000{q}");
    // PNG colour type: 0 grey, 4 grey and alpha; 2 and 6 colour, 3 a palette.
    for q in ["", "&resampling=bilinear"] {
        let png = get(tile(q)).await;
        assert!(matches!(png[25], 0 | 4), "one band in grey{q}: colour type {}", png[25]);
    }
    let mapped = get(tile("&colormap_name=viridis")).await;
    assert!(matches!(mapped[25], 2 | 3 | 6), "a colour map is colour: {}", mapped[25]);
    let jpg = get(tile("").replace(".png", ".jpg")).await;
    // The frame header's component count: one for grey.
    let sof = jpg.windows(2).position(|w| w[0] == 0xFF && matches!(w[1], 0xC0..=0xC2)).unwrap();
    assert_eq!(jpg[sof + 9], 1, "a grey JPEG");
}

#[tokio::test]
async fn ogc_wms_getcapabilities_and_getmap_end_to_end() {
    let sat = fixture("satellite_utm10n_rgb_zstd_p3.tif");
    if !sat.exists() {
        eprintln!("fixtures not generated; skipping");
        return;
    }
    let sat = sat.canonicalize().unwrap();
    let sat_url = sat.to_string_lossy();
    let engine = Arc::new(TileEngine::new(64 * 1024 * 1024));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = serve_listener(engine, listener).await;
    });
    let base = format!("http://{addr}");
    let client = reqwest::Client::new();

    // 1. GetCapabilities XML (uppercase WMS keys)
    let cap_resp = client
        .get(format!("{base}/cog/wms?url={sat_url}&SERVICE=WMS&REQUEST=GetCapabilities&VERSION=1.3.0"))
        .send()
        .await
        .unwrap();
    assert_eq!(cap_resp.status().as_u16(), 200);
    let cap_xml = cap_resp.text().await.unwrap();
    assert!(cap_xml.contains("<WMS_Capabilities version=\"1.3.0\""));
    assert!(!cap_xml.contains("REQUEST=GetCapabilities"));

    // 2. GetMap WMS 1.3.0 with uppercase keys & EPSG:4326 (lat,lon axis order) vs CRS:84 (lon,lat)
    let png_4326 = client
        .get(format!(
            "{base}/cog/wms?url={sat_url}&SERVICE=WMS&REQUEST=GetMap&VERSION=1.3.0&CRS=EPSG:4326&BBOX=37.6,-122.8,37.8,-122.5&WIDTH=64&HEIGHT=64&FORMAT=image/png"
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(png_4326.status().as_u16(), 200);
    let b_4326 = png_4326.bytes().await.unwrap();
    assert_eq!(&b_4326[0..8], b"\x89PNG\r\n\x1a\n");

    let png_crs84 = client
        .get(format!(
            "{base}/cog/wms?url={sat_url}&SERVICE=WMS&REQUEST=GetMap&VERSION=1.3.0&CRS=CRS:84&BBOX=-122.8,37.6,-122.5,37.8&WIDTH=64&HEIGHT=64&FORMAT=image/png"
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(png_crs84.status().as_u16(), 200);
    let b_crs84 = png_crs84.bytes().await.unwrap();
    assert_eq!(b_4326, b_crs84, "WMS 1.3.0 EPSG:4326 (lat,lon) and CRS:84 (lon,lat) must produce identical PNGs");

    // 3. GetMap WMS 1.1.1 SRS=EPSG:4326 (lon,lat)
    let png_111 = client
        .get(format!(
            "{base}/cog/wms?url={sat_url}&SERVICE=WMS&REQUEST=GetMap&VERSION=1.1.1&SRS=EPSG:4326&BBOX=-122.8,37.6,-122.5,37.8&WIDTH=64&HEIGHT=64&FORMAT=image/png"
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(png_111.status().as_u16(), 200);
    assert_eq!(png_111.bytes().await.unwrap(), b_crs84, "WMS 1.1.1 SRS=EPSG:4326 must match CRS:84");

    // 4. Invalid TRANSPARENT & missing BBOX return 400
    let bad_t = client
        .get(format!(
            "{base}/cog/wms?url={sat_url}&REQUEST=GetMap&CRS=CRS:84&BBOX=-122.8,37.6,-122.5,37.8&WIDTH=64&HEIGHT=64&TRANSPARENT=maybe"
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(bad_t.status().as_u16(), 400);
}

/// The shape and values of a `.npy` in any of the types an array is written in.
fn decode_npy_any(bytes: &[u8]) -> (Vec<usize>, Vec<f64>) {
    assert_eq!(&bytes[..6], b"\x93NUMPY", "not a .npy");
    let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let header = std::str::from_utf8(&bytes[10..10 + header_len]).unwrap();
    let descr = header.split("'descr': '").nth(1).unwrap().split('\'').next().unwrap();
    let shape: Vec<usize> = header.split("'shape': (").nth(1).unwrap().split(')').next().unwrap()
        .split(',').filter_map(|d| d.trim().parse().ok()).collect();
    let data = &bytes[10 + header_len..];
    let values = match descr {
        "|u1" => data.iter().map(|&v| f64::from(v)).collect(),
        "<u2" => data.chunks_exact(2).map(|c| f64::from(u16::from_le_bytes([c[0], c[1]]))).collect(),
        "<i4" => data.chunks_exact(4).map(|c| f64::from(i32::from_le_bytes([c[0], c[1], c[2], c[3]]))).collect(),
        "<f4" => data.chunks_exact(4).map(|c| f64::from(f32::from_le_bytes([c[0], c[1], c[2], c[3]]))).collect(),
        "<f8" => data.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect(),
        other => panic!("unexpected dtype {other}"),
    };
    (shape, values)
}

/// Several assets of more than one band are stacked as rio-tiler stacks a STAC item: every band
/// of each, one after another, so `bN` is the Nth band of the stack; an asset's own band math
/// is at its place in the stack, and a request expression is over the results.
#[tokio::test]
async fn stac_stacks_every_band_of_several_assets() {
    let tmp = std::env::temp_dir().join(format!("cogzilla_stac_stack_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let bounds = [10.0, 40.0, 10.64, 40.64];
    let (a, b) = (tmp.join("a.tif"), tmp.join("b.tif"));
    write_synthetic_u16_2band_4326_geotiff(&a, 64, 64, bounds, 3, 5, None);
    write_synthetic_u16_2band_4326_geotiff(&b, 64, 64, bounds, 11, 13, None);
    let asset = |p: &PathBuf| serde_json::json!({"href": p.to_string_lossy(), "type": "image/tiff; application=geotiff", "roles": ["data"]});
    let item = serde_json::json!({
        "type": "Feature", "stac_version": "1.0.0", "id": "two", "bbox": bounds,
        "geometry": {"type": "Polygon", "coordinates": [[[10.0, 40.0], [10.64, 40.0], [10.64, 40.64], [10.0, 40.64], [10.0, 40.0]]]},
        "properties": {"datetime": "2024-01-01T00:00:00Z"},
        "assets": {"a": asset(&a), "b": asset(&b), "b2": asset(&b), "SR-B5": asset(&a)},
    });
    let item_path = tmp.join("item.json");
    std::fs::write(&item_path, item.to_string()).unwrap();
    let url = item_path.to_string_lossy().to_string();
    let engine = Arc::new(TileEngine::new(64 * 1024 * 1024));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = serve_listener(engine, listener).await;
    });
    let base = format!("http://{addr}");
    let client = reqwest::Client::new();
    let pixel = |q: &str| {
        let q = q.to_string();
        let (client, base, url) = (client.clone(), base.clone(), url.clone());
        async move {
            let resp = client.get(format!("{base}/stac/preview.npy?url={url}&max_size=16{q}")).send().await.unwrap();
            assert_eq!(resp.status().as_u16(), 200, "{q}");
            let (shape, values) = decode_npy_any(&resp.bytes().await.unwrap());
            let plane = shape[1] * shape[2];
            // The data bands at the middle pixel, without the mask.
            (0..shape[0] - 1).map(|k| values[k * plane + plane / 2]).collect::<Vec<f64>>()
        }
    };
    assert_eq!(pixel("&assets=a&assets=b").await, [3.0, 5.0, 11.0, 13.0]);
    assert_eq!(pixel("&assets=a&assets=b&expression=b2;b3*2;b4").await, [5.0, 22.0, 13.0]);
    assert_eq!(pixel("&assets=a%7Cbidx%3D2&assets=b").await, [5.0, 11.0, 13.0]);
    assert_eq!(pixel("&assets=a%7Cexpression%3Db1%2Bb2&assets=b").await, [8.0, 11.0, 13.0]);
    assert_eq!(pixel("&assets=a%7Cexpression%3Db1%2Bb2&assets=b&expression=b1*b3").await, [104.0]);
    // A terrain algorithm over band math reads its first output, as rio-tiler does: terrarium
    // encodes 32768 + the value as red, green and blue.
    assert_eq!(pixel("&assets=a&algorithm=terrarium").await, [128.0, 3.0, 0.0]);
    assert_eq!(pixel("&assets=a&expression=b2*2&algorithm=terrarium").await, [128.0, 10.0, 0.0]);
    assert_eq!(pixel("&assets=a&assets=b&expression=b3-b1;b1&algorithm=terrarium").await, [128.0, 8.0, 0.0]);
    // Its statistics read the same stack, asset names included (terrarium's green is the value).
    // An asset's own band math is read there too: `a|expression=b1+b2` is 3 + 5.
    for q in [
        "&assets=a&assets=b&expression=b3-b1",
        "&assets=a&assets=b&expression=b_b1-a_b1",
        "&assets=a%7Cexpression%3Db1%2Bb2",
    ] {
        let resp = client
            .get(format!("{base}/stac/statistics?url={url}&max_size=16&algorithm=terrarium{q}"))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status().as_u16(), 200, "{q}");
        let stats: serde_json::Value = serde_json::from_slice(&resp.bytes().await.unwrap()).unwrap();
        assert_eq!(stats["b2"]["mean"].as_f64(), Some(8.0), "{q}: {stats}");
    }
    // An expression beside an algorithm that is not a terrain one is left to that algorithm.
    let resp = client
        .get(format!("{base}/stac/preview.png?url={url}&max_size=16&assets=a&expression=ndvi&algorithm=cast"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 200);
    // ... and a spectral preset beside a terrain algorithm reads the band, as before.
    assert_eq!(pixel("&assets=a&expression=ndvi&algorithm=terrarium").await, [128.0, 3.0, 0.0]);
    let resp = client
        .get(format!("{base}/stac/statistics?url={url}&max_size=16&assets=a&expression=ndvi&algorithm=terrarium"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 200);
    let stats: serde_json::Value = serde_json::from_slice(&resp.bytes().await.unwrap()).unwrap();
    assert_eq!(stats["b2"]["mean"].as_f64(), Some(3.0), "{stats}");
    // An expression that reads no band has nowhere to read elevations.
    let resp = client
        .get(format!("{base}/stac/preview.png?url={url}&assets=a&expression=42&algorithm=hillshade"))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 400);
    // Band math over assets on different grids is refused, not sent round the mixed-grid
    // fallback again.
    let coarse = tmp.join("coarse.tif");
    write_synthetic_u16_2band_4326_geotiff(&coarse, 32, 32, bounds, 7, 9, None);
    let mixed = serde_json::json!({
        "type": "Feature", "stac_version": "1.0.0", "id": "mixed", "bbox": bounds,
        "geometry": item["geometry"].clone(),
        "properties": {"datetime": "2024-01-01T00:00:00Z"},
        "assets": {"a": asset(&a), "c": asset(&coarse)},
    });
    let mixed_path = tmp.join("mixed.json");
    std::fs::write(&mixed_path, mixed.to_string()).unwrap();
    let resp = client
        .get(format!(
            "{base}/stac/preview.png?url={}&max_size=16&assets=a&assets=c&asset_bidx=a%7C1&asset_bidx=c%7C1&expression=b1%2Bb2&algorithm=slope",
            mixed_path.to_string_lossy()
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status().as_u16(), 400);
    assert!(String::from_utf8_lossy(&resp.bytes().await.unwrap()).contains("one grid"));
    // An asset's name in the request expression is its first band.
    assert_eq!(pixel("&assets=a&assets=b&expression=b_b1-a_b1").await, [8.0]);
    // ... and `bN` beside it is still the Nth band of the stack.
    assert_eq!(pixel("&assets=a&assets=b&expression=b_b1%2Bb2").await, [16.0]);
    // An asset named as a band would be is read as the asset, as before stacking.
    assert_eq!(pixel("&assets=a&assets=b2&expression=b2").await, [11.0]);
    // An asset asked for twice keeps what each time asked.
    assert_eq!(pixel("&assets=a&assets=a%7Cexpression%3Db1*2").await, [3.0, 5.0, 6.0]);
    assert_eq!(pixel("&assets=a%7Cexpression%3Db1*2&assets=a").await, [6.0, 3.0, 5.0]);
    assert_eq!(pixel("&assets=a%7Cbidx%3D2&assets=a%7Cbidx%3D1").await, [5.0, 3.0]);
    // `asset_bidx` for one asset beside another's own `|bidx=`.
    assert_eq!(pixel("&assets=a%7Cbidx%3D2&assets=b&asset_bidx=b%7C2").await, [5.0, 13.0]);
    // `:all:` reads every asset of the item, and an asset's band math named beside it goes to
    // that asset; its array is typed as numpy types that math (b1*2 over uint16 is int32).
    let all = client.get(format!("{base}/stac/preview.npy?url={url}&max_size=16&assets=:all:&assets=b%7Cexpression%3Db1*2")).send().await.unwrap();
    assert_eq!(all.status().as_u16(), 200);
    let all = all.bytes().await.unwrap();
    assert!(String::from_utf8_lossy(&all[..64]).contains("'<i4'"), "{}", String::from_utf8_lossy(&all[..80]));
    // Statistics of band math read the same stack.
    let means = |q: &str| {
        let (client, base, url) = (client.clone(), base.clone(), url.clone());
        let q = q.to_string();
        async move {
            let resp = client.get(format!("{base}/stac/statistics?url={url}&max_size=16{q}")).send().await.unwrap();
            assert_eq!(resp.status().as_u16(), 200, "{q}");
            let doc: serde_json::Value = serde_json::from_slice(&resp.bytes().await.unwrap()).unwrap();
            (1..=doc.as_object().unwrap().len())
                .map(|k| {
                    let band = &doc[format!("b{k}")];
                    (band["mean"].as_f64().unwrap(), band["description"].as_str().unwrap().to_string())
                })
                .collect::<Vec<(f64, String)>>()
        }
    };
    // Each result is described, as TiTiler describes it, in the names of the stacked bands.
    let described = |pairs: &[(f64, &str)]| pairs.iter().map(|&(m, d)| (m, d.to_string())).collect::<Vec<_>>();
    assert_eq!(means("&assets=a&assets=b&expression=b2;b3*2").await, described(&[(5.0, "a_b2"), (22.0, "b_b1*2")]));
    assert_eq!(
        means("&assets=a%7Cexpression%3Db1%2Bb2&assets=b&expression=b1*b3").await,
        described(&[(104.0, "a_b1+b2*b_b2")])
    );
    assert_eq!(means("&assets=a%7Cbidx%3D2&assets=b&expression=b1%2Bb3").await, described(&[(18.0, "a_b2+b_b2")]));
    assert_eq!(
        means("&assets=a&assets=b").await,
        described(&[(3.0, "a_b1"), (5.0, "a_b2"), (11.0, "b_b1"), (13.0, "b_b2")])
    );
    // One asset's chosen bands are read in their order by band math.
    assert_eq!(means("&assets=a%7Cbidx%3D2&expression=b1").await, described(&[(5.0, "a_b2")]));
    assert_eq!(
        means("&assets=a%7Cbidx%3D2,1&expression=b1;b2*2").await,
        described(&[(5.0, "a_b2"), (6.0, "a_b1*2")])
    );
    // Whatever top-level `bidx` came with it; and an asset's name reads its own band.
    assert_eq!(means("&assets=a%7Cbidx%3D2&bidx=1&expression=b1").await, described(&[(5.0, "a_b2")]));
    assert_eq!(means("&assets=a%7Cbidx%3D2,1&expression=a_b2%2Bb1").await, described(&[(10.0, "a_b2+a_b2")]));
    // A name is read whole first: a band token inside it, or a name like a band, is the name's.
    assert_eq!(means("&assets=SR-B5%7Cbidx%3D2,1&expression=SR-B5%2Bb1").await[0].0, 10.0);
    assert_eq!(means("&assets=b2%7Cbidx%3D2,1&expression=b2_b2%2Bb1").await[0].0, 26.0);
    let past = client.get(format!("{base}/stac/statistics?url={url}&assets=a%7Cbidx%3D2&expression=b2")).send().await.unwrap();
    assert_eq!(past.status().as_u16(), 400);
    // An asset named like a band (`b2`) stays as written.
    assert_eq!(means("&assets=a&assets=b2&expression=b2%2B1").await[0].1, "b2+1");
    // A point and the statistics of a posted feature read the same stack, and describe it so.
    let point = |q: &str| {
        let (client, base, url) = (client.clone(), base.clone(), url.clone());
        let q = q.to_string();
        async move {
            let resp = client.get(format!("{base}/stac/point/10.3,40.3?url={url}{q}")).send().await.unwrap();
            assert_eq!(resp.status().as_u16(), 200, "{q}");
            let doc: serde_json::Value = serde_json::from_slice(&resp.bytes().await.unwrap()).unwrap();
            let values = doc["values"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap());
            let descs = doc["band_descriptions"].as_array().unwrap().iter().map(|d| d.as_str().unwrap().to_string());
            values.zip(descs).collect::<Vec<(f64, String)>>()
        }
    };
    assert_eq!(point("&assets=a&assets=b&expression=b2;b3*2").await, described(&[(5.0, "a_b2"), (22.0, "b_b1*2")]));
    assert_eq!(
        point("&assets=a%7Cexpression%3Db1%2Bb2&assets=b&expression=b1*b3").await,
        described(&[(104.0, "a_b1+b2*b_b2")])
    );
    assert_eq!(point("&assets=a%7Cexpression%3Db1*2;b2&expression=b1%2Bb2").await, described(&[(11.0, "a_b1*2+a_b2")]));
    assert_eq!(point("&assets=a%7Cbidx%3D2&expression=b1").await, described(&[(5.0, "a_b2")]));
    // An asset's `{asset}_b{k}` is its band `k` whatever was picked; `bN` the Nth picked.
    assert_eq!(point("&assets=a%7Cbidx%3D2&expression=a_b2").await, described(&[(5.0, "a_b2")]));
    assert_eq!(point("&assets=a%7Cbidx%3D2,1&expression=a_b1;b1").await, described(&[(3.0, "a_b1"), (5.0, "a_b2")]));
    // ... and under `asset_as_band` the band picked is the asset's, named after it.
    assert_eq!(point("&assets=a%7Cbidx%3D2&asset_as_band=true&expression=b1").await, described(&[(5.0, "a")]));
    let feature = serde_json::json!({"type": "Feature", "properties": {}, "geometry": {"type": "Polygon",
        "coordinates": [[[10.1, 40.1], [10.5, 40.1], [10.5, 40.5], [10.1, 40.5], [10.1, 40.1]]]}});
    let posted = |q: &str| {
        let (client, base, url, feature) = (client.clone(), base.clone(), url.clone(), feature.clone());
        let q = q.to_string();
        async move {
            let resp = client
                .post(format!("{base}/stac/statistics?url={url}&max_size=16{q}"))
                .header("content-type", "application/json")
                .body(feature.to_string())
                .send()
                .await
                .unwrap();
            assert_eq!(resp.status().as_u16(), 200, "{q}");
            let doc: serde_json::Value = serde_json::from_slice(&resp.bytes().await.unwrap()).unwrap();
            let stats = &doc["properties"]["statistics"];
            (1..=stats.as_object().unwrap().len())
                .map(|k| {
                    let band = &stats[format!("b{k}")];
                    // Means weighted by each pixel's area, to a millionth.
                    let mean = (band["mean"].as_f64().unwrap() * 1e6).round() / 1e6;
                    (mean, band["description"].as_str().unwrap().to_string())
                })
                .collect::<Vec<(f64, String)>>()
        }
    };
    assert_eq!(posted("&assets=a&assets=b&expression=b2;b3*2").await, described(&[(5.0, "a_b2"), (22.0, "b_b1*2")]));
    assert_eq!(
        posted("&assets=a%7Cexpression%3Db1%2Bb2&assets=b&expression=b1*b3").await,
        described(&[(104.0, "a_b1+b2*b_b2")])
    );
    // ... the same when an output size makes it a picture first.
    assert_eq!(
        posted("&assets=a&assets=b&expression=b2;b3*2&width=8&height=8").await,
        described(&[(5.0, "a_b2"), (22.0, "b_b1*2")])
    );
    // Sixteen bands are read for band math, but an answer of them all holds at most fifteen.
    let sixteen = "&assets=a".repeat(8);
    let too_many = client.get(format!("{base}/stac/preview.npy?url={url}{sixteen}")).send().await.unwrap();
    assert_eq!(too_many.status().as_u16(), 400, "sixteen bands with no band math");
    assert_eq!(pixel(&format!("{sixteen}&expression=b1%2Bb16")).await, [8.0]);
    assert_eq!(pixel(&format!("{sixteen}&bidx=16")).await, [5.0]);
    // Under `asset_as_band` each asset gives one band, however it is chosen.
    assert_eq!(pixel("&assets=a%7Cbidx%3D1&assets=b%7Cexpression%3Db2&asset_as_band=true").await, [3.0, 13.0]);
    for q in ["&assets=a&assets=b", "&assets=a%7Cexpression%3Db1;b2&assets=b%7Cbidx%3D1", "&assets=a%7Cbidx%3D1,2&assets=b%7Cbidx%3D1", "&assets=a"] {
        let refused = client.get(format!("{base}/stac/preview.npy?url={url}{q}&asset_as_band=true")).send().await.unwrap();
        assert_eq!(refused.status().as_u16(), 400, "asset_as_band over an asset of two bands: {q}");
    }
    let _ = std::fs::remove_dir_all(&tmp);
}

/// An algorithm with band math works over the expression's outputs, as rio-tiler applies it
/// after the read: `floor(b1 * 101 / 2)` is 151 where the band is 3, not `floor(3)`, and the
/// array is the algorithm's type.
#[tokio::test]
async fn an_algorithm_works_over_band_math_results() {
    let tmp = std::env::temp_dir().join(format!("cogzilla_alg_over_math_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let cog = tmp.join("ab.tif");
    write_synthetic_u16_2band_4326_geotiff(&cog, 64, 64, [10.0, 40.0, 10.64, 40.64], 3, 5, None);
    let engine = Arc::new(TileEngine::new(64 * 1024 * 1024));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = serve_listener(engine, listener).await;
    });
    let base = format!("http://{addr}/cog");
    let url = cog.to_string_lossy().to_string();
    let client = reqwest::Client::new();
    let get = |path: String| {
        let client = client.clone();
        async move {
            let resp = client.get(path).send().await.unwrap();
            assert_eq!(resp.status().as_u16(), 200);
            resp.bytes().await.unwrap()
        }
    };
    let middle = |bytes: &[u8]| {
        let (shape, values) = decode_npy_any(bytes);
        let plane = shape[1] * shape[2];
        (0..shape[0] - 1).map(|k| values[k * plane + plane / 2]).collect::<Vec<f64>>()
    };
    let npy = get(format!("{base}/preview.npy?url={url}&max_size=16&expression=b1*101/2;b2*100&algorithm=floor")).await;
    assert!(String::from_utf8_lossy(&npy[..64]).contains("'|u1'"), "floor writes bytes");
    // floor(151.5) and 500 wrapped into a byte, as numpy's astype wraps.
    assert_eq!(middle(&npy), [151.0, 244.0]);
    let max = get(format!("{base}/preview.npy?url={url}&max_size=16&expression=b1/2;b2&algorithm=max")).await;
    assert_eq!(middle(&max), [5.0]);
    let stats: serde_json::Value = serde_json::from_slice(
        &get(format!("{base}/statistics?url={url}&expression=b1;b2*3&algorithm=normalizedIndex")).await,
    )
    .unwrap();
    // (15 - 3) / (15 + 3), described over the expression's outputs.
    assert!((stats["b1"]["mean"].as_f64().unwrap() - 2.0 / 3.0).abs() < 1e-6, "{stats}");
    assert_eq!(stats["b1"]["description"], "(b2*3 - b1) / (b2*3 + b1)");
    let _ = std::fs::remove_dir_all(&tmp);
}

/// A TIFF using a compression code this build does not implement returns `501 Not Implemented`
/// uniformly across `/cog`, `/stac`, and `/mosaicjson`, matching `/zarr`'s `501` for
/// `UnsupportedCodec` (#556).
#[tokio::test]
async fn unsupported_tiff_compression_returns_501_across_cog_stac_and_mosaicjson() {
    let tmp = std::env::temp_dir().join(format!("cogzilla_unsupported_comp_556_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    std::fs::create_dir_all(&tmp).unwrap();

    // Start from a valid synthetic GeoTIFF, then patch IFD0's Compression tag (259) to 34712 (JPEG2000).
    let cog = tmp.join("jp2_unsupported.tif");
    write_synthetic_u16_2band_4326_geotiff(&cog, 64, 64, [10.0, 40.0, 10.64, 40.64], 3, 5, None);
    let mut raw = std::fs::read(&cog).unwrap();
    let ifd0_off = u32::from_le_bytes(raw[4..8].try_into().unwrap()) as usize;
    let num_entries = u16::from_le_bytes(raw[ifd0_off..ifd0_off + 2].try_into().unwrap()) as usize;
    let mut patched = false;
    for i in 0..num_entries {
        let off = ifd0_off + 2 + i * 12;
        let tag = u16::from_le_bytes(raw[off..off + 2].try_into().unwrap());
        if tag == 259 {
            raw[off + 8..off + 10].copy_from_slice(&34712u16.to_le_bytes());
            patched = true;
            break;
        }
    }
    assert!(patched, "synthetic GeoTIFF should contain Compression tag 259");
    std::fs::write(&cog, &raw).unwrap();

    let cog_url = cog.to_string_lossy().to_string();
    let stac_path = tmp.join("item.json");
    std::fs::write(
        &stac_path,
        serde_json::json!({
            "type": "Feature",
            "stac_version": "1.0.0",
            "id": "jp2-item",
            "bbox": [10.0, 40.0, 10.64, 40.64],
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[10.0, 40.0], [10.64, 40.0], [10.64, 40.64], [10.0, 40.64], [10.0, 40.0]]]
            },
            "properties": { "datetime": "2026-01-01T00:00:00Z" },
            "assets": {
                "data": { "href": cog_url, "type": "image/tiff; application=geotiff" }
            }
        })
        .to_string(),
    )
    .unwrap();
    let mosaic_path = tmp.join("mosaic.json");
    std::fs::write(
        &mosaic_path,
        serde_json::json!({
            "mosaicjson": "0.0.3",
            "minzoom": 6,
            "maxzoom": 10,
            "quadkey_zoom": 6,
            "bounds": [10.0, 40.0, 10.64, 40.64],
            "tiles": {
                "122001": [cog_url]
            }
        })
        .to_string(),
    )
    .unwrap();

    let engine = Arc::new(TileEngine::new(64 * 1024 * 1024));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = serve_listener(engine, listener).await;
    });
    let base = format!("http://{addr}");
    let client = reqwest::Client::new();

    for (family, url) in [
        ("/cog/info", format!("{base}/cog/info?url={}", cog.display())),
        ("/stac/info", format!("{base}/stac/info?url={}&assets=data", stac_path.display())),
        (
            "/mosaicjson/tiles",
            format!("{base}/mosaicjson/tiles/WebMercatorQuad/6/33/24.png?url={}", mosaic_path.display()),
        ),
    ] {
        let resp = client.get(&url).send().await.unwrap();
        let status = resp.status().as_u16();
        let body = resp.text().await.unwrap();
        assert_eq!(status, 501, "{family} should return 501 for unsupported TIFF compression: {body}");
        assert!(body.contains("34712"), "{family} body should name compression code 34712: {body}");
    }

    let _ = std::fs::remove_dir_all(&tmp);
}

#[tokio::test]
async fn multi_url_cog_point_stacks_bands_and_evaluates_expression() {
    let tmp = std::env::temp_dir().join(format!(
        "cogzilla_multiurl_pt_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&tmp).unwrap();
    let cog1 = tmp.join("b1.tif");
    let cog2 = tmp.join("b2.tif");
    let cog_far = tmp.join("far.tif");
    write_synthetic_u16_2band_4326_geotiff(&cog1, 32, 32, [10.0, 40.0, 11.0, 41.0], 100, 999, None);
    write_synthetic_u16_2band_4326_geotiff(&cog2, 32, 32, [10.0, 40.0, 11.0, 41.0], 300, 888, None);
    write_synthetic_u16_2band_4326_geotiff(&cog_far, 32, 32, [20.0, 50.0, 21.0, 51.0], 500, 777, None);

    let engine = Arc::new(TileEngine::new(64 * 1024 * 1024));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = serve_listener(engine, listener).await;
    });
    let base = format!("http://{addr}");
    let client = reqwest::Client::new();

    // 1. Stacked multi-URL /cog/point returns band 1 of each URL as b1, b2
    let r1 = client
        .get(format!("{base}/cog/point/10.5,40.5?url={}&url={}", cog1.display(), cog2.display()))
        .send()
        .await
        .unwrap();
    assert_eq!(r1.status(), 200);
    let j1: serde_json::Value = serde_json::from_str(&r1.text().await.unwrap()).unwrap();
    assert_eq!(j1["band_names"], serde_json::json!(["b1", "b2"]));
    assert_eq!(j1["values"], serde_json::json!([100.0, 300.0]));

    // 2. Multi-URL /cog/point with expression=(b2-b1)/(b2+b1) -> (300-100)/(300+100) = 0.5
    let r2 = client
        .get(format!(
            "{base}/cog/point/10.5,40.5?url={}&url={}&expression=(b2-b1)/(b2%2Bb1)",
            cog1.display(),
            cog2.display()
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(r2.status(), 200);
    let j2: serde_json::Value = serde_json::from_str(&r2.text().await.unwrap()).unwrap();
    let v = j2["values"][0].as_f64().unwrap();
    assert!((v - 0.5).abs() < 1e-6, "expected 0.5, got {v}");

    // 3. Partial OOB (cog1 in-bounds, cog_far out-of-bounds) -> [100.0, null]
    let r3 = client
        .get(format!("{base}/cog/point/10.5,40.5?url={}&url={}", cog1.display(), cog_far.display()))
        .send()
        .await
        .unwrap();
    assert_eq!(r3.status(), 200);
    let j3: serde_json::Value = serde_json::from_str(&r3.text().await.unwrap()).unwrap();
    assert_eq!(j3["values"], serde_json::json!([100.0, null]));

    // 4. All OOB -> 404
    let r4 = client
        .get(format!("{base}/cog/point/0.0,0.0?url={}&url={}", cog1.display(), cog2.display()))
        .send()
        .await
        .unwrap();
    assert_eq!(r4.status(), 404);

    let _ = std::fs::remove_dir_all(&tmp);
}

#[tokio::test]
async fn rgba_and_gray_alpha_preserve_continuous_alpha_across_png_and_npy() {
    let tmp = std::env::temp_dir().join(format!(
        "cogzilla_cont_alpha_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&tmp).unwrap();

    // Minimal PNG RGBA8 / Gray+Alpha8 decoder using libdeflater.
    let decode_png_alpha = |png: &[u8]| -> (u32, u32, Vec<u8>) {
        assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n");
        let mut pos = 8usize;
        let mut w = 0u32;
        let mut h = 0u32;
        let mut color_type = 0u8;
        let mut idat = Vec::new();
        while pos + 8 <= png.len() {
            let len = u32::from_be_bytes(png[pos..pos + 4].try_into().unwrap()) as usize;
            let ctype = &png[pos + 4..pos + 8];
            let data = &png[pos + 8..pos + 8 + len];
            if ctype == b"IHDR" {
                w = u32::from_be_bytes(data[0..4].try_into().unwrap());
                h = u32::from_be_bytes(data[4..8].try_into().unwrap());
                color_type = data[9];
            } else if ctype == b"IDAT" {
                idat.extend_from_slice(data);
            }
            pos += 12 + len;
        }
        let bpp: usize = match color_type {
            6 => 4, // RGBA8
            4 => 2, // Gray+Alpha8
            other => panic!("expected PNG with alpha channel (color_type 4 or 6), got {other}"),
        };
        let stride = (w as usize) * bpp;
        let mut raw = vec![0u8; (h as usize) * (1 + stride)];
        let n = libdeflater::Decompressor::new()
            .zlib_decompress(&idat, &mut raw)
            .expect("zlib decompress PNG IDAT");
        assert_eq!(n, raw.len());
        let mut unfilt = vec![0u8; (h as usize) * stride];
        for y in 0..h as usize {
            let filter = raw[y * (1 + stride)];
            let src = &raw[y * (1 + stride) + 1..(y + 1) * (1 + stride)];
            for x in 0..stride {
                let left = if x >= bpp { unfilt[y * stride + x - bpp] } else { 0 };
                let up = if y > 0 { unfilt[(y - 1) * stride + x] } else { 0 };
                let up_left = if x >= bpp && y > 0 {
                    unfilt[(y - 1) * stride + x - bpp]
                } else {
                    0
                };
                unfilt[y * stride + x] = match filter {
                    0 => src[x],
                    1 => src[x].wrapping_add(left),
                    2 => src[x].wrapping_add(up),
                    3 => src[x].wrapping_add(((u16::from(left) + u16::from(up)) / 2) as u8),
                    4 => {
                        let p = i32::from(left) + i32::from(up) - i32::from(up_left);
                        let pa = (p - i32::from(left)).abs();
                        let pb = (p - i32::from(up)).abs();
                        let pc = (p - i32::from(up_left)).abs();
                        let pr = if pa <= pb && pa <= pc {
                            left
                        } else if pb <= pc {
                            up
                        } else {
                            up_left
                        };
                        src[x].wrapping_add(pr)
                    }
                    other => panic!("unexpected PNG filter {other}"),
                };
            }
        }
        let alphas: Vec<u8> = unfilt
            .chunks_exact(bpp)
            .map(|px| px[bpp - 1])
            .collect();
        (w, h, alphas)
    };

    // Helper: write a minimal uncompressed EPSG:4326 RGBA (spp=4) or Gray+Alpha (spp=2) GeoTIFF
    // with ExtraSamples = [2] (UnassociatedAlpha).
    let write_alpha_geotiff = |path: &std::path::Path, spp: u16, pixels: &[u8]| {
        let w = 16u32;
        let h = 16u32;
        // Covers WebMercator tile (6, 33, 24) around lon [5.625..11.25], lat [36.6..40.98]
        let bounds = [5.0f64, 36.0, 12.0, 42.0];
        let mut buf = Vec::<u8>::new();
        buf.extend_from_slice(b"II");
        buf.extend_from_slice(&42u16.to_le_bytes());
        buf.extend_from_slice(&8u32.to_le_bytes());

        let num_entries: u16 = 14;
        let ifd_size = 2 + (num_entries as usize) * 12 + 4;
        let data_start = 8 + ifd_size;
        let bps_off = data_start as u32;
        let bps_bytes = if spp > 2 { (spp as u32) * 2 } else { 0 };
        let scale_off = bps_off + bps_bytes;
        let tiepoint_off = scale_off + 24;
        let geokey_off = tiepoint_off + 48;
        let pixel_off = geokey_off + 24;

        let push_entry = |b: &mut Vec<u8>, tag: u16, ty: u16, count: u32, val: u32| {
            b.extend_from_slice(&tag.to_le_bytes());
            b.extend_from_slice(&ty.to_le_bytes());
            b.extend_from_slice(&count.to_le_bytes());
            b.extend_from_slice(&val.to_le_bytes());
        };

        buf.extend_from_slice(&num_entries.to_le_bytes());
        push_entry(&mut buf, 256, 4, 1, w); // ImageWidth
        push_entry(&mut buf, 257, 4, 1, h); // ImageLength
        if spp > 2 {
            push_entry(&mut buf, 258, 3, spp as u32, bps_off); // BitsPerSample
        } else {
            // 2 inline u16 = [8, 8]
            push_entry(&mut buf, 258, 3, 2, 8u32 | (8u32 << 16));
        }
        push_entry(&mut buf, 259, 3, 1, 1); // Compression = None
        push_entry(&mut buf, 262, 3, 1, if spp == 4 { 2 } else { 1 }); // Photometric RGB / MinIsBlack
        push_entry(&mut buf, 273, 4, 1, pixel_off); // StripOffsets
        push_entry(&mut buf, 277, 3, 1, spp as u32); // SamplesPerPixel
        push_entry(&mut buf, 278, 4, 1, h); // RowsPerStrip
        push_entry(&mut buf, 279, 4, 1, pixels.len() as u32); // StripByteCounts
        push_entry(&mut buf, 284, 3, 1, 1); // PlanarConfiguration = Chunky
        push_entry(&mut buf, 338, 3, 1, 2); // ExtraSamples = UnassociatedAlpha
        push_entry(&mut buf, 33550, 12, 3, scale_off); // ModelPixelScaleTag
        push_entry(&mut buf, 33922, 12, 6, tiepoint_off); // ModelTiepointTag
        push_entry(&mut buf, 34735, 3, 12, geokey_off); // GeoKeyDirectoryTag
        buf.extend_from_slice(&0u32.to_le_bytes());

        if spp > 2 {
            for _ in 0..spp {
                buf.extend_from_slice(&8u16.to_le_bytes());
            }
        }
        let sx = (bounds[2] - bounds[0]) / w as f64;
        let sy = (bounds[3] - bounds[1]) / h as f64;
        for v in [sx, sy, 0.0f64] {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        for v in [0.0f64, 0.0, 0.0, bounds[0], bounds[3], 0.0] {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        for v in [1u16, 1, 0, 2, 1024, 0, 1, 2, 2048, 0, 1, 4326] {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf.extend_from_slice(pixels);
        std::fs::write(path, buf).unwrap();
    };

    // 16x16 RGBA GeoTIFF:
    // - top half (rows 0..8): alpha = 128 (translucent)
    // - bottom-left quadrant (rows 8..16, cols 0..8): alpha = 0 (transparent)
    // - bottom-right quadrant (rows 8..16, cols 8..16): alpha = 255 (opaque)
    let mut rgba_px = Vec::with_capacity(16 * 16 * 4);
    for y in 0..16 {
        for x in 0..16 {
            let a = if y < 8 {
                128u8
            } else if x < 8 {
                0u8
            } else {
                255u8
            };
            rgba_px.extend_from_slice(&[200u8, 100, 50, a]);
        }
    }
    let rgba_path = tmp.join("translucent_rgba.tif");
    write_alpha_geotiff(&rgba_path, 4, &rgba_px);

    // 16x16 Gray+Alpha GeoTIFF: constant gray=180, alpha=64.
    let mut ga_px = Vec::with_capacity(16 * 16 * 2);
    for _ in 0..(16 * 16) {
        ga_px.extend_from_slice(&[180u8, 64]);
    }
    let ga_path = tmp.join("translucent_ga.tif");
    write_alpha_geotiff(&ga_path, 2, &ga_px);

    let engine = Arc::new(TileEngine::new(64 * 1024 * 1024));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = serve_listener(engine, listener).await;
    });
    let base = format!("http://{addr}");
    let client = reqwest::Client::new();

    // 1. /cog/tiles/WebMercatorQuad/6/33/24.png (nearest) must preserve alpha=128 (not 255).
    let r_tile_png = client
        .get(format!(
            "{base}/cog/tiles/WebMercatorQuad/6/33/24.png?url={}&resampling=nearest",
            rgba_path.display()
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(r_tile_png.status(), 200);
    let tile_png_bytes = r_tile_png.bytes().await.unwrap();
    let (tw, th, tile_alphas) = decode_png_alpha(&tile_png_bytes);
    assert_eq!((tw, th), (256, 256));
    assert!(
        tile_alphas.contains(&128),
        "expected continuous alpha=128 in /cog/tiles PNG, unique alphas={:?}",
        tile_alphas
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>()
    );
    assert!(tile_alphas.contains(&0), "expected masked alpha=0 in /cog/tiles PNG");
    assert!(tile_alphas.contains(&255), "expected opaque alpha=255 in /cog/tiles PNG");

    // 2. /cog/tiles/WebMercatorQuad/6/33/24.npy (nearest) must preserve uint8 mask=128.
    let r_tile_npy = client
        .get(format!(
            "{base}/cog/tiles/WebMercatorQuad/6/33/24.npy?url={}&resampling=nearest",
            rgba_path.display()
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(r_tile_npy.status(), 200);
    let npy_bytes = r_tile_npy.bytes().await.unwrap();
    assert_eq!(&npy_bytes[..8], b"\x93NUMPY\x01\x00");
    let hdr_len = usize::from(u16::from_le_bytes([npy_bytes[8], npy_bytes[9]]));
    let hdr = std::str::from_utf8(&npy_bytes[10..10 + hdr_len]).unwrap();
    assert!(hdr.contains("'|u1'"), "expected uint8 .npy, got {hdr}");
    let body = &npy_bytes[10 + hdr_len..];
    assert_eq!(body.len(), 4 * 256 * 256);
    let mask_plane = &body[3 * 256 * 256..];
    assert!(mask_plane.contains(&128), "expected mask=128 in /cog/tiles .npy");
    assert!(mask_plane.contains(&0), "expected mask=0 in /cog/tiles .npy");
    assert!(mask_plane.contains(&255), "expected mask=255 in /cog/tiles .npy");

    // 3. /cog/preview.png and /cog/preview.npy on RGBA (nearest) preserve alpha=128.
    let r_prev_png = client
        .get(format!(
            "{base}/cog/preview.png?url={}&resampling=nearest",
            rgba_path.display()
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(r_prev_png.status(), 200);
    let (_, _, prev_alphas) = decode_png_alpha(&r_prev_png.bytes().await.unwrap());
    assert!(prev_alphas.contains(&128), "expected alpha=128 in /cog/preview.png");
    assert!(prev_alphas.contains(&0), "expected alpha=0 in /cog/preview.png");
    assert!(prev_alphas.contains(&255), "expected alpha=255 in /cog/preview.png");

    // 4. Gray+Alpha /cog/preview.png preserves alpha=64.
    let r_ga_png = client
        .get(format!(
            "{base}/cog/preview.png?url={}&resampling=nearest",
            ga_path.display()
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(r_ga_png.status(), 200);
    let (_, _, ga_alphas) = decode_png_alpha(&r_ga_png.bytes().await.unwrap());
    assert!(
        ga_alphas.iter().all(|&a| a == 64),
        "expected all alpha=64 in Gray+Alpha /cog/preview.png, got {:?}",
        ga_alphas
            .iter()
            .copied()
            .collect::<std::collections::BTreeSet<_>>()
    );

    // 5. /cog/statistics still treats alpha=128 as valid (75% valid = 192 / 256 pixels).
    let r_stats = client
        .get(format!("{base}/cog/statistics?url={}", rgba_path.display()))
        .send()
        .await
        .unwrap();
    assert_eq!(r_stats.status(), 200);
    let stats_json: serde_json::Value = serde_json::from_str(&r_stats.text().await.unwrap()).unwrap();
    assert_eq!(stats_json["b1"]["count"].as_f64().unwrap(), 192.0);
    assert_eq!(stats_json["b1"]["mean"].as_f64().unwrap(), 200.0);

    let _ = std::fs::remove_dir_all(&tmp);
}

#[tokio::test]
async fn rgb_scalar_nodata_masks_only_when_all_bands_match_gdal_dataset_mask() {
    let tmp = std::env::temp_dir().join(format!(
        "cogzilla_rgb_nodata_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&tmp).unwrap();

    // Minimal PNG RGBA8 decoder returning (w, h, Vec<[u8; 4]>).
    let decode_png_rgba = |png: &[u8]| -> (u32, u32, Vec<[u8; 4]>) {
        assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n");
        let mut pos = 8usize;
        let mut w = 0u32;
        let mut h = 0u32;
        let mut color_type = 0u8;
        let mut idat = Vec::new();
        while pos + 8 <= png.len() {
            let len = u32::from_be_bytes(png[pos..pos + 4].try_into().unwrap()) as usize;
            let ctype = &png[pos + 4..pos + 8];
            let data = &png[pos + 8..pos + 8 + len];
            if ctype == b"IHDR" {
                w = u32::from_be_bytes(data[0..4].try_into().unwrap());
                h = u32::from_be_bytes(data[4..8].try_into().unwrap());
                color_type = data[9];
            } else if ctype == b"IDAT" {
                idat.extend_from_slice(data);
            }
            pos += 12 + len;
        }
        assert_eq!(color_type, 6, "expected RGBA8 PNG");
        let bpp = 4usize;
        let stride = (w as usize) * bpp;
        let mut raw = vec![0u8; (h as usize) * (1 + stride)];
        let n = libdeflater::Decompressor::new()
            .zlib_decompress(&idat, &mut raw)
            .expect("zlib decompress PNG");
        assert_eq!(n, raw.len());
        let mut unfilt = vec![0u8; (h as usize) * stride];
        for y in 0..h as usize {
            let filter = raw[y * (1 + stride)];
            let src = &raw[y * (1 + stride) + 1..(y + 1) * (1 + stride)];
            for x in 0..stride {
                let left = if x >= bpp { unfilt[y * stride + x - bpp] } else { 0 };
                let up = if y > 0 { unfilt[(y - 1) * stride + x] } else { 0 };
                let up_left = if x >= bpp && y > 0 {
                    unfilt[(y - 1) * stride + x - bpp]
                } else {
                    0
                };
                unfilt[y * stride + x] = match filter {
                    0 => src[x],
                    1 => src[x].wrapping_add(left),
                    2 => src[x].wrapping_add(up),
                    3 => src[x].wrapping_add(((u16::from(left) + u16::from(up)) / 2) as u8),
                    4 => {
                        let p = i32::from(left) + i32::from(up) - i32::from(up_left);
                        let pa = (p - i32::from(left)).abs();
                        let pb = (p - i32::from(up)).abs();
                        let pc = (p - i32::from(up_left)).abs();
                        let pr = if pa <= pb && pa <= pc {
                            left
                        } else if pb <= pc {
                            up
                        } else {
                            up_left
                        };
                        src[x].wrapping_add(pr)
                    }
                    other => panic!("unexpected PNG filter {other}"),
                };
            }
        }
        let pixels = unfilt
            .chunks_exact(4)
            .map(|c| [c[0], c[1], c[2], c[3]])
            .collect();
        (w, h, pixels)
    };

    // Helper: write a 16x16 3-band uint8 RGB EPSG:4326 GeoTIFF with optional GDAL_NODATA.
    let write_rgb_4326 = |path: &std::path::Path, pixels: &[u8], nodata_str: Option<&str>| {
        let w = 16u32;
        let h = 16u32;
        let bounds = [5.0f64, 36.0, 12.0, 42.0];
        let mut buf = Vec::<u8>::new();
        buf.extend_from_slice(b"II");
        buf.extend_from_slice(&42u16.to_le_bytes());
        buf.extend_from_slice(&8u32.to_le_bytes());

        let nd_bytes = nodata_str.map(|s| {
            let mut v = s.as_bytes().to_vec();
            v.push(0);
            v
        });
        let num_entries: u16 = 13 + u16::from(nd_bytes.is_some());
        let ifd_size = 2 + (num_entries as usize) * 12 + 4;
        let data_start = 8 + ifd_size;
        let bps_off = data_start as u32;
        let scale_off = bps_off + 6;
        let tiepoint_off = scale_off + 24;
        let geokey_off = tiepoint_off + 48;
        let nd_off = geokey_off + 24;
        let nd_len = nd_bytes.as_ref().map_or(0, Vec::len) as u32;
        let pixel_off = nd_off + nd_len;

        let push_entry = |b: &mut Vec<u8>, tag: u16, ty: u16, count: u32, val: u32| {
            b.extend_from_slice(&tag.to_le_bytes());
            b.extend_from_slice(&ty.to_le_bytes());
            b.extend_from_slice(&count.to_le_bytes());
            b.extend_from_slice(&val.to_le_bytes());
        };

        buf.extend_from_slice(&num_entries.to_le_bytes());
        push_entry(&mut buf, 256, 4, 1, w);
        push_entry(&mut buf, 257, 4, 1, h);
        push_entry(&mut buf, 258, 3, 3, bps_off);
        push_entry(&mut buf, 259, 3, 1, 1);
        push_entry(&mut buf, 262, 3, 1, 2); // RGB
        push_entry(&mut buf, 273, 4, 1, pixel_off);
        push_entry(&mut buf, 277, 3, 1, 3);
        push_entry(&mut buf, 278, 4, 1, h);
        push_entry(&mut buf, 279, 4, 1, pixels.len() as u32);
        push_entry(&mut buf, 284, 3, 1, 1); // Chunky
        push_entry(&mut buf, 33550, 12, 3, scale_off);
        push_entry(&mut buf, 33922, 12, 6, tiepoint_off);
        push_entry(&mut buf, 34735, 3, 12, geokey_off);
        if let Some(ref ndb) = nd_bytes {
            if ndb.len() <= 4 {
                let mut val_bytes = [0u8; 4];
                val_bytes[..ndb.len()].copy_from_slice(ndb);
                push_entry(&mut buf, 42113, 2, ndb.len() as u32, u32::from_le_bytes(val_bytes));
            } else {
                push_entry(&mut buf, 42113, 2, ndb.len() as u32, nd_off);
            }
        }
        buf.extend_from_slice(&0u32.to_le_bytes());

        for _ in 0..3 {
            buf.extend_from_slice(&8u16.to_le_bytes());
        }
        let sx = (bounds[2] - bounds[0]) / w as f64;
        let sy = (bounds[3] - bounds[1]) / h as f64;
        for v in [sx, sy, 0.0f64] {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        for v in [0.0f64, 0.0, 0.0, bounds[0], bounds[3], 0.0] {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        for v in [1u16, 1, 0, 2, 1024, 0, 1, 2, 2048, 0, 1, 4326] {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        if let Some(ref ndb) = nd_bytes {
            buf.extend_from_slice(ndb);
        }
        buf.extend_from_slice(pixels);
        std::fs::write(path, buf).unwrap();
    };

    // 16x16 RGB image with 4 quadrants:
    // - top-left:     [0, 0, 0]       (all-zero hole -> masked under nodata=0)
    // - top-right:    [0, 120, 200]   (only R=0 -> VALID & opaque [0, 120, 200, 255] under nodata=0!)
    // - bottom-left:  [255, 255, 255] (all-255 hole -> masked under nodata=255)
    // - bottom-right: [255, 120, 200] (only R=255 -> VALID & opaque [255, 120, 200, 255] under nodata=255!)
    let mut rgb_px = Vec::with_capacity(16 * 16 * 3);
    for y in 0..16 {
        for x in 0..16 {
            let p = match (y < 8, x < 8) {
                (true, true) => [0u8, 0, 0],
                (true, false) => [0u8, 120, 200],
                (false, true) => [255u8, 255, 255],
                (false, false) => [255u8, 120, 200],
            };
            rgb_px.extend_from_slice(&p);
        }
    }
    let rgb_no_nd = tmp.join("rgb_plain.tif");
    let rgb_nd0 = tmp.join("rgb_nd0.tif");
    let rgb_nd255 = tmp.join("rgb_nd255.tif");
    write_rgb_4326(&rgb_no_nd, &rgb_px, None);
    write_rgb_4326(&rgb_nd0, &rgb_px, Some("0"));
    write_rgb_4326(&rgb_nd255, &rgb_px, Some("255"));

    let engine = Arc::new(TileEngine::new(64 * 1024 * 1024));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = serve_listener(engine, listener).await;
    });
    let base = format!("http://{addr}");
    let client = reqwest::Client::new();

    // 1. Runtime ?nodata=0 on /cog/tiles PNG (general path) AND embedded GDAL_NODATA=0 (tile fast path)
    for url_q in [
        format!("url={}&nodata=0&resampling=nearest", rgb_no_nd.display()),
        format!("url={}&resampling=nearest", rgb_nd0.display()),
    ] {
        let r = client
            .get(format!("{base}/cog/tiles/WebMercatorQuad/6/33/24.png?{url_q}"))
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 200);
        let (_, _, px) = decode_png_rgba(&r.bytes().await.unwrap());
        assert!(
            px.contains(&[0, 120, 200, 255]),
            "expected valid opaque [0, 120, 200, 255] under nodata=0 ({url_q})"
        );
        assert!(
            px.contains(&[0, 0, 0, 0]),
            "expected transparent [0, 0, 0, 0] hole under nodata=0 ({url_q})"
        );
    }

    // 2. Runtime ?nodata=255 AND embedded GDAL_NODATA=255:
    //    [255, 120, 200] must stay opaque with exact R=255 (NOT 0 or transparent!).
    for url_q in [
        format!("url={}&nodata=255&resampling=nearest", rgb_no_nd.display()),
        format!("url={}&resampling=nearest", rgb_nd255.display()),
    ] {
        let r = client
            .get(format!("{base}/cog/tiles/WebMercatorQuad/6/33/24.png?{url_q}"))
            .send()
            .await
            .unwrap();
        assert_eq!(r.status(), 200);
        let (_, _, px) = decode_png_rgba(&r.bytes().await.unwrap());
        assert!(
            px.contains(&[255, 120, 200, 255]),
            "expected valid opaque [255, 120, 200, 255] under nodata=255 ({url_q})"
        );
        assert!(
            px.contains(&[0, 0, 0, 0]),
            "expected transparent [0, 0, 0, 0] hole for [255,255,255] under nodata=255 ({url_q})"
        );
    }

    // 3. /cog/preview.png with ?nodata=0 and ?nodata=255
    let r_prev = client
        .get(format!(
            "{base}/cog/preview.png?url={}&nodata=255&resampling=nearest",
            rgb_no_nd.display()
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(r_prev.status(), 200);
    let (_, _, prev_px) = decode_png_rgba(&r_prev.bytes().await.unwrap());
    assert!(prev_px.contains(&[255, 120, 200, 255]));
    assert!(prev_px.contains(&[0, 0, 0, 0]));

    let _ = std::fs::remove_dir_all(&tmp);
}

#[tokio::test]
async fn multiband_npy_preserves_file_nodata_on_partially_valid_pixels() {
    let tmp = std::env::temp_dir().join(format!(
        "cogzilla_multiband_nd_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&tmp).unwrap();

    // 16x16 3-band Float32 EPSG:4326 GeoTIFF with GDAL_NODATA = "-9999.0":
    // - top half: b1=-9999.0, b2=-9999.0, b3=154.2 (partially valid -> mask valid, b1/b2 must stay -9999.0)
    // - bottom half: b1=-9999.0, b2=-9999.0, b3=-9999.0 (all nodata -> masked)
    let w = 16u32;
    let h = 16u32;
    let bounds = [5.0f64, 36.0, 12.0, 42.0];
    let mut pixels = Vec::with_capacity((w * h * 3 * 4) as usize);
    for y in 0..h {
        for _x in 0..w {
            let b3 = if y < 8 { 154.2f32 } else { -9999.0f32 };
            for v in [-9999.0f32, -9999.0f32, b3] {
                pixels.extend_from_slice(&v.to_le_bytes());
            }
        }
    }
    let nd_bytes = b"-9999.0\0";
    let mut buf = Vec::<u8>::new();
    buf.extend_from_slice(b"II");
    buf.extend_from_slice(&42u16.to_le_bytes());
    buf.extend_from_slice(&8u32.to_le_bytes());

    let num_entries: u16 = 15;
    let ifd_size = 2 + (num_entries as usize) * 12 + 4;
    let data_start = 8 + ifd_size;
    let bps_off = data_start as u32;
    let sf_off = bps_off + 6;
    let scale_off = sf_off + 6;
    let tiepoint_off = scale_off + 24;
    let geokey_off = tiepoint_off + 48;
    let nd_off = geokey_off + 24;
    let pixel_off = nd_off + nd_bytes.len() as u32;

    let push_entry = |b: &mut Vec<u8>, tag: u16, ty: u16, count: u32, val: u32| {
        b.extend_from_slice(&tag.to_le_bytes());
        b.extend_from_slice(&ty.to_le_bytes());
        b.extend_from_slice(&count.to_le_bytes());
        b.extend_from_slice(&val.to_le_bytes());
    };
    buf.extend_from_slice(&num_entries.to_le_bytes());
    push_entry(&mut buf, 256, 4, 1, w);
    push_entry(&mut buf, 257, 4, 1, h);
    push_entry(&mut buf, 258, 3, 3, bps_off);
    push_entry(&mut buf, 259, 3, 1, 1);
    push_entry(&mut buf, 262, 3, 1, 1);
    push_entry(&mut buf, 273, 4, 1, pixel_off);
    push_entry(&mut buf, 277, 3, 1, 3);
    push_entry(&mut buf, 278, 4, 1, h);
    push_entry(&mut buf, 279, 4, 1, pixels.len() as u32);
    push_entry(&mut buf, 284, 3, 1, 1);
    push_entry(&mut buf, 339, 3, 3, sf_off); // SampleFormat = [3, 3, 3] (IEEEFP)
    push_entry(&mut buf, 33550, 12, 3, scale_off);
    push_entry(&mut buf, 33922, 12, 6, tiepoint_off);
    push_entry(&mut buf, 34735, 3, 12, geokey_off);
    push_entry(&mut buf, 42113, 2, nd_bytes.len() as u32, nd_off);
    buf.extend_from_slice(&0u32.to_le_bytes());

    for _ in 0..3 {
        buf.extend_from_slice(&32u16.to_le_bytes());
    }
    for _ in 0..3 {
        buf.extend_from_slice(&3u16.to_le_bytes());
    }
    let sx = (bounds[2] - bounds[0]) / w as f64;
    let sy = (bounds[3] - bounds[1]) / h as f64;
    for v in [sx, sy, 0.0f64] {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    for v in [0.0f64, 0.0, 0.0, bounds[0], bounds[3], 0.0] {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    for v in [1u16, 1, 0, 2, 1024, 0, 1, 2, 2048, 0, 1, 4326] {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    buf.extend_from_slice(nd_bytes);
    buf.extend_from_slice(&pixels);
    let path = tmp.join("asym_f32_nd9999.tif");
    std::fs::write(&path, buf).unwrap();

    let engine = Arc::new(TileEngine::new(64 * 1024 * 1024));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = serve_listener(engine, listener).await;
    });
    let base = format!("http://{addr}");
    let client = reqwest::Client::new();

    let r = client
        .get(format!(
            "{base}/cog/preview.npy?url={}&resampling=nearest",
            path.display()
        ))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let npy = r.bytes().await.unwrap();
    let (shape, start) = cogzilla_kernel::f32_npy_shape(&npy).expect("f32 npy");
    assert_eq!(shape, vec![4, 16, 16]);
    let vals: Vec<f32> = npy[start..]
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let npix = 16 * 16;
    // Top-half pixel (row 0, col 0): b1 == -9999.0 (NOT 0.0!), b2 == -9999.0, b3 == 154.2, mask == f32::MAX
    assert_eq!(vals[0], -9999.0, "b1 must preserve file nodata -9999.0 on partially-valid pixel");
    assert_eq!(vals[npix], -9999.0, "b2 must preserve file nodata -9999.0 on partially-valid pixel");
    assert!((vals[2 * npix] - 154.2).abs() < 1e-4);
    assert_eq!(vals[3 * npix], f32::MAX);
    // Bottom-half pixel (row 15, col 0): mask == f32::MIN
    assert_eq!(vals[3 * npix + 15 * 16], f32::MIN);

    let _ = std::fs::remove_dir_all(&tmp);
}
