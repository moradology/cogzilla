use super::*;
use tokio::net::TcpListener;

fn single(rescale: (f32, f32), colormap: ColormapName) -> BandRequest {
    BandRequest::Single { band_1based: 1, rescale, colormap }
}

fn selected(bands: usize, values: &[f32]) -> Selected {
    Selected { bands, values: values.to_vec() }
}

#[test]
fn a_pixel_selection_is_painted_by_rio_tilers_rules() {
    let pq = parse_query(None);
    let nan = f32::NAN;
    // One band without a colour map: the byte is the grey.
    let grey = single((0.0, 255.0), ColormapName::Grayscale);
    let rgba = paint_selected(selected(1, &[33.0, 244.0, nan]), 3, &grey, &pq, None).unwrap();
    assert_eq!(rgba, [33, 33, 33, 255, 244, 244, 244, 255, 0, 0, 0, 0]);
    // The byte is `linear_rescale`'s: the top of the range is 255.
    let rgba = paint_selected(selected(1, &[0.3]), 1, &single((0.0, 0.3), ColormapName::Grayscale), &pq, None).unwrap();
    assert_eq!(rgba, [255, 255, 255, 255]);
    // Red, green and blue: there where any band has a value, with 0 for one that has not.
    let rgb = BandRequest::Rgb { bands_1based: [1, 2, 3], rescale_min: [0.0; 3], rescale_max: [255.0; 3] };
    let rgba = paint_selected(selected(3, &[10.0, nan, nan, 20.0, 20.0, nan, 30.0, nan, nan]), 3, &rgb, &pq, None).unwrap();
    assert_eq!(rgba, [10, 20, 30, 255, 0, 20, 0, 255, 0, 0, 0, 0]);
    // An infinity is a value, clipped to the range.
    let rgba = paint_selected(selected(1, &[f32::INFINITY, f32::NEG_INFINITY]), 2, &grey, &pq, None).unwrap();
    assert_eq!(rgba, [255, 255, 255, 255, 0, 0, 0, 255]);
    // A range with no width is 0 throughout.
    let rgba = paint_selected(selected(1, &[2.0, 3.0, 4.0]), 3, &single((3.0, 3.0), ColormapName::Grayscale), &pq, None).unwrap();
    assert_eq!(rgba, [0, 0, 0, 255, 0, 0, 0, 255, 0, 0, 0, 255]);
}

#[test]
fn a_pixel_selection_keeps_the_colour_maps_a_raster_or_a_request_brings() {
    let pq = parse_query(None);
    let grey = single((0.0, 255.0), ColormapName::Grayscale);
    let values = selected(1, &[1.0, 2.0, 7.0, f32::NAN, 300.0]);
    // A raster's own palette: the class's colour and its transparency, by the value.
    let mut palette = vec![[0u8; 4]; 256];
    palette[1] = [10, 20, 30, 255];
    palette[2] = [40, 50, 60, 128];
    let own = CustomColormap::from_rgba_palette(&palette);
    let rgba = paint_selected(values.clone(), 5, &grey, &pq, Some(&own)).unwrap();
    assert_eq!(rgba[..8], [10, 20, 30, 255, 40, 50, 60, 128]);
    assert_eq!(rgba[8..], [0; 12]);
    // A colour map given as JSON, classes and intervals.
    let classes = CustomColormap::from_json_str(r##"{"1": [1, 2, 3, 200], "300": "#ff0000"}"##, false).unwrap();
    let rgba = paint_selected(values.clone(), 5, &grey, &pq, Some(&classes)).unwrap();
    assert_eq!(rgba[..4], [1, 2, 3, 200]);
    assert_eq!(rgba[4..16], [0; 12]);
    assert_eq!(rgba[16..], [255, 0, 0, 255]);
    let spans = CustomColormap::from_json_str("[[[0, 5], [9, 9, 9, 255]], [[5, 10], [7, 7, 7, 255]]]", false).unwrap();
    let rgba = paint_selected(values.clone(), 5, &grey, &pq, Some(&spans)).unwrap();
    assert_eq!(rgba[..12], [9, 9, 9, 255, 9, 9, 9, 255, 7, 7, 7, 255]);
    assert_eq!(rgba[12..], [0; 8]);
    // A ramp between stops, over the rescaled byte.
    let ramp = CustomColormap::from_json_str_with_type(r#"{"0": [0, 0, 0, 255], "255": [255, 255, 255, 255]}"#, true, true).unwrap();
    let rgba = paint_selected(selected(1, &[0.0, 50.0, 100.0]), 3, &single((0.0, 100.0), ColormapName::Grayscale), &pq, Some(&ramp)).unwrap();
    assert_eq!([rgba[0], rgba[8]], [0, 255]);
    assert!((126..=128).contains(&rgba[4]), "{}", rgba[4]);
    // Three bands are a picture already: no colour map is looked at.
    let rgb = BandRequest::Rgb { bands_1based: [1, 2, 3], rescale_min: [0.0; 3], rescale_max: [255.0; 3] };
    let rgba = paint_selected(selected(3, &[1.0, 2.0, 3.0]), 1, &rgb, &pq, Some(&own)).unwrap();
    assert_eq!(rgba, [1, 2, 3, 255]);
}

#[test]
fn an_array_asked_with_a_rescale_is_the_pictures_bytes() {
    // One band of 50 and a pixel that is not there, with its mask, as the renderer writes it.
    let f32_npy = |values: &[f32], mask: bool| {
        let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        cogzilla_kernel::wrap_raw_f32_as_npy_with_mask(Bytes::from(raw), values.len() / 2, 1, 2, mask)
    };
    let body = |npy: &[u8]| npy[10 + usize::from(u16::from_le_bytes([npy[8], npy[9]]))..].to_vec();
    let shown = |q: &str, npy: &[u8]| shown_array(&parse_query(Some(q)), None, npy).unwrap();
    let grey = shown("rescale=0,100", &f32_npy(&[50.0, f32::NAN], true));
    assert!(std::str::from_utf8(&grey[10..60]).unwrap().contains("'descr': '|u1'"));
    assert!(std::str::from_utf8(&grey[10..80]).unwrap().contains("'shape': (2, 1, 2)"));
    // 50 of 0..100 is 127, as rio-tiler makes it; the mask is 255 where the pixel is.
    assert_eq!(body(&grey), [127, 0, 255, 0]);
    // A colour map is red, green and blue.
    let mapped = shown("rescale=0,100&colormap_name=viridis", &f32_npy(&[50.0, f32::NAN], true));
    let rgb = ColormapName::Viridis.lut().rgb[127];
    assert_eq!(body(&mapped), [rgb[0], 0, rgb[1], 0, rgb[2], 0, 255, 0]);
    // Without the mask, the band alone.
    let bare = shown("rescale=0,100&return_mask=false", &f32_npy(&[50.0, 100.0], false));
    assert_eq!(body(&bare), [127, 255]);
    // 2-band and 4-band rescaled arrays preserve every band and the any-band-valid mask.
    let two = shown("rescale=0,100&rescale=0,200", &f32_npy(&[50.0, f32::NAN, 100.0, 200.0], true));
    assert!(std::str::from_utf8(&two[10..80]).unwrap().contains("'shape': (3, 1, 2)"));
    assert_eq!(body(&two), [127, 0, 127, 255, 255, 255]);
    let four = shown("rescale=0,100", &f32_npy(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0], true));
    assert!(std::str::from_utf8(&four[10..80]).unwrap().contains("'shape': (5, 1, 2)"));
    assert_eq!(body(&four).len(), 10);
    // STAC per-asset expression with rescale still asks for painted bytes.
    let pq_ae = parse_query(Some("assets=red|expression=b1*2&rescale=0,100"));
    assert!(matches!(array_asked(&pq_ae, Some(&"tile.npy")), Some(ArrayAsked::Shown(_))));
    // What is not a float32 array is not painted.
    assert!(shown_array(&parse_query(Some("rescale=0,100")), None, b"not an array").is_none());
}

#[test]
fn a_pixel_selection_keeps_a_range_given_upside_down() {
    let pq = parse_query(None);
    let rgba = paint_selected(selected(1, &[0.0, 51.0, 255.0, f32::NAN]), 4, &single((255.0, 0.0), ColormapName::Grayscale), &pq, None).unwrap();
    assert_eq!(rgba, [255, 255, 255, 255, 204, 204, 204, 255, 0, 0, 0, 255, 0, 0, 0, 0]);
    let table = ColormapName::Viridis.lut();
    let rgba = paint_selected(selected(1, &[0.0]), 1, &single((255.0, 0.0), ColormapName::Viridis), &pq, None).unwrap();
    assert_eq!(rgba[..3], table.rgb[255]);
    let rgb = BandRequest::Rgb { bands_1based: [1, 2, 3], rescale_min: [0.0, 100.0, 0.0], rescale_max: [100.0, 0.0, 100.0] };
    let rgba = paint_selected(selected(3, &[20.0, 20.0, 20.0]), 1, &rgb, &pq, None).unwrap();
    assert_eq!(rgba, [51, 204, 51, 255]);
}

#[test]
fn a_pixel_selection_paints_whatever_bands_the_request_leaves() {
    let pq = parse_query(None);
    // Band math for red, green and blue keeps its own ranges.
    let prog = || ExprProgram::compile("b1").unwrap();
    let math = BandRequest::CustomExprRgb { progs: [prog(), prog(), prog()], rescale_min: [0.0; 3], rescale_max: [100.0, 200.0, 400.0] };
    let rgba = paint_selected(selected(3, &[100.0, 100.0, 100.0]), 1, &math, &pq, None).unwrap();
    assert_eq!(rgba, [255, 127, 63, 255]);
    // Bands with no range of their own are bytes as they are: a terrain encoding's
    // three, and the first three of however many were asked for.
    let terrain = BandRequest::Terrain { band_1based: 1, algorithm: DemAlgorithm::Terrarium, math: None };
    let rgba = paint_selected(selected(3, &[128.0, 7.0, 255.0]), 1, &terrain, &pq, None).unwrap();
    assert_eq!(rgba, [128, 7, 255, 255]);
    let four = BandRequest::MultiBand { bands_1based: vec![1, 2, 3, 4] };
    let rgba = paint_selected(selected(4, &[1.0, 2.0, 3.0, 4.0]), 1, &four, &pq, None).unwrap();
    assert_eq!(rgba, [1, 2, 3, 255]);
    // Two bands: the first, under the request's colour map.
    let two = BandRequest::MultiBand { bands_1based: vec![1, 2] };
    let mapped = parse_query(Some("colormap_name=viridis"));
    let table = ColormapName::Viridis.lut();
    let rgba = paint_selected(selected(2, &[64.0, 200.0]), 1, &two, &mapped, None).unwrap();
    assert_eq!(rgba[..3], table.rgb[64]);
    // One band left of a request for three (a count): bytes as they are, not the
    // request's ranges.
    let rgb = BandRequest::Rgb { bands_1based: [1, 2, 3], rescale_min: [0.0; 3], rescale_max: [4000.0; 3] };
    let rgba = paint_selected(selected(1, &[2.0]), 1, &rgb, &pq, None).unwrap();
    assert_eq!(rgba, [2, 2, 2, 255]);
    // A request for one band that three came of: red, green and blue as they are.
    let rgba = paint_selected(selected(3, &[1.0, 2.0, 3.0]), 1, &single((0.0, 4000.0), ColormapName::Viridis), &pq, None).unwrap();
    assert_eq!(rgba, [1, 2, 3, 255]);
}

#[test]
fn a_pixel_selection_takes_a_colour_formula_before_the_colour_map() {
    let pq = parse_query(Some("color_formula=gamma%20rgb%202"));
    let formula = ColorFormula::parse("gamma rgb 2").unwrap();
    let levels = paint::formula_levels(&formula);
    assert_ne!(levels[64], 64);
    // One band: the formula on its byte, then the map, or the grey that byte is.
    let table = ColormapName::Viridis.lut();
    let rgba = paint_selected(selected(1, &[64.0, f32::NAN]), 2, &single((0.0, 255.0), ColormapName::Viridis), &pq, None).unwrap();
    assert_eq!(rgba[..3], table.rgb[usize::from(levels[64])]);
    assert_eq!(rgba[3..], [255, 0, 0, 0, 0]);
    let rgba = paint_selected(selected(1, &[64.0]), 1, &single((0.0, 255.0), ColormapName::Grayscale), &pq, None).unwrap();
    assert_eq!(rgba, [levels[64], levels[64], levels[64], 255]);
    // Three bands: the formula on the finished picture.
    let rgb = BandRequest::Rgb { bands_1based: [1, 2, 3], rescale_min: [0.0; 3], rescale_max: [255.0; 3] };
    let rgba = paint_selected(selected(3, &[64.0, 0.0, 255.0]), 1, &rgb, &pq, None).unwrap();
    assert_eq!(rgba, [levels[64], 0, 255, 255]);
    // Under a colour map of the raster's or the request's own it is for the finished picture.
    let classes = CustomColormap::from_json_str(r#"{"64": [1, 2, 3, 255]}"#, false).unwrap();
    let rgba = paint_selected(selected(1, &[64.0]), 1, &single((0.0, 255.0), ColormapName::Grayscale), &pq, Some(&classes)).unwrap();
    let mut mapped = [1u8, 2, 3, 255];
    formula.apply_buffer_in_place(&mut mapped, 4);
    assert_ne!(mapped, [1, 2, 3, 255]);
    assert_eq!(rgba, mapped);
    // A formula that does not parse is the request's fault.
    let bad = parse_query(Some("color_formula=nonsense"));
    assert!(matches!(
        paint_selected(selected(1, &[1.0]), 1, &single((0.0, 255.0), ColormapName::Grayscale), &bad, None),
        Err(EngineError::BadRequest(_))
    ));
}

#[test]
fn a_painted_pixel_selection_is_encoded_as_any_other_picture() {
    let pq = parse_query(None);
    let grey = single((0.0, 255.0), ColormapName::Grayscale);
    let rgba = paint_selected(selected(1, &[33.0, f32::NAN, 200.0, 90.0]), 4, &grey, &pq, None).unwrap();
    // The bare pixels, with and without the mask.
    assert_eq!(encode_rgba_output(2, 2, &rgba, OutputFormat::RawRgba, true).as_ref(), rgba.as_slice());
    let png = encode_rgba_output(2, 2, &rgba, OutputFormat::Png, true);
    assert_eq!(png[..8], [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]);
    let unmasked = encode_rgba_output(2, 2, &rgba, OutputFormat::Png, false);
    assert_eq!(unmasked[..8], png[..8]);
    assert_ne!(unmasked, png, "return_mask=false leaves the mask out");
    let jpeg = encode_rgba_output(2, 2, &rgba, OutputFormat::Jpeg { quality: 75 }, true);
    assert_eq!(jpeg[..2], [0xff, 0xd8]);
}

#[test]
fn unrescaled_uint16_png_emits_16_bit_png_and_clamps_overshoot() {
    let pq_plain = parse_query(None);
    let pq_rescaled = parse_query(Some("rescale=0,1000"));
    let pq_cmap = parse_query(Some("colormap_name=viridis"));
    let single = BandRequest::Single { band_1based: 1, rescale: (0.0, 65535.0), colormap: ColormapName::Grayscale };
    let rgb = BandRequest::Rgb { bands_1based: [1, 2, 3], rescale_min: [0.0; 3], rescale_max: [65535.0; 3] };
    let two_band = BandRequest::MultiBand { bands_1based: vec![1, 2] };
    let four_band = BandRequest::MultiBand { bands_1based: vec![1, 2, 3, 4] };
    let pq_no_mask = parse_query(Some("return_mask=false"));
    assert!(wants_u16_png(OutputFormat::Png, "image/png", &pq_plain, None, &single, cogzilla_raster::SampleType::Uint16));
    assert!(wants_u16_png(OutputFormat::Png, "image/png", &pq_plain, None, &rgb, cogzilla_raster::SampleType::Uint16));
    assert!(!wants_u16_png(OutputFormat::Png, "image/png", &pq_plain, None, &two_band, cogzilla_raster::SampleType::Uint16));
    assert!(!wants_u16_png(OutputFormat::Png, "image/png", &pq_no_mask, None, &two_band, cogzilla_raster::SampleType::Uint16));
    assert!(!wants_u16_png(OutputFormat::Png, "image/png", &pq_no_mask, None, &four_band, cogzilla_raster::SampleType::Uint16));
    assert!(!wants_u16_png(OutputFormat::Png, "image/png", &pq_plain, None, &single, cogzilla_raster::SampleType::Uint8));
    assert!(!wants_u16_png(OutputFormat::Png, "image/png", &pq_rescaled, None, &single, cogzilla_raster::SampleType::Uint16));
    assert!(!wants_u16_png(OutputFormat::Png, "image/png", &pq_cmap, None, &single, cogzilla_raster::SampleType::Uint16));

    // 4x1 single-band f32 buffer testing clamp (-1.0 -> 0, 70000.0 -> 65535),
    // rounding (123.6 -> 124), and NaN (0, alpha=0) in 16-bit Grey+Alpha PNG.
    let mut raw = Vec::new();
    for v in [-1.0f32, 123.6, 70000.0, f32::NAN] {
        raw.extend_from_slice(&v.to_le_bytes());
    }
    let png = wrap_geotiff_fmt(Bytes::from(raw), "image/png", OutputFormat::RawF32, 4, 1, [0.0; 4], 4326, true);
    assert_eq!(&png[..8], &[0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a]);
    // IHDR: byte 24 is bit depth (16), byte 25 is colour type (4 = Greyscale+Alpha).
    assert_eq!(png[24], 16, "expected 16-bit PNG");
    assert_eq!(png[25], 4, "expected Greyscale+Alpha colour type");
    // Extract and inflate IDAT payload to verify decoded 16-bit BE samples & alpha.
    let mut idat = Vec::new();
    let mut pos = 8usize;
    while pos + 12 <= png.len() {
        let len = u32::from_be_bytes([png[pos], png[pos + 1], png[pos + 2], png[pos + 3]]) as usize;
        let chunk_type = &png[pos + 4..pos + 8];
        if chunk_type == b"IDAT" {
            idat.extend_from_slice(&png[pos + 8..pos + 8 + len]);
        }
        pos += 12 + len;
    }
    let mut scanline = vec![0u8; 17];
    let n = libdeflater::Decompressor::new()
        .zlib_decompress(&idat, &mut scanline)
        .expect("valid zlib IDAT");
    // Filter byte (0) + 4 pixels * (u16 value + u16 alpha) = 1 + 16 = 17 bytes.
    assert_eq!(n, 17);
    assert_eq!(scanline[0], 0);
    let decoded: Vec<(u16, u16)> = scanline[1..]
        .chunks_exact(4)
        .map(|c| (u16::from_be_bytes([c[0], c[1]]), u16::from_be_bytes([c[2], c[3]])))
        .collect();
    assert_eq!(decoded, vec![(0, 65535), (124, 65535), (65535, 65535), (0, 0)]);

    // 2-band Uint16 `.png` tile request is consistently rejected with 400 (matching Uint8/Float32).
    let pq_2b = parse_query(Some("bidx=1&bidx=2"));
    let res = build_tile_request_for_tms(
        None,
        TileCoord { z: 0, x: 0, y: 0 },
        None,
        Some("png"),
        &pq_2b,
        2,
        cogzilla_raster::SampleType::Uint16,
        "WebMercatorQuad",
        None,
        false,
    );
    assert!(
        matches!(res, Err(EngineError::BadRequest(_))),
        "2-band Uint16 .png tile should return 400 BadRequest, got: {:?}",
        res
    );
}

#[test]
fn pixel_selection_names_keep_their_aliases_and_any_case() {
    for (name, method) in [
        ("first", Selection::First),
        ("highest", Selection::Highest),
        ("lowest", Selection::Lowest),
        ("mean", Selection::Mean),
        ("median", Selection::Median),
        ("stdev", Selection::Stdev),
        ("std", Selection::Stdev),
        ("STD", Selection::Stdev),
        ("StDev", Selection::Stdev),
        ("count", Selection::Count),
        ("Highest", Selection::Highest),
        ("LastBandLow", Selection::LastBandLow),
        ("LASTBANDHIGH", Selection::LastBandHigh),
    ] {
        assert_eq!(parse_pixel_selection(name), Some(method), "{name}");
        let pq = parse_query(Some(&format!("pixel_selection={name}")));
        assert_eq!(pq.pixel_selection, method, "{name}");
        assert!(pq.errors.is_empty(), "{name}: {:?}", pq.errors);
    }
    assert_eq!(parse_query(None).pixel_selection, Selection::First);
    for name in ["sum", "", "stddev", "last"] {
        assert_eq!(parse_pixel_selection(name), None, "{name}");
    }
    let pq = parse_query(Some("pixel_selection=sum"));
    assert_eq!(
        pq.errors,
        ["unsupported pixel_selection 'sum' (expected first, highest, lowest, mean, median, stdev, count, lastbandlow, or lastbandhigh)"]
    );
}

#[test]
fn mean_and_median_are_cast_back_only_for_a_raster_s_own_whole_numbers() {
    use cogzilla_raster::SampleType as T;
    let single = BandRequest::Single { band_1based: 1, rescale: (0.0, 255.0), colormap: ColormapName::Grayscale };
    let rgb = BandRequest::Rgb { bands_1based: [1, 2, 3], rescale_min: [0.0; 3], rescale_max: [255.0; 3] };
    let multi = BandRequest::MultiBand { bands_1based: vec![1, 2] };
    for bands in [&single, &rgb, &multi] {
        for kind in [T::Uint8, T::Int8, T::Uint16, T::Int16, T::Uint32, T::Int32] {
            assert!(reads_whole_numbers(false, bands, kind), "{bands:?} {kind:?}");
            assert!(!reads_whole_numbers(true, bands, kind), "unscaled {bands:?} {kind:?}");
        }
        for kind in [T::Float32, T::Float64] {
            assert!(!reads_whole_numbers(false, bands, kind), "{bands:?} {kind:?}");
        }
    }
    let pq = parse_query(Some("expression=b1/2"));
    let math = build_band_request(&pq, 1, T::Uint8).expect("a band math request");
    assert!(!reads_whole_numbers(false, &math, T::Uint8), "{math:?}");
}

#[test]
fn the_median_budget_is_64_mib_of_held_scenes() {
    assert_eq!(MAX_MEDIAN_BUFFER_BYTES, 64 * 1024 * 1024);
    let (bands, pixels) = (3, 256 * 256);
    // The 64 scenes a request may name fit at the default tile size.
    assert!(Fold::<f32>::held_bytes(Selection::Median, 64, bands, pixels) <= MAX_MEDIAN_BUFFER_BYTES);
    // At 2048 px a side, one three-band scene fits and the second does not.
    let pixels = 2048 * 2048;
    assert!(Fold::<f32>::held_bytes(Selection::Median, 1, bands, pixels) <= MAX_MEDIAN_BUFFER_BYTES);
    assert!(Fold::<f32>::held_bytes(Selection::Median, 2, bands, pixels) > MAX_MEDIAN_BUFFER_BYTES);
    // No other method holds its scenes, so none is bounded by their number.
    assert_eq!(Fold::<f32>::held_bytes(Selection::Mean, 64, bands, pixels), 0);
}

#[test]
fn colormap_param_sets_explicit_for_hillshade_and_stac_bbox_accepts_coord_crs() {
    let pq = parse_query(Some("algorithm=hillshade&colormap=terrain"));
    assert!(pq.colormap_explicit);
    let bands = build_band_request(&pq, 1, cogzilla_raster::SampleType::Float32)
        .expect("valid hillshade request");
    match bands {
        BandRequest::Dem { colormap, .. } => {
            assert_ne!(colormap, ColormapName::Grayscale);
            assert_eq!(colormap, ColormapName::from_str_opt("terrain").unwrap());
        }
        other => panic!("expected BandRequest::Dem, got {other:?}"),
    }

    // /stac/bbox must accept coord_crs=EPSG:3857 (matching /cog/bbox)
    let pq_bbox = parse_query(Some("url=foo.json&coord_crs=EPSG:3857"));
    let res = validate_query(&["stac", "bbox", "0,0,1000,1000.png"], &pq_bbox);
    assert!(res.is_ok(), "unexpected error: {res:?}");
}

#[test]
fn xyz_bounds_and_http_error_status_mapping() {
    let pq = parse_query(None);
    assert_eq!(pq.resample, ResampleMode::Nearest);
    assert_eq!(pq.reproject, ResampleMode::Nearest);
    let u8t = cogzilla_raster::SampleType::Uint8;
    // WebMercatorQuad at z=2 has 4x4 tiles (0..=3); out-of-matrix coords return 404 NotFound (#624)
    assert!(build_tile_request(None, TileCoord { z: 2, x: 3, y: 3 }, None, Some("png"), &pq, 3, u8t, false, None).is_ok());
    assert!(matches!(
        build_tile_request(None, TileCoord { z: 2, x: 4, y: 0 }, None, Some("png"), &pq, 3, u8t, false, None),
        Err(EngineError::NotFound(_))
    ));
    assert!(matches!(
        build_tile_request(None, TileCoord { z: 2, x: 0, y: 4 }, None, Some("png"), &pq, 3, u8t, false, None),
        Err(EngineError::NotFound(_))
    ));
    // WorldCRS84Quad at z=2 has 8x4 tiles (x: 0..=7, y: 0..=3)
    assert!(build_tile_request(None, TileCoord { z: 2, x: 7, y: 3 }, None, Some("png"), &pq, 3, u8t, true, None).is_ok());
    assert!(matches!(
        build_tile_request(None, TileCoord { z: 2, x: 8, y: 0 }, None, Some("png"), &pq, 3, u8t, true, None),
        Err(EngineError::NotFound(_))
    ));
    assert!(matches!(
        build_tile_request(None, TileCoord { z: 2, x: 0, y: 4 }, None, Some("png"), &pq, 3, u8t, true, None),
        Err(EngineError::NotFound(_))
    ));

    // ?reproject= is accepted and parsed independently from ?resampling=
    let pq_rp = parse_query(Some("resampling=bilinear&reproject=cubic"));
    assert!(pq_rp.errors.is_empty());
    assert_eq!(pq_rp.resample, ResampleMode::Bilinear);
    assert_eq!(pq_rp.reproject, ResampleMode::Cubic);
    assert!(validate_query(&["cog", "tiles", "WebMercatorQuad", "0", "0", "0.png"], &pq_rp).is_ok());

    // Single-band default colormap is Grayscale (matching TiTiler)
    assert_eq!(pq.colormap, ColormapName::Grayscale);

    // Form-encoded ?expression= with spaces (`+`), `%2B`, `%2A`, `%5E`, `%28`/`%29` decodes cleanly, while raw `(b1+b2)` still works
    let pq_form = parse_query(Some("expression=%28b1+-+b2%29+%2F+%28b1+%2B+b2%29"));
    assert_eq!(pq_form.expr_raw.as_deref(), Some("(b1 - b2) / (b1 + b2)"));
    let pq_mul = parse_query(Some("expression=10+%2A+log10%28b1%29"));
    assert_eq!(pq_mul.expr_raw.as_deref(), Some("10 * log10(b1)"));
    let pq_raw = parse_query(Some("expression=(b1-b2)/(b1+b2)"));
    assert_eq!(pq_raw.expr_raw.as_deref(), Some("(b1-b2)/(b1+b2)"));

    // ?format= applies on format-less tile routes parsed via parse_y_scale_ext
    let (y_parsed, sc_parsed, ext_opt) = parse_y_scale_ext("1@2x").unwrap();
    assert_eq!((y_parsed, sc_parsed, ext_opt), (1, Some(2), None));
    let pq_fmt = parse_query(Some("format=webp"));
    let (treq, ct) =
        build_tile_request(None, TileCoord { z: 2, x: 1, y: y_parsed }, sc_parsed, ext_opt, &pq_fmt, 3, u8t, false, None)
            .unwrap();
    assert_eq!(ct, "image/webp");
    assert!(matches!(treq.format, OutputFormat::Webp { .. }));

    // Invalid minzoom > maxzoom rejected on tilejson.json
    let pq_bad_z = parse_query(Some("minzoom=14&maxzoom=8"));
    assert!(validate_query(&["cog", "WebMercatorQuad", "tilejson.json"], &pq_bad_z).is_err());

    // Default dtype rescale ranges & aspect-preserving dims & hex colormaps
    assert_eq!(default_rescale_for_dtype(cogzilla_raster::SampleType::Uint16, false), (0.0, 65535.0));
    assert_eq!(default_rescale_for_dtype(cogzilla_raster::SampleType::Int16, true), (-32768.0, 32767.0));
    assert_eq!(aspect_preserving_dims([0.0, 0.0, 2.0, 1.0], 512), (512, 256));
    assert_eq!(aspect_preserving_dims([0.0, 0.0, 1.0, 4.0], 512), (128, 512));
    let hex_lut = cogzilla_kernel::ColormapLut::from_json_str(r##"{"0":"#112233","255":"#ff000080"}"##).unwrap();
    assert_eq!(hex_lut.rgb[0], [0x11, 0x22, 0x33]);
    assert_eq!(hex_lut.rgba_u32[255].to_le_bytes(), [0xff, 0x00, 0x00, 0x80]);

    // TiTiler query param parity: nodata=nan/inf/-inf, z_exaggeration, [0,1000] rescale,
    // colormap_name precedence over inline colormap, and tilejson minzoom/maxzoom/tile_format.
    let pq_nd = parse_query(Some("nodata=nan&rescale=[0,%201000]&algorithm_params=%7B%22z_exaggeration%22:2.5%7D"));
    assert!(pq_nd.errors.is_empty());
    assert!(pq_nd.nodata.unwrap().is_nan());
    assert_eq!(pq_nd.rescale, vec![(0.0, 1000.0)]);
    assert!((pq_nd.z_factor - 2.5).abs() < 1e-6);

    let pq_cm = parse_query(Some("colormap=%7B%220%22:[0,0,0,255]%7D&colormap_name=viridis"));
    assert!(pq_cm.errors.is_empty());
    assert!(pq_cm.custom_colormap_json.is_none());
    assert_eq!(pq_cm.colormap, ColormapName::from_str_opt("viridis").unwrap());

    let pq_tj = parse_query(Some("url=a.tif&minzoom=4&maxzoom=14&tile_format=webp"));
    assert!(pq_tj.errors.is_empty());
    assert!(validate_query(&["cog", "WebMercatorQuad", "tilejson.json"], &pq_tj).is_ok());
    assert!(validate_query(&["cog", "tiles", "WebMercatorQuad", "4", "0", "0.png"], &pq_tj).is_err());
    let mut tj = serde_json::json!({"minzoom": 0, "maxzoom": 10, "center": [0.0, 0.0, 0], "tiles": []});
    apply_tilejson_overrides(&mut tj, &pq_tj, "http://host", "cog", "WebMercatorQuad", "url=a.tif&minzoom=4&maxzoom=14&tile_format=webp");
    assert_eq!(tj["minzoom"], 4);
    assert_eq!(tj["maxzoom"], 14);
    assert_eq!(tj["center"][2], 4);
    assert_eq!(tj["tiles"][0], "http://host/cog/tiles/WebMercatorQuad/{z}/{x}/{y}.webp?url=a.tif");

    // HTTP status mapping for upstream 404, local NotFound, and invalid TIFF
    let r404 = error_to_http_response(&EngineError::Io(cogzilla_io::IoError::HttpStatus {
        status: 404,
        url: "https://example.com/missing.tif".into(),
    }));
    assert_eq!(r404.status(), StatusCode::NOT_FOUND);
    let r_enoent = error_to_http_response(&EngineError::Io(cogzilla_io::IoError::Fs(
        std::io::Error::new(std::io::ErrorKind::NotFound, "missing"),
    )));
    assert_eq!(r_enoent.status(), StatusCode::NOT_FOUND);
    let r_tiff = error_to_http_response(&EngineError::Io(cogzilla_io::IoError::Tiff(
        cogzilla_tiff::TiffError::Corrupt("not a tiff"),
    )));
    assert_eq!(r_tiff.status(), StatusCode::BAD_REQUEST);
    for tiff_err in [
        cogzilla_tiff::TiffError::UnsupportedCompression(34712),
        cogzilla_tiff::TiffError::UnsupportedPredictor(99),
        cogzilla_tiff::TiffError::UnsupportedSampleFormat {
            format: 5,
            bits: 128,
        },
        cogzilla_tiff::TiffError::UnsupportedGcpGeoreferencing(2),
    ] {
        let resp = error_to_http_response(&EngineError::Io(cogzilla_io::IoError::Tiff(tiff_err)));
        assert_eq!(resp.status(), StatusCode::NOT_IMPLEMENTED);
    }
    assert_eq!(
        zarr::zarr_error_status(&cogzilla_zarr::ZarrError::UnsupportedCodec {
            chunk_key: "0.0".into(),
            codec: "jpeg2000".into(),
            supported: "gzip, zstd",
        }),
        StatusCode::NOT_IMPLEMENTED
    );
    let r_codec = error_to_http_response(&EngineError::Io(cogzilla_io::IoError::Codec(
        cogzilla_codec::CodecError::Deflate("corrupt deflate payload".into()),
    )));
    assert_eq!(r_codec.status(), StatusCode::BAD_GATEWAY);

    // AlgorithmFactory JSON responses (/algorithms and /algorithms/{id})
    let algs = list_algorithms_json("http://localhost:8080");
    let list = algs["algorithms"].as_array().unwrap();
    assert_eq!(list.len(), 19);
    assert_eq!(list[0]["id"], "hillshade");
    assert_eq!(get_algorithm_json("contours").unwrap()["title"], "Contours");
    assert_eq!(
        list[0]["links"][0]["href"],
        "http://localhost:8080/algorithms/hillshade"
    );
    let hs = get_algorithm_json("hillshade").unwrap();
    assert_eq!(hs["title"], "Hillshade");
    assert_eq!(hs["inputs"]["nbands"], 1);
    assert_eq!(hs["outputs"]["dtype"], "uint8");
    assert_eq!(get_algorithm_json("normalizedIndex").unwrap()["title"], "Normalized Difference Index");
    assert_eq!(get_algorithm_json("grayscale").unwrap()["title"], "Transform a RGB Image to Grayscale");
    assert_eq!(get_algorithm_json("bitonal").unwrap()["title"], "Transform an Image to Bitonal");
    assert!(matches!(
        get_algorithm_json("nonexistent"),
        Err(EngineError::NotFound(_))
    ));

    // WMTSCapabilities.xml generation & XML escaping
    let xml = build_wmts_capabilities(
        "http://localhost:8080",
        "cog",
        "WebMercatorQuad",
        "Test & <Layer>",
        [-122.7, 37.7, -122.5, 37.8],
        8,
        10,
        "webp",
        2,
        "url=https://ex.com/a.tif&rescale=0,100",
        "url=https://ex.com/a.tif&rescale=0,100&tile_format=webp&tile_scale=2",
        false,
    );
    assert!(xml.contains("xmlns=\"http://www.opengis.net/wmts/1.0\""));
    assert!(xml.contains("Test &amp; &lt;Layer&gt;"));
    assert!(xml.contains(
        "http://localhost:8080/cog/tiles/WebMercatorQuad/{TileMatrix}/{TileCol}/{TileRow}@2x.webp?url=https://ex.com/a.tif&amp;rescale=0,100"
    ));
    assert!(xml.contains("<TileWidth>512</TileWidth>"));
    assert!(xml.contains("<ows:Identifier>8</ows:Identifier>"));
    assert!(xml.contains("<ows:Identifier>10</ows:Identifier>"));

    // Vertical colormap PNG (20x256) + orientation/width/height query validation
    let vpng = get_colormap_image("viridis", OutputFormat::Png, true, None, None).unwrap();
    let sized = get_colormap_image("viridis", OutputFormat::Png, false, Some(128), Some(16)).unwrap();
    assert_eq!(&sized[16..20], &128u32.to_be_bytes());
    assert_eq!(&sized[20..24], &16u32.to_be_bytes());
    assert_eq!(&vpng[0..8], b"\x89PNG\r\n\x1a\n");
    let w = u32::from_be_bytes([vpng[16], vpng[17], vpng[18], vpng[19]]);
    let h = u32::from_be_bytes([vpng[20], vpng[21], vpng[22], vpng[23]]);
    assert_eq!((w, h), (20, 256));

    let pq_wh = parse_query(Some("url=a.tif&width=320&height=240"));
    assert!(validate_query(&["cog", "preview.png"], &pq_wh).is_ok());
    assert!(validate_query(&["cog", "preview", "256x256.png"], &pq_wh).is_err());
    assert_eq!(resolve_query_dims(512, 256, Some(200), None), (200, 100));
    assert_eq!(resolve_query_dims(512, 256, None, Some(100)), (200, 100));
    // Unrounded float aspect ratio avoids integer-rounding distortion on small native axes
    // (204.8 / 16.64 * 32 = 393.85 -> 394, whereas rounded 205 / 17 * 32 = 385.88 -> 386).
    assert_eq!(resolve_query_dims_f64(204.8, 16.64, None, Some(32)), (394, 32));

    let pq_ori = parse_query(Some("f=png&orientation=vertical"));
    assert!(validate_query(&["colorMaps", "viridis"], &pq_ori).is_ok());
    assert!(validate_query(&["cog", "info"], &pq_ori).is_err());

    let pq_wmts = parse_query(Some("url=a.tif&minzoom=4&maxzoom=12&tile_format=png&tile_scale=2"));
    assert!(validate_query(&["cog", "WebMercatorQuad", "WMTSCapabilities.xml"], &pq_wmts).is_ok());
    let pq_wms = parse_query(Some("url=https://example.com/a.tif&SERVICE=WMS&REQUEST=GetMap&VERSION=1.3.0&CRS=CRS:84&BBOX=-123,37,-122,38&WIDTH=256&HEIGHT=256&FORMAT=image/png"));
    assert!(pq_wms.errors.is_empty(), "wms errors: {:?}", pq_wms.errors);
    assert!(validate_query(&["cog", "wms"], &pq_wms).is_ok());
    assert_eq!(parse_coord_crs_with_epsg("CRS:84").unwrap().1, 4326);
    assert!(parse_wms_format(Some("image/png"), None, 85).is_ok());
    assert!(parse_wms_format(Some("image/tiff"), None, 85).is_ok());
    assert!(parse_wms_format(Some("image/gif"), None, 85).is_err());
    let cap = build_wms_capabilities("https://example.com", "url=https://example.com/a.tif&SERVICE=WMS&REQUEST=GetCapabilities&VERSION=1.3.0", [-123.0, 37.0, -122.0, 38.0], Some(32610));
    assert!(cap.contains("<WMS_Capabilities version=\"1.3.0\""));
    assert!(cap.contains("<CRS>EPSG:32610</CRS>"));
    assert!(cap.contains("<BoundingBox CRS=\"EPSG:4326\" minx=\"37\" miny=\"-123\" maxx=\"38\" maxy=\"-122\"/>"));
    assert!(cap.contains("xlink:href=\"https://example.com/cog/wms?url=https://example.com/a.tif&amp;\""), "OnlineResource should strip WMS protocol params and end with &: {cap}");
    assert!(!cap.contains("REQUEST=GetCapabilities"));
    assert!(parse_wms_format(Some("IMAGE/PNG"), None, 85).is_ok());
    assert!(validate_query(&["mosaicjson", "WorldCRS84Quad", "WMTSCapabilities.xml"], &pq_wmts).is_ok());
    assert!(validate_query(&["mosaicjson", "NotATms", "WMTSCapabilities.xml"], &pq_wmts).is_err());
}

#[test]
fn geospatial_bounds_reproject_and_user_defined_crs_parity() {
    // 1. reproject_wgs84_bounds on global COG into UTM31 (EPSG:32631), NZTM (EPSG:2193),
    //    and LAEA (EPSG:3035) must include interior points (Rome/Berlin/NZ/Europe), not just edges.
    let global = [-180.0, -85.0, 180.0, 85.0];
    let utm31 = cogzilla_proj::CrsModel::from_epsg(32631).unwrap();
    let utm_b = reproject_wgs84_bounds(global, &utm31).unwrap();
    let (rome_x, rome_y) = utm31.from_lonlat_deg(12.5, 41.9);
    assert!(
        rome_x >= utm_b[0] && rome_x <= utm_b[2] && rome_y >= utm_b[1] && rome_y <= utm_b[3],
        "Rome ({rome_x}, {rome_y}) outside projected UTM31 bounds {utm_b:?}"
    );

    let nztm = cogzilla_proj::CrsModel::from_epsg(2193).unwrap();
    let nztm_b = reproject_wgs84_bounds(global, &nztm).unwrap();
    let (nz_x, nz_y) = nztm.from_lonlat_deg(170.5, -43.5);
    assert!(
        nz_x >= nztm_b[0] && nz_x <= nztm_b[2] && nz_y >= nztm_b[1] && nz_y <= nztm_b[3],
        "NZ ({nz_x}, {nz_y}) outside projected NZTM bounds {nztm_b:?}"
    );

    let laea = cogzilla_proj::CrsModel::from_epsg(3035).unwrap();
    let laea_b = reproject_wgs84_bounds([-180.0, -90.0, 180.0, 90.0], &laea).unwrap();
    let (ber_x, ber_y) = laea.from_lonlat_deg(13.4, 52.5);
    assert!(
        ber_x >= laea_b[0] && ber_x <= laea_b[2] && ber_y >= laea_b[1] && ber_y <= laea_b[3],
        "Berlin ({ber_x}, {ber_y}) outside projected LAEA bounds {laea_b:?}"
    );

    // 2. cog_standard_epsg returns None for user-defined projected CRS (32767) even with geographic_type=4326
    let mut gk = cogzilla_tiff::GeoKeys {
        model_type: Some(1),
        projected_cs_type: Some(32767),
        geographic_type: Some(4326),
        ..cogzilla_tiff::GeoKeys::default()
    };
    assert_eq!(cog_standard_epsg(&gk), None);

    // Standard projected EPSG returns Some(32610)
    gk.projected_cs_type = Some(32610);
    assert_eq!(cog_standard_epsg(&gk), Some(32610));

    // Pure geographic returns Some(4326)
    let gk_geo = cogzilla_tiff::GeoKeys {
        model_type: Some(2),
        geographic_type: Some(4326),
        ..cogzilla_tiff::GeoKeys::default()
    };
    assert_eq!(cog_standard_epsg(&gk_geo), Some(4326));

    // 3. GEOTIFF_CONTENT_TYPE matches parse_ext_format("tif", ...)
    let (_, tif_ct) = parse_ext_format("tif", None, 85).unwrap();
    assert_eq!(tif_ct, GEOTIFF_CONTENT_TYPE);

    // 4. PolygonMask::check_work_budget rejects O(W*H*V) DoS payloads
    let ring: Vec<serde_json::Value> = (0..10_000)
        .map(|i| {
            let a = (i as f64) * std::f64::consts::TAU / 10_000.0;
            serde_json::json!([a.cos(), a.sin()])
        })
        .collect();
    let poly_json = serde_json::json!({
        "type": "Polygon",
        "coordinates": [ring],
    });
    let (pm, _) = PolygonMask::from_geojson(&poly_json).unwrap();
    assert!(pm.check_work_budget(256 * 256).is_err());
    assert!(pm.check_work_budget(64 * 64).is_ok());

    // 5. ?assets=visual|bidx=2&assets=nir stores per-asset bidx without leaking into global bidx
    let pq_ab = parse_query(Some("assets=visual|bidx=2&assets=nir&asset_as_band=true"));
    assert_eq!(pq_ab.assets, vec!["visual", "nir"]);
    assert!(pq_ab.bidx.is_empty());
    assert_eq!(pq_ab.asset_bidx, vec![("visual".to_string(), vec![2])]);
    assert!(pq_ab.asset_as_band);

    // 6. CrsModel::linear_unit_metres returns US survey foot for EPSG:2227 and 1.0 for EPSG:32610
    let ca_ft = cogzilla_proj::CrsModel::from_epsg(2227).unwrap();
    assert!((ca_ft.linear_unit_metres() - 1200.0 / 3937.0).abs() < 1e-6);
    let utm10 = cogzilla_proj::CrsModel::from_epsg(32610).unwrap();
    assert!((utm10.linear_unit_metres() - 1.0).abs() < 1e-9);

    // 7. resolve_bbox_dst_crs defaults to coord_crs when dst_crs is omitted
    let pq_utm = parse_query(Some("url=a.tif&coord_crs=EPSG:32610"));
    assert_eq!(resolve_bbox_dst_crs(&pq_utm).map(|(_, e)| e), Some(32610));

    // 8. Form-encoded comparison & semicolon expressions decode `+` as space
    let pq_cmp = parse_query(Some("expression=b1+%3E+100"));
    assert_eq!(pq_cmp.expr_raw.as_deref(), Some("b1 > 100"));
    let pq_semi = parse_query(Some("expression=b1%3B+b2%3B+b3"));
    assert_eq!(pq_semi.expr_raw.as_deref(), Some("b1; b2; b3"));

    // 9. resolve_zoom_range honors ?maxzoom < def_min and WMTS tif MIME type
    assert_eq!(resolve_zoom_range(10, 15, None, Some(5)), (5, 5));
    let tif_xml = build_wmts_capabilities(
        "http://localhost:8080",
        "cog",
        "WebMercatorQuad",
        "Test",
        [-122.7, 37.7, -122.5, 37.8],
        5,
        5,
        "tif",
        1,
        "url=a.tif",
        "url=a.tif&tile_format=tif",
        false,
    );
    assert!(tif_xml.contains("<Format>image/tiff; application=geotiff</Format>"));

    // 10. /{family}/map.html must reject OGC Maps-only width/dst_crs params
    let pq_map_html = parse_query(Some("url=a.tif&width=256"));
    assert!(validate_query(&["cog", "map.html"], &pq_map_html).is_err());
    assert!(validate_query(&["cog", "map"], &pq_map_html).is_ok());

    // 11. Antimeridian-crossing WGS84 bbox (w > e, Fiji) projects across 180°, not 0°
    let utm01s = cogzilla_proj::CrsModel::from_epsg(32701).unwrap();
    let fiji_b = reproject_wgs84_bounds([177.0, -20.0, -178.0, -16.0], &utm01s).unwrap();
    let (fiji_x, fiji_y) = utm01s.from_lonlat_deg(179.5, -18.0);
    assert!(
        fiji_x >= fiji_b[0] && fiji_x <= fiji_b[2] && fiji_y >= fiji_b[1] && fiji_y <= fiji_b[3],
        "Fiji ({fiji_x}, {fiji_y}) outside antimeridian projected bounds {fiji_b:?}"
    );
    assert!(
        (fiji_b[2] - fiji_b[0]) < 1_000_000.0,
        "antimeridian bounds swept wrong hemisphere: {fiji_b:?}"
    );

    // 12. normalize_render_bbox unwraps antimeridian Geographic bboxes & rejects inverted boxes
    let fiji_geo = normalize_render_bbox(
        [177.0, -20.0, -178.0, -16.0],
        &cogzilla_proj::CrsModel::Geographic,
    )
    .unwrap();
    assert_eq!(fiji_geo, [177.0, -20.0, 182.0, -16.0]);
    assert_eq!(aspect_preserving_dims(fiji_geo, 512), (512, 410));
    assert!(normalize_render_bbox(
        [0.0, 10.0, 1.0, -10.0],
        &cogzilla_proj::CrsModel::Geographic,
    )
    .is_err());
    assert!(normalize_render_bbox(
        [10.0, 0.0, 5.0, 1.0],
        &cogzilla_proj::CrsModel::Epsg3857,
    )
    .is_err());

    // 13. bbox_to_wgs84 unwraps antimeridian-crossing projected bboxes instead of inverting
    let utm60n = cogzilla_proj::CrsModel::from_epsg(32660).unwrap();
    let am_wgs = bbox_to_wgs84([800_000.0, 0.0, 900_000.0, 100_000.0], Some(&utm60n)).unwrap();
    assert!(
        am_wgs[0] > 179.0 && am_wgs[2] > 180.0 && (am_wgs[2] - am_wgs[0]) < 2.0,
        "antimeridian UTM60N bbox inverted to complement: {am_wgs:?}"
    );

    // 14. reproject_wgs84_bounds into UTM on a global [-180,-90,180,90] COG covers >67.5°N
    let utm31_global = reproject_wgs84_bounds([-180.0, -90.0, 180.0, 90.0], &utm31).unwrap();
    let (_, sval_y) = utm31.from_lonlat_deg(3.0, 78.0);
    assert!(
        sval_y <= utm31_global[3],
        "global UTM31 bounds truncated below 78°N ({sval_y} > {})",
        utm31_global[3]
    );

    // 15. Buffered CDB1GlobalGrid @4x tile exceeding 4096 px must return 400 (not silent clamp)
    let pq_cdb_buf = parse_query(Some("url=a.tif&buffer=64"));
    assert!(build_tile_request_for_tms(
        None,
        TileCoord { z: 0, x: 0, y: 0 },
        Some(4),
        Some("png"),
        &pq_cdb_buf,
        1,
        cogzilla_raster::SampleType::Uint8,
        "CDB1GlobalGrid",
        None,
        true,
    )
    .is_err());

    // 16. bbox_to_wgs84 preserves full [-180, 180] on global EPSG:3857 and polar EPSG:3413,
    // and reproject_wgs84_bounds rejects polar-cap bboxes wholly outside TM's ±89.35° domain
    let m = 20_037_508.342_789_244;
    let glob_3857 =
        bbox_to_wgs84([-m, -m, m, m], Some(&cogzilla_proj::CrsModel::Epsg3857)).unwrap();
    assert!(
        (glob_3857[0] + 180.0).abs() < 1e-6 && (glob_3857[2] - 180.0).abs() < 1e-6,
        "global EPSG:3857 bbox must span [-180, 180], got {glob_3857:?}"
    );
    let ps_north = cogzilla_proj::CrsModel::from_epsg(3413).unwrap();
    let polar_wgs =
        bbox_to_wgs84([-2_000_000.0, -2_000_000.0, 2_000_000.0, 2_000_000.0], Some(&ps_north))
            .unwrap();
    assert_eq!(polar_wgs[0], -180.0);
    assert_eq!(polar_wgs[2], 180.0);
    assert_eq!(polar_wgs[3], 90.0);
    assert!(reproject_wgs84_bounds([0.0, 89.5, 10.0, 90.0], &utm31).is_err());

    // 17. Narrow polar strip in Cylindrical Equal Area (EPSG:6933) must NOT expand to [-180, 180]
    let cea = cogzilla_proj::CrsModel::from_epsg(6933).unwrap();
    let cea_strip = bbox_to_wgs84([-100_000.0, 7_000_000.0, 100_000.0, 7_400_000.0], Some(&cea))
        .unwrap();
    assert!(
        cea_strip[2] - cea_strip[0] < 10.0,
        "narrow EPSG:6933 strip must not expand to [-180, 180], got {cea_strip:?}"
    );

    // 18. wgs84_bboxes_intersect handles regional 0..360 and -360..0 bboxes without false positives
    assert!(!wgs84_bboxes_intersect(
        [235.0, 32.0, 245.0, 42.0],
        [-170.0, 35.0, -160.0, 40.0]
    ));
    assert!(wgs84_bboxes_intersect(
        [235.0, 32.0, 245.0, 42.0],
        [-122.0, 35.0, -118.0, 40.0]
    ));
    assert!(!wgs84_bboxes_intersect(
        [-245.0, 32.0, -235.0, 42.0],
        [160.0, 35.0, 170.0, 40.0]
    ));
    assert!(wgs84_bboxes_intersect(
        [-245.0, 32.0, -235.0, 42.0],
        [118.0, 35.0, 122.0, 40.0]
    ));

    // 19. OGC tileset & STAC render links must never double X-Forwarded-Prefix
    let tms = crate::tms::lookup_tms("WebMercatorQuad").unwrap();
    let ts_json = build_ogc_tileset_json(
        "https://api.example.com/prod",
        "/prod",
        "cog",
        tms,
        "url=a.tif",
        Some([-122.5, 37.5, -122.0, 38.0]),
        Some(0),
        Some(12),
    );
    let ts_str = ts_json.to_string();
    assert!(
        !ts_str.contains("/prod/prod"),
        "OGC tileset links doubled X-Forwarded-Prefix: {ts_str}"
    );
    assert!(ts_str.contains("https://api.example.com/prod/cog/"));
}

#[tokio::test]
async fn sweep22_body_limits_timeout_and_feature_collection_cap() {
    // BB1: default FeatureCollection limit allows thousands of features (not capped at 64)
    assert!(check_feature_collection_len(200).is_ok());
    assert!(check_feature_collection_len(1_000).is_ok());
    assert!(check_feature_collection_len(crate::config::DEFAULT_MAX_STATISTICS_FEATURES).is_ok());
    assert!(check_feature_collection_len(crate::config::DEFAULT_MAX_STATISTICS_FEATURES + 1).is_err());

    // BB2: oversized POST body returns 413 Content Too Large with exact message
    let big = Bytes::from(vec![b'a'; MAX_POST_BODY_BYTES + 1]);
    let err = read_limited_body_with_timeout(http_body_util::Full::new(big), 5)
        .await
        .unwrap_err();
    let resp = error_to_http_response(&err);
    assert_eq!(resp.status(), StatusCode::PAYLOAD_TOO_LARGE);
    let msg = err.to_string();
    assert!(
        msg.contains("request body exceeds 16 MiB limit"),
        "unexpected 413 message: {msg}"
    );

    // BB3: stalled POST body returns 408 Request Timeout
    struct NeverBody;
    impl hyper::body::Body for NeverBody {
        type Data = Bytes;
        type Error = std::io::Error;
        fn poll_frame(
            self: std::pin::Pin<&mut Self>,
            _cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Option<Result<hyper::body::Frame<Self::Data>, Self::Error>>>
        {
            std::task::Poll::Pending
        }
    }
    let err_to = read_limited_body_with_timeout(NeverBody, 1)
        .await
        .unwrap_err();
    let resp_to = error_to_http_response(&err_to);
    assert_eq!(resp_to.status(), StatusCode::REQUEST_TIMEOUT);
}

#[test]
fn ogc_tileset_filters_template_query_and_skips_negative_cdb_matrices_and_stac_renders_encodes() {
    // 1. OGC tileset strips minzoom/maxzoom/crs from item href and skips CDB1GlobalGrid negative matrices.
    let cdb = crate::tms::lookup_tms("CDB1GlobalGrid").unwrap();
    let ts = build_ogc_tileset_json(
        "example.com",
        "",
        "cog",
        cdb,
        "url=a.tif&minzoom=0&maxzoom=2&crs=EPSG:4326&bidx=1",
        Some([-10.0, -10.0, 10.0, 10.0]),
        Some(0),
        Some(2),
    );
    let limits = ts["tileMatrixSetLimits"].as_array().unwrap();
    assert!(
        limits.iter().all(|l| !l["tileMatrix"].as_str().unwrap().starts_with('-')),
        "CDB1GlobalGrid must not advertise negative matrix IDs: {limits:?}"
    );
    let item_href = ts["links"]
        .as_array()
        .unwrap()
        .iter()
        .find(|l| l["rel"] == "item")
        .unwrap()["href"]
        .as_str()
        .unwrap();
    assert!(item_href.contains("url=a.tif") && item_href.contains("bidx=1"));
    assert!(!item_href.contains("minzoom=") && !item_href.contains("maxzoom=") && !item_href.contains("crs="));

    // 2. STAC render links percent-encode presigned URLs, render IDs, and inline object colormaps.
    let render = serde_json::json!({
        "assets": ["red"],
        "colormap": {"1": [255, 0, 0, 255]}
    });
    let entry = build_stac_render_entry(
        "example.com",
        "",
        "https://s3.example/item.json?a=1&b=2",
        "true color",
        &render,
    );
    let links = entry["links"].as_array().unwrap();
    let self_href = links[0]["href"].as_str().unwrap();
    let tj_href = links[1]["href"].as_str().unwrap();
    assert!(self_href.contains("/stac/renders/true%20color?url=https%3A%2F%2Fs3.example%2Fitem.json%3Fa%3D1%26b%3D2"), "{self_href}");
    assert!(tj_href.contains("colormap=%7B%221%22%3A%5B255%2C0%2C0%2C255%5D%7D"), "{tj_href}");
}

#[test]
fn audit87_header_accepts_gzip_rfc9110_precedence_and_nan() {
    assert_eq!(negotiate_encoding(["gzip;q=0, *"]), PreferredEncoding::Brotli);
    assert_eq!(negotiate_encoding(["*, gzip;q=0"]), PreferredEncoding::Brotli);
    assert_eq!(negotiate_encoding(["x-gzip;q=0, *;q=1"]), PreferredEncoding::Brotli);
    assert_eq!(negotiate_encoding(["gzip;q=NaN"]), PreferredEncoding::None);
    assert_eq!(negotiate_encoding(["gzip;q=inf"]), PreferredEncoding::None);
    assert_eq!(negotiate_encoding(["gzip;q=-0.5"]), PreferredEncoding::None);
    assert_eq!(negotiate_encoding(["br, deflate"]), PreferredEncoding::Brotli);
    assert_eq!(negotiate_encoding(["br, *;q=0.5"]), PreferredEncoding::Brotli);
    assert_eq!(negotiate_encoding(["x-gzip;q=0.1"]), PreferredEncoding::Gzip);
    assert_eq!(negotiate_encoding(["gzip, deflate"]), PreferredEncoding::Gzip);
}

#[test]
fn audit88_attach_x_failed_assets_redacts_credentials_and_presigned_queries() {
    let mut resp = Response::builder()
        .status(StatusCode::OK)
        .body(Full::new(Bytes::new()))
        .unwrap();
    attach_x_failed_assets(
        &mut resp,
        &[
            "https://user:secret@s3.example.com/bucket/a.tif?X-Amz-Signature=deadbeef".into(),
            "/local/ok.tif".into(),
        ],
    );
    let hv = resp.headers().get("x-failed-assets").unwrap().to_str().unwrap();
    assert!(!hv.contains("secret") && !hv.contains("deadbeef"), "leaked creds in x-failed-assets: {hv}");
    assert!(hv.contains("https://s3.example.com/bucket/a.tif") && hv.contains("/local/ok.tif"), "{hv}");
}

#[test]
fn audit91_native_bbox_dims_aspect_antimeridian_and_0_360() {
    let make_ifd = |w: u32, h: u32, a: f64, e: f64, c: f64, f: f64| {
        let p2c = cogzilla_raster::Affine2D { a, b: 0.0, c, d: 0.0, e, f };
        cogzilla_tiff::IfdMeta {
            width: w,
            height: h,
            tile_width: 256,
            tile_height: 256,
            tiles_across: w.div_ceil(256),
            tiles_down: h.div_ceil(256),
            sample_type: cogzilla_raster::SampleType::Uint8,
            samples_per_pixel: 1,
            planar_config: cogzilla_tiff::PlanarConfig::Chunky,
            compression: cogzilla_tiff::Compression::None,
            predictor: cogzilla_tiff::Predictor::None,
            photometric: 1,
            extra_samples: Vec::new(),
            jpeg_tables: None,
            mask_ifd: None,
            is_stripped: false,
            pixel_to_crs: p2c,
            crs_to_pixel: p2c.invert().unwrap(),
            gcp_poly: None,
            tile_offsets: Box::default(),
            tile_byte_counts: Box::default(),
        }
    };
    let geo = cogzilla_proj::CrsModel::Geographic;
    let merc = cogzilla_proj::CrsModel::Epsg3857;

    // 1. Large 2:1 global COG preserves 2:1 aspect ratio at the 4096 cap (4096x2048, not 4096x4096).
    let ifd_2to1 = make_ifd(21600, 10800, 360.0 / 21600.0, -180.0 / 10800.0, -180.0, 90.0);
    let dims = native_bbox_dims_in_crs(&ifd_2to1, &geo, &geo, [-180.0, -90.0, 180.0, 90.0], [-180.0, -90.0, 180.0, 90.0]);
    assert_eq!(dims, (4096, 2048));

    // 2. Global 0..360° Geographic COG (1440x720) keeps 1440x720 in Geographic and does not collapse ddx in 3857.
    let ifd_360 = make_ifd(1440, 720, 0.25, -0.25, 0.0, 90.0);
    assert_eq!(
        native_bbox_dims_in_crs(&ifd_360, &geo, &geo, [-180.0, -90.0, 180.0, 90.0], [-180.0, -90.0, 180.0, 90.0]),
        (1440, 720)
    );
    let merc_bb = [-20_037_508.342_789_244, -20_037_508.342_789_244, 20_037_508.342_789_244, 20_037_508.342_789_244];
    let (mw, mh) = native_bbox_dims_in_crs(&ifd_360, &geo, &merc, [-180.0, -85.05, 180.0, 85.05], merc_bb);
    assert!((1100..=1500).contains(&mw) && (1100..=1500).contains(&mh), "got ({mw}, {mh})");

    // 3. Global 1024x1024 EPSG:3857 -> EPSG:4326 does NOT falsely wrap ddx (stays ~1309x619, not 3065x1448).
    let span = 2.0 * 20_037_508.342_789_244;
    let ifd_merc = make_ifd(1024, 1024, span / 1024.0, -span / 1024.0, -20_037_508.342_789_244, 20_037_508.342_789_244);
    let (gw, gh) = native_bbox_dims_in_crs(&ifd_merc, &merc, &geo, [-180.0, -85.05, 180.0, 85.05], [-180.0, -85.05, 180.0, 85.05]);
    assert!((1200..=1400).contains(&gw) && (550..=700).contains(&gh), "got ({gw}, {gh})");

    // 4. Antimeridian-crossing UTM 60N (179°E..179°W) -> EPSG:4326 wraps ddx properly (~512x512, not 4x4).
    let utm60 = cogzilla_proj::CrsModel::from_epsg(32660).unwrap();
    let (x0, y0) = utm60.from_lonlat_deg(179.0, 1.0);
    let (x1, y1) = utm60.from_lonlat_deg(-179.0, -1.0);
    let ifd_am = make_ifd(512, 512, (x1 - x0) / 512.0, (y1 - y0) / 512.0, x0, y0);
    let (aw, ah) = native_bbox_dims_in_crs(&ifd_am, &utm60, &geo, [179.0, -1.0, 181.0, 1.0], [179.0, -1.0, 181.0, 1.0]);
    assert!((450..=580).contains(&aw) && (450..=580).contains(&ah), "got ({aw}, {ah})");
}

#[test]
fn attach_x_assets_redacts_secrets_and_preserves_plain_names() {
    let mut resp = Response::builder()
        .status(StatusCode::OK)
        .body(Full::new(Bytes::new()))
        .unwrap();
    attach_x_assets(
        &mut resp,
        &[
            "https://user:secret@example.com/a.tif?X-Amz-Signature=deadbeef&sig=123".to_string(),
            "red".to_string(),
            "B04".to_string(),
        ],
    );
    let hv = resp.headers().get("x-assets").unwrap().to_str().unwrap();
    assert_eq!(hv, "https://example.com/a.tif,red,B04");
    assert!(!hv.contains("secret") && !hv.contains("deadbeef"));
}

#[tokio::test]
async fn forwarded_prefix_rejects_protocol_relative_and_invalid_host() {
    let engine = Arc::new(TileEngine::new(4 * 1024 * 1024));
    let req = Request::builder()
        .method(Method::GET)
        .uri("/docs")
        .header("x-forwarded-prefix", "//evil.com")
        .header("x-forwarded-host", "attacker@evil.com/bad")
        .header("host", "127.0.0.1:8080")
        .body(Full::new(Bytes::new()))
        .unwrap();
    let resp = handle_request(engine, req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = http_body_util::BodyExt::collect(resp.into_body())
        .await
        .unwrap()
        .to_bytes();
    let html = std::str::from_utf8(&body).unwrap();
    assert!(!html.contains("//evil.com"), "protocol-relative prefix leaked into HTML: {html}");
    assert!(!html.contains("attacker@"), "invalid forwarded host leaked into HTML: {html}");
}

#[tokio::test]
async fn sweep45_qq1_colormaps_html_npy_and_rr1_accept_negotiation_vary() {
    let engine = Arc::new(TileEngine::new(4 * 1024 * 1024));
    let browser_accept = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8";

    // QQ1: /colorMaps/viridis?f=html -> 200 text/html + Vary: Accept
    let req = Request::builder()
        .method(Method::GET)
        .uri("/colorMaps/viridis?f=html")
        .body(Full::new(Bytes::new()))
        .unwrap();
    let resp = handle_request(engine.clone(), req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert!(resp.headers().get(CONTENT_TYPE).unwrap().to_str().unwrap().starts_with("text/html"));
    let vary = resp.headers().get(hyper::header::VARY).unwrap().to_str().unwrap().to_ascii_lowercase();
    assert!(vary.contains("accept"), "missing Vary: Accept: {vary}");

    // QQ1: /colorMaps/viridis?f=npy -> 200 (4, 20, 256) uint8 NumPy array
    let req = Request::builder()
        .method(Method::GET)
        .uri("/colorMaps/viridis?f=npy")
        .body(Full::new(Bytes::new()))
        .unwrap();
    let resp = handle_request(engine.clone(), req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let body = http_body_util::BodyExt::collect(resp.into_body()).await.unwrap().to_bytes();
    assert_eq!(&body[0..6], b"\x93NUMPY");
    let hlen = u16::from_le_bytes([body[8], body[9]]) as usize;
    let hdr = std::str::from_utf8(&body[10..10 + hlen]).unwrap();
    assert!(hdr.contains("'|u1'") && hdr.contains("(4, 20, 256)"), "unexpected npy header: {hdr}");
    assert_eq!(body.len(), 10 + hlen + 4 * 20 * 256);

    // RR1: browser Accept -> text/html + Vary: Accept on all offers_html routes
    for uri in [
        "/tileMatrixSets",
        "/tileMatrixSets/WebMercatorQuad",
        "/conformance",
        "/algorithms",
        "/colorMaps/viridis",
    ] {
        let req = Request::builder()
            .method(Method::GET)
            .uri(uri)
            .header(hyper::header::ACCEPT, browser_accept)
            .body(Full::new(Bytes::new()))
            .unwrap();
        let resp = handle_request(engine.clone(), req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK, "{uri}");
        assert!(
            resp.headers().get(CONTENT_TYPE).unwrap().to_str().unwrap().starts_with("text/html"),
            "expected text/html for {uri}"
        );
        let vary = resp.headers().get(hyper::header::VARY).unwrap().to_str().unwrap().to_ascii_lowercase();
        assert!(vary.contains("accept"), "missing Vary: Accept for {uri}: {vary}");
    }

    // RR1: explicit ?f=json wins over browser Accept, and still carries Vary: Accept
    let req = Request::builder()
        .method(Method::GET)
        .uri("/tileMatrixSets?f=json")
        .header(hyper::header::ACCEPT, browser_accept)
        .body(Full::new(Bytes::new()))
        .unwrap();
    let resp = handle_request(engine.clone(), req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert!(resp.headers().get(CONTENT_TYPE).unwrap().to_str().unwrap().starts_with("application/json"));
    let vary = resp.headers().get(hyper::header::VARY).unwrap().to_str().unwrap().to_ascii_lowercase();
    assert!(vary.contains("accept"), "missing Vary: Accept on ?f=json: {vary}");

    // RR1: API client (no Accept or Accept: */*) gets application/json + Vary: Accept
    let req = Request::builder()
        .method(Method::GET)
        .uri("/tileMatrixSets")
        .header(hyper::header::ACCEPT, "*/*")
        .body(Full::new(Bytes::new()))
        .unwrap();
    let resp = handle_request(engine, req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert!(resp.headers().get(CONTENT_TYPE).unwrap().to_str().unwrap().starts_with("application/json"));
    let vary = resp.headers().get(hyper::header::VARY).unwrap().to_str().unwrap().to_ascii_lowercase();
    assert!(vary.contains("accept"), "missing Vary: Accept on */*: {vary}");

    // Strict boolean query parameter validation (rejects bogus booleans with 422).
    for q in [
        "return_mask=bogus",
        "unscale=maybe",
        "strict=nope",
        "categorical=2",
        "area_weighted=invalid",
        "use_epsg=nah",
        "asset_as_band=wat",
    ] {
        let pq = parse_query(Some(q));
        assert!(!pq.errors.is_empty(), "expected validation error for {q}");
    }
    assert!(parse_query(Some("return_mask=false&unscale=1&categorical=yes")).errors.is_empty());
}

#[tokio::test]
async fn readyz_and_graceful_shutdown_drain() {
    let engine = Arc::new(TileEngine::new(8 * 1024 * 1024));
    let req = Request::builder()
        .method(Method::GET)
        .uri("/readyz")
        .body(Full::new(Bytes::new()))
        .unwrap();
    let resp = handle_request(engine.clone(), req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    assert_eq!(
        resp.headers().get(CACHE_CONTROL).and_then(|v| v.to_str().ok()),
        Some("no-store")
    );

    // HEAD /readyz -> 200 with no-store and empty body; POST /readyz -> 405.
    let head_req = Request::builder()
        .method(Method::HEAD)
        .uri("/readyz")
        .body(Full::new(Bytes::new()))
        .unwrap();
    let head_resp = handle_request(engine.clone(), head_req).await.unwrap();
    assert_eq!(head_resp.status(), StatusCode::OK);
    assert_eq!(
        head_resp.headers().get(CACHE_CONTROL).and_then(|v| v.to_str().ok()),
        Some("no-store")
    );
    let post_req = Request::builder()
        .method(Method::POST)
        .uri("/readyz")
        .body(Full::new(Bytes::new()))
        .unwrap();
    let post_resp = handle_request(engine.clone(), post_req).await.unwrap();
    assert_eq!(post_resp.status(), StatusCode::METHOD_NOT_ALLOWED);

    // Unreachable pgSTAC -> 503 with Cache-Control: no-store.
    let dead_port = TcpListener::bind("127.0.0.1:0").await.unwrap().local_addr().unwrap().port();
    let pg_cfg = cogzilla_pgstac::PgStacConfig::from_connection_string(&format!(
        "postgresql://u:p@127.0.0.1:{dead_port}/db?connect_timeout=1"
    ))
    .unwrap();
    let pg = Arc::new(cogzilla_pgstac::PgStac::new(pg_cfg).unwrap());
    let pg_engine = Arc::new(TileEngine::new(8 * 1024 * 1024));
    pg_engine.set_pgstac(pg);
    let unready_req = Request::builder()
        .method(Method::GET)
        .uri("/readyz")
        .body(Full::new(Bytes::new()))
        .unwrap();
    let unready_resp = handle_request(pg_engine, unready_req).await.unwrap();
    assert_eq!(unready_resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        unready_resp.headers().get(CACHE_CONTROL).and_then(|v| v.to_str().ok()),
        Some("no-store")
    );

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    let srv = tokio::spawn(async move {
        serve_listener_with_shutdown(engine, listener, async move {
            let _ = rx.await;
        })
        .await
        .unwrap();
    });

    let client = reqwest::Client::new();
    let r = client.get(format!("http://{addr}/readyz")).send().await.unwrap();
    assert_eq!(r.status().as_u16(), 200);
    assert_eq!(r.text().await.unwrap(), "ok");

    let _ = tx.send(());
    tokio::time::timeout(std::time::Duration::from_secs(5), srv)
        .await
        .expect("server should drain and exit promptly")
        .unwrap();
}

#[test]
fn negotiate_encoding_prefers_brotli_and_honours_rfc9110_q_values() {
    assert_eq!(negotiate_encoding(["gzip, deflate, br"]), PreferredEncoding::Brotli);
    assert_eq!(negotiate_encoding(["br"]), PreferredEncoding::Brotli);
    assert_eq!(negotiate_encoding(["gzip, deflate"]), PreferredEncoding::Gzip);
    assert_eq!(negotiate_encoding(["br;q=0, gzip"]), PreferredEncoding::Gzip);
    assert_eq!(negotiate_encoding(["gzip;q=1.0, br;q=0.5"]), PreferredEncoding::Gzip);
    assert_eq!(negotiate_encoding(["gzip;q=0.5, br;q=0.8"]), PreferredEncoding::Brotli);
    assert_eq!(negotiate_encoding(["gzip;q=0.5, *"]), PreferredEncoding::Brotli);
    assert_eq!(negotiate_encoding(["br;q=0.5, *"]), PreferredEncoding::Gzip);
    assert_eq!(negotiate_encoding(["*"]), PreferredEncoding::Gzip);
    assert_eq!(negotiate_encoding(["gzip;q=0, *"]), PreferredEncoding::Brotli);
    assert_eq!(negotiate_encoding(["gzip;q=0, br;q=0, *"]), PreferredEncoding::None);
    assert_eq!(negotiate_encoding(["identity;q=1.0, br;q=0.5"]), PreferredEncoding::None);
    assert_eq!(negotiate_encoding(["br;q=NaN"]), PreferredEncoding::None);
    // Multi-header HTTP/1.1 & HTTP/2 Accept-Encoding lines
    assert_eq!(negotiate_encoding(["gzip", "br"]), PreferredEncoding::Brotli);
    assert_eq!(negotiate_encoding(["*", "gzip;q=0", "br;q=0"]), PreferredEncoding::None);
}

#[tokio::test]
async fn x_request_id_echoes_valid_generates_when_absent_or_invalid_and_on_429() {
    let engine = Arc::new(TileEngine::new(8 * 1024 * 1024));

    // Absent -> generated 21-byte <8-hex>-<12-hex>, unique across calls.
    let r1 = handle_request(
        engine.clone(),
        Request::builder().uri("/healthz").body(Full::new(Bytes::new())).unwrap(),
    )
    .await
    .unwrap();
    let r2 = handle_request(
        engine.clone(),
        Request::builder().uri("/healthz").body(Full::new(Bytes::new())).unwrap(),
    )
    .await
    .unwrap();
    let id1 = r1.headers().get(&X_REQUEST_ID).unwrap().to_str().unwrap();
    let id2 = r2.headers().get(&X_REQUEST_ID).unwrap().to_str().unwrap();
    assert_eq!(id1.len(), 21);
    assert_eq!(id1.as_bytes()[8], b'-');
    assert!(id1[..8].bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()));
    assert!(id1[9..].bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()));
    assert_ne!(id1, id2);
    assert_eq!(
        r1.headers()
            .get(hyper::header::ACCESS_CONTROL_EXPOSE_HEADERS)
            .and_then(|v| v.to_str().ok()),
        Some("X-Request-Id")
    );

    // Valid caller-supplied X-Request-Id (including exact 128-byte boundary) is echoed verbatim.
    let exact_128 = "a".repeat(128);
    for valid in ["trace-abc-123_XYZ", exact_128.as_str()] {
        let r_echo = handle_request(
            engine.clone(),
            Request::builder()
                .uri("/healthz")
                .header("x-request-id", valid)
                .body(Full::new(Bytes::new()))
                .unwrap(),
        )
        .await
        .unwrap();
        assert_eq!(
            r_echo.headers().get(&X_REQUEST_ID).unwrap().to_str().unwrap(),
            valid
        );
    }

    // Invalid (empty, contains whitespace/tab, or >128 bytes) -> replaced with fresh generated ID.
    for invalid in ["", "bad id with spaces", "bad\tid"] {
        let r_bad = handle_request(
            engine.clone(),
            Request::builder()
                .uri("/healthz")
                .header("x-request-id", invalid)
                .body(Full::new(Bytes::new()))
                .unwrap(),
        )
        .await
        .unwrap();
        let bad_out = r_bad.headers().get(&X_REQUEST_ID).unwrap().to_str().unwrap();
        assert_ne!(bad_out, invalid);
        assert_eq!(bad_out.len(), 21);
    }

    let too_long = "a".repeat(129);
    let r_long = handle_request(
        engine.clone(),
        Request::builder()
            .uri("/healthz")
            .header("x-request-id", &too_long)
            .body(Full::new(Bytes::new()))
            .unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(r_long.headers().get(&X_REQUEST_ID).unwrap().len(), 21);

    // AIMD 429 shedding still carries X-Request-Id.
    let tight = Arc::new(
        TileEngine::new(8 * 1024 * 1024)
            .with_aimd(crate::aimd::AimdLimiter::new(true, 1, 1, 1, 150)),
    );
    let _held = tight.aimd.try_acquire().unwrap();
    let r_429 = handle_request(
        tight.clone(),
        Request::builder()
            .uri("/cog/info")
            .header("x-request-id", "shed-req-42")
            .body(Full::new(Bytes::new()))
            .unwrap(),
    )
    .await
    .unwrap();
    assert_eq!(r_429.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(
        r_429.headers().get(&X_REQUEST_ID).unwrap().to_str().unwrap(),
        "shed-req-42"
    );
}

#[test]
fn a_cast_to_float_over_band_math_keeps_the_band_math_s_range() {
    use cogzilla_raster::SampleType as T;
    // No integer range to cast into: the picture spans what the band math gives, as
    // without the expression.
    let pq = parse_query(Some("expression=b1&algorithm=cast&algorithm_params=%7B%22dtype%22%3A%22float32%22%7D"));
    let composed = algorithm_over_expression(&pq, Some(T::Float32)).unwrap().expect("composed");
    assert_eq!(composed.rescale, [default_rescale_for_dtype(T::Float32, true)]);
    let pq = parse_query(Some("expression=b1&algorithm=cast&algorithm_params=%7B%22dtype%22%3A%22int16%22%7D"));
    let composed = algorithm_over_expression(&pq, Some(T::Float32)).unwrap().expect("composed");
    assert_eq!(composed.rescale, [(-32768.0, 32767.0)]);
}
