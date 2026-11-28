//! Unit tests for `cogzilla_kernel::render`.

use cogzilla_raster::SampleType;

use crate::colormap::nodata_index;
use super::dem::eval_hillshade_row_clean;
use super::single_band::render_single_specialized;
use super::rgb::{
    CleanRgbNearestPlan, bilinear_rgb_single_block_row, sample_rgb_bilinear_row_clean,
    sample_rgb_nearest_row_clean,
};
use super::*;
use cogzilla_codec::{AlignedF32Slab, BandData, BlockStats, DecodedBand};

#[test]
fn a_nodata_palette_keeps_the_colour_map_s_colours() {
    use crate::colormap::ColormapName;
    for name in [ColormapName::Viridis, ColormapName::Terrain, ColormapName::Grayscale] {
        let lut = name.lut();
        let (plte, merged) = build_plte_nodata(lut);
        let mut off = 0;
        for v in 0..=255u8 {
            let at = usize::from(nodata_index(v, merged));
            assert!(at >= 1, "index 0 is the transparent one");
            if plte[at * 3..at * 3 + 3] != lut.rgb[usize::from(v)] {
                off += 1;
                assert_eq!(v, merged, "only the merged entry may take another colour");
                let got = &plte[at * 3..at * 3 + 3];
                let want = lut.rgb[usize::from(v)];
                assert!((0..3).all(|k| got[k].abs_diff(want[k]) <= 4), "{name:?}: {got:?} for {want:?}");
            }
        }
        // Viridis repeats a colour next to itself, so it is drawn exactly.
        if matches!(name, ColormapName::Viridis) {
            assert_eq!(off, 0);
        }
    }
}

fn make_plan_32() -> CoordPlan<32> {
    let mut u = [0.0f32; 32];
    let mut v = [0.0f32; 32];
    for i in 0..32 {
        u[i] = 1.25 + (i as f32) * 1.73;
        v[i] = 2.35 + (i as f32) * 1.61;
    }
    CoordPlan::Separable1D {
        u: Box::new(u),
        v: Box::new(v),
        src_min_col: 1,
        src_max_col: 60,
        src_min_row: 2,
        src_max_row: 60,
        all_in_bounds: true,
    }
}

#[test]
fn fused_rgb_bilinear_matches_scalar_for_u8_u16_and_f32() {
    let plan = make_plan_32();
    let tw = 64u32;
    let th = 64u32;
    let len = (tw * th) as usize;
    let stats = BlockStats {
        all_nodata: false,
        no_nodata: true,
        min: 0.0,
        max: 255.0,
    };

    // 1. U8 single-block
    let b0_u8 = DecodedBand {
        width: tw,
        height: th,
        data: BandData::U8 {
            data: (0..len).map(|i| ((i * 7 + 3) % 256) as u8).collect(),
            nodata: None,
        },
        stats,
    };
    let b1_u8 = DecodedBand {
        width: tw,
        height: th,
        data: BandData::U8 {
            data: (0..len).map(|i| ((i * 11 + 17) % 256) as u8).collect(),
            nodata: None,
        },
        stats,
    };
    let b2_u8 = DecodedBand {
        width: tw,
        height: th,
        data: BandData::U8 {
            data: (0..len).map(|i| ((i * 13 + 29) % 256) as u8).collect(),
            nodata: None,
        },
        stats,
    };
    let blks0 = [Some(&b0_u8)];
    let blks1 = [Some(&b1_u8)];
    let blks2 = [Some(&b2_u8)];
    let srcs_u8 = [
        BandSource::new(tw, th, tw, th, 0, 0, 1, 1, &blks0),
        BandSource::new(tw, th, tw, th, 0, 0, 1, 1, &blks1),
        BandSource::new(tw, th, tw, th, 0, 0, 1, 1, &blks2),
    ];

    // 2. U16 single-block
    let b0_u16 = DecodedBand {
        width: tw,
        height: th,
        data: BandData::U16 {
            data: (0..len).map(|i| ((i * 37 + 100) % 20000) as u16).collect(),
            nodata: None,
        },
        stats,
    };
    let b1_u16 = DecodedBand {
        width: tw,
        height: th,
        data: BandData::U16 {
            data: (0..len).map(|i| ((i * 53 + 200) % 20000) as u16).collect(),
            nodata: None,
        },
        stats,
    };
    let b2_u16 = DecodedBand {
        width: tw,
        height: th,
        data: BandData::U16 {
            data: (0..len).map(|i| ((i * 71 + 300) % 20000) as u16).collect(),
            nodata: None,
        },
        stats,
    };
    let u0 = [Some(&b0_u16)];
    let u1 = [Some(&b1_u16)];
    let u2 = [Some(&b2_u16)];
    let srcs_u16 = [
        BandSource::new(tw, th, tw, th, 0, 0, 1, 1, &u0),
        BandSource::new(tw, th, tw, th, 0, 0, 1, 1, &u1),
        BandSource::new(tw, th, tw, th, 0, 0, 1, 1, &u2),
    ];

    // 3. F32 single-block
    let mut s0 = AlignedF32Slab::filled(len, 0.0);
    let mut s1 = AlignedF32Slab::filled(len, 0.0);
    let mut s2 = AlignedF32Slab::filled(len, 0.0);
    for i in 0..len {
        s0.as_mut_slice()[i] = ((i * 19) % 1000) as f32 * 0.001;
        s1.as_mut_slice()[i] = ((i * 23) % 1000) as f32 * 0.001;
        s2.as_mut_slice()[i] = ((i * 29) % 1000) as f32 * 0.001;
    }
    let b0_f32 = DecodedBand { width: tw, height: th, data: BandData::F32(s0), stats };
    let b1_f32 = DecodedBand { width: tw, height: th, data: BandData::F32(s1), stats };
    let b2_f32 = DecodedBand { width: tw, height: th, data: BandData::F32(s2), stats };
    let f0 = [Some(&b0_f32)];
    let f1 = [Some(&b1_f32)];
    let f2 = [Some(&b2_f32)];
    let srcs_f32 = [
        BandSource::new(tw, th, tw, th, 0, 0, 1, 1, &f0),
        BandSource::new(tw, th, tw, th, 0, 0, 1, 1, &f1),
        BandSource::new(tw, th, tw, th, 0, 0, 1, 1, &f2),
    ];

    for (name, srcs, mins, maxs) in [
        ("u8", &srcs_u8, [0.0f32; 3], [255.0f32; 3]),
        ("u16", &srcs_u16, [500.0; 3], [19500.0; 3]),
        ("f32", &srcs_f32, [0.0; 3], [1.0; 3]),
    ] {
        let inv = [
            255.0 / (maxs[0] - mins[0]),
            255.0 / (maxs[1] - mins[1]),
            255.0 / (maxs[2] - mins[2]),
        ];
        for py in 0..32 {
            let mut fused = [0u8; 32 * 3];
            let mut scalar = [0u8; 32 * 3];
            sample_rgb_bilinear_row_clean(&plan, py, srcs, inv, mins, &mut fused);
            for_each_pixel_in_row(&plan, py, |px, u, v| {
                let off = px * 3;
                scalar[off] = rescale_u8(sample_bilinear::<true>(&srcs[0], u, v), inv[0], mins[0]);
                scalar[off + 1] = rescale_u8(sample_bilinear::<true>(&srcs[1], u, v), inv[1], mins[1]);
                scalar[off + 2] = rescale_u8(sample_bilinear::<true>(&srcs[2], u, v), inv[2], mins[2]);
            });
            for i in 0..fused.len() {
                let diff = (fused[i] as i32 - scalar[i] as i32).abs();
                assert!(
                    diff <= 1,
                    "{name} row {py} idx {i}: fused={} scalar={}",
                    fused[i],
                    scalar[i]
                );
            }
        }
    }
}

#[test]
fn terrain_encodings_round_trip_accurately() {
    for &elev in &[-430.5f32, -10.25, 0.0, 123.45, 2450.7, 8848.86] {
        let [r, g, b] = encode_terrarium_rgb(elev);
        let dec_terr = (r as f64 * 256.0 + g as f64 + b as f64 / 256.0) - 32768.0;
        assert!(
            (dec_terr - elev as f64).abs() < 0.005,
            "terrarium mismatch for {elev}: got {dec_terr}"
        );

        let [mr, mg, mb] = encode_terrain_rgb(elev);
        let dec_mb = -10000.0 + ((mr as u32 * 65536 + mg as u32 * 256 + mb as u32) as f64 * 0.1);
        assert!(
            (dec_mb - elev as f64).abs() <= 0.055,
            "terrainrgb mismatch for {elev}: got {dec_mb}"
        );
    }
}

#[test]
fn dem_stencil_matches_analytical_flat_and_45deg_east_ramp() {
    let inv_dx = 1.0 / 10.0f32;

    // Flat plane z = 500m -> slope = 0 deg, TiTiler's hillshade(alt=45) = 255 * (sin(45 deg) + 1) / 2 ~= 217.7, as a byte 217
    let flat = |alg| eval_dem_stencil(500.0, 500.0, 500.0, 500.0, 500.0, inv_dx, inv_dx, alg);
    let flat_slope = flat(DemAlgorithm::Slope { z_factor: 1.0 });
    assert!(flat_slope.abs() < 1e-4, "expected 0 deg slope, got {flat_slope}");
    let flat_hs = flat(DemAlgorithm::Hillshade { azimuth_deg: 315.0, altitude_deg: 45.0, z_factor: 1.0 });
    assert_eq!(flat_hs, 217.0, "expected 217.66 written as a byte");

    // 45-degree east-facing ramp: z drops by dx (=10m) each step East -> dz/dx = -1, dz/dy = 0
    let ramp = |alg| eval_dem_stencil(100.0, 110.0, 100.0, 90.0, 100.0, inv_dx, inv_dx, alg);
    let ramp_slope = ramp(DemAlgorithm::Slope { z_factor: 1.0 });
    assert!((ramp_slope - 45.0).abs() < 1e-3, "expected 45 deg slope, got {ramp_slope}");
    let ramp_aspect = ramp(DemAlgorithm::Aspect);
    assert!((ramp_aspect - 90.0).abs() < 1e-3, "expected 90 deg (East) aspect, got {ramp_aspect}");
    // A missing neighbour is numpy's one-sided difference, over one pixel.
    let edge = eval_dem_stencil(100.0, f32::NAN, 100.0, 90.0, 100.0, inv_dx, inv_dx, DemAlgorithm::Slope { z_factor: 1.0 });
    assert!((edge - 45.0).abs() < 1e-3, "expected 45 deg slope from one side, got {edge}");
}

#[test]
fn hillshade_is_titilers() {
    // titiler.core.algorithm.dem.HillShade, written out in double precision: numpy's
    // gradient per pixel (axis 0 down, axis 1 across), its slope and aspect, its azimuth.
    let titiler = |z: &[[f64; 3]; 3], azimuth: f64, altitude: f64, zx: f64| -> f64 {
        let x = 0.5 * (z[2][1] - z[0][1]) * zx;
        let y = 0.5 * (z[1][2] - z[1][0]) * zx;
        let slope = std::f64::consts::FRAC_PI_2 - (x * x + y * y).sqrt().atan();
        let aspect = (-x).atan2(y);
        let az = (360.0 - azimuth).to_radians();
        let alt = altitude.to_radians();
        let shaded = alt.sin() * slope.sin() + alt.cos() * slope.cos() * (az - aspect).cos();
        (255.0 * (shaded + 1.0) / 2.0).max(0.0)
    };
    let patches = [
        [[100.0, 103.0, 109.0], [98.0, 101.0, 107.0], [91.0, 97.0, 104.0]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[500.0, 480.0, 470.0], [520.0, 505.0, 471.0], [560.0, 530.0, 500.0]],
    ];
    for z in &patches {
        for (azimuth, altitude, zx) in [(45.0, 45.0, 1.0), (200.0, 20.0, 1.0), (0.0, 60.0, 3.0), (360.0, -30.0, 0.5)] {
            let want = titiler(z, azimuth, altitude, zx);
            // Compass degrees, as the query is converted: TiTiler's 45 is 315 (North-West).
            let alg = DemAlgorithm::Hillshade {
                azimuth_deg: ((azimuth - 90.0) as f32).rem_euclid(360.0),
                altitude_deg: altitude as f32,
                z_factor: zx as f32,
            };
            let f = |r: usize, c: usize| z[r][c] as f32;
            let got = eval_dem_stencil(f(0, 1), f(1, 0), f(1, 1), f(1, 2), f(2, 1), 1.0, 1.0, alg);
            // Written as a byte, as TiTiler's `astype("uint8")`.
            assert!(got == got.trunc() && (got as f64 - want.trunc()).abs() < 1e-9, "{z:?} {azimuth} {altitude} {zx}: {got} vs {want}");
        }
    }
}

#[test]
fn dem_adjacent_tiles_have_zero_boundary_seam() {
    // Smooth tilted plane z(c, r) = 1000 - 10*c - 5*r sampled at 4x zoom (step = 0.25 px)
    // across two adjacent 32x32 tiles. Every border pixel must produce the exact same slope
    // as the interior (no 1-pixel staircase spike along tile boundaries!).
    let tw = 64u32;
    let th = 64u32;
    let len = (tw * th) as usize;
    let mut slab = AlignedF32Slab::filled(len, 0.0);
    for r in 0..th {
        for c in 0..tw {
            slab.as_mut_slice()[(r * tw + c) as usize] =
                1000.0 - 10.0 * (c as f32) - 5.0 * (r as f32);
        }
    }
    let band = DecodedBand {
        width: tw,
        height: th,
        data: BandData::F32(slab),
        stats: BlockStats {
            all_nodata: false,
            no_nodata: true,
            min: 0.0,
            max: 1000.0,
        },
    };
    let blks = [Some(&band)];
    let src = BandSource::new(tw, th, tw, th, 0, 0, 1, 1, &blks);

    let make_tile_plan = |u0: f32, v0: f32| -> CoordPlan<32> {
        let mut u = [0.0f32; 32];
        let mut v = [0.0f32; 32];
        for i in 0..32 {
            u[i] = u0 + (i as f32) * 0.25;
            v[i] = v0 + (i as f32) * 0.25;
        }
        CoordPlan::Separable1D {
            u: Box::new(u),
            v: Box::new(v),
            src_min_col: 2,
            src_max_col: 50,
            src_min_row: 2,
            src_max_row: 50,
            all_in_bounds: true,
        }
    };

    let left_plan = make_tile_plan(10.0, 10.0);
    let right_plan = make_tile_plan(18.0, 10.0);
    let style = SingleBandStyle {
        rescale_min: 0.0,
        rescale_max: 90.0,
        colormap: ColormapName::Viridis,
        resample: ResampleMode::Bilinear,
        format: OutputFormat::RawF32,
    };
    let left_bytes = render_dem_algorithm(
        &left_plan,
        &src,
        DemAlgorithm::Slope { z_factor: 1.0 },
        2.5,
        &style,
    );
    let right_bytes = render_dem_algorithm(
        &right_plan,
        &src,
        DemAlgorithm::Slope { z_factor: 1.0 },
        2.5,
        &style,
    );
    let read_f32 = |b: &[u8], py: usize, px: usize| -> f32 {
        let off = (py * 32 + px) * 4;
        f32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
    };
    let expected = read_f32(&left_bytes, 16, 16);
    for py in 0..32 {
        let l_edge = read_f32(&left_bytes, py, 31);
        let r_edge = read_f32(&right_bytes, py, 0);
        let top_edge = read_f32(&left_bytes, 0, py);
        let bot_edge = read_f32(&left_bytes, 31, py);
        assert!((l_edge - expected).abs() < 1e-3, "left edge seam at py={py}: {l_edge} vs {expected}");
        assert!((r_edge - expected).abs() < 1e-3, "right edge seam at py={py}: {r_edge} vs {expected}");
        assert!((top_edge - expected).abs() < 1e-3, "top edge seam at px={py}: {top_edge} vs {expected}");
        assert!((bot_edge - expected).abs() < 1e-3, "bot edge seam at px={py}: {bot_edge} vs {expected}");
    }
}

#[test]
fn simd_cubic_and_hillshade_match_scalar() {
    let tw = 512u32;
    let th = 512u32;
    let len = (tw * th) as usize;
    let mut slab = AlignedF32Slab::filled(len, 0.0);
    for r in 0..th {
        for c in 0..tw {
            let x = c as f32 * 0.07;
            let y = r as f32 * 0.09;
            slab.as_mut_slice()[(r * tw + c) as usize] =
                1500.0 + 400.0 * x.sin() * y.cos() + 120.0 * (x * 1.7 + y * 0.5).cos();
        }
    }
    let band = DecodedBand {
        width: tw,
        height: th,
        data: BandData::F32(slab),
        stats: BlockStats {
            all_nodata: false,
            no_nodata: true,
            min: 900.0,
            max: 2100.0,
        },
    };
    let blks = [Some(&band)];
    let src = BandSource::new(tw, th, tw, th, 0, 0, 1, 1, &blks);

    // 1. Verify SIMD sample_cubic against scalar reference
    for i in 0..64 {
        let u = 10.13 + (i as f32) * 3.71;
        let v = 12.29 + (i as f32) * 3.53;
        let fast = sample_cubic(&src, u, v);
        let (x0, fx) = fast_floor_i32(u);
        let (y0, fy) = fast_floor_i32(v);
        let wx = cubic_weights(fx);
        let wy = cubic_weights(fy);
        let mut ref_val = 0.0f32;
        for j in 0..4 {
            let mut r_acc = 0.0f32;
            for k in 0..4 {
                r_acc += src.sample_checked(x0 + k - 1, y0 + j - 1) * wx[k as usize];
            }
            ref_val += r_acc * wy[j as usize];
        }
        assert!(
            (fast - ref_val).abs() < 1e-3,
            "cubic mismatch at ({u},{v}): fast={fast} ref={ref_val}"
        );
    }

    // 2. Verify the SIMD hillshade row against scalar eval_dem_stencil
    let mut r0 = [0.0f32; 258];
    let mut r1 = [0.0f32; 258];
    let mut r2 = [0.0f32; 258];
    for i in 0..258 {
        r0[i] = src.sample_inbounds(i as u32 + 10, 10);
        r1[i] = src.sample_inbounds(i as u32 + 10, 11);
        r2[i] = src.sample_inbounds(i as u32 + 10, 12);
    }
    let mut simd_out = [0u8; 256];
    eval_hillshade_row_clean::<256>(
        &r0, &r1, &r2, 315.0, 45.0, 2.0, 1.0, 0.0, &mut simd_out,
    );
    for px in 0..256 {
        let c = px + 1;
        let s = eval_dem_stencil(
            r0[c], r1[c - 1], r1[c], r1[c + 1], r2[c],
            1.0,
            1.0,
            DemAlgorithm::Hillshade {
                azimuth_deg: 315.0,
                altitude_deg: 45.0,
                z_factor: 2.0,
            },
        );
        let scalar_u8 = rescale_u8(s, 1.0, 0.0);
        let diff = (simd_out[px] as i32 - scalar_u8 as i32).abs();
        assert!(diff <= 1, "hillshade mismatch at px={px}: simd={} scalar={scalar_u8}", simd_out[px]);
    }

    // 3. Time 100 iterations of 256x256 cubic hillshade PNG
    let mut u = [0.0f32; 256];
    let mut v = [0.0f32; 256];
    for i in 0..256 {
        u[i] = 10.0 + (i as f32) * 1.5;
        v[i] = 10.0 + (i as f32) * 1.5;
    }
    let plan = CoordPlan::Separable1D {
        u: Box::new(u),
        v: Box::new(v),
        src_min_col: 10,
        src_max_col: 400,
        src_min_row: 10,
        src_max_row: 400,
        all_in_bounds: true,
    };
    let style = SingleBandStyle {
        rescale_min: 0.0,
        rescale_max: 255.0,
        colormap: ColormapName::from_str_opt("gray").unwrap(),
        resample: ResampleMode::Cubic,
        format: OutputFormat::Png,
    };
    let alg = DemAlgorithm::Hillshade {
        azimuth_deg: 315.0,
        altitude_deg: 45.0,
        z_factor: 2.0,
    };
    // warmup
    let _ = render_dem_algorithm(&plan, &src, alg, 30.0, &style);
    let t0 = std::time::Instant::now();
    for _ in 0..100 {
        let png = render_dem_algorithm(&plan, &src, alg, 30.0, &style);
        std::hint::black_box(png);
    }
    let avg_us = t0.elapsed().as_secs_f64() * 1e6 / 100.0;
    eprintln!("256x256 cubic SIMD hillshade PNG avg: {avg_us:.1} µs");
}

#[test]
fn dem_anisotropic_spacing_matches_exact_slope() {
    // Pixels 5 units wide and 10 high.
    // A 45° East-facing ramp rises 5 m per East pixel (dz/dx = 1.0, dz/dy = 0.0).
    let slope = eval_dem_stencil(
        5.0, 0.0, 5.0, 10.0, 5.0,
        1.0 / 5.0,
        1.0 / 10.0,
        DemAlgorithm::Slope { z_factor: 1.0 },
    );
    assert!((slope - 45.0).abs() < 1e-4, "expected 45.0° slope, got {slope}");
}

#[test]
fn npy_v1_header_is_64_byte_aligned_and_valid() {
    let raw = Bytes::from_iter([1.0f32, 2.0, 3.0, 4.0].iter().flat_map(|v| v.to_le_bytes()));
    let npy = wrap_raw_f32_as_npy(raw.clone(), 1, 2, 2);
    assert_eq!(&npy[..8], b"\x93NUMPY\x01\x00");
    let hlen = u16::from_le_bytes([npy[8], npy[9]]) as usize;
    let total_hdr = 10 + hlen;
    assert_eq!(total_hdr % 64, 0, "NPY v1.0 header must be 64-byte aligned");
    let dict = std::str::from_utf8(&npy[10..total_hdr]).unwrap();
    assert!(dict.contains("'descr': '<f4'"));
    assert!(dict.contains("'fortran_order': False"));
    assert!(dict.contains("'shape': (1, 2, 2)"));
    assert!(dict.ends_with('\n'));
    assert_eq!(&npy[total_hdr..], raw.as_ref());
}

#[test]
fn sample_rgb_nearest_row_clean_matches_scalar() {
    use cogzilla_codec::{AlignedF32Slab, BandData, BlockStats, DecodedBand};
    use std::sync::Arc;

    let tw = 64u32;
    let th = 64u32;
    let n_px = (tw * th) as usize;
    let mut u_arr = [0.0f32; 256];
    let mut v_arr = [0.0f32; 256];
    for i in 0..256 {
        u_arr[i] = 1.2 + (i as f32) * 0.23;
        v_arr[i] = 2.4 + (i as f32) * 0.22;
    }
    let plan = CoordPlan::Separable1D {
        u: Box::new(u_arr),
        v: Box::new(v_arr),
        src_min_col: 1,
        src_max_col: 62,
        src_min_row: 2,
        src_max_row: 62,
        all_in_bounds: true,
    };
    let stats = BlockStats {
        all_nodata: false,
        no_nodata: true,
        min: 0.0,
        max: 255.0,
    };

    let check = |srcs: [BandSource<'_>; 3], inv: [f32; 3], mins: [f32; 3], label: &str| {
        let mut fast = vec![0u8; 256 * 3];
        let mut ref_row = vec![0u8; 256 * 3];
        for py in [0usize, 17, 128, 255] {
            sample_rgb_nearest_row_clean(&plan, py, &srcs, inv, mins, &mut fast);
            for_each_pixel_in_row(&plan, py, |px, u, v| {
                ref_row[px * 3] = rescale_u8(sample_nearest::<true>(&srcs[0], u, v), inv[0], mins[0]);
                ref_row[px * 3 + 1] = rescale_u8(sample_nearest::<true>(&srcs[1], u, v), inv[1], mins[1]);
                ref_row[px * 3 + 2] = rescale_u8(sample_nearest::<true>(&srcs[2], u, v), inv[2], mins[2]);
            });
            assert_eq!(fast, ref_row, "{label} mismatch at py={py}");
        }
    };

    // 1. Interleaved RgbU8Channel (identity + rescaled)
    let rgb_vec: Vec<u8> = (0..n_px * 3).map(|i| ((i * 37 + 11) % 256) as u8).collect();
    let rgb_arc: Arc<[u8]> = Arc::from(rgb_vec);
    let mk_rgb = |ch: u8| DecodedBand {
        data: BandData::RgbU8Channel { rgb: rgb_arc.clone(), channel: ch, nodata: None },
        width: tw,
        height: th,
        stats,
    };
    let (b0, b1, b2) = (mk_rgb(0), mk_rgb(1), mk_rgb(2));
    let (s0, s1, s2) = ([Some(&b0)], [Some(&b1)], [Some(&b2)]);
    check(
        [
            BandSource::new(tw, th, tw, th, 0, 0, 1, 1, &s0),
            BandSource::new(tw, th, tw, th, 0, 0, 1, 1, &s1),
            BandSource::new(tw, th, tw, th, 0, 0, 1, 1, &s2),
        ],
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0],
        "RgbU8 identity",
    );
    check(
        [
            BandSource::new(tw, th, tw, th, 0, 0, 1, 1, &s0),
            BandSource::new(tw, th, tw, th, 0, 0, 1, 1, &s1),
            BandSource::new(tw, th, tw, th, 0, 0, 1, 1, &s2),
        ],
        [1.5, 0.8, 2.0],
        [10.0, 20.0, 5.0],
        "RgbU8 rescaled",
    );

    // 2. Planar U8
    let mk_u8 = |seed: usize| DecodedBand {
        data: BandData::U8 {
            data: (0..n_px).map(|i| ((i * seed + 7) % 256) as u8).collect(),
            nodata: None,
        },
        width: tw,
        height: th,
        stats,
    };
    let (u0, u1, u2) = (mk_u8(13), mk_u8(29), mk_u8(53));
    let (su0, su1, su2) = ([Some(&u0)], [Some(&u1)], [Some(&u2)]);
    check(
        [
            BandSource::new(tw, th, tw, th, 0, 0, 1, 1, &su0),
            BandSource::new(tw, th, tw, th, 0, 0, 1, 1, &su1),
            BandSource::new(tw, th, tw, th, 0, 0, 1, 1, &su2),
        ],
        [1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0],
        "U8 identity",
    );

    // 3. Planar U16
    let mk_u16 = |seed: usize| DecodedBand {
        data: BandData::U16 {
            data: (0..n_px).map(|i| ((i * seed + 100) % 4096) as u16).collect(),
            nodata: None,
        },
        width: tw,
        height: th,
        stats,
    };
    let (w0, w1, w2) = (mk_u16(17), mk_u16(31), mk_u16(47));
    let (sw0, sw1, sw2) = ([Some(&w0)], [Some(&w1)], [Some(&w2)]);
    check(
        [
            BandSource::new(tw, th, tw, th, 0, 0, 1, 1, &sw0),
            BandSource::new(tw, th, tw, th, 0, 0, 1, 1, &sw1),
            BandSource::new(tw, th, tw, th, 0, 0, 1, 1, &sw2),
        ],
        [255.0 / 3000.0; 3],
        [100.0; 3],
        "U16 rescaled",
    );

    // 4. Planar F32
    let mk_f32 = |seed: f32| {
        let mut slab = AlignedF32Slab::filled(n_px, 0.0);
        for (i, v) in slab.as_mut_slice().iter_mut().enumerate() {
            *v = (i as f32) * 0.01 * seed;
        }
        DecodedBand {
            data: BandData::F32(slab),
            width: tw,
            height: th,
            stats,
        }
    };
    let (f0, f1, f2) = (mk_f32(1.0), mk_f32(1.7), mk_f32(2.3));
    let (sf0, sf1, sf2) = ([Some(&f0)], [Some(&f1)], [Some(&f2)]);
    check(
        [
            BandSource::new(tw, th, tw, th, 0, 0, 1, 1, &sf0),
            BandSource::new(tw, th, tw, th, 0, 0, 1, 1, &sf1),
            BandSource::new(tw, th, tw, th, 0, 0, 1, 1, &sf2),
        ],
        [255.0 / 100.0; 3],
        [0.0; 3],
        "F32 rescaled",
    );
}

#[test]
fn single_block_u16_u8_and_rgb_separable_fast_paths_match_scalar() {
    use crate::colormap::ColormapName;
    use cogzilla_codec::{BandData, BlockStats, DecodedBand};
    use std::sync::Arc;

    let tw = 64u32;
    let th = 64u32;
    let n_px = (tw * th) as usize;
    let stats = BlockStats {
        min: 0.0,
        max: 4000.0,
        all_nodata: false,
        no_nodata: true,
    };
    let mut u = [0.0f32; 256];
    let mut v = [0.0f32; 256];
    for i in 0..256 {
        u[i] = 1.25 + (i as f32) * (60.0 / 255.0);
        v[i] = 1.75 + (i as f32) * (59.5 / 255.0);
    }
    let plan = CoordPlan::Separable1D {
        u: Box::new(u),
        v: Box::new(v),
        src_min_col: 1,
        src_max_col: 62,
        src_min_row: 1,
        src_max_row: 62,
        all_in_bounds: true,
    };

    // 1. Single-band U16 & U8 (Nearest + Bilinear) in render_single_specialized + render_expr_vm
    let u16_a = DecodedBand {
        data: BandData::U16 {
            data: (0..n_px).map(|i| ((i * 29 + 17) % 3500) as u16).collect(),
            nodata: None,
        },
        width: tw,
        height: th,
        stats,
    };
    let u16_b = DecodedBand {
        data: BandData::U16 {
            data: (0..n_px).map(|i| ((i * 13 + 211) % 3500) as u16).collect(),
            nodata: None,
        },
        width: tw,
        height: th,
        stats,
    };
    let u8_a = DecodedBand {
        data: BandData::U8 {
            data: (0..n_px).map(|i| ((i * 7 + 5) & 0xFF) as u8).collect(),
            nodata: None,
        },
        width: tw,
        height: th,
        stats,
    };
    let s_u16_a = [Some(&u16_a)];
    let s_u16_b = [Some(&u16_b)];
    let s_u8_a = [Some(&u8_a)];
    let bs_u16_a = BandSource::new(tw, th, tw, th, 0, 0, 1, 1, &s_u16_a);
    let bs_u8_a = BandSource::new(tw, th, tw, th, 0, 0, 1, 1, &s_u8_a);

    for (label, bs) in [("U16", &bs_u16_a), ("U8", &bs_u8_a)] {
        for (is_bilin, resample) in [
            (false, ResampleMode::Nearest),
            (true, ResampleMode::Bilinear),
        ] {
            let style = SingleBandStyle {
                rescale_min: 0.0,
                rescale_max: 255.0,
                colormap: ColormapName::Viridis,
                format: OutputFormat::RawF32,
                resample,
            };
            let got = if is_bilin {
                render_single_specialized::<256, true, true>(&plan, bs, &style)
            } else {
                render_single_specialized::<256, false, true>(&plan, bs, &style)
            };
            for py in 0..256 {
                for px in 0..256 {
                    let idx = (py * 256 + px) * 4;
                    let g = f32::from_le_bytes(got[idx..idx + 4].try_into().unwrap());
                    let e = if is_bilin {
                        sample_bilinear::<true>(bs, u[px], v[py])
                    } else {
                        sample_nearest::<true>(bs, u[px], v[py])
                    };
                    assert!(
                        (g - e).abs() <= 1e-3,
                        "{label} bilin={is_bilin} mismatch at ({px},{py}): got {g}, exp {e}"
                    );
                }
            }
        }
    }

    // 2. render_expr_vm on U16 pair (Nearest + Bilinear)
    let prog = crate::expr_vm::ExprProgram::compile("(b1 - b2) / (b1 + b2 + 1.0)").unwrap();
    let pair = [
        BandSource::new(tw, th, tw, th, 0, 0, 1, 1, &s_u16_a),
        BandSource::new(tw, th, tw, th, 0, 0, 1, 1, &s_u16_b),
    ];
    for resample in [ResampleMode::Nearest, ResampleMode::Bilinear] {
        let style = SingleBandStyle {
            rescale_min: -1.0,
            rescale_max: 1.0,
            colormap: ColormapName::Viridis,
            format: OutputFormat::RawF32,
            resample,
        };
        let got = render_expr_vm(&plan, &pair, &prog, &style, (tw, th));
        for py in 0..256 {
            for px in 0..256 {
                let idx = (py * 256 + px) * 4;
                let g = f32::from_le_bytes(got[idx..idx + 4].try_into().unwrap());
                let v1 = match resample {
                    ResampleMode::Bilinear => sample_bilinear::<true>(&pair[0], u[px], v[py]),
                    _ => sample_nearest::<true>(&pair[0], u[px], v[py]),
                };
                let v2 = match resample {
                    ResampleMode::Bilinear => sample_bilinear::<true>(&pair[1], u[px], v[py]),
                    _ => sample_nearest::<true>(&pair[1], u[px], v[py]),
                };
                let e = (v1 - v2) / (v1 + v2 + 1.0);
                assert!(
                    (g - e).abs() <= 1e-4,
                    "expr_vm {resample:?} mismatch at ({px},{py}): got {g}, exp {e}"
                );
            }
        }
    }

    // 3. Interleaved RgbU8Channel on Separable1D (hoisted Nearest + SIMD Bilinear)
    let rgb_raw: Arc<[u8]> = (0..n_px * 3).map(|i| ((i * 37 + 11) & 0xFF) as u8).collect();
    let ch = |c: u8| DecodedBand {
        data: BandData::RgbU8Channel {
            rgb: Arc::clone(&rgb_raw),
            channel: c,
            nodata: None,
        },
        width: tw,
        height: th,
        stats,
    };
    let (c0, c1, c2) = (ch(0), ch(1), ch(2));
    let (sc0, sc1, sc2) = ([Some(&c0)], [Some(&c1)], [Some(&c2)]);
    let rgb_srcs = [
        BandSource::new(tw, th, tw, th, 0, 0, 1, 1, &sc0),
        BandSource::new(tw, th, tw, th, 0, 0, 1, 1, &sc1),
        BandSource::new(tw, th, tw, th, 0, 0, 1, 1, &sc2),
    ];
    for (mins, maxs) in [([0.0; 3], [255.0; 3]), ([10.0, 20.0, 30.0], [220.0, 230.0, 240.0])] {
        let inv = [
            255.0 / (maxs[0] - mins[0]),
            255.0 / (maxs[1] - mins[1]),
            255.0 / (maxs[2] - mins[2]),
        ];
        let np = CleanRgbNearestPlan::build(&plan, &rgb_srcs, inv, mins);
        let mut row_fast = [0u8; 256 * 3];
        let mut row_ref = [0u8; 256 * 3];
        for py in [0usize, 13, 128, 255] {
            np.sample_row(&plan, py, &rgb_srcs, inv, mins, &mut row_fast);
            sample_rgb_nearest_row_clean(&plan, py, &rgb_srcs, inv, mins, &mut row_ref);
            assert_eq!(row_fast, row_ref, "CleanRgbNearestPlan mismatch at py={py}");

            sample_rgb_bilinear_row_clean(&plan, py, &rgb_srcs, inv, mins, &mut row_fast);
            for px in 0..256 {
                for ch_i in 0..3 {
                    let val = sample_bilinear::<true>(&rgb_srcs[ch_i], u[px], v[py]);
                    let exp = rescale_u8(val, inv[ch_i], mins[ch_i]);
                    let diff = (row_fast[px * 3 + ch_i] as i32 - exp as i32).abs();
                    assert!(
                        diff <= 1,
                        "RgbU8 bilinear mismatch at ({px},{py}) ch={ch_i}: got {}, exp {exp}",
                        row_fast[px * 3 + ch_i]
                    );
                }
            }
        }
    }
}

#[test]
fn render_expr_vm_masks_oob_pixels_for_constant_and_where_expressions() {
    let mut u = [0.0f32; 16];
    let mut v = [0.0f32; 16];
    // Left half (-8..-1) is out-of-bounds, right half (0..7) is inside a 16x16 IFD.
    for i in 0..16 {
        u[i] = i as f32 - 8.0;
        v[i] = i as f32;
    }
    let plan = CoordPlan::Separable1D {
        u: Box::new(u),
        v: Box::new(v),
        src_min_col: -8,
        src_max_col: 7,
        src_min_row: 0,
        src_max_row: 15,
        all_in_bounds: false,
    };
    let style = SingleBandStyle {
        rescale_min: 0.0,
        rescale_max: 10.0,
        colormap: ColormapName::Grayscale,
        format: OutputFormat::RawF32,
        resample: ResampleMode::Nearest,
    };

    // 1. Constant expression (`1`) with empty sources
    let const_prog = crate::expr_vm::ExprProgram::compile("1").unwrap();
    let out_const = render_expr_vm(&plan, &[], &const_prog, &style, (16, 16));
    let oob_val = f32::from_le_bytes(out_const[0..4].try_into().unwrap());
    let in_val = f32::from_le_bytes(out_const[8 * 4..9 * 4].try_into().unwrap());
    assert!(oob_val.is_nan(), "OOB pixel for constant expr must be NaN, got {oob_val}");
    assert_eq!(in_val, 1.0, "in-bounds pixel for constant expr must be 1.0");

    // 2. where(1, 5, b1) with a real band source
    let band = cogzilla_codec::DecodedBand {
        width: 16,
        height: 16,
        data: cogzilla_codec::BandData::U8 {
            data: vec![2u8; 16 * 16].into_boxed_slice(),
            nodata: None,
        },
        stats: cogzilla_codec::BlockStats {
            min: 2.0,
            max: 2.0,
            all_nodata: false,
            no_nodata: true,
        },
    };
    let refs = [Some(&band)];
    let src = [BandSource::new(16, 16, 16, 16, 0, 0, 1, 1, &refs)];
    let where_prog = crate::expr_vm::ExprProgram::compile("where(1, 5, b1)").unwrap();
    let out_where = render_expr_vm(&plan, &src, &where_prog, &style, (16, 16));
    let oob_w = f32::from_le_bytes(out_where[0..4].try_into().unwrap());
    let in_w = f32::from_le_bytes(out_where[8 * 4..9 * 4].try_into().unwrap());
    assert!(oob_w.is_nan(), "OOB pixel for where() expr must be NaN, got {oob_w}");
    assert_eq!(in_w, 5.0, "in-bounds pixel for where(1, 5, b1) must be 5.0");
}

/// Golden bytes of the `.npy` float32 writers: the header in full for a spread of shapes,
/// and the length and SHA-256 of whole arrays. These pin the response bytes, so a change
/// here is a change to what clients receive.
mod npy_golden_tests {
    use super::*;
    use sha2::{Digest, Sha256};

    fn sha256_hex(bytes: &[u8]) -> String {
        Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
    }

    /// The header as it must be written: magic, version 1.0, dict length, the dict, `pad`
    /// spaces and a newline.
    fn header(dict: &str, pad: usize) -> Vec<u8> {
        let mut out = b"\x93NUMPY\x01\x00".to_vec();
        out.extend_from_slice(&((dict.len() + pad + 1) as u16).to_le_bytes());
        out.extend_from_slice(dict.as_bytes());
        out.extend(std::iter::repeat_n(b' ', pad));
        out.push(b'\n');
        out
    }

    /// Reproducible float32 planes with NaN, both infinities, negatives and fractions.
    fn planes(bands: usize, height: u32, width: u32, seed: u32) -> Vec<u8> {
        let count = bands * height as usize * width as usize;
        let mut state = seed;
        let mut out = Vec::with_capacity(count * 4);
        for _ in 0..count {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let v = match (state >> 16) % 11 {
                0 => f32::NAN,
                1 => f32::INFINITY,
                2 => f32::NEG_INFINITY,
                _ => ((state >> 8) as i32 - (1 << 23)) as f32 / 1000.0,
            };
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }

    #[test]
    fn f32_header_bytes_are_pinned() {
        // (bands, height, width), the dict, and the spaces that pad it.
        let cases: [(usize, u32, u32, &str, usize); 7] = [
            (1, 2, 2, "{'descr': '<f4', 'fortran_order': False, 'shape': (1, 2, 2), }", 55),
            (3, 256, 256, "{'descr': '<f4', 'fortran_order': False, 'shape': (3, 256, 256), }", 51),
            (4, 7, 13, "{'descr': '<f4', 'fortran_order': False, 'shape': (4, 7, 13), }", 54),
            (10, 1024, 512, "{'descr': '<f4', 'fortran_order': False, 'shape': (10, 1024, 512), }", 49),
            (2, 12345, 1, "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 12345, 1), }", 51),
            (0, 0, 0, "{'descr': '<f4', 'fortran_order': False, 'shape': (0, 0, 0), }", 55),
            (
                usize::MAX,
                u32::MAX,
                u32::MAX,
                "{'descr': '<f4', 'fortran_order': False, 'shape': (18446744073709551615, 4294967295, 4294967295), }",
                18,
            ),
        ];
        for (bands, height, width, dict, pad) in cases {
            let got = encode_npy_f32_header(bands, height, width);
            assert_eq!(got.len(), 128, "header length for ({bands}, {height}, {width})");
            assert_eq!(got, header(dict, pad), "header bytes for ({bands}, {height}, {width})");
        }
    }

    #[test]
    fn f32_small_arrays_are_pinned_in_full() {
        let le = |values: &[f32]| values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>();
        let nan = f32::NAN;

        // One band, 2x2: the planes follow the header untouched, then the mask.
        let values = [1.5, nan, f32::NEG_INFINITY, -2.0];
        let mut exp = header("{'descr': '<f4', 'fortran_order': False, 'shape': (1, 2, 2), }", 55);
        exp.extend_from_slice(&le(&values));
        let got = wrap_raw_f32_as_npy(Bytes::from(le(&values)), 1, 2, 2);
        assert_eq!(got.as_ref(), exp.as_slice());
        let got = wrap_raw_f32_as_npy_with_mask(Bytes::from(le(&values)), 1, 2, 2, false);
        assert_eq!(got.as_ref(), exp.as_slice());

        let mut exp = header("{'descr': '<f4', 'fortran_order': False, 'shape': (2, 2, 2), }", 55);
        exp.extend_from_slice(&le(&values));
        exp.extend_from_slice(&le(&[255.0, 0.0, 0.0, 255.0]));
        let got = wrap_raw_f32_as_npy_with_mask(Bytes::from(le(&values)), 1, 2, 2, true);
        assert_eq!(got.as_ref(), exp.as_slice());

        // Three bands, 1 row of 2: a pixel is masked out only when no band is finite there.
        let values = [nan, 1.0, f32::INFINITY, nan, nan, -0.0];
        let mut exp = header("{'descr': '<f4', 'fortran_order': False, 'shape': (4, 1, 2), }", 55);
        exp.extend_from_slice(&le(&values));
        exp.extend_from_slice(&le(&[0.0, 255.0]));
        let got = wrap_raw_f32_as_npy_with_mask(Bytes::from(le(&values)), 3, 1, 2, true);
        assert_eq!(got.as_ref(), exp.as_slice());

        // An empty tile is the header and NaN throughout.
        let mut exp = header("{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3, 3), }", 55);
        exp.extend_from_slice(&le(&[nan; 18]));
        assert_eq!(empty_tile_bytes(3, OutputFormat::Npy, 2).as_ref(), exp.as_slice());
    }

    #[test]
    fn f32_arrays_are_pinned_by_length_and_digest() {
        // bands, height, width, mask, bytes cut from the end of the planes, length, SHA-256.
        let cases: [(usize, u32, u32, bool, usize, usize, &str); 14] = [
            (1, 2, 2, false, 0, 144, "a2f108947a23fd0bf4a45863a61642c8737f4a737de4ae562e04a90bc71c05f4"),
            (1, 2, 2, true, 0, 160, "d0037009d0ba0f13e146d568787c345a423ec2907d6cb00b31000ec65258a30b"),
            (3, 1, 5, true, 0, 208, "bf56f2c2b8000609fb42496e22e5f943de6b0a877470fe254d27cb89cab7e05f"),
            (3, 7, 13, false, 0, 1220, "17cdaaea973e5db420409f60adeab2ee13586b2544a1a5fc1ba1ef7f07e74e6d"),
            (3, 7, 13, true, 0, 1584, "9d14efff4748fd24de1c2a71c6ad4847fa1ea5b8f5f522fcfe69edefd28c9a39"),
            (1, 256, 256, false, 0, 262272, "7d0665ee74caf962f474512cae016c2846cd45d6196a7f1651a73156b3dcc705"),
            (1, 256, 256, true, 0, 524416, "020d46b447418a8fe60c40d93f4fc624d2149590ad38d6d38ee90130a725a835"),
            (3, 256, 256, true, 0, 1048704, "2eb4f63e99fb4a12b94f3a397ed4bd1e6e33e7ce715ad8de64aa5eca21adc716"),
            (3, 512, 512, false, 0, 3145856, "87028dd38eb31c700fef51f6918423e0674e54af6bf0106d9cef4eedbd1e7337"),
            (1, 1000, 3, true, 0, 24128, "0ec5029e87634968585a837ed56567549724a34e4a8c77bb196626a3269fe510"),
            (2, 1, 1024, true, 0, 12416, "58b72da7c4d034a04afe596f22dda8eea4c670780a1258d5629959694db7ac6f"),
            (4, 12345, 1, true, 0, 247028, "8e8bf86438d0a349a9780d8e920d24537ea1292fbf9e629eb88273ef25ba4c34"),
            (3, 9, 17, true, 150, 2426, "530137fad2f5a61fa10aa1043ed85832f2155ed0a9624fd94ecb8a9e1b9c4e33"),
            (0, 5, 6, true, 0, 248, "7225ef713825a642977bb6c7f903da5f140d1920d95aae073503dd2be4fdcc62"),
        ];
        for (i, (bands, height, width, mask, cut, len, digest)) in cases.into_iter().enumerate() {
            let mut raw = planes(bands, height, width, 0x9E37_79B9 ^ i as u32);
            raw.truncate(raw.len() - cut);
            let got = wrap_raw_f32_as_npy_with_mask(Bytes::from(raw.clone()), bands, height, width, mask);
            let what = format!("case {i}: ({bands}, {height}, {width}), mask {mask}, cut {cut}");
            assert_eq!(got.len(), len, "{what}");
            assert_eq!(sha256_hex(&got), digest, "{what}");
            if !mask {
                assert_eq!(got, wrap_raw_f32_as_npy(Bytes::from(raw), bands, height, width));
            }
        }
    }

    #[test]
    fn empty_npy_tiles_are_pinned_by_length_and_digest() {
        // tile size, bands, length, SHA-256.
        let cases: [(u32, usize, usize, &str); 5] = [
            (256, 1, 262272, "820ec7e5f3dfd5b867b3fd2e8d56e43f8c6afe3bde4a76720a77eafefdfd002f"),
            (256, 3, 786560, "803f28ae94e98efe8807f4fe51b0772f36d8318ba550ec45e5135cb70496d5de"),
            (512, 1, 1048704, "845324c7127f73c5a480b5b24d1a9b228c802e837171d77a33994bae555550cd"),
            (64, 2, 32896, "85a5dff8aa70b4e8df6b7455d9c44fcce85905cec33692d1b4c2776edfee5cee"),
            (1, 5, 148, "a5497f607d665ebcbf609c18495d17f61a4f30eb9107430f00f8c94f791dc6a5"),
        ];
        for (size, bands, len, digest) in cases {
            let got = empty_tile_bytes(size, OutputFormat::Npy, bands);
            assert_eq!(got.len(), len, "empty {size} tile of {bands} bands");
            assert_eq!(sha256_hex(&got), digest, "empty {size} tile of {bands} bands");
        }
    }
}

/// The typed `.npy` writer and the header it shares with the float32 one.
mod npy_typed_tests {
    use super::*;

    const KINDS: [(SampleType, &str, usize); 8] = [
        (SampleType::Uint8, "|u1", 1),
        (SampleType::Int8, "|i1", 1),
        (SampleType::Uint16, "<u2", 2),
        (SampleType::Int16, "<i2", 2),
        (SampleType::Uint32, "<u4", 4),
        (SampleType::Int32, "<i4", 4),
        (SampleType::Float32, "<f4", 4),
        (SampleType::Float64, "<f8", 8),
    ];

    /// Read a header back: the dtype, the order, the shape, and where the data starts.
    fn parse(npy: &[u8]) -> (String, String, Vec<usize>, usize) {
        assert_eq!(&npy[..8], b"\x93NUMPY\x01\x00");
        let start = 10 + usize::from(u16::from_le_bytes([npy[8], npy[9]]));
        assert_eq!(start % 64, 0, "the data starts on a multiple of 64 bytes");
        let dict = std::str::from_utf8(&npy[10..start]).unwrap();
        assert!(dict.ends_with('\n'));
        assert!(dict[..dict.len() - 1].trim_end_matches(' ').ends_with(", }"));
        let field = |name: &str, end: &str| {
            let from = dict.find(name).unwrap_or_else(|| panic!("{name} in {dict}")) + name.len();
            dict[from..from + dict[from..].find(end).unwrap()].to_string()
        };
        let shape = field("'shape': (", ")")
            .split(',')
            .map(str::trim)
            .filter(|dim| !dim.is_empty())
            .map(|dim| dim.parse().unwrap())
            .collect();
        (field("'descr': '", "'"), field("'fortran_order': ", ","), shape, start)
    }

    #[test]
    fn header_is_general_over_dtype_and_shape() {
        for (_, descr, _) in KINDS {
            for shape in [vec![], vec![7], vec![1, 2, 2], vec![4, 1024, 512], vec![2, 3, 4, 5]] {
                let hdr = encode_npy_header(descr, &shape);
                let (got_descr, order, got_shape, start) = parse(&hdr);
                assert_eq!((got_descr.as_str(), order.as_str()), (descr, "False"));
                assert_eq!(got_shape, shape);
                assert_eq!(start, hdr.len());
            }
        }
        // NumPy writes a tuple of one with its comma, and of none as `()`.
        let dict = |hdr: Vec<u8>| String::from_utf8(hdr[10..].to_vec()).unwrap();
        assert!(dict(encode_npy_header("<i2", &[7])).starts_with("{'descr': '<i2', 'fortran_order': False, 'shape': (7,), }"));
        assert!(dict(encode_npy_header("<f8", &[])).starts_with("{'descr': '<f8', 'fortran_order': False, 'shape': (), }"));
    }

    #[test]
    fn header_pads_to_the_next_64_bytes() {
        // Shapes of 2 to 40 dimensions, led by a number of one or two digits, so the dict
        // runs through every length from short to past 128 bytes.
        let (mut unpadded, mut short, mut long) = (false, false, false);
        for first in [1usize, 10] {
            for dims in 2..=40usize {
                let mut shape = vec![1usize; dims];
                shape[0] = first;
                let inner = shape.iter().map(usize::to_string).collect::<Vec<_>>().join(", ");
                let dict = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({inner}), }}");
                let total = (10 + dict.len() + 1).div_ceil(64) * 64;
                let pad = total - 11 - dict.len();
                let hdr = encode_npy_header("<f4", &shape);
                assert_eq!(hdr.len(), total, "{dict}");
                assert_eq!(usize::from(u16::from_le_bytes([hdr[8], hdr[9]])), total - 10);
                assert_eq!(&hdr[10..10 + dict.len()], dict.as_bytes());
                assert!(hdr[10 + dict.len()..total - 1].iter().all(|b| *b == b' '));
                assert_eq!(hdr[total - 1], b'\n');
                unpadded |= pad == 0;
                short |= total == 128;
                long |= total == 192;
            }
        }
        assert!(unpadded && short && long, "a dict that fills its block, and both block counts");
    }

    #[test]
    fn f32_header_is_the_general_header() {
        for (bands, height, width) in [(1usize, 2u32, 2u32), (3, 256, 256), (10, 1024, 512), (usize::MAX, u32::MAX, u32::MAX)] {
            assert_eq!(
                encode_npy_f32_header(bands, height, width),
                encode_npy_header("<f4", &[bands, height as usize, width as usize])
            );
        }
    }

    #[test]
    fn every_dtype_parses_back_with_a_body_of_the_right_length() {
        let values: Vec<f32> = (0..3 * 5 * 7).map(|i| if i % 4 == 0 { f32::NAN } else { i as f32 }).collect();
        for (kind, descr, size) in KINDS {
            assert_eq!(npy_descr(kind), descr);
            assert_eq!(kind.bytes_per_sample(), size);
            for bands in [1usize, 3] {
                for mask in [false, true] {
                    let npy = encode_npy_typed(&values, bands, 5, 7, kind, None, mask);
                    let (got_descr, order, shape, start) = parse(&npy);
                    assert_eq!((got_descr.as_str(), order.as_str()), (descr, "False"));
                    assert_eq!(shape, [bands + usize::from(mask), 5, 7]);
                    assert_eq!(npy.len() - start, (bands + usize::from(mask)) * 5 * 7 * size);
                }
            }
        }
    }

    /// The body of a 1x4 single-band array of `values`, with its mask plane.
    fn body(values: &[f32], kind: SampleType, absent: Option<f32>, mask: bool) -> Vec<u8> {
        let npy = encode_npy_typed(values, 1, 1, values.len(), kind, absent, mask);
        npy[parse(&npy).3..].to_vec()
    }

    #[test]
    fn values_fill_and_mask_in_each_dtype() {
        let nan = f32::NAN;
        let values = [1.9, nan, -2.9, 70000.0];
        macro_rules! le {
            ($t:ty, $($v:expr),*) => {{
                let mut out = Vec::new();
                $(out.extend_from_slice(&<$t>::to_le_bytes($v));)*
                out
            }};
        }
        // Truncated toward zero, held to the range, absent as zero; then the mask, the
        // type's largest number where there is a value and its smallest where there is not.
        assert_eq!(body(&values, SampleType::Uint8, None, true), le!(u8, 1, 0, 0, 255, 255, 0, 255, 255));
        assert_eq!(body(&values, SampleType::Int8, None, true), le!(i8, 1, 0, -2, 127, 127, -128, 127, 127));
        assert_eq!(body(&values, SampleType::Uint16, None, true), le!(u16, 1, 0, 0, 65535, 65535, 0, 65535, 65535));
        assert_eq!(body(&values, SampleType::Int16, None, true), le!(i16, 1, 0, -2, 32767, 32767, -32768, 32767, 32767));
        assert_eq!(body(&values, SampleType::Uint32, None, true), le!(u32, 1, 0, 0, 70000, u32::MAX, 0, u32::MAX, u32::MAX));
        assert_eq!(body(&values, SampleType::Int32, None, true), le!(i32, 1, 0, -2, 70000, i32::MAX, i32::MIN, i32::MAX, i32::MAX));
        assert_eq!(
            body(&values, SampleType::Float32, None, true),
            le!(f32, 1.9, 0.0, -2.9, 70000.0, f32::MAX, f32::MIN, f32::MAX, f32::MAX)
        );
        assert_eq!(
            body(&values, SampleType::Float64, None, true),
            le!(f64, f64::from(1.9f32), 0.0, f64::from(-2.9f32), 70000.0, f64::MAX, f64::MIN, f64::MAX, f64::MAX)
        );
        // Without the mask the body is the bands alone.
        assert_eq!(body(&values, SampleType::Uint16, None, false), le!(u16, 1, 0, 0, 65535));
        assert_eq!(body(&values, SampleType::Float32, None, false), le!(f32, 1.9, 0.0, -2.9, 70000.0));
    }

    #[test]
    fn absent_values_take_the_named_fill() {
        let nan = f32::NAN;
        let values = [nan, 5.0, nan];
        assert_eq!(body(&values, SampleType::Uint8, Some(9.0), false), [9, 5, 9]);
        assert_eq!(body(&values, SampleType::Uint8, None, false), [0, 5, 0]);
        // The fill is cast as a value is: out of range it is held to the range.
        assert_eq!(body(&values, SampleType::Uint8, Some(-9999.0), false), [0, 5, 0]);
        assert_eq!(body(&values, SampleType::Int16, Some(-9999.0), false), [(-9999i16).to_le_bytes(), 5i16.to_le_bytes(), (-9999i16).to_le_bytes()].concat());
        assert_eq!(body(&values, SampleType::Float32, Some(-9999.0), false), [(-9999f32).to_le_bytes(), 5f32.to_le_bytes(), (-9999f32).to_le_bytes()].concat());
        // A fill of NaN is zero in an integer and stays NaN in a float.
        assert_eq!(body(&values, SampleType::Int32, Some(nan), false), [0i32.to_le_bytes(), 5i32.to_le_bytes(), 0i32.to_le_bytes()].concat());
        assert!(f32::from_le_bytes(body(&values, SampleType::Float32, Some(nan), false)[..4].try_into().unwrap()).is_nan());
        // The fill does not reach the mask, which reads the values given.
        assert_eq!(body(&values, SampleType::Uint8, Some(9.0), true), [9, 5, 9, 0, 255, 0]);
    }

    #[test]
    fn floats_hold_infinities_to_the_largest_finite_number() {
        let values = [f32::INFINITY, f32::NEG_INFINITY];
        assert_eq!(body(&values, SampleType::Float32, None, false), [f32::MAX.to_le_bytes(), f32::MIN.to_le_bytes()].concat());
        assert_eq!(
            body(&values, SampleType::Float64, None, false),
            [f64::MAX.to_le_bytes(), f64::MIN.to_le_bytes()].concat()
        );
        assert_eq!(body(&values, SampleType::Int16, None, true), [i16::MAX.to_le_bytes(), i16::MIN.to_le_bytes(), i16::MAX.to_le_bytes(), i16::MAX.to_le_bytes()].concat());
    }

    #[test]
    fn mask_is_set_where_any_band_has_a_value() {
        let nan = f32::NAN;
        // Three bands of 1x3: nothing at the first pixel, one band at the second, all at the third.
        let values = [nan, nan, 1.0, nan, 2.0, 3.0, nan, nan, 0.0];
        let npy = encode_npy_typed(&values, 3, 1, 3, SampleType::Uint16, None, true);
        let (_, _, shape, start) = parse(&npy);
        assert_eq!(shape, [4, 1, 3]);
        assert_eq!(&npy[start + 18..], [0u16.to_le_bytes(), u16::MAX.to_le_bytes(), u16::MAX.to_le_bytes()].concat().as_slice());
    }

    #[test]
    fn a_float32_array_is_rewritten_in_another_type() {
        // Two planes of a band and its mask, as the float32 writer makes them.
        let raw: Vec<u8> = [1.9f32, 70000.0, -2.5, 0.0, 255.0, 255.0, 0.0, 255.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let f32_npy = wrap_raw_f32_as_npy(Bytes::from(raw), 2, 2, 2);
        let typed = retype_f32_npy(&f32_npy, SampleType::Uint16, true).unwrap();
        let (descr, _, shape, start) = parse(&typed);
        assert_eq!((descr.as_str(), shape), ("<u2", vec![2, 2, 2]));
        let words: Vec<u16> = typed[start..].chunks_exact(2).map(|w| u16::from_le_bytes([w[0], w[1]])).collect();
        assert_eq!(words, [1, 65535, 0, 0, 65535, 65535, 0, 65535]);
        // Signed: the mask's nothing is the smallest number; the value under it is kept.
        let typed = retype_f32_npy(&f32_npy, SampleType::Int16, true).unwrap();
        let start = parse(&typed).3;
        let words: Vec<i16> = typed[start..].chunks_exact(2).map(|w| i16::from_le_bytes([w[0], w[1]])).collect();
        assert_eq!(words, [1, 32767, -2, 0, 32767, 32767, -32768, 32767]);
        // Without a mask every plane is a band.
        let typed = retype_f32_npy(&f32_npy, SampleType::Uint8, false).unwrap();
        assert_eq!(&typed[parse(&typed).3..], [1, 255, 0, 0, 255, 255, 0, 255]);
        // float32 to float32 changes only the mask.
        let typed = retype_f32_npy(&f32_npy, SampleType::Float32, true).unwrap();
        let start = parse(&typed).3;
        let floats: Vec<f32> = typed[start..].chunks_exact(4).map(|w| f32::from_le_bytes([w[0], w[1], w[2], w[3]])).collect();
        assert_eq!(floats, [1.9, 70000.0, -2.5, 0.0, f32::MAX, f32::MAX, f32::MIN, f32::MAX]);
        // What is not a float32 array of three dimensions, or is cut short, is left alone.
        let typed_already = encode_npy_typed(&[1.0], 1, 1, 1, SampleType::Uint8, None, false);
        assert!(retype_f32_npy(&typed_already, SampleType::Uint16, false).is_none());
        assert!(retype_f32_npy(&f32_npy[..f32_npy.len() - 1], SampleType::Uint16, true).is_none());
        assert!(retype_f32_npy(b"not an array", SampleType::Uint16, true).is_none());
        let flat = [encode_npy_header("<f4", &[2]), 1f32.to_le_bytes().repeat(2)].concat();
        assert!(retype_f32_npy(&flat, SampleType::Uint16, false).is_none());
        // NaN in the mask plane maps to the type's minimum (invalid), not maximum.
        let raw_nan_mask: Vec<u8> = [5.0f32, 6.0, f32::NAN, 255.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let f32_nan_mask = wrap_raw_f32_as_npy(Bytes::from(raw_nan_mask), 2, 1, 2);
        let typed_u16 = retype_f32_npy(&f32_nan_mask, SampleType::Uint16, true).unwrap();
        let words_u16: Vec<u16> = typed_u16[parse(&typed_u16).3..].chunks_exact(2).map(|w| u16::from_le_bytes([w[0], w[1]])).collect();
        assert_eq!(words_u16, [5, 6, 0, u16::MAX]);
        let typed_i16 = retype_f32_npy(&f32_nan_mask, SampleType::Int16, true).unwrap();
        let words_i16: Vec<i16> = typed_i16[parse(&typed_i16).3..].chunks_exact(2).map(|w| i16::from_le_bytes([w[0], w[1]])).collect();
        assert_eq!(words_i16, [5, 6, i16::MIN, i16::MAX]);
    }

    #[test]
    fn only_the_named_planes_are_read() {
        let values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        // More values than the planes hold: the rest is left out.
        let npy = encode_npy_typed(&values, 1, 2, 2, SampleType::Uint8, None, false);
        assert_eq!(&npy[parse(&npy).3..], [1, 2, 3, 4]);
        // Fewer: the body is short, and the mask counts what is missing as absent.
        let npy = encode_npy_typed(&values[..3], 1, 2, 2, SampleType::Uint8, None, true);
        let (_, _, shape, start) = parse(&npy);
        assert_eq!(shape, [2, 2, 2]);
        assert_eq!(&npy[start..], [1, 2, 3, 255, 255, 255, 0]);
        // No bands: a mask of nothing there, the type's smallest number.
        let npy = encode_npy_typed(&[], 0, 2, 2, SampleType::Int16, None, true);
        assert_eq!(&npy[parse(&npy).3..], i16::MIN.to_le_bytes().repeat(4).as_slice());
    }

    #[test]
    fn typed_float32_is_not_the_raw_float32_wrapper() {
        // The two float32 writers agree on the header and on finite values, and differ
        // in what they write for an absent value, an infinity and the mask.
        let values = [1.5f32, -2.0, 3.25, 4.0];
        let raw = Bytes::from(values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
        assert_eq!(
            encode_npy_typed(&values, 1, 2, 2, SampleType::Float32, None, false),
            wrap_raw_f32_as_npy(raw.clone(), 1, 2, 2)
        );
        let typed = encode_npy_typed(&values, 1, 2, 2, SampleType::Float32, None, true);
        let wrapped = wrap_raw_f32_as_npy_with_mask(raw, 1, 2, 2, true);
        assert_eq!(typed[..144], wrapped[..144]);
        assert_eq!(&typed[144..148], f32::MAX.to_le_bytes());
        assert_eq!(&wrapped[144..148], 255.0f32.to_le_bytes());
    }

    #[test]
    fn simd_tail_loops_write_all_pixels_for_non_multiple_widths() {
        // N = 5 is neither a multiple of 4 (NEON) nor 8 (AVX2).
        let u = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let v = [1.0f32, 1.0, 1.0, 1.0, 1.0];
        let band = vec![100u8; 64];
        let mut rgb = [0xAAu8; 5 * 3];
        bilinear_rgb_single_block_row::<5, u8, _>(
            &u,
            &v,
            &band,
            &band,
            &band,
            0,
            0,
            8,
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            &mut rgb,
            |x| x as f32,
        );
        assert!(
            rgb.iter().all(|&b| b == 100),
            "tail pixel in bilinear_rgb_single_block_row::<5> was not written: {rgb:?}"
        );

        let r0 = [10.0f32; 7];
        let r1 = [10.0f32; 7];
        let r2 = [10.0f32; 7];
        let mut hs = [0xEEu8; 5];
        eval_hillshade_row_clean::<5>(&r0, &r1, &r2, 315.0, 45.0, 1.0, 1.0, 0.0, &mut hs);
        assert!(
            hs.iter().all(|&b| b == hs[0] && b != 0xEE),
            "tail pixel in eval_hillshade_row_clean::<5> was not written: {hs:?}"
        );
    }

    #[test]
    fn exact2d_coord_plan_scale_handles_nan_center_and_n1() {
        // Audit #68 #2: N == 1 must not panic OOB
        let plan1 = CoordPlan::Exact2D {
            u: vec![5.0].into_boxed_slice(),
            v: vec![5.0].into_boxed_slice(),
            all_in_bounds: true,
            src_min_col: 0,
            src_max_col: 10,
            src_min_row: 0,
            src_max_row: 10,
        };
        assert_eq!(coord_plan_scale::<1>(&plan1), (1.0, 1.0));

        // Exact2D with NaN center (N/2, N/2) but valid 16x downscale in top-left quadrant
        let n = 8usize;
        let mut u = vec![f32::NAN; n * n];
        let mut v = vec![f32::NAN; n * n];
        for py in 0..3 {
            for px in 0..3 {
                u[py * n + px] = (px as f32) * 16.0;
                v[py * n + px] = (py as f32) * 16.0;
            }
        }
        let plan8 = CoordPlan::Exact2D {
            u: u.into_boxed_slice(),
            v: v.into_boxed_slice(),
            all_in_bounds: false,
            src_min_col: 0,
            src_max_col: 48,
            src_min_row: 0,
            src_max_row: 48,
        };
        let (sx, sy) = coord_plan_scale::<8>(&plan8);
        assert!((sx - 16.0).abs() < 1e-4 && (sy - 16.0).abs() < 1e-4, "got ({sx}, {sy})");
    }

    #[test]
    fn audit87_simd_scalar_rounding_parity_and_downscale_footprint_edge() {
        // 1. Clean hillshade: SIMD lanes (0..4) and scalar tail (4) must match exactly.
        // Flat at 40°: TiTiler's 255 * (sin(40°) + 1) / 2 = 209.46, truncated.
        let r0 = [10.0f32; 7];
        let r1 = [10.0f32; 7];
        let r2 = [10.0f32; 7];
        let mut out = [0u8; 5];
        eval_hillshade_row_clean::<5>(&r0, &r1, &r2, 315.0, 40.0, 1.0, 1.0, 0.0, &mut out);
        assert!(
            out.iter().all(|&v| v == out[0]),
            "hillshade SIMD lanes and scalar tail diverged: {out:?}"
        );
        assert_eq!(out[0], 209);

        // 2. Single-block RGB bilinear: SIMD lanes (0..4) and scalar tail (4) must match exactly.
        let r_blk: [u8; 16] = [
            100, 100, 101, 101, 100, 100, 101, 101, 100, 100, 101, 101, 100, 100, 101, 101,
        ];
        let u_row = [1.6f32; 5];
        let v_row = [0.0f32; 5];
        let mut row_rgb = [0u8; 15];
        bilinear_rgb_single_block_row::<5, u8, _>(
            &u_row,
            &v_row,
            &r_blk,
            &r_blk,
            &r_blk,
            0,
            0,
            4,
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
            &mut row_rgb,
            |x| x as f32,
        );
        for px in 0..5 {
            assert_eq!(
                &row_rgb[px * 3..px * 3 + 3],
                &[100, 100, 100],
                "RGB bilinear SIMD/scalar mismatch at px={px}: {row_rgb:?}"
            );
        }

        // 3. Downscaled footprint resampling must not clip centers outside [-0.5, w - 0.5]
        // when the footprint box still overlaps the raster.
        let band = cogzilla_codec::DecodedBand {
            data: cogzilla_codec::BandData::U8 {
                data: vec![10u8; 16 * 16].into_boxed_slice(),
                nodata: None,
            },
            width: 16,
            height: 16,
            stats: cogzilla_codec::BlockStats {
                all_nodata: false,
                no_nodata: true,
                min: 10.0,
                max: 10.0,
            },
        };
        let blocks = [Some(&band)];
        let src = BandSource::new(16, 16, 16, 16, 0, 0, 1, 1, &blocks);
        let avg = sample_general_scaled(&src, -1.0, 4.0, 4.0, 4.0, ResampleMode::Average);
        assert!(
            avg.is_finite() && (avg - 10.0).abs() < 1e-4,
            "expected overlap Average=10.0, got {avg}"
        );
        let sum = sample_general_scaled(&src, -1.0, 4.0, 4.0, 4.0, ResampleMode::Sum);
        // cx = -0.5, footprint x in [-2.5, 1.5] -> overlap width 1.5, height 4.0 -> area 6.0 * 10.0 = 60.0
        assert!(
            sum.is_finite() && (sum - 60.0).abs() < 1e-3,
            "expected overlap Sum=60.0, got {sum}"
        );
        // Completely outside footprint still returns NaN
        assert!(sample_general_scaled(&src, -5.0, 4.0, 4.0, 4.0, ResampleMode::Average).is_nan());
    }

    #[test]
    fn smoothing_resamplers_require_majority_valid_weight() {
        // 8x8 raster where x < 4 is valid (100.0) and x >= 4 is NaN.
        let mut slab = cogzilla_codec::AlignedF32Slab::filled(64, f32::NAN);
        let data = slab.as_mut_slice();
        for y in 0..8 {
            for x in 0..4 {
                data[y * 8 + x] = 100.0;
            }
        }
        let band = cogzilla_codec::DecodedBand {
            data: cogzilla_codec::BandData::F32(slab),
            width: 8,
            height: 8,
            stats: cogzilla_codec::BlockStats {
                all_nodata: false,
                no_nodata: false,
                min: 100.0,
                max: 100.0,
            },
        };
        let blocks = [Some(&band)];
        let src = BandSource::new(8, 8, 8, 8, 0, 0, 1, 1, &blocks);

        // Bilinear at u=3.25, v=4.0: valid weight = 0.75 >= 0.5 -> valid (100.0).
        let v_maj = sample_bilinear::<false>(&src, 3.25, 4.0);
        assert!((v_maj - 100.0).abs() < 1e-4, "expected 100.0 for wsum=0.75, got {v_maj}");
        // Bilinear at u=3.5, v=4.0: valid weight = 0.5 >= 0.5 -> valid (100.0).
        let v_half = sample_bilinear::<false>(&src, 3.5, 4.0);
        assert!((v_half - 100.0).abs() < 1e-4, "expected 100.0 for wsum=0.5, got {v_half}");
        // Bilinear at u=3.75, v=4.0: valid weight = 0.25 < 0.5 -> NaN (no nodata-collar bloat).
        let v_min = sample_bilinear::<false>(&src, 3.75, 4.0);
        assert!(v_min.is_nan(), "expected NaN for wsum=0.25, got {v_min}");

        // CubicSpline & Gauss & widened Bilinear all follow majority-valid (>= 0.5).
        for mode in [ResampleMode::CubicSpline, ResampleMode::Gauss] {
            let maj = sample_general_unwidened(&src, 3.25, 4.0, 1.0, 1.0, mode);
            assert!((maj - 100.0).abs() < 1e-4, "{mode:?} majority expected 100.0, got {maj}");
            let min = sample_general_unwidened(&src, 3.75, 4.0, 1.0, 1.0, mode);
            assert!(min.is_nan(), "{mode:?} minority expected NaN, got {min}");
        }
        let w_maj = sample_general_scaled(&src, 3.25, 4.0, 2.0, 2.0, ResampleMode::Bilinear);
        assert!((w_maj - 100.0).abs() < 1e-4, "widened Bilinear majority expected 100.0, got {w_maj}");
        let w_min = sample_general_scaled(&src, 3.75, 4.0, 2.0, 2.0, ResampleMode::Bilinear);
        assert!(w_min.is_nan(), "widened Bilinear minority expected NaN, got {w_min}");

        // f64 Bilinear also requires majority-valid (>= 0.5).
        let f64_maj = sample_f64_general(&src, 3.25, 4.0, ResampleMode::Bilinear);
        assert!((f64_maj - 100.0).abs() < 1e-9, "f64 Bilinear majority expected 100.0, got {f64_maj}");
        let f64_min = sample_f64_general(&src, 3.75, 4.0, ResampleMode::Bilinear);
        assert!(f64_min.is_nan(), "f64 Bilinear minority expected NaN, got {f64_min}");
    }
}

