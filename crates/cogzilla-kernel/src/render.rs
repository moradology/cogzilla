use std::cell::RefCell;

use bytes::{BufMut, Bytes, BytesMut};
use cogzilla_proj::{CoordPlan, GRID_DIM};

use crate::colormap::{ColormapLut, ColormapName};
use crate::jpeg_fast::{empty_black_jpeg, encode_jpeg_rgba};
use crate::png_fast::{
    empty_transparent_png, encode_filtered_png, filter_sub_ga_row,
    filter_sub_rgb_row, filter_sub_rgba_row, filter_sub_u8_row,
};

#[inline]
fn build_plte_opaque(lut: &ColormapLut) -> &[u8] {
    lut.rgb.as_flattened()
}

mod dem;
mod expr;
mod npy;
mod rgb;
mod single_band;
mod widened;

pub use self::dem::{
    DemAlgorithm, dem_gradient, encode_contours_rgb, encode_terrain_pixel,
    encode_terrain_rgb, encode_terrarium_rgb, eval_dem_stencil, render_dem_algorithm,
    render_terrain_tile, safe_rescale_denom,
};
pub use self::npy::{
    encode_npy_f32_header, encode_npy_header, encode_npy_typed, encode_npy_typed_f64,
    f32_npy_shape, is_valid_mask_sample, npy_descr, retype_f32_npy, retype_f32_npy_with,
    retype_value_with_mask, wrap_raw_f32_as_npy, wrap_raw_f32_as_npy_with_alpha_mask,
    wrap_raw_f32_as_npy_with_mask,
};
use self::widened::{
    AxisTapTable, lanczos3_lut, plan_separable_conv_scale, widened_plane_separable_into_with_tables,
    widened_rgb_separable_stream_clean,
};
pub use self::rgb::render_rgb_bands;
pub use self::single_band::render_single_band;
pub use self::widened::{
    WidenedSampler, gdal_kernel_scale, kernel_widens, plan_widened_scale,
    sample_f64_kernel_widened, sample_kernel_widened, separable_axes, widened_kernel_radius,
    widened_plane, widened_plane_separable,
};
use self::expr::{render_eval_fn, render_plane_single};
pub use self::expr::{
    render_bandmath1, render_bandmath2, render_bandmath3, render_expr_vm, render_sar_rgb,
};

/// A palette for a colour-mapped tile that may have empty pixels, and the colour-map entry
/// that gives up its index (see `ColormapLut::nodata_plte`; `nodata_index` places a byte value).
#[inline]
fn build_plte_nodata(lut: &ColormapLut) -> (&[u8; 768], u8) {
    (&lut.nodata_plte, lut.nodata_merged)
}

use crate::source::BandSource;
use crate::webp_fast::{empty_transparent_webp, encode_webp_rgba};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResampleMode {
    Nearest,
    Bilinear,
    Cubic,
    CubicSpline,
    Lanczos,
    Average,
    Mode,
    Gauss,
    Rms,
    Min,
    Max,
    Med,
    Q1,
    Q3,
    Sum,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OutputFormat {
    Png,
    /// `quality: None` = lossless WebP, `Some(q)` = lossy quality (e.g. `82.0`).
    Webp { quality: Option<f32> },
    /// JPEG (lossy quality `1..=100`, default `85`).
    Jpeg { quality: u8 },
    RawF32,
    /// Raw CHW little-endian `f64` array (`8` bytes/sample), used for unstyled wide-dtype GeoTIFFs.
    RawF64,
    /// NumPy v1.0 `.npy` array (`shape = (C, H, W)`).
    Npy,
    /// Exact wide-dtype NumPy v1.0 `.npy` array rendered via the `f64` data path.
    NpyF64,
    /// Uncompressed RGBA8 (`N * N * 4` bytes), used internally for multi-scene mosaic compositing.
    RawRgba,
}

#[derive(Debug, Clone, Copy)]
pub struct SingleBandStyle {
    pub rescale_min: f32,
    pub rescale_max: f32,
    pub colormap: ColormapName,
    pub resample: ResampleMode,
    pub format: OutputFormat,
}

#[derive(Debug, Clone, Copy)]
pub struct RgbStyle {
    pub rescale_min: [f32; 3],
    pub rescale_max: [f32; 3],
    pub resample: ResampleMode,
    pub format: OutputFormat,
}

struct KernelScratch {
    img_buf: Vec<u8>,
    row_buf: Vec<u8>,
    f32_row: Vec<f32>,
    dem_elev: Vec<f32>,
}

impl KernelScratch {
    fn new() -> Self {
        Self {
            img_buf: Vec::with_capacity(512 * (1 + 512 * 4)),
            row_buf: vec![0u8; 512 * 4],
            f32_row: vec![0.0f32; 512],
            dem_elev: Vec::new(),
        }
    }
}

thread_local! {
    static SCRATCH: RefCell<KernelScratch> = RefCell::new(KernelScratch::new());
}

// ============================================================================
// SIMD Fast-Path Row Bilinear Samplers (Separable1D + Clean + Single-Block)
// ============================================================================

/// Compute an entire output row of `N` bilinearly resampled `f32` values directly from a single
/// flat source block using ARM64 NEON 4-wide FMA (hoisting `row0_ptr`, `row1_ptr`, and `fy`).
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn bilinear_row_flat_neon<const N: usize>(
    u_coords: &[f32; N],
    base_col: u32,
    row0_ptr: *const f32,
    row1_ptr: *const f32,
    max_x: i32,
    fy: f32,
    out_row: &mut [f32],
) {
    use std::arch::aarch64::*;
    let v_fy = unsafe { vdupq_n_f32(fy) };
    let v_base = unsafe { vdupq_n_f32(base_col as f32) };
    let v_zero = unsafe { vdupq_n_s32(0) };
    let v_max_x = unsafe { vdupq_n_s32(max_x) };
    let mut px = 0usize;

    while px + 4 <= N {
        unsafe {
            let u = vsubq_f32(vld1q_f32(u_coords.as_ptr().add(px)), v_base);
            let x0_f = vrndmq_f32(u);
            let fx = vsubq_f32(u, x0_f);
            let x0_i = vminq_s32(vmaxq_s32(vcvtq_s32_f32(x0_f), v_zero), v_max_x);

            let c0 = vgetq_lane_s32::<0>(x0_i) as usize;
            let c1 = vgetq_lane_s32::<1>(x0_i) as usize;
            let c2 = vgetq_lane_s32::<2>(x0_i) as usize;
            let c3 = vgetq_lane_s32::<3>(x0_i) as usize;

            // Load adjacent (v00, v10) pairs with single 64-bit loads + unzip
            let t0 = vld1_f32(row0_ptr.add(c0));
            let t1 = vld1_f32(row0_ptr.add(c1));
            let t2 = vld1_f32(row0_ptr.add(c2));
            let t3 = vld1_f32(row0_ptr.add(c3));
            let t01 = vcombine_f32(t0, t1);
            let t23 = vcombine_f32(t2, t3);
            let v00 = vuzp1q_f32(t01, t23);
            let v10 = vuzp2q_f32(t01, t23);

            let b0 = vld1_f32(row1_ptr.add(c0));
            let b1 = vld1_f32(row1_ptr.add(c1));
            let b2 = vld1_f32(row1_ptr.add(c2));
            let b3 = vld1_f32(row1_ptr.add(c3));
            let b01 = vcombine_f32(b0, b1);
            let b23 = vcombine_f32(b2, b3);
            let v01 = vuzp1q_f32(b01, b23);
            let v11 = vuzp2q_f32(b01, b23);

            let top = vfmaq_f32(v00, vsubq_f32(v10, v00), fx);
            let bot = vfmaq_f32(v01, vsubq_f32(v11, v01), fx);
            let res = vfmaq_f32(top, vsubq_f32(bot, top), v_fy);

            vst1q_f32(out_row.as_mut_ptr().add(px), res);
        }
        px += 4;
    }
    let bc_f = base_col as f32;
    while px < N {
        let (x0_raw, fx) = fast_floor_i32(u_coords[px] - bc_f);
        let c = x0_raw.clamp(0, max_x) as usize;
        unsafe {
            let v00 = *row0_ptr.add(c);
            let v10 = *row0_ptr.add(c + 1);
            let v01 = *row1_ptr.add(c);
            let v11 = *row1_ptr.add(c + 1);
            let top = v00 + (v10 - v00) * fx;
            let bot = v01 + (v11 - v01) * fx;
            *out_row.get_unchecked_mut(px) = top + (bot - top) * fy;
        }
        px += 1;
    }
}

/// x86_64 AVX2+FMA 8-wide bilinear row sampler for flat single-block `Separable1D + CleanFastPath`.
/// Uses 64-bit adjacent-pair loads + shuffle instead of microcoded `vgatherdps` (much faster on
/// AMD Zen 4 / EPYC as well as Intel).
// SAFETY: Caller checks AVX2+FMA and guarantees `row0_ptr`/`row1_ptr` have `>= max_x + 2` elements and `out_row.len() >= N`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn bilinear_row_flat_avx2<const N: usize>(
    u_coords: &[f32; N],
    base_col: u32,
    row0_ptr: *const f32,
    row1_ptr: *const f32,
    max_x: i32,
    fy: f32,
    out_row: &mut [f32],
) {
    use std::arch::x86_64::*;
    let v_fy = _mm256_set1_ps(fy);
    let v_base = _mm256_set1_ps(base_col as f32);
    let v_zero = _mm256_setzero_si256();
    let v_max_x = _mm256_set1_epi32(max_x);
    let mut idxs = [0i32; 8];
    let mut px = 0usize;

    macro_rules! load_8_pairs {
        ($ptr:expr) => {{
            let c0 = idxs[0] as usize;
            let c1 = idxs[1] as usize;
            let c2 = idxs[2] as usize;
            let c3 = idxs[3] as usize;
            let c4 = idxs[4] as usize;
            let c5 = idxs[5] as usize;
            let c6 = idxs[6] as usize;
            let c7 = idxs[7] as usize;

            // SAFETY: Each `c` is clamped to `0..=max_x`, so `c` and `c + 1` (8 bytes) are in-bounds.
            let (p01, p23, p45, p67) = unsafe {
                (
                    _mm_castpd_ps(_mm_set_pd(
                        ($ptr.add(c1) as *const f64).read_unaligned(),
                        ($ptr.add(c0) as *const f64).read_unaligned(),
                    )),
                    _mm_castpd_ps(_mm_set_pd(
                        ($ptr.add(c3) as *const f64).read_unaligned(),
                        ($ptr.add(c2) as *const f64).read_unaligned(),
                    )),
                    _mm_castpd_ps(_mm_set_pd(
                        ($ptr.add(c5) as *const f64).read_unaligned(),
                        ($ptr.add(c4) as *const f64).read_unaligned(),
                    )),
                    _mm_castpd_ps(_mm_set_pd(
                        ($ptr.add(c7) as *const f64).read_unaligned(),
                        ($ptr.add(c6) as *const f64).read_unaligned(),
                    )),
                )
            };

            let lo0 = _mm_shuffle_ps::<0x88>(p01, p23);
            let lo1 = _mm_shuffle_ps::<0xDD>(p01, p23);
            let hi0 = _mm_shuffle_ps::<0x88>(p45, p67);
            let hi1 = _mm_shuffle_ps::<0xDD>(p45, p67);
            (
                _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(lo0), hi0),
                _mm256_insertf128_ps::<1>(_mm256_castps128_ps256(lo1), hi1),
            )
        }};
    }

    while px + 8 <= N {
        // SAFETY: `px + 8 <= N` guarantees 8 readable `f32`s in `u_coords` and 8 writable `i32`s in `idxs`.
        let u = _mm256_sub_ps(unsafe { _mm256_loadu_ps(u_coords.as_ptr().add(px)) }, v_base);
        let x0_f = _mm256_floor_ps(u);
        let fx = _mm256_sub_ps(u, x0_f);
        let idx = _mm256_min_epi32(_mm256_max_epi32(_mm256_cvtps_epi32(x0_f), v_zero), v_max_x);
        unsafe { _mm256_storeu_si256(idxs.as_mut_ptr() as *mut __m256i, idx) };

        let (v00, v10) = load_8_pairs!(row0_ptr);
        let (v01, v11) = load_8_pairs!(row1_ptr);

        let top = _mm256_fmadd_ps(_mm256_sub_ps(v10, v00), fx, v00);
        let bot = _mm256_fmadd_ps(_mm256_sub_ps(v11, v01), fx, v01);
        let res = _mm256_fmadd_ps(_mm256_sub_ps(bot, top), v_fy, top);

        // SAFETY: `px + 8 <= N <= out_row.len()`.
        unsafe { _mm256_storeu_ps(out_row.as_mut_ptr().add(px), res) };
        px += 8;
    }
    let bc_f = base_col as f32;
    while px < N {
        let (x0_raw, fx) = fast_floor_i32(u_coords[px] - bc_f);
        let c = x0_raw.clamp(0, max_x) as usize;
        // SAFETY: `c <= max_x` and `px < N <= out_row.len()`.
        unsafe {
            let v00 = *row0_ptr.add(c);
            let v10 = *row0_ptr.add(c + 1);
            let v01 = *row1_ptr.add(c);
            let v11 = *row1_ptr.add(c + 1);
            let top = v00 + (v10 - v00) * fx;
            let bot = v01 + (v11 - v01) * fx;
            *out_row.get_unchecked_mut(px) = top + (bot - top) * fy;
        }
        px += 1;
    }
}

#[inline(always)]
fn bilinear_row_flat_dispatch<const N: usize>(
    u_coords: &[f32; N],
    base_col: u32,
    row0: &[f32],
    row1: &[f32],
    fy: f32,
    out_row: &mut [f32],
) {
    let row_len = row0.len().min(row1.len());
    if row_len < 2 || out_row.len() < N {
        return;
    }
    let max_x = (row_len as i32) - 2;
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            bilinear_row_flat_neon::<N>(
                u_coords,
                base_col,
                row0.as_ptr(),
                row1.as_ptr(),
                max_x,
                fy,
                out_row,
            );
        }
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            unsafe {
                bilinear_row_flat_avx2::<N>(
                    u_coords,
                    base_col,
                    row0.as_ptr(),
                    row1.as_ptr(),
                    max_x,
                    fy,
                    out_row,
                );
            }
            return;
        }
    }
    #[allow(unreachable_code)]
    {
        let bc = base_col as f32;
        let max_x = row0.len().min(row1.len()).saturating_sub(2);
        for px in 0..N {
            let u = u_coords[px] - bc;
            let x0_f = u.floor();
            let fx = u - x0_f;
            let x0 = (x0_f.max(0.0) as usize).min(max_x);
            let v00 = unsafe { *row0.get_unchecked(x0) };
            let v10 = unsafe { *row0.get_unchecked(x0 + 1) };
            let v01 = unsafe { *row1.get_unchecked(x0) };
            let v11 = unsafe { *row1.get_unchecked(x0 + 1) };
            let top = v00 + (v10 - v00) * fx;
            let bot = v01 + (v11 - v01) * fx;
            out_row[px] = top + (bot - top) * fy;
        }
    }
}

/// ARM NEON 4-wide bilinear row sampler for single-block `U16`/`U8` (`Separable1D + CleanFastPath`).
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn bilinear_row_typed_neon<T: Copy, F: Fn(T) -> f32, const N: usize>(
    u_coords: &[f32; N],
    base_col: u32,
    row0_ptr: *const T,
    row1_ptr: *const T,
    max_x: i32,
    fy: f32,
    out_row: &mut [f32],
    to_f32: &F,
) {
    use std::arch::aarch64::*;
    let (v_fy, v_base, v_zero, v_max_x) = unsafe {
        (
            vdupq_n_f32(fy),
            vdupq_n_f32(base_col as f32),
            vdupq_n_s32(0),
            vdupq_n_s32(max_x),
        )
    };
    let mut idxs = [0i32; 4];
    let mut px = 0usize;
    while px + 4 <= N {
        unsafe {
            let u = vsubq_f32(vld1q_f32(u_coords.as_ptr().add(px)), v_base);
            let x0_f = vrndmq_f32(u);
            let fx = vsubq_f32(u, x0_f);
            let idx = vminq_s32(vmaxq_s32(vcvtq_s32_f32(x0_f), v_zero), v_max_x);
            vst1q_s32(idxs.as_mut_ptr(), idx);

            let c0 = idxs[0] as usize;
            let c1 = idxs[1] as usize;
            let c2 = idxs[2] as usize;
            let c3 = idxs[3] as usize;

            let a00 = [
                to_f32(*row0_ptr.add(c0)),
                to_f32(*row0_ptr.add(c1)),
                to_f32(*row0_ptr.add(c2)),
                to_f32(*row0_ptr.add(c3)),
            ];
            let a10 = [
                to_f32(*row0_ptr.add(c0 + 1)),
                to_f32(*row0_ptr.add(c1 + 1)),
                to_f32(*row0_ptr.add(c2 + 1)),
                to_f32(*row0_ptr.add(c3 + 1)),
            ];
            let a01 = [
                to_f32(*row1_ptr.add(c0)),
                to_f32(*row1_ptr.add(c1)),
                to_f32(*row1_ptr.add(c2)),
                to_f32(*row1_ptr.add(c3)),
            ];
            let a11 = [
                to_f32(*row1_ptr.add(c0 + 1)),
                to_f32(*row1_ptr.add(c1 + 1)),
                to_f32(*row1_ptr.add(c2 + 1)),
                to_f32(*row1_ptr.add(c3 + 1)),
            ];

            let v00 = vld1q_f32(a00.as_ptr());
            let v10 = vld1q_f32(a10.as_ptr());
            let v01 = vld1q_f32(a01.as_ptr());
            let v11 = vld1q_f32(a11.as_ptr());

            let top = vfmaq_f32(v00, vsubq_f32(v10, v00), fx);
            let bot = vfmaq_f32(v01, vsubq_f32(v11, v01), fx);
            let res = vfmaq_f32(top, vsubq_f32(bot, top), v_fy);
            vst1q_f32(out_row.as_mut_ptr().add(px), res);
        }
        px += 4;
    }
    let bc_f = base_col as f32;
    while px < N {
        let (x0_raw, fx) = fast_floor_i32(u_coords[px] - bc_f);
        let c = x0_raw.clamp(0, max_x) as usize;
        unsafe {
            let v00 = to_f32(*row0_ptr.add(c));
            let v10 = to_f32(*row0_ptr.add(c + 1));
            let v01 = to_f32(*row1_ptr.add(c));
            let v11 = to_f32(*row1_ptr.add(c + 1));
            let top = v00 + (v10 - v00) * fx;
            let bot = v01 + (v11 - v01) * fx;
            *out_row.get_unchecked_mut(px) = top + (bot - top) * fy;
        }
        px += 1;
    }
}

/// x86_64 AVX2+FMA 8-wide bilinear row sampler for single-block `U16`/`U8` (`Separable1D + CleanFastPath`).
// SAFETY: Caller checks AVX2+FMA and guarantees `row0_ptr`/`row1_ptr` have `>= max_x + 2` elements and `out_row.len() >= N`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn bilinear_row_typed_avx2<T: Copy, F: Fn(T) -> f32, const N: usize>(
    u_coords: &[f32; N],
    base_col: u32,
    row0_ptr: *const T,
    row1_ptr: *const T,
    max_x: i32,
    fy: f32,
    out_row: &mut [f32],
    to_f32: &F,
) {
    use std::arch::x86_64::*;
    let v_fy = _mm256_set1_ps(fy);
    let v_base = _mm256_set1_ps(base_col as f32);
    let v_zero = _mm256_setzero_si256();
    let v_max_x = _mm256_set1_epi32(max_x);
    let mut idxs = [0i32; 8];
    let mut px = 0usize;
    while px + 8 <= N {
        // SAFETY: `px + 8 <= N` guarantees 8 readable `f32`s in `u_coords` and 8 writable `i32`s in `idxs`.
        let u = _mm256_sub_ps(unsafe { _mm256_loadu_ps(u_coords.as_ptr().add(px)) }, v_base);
        let x0_f = _mm256_floor_ps(u);
        let fx = _mm256_sub_ps(u, x0_f);
        let idx = _mm256_min_epi32(_mm256_max_epi32(_mm256_cvtps_epi32(x0_f), v_zero), v_max_x);
        unsafe { _mm256_storeu_si256(idxs.as_mut_ptr() as *mut __m256i, idx) };

        let mut a00 = [0.0f32; 8];
        let mut a10 = [0.0f32; 8];
        let mut a01 = [0.0f32; 8];
        let mut a11 = [0.0f32; 8];
        for k in 0..8 {
            let c = idxs[k] as usize;
            // SAFETY: `c` is clamped to `0..=max_x`, so `c` and `c + 1` are in-bounds.
            unsafe {
                a00[k] = to_f32(*row0_ptr.add(c));
                a10[k] = to_f32(*row0_ptr.add(c + 1));
                a01[k] = to_f32(*row1_ptr.add(c));
                a11[k] = to_f32(*row1_ptr.add(c + 1));
            }
        }

        // SAFETY: `a00..a11` are 8-element `f32` stack arrays (32 bytes each).
        let (v00, v10, v01, v11) = unsafe {
            (
                _mm256_loadu_ps(a00.as_ptr()),
                _mm256_loadu_ps(a10.as_ptr()),
                _mm256_loadu_ps(a01.as_ptr()),
                _mm256_loadu_ps(a11.as_ptr()),
            )
        };

        let top = _mm256_fmadd_ps(_mm256_sub_ps(v10, v00), fx, v00);
        let bot = _mm256_fmadd_ps(_mm256_sub_ps(v11, v01), fx, v01);
        let res = _mm256_fmadd_ps(_mm256_sub_ps(bot, top), v_fy, top);
        // SAFETY: `px + 8 <= N <= out_row.len()`.
        unsafe { _mm256_storeu_ps(out_row.as_mut_ptr().add(px), res) };
        px += 8;
    }
    let bc_f = base_col as f32;
    while px < N {
        let (x0_raw, fx) = fast_floor_i32(u_coords[px] - bc_f);
        let c = x0_raw.clamp(0, max_x) as usize;
        // SAFETY: `c <= max_x` and `px < N <= out_row.len()`.
        unsafe {
            let v00 = to_f32(*row0_ptr.add(c));
            let v10 = to_f32(*row0_ptr.add(c + 1));
            let v01 = to_f32(*row1_ptr.add(c));
            let v11 = to_f32(*row1_ptr.add(c + 1));
            let top = v00 + (v10 - v00) * fx;
            let bot = v01 + (v11 - v01) * fx;
            *out_row.get_unchecked_mut(px) = top + (bot - top) * fy;
        }
        px += 1;
    }
}

#[inline(always)]
fn bilinear_row_typed_dispatch<T: Copy, F: Fn(T) -> f32, const N: usize>(
    u_coords: &[f32; N],
    base_col: u32,
    row0: &[T],
    row1: &[T],
    fy: f32,
    out_row: &mut [f32],
    to_f32: F,
) {
    let row_len = row0.len().min(row1.len());
    if row_len < 2 || out_row.len() < N {
        return;
    }
    let max_x = (row_len as i32) - 2;
    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            bilinear_row_typed_neon::<T, F, N>(
                u_coords,
                base_col,
                row0.as_ptr(),
                row1.as_ptr(),
                max_x,
                fy,
                out_row,
                &to_f32,
            );
        }
        return;
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            unsafe {
                bilinear_row_typed_avx2::<T, F, N>(
                    u_coords,
                    base_col,
                    row0.as_ptr(),
                    row1.as_ptr(),
                    max_x,
                    fy,
                    out_row,
                    &to_f32,
                );
            }
            return;
        }
    }
    #[allow(unreachable_code)]
    {
        let bc = base_col as f32;
        let max_x_u = row_len.saturating_sub(2);
        for px in 0..N {
            let (x0_raw, fx) = fast_floor_i32(u_coords[px] - bc);
            let x0 = (x0_raw.max(0) as usize).min(max_x_u);
            let v00 = to_f32(unsafe { *row0.get_unchecked(x0) });
            let v10 = to_f32(unsafe { *row0.get_unchecked(x0 + 1) });
            let v01 = to_f32(unsafe { *row1.get_unchecked(x0) });
            let v11 = to_f32(unsafe { *row1.get_unchecked(x0 + 1) });
            let top = v00 + (v10 - v00) * fx;
            let bot = v01 + (v11 - v01) * fx;
            out_row[px] = top + (bot - top) * fy;
        }
    }
}

/// Unified single-block typed view for clean (`no_nodata`, exact `f32` base, `w/h >= 2`) bands.
#[derive(Clone, Copy)]
enum SingleBlockRef<'a> {
    F32(&'a [f32], u32, u32, usize),
    U16(&'a [u16], u32, u32, usize),
    U8(&'a [u8], u32, u32, usize),
}

impl<'a> SingleBlockRef<'a> {
    #[inline(always)]
    fn from_source(src: &BandSource<'a>) -> Option<Self> {
        if let Some((s, bc, br, st)) = src.single_block_flat() {
            Some(Self::F32(s, bc, br, st))
        } else if let Some((s, bc, br, st)) = src.single_block_u16() {
            Some(Self::U16(s, bc, br, st))
        } else if let Some((s, bc, br, st)) = src.single_block_u8() {
            Some(Self::U8(s, bc, br, st))
        } else {
            None
        }
    }

    #[inline(always)]
    fn base_col(&self) -> u32 {
        match *self {
            Self::F32(_, bc, _, _) | Self::U16(_, bc, _, _) | Self::U8(_, bc, _, _) => bc,
        }
    }

    #[inline(always)]
    fn base_row(&self) -> u32 {
        match *self {
            Self::F32(_, _, br, _) | Self::U16(_, _, br, _) | Self::U8(_, _, br, _) => br,
        }
    }

    #[inline(always)]
    fn row_stride(&self) -> usize {
        match *self {
            Self::F32(_, _, _, st) | Self::U16(_, _, _, st) | Self::U8(_, _, _, st) => st,
        }
    }

    #[inline(always)]
    fn height(&self) -> usize {
        match *self {
            Self::F32(s, _, _, st) => s.len() / st,
            Self::U16(s, _, _, st) => s.len() / st,
            Self::U8(s, _, _, st) => s.len() / st,
        }
    }

    #[inline(always)]
    fn precompute_nearest_x<const N: usize>(&self, u: &[f32; N]) -> [usize; N] {
        let max_x = (self.row_stride() as i32 - 1).max(0);
        let bc_f = self.base_col() as f32;
        let mut x_idx = [0usize; N];
        for px in 0..N {
            x_idx[px] = fast_floor_i32(u[px] + 0.5 - bc_f).0.clamp(0, max_x) as usize;
        }
        x_idx
    }

    #[inline(always)]
    fn sample_nearest_row_f32<const N: usize>(
        &self,
        x_idx: &[usize; N],
        vy: f32,
        out: &mut [f32],
    ) {
        let max_y = (self.height() as i32 - 1).max(0);
        let y0 = fast_floor_i32(vy + 0.5 - self.base_row() as f32)
            .0
            .clamp(0, max_y) as usize;
        match *self {
            Self::F32(s, _, _, st) => {
                let row = &s[y0 * st..];
                for px in 0..N {
                    out[px] = row[x_idx[px]];
                }
            }
            Self::U16(s, _, _, st) => {
                let row = &s[y0 * st..];
                for px in 0..N {
                    out[px] = row[x_idx[px]] as f32;
                }
            }
            Self::U8(s, _, _, st) => {
                let row = &s[y0 * st..];
                for px in 0..N {
                    out[px] = row[x_idx[px]] as f32;
                }
            }
        }
    }

    #[inline(always)]
    fn sample_bilinear_row_f32<const N: usize>(&self, u: &[f32; N], vy: f32, out: &mut [f32]) {
        let max_y = (self.height() as i32 - 2).max(0);
        let (y0_raw, fy) = fast_floor_i32(vy - self.base_row() as f32);
        let y0 = y0_raw.clamp(0, max_y) as usize;
        match *self {
            Self::F32(s, bc, _, st) => {
                let r0 = &s[y0 * st..(y0 + 1) * st];
                let r1 = &s[(y0 + 1) * st..(y0 + 2) * st];
                bilinear_row_flat_dispatch::<N>(u, bc, r0, r1, fy, out);
            }
            Self::U16(s, bc, _, st) => {
                let r0 = &s[y0 * st..(y0 + 1) * st];
                let r1 = &s[(y0 + 1) * st..(y0 + 2) * st];
                bilinear_row_typed_dispatch::<u16, _, N>(u, bc, r0, r1, fy, out, |v| v as f32);
            }
            Self::U8(s, bc, _, st) => {
                let r0 = &s[y0 * st..(y0 + 1) * st];
                let r1 = &s[(y0 + 1) * st..(y0 + 2) * st];
                bilinear_row_typed_dispatch::<u8, _, N>(u, bc, r0, r1, fy, out, |v| v as f32);
            }
        }
    }
}

// ============================================================================
// General Per-Pixel Samplers
// ============================================================================

/// Branchless inline floor returning `(floor_i32, frac)` without calling libm `floorf` on
/// baseline x86-64 SSE2 builds (where `f32::floor` emits a `floorf@PLT` call per pixel).
#[inline(always)]
fn fast_floor_i32(x: f32) -> (i32, f32) {
    #[cfg(any(target_arch = "aarch64", target_feature = "sse4.1"))]
    {
        let f = x.floor();
        (f as i32, x - f)
    }
    #[cfg(not(any(target_arch = "aarch64", target_feature = "sse4.1")))]
    {
        let i = x as i32;
        let adj = i.saturating_sub((i as f32 > x) as i32);
        (adj, x - (adj as f32))
    }
}

#[inline(always)]
fn sample_bilinear<const CLEAN: bool>(src: &BandSource<'_>, u: f32, v: f32) -> f32 {
    if !CLEAN && !src.contains_sample(u, v) {
        return f32::NAN;
    }
    let (x0, fx) = fast_floor_i32(u);
    let (y0, fy) = fast_floor_i32(v);

    if CLEAN {
        let (v00, v10, v01, v11) = src.sample_quad_inbounds(x0 as u32, y0 as u32);
        let top = v00 + (v10 - v00) * fx;
        let bot = v01 + (v11 - v01) * fx;
        top + (bot - top) * fy
    } else {
        let (v00, v10, v01, v11) = if x0 >= 0
            && y0 >= 0
            && x0 + 1 < src.ifd_w
            && y0 + 1 < src.ifd_h
            && src.nodata_override.is_none()
            && src.scale_offset.is_none()
        {
            src.sample_quad_inbounds(x0 as u32, y0 as u32)
        } else {
            (
                src.sample_checked(x0, y0),
                src.sample_checked(x0 + 1, y0),
                src.sample_checked(x0, y0 + 1),
                src.sample_checked(x0 + 1, y0 + 1),
            )
        };

        let w00 = (1.0 - fx) * (1.0 - fy);
        let w10 = fx * (1.0 - fy);
        let w01 = (1.0 - fx) * fy;
        let w11 = fx * fy;

        let w_lim = src.ifd_w;
        let h_lim = src.ifd_h;
        let in0x = x0 >= 0 && x0 < w_lim;
        let in1x = x0 + 1 >= 0 && x0 + 1 < w_lim;
        let in0y = y0 >= 0 && y0 < h_lim;
        let in1y = y0 + 1 >= 0 && y0 + 1 < h_lim;

        let mut sum = 0.0f32;
        let mut wsum = 0.0f32;
        let mut wtot = 0.0f32;
        if in0x && in0y {
            wtot += w00;
            if !v00.is_nan() {
                sum += v00 * w00;
                wsum += w00;
            }
        }
        if in1x && in0y {
            wtot += w10;
            if !v10.is_nan() {
                sum += v10 * w10;
                wsum += w10;
            }
        }
        if in0x && in1y {
            wtot += w01;
            if !v01.is_nan() {
                sum += v01 * w01;
                wsum += w01;
            }
        }
        if in1x && in1y {
            wtot += w11;
            if !v11.is_nan() {
                sum += v11 * w11;
                wsum += w11;
            }
        }
        if wsum >= 0.5 * wtot && wsum > 1e-6 {
            sum / wsum
        } else {
            f32::NAN
        }
    }
}

#[inline(always)]
fn sample_nearest<const CLEAN: bool>(src: &BandSource<'_>, u: f32, v: f32) -> f32 {
    if !CLEAN && (!u.is_finite() || !v.is_finite()) {
        return f32::NAN;
    }
    let (x, _) = fast_floor_i32(u + 0.5);
    let (y, _) = fast_floor_i32(v + 0.5);
    if CLEAN {
        src.sample_inbounds(x as u32, y as u32)
    } else {
        src.sample_checked(x, y)
    }
}

/// Branchless Keys/Catmull-Rom bicubic weights (`a = -0.5`, matching GDAL `GRA_Cubic`)
/// for fractional offset `t in [0.0, 1.0]`.
#[inline(always)]
fn cubic_weights(t: f32) -> [f32; 4] {
    let t2 = t * t;
    let t3 = t2 * t;
    [
        (-0.5 * t3) + t2 - (0.5 * t),
        (1.5 * t3) - (2.5 * t2) + 1.0,
        (-1.5 * t3) + (2.0 * t2) + (0.5 * t),
        (0.5 * t3) - (0.5 * t2),
    ]
}

#[inline(always)]
pub fn sample_cubic(src: &BandSource<'_>, u: f32, v: f32) -> f32 {
    if !src.contains_sample(u, v) {
        return f32::NAN;
    }
    let (x0, fx) = fast_floor_i32(u);
    let (y0, fy) = fast_floor_i32(v);
    let wx = cubic_weights(fx);
    let wy = cubic_weights(fy);

    // Fast SIMD path: all 16 samples lie strictly inside one clean f32 block -> 4 contiguous 16B loads!
    if let Some((base_ptr, stride)) = src.interior_4x4_f32(x0, y0) {
        #[cfg(target_arch = "aarch64")]
        unsafe {
            use std::arch::aarch64::*;
            let vwx = vld1q_f32(wx.as_ptr());
            let r0 = vld1q_f32(base_ptr);
            let r1 = vld1q_f32(base_ptr.add(stride));
            let r2 = vld1q_f32(base_ptr.add(stride * 2));
            let r3 = vld1q_f32(base_ptr.add(stride * 3));
            // Vertical blend of the 4 rows first: acc = r0*wy[0] + r1*wy[1] + r2*wy[2] + r3*wy[3]
            let mut col = vmulq_n_f32(r0, wy[0]);
            col = vfmaq_n_f32(col, r1, wy[1]);
            col = vfmaq_n_f32(col, r2, wy[2]);
            col = vfmaq_n_f32(col, r3, wy[3]);
            return vaddvq_f32(vmulq_f32(col, vwx));
        }
        #[cfg(target_arch = "x86_64")]
        unsafe {
            use std::arch::x86_64::*;
            let vwx = _mm_loadu_ps(wx.as_ptr());
            let r0 = _mm_loadu_ps(base_ptr);
            let r1 = _mm_loadu_ps(base_ptr.add(stride));
            let r2 = _mm_loadu_ps(base_ptr.add(stride * 2));
            let r3 = _mm_loadu_ps(base_ptr.add(stride * 3));
            let col = _mm_add_ps(
                _mm_add_ps(_mm_mul_ps(r0, _mm_set1_ps(wy[0])), _mm_mul_ps(r1, _mm_set1_ps(wy[1]))),
                _mm_add_ps(_mm_mul_ps(r2, _mm_set1_ps(wy[2])), _mm_mul_ps(r3, _mm_set1_ps(wy[3]))),
            );
            let prod = _mm_mul_ps(col, vwx);
            let shuf = _mm_movehl_ps(prod, prod);
            let sums = _mm_add_ps(prod, shuf);
            let shuf2 = _mm_shuffle_ps::<1>(sums, sums);
            return _mm_cvtss_f32(_mm_add_ss(sums, shuf2));
        }
        #[allow(unreachable_code)]
        unsafe {
            let mut acc = 0.0f32;
            for j in 0..4 {
                let rp = base_ptr.add(j * stride);
                let row = *rp * wx[0] + *rp.add(1) * wx[1] + *rp.add(2) * wx[2] + *rp.add(3) * wx[3];
                acc += row * wy[j];
            }
            return acc;
        }
    }

    let mut acc = 0.0f32;
    for j in 0..4 {
        let mut row_acc = 0.0f32;
        let ry = y0 + (j as i32) - 1;
        for i in 0..4 {
            let v_s = src.sample_checked(x0 + (i as i32) - 1, ry);
            if v_s.is_nan() {
                return sample_bilinear::<false>(src, u, v);
            }
            row_acc += v_s * wx[i];
        }
        acc += row_acc * wy[j];
    }
    acc
}

#[inline(always)]
fn cubic_spline_weights(t: f32) -> [f32; 4] {
    let t2 = t * t;
    let t3 = t2 * t;
    let omt = 1.0 - t;
    [
        (omt * omt * omt) * (1.0 / 6.0),
        (3.0 * t3 - 6.0 * t2 + 4.0) * (1.0 / 6.0),
        (-3.0 * t3 + 3.0 * t2 + 3.0 * t + 1.0) * (1.0 / 6.0),
        t3 * (1.0 / 6.0),
    ]
}

#[inline(always)]
fn lanczos3_weight(x: f32) -> f32 {
    lanczos3_lut(x.abs())
}

/// Separable plan over output pixel indices (`u[px] = px`, `v[py] = py`), used to feed a
/// precomputed `N x N` plane through the per-pixel encoders.
fn index_plan<const N: usize>() -> CoordPlan<N> {
    let mut u = Box::new([0.0f32; N]);
    for (i, x) in u.iter_mut().enumerate() {
        *x = i as f32;
    }
    CoordPlan::Separable1D {
        v: u.clone(),
        u,
        src_min_col: 0,
        src_max_col: N as i32 - 1,
        src_min_row: 0,
        src_max_row: N as i32 - 1,
        all_in_bounds: true,
    }
}

#[inline(always)]
fn plane_at<const N: usize>(plane: &[f32], u: f32, v: f32) -> f32 {
    plane[(v as usize) * N + (u as usize)]
}

/// General sampler supporting all 15 GDAL/TiTiler resampling methods at 1:1 source scale.
#[inline]
pub fn sample_general(src: &BandSource<'_>, u: f32, v: f32, mode: ResampleMode) -> f32 {
    sample_general_scaled(src, u, v, 1.0, 1.0, mode)
}

/// General sampler supporting all 15 GDAL/TiTiler resampling methods with an explicit
/// destination-to-source pixel footprint `(scale_x, scale_y)` in source pixel units.
///
/// For footprint-based modes (`Average`, `Mode`, `Rms`, `Min`, `Max`, `Med`, `Q1`, `Q3`, `Sum`),
/// integrates over the exact source-pixel box `[u + 0.5 - 0.5*scale_x, u + 0.5 + 0.5*scale_x] x
/// [v + 0.5 - 0.5*scale_y, v + 0.5 + 0.5*scale_y]` weighted by exact overlap area:
/// - At 1:1 scale (`scale_x == 1, scale_y == 1`), reads the exact single source pixel (0 blur).
/// - At exact integer factors (2x, 4x, 8x, ...), computes the exact block mean / mode / stat.
/// - At non-integer factors, weights each overlapped source pixel by its exact overlap area.
///
/// `Bilinear`, `Cubic`, `CubicSpline` and `Lanczos` widen their kernel like GDAL when the footprint
/// shrinks the image (see [`sample_kernel_widened`]).
#[inline]
pub fn sample_general_scaled(
    src: &BandSource<'_>,
    u: f32,
    v: f32,
    scale_x: f32,
    scale_y: f32,
    mode: ResampleMode,
) -> f32 {
    if kernel_widens(mode, scale_x, scale_y) {
        return sample_kernel_widened(src, u, v, scale_x, scale_y, mode);
    }
    sample_general_unwidened(src, u, v, scale_x, scale_y, mode)
}

/// [`sample_general_scaled`] for callers that already sent widened convolution kernels to
/// [`sample_kernel_widened`]: fixed-size kernels (1:1 and upsampling) and footprint modes.
#[inline]
pub fn sample_general_unwidened(
    src: &BandSource<'_>,
    u: f32,
    v: f32,
    scale_x: f32,
    scale_y: f32,
    mode: ResampleMode,
) -> f32 {
    if !u.is_finite() || !v.is_finite() {
        return f32::NAN;
    }
    if matches!(
        mode,
        ResampleMode::Average
            | ResampleMode::Mode
            | ResampleMode::Rms
            | ResampleMode::Min
            | ResampleMode::Max
            | ResampleMode::Med
            | ResampleMode::Q1
            | ResampleMode::Q3
            | ResampleMode::Sum
    ) {
        return sample_footprint_exact(src, u, v, scale_x, scale_y, mode);
    }
    if !src.contains_sample(u, v) {
        return f32::NAN;
    }
    match mode {
        ResampleMode::Nearest => sample_nearest::<false>(src, u, v),
        ResampleMode::Bilinear => sample_bilinear::<false>(src, u, v),
        ResampleMode::Cubic => sample_cubic(src, u, v),
        ResampleMode::CubicSpline => {
            let (x0, fx) = fast_floor_i32(u);
            let (y0, fy) = fast_floor_i32(v);
            let wx = cubic_spline_weights(fx);
            let wy = cubic_spline_weights(fy);
            let mut sum = 0.0f32;
            let mut wsum = 0.0f32;
            for j in 0..4 {
                let ry = y0 + (j as i32) - 1;
                let wyj = wy[j];
                for i in 0..4 {
                    let s = src.sample_checked(x0 + (i as i32) - 1, ry);
                    if !s.is_nan() {
                        let w = wx[i] * wyj;
                        sum += s * w;
                        wsum += w;
                    }
                }
            }
            if wsum >= 0.5 { sum / wsum } else { f32::NAN }
        }
        ResampleMode::Lanczos => {
            let (x0, fx) = fast_floor_i32(u);
            let (y0, fy) = fast_floor_i32(v);
            let wx = [
                lanczos3_weight(-2.0 - fx),
                lanczos3_weight(-1.0 - fx),
                lanczos3_weight(-fx),
                lanczos3_weight(1.0 - fx),
                lanczos3_weight(2.0 - fx),
                lanczos3_weight(3.0 - fx),
            ];
            let wy = [
                lanczos3_weight(-2.0 - fy),
                lanczos3_weight(-1.0 - fy),
                lanczos3_weight(-fy),
                lanczos3_weight(1.0 - fy),
                lanczos3_weight(2.0 - fy),
                lanczos3_weight(3.0 - fy),
            ];
            if src.no_nodata
                && let Some((band, base, stride)) = src.rect_in_block(x0 - 2, y0 - 2, 6, 6)
                && let Some(sl) = band.data.as_f32_slice()
            {
                let mut col = [0.0f32; 6];
                for j in 0..6 {
                    let off = base + j * stride;
                    let row = &sl[off..off + 6];
                    let wyj = wy[j];
                    for i in 0..6 {
                        col[i] += row[i] * wyj;
                    }
                }
                let mut sum = 0.0f32;
                let mut wxs = 0.0f32;
                for i in 0..6 {
                    sum += col[i] * wx[i];
                    wxs += wx[i];
                }
                let wys: f32 = wy[0] + wy[1] + wy[2] + wy[3] + wy[4] + wy[5];
                return sum / (wxs * wys);
            }
            let mut sum = 0.0f32;
            let mut wsum = 0.0f32;
            for j in 0..6 {
                let ry = y0 + (j as i32) - 2;
                let wyj = wy[j];
                for i in 0..6 {
                    let s = src.sample_checked(x0 + (i as i32) - 2, ry);
                    if !s.is_nan() {
                        let w = wx[i] * wyj;
                        sum += s * w;
                        wsum += w;
                    }
                }
            }
            if wsum >= 0.5 {
                sum / wsum
            } else {
                sample_bilinear::<false>(src, u, v)
            }
        }
        ResampleMode::Gauss => {
            let cx = u.round() as i32;
            let cy = v.round() as i32;
            let mut sum = 0.0f32;
            let mut wsum = 0.0f32;
            let mut wtot = 0.0f32;
            for dy in -1..=1 {
                let ry = cy + dy;
                let ydiff = ry as f32 - v;
                for dx in -1..=1 {
                    let rx = cx + dx;
                    let xdiff = rx as f32 - u;
                    let w = (-0.888_888_9 * (xdiff * xdiff + ydiff * ydiff)).exp();
                    wtot += w;
                    let s = src.sample_checked(rx, ry);
                    if !s.is_nan() {
                        sum += s * w;
                        wsum += w;
                    }
                }
            }
            if wsum >= 0.5 * wtot && wsum > 1e-6 { sum / wsum } else { f32::NAN }
        }
        ResampleMode::Average
        | ResampleMode::Mode
        | ResampleMode::Rms
        | ResampleMode::Min
        | ResampleMode::Max
        | ResampleMode::Med
        | ResampleMode::Q1
        | ResampleMode::Q3
        | ResampleMode::Sum => {
            sample_footprint_exact(src, u, v, scale_x, scale_y, mode)
        }
    }
}

/// Area-exact footprint sampler for `Average`, `Mode`, `Rms`, `Min`, `Max`, `Med`, `Q1`, `Q3`, `Sum`.
///
/// Maps destination pixel center `(u, v)` and footprint width/height `(scale_x, scale_y)` to
/// source corner box `[x0, x1] x [y0, y1]` (where source pixel `(ix, iy)` spans `[ix, ix+1] x
/// [iy, iy+1]`) and weights every overlapped valid source pixel by its exact overlap area.
#[inline]
fn sample_footprint_exact(
    src: &BandSource<'_>,
    u: f32,
    v: f32,
    scale_x: f32,
    scale_y: f32,
    mode: ResampleMode,
) -> f32 {
    let sx = (scale_x as f64).abs().max(1e-6);
    let sy = (scale_y as f64).abs().max(1e-6);
    let cx = u as f64 + 0.5;
    let cy = v as f64 + 0.5;
    let x0 = (cx - 0.5 * sx).clamp(0.0, src.ifd_w as f64);
    let x1 = (cx + 0.5 * sx).clamp(0.0, src.ifd_w as f64);
    let y0 = (cy - 0.5 * sy).clamp(0.0, src.ifd_h as f64);
    let y1 = (cy + 0.5 * sy).clamp(0.0, src.ifd_h as f64);
    if x1 <= x0 + 1e-12 || y1 <= y0 + 1e-12 {
        return f32::NAN;
    }

    let ix0 = x0.floor() as i32;
    let ix1 = x1.next_down().floor() as i32;
    let iy0 = y0.floor() as i32;
    let iy1 = y1.next_down().floor() as i32;

    // Fast 1-pixel path: exact 1:1 overview read or upsampling strictly inside one source pixel.
    if ix0 == ix1 && iy0 == iy1 {
        let s = src.sample_checked(ix0, iy0);
        return if matches!(mode, ResampleMode::Sum) && !s.is_nan() {
            ((s as f64) * (x1 - x0) * (y1 - y0)) as f32
        } else {
            s
        };
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn visit_footprint<F: FnMut(f32, f64)>(
        src: &BandSource<'_>,
        x0: f64,
        x1: f64,
        y0: f64,
        y1: f64,
        ix0: i32,
        ix1: i32,
        iy0: i32,
        iy1: i32,
        mut cb: F,
    ) {
        const MAX_EXACT_SPAN: i32 = 64;
        let nx = ix1 - ix0 + 1;
        let ny = iy1 - iy0 + 1;
        if nx <= MAX_EXACT_SPAN && ny <= MAX_EXACT_SPAN {
            for iy in iy0..=iy1 {
                let wy = ((iy + 1) as f64).min(y1) - (iy as f64).max(y0);
                if wy <= 1e-9 {
                    continue;
                }
                for ix in ix0..=ix1 {
                    let wx = ((ix + 1) as f64).min(x1) - (ix as f64).max(x0);
                    if wx <= 1e-9 {
                        continue;
                    }
                    let s = src.sample_checked(ix, iy);
                    if !s.is_nan() {
                        cb(s, wx * wy);
                    }
                }
            }
        } else {
            let steps_x = nx.min(MAX_EXACT_SPAN) as usize;
            let steps_y = ny.min(MAX_EXACT_SPAN) as usize;
            let dx = (x1 - x0) / (steps_x as f64);
            let dy = (y1 - y0) / (steps_y as f64);
            let cell_w = dx * dy;
            for gy in 0..steps_y {
                let iy = (y0 + (gy as f64 + 0.5) * dy).floor() as i32;
                for gx in 0..steps_x {
                    let ix = (x0 + (gx as f64 + 0.5) * dx).floor() as i32;
                    let s = src.sample_checked(ix, iy);
                    if !s.is_nan() {
                        cb(s, cell_w);
                    }
                }
            }
        }
    }

    match mode {
        ResampleMode::Average => {
            let mut sum = 0.0f64;
            let mut wsum = 0.0f64;
            visit_footprint(src, x0, x1, y0, y1, ix0, ix1, iy0, iy1, |s, w| {
                sum += (s as f64) * w;
                wsum += w;
            });
            if wsum > 0.0 {
                (sum / wsum) as f32
            } else {
                f32::NAN
            }
        }
        ResampleMode::Sum => {
            let mut sum = 0.0f64;
            let mut any = false;
            visit_footprint(src, x0, x1, y0, y1, ix0, ix1, iy0, iy1, |s, w| {
                sum += (s as f64) * w;
                any = true;
            });
            if any { sum as f32 } else { f32::NAN }
        }
        ResampleMode::Rms => {
            let mut sq = 0.0f64;
            let mut wsum = 0.0f64;
            visit_footprint(src, x0, x1, y0, y1, ix0, ix1, iy0, iy1, |s, w| {
                let sf = s as f64;
                sq += sf * sf * w;
                wsum += w;
            });
            if wsum > 0.0 {
                (sq / wsum).sqrt() as f32
            } else {
                f32::NAN
            }
        }
        ResampleMode::Min => {
            let mut best = f32::INFINITY;
            let mut any = false;
            visit_footprint(src, x0, x1, y0, y1, ix0, ix1, iy0, iy1, |s, _| {
                if s < best {
                    best = s;
                }
                any = true;
            });
            if any { best } else { f32::NAN }
        }
        ResampleMode::Max => {
            let mut best = f32::NEG_INFINITY;
            let mut any = false;
            visit_footprint(src, x0, x1, y0, y1, ix0, ix1, iy0, iy1, |s, _| {
                if s > best {
                    best = s;
                }
                any = true;
            });
            if any { best } else { f32::NAN }
        }
        ResampleMode::Mode => {
            // Stack-allocated fast path for <= 64 distinct values (covers up to 8x8 exact blocks
            // with zero heap allocation); falls back to Vec only for larger footprints.
            let mut stack_bins = [(0u32, 0.0f32, 0.0f64); 64];
            let mut stack_len = 0usize;
            let mut heap_bins: Option<Vec<(u32, f32, f64)>> = None;
            visit_footprint(src, x0, x1, y0, y1, ix0, ix1, iy0, iy1, |s, w| {
                // Normalize -0.0 to +0.0 so they bin together.
                let norm = if s == 0.0 { 0.0f32 } else { s };
                let bits = norm.to_bits();
                if let Some(ref mut bins) = heap_bins {
                    for b in bins.iter_mut() {
                        if b.0 == bits {
                            b.2 += w;
                            return;
                        }
                    }
                    bins.push((bits, norm, w));
                } else {
                    for b in &mut stack_bins[..stack_len] {
                        if b.0 == bits {
                            b.2 += w;
                            return;
                        }
                    }
                    if stack_len < stack_bins.len() {
                        stack_bins[stack_len] = (bits, norm, w);
                        stack_len += 1;
                    } else {
                        let mut v = Vec::with_capacity(128);
                        v.extend_from_slice(&stack_bins);
                        v.push((bits, norm, w));
                        heap_bins = Some(v);
                    }
                }
            });
            let bins: &[(u32, f32, f64)] = match heap_bins {
                Some(ref v) => v.as_slice(),
                None => &stack_bins[..stack_len],
            };
            if bins.is_empty() {
                return f32::NAN;
            }
            let mut best_val = bins[0].1;
            let mut best_w = bins[0].2;
            for &(_, val, wt) in &bins[1..] {
                if wt > best_w {
                    best_w = wt;
                    best_val = val;
                }
            }
            best_val
        }
        ResampleMode::Med | ResampleMode::Q1 | ResampleMode::Q3 => {
            let mut stack_vals = [0.0f32; 64];
            let mut stack_len = 0usize;
            let mut heap_vals: Option<Vec<f32>> = None;
            visit_footprint(src, x0, x1, y0, y1, ix0, ix1, iy0, iy1, |s, _| {
                if let Some(ref mut v) = heap_vals {
                    v.push(s);
                } else if stack_len < stack_vals.len() {
                    stack_vals[stack_len] = s;
                    stack_len += 1;
                } else {
                    let mut v = Vec::with_capacity(128);
                    v.extend_from_slice(&stack_vals);
                    v.push(s);
                    heap_vals = Some(v);
                }
            });
            let slice: &mut [f32] = match heap_vals {
                Some(ref mut v) => v.as_mut_slice(),
                None => &mut stack_vals[..stack_len],
            };
            if slice.is_empty() {
                return f32::NAN;
            }
            slice.sort_unstable_by(f32::total_cmp);
            let q = match mode {
                ResampleMode::Q1 => 0.25f32,
                ResampleMode::Med => 0.50f32,
                ResampleMode::Q3 => 0.75f32,
                _ => 0.50f32,
            };
            let pos = q * ((slice.len() - 1) as f32);
            let lo = pos.floor() as usize;
            let hi = (lo + 1).min(slice.len() - 1);
            let frac = pos - lo as f32;
            if frac == 0.0 || lo == hi || slice[lo] == slice[hi] {
                slice[lo]
            } else {
                slice[lo] * (1.0 - frac) + slice[hi] * frac
            }
        }
        _ => unreachable!(),
    }
}

/// Estimate source-pixel footprint `(scale_x, scale_y)` per output pixel from a `CoordPlan`.
#[inline]
pub fn coord_plan_scale<const N: usize>(plan: &CoordPlan<N>) -> (f32, f32) {
    match plan {
        CoordPlan::Separable1D { u, v, .. } => separable_coord_scale(&u[..], &v[..]),
        CoordPlan::Grid2D { grid_u, grid_v, .. } => grid_coord_scale(grid_u, grid_v, N, N),
        CoordPlan::Exact2D { u, v, .. } => exact_coord_scale(&u[..], &v[..], N, N),
    }
}

#[inline(always)]
fn usable_scale(s: f32) -> f32 {
    if s.is_finite() && s > 1e-4 { s } else { 1.0 }
}

/// [`coord_plan_scale`] of separable axes `u` (columns) x `v` (rows).
#[inline]
pub fn separable_coord_scale(u: &[f32], v: &[f32]) -> (f32, f32) {
    let span = |c: &[f32]| match (c.first(), c.last()) {
        (Some(a), Some(b)) => (b - a).abs() / (c.len().saturating_sub(1)).max(1) as f32,
        _ => 1.0,
    };
    (usable_scale(span(u)), usable_scale(span(v)))
}

/// [`coord_plan_scale`] of a `GRID_DIM` x `GRID_DIM` warp grid over a `dst_w` x `dst_h` output.
#[inline]
pub fn grid_coord_scale(grid_u: &[f32], grid_v: &[f32], dst_w: usize, dst_h: usize) -> (f32, f32) {
    let mid = (GRID_DIM - 1) / 2;
    let i00 = mid * GRID_DIM + mid;
    let i10 = i00 + 1;
    let i01 = (mid + 1) * GRID_DIM + mid;
    let cell_x = (dst_w / (GRID_DIM - 1)).max(1) as f32;
    let cell_y = (dst_h / (GRID_DIM - 1)).max(1) as f32;
    let sx = (grid_u[i10] - grid_u[i00]).hypot(grid_v[i10] - grid_v[i00]) / cell_x;
    let sy = (grid_u[i01] - grid_u[i00]).hypot(grid_v[i01] - grid_v[i00]) / cell_y;
    (usable_scale(sx), usable_scale(sy))
}

/// [`coord_plan_scale`] of dense row-major `w` x `h` coordinate maps: the first pixel step found
/// with finite, non-degenerate neighbours (centre first, then a coarse and a full scan).
pub fn exact_coord_scale(u: &[f32], v: &[f32], w: usize, h: usize) -> (f32, f32) {
    if w <= 1 || h <= 1 || u.len() < w * h || v.len() < w * h {
        return (1.0, 1.0);
    }
    let probe = |py: usize, px: usize| -> Option<(f32, f32)> {
        let i00 = py * w + px;
        let i10 = i00 + 1;
        let i01 = (py + 1) * w + px;
        let sx = (u[i10] - u[i00]).hypot(v[i10] - v[i00]);
        let sy = (u[i01] - u[i00]).hypot(v[i01] - v[i00]);
        if sx.is_finite() && sx > 1e-4 && sy.is_finite() && sy > 1e-4 {
            Some((sx, sy))
        } else {
            None
        }
    };
    if let Some(s) = probe((h / 2).min(h - 2), (w / 2).min(w - 2)) {
        return s;
    }
    let (step_y, step_x) = ((h / 16).max(1), (w / 16).max(1));
    for py in (0..h - 1).step_by(step_y) {
        for px in (0..w - 1).step_by(step_x) {
            if let Some(s) = probe(py, px) {
                return s;
            }
        }
    }
    for py in 0..h - 1 {
        for px in 0..w - 1 {
            if let Some(s) = probe(py, px) {
                return s;
            }
        }
    }
    (1.0, 1.0)
}

#[inline(always)]
fn sample_val<const BILINEAR: bool, const CLEAN: bool>(
    src: &BandSource<'_>,
    u: f32,
    v: f32,
) -> f32 {
    if BILINEAR {
        sample_bilinear::<CLEAN>(src, u, v)
    } else {
        sample_nearest::<CLEAN>(src, u, v)
    }
}

#[inline(always)]
fn rescale_u8(val: f32, inv_range: f32, min_val: f32) -> u8 {
    ((val - min_val) * inv_range).clamp(0.0, 255.0) as u8
}

#[inline(always)]
fn for_each_pixel_in_row<const N: usize, F: FnMut(usize, f32, f32)>(
    plan: &CoordPlan<N>,
    py: usize,
    mut f: F,
) {
    match plan {
        CoordPlan::Separable1D { u, v, .. } => {
            let vy = v[py];
            for px in 0..N {
                f(px, u[px], vy);
            }
        }
        CoordPlan::Grid2D { grid_u, grid_v, .. } => {
            let cell_px = N / (GRID_DIM - 1);
            let inv_cell = 1.0 / (cell_px as f32);
            let gy = (py / cell_px).min(GRID_DIM - 2);
            let ty = (py - gy * cell_px) as f32 * inv_cell;
            let row0 = gy * GRID_DIM;
            let row1 = (gy + 1) * GRID_DIM;

            for gx in 0..(GRID_DIM - 1) {
                let u_l = grid_u[row0 + gx] + (grid_u[row1 + gx] - grid_u[row0 + gx]) * ty;
                let u_r = grid_u[row0 + gx + 1] + (grid_u[row1 + gx + 1] - grid_u[row0 + gx + 1]) * ty;
                let v_l = grid_v[row0 + gx] + (grid_v[row1 + gx] - grid_v[row0 + gx]) * ty;
                let v_r = grid_v[row0 + gx + 1] + (grid_v[row1 + gx + 1] - grid_v[row0 + gx + 1]) * ty;

                let du = u_r - u_l;
                let dv = v_r - v_l;
                let base_x = gx * cell_px;
                for k in 0..cell_px {
                    let tx = (k as f32) * inv_cell;
                    f(base_x + k, u_l + du * tx, v_l + dv * tx);
                }
            }
        }
        CoordPlan::Exact2D { u, v, .. } => {
            let row_off = py * N;
            for px in 0..N {
                f(px, u[row_off + px], v[row_off + px]);
            }
        }
    }
}

/// Full-precision `f64` point/resampling sampler for wide dtypes (`Uint32`, `Int32`, `Uint64`,
/// `Int64`, `Float64`) on data-array routes (`.npy` / unstyled data `.tif`).
#[inline]
pub fn sample_f64_general(
    src: &BandSource<'_>,
    u: f64,
    v: f64,
    mode: ResampleMode,
) -> f64 {
    if !src.contains_sample_f64(u, v) {
        return f64::NAN;
    }
    match mode {
        ResampleMode::Nearest => {
            let ix = (u + 0.5).floor() as i32;
            let iy = (v + 0.5).floor() as i32;
            src.sample_f64_checked(ix, iy)
        }
        ResampleMode::Bilinear => {
            let x0 = u.floor() as i32;
            let y0 = v.floor() as i32;
            let fx = u - x0 as f64;
            let fy = v - y0 as f64;
            let s00 = src.sample_f64_checked(x0, y0);
            let s10 = src.sample_f64_checked(x0 + 1, y0);
            let s01 = src.sample_f64_checked(x0, y0 + 1);
            let s11 = src.sample_f64_checked(x0 + 1, y0 + 1);
            let w00 = (1.0 - fx) * (1.0 - fy);
            let w10 = fx * (1.0 - fy);
            let w01 = (1.0 - fx) * fy;
            let w11 = fx * fy;
            let mut sum = 0.0f64;
            let mut wsum = 0.0f64;
            for (s, w) in [(s00, w00), (s10, w10), (s01, w01), (s11, w11)] {
                if !s.is_nan() {
                    sum += s * w;
                    wsum += w;
                }
            }
            if wsum >= 0.5 { sum / wsum } else { f64::NAN }
        }
        _ => {
            // For higher-order / footprint modes on wide dtypes, fall back to nearest when exact
            // 1:1 or bilinear in f64 to avoid f32 quantization on large integers/doubles.
            let x0 = u.floor() as i32;
            let y0 = v.floor() as i32;
            let fx = u - x0 as f64;
            let fy = v - y0 as f64;
            if fx.abs() < 1e-6 && fy.abs() < 1e-6 {
                src.sample_f64_checked(x0, y0)
            } else {
                sample_f64_general(src, u, v, ResampleMode::Bilinear)
            }
        }
    }
}

pub fn empty_tile_bytes(tile_size: u32, format: OutputFormat, num_bands: usize) -> Bytes {
    use std::sync::OnceLock;
    match format {
        OutputFormat::Png => empty_transparent_png(tile_size),
        OutputFormat::Webp { .. } => empty_transparent_webp(tile_size),
        OutputFormat::Jpeg { .. } => empty_black_jpeg(tile_size),
        OutputFormat::RawF64 => {
            let count = (tile_size as usize) * (tile_size as usize) * num_bands;
            let mut buf = BytesMut::with_capacity(count * 8);
            let nan_bytes = f64::NAN.to_le_bytes();
            for _ in 0..count {
                buf.put_slice(&nan_bytes);
            }
            buf.freeze()
        }
        OutputFormat::RawF32 => {
            let build = |sz: u32, nb: usize| -> Bytes {
                let count = (sz as usize) * (sz as usize) * nb;
                let mut buf = BytesMut::with_capacity(count * 4);
                let nan_bytes = f32::NAN.to_le_bytes();
                for _ in 0..count {
                    buf.put_slice(&nan_bytes);
                }
                buf.freeze()
            };
            static F32_256_1: OnceLock<Bytes> = OnceLock::new();
            static F32_256_3: OnceLock<Bytes> = OnceLock::new();
            static F32_512_1: OnceLock<Bytes> = OnceLock::new();
            static F32_512_3: OnceLock<Bytes> = OnceLock::new();
            match (tile_size, num_bands) {
                (256, 1) => F32_256_1.get_or_init(|| build(256, 1)).clone(),
                (256, 3) => F32_256_3.get_or_init(|| build(256, 3)).clone(),
                (512, 1) => F32_512_1.get_or_init(|| build(512, 1)).clone(),
                (512, 3) => F32_512_3.get_or_init(|| build(512, 3)).clone(),
                _ => build(tile_size, num_bands),
            }
        }
        OutputFormat::Npy | OutputFormat::NpyF64 => {
            let hdr = encode_npy_f32_header(num_bands, tile_size, tile_size);
            let count = (tile_size as usize) * (tile_size as usize) * num_bands;
            let mut buf = BytesMut::with_capacity(hdr.len() + count * 4);
            buf.put_slice(&hdr);
            let nan_bytes = f32::NAN.to_le_bytes();
            for _ in 0..count {
                buf.put_slice(&nan_bytes);
            }
            buf.freeze()
        }
        OutputFormat::RawRgba => {
            static RGBA_256: OnceLock<Bytes> = OnceLock::new();
            static RGBA_512: OnceLock<Bytes> = OnceLock::new();
            match tile_size {
                256 => RGBA_256.get_or_init(|| Bytes::from(vec![0u8; 256 * 256 * 4])).clone(),
                512 => RGBA_512.get_or_init(|| Bytes::from(vec![0u8; 512 * 512 * 4])).clone(),
                _ => Bytes::from(vec![0u8; (tile_size as usize) * (tile_size as usize) * 4]),
            }
        }
    }
}

#[inline]
fn rgb_to_rgba_bytes(rgb: &[u8]) -> Bytes {
    let pixels = rgb.len() / 3;
    let mut out = Vec::with_capacity(pixels * 4);
    for chunk in rgb.chunks_exact(3) {
        out.extend_from_slice(&[chunk[0], chunk[1], chunk[2], 255]);
    }
    Bytes::from(out)
}

/// Encode an RGBA8 tile buffer into PNG (using opaque RGB when `all_opaque`), WebP, JPEG, or raw RGBA.
pub fn encode_rgba_tile(rgba: &[u8], size: u32, all_opaque: bool, format: OutputFormat) -> Bytes {
    let n = size as usize;
    match format {
        OutputFormat::Png => {
            let encode_with = |filtered: &mut Vec<u8>, rgb_row: &mut Vec<u8>| -> Bytes {
                if all_opaque {
                    let row_len = n * 3 + 1;
                    let total = n * row_len;
                    if filtered.len() < total {
                        filtered.resize(total, 0);
                    }
                    if rgb_row.len() < n * 3 {
                        rgb_row.resize(n * 3, 0);
                    }
                    let row_buf = &mut rgb_row[..n * 3];
                    for y in 0..n {
                        let src = &rgba[y * n * 4..(y + 1) * n * 4];
                        for x in 0..n {
                            row_buf[x * 3] = src[x * 4];
                            row_buf[x * 3 + 1] = src[x * 4 + 1];
                            row_buf[x * 3 + 2] = src[x * 4 + 2];
                        }
                        filter_sub_rgb_row(
                            row_buf,
                            &mut filtered[y * row_len..(y + 1) * row_len],
                        );
                    }
                    encode_filtered_png(size, size, 2, &filtered[..total])
                } else {
                    let row_len = n * 4 + 1;
                    let total = n * row_len;
                    if filtered.len() < total {
                        filtered.resize(total, 0);
                    }
                    for y in 0..n {
                        let src = &rgba[y * n * 4..(y + 1) * n * 4];
                        filter_sub_rgba_row(src, &mut filtered[y * row_len..(y + 1) * row_len]);
                    }
                    encode_filtered_png(size, size, 6, &filtered[..total])
                }
            };
            SCRATCH.with(|cell| {
                if let Ok(mut guard) = cell.try_borrow_mut() {
                    let sc = &mut *guard;
                    encode_with(&mut sc.img_buf, &mut sc.row_buf)
                } else {
                    let mut filtered = Vec::new();
                    let mut rgb_row = Vec::new();
                    encode_with(&mut filtered, &mut rgb_row)
                }
            })
        }
        OutputFormat::Webp { quality } => encode_webp_rgba(size, size, rgba, quality),
        OutputFormat::Jpeg { quality } => encode_jpeg_rgba(size, size, rgba, quality),
        OutputFormat::RawRgba | OutputFormat::RawF32 | OutputFormat::RawF64 | OutputFormat::Npy | OutputFormat::NpyF64 => {
            Bytes::copy_from_slice(rgba)
        }
    }
}

/// [`encode_rgba_tile`] for a tile of one band drawn in grey (every pixel that is there has
/// `r == g == b`), written as TiTiler writes one band: a PNG of one channel (`L`) when every
/// pixel is there and of two (`LA`) when not, and a JPEG of one component. Other formats are
/// written as [`encode_rgba_tile`] writes them.
pub fn encode_grey_rgba_tile(rgba: &[u8], size: u32, all_opaque: bool, format: OutputFormat) -> Bytes {
    let n = size as usize;
    let luma = |p: &[u8]| if p[3] == 0 { 0 } else { p[0] };
    match format {
        OutputFormat::Png => {
            let channels = if all_opaque { 1 } else { 2 };
            let row_len = n * channels + 1;
            let mut filtered = vec![0u8; n * row_len];
            let mut row = vec![0u8; n * channels];
            for y in 0..n {
                let src = &rgba[y * n * 4..(y + 1) * n * 4];
                let dst = &mut filtered[y * row_len..(y + 1) * row_len];
                if all_opaque {
                    for (p, v) in src.chunks_exact(4).zip(row.iter_mut()) {
                        *v = p[0];
                    }
                    filter_sub_u8_row(&row, dst);
                } else {
                    for (p, v) in src.chunks_exact(4).zip(row.chunks_exact_mut(2)) {
                        v[0] = luma(p);
                        v[1] = p[3];
                    }
                    filter_sub_ga_row(&row, dst);
                }
            }
            encode_filtered_png(size, size, if all_opaque { 0 } else { 4 }, &filtered)
        }
        OutputFormat::Jpeg { quality } => {
            let grey: Vec<u8> = rgba.chunks_exact(4).take(n * n).map(luma).collect();
            crate::jpeg_fast::encode_jpeg_gray(size, size, &grey, quality)
        }
        _ => encode_rgba_tile(rgba, size, all_opaque, format),
    }
}

#[inline(always)]
fn fill_uv_row<const N: usize>(
    plan: &CoordPlan<N>,
    py: usize,
    u_row: &mut [f32; N],
    v_row: &mut [f32; N],
) {
    match plan {
        CoordPlan::Separable1D { u, v, .. } => {
            u_row.copy_from_slice(&u[..]);
            v_row.fill(v[py]);
        }
        CoordPlan::Grid2D { grid_u, grid_v, .. } => {
            let cell_px = N / (GRID_DIM - 1);
            let inv_cell = 1.0 / (cell_px as f32);
            let gy = (py / cell_px).min(GRID_DIM - 2);
            let ty = (py - gy * cell_px) as f32 * inv_cell;
            let row0 = gy * GRID_DIM;
            let row1 = (gy + 1) * GRID_DIM;
            for gx in 0..(GRID_DIM - 1) {
                let u_l = grid_u[row0 + gx] + (grid_u[row1 + gx] - grid_u[row0 + gx]) * ty;
                let u_r = grid_u[row0 + gx + 1] + (grid_u[row1 + gx + 1] - grid_u[row0 + gx + 1]) * ty;
                let v_l = grid_v[row0 + gx] + (grid_v[row1 + gx] - grid_v[row0 + gx]) * ty;
                let v_r = grid_v[row0 + gx + 1] + (grid_v[row1 + gx + 1] - grid_v[row0 + gx + 1]) * ty;
                let du = u_r - u_l;
                let dv = v_r - v_l;
                let base_x = gx * cell_px;
                for k in 0..cell_px {
                    let tx = (k as f32) * inv_cell;
                    u_row[base_x + k] = u_l + du * tx;
                    v_row[base_x + k] = v_l + dv * tx;
                }
            }
        }
        CoordPlan::Exact2D { u, v, .. } => {
            let row_off = py * N;
            u_row.copy_from_slice(&u[row_off..row_off + N]);
            v_row.copy_from_slice(&v[row_off..row_off + N]);
        }
    }
}

#[cfg(test)]
mod tests;
