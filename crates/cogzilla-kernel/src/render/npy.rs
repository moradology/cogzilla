//! NumPy v1.0 `.npy` array serialization and retyping.

use bytes::{BufMut, Bytes, BytesMut};
use cogzilla_raster::SampleType;

/// Encode a 64-byte-aligned NumPy v1.0 `.npy` header for a C-contiguous array of the
/// dtype `descr` (`<f4`, `|u1`, ...) and of `shape`: the magic, the version, the length of
/// what follows, then the dict padded with spaces and ended by a newline so that the data
/// starts on a multiple of 64 bytes. Every `.npy` this crate writes begins with it.
pub fn encode_npy_header(descr: &str, shape: &[usize]) -> Vec<u8> {
    use std::fmt::Write;
    let mut dict = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': (");
    for (i, dim) in shape.iter().enumerate() {
        if i > 0 {
            dict.push_str(", ");
        }
        let _ = write!(dict, "{dim}");
    }
    // A tuple of one is written `(n,)`.
    dict.push_str(if shape.len() == 1 { ",), }" } else { "), }" });
    let unpadded = 10 + dict.len() + 1;
    let pad = (64 - (unpadded % 64)) % 64;
    for _ in 0..pad {
        dict.push(' ');
    }
    dict.push('\n');
    let hlen = dict.len() as u16;
    let mut hdr = Vec::with_capacity(10 + dict.len());
    hdr.extend_from_slice(b"\x93NUMPY\x01\x00");
    hdr.extend_from_slice(&hlen.to_le_bytes());
    hdr.extend_from_slice(dict.as_bytes());
    hdr
}

/// Encode a 64-byte-aligned NumPy v1.0 `.npy` header for a C-contiguous `<f4` array
/// of shape `(bands, height, width)`.
pub fn encode_npy_f32_header(bands: usize, height: u32, width: u32) -> Vec<u8> {
    encode_npy_header("<f4", &[bands, height as usize, width as usize])
}

/// The NumPy dtype string of a sample type: little-endian, and `|` for the single bytes.
#[inline(always)]
pub const fn npy_descr(kind: SampleType) -> &'static str {
    kind.npy_descr()
}

fn write_npy_typed_planes(
    out: &mut Vec<u8>,
    kind: SampleType,
    data: impl Iterator<Item = f64>,
    mask_bits: Option<impl Iterator<Item = bool>>,
) {
    macro_rules! write_planes {
        ($t:ty, $cast:expr) => {{
            let cast = $cast;
            for v in data {
                let x: $t = cast(v);
                out.extend_from_slice(&x.to_le_bytes());
            }
            if let Some(bits) = mask_bits {
                for there in bits {
                    let x: $t = if there { <$t>::MAX } else { <$t>::MIN };
                    out.extend_from_slice(&x.to_le_bytes());
                }
            }
        }};
    }
    match kind {
        SampleType::Bool => {
            for v in data {
                out.push(u8::from(v != 0.0 && !v.is_nan()));
            }
            if let Some(bits) = mask_bits {
                for there in bits {
                    out.push(u8::from(there));
                }
            }
        }
        SampleType::Uint1 | SampleType::Uint4 | SampleType::Uint8 => {
            write_planes!(u8, |v: f64| v as u8)
        }
        SampleType::Int8 => write_planes!(i8, |v: f64| v as i8),
        SampleType::Uint12 | SampleType::Uint16 => write_planes!(u16, |v: f64| v as u16),
        SampleType::Int16 => write_planes!(i16, |v: f64| v as i16),
        SampleType::Uint32 => write_planes!(u32, |v: f64| v as u32),
        SampleType::Int32 => write_planes!(i32, |v: f64| v as i32),
        SampleType::Uint64 => write_planes!(u64, |v: f64| v as u64),
        SampleType::Int64 => write_planes!(i64, |v: f64| v as i64),
        SampleType::Float16 => {
            for v in data {
                out.extend_from_slice(&f32_to_f16_bits(v as f32).to_le_bytes());
            }
            if let Some(bits) = mask_bits {
                for there in bits {
                    // Max finite f16 = 65504.0 (0x7BFF), min finite f16 = -65504.0 (0xFBFF).
                    let bits: u16 = if there { 0x7BFF } else { 0xFBFF };
                    out.extend_from_slice(&bits.to_le_bytes());
                }
            }
        }
        SampleType::Float32 => write_planes!(f32, |v: f64| (v as f32).clamp(f32::MIN, f32::MAX)),
        SampleType::Float64 => write_planes!(f64, |v: f64| v.clamp(f64::MIN, f64::MAX)),
    }
}

/// Convert `f32` to IEEE-754 binary16 bits (preserving `NaN` as canonical quiet NaN `0x7E00`
/// and clamping infinities to `±65504.0`).
fn f32_to_f16_bits(v: f32) -> u16 {
    if v.is_nan() {
        return 0x7E00;
    }
    let clamped = v.clamp(-65504.0, 65504.0);
    let bits = clamped.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let frac = bits & 0x007F_FFFF;
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        let m = (frac | 0x0080_0000) >> (1 - exp + 13);
        return sign | (m as u16);
    }
    if exp >= 31 {
        return sign | 0x7BFF;
    }
    sign | ((exp as u16) << 10) | ((frac >> 13) as u16)
}

/// A `.npy` array of `bands` planes of `height` by `width`, written in the dtype of `kind`
/// from CHW float32 `values` in which NaN marks a value that is not there.
///
/// A value that is not there is written as `absent`, or as zero without one. Integers take
/// the value truncated toward zero and held to the type's range (a NaN `absent` is zero);
/// the float types hold an infinity to their largest finite number. With `mask`, one more
/// plane follows the bands, as rio-tiler's `ImageData.mask` is: the type's largest number
/// (`255` for `u1`, `65535` for `u2`, `f32::MAX` for `f4`, ...) where any band has a value,
/// and its smallest elsewhere (`0` for the unsigned types, `-32768` for `i2`, `f32::MIN`
/// for `f4`, ...). Only the first
/// `bands * height * width` of `values` are written, and a shorter slice makes a shorter
/// body under a header that still names every plane.
///
/// This is not the float32 path of [`wrap_raw_f32_as_npy_with_mask`], which copies its
/// planes as they are, counts an infinity as absent and writes `255.0` in its mask.
pub fn encode_npy_typed(
    values: &[f32],
    bands: usize,
    height: usize,
    width: usize,
    kind: SampleType,
    absent: Option<f32>,
    mask: bool,
) -> Bytes {
    let pixels = width * height;
    let planes = bands + usize::from(mask);
    let hdr = encode_npy_header(npy_descr(kind), &[planes, height, width]);
    let mut out = Vec::with_capacity(hdr.len() + planes * pixels * kind.bytes_per_sample());
    out.extend_from_slice(&hdr);
    let absent = f64::from(absent.unwrap_or(0.0));
    let data = &values[..(bands * pixels).min(values.len())];
    let vals = data
        .iter()
        .map(|v| if v.is_nan() { absent } else { f64::from(*v) });
    let mask_bits = mask.then(|| {
        (0..pixels).map(|i| {
            (0..bands).any(|band| data.get(band * pixels + i).is_some_and(|v| !v.is_nan()))
        })
    });
    write_npy_typed_planes(&mut out, kind, vals, mask_bits);
    Bytes::from(out)
}

/// The shape of a float32 `.npy` this crate wrote (`<f4`, C order), and where its data starts.
pub fn f32_npy_shape(npy: &[u8]) -> Option<(Vec<usize>, usize)> {
    if npy.len() < 10 || &npy[..8] != b"\x93NUMPY\x01\x00" {
        return None;
    }
    let start = 10 + usize::from(u16::from_le_bytes([npy[8], npy[9]]));
    let dict = std::str::from_utf8(npy.get(10..start)?).ok()?;
    if !dict.contains("'descr': '<f4'") || !dict.contains("'fortran_order': False") {
        return None;
    }
    let from = dict.find("'shape': (")? + "'shape': (".len();
    let shape = dict[from..from + dict[from..].find(')')?]
        .split(',')
        .map(str::trim)
        .filter(|dim| !dim.is_empty())
        .map(|dim| dim.parse().ok())
        .collect::<Option<Vec<usize>>>()?;
    Some((shape, start))
}

/// A float32 `.npy` of CHW planes rewritten in the dtype of `kind`, or `None` when `npy` is
/// not such an array or its body is not the size its header names.
///
/// The values are cast as [`encode_npy_typed`] casts them: integers truncated toward zero
/// and held to the type's range (NaN as zero), floats with an infinity held to their largest
/// finite number. With `mask` the last plane is a validity mask (any value but zero where a
/// pixel is there), written as the type's largest number there and its smallest elsewhere,
/// as rio-tiler's `ImageData.mask` is. The values under the mask are kept as they are: what
/// the renderer put there is the raster's own fill.
pub fn retype_f32_npy(npy: &[u8], kind: SampleType, mask: bool) -> Option<Bytes> {
    retype_f32_npy_with(npy, kind, mask, None, |v| v)
}

/// Whether a floating-point validity mask sample represents a valid pixel.
///
/// Works uniformly across both mask conventions used in cogzilla:
/// - `.npy` masks (`255.0` valid / `0.0` masked), and
/// - retyped float GeoTIFF / `.npy` masks (`MAX` valid / `MIN` masked).
///
/// Checking `m.is_finite() && m > 0.0` ensures negative mask sentinels (such as `f32::MIN`)
/// are treated as masked rather than valid (#574).
#[inline(always)]
pub fn is_valid_mask_sample(m: f64) -> bool {
    m.is_finite() && m > 0.0
}

/// Normalize a sample `v` against its pixel's optional mask sample before retyping:
/// a `NaN` on a pixel the mask marks valid (e.g. an asymmetric per-band nodata pixel in a
/// multi-band raster) becomes `nodata_fill.unwrap_or(0.0)` so the raster's own nodata
/// sentinel (such as `-9999.0`) is preserved instead of silently turning into `0.0` (#636).
#[inline(always)]
pub fn retype_value_with_mask(v: f64, mask_sample: Option<f64>, nodata_fill: Option<f64>) -> f64 {
    if v.is_nan() && mask_sample.is_some_and(is_valid_mask_sample) {
        nodata_fill.unwrap_or(0.0)
    } else {
        v
    }
}

/// [`retype_f32_npy`], each value (not the mask) passed through `prepare` before it is cast:
/// how an algorithm rounds or wraps it into `kind`. `nodata_fill` is substituted for per-band
/// `NaN`s on pixels whose dataset mask is valid (#636).
pub fn retype_f32_npy_with(
    npy: &[u8],
    kind: SampleType,
    mask: bool,
    nodata_fill: Option<f64>,
    prepare: impl Fn(f64) -> f64,
) -> Option<Bytes> {
    let (shape, start) = f32_npy_shape(npy)?;
    let [planes, height, width] = shape[..] else { return None };
    let body = &npy[start..];
    let pixels = height.checked_mul(width)?;
    if planes == 0 || body.len() != planes.checked_mul(pixels)?.checked_mul(4)? {
        return None;
    }
    let bands = planes - usize::from(mask);
    let data_bytes = bands * pixels * 4;
    let hdr = encode_npy_header(npy_descr(kind), &[planes, height, width]);
    let mut out = Vec::with_capacity(hdr.len() + planes * pixels * kind.bytes_per_sample());
    out.extend_from_slice(&hdr);
    let mask_slice = mask.then(|| &body[data_bytes..]);
    let data = body[..data_bytes].chunks_exact(4).enumerate().map(|(i, v)| {
        let x = f64::from(f32::from_le_bytes([v[0], v[1], v[2], v[3]]));
        let mask_sample = if x.is_nan() {
            mask_slice.map(|ms| {
                let off = (i % pixels) * 4;
                f64::from(f32::from_le_bytes([ms[off], ms[off + 1], ms[off + 2], ms[off + 3]]))
            })
        } else {
            None
        };
        prepare(retype_value_with_mask(x, mask_sample, nodata_fill))
    });
    if matches!(kind, SampleType::Uint8 | SampleType::Uint16)
        && let Some(ms) = mask_slice
    {
        write_npy_typed_planes(&mut out, kind, data, None::<std::iter::Empty<bool>>);
        if kind == SampleType::Uint8 {
            for v in ms.chunks_exact(4) {
                let m = f32::from_le_bytes([v[0], v[1], v[2], v[3]]);
                let b = if !m.is_finite() || m <= 0.0 {
                    0u8
                } else if m >= 255.0 {
                    255u8
                } else {
                    m.round() as u8
                };
                out.push(b);
            }
        } else {
            // Preserve continuous 0..=65535 alpha for uint16 RGBA / Gray+Alpha rasters.
            // f32 has 24-bit significand precision, so scaling 0..=65535 -> 0..=255 -> 0..=65535
            // round-trips every u16 value bit-exactly.
            for v in ms.chunks_exact(4) {
                let m = f32::from_le_bytes([v[0], v[1], v[2], v[3]]);
                let u = if !m.is_finite() || m <= 0.0 {
                    0u16
                } else if m >= 255.0 {
                    u16::MAX
                } else {
                    (f64::from(m) * (f64::from(u16::MAX) / 255.0)).round() as u16
                };
                out.extend_from_slice(&u.to_le_bytes());
            }
        }
        return Some(Bytes::from(out));
    }
    let mask_bits = mask_slice.map(|ms| {
        ms.chunks_exact(4).map(|v| {
            is_valid_mask_sample(f64::from(f32::from_le_bytes([v[0], v[1], v[2], v[3]])))
        })
    });
    write_npy_typed_planes(&mut out, kind, data, mask_bits);
    Some(Bytes::from(out))
}

/// Like [`encode_npy_typed`], taking full-precision CHW `f64` values (preserving exact
/// `uint32`/`int32`/`int64` up to ±2^53 and 17-digit `float64` without `f32` quantization).
pub fn encode_npy_typed_f64(
    values: &[f64],
    bands: usize,
    height: usize,
    width: usize,
    kind: SampleType,
    absent: Option<f64>,
    mask: bool,
) -> Bytes {
    let pixels = width * height;
    let planes = bands + usize::from(mask);
    let hdr = encode_npy_header(npy_descr(kind), &[planes, height, width]);
    let mut out = Vec::with_capacity(hdr.len() + planes * pixels * kind.bytes_per_sample());
    out.extend_from_slice(&hdr);
    let absent = absent.unwrap_or(if kind.storable_range().is_none() {
        f64::NAN
    } else {
        0.0
    });
    let data = &values[..(bands * pixels).min(values.len())];
    let vals = data.iter().map(|&v| if v.is_nan() { absent } else { v });
    let mask_bits = mask.then(|| {
        (0..pixels).map(|i| {
            (0..bands).any(|band| data.get(band * pixels + i).is_some_and(|v| !v.is_nan()))
        })
    });
    write_npy_typed_planes(&mut out, kind, vals, mask_bits);
    Bytes::from(out)
}

/// Prepend a NumPy v1.0 `.npy` `<f4` header to raw CHW little-endian float32 bytes.
pub fn wrap_raw_f32_as_npy(raw: Bytes, bands: usize, height: u32, width: u32) -> Bytes {
    wrap_raw_f32_as_npy_with_mask(raw, bands, height, width, false)
}

/// Prepend a NumPy v1.0 `.npy` `<f4` header to raw CHW little-endian float32 bytes,
/// optionally stacking a `(1, H, W)` float32 validity mask band (`255.0` valid / `0.0` nodata)
/// when `return_mask` is `true` (matching TiTiler / `rio-tiler` `.npy` output).
pub fn wrap_raw_f32_as_npy_with_mask(
    raw: Bytes,
    bands: usize,
    height: u32,
    width: u32,
    return_mask: bool,
) -> Bytes {
    wrap_raw_f32_as_npy_with_alpha_mask(raw, bands, height, width, None, return_mask)
}

/// Like [`wrap_raw_f32_as_npy_with_mask`], but when `alpha_mask` is `Some` and `return_mask`
/// is `true`, writes the continuous `0.0..=255.0` resampled alpha values for valid pixels
/// instead of binarizing to `255.0`.
pub fn wrap_raw_f32_as_npy_with_alpha_mask(
    raw: Bytes,
    bands: usize,
    height: u32,
    width: u32,
    alpha_mask: Option<&[f32]>,
    return_mask: bool,
) -> Bytes {
    if !return_mask {
        let hdr = encode_npy_f32_header(bands, height, width);
        let mut out = BytesMut::with_capacity(hdr.len() + raw.len());
        out.put_slice(&hdr);
        out.put_slice(&raw);
        return out.freeze();
    }
    let n_px = (height as usize).saturating_mul(width as usize);
    let plane_bytes = n_px * 4;
    let hdr = encode_npy_f32_header(bands + 1, height, width);
    let mut out = BytesMut::with_capacity(hdr.len() + raw.len() + plane_bytes);
    out.put_slice(&hdr);
    out.put_slice(&raw);
    let valid_le = 255.0f32.to_le_bytes();
    let zero_le = 0.0f32.to_le_bytes();
    for px in 0..n_px {
        let px_off = px * 4;
        let mut any_valid = false;
        for b in 0..bands {
            let off = b * plane_bytes + px_off;
            if off + 4 <= raw.len() {
                let v = f32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]);
                if v.is_finite() {
                    any_valid = true;
                    break;
                }
            }
        }
        if !any_valid {
            out.put_slice(&zero_le);
        } else if let Some(alpha) = alpha_mask {
            let a = alpha.get(px).copied().unwrap_or(0.0);
            let clamped = if a.is_finite() && a > 0.0 {
                a.clamp(0.0, 255.0)
            } else {
                0.0
            };
            out.put_slice(&clamped.to_le_bytes());
        } else {
            out.put_slice(&valid_le);
        }
    }
    out.freeze()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn npy_body(bytes: &[u8]) -> &[u8] {
        assert_eq!(&bytes[..8], b"\x93NUMPY\x01\x00");
        let hdr_len = usize::from(u16::from_le_bytes([bytes[8], bytes[9]]));
        &bytes[10 + hdr_len..]
    }

    #[test]
    fn float_dtypes_write_nan_for_absent_pixels_in_npy() {
        // Float16: absent pixel -> canonical binary16 quiet NaN (0x7E00); finite 1.0 -> 0x3C00.
        let f16_npy = encode_npy_typed_f64(&[1.0, f64::NAN], 1, 1, 2, SampleType::Float16, None, false);
        let body = npy_body(&f16_npy);
        assert_eq!(body.len(), 4);
        assert_eq!(u16::from_le_bytes([body[0], body[1]]), 0x3C00);
        let nan_bits = u16::from_le_bytes([body[2], body[3]]);
        assert_eq!(nan_bits & 0x7C00, 0x7C00, "Float16 absent pixel must have NaN exponent");
        assert_ne!(nan_bits & 0x03FF, 0, "Float16 absent pixel must have non-zero mantissa (NaN, not Inf)");

        // Float32: absent pixel -> NaN.
        let f32_npy = encode_npy_typed_f64(&[1.0, f64::NAN], 1, 1, 2, SampleType::Float32, None, false);
        let body = npy_body(&f32_npy);
        assert_eq!(f32::from_le_bytes(body[0..4].try_into().unwrap()), 1.0);
        assert!(f32::from_le_bytes(body[4..8].try_into().unwrap()).is_nan());

        // Float64: absent pixel -> NaN.
        let f64_npy = encode_npy_typed_f64(&[1.0, f64::NAN], 1, 1, 2, SampleType::Float64, None, false);
        let body = npy_body(&f64_npy);
        assert_eq!(f64::from_le_bytes(body[0..8].try_into().unwrap()), 1.0);
        assert!(f64::from_le_bytes(body[8..16].try_into().unwrap()).is_nan());

        // Integer/Bool dtypes: absent pixel -> 0.
        let u8_npy = encode_npy_typed_f64(&[7.0, f64::NAN], 1, 1, 2, SampleType::Uint8, None, false);
        assert_eq!(npy_body(&u8_npy), &[7u8, 0u8]);
        let bool_npy = encode_npy_typed_f64(&[1.0, f64::NAN], 1, 1, 2, SampleType::Bool, None, false);
        assert_eq!(npy_body(&bool_npy), &[1u8, 0u8]);
    }

    #[test]
    fn retype_f32_npy_replaces_nan_with_zero_on_valid_masked_pixels() {
        // 2-band, 1x2 CHW float32 array + mask plane:
        // pixel 0: band0=NaN (nodata), band1=100.0 (valid) -> mask=255.0 (valid)
        // pixel 1: band0=NaN, band1=NaN -> mask=0.0 (masked)
        let raw_vals: [f32; 4] = [f32::NAN, f32::NAN, 100.0, f32::NAN];
        let mut raw_bytes = Vec::with_capacity(16);
        for v in raw_vals {
            raw_bytes.extend_from_slice(&v.to_le_bytes());
        }
        let wrapped = wrap_raw_f32_as_npy_with_mask(Bytes::from(raw_bytes), 2, 1, 2, true);
        let retyped = retype_f32_npy(&wrapped, SampleType::Float32, true).expect("retype f32");
        let body = npy_body(&retyped);
        let out: Vec<f32> = body
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        // Band 0: pixel 0 (valid!) must be 0.0 (not NaN); pixel 1 (masked) stays NaN.
        assert_eq!(out[0], 0.0);
        assert!(out[1].is_nan());
        // Band 1: pixel 0 = 100.0, pixel 1 = NaN.
        assert_eq!(out[2], 100.0);
        assert!(out[3].is_nan());
        // Mask plane: pixel 0 = f32::MAX, pixel 1 = f32::MIN.
        assert_eq!(out[4], f32::MAX);
        assert_eq!(out[5], f32::MIN);
    }

    #[test]
    fn continuous_alpha_mask_is_preserved_in_f32_and_retyped_u8_npy() {
        // 1-band, 1x4 CHW float32 array with continuous alpha [0, 64, 128, 255].
        let raw_vals: [f32; 4] = [f32::NAN, 10.0, 20.0, 30.0];
        let mut raw_bytes = Vec::with_capacity(16);
        for v in raw_vals {
            raw_bytes.extend_from_slice(&v.to_le_bytes());
        }
        let alpha = [0.0f32, 64.0, 128.0, 255.0];
        let wrapped = wrap_raw_f32_as_npy_with_alpha_mask(
            Bytes::from(raw_bytes),
            1,
            1,
            4,
            Some(&alpha),
            true,
        );
        let f32_body = npy_body(&wrapped);
        let f32_out: Vec<f32> = f32_body
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(&f32_out[4..8], &[0.0, 64.0, 128.0, 255.0]);

        let u8_npy = retype_f32_npy(&wrapped, SampleType::Uint8, true).expect("retype u8");
        let u8_body = npy_body(&u8_npy);
        assert_eq!(&u8_body[0..4], &[0u8, 10, 20, 30]);
        assert_eq!(&u8_body[4..8], &[0u8, 64, 128, 255]);

        // Verify continuous uint16 alpha round-trips bit-exactly through 0..=255 f32 scaling.
        let sample_u16s: [u16; 6] = [0, 1, 5000, 32768, 65534, 65535];
        let mut raw_u16_bytes = Vec::with_capacity(sample_u16s.len() * 4);
        let mut alpha_u16_f32 = Vec::with_capacity(sample_u16s.len());
        for (i, &a) in sample_u16s.iter().enumerate() {
            let v = if a == 0 { f32::NAN } else { (1000 + i * 500) as f32 };
            raw_u16_bytes.extend_from_slice(&v.to_le_bytes());
            alpha_u16_f32.push(a as f32 * (255.0 / 65535.0));
        }
        let wrapped16 = wrap_raw_f32_as_npy_with_alpha_mask(
            Bytes::from(raw_u16_bytes),
            1,
            1,
            sample_u16s.len() as u32,
            Some(&alpha_u16_f32),
            true,
        );
        let u16_npy = retype_f32_npy(&wrapped16, SampleType::Uint16, true).expect("retype u16");
        let u16_body = npy_body(&u16_npy);
        let u16_vals: Vec<u16> = u16_body
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        assert_eq!(&u16_vals[..6], &[0, 1500, 2000, 2500, 3000, 3500]);
        assert_eq!(&u16_vals[6..], &sample_u16s);
    }

    #[test]
    fn retype_f32_npy_preserves_file_nodata_sentinel_on_partially_valid_pixels() {
        // 3-band, 1x2 CHW float32:
        // pixel 0: b1=NaN, b2=NaN, b3=154.2 (partially valid -> mask valid, b1/b2 -> -9999.0)
        // pixel 1: b1=NaN, b2=NaN, b3=NaN   (fully masked -> mask MIN, bands stay NaN)
        let raw_vals: [f32; 6] = [f32::NAN, f32::NAN, f32::NAN, f32::NAN, 154.2, f32::NAN];
        let mut raw_bytes = Vec::with_capacity(24);
        for v in raw_vals {
            raw_bytes.extend_from_slice(&v.to_le_bytes());
        }
        let wrapped = wrap_raw_f32_as_npy_with_mask(Bytes::from(raw_bytes), 3, 1, 2, true);
        let retyped = retype_f32_npy_with(&wrapped, SampleType::Float32, true, Some(-9999.0), |v| v)
            .expect("retype f32 with nodata");
        let body = npy_body(&retyped);
        let out: Vec<f32> = body
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(out[0], -9999.0);
        assert!(out[1].is_nan());
        assert_eq!(out[2], -9999.0);
        assert!(out[3].is_nan());
        assert!((out[4] - 154.2).abs() < 1e-4);
        assert!(out[5].is_nan());
        assert_eq!(out[6], f32::MAX);
        assert_eq!(out[7], f32::MIN);
    }
}
