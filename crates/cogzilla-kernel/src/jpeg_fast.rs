use std::cell::RefCell;
use std::sync::OnceLock;

use bytes::Bytes;
use mozjpeg::{ColorSpace, Compress};

thread_local! {
    static JPEG_RGB_SCRATCH: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
    static JPEG_OUT_SCRATCH: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

fn encode_mozjpeg(width: u32, height: u32, color: ColorSpace, pixels: &[u8], quality: u8) -> Bytes {
    let (w, h) = (width as usize, height as usize);
    if w == 0 || h == 0 {
        return Bytes::new();
    }
    JPEG_OUT_SCRATCH.with(|cell| {
        let mut out = cell.borrow_mut();
        out.clear();
        let target_cap = (pixels.len() / 6).max(16 * 1024);
        if out.capacity() < target_cap {
            out.reserve(target_cap);
        }
        let encoded = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> std::io::Result<()> {
            let mut comp = Compress::new(color);
            comp.set_fastest_defaults();
            comp.set_size(w, h);
            comp.set_quality(quality.clamp(1, 100) as f32);
            let mut started = comp.start_compress(&mut *out)?;
            started.write_scanlines(pixels)?;
            started.finish()?;
            Ok(())
        }));
        if !matches!(encoded, Ok(Ok(()))) {
            return Bytes::new();
        }
        let bytes = Bytes::copy_from_slice(&out);
        if out.capacity() > 4 * 1024 * 1024 {
            *out = Vec::new();
        }
        bytes
    })
}

fn encode_jpeg_rgb_raw(width: u32, height: u32, rgb: &[u8], quality: u8) -> Bytes {
    encode_mozjpeg(width, height, ColorSpace::JCS_RGB, rgb, quality)
}

/// Encode an RGB8 buffer to JPEG via SIMD `mozjpeg` (baseline fastest).
/// Automatically emits 1-component `Luma` JPEG when all pixels have `r == g == b`.
pub fn encode_jpeg_rgb(width: u32, height: u32, rgb: &[u8], quality: u8) -> Bytes {
    let pixels = (width as usize).saturating_mul(height as usize);
    let is_gray = rgb
        .chunks_exact(3)
        .take(pixels)
        .all(|p| p[0] == p[1] && p[1] == p[2]);
    if is_gray {
        return JPEG_RGB_SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            scratch.clear();
            scratch.resize(pixels, 0);
            for (src, dst) in rgb.chunks_exact(3).take(pixels).zip(scratch.iter_mut()) {
                *dst = src[0];
            }
            let encoded = encode_jpeg_gray(width, height, &scratch, quality);
            if scratch.capacity() > 4 * 1024 * 1024 {
                *scratch = Vec::new();
            }
            encoded
        });
    }
    encode_jpeg_rgb_raw(width, height, rgb, quality)
}

/// Encode a 1-channel Grayscale (`Luma`) buffer to JPEG via SIMD `mozjpeg` (baseline fastest).
pub fn encode_jpeg_gray(width: u32, height: u32, gray: &[u8], quality: u8) -> Bytes {
    encode_mozjpeg(width, height, ColorSpace::JCS_GRAYSCALE, gray, quality)
}

/// Encode an RGBA8 buffer to JPEG (stripping alpha onto black background, matching TiTiler).
/// Automatically emits a 1-component `Luma` JPEG when all opaque pixels are grayscale (`r==g==b`),
/// matching TiTiler's 1-band greyscale JPEG output and reducing size by ~30%.
/// Reuses a thread-local scratch buffer to avoid per-tile heap allocations.
pub fn encode_jpeg_rgba(width: u32, height: u32, rgba: &[u8], quality: u8) -> Bytes {
    encode_jpeg_rgba_as(width, height, rgba, quality, true)
}

/// [`encode_jpeg_rgba`], always in three components: for a caller that knows the picture
/// is of three bands, whatever its pixels happen to hold.
pub fn encode_jpeg_rgba_colour(width: u32, height: u32, rgba: &[u8], quality: u8) -> Bytes {
    encode_jpeg_rgba_as(width, height, rgba, quality, false)
}

fn encode_jpeg_rgba_as(width: u32, height: u32, rgba: &[u8], quality: u8, grey_if_grey: bool) -> Bytes {
    let pixels = (width as usize).saturating_mul(height as usize);
    let is_gray = grey_if_grey
        && rgba
        .chunks_exact(4)
        .take(pixels)
        .all(|p| p[3] == 0 || (p[0] == p[1] && p[1] == p[2]));
    JPEG_RGB_SCRATCH.with(|cell| {
        let mut scratch = cell.borrow_mut();
        scratch.clear();
        let encoded = if is_gray {
            scratch.resize(pixels, 0);
            for (src, dst) in rgba.chunks_exact(4).take(pixels).zip(scratch.iter_mut()) {
                *dst = if src[3] != 0 { src[0] } else { 0 };
            }
            encode_jpeg_gray(width, height, &scratch, quality)
        } else {
            let needed = pixels.saturating_mul(3);
            scratch.resize(needed, 0);
            for (src, dst) in rgba
                .chunks_exact(4)
                .take(pixels)
                .zip(scratch.chunks_exact_mut(3))
            {
                if src[3] != 0 {
                    dst[0] = src[0];
                    dst[1] = src[1];
                    dst[2] = src[2];
                }
            }
            encode_jpeg_rgb_raw(width, height, &scratch, quality)
        };
        if scratch.capacity() > 4 * 1024 * 1024 {
            *scratch = Vec::new();
        }
        encoded
    })
}

/// Pre-encoded static black JPEG for empty/out-of-bounds tiles.
pub fn empty_black_jpeg(size: u32) -> Bytes {
    match size {
        256 => {
            static JPG_256: OnceLock<Bytes> = OnceLock::new();
            JPG_256
                .get_or_init(|| encode_jpeg_rgb(256, 256, &vec![0u8; 256 * 256 * 3], 85))
                .clone()
        }
        512 => {
            static JPG_512: OnceLock<Bytes> = OnceLock::new();
            JPG_512
                .get_or_init(|| encode_jpeg_rgb(512, 512, &vec![0u8; 512 * 512 * 3], 85))
                .clone()
        }
        other => encode_jpeg_rgb(
            other,
            other,
            &vec![0u8; (other as usize) * (other as usize) * 3],
            85,
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mozjpeg_encodes_valid_rgb_gray_rgba_and_empty_black() {
        let mut rgb = vec![0u8; 64 * 64 * 3];
        for (i, px) in rgb.chunks_exact_mut(3).enumerate() {
            px[0] = (i & 0xff) as u8;
            px[1] = ((i * 3) & 0xff) as u8;
            px[2] = ((i * 7) & 0xff) as u8;
        }
        let jpg_rgb = encode_jpeg_rgb(64, 64, &rgb, 85);
        assert!(jpg_rgb.starts_with(&[0xFF, 0xD8, 0xFF]));
        assert!(jpg_rgb.ends_with(&[0xFF, 0xD9]));

        let gray: Vec<u8> = (0..64 * 64).map(|i| (i & 0xff) as u8).collect();
        let jpg_gray = encode_jpeg_gray(64, 64, &gray, 85);
        assert!(jpg_gray.starts_with(&[0xFF, 0xD8, 0xFF]));
        assert!(jpg_gray.ends_with(&[0xFF, 0xD9]));

        let rgba = vec![128u8; 32 * 32 * 4];
        let jpg_rgba = encode_jpeg_rgba(32, 32, &rgba, 85);
        assert!(jpg_rgba.starts_with(&[0xFF, 0xD8, 0xFF]));
        assert!(jpg_rgba.ends_with(&[0xFF, 0xD9]));

        let empty = empty_black_jpeg(256);
        assert!(empty.starts_with(&[0xFF, 0xD8, 0xFF]));
        assert!(empty.ends_with(&[0xFF, 0xD9]));
    }
}
