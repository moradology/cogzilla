pub mod lzw;
pub mod predictor;
pub mod slab;

use std::cell::RefCell;

use cogzilla_raster::SampleType;
use cogzilla_tiff::{ByteOrder, Compression, IfdMeta, MaskIfdMeta, PlanarConfig, Predictor};
use thiserror::Error;

pub use slab::{AlignedF32Slab, BandData, BlockStats, DecodedBand, DecodedTile};

/// Decode TIFF PackBits (compression code 32773) RLE stream into `dst`.
fn decode_packbits(src: &[u8], dst: &mut [u8]) -> Result<usize, CodecError> {
    let mut si = 0usize;
    let mut di = 0usize;
    while si < src.len() && di < dst.len() {
        let hdr = src[si] as i8;
        si += 1;
        if hdr >= 0 {
            let count = (hdr as usize) + 1;
            if si + count > src.len() {
                return Err(CodecError::Deflate("truncated PackBits literal run".into()));
            }
            let copy = count.min(dst.len() - di);
            dst[di..di + copy].copy_from_slice(&src[si..si + copy]);
            si += count;
            di += copy;
        } else if hdr != -128 {
            let count = (1i16 - hdr as i16) as usize;
            if si >= src.len() {
                return Err(CodecError::Deflate("truncated PackBits repeat run".into()));
            }
            let byte = src[si];
            si += 1;
            let copy = count.min(dst.len() - di);
            dst[di..di + copy].fill(byte);
            di += copy;
        }
    }
    Ok(di)
}

/// Compress a byte slice into a gzip stream (RFC 1952) using `libdeflater` at default level.
pub fn compress_gzip(input: &[u8]) -> Result<Vec<u8>, String> {
    let mut compressor = libdeflater::Compressor::new(libdeflater::CompressionLvl::default());
    let max_sz = compressor.gzip_compress_bound(input.len());
    let mut out = vec![0u8; max_sz];
    let n = compressor
        .gzip_compress(input, &mut out)
        .map_err(|e| format!("gzip compress error: {e:?}"))?;
    out.truncate(n);
    Ok(out)
}

/// Compress a byte slice into a zlib stream (RFC 1950) using `libdeflater` at default level.
pub fn compress_zlib(input: &[u8]) -> Result<Vec<u8>, String> {
    let mut compressor = libdeflater::Compressor::new(libdeflater::CompressionLvl::default());
    let max_sz = compressor.zlib_compress_bound(input.len());
    let mut out = vec![0u8; max_sz];
    let n = compressor
        .zlib_compress(input, &mut out)
        .map_err(|e| format!("zlib compress error: {e:?}"))?;
    out.truncate(n);
    Ok(out)
}

/// Compress a byte slice into a Zstd frame at `level` (e.g. `3`).
pub fn compress_zstd(input: &[u8], level: i32) -> Result<Vec<u8>, String> {
    // SAFETY: `ZSTD_compressBound` is a pure size calculation.
    let bound = unsafe { zstd_sys::ZSTD_compressBound(input.len()) };
    let mut out = vec![0u8; bound];
    // SAFETY: `out` has `out.len()` writable bytes and `input` has `input.len()` readable bytes.
    let rc = unsafe {
        zstd_sys::ZSTD_compress(
            out.as_mut_ptr().cast(),
            out.len(),
            input.as_ptr().cast(),
            input.len(),
            level,
        )
    };
    // SAFETY: `ZSTD_isError` inspects the return code only.
    if unsafe { zstd_sys::ZSTD_isError(rc) } != 0 {
        return Err(format!("zstd compress error: code {rc}"));
    }
    out.truncate(rc);
    Ok(out)
}

/// Compress a byte slice into a Brotli stream (RFC 7932) at fast HTTP quality (level 4, lgwin 22).
pub fn compress_brotli(input: &[u8]) -> Result<Vec<u8>, String> {
    use std::io::Write;
    let mut out = Vec::with_capacity(input.len() / 2);
    {
        let mut writer = brotli::CompressorWriter::new(&mut out, 4096, 4, 22);
        writer
            .write_all(input)
            .map_err(|e| format!("brotli compress error: {e}"))?;
    }
    Ok(out)
}

/// Decompress a Brotli-encoded buffer with an upper bound on decompressed size.
pub fn decompress_brotli_limited(input: &[u8], max_out_bytes: usize) -> Result<Vec<u8>, String> {
    use std::io::Read;
    if max_out_bytes == 0 {
        return Err("brotli payload exceeds 0 B limit".into());
    }
    let mut reader = brotli::Decompressor::new(input, 4096);
    let mut out = Vec::with_capacity(input.len().min(max_out_bytes));
    let mut chunk = [0u8; 8192];
    loop {
        let n = reader
            .read(&mut chunk)
            .map_err(|e| format!("brotli decompress error: {e}"))?;
        if n == 0 {
            break;
        }
        if out.len().saturating_add(n) > max_out_bytes {
            return Err(format!("brotli payload exceeds {max_out_bytes} B limit"));
        }
        out.extend_from_slice(&chunk[..n]);
    }
    Ok(out)
}

/// Decompress a gzip-encoded buffer using `libdeflater` with an upper bound on decompressed size
/// to guard against decompression bombs.
pub fn decompress_gzip_limited(input: &[u8], max_out_bytes: usize) -> Result<Vec<u8>, String> {
    if max_out_bytes == 0 {
        return Err("gzip payload exceeds 0 B limit".into());
    }
    if input.len() < 18 || input[0] != 0x1f || input[1] != 0x8b {
        return Err("invalid gzip header".into());
    }
    let isize_hint = u32::from_le_bytes([
        input[input.len() - 4],
        input[input.len() - 3],
        input[input.len() - 2],
        input[input.len() - 1],
    ]) as usize;
    // Cap initial allocation against an attacker-controlled ISIZE trailer; legitimate high-ratio
    // streams still grow smoothly via the 2x `InsufficientSpace` loop up to `max_out_bytes`.
    let max_initial = input.len().saturating_mul(16).max(65_536);
    let min_cap = 4096usize.min(max_out_bytes);
    let mut cap = isize_hint
        .min(max_initial)
        .max(input.len().saturating_mul(2))
        .clamp(min_cap, max_out_bytes);
    loop {
        let mut out = vec![0u8; cap];
        let res = CODEC_STATE.with(|cell| cell.borrow_mut().deflate.gzip_decompress(input, &mut out));
        match res {
            Ok(n) => {
                out.truncate(n);
                return Ok(out);
            }
            Err(libdeflater::DecompressionError::InsufficientSpace) => {
                if cap >= max_out_bytes {
                    return Err(format!(
                        "gzip payload exceeds {} MiB limit",
                        max_out_bytes / (1024 * 1024)
                    ));
                }
                cap = cap.saturating_mul(2).min(max_out_bytes);
            }
            Err(e) => return Err(format!("gzip decompression failed: {e:?}")),
        }
    }
}

#[derive(Debug, Error)]
pub enum CodecError {
    #[error("deflate decompression failed: {0}")]
    Deflate(String),
    #[error("zstd decompression failed: code {0}")]
    Zstd(usize),
    #[error("zstd decompression failed: {0}")]
    ZstdHeader(&'static str),
    #[error("LZW decompression failed: {0}")]
    Lzw(&'static str),
    #[error("JPEG decompression failed: {0}")]
    Jpeg(String),
    #[error("WebP decompression failed")]
    Webp,
    #[error("LERC decompression failed: {0}")]
    Lerc(String),
    #[error("uncompressed tile size mismatch: expected {expected} bytes, got {got}")]
    SizeMismatch { expected: usize, got: usize },
}

struct ThreadCodecState {
    deflate: libdeflater::Decompressor,
    zstd_dctx: *mut zstd_sys::ZSTD_DCtx,
    lzw: lzw::LzwDecoder,
    scratch: Vec<u8>,
    jpeg_merge: Vec<u8>,
}

impl ThreadCodecState {
    fn new() -> Self {
        let zstd_dctx = unsafe { zstd_sys::ZSTD_createDCtx() };
        assert!(!zstd_dctx.is_null(), "failed to allocate ZSTD_DCtx");
        Self {
            deflate: libdeflater::Decompressor::new(),
            zstd_dctx,
            lzw: lzw::LzwDecoder::new(),
            scratch: Vec::with_capacity(512 * 512 * 4),
            jpeg_merge: Vec::new(),
        }
    }
}

impl Drop for ThreadCodecState {
    fn drop(&mut self) {
        if !self.zstd_dctx.is_null() {
            unsafe {
                zstd_sys::ZSTD_freeDCtx(self.zstd_dctx);
            }
        }
    }
}

thread_local! {
    static CODEC_STATE: RefCell<ThreadCodecState> = RefCell::new(ThreadCodecState::new());
    static EXTERNAL_SCRATCH: RefCell<Vec<u8>> = RefCell::new(Vec::with_capacity(512 * 512 * 4));
}

/// Framing wrapper for deflate decompression via [`inflate_into`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeflateWrapper {
    /// RFC 1952 gzip stream.
    Gzip,
    /// RFC 1950 zlib stream.
    Zlib,
    /// RFC 1951 raw deflate stream.
    Raw,
}

/// Reusable thread-local byte buffer exposed by [`with_codec_scratch`].
pub struct CodecScratch<'a> {
    buf: &'a mut Vec<u8>,
}

impl CodecScratch<'_> {
    /// Resize the scratch buffer to `len` zero-initialized bytes and return it as a mutable slice.
    pub fn resize_zeroed(&mut self, len: usize) -> &mut [u8] {
        self.buf.clear();
        self.buf.resize(len, 0);
        self.buf.as_mut_slice()
    }

    /// Access the underlying reusable `Vec<u8>`.
    pub fn as_vec_mut(&mut self) -> &mut Vec<u8> {
        self.buf
    }
}

/// Borrow a thread-local reusable byte scratch buffer without per-chunk heap allocation.
///
/// Safe to combine with [`inflate_into`] and [`zstd_into`] inside `f` (uses a separate
/// thread-local cell from the codec context state). Clears the buffer on entry and exit,
/// shrinks oversized (>4 MiB) buffers back to 1 MiB on drop, and falls back to a local
/// `Vec` if called re-entrantly.
pub fn with_codec_scratch<R>(f: impl FnOnce(&mut CodecScratch<'_>) -> R) -> R {
    struct ExternalScratchGuard<'a>(&'a mut Vec<u8>);
    impl Drop for ExternalScratchGuard<'_> {
        fn drop(&mut self) {
            self.0.clear();
            if self.0.capacity() > 4 * 1024 * 1024 {
                *self.0 = Vec::with_capacity(512 * 512 * 4);
            }
        }
    }
    EXTERNAL_SCRATCH.with(|cell| {
        if let Ok(mut buf) = cell.try_borrow_mut() {
            buf.clear();
            let guard = ExternalScratchGuard(&mut buf);
            let mut scratch = CodecScratch { buf: guard.0 };
            f(&mut scratch)
        } else {
            let mut local = Vec::new();
            let mut scratch = CodecScratch { buf: &mut local };
            f(&mut scratch)
        }
    })
}

/// Validate minimal framing headers for gzip/zlib before allocating output buffers.
pub fn validate_deflate_header(src: &[u8], wrapper: DeflateWrapper) -> Result<(), CodecError> {
    match wrapper {
        DeflateWrapper::Gzip => {
            if src.len() < 18 || src[0] != 0x1f || src[1] != 0x8b {
                return Err(CodecError::Deflate(
                    "invalid or truncated gzip header".into(),
                ));
            }
        }
        DeflateWrapper::Zlib => {
            if src.len() < 6 {
                return Err(CodecError::Deflate("truncated zlib stream".into()));
            }
        }
        DeflateWrapper::Raw => {}
    }
    Ok(())
}

/// Validate Zstd frame header and declared content size against `max_out` before allocating.
pub fn validate_zstd_header(src: &[u8], max_out: usize) -> Result<(), CodecError> {
    if src.len() < 4 {
        return Err(CodecError::ZstdHeader("truncated zstd frame"));
    }
    // SAFETY: `src.as_ptr()` is valid for `src.len()` readable bytes.
    let content_size =
        unsafe { zstd_sys::ZSTD_getFrameContentSize(src.as_ptr().cast(), src.len()) };
    if content_size == zstd_sys::ZSTD_CONTENTSIZE_ERROR as u64 {
        return Err(CodecError::ZstdHeader("corrupt zstd frame header"));
    }
    if content_size != zstd_sys::ZSTD_CONTENTSIZE_UNKNOWN as u64 && content_size > max_out as u64 {
        return Err(CodecError::SizeMismatch {
            expected: max_out,
            got: usize::try_from(content_size).unwrap_or(usize::MAX),
        });
    }
    Ok(())
}

/// Decompress a gzip, zlib, or raw-deflate payload directly into `dst`, reusing the
/// thread-local `libdeflater::Decompressor`.
///
/// Enforces the caller's exact `dst.len()` bound without growing or over-allocating.
pub fn inflate_into(
    src: &[u8],
    dst: &mut [u8],
    wrapper: DeflateWrapper,
) -> Result<usize, CodecError> {
    validate_deflate_header(src, wrapper)?;
    CODEC_STATE.with(|cell| {
        let mut state = cell.borrow_mut();
        let res = match wrapper {
            DeflateWrapper::Gzip => state.deflate.gzip_decompress(src, dst),
            DeflateWrapper::Zlib => state.deflate.zlib_decompress(src, dst),
            DeflateWrapper::Raw => state.deflate.deflate_decompress(src, dst),
        };
        res.map_err(|e| CodecError::Deflate(format!("libdeflater {wrapper:?} error: {e:?}")))
    })
}

/// Validate header, allocate at most `expected` bytes, and decompress gzip/zlib/raw-deflate
/// into an owned `Vec<u8>` without an intermediate scratch copy.
pub fn inflate_to_vec(
    src: &[u8],
    expected: usize,
    wrapper: DeflateWrapper,
) -> Result<Vec<u8>, CodecError> {
    validate_deflate_header(src, wrapper)?;
    let mut dst = vec![0u8; expected];
    let n = inflate_into(src, &mut dst, wrapper)?;
    dst.truncate(n);
    Ok(dst)
}

/// Decompress a Zstd frame directly into `dst`, reusing the thread-local `ZSTD_DCtx`.
///
/// Validates `ZSTD_getFrameContentSize` against `dst.len()` before decompressing so hostile
/// frames declaring more bytes than the expected chunk size fail immediately.
pub fn zstd_into(src: &[u8], dst: &mut [u8]) -> Result<usize, CodecError> {
    validate_zstd_header(src, dst.len())?;
    CODEC_STATE.with(|cell| {
        let state = cell.borrow_mut();
        // SAFETY: `state.zstd_dctx` is a valid thread-local `ZSTD_DCtx`, `dst` is writable
        // for `dst.len()` bytes, and `src` is readable for `src.len()` bytes.
        let rc = unsafe {
            zstd_sys::ZSTD_decompressDCtx(
                state.zstd_dctx,
                dst.as_mut_ptr().cast(),
                dst.len(),
                src.as_ptr().cast(),
                src.len(),
            )
        };
        // SAFETY: `ZSTD_isError` inspects the return code only.
        if unsafe { zstd_sys::ZSTD_isError(rc) } != 0 {
            return Err(CodecError::Zstd(rc));
        }
        Ok(rc)
    })
}

/// Validate Zstd header/content-size, allocate at most `expected` bytes, and decompress into an
/// owned `Vec<u8>` without an intermediate scratch copy.
pub fn zstd_to_vec(src: &[u8], expected: usize) -> Result<Vec<u8>, CodecError> {
    validate_zstd_header(src, expected)?;
    let mut dst = vec![0u8; expected];
    let n = zstd_into(src, &mut dst)?;
    dst.truncate(n);
    Ok(dst)
}

/// Hard upper bound on a single decoded TIFF tile/strip (128 MiB) to prevent integer wrap
/// overflows and zip-bomb OOMs on hostile remote GeoTIFFs.
const MAX_DECODED_TILE_BYTES: usize = 128 * 1024 * 1024;

/// Resident in-memory bytes per sample of the [`BandData`] produced by [`decode_tile_rows`]
/// for a given TIFF [`SampleType`].
///
/// Differs from [`SampleType::bytes_per_sample`] (on-disk width) because `Int8`/`Int16`
/// reconstruct into `BandData::F32` (4 B) and `Uint32`/`Int32` reconstruct into
/// `BandData::F64` (8 B) to preserve full 32-bit integer precision.
#[inline(always)]
pub const fn decoded_bytes_per_sample(st: SampleType) -> usize {
    match st {
        SampleType::Bool | SampleType::Uint1 | SampleType::Uint4 | SampleType::Uint8 => 1,
        SampleType::Uint12 | SampleType::Uint16 => 2,
        SampleType::Int8 | SampleType::Int16 | SampleType::Float16 | SampleType::Float32 => 4,
        SampleType::Uint32
        | SampleType::Int32
        | SampleType::Uint64
        | SampleType::Int64
        | SampleType::Float64 => 8,
    }
}

/// Decompress and reconstruct a single internal COG tile into planar 64-byte-aligned `f32` bands
/// with precomputed `BlockStats`, reusing thread-local codec contexts and scratch memory.
pub fn decode_tile(
    compressed: &[u8],
    ifd: &IfdMeta,
    byte_order: ByteOrder,
    nodata: Option<f32>,
) -> Result<DecodedTile, CodecError> {
    decode_tile_rows(compressed, ifd, byte_order, nodata, ifd.tile_height)
}

/// Decompress and reconstruct a single TIFF tile or strip with an explicit row count `actual_rows`
/// (which is `< ifd.tile_height` for the final partial strip of a stripped TIFF).
///
/// Decoding only `actual_rows` avoids trailing zero-padding in `BlockStats`, prevents false
/// `alpha == 0` detection on stripped RGBA rasters, and eliminates stale scratch bytes on WebP strips.
pub fn decode_tile_rows(
    compressed: &[u8],
    ifd: &IfdMeta,
    byte_order: ByteOrder,
    nodata: Option<f32>,
    actual_rows: u32,
) -> Result<DecodedTile, CodecError> {
    decode_tile_rows_exact(
        compressed,
        ifd,
        byte_order,
        nodata.map(|v| v as f64),
        actual_rows,
    )
}

/// Like [`decode_tile_rows`], accepting full-precision `Option<f64>` nodata for wide sample types
/// (`Uint32`, `Int32`, `Uint64`, `Int64`, `Float64`).
pub fn decode_tile_rows_exact(
    compressed: &[u8],
    ifd: &IfdMeta,
    byte_order: ByteOrder,
    nodata_f64: Option<f64>,
    actual_rows: u32,
) -> Result<DecodedTile, CodecError> {
    decode_tile_rows_exact_impl(compressed, ifd, byte_order, nodata_f64, actual_rows, true)
}

/// Like [`decode_tile_rows_exact`], but preserves underlying color samples behind `alpha == 0`
/// (and skips LERC internal validity-mask zeroing) for `return_mask=false` requests (#620).
pub fn decode_tile_rows_exact_unmasked(
    compressed: &[u8],
    ifd: &IfdMeta,
    byte_order: ByteOrder,
    nodata_f64: Option<f64>,
    actual_rows: u32,
) -> Result<DecodedTile, CodecError> {
    decode_tile_rows_exact_impl(compressed, ifd, byte_order, nodata_f64, actual_rows, false)
}

fn decode_tile_rows_exact_impl(
    compressed: &[u8],
    ifd: &IfdMeta,
    byte_order: ByteOrder,
    nodata_f64: Option<f64>,
    actual_rows: u32,
    apply_mask: bool,
) -> Result<DecodedTile, CodecError> {
    let nodata: Option<f32> = nodata_f64.map(|v| v as f32);
    let tile_h = actual_rows.min(ifd.tile_height).max(1);
    let bands_in_tile = match ifd.planar_config {
        PlanarConfig::Chunky => ifd.samples_per_pixel as usize,
        PlanarConfig::Planar => 1,
    };
    let expected_bytes = ifd
        .sample_type
        .row_bytes(ifd.tile_width as usize, bands_in_tile)
        .and_then(|rb| rb.checked_mul(tile_h as usize))
        .filter(|&b| b > 0 && b <= MAX_DECODED_TILE_BYTES)
        .ok_or(CodecError::SizeMismatch {
            expected: MAX_DECODED_TILE_BYTES,
            got: compressed.len(),
        })?;
    // For sub-byte types (`Uint1`/`Uint4`), also bound the unpacked 1-byte-per-sample allocation.
    let _unpacked_bytes = (ifd.tile_width as usize)
        .checked_mul(tile_h as usize)
        .and_then(|n| n.checked_mul(bands_in_tile))
        .and_then(|n| n.checked_mul(ifd.sample_type.bytes_per_sample()))
        .filter(|&b| b > 0 && b <= MAX_DECODED_TILE_BYTES)
        .ok_or(CodecError::SizeMismatch {
            expected: MAX_DECODED_TILE_BYTES,
            got: compressed.len(),
        })?;

    // Never apply GDAL_NODATA to an ExtraSamples alpha channel (e.g. GDAL_NODATA=255 on RGBA
    // would turn opaque alpha=255 into NaN and blank every pixel). For U8/U16, raw samples stay
    // intact so apply_rgba_alpha_if_present simply clears nodata on the alpha band; for
    // F32-backed dtypes we defer nodata masking to the non-alpha color bands only.
    let chunky_alpha_idx = if ifd.planar_config == PlanarConfig::Chunky {
        alpha_channel_index(bands_in_tile, &ifd.extra_samples)
    } else {
        None
    };
    let defer_color_nodata = chunky_alpha_idx.is_some()
        && nodata.is_some()
        && !matches!(ifd.sample_type, SampleType::Uint8 | SampleType::Uint16);
    let recon_nodata = if defer_color_nodata { None } else { nodata };
    let recon_nodata_f64 = if defer_color_nodata { None } else { nodata_f64 };
    let deferred_nodata = if defer_color_nodata { nodata_f64 } else { None };

    // Fast path: Uncompressed + Predictor::None avoids touching thread-local scratch completely!
    if ifd.compression == Compression::None
        && ifd.predictor == Predictor::None
        && compressed.len() >= expected_bytes
    {
        let mut bands = reconstruct_bands(
            RawTileBuf::Shared(&compressed[..expected_bytes]),
            ifd,
            tile_h,
            bands_in_tile,
            byte_order,
            Predictor::None,
            recon_nodata,
            recon_nodata_f64,
        );
        apply_rgba_alpha_if_present(&mut bands, ifd, deferred_nodata, apply_mask);
        return Ok(DecodedTile { bands });
    }

    CODEC_STATE.with(|cell| {
        struct ScratchShrinkGuard<'a>(&'a mut ThreadCodecState);
        impl Drop for ScratchShrinkGuard<'_> {
            fn drop(&mut self) {
                if self.0.scratch.capacity() > 4 * 1024 * 1024 {
                    self.0.scratch = Vec::with_capacity(512 * 512 * 4);
                }
                if self.0.jpeg_merge.capacity() > 1024 * 1024 {
                    self.0.jpeg_merge = Vec::new();
                }
            }
        }
        let mut guard = cell.borrow_mut();
        let st_guard = ScratchShrinkGuard(&mut guard);
        let st = &mut *st_guard.0;
        if st.scratch.len() < expected_bytes {
            st.scratch.resize(expected_bytes, 0);
        }
        let raw_slice = &mut st.scratch[..expected_bytes];
        let mut lerc_int_mask: Option<Vec<u8>> = None;

        match ifd.compression {
            Compression::None => {
                if compressed.len() < expected_bytes {
                    if ifd.is_stripped {
                        raw_slice[..compressed.len()].copy_from_slice(compressed);
                        raw_slice[compressed.len()..].fill(0);
                    } else {
                        return Err(CodecError::SizeMismatch {
                            expected: expected_bytes,
                            got: compressed.len(),
                        });
                    }
                } else {
                    raw_slice.copy_from_slice(&compressed[..expected_bytes]);
                }
            }
            Compression::PackBits => {
                let written = decode_packbits(compressed, raw_slice)?;
                if written < expected_bytes {
                    if ifd.is_stripped {
                        raw_slice[written..].fill(0);
                    } else {
                        return Err(CodecError::SizeMismatch {
                            expected: expected_bytes,
                            got: written,
                        });
                    }
                }
            }
            Compression::Deflate => {
                // TIFF Adobe Deflate (code 8 / 32946) uses zlib framing; fall back to raw deflate if needed.
                let written = match st.deflate.zlib_decompress(compressed, raw_slice) {
                    Ok(n) => n,
                    Err(e1) => st
                        .deflate
                        .deflate_decompress(compressed, raw_slice)
                        .map_err(|_| CodecError::Deflate(format!("{e1:?}")))?,
                };
                if written < expected_bytes {
                    raw_slice[written..].fill(0);
                }
            }
            Compression::Zstd => {
                let code = unsafe {
                    zstd_sys::ZSTD_decompressDCtx(
                        st.zstd_dctx,
                        raw_slice.as_mut_ptr() as *mut std::ffi::c_void,
                        expected_bytes,
                        compressed.as_ptr() as *const std::ffi::c_void,
                        compressed.len(),
                    )
                };
                if unsafe { zstd_sys::ZSTD_isError(code) } != 0 {
                    return Err(CodecError::Zstd(code));
                }
                if code < expected_bytes {
                    raw_slice[code..].fill(0);
                }
            }
            Compression::Lzw => {
                let written = st.lzw.decode(compressed, raw_slice)?;
                if written < expected_bytes {
                    raw_slice[written..].fill(0);
                }
            }
            Compression::Jpeg => {
                if ifd.sample_type != SampleType::Uint8 {
                    return Err(CodecError::Jpeg(format!(
                        "JPEG compression requires uint8 samples, got {:?}",
                        ifd.sample_type
                    )));
                }
                // Merge shared TIFF JPEGTables (tag 347) with abbreviated tile bitstream if present
                let jpeg_bytes: &[u8] = if let Some(tables) = ifd.jpeg_tables.as_deref() {
                    if tables.len() >= 4 && compressed.len() >= 2 {
                        let t_end = if tables.ends_with(&[0xFF, 0xD9]) {
                            tables.len() - 2
                        } else {
                            tables.len()
                        };
                        let c_start = if compressed.starts_with(&[0xFF, 0xD8]) {
                            2
                        } else {
                            0
                        };
                        st.jpeg_merge.clear();
                        st.jpeg_merge.reserve(t_end + (compressed.len() - c_start));
                        st.jpeg_merge.extend_from_slice(&tables[..t_end]);
                        st.jpeg_merge.extend_from_slice(&compressed[c_start..]);
                        &st.jpeg_merge
                    } else {
                        compressed
                    }
                } else {
                    compressed
                };

                use zune_jpeg::zune_core::colorspace::ColorSpace;
                use zune_jpeg::zune_core::options::DecoderOptions;

                let opts = DecoderOptions::default()
                    .set_max_width((ifd.tile_width as usize).next_multiple_of(16))
                    .set_max_height((tile_h as usize).next_multiple_of(16));
                let mut dec = zune_jpeg::JpegDecoder::new_with_options(jpeg_bytes, opts);
                dec.decode_headers()
                    .map_err(|e| CodecError::Jpeg(format!("{e:?}")))?;
                let out_cs = match (bands_in_tile, ifd.photometric) {
                    (1, _) => ColorSpace::Luma,
                    // Photometric=2 (RGB): match the stream's own input colorspace (RGB if an Adobe
                    // APP14 transform=0 marker is present, YCbCr if stripped) so zune-jpeg performs
                    // zero colorspace conversion instead of failing on RGB->YCbCr.
                    (3, 2) => dec.get_input_colorspace().unwrap_or(ColorSpace::YCbCr),
                    (3, _) => ColorSpace::RGB,
                    (4, _) => dec.get_input_colorspace().unwrap_or(ColorSpace::RGBA),
                    _ => {
                        return Err(CodecError::Jpeg(format!(
                            "unsupported JPEG band count: {bands_in_tile}"
                        )));
                    }
                };
                if out_cs.num_components() != bands_in_tile {
                    return Err(CodecError::Jpeg(format!(
                        "JPEG colorspace {out_cs:?} has {} components, expected {bands_in_tile}",
                        out_cs.num_components()
                    )));
                }
                let (jw, jh) = dec.dimensions().ok_or_else(|| {
                    CodecError::Jpeg("JPEG header missing dimensions".into())
                })?;
                let decoded_jpeg_bytes = jw
                    .checked_mul(jh)
                    .and_then(|px| px.checked_mul(bands_in_tile))
                    .filter(|&b| b > 0 && b <= MAX_DECODED_TILE_BYTES)
                    .ok_or_else(|| {
                        CodecError::Jpeg(format!(
                            "JPEG dimensions {jw}x{jh}x{bands_in_tile} exceed max decoded tile size"
                        ))
                    })?;
                let _ = decoded_jpeg_bytes;
                dec.set_options(opts.jpeg_set_out_colorspace(out_cs));
                let tile_w = ifd.tile_width as usize;
                let tile_h_usize = tile_h as usize;
                if jw == tile_w {
                    let out_len = dec.output_buffer_size().unwrap_or(expected_bytes);
                    if out_len <= expected_bytes {
                        dec.decode_into(&mut raw_slice[..out_len])
                            .map_err(|e| CodecError::Jpeg(format!("{e:?}")))?;
                        if out_len < expected_bytes {
                            raw_slice[out_len..expected_bytes].fill(0);
                        }
                    } else {
                        let pixels = dec
                            .decode()
                            .map_err(|e| CodecError::Jpeg(format!("{e:?}")))?;
                        if pixels.len() < expected_bytes {
                            return Err(CodecError::Jpeg(format!(
                                "decoded JPEG buffer too small: {} < {expected_bytes}",
                                pixels.len()
                            )));
                        }
                        raw_slice[..expected_bytes].copy_from_slice(&pixels[..expected_bytes]);
                    }
                } else {
                    // MCU-padded or narrower JPEG stream: copy row-by-row so rows don't shear.
                    let pixels = dec
                        .decode()
                        .map_err(|e| CodecError::Jpeg(format!("{e:?}")))?;
                    let src_stride = jw * bands_in_tile;
                    let dst_stride = tile_w * bands_in_tile;
                    let copy_bytes = jw.min(tile_w) * bands_in_tile;
                    let copy_rows = jh.min(tile_h_usize);
                    raw_slice[..expected_bytes].fill(0);
                    for y in 0..copy_rows {
                        let src_off = y * src_stride;
                        if src_off + copy_bytes > pixels.len() {
                            break;
                        }
                        let dst_off = y * dst_stride;
                        raw_slice[dst_off..dst_off + copy_bytes]
                            .copy_from_slice(&pixels[src_off..src_off + copy_bytes]);
                    }
                }
            }
            Compression::Webp => {
                if ifd.sample_type != SampleType::Uint8 || (bands_in_tile != 3 && bands_in_tile != 4)
                {
                    return Err(CodecError::Webp);
                }
                let mut webp_w: std::ffi::c_int = 0;
                let mut webp_h: std::ffi::c_int = 0;
                let info_ok = unsafe {
                    libwebp_sys::WebPGetInfo(
                        compressed.as_ptr(),
                        compressed.len(),
                        &mut webp_w,
                        &mut webp_h,
                    )
                };
                if info_ok == 0
                    || webp_w <= 0
                    || webp_h <= 0
                    || webp_w as u32 > ifd.tile_width
                    || webp_h as u32 > tile_h
                {
                    return Err(CodecError::Webp);
                }
                // Zero thread-local scratch if WebP dimensions are smaller than the tile/strip so
                // unwritten rows/columns never leak bytes from prior tiles on this thread.
                if webp_w as u32 != ifd.tile_width || webp_h as u32 != tile_h {
                    raw_slice.fill(0);
                }
                let stride = (ifd.tile_width as i32) * (bands_in_tile as i32);
                let ptr = unsafe {
                    if bands_in_tile == 4 {
                        libwebp_sys::WebPDecodeRGBAInto(
                            compressed.as_ptr(),
                            compressed.len(),
                            raw_slice.as_mut_ptr(),
                            expected_bytes,
                            stride,
                        )
                    } else {
                        libwebp_sys::WebPDecodeRGBInto(
                            compressed.as_ptr(),
                            compressed.len(),
                            raw_slice.as_mut_ptr(),
                            expected_bytes,
                            stride,
                        )
                    }
                };
                if ptr.is_null() {
                    return Err(CodecError::Webp);
                }
            }
            Compression::Lerc | Compression::LercDeflate | Compression::LercZstd => {
                let unwrapped: Option<Vec<u8>> = match ifd.compression {
                    Compression::Lerc => None,
                    Compression::LercDeflate => {
                        let max_cap = expected_bytes.saturating_mul(2).clamp(65_536, MAX_DECODED_TILE_BYTES);
                        let mut cap = compressed
                            .len()
                            .saturating_mul(16)
                            .clamp(4096usize.min(max_cap), max_cap);
                        let buf = loop {
                            let mut out = vec![0u8; cap];
                            match st.deflate.zlib_decompress(compressed, &mut out) {
                                Ok(n) => {
                                    out.truncate(n);
                                    break out;
                                }
                                Err(libdeflater::DecompressionError::InsufficientSpace) => {
                                    if cap >= max_cap {
                                        return Err(CodecError::Lerc(
                                            "LERC_DEFLATE payload exceeds tile size cap".into(),
                                        ));
                                    }
                                    cap = cap.saturating_mul(2).min(max_cap);
                                }
                                Err(e1) => match st.deflate.deflate_decompress(compressed, &mut out) {
                                    Ok(n) => {
                                        out.truncate(n);
                                        break out;
                                    }
                                    Err(libdeflater::DecompressionError::InsufficientSpace) => {
                                        if cap >= max_cap {
                                            return Err(CodecError::Lerc(
                                                "LERC_DEFLATE payload exceeds tile size cap".into(),
                                            ));
                                        }
                                        cap = cap.saturating_mul(2).min(max_cap);
                                    }
                                    Err(_) => {
                                        return Err(CodecError::Lerc(format!(
                                            "LERC_DEFLATE outer inflate failed: {e1:?}"
                                        )));
                                    }
                                },
                            }
                        };
                        Some(buf)
                    }
                    Compression::LercZstd => {
                        let max_cap = expected_bytes.saturating_mul(2).clamp(65_536, MAX_DECODED_TILE_BYTES);
                        let content_sz = unsafe {
                            zstd_sys::ZSTD_getFrameContentSize(
                                compressed.as_ptr() as *const std::ffi::c_void,
                                compressed.len(),
                            )
                        };
                        if content_sz == zstd_sys::ZSTD_CONTENTSIZE_ERROR as u64 {
                            return Err(CodecError::Lerc(
                                "invalid LERC_ZSTD frame header".into(),
                            ));
                        }
                        let mut cap = if content_sz != zstd_sys::ZSTD_CONTENTSIZE_UNKNOWN as u64 {
                            let sz = content_sz as usize;
                            if sz > max_cap {
                                return Err(CodecError::Lerc(
                                    "LERC_ZSTD payload exceeds tile size cap".into(),
                                ));
                            }
                            let ratio_cap = compressed
                                .len()
                                .saturating_mul(16)
                                .clamp(4096usize.min(max_cap), max_cap);
                            sz.min(ratio_cap).max(1)
                        } else {
                            compressed
                                .len()
                                .saturating_mul(8)
                                .clamp(65_536usize.min(max_cap), max_cap)
                        };
                        let out = loop {
                            let mut buf = vec![0u8; cap];
                            let code = unsafe {
                                zstd_sys::ZSTD_decompressDCtx(
                                    st.zstd_dctx,
                                    buf.as_mut_ptr() as *mut std::ffi::c_void,
                                    buf.len(),
                                    compressed.as_ptr() as *const std::ffi::c_void,
                                    compressed.len(),
                                )
                            };
                            if unsafe { zstd_sys::ZSTD_isError(code) } == 0 {
                                buf.truncate(code);
                                break buf;
                            }
                            let err_code = unsafe { zstd_sys::ZSTD_getErrorCode(code) };
                            if err_code == zstd_sys::ZSTD_ErrorCode::ZSTD_error_dstSize_tooSmall
                                && cap < max_cap
                            {
                                cap = cap.saturating_mul(2).min(max_cap);
                                continue;
                            }
                            return Err(CodecError::Zstd(code));
                        };
                        Some(out)
                    }
                    _ => None,
                };
                let lerc_blob: &[u8] = unwrapped.as_deref().unwrap_or(compressed);
                // Pre-decode LERC1/LERC2 header dimension guard so a crafted tiny tile cannot
                // trigger a multi-hundred-MiB allocation inside `lerc_reader::decode_first`.
                if let Some((hdr_w, hdr_h, hdr_d)) = parse_lerc_header_dims(lerc_blob)
                    && (hdr_w == 0
                        || hdr_h == 0
                        || hdr_d == 0
                        || hdr_w > ifd.tile_width as usize
                        || hdr_h > tile_h as usize
                        || hdr_d > bands_in_tile)
                {
                    return Err(CodecError::Lerc(format!(
                        "LERC header dimensions {}x{}x{} exceed tile {}x{}x{}",
                        hdr_w, hdr_h, hdr_d, ifd.tile_width, tile_h, bands_in_tile
                    )));
                }
                let decoded = lerc_reader::decode_first(lerc_blob)
                    .map_err(|e| CodecError::Lerc(e.to_string()))?;
                let w = decoded.info.width as usize;
                let h = decoded.info.height as usize;
                let depth = decoded.info.depth as usize;
                if w == 0 || h == 0 || depth == 0 {
                    return Err(CodecError::Lerc("empty LERC block".into()));
                }
                if w > ifd.tile_width as usize
                    || h > tile_h as usize
                    || depth != bands_in_tile
                {
                    return Err(CodecError::Lerc(format!(
                        "LERC shape {}x{}x{} does not match tile {}x{}x{}",
                        w, h, depth, ifd.tile_width, tile_h, bands_in_tile
                    )));
                }
                raw_slice.fill(0);
                let tile_w = ifd.tile_width as usize;
                let mask = decoded.mask.as_deref();
                let copy_row_major = |raw: &mut [u8], bps: usize, write_sample: &dyn Fn(usize, &mut [u8])| {
                    for row in 0..h {
                        for col in 0..w {
                            let src_px = row * w + col;
                            let dst_px = row * tile_w + col;
                            for b in 0..depth {
                                let si = src_px * depth + b;
                                let di = (dst_px * depth + b) * bps;
                                write_sample(si, &mut raw[di..di + bps]);
                            }
                        }
                    }
                };
                let is_masked = |si: usize| -> bool {
                    mask.is_some_and(|m| m.get(si / depth).copied().unwrap_or(1) == 0)
                };
                use lerc_core::PixelData;
                match (&decoded.pixels, ifd.sample_type) {
                    (PixelData::F32(v), SampleType::Float32) => {
                        copy_row_major(raw_slice, 4, &|si, dst| {
                            let val = if is_masked(si) { f32::NAN } else { v[si] };
                            dst.copy_from_slice(&val.to_le_bytes());
                        });
                    }
                    (PixelData::F64(v), SampleType::Float64) => {
                        copy_row_major(raw_slice, 8, &|si, dst| {
                            let val = if is_masked(si) { f64::NAN } else { v[si] };
                            dst.copy_from_slice(&val.to_le_bytes());
                        });
                    }
                    (PixelData::U8(v), SampleType::Uint8) => {
                        let nd = nodata.map(|n| n as u8).unwrap_or(0);
                        copy_row_major(raw_slice, 1, &|si, dst| {
                            dst[0] = if is_masked(si) { nd } else { v[si] };
                        });
                    }
                    (PixelData::I8(v), SampleType::Int8) => {
                        let nd = nodata.map(|n| n as i8).unwrap_or(0);
                        copy_row_major(raw_slice, 1, &|si, dst| {
                            dst[0] = (if is_masked(si) { nd } else { v[si] }) as u8;
                        });
                    }
                    (PixelData::U16(v), SampleType::Uint16) => {
                        let nd = nodata.map(|n| n as u16).unwrap_or(0);
                        copy_row_major(raw_slice, 2, &|si, dst| {
                            let val = if is_masked(si) { nd } else { v[si] };
                            dst.copy_from_slice(&val.to_le_bytes());
                        });
                    }
                    (PixelData::I16(v), SampleType::Int16) => {
                        let nd = nodata.map(|n| n as i16).unwrap_or(0);
                        copy_row_major(raw_slice, 2, &|si, dst| {
                            let val = if is_masked(si) { nd } else { v[si] };
                            dst.copy_from_slice(&val.to_le_bytes());
                        });
                    }
                    (PixelData::U32(v), SampleType::Uint32) => {
                        let nd = nodata.map(|n| n as u32).unwrap_or(0);
                        copy_row_major(raw_slice, 4, &|si, dst| {
                            let val = if is_masked(si) { nd } else { v[si] };
                            dst.copy_from_slice(&val.to_le_bytes());
                        });
                    }
                    (PixelData::I32(v), SampleType::Int32) => {
                        let nd = nodata.map(|n| n as i32).unwrap_or(0);
                        copy_row_major(raw_slice, 4, &|si, dst| {
                            let val = if is_masked(si) { nd } else { v[si] };
                            dst.copy_from_slice(&val.to_le_bytes());
                        });
                    }
                    (other, st) => {
                        return Err(CodecError::Lerc(format!(
                            "LERC pixel type {:?} does not match TIFF sample type {:?}",
                            other.data_type(),
                            st
                        )));
                    }
                }
                // Preserve LERC per-pixel validity mask on integer dtypes even when no GDAL_NODATA
                // is configured (or nodata is out of range for the integer type).
                if !matches!(ifd.sample_type, SampleType::Float32 | SampleType::Float64)
                    && let Some(m) = mask
                    && m.contains(&0)
                {
                    let mut padded = vec![1u8; tile_w * tile_h as usize];
                    for row in 0..h {
                        for col in 0..w {
                            if m.get(row * w + col).copied().unwrap_or(1) == 0 {
                                padded[row * tile_w + col] = 0;
                            }
                        }
                    }
                    lerc_int_mask = Some(padded);
                }
            }
        }

        let (eff_byte_order, eff_predictor) = if matches!(
            ifd.compression,
            Compression::Lerc | Compression::LercDeflate | Compression::LercZstd
        ) {
            (ByteOrder::Little, Predictor::None)
        } else {
            (byte_order, ifd.predictor)
        };

        let mut bands = reconstruct_bands(
            RawTileBuf::Mut(raw_slice),
            ifd,
            tile_h,
            bands_in_tile,
            eff_byte_order,
            eff_predictor,
            recon_nodata,
            recon_nodata_f64,
        );

        apply_rgba_alpha_if_present(&mut bands, ifd, deferred_nodata, apply_mask);
        let mut tile = DecodedTile { bands };
        if apply_mask && let Some(ref m) = lerc_int_mask {
            apply_mask_to_tile(&mut tile, m);
        }
        Ok(tile)
    })
}

/// Return the 0-based alpha channel index from `ExtraSamples` (tag 338 = 1 or 2), supporting
/// both 2-band Gray+Alpha (`samples_per_pixel == 2`) and 4-band RGBA (`samples_per_pixel >= 4`).
pub fn alpha_channel_index(samples_per_pixel: usize, extra_samples: &[u16]) -> Option<usize> {
    if samples_per_pixel < 2 || extra_samples.is_empty() || extra_samples.len() >= samples_per_pixel
    {
        return None;
    }
    let pos = extra_samples.iter().position(|&v| v == 1 || v == 2)?;
    let base_samples = samples_per_pixel - extra_samples.len();
    Some(base_samples + pos)
}

/// If a chunky multi-band tile carries an alpha channel via `ExtraSamples` (tag 338 = 1 or 2,
/// e.g. 2-band Gray+Alpha or 4-band RGBA), mask transparent pixels (`alpha == 0`) to `f32::NAN`
/// in the color bands and recompute their `BlockStats` so every downstream kernel renders alpha
/// identically to TiTiler.
/// Parse `(width, height, depth)` directly from a LERC1 (`CntZImage`) or LERC2 (`Lerc2 `) header
/// before calling `lerc_reader::decode_first`, preventing crafted tiny tiles from triggering
/// multi-hundred-MiB allocations.
fn parse_lerc_header_dims(blob: &[u8]) -> Option<(usize, usize, usize)> {
    if blob.len() >= 10 && blob.starts_with(b"Lerc2 ") {
        let version = i32::from_le_bytes([blob[6], blob[7], blob[8], blob[9]]);
        if !(1..=6).contains(&version) {
            return None;
        }
        let (h_off, w_off, d_off) = if version <= 2 {
            if blob.len() < 18 {
                return None;
            }
            (10usize, 14usize, None)
        } else if version == 3 {
            if blob.len() < 22 {
                return None;
            }
            (14usize, 18usize, None)
        } else {
            if blob.len() < 26 {
                return None;
            }
            (14usize, 18usize, Some(22usize))
        };
        let h = u32::from_le_bytes([blob[h_off], blob[h_off + 1], blob[h_off + 2], blob[h_off + 3]]) as usize;
        let w = u32::from_le_bytes([blob[w_off], blob[w_off + 1], blob[w_off + 2], blob[w_off + 3]]) as usize;
        let d = d_off
            .map(|o| u32::from_le_bytes([blob[o], blob[o + 1], blob[o + 2], blob[o + 3]]) as usize)
            .unwrap_or(1);
        return Some((w, h, d));
    }
    if blob.len() >= 26 && blob.starts_with(b"CntZImage") {
        let h = u32::from_le_bytes([blob[18], blob[19], blob[20], blob[21]]) as usize;
        let w = u32::from_le_bytes([blob[22], blob[23], blob[24], blob[25]]) as usize;
        return Some((w, h, 1));
    }
    None
}

fn apply_rgba_alpha_if_present(
    bands: &mut [DecodedBand],
    ifd: &IfdMeta,
    deferred_color_nodata: Option<f64>,
    apply_alpha_mask: bool,
) {
    if ifd.planar_config != PlanarConfig::Chunky {
        return;
    }
    let Some(alpha_idx) = alpha_channel_index(bands.len(), &ifd.extra_samples) else {
        return;
    };
    // Clear any GDAL_NODATA metadata accidentally attached to a U8/U16 alpha band so e.g.
    // GDAL_NODATA=255 does not treat opaque alpha=255 as NaN.
    match &mut bands[alpha_idx].data {
        BandData::U8 { data, nodata } if nodata.is_some() => {
            *nodata = None;
            let mut mn = u8::MAX;
            let mut mx = u8::MIN;
            for &v in data.iter() {
                mn = mn.min(v);
                mx = mx.max(v);
            }
            bands[alpha_idx].stats = BlockStats {
                all_nodata: data.is_empty(),
                no_nodata: true,
                min: if data.is_empty() { f32::NAN } else { mn as f32 },
                max: if data.is_empty() { f32::NAN } else { mx as f32 },
            };
        }
        BandData::U16 { data, nodata } if nodata.is_some() => {
            *nodata = None;
            let mut mn = u16::MAX;
            let mut mx = u16::MIN;
            for &v in data.iter() {
                mn = mn.min(v);
                mx = mx.max(v);
            }
            bands[alpha_idx].stats = BlockStats {
                all_nodata: data.is_empty(),
                no_nodata: true,
                min: if data.is_empty() { f32::NAN } else { mn as f32 },
                max: if data.is_empty() { f32::NAN } else { mx as f32 },
            };
        }
        BandData::RgbU8Channel { rgb, channel, nodata } if nodata.is_some() => {
            *nodata = None;
            let ch = *channel as usize;
            let n = rgb.len() / 3;
            let mut mn = u8::MAX;
            let mut mx = u8::MIN;
            for i in 0..n {
                let v = rgb[i * 3 + ch];
                mn = mn.min(v);
                mx = mx.max(v);
            }
            bands[alpha_idx].stats = BlockStats {
                all_nodata: n == 0,
                no_nodata: true,
                min: if n == 0 { f32::NAN } else { mn as f32 },
                max: if n == 0 { f32::NAN } else { mx as f32 },
            };
        }
        _ => {}
    }

    // Apply deferred GDAL_NODATA to non-alpha color bands only.
    if let Some(nd_f64) = deferred_color_nodata {
        let nd_f32 = nd_f64 as f32;
        for (idx, b) in bands.iter_mut().enumerate() {
            if idx == alpha_idx {
                continue;
            }
            let is_f64 = matches!(b.data, BandData::F64(_));
            let nd_cmp = if is_f64 { nd_f64 } else { f64::from(nd_f32) };
            let limit = b.data.len();
            mask_band_where(b, limit, |_, v| !nd_cmp.is_nan() && v == nd_cmp);
        }
    }

    if !apply_alpha_mask {
        return;
    }

    // O(1) fast-path: if the alpha band has no nodata and min > 0, every pixel is opaque!
    let alpha_stats = bands[alpha_idx].stats;
    if alpha_stats.no_nodata && alpha_stats.min > 0.0 {
        return;
    }
    let n = bands[alpha_idx].data.len();
    if (0..n).all(|i| bands[alpha_idx].data.get_f32(i) > 0.0) {
        return;
    }
    let (before, rest) = bands.split_at_mut(alpha_idx);
    let (alpha_band, after) = rest.split_first_mut().unwrap();
    for b in before.iter_mut().chain(after.iter_mut()) {
        apply_alpha_data_to_band(b, &alpha_band.data);
    }
}

enum RawTileBuf<'a> {
    Shared(&'a [u8]),
    Mut(&'a mut [u8]),
}

impl RawTileBuf<'_> {
    fn as_slice(&self) -> &[u8] {
        match self {
            Self::Shared(s) => s,
            Self::Mut(s) => s,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn reconstruct_bands(
    raw: RawTileBuf<'_>,
    ifd: &IfdMeta,
    tile_h: u32,
    bands_in_tile: usize,
    byte_order: ByteOrder,
    pred: Predictor,
    recon_nodata: Option<f32>,
    recon_nodata_f64: Option<f64>,
) -> Vec<DecodedBand> {
    let horiz = pred == Predictor::HorizontalInt;
    match (ifd.sample_type, pred, raw) {
        (SampleType::Float32, Predictor::None, r) => predictor::reconstruct_predictor1(
            r.as_slice(),
            ifd.tile_width,
            tile_h,
            bands_in_tile,
            byte_order,
            recon_nodata,
        ),
        (SampleType::Float32, Predictor::HorizontalInt, r) => predictor::reconstruct_predictor2(
            r.as_slice(),
            ifd.tile_width,
            tile_h,
            bands_in_tile,
            byte_order,
            recon_nodata,
        ),
        (SampleType::Float32, Predictor::FloatingPoint, RawTileBuf::Mut(r)) => {
            predictor::reconstruct_predictor3(r, ifd.tile_width, tile_h, bands_in_tile, recon_nodata)
        }
        (SampleType::Float32, Predictor::FloatingPoint, RawTileBuf::Shared(_)) => {
            unreachable!("Predictor::FloatingPoint always uses scratch buffer")
        }
        (SampleType::Uint16, _, r) => predictor::reconstruct_u16(
            r.as_slice(),
            ifd.tile_width,
            tile_h,
            bands_in_tile,
            byte_order,
            horiz,
            recon_nodata,
        ),
        (SampleType::Int16, _, r) => predictor::reconstruct_i16(
            r.as_slice(),
            ifd.tile_width,
            tile_h,
            bands_in_tile,
            byte_order,
            horiz,
            recon_nodata,
        ),
        (SampleType::Uint8, _, r) => predictor::reconstruct_u8(
            r.as_slice(),
            ifd.tile_width,
            tile_h,
            bands_in_tile,
            horiz,
            recon_nodata,
        ),
        (SampleType::Int8, _, r) => predictor::reconstruct_i8(
            r.as_slice(),
            ifd.tile_width,
            tile_h,
            bands_in_tile,
            horiz,
            recon_nodata,
        ),
        (SampleType::Uint32, _, r) => predictor::reconstruct_int32(
            r.as_slice(),
            ifd.tile_width,
            tile_h,
            bands_in_tile,
            byte_order,
            false,
            horiz,
            recon_nodata_f64,
        ),
        (SampleType::Int32, _, r) => predictor::reconstruct_int32(
            r.as_slice(),
            ifd.tile_width,
            tile_h,
            bands_in_tile,
            byte_order,
            true,
            horiz,
            recon_nodata_f64,
        ),
        (SampleType::Uint1, _, r) => predictor::reconstruct_subbyte(
            r.as_slice(),
            ifd.tile_width,
            tile_h,
            bands_in_tile,
            1,
            recon_nodata,
        ),
        (SampleType::Uint4, _, r) => predictor::reconstruct_subbyte(
            r.as_slice(),
            ifd.tile_width,
            tile_h,
            bands_in_tile,
            4,
            recon_nodata,
        ),
        (SampleType::Uint12, _, r) => predictor::reconstruct_uint12(
            r.as_slice(),
            ifd.tile_width,
            tile_h,
            bands_in_tile,
            recon_nodata,
        ),
        (SampleType::Uint64, _, r) => predictor::reconstruct_int64(
            r.as_slice(),
            ifd.tile_width,
            tile_h,
            bands_in_tile,
            byte_order,
            false,
            horiz,
            recon_nodata_f64,
        ),
        (SampleType::Int64, _, r) => predictor::reconstruct_int64(
            r.as_slice(),
            ifd.tile_width,
            tile_h,
            bands_in_tile,
            byte_order,
            true,
            horiz,
            recon_nodata_f64,
        ),
        (SampleType::Float16, Predictor::None, r) => predictor::reconstruct_f16_uncompressed(
            r.as_slice(),
            ifd.tile_width,
            tile_h,
            bands_in_tile,
            byte_order,
            recon_nodata,
        ),
        (SampleType::Float16, p, RawTileBuf::Mut(r)) => predictor::reconstruct_f16(
            r,
            ifd.tile_width,
            tile_h,
            bands_in_tile,
            byte_order,
            p,
            recon_nodata,
        ),
        (SampleType::Float16, _, RawTileBuf::Shared(_)) => {
            unreachable!("Float16 with predictor always uses scratch buffer")
        }
        (SampleType::Float64, Predictor::None, r) => predictor::reconstruct_f64_uncompressed(
            r.as_slice(),
            ifd.tile_width,
            tile_h,
            bands_in_tile,
            byte_order,
            recon_nodata_f64,
        ),
        (SampleType::Float64, p, RawTileBuf::Mut(r)) => predictor::reconstruct_f64(
            r,
            ifd.tile_width,
            tile_h,
            bands_in_tile,
            byte_order,
            p,
            recon_nodata_f64,
        ),
        (SampleType::Float64, _, RawTileBuf::Shared(_)) => {
            unreachable!("Float64 with predictor always uses scratch buffer")
        }
        (SampleType::Bool, _, _) => {
            unreachable!("TIFF IFDs never produce Bool sample types")
        }
    }
}

/// Mask pixels in `b` where `is_masked(i, value)` returns `true` (setting them to `NaN`)
/// and recompute `b.stats` in a single pass across both `BandData::F64` and `f32` storage.
fn mask_band_where(b: &mut DecodedBand, limit: usize, mut is_masked: impl FnMut(usize, f64) -> bool) {
    if let BandData::F64(ref mut slice) = b.data {
        let n = slice.len().min(limit);
        let mut all_nd = true;
        let mut no_nd = true;
        let mut mn = f64::INFINITY;
        let mut mx = f64::NEG_INFINITY;
        for (i, slot) in slice[..n].iter_mut().enumerate() {
            let v = *slot;
            if v.is_nan() || is_masked(i, v) {
                *slot = f64::NAN;
                no_nd = false;
            } else {
                all_nd = false;
                if v < mn {
                    mn = v;
                }
                if v > mx {
                    mx = v;
                }
            }
        }
        b.stats = BlockStats {
            all_nodata: all_nd,
            no_nodata: no_nd,
            min: if all_nd {
                f32::NAN
            } else {
                (mn as f32).clamp(f32::MIN, f32::MAX)
            },
            max: if all_nd {
                f32::NAN
            } else {
                (mx as f32).clamp(f32::MIN, f32::MAX)
            },
        };
        return;
    }
    let slice = b.data.make_f32_mut();
    let n = slice.len().min(limit);
    let mut all_nd = true;
    let mut no_nd = true;
    let mut mn = f32::INFINITY;
    let mut mx = f32::NEG_INFINITY;
    for (i, slot) in slice[..n].iter_mut().enumerate() {
        let v = *slot;
        if v.is_nan() || is_masked(i, f64::from(v)) {
            *slot = f32::NAN;
            no_nd = false;
        } else {
            all_nd = false;
            if v < mn {
                mn = v;
            }
            if v > mx {
                mx = v;
            }
        }
    }
    b.stats = BlockStats {
        all_nodata: all_nd,
        no_nodata: no_nd,
        min: if all_nd { f32::NAN } else { mn },
        max: if all_nd { f32::NAN } else { mx },
    };
}

/// Mask transparent pixels (`alpha == 0` or `NaN`) to `f32::NAN` in a single-band decoded tile
/// (used for `PlanarConfig::Planar` Gray+Alpha / RGBA COGs where the alpha channel lives in a separate plane).
pub fn apply_alpha_band_to_tile(tile: &mut DecodedTile, alpha: &DecodedBand) {
    if alpha.stats.no_nodata && alpha.stats.min > 0.0 {
        return;
    }
    let n = alpha.data.len();
    if (0..n).all(|i| alpha.data.get_f32(i) > 0.0) {
        return;
    }
    for b in &mut tile.bands {
        apply_alpha_data_to_band(b, &alpha.data);
    }
}

fn apply_alpha_data_to_band(b: &mut DecodedBand, alpha: &BandData) {
    mask_band_where(b, alpha.len(), |i, _| {
        let a = alpha.get_f32(i);
        a <= 0.0 || a.is_nan()
    });
}

/// Decompress a 1-bit (packed MSB-first per row) or 8-bit internal transparency mask tile
/// (`NewSubfileType & 4 != 0`) into a `tile_width * tile_height` byte mask (`0` = transparent, `255` = opaque).
pub fn decode_mask_tile(compressed: &[u8], mask: &MaskIfdMeta) -> Result<Vec<u8>, CodecError> {
    decode_mask_tile_rows(compressed, mask, mask.tile_height)
}

/// Decompress a 1-bit or 8-bit internal transparency mask tile or strip with an explicit row count
/// `actual_rows` (which is `< mask.tile_height` for the final partial strip of a stripped mask IFD).
pub fn decode_mask_tile_rows(
    compressed: &[u8],
    mask: &MaskIfdMeta,
    actual_rows: u32,
) -> Result<Vec<u8>, CodecError> {
    let w = mask.tile_width as usize;
    let h = actual_rows.min(mask.tile_height).max(1) as usize;
    // Enforce MAX_DECODED_TILE_BYTES on the unpacked output pixel count (1 byte/px),
    // not just packed 1-bit row_bytes (which would allow 8x larger 1 GiB allocations).
    let unpacked_bytes = w
        .checked_mul(h)
        .filter(|&px| px > 0 && px <= MAX_DECODED_TILE_BYTES)
        .ok_or(CodecError::SizeMismatch {
            expected: MAX_DECODED_TILE_BYTES,
            got: compressed.len(),
        })?;
    let row_bytes = if mask.bits_per_sample == 1 {
        w.div_ceil(8)
    } else {
        w
    };
    let expected = row_bytes
        .checked_mul(h)
        .filter(|&b| b > 0 && b <= unpacked_bytes)
        .ok_or(CodecError::SizeMismatch {
            expected: MAX_DECODED_TILE_BYTES,
            got: compressed.len(),
        })?;

    let decompress_into = |st: &mut ThreadCodecState, dst: &mut [u8]| -> Result<(), CodecError> {
        let written = match mask.compression {
            Compression::None => {
                let n = compressed.len().min(dst.len());
                dst[..n].copy_from_slice(&compressed[..n]);
                n
            }
            Compression::Deflate => match st.deflate.zlib_decompress(compressed, dst) {
                Ok(n) => n,
                Err(e1) => st
                    .deflate
                    .deflate_decompress(compressed, dst)
                    .map_err(|_| CodecError::Deflate(format!("{e1:?}")))?,
            },
            Compression::Zstd => {
                let code = unsafe {
                    zstd_sys::ZSTD_decompressDCtx(
                        st.zstd_dctx,
                        dst.as_mut_ptr() as *mut std::ffi::c_void,
                        dst.len(),
                        compressed.as_ptr() as *const std::ffi::c_void,
                        compressed.len(),
                    )
                };
                if unsafe { zstd_sys::ZSTD_isError(code) } != 0 {
                    return Err(CodecError::Zstd(code));
                }
                code
            }
            Compression::Lzw => st.lzw.decode(compressed, dst)?,
            Compression::PackBits => decode_packbits(compressed, dst)?,
            _ => {
                return Err(CodecError::Deflate(
                    "unsupported compression in internal mask IFD".into(),
                ));
            }
        };
        if written < dst.len() {
            dst[written..].fill(0);
        }
        Ok(())
    };

    if mask.bits_per_sample == 1 {
        CODEC_STATE.with(|cell| {
            let mut guard = cell.borrow_mut();
            let st = &mut *guard;
            if st.scratch.len() < expected {
                st.scratch.resize(expected, 0);
            }
            let mut tmp_scratch = std::mem::take(&mut st.scratch);
            let res = decompress_into(st, &mut tmp_scratch[..expected]);
            let unpacked = res.map(|()| {
                let packed = &tmp_scratch[..expected];
                if w.is_multiple_of(8) && packed.iter().all(|&b| b == 0xFF) {
                    return vec![255u8; w * h];
                }
                let mut out = vec![0u8; w * h];
                for r in 0..h {
                    let src_row = &packed[r * row_bytes..(r + 1) * row_bytes];
                    let dst_row = &mut out[r * w..(r + 1) * w];
                    for c in 0..w {
                        let bit = (src_row[c >> 3] >> (7 - (c & 7))) & 1;
                        dst_row[c] = if bit != 0 { 255 } else { 0 };
                    }
                }
                out
            });
            st.scratch = if tmp_scratch.capacity() > 4 * 1024 * 1024 {
                Vec::with_capacity(512 * 512 * 4)
            } else {
                tmp_scratch
            };
            unpacked
        })
    } else {
        let mut out = vec![0u8; expected];
        CODEC_STATE.with(|cell| {
            let mut guard = cell.borrow_mut();
            decompress_into(&mut guard, &mut out)
        })?;
        if mask.predictor == Predictor::HorizontalInt {
            for r in 0..h {
                let row = &mut out[r * w..(r + 1) * w];
                let mut acc = 0u8;
                for px in row.iter_mut() {
                    acc = acc.wrapping_add(*px);
                    *px = acc;
                }
            }
        }
        Ok(out)
    }
}

/// Apply an unpacked `0`/`255` transparency mask to every band of a `DecodedTile`,
/// setting masked pixels to `f32::NAN` and updating `BlockStats`.
pub fn apply_mask_to_tile(tile: &mut DecodedTile, mask: &[u8]) {
    if mask.iter().all(|&m| m != 0) {
        return;
    }
    for b in &mut tile.bands {
        mask_band_where(b, mask.len(), |i, _| mask[i] == 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cogzilla_raster::Affine2D;

    #[test]
    fn hostile_tile_dimension_overflow_is_rejected_without_allocation() {
        // 2_147_418_113 * 2_147_549_185 * 1 * 4 wraps mod 2^64 to 4 bytes in release mode!
        let ident = Affine2D {
            a: 1.0,
            b: 0.0,
            c: 0.0,
            d: 0.0,
            e: 1.0,
            f: 0.0,
        };
        let ifd = IfdMeta {
            width: 2_147_418_113,
            height: 2_147_549_185,
            tile_width: 2_147_418_113,
            tile_height: 2_147_549_185,
            tiles_across: 1,
            tiles_down: 1,
            samples_per_pixel: 1,
            sample_type: SampleType::Float32,
            compression: Compression::None,
            predictor: Predictor::None,
            planar_config: PlanarConfig::Chunky,
            photometric: 1,
            extra_samples: Vec::new(),
            is_stripped: false,
            tile_offsets: Box::new([0]),
            tile_byte_counts: Box::new([4]),
            jpeg_tables: None,
            mask_ifd: None,
            pixel_to_crs: ident,
            crs_to_pixel: ident,
            gcp_poly: None,
        };
        assert!(matches!(
            decode_tile(&[0u8; 4], &ifd, ByteOrder::Little, None),
            Err(CodecError::SizeMismatch { .. })
        ));
    }

    #[test]
    fn apply_alpha_band_to_tile_masks_zero_alpha_and_updates_stats() {
        let mut tile = DecodedTile {
            bands: vec![DecodedBand {
                width: 2,
                height: 2,
                data: BandData::U8 {
                    data: vec![10u8, 20, 30, 40].into_boxed_slice(),
                    nodata: None,
                },
                stats: BlockStats {
                    all_nodata: false,
                    no_nodata: true,
                    min: 10.0,
                    max: 40.0,
                },
            }],
        };
        let alpha = DecodedBand {
            width: 2,
            height: 2,
            data: BandData::U8 {
                data: vec![255u8, 0, 255, 0].into_boxed_slice(),
                nodata: None,
            },
            stats: BlockStats {
                all_nodata: false,
                no_nodata: true,
                min: 0.0,
                max: 255.0,
            },
        };
        apply_alpha_band_to_tile(&mut tile, &alpha);
        let b = &tile.bands[0];
        assert!(!b.stats.all_nodata);
        assert!(!b.stats.no_nodata);
        assert_eq!(b.data.get_f32(0), 10.0);
        assert!(b.data.get_f32(1).is_nan());
        assert_eq!(b.data.get_f32(2), 30.0);
        assert!(b.data.get_f32(3).is_nan());
    }

    #[test]
    fn decode_mask_tile_undoes_horizontal_predictor_on_8bit_mask_and_gray_alpha_index() {
        // Horizontal-differenced opaque 4x2 8-bit mask: [255, 0, 0, 0] per row -> [255, 255, 255, 255]
        let mask_meta = MaskIfdMeta {
            width: 4,
            height: 2,
            tile_width: 4,
            tile_height: 2,
            tiles_across: 1,
            tiles_down: 1,
            bits_per_sample: 8,
            compression: Compression::None,
            predictor: Predictor::HorizontalInt,
            is_stripped: false,
            tile_offsets: Box::new([0]),
            tile_byte_counts: Box::new([8]),
        };
        let raw_diff = [255u8, 0, 0, 0, 0, 255, 0, 1];
        let decoded = decode_mask_tile(&raw_diff, &mask_meta).unwrap();
        assert_eq!(decoded, vec![255, 255, 255, 255, 0, 255, 255, 0]);

        // 2-band Gray+Alpha and 4-band RGBA ExtraSamples index detection
        assert_eq!(alpha_channel_index(2, &[2]), Some(1));
        assert_eq!(alpha_channel_index(4, &[1]), Some(3));
        assert_eq!(alpha_channel_index(4, &[0]), None);
    }

    #[test]
    fn decode_tile_rows_clamps_partial_final_strip_stats_and_rgba_alpha() {
        // 2x5 stripped RGBA image with tile_height=4: final strip has actual_rows=1 (2 pixels = 8 bytes).
        // All alpha values in that 1 row are 255 (opaque), so stats must stay no_nodata=true
        // and min=100.0 (not polluted by zero-padded rows 1..4!).
        let ifd = IfdMeta {
            width: 2,
            height: 5,
            tile_width: 2,
            tile_height: 4,
            tiles_across: 1,
            tiles_down: 2,
            samples_per_pixel: 4,
            sample_type: SampleType::Uint8,
            planar_config: PlanarConfig::Chunky,
            compression: Compression::None,
            predictor: Predictor::None,
            photometric: 2,
            extra_samples: vec![2],
            jpeg_tables: None,
            mask_ifd: None,
            is_stripped: true,
            pixel_to_crs: cogzilla_raster::Affine2D {
                a: 1.0,
                b: 0.0,
                c: 0.0,
                d: 0.0,
                e: 1.0,
                f: 0.0,
            },
            crs_to_pixel: cogzilla_raster::Affine2D {
                a: 1.0,
                b: 0.0,
                c: 0.0,
                d: 0.0,
                e: 1.0,
                f: 0.0,
            },
            gcp_poly: None,
            tile_offsets: Box::new([0, 32]),
            tile_byte_counts: Box::new([32, 8]),
        };
        assert_eq!(ifd.block_height(0), 4);
        assert_eq!(ifd.block_height(1), 1);

        let raw_last_strip = [100u8, 120, 140, 255, 200, 210, 220, 255];
        let tile = decode_tile_rows(
            &raw_last_strip,
            &ifd,
            ByteOrder::Little,
            None,
            ifd.block_height(1),
        )
        .unwrap();
        assert_eq!(tile.bands[0].height, 1);
        assert_eq!(tile.bands[0].data.len(), 2);
        assert!(tile.bands[0].stats.no_nodata);
        assert_eq!(tile.bands[0].stats.min, 100.0);
        assert_eq!(tile.bands[0].stats.max, 200.0);
        assert_eq!(tile.bands[3].stats.min, 255.0);
    }

    #[test]
    fn one_bit_mask_enforces_unpacked_byte_limit_and_recomputes_stats() {
        let huge_mask = MaskIfdMeta {
            width: 32768,
            height: 32768,
            tile_width: 32768,
            tile_height: 32768,
            tiles_across: 1,
            tiles_down: 1,
            bits_per_sample: 1,
            compression: Compression::None,
            predictor: Predictor::None,
            is_stripped: false,
            tile_offsets: Box::new([0]),
            tile_byte_counts: Box::new([1]),
        };
        assert!(
            matches!(decode_mask_tile(&[0u8], &huge_mask), Err(CodecError::SizeMismatch { .. })),
            "32768x32768 1-bit mask unpacks to 1 GiB and must be rejected"
        );

        let mut slab = AlignedF32Slab::new_uninit(4);
        slab.as_mut_slice().copy_from_slice(&[1.0, 10.0, 20.0, 999.0]);
        let mut tile = DecodedTile {
            bands: vec![DecodedBand {
                width: 2,
                height: 2,
                data: BandData::F32(slab),
                stats: BlockStats {
                    all_nodata: false,
                    no_nodata: true,
                    min: 1.0,
                    max: 999.0,
                },
            }],
        };
        apply_mask_to_tile(&mut tile, &[0, 255, 255, 0]);
        assert!(!tile.bands[0].stats.all_nodata);
        assert!(!tile.bands[0].stats.no_nodata);
        assert_eq!(tile.bands[0].stats.min, 10.0);
        assert_eq!(tile.bands[0].stats.max, 20.0);

        apply_mask_to_tile(&mut tile, &[0, 0, 0, 0]);
        assert!(tile.bands[0].stats.all_nodata);
        assert!(tile.bands[0].stats.min.is_nan());
        assert!(tile.bands[0].stats.max.is_nan());
    }

    #[test]
    fn chunky_rgba_and_gray_alpha_ignore_gdal_nodata_on_alpha_channel() {
        let mut ifd_u8 = IfdMeta {
            width: 2,
            height: 1,
            tile_width: 2,
            tile_height: 1,
            tiles_across: 1,
            tiles_down: 1,
            samples_per_pixel: 4,
            sample_type: SampleType::Uint8,
            planar_config: PlanarConfig::Chunky,
            compression: Compression::None,
            predictor: Predictor::None,
            photometric: 2,
            extra_samples: vec![2],
            jpeg_tables: None,
            mask_ifd: None,
            is_stripped: false,
            pixel_to_crs: cogzilla_raster::Affine2D {
                a: 1.0,
                b: 0.0,
                c: 0.0,
                d: 0.0,
                e: 1.0,
                f: 0.0,
            },
            crs_to_pixel: cogzilla_raster::Affine2D {
                a: 1.0,
                b: 0.0,
                c: 0.0,
                d: 0.0,
                e: 1.0,
                f: 0.0,
            },
            gcp_poly: None,
            tile_offsets: Box::new([0]),
            tile_byte_counts: Box::new([8]),
        };
        // Pixel 0: opaque (R=10, G=20, B=30, A=255), Pixel 1: transparent (R=40, G=50, B=60, A=0)
        // With GDAL_NODATA = 255, opaque alpha=255 MUST NOT become NaN and blank pixel 0!
        let raw_u8 = [10u8, 20, 30, 255, 40, 50, 60, 0];
        let tile_u8 = decode_tile(&raw_u8, &ifd_u8, ByteOrder::Little, Some(255.0)).unwrap();
        assert_eq!(tile_u8.bands[0].data.get_f32(0), 10.0);
        assert_eq!(tile_u8.bands[1].data.get_f32(0), 20.0);
        assert_eq!(tile_u8.bands[2].data.get_f32(0), 30.0);
        assert!(tile_u8.bands[0].data.get_f32(1).is_nan());

        // Float32 Gray+Alpha with GDAL_NODATA = 1.0: opaque alpha=1.0 must stay opaque,
        // while color value 1.0 is masked to NaN.
        ifd_u8.samples_per_pixel = 2;
        ifd_u8.sample_type = SampleType::Float32;
        ifd_u8.photometric = 1;
        let mut raw_f32 = Vec::new();
        for v in [42.0f32, 1.0f32, 1.0f32, 1.0f32] {
            raw_f32.extend_from_slice(&v.to_le_bytes());
        }
        let tile_f32 = decode_tile(&raw_f32, &ifd_u8, ByteOrder::Little, Some(1.0)).unwrap();
        assert_eq!(tile_f32.bands[0].data.get_f32(0), 42.0);
        assert!(tile_f32.bands[0].data.get_f32(1).is_nan());
    }

    #[test]
    fn parse_lerc_header_dims_respects_lerc1_and_lerc2_versions() {
        // LERC2 v2: 0..6 magic, 6..10 version=2, 10..14 height, 14..18 width (depth=1)
        let mut v2 = Vec::new();
        v2.extend_from_slice(b"Lerc2 ");
        v2.extend_from_slice(&2i32.to_le_bytes());
        v2.extend_from_slice(&256u32.to_le_bytes()); // height
        v2.extend_from_slice(&128u32.to_le_bytes()); // width
        v2.extend_from_slice(&32768u32.to_le_bytes()); // valid_pixel_count (must not be read as w/h)
        assert_eq!(parse_lerc_header_dims(&v2), Some((128, 256, 1)));

        // LERC2 v3: 10..14 checksum, 14..18 height, 18..22 width (depth=1)
        let mut v3 = Vec::new();
        v3.extend_from_slice(b"Lerc2 ");
        v3.extend_from_slice(&3i32.to_le_bytes());
        v3.extend_from_slice(&0xDEAD_BEEFu32.to_le_bytes());
        v3.extend_from_slice(&64u32.to_le_bytes()); // height
        v3.extend_from_slice(&512u32.to_le_bytes()); // width
        v3.extend_from_slice(&32768u32.to_le_bytes());
        assert_eq!(parse_lerc_header_dims(&v3), Some((512, 64, 1)));

        // LERC2 v4: 10..14 checksum, 14..18 height, 18..22 width, 22..26 depth
        let mut v4 = Vec::new();
        v4.extend_from_slice(b"Lerc2 ");
        v4.extend_from_slice(&4i32.to_le_bytes());
        v4.extend_from_slice(&0u32.to_le_bytes());
        v4.extend_from_slice(&1u32.to_le_bytes()); // partial strip height=1
        v4.extend_from_slice(&256u32.to_le_bytes()); // width=256
        v4.extend_from_slice(&3u32.to_le_bytes()); // depth=3 RGB
        assert_eq!(parse_lerc_header_dims(&v4), Some((256, 1, 3)));

        // Crafted LERC2 v4 OOM attempt: height = 100_000_000, width = 1, depth = 1
        let mut v4_oom = Vec::new();
        v4_oom.extend_from_slice(b"Lerc2 ");
        v4_oom.extend_from_slice(&4i32.to_le_bytes());
        v4_oom.extend_from_slice(&0u32.to_le_bytes());
        v4_oom.extend_from_slice(&100_000_000u32.to_le_bytes());
        v4_oom.extend_from_slice(&1u32.to_le_bytes());
        v4_oom.extend_from_slice(&1u32.to_le_bytes());
        assert_eq!(parse_lerc_header_dims(&v4_oom), Some((1, 100_000_000, 1)));

        // LERC1 with 9-byte prefix + NUL (no trailing space)
        let mut v1 = Vec::new();
        v1.extend_from_slice(b"CntZImage\0");
        v1.extend_from_slice(&11i32.to_le_bytes()); // version
        v1.extend_from_slice(&8i32.to_le_bytes()); // image_type
        v1.extend_from_slice(&1024u32.to_le_bytes()); // height
        v1.extend_from_slice(&2048u32.to_le_bytes()); // width
        assert_eq!(parse_lerc_header_dims(&v1), Some((2048, 1024, 1)));
    }

    #[test]
    fn decompress_gzip_limited_handles_small_cap_and_hostile_isize_trailer() {
        // Must not panic when max_out_bytes < 4096 or == 0.
        let gz = compress_gzip(b"hello cogzilla").unwrap();
        assert_eq!(decompress_gzip_limited(&gz, 64).unwrap(), b"hello cogzilla");
        assert!(decompress_gzip_limited(&gz, 4).is_err());
        assert!(decompress_gzip_limited(&gz, 0).is_err());
        // Hostile 18-byte gzip header with 0xFFFF_FFFF ISIZE trailer must fail cleanly without OOM.
        let mut hostile = [0u8; 18];
        hostile[0] = 0x1f;
        hostile[1] = 0x8b;
        hostile[14..18].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(decompress_gzip_limited(&hostile, 64 * 1024 * 1024).is_err());
    }

    #[test]
    fn compress_and_decompress_brotli_roundtrip_and_limit() {
        let payload = b"cogzilla brotli roundtrip test payload ".repeat(64);
        let br = compress_brotli(&payload).unwrap();
        assert!(br.len() < payload.len());
        assert_eq!(decompress_brotli_limited(&br, payload.len()).unwrap(), payload);
        assert!(decompress_brotli_limited(&br, 16).is_err());
        assert!(decompress_brotli_limited(&br, 0).is_err());
    }

    #[test]
    fn inflate_into_zstd_into_and_scratch_reuse_roundtrip_and_bounds() {
        let payload: Vec<u8> = (0..1024u32).flat_map(|v| v.to_le_bytes()).collect();
        let gz = compress_gzip(&payload).unwrap();
        let zl = compress_zlib(&payload).unwrap();
        let zs = compress_zstd(&payload, 3).unwrap();

        with_codec_scratch(|scratch| {
            let dst = scratch.resize_zeroed(payload.len());
            let n_gz = inflate_into(&gz, dst, DeflateWrapper::Gzip).unwrap();
            assert_eq!(&dst[..n_gz], payload.as_slice());

            let dst = scratch.resize_zeroed(payload.len());
            let n_zl = inflate_into(&zl, dst, DeflateWrapper::Zlib).unwrap();
            assert_eq!(&dst[..n_zl], payload.as_slice());

            let dst = scratch.resize_zeroed(payload.len());
            let n_zs = zstd_into(&zs, dst).unwrap();
            assert_eq!(&dst[..n_zs], payload.as_slice());

            // Oversized frame must be refused before decompressing without over-allocating.
            let tiny = scratch.resize_zeroed(16);
            assert!(matches!(
                zstd_into(&zs, tiny),
                Err(CodecError::SizeMismatch { expected: 16, got }) if got == payload.len()
            ));
            assert!(inflate_into(&gz, tiny, DeflateWrapper::Gzip).is_err());
            assert!(inflate_into(&zl, tiny, DeflateWrapper::Zlib).is_err());

            // Truncated/corrupt zstd headers report ZstdHeader, not Deflate.
            assert!(matches!(
                zstd_into(&[0, 1], tiny),
                Err(CodecError::ZstdHeader(_))
            ));
            assert!(matches!(
                zstd_into(&[0, 0, 0, 0], tiny),
                Err(CodecError::ZstdHeader(_))
            ));

            // Re-entrant with_codec_scratch falls back safely without panicking.
            with_codec_scratch(|inner| {
                let d = inner.resize_zeroed(payload.len());
                assert_eq!(zstd_into(&zs, d).unwrap(), payload.len());
            });
        });

        // Oversized (>4 MiB) scratch allocation is shrunk back to 1 MiB on drop, and cleared on entry.
        with_codec_scratch(|scratch| {
            let _ = scratch.resize_zeroed(5 * 1024 * 1024);
        });
        with_codec_scratch(|scratch| {
            assert!(scratch.as_vec_mut().is_empty());
            assert!(scratch.as_vec_mut().capacity() <= 4 * 1024 * 1024);
        });

        // Direct owned-vec helpers roundtrip and reject hostile headers before allocating.
        assert_eq!(inflate_to_vec(&gz, payload.len(), DeflateWrapper::Gzip).unwrap(), payload);
        assert_eq!(inflate_to_vec(&zl, payload.len(), DeflateWrapper::Zlib).unwrap(), payload);
        assert_eq!(zstd_to_vec(&zs, payload.len()).unwrap(), payload);
        assert!(matches!(
            zstd_to_vec(&[0, 1], 128 * 1024 * 1024),
            Err(CodecError::ZstdHeader(_))
        ));
    }
}
