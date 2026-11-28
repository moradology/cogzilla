//! The one uncompressed Classic (little-endian) GeoTIFF writer behind every `.tif` response.
//!
//! [`write_geotiff`] owns the file layout: the TIFF header, a single IFD with its entries in
//! ascending tag order, the out-of-line tag arrays, the georeferencing tail (`ModelPixelScale`,
//! `ModelTiepoint`, `GeoKeyDirectory`, optional `GDAL_NODATA`) and one strip (chunky) or one strip
//! per band (planar) of pixel bytes. Callers decide what the samples mean and produce them.

use bytes::{BufMut, Bytes, BytesMut};

/// Sample type of every band (`BitsPerSample` + `SampleFormat`). Samples are little-endian.
// The integer and float64 types are for raw-array GeoTIFFs; this crate's own encoders write
// `U8` and `F32` only.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GeoTiffSample {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    U64,
    I64,
    F32,
    F64,
}

impl GeoTiffSample {
    /// `BitsPerSample` (tag 258).
    pub(crate) const fn bits(self) -> u16 {
        match self {
            Self::U8 | Self::I8 => 8,
            Self::U16 | Self::I16 => 16,
            Self::U32 | Self::I32 | Self::F32 => 32,
            Self::U64 | Self::I64 | Self::F64 => 64,
        }
    }

    /// `SampleFormat` (tag 339): 1 unsigned integer, 2 signed integer, 3 IEEE float.
    pub(crate) const fn format(self) -> u16 {
        match self {
            Self::U8 | Self::U16 | Self::U32 | Self::U64 => 1,
            Self::I8 | Self::I16 | Self::I32 | Self::I64 => 2,
            Self::F32 | Self::F64 => 3,
        }
    }
}

/// How the pixel bytes handed to [`write_geotiff`] are ordered.
#[allow(dead_code)] // `Planar` is for band-major raw arrays; this crate's own encoders interleave.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GeoTiffLayout {
    /// Pixel-interleaved (`b0 b1 .. bn` per pixel) in one strip: `PlanarConfiguration = 1`.
    Chunky,
    /// Band-major (all of band 0, then band 1, ..), one strip per band: `PlanarConfiguration = 2`.
    /// A single band is the same bytes either way and is written as `PlanarConfiguration = 1`
    /// with its one strip referenced inline.
    Planar,
}

/// `PhotometricInterpretation` (tag 262).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GeoTiffPhotometric {
    /// First band is gray; every further band must be described by `extra_samples`.
    MinIsBlack = 1,
    /// First three bands are red, green, blue; every further band must be described by
    /// `extra_samples`.
    Rgb = 2,
}

/// Everything about a GeoTIFF except its pixel bytes.
#[derive(Clone, Copy, Debug)]
pub(crate) struct GeoTiffSpec<'a> {
    /// `ImageWidth`, at least 1. Not clamped here.
    pub width: u32,
    /// `ImageLength`, at least 1. Not clamped here.
    pub height: u32,
    /// `SamplesPerPixel`, at least 1, including any mask / alpha band.
    pub bands: u16,
    /// Type of every band.
    pub sample: GeoTiffSample,
    /// Write the `SampleFormat` tag (339). Readers default to unsigned integer without it, so
    /// only `U8` / `U16` / `U32` may leave it out.
    pub sample_format_tag: bool,
    pub layout: GeoTiffLayout,
    pub photometric: GeoTiffPhotometric,
    /// `ExtraSamples` (tag 338), one value per band beyond the photometric ones: `0` unspecified
    /// data, `1` associated alpha, `2` unassociated alpha (a validity mask). Empty writes no tag.
    pub extra_samples: &'a [u16],
    /// `GDAL_NODATA` (tag 42113) text without the trailing NUL, e.g. `"nan"`. `None` writes no tag.
    pub nodata: Option<&'a str>,
    /// `[min_x, min_y, max_x, max_y]` of the image in `epsg` units. A geographic bbox whose
    /// `max_x < min_x` crosses the antimeridian and is unwrapped for the pixel scale.
    pub bbox: [f64; 4],
    /// EPSG code for the GeoKey directory: written as `GeographicTypeGeoKey` when
    /// `cogzilla_proj` knows the code as geographic, as `ProjectedCSTypeGeoKey` otherwise.
    pub epsg: u16,
}

impl GeoTiffSpec<'_> {
    /// Exact number of pixel bytes the file holds:
    /// `width * height * bands * bytes-per-sample`.
    pub(crate) fn pixel_bytes(&self) -> usize {
        (self.width as usize)
            * (self.height as usize)
            * (self.bands as usize)
            * (self.sample.bits() as usize / 8)
    }
}

const TIFF_ASCII: u16 = 2;
const TIFF_SHORT: u16 = 3;
const TIFF_LONG: u16 = 4;
const TIFF_DOUBLE: u16 = 12;

/// Bytes a SHORT array of `count` values takes outside its IFD entry (up to two fit inside).
const fn shorts_out_of_line(count: usize) -> u32 {
    if count > 2 { (count as u32) * 2 } else { 0 }
}

/// Value field of a SHORT array entry: the values themselves when they fit, else their offset.
fn shorts_value(count: usize, at: u32, value: impl Fn(usize) -> u16) -> u32 {
    match count {
        0 => 0,
        1 => value(0) as u32,
        2 => (value(0) as u32) | ((value(1) as u32) << 16),
        _ => at,
    }
}

/// Write an uncompressed Classic GeoTIFF described by `spec`.
///
/// `write_pixels` appends the pixel bytes straight into the output buffer (already sized for the
/// whole file), in `spec.layout` order: exactly [`GeoTiffSpec::pixel_bytes`] of them. If it
/// appends fewer the strip is zero-filled; anything beyond is cut off, so the file is always
/// structurally valid.
///
/// Classic TIFF offsets are 32-bit: the caller keeps the file under 4 GiB.
pub(crate) fn write_geotiff(
    spec: &GeoTiffSpec<'_>,
    write_pixels: impl FnOnce(&mut BytesMut),
) -> Bytes {
    let is_geographic = cogzilla_proj::CrsModel::from_epsg(spec.epsg)
        .is_ok_and(|c| c.is_geographic());
    let bbox = spec.bbox;
    // ModelPixelScaleTag: [dx, dy, 0.0] (unwrap antimeridian-crossing geographic bboxes w > e)
    let span_x = if is_geographic && bbox[2] < bbox[0] {
        (bbox[2] + 360.0) - bbox[0]
    } else {
        (bbox[2] - bbox[0]).abs()
    };
    let dx = (span_x / (spec.width.max(1) as f64)).max(1e-12);
    let dy = ((bbox[3] - bbox[1]).abs() / (spec.height.max(1) as f64)).max(1e-12);
    write_geotiff_placed(spec, [dx, dy], [bbox[0], bbox[3]], write_pixels)
}

/// [`write_geotiff`] with its pixel size and the corner of its first pixel given as they are
/// written (`ModelPixelScale`, `ModelTiepoint`), for a file rewritten from one of its own.
/// `spec.bbox` is not read.
fn write_geotiff_placed(
    spec: &GeoTiffSpec<'_>,
    pixel_size: [f64; 2],
    corner: [f64; 2],
    write_pixels: impl FnOnce(&mut BytesMut),
) -> Bytes {
    let bands = spec.bands as usize;
    let pixel_bytes = spec.pixel_bytes();
    let planar = spec.layout == GeoTiffLayout::Planar && bands > 1;
    let n_strips = if planar { bands } else { 1 };
    let strip_bytes = pixel_bytes / n_strips;
    let n_extra = spec.extra_samples.len();
    // ASCII count includes the terminating NUL; up to four bytes live in the entry itself.
    let nodata_count = spec.nodata.map_or(0, |s| s.len() + 1);
    debug_assert!(spec.width >= 1 && spec.height >= 1 && bands >= 1);
    debug_assert!(spec.sample_format_tag || spec.sample.format() == 1);

    let n_entries: u16 = 13
        + u16::from(n_extra > 0)
        + u16::from(spec.sample_format_tag)
        + u16::from(spec.nodata.is_some());

    // Layout:
    // 0..8      : TIFF header ("II", 42, ifd_offset = 8)
    // 8..       : IFD (count, 12-byte entries, next-IFD = 0)
    // then      : out-of-line BitsPerSample / ExtraSamples / SampleFormat (3+ values each),
    //             StripOffsets + StripByteCounts (planar), ModelPixelScale (24),
    //             ModelTiepoint (48), GeoKeyDirectory (32), GDAL_NODATA (5+ bytes, even-padded)
    // pixel_off : strip data
    let mut next = (8 + 2 + 12 * (n_entries as usize) + 4) as u32;
    let mut place = |len: u32| {
        let at = next;
        next += len;
        at
    };
    let bps_off = place(shorts_out_of_line(bands));
    let extra_off = place(shorts_out_of_line(n_extra));
    let sf_off = place(if spec.sample_format_tag { shorts_out_of_line(bands) } else { 0 });
    let strips_off = place(if planar { 8 * bands as u32 } else { 0 });
    let scale_off = place(24);
    let tie_off = place(48);
    let gkd_off = place(32);
    let nodata_off =
        place(if nodata_count > 4 { (nodata_count as u32).next_multiple_of(2) } else { 0 });
    let pixel_off = next;
    let file_len = (pixel_off as usize) + pixel_bytes;

    let mut buf = BytesMut::with_capacity(file_len);
    // TIFF little-endian header
    buf.put_slice(b"II");
    buf.put_u16_le(42);
    buf.put_u32_le(8);

    // IFD entries, in strictly ascending tag order
    buf.put_u16_le(n_entries);
    let mut put_entry = |tag: u16, ty: u16, count: u32, val: u32| {
        buf.put_u16_le(tag);
        buf.put_u16_le(ty);
        buf.put_u32_le(count);
        buf.put_u32_le(val);
    };
    let bits = spec.sample.bits();
    let format = spec.sample.format();
    let (strip_offsets_val, strip_counts_val) = if planar {
        (strips_off, strips_off + 4 * bands as u32)
    } else {
        (pixel_off, strip_bytes as u32)
    };
    put_entry(256, TIFF_LONG, 1, spec.width);
    put_entry(257, TIFF_LONG, 1, spec.height);
    put_entry(258, TIFF_SHORT, bands as u32, shorts_value(bands, bps_off, |_| bits)); // BitsPerSample
    put_entry(259, TIFF_SHORT, 1, 1); // Compression = None
    put_entry(262, TIFF_SHORT, 1, spec.photometric as u32); // PhotometricInterpretation
    put_entry(273, TIFF_LONG, n_strips as u32, strip_offsets_val); // StripOffsets
    put_entry(277, TIFF_SHORT, 1, bands as u32); // SamplesPerPixel
    put_entry(278, TIFF_LONG, 1, spec.height); // RowsPerStrip
    put_entry(279, TIFF_LONG, n_strips as u32, strip_counts_val); // StripByteCounts
    put_entry(284, TIFF_SHORT, 1, if planar { 2 } else { 1 }); // PlanarConfiguration
    if n_extra > 0 {
        let extra = spec.extra_samples;
        put_entry(338, TIFF_SHORT, n_extra as u32, shorts_value(n_extra, extra_off, |i| extra[i])); // ExtraSamples
    }
    if spec.sample_format_tag {
        put_entry(339, TIFF_SHORT, bands as u32, shorts_value(bands, sf_off, |_| format)); // SampleFormat
    }
    put_entry(33550, TIFF_DOUBLE, 3, scale_off); // ModelPixelScaleTag
    put_entry(33922, TIFF_DOUBLE, 6, tie_off); // ModelTiepointTag
    put_entry(34735, TIFF_SHORT, 16, gkd_off); // GeoKeyDirectoryTag
    if let Some(nodata) = spec.nodata {
        let val = if nodata_count > 4 {
            nodata_off
        } else {
            let mut inline = [0u8; 4];
            inline[..nodata.len()].copy_from_slice(nodata.as_bytes());
            u32::from_le_bytes(inline)
        };
        put_entry(42113, TIFF_ASCII, nodata_count as u32, val); // GDAL_NODATA
    }
    buf.put_u32_le(0); // Next IFD = 0

    // Out-of-line SHORT arrays
    if bands > 2 {
        for _ in 0..bands {
            buf.put_u16_le(bits);
        }
    }
    if n_extra > 2 {
        for &es in spec.extra_samples {
            buf.put_u16_le(es);
        }
    }
    if spec.sample_format_tag && bands > 2 {
        for _ in 0..bands {
            buf.put_u16_le(format);
        }
    }
    // One strip per band
    if planar {
        for band in 0..bands {
            buf.put_u32_le(pixel_off + (band * strip_bytes) as u32);
        }
        for _ in 0..bands {
            buf.put_u32_le(strip_bytes as u32);
        }
    }

    let is_geographic = cogzilla_proj::CrsModel::from_epsg(spec.epsg)
        .is_ok_and(|c| c.is_geographic());
    buf.put_f64_le(pixel_size[0]);
    buf.put_f64_le(pixel_size[1]);
    buf.put_f64_le(0.0);
    // ModelTiepointTag: [0, 0, 0, minx, maxy, 0]
    buf.put_f64_le(0.0);
    buf.put_f64_le(0.0);
    buf.put_f64_le(0.0);
    buf.put_f64_le(corner[0]);
    buf.put_f64_le(corner[1]);
    buf.put_f64_le(0.0);
    // GeoKeyDirectoryTag (16 u16s = 32 bytes)
    let (model_type, crs_key): (u16, u16) = if is_geographic {
        (2, 2048) // ModelTypeGeographic, GeographicTypeGeoKey
    } else {
        (1, 3072) // ModelTypeProjected, ProjectedCSTypeGeoKey
    };
    #[rustfmt::skip]
    let gkd: [u16; 16] = [
        1, 1, 0, 3, // version 1.1.0, 3 keys
        1024, 0, 1, model_type, // GTModelTypeGeoKey
        1025, 0, 1, 1, // GTRasterTypeGeoKey = PixelIsArea
        crs_key, 0, 1, spec.epsg, // EPSG code
    ];
    for &k in &gkd {
        buf.put_u16_le(k);
    }
    if let Some(nodata) = spec.nodata
        && nodata_count > 4
    {
        buf.put_slice(nodata.as_bytes());
        buf.resize(pixel_off as usize, 0); // NUL terminator and even padding
    }
    debug_assert_eq!(buf.len(), pixel_off as usize);

    write_pixels(&mut buf);
    debug_assert!(buf.len() <= file_len, "pixel writer overran the strip data");
    buf.resize(file_len, 0);
    buf.freeze()
}

/// The sample type of a raw-array GeoTIFF of `kind`.
pub(crate) fn geotiff_sample(kind: cogzilla_raster::SampleType) -> GeoTiffSample {
    use cogzilla_raster::SampleType as T;
    match kind {
        T::Bool | T::Uint1 | T::Uint4 | T::Uint8 => GeoTiffSample::U8,
        T::Int8 => GeoTiffSample::I8,
        T::Uint12 | T::Uint16 => GeoTiffSample::U16,
        T::Int16 => GeoTiffSample::I16,
        T::Uint32 => GeoTiffSample::U32,
        T::Int32 => GeoTiffSample::I32,
        T::Uint64 => GeoTiffSample::U64,
        T::Int64 => GeoTiffSample::I64,
        T::Float16 | T::Float32 => GeoTiffSample::F32,
        T::Float64 => GeoTiffSample::F64,
    }
}

/// A float32 GeoTIFF that [`write_geotiff`] made, rewritten in the sample type of `kind`, or
/// `None` when `tif` is not such a file. Size, layout, pixel size, corner and EPSG code are
/// kept; photometric interpretation and `ExtraSamples` are derived from `(data_bands, kind,
/// mask)` to match GDAL (only 3-band `Uint8` uses `RGB`, all others use `MinIsBlack`). The
/// `GDAL_NODATA` of `nan` goes, as TiTiler writes none. Values are cast as
/// `cogzilla_kernel::retype_f32_npy` casts them; with `mask`, the last band is the mask,
/// written as the type's largest number where it is not zero and its smallest where it is.
pub(crate) fn retype_f32_geotiff(tif: &[u8], kind: cogzilla_raster::SampleType, mask: bool) -> Option<Bytes> {
    retype_f32_geotiff_with(tif, kind, mask, |v| v)
}

/// [`retype_f32_geotiff`], each sample (not the mask) passed through `prepare` before it is
/// cast, as `cogzilla_kernel::retype_f32_npy_with` does.
pub(crate) fn retype_f32_geotiff_with(
    tif: &[u8],
    kind: cogzilla_raster::SampleType,
    mask: bool,
    prepare: impl Fn(f64) -> f64,
) -> Option<Bytes> {
    let u16_at = |at: usize| tif.get(at..at + 2).map(|b| u16::from_le_bytes([b[0], b[1]]));
    let u32_at = |at: usize| tif.get(at..at + 4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]));
    let f64_at = |at: usize| tif.get(at..at + 8).map(|b| f64::from_le_bytes(b.try_into().unwrap()));
    if tif.get(..4)? != b"II*\0" || u32_at(4)? != 8 {
        return None;
    }
    // Tag -> (count, the value or offset field's position).
    let n = usize::from(u16_at(8)?);
    let mut tags = std::collections::HashMap::with_capacity(n);
    for e in 0..n {
        let at = 10 + 12 * e;
        tags.insert(u16_at(at)?, (u32_at(at + 4)? as usize, at + 8));
    }
    let long = |tag: u16| tags.get(&tag).and_then(|&(_, at)| u32_at(at));
    let short = |tag: u16| tags.get(&tag).and_then(|&(_, at)| u16_at(at));
    // SHORT arrays: inline up to two values, else at the offset.
    let shorts = |tag: u16| -> Option<Vec<u16>> {
        let &(count, at) = tags.get(&tag)?;
        let from = if count > 2 { u32_at(at)? as usize } else { at };
        (0..count).map(|i| u16_at(from + 2 * i)).collect()
    };
    let (width, height, bands) = (long(256)?, long(257)?, short(277)?);
    let bands_us = usize::from(bands);
    if width == 0 || height == 0 || bands == 0 || (mask && bands < 2) {
        return None;
    }
    let bps = shorts(258)?;
    let sf = shorts(339)?;
    let src_bps = *bps.first()?;
    let is_f32 = src_bps == 32 && bps.iter().all(|b| *b == 32);
    let is_f64 = src_bps == 64 && bps.iter().all(|b| *b == 64);
    if bps.len() != bands_us
        || sf.len() != bands_us
        || (!is_f32 && !is_f64)
        || sf.iter().any(|f| *f != 3)
        || short(259)? != 1
    {
        return None;
    }
    let src_bytes_per_sample = if is_f64 { 8usize } else { 4usize };
    // Validate input photometric tag (1 = MinIsBlack, 2 = Rgb), then derive the output
    // photometric & ExtraSamples from the target dtype matching GDAL: only 3-band uint8 is RGB.
    match short(262)? {
        1 | 2 => {}
        _ => return None,
    }
    let data_bands = bands - u16::from(mask);
    let is_rgb = data_bands == 3 && matches!(kind, cogzilla_raster::SampleType::Uint8);
    let photometric = if is_rgb {
        GeoTiffPhotometric::Rgb
    } else {
        GeoTiffPhotometric::MinIsBlack
    };
    let photo_bands = if is_rgb { 3u16 } else { 1u16 };
    let extra: Vec<u16> = (photo_bands..bands)
        .map(|b| if mask && b + 1 == bands { 2 } else { 0 })
        .collect();
    let planar = short(284)? == 2;
    let scale_at = long(33550)? as usize;
    let tie_at = long(33922)? as usize;
    let keys_at = long(34735)? as usize;
    let epsg = u16_at(keys_at + 2 * 15)?;
    let (pixel_size, corner) = ([f64_at(scale_at)?, f64_at(scale_at + 8)?], [f64_at(tie_at + 24)?, f64_at(tie_at + 32)?]);
    let pixels = (width as usize).checked_mul(height as usize)?;
    let in_bytes = bands_us.checked_mul(pixels)?.checked_mul(src_bytes_per_sample)?;
    let out_bytes = bands_us.checked_mul(pixels)?.checked_mul(kind.bytes_per_sample())?;
    // Classic TIFF uses 32-bit strip offsets and byte counts.
    if out_bytes > (u32::MAX as usize) - 4096 {
        return None;
    }
    let start = long(273).filter(|_| !planar).or_else(|| u32_at(long(273)? as usize))? as usize;
    let samples = tif.get(start..start.checked_add(in_bytes)?)?;
    let layout = if planar { GeoTiffLayout::Planar } else { GeoTiffLayout::Chunky };
    let spec = GeoTiffSpec {
        width,
        height,
        bands,
        sample: geotiff_sample(kind),
        sample_format_tag: true,
        layout,
        photometric,
        extra_samples: &extra,
        nodata: None,
        bbox: [0.0; 4],
        epsg,
    };
    // Which band a sample belongs to, from its place among the samples.
    let band_of = |i: usize| if planar { i / pixels } else { i % bands_us };
    let is_mask = |i: usize| mask && band_of(i) + 1 == bands_us;
    let mask_index_of = |i: usize| {
        if planar {
            (bands_us - 1) * pixels + (i % pixels)
        } else {
            (i / bands_us) * bands_us + (bands_us - 1)
        }
    };
    Some(write_geotiff_placed(&spec, pixel_size, corner, |buf| {
        let total_samples = bands_us * pixels;
        let read_sample = |i: usize| -> f64 {
            if is_f64 {
                let off = i * 8;
                f64::from_le_bytes(samples[off..off + 8].try_into().unwrap())
            } else {
                let off = i * 4;
                f64::from(f32::from_le_bytes([
                    samples[off],
                    samples[off + 1],
                    samples[off + 2],
                    samples[off + 3],
                ]))
            }
        };
        macro_rules! put {
            ($t:ty, $cast:expr) => {{
                let cast = $cast;
                for i in 0..total_samples {
                    let v = read_sample(i);
                    let x: $t = if !is_mask(i) {
                        let mask_sample = (mask && v.is_nan()).then(|| read_sample(mask_index_of(i)));
                        cast(prepare(cogzilla_kernel::retype_value_with_mask(v, mask_sample)))
                    } else if cogzilla_kernel::is_valid_mask_sample(v) {
                        <$t>::MAX
                    } else {
                        <$t>::MIN
                    };
                    buf.put_slice(&x.to_le_bytes());
                }
            }};
        }
        use cogzilla_raster::SampleType as T;
        match kind {
            T::Bool | T::Uint1 | T::Uint4 | T::Uint8 => put!(u8, |v: f64| v as u8),
            T::Int8 => put!(i8, |v: f64| v as i8),
            T::Uint12 | T::Uint16 => put!(u16, |v: f64| v as u16),
            T::Int16 => put!(i16, |v: f64| v as i16),
            T::Uint32 => put!(u32, |v: f64| v as u32),
            T::Int32 => put!(i32, |v: f64| v as i32),
            T::Uint64 => put!(u64, |v: f64| v as u64),
            T::Int64 => put!(i64, |v: f64| v as i64),
            T::Float16 | T::Float32 => put!(f32, |v: f64| v as f32),
            T::Float64 => put!(f64, |v: f64| v),
        }
    }))
}

/// Write a band-major CHW `f64` array directly as an uncompressed GeoTIFF in `kind`
/// (preserving exact `uint32`/`int32`/`int64` up to ±2^53 and 17-digit `float64` without `f32` quantization).
#[allow(clippy::too_many_arguments)]
pub(crate) fn write_geotiff_typed_f64(
    values: &[f64],
    bands: usize,
    width: u32,
    height: u32,
    kind: cogzilla_raster::SampleType,
    bbox: [f64; 4],
    epsg: u16,
    return_mask: bool,
) -> Bytes {
    let pixels = (width as usize) * (height as usize);
    let total_bands = (bands + usize::from(return_mask)) as u16;
    // Match GDAL: only 3-band uint8 defaults to Photometric=RGB; all other dtypes use MinIsBlack.
    let is_rgb = bands == 3 && matches!(kind, cogzilla_raster::SampleType::Uint8);
    let photometric = if is_rgb {
        GeoTiffPhotometric::Rgb
    } else {
        GeoTiffPhotometric::MinIsBlack
    };
    let photo_bands = if is_rgb { 3 } else { 1 };
    let mut extra: Vec<u16> = Vec::new();
    for b in photo_bands..bands {
        let _ = b;
        extra.push(0); // unspecified data band
    }
    if return_mask {
        extra.push(2); // unassociated alpha / validity mask
    }
    let spec = GeoTiffSpec {
        width,
        height,
        bands: total_bands,
        sample: geotiff_sample(kind),
        sample_format_tag: true,
        layout: GeoTiffLayout::Planar,
        photometric,
        extra_samples: &extra,
        nodata: None,
        bbox,
        epsg,
    };
    write_geotiff(&spec, |buf| {
        let data = &values[..(bands * pixels).min(values.len())];
        macro_rules! put_planes {
            ($t:ty, $cast:expr, $nan_to_zero:expr) => {{
                let cast = $cast;
                for &v in data {
                    let x: $t = cast(if $nan_to_zero && v.is_nan() { 0.0 } else { v });
                    buf.put_slice(&x.to_le_bytes());
                }
                if return_mask {
                    for i in 0..pixels {
                        let there = (0..bands)
                            .any(|b| data.get(b * pixels + i).is_some_and(|v| !v.is_nan()));
                        let x: $t = if there { <$t>::MAX } else { <$t>::MIN };
                        buf.put_slice(&x.to_le_bytes());
                    }
                }
            }};
        }
        use cogzilla_raster::SampleType as T;
        match kind {
            T::Bool | T::Uint1 | T::Uint4 | T::Uint8 => put_planes!(u8, |v: f64| v as u8, true),
            T::Int8 => put_planes!(i8, |v: f64| v as i8, true),
            T::Uint12 | T::Uint16 => put_planes!(u16, |v: f64| v as u16, true),
            T::Int16 => put_planes!(i16, |v: f64| v as i16, true),
            T::Uint32 => put_planes!(u32, |v: f64| v as u32, true),
            T::Int32 => put_planes!(i32, |v: f64| v as i32, true),
            T::Uint64 => put_planes!(u64, |v: f64| v as u64, true),
            T::Int64 => put_planes!(i64, |v: f64| v as i64, true),
            T::Float16 | T::Float32 => put_planes!(f32, |v: f64| v as f32, false),
            T::Float64 => put_planes!(f64, |v: f64| v, false),
        }
    })
}

/// Encode an RGBA buffer as a spec-compliant uncompressed Classic GeoTIFF (`EPSG:3857` or `EPSG:4326`).
pub fn encode_geotiff_rgba(
    width: u32,
    height: u32,
    rgba: &[u8],
    bbox: [f64; 4],
    epsg: u16,
    return_mask: bool,
) -> Bytes {
    let width = width.clamp(1, 4096);
    let height = height.clamp(1, 4096);
    let n_px = (width as usize) * (height as usize);
    let mut has_opaque = false;
    let mut all_gray = true;
    for px in rgba.chunks_exact(4).take(n_px) {
        if px[3] > 0 {
            has_opaque = true;
            if px[0] != px[1] || px[1] != px[2] {
                all_gray = false;
                break;
            }
        }
    }
    let is_gray = has_opaque && all_gray;
    let spp: u16 = if is_gray {
        if return_mask { 2 } else { 1 }
    } else if return_mask {
        4
    } else {
        3
    };
    let spec = GeoTiffSpec {
        width,
        height,
        bands: spp,
        sample: GeoTiffSample::U8,
        // 8-bit unsigned is the TIFF default, and GDAL's own byte GeoTIFFs carry no SampleFormat.
        sample_format_tag: false,
        layout: GeoTiffLayout::Chunky,
        photometric: if is_gray { GeoTiffPhotometric::MinIsBlack } else { GeoTiffPhotometric::Rgb },
        extra_samples: if return_mask { &[2] } else { &[] }, // UnassociatedAlpha
        nodata: None,
        bbox,
        epsg,
    };
    // A short `rgba` leaves the rest of the strip zero-filled by the writer.
    write_geotiff(&spec, |buf| {
        if is_gray {
            if return_mask {
                for px in rgba.chunks_exact(4).take(n_px) {
                    buf.put_u8(px[0]);
                    buf.put_u8(px[3]);
                }
            } else {
                for px in rgba.chunks_exact(4).take(n_px) {
                    buf.put_u8(px[0]);
                }
            }
        } else if return_mask {
            let take = (n_px * 4).min(rgba.len());
            buf.put_slice(&rgba[..take]);
        } else {
            for px in rgba.chunks_exact(4).take(n_px) {
                buf.put_u8(px[0]);
                buf.put_u8(px[1]);
                buf.put_u8(px[2]);
            }
        }
    })
}

/// Encode a CHW `f32` little-endian buffer as a spec-compliant uncompressed Classic IEEE Float32
/// GeoTIFF (`SampleFormat=3`, `BitsPerSample=32`, `GDAL_NODATA="nan"`). When `return_mask` is
/// `true`, appends an extra float32 validity mask band (`255.0` valid / `0.0` nodata) matching
/// TiTiler / `rio-tiler` GeoTIFF output.
pub fn encode_geotiff_f32(
    width: u32,
    height: u32,
    raw_f32_chw: &[u8],
    bbox: [f64; 4],
    epsg: u16,
    return_mask: bool,
) -> Bytes {
    let width = width.clamp(1, 4096);
    let height = height.clamp(1, 4096);
    let n_px = (width as usize) * (height as usize);
    let data_spp: u16 = if n_px > 0 && raw_f32_chw.len() >= n_px * 4 {
        (raw_f32_chw.len() / (n_px * 4)).clamp(1, 15) as u16
    } else {
        1
    };
    let spp: u16 = if return_mask { data_spp + 1 } else { data_spp };
    let pixel_bytes = n_px * (spp as usize) * 4;
    // Float32 GeoTIFFs always use MinIsBlack (GDAL only defaults to RGB for 3-band uint8).
    let photometric = GeoTiffPhotometric::MinIsBlack;
    let extra_samples_count: u16 = spp.saturating_sub(1);
    let mut extra_samples = Vec::with_capacity(extra_samples_count as usize);
    for idx in 0..extra_samples_count {
        let is_mask = return_mask && idx + 1 == extra_samples_count;
        // 2 = UnassociatedAlpha for the appended validity mask, 0 = Unspecified for extra data bands.
        extra_samples.push(if is_mask { 2u16 } else { 0u16 });
    }
    let spec = GeoTiffSpec {
        width,
        height,
        bands: spp,
        sample: GeoTiffSample::F32,
        sample_format_tag: true,
        layout: GeoTiffLayout::Chunky,
        photometric,
        extra_samples: &extra_samples,
        nodata: Some("nan"),
        bbox,
        epsg,
    };
    write_geotiff(&spec, |buf| {
        // Pixel data: interleave CHW f32 planes into Chunky f32 (plus optional validity mask band)
        if spp == 1 && !return_mask {
            let take = (pixel_bytes.min(raw_f32_chw.len()) / 4) * 4;
            buf.put_slice(&raw_f32_chw[..take]);
            if take < pixel_bytes {
                let rem_f32 = (pixel_bytes - take) / 4;
                for _ in 0..rem_f32 {
                    buf.put_f32_le(f32::NAN);
                }
            }
        } else {
            let data_spp_us = data_spp as usize;
            let plane_bytes = n_px * 4;
            let valid_le = 255.0f32.to_le_bytes();
            let zero_le = 0.0f32.to_le_bytes();
            for px in 0..n_px {
                let px_off = px * 4;
                let mut any_valid = false;
                for b in 0..data_spp_us {
                    let off = b * plane_bytes + px_off;
                    if off + 4 <= raw_f32_chw.len() {
                        let sl = &raw_f32_chw[off..off + 4];
                        if return_mask
                            && !any_valid
                            && f32::from_le_bytes([sl[0], sl[1], sl[2], sl[3]]).is_finite()
                        {
                            any_valid = true;
                        }
                        buf.put_slice(sl);
                    } else {
                        buf.put_f32_le(f32::NAN);
                    }
                }
                if return_mask {
                    buf.put_slice(if any_valid { &valid_le } else { &zero_le });
                }
            }
        }
    })
}


#[cfg(test)]
mod golden_tests {
    //! Byte-exact pins of `encode_geotiff_rgba` / `encode_geotiff_f32` output: every case records
    //! the exact length and a 128-bit FNV-1a digest of the encoded file.

    use super::{encode_geotiff_f32, encode_geotiff_rgba};
    use bytes::Bytes;

    
    const MERC: [f64; 4] = [-20037508.342789244, -7.081154551613622e-10, 0.0, 20037508.342789244];
    const UTM: [f64; 4] = [399960.0, 4890240.0, 509760.0, 5000040.0];
    const GEO: [f64; 4] = [8.25, 45.5, 9.75, 46.125];
    const ANTIMERIDIAN: [f64; 4] = [170.0, -10.5, -170.0, 10.25];

    fn fnv1a_128(bytes: &[u8]) -> u128 {
        let mut hash: u128 = 0x6c62272e07bb014262b821756295c58d;
        for &b in bytes {
            hash ^= u128::from(b);
            hash = hash.wrapping_mul(0x0000000001000000000000000000013b);
        }
        hash
    }

    /// Deterministic byte stream (64-bit LCG, high byte of each step).
    fn noise(seed: u64, len: usize) -> Vec<u8> {
        let mut state = seed.wrapping_mul(0x9e3779b97f4a7c15).wrapping_add(1);
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (state >> 56) as u8
            })
            .collect()
    }

    #[derive(Clone, Copy)]
    enum Alpha {
        Opaque,
        Mixed,
        Clear,
    }

    fn rgba(seed: u64, w: usize, h: usize, gray: bool, alpha: Alpha) -> Vec<u8> {
        let mut px = noise(seed, w * h * 4);
        for (i, p) in px.chunks_exact_mut(4).enumerate() {
            if gray {
                p[1] = p[0];
                p[2] = p[0];
            }
            p[3] = match alpha {
                Alpha::Opaque => 255,
                Alpha::Mixed => [0, 255, 0, 128, 255][i % 5],
                Alpha::Clear => 0,
            };
        }
        px
    }

    /// Band-major little-endian `f32` planes; with `holes`, some pixels are NaN in every band
    /// (mask 0), some in one band only (mask stays valid), and one sample is infinite.
    fn f32_chw(seed: u64, w: usize, h: usize, bands: usize, holes: bool) -> Vec<u8> {
        let n_px = w * h;
        let raw = noise(seed, n_px * bands * 2);
        let mut out = Vec::with_capacity(n_px * bands * 4);
        for b in 0..bands {
            for px in 0..n_px {
                let i = (b * n_px + px) * 2;
                let mut v = f32::from(u16::from_le_bytes([raw[i], raw[i + 1]])) / 97.0 - 300.0;
                if holes {
                    if px % 4 == 1 || (px % 7 == 3 && b == 0) {
                        v = f32::NAN;
                    } else if px % 11 == 5 && b + 1 == bands {
                        v = f32::INFINITY;
                    }
                }
                out.extend_from_slice(&v.to_le_bytes());
            }
        }
        out
    }

    fn cases() -> Vec<(&'static str, Bytes)> {
        let mut out: Vec<(&'static str, Bytes)> = Vec::new();
        let mut rgba_case = |name, w: u32, h: u32, px: &[u8], bbox, epsg, mask| {
            out.push((name, encode_geotiff_rgba(w, h, px, bbox, epsg, mask)));
        };
        let colour = rgba(1, 8, 8, false, Alpha::Opaque);
        rgba_case("rgba colour opaque 8x8 3857", 8, 8, &colour, MERC, 3857, false);
        rgba_case("rgba colour opaque 8x8 3857 mask", 8, 8, &colour, MERC, 3857, true);
        let colour_holes = rgba(2, 8, 8, false, Alpha::Mixed);
        rgba_case("rgba colour transparent 8x8 3857", 8, 8, &colour_holes, MERC, 3857, false);
        rgba_case("rgba colour transparent 8x8 3857 mask", 8, 8, &colour_holes, MERC, 3857, true);
        let gray = rgba(3, 8, 8, true, Alpha::Opaque);
        rgba_case("rgba gray opaque 8x8 4326", 8, 8, &gray, GEO, 4326, false);
        rgba_case("rgba gray opaque 8x8 4326 mask", 8, 8, &gray, GEO, 4326, true);
        // Gray detection only looks at pixels with alpha > 0: colour under transparent pixels.
        let mut gray_holes = rgba(4, 8, 8, true, Alpha::Mixed);
        for p in gray_holes.chunks_exact_mut(4).filter(|p| p[3] == 0) {
            p[1] = p[0].wrapping_add(17);
        }
        rgba_case("rgba gray transparent 8x8 4326", 8, 8, &gray_holes, GEO, 4326, false);
        rgba_case("rgba gray transparent 8x8 4326 mask", 8, 8, &gray_holes, GEO, 4326, true);
        let clear = rgba(5, 8, 8, true, Alpha::Clear);
        rgba_case("rgba all transparent 8x8 3857", 8, 8, &clear, MERC, 3857, false);
        rgba_case("rgba all transparent 8x8 3857 mask", 8, 8, &clear, MERC, 3857, true);
        let wide = rgba(6, 7, 3, false, Alpha::Mixed);
        rgba_case("rgba colour 7x3 4326", 7, 3, &wide, GEO, 4326, false);
        rgba_case("rgba colour 7x3 4326 mask", 7, 3, &wide, GEO, 4326, true);
        rgba_case("rgba colour 7x3 antimeridian 4326 mask", 7, 3, &wide, ANTIMERIDIAN, 4326, true);
        rgba_case("rgba colour 7x3 antimeridian 3857 mask", 7, 3, &wide, ANTIMERIDIAN, 3857, true);
        rgba_case("rgba colour 7x3 32633", 7, 3, &wide, UTM, 32633, false);
        rgba_case("rgba colour 7x3 4269 geographic", 7, 3, &wide, GEO, 4269, false);
        rgba_case("rgba colour 7x3 unknown epsg", 7, 3, &wide, GEO, 1, true);
        let tall = rgba(7, 3, 9, true, Alpha::Mixed);
        rgba_case("rgba gray 3x9 3857 mask", 3, 9, &tall, MERC, 3857, true);
        // Buffer shorter than width*height*4: the strip is zero-padded.
        rgba_case("rgba colour short buffer", 8, 8, &colour[..100], MERC, 3857, false);
        rgba_case("rgba colour short buffer mask", 8, 8, &colour_holes[..101], MERC, 3857, true);
        rgba_case("rgba gray short buffer", 8, 8, &gray[..100], MERC, 3857, false);
        rgba_case("rgba gray short buffer mask", 8, 8, &gray[..102], MERC, 3857, true);
        // Buffer longer than width*height*4: the tail is ignored.
        rgba_case("rgba colour long buffer", 4, 8, &colour, MERC, 3857, false);
        rgba_case("rgba colour long buffer mask", 4, 8, &colour, MERC, 3857, true);
        rgba_case("rgba empty buffer", 2, 2, &[], MERC, 3857, true);
        // Dimensions clamp to 1..=4096.
        rgba_case("rgba zero width", 0, 4, &colour, MERC, 3857, false);
        rgba_case("rgba zero size mask", 0, 0, &colour, GEO, 4326, true);
        rgba_case("rgba 5000x2 clamps to 4096", 5000, 2, &colour, MERC, 3857, true);
        rgba_case("rgba 2x5000 clamps to 4096", 2, 5000, &gray, GEO, 4326, false);
        rgba_case("rgba degenerate bbox", 8, 8, &colour, [5.0, 5.0, 5.0, 5.0], 4326, false);

        let mut f32_case = |name, w: u32, h: u32, data: &[u8], bbox, epsg, mask| {
            out.push((name, encode_geotiff_f32(w, h, data, bbox, epsg, mask)));
        };
        let one = f32_chw(11, 8, 8, 1, false);
        f32_case("f32 1 band 8x8 3857", 8, 8, &one, MERC, 3857, false);
        f32_case("f32 1 band 8x8 3857 mask", 8, 8, &one, MERC, 3857, true);
        let one_holes = f32_chw(12, 8, 8, 1, true);
        f32_case("f32 1 band holes 8x8 4326", 8, 8, &one_holes, GEO, 4326, false);
        f32_case("f32 1 band holes 8x8 4326 mask", 8, 8, &one_holes, GEO, 4326, true);
        let two = f32_chw(13, 8, 8, 2, true);
        f32_case("f32 2 bands 8x8 3857", 8, 8, &two, MERC, 3857, false);
        f32_case("f32 2 bands 8x8 3857 mask", 8, 8, &two, MERC, 3857, true);
        let three = f32_chw(14, 7, 3, 3, true);
        f32_case("f32 3 bands 7x3 4326", 7, 3, &three, GEO, 4326, false);
        f32_case("f32 3 bands 7x3 4326 mask", 7, 3, &three, GEO, 4326, true);
        f32_case("f32 3 bands 7x3 antimeridian 4326", 7, 3, &three, ANTIMERIDIAN, 4326, false);
        f32_case("f32 3 bands 7x3 antimeridian 3857", 7, 3, &three, ANTIMERIDIAN, 3857, false);
        f32_case("f32 3 bands 7x3 32633 mask", 7, 3, &three, UTM, 32633, true);
        f32_case("f32 3 bands 7x3 4269 geographic", 7, 3, &three, GEO, 4269, false);
        f32_case("f32 3 bands 7x3 unknown epsg", 7, 3, &three, GEO, 1, true);
        let four = f32_chw(15, 3, 9, 4, true);
        f32_case("f32 4 bands 3x9 3857", 3, 9, &four, MERC, 3857, false);
        f32_case("f32 4 bands 3x9 3857 mask", 3, 9, &four, MERC, 3857, true);
        let five = f32_chw(16, 4, 4, 5, true);
        f32_case("f32 5 bands 4x4 3857", 4, 4, &five, MERC, 3857, false);
        f32_case("f32 5 bands 4x4 3857 mask", 4, 4, &five, MERC, 3857, true);
        // More than 15 planes: the band count clamps to 15.
        let many = f32_chw(17, 2, 2, 17, false);
        f32_case("f32 17 bands 2x2 clamps to 15", 2, 2, &many, MERC, 3857, false);
        f32_case("f32 17 bands 2x2 clamps to 15 mask", 2, 2, &many, MERC, 3857, true);
        // Buffer shorter than one plane: one band, NaN-padded.
        f32_case("f32 short buffer", 8, 8, &one[..100], MERC, 3857, false);
        f32_case("f32 short ragged buffer", 8, 8, &one[..103], MERC, 3857, false);
        f32_case("f32 short buffer mask", 8, 8, &one_holes[..100], MERC, 3857, true);
        f32_case("f32 empty buffer", 2, 2, &[], MERC, 3857, false);
        f32_case("f32 empty buffer mask", 2, 2, &[], GEO, 4326, true);
        // Trailing partial plane: the band count rounds down and the tail is ignored.
        f32_case("f32 2.5 planes", 8, 8, &two[..two.len() - 30], MERC, 3857, false);
        f32_case("f32 1 band ragged tail", 8, 8, &two[..one.len() + 6], MERC, 3857, false);
        f32_case("f32 2.5 planes mask", 4, 8, &two[..320], MERC, 3857, true);
        // Dimensions clamp to 1..=4096.
        f32_case("f32 zero width", 0, 4, &one, MERC, 3857, false);
        f32_case("f32 zero size mask", 0, 0, &one, GEO, 4326, true);
        f32_case("f32 5000x1 clamps to 4096", 5000, 1, &one, MERC, 3857, true);
        f32_case("f32 1x5000 clamps to 4096", 1, 5000, &one, GEO, 4326, false);
        f32_case("f32 degenerate bbox", 8, 8, &one, [5.0, 5.0, 5.0, 5.0], 4326, false);
        out
    }

    /// `(case, exact byte length, FNV-1a 128 of the bytes)`.
    #[rustfmt::skip]
    const GOLDEN: &[(&str, usize, u128)] = &[
        ("rgba colour opaque 8x8 3857", 472, 0x40b39d95e1dd8242cd4e61314b2c883e),
        ("rgba colour opaque 8x8 3857 mask", 550, 0xf2788e72fef190c2149992bd0ad51142),
        ("rgba colour transparent 8x8 3857", 472, 0x9ea75746088952c6fd1c2110644f8033),
        ("rgba colour transparent 8x8 3857 mask", 550, 0x0daaeb6cb274306c0bb60d18c0313598),
        ("rgba gray opaque 8x8 4326", 338, 0x2794fc34fb532e35d94759148881cdea),
        ("rgba gray opaque 8x8 4326 mask", 414, 0x8db51fc3e8a7f4d72a4c1972c2b997e4),
        ("rgba gray transparent 8x8 4326", 338, 0x7493896440638fc67fb76cef663c8efd),
        ("rgba gray transparent 8x8 4326 mask", 414, 0x9319f6be7aa1a7b185529ec0ac966d8e),
        ("rgba all transparent 8x8 3857", 472, 0x2b4daaa07709ef679a011a6f8c8bcc58),
        ("rgba all transparent 8x8 3857 mask", 550, 0xa94bbefbb0a18655d44f2a5190253fc6),
        ("rgba colour 7x3 4326", 343, 0x076b1662af032024a2adeb99dadc0ca5),
        ("rgba colour 7x3 4326 mask", 378, 0x1e3ca78cdc500be2e43a2db7d16c7e51),
        ("rgba colour 7x3 antimeridian 4326 mask", 378, 0x7978095b925f0cb102652db90ecc241e),
        ("rgba colour 7x3 antimeridian 3857 mask", 378, 0x6020e3a1afdc59f9484f145772bbb8b2),
        ("rgba colour 7x3 32633", 343, 0xe73936a6b077f9bd9848725a02c50344),
        ("rgba colour 7x3 4269 geographic", 343, 0x6a0ec148dad417d07ff62d403dc9a944),
        ("rgba colour 7x3 unknown epsg", 378, 0x590b512ee96a0a1eedb1333cfaf81a39),
        ("rgba gray 3x9 3857 mask", 340, 0xf88e601763d18fab5dd4cb008a765fd8),
        ("rgba colour short buffer", 472, 0xe1df2d707d22aaccc0a616b02d35bc34),
        ("rgba colour short buffer mask", 550, 0x41fc6a168ec981eae90169704fb6fd16),
        ("rgba gray short buffer", 338, 0x16b6a1de3f464a033cc7b15f5ec0a1c5),
        ("rgba gray short buffer mask", 414, 0x19d4b5c05f9ce6a4e6150105b0cf3c70),
        ("rgba colour long buffer", 376, 0x6dfd48bae9edc6882312c2e69452dd5d),
        ("rgba colour long buffer mask", 422, 0xcfdc98d6088fd5459184a0b51dc54faa),
        ("rgba empty buffer", 310, 0xaa88dbae3e02844be81b95f5a3674e63),
        ("rgba zero width", 292, 0xd385a0fd37e237113d972b6fdff74d24),
        ("rgba zero size mask", 298, 0x1c3546afc2f5571eed355c28ee240086),
        ("rgba 5000x2 clamps to 4096", 33062, 0x37a63c2171ea4e25e7c3486ea9dbbd1a),
        ("rgba 2x5000 clamps to 4096", 8466, 0x91a0617ef64144c3b603af220996a770),
        ("rgba degenerate bbox", 472, 0x3355ba441ba3ada133eafe2c30eba037),
        ("f32 1 band 8x8 3857", 554, 0xdde4d748674ef2bc81d021240a98ebcf),
        ("f32 1 band 8x8 3857 mask", 822, 0x08db91822f415632d017d03b0bdf0c44),
        ("f32 1 band holes 8x8 4326", 554, 0x73fa21f219cfcb212fd0c1e6cca7c510),
        ("f32 1 band holes 8x8 4326 mask", 822, 0xe58604386c30557b1ce5dbc5dfdfa66f),
        ("f32 2 bands 8x8 3857", 822, 0xfd4ce31b6df242eeb08d326262699eac),
        ("f32 2 bands 8x8 3857 mask", 1090, 0x11265b727a01fe9311caf16178e1ac38),
        ("f32 3 bands 7x3 4326", 574, 0x0c890f0aa9924e51f88e4f308d1348c2),
        ("f32 3 bands 7x3 4326 mask", 668, 0x8a401a604a138d3f2e1ec5dea40727fc),
        ("f32 3 bands 7x3 antimeridian 4326", 574, 0x95d0d5edc0507cd44bb6154c2b0c3cb1),
        ("f32 3 bands 7x3 antimeridian 3857", 574, 0xa89793beb7fec3fd6cffcda94e003f8d),
        ("f32 3 bands 7x3 32633 mask", 668, 0x4535b7b0062e7d069cecc358c66c4f4f),
        ("f32 3 bands 7x3 4269 geographic", 574, 0x5abb9c439762c34563b8bdedfd76d8d1),
        ("f32 3 bands 7x3 unknown epsg", 668, 0xdb6fe5cae20f1df19914485f82cae1ac),
        ("f32 4 bands 3x9 3857", 764, 0xc4bf027ac462ad26a47c425ad6cb078a),
        ("f32 4 bands 3x9 3857 mask", 878, 0x5cb9dbbfdb3238bd60657adf68bcb841),
        ("f32 5 bands 4x4 3857", 658, 0x205b3b62431759c71f590198d3831d3c),
        ("f32 5 bands 4x4 3857 mask", 728, 0x74f1828bfabd673cde86a4fc34f5d3ed),
        ("f32 17 bands 2x2 clamps to 15", 638, 0x2a738330881c87d8e93c49f6ade4e1e8),
        ("f32 17 bands 2x2 clamps to 15 mask", 660, 0x61e6dcc95dccf30856b256f2c8cc804a),
        ("f32 short buffer", 554, 0x1be98b50444e6b34ec3b712059699c72),
        ("f32 short ragged buffer", 554, 0x1be98b50444e6b34ec3b712059699c72),
        ("f32 short buffer mask", 822, 0xdafcc46b7bc612559c77c3783340626a),
        ("f32 empty buffer", 314, 0xa26b64f76a69388a8bcb8436089db1bf),
        ("f32 empty buffer mask", 342, 0xdde2eb1040e5b226fb4653f317f95ae7),
        ("f32 2.5 planes", 554, 0x0b296f29fb92d4025d623bdc68dca791),
        ("f32 1 band ragged tail", 554, 0x0b296f29fb92d4025d623bdc68dca791),
        ("f32 2.5 planes mask", 706, 0x15259173be79ae4d8dcd26193187e3ac),
        ("f32 zero width", 638, 0xe1a00199c0d3744b183d8994904236e8),
        ("f32 zero size mask", 468, 0x9a8bb8bbb1d3f5199de9019207180a83),
        ("f32 5000x1 clamps to 4096", 33078, 0xd67aadf53f8cbde3572bd480f11453c9),
        ("f32 1x5000 clamps to 4096", 16682, 0xc2d46d38acc6ae0e082da46b4bf4970b),
        ("f32 degenerate bbox", 554, 0x3b22078a2854a2e35ee1684eeb246636),
    ];

    #[test]
    fn existing_geotiff_encoders_are_byte_stable() {
        let cases = cases();
        if std::env::var_os("COGZILLA_PRINT_GEOTIFF_GOLDEN").is_some() {
            for (name, bytes) in &cases {
                println!("        ({name:?}, {}, 0x{:032x}),", bytes.len(), fnv1a_128(bytes));
            }
        }
        assert_eq!(cases.len(), GOLDEN.len(), "golden table and case list differ in length");
        for ((name, bytes), (want_name, want_len, want_hash)) in cases.iter().zip(GOLDEN) {
            assert_eq!(name, want_name, "golden table order");
            assert_eq!(bytes.len(), *want_len, "{name}: length changed");
            assert_eq!(fnv1a_128(bytes), *want_hash, "{name}: bytes changed");
        }
    }

    /// The digest is only a pin if it distinguishes neighbouring files.
    #[test]
    fn golden_digest_is_sensitive_to_single_bytes() {
        let a = noise(1, 512);
        let mut b = a.clone();
        b[511] ^= 1;
        assert_ne!(fnv1a_128(&a), fnv1a_128(&b));
        assert_eq!(fnv1a_128(b""), 0x6c62272e07bb014262b821756295c58d);
        assert_eq!(fnv1a_128(b"a"), 0xd228cb696f1a8caf78912b704e4a8964);
    }
}

#[cfg(test)]
mod writer_tests {
    //! Reads `write_geotiff` output back, both with a minimal IFD walker (tag order, counts,
    //! offsets inside the file) and with `cogzilla_tiff::parse_cog`.

    use bytes::BufMut;
    use cogzilla_raster::SampleType;
    use cogzilla_tiff::{PlanarConfig, parse_cog};

    use super::*;

    const GEO: [f64; 4] = [8.0, 45.0, 9.0, 46.0];
    const MERC: [f64; 4] = [-1000.0, 2000.0, 3000.0, 5000.0];

    struct Entry {
        tag: u16,
        ty: u16,
        count: u32,
        value: [u8; 4],
    }

    struct Ifd<'a> {
        file: &'a [u8],
        entries: Vec<Entry>,
        /// End of the IFD itself (after the next-IFD pointer).
        end: usize,
    }

    fn u16_at(b: &[u8], at: usize) -> u16 {
        u16::from_le_bytes([b[at], b[at + 1]])
    }

    fn u32_at(b: &[u8], at: usize) -> u32 {
        u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
    }

    /// Walk the single IFD, checking the header and that tags strictly ascend.
    fn read_ifd(file: &[u8]) -> Ifd<'_> {
        assert_eq!(&file[..4], b"II\x2a\x00", "little-endian Classic TIFF magic");
        let at = u32_at(file, 4) as usize;
        assert_eq!(at, 8);
        let n = u16_at(file, at) as usize;
        let entries: Vec<Entry> = (0..n)
            .map(|i| {
                let e = at + 2 + 12 * i;
                Entry {
                    tag: u16_at(file, e),
                    ty: u16_at(file, e + 2),
                    count: u32_at(file, e + 4),
                    value: [file[e + 8], file[e + 9], file[e + 10], file[e + 11]],
                }
            })
            .collect();
        assert!(entries.windows(2).all(|w| w[0].tag < w[1].tag), "tags strictly ascending");
        let end = at + 2 + 12 * n + 4;
        assert_eq!(u32_at(file, end - 4), 0, "no second IFD");
        Ifd { file, entries, end }
    }

    impl Ifd<'_> {
        fn tags(&self) -> Vec<u16> {
            self.entries.iter().map(|e| e.tag).collect()
        }

        fn entry(&self, tag: u16) -> &Entry {
            self.entries.iter().find(|e| e.tag == tag).unwrap_or_else(|| panic!("tag {tag} missing"))
        }

        fn has(&self, tag: u16) -> bool {
            self.entries.iter().any(|e| e.tag == tag)
        }

        /// File range of an out-of-line value, `None` when it sits in the entry.
        fn out_of_line(&self, e: &Entry) -> Option<std::ops::Range<usize>> {
            let size = match e.ty {
                2 => 1,
                3 => 2,
                4 => 4,
                12 => 8,
                other => panic!("unexpected field type {other} on tag {}", e.tag),
            } * e.count as usize;
            (size > 4).then(|| {
                let at = u32::from_le_bytes(e.value) as usize;
                assert_eq!(at % 2, 0, "tag {} value on a word boundary", e.tag);
                assert!(at >= self.end && at + size <= self.file.len(), "tag {} value in file", e.tag);
                at..at + size
            })
        }

        fn payload(&self, tag: u16, ty: u16) -> Vec<u8> {
            let e = self.entry(tag);
            assert_eq!(e.ty, ty, "field type of tag {tag}");
            match self.out_of_line(e) {
                Some(range) => self.file[range].to_vec(),
                None => {
                    let size = e.count as usize * if ty == 4 { 4 } else { ty as usize - 1 };
                    e.value[..size].to_vec()
                }
            }
        }

        fn shorts(&self, tag: u16) -> Vec<u16> {
            self.payload(tag, 3).chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
        }

        fn longs(&self, tag: u16) -> Vec<u32> {
            self.payload(tag, 4).chunks_exact(4).map(|c| u32_at(c, 0)).collect()
        }

        fn doubles(&self, tag: u16) -> Vec<f64> {
            self.payload(tag, 12)
                .chunks_exact(8)
                .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
                .collect()
        }
    }

    fn sample_type(sample: GeoTiffSample) -> SampleType {
        match sample {
            GeoTiffSample::U8 => SampleType::Uint8,
            GeoTiffSample::I8 => SampleType::Int8,
            GeoTiffSample::U16 => SampleType::Uint16,
            GeoTiffSample::I16 => SampleType::Int16,
            GeoTiffSample::U32 => SampleType::Uint32,
            GeoTiffSample::I32 => SampleType::Int32,
            GeoTiffSample::U64 => SampleType::Uint64,
            GeoTiffSample::I64 => SampleType::Int64,
            GeoTiffSample::F32 => SampleType::Float32,
            GeoTiffSample::F64 => SampleType::Float64,
        }
    }

    /// Distinct bytes per position so a misplaced strip cannot go unnoticed.
    fn pixels(spec: &GeoTiffSpec<'_>) -> Vec<u8> {
        (0..spec.pixel_bytes()).map(|i| (i * 31 + 7) as u8).collect()
    }

    /// Write `spec` + `data` and check everything that must hold for any spec. Returns the file.
    fn write_and_check(spec: &GeoTiffSpec<'_>, data: &[u8]) -> Bytes {
        assert_eq!(data.len(), spec.pixel_bytes());
        let file = write_geotiff(spec, |buf| buf.put_slice(data));
        let ifd = read_ifd(&file);
        let bands = spec.bands as usize;
        let planar = spec.layout == GeoTiffLayout::Planar && bands > 1;

        let mut want_tags = vec![256, 257, 258, 259, 262, 273, 277, 278, 279, 284];
        if !spec.extra_samples.is_empty() {
            want_tags.push(338);
        }
        if spec.sample_format_tag {
            want_tags.push(339);
        }
        want_tags.extend([33550, 33922, 34735]);
        if spec.nodata.is_some() {
            want_tags.push(42113);
        }
        assert_eq!(ifd.tags(), want_tags);

        assert_eq!(ifd.longs(256), [spec.width]);
        assert_eq!(ifd.longs(257), [spec.height]);
        assert_eq!(ifd.shorts(258), vec![spec.sample.bits(); bands]);
        assert_eq!(ifd.shorts(259), [1]);
        assert_eq!(ifd.shorts(262), [spec.photometric as u16]);
        assert_eq!(ifd.shorts(277), [spec.bands]);
        assert_eq!(ifd.longs(278), [spec.height]);
        assert_eq!(ifd.shorts(284), [if planar { 2 } else { 1 }]);
        if !spec.extra_samples.is_empty() {
            assert_eq!(ifd.shorts(338), spec.extra_samples);
        }
        if spec.sample_format_tag {
            assert_eq!(ifd.shorts(339), vec![spec.sample.format(); bands]);
        }

        // Strips: after every tag value, back to back, ending exactly at end of file, and
        // holding the input bytes in order.
        let offsets = ifd.longs(273);
        let counts = ifd.longs(279);
        let n_strips = if planar { bands } else { 1 };
        assert_eq!(offsets.len(), n_strips);
        assert_eq!(counts.len(), n_strips);
        let meta_end = ifd
            .entries
            .iter()
            .filter_map(|e| ifd.out_of_line(e))
            .map(|r| r.end)
            .max()
            .unwrap();
        // (rounded up to a word boundary after odd-length GDAL_NODATA text)
        assert_eq!(
            offsets[0] as usize,
            meta_end.next_multiple_of(2),
            "pixels follow the last tag value"
        );
        let mut at = offsets[0] as usize;
        for (&off, &count) in offsets.iter().zip(&counts) {
            assert_eq!(off as usize, at);
            assert_eq!(count as usize, data.len() / n_strips);
            at += count as usize;
        }
        assert_eq!(at, file.len());
        assert_eq!(&file[offsets[0] as usize..], data);

        // Georeferencing tail.
        let dx = (spec.bbox[2] - spec.bbox[0]) / f64::from(spec.width);
        let dy = (spec.bbox[3] - spec.bbox[1]) / f64::from(spec.height);
        assert_eq!(ifd.doubles(33550), [dx, dy, 0.0]);
        assert_eq!(ifd.doubles(33922), [0.0, 0.0, 0.0, spec.bbox[0], spec.bbox[3], 0.0]);
        let (model, key) = if spec.epsg == 4326 { (2, 2048) } else { (1, 3072) };
        assert_eq!(
            ifd.shorts(34735),
            [1, 1, 0, 3, 1024, 0, 1, model, 1025, 0, 1, 1, key, 0, 1, spec.epsg]
        );
        if let Some(nodata) = spec.nodata {
            let mut want = nodata.as_bytes().to_vec();
            want.push(0);
            assert_eq!(ifd.payload(42113, 2), want);
        }

        // The repo's own reader agrees.
        let index = parse_cog(&file).expect("cogzilla-tiff opens the file");
        assert_eq!(index.ifds.len(), 1);
        assert!(index.is_georeferenced);
        let full = index.full_res();
        assert_eq!((full.width, full.height), (spec.width, spec.height));
        assert_eq!(full.samples_per_pixel, spec.bands);
        assert_eq!(full.sample_type, sample_type(spec.sample));
        assert_eq!(
            full.planar_config,
            if planar { PlanarConfig::Planar } else { PlanarConfig::Chunky }
        );
        assert!(full.is_stripped);
        assert_eq!((full.tile_width, full.tile_height), (spec.width, spec.height));
        assert_eq!(full.photometric, spec.photometric as u16);
        assert_eq!(full.extra_samples, spec.extra_samples);
        for band in 0..n_strips {
            assert_eq!(
                full.tile_byte_range(0, 0, band as u16),
                Some((u64::from(offsets[band]), counts[band]))
            );
        }
        assert_eq!(full.pixel_to_crs_pt(0.0, 0.0), (spec.bbox[0], spec.bbox[3]));
        let (x1, y1) = full.pixel_to_crs_pt(f64::from(spec.width), f64::from(spec.height));
        assert!((x1 - spec.bbox[2]).abs() < 1e-9 && (y1 - spec.bbox[1]).abs() < 1e-9);
        if spec.epsg == 4326 {
            assert_eq!(index.geo_keys.model_type, Some(2));
            assert_eq!(index.geo_keys.geographic_type, Some(4326));
        } else {
            assert_eq!(index.geo_keys.model_type, Some(1));
            assert_eq!(index.geo_keys.projected_cs_type, Some(spec.epsg));
        }
        match spec.nodata {
            None => assert_eq!(index.nodata, None),
            Some("nan") => assert!(index.nodata.is_some_and(f32::is_nan)),
            Some(text) => assert_eq!(index.nodata, text.parse::<f32>().ok()),
        }
        file
    }

    fn planar(
        sample: GeoTiffSample,
        bands: u16,
        photometric: GeoTiffPhotometric,
        extra_samples: &[u16],
    ) -> GeoTiffSpec<'_> {
        GeoTiffSpec {
            width: 5,
            height: 3,
            bands,
            sample,
            sample_format_tag: true,
            layout: GeoTiffLayout::Planar,
            photometric,
            extra_samples,
            nodata: None,
            bbox: GEO,
            epsg: 4326,
        }
    }

    #[test]
    fn planar_single_band_uint16_is_one_inline_strip() {
        let spec = planar(GeoTiffSample::U16, 1, GeoTiffPhotometric::MinIsBlack, &[]);
        let file = write_and_check(&spec, &pixels(&spec));
        let ifd = read_ifd(&file);
        // One band is written as chunky with StripOffsets / StripByteCounts in their entries.
        assert_eq!(ifd.shorts(284), [1]);
        assert_eq!(ifd.entry(273).count, 1);
        assert_eq!(ifd.longs(279), [5 * 3 * 2]);
        assert_eq!(ifd.shorts(258), [16]);
        assert_eq!(ifd.shorts(339), [1]);
        assert!(!ifd.has(338));
        // 14 entries, nothing out of line but the 104-byte geo tail.
        assert_eq!(file.len(), 8 + 2 + 12 * 14 + 4 + 104 + 30);
    }

    #[test]
    fn planar_rgb_uint8_with_mask_band() {
        let spec = planar(GeoTiffSample::U8, 4, GeoTiffPhotometric::Rgb, &[2]);
        let data = pixels(&spec);
        let file = write_and_check(&spec, &data);
        let ifd = read_ifd(&file);
        assert_eq!(ifd.shorts(262), [2]);
        assert_eq!(ifd.shorts(284), [2]);
        assert_eq!(ifd.shorts(338), [2]);
        assert_eq!(ifd.shorts(258), [8, 8, 8, 8]);
        let offsets = ifd.longs(273);
        assert_eq!(ifd.longs(279), [15, 15, 15, 15]);
        // Each strip is one whole band of the band-major input.
        for (band, &off) in offsets.iter().enumerate() {
            assert_eq!(&file[off as usize..off as usize + 15], &data[band * 15..(band + 1) * 15]);
        }
    }

    #[test]
    fn planar_two_band_int16() {
        let spec = planar(GeoTiffSample::I16, 2, GeoTiffPhotometric::MinIsBlack, &[0]);
        let file = write_and_check(&spec, &pixels(&spec));
        let ifd = read_ifd(&file);
        // Two SHORTs fit in the entry; two LONG strip offsets do not.
        assert!(ifd.out_of_line(ifd.entry(258)).is_none());
        assert!(ifd.out_of_line(ifd.entry(339)).is_none());
        assert!(ifd.out_of_line(ifd.entry(273)).is_some());
        assert!(ifd.out_of_line(ifd.entry(279)).is_some());
        assert_eq!(ifd.shorts(258), [16, 16]);
        assert_eq!(ifd.shorts(339), [2, 2]);
        assert_eq!(ifd.shorts(338), [0]);
        assert_eq!(ifd.longs(279), [30, 30]);
    }

    #[test]
    fn planar_float64() {
        let one = planar(GeoTiffSample::F64, 1, GeoTiffPhotometric::MinIsBlack, &[]);
        let file = write_and_check(&one, &pixels(&one));
        let ifd = read_ifd(&file);
        assert_eq!((ifd.shorts(258), ifd.shorts(339)), (vec![64], vec![3]));
        assert_eq!(ifd.longs(279), [5 * 3 * 8]);

        // Five data bands and a mask: every SHORT array is out of line.
        let six = planar(GeoTiffSample::F64, 6, GeoTiffPhotometric::MinIsBlack, &[0, 0, 0, 0, 2]);
        let file = write_and_check(&six, &pixels(&six));
        let ifd = read_ifd(&file);
        for tag in [258, 338, 339, 273, 279] {
            assert!(ifd.out_of_line(ifd.entry(tag)).is_some(), "tag {tag}");
        }
        assert_eq!(ifd.longs(279), [120; 6]);
    }

    #[test]
    fn non_uint8_three_band_geotiffs_use_minisblack_matching_gdal() {
        use cogzilla_raster::SampleType as T;
        let bbox = [0.0, 0.0, 2.0, 1.0];
        let f32s: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        // 3-band Float32 with mask -> MinIsBlack (1), ExtraSamples=[0, 0, 2]
        let f32_tif = encode_geotiff_f32(2, 1, &f32s, bbox, 4326, true);
        let ifd = read_ifd(&f32_tif);
        assert_eq!(ifd.shorts(262), [1]);
        assert_eq!(ifd.shorts(338), [0, 0, 2]);
        // 3-band Float64 with mask -> MinIsBlack (1), ExtraSamples=[0, 0, 2]
        let f64_tif = write_geotiff_typed_f64(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2, 1, T::Float64, bbox, 4326, true);
        let ifd = read_ifd(&f64_tif);
        assert_eq!(ifd.shorts(262), [1]);
        assert_eq!(ifd.shorts(338), [0, 0, 2]);
        // 3-band Uint16 retyped from f32 with mask -> MinIsBlack (1), ExtraSamples=[0, 0, 2]
        let u16_tif = retype_f32_geotiff(&f32_tif, T::Uint16, true).unwrap();
        let ifd = read_ifd(&u16_tif);
        assert_eq!(ifd.shorts(262), [1]);
        assert_eq!(ifd.shorts(338), [0, 0, 2]);
        // 3-band Uint8 retyped from f32 with mask -> Rgb (2), ExtraSamples=[2]
        let u8_tif = retype_f32_geotiff(&f32_tif, T::Uint8, true).unwrap();
        let ifd = read_ifd(&u8_tif);
        assert_eq!(ifd.shorts(262), [2]);
        assert_eq!(ifd.shorts(338), [2]);
    }

    #[test]
    fn every_sample_type_in_both_layouts() {
        use GeoTiffSample::*;
        for sample in [U8, I8, U16, I16, U32, I32, F32, F64] {
            for layout in [GeoTiffLayout::Chunky, GeoTiffLayout::Planar] {
                for (bands, extra) in [(1u16, &[][..]), (2, &[2]), (3, &[0, 0]), (4, &[0, 0, 2])] {
                    let spec = GeoTiffSpec {
                        layout,
                        width: 3,
                        height: 7,
                        bbox: MERC,
                        epsg: 3857,
                        ..planar(sample, bands, GeoTiffPhotometric::MinIsBlack, extra)
                    };
                    write_and_check(&spec, &pixels(&spec));
                }
            }
        }
    }

    #[test]
    fn unsigned_samples_may_omit_sample_format() {
        for sample in [GeoTiffSample::U8, GeoTiffSample::U16, GeoTiffSample::U32] {
            let spec = GeoTiffSpec {
                sample_format_tag: false,
                ..planar(sample, 3, GeoTiffPhotometric::MinIsBlack, &[0, 0])
            };
            let file = write_and_check(&spec, &pixels(&spec));
            assert!(!read_ifd(&file).has(339));
        }
    }

    #[test]
    fn nodata_text_inline_and_out_of_line() {
        // Up to three characters and the NUL fit in the entry; longer text goes after the
        // GeoKey directory, padded so the pixels start on a word boundary.
        for (nodata, out_of_line) in [
            ("0", false),
            ("nan", false),
            ("-999", true),
            ("-32768", true),
            ("-9999.5", true),
        ] {
            for layout in [GeoTiffLayout::Chunky, GeoTiffLayout::Planar] {
                let spec = GeoTiffSpec {
                    nodata: Some(nodata),
                    layout,
                    ..planar(GeoTiffSample::F32, 3, GeoTiffPhotometric::MinIsBlack, &[0, 0])
                };
                let file = write_and_check(&spec, &pixels(&spec));
                let ifd = read_ifd(&file);
                let entry = ifd.entry(42113);
                assert_eq!(entry.count as usize, nodata.len() + 1);
                assert_eq!(ifd.out_of_line(entry).is_some(), out_of_line, "{nodata}");
                assert_eq!(ifd.longs(273)[0] % 2, 0);
            }
        }
    }

    #[test]
    fn pixel_writer_shortfall_is_zero_filled() {
        let spec = planar(GeoTiffSample::U16, 2, GeoTiffPhotometric::MinIsBlack, &[0]);
        let full = write_and_check(&spec, &pixels(&spec));
        let short = write_geotiff(&spec, |buf| buf.put_slice(&[9; 7]));
        assert_eq!(short.len(), full.len());
        let pixel_off = full.len() - spec.pixel_bytes();
        assert_eq!(short[..pixel_off], full[..pixel_off]);
        assert_eq!(short[pixel_off..pixel_off + 7], [9; 7]);
        assert!(short[pixel_off + 7..].iter().all(|&b| b == 0));
    }

    #[test]
    fn geographic_antimeridian_bbox_is_unwrapped() {
        let crossing = [170.0, -10.0, -170.0, 10.0];
        let spec = GeoTiffSpec {
            bbox: crossing,
            ..planar(GeoTiffSample::U8, 1, GeoTiffPhotometric::MinIsBlack, &[])
        };
        let file = write_geotiff(&spec, |buf| buf.put_slice(&pixels(&spec)));
        let ifd = read_ifd(&file);
        assert_eq!(ifd.doubles(33550), [20.0 / 5.0, 20.0 / 3.0, 0.0]);
        assert_eq!(ifd.doubles(33922), [0.0, 0.0, 0.0, 170.0, 10.0, 0.0]);
        // The same numbers in a projected CRS are not a wrap: |max_x - min_x|.
        let projected = GeoTiffSpec { epsg: 3857, ..spec };
        let file = write_geotiff(&projected, |buf| buf.put_slice(&pixels(&projected)));
        assert_eq!(read_ifd(&file).doubles(33550), [340.0 / 5.0, 20.0 / 3.0, 0.0]);
    }
}

#[cfg(test)]
mod retype_tests {
    use super::*;
        use cogzilla_raster::SampleType as T;

    fn f32s(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    /// The file `write_geotiff` makes of these samples in this type, as the retyped one must be.
    fn made(sample: GeoTiffSample, bands: u16, extra: &[u16], bbox: [f64; 4], epsg: u16, pixels: &[u8]) -> Bytes {
        let photometric = if bands - extra.len() as u16 == 3 { GeoTiffPhotometric::Rgb } else { GeoTiffPhotometric::MinIsBlack };
        let spec = GeoTiffSpec {
            width: 2,
            height: 1,
            bands,
            sample,
            sample_format_tag: true,
            layout: GeoTiffLayout::Chunky,
            photometric,
            extra_samples: extra,
            nodata: None,
            bbox,
            epsg,
        };
        write_geotiff(&spec, |buf| buf.put_slice(pixels))
    }

    #[test]
    fn a_float32_geotiff_is_rewritten_in_another_type() {
        let bbox = [-13_580_000.123_456_7, 4_540_000.987_654_3, -13_579_961.9, 4_540_019.1];
        // One band and its mask: a value, then a pixel that is not there.
        let f32_tif = encode_geotiff_f32(2, 1, &f32s(&[1.9, f32::NAN]), bbox, 3857, true);
        let u16_tif = retype_f32_geotiff(&f32_tif, T::Uint16, true).unwrap();
        let pixels: Vec<u8> = [1u16, u16::MAX, 0, u16::MIN].iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(u16_tif, made(GeoTiffSample::U16, 2, &[2], bbox, 3857, &pixels));
        // Signed, and degrees: the mask's nothing is the type's smallest number.
        let f32_tif = encode_geotiff_f32(2, 1, &f32s(&[-2.5, f32::NAN]), [10.0, 40.0, 10.5, 40.25], 4326, true);
        let i16_tif = retype_f32_geotiff(&f32_tif, T::Int16, true).unwrap();
        let pixels: Vec<u8> = [-2i16, i16::MAX, 0, i16::MIN].iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(i16_tif, made(GeoTiffSample::I16, 2, &[2], [10.0, 40.0, 10.5, 40.25], 4326, &pixels));
        // Three bands and no mask: every band is data.
        let f32_tif = encode_geotiff_f32(2, 1, &f32s(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), bbox, 32610, false);
        let u8_tif = retype_f32_geotiff(&f32_tif, T::Uint8, false).unwrap();
        assert_eq!(u8_tif, made(GeoTiffSample::U8, 3, &[], bbox, 32610, &[1, 3, 5, 2, 4, 6]));
        // float32 stays float32 and loses its `nan` no-data; the mask is the largest float.
        let f32_tif = encode_geotiff_f32(2, 1, &f32s(&[0.5, 7.0]), bbox, 3857, true);
        let again = retype_f32_geotiff(&f32_tif, T::Float32, true).unwrap();
        assert_eq!(again, made(GeoTiffSample::F32, 2, &[2], bbox, 3857, &f32s(&[0.5, f32::MAX, 7.0, f32::MAX])));
        // NaN in the mask band maps to `<T>::MIN` (invalid), not `<T>::MAX`.
        let nan_mask_tif = made(GeoTiffSample::F32, 2, &[2], bbox, 3857, &f32s(&[5.0, f32::NAN, 6.0, 255.0]));
        let u16_nan = retype_f32_geotiff(&nan_mask_tif, T::Uint16, true).unwrap();
        let px_u16: Vec<u8> = [5u16, u16::MIN, 6, u16::MAX].iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(u16_nan, made(GeoTiffSample::U16, 2, &[2], bbox, 3857, &px_u16));
    }

    #[test]
    fn what_is_not_a_float32_geotiff_is_left_alone() {
        let rgba = encode_geotiff_rgba(2, 1, &[1, 2, 3, 255, 4, 5, 6, 255], [0.0, 0.0, 2.0, 1.0], 3857, true);
        assert!(retype_f32_geotiff(&rgba, T::Uint16, true).is_none());
        let f32_tif = encode_geotiff_f32(2, 1, &f32s(&[1.0, 2.0]), [0.0, 0.0, 2.0, 1.0], 3857, false);
        assert!(retype_f32_geotiff(&f32_tif, T::Uint16, true).is_none());
        assert!(retype_f32_geotiff(&f32_tif[..f32_tif.len() - 1], T::Uint16, false).is_none());
        assert!(retype_f32_geotiff(b"II*\0 not a tiff at all", T::Uint16, false).is_none());
        assert!(retype_f32_geotiff(b"\x93NUMPY", T::Uint16, false).is_none());
    }

    #[test]
    fn retype_f32_geotiff_replaces_nan_with_zero_on_valid_masked_pixels() {
        let bbox = [0.0, 0.0, 2.0, 1.0];
        // 2-band CHW + mask (2x1): pixel 0 has band0=NaN, band1=100.0 (valid); pixel 1 has both NaN (masked).
        let f32_tif = encode_geotiff_f32(2, 1, &f32s(&[f32::NAN, f32::NAN, 100.0, f32::NAN]), bbox, 3857, true);
        let retyped = retype_f32_geotiff(&f32_tif, T::Float32, true).expect("retype f32 geotiff");
        // Chunky output layout: [px0_b0, px0_b1, px0_mask, px1_b0, px1_b1, px1_mask]
        let expected_px = f32s(&[0.0, 100.0, f32::MAX, f32::NAN, f32::NAN, f32::MIN]);
        let expected = made(GeoTiffSample::F32, 3, &[0, 2], bbox, 3857, &expected_px);
        assert_eq!(retyped.len(), expected.len());
        let data_off = retyped.len() - expected_px.len();
        let got: Vec<f32> = retyped[data_off..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(got[0], 0.0, "valid pixel band0 NaN must become 0.0");
        assert_eq!(got[1], 100.0);
        assert_eq!(got[2], f32::MAX);
        assert!(got[3].is_nan(), "masked pixel band0 stays NaN");
        assert!(got[4].is_nan(), "masked pixel band1 stays NaN");
        assert_eq!(got[5], f32::MIN);
    }

    #[test]
    fn retype_npy_and_geotiff_agree_across_both_mask_conventions() {
        let bbox = [0.0, 0.0, 2.0, 1.0];
        // Both mask conventions: (255.0 / 0.0) and (f32::MAX / f32::MIN).
        for (valid_m, masked_m) in [(255.0f32, 0.0f32), (f32::MAX, f32::MIN)] {
            // 2-band 2x1 CHW + explicit mask plane:
            // px0: b0=NaN, b1=100.0, mask=valid_m -> b0 must become 0, b1=100, mask=MAX
            // px1: b0=NaN, b1=NaN,   mask=masked_m -> b0/b1 stay absent (NaN for f32, 0 for int), mask=MIN
            let chw = [f32::NAN, f32::NAN, 100.0, f32::NAN, valid_m, masked_m];
            let hdr = cogzilla_kernel::encode_npy_f32_header(3, 1, 2);
            let mut npy_in = Vec::with_capacity(hdr.len() + chw.len() * 4);
            npy_in.extend_from_slice(&hdr);
            for v in chw {
                npy_in.extend_from_slice(&v.to_le_bytes());
            }
            // Chunky GeoTIFF with the exact same 2 pixels: [px0_b0, px0_b1, px0_m, px1_b0, px1_b1, px1_m].
            let chunky = [f32::NAN, 100.0, valid_m, f32::NAN, f32::NAN, masked_m];
            let tif_in = made(GeoTiffSample::F32, 3, &[0, 2], bbox, 3857, &f32s(&chunky));

            for kind in [T::Float32, T::Int16, T::Uint16] {
                let npy_out = cogzilla_kernel::retype_f32_npy(&npy_in, kind, true).expect("retype npy");
                let tif_out = retype_f32_geotiff(&tif_in, kind, true).expect("retype tif");

                let npy_hdr_len = 10 + usize::from(u16::from_le_bytes([npy_out[8], npy_out[9]]));
                let npy_body = &npy_out[npy_hdr_len..];
                let tif_body = &tif_out[tif_out.len() - npy_body.len()..];

                assert_eq!(npy_body.len(), tif_body.len(), "length mismatch for {kind:?}");
                match kind {
                    T::Float32 => {
                        let nf: Vec<f32> = npy_body.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
                        let tf_chunky: Vec<f32> = tif_body.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
                        // Reorder Chunky [p0b0, p0b1, p0m, p1b0, p1b1, p1m] -> CHW [p0b0, p1b0, p0b1, p1b1, p0m, p1m].
                        let tf = [tf_chunky[0], tf_chunky[3], tf_chunky[1], tf_chunky[4], tf_chunky[2], tf_chunky[5]];
                        assert_eq!(nf[0], 0.0);
                        assert_eq!(tf[0], 0.0);
                        assert!(nf[1].is_nan() && tf[1].is_nan(), "masked px1 b0 must stay NaN under ({valid_m}, {masked_m})");
                        assert_eq!(nf[2], 100.0);
                        assert_eq!(tf[2], 100.0);
                        assert!(nf[3].is_nan() && tf[3].is_nan(), "masked px1 b1 must stay NaN under ({valid_m}, {masked_m})");
                        assert_eq!(nf[4], f32::MAX);
                        assert_eq!(tf[4], f32::MAX);
                        assert_eq!(nf[5], f32::MIN);
                        assert_eq!(tf[5], f32::MIN);
                    }
                    T::Int16 => {
                        let nv: Vec<i16> = npy_body.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect();
                        let tc: Vec<i16> = tif_body.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect();
                        let tv = vec![tc[0], tc[3], tc[1], tc[4], tc[2], tc[5]];
                        assert_eq!(nv, vec![0, 0, 100, 0, i16::MAX, i16::MIN]);
                        assert_eq!(tv, nv, "Int16 values must match under ({valid_m}, {masked_m})");
                    }
                    T::Uint16 => {
                        let nv: Vec<u16> = npy_body.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
                        let tc: Vec<u16> = tif_body.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
                        let tv = vec![tc[0], tc[3], tc[1], tc[4], tc[2], tc[5]];
                        assert_eq!(nv, vec![0, 0, 100, 0, u16::MAX, u16::MIN]);
                        assert_eq!(tv, nv, "Uint16 values must match under ({valid_m}, {masked_m})");
                    }
                    _ => unreachable!(),
                }
            }
        }
    }
}
