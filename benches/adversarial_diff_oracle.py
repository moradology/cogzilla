#!/usr/bin/env python3
"""
Adversarial Differential Oracle: proves cogzilla is EQUAL or INTENTIONALLY BETTER
than stock TiTiler 2.4.1 across ~240 hard edge cases and synthetic torture COGs.

Every check is classified into one of two strict contracts:
  - PARITY: cogzilla MUST match TiTiler 2.4.1 within tight tolerances (status, shape,
    dtype, pixel PSNR/MAE, stats, point values, GeoTIFF georeferencing).
  - INTENTIONALLY_BETTER: cogzilla MUST satisfy its documented improved contract
    (docs/titiler-divergences.md) while confirming TiTiler's known limitation.

Supports:
  --cpu-only           Run cogzilla with COGZILLA_GPU_POLICY=never (primary CPU path)
  --check-cpu-vs-gpu   Also compare CPU-only cogzilla vs default GPU-enabled cogzilla
  --only SUBSTR        Filter checks by substring
  --json-out PATH      Write machine-readable JSON report

Exits 0 iff 100% of checks pass (zero unexplained discrepancies).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import http.server
import io
import json
import math
import os
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import warnings
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path
from urllib.parse import quote

import httpx
import mercantile
import morecantile
import numpy as np
from PIL import Image
from pyproj import Transformer
import rasterio
from rasterio.control import GroundControlPoint
from rasterio.crs import CRS as RasterioCRS
from rasterio.enums import ColorInterp, Resampling
from rasterio.transform import Affine, from_bounds
from rasterio.warp import transform_bounds

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "benches" / "fixtures"
CORPUS = FIXTURES / "parity_corpus"
EXTRA = FIXTURES / "oracle_extra"
DTYPES = FIXTURES / "dtypes"
RESULTS = ROOT / "benches" / "results"
COGZILLA_BIN = ROOT / "target" / "release" / "cogzilla-server"

sys.path.insert(0, str(ROOT / "benches"))
from oracle_lib import (
    compare_geotiffs,
    compare_images,
    compare_npy,
    first_json_diff,
    same_place,
    wait_ready,
)


def pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def q(path: Path) -> str:
    return quote(str(path.resolve()), safe="")


def qe(s: str) -> str:
    return quote(s, safe="")


def center_lonlat_and_tile(path: Path, z: int | None = None) -> tuple[float, float, int, int, int, tuple[float, float, float, float]]:
    with rasterio.open(path) as ds:
        wgs = transform_bounds(ds.crs, "EPSG:4326", *ds.bounds, densify_pts=21)
    minx, miny, maxx, maxy = wgs
    lon = 0.5 * (minx + maxx)
    lat = max(-84.0, min(84.0, 0.5 * (miny + maxy)))
    if z is None:
        span = max(maxx - minx, maxy - miny, 1e-4)
        z = max(1, min(14, int(round(math.log2(360.0 / span))) + 1))
    n = 1 << z
    tx = int((lon + 180.0) / 360.0 * n) % n
    lat_rad = math.radians(lat)
    ty = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    ty = max(0, min(n - 1, ty))
    return lon, lat, z, tx, ty, (minx, miny, maxx, maxy)


def build_synthetic_fixtures(synth_dir: Path) -> dict[str, Path]:
    synth_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    tf_sf = from_bounds(-13630000.0, 4530000.0, -13600000.0, 4560000.0, 256, 256)

    # Rotated 15-deg GeoTIFF
    p = synth_dir / "rotated_15deg_3857.tif"
    yy, xx = np.mgrid[0:256, 0:256].astype(np.float32)
    base = Affine.translation(-13630000.0, 4560000.0) * Affine.scale(100.0, -100.0) * Affine.rotation(15.0)
    with rasterio.open(p, "w", driver="GTiff", width=256, height=256, count=1, dtype="float32", crs="EPSG:3857", transform=base, tiled=True, blockxsize=256, blockysize=256, compress="deflate", nodata=-9999.0) as ds:
        ds.write((xx * 2.0 + yy * 3.0 + 10.0)[None, ...])
    out["rotated"] = p

    # South-up (positive e) GeoTIFF
    p = synth_dir / "south_up_4326.tif"
    tf_su = Affine(0.01, 0.0, -123.0, 0.0, 0.01, 37.0)
    with rasterio.open(p, "w", driver="GTiff", width=256, height=256, count=1, dtype="float32", crs="EPSG:4326", transform=tf_su, tiled=True, blockxsize=256, blockysize=256, compress="deflate", nodata=-9999.0) as ds:
        ds.write((xx + yy * 2.0 + 5.0)[None, ...])
    out["south_up"] = p

    # Ungeoreferenced RGB u8
    p = synth_dir / "ungeoref_rgb_u8.tif"
    yy8, xx8 = np.mgrid[0:128, 0:128]
    with rasterio.open(p, "w", driver="GTiff", width=128, height=128, count=3, dtype="uint8", tiled=True, blockxsize=128, blockysize=128, compress="deflate") as ds:
        ds.write(np.stack([(xx8 * 2).astype(np.uint8), (yy8 * 2).astype(np.uint8), ((xx8 + yy8) % 256).astype(np.uint8)], axis=0))
    out["ungeoref"] = p

    # High-precision int32 (> 2^24)
    p = synth_dir / "highprec_int32.tif"
    yy32, xx32 = np.mgrid[0:256, 0:256].astype(np.int32)
    arr32 = np.int32(100_000_000) + xx32 + yy32 * np.int32(256)
    arr32[0:32, 0:32] = -2147483648
    with rasterio.open(p, "w", driver="GTiff", width=256, height=256, count=1, dtype="int32", crs="EPSG:3857", transform=tf_sf, tiled=True, blockxsize=256, blockysize=256, compress="deflate", nodata=-2147483648) as ds:
        ds.write(arr32[None, ...])
    out["highprec_i32"] = p

    # Per-band distinct scale & offset (3-band uint16)
    p = synth_dir / "perband_scale_offset_u16.tif"
    yy16, xx16 = np.mgrid[0:256, 0:256].astype(np.uint16)
    with rasterio.open(p, "w", driver="GTiff", width=256, height=256, count=3, dtype="uint16", crs="EPSG:3857", transform=tf_sf, tiled=True, blockxsize=256, blockysize=256, compress="deflate", nodata=0) as ds:
        ds.write(np.stack([(xx16 * 10 + 100).astype(np.uint16), (yy16 * 20 + 500).astype(np.uint16), ((xx16 + yy16) * 5 + 50).astype(np.uint16)], axis=0))
        ds.scales = (0.01, 0.001, 0.1)
        ds.offsets = (-10.0, 5.0, -2.0)
    out["perband_so"] = p

    # Sparse COG (SPARSE_OK=TRUE)
    p = synth_dir / "sparse_i16.tif"
    tf_sp = from_bounds(-13640000.0, 4520000.0, -13600000.0, 4560000.0, 512, 512)
    with rasterio.open(p, "w", driver="GTiff", width=512, height=512, count=1, dtype="int16", crs="EPSG:3857", transform=tf_sp, tiled=True, blockxsize=256, blockysize=256, compress="deflate", nodata=-9999, SPARSE_OK="TRUE") as ds:
        yy_s, xx_s = np.mgrid[0:256, 0:256].astype(np.int16)
        ds.write((xx_s + yy_s + 100).astype(np.int16), 1, window=rasterio.windows.Window(0, 0, 256, 256))
    out["sparse_i16"] = p

    # Categorical uint8
    p = synth_dir / "categorical_u8.tif"
    classes = np.array([0, 1, 2, 3, 5, 10], dtype=np.uint8)
    yy_c, xx_c = np.mgrid[0:256, 0:256]
    with rasterio.open(p, "w", driver="GTiff", width=256, height=256, count=1, dtype="uint8", crs="EPSG:3857", transform=tf_sf, tiled=True, blockxsize=256, blockysize=256, compress="deflate", nodata=0) as ds:
        ds.write(classes[((xx_c // 32) + (yy_c // 32)) % len(classes)][None, ...])
    out["cat_u8"] = p

    # Negative float32 with FLT_MIN nodata
    p = synth_dir / "neg_f32_min_nodata.tif"
    nd_fmin = float(np.finfo(np.float32).min)
    arr_nf = ((xx - 128.0) * 0.5 + (yy - 128.0) * 0.25).astype(np.float32)
    arr_nf[0:64, 0:64] = nd_fmin
    with rasterio.open(p, "w", driver="GTiff", width=256, height=256, count=1, dtype="float32", crs="EPSG:3857", transform=tf_sf, tiled=True, blockxsize=256, blockysize=256, compress="deflate", nodata=nd_fmin) as ds:
        ds.write(arr_nf[None, ...])
    out["neg_f32"] = p

    # Tiny 1x1 & 3x3
    p = synth_dir / "tiny_1x1.tif"
    with rasterio.open(p, "w", driver="GTiff", width=1, height=1, count=1, dtype="float32", crs="EPSG:3857", transform=from_bounds(-13630000.0, 4530000.0, -13600000.0, 4560000.0, 1, 1), nodata=-9999.0) as ds:
        ds.write(np.array([[[123.5]]], dtype=np.float32))
    out["tiny_1x1"] = p

    p = synth_dir / "tiny_3x3.tif"
    with rasterio.open(p, "w", driver="GTiff", width=3, height=3, count=1, dtype="float32", crs="EPSG:3857", transform=from_bounds(-13630000.0, 4530000.0, -13600000.0, 4560000.0, 3, 3), nodata=-9999.0) as ds:
        ds.write(np.arange(1, 10, dtype=np.float32).reshape(1, 3, 3) * 10.0)
    out["tiny_3x3"] = p

    # Constant 42.0 & All-nodata
    p = synth_dir / "constant_42.tif"
    with rasterio.open(p, "w", driver="GTiff", width=256, height=256, count=1, dtype="float32", crs="EPSG:3857", transform=tf_sf, tiled=True, blockxsize=256, blockysize=256, compress="deflate", nodata=-9999.0) as ds:
        ds.write(np.full((1, 256, 256), 42.0, dtype=np.float32))
    out["constant"] = p

    p = synth_dir / "all_nodata.tif"
    with rasterio.open(p, "w", driver="GTiff", width=256, height=256, count=1, dtype="float32", crs="EPSG:3857", transform=tf_sf, tiled=True, blockxsize=256, blockysize=256, compress="deflate", nodata=-9999.0) as ds:
        ds.write(np.full((1, 256, 256), -9999.0, dtype=np.float32))
    out["all_nodata"] = p

    # Sentinel boundary nodata: int16(-32768), uint8(255), uint16(65535)
    for tag, dt, nd, lo, hi in [
        ("nd_i16_min", "int16", -32768, -1000, 1000),
        ("nd_u8_255", "uint8", 255, 0, 250),
        ("nd_u16_max", "uint16", 65535, 0, 60000),
    ]:
        p = synth_dir / f"{tag}.tif"
        arr = (lo + ((xx_c + yy_c) * (hi - lo)) // 510).astype(dt)
        arr[0:64, 0:64] = nd
        with rasterio.open(p, "w", driver="GTiff", width=256, height=256, count=1, dtype=dt, crs="EPSG:3857", transform=tf_sf, tiled=True, blockxsize=256, blockysize=256, compress="deflate", nodata=nd) as ds:
            ds.write(arr[None, ...])
        out[tag] = p

    # Float32 with +inf / -inf / NaN
    p = synth_dir / "f32_with_infs.tif"
    arr_inf = ((xx + yy) * 0.5).astype(np.float32)
    arr_inf[0:16, 0:16] = np.inf
    arr_inf[16:32, 0:16] = -np.inf
    arr_inf[32:48, 0:16] = np.nan
    with rasterio.open(p, "w", driver="GTiff", width=256, height=256, count=1, dtype="float32", crs="EPSG:3857", transform=tf_sf, tiled=True, blockxsize=256, blockysize=256, compress="deflate", nodata=float("nan")) as ds:
        ds.write(arr_inf[None, ...])
    out["f32_infs"] = p

    # Planar RGB + internal 1-bit mask & Planar RGBA
    p = synth_dir / "planar_rgb_1bit_mask.tif"
    rgb = np.stack([xx_c.astype(np.uint8), yy_c.astype(np.uint8), ((xx_c + yy_c) // 2).astype(np.uint8)], axis=0)
    mask = np.full((256, 256), 255, dtype=np.uint8)
    mask[0:96, 0:96] = 0
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rasterio.open(p, "w", driver="GTiff", width=256, height=256, count=3, dtype="uint8", crs="EPSG:3857", transform=tf_sf, tiled=True, blockxsize=256, blockysize=256, compress="deflate", interleave="band") as ds:
            ds.write(rgb)
            ds.write_mask(mask)
    out["planar_1bit"] = p

    p = synth_dir / "planar_rgba.tif"
    rgba = np.concatenate([rgb, mask[None, ...]], axis=0)
    with rasterio.open(p, "w", driver="GTiff", width=256, height=256, count=4, dtype="uint8", crs="EPSG:3857", transform=tf_sf, tiled=True, blockxsize=256, blockysize=256, compress="deflate", interleave="band", photometric="rgb") as ds:
        ds.colorinterp = (rasterio.enums.ColorInterp.red, rasterio.enums.ColorInterp.green, rasterio.enums.ColorInterp.blue, rasterio.enums.ColorInterp.alpha)
        ds.write(rgba)
    out["planar_rgba"] = p

    # Antimeridian-crossing EPSG:4326 COG (179..181)
    p = synth_dir / "antimeridian_4326.tif"
    with rasterio.open(p, "w", driver="GTiff", width=256, height=256, count=1, dtype="float32", crs="EPSG:4326", transform=from_bounds(179.0, -18.0, 181.0, -16.0, 256, 256), tiled=True, blockxsize=256, blockysize=256, compress="deflate", nodata=-9999.0) as ds:
        ds.write((xx + yy * 2.0)[None, ...])
    out["antimeridian"] = p

    # Mixed-dtype / mixed-nodata STAC item
    p_u8, p_f32, p_u16 = synth_dir / "stac_a_u8.tif", synth_dir / "stac_b_f32.tif", synth_dir / "stac_c_u16.tif"
    with rasterio.open(p_u8, "w", driver="GTiff", width=256, height=256, count=1, dtype="uint8", crs="EPSG:3857", transform=tf_sf, tiled=True, blockxsize=256, blockysize=256, compress="deflate", nodata=0) as ds:
        a = ((xx_c // 2) + 10).astype(np.uint8); a[0:32, 0:32] = 0; ds.write(a[None, ...])
    with rasterio.open(p_f32, "w", driver="GTiff", width=256, height=256, count=1, dtype="float32", crs="EPSG:3857", transform=tf_sf, tiled=True, blockxsize=256, blockysize=256, compress="deflate", nodata=-9999.0) as ds:
        a = (yy * 0.5 + 1.0).astype(np.float32); a[224:256, 224:256] = -9999.0; ds.write(a[None, ...])
    with rasterio.open(p_u16, "w", driver="GTiff", width=256, height=256, count=1, dtype="uint16", crs="EPSG:3857", transform=tf_sf, tiled=True, blockxsize=256, blockysize=256, compress="deflate", nodata=65535) as ds:
        a = ((xx_c + yy_c) * 10 + 100).astype(np.uint16); a[0:32, 224:256] = 65535; ds.write(a[None, ...])
    p_stac_mix = synth_dir / "mixed_dtype_nodata_stac.json"
    p_stac_mix.write_text(json.dumps({
        "type": "Feature", "stac_version": "1.0.0", "id": "mixed-dtype-nodata",
        "geometry": {"type": "Polygon", "coordinates": [[[-122.44, 37.64], [-122.17, 37.64], [-122.17, 37.86], [-122.44, 37.86], [-122.44, 37.64]]]},
        "bbox": [-122.44, 37.64, -122.17, 37.86],
        "properties": {"datetime": "2024-01-01T00:00:00Z"},
        "assets": {
            "a": {"href": str(p_u8.resolve()), "type": "image/tiff; application=geotiff"},
            "b": {"href": str(p_f32.resolve()), "type": "image/tiff; application=geotiff"},
            "c": {"href": str(p_u16.resolve()), "type": "image/tiff; application=geotiff"},
        },
        "links": [],
    }))
    out["stac_mix"] = p_stac_mix

    # Wave 5: Exotic CRSs (LAEA 3035, LCC 3978, NZTM 2193, EASE 6933, UPS 32661/32761, BNG 27700, NAD83 4269, MODIS Sinusoidal)
    for name, crs, bounds in [
        ("laea_3035", "EPSG:3035", (4000000.0, 2800000.0, 4200000.0, 3000000.0)),
        ("lcc_3978", "EPSG:3978", (-1000000.0, 0.0, -800000.0, 200000.0)),
        ("nztm_2193", "EPSG:2193", (1700000.0, 5400000.0, 1800000.0, 5500000.0)),
        ("ease_6933", "EPSG:6933", (-11800000.0, 4400000.0, -11600000.0, 4600000.0)),
        ("ups_n_32661", "EPSG:32661", (1900000.0, 1000000.0, 2100000.0, 1200000.0)),
        ("ups_s_32761", "EPSG:32761", (1900000.0, 2800000.0, 2100000.0, 3000000.0)),
        ("bng_27700", "EPSG:27700", (400000.0, 200000.0, 500000.0, 300000.0)),
        ("nad83_4269", "EPSG:4269", (-123.0, 37.0, -122.0, 38.0)),
        ("modis_sinu", "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +R=6371007.181 +units=m +no_defs", (-10000000.0, 4000000.0, -9800000.0, 4200000.0)),
    ]:
        p_crs = synth_dir / f"{name}.tif"
        arr_c = (xx * 1.5 + yy * 2.5 + 10.0).astype(np.float32)
        arr_c[0:24, 0:24] = -9999.0
        with rasterio.open(p_crs, "w", driver="GTiff", width=256, height=256, count=1, dtype="float32", crs=crs, transform=from_bounds(*bounds, 256, 256), tiled=True, blockxsize=256, blockysize=256, compress="deflate", nodata=-9999.0) as ds:
            ds.write(arr_c[None, ...])
        out[name] = p_crs

    # GCP-only affine GeoTIFF (no geotransform)
    p_gcp = synth_dir / "gcp_affine_4326.tif"
    gcps = [
        GroundControlPoint(row=0, col=0, x=-123.0, y=38.0, id="1"),
        GroundControlPoint(row=0, col=256, x=-122.0, y=38.0, id="2"),
        GroundControlPoint(row=256, col=0, x=-123.0, y=37.0, id="3"),
        GroundControlPoint(row=256, col=256, x=-122.0, y=37.0, id="4"),
    ]
    with rasterio.open(p_gcp, "w", driver="GTiff", width=256, height=256, count=1, dtype="float32", gcps=gcps, crs="EPSG:4326", tiled=True, blockxsize=256, blockysize=256, compress="deflate", nodata=-9999.0) as ds:
        ds.write((xx + yy * 2.0 + 10.0)[None, ...])
    out["gcp_affine"] = p_gcp

    # Multi-overview + unscale + nodata COG
    p_ov = synth_dir / "multi_ovr_unscale_u16.tif"
    yy1k, xx1k = np.mgrid[0:1024, 0:1024].astype(np.uint16)
    arr_ov = ((xx1k * 20 + yy1k * 30 + 1000) % 50000 + 100).astype(np.uint16)
    arr_ov[0:128, 0:128] = 0
    with rasterio.open(p_ov, "w", driver="GTiff", width=1024, height=1024, count=1, dtype="uint16", crs="EPSG:3857", transform=from_bounds(-13650000.0, 4500000.0, -13550000.0, 4600000.0, 1024, 1024), tiled=True, blockxsize=256, blockysize=256, compress="deflate", predictor=2, nodata=0) as ds:
        ds.write(arr_ov[None, ...])
        ds.scales = (0.0001,)
        ds.offsets = (-0.1,)
        ds.build_overviews([2, 4, 8], Resampling.nearest)
    out["multi_ovr_unscale"] = p_ov

    # Stripped float64 predictor=3 deflate, BigTIFF int32 predictor=2 zstd, int8 nodata=-128 unscale
    p_f64p3 = synth_dir / "stripped_f64_p3.tif"
    arr_f64 = (xx.astype(np.float64) * 1.234567890123 + yy.astype(np.float64) * 9.876543210987 + 1e9)
    arr_f64[0:32, 0:32] = -9999.0
    with rasterio.open(p_f64p3, "w", driver="GTiff", width=256, height=256, count=1, dtype="float64", crs="EPSG:3857", transform=tf_sf, tiled=False, compress="deflate", predictor=3, nodata=-9999.0) as ds:
        ds.write(arr_f64[None, ...])
    out["stripped_f64_p3"] = p_f64p3

    p_big = synth_dir / "bigtiff_i32_p2_zstd.tif"
    arr_big = (xx.astype(np.int32) * 100000 - yy.astype(np.int32) * 50000 + 50000000)
    arr_big[0:32, 0:32] = -2147483648
    with rasterio.open(p_big, "w", driver="GTiff", width=256, height=256, count=1, dtype="int32", crs="EPSG:3857", transform=tf_sf, tiled=True, blockxsize=256, blockysize=256, compress="zstd", predictor=2, nodata=-2147483648, BIGTIFF="YES") as ds:
        ds.write(arr_big[None, ...])
    out["bigtiff_i32"] = p_big

    p_i8 = synth_dir / "int8_nd128_unscale.tif"
    arr_i8 = (((xx + yy) % 200) - 100).astype(np.int8)
    arr_i8[0:32, 0:32] = -128
    with rasterio.open(p_i8, "w", driver="GTiff", width=256, height=256, count=1, dtype="int8", crs="EPSG:3857", transform=tf_sf, tiled=True, blockxsize=256, blockysize=256, compress="deflate", nodata=-128) as ds:
        ds.write(arr_i8[None, ...])
        ds.scales = (0.1,)
        ds.offsets = (10.0,)
    out["int8_unscale"] = p_i8

    # Wave 9: Exotic codecs & bit packing (LERC/DEFLATE/ZSTD, WebP lossless/lossy, JPEG YCbCr,
    # 1-bit, 4-bit, 12-bit, Predictor=2 int16, Predictor=3 float64, PlanarConfig=SEPARATE).
    tf_w9 = from_bounds(-122.5, 37.5, -122.0, 38.0, 128, 128)
    yy9, xx9 = np.mgrid[0:128, 0:128]
    f32_w9 = (100.0 + xx9.astype(np.float32) * 2.5 + yy9.astype(np.float32) * 1.25)
    f32_w9_nd = f32_w9.copy()
    f32_w9_nd[0:24, 0:24] = -9999.0
    rgb_w9 = np.stack([
        np.clip(20 + xx9 * 1.7, 0, 255).astype(np.uint8),
        np.clip(40 + yy9 * 1.5, 0, 255).astype(np.uint8),
        np.clip(220 - (xx9 + yy9) * 0.7, 0, 255).astype(np.uint8),
    ], axis=0)
    w9_specs = [
        ("w9_lerc", f32_w9[None], dict(compress="lerc", max_z_error=0.01)),
        ("w9_lerc_deflate", f32_w9_nd[None], dict(compress="lerc_deflate", max_z_error=0.01, nodata=-9999.0)),
        ("w9_lerc_zstd", f32_w9_nd[None], dict(compress="lerc_zstd", max_z_error=0.01, nodata=-9999.0)),
        ("w9_webp_lossless", rgb_w9, dict(compress="webp", webp_lossless=True)),
        ("w9_webp_lossy", rgb_w9, dict(compress="webp", webp_level=90)),
        ("w9_jpeg_ycbcr", rgb_w9, dict(compress="jpeg", photometric="ycbcr", jpeg_quality=95)),
        ("w9_nbits_1", ((xx9 // 16 + yy9 // 16) % 2).astype(np.uint8)[None], dict(compress="deflate", nbits=1)),
        ("w9_nbits_4", ((xx9 // 8 + yy9 // 16) % 16).astype(np.uint8)[None], dict(compress="deflate", nbits=4)),
        ("w9_nbits_12", ((xx9 * 15 + yy9 * 17) % 4096).astype(np.uint16)[None], dict(compress="deflate", nbits=12)),
        ("w9_pred2_i16", ((xx9 * 10 - yy9 * 8 - 500).astype(np.int16))[None], dict(compress="deflate", predictor=2, nodata=-32768)),
        ("w9_pred3_f64", (f32_w9.astype(np.float64) * 1.000123)[None], dict(compress="zstd", predictor=3)),
        ("w9_planar_sep_rgb", rgb_w9, dict(compress="deflate", interleave="band")),
    ]
    for name, arr, kw in w9_specs:
        p = synth_dir / f"{name}.tif"
        with rasterio.open(p, "w", driver="GTiff", width=128, height=128, count=arr.shape[0], dtype=arr.dtype, crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64, **kw) as ds:
            ds.write(arr)
        out[name] = p

    # Wave 11: Rotated (30 deg EPSG:3857), Sheared (EPSG:32610), South-Up (e > 0 EPSG:4326),
    # and additional regional/global CRSs (Equal Earth EPSG:8857, CONUS Albers EPSG:5070,
    # France Lambert-93 EPSG:2154, UTM 56S EPSG:32756), plus a local STAC item.
    arr_w11 = (100.0 + xx9.astype(np.float32) * 2.0 + yy9.astype(np.float32) * 1.5)
    arr_w11[0:16, 0:16] = -9999.0
    tf_rot30 = Affine.translation(-13640000.0, 4560000.0) * Affine.rotation(30.0) * Affine.scale(150.0, -150.0)
    for name, crs_s, tf_m in [
        ("w11_rot30", "EPSG:3857", tf_rot30),
        ("w11_sheared", "EPSG:32610", Affine(100.0, 25.0, 540000.0, 15.0, -100.0, 4190000.0)),
        ("w11_southup", "EPSG:4326", Affine(0.004, 0.0, -122.5, 0.0, 0.004, 37.5)),
        ("w11_equal_earth", "EPSG:8857", from_bounds(-1000000.0, 4500000.0, 1000000.0, 5500000.0, 128, 128)),
        ("w11_albers_5070", "EPSG:5070", from_bounds(-500000.0, 1500000.0, 500000.0, 2200000.0, 128, 128)),
        ("w11_lambert93_2154", "EPSG:2154", from_bounds(600000.0, 6700000.0, 750000.0, 6850000.0, 128, 128)),
        ("w11_utm56s_32756", "EPSG:32756", from_bounds(300000.0, 6220000.0, 350000.0, 6270000.0, 128, 128)),
    ]:
        p = synth_dir / f"{name}.tif"
        with rasterio.open(p, "w", driver="GTiff", width=128, height=128, count=1, dtype="float32", crs=crs_s, transform=tf_m, tiled=True, blockxsize=64, blockysize=64, compress="deflate", nodata=-9999.0) as ds:
            ds.write(arr_w11[None])
        out[name] = p

    # Local self-contained STAC item with two 1-band assets (like Landsat/Sentinel STAC items)
    p_stac_local = synth_dir / "w11_local_stac.json"
    p_stac_a = synth_dir / "w11_stac_a.tif"
    p_stac_b = synth_dir / "w11_stac_b.tif"
    tf_stac = from_bounds(-122.75, 37.65, -122.45, 37.95, 128, 128)
    with rasterio.open(p_stac_a, "w", driver="GTiff", width=128, height=128, count=1, dtype="float32", crs="EPSG:4326", transform=tf_stac, tiled=True, blockxsize=64, blockysize=64, compress="deflate") as ds:
        ds.write((100.0 + xx9.astype(np.float32) * 1.5)[None])
    with rasterio.open(p_stac_b, "w", driver="GTiff", width=128, height=128, count=1, dtype="float32", crs="EPSG:4326", transform=tf_stac, tiled=True, blockxsize=64, blockysize=64, compress="deflate") as ds:
        ds.write((200.0 + yy9.astype(np.float32) * 2.0)[None])
    p_stac_local.write_text(json.dumps({
        "type": "Feature",
        "stac_version": "1.0.0",
        "id": "w11-local-stac",
        "bbox": [-122.75, 37.65, -122.45, 37.95],
        "geometry": {"type": "Polygon", "coordinates": [[[-122.75, 37.65], [-122.45, 37.65], [-122.45, 37.95], [-122.75, 37.95], [-122.75, 37.65]]]},
        "properties": {"datetime": "2024-06-01T00:00:00Z"},
        "assets": {
            "a": {"href": str(p_stac_a.resolve()), "type": "image/tiff; application=geotiff"},
            "b": {"href": str(p_stac_b.resolve()), "type": "image/tiff; application=geotiff"},
        },
    }))
    out["w11_stac"] = p_stac_local

    # Wave 14: High-risk qualitative blind spots (Big-Endian MM TIFFs with predictors,
    # overview mask IFDs, external .msk sidecar, per-band scales/offsets, negative scale,
    # 6-band multispectral, Gray+Alpha, 16-bit RGBA, +/-Inf, constant-value, all-nodata,
    # South Pole EPSG:3031, antimeridian [170..190]).
    for name, arr_be, kw_be in [
        ("w14_be_u16", (1000 + xx9 * 100 + yy9 * 50).astype(np.uint16)[None], dict(compress="deflate", predictor=2, endianness="BIG")),
        ("w14_be_i16", ((xx9 * 50 - yy9 * 40 - 1000).astype(np.int16))[None], dict(compress="deflate", predictor=2, nodata=-32768, endianness="BIG")),
        ("w14_be_f32_p3", (100.5 + xx9.astype(np.float32) * 2.25 + yy9.astype(np.float32) * 1.125)[None], dict(compress="deflate", predictor=3, endianness="BIG")),
        ("w14_be_f64_p3", (100.123456 + xx9.astype(np.float64) * 2.5 + yy9.astype(np.float64) * 1.25)[None], dict(compress="zstd", predictor=3, endianness="BIG")),
    ]:
        p = synth_dir / f"{name}.tif"
        with rasterio.open(p, "w", driver="GTiff", width=128, height=128, count=1, dtype=arr_be.dtype, crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64, **kw_be) as ds:
            ds.write(arr_be)
        out[name] = p

    p_ovrmask = synth_dir / "w14_ovr_with_mask.tif"
    yy5, xx5 = np.mgrid[0:512, 0:512]
    rgb5 = np.stack([np.clip(40 + xx5 // 3, 1, 255).astype(np.uint8), np.clip(60 + yy5 // 3, 1, 255).astype(np.uint8), np.clip(200 - (xx5 + yy5) // 6, 1, 255).astype(np.uint8)], axis=0)
    mask5 = np.where((xx5 < 256) & (yy5 < 256), 0, 255).astype(np.uint8)
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rasterio.open(p_ovrmask, "w", driver="GTiff", width=512, height=512, count=3, dtype="uint8", crs="EPSG:4326", transform=from_bounds(-123.0, 37.0, -122.0, 38.0, 512, 512), tiled=True, blockxsize=256, blockysize=256, compress="deflate") as ds:
            ds.write(rgb5)
            ds.write_mask(mask5)
            ds.build_overviews([2, 4], Resampling.average)
    out["w14_ovr_mask"] = p_ovrmask

    p_extmsk = synth_dir / "w14_ext_mask.tif"
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=False):
        with rasterio.open(p_extmsk, "w", driver="GTiff", width=128, height=128, count=3, dtype="uint8", crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64, compress="deflate") as ds:
            ds.write(rgb_w9)
            # Place mask=0 inside [80:112, 64:96] so tile 10/164/396 (row 72..128, col 40..128) genuinely hits it!
            ds.write_mask(np.where((xx9 >= 64) & (xx9 < 96) & (yy9 >= 80) & (yy9 < 112), 0, 255).astype(np.uint8))
    out["w14_ext_mask"] = p_extmsk

    p_pb = synth_dir / "w14_per_band_scale.tif"
    pb3 = np.stack([(100 + xx9 * 2).astype(np.int16), (200 + yy9 * 3).astype(np.int16), (50 + xx9 + yy9).astype(np.int16)], axis=0)
    pb3[:, 80:96, 64:80] = -32768
    with rasterio.open(p_pb, "w", driver="GTiff", width=128, height=128, count=3, dtype="int16", crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64, compress="deflate", nodata=-32768) as ds:
        ds.write(pb3)
        ds.scales = (0.1, 0.5, 2.0)
        ds.offsets = (10.0, -20.0, 5.0)
    out["w14_per_band_scale"] = p_pb

    p_neg_s = synth_dir / "w14_neg_scale.tif"
    u16_s = (1000 + xx9 * 20 + yy9 * 10).astype(np.uint16)
    u16_s[80:96, 64:80] = 0
    with rasterio.open(p_neg_s, "w", driver="GTiff", width=128, height=128, count=1, dtype="uint16", crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64, compress="deflate", nodata=0) as ds:
        ds.write(u16_s[None])
        ds.scales = (-0.01,)
        ds.offsets = (100.0,)
    out["w14_neg_scale"] = p_neg_s

    p_6b = synth_dir / "w14_multispec_6b.tif"
    b6 = np.stack([((bi + 1) * 500 + xx9 * (bi + 2) + yy9 * (7 - bi)).astype(np.uint16) for bi in range(6)], axis=0)
    b6[:, 80:96, 64:80] = 0
    with rasterio.open(p_6b, "w", driver="GTiff", width=128, height=128, count=6, dtype="uint16", crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64, compress="deflate", nodata=0) as ds:
        ds.write(b6)
    out["w14_6b"] = p_6b

    p_ga = synth_dir / "w14_gray_alpha.tif"
    with rasterio.open(p_ga, "w", driver="GTiff", width=128, height=128, count=2, dtype="uint8", crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64, compress="deflate", photometric="minisblack", alpha="yes") as ds:
        ds.write(np.stack([np.clip(40 + xx9 + yy9 // 2, 1, 255).astype(np.uint8), np.where((xx9 >= 64) & (xx9 < 96) & (yy9 >= 80) & (yy9 < 112), 0, 255).astype(np.uint8)], axis=0))
        ds.colorinterp = (ColorInterp.gray, ColorInterp.alpha)
    out["w14_gray_alpha"] = p_ga

    p_rgba16 = synth_dir / "w14_rgba_u16.tif"
    with rasterio.open(p_rgba16, "w", driver="GTiff", width=128, height=128, count=4, dtype="uint16", crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64, compress="deflate", photometric="rgb", alpha="yes") as ds:
        ds.write(np.stack([(2000 + xx9 * 200).astype(np.uint16), (4000 + yy9 * 200).astype(np.uint16), (6000 + (xx9 + yy9) * 100).astype(np.uint16), np.where((xx9 >= 64) & (xx9 < 96) & (yy9 >= 80) & (yy9 < 112), 0, 65535).astype(np.uint16)], axis=0))
        ds.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha)
    out["w14_rgba_u16"] = p_rgba16

    p_const = synth_dir / "w14_const_42.tif"
    with rasterio.open(p_const, "w", driver="GTiff", width=128, height=128, count=1, dtype="float32", crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64, compress="deflate", nodata=-9999.0) as ds:
        ds.write(np.full((1, 128, 128), 42.0, dtype=np.float32))
    out["w14_const"] = p_const

    p_allnd = synth_dir / "w14_all_nodata.tif"
    with rasterio.open(p_allnd, "w", driver="GTiff", width=128, height=128, count=1, dtype="float32", crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64, compress="deflate", nodata=-9999.0) as ds:
        ds.write(np.full((1, 128, 128), -9999.0, dtype=np.float32))
    out["w14_allnd"] = p_allnd

    p_inf = synth_dir / "w14_f32_inf.tif"
    arr_inf = (100.0 + xx9.astype(np.float32) + yy9.astype(np.float32))
    arr_inf[80:96, 64:80] = np.inf
    arr_inf[96:112, 64:80] = -np.inf
    with rasterio.open(p_inf, "w", driver="GTiff", width=128, height=128, count=1, dtype="float32", crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64, compress="deflate") as ds:
        ds.write(arr_inf[None])
    out["w14_inf"] = p_inf

    p_sp = synth_dir / "w14_south_pole_3031.tif"
    with rasterio.open(p_sp, "w", driver="GTiff", width=128, height=128, count=1, dtype="float32", crs="EPSG:3031", transform=from_bounds(-500000.0, -500000.0, 500000.0, 500000.0, 128, 128), tiled=True, blockxsize=64, blockysize=64, compress="deflate") as ds:
        ds.write((1000.0 + xx9.astype(np.float32) * 5.0 + yy9.astype(np.float32) * 3.0)[None])
    out["w14_south_pole"] = p_sp

    p_am = synth_dir / "w14_antimeridian_170_190.tif"
    with rasterio.open(p_am, "w", driver="GTiff", width=128, height=128, count=1, dtype="float32", crs="EPSG:4326", transform=from_bounds(170.0, -10.0, 190.0, 10.0, 128, 128), tiled=True, blockxsize=64, blockysize=64, compress="deflate") as ds:
        ds.write((100.0 + xx9.astype(np.float32) * 2.0 + yy9.astype(np.float32))[None])
    out["w14_antimeridian"] = p_am

    # Wave 16: Photometric=MinIsWhite (uint8 & 1-bit), partial final strips (CONTIG & SEPARATE RGBA), Palette
    def _patch_short_tag(path: Path, tag_id: int, new_val: int) -> None:
        buf = bytearray(path.read_bytes())
        assert buf[:4] == b"II*\x00"
        ifd_off = struct.unpack_from("<I", buf, 4)[0]
        n_ent = struct.unpack_from("<H", buf, ifd_off)[0]
        for i in range(n_ent):
            off = ifd_off + 2 + i * 12
            t, typ, cnt = struct.unpack_from("<HHI", buf, off)
            if t == tag_id and typ == 3 and cnt == 1:
                struct.pack_into("<H", buf, off + 8, new_val)
                path.write_bytes(bytes(buf))
                return
        raise RuntimeError(f"tag {tag_id} not found in {path}")

    p_miw8 = synth_dir / "w16_miniswhite_u8.tif"
    with rasterio.open(p_miw8, "w", driver="GTiff", width=128, height=128, count=1, dtype="uint8", crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64, compress="deflate") as ds:
        ds.write(((xx9 * 2 + yy9) % 256).astype(np.uint8), 1)
    _patch_short_tag(p_miw8, 262, 0)  # Photometric = MinIsWhite
    out["w16_miniswhite_u8"] = p_miw8

    p_miw1 = synth_dir / "w16_miniswhite_1bit.tif"
    with rasterio.open(p_miw1, "w", driver="GTiff", width=128, height=128, count=1, dtype="uint8", nbits=1, crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64, compress="deflate") as ds:
        ds.write((((xx9 // 4) + (yy9 // 8)) % 2).astype(np.uint8), 1)
    _patch_short_tag(p_miw1, 262, 0)  # Photometric = MinIsWhite
    out["w16_miniswhite_1bit"] = p_miw1

    tf_100 = from_bounds(-122.5, 37.5, -122.0, 38.0, 128, 100)
    yy100, xx100 = np.mgrid[0:100, 0:128]
    p_sc = synth_dir / "w16_strip_partial_contig.tif"
    with rasterio.open(p_sc, "w", driver="GTiff", width=128, height=100, count=3, dtype="uint8", crs="EPSG:4326", transform=tf_100, tiled=False, blockysize=32, interleave="pixel", compress="deflate") as ds:
        ds.write(np.stack([((xx100 * 2) % 256).astype(np.uint8), ((yy100 * 2) % 256).astype(np.uint8), ((xx100 + yy100) % 256).astype(np.uint8)]))
    out["w16_strip_partial_contig"] = p_sc

    p_ss = synth_dir / "w16_strip_partial_sep_rgba.tif"
    with rasterio.open(p_ss, "w", driver="GTiff", width=128, height=100, count=4, dtype="uint8", crs="EPSG:4326", transform=tf_100, tiled=False, blockysize=32, interleave="band", compress="deflate", photometric="RGB", alpha="YES") as ds:
        a100 = np.full((100, 128), 255, dtype=np.uint8)
        a100[60:95, 60:100] = 0
        ds.write(np.stack([((xx100 * 2) % 256).astype(np.uint8), ((yy100 * 2) % 256).astype(np.uint8), ((xx100 + yy100) % 256).astype(np.uint8), a100]))
    out["w16_strip_partial_sep_rgba"] = p_ss

    p_pal16 = synth_dir / "w16_palette.tif"
    with rasterio.open(p_pal16, "w", driver="GTiff", width=128, height=128, count=1, dtype="uint8", crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64, compress="deflate") as ds:
        ds.write(((xx9 + yy9) % 8).astype(np.uint8), 1)
        ds.write_colormap(1, {i: (i * 31, (255 - i * 29) % 256, (i * 53) % 256, 255) for i in range(8)})
    out["w16_palette"] = p_pal16

    # Wave 18: GCP-only float32 GeoTIFFs (affine & rotated 9-point) + 3-band float32 COG with scales/offsets
    grad18 = 100.0 + xx9.astype(np.float32) * 2.0 + yy9.astype(np.float32) * 1.5
    p_gcp_aff = synth_dir / "w18_gcp_affine.tif"
    gcps_aff = [
        GroundControlPoint(row=0, col=0, x=-122.5, y=38.0, id="1"),
        GroundControlPoint(row=0, col=128, x=-122.0, y=38.0, id="2"),
        GroundControlPoint(row=128, col=0, x=-122.5, y=37.5, id="3"),
        GroundControlPoint(row=128, col=128, x=-122.0, y=37.5, id="4"),
    ]
    with rasterio.open(p_gcp_aff, "w", driver="GTiff", width=128, height=128, count=1, dtype="float32", gcps=gcps_aff, crs="EPSG:4326", tiled=True, blockxsize=64, blockysize=64, compress="deflate") as ds:
        ds.write(grad18[None])
    out["w18_gcp_aff"] = p_gcp_aff

    p_gcp_rot = synth_dir / "w18_gcp_rotated.tif"
    gcps_rot = []
    gid = 1
    for r in (0, 64, 128):
        for c_i in (0, 64, 128):
            u = (c_i - 64) / 128.0 * 0.5
            v = (64 - r) / 128.0 * 0.5
            gcps_rot.append(GroundControlPoint(row=r, col=c_i, x=-122.25 + u * 0.985 - v * 0.174, y=37.75 + u * 0.174 + v * 0.985, id=str(gid)))
            gid += 1
    with rasterio.open(p_gcp_rot, "w", driver="GTiff", width=128, height=128, count=1, dtype="float32", gcps=gcps_rot, crs="EPSG:4326", tiled=True, blockxsize=64, blockysize=64, compress="deflate") as ds:
        ds.write(grad18[None])
    out["w18_gcp_rot"] = p_gcp_rot

    p_cog18 = synth_dir / "w18_cog3b.tif"
    with rasterio.open(p_cog18, "w", driver="GTiff", width=128, height=128, count=3, dtype="float32", crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64, compress="deflate", nodata=-9999.0) as ds:
        b1 = 10.0 + xx9.astype(np.float32) * 1.5 + yy9.astype(np.float32) * 0.5
        b2 = 20.0 + xx9.astype(np.float32) * 0.5 + yy9.astype(np.float32) * 1.5
        b3 = 30.0 + (xx9 + yy9).astype(np.float32) * 0.8
        b1[80:100, 64:84] = -9999.0
        b2[80:100, 64:84] = -9999.0
        b3[80:100, 64:84] = -9999.0
        ds.write(np.stack([b1, b2, b3]))
        ds.scales = (0.1, 0.2, 0.5)
        ds.offsets = (1.0, 2.0, 3.0)
    out["w18_cog3b"] = p_cog18

    # Wave 19: Custom JSON colormaps (discrete/hex/transparent/interval/linear/negative int16) & built-in algorithms
    p_cat19 = synth_dir / "w19_cat_u8.tif"
    with rasterio.open(p_cat19, "w", driver="GTiff", width=128, height=128, count=1, dtype="uint8", crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64, compress="deflate") as ds:
        ds.write((((xx9 // 16) + (yy9 // 16)) % 4).astype(np.uint8), 1)
    out["w19_cat_u8"] = p_cat19

    p_i16_19 = synth_dir / "w19_neg_i16.tif"
    vals_i16_19 = np.where((xx9 // 32) % 3 == 0, -10, np.where((xx9 // 32) % 3 == 1, 0, 10)).astype(np.int16)
    with rasterio.open(p_i16_19, "w", driver="GTiff", width=128, height=128, count=1, dtype="int16", crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64, compress="deflate") as ds:
        ds.write(vals_i16_19, 1)
    out["w19_neg_i16"] = p_i16_19

    p_dem19 = synth_dir / "w19_dem_f32.tif"
    with rasterio.open(p_dem19, "w", driver="GTiff", width=128, height=128, count=1, dtype="float32", crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64, compress="deflate") as ds:
        ds.write((-160.5 + xx9.astype(np.float32) * 2.0 + yy9.astype(np.float32) * 0.5), 1)
    out["w19_dem_f32"] = p_dem19

    p_rgb19 = synth_dir / "w19_rgb_u8.tif"
    with rasterio.open(p_rgb19, "w", driver="GTiff", width=128, height=128, count=3, dtype="uint8", crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64, compress="deflate") as ds:
        ds.write(np.stack([((xx9 * 4) % 256).astype(np.uint8), ((yy9 * 4) % 256).astype(np.uint8), (((xx9 + yy9) * 3) % 256).astype(np.uint8)]))
    out["w19_rgb_u8"] = p_rgb19

    p_2b19 = synth_dir / "w19_nir_red.tif"
    with rasterio.open(p_2b19, "w", driver="GTiff", width=128, height=128, count=2, dtype="float32", crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64, compress="deflate") as ds:
        ds.write(np.stack([50.0 + xx9.astype(np.float32) * 2.0, 20.0 + yy9.astype(np.float32) * 1.5]))
    out["w19_nir_red"] = p_2b19

    # Wave 20: BigTIFF (LE & BE, Predictor=3), PackBits, non-square tiles (128x32), 4:1 anisotropic pixels, multi-overview nodata override + unscale
    grad20 = 50.0 + xx9.astype(np.float32) * 1.75 + yy9.astype(np.float32) * 1.25
    p_bt_le = synth_dir / "w20_bigtiff_le.tif"
    with rasterio.open(p_bt_le, "w", driver="GTiff", width=128, height=128, count=1, dtype="float32", crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64, compress="deflate", predictor=3, BIGTIFF="YES") as ds:
        ds.write(grad20[None])
    out["w20_bigtiff_le"] = p_bt_le

    p_bt_be = synth_dir / "w20_bigtiff_be.tif"
    with rasterio.open(p_bt_be, "w", driver="GTiff", width=128, height=128, count=1, dtype="float32", crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64, compress="deflate", predictor=3, BIGTIFF="YES", endianness="BIG") as ds:
        ds.write(grad20[None])
    assert p_bt_be.read_bytes()[:4] == b"MM\x00+", "expected Big-Endian BigTIFF header"
    out["w20_bigtiff_be"] = p_bt_be

    p_pb20 = synth_dir / "w20_packbits_rgb.tif"
    pb_rgb = np.stack([((xx9 // 8) * 15 % 256).astype(np.uint8), ((yy9 * 3) % 256).astype(np.uint8), (((xx9 + yy9) * 2) % 256).astype(np.uint8)])
    pb_rgb[:, 80:112, 64:96] = 128  # constant patch inside tile 10/164/396 so PackBits emits genuine RLE repeat runs
    with rasterio.open(p_pb20, "w", driver="GTiff", width=128, height=128, count=3, dtype="uint8", crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64, compress="packbits") as ds:
        ds.write(pb_rgb)
    out["w20_packbits_rgb"] = p_pb20

    p_ns20 = synth_dir / "w20_nonsq_128x32.tif"
    tf_ns20 = from_bounds(-122.5, 37.5, -122.0, 38.0, 256, 96)
    yy_ns20, xx_ns20 = np.mgrid[0:96, 0:256].astype(np.float32)
    with rasterio.open(p_ns20, "w", driver="GTiff", width=256, height=96, count=1, dtype="float32", crs="EPSG:4326", transform=tf_ns20, tiled=True, blockxsize=128, blockysize=32, compress="deflate") as ds:
        ds.write((10.0 + xx_ns20 * 0.5 + yy_ns20 * 3.0)[None])
    out["w20_nonsq"] = p_ns20

    p_an20 = synth_dir / "w20_aniso_4to1.tif"
    tf_an20 = from_bounds(-122.5, 37.5, -122.0, 38.0, 256, 64)
    yy_an20, xx_an20 = np.mgrid[0:64, 0:256].astype(np.float32)
    with rasterio.open(p_an20, "w", driver="GTiff", width=256, height=64, count=1, dtype="float32", crs="EPSG:4326", transform=tf_an20, tiled=True, blockxsize=64, blockysize=64, compress="deflate") as ds:
        ds.write((20.0 + xx_an20 * 0.75 + yy_an20 * 4.0)[None])
    out["w20_aniso"] = p_an20

    p_ov20 = synth_dir / "w20_multi_ov_unscale.tif"
    tf_512 = from_bounds(-122.5, 37.5, -122.0, 38.0, 512, 512)
    yy5, xx5 = np.mgrid[0:512, 0:512]
    arr5 = (100 + (xx5 // 2) + (yy5 // 4)).astype(np.int16)
    arr5[300:360, 240:320] = -9999
    arr5[360:400, 240:320] = 200
    with rasterio.open(p_ov20, "w", driver="GTiff", width=512, height=512, count=1, dtype="int16", crs="EPSG:4326", transform=tf_512, tiled=True, blockxsize=128, blockysize=128, compress="deflate", nodata=-9999) as ds:
        ds.write(arr5[None])
        ds.scales = (0.1,)
        ds.offsets = (10.0,)
        ds.build_overviews([2, 4], Resampling.nearest)
    out["w20_multi_ov"] = p_ov20

    # Wave 21: Heterogeneous multi-CRS / multi-resolution / multi-dtype STAC item
    p_w21_red = synth_dir / "w21_red_4326_u16.tif"
    red21 = (500 + xx9 * 10 + yy9 * 5).astype(np.uint16)
    red21[85:100, 70:85] = 0  # nodata hole inside tile 10/164/396
    with rasterio.open(p_w21_red, "w", driver="GTiff", width=128, height=128, count=1, dtype="uint16", crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64, compress="deflate", nodata=0) as ds:
        ds.write(red21, 1)

    t_utm21 = Transformer.from_crs("EPSG:4326", "EPSG:32610", always_xy=True)
    ux0, uy0 = t_utm21.transform(-122.55, 37.45)
    ux1, uy1 = t_utm21.transform(-121.95, 38.05)
    yy256, xx256 = np.mgrid[0:256, 0:256]
    p_w21_nir = synth_dir / "w21_nir_32610_f32.tif"
    with rasterio.open(p_w21_nir, "w", driver="GTiff", width=256, height=256, count=1, dtype="float32", crs="EPSG:32610", transform=from_bounds(ux0, uy0, ux1, uy1, 256, 256), tiled=True, blockxsize=64, blockysize=64, compress="deflate", nodata=-9999.0) as ds:
        ds.write((1200.0 + xx256.astype(np.float32) * 8.0 + yy256.astype(np.float32) * 4.0), 1)

    p_w21_stac = synth_dir / "w21_hetero_item.json"
    p_w21_stac.write_text(json.dumps({
        "type": "Feature",
        "stac_version": "1.0.0",
        "id": "w21-hetero",
        "geometry": {"type": "Polygon", "coordinates": [[[-122.5, 37.5], [-122.0, 37.5], [-122.0, 38.0], [-122.5, 38.0], [-122.5, 37.5]]]},
        "bbox": [-122.5, 37.5, -122.0, 38.0],
        "properties": {"datetime": "2024-06-01T00:00:00Z"},
        "links": [],
        "assets": {
            "red": {"href": str(p_w21_red), "type": "image/tiff; application=geotiff"},
            "nir": {"href": str(p_w21_nir), "type": "image/tiff; application=geotiff"},
        },
    }))
    out["w21_stac"] = p_w21_stac

    # Wave 22: 2-band float32 straddling 0 inside tile 10/164/396 (b1 in [-20, 23.5], b2 in [-9, 18.5]) for trig, nested where & domain-error checks
    p_w22_2b = synth_dir / "w22_f32_2b.tif"
    with rasterio.open(p_w22_2b, "w", driver="GTiff", width=128, height=128, count=2, dtype="float32", crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64, compress="deflate") as ds:
        ds.write(np.stack([(-40.0 + xx9.astype(np.float32) * 0.5), (-45.0 + yy9.astype(np.float32) * 0.5)]))
    out["w22_f32_2b"] = p_w22_2b

    # Wave 23: Categorical uint8 (with nodata=255), negative int16 (with nodata=-9999), and 3-band float32 for stats params & per-band rescale
    p_w23_cat = synth_dir / "w23_cat_u8.tif"
    cat23 = (((xx9 // 16) + (yy9 // 16)) % 4).astype(np.uint8)
    cat23[0:16, 0:16] = 255
    with rasterio.open(p_w23_cat, "w", driver="GTiff", width=128, height=128, count=1, dtype="uint8", crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64, compress="deflate", nodata=255) as ds:
        ds.write(cat23, 1)
    out["w23_cat_u8"] = p_w23_cat

    p_w23_neg = synth_dir / "w23_cat_i16.tif"
    neg23 = np.where((xx9 // 32) % 3 == 0, -10, np.where((xx9 // 32) % 3 == 1, 0, 10)).astype(np.int16)
    neg23[0:16, 0:16] = -9999
    with rasterio.open(p_w23_neg, "w", driver="GTiff", width=128, height=128, count=1, dtype="int16", crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64, compress="deflate", nodata=-9999) as ds:
        ds.write(neg23, 1)
    out["w23_cat_i16"] = p_w23_neg

    p_w23_3b = synth_dir / "w23_f32_3b.tif"
    with rasterio.open(p_w23_3b, "w", driver="GTiff", width=128, height=128, count=3, dtype="float32", crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64, compress="deflate") as ds:
        ds.write(np.stack([
            (10.0 + xx9.astype(np.float32) * 1.5 + yy9.astype(np.float32) * 0.5),
            (40.0 + xx9.astype(np.float32) * 0.8 + yy9.astype(np.float32) * 1.2),
            (80.0 + (xx9 + yy9).astype(np.float32) * 0.9),
        ]))
        ds.scales = (0.1, 0.2, 0.5)
        ds.offsets = (1.0, 2.0, 3.0)
    out["w23_f32_3b"] = p_w23_3b

    # Wave 24: Regional EPSG:4326 COGs for non-Mercator OGC TileMatrixSets
    grad24 = 100.0 + xx9.astype(np.float32) * 2.0 + yy9.astype(np.float32) * 1.5
    for key, bounds in (
        ("w24_eu", (9.0, 47.0, 12.0, 50.0)),
        ("w24_nz", (171.0, -43.5, 174.0, -40.5)),
        ("w24_ca", (-100.0, 50.0, -94.0, 56.0)),
    ):
        p_reg = synth_dir / f"{key}.tif"
        with rasterio.open(p_reg, "w", driver="GTiff", width=128, height=128, count=1, dtype="float32", crs="EPSG:4326", transform=from_bounds(*bounds, 128, 128), tiled=True, blockxsize=64, blockysize=64, compress="deflate") as ds:
            ds.write(grad24[None])
        out[key] = p_reg

    # Wave 25: High-latitude 84°N (EPSG:3413 -> WebMercatorQuad) & Global (-180..180, -85..85) zoom-0 COGs
    t_3413_25 = Transformer.from_crs("EPSG:4326", "EPSG:3413", always_xy=True)
    px0, py0 = t_3413_25.transform(10.0, 82.5)
    px1, py1 = t_3413_25.transform(25.0, 84.5)
    p_w25_hi = synth_dir / "w25_highlat_84n_3413.tif"
    with rasterio.open(
        p_w25_hi, "w", driver="GTiff", width=128, height=128, count=1, dtype="float32",
        crs="EPSG:3413", transform=from_bounds(min(px0, px1) - 100000, min(py0, py1) - 100000, max(px0, px1) + 100000, max(py0, py1) + 100000, 128, 128),
        tiled=True, blockxsize=64, blockysize=64, compress="deflate",
    ) as ds:
        ds.write(grad24[None])
    out["w25_highlat_84n"] = p_w25_hi

    p_w25_glob = synth_dir / "w25_global_4326.tif"
    yy64, xx128g = np.mgrid[0:64, 0:128].astype(np.float32)
    with rasterio.open(
        p_w25_glob, "w", driver="GTiff", width=128, height=64, count=1, dtype="float32",
        crs="EPSG:4326", transform=from_bounds(-180.0, -85.0, 180.0, 85.0, 128, 64),
        tiled=True, blockxsize=64, blockysize=64, compress="deflate",
    ) as ds:
        ds.write((50.0 + xx128g * 1.5 + yy64 * 2.0)[None])
    out["w25_global"] = p_w25_glob

    # Wave 27: 100%-opaque RGB COG covering all of tile 10/164/396 for auto-format (opaque -> JPEG) & @2x Retina checks
    p_w27_op = synth_dir / "w27_opaque_rgb.tif"
    with rasterio.open(
        p_w27_op, "w", driver="GTiff", width=128, height=128, count=3, dtype="uint8",
        crs="EPSG:4326", transform=from_bounds(-122.5, 37.3, -121.9, 37.8, 128, 128),
        tiled=True, blockxsize=64, blockysize=64, compress="deflate",
    ) as ds:
        ds.write(np.stack([((xx9 * 2) % 256).astype(np.uint8), ((yy9 * 2) % 256).astype(np.uint8), ((xx9 + yy9) % 256).astype(np.uint8)]))
    out["w27_opaque_rgb"] = p_w27_op

    # Wave 29: Unprobed TIFF codecs/predictors (Grayscale JPEG, LZW P2 u16/i16, LZW P3 f32, ZSTD P2 u16, YCbCr JPEG + internal mask) & STAC asset_as_band=true
    p_w29_gj = synth_dir / "w29_gray_jpeg.tif"
    with rasterio.open(
        p_w29_gj, "w", driver="GTiff", width=128, height=128, count=1, dtype="uint8",
        crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64,
        compress="JPEG", jpeg_quality=90,
    ) as ds:
        ds.write(((xx9 * 2 + yy9) % 256).astype(np.uint8), 1)
    out["w29_gray_jpeg"] = p_w29_gj

    p_w29_lzw_u16 = synth_dir / "w29_lzw_p2_u16.tif"
    u16_w29 = (xx9.astype(np.uint32) * 300 + yy9.astype(np.uint32) * 150 + 500).astype(np.uint16)
    with rasterio.open(
        p_w29_lzw_u16, "w", driver="GTiff", width=128, height=128, count=1, dtype="uint16",
        crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64,
        compress="LZW", predictor=2,
    ) as ds:
        ds.write(u16_w29, 1)
    out["w29_lzw_p2_u16"] = p_w29_lzw_u16

    p_w29_lzw_i16 = synth_dir / "w29_lzw_p2_i16.tif"
    i16_w29 = ((xx9.astype(np.int32) - 64) * 200 + (yy9.astype(np.int32) - 64) * 100).astype(np.int16)
    i16_w29[80:110, 60:100] = -32768  # interior nodata hole inside tile 10/164/396
    with rasterio.open(
        p_w29_lzw_i16, "w", driver="GTiff", width=128, height=128, count=1, dtype="int16",
        crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64,
        compress="LZW", predictor=2, nodata=-32768,
    ) as ds:
        ds.write(i16_w29, 1)
    out["w29_lzw_p2_i16"] = p_w29_lzw_i16

    p_w29_lzw_f32 = synth_dir / "w29_lzw_p3_f32.tif"
    f32_w29 = ((xx9 - 64) * 1.375 + (yy9 - 64) * 0.625).astype(np.float32)
    with rasterio.open(
        p_w29_lzw_f32, "w", driver="GTiff", width=128, height=128, count=1, dtype="float32",
        crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64,
        compress="LZW", predictor=3,
    ) as ds:
        ds.write(f32_w29, 1)
    out["w29_lzw_p3_f32"] = p_w29_lzw_f32

    p_w29_zstd_u16 = synth_dir / "w29_zstd_p2_u16.tif"
    with rasterio.open(
        p_w29_zstd_u16, "w", driver="GTiff", width=128, height=128, count=1, dtype="uint16",
        crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64,
        compress="ZSTD", predictor=2,
    ) as ds:
        ds.write(u16_w29, 1)
    out["w29_zstd_p2_u16"] = p_w29_zstd_u16

    p_w29_jmsk = synth_dir / "w29_ycbcr_jpeg_mask.tif"
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rasterio.open(
            p_w29_jmsk, "w", driver="GTiff", width=128, height=128, count=3, dtype="uint8",
            crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64,
            compress="JPEG", photometric="YCBCR", jpeg_quality=90,
        ) as ds:
            ds.write(np.stack([((xx9 * 2) % 256).astype(np.uint8), ((yy9 * 2) % 256).astype(np.uint8), ((xx9 + yy9) % 256).astype(np.uint8)]))
            msk_j = np.full((128, 128), 255, dtype=np.uint8)
            msk_j[80:110, 60:100] = 0
            ds.write_mask(msk_j)
    out["w29_ycbcr_jpeg_mask"] = p_w29_jmsk

    p_w29_red = synth_dir / "w29_stac_red.tif"
    p_w29_nir = synth_dir / "w29_stac_nir.tif"
    with rasterio.open(
        p_w29_red, "w", driver="GTiff", width=128, height=128, count=1, dtype="float32",
        crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64, compress="deflate",
    ) as ds:
        ds.write((xx9.astype(np.float32) + 10.0), 1)
    with rasterio.open(
        p_w29_nir, "w", driver="GTiff", width=128, height=128, count=1, dtype="float32",
        crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64, compress="deflate",
    ) as ds:
        ds.write((yy9.astype(np.float32) * 2.0 + 20.0), 1)
    p_w29_stac = synth_dir / "w29_stac_ab.json"
    p_w29_stac.write_text(json.dumps({
        "type": "Feature", "stac_version": "1.0.0", "id": "w29-ab",
        "geometry": {"type": "Polygon", "coordinates": [[[-122.5, 37.5], [-122.0, 37.5], [-122.0, 38.0], [-122.5, 38.0], [-122.5, 37.5]]]},
        "bbox": [-122.5, 37.5, -122.0, 38.0], "properties": {"datetime": "2024-06-01T00:00:00Z"}, "links": [],
        "assets": {"red": {"href": str(p_w29_red)}, "nir": {"href": str(p_w29_nir)}},
    }))
    out["w29_stac_ab"] = p_w29_stac

    # Wave 31: Rescale/color_formula/BGR-reorder/non-square preview fixtures
    p_w31_f32 = synth_dir / "w31_f32_neg_pos.tif"
    f32_w31 = ((xx9.astype(np.float32) - 64.0) * 3.0 + (yy9.astype(np.float32) - 64.0) * 1.5)
    f32_w31[80:100, 60:90] = -9999.0  # interior nodata hole inside tile 10/164/396
    with rasterio.open(
        p_w31_f32, "w", driver="GTiff", width=128, height=128, count=1, dtype="float32",
        crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64,
        compress="deflate", nodata=-9999.0,
    ) as ds:
        ds.write(f32_w31, 1)
    out["w31_f32"] = p_w31_f32

    p_w31_rgb = synth_dir / "w31_rgb_asym.tif"
    rgb_w31 = np.stack([
        np.maximum(((xx9 * 2) % 255).astype(np.uint8), 1),
        np.maximum(((yy9 * 2) % 255).astype(np.uint8), 1),
        np.maximum((((127 - xx9) * 2) % 255).astype(np.uint8), 1),
    ], axis=0)
    rgb_w31[:, 80:100, 60:90] = 0
    with rasterio.open(
        p_w31_rgb, "w", driver="GTiff", width=128, height=128, count=3, dtype="uint8",
        crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64,
        compress="deflate", nodata=0,
    ) as ds:
        ds.write(rgb_w31)
    out["w31_rgb"] = p_w31_rgb

    # Wave 32: DEM + STAC fixtures for non-256 tilesize (128/512) cubic/hillshade and STAC OGC (/stac/map, tilejson, WMTS)
    p_w32_dem = synth_dir / "w32_dem.tif"
    dem_w32 = (100.0 + xx9.astype(np.float32) * 4.0 + yy9.astype(np.float32) * 2.5 + 30.0 * np.sin(xx9 / 10.0)).astype(np.float32)
    with rasterio.open(
        p_w32_dem, "w", driver="GTiff", width=128, height=128, count=1, dtype="float32",
        crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64, compress="deflate",
    ) as ds:
        ds.write(dem_w32, 1)
    out["w32_dem"] = p_w32_dem

    p_w32_stac = synth_dir / "w32_stac.json"
    p_w32_stac.write_text(json.dumps({
        "type": "Feature", "stac_version": "1.0.0", "id": "w32-stac",
        "geometry": {"type": "Polygon", "coordinates": [[[-122.5, 37.5], [-122.0, 37.5], [-122.0, 38.0], [-122.5, 38.0], [-122.5, 37.5]]]},
        "bbox": [-122.5, 37.5, -122.0, 38.0], "properties": {"datetime": "2024-06-01T00:00:00Z"}, "links": [],
        "assets": {"dem": {"href": str(p_w32_dem)}},
    }))
    out["w32_stac"] = p_w32_stac

    # Wave 33: US Survey Feet (EPSG:2227), Intl Feet (EPSG:2913), Ellipsoidal World Mercator (EPSG:3395),
    # NAD83 UTM 10N (EPSG:26910), Arctic Polar Stereographic (EPSG:3995), and PlanarConfiguration=SEPARATE f32 ZSTD P3
    grad_w33 = (100.0 + xx9.astype(np.float32) * 3.0 + yy9.astype(np.float32) * 2.0).astype(np.float32)
    for cname, crs_str, wgs_bb in [
        ("w33_ca_usft_2227", "EPSG:2227", (-122.4, 37.6, -122.1, 37.9)),
        ("w33_or_intlft_2913", "EPSG:2913", (-122.8, 45.4, -122.5, 45.7)),
        ("w33_merc_3395", "EPSG:3395", (-122.5, 37.5, -122.0, 38.0)),
        ("w33_nad83_26910", "EPSG:26910", (-122.5, 37.5, -122.0, 38.0)),
        ("w33_arctic_3995", "EPSG:3995", (10.0, 78.0, 16.0, 80.0)),
    ]:
        p_c = synth_dir / f"{cname}.tif"
        bb_c = transform_bounds("EPSG:4326", crs_str, *wgs_bb, densify_pts=21)
        with rasterio.open(
            p_c, "w", driver="GTiff", width=128, height=128, count=1, dtype="float32",
            crs=crs_str, transform=from_bounds(*bb_c, 128, 128),
            tiled=True, blockxsize=64, blockysize=64, compress="deflate",
        ) as ds:
            ds.write(grad_w33, 1)
        out[cname] = p_c

    p_w33_sep = synth_dir / "w33_planar_f32_zstd_p3.tif"
    with rasterio.open(
        p_w33_sep, "w", driver="GTiff", width=128, height=128, count=3, dtype="float32",
        crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64,
        compress="ZSTD", predictor=3, interleave="BAND",
    ) as ds:
        ds.write(np.stack([grad_w33, grad_w33 * 0.5 + 10.0, grad_w33 * 0.25 - 5.0]))
    out["w33_planar_f32_zstd_p3"] = p_w33_sep

    # Wave 34: int64 & uint64 BigTIFF COGs + multi-band STAC asset for inline pipe syntax & dst_crs/coord_crs bbox
    p_w34_i64 = synth_dir / "w34_i64.tif"
    i64_w34 = ((xx9.astype(np.int64) - 64) * 100000 + (yy9.astype(np.int64) - 64) * 50000)
    with rasterio.open(
        p_w34_i64, "w", driver="GTiff", width=128, height=128, count=1, dtype="int64",
        crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64, compress="deflate",
    ) as ds:
        ds.write(i64_w34, 1)
    out["w34_i64"] = p_w34_i64

    p_w34_u64 = synth_dir / "w34_u64.tif"
    u64_w34 = (xx9.astype(np.uint64) * 100000 + yy9.astype(np.uint64) * 50000 + 1000)
    with rasterio.open(
        p_w34_u64, "w", driver="GTiff", width=128, height=128, count=1, dtype="uint64",
        crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64, compress="deflate",
    ) as ds:
        ds.write(u64_w34, 1)
    out["w34_u64"] = p_w34_u64

    p_w34_mb = synth_dir / "w34_multi3b.tif"
    with rasterio.open(
        p_w34_mb, "w", driver="GTiff", width=128, height=128, count=3, dtype="float32",
        crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64, compress="deflate",
    ) as ds:
        ds.write(np.stack([
            (xx9.astype(np.float32) + 10.0),
            (yy9.astype(np.float32) * 2.0 + 20.0),
            ((xx9 + yy9).astype(np.float32) + 5.0),
        ]))
    out["w34_multi3b"] = p_w34_mb

    p_w34_stac = synth_dir / "w34_stac_pipe.json"
    p_w34_stac.write_text(json.dumps({
        "type": "Feature", "stac_version": "1.0.0", "id": "w34-pipe",
        "geometry": {"type": "Polygon", "coordinates": [[[-122.5, 37.5], [-122.0, 37.5], [-122.0, 38.0], [-122.5, 38.0], [-122.5, 37.5]]]},
        "bbox": [-122.5, 37.5, -122.0, 38.0], "properties": {"datetime": "2024-06-01T00:00:00Z"}, "links": [],
        "assets": {"multi": {"href": str(p_w34_mb)}},
    }))
    out["w34_stac_pipe"] = p_w34_stac

    # Wave 35: Scaled int16 DEM (scale=0.1, offset=500.0), heterogeneous per-band scaled 2-band int16
    # (Landsat/Sentinel-2 style), and tagless float32 with NaN hole for ?nodata=nan override
    p_w35_sdem = synth_dir / "w35_scaled_dem.tif"
    raw_dem_w35 = (xx9.astype(np.int32) * 50 + yy9.astype(np.int32) * 25).astype(np.int16)
    raw_dem_w35[80:100, 60:90] = -32768
    with rasterio.open(
        p_w35_sdem, "w", driver="GTiff", width=128, height=128, count=1, dtype="int16",
        crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64,
        compress="deflate", nodata=-32768,
    ) as ds:
        ds.write(raw_dem_w35, 1)
        ds.scales = (0.1,)
        ds.offsets = (500.0,)
    out["w35_scaled_dem"] = p_w35_sdem

    p_w35_s2b = synth_dir / "w35_scaled_2b.tif"
    b1_w35 = (xx9.astype(np.int32) * 100 + 2000).astype(np.int16)
    b2_w35 = (yy9.astype(np.int32) * 50 + 5000).astype(np.int16)
    with rasterio.open(
        p_w35_s2b, "w", driver="GTiff", width=128, height=128, count=2, dtype="int16",
        crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64,
        compress="deflate", nodata=-9999,
    ) as ds:
        ds.write(np.stack([b1_w35, b2_w35]))
        ds.scales = (0.0001, 0.0002)
        ds.offsets = (-0.1, -0.2)
    out["w35_scaled_2b"] = p_w35_s2b

    p_w35_nan = synth_dir / "w35_f32_nan_notag.tif"
    f32_nan_w35 = (10.0 + xx9.astype(np.float32) * 2.0 + yy9.astype(np.float32)).astype(np.float32)
    f32_nan_w35[80:100, 60:90] = np.nan
    with rasterio.open(
        p_w35_nan, "w", driver="GTiff", width=128, height=128, count=1, dtype="float32",
        crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64, compress="deflate",
    ) as ds:
        ds.write(f32_nan_w35, 1)
    out["w35_f32_nan_notag"] = p_w35_nan

    # Wave 36: 5-band multispectral COG (B, G, R, NIR, SWIR), DEM for /bbox & /feature algorithms, and STAC item
    p_w36_5b = synth_dir / "w36_5band.tif"
    b5_w36 = np.stack([
        (10.0 + xx9 * 0.5).astype(np.float32),
        (20.0 + yy9 * 0.5).astype(np.float32),
        (30.0 + xx9 * 0.8 + yy9 * 0.2).astype(np.float32),
        (80.0 + yy9 * 1.2 + xx9 * 0.3).astype(np.float32),
        (50.0 + (xx9 + yy9) * 0.4).astype(np.float32),
    ])
    with rasterio.open(
        p_w36_5b, "w", driver="GTiff", width=128, height=128, count=5, dtype="float32",
        crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64, compress="deflate",
    ) as ds:
        ds.write(b5_w36)
    out["w36_5band"] = p_w36_5b

    p_w36_dem = synth_dir / "w36_dem.tif"
    dem_w36 = (200.0 + xx9 * 5.0 + yy9 * 3.0 + 40.0 * np.sin(xx9 / 8.0)).astype(np.float32)
    with rasterio.open(
        p_w36_dem, "w", driver="GTiff", width=128, height=128, count=1, dtype="float32",
        crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64, compress="deflate",
    ) as ds:
        ds.write(dem_w36, 1)
    out["w36_dem"] = p_w36_dem

    p_w36_stac = synth_dir / "w36_stac5b.json"
    p_w36_stac.write_text(json.dumps({
        "type": "Feature", "stac_version": "1.0.0", "id": "w36-5b",
        "geometry": {"type": "Polygon", "coordinates": [[[-122.5, 37.5], [-122.0, 37.5], [-122.0, 38.0], [-122.5, 38.0], [-122.5, 37.5]]]},
        "bbox": [-122.5, 37.5, -122.0, 38.0], "properties": {"datetime": "2024-06-01T00:00:00Z"}, "links": [],
        "assets": {"ms": {"href": str(p_w36_5b)}},
    }))
    out["w36_stac5b"] = p_w36_stac

    # Wave 37: float32 COG for projected GeoJSON coord_crs (EPSG:3857, EPSG:32610), US-feet /point (EPSG:2227), /cog/bounds?crs=, /cog/info?crs=, /cog/stac
    p_w37_cog = synth_dir / "w37_cog.tif"
    arr_w37 = (10.0 + xx9.astype(np.float32) * 2.5 + yy9.astype(np.float32) * 1.5).astype(np.float32)
    with rasterio.open(
        p_w37_cog, "w", driver="GTiff", width=128, height=128, count=1, dtype="float32",
        crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64, compress="deflate",
    ) as ds:
        ds.write(arr_w37, 1)
    out["w37_cog"] = p_w37_cog

    # Wave 38: High-precision uint32 (> 2^24) & signed int32 (< -2^24 + INT32_MIN nodata hole) to catch any f32 truncation
    p_w38_u32 = synth_dir / "w38_u32_hi.tif"
    u32_w38 = (np.uint32(2_000_000_000) + xx9.astype(np.uint32) + yy9.astype(np.uint32) * np.uint32(3))
    with rasterio.open(
        p_w38_u32, "w", driver="GTiff", width=128, height=128, count=1, dtype="uint32",
        crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64, compress="deflate",
    ) as ds:
        ds.write(u32_w38, 1)
    out["w38_u32_hi"] = p_w38_u32

    p_w38_i32 = synth_dir / "w38_i32_hi.tif"
    i32_w38 = (np.int32(-2_000_000_000) + xx9.astype(np.int32) + yy9.astype(np.int32) * np.int32(3))
    i32_w38[80:100, 60:90] = -2147483648
    with rasterio.open(
        p_w38_i32, "w", driver="GTiff", width=128, height=128, count=1, dtype="int32",
        crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64,
        compress="deflate", nodata=-2147483648,
    ) as ds:
        ds.write(i32_w38, 1)
    out["w38_i32_hi"] = p_w38_i32

    # Wave 39: High-precision float64 (> 1e9, where f32 ULP=64 wipes out 0.001/0.003 ramps) + nodata hole
    p_w39_f64 = synth_dir / "w39_f64_hi.tif"
    f64_w39 = (1_000_000_000.0 + xx9.astype(np.float64) * 0.001 + yy9.astype(np.float64) * 0.003)
    f64_w39[80:100, 60:90] = -9999.0
    with rasterio.open(
        p_w39_f64, "w", driver="GTiff", width=128, height=128, count=1, dtype="float64",
        crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64,
        compress="deflate", predictor=3, nodata=-9999.0,
    ) as ds:
        ds.write(f64_w39, 1)
    out["w39_f64_hi"] = p_w39_f64

    # Wave 40: 2:1 non-square 256x128 DEM for unsized /preview, /bbox, /feature (max_size/width/height) & algorithm=aspect
    p_w40_rect = synth_dir / "w40_rect_dem.tif"
    yy40, xx40 = np.mgrid[0:128, 0:256]
    dem_w40 = (200.0 + xx40.astype(np.float32) * 3.0 + yy40.astype(np.float32) * 4.0 + 50.0 * np.sin(xx40 / 15.0) * np.cos(yy40 / 12.0)).astype(np.float32)
    with rasterio.open(
        p_w40_rect, "w", driver="GTiff", width=256, height=128, count=1, dtype="float32",
        crs="EPSG:4326", transform=from_bounds(-122.5, 37.5, -121.5, 38.0, 256, 128),
        tiled=True, blockxsize=64, blockysize=64, compress="deflate",
    ) as ds:
        ds.write(dem_w40, 1)
    out["w40_rect_dem"] = p_w40_rect

    # Wave 41: 4-band float32 (even band count for median/std/var/min/max/mean/sum) & 3-band high-value uint16 (sum > 65535 -> uint64)
    p_w41_4b = synth_dir / "w41_f32_4b.tif"
    b4_w41 = np.stack([
        (10.0 + xx9 * 0.5).astype(np.float32),
        (50.0 + yy9 * 0.8).astype(np.float32),
        (20.0 + (xx9 + yy9) * 0.4).astype(np.float32),
        (120.0 - xx9 * 0.3 + yy9 * 0.2).astype(np.float32),
    ])
    with rasterio.open(
        p_w41_4b, "w", driver="GTiff", width=128, height=128, count=4, dtype="float32",
        crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64, compress="deflate",
    ) as ds:
        ds.write(b4_w41)
    out["w41_f32_4b"] = p_w41_4b

    p_w41_u16 = synth_dir / "w41_u16_3b_hi.tif"
    b3_w41 = np.stack([
        (30000 + xx9.astype(np.uint32) * 100).astype(np.uint16),
        (35000 + yy9.astype(np.uint32) * 100).astype(np.uint16),
        (40000 + (xx9 + yy9).astype(np.uint32) * 50).astype(np.uint16),
    ])
    with rasterio.open(
        p_w41_u16, "w", driver="GTiff", width=128, height=128, count=3, dtype="uint16",
        crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64, compress="deflate",
    ) as ds:
        ds.write(b3_w41)
    out["w41_u16_3b_hi"] = p_w41_u16

    # Wave 42: IEEE-754 float16 (NBITS=16, SampleFormat=3) COGs — Predictor=1 and Predictor=3 (2-plane byte shuffle)
    f16_arr = (-10.0 + xx9 * 0.25 + yy9 * 0.125).astype(np.float32)
    f16_arr[80:110, 60:90] = -1024.0
    p_w42_f16 = synth_dir / "w42_f16.tif"
    with rasterio.open(
        p_w42_f16, "w", driver="GTiff", width=128, height=128, count=1, dtype="float32",
        crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64,
        compress="deflate", predictor=1, nbits=16, nodata=-1024.0,
    ) as ds:
        ds.write(f16_arr, 1)
    out["w42_f16"] = p_w42_f16

    p_w42_f16_p3 = synth_dir / "w42_f16_p3.tif"
    with rasterio.open(
        p_w42_f16_p3, "w", driver="GTiff", width=128, height=128, count=1, dtype="float32",
        crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64,
        compress="deflate", predictor=3, nbits=16, nodata=-1024.0,
    ) as ds:
        ds.write(f16_arr, 1)
    out["w42_f16_p3"] = p_w42_f16_p3

    # Wave 43: 3-band RGB ?nodata=0 (pure [0,0,0] hole + partial [0,120,200] valid region),
    # multi-overview float32 (dual resampling= + reproject=), and stripped (non-tiled) 3-band TIFF with internal overviews
    tf_w43 = from_bounds(-122.5, 37.5, -122.0, 38.0, 256, 256)
    yy43, xx43 = np.mgrid[0:256, 0:256]
    r43 = np.clip(40 + xx43 // 2, 1, 250).astype(np.uint8)
    g43 = np.clip(60 + yy43 // 2, 1, 250).astype(np.uint8)
    b43 = np.clip(80 + (xx43 + yy43) // 4, 1, 250).astype(np.uint8)
    # Pure [0,0,0] hole inside tile 10/164/396 (rows 144..256, cols 80..256)
    r43[160:200, 110:150] = 0
    g43[160:200, 110:150] = 0
    b43[160:200, 110:150] = 0
    # Partial [0,120,200] (only R=0): valid under GDAL dataset_mask all-band nodata semantics
    r43[200:240, 150:200] = 0
    g43[200:240, 150:200] = 120
    b43[200:240, 150:200] = 200

    p_w43_rgb = synth_dir / "w43_rgb_nodata.tif"
    with rasterio.open(
        p_w43_rgb, "w", driver="GTiff", width=256, height=256, count=3, dtype="uint8",
        crs="EPSG:4326", transform=tf_w43, tiled=True, blockxsize=64, blockysize=64, compress="deflate",
    ) as ds:
        ds.write(np.stack([r43, g43, b43]))
    out["w43_rgb_nodata"] = p_w43_rgb

    p_w43_ovr = synth_dir / "w43_multi_ovr.tif"
    dem43 = (100.0 + xx43 * 2.0 + yy43 * 1.5 + 30.0 * np.sin(xx43 * 0.2)).astype(np.float32)
    with rasterio.open(
        p_w43_ovr, "w", driver="GTiff", width=256, height=256, count=1, dtype="float32",
        crs="EPSG:4326", transform=tf_w43, tiled=True, blockxsize=64, blockysize=64, compress="deflate",
    ) as ds:
        ds.write(dem43, 1)
        ds.build_overviews([2, 4], Resampling.average)
    out["w43_multi_ovr"] = p_w43_ovr

    p_w43_strip = synth_dir / "w43_stripped_ovr.tif"
    with rasterio.open(
        p_w43_strip, "w", driver="GTiff", width=256, height=256, count=3, dtype="uint8",
        crs="EPSG:4326", transform=tf_w43, tiled=False, blockysize=16, compress="deflate",
    ) as ds:
        ds.write(np.stack([r43, g43, b43]))
        ds.build_overviews([2, 4], Resampling.average)
    out["w43_stripped_ovr"] = p_w43_strip

    # Wave 45: unrescaled 16-bit uint16 PNG (with nodata=0 hole), categorical uint8 (classes 1/2/3 + nodata=0)
    # for discrete RGBA JSON colormap, and float32 DEM for ceil/floor/cast(output_dtype)/terrainrgb/contours/expr+alg
    p_w45_u16 = synth_dir / "w45_u16.tif"
    u16_45 = (1000 + xx9.astype(np.uint32) * 300 + yy9.astype(np.uint32) * 200).astype(np.uint16)
    u16_45[80:110, 60:90] = 0
    with rasterio.open(
        p_w45_u16, "w", driver="GTiff", width=128, height=128, count=1, dtype="uint16",
        crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64, compress="deflate", nodata=0,
    ) as ds:
        ds.write(u16_45, 1)
    out["w45_u16"] = p_w45_u16

    p_w45_cat = synth_dir / "w45_cat.tif"
    cat_45 = np.where(xx9 < 64, 1, np.where(yy9 < 96, 2, 3)).astype(np.uint8)
    cat_45[80:100, 60:80] = 0
    with rasterio.open(
        p_w45_cat, "w", driver="GTiff", width=128, height=128, count=1, dtype="uint8",
        crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64, compress="deflate", nodata=0,
    ) as ds:
        ds.write(cat_45, 1)
    out["w45_cat"] = p_w45_cat

    p_w45_dem = synth_dir / "w45_dem.tif"
    dem_45 = (100.25 + xx9 * 3.7 + yy9 * 2.3).astype(np.float32)
    with rasterio.open(
        p_w45_dem, "w", driver="GTiff", width=128, height=128, count=1, dtype="float32",
        crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64, compress="deflate",
    ) as ds:
        ds.write(dem_45, 1)
    out["w45_dem"] = p_w45_dem

    # Wave 46: LERC / LERC_DEFLATE (int16) / LERC_ZSTD (lossy float32) COGs + 6 projected CRS families
    # (Conus Albers 5070, Lambert-93 2154, LAEA Europe 3035, EASE-Grid 6933, Antarctic Polar Stereo 3031, British National Grid 27700)
    a46 = (100.0 + xx9 * 2.5 + yy9 * 1.5).astype(np.float32)
    a46_hole = a46.copy()
    a46_hole[80:110, 60:90] = -9999.0
    p_w46_lerc = synth_dir / "w46_lerc_f32.tif"
    with rasterio.open(
        p_w46_lerc, "w", driver="GTiff", width=128, height=128, count=1, dtype="float32",
        crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64,
        compress="lerc", max_z_error=0.0, nodata=-9999.0,
    ) as ds:
        ds.write(a46_hole, 1)
    out["w46_lerc_f32"] = p_w46_lerc

    i16_46 = (-500 + xx9.astype(np.int32) * 20 + yy9.astype(np.int32) * 15).astype(np.int16)
    i16_46[80:110, 60:90] = -32768
    p_w46_lerc_def = synth_dir / "w46_lerc_deflate_i16.tif"
    with rasterio.open(
        p_w46_lerc_def, "w", driver="GTiff", width=128, height=128, count=1, dtype="int16",
        crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64,
        compress="lerc_deflate", nodata=-32768,
    ) as ds:
        ds.write(i16_46, 1)
    out["w46_lerc_deflate_i16"] = p_w46_lerc_def

    p_w46_lerc_zstd = synth_dir / "w46_lerc_zstd_lossy.tif"
    with rasterio.open(
        p_w46_lerc_zstd, "w", driver="GTiff", width=128, height=128, count=1, dtype="float32",
        crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64,
        compress="lerc_zstd", max_z_error=0.5, nodata=-9999.0,
    ) as ds:
        ds.write(a46_hole, 1)
    out["w46_lerc_zstd_lossy"] = p_w46_lerc_zstd

    for tag, epsg, (bw, bs, be, bn) in [
        ("5070", "EPSG:5070", (-98.0, 38.0, -97.0, 39.0)),
        ("2154", "EPSG:2154", (2.0, 48.0, 3.0, 49.0)),
        ("3035", "EPSG:3035", (10.0, 50.0, 11.0, 51.0)),
        ("6933", "EPSG:6933", (15.0, 20.0, 16.0, 21.0)),
        ("3031", "EPSG:3031", (0.0, -76.0, 2.0, -75.0)),
        ("27700", "EPSG:27700", (-1.5, 51.5, -0.5, 52.5)),
    ]:
        l_, b_, r_, t_ = transform_bounds("EPSG:4326", epsg, bw, bs, be, bn)
        p_crs = synth_dir / f"w46_crs_{tag}.tif"
        with rasterio.open(
            p_crs, "w", driver="GTiff", width=128, height=128, count=1, dtype="float32",
            crs=epsg, transform=from_bounds(l_, b_, r_, t_, 128, 128),
            tiled=True, blockxsize=64, blockysize=64, compress="deflate",
        ) as ds:
            ds.write(a46, 1)
        out[f"w46_crs_{tag}"] = p_crs

    # Wave 47: Antimeridian-crossing COG (179E..181E) + 6 exotic CRS families
    # (Equal Earth 8857, NSIDC Polar Stereo N 3413, Australian Albers 3577, Dutch RD New 28992, Swiss LV95 2056, NAD83 4269)
    p_w47_anti = synth_dir / "w47_antimeridian.tif"
    with rasterio.open(
        p_w47_anti, "w", driver="GTiff", width=128, height=128, count=1, dtype="float32",
        crs="EPSG:4326", transform=from_bounds(179.0, -18.0, 181.0, -16.0, 128, 128),
        tiled=True, blockxsize=64, blockysize=64, compress="deflate",
    ) as ds:
        ds.write(a46, 1)
    out["w47_antimeridian"] = p_w47_anti

    for tag, epsg, (bw, bs, be, bn) in [
        ("8857", "EPSG:8857", (10.0, 20.0, 11.0, 21.0)),
        ("3413", "EPSG:3413", (-45.0, 72.0, -43.0, 73.0)),
        ("3577", "EPSG:3577", (135.0, -26.0, 136.0, -25.0)),
        ("28992", "EPSG:28992", (5.0, 52.0, 6.0, 52.5)),
        ("2056", "EPSG:2056", (7.5, 46.5, 8.5, 47.0)),
        ("4269", "EPSG:4269", (-105.5, 39.5, -104.5, 40.5)),
    ]:
        l_, b_, r_, t_ = transform_bounds("EPSG:4326", epsg, bw, bs, be, bn)
        p_crs = synth_dir / f"w47_crs_{tag}.tif"
        with rasterio.open(
            p_crs, "w", driver="GTiff", width=128, height=128, count=1, dtype="float32",
            crs=epsg, transform=from_bounds(l_, b_, r_, t_, 128, 128),
            tiled=True, blockxsize=64, blockysize=64, compress="deflate",
        ) as ds:
            ds.write(a46, 1)
        out[f"w47_crs_{tag}"] = p_crs

    # Wave 48: 3-band float32 COG for deep Expression VM math/conditionals/div-by-zero checks
    p_w48_3b = synth_dir / "w48_expr3b.tif"
    b1_48 = (10.0 + xx9 * 1.5).astype(np.float32)
    b2_48 = (20.0 + yy9 * 1.2).astype(np.float32)
    b3_48 = (5.0 + (xx9 + yy9) * 0.5).astype(np.float32)
    with rasterio.open(
        p_w48_3b, "w", driver="GTiff", width=128, height=128, count=3, dtype="float32",
        crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64, compress="deflate",
    ) as ds:
        ds.write(np.stack([b1_48, b2_48, b3_48]))
    out["w48_expr3b"] = p_w48_3b

    # Wave 49: 3-band RGB ?nodata=255 (pure [255,255,255] hole + partial [255,120,200] valid) & 2-band Gray+Alpha (ExtraSamples=2)
    p_w49_rgb255 = synth_dir / "w49_rgb_nd255.tif"
    r49 = np.clip(40 + xx9, 1, 250).astype(np.uint8)
    g49 = np.clip(60 + yy9, 1, 250).astype(np.uint8)
    b49 = np.clip(80 + (xx9 + yy9) // 2, 1, 250).astype(np.uint8)
    r49[80:100, 55:75] = 255
    g49[80:100, 55:75] = 255
    b49[80:100, 55:75] = 255
    r49[100:120, 75:100] = 255
    g49[100:120, 75:100] = 120
    b49[100:120, 75:100] = 200
    with rasterio.open(
        p_w49_rgb255, "w", driver="GTiff", width=128, height=128, count=3, dtype="uint8",
        crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64, compress="deflate",
    ) as ds:
        ds.write(np.stack([r49, g49, b49]))
    out["w49_rgb_nd255"] = p_w49_rgb255

    p_w49_ga = synth_dir / "w49_gray_alpha.tif"
    gray49 = np.clip(30 + xx9 + yy9 // 2, 1, 250).astype(np.uint8)
    alpha49 = np.full((128, 128), 255, dtype=np.uint8)
    alpha49[80:110, 60:90] = 0
    with rasterio.open(
        p_w49_ga, "w", driver="GTiff", width=128, height=128, count=2, dtype="uint8",
        crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64, compress="deflate",
        photometric="minisblack",
    ) as ds:
        ds.write(gray49, 1)
        ds.write(alpha49, 2)
        ds.colorinterp = (rasterio.enums.ColorInterp.gray, rasterio.enums.ColorInterp.alpha)
    out["w49_gray_alpha"] = p_w49_ga

    # Wave 50: Donut polygon (exterior + interior hole ring) & MultiPolygon COG with 5000.0 center spike
    p_w50_donut = synth_dir / "w50_donut.tif"
    arr50 = (10.0 + xx9 * 0.5 + yy9 * 0.5).astype(np.float32)
    arr50[48:80, 48:80] = 5000.0
    with rasterio.open(
        p_w50_donut, "w", driver="GTiff", width=128, height=128, count=1, dtype="float32",
        crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64, compress="deflate",
    ) as ds:
        ds.write(arr50, 1)
    out["w50_donut"] = p_w50_donut

    # Wave 51: Big-Endian (MM) int16 Predictor=2 & float32 Predictor=3 COGs (with interior nodata holes)
    p_w51_be_i16 = synth_dir / "w51_be_i16_p2.tif"
    i16_51 = (-1000 + xx9.astype(np.int32) * 30 + yy9.astype(np.int32) * 20).astype(np.int16)
    i16_51[80:110, 60:90] = -32768
    with rasterio.open(
        p_w51_be_i16, "w", driver="GTiff", width=128, height=128, count=1, dtype="int16",
        crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64,
        compress="deflate", predictor=2, nodata=-32768, endianness="BIG",
    ) as ds:
        ds.write(i16_51, 1)
    assert p_w51_be_i16.read_bytes()[:2] == b"MM", "w51_be_i16_p2 must be Big-Endian MM"
    out["w51_be_i16_p2"] = p_w51_be_i16

    p_w51_be_f32 = synth_dir / "w51_be_f32_p3.tif"
    f32_51 = (100.5 + xx9 * 2.5 + yy9 * 1.25).astype(np.float32)
    f32_51[80:110, 60:90] = -9999.0
    with rasterio.open(
        p_w51_be_f32, "w", driver="GTiff", width=128, height=128, count=1, dtype="float32",
        crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64,
        compress="deflate", predictor=3, nodata=-9999.0, endianness="BIG",
    ) as ds:
        ds.write(f32_51, 1)
    assert p_w51_be_f32.read_bytes()[:2] == b"MM", "w51_be_f32_p3 must be Big-Endian MM"
    out["w51_be_f32_p3"] = p_w51_be_f32

    # Wave 52: 3-band uint16 RGB COG with nodata=0 hole (unrescaled 16-bit RGBA PNG + .npy)
    p_w52_u16_rgb = synth_dir / "w52_u16_rgb.tif"
    r52 = (1000 + xx9.astype(np.uint32) * 300).astype(np.uint16)
    g52 = (2000 + yy9.astype(np.uint32) * 300).astype(np.uint16)
    b52 = (3000 + (xx9 + yy9).astype(np.uint32) * 150).astype(np.uint16)
    r52[80:100, 60:80] = 0
    g52[80:100, 60:80] = 0
    b52[80:100, 60:80] = 0
    with rasterio.open(
        p_w52_u16_rgb, "w", driver="GTiff", width=128, height=128, count=3, dtype="uint16",
        crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64,
        compress="deflate", photometric="rgb", nodata=0,
    ) as ds:
        ds.write(np.stack([r52, g52, b52]))
    out["w52_u16_rgb"] = p_w52_u16_rgb

    # Wave 53: Negative scale (scale=-0.01, offset=100.0) uint16, WebP lossless 3-band RGB, ZSTD float64 Predictor=3 (>1e9)
    p_w53_neg = synth_dir / "w53_neg_scale.tif"
    dn53 = (1000 + xx9.astype(np.uint32) * 30 + yy9.astype(np.uint32) * 20).astype(np.uint16)
    dn53[80:110, 60:90] = 65535
    with rasterio.open(
        p_w53_neg, "w", driver="GTiff", width=128, height=128, count=1, dtype="uint16",
        crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64, compress="deflate", nodata=65535,
    ) as ds:
        ds.write(dn53, 1)
        ds.scales = (-0.01,)
        ds.offsets = (100.0,)
    out["w53_neg_scale"] = p_w53_neg

    p_w53_webp = synth_dir / "w53_webp_lossless.tif"
    rgb53 = np.stack([
        np.clip(20 + xx9 * 1.5, 1, 250).astype(np.uint8),
        np.clip(40 + yy9 * 1.5, 1, 250).astype(np.uint8),
        np.clip(60 + (xx9 + yy9) * 0.7, 1, 250).astype(np.uint8),
    ])
    with rasterio.open(
        p_w53_webp, "w", driver="GTiff", width=128, height=128, count=3, dtype="uint8",
        crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64,
        compress="webp", webp_lossless=True,
    ) as ds:
        ds.write(rgb53)
    out["w53_webp_lossless"] = p_w53_webp

    p_w53_f64 = synth_dir / "w53_zstd_f64_p3.tif"
    f64_53 = (1_000_000_000.0 + xx9.astype(np.float64) * 0.002 + yy9.astype(np.float64) * 0.005)
    f64_53[80:110, 60:90] = -9999.0
    with rasterio.open(
        p_w53_f64, "w", driver="GTiff", width=128, height=128, count=1, dtype="float64",
        crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64,
        compress="zstd", predictor=3, nodata=-9999.0,
    ) as ds:
        ds.write(f64_53, 1)
    out["w53_zstd_f64_p3"] = p_w53_f64

    # Wave 54: WebMercator-aligned (1:1 tile 10/164/396, 256x256) JPEG-in-TIFF PHOTOMETRIC=RGB & YCBCR
    # with shared tag 347 JPEGTables — genuinely exercises zero-decode passthrough + splice_jpeg_tables
    # (YCBCR .jpg), the photometric=2 RGB passthrough guard (RGB .jpg), and decode-path JPEGTables (.png).
    b54 = mercantile.xy_bounds(164, 396, 10)
    tf_w54 = from_bounds(b54.left, b54.bottom, b54.right, b54.top, 256, 256)
    yy54, xx54 = np.mgrid[0:256, 0:256]
    r54 = np.clip(40 + xx54 * 0.75, 0, 255).astype(np.uint8)
    g54 = np.clip(30 + yy54 * 0.70, 0, 255).astype(np.uint8)
    b54_ch = np.clip(220 - xx54 * 0.60, 0, 255).astype(np.uint8)
    rgb54 = np.stack([r54, g54, b54_ch])

    def _has_tiff_tag(path: Path, tag_id: int) -> bool:
        raw = path.read_bytes()
        bo = "<" if raw[:2] == b"II" else ">"
        ifd_off = struct.unpack(bo + "I", raw[4:8])[0]
        n_ent = struct.unpack(bo + "H", raw[ifd_off:ifd_off + 2])[0]
        for i in range(n_ent):
            t = struct.unpack(bo + "H", raw[ifd_off + 2 + i * 12:ifd_off + 4 + i * 12])[0]
            if t == tag_id:
                return True
        return False

    p_w54_rgb_jt = synth_dir / "w54_rgb_jpeg_tables.tif"
    with rasterio.open(
        p_w54_rgb_jt, "w", driver="GTiff", width=256, height=256, count=3, dtype="uint8",
        crs="EPSG:3857", transform=tf_w54, tiled=True, blockxsize=256, blockysize=256,
        compress="JPEG", photometric="RGB", jpeg_quality=92, JPEGTABLESMODE=1,
    ) as ds:
        ds.write(rgb54)
    assert _has_tiff_tag(p_w54_rgb_jt, 347), "w54_rgb_jpeg_tables IFD must contain tag 347 JPEGTables"
    out["w54_rgb_jpeg_tables"] = p_w54_rgb_jt

    p_w54_ycbcr_jt = synth_dir / "w54_ycbcr_jpeg_tables.tif"
    with rasterio.open(
        p_w54_ycbcr_jt, "w", driver="GTiff", width=256, height=256, count=3, dtype="uint8",
        crs="EPSG:3857", transform=tf_w54, tiled=True, blockxsize=256, blockysize=256,
        compress="JPEG", photometric="YCBCR", jpeg_quality=92, JPEGTABLESMODE=1,
    ) as ds:
        ds.write(rgb54)
    assert _has_tiff_tag(p_w54_ycbcr_jt, 347), "w54_ycbcr_jpeg_tables IFD must contain tag 347 JPEGTables"
    out["w54_ycbcr_jpeg_tables"] = p_w54_ycbcr_jt

    # Wave 55: Float32 with 1-bit GDAL_TIFF_INTERNAL_MASK (no nodata tag), 4:1 non-square pixel aspect,
    # and 512x512 multi-overview (2x/4x/8x) uint16 with scale/offset + nodata hole
    p_w55_imask = synth_dir / "w55_f32_imask.tif"
    f32_55 = (10.0 + xx9 * 1.5 + yy9 * 0.8).astype(np.float32)
    mask55 = np.full((128, 128), 255, dtype=np.uint8)
    mask55[80:110, 60:90] = 0
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rasterio.open(
            p_w55_imask, "w", driver="GTiff", width=128, height=128, count=1, dtype="float32",
            crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64, compress="deflate",
        ) as ds:
            ds.write(f32_55, 1)
            ds.write_mask(mask55)
    out["w55_f32_imask"] = p_w55_imask

    p_w55_ns = synth_dir / "w55_nonsquare_4to1.tif"
    yy55, xx55 = np.mgrid[0:256, 0:64]
    f32_ns = (50.0 + xx55 * 3.0 + yy55 * 0.75).astype(np.float32)
    f32_ns[160:220, 30:45] = -9999.0
    tf_w55_ns = from_bounds(-122.5, 37.5, -122.0, 38.0, 64, 256)
    with rasterio.open(
        p_w55_ns, "w", driver="GTiff", width=64, height=256, count=1, dtype="float32",
        crs="EPSG:4326", transform=tf_w55_ns, tiled=True, blockxsize=64, blockysize=64,
        compress="deflate", nodata=-9999.0,
    ) as ds:
        ds.write(f32_ns, 1)
    out["w55_nonsquare_4to1"] = p_w55_ns

    p_w55_ovr = synth_dir / "w55_multi_ovr_unscale.tif"
    yy55o, xx55o = np.mgrid[0:512, 0:512]
    u16_ovr = (5000 + xx55o.astype(np.uint32) * 40 + yy55o.astype(np.uint32) * 30).astype(np.uint16)
    u16_ovr[300:420, 220:360] = 65535
    tf_w55_ovr = from_bounds(-122.5, 37.5, -122.0, 38.0, 512, 512)
    with rasterio.open(
        p_w55_ovr, "w", driver="GTiff", width=512, height=512, count=1, dtype="uint16",
        crs="EPSG:4326", transform=tf_w55_ovr, tiled=True, blockxsize=128, blockysize=128,
        compress="deflate", nodata=65535,
    ) as ds:
        ds.write(u16_ovr, 1)
        ds.scales = (0.001,)
        ds.offsets = (-10.0,)
        ds.build_overviews([2, 4, 8], rasterio.enums.Resampling.average)
    out["w55_multi_ovr_unscale"] = p_w55_ovr

    # Wave 56: ?nodata= query override + unscale=true, and prime-dimension (137x113) partial edge tiles
    # with rectangular 128x32 blocks + Predictor=2 (int16 ZSTD) & Predictor=3 (float32 DEFLATE)
    p_w56_sc = synth_dir / "w56_scaled_no_nd.tif"
    dn56 = (2000 + xx9.astype(np.uint32) * 35 + yy9.astype(np.uint32) * 25).astype(np.uint16)
    dn56[80:110, 60:90] = 1000  # sentinel DN (not declared as TIFF nodata; supplied via ?nodata=1000)
    with rasterio.open(
        p_w56_sc, "w", driver="GTiff", width=128, height=128, count=1, dtype="uint16",
        crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64, compress="deflate",
    ) as ds:
        ds.write(dn56, 1)
        ds.scales = (0.01,)
        ds.offsets = (-50.0,)
    out["w56_scaled_no_nd"] = p_w56_sc

    yy56, xx56 = np.mgrid[0:113, 0:137]
    tf_w56_p = from_bounds(-122.5, 37.5, -122.0, 38.0, 137, 113)

    p_w56_pi16 = synth_dir / "w56_prime_i16_p2.tif"
    i16_56 = (-500 + xx56.astype(np.int32) * 25 + yy56.astype(np.int32) * 15).astype(np.int16)
    i16_56[70:100, 65:135] = -32768  # crosses right partial edge tile (cols 128..137)
    with rasterio.open(
        p_w56_pi16, "w", driver="GTiff", width=137, height=113, count=1, dtype="int16",
        crs="EPSG:4326", transform=tf_w56_p, tiled=True, blockxsize=128, blockysize=32,
        compress="zstd", predictor=2, nodata=-32768,
    ) as ds:
        ds.write(i16_56, 1)
    out["w56_prime_i16_p2"] = p_w56_pi16

    p_w56_pf32 = synth_dir / "w56_prime_f32_p3.tif"
    f32_56 = (10.25 + xx56 * 1.75 + yy56 * 0.625).astype(np.float32)
    f32_56[70:112, 65:135] = -9999.0  # crosses both right (128..137) & bottom (96..113) partial edge tiles
    with rasterio.open(
        p_w56_pf32, "w", driver="GTiff", width=137, height=113, count=1, dtype="float32",
        crs="EPSG:4326", transform=tf_w56_p, tiled=True, blockxsize=128, blockysize=32,
        compress="deflate", predictor=3, nodata=-9999.0,
    ) as ds:
        ds.write(f32_56, 1)
    out["w56_prime_f32_p3"] = p_w56_pf32

    # Wave 57: Sparse GeoTIFFs (SPARSE_OK=TRUE, TileByteCounts==0 omitted tiles) with & without nodata,
    # and 3-band uint16 with heterogeneous per-band (scale, offset)
    yy57q, xx57q = np.mgrid[0:64, 0:64]
    p_w57_sp_nd = synth_dir / "w57_sparse_nodata.tif"
    with rasterio.open(
        p_w57_sp_nd, "w", driver="GTiff", width=128, height=128, count=1, dtype="float32",
        crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64,
        compress="deflate", nodata=-9999.0, SPARSE_OK="TRUE",
    ) as ds:
        ds.write((20.0 + xx57q * 1.2 + yy57q * 0.8).astype(np.float32), 1, window=rasterio.windows.Window(0, 0, 64, 64))
        ds.write((100.0 + xx57q * 1.5 + yy57q * 1.0).astype(np.float32), 1, window=rasterio.windows.Window(64, 64, 64, 64))
    out["w57_sparse_nodata"] = p_w57_sp_nd

    p_w57_sp_no_nd = synth_dir / "w57_sparse_no_nd.tif"
    with rasterio.open(
        p_w57_sp_no_nd, "w", driver="GTiff", width=128, height=128, count=1, dtype="uint16",
        crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64,
        compress="deflate", SPARSE_OK="TRUE",
    ) as ds:
        ds.write((100 + xx57q.astype(np.uint32) * 10 + yy57q.astype(np.uint32) * 5).astype(np.uint16), 1, window=rasterio.windows.Window(0, 0, 64, 64))
        ds.write((500 + xx57q.astype(np.uint32) * 12 + yy57q.astype(np.uint32) * 8).astype(np.uint16), 1, window=rasterio.windows.Window(64, 64, 64, 64))
    out["w57_sparse_no_nd"] = p_w57_sp_no_nd

    p_w57_pb_sc = synth_dir / "w57_perband_scale_3b.tif"
    b1_57 = (1000 + xx9.astype(np.uint32) * 20).astype(np.uint16)
    b2_57 = (2000 + yy9.astype(np.uint32) * 30).astype(np.uint16)
    b3_57 = (3000 + (xx9 + yy9).astype(np.uint32) * 15).astype(np.uint16)
    b1_57[80:105, 60:85] = 65535
    b2_57[80:105, 60:85] = 65535
    b3_57[80:105, 60:85] = 65535
    with rasterio.open(
        p_w57_pb_sc, "w", driver="GTiff", width=128, height=128, count=3, dtype="uint16",
        crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64,
        compress="deflate", nodata=65535,
    ) as ds:
        ds.write(np.stack([b1_57, b2_57, b3_57]))
        ds.scales = (0.001, 0.002, 0.005)
        ds.offsets = (0.0, -1.0, 10.0)
    out["w57_perband_scale_3b"] = p_w57_pb_sc

    # Wave 58: Planar-separate (INTERLEAVE=BAND) asymmetric sparse TIFFs (with & without nodata),
    # sparse nodata=0 TIFF with ?nodata=999 override, and float32 -0.0 vs nodata=0.0
    p_w58_pl_nd = synth_dir / "w58_planar_sparse_nd.tif"
    with rasterio.open(
        p_w58_pl_nd, "w", driver="GTiff", width=128, height=128, count=3, dtype="float32",
        crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64,
        interleave="band", compress="deflate", nodata=-9999.0, SPARSE_OK="TRUE",
    ) as ds:
        w_tl = rasterio.windows.Window(0, 0, 64, 64)
        w_tr = rasterio.windows.Window(64, 0, 64, 64)
        w_bl = rasterio.windows.Window(0, 64, 64, 64)
        w_br = rasterio.windows.Window(64, 64, 64, 64)
        blk1 = (10.0 + xx57q * 1.0 + yy57q * 0.5).astype(np.float32)
        blk2 = (50.0 + xx57q * 0.8 + yy57q * 1.2).astype(np.float32)
        blk3 = (90.0 + xx57q * 0.6 + yy57q * 0.9).astype(np.float32)
        # Band 1: TL + BR; Band 2: TL + TR + BR; Band 3: TL + BL + BR (asymmetric sparsity per band!)
        ds.write(blk1, 1, window=w_tl); ds.write(blk1 + 20.0, 1, window=w_br)
        ds.write(blk2, 2, window=w_tl); ds.write(blk2 + 10.0, 2, window=w_tr); ds.write(blk2 + 20.0, 2, window=w_br)
        ds.write(blk3, 3, window=w_tl); ds.write(blk3 + 15.0, 3, window=w_bl); ds.write(blk3 + 25.0, 3, window=w_br)
    out["w58_planar_sparse_nd"] = p_w58_pl_nd

    p_w58_pl_nond = synth_dir / "w58_planar_sparse_no_nd.tif"
    with rasterio.open(
        p_w58_pl_nond, "w", driver="GTiff", width=128, height=128, count=3, dtype="uint16",
        crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64,
        interleave="band", compress="deflate", SPARSE_OK="TRUE",
    ) as ds:
        ub1 = (100 + xx57q.astype(np.uint32) * 5).astype(np.uint16)
        ub2 = (200 + yy57q.astype(np.uint32) * 6).astype(np.uint16)
        ub3 = (300 + (xx57q + yy57q).astype(np.uint32) * 3).astype(np.uint16)
        ds.write(ub1, 1, window=w_tl); ds.write(ub1 + 50, 1, window=w_br)
        ds.write(ub2, 2, window=w_tl); ds.write(ub2 + 30, 2, window=w_tr); ds.write(ub2 + 60, 2, window=w_br)
        ds.write(ub3, 3, window=w_tl); ds.write(ub3 + 40, 3, window=w_bl); ds.write(ub3 + 80, 3, window=w_br)
    out["w58_planar_sparse_no_nd"] = p_w58_pl_nond

    p_w58_sp_nd0 = synth_dir / "w58_sparse_nd0.tif"
    with rasterio.open(
        p_w58_sp_nd0, "w", driver="GTiff", width=128, height=128, count=1, dtype="uint16",
        crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64,
        compress="deflate", nodata=0, SPARSE_OK="TRUE",
    ) as ds:
        tl58 = (100 + xx57q.astype(np.uint32) * 4 + yy57q.astype(np.uint32) * 3).astype(np.uint16)
        br58 = (200 + xx57q.astype(np.uint32) * 5 + yy57q.astype(np.uint32) * 4).astype(np.uint16)
        br58[15:40, 15:40] = 999
        ds.write(tl58, 1, window=w_tl)
        ds.write(br58, 1, window=w_br)
    out["w58_sparse_nd0"] = p_w58_sp_nd0

    p_w58_neg0 = synth_dir / "w58_neg_zero_f32.tif"
    f32_58 = (10.0 + xx9 * 1.2 + yy9 * 0.8).astype(np.float32)
    f32_58[75:95, 55:75] = np.float32(-0.0)
    f32_58[95:115, 75:95] = np.float32(0.0)
    with rasterio.open(
        p_w58_neg0, "w", driver="GTiff", width=128, height=128, count=1, dtype="float32",
        crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64,
        compress="deflate", nodata=0.0,
    ) as ds:
        ds.write(f32_58, 1)
    out["w58_neg_zero_f32"] = p_w58_neg0

    # Wave 59: Cubic/Lanczos step-edge ringing clamping (uint8 0/255, int8 -127/127 w/ nodata=-128),
    # single-row strips (ROWS_PER_STRIP=1) & prime strips (ROWS_PER_STRIP=13), and BigTIFF f64 P=3 w/ overviews
    p_w59_u8 = synth_dir / "w59_step_u8.tif"
    u8_59 = np.where(((xx9 // 16) + (yy9 // 16)) % 2 == 0, 0, 255).astype(np.uint8)
    with rasterio.open(
        p_w59_u8, "w", driver="GTiff", width=128, height=128, count=1, dtype="uint8",
        crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64, compress="deflate",
    ) as ds:
        ds.write(u8_59, 1)
    out["w59_step_u8"] = p_w59_u8

    p_w59_i8 = synth_dir / "w59_step_i8.tif"
    i8_59 = np.where(((xx9 // 16) + (yy9 // 16)) % 2 == 0, -127, 127).astype(np.int8)
    i8_59[80:100, 60:80] = -128
    with rasterio.open(
        p_w59_i8, "w", driver="GTiff", width=128, height=128, count=1, dtype="int8",
        crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64,
        compress="deflate", nodata=-128,
    ) as ds:
        ds.write(i8_59, 1)
    out["w59_step_i8"] = p_w59_i8

    p_w59_s1 = synth_dir / "w59_strip1_i16_p2.tif"
    i16_59 = (-800 + xx9.astype(np.int32) * 20 + yy9.astype(np.int32) * 15).astype(np.int16)
    i16_59[80:105, 60:90] = -32768
    with rasterio.open(
        p_w59_s1, "w", driver="GTiff", width=128, height=128, count=1, dtype="int16",
        crs="EPSG:4326", transform=tf_w9, tiled=False, blockysize=1,
        compress="deflate", predictor=2, nodata=-32768,
    ) as ds:
        ds.write(i16_59, 1)
    out["w59_strip1_i16_p2"] = p_w59_s1

    p_w59_s13 = synth_dir / "w59_strip13_f32_p3.tif"
    f32_59 = (25.5 + xx9 * 1.4 + yy9 * 0.9).astype(np.float32)
    f32_59[85:125, 60:95] = -9999.0  # crosses into final partial 11-row strip (rows 117..128)
    with rasterio.open(
        p_w59_s13, "w", driver="GTiff", width=128, height=128, count=1, dtype="float32",
        crs="EPSG:4326", transform=tf_w9, tiled=False, blockysize=13,
        compress="zstd", predictor=3, nodata=-9999.0,
    ) as ds:
        ds.write(f32_59, 1)
    out["w59_strip13_f32_p3"] = p_w59_s13

    p_w59_bt = synth_dir / "w59_bigtiff_f64_ovr.tif"
    yy59b, xx59b = np.mgrid[0:256, 0:256]
    f64_59 = (1_000_000_000.0 + xx59b.astype(np.float64) * 0.004 + yy59b.astype(np.float64) * 0.008)
    f64_59[160:210, 120:170] = -9999.0
    tf_w59_bt = from_bounds(-122.5, 37.5, -122.0, 38.0, 256, 256)
    with rasterio.open(
        p_w59_bt, "w", driver="GTiff", width=256, height=256, count=1, dtype="float64",
        crs="EPSG:4326", transform=tf_w59_bt, tiled=True, blockxsize=64, blockysize=64,
        compress="deflate", predictor=3, nodata=-9999.0, BIGTIFF="YES",
    ) as ds:
        ds.write(f64_59, 1)
        ds.build_overviews([2, 4], rasterio.enums.Resampling.average)
    assert struct.unpack("<H", p_w59_bt.read_bytes()[2:4])[0] == 43, "w59_bigtiff_f64_ovr must be BigTIFF (version 43)"
    out["w59_bigtiff_f64_ovr"] = p_w59_bt

    # Wave 60: Non-square 4:1 wide (256x64) float32 COG for preview/bbox/feature aspect-ratio sizing & max_size=0
    p_w60_wide = synth_dir / "w60_wide_4to1.tif"
    yy60, xx60 = np.mgrid[0:64, 0:256]
    f32_60 = (20.0 + xx60 * 0.8 + yy60 * 2.5).astype(np.float32)
    f32_60[20:44, 100:150] = -9999.0
    tf_w60 = from_bounds(-122.5, 37.5, -122.0, 38.0, 256, 64)
    with rasterio.open(
        p_w60_wide, "w", driver="GTiff", width=256, height=64, count=1, dtype="float32",
        crs="EPSG:4326", transform=tf_w60, tiled=True, blockxsize=64, blockysize=64,
        compress="deflate", nodata=-9999.0,
    ) as ds:
        ds.write(f32_60, 1)
    out["w60_wide_4to1"] = p_w60_wide

    # Wave 65: Float32 COG whose bottom half (rows 64..128, covering all of tile 10/164/396) is 100% nodata
    p_w65_and = synth_dir / "w65_all_nodata_tile.tif"
    f32_65 = np.full((128, 128), -9999.0, dtype=np.float32)
    f32_65[:64, :64] = (20.0 + xx9[:64, :64] * 1.5 + yy9[:64, :64] * 0.8).astype(np.float32)
    with rasterio.open(
        p_w65_and, "w", driver="GTiff", width=128, height=128, count=1, dtype="float32",
        crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64,
        compress="deflate", nodata=-9999.0,
    ) as ds:
        ds.write(f32_65, 1)
    out["w65_all_nodata_tile"] = p_w65_and

    # Wave 66: Constant-value (zero-variance) float32 COG + nodata hole, and STAC item with
    # heterogeneous CRS (EPSG:32610 vs EPSG:4326) AND heterogeneous dtype (uint16 vs float32)
    p_w66_const = synth_dir / "w66_const_f32.tif"
    f32_66 = np.full((128, 128), 42.0, dtype=np.float32)
    f32_66[80:105, 60:85] = -9999.0
    with rasterio.open(
        p_w66_const, "w", driver="GTiff", width=128, height=128, count=1, dtype="float32",
        crs="EPSG:4326", transform=tf_w9, tiled=True, blockxsize=64, blockysize=64,
        compress="deflate", nodata=-9999.0,
    ) as ds:
        ds.write(f32_66, 1)
    out["w66_const_f32"] = p_w66_const

    p_w66_flat = synth_dir / "w66_const_flat.tif"
    with rasterio.open(
        p_w66_flat, "w", driver="GTiff", width=128, height=128, count=1, dtype="float32",
        crs="EPSG:4326", transform=from_bounds(-122.6, 37.4, -121.9, 38.1, 128, 128),
        tiled=True, blockxsize=64, blockysize=64, compress="deflate",
    ) as ds:
        ds.write(np.full((128, 128), 42.0, dtype=np.float32), 1)
    out["w66_const_flat"] = p_w66_flat

    p_w66_red = synth_dir / "w66_red_utm_u16.tif"
    b66_utm = rasterio.warp.transform_bounds("EPSG:4326", "EPSG:32610", -122.6, 37.4, -121.9, 38.1)
    tf_w66_utm = from_bounds(*b66_utm, 128, 128)
    u16_66 = (100 + xx9.astype(np.uint32) * 3 + yy9.astype(np.uint32) * 2).astype(np.uint16)
    with rasterio.open(
        p_w66_red, "w", driver="GTiff", width=128, height=128, count=1, dtype="uint16",
        crs="EPSG:32610", transform=tf_w66_utm, tiled=True, blockxsize=64, blockysize=64,
        compress="deflate", nodata=0,
    ) as ds:
        ds.write(u16_66, 1)

    p_w66_nir = synth_dir / "w66_nir_4326_f32.tif"
    f32_66_nir = (200.0 + xx9 * 2.5 + yy9 * 1.5).astype(np.float32)
    with rasterio.open(
        p_w66_nir, "w", driver="GTiff", width=128, height=128, count=1, dtype="float32",
        crs="EPSG:4326", transform=from_bounds(-122.6, 37.4, -121.9, 38.1, 128, 128),
        tiled=True, blockxsize=64, blockysize=64, compress="deflate", nodata=-9999.0,
    ) as ds:
        ds.write(f32_66_nir, 1)

    p_w66_stac = synth_dir / "w66_stac_hetero.json"
    p_w66_stac.write_text(json.dumps({
        "type": "Feature",
        "stac_version": "1.0.0",
        "id": "w66-hetero-crs-dtype",
        "geometry": {"type": "Polygon", "coordinates": [[[-122.6, 37.4], [-121.9, 37.4], [-121.9, 38.1], [-122.6, 38.1], [-122.6, 37.4]]]},
        "bbox": [-122.6, 37.4, -121.9, 38.1],
        "properties": {"datetime": "2026-01-01T00:00:00Z"},
        "links": [],
        "assets": {
            "red": {"href": str(p_w66_red), "type": "image/tiff; application=geotiff"},
            "nir": {"href": str(p_w66_nir), "type": "image/tiff; application=geotiff"},
        },
    }))
    out["w66_stac_hetero"] = p_w66_stac

    # Wave 67: STAC unscale=true with heterogeneous per-asset & per-band (scale, offset) + sub-byte NBITS=12/4 with nodata
    tf_w67 = from_bounds(-122.6, 37.4, -121.9, 38.1, 128, 128)
    p_w67_red = synth_dir / "w67_red_scaled_u16.tif"
    u16_67_r = (2000 + xx9 * 25 + yy9 * 15).astype(np.uint16)
    u16_67_r[75:95, 60:80] = 0
    with rasterio.open(
        p_w67_red, "w", driver="GTiff", width=128, height=128, count=1, dtype="uint16",
        crs="EPSG:4326", transform=tf_w67, tiled=True, blockxsize=64, blockysize=64,
        compress="deflate", nodata=0,
    ) as ds:
        ds.write(u16_67_r, 1)
        ds.scales = (0.0001,)
        ds.offsets = (-0.1,)

    p_w67_nir = synth_dir / "w67_nir_scaled_u16.tif"
    u16_67_n = (3000 + xx9 * 30 + yy9 * 20).astype(np.uint16)
    u16_67_n[75:95, 60:80] = 0
    with rasterio.open(
        p_w67_nir, "w", driver="GTiff", width=128, height=128, count=1, dtype="uint16",
        crs="EPSG:4326", transform=tf_w67, tiled=True, blockxsize=64, blockysize=64,
        compress="deflate", nodata=0,
    ) as ds:
        ds.write(u16_67_n, 1)
        ds.scales = (0.0002,)
        ds.offsets = (0.05,)

    p_w67_ms = synth_dir / "w67_ms_hetero_scale_u16.tif"
    u16_67_ms = np.stack([
        (1000 + xx9 * 10 + yy9 * 5).astype(np.uint16),
        (2000 + xx9 * 15 + yy9 * 10).astype(np.uint16),
        (4000 + xx9 * 20 + yy9 * 15).astype(np.uint16),
    ])
    u16_67_ms[:, 75:95, 60:80] = 0
    with rasterio.open(
        p_w67_ms, "w", driver="GTiff", width=128, height=128, count=3, dtype="uint16",
        crs="EPSG:4326", transform=tf_w67, tiled=True, blockxsize=64, blockysize=64,
        compress="deflate", nodata=0,
    ) as ds:
        ds.write(u16_67_ms)
        ds.scales = (0.01, 0.001, -0.005)
        ds.offsets = (1.0, -2.0, 50.0)

    p_w67_stac = synth_dir / "w67_stac_scaled.json"
    p_w67_stac.write_text(json.dumps({
        "type": "Feature",
        "stac_version": "1.0.0",
        "id": "w67-scaled-assets",
        "geometry": {"type": "Polygon", "coordinates": [[[-122.6, 37.4], [-121.9, 37.4], [-121.9, 38.1], [-122.6, 38.1], [-122.6, 37.4]]]},
        "bbox": [-122.6, 37.4, -121.9, 38.1],
        "properties": {"datetime": "2026-01-01T00:00:00Z"},
        "links": [],
        "assets": {
            "red": {"href": str(p_w67_red), "type": "image/tiff; application=geotiff"},
            "nir": {"href": str(p_w67_nir), "type": "image/tiff; application=geotiff"},
            "ms": {"href": str(p_w67_ms), "type": "image/tiff; application=geotiff"},
        },
    }))
    out["w67_stac_scaled"] = p_w67_stac

    p_w67_nb12 = synth_dir / "w67_nbits12_nodata_scaled.tif"
    u16_nb12 = (200 + xx9 * 18 + yy9 * 12).astype(np.uint16)
    u16_nb12[75:95, 60:80] = 4095
    with rasterio.open(
        p_w67_nb12, "w", driver="GTiff", width=128, height=128, count=1, dtype="uint16",
        crs="EPSG:4326", transform=tf_w67, tiled=True, blockxsize=64, blockysize=64,
        compress="deflate", nbits=12, nodata=4095,
    ) as ds:
        ds.write(u16_nb12, 1)
        ds.scales = (0.001,)
        ds.offsets = (-0.5,)
    out["w67_nbits12_nodata_scaled"] = p_w67_nb12

    p_w67_nb4 = synth_dir / "w67_nbits4_nodata.tif"
    u8_nb4 = ((xx9 // 10 + yy9 // 20) % 15).astype(np.uint8)
    u8_nb4[75:95, 60:80] = 15
    with rasterio.open(
        p_w67_nb4, "w", driver="GTiff", width=128, height=128, count=1, dtype="uint8",
        crs="EPSG:4326", transform=tf_w67, tiled=True, blockxsize=64, blockysize=64,
        compress="deflate", nbits=4, nodata=15,
    ) as ds:
        ds.write(u8_nb4, 1)
    out["w67_nbits4_nodata"] = p_w67_nb4

    # Wave 68: Odd prime-dimension (137x113) YCbCr 4:2:0 JPEG-in-TIFF (tiled partial MCU edge + striped)
    yy68, xx68 = np.mgrid[0:113, 0:137]
    rgb_w68 = np.stack([
        (30 + xx68 * 1.4 + yy68 * 0.3).clip(0, 255).astype(np.uint8),
        (220 - xx68 * 0.8 + yy68 * 0.2).clip(0, 255).astype(np.uint8),
        (40 + yy68 * 1.6).clip(0, 255).astype(np.uint8),
    ])
    tf_w68 = from_bounds(-122.6, 37.4, -121.9, 38.1, 137, 113)
    p_w68_yt = synth_dir / "w68_ycbcr_odd_tiled.tif"
    with rasterio.open(
        p_w68_yt, "w", driver="GTiff", width=137, height=113, count=3, dtype="uint8",
        crs="EPSG:4326", transform=tf_w68, tiled=True, blockxsize=64, blockysize=64,
        compress="JPEG", photometric="YCBCR", jpeg_quality=92,
    ) as ds:
        ds.write(rgb_w68)
    out["w68_ycbcr_odd_tiled"] = p_w68_yt

    p_w68_ys = synth_dir / "w68_ycbcr_odd_strip.tif"
    with rasterio.open(
        p_w68_ys, "w", driver="GTiff", width=137, height=113, count=3, dtype="uint8",
        crs="EPSG:4326", transform=tf_w68, tiled=False, blockysize=16,
        compress="JPEG", photometric="YCBCR", jpeg_quality=92,
    ) as ds:
        ds.write(rgb_w68)
    out["w68_ycbcr_odd_strip"] = p_w68_ys

    # Wave 69: Pure 90-deg rotated (a=0,e=0), East-to-West mirrored (a<0), South-up DEM (e>0) & negative-key discrete int16 colormap
    from rasterio.transform import Affine as _Affine
    step69 = 0.7 / 128.0
    p_w69_rot90 = synth_dir / "w69_rot90_4326.tif"
    f32_rot90 = (100.0 + xx9 * 2.0 + yy9 * 1.5).astype(np.float32)
    f32_rot90[60:80, 75:95] = -9999.0
    with rasterio.open(
        p_w69_rot90, "w", driver="GTiff", width=128, height=128, count=1, dtype="float32",
        crs="EPSG:4326", transform=_Affine(0.0, step69, -122.6, -step69, 0.0, 38.1),
        tiled=True, blockxsize=64, blockysize=64, compress="deflate", nodata=-9999.0,
    ) as ds:
        ds.write(f32_rot90, 1)
    out["w69_rot90_4326"] = p_w69_rot90

    p_w69_mirror = synth_dir / "w69_mirror_x_4326.tif"
    f32_mirror = (50.0 + xx9 * 3.0 + yy9 * 1.2).astype(np.float32)
    f32_mirror[75:95, 48:68] = -9999.0
    with rasterio.open(
        p_w69_mirror, "w", driver="GTiff", width=128, height=128, count=1, dtype="float32",
        crs="EPSG:4326", transform=_Affine(-step69, 0.0, -121.9, 0.0, -step69, 38.1),
        tiled=True, blockxsize=64, blockysize=64, compress="deflate", nodata=-9999.0,
    ) as ds:
        ds.write(f32_mirror, 1)
    out["w69_mirror_x_4326"] = p_w69_mirror

    p_w69_su = synth_dir / "w69_southup_dem.tif"
    f32_su = (100.0 + xx9 * 4.0 + yy9 * 6.0).astype(np.float32)
    with rasterio.open(
        p_w69_su, "w", driver="GTiff", width=128, height=128, count=1, dtype="float32",
        crs="EPSG:4326", transform=_Affine(step69, 0.0, -122.6, 0.0, step69, 37.4),
        tiled=True, blockxsize=64, blockysize=64, compress="deflate",
    ) as ds:
        ds.write(f32_su, 1)
    out["w69_southup_dem"] = p_w69_su

    p_w69_negcat = synth_dir / "w69_neg_cat_i16.tif"
    cats69 = np.array([-10, 0, 10, 20], dtype=np.int16)
    i16_cat = cats69[(xx9 // 32 + yy9 // 32) % 4]
    i16_cat[75:95, 60:80] = -9999
    with rasterio.open(
        p_w69_negcat, "w", driver="GTiff", width=128, height=128, count=1, dtype="int16",
        crs="EPSG:4326", transform=tf_w67, tiled=True, blockxsize=64, blockysize=64,
        compress="deflate", nodata=-9999,
    ) as ds:
        ds.write(i16_cat, 1)
    out["w69_neg_cat_i16"] = p_w69_negcat

    # Wave 74: Multi-overview ([2, 4]) COGs with per-overview 1-bit internal masks (NewSubfileType=5)
    yy74, xx74 = np.mgrid[0:512, 0:512]
    tf_w74 = from_bounds(-122.6, 37.4, -121.9, 38.1, 512, 512)
    mask74 = np.full((512, 512), 255, dtype=np.uint8)
    mask74[280:380, 220:340] = 0  # large interior masked hole visible at both z10 and z8 overviews
    mask74[:24, :] = 0            # northern collar masked
    rgb74 = np.stack([
        (30 + (xx74 * 0.42)).clip(0, 255).astype(np.uint8),
        (220 - (yy74 * 0.35)).clip(0, 255).astype(np.uint8),
        (40 + ((xx74 + yy74) * 0.2)).clip(0, 255).astype(np.uint8),
    ])
    p_w74_ycbcr = synth_dir / "w74_ovr_imask_ycbcr.tif"
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rasterio.open(
            p_w74_ycbcr, "w", driver="GTiff", width=512, height=512, count=3, dtype="uint8",
            crs="EPSG:4326", transform=tf_w74, tiled=True, blockxsize=128, blockysize=128,
            compress="JPEG", photometric="YCBCR", jpeg_quality=92,
        ) as ds:
            ds.write(rgb74)
            ds.write_mask(mask74)
            ds.build_overviews([2, 4], rasterio.enums.Resampling.average)
    out["w74_ovr_imask_ycbcr"] = p_w74_ycbcr

    f32_74 = (1500.0 + xx74 * 3.0 + yy74 * 2.0).astype(np.float32)
    p_w74_f32 = synth_dir / "w74_ovr_imask_f32_scaled.tif"
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        with rasterio.open(
            p_w74_f32, "w", driver="GTiff", width=512, height=512, count=1, dtype="float32",
            crs="EPSG:4326", transform=tf_w74, tiled=True, blockxsize=128, blockysize=128,
            compress="deflate", predictor=3,
        ) as ds:
            ds.write(f32_74, 1)
            ds.scales = (0.01,)
            ds.offsets = (-10.0,)
            ds.write_mask(mask74)
            ds.build_overviews([2, 4], rasterio.enums.Resampling.average)
    out["w74_ovr_imask_f32_scaled"] = p_w74_f32

    return out


def compare_stats_strict(j_cw: dict, j_tt: dict) -> tuple[bool, str]:
    if set(j_cw.keys()) != set(j_tt.keys()):
        return False, f"KEYS DIFF cw={sorted(j_cw.keys())} vs tt={sorted(j_tt.keys())}"
    diffs = []
    for k in sorted(j_tt.keys()):
        a, b = j_cw[k], j_tt[k]
        for fld in ("min", "max", "mean", "std", "median", "percentile_2", "percentile_98"):
            va, vb = a.get(fld), b.get(fld)
            if va is None and vb is None:
                continue
            if (va is None) != (vb is None):
                diffs.append(f"{k}.{fld}: cw={va} vs tt={vb}")
                continue
            fa, fb = float(va), float(vb)
            if not (math.isfinite(fa) and math.isfinite(fb)):
                if not (math.isnan(fa) and math.isnan(fb)) and fa != fb:
                    diffs.append(f"{k}.{fld}: cw={fa} vs tt={fb}")
                continue
            scale = max(1e-3, abs(fb))
            if abs(fa - fb) / scale > 2e-2 and abs(fa - fb) > 1e-3:
                diffs.append(f"{k}.{fld}: cw={fa:.6g} vs tt={fb:.6g}")
        for fld in b:
            if fld.startswith("percentile_") and fld not in ("percentile_2", "percentile_98"):
                va, vb = a.get(fld), b.get(fld)
                if va is None and vb is None:
                    continue
                if (va is None) != (vb is None):
                    diffs.append(f"{k}.{fld}: cw={va} vs tt={vb}")
                    continue
                fa, fb = float(va), float(vb)
                if abs(fa - fb) / max(1e-3, abs(fb)) > 2e-2 and abs(fa - fb) > 1e-3:
                    diffs.append(f"{k}.{fld}: cw={fa} vs tt={fb}")
        if "values" in b and a.get("values") != b.get("values"):
            diffs.append(f"{k}.values: cw={a.get('values')} vs tt={b.get('values')}")
        if "histogram" in b:
            ha, hb = a.get("histogram"), b.get("histogram")
            if ha is None or hb is None or len(ha) != 2 or len(hb) != 2:
                diffs.append(f"{k}.histogram: missing/malformed")
            else:
                ca_h, ea_h = ha
                cb_h, eb_h = hb
                if len(ca_h) != len(cb_h) or len(ea_h) != len(eb_h):
                    diffs.append(f"{k}.histogram len: cw=({len(ca_h)},{len(ea_h)}) vs tt=({len(cb_h)},{len(eb_h)})")
                elif eb_h:
                    edge_err = max(abs(float(x) - float(y)) for x, y in zip(ea_h, eb_h))
                    edge_scale = max(1.0, max(abs(float(y)) for y in eb_h))
                    if edge_err / edge_scale > 2e-2:
                        diffs.append(f"{k}.histogram edges err={edge_err:.4g}")
                    sa_h = sum(float(x) for x in ca_h)
                    sb_h = sum(float(y) for y in cb_h)
                    ca_ref = float(a.get("count") or a.get("valid_pixels") or 0.0)
                    cb_ref = float((b.get("count") if "area" in b else b.get("valid_pixels")) or b.get("count") or 0.0)
                    if sb_h > 0.0:
                        if ca_ref > 0.0 and cb_ref > 0.0:
                            if abs(sa_h / ca_ref - sb_h / cb_ref) > 3e-2:
                                diffs.append(f"{k}.histogram frac: cw={sa_h/ca_ref:.3f} vs tt={sb_h/cb_ref:.3f}")
                        elif abs(sa_h - sb_h) / sb_h > 5e-2:
                            diffs.append(f"{k}.histogram sum: cw={sa_h:.1f} vs tt={sb_h:.1f}")
                        if sa_h > 0.0:
                            l1 = sum(abs(float(x) / sa_h - float(y) / sb_h) for x, y in zip(ca_h, cb_h))
                            if l1 > 0.08:
                                diffs.append(f"{k}.histogram counts L1={l1:.3f}")
                    elif sa_h > 0.0:
                        diffs.append(f"{k}.histogram sum: cw={sa_h:.1f} vs tt=0")
        ca, cb = a.get("count"), b.get("count")
        if (ca is None) != (cb is None):
            diffs.append(f"{k}.count: cw={ca} vs tt={cb}")
        elif ca is not None and cb is not None:
            if abs(float(ca) - float(cb)) / max(1.0, float(cb)) > 3e-2:
                diffs.append(f"{k}.count: cw={ca} vs tt={cb}")
    if diffs:
        return False, "; ".join(diffs[:4])
    return True, f"keys={sorted(j_cw.keys())} ok"


def compare_point_strict(j_cw: dict, j_tt: dict) -> tuple[bool, str]:
    vcw, vtt = j_cw.get("values", []), j_tt.get("values", [])
    if len(vcw) != len(vtt):
        return False, f"len cw={len(vcw)} vs tt={len(vtt)}"
    diffs = []
    for i, (a, b) in enumerate(zip(vcw, vtt)):
        if a is None and b is None:
            continue
        if (a is None) != (b is None):
            diffs.append(f"[{i}] cw={a} vs tt={b}")
            continue
        scale = max(1.0, abs(float(b)))
        if abs(float(a) - float(b)) / scale > 1e-3:
            diffs.append(f"[{i}] cw={a} vs tt={b}")
    if diffs:
        return False, "; ".join(diffs)
    return True, f"values={vcw[:4]}"


def start_cogzilla(bin_path: Path, port: int, cpu_only: bool) -> subprocess.Popen:
    env = os.environ.copy()
    if cpu_only:
        env["COGZILLA_GPU_POLICY"] = "never"
    else:
        env.pop("COGZILLA_GPU_POLICY", None)
    return subprocess.Popen(
        [str(bin_path), "--bind", f"127.0.0.1:{port}", "--cache-mb", "256"],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def build_mosaic_fixtures(d: Path) -> dict:
    """Build 1-band, 2-band, and 3-band RGB aligned COGs on exact z=12/657/1582 grid."""
    d.mkdir(parents=True, exist_ok=True)
    b = mercantile.xy_bounds(657, 1582, 12)
    tf = from_bounds(b.left, b.bottom, b.right, b.top, 256, 256)

    s1 = np.full((256, 256), 10.0, dtype=np.float32)
    s2 = np.full((256, 256), 50.0, dtype=np.float32)
    s3 = np.full((256, 256), 20.0, dtype=np.float32)
    yy, xx = np.mgrid[0:256, 0:256].astype(np.float32)
    s1 += xx * 0.1
    s2 += yy * 0.2
    s3 += (xx + yy) * 0.05

    # Staggered nodata holes (-9999.0) covering overlap cardinalities 0, 1, 2, 3:
    s1[0:64, 0:128] = -9999.0
    s2[0:64, 0:64] = -9999.0
    s2[64:128, 0:64] = -9999.0
    s3[0:64, 0:128] = -9999.0

    paths_1b = []
    for idx, arr in enumerate((s1, s2, s3), 1):
        p = d / f"scene_{idx}_1b.tif"
        with rasterio.open(p, "w", driver="GTiff", width=256, height=256, count=1, dtype="float32", crs="EPSG:3857", transform=tf, tiled=True, blockxsize=256, blockysize=256, compress="deflate", nodata=-9999.0) as ds:
            ds.write(arr[None, ...])
        paths_1b.append(str(p.resolve()))

    scores = [30.0, 10.0, 20.0]
    vals = [100.0, 200.0, 300.0]
    paths_2b, s2b_data, s2b_score = [], [], []
    for idx, (val, sc, mask_src) in enumerate(zip(vals, scores, (s1, s2, s3)), 1):
        b1 = np.where(mask_src == -9999.0, -9999.0, val + xx * 0.01).astype(np.float32)
        b2 = np.where(mask_src == -9999.0, -9999.0, sc + yy * 0.01).astype(np.float32)
        p = d / f"scene_{idx}_2b.tif"
        with rasterio.open(p, "w", driver="GTiff", width=256, height=256, count=2, dtype="float32", crs="EPSG:3857", transform=tf, tiled=True, blockxsize=256, blockysize=256, compress="deflate", nodata=-9999.0) as ds:
            ds.write(np.stack([b1, b2], axis=0))
        paths_2b.append(str(p.resolve()))
        s2b_data.append(np.where(b1 == -9999.0, np.nan, b1))
        s2b_score.append(np.where(b2 == -9999.0, np.nan, b2))

    paths_rgb, rgb_scenes = [], []
    rgb_bases = [(40, 120, 200), (180, 60, 100), (100, 220, 50)]
    for idx, ((r0, g0, b0), mask_src) in enumerate(zip(rgb_bases, (s1, s2, s3)), 1):
        r = np.where(mask_src == -9999.0, 0, np.clip(r0 + (xx // 8).astype(np.int32), 1, 255)).astype(np.uint8)
        g = np.where(mask_src == -9999.0, 0, np.clip(g0 + (yy // 8).astype(np.int32), 1, 255)).astype(np.uint8)
        bl = np.where(mask_src == -9999.0, 0, np.clip(b0 + ((xx + yy) // 16).astype(np.int32), 1, 255)).astype(np.uint8)
        arr3 = np.stack([r, g, bl], axis=0)
        p = d / f"scene_{idx}_rgb.tif"
        with rasterio.open(p, "w", driver="GTiff", width=256, height=256, count=3, dtype="uint8", crs="EPSG:3857", transform=tf, tiled=True, blockxsize=256, blockysize=256, compress="deflate", nodata=0) as ds:
            ds.write(arr3)
        paths_rgb.append(str(p.resolve()))
        rgb_scenes.append(np.where(arr3 == 0, np.nan, arr3.astype(np.float64)))

    qk = mercantile.quadkey(657, 1582, 12)
    lonlat = mercantile.bounds(657, 1582, 12)
    bounds_wgs = [lonlat.west, lonlat.south, lonlat.east, lonlat.north]

    mj_1b = d / "mosaic_1b.json"
    mj_1b.write_text(json.dumps({"mosaicjson": "0.0.2", "name": "test-1b", "minzoom": 12, "maxzoom": 12, "quadkey_zoom": 12, "bounds": bounds_wgs, "tiles": {qk: paths_1b}}))
    mj_2b = d / "mosaic_2b.json"
    mj_2b.write_text(json.dumps({"mosaicjson": "0.0.2", "name": "test-2b", "minzoom": 12, "maxzoom": 12, "quadkey_zoom": 12, "bounds": bounds_wgs, "tiles": {qk: paths_2b}}))
    mj_rgb = d / "mosaic_rgb.json"
    mj_rgb.write_text(json.dumps({"mosaicjson": "0.0.2", "name": "test-rgb", "minzoom": 12, "maxzoom": 12, "quadkey_zoom": 12, "bounds": bounds_wgs, "tiles": {qk: paths_rgb}}))

    stack = np.stack([np.where(a == -9999.0, np.nan, a) for a in (s1, s2, s3)], axis=0)
    valid_cnt = np.isfinite(stack).sum(axis=0)
    rgb_stack = np.stack(rgb_scenes, axis=0)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        gt: dict[str, np.ndarray] = {}
        gt["first"] = np.where(np.isfinite(stack[0]), stack[0], np.where(np.isfinite(stack[1]), stack[1], stack[2]))
        gt["highest"] = np.nanmax(stack, axis=0)
        gt["lowest"] = np.nanmin(stack, axis=0)
        gt["mean"] = np.nanmean(stack, axis=0)
        gt["median"] = np.nanmedian(stack, axis=0)
        gt["stdev"] = np.nanstd(stack, axis=0, ddof=0)
        gt["count"] = valid_cnt.astype(np.float32)

        d_stack = np.stack(s2b_data, axis=0)
        sc_stack = np.stack(s2b_score, axis=0)
        idx_low = np.argmin(np.where(np.isfinite(sc_stack), sc_stack, np.inf), axis=0)
        gt["lastbandlow"] = np.where(valid_cnt > 0, np.take_along_axis(d_stack, idx_low[None], axis=0)[0], np.nan)
        idx_high = np.argmax(np.where(np.isfinite(sc_stack), sc_stack, -np.inf), axis=0)
        gt["lastbandhigh"] = np.where(valid_cnt > 0, np.take_along_axis(d_stack, idx_high[None], axis=0)[0], np.nan)

        gt_rgb: dict[str, np.ndarray] = {}
        gt_rgb["first"] = np.where(np.isfinite(rgb_stack[0]), rgb_stack[0], np.where(np.isfinite(rgb_stack[1]), rgb_stack[1], rgb_stack[2]))
        gt_rgb["highest"] = np.nanmax(rgb_stack, axis=0)
        gt_rgb["lowest"] = np.nanmin(rgb_stack, axis=0)
        gt_rgb["mean"] = np.nanmean(rgb_stack, axis=0)
        gt_rgb["median"] = np.nanmedian(rgb_stack, axis=0)
        gt_rgb["stdev"] = np.nanstd(rgb_stack, axis=0, ddof=0)

    return {"mj_1b": mj_1b, "mj_2b": mj_2b, "mj_rgb": mj_rgb, "gt": gt, "gt_rgb": gt_rgb, "bounds_wgs": bounds_wgs, "xy_bounds": b, "s1": s1, "s2": s2, "s3": s3}


def run_wave7_checks(
    client: httpx.Client,
    cw_base: str,
    gpu_base: str | None,
    tmp_dir: Path,
    only_substr: str | None,
    failures: list[dict],
    gpu_diffs: list[dict],
) -> int:
    """Wave 7: MosaicJSON (all 9 pixel_selection methods, 3-band RGB, expr, bbox, point, POST stats) + HTTP protocol contracts."""
    def wants(label: str) -> bool:
        return not only_substr or only_substr.lower() in label.lower()

    all_labels = [
        *[f"W7 MOSAIC {m} .npy" for m in ("first", "highest", "lowest", "mean", "median", "stdev", "count", "lastbandlow", "lastbandhigh")],
        *[f"W7 MOSAIC {m} .png" for m in ("first", "highest", "lowest", "mean", "median", "stdev", "count", "lastbandlow", "lastbandhigh")],
        *[f"W7 MOSAIC RGB {m} .npy" for m in ("first", "highest", "lowest", "mean", "median", "stdev")],
        *[f"W7 MOSAIC expr+{m} .npy" for m in ("first", "mean", "highest")],
        "W7 MOSAIC /point multi-asset",
        *[f"W7 MOSAIC /bbox {m}" for m in ("first", "mean", "highest")],
        *[f"W7 MOSAIC POST /statistics {m}" for m in ("first", "mean", "highest")],
        "W7 HTTP ETag + If-None-Match -> 304",
        "W7 HTTP If-None-Match:* -> 304",
        "W7 HTTP HEAD tile -> 200 empty",
        "W7 HTTP OPTIONS -> 204",
        "W7 HTTP trailing slash /cog/info/ -> 200",
        "W7 HTTP X-Forwarded-{Proto,Host,Prefix}",
        "W7 HTTP Vary: Accept on /colorMaps",
    ]
    if not any(wants(lb) for lb in all_labels):
        return 0

    fx = build_mosaic_fixtures(tmp_dir / "wave7_mosaic")
    mj_1b, mj_2b, mj_rgb = fx["mj_1b"], fx["mj_2b"], fx["mj_rgb"]
    gt, gt_rgb, xb = fx["gt"], fx["gt_rgb"], fx["xy_bounds"]
    sat = FIXTURES / "satellite_utm10n_rgb_zstd_p3.tif"
    merc = FIXTURES / "mercator_4k_zstd_p3.tif"

    ran = 0

    def record(label: str, path: str, kind: str, ok: bool, detail: str, method: str = "GET"):
        nonlocal ran
        ran += 1
        if not ok:
            failures.append({"label": label, "method": method, "path": path, "kind": kind, "detail": detail})
        print(f"{'PASS' if ok else 'FAIL'} | {label:54s} | {detail}")

    # 1. All 9 pixel_selection methods on 1-band (or 2-band for lastband*) .npy + .png
    for method in ("first", "highest", "lowest", "mean", "median", "stdev", "count", "lastbandlow", "lastbandhigh"):
        mj = mj_2b if method in ("lastbandlow", "lastbandhigh") else mj_1b
        lb_npy = f"W7 MOSAIC {method} .npy"
        if wants(lb_npy):
            path = f"/mosaicjson/tiles/WebMercatorQuad/12/657/1582.npy?url={q(mj)}&pixel_selection={method}"
            rc = client.get(cw_base + path)
            if gpu_base is not None:
                rg = client.get(gpu_base + path)
                if rc.status_code != rg.status_code:
                    gpu_diffs.append({"label": lb_npy, "detail": f"status cpu={rc.status_code} gpu={rg.status_code}"})
                elif rc.status_code == 200:
                    gok, gd = compare_npy(rc.content, rg.content, check_values=True)
                    if not gok:
                        gpu_diffs.append({"label": lb_npy, "detail": gd})
            if rc.status_code != 200:
                record(lb_npy, path, "mosaic_gt", False, f"HTTP {rc.status_code} {rc.text[:100]!r}")
            else:
                arr = np.load(io.BytesIO(rc.content))
                low = np.finfo(arr.dtype).min if arr.dtype.kind == "f" else np.iinfo(arr.dtype).min
                mask = arr[-1] > low
                exp = gt[method]
                shape_ok = arr.shape == (2, 256, 256)
                if method == "count":
                    max_err = float(np.max(np.abs(arr[0].astype(np.float64) - exp.astype(np.float64))))
                    all_valid = bool(mask.all())
                    record(lb_npy, path, "mosaic_gt", shape_ok and all_valid and max_err < 1e-4, f"shape={arr.shape} dtype={arr.dtype} all_valid={all_valid} max_err={max_err:.6f}")
                else:
                    exp_mask = np.isfinite(exp)
                    mask_eq = float((mask == exp_mask).mean()) * 100.0
                    both = mask & exp_mask
                    max_err = float(np.max(np.abs(arr[0][both].astype(np.float64) - exp[both].astype(np.float64)))) if both.any() else 0.0
                    record(lb_npy, path, "mosaic_gt", shape_ok and mask_eq == 100.0 and max_err < 1e-2, f"shape={arr.shape} dtype={arr.dtype} mask_eq={mask_eq:.1f}% max_err={max_err:.5f}")

        lb_png = f"W7 MOSAIC {method} .png"
        if wants(lb_png):
            path = f"/mosaicjson/tiles/WebMercatorQuad/12/657/1582.png?url={q(mj)}&pixel_selection={method}&rescale=0,300&colormap_name=viridis"
            rc = client.get(cw_base + path)
            if rc.status_code != 200:
                record(lb_png, path, "mosaic_png", False, f"HTTP {rc.status_code} {rc.text[:100]!r}")
            else:
                try:
                    rgba = np.array(Image.open(io.BytesIO(rc.content)).convert("RGBA"))
                except Exception as exc:
                    record(lb_png, path, "mosaic_png", False, f"invalid PNG: {exc}")
                    continue
                alpha = rgba[..., 3]
                if method == "count":
                    alpha_ok = bool((alpha == 255).all())
                    valid_rgb = rgba[..., :3]
                else:
                    exp_mask = np.isfinite(gt[method])
                    alpha_ok = bool(((alpha > 0) == exp_mask).all())
                    valid_rgb = rgba[exp_mask, :3]
                rgb_std = float(valid_rgb.std()) if valid_rgb.size else 0.0
                ok = rgba.shape == (256, 256, 4) and alpha_ok and rgb_std > 0.5
                detail = f"shape={rgba.shape} alpha_ok={alpha_ok} rgb_std={rgb_std:.1f}"
                if gpu_base is not None:
                    rg = client.get(gpu_base + path)
                    if rg.status_code != 200:
                        gpu_diffs.append({"label": lb_png, "detail": f"status cpu=200 gpu={rg.status_code}"})
                    else:
                        iok, idet = compare_images(rc.content, rg.content)
                        if not iok:
                            gpu_diffs.append({"label": lb_png, "detail": idet})
                        detail = f"{detail} | CPU==GPU {idet}"
                record(lb_png, path, "mosaic_png", ok, detail)

    # 2. 3-band RGB mosaic with pixel_selection
    for method in ("first", "highest", "lowest", "mean", "median", "stdev"):
        lb = f"W7 MOSAIC RGB {method} .npy"
        if not wants(lb):
            continue
        path = f"/mosaicjson/tiles/WebMercatorQuad/12/657/1582.npy?url={q(mj_rgb)}&pixel_selection={method}"
        rc = client.get(cw_base + path)
        if gpu_base is not None:
            rg = client.get(gpu_base + path)
            if rc.status_code != rg.status_code:
                gpu_diffs.append({"label": lb, "detail": f"status cpu={rc.status_code} gpu={rg.status_code}"})
            elif rc.status_code == 200:
                gok, gd = compare_npy(rc.content, rg.content, check_values=True)
                if not gok:
                    gpu_diffs.append({"label": lb, "detail": gd})
        if rc.status_code != 200:
            record(lb, path, "mosaic_gt", False, f"HTTP {rc.status_code} {rc.text[:100]!r}")
        else:
            arr = np.load(io.BytesIO(rc.content))
            low = np.finfo(arr.dtype).min if arr.dtype.kind == "f" else np.iinfo(arr.dtype).min
            mask = arr[-1] > low
            exp = gt_rgb[method]
            exp_mask = np.isfinite(exp[0])
            mask_eq = float((mask == exp_mask).mean()) * 100.0
            both = mask & exp_mask
            shape_ok = arr.shape == (4, 256, 256)
            if arr.dtype == np.uint8:
                # Integer mean/median truncates toward zero in cogzilla; first/highest/lowest are exact.
                ref = np.trunc(exp) if method in ("mean", "median") else exp
                max_err = max(float(np.max(np.abs(arr[bi][both].astype(np.float64) - ref[bi][both]))) for bi in range(3)) if both.any() else 0.0
                tol = 0.0
            else:
                max_err = max(float(np.max(np.abs(arr[bi][both].astype(np.float64) - exp[bi][both]))) for bi in range(3)) if both.any() else 0.0
                tol = 1e-2
            record(lb, path, "mosaic_gt", shape_ok and mask_eq == 100.0 and max_err <= tol, f"shape={arr.shape} dtype={arr.dtype} mask_eq={mask_eq:.1f}% max_err={max_err:.4f}")

    # 3. MosaicJSON expression + pixel_selection
    for method in ("first", "mean", "highest"):
        lb = f"W7 MOSAIC expr+{method} .npy"
        if not wants(lb):
            continue
        path = f"/mosaicjson/tiles/WebMercatorQuad/12/657/1582.npy?url={q(mj_1b)}&pixel_selection={method}&expression={qe('b1*2.0+5.0')}"
        rc = client.get(cw_base + path)
        if gpu_base is not None:
            rg = client.get(gpu_base + path)
            if rc.status_code != rg.status_code:
                gpu_diffs.append({"label": lb, "detail": f"status cpu={rc.status_code} gpu={rg.status_code}"})
            elif rc.status_code == 200:
                gok, gd = compare_npy(rc.content, rg.content, check_values=True)
                if not gok:
                    gpu_diffs.append({"label": lb, "detail": gd})
        if rc.status_code != 200:
            record(lb, path, "mosaic_gt", False, f"HTTP {rc.status_code} {rc.text[:100]!r}")
        else:
            arr = np.load(io.BytesIO(rc.content))
            low = np.finfo(arr.dtype).min if arr.dtype.kind == "f" else np.iinfo(arr.dtype).min
            mask = arr[-1] > low
            exp = gt[method] * 2.0 + 5.0
            exp_mask = np.isfinite(exp)
            both = mask & exp_mask
            mask_eq = float((mask == exp_mask).mean()) * 100.0
            max_err = float(np.max(np.abs(arr[0][both].astype(np.float64) - exp[both]))) if both.any() else 0.0
            record(lb, path, "mosaic_gt", arr.shape == (2, 256, 256) and mask_eq == 100.0 and max_err < 1e-2, f"shape={arr.shape} dtype={arr.dtype} mask_eq={mask_eq:.1f}% max_err={max_err:.5f}")

    # 4. MosaicJSON /point (multi-asset response at exact EPSG:3857 pixel center [row=128, col=128])
    dx = (xb.right - xb.left) / 256.0
    dy = (xb.top - xb.bottom) / 256.0
    pt_ll = mercantile.lnglat(xb.left + 128.5 * dx, xb.top - 128.5 * dy)
    lb_pt = "W7 MOSAIC /point multi-asset"
    if wants(lb_pt):
        path = f"/mosaicjson/point/{pt_ll.lng:.8f},{pt_ll.lat:.8f}?url={q(mj_1b)}"
        rc = client.get(cw_base + path)
        if gpu_base is not None:
            rg = client.get(gpu_base + path)
            if rc.status_code != rg.status_code or (rc.status_code == 200 and first_json_diff(rc.json(), rg.json()) is not None):
                gpu_diffs.append({"label": lb_pt, "detail": f"cpu={rc.text[:80]} gpu={rg.text[:80]}"})
        if rc.status_code != 200:
            record(lb_pt, path, "mosaic_point", False, f"HTTP {rc.status_code} {rc.text[:100]!r}")
        else:
            assets = rc.json().get("assets", [])
            got_vals = [a["values"][0] for a in assets]
            exp_vals = [float(fx["s1"][128, 128]), float(fx["s2"][128, 128]), float(fx["s3"][128, 128])]
            ok = len(got_vals) == 3 and all(abs(g - e) < 1e-3 for g, e in zip(got_vals, exp_vals))
            record(lb_pt, path, "mosaic_point", ok, f"got={[round(v, 4) for v in got_vals]} exp={[round(v, 4) for v in exp_vals]}")

    # 5. MosaicJSON /bbox
    bbox_s = f"{xb.left:.4f},{xb.bottom:.4f},{xb.right:.4f},{xb.top:.4f}"
    for method in ("first", "mean", "highest"):
        lb = f"W7 MOSAIC /bbox {method}"
        if not wants(lb):
            continue
        path = f"/mosaicjson/bbox/{bbox_s}/256x256.npy?url={q(mj_1b)}&coord_crs=EPSG:3857&dst_crs=EPSG:3857&pixel_selection={method}"
        rc = client.get(cw_base + path)
        if gpu_base is not None:
            rg = client.get(gpu_base + path)
            if rc.status_code != rg.status_code:
                gpu_diffs.append({"label": lb, "detail": f"status cpu={rc.status_code} gpu={rg.status_code}"})
            elif rc.status_code == 200:
                gok, gd = compare_npy(rc.content, rg.content, check_values=True)
                if not gok:
                    gpu_diffs.append({"label": lb, "detail": gd})
        if rc.status_code != 200:
            record(lb, path, "mosaic_gt", False, f"HTTP {rc.status_code} {rc.text[:100]!r}")
        else:
            arr = np.load(io.BytesIO(rc.content))
            low = np.finfo(arr.dtype).min if arr.dtype.kind == "f" else np.iinfo(arr.dtype).min
            mask = arr[-1] > low
            exp = gt[method]
            exp_mask = np.isfinite(exp)
            both = mask & exp_mask
            mask_eq = float((mask == exp_mask).mean()) * 100.0
            max_err = float(np.max(np.abs(arr[0][both].astype(np.float64) - exp[both]))) if both.any() else 0.0
            record(lb, path, "mosaic_gt", arr.shape == (2, 256, 256) and mask_eq == 100.0 and max_err < 1e-2, f"shape={arr.shape} mask_eq={mask_eq:.1f}% max_err={max_err:.5f}")

    # 6. MosaicJSON POST /statistics over west half [0:256, 0:128] covering overlap cardinalities 0, 1, 2, and 3!
    mid_x = 0.5 * (xb.left + xb.right)
    sw = mercantile.lnglat(xb.left, xb.bottom)
    se_mid = mercantile.lnglat(mid_x, xb.bottom)
    ne_mid = mercantile.lnglat(mid_x, xb.top)
    nw = mercantile.lnglat(xb.left, xb.top)
    poly_west = {
        "type": "Feature",
        "properties": {},
        "geometry": {"type": "Polygon", "coordinates": [[[sw.lng, sw.lat], [se_mid.lng, se_mid.lat], [ne_mid.lng, ne_mid.lat], [nw.lng, nw.lat], [sw.lng, sw.lat]]]},
    }
    for method in ("first", "mean", "highest"):
        lb = f"W7 MOSAIC POST /statistics {method}"
        if not wants(lb):
            continue
        path = f"/mosaicjson/statistics?url={q(mj_1b)}&pixel_selection={method}"
        rc = client.post(cw_base + path, json=poly_west)
        if gpu_base is not None:
            rg = client.post(gpu_base + path, json=poly_west)
            if rc.status_code != rg.status_code:
                gpu_diffs.append({"label": lb, "detail": f"status cpu={rc.status_code} gpu={rg.status_code}"})
            elif rc.status_code == 200:
                jd = first_json_diff(rc.json(), rg.json())
                if jd is not None:
                    gpu_diffs.append({"label": lb, "detail": f"CPU vs GPU stats diff: {jd}"})
        if rc.status_code != 200:
            record(lb, path, "mosaic_stats", False, f"HTTP {rc.status_code} {rc.text[:100]!r}", method="POST")
        else:
            st = rc.json()["properties"]["statistics"]["b1"]
            sub = gt[method][:, 0:128]
            valid_sub = sub[np.isfinite(sub)]
            exp_mean = float(valid_sub.mean())
            exp_min = float(valid_sub.min())
            exp_max = float(valid_sub.max())
            exp_count = float(valid_sub.size)
            mean_err = abs(st["mean"] - exp_mean)
            min_err = abs(st["min"] - exp_min)
            max_err = abs(st["max"] - exp_max)
            cnt_rel = abs(st["count"] - exp_count) / exp_count
            ok = mean_err < 0.1 and min_err < 0.2 and max_err < 0.2 and cnt_rel < 0.02
            record(lb, path, "mosaic_stats", ok, f"mean={st['mean']:.3f}/{exp_mean:.3f} min={st['min']:.2f}/{exp_min:.2f} max={st['max']:.2f}/{exp_max:.2f} cnt={st['count']:.0f}/{exp_count:.0f}", method="POST")

    # 7. HTTP protocol contracts (ETag/304, HEAD, OPTIONS, trailing slash, X-Forwarded-*, Vary: Accept)
    tile_u = f"/cog/tiles/WebMercatorQuad/12/657/1582.png?url={q(merc)}&rescale=0,1000&colormap_name=viridis"
    etag: str | None = None
    if wants("W7 HTTP ETag + If-None-Match -> 304"):
        etag = client.get(cw_base + tile_u).headers.get("etag")

    def check_x_forwarded() -> tuple[bool, str]:
        r = client.get(
            cw_base + f"/cog/WebMercatorQuad/tilejson.json?url={q(sat)}",
            headers={"X-Forwarded-Proto": "https", "X-Forwarded-Host": "tiles.example.com", "X-Forwarded-Prefix": "/v1"},
        )
        if r.status_code != 200:
            return False, f"HTTP {r.status_code} {r.text[:80]!r}"
        tiles = r.json().get("tiles") or []
        tile0 = tiles[0] if tiles else ""
        return tile0.startswith("https://tiles.example.com/v1/"), f"tile0={tile0[:60]}"

    proto_specs = [
        ("W7 HTTP ETag + If-None-Match -> 304", "GET", tile_u, lambda: (
            (r := client.get(cw_base + tile_u, headers={"If-None-Match": etag or ""})),
            bool(etag) and r.status_code == 304 and len(r.content) == 0 and r.headers.get("etag") == etag,
            f"etag={etag} status={r.status_code}",
        )[1:]),
        ("W7 HTTP If-None-Match:* -> 304", "GET", tile_u, lambda: (
            (r := client.get(cw_base + tile_u, headers={"If-None-Match": "*"})),
            r.status_code == 304 and len(r.content) == 0,
            f"status={r.status_code}",
        )[1:]),
        ("W7 HTTP HEAD tile -> 200 empty", "HEAD", tile_u, lambda: (
            (r := client.head(cw_base + tile_u)),
            r.status_code == 200 and len(r.content) == 0,
            f"status={r.status_code} len={len(r.content)}",
        )[1:]),
        ("W7 HTTP OPTIONS -> 204", "OPTIONS", f"/cog/info?url={q(sat)}", lambda: (
            (r := client.options(cw_base + f"/cog/info?url={q(sat)}")),
            r.status_code == 204,
            f"status={r.status_code}",
        )[1:]),
        ("W7 HTTP trailing slash /cog/info/ -> 200", "GET", f"/cog/info/?url={q(sat)}", lambda: (
            (r := client.get(cw_base + f"/cog/info/?url={q(sat)}")),
            r.status_code == 200,
            f"status={r.status_code}",
        )[1:]),
        ("W7 HTTP X-Forwarded-{Proto,Host,Prefix}", "GET", f"/cog/WebMercatorQuad/tilejson.json?url={q(sat)}", check_x_forwarded),
        ("W7 HTTP Vary: Accept on /colorMaps", "GET", "/colorMaps", lambda: (
            (r := client.get(cw_base + "/colorMaps")),
            r.status_code == 200 and "accept" in r.headers.get("vary", "").lower(),
            f"status={r.status_code} vary={r.headers.get('vary')!r}",
        )[1:]),
    ]
    for lb, http_method, path, fn in proto_specs:
        if not wants(lb):
            continue
        ok, det = fn()
        record(lb, path, "http_proto", ok, det, method=http_method)

    return ran


def _write_v2_zarr_array(store: Path, name: str, arr: np.ndarray, chunks: tuple[int, ...], attrs: dict, compressor: dict | None = None, fill_value=None):
    ad = store / name
    ad.mkdir(parents=True, exist_ok=True)
    meta = {
        "chunks": list(chunks),
        "compressor": compressor,
        "dtype": arr.dtype.str,
        "fill_value": fill_value,
        "filters": None,
        "order": "C",
        "shape": list(arr.shape),
        "zarr_format": 2,
    }
    (ad / ".zarray").write_text(json.dumps(meta))
    (ad / ".zattrs").write_text(json.dumps(attrs))
    grid = [int(np.ceil(s / c)) for s, c in zip(arr.shape, chunks)]
    for idx in np.ndindex(*grid):
        slices = tuple(slice(i * c, min((i + 1) * c, s)) for i, c, s in zip(idx, chunks, arr.shape))
        sub = arr[slices]
        if sub.shape != chunks:
            pad = np.full(chunks, fill_value if fill_value is not None else 0, dtype=arr.dtype)
            pad[tuple(slice(0, x) for x in sub.shape)] = sub
            sub = pad
        raw = np.ascontiguousarray(sub).tobytes()
        if compressor is None:
            enc = raw
        elif compressor["id"] == "zlib":
            enc = zlib.compress(raw, compressor.get("level", 1))
        elif compressor["id"] == "gzip":
            enc = gzip.compress(raw, compresslevel=compressor.get("level", 1))
        else:
            raise ValueError(compressor)
        key = ".".join(str(i) for i in idx) if idx else "0"
        (ad / key).write_bytes(enc)


def _write_v3_zarr_array(store: Path, name: str, arr: np.ndarray, chunks: tuple[int, ...], attrs: dict, dim_names: list[str], fill_value, use_gzip: bool = True):
    ad = store / name
    ad.mkdir(parents=True, exist_ok=True)
    codecs = [{"name": "bytes", "configuration": {"endian": "little"}}]
    if use_gzip:
        codecs.append({"name": "gzip", "configuration": {"level": 1}})
    meta = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": list(arr.shape),
        "data_type": arr.dtype.name,
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": list(chunks)}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "fill_value": fill_value,
        "codecs": codecs,
        "dimension_names": dim_names,
        "attributes": attrs,
    }
    (ad / "zarr.json").write_text(json.dumps(meta))
    grid = [int(np.ceil(s / c)) for s, c in zip(arr.shape, chunks)]
    for idx in np.ndindex(*grid):
        slices = tuple(slice(i * c, min((i + 1) * c, s)) for i, c, s in zip(idx, chunks, arr.shape))
        sub = np.ascontiguousarray(arr[slices]).tobytes()
        enc = gzip.compress(sub, compresslevel=1) if use_gzip else sub
        cp = ad / "c" / Path(*[str(i) for i in idx])
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_bytes(enc)


def build_zarr_fixtures(d: Path) -> dict:
    """Build 4 synthetic Zarr v2/v3 stores covering 3D time slicing, EPSG:3857, south-up v3, and 0..360° lon."""
    d.mkdir(parents=True, exist_ok=True)
    lat = np.linspace(38.0 - 0.5 / 128, 37.0 + 0.5 / 128, 128, dtype=np.float64)
    lon = np.linspace(-123.0 + 0.5 / 128, -122.0 - 0.5 / 128, 128, dtype=np.float64)
    yy, xx = np.mgrid[0:128, 0:128].astype(np.float32)
    t0 = 20.0 + xx * 0.5 + yy * 0.25
    t1 = 100.0 + xx * 1.0 - yy * 0.25
    data_3d = np.stack([t0, t1], axis=0).astype(np.float32)
    data_3d[:, 0:32, 0:32] = -9999.0

    # 1. Zarr v2 Geographic 3D (time=2, lat=128, lon=128)
    s_v2_geo = d / "v2_geo_3d.zarr"
    s_v2_geo.mkdir()
    (s_v2_geo / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
    (s_v2_geo / ".zattrs").write_text(json.dumps({"Conventions": "CF-1.8"}))
    _write_v2_zarr_array(s_v2_geo, "time", np.array([0, 1], dtype=np.int64), (2,), {"_ARRAY_DIMENSIONS": ["time"]})
    _write_v2_zarr_array(s_v2_geo, "lat", lat, (128,), {"_ARRAY_DIMENSIONS": ["lat"], "standard_name": "latitude", "units": "degrees_north"})
    _write_v2_zarr_array(s_v2_geo, "lon", lon, (128,), {"_ARRAY_DIMENSIONS": ["lon"], "standard_name": "longitude", "units": "degrees_east"})
    _write_v2_zarr_array(s_v2_geo, "temp", data_3d, (1, 64, 64), {"_ARRAY_DIMENSIONS": ["time", "lat", "lon"], "scale_factor": 0.5, "add_offset": 10.0}, compressor={"id": "zlib", "level": 1}, fill_value=-9999.0)

    # 2. Zarr v2 Projected EPSG:3857 2D (256x256) aligned to z=12/657/1582
    s_v2_proj = d / "v2_proj_3857.zarr"
    s_v2_proj.mkdir()
    (s_v2_proj / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
    (s_v2_proj / ".zattrs").write_text(json.dumps({"Conventions": "CF-1.8"}))
    b = mercantile.xy_bounds(657, 1582, 12)
    dx, dy = (b.right - b.left) / 256.0, (b.top - b.bottom) / 256.0
    x_c = np.linspace(b.left + 0.5 * dx, b.right - 0.5 * dx, 256, dtype=np.float64)
    y_c = np.linspace(b.top - 0.5 * dy, b.bottom + 0.5 * dy, 256, dtype=np.float64)
    yy2, xx2 = np.mgrid[0:256, 0:256]
    elev = (500 + xx2 * 4 + yy2 * 2).astype(np.int16)
    elev[0:64, 0:64] = -32768
    wkt_3857 = RasterioCRS.from_epsg(3857).to_wkt()
    _write_v2_zarr_array(s_v2_proj, "x", x_c, (256,), {"_ARRAY_DIMENSIONS": ["x"], "standard_name": "projection_x_coordinate", "units": "m"})
    _write_v2_zarr_array(s_v2_proj, "y", y_c, (256,), {"_ARRAY_DIMENSIONS": ["y"], "standard_name": "projection_y_coordinate", "units": "m"})
    _write_v2_zarr_array(s_v2_proj, "spatial_ref", np.array(0, dtype=np.int64), (), {"_ARRAY_DIMENSIONS": [], "crs_wkt": wkt_3857, "spatial_ref": wkt_3857})
    _write_v2_zarr_array(s_v2_proj, "elev", elev, (128, 128), {"_ARRAY_DIMENSIONS": ["y", "x"], "grid_mapping": "spatial_ref", "scale_factor": 0.1, "add_offset": 100.0}, compressor={"id": "gzip", "level": 1}, fill_value=-32768)

    # 3. Zarr v3 Geographic 2D with ASCENDING latitude (south-up!), float64, NaN hole
    s_v3_southup = d / "v3_southup.zarr"
    s_v3_southup.mkdir()
    (s_v3_southup / "zarr.json").write_text(json.dumps({"zarr_format": 3, "node_type": "group", "attributes": {"Conventions": "CF-1.8"}}))
    lat_asc = np.linspace(37.0 + 0.5 / 128, 38.0 - 0.5 / 128, 128, dtype=np.float64)
    v3_arr = (lat_asc[:, None] * 10.0 + (lon[None, :] + 130.0)).astype(np.float64)
    v3_arr[0:32, 0:32] = np.nan
    _write_v3_zarr_array(s_v3_southup, "lat", lat_asc, (128,), {"standard_name": "latitude", "units": "degrees_north"}, ["lat"], "NaN", use_gzip=False)
    _write_v3_zarr_array(s_v3_southup, "lon", lon, (128,), {"standard_name": "longitude", "units": "degrees_east"}, ["lon"], "NaN", use_gzip=False)
    _write_v3_zarr_array(s_v3_southup, "sst", v3_arr, (64, 64), {"units": "K"}, ["lat", "lon"], "NaN", use_gzip=True)

    # 4. Zarr v2 0..360° longitude grid
    s_v2_360 = d / "v2_lon360.zarr"
    s_v2_360.mkdir()
    (s_v2_360 / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
    (s_v2_360 / ".zattrs").write_text(json.dumps({"Conventions": "CF-1.8"}))
    _write_v2_zarr_array(s_v2_360, "lat", lat, (128,), {"_ARRAY_DIMENSIONS": ["lat"], "standard_name": "latitude"})
    _write_v2_zarr_array(s_v2_360, "lon", lon + 360.0, (128,), {"_ARRAY_DIMENSIONS": ["lon"], "standard_name": "longitude"})
    _write_v2_zarr_array(s_v2_360, "var360", t0.astype(np.float32), (64, 64), {"_ARRAY_DIMENSIONS": ["lat", "lon"]}, compressor={"id": "zlib", "level": 1}, fill_value=-9999.0)

    return {"v2_geo": s_v2_geo, "v2_proj": s_v2_proj, "v3_southup": s_v3_southup, "v2_360": s_v2_360, "data_3d": data_3d, "elev": elev, "v3_arr": v3_arr, "lat": lat, "lon": lon, "lat_asc": lat_asc}


def run_wave8_checks(
    client: httpx.Client,
    cw_base: str,
    gpu_base: str | None,
    tmp_dir: Path,
    only_substr: str | None,
    failures: list[dict],
    gpu_diffs: list[dict],
) -> int:
    """Wave 8: Zarr v2 & v3 adversarial oracle (tiles .npy/.png, return_mask=false, sel=, unscale, south-up v3, 0..360° lon, preview, bbox, GET/POST stats, ETag/HEAD)."""
    def wants(label: str) -> bool:
        return not only_substr or only_substr.lower() in label.lower()

    all_labels = [
        "W8 ZARR v2 /dataset/keys",
        "W8 ZARR v2 /info",
        "W8 ZARR v3 /validate",
        "W8 ZARR v2 proj tile .npy unscale=False",
        "W8 ZARR v2 proj tile .npy unscale=True",
        "W8 ZARR v2 proj tile .npy return_mask=false",
        "W8 ZARR v2 proj tile .png",
        *[f"W8 ZARR v2 3D /point sel=time={t} unscale={u}" for t in (0, 1) for u in (False, True)],
        "W8 ZARR v2 3D /preview/128x128.npy sel=time=1",
        "W8 ZARR v2 3D /bbox 128x128.npy sel=time=0",
        "W8 ZARR v2 3D GET /statistics sel=time=0",
        "W8 ZARR v2 3D POST /statistics west half",
        "W8 ZARR v3 south-up /point NE",
        "W8 ZARR v3 south-up /bbox 128x128.npy",
        "W8 ZARR v2 0..360° lon /point (-122.5, 37.5)",
        "W8 ZARR tile ETag + If-None-Match -> 304",
        "W8 ZARR tile HEAD -> 200 empty",
        "W8 ZARR tile cache isolates .npy vs .png",
    ]
    if not any(wants(lb) for lb in all_labels):
        return 0

    fx = build_zarr_fixtures(tmp_dir / "wave8_zarr")
    ran = 0

    def rec(label: str, path: str, kind: str, ok: bool, detail: str, method: str = "GET"):
        nonlocal ran
        ran += 1
        if not ok:
            failures.append({"label": label, "method": method, "path": path, "kind": kind, "detail": detail})
        print(f"{'PASS' if ok else 'FAIL'} | {label:54s} | {detail}")

    def try_npy(label: str, path: str, raw: bytes):
        try:
            return np.load(io.BytesIO(raw))
        except Exception as exc:
            rec(label, path, "zarr_npy", False, f"invalid .npy (len={len(raw)}, magic={raw[:8]!r}): {exc}")
            return None

    # 1. Metadata & introspection
    for lb, path, check_fn in [
        ("W8 ZARR v2 /dataset/keys", f"/zarr/dataset/keys?url={q(fx['v2_geo'])}", lambda r: r.status_code == 200 and "temp" in r.json()),
        ("W8 ZARR v2 /info", f"/zarr/info?url={q(fx['v2_geo'])}&variable=temp", lambda r: r.status_code == 200 and r.json().get("width") == 128 and r.json().get("height") == 128),
        ("W8 ZARR v3 /validate", f"/zarr/validate?url={q(fx['v3_southup'])}&variable=sst", lambda r: r.status_code == 200 and r.json().get("bounds") == [-123.0, 37.0, -122.0, 38.0]),
    ]:
        if wants(lb):
            r = client.get(cw_base + path)
            ok = check_fn(r) if r.status_code == 200 else False
            rec(lb, path, "zarr_meta", ok, f"status={r.status_code} {r.text[:80]}")

    # 2. Zarr v2 Projected EPSG:3857 exact tile z=12/657/1582 (.npy unscale=False/True, return_mask=false, .png)
    for unscale in (False, True):
        lb = f"W8 ZARR v2 proj tile .npy unscale={unscale}"
        if not wants(lb):
            continue
        path = f"/zarr/tiles/WebMercatorQuad/12/657/1582.npy?url={q(fx['v2_proj'])}&variable=elev&unscale={'true' if unscale else 'false'}"
        rc = client.get(cw_base + path)
        if gpu_base is not None:
            rg = client.get(gpu_base + path)
            if rc.status_code != rg.status_code:
                gpu_diffs.append({"label": lb, "detail": f"status cpu={rc.status_code} gpu={rg.status_code}"})
            elif rc.status_code == 200:
                gok, gd = compare_npy(rc.content, rg.content, check_values=True)
                if not gok:
                    gpu_diffs.append({"label": lb, "detail": gd})
        if rc.status_code != 200:
            rec(lb, path, "zarr_npy", False, f"HTTP {rc.status_code} {rc.text[:100]!r}")
        else:
            arr = try_npy(lb, path, rc.content)
            if arr is not None:
                low = np.finfo(arr.dtype).min if arr.dtype.kind == "f" else np.iinfo(arr.dtype).min
                mask = arr[-1] > low
                exp_mask = fx["elev"] != -32768
                exp_val = fx["elev"].astype(np.float64) * 0.1 + 100.0 if unscale else fx["elev"].astype(np.float64)
                mask_eq = float((mask == exp_mask).mean()) * 100.0
                both = mask & exp_mask
                max_err = float(np.max(np.abs(arr[0][both].astype(np.float64) - exp_val[both]))) if both.any() else 0.0
                rec(lb, path, "zarr_npy", arr.shape == (2, 256, 256) and mask_eq == 100.0 and max_err < 1e-2, f"shape={arr.shape} dtype={arr.dtype} mask_eq={mask_eq:.1f}% max_err={max_err:.5f}")

    lb_nm = "W8 ZARR v2 proj tile .npy return_mask=false"
    if wants(lb_nm):
        path = f"/zarr/tiles/WebMercatorQuad/12/657/1582.npy?url={q(fx['v2_proj'])}&variable=elev&return_mask=false"
        rc = client.get(cw_base + path)
        if rc.status_code != 200:
            rec(lb_nm, path, "zarr_npy", False, f"HTTP {rc.status_code}")
        else:
            arr = try_npy(lb_nm, path, rc.content)
            if arr is not None:
                valid = fx["elev"] != -32768
                max_err = float(np.max(np.abs(arr[0][valid].astype(np.float64) - fx["elev"][valid].astype(np.float64))))
                zeroed = bool((arr[0][~valid] == 0.0).all())
                rec(lb_nm, path, "zarr_npy", arr.shape == (1, 256, 256) and zeroed and max_err < 1e-3, f"shape={arr.shape} zeroed={zeroed} max_err={max_err:.5f}")

    lb_png = "W8 ZARR v2 proj tile .png"
    if wants(lb_png):
        path = f"/zarr/tiles/WebMercatorQuad/12/657/1582.png?url={q(fx['v2_proj'])}&variable=elev&rescale=500,2000&colormap_name=viridis"
        rc = client.get(cw_base + path)
        if rc.status_code != 200:
            rec(lb_png, path, "zarr_png", False, f"HTTP {rc.status_code}")
        else:
            rgba = np.array(Image.open(io.BytesIO(rc.content)).convert("RGBA"))
            exp_mask = fx["elev"] != -32768
            alpha_ok = bool(((rgba[..., 3] > 0) == exp_mask).all())
            rgb_std = float(rgba[exp_mask, :3].std())
            detail = f"shape={rgba.shape} alpha_ok={alpha_ok} rgb_std={rgb_std:.1f}"
            if gpu_base is not None:
                rg = client.get(gpu_base + path)
                if rg.status_code != 200:
                    gpu_diffs.append({"label": lb_png, "detail": f"status cpu=200 gpu={rg.status_code}"})
                else:
                    iok, idet = compare_images(rc.content, rg.content)
                    if not iok:
                        gpu_diffs.append({"label": lb_png, "detail": idet})
                    detail = f"{detail} | CPU==GPU {idet}"
            rec(lb_png, path, "zarr_png", rgba.shape == (256, 256, 4) and alpha_ok and rgb_std > 1.0, detail)

    # 3. Zarr v2 Geographic 3D: sel=time=0/1 & unscale=False/True on /point
    lat_p, lon_p = float(fx["lat"][64]), float(fx["lon"][64])
    for t_idx in (0, 1):
        for unscale in (False, True):
            lb = f"W8 ZARR v2 3D /point sel=time={t_idx} unscale={unscale}"
            if not wants(lb):
                continue
            path = f"/zarr/point/{lon_p:.8f},{lat_p:.8f}?url={q(fx['v2_geo'])}&variable=temp&sel={qe(f'time={t_idx}')}&unscale={'true' if unscale else 'false'}"
            rc = client.get(cw_base + path)
            if gpu_base is not None:
                rg = client.get(gpu_base + path)
                if rc.status_code != rg.status_code or (rc.status_code == 200 and first_json_diff(rc.json(), rg.json()) is not None):
                    gpu_diffs.append({"label": lb, "detail": f"cpu={rc.text[:80]} gpu={rg.text[:80]}"})
            if rc.status_code != 200:
                rec(lb, path, "zarr_point", False, f"HTTP {rc.status_code} {rc.text[:100]!r}")
            else:
                got = rc.json()["values"][0]
                raw_v = float(fx["data_3d"][t_idx, 64, 64])
                exp = raw_v * 0.5 + 10.0 if unscale else raw_v
                rec(lb, path, "zarr_point", abs(got - exp) < 1e-3, f"got={got:.4f} exp={exp:.4f}")

    # 4. Zarr v2 Geographic 3D /preview & /bbox
    for lb, path, exp_arr in [
        ("W8 ZARR v2 3D /preview/128x128.npy sel=time=1", f"/zarr/preview/128x128.npy?url={q(fx['v2_geo'])}&variable=temp&sel={qe('time=1')}", fx["data_3d"][1]),
        ("W8 ZARR v2 3D /bbox 128x128.npy sel=time=0", f"/zarr/bbox/-123,37,-122,38/128x128.npy?url={q(fx['v2_geo'])}&variable=temp&sel={qe('time=0')}", fx["data_3d"][0]),
    ]:
        if not wants(lb):
            continue
        rc = client.get(cw_base + path)
        if gpu_base is not None:
            rg = client.get(gpu_base + path)
            if rc.status_code != rg.status_code:
                gpu_diffs.append({"label": lb, "detail": f"status cpu={rc.status_code} gpu={rg.status_code}"})
            elif rc.status_code == 200:
                gok, gd = compare_npy(rc.content, rg.content, check_values=True)
                if not gok:
                    gpu_diffs.append({"label": lb, "detail": gd})
        if rc.status_code != 200:
            rec(lb, path, "zarr_npy", False, f"HTTP {rc.status_code} {rc.text[:100]!r}")
        else:
            arr = try_npy(lb, path, rc.content)
            if arr is not None:
                low = np.finfo(arr.dtype).min if arr.dtype.kind == "f" else np.iinfo(arr.dtype).min
                mask = arr[-1] > low
                exp_mask = exp_arr != -9999.0
                both = mask & exp_mask
                mask_eq = float((mask == exp_mask).mean()) * 100.0
                max_err = float(np.max(np.abs(arr[0][both].astype(np.float64) - exp_arr[both]))) if both.any() else 0.0
                rec(lb, path, "zarr_npy", arr.shape == (2, 128, 128) and mask_eq == 100.0 and max_err < 1e-3, f"shape={arr.shape} mask_eq={mask_eq:.1f}% max_err={max_err:.5f}")

    # 5. Zarr v2 GET & POST /statistics
    stats_path = f"/zarr/statistics?url={q(fx['v2_geo'])}&variable=temp&sel={qe('time=0')}"
    if wants("W8 ZARR v2 3D GET /statistics sel=time=0"):
        rc = client.get(cw_base + stats_path)
        if rc.status_code != 200:
            rec("W8 ZARR v2 3D GET /statistics sel=time=0", stats_path, "zarr_stats", False, f"HTTP {rc.status_code}")
        else:
            st = next(iter(rc.json().values()))
            valid = fx["data_3d"][0][fx["data_3d"][0] != -9999.0]
            rec("W8 ZARR v2 3D GET /statistics sel=time=0", stats_path, "zarr_stats", abs(st["mean"] - float(valid.mean())) < 1e-2 and int(st["count"]) == valid.size, f"mean={st['mean']:.4f}/{valid.mean():.4f} cnt={st['count']}/{valid.size}")

    if wants("W8 ZARR v2 3D POST /statistics west half"):
        poly_west = {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[[-123.0, 37.0], [-122.5, 37.0], [-122.5, 38.0], [-123.0, 38.0], [-123.0, 37.0]]]}}
        rc = client.post(cw_base + stats_path, json=poly_west)
        if gpu_base is not None:
            rg = client.post(gpu_base + stats_path, json=poly_west)
            if rc.status_code != rg.status_code or (rc.status_code == 200 and first_json_diff(rc.json(), rg.json()) is not None):
                gpu_diffs.append({"label": "W8 ZARR v2 3D POST /statistics west half", "detail": "CPU vs GPU stats diff"})
        if rc.status_code != 200:
            rec("W8 ZARR v2 3D POST /statistics west half", stats_path, "zarr_stats", False, f"HTTP {rc.status_code}", method="POST")
        else:
            st = next(iter(rc.json()["properties"]["statistics"].values()))
            sub = fx["data_3d"][0][:, 0:64]
            valid = sub[sub != -9999.0]
            rec("W8 ZARR v2 3D POST /statistics west half", stats_path, "zarr_stats", abs(st["mean"] - float(valid.mean())) < 0.2 and int(st["count"]) == valid.size, f"mean={st['mean']:.4f}/{valid.mean():.4f} cnt={st['count']:.0f}/{valid.size}", method="POST")

    # 6. Zarr v3 ASCENDING latitude (south-up!) /point + /bbox
    if wants("W8 ZARR v3 south-up /point NE"):
        lat_ne, lon_ne = float(fx["lat_asc"][100]), float(fx["lon"][100])
        exp_ne = float(fx["v3_arr"][100, 100])
        path = f"/zarr/point/{lon_ne:.8f},{lat_ne:.8f}?url={q(fx['v3_southup'])}&variable=sst"
        rc = client.get(cw_base + path)
        if rc.status_code != 200:
            rec("W8 ZARR v3 south-up /point NE", path, "zarr_point", False, f"HTTP {rc.status_code}")
        else:
            got = rc.json()["values"][0]
            rec("W8 ZARR v3 south-up /point NE", path, "zarr_point", abs(got - exp_ne) < 1e-3, f"got={got:.4f} exp={exp_ne:.4f}")

    if wants("W8 ZARR v3 south-up /bbox 128x128.npy"):
        path = f"/zarr/bbox/-123,37,-122,38/128x128.npy?url={q(fx['v3_southup'])}&variable=sst"
        rc = client.get(cw_base + path)
        if gpu_base is not None:
            rg = client.get(gpu_base + path)
            if rc.status_code != rg.status_code:
                gpu_diffs.append({"label": "W8 ZARR v3 south-up /bbox 128x128.npy", "detail": f"status cpu={rc.status_code} gpu={rg.status_code}"})
            elif rc.status_code == 200:
                gok, gd = compare_npy(rc.content, rg.content, check_values=True)
                if not gok:
                    gpu_diffs.append({"label": "W8 ZARR v3 south-up /bbox 128x128.npy", "detail": gd})
        if rc.status_code != 200:
            rec("W8 ZARR v3 south-up /bbox 128x128.npy", path, "zarr_npy", False, f"HTTP {rc.status_code}")
        else:
            arr = try_npy("W8 ZARR v3 south-up /bbox 128x128.npy", path, rc.content)
            if arr is not None:
                low = np.finfo(arr.dtype).min if arr.dtype.kind == "f" else np.iinfo(arr.dtype).min
                mask = arr[-1] > low
                exp = fx["v3_arr"][::-1, :]
                exp_mask = np.isfinite(exp)
                both = mask & exp_mask
                mask_eq = float((mask == exp_mask).mean()) * 100.0
                max_err = float(np.max(np.abs(arr[0][both].astype(np.float64) - exp[both]))) if both.any() else 0.0
                rec("W8 ZARR v3 south-up /bbox 128x128.npy", path, "zarr_npy", arr.shape == (2, 128, 128) and mask_eq == 100.0 and max_err < 1e-3, f"shape={arr.shape} dtype={arr.dtype} mask_eq={mask_eq:.1f}% max_err={max_err:.5f}")

    # 7. Zarr v2 0..360° longitude /point
    if wants("W8 ZARR v2 0..360° lon /point (-122.5, 37.5)"):
        path = f"/zarr/point/{lon_p:.8f},{lat_p:.8f}?url={q(fx['v2_360'])}&variable=var360"
        rc = client.get(cw_base + path)
        if rc.status_code != 200:
            rec("W8 ZARR v2 0..360° lon /point (-122.5, 37.5)", path, "zarr_point", False, f"HTTP {rc.status_code}")
        else:
            got = rc.json()["values"][0]
            exp = float(fx["data_3d"][0, 64, 64])
            rec("W8 ZARR v2 0..360° lon /point (-122.5, 37.5)", path, "zarr_point", abs(got - exp) < 1e-3, f"got={got:.4f} exp={exp:.4f}")

    # 8. Zarr tile ETag/304 & HEAD
    tu = f"/zarr/tiles/WebMercatorQuad/12/657/1582.png?url={q(fx['v2_proj'])}&variable=elev&rescale=500,2000&colormap_name=viridis"
    if wants("W8 ZARR tile ETag + If-None-Match -> 304"):
        etag = client.get(cw_base + tu).headers.get("etag")
        r304 = client.get(cw_base + tu, headers={"If-None-Match": etag or ""})
        rec("W8 ZARR tile ETag + If-None-Match -> 304", tu, "zarr_http", bool(etag) and r304.status_code == 304 and len(r304.content) == 0, f"etag={etag} status={r304.status_code}")
    if wants("W8 ZARR tile HEAD -> 200 empty"):
        rh = client.head(cw_base + tu)
        rec("W8 ZARR tile HEAD -> 200 empty", tu, "zarr_http", rh.status_code == 200 and len(rh.content) == 0, f"status={rh.status_code} len={len(rh.content)}", method="HEAD")

    # 9. Regression: identical query string with .npy then .png must NOT collide in tile_cache
    if wants("W8 ZARR tile cache isolates .npy vs .png"):
        qs = f"url={q(fx['v2_proj'])}&variable=elev"
        p_npy = f"/zarr/tiles/WebMercatorQuad/12/657/1582.npy?{qs}"
        p_png = f"/zarr/tiles/WebMercatorQuad/12/657/1582.png?{qs}"
        r_npy = client.get(cw_base + p_npy)
        r_png = client.get(cw_base + p_png)
        ok_cache = (
            r_npy.status_code == 200
            and r_png.status_code == 200
            and r_npy.content.startswith(b"\x93NUMPY")
            and r_png.content.startswith(b"\x89PNG\r\n\x1a\n")
            and r_npy.headers.get("etag") != r_png.headers.get("etag")
        )
        rec("W8 ZARR tile cache isolates .npy vs .png", p_png, "zarr_http", ok_cache, f"npy_magic={r_npy.content[:6]!r} png_magic={r_png.content[:4]!r}")

    return ran


def run_wave14_fault_checks(
    client: httpx.Client,
    cw_base: str,
    tmp_dir: Path,
    s: dict[str, Path],
    only_substr: str | None,
    failures: list[dict],
) -> int:
    """Wave 14 operational fault-injection, secret redaction, cache invalidation & antimeridian bbox checks."""
    def wants(label: str) -> bool:
        return not only_substr or only_substr.lower() in label.lower()

    all_labels = [
        "W14 FAULT HTTP 403->403 + secret redacted",
        "W14 FAULT HTTP 500->502 + secret redacted",
        "W14 FAULT MOSAIC partial fail + redacted X-Failed-Assets",
        "W14 MUTABLE local COG overwrite invalidates cache",
        "W14 ZARR CFTime ISO-8601 sel=time=2024-06-02",
        "W14 MUTABLE local Zarr ETag invalidation (200 not 304)",
        "W14 ANTIMERIDIAN west>east /cog/bbox == unwrapped",
    ]
    if not any(wants(lb) for lb in all_labels):
        return 0

    ran = 0

    def rec(label: str, path: str, kind: str, ok: bool, detail: str):
        nonlocal ran
        ran += 1
        if not ok:
            failures.append({"label": label, "method": "GET", "path": path, "kind": kind, "detail": detail})
        print(f"{'PASS' if ok else 'FAIL'} | {label:54s} | {detail}")

    d = tmp_dir / "wave14_ops"
    d.mkdir(parents=True, exist_ok=True)
    tf = from_bounds(-122.5, 37.5, -122.0, 38.0, 128, 128)

    # 1. Local HTTP fault-injection server (403, 500, and 206 Range OK)
    if any(wants(lb) for lb in all_labels[:3]):
        ok_bytes = s["w14_const"].read_bytes()
        srv_port = pick_free_port()

        class FaultHandler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_HEAD(self):
                if self.path.startswith("/fail500"):
                    self.send_response(500)
                    self.end_headers()
                    return
                if self.path.startswith("/forbidden403"):
                    self.send_response(403)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Length", str(len(ok_bytes)))
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("ETag", '"w14-ok"')
                self.end_headers()

            def do_GET(self):
                if self.path.startswith("/fail500"):
                    self.send_response(500)
                    self.end_headers()
                    self.wfile.write(b"upstream boom")
                    return
                if self.path.startswith("/forbidden403"):
                    self.send_response(403)
                    self.end_headers()
                    return
                rng = self.headers.get("Range")
                if rng and rng.startswith("bytes="):
                    s0, e0 = rng[6:].split("-", 1)
                    start = int(s0)
                    end = min(int(e0) if e0 else len(ok_bytes) - 1, len(ok_bytes) - 1)
                    chunk = ok_bytes[start:end + 1]
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{end}/{len(ok_bytes)}")
                    self.send_header("Content-Length", str(len(chunk)))
                    self.send_header("ETag", '"w14-ok"')
                    self.end_headers()
                    self.wfile.write(chunk)
                else:
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(ok_bytes)))
                    self.send_header("ETag", '"w14-ok"')
                    self.end_headers()
                    self.wfile.write(ok_bytes)

        httpd = http.server.ThreadingHTTPServer(("127.0.0.1", srv_port), FaultHandler)
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        try:
            remote_ok = f"http://127.0.0.1:{srv_port}/ok.tif"
            remote_500 = f"http://127.0.0.1:{srv_port}/fail500.tif?X-Amz-Signature=SUPERSECRET123"
            remote_403 = f"http://127.0.0.1:{srv_port}/forbidden403.tif?token=TOPSECRET999"

            for lb, u, exp_code, secret in [
                ("W14 FAULT HTTP 403->403 + secret redacted", remote_403, 403, "TOPSECRET999"),
                ("W14 FAULT HTTP 500->502 + secret redacted", remote_500, 502, "SUPERSECRET123"),
            ]:
                if wants(lb):
                    path = f"/cog/info?url={qe(u)}"
                    r = client.get(cw_base + path)
                    rec(lb, path, "fault_http", r.status_code == exp_code and secret not in r.text, f"status={r.status_code} secret_redacted={secret not in r.text}")

            lb_m = "W14 FAULT MOSAIC partial fail + redacted X-Failed-Assets"
            if wants(lb_m):
                qk = mercantile.quadkey(164, 396, 10)
                mj = d / "mosaic_fault.json"
                mj.write_text(json.dumps({"mosaicjson": "0.0.2", "minzoom": 10, "maxzoom": 10, "quadkey_zoom": 10, "bounds": [-122.5, 37.5, -122.0, 38.0], "tiles": {qk: [remote_500, remote_ok]}}))
                path = f"/mosaicjson/tiles/WebMercatorQuad/10/164/396.npy?url={q(mj)}"
                rm = client.get(cw_base + path)
                xfa = rm.headers.get("x-failed-assets", "")
                rec(lb_m, path, "fault_mosaic", rm.status_code == 200 and bool(xfa) and "SUPERSECRET123" not in xfa, f"status={rm.status_code} xfa={xfa!r}")
        finally:
            httpd.shutdown()
            t.join(timeout=5)
            httpd.server_close()

    # 2. Mutable local COG overwrite invalidates cache immediately
    lb_mut = "W14 MUTABLE local COG overwrite invalidates cache"
    if wants(lb_mut):
        p_mut = d / "mut_local.tif"
        with rasterio.open(p_mut, "w", driver="GTiff", width=128, height=128, count=1, dtype="float32", crs="EPSG:4326", transform=tf, tiled=True, blockxsize=64, blockysize=64, compress="deflate") as ds:
            ds.write(np.full((1, 128, 128), 111.0, dtype=np.float32))
        path = f"/cog/point/-122.25,37.75?url={q(p_mut)}"
        r1 = client.get(cw_base + path)
        v1 = r1.json().get("values", [None])[0] if r1.status_code == 200 else None
        with rasterio.open(p_mut, "w", driver="GTiff", width=128, height=128, count=1, dtype="float32", crs="EPSG:4326", transform=tf, tiled=True, blockxsize=64, blockysize=64, compress="deflate") as ds:
            ds.write(np.full((1, 128, 128), 222.0, dtype=np.float32))
        st = os.stat(p_mut)
        os.utime(p_mut, (st.st_atime, st.st_mtime + 2))
        r2 = client.get(cw_base + path)
        v2 = r2.json().get("values", [None])[0] if r2.status_code == 200 else None
        rec(lb_mut, path, "mutable_cog", r1.status_code == 200 and r2.status_code == 200 and v1 == 111.0 and v2 == 222.0, f"v1={v1} -> v2={v2}")

    # 3. Zarr CFTime ISO-8601 + Mutable local Zarr ETag invalidation
    if wants("W14 ZARR CFTime ISO-8601 sel=time=2024-06-02") or wants("W14 MUTABLE local Zarr ETag invalidation (200 not 304)"):
        s_zmut = d / "zarr_mut_cftime.zarr"
        s_zmut.mkdir(exist_ok=True)
        (s_zmut / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
        (s_zmut / ".zattrs").write_text(json.dumps({"Conventions": "CF-1.8"}))
        lat_z = np.linspace(38.0 - 0.5 / 64, 37.5 + 0.5 / 64, 64, dtype=np.float64)
        lon_z = np.linspace(-122.5 + 0.5 / 64, -122.0 - 0.5 / 64, 64, dtype=np.float64)
        _write_v2_zarr_array(s_zmut, "time", np.array([0, 1], dtype=np.int64), (2,), {"_ARRAY_DIMENSIONS": ["time"], "units": "days since 2024-06-01", "calendar": "standard"})
        _write_v2_zarr_array(s_zmut, "lat", lat_z, (64,), {"_ARRAY_DIMENSIONS": ["lat"], "standard_name": "latitude"})
        _write_v2_zarr_array(s_zmut, "lon", lon_z, (64,), {"_ARRAY_DIMENSIONS": ["lon"], "standard_name": "longitude"})
        _write_v2_zarr_array(s_zmut, "v", np.stack([np.full((64, 64), 10.0, dtype=np.float32), np.full((64, 64), 77.0, dtype=np.float32)], axis=0), (1, 64, 64), {"_ARRAY_DIMENSIONS": ["time", "lat", "lon"]}, compressor={"id": "zlib", "level": 1})

        if wants("W14 ZARR CFTime ISO-8601 sel=time=2024-06-02"):
            path = f"/zarr/point/-122.25,37.75?url={q(s_zmut)}&variable=v&sel={qe('time=2024-06-02')}"
            r = client.get(cw_base + path)
            ok = r.status_code == 200 and r.json()["values"][0] == 77.0
            rec("W14 ZARR CFTime ISO-8601 sel=time=2024-06-02", path, "zarr_cftime", ok, f"status={r.status_code} val={r.json().get('values') if r.status_code == 200 else r.text[:60]}")

        if wants("W14 MUTABLE local Zarr ETag invalidation (200 not 304)"):
            zu = f"/zarr/tiles/WebMercatorQuad/10/164/396.npy?url={q(s_zmut)}&variable=v&sel={qe('time=2024-06-01')}"
            zr1 = client.get(cw_base + zu)
            etag1 = zr1.headers.get("etag")
            v1 = float(np.load(io.BytesIO(zr1.content))[0, 128, 128]) if zr1.status_code == 200 else None
            (s_zmut / "v" / "0.0.0").write_bytes(zlib.compress(np.full((64, 64), 99.0, dtype=np.float32).tobytes(), 1))
            for mf in (s_zmut / "v" / "0.0.0", s_zmut / "v" / ".zarray", s_zmut / ".zgroup"):
                st = os.stat(mf)
                os.utime(mf, (st.st_atime, st.st_mtime + 2))
            zr2 = client.get(cw_base + zu, headers={"If-None-Match": etag1 or ""})
            etag2 = zr2.headers.get("etag")
            v2 = float(np.load(io.BytesIO(zr2.content))[0, 128, 128]) if zr2.status_code == 200 else None
            ok_z = zr1.status_code == 200 and bool(etag1) and zr2.status_code == 200 and etag2 != etag1 and v1 == 10.0 and v2 == 99.0
            rec("W14 MUTABLE local Zarr ETag invalidation (200 not 304)", zu, "mutable_zarr", ok_z, f"status={zr2.status_code} v1={v1} -> v2={v2} etag_changed={etag2 != etag1}")

    # 4. Antimeridian west>east /cog/bbox equivalence
    lb_am = "W14 ANTIMERIDIAN west>east /cog/bbox == unwrapped"
    if wants(lb_am):
        p_wrap = f"/cog/bbox/175,-10,-175,10/64x64.npy?url={q(s['w14_antimeridian'])}"
        p_unwrap = f"/cog/bbox/175,-10,185,10/64x64.npy?url={q(s['w14_antimeridian'])}"
        rw, ru = client.get(cw_base + p_wrap), client.get(cw_base + p_unwrap)
        ok_am, det_am = compare_npy(rw.content, ru.content, check_values=True) if (rw.status_code == 200 and ru.status_code == 200) else (False, f"HTTP wrap={rw.status_code} unwrap={ru.status_code}")
        rec(lb_am, p_wrap, "antimeridian_bbox", ok_am, det_am)

    return ran


def _crc32c(data: bytes) -> int:
    """Pure-Python Castagnoli CRC32C (polynomial 0x82F63B78) for Zarr v3 sharding_indexed."""
    crc = 0xFFFFFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0x82F63B78
            else:
                crc >>= 1
    return crc ^ 0xFFFFFFFF


def _build_zarr_v3_sharded(store: Path) -> tuple[Path, np.ndarray]:
    """Build a Zarr v3 store with `sharding_indexed` (1 shard of 128x128 = 2x2 inner chunks of 64x64)."""
    store.mkdir(parents=True, exist_ok=True)
    (store / "zarr.json").write_text(
        json.dumps({"zarr_format": 3, "node_type": "group", "attributes": {"Conventions": "CF-1.8"}})
    )

    lat = np.linspace(38.0 - 0.5 / 128, 37.0 + 0.5 / 128, 128, dtype=np.float64)
    lon = np.linspace(-123.0 + 0.5 / 128, -122.0 - 0.5 / 128, 128, dtype=np.float64)
    for name, arr, std in (("lat", lat, "latitude"), ("lon", lon, "longitude")):
        ad = store / name
        ad.mkdir()
        (ad / "zarr.json").write_text(json.dumps({
            "zarr_format": 3,
            "node_type": "array",
            "shape": [128],
            "data_type": "float64",
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [128]}},
            "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
            "fill_value": "NaN",
            "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
            "dimension_names": [name],
            "attributes": {"standard_name": std},
        }))
        cp = ad / "c" / "0"
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_bytes(np.ascontiguousarray(arr).tobytes())

    yy, xx = np.mgrid[0:128, 0:128].astype(np.float32)
    data = 50.0 + xx * 1.5 + yy * 0.75
    data[0:32, 0:32] = -9999.0

    ad = store / "sharded_var"
    ad.mkdir()
    (ad / "zarr.json").write_text(json.dumps({
        "zarr_format": 3,
        "node_type": "array",
        "shape": [128, 128],
        "data_type": "float32",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [128, 128]}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "fill_value": -9999.0,
        "codecs": [{
            "name": "sharding_indexed",
            "configuration": {
                "chunk_shape": [64, 64],
                "codecs": [
                    {"name": "bytes", "configuration": {"endian": "little"}},
                    {"name": "gzip", "configuration": {"level": 1}},
                ],
                "index_codecs": [
                    {"name": "bytes", "configuration": {"endian": "little"}},
                    {"name": "crc32c"},
                ],
                "index_location": "end",
            },
        }],
        "dimension_names": ["lat", "lon"],
        "attributes": {},
    }))

    buf = bytearray()
    index_entries = []
    for cy in range(2):
        for cx in range(2):
            sub = np.ascontiguousarray(data[cy * 64:(cy + 1) * 64, cx * 64:(cx + 1) * 64])
            comp = gzip.compress(sub.tobytes(), compresslevel=1)
            off = len(buf)
            buf.extend(comp)
            index_entries.append((off, len(comp)))
    idx_bytes = b"".join(struct.pack("<QQ", o, n) for o, n in index_entries)
    chk = struct.pack("<I", _crc32c(idx_bytes))
    buf.extend(idx_bytes)
    buf.extend(chk)

    sp = ad / "c" / "0" / "0"
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_bytes(bytes(buf))
    return store, data


def run_wave15_resilience_checks(
    client: httpx.Client,
    cw_base: str,
    gpu_base: str | None,
    tmp_dir: Path,
    only_filter: str,
    failures: list[dict],
    gpu_diffs: list[dict],
) -> int:
    """Wave 15: HTTP If-Match 412 race recovery, range-misbehavior fault injection, hostile/corrupt TIFF+Zarr fuzzing, and Zarr v3 sharding_indexed."""
    w15_dir = tmp_dir / "w15"
    w15_dir.mkdir(parents=True, exist_ok=True)
    tf = from_bounds(-122.5, 37.5, -122.0, 38.0, 128, 128)

    for ver, val in (("v1", 100.0), ("v2", 250.0)):
        with rasterio.open(
            w15_dir / f"{ver}.tif", "w", driver="GTiff", width=128, height=128, count=1,
            dtype="float32", crs="EPSG:4326", transform=tf,
            tiled=True, blockxsize=64, blockysize=64, compress="deflate",
        ) as ds:
            ds.write(np.full((1, 128, 128), val, dtype=np.float32))
    v1_bytes = (w15_dir / "v1.tif").read_bytes()
    v2_bytes = (w15_dir / "v2.tif").read_bytes()

    # > 64 KB uncompressed tiled COG so opening + tile fetch genuinely requires a second HTTP Range GET
    tf_big = from_bounds(-122.5, 37.5, -122.0, 38.0, 512, 512)
    with rasterio.open(
        w15_dir / "big_norange.tif", "w", driver="GTiff", width=512, height=512, count=1,
        dtype="float32", crs="EPSG:4326", transform=tf_big,
        tiled=True, blockxsize=64, blockysize=64,
    ) as ds:
        ds.write(np.full((1, 512, 512), 42.0, dtype=np.float32))
    big_bytes = (w15_dir / "big_norange.tif").read_bytes()

    # Hostile / corrupt TIFF fixtures
    p_trunc = w15_dir / "truncated.tif"
    p_trunc.write_bytes(v1_bytes[:24])
    p_garbage = w15_dir / "garbage.tif"
    p_garbage.write_bytes(b"NOT_A_TIFF_AT_ALL_0123456789" * 20)
    p_corrupt_tile = w15_dir / "corrupt_tile.tif"
    ba = bytearray(v1_bytes)
    for i in range(max(0, len(ba) - 128), len(ba)):
        ba[i] ^= 0xFF
    p_corrupt_tile.write_bytes(bytes(ba))

    # Zarr v3 sharded store + Zarr error-table fixtures
    z_sharded, z_sharded_gt = _build_zarr_v3_sharded(w15_dir / "v3_sharded.zarr")

    z_err = w15_dir / "zarr_errors.zarr"
    z_err.mkdir()
    (z_err / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
    lat64 = np.linspace(38.0 - 0.5 / 64, 37.5 + 0.5 / 64, 64, dtype=np.float64)
    lon64 = np.linspace(-122.5 + 0.5 / 64, -122.0 - 0.5 / 64, 64, dtype=np.float64)
    _write_v2_zarr_array(z_err, "time", np.array([0, 1], dtype=np.int64), (2,), {"_ARRAY_DIMENSIONS": ["time"]})
    _write_v2_zarr_array(z_err, "lat", lat64, (64,), {"_ARRAY_DIMENSIONS": ["lat"], "standard_name": "latitude"})
    _write_v2_zarr_array(z_err, "lon", lon64, (64,), {"_ARRAY_DIMENSIONS": ["lon"], "standard_name": "longitude"})
    _write_v2_zarr_array(
        z_err, "a", np.ones((2, 64, 64), dtype=np.float32), (1, 64, 64),
        {"_ARRAY_DIMENSIONS": ["time", "lat", "lon"]}, compressor={"id": "gzip", "level": 1},
    )
    _write_v2_zarr_array(
        z_err, "b", np.ones((2, 64, 64), dtype=np.float32), (1, 64, 64),
        {"_ARRAY_DIMENSIONS": ["time", "lat", "lon"]}, compressor={"id": "gzip", "level": 1},
    )
    (z_err / "a" / "0.0.0").write_bytes(b"\x1f\x8bCORRUPT_GZIP_GARBAGE")
    (z_err / "mystery").mkdir()
    (z_err / "mystery" / ".zarray").write_text(json.dumps({
        "chunks": [64, 64], "compressor": {"id": "alien_lz99"}, "dtype": "<f4",
        "fill_value": None, "filters": None, "order": "C", "shape": [64, 64], "zarr_format": 2,
    }))
    (z_err / "mystery" / ".zattrs").write_text(json.dumps({"_ARRAY_DIMENSIONS": ["lat", "lon"]}))
    (z_err / "mystery" / "0.0").write_bytes(b"xxxx")

    state = {"bytes": v1_bytes, "etag": '"v1"', "saw_412": False}
    srv_port = pick_free_port()

    class RaceFaultHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_HEAD(self):
            if self.path.startswith("/redir"):
                self.send_response(302)
                self.send_header("Location", f"http://127.0.0.1:{srv_port}/target.tif")
                self.end_headers()
                return
            if self.path.startswith("/loop"):
                self.send_response(302)
                self.send_header("Location", f"http://127.0.0.1:{srv_port}/loop.tif")
                self.end_headers()
                return
            if self.path.startswith("/target"):
                self.send_response(200)
                self.send_header("Content-Length", str(len(v1_bytes)))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                return
            if self.path.startswith("/norange"):
                self.send_response(200)
                self.send_header("Content-Length", str(len(big_bytes)))
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Length", str(len(state["bytes"])))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("ETag", state["etag"])
            self.end_headers()

        def do_GET(self):
            if self.path.startswith("/redir"):
                self.send_response(302)
                self.send_header("Location", f"http://127.0.0.1:{srv_port}/target.tif")
                self.end_headers()
                return
            if self.path.startswith("/loop"):
                self.send_response(302)
                self.send_header("Location", f"http://127.0.0.1:{srv_port}/loop.tif")
                self.end_headers()
                return
            if self.path.startswith("/target"):
                rng = self.headers.get("Range")
                if rng and rng.startswith("bytes="):
                    s0, e0 = rng[6:].split("-", 1)
                    start = int(s0)
                    end = min(int(e0) if e0 else len(v1_bytes) - 1, len(v1_bytes) - 1)
                    chunk = v1_bytes[start:end + 1]
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{end}/{len(v1_bytes)}")
                    self.send_header("Content-Length", str(len(chunk)))
                    self.end_headers()
                    self.wfile.write(chunk)
                else:
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(v1_bytes)))
                    self.send_header("Accept-Ranges", "bytes")
                    self.end_headers()
                    self.wfile.write(v1_bytes)
                return
            if self.path.startswith("/norange"):
                self.send_response(200)
                self.send_header("Content-Length", str(len(big_bytes)))
                self.end_headers()
                try:
                    self.wfile.write(big_bytes)
                except (BrokenPipeError, ConnectionResetError):
                    pass
                return
            if self.path.startswith("/truncrange"):
                rng = self.headers.get("Range")
                if rng and rng.startswith("bytes="):
                    chunk = v1_bytes[:8]
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes 0-7/{len(v1_bytes)}")
                    self.send_header("Content-Length", "8")
                    self.end_headers()
                    self.wfile.write(chunk)
                    return
            ifm = self.headers.get("If-Match")
            if ifm and ifm != state["etag"]:
                state["saw_412"] = True
                self.send_response(412)
                self.send_header("ETag", state["etag"])
                self.end_headers()
                return
            rng = self.headers.get("Range")
            data = state["bytes"]
            if rng and rng.startswith("bytes="):
                s0, e0 = rng[6:].split("-", 1)
                start = int(s0)
                end = min(int(e0) if e0 else len(data) - 1, len(data) - 1)
                chunk = data[start:end + 1]
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
                self.send_header("Content-Length", str(len(chunk)))
                self.send_header("ETag", state["etag"])
                self.end_headers()
                self.wfile.write(chunk)
            else:
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("ETag", state["etag"])
                self.end_headers()
                self.wfile.write(data)

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", srv_port), RaceFaultHandler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()

    ran = 0

    def rec(lb: str, ok: bool, detail: str, path: str = ""):
        nonlocal ran
        ran += 1
        if not ok:
            failures.append({"label": lb, "method": "GET", "path": path, "kind": "w15", "detail": detail})
        print(f"{'PASS' if ok else 'FAIL'} | {lb:54s} | {detail}")

    def wanted(lb: str) -> bool:
        return not only_filter or only_filter.lower() in lb.lower()

    try:
        # 1. HTTP If-Match 412 mid-read race recovery -> 200 fresh v2
        lb = "W15 HTTP If-Match 412 race -> 200 fresh v2"
        if wanted(lb):
            race_url = f"http://127.0.0.1:{srv_port}/race.tif"
            r_info = client.get(f"{cw_base}/cog/info?url={qe(race_url)}")
            state["bytes"] = v2_bytes
            state["etag"] = '"v2"'
            tile_path = f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={qe(race_url)}"
            r_tile = client.get(cw_base + tile_path)
            if r_tile.status_code == 200:
                arr = np.load(io.BytesIO(r_tile.content))
                v = float(arr[0, 128, 128])
                rec(lb, r_info.status_code == 200 and state["saw_412"] and v == 250.0, f"saw_412={state['saw_412']} val={v} (exp 250.0)", tile_path)
            else:
                rec(lb, False, f"HTTP {r_tile.status_code} saw_412={state['saw_412']} {r_tile.text[:80]!r}", tile_path)

        # 2. HTTP Range misbehavior: no-Range 200 on >64KB COG -> 502; truncated Range -> 502
        lb = "W15 HTTP no-Range 200 (>64KB COG) -> 502"
        if wanted(lb):
            p_nr = f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={qe(f'http://127.0.0.1:{srv_port}/norange.tif')}"
            r_nr = client.get(cw_base + p_nr)
            rec(lb, r_nr.status_code == 502, f"status={r_nr.status_code} {r_nr.text[:80]!r}", p_nr)

        lb = "W15 HTTP truncated Range -> 502"
        if wanted(lb):
            p_tr = f"/cog/info?url={qe(f'http://127.0.0.1:{srv_port}/truncrange.tif')}"
            r_tr = client.get(cw_base + p_tr)
            rec(lb, r_tr.status_code in (400, 502), f"status={r_tr.status_code} {r_tr.text[:80]!r}", p_tr)

        # 3. Hostile / corrupt TIFF & Zarr fuzzing (zero 500s!)
        for lb, path, allowed in [
            ("W15 FUZZ truncated TIFF -> 400", f"/cog/info?url={q(p_trunc)}", (400,)),
            ("W15 FUZZ garbage bytes -> 400", f"/cog/info?url={q(p_garbage)}", (400,)),
            ("W15 FUZZ corrupt compressed tile -> 502 (not 500)", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(p_corrupt_tile)}", (502,)),
            ("W15 FUZZ ftp:// scheme -> 400", f"/cog/info?url={qe('ftp://example.com/a.tif')}", (400,)),
            ("W15 ZARR ambiguous variable -> 400", f"/zarr/info?url={q(z_err)}", (400,)),
            ("W15 ZARR out-of-range sel=time=999 -> 400", f"/zarr/point/-122.25,37.75?url={q(z_err)}&variable=b&sel={qe('time=999')}", (400,)),
            ("W15 ZARR unknown codec -> 501", f"/zarr/point/-122.25,37.75?url={q(z_err)}&variable=mystery", (501,)),
            ("W15 ZARR corrupt gzip chunk -> 502", f"/zarr/point/-122.25,37.75?url={q(z_err)}&variable=a&sel={qe('time=0')}", (502,)),
        ]:
            if wanted(lb):
                r = client.get(cw_base + path)
                rec(lb, r.status_code in allowed, f"status={r.status_code} {r.text[:80]!r}", path)

        # 4. Zarr v3 sharding_indexed exact pixel parity against NumPy ground truth (+ CPU vs GPU)
        lb = "W15 ZARR v3 sharding_indexed /bbox .npy"
        if wanted(lb):
            sh_path = f"/zarr/bbox/-123,37,-122,38/128x128.npy?url={q(z_sharded)}&variable=sharded_var"
            r_sh = client.get(cw_base + sh_path)
            if r_sh.status_code != 200:
                rec(lb, False, f"HTTP {r_sh.status_code} {r_sh.text[:100]!r}", sh_path)
            else:
                arr = np.load(io.BytesIO(r_sh.content))
                mask = arr[1] > np.finfo(np.float32).min
                exp_mask = z_sharded_gt != -9999.0
                both = mask & exp_mask
                max_err = float(np.max(np.abs(arr[0][both] - z_sharded_gt[both]))) if both.any() else 999.0
                rec(
                    lb,
                    arr.shape == (2, 128, 128) and bool((mask == exp_mask).all()) and max_err < 1e-3,
                    f"shape={arr.shape} mask_eq={float((mask == exp_mask).mean()) * 100:.1f}% max_err={max_err:.5f}",
                    sh_path,
                )
                if gpu_base is not None:
                    r_g = client.get(gpu_base + sh_path)
                    if r_g.status_code != 200:
                        gpu_diffs.append({"label": lb, "detail": f"status cpu=200 gpu={r_g.status_code}"})
                    else:
                        gok, gd = compare_npy(r_sh.content, r_g.content, check_values=True, masked=True)
                        if not gok:
                            gpu_diffs.append({"label": lb, "detail": gd})

        # 5. Wave 16 HTTP 302 redirect, redirect loop -> 502, and 16-way concurrent singleflight burst
        lb = "W16 HTTP 302 redirect -> 200"
        if wanted(lb):
            p_rd = f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={qe(f'http://127.0.0.1:{srv_port}/redir.tif')}"
            r_rd = client.get(cw_base + p_rd)
            if r_rd.status_code == 200:
                arr = np.load(io.BytesIO(r_rd.content))
                v = float(arr[0, 128, 128])
                rec(lb, v == 100.0, f"val={v} (exp 100.0)", p_rd)
            else:
                rec(lb, False, f"HTTP {r_rd.status_code} {r_rd.text[:80]!r}", p_rd)

        lb = "W16 HTTP redirect loop -> 502"
        if wanted(lb):
            p_lp = f"/cog/info?url={qe(f'http://127.0.0.1:{srv_port}/loop.tif')}"
            r_lp = client.get(cw_base + p_lp)
            rec(lb, r_lp.status_code == 502, f"status={r_lp.status_code} {r_lp.text[:80]!r}", p_lp)

        lb = "W16 CONCURRENT 16x singleflight burst"
        if wanted(lb):
            p_burst = f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(w15_dir / 'v1.tif')}"
            def _fetch_one(_):
                r = httpx.get(cw_base + p_burst, timeout=30.0)
                return r.status_code, r.content
            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
                res = list(ex.map(_fetch_one, range(16)))
            all_200 = all(st == 200 for st, _ in res)
            all_eq = len({b for _, b in res}) == 1
            rec(lb, all_200 and all_eq and len(res[0][1]) > 0, f"all_200={all_200} identical_bytes={all_eq} len={len(res[0][1])}", p_burst)

        # 6. Server health check after all hostile fuzzing
        lb = "W15 SERVER healthy after hostile fuzzing -> 200"
        if wanted(lb):
            rh = client.get(f"{cw_base}/healthz")
            rec(lb, rh.status_code == 200, f"status={rh.status_code}", "/healthz")
    finally:
        httpd.shutdown()
        t.join(timeout=5)
        httpd.server_close()

    return ran


def run_wave17_zarr_deep_checks(
    client: httpx.Client,
    cw_base: str,
    gpu_base: str | None,
    tmp_dir: Path,
    only_filter: str,
    failures: list[dict],
    gpu_diffs: list[dict],
) -> int:
    """Wave 17: Zarr v2 Fortran-order, Zarr v3 transpose codec, CF scale_factor/add_offset, POST /zarr/feature, cross-CRS multiscales overview selection, and styled tile CPU-vs-GPU parity."""
    w17_dir = tmp_dir / "w17"
    w17_dir.mkdir(parents=True, exist_ok=True)
    lat = np.linspace(38.0 - 0.5 / 64, 37.0 + 0.5 / 64, 64, dtype=np.float64)
    lon = np.linspace(-123.0 + 0.5 / 64, -122.0 - 0.5 / 64, 64, dtype=np.float64)
    yy, xx = np.mgrid[0:64, 0:64]
    grad = 10.0 + xx.astype(np.float32) * 10.0 + yy.astype(np.float32) * 0.25

    # 1. Zarr v2 Fortran-order ("order": "F")
    z_f = w17_dir / "v2_fortran.zarr"
    z_f.mkdir()
    (z_f / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
    _write_v2_zarr_array(z_f, "lat", lat, (64,), {"_ARRAY_DIMENSIONS": ["lat"], "standard_name": "latitude"})
    _write_v2_zarr_array(z_f, "lon", lon, (64,), {"_ARRAY_DIMENSIONS": ["lon"], "standard_name": "longitude"})
    ad = z_f / "f_var"
    ad.mkdir()
    (ad / ".zarray").write_text(json.dumps({
        "chunks": [64, 64], "compressor": None, "dtype": "<f4",
        "fill_value": -9999.0, "filters": None, "order": "F", "shape": [64, 64], "zarr_format": 2,
    }))
    (ad / ".zattrs").write_text(json.dumps({"_ARRAY_DIMENSIONS": ["lat", "lon"]}))
    (ad / "0.0").write_bytes(np.asfortranarray(grad).tobytes(order="F"))

    # 2. Zarr v3 transpose codec (order: [1, 0])
    z_v3t = w17_dir / "v3_transpose.zarr"
    z_v3t.mkdir()
    (z_v3t / "zarr.json").write_text(json.dumps({"zarr_format": 3, "node_type": "group", "attributes": {"Conventions": "CF-1.8"}}))
    for nm, arr, std in (("lat", lat, "latitude"), ("lon", lon, "longitude")):
        vd = z_v3t / nm
        vd.mkdir()
        (vd / "zarr.json").write_text(json.dumps({
            "zarr_format": 3, "node_type": "array", "shape": [64], "data_type": "float64",
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [64]}},
            "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
            "fill_value": "NaN", "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
            "dimension_names": [nm], "attributes": {"standard_name": std},
        }))
        (vd / "c").mkdir()
        (vd / "c" / "0").write_bytes(np.ascontiguousarray(arr).tobytes())
    vd = z_v3t / "t_var"
    vd.mkdir()
    (vd / "zarr.json").write_text(json.dumps({
        "zarr_format": 3, "node_type": "array", "shape": [64, 64], "data_type": "float32",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [64, 64]}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "fill_value": -9999.0,
        "codecs": [
            {"name": "transpose", "configuration": {"order": [1, 0]}},
            {"name": "bytes", "configuration": {"endian": "little"}},
        ],
        "dimension_names": ["lat", "lon"], "attributes": {},
    }))
    (vd / "c" / "0").mkdir(parents=True)
    (vd / "c" / "0" / "0").write_bytes(np.ascontiguousarray(grad.T).tobytes())

    # 3. Zarr v2 CF scale_factor + add_offset + int16 _FillValue=-32768
    z_cf = w17_dir / "v2_cf_scale.zarr"
    z_cf.mkdir()
    (z_cf / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
    _write_v2_zarr_array(z_cf, "lat", lat, (64,), {"_ARRAY_DIMENSIONS": ["lat"], "standard_name": "latitude"})
    _write_v2_zarr_array(z_cf, "lon", lon, (64,), {"_ARRAY_DIMENSIONS": ["lon"], "standard_name": "longitude"})
    i16 = (100 + xx * 5 + yy * 2).astype(np.int16)
    i16[0:16, 0:16] = -32768
    _write_v2_zarr_array(
        z_cf, "temp", i16, (64, 64),
        {"_ARRAY_DIMENSIONS": ["lat", "lon"], "scale_factor": 0.01, "add_offset": 273.15, "_FillValue": -32768},
        fill_value=-32768,
    )

    # 4. Zarr v2 multiscales pyramid (EPSG:4326 queried in WebMercatorQuad)
    z_ms = w17_dir / "v2_multiscales.zarr"
    z_ms.mkdir()
    (z_ms / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
    (z_ms / ".zattrs").write_text(json.dumps({
        "multiscales": [{"datasets": [{"path": "0"}, {"path": "1"}], "name": "pyramid", "version": "0.4"}]
    }))
    for lvl, n, val in (("0", 128, 100.0), ("1", 32, 200.0)):
        gd = z_ms / lvl
        gd.mkdir()
        (gd / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
        la = np.linspace(38.0 - 0.5 / n, 37.0 + 0.5 / n, n, dtype=np.float64)
        lo = np.linspace(-123.0 + 0.5 / n, -122.0 - 0.5 / n, n, dtype=np.float64)
        _write_v2_zarr_array(gd, "lat", la, (n,), {"_ARRAY_DIMENSIONS": ["lat"], "standard_name": "latitude"})
        _write_v2_zarr_array(gd, "lon", lo, (n,), {"_ARRAY_DIMENSIONS": ["lon"], "standard_name": "longitude"})
        _write_v2_zarr_array(gd, "v", np.full((n, n), val, dtype=np.float32), (n, n), {"_ARRAY_DIMENSIONS": ["lat", "lon"]})

    ran = 0

    def rec(lb: str, ok: bool, detail: str, path: str = "", method: str = "GET"):
        nonlocal ran
        ran += 1
        if not ok:
            failures.append({"label": lb, "method": method, "path": path, "kind": "w17", "detail": detail})
        print(f"{'PASS' if ok else 'FAIL'} | {lb:54s} | {detail}")

    def wanted(lb: str) -> bool:
        return not only_filter or only_filter.lower() in lb.lower()

    # 1. Zarr v2 order=F
    lb = "W17 ZARR v2 order=F /bbox .npy"
    if wanted(lb):
        p = f"/zarr/bbox/-123,37,-122,38/64x64.npy?url={q(z_f)}&variable=f_var"
        r = client.get(cw_base + p)
        if r.status_code == 200:
            arr = np.load(io.BytesIO(r.content))
            err = float(np.max(np.abs(arr[0] - grad)))
            rec(lb, arr.shape == (2, 64, 64) and err < 1e-3, f"err={err:.5f}", p)
        else:
            rec(lb, False, f"HTTP {r.status_code} {r.text[:80]!r}", p)

    # 2. Zarr v3 transpose codec [1, 0]
    lb = "W17 ZARR v3 transpose [1,0] /bbox .npy"
    if wanted(lb):
        p = f"/zarr/bbox/-123,37,-122,38/64x64.npy?url={q(z_v3t)}&variable=t_var"
        r = client.get(cw_base + p)
        if r.status_code == 200:
            arr = np.load(io.BytesIO(r.content))
            err = float(np.max(np.abs(arr[0] - grad)))
            rec(lb, arr.shape == (2, 64, 64) and err < 1e-3, f"err={err:.5f}", p)
        else:
            rec(lb, False, f"HTTP {r.status_code} {r.text[:80]!r}", p)

    # 3. Zarr v2 CF scale_factor + add_offset + _FillValue=-32768
    for unscale in (False, True):
        lb = f"W17 ZARR CF int16 unscale={unscale}"
        if wanted(lb):
            p = f"/zarr/bbox/-123,37,-122,38/64x64.npy?url={q(z_cf)}&variable=temp&unscale={'true' if unscale else 'false'}"
            r = client.get(cw_base + p)
            if r.status_code == 200:
                arr = np.load(io.BytesIO(r.content))
                low = np.finfo(arr.dtype).min if arr.dtype.kind == "f" else np.iinfo(arr.dtype).min
                mask = arr[1] > low
                exp_mask = i16 != -32768
                exp_vals = i16.astype(np.float64) * (0.01 if unscale else 1.0) + (273.15 if unscale else 0.0)
                both = mask & exp_mask
                err = float(np.max(np.abs(arr[0][both].astype(np.float64) - exp_vals[both]))) if both.any() else 999.0
                rec(lb, bool((mask == exp_mask).all()) and err < 1e-2, f"mask_eq={float((mask == exp_mask).mean()) * 100:.1f}% err={err:.5f}", p)
            else:
                rec(lb, False, f"HTTP {r.status_code} {r.text[:80]!r}", p)

    # 4. POST /zarr/feature GeoJSON triangle crop (verifies exterior polygon masking + cropped value range)
    lb = "W17 ZARR POST /zarr/feature .npy"
    if wanted(lb):
        poly = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[-122.75, 37.25], [-122.25, 37.25], [-122.75, 37.75], [-122.75, 37.25]]],
            },
            "properties": {},
        }
        p = f"/zarr/feature/64x64.npy?url={q(z_f)}&variable=f_var"
        r = client.post(cw_base + p, json=poly)
        if r.status_code == 200:
            arr = np.load(io.BytesIO(r.content))
            mask = arr[1] > np.finfo(np.float32).min
            frac = float(mask.mean())
            vals = arr[0][mask]
            vmin = float(vals.min()) if vals.size else -1.0
            vmax = float(vals.max()) if vals.size else -1.0
            rec(
                lb,
                arr.shape == (2, 64, 64) and 0.35 < frac < 0.65 and vmin >= 165.0 and vmax <= 500.0,
                f"shape={arr.shape} valid={frac * 100:.1f}% range=[{vmin:.1f}, {vmax:.1f}]",
                p,
                "POST",
            )
        else:
            rec(lb, False, f"HTTP {r.status_code} {r.text[:80]!r}", p, "POST")

    # 5. Zarr multiscales cross-CRS (WebMercatorQuad over EPSG:4326): z10 -> level 0 (100.0), z5 -> level 1 (200.0)
    lb = "W17 ZARR multiscales cross-CRS z10=lvl0 z5=lvl1"
    if wanted(lb):
        p_hi = f"/zarr/tiles/WebMercatorQuad/10/163/396.npy?url={q(z_ms)}&variable=v"
        p_lo = f"/zarr/tiles/WebMercatorQuad/5/5/12.npy?url={q(z_ms)}&variable=v"
        r_hi = client.get(cw_base + p_hi)
        r_lo = client.get(cw_base + p_lo)
        if r_hi.status_code == 200 and r_lo.status_code == 200:
            ah = np.load(io.BytesIO(r_hi.content))
            al = np.load(io.BytesIO(r_lo.content))
            mh = ah[1] > np.finfo(np.float32).min
            ml = al[1] > np.finfo(np.float32).min
            vh = float(np.median(ah[0][mh])) if mh.any() else -1.0
            vl = float(np.median(al[0][ml])) if ml.any() else -1.0
            rec(lb, abs(vh - 100.0) < 1e-3 and abs(vl - 200.0) < 1e-3, f"z10={vh} (exp 100.0) z5={vl} (exp 200.0)", p_hi)
        else:
            rec(lb, False, f"HTTP z10={r_hi.status_code} z5={r_lo.status_code}", p_hi)

    # 6. Zarr tile PNG with colormap_name + rescale + color_formula (CPU vs GPU)
    lb = "W17 ZARR tile PNG colormap+formula"
    if wanted(lb):
        p = f"/zarr/tiles/WebMercatorQuad/8/41/99.png?url={q(z_f)}&variable=f_var&rescale=0,700&colormap_name=viridis&color_formula={qe('Gamma RGB 1.5')}"
        rc = client.get(cw_base + p)
        if rc.status_code == 200:
            rgba = np.array(Image.open(io.BytesIO(rc.content)).convert("RGBA"))
            valid = rgba[..., 3] > 0
            non_gray = bool((rgba[valid, 0] != rgba[valid, 1]).any()) if valid.any() else False
            ok = rgba.shape == (256, 256, 4) and bool(valid.any()) and non_gray
            rec(lb, ok, f"shape={rgba.shape} valid={float(valid.mean()) * 100:.1f}% non_gray={non_gray}", p)
            if gpu_base is not None:
                rg = client.get(gpu_base + p)
                if rg.status_code != 200:
                    gpu_diffs.append({"label": lb, "detail": f"status cpu=200 gpu={rg.status_code}"})
                else:
                    gok, gd = compare_images(rc.content, rg.content)
                    if not gok:
                        gpu_diffs.append({"label": lb, "detail": gd})
        else:
            rec(lb, False, f"HTTP {rc.status_code} {rc.text[:80]!r}", p)

    # --- Wave 25: Big-Endian Zarr v2 (>f4, >i2), Zarr v3 ("endian": "big"), POST /zarr/statistics hole exclusion, TileJSON & WMTS ---
    z_be2 = w17_dir / "w25_v2_be.zarr"
    z_be2.mkdir()
    (z_be2 / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
    _write_v2_zarr_array(z_be2, "lat", lat, (64,), {"_ARRAY_DIMENSIONS": ["lat"], "standard_name": "latitude"})
    _write_v2_zarr_array(z_be2, "lon", lon, (64,), {"_ARRAY_DIMENSIONS": ["lon"], "standard_name": "longitude"})
    _write_v2_zarr_array(z_be2, "be_f32", grad.astype(">f4"), (64, 64), {"_ARRAY_DIMENSIONS": ["lat", "lon"]}, fill_value=-9999.0)
    be_i16 = (-500 + xx * 10 + yy * 5).astype(">i2")
    be_i16[0:16, 0:16] = -32768
    _write_v2_zarr_array(z_be2, "be_i16", be_i16, (64, 64), {"_ARRAY_DIMENSIONS": ["lat", "lon"], "_FillValue": -32768}, fill_value=-32768)

    z_be3 = w17_dir / "w25_v3_be.zarr"
    z_be3.mkdir()
    (z_be3 / "zarr.json").write_text(json.dumps({"zarr_format": 3, "node_type": "group", "attributes": {"Conventions": "CF-1.8"}}))
    for nm, arr_c, std in (("lat", lat, "latitude"), ("lon", lon, "longitude")):
        vd = z_be3 / nm
        vd.mkdir()
        (vd / "zarr.json").write_text(json.dumps({
            "zarr_format": 3, "node_type": "array", "shape": [64], "data_type": "float64",
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [64]}},
            "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
            "fill_value": "NaN", "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
            "dimension_names": [nm], "attributes": {"standard_name": std},
        }))
        (vd / "c").mkdir()
        (vd / "c" / "0").write_bytes(np.ascontiguousarray(arr_c).tobytes())
    vd = z_be3 / "be_var"
    vd.mkdir()
    (vd / "zarr.json").write_text(json.dumps({
        "zarr_format": 3, "node_type": "array", "shape": [64, 64], "data_type": "float32",
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [64, 64]}},
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "fill_value": -9999.0,
        "codecs": [{"name": "bytes", "configuration": {"endian": "big"}}],
        "dimension_names": ["lat", "lon"], "attributes": {},
    }))
    (vd / "c" / "0").mkdir(parents=True)
    (vd / "c" / "0" / "0").write_bytes(np.ascontiguousarray(grad.astype(">f4")).tobytes())

    lb = "W25 ZARR v2 Big-Endian >f4 /bbox .npy"
    if wanted(lb):
        p = f"/zarr/bbox/-123,37,-122,38/64x64.npy?url={q(z_be2)}&variable=be_f32"
        r = client.get(cw_base + p)
        if r.status_code == 200:
            arr = np.load(io.BytesIO(r.content))
            err = float(np.max(np.abs(arr[0] - grad)))
            rec(lb, arr.shape == (2, 64, 64) and err < 1e-3, f"err={err:.5f}", p)
        else:
            rec(lb, False, f"HTTP {r.status_code} {r.text[:80]!r}", p)

    lb = "W25 ZARR v2 Big-Endian >i2 /bbox .npy"
    if wanted(lb):
        p = f"/zarr/bbox/-123,37,-122,38/64x64.npy?url={q(z_be2)}&variable=be_i16"
        r = client.get(cw_base + p)
        if r.status_code == 200:
            arr = np.load(io.BytesIO(r.content))
            mask = arr[1] > np.finfo(np.float32).min
            exp_m = be_i16 != -32768
            err = float(np.max(np.abs(arr[0][exp_m] - be_i16.astype(np.float32)[exp_m])))
            rec(lb, bool((mask == exp_m).all()) and err < 1e-3, f"mask_eq={float((mask == exp_m).mean()) * 100:.1f}% err={err:.5f}", p)
        else:
            rec(lb, False, f"HTTP {r.status_code} {r.text[:80]!r}", p)

    lb = "W25 ZARR v3 Big-Endian /bbox .npy"
    if wanted(lb):
        p = f"/zarr/bbox/-123,37,-122,38/64x64.npy?url={q(z_be3)}&variable=be_var"
        r = client.get(cw_base + p)
        if r.status_code == 200:
            arr = np.load(io.BytesIO(r.content))
            err = float(np.max(np.abs(arr[0] - grad)))
            rec(lb, arr.shape == (2, 64, 64) and err < 1e-3, f"err={err:.5f}", p)
        else:
            rec(lb, False, f"HTTP {r.status_code} {r.text[:80]!r}", p)

    lb = "W25 ZARR POST /statistics hole exclusion"
    if wanted(lb):
        p = f"/zarr/statistics?url={q(z_be2)}&variable=be_f32"
        poly_nh = {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[[-122.75, 37.25], [-122.25, 37.25], [-122.25, 37.75], [-122.75, 37.75], [-122.75, 37.25]]]}, "properties": {}}
        poly_wh = {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[[-122.75, 37.25], [-122.25, 37.25], [-122.25, 37.75], [-122.75, 37.75], [-122.75, 37.25]], [[-122.625, 37.375], [-122.375, 37.375], [-122.375, 37.625], [-122.625, 37.625], [-122.625, 37.375]]]}, "properties": {}}
        r_nh = client.post(cw_base + p, json=poly_nh)
        r_wh = client.post(cw_base + p, json=poly_wh)
        if r_nh.status_code == 200 and r_wh.status_code == 200:
            c_nh = r_nh.json()["properties"]["statistics"]["b1"]["count"]
            c_wh = r_wh.json()["properties"]["statistics"]["b1"]["count"]
            ratio = c_wh / c_nh if c_nh > 0 else 0.0
            rec(lb, abs(c_nh - 1024.0) < 1.0 and abs(ratio - 0.75) < 0.01, f"outer={c_nh:.1f} w_hole={c_wh:.1f} ratio={ratio:.4f}", p, "POST")
        else:
            rec(lb, False, f"HTTP nh={r_nh.status_code} wh={r_wh.status_code}", p, "POST")

    lb = "W25 ZARR TileJSON WebMercatorQuad"
    if wanted(lb):
        p = f"/zarr/WebMercatorQuad/tilejson.json?url={q(z_be2)}&variable=be_f32&rescale=0,300&colormap_name=viridis"
        r = client.get(cw_base + p)
        ok_tj = r.status_code == 200 and "variable=be_f32" in r.json().get("tiles", [""])[0]
        rec(lb, ok_tj, f"status={r.status_code}", p)

    lb = "W25 ZARR WMTS Capabilities XML"
    if wanted(lb):
        p = f"/zarr/WebMercatorQuad/WMTSCapabilities.xml?url={q(z_be2)}&variable=be_f32"
        r = client.get(cw_base + p)
        ok_wm = r.status_code == 200 and ET.fromstring(r.content).tag.endswith("Capabilities") and "variable=be_f32" in r.text
        rec(lb, ok_wm, f"status={r.status_code} len={len(r.text)}", p)

    # --- Wave 26: Zarr CF unscale=true on /point & POST /statistics, plus nodata= query override ---
    z_cf26 = w17_dir / "w26_cf_unscale.zarr"
    z_cf26.mkdir()
    (z_cf26 / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
    _write_v2_zarr_array(z_cf26, "lat", lat, (64,), {"_ARRAY_DIMENSIONS": ["lat"], "standard_name": "latitude"})
    _write_v2_zarr_array(z_cf26, "lon", lon, (64,), {"_ARRAY_DIMENSIONS": ["lon"], "standard_name": "longitude"})
    i16_26 = np.full((64, 64), 1000, dtype=np.int16)
    i16_26[0:16, 0:16] = -32768
    i16_26[16:32, 0:16] = 500
    _write_v2_zarr_array(
        z_cf26, "temp", i16_26, (64, 64),
        {"_ARRAY_DIMENSIONS": ["lat", "lon"], "scale_factor": 0.1, "add_offset": 200.0, "_FillValue": -32768},
        fill_value=-32768,
    )

    lb = "W26 ZARR /point unscale=false vs true"
    if wanted(lb):
        p_raw = f"/zarr/point/-122.25,37.25?url={q(z_cf26)}&variable=temp&unscale=false"
        p_uns = f"/zarr/point/-122.25,37.25?url={q(z_cf26)}&variable=temp&unscale=true"
        r_raw, r_uns = client.get(cw_base + p_raw), client.get(cw_base + p_uns)
        vr = r_raw.json()["values"][0] if r_raw.status_code == 200 else None
        vu = r_uns.json()["values"][0] if r_uns.status_code == 200 else None
        rec(lb, vr == 1000.0 and vu is not None and abs(vu - 300.0) < 1e-3, f"raw={vr} unscale={vu}", p_uns)

    lb = "W26 ZARR POST /statistics unscale=false vs true"
    if wanted(lb):
        poly26 = {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[[-122.5, 37.0], [-122.0, 37.0], [-122.0, 37.5], [-122.5, 37.5], [-122.5, 37.0]]]}, "properties": {}}
        p_sr = f"/zarr/statistics?url={q(z_cf26)}&variable=temp&unscale=false"
        p_su = f"/zarr/statistics?url={q(z_cf26)}&variable=temp&unscale=true"
        r_sr, r_su = client.post(cw_base + p_sr, json=poly26), client.post(cw_base + p_su, json=poly26)
        mr = r_sr.json()["properties"]["statistics"]["b1"]["mean"] if r_sr.status_code == 200 else None
        mu = r_su.json()["properties"]["statistics"]["b1"]["mean"] if r_su.status_code == 200 else None
        rec(lb, mr is not None and abs(mr - 1000.0) < 1e-3 and mu is not None and abs(mu - 300.0) < 1e-3, f"raw={mr} unscale={mu}", p_su, "POST")

    lb = "W26 ZARR nodata=500 query override /bbox .npy"
    if wanted(lb):
        p_nd = f"/zarr/bbox/-123,37,-122,38/64x64.npy?url={q(z_cf26)}&variable=temp&nodata=500"
        r_nd = client.get(cw_base + p_nd)
        if r_nd.status_code == 200:
            arr = np.load(io.BytesIO(r_nd.content))
            m = arr[1] > np.finfo(np.float32).min
            ok_nd = int(m[16:32, 0:16].sum()) == 0 and int(m[0:16, 0:16].sum()) == 0 and int(m[32:64, 32:64].sum()) == 1024
            rec(lb, ok_nd, f"nd500_valid={int(m[16:32,0:16].sum())} fill_valid={int(m[0:16,0:16].sum())} data_valid={int(m[32:64,32:64].sum())}", p_nd)
        else:
            rec(lb, False, f"HTTP {r_nd.status_code}", p_nd)

    return ran


def run_wave26_security_checks(
    client: httpx.Client,
    cw_base: str,
    bin_path: Path,
    tmp_dir: Path,
    only_filter: str,
    failures: list[dict],
) -> int:
    """Wave 26: Live COGZILLA_ALLOWED_URL_PREFIXES SSRF enforcement (direct 403, cross-allowlist 302 redirect, STAC & MosaicJSON fatal 403) + X-Request-ID echo/sanitization."""
    ran = 0

    def wanted(lb: str) -> bool:
        return not only_filter or only_filter.lower() in lb.lower()

    def rec(lb: str, ok: bool, detail: str, path: str = ""):
        nonlocal ran
        ran += 1
        if not ok:
            failures.append({"label": lb, "method": "GET", "path": path, "kind": "w26", "detail": detail})
        print(f"{'PASS' if ok else 'FAIL'} | {lb:54s} | {detail}")

    lb_id = "W26 HTTP X-Request-ID echo & sanitize"
    if wanted(lb_id):
        r1 = client.get(f"{cw_base}/healthz", headers={"X-Request-ID": "trace-abc-123"})
        r2 = client.get(f"{cw_base}/healthz", headers={"X-Request-ID": "bad id with spaces!"})
        ok_id = r1.headers.get("x-request-id") == "trace-abc-123" and r2.headers.get("x-request-id") not in (None, "", "bad id with spaces!")
        rec(lb_id, ok_id, f"echo={r1.headers.get('x-request-id')!r} sanitized={r2.headers.get('x-request-id')!r}", "/healthz")

    sec_labels = [
        "W26 SEC allowed URL -> 200",
        "W26 SEC disallowed URL -> 403",
        "W26 SEC cross-allowlist 302 redirect -> 403",
        "W26 SEC STAC disallowed asset -> 403",
        "W26 SEC MosaicJSON mixed allowed+disallowed -> 403",
    ]
    if not any(wanted(lb) for lb in sec_labels):
        return ran

    w26_dir = tmp_dir / "w26"
    w26_dir.mkdir(parents=True, exist_ok=True)
    tf = from_bounds(-122.5, 37.5, -122.0, 38.0, 64, 64)
    p_ok = w26_dir / "ok.tif"
    with rasterio.open(p_ok, "w", driver="GTiff", width=64, height=64, count=1, dtype="float32", crs="EPSG:4326", transform=tf, tiled=True, blockxsize=64, blockysize=64, compress="deflate") as ds:
        ds.write(np.full((1, 64, 64), 42.0, dtype=np.float32))
    ok_bytes = p_ok.read_bytes()

    srv_port = pick_free_port()
    allowed_prefix = f"http://127.0.0.1:{srv_port}/allowed/"
    stac_evil = json.dumps({
        "type": "Feature", "stac_version": "1.0.0", "id": "evil-stac",
        "geometry": {"type": "Polygon", "coordinates": [[[-122.5, 37.5], [-122.0, 37.5], [-122.0, 38.0], [-122.5, 38.0], [-122.5, 37.5]]]},
        "bbox": [-122.5, 37.5, -122.0, 38.0], "properties": {"datetime": "2024-06-01T00:00:00Z"}, "links": [],
        "assets": {"b1": {"href": f"http://127.0.0.1:{srv_port}/forbidden/secret.tif"}},
    }).encode()
    qk = mercantile.quadkey(164, 396, 10)
    mj_mixed = json.dumps({
        "mosaicjson": "0.0.2", "minzoom": 10, "maxzoom": 10,
        "bounds": [-122.5, 37.5, -122.0, 38.0],
        "tiles": {qk: [f"{allowed_prefix}ok.tif", f"http://127.0.0.1:{srv_port}/forbidden/secret.tif"]},
    }).encode()

    class SecHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _serve(self, is_head: bool):
            if self.path == "/allowed/redir_to_forbidden.tif":
                self.send_response(302)
                self.send_header("Location", f"http://127.0.0.1:{srv_port}/forbidden/secret.tif")
                self.end_headers()
                return
            if self.path == "/allowed/stac_evil.json":
                body, ct = stac_evil, "application/json"
            elif self.path == "/allowed/mj_mixed.json":
                body, ct = mj_mixed, "application/json"
            elif self.path in ("/allowed/ok.tif", "/forbidden/secret.tif"):
                body, ct = ok_bytes, "image/tiff"
            else:
                self.send_response(404)
                self.end_headers()
                return
            rng = self.headers.get("Range")
            if not is_head and rng and rng.startswith("bytes="):
                s0, e0 = rng[6:].split("-", 1)
                start = int(s0)
                end = min(int(e0) if e0 else len(body) - 1, len(body) - 1)
                chunk = body[start:end + 1]
                self.send_response(206)
                self.send_header("Content-Type", ct)
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(body)}")
                self.send_header("Content-Length", str(len(chunk)))
                self.end_headers()
                self.wfile.write(chunk)
            else:
                self.send_response(200)
                self.send_header("Content-Type", ct)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                if not is_head:
                    self.wfile.write(body)

        def do_HEAD(self):
            self._serve(True)

        def do_GET(self):
            self._serve(False)

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", srv_port), SecHandler)
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()

    ps: subprocess.Popen | None = None
    try:
        p_sec = pick_free_port()
        env_sec = os.environ.copy()
        env_sec["COGZILLA_GPU_POLICY"] = "never"
        env_sec["COGZILLA_ALLOWED_URL_PREFIXES"] = allowed_prefix
        ps = subprocess.Popen(
            [str(bin_path), "--bind", f"127.0.0.1:{p_sec}", "--cache-mb", "64"],
            env=env_sec, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        wait_ready(f"http://127.0.0.1:{p_sec}/healthz")
        sec = f"http://127.0.0.1:{p_sec}"
        for lb, path, exp_st in [
            ("W26 SEC allowed URL -> 200", f"/cog/info?url={qe(allowed_prefix + 'ok.tif')}", 200),
            ("W26 SEC disallowed URL -> 403", f"/cog/info?url={qe(f'http://127.0.0.1:{srv_port}/forbidden/secret.tif')}", 403),
            ("W26 SEC cross-allowlist 302 redirect -> 403", f"/cog/info?url={qe(allowed_prefix + 'redir_to_forbidden.tif')}", 403),
            ("W26 SEC STAC disallowed asset -> 403", f"/stac/tiles/WebMercatorQuad/10/164/396.npy?url={qe(allowed_prefix + 'stac_evil.json')}&assets=b1", 403),
            ("W26 SEC MosaicJSON mixed allowed+disallowed -> 403", f"/mosaicjson/tiles/WebMercatorQuad/10/164/396.npy?url={qe(allowed_prefix + 'mj_mixed.json')}", 403),
        ]:
            if wanted(lb):
                r = client.get(sec + path)
                rec(lb, r.status_code == exp_st, f"status={r.status_code} (exp {exp_st})", path)
    finally:
        if ps is not None:
            try:
                if ps.poll() is None:
                    ps.terminate()
                ps.wait(timeout=5)
            except subprocess.TimeoutExpired:
                ps.kill()
                ps.wait(timeout=5)
            except OSError:
                pass
        httpd.shutdown()
        th.join(timeout=5)
        httpd.server_close()

    return ran


def run_wave28_wms_and_mosaic_checks(
    client: httpx.Client,
    cw_base: str,
    gpu_base: str | None,
    tmp_dir: Path,
    only_filter: str | None,
    failures: list[dict],
    gpu_diffs: list[dict],
) -> int:
    """Wave 28: OGC WMS 1.3.0 / 1.1.1 (GetCapabilities, axis order, CRS:84, TRANSPARENT, JPEG/GeoTIFF)
    + MosaicJSON TileJSON / WMTS / FeatureCollection POST /statistics."""
    ran = 0

    def wanted(lb: str) -> bool:
        return not only_filter or only_filter.lower() in lb.lower()

    def rec(lb: str, ok: bool, detail: str, path: str, method: str = "GET") -> None:
        nonlocal ran
        ran += 1
        if not ok:
            failures.append({"label": lb, "method": method, "path": path, "kind": "w28", "detail": detail})
        print(f"{'PASS' if ok else 'FAIL'} | {lb:54s} | {detail}")

    def check_gpu_img(lb: str, path: str, cpu_bytes: bytes) -> None:
        if not gpu_base:
            return
        rg = client.get(gpu_base + path)
        if rg.status_code != 200:
            gpu_diffs.append({"label": lb, "detail": f"GPU HTTP {rg.status_code}"})
            return
        ok_g, det_g = compare_images(cpu_bytes, rg.content)
        if not ok_g:
            gpu_diffs.append({"label": lb, "detail": det_g})

    labels = [
        "W28 WMS 1.3.0 GetCapabilities XML axis order",
        "W28 WMS 1.3.0 GetMap EPSG:4326 (lat,lon) vs GDAL",
        "W28 WMS 1.3.0 GetMap CRS:84 (lon,lat) + Content-Crs",
        "W28 WMS 1.1.1 GetMap SRS=EPSG:4326 (lon,lat) vs GDAL",
        "W28 WMS 1.3.0 GetMap EPSG:3857 (x,y) vs GDAL",
        "W28 WMS TRANSPARENT=FALSE opaque RGB",
        "W28 WMS FORMAT=image/jpeg",
        "W28 WMS FORMAT=image/tiff georeferenced GeoTIFF",
        "W28 WMS invalid REQUEST=Bogus -> 400",
        "W68 WMS 1.3.0 GetMap CRS=EPSG:32610 (projected UTM) vs GDAL",
        "W68 WMS 1.3.0 GetMap CRS=EPSG:4258 (ETRS89 lat,lon swap) vs GDAL",
        "W68 WMS 1.3.0 GetMap COLORMAP_NAME=viridis + RESCALE + UNSCALE=true",
        "W28 MOSAIC TileJSON WebMercatorQuad",
        "W28 MOSAIC WMTS Capabilities XML",
        "W28 MOSAIC POST /statistics FeatureCollection",
        "W38 MOSAIC POST /feature 64x64.npy",
        "W38 MOSAIC GET /bbox 64x64.npy",
        "W38 MOSAIC GET /point",
        "W43 MOSAIC expression + resampling=cubic .npy",
        "W43 MOSAIC expression + colormap_name=viridis .png",
        "W54 MOSAIC bidx=3,2,1 + pixel_selection=median + color_formula .png",
        "W54 MOSAIC POST /statistics donut coord_crs=EPSG:3857",
    ]
    if not any(wanted(lb) for lb in labels):
        return 0

    w28_dir = tmp_dir / "w28"
    w28_dir.mkdir(parents=True, exist_ok=True)
    w, h = 128, 128
    tf = from_bounds(-122.5, 37.5, -122.0, 38.0, w, h)
    yy, xx = np.mgrid[0:h, 0:w]
    arr = np.stack([
        np.maximum(((xx * 2) % 255).astype(np.uint8), 1),
        np.maximum(((yy * 2) % 255).astype(np.uint8), 1),
        np.maximum(((xx + yy) % 255).astype(np.uint8), 1),
    ], axis=0)
    arr[:, 48:80, 48:80] = 0  # interior nodata hole
    cog_p = w28_dir / "wms_hole.tif"
    with rasterio.open(
        cog_p, "w", driver="GTiff", width=w, height=h, count=3, dtype="uint8",
        crs="EPSG:4326", transform=tf, tiled=True, blockxsize=64, blockysize=64,
        compress="deflate", nodata=0,
    ) as ds:
        ds.write(arr)
    gt_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    gt_rgba[:, :, :3] = np.transpose(arr, (1, 2, 0))
    gt_rgba[:, :, 3] = np.where((arr[0] == 0) & (arr[1] == 0) & (arr[2] == 0), 0, 255).astype(np.uint8)
    valid_mask = gt_rgba[:, :, 3] == 255

    # Build MosaicJSON covering [-122.5, 37.5, -122.0, 38.0] at z=10
    qks = sorted({
        mercantile.quadkey(mercantile.tile(lo, la, 10))
        for lo in (-122.45, -122.25, -122.05)
        for la in (37.55, 37.75, 37.95)
    })
    mj_p = w28_dir / "w28_mosaic.json"
    mj_p.write_text(json.dumps({
        "mosaicjson": "0.0.3",
        "minzoom": 8,
        "maxzoom": 14,
        "bounds": [-122.5, 37.5, -122.0, 38.0],
        "quadkey_zoom": 10,
        "tiles": {qk: [str(cog_p)] for qk in qks},
    }))

    lb = "W28 WMS 1.3.0 GetCapabilities XML axis order"
    if wanted(lb):
        p = f"/cog/wms?url={q(cog_p)}&SERVICE=WMS&VERSION=1.3.0&REQUEST=GetCapabilities"
        r = client.get(cw_base + p)
        if r.status_code == 200:
            root = ET.fromstring(r.content)
            ns = {"w": "http://www.opengis.net/wms"}
            bbs = {
                el.attrib.get("CRS"): tuple(float(el.attrib[k]) for k in ("minx", "miny", "maxx", "maxy"))
                for el in root.findall(".//w:BoundingBox", ns)
            }
            ok = (
                root.tag.endswith("WMS_Capabilities")
                and bbs.get("CRS:84") == (-122.5, 37.5, -122.0, 38.0)
                and bbs.get("EPSG:4326") == (37.5, -122.5, 38.0, -122.0)
                and "EPSG:3857" in bbs
            )
            rec(lb, ok, f"root={root.tag.split('}')[-1]} crs84={bbs.get('CRS:84')} epsg4326={bbs.get('EPSG:4326')}", p)
        else:
            rec(lb, False, f"HTTP {r.status_code} {r.text[:80]!r}", p)

    lb = "W28 WMS 1.3.0 GetMap EPSG:4326 (lat,lon) vs GDAL"
    if wanted(lb):
        p = (
            f"/cog/wms?url={q(cog_p)}&SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap"
            f"&LAYERS=cog&STYLES=&CRS=EPSG:4326&BBOX=37.5,-122.5,38.0,-122.0&WIDTH=128&HEIGHT=128&FORMAT=image/png"
        )
        r = client.get(cw_base + p)
        if r.status_code == 200:
            im = np.array(Image.open(io.BytesIO(r.content)).convert("RGBA"))
            mae = float(np.abs(im[valid_mask, :3].astype(float) - gt_rgba[valid_mask, :3].astype(float)).mean())
            ok = im.shape == (128, 128, 4) and mae < 0.01 and int(im[64, 64, 3]) == 0 and int(im[10, 10, 3]) == 255
            rec(lb, ok, f"MAE={mae:.2f} hole_a={im[64,64,3]} valid_a={im[10,10,3]}", p)
            check_gpu_img(lb, p, r.content)
        else:
            rec(lb, False, f"HTTP {r.status_code} {r.text[:80]!r}", p)

    lb = "W28 WMS 1.3.0 GetMap CRS:84 (lon,lat) + Content-Crs"
    if wanted(lb):
        p = (
            f"/cog/wms?url={q(cog_p)}&SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap"
            f"&LAYERS=cog&STYLES=&CRS=CRS:84&BBOX=-122.5,37.5,-122.0,38.0&WIDTH=128&HEIGHT=128&FORMAT=image/png&TRANSPARENT=TRUE"
        )
        r = client.get(cw_base + p)
        if r.status_code == 200:
            im = np.array(Image.open(io.BytesIO(r.content)).convert("RGBA"))
            mae = float(np.abs(im[valid_mask, :3].astype(float) - gt_rgba[valid_mask, :3].astype(float)).mean())
            crs_hdr = r.headers.get("content-crs", "")
            ok = mae < 0.01 and int(im[64, 64, 3]) == 0 and "CRS84" in crs_hdr
            rec(lb, ok, f"MAE={mae:.2f} hole_a={im[64,64,3]} Content-Crs={crs_hdr}", p)
            check_gpu_img(lb, p, r.content)
        else:
            rec(lb, False, f"HTTP {r.status_code} {r.text[:80]!r}", p)

    lb = "W28 WMS 1.1.1 GetMap SRS=EPSG:4326 (lon,lat) vs GDAL"
    if wanted(lb):
        p = (
            f"/cog/wms?url={q(cog_p)}&SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap"
            f"&SRS=EPSG:4326&BBOX=-122.5,37.5,-122.0,38.0&WIDTH=128&HEIGHT=128&FORMAT=image/png"
        )
        r = client.get(cw_base + p)
        if r.status_code == 200:
            im = np.array(Image.open(io.BytesIO(r.content)).convert("RGBA"))
            mae = float(np.abs(im[valid_mask, :3].astype(float) - gt_rgba[valid_mask, :3].astype(float)).mean())
            ok = mae < 0.01 and int(im[64, 64, 3]) == 0
            rec(lb, ok, f"MAE={mae:.2f} hole_a={im[64,64,3]}", p)
            check_gpu_img(lb, p, r.content)
        else:
            rec(lb, False, f"HTTP {r.status_code} {r.text[:80]!r}", p)

    lb = "W28 WMS 1.3.0 GetMap EPSG:3857 (x,y) vs GDAL"
    if wanted(lb):
        mb = transform_bounds("EPSG:4326", "EPSG:3857", -122.5, 37.5, -122.0, 38.0)
        p = (
            f"/cog/wms?url={q(cog_p)}&SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap"
            f"&CRS=EPSG:3857&BBOX={mb[0]},{mb[1]},{mb[2]},{mb[3]}&WIDTH=128&HEIGHT=128&FORMAT=image/png"
        )
        r = client.get(cw_base + p)
        if r.status_code == 200:
            from rasterio.warp import reproject, Resampling as RioResampling
            dst_tf = from_bounds(*mb, 128, 128)
            gt3857 = np.zeros((3, 128, 128), dtype=np.uint8)
            for b in range(3):
                reproject(
                    source=arr[b], destination=gt3857[b],
                    src_transform=tf, src_crs="EPSG:4326", src_nodata=0,
                    dst_transform=dst_tf, dst_crs="EPSG:3857", dst_nodata=0,
                    resampling=RioResampling.nearest,
                )
            im = np.array(Image.open(io.BytesIO(r.content)).convert("RGBA"))
            vm = (im[:, :, 3] == 255) & (gt3857[0] > 0)
            mae = float(np.abs(im[vm, :3].astype(float) - np.transpose(gt3857, (1, 2, 0))[vm].astype(float)).mean())
            ok = bool(vm.sum() > 10000) and mae < 1.0 and int(im[64, 64, 3]) == 0
            rec(lb, ok, f"MAE={mae:.2f} valid_px={int(vm.sum())} hole_a={im[64,64,3]}", p)
            check_gpu_img(lb, p, r.content)
        else:
            rec(lb, False, f"HTTP {r.status_code} {r.text[:80]!r}", p)

    lb = "W28 WMS TRANSPARENT=FALSE opaque RGB"
    if wanted(lb):
        p = (
            f"/cog/wms?url={q(cog_p)}&SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap"
            f"&CRS=CRS:84&BBOX=-122.5,37.5,-122.0,38.0&WIDTH=128&HEIGHT=128&FORMAT=image/png&TRANSPARENT=FALSE"
        )
        r = client.get(cw_base + p)
        if r.status_code == 200:
            pil = Image.open(io.BytesIO(r.content))
            im = np.array(pil)
            ok = pil.mode == "RGB" and im[64, 64].tolist() == [0, 0, 0] and im[10, 10].tolist() == gt_rgba[10, 10, :3].tolist()
            rec(lb, ok, f"mode={pil.mode} hole={im[64,64].tolist()} valid={im[10,10].tolist()}", p)
            check_gpu_img(lb, p, r.content)
        else:
            rec(lb, False, f"HTTP {r.status_code} {r.text[:80]!r}", p)

    lb = "W28 WMS FORMAT=image/jpeg"
    if wanted(lb):
        p = (
            f"/cog/wms?url={q(cog_p)}&SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap"
            f"&CRS=CRS:84&BBOX=-122.5,37.5,-122.0,38.0&WIDTH=128&HEIGHT=128&FORMAT=image/jpeg"
        )
        r = client.get(cw_base + p)
        if r.status_code == 200:
            pil = Image.open(io.BytesIO(r.content))
            im = np.array(pil)
            mae = float(np.abs(im[valid_mask].astype(float) - gt_rgba[valid_mask, :3].astype(float)).mean())
            ok = r.headers.get("content-type", "").startswith("image/jpeg") and pil.mode == "RGB" and mae < 5.0
            rec(lb, ok, f"ct={r.headers.get('content-type')} mode={pil.mode} MAE={mae:.2f}", p)
            check_gpu_img(lb, p, r.content)
        else:
            rec(lb, False, f"HTTP {r.status_code} {r.text[:80]!r}", p)

    lb = "W28 WMS FORMAT=image/tiff georeferenced GeoTIFF"
    if wanted(lb):
        p = (
            f"/cog/wms?url={q(cog_p)}&SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap"
            f"&CRS=EPSG:4326&BBOX=37.5,-122.5,38.0,-122.0&WIDTH=128&HEIGHT=128&FORMAT=image/tiff"
        )
        r = client.get(cw_base + p)
        if r.status_code == 200:
            with rasterio.MemoryFile(r.content) as mf, mf.open() as ds:
                d = ds.read()
                b = ds.bounds
                max_diff = int(np.max(np.abs(d[:3, 10:30, 10:30].astype(int) - arr[:, 10:30, 10:30].astype(int))))
                ok = (
                    ds.crs is not None and ds.crs.to_epsg() == 4326
                    and abs(b.left - (-122.5)) < 1e-6 and abs(b.bottom - 37.5) < 1e-6
                    and abs(b.right - (-122.0)) < 1e-6 and abs(b.top - 38.0) < 1e-6
                    and max_diff == 0
                )
                rec(lb, ok, f"crs={ds.crs} shape={d.shape} max_diff={max_diff}", p)
            if gpu_base:
                rg = client.get(gpu_base + p)
                if rg.status_code != 200:
                    gpu_diffs.append({"label": lb, "detail": f"GPU HTTP {rg.status_code}"})
                else:
                    ok_g, det_g = compare_geotiffs(r.content, rg.content, compare_pixels=True)
                    if not ok_g:
                        gpu_diffs.append({"label": lb, "detail": det_g})
        else:
            rec(lb, False, f"HTTP {r.status_code} {r.text[:80]!r}", p)

    lb = "W28 WMS invalid REQUEST=Bogus -> 400"
    if wanted(lb):
        p = f"/cog/wms?url={q(cog_p)}&SERVICE=WMS&VERSION=1.3.0&REQUEST=Bogus"
        r = client.get(cw_base + p)
        rec(lb, r.status_code == 400, f"status={r.status_code}", p)

    def _gdal_reproject(dst_crs: str, dst_bounds: tuple[float, float, float, float], width: int, height: int) -> np.ndarray:
        from rasterio.warp import reproject as _warp_reproject, Resampling as _WarpResampling
        dst = np.zeros((3, height, width), dtype=np.uint8)
        dst_tf = from_bounds(*dst_bounds, width, height)
        for bi in range(3):
            _warp_reproject(
                source=arr[bi], destination=dst[bi],
                src_transform=tf, src_crs="EPSG:4326", src_nodata=0,
                dst_transform=dst_tf, dst_crs=dst_crs, dst_nodata=0,
                resampling=_WarpResampling.nearest,
            )
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        rgba[:, :, :3] = np.transpose(dst, (1, 2, 0))
        rgba[:, :, 3] = np.where((dst[0] == 0) & (dst[1] == 0) & (dst[2] == 0), 0, 255).astype(np.uint8)
        return rgba

    lb = "W68 WMS 1.3.0 GetMap CRS=EPSG:32610 (projected UTM) vs GDAL"
    if wanted(lb):
        utm_b = transform_bounds("EPSG:4326", "EPSG:32610", -122.45, 37.55, -122.05, 37.95)
        p = (
            f"/cog/wms?url={q(cog_p)}&SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap"
            f"&CRS=EPSG:32610&BBOX={utm_b[0]:.2f},{utm_b[1]:.2f},{utm_b[2]:.2f},{utm_b[3]:.2f}"
            f"&WIDTH=128&HEIGHT=128&FORMAT=image/png"
        )
        r = client.get(cw_base + p)
        if r.status_code == 200:
            got = np.array(Image.open(io.BytesIO(r.content)).convert("RGBA"))
            ref = _gdal_reproject("EPSG:32610", utm_b, 128, 128)
            valid = (got[:, :, 3] == 255) & (ref[:, :, 3] == 255)
            mae = float(np.abs(got[valid, :3].astype(float) - ref[valid, :3].astype(float)).mean()) if valid.any() else 999.0
            ok = mae < 1.5 and int(valid.sum()) > 10000
            rec(lb, ok, f"MAE={mae:.2f} valid_px={int(valid.sum())}", p)
            check_gpu_img(lb, p, r.content)
        else:
            rec(lb, False, f"HTTP {r.status_code} {r.text[:80]!r}", p)

    lb = "W68 WMS 1.3.0 GetMap CRS=EPSG:4258 (ETRS89 lat,lon swap) vs GDAL"
    if wanted(lb):
        p = (
            f"/cog/wms?url={q(cog_p)}&SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap"
            f"&CRS=EPSG:4258&BBOX=37.5,-122.5,38.0,-122.0&WIDTH=128&HEIGHT=128&FORMAT=image/png"
        )
        r = client.get(cw_base + p)
        if r.status_code == 200:
            got = np.array(Image.open(io.BytesIO(r.content)).convert("RGBA"))
            ref = _gdal_reproject("EPSG:4258", (-122.5, 37.5, -122.0, 38.0), 128, 128)
            valid = (got[:, :, 3] == 255) & (ref[:, :, 3] == 255)
            mae = float(np.abs(got[valid, :3].astype(float) - ref[valid, :3].astype(float)).mean()) if valid.any() else 999.0
            hole_a = int(got[56, 56, 3])
            ok = mae < 1.0 and hole_a == 0 and int(valid.sum()) > 14000
            rec(lb, ok, f"MAE={mae:.2f} hole_a={hole_a} valid_px={int(valid.sum())}", p)
            check_gpu_img(lb, p, r.content)
        else:
            rec(lb, False, f"HTTP {r.status_code} {r.text[:80]!r}", p)

    lb = "W68 WMS 1.3.0 GetMap COLORMAP_NAME=viridis + RESCALE + UNSCALE=true"
    if wanted(lb):
        wms_scaled = w28_dir / "w68_wms_scaled.tif"
        yy_s, xx_s = np.mgrid[0:128, 0:128]
        u16_s = (1000 + xx_s * 20 + yy_s * 15).astype(np.uint16)
        u16_s[40:60, 40:60] = 0
        with rasterio.open(
            wms_scaled, "w", driver="GTiff", width=128, height=128, count=1, dtype="uint16",
            crs="EPSG:4326", transform=tf, tiled=True, blockxsize=64, blockysize=64,
            compress="deflate", nodata=0,
        ) as ds:
            ds.write(u16_s, 1)
            ds.scales = (0.001,)
            ds.offsets = (-0.5,)
        p = (
            f"/cog/wms?url={q(wms_scaled)}&SERVICE=WMS&VERSION=1.3.0&REQUEST=GetMap"
            f"&CRS=EPSG:4326&BBOX=37.5,-122.5,38.0,-122.0&WIDTH=128&HEIGHT=128&FORMAT=image/png"
            f"&unscale=true&rescale=0.5,5.0&colormap_name=viridis"
        )
        r = client.get(cw_base + p)
        if r.status_code == 200:
            got = np.array(Image.open(io.BytesIO(r.content)).convert("RGBA"))
            # Compare against /cog/bbox with identical params
            pb = f"/cog/bbox/-122.5,37.5,-122.0,38.0/128x128.png?url={q(wms_scaled)}&unscale=true&rescale=0.5,5.0&colormap_name=viridis"
            rb = client.get(cw_base + pb)
            if rb.status_code == 200:
                ref = np.array(Image.open(io.BytesIO(rb.content)).convert("RGBA"))
                mae = float(np.abs(got.astype(float) - ref.astype(float)).mean())
                hole_a = int(got[50, 50, 3])
                ok = mae == 0.0 and hole_a == 0 and int((got[:, :, 3] == 255).sum()) > 15000
                rec(lb, ok, f"MAE_vs_bbox={mae:.2f} hole_a={hole_a}", p)
                check_gpu_img(lb, p, r.content)
            else:
                rec(lb, False, f"ref /bbox HTTP {rb.status_code}", p)
        else:
            rec(lb, False, f"HTTP {r.status_code} {r.text[:80]!r}", p)

    lb = "W28 MOSAIC TileJSON WebMercatorQuad"
    if wanted(lb):
        p = f"/mosaicjson/WebMercatorQuad/tilejson.json?url={q(mj_p)}"
        r = client.get(cw_base + p)
        if r.status_code == 200:
            j = r.json()
            ok = (
                j.get("tilejson") == "3.0.0"
                and j.get("bounds") == [-122.5, 37.5, -122.0, 38.0]
                and j.get("minzoom") == 8 and j.get("maxzoom") == 14
                and "/mosaicjson/tiles/WebMercatorQuad/" in j.get("tiles", [""])[0]
            )
            rec(lb, ok, f"tilejson={j.get('tilejson')} z={j.get('minzoom')}..{j.get('maxzoom')}", p)
        else:
            rec(lb, False, f"HTTP {r.status_code} {r.text[:80]!r}", p)

    lb = "W28 MOSAIC WMTS Capabilities XML"
    if wanted(lb):
        p = f"/mosaicjson/WebMercatorQuad/WMTSCapabilities.xml?url={q(mj_p)}"
        r = client.get(cw_base + p)
        if r.status_code == 200:
            root = ET.fromstring(r.content)
            ok = root.tag.endswith("Capabilities") and "WebMercatorQuad" in r.text
            rec(lb, ok, f"root={root.tag.split('}')[-1]} len={len(r.text)}", p)
        else:
            rec(lb, False, f"HTTP {r.status_code} {r.text[:80]!r}", p)

    lb = "W28 MOSAIC POST /statistics FeatureCollection"
    if wanted(lb):
        p = f"/mosaicjson/statistics?url={q(mj_p)}"
        fc = {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "id": "f1", "properties": {}, "geometry": {
                    "type": "Polygon", "coordinates": [[[-122.45, 37.85], [-122.35, 37.85], [-122.35, 37.95], [-122.45, 37.95], [-122.45, 37.85]]]
                }},
                {"type": "Feature", "id": "f2", "properties": {}, "geometry": {
                    "type": "Polygon", "coordinates": [[[-122.2, 37.55], [-122.1, 37.55], [-122.1, 37.65], [-122.2, 37.65], [-122.2, 37.55]]]
                }},
            ],
        }
        r = client.post(cw_base + p, json=fc)
        if r.status_code == 200:
            j = r.json()
            feats = j.get("features", [])
            if j.get("type") == "FeatureCollection" and len(feats) == 2:
                m1 = feats[0]["properties"]["statistics"]["b1"]["mean"]
                m2 = feats[1]["properties"]["statistics"]["b1"]["mean"]
                # f1 is in left columns (x ~ 13..38 -> b1 ~ 26..76 -> mean ~ 51)
                # f2 is in right columns (x ~ 77..102 -> b1 ~ 154..204 -> mean ~ 179)
                ok = 40.0 < m1 < 65.0 and 165.0 < m2 < 195.0
                rec(lb, ok, f"n_feat=2 f1.b1.mean={m1:.1f} f2.b1.mean={m2:.1f}", p, "POST")
            else:
                rec(lb, False, f"unexpected FC shape: {j.get('type')} n={len(feats)}", p, "POST")
        else:
            rec(lb, False, f"HTTP {r.status_code} {r.text[:80]!r}", p, "POST")

    # --- Wave 38: MosaicJSON POST /feature, GET /bbox & GET /point ---
    lb = "W38 MOSAIC POST /feature 64x64.npy"
    if wanted(lb):
        p = f"/mosaicjson/feature/64x64.npy?url={q(mj_p)}"
        tri = {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[[-122.45, 37.55], [-122.05, 37.55], [-122.25, 37.95], [-122.45, 37.55]]]}}
        r = client.post(cw_base + p, json=tri)
        if r.status_code == 200:
            a = np.load(io.BytesIO(r.content))
            vm = a[3] > 0
            rec(lb, a.shape == (4, 64, 64) and a.dtype == np.uint8 and 1500 < int(vm.sum()) < 2600, f"shape={a.shape} valid={int(vm.sum())}", p, "POST")
            if gpu_base:
                rg = client.post(gpu_base + p, json=tri)
                if rg.status_code != 200:
                    gpu_diffs.append({"label": lb, "detail": f"GPU HTTP {rg.status_code}"})
                else:
                    ok_g, det_g = compare_npy(r.content, rg.content, check_values=True)
                    if not ok_g:
                        gpu_diffs.append({"label": lb, "detail": det_g})
        else:
            rec(lb, False, f"HTTP {r.status_code} {r.text[:80]!r}", p, "POST")

    lb = "W38 MOSAIC GET /bbox 64x64.npy"
    if wanted(lb):
        p = f"/mosaicjson/bbox/-122.45,37.85,-122.35,37.95/64x64.npy?url={q(mj_p)}"
        r = client.get(cw_base + p)
        if r.status_code == 200:
            a = np.load(io.BytesIO(r.content))
            vm = a[3] > 0
            rec(lb, a.shape == (4, 64, 64) and int(vm.sum()) == 4096 and 40.0 < float(a[0].mean()) < 65.0, f"shape={a.shape} valid={int(vm.sum())} b1_mean={float(a[0].mean()):.1f}", p)
            if gpu_base:
                rg = client.get(gpu_base + p)
                if rg.status_code != 200:
                    gpu_diffs.append({"label": lb, "detail": f"GPU HTTP {rg.status_code}"})
                else:
                    ok_g, det_g = compare_npy(r.content, rg.content, check_values=True)
                    if not ok_g:
                        gpu_diffs.append({"label": lb, "detail": det_g})
        else:
            rec(lb, False, f"HTTP {r.status_code} {r.text[:80]!r}", p)

    lb = "W38 MOSAIC GET /point"
    if wanted(lb):
        p = f"/mosaicjson/point/-122.45,37.95?url={q(mj_p)}"
        r = client.get(cw_base + p)
        if r.status_code == 200:
            j = r.json()
            assets = j.get("assets", [])
            vals = assets[0].get("values", []) if assets else []
            ok = len(assets) >= 1 and len(vals) == 3 and all(v is not None and v > 0 for v in vals)
            rec(lb, ok, f"n_assets={len(assets)} vals={vals}", p)
            if gpu_base:
                rg = client.get(gpu_base + p)
                if rg.status_code != 200:
                    gpu_diffs.append({"label": lb, "detail": f"GPU HTTP {rg.status_code}"})
                else:
                    d_pt = first_json_diff(j, rg.json())
                    if d_pt is not None:
                        gpu_diffs.append({"label": lb, "detail": d_pt})
        else:
            rec(lb, False, f"HTTP {r.status_code} {r.text[:80]!r}", p)

    lb = "W43 MOSAIC expression + resampling=cubic .npy"
    if wanted(lb):
        p = f"/mosaicjson/tiles/WebMercatorQuad/10/164/396.npy?url={q(mj_p)}&expression={qe('(b2-b1)/(b2+b1+100)')}&resampling=cubic"
        r = client.get(cw_base + p)
        if r.status_code == 200:
            a = np.load(io.BytesIO(r.content))
            vm = a[1] > np.finfo(a.dtype).min
            astd = float(a[0][vm].std()) if vm.any() else 0.0
            amax = float(np.abs(a[0][vm]).max()) if vm.any() else 999.0
            ok = a.shape == (2, 256, 256) and a.dtype == np.float64 and int(vm.sum()) > 10000 and astd > 0.05 and amax <= 1.5
            rec(lb, ok, f"shape={a.shape} dtype={a.dtype} valid={int(vm.sum())} std={astd:.4f} max={amax:.4f}", p)
            if gpu_base:
                rg = client.get(gpu_base + p)
                if rg.status_code != 200:
                    gpu_diffs.append({"label": lb, "detail": f"GPU HTTP {rg.status_code}"})
                else:
                    ok_g, det_g = compare_npy(r.content, rg.content, check_values=True, smooth=True)
                    if not ok_g:
                        gpu_diffs.append({"label": lb, "detail": det_g})
        else:
            rec(lb, False, f"HTTP {r.status_code} {r.text[:80]!r}", p)

    lb = "W43 MOSAIC expression + colormap_name=viridis .png"
    if wanted(lb):
        p = f"/mosaicjson/tiles/WebMercatorQuad/10/164/396.png?url={q(mj_p)}&expression={qe('(b2-b1)/(b2+b1+100)')}&rescale=-1,1&colormap_name=viridis"
        r = client.get(cw_base + p)
        if r.status_code == 200:
            im = np.array(Image.open(io.BytesIO(r.content)).convert("RGBA"))
            vm = im[..., 3] == 255
            cstd = float(im[vm, :3].std(axis=0).max()) if vm.any() else 0.0
            ok = im.shape == (256, 256, 4) and int(vm.sum()) > 10000 and cstd > 5.0
            rec(lb, ok, f"shape={im.shape} valid={int(vm.sum())} std={cstd:.1f}", p)
            if gpu_base:
                rg = client.get(gpu_base + p)
                if rg.status_code != 200:
                    gpu_diffs.append({"label": lb, "detail": f"GPU HTTP {rg.status_code}"})
                else:
                    ok_g, det_g = compare_images(r.content, rg.content)
                    if not ok_g:
                        gpu_diffs.append({"label": lb, "detail": det_g})
        else:
            rec(lb, False, f"HTTP {r.status_code} {r.text[:80]!r}", p)

    lb = "W54 MOSAIC bidx=3,2,1 + pixel_selection=median + color_formula .png"
    if wanted(lb):
        p = f"/mosaicjson/tiles/WebMercatorQuad/10/164/396.png?url={q(mj_p)}&bidx=3&bidx=2&bidx=1&pixel_selection=median&color_formula={qe('Gamma RGB 1.4')}"
        r = client.get(cw_base + p)
        if r.status_code == 200:
            im = np.array(Image.open(io.BytesIO(r.content)).convert("RGBA"))
            vm = im[..., 3] == 255
            cstd = float(im[vm, :3].std(axis=0).max()) if vm.any() else 0.0
            ok = im.shape == (256, 256, 4) and int(vm.sum()) > 10000 and cstd > 5.0
            rec(lb, ok, f"shape={im.shape} valid={int(vm.sum())} std={cstd:.1f}", p)
            if gpu_base:
                rg = client.get(gpu_base + p)
                if rg.status_code != 200:
                    gpu_diffs.append({"label": lb, "detail": f"GPU HTTP {rg.status_code}"})
                else:
                    ok_g, det_g = compare_images(r.content, rg.content)
                    if not ok_g:
                        gpu_diffs.append({"label": lb, "detail": det_g})
        else:
            rec(lb, False, f"HTTP {r.status_code} {r.text[:80]!r}", p)

    lb = "W54 MOSAIC POST /statistics donut coord_crs=EPSG:3857"
    if wanted(lb):
        ox, oy = rasterio.warp.transform("EPSG:4326", "EPSG:3857", [-122.48, -122.02, -122.02, -122.48, -122.48], [37.52, 37.52, 37.98, 37.98, 37.52])
        ix, iy = rasterio.warp.transform("EPSG:4326", "EPSG:3857", [-122.34, -122.34, -122.16, -122.16, -122.34], [37.66, 37.84, 37.84, 37.66, 37.66])
        donut_3857 = {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [list(zip(ox, oy)), list(zip(ix, iy))]}}
        p = f"/mosaicjson/statistics?url={q(mj_p)}&coord_crs=EPSG:3857"
        r = client.post(cw_base + p, json=donut_3857)
        if r.status_code == 200:
            st = r.json().get("properties", {}).get("statistics", {})
            b1 = st.get("b1", {})
            vp = float(b1.get("valid_percent", 0.0))
            cnt = float(b1.get("count", 0.0))
            # Exterior 0.46x0.46 deg (~13867 px) minus interior hole 0.18x0.18 deg (~2123 px) = ~11744 px
            ok = "b1" in st and abs(vp - 100.0) < 0.5 and 11000.0 < cnt < 12500.0
            rec(lb, ok, f"bands={sorted(st.keys())} b1.valid_percent={vp:.1f} count={cnt:.1f} (exp ~11744)", p)
            if gpu_base:
                rg = client.post(gpu_base + p, json=donut_3857)
                if rg.status_code != 200:
                    gpu_diffs.append({"label": lb, "detail": f"GPU HTTP {rg.status_code}"})
                else:
                    d_st = first_json_diff(r.json(), rg.json())
                    if d_st is not None:
                        gpu_diffs.append({"label": lb, "detail": d_st})
        else:
            rec(lb, False, f"HTTP {r.status_code} {r.text[:80]!r}", p)

    return ran


def run_wave29_http_protocol_checks(
    client: httpx.Client,
    cw_base: str,
    s: dict[str, Path],
    only_filter: str | None,
    failures: list[dict],
) -> int:
    """Wave 29: HTTP ETag / If-None-Match 304 revalidation, HEAD empty body + ETag, and CORS OPTIONS preflight."""
    ran = 0

    def wanted(lb: str) -> bool:
        return not only_filter or only_filter.lower() in lb.lower()

    def rec(lb: str, ok: bool, detail: str, path: str, method: str = "GET") -> None:
        nonlocal ran
        ran += 1
        if not ok:
            failures.append({"label": lb, "method": method, "path": path, "kind": "w29_http", "detail": detail})
        print(f"{'PASS' if ok else 'FAIL'} | {lb:54s} | {detail}")

    tile_p = f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w29_lzw_p2_u16'])}&rescale=0,60000&colormap_name=viridis"

    lb = "W29 HTTP ETag + If-None-Match -> 304 Not Modified"
    if wanted(lb):
        r1 = client.get(cw_base + tile_p, headers={"Origin": "https://example.com"})
        etag = r1.headers.get("etag")
        if r1.status_code == 200 and etag:
            r2 = client.get(cw_base + tile_p, headers={"If-None-Match": etag})
            ok = r2.status_code == 304 and len(r2.content) == 0 and r2.headers.get("etag") == etag
            rec(lb, ok, f"etag={etag!r} 304_st={r2.status_code} 304_len={len(r2.content)}", tile_p)
        else:
            rec(lb, False, f"GET st={r1.status_code} etag={etag!r}", tile_p)

    lb = "W29 HTTP HEAD tile -> 200 empty body + matching ETag"
    if wanted(lb):
        rg = client.get(cw_base + tile_p)
        rh = client.head(cw_base + tile_p)
        ok = (
            rg.status_code == 200
            and rh.status_code == 200
            and len(rh.content) == 0
            and rh.headers.get("content-type") == "image/png"
            and rh.headers.get("etag") is not None
            and rh.headers.get("etag") == rg.headers.get("etag")
        )
        rec(lb, ok, f"st={rh.status_code} len={len(rh.content)} etag_match={rh.headers.get('etag') == rg.headers.get('etag')}", tile_p, "HEAD")

    lb = "W29 HTTP OPTIONS CORS preflight -> 204 + ACAO/ACAM"
    if wanted(lb):
        ro = client.options(
            cw_base + tile_p,
            headers={"Origin": "https://example.com", "Access-Control-Request-Method": "GET"},
        )
        acao = ro.headers.get("access-control-allow-origin", "")
        acam = ro.headers.get("access-control-allow-methods", "")
        ok = ro.status_code == 204 and acao == "*" and "GET" in acam and "POST" in acam
        rec(lb, ok, f"st={ro.status_code} acao={acao!r} acam={acam!r}", tile_p, "OPTIONS")

    return ran


def run_wave30_zarr_codecs_multiurl_rpc_checks(
    client: httpx.Client,
    cw_base: str,
    gpu_base: str | None,
    tmp_dir: Path,
    only_filter: str | None,
    failures: list[dict],
    gpu_diffs: list[dict],
) -> int:
    """Wave 30: Zarr v2 codecs (Gzip, LZ4 numcodecs block, Blosc zlib+byte-shuffle, Blosc zlib+bit-shuffle,
    Blosc memcpy), multi-URL COG /point & /bbox stacking, and RPC-only / ungeoreferenced TIFF diagnostics."""
    ran = 0

    def wanted(lb: str) -> bool:
        return not only_filter or only_filter.lower() in lb.lower()

    def rec(lb: str, ok: bool, detail: str, path: str, method: str = "GET") -> None:
        nonlocal ran
        ran += 1
        if not ok:
            failures.append({"label": lb, "method": method, "path": path, "kind": "w30", "detail": detail})
        print(f"{'PASS' if ok else 'FAIL'} | {lb:54s} | {detail}")

    labels = [
        "W30 ZARR v2 Gzip /bbox .npy",
        "W30 ZARR v2 LZ4 raw block /bbox .npy",
        "W30 ZARR v2 Blosc zlib+byte-shuffle /bbox .npy",
        "W30 ZARR v2 Blosc zlib+bit-shuffle /bbox .npy",
        "W30 ZARR v2 Blosc memcpy /bbox .npy",
        "W30 MULTI-URL /cog/point 2-band stack",
        "W30 MULTI-URL /cog/point expr (b2-b1)/(b2+b1)",
        "W30 MULTI-URL /cog/bbox b1+b2 .npy",
        "W30 TIFF ungeoreferenced -> 400 tile & 200 validate",
        "W30 TIFF RPC-only (tag 50844) -> 400 RPC msg & 200 validate",
        "W32 ZARR v3 sharding_indexed + gzip + missing chunk .npy",
        "W32 ZARR v3 /zarr/preview cubic .png",
        "W39 ZARR f64 >1e9 /zarr/point exact",
        "W44 ZARR v3 transpose [1,0] Fortran /bbox .npy & /point",
        "W44 ZARR v3 crc32c standalone codec /bbox .npy & /point",
        "W44 ZARR v3 crc32c corrupt checksum -> 502",
        "W44 ZARR v3 gzip codec /bbox .npy",
        "W44 ZARR POST /statistics 2-feature FeatureCollection",
        "W44 ZARR POST /statistics coord_crs=EPSG:3857",
        "W50 ZARR v3 4-stage (transpose+big-endian+gzip+crc32c) .npy",
        "W52 ZARR v2 shuffle filter + zlib compressor /bbox .npy",
        "W54 ZARR v3 blosc (v3 spec byte-shuffle+zlib) /bbox .npy",
        "W54 ZARR GET /tiles algorithm=hillshade .png",
        "W54 ZARR POST /feature donut coord_crs=EPSG:3857 .npy",
        "W57 ZARR v3 standalone bitshuffle + gzip /bbox .npy",
        "W68 ZARR v3 zstd codec /bbox .npy",
        "W68 ZARR v2 3D sel=time=nearest/pad/backfill /point",
        "W68 ZARR v2 3D sel= conflicting methods -> 400",
    ]
    if not any(wanted(lb) for lb in labels):
        return 0

    w30_dir = tmp_dir / "w30"
    w30_dir.mkdir(parents=True, exist_ok=True)
    tf = from_bounds(-122.5, 37.5, -122.0, 38.0, 64, 64)
    yy, xx = np.mgrid[0:64, 0:64]
    grad = (xx * 3.25 + yy * 1.75 - 50.0).astype(np.float32)

    def _byte_shuffle(raw: bytes, typesize: int) -> bytes:
        n = len(raw) // typesize
        return np.frombuffer(raw, dtype=np.uint8).reshape(n, typesize).T.tobytes()

    def _bit_shuffle(raw: bytes, typesize: int) -> bytes:
        n_elems = len(raw) // typesize
        n_groups = n_elems // 8
        block_elems = n_groups * 8
        arr = np.frombuffer(raw[:block_elems * typesize], dtype=np.uint8).reshape(block_elems, typesize)
        bits = np.unpackbits(arr, axis=1, bitorder="little")  # (block_elems, typesize*8)
        packed = np.packbits(bits.T.reshape(typesize * 8, n_groups, 8), axis=2, bitorder="little")
        return packed.tobytes() + raw[block_elems * typesize:]

    def _encode_blosc1(raw: bytes, typesize: int, shuffle: int = 1, use_zlib: bool = True) -> bytes:
        # Blosc 1.x header: version=2, versionlz=1, flags, typesize, nbytes, blocksize, cbytes
        # flags: 0x01=DOSHUFFLE, 0x02=MEMCPYED, 0x04=DOBITSHUFFLE, 0x10=NON_SPLIT, (3<<5)=COMP_ZLIB
        nbytes = len(raw)
        if not use_zlib:
            flags = 0x02  # BLOSC_MEMCPYED
            hdr = struct.pack("<BBBBIII", 2, 1, flags, typesize, nbytes, nbytes, 16 + nbytes)
            return hdr + raw
        if shuffle == 1:
            data = _byte_shuffle(raw, typesize)
            flags = 0x01 | 0x10 | (3 << 5)
        elif shuffle == 2:
            data = _bit_shuffle(raw, typesize)
            flags = 0x04 | 0x10 | (3 << 5)
        else:
            data = raw
            flags = 0x10 | (3 << 5)
        comp = zlib.compress(data, 6)
        bstarts = struct.pack("<I", 20)
        blk = struct.pack("<I", len(comp)) + comp
        cbytes = 16 + 4 + len(blk)
        hdr = struct.pack("<BBBBIII", 2, 1, flags, typesize, nbytes, nbytes, cbytes)
        return hdr + bstarts + blk

    def _encode_lz4_numcodecs(raw: bytes) -> bytes:
        n = len(raw)
        out = bytearray(struct.pack("<I", n))
        if n < 15:
            out.append(n << 4)
        else:
            out.append(0xF0)
            rem = n - 15
            while rem >= 255:
                out.append(255)
                rem -= 255
            out.append(rem)
        out.extend(raw)
        return bytes(out)

    z_root = w30_dir / "w30_codecs.zarr"
    z_root.mkdir()
    (z_root / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
    lon = np.linspace(-122.5, -122.0, 64, dtype=np.float64)
    lat = np.linspace(38.0, 37.5, 64, dtype=np.float64)

    def _write_zarr_var(name: str, data: np.ndarray, chunks: tuple[int, ...], dims: list[str], comp: dict | None, enc):
        d = z_root / name
        d.mkdir()
        (d / ".zarray").write_text(json.dumps({
            "zarr_format": 2, "shape": list(data.shape), "chunks": list(chunks),
            "dtype": data.dtype.str, "compressor": comp, "fill_value": None, "order": "C", "filters": None,
        }))
        (d / ".zattrs").write_text(json.dumps({"_ARRAY_DIMENSIONS": dims}))
        key = "0" if data.ndim == 1 else "0.0"
        (d / key).write_bytes(enc(np.ascontiguousarray(data).tobytes()))

    _write_zarr_var("lon", lon, (64,), ["lon"], None, lambda b: b)
    _write_zarr_var("lat", lat, (64,), ["lat"], None, lambda b: b)
    _write_zarr_var("v_gzip", grad, (64, 64), ["lat", "lon"], {"id": "gzip", "level": 6}, lambda b: gzip.compress(b))
    _write_zarr_var("v_lz4", grad, (64, 64), ["lat", "lon"], {"id": "lz4", "acceleration": 1}, _encode_lz4_numcodecs)
    _write_zarr_var("v_blosc_shuf", grad, (64, 64), ["lat", "lon"], {"id": "blosc", "cname": "zlib", "clevel": 6, "shuffle": 1}, lambda b: _encode_blosc1(b, 4, shuffle=1, use_zlib=True))
    _write_zarr_var("v_blosc_bitshuf", grad, (64, 64), ["lat", "lon"], {"id": "blosc", "cname": "zlib", "clevel": 6, "shuffle": 2}, lambda b: _encode_blosc1(b, 4, shuffle=2, use_zlib=True))
    _write_zarr_var("v_blosc_memcpy", grad, (64, 64), ["lat", "lon"], {"id": "blosc", "cname": "lz4", "clevel": 5, "shuffle": 0}, lambda b: _encode_blosc1(b, 4, shuffle=0, use_zlib=False))

    for lb, var in [
        ("W30 ZARR v2 Gzip /bbox .npy", "v_gzip"),
        ("W30 ZARR v2 LZ4 raw block /bbox .npy", "v_lz4"),
        ("W30 ZARR v2 Blosc zlib+byte-shuffle /bbox .npy", "v_blosc_shuf"),
        ("W30 ZARR v2 Blosc zlib+bit-shuffle /bbox .npy", "v_blosc_bitshuf"),
        ("W30 ZARR v2 Blosc memcpy /bbox .npy", "v_blosc_memcpy"),
    ]:
        if wanted(lb):
            p = f"/zarr/bbox/-122.5,37.5,-122.0,38.0/64x64.npy?url={q(z_root)}&variable={var}"
            r = client.get(cw_base + p)
            if r.status_code == 200:
                arr = np.load(io.BytesIO(r.content))
                err = float(np.max(np.abs(arr[0] - grad)))
                rec(lb, arr.shape == (2, 64, 64) and err < 1e-4, f"max_err={err:.5f}", p)
            else:
                rec(lb, False, f"HTTP {r.status_code} {r.text[:100]!r}", p)

    # Multi-URL COG fixtures
    p1 = w30_dir / "b1.tif"
    p2 = w30_dir / "b2.tif"
    b1_arr = (xx.astype(np.float32) + 10.0)
    b2_arr = (yy.astype(np.float32) * 2.0 + 20.0)
    with rasterio.open(p1, "w", driver="GTiff", width=64, height=64, count=1, dtype="float32", crs="EPSG:4326", transform=tf, tiled=True, blockxsize=64, blockysize=64, compress="deflate") as ds:
        ds.write(b1_arr, 1)
    with rasterio.open(p2, "w", driver="GTiff", width=64, height=64, count=1, dtype="float32", crs="EPSG:4326", transform=tf, tiled=True, blockxsize=64, blockysize=64, compress="deflate") as ds:
        ds.write(b2_arr, 1)

    lb = "W30 MULTI-URL /cog/point 2-band stack"
    if wanted(lb):
        p = f"/cog/point/-122.25,37.75?url={q(p1)}&url={q(p2)}"
        r = client.get(cw_base + p)
        if r.status_code == 200:
            j = r.json()
            vals = j.get("values", [])
            ok = j.get("band_names") == ["b1", "b2"] and len(vals) == 2 and abs(vals[0] - 42.0) < 0.01 and abs(vals[1] - 84.0) < 0.01
            rec(lb, ok, f"names={j.get('band_names')} values={vals}", p)
        else:
            rec(lb, False, f"HTTP {r.status_code} {r.text[:100]!r}", p)

    lb = "W30 MULTI-URL /cog/point expr (b2-b1)/(b2+b1)"
    if wanted(lb):
        p = f"/cog/point/-122.25,37.75?url={q(p1)}&url={q(p2)}&expression={qe('(b2-b1)/(b2+b1)')}"
        r = client.get(cw_base + p)
        if r.status_code == 200:
            j = r.json()
            vals = j.get("values", [])
            exp_ndvi = (84.0 - 42.0) / (84.0 + 42.0)  # 1/3
            ok = len(vals) == 1 and abs(vals[0] - exp_ndvi) < 1e-4
            rec(lb, ok, f"values={vals} exp={exp_ndvi:.5f}", p)
        else:
            rec(lb, False, f"HTTP {r.status_code} {r.text[:100]!r}", p)

    lb = "W30 MULTI-URL /cog/bbox b1+b2 .npy"
    if wanted(lb):
        p = f"/cog/bbox/-122.5,37.5,-122.0,38.0/64x64.npy?url={q(p1)}&url={q(p2)}&expression={qe('b1+b2')}"
        r = client.get(cw_base + p)
        if r.status_code == 200:
            arr = np.load(io.BytesIO(r.content))
            err = float(np.max(np.abs(arr[0] - (b1_arr + b2_arr))))
            rec(lb, arr.shape == (2, 64, 64) and err < 1e-4, f"max_err={err:.5f}", p)
            if gpu_base:
                rg = client.get(gpu_base + p)
                if rg.status_code != 200:
                    gpu_diffs.append({"label": lb, "detail": f"GPU HTTP {rg.status_code}"})
                else:
                    ok_g, det_g = compare_npy(r.content, rg.content, check_values=True)
                    if not ok_g:
                        gpu_diffs.append({"label": lb, "detail": det_g})
        else:
            rec(lb, False, f"HTTP {r.status_code} {r.text[:100]!r}", p)

    # Ungeoreferenced & RPC-only TIFF diagnostics
    p_nogeo = w30_dir / "nogeo.tif"
    with rasterio.open(p_nogeo, "w", driver="GTiff", width=64, height=64, count=1, dtype="uint8", tiled=True, blockxsize=64, blockysize=64) as ds:
        ds.write((xx % 256).astype(np.uint8), 1)
    p_rpc = w30_dir / "rpc_only.tif"
    with rasterio.open(p_rpc, "w", driver="GTiff", width=64, height=64, count=1, dtype="uint8", tiled=True, blockxsize=64, blockysize=64) as ds:
        ds.write((xx % 256).astype(np.uint8), 1)
        ds.update_tags(
            ns="RPC", ERR_BIAS="0", ERR_RAND="0", LINE_OFF="32", SAMP_OFF="32",
            LAT_OFF="37.75", LONG_OFF="-122.25", HEIGHT_OFF="0", LINE_SCALE="32",
            SAMP_SCALE="32", LAT_SCALE="0.25", LONG_SCALE="0.25", HEIGHT_SCALE="100",
            LINE_NUM_COEFF="0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0",
            LINE_DEN_COEFF="1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0",
            SAMP_NUM_COEFF="0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0",
            SAMP_DEN_COEFF="1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0",
        )

    lb = "W30 TIFF ungeoreferenced -> 400 tile & 200 validate"
    if wanted(lb):
        pt = f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(p_nogeo)}"
        pv = f"/cog/validate?url={q(p_nogeo)}"
        rt, rv = client.get(cw_base + pt), client.get(cw_base + pv)
        ok = rt.status_code == 400 and "no georeferencing" in rt.text.lower() and rv.status_code == 200
        rec(lb, ok, f"tile={rt.status_code} validate={rv.status_code}", pt)

    lb = "W30 TIFF RPC-only (tag 50844) -> 400 RPC msg & 200 validate"
    if wanted(lb):
        pt = f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(p_rpc)}"
        pv = f"/cog/validate?url={q(p_rpc)}"
        rt, rv = client.get(cw_base + pt), client.get(cw_base + pv)
        ok = rt.status_code == 400 and "rpc" in rt.text.lower() and rv.status_code == 200
        rec(lb, ok, f"tile={rt.status_code} validate={rv.status_code} msg={rt.text[:70]!r}", pt)

    # --- Wave 32: Zarr v3 sharding_indexed with gzip inner codec + missing inner chunk + cubic /zarr/preview ---
    if wanted("W32 ZARR v3 sharding_indexed + gzip + missing chunk .npy") or wanted("W32 ZARR v3 /zarr/preview cubic .png"):
        z3 = w30_dir / "w32_shard_gzip.zarr"
        (z3 / "v" / "c" / "0").mkdir(parents=True)
        (z3 / "zarr.json").write_text(json.dumps({"zarr_format": 3, "node_type": "group", "attributes": {}}))
        for cname, arr_c in [("lat", lat), ("lon", lon)]:
            (z3 / cname / "c").mkdir(parents=True)
            (z3 / cname / "zarr.json").write_text(json.dumps({
                "zarr_format": 3, "node_type": "array", "shape": [64], "data_type": "float64",
                "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [64]}},
                "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
                "fill_value": 0.0, "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
                "dimension_names": [cname], "attributes": {},
            }))
            (z3 / cname / "c" / "0").write_bytes(arr_c.tobytes())
        g64 = (10.0 + xx * 2.0 + yy * 3.0).astype(np.float32)
        sbuf = bytearray()
        sidx = []
        for iy in range(2):
            for ix in range(2):
                if iy == 1 and ix == 1:
                    sidx.append((0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF))
                else:
                    comp = gzip.compress(np.ascontiguousarray(g64[iy * 32:(iy + 1) * 32, ix * 32:(ix + 1) * 32]).tobytes())
                    sidx.append((len(sbuf), len(comp)))
                    sbuf.extend(comp)
        idx_b = b"".join(struct.pack("<QQ", o, l) for o, l in sidx)
        def _crc32c(data: bytes) -> int:
            crc = 0xFFFFFFFF
            for b in data:
                crc ^= b
                for _ in range(8):
                    crc = (crc >> 1) ^ 0x82F63B78 if (crc & 1) else (crc >> 1)
            return crc ^ 0xFFFFFFFF
        (z3 / "v" / "c" / "0" / "0").write_bytes(bytes(sbuf) + idx_b + struct.pack("<I", _crc32c(idx_b)))
        (z3 / "v" / "zarr.json").write_text(json.dumps({
            "zarr_format": 3, "node_type": "array", "shape": [64, 64], "data_type": "float32",
            "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [64, 64]}},
            "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
            "fill_value": -9999.0,
            "codecs": [{
                "name": "sharding_indexed",
                "configuration": {
                    "chunk_shape": [32, 32],
                    "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}, {"name": "gzip", "configuration": {"level": 5}}],
                    "index_codecs": [{"name": "bytes", "configuration": {"endian": "little"}}, {"name": "crc32c"}],
                    "index_location": "end",
                },
            }],
            "dimension_names": ["lat", "lon"], "attributes": {},
        }))

        lb = "W32 ZARR v3 sharding_indexed + gzip + missing chunk .npy"
        if wanted(lb):
            p = f"/zarr/bbox/-122.5,37.5,-122.0,38.0/64x64.npy?url={q(z3)}&variable=v"
            r = client.get(cw_base + p)
            if r.status_code == 200:
                arr = np.load(io.BytesIO(r.content))
                miss_masked = int((arr[1, 32:, 32:] == np.finfo(np.float32).min).sum())
                valid_err = float(np.max(np.abs(arr[0, :32, :32] - g64[:32, :32])))
                rec(lb, arr.shape == (2, 64, 64) and valid_err < 1e-4 and miss_masked == 1024, f"valid_err={valid_err:.5f} missing_masked={miss_masked}/1024", p)
            else:
                rec(lb, False, f"HTTP {r.status_code} {r.text[:100]!r}", p)

        lb = "W32 ZARR v3 /zarr/preview cubic .png"
        if wanted(lb):
            p = f"/zarr/preview/64x64.png?url={q(z3)}&variable=v&resampling=cubic&rescale=10,330&colormap_name=viridis"
            r = client.get(cw_base + p)
            if r.status_code == 200:
                im = np.array(Image.open(io.BytesIO(r.content)).convert("RGBA"))
                ok = im.shape == (64, 64, 4) and int(im[48, 48, 3]) == 0 and int(im[16, 16, 3]) == 255
                rec(lb, ok, f"shape={im.shape} miss_a={im[48,48,3]} valid_a={im[16,16,3]}", p)
                if gpu_base:
                    rg = client.get(gpu_base + p)
                    if rg.status_code != 200:
                        gpu_diffs.append({"label": lb, "detail": f"GPU HTTP {rg.status_code}"})
                    else:
                        ok_g, det_g = compare_images(r.content, rg.content)
                        if not ok_g:
                            gpu_diffs.append({"label": lb, "detail": det_g})
            else:
                rec(lb, False, f"HTTP {r.status_code} {r.text[:100]!r}", p)

    # --- Wave 39: High-precision float64 Zarr /point (1e9 + 0.001*x + 0.003*y, where f32 ULP=64) ---
    lb = "W39 ZARR f64 >1e9 /zarr/point exact"
    if wanted(lb):
        zf64 = w30_dir / "w39_f64_hi.zarr"
        zf64.mkdir()
        (zf64 / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
        zf64_arr = (1_000_000_000.0 + xx.astype(np.float64) * 0.001 + yy.astype(np.float64) * 0.003)
        for vname, varr, vdims in [("lon", lon, ["lon"]), ("lat", lat, ["lat"]), ("val", zf64_arr, ["lat", "lon"])]:
            vd = zf64 / vname
            vd.mkdir()
            (vd / ".zarray").write_text(json.dumps({
                "zarr_format": 2, "shape": list(varr.shape), "chunks": list(varr.shape),
                "dtype": "<f8", "compressor": None, "fill_value": None, "order": "C", "filters": None,
            }))
            (vd / ".zattrs").write_text(json.dumps({"_ARRAY_DIMENSIONS": vdims}))
            (vd / ("0" if varr.ndim == 1 else "0.0")).write_bytes(np.ascontiguousarray(varr).tobytes())
        p = f"/zarr/point/-122.25,37.75?url={q(zf64)}&variable=val"
        r = client.get(cw_base + p)
        if r.status_code == 200:
            vals = r.json().get("values", [])
            ok = len(vals) == 1 and abs(vals[0] - 1_000_000_000.128) < 1e-3
            rec(lb, ok, f"values={vals} (exp ~1000000000.128)", p)
            if gpu_base:
                rg = client.get(gpu_base + p)
                if rg.status_code != 200:
                    gpu_diffs.append({"label": lb, "detail": f"GPU HTTP {rg.status_code}"})
                else:
                    gvals = rg.json().get("values", [])
                    if len(gvals) != 1 or abs(vals[0] - gvals[0]) >= 1e-4:
                        gpu_diffs.append({"label": lb, "detail": f"cpu={vals} gpu={gvals}"})
        else:
            rec(lb, False, f"HTTP {r.status_code} {r.text[:100]!r}", p)

    # --- Wave 44: Zarr v3 transpose [1,0], standalone crc32c (+ corrupt rejection), gzip, POST /zarr/statistics FC + coord_crs=EPSG:3857 ---
    w44_labels = [
        "W44 ZARR v3 transpose [1,0] Fortran /bbox .npy & /point",
        "W44 ZARR v3 crc32c standalone codec /bbox .npy & /point",
        "W44 ZARR v3 crc32c corrupt checksum -> 502",
        "W44 ZARR v3 gzip codec /bbox .npy",
        "W44 ZARR POST /statistics 2-feature FeatureCollection",
        "W44 ZARR POST /statistics coord_crs=EPSG:3857",
        "W50 ZARR v3 4-stage (transpose+big-endian+gzip+crc32c) .npy",
        "W52 ZARR v2 shuffle filter + zlib compressor /bbox .npy",
        "W54 ZARR v3 blosc (v3 spec byte-shuffle+zlib) /bbox .npy",
        "W54 ZARR GET /tiles algorithm=hillshade .png",
        "W54 ZARR POST /feature donut coord_crs=EPSG:3857 .npy",
        "W57 ZARR v3 standalone bitshuffle + gzip /bbox .npy",
        "W68 ZARR v3 zstd codec /bbox .npy",
        "W68 ZARR v2 3D sel=time=nearest/pad/backfill /point",
        "W68 ZARR v2 3D sel= conflicting methods -> 400",
    ]
    if any(wanted(l) for l in w44_labels):
        def _crc32c_w44(data: bytes) -> int:
            poly = 0x82F63B78
            crc = 0xFFFFFFFF
            for b_ in data:
                crc ^= b_
                for _ in range(8):
                    crc = (crc >> 1) ^ poly if (crc & 1) else (crc >> 1)
            return crc ^ 0xFFFFFFFF

        h44, w44 = 64, 64
        yy44, xx44 = np.mgrid[0:h44, 0:w44].astype(np.float32)
        arr44 = (10.0 + xx44 * 10.0 + yy44 * 0.5).astype(np.float32)
        raw44 = np.ascontiguousarray(arr44).tobytes()
        xs44 = np.linspace(-122.5 + 0.5 / (2 * w44), -122.0 - 0.5 / (2 * w44), w44, dtype=np.float64)
        ys44 = np.linspace(38.0 - 0.5 / (2 * h44), 37.5 + 0.5 / (2 * h44), h44, dtype=np.float64)

        def _mk_z3_w44(name: str, codecs_cfg: list, chunk_bytes: bytes) -> Path:
            root = w30_dir / name
            root.mkdir(parents=True, exist_ok=True)
            (root / "zarr.json").write_text(json.dumps({
                "zarr_format": 3, "node_type": "group",
                "attributes": {"multiscales": [{"version": "0.4", "datasets": [{"path": "data"}]}]},
            }))
            dd = root / "data"
            dd.mkdir()
            (dd / "zarr.json").write_text(json.dumps({
                "zarr_format": 3, "node_type": "array",
                "shape": [h44, w44], "data_type": "float32",
                "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [h44, w44]}},
                "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
                "fill_value": -9999.0, "codecs": codecs_cfg,
                "dimension_names": ["y", "x"], "attributes": {"_ARRAY_DIMENSIONS": ["y", "x"]},
            }))
            (dd / "c" / "0").mkdir(parents=True)
            (dd / "c" / "0" / "0").write_bytes(chunk_bytes)
            for cname, cvals in (("x", xs44), ("y", ys44)):
                cd = root / cname
                cd.mkdir()
                (cd / "zarr.json").write_text(json.dumps({
                    "zarr_format": 3, "node_type": "array",
                    "shape": [len(cvals)], "data_type": "float64",
                    "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [len(cvals)]}},
                    "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
                    "fill_value": "NaN",
                    "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
                    "dimension_names": [cname], "attributes": {"_ARRAY_DIMENSIONS": [cname]},
                }))
                (cd / "c").mkdir()
                (cd / "c" / "0").write_bytes(cvals.tobytes())
            return root

        lon44 = -122.5 + (48 + 0.5) * 0.5 / 64
        lat44 = 38.0 - (16 + 0.5) * 0.5 / 64

        lb = "W44 ZARR v3 transpose [1,0] Fortran /bbox .npy & /point"
        if wanted(lb):
            z_tr = _mk_z3_w44(
                "w44_z3_transpose.zarr",
                [{"name": "transpose", "configuration": {"order": [1, 0]}}, {"name": "bytes", "configuration": {"endian": "little"}}],
                np.asfortranarray(arr44).tobytes(order="F"),
            )
            p_bb = f"/zarr/bbox/-122.5,37.5,-122.0,38.0/64x64.npy?url={q(z_tr)}&variable=data"
            p_pt = f"/zarr/point/{lon44},{lat44}?url={q(z_tr)}&variable=data"
            rb, rp = client.get(cw_base + p_bb), client.get(cw_base + p_pt)
            if rb.status_code == 200 and rp.status_code == 200:
                ab = np.load(io.BytesIO(rb.content))
                err = float(np.abs(ab[0] - arr44).max())
                vals = rp.json().get("values", [])
                ok = err < 1e-4 and len(vals) == 1 and abs(vals[0] - 498.0) < 1e-3
                rec(lb, ok, f"max_err={err:.5f} pt={vals} (exp 498.0)", p_bb)
                if gpu_base:
                    rbg = client.get(gpu_base + p_bb)
                    if rbg.status_code != 200:
                        gpu_diffs.append({"label": lb, "detail": f"GPU HTTP {rbg.status_code}"})
                    else:
                        ok_g, det_g = compare_npy(rb.content, rbg.content, check_values=True)
                        if not ok_g:
                            gpu_diffs.append({"label": lb, "detail": det_g})
            else:
                rec(lb, False, f"HTTP bbox={rb.status_code} pt={rp.status_code}", p_bb)

        lb = "W44 ZARR v3 crc32c standalone codec /bbox .npy & /point"
        if wanted(lb):
            z_crc = _mk_z3_w44(
                "w44_z3_crc32c.zarr",
                [{"name": "bytes", "configuration": {"endian": "little"}}, {"name": "crc32c", "configuration": {}}],
                raw44 + struct.pack("<I", _crc32c_w44(raw44)),
            )
            p_bb = f"/zarr/bbox/-122.5,37.5,-122.0,38.0/64x64.npy?url={q(z_crc)}&variable=data"
            p_pt = f"/zarr/point/{lon44},{lat44}?url={q(z_crc)}&variable=data"
            rb, rp = client.get(cw_base + p_bb), client.get(cw_base + p_pt)
            if rb.status_code == 200 and rp.status_code == 200:
                ab = np.load(io.BytesIO(rb.content))
                err = float(np.abs(ab[0] - arr44).max())
                vals = rp.json().get("values", [])
                ok = err < 1e-4 and len(vals) == 1 and abs(vals[0] - 498.0) < 1e-3
                rec(lb, ok, f"max_err={err:.5f} pt={vals} (exp 498.0)", p_bb)
                if gpu_base:
                    rbg = client.get(gpu_base + p_bb)
                    if rbg.status_code != 200:
                        gpu_diffs.append({"label": lb, "detail": f"GPU HTTP {rbg.status_code}"})
                    else:
                        ok_g, det_g = compare_npy(rb.content, rbg.content, check_values=True)
                        if not ok_g:
                            gpu_diffs.append({"label": lb, "detail": det_g})
            else:
                rec(lb, False, f"HTTP bbox={rb.status_code} pt={rp.status_code}", p_bb)

        lb = "W44 ZARR v3 crc32c corrupt checksum -> 502"
        if wanted(lb):
            bad_crc = _crc32c_w44(raw44) ^ 0xDEAD0001
            z_bad = _mk_z3_w44(
                "w44_z3_crc32c_bad.zarr",
                [{"name": "bytes", "configuration": {"endian": "little"}}, {"name": "crc32c", "configuration": {}}],
                raw44 + struct.pack("<I", bad_crc),
            )
            p_bb = f"/zarr/bbox/-122.5,37.5,-122.0,38.0/64x64.npy?url={q(z_bad)}&variable=data"
            rb = client.get(cw_base + p_bb)
            ok = rb.status_code == 502 and "crc32c" in rb.text.lower()
            rec(lb, ok, f"status={rb.status_code} msg={rb.text[:70]!r}", p_bb)

        z_gz44 = None
        if any(wanted(l) for l in (
            "W44 ZARR v3 gzip codec /bbox .npy",
            "W44 ZARR POST /statistics 2-feature FeatureCollection",
            "W44 ZARR POST /statistics coord_crs=EPSG:3857",
            "W54 ZARR GET /tiles algorithm=hillshade .png",
            "W54 ZARR POST /feature donut coord_crs=EPSG:3857 .npy",
        )):
            z_gz44 = _mk_z3_w44(
                "w44_z3_gzip.zarr",
                [{"name": "bytes", "configuration": {"endian": "little"}}, {"name": "gzip", "configuration": {"level": 5}}],
                zlib.compress(raw44, level=5, wbits=31),
            )

        lb = "W44 ZARR v3 gzip codec /bbox .npy"
        if wanted(lb):
            p_bb = f"/zarr/bbox/-122.5,37.5,-122.0,38.0/64x64.npy?url={q(z_gz44)}&variable=data"
            rb = client.get(cw_base + p_bb)
            if rb.status_code == 200:
                ab = np.load(io.BytesIO(rb.content))
                err = float(np.abs(ab[0] - arr44).max())
                rec(lb, err < 1e-4, f"max_err={err:.5f}", p_bb)
                if gpu_base:
                    rbg = client.get(gpu_base + p_bb)
                    if rbg.status_code != 200:
                        gpu_diffs.append({"label": lb, "detail": f"GPU HTTP {rbg.status_code}"})
                    else:
                        ok_g, det_g = compare_npy(rb.content, rbg.content, check_values=True)
                        if not ok_g:
                            gpu_diffs.append({"label": lb, "detail": det_g})
            else:
                rec(lb, False, f"HTTP {rb.status_code} {rb.text[:80]!r}", p_bb)

        west_mean_ref = None
        lb = "W44 ZARR POST /statistics 2-feature FeatureCollection"
        if wanted(lb):
            fc44 = {
                "type": "FeatureCollection",
                "features": [
                    {"type": "Feature", "properties": {"id": "west"}, "geometry": {"type": "Polygon", "coordinates": [[[-122.45, 37.6], [-122.3, 37.6], [-122.3, 37.9], [-122.45, 37.9], [-122.45, 37.6]]]}},
                    {"type": "Feature", "properties": {"id": "east"}, "geometry": {"type": "Polygon", "coordinates": [[[-122.2, 37.6], [-122.05, 37.6], [-122.05, 37.9], [-122.2, 37.9], [-122.2, 37.6]]]}},
                ],
            }
            p_fc = f"/zarr/statistics?url={q(z_gz44)}&variable=data"
            r = client.post(cw_base + p_fc, json=fc44)
            if r.status_code == 200:
                feats = r.json().get("features", [])
                m_w = feats[0]["properties"]["statistics"]["b1"]["mean"] if len(feats) == 2 else 0.0
                m_e = feats[1]["properties"]["statistics"]["b1"]["mean"] if len(feats) == 2 else 0.0
                west_mean_ref = m_w
                ok = len(feats) == 2 and 120.0 < m_w < 220.0 and 450.0 < m_e < 560.0 and (m_e - m_w) > 250.0
                rec(lb, ok, f"west_mean={m_w:.2f} east_mean={m_e:.2f}", p_fc)
                if gpu_base:
                    rg = client.post(gpu_base + p_fc, json=fc44)
                    if rg.status_code != 200:
                        gpu_diffs.append({"label": lb, "detail": f"GPU HTTP {rg.status_code}"})
                    else:
                        d_fc = first_json_diff(r.json(), rg.json())
                        if d_fc is not None:
                            gpu_diffs.append({"label": lb, "detail": d_fc})
            else:
                rec(lb, False, f"HTTP {r.status_code} {r.text[:80]!r}", p_fc)

        lb = "W44 ZARR POST /statistics coord_crs=EPSG:3857"
        if wanted(lb):
            x3857, y3857 = rasterio.warp.transform("EPSG:4326", "EPSG:3857", [-122.45, -122.3, -122.3, -122.45, -122.45], [37.6, 37.6, 37.9, 37.9, 37.6])
            f3857 = {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [list(zip(x3857, y3857))]}}
            p_38 = f"/zarr/statistics?url={q(z_gz44)}&variable=data&coord_crs=EPSG:3857"
            r = client.post(cw_base + p_38, json=f3857)
            if r.status_code == 200:
                m38 = r.json()["properties"]["statistics"]["b1"]["mean"]
                ok = 120.0 < m38 < 220.0 and (west_mean_ref is None or abs(m38 - west_mean_ref) < 5.0)
                rec(lb, ok, f"mean_3857={m38:.2f}" + (f" (west_4326={west_mean_ref:.2f})" if west_mean_ref is not None else ""), p_38)
                if gpu_base:
                    rg = client.post(gpu_base + p_38, json=f3857)
                    if rg.status_code != 200:
                        gpu_diffs.append({"label": lb, "detail": f"GPU HTTP {rg.status_code}"})
                    else:
                        d_38 = first_json_diff(r.json(), rg.json())
                        if d_38 is not None:
                            gpu_diffs.append({"label": lb, "detail": d_38})
            else:
                rec(lb, False, f"HTTP {r.status_code} {r.text[:80]!r}", p_38)

        lb = "W50 ZARR v3 4-stage (transpose+big-endian+gzip+crc32c) .npy"
        if wanted(lb):
            f_be = np.asfortranarray(arr44).astype(">f4").tobytes(order="F")
            gz_be = zlib.compress(f_be, level=5, wbits=31)
            z_4s = _mk_z3_w44(
                "w50_z3_4stage.zarr",
                [
                    {"name": "transpose", "configuration": {"order": [1, 0]}},
                    {"name": "bytes", "configuration": {"endian": "big"}},
                    {"name": "gzip", "configuration": {"level": 5}},
                    {"name": "crc32c", "configuration": {}},
                ],
                gz_be + struct.pack("<I", _crc32c_w44(gz_be)),
            )
            p_bb = f"/zarr/bbox/-122.5,37.5,-122.0,38.0/64x64.npy?url={q(z_4s)}&variable=data"
            rb = client.get(cw_base + p_bb)
            if rb.status_code == 200:
                ab = np.load(io.BytesIO(rb.content))
                err = float(np.abs(ab[0] - arr44).max())
                rec(lb, err < 1e-4, f"max_err={err:.5f}", p_bb)
                if gpu_base:
                    rbg = client.get(gpu_base + p_bb)
                    if rbg.status_code != 200:
                        gpu_diffs.append({"label": lb, "detail": f"GPU HTTP {rbg.status_code}"})
                    else:
                        ok_g, det_g = compare_npy(rb.content, rbg.content, check_values=True)
                        if not ok_g:
                            gpu_diffs.append({"label": lb, "detail": det_g})
            else:
                rec(lb, False, f"HTTP {rb.status_code} {rb.text[:80]!r}", p_bb)

        lb = "W52 ZARR v2 shuffle filter + zlib compressor /bbox .npy"
        if wanted(lb):
            z2_sh = w30_dir / "w52_z2_shuf_zlib.zarr"
            z2_sh.mkdir(parents=True, exist_ok=True)
            (z2_sh / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
            for vn, va, vd in [("lon", xs44, ["lon"]), ("lat", ys44, ["lat"])]:
                d_ = z2_sh / vn
                d_.mkdir()
                (d_ / ".zarray").write_text(json.dumps({
                    "zarr_format": 2, "shape": [len(va)], "chunks": [len(va)], "dtype": "<f8",
                    "compressor": None, "fill_value": None, "order": "C", "filters": None,
                }))
                (d_ / ".zattrs").write_text(json.dumps({"_ARRAY_DIMENSIONS": vd}))
                (d_ / "0").write_bytes(va.tobytes())
            dv = z2_sh / "val"
            dv.mkdir()
            (dv / ".zarray").write_text(json.dumps({
                "zarr_format": 2, "shape": [h44, w44], "chunks": [h44, w44], "dtype": "<f4",
                "compressor": {"id": "zlib", "level": 5},
                "filters": [{"id": "shuffle", "elementsize": 4}],
                "fill_value": -9999.0, "order": "C",
            }))
            (dv / ".zattrs").write_text(json.dumps({"_ARRAY_DIMENSIONS": ["lat", "lon"]}))
            shuf = np.frombuffer(raw44, dtype=np.uint8).reshape(-1, 4).T.tobytes()
            (dv / "0.0").write_bytes(zlib.compress(shuf, level=5))
            p_bb = f"/zarr/bbox/-122.5,37.5,-122.0,38.0/64x64.npy?url={q(z2_sh)}&variable=val"
            rb = client.get(cw_base + p_bb)
            if rb.status_code == 200:
                ab = np.load(io.BytesIO(rb.content))
                err = float(np.abs(ab[0] - arr44).max())
                rec(lb, err < 1e-4, f"max_err={err:.5f}", p_bb)
                if gpu_base:
                    rbg = client.get(gpu_base + p_bb)
                    if rbg.status_code != 200:
                        gpu_diffs.append({"label": lb, "detail": f"GPU HTTP {rbg.status_code}"})
                    else:
                        ok_g, det_g = compare_npy(rb.content, rbg.content, check_values=True)
                        if not ok_g:
                            gpu_diffs.append({"label": lb, "detail": det_g})
            else:
                rec(lb, False, f"HTTP {rb.status_code} {rb.text[:80]!r}", p_bb)

        lb = "W54 ZARR v3 blosc (v3 spec byte-shuffle+zlib) /bbox .npy"
        if wanted(lb):
            z_blosc3 = _mk_z3_w44(
                "w54_z3_blosc.zarr",
                [
                    {"name": "bytes", "configuration": {"endian": "little"}},
                    {"name": "blosc", "configuration": {"cname": "zlib", "clevel": 5, "shuffle": "shuffle", "typesize": 4, "blocksize": 0}},
                ],
                _encode_blosc1(raw44, 4, shuffle=1, use_zlib=True),
            )
            p_bb = f"/zarr/bbox/-122.5,37.5,-122.0,38.0/64x64.npy?url={q(z_blosc3)}&variable=data"
            rb = client.get(cw_base + p_bb)
            if rb.status_code == 200:
                ab = np.load(io.BytesIO(rb.content))
                err = float(np.abs(ab[0] - arr44).max())
                rec(lb, err < 1e-4, f"max_err={err:.5f}", p_bb)
                if gpu_base:
                    rbg = client.get(gpu_base + p_bb)
                    if rbg.status_code != 200:
                        gpu_diffs.append({"label": lb, "detail": f"GPU HTTP {rbg.status_code}"})
                    else:
                        ok_g, det_g = compare_npy(rb.content, rbg.content, check_values=True)
                        if not ok_g:
                            gpu_diffs.append({"label": lb, "detail": det_g})
            else:
                rec(lb, False, f"HTTP {rb.status_code} {rb.text[:80]!r}", p_bb)

        lb = "W54 ZARR GET /tiles algorithm=hillshade .png"
        if wanted(lb):
            p_hs = f"/zarr/tiles/WebMercatorQuad/10/164/396.png?url={q(z_gz44)}&variable=data&algorithm=hillshade"
            r_hs = client.get(cw_base + p_hs)
            if r_hs.status_code == 200:
                im = np.array(Image.open(io.BytesIO(r_hs.content)).convert("RGBA"))
                vm = im[..., 3] == 255
                ok = im.shape == (256, 256, 4) and int(vm.sum()) > 10000 and int(im[vm, 0].max()) > 100
                rec(lb, ok, f"shape={im.shape} valid={int(vm.sum())} max_r={int(im[vm, 0].max()) if vm.any() else 0}", p_hs)
                if gpu_base:
                    rg = client.get(gpu_base + p_hs)
                    if rg.status_code != 200:
                        gpu_diffs.append({"label": lb, "detail": f"GPU HTTP {rg.status_code}"})
                    else:
                        ok_g, det_g = compare_images(r_hs.content, rg.content)
                        if not ok_g:
                            gpu_diffs.append({"label": lb, "detail": det_g})
            else:
                rec(lb, False, f"HTTP {r_hs.status_code} {r_hs.text[:80]!r}", p_hs)

        lb = "W54 ZARR POST /feature donut coord_crs=EPSG:3857 .npy"
        if wanted(lb):
            ox, oy = rasterio.warp.transform("EPSG:4326", "EPSG:3857", [-122.45, -122.05, -122.05, -122.45, -122.45], [37.55, 37.55, 37.95, 37.95, 37.55])
            ix, iy = rasterio.warp.transform("EPSG:4326", "EPSG:3857", [-122.33, -122.33, -122.17, -122.17, -122.33], [37.67, 37.83, 37.83, 37.67, 37.67])
            donut_z38 = {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [list(zip(ox, oy)), list(zip(ix, iy))]}}
            p_ft = f"/zarr/feature/64x64.npy?url={q(z_gz44)}&variable=data&coord_crs=EPSG:3857"
            r_ft = client.post(cw_base + p_ft, json=donut_z38)
            if r_ft.status_code == 200:
                af = np.load(io.BytesIO(r_ft.content))
                fmin = np.finfo(af.dtype).min
                center_masked = bool(af[1, 32, 32] == fmin)
                ring_valid = bool(af[1, 8, 32] > fmin)
                nvalid = int((af[1] > fmin).sum())
                ok = af.shape == (2, 64, 64) and center_masked and ring_valid and 2000 < nvalid < 3600
                rec(lb, ok, f"shape={af.shape} center_masked={center_masked} ring_valid={ring_valid} valid={nvalid}", p_ft)
                if gpu_base:
                    rg = client.post(gpu_base + p_ft, json=donut_z38)
                    if rg.status_code != 200:
                        gpu_diffs.append({"label": lb, "detail": f"GPU HTTP {rg.status_code}"})
                    else:
                        ok_g, det_g = compare_npy(r_ft.content, rg.content, check_values=True)
                        if not ok_g:
                            gpu_diffs.append({"label": lb, "detail": det_g})
            else:
                rec(lb, False, f"HTTP {r_ft.status_code} {r_ft.text[:80]!r}", p_ft)

        lb = "W57 ZARR v3 standalone bitshuffle + gzip /bbox .npy"
        if wanted(lb):
            bs_gz = zlib.compress(_bit_shuffle(raw44, 4), level=5, wbits=31)
            z_bs3 = _mk_z3_w44(
                "w57_z3_bitshuffle_gzip.zarr",
                [
                    {"name": "bytes", "configuration": {"endian": "little"}},
                    {"name": "bitshuffle", "configuration": {"elementsize": 4}},
                    {"name": "gzip", "configuration": {"level": 5}},
                ],
                bs_gz,
            )
            p_bb = f"/zarr/bbox/-122.5,37.5,-122.0,38.0/64x64.npy?url={q(z_bs3)}&variable=data"
            rb = client.get(cw_base + p_bb)
            if rb.status_code == 200:
                ab = np.load(io.BytesIO(rb.content))
                err = float(np.abs(ab[0] - arr44).max())
                rec(lb, err < 1e-4, f"max_err={err:.5f}", p_bb)
                if gpu_base:
                    rbg = client.get(gpu_base + p_bb)
                    if rbg.status_code != 200:
                        gpu_diffs.append({"label": lb, "detail": f"GPU HTTP {rbg.status_code}"})
                    else:
                        ok_g, det_g = compare_npy(rb.content, rbg.content, check_values=True)
                        if not ok_g:
                            gpu_diffs.append({"label": lb, "detail": det_g})
            else:
                rec(lb, False, f"HTTP {rb.status_code} {rb.text[:80]!r}", p_bb)

        lb = "W68 ZARR v3 zstd codec /bbox .npy"
        if wanted(lb):
            import shutil as _shutil
            zstd_bin = _shutil.which("zstd")
            if zstd_bin:
                zstd_payload = subprocess.run([zstd_bin, "-3", "-c"], input=raw44, capture_output=True, check=True).stdout
            else:
                # Valid RFC 8878 Zstd frame with a single last raw (uncompressed) block
                n = len(raw44)
                bh = (1 | (0 << 1) | (n << 3)).to_bytes(3, "little")
                zstd_payload = b"\x28\xb5\x2f\xfd\xa0" + n.to_bytes(4, "little") + bh + raw44
            z_zstd3 = _mk_z3_w44(
                "w68_z3_zstd.zarr",
                [
                    {"name": "bytes", "configuration": {"endian": "little"}},
                    {"name": "zstd", "configuration": {"level": 3}},
                ],
                zstd_payload,
            )
            p_bb = f"/zarr/bbox/-122.5,37.5,-122.0,38.0/64x64.npy?url={q(z_zstd3)}&variable=data"
            rb = client.get(cw_base + p_bb)
            if rb.status_code == 200:
                ab = np.load(io.BytesIO(rb.content))
                err = float(np.abs(ab[0] - arr44).max())
                rec(lb, err < 1e-4, f"max_err={err:.5f}", p_bb)
                if gpu_base:
                    rbg = client.get(gpu_base + p_bb)
                    if rbg.status_code != 200:
                        gpu_diffs.append({"label": lb, "detail": f"GPU HTTP {rbg.status_code}"})
                    else:
                        ok_g, det_g = compare_npy(rb.content, rbg.content, check_values=True)
                        if not ok_g:
                            gpu_diffs.append({"label": lb, "detail": det_g})
            else:
                rec(lb, False, f"HTTP {rb.status_code} {rb.text[:80]!r}", p_bb)

        if wanted("W68 ZARR v2 3D sel=time=nearest/pad/backfill /point") or wanted("W68 ZARR v2 3D sel= conflicting methods -> 400"):
            z_sel = w30_dir / "w68_sel3d.zarr"
            z_sel.mkdir(exist_ok=True)
            (z_sel / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
            t_coords = np.array([10.0, 20.0, 30.0], dtype=np.float64)
            d3 = np.stack([
                np.full((64, 64), 111.0, dtype=np.float32),
                np.full((64, 64), 222.0, dtype=np.float32),
                np.full((64, 64), 333.0, dtype=np.float32),
            ])
            for nm, arr_v, ch, dims, key in [
                ("time", t_coords, [3], ["time"], "0"),
                ("lon", lon, [64], ["lon"], "0"),
                ("lat", lat, [64], ["lat"], "0"),
                ("temp", d3, [3, 64, 64], ["time", "lat", "lon"], "0.0.0"),
            ]:
                vd = z_sel / nm
                vd.mkdir(exist_ok=True)
                (vd / ".zarray").write_text(json.dumps({
                    "zarr_format": 2, "shape": list(arr_v.shape), "chunks": ch,
                    "dtype": arr_v.dtype.str, "compressor": None, "fill_value": None, "order": "C", "filters": None,
                }))
                (vd / ".zattrs").write_text(json.dumps({"_ARRAY_DIMENSIONS": dims}))
                (vd / key).write_bytes(np.ascontiguousarray(arr_v).tobytes())

            lb = "W68 ZARR v2 3D sel=time=nearest/pad/backfill /point"
            if wanted(lb):
                checks_sel = [
                    ("time=nearest::14.0", 111.0),
                    ("time=nearest::16.0", 222.0),
                    ("time=pad::29.0", 222.0),
                    ("time=backfill::11.0", 222.0),
                ]
                got_vals = []
                ok_all = True
                for expr_s, exp_v in checks_sel:
                    ps = f"/zarr/point/-122.25,37.75?url={q(z_sel)}&variable=temp&sel={qe(expr_s)}"
                    rs = client.get(cw_base + ps)
                    if rs.status_code != 200 or abs(rs.json()["values"][0] - exp_v) > 1e-4:
                        ok_all = False
                    got_vals.append(rs.json()["values"][0] if rs.status_code == 200 else f"HTTP{rs.status_code}")
                rec(lb, ok_all, f"vals={got_vals} exp=[111,222,222,222]", ps)

            lb = "W68 ZARR v2 3D sel= conflicting methods -> 400"
            if wanted(lb):
                pc = f"/zarr/point/-122.25,37.75?url={q(z_sel)}&variable=temp&sel={qe('time=nearest::14.0')}&sel={qe('time=pad::29.0')}"
                rc = client.get(cw_base + pc)
                rec(lb, rc.status_code == 400, f"status={rc.status_code}", pc)

    return ran


def main() -> int:
    parser = argparse.ArgumentParser(description="Adversarial differential oracle proving cogzilla is equal or intentionally better than TiTiler 2.4.1.")
    parser.add_argument("--cpu-only", action="store_true", help="Run cogzilla with COGZILLA_GPU_POLICY=never")
    parser.add_argument("--check-cpu-vs-gpu", action="store_true", help="Also verify CPU-only cogzilla matches GPU-enabled cogzilla")
    parser.add_argument("--cogzilla-bin", default=str(COGZILLA_BIN))
    parser.add_argument("--only", default="", help="Substring filter on check label")
    parser.add_argument("--json-out", default="", help="Optional JSON output path")
    args = parser.parse_args()

    bin_path = Path(args.cogzilla_bin)
    if not bin_path.exists():
        raise SystemExit(f"cogzilla binary not found at {bin_path}; run `cargo build --release --bin cogzilla-server --locked`")

    with tempfile.TemporaryDirectory(prefix="cogzilla_adv_oracle_") as tmp:
        tmp_dir = Path(tmp)
        s = build_synthetic_fixtures(tmp_dir / "synth")
        tt_app = tmp_dir / "_tt_app.py"
        tt_app.write_text(
            "from fastapi import FastAPI\n"
            "from rio_tiler.errors import PointOutsideBounds\n"
            "from rio_tiler.io import STACReader\n"
            "from titiler.core.errors import DEFAULT_STATUS_CODES, add_exception_handlers\n"
            "from titiler.core.factory import TilerFactory, MultiBaseTilerFactory, ColorMapFactory, TMSFactory, AlgorithmFactory\n"
            "app = FastAPI()\n"
            "add_exception_handlers(app, {**DEFAULT_STATUS_CODES, PointOutsideBounds: 404})\n"
            "cog = TilerFactory(router_prefix='/cog', add_ogc_maps=True)\n"
            "stac = MultiBaseTilerFactory(reader=STACReader, router_prefix='/stac', add_ogc_maps=True)\n"
            "app.include_router(cog.router, prefix='/cog')\n"
            "app.include_router(stac.router, prefix='/stac')\n"
            "app.include_router(ColorMapFactory().router)\n"
            "app.include_router(TMSFactory().router)\n"
            "app.include_router(AlgorithmFactory().router)\n"
        )

        port_cw = pick_free_port()
        port_tt = pick_free_port()
        procs: list[subprocess.Popen] = []

        try:
            # When --check-cpu-vs-gpu is enabled, always run primary cw_proc as CPU-only
            # and gpu_proc with GPU enabled so we genuinely compare CPU vs GPU.
            primary_cpu_only = args.cpu_only or args.check_cpu_vs_gpu
            cw_proc = start_cogzilla(bin_path, port_cw, cpu_only=primary_cpu_only)
            procs.append(cw_proc)

            port_gpu = None
            if args.check_cpu_vs_gpu:
                port_gpu = pick_free_port()
                gpu_proc = start_cogzilla(bin_path, port_gpu, cpu_only=False)
                procs.append(gpu_proc)

            tt_env = os.environ.copy()
            tt_env.update({"GDAL_CACHEMAX": "256", "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR", "VSI_CACHE": "TRUE"})
            tt_proc = subprocess.Popen(
                [
                    sys.executable, "-m", "uvicorn", "_tt_app:app",
                    "--app-dir", str(tmp_dir), "--host", "127.0.0.1", "--port", str(port_tt),
                    "--workers", "1", "--log-level", "error",
                ],
                env=tt_env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            procs.append(tt_proc)

            wait_ready(f"http://127.0.0.1:{port_cw}/healthz")
            wait_ready(f"http://127.0.0.1:{port_tt}/openapi.json")
            if port_gpu is not None:
                wait_ready(f"http://127.0.0.1:{port_gpu}/healthz")
            for p in procs:
                assert p.poll() is None, "child server exited early"

            cw_base = f"http://127.0.0.1:{port_cw}"
            tt_base = f"http://127.0.0.1:{port_tt}"
            gpu_base = f"http://127.0.0.1:{port_gpu}" if port_gpu else None

            sat = FIXTURES / "satellite_utm10n_rgb_zstd_p3.tif"
            merc = FIXTURES / "mercator_4k_zstd_p3.tif"
            geo4326 = FIXTURES / "geo4326_4k_deflate_p3.tif"
            scale_off = CORPUS / "meta_landsat_scale_offset.tif"
            nd_f32 = CORPUS / "nodata_float32_nan_swath.tif"
            nd_i16 = CORPUS / "nodata_int16_minus9999.tif"
            mask_1bit = CORPUS / "mask_jpeg_internal_1bit.tif"
            mask_rgba = CORPUS / "mask_rgba_4band_u8.tif"
            rgbn_u16 = CORPUS / "layout_4band_rgbn_u16.tif"
            no_ovr = CORPUS / "struct_512tiles_no_ovr.tif"
            polar3413 = CORPUS / "proj_3413_arctic_polar.tif"
            palette = EXTRA / "palette_u8.tif"
            geo360 = EXTRA / "geo_0_360.tif"
            landsat_stac = FIXTURES / "landsat_c2_stac.json"

            # (label, method, path, body, kind)
            cases: list[tuple[str, str, str, dict | None, str]] = []

            # --- 1. Unscale across endpoints ---
            lon, lat, z, tx, ty, b = center_lonlat_and_tile(scale_off)
            poly_so = {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[
                [b[0] + 0.25 * (b[2] - b[0]), b[1] + 0.25 * (b[3] - b[1])],
                [b[0] + 0.75 * (b[2] - b[0]), b[1] + 0.25 * (b[3] - b[1])],
                [b[0] + 0.75 * (b[2] - b[0]), b[1] + 0.75 * (b[3] - b[1])],
                [b[0] + 0.25 * (b[2] - b[0]), b[1] + 0.75 * (b[3] - b[1])],
                [b[0] + 0.25 * (b[2] - b[0]), b[1] + 0.25 * (b[3] - b[1])],
            ]]}, "properties": {}}
            cases += [
                ("PARITY UNSCALE point true", "GET", f"/cog/point/{lon:.5f},{lat:.5f}?url={q(scale_off)}&unscale=true", None, "point"),
                ("PARITY UNSCALE tile .npy true", "GET", f"/cog/tiles/WebMercatorQuad/{z}/{tx}/{ty}.npy?url={q(scale_off)}&unscale=true", None, "npy"),
                ("PARITY UNSCALE tile .tif true", "GET", f"/cog/tiles/WebMercatorQuad/{z}/{tx}/{ty}.tif?url={q(scale_off)}&unscale=true", None, "geotiff"),
                ("PARITY UNSCALE stats GET true", "GET", f"/cog/statistics?url={q(scale_off)}&unscale=true&max_size=256", None, "stats"),
                ("PARITY UNSCALE stats POST max_size=256", "POST", f"/cog/statistics?url={q(scale_off)}&unscale=true&max_size=256", poly_so, "post_stats"),
                ("PARITY UNSCALE feature .npy true", "POST", f"/cog/feature/64x64.npy?url={q(scale_off)}&unscale=true", poly_so, "npy"),
            ]

            # --- 2. All 8 GDAL dtypes (.npy, .tif, return_mask=false, stats, point) ---
            for dt_name in ("uint8", "int8", "uint16", "int16", "uint32", "int32", "float32", "float64"):
                p = DTYPES / f"dtype_{dt_name}.tif"
                lon, lat, _, _, _, _ = center_lonlat_and_tile(p, z=12)
                cases += [
                    (f"PARITY DTYPE [{dt_name}] .npy", "GET", f"/cog/tiles/WebMercatorQuad/12/657/1582.npy?url={q(p)}", None, "npy"),
                    (f"PARITY DTYPE [{dt_name}] .npy no-mask", "GET", f"/cog/tiles/WebMercatorQuad/12/657/1582.npy?url={q(p)}&return_mask=false", None, "npy"),
                    (f"PARITY DTYPE [{dt_name}] .tif", "GET", f"/cog/tiles/WebMercatorQuad/12/657/1582.tif?url={q(p)}", None, "geotiff"),
                    (f"PARITY DTYPE [{dt_name}] stats", "GET", f"/cog/statistics?url={q(p)}&max_size=128", None, "stats"),
                    (f"PARITY DTYPE [{dt_name}] point", "GET", f"/cog/point/{lon:.5f},{lat:.5f}?url={q(p)}", None, "point"),
                ]

            # --- 3. Synthetic torture COGs ---
            cases += [
                ("PARITY SYNTH rotated point", "GET", f"/cog/point/-122.30,37.75?url={q(s['rotated'])}", None, "point"),
                ("PARITY SYNTH south_up .npy", "GET", f"/cog/tiles/WebMercatorQuad/9/82/197.npy?url={q(s['south_up'])}", None, "npy"),
                ("PARITY SYNTH south_up point", "GET", f"/cog/point/-121.705,38.205?url={q(s['south_up'])}", None, "point"),
                ("PARITY SYNTH highprec_i32 .npy", "GET", f"/cog/tiles/WebMercatorQuad/11/327/792.npy?url={q(s['highprec_i32'])}", None, "npy"),
                ("PARITY SYNTH highprec_i32 point", "GET", f"/cog/point/-122.30,37.75?url={q(s['highprec_i32'])}", None, "point"),
                ("PARITY SYNTH highprec_i32 stats", "GET", f"/cog/statistics?url={q(s['highprec_i32'])}&max_size=256", None, "stats"),
                ("PARITY SYNTH perband_so unscale .npy", "GET", f"/cog/tiles/WebMercatorQuad/11/327/792.npy?url={q(s['perband_so'])}&unscale=true", None, "npy"),
                ("PARITY SYNTH perband_so expr unscale", "GET", f"/cog/tiles/WebMercatorQuad/11/327/792.npy?url={q(s['perband_so'])}&unscale=true&expression={qe('b1+b2+b3')}", None, "npy"),
                ("PARITY SYNTH sparse default .npy", "GET", f"/cog/tiles/WebMercatorQuad/11/327/792.npy?url={q(s['sparse_i16'])}", None, "npy"),
                ("PARITY SYNTH sparse nodata=0 .npy", "GET", f"/cog/tiles/WebMercatorQuad/11/327/792.npy?url={q(s['sparse_i16'])}&nodata=0", None, "npy"),
                ("PARITY SYNTH sparse stats nodata=0", "GET", f"/cog/statistics?url={q(s['sparse_i16'])}&nodata=0&max_size=256", None, "stats"),
                ("PARITY SYNTH categorical stats", "GET", f"/cog/statistics?url={q(s['cat_u8'])}&categorical=true&max_size=256", None, "stats"),
                ("PARITY SYNTH categorical subset", "GET", f"/cog/statistics?url={q(s['cat_u8'])}&categorical=true&c=1&c=5&max_size=256", None, "stats"),
                ("PARITY SYNTH custom hist edges", "GET", f"/cog/statistics?url={q(s['cat_u8'])}&histogram_bins=0,2,6,12&max_size=256", None, "stats"),
                ("PARITY SYNTH custom percentiles", "GET", f"/cog/statistics?url={q(s['cat_u8'])}&p=10&p=50&p=90&max_size=256", None, "stats"),
                ("PARITY SYNTH neg_f32 ceil .npy", "GET", f"/cog/tiles/WebMercatorQuad/11/327/792.npy?url={q(s['neg_f32'])}&algorithm=ceil", None, "npy"),
                ("PARITY SYNTH neg_f32 floor .npy", "GET", f"/cog/tiles/WebMercatorQuad/11/327/792.npy?url={q(s['neg_f32'])}&algorithm=floor", None, "npy"),
                ("PARITY SYNTH neg_f32 cast default .npy", "GET", f"/cog/tiles/WebMercatorQuad/11/327/792.npy?url={q(s['neg_f32'])}&algorithm=cast", None, "npy"),
                ("PARITY SYNTH tiny_1x1 cubic .npy", "GET", f"/cog/tiles/WebMercatorQuad/11/327/792.npy?url={q(s['tiny_1x1'])}&resampling=cubic", None, "npy"),
                ("PARITY SYNTH tiny_3x3 lanczos .npy", "GET", f"/cog/tiles/WebMercatorQuad/11/327/792.npy?url={q(s['tiny_3x3'])}&resampling=lanczos", None, "npy"),
                ("PARITY SYNTH constant_42 stats", "GET", f"/cog/statistics?url={q(s['constant'])}", None, "stats"),
                ("PARITY SYNTH constant_42 rescale=42,42", "GET", f"/cog/tiles/WebMercatorQuad/11/327/792.png?url={q(s['constant'])}&rescale=42,42&colormap_name=viridis", None, "png"),
                ("PARITY SYNTH all_nodata .png", "GET", f"/cog/tiles/WebMercatorQuad/11/327/792.png?url={q(s['all_nodata'])}&rescale=0,100&colormap_name=viridis", None, "png"),
                ("PARITY SYNTH all_nodata .npy", "GET", f"/cog/tiles/WebMercatorQuad/11/327/792.npy?url={q(s['all_nodata'])}", None, "npy"),
                ("PARITY SYNTH all_nodata stats", "GET", f"/cog/statistics?url={q(s['all_nodata'])}", None, "stats"),
                ("PARITY SYNTH nd_i16_min .npy", "GET", f"/cog/tiles/WebMercatorQuad/11/327/792.npy?url={q(s['nd_i16_min'])}", None, "npy"),
                ("PARITY SYNTH nd_u8_255 .npy", "GET", f"/cog/tiles/WebMercatorQuad/11/327/792.npy?url={q(s['nd_u8_255'])}", None, "npy"),
                ("PARITY SYNTH nd_u16_max .npy", "GET", f"/cog/tiles/WebMercatorQuad/11/327/792.npy?url={q(s['nd_u16_max'])}", None, "npy"),
                ("PARITY SYNTH f32_infs .npy", "GET", f"/cog/tiles/WebMercatorQuad/11/327/792.npy?url={q(s['f32_infs'])}", None, "npy"),
                ("PARITY SYNTH f32_infs stats", "GET", f"/cog/statistics?url={q(s['f32_infs'])}", None, "stats"),
                ("PARITY SYNTH planar_1bit .png", "GET", f"/cog/tiles/WebMercatorQuad/11/327/792.png?url={q(s['planar_1bit'])}", None, "png"),
                ("PARITY SYNTH planar_rgba .png", "GET", f"/cog/tiles/WebMercatorQuad/11/327/792.png?url={q(s['planar_rgba'])}", None, "png"),
                ("PARITY SYNTH antimeridian east .npy", "GET", f"/cog/tiles/WebMercatorQuad/6/63/35.npy?url={q(s['antimeridian'])}", None, "npy"),
            ]

            # --- 4. Nodata-boundary resampling & /statistics resampling ---
            for rm in ("bilinear", "cubic", "cubic_spline", "lanczos", "average", "gauss"):
                cases.append((f"PARITY NDBND f32_swath [{rm}] .npy", "GET", f"/cog/tiles/WebMercatorQuad/12/657/1582.npy?url={q(nd_f32)}&resampling={rm}", None, "npy"))
            for rm in ("bilinear", "cubic", "average"):
                cases.append((f"PARITY STATRESAMP i16_nd [{rm}]", "GET", f"/cog/statistics?url={q(nd_i16)}&resampling={rm}&max_size=128", None, "stats"))

            # --- 5. Non-Mercator & Polar TMS, 16-bit PNGs, POST /statistics & /feature, OGC /map, STAC ---
            donut_poly = {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [
                [[-122.80, 37.60], [-122.50, 37.60], [-122.50, 37.90], [-122.80, 37.90], [-122.80, 37.60]],
                [[-122.70, 37.70], [-122.60, 37.70], [-122.60, 37.80], [-122.70, 37.80], [-122.70, 37.70]],
            ]}, "properties": {}}
            multi_poly = {"type": "Feature", "geometry": {"type": "MultiPolygon", "coordinates": [
                [[[-122.80, 37.60], [-122.70, 37.60], [-122.70, 37.70], [-122.80, 37.70], [-122.80, 37.60]]],
                [[[-122.60, 37.80], [-122.50, 37.80], [-122.50, 37.90], [-122.60, 37.90], [-122.60, 37.80]]],
            ]}, "properties": {}}
            with rasterio.open(no_ovr) as ds:
                b2 = transform_bounds(ds.crs, "EPSG:4326", *ds.bounds, densify_pts=21)
            poly_no_ovr = {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[
                [b2[0] + 0.1 * (b2[2] - b2[0]), b2[1] + 0.1 * (b2[3] - b2[1])],
                [b2[0] + 0.9 * (b2[2] - b2[0]), b2[1] + 0.1 * (b2[3] - b2[1])],
                [b2[0] + 0.9 * (b2[2] - b2[0]), b2[1] + 0.9 * (b2[3] - b2[1])],
                [b2[0] + 0.1 * (b2[2] - b2[0]), b2[1] + 0.9 * (b2[3] - b2[1])],
                [b2[0] + 0.1 * (b2[2] - b2[0]), b2[1] + 0.1 * (b2[3] - b2[1])],
            ]]}, "properties": {}}
            cases += [
                ("PARITY TMS WorldCRS84Quad .npy", "GET", f"/cog/tiles/WorldCRS84Quad/6/20/18.npy?url={q(geo4326)}", None, "npy"),
                ("PARITY TMS UPSArctic (2,1,2) .npy", "GET", f"/cog/tiles/UPSArcticWGS84Quad/2/1/2.npy?url={q(polar3413)}", None, "npy"),
                ("PARITY U16PNG 1-band .png", "GET", f"/cog/tiles/WebMercatorQuad/12/657/1582.png?url={q(DTYPES / 'dtype_uint16.tif')}", None, "png"),
                ("PARITY U16PNG 3-band RGB .png", "GET", f"/cog/tiles/WebMercatorQuad/10/164/395.png?url={q(rgbn_u16)}&bidx=1&bidx=2&bidx=3", None, "png"),
                ("PARITY POSTSTATS MultiPolygon max_size=256", "POST", f"/cog/statistics?url={q(sat)}&max_size=256", multi_poly, "post_stats"),
                ("PARITY POSTSTATS donut max_size=256", "POST", f"/cog/statistics?url={q(sat)}&max_size=256", donut_poly, "post_stats"),
                ("PARITY POSTSTATS no-overview max_size=128", "POST", f"/cog/statistics?url={q(no_ovr)}&max_size=128", poly_no_ovr, "post_stats"),
                ("PARITY FEAT donut .png", "POST", f"/cog/feature/64x64.png?url={q(sat)}&rescale=0,0.35", donut_poly, "png"),
                ("PARITY OGC map bbox-crs=3857 .png", "GET", f"/cog/map?url={q(sat)}&bbox=-13665000,4530000,-13635000,4560000&bbox-crs=http://www.opengis.net/def/crs/EPSG/0/3857&width=64&height=64&rescale=0,0.35&f=png", None, "png"),
                ("PARITY BUFFER=3 hillshade .png", "GET", f"/cog/tiles/WebMercatorQuad/12/657/1582.png?url={q(merc)}&algorithm=hillshade&buffer=3", None, "png"),
                ("PARITY BUFFER=3 slope .npy", "GET", f"/cog/tiles/WebMercatorQuad/12/657/1582.npy?url={q(merc)}&algorithm=slope&buffer=3", None, "npy"),
                ("PARITY EXPR+ALG grayscale .png", "GET", f"/cog/tiles/WebMercatorQuad/12/652/1583.png?url={q(sat)}&expression=b1*2;b2*2;b3*2&algorithm=grayscale&rescale=0,0.7", None, "png"),
                ("PARITY NODATA i16 override=0 stats", "GET", f"/cog/statistics?url={q(nd_i16)}&nodata=0&max_size=128", None, "stats"),
                ("PARITY STACMIX u8+f32+u16 .npy", "GET", f"/stac/tiles/WebMercatorQuad/11/327/792.npy?url={q(s['stac_mix'])}&assets=a&assets=b&assets=c", None, "npy"),
                ("PARITY STACMIX expr a+b .npy", "GET", f"/stac/tiles/WebMercatorQuad/11/327/792.npy?url={q(s['stac_mix'])}&assets=a&assets=b&expression={qe('b1+b2')}", None, "npy"),
                ("PARITY STAC asset_bidx .npy", "GET", f"/stac/tiles/WebMercatorQuad/12/652/1583.npy?url={q(landsat_stac)}&assets=red&assets=green&asset_bidx=red|1&asset_bidx=green|1", None, "npy"),
            ]

            # --- 5b. Wave 5: Exotic CRSs, GCP-only, multi-overview+unscale, exotic codecs ---
            for name in ("laea_3035", "lcc_3978", "nztm_2193", "ease_6933", "ups_n_32661", "ups_s_32761", "bng_27700", "nad83_4269", "modis_sinu"):
                p = s[name]
                lon, lat, z, tx, ty, _ = center_lonlat_and_tile(p)
                cases += [
                    (f"PARITY CRS [{name}] .npy", "GET", f"/cog/tiles/WebMercatorQuad/{z}/{tx}/{ty}.npy?url={q(p)}", None, "npy"),
                    (f"PARITY CRS [{name}] point", "GET", f"/cog/point/{lon:.5f},{lat:.5f}?url={q(p)}", None, "point"),
                ]
            cases += [
                ("PARITY GCP affine .npy", "GET", f"/cog/tiles/WebMercatorQuad/9/81/197.npy?url={q(s['gcp_affine'])}", None, "npy"),
                ("PARITY GCP affine point", "GET", f"/cog/point/-122.5,37.5?url={q(s['gcp_affine'])}", None, "point"),
                ("PARITY OVR+UNSCALE z9 cubic .npy", "GET", f"/cog/tiles/WebMercatorQuad/9/81/197.npy?url={q(s['multi_ovr_unscale'])}&unscale=true&resampling=cubic", None, "npy"),
                ("PARITY OVR+UNSCALE z9 average .npy", "GET", f"/cog/tiles/WebMercatorQuad/9/81/197.npy?url={q(s['multi_ovr_unscale'])}&unscale=true&resampling=average", None, "npy"),
                ("PARITY CODEC stripped_f64_p3 .npy", "GET", f"/cog/tiles/WebMercatorQuad/11/327/792.npy?url={q(s['stripped_f64_p3'])}", None, "npy"),
                ("PARITY CODEC bigtiff_i32 .npy", "GET", f"/cog/tiles/WebMercatorQuad/11/327/792.npy?url={q(s['bigtiff_i32'])}", None, "npy"),
                ("PARITY CODEC int8_unscale .npy", "GET", f"/cog/tiles/WebMercatorQuad/11/327/792.npy?url={q(s['int8_unscale'])}&unscale=true", None, "npy"),
            ]

            # --- 5c. Wave 6: Colormap LUTs, Content-Crs/Content-Bbox headers, expr VM, custom colormaps ---
            for cm in ("viridis", "plasma", "inferno", "terrain", "coolwarm", "turbo", "gray", "twilight"):
                cases.append((f"PARITY CMAP LUT [{cm}]", "GET", f"/colorMaps/{cm}", None, "json_exact"))
            cases += [
                ("PARITY HDR tile WebMercatorQuad", "GET", f"/cog/tiles/WebMercatorQuad/12/657/1582.png?url={q(merc)}&rescale=0,1000", None, "headers"),
                ("PARITY HDR bbox dst_crs=3857", "GET", f"/cog/bbox/-122.8,37.6,-122.5,37.9/64x64.png?url={q(sat)}&dst_crs=epsg:3857&rescale=0,0.35", None, "headers"),
                ("PARITY HDR preview native", "GET", f"/cog/preview/64x64.png?url={q(sat)}&rescale=0,0.35", None, "headers"),
                ("PARITY HDR feature 4326", "POST", f"/cog/feature/64x64.png?url={q(sat)}&rescale=0,0.35", donut_poly, "headers"),
                ("PARITY EXPR trig+log+pow .npy", "GET", f"/cog/tiles/WebMercatorQuad/12/652/1583.npy?url={q(sat)}&expression={qe('sin(b1)+log1p(b2)+(b3+0.1)**-0.5')}", None, "npy"),
                ("PARITY EXPR where+and int32 .npy", "GET", f"/cog/tiles/WebMercatorQuad/12/652/1583.npy?url={q(sat)}&expression={qe('where((b1>0.1)&(b2<0.3),1,0)')}", None, "npy"),
                ("PARITY EXPR where+or f32 .npy", "GET", f"/cog/tiles/WebMercatorQuad/12/652/1583.npy?url={q(sat)}&expression={qe('where((b1>0.25)|(b3>0.25),b1,b2)')}", None, "npy"),
                ("PARITY CMAP neg float interval .png", "GET", f"/cog/tiles/WebMercatorQuad/11/327/792.png?url={q(s['neg_f32'])}&colormap={qe(json.dumps([[[-100,0],[0,0,255,255]],[[0,100],[255,0,0,255]]]))}", None, "png"),
                ("PARITY CMAP semi-transparent alpha .png", "GET", f"/cog/tiles/WebMercatorQuad/11/327/792.png?url={q(s['cat_u8'])}&colormap={qe(json.dumps({'1':[255,0,0,128],'2':[0,255,0,0],'3':[0,0,255,255],'5':[255,255,0,200],'10':[255,0,255,255]}))}", None, "png"),
            ]

            # --- 5d. Wave 9: Exotic codecs (LERC/WebP/JPEG/1-bit/4-bit/12-bit/Pred2/Pred3/PlanarSep), chained color_formula, algorithm_params ---
            for name in ("w9_lerc", "w9_lerc_deflate", "w9_lerc_zstd", "w9_webp_lossless", "w9_webp_lossy", "w9_nbits_4", "w9_nbits_12", "w9_pred2_i16", "w9_pred3_f64", "w9_planar_sep_rgb"):
                p = s[name]
                lon, lat, z, tx, ty, _ = center_lonlat_and_tile(p, z=10)
                cases += [
                    (f"PARITY W9 [{name}] .npy", "GET", f"/cog/tiles/WebMercatorQuad/{z}/{tx}/{ty}.npy?url={q(p)}", None, "npy"),
                    (f"PARITY W9 [{name}] point", "GET", f"/cog/point/{lon:.5f},{lat:.5f}?url={q(p)}", None, "point"),
                    (f"PARITY W9 [{name}] stats", "GET", f"/cog/statistics?url={q(p)}", None, "stats"),
                ]
            lon_j, lat_j, z_j, tx_j, ty_j, _ = center_lonlat_and_tile(s["w9_jpeg_ycbcr"], z=10)
            cases += [
                ("PARITY W9 [w9_jpeg_ycbcr] .png", "GET", f"/cog/tiles/WebMercatorQuad/{z_j}/{tx_j}/{ty_j}.png?url={q(s['w9_jpeg_ycbcr'])}", None, "png"),
                ("PARITY W9 [w9_jpeg_ycbcr] point", "GET", f"/cog/point/{lon_j:.5f},{lat_j:.5f}?url={q(s['w9_jpeg_ycbcr'])}", None, "point"),
                ("PARITY W9 [w9_nbits_1] point", "GET", f"/cog/point/{lon_j:.5f},{lat_j:.5f}?url={q(s['w9_nbits_1'])}", None, "point"),
                ("PARITY W9 [w9_nbits_1] stats", "GET", f"/cog/statistics?url={q(s['w9_nbits_1'])}", None, "stats"),
                ("PARITY W9 [w9_nbits_12] .png", "GET", f"/cog/tiles/WebMercatorQuad/{z_j}/{tx_j}/{ty_j}.png?url={q(s['w9_nbits_12'])}", None, "png"),
                ("PARITY W9 CF chained Gamma+Sigmoidal+Sat", "GET", f"/cog/tiles/WebMercatorQuad/12/652/1583.png?url={q(sat)}&rescale=0,0.35&color_formula={qe('Gamma RGB 1.5 Sigmoidal RGB 6 0.5 Saturation 1.3')}", None, "png"),
                ("PARITY W9 CF per-channel Gamma R/G/B", "GET", f"/cog/tiles/WebMercatorQuad/12/652/1583.png?url={q(sat)}&rescale=0,0.35&color_formula={qe('Gamma R 1.2 Gamma G 0.8 Gamma B 1.4')}", None, "png"),
                ("PARITY W9 ALG contours custom params", "GET", f"/cog/tiles/WebMercatorQuad/12/657/1582.png?url={q(merc)}&algorithm=contours&algorithm_params={qe(json.dumps({'increment':50,'thickness':2,'minz':0,'maxz':1000}))}", None, "png"),
                ("PARITY W9 ALG normalizedIndex .npy", "GET", f"/cog/tiles/WebMercatorQuad/12/652/1583.npy?url={q(sat)}&bidx=1&bidx=2&algorithm=normalizedIndex", None, "npy"),
                ("PARITY W9 ALG terrarium .png", "GET", f"/cog/tiles/WebMercatorQuad/12/657/1582.png?url={q(merc)}&algorithm=terrarium", None, "png"),
                ("PARITY W9 ALG terrainrgb .png", "GET", f"/cog/tiles/WebMercatorQuad/12/657/1582.png?url={q(merc)}&algorithm=terrainrgb", None, "png"),
            ]

            # --- 5e. Wave 10: Expression VM torture, multi-expr (;), WebP/JPEG/GeoTIFF, WorldCRS84Quad ---
            cases += [
                ("PARITY W10 EXPR abs+sqrt+neg .npy", "GET", f"/cog/tiles/WebMercatorQuad/12/652/1583.npy?url={q(sat)}&expression={qe('abs(-b1)+sqrt(abs(b2))')}", None, "npy"),
                ("PARITY W10 EXPR exp+log+log10 .npy", "GET", f"/cog/tiles/WebMercatorQuad/12/652/1583.npy?url={q(sat)}&expression={qe('exp(b1)+log(b2+1.0)+log10(b3+1.0)')}", None, "npy"),
                ("PARITY W10 EXPR cos+tan+arctan2 .npy", "GET", f"/cog/tiles/WebMercatorQuad/12/652/1583.npy?url={q(sat)}&expression={qe('cos(b1)+tan(b2*0.5)+arctan2(b1,b2+0.1)')}", None, "npy"),
                ("PARITY W10 EXPR arcsin+arccos .npy", "GET", f"/cog/tiles/WebMercatorQuad/12/652/1583.npy?url={q(sat)}&expression={qe('arcsin(b1*0.5)+arccos(b2*0.5)')}", None, "npy"),
                ("PARITY W10 EXPR floor+ceil .npy", "GET", f"/cog/tiles/WebMercatorQuad/12/652/1583.npy?url={q(sat)}&expression={qe('floor(b1*10.0)+ceil(b2*10.0)')}", None, "npy"),
                ("PARITY W10 EXPR multi-expr 3-band .npy", "GET", f"/cog/tiles/WebMercatorQuad/12/652/1583.npy?url={q(sat)}&expression={qe('b1+b2;b2-b1;b3*2.0')}", None, "npy"),
                ("PARITY W10 EXPR multi-expr 3-band .png", "GET", f"/cog/tiles/WebMercatorQuad/12/652/1583.png?url={q(sat)}&expression={qe('b1+b2;b2-b1;b3*2.0')}&rescale=0,0.6&rescale=-0.2,0.2&rescale=0,0.6", None, "png"),
                ("PARITY W10 EXPR div-by-zero safe .npy", "GET", f"/cog/tiles/WebMercatorQuad/12/652/1583.npy?url={q(sat)}&expression={qe('where(b1>0.15,b2/b1,0.0)')}", None, "npy"),
                ("PARITY W10 EXPR nested where int32 .npy", "GET", f"/cog/tiles/WebMercatorQuad/12/652/1583.npy?url={q(sat)}&expression={qe('where(b1>0.2,2,where(b1>0.1,1,0))')}", None, "npy"),
                ("PARITY W10 FMT WebP RGB .webp", "GET", f"/cog/tiles/WebMercatorQuad/12/652/1583.webp?url={q(sat)}&rescale=0,0.35", None, "png"),
                ("PARITY W10 FMT JPEG RGB .jpg", "GET", f"/cog/tiles/WebMercatorQuad/12/652/1583.jpg?url={q(sat)}&rescale=0,0.35", None, "png"),
                ("PARITY W10 FMT GeoTIFF DEM .tif", "GET", f"/cog/tiles/WebMercatorQuad/12/657/1582.tif?url={q(merc)}", None, "geotiff"),
                ("PARITY W10 FMT GeoTIFF DEM return_mask=false", "GET", f"/cog/tiles/WebMercatorQuad/12/657/1582.tif?url={q(merc)}&return_mask=false", None, "geotiff"),
                ("PARITY W10 TMS WorldCRS84Quad .npy", "GET", f"/cog/tiles/WorldCRS84Quad/11/652/593.npy?url={q(sat)}", None, "npy"),
            ]

            # --- 5f. Wave 11: Rotated/Sheared/South-Up geotransforms, Global/Regional CRSs, Deep /stac/* ---
            for name in ("w11_rot30", "w11_sheared", "w11_southup"):
                p = s[name]
                lon, lat, z, tx, ty, wgs = center_lonlat_and_tile(p)
                miny, maxy = min(wgs[1], wgs[3]), max(wgs[1], wgs[3])
                bb = f"{wgs[0]:.5f},{miny:.5f},{wgs[2]:.5f},{maxy:.5f}"
                if name != "w11_rot30":
                    cases.append((f"PARITY W11 XFORM [{name}] .npy", "GET", f"/cog/tiles/WebMercatorQuad/{z}/{tx}/{ty}.npy?url={q(p)}", None, "npy"))
                cases.append((f"PARITY W11 XFORM [{name}] point", "GET", f"/cog/point/{lon:.5f},{lat:.5f}?url={q(p)}", None, "point"))
                if name == "w11_southup":
                    cases.append(("BETTER W11 south-up /bbox (cw 200, tt 500)", "GET", f"/cog/bbox/{bb}/64x64.npy?url={q(p)}", None, "better_cw200_tt500"))
                else:
                    cases.append((f"PARITY W11 XFORM [{name}] bbox", "GET", f"/cog/bbox/{bb}/64x64.npy?url={q(p)}", None, "npy"))
                cases.append((f"PARITY W11 XFORM [{name}] stats", "GET", f"/cog/statistics?url={q(p)}", None, "stats"))
            for name in ("w11_equal_earth", "w11_albers_5070", "w11_lambert93_2154", "w11_utm56s_32756"):
                p = s[name]
                lon, lat, z, tx, ty, _ = center_lonlat_and_tile(p)
                cases += [
                    (f"PARITY W11 CRS [{name}] .npy", "GET", f"/cog/tiles/WebMercatorQuad/{z}/{tx}/{ty}.npy?url={q(p)}", None, "npy"),
                    (f"PARITY W11 CRS [{name}] point", "GET", f"/cog/point/{lon:.5f},{lat:.5f}?url={q(p)}", None, "point"),
                ]
            poly_stac = {
                "type": "Feature", "properties": {},
                "geometry": {"type": "Polygon", "coordinates": [[[-122.70, 37.70], [-122.55, 37.70], [-122.55, 37.85], [-122.70, 37.85], [-122.70, 37.70]]]},
            }
            cases += [
                ("PARITY W11 STAC /point multi-asset", "GET", f"/stac/point/-122.62,37.78?url={q(s['w11_stac'])}&assets=a&assets=b", None, "point"),
                ("PARITY W11 STAC /point expr", "GET", f"/stac/point/-122.62,37.78?url={q(s['w11_stac'])}&assets=a&assets=b&expression={qe('b1-b2')}", None, "point"),
                ("PARITY W11 STAC /bbox 64x64.npy", "GET", f"/stac/bbox/-122.70,37.70,-122.55,37.85/64x64.npy?url={q(s['w11_stac'])}&assets=a&assets=b", None, "npy"),
                ("PARITY W11 STAC /bbox 64x64.png expr", "GET", f"/stac/bbox/-122.70,37.70,-122.55,37.85/64x64.png?url={q(s['w11_stac'])}&assets=a&assets=b&expression={qe('(b1-b2)/(b1+b2)')}&rescale=-1,1&colormap_name=viridis", None, "png"),
                ("PARITY W11 STAC /preview/64x64.npy", "GET", f"/stac/preview/64x64.npy?url={q(s['w11_stac'])}&assets=a", None, "npy"),
                ("PARITY W11 STAC POST /feature 64x64.npy", "POST", f"/stac/feature/64x64.npy?url={q(s['w11_stac'])}&assets=a", poly_stac, "npy"),
                ("PARITY W11 STAC GET /statistics", "GET", f"/stac/statistics?url={q(s['w11_stac'])}&assets=a&max_size=128", None, "stats"),
                ("PARITY W11 STAC POST /statistics", "POST", f"/stac/statistics?url={q(s['w11_stac'])}&assets=a&max_size=128", poly_stac, "post_stats"),
            ]

            # --- 5g. Wave 12: Zonal geometry torture (2-hole Polygon, MultiPolygon, GeometryCollection, MultiPoint) & stats params ---
            w12_swiss = {
                "type": "Feature", "properties": {"id": "swiss"},
                "geometry": {"type": "Polygon", "coordinates": [
                    [[-122.72, 37.68], [-122.48, 37.68], [-122.48, 37.92], [-122.72, 37.92], [-122.72, 37.68]],
                    [[-122.68, 37.72], [-122.62, 37.72], [-122.62, 37.78], [-122.68, 37.78], [-122.68, 37.72]],
                    [[-122.58, 37.82], [-122.52, 37.82], [-122.52, 37.88], [-122.58, 37.88], [-122.58, 37.82]],
                ]},
            }
            w12_mp = {
                "type": "Feature", "properties": {"id": "mp"},
                "geometry": {"type": "MultiPolygon", "coordinates": [
                    [[[-122.72, 37.68], [-122.60, 37.68], [-122.60, 37.80], [-122.72, 37.80], [-122.72, 37.68]],
                     [[-122.68, 37.71], [-122.64, 37.71], [-122.64, 37.75], [-122.68, 37.75], [-122.68, 37.71]]],
                    [[[-122.56, 37.82], [-122.48, 37.82], [-122.48, 37.92], [-122.56, 37.92], [-122.56, 37.82]]],
                ]},
            }
            w12_ls = {
                "type": "Feature", "properties": {"id": "ls"},
                "geometry": {"type": "LineString", "coordinates": [[-122.72, 37.68], [-122.60, 37.80], [-122.48, 37.90]]},
            }
            w12_gc = {
                "type": "Feature", "properties": {"id": "gc"},
                "geometry": {"type": "GeometryCollection", "geometries": [
                    {"type": "Polygon", "coordinates": [[[-122.70, 37.70], [-122.60, 37.70], [-122.60, 37.80], [-122.70, 37.80], [-122.70, 37.70]]]},
                    {"type": "LineString", "coordinates": [[-122.58, 37.80], [-122.50, 37.90]]},
                ]},
            }
            w12_mpt = {
                "type": "Feature", "properties": {"id": "mpt"},
                "geometry": {"type": "MultiPoint", "coordinates": [[-122.70, 37.70], [-122.60, 37.80], [-122.50, 37.90]]},
            }
            cases += [
                ("PARITY W12 ZONAL Polygon 2-holes stats", "POST", f"/cog/statistics?url={q(sat)}&max_size=128", w12_swiss, "post_stats"),
                ("PARITY W12 ZONAL MultiPolygon+hole stats", "POST", f"/cog/statistics?url={q(sat)}&max_size=128", w12_mp, "post_stats"),
                ("PARITY W12 ZONAL GeometryCollection stats", "POST", f"/cog/statistics?url={q(sat)}&max_size=128", w12_gc, "post_stats"),
                ("PARITY W12 FEAT MultiPolygon 64x64.npy", "POST", f"/cog/feature/64x64.npy?url={q(sat)}", w12_mp, "npy"),
                ("PARITY W12 FEAT Polygon 2-holes 64x64.png", "POST", f"/cog/feature/64x64.png?url={q(sat)}&rescale=0,0.35", w12_swiss, "png"),
                ("PARITY W12 FEAT LineString 64x64.npy", "POST", f"/cog/feature/64x64.npy?url={q(sat)}", w12_ls, "npy"),
                ("PARITY W12 FEAT GeometryCollection 64x64.npy", "POST", f"/cog/feature/64x64.npy?url={q(sat)}", w12_gc, "npy"),
                ("PARITY W12 STATS categorical=true", "GET", f"/cog/statistics?url={q(s['cat_u8'])}&categorical=true", None, "stats"),
                ("PARITY W12 STATS categorical+c=1,3,5,99", "GET", f"/cog/statistics?url={q(s['cat_u8'])}&categorical=true&c=1&c=3&c=5&c=99", None, "stats"),
                ("PARITY W12 STATS histogram_bins=4+range", "GET", f"/cog/statistics?url={q(merc)}&histogram_bins=4&histogram_range=0,500", None, "stats"),
                ("PARITY W12 STATS explicit histogram_bins", "GET", f"/cog/statistics?url={q(merc)}&histogram_bins=0,100,300,600,1200", None, "stats"),
                ("PARITY W12 STATS custom p=10,25,75,90", "GET", f"/cog/statistics?url={q(merc)}&p=10&p=25&p=75&p=90", None, "stats"),
                ("PARITY W12 STATS bidx=3&bidx=1 reorder", "GET", f"/cog/statistics?url={q(sat)}&bidx=3&bidx=1&max_size=128", None, "stats"),
                ("PARITY W12 STATS nodata=1 override", "GET", f"/cog/statistics?url={q(s['cat_u8'])}&nodata=1", None, "stats"),
                ("PARITY W12 STATS GET expr NDVI", "GET", f"/cog/statistics?url={q(sat)}&expression={qe('(b1-b2)/(b1+b2)')}&max_size=128", None, "stats"),
                ("PARITY W12 STATS POST expr NDVI", "POST", f"/cog/statistics?url={q(sat)}&expression={qe('(b1-b2)/(b1+b2)')}&max_size=128", w12_swiss, "post_stats"),
            ]

            # --- 5h. Wave 13: Full 9-mode resampling/reproject matrix, buffer=, coord_crs/dst_crs, tilesize=128/512 ---
            for mode in ("nearest", "bilinear", "cubic", "cubic_spline", "lanczos", "average", "mode", "gauss", "rms"):
                cases.append((f"PARITY W13 RESAMP same-CRS [{mode}] .npy", "GET", f"/cog/tiles/WebMercatorQuad/12/657/1582.npy?url={q(merc)}&resampling={mode}", None, "npy"))
                if mode != "gauss":
                    cases.append((f"PARITY W13 REPROJ [{mode}] .npy", "GET", f"/cog/tiles/WebMercatorQuad/12/652/1583.npy?url={q(sat)}&reproject={mode}", None, "npy"))
            # Precomputed EPSG:3857 and EPSG:32610 coords for SF point (-122.62, 37.78) and bbox (-122.70, 37.70, -122.55, 37.85)
            w13_poly_3857 = {
                "type": "Feature", "properties": {},
                "geometry": {"type": "Polygon", "coordinates": [[
                    [-13658901.07, 4537178.90], [-13642203.15, 4537178.90],
                    [-13642203.15, 4558308.60], [-13658901.07, 4558308.60],
                    [-13658901.07, 4537178.90],
                ]]},
            }
            cases += [
                ("PARITY W13 RESAMP+REPROJ bilinear+cubic .npy", "GET", f"/cog/tiles/WebMercatorQuad/12/652/1583.npy?url={q(sat)}&resampling=bilinear&reproject=cubic", None, "npy"),
                ("PARITY W13 RESAMP+REPROJ average+lanczos .npy", "GET", f"/cog/tiles/WebMercatorQuad/12/652/1583.npy?url={q(sat)}&resampling=average&reproject=lanczos", None, "npy"),
                ("PARITY W13 BUFFER=4 DEM .npy", "GET", f"/cog/tiles/WebMercatorQuad/12/657/1582.npy?url={q(merc)}&buffer=4", None, "npy"),
                ("PARITY W13 BUFFER=16 RGB .png", "GET", f"/cog/tiles/WebMercatorQuad/12/652/1583.png?url={q(sat)}&rescale=0,0.35&buffer=16", None, "png"),
                ("PARITY W13 BUFFER=4 DEM .tif", "GET", f"/cog/tiles/WebMercatorQuad/12/657/1582.tif?url={q(merc)}&buffer=4", None, "geotiff"),
                ("PARITY W13 /point coord_crs=3857", "GET", f"/cog/point/-13649995.51,4548436.11?url={q(sat)}&coord_crs=EPSG:3857", None, "point"),
                ("PARITY W13 /point coord_crs=32610", "GET", f"/cog/point/533447.82,4181384.60?url={q(sat)}&coord_crs=EPSG:32610", None, "point"),
                ("PARITY W13 /bbox coord=32610 dst=3857 .npy", "GET", f"/cog/bbox/526430.92,4172485.08,539579.88,4189174.16/64x64.npy?url={q(sat)}&coord_crs=EPSG:32610&dst_crs=EPSG:3857", None, "npy"),
                ("PARITY W13 /bbox coord=3857 dst=4326 .npy", "GET", f"/cog/bbox/-13658901.07,4537178.90,-13642203.15,4558308.60/64x64.npy?url={q(sat)}&coord_crs=EPSG:3857&dst_crs=EPSG:4326", None, "npy"),
                ("PARITY W13 POST /feature coord=3857 dst=32610", "POST", f"/cog/feature/64x64.npy?url={q(sat)}&coord_crs=EPSG:3857&dst_crs=EPSG:32610", w13_poly_3857, "npy"),
                ("PARITY W13 POST /statistics coord=3857 native", "POST", f"/cog/statistics?url={q(sat)}&coord_crs=EPSG:3857", w13_poly_3857, "post_stats"),
                ("PARITY W13 TILESIZE=128 .npy", "GET", f"/cog/tiles/WebMercatorQuad/12/657/1582.npy?url={q(merc)}&tilesize=128", None, "npy"),
                ("PARITY W13 TILESIZE=512 .png", "GET", f"/cog/tiles/WebMercatorQuad/12/652/1583.png?url={q(sat)}&rescale=0,0.35&tilesize=512", None, "png"),
            ]

            # --- 5i. Wave 14: Qualitative blind spots (Big-Endian, overview masks, .msk sidecar, per-band unscale, 6-band, Gray+Alpha, u16 RGBA, const/all-nodata/Inf, South Pole) ---
            for name in ("w14_be_u16", "w14_be_i16", "w14_be_f32_p3", "w14_be_f64_p3"):
                cases.append((f"PARITY W14 BE [{name}] .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s[name])}", None, "npy"))
            cases += [
                ("PARITY W14 OVRMASK z=8 overview .npy", "GET", f"/cog/tiles/WebMercatorQuad/8/40/99.npy?url={q(s['w14_ovr_mask'])}", None, "npy"),
                ("PARITY W14 OVRMASK z=8 overview .png", "GET", f"/cog/tiles/WebMercatorQuad/8/40/99.png?url={q(s['w14_ovr_mask'])}", None, "png"),
                ("PARITY W14 EXT .msk sidecar .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w14_ext_mask'])}", None, "npy"),
                ("PARITY W14 PER-BAND unscale .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w14_per_band_scale'])}&unscale=true", None, "npy"),
                ("PARITY W14 PER-BAND unscale expr .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w14_per_band_scale'])}&unscale=true&expression={qe('b1+b2+b3')}", None, "npy"),
                ("PARITY W14 NEG-SCALE unscale .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w14_neg_scale'])}&unscale=true", None, "npy"),
                ("PARITY W14 6-BAND bidx=6,4,2 .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w14_6b'])}&bidx=6&bidx=4&bidx=2", None, "npy"),
                ("PARITY W14 6-BAND expr (b6-b2)/(b6+b2)", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w14_6b'])}&expression={qe('(b6-b2)/(b6+b2)')}", None, "npy"),
                ("PARITY W14 GRAY+ALPHA .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w14_gray_alpha'])}", None, "npy"),
                ("PARITY W14 RGBA-U16 .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w14_rgba_u16'])}", None, "npy"),
                ("PARITY W14 CONST-42 rescale=42,42 .png", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w14_const'])}&rescale=42,42", None, "png"),
                ("PARITY W14 CONST-42 /statistics", "GET", f"/cog/statistics?url={q(s['w14_const'])}", None, "stats"),
                ("PARITY W14 ALL-NODATA .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w14_allnd'])}", None, "npy"),
                ("PARITY W14 ALL-NODATA /statistics", "GET", f"/cog/statistics?url={q(s['w14_allnd'])}", None, "stats"),
                ("BETTER W14 F32 +/-Inf .npy (cw masks Inf, tt leaves Inf)", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w14_inf'])}", None, "better_inf_masked"),
                ("PARITY W14 F32 +/-Inf /statistics", "GET", f"/cog/statistics?url={q(s['w14_inf'])}", None, "stats"),
                ("PARITY W14 SOUTH-POLE EPSG:3031 UPSAntarctic", "GET", f"/cog/tiles/UPSAntarcticWGS84Quad/2/1/1.npy?url={q(s['w14_south_pole'])}", None, "npy"),
                # --- Wave 16: Photometric=MinIsWhite, partial final strips, Palette ---
                ("PARITY W16 MINISWHITE u8 .png", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w16_miniswhite_u8'])}", None, "png"),
                ("PARITY W16 MINISWHITE 1bit .png", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w16_miniswhite_1bit'])}", None, "png"),
                ("PARITY W16 STRIP partial final CONTIG .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w16_strip_partial_contig'])}", None, "npy"),
                ("PARITY W16 STRIP partial final SEPARATE RGBA .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w16_strip_partial_sep_rgba'])}", None, "npy"),
                ("PARITY W16 PALETTE u8 .png", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w16_palette'])}", None, "png"),
                ("BETTER W16 MINISWHITE u8 .npy raw (cw 2ch, tt 4ch)", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w16_miniswhite_u8'])}", None, "better_palette_npy"),
                ("BETTER W16 PALETTE u8 .npy raw (cw 2ch, tt 4ch)", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w16_palette'])}", None, "better_palette_npy"),
                # --- Wave 18: GCP-only GeoTIFFs, OGC /cog/map, /cog/preview, TileJSON, GPU torture ---
                ("BETTER W18 GCP affine /cog/point (cw 324.0, tt None)", "GET", f"/cog/point/-122.25,37.75?url={q(s['w18_gcp_aff'])}", None, "better_gcp_point"),
                ("BETTER W18 GCP rotated /cog/point (cw 324.0, tt None)", "GET", f"/cog/point/-122.25,37.75?url={q(s['w18_gcp_rot'])}", None, "better_gcp_point"),
                ("BETTER W18 GCP affine .png (cw gradient, tt solid purple)", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w18_gcp_aff'])}&rescale=100,600&colormap_name=viridis", None, "better_gcp_png"),
                ("PARITY W18 OGC /cog/map 4326 .png", "GET", f"/cog/map?url={q(s['w18_cog3b'])}&bbox=-122.5,37.5,-122.0,38.0&width=128&height=128&rescale=0,300&f=png", None, "png"),
                ("PARITY W18 OGC /cog/map bbox-crs=3857 crs=4326 .png", "GET", f"/cog/map?url={q(s['w18_cog3b'])}&bbox=-13636637,4509036,-13580978,4579425&bbox-crs={qe('http://www.opengis.net/def/crs/EPSG/0/3857')}&crs={qe('http://www.opengis.net/def/crs/EPSG/0/4326')}&width=128&height=128&rescale=0,300&f=png", None, "png"),
                ("PARITY W18 PREVIEW max_size=64 .npy", "GET", f"/cog/preview.npy?url={q(s['w18_cog3b'])}&max_size=64", None, "npy"),
                ("PARITY W18 PREVIEW max_size=64 unscale expr .png", "GET", f"/cog/preview.png?url={q(s['w18_cog3b'])}&max_size=64&unscale=true&expression={qe('(b2-b1)/(b2+b1)')}&rescale=-1,1&colormap_name=viridis", None, "png"),
                ("PARITY W18 GPU torture cubic+RGB+formula .png", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w18_cog3b'])}&resampling=cubic&rescale=0,300&color_formula={qe('Gamma RGB 1.4')}", None, "png"),
                ("PARITY W18 TILEJSON WebMercatorQuad", "GET", f"/cog/WebMercatorQuad/tilejson.json?url={q(s['w18_cog3b'])}&bidx=1&rescale=0,300&colormap_name=viridis", None, "tilejson_w18"),
                # --- Wave 19: Custom JSON Colormaps & Built-In Algorithms ---
                ("PARITY W19 CMAP discrete w/ transparent class .png", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w19_cat_u8'])}&colormap={qe(json.dumps({'0':[255,0,0,255],'1':[0,255,0,255],'2':[0,0,255,0],'3':[255,255,0,255]}))}", None, "png"),
                ("PARITY W19 CMAP discrete hex strings .png", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w19_cat_u8'])}&colormap={qe(json.dumps({'0':'#ff0000','1':'#00ff00','2':'#0000ff','3':'#ffff00'}))}", None, "png"),
                ("PARITY W19 CMAP interval negative-to-pos f32 .png", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w19_dem_f32'])}&colormap={qe(json.dumps([[[-100,0],[0,0,255,255]],[[0,100],[0,255,0,255]],[[100,300],[255,0,0,255]]]))}", None, "png"),
                ("BETTER W19 CMAP linear + rescale (cw interp, tt sparse)", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w19_dem_f32'])}&rescale=-60,200&colormap={qe(json.dumps({'0':[0,0,0,255],'128':[255,0,0,255],'255':[255,255,255,255]}))}&colormap_type=linear", None, "better_linear_cmap"),
                ("PARITY W19 CMAP negative int16 keys .png", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w19_neg_i16'])}&colormap={qe(json.dumps({'-10':[255,0,0,255],'0':[0,255,0,255],'10':[0,0,255,255]}))}", None, "png"),
                ("PARITY W19 ALG terrarium .png", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w19_dem_f32'])}&algorithm=terrarium", None, "png"),
                ("PARITY W19 ALG terrainrgb .png", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w19_dem_f32'])}&algorithm=terrainrgb", None, "png"),
                ("PARITY W19 ALG normalizedIndex .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w19_nir_red'])}&algorithm=normalizedIndex", None, "npy"),
                ("PARITY W19 ALG grayscale .png", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w19_rgb_u8'])}&algorithm=grayscale", None, "png"),
                ("PARITY W19 ALG bitonal .png", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w19_rgb_u8'])}&algorithm=bitonal", None, "png"),
                ("PARITY W19 ALG ceil negative f32 .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w19_dem_f32'])}&algorithm=ceil", None, "npy"),
                ("PARITY W19 ALG floor negative f32 .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w19_dem_f32'])}&algorithm=floor", None, "npy"),
                ("PARITY W19 ALG contours increment=50 .png", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w19_dem_f32'])}&algorithm=contours&algorithm_params={qe(json.dumps({'increment':50}))}", None, "png"),
                # --- Wave 20: BigTIFF (LE/BE), PackBits, Non-Square Tiles, 4:1 Anisotropic Pixels & Multi-Overview Nodata ---
                ("PARITY W20 BIGTIFF LE predictor=3 .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w20_bigtiff_le'])}", None, "npy"),
                ("PARITY W20 BIGTIFF BE predictor=3 .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w20_bigtiff_be'])}", None, "npy"),
                ("PARITY W20 PACKBITS RGB .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w20_packbits_rgb'])}", None, "npy"),
                ("PARITY W20 PACKBITS RGB .png", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w20_packbits_rgb'])}", None, "png"),
                ("PARITY W20 NONSQ tile 128x32 .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w20_nonsq'])}", None, "npy"),
                ("PARITY W20 ANISO 4:1 bilinear .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w20_aniso'])}&resampling=bilinear", None, "npy"),
                ("PARITY W20 ANISO 4:1 cubic .png", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w20_aniso'])}&resampling=cubic&rescale=0,500&colormap_name=viridis", None, "png"),
                ("PARITY W20 ANISO 4:1 /statistics", "GET", f"/cog/statistics?url={q(s['w20_aniso'])}", None, "stats"),
                ("PARITY W20 MULTI-OV z11 unscale=true .npy", "GET", f"/cog/tiles/WebMercatorQuad/11/328/793.npy?url={q(s['w20_multi_ov'])}&unscale=true", None, "npy"),
                ("PARITY W20 MULTI-OV z8 (4x ov) unscale=true .npy", "GET", f"/cog/tiles/WebMercatorQuad/8/41/99.npy?url={q(s['w20_multi_ov'])}&unscale=true", None, "npy"),
                ("PARITY W20 MULTI-OV nodata=200 override .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w20_multi_ov'])}&nodata=200", None, "npy"),
                ("PARITY W20 MULTI-OV nodata=200 override /statistics", "GET", f"/cog/statistics?url={q(s['w20_multi_ov'])}&nodata=200", None, "stats"),
                ("PARITY W20 MULTI-OV nodata=200 override /point (-9999)", "GET", f"/cog/point/-122.22,37.67?url={q(s['w20_multi_ov'])}&nodata=200", None, "point"),
                # --- Wave 21: Heterogeneous multi-CRS / multi-resolution / multi-dtype STAC ---
                ("PARITY W21 STAC hetero stack red(4326 u16)+nir(32610 f32) .npy", "GET", f"/stac/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w21_stac'])}&assets=red&assets=nir", None, "npy"),
                ("PARITY W21 STAC hetero /point red+nir", "GET", f"/stac/point/-122.25,37.75?url={q(s['w21_stac'])}&assets=red&assets=nir", None, "point"),
                ("BETTER W21 STAC hetero NDVI asset_as_band .png (cw 200, tt 400)", "GET", f"/stac/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w21_stac'])}&assets=red&assets=nir&asset_as_band=true&expression={qe('(nir-red)/(nir+red)')}&rescale=-1,1&colormap_name=viridis", None, "better_cw200_tt400"),
                ("BETTER W21 STAC hetero NDVI without assets= (cw 200, tt 422)", "GET", f"/stac/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w21_stac'])}&asset_as_band=true&expression={qe('(nir-red)/(nir+red)')}&rescale=-1,1&colormap_name=viridis", None, "better_cw200_tt422"),
                ("BETTER W21 STAC hetero /point NDVI without assets= (cw 200, tt 422)", "GET", f"/stac/point/-122.25,37.75?url={q(s['w21_stac'])}&asset_as_band=true&expression={qe('(nir-red)/(nir+red)')}", None, "better_cw200_tt422"),
                # --- Wave 22: GeoTIFF NaN-leak fix on hetero STAC .tif & Expression VM trig/where/domain ops ---
                ("PARITY W22 STAC hetero stack red+nir .tif", "GET", f"/stac/tiles/WebMercatorQuad/10/164/396.tif?url={q(s['w21_stac'])}&assets=red&assets=nir", None, "geotiff"),
                ("PARITY W22 EXPR nested where .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w22_f32_2b'])}&expression={qe('where(b1>10,where(b2>15,b1+b2,b1-b2),0)')}", None, "npy"),
                ("PARITY W22 EXPR sin+cos .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w22_f32_2b'])}&expression={qe('sin(b1*0.1)+cos(b2*0.1)')}", None, "npy"),
                ("PARITY W22 EXPR arctan2 .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w22_f32_2b'])}&expression={qe('arctan2(b2,b1)')}", None, "npy"),
                ("PARITY W22 EXPR abs+sqrt(abs) .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w22_f32_2b'])}&expression={qe('sqrt(abs(b1))+abs(b2)')}", None, "npy"),
                ("PARITY W22 EXPR log1p(abs) .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w22_f32_2b'])}&expression={qe('log1p(abs(b1))')}", None, "npy"),
                ("BETTER W22 EXPR domain sqrt(b1) w/ negs .png (cw masks NaN, tt purple)", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w22_f32_2b'])}&expression={qe('sqrt(b1)')}&rescale=0,10&colormap_name=viridis", None, "better_expr_nan_masked"),
                # --- Wave 23: /info.geojson, stats params (categorical, p=, histogram_bins/range, FeatureCollection), per-band rescale & dup bidx ---
                ("PARITY W23 /cog/info.geojson", "GET", f"/cog/info.geojson?url={q(s['w23_f32_3b'])}", None, "info_geojson_w23"),
                ("PARITY W23 STATS categorical uint8", "GET", f"/cog/statistics?url={q(s['w23_cat_u8'])}&categorical=true", None, "stats_w23"),
                ("PARITY W23 STATS categorical neg int16", "GET", f"/cog/statistics?url={q(s['w23_cat_i16'])}&categorical=true", None, "stats_w23"),
                ("PARITY W23 STATS custom p=10,50,90", "GET", f"/cog/statistics?url={q(s['w23_f32_3b'])}&bidx=1&p=10&p=50&p=90", None, "stats_w23"),
                ("PARITY W23 STATS histogram_bins=5 range=0,200", "GET", f"/cog/statistics?url={q(s['w23_f32_3b'])}&bidx=1&histogram_bins=5&histogram_range=0,200", None, "stats_w23"),
                ("PARITY W23 STATS unscale=true", "GET", f"/cog/statistics?url={q(s['w23_f32_3b'])}&unscale=true", None, "stats"),
                ("PARITY W23 STATS expression (b2-b1)/(b2+b1)", "GET", f"/cog/statistics?url={q(s['w23_f32_3b'])}&expression={qe('(b2-b1)/(b2+b1)')}", None, "stats"),
                ("PARITY W23 POST /statistics FeatureCollection 2x", "POST", f"/cog/statistics?url={q(s['w23_cat_u8'])}&categorical=true&p=25&p=75", {
                    "type": "FeatureCollection",
                    "features": [
                        {"type": "Feature", "id": "west", "geometry": {"type": "Polygon", "coordinates": [[[-122.45, 37.55], [-122.25, 37.55], [-122.25, 37.95], [-122.45, 37.95], [-122.45, 37.55]]]}, "properties": {"name": "west"}},
                        {"type": "Feature", "id": "east", "geometry": {"type": "Polygon", "coordinates": [[[-122.25, 37.55], [-122.05, 37.55], [-122.05, 37.95], [-122.25, 37.95], [-122.25, 37.55]]]}, "properties": {"name": "east"}},
                    ],
                }, "post_fc_stats_w23"),
                ("PARITY W23 PER-BAND rescale 3-band .png", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w23_f32_3b'])}&rescale=0,100&rescale=50,200&rescale=100,300", None, "png"),
                ("PARITY W23 DUP bidx=1,1,1 pseudo-RGB .png", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w23_f32_3b'])}&bidx=1&bidx=1&bidx=1&rescale=0,250", None, "png"),
                ("PARITY W23 DUP bidx=1,1,1 pseudo-RGB .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w23_f32_3b'])}&bidx=1&bidx=1&bidx=1", None, "npy"),
            ]
            # --- Wave 24: Non-Mercator OGC TileMatrixSets (pixel + Content-Crs/Content-Bbox parity) ---
            for tms_id, fpath, lon, lat, z in (
                ("WGS1984Quad", s["w23_f32_3b"], -122.25, 37.75, 8),
                ("WorldMercatorWGS84Quad", s["w23_f32_3b"], -122.25, 37.75, 9),
                ("EuropeanETRS89_LAEAQuad", s["w24_eu"], 10.5, 48.5, 6),
                ("NZTM2000Quad", s["w24_nz"], 172.5, -42.0, 5),
                ("CanadianNAD83_LCC", s["w24_ca"], -97.0, 53.0, 5),
            ):
                t24 = morecantile.tms.get(tms_id).tile(lon, lat, z)
                cases.append((f"PARITY W24 TMS {tms_id} {z}/{t24.x}/{t24.y}.npy", "GET", f"/cog/tiles/{tms_id}/{z}/{t24.x}/{t24.y}.npy?url={q(fpath)}", None, "npy"))
                cases.append((f"PARITY W24 TMS {tms_id} {z}/{t24.x}/{t24.y}.png+hdr", "GET", f"/cog/tiles/{tms_id}/{z}/{t24.x}/{t24.y}.png?url={q(fpath)}&bidx=1&rescale=100,600&colormap_name=viridis", None, "headers"))
            # --- Wave 25: High-latitude 84°N (10:1 Mercator stretch) & Global zoom-0 tiles ---
            t25_hi = morecantile.tms.get("WebMercatorQuad").tile(17.5, 83.5, 5)
            cases += [
                ("PARITY W25 HIGH-LAT 84N 3413->3857 .npy", "GET", f"/cog/tiles/WebMercatorQuad/5/{t25_hi.x}/{t25_hi.y}.npy?url={q(s['w25_highlat_84n'])}", None, "npy"),
                ("PARITY W25 HIGH-LAT 84N 3413->3857 bilinear .png", "GET", f"/cog/tiles/WebMercatorQuad/5/{t25_hi.x}/{t25_hi.y}.png?url={q(s['w25_highlat_84n'])}&resampling=bilinear&rescale=100,550&colormap_name=viridis", None, "png"),
                ("PARITY W25 GLOBAL WebMercatorQuad 0/0/0 .npy", "GET", f"/cog/tiles/WebMercatorQuad/0/0/0.npy?url={q(s['w25_global'])}", None, "npy"),
                ("PARITY W25 GLOBAL WGS1984Quad 0/0/0 .npy", "GET", f"/cog/tiles/WGS1984Quad/0/0/0.npy?url={q(s['w25_global'])}", None, "npy"),
                ("PARITY W25 GLOBAL WGS1984Quad 0/1/0 .png", "GET", f"/cog/tiles/WGS1984Quad/0/1/0.png?url={q(s['w25_global'])}&rescale=50,370&colormap_name=viridis", None, "png"),
                # --- Wave 27: @2x Retina tiles, POST /feature formats (.png/.tif/.webp), LineString/MultiLineString stats & auto-format ---
                ("BETTER W27 @2x Retina 512x512 .png (cw 200, tt 422)", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396@2x.png?url={q(s['w23_f32_3b'])}&bidx=1&rescale=10,270&colormap_name=viridis", None, "better_retina_2x"),
                ("BETTER W27 @2x Retina 512x512 .npy (cw 200, tt 422)", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396@2x.npy?url={q(s['w23_f32_3b'])}&bidx=1", None, "better_retina_2x"),
                ("PARITY W27 POST /feature triangle .png", "POST", f"/cog/feature/128x128.png?url={q(s['w23_f32_3b'])}&bidx=1&rescale=10,270&colormap_name=viridis", {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[[-122.45, 37.55], [-122.05, 37.55], [-122.25, 37.95], [-122.45, 37.55]]]}, "properties": {}}, "png"),
                ("PARITY W27 POST /feature triangle .tif", "POST", f"/cog/feature/128x128.tif?url={q(s['w23_f32_3b'])}&bidx=1", {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[[-122.45, 37.55], [-122.05, 37.55], [-122.25, 37.95], [-122.45, 37.55]]]}, "properties": {}}, "geotiff"),
                ("PARITY W27 POST /feature triangle .webp", "POST", f"/cog/feature/128x128.webp?url={q(s['w19_rgb_u8'])}", {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [[[-122.45, 37.55], [-122.05, 37.55], [-122.25, 37.95], [-122.45, 37.55]]]}, "properties": {}}, "png"),
                ("BETTER W27 POST /statistics LineString (cw exact, tt 10x low)", "POST", f"/cog/statistics?url={q(s['w23_f32_3b'])}&bidx=1", {"type": "Feature", "geometry": {"type": "LineString", "coordinates": [[-122.45, 37.55], [-122.25, 37.75], [-122.05, 37.95]]}, "properties": {}}, "better_linestring_stats"),
                ("BETTER W27 POST /statistics MultiLineString (cw 204.8, tt 20.5)", "POST", f"/cog/statistics?url={q(s['w23_f32_3b'])}&bidx=1", {"type": "Feature", "geometry": {"type": "MultiLineString", "coordinates": [[[-122.45, 37.60], [-122.05, 37.60]], [[-122.45, 37.90], [-122.05, 37.90]]]}, "properties": {}}, "better_linestring_stats"),
                ("PARITY W27 AUTO-FMT extensionless opaque -> JPEG", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396?url={q(s['w27_opaque_rgb'])}", None, "auto_jpeg"),
                # --- Wave 29: Unprobed TIFF codecs/predictors & STAC asset_as_band=true named expressions ---
                ("PARITY W29 CODEC Grayscale JPEG .png", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w29_gray_jpeg'])}", None, "png"),
                ("PARITY W29 CODEC Grayscale JPEG /point", "GET", f"/cog/point/-122.25,37.75?url={q(s['w29_gray_jpeg'])}", None, "point"),
                ("PARITY W29 CODEC LZW Predictor=2 uint16 .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w29_lzw_p2_u16'])}", None, "npy"),
                ("PARITY W29 CODEC LZW Predictor=2 int16+nodata .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w29_lzw_p2_i16'])}", None, "npy"),
                ("PARITY W29 CODEC LZW Predictor=3 float32 .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w29_lzw_p3_f32'])}", None, "npy"),
                ("PARITY W29 CODEC ZSTD Predictor=2 uint16 .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w29_zstd_p2_u16'])}", None, "npy"),
                ("PARITY W29 CODEC YCbCr JPEG internal mask .png", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w29_ycbcr_jpeg_mask'])}", None, "png"),
                ("PARITY W29 STAC NDVI positional (b2-b1)/(b2+b1) .npy", "GET", f"/stac/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w29_stac_ab'])}&assets=red&assets=nir&expression={qe('(b2-b1)/(b2+b1)')}", None, "npy"),
                ("BETTER W29 STAC asset_as_band=true named expr (cw 200, tt 400)", "GET", f"/stac/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w29_stac_ab'])}&assets=red&assets=nir&asset_as_band=true&expression={qe('(nir-red)/(nir+red)')}", None, "better_stac_asset_as_band"),
                # --- Wave 31: Inverted & negative rescale, BGR band reorder, colormap+color_formula, opaque WebP, non-square preview ---
                ("BETTER W31 RESCALE inverted 200,0 gray (cw ramp, tt solid 255)", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w31_f32'])}&rescale=200,0", None, "better_inverted_rescale_w31"),
                ("BETTER W31 RESCALE inverted 200,0 viridis (cw ramp, tt flat)", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w31_f32'])}&rescale=200,0&colormap_name=viridis", None, "better_inverted_rescale_w31"),
                ("PARITY W31 RESCALE all-negative -150,-10 viridis .png", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w31_f32'])}&rescale=-150,-10&colormap_name=viridis", None, "png"),
                ("PARITY W31 BIDX BGR 3,2,1 .png", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w31_rgb'])}&bidx=3&bidx=2&bidx=1", None, "png"),
                ("PARITY W31 BIDX BGR 3,2,1 .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w31_rgb'])}&bidx=3&bidx=2&bidx=1", None, "npy"),
                ("PARITY W31 BIDX BGR 3,2,1 .tif", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.tif?url={q(s['w31_rgb'])}&bidx=3&bidx=2&bidx=1", None, "geotiff"),
                ("BETTER W31 CMAP+CF viridis + Gamma RGB 1.8 (cw 200, tt 500)", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w31_f32'])}&rescale=-100,200&colormap_name=viridis&color_formula={qe('Gamma RGB 1.8')}", None, "better_cmap_color_formula"),
                ("PARITY W31 RETURN_MASK=false .webp", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.webp?url={q(s['w31_rgb'])}&return_mask=false", None, "png"),
                ("PARITY W31 PREVIEW non-square 96x48 .png", "GET", f"/cog/preview/96x48.png?url={q(s['w31_rgb'])}", None, "png"),
                # --- Wave 32: Non-256 tilesize (128/512) cubic/hillshade + STAC OGC (/stac/map, TileJSON, WMTS) ---
                ("PARITY W32 TILESIZE=128 cubic viridis .png", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w32_dem'])}&tilesize=128&resampling=cubic&rescale=100,900&colormap_name=viridis", None, "png"),
                ("PARITY W32 TILESIZE=512 cubic viridis .png", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w32_dem'])}&tilesize=512&resampling=cubic&rescale=100,900&colormap_name=viridis", None, "png"),
                ("PARITY W32 TILESIZE=128 .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w32_dem'])}&tilesize=128", None, "npy"),
                ("BETTER W32 TILESIZE=512 hillshade seamless (cw 512, tt 506)", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w32_dem'])}&tilesize=512&algorithm=hillshade", None, "better_hillshade_512"),
                ("PARITY W32 STAC /stac/map 4326 64x64 .png", "GET", f"/stac/map?url={q(s['w32_stac'])}&assets=dem&bbox=-122.5,37.5,-122.0,38.0&width=64&height=64&rescale=100,900&colormap_name=viridis&f=png", None, "png"),
                ("PARITY W32 STAC TileJSON WebMercatorQuad", "GET", f"/stac/WebMercatorQuad/tilejson.json?url={q(s['w32_stac'])}&assets=dem", None, "tilejson_w32"),
                ("BETTER W32 STAC WMTS Capabilities XML (cw 200, tt 404)", "GET", f"/stac/WebMercatorQuad/WMTSCapabilities.xml?url={q(s['w32_stac'])}&assets=dem", None, "better_stac_wmts"),
                # --- Wave 33: US Survey Feet, Intl Feet, Ellipsoidal World Mercator 3395, NAD83 UTM 10N, Arctic 3995 & Planar SEPARATE f32 ZSTD P3 ---
                ("PARITY W33 PLANAR SEPARATE 3-band f32 ZSTD P3 .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w33_planar_f32_zstd_p3'])}", None, "npy"),
            ]
            for name in ("w33_ca_usft_2227", "w33_or_intlft_2913", "w33_merc_3395", "w33_nad83_26910", "w33_arctic_3995"):
                p = s[name]
                lon_c, lat_c, z_c, tx_c, ty_c, _ = center_lonlat_and_tile(p, z=6 if name == "w33_arctic_3995" else 10)
                cases += [
                    (f"PARITY W33 CRS [{name}] .npy", "GET", f"/cog/tiles/WebMercatorQuad/{z_c}/{tx_c}/{ty_c}.npy?url={q(p)}", None, "npy"),
                    (f"PARITY W33 CRS [{name}] point", "GET", f"/cog/point/{lon_c:.5f},{lat_c:.5f}?url={q(p)}", None, "point"),
                ]
            # --- Wave 34: int64/uint64 64-bit integer COGs, STAC inline pipe syntax & dst_crs/coord_crs bbox ---
            cases += [
                ("PARITY W34 DTYPE int64 tile .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w34_i64'])}", None, "npy"),
                ("PARITY W34 DTYPE int64 /point", "GET", f"/cog/point/-122.25,37.75?url={q(s['w34_i64'])}", None, "point"),
                ("PARITY W34 DTYPE int64 /statistics", "GET", f"/cog/statistics?url={q(s['w34_i64'])}", None, "stats"),
                ("BETTER W34 DTYPE uint64 tile .npy (cw 200 u64, tt 500)", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w34_u64'])}", None, "better_u64_npy"),
                ("PARITY W34 DTYPE uint64 /point", "GET", f"/cog/point/-122.25,37.75?url={q(s['w34_u64'])}", None, "point"),
                ("PARITY W34 STAC inline assets=multi|bidx=2 .npy", "GET", f"/stac/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w34_stac_pipe'])}&assets={qe('multi|bidx=2')}", None, "npy"),
                ("PARITY W34 STAC inline assets=multi|expression=b2-b1 .npy", "GET", f"/stac/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w34_stac_pipe'])}&assets={qe('multi|expression=b2-b1')}", None, "npy"),
                ("PARITY W34 STAC inline assets=multi|bidx=2 /point", "GET", f"/stac/point/-122.25,37.75?url={q(s['w34_stac_pipe'])}&assets={qe('multi|bidx=2')}", None, "point"),
                ("PARITY W34 BBOX dst_crs=EPSG:32610 cubic .npy", "GET", f"/cog/bbox/-122.4,37.6,-122.1,37.9/64x64.npy?url={q(s['w34_multi3b'])}&bidx=1&dst_crs=EPSG:32610&resampling=cubic", None, "npy"),
                ("PARITY W34 BBOX coord_crs=EPSG:32610 dst_crs=32610 .npy", "GET", f"/cog/bbox/550000,4165000,575000,4190000/64x64.npy?url={q(s['w34_multi3b'])}&bidx=1&coord_crs=EPSG:32610&dst_crs=EPSG:32610", None, "npy"),
                # --- Wave 35: unscale=true + DEM/colormap/per-band-scaled expression & ?nodata=nan query override ---
                ("PARITY W35 UNSCALE+VIRIDIS scaled DEM .png", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w35_scaled_dem'])}&unscale=true&rescale=500,1500&colormap_name=viridis", None, "png"),
                ("PARITY W35 UNSCALE+SLOPE buffer=3 scaled b1 .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w35_scaled_2b'])}&bidx=1&unscale=true&algorithm=slope&buffer=3", None, "npy"),
                ("PARITY W35 UNSCALE+TERRARIUM scaled DEM .png", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w35_scaled_dem'])}&unscale=true&algorithm=terrarium", None, "png"),
                ("PARITY W35 UNSCALE+EXPR hetero 2-band NDVI .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w35_scaled_2b'])}&unscale=true&expression={qe('(b2-b1)/(b2+b1)')}", None, "npy"),
                ("PARITY W35 UNSCALE+EXPR hetero 2-band NDVI .png", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w35_scaled_2b'])}&unscale=true&expression={qe('(b2-b1)/(b2+b1)')}&rescale=-1,1&colormap_name=viridis", None, "png"),
                ("PARITY W35 UNSCALE hetero 2-band /point", "GET", f"/cog/point/-122.25,37.75?url={q(s['w35_scaled_2b'])}&unscale=true", None, "point"),
                ("PARITY W35 UNSCALE+EXPR hetero 2-band /point", "GET", f"/cog/point/-122.25,37.75?url={q(s['w35_scaled_2b'])}&unscale=true&expression={qe('(b2-b1)/(b2+b1)')}", None, "point"),
                ("PARITY W35 NODATA=nan tagless f32 .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w35_f32_nan_notag'])}&nodata=nan", None, "npy"),
                ("PARITY W35 NODATA=nan tagless f32 .png", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w35_f32_nan_notag'])}&nodata=nan&rescale=0,400&colormap_name=viridis", None, "png"),
                ("PARITY W35 NODATA=nan tagless f32 /statistics", "GET", f"/cog/statistics?url={q(s['w35_f32_nan_notag'])}&nodata=nan", None, "stats"),
                # --- Wave 36: 5-band multispectral COG, /bbox & /feature algorithms, STAC FC stats & expr sci-notation ---
                ("PARITY W36 5-BAND CIR bidx=4,3,2 .png", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w36_5band'])}&bidx=4&bidx=3&bidx=2&rescale=0,300", None, "png"),
                ("PARITY W36 5-BAND NDVI (b4-b3)/(b4+b3) .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w36_5band'])}&expression={qe('(b4-b3)/(b4+b3)')}", None, "npy"),
                ("PARITY W36 5-BAND /statistics (b1..b5)", "GET", f"/cog/statistics?url={q(s['w36_5band'])}", None, "stats"),
                ("PARITY W36 BBOX algorithm=terrarium .png", "GET", f"/cog/bbox/-122.45,37.55,-122.05,37.95/64x64.png?url={q(s['w36_dem'])}&algorithm=terrarium", None, "png"),
                ("BETTER W36 BBOX algorithm=hillshade (cw 64x64, tt 58x58)", "GET", f"/cog/bbox/-122.45,37.55,-122.05,37.95/64x64.png?url={q(s['w36_dem'])}&algorithm=hillshade", None, "better_bbox_hillshade_64"),
                ("PARITY W36 BBOX algorithm=contours inc=100 .png", "GET", f"/cog/bbox/-122.45,37.55,-122.05,37.95/64x64.png?url={q(s['w36_dem'])}&algorithm=contours&algorithm_params={qe(json.dumps({'increment':100}))}", None, "png"),
                ("PARITY W36 POST /feature algorithm=terrarium .png", "POST", f"/cog/feature/64x64.png?url={q(s['w36_dem'])}&algorithm=terrarium", {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[[-122.45, 37.55], [-122.05, 37.55], [-122.25, 37.95], [-122.45, 37.55]]]}}, "png"),
                ("PARITY W36 POST /feature expr NDVI .npy", "POST", f"/cog/feature/64x64.npy?url={q(s['w36_5band'])}&expression={qe('(b4-b3)/(b4+b3)')}", {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[[-122.45, 37.55], [-122.05, 37.55], [-122.25, 37.95], [-122.45, 37.55]]]}}, "npy"),
                ("PARITY W36 POST /stac/statistics FC 2-feat ms|bidx=4", "POST", f"/stac/statistics?url={q(s['w36_stac5b'])}&assets={qe('ms|bidx=4')}", {"type": "FeatureCollection", "features": [
                    {"type": "Feature", "id": "a", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[[-122.45, 37.80], [-122.30, 37.80], [-122.30, 37.95], [-122.45, 37.95], [-122.45, 37.80]]]}},
                    {"type": "Feature", "id": "b", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[[-122.20, 37.55], [-122.05, 37.55], [-122.05, 37.70], [-122.20, 37.70], [-122.20, 37.55]]]}},
                ]}, "post_fc_stats_w23"),
                ("PARITY W36 EXPR sci-notation + arctan .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w36_5band'])}&expression={qe('arctan(b1*1e-2)+2.5e1')}", None, "npy"),
            ]
            # --- Wave 37: Projected GeoJSON coord_crs (3857/32610) on POST /statistics & /feature, US-feet /point (2227), /cog/info?crs=3857, /cog/bounds?crs=3857 & /cog/stac ---
            from rasterio.warp import transform as rio_transform
            w37_x3857, w37_y3857 = rio_transform("EPSG:4326", "EPSG:3857", [-122.4, -122.1, -122.1, -122.4, -122.4], [37.6, 37.6, 37.9, 37.9, 37.6])
            w37_poly_3857 = {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [list(zip(w37_x3857, w37_y3857))]}}
            w37_xutm, w37_yutm = rio_transform("EPSG:4326", "EPSG:32610", [-122.4, -122.1, -122.1, -122.4, -122.4], [37.6, 37.6, 37.9, 37.9, 37.6])
            w37_poly_utm = {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [list(zip(w37_xutm, w37_yutm))]}}
            w37_px2227, w37_py2227 = rio_transform("EPSG:4326", "EPSG:2227", [-122.25], [37.75])
            cases += [
                ("PARITY W37 POST /statistics coord_crs=EPSG:3857", "POST", f"/cog/statistics?url={q(s['w37_cog'])}&coord_crs=EPSG:3857", w37_poly_3857, "post_stats"),
                ("PARITY W37 POST /statistics coord_crs=EPSG:32610", "POST", f"/cog/statistics?url={q(s['w37_cog'])}&coord_crs=EPSG:32610", w37_poly_utm, "post_stats"),
                ("PARITY W37 POST /feature coord_crs=EPSG:3857 .npy", "POST", f"/cog/feature/64x64.npy?url={q(s['w37_cog'])}&coord_crs=EPSG:3857", w37_poly_3857, "npy"),
                ("PARITY W37 POST /feature coord_crs=32610 dst_crs=32610 .npy", "POST", f"/cog/feature/64x64.npy?url={q(s['w37_cog'])}&coord_crs=EPSG:32610&dst_crs=EPSG:32610", w37_poly_utm, "npy"),
                ("PARITY W37 /point coord_crs=EPSG:2227 (US Survey Feet)", "GET", f"/cog/point/{w37_px2227[0]:.2f},{w37_py2227[0]:.2f}?url={q(s['w37_cog'])}&coord_crs=EPSG:2227", None, "point"),
                ("BETTER W37 GET /cog/info?crs=EPSG:3857 (cw 3857, tt ignores)", "GET", f"/cog/info?url={q(s['w37_cog'])}&crs=EPSG:3857", None, "better_info_crs3857"),
                ("BETTER W37 GET /cog/bounds?crs=EPSG:3857 (cw 200, tt 404)", "GET", f"/cog/bounds?url={q(s['w37_cog'])}&crs=EPSG:3857", None, "better_bounds_crs3857"),
                ("BETTER W37 GET /cog/stac item JSON (cw 200, tt 404)", "GET", f"/cog/stac?url={q(s['w37_cog'])}", None, "better_cog_stac"),
                # --- Wave 38: High-precision uint32 (>2^24) & int32 (<-2^24) + OGC/TiTiler discovery endpoints ---
                ("PARITY W38 DTYPE u32 >2^24 exact .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w38_u32_hi'])}", None, "npy"),
                ("PARITY W38 DTYPE u32 >2^24 exact .tif", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.tif?url={q(s['w38_u32_hi'])}", None, "geotiff"),
                ("PARITY W38 DTYPE u32 >2^24 /point", "GET", f"/cog/point/-122.25,37.75?url={q(s['w38_u32_hi'])}", None, "point"),
                ("PARITY W38 DTYPE u32 >2^24 /statistics", "GET", f"/cog/statistics?url={q(s['w38_u32_hi'])}", None, "stats"),
                ("PARITY W38 DTYPE i32 <-2^24+nodata .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w38_i32_hi'])}", None, "npy"),
                ("PARITY W38 DTYPE i32 <-2^24+nodata /point", "GET", f"/cog/point/-122.25,37.75?url={q(s['w38_i32_hi'])}", None, "point"),
                ("PARITY W38 DISCOVERY /tileMatrixSets & /algorithms", "GET", "/algorithms", None, "discovery_w38"),
                ("PARITY W38 DISCOVERY /colorMaps", "GET", "/colorMaps", None, "discovery_w38"),
                # --- Wave 39: High-precision float64 (>1e9, f32 ULP=64), OGC /cog/tiles metadata & STAC expr feature/stats ---
                ("PARITY W39 DTYPE f64 >1e9 .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w39_f64_hi'])}", None, "npy_f64_exact"),
                ("PARITY W39 DTYPE f64 >1e9 bilinear .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w39_f64_hi'])}&resampling=bilinear", None, "npy_f64_exact"),
                ("PARITY W39 DTYPE f64 >1e9 .tif", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.tif?url={q(s['w39_f64_hi'])}", None, "geotiff_f64_exact"),
                ("PARITY W39 DTYPE f64 >1e9 /point", "GET", f"/cog/point/-122.25,37.75?url={q(s['w39_f64_hi'])}", None, "point_f64_exact"),
                ("PARITY W39 DTYPE f64 >1e9 /statistics", "GET", f"/cog/statistics?url={q(s['w39_f64_hi'])}", None, "stats"),
                ("PARITY W39 OGC /cog/tiles tilesets list", "GET", f"/cog/tiles?url={q(s['w39_f64_hi'])}", None, "ogc_tiles_w39"),
                ("PARITY W39 OGC /cog/tiles/WebMercatorQuad tileset", "GET", f"/cog/tiles/WebMercatorQuad?url={q(s['w39_f64_hi'])}", None, "ogc_tiles_w39"),
                ("PARITY W39 STAC POST /feature NDVI .npy", "POST", f"/stac/feature/64x64.npy?url={q(s['w29_stac_ab'])}&assets=red&assets=nir&expression={qe('(b2-b1)/(b2+b1)')}", {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[[-122.45, 37.55], [-122.05, 37.55], [-122.25, 37.95], [-122.45, 37.55]]]}}, "npy"),
                ("PARITY W39 STAC POST /statistics NDVI", "POST", f"/stac/statistics?url={q(s['w29_stac_ab'])}&assets=red&assets=nir&expression={qe('(b2-b1)/(b2+b1)')}", {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[[-122.45, 37.55], [-122.05, 37.55], [-122.25, 37.95], [-122.45, 37.55]]]}}, "post_stats"),
                # --- Wave 40: Unsized /preview, /bbox & POST /feature (max_size/width/height on 2:1 non-square COG) + algorithm=aspect ---
                ("PARITY W40 UNSIZED /cog/preview.png max_size=64", "GET", f"/cog/preview.png?url={q(s['w40_rect_dem'])}&max_size=64&rescale=200,1500&colormap_name=viridis", None, "png"),
                ("PARITY W40 UNSIZED /cog/bbox .png max_size=64", "GET", f"/cog/bbox/-122.4,37.6,-121.6,37.9.png?url={q(s['w40_rect_dem'])}&max_size=64&rescale=200,1500&colormap_name=viridis", None, "png"),
                ("PARITY W40 UNSIZED /cog/bbox .npy width=80&height=40", "GET", f"/cog/bbox/-122.4,37.6,-121.6,37.9.npy?url={q(s['w40_rect_dem'])}&width=80&height=40", None, "npy"),
                ("PARITY W40 UNSIZED POST /cog/feature.png max_size=128", "POST", f"/cog/feature.png?url={q(s['w40_rect_dem'])}&max_size=128&rescale=200,1500&colormap_name=viridis", {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[[-122.4, 37.55], [-121.6, 37.55], [-122.0, 37.95], [-122.4, 37.55]]]}}, "png"),
                ("PARITY W40 UNSIZED POST /cog/feature.npy w=48&h=48", "POST", f"/cog/feature.npy?url={q(s['w40_rect_dem'])}&width=48&height=48", {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[[-122.4, 37.55], [-121.6, 37.55], [-122.0, 37.95], [-122.4, 37.55]]]}}, "npy"),
                ("BETTER W40 ALG aspect .npy (cw 200 0..360 deg, tt 422)", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w40_rect_dem'])}&algorithm=aspect", None, "better_aspect_w40"),
                ("BETTER W40 ALG aspect&buffer=3 .npy (cw 262x262, tt 422)", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w40_rect_dem'])}&algorithm=aspect&buffer=3", None, "better_aspect_w40"),
                ("BETTER W40 ALG aspect twilight .png (cw 200, tt 422)", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w40_rect_dem'])}&algorithm=aspect&rescale=0,360&colormap_name=twilight", None, "better_aspect_w40"),
                # --- Wave 41: All 7 multi-band reduction algorithms (min/max/mean/std/var/median/sum) + u16 sum>65535 promotion ---
                ("PARITY W41 ALG 4-band f32 min .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w41_f32_4b'])}&algorithm=min", None, "npy"),
                ("PARITY W41 ALG 4-band f32 max .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w41_f32_4b'])}&algorithm=max", None, "npy"),
                ("PARITY W41 ALG 4-band f32 mean .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w41_f32_4b'])}&algorithm=mean", None, "npy"),
                ("PARITY W41 ALG 4-band f32 std .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w41_f32_4b'])}&algorithm=std", None, "npy"),
                ("PARITY W41 ALG 4-band f32 var .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w41_f32_4b'])}&algorithm=var", None, "npy"),
                ("PARITY W41 ALG 4-band f32 median (even) .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w41_f32_4b'])}&algorithm=median", None, "npy"),
                ("PARITY W41 ALG 4-band f32 sum .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w41_f32_4b'])}&algorithm=sum", None, "npy"),
                ("BETTER W41 ALG 3-band u16 sum >65535 (cw 200 u64, tt 500)", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w41_u16_3b_hi'])}&algorithm=sum", None, "better_u16_sum_u64"),
                ("PARITY W41 ALG 3-band u16 median (odd) .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w41_u16_3b_hi'])}&algorithm=median", None, "npy"),
                ("PARITY W41 ALG 3-band u16 min .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w41_u16_3b_hi'])}&algorithm=min", None, "npy"),
                # --- Wave 42: IEEE-754 float16 GeoTIFF (Predictor=1 & Predictor=3), bidx+algorithm composition, POST /statistics expression+unscale ---
                ("PARITY W42 FLOAT16 tile .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w42_f16'])}", None, "npy"),
                ("PARITY W42 FLOAT16 tile .png (viridis)", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w42_f16'])}&rescale=-10,40&colormap_name=viridis", None, "png"),
                ("PARITY W42 FLOAT16 /point", "GET", f"/cog/point/-122.25,37.75?url={q(s['w42_f16'])}", None, "point"),
                ("PARITY W42 FLOAT16 /statistics", "GET", f"/cog/statistics?url={q(s['w42_f16'])}", None, "stats"),
                ("PARITY W42 FLOAT16 Predictor=3 tile .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w42_f16_p3'])}", None, "npy"),
                ("PARITY W42 FLOAT16 Predictor=3 /point", "GET", f"/cog/point/-122.25,37.75?url={q(s['w42_f16_p3'])}", None, "point"),
                ("PARITY W42 bidx=1,3 + algorithm=mean .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w41_f32_4b'])}&bidx=1&bidx=3&algorithm=mean", None, "npy"),
                ("PARITY W42 bidx=1,3 + algorithm=normalizedIndex .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w41_f32_4b'])}&bidx=1&bidx=3&algorithm=normalizedIndex", None, "npy"),
                ("PARITY W42 bidx=1,3 + algorithm=normalizedIndex .png", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w41_f32_4b'])}&bidx=1&bidx=3&algorithm=normalizedIndex&rescale=-1,1&colormap_name=rdylgn", None, "png"),
                ("PARITY W42 POST /statistics expression=(b3-b1)/(b3+b1)", "POST", f"/cog/statistics?url={q(s['w41_f32_4b'])}&expression={qe('(b3-b1)/(b3+b1)')}", {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[[-122.4, 37.6], [-122.1, 37.6], [-122.1, 37.9], [-122.4, 37.9], [-122.4, 37.6]]]}}, "post_stats"),
                ("PARITY W42 POST /statistics unscale+expression NDVI", "POST", f"/cog/statistics?url={q(s['w35_scaled_2b'])}&unscale=true&expression={qe('(b2-b1)/(b2+b1)')}", {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[[-122.4, 37.6], [-122.1, 37.6], [-122.1, 37.9], [-122.4, 37.9], [-122.4, 37.6]]]}}, "post_stats"),
                # --- Wave 43: 3-band RGB ?nodata=0 .npy (all-band dataset_mask), dual resampling=+reproject=, stripped multi-band TIFF with internal overviews ---
                ("PARITY W43 RGB nodata=0 .npy (all-band dataset_mask)", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w43_rgb_nodata'])}&nodata=0", None, "npy"),
                ("PARITY W43 DUAL resampling=average&reproject=cubic .npy", "GET", f"/cog/tiles/WebMercatorQuad/9/82/198.npy?url={q(s['w43_multi_ovr'])}&resampling=average&reproject=cubic", None, "npy"),
                ("PARITY W43 DUAL resampling=nearest&reproject=bilinear .npy", "GET", f"/cog/tiles/WebMercatorQuad/9/82/198.npy?url={q(s['w43_multi_ovr'])}&resampling=nearest&reproject=bilinear", None, "npy"),
                ("PARITY W43 DUAL resampling=bilinear&reproject=nearest .npy", "GET", f"/cog/tiles/WebMercatorQuad/9/82/198.npy?url={q(s['w43_multi_ovr'])}&resampling=bilinear&reproject=nearest", None, "npy"),
                ("PARITY W43 STRIPPED+OVR z10 full-res .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w43_stripped_ovr'])}", None, "npy"),
                ("PARITY W43 STRIPPED+OVR z8 overview .npy", "GET", f"/cog/tiles/WebMercatorQuad/8/41/99.npy?url={q(s['w43_stripped_ovr'])}", None, "npy"),
                ("PARITY W43 STRIPPED+OVR z10 full-res .png", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w43_stripped_ovr'])}", None, "png"),
                ("PARITY W43 STRIPPED+OVR /point", "GET", f"/cog/point/-122.25,37.75?url={q(s['w43_stripped_ovr'])}", None, "point"),
                ("PARITY W43 STRIPPED+OVR /statistics", "GET", f"/cog/statistics?url={q(s['w43_stripped_ovr'])}", None, "stats"),
                # --- Wave 44: Chained & per-channel color_formula on 3-band RGB ---
                ("PARITY W44 color_formula chained Gamma+Sigmoidal+Sat .png", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w43_rgb_nodata'])}&color_formula={qe('Gamma RGB 1.5 Sigmoidal RGB 15 0.35 Saturation 1.3')}", None, "png"),
                ("PARITY W44 color_formula per-channel Gamma R/B .png", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w43_rgb_nodata'])}&color_formula={qe('Gamma R 1.8 Gamma B 0.7')}", None, "png"),
                ("PARITY W44 color_formula per-channel Sigmoidal G .png", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w43_rgb_nodata'])}&color_formula={qe('Sigmoidal G 10 0.5')}", None, "png"),
                # --- Wave 45: 16-bit unrescaled uint16 PNG, discrete RGBA JSON colormap, cast(output_dtype), ceil/floor/grayscale/terrainrgb/contours, expr+alg ---
                ("PARITY W45 16-bit unrescaled uint16 .png", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w45_u16'])}", None, "png"),
                ("PARITY W45 discrete RGBA JSON colormap (alpha 0/128/255)", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w45_cat'])}&colormap={qe(json.dumps({'1':[255,0,0,255],'2':[0,255,0,128],'3':[0,0,255,0]}))}", None, "png"),
                ("PARITY W45 ALG ceil .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w45_dem'])}&algorithm=ceil", None, "npy"),
                ("PARITY W45 ALG floor .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w45_dem'])}&algorithm=floor", None, "npy"),
                ("PARITY W45 ALG cast default uint8 .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w45_dem'])}&algorithm=cast", None, "npy"),
                ("BETTER W45 ALG cast output_dtype=int16 (cw i16, tt u8)", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w45_dem'])}&algorithm=cast&algorithm_params={qe(json.dumps({'output_dtype':'int16'}))}", None, "better_cast_int16"),
                ("PARITY W45 ALG grayscale (3-band RGB) .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w43_rgb_nodata'])}&algorithm=grayscale", None, "npy"),
                ("PARITY W45 ALG grayscale (3-band RGB) .png", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w43_rgb_nodata'])}&algorithm=grayscale", None, "png"),
                ("PARITY W45 ALG terrainrgb .png", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w45_dem'])}&algorithm=terrainrgb", None, "png"),
                ("PARITY W45 ALG contours custom (inc=50, thick=2) .png", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w45_dem'])}&algorithm=contours&algorithm_params={qe(json.dumps({'increment':50,'thickness':2}))}", None, "png"),
                ("PARITY W45 EXPR+ALG expression=b1*0.5+10 & ceil .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w45_dem'])}&expression={qe('b1*0.5+10')}&algorithm=ceil", None, "npy"),
                ("PARITY W45 EXPR+ALG expression=b1*2.0 & terrarium .png", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w45_dem'])}&expression={qe('b1*2.0')}&algorithm=terrarium", None, "png"),
                # --- Wave 46: LERC / LERC_DEFLATE (int16) / LERC_ZSTD (lossy f32) + 6 projected CRS families ---
                ("PARITY W46 LERC lossless f32 + hole .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w46_lerc_f32'])}", None, "npy"),
                ("PARITY W46 LERC_DEFLATE int16 + hole .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w46_lerc_deflate_i16'])}", None, "npy"),
                ("PARITY W46 LERC_DEFLATE int16 /statistics", "GET", f"/cog/statistics?url={q(s['w46_lerc_deflate_i16'])}", None, "stats"),
                ("PARITY W46 LERC_ZSTD lossy f32 + hole .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w46_lerc_zstd_lossy'])}", None, "npy"),
                ("PARITY W46 LERC_ZSTD lossy f32 /point", "GET", f"/cog/point/-122.25,37.75?url={q(s['w46_lerc_zstd_lossy'])}", None, "point"),
            ]
            for tag, desc, lon_c, lat_c in [
                ("5070", "EPSG:5070 Conus Albers", -97.5, 38.5),
                ("2154", "EPSG:2154 Lambert-93 LCC", 2.5, 48.5),
                ("3035", "EPSG:3035 LAEA Europe", 10.5, 50.5),
                ("6933", "EPSG:6933 EASE-Grid CEA", 15.5, 20.5),
                ("3031", "EPSG:3031 Antarctic Stereo", 1.0, -75.5),
                ("27700", "EPSG:27700 British Natl Grid", -1.0, 52.0),
            ]:
                t46 = mercantile.tile(lon_c, lat_c, 9)
                cases += [
                    (f"PARITY W46 CRS {desc} .npy", "GET", f"/cog/tiles/WebMercatorQuad/{t46.z}/{t46.x}/{t46.y}.npy?url={q(s[f'w46_crs_{tag}'])}", None, "npy"),
                    (f"PARITY W46 CRS {desc} /point", "GET", f"/cog/point/{lon_c},{lat_c}?url={q(s[f'w46_crs_{tag}'])}", None, "point"),
                ]
            # --- Wave 47: Antimeridian-crossing COG (179E..181E) + 6 exotic CRS families ---
            tw47 = mercantile.tile(179.5, -17.0, 9)
            te47 = mercantile.tile(-179.5, -17.0, 9)
            cases += [
                ("PARITY W47 ANTIMERIDIAN west 179.5E tile .npy", "GET", f"/cog/tiles/WebMercatorQuad/{tw47.z}/{tw47.x}/{tw47.y}.npy?url={q(s['w47_antimeridian'])}", None, "npy"),
                ("PARITY W47 ANTIMERIDIAN west 179.5E /point", "GET", f"/cog/point/179.5,-17.0?url={q(s['w47_antimeridian'])}", None, "point"),
                ("BETTER W47 ANTIMERIDIAN east -179.5W tile .npy (cw 200, tt 404)", "GET", f"/cog/tiles/WebMercatorQuad/{te47.z}/{te47.x}/{te47.y}.npy?url={q(s['w47_antimeridian'])}", None, "better_antimeridian_w47"),
                ("BETTER W47 ANTIMERIDIAN east -179.5W /point (cw 200, tt 404)", "GET", f"/cog/point/-179.5,-17.0?url={q(s['w47_antimeridian'])}", None, "better_antimeridian_w47"),
            ]
            for tag, desc, lon_c, lat_c in [
                ("8857", "EPSG:8857 Equal Earth", 10.5, 20.5),
                ("3413", "EPSG:3413 Polar Stereo N", -44.0, 72.5),
                ("3577", "EPSG:3577 Australian Albers", 135.5, -25.5),
                ("28992", "EPSG:28992 Dutch RD New", 5.5, 52.25),
                ("2056", "EPSG:2056 Swiss LV95", 8.0, 46.75),
                ("4269", "EPSG:4269 NAD83 Geog", -105.0, 40.0),
            ]:
                t47 = mercantile.tile(lon_c, lat_c, 9)
                cases += [
                    (f"PARITY W47 CRS {desc} .npy", "GET", f"/cog/tiles/WebMercatorQuad/{t47.z}/{t47.x}/{t47.y}.npy?url={q(s[f'w47_crs_{tag}'])}", None, "npy"),
                    (f"PARITY W47 CRS {desc} /point", "GET", f"/cog/point/{lon_c},{lat_c}?url={q(s[f'w47_crs_{tag}'])}", None, "point"),
                ]
            # --- Wave 48: Expression VM deep math/conditionals/div-by-zero + STAC algorithm/feature-3857/bbox-expr ---
            cases += [
                ("PARITY W48 EXPR where(b1>120,b2,b3) .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w48_expr3b'])}&expression={qe('where(b1>120,b2,b3)')}", None, "npy"),
                ("PARITY W48 EXPR log10(b1)+log(b2) .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w48_expr3b'])}&expression={qe('log10(b1)+log(b2)')}", None, "npy"),
                ("PARITY W48 EXPR exp(b1*0.01)+abs(b2-120) .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w48_expr3b'])}&expression={qe('exp(b1*0.01)+abs(b2-120)')}", None, "npy"),
                ("PARITY W48 EXPR sin(b1*0.02)+cos(b2*0.02) .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w48_expr3b'])}&expression={qe('sin(b1*0.02)+cos(b2*0.02)')}", None, "npy"),
                ("PARITY W48 EXPR minimum(b1,b2)+maximum(b2,b3) .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w48_expr3b'])}&expression={qe('minimum(b1,b2)+maximum(b2,b3)')}", None, "npy"),
                ("BETTER W48 EXPR div-by-zero masked (cw clean, tt leaks)", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w48_expr3b'])}&expression={qe('b2/where(b1>130,b1,0)')}", None, "better_div0_w48"),
                ("PARITY W48 STAC tile algorithm=mean .npy", "GET", f"/stac/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w36_stac5b'])}&assets=ms&algorithm=mean", None, "npy"),
                ("PARITY W48 STAC POST /feature coord_crs=EPSG:3857 .npy", "POST", f"/stac/feature/64x64.npy?url={q(s['w36_stac5b'])}&assets=ms&coord_crs=EPSG:3857", w37_poly_3857, "npy"),
                ("BETTER W48 STAC /bbox expr auto-asset (cw 200, tt 422)", "GET", f"/stac/bbox/-122.4,37.6,-122.1,37.9/64x64.npy?url={q(s['w36_stac5b'])}&expression={qe('(ms_b4-ms_b3)/(ms_b4+ms_b3)')}", None, "better_stac_bbox_auto_asset"),
                # --- Wave 49: Issue #602 RGB scalar nodata .png (0/255 + bilinear), 2-band Gray+Alpha (ExtraSamples=2), semicolon multi-band expressions ---
                ("PARITY W49 RGB nodata=0 .png (all-band dataset_mask)", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w43_rgb_nodata'])}&nodata=0", None, "png"),
                ("PARITY W49 RGB nodata=0 bilinear .png (all-band mask)", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w43_rgb_nodata'])}&nodata=0&resampling=bilinear", None, "png"),
                ("PARITY W49 RGB nodata=255 .png (all-band dataset_mask)", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w49_rgb_nd255'])}&nodata=255", None, "png"),
                ("PARITY W49 GRAY+ALPHA (ExtraSamples=2) .png", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w49_gray_alpha'])}", None, "png"),
                ("PARITY W49 GRAY+ALPHA (ExtraSamples=2) .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w49_gray_alpha'])}", None, "npy"),
                ("PARITY W49 GRAY+ALPHA (ExtraSamples=2) /statistics", "GET", f"/cog/statistics?url={q(s['w49_gray_alpha'])}", None, "stats"),
                ("PARITY W49 GRAY+ALPHA (ExtraSamples=2) /point", "GET", f"/cog/point/-122.25,37.75?url={q(s['w49_gray_alpha'])}", None, "point"),
                ("PARITY W49 EXPR 3-band ; multi-output .npy (f64)", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w43_rgb_nodata'])}&expression={qe('b1*0.5;b2*0.5;b3*0.5')}", None, "npy"),
                ("PARITY W49 EXPR 3-band ; multi-output .png (RGB)", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w43_rgb_nodata'])}&expression={qe('b1*0.5;b2*0.5;b3*0.5')}&rescale=0,128", None, "png"),
                ("PARITY W49 EXPR 2-band ; multi-output .npy (i32)", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w43_rgb_nodata'])}&expression={qe('b1+b2;b3-b1')}", None, "npy"),
                # --- Wave 50: Donut Polygon (interior hole ring), MultiPolygon (2 disjoint squares), POST /statistics params ---
                ("PARITY W50 DONUT Polygon (interior hole) POST /statistics", "POST", f"/cog/statistics?url={q(s['w50_donut'])}", {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[[-122.45, 37.55], [-122.05, 37.55], [-122.05, 37.95], [-122.45, 37.95], [-122.45, 37.55]], [[-122.33, 37.67], [-122.33, 37.83], [-122.17, 37.83], [-122.17, 37.67], [-122.33, 37.67]]]}}, "post_stats"),
                ("PARITY W50 DONUT Polygon (interior hole) POST /feature .npy", "POST", f"/cog/feature/64x64.npy?url={q(s['w50_donut'])}", {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[[-122.45, 37.55], [-122.05, 37.55], [-122.05, 37.95], [-122.45, 37.95], [-122.45, 37.55]], [[-122.33, 37.67], [-122.33, 37.83], [-122.17, 37.83], [-122.17, 37.67], [-122.33, 37.67]]]}}, "npy"),
                ("PARITY W50 DONUT Polygon (interior hole) POST /feature .png", "POST", f"/cog/feature/64x64.png?url={q(s['w50_donut'])}&rescale=0,150&colormap_name=viridis", {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[[-122.45, 37.55], [-122.05, 37.55], [-122.05, 37.95], [-122.45, 37.95], [-122.45, 37.55]], [[-122.33, 37.67], [-122.33, 37.83], [-122.17, 37.83], [-122.17, 37.67], [-122.33, 37.67]]]}}, "png"),
                ("PARITY W50 MultiPolygon (2 disjoint) POST /statistics", "POST", f"/cog/statistics?url={q(s['w50_donut'])}", {"type": "Feature", "properties": {}, "geometry": {"type": "MultiPolygon", "coordinates": [[[[-122.46, 37.54], [-122.36, 37.54], [-122.36, 37.64], [-122.46, 37.64], [-122.46, 37.54]]], [[[-122.14, 37.86], [-122.04, 37.86], [-122.04, 37.96], [-122.14, 37.96], [-122.14, 37.86]]]]}}, "post_stats"),
                ("PARITY W50 MultiPolygon (2 disjoint) POST /feature .npy", "POST", f"/cog/feature/64x64.npy?url={q(s['w50_donut'])}", {"type": "Feature", "properties": {}, "geometry": {"type": "MultiPolygon", "coordinates": [[[[-122.46, 37.54], [-122.36, 37.54], [-122.36, 37.64], [-122.46, 37.64], [-122.46, 37.54]]], [[[-122.14, 37.86], [-122.04, 37.86], [-122.04, 37.96], [-122.14, 37.96], [-122.14, 37.86]]]]}}, "npy"),
                ("PARITY W50 POST /statistics p=25,75 & histogram_bins=6", "POST", f"/cog/statistics?url={q(s['w50_donut'])}&p=25&p=75&histogram_bins=6", {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[[-122.45, 37.55], [-122.05, 37.55], [-122.05, 37.95], [-122.45, 37.95], [-122.45, 37.55]], [[-122.33, 37.67], [-122.33, 37.83], [-122.17, 37.83], [-122.17, 37.67], [-122.33, 37.67]]]}}, "post_stats"),
                # --- Wave 51: Big-Endian (MM) Predictor=2/3, all_touched=true exact coverage, per-band rescale + color_formula on /bbox & /feature ---
                ("PARITY W51 Big-Endian int16 Predictor=2 + hole .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w51_be_i16_p2'])}", None, "npy"),
                ("PARITY W51 Big-Endian int16 Predictor=2 /statistics", "GET", f"/cog/statistics?url={q(s['w51_be_i16_p2'])}", None, "stats"),
                ("PARITY W51 Big-Endian float32 Predictor=3 + hole .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w51_be_f32_p3'])}", None, "npy"),
                ("PARITY W51 Big-Endian float32 Predictor=3 /point", "GET", f"/cog/point/-122.25,37.75?url={q(s['w51_be_f32_p3'])}", None, "point"),
                ("PARITY W51 POST /statistics all_touched=true", "POST", f"/cog/statistics?url={q(s['w51_be_f32_p3'])}&all_touched=true", {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[[-122.40, 37.60], [-122.10, 37.60], [-122.10, 37.90], [-122.40, 37.90], [-122.40, 37.60]]]}}, "post_stats"),
                ("PARITY W51 GET /bbox per-band rescale + color_formula .png", "GET", f"/cog/bbox/-122.4,37.6,-122.1,37.9/64x64.png?url={q(s['w43_rgb_nodata'])}&rescale=0,200&rescale=0,220&rescale=0,240&color_formula={qe('Gamma RGB 1.4 Saturation 1.2')}", None, "png"),
                ("PARITY W51 POST /feature per-band rescale + color_formula .png", "POST", f"/cog/feature/64x64.png?url={q(s['w43_rgb_nodata'])}&rescale=0,200&rescale=0,220&rescale=0,240&color_formula={qe('Gamma RGB 1.4 Saturation 1.2')}", {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[[-122.40, 37.60], [-122.10, 37.60], [-122.10, 37.90], [-122.40, 37.90], [-122.40, 37.60]]]}}, "png"),
                # --- Wave 52: 16-bit unrescaled 3-band uint16 RGBA PNG & .npy, STAC POST /statistics coord_crs=3857 + inline pipe expr ---
                ("PARITY W52 16-bit unrescaled 3-band uint16 RGBA .png", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w52_u16_rgb'])}", None, "png"),
                ("PARITY W52 16-bit 3-band uint16 .npy + nodata=0 hole", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w52_u16_rgb'])}", None, "npy"),
                ("PARITY W52 STAC POST /statistics coord_crs=EPSG:3857", "POST", f"/stac/statistics?url={q(s['w36_stac5b'])}&assets={qe('ms|bidx=4')}&coord_crs=EPSG:3857", w37_poly_3857, "post_stats"),
                ("PARITY W52 STAC POST /statistics inline expr + 3857", "POST", f"/stac/statistics?url={q(s['w36_stac5b'])}&assets={qe('ms|expression=(b4-b3)/(b4+b3)')}&coord_crs=EPSG:3857", w37_poly_3857, "post_stats"),
                # --- Wave 53: Negative scale unscale=true, WebP lossless RGB, ZSTD float64 Predictor=3 (>1e9 strict) ---
                ("PARITY W53 NEG-SCALE unscale=true .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w53_neg_scale'])}&unscale=true", None, "npy"),
                ("PARITY W53 NEG-SCALE unscale=true viridis .png", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w53_neg_scale'])}&unscale=true&rescale=25,90&colormap_name=viridis", None, "png"),
                ("PARITY W53 NEG-SCALE unscale=true /point", "GET", f"/cog/point/-122.25,37.75?url={q(s['w53_neg_scale'])}&unscale=true", None, "point"),
                ("PARITY W53 NEG-SCALE unscale=true /statistics", "GET", f"/cog/statistics?url={q(s['w53_neg_scale'])}&unscale=true", None, "stats"),
                ("PARITY W53 WEBP lossless 3-band RGB .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w53_webp_lossless'])}", None, "npy"),
                ("PARITY W53 ZSTD float64 Predictor=3 (>1e9) .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w53_zstd_f64_p3'])}", None, "npy_f64_exact"),
                # --- Wave 54: WebMercator-aligned JPEG-in-TIFF PHOTOMETRIC=RGB & YCBCR with shared tag 347 JPEGTables ---
                ("PARITY W54 RGB JPEG+JPEGTables .png", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w54_rgb_jpeg_tables'])}", None, "png"),
                ("PARITY W54 RGB JPEG+JPEGTables .jpg", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.jpg?url={q(s['w54_rgb_jpeg_tables'])}", None, "png"),
                ("PARITY W54 RGB JPEG+JPEGTables /point", "GET", f"/cog/point/{(mercantile.bounds(164, 396, 10).west + mercantile.bounds(164, 396, 10).east) / 2:.5f},{(mercantile.bounds(164, 396, 10).south + mercantile.bounds(164, 396, 10).north) / 2:.5f}?url={q(s['w54_rgb_jpeg_tables'])}", None, "point"),
                ("PARITY W54 YCbCr JPEG+JPEGTables .png", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w54_ycbcr_jpeg_tables'])}", None, "png"),
                ("PARITY W54 YCbCr JPEG+JPEGTables .jpg", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.jpg?url={q(s['w54_ycbcr_jpeg_tables'])}", None, "png"),
                # --- Wave 55: Float32 1-bit internal mask (no nodata), 4:1 non-square pixel aspect, multi-overview unscale ---
                ("PARITY W55 F32 internal-mask (no nodata) .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w55_f32_imask'])}", None, "npy"),
                ("PARITY W55 F32 internal-mask bilinear viridis .png", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w55_f32_imask'])}&resampling=bilinear&rescale=10,300&colormap_name=viridis", None, "png"),
                ("PARITY W55 F32 internal-mask /point valid", "GET", f"/cog/point/-122.25,37.75?url={q(s['w55_f32_imask'])}", None, "point"),
                ("PARITY W55 F32 internal-mask /point inside hole", "GET", f"/cog/point/-122.20,37.65?url={q(s['w55_f32_imask'])}", None, "point"),
                ("PARITY W55 F32 internal-mask /statistics", "GET", f"/cog/statistics?url={q(s['w55_f32_imask'])}", None, "stats"),
                ("PARITY W55 F32 internal-mask expr b1*2+10 .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w55_f32_imask'])}&expression={qe('b1*2+10')}", None, "npy"),
                ("PARITY W55 NON-SQUARE 4:1 pixel aspect .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w55_nonsquare_4to1'])}", None, "npy"),
                ("PARITY W55 NON-SQUARE 4:1 cubic viridis .png", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w55_nonsquare_4to1'])}&resampling=cubic&rescale=50,450&colormap_name=viridis", None, "png"),
                ("PARITY W55 NON-SQUARE 4:1 /point", "GET", f"/cog/point/-122.25,37.75?url={q(s['w55_nonsquare_4to1'])}", None, "point"),
                ("PARITY W55 MULTI-OVR unscale=true z10 bilinear .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w55_multi_ovr_unscale'])}&unscale=true&resampling=bilinear", None, "npy"),
                ("PARITY W55 MULTI-OVR unscale=true z8 bilinear .npy", "GET", f"/cog/tiles/WebMercatorQuad/8/41/99.npy?url={q(s['w55_multi_ovr_unscale'])}&unscale=true&resampling=bilinear", None, "npy"),
                # --- Wave 56: ?nodata= override + unscale=true, prime-dim 137x113 rectangular 128x32 partial edge tiles (P2/P3), WorldCRS84Quad expr+bilinear ---
                ("PARITY W56 NODATA+UNSCALE ?nodata=1000&unscale=true .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w56_scaled_no_nd'])}&nodata=1000&unscale=true", None, "npy"),
                ("PARITY W56 NODATA+UNSCALE ?nodata=1000&unscale=true .png", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w56_scaled_no_nd'])}&nodata=1000&unscale=true&rescale=-30,50&colormap_name=viridis", None, "png"),
                ("PARITY W56 NODATA+UNSCALE ?nodata=1000&unscale=true /point hole", "GET", f"/cog/point/-122.20,37.65?url={q(s['w56_scaled_no_nd'])}&nodata=1000&unscale=true", None, "point"),
                ("PARITY W56 NODATA+UNSCALE ?nodata=1000&unscale=true /statistics", "GET", f"/cog/statistics?url={q(s['w56_scaled_no_nd'])}&nodata=1000&unscale=true", None, "stats"),
                ("PARITY W56 PRIME 137x113 128x32 i16 ZSTD P2 .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w56_prime_i16_p2'])}", None, "npy"),
                ("PARITY W56 PRIME 137x113 128x32 i16 ZSTD P2 /statistics", "GET", f"/cog/statistics?url={q(s['w56_prime_i16_p2'])}", None, "stats"),
                ("PARITY W56 PRIME 137x113 128x32 f32 DEFLATE P3 .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w56_prime_f32_p3'])}", None, "npy"),
                ("PARITY W56 PRIME 137x113 128x32 f32 DEFLATE P3 /point", "GET", f"/cog/point/-122.25,37.75?url={q(s['w56_prime_f32_p3'])}", None, "point"),
                ("PARITY W56 TMS WorldCRS84Quad expr+bilinear .npy", "GET", f"/cog/tiles/WorldCRS84Quad/11/652/593.npy?url={q(sat)}&expression={qe('(b3-b1)/(b3+b1+0.01)')}&resampling=bilinear", None, "npy"),
                ("PARITY W56 TMS WorldCRS84Quad expr+bilinear viridis .png", "GET", f"/cog/tiles/WorldCRS84Quad/11/652/593.png?url={q(sat)}&expression={qe('(b3-b1)/(b3+b1+0.01)')}&resampling=bilinear&rescale=-1,1&colormap_name=viridis", None, "png"),
                # --- Wave 57: Sparse GeoTIFFs (SPARSE_OK=TRUE) with & without nodata, per-band heterogeneous (scale, offset) ---
                ("PARITY W57 SPARSE TIFF nodata=-9999 .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w57_sparse_nodata'])}", None, "npy"),
                ("PARITY W57 SPARSE TIFF nodata=-9999 bilinear .png", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w57_sparse_nodata'])}&resampling=bilinear&rescale=20,260&colormap_name=viridis", None, "png"),
                ("PARITY W57 SPARSE TIFF nodata=-9999 /point sparse tile", "GET", f"/cog/point/-122.40,37.60?url={q(s['w57_sparse_nodata'])}", None, "point"),
                ("PARITY W57 SPARSE TIFF nodata=-9999 /statistics", "GET", f"/cog/statistics?url={q(s['w57_sparse_nodata'])}", None, "stats"),
                ("PARITY W57 SPARSE TIFF no-nodata (valid 0s) .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w57_sparse_no_nd'])}", None, "npy"),
                ("PARITY W57 SPARSE TIFF no-nodata (valid 0s) /statistics", "GET", f"/cog/statistics?url={q(s['w57_sparse_no_nd'])}", None, "stats"),
                ("PARITY W57 PER-BAND scale/offset 3b unscale=true .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w57_perband_scale_3b'])}&unscale=true", None, "npy"),
                ("PARITY W57 PER-BAND scale/offset 3b unscale+expr .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w57_perband_scale_3b'])}&unscale=true&expression={qe('(b3-b2)/(b1+1)')}", None, "npy"),
                ("PARITY W57 PER-BAND scale/offset 3b unscale /point", "GET", f"/cog/point/-122.25,37.75?url={q(s['w57_perband_scale_3b'])}&unscale=true", None, "point"),
                ("PARITY W57 PER-BAND scale/offset 3b unscale /statistics", "GET", f"/cog/statistics?url={q(s['w57_perband_scale_3b'])}&unscale=true", None, "stats"),
                # --- Wave 58: Planar-separate asymmetric sparse TIFFs, sparse nodata=0 + ?nodata=999 override, float32 -0.0 vs nodata=0.0 ---
                ("PARITY W58 PLANAR SPARSE nodata=-9999 bidx=3 .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w58_planar_sparse_nd'])}&bidx=3", None, "npy"),
                ("PARITY W58 PLANAR SPARSE nodata=-9999 expr b1+b2+b3 .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w58_planar_sparse_nd'])}&expression={qe('b1+b2+b3')}", None, "npy"),
                ("PARITY W58 PLANAR SPARSE nodata=-9999 /point asym BL", "GET", f"/cog/point/-122.40,37.60?url={q(s['w58_planar_sparse_nd'])}", None, "point"),
                ("PARITY W58 PLANAR SPARSE nodata=-9999 /statistics", "GET", f"/cog/statistics?url={q(s['w58_planar_sparse_nd'])}", None, "stats"),
                ("PARITY W58 PLANAR SPARSE no-nodata (valid 0s) .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w58_planar_sparse_no_nd'])}", None, "npy"),
                ("PARITY W58 PLANAR SPARSE no-nodata (valid 0s) /statistics", "GET", f"/cog/statistics?url={q(s['w58_planar_sparse_no_nd'])}", None, "stats"),
                ("PARITY W58 SPARSE nd=0 ?nodata=999 override .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w58_sparse_nd0'])}&nodata=999", None, "npy"),
                ("PARITY W58 SPARSE nd=0 ?nodata=999 override .png", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w58_sparse_nd0'])}&nodata=999&rescale=0,800&colormap_name=viridis", None, "png"),
                ("PARITY W58 SPARSE nd=0 ?nodata=999 /point sparse BL", "GET", f"/cog/point/-122.40,37.60?url={q(s['w58_sparse_nd0'])}&nodata=999", None, "point"),
                ("PARITY W58 SPARSE nd=0 ?nodata=999 /statistics", "GET", f"/cog/statistics?url={q(s['w58_sparse_nd0'])}&nodata=999", None, "stats"),
                ("PARITY W58 NEG-ZERO -0.0 vs nodata=0.0 .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w58_neg_zero_f32'])}", None, "npy"),
                ("PARITY W58 NEG-ZERO -0.0 vs nodata=0.0 /statistics", "GET", f"/cog/statistics?url={q(s['w58_neg_zero_f32'])}", None, "stats"),
                # --- Wave 59: Lanczos/Cubic ringing clamping (u8/i8), single-row & prime strips, BigTIFF f64 P=3 w/ overviews ---
                ("PARITY W59 RINGING uint8 step lanczos .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w59_step_u8'])}&resampling=lanczos", None, "npy"),
                ("PARITY W59 RINGING uint8 step cubic .png", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w59_step_u8'])}&resampling=cubic", None, "png"),
                ("PARITY W59 RINGING int8 step lanczos .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w59_step_i8'])}&resampling=lanczos", None, "npy"),
                ("PARITY W59 RINGING int8 step cubic .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w59_step_i8'])}&resampling=cubic", None, "npy"),
                ("PARITY W59 STRIP-1 i16 DEFLATE P2 .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w59_strip1_i16_p2'])}", None, "npy"),
                ("PARITY W59 STRIP-1 i16 DEFLATE P2 /statistics", "GET", f"/cog/statistics?url={q(s['w59_strip1_i16_p2'])}", None, "stats"),
                ("PARITY W59 STRIP-13 f32 ZSTD P3 .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w59_strip13_f32_p3'])}", None, "npy"),
                ("PARITY W59 STRIP-13 f32 ZSTD P3 /point", "GET", f"/cog/point/-122.25,37.75?url={q(s['w59_strip13_f32_p3'])}", None, "point"),
                ("PARITY W59 BIGTIFF f64 P3 base z10 .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w59_bigtiff_f64_ovr'])}", None, "npy_f64_exact"),
                ("PARITY W59 BIGTIFF f64 P3 ovr z8 .npy", "GET", f"/cog/tiles/WebMercatorQuad/8/41/99.npy?url={q(s['w59_bigtiff_f64_ovr'])}", None, "npy_f64_exact"),
                # --- Wave 60: Fractional buffer=0.5/1.5, non-square 4:1 preview/bbox/feature max_size=0/64 & single-axis width/height ---
                ("PARITY W60 FRAC-BUFFER buffer=0.5 .npy (257x257)", "GET", f"/cog/tiles/WebMercatorQuad/12/657/1582.npy?url={q(merc)}&buffer=0.5", None, "npy"),
                ("PARITY W60 FRAC-BUFFER buffer=1.5 bilinear .png (259x259)", "GET", f"/cog/tiles/WebMercatorQuad/12/657/1582.png?url={q(merc)}&buffer=1.5&resampling=bilinear&rescale=0,1000&colormap_name=viridis", None, "png"),
                ("PARITY W60 PREVIEW 4:1 max_size=64 .npy (64x16)", "GET", f"/cog/preview.npy?url={q(s['w60_wide_4to1'])}&max_size=64", None, "npy"),
                ("PARITY W60 PREVIEW 4:1 max_size=0 native .npy (256x64)", "GET", f"/cog/preview.npy?url={q(s['w60_wide_4to1'])}&max_size=0", None, "npy"),
                ("PARITY W60 PREVIEW 4:1 width=80 auto-height .npy", "GET", f"/cog/preview.npy?url={q(s['w60_wide_4to1'])}&width=80", None, "npy"),
                ("PARITY W60 PREVIEW 4:1 height=32 auto-width .png", "GET", f"/cog/preview.png?url={q(s['w60_wide_4to1'])}&height=32&rescale=20,400&colormap_name=viridis", None, "png"),
                ("PARITY W60 BBOX 2:1 unsized max_size=64 .npy", "GET", f"/cog/bbox/-122.45,37.55,-122.05,37.75.npy?url={q(s['w60_wide_4to1'])}&max_size=64", None, "npy"),
                ("PARITY W60 BBOX 2:1 width=80 auto-height .npy", "GET", f"/cog/bbox/-122.45,37.55,-122.05,37.75.npy?url={q(s['w60_wide_4to1'])}&width=80", None, "npy"),
                ("PARITY W60 FEATURE 3:1 unsized max_size=64 .npy", "POST", f"/cog/feature.npy?url={q(s['w60_wide_4to1'])}&max_size=64", {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[[-122.45, 37.65], [-122.05, 37.65], [-122.05, 37.78], [-122.45, 37.78], [-122.45, 37.65]]]}}, "npy"),
                ("PARITY W60 FEATURE 3:1 height=32 auto-width .npy", "POST", f"/cog/feature.npy?url={q(s['w60_wide_4to1'])}&height=32", {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[[-122.45, 37.65], [-122.05, 37.65], [-122.05, 37.78], [-122.45, 37.78], [-122.45, 37.65]]]}}, "npy"),
                # --- Wave 61: GeoTIFF (.tif) unscale/expr/bbox/feature/preview/RGB parity & WebP alpha transparency ---
                ("PARITY W61 GEOTIFF tile unscale=true .tif", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.tif?url={q(s['w53_neg_scale'])}&unscale=true", None, "geotiff"),
                ("PARITY W61 GEOTIFF tile expr .tif", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.tif?url={q(s['w43_rgb_nodata'])}&expression={qe('(b2-b1)/(b2+b1+100)')}", None, "geotiff"),
                ("PARITY W61 GEOTIFF tile f32 internal-mask bilinear .tif", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.tif?url={q(s['w55_f32_imask'])}&resampling=bilinear", None, "geotiff"),
                ("PARITY W61 GEOTIFF GET /bbox 64x64.tif", "GET", f"/cog/bbox/-122.45,37.55,-122.05,37.95/64x64.tif?url={q(s['w60_wide_4to1'])}", None, "geotiff"),
                ("PARITY W61 GEOTIFF GET /bbox dst_crs=3857 64x64.tif", "GET", f"/cog/bbox/-122.45,37.55,-122.05,37.95/64x64.tif?url={q(s['w60_wide_4to1'])}&dst_crs=EPSG:3857", None, "geotiff"),
                ("PARITY W61 GEOTIFF POST /feature donut 64x64.tif", "POST", f"/cog/feature/64x64.tif?url={q(s['w50_donut'])}", {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[[-122.45, 37.55], [-122.05, 37.55], [-122.05, 37.95], [-122.45, 37.95], [-122.45, 37.55]], [[-122.33, 37.67], [-122.33, 37.83], [-122.17, 37.83], [-122.17, 37.67], [-122.33, 37.67]]]}}, "geotiff"),
                ("PARITY W61 WEBP tile viridis + nodata hole .webp", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.webp?url={q(s['w55_f32_imask'])}&rescale=10,300&colormap_name=viridis", None, "png"),
                ("PARITY W61 WEBP POST /feature donut 64x64.webp", "POST", f"/cog/feature/64x64.webp?url={q(s['w50_donut'])}&rescale=0,150&colormap_name=viridis", {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[[-122.45, 37.55], [-122.05, 37.55], [-122.05, 37.95], [-122.45, 37.95], [-122.45, 37.55]], [[-122.33, 37.67], [-122.33, 37.83], [-122.17, 37.83], [-122.17, 37.67], [-122.33, 37.67]]]}}, "png"),
                ("PARITY W61 GEOTIFF GET /preview 64x32.tif", "GET", f"/cog/preview/64x32.tif?url={q(s['w60_wide_4to1'])}", None, "geotiff"),
                ("PARITY W61 GEOTIFF tile 3-band RGB uint8 .tif", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.tif?url={q(s['w43_rgb_nodata'])}", None, "geotiff"),
                # --- Wave 62: Interval JSON colormaps (neg/pos, semi-transparent alpha, out-of-range, unscale=true) & STAC GeoTIFF/WebP/3857 point ---
                ("PARITY W62 INTERVAL CMAP neg/pos + alpha=128 + OOR .png", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w19_dem_f32'])}&colormap={qe(json.dumps([[[-30, 0], [0, 0, 255, 255]], [[0, 80], [0, 255, 0, 128]], [[80, 120], [255, 0, 0, 255]]]))}", None, "png"),
                ("PARITY W62 INTERVAL CMAP GET /bbox 64x64.png", "GET", f"/cog/bbox/-122.45,37.55,-122.05,37.95/64x64.png?url={q(s['w19_dem_f32'])}&colormap={qe(json.dumps([[[-30, 0], [0, 0, 255, 255]], [[0, 80], [0, 255, 0, 128]], [[80, 120], [255, 0, 0, 255]]]))}", None, "png"),
                ("PARITY W62 INTERVAL CMAP POST /feature 64x64.png", "POST", f"/cog/feature/64x64.png?url={q(s['w19_dem_f32'])}&colormap={qe(json.dumps([[[-30, 0], [0, 0, 255, 255]], [[0, 80], [0, 255, 0, 128]], [[80, 120], [255, 0, 0, 255]]]))}", {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[[-122.45, 37.55], [-122.05, 37.55], [-122.05, 37.95], [-122.45, 37.95], [-122.45, 37.55]]]}}, "png"),
                ("PARITY W62 INTERVAL CMAP + unscale=true .png", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w53_neg_scale'])}&unscale=true&colormap={qe(json.dumps([[[30, 42], [0, 128, 255, 255]], [[42, 52], [255, 200, 0, 180]], [[52, 60], [200, 0, 80, 255]]]))}", None, "png"),
                ("PARITY W62 STAC GET /bbox 5-band 64x64.tif", "GET", f"/stac/bbox/-122.4,37.6,-122.1,37.9/64x64.tif?url={q(s['w36_stac5b'])}&assets=ms", None, "geotiff"),
                ("PARITY W62 STAC GET /bbox dst_crs=3857 64x64.tif", "GET", f"/stac/bbox/-122.4,37.6,-122.1,37.9/64x64.tif?url={q(s['w36_stac5b'])}&assets=ms&dst_crs=EPSG:3857", None, "geotiff"),
                ("PARITY W62 STAC POST /feature coord_crs=3857 64x64.tif", "POST", f"/stac/feature/64x64.tif?url={q(s['w36_stac5b'])}&assets=ms&coord_crs=EPSG:3857", w37_poly_3857, "geotiff"),
                ("PARITY W62 STAC tile expr .tif", "GET", f"/stac/tiles/WebMercatorQuad/10/164/396.tif?url={q(s['w36_stac5b'])}&assets={qe('ms|expression=(b4-b3)/(b4+b3)')}", None, "geotiff"),
                ("PARITY W62 STAC /point coord_crs=EPSG:3857", "GET", f"/stac/point/-13608804.60,4544181.28?url={q(s['w36_stac5b'])}&assets=ms&coord_crs=EPSG:3857", None, "point"),
                ("PARITY W62 STAC tile RGB .webp", "GET", f"/stac/tiles/WebMercatorQuad/10/164/396.webp?url={q(s['w36_stac5b'])}&assets={qe('ms|bidx=1,2,3')}&rescale=0,255", None, "png"),
                # --- Wave 63: bidx reordering + per-band rescale/CF, normalizedIndex/terrarium/contours on bbox/feature/preview, POST /statistics non-uniform histogram_bins & histogram_range ---
                ("PARITY W63 BIDX REORDER 3,1,2 + per-band rescale .png", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w43_rgb_nodata'])}&bidx=3&bidx=1&bidx=2&rescale=0,250&rescale=0,120&rescale=0,180", None, "png"),
                ("PARITY W63 BIDX REORDER 3,1,2 + per-band rescale + CF .png", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w43_rgb_nodata'])}&bidx=3&bidx=1&bidx=2&rescale=0,250&rescale=0,120&rescale=0,180&color_formula={qe('Gamma R 1.3 Gamma G 0.9 Gamma B 1.2')}", None, "png"),
                ("PARITY W63 BIDX REORDER 3,1,2 + per-band rescale /preview .png", "GET", f"/cog/preview/64x64.png?url={q(s['w43_rgb_nodata'])}&bidx=3&bidx=1&bidx=2&rescale=0,250&rescale=0,120&rescale=0,180", None, "png"),
                ("PARITY W63 BIDX REORDER 3,1,2 + per-band rescale POST /feature .png", "POST", f"/cog/feature/64x64.png?url={q(s['w43_rgb_nodata'])}&bidx=3&bidx=1&bidx=2&rescale=0,250&rescale=0,120&rescale=0,180", {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[[-122.45, 37.55], [-122.05, 37.55], [-122.05, 37.95], [-122.45, 37.95], [-122.45, 37.55]]]}}, "png"),
                ("PARITY W63 ALG normalizedIndex GET /bbox .npy", "GET", f"/cog/bbox/-122.45,37.55,-122.05,37.95/64x64.npy?url={q(s['w43_rgb_nodata'])}&bidx=3&bidx=1&algorithm=normalizedIndex", None, "npy"),
                ("PARITY W63 ALG normalizedIndex POST /feature .npy", "POST", f"/cog/feature/64x64.npy?url={q(s['w43_rgb_nodata'])}&bidx=3&bidx=1&algorithm=normalizedIndex", {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[[-122.45, 37.55], [-122.05, 37.55], [-122.05, 37.95], [-122.45, 37.95], [-122.45, 37.55]]]}}, "npy"),
                ("PARITY W63 ALG terrarium GET /bbox .png", "GET", f"/cog/bbox/-122.45,37.55,-122.05,37.95/64x64.png?url={q(s['w19_dem_f32'])}&algorithm=terrarium", None, "png"),
                ("PARITY W63 ALG contours GET /preview .png", "GET", f"/cog/preview/64x64.png?url={q(s['w19_dem_f32'])}&algorithm=contours&algorithm_params={qe(json.dumps({'increment': 25, 'thickness': 2}))}", None, "png"),
                ("PARITY W63 POST /statistics non-uniform hist + p=5,50,95", "POST", f"/cog/statistics?url={q(s['w50_donut'])}&histogram_bins=20,40,70,110,130&p=5&p=50&p=95", {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[[-122.45, 37.55], [-122.05, 37.55], [-122.05, 37.95], [-122.45, 37.95], [-122.45, 37.55]], [[-122.33, 37.67], [-122.33, 37.83], [-122.17, 37.83], [-122.17, 37.67], [-122.33, 37.67]]]}}, "post_stats"),
                ("PARITY W63 POST /statistics histogram_range=40,100 bins=4", "POST", f"/cog/statistics?url={q(s['w50_donut'])}&histogram_bins=4&histogram_range=40,100", {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[[-122.45, 37.55], [-122.05, 37.55], [-122.05, 37.95], [-122.45, 37.95], [-122.45, 37.55]], [[-122.33, 37.67], [-122.33, 37.83], [-122.17, 37.83], [-122.17, 37.67], [-122.33, 37.67]]]}}, "post_stats"),
                # --- Wave 64: return_mask=false across PNG/NPY/GeoTIFF/bbox/feature/preview/STAC, STAC info.geojson, nested where() ---
                ("PARITY W64 RETURN_MASK=false 1-band viridis .png", "GET", f"/cog/tiles/WebMercatorQuad/12/657/1582.png?url={q(merc)}&rescale=0,1000&colormap_name=viridis&return_mask=false", None, "png"),
                ("PARITY W64 RETURN_MASK=false 3-band RGB .png", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w27_opaque_rgb'])}&return_mask=false", None, "png"),
                ("PARITY W64 RETURN_MASK=false GET /bbox 64x64.npy", "GET", f"/cog/bbox/-122.45,37.55,-122.05,37.95/64x64.npy?url={q(s['w19_dem_f32'])}&return_mask=false", None, "npy"),
                ("PARITY W64 RETURN_MASK=false GET /bbox 64x64.tif", "GET", f"/cog/bbox/-122.45,37.55,-122.05,37.95/64x64.tif?url={q(s['w19_dem_f32'])}&return_mask=false", None, "geotiff"),
                ("PARITY W64 RETURN_MASK=false POST /feature 64x64.npy", "POST", f"/cog/feature/64x64.npy?url={q(s['w19_dem_f32'])}&return_mask=false", {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[[-122.45, 37.55], [-122.05, 37.55], [-122.05, 37.95], [-122.45, 37.95], [-122.45, 37.55]]]}}, "npy"),
                ("PARITY W64 RETURN_MASK=false POST /feature 64x64.tif", "POST", f"/cog/feature/64x64.tif?url={q(s['w19_dem_f32'])}&return_mask=false", {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[[-122.45, 37.55], [-122.05, 37.55], [-122.05, 37.95], [-122.45, 37.95], [-122.45, 37.55]]]}}, "geotiff"),
                ("PARITY W64 RETURN_MASK=false GET /preview viridis .png", "GET", f"/cog/preview/64x64.png?url={q(s['w19_dem_f32'])}&rescale=-160,160&colormap_name=viridis&return_mask=false", None, "png"),
                ("PARITY W64 RETURN_MASK=false STAC tile 5-band .npy", "GET", f"/stac/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w36_stac5b'])}&assets=ms&return_mask=false", None, "npy"),
                ("PARITY W64 STAC /stac/info.geojson", "GET", f"/stac/info.geojson?url={q(s['w36_stac5b'])}&assets=ms", None, "stac_info_geojson"),
                ("PARITY W64 EXPR nested where(b1>100,where(b2>120,b3,b2),b1) .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w48_expr3b'])}&expression={qe('where(b1>100,where(b2>120,b3,b2),b1)')}", None, "npy"),
                # --- Wave 65: Compound boolean & transcendental expr VM, dual non-4326 CRSs (32610 -> 3857), all-nodata interior tile/polygon ---
                ("PARITY W65 EXPR where((b1>100)&(b2<150),b3,0) .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w48_expr3b'])}&expression={qe('where((b1>100)&(b2<150),b3,0)')}", None, "npy"),
                ("PARITY W65 EXPR where((b1<95)|(b2>155),255,50) .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w48_expr3b'])}&expression={qe('where((b1<95)|(b2>155),255,50)')}", None, "npy"),
                ("PARITY W65 EXPR power (b1**0.5)+(b2**1.5) .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w48_expr3b'])}&expression={qe('(b1**0.5)+(b2**1.5)')}", None, "npy"),
                ("PARITY W65 EXPR arctan2(b2-120,b1-100) .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w48_expr3b'])}&expression={qe('arctan2(b2-120,b1-100)')}", None, "npy"),
                ("PARITY W65 EXPR tanh+tan .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w48_expr3b'])}&expression={qe('tanh((b1-100)*0.02)+tan(b2*0.004)')}", None, "npy"),
                ("PARITY W65 DUAL-CRS bbox coord=32610 dst=3857 .npy", "GET", f"/cog/bbox/{','.join(f'{v:.2f}' for v in rasterio.warp.transform_bounds('EPSG:4326', 'EPSG:32610', -122.4, 37.6, -122.1, 37.9))}/64x64.npy?url={q(s['w37_cog'])}&coord_crs=EPSG:32610&dst_crs=EPSG:3857", None, "npy"),
                ("PARITY W65 DUAL-CRS bbox coord=32610 dst=3857 .tif", "GET", f"/cog/bbox/{','.join(f'{v:.2f}' for v in rasterio.warp.transform_bounds('EPSG:4326', 'EPSG:32610', -122.4, 37.6, -122.1, 37.9))}/64x64.tif?url={q(s['w37_cog'])}&coord_crs=EPSG:32610&dst_crs=EPSG:3857", None, "geotiff"),
                ("PARITY W65 DUAL-CRS feature coord=32610 dst=3857 .npy", "POST", f"/cog/feature/64x64.npy?url={q(s['w37_cog'])}&coord_crs=EPSG:32610&dst_crs=EPSG:3857", w37_poly_utm, "npy"),
                ("PARITY W65 DUAL-CRS STAC bbox coord=32610 dst=3857 .npy", "GET", f"/stac/bbox/{','.join(f'{v:.2f}' for v in rasterio.warp.transform_bounds('EPSG:4326', 'EPSG:32610', -122.4, 37.6, -122.1, 37.9))}/64x64.npy?url={q(s['w36_stac5b'])}&assets=ms&coord_crs=EPSG:32610&dst_crs=EPSG:3857", None, "npy"),
                ("PARITY W65 ALL-NODATA interior tile .png", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w65_all_nodata_tile'])}&rescale=0,200&colormap_name=viridis", None, "png"),
                ("PARITY W65 ALL-NODATA interior tile .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w65_all_nodata_tile'])}", None, "npy"),
                ("PARITY W65 ALL-NODATA interior /point", "GET", f"/cog/point/-122.25,37.65?url={q(s['w65_all_nodata_tile'])}", None, "point"),
                ("PARITY W65 ALL-NODATA interior POST /statistics", "POST", f"/cog/statistics?url={q(s['w65_all_nodata_tile'])}", {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[[-122.4, 37.55], [-122.1, 37.55], [-122.1, 37.70], [-122.4, 37.70], [-122.4, 37.55]]]}}, "post_stats"),
                # --- Wave 66: Constant-value (zero-variance) raster, STAC hetero-CRS (32610+4326) + hetero-dtype (u16+f32), bbox average/mode ---
                ("PARITY W66 CONST raster GET /statistics", "GET", f"/cog/statistics?url={q(s['w66_const_f32'])}", None, "stats"),
                ("PARITY W66 CONST raster POST /statistics", "POST", f"/cog/statistics?url={q(s['w66_const_f32'])}", {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[[-122.45, 37.55], [-122.05, 37.55], [-122.05, 37.95], [-122.45, 37.95], [-122.45, 37.55]]]}}, "post_stats"),
                ("PARITY W66 CONST raster degenerate rescale=42,42 .png", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w66_const_f32'])}&rescale=42,42&colormap_name=viridis", None, "png"),
                ("PARITY W66 CONST raster slope buffer=3 .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w66_const_flat'])}&algorithm=slope&buffer=3", None, "npy"),
                ("PARITY W66 CONST raster hillshade buffer=3 .png", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w66_const_flat'])}&algorithm=hillshade&buffer=3", None, "png"),
                ("PARITY W66 STAC HETERO-CRS+DTYPE red+nir .npy", "GET", f"/stac/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w66_stac_hetero'])}&assets=red&assets=nir", None, "npy"),
                ("PARITY W66 STAC HETERO-CRS+DTYPE red+nir .tif", "GET", f"/stac/tiles/WebMercatorQuad/10/164/396.tif?url={q(s['w66_stac_hetero'])}&assets=red&assets=nir", None, "geotiff"),
                ("PARITY W66 STAC HETERO-CRS+DTYPE red+nir /point", "GET", f"/stac/point/-122.25,37.75?url={q(s['w66_stac_hetero'])}&assets=red&assets=nir", None, "point"),
                ("PARITY W66 STAC HETERO-CRS+DTYPE red+nir /statistics", "GET", f"/stac/statistics?url={q(s['w66_stac_hetero'])}&assets=red&assets=nir", None, "stats"),
                ("PARITY W66 BBOX resampling=average f32+hole 32x32.npy", "GET", f"/cog/bbox/-122.45,37.55,-122.05,37.95/32x32.npy?url={q(s['w55_f32_imask'])}&resampling=average", None, "npy"),
                ("PARITY W66 BBOX resampling=mode cat u8 32x32.npy", "GET", f"/cog/bbox/-122.45,37.55,-122.05,37.95/32x32.npy?url={q(s['w23_cat_u8'])}&resampling=mode", None, "npy"),
                # --- Wave 67: STAC unscale=true hetero scale+offset + pipe |bidx=3,1 + expression, & NBITS=12/4 + nodata + smooth/unscale ---
                ("PARITY W67 STAC unscale=true hetero red+nir .npy", "GET", f"/stac/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w67_stac_scaled'])}&assets=red&assets=nir&unscale=true", None, "npy"),
                ("PARITY W67 STAC unscale=true hetero red+nir /point", "GET", f"/stac/point/-122.25,37.75?url={q(s['w67_stac_scaled'])}&assets=red&assets=nir&unscale=true", None, "point"),
                ("PARITY W67 STAC unscale=true hetero red+nir GET /statistics", "GET", f"/stac/statistics?url={q(s['w67_stac_scaled'])}&assets=red&assets=nir&unscale=true", None, "stats"),
                ("PARITY W67 STAC unscale=true hetero red+nir POST /statistics", "POST", f"/stac/statistics?url={q(s['w67_stac_scaled'])}&assets=red&assets=nir&unscale=true", {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[[-122.45, 37.55], [-122.05, 37.55], [-122.05, 37.95], [-122.45, 37.95], [-122.45, 37.55]]]}}, "post_stats"),
                ("PARITY W67 STAC unscale=true hetero red+nir /bbox .npy", "GET", f"/stac/bbox/-122.45,37.55,-122.05,37.95/64x64.npy?url={q(s['w67_stac_scaled'])}&assets=red&assets=nir&unscale=true", None, "npy"),
                ("PARITY W67 STAC unscale=true hetero red+nir POST /feature .npy", "POST", f"/stac/feature/64x64.npy?url={q(s['w67_stac_scaled'])}&assets=red&assets=nir&unscale=true", {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[[-122.45, 37.55], [-122.05, 37.55], [-122.05, 37.95], [-122.45, 37.95], [-122.45, 37.55]]]}}, "npy"),
                ("PARITY W67 STAC unscale=true + expr NDVI red+nir .npy", "GET", f"/stac/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w67_stac_scaled'])}&assets=red&assets=nir&unscale=true&expression={qe('(b2-b1)/(b2+b1)')}", None, "npy"),
                ("PARITY W67 STAC unscale=true + inline expr ms .npy", "GET", f"/stac/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w67_stac_scaled'])}&assets={qe('ms|expression=(b3-b1)/(b3+b1)')}&unscale=true", None, "npy"),
                ("PARITY W67 STAC unscale=true + pipe ms|bidx=3,1 .npy", "GET", f"/stac/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w67_stac_scaled'])}&assets={qe('ms|bidx=3,1')}&unscale=true", None, "npy"),
                ("PARITY W67 NBITS=12 nodata=4095 resampling=bilinear .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w67_nbits12_nodata_scaled'])}&resampling=bilinear", None, "npy"),
                ("PARITY W67 NBITS=12 nodata=4095 unscale=true .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w67_nbits12_nodata_scaled'])}&unscale=true", None, "npy"),
                ("PARITY W67 NBITS=12 nodata=4095 GET /statistics", "GET", f"/cog/statistics?url={q(s['w67_nbits12_nodata_scaled'])}", None, "stats"),
                ("PARITY W67 NBITS=4 nodata=15 resampling=cubic .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w67_nbits4_nodata'])}&resampling=cubic", None, "npy"),
                # --- Wave 68: Odd prime-dim (137x113) YCbCr JPEG-in-TIFF + non-default tilesize (512/128) combos ---
                ("PARITY W68 YCbCr JPEG odd 137x113 tiled .png", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w68_ycbcr_odd_tiled'])}", None, "png"),
                ("PARITY W68 YCbCr JPEG odd 137x113 tiled .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w68_ycbcr_odd_tiled'])}", None, "npy"),
                ("PARITY W68 YCbCr JPEG odd 137x113 tiled /point edge", "GET", f"/cog/point/-121.92,37.42?url={q(s['w68_ycbcr_odd_tiled'])}", None, "point"),
                ("PARITY W68 YCbCr JPEG odd 137x113 striped .png", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w68_ycbcr_odd_strip'])}", None, "png"),
                ("PARITY W68 TILESIZE=512 hillshade buffer=3 .png", "GET", f"/cog/tiles/WebMercatorQuad/12/657/1582.png?url={q(merc)}&tilesize=512&algorithm=hillshade&buffer=3", None, "png"),
                ("PARITY W68 TILESIZE=512 slope buffer=3 .npy", "GET", f"/cog/tiles/WebMercatorQuad/12/657/1582.npy?url={q(merc)}&tilesize=512&algorithm=slope&buffer=3", None, "npy"),
                ("PARITY W68 TILESIZE=128 cubic + unscale=true .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w67_nbits12_nodata_scaled'])}&tilesize=128&resampling=cubic&unscale=true", None, "npy"),
                ("PARITY W68 TILESIZE=512 interval colormap .png", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w55_f32_imask'])}&tilesize=512&colormap={qe(json.dumps([[[-30,0],[255,0,0,255]],[[0,80],[0,255,0,255]],[[80,120],[0,0,255,128]]]))}", None, "png"),
                ("PARITY W68 TILESIZE=128 STAC hetero unscale=true .npy", "GET", f"/stac/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w67_stac_scaled'])}&assets=red&assets=nir&tilesize=128&unscale=true", None, "npy"),
                # --- Wave 69: Pure 90-deg rotated (a=0,e=0), East-to-West mirrored (a<0), South-up DEM (e>0) & negative-key discrete int16 colormap ---
                ("PARITY W69 ROT90 (a=0,e=0) tile .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w69_rot90_4326'])}", None, "npy"),
                ("PARITY W69 ROT90 (a=0,e=0) /point", "GET", f"/cog/point/-122.15,37.85?url={q(s['w69_rot90_4326'])}", None, "point"),
                ("BETTER W69 ROT90 (a=0,e=0) /bbox 64x64.npy (cw exact, tt 500)", "GET", f"/cog/bbox/-122.45,37.55,-122.05,37.95/64x64.npy?url={q(s['w69_rot90_4326'])}", None, "better_w69_non_northup"),
                ("PARITY W69 ROT90 (a=0,e=0) GET /statistics", "GET", f"/cog/statistics?url={q(s['w69_rot90_4326'])}", None, "stats"),
                ("PARITY W69 MIRROR-X (a<0) tile .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w69_mirror_x_4326'])}", None, "npy"),
                ("BETTER W69 MIRROR-X (a<0) /point (cw exact 239.0, tt 404)", "GET", f"/cog/point/-122.15,37.85?url={q(s['w69_mirror_x_4326'])}", None, "better_w69_non_northup"),
                ("BETTER W69 MIRROR-X (a<0) /bbox 64x64.npy (cw exact, tt 500)", "GET", f"/cog/bbox/-122.45,37.55,-122.05,37.95/64x64.npy?url={q(s['w69_mirror_x_4326'])}", None, "better_w69_non_northup"),
                ("BETTER W69 MIRROR-X (a<0) POST /statistics (cw exact, tt 500)", "POST", f"/cog/statistics?url={q(s['w69_mirror_x_4326'])}", {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[[-122.45, 37.55], [-122.05, 37.55], [-122.05, 37.95], [-122.45, 37.95], [-122.45, 37.55]]]}}, "better_w69_non_northup"),
                ("PARITY W69 SOUTH-UP DEM slope buffer=3 .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w69_southup_dem'])}&algorithm=slope&buffer=3", None, "npy"),
                ("PARITY W69 SOUTH-UP DEM hillshade buffer=3 .png", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w69_southup_dem'])}&algorithm=hillshade&buffer=3", None, "png"),
                ("PARITY W69 SOUTH-UP DEM resampling=bilinear .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w69_southup_dem'])}&resampling=bilinear", None, "npy"),
                ("BETTER W69 SOUTH-UP DEM POST /statistics (cw exact, tt 500)", "POST", f"/cog/statistics?url={q(s['w69_southup_dem'])}", {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[[-122.45, 37.55], [-122.05, 37.55], [-122.05, 37.95], [-122.45, 37.95], [-122.45, 37.55]]]}}, "better_w69_non_northup"),
                ("PARITY W69 NEG-KEY discrete JSON colormap int16 .png", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w69_neg_cat_i16'])}&colormap={qe(json.dumps({'-10':[255,0,0,255],'0':[0,255,0,255],'10':[0,0,255,255],'20':[255,255,0,128]}))}", None, "png"),
                # --- Wave 70: Deep /statistics parameters (custom percentiles p=, category filter c= incl. negative, custom histogram_bins edge list, GET ?bbox=, algorithm=slope, STAC FC+unscale+p=) ---
                ("PARITY W70 STATS GET custom percentiles p=10,25,50,75,90", "GET", f"/cog/statistics?url={q(s['w55_f32_imask'])}&p=10&p=25&p=50&p=75&p=90", None, "stats"),
                ("PARITY W70 STATS POST custom percentiles p=5,50,95", "POST", f"/cog/statistics?url={q(s['w55_f32_imask'])}&p=5&p=50&p=95", {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[[-122.45, 37.55], [-122.05, 37.55], [-122.05, 37.95], [-122.45, 37.95], [-122.45, 37.55]]]}}, "post_stats"),
                ("PARITY W70 STATS GET categorical=true + c=-10,10 neg filter", "GET", f"/cog/statistics?url={q(s['w69_neg_cat_i16'])}&categorical=true&c=-10&c=10", None, "stats"),
                ("PARITY W70 STATS POST categorical=true + c=-10,20 neg filter", "POST", f"/cog/statistics?url={q(s['w69_neg_cat_i16'])}&categorical=true&c=-10&c=20", {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[[-122.45, 37.55], [-122.05, 37.55], [-122.05, 37.95], [-122.45, 37.95], [-122.45, 37.55]]]}}, "post_stats"),
                ("PARITY W70 STATS GET custom histogram_bins edge list", "GET", f"/cog/statistics?url={q(s['w55_f32_imask'])}&histogram_bins=0,50,150,250,400", None, "stats"),
                ("BETTER W70 STATS GET negative histogram_bins edges (cw 200, tt 422)", "GET", f"/cog/statistics?url={q(s['w55_f32_imask'])}&histogram_bins=-20,50,150,250,400", None, "better_w70_neg_bins"),
                ("PARITY W70 STATS POST custom histogram_bins edge list", "POST", f"/cog/statistics?url={q(s['w55_f32_imask'])}&histogram_bins=0,100,200,350", {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[[-122.45, 37.55], [-122.05, 37.55], [-122.05, 37.95], [-122.45, 37.95], [-122.45, 37.55]]]}}, "post_stats"),
                ("PARITY W70 STATS GET histogram_bins=5 + histogram_range", "GET", f"/cog/statistics?url={q(s['w55_f32_imask'])}&histogram_bins=5&histogram_range=50,300", None, "stats"),
                ("PARITY W70 STATS GET ?bbox= subset filter", "GET", f"/cog/statistics?url={q(s['w55_f32_imask'])}&bbox=-122.45,37.55,-122.05,37.95", None, "stats"),
                ("BETTER W70 STATS GET algorithm=slope full 128x128 (cw 16384, tt 14884)", "GET", f"/cog/statistics?url={q(s['w69_southup_dem'])}&algorithm=slope", None, "better_w70_slope_stats"),
                ("BETTER W70 STATS POST algorithm=slope (cw 400, tt 500)", "POST", f"/cog/statistics?url={q(merc)}&algorithm=slope", {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[[-122.45, 37.65], [-122.15, 37.65], [-122.15, 37.85], [-122.45, 37.85], [-122.45, 37.65]]]}}, "better_cw4xx_tt500"),
                ("PARITY W70 STAC GET /statistics custom p=10,90 + bins=5 + unscale", "GET", f"/stac/statistics?url={q(s['w67_stac_scaled'])}&assets=red&assets=nir&unscale=true&p=10&p=90&histogram_bins=5", None, "stats"),
                ("PARITY W70 STAC POST /statistics FC 2-poly + unscale + p=25,75", "POST", f"/stac/statistics?url={q(s['w67_stac_scaled'])}&assets=red&assets=nir&unscale=true&p=25&p=75", {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {"id": 1}, "geometry": {"type": "Polygon", "coordinates": [[[-122.50, 37.50], [-122.30, 37.50], [-122.30, 37.70], [-122.50, 37.70], [-122.50, 37.50]]]}}, {"type": "Feature", "properties": {"id": 2}, "geometry": {"type": "Polygon", "coordinates": [[[-122.20, 37.75], [-122.00, 37.75], [-122.00, 37.95], [-122.20, 37.95], [-122.20, 37.75]]]}}]}, "post_fc_stats_w23"),
                # --- Wave 71: Multi-output semicolon expressions across /point, GET/POST /statistics, /bbox, /feature, /preview, mixed-dtype, color_formula & STAC+unscale ---
                ("PARITY W71 MULTI-EXPR 3-band /cog/point", "GET", f"/cog/point/-122.35,37.80?url={q(s['w43_rgb_nodata'])}&expression={qe('b1+b2;b2-b1;b3*0.5')}", None, "point"),
                ("PARITY W71 MULTI-EXPR 3-band GET /cog/statistics", "GET", f"/cog/statistics?url={q(s['w43_rgb_nodata'])}&expression={qe('b1+b2;b2-b1;b3*0.5')}", None, "stats"),
                ("PARITY W71 MULTI-EXPR 3-band POST /cog/statistics", "POST", f"/cog/statistics?url={q(s['w43_rgb_nodata'])}&expression={qe('b1+b2;b2-b1;b3*0.5')}", {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[[-122.40, 37.55], [-122.10, 37.55], [-122.10, 37.85], [-122.40, 37.85], [-122.40, 37.55]]]}}, "post_stats"),
                ("PARITY W71 MULTI-EXPR 3-band /bbox .tif", "GET", f"/cog/bbox/-122.40,37.55,-122.10,37.85/64x64.tif?url={q(s['w43_rgb_nodata'])}&expression={qe('b1+b2;b2-b1;b3*0.5')}", None, "geotiff"),
                ("PARITY W71 MULTI-EXPR 3-band /bbox per-band rescale + CF .png", "GET", f"/cog/bbox/-122.40,37.55,-122.10,37.85/64x64.png?url={q(s['w43_rgb_nodata'])}&expression={qe('b1+b2;b2-b1;b3*0.5')}&rescale=0,400&rescale=-200,200&rescale=0,130&color_formula={qe('Gamma RGB 1.3')}", None, "png"),
                ("PARITY W71 MULTI-EXPR 3-band POST /feature .tif", "POST", f"/cog/feature/64x64.tif?url={q(s['w43_rgb_nodata'])}&expression={qe('b1+b2;b2-b1;b3*0.5')}", {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[[-122.40, 37.55], [-122.10, 37.55], [-122.10, 37.85], [-122.40, 37.85], [-122.40, 37.55]]]}}, "geotiff"),
                ("PARITY W71 MULTI-EXPR 3-band /preview .npy", "GET", f"/cog/preview/64x64.npy?url={q(s['w43_rgb_nodata'])}&expression={qe('b1+b2;b2-b1;b3*0.5')}", None, "npy"),
                ("PARITY W71 MULTI-EXPR mixed-dtype (int;float;int) tile .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w43_rgb_nodata'])}&expression={qe('b1+1;b2*0.5;b3-1')}", None, "npy"),
                ("PARITY W71 MULTI-EXPR STAC unscale=true 3-band tile .npy", "GET", f"/stac/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w67_stac_scaled'])}&assets=red&assets=nir&unscale=true&expression={qe('b2-b1;b2+b1;(b2-b1)/(b2+b1)')}", None, "npy"),
                ("PARITY W71 MULTI-EXPR STAC unscale=true 3-band tile + CF .png", "GET", f"/stac/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w67_stac_scaled'])}&assets=red&assets=nir&unscale=true&expression={qe('b2-b1;b2+b1;(b2-b1)/(b2+b1)')}&rescale=0,2&rescale=0,3&rescale=0,1&color_formula={qe('Gamma RGB 1.2')}", None, "png"),
                ("PARITY W71 MULTI-EXPR STAC unscale=true 3-band /point", "GET", f"/stac/point/-122.25,37.75?url={q(s['w67_stac_scaled'])}&assets=red&assets=nir&unscale=true&expression={qe('b2-b1;b2+b1;(b2-b1)/(b2+b1)')}", None, "point"),
                ("PARITY W71 MULTI-EXPR STAC unscale=true 3-band GET /statistics", "GET", f"/stac/statistics?url={q(s['w67_stac_scaled'])}&assets=red&assets=nir&unscale=true&expression={qe('b2-b1;b2+b1;(b2-b1)/(b2+b1)')}", None, "stats"),
                ("PARITY W71 MULTI-EXPR STAC unscale=true 3-band POST /statistics", "POST", f"/stac/statistics?url={q(s['w67_stac_scaled'])}&assets=red&assets=nir&unscale=true&expression={qe('b2-b1;b2+b1;(b2-b1)/(b2+b1)')}", {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[[-122.45, 37.55], [-122.05, 37.55], [-122.05, 37.95], [-122.45, 37.95], [-122.45, 37.55]]]}}, "post_stats"),
                # --- Wave 72: Expression VM deep ops (signed modulo %, ModPow2, log1p/expm1, arcsin/arccos/arctan, sinh/cosh, floor/ceil) & 3-band non-u8 GeoTIFF matrix ---
                ("PARITY W72 EXPR signed modulo (-b1)%10 + (-b1)%8 .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w55_f32_imask'])}&expression={qe('((-b1)%10)+((-b1)%8)')}", None, "npy"),
                ("PARITY W72 EXPR negative-divisor modulo b1%-10 .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w55_f32_imask'])}&expression={qe('b1%-10')}", None, "npy"),
                ("PARITY W72 EXPR log1p + expm1 .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w55_f32_imask'])}&expression={qe('log1p(abs(b1)*0.01)+expm1(abs(b1)*0.002)')}", None, "npy"),
                ("PARITY W72 EXPR arcsin + arccos + arctan .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w55_f32_imask'])}&expression={qe('arcsin(b1/400.0)+arccos(b1/400.0)+arctan(b1*0.01)')}", None, "npy"),
                ("PARITY W72 EXPR sinh + cosh .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w55_f32_imask'])}&expression={qe('sinh(b1*0.005)+cosh(b1*0.005)')}", None, "npy"),
                ("PARITY W72 EXPR floor + ceil on signed float .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w55_f32_imask'])}&expression={qe('floor(b1*0.13)-ceil(-b1*0.13)')}", None, "npy"),
                ("PARITY W72 EXPR double-neg & unary -(b1-b2) .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w43_rgb_nodata'])}&expression={qe('--b1-(b2-b3)')}", None, "npy"),
                ("PARITY W72 GEOTIFF 3-band uint16 tile .tif (MinIsBlack)", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.tif?url={q(s['w67_stac_scaled'].parent / 'w67_ms_hetero_scale_u16.tif')}", None, "geotiff"),
                ("PARITY W72 GEOTIFF 3-band float32 tile .tif (MinIsBlack)", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.tif?url={q(s['w33_planar_f32_zstd_p3'])}", None, "geotiff"),
                ("PARITY W72 GEOTIFF 3-band uint8 return_mask=false .tif (RGB)", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.tif?url={q(s['w43_rgb_nodata'])}&return_mask=false", None, "geotiff"),
                ("PARITY W72 GEOTIFF 3-band uint16 return_mask=false .tif (MinIsBlack)", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.tif?url={q(s['w67_stac_scaled'].parent / 'w67_ms_hetero_scale_u16.tif')}&return_mask=false", None, "geotiff"),
                ("PARITY W72 INFO.GEOJSON internal 1-bit mask float32", "GET", f"/cog/info.geojson?url={q(s['w55_f32_imask'])}", None, "info_geojson_w23"),
                ("PARITY W72 INFO.GEOJSON NBITS=12 nodata=4095 scaled", "GET", f"/cog/info.geojson?url={q(s['w67_nbits12_nodata_scaled'])}", None, "info_geojson_w23"),
                ("PARITY W72 INFO.GEOJSON ROT90 (a=0,e=0)", "GET", f"/cog/info.geojson?url={q(s['w69_rot90_4326'])}", None, "info_geojson_w23"),
                ("BETTER W72 INFO.GEOJSON MIRROR-X (a<0) valid bounds (cw minx<maxx, tt inverted)", "GET", f"/cog/info.geojson?url={q(s['w69_mirror_x_4326'])}", None, "better_w72_mirror_info"),
                # --- Wave 73: Non-3857 TMS GeoTIFF (.tif) outputs + deep OGC Maps API (/cog/map & /stac/map) ---
                ("PARITY W73 TMS WorldCRS84Quad tile .tif (EPSG:4326)", "GET", f"/cog/tiles/WorldCRS84Quad/10/{morecantile.tms.get('WorldCRS84Quad').tile(-122.25, 37.75, 10).x}/{morecantile.tms.get('WorldCRS84Quad').tile(-122.25, 37.75, 10).y}.tif?url={q(s['w55_f32_imask'])}", None, "geotiff"),
                ("PARITY W73 TMS WorldMercatorWGS84Quad tile .tif (EPSG:3395)", "GET", f"/cog/tiles/WorldMercatorWGS84Quad/10/{morecantile.tms.get('WorldMercatorWGS84Quad').tile(-122.25, 37.75, 10).x}/{morecantile.tms.get('WorldMercatorWGS84Quad').tile(-122.25, 37.75, 10).y}.tif?url={q(s['w55_f32_imask'])}", None, "geotiff"),
                ("PARITY W73 TMS EuropeanETRS89_LAEAQuad tile .tif (EPSG:3035)", "GET", f"/cog/tiles/EuropeanETRS89_LAEAQuad/6/{morecantile.tms.get('EuropeanETRS89_LAEAQuad').tile(10.5, 48.5, 6).x}/{morecantile.tms.get('EuropeanETRS89_LAEAQuad').tile(10.5, 48.5, 6).y}.tif?url={q(s['w24_eu'])}", None, "geotiff"),
                ("PARITY W73 STAC TMS WorldCRS84Quad unscale=true .tif", "GET", f"/stac/tiles/WorldCRS84Quad/10/{morecantile.tms.get('WorldCRS84Quad').tile(-122.25, 37.65, 10).x}/{morecantile.tms.get('WorldCRS84Quad').tile(-122.25, 37.65, 10).y}.tif?url={q(s['w67_stac_scaled'])}&assets=red&assets=nir&unscale=true", None, "geotiff"),
                ("PARITY W73 OGC /cog/map f=tif 4326 unscale=true", "GET", f"/cog/map?url={q(s['w67_nbits12_nodata_scaled'])}&bbox=-122.45,37.55,-122.05,37.95&width=64&height=64&unscale=true&f=tif", None, "geotiff"),
                ("PARITY W73 OGC /cog/map f=tif bbox-crs=3857 crs=3857", "GET", f"/cog/map?url={q(s['w55_f32_imask'])}&bbox=-13631071,4516059,-13586543,4572377&bbox-crs={qe('http://www.opengis.net/def/crs/EPSG/0/3857')}&crs={qe('http://www.opengis.net/def/crs/EPSG/0/3857')}&width=64&height=64&f=tif", None, "geotiff"),
                ("PARITY W73 OGC /cog/map f=tif crs=32610 (UTM 10N)", "GET", f"/cog/map?url={q(s['w55_f32_imask'])}&bbox=-122.45,37.55,-122.05,37.95&crs={qe('http://www.opengis.net/def/crs/EPSG/0/32610')}&width=64&height=64&f=tif", None, "geotiff"),
                ("PARITY W73 OGC /cog/map f=webp viridis + hole", "GET", f"/cog/map?url={q(s['w55_f32_imask'])}&bbox=-122.45,37.55,-122.05,37.95&width=64&height=64&rescale=0,300&colormap_name=viridis&f=webp", None, "png"),
                ("PARITY W73 OGC /cog/map f=jpeg + color_formula", "GET", f"/cog/map?url={q(s['w43_rgb_nodata'])}&bbox=-122.45,37.55,-122.05,37.95&width=64&height=64&color_formula={qe('Gamma RGB 1.3')}&f=jpeg", None, "png"),
                ("PARITY W73 OGC /cog/map f=png expression + viridis", "GET", f"/cog/map?url={q(s['w43_rgb_nodata'])}&bbox=-122.45,37.55,-122.05,37.95&width=64&height=64&expression={qe('(b2-b1)/(b2+b1+1)')}&rescale=-1,1&colormap_name=viridis&f=png", None, "png"),
                ("BETTER W73 OGC /cog/map f=png algorithm=hillshade exact 64x64 (cw 64, tt 58)", "GET", f"/cog/map?url={q(merc)}&bbox=-122.45,37.65,-122.15,37.85&width=64&height=64&algorithm=hillshade&f=png", None, "better_w73_map_hillshade"),
                ("PARITY W73 OGC /stac/map f=tif red+nir unscale=true", "GET", f"/stac/map?url={q(s['w67_stac_scaled'])}&assets=red&assets=nir&bbox=-122.45,37.55,-122.05,37.95&width=64&height=64&unscale=true&f=tif", None, "geotiff"),
                ("PARITY W73 OGC /stac/map f=png 3857 expr+unscale+viridis", "GET", f"/stac/map?url={q(s['w67_stac_scaled'])}&assets=red&assets=nir&bbox=-13631071,4516059,-13586543,4572377&bbox-crs={qe('http://www.opengis.net/def/crs/EPSG/0/3857')}&crs={qe('http://www.opengis.net/def/crs/EPSG/0/3857')}&width=64&height=64&unscale=true&expression={qe('(b2-b1)/(b2+b1)')}&rescale=0,1&colormap_name=viridis&f=png", None, "png"),
                # --- Wave 74: Multi-overview ([2, 4]) COGs with per-overview 1-bit internal masks (NewSubfileType=5) ---
                ("PARITY W74 OVR+IMASK YCbCr JPEG base z10 .png", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.png?url={q(s['w74_ovr_imask_ycbcr'])}", None, "png"),
                ("PARITY W74 OVR+IMASK YCbCr JPEG base z10 .tif", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.tif?url={q(s['w74_ovr_imask_ycbcr'])}", None, "geotiff"),
                ("PARITY W74 OVR+IMASK YCbCr JPEG ovr z8 .png", "GET", f"/cog/tiles/WebMercatorQuad/8/41/99.png?url={q(s['w74_ovr_imask_ycbcr'])}", None, "png"),
                ("PARITY W74 OVR+IMASK YCbCr JPEG ovr z9 .png", "GET", f"/cog/tiles/WebMercatorQuad/9/82/198.png?url={q(s['w74_ovr_imask_ycbcr'])}", None, "png"),
                ("PARITY W74 OVR+IMASK YCbCr JPEG /preview 64x64.png", "GET", f"/cog/preview/64x64.png?url={q(s['w74_ovr_imask_ycbcr'])}", None, "png"),
                ("PARITY W74 OVR+IMASK YCbCr JPEG /bbox 64x64.png", "GET", f"/cog/bbox/-122.55,37.45,-121.95,38.08/64x64.png?url={q(s['w74_ovr_imask_ycbcr'])}", None, "png"),
                ("PARITY W74 OVR+IMASK f32+unscale base z10 .npy", "GET", f"/cog/tiles/WebMercatorQuad/10/164/396.npy?url={q(s['w74_ovr_imask_f32_scaled'])}&unscale=true", None, "npy"),
                ("PARITY W74 OVR+IMASK f32+unscale ovr z8 .npy", "GET", f"/cog/tiles/WebMercatorQuad/8/41/99.npy?url={q(s['w74_ovr_imask_f32_scaled'])}&unscale=true", None, "npy"),
                ("PARITY W74 OVR+IMASK f32+unscale ovr z8 viridis .png", "GET", f"/cog/tiles/WebMercatorQuad/8/41/99.png?url={q(s['w74_ovr_imask_f32_scaled'])}&unscale=true&rescale=5,30&colormap_name=viridis", None, "png"),
                ("PARITY W74 OVR+IMASK f32+unscale GET /statistics", "GET", f"/cog/statistics?url={q(s['w74_ovr_imask_f32_scaled'])}&unscale=true", None, "stats"),
                ("PARITY W74 OVR+IMASK f32+unscale POST /statistics", "POST", f"/cog/statistics?url={q(s['w74_ovr_imask_f32_scaled'])}&unscale=true", {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[[-122.50, 37.50], [-122.00, 37.50], [-122.00, 38.05], [-122.50, 38.05], [-122.50, 37.50]]]}}, "post_stats"),
                ("PARITY W74 OVR+IMASK f32 /cog/info.geojson", "GET", f"/cog/info.geojson?url={q(s['w74_ovr_imask_f32_scaled'])}", None, "info_geojson_w23"),
            ]

            # --- 6. INTENTIONALLY BETTER contracts (docs/titiler-divergences.md) ---
            cases += [
                ("BETTER W13 reproject=gauss (cw 200, tt 422)", "GET", f"/cog/tiles/WebMercatorQuad/12/652/1583.npy?url={q(sat)}&reproject=gauss", None, "better_cw200_tt422"),
                ("BETTER W12 ZONAL MultiPoint exact count=3 (cw 3.0, tt 0.03)", "POST", f"/cog/statistics?url={q(sat)}&max_size=128", w12_mpt, "better_multipoint_3"),
                ("BETTER W11 rot30 EPSG:3857 tile (cw exact, tt unrotated)", "GET", f"/cog/tiles/WebMercatorQuad/12/655/1582.npy?url={q(s['w11_rot30'])}", None, "better_rot30_3857"),
                ("BETTER W9 nbits_1 .npy raw 0/1 (cw 2ch, tt 4ch)", "GET", f"/cog/tiles/WebMercatorQuad/{z_j}/{tx_j}/{ty_j}.npy?url={q(s['w9_nbits_1'])}", None, "better_palette_npy"),
                ("BETTER W9 hillshade custom 256x256 (cw 256, tt 250)", "GET", f"/cog/tiles/WebMercatorQuad/12/657/1582.png?url={q(merc)}&algorithm=hillshade&algorithm_params={qe(json.dumps({'azimuth':135,'angle_altitude':30,'z_exaggeration':2.0}))}", None, "better_hillshade_256"),
                ("BETTER inverted rescale (cw inverts, tt clips 255)", "GET", f"/cog/tiles/WebMercatorQuad/12/657/1582.png?url={q(merc)}&rescale=1000,0&colormap_name=viridis", None, "better_inverted_rescale"),
                ("BETTER resampling=min (cw 200, tt 422)", "GET", f"/cog/tiles/WebMercatorQuad/12/657/1582.png?url={q(merc)}&rescale=0,1000&colormap_name=viridis&resampling=min", None, "better_cw200_tt422"),
                ("BETTER resampling=med (cw 200, tt 422)", "GET", f"/cog/tiles/WebMercatorQuad/12/657/1582.png?url={q(merc)}&rescale=0,1000&colormap_name=viridis&resampling=med", None, "better_cw200_tt422"),
                ("BETTER color_formula 1-band gray (cw 200, tt 500)", "GET", f"/cog/tiles/WebMercatorQuad/12/657/1582.png?url={q(merc)}&rescale=0,1000&color_formula={qe('Gamma RGB 1.8')}", None, "better_cw200_tt500"),
                ("BETTER 0..360 tile west (cw 200, tt 404)", "GET", f"/cog/tiles/WebMercatorQuad/2/0/1.png?url={q(geo360)}&rescale=0,255&colormap_name=viridis", None, "better_cw200_tt404"),
                ("BETTER 0..360 point -122.5,37.5 (cw 200, tt 404)", "GET", f"/cog/point/-122.5,37.5?url={q(geo360)}", None, "better_cw200_tt404"),
                ("BETTER antimeridian west tile (cw 200, tt 404)", "GET", f"/cog/tiles/WebMercatorQuad/6/0/35.npy?url={q(s['antimeridian'])}", None, "better_cw200_tt404"),
                ("BETTER polar TMS OOB (cw 404, tt blank 200)", "GET", f"/cog/tiles/UPSArcticWGS84Quad/2/1/1.npy?url={q(polar3413)}", None, "better_cw404_tt200"),
                ("BETTER ungeoref preview (cw 200, tt 500)", "GET", f"/cog/preview/64x64.png?url={q(s['ungeoref'])}", None, "better_cw200_tt500"),
                ("BETTER out-of-range bidx=99 (cw 400, tt 500)", "GET", f"/cog/tiles/WebMercatorQuad/12/652/1583.png?url={q(sat)}&bidx=99", None, "better_cw4xx_tt500"),
                ("BETTER palette .npy raw indices (cw 2ch, tt 4ch)", "GET", f"/cog/tiles/WebMercatorQuad/10/164/395.npy?url={q(palette)}", None, "better_palette_npy"),
                ("BETTER 2-band Float32 .png (cw 400, tt 200)", "GET", f"/cog/tiles/WebMercatorQuad/12/652/1583.png?url={q(sat)}&bidx=1&bidx=2&rescale=0,0.35", None, "better_cw400_tt200"),
                ("BETTER 2-band Uint16 .png (cw 400)", "GET", f"/cog/tiles/WebMercatorQuad/10/164/395.png?url={q(rgbn_u16)}&bidx=1&bidx=2", None, "better_cw400"),
                ("BETTER mask_1bit return_mask=false (valid match, collar zeroed)", "GET", f"/cog/tiles/WebMercatorQuad/10/164/395.png?url={q(mask_1bit)}&return_mask=false", None, "better_mask_zeroed"),
                ("BETTER mask_rgba return_mask=false (valid match, collar zeroed)", "GET", f"/cog/tiles/WebMercatorQuad/10/164/395.png?url={q(mask_rgba)}&return_mask=false", None, "better_mask_zeroed"),
                ("BETTER slope seamless 256x256 (cw 256, tt 250)", "GET", f"/cog/tiles/WebMercatorQuad/12/657/1582.npy?url={q(merc)}&algorithm=slope", None, "better_slope_256"),
                ("BETTER cast dtype=int16 (cw int16, tt hardcoded u8)", "GET", f"/cog/tiles/WebMercatorQuad/11/327/792.npy?url={q(s['neg_f32'])}&algorithm=cast&algorithm_params={qe(json.dumps({'dtype':'int16'}))}", None, "better_cast_i16"),
            ]

            if args.only:
                cases = [c for c in cases if args.only.lower() in c[0].lower()]

            mode_str = "CPU-ONLY" if primary_cpu_only else "DEFAULT"
            print(f"\n=== Adversarial Differential Oracle ({mode_str}): {len(cases)} TiTiler-diff checks + Wave 7 Mosaic/HTTP checks ===")
            failures = []
            gpu_diffs = []
            with httpx.Client(timeout=30.0) as client:
                for label, method, path, body, kind in cases:
                    r_cw = client.request(method, cw_base + path, json=body)
                    try:
                        r_tt = client.request(method, tt_base + path, json=body)
                    except httpx.HTTPError:
                        with httpx.Client(timeout=30.0) as c2:
                            r_tt = c2.request(method, tt_base + path, json=body)

                    is_unmasked = "return_mask=false" in path
                    is_smooth = ("resampling=" in path and "resampling=nearest" not in path) or ("reproject=" in path and "reproject=nearest" not in path)

                    if gpu_base is not None:
                        r_gpu = client.request(method, gpu_base + path, json=body)
                        if r_cw.status_code != r_gpu.status_code:
                            gpu_diffs.append({"label": label, "detail": f"status cpu={r_cw.status_code} gpu={r_gpu.status_code}"})
                        elif r_cw.status_code == 200:
                            ct = r_cw.headers.get("content-type", "")
                            if kind == "headers":
                                hok, hd = same_place(r_cw, r_gpu)
                                iok, idet = compare_images(r_cw.content, r_gpu.content)
                                gok, gd = (hok and iok), (hd if not hok else idet)
                            elif kind == "json_exact":
                                d = first_json_diff(r_cw.json(), r_gpu.json())
                                gok, gd = (d is None), (d or "exact JSON match")
                            elif kind == "auto_jpeg":
                                gct = r_gpu.headers.get("content-type", "")
                                iok, idet = compare_images(r_cw.content, r_gpu.content)
                                gok = ("image/jpeg" in gct) and iok
                                gd = idet if "image/jpeg" in gct else f"gpu content-type={gct!r}"
                            elif "image/png" in ct or "image/jpeg" in ct or "image/webp" in ct:
                                gok, gd = compare_images(r_cw.content, r_gpu.content)
                                if gok and is_unmasked and "image/png" in ct:
                                    im_g = Image.open(io.BytesIO(r_gpu.content))
                                    if "A" in im_g.mode or "transparency" in im_g.info:
                                        gok, gd = False, f"gpu unexpectedly has alpha on return_mask=false: mode={im_g.mode}"
                            elif kind == "npy_f64_exact":
                                ac, ag = np.load(io.BytesIO(r_cw.content)), np.load(io.BytesIO(r_gpu.content))
                                low = np.finfo(np.float64).min
                                mc, mg = ac[1] > low, ag[1] > low
                                both = mc & mg
                                merr = float(np.max(np.abs(ac[0][both] - ag[0][both]))) if both.any() else 999.0
                                gok = ac.dtype == np.float64 and ag.dtype == np.float64 and bool((mc == mg).all()) and merr < 1e-4
                                gd = f"f64 max_abs_err={merr:.6f}"
                            elif kind == "geotiff_f64_exact":
                                with rasterio.MemoryFile(r_cw.content) as mfc, mfc.open() as dc, rasterio.MemoryFile(r_gpu.content) as mfg, mfg.open() as dg:
                                    ac, ag = dc.read(), dg.read()
                                    mc, mg = ac[1] > 0, ag[1] > 0
                                    both = mc & mg
                                    merr = float(np.max(np.abs(ac[0][both] - ag[0][both]))) if both.any() else 999.0
                                    gok = ac.dtype == np.float64 and ag.dtype == np.float64 and bool((mc == mg).all()) and merr < 1e-4
                                    gd = f"f64 tif max_abs_err={merr:.6f}"
                            elif kind == "point_f64_exact":
                                vc, vg = r_cw.json().get("values", []), r_gpu.json().get("values", [])
                                gok = len(vc) == 1 and len(vg) == 1 and abs(vc[0] - vg[0]) < 1e-4
                                gd = f"cpu={vc} gpu={vg}"
                            elif kind == "better_aspect_w40" and ".npy" in path:
                                ac, ag = np.load(io.BytesIO(r_cw.content)), np.load(io.BytesIO(r_gpu.content))
                                mc, mg = ac[1] > 0.0, ag[1] > 0.0
                                merr = float(np.max(np.abs(ac[0][mc] - ag[0][mc]))) if mc.any() else 999.0
                                gok = ac.shape == ag.shape and bool((mc == mg).all()) and merr < 1e-4
                                gd = f"aspect max_diff={merr:.6f}"
                            elif "application/x-binary" in ct or ".npy" in path:
                                gok, gd = compare_npy(r_cw.content, r_gpu.content, check_values=True, masked=(not is_unmasked) or ("/stac/" in path), smooth=is_smooth)
                            elif "image/tiff" in ct:
                                gok, gd = compare_geotiffs(r_cw.content, r_gpu.content, compare_pixels=True, masked=not is_unmasked, smooth=is_smooth)
                            elif kind in ("stats", "stats_w23"):
                                gok, gd = compare_stats_strict(r_cw.json(), r_gpu.json())
                            elif kind in ("post_stats", "better_linestring_stats"):
                                gok, gd = compare_stats_strict(r_cw.json()["properties"]["statistics"], r_gpu.json()["properties"]["statistics"])
                            elif kind == "post_fc_stats_w23":
                                fc_c, fc_g = r_cw.json()["features"], r_gpu.json()["features"]
                                gok, gd = True, "ok"
                                for fi in range(len(fc_c)):
                                    fok, fgd = compare_stats_strict(fc_c[fi]["properties"]["statistics"], fc_g[fi]["properties"]["statistics"])
                                    if not fok:
                                        gok, gd = False, f"f{fi}: {fgd}"
                                        break
                            elif kind in ("info_geojson_w23", "stac_info_geojson"):
                                d_ig = first_json_diff(r_cw.json(), r_gpu.json())
                                gok, gd = (d_ig is None), (d_ig or "exact JSON match")
                            elif kind == "point" or "/point/" in path:
                                gok, gd = compare_point_strict(r_cw.json(), r_gpu.json())
                            elif kind in ("tilejson_w18", "tilejson_w32"):
                                jc, jg = r_cw.json(), r_gpu.json()
                                gok = (jc["bounds"] == jg["bounds"] and jc["minzoom"] == jg["minzoom"] and jc["maxzoom"] == jg["maxzoom"] and jc["center"] == jg["center"])
                                gd = "tilejson metadata match" if gok else f"cw={jc} gpu={jg}"
                            elif kind == "better_stac_wmts":
                                gok = ET.fromstring(r_gpu.content).tag.endswith("Capabilities") and "WebMercatorQuad" in r_gpu.text
                                gd = "wmts xml match" if gok else "gpu wmts invalid"
                            elif kind in ("better_info_crs3857", "better_bounds_crs3857", "better_cog_stac"):
                                d = first_json_diff(r_cw.json(), r_gpu.json())
                                gok, gd = (d is None), (d or "exact JSON match")
                            elif kind == "discovery_w38":
                                jc, jg = r_cw.json(), r_gpu.json()
                                k = "algorithms" if path == "/algorithms" else "colormaps"
                                ic = [x["id"] if isinstance(x, dict) else x for x in jc.get(k, [])]
                                ig = [x["id"] if isinstance(x, dict) else x for x in jg.get(k, [])]
                                gok, gd = (ic == ig), f"ids match ({len(ic)})" if ic == ig else f"cw={len(ic)} gpu={len(ig)}"
                            elif kind == "ogc_tiles_w39":
                                jc, jg = r_cw.json(), r_gpu.json()
                                if "tilesets" in jc:
                                    ic = [t.get("tileMatrixSetId") or t.get("title") for t in jc["tilesets"]]
                                    ig = [t.get("tileMatrixSetId") or t.get("title") for t in jg.get("tilesets", [])]
                                    gok, gd = (ic == ig), f"tilesets match ({len(ic)})"
                                else:
                                    gok = jc.get("dataType") == jg.get("dataType") and jc.get("boundingBox") == jg.get("boundingBox")
                                    gd = "tileset meta match" if gok else "tileset meta diff"
                            else:
                                gok, gd = (r_cw.content == r_gpu.content), "bytes diff"
                            if not gok:
                                gpu_diffs.append({"label": label, "detail": gd})

                    if kind == "better_inverted_rescale":
                        if r_cw.status_code != 200 or r_tt.status_code != 200:
                            ok, detail = False, f"HTTP cw={r_cw.status_code} tt={r_tt.status_code}"
                        else:
                            ac = np.array(Image.open(io.BytesIO(r_cw.content)).convert("RGB"))
                            at = np.array(Image.open(io.BytesIO(r_tt.content)).convert("RGB"))
                            cw_spatial_std = max(float(ac[..., c].std()) for c in range(3))
                            tt_spatial_std = max(float(at[..., c].std()) for c in range(3))
                            ok = cw_spatial_std > 1.0 and tt_spatial_std < 1e-6
                            detail = f"cw spatial_std={cw_spatial_std:.1f} vs tt spatial_std={tt_spatial_std:.1f}"
                    elif kind in ("better_cw200_tt400", "better_cw200_tt422", "better_cw200_tt500", "better_cw200_tt404"):
                        exp_tt = int(kind.split("_tt")[1])
                        payload_ok = False
                        if r_cw.status_code == 200:
                            ct = r_cw.headers.get("content-type", "")
                            if "image/" in ct:
                                im = Image.open(io.BytesIO(r_cw.content))
                                payload_ok = im.size[0] > 0 and im.size[1] > 0
                            elif ".npy" in path:
                                arr = np.load(io.BytesIO(r_cw.content))
                                payload_ok = arr.size > 0
                            elif "json" in ct:
                                payload_ok = len(r_cw.json().get("values", [])) > 0
                        ok = r_cw.status_code == 200 and payload_ok and r_tt.status_code == exp_tt
                        detail = f"cw={r_cw.status_code}(valid={payload_ok}) tt={r_tt.status_code}"
                    elif kind == "better_w69_non_northup":
                        if "/point/" in path:
                            ok = (
                                r_cw.status_code == 200 and r_tt.status_code == 404
                                and abs(float(r_cw.json()["values"][0]) - 239.0) < 1e-4
                            )
                            detail = f"cw={r_cw.status_code} val={r_cw.json().get('values') if r_cw.status_code == 200 else None} tt={r_tt.status_code}"
                        elif ".npy" in path:
                            if r_cw.status_code == 200 and r_tt.status_code == 500:
                                arr = np.load(io.BytesIO(r_cw.content))
                                fmin = np.finfo(arr.dtype).min
                                valid = arr[1] > fmin
                                nvalid = int(valid.sum())
                                mean_v = float(arr[0][valid].mean()) if valid.any() else 0.0
                                exp_lo, exp_hi = (310.0, 326.0) if "rot90" in path else (312.0, 328.0)
                                ok = arr.shape == (2, 64, 64) and 3500 < nvalid < 4000 and exp_lo < mean_v < exp_hi
                                detail = f"cw=200 shape={arr.shape} valid={nvalid} mean={mean_v:.2f} tt=500"
                            else:
                                ok, detail = False, f"cw={r_cw.status_code} tt={r_tt.status_code}"
                        else:
                            if r_cw.status_code == 200 and r_tt.status_code == 500:
                                st = r_cw.json()["properties"]["statistics"]["b1"]
                                mean_v, cnt = float(st["mean"]), float(st["count"])
                                if "southup" in path:
                                    ok = 730.0 < mean_v < 740.0 and 5300.0 < cnt < 5400.0
                                else:
                                    ok = 310.0 < mean_v < 330.0 and 4900.0 < cnt < 5000.0
                                detail = f"cw=200 mean={mean_v:.2f} count={cnt:.1f} tt=500"
                            else:
                                ok, detail = False, f"cw={r_cw.status_code} tt={r_tt.status_code}"
                    elif kind == "better_w70_neg_bins":
                        if r_cw.status_code == 200 and r_tt.status_code == 422:
                            b1 = r_cw.json()["b1"]
                            counts, edges = b1["histogram"]
                            ok = edges == [-20.0, 50.0, 150.0, 250.0, 400.0] and sum(counts) == b1["count"] and counts[0] > 0
                            detail = f"cw=200 edges={edges} counts={counts} tt=422"
                        else:
                            ok, detail = False, f"cw={r_cw.status_code} tt={r_tt.status_code}"
                    elif kind == "better_w70_slope_stats":
                        if r_cw.status_code == 200 and r_tt.status_code == 200:
                            bc, bt = r_cw.json()["b1"], r_tt.json()["b1"]
                            ok = (
                                int(bc["count"]) == 16384 and int(bt["count"]) == 14884
                                and abs(float(bc["mean"]) - float(bt["mean"])) < 1e-2
                            )
                            detail = f"cw count={bc['count']} mean={bc['mean']:.4f} vs tt count={bt['count']} mean={bt['mean']:.4f}"
                        else:
                            ok, detail = False, f"cw={r_cw.status_code} tt={r_tt.status_code}"
                    elif kind == "better_cw404_tt200":
                        ok = r_cw.status_code == 404 and r_tt.status_code == 200
                        detail = f"cw={r_cw.status_code} tt={r_tt.status_code}"
                    elif kind == "better_cw4xx_tt500":
                        ok = (400 <= r_cw.status_code < 500) and r_tt.status_code == 500
                        detail = f"cw={r_cw.status_code} tt={r_tt.status_code}"
                    elif kind == "better_cw400_tt200":
                        ok = r_cw.status_code == 400 and r_tt.status_code == 200
                        detail = f"cw={r_cw.status_code} tt={r_tt.status_code}"
                    elif kind == "better_cw400":
                        ok = r_cw.status_code == 400
                        detail = f"cw={r_cw.status_code} tt={r_tt.status_code}"
                    elif kind == "better_palette_npy":
                        if r_cw.status_code != 200 or r_tt.status_code != 200:
                            ok, detail = False, f"HTTP cw={r_cw.status_code} tt={r_tt.status_code}"
                        else:
                            ac = np.load(io.BytesIO(r_cw.content))
                            at = np.load(io.BytesIO(r_tt.content))
                            uniq = set(np.unique(ac[0]).tolist())
                            ok = ac.shape == (2, 256, 256) and ac.dtype == np.uint8 and at.shape == (4, 256, 256) and len(uniq) >= 2
                            detail = f"cw={ac.shape}:{ac.dtype} uniq={sorted(uniq)[:6]} tt={at.shape}:{at.dtype}"
                    elif kind == "better_mask_zeroed":
                        if r_cw.status_code != 200 or r_tt.status_code != 200:
                            ok, detail = False, f"HTTP cw={r_cw.status_code} tt={r_tt.status_code}"
                        else:
                            ac = np.array(Image.open(io.BytesIO(r_cw.content)).convert("RGB"))
                            at = np.array(Image.open(io.BytesIO(r_tt.content)).convert("RGB"))
                            valid = ac.sum(axis=-1) > 0
                            zeroed = ~valid
                            mae = float(np.abs(ac[valid].astype(np.float32) - at[valid].astype(np.float32)).mean()) if valid.any() else 999.0
                            ok = ac.shape == at.shape and int(valid.sum()) > 1000 and int(zeroed.sum()) > 1000 and mae < 1.0
                            detail = f"valid_px={int(valid.sum())} zeroed_px={int(zeroed.sum())} valid_MAE={mae:.2f}"
                    elif kind == "better_slope_256":
                        if r_cw.status_code != 200 or r_tt.status_code != 200:
                            ok, detail = False, f"HTTP cw={r_cw.status_code} tt={r_tt.status_code}"
                        else:
                            ac = np.load(io.BytesIO(r_cw.content))
                            at = np.load(io.BytesIO(r_tt.content))
                            diff = float(np.nanmax(np.abs(ac[0, 3:253, 3:253] - at[0])))
                            border_valid = bool((ac[1, 0, :] > 0).all() and (ac[1, -1, :] > 0).all() and (ac[1, :, 0] > 0).all() and (ac[1, :, -1] > 0).all())
                            ok = ac.shape == (2, 256, 256) and at.shape == (2, 250, 250) and diff < 1e-2 and border_valid
                            detail = f"cw={ac.shape} tt={at.shape} inner_max_diff={diff:.4f} border_valid={border_valid}"
                    elif kind == "better_hillshade_256":
                        if r_cw.status_code != 200 or r_tt.status_code != 200:
                            ok, detail = False, f"HTTP cw={r_cw.status_code} tt={r_tt.status_code}"
                        else:
                            ac = np.array(Image.open(io.BytesIO(r_cw.content)).convert("RGBA"))
                            at = np.array(Image.open(io.BytesIO(r_tt.content)).convert("RGBA"))
                            if ac.shape != (256, 256, 4) or at.shape != (250, 250, 4):
                                ok, detail = False, f"shape mismatch cw={ac.shape} tt={at.shape}"
                            else:
                                inner_mae = float(np.abs(ac[3:253, 3:253, :3].astype(np.float32) - at[:, :, :3].astype(np.float32)).mean())
                                border_opaque = bool((ac[0, :, 3] == 255).all() and (ac[-1, :, 3] == 255).all() and (ac[:, 0, 3] == 255).all() and (ac[:, -1, 3] == 255).all())
                                ok = inner_mae < 2.0 and border_opaque
                                detail = f"cw={ac.shape} tt={at.shape} inner_MAE={inner_mae:.2f} border_opaque={border_opaque}"
                    elif kind == "better_cast_i16":
                        if r_cw.status_code != 200 or r_tt.status_code != 200:
                            ok, detail = False, f"HTTP cw={r_cw.status_code} tt={r_tt.status_code}"
                        else:
                            ac = np.load(io.BytesIO(r_cw.content))
                            at = np.load(io.BytesIO(r_tt.content))
                            valid = ac[1] > np.iinfo(np.int16).min
                            ok = ac.dtype == np.int16 and at.dtype == np.uint8 and bool(valid.any()) and int(ac[0][valid].min()) < 0
                            detail = f"cw={ac.dtype} min={int(ac[0][valid].min())} tt={at.dtype}"
                    elif kind == "better_rot30_3857":
                        if r_cw.status_code != 200 or r_tt.status_code != 200:
                            ok, detail = False, f"HTTP cw={r_cw.status_code} tt={r_tt.status_code}"
                        else:
                            ac = np.load(io.BytesIO(r_cw.content))
                            at = np.load(io.BytesIO(r_tt.content))
                            # Exact affine ground truth at tile 12/655/1582 pixel (128, 128) = 373.5
                            cw_v, tt_v = float(ac[0, 128, 128]), float(at[0, 128, 128])
                            ok = ac.shape == (2, 256, 256) and abs(cw_v - 373.5) < 1e-3 and abs(tt_v - 373.5) > 5.0
                            detail = f"cw[128,128]={cw_v:.2f} (exact=373.50) vs tt[128,128]={tt_v:.2f}"
                    elif kind == "better_multipoint_3":
                        if r_cw.status_code != 200 or r_tt.status_code != 200:
                            ok, detail = False, f"HTTP cw={r_cw.status_code} tt={r_tt.status_code}"
                        else:
                            sc = r_cw.json()["properties"]["statistics"]["b1"]
                            st = r_tt.json()["properties"]["statistics"]["b1"]
                            # cogzilla samples all 3 points (including extrema on bbox corners!) with exact count=3.0;
                            # TiTiler rasterizes on 10x10 sub-pixels so count ~= 0.03 while mean/min/max match.
                            ok = (
                                abs(sc["count"] - 3.0) < 1e-6
                                and st["count"] < 0.1
                                and sc["std"] > 0.005
                                and abs(sc["mean"] - st["mean"]) < 0.02
                            )
                            detail = f"cw count={sc['count']:.1f} mean={sc['mean']:.4f} std={sc['std']:.4f} vs tt count={st['count']:.3f} mean={st['mean']:.4f}"
                    elif kind == "better_linestring_stats":
                        if r_cw.status_code != 200 or r_tt.status_code != 200:
                            ok, detail = False, f"HTTP cw={r_cw.status_code} tt={r_tt.status_code}"
                        else:
                            sc = r_cw.json()["properties"]["statistics"]["b1"]
                            st = r_tt.json()["properties"]["statistics"]["b1"]
                            # cogzilla measures exact geometric line pixel length (~145 / ~205 px),
                            # whereas TiTiler's 10x10 sub-pixel rasterization undercounts 1D lines by ~7-10x.
                            ok = (
                                sc["count"] > 100.0
                                and st["count"] < 30.0
                                and abs(sc["mean"] - st["mean"]) < 2.0
                            )
                            detail = f"cw count={sc['count']:.1f} mean={sc['mean']:.2f} vs tt count={st['count']:.2f} mean={st['mean']:.2f}"
                    elif kind == "better_retina_2x":
                        size_ok = False
                        got_sz = None
                        if r_cw.status_code == 200:
                            if ".npy" in path:
                                arr = np.load(io.BytesIO(r_cw.content))
                                got_sz = arr.shape
                                size_ok = arr.shape == (2, 512, 512)
                            else:
                                im = Image.open(io.BytesIO(r_cw.content))
                                got_sz = im.size
                                size_ok = im.size == (512, 512)
                        ok = r_cw.status_code == 200 and size_ok and r_tt.status_code == 422
                        detail = f"cw={r_cw.status_code} size={got_sz} tt={r_tt.status_code}"
                    elif kind == "auto_jpeg":
                        if r_cw.status_code != 200 or r_tt.status_code != 200:
                            ok, detail = False, f"HTTP cw={r_cw.status_code} tt={r_tt.status_code}"
                        else:
                            ct_c = r_cw.headers.get("content-type", "")
                            ct_t = r_tt.headers.get("content-type", "")
                            iok, idet = compare_images(r_cw.content, r_tt.content)
                            ok = ("image/jpeg" in ct_c) and ("image/jpeg" in ct_t) and iok
                            detail = f"ct_cw={ct_c!r} ct_tt={ct_t!r} {idet}"
                    elif kind == "better_inf_masked":
                        if r_cw.status_code != 200 or r_tt.status_code != 200:
                            ok, detail = False, f"HTTP cw={r_cw.status_code} tt={r_tt.status_code}"
                        else:
                            ac = np.load(io.BytesIO(r_cw.content))
                            at = np.load(io.BytesIO(r_tt.content))
                            low = np.finfo(np.float32).min
                            mc, mt = ac[1] > low, at[1] > low
                            cw_all_finite = bool(np.isfinite(ac[0][mc]).all()) if mc.any() else False
                            tt_has_inf = bool(np.isinf(at[0][mt]).any()) if mt.any() else False
                            both_fin = mc & mt & np.isfinite(at[0])
                            max_err = float(np.max(np.abs(ac[0][both_fin] - at[0][both_fin]))) if both_fin.any() else 999.0
                            ok = ac.shape == (2, 256, 256) and cw_all_finite and tt_has_inf and max_err < 1e-3
                            detail = f"cw_all_finite={cw_all_finite} tt_has_inf={tt_has_inf} finite_max_err={max_err:.5f}"
                    elif kind == "better_gcp_point":
                        if r_cw.status_code != 200 or r_tt.status_code != 200:
                            ok, detail = False, f"HTTP cw={r_cw.status_code} tt={r_tt.status_code}"
                        else:
                            vc = r_cw.json()["values"][0]
                            vt = r_tt.json()["values"][0]
                            ok = vc is not None and abs(float(vc) - 324.0) < 1e-3 and vt is None
                            detail = f"cw={vc} (exact=324.0) vs tt={vt}"
                    elif kind == "better_gcp_png":
                        if r_cw.status_code != 200 or r_tt.status_code != 200:
                            ok, detail = False, f"HTTP cw={r_cw.status_code} tt={r_tt.status_code}"
                        else:
                            imc = np.array(Image.open(io.BytesIO(r_cw.content)).convert("RGBA"))
                            imt = np.array(Image.open(io.BytesIO(r_tt.content)).convert("RGBA"))
                            # cogzilla renders the exact viridis gradient ([50, 181, 122] at center),
                            # whereas TiTiler collapses the entire float32 GCP tile to viridis(0) ([68, 1, 84]).
                            cw_std = float(imc[imc[..., 3] > 0, :3].std(axis=0).max())
                            tt_std = float(imt[imt[..., 3] > 0, :3].std(axis=0).max())
                            ok = (
                                imc.shape == (256, 256, 4)
                                and int(imc[128, 128, 1]) > 150
                                and cw_std > 30.0
                                and tt_std < 1.0
                            )
                            detail = f"cw[128,128]={imc[128,128,:3].tolist()} spatial_std={cw_std:.1f} vs tt[128,128]={imt[128,128,:3].tolist()} spatial_std={tt_std:.1f}"
                    elif kind == "better_linear_cmap":
                        if r_cw.status_code != 200 or r_tt.status_code != 200:
                            ok, detail = False, f"HTTP cw={r_cw.status_code} tt={r_tt.status_code}"
                        else:
                            imc = np.array(Image.open(io.BytesIO(r_cw.content)).convert("RGBA"))
                            imt = np.array(Image.open(io.BytesIO(r_tt.content)).convert("RGBA"))
                            cw_cov = float((imc[..., 3] > 0).mean()) * 100.0
                            tt_cov = float((imt[..., 3] > 0).mean()) * 100.0
                            # cogzilla linearly interpolates custom JSON stops across all valid pixels (~76.7%),
                            # whereas TiTiler treats the 3-stop JSON dict as discrete (~0.6% coverage).
                            ok = (
                                imc.shape == (256, 256, 4)
                                and cw_cov > 70.0
                                and tt_cov < 5.0
                                and 180 < int(imc[128, 128, 0]) < 255
                                and int(imc[128, 128, 1]) == 0
                            )
                            detail = f"cw_cov={cw_cov:.1f}% px[128,128]={imc[128,128,:3].tolist()} vs tt_cov={tt_cov:.2f}%"
                    elif kind == "better_stac_asset_as_band":
                        if r_cw.status_code != 200 or r_tt.status_code != 400:
                            ok, detail = False, f"HTTP cw={r_cw.status_code} (exp 200) tt={r_tt.status_code} (exp 400)"
                        else:
                            ac = np.load(io.BytesIO(r_cw.content))
                            vm = ac[1] > np.finfo(np.float32).min
                            vmin = float(ac[0][vm].min()) if vm.any() else -999.0
                            vmax = float(ac[0][vm].max()) if vm.any() else -999.0
                            ok = ac.shape == (2, 256, 256) and ac.dtype == np.float32 and int(vm.sum()) > 10000 and 0.08 < vmin < 0.11 and 0.67 < vmax < 0.72
                            detail = f"cw=200 shape={ac.shape} valid={int(vm.sum())} ndvi=[{vmin:.4f}..{vmax:.4f}] tt=400"
                    elif kind == "better_inverted_rescale_w31":
                        if r_cw.status_code != 200 or r_tt.status_code != 200:
                            ok, detail = False, f"HTTP cw={r_cw.status_code} tt={r_tt.status_code}"
                        else:
                            imc = np.array(Image.open(io.BytesIO(r_cw.content)).convert("RGBA"))
                            imt = np.array(Image.open(io.BytesIO(r_tt.content)).convert("RGBA"))
                            vm = (imc[..., 3] == 255) & (imt[..., 3] == 255)
                            cw_std = float(imc[vm, :3].std(axis=0).max()) if vm.any() else 0.0
                            tt_std = float(imt[vm, :3].std(axis=0).max()) if vm.any() else 999.0
                            # cogzilla renders a true inverted ramp (std > 40), whereas TiTiler's np.clip(x, 200, 0) collapses to a flat constant (std == 0).
                            ok = imc.shape == (256, 256, 4) and int(vm.sum()) > 10000 and cw_std > 40.0 and tt_std < 0.5
                            detail = f"cw_std={cw_std:.1f} vs tt_std={tt_std:.2f} (tt flat={imt[vm,:3][0].tolist() if vm.any() else []})"
                    elif kind == "better_cmap_color_formula":
                        if r_cw.status_code != 200 or r_tt.status_code != 500:
                            ok, detail = False, f"HTTP cw={r_cw.status_code} (exp 200) tt={r_tt.status_code} (exp 500)"
                        else:
                            imc = np.array(Image.open(io.BytesIO(r_cw.content)).convert("RGBA"))
                            vm = imc[..., 3] == 255
                            cw_std = float(imc[vm, :3].std(axis=0).max()) if vm.any() else 0.0
                            ok = imc.shape == (256, 256, 4) and int(vm.sum()) > 10000 and cw_std > 20.0
                            detail = f"cw=200 valid={int(vm.sum())} std={cw_std:.1f} tt=500"
                    elif kind == "better_hillshade_512":
                        if r_cw.status_code != 200 or r_tt.status_code != 200:
                            ok, detail = False, f"HTTP cw={r_cw.status_code} tt={r_tt.status_code}"
                        else:
                            imc = np.array(Image.open(io.BytesIO(r_cw.content)).convert("RGBA"))
                            imt = np.array(Image.open(io.BytesIO(r_tt.content)).convert("RGBA"))
                            ok = imc.shape == (512, 512, 4) and imt.shape[:2] == (506, 506) and int((imc[..., 3] == 255).sum()) > 100000
                            detail = f"cw={imc.shape[:2]} tt={imt.shape[:2]}"
                    elif kind == "tilejson_w32":
                        if r_cw.status_code != 200 or r_tt.status_code != 200:
                            ok, detail = False, f"HTTP cw={r_cw.status_code} tt={r_tt.status_code}"
                        else:
                            jc, jt = r_cw.json(), r_tt.json()
                            berr = max(abs(a - b) for a, b in zip(jc["bounds"], jt["bounds"]))
                            tile0 = jc.get("tiles", [""])[0]
                            ok = berr < 1e-4 and 0 <= jc["minzoom"] <= jc["maxzoom"] <= 24 and "assets=dem" in tile0
                            detail = f"bounds_err={berr:.6f} cw_z={jc['minzoom']}..{jc['maxzoom']} tt_z={jt['minzoom']}..{jt['maxzoom']}"
                    elif kind == "better_stac_wmts":
                        if r_cw.status_code != 200 or r_tt.status_code != 404:
                            ok, detail = False, f"HTTP cw={r_cw.status_code} (exp 200) tt={r_tt.status_code} (exp 404)"
                        else:
                            root = ET.fromstring(r_cw.content)
                            ok = root.tag.endswith("Capabilities") and "WebMercatorQuad" in r_cw.text and "assets=dem" in r_cw.text
                            detail = f"cw=200 root={root.tag.split('}')[-1]} len={len(r_cw.text)} tt=404"
                    elif kind == "better_u64_npy":
                        if r_cw.status_code != 200 or r_tt.status_code != 500:
                            ok, detail = False, f"HTTP cw={r_cw.status_code} (exp 200) tt={r_tt.status_code} (exp 500)"
                        else:
                            ac = np.load(io.BytesIO(r_cw.content))
                            vm = ac[1] > 0
                            vmin = int(ac[0][vm].min()) if vm.any() else 0
                            vmax = int(ac[0][vm].max()) if vm.any() else 0
                            ok = ac.shape == (2, 256, 256) and ac.dtype == np.uint64 and int(vm.sum()) > 10000 and vmin >= 1000 and vmax > 10000000
                            detail = f"cw=200 dtype={ac.dtype} valid={int(vm.sum())} range=[{vmin}..{vmax}] tt=500"
                    elif kind == "better_bbox_hillshade_64":
                        if r_cw.status_code != 200 or r_tt.status_code != 200:
                            ok, detail = False, f"HTTP cw={r_cw.status_code} tt={r_tt.status_code}"
                        else:
                            imc = np.array(Image.open(io.BytesIO(r_cw.content)).convert("RGBA"))
                            imt = np.array(Image.open(io.BytesIO(r_tt.content)).convert("RGBA"))
                            ok = imc.shape == (64, 64, 4) and imt.shape[:2] == (58, 58) and int((imc[..., 3] == 255).sum()) == 64 * 64
                            detail = f"cw={imc.shape[:2]} tt={imt.shape[:2]}"
                    elif kind == "better_info_crs3857":
                        if r_cw.status_code != 200 or r_tt.status_code != 200:
                            ok, detail = False, f"HTTP cw={r_cw.status_code} tt={r_tt.status_code}"
                        else:
                            jc, jt = r_cw.json(), r_tt.json()
                            exp_b = transform_bounds("EPSG:4326", "EPSG:3857", -122.5, 37.5, -122.0, 38.0)
                            berr = max(abs(a - b) for a, b in zip(jc["bounds"], exp_b))
                            ok = berr < 1.0 and "3857" in str(jc.get("crs", "")) and abs(jt["bounds"][0] - (-122.5)) < 1e-4
                            detail = f"cw_crs={jc.get('crs')} berr={berr:.2f}m vs tt_bounds={jt['bounds']}"
                    elif kind == "better_bounds_crs3857":
                        if r_cw.status_code != 200 or r_tt.status_code != 404:
                            ok, detail = False, f"HTTP cw={r_cw.status_code} (exp 200) tt={r_tt.status_code} (exp 404)"
                        else:
                            jc = r_cw.json()
                            exp_b = transform_bounds("EPSG:4326", "EPSG:3857", -122.5, 37.5, -122.0, 38.0)
                            berr = max(abs(a - b) for a, b in zip(jc["bounds"], exp_b))
                            ok = berr < 1.0
                            detail = f"cw=200 berr={berr:.2f}m tt=404"
                    elif kind == "better_cog_stac":
                        if r_cw.status_code != 200 or r_tt.status_code != 404:
                            ok, detail = False, f"HTTP cw={r_cw.status_code} (exp 200) tt={r_tt.status_code} (exp 404)"
                        else:
                            jc = r_cw.json()
                            ok = jc.get("type") == "Feature" and "assets" in jc and "data" in jc["assets"] and jc.get("bbox") == [-122.5, 37.5, -122.0, 38.0]
                            detail = f"cw=200 stac_version={jc.get('stac_version')} bbox={jc.get('bbox')} tt=404"
                    elif kind == "discovery_w38":
                        if r_cw.status_code != 200 or r_tt.status_code != 200:
                            ok, detail = False, f"HTTP cw={r_cw.status_code} tt={r_tt.status_code}"
                        elif path == "/algorithms":
                            def _alg_ids(obj):
                                a = obj.get("algorithms", [])
                                if isinstance(a, dict):
                                    return set(a.keys())
                                return {x["id"] if isinstance(x, dict) else x for x in a}
                            ac, at = _alg_ids(r_cw.json()), _alg_ids(r_tt.json())
                            missing = sorted(at - ac)
                            rtms_c, rtms_t = client.get(cw_base + "/tileMatrixSets"), client.get(tt_base + "/tileMatrixSets")
                            tms_c = {t["id"] for t in rtms_c.json().get("tileMatrixSets", [])}
                            tms_t = {t["id"] for t in rtms_t.json().get("tileMatrixSets", [])}
                            ok = len(missing) == 0 and "WebMercatorQuad" in tms_c and "WorldCRS84Quad" in tms_c and len(tms_c & tms_t) >= 10
                            detail = f"algs={len(ac)} (tt={len(at)}, missing={missing}) tms={len(tms_c)}"
                        else:
                            def _cm_ids(obj):
                                c = obj.get("colormaps") or obj.get("colorMaps") or []
                                return {x["id"] if isinstance(x, dict) else x for x in c}
                            cc, ct = _cm_ids(r_cw.json()), _cm_ids(r_tt.json())
                            ok = {"viridis", "plasma", "inferno", "terrain", "turbo"}.issubset(cc) and len(cc & ct) >= 100
                            detail = f"cw_cmaps={len(cc)} tt_cmaps={len(ct)} shared={len(cc & ct)}"
                    elif kind == "ogc_tiles_w39":
                        if r_cw.status_code != 200 or r_tt.status_code != 200:
                            ok, detail = False, f"HTTP cw={r_cw.status_code} tt={r_tt.status_code}"
                        else:
                            jc, jt = r_cw.json(), r_tt.json()
                            if "tilesets" in jc:
                                ok = len(jc["tilesets"]) >= 10 and len(jt.get("tilesets", [])) >= 10
                                detail = f"cw_tilesets={len(jc['tilesets'])} tt_tilesets={len(jt.get('tilesets', []))}"
                            else:
                                ok = "boundingBox" in jc and len(jc.get("links", [])) > 0 and "boundingBox" in jt
                                detail = f"dataType={jc.get('dataType')} links={len(jc.get('links', []))}"
                    elif kind == "npy_f64_exact":
                        if r_cw.status_code != 200 or r_tt.status_code != 200:
                            ok, detail = False, f"HTTP cw={r_cw.status_code} tt={r_tt.status_code}"
                        else:
                            ac, at = np.load(io.BytesIO(r_cw.content)), np.load(io.BytesIO(r_tt.content))
                            low = np.finfo(np.float64).min
                            mc, mt = ac[1] > low, at[1] > low
                            both = mc & mt
                            mask_eq = float((mc == mt).mean()) * 100.0
                            merr = float(np.max(np.abs(ac[0][both] - at[0][both]))) if both.any() else 999.0
                            cstd = float(ac[0][mc].std()) if mc.any() else 0.0
                            ok = ac.dtype == np.float64 and at.dtype == np.float64 and mask_eq >= 99.0 and merr < 1e-4 and cstd > 0.05
                            detail = f"f64 mask={mask_eq:.2f}% max_abs_err={merr:.6f} std={cstd:.4f}"
                    elif kind == "geotiff_f64_exact":
                        if r_cw.status_code != 200 or r_tt.status_code != 200:
                            ok, detail = False, f"HTTP cw={r_cw.status_code} tt={r_tt.status_code}"
                        else:
                            with rasterio.MemoryFile(r_cw.content) as mfc, mfc.open() as dc, rasterio.MemoryFile(r_tt.content) as mft, mft.open() as dt:
                                ac, at = dc.read(), dt.read()
                                mc, mt = ac[1] > 0, at[1] > 0
                                both = mc & mt
                                mask_eq = float((mc == mt).mean()) * 100.0
                                merr = float(np.max(np.abs(ac[0][both] - at[0][both]))) if both.any() else 999.0
                                cstd = float(ac[0][mc].std()) if mc.any() else 0.0
                                ok = ac.dtype == np.float64 and at.dtype == np.float64 and mask_eq >= 99.0 and merr < 1e-4 and cstd > 0.05
                                detail = f"f64 tif mask={mask_eq:.2f}% max_abs_err={merr:.6f} std={cstd:.4f}"
                    elif kind == "point_f64_exact":
                        if r_cw.status_code != 200 or r_tt.status_code != 200:
                            ok, detail = False, f"HTTP cw={r_cw.status_code} tt={r_tt.status_code}"
                        else:
                            vc, vt = r_cw.json().get("values", []), r_tt.json().get("values", [])
                            aerr = abs(vc[0] - vt[0]) if len(vc) == 1 and len(vt) == 1 else 999.0
                            ok = aerr < 1e-4 and abs(vc[0] - 1_000_000_000.256) < 1e-4
                            detail = f"cw={vc} tt={vt} abs_err={aerr:.6f}"
                    elif kind == "better_aspect_w40":
                        if r_cw.status_code != 200 or r_tt.status_code != 422:
                            ok, detail = False, f"HTTP cw={r_cw.status_code} (exp 200) tt={r_tt.status_code} (exp 422)"
                        elif ".png" in path:
                            imc = np.array(Image.open(io.BytesIO(r_cw.content)).convert("RGBA"))
                            vm = imc[..., 3] == 255
                            cstd = float(imc[vm, :3].std(axis=0).max()) if vm.any() else 0.0
                            ok = imc.shape == (256, 256, 4) and int(vm.sum()) > 10000 and cstd > 10.0
                            detail = f"cw=200 shape={imc.shape} valid={int(vm.sum())} std={cstd:.1f} tt=422"
                        else:
                            ac = np.load(io.BytesIO(r_cw.content))
                            exp_dim = 262 if "buffer=3" in path else 256
                            vm = ac[1] > 0.0
                            vmin = float(ac[0][vm].min()) if vm.any() else -1.0
                            vmax = float(ac[0][vm].max()) if vm.any() else -1.0
                            ok = ac.shape == (2, exp_dim, exp_dim) and ac.dtype == np.float32 and int(vm.sum()) > 10000 and 0.0 <= vmin <= vmax <= 360.0 and (vmax - vmin) > 10.0
                            detail = f"cw=200 shape={ac.shape} range=[{vmin:.1f}..{vmax:.1f}] tt=422"
                    elif kind == "better_u16_sum_u64":
                        if r_cw.status_code != 200 or r_tt.status_code != 500:
                            ok, detail = False, f"HTTP cw={r_cw.status_code} (exp 200) tt={r_tt.status_code} (exp 500)"
                        else:
                            ac = np.load(io.BytesIO(r_cw.content))
                            vm = ac[1] > 0
                            vmin = int(ac[0][vm].min()) if vm.any() else 0
                            vmax = int(ac[0][vm].max()) if vm.any() else 0
                            ok = ac.shape == (2, 256, 256) and ac.dtype == np.uint64 and int(vm.sum()) > 10000 and vmin > 100000 and vmax < 160000
                            detail = f"cw=200 dtype={ac.dtype} valid={int(vm.sum())} sum_range=[{vmin}..{vmax}] tt=500"
                    elif kind == "better_cast_int16":
                        if r_cw.status_code != 200 or r_tt.status_code != 200:
                            ok, detail = False, f"HTTP cw={r_cw.status_code} tt={r_tt.status_code}"
                        else:
                            ac = np.load(io.BytesIO(r_cw.content))
                            at = np.load(io.BytesIO(r_tt.content))
                            vm = ac[1] > 0
                            vmin = int(ac[0][vm].min()) if vm.any() else 0
                            vmax = int(ac[0][vm].max()) if vm.any() else 0
                            ok = (
                                ac.shape == (2, 256, 256)
                                and ac.dtype == np.int16
                                and at.dtype == np.uint8
                                and int(vm.sum()) > 10000
                                and vmin > 300
                                and vmax < 900
                            )
                            detail = f"cw={ac.dtype} [{vmin}..{vmax}] vs tt={at.dtype} (hardcoded uint8)"
                    elif kind == "better_antimeridian_w47":
                        if r_cw.status_code != 200 or r_tt.status_code != 404:
                            ok, detail = False, f"HTTP cw={r_cw.status_code} (exp 200) tt={r_tt.status_code} (exp 404)"
                        elif ".npy" in path:
                            ac = np.load(io.BytesIO(r_cw.content))
                            vm = ac[1] > np.finfo(np.float32).min
                            vmin = float(ac[0][vm].min()) if vm.any() else -1.0
                            vmax = float(ac[0][vm].max()) if vm.any() else -1.0
                            ok = ac.shape == (2, 256, 256) and ac.dtype == np.float32 and int(vm.sum()) == 256 * 256 and 250.0 < vmin < 400.0 and 400.0 < vmax < 620.0
                            detail = f"cw=200 valid={int(vm.sum())} range=[{vmin:.1f}..{vmax:.1f}] tt=404"
                        else:
                            vc = r_cw.json().get("values", [])
                            ok = len(vc) == 1 and abs(vc[0] - 436.0) < 1e-3
                            detail = f"cw=200 values={vc} (exp 436.0) tt=404"
                    elif kind == "better_div0_w48":
                        if r_cw.status_code != 200 or r_tt.status_code != 200:
                            ok, detail = False, f"HTTP cw={r_cw.status_code} tt={r_tt.status_code}"
                        else:
                            ac, at = np.load(io.BytesIO(r_cw.content)), np.load(io.BytesIO(r_tt.content))
                            mc = ac[1] > np.finfo(ac.dtype).min
                            mt = at[1] > np.finfo(at.dtype).min
                            both = mc & mt & np.isfinite(ac[0]) & np.isfinite(at[0])
                            mae = float(np.abs(ac[0][both] - at[0][both]).mean()) if both.any() else 999.0
                            masked_div0 = int((mt & ~mc).sum())
                            ok = ac.shape == (2, 256, 256) and int(both.sum()) > 20000 and mae < 1e-4 and masked_div0 > 20000 and int(np.isinf(ac[0][mc]).sum()) == 0
                            detail = f"valid_MAE={mae:.5f} cw_masked_div0={masked_div0} cw_valid={int(mc.sum())} tt_valid={int(mt.sum())}"
                    elif kind == "better_stac_bbox_auto_asset":
                        if r_cw.status_code != 200 or r_tt.status_code != 422:
                            ok, detail = False, f"HTTP cw={r_cw.status_code} (exp 200) tt={r_tt.status_code} (exp 422)"
                        else:
                            ac = np.load(io.BytesIO(r_cw.content))
                            vm = ac[1] > np.finfo(ac.dtype).min
                            vmin = float(ac[0][vm].min()) if vm.any() else -999.0
                            vmax = float(ac[0][vm].max()) if vm.any() else -999.0
                            ok = ac.shape == (2, 64, 64) and ac.dtype == np.float32 and int(vm.sum()) == 64 * 64 and -1.0 <= vmin < vmax <= 1.0
                            detail = f"cw=200 shape={ac.shape} range=[{vmin:.4f}..{vmax:.4f}] tt=422"
                    elif kind == "better_expr_nan_masked":
                        if r_cw.status_code != 200 or r_tt.status_code != 200:
                            ok, detail = False, f"HTTP cw={r_cw.status_code} tt={r_tt.status_code}"
                        else:
                            imc = np.array(Image.open(io.BytesIO(r_cw.content)).convert("RGBA"))
                            imt = np.array(Image.open(io.BytesIO(r_tt.content)).convert("RGBA"))
                            mc, mt = imc[..., 3] > 0, imt[..., 3] > 0
                            both = mc & mt
                            mae = float(np.abs(imc[both, :3].astype(np.float32) - imt[both, :3].astype(np.float32)).mean()) if both.any() else 999.0
                            # On sqrt(b1) where b1 < 0: cogzilla masks NaN to alpha=0, while TiTiler paints opaque viridis(0).
                            only_tt = mt & ~mc
                            ok = (
                                imc.shape == (256, 256, 4)
                                and bool(both.any())
                                and mae < 0.5
                                and int(only_tt.sum()) > 10000
                                and int(imc[100, 30, 3]) == 0
                                and imt[100, 30].tolist() == [68, 1, 84, 255]
                            )
                            detail = f"valid_MAE={mae:.2f} cw_masked_nan_px={int(only_tt.sum())} tt_nan_rgba={imt[100,30].tolist()}"
                    elif kind == "tilejson_w18":
                        if r_cw.status_code != 200 or r_tt.status_code != 200:
                            ok, detail = False, f"HTTP cw={r_cw.status_code} tt={r_tt.status_code}"
                        else:
                            jc, jt = r_cw.json(), r_tt.json()
                            berr = max(abs(a - b) for a, b in zip(jc["bounds"], jt["bounds"]))
                            tile0 = jc["tiles"][0]
                            ok = (
                                berr < 1e-4
                                and jc["minzoom"] == jt["minzoom"]
                                and jc["maxzoom"] == jt["maxzoom"]
                                and "colormap_name=viridis" in tile0
                                and "rescale=0,300" in tile0
                            )
                            detail = f"bounds_err={berr:.6f} z={jc['minzoom']}..{jc['maxzoom']} vs {jt['minzoom']}..{jt['maxzoom']}"
                    elif r_cw.status_code != 200 or r_tt.status_code != 200:
                        ok = False
                        detail = f"HTTP cw={r_cw.status_code} ({r_cw.text[:100]!r}) vs tt={r_tt.status_code} ({r_tt.text[:100]!r})"
                    elif kind == "png":
                        ok, detail = compare_images(r_cw.content, r_tt.content)
                        if ok and is_unmasked:
                            im_c = Image.open(io.BytesIO(r_cw.content))
                            im_t = Image.open(io.BytesIO(r_tt.content))
                            no_a_c = "A" not in im_c.mode and "transparency" not in im_c.info
                            no_a_t = "A" not in im_t.mode and "transparency" not in im_t.info
                            if not (no_a_c and no_a_t):
                                ok = False
                                detail = f"expected no-alpha on return_mask=false: cw={im_c.mode} tt={im_t.mode}"
                            else:
                                detail += f" mode={im_c.mode}"
                    elif kind == "npy":
                        ok, detail = compare_npy(r_cw.content, r_tt.content, check_values=True, masked=not is_unmasked, smooth=is_smooth)
                    elif kind == "geotiff":
                        ok, detail = compare_geotiffs(r_cw.content, r_tt.content, compare_pixels=True, masked=not is_unmasked, smooth=is_smooth)
                    elif kind == "json_exact":
                        d = first_json_diff(r_cw.json(), r_tt.json())
                        ok, detail = (d is None), (d or "exact JSON match")
                    elif kind == "headers":
                        if not (r_cw.headers.get("content-crs") and r_cw.headers.get("content-bbox") and r_tt.headers.get("content-crs") and r_tt.headers.get("content-bbox")):
                            ok, detail = False, f"missing geo headers: cw=({r_cw.headers.get('content-crs')},{r_cw.headers.get('content-bbox')}) tt=({r_tt.headers.get('content-crs')},{r_tt.headers.get('content-bbox')})"
                        else:
                            ok, detail = same_place(r_cw, r_tt)
                    elif kind == "stats":
                        ok, detail = compare_stats_strict(r_cw.json(), r_tt.json())
                    elif kind == "stats_w23":
                        jc, jt = r_cw.json(), r_tt.json()
                        ok, detail = compare_stats_strict(jc, jt)
                        if ok and "categorical=true" in path:
                            hc, ht = jc["b1"].get("histogram", [[], []]), jt["b1"].get("histogram", [[], []])
                            ok = len(hc[0]) == len(ht[0]) and len(hc[0]) >= 3 and hc[1] == ht[1] and hc[0] == ht[0]
                            detail += f" hist_vals={hc[1]} counts={hc[0]}"
                        if ok and "&p=10" in path:
                            for pk in ("percentile_10", "percentile_50", "percentile_90"):
                                vc, vt = jc["b1"].get(pk), jt["b1"].get(pk)
                                if vc is None or vt is None or abs(vc - vt) > 0.5:
                                    ok = False
                                    detail += f" {pk}: cw={vc} tt={vt}"
                        if ok and "histogram_bins=5" in path:
                            hc, ht = jc["b1"]["histogram"], jt["b1"]["histogram"]
                            ok = len(hc[0]) == 5 and len(ht[0]) == 5 and max(abs(a - b) for a, b in zip(hc[1], ht[1])) < 1e-4
                            detail += f" bins={len(hc[0])} edges=[{hc[1][0]}..{hc[1][-1]}]"
                    elif kind == "better_w73_map_hillshade":
                        ac = np.array(Image.open(io.BytesIO(r_cw.content)).convert("RGBA"))
                        at = np.array(Image.open(io.BytesIO(r_tt.content)).convert("RGBA"))
                        interior_mae = float(np.abs(ac[3:61, 3:61, :3].astype(float) - at[:, :, :3].astype(float)).mean()) if ac.shape == (64, 64, 4) and at.shape == (58, 58, 4) else 999.0
                        ok = ac.shape == (64, 64, 4) and at.shape == (58, 58, 4) and interior_mae < 2.0 and float(ac[..., 0].std()) > 10.0
                        detail = f"cw={ac.shape} tt={at.shape} interior_MAE={interior_mae:.2f}"
                    elif kind == "better_w72_mirror_info":
                        jc, jt = r_cw.json(), r_tt.json()
                        pc, pt = jc["properties"], jt["properties"]
                        bc, bt = pc["bounds"], pt["bounds"]
                        ok = (
                            jc.get("type") == "Feature"
                            and jc.get("geometry", {}).get("type") == "Polygon"
                            and len(bc) == 4 and len(bt) == 4
                            and bc[0] < bc[2] and bt[0] > bt[2]
                            and max(abs(a - b) for a, b in zip(bc, [-122.6, 37.4, -121.9, 38.1])) < 1e-6
                            and pc["count"] == pt["count"] and pc["dtype"] == pt["dtype"]
                        )
                        detail = f"cw bounds={[round(x, 2) for x in bc]} vs tt bounds={[round(x, 2) for x in bt]}"
                    elif kind == "info_geojson_w23":
                        jc, jt = r_cw.json(), r_tt.json()
                        ok = (
                            jc.get("type") == "Feature"
                            and jc.get("geometry", {}).get("type") == "Polygon"
                            and jc["properties"]["count"] == jt["properties"]["count"]
                            and jc["properties"]["dtype"] == jt["properties"]["dtype"]
                            and jc["properties"]["bounds"] == jt["properties"]["bounds"]
                        )
                        detail = f"type={jc.get('type')} count={jc['properties']['count']} dtype={jc['properties']['dtype']}"
                    elif kind == "stac_info_geojson":
                        jc, jt = r_cw.json(), r_tt.json()
                        pc, pt = jc.get("properties", {}), jt.get("properties", {})
                        ok = (
                            jc.get("type") == "Feature"
                            and jc.get("geometry", {}).get("type") == "Polygon"
                            and set(pc.keys()) == set(pt.keys())
                            and len(pc) > 0
                            and all(
                                pc[k].get("count") == pt[k].get("count")
                                and pc[k].get("dtype") == pt[k].get("dtype")
                                and pc[k].get("width") == pt[k].get("width")
                                and pc[k].get("height") == pt[k].get("height")
                                and pc[k].get("bounds") == pt[k].get("bounds")
                                for k in pt
                            )
                        )
                        detail = f"type={jc.get('type')} assets={sorted(pc.keys())}"
                    elif kind == "post_fc_stats_w23":
                        fc_c, fc_t = r_cw.json(), r_tt.json()
                        ok = fc_c.get("type") == "FeatureCollection" and len(fc_c.get("features", [])) == 2
                        dets = []
                        for fi in range(2):
                            sc = fc_c["features"][fi]["properties"]["statistics"]
                            st = fc_t["features"][fi]["properties"]["statistics"]
                            fok, fdet = compare_stats_strict(sc, st)
                            ok = ok and fok
                            dets.append(f"f{fi}:{fdet}")
                        detail = " | ".join(dets)
                    elif kind == "post_stats":
                        ok, detail = compare_stats_strict(r_cw.json()["properties"]["statistics"], r_tt.json()["properties"]["statistics"])
                    elif kind == "point":
                        ok, detail = compare_point_strict(r_cw.json(), r_tt.json())
                    else:
                        raise ValueError(f"Unknown check kind: {kind!r}")

                    tag = "PASS" if ok else "FAIL"
                    if not ok:
                        failures.append({"label": label, "method": method, "path": path, "kind": kind, "detail": detail})
                    print(f"{tag} | {label:54s} | {detail}")

                w7_ran = run_wave7_checks(client, cw_base, gpu_base, tmp_dir, args.only, failures, gpu_diffs)
                w8_ran = run_wave8_checks(client, cw_base, gpu_base, tmp_dir, args.only, failures, gpu_diffs)
                w14_ran = run_wave14_fault_checks(client, cw_base, tmp_dir, s, args.only, failures)
                w15_ran = run_wave15_resilience_checks(client, cw_base, gpu_base, tmp_dir, args.only, failures, gpu_diffs)
                w17_ran = run_wave17_zarr_deep_checks(client, cw_base, gpu_base, tmp_dir, args.only, failures, gpu_diffs)
                w26_ran = run_wave26_security_checks(client, cw_base, bin_path, tmp_dir, args.only, failures)
                w28_ran = run_wave28_wms_and_mosaic_checks(client, cw_base, gpu_base, tmp_dir, args.only, failures, gpu_diffs)
                w29_ran = run_wave29_http_protocol_checks(client, cw_base, s, args.only, failures)
                w30_ran = run_wave30_zarr_codecs_multiurl_rpc_checks(client, cw_base, gpu_base, tmp_dir, args.only, failures, gpu_diffs)
                w18_extra = 0
                lb_wmts = "W18 WMTS Capabilities XML WebMercatorQuad"
                if not args.only or args.only.lower() in lb_wmts.lower():
                    w18_extra = 1
                    wmts_path = f"/cog/WebMercatorQuad/WMTSCapabilities.xml?url={q(s['w18_cog3b'])}&rescale=0,300&colormap_name=viridis"
                    rw = client.get(cw_base + wmts_path)
                    if rw.status_code == 200:
                        root_xml = ET.fromstring(rw.content)
                        ok_w = root_xml.tag.endswith("Capabilities") and "WebMercatorQuad" in rw.text and "colormap_name=viridis" in rw.text
                        det_w = f"root={root_xml.tag.split('}')[-1]} len={len(rw.text)}"
                    else:
                        ok_w, det_w = False, f"HTTP {rw.status_code} {rw.text[:80]!r}"
                    if not ok_w:
                        failures.append({"label": lb_wmts, "method": "GET", "path": wmts_path, "kind": "w18_wmts", "detail": det_w})
                    print(f"{'PASS' if ok_w else 'FAIL'} | {lb_wmts:54s} | {det_w}")

            total_checks = len(cases) + w7_ran + w8_ran + w14_ran + w15_ran + w17_ran + w26_ran + w28_ran + w29_ran + w30_ran + w18_extra

            if gpu_diffs:
                print("\nCPU-vs-GPU differences:")
                for gd in gpu_diffs:
                    print(f"  GPU_DIFF | {gd['label']:54s} | {gd['detail']}")

            if args.json_out:
                Path(args.json_out).write_text(json.dumps({
                    "mode": mode_str,
                    "total": total_checks,
                    "failed": len(failures),
                    "gpu_diffs": gpu_diffs,
                    "failures": failures,
                }, indent=2))
            print(f"\nSummary ({mode_str}): {total_checks - len(failures)}/{total_checks} passed, {len(failures)} failed" + (f", {len(gpu_diffs)} CPU-vs-GPU diffs" if args.check_cpu_vs_gpu else ""))
            return 1 if (failures or gpu_diffs) else 0
        finally:
            for p in procs:
                try:
                    if p.poll() is None:
                        p.terminate()
                except OSError:
                    pass
            for p in procs:
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:
                        p.kill()
                        p.wait(timeout=5)
                    except OSError:
                        pass


if __name__ == "__main__":
    sys.exit(main())
