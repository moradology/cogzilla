#!/usr/bin/env python3
"""Shared differential-oracle comparison library for cogzilla (#505).

Provides one set of response-comparison and server-readiness primitives used across
`titiler_diff_oracle.py`, `pgstac_diff_oracle.py`, `zarr_parity_oracle.py`, and
`parity_gauntlet.py`, so parity verdicts and tolerances are defined once and shared
across all route families (`/cog`, `/stac`, `/mosaicjson`, `/searches`, `/zarr`).
"""

from __future__ import annotations

import io
import json
import math
import time
import warnings
import xml.etree.ElementTree as ET
from typing import Any

Diff = tuple[bool, str]

DEFAULT_VOLATILE_KEYS: frozenset[str] = frozenset({"lastused", "usecount"})


def wait_ready(url: str, timeout_s: float = 15.0, *, poll_s: float = 0.1, req_timeout_s: float = 1.0) -> None:
    """Poll `url` until it returns HTTP 200 or `timeout_s` elapses."""
    import httpx

    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            r = httpx.get(url, timeout=req_timeout_s)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(poll_s)
    raise RuntimeError(f"Server at {url} did not become ready in {timeout_s}s")


def time_request(client: Any, method: str, url: str, json_body: Any = None, iters: int = 5):
    """Run one warm-up request plus `iters` timed requests; return `(last_response, median_ms)`."""
    r = client.request(method, url, json=json_body)
    if r.status_code != 200:
        return r, 0.0
    samples: list[float] = []
    last = r
    for _ in range(iters):
        t0 = time.perf_counter()
        last = client.request(method, url, json=json_body)
        samples.append((time.perf_counter() - t0) * 1e3)
    samples.sort()
    return last, samples[len(samples) // 2]


def compare_images(
    a: bytes,
    b: bytes,
    *,
    min_mask_pct: float = 99.0,
    min_psnr: float = 25.0,
) -> Diff:
    """Compare two encoded raster images (PNG/JPEG/WebP) by shape, alpha mask agreement, MAE, and PSNR."""
    import numpy as np
    from PIL import Image

    try:
        im_a = np.array(Image.open(io.BytesIO(a)).convert("RGBA"), dtype=np.float32)
        im_b = np.array(Image.open(io.BytesIO(b)).convert("RGBA"), dtype=np.float32)
    except Exception as e:  # noqa: BLE001 - an undecodable image is a failed check
        return False, f"undecodable image: {e}"
    if im_a.shape != im_b.shape:
        return False, f"SHAPE DIFF cw={im_a.shape} vs tt={im_b.shape}"
    ma = im_a[:, :, 3] > 0
    mb = im_b[:, :, 3] > 0
    mask_pct = float(np.mean(ma == mb)) * 100.0
    both = ma & mb
    if np.any(both):
        diff = im_a[:, :, :3][both] - im_b[:, :, :3][both]
        mae = float(np.mean(np.abs(diff)))
        mse = float(np.mean(diff * diff))
        psnr = 99.0 if mse < 1e-9 else 20.0 * math.log10(255.0 / math.sqrt(mse))
        adiff = im_a[:, :, 3][both] - im_b[:, :, 3][both]
        amae = float(np.mean(np.abs(adiff)))
        amse = float(np.mean(adiff * adiff))
        apsnr = 99.0 if amse < 1e-9 else 20.0 * math.log10(255.0 / math.sqrt(amse))
    else:
        mae, psnr, amae, apsnr = 0.0, 99.0, 0.0, 99.0
    ok = mask_pct >= min_mask_pct and psnr >= min_psnr and apsnr >= min_psnr
    alpha_note = f" aMAE={amae:.2f} aPSNR={apsnr:.1f}dB" if amae > 0.0 else ""
    return ok, f"shape={im_a.shape[:2]} mask={mask_pct:.2f}% MAE={mae:.2f} PSNR={psnr:.1f}dB{alpha_note}"


def compare_arrays(
    a_bytes: bytes,
    b_bytes: bytes,
    *,
    smooth: bool = False,
    masked: bool = True,
    rtol: float = 0.02,
    atol: float = 1e-9,
    min_mask_pct: float = 99.0,
) -> Diff:
    """Compare two `.npy` raster arrays: dtype, shape, mask band agreement, and numeric tolerance.

    When `masked=False` (`return_mask=false`), a pixel counts as valid where any band of the
    reference array is non-zero.
    """
    import numpy as np

    try:
        a, b = np.load(io.BytesIO(a_bytes)), np.load(io.BytesIO(b_bytes))
    except Exception as e:  # noqa: BLE001 - an unreadable array is a failed check
        return False, f"unreadable array: {e}"
    if a.dtype != b.dtype or a.shape != b.shape:
        return False, f"{a.dtype} {a.shape} vs {b.dtype} {b.shape}"
    if not masked:
        there = (b != 0).any(axis=0)
        top = np.iinfo(a.dtype).max if a.dtype.kind in "ui" else 1
        a = np.concatenate([a, (there * top).astype(a.dtype)[None]])
        b = np.concatenate([b, (there * top).astype(b.dtype)[None]])
        ma, mb = a[-1] != 0, b[-1] != 0
    else:
        # Signed int and float `.npy` mask bands use dtype min for masked and dtype max for valid
        # (unsigned ints use 0 for masked and 255 / 65535 for valid).
        low = np.finfo(a.dtype).min if a.dtype.kind == "f" else np.iinfo(a.dtype).min
        ma, mb = a[-1] > low, b[-1] > low
    mask = float((ma == mb).mean()) * 100.0
    tops = set(np.unique(a[-1][ma]).tolist()) == set(np.unique(b[-1][mb]).tolist())
    both = ma & mb
    if not both.any():
        return mask >= min_mask_pct and tops, f"{a.dtype} {a.shape}, mask {mask:.2f}%"
    va, vb = a[:-1][:, both].astype(np.float64), b[:-1][:, both].astype(np.float64)
    if a.dtype.kind == "f":
        # Halved first: FLT_MAX / DBL_MAX sentinel values overflow an unscaled sum.
        ha, hb = (va / 2.0).mean(), (vb / 2.0).mean()
        close = ha == hb or abs(ha - hb) <= rtol * abs(hb) + atol
        return (
            mask >= min_mask_pct and tops and close,
            f"{a.dtype} {a.shape}, mask {mask:.2f}%, mean {va.mean():.5g} vs {vb.mean():.5g}",
        )
    # Band by band: one wrong band is not hidden by the others.
    same = min(float((va[i] == vb[i]).mean()) * 100.0 for i in range(va.shape[0]))
    if smooth:
        off = max(
            float(np.abs(va[i] - vb[i]).mean() / (np.abs(vb[i]).mean() + 1e-9))
            for i in range(va.shape[0])
        ) * 100.0
        return (
            mask >= min_mask_pct and tops and off <= 10.0,
            f"{a.dtype} {a.shape}, mask {mask:.2f}%, values {off:.1f}% apart in the worst band",
        )
    return (
        mask >= min_mask_pct and tops and same >= 75.0,
        f"{a.dtype} {a.shape}, mask {mask:.2f}%, {same:.1f}% of values equal in the worst band",
    )


def compare_npy(
    a_bytes: bytes,
    b_bytes: bytes,
    *,
    check_values: bool = False,
    smooth: bool = False,
    masked: bool = True,
) -> Diff:
    """Compare two `.npy` payloads (shape-only by default, or full pixel/mask check when `check_values=True`)."""
    if check_values:
        return compare_arrays(a_bytes, b_bytes, smooth=smooth, masked=masked)
    import numpy as np

    try:
        a_cw = np.load(io.BytesIO(a_bytes))
        a_tt = np.load(io.BytesIO(b_bytes))
    except Exception as e:  # noqa: BLE001
        return False, f"unreadable array: {e}"
    if a_cw.shape != a_tt.shape:
        return False, f"shape cw={a_cw.shape} vs tt={a_tt.shape}"
    return True, f"cw={a_cw.shape}:{a_cw.dtype} tt={a_tt.shape}:{a_tt.dtype}"


def geotiff_array(body: bytes) -> tuple[bytes, str]:
    """Decode a GeoTIFF into a `.npy` buffer and a summary of its dtype, bands, colorinterp, EPSG, and bounds."""
    import numpy as np
    from rasterio.io import MemoryFile

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with MemoryFile(body) as memory, memory.open() as ds:
            said = (
                f"{ds.dtypes[0]} x{ds.count} {[c.name for c in ds.colorinterp]} "
                f"epsg {ds.crs.to_epsg() if ds.crs else None} {[round(v, 3 if abs(v) > 1000 else 6) for v in ds.bounds]}"
            )
            out = io.BytesIO()
            np.save(out, ds.read())
            return out.getvalue(), said


def compare_geotiffs(
    a_bytes: bytes,
    b_bytes: bytes,
    *,
    compare_pixels: bool = False,
    smooth: bool = False,
    masked: bool = True,
    bounds_rel_tol: float = 1e-3,
) -> Diff:
    """Compare two GeoTIFFs.

    When `compare_pixels=True`, verifies exact metadata summary (dtype, band count, colorinterp,
    EPSG, rounded bounds) plus pixel agreement via `compare_arrays`.
    When `compare_pixels=False`, verifies dimensions, EPSG, and bounds within `bounds_rel_tol`.
    """
    if compare_pixels:
        try:
            (arr_a, said_a), (arr_b, said_b) = geotiff_array(a_bytes), geotiff_array(b_bytes)
        except Exception as e:  # noqa: BLE001 - an unreadable GeoTIFF is a failed check
            return False, f"unreadable GeoTIFF: {e}"
        if said_a != said_b:
            return False, f"{said_a} vs {said_b}"
        ok, note = compare_arrays(arr_a, arr_b, smooth=smooth, masked=masked)
        return ok, f"GeoTIFF, {note}"

    from rasterio.io import MemoryFile

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with MemoryFile(a_bytes) as ma, ma.open() as da, MemoryFile(b_bytes) as mb, mb.open() as db:
                if (da.height, da.width) != (db.height, db.width):
                    return False, f"shape cw={(da.height, da.width)} vs tt={(db.height, db.width)}"
                a_epsg = da.crs.to_epsg() if da.crs else None
                b_epsg = db.crs.to_epsg() if db.crs else None
                if a_epsg != b_epsg:
                    return False, f"epsg cw={a_epsg} vs tt={b_epsg}"
                b_err = max(abs(x - y) for x, y in zip(da.bounds, db.bounds))
                scale = max(1.0, abs(db.bounds.right - db.bounds.left))
                if b_err / scale > bounds_rel_tol:
                    return False, f"bounds_err={b_err:.4f} (rel={b_err / scale:.4f})"
                return (
                    True,
                    f"shape={(da.height, da.width)} epsg={a_epsg} bands={da.count}/{db.count} dtype={da.dtypes[0]}/{db.dtypes[0]}",
                )
    except Exception as e:  # noqa: BLE001
        return False, f"unreadable GeoTIFF: {e}"


def normalize_json(
    value: Any,
    base: str = "",
    *,
    ignore: set[str] | frozenset[str] = DEFAULT_VOLATILE_KEYS,
) -> Any:
    """Recursively drop `ignore` keys and replace server-specific `base` URLs with `{base}`."""
    if isinstance(value, dict):
        return {k: normalize_json(v, base, ignore=ignore) for k, v in value.items() if k not in ignore}
    if isinstance(value, list):
        return [normalize_json(v, base, ignore=ignore) for v in value]
    if isinstance(value, str) and base:
        return value.replace(base, "{base}")
    return value


def first_json_diff(a: Any, b: Any, path: str = "$", *, rel_tol: float = 1e-9, abs_tol: float = 1e-9) -> str | None:
    """Return a human-readable path to the first structural/value difference, or `None` if equal."""
    if type(a) is not type(b) and not (
        isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool) and not isinstance(b, bool)
    ):
        return f"{path}: {type(a).__name__} vs {type(b).__name__}"
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                return f"{path}.{k}: missing in cogzilla"
            if k not in b:
                return f"{path}.{k}: only in cogzilla"
            d = first_json_diff(a[k], b[k], f"{path}.{k}", rel_tol=rel_tol, abs_tol=abs_tol)
            if d:
                return d
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: {len(a)} vs {len(b)} entries"
        for i, (x, y) in enumerate(zip(a, b)):
            d = first_json_diff(x, y, f"{path}[{i}]", rel_tol=rel_tol, abs_tol=abs_tol)
            if d:
                return d
        return None
    if isinstance(a, float) or isinstance(b, float):
        return None if math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol) else f"{path}: {a} vs {b}"
    return None if a == b else f"{path}: {str(a)[:60]!r} vs {str(b)[:60]!r}"


def compare_json(
    a: Any,
    b: Any,
    *,
    base_a: str = "",
    base_b: str = "",
    ignore: set[str] | frozenset[str] = DEFAULT_VOLATILE_KEYS,
    rel_tol: float = 1e-9,
    abs_tol: float = 1e-9,
) -> Diff:
    """Normalize and structurally compare two JSON objects (or JSON-encoded bytes/strings)."""
    try:
        obj_a = json.loads(a) if isinstance(a, (bytes, bytearray, str)) else a
        obj_b = json.loads(b) if isinstance(b, (bytes, bytearray, str)) else b
    except ValueError as e:
        return False, f"not JSON: {e}"
    na = normalize_json(obj_a, base_a, ignore=ignore)
    nb = normalize_json(obj_b, base_b, ignore=ignore)
    diff = first_json_diff(na, nb, rel_tol=rel_tol, abs_tol=abs_tol)
    return diff is None, diff or "JSON bodies equal"


def wmts_view(text: str, base: str, *, tms_id: str = "WebMercatorQuad") -> dict[str, Any]:
    """Parse a WMTS capabilities XML into plain lists, keeping only `tms_id` layers/matrices."""
    wm = "{http://www.opengis.net/wmts/1.0}"
    ows = "{http://www.opengis.net/ows/1.1}"

    def plain(el: ET.Element):
        attrs = {k: v.replace(base, "{base}") for k, v in sorted(el.attrib.items())}
        txt: Any = (el.text or "").strip().replace(base, "{base}")
        if el.tag.endswith("Corner"):
            txt = [float(v) for v in txt.split()]
        return [el.tag, attrs, txt, [plain(c) for c in el]]

    root = ET.fromstring(text)
    contents = root.find(f"{wm}Contents")
    layers = [
        layer
        for layer in contents.findall(f"{wm}Layer")
        if layer.find(f"{wm}TileMatrixSetLink/{wm}TileMatrixSet").text == tms_id
    ]
    matrix_sets = [
        tms
        for tms in contents.findall(f"{wm}TileMatrixSet")
        if tms.find(f"{ows}Identifier").text == tms_id
    ]
    rest = [plain(c) for c in root if c.tag != f"{wm}Contents"]
    return {
        "root": [root.tag, dict(sorted(root.attrib.items()))],
        "outside_contents": rest,
        "layers": [plain(layer) for layer in layers],
        "tile_matrix_sets": [plain(tms) for tms in matrix_sets],
    }


def compare_wmts_xml(
    a: bytes | str,
    b: bytes | str,
    *,
    base_a: str | None = None,
    base_b: str | None = None,
    tms_id: str = "WebMercatorQuad",
) -> Diff:
    """Compare two WMTS Capabilities XML documents.

    When `base_a` and `base_b` are provided, performs a deep structural diff of the `tms_id`
    layers and tile matrix sets (stripping server-specific base URLs). Otherwise verifies that
    the advertised `TileMatrixSetLink` list matches and reports matrix counts.
    """
    if base_a is not None and base_b is not None:
        text_a = a.decode("utf-8") if isinstance(a, (bytes, bytearray)) else a
        text_b = b.decode("utf-8") if isinstance(b, (bytes, bytearray)) else b
        va, vb = wmts_view(text_a, base_a, tms_id=tms_id), wmts_view(text_b, base_b, tms_id=tms_id)
        diff = first_json_diff(va, vb)
        return diff is None, diff or f"{len(va['layers'])} layer(s), {tms_id} parts equal"

    rcw = ET.fromstring(a)
    rtt = ET.fromstring(b)
    ns = {"wmts": "http://www.opengis.net/wmts/1.0", "ows": "http://www.opengis.net/ows/1.1"}
    cw_tms = [e.text for e in rcw.findall(".//wmts:TileMatrixSetLink/wmts:TileMatrixSet", ns)]
    tt_tms = [e.text for e in rtt.findall(".//wmts:TileMatrixSetLink/wmts:TileMatrixSet", ns)]
    if cw_tms != tt_tms:
        return False, f"TMS link cw={cw_tms} vs tt={tt_tms}"
    cw_mats = len(rcw.findall(".//wmts:TileMatrix", ns))
    tt_mats = len(rtt.findall(".//wmts:TileMatrix", ns))
    return True, f"tms={cw_tms} matrices=cw:{cw_mats}/tt:{tt_mats}"


def same_place(resp_a: Any, resp_b: Any, *, bbox_rel_tol: float = 1e-6) -> Diff:
    """Verify `Content-Crs` is equal and `Content-Bbox` agrees within `bbox_rel_tol`."""
    crs_a, crs_b = resp_a.headers.get("content-crs"), resp_b.headers.get("content-crs")
    if crs_a != crs_b:
        return False, f"content-crs {crs_a} vs {crs_b}"
    box_a, box_b = resp_a.headers.get("content-bbox"), resp_b.headers.get("content-bbox")
    if (box_a is None) != (box_b is None):
        return False, f"content-bbox {box_a} vs {box_b}"
    if box_a is None:
        return True, "neither placed"
    va, vb = [float(v) for v in box_a.split(",")], [float(v) for v in box_b.split(",")]
    if len(va) != 4 or any(abs(x - y) > bbox_rel_tol * max(1.0, abs(y)) for x, y in zip(va, vb)):
        return False, f"content-bbox {box_a} vs {box_b}"
    return True, f"{crs_a.rsplit('/', 1)[-1]} {box_a}"


def stats_diff(a: Any, b: Any, path: str = "$") -> str | None:
    """Compare statistics JSON structurally and summary numbers loosely (5% of reference, or 2 units)."""
    if isinstance(b, dict):
        if not isinstance(a, dict) or set(a) != set(b):
            return f"{path}: keys differ: {sorted(set(a) ^ set(b)) if isinstance(a, dict) else type(a).__name__}"
        for k in b:
            if k in (
                "histogram",
                "majority",
                "minority",
                "unique",
                "count",
                "sum",
                "masked_pixels",
                "valid_pixels",
                "valid_percent",
            ):
                continue
            d = stats_diff(a[k], b[k], f"{path}.{k}")
            if d:
                return d
        return None
    if isinstance(b, (int, float)) and not isinstance(b, bool):
        if not isinstance(a, (int, float)) or abs(a - b) > max(2.0, 0.05 * abs(b)):
            return f"{path}: {a} vs {b}"
        return None
    return None if a == b else f"{path}: {a!r} vs {b!r}"


def zonal_diff(a: Any, b: Any, path: str = "$", key: str = "") -> str | None:
    """Compare exact zonal statistics JSON (1e-6 rel tol for mean/count/sum/std, 1e-9 for other numbers)."""
    if isinstance(b, dict):
        if not isinstance(a, dict) or set(a) != set(b):
            return f"{path}: keys differ: {sorted(set(a) ^ set(b)) if isinstance(a, dict) else type(a).__name__}"
        for k in b:
            d = zonal_diff(a[k], b[k], f"{path}.{k}", k)
            if d:
                return d
        return None
    if isinstance(b, list):
        if not isinstance(a, list) or len(a) != len(b):
            return f"{path}: lists differ"
        for i, (x, y) in enumerate(zip(a, b)):
            d = zonal_diff(x, y, f"{path}[{i}]", key)
            if d:
                return d
        return None
    if isinstance(b, (int, float)) and not isinstance(b, bool):
        tol = 1e-6 if key in ("mean", "count", "sum", "std") else 1e-9
        if not isinstance(a, (int, float)) or abs(a - b) > tol * max(1.0, abs(b)):
            return f"{path}: {a!r} vs {b!r}"
        return None
    return None if a == b else f"{path}: {a!r} vs {b!r}"


def compare_response(content_type: str, a: bytes, b: bytes, **kw: Any) -> Diff:
    """Dispatch response comparison by `content_type` (and payload magic bytes)."""
    ct = (content_type or "").split(";")[0].strip().lower()
    if a[:6] == b"\x93NUMPY" or b[:6] == b"\x93NUMPY" or ct == "application/x-binary":
        return compare_arrays(
            a,
            b,
            smooth=kw.get("smooth", False),
            masked=kw.get("masked", True),
            rtol=kw.get("rtol", 0.02),
            atol=kw.get("atol", 1e-9),
            min_mask_pct=kw.get("min_mask_pct", 99.0),
        )
    if a[:4] in (b"II*\x00", b"MM\x00*") or b[:4] in (b"II*\x00", b"MM\x00*") or "tiff" in ct:
        return compare_geotiffs(
            a,
            b,
            compare_pixels=kw.get("compare_pixels", True),
            smooth=kw.get("smooth", False),
            masked=kw.get("masked", True),
            bounds_rel_tol=kw.get("bounds_rel_tol", 1e-3),
        )
    if ct.startswith("image/"):
        ok, note = compare_images(
            a,
            b,
            min_mask_pct=kw.get("min_mask_pct", 99.0),
            min_psnr=kw.get("min_psnr", 25.0),
        )
        return ok, f"{ct}, {note}"
    if ct in ("application/xml", "text/xml") or ct.endswith("+xml"):
        return compare_wmts_xml(
            a,
            b,
            base_a=kw.get("base_a"),
            base_b=kw.get("base_b"),
            tms_id=kw.get("tms_id", "WebMercatorQuad"),
        )
    if ct == "application/json" or ct.endswith("+json"):
        return compare_json(
            a,
            b,
            base_a=kw.get("base_a", ""),
            base_b=kw.get("base_b", ""),
            ignore=kw.get("ignore", DEFAULT_VOLATILE_KEYS),
        )
    if a == b:
        return True, f"bytes identical ({len(a)} B)"
    return False, f"bytes differ ({len(a)} B vs {len(b)} B)"
