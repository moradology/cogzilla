struct ExprParams {
    src_w: u32,
    src_h: u32,
    src_stride: u32,
    band_stride: u32,
    dst_w: u32,
    dst_h: u32,
    num_bands: u32,
    num_instrs: u32,
    /// 0 = Bilinear, 1 = Cubic, 2 = Lanczos3
    resample_mode: u32,
    /// Bit 0: apply RGB LUT, Bit 1: apply LCH saturation
    formula_flags: u32,
    /// 0 = Separable1D, 1 = Grid2D (17x17 bilinear), 2 = Exact2D (dst_w * dst_h)
    coord_mode: u32,
    _pad1: u32,
    rescale_min: f32,
    inv_range_255: f32,
    inv_range_254: f32,
    saturation: f32,
    rgb_min: vec4<f32>,
    rgb_inv_255: vec4<f32>,
    rgb_nodata_fill: vec4<f32>,
}

@group(0) @binding(0) var<uniform> params: ExprParams;
/// Stacked planar f32 slabs: band b at offset b * params.band_stride
@group(0) @binding(1) var<storage, read> bands_slab: array<f32>;
@group(0) @binding(2) var<storage, read> u_coords: array<f32>;
@group(0) @binding(3) var<storage, read> v_coords: array<f32>;
@group(0) @binding(4) var<storage, read> instrs: array<GpuInstr>;
/// 256 packed RGBA8 u32 entries (for expr colormap) OR 256 packed RGB-LUT u32 entries (r_lut | g_lut<<8 | b_lut<<16)
@group(0) @binding(5) var<storage, read> lut_u32: array<u32>;
@group(0) @binding(6) var<storage, read_write> dst_words: array<u32>;

fn sample_checked(band: u32, x: i32, y: i32) -> f32 {
    if (x < 0 || y < 0 || x >= i32(params.src_w) || y >= i32(params.src_h)) {
        return nan_f32();
    }
    return bands_slab[band * params.band_stride + u32(y) * params.src_stride + u32(x)];
}

fn sample_bilinear_raw(band: u32, u: f32, v: f32) -> f32 {
    let x0_f = floor(u);
    let y0_f = floor(v);
    let x0 = i32(x0_f);
    let y0 = i32(y0_f);
    let fx = u - x0_f;
    let fy = v - y0_f;

    let v00 = sample_checked(band, x0, y0);
    let v10 = sample_checked(band, x0 + 1, y0);
    let v01 = sample_checked(band, x0, y0 + 1);
    let v11 = sample_checked(band, x0 + 1, y0 + 1);

    let w00 = (1.0 - fx) * (1.0 - fy);
    let w10 = fx * (1.0 - fy);
    let w01 = (1.0 - fx) * fy;
    let w11 = fx * fy;

    let w_lim = i32(params.src_w);
    let h_lim = i32(params.src_h);
    let in0x = x0 >= 0 && x0 < w_lim;
    let in1x = x0 + 1 >= 0 && x0 + 1 < w_lim;
    let in0y = y0 >= 0 && y0 < h_lim;
    let in1y = y0 + 1 >= 0 && y0 + 1 < h_lim;

    var sum = 0.0;
    var wsum = 0.0;
    var wtot = 0.0;
    if (in0x && in0y) { wtot += w00; if (!is_nan_f32(v00)) { sum += v00 * w00; wsum += w00; } }
    if (in1x && in0y) { wtot += w10; if (!is_nan_f32(v10)) { sum += v10 * w10; wsum += w10; } }
    if (in0x && in1y) { wtot += w01; if (!is_nan_f32(v01)) { sum += v01 * w01; wsum += w01; } }
    if (in1x && in1y) { wtot += w11; if (!is_nan_f32(v11)) { sum += v11 * w11; wsum += w11; } }

    if (wsum >= 0.5 * wtot && wsum > 1e-6) {
        return sum / wsum;
    }
    return nan_f32();
}

fn sample_cubic(band: u32, u: f32, v: f32) -> f32 {
    if (!in_sample_extent(u, v)) {
        return nan_f32();
    }
    let x0_f = floor(u);
    let y0_f = floor(v);
    let x0 = i32(x0_f);
    let y0 = i32(y0_f);
    let wx = cubic_weights(u - x0_f);
    let wy = cubic_weights(v - y0_f);

    var acc = 0.0;
    for (var j: i32 = 0; j < 4; j = j + 1) {
        let ry = y0 + j - 1;
        var row_acc = 0.0;
        for (var i: i32 = 0; i < 4; i = i + 1) {
            let s = sample_checked(band, x0 + i - 1, ry);
            if (is_nan_f32(s)) {
                return sample_bilinear_raw(band, u, v);
            }
            row_acc += s * wx[i];
        }
        acc += row_acc * wy[j];
    }
    return acc;
}

fn sample_lanczos3(band: u32, u: f32, v: f32) -> f32 {
    if (!in_sample_extent(u, v)) {
        return nan_f32();
    }
    let x0_f = floor(u);
    let y0_f = floor(v);
    let x0 = i32(x0_f);
    let y0 = i32(y0_f);
    let fx = u - x0_f;
    let fy = v - y0_f;

    var wx: array<f32, 6>;
    var wy: array<f32, 6>;
    for (var k: i32 = 0; k < 6; k = k + 1) {
        let d = f32(k - 2);
        wx[k] = lanczos3_weight(d - fx);
        wy[k] = lanczos3_weight(d - fy);
    }

    var sum = 0.0;
    var wsum = 0.0;
    for (var j: i32 = 0; j < 6; j = j + 1) {
        let ry = y0 + j - 2;
        let wyj = wy[j];
        for (var i: i32 = 0; i < 6; i = i + 1) {
            let s = sample_checked(band, x0 + i - 2, ry);
            if (!is_nan_f32(s)) {
                let w = wx[i] * wyj;
                sum += s * w;
                wsum += w;
            }
        }
    }

    if (wsum >= 0.5) {
        return sum / wsum;
    }
    return sample_bilinear_raw(band, u, v);
}

fn sample_band(band: u32, u: f32, v: f32) -> f32 {
    if (params.resample_mode == 0u) {
        if (!in_sample_extent(u, v)) {
            return nan_f32();
        }
        return sample_bilinear_raw(band, u, v);
    } else if (params.resample_mode == 1u) {
        return sample_cubic(band, u, v);
    } else if (params.resample_mode == 3u) {
        if (!in_sample_extent(u, v)) {
            return nan_f32();
        }
        return sample_checked(band, i32(floor(u + 0.5)), i32(floor(v + 0.5)));
    } else {
        return sample_lanczos3(band, u, v);
    }
}

fn eval_expr_vm(u: f32, v: f32) -> f32 {
    var bvals: array<f32, 8>;
    let nb = min(params.num_bands, 8u);
    for (var b: u32 = 0u; b < nb; b = b + 1u) {
        bvals[b] = sample_band(b, u, v);
    }
    return eval_expr_vm_range(&bvals, 0u, params.num_instrs);
}

@compute @workgroup_size(16, 16, 1)
fn expr_eval_rgba8(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x;
    let py = gid.y;
    if (px >= params.dst_w || py >= params.dst_h) {
        return;
    }
    let uv = eval_uv(px, py);
    let out_idx = py * params.dst_w + px;
    if (!is_finite_f32(uv.x) || !is_finite_f32(uv.y)) {
        dst_words[out_idx] = 0u;
        return;
    }
    let val = eval_expr_vm(uv.x, uv.y);
    if (!is_finite_f32(val)) {
        dst_words[out_idx] = 0u;
    } else {
        let scaled = clamp((val - params.rescale_min) * params.inv_range_255, 0.0, 255.0);
        dst_words[out_idx] = lut_u32[u32(scaled)];
    }
}

@compute @workgroup_size(16, 16, 1)
fn expr_eval_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x;
    let py = gid.y;
    if (px >= params.dst_w || py >= params.dst_h) {
        return;
    }
    let uv = eval_uv(px, py);
    if (!is_finite_f32(uv.x) || !is_finite_f32(uv.y)) {
        dst_words[py * params.dst_w + px] = bitcast<u32>(nan_f32());
        return;
    }
    let val = eval_expr_vm(uv.x, uv.y);
    dst_words[py * params.dst_w + px] = bitcast<u32>(val);
}

@compute @workgroup_size(16, 16, 1)
fn expr_eval_indexed_u8(@builtin(global_invocation_id) gid: vec3<u32>) {
    let word_x = gid.x;
    let py = gid.y;
    let words_per_row = (params.dst_w + 3u) / 4u;
    if (word_x >= words_per_row || py >= params.dst_h) {
        return;
    }

    let base_x = word_x * 4u;
    var packed: u32 = 0u;
    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
        let px = base_x + k;
        if (px < params.dst_w) {
            let uv = eval_uv(px, py);
            var idx: u32 = 0u;
            if (is_finite_f32(uv.x) && is_finite_f32(uv.y)) {
                let val = eval_expr_vm(uv.x, uv.y);
                if (is_finite_f32(val)) {
                    let scaled = clamp((val - params.rescale_min) * params.inv_range_254, 0.0, 254.0);
                    idx = 1u + u32(scaled);
                }
            }
            packed = packed | ((idx & 0xffu) << (k * 8u));
        }
    }
    dst_words[py * words_per_row + word_x] = packed;
}

// ---- CIE LCH (D65) Saturation matching cogzilla-kernel::color_formula ----

fn srgb_to_lin(u: f32) -> f32 {
    if (u <= 0.04045) {
        return u * (1.0 / 12.92);
    }
    return pow((u + 0.055) * (1.0 / 1.055), 2.4);
}

fn lin_to_srgb_u32(u: f32) -> u32 {
    let c = clamp(u, 0.0, 1.0);
    var s: f32;
    if (c <= 0.0031308) {
        s = c * 12.92;
    } else {
        s = 1.055 * pow(c, 1.0 / 2.4) - 0.055;
    }
    return u32(clamp(round(s * 255.0), 0.0, 255.0));
}

fn lab_f(t: f32) -> f32 {
    if (t > 0.0088564516) {
        return pow(t, 1.0 / 3.0);
    }
    return 7.787037 * t + (16.0 / 116.0);
}

fn lab_f_inv(t: f32) -> f32 {
    if (t > 0.20689655) {
        return t * t * t;
    }
    return (t - (16.0 / 116.0)) * (1.0 / 7.787037);
}

fn apply_lch_sat(r: u32, g: u32, b: u32, sat: f32) -> vec3<u32> {
    let rl = srgb_to_lin(f32(r) * (1.0 / 255.0));
    let gl = srgb_to_lin(f32(g) * (1.0 / 255.0));
    let bl = srgb_to_lin(f32(b) * (1.0 / 255.0));

    let x = 0.4124564 * rl + 0.3575761 * gl + 0.1804375 * bl;
    let y = 0.2126729 * rl + 0.7151522 * gl + 0.0721750 * bl;
    let z = 0.0193339 * rl + 0.1191920 * gl + 0.9503041 * bl;

    let fx = lab_f(x * (1.0 / 0.95047));
    let fy = lab_f(y);
    let fz = lab_f(z * (1.0 / 1.08883));
    let a = 500.0 * (fx - fy) * sat;
    let b_lab = 200.0 * (fy - fz) * sat;

    let fx2 = fy + a * (1.0 / 500.0);
    let fz2 = fy - b_lab * (1.0 / 200.0);
    let x2 = 0.95047 * lab_f_inv(fx2);
    let y2 = lab_f_inv(fy);
    let z2 = 1.08883 * lab_f_inv(fz2);

    let rl2 = 3.2404542 * x2 - 1.5371385 * y2 - 0.4985314 * z2;
    let gl2 = -0.9692660 * x2 + 1.8760108 * y2 + 0.0415560 * z2;
    let bl2 = 0.0556434 * x2 - 0.2040259 * y2 + 1.0572252 * z2;

    return vec3<u32>(lin_to_srgb_u32(rl2), lin_to_srgb_u32(gl2), lin_to_srgb_u32(bl2));
}

fn pack_rgb_formula(vr: f32, vg: f32, vb: f32) -> u32 {
    var r = u32(clamp((vr - params.rgb_min.x) * params.rgb_inv_255.x, 0.0, 255.0));
    var g = u32(clamp((vg - params.rgb_min.y) * params.rgb_inv_255.y, 0.0, 255.0));
    var b = u32(clamp((vb - params.rgb_min.z) * params.rgb_inv_255.z, 0.0, 255.0));

    if ((params.formula_flags & 1u) != 0u) {
        r = lut_u32[r] & 0xffu;
        g = (lut_u32[g] >> 8u) & 0xffu;
        b = (lut_u32[b] >> 16u) & 0xffu;
    }
    if ((params.formula_flags & 2u) != 0u) {
        let sat_rgb = apply_lch_sat(r, g, b, params.saturation);
        r = sat_rgb.x;
        g = sat_rgb.y;
        b = sat_rgb.z;
    }

    return r | (g << 8u) | (b << 16u) | (0xffu << 24u);
}

/// Fused 3-band RGB resample -> per-channel rescale -> Gamma/Sigmoidal LUT -> LCH Saturation -> RGBA8.
@compute @workgroup_size(16, 16, 1)
fn rgb_color_formula_rgba8(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x;
    let py = gid.y;
    if (px >= params.dst_w || py >= params.dst_h) {
        return;
    }

    let uv = eval_uv(px, py);
    let vr = sample_band(0u, uv.x, uv.y);
    let vg = sample_band(1u, uv.x, uv.y);
    let vb = sample_band(2u, uv.x, uv.y);
    let out_idx = py * params.dst_w + px;

    let r_ok = is_finite_f32(vr);
    let g_ok = is_finite_f32(vg);
    let b_ok = is_finite_f32(vb);
    if (!r_ok && !g_ok && !b_ok) {
        dst_words[out_idx] = 0u;
        return;
    }

    let r_val = select(params.rgb_nodata_fill.x, vr, r_ok);
    let g_val = select(params.rgb_nodata_fill.y, vg, g_ok);
    let b_val = select(params.rgb_nodata_fill.z, vb, b_ok);
    dst_words[out_idx] = pack_rgb_formula(r_val, g_val, b_val);
}

/// Fused Dual-Pol SAR False-Color RGB ([VV_dB, VH_dB, VV_dB - VH_dB]) -> rescale ->
/// Gamma/Sigmoidal LUT -> CIE LCH Saturation -> RGBA8.
@compute @workgroup_size(16, 16, 1)
fn sar_rgb_color_formula_rgba8(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x;
    let py = gid.y;
    if (px >= params.dst_w || py >= params.dst_h) {
        return;
    }

    let uv = eval_uv(px, py);
    let p_vv = sample_band(0u, uv.x, uv.y);
    let p_vh = sample_band(1u, uv.x, uv.y);
    let out_idx = py * params.dst_w + px;

    if (!is_finite_f32(p_vv) || !is_finite_f32(p_vh) || p_vv <= 0.0 || p_vh <= 0.0) {
        dst_words[out_idx] = 0u;
        return;
    }

    let vv = 10.0 * log(p_vv) * INV_LN10;
    let vh = 10.0 * log(p_vh) * INV_LN10;
    let diff = vv - vh;

    dst_words[out_idx] = pack_rgb_formula(vv, vh, diff);
}

/// Fused 3-expression CustomExprRgb (`[prog_r, prog_g, prog_b]`) -> rescale ->
/// Gamma/Sigmoidal LUT -> CIE LCH Saturation -> RGBA8.
/// `params.num_instrs` packs `len_r | (len_g << 8u) | (len_b << 16u)`.
@compute @workgroup_size(16, 16, 1)
fn expr_rgb_color_formula_rgba8(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x;
    let py = gid.y;
    if (px >= params.dst_w || py >= params.dst_h) {
        return;
    }

    let uv = eval_uv(px, py);
    let out_idx = py * params.dst_w + px;
    if (!is_finite_f32(uv.x) || !is_finite_f32(uv.y)) {
        dst_words[out_idx] = 0u;
        return;
    }
    var bvals: array<f32, 8>;
    let nb = min(params.num_bands, 8u);
    for (var b: u32 = 0u; b < nb; b = b + 1u) {
        bvals[b] = sample_band(b, uv.x, uv.y);
    }

    let len0 = params.num_instrs & 0xffu;
    let len1 = (params.num_instrs >> 8u) & 0xffu;
    let len2 = (params.num_instrs >> 16u) & 0xffu;
    let vr = eval_expr_vm_range(&bvals, 0u, len0);
    let vg = eval_expr_vm_range(&bvals, len0, len0 + len1);
    let vb = eval_expr_vm_range(&bvals, len0 + len1, len0 + len1 + len2);

    let r_ok = is_finite_f32(vr);
    let g_ok = is_finite_f32(vg);
    let b_ok = is_finite_f32(vb);
    if (!r_ok && !g_ok && !b_ok) {
        dst_words[out_idx] = 0u;
        return;
    }

    let r_val = select(params.rgb_nodata_fill.x, vr, r_ok);
    let g_val = select(params.rgb_nodata_fill.y, vg, g_ok);
    let b_val = select(params.rgb_nodata_fill.z, vb, b_ok);
    dst_words[out_idx] = pack_rgb_formula(r_val, g_val, b_val);
}

/// Multi-band planar `(num_bands, dst_h, dst_w)` Float32 resampling for scientific `.npy` / `.tif`
/// ML tensor cubes (3D Z-dispatch over `gid.z in 0..params.num_bands`).
@compute @workgroup_size(16, 16, 1)
fn multiband_resample_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x;
    let py = gid.y;
    let band = gid.z;
    if (px >= params.dst_w || py >= params.dst_h || band >= params.num_bands) {
        return;
    }
    let uv = eval_uv(px, py);
    let val = sample_band(band, uv.x, uv.y);
    let plane_pix = params.dst_w * params.dst_h;
    dst_words[band * plane_pix + py * params.dst_w + px] = bitcast<u32>(val);
}

/// Planar `(3, dst_h, dst_w)` Float32 evaluation for 3-expression `CustomExprRgb` `.npy` / `.tif`.
@compute @workgroup_size(16, 16, 1)
fn expr_rgb_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x;
    let py = gid.y;
    if (px >= params.dst_w || py >= params.dst_h) {
        return;
    }
    let uv = eval_uv(px, py);
    let plane_pix = params.dst_w * params.dst_h;
    let pix = py * params.dst_w + px;
    if (!is_finite_f32(uv.x) || !is_finite_f32(uv.y)) {
        let nan_bits = bitcast<u32>(nan_f32());
        dst_words[pix] = nan_bits;
        dst_words[plane_pix + pix] = nan_bits;
        dst_words[2u * plane_pix + pix] = nan_bits;
        return;
    }
    var bvals: array<f32, 8>;
    let nb = min(params.num_bands, 8u);
    for (var b: u32 = 0u; b < nb; b = b + 1u) {
        bvals[b] = sample_band(b, uv.x, uv.y);
    }
    let len0 = params.num_instrs & 0xffu;
    let len1 = (params.num_instrs >> 8u) & 0xffu;
    let len2 = (params.num_instrs >> 16u) & 0xffu;
    let vr = eval_expr_vm_range(&bvals, 0u, len0);
    let vg = eval_expr_vm_range(&bvals, len0, len0 + len1);
    let vb = eval_expr_vm_range(&bvals, len0 + len1, len0 + len1 + len2);
    dst_words[pix] = bitcast<u32>(vr);
    dst_words[plane_pix + pix] = bitcast<u32>(vg);
    dst_words[2u * plane_pix + pix] = bitcast<u32>(vb);
}
