fn sample_checked(x: i32, y: i32) -> f32 {
    if (x < 0 || y < 0 || x >= i32(params.src_w) || y >= i32(params.src_h)) {
        return nan_f32();
    }
    return src_slab[u32(y) * params.src_stride + u32(x)];
}

fn sample_bilinear_raw(u: f32, v: f32) -> f32 {
    let x0_f = floor(u);
    let y0_f = floor(v);
    let x0 = i32(x0_f);
    let y0 = i32(y0_f);
    let fx = u - x0_f;
    let fy = v - y0_f;

    let v00 = sample_checked(x0, y0);
    let v10 = sample_checked(x0 + 1, y0);
    let v01 = sample_checked(x0, y0 + 1);
    let v11 = sample_checked(x0 + 1, y0 + 1);

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

fn sample_bilinear(u: f32, v: f32) -> f32 {
    if (!in_sample_extent(u, v)) {
        return nan_f32();
    }
    return sample_bilinear_raw(u, v);
}

fn sample_cubic(u: f32, v: f32) -> f32 {
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
            let s = sample_checked(x0 + i - 1, ry);
            if (is_nan_f32(s)) {
                return sample_bilinear_raw(u, v);
            }
            row_acc += s * wx[i];
        }
        acc += row_acc * wy[j];
    }
    return acc;
}

fn sample_lanczos3(u: f32, v: f32) -> f32 {
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
            let s = sample_checked(x0 + i - 2, ry);
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
    return sample_bilinear_raw(u, v);
}

fn sample_nearest(u: f32, v: f32) -> f32 {
    if (!in_sample_extent(u, v)) {
        return nan_f32();
    }
    return sample_checked(i32(floor(u + 0.5)), i32(floor(v + 0.5)));
}

fn sample_mode_1band(mode: u32, u: f32, v: f32) -> f32 {
    if (mode == 0u) {
        return sample_bilinear(u, v);
    } else if (mode == 1u) {
        return sample_cubic(u, v);
    } else if (mode == 3u) {
        return sample_nearest(u, v);
    } else {
        return sample_lanczos3(u, v);
    }
}
