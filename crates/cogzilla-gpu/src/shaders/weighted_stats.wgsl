// Single-Submit GPU Weighted Zonal Statistics, Shared-Memory Histogram & Percentile CDF.
//
// Executes all 3 stages inside ONE bind group (bindings 0..=7) and ONE CommandEncoder
// with ZERO CPU round-trips between passes:
//   Stage 1 (`wstats_reduce_pass1`):   Grid-stride workgroup Kahan reduction (min, max,
//                                      w_total, w_sum, c_total, c_sum, area_valid, valid/masked counts, max_w).
//   Stage 1b (`wstats_finalize_pass1`): Single workgroup (1,1,1) reducing <=1024 partials -> WeightedResolved.
//   Stage 2 (`wstats_hist_pass2`):     9 KB workgroup-privatized shared-memory weighted atomic histograms
//                                      (user N-bin + 1024-bin CDF, scale=64) + exact centered weighted variance.

struct WeightedStatsParams {
    len: u32,
    win_w: u32,
    num_wg: u32,
    num_bins: u32,
    has_hist_range: u32,
    area_weighted: u32,
    h_min: f32,
    h_max: f32,
    cov_eps: f32,
    pixel_set_eps: f32,
    _pad0: u32,
    _pad1: u32,
}

struct WeightedPartial {
    min_val: f32,
    max_val: f32,
    w_total: f32,
    w_sum: f32,
    c_total: f32,
    c_sum: f32,
    area_valid: f32,
    max_w: f32,
    valid_px: u32,
    masked_px: u32,
    _pad0: u32,
    _pad1: u32,
}

struct WeightedResolved {
    min_val: f32,
    max_val: f32,
    w_total: f32,
    w_sum: f32,
    mean: f32,
    c_total: f32,
    c_sum: f32,
    area_valid: f32,
    p_inv_span: f32,
    h_min: f32,
    h_max: f32,
    h_inv_span: f32,
    inv_max_w: f32,
    valid_px: u32,
    masked_px: u32,
    _pad0: u32,
}

@group(0) @binding(0) var<uniform> params: WeightedStatsParams;
@group(0) @binding(1) var<storage, read> pixels: array<f32>;
@group(0) @binding(2) var<storage, read> covs: array<f32>;
@group(0) @binding(3) var<storage, read> row_areas: array<f32>;
@group(0) @binding(4) var<storage, read_write> partials: array<WeightedPartial>;
@group(0) @binding(5) var<storage, read_write> resolved: WeightedResolved;
@group(0) @binding(6) var<storage, read_write> hist_out: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read_write> var_partials: array<f32>;

var<workgroup> sh_min: array<f32, 256>;
var<workgroup> sh_max: array<f32, 256>;
var<workgroup> sh_wtot: array<f32, 256>;
var<workgroup> sh_wsum: array<f32, 256>;
var<workgroup> sh_ctot: array<f32, 256>;
var<workgroup> sh_csum: array<f32, 256>;
var<workgroup> sh_area: array<f32, 256>;
var<workgroup> sh_maxw: array<f32, 256>;
var<workgroup> sh_vpx: array<u32, 256>;
var<workgroup> sh_mpx: array<u32, 256>;

var<workgroup> sh_hist: array<atomic<u32>, 1024>;
var<workgroup> sh_cdf: array<atomic<u32>, 1024>;
var<workgroup> sh_var: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn wstats_reduce_pass1(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    let stride = num_wg.x * 256u;
    var idx = wgid.x * 256u + tid;

    var l_min: f32 = 3.402823466e+38;
    var l_max: f32 = -3.402823466e+38;
    var l_wtot: f32 = 0.0;
    var l_wtot_c: f32 = 0.0;
    var l_wsum: f32 = 0.0;
    var l_wsum_c: f32 = 0.0;
    var l_ctot: f32 = 0.0;
    var l_ctot_c: f32 = 0.0;
    var l_csum: f32 = 0.0;
    var l_csum_c: f32 = 0.0;
    var l_area: f32 = 0.0;
    var l_area_c: f32 = 0.0;
    var l_maxw: f32 = 0.0;
    var l_vpx: u32 = 0u;
    var l_mpx: u32 = 0u;

    while (idx < params.len) {
        let c = covs[idx];
        if (c > params.cov_eps) {
            let v = pixels[idx];
            let in_set = c > params.pixel_set_eps;
            if (!is_finite_f32(v)) {
                if (in_set) {
                    l_mpx = l_mpx + 1u;
                }
            } else {
                let py = idx / params.win_w;
                let a_px = row_areas[py];
                let cell_area = c * a_px;
                let w = select(c, cell_area, params.area_weighted != 0u);
                if (in_set) {
                    l_vpx = l_vpx + 1u;
                    l_min = min(l_min, v);
                    l_max = max(l_max, v);
                }
                l_maxw = max(l_maxw, w);

                // Kahan accumulations
                var y = w - l_wtot_c;
                var t = l_wtot + y;
                l_wtot_c = (t - l_wtot) - y;
                l_wtot = t;

                y = w * v - l_wsum_c;
                t = l_wsum + y;
                l_wsum_c = (t - l_wsum) - y;
                l_wsum = t;

                y = c - l_ctot_c;
                t = l_ctot + y;
                l_ctot_c = (t - l_ctot) - y;
                l_ctot = t;

                y = c * v - l_csum_c;
                t = l_csum + y;
                l_csum_c = (t - l_csum) - y;
                l_csum = t;

                y = cell_area - l_area_c;
                t = l_area + y;
                l_area_c = (t - l_area) - y;
                l_area = t;
            }
        }
        idx = idx + stride;
    }

    sh_min[tid] = l_min;
    sh_max[tid] = l_max;
    sh_wtot[tid] = l_wtot;
    sh_wsum[tid] = l_wsum;
    sh_ctot[tid] = l_ctot;
    sh_csum[tid] = l_csum;
    sh_area[tid] = l_area;
    sh_maxw[tid] = l_maxw;
    sh_vpx[tid] = l_vpx;
    sh_mpx[tid] = l_mpx;
    workgroupBarrier();

    var s: u32 = 128u;
    while (s > 0u) {
        if (tid < s) {
            let v_other = sh_vpx[tid + s];
            if (v_other > 0u) {
                if (sh_vpx[tid] == 0u) {
                    sh_min[tid] = sh_min[tid + s];
                    sh_max[tid] = sh_max[tid + s];
                } else {
                    sh_min[tid] = min(sh_min[tid], sh_min[tid + s]);
                    sh_max[tid] = max(sh_max[tid], sh_max[tid + s]);
                }
                sh_vpx[tid] = sh_vpx[tid] + v_other;
            }
            sh_mpx[tid] = sh_mpx[tid] + sh_mpx[tid + s];
            sh_wtot[tid] = sh_wtot[tid] + sh_wtot[tid + s];
            sh_wsum[tid] = sh_wsum[tid] + sh_wsum[tid + s];
            sh_ctot[tid] = sh_ctot[tid] + sh_ctot[tid + s];
            sh_csum[tid] = sh_csum[tid] + sh_csum[tid + s];
            sh_area[tid] = sh_area[tid] + sh_area[tid + s];
            sh_maxw[tid] = max(sh_maxw[tid], sh_maxw[tid + s]);
        }
        workgroupBarrier();
        s = s >> 1u;
    }

    if (tid == 0u) {
        partials[wgid.x] = WeightedPartial(
            sh_min[0],
            sh_max[0],
            sh_wtot[0],
            sh_wsum[0],
            sh_ctot[0],
            sh_csum[0],
            sh_area[0],
            sh_maxw[0],
            sh_vpx[0],
            sh_mpx[0],
            0u,
            0u,
        );
    }
}

@compute @workgroup_size(256, 1, 1)
fn wstats_finalize_pass1(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let n = params.num_wg;

    var l_min: f32 = 3.402823466e+38;
    var l_max: f32 = -3.402823466e+38;
    var l_wtot: f32 = 0.0;
    var l_wsum: f32 = 0.0;
    var l_ctot: f32 = 0.0;
    var l_csum: f32 = 0.0;
    var l_area: f32 = 0.0;
    var l_maxw: f32 = 0.0;
    var l_vpx: u32 = 0u;
    var l_mpx: u32 = 0u;

    var i = tid;
    while (i < n) {
        let p = partials[i];
        if (p.valid_px > 0u) {
            if (l_vpx == 0u) {
                l_min = p.min_val;
                l_max = p.max_val;
            } else {
                l_min = min(l_min, p.min_val);
                l_max = max(l_max, p.max_val);
            }
            l_vpx = l_vpx + p.valid_px;
        }
        l_mpx = l_mpx + p.masked_px;
        l_wtot = l_wtot + p.w_total;
        l_wsum = l_wsum + p.w_sum;
        l_ctot = l_ctot + p.c_total;
        l_csum = l_csum + p.c_sum;
        l_area = l_area + p.area_valid;
        l_maxw = max(l_maxw, p.max_w);
        i = i + 256u;
    }

    sh_min[tid] = l_min;
    sh_max[tid] = l_max;
    sh_wtot[tid] = l_wtot;
    sh_wsum[tid] = l_wsum;
    sh_ctot[tid] = l_ctot;
    sh_csum[tid] = l_csum;
    sh_area[tid] = l_area;
    sh_maxw[tid] = l_maxw;
    sh_vpx[tid] = l_vpx;
    sh_mpx[tid] = l_mpx;
    workgroupBarrier();

    var s: u32 = 128u;
    while (s > 0u) {
        if (tid < s) {
            let v_other = sh_vpx[tid + s];
            if (v_other > 0u) {
                if (sh_vpx[tid] == 0u) {
                    sh_min[tid] = sh_min[tid + s];
                    sh_max[tid] = sh_max[tid + s];
                } else {
                    sh_min[tid] = min(sh_min[tid], sh_min[tid + s]);
                    sh_max[tid] = max(sh_max[tid], sh_max[tid + s]);
                }
                sh_vpx[tid] = sh_vpx[tid] + v_other;
            }
            sh_mpx[tid] = sh_mpx[tid] + sh_mpx[tid + s];
            sh_wtot[tid] = sh_wtot[tid] + sh_wtot[tid + s];
            sh_wsum[tid] = sh_wsum[tid] + sh_wsum[tid + s];
            sh_ctot[tid] = sh_ctot[tid] + sh_ctot[tid + s];
            sh_csum[tid] = sh_csum[tid] + sh_csum[tid + s];
            sh_area[tid] = sh_area[tid] + sh_area[tid + s];
            sh_maxw[tid] = max(sh_maxw[tid], sh_maxw[tid + s]);
        }
        workgroupBarrier();
        s = s >> 1u;
    }

    if (tid == 0u) {
        let vpx = sh_vpx[0];
        let wtot = sh_wtot[0];
        if (vpx == 0u || wtot <= 0.0) {
            resolved = WeightedResolved(
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0u, sh_mpx[0], 0u,
            );
        } else {
            let mn = sh_min[0];
            let mx = sh_max[0];
            let mean = sh_wsum[0] / wtot;
            let p_span = mx - mn;
            let p_inv_span = select(0.0, 1.0 / p_span, p_span > 1e-20);
            let h_min = select(mn, params.h_min, params.has_hist_range != 0u);
            let h_max = select(mx, params.h_max, params.has_hist_range != 0u);
            let h_span = h_max - h_min;
            let h_inv_span = select(0.0, 1.0 / h_span, h_span > 1e-20);
            let inv_max_w = select(0.0, 64.0 / sh_maxw[0], sh_maxw[0] > 1e-30);
            resolved = WeightedResolved(
                mn,
                mx,
                wtot,
                sh_wsum[0],
                mean,
                sh_ctot[0],
                sh_csum[0],
                sh_area[0],
                p_inv_span,
                h_min,
                h_max,
                h_inv_span,
                inv_max_w,
                vpx,
                sh_mpx[0],
                0u,
            );
        }
    }
}

@compute @workgroup_size(256, 1, 1)
fn wstats_hist_pass2(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    if (resolved.valid_px == 0u || resolved.w_total <= 0.0) {
        if (tid == 0u) {
            var_partials[wgid.x] = 0.0;
        }
        return;
    }

    let n_bins = params.num_bins;
    var b = tid;
    while (b < 1024u) {
        atomicStore(&sh_hist[b], 0u);
        atomicStore(&sh_cdf[b], 0u);
        b = b + 256u;
    }
    workgroupBarrier();

    let stride = num_wg.x * 256u;
    var idx = wgid.x * 256u + tid;
    var l_var: f32 = 0.0;
    var l_var_c: f32 = 0.0;
    let mean = resolved.mean;
    let mn = resolved.min_val;
    let p_inv = resolved.p_inv_span;
    let h_min = resolved.h_min;
    let h_max = resolved.h_max;
    let h_inv = resolved.h_inv_span;
    let inv_max_w = resolved.inv_max_w;
    let n_bins_f = f32(n_bins);
    let last_bin = n_bins - 1u;
    let h_step = (h_max - h_min) / n_bins_f;

    while (idx < params.len) {
        let c = covs[idx];
        if (c > params.cov_eps) {
            let v = pixels[idx];
            if (is_finite_f32(v)) {
                let py = idx / params.win_w;
                let w = select(c, c * row_areas[py], params.area_weighted != 0u);
                let wq = max(u32(round(w * inv_max_w)), 1u);

                // Exact centered weighted variance
                let d = v - mean;
                let y = w * d * d - l_var_c;
                let t = l_var + y;
                l_var_c = (t - l_var) - y;
                l_var = t;

                // 1024-bin weighted percentile CDF
                var cb: u32 = 0u;
                if (p_inv > 0.0) {
                    let f = clamp((v - mn) * p_inv, 0.0, 1.0);
                    cb = min(u32(f * 1024.0), 1023u);
                }
                atomicAdd(&sh_cdf[cb], wq);

                // User N-bin coverage-weighted histogram (always in coverage units, matching zonal.rs)
                if (v >= h_min && v <= h_max) {
                    var hb: u32 = 0u;
                    if (h_inv > 0.0) {
                        let hf = (v - h_min) * h_inv;
                        hb = min(u32(hf * n_bins_f), last_bin);
                        if (hb < last_bin && v >= f32(hb + 1u) * h_step + h_min) {
                            hb = hb + 1u;
                        } else if (hb > 0u && v < f32(hb) * h_step + h_min) {
                            hb = hb - 1u;
                        }
                    }
                    let cq = u32(round(clamp(c, 0.0, 1.0) * 64.0));
                    if (cq > 0u) {
                        atomicAdd(&sh_hist[hb], cq);
                    }
                }
            }
        }
        idx = idx + stride;
    }

    sh_var[tid] = l_var;
    workgroupBarrier();

    var s: u32 = 128u;
    while (s > 0u) {
        if (tid < s) {
            sh_var[tid] = sh_var[tid] + sh_var[tid + s];
        }
        workgroupBarrier();
        s = s >> 1u;
    }
    if (tid == 0u) {
        var_partials[wgid.x] = sh_var[0];
    }

    var k = tid;
    while (k < 1024u) {
        let hc = atomicLoad(&sh_hist[k]);
        if (hc > 0u) {
            atomicAdd(&hist_out[k], hc);
        }
        let cc = atomicLoad(&sh_cdf[k]);
        if (cc > 0u) {
            atomicAdd(&hist_out[1024u + k], cc);
        }
        k = k + 256u;
    }
}
