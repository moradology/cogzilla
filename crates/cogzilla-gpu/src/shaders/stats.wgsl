// Single-Submit GPU Raster Band Statistics, Shared-Memory Atomic Histogram & Percentile CDF.
//
// All 3 stages share ONE bind group (bindings 0..=5) and execute inside ONE CommandEncoder
// with ZERO CPU round-trips between passes:
//   Stage 1 (`stats_reduce_pass1`):   Grid-stride workgroup reduction (Kahan sum, min, max, count).
//   Stage 1b (`stats_finalize_pass1`): Single workgroup (1,1,1) reducing <=1024 partials -> ResolvedStats.
//   Stage 2 (`stats_hist_pass2`):     9 KB workgroup-privatized shared-memory atomic histograms
//                                     (user N-bin + 1024-bin CDF) + exact centered 2-pass variance.

struct StatsParams {
    len: u32,
    num_wg: u32,
    has_nodata: u32,
    num_bins: u32,
    has_hist_range: u32,
    nodata_val: f32,
    h_min: f32,
    h_max: f32,
}

struct PartialStats {
    min_val: f32,
    max_val: f32,
    sum: f32,
    count: u32,
}

struct ResolvedStats {
    min_val: f32,
    max_val: f32,
    sum: f32,
    mean: f32,
    p_inv_span: f32,
    h_min: f32,
    h_max: f32,
    h_inv_span: f32,
    valid_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: StatsParams;
@group(0) @binding(1) var<storage, read> pixels: array<f32>;
@group(0) @binding(2) var<storage, read_write> partials: array<PartialStats>;
@group(0) @binding(3) var<storage, read_write> resolved: ResolvedStats;
@group(0) @binding(4) var<storage, read_write> hist_out: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> var_partials: array<f32>;

var<workgroup> sh_min: array<f32, 256>;
var<workgroup> sh_max: array<f32, 256>;
var<workgroup> sh_sum: array<f32, 256>;
var<workgroup> sh_cnt: array<u32, 256>;

var<workgroup> sh_hist: array<atomic<u32>, 1024>;
var<workgroup> sh_cdf: array<atomic<u32>, 1024>;
var<workgroup> sh_var: array<f32, 256>;

fn is_valid_pixel(v: f32) -> bool {
    if (!is_finite_f32(v)) {
        return false;
    }
    if (params.has_nodata != 0u && v == params.nodata_val) {
        return false;
    }
    return true;
}

@compute @workgroup_size(256, 1, 1)
fn stats_reduce_pass1(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    let stride = num_wg.x * 256u;
    var idx = wgid.x * 256u + tid;

    var local_min: f32 = 3.402823466e+38;
    var local_max: f32 = -3.402823466e+38;
    var local_sum: f32 = 0.0;
    var local_comp: f32 = 0.0;
    var local_cnt: u32 = 0u;

    while (idx < params.len) {
        let v = pixels[idx];
        if (is_valid_pixel(v)) {
            local_min = min(local_min, v);
            local_max = max(local_max, v);
            let y = v - local_comp;
            let t = local_sum + y;
            local_comp = (t - local_sum) - y;
            local_sum = t;
            local_cnt = local_cnt + 1u;
        }
        idx = idx + stride;
    }

    sh_min[tid] = local_min;
    sh_max[tid] = local_max;
    sh_sum[tid] = local_sum;
    sh_cnt[tid] = local_cnt;
    workgroupBarrier();

    var s: u32 = 128u;
    while (s > 0u) {
        if (tid < s) {
            let c_other = sh_cnt[tid + s];
            if (c_other > 0u) {
                if (sh_cnt[tid] == 0u) {
                    sh_min[tid] = sh_min[tid + s];
                    sh_max[tid] = sh_max[tid + s];
                } else {
                    sh_min[tid] = min(sh_min[tid], sh_min[tid + s]);
                    sh_max[tid] = max(sh_max[tid], sh_max[tid + s]);
                }
                sh_sum[tid] = sh_sum[tid] + sh_sum[tid + s];
                sh_cnt[tid] = sh_cnt[tid] + c_other;
            }
        }
        workgroupBarrier();
        s = s >> 1u;
    }

    if (tid == 0u) {
        partials[wgid.x] = PartialStats(sh_min[0], sh_max[0], sh_sum[0], sh_cnt[0]);
    }
}

@compute @workgroup_size(256, 1, 1)
fn stats_finalize_pass1(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    var lmin: f32 = 3.402823466e+38;
    var lmax: f32 = -3.402823466e+38;
    var lsum: f32 = 0.0;
    var lcnt: u32 = 0u;

    var i = tid;
    while (i < params.num_wg) {
        let p = partials[i];
        if (p.count > 0u) {
            if (lcnt == 0u) {
                lmin = p.min_val;
                lmax = p.max_val;
            } else {
                lmin = min(lmin, p.min_val);
                lmax = max(lmax, p.max_val);
            }
            lsum = lsum + p.sum;
            lcnt = lcnt + p.count;
        }
        i = i + 256u;
    }

    sh_min[tid] = lmin;
    sh_max[tid] = lmax;
    sh_sum[tid] = lsum;
    sh_cnt[tid] = lcnt;
    workgroupBarrier();

    var s: u32 = 128u;
    while (s > 0u) {
        if (tid < s) {
            let c_other = sh_cnt[tid + s];
            if (c_other > 0u) {
                if (sh_cnt[tid] == 0u) {
                    sh_min[tid] = sh_min[tid + s];
                    sh_max[tid] = sh_max[tid + s];
                } else {
                    sh_min[tid] = min(sh_min[tid], sh_min[tid + s]);
                    sh_max[tid] = max(sh_max[tid], sh_max[tid + s]);
                }
                sh_sum[tid] = sh_sum[tid] + sh_sum[tid + s];
                sh_cnt[tid] = sh_cnt[tid] + c_other;
            }
        }
        workgroupBarrier();
        s = s >> 1u;
    }

    if (tid == 0u) {
        let vc = sh_cnt[0];
        if (vc == 0u) {
            resolved = ResolvedStats(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0u, 0u, 0u, 0u);
        } else {
            let mn = sh_min[0];
            let mx = sh_max[0];
            let sm = sh_sum[0];
            let avg = sm / f32(vc);
            let p_span = max(mx - mn, 1e-12);
            let hm0 = select(mn, params.h_min, params.has_hist_range != 0u);
            let hm1 = select(mx, params.h_max, params.has_hist_range != 0u);
            let h_span = max(hm1 - hm0, 1e-12);
            resolved = ResolvedStats(
                mn,
                mx,
                sm,
                avg,
                1.0 / p_span,
                hm0,
                hm1,
                1.0 / h_span,
                vc,
                0u,
                0u,
                0u,
            );
        }
    }
}

@compute @workgroup_size(256, 1, 1)
fn stats_hist_pass2(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(num_workgroups) num_wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    if (resolved.valid_count == 0u) {
        if (tid == 0u) {
            var_partials[wgid.x] = 0.0;
        }
        return;
    }

    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
        let b = tid + k * 256u;
        atomicStore(&sh_hist[b], 0u);
        atomicStore(&sh_cdf[b], 0u);
    }
    workgroupBarrier();

    let stride = num_wg.x * 256u;
    var idx = wgid.x * 256u + tid;
    var local_var: f32 = 0.0;
    let n_bins_f = f32(params.num_bins);
    let max_bin = params.num_bins - 1u;
    let avg = resolved.mean;
    let p_min = resolved.min_val;
    let p_inv = resolved.p_inv_span;
    let h_min = resolved.h_min;
    let h_max = resolved.h_max;
    let h_inv = resolved.h_inv_span;
    let h_step = (h_max - h_min) / n_bins_f;

    while (idx < params.len) {
        let v = pixels[idx];
        if (is_valid_pixel(v)) {
            let d = v - avg;
            local_var = local_var + d * d;

            let cdf_f = floor((v - p_min) * p_inv * 1024.0);
            let cdf_bin = u32(clamp(cdf_f, 0.0, 1023.0));
            atomicAdd(&sh_cdf[cdf_bin], 1u);

            if (params.has_hist_range == 0u || (v >= h_min && v <= h_max)) {
                let hf = floor((v - h_min) * h_inv * n_bins_f);
                var hbin = u32(clamp(hf, 0.0, f32(max_bin)));
                // Match numpy.histogram (`edges[i] <= v < edges[i + 1]`, last bin closed):
                // correct 1-ULP rounding of `1.0 / (h_max - h_min)` at exact inner bin edges (#626).
                if (hbin < max_bin && v >= f32(hbin + 1u) * h_step + h_min) {
                    hbin = hbin + 1u;
                } else if (hbin > 0u && v < f32(hbin) * h_step + h_min) {
                    hbin = hbin - 1u;
                }
                atomicAdd(&sh_hist[hbin], 1u);
            }
        }
        idx = idx + stride;
    }

    sh_var[tid] = local_var;
    workgroupBarrier();

    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
        let b = tid + k * 256u;
        if (b < params.num_bins) {
            let c = atomicLoad(&sh_hist[b]);
            if (c > 0u) {
                atomicAdd(&hist_out[b], c);
            }
        }
        let cc = atomicLoad(&sh_cdf[b]);
        if (cc > 0u) {
            atomicAdd(&hist_out[1024u + b], cc);
        }
    }

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
}
