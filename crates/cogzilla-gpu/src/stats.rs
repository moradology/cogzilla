//! Single-Submit GPU Raster Band Statistics, Shared-Memory Atomic Histogram & Percentile CDF.
//!
//! Executes all 3 WGSL stages inside **ONE `wgpu::CommandEncoder` → ONE `queue.submit()` →
//! ONE 12 KB `map_async()`** with zero CPU round-trips between passes:
//! 1. **Stage 1 (`stats_reduce_pass1`)**: Grid-stride workgroup reduction (Kahan sum, min, max, count).
//! 2. **Stage 1b (`stats_finalize_pass1`)**: Single-workgroup GPU reduction of `<= 1024` partials →
//!    `ResolvedStats` (`min`, `max`, `mean`, `valid_count`, histogram spans).
//! 3. **Stage 2 (`stats_hist_pass2`)**: 9 KB workgroup-privatized shared-memory atomic histograms
//!    (user `N`-bin + 1024-bin percentile CDF) + exact centered 2-pass variance (`std`).

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};

use crate::context::{GpuContext, GpuError};
use crate::resample::GpuSlab;

pub(crate) const MAX_WG: u32 = 1024;
pub(crate) const CDF_BINS: usize = 1024;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct StatsParams {
    pub(crate) len: u32,
    pub(crate) num_wg: u32,
    pub(crate) has_nodata: u32,
    pub(crate) num_bins: u32,
    pub(crate) has_hist_range: u32,
    pub(crate) nodata_val: f32,
    pub(crate) h_min: f32,
    pub(crate) h_max: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct PartialStats {
    pub(crate) min_val: f32,
    pub(crate) max_val: f32,
    pub(crate) sum: f32,
    pub(crate) count: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub(crate) struct ResolvedStats {
    pub(crate) min_val: f32,
    pub(crate) max_val: f32,
    pub(crate) sum: f32,
    pub(crate) mean: f32,
    pub(crate) p_inv_span: f32,
    pub(crate) h_min: f32,
    pub(crate) h_max: f32,
    pub(crate) h_inv_span: f32,
    pub(crate) valid_count: u32,
    pub(crate) _pad0: u32,
    pub(crate) _pad1: u32,
    pub(crate) _pad2: u32,
}

/// Complete TiTiler-compatible band statistics computed on the GPU (defined in `cogzilla-raster`).
pub use cogzilla_raster::GpuBandStats;

/// Single-submit GPU raster statistics & histogram reducer.
pub struct GpuStatsReducer {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    bgl: wgpu::BindGroupLayout,
    pipe_pass1: wgpu::ComputePipeline,
    pipe_finalize1: wgpu::ComputePipeline,
    pipe_pass2: wgpu::ComputePipeline,
    params_buf: wgpu::Buffer,
    src_buf: wgpu::Buffer,
    src_cap_elems: usize,
    partials_buf: wgpu::Buffer,
    resolved_buf: wgpu::Buffer,
    hist_buf: wgpu::Buffer,
    var_buf: wgpu::Buffer,
    staging_buf: wgpu::Buffer,
}

impl GpuStatsReducer {
    pub fn new(ctx: &GpuContext) -> Self {
        let device = Arc::clone(&ctx.device);
        let queue = Arc::clone(&ctx.queue);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("cogzilla-stats-wgsl"),
            source: wgpu::ShaderSource::Wgsl(crate::shaders::STATS_WGSL.into()),
        });

        use crate::wgpu_util::{bgl_storage_ro, bgl_storage_rw, bgl_uniform};
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("cogzilla-stats-bgl"),
            entries: &[
                bgl_uniform(0),
                bgl_storage_ro(1),
                bgl_storage_rw(2),
                bgl_storage_rw(3),
                bgl_storage_rw(4),
                bgl_storage_rw(5),
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("cogzilla-stats-pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let mk_pipe = |label: &str, entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pl),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let pipe_pass1 = mk_pipe("cogzilla-stats-pass1", "stats_reduce_pass1");
        let pipe_finalize1 = mk_pipe("cogzilla-stats-finalize1", "stats_finalize_pass1");
        let pipe_pass2 = mk_pipe("cogzilla-stats-pass2", "stats_hist_pass2");

        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cogzilla-stats-params"),
            size: std::mem::size_of::<StatsParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let src_cap_elems = 1024 * 1024;
        let src_buf = Self::alloc_src(&device, src_cap_elems);

        let partials_bytes = (MAX_WG as usize * std::mem::size_of::<PartialStats>()) as u64;
        let partials_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cogzilla-stats-partials"),
            size: partials_bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let resolved_bytes = std::mem::size_of::<ResolvedStats>() as u64;
        let resolved_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cogzilla-stats-resolved"),
            size: resolved_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let hist_bytes = (2 * CDF_BINS * std::mem::size_of::<u32>()) as u64;
        let hist_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cogzilla-stats-hist"),
            size: hist_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let var_bytes = (MAX_WG as usize * std::mem::size_of::<f32>()) as u64;
        let var_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cogzilla-stats-var"),
            size: var_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Single packed staging buffer: [ResolvedStats (48 B) | hist_out (8192 B) | var_partials (<=4096 B)]
        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cogzilla-stats-staging"),
            size: resolved_bytes + hist_bytes + var_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            device,
            queue,
            bgl,
            pipe_pass1,
            pipe_finalize1,
            pipe_pass2,
            params_buf,
            src_buf,
            src_cap_elems,
            partials_buf,
            resolved_buf,
            hist_buf,
            var_buf,
            staging_buf,
        }
    }

    fn alloc_src(device: &wgpu::Device, elems: usize) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cogzilla-stats-src"),
            size: (elems.max(1) * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Compute full band statistics, histogram, and percentiles from a host `f32` slice
    /// in a single GPU submission.
    pub fn reduce_f32(
        &mut self,
        pixels: &[f32],
        nodata: Option<f32>,
        num_bins: usize,
        hist_range: Option<(f64, f64)>,
        percentiles: &[f64],
    ) -> Result<GpuBandStats, GpuError> {
        if pixels.is_empty() {
            return Ok(empty_stats(0, percentiles));
        }
        let needed_bytes = (pixels.len() as u64).saturating_mul(4);
        let max_binding = self.device.limits().max_storage_buffer_binding_size as u64;
        if needed_bytes > max_binding {
            return Err(GpuError::InvalidDimensions(format!(
                "stats input ({needed_bytes} bytes) exceeds adapter max_storage_buffer_binding_size ({max_binding} bytes)"
            )));
        }
        if pixels.len() > self.src_cap_elems {
            let max_elems = (max_binding / 4) as usize;
            self.src_cap_elems = pixels.len().next_power_of_two().min(max_elems);
            self.src_buf = Self::alloc_src(&self.device, self.src_cap_elems);
        }
        self.queue
            .write_buffer(&self.src_buf, 0, bytemuck::cast_slice(pixels));
        let src_buf = self.src_buf.clone();
        self.reduce_buffer(&src_buf, pixels.len(), nodata, num_bins, hist_range, percentiles)
    }

    /// Compute full band statistics directly on a GPU-resident `GpuSlab` (zero CPU→GPU upload!).
    pub fn reduce_slab(
        &mut self,
        slab: &GpuSlab,
        nodata: Option<f32>,
        num_bins: usize,
        hist_range: Option<(f64, f64)>,
        percentiles: &[f64],
    ) -> Result<GpuBandStats, GpuError> {
        let len = (slab.stride as usize).saturating_mul(slab.height as usize);
        if len == 0 {
            return Ok(empty_stats(0, percentiles));
        }
        self.reduce_buffer(
            &slab.buffer,
            len,
            nodata,
            num_bins,
            hist_range,
            percentiles,
        )
    }

    /// Compute full band statistics directly on an arbitrary GPU `STORAGE` buffer of `f32` pixels
    /// in **one single GPU submission** (zero CPU→GPU upload, 12 KB total readback).
    pub fn reduce_buffer(
        &mut self,
        src_buf: &wgpu::Buffer,
        len: usize,
        nodata: Option<f32>,
        num_bins: usize,
        hist_range: Option<(f64, f64)>,
        percentiles: &[f64],
    ) -> Result<GpuBandStats, GpuError> {
        let len_u32 = u32::try_from(len)
            .map_err(|_| GpuError::InvalidDimensions("stats len exceeds u32::MAX".into()))?;
        let num_wg = len_u32.div_ceil(256 * 16).clamp(1, MAX_WG);
        let n_bins = num_bins.clamp(1, CDF_BINS) as u32;
        let (hr_min, hr_max) = hist_range.unwrap_or((0.0, 0.0));

        let params = StatsParams {
            len: len_u32,
            num_wg,
            has_nodata: u32::from(nodata.is_some()),
            num_bins: n_bins,
            has_hist_range: u32::from(hist_range.is_some()),
            nodata_val: nodata.unwrap_or(0.0),
            h_min: hr_min as f32,
            h_max: hr_max as f32,
        };
        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("cogzilla-stats-bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: src_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.partials_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.resolved_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.hist_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.var_buf.as_entire_binding(),
                },
            ],
        });

        let resolved_bytes = std::mem::size_of::<ResolvedStats>() as u64;
        let hist_bytes = (2 * CDF_BINS * std::mem::size_of::<u32>()) as u64;
        let var_bytes = (num_wg as usize * std::mem::size_of::<f32>()) as u64;
        let total_readback = resolved_bytes + hist_bytes + var_bytes;

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("cogzilla-stats-enc"),
            });
        encoder.clear_buffer(&self.hist_buf, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cogzilla-stats-pass1"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipe_pass1);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(num_wg, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cogzilla-stats-finalize1"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipe_finalize1);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("cogzilla-stats-pass2"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipe_pass2);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(num_wg, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.resolved_buf, 0, &self.staging_buf, 0, resolved_bytes);
        encoder.copy_buffer_to_buffer(
            &self.hist_buf,
            0,
            &self.staging_buf,
            resolved_bytes,
            hist_bytes,
        );
        encoder.copy_buffer_to_buffer(
            &self.var_buf,
            0,
            &self.staging_buf,
            resolved_bytes + hist_bytes,
            var_bytes,
        );
        self.queue.submit(Some(encoder.finish()));

        crate::wgpu_util::readback_staging_sync(
            &self.device,
            &self.staging_buf,
            total_readback,
            |mapped| decode_stats_readback(mapped, len, n_bins as usize, hist_range, percentiles),
        )
    }
}

/// Decode the packed 12 KB readback buffer (`[ResolvedStats | hist_out | var_partials]`)
/// emitted by `GpuStatsReducer` or `CoalescedJob::WithStats` into a `GpuBandStats`.
pub fn decode_stats_readback(
    bytes: &[u8],
    total_pixels: usize,
    num_bins: usize,
    hist_range: Option<(f64, f64)>,
    percentiles: &[f64],
) -> GpuBandStats {
    let rb = std::mem::size_of::<ResolvedStats>();
    let hb = 2 * CDF_BINS * std::mem::size_of::<u32>();
    if bytes.len() < rb + hb || !(bytes.len() - rb - hb).is_multiple_of(4) {
        return empty_stats(total_pixels, percentiles);
    }
    let n_bins = num_bins.clamp(1, CDF_BINS);
    let res: ResolvedStats = bytemuck::pod_read_unaligned(&bytes[..rb]);
    let valid_px = res.valid_count as usize;
    if valid_px == 0 {
        return empty_stats(total_pixels, percentiles);
    }
    let h_owned: Vec<u32>;
    let h_all: &[u32] = match bytemuck::try_cast_slice(&bytes[rb..rb + hb]) {
        Ok(s) => s,
        Err(_) => {
            h_owned = bytes[rb..rb + hb]
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            &h_owned
        }
    };
    let v_owned: Vec<f32>;
    let v_all: &[f32] = match bytemuck::try_cast_slice(&bytes[rb + hb..]) {
        Ok(s) => s,
        Err(_) => {
            v_owned = bytes[rb + hb..]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            &v_owned
        }
    };
    let hist_counts: Vec<u64> = h_all[..n_bins].iter().map(|&c| c as u64).collect();
    let cdf_counts: Vec<u64> = h_all[CDF_BINS..2 * CDF_BINS]
        .iter()
        .map(|&c| c as u64)
        .collect();
    let var_sum: f64 = v_all.iter().map(|&x| x as f64).sum();

    let min_v = res.min_val as f64;
    let max_v = res.max_val as f64;
    let sum = res.sum as f64;
    let mean = sum / valid_px as f64;
    let std_dev = (var_sum / valid_px as f64).sqrt();
    let median = Some(cdf_percentile(&cdf_counts, valid_px as u64, min_v, max_v, 50.0));
    let p_vals = percentiles
        .iter()
        .map(|&p| (p, Some(cdf_percentile(&cdf_counts, valid_px as u64, min_v, max_v, p))))
        .collect();
    let (h_min, h_max) = hist_range.unwrap_or((min_v, max_v));
    let h_span = (h_max - h_min).max(1e-12);
    let hist_edges: Vec<f64> = (0..=n_bins)
        .map(|i| h_min + h_span * (i as f64 / n_bins as f64))
        .collect();
    let masked_pixels = total_pixels.saturating_sub(valid_px);
    let valid_percent = if total_pixels > 0 {
        ((valid_px as f64 / total_pixels as f64) * 100.0 * 100.0).round() / 100.0
    } else {
        0.0
    };

    let mut maj_cnt = 0u64;
    let mut maj_bin = 0usize;
    let mut min_cnt = u64::MAX;
    let mut min_bin = 0usize;
    let mut unique_bins = 0usize;
    for (i, &c) in cdf_counts.iter().enumerate() {
        if c > 0 {
            unique_bins += 1;
            if c > maj_cnt {
                maj_cnt = c;
                maj_bin = i;
            }
            if c < min_cnt {
                min_cnt = c;
                min_bin = i;
            }
        }
    }
    let bin_w = (max_v - min_v) / (CDF_BINS as f64);
    let majority = Some(min_v + (maj_bin as f64 + 0.5) * bin_w);
    let minority = Some(min_v + (min_bin as f64 + 0.5) * bin_w);

    GpuBandStats {
        total_pixels,
        valid_pixels: valid_px,
        masked_pixels,
        valid_percent,
        min: Some(min_v),
        max: Some(max_v),
        mean: Some(mean),
        sum,
        std: Some(std_dev),
        median,
        majority,
        minority,
        unique_bins,
        percentiles: p_vals,
        hist_counts,
        hist_edges,
    }
}

fn cdf_percentile(cdf: &[u64], total: u64, min_v: f64, max_v: f64, p: f64) -> f64 {
    if total == 0 || max_v <= min_v {
        return min_v;
    }
    let rank = (p.clamp(0.0, 100.0) / 100.0) * (total - 1) as f64;
    let mut acc = 0u64;
    let span = max_v - min_v;
    let bin_w = span / (cdf.len() as f64);
    for (i, &c) in cdf.iter().enumerate() {
        if c == 0 {
            continue;
        }
        let next = acc + c;
        if (next as f64) > rank {
            let frac = ((rank - acc as f64) / (c as f64)).clamp(0.0, 1.0);
            return min_v + (i as f64 + frac) * bin_w;
        }
        acc = next;
    }
    max_v
}

fn empty_stats(total: usize, percentiles: &[f64]) -> GpuBandStats {
    GpuBandStats {
        total_pixels: total,
        valid_pixels: 0,
        masked_pixels: total,
        valid_percent: 0.0,
        min: None,
        max: None,
        mean: None,
        sum: 0.0,
        std: None,
        median: None,
        majority: None,
        minority: None,
        unique_bins: 0,
        percentiles: percentiles.iter().map(|&p| (p, None)).collect(),
        hist_counts: Vec::new(),
        hist_edges: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::GpuOptions;

    #[test]
    fn gpu_stats_reducer_matches_cpu_exact_and_percentiles() {
        let Ok(ctx) = GpuContext::new(GpuOptions::default()) else {
            eprintln!("skipping gpu_stats test (no GPU adapter)");
            return;
        };
        let mut reducer = GpuStatsReducer::new(&ctx);
        let n = 512 * 512usize;
        let mut data = Vec::with_capacity(n);
        for i in 0..n {
            if i % 17 == 0 {
                data.push(f32::NAN);
            } else if i % 29 == 0 {
                data.push(-9999.0);
            } else {
                let x = (i as f32) / (n as f32);
                data.push(100.0 + 900.0 * x + 50.0 * (x * 37.0).sin());
            }
        }
        let stats = reducer
            .reduce_f32(&data, Some(-9999.0), 10, None, &[2.0, 50.0, 98.0])
            .expect("gpu stats");

        let mut valid: Vec<f32> = data
            .iter()
            .copied()
            .filter(|v| v.is_finite() && *v != -9999.0)
            .collect();
        valid.sort_unstable_by(|a, b| a.total_cmp(b));
        let cpu_min = valid[0] as f64;
        let cpu_max = *valid.last().unwrap() as f64;
        let cpu_mean = valid.iter().map(|&v| v as f64).sum::<f64>() / valid.len() as f64;
        let cpu_std = (valid
            .iter()
            .map(|&v| {
                let d = v as f64 - cpu_mean;
                d * d
            })
            .sum::<f64>()
            / valid.len() as f64)
            .sqrt();
        let cpu_p2 = valid[((valid.len() - 1) as f64 * 0.02) as usize] as f64;
        let cpu_p50 = valid[((valid.len() - 1) as f64 * 0.50) as usize] as f64;
        let cpu_p98 = valid[((valid.len() - 1) as f64 * 0.98) as usize] as f64;

        assert_eq!(stats.valid_pixels, valid.len());
        assert_eq!(stats.masked_pixels, n - valid.len());
        assert!((stats.min.unwrap() - cpu_min).abs() < 1e-4);
        assert!((stats.max.unwrap() - cpu_max).abs() < 1e-4);
        assert!((stats.mean.unwrap() - cpu_mean).abs() < 1e-1);
        assert!((stats.std.unwrap() - cpu_std).abs() < 1e-1);
        assert_eq!(stats.hist_counts.iter().sum::<u64>(), valid.len() as u64);
        let span = cpu_max - cpu_min;
        assert!((stats.percentiles[0].1.unwrap() - cpu_p2).abs() < span * 0.002);
        assert!((stats.percentiles[1].1.unwrap() - cpu_p50).abs() < span * 0.002);
        assert!((stats.percentiles[2].1.unwrap() - cpu_p98).abs() < span * 0.002);
    }

    #[test]
    fn gpu_histogram_inner_bin_edge_ties_match_numpy() {
        let Ok(ctx) = GpuContext::new(GpuOptions::default()) else {
            eprintln!("skipping gpu_stats bin-edge test (no GPU adapter)");
            return;
        };
        let mut reducer = GpuStatsReducer::new(&ctx);
        for &(lo, hi, n_bins) in &[
            (0.0f32, 3.0f32, 3usize),
            (0.0, 7.0, 7),
            (0.0, 10.0, 10),
            (0.1, 0.8, 7),
            (-5.0, 8.0, 13),
        ] {
            let step = (hi - lo) / (n_bins as f32);
            let mut edges: Vec<f32> = (0..=n_bins).map(|k| (k as f32) * step + lo).collect();
            edges[n_bins] = hi;
            // Place 1 sample on every edge (0..=n_bins), plus 1 extra on each inner edge.
            // Under numpy.histogram (`edges[i] <= v < edges[i+1]`, last bin closed):
            // bin 0 gets edge 0 (count 1); inner bins 1..n_bins-1 get 2 samples each;
            // last bin gets edge n_bins-1 (2 samples) + edge n_bins (1 sample) = 3 samples.
            let mut data = Vec::new();
            data.push(edges[0]);
            for &e in &edges[1..n_bins] {
                data.push(e);
                data.push(e);
            }
            data.push(edges[n_bins]);

            let stats = reducer
                .reduce_f32(&data, None, n_bins, None, &[50.0])
                .expect("gpu stats");
            let mut expected = vec![0u64; n_bins];
            for &v in &data {
                let bin = edges[1..n_bins].partition_point(|&e| e <= v);
                expected[bin] += 1;
            }
            assert_eq!(
                stats.hist_counts, expected,
                "GPU histogram bin-edge mismatch for [{lo}, {hi}] with {n_bins} bins"
            );
        }
    }
}
