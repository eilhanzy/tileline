//! Runtime tuning heuristics for GMS.
//!
//! These hints are intentionally conservative and portable. They centralize platform and
//! memory-topology-specific knobs (especially unified-memory behavior) so examples and future
//! runtime code do not each reinvent their own heuristics.

use wgpu::DeviceType;

/// Reusable runtime tuning hints derived from adapter characteristics.
#[derive(Debug, Clone, Copy)]
pub struct GmsRuntimeTuningProfile {
    /// True when the adapter behaves like a unified-memory/integrated GPU path.
    pub prefer_unified_memory_tuning: bool,
    /// True for Apple Silicon-style adapters (best-effort name/device-type heuristic).
    pub is_apple_silicon_like: bool,
    /// Suggested `SurfaceConfiguration::desired_maximum_frame_latency`.
    pub recommended_surface_frame_latency: u32,
    /// Suggested ring length for reusable offscreen throughput targets.
    pub throughput_offscreen_target_ring_len: usize,
    /// Suggested throughput burst for integrated/unified GPUs in synthetic benchmarks.
    pub integrated_throughput_burst_work_units: u32,
    /// Suggested capacity for benchmark timing vectors to avoid growth spikes.
    pub benchmark_timing_capacity: usize,
    /// Suggested UI title refresh interval to reduce host/compositor noise in benchmarks.
    pub benchmark_title_update_interval_ms: u64,
    /// Throughput benchmark score weighting for work-time stability vs present-time stability.
    ///
    /// `0.0` means score stability uses only present/compositor intervals.
    /// `1.0` means score stability uses only per-work-unit render durations.
    pub throughput_work_stability_blend: f64,
    /// Number of presents used to smoothly ramp into full throughput burst mode.
    pub throughput_startup_ramp_frames: u32,
    /// Number of offscreen prewarm submissions to issue during startup (before first visible frame).
    pub throughput_startup_prewarm_submits: u32,
}

impl GmsRuntimeTuningProfile {
    /// Build tuning hints from `wgpu::AdapterInfo`.
    ///
    /// This is name-heuristic based because portable `wgpu` does not expose direct UMA metadata.
    pub fn from_adapter_info(adapter_info: &wgpu::AdapterInfo) -> Self {
        let name = adapter_info.name.to_ascii_lowercase();
        let is_apple_silicon_like = name.contains("apple")
            || name.contains("m1")
            || name.contains("m2")
            || name.contains("m3")
            || name.contains("m4");
        let prefer_unified_memory_tuning =
            matches!(adapter_info.device_type, DeviceType::IntegratedGpu) || is_apple_silicon_like;

        if is_apple_silicon_like {
            // Apple Silicon tends to expose excellent raw throughput but visible jitter can
            // increase when synthetic benchmarks aggressively hammer a single reusable target.
            // We bias toward more reuse buffers and lower burst intensity to stabilize p95/p99.
            return Self {
                prefer_unified_memory_tuning: true,
                is_apple_silicon_like: true,
                recommended_surface_frame_latency: 4,
                throughput_offscreen_target_ring_len: 6,
                integrated_throughput_burst_work_units: 3,
                throughput_work_stability_blend: 0.85,
                throughput_startup_ramp_frames: 90,
                throughput_startup_prewarm_submits: 2,
                benchmark_timing_capacity: 131_072,
                benchmark_title_update_interval_ms: 500,
            };
        }

        if prefer_unified_memory_tuning {
            return Self {
                prefer_unified_memory_tuning: true,
                is_apple_silicon_like: false,
                recommended_surface_frame_latency: 3,
                throughput_offscreen_target_ring_len: 4,
                integrated_throughput_burst_work_units: 6,
                throughput_work_stability_blend: 0.65,
                throughput_startup_ramp_frames: 45,
                throughput_startup_prewarm_submits: 1,
                benchmark_timing_capacity: 98_304,
                benchmark_title_update_interval_ms: 500,
            };
        }

        Self {
            prefer_unified_memory_tuning: false,
            is_apple_silicon_like: false,
            recommended_surface_frame_latency: 2,
            throughput_offscreen_target_ring_len: 2,
            integrated_throughput_burst_work_units: 8,
            throughput_work_stability_blend: 0.0,
            throughput_startup_ramp_frames: 0,
            throughput_startup_prewarm_submits: 0,
            benchmark_timing_capacity: 131_072,
            benchmark_title_update_interval_ms: 500,
        }
    }
}
