//! Runtime policies for MGS clients.
//!
//! This module centralizes frame pacing, present-mode fallback, UMA memory
//! strategy, and adaptive throughput burst control so applications can share
//! the same behavior without duplicating benchmark-local code.

use std::time::{Duration, Instant};

use wgpu::{Limits, PresentMode, StoreOp};

use crate::hardware::{MobileGpuFamily, MobileGpuProfile, TbdrArchitecture};
use crate::zram::MpsZramConfig;

/// Runtime throughput mode preference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeMode {
    Auto,
    Stable,
    MaxThroughput,
}

/// Runtime present synchronization preference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VsyncMode {
    Auto,
    On,
    Off,
}

/// Effective runtime pacing mode after applying policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimePacingMode {
    Stable,
    MaxThroughput,
}

/// Device-limit clamp report for portability-sensitive limits.
#[derive(Debug, Clone, Copy, Default)]
pub struct DeviceLimitClampReport {
    pub max_texture_dimension_1d: bool,
    pub max_texture_dimension_2d: bool,
    pub max_texture_dimension_3d: bool,
    pub max_texture_array_layers: bool,
}

impl DeviceLimitClampReport {
    #[inline]
    pub fn any_clamped(self) -> bool {
        self.max_texture_dimension_1d
            || self.max_texture_dimension_2d
            || self.max_texture_dimension_3d
            || self.max_texture_array_layers
    }
}

/// Clamp requested `wgpu::Limits` to supported adapter limits.
pub fn clamp_required_limits_to_supported(
    mut requested: Limits,
    supported: &Limits,
) -> (Limits, DeviceLimitClampReport) {
    let mut report = DeviceLimitClampReport::default();

    if requested.max_texture_dimension_1d > supported.max_texture_dimension_1d {
        requested.max_texture_dimension_1d = supported.max_texture_dimension_1d;
        report.max_texture_dimension_1d = true;
    }
    if requested.max_texture_dimension_2d > supported.max_texture_dimension_2d {
        requested.max_texture_dimension_2d = supported.max_texture_dimension_2d;
        report.max_texture_dimension_2d = true;
    }
    if requested.max_texture_dimension_3d > supported.max_texture_dimension_3d {
        requested.max_texture_dimension_3d = supported.max_texture_dimension_3d;
        report.max_texture_dimension_3d = true;
    }
    if requested.max_texture_array_layers > supported.max_texture_array_layers {
        requested.max_texture_array_layers = supported.max_texture_array_layers;
        report.max_texture_array_layers = true;
    }

    (requested, report)
}

/// Conservative default required-limits policy for MGS device creation.
pub fn safe_default_required_limits_for_adapter(
    adapter: &wgpu::Adapter,
) -> (Limits, DeviceLimitClampReport) {
    let supported = adapter.limits();
    clamp_required_limits_to_supported(Limits::default(), &supported)
}

/// Frame pacer used by runtime clients to avoid busy-poll CPU saturation.
#[derive(Debug, Clone)]
pub struct ThroughputFramePacer {
    pacing_mode: RuntimePacingMode,
    min_frame_interval: Duration,
    cpu_relief_interval: Duration,
    next_frame_at: Instant,
}

impl ThroughputFramePacer {
    /// Create a new frame pacer.
    pub fn new(
        pacing_mode: RuntimePacingMode,
        min_frame_interval: Duration,
        cpu_relief_interval: Duration,
        now: Instant,
    ) -> Self {
        Self {
            pacing_mode,
            min_frame_interval,
            cpu_relief_interval,
            next_frame_at: now,
        }
    }

    /// Returns true when render work should execute at `now`.
    pub fn ready_for_next_frame(&self, now: Instant) -> bool {
        if !matches!(self.pacing_mode, RuntimePacingMode::MaxThroughput) {
            return true;
        }
        now >= self.next_frame_at
    }

    /// Returns the next wake target for event loops in max-throughput mode.
    pub fn preferred_wait_until(&self, now: Instant) -> Option<Instant> {
        if !matches!(self.pacing_mode, RuntimePacingMode::MaxThroughput) {
            return None;
        }
        let floor = self.cpu_relief_interval.max(Duration::from_micros(500));
        Some(if self.next_frame_at > now {
            self.next_frame_at
        } else {
            now + floor
        })
    }

    /// Updates next-frame target after one frame completes.
    pub fn on_frame_complete(&mut self, frame_start: Instant, frame_end: Instant) {
        if !matches!(self.pacing_mode, RuntimePacingMode::MaxThroughput) {
            return;
        }
        let target = if self.min_frame_interval.is_zero() {
            frame_end + self.cpu_relief_interval.max(Duration::from_micros(500))
        } else {
            let elapsed = frame_end.saturating_duration_since(frame_start);
            if elapsed >= self.min_frame_interval {
                frame_end + self.cpu_relief_interval.max(Duration::from_micros(500))
            } else {
                frame_start + self.min_frame_interval
            }
        };
        self.next_frame_at = target;
    }
}

/// Detects whether the adapter/profile is likely backed by unified memory.
pub fn is_unified_memory_profile(
    profile: &MobileGpuProfile,
    adapter_info: &wgpu::AdapterInfo,
) -> bool {
    matches!(profile.architecture, TbdrArchitecture::AppleTbdr)
        || matches!(profile.family, MobileGpuFamily::Apple)
        || matches!(adapter_info.backend, wgpu::Backend::Metal)
        // Most mobile TBDR iGPU paths are system-memory-backed and should use
        // conservative memory/throughput policy to avoid OS-wide contention.
        || (profile.is_mobile_tbdr()
            && matches!(
                adapter_info.device_type,
                wgpu::DeviceType::IntegratedGpu | wgpu::DeviceType::Other
            ))
}

/// Returns true when the selected present mode is expected to be uncapped.
pub fn present_mode_allows_uncapped(mode: PresentMode) -> bool {
    matches!(mode, PresentMode::Immediate | PresentMode::AutoNoVsync)
}

/// Select a present mode from adapter capabilities according to the preference.
pub fn select_present_mode(
    available: &[PresentMode],
    prefer_stable: bool,
    vsync_mode: VsyncMode,
) -> PresentMode {
    let has = |mode: PresentMode| available.contains(&mode);
    match vsync_mode {
        VsyncMode::On => {
            if has(PresentMode::Fifo) {
                PresentMode::Fifo
            } else {
                available.first().copied().unwrap_or(PresentMode::AutoVsync)
            }
        }
        VsyncMode::Off => {
            if has(PresentMode::Immediate) {
                PresentMode::Immediate
            } else if has(PresentMode::AutoNoVsync) {
                PresentMode::AutoNoVsync
            } else if has(PresentMode::Mailbox) {
                PresentMode::Mailbox
            } else {
                available
                    .first()
                    .copied()
                    .unwrap_or(PresentMode::AutoNoVsync)
            }
        }
        VsyncMode::Auto => {
            if prefer_stable && has(PresentMode::Fifo) {
                PresentMode::Fifo
            } else if has(PresentMode::AutoVsync) {
                PresentMode::AutoVsync
            } else {
                available.first().copied().unwrap_or(PresentMode::AutoVsync)
            }
        }
    }
}

/// Chooses stable vs throughput pacing from user-facing mode + vsync preference.
pub fn choose_pacing_mode(mode: RuntimeMode, vsync_mode: VsyncMode) -> RuntimePacingMode {
    match mode {
        RuntimeMode::Stable => RuntimePacingMode::Stable,
        RuntimeMode::MaxThroughput => RuntimePacingMode::MaxThroughput,
        RuntimeMode::Auto => match vsync_mode {
            VsyncMode::On => RuntimePacingMode::Stable,
            VsyncMode::Off => RuntimePacingMode::MaxThroughput,
            VsyncMode::Auto => RuntimePacingMode::Stable,
        },
    }
}

/// Selects baseline throughput burst from profile + mode.
pub fn select_throughput_burst(
    profile: &MobileGpuProfile,
    mode: RuntimeMode,
    uma_shared_memory: bool,
) -> u32 {
    let mut burst = match mode {
        RuntimeMode::Stable => 1,
        RuntimeMode::MaxThroughput => match profile.architecture {
            TbdrArchitecture::AppleTbdr => 4,
            TbdrArchitecture::MaliTbdr => mali_burst_from_cores(profile.estimated_cores, true),
            TbdrArchitecture::FlexRender => 8,
            TbdrArchitecture::PowerVrTbdr => 4,
            TbdrArchitecture::Unknown => 3,
        },
        RuntimeMode::Auto => {
            if profile.is_mobile_tbdr() {
                if matches!(profile.architecture, TbdrArchitecture::AppleTbdr) {
                    2
                } else if matches!(profile.architecture, TbdrArchitecture::MaliTbdr) {
                    mali_burst_from_cores(profile.estimated_cores, false)
                } else {
                    3
                }
            } else {
                2
            }
        }
    };

    if uma_shared_memory {
        burst = burst.min(match mode {
            RuntimeMode::Stable => 1,
            RuntimeMode::Auto => 2,
            RuntimeMode::MaxThroughput => 4,
        });
    }
    burst.max(1)
}

fn mali_burst_from_cores(estimated_cores: u32, max_mode: bool) -> u32 {
    if max_mode {
        match estimated_cores {
            0..=7 => 4,
            8..=13 => 6,
            _ => 8,
        }
    } else {
        match estimated_cores {
            0..=7 => 3,
            8..=13 => 4,
            _ => 5,
        }
    }
}

/// Startup easing to avoid first-frame spikes.
pub fn startup_ramp(target: u32, submitted_frames: u64) -> u32 {
    if target <= 1 {
        return target;
    }
    let ramp_frames = 60.0;
    let progress = ((submitted_frames.saturating_add(1)) as f64 / ramp_frames).clamp(0.0, 1.0);
    let eased = progress * progress * (3.0 - 2.0 * progress);
    let scaled = 1.0 + (target.saturating_sub(1) as f64) * eased;
    scaled.round().clamp(1.0, target as f64) as u32
}

/// Heuristic minimum frame interval for max-throughput mode.
pub fn recommended_min_frame_interval(
    profile: &MobileGpuProfile,
    uma_shared_memory: bool,
    throughput_target_burst: u32,
    pacing_mode: RuntimePacingMode,
) -> Duration {
    if !matches!(pacing_mode, RuntimePacingMode::MaxThroughput) || throughput_target_burst <= 1 {
        return Duration::ZERO;
    }
    if uma_shared_memory {
        Duration::from_micros(2_400)
    } else if profile.is_mobile_tbdr() {
        Duration::from_micros(1_100)
    } else {
        Duration::from_micros(700)
    }
}

/// Policy for no-vsync fallback when compositor still caps presentation.
#[derive(Debug, Clone, Copy)]
pub struct AggressiveNoVsyncPolicy {
    pub enabled: bool,
    pub forced_present_interval: u32,
}

impl AggressiveNoVsyncPolicy {
    /// Builds policy from selected present mode and user preference.
    pub fn from_selection(
        vsync_mode: VsyncMode,
        selected_present_mode: PresentMode,
        uma_shared_memory: bool,
    ) -> Self {
        let enabled = matches!(vsync_mode, VsyncMode::Off)
            && !present_mode_allows_uncapped(selected_present_mode);
        let forced_present_interval = if enabled {
            if uma_shared_memory {
                4
            } else {
                3
            }
        } else {
            1
        };
        Self {
            enabled,
            forced_present_interval,
        }
    }

    /// Returns true when this frame should be presented to the surface.
    pub fn should_present(&self, submitted_frames: u64) -> bool {
        if !self.enabled {
            return true;
        }
        submitted_frames % u64::from(self.forced_present_interval.max(1)) == 0
    }
}

/// Memory policy for throughput resources.
#[derive(Debug, Clone, Copy)]
pub struct ThroughputMemoryPolicy {
    uma_shared_memory: bool,
}

impl ThroughputMemoryPolicy {
    pub fn new(uma_shared_memory: bool) -> Self {
        Self { uma_shared_memory }
    }

    /// Recommended surface frame latency.
    pub fn desired_surface_frame_latency(&self) -> u32 {
        if self.uma_shared_memory {
            2
        } else {
            3
        }
    }

    /// Recommended ring length for offscreen throughput targets.
    pub fn desired_ring_len(&self, required_work_units: usize) -> usize {
        let max_ring_len = if self.uma_shared_memory { 6 } else { 12 };
        required_work_units.max(2).min(max_ring_len)
    }

    /// Offscreen throughput target dimensions.
    pub fn offscreen_dimensions(&self, width: u32, height: u32) -> (u32, u32) {
        if self.uma_shared_memory {
            (width.max(1).div_ceil(2), height.max(1).div_ceil(2))
        } else {
            (width.max(1), height.max(1))
        }
    }

    /// Store operation for throughput-only offscreen passes.
    pub fn throughput_store_op(&self) -> StoreOp {
        if self.uma_shared_memory {
            StoreOp::Discard
        } else {
            StoreOp::Store
        }
    }

    /// Initial preallocation target count.
    pub fn initial_target_count(&self, throughput_target_burst: u32) -> usize {
        if self.uma_shared_memory && throughput_target_burst > 1 {
            12
        } else {
            0
        }
    }

    /// Suggested target trim size under memory pressure.
    pub fn pressure_trim_keep_len(&self) -> Option<usize> {
        if self.uma_shared_memory {
            Some(2)
        } else {
            None
        }
    }

    pub fn is_uma(&self) -> bool {
        self.uma_shared_memory
    }

    /// Recommended MPS-style compressed spill policy for this memory profile.
    pub fn recommended_zram_config(
        &self,
        profile: &MobileGpuProfile,
        logical_threads: usize,
    ) -> MpsZramConfig {
        MpsZramConfig::for_profile(profile, logical_threads, self.uma_shared_memory)
    }
}

/// Adaptive controller for throughput burst stability.
#[derive(Debug, Clone)]
pub struct AdaptiveBurstController {
    enabled: bool,
    uma_shared_memory: bool,
    min_units: u32,
    max_units_soft: u32,
    cap_units: u32,
    initialized: bool,
    ema_ms_per_unit: f64,
    ema_abs_delta: f64,
    cooldown_frames: u32,
    recovery_frames: u32,
    low_jitter_streak: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::MobileGpuProfile;

    fn make_adapter_info(
        name: &str,
        backend: wgpu::Backend,
        device_type: wgpu::DeviceType,
    ) -> wgpu::AdapterInfo {
        wgpu::AdapterInfo {
            name: name.to_string(),
            vendor: 0,
            device: 0,
            device_type,
            device_pci_bus_id: String::new(),
            driver: String::new(),
            driver_info: String::new(),
            backend,
            subgroup_min_size: 1,
            subgroup_max_size: 1,
            transient_saves_memory: false,
        }
    }

    #[test]
    fn vsync_off_prefers_immediate() {
        let selected = select_present_mode(
            &[
                PresentMode::Fifo,
                PresentMode::Mailbox,
                PresentMode::Immediate,
            ],
            false,
            VsyncMode::Off,
        );
        assert_eq!(selected, PresentMode::Immediate);
    }

    #[test]
    fn aggressive_no_vsync_enables_when_uncapped_unavailable() {
        let policy =
            AggressiveNoVsyncPolicy::from_selection(VsyncMode::Off, PresentMode::Fifo, false);
        assert!(policy.enabled);
        assert_eq!(policy.forced_present_interval, 3);
        assert!(policy.should_present(0));
        assert!(!policy.should_present(1));
    }

    #[test]
    fn uma_memory_policy_scales_offscreen_target() {
        let policy = ThroughputMemoryPolicy::new(true);
        let (w, h) = policy.offscreen_dimensions(1280, 720);
        assert_eq!(w, 640);
        assert_eq!(h, 360);
        assert_eq!(policy.throughput_store_op(), StoreOp::Discard);
    }

    #[test]
    fn uma_memory_policy_recommends_conservative_zram_budget() {
        let policy = ThroughputMemoryPolicy::new(true);
        let profile = MobileGpuProfile::detect("Mali-G610");
        let cfg = policy.recommended_zram_config(&profile, 8);
        assert!(cfg.enabled);
        assert!(cfg.target_compressed_bytes <= 24 * 1024 * 1024);
    }

    #[test]
    fn adaptive_burst_controller_clamps_to_bounds() {
        let mut controller = AdaptiveBurstController::new(
            4,
            RuntimeMode::MaxThroughput,
            PresentMode::Immediate,
            true,
        );
        let resolved = controller.resolve(12);
        assert!(resolved >= 1);
        controller.observe(16.0, resolved, 12, false);
        assert!(controller.current_cap() >= 1);
    }

    #[test]
    fn burst_profile_for_apple_auto_is_conservative() {
        let profile = MobileGpuProfile::detect("Apple M4");
        let burst = select_throughput_burst(&profile, RuntimeMode::Auto, true);
        assert_eq!(burst, 2);
    }

    #[test]
    fn burst_profile_for_high_core_mali_scales_up() {
        let profile = MobileGpuProfile::detect("Mali-G78 MC24");
        let burst_auto = select_throughput_burst(&profile, RuntimeMode::Auto, false);
        let burst_max = select_throughput_burst(&profile, RuntimeMode::MaxThroughput, false);
        assert!(burst_auto >= 5);
        assert!(burst_max >= 8);
    }

    #[test]
    fn burst_profile_for_low_core_mali_stays_bounded() {
        let profile = MobileGpuProfile::detect("Mali-G52");
        let burst_auto = select_throughput_burst(&profile, RuntimeMode::Auto, false);
        let burst_max = select_throughput_burst(&profile, RuntimeMode::MaxThroughput, false);
        assert_eq!(burst_auto, 3);
        assert_eq!(burst_max, 4);
    }

    #[test]
    fn mali_integrated_profile_is_treated_as_unified_memory() {
        let profile = MobileGpuProfile::detect("Mali-G610");
        let info = make_adapter_info(
            "Mali-G610",
            wgpu::Backend::Vulkan,
            wgpu::DeviceType::IntegratedGpu,
        );
        assert!(is_unified_memory_profile(&profile, &info));
    }

    #[test]
    fn discrete_desktop_profile_is_not_unified_by_mobile_rule() {
        let profile = MobileGpuProfile::detect("NVIDIA GeForce RTX 5060 Ti");
        let info = make_adapter_info(
            "NVIDIA GeForce RTX 5060 Ti",
            wgpu::Backend::Vulkan,
            wgpu::DeviceType::DiscreteGpu,
        );
        assert!(!is_unified_memory_profile(&profile, &info));
    }

    #[test]
    fn clamps_3d_dimension_to_supported_limit() {
        let requested = Limits::default();
        let mut supported = Limits::default();
        supported.max_texture_dimension_3d = 512;
        let (clamped, report) = clamp_required_limits_to_supported(requested, &supported);
        assert!(report.max_texture_dimension_3d);
        assert_eq!(clamped.max_texture_dimension_3d, 512);
    }

    #[test]
    fn frame_pacer_wait_target_advances_in_max_mode() {
        let now = Instant::now();
        let mut pacer = ThroughputFramePacer::new(
            RuntimePacingMode::MaxThroughput,
            Duration::from_millis(2),
            Duration::from_millis(1),
            now,
        );
        let wait_until = pacer
            .preferred_wait_until(now)
            .expect("max mode should provide wait target");
        assert!(wait_until >= now);
        pacer.on_frame_complete(now, now + Duration::from_millis(1));
        assert!(pacer.preferred_wait_until(now).is_some());
    }
}

impl AdaptiveBurstController {
    pub fn new(
        base_units: u32,
        mode: RuntimeMode,
        present_mode: PresentMode,
        uma_shared_memory: bool,
    ) -> Self {
        let enabled = base_units > 1
            && !matches!(mode, RuntimeMode::Stable)
            && present_mode_allows_uncapped(present_mode);
        let min_units = if !enabled {
            1
        } else if uma_shared_memory {
            1
        } else {
            2
        };
        let max_units_soft = if uma_shared_memory { 8 } else { 24 };
        Self {
            enabled,
            uma_shared_memory,
            min_units,
            max_units_soft,
            cap_units: base_units.max(1),
            initialized: false,
            ema_ms_per_unit: 0.0,
            ema_abs_delta: 0.0,
            cooldown_frames: 0,
            recovery_frames: 0,
            low_jitter_streak: 0,
        }
    }

    pub fn reset(&mut self, base_units: u32) {
        self.cap_units = base_units.max(self.min_units).min(self.max_units_soft);
        self.initialized = false;
        self.ema_ms_per_unit = 0.0;
        self.ema_abs_delta = 0.0;
        self.cooldown_frames = 0;
        self.recovery_frames = 0;
        self.low_jitter_streak = 0;
    }

    pub fn resolve(&self, planned_units: u32) -> u32 {
        let planned_units = planned_units.max(1);
        if !self.enabled {
            return planned_units;
        }
        planned_units.min(self.cap_units.max(self.min_units))
    }

    pub fn observe(
        &mut self,
        frame_ms: f64,
        used_units: u32,
        planned_units: u32,
        memory_pressure: bool,
    ) {
        if !self.enabled {
            return;
        }

        let upper_bound = planned_units.max(1).min(self.max_units_soft).clamp(1, 24);
        let lower_bound = self.min_units.min(upper_bound);
        self.cap_units = self.cap_units.clamp(lower_bound, upper_bound);

        let sample = (frame_ms / f64::from(used_units.max(1))).max(0.0001);
        if !self.initialized {
            self.initialized = true;
            self.ema_ms_per_unit = sample;
            self.ema_abs_delta = 0.0;
            return;
        }

        let alpha = 0.08;
        self.ema_ms_per_unit = self.ema_ms_per_unit * (1.0 - alpha) + sample * alpha;
        let abs_delta = (sample - self.ema_ms_per_unit).abs();
        self.ema_abs_delta = self.ema_abs_delta * (1.0 - alpha) + abs_delta * alpha;
        let jitter_ratio = self.ema_abs_delta / self.ema_ms_per_unit.max(0.0001);

        if memory_pressure {
            self.cap_units = self.cap_units.saturating_sub(1).max(lower_bound);
            self.cooldown_frames = if self.uma_shared_memory { 28 } else { 20 };
            self.recovery_frames = if self.uma_shared_memory { 32 } else { 0 };
            self.low_jitter_streak = 0;
            return;
        }

        if self.cooldown_frames > 0 {
            self.cooldown_frames = self.cooldown_frames.saturating_sub(1);
        }
        if self.recovery_frames > 0 {
            self.recovery_frames = self.recovery_frames.saturating_sub(1);
            if self.uma_shared_memory && self.cap_units > 2 {
                self.cap_units = self.cap_units.saturating_sub(1).max(lower_bound);
            }
        }

        let spike = if self.uma_shared_memory {
            sample > self.ema_ms_per_unit * 1.70
        } else {
            sample > self.ema_ms_per_unit * 1.85
        };
        let high_jitter = if self.uma_shared_memory {
            jitter_ratio > 0.18 || spike
        } else {
            jitter_ratio > 0.24 || spike
        };

        if high_jitter {
            // Keep decrease policy relaxed to avoid over-correcting on short spikes.
            let reduction = if self.uma_shared_memory {
                if jitter_ratio > 0.34 || sample > self.ema_ms_per_unit * 2.25 {
                    2
                } else {
                    1
                }
            } else if jitter_ratio > 0.45 || sample > self.ema_ms_per_unit * 2.40 {
                2
            } else {
                1
            };
            self.cap_units = self.cap_units.saturating_sub(reduction).max(lower_bound);
            self.cooldown_frames = if self.uma_shared_memory { 22 } else { 14 };
            if self.uma_shared_memory {
                self.recovery_frames = 24;
            }
            self.low_jitter_streak = 0;
        } else if (if self.uma_shared_memory {
            jitter_ratio < 0.06
        } else {
            jitter_ratio < 0.10
        }) && self.cap_units < upper_bound
        {
            self.low_jitter_streak = self.low_jitter_streak.saturating_add(1);
            let growth_streak_target = if self.uma_shared_memory { 48 } else { 24 };
            if self.low_jitter_streak >= growth_streak_target
                && self.cooldown_frames == 0
                && self.recovery_frames == 0
            {
                self.cap_units = (self.cap_units + 1).min(upper_bound);
                self.low_jitter_streak = 0;
            }
        } else {
            self.low_jitter_streak = 0;
        }

        self.cap_units = self.cap_units.clamp(lower_bound, upper_bound);
    }

    pub fn current_cap(&self) -> u32 {
        self.cap_units.max(1)
    }
}
