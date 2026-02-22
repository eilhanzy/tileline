//! Adaptive UMA buffer and command submission regulator for GMS.
//!
//! This module is designed for unified-memory GPUs (such as Apple M-series on Metal), but it is
//! generic enough to be reused by other integrated GPU paths. It combines:
//! - a frame stability monitor (stddev over the last N frames),
//! - a sliding-window command encoder submission regulator,
//! - and a shared-buffer arbiter that bounds CPU/GPU access to mapped ranges.
//!
//! The primary entry point is [`AdaptiveBuffer::reconcile`], which consumes per-frame telemetry
//! and returns a decision structure that balances throughput and stability.

use std::collections::{HashMap, VecDeque};

/// Configuration for [`AdaptiveBuffer`].
#[derive(Debug, Clone, Copy)]
pub struct AdaptiveBufferConfig {
    /// Number of recent frames tracked by the stability monitor.
    pub stability_window_frames: usize,
    /// Enter recovery mode when the frame-time stddev exceeds this threshold.
    pub recovery_stddev_threshold_ms: f64,
    /// Exit recovery mode once stddev falls below this threshold (hysteresis).
    pub recovery_exit_stddev_threshold_ms: f64,
    /// Minimum concurrent encoder submissions allowed by the regulator.
    pub encoder_window_min: u32,
    /// Maximum concurrent encoder submissions allowed by the regulator.
    pub encoder_window_max: u32,
    /// Initial concurrent encoder submission window.
    pub encoder_window_initial: u32,
    /// Sliding window length for recent encoder submission history.
    pub encoder_history_frames: usize,
    /// Base shared CPU map budget for UMA locking.
    pub uma_cpu_map_budget_bytes: u64,
    /// Base shared GPU access budget for UMA locking.
    pub uma_gpu_shared_budget_bytes: u64,
    /// M4 integrated GPU GMS hardware score baseline used to avoid iGPU overuse.
    pub igpu_hardware_score_baseline: u64,
    /// Number of consecutive stable frames required before gentle recovery exit expansion.
    pub stable_frames_for_relaxation: u32,
}

impl Default for AdaptiveBufferConfig {
    fn default() -> Self {
        Self {
            stability_window_frames: 100,
            recovery_stddev_threshold_ms: 0.5,
            recovery_exit_stddev_threshold_ms: 0.35,
            encoder_window_min: 1,
            encoder_window_max: 8,
            encoder_window_initial: 4,
            encoder_history_frames: 16,
            uma_cpu_map_budget_bytes: 8 * 1024 * 1024,
            uma_gpu_shared_budget_bytes: 16 * 1024 * 1024,
            igpu_hardware_score_baseline: 1_229,
            stable_frames_for_relaxation: 24,
        }
    }
}

/// Operating mode of the adaptive regulator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdaptiveBufferMode {
    /// Normal mode prioritizes throughput while respecting current budgets.
    Normal,
    /// Recovery mode prioritizes frame-time stability and reduces concurrency pressure.
    StabilityRecovery,
}

/// Per-frame telemetry consumed by [`AdaptiveBuffer::reconcile`].
#[derive(Debug, Clone, Copy)]
pub struct AdaptiveFrameTelemetry {
    /// Frame time in milliseconds (or normalized work-unit time, depending on caller policy).
    pub frame_time_ms: f64,
    /// Number of encoders submitted in the current frame.
    pub submitted_encoders: u32,
    /// Current in-flight encoders/submissions at the time of reconciliation.
    pub in_flight_encoders: u32,
    /// GMS hardware score for the integrated GPU. `0` falls back to baseline.
    pub igpu_gms_hardware_score: u64,
}

impl Default for AdaptiveFrameTelemetry {
    fn default() -> Self {
        Self {
            frame_time_ms: 0.0,
            submitted_encoders: 0,
            in_flight_encoders: 0,
            igpu_gms_hardware_score: 0,
        }
    }
}

/// Output of [`AdaptiveBuffer::reconcile`].
#[derive(Debug, Clone, Copy)]
pub struct AdaptiveBufferDecision {
    /// Current operating mode.
    pub mode: AdaptiveBufferMode,
    /// Stddev of the tracked frame window.
    pub frame_stddev_ms: f64,
    /// Mean of the tracked frame window.
    pub frame_mean_ms: f64,
    /// Current allowed concurrent encoder submissions.
    pub max_concurrent_encoders: u32,
    /// Recommended encoder submissions for the next frame.
    pub recommended_encoder_submissions: u32,
    /// Whether a new encoder submission should be deferred this frame.
    pub should_defer_encoder_submission: bool,
    /// Current effective CPU map budget in bytes.
    pub cpu_map_budget_bytes: u64,
    /// Current effective GPU shared-buffer budget in bytes.
    pub gpu_shared_budget_bytes: u64,
    /// Total bytes currently held by CPU-side mapped `BufferView`s.
    pub cpu_mapped_bytes_in_use: u64,
    /// Total bytes currently reserved for GPU-side shared access.
    pub gpu_shared_bytes_in_use: u64,
    /// Total UMA contention events observed since the last reconcile.
    pub contention_events_since_last_reconcile: u32,
    /// Score normalization factor vs baseline (1.0 ~= M4 baseline score 1229).
    pub hardware_score_factor: f64,
}

/// Identifies a logical shared buffer region managed by the UMA arbiter.
pub type SharedBufferKey = u64;

/// Which side currently holds a shared buffer lease.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SharedBufferOwner {
    CpuRead,
    CpuWrite,
    Gpu,
}

/// Opaque lease token returned when buffer access is granted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SharedBufferLease {
    key: SharedBufferKey,
    owner: SharedBufferOwner,
    bytes: u64,
    generation: u64,
}

/// Lock failure reason for shared UMA buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SharedBufferLockError {
    /// CPU and GPU attempted overlapping ownership without sufficient budget/slack.
    Contended,
    /// The requested lease exceeds the current adaptive budget.
    BudgetExceeded,
    /// The lease token was invalid for release (stale generation or wrong owner).
    InvalidLease,
}

#[derive(Debug, Clone, Copy, Default)]
struct SharedBufferEntry {
    cpu_readers: u32,
    cpu_writers: u32,
    gpu_holders: u32,
    cpu_bytes: u64,
    gpu_bytes: u64,
    generation: u64,
}

/// Adaptive UMA regulator used by GMS runtimes to stabilize integrated GPU command submission.
///
/// Typical usage per frame:
/// 1. Register/limit shared buffer access via `try_acquire_*` / `release_lease`.
/// 2. Submit work according to the previous decision.
/// 3. Call [`reconcile`](Self::reconcile) with frame telemetry.
pub struct AdaptiveBuffer {
    config: AdaptiveBufferConfig,
    mode: AdaptiveBufferMode,
    frame_times_ms: VecDeque<f64>,
    encoder_submissions_window: VecDeque<u32>,
    max_concurrent_encoders: u32,
    cpu_map_budget_bytes: u64,
    gpu_shared_budget_bytes: u64,
    cpu_mapped_bytes_in_use: u64,
    gpu_shared_bytes_in_use: u64,
    contention_events_since_last_reconcile: u32,
    stable_frames_streak: u32,
    shared_buffers: HashMap<SharedBufferKey, SharedBufferEntry>,
    lease_generation_counter: u64,
}

impl AdaptiveBuffer {
    /// Create a new adaptive regulator with custom configuration.
    pub fn new(config: AdaptiveBufferConfig) -> Self {
        let encoder_initial = config
            .encoder_window_initial
            .clamp(config.encoder_window_min, config.encoder_window_max);

        Self {
            config,
            mode: AdaptiveBufferMode::Normal,
            frame_times_ms: VecDeque::with_capacity(config.stability_window_frames.max(4)),
            encoder_submissions_window: VecDeque::with_capacity(
                config.encoder_history_frames.max(4),
            ),
            max_concurrent_encoders: encoder_initial,
            cpu_map_budget_bytes: config.uma_cpu_map_budget_bytes,
            gpu_shared_budget_bytes: config.uma_gpu_shared_budget_bytes,
            cpu_mapped_bytes_in_use: 0,
            gpu_shared_bytes_in_use: 0,
            contention_events_since_last_reconcile: 0,
            stable_frames_streak: 0,
            shared_buffers: HashMap::new(),
            lease_generation_counter: 1,
        }
    }

    /// Create with [`AdaptiveBufferConfig::default`].
    pub fn with_defaults() -> Self {
        Self::new(AdaptiveBufferConfig::default())
    }

    /// Access the current mode without mutating internal state.
    pub fn mode(&self) -> AdaptiveBufferMode {
        self.mode
    }

    /// Returns the current concurrent encoder window cap.
    pub fn max_concurrent_encoders(&self) -> u32 {
        self.max_concurrent_encoders
    }

    /// Attempt to acquire a CPU read lease using a mapped `wgpu::BufferView`.
    ///
    /// The requested byte size is derived from `view.len()`, which couples the arbiter directly
    /// to `wgpu::BufferView` usage and reduces accidental mismatches in UMA accounting.
    pub fn try_acquire_cpu_read_view(
        &mut self,
        key: SharedBufferKey,
        view: &wgpu::BufferView,
    ) -> Result<SharedBufferLease, SharedBufferLockError> {
        self.try_acquire_cpu_lease(key, view.len() as u64, SharedBufferOwner::CpuRead)
    }

    /// Attempt to acquire a CPU write lease using a mapped `wgpu::BufferViewMut`.
    pub fn try_acquire_cpu_write_view(
        &mut self,
        key: SharedBufferKey,
        view: &wgpu::BufferViewMut,
    ) -> Result<SharedBufferLease, SharedBufferLockError> {
        self.try_acquire_cpu_lease(key, view.len() as u64, SharedBufferOwner::CpuWrite)
    }

    /// Attempt to acquire a GPU lease for a shared buffer byte range.
    ///
    /// Callers should pass the exact byte span they intend to read/write on the GPU for better
    /// pressure accounting.
    pub fn try_acquire_gpu_range(
        &mut self,
        key: SharedBufferKey,
        bytes: u64,
    ) -> Result<SharedBufferLease, SharedBufferLockError> {
        let bytes = bytes.max(1);

        if self.gpu_shared_bytes_in_use.saturating_add(bytes) > self.gpu_shared_budget_bytes {
            self.contention_events_since_last_reconcile = self
                .contention_events_since_last_reconcile
                .saturating_add(1);
            return Err(SharedBufferLockError::BudgetExceeded);
        }

        let entry = self.shared_buffers.entry(key).or_default();
        let cpu_active = entry.cpu_readers > 0 || entry.cpu_writers > 0;
        if cpu_active {
            self.contention_events_since_last_reconcile = self
                .contention_events_since_last_reconcile
                .saturating_add(1);
            return Err(SharedBufferLockError::Contended);
        }

        entry.gpu_holders = entry.gpu_holders.saturating_add(1);
        entry.gpu_bytes = entry.gpu_bytes.saturating_add(bytes);
        self.gpu_shared_bytes_in_use = self.gpu_shared_bytes_in_use.saturating_add(bytes);

        Ok(self.new_lease(key, SharedBufferOwner::Gpu, bytes))
    }

    /// Release a previously acquired CPU/GPU lease.
    pub fn release_lease(&mut self, lease: SharedBufferLease) -> Result<(), SharedBufferLockError> {
        let Some(entry) = self.shared_buffers.get_mut(&lease.key) else {
            return Err(SharedBufferLockError::InvalidLease);
        };

        if lease.generation == 0 || lease.generation > self.lease_generation_counter {
            return Err(SharedBufferLockError::InvalidLease);
        }

        match lease.owner {
            SharedBufferOwner::CpuRead => {
                if entry.cpu_readers == 0 || entry.cpu_bytes < lease.bytes {
                    return Err(SharedBufferLockError::InvalidLease);
                }
                entry.cpu_readers -= 1;
                entry.cpu_bytes -= lease.bytes;
                self.cpu_mapped_bytes_in_use =
                    self.cpu_mapped_bytes_in_use.saturating_sub(lease.bytes);
            }
            SharedBufferOwner::CpuWrite => {
                if entry.cpu_writers == 0 || entry.cpu_bytes < lease.bytes {
                    return Err(SharedBufferLockError::InvalidLease);
                }
                entry.cpu_writers -= 1;
                entry.cpu_bytes -= lease.bytes;
                self.cpu_mapped_bytes_in_use =
                    self.cpu_mapped_bytes_in_use.saturating_sub(lease.bytes);
            }
            SharedBufferOwner::Gpu => {
                if entry.gpu_holders == 0 || entry.gpu_bytes < lease.bytes {
                    return Err(SharedBufferLockError::InvalidLease);
                }
                entry.gpu_holders -= 1;
                entry.gpu_bytes -= lease.bytes;
                self.gpu_shared_bytes_in_use =
                    self.gpu_shared_bytes_in_use.saturating_sub(lease.bytes);
            }
        }

        if entry.cpu_readers == 0 && entry.cpu_writers == 0 && entry.gpu_holders == 0 {
            // Remove empty bookkeeping entries to keep the map small.
            self.shared_buffers.remove(&lease.key);
        }

        Ok(())
    }

    /// Reconcile the regulator state using the latest frame telemetry.
    ///
    /// This method is the main adaptive loop. It:
    /// - updates the 100-frame stability monitor,
    /// - enters/exits recovery mode based on stddev thresholds,
    /// - scales encoder concurrency with a score-aware heuristic (M4 baseline: 1229),
    /// - and tightens/relaxes UMA budgets based on stability and contention.
    pub fn reconcile(&mut self, telemetry: AdaptiveFrameTelemetry) -> AdaptiveBufferDecision {
        if telemetry.frame_time_ms.is_finite() && telemetry.frame_time_ms > 0.0 {
            self.frame_times_ms.push_back(telemetry.frame_time_ms);
            while self.frame_times_ms.len() > self.config.stability_window_frames.max(4) {
                self.frame_times_ms.pop_front();
            }
        }

        self.encoder_submissions_window
            .push_back(telemetry.submitted_encoders);
        while self.encoder_submissions_window.len() > self.config.encoder_history_frames.max(4) {
            self.encoder_submissions_window.pop_front();
        }

        let frame_mean_ms = mean_f64(self.frame_times_ms.iter().copied()).unwrap_or(0.0);
        let frame_stddev_ms = stddev_f64(self.frame_times_ms.iter().copied()).unwrap_or(0.0);

        let enter_recovery = frame_stddev_ms > self.config.recovery_stddev_threshold_ms;
        let exit_recovery = frame_stddev_ms <= self.config.recovery_exit_stddev_threshold_ms;

        if enter_recovery {
            self.mode = AdaptiveBufferMode::StabilityRecovery;
            self.stable_frames_streak = 0;
        } else if exit_recovery {
            self.stable_frames_streak = self.stable_frames_streak.saturating_add(1);
            if self.mode == AdaptiveBufferMode::StabilityRecovery
                && self.stable_frames_streak >= self.config.stable_frames_for_relaxation
            {
                self.mode = AdaptiveBufferMode::Normal;
            }
        } else {
            self.stable_frames_streak = 0;
        }

        let score = telemetry
            .igpu_gms_hardware_score
            .max(self.config.igpu_hardware_score_baseline);
        let hardware_score_factor =
            (score as f64 / self.config.igpu_hardware_score_baseline.max(1) as f64).clamp(0.5, 4.0);

        self.reconcile_encoder_window(telemetry, frame_stddev_ms, hardware_score_factor);
        self.reconcile_uma_budgets(frame_stddev_ms, hardware_score_factor);

        let recent_encoder_mean = mean_f64(
            self.encoder_submissions_window
                .iter()
                .copied()
                .map(f64::from),
        )
        .unwrap_or(self.max_concurrent_encoders as f64);

        let recommended_encoder_submissions = match self.mode {
            AdaptiveBufferMode::StabilityRecovery => recent_encoder_mean
                .floor()
                .max(1.0)
                .min(self.max_concurrent_encoders as f64)
                as u32,
            AdaptiveBufferMode::Normal => recent_encoder_mean
                .ceil()
                .max(1.0)
                .min(self.max_concurrent_encoders as f64)
                as u32,
        };

        let should_defer_encoder_submission =
            telemetry.in_flight_encoders >= self.max_concurrent_encoders;

        let contention_events = self.contention_events_since_last_reconcile;
        self.contention_events_since_last_reconcile = 0;

        AdaptiveBufferDecision {
            mode: self.mode,
            frame_stddev_ms,
            frame_mean_ms,
            max_concurrent_encoders: self.max_concurrent_encoders,
            recommended_encoder_submissions,
            should_defer_encoder_submission,
            cpu_map_budget_bytes: self.cpu_map_budget_bytes,
            gpu_shared_budget_bytes: self.gpu_shared_budget_bytes,
            cpu_mapped_bytes_in_use: self.cpu_mapped_bytes_in_use,
            gpu_shared_bytes_in_use: self.gpu_shared_bytes_in_use,
            contention_events_since_last_reconcile: contention_events,
            hardware_score_factor,
        }
    }

    fn try_acquire_cpu_lease(
        &mut self,
        key: SharedBufferKey,
        bytes: u64,
        owner: SharedBufferOwner,
    ) -> Result<SharedBufferLease, SharedBufferLockError> {
        let bytes = bytes.max(1);

        if self.cpu_mapped_bytes_in_use.saturating_add(bytes) > self.cpu_map_budget_bytes {
            self.contention_events_since_last_reconcile = self
                .contention_events_since_last_reconcile
                .saturating_add(1);
            return Err(SharedBufferLockError::BudgetExceeded);
        }

        let entry = self.shared_buffers.entry(key).or_default();
        let gpu_active = entry.gpu_holders > 0;
        let cpu_write_active = entry.cpu_writers > 0;

        let is_write = matches!(owner, SharedBufferOwner::CpuWrite);
        if gpu_active
            || (is_write && (entry.cpu_readers > 0 || cpu_write_active))
            || (!is_write && cpu_write_active)
        {
            self.contention_events_since_last_reconcile = self
                .contention_events_since_last_reconcile
                .saturating_add(1);
            return Err(SharedBufferLockError::Contended);
        }

        match owner {
            SharedBufferOwner::CpuRead => {
                entry.cpu_readers = entry.cpu_readers.saturating_add(1);
            }
            SharedBufferOwner::CpuWrite => {
                entry.cpu_writers = entry.cpu_writers.saturating_add(1);
            }
            SharedBufferOwner::Gpu => unreachable!("GPU leases use try_acquire_gpu_range"),
        }
        entry.cpu_bytes = entry.cpu_bytes.saturating_add(bytes);
        self.cpu_mapped_bytes_in_use = self.cpu_mapped_bytes_in_use.saturating_add(bytes);

        Ok(self.new_lease(key, owner, bytes))
    }

    fn new_lease(
        &mut self,
        key: SharedBufferKey,
        owner: SharedBufferOwner,
        bytes: u64,
    ) -> SharedBufferLease {
        let generation = self.lease_generation_counter;
        self.lease_generation_counter = self.lease_generation_counter.saturating_add(1);

        if let Some(entry) = self.shared_buffers.get_mut(&key) {
            entry.generation = generation;
        }

        SharedBufferLease {
            key,
            owner,
            bytes,
            generation,
        }
    }

    fn reconcile_encoder_window(
        &mut self,
        telemetry: AdaptiveFrameTelemetry,
        frame_stddev_ms: f64,
        hardware_score_factor: f64,
    ) {
        let nominal = ((self.config.encoder_window_initial as f64) * hardware_score_factor.sqrt())
            .round()
            .clamp(
                self.config.encoder_window_min as f64,
                self.config.encoder_window_max as f64,
            ) as u32;

        let recent_mean = mean_f64(
            self.encoder_submissions_window
                .iter()
                .copied()
                .map(f64::from),
        )
        .unwrap_or(nominal as f64);

        let recent_target = recent_mean.round().clamp(
            self.config.encoder_window_min as f64,
            self.config.encoder_window_max as f64,
        ) as u32;

        if self.mode == AdaptiveBufferMode::StabilityRecovery {
            // Reduce pressure quickly when frame pacing is unstable or UMA contention spikes.
            let pressure = (frame_stddev_ms / self.config.recovery_stddev_threshold_ms.max(0.001))
                .clamp(1.0, 4.0);
            let reduction = pressure.ceil() as u32;
            let target = recent_target
                .min(nominal)
                .saturating_sub(reduction)
                .max(self.config.encoder_window_min);
            self.max_concurrent_encoders =
                self.max_concurrent_encoders.saturating_sub(1).max(target);
        } else {
            // Relax slowly to avoid oscillation.
            let target = nominal.max(recent_target);
            if self.max_concurrent_encoders < target {
                self.max_concurrent_encoders = self.max_concurrent_encoders.saturating_add(1);
            } else if telemetry.in_flight_encoders + 2 < self.max_concurrent_encoders {
                self.max_concurrent_encoders = self.max_concurrent_encoders.saturating_sub(1);
            }
        }

        self.max_concurrent_encoders = self.max_concurrent_encoders.clamp(
            self.config.encoder_window_min,
            self.config.encoder_window_max,
        );
    }

    fn reconcile_uma_budgets(&mut self, frame_stddev_ms: f64, hardware_score_factor: f64) {
        let score_guard = (hardware_score_factor / 1.0).clamp(0.75, 1.5);

        let base_cpu = (self.config.uma_cpu_map_budget_bytes as f64 * score_guard).round() as u64;
        let base_gpu =
            (self.config.uma_gpu_shared_budget_bytes as f64 * score_guard).round() as u64;

        let (cpu_factor, gpu_factor) = if self.mode == AdaptiveBufferMode::StabilityRecovery {
            // Tighten budgets in recovery mode to reduce CPU/GPU overlap on shared UMA buffers.
            let severity = (frame_stddev_ms / self.config.recovery_stddev_threshold_ms.max(0.001))
                .clamp(1.0, 3.0);
            let tighten = (1.0 / severity).clamp(0.35, 1.0);
            (tighten, tighten)
        } else {
            (1.0, 1.0)
        };

        self.cpu_map_budget_bytes = ((base_cpu as f64) * cpu_factor).round().max(256.0) as u64;
        self.gpu_shared_budget_bytes = ((base_gpu as f64) * gpu_factor).round().max(256.0) as u64;

        // Never shrink below what is already actively leased; defer effective reductions until
        // outstanding leases are released.
        self.cpu_map_budget_bytes = self.cpu_map_budget_bytes.max(self.cpu_mapped_bytes_in_use);
        self.gpu_shared_budget_bytes = self
            .gpu_shared_budget_bytes
            .max(self.gpu_shared_bytes_in_use);
    }
}

fn mean_f64<I>(values: I) -> Option<f64>
where
    I: IntoIterator<Item = f64>,
{
    let mut sum = 0.0f64;
    let mut count = 0usize;
    for value in values {
        if !value.is_finite() {
            continue;
        }
        sum += value;
        count += 1;
    }
    if count == 0 {
        None
    } else {
        Some(sum / count as f64)
    }
}

fn stddev_f64<I>(values: I) -> Option<f64>
where
    I: IntoIterator<Item = f64>,
{
    let filtered = values
        .into_iter()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();

    if filtered.len() < 2 {
        return None;
    }

    let mean = filtered.iter().sum::<f64>() / filtered.len() as f64;
    let variance = filtered
        .iter()
        .map(|value| {
            let delta = *value - mean;
            delta * delta
        })
        .sum::<f64>()
        / filtered.len() as f64;

    Some(variance.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enters_recovery_when_stddev_exceeds_threshold() {
        let mut adaptive = AdaptiveBuffer::with_defaults();

        // Prime with stable frames.
        for _ in 0..100 {
            let decision = adaptive.reconcile(AdaptiveFrameTelemetry {
                frame_time_ms: 1.0,
                submitted_encoders: 2,
                in_flight_encoders: 1,
                igpu_gms_hardware_score: 1_229,
            });
            assert!(decision.frame_stddev_ms <= 0.001);
        }

        // Inject jitter to exceed the 0.5ms threshold.
        let mut last_decision = adaptive.reconcile(AdaptiveFrameTelemetry::default());
        for i in 0..120 {
            let frame = if i % 2 == 0 { 0.1 } else { 3.0 };
            last_decision = adaptive.reconcile(AdaptiveFrameTelemetry {
                frame_time_ms: frame,
                submitted_encoders: 4,
                in_flight_encoders: 4,
                igpu_gms_hardware_score: 1_229,
            });
        }

        assert_eq!(adaptive.mode(), AdaptiveBufferMode::StabilityRecovery);
        assert!(last_decision.frame_stddev_ms > adaptive.config.recovery_stddev_threshold_ms);
        assert!(adaptive.max_concurrent_encoders() <= 4);
    }

    #[test]
    fn score_baseline_limits_encoder_growth() {
        let mut adaptive = AdaptiveBuffer::with_defaults();

        for _ in 0..150 {
            adaptive.reconcile(AdaptiveFrameTelemetry {
                frame_time_ms: 1.0,
                submitted_encoders: 8,
                in_flight_encoders: 0,
                igpu_gms_hardware_score: 1_229, // M4 baseline
            });
        }

        assert!(adaptive.max_concurrent_encoders() <= adaptive.config.encoder_window_max);
    }
}
