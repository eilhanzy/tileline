//! Multi-GPU synchronization primitives for Tileline core.
//!
//! This module provides a portable synchronization layer for explicit multi-GPU execution using
//! `wgpu` fence/semaphore equivalents:
//! - queue submission indices (`SubmissionIndex`) as timeline markers
//! - `Device::poll(PollType::Wait { .. })` as bounded fence waits
//! - a frame-scoped compose barrier with a strict sub-millisecond wait budget
//!
//! The implementation is intentionally backend-aware (Vulkan / Metal hints) while remaining
//! portable. Native semaphores/fences are not directly exposed by `wgpu`, so the runtime uses
//! submission timelines and bounded waits to avoid long CPU stalls.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use gms::{
    AdaptiveBuffer, AdaptiveBufferConfig, AdaptiveBufferDecision, AdaptiveFrameTelemetry,
    GpuAdapterProfile, MemoryTopology, MultiGpuSyncPlan, SharedBufferKey, SharedBufferLease,
    SharedBufferLockError,
};
use wgpu::{Device, PollError, PollStatus};

const APPLE_VENDOR_ID: u32 = 0x106B;

/// Logical queue/lane tracked by the synchronizer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuQueueLane {
    /// Primary/present GPU compute+graphics queue.
    Primary,
    /// Secondary helper GPU queue.
    Secondary,
    /// Transfer/copy queue (host-bridge uploads/readbacks), if the runtime separates it.
    Transfer,
}

/// Backend-neutral submission handle recorded against a queue lane.
#[derive(Debug, Clone)]
pub enum GpuSubmissionHandle {
    Wgpu(wgpu::SubmissionIndex),
    Serial(u64),
}

/// Portable result of polling or waiting on a GPU queue submission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuSubmissionWaitStatus {
    Ready,
    Pending,
    TimedOut,
    Invalid,
}

/// Backend-neutral submission wait surface used by the synchronizer.
pub trait GpuSubmissionWaiter {
    fn wait_submission(
        &self,
        submission: &GpuSubmissionHandle,
        timeout: Option<Duration>,
    ) -> GpuSubmissionWaitStatus;
}

/// `wgpu` adapter that preserves the old queue-submission wait behavior behind a neutral trait.
pub struct WgpuSubmissionWaiter<'a> {
    device: &'a Device,
}

impl<'a> WgpuSubmissionWaiter<'a> {
    pub fn new(device: &'a Device) -> Self {
        Self { device }
    }
}

impl GpuSubmissionWaiter for WgpuSubmissionWaiter<'_> {
    fn wait_submission(
        &self,
        submission: &GpuSubmissionHandle,
        timeout: Option<Duration>,
    ) -> GpuSubmissionWaitStatus {
        let GpuSubmissionHandle::Wgpu(submission) = submission else {
            return GpuSubmissionWaitStatus::Invalid;
        };
        match self.device.poll(wgpu::PollType::Wait {
            submission_index: Some(submission.clone()),
            timeout,
        }) {
            Ok(status) if status.wait_finished() => GpuSubmissionWaitStatus::Ready,
            Ok(PollStatus::Poll) => GpuSubmissionWaitStatus::Pending,
            Ok(_) => GpuSubmissionWaitStatus::Pending,
            Err(PollError::Timeout) => GpuSubmissionWaitStatus::TimedOut,
            Err(PollError::WrongSubmissionIndex(_, _)) => GpuSubmissionWaitStatus::Invalid,
        }
    }
}

/// Backend sync style hint for the runtime.
///
/// This does not expose native handles. It tells the runtime which synchronization strategy is the
/// closest conceptual match for profiling, diagnostics, and later native interop extensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncBackendHint {
    /// Portable `wgpu` queue timeline + bounded `poll` waits.
    PortableWgpu,
    /// Vulkan-like timeline semaphore + fence semantics (modeled on top of `wgpu`).
    VulkanTimelineFenceLike,
    /// Metal shared-event + completion handler semantics (modeled on top of `wgpu`).
    MetalSharedEventLike,
}

/// Shared-memory placement policy recommended by the synchronizer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SharedPlacementPolicy {
    /// Host-mapped upload/readback bridge (discrete GPU or mixed topology paths).
    HostMappedBridge,
    /// Unified/shared placement without Metal-specific strictness.
    UnifiedShared,
    /// Apple Silicon UMA: keep shared buffers on strict shared placement semantics.
    ///
    /// In `wgpu` this is realized as a shared/MAP_WRITE-oriented path and stable reuse policy.
    /// A future Metal-native path may bind this to `MTLStorageModeShared` explicitly.
    MetalSharedStrict,
}

/// Per-frame admission result for the synchronizer.
#[derive(Debug, Clone, Copy)]
pub struct FrameSyncAdmission {
    /// Whether the frame was accepted into the in-flight tracking window.
    pub accepted: bool,
    /// Current tracked frames in flight after the admission attempt.
    pub pending_frames: usize,
    /// Signal for upper layers to spill latency-sensitive work back to the primary GPU.
    pub should_spill_secondary: bool,
    /// Best-effort scale hint for secondary helper workload (1.0 means no reduction).
    pub secondary_scale_hint: f64,
}

/// Result of compose-barrier reconciliation for the oldest pending frame.
#[derive(Debug, Clone)]
pub struct ComposeBarrierState {
    /// Frame identifier targeted by this reconcile step.
    pub frame_id: Option<u64>,
    /// `true` when all required queue submissions are complete and composition may proceed.
    pub ready: bool,
    /// `true` if the call performed at least one blocking wait (bounded by config budget).
    pub waited: bool,
    /// `true` if a bounded wait timed out before all required submissions completed.
    pub timed_out: bool,
    /// Total wait time spent in this reconcile call.
    pub wait_time: Duration,
    /// `true` if the configured compose wait budget was exhausted.
    pub compose_budget_exhausted: bool,
    /// How late the oldest tracked frame is relative to the desired compose budget window.
    pub late_by: Option<Duration>,
    /// Backend-specific sync flavor hint.
    pub backend_hint: SyncBackendHint,
    /// Current shared-memory placement recommendation.
    pub shared_placement: SharedPlacementPolicy,
    /// Last adaptive UMA decision if the synchronizer is managing Apple unified memory stability.
    pub apple_uma_decision: Option<AdaptiveBufferDecision>,
}

impl Default for ComposeBarrierState {
    fn default() -> Self {
        Self {
            frame_id: None,
            ready: false,
            waited: false,
            timed_out: false,
            wait_time: Duration::ZERO,
            compose_budget_exhausted: false,
            late_by: None,
            backend_hint: SyncBackendHint::PortableWgpu,
            shared_placement: SharedPlacementPolicy::HostMappedBridge,
            apple_uma_decision: None,
        }
    }
}

/// Runtime configuration for [`MultiGpuFrameSynchronizer`].
#[derive(Debug, Clone, Copy)]
pub struct MultiGpuFrameSyncConfig {
    /// Max bounded wait time for a single compose reconcile call.
    ///
    /// The default is 0.8ms to match the engine requirement for sub-millisecond compose sync.
    pub compose_wait_budget: Duration,
    /// Soft lateness target used for diagnostics and spillback hints.
    pub target_compose_budget: Duration,
    /// Optional override for in-flight frame count (otherwise GMS planner sync plan is used).
    pub max_frames_in_flight_override: Option<u32>,
    /// Target stability percentage for Apple UMA paths (e.g. 0.85 for 85%).
    pub apple_target_stability: f64,
    /// Thermal-throttle guardrail used by Apple UMA mode for diagnostics/policy bias.
    pub apple_thermal_floor_fps: f64,
}

impl Default for MultiGpuFrameSyncConfig {
    fn default() -> Self {
        Self {
            compose_wait_budget: Duration::from_micros(800),
            target_compose_budget: Duration::from_micros(800),
            max_frames_in_flight_override: None,
            apple_target_stability: 0.85,
            apple_thermal_floor_fps: 60.0,
        }
    }
}

/// Snapshot of synchronizer counters and active policy state.
#[derive(Debug, Clone)]
pub struct MultiGpuSyncSnapshot {
    /// Backend synchronization flavor currently used for diagnostics/profiling.
    pub backend_hint: SyncBackendHint,
    /// Recommended shared-memory placement policy for current adapter topology.
    pub shared_placement: SharedPlacementPolicy,
    /// Effective in-flight frame limit enforced by the synchronizer.
    pub frames_in_flight_limit: u32,
    /// Number of frames currently tracked as pending for compose readiness.
    pub tracked_frames: usize,
    /// Number of non-blocking `Device::poll` calls issued by the synchronizer.
    pub poll_calls: u64,
    /// Number of bounded blocking waits issued during compose reconciliation.
    pub wait_calls: u64,
    /// Count of bounded waits that hit timeout before readiness.
    pub wait_timeouts: u64,
    /// Number of waits attempted for invalid or stale submission indices.
    pub invalid_submission_waits: u64,
    /// Number of frames fully completed and retired from tracking.
    pub completed_frames: u64,
    /// Number of frames rejected at admission due to in-flight pressure.
    pub rejected_frames: u64,
    /// `true` when Apple UMA adaptive stabilization is active.
    pub apple_uma_enabled: bool,
    /// Last adaptive UMA decision produced during reconciliation.
    pub last_apple_uma_decision: Option<AdaptiveBufferDecision>,
}

#[derive(Clone)]
struct TrackedFrame {
    frame_id: u64,
    registered_at: Instant,
    require_primary: bool,
    require_secondary: bool,
    require_transfer: bool,
    primary_submission: Option<GpuSubmissionHandle>,
    secondary_submission: Option<GpuSubmissionHandle>,
    transfer_submission: Option<GpuSubmissionHandle>,
}

impl TrackedFrame {
    fn new(frame_id: u64, require_secondary: bool, require_transfer: bool) -> Self {
        Self {
            frame_id,
            registered_at: Instant::now(),
            require_primary: true,
            require_secondary,
            require_transfer,
            primary_submission: None,
            secondary_submission: None,
            transfer_submission: None,
        }
    }

    fn submission_mut(&mut self, lane: GpuQueueLane) -> &mut Option<GpuSubmissionHandle> {
        match lane {
            GpuQueueLane::Primary => &mut self.primary_submission,
            GpuQueueLane::Secondary => &mut self.secondary_submission,
            GpuQueueLane::Transfer => &mut self.transfer_submission,
        }
    }

    fn is_registered_for_lane(&self, lane: GpuQueueLane) -> bool {
        match lane {
            GpuQueueLane::Primary => self.require_primary,
            GpuQueueLane::Secondary => self.require_secondary,
            GpuQueueLane::Transfer => self.require_transfer,
        }
    }
}

struct AppleUmaSyncState {
    adaptive: AdaptiveBuffer,
    last_decision: Option<AdaptiveBufferDecision>,
    _strict_shared_placement: bool,
    _target_stability: f64,
    _thermal_floor_fps: f64,
}

/// Portable explicit multi-GPU frame synchronizer.
///
/// The synchronizer is designed to be called from a render/runtime thread (single owner), while
/// CPU production continues on MPS threads. It avoids `Mutex` and blocking queues; all blocking is
/// opt-in and bounded to a sub-millisecond compose wait budget.
pub struct MultiGpuFrameSynchronizer {
    _plan: MultiGpuSyncPlan,
    config: MultiGpuFrameSyncConfig,
    backend_hint: SyncBackendHint,
    shared_placement: SharedPlacementPolicy,
    frames_in_flight_limit: u32,
    pending_frames: VecDeque<TrackedFrame>,
    poll_calls: u64,
    wait_calls: u64,
    wait_timeouts: u64,
    invalid_submission_waits: u64,
    completed_frames: u64,
    rejected_frames: u64,
    apple_uma: Option<AppleUmaSyncState>,
}

impl MultiGpuFrameSynchronizer {
    /// Create a new synchronizer from a GMS sync plan and adapter topology.
    pub fn new(
        plan: MultiGpuSyncPlan,
        config: MultiGpuFrameSyncConfig,
        primary: &GpuAdapterProfile,
        secondary: Option<&GpuAdapterProfile>,
    ) -> Self {
        let backend_hint = detect_sync_backend_hint(primary, secondary);
        let shared_placement = detect_shared_placement_policy(primary, secondary);
        let frames_in_flight_limit = config
            .max_frames_in_flight_override
            .unwrap_or(plan.frames_in_flight)
            .max(1);
        let apple_uma = build_apple_uma_state(shared_placement, config, primary, secondary);

        Self {
            _plan: plan,
            config,
            backend_hint,
            shared_placement,
            frames_in_flight_limit,
            pending_frames: VecDeque::new(),
            poll_calls: 0,
            wait_calls: 0,
            wait_timeouts: 0,
            invalid_submission_waits: 0,
            completed_frames: 0,
            rejected_frames: 0,
            apple_uma,
        }
    }

    /// Frames-in-flight cap currently enforced by the synchronizer.
    pub fn frames_in_flight_limit(&self) -> u32 {
        self.frames_in_flight_limit
    }

    /// Backend sync hint (Vulkan-like, Metal-like, or portable fallback).
    pub fn backend_hint(&self) -> SyncBackendHint {
        self.backend_hint
    }

    /// Current shared placement recommendation for interop/upload buffers.
    pub fn shared_placement(&self) -> SharedPlacementPolicy {
        self.shared_placement
    }

    /// Non-blocking frame admission. Rejects when the in-flight window is saturated.
    pub fn admit_frame(
        &mut self,
        frame_id: u64,
        require_secondary: bool,
        require_transfer: bool,
    ) -> FrameSyncAdmission {
        let pending = self.pending_frames.len();
        let limit = self.frames_in_flight_limit.max(1) as usize;
        if pending >= limit {
            self.rejected_frames = self.rejected_frames.saturating_add(1);
            let over = pending.saturating_sub(limit).saturating_add(1);
            let pressure = (over as f64 / limit as f64).clamp(0.0, 1.5);
            let secondary_scale_hint = (1.0 - pressure * 0.5).clamp(0.25, 1.0);
            return FrameSyncAdmission {
                accepted: false,
                pending_frames: pending,
                should_spill_secondary: true,
                secondary_scale_hint,
            };
        }

        self.pending_frames.push_back(TrackedFrame::new(
            frame_id,
            require_secondary,
            require_transfer,
        ));

        FrameSyncAdmission {
            accepted: true,
            pending_frames: self.pending_frames.len(),
            should_spill_secondary: false,
            secondary_scale_hint: 1.0,
        }
    }

    /// Record a queue submission index for a tracked frame/lane.
    ///
    /// Returns `true` if the frame was found and the lane is part of that frame.
    pub fn record_submission(
        &mut self,
        frame_id: u64,
        lane: GpuQueueLane,
        submission: GpuSubmissionHandle,
    ) -> bool {
        if let Some(frame) = self
            .pending_frames
            .iter_mut()
            .find(|frame| frame.frame_id == frame_id)
        {
            if !frame.is_registered_for_lane(lane) {
                return false;
            }
            *frame.submission_mut(lane) = Some(submission);
            return true;
        }
        false
    }

    /// Reconcile the oldest pending frame without blocking. This polls devices once and reports
    /// whether composition may proceed.
    pub fn try_reconcile_nonblocking(
        &mut self,
        primary_waiter: &dyn GpuSubmissionWaiter,
        secondary_waiter: Option<&dyn GpuSubmissionWaiter>,
        transfer_waiter: Option<&dyn GpuSubmissionWaiter>,
    ) -> ComposeBarrierState {
        self.reconcile_impl(
            primary_waiter,
            secondary_waiter,
            transfer_waiter,
            WaitMode::NonBlocking,
        )
    }

    /// Reconcile the oldest pending frame using a strict bounded wait (default 0.8ms).
    ///
    /// This is the portable equivalent of waiting on per-frame fence/semaphore readiness before
    /// final composition/present. The wait is bounded so the CPU does not park for long.
    pub fn reconcile_for_present(
        &mut self,
        primary_waiter: &dyn GpuSubmissionWaiter,
        secondary_waiter: Option<&dyn GpuSubmissionWaiter>,
        transfer_waiter: Option<&dyn GpuSubmissionWaiter>,
    ) -> ComposeBarrierState {
        self.reconcile_impl(
            primary_waiter,
            secondary_waiter,
            transfer_waiter,
            WaitMode::Budgeted(self.config.compose_wait_budget),
        )
    }

    /// Feed Apple UMA telemetry into the adaptive buffer controller (Metal/shared-memory paths).
    pub fn reconcile_apple_uma(
        &mut self,
        telemetry: AdaptiveFrameTelemetry,
    ) -> Option<AdaptiveBufferDecision> {
        let state = self.apple_uma.as_mut()?;
        let decision = state.adaptive.reconcile(telemetry);
        state.last_decision = Some(decision);
        Some(decision)
    }

    /// Try to acquire a CPU read lease on a shared UMA buffer view.
    pub fn try_acquire_apple_uma_cpu_read_view(
        &mut self,
        key: SharedBufferKey,
        view: &wgpu::BufferView,
    ) -> Result<SharedBufferLease, SharedBufferLockError> {
        let state = self
            .apple_uma
            .as_mut()
            .ok_or(SharedBufferLockError::InvalidLease)?;
        state.adaptive.try_acquire_cpu_read_view(key, view)
    }

    /// Try to acquire a CPU write lease on a shared UMA buffer view.
    pub fn try_acquire_apple_uma_cpu_write_view(
        &mut self,
        key: SharedBufferKey,
        view: &wgpu::BufferViewMut,
    ) -> Result<SharedBufferLease, SharedBufferLockError> {
        let state = self
            .apple_uma
            .as_mut()
            .ok_or(SharedBufferLockError::InvalidLease)?;
        state.adaptive.try_acquire_cpu_write_view(key, view)
    }

    /// Try to acquire a GPU lease for a shared UMA buffer byte range.
    pub fn try_acquire_apple_uma_gpu_range(
        &mut self,
        key: SharedBufferKey,
        bytes: u64,
    ) -> Result<SharedBufferLease, SharedBufferLockError> {
        let state = self
            .apple_uma
            .as_mut()
            .ok_or(SharedBufferLockError::InvalidLease)?;
        state.adaptive.try_acquire_gpu_range(key, bytes)
    }

    /// Release a previously acquired shared UMA lease.
    pub fn release_apple_uma_lease(
        &mut self,
        lease: SharedBufferLease,
    ) -> Result<(), SharedBufferLockError> {
        let state = self
            .apple_uma
            .as_mut()
            .ok_or(SharedBufferLockError::InvalidLease)?;
        state.adaptive.release_lease(lease)
    }

    /// Immutable snapshot for diagnostics and telemetry export.
    pub fn snapshot(&self) -> MultiGpuSyncSnapshot {
        MultiGpuSyncSnapshot {
            backend_hint: self.backend_hint,
            shared_placement: self.shared_placement,
            frames_in_flight_limit: self.frames_in_flight_limit,
            tracked_frames: self.pending_frames.len(),
            poll_calls: self.poll_calls,
            wait_calls: self.wait_calls,
            wait_timeouts: self.wait_timeouts,
            invalid_submission_waits: self.invalid_submission_waits,
            completed_frames: self.completed_frames,
            rejected_frames: self.rejected_frames,
            apple_uma_enabled: self.apple_uma.is_some(),
            last_apple_uma_decision: self
                .apple_uma
                .as_ref()
                .and_then(|state| state.last_decision),
        }
    }

    fn reconcile_impl(
        &mut self,
        primary_waiter: &dyn GpuSubmissionWaiter,
        secondary_waiter: Option<&dyn GpuSubmissionWaiter>,
        transfer_waiter: Option<&dyn GpuSubmissionWaiter>,
        wait_mode: WaitMode,
    ) -> ComposeBarrierState {
        let Some(frame) = self.pending_frames.front().cloned() else {
            return ComposeBarrierState {
                backend_hint: self.backend_hint,
                shared_placement: self.shared_placement,
                apple_uma_decision: self.apple_uma.as_ref().and_then(|s| s.last_decision),
                ..ComposeBarrierState::default()
            };
        };

        let now = Instant::now();
        let mut state = ComposeBarrierState {
            frame_id: Some(frame.frame_id),
            backend_hint: self.backend_hint,
            shared_placement: self.shared_placement,
            late_by: now
                .checked_duration_since(frame.registered_at)
                .and_then(|age| age.checked_sub(self.config.target_compose_budget)),
            apple_uma_decision: self.apple_uma.as_ref().and_then(|s| s.last_decision),
            ..ComposeBarrierState::default()
        };

        let mut remaining_wait_budget = match wait_mode {
            WaitMode::NonBlocking => Duration::ZERO,
            WaitMode::Budgeted(budget) => budget,
        };

        let primary_ready = self.resolve_lane_readiness(
            primary_waiter,
            frame.primary_submission,
            frame.require_primary,
            &mut remaining_wait_budget,
            &mut state,
        );
        let secondary_ready = self.resolve_lane_readiness(
            secondary_waiter.unwrap_or(primary_waiter),
            frame.secondary_submission,
            frame.require_secondary,
            &mut remaining_wait_budget,
            &mut state,
        );
        let transfer_ready = self.resolve_lane_readiness(
            transfer_waiter
                .or(secondary_waiter)
                .unwrap_or(primary_waiter),
            frame.transfer_submission,
            frame.require_transfer,
            &mut remaining_wait_budget,
            &mut state,
        );

        state.ready = primary_ready && secondary_ready && transfer_ready && !state.timed_out;
        if state.ready {
            let _ = self.pending_frames.pop_front();
            self.completed_frames = self.completed_frames.saturating_add(1);
        } else if matches!(wait_mode, WaitMode::Budgeted(_)) && remaining_wait_budget.is_zero() {
            state.compose_budget_exhausted = true;
        }

        state
    }

    fn resolve_lane_readiness(
        &mut self,
        waiter: &dyn GpuSubmissionWaiter,
        submission: Option<GpuSubmissionHandle>,
        required: bool,
        remaining_wait_budget: &mut Duration,
        state: &mut ComposeBarrierState,
    ) -> bool {
        if !required {
            return true;
        }

        let Some(submission) = submission else {
            return false;
        };

        // First perform a non-blocking targeted readiness check. This gives us a portable
        // fence-like probe without stalling the CPU.
        self.poll_calls = self.poll_calls.saturating_add(1);
        match waiter.wait_submission(&submission, Some(Duration::ZERO)) {
            GpuSubmissionWaitStatus::Ready => return true,
            GpuSubmissionWaitStatus::Pending | GpuSubmissionWaitStatus::TimedOut => {}
            GpuSubmissionWaitStatus::Invalid => {
                self.invalid_submission_waits = self.invalid_submission_waits.saturating_add(1);
                return false;
            }
        }

        if remaining_wait_budget.is_zero() {
            return false;
        }

        let wait_start = Instant::now();
        self.wait_calls = self.wait_calls.saturating_add(1);
        state.waited = true;
        match waiter.wait_submission(&submission, Some(*remaining_wait_budget)) {
            GpuSubmissionWaitStatus::Ready => {
                let waited = wait_start.elapsed();
                state.wait_time = state.wait_time.saturating_add(waited);
                *remaining_wait_budget = remaining_wait_budget.saturating_sub(waited);
                true
            }
            GpuSubmissionWaitStatus::TimedOut | GpuSubmissionWaitStatus::Pending => {
                let waited = wait_start.elapsed();
                state.wait_time = state.wait_time.saturating_add(waited);
                *remaining_wait_budget = remaining_wait_budget.saturating_sub(waited);
                self.wait_timeouts = self.wait_timeouts.saturating_add(1);
                state.timed_out = true;
                false
            }
            GpuSubmissionWaitStatus::Invalid => {
                let waited = wait_start.elapsed();
                state.wait_time = state.wait_time.saturating_add(waited);
                *remaining_wait_budget = remaining_wait_budget.saturating_sub(waited);
                self.invalid_submission_waits = self.invalid_submission_waits.saturating_add(1);
                false
            }
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum WaitMode {
    NonBlocking,
    Budgeted(Duration),
}

fn detect_sync_backend_hint(
    primary: &GpuAdapterProfile,
    secondary: Option<&GpuAdapterProfile>,
) -> SyncBackendHint {
    let primary_backend = primary.backend;
    let secondary_backend = secondary.map(|gpu| gpu.backend);

    if primary_backend == wgpu::Backend::Metal || secondary_backend == Some(wgpu::Backend::Metal) {
        SyncBackendHint::MetalSharedEventLike
    } else if primary_backend == wgpu::Backend::Vulkan
        || secondary_backend == Some(wgpu::Backend::Vulkan)
    {
        SyncBackendHint::VulkanTimelineFenceLike
    } else {
        SyncBackendHint::PortableWgpu
    }
}

fn detect_shared_placement_policy(
    primary: &GpuAdapterProfile,
    secondary: Option<&GpuAdapterProfile>,
) -> SharedPlacementPolicy {
    let apple_metal =
        is_apple_metal_adapter(primary) || secondary.map(is_apple_metal_adapter).unwrap_or(false);
    if apple_metal {
        return SharedPlacementPolicy::MetalSharedStrict;
    }

    if matches!(primary.memory_topology, MemoryTopology::Unified)
        && secondary
            .map(|gpu| matches!(gpu.memory_topology, MemoryTopology::Unified))
            .unwrap_or(false)
    {
        SharedPlacementPolicy::UnifiedShared
    } else {
        SharedPlacementPolicy::HostMappedBridge
    }
}

fn build_apple_uma_state(
    shared_placement: SharedPlacementPolicy,
    config: MultiGpuFrameSyncConfig,
    primary: &GpuAdapterProfile,
    secondary: Option<&GpuAdapterProfile>,
) -> Option<AppleUmaSyncState> {
    let has_apple =
        is_apple_metal_adapter(primary) || secondary.map(is_apple_metal_adapter).unwrap_or(false);
    if !has_apple {
        return None;
    }

    let adaptive = AdaptiveBuffer::new(AdaptiveBufferConfig::default());
    Some(AppleUmaSyncState {
        adaptive,
        last_decision: None,
        _strict_shared_placement: matches!(
            shared_placement,
            SharedPlacementPolicy::MetalSharedStrict
        ),
        _target_stability: config.apple_target_stability,
        _thermal_floor_fps: config.apple_thermal_floor_fps,
    })
}

fn is_apple_metal_adapter(profile: &GpuAdapterProfile) -> bool {
    profile.backend == wgpu::Backend::Metal
        && (profile.vendor_id == APPLE_VENDOR_ID
            || profile.name.to_ascii_lowercase().contains("apple"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_uses_sub_ms_budget() {
        let cfg = MultiGpuFrameSyncConfig::default();
        assert_eq!(cfg.compose_wait_budget, Duration::from_micros(800));
    }

    #[test]
    fn admission_rejects_when_window_is_full() {
        let dummy_plan = MultiGpuSyncPlan {
            fence_equivalent: gms::SyncEquivalent::QueueCompletionFence,
            semaphore_equivalent: gms::SyncEquivalent::QueueSubmissionTimeline,
            frames_in_flight: 1,
            queue_timeline_stages: 3,
            primary_command_buffers_preallocated: 3,
            secondary_command_buffers_preallocated: 3,
            transfer_command_buffers_preallocated: 2,
            aggressive_integrated_preallocation: false,
            integrated_encoder_pool: 0,
            integrated_ring_segments: 0,
            secondary_budget_ms: 0.5,
            estimated_secondary_ms: 0.0,
            secondary_slack_ms: 0.0,
        };

        let profile = GpuAdapterProfile {
            index: 0,
            name: "Dummy".into(),
            vendor_id: 0,
            device_id: 0,
            backend: wgpu::Backend::Vulkan,
            device_type: wgpu::DeviceType::IntegratedGpu,
            memory_topology: MemoryTopology::Unified,
            compute_unit_kind: gms::ComputeUnitKind::CoreCluster,
            estimated_compute_units: 8,
            unit_grouping: 2,
            unit_perf_score: 125.0,
            thermal_headroom: 0.65,
            compute_unit_source: gms::ComputeUnitEstimateSource::DeviceNameTable,
            compute_unit_probe_note: None,
            arm_shader_core_count: None,
            estimated_vram_mb: 4096,
            estimated_bandwidth_gbps: 50.0,
            supports_mappable_primary_buffers: true,
            limits: gms::hardware::GpuLimitsSummary {
                max_buffer_size: 1 << 20,
                max_storage_buffer_binding_size: 1 << 20,
                max_compute_invocations_per_workgroup: 256,
                max_compute_workgroup_storage_size: 16384,
                max_compute_workgroups_per_dimension: 65535,
            },
            score: 1000,
            score_breakdown: gms::GpuScoreBreakdown::default(),
        };

        let mut sync = MultiGpuFrameSynchronizer::new(
            dummy_plan,
            MultiGpuFrameSyncConfig::default(),
            &profile,
            None,
        );
        assert!(sync.admit_frame(1, false, false).accepted);
        let second = sync.admit_frame(2, true, true);
        assert!(!second.accepted);
        assert!(second.should_spill_secondary);
    }
}
