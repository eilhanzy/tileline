//! MPS <-> MGS bridge (Tileline core).
//!
//! Connects CPU-side MPS execution output to GPU-side MGS planning
//! (TBDR tile assignment and fallback chain) in a lock-free manner.
//!
//! Design goals:
//! - same lock-free MPS -> MGS completion queue as bridge.rs (crossbeam SegQueue)
//! - deterministic plan production via frame sealing
//! - workload-hint accumulator that feeds the MGS tile planner
//! - consistent API surface with `MpsGmsBridge`

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crossbeam::queue::SegQueue;
use mgs::{MgsBridge, MgsBridgePlan, MobileGpuProfile, MpsWorkloadHint};
use mps::{CorePreference, MpsScheduler, NativeTask, SchedulerMetrics, TaskPriority};

// ── Frame / Submission identifiers ──────────────────────────────────────────

/// Bridge-local unique identifier for a submitted MGS pipeline task.
pub type MgsBridgeSubmissionId = u64;

/// Frame identifier used by the MGS bridge and downstream tile planner.
pub type MgsBridgeFrameId = u64;

// ── Task Descriptor ──────────────────────────────────────────────────────────

/// CPU task descriptor for the MPS -> MGS pipeline.
///
/// Each completed CPU task is folded into an `MgsFrameAccumulator`; once the
/// frame is sealed the accumulator is converted to an `MpsWorkloadHint` and
/// forwarded to MGS.
#[derive(Debug, Clone, Copy)]
pub struct MgsBridgeTaskDescriptor {
    /// Frame bucket this task contributes to.
    pub frame_id: MgsBridgeFrameId,
    /// MPS scheduler priority for the CPU-side preprocessing stage.
    pub cpu_priority: TaskPriority,
    /// Preferred CPU core class for the preprocessing stage.
    pub cpu_preference: CorePreference,
    /// Number of objects / draw calls represented by this task.
    pub object_count: u32,
    /// Estimated memory transfer size (KB).
    pub transfer_size_kb: u32,
    /// Render target width (pixels).
    pub target_width: u32,
    /// Render target height (pixels).
    pub target_height: u32,
    /// Maximum latency budget suggested by MPS (ms); 0 = unconstrained.
    pub latency_budget_ms: u32,
}

impl Default for MgsBridgeTaskDescriptor {
    fn default() -> Self {
        Self {
            frame_id: 0,
            cpu_priority: TaskPriority::Normal,
            cpu_preference: CorePreference::Auto,
            object_count: 1,
            transfer_size_kb: 0,
            target_width: 1920,
            target_height: 1080,
            latency_budget_ms: 0,
        }
    }
}

// ── Submission / Receipt ─────────────────────────────────────────────────────

/// Submission payload for MPS in the MGS path.
pub struct MgsBridgeMpsSubmission {
    /// Task descriptor — folded into the frame accumulator on completion.
    pub descriptor: MgsBridgeTaskDescriptor,
    /// CPU preprocessing closure executed by MPS.
    pub task: NativeTask,
}

/// Receipt returned to the caller when a bridge task is submitted to MPS.
#[derive(Debug, Clone, Copy)]
pub struct MgsBridgeSubmitReceipt {
    /// Bridge-local submission ID.
    pub bridge_submission_id: MgsBridgeSubmissionId,
    /// MPS scheduler task ID returned by `mps::MpsScheduler`.
    pub mps_task_id: u64,
}

// ── Frame Plan ───────────────────────────────────────────────────────────────

/// MGS tile plan produced by the bridge for a sealed frame.
#[derive(Debug, Clone)]
pub struct MgsBridgeFramePlan {
    /// Frame ID associated with this plan.
    pub frame_id: MgsBridgeFrameId,
    /// Timestamp when the bridge published the plan.
    pub created_at: Instant,
    /// Number of CPU tasks that completed and were folded into this frame.
    pub cpu_completed_tasks: u32,
    /// Aggregate CPU execution time measured on MPS workers (ms).
    pub cpu_execution_time_ms: f64,
    /// MGS tile plan and fallback level produced for this frame.
    pub plan: MgsBridgePlan,
    /// Combined workload hint reflected in the plan (for diagnostics / telemetry).
    pub workload_hint: MpsWorkloadHint,
}

// ── Bridge Metrics ───────────────────────────────────────────────────────────

/// Metrics snapshot for `MpsMgsBridge`.
#[derive(Debug, Clone)]
pub struct MpsMgsBridgeMetrics {
    /// Total submissions accepted by the bridge.
    pub bridge_submitted: u64,
    /// Total completion records pushed into the lock-free completion queue.
    pub cpu_completions_queued: u64,
    /// Total completion records drained from the lock-free completion queue.
    pub cpu_completions_drained: u64,
    /// Total frame plans published to the GPU-plan queue.
    pub frame_plans_published: u64,
    /// Number of sealed frames waiting for sufficient CPU completions.
    pub sealed_frames_pending: usize,
    /// Number of frame accumulators currently tracked by the bridge.
    pub accumulators_pending: usize,
    /// Approximate number of published frame plans waiting for consumption.
    pub gpu_plan_queue_depth: usize,
    /// Snapshot of MPS scheduler metrics.
    pub mps: SchedulerMetrics,
}

// ── Bridge Configuration ─────────────────────────────────────────────────────

/// Configuration for `MpsMgsBridge`.
#[derive(Debug, Clone)]
pub struct MpsMgsBridgeConfig {
    /// Max completion records drained from the lock-free queue per `pump` call.
    pub drain_batch_limit: usize,
    /// Default render target width used when a task descriptor reports 0.
    pub default_target_width: u32,
    /// Default render target height used when a task descriptor reports 0.
    pub default_target_height: u32,
}

impl Default for MpsMgsBridgeConfig {
    fn default() -> Self {
        Self {
            drain_batch_limit: 4096,
            default_target_width: 1920,
            default_target_height: 1080,
        }
    }
}

// ── Internal Types ───────────────────────────────────────────────────────────

#[derive(Debug)]
struct MgsCpuCompletionRecord {
    submission_id: MgsBridgeSubmissionId,
    descriptor: MgsBridgeTaskDescriptor,
    cpu_elapsed: Duration,
    completed_at: Instant,
}

#[derive(Debug)]
struct MgsFrameAccumulator {
    _first_completion_at: Instant,
    completed_tasks: u32,
    cpu_execution_ns: u64,
    total_object_count: u32,
    total_transfer_kb: u32,
    // Take the largest resolution reported across tasks in the same frame.
    max_target_width: u32,
    max_target_height: u32,
    // Take the smallest non-zero latency budget (most constrained).
    min_latency_budget_ms: u32,
}

impl MgsFrameAccumulator {
    fn new(now: Instant, default_width: u32, default_height: u32) -> Self {
        Self {
            _first_completion_at: now,
            completed_tasks: 0,
            cpu_execution_ns: 0,
            total_object_count: 0,
            total_transfer_kb: 0,
            max_target_width: default_width,
            max_target_height: default_height,
            min_latency_budget_ms: 0,
        }
    }

    fn apply(&mut self, record: &MgsCpuCompletionRecord) {
        self.completed_tasks = self.completed_tasks.saturating_add(1);
        self.cpu_execution_ns = self
            .cpu_execution_ns
            .saturating_add(record.cpu_elapsed.as_nanos() as u64);
        self.total_object_count = self
            .total_object_count
            .saturating_add(record.descriptor.object_count);
        self.total_transfer_kb = self
            .total_transfer_kb
            .saturating_add(record.descriptor.transfer_size_kb);

        if record.descriptor.target_width > 0 {
            self.max_target_width =
                self.max_target_width.max(record.descriptor.target_width);
        }
        if record.descriptor.target_height > 0 {
            self.max_target_height =
                self.max_target_height.max(record.descriptor.target_height);
        }

        let budget = record.descriptor.latency_budget_ms;
        if budget > 0 {
            self.min_latency_budget_ms = if self.min_latency_budget_ms == 0 {
                budget
            } else {
                self.min_latency_budget_ms.min(budget)
            };
        }
    }

    fn is_empty(&self) -> bool {
        self.total_object_count == 0
    }

    fn to_workload_hint(&self) -> MpsWorkloadHint {
        MpsWorkloadHint {
            transfer_size_kb: self.total_transfer_kb.max(1),
            object_count: self.total_object_count.max(1),
            target_width: self.max_target_width,
            target_height: self.max_target_height,
            latency_budget_ms: self.min_latency_budget_ms,
        }
    }
}

// ── Main Struct ──────────────────────────────────────────────────────────────

/// Bridge connecting the MPS CPU scheduler to the MGS mobile GPU planner.
///
/// Usage pattern:
/// 1. Submit CPU preprocessing closures with [`submit_mps_task`](Self::submit_mps_task).
/// 2. Seal a frame with [`seal_frame`](Self::seal_frame) when no more CPU tasks are expected.
/// 3. Periodically call [`pump`](Self::pump) to drain completions and publish MGS plans.
/// 4. Consume [`MgsBridgeFramePlan`] objects from [`try_pop_frame_plan`](Self::try_pop_frame_plan).
pub struct MpsMgsBridge {
    config: MpsMgsBridgeConfig,
    mps: MpsScheduler,
    mgs_bridge: MgsBridge,
    cpu_completion_queue: Arc<SegQueue<MgsCpuCompletionRecord>>,
    gpu_plan_queue: Arc<SegQueue<MgsBridgeFramePlan>>,
    pending_frames: BTreeMap<MgsBridgeFrameId, MgsFrameAccumulator>,
    sealed_frames: BTreeSet<MgsBridgeFrameId>,
    bridge_submitted: AtomicU64,
    cpu_completions_queued: Arc<AtomicU64>,
    cpu_completions_drained: AtomicU64,
    frame_plans_published: AtomicU64,
    next_bridge_submission_id: AtomicU64,
}

impl MpsMgsBridge {
    /// Create a bridge with a newly constructed MPS scheduler and the given GPU profile.
    pub fn new(profile: MobileGpuProfile, config: MpsMgsBridgeConfig) -> Self {
        Self::with_scheduler(MpsScheduler::new(), profile, config)
    }

    /// Create a bridge from a caller-provided MPS scheduler.
    pub fn with_scheduler(
        mps: MpsScheduler,
        profile: MobileGpuProfile,
        config: MpsMgsBridgeConfig,
    ) -> Self {
        let mgs_bridge = MgsBridge::new(profile);
        Self {
            config,
            mps,
            mgs_bridge,
            cpu_completion_queue: Arc::new(SegQueue::new()),
            gpu_plan_queue: Arc::new(SegQueue::new()),
            pending_frames: BTreeMap::new(),
            sealed_frames: BTreeSet::new(),
            bridge_submitted: AtomicU64::new(0),
            cpu_completions_queued: Arc::new(AtomicU64::new(0)),
            cpu_completions_drained: AtomicU64::new(0),
            frame_plans_published: AtomicU64::new(0),
            next_bridge_submission_id: AtomicU64::new(1),
        }
    }

    /// Access the underlying MPS scheduler.
    pub fn mps(&self) -> &MpsScheduler {
        &self.mps
    }

    /// Access the underlying MGS bridge.
    pub fn mgs_bridge(&self) -> &MgsBridge {
        &self.mgs_bridge
    }

    /// Submit a CPU task to MPS and register a lock-free completion record for MGS planning.
    pub fn submit_mps_task(
        &self,
        submission: MgsBridgeMpsSubmission,
    ) -> MgsBridgeSubmitReceipt {
        let MgsBridgeMpsSubmission { descriptor, task } = submission;
        let bridge_submission_id = self
            .next_bridge_submission_id
            .fetch_add(1, Ordering::Relaxed);
        let priority = descriptor.cpu_priority;
        let preference = descriptor.cpu_preference;

        let mps_task_id = self.mps.submit_native_boxed(
            priority,
            preference,
            wrap_completion_task(
                bridge_submission_id,
                descriptor,
                task,
                Arc::clone(&self.cpu_completion_queue),
                Arc::clone(&self.cpu_completions_queued),
            ),
        );

        self.bridge_submitted.fetch_add(1, Ordering::Relaxed);
        MgsBridgeSubmitReceipt {
            bridge_submission_id,
            mps_task_id,
        }
    }

    /// Enqueue a synthetic completion directly (useful when CPU work is executed outside MPS).
    pub fn enqueue_completed_descriptor(
        &self,
        descriptor: MgsBridgeTaskDescriptor,
    ) -> MgsBridgeSubmissionId {
        let id = self
            .next_bridge_submission_id
            .fetch_add(1, Ordering::Relaxed);
        self.cpu_completion_queue.push(MgsCpuCompletionRecord {
            submission_id: id,
            descriptor,
            cpu_elapsed: Duration::ZERO,
            completed_at: Instant::now(),
        });
        self.cpu_completions_queued.fetch_add(1, Ordering::Relaxed);
        id
    }

    /// Mark a frame as complete from the CPU producer perspective.
    ///
    /// The bridge will only publish a GPU plan for frames that have been sealed
    /// and have at least one completion record.
    pub fn seal_frame(&mut self, frame_id: MgsBridgeFrameId) {
        self.sealed_frames.insert(frame_id);
    }

    /// Drain MPS completion events into frame accumulators without blocking.
    pub fn drain_cpu_completions(&mut self, max_events: usize) -> usize {
        let limit = max_events.min(self.config.drain_batch_limit).max(1);
        let mut drained = 0usize;

        while drained < limit {
            let Some(record) = self.cpu_completion_queue.pop() else {
                break;
            };
            let _ = record.submission_id;
            let now = record.completed_at;
            let entry = self
                .pending_frames
                .entry(record.descriptor.frame_id)
                .or_insert_with(|| {
                    MgsFrameAccumulator::new(
                        now,
                        self.config.default_target_width,
                        self.config.default_target_height,
                    )
                });
            entry.apply(&record);
            drained += 1;
        }

        if drained > 0 {
            self.cpu_completions_drained
                .fetch_add(drained as u64, Ordering::Relaxed);
        }
        drained
    }

    /// Drain completions and publish up to `max_frames` sealed frame plans.
    pub fn pump(&mut self, max_frames: usize) -> usize {
        self.drain_cpu_completions(self.config.drain_batch_limit);
        let mut published = 0usize;
        let max_frames = max_frames.max(1);

        while published < max_frames {
            let Some(frame_id) = self.next_plannable_frame_id() else {
                break;
            };
            let Some(plan) = self.build_frame_plan(frame_id) else {
                break;
            };
            self.gpu_plan_queue.push(plan);
            self.frame_plans_published.fetch_add(1, Ordering::Relaxed);
            published += 1;
        }

        published
    }

    /// Try to pop a previously published MGS frame plan.
    pub fn try_pop_frame_plan(&self) -> Option<MgsBridgeFramePlan> {
        self.gpu_plan_queue.pop()
    }

    /// Snapshot metrics across the bridge and MPS scheduler.
    pub fn metrics(&self) -> MpsMgsBridgeMetrics {
        MpsMgsBridgeMetrics {
            bridge_submitted: self.bridge_submitted.load(Ordering::Relaxed),
            cpu_completions_queued: self.cpu_completions_queued.load(Ordering::Relaxed),
            cpu_completions_drained: self.cpu_completions_drained.load(Ordering::Relaxed),
            frame_plans_published: self.frame_plans_published.load(Ordering::Relaxed),
            sealed_frames_pending: self.sealed_frames.len(),
            accumulators_pending: self.pending_frames.len(),
            gpu_plan_queue_depth: self.gpu_plan_queue.len(),
            mps: self.mps.metrics(),
        }
    }

    // ── Internal helpers ─────────────────────────────────────────────────────

    fn next_plannable_frame_id(&self) -> Option<MgsBridgeFrameId> {
        self.sealed_frames
            .iter()
            .copied()
            .find(|fid| self.pending_frames.contains_key(fid))
    }

    fn build_frame_plan(&mut self, frame_id: MgsBridgeFrameId) -> Option<MgsBridgeFramePlan> {
        let accumulator = self.pending_frames.remove(&frame_id)?;
        self.sealed_frames.remove(&frame_id);

        if accumulator.is_empty() {
            return None;
        }

        let hint = accumulator.to_workload_hint();
        let plan = self.mgs_bridge.translate(hint);

        Some(MgsBridgeFramePlan {
            frame_id,
            created_at: Instant::now(),
            cpu_completed_tasks: accumulator.completed_tasks,
            cpu_execution_time_ms: accumulator.cpu_execution_ns as f64 / 1_000_000.0,
            plan,
            workload_hint: hint,
        })
    }
}

// ── Completion wrapper ───────────────────────────────────────────────────────

fn wrap_completion_task(
    bridge_submission_id: MgsBridgeSubmissionId,
    descriptor: MgsBridgeTaskDescriptor,
    task: NativeTask,
    completion_queue: Arc<SegQueue<MgsCpuCompletionRecord>>,
    cpu_completions_queued: Arc<AtomicU64>,
) -> NativeTask {
    Box::new(move || {
        let started = Instant::now();
        task();
        let completed_at = Instant::now();
        let cpu_elapsed = completed_at.saturating_duration_since(started);
        completion_queue.push(MgsCpuCompletionRecord {
            submission_id: bridge_submission_id,
            descriptor,
            cpu_elapsed,
            completed_at,
        });
        cpu_completions_queued.fetch_add(1, Ordering::Relaxed);
    })
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_bridge(gpu_name: &str) -> MpsMgsBridge {
        let profile = MobileGpuProfile::detect(gpu_name);
        MpsMgsBridge::new(profile, MpsMgsBridgeConfig::default())
    }

    #[test]
    fn pump_produces_plan_after_seal() {
        let mut bridge = make_bridge("Adreno 750");

        bridge.enqueue_completed_descriptor(MgsBridgeTaskDescriptor {
            frame_id: 1,
            object_count: 64,
            transfer_size_kb: 256,
            ..Default::default()
        });
        bridge.seal_frame(1);

        let published = bridge.pump(1);
        assert_eq!(published, 1);

        let plan = bridge.try_pop_frame_plan().expect("expected a frame plan");
        assert_eq!(plan.frame_id, 1);
        assert_eq!(plan.cpu_completed_tasks, 1);
        assert!(!plan.plan.tile_plan.assignments.is_empty());
    }

    #[test]
    fn unsealed_frame_is_not_published() {
        let mut bridge = make_bridge("Mali-G78 MC24");

        bridge.enqueue_completed_descriptor(MgsBridgeTaskDescriptor {
            frame_id: 42,
            object_count: 16,
            transfer_size_kb: 64,
            ..Default::default()
        });
        // Not sealed — must not be published.
        let published = bridge.pump(1);
        assert_eq!(published, 0);
    }

    #[test]
    fn multiple_tasks_accumulate_into_single_plan() {
        let mut bridge = make_bridge("Adreno 750");

        for _ in 0..4 {
            bridge.enqueue_completed_descriptor(MgsBridgeTaskDescriptor {
                frame_id: 7,
                object_count: 32,
                transfer_size_kb: 128,
                ..Default::default()
            });
        }
        bridge.seal_frame(7);
        bridge.pump(1);

        let plan = bridge.try_pop_frame_plan().expect("expected a frame plan");
        assert_eq!(plan.cpu_completed_tasks, 4);
        // 4 * 32 = 128 objects must be accumulated.
        assert_eq!(plan.workload_hint.object_count, 128);
    }

    #[test]
    fn metrics_reflect_activity() {
        let mut bridge = make_bridge("Adreno 750");

        bridge.enqueue_completed_descriptor(MgsBridgeTaskDescriptor {
            frame_id: 1,
            object_count: 8,
            transfer_size_kb: 32,
            ..Default::default()
        });
        bridge.seal_frame(1);
        bridge.pump(1);

        let m = bridge.metrics();
        assert_eq!(m.cpu_completions_queued, 1);
        assert_eq!(m.cpu_completions_drained, 1);
        assert_eq!(m.frame_plans_published, 1);
    }
}
