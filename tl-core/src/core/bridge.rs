//! MPS <-> GMS bridge for Tileline core.
//!
//! This module connects CPU-side MPS execution (task production/preprocessing) to GPU-side GMS
//! planning (workgroup allocation and multi-GPU lane splitting) without blocking the CPU.
//!
//! Design goals:
//! - lock-free MPS -> GMS completion transport (`crossbeam::queue::SegQueue`)
//! - explicit heavy-vs-latency task classification for asymmetric multi-GPU dispatch
//! - frame sealing so planning can happen deterministically without partial-frame stalls
//! - Apple UMA compatibility via the shared `graphics::multigpu::sync` adaptive path

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crossbeam::queue::SegQueue;
use gms::{
    AdaptiveBufferDecision, AdaptiveFrameTelemetry, DispatchPlan, GmsDispatcher, GpuInventory,
    MultiGpuDispatchPlan, MultiGpuDispatcher, MultiGpuWorkloadRequest, TaskClass, WorkloadRequest,
};
use mps::{CorePreference, MpsScheduler, NativeTask, SchedulerMetrics, TaskPriority};

use crate::graphics::multigpu::sync::{
    ComposeBarrierState, GpuQueueLane, GpuSubmissionHandle, GpuSubmissionWaiter,
    MultiGpuFrameSyncConfig, MultiGpuFrameSynchronizer, MultiGpuSyncSnapshot,
};

/// Bridge-local unique identifier for a submitted CPU->GPU pipeline task.
pub type BridgeSubmissionId = u64;

/// Frame identifier used by the bridge and downstream GPU planners.
pub type BridgeFrameId = u64;

/// GPU-oriented task taxonomy used by the bridge.
///
/// Variants intentionally align with GMS workload classes and the engine domains described in the
/// architecture request (TractionVF physics, large particle sets, sampled meshes, UI/Post-FX).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BridgeGpuTaskKind {
    /// TractionVF or similarly heavy physics buffer updates.
    TractionVfPhysics,
    /// Large-scale particle simulation/update chunks.
    MassiveParticles,
    /// Sampled mesh / sampled processing work (texture-heavy compute).
    SampledMesh,
    /// 3D object updates (transforms, culling metadata, skinning metadata packing).
    ObjectUpdate,
    /// AI/ML lane (inference, gameplay model evaluation, utility kernels).
    AiMl,
    /// Latency-sensitive UI composition work.
    Ui,
    /// Latency-sensitive post-processing/composition work.
    PostFx,
    /// Direct bridge to a specific GMS task class.
    Custom(TaskClass),
}

/// Optional routing override before GMS's own multi-GPU asymmetry planner runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BridgeTaskRouting {
    /// Use the default mapping from [`BridgeGpuTaskKind`] to GMS task classes.
    Auto,
    /// Bias the task into a heavy/throughput lane (primary GPU preferred by planner).
    ForcePrimaryHeavy,
    /// Bias the task into a latency lane (secondary GPU preferred by planner when available).
    ForceSecondaryLatency,
}

/// CPU+GPU metadata for a bridge task.
///
/// The CPU work runs on MPS, and on completion the descriptor is converted into GMS workload units.
#[derive(Debug, Clone, Copy)]
pub struct BridgeTaskDescriptor {
    /// Frame bucket this task contributes to.
    pub frame_id: BridgeFrameId,
    /// GPU-oriented task classification used for GMS workload mapping.
    pub gpu_kind: BridgeGpuTaskKind,
    /// Optional routing override before GMS planner weighting.
    pub routing: BridgeTaskRouting,
    /// MPS scheduler priority for the CPU-side preprocessing stage.
    pub cpu_priority: TaskPriority,
    /// Preferred CPU core class for the preprocessing stage.
    pub cpu_preference: CorePreference,
    /// Number of logical jobs/chunks contributed to the target GMS task class.
    pub job_count: u32,
    /// Total bytes of payload touched/produced by this task (used for zero-copy planning).
    pub payload_bytes: u64,
    /// Optional processed texture bytes that will cross adapters before present.
    pub processed_texture_bytes: u64,
    /// Optional per-task workgroup size hint (frame-level planner uses the max seen per frame).
    pub base_workgroup_size_hint: Option<u32>,
}

impl Default for BridgeTaskDescriptor {
    fn default() -> Self {
        Self {
            frame_id: 0,
            gpu_kind: BridgeGpuTaskKind::ObjectUpdate,
            routing: BridgeTaskRouting::Auto,
            cpu_priority: TaskPriority::Normal,
            cpu_preference: CorePreference::Auto,
            job_count: 1,
            payload_bytes: 0,
            processed_texture_bytes: 0,
            base_workgroup_size_hint: None,
        }
    }
}

/// Submission payload for MPS. The closure executes on the CPU and the bridge receives a
/// completion record through a lock-free queue.
pub struct BridgeMpsSubmission {
    /// Data-oriented descriptor folded into a frame-level GMS workload request on completion.
    pub descriptor: BridgeTaskDescriptor,
    /// CPU preprocessing closure executed by MPS.
    pub task: NativeTask,
}

/// Receipt returned to the caller when a bridge task is submitted to MPS.
#[derive(Debug, Clone, Copy)]
pub struct BridgeSubmitReceipt {
    /// Bridge-local submission ID used for bridge-side diagnostics.
    pub bridge_submission_id: BridgeSubmissionId,
    /// MPS scheduler task ID returned by `mps::MpsScheduler`.
    pub mps_task_id: u64,
}

/// Published frame plan produced by the bridge after draining MPS completions.
#[derive(Debug, Clone)]
pub struct BridgeFramePlan {
    /// Frame ID associated with this published plan.
    pub frame_id: BridgeFrameId,
    /// Timestamp when the bridge published the plan.
    pub created_at: Instant,
    /// Number of CPU tasks that completed and were folded into this frame.
    pub cpu_completed_tasks: u32,
    /// Optional aggregate CPU execution time (measured on MPS workers) for those tasks.
    pub cpu_execution_time_ms: f64,
    /// GMS single-GPU fallback/diagnostic plan.
    pub single_gpu_plan: DispatchPlan,
    /// GMS explicit multi-GPU plan (may still collapse to primary-only if no helper GPU exists).
    pub multi_gpu_plan: MultiGpuDispatchPlan,
    /// Workload request produced from bridge descriptors (data-oriented mapping).
    pub workload_request: MultiGpuWorkloadRequest,
    /// Result of frame admission into the multi-GPU synchronizer window.
    pub sync_admitted: bool,
    /// Whether the synchronizer suggested spilling latency work back to primary due to pressure.
    pub sync_spill_secondary_hint: bool,
    /// Secondary scaling hint from the synchronizer (1.0 = full helper lane budget).
    pub sync_secondary_scale_hint: f64,
    /// `true` if any assigned adapter strongly prefers mapped/zero-copy uploads.
    pub prefer_map_write_uploads: bool,
    /// Latest Apple UMA adaptive decision (if available).
    pub apple_uma_decision: Option<AdaptiveBufferDecision>,
}

/// Bridge metrics snapshot.
#[derive(Debug, Clone)]
pub struct MpsGmsBridgeMetrics {
    /// Total MPS-bound submissions accepted by the bridge.
    pub bridge_submitted: u64,
    /// Total CPU completion records pushed into the lock-free completion queue.
    pub cpu_completions_queued: u64,
    /// Total CPU completion records drained from the lock-free completion queue.
    pub cpu_completions_drained: u64,
    /// Total frame plans published to the GPU-plan queue.
    pub frame_plans_published: u64,
    /// Number of sealed frames still waiting for sufficient CPU completions.
    pub sealed_frames_pending: usize,
    /// Number of frame accumulators currently tracked by the bridge.
    pub accumulators_pending: usize,
    /// Approximate number of published frame plans waiting for runtime consumption.
    pub gpu_plan_queue_depth: usize,
    /// Number of frames rejected by sync admission due to in-flight pressure.
    pub sync_rejections: u64,
    /// Snapshot of MPS scheduler metrics.
    pub mps: SchedulerMetrics,
    /// Optional multi-GPU sync snapshot when explicit sync is active.
    pub sync: Option<MultiGpuSyncSnapshot>,
}

/// Bridge configuration.
#[derive(Debug, Clone, Copy)]
pub struct MpsGmsBridgeConfig {
    /// Target frame budget passed into GMS multi-GPU planning.
    pub target_frame_budget_ms: f64,
    /// Default base workgroup size used when no task-specific hint is provided.
    pub default_base_workgroup_size: u32,
    /// Max completions drained from the lock-free queue per pump call.
    pub drain_batch_limit: usize,
    /// Synchronizer configuration for explicit multi-GPU compose barriers.
    pub sync: MultiGpuFrameSyncConfig,
}

impl Default for MpsGmsBridgeConfig {
    fn default() -> Self {
        Self {
            target_frame_budget_ms: 0.80,
            default_base_workgroup_size: 64,
            drain_batch_limit: 4096,
            sync: MultiGpuFrameSyncConfig::default(),
        }
    }
}

#[derive(Debug, Clone)]
struct CpuCompletionRecord {
    submission_id: BridgeSubmissionId,
    descriptor: BridgeTaskDescriptor,
    cpu_elapsed: Duration,
    completed_at: Instant,
}

#[derive(Debug, Clone)]
struct FrameAccumulator {
    _first_completion_at: Instant,
    last_completion_at: Instant,
    completed_tasks: u32,
    cpu_execution_ns: u64,
    sampled_processing_jobs: u32,
    object_updates: u32,
    physics_jobs: u32,
    ai_ml_jobs: u32,
    ui_jobs: u32,
    post_fx_jobs: u32,
    sampled_bytes: u64,
    object_bytes: u64,
    physics_bytes: u64,
    ai_ml_bytes: u64,
    ui_bytes: u64,
    post_fx_bytes: u64,
    processed_texture_bytes_per_frame: u64,
    base_workgroup_size: u32,
}

impl FrameAccumulator {
    fn new(now: Instant, base_workgroup_size: u32) -> Self {
        Self {
            _first_completion_at: now,
            last_completion_at: now,
            completed_tasks: 0,
            cpu_execution_ns: 0,
            sampled_processing_jobs: 0,
            object_updates: 0,
            physics_jobs: 0,
            ai_ml_jobs: 0,
            ui_jobs: 0,
            post_fx_jobs: 0,
            sampled_bytes: 0,
            object_bytes: 0,
            physics_bytes: 0,
            ai_ml_bytes: 0,
            ui_bytes: 0,
            post_fx_bytes: 0,
            processed_texture_bytes_per_frame: 0,
            base_workgroup_size,
        }
    }

    fn apply_completion(&mut self, record: &CpuCompletionRecord) {
        self.last_completion_at = record.completed_at;
        self.completed_tasks = self.completed_tasks.saturating_add(1);
        self.cpu_execution_ns = self
            .cpu_execution_ns
            .saturating_add(record.cpu_elapsed.as_nanos() as u64);
        self.processed_texture_bytes_per_frame = self
            .processed_texture_bytes_per_frame
            .saturating_add(record.descriptor.processed_texture_bytes);
        if let Some(hint) = record.descriptor.base_workgroup_size_hint {
            self.base_workgroup_size = self.base_workgroup_size.max(hint);
        }

        let jobs = record.descriptor.job_count.max(1);
        let bytes = record.descriptor.payload_bytes;
        match effective_task_class(record.descriptor) {
            TaskClass::SampledProcessing => {
                self.sampled_processing_jobs = self.sampled_processing_jobs.saturating_add(jobs);
                self.sampled_bytes = self.sampled_bytes.saturating_add(bytes);
            }
            TaskClass::ObjectUpdate => {
                self.object_updates = self.object_updates.saturating_add(jobs);
                self.object_bytes = self.object_bytes.saturating_add(bytes);
            }
            TaskClass::Physics => {
                self.physics_jobs = self.physics_jobs.saturating_add(jobs);
                self.physics_bytes = self.physics_bytes.saturating_add(bytes);
            }
            TaskClass::AiMl => {
                self.ai_ml_jobs = self.ai_ml_jobs.saturating_add(jobs);
                self.ai_ml_bytes = self.ai_ml_bytes.saturating_add(bytes);
            }
            TaskClass::Ui => {
                self.ui_jobs = self.ui_jobs.saturating_add(jobs);
                self.ui_bytes = self.ui_bytes.saturating_add(bytes);
            }
            TaskClass::PostFx => {
                self.post_fx_jobs = self.post_fx_jobs.saturating_add(jobs);
                self.post_fx_bytes = self.post_fx_bytes.saturating_add(bytes);
            }
        }
    }

    fn is_empty(&self) -> bool {
        self.sampled_processing_jobs == 0
            && self.object_updates == 0
            && self.physics_jobs == 0
            && self.ai_ml_jobs == 0
            && self.ui_jobs == 0
            && self.post_fx_jobs == 0
    }

    fn to_single_gpu_request(&self) -> WorkloadRequest {
        let object_jobs = self.object_updates.saturating_add(self.ai_ml_jobs / 2);
        let physics_jobs = self.physics_jobs.saturating_add(self.ai_ml_jobs / 2);
        WorkloadRequest {
            object_updates: object_jobs,
            physics_jobs,
            bytes_per_object: average_bytes_per_job(
                self.object_bytes.saturating_add(self.ai_ml_bytes / 2),
                object_jobs,
                256,
            ),
            bytes_per_physics_job: average_bytes_per_job(
                self.physics_bytes.saturating_add(self.ai_ml_bytes / 2),
                physics_jobs,
                1024,
            ),
            base_workgroup_size: self.base_workgroup_size,
        }
    }

    fn to_multi_gpu_request(&self, target_frame_budget_ms: f64) -> MultiGpuWorkloadRequest {
        MultiGpuWorkloadRequest {
            sampled_processing_jobs: self.sampled_processing_jobs,
            object_updates: self.object_updates,
            physics_jobs: self.physics_jobs,
            ai_ml_jobs: self.ai_ml_jobs,
            ui_jobs: self.ui_jobs,
            post_fx_jobs: self.post_fx_jobs,
            bytes_per_sampled_job: average_bytes_per_job(
                self.sampled_bytes,
                self.sampled_processing_jobs,
                4096,
            ),
            bytes_per_object: average_bytes_per_job(self.object_bytes, self.object_updates, 256),
            bytes_per_physics_job: average_bytes_per_job(
                self.physics_bytes,
                self.physics_jobs,
                1024,
            ),
            bytes_per_ai_ml_job: average_bytes_per_job(self.ai_ml_bytes, self.ai_ml_jobs, 2048),
            bytes_per_ui_job: average_bytes_per_job(self.ui_bytes, self.ui_jobs, 512),
            bytes_per_post_fx_job: average_bytes_per_job(
                self.post_fx_bytes,
                self.post_fx_jobs,
                1024,
            ),
            processed_texture_bytes_per_frame: self
                .processed_texture_bytes_per_frame
                .max(512 * 1024),
            base_workgroup_size: self.base_workgroup_size,
            target_frame_budget_ms,
        }
    }
}

/// Main CPU<->GPU bridge orchestrator.
///
/// Usage pattern:
/// 1. Submit CPU preprocessing closures with [`submit_mps_task`](Self::submit_mps_task).
/// 2. Seal a frame with [`seal_frame`](Self::seal_frame) when no more CPU tasks are expected.
/// 3. Periodically call [`pump`](Self::pump) to drain completions and publish GMS plans.
/// 4. Consume [`BridgeFramePlan`] objects from [`try_pop_frame_plan`](Self::try_pop_frame_plan).
pub struct MpsGmsBridge {
    config: MpsGmsBridgeConfig,
    mps: MpsScheduler,
    gpu_inventory: GpuInventory,
    gms_dispatcher: GmsDispatcher,
    multi_gpu_dispatcher: MultiGpuDispatcher,
    cpu_completion_queue: Arc<SegQueue<CpuCompletionRecord>>,
    gpu_plan_queue: Arc<SegQueue<BridgeFramePlan>>,
    pending_frames: BTreeMap<BridgeFrameId, FrameAccumulator>,
    sealed_frames: BTreeSet<BridgeFrameId>,
    sync: Option<MultiGpuFrameSynchronizer>,
    bridge_submitted: AtomicU64,
    cpu_completions_queued: Arc<AtomicU64>,
    cpu_completions_drained: AtomicU64,
    frame_plans_published: AtomicU64,
    sync_rejections: AtomicU64,
    next_bridge_submission_id: AtomicU64,
}

impl MpsGmsBridge {
    /// Create a bridge with a newly detected MPS scheduler and discovered GMS inventory.
    pub fn new(config: MpsGmsBridgeConfig) -> Self {
        Self::with_scheduler(MpsScheduler::new(), config)
    }

    /// Create a bridge from a caller-provided MPS scheduler.
    pub fn with_scheduler(mps: MpsScheduler, config: MpsGmsBridgeConfig) -> Self {
        let inventory = GpuInventory::discover();
        let gms_dispatcher = GmsDispatcher::new(inventory.clone());
        let multi_gpu_dispatcher = MultiGpuDispatcher::new(inventory.clone());
        Self {
            config,
            mps,
            gpu_inventory: inventory,
            gms_dispatcher,
            multi_gpu_dispatcher,
            cpu_completion_queue: Arc::new(SegQueue::new()),
            gpu_plan_queue: Arc::new(SegQueue::new()),
            pending_frames: BTreeMap::new(),
            sealed_frames: BTreeSet::new(),
            sync: None,
            bridge_submitted: AtomicU64::new(0),
            cpu_completions_queued: Arc::new(AtomicU64::new(0)),
            cpu_completions_drained: AtomicU64::new(0),
            frame_plans_published: AtomicU64::new(0),
            sync_rejections: AtomicU64::new(0),
            next_bridge_submission_id: AtomicU64::new(1),
        }
    }

    /// Access the underlying MPS scheduler.
    pub fn mps(&self) -> &MpsScheduler {
        &self.mps
    }

    /// Access the discovered GPU inventory used by GMS planners.
    pub fn gpu_inventory(&self) -> &GpuInventory {
        &self.gpu_inventory
    }

    /// Access the explicit multi-GPU synchronizer if it has been initialized.
    pub fn sync(&self) -> Option<&MultiGpuFrameSynchronizer> {
        self.sync.as_ref()
    }

    /// Mutable access to the explicit multi-GPU synchronizer.
    pub fn sync_mut(&mut self) -> Option<&mut MultiGpuFrameSynchronizer> {
        self.sync.as_mut()
    }

    /// Submit a CPU task to MPS and register a lock-free completion record for GPU planning.
    pub fn submit_mps_task(&self, submission: BridgeMpsSubmission) -> BridgeSubmitReceipt {
        let BridgeMpsSubmission { descriptor, task } = submission;
        let bridge_submission_id = self
            .next_bridge_submission_id
            .fetch_add(1, Ordering::Relaxed);
        let priority = descriptor.cpu_priority;
        let preference = descriptor.cpu_preference;

        let mps_task_id = self.mps.submit_native_boxed(
            priority,
            preference,
            wrap_bridge_completion_task(
                bridge_submission_id,
                descriptor,
                task,
                Arc::clone(&self.cpu_completion_queue),
                Arc::clone(&self.cpu_completions_queued),
            ),
        );

        self.bridge_submitted.fetch_add(1, Ordering::Relaxed);
        BridgeSubmitReceipt {
            bridge_submission_id,
            mps_task_id,
        }
    }

    /// Enqueue a synthetic completion directly (useful when CPU work is executed outside MPS).
    pub fn enqueue_completed_descriptor(
        &self,
        descriptor: BridgeTaskDescriptor,
    ) -> BridgeSubmissionId {
        let bridge_submission_id = self
            .next_bridge_submission_id
            .fetch_add(1, Ordering::Relaxed);
        self.cpu_completion_queue.push(CpuCompletionRecord {
            submission_id: bridge_submission_id,
            descriptor,
            cpu_elapsed: Duration::ZERO,
            completed_at: Instant::now(),
        });
        self.cpu_completions_queued.fetch_add(1, Ordering::Relaxed);
        bridge_submission_id
    }

    /// Mark a frame as complete from the CPU producer perspective.
    ///
    /// The bridge will only publish a GPU plan for frames that have been sealed.
    pub fn seal_frame(&mut self, frame_id: BridgeFrameId) {
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
            let entry = self
                .pending_frames
                .entry(record.descriptor.frame_id)
                .or_insert_with(|| {
                    FrameAccumulator::new(
                        record.completed_at,
                        self.config.default_base_workgroup_size,
                    )
                });
            entry.apply_completion(&record);
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

    /// Try to pop a previously published GPU frame plan.
    pub fn try_pop_frame_plan(&self) -> Option<BridgeFramePlan> {
        self.gpu_plan_queue.pop()
    }

    /// Feed runtime telemetry back into the Apple UMA adaptive controller (if active).
    pub fn reconcile_apple_uma(
        &mut self,
        telemetry: AdaptiveFrameTelemetry,
    ) -> Option<AdaptiveBufferDecision> {
        self.sync.as_mut()?.reconcile_apple_uma(telemetry)
    }

    /// Record a queue submission for a tracked frame/lane.
    pub fn record_gpu_submission(
        &mut self,
        frame_id: BridgeFrameId,
        lane: GpuQueueLane,
        submission: GpuSubmissionHandle,
    ) -> bool {
        self.sync
            .as_mut()
            .map(|sync| sync.record_submission(frame_id, lane, submission))
            .unwrap_or(false)
    }

    /// Non-blocking compose reconcile (portable fence poll equivalent).
    pub fn try_reconcile_present_nonblocking(
        &mut self,
        primary_waiter: &dyn GpuSubmissionWaiter,
        secondary_waiter: Option<&dyn GpuSubmissionWaiter>,
        transfer_waiter: Option<&dyn GpuSubmissionWaiter>,
    ) -> Option<ComposeBarrierState> {
        self.sync.as_mut().map(|sync| {
            sync.try_reconcile_nonblocking(primary_waiter, secondary_waiter, transfer_waiter)
        })
    }

    /// Budgeted compose reconcile (bounded fence wait equivalent, default 0.8ms).
    pub fn reconcile_present(
        &mut self,
        primary_waiter: &dyn GpuSubmissionWaiter,
        secondary_waiter: Option<&dyn GpuSubmissionWaiter>,
        transfer_waiter: Option<&dyn GpuSubmissionWaiter>,
    ) -> Option<ComposeBarrierState> {
        self.sync.as_mut().map(|sync| {
            sync.reconcile_for_present(primary_waiter, secondary_waiter, transfer_waiter)
        })
    }

    /// Snapshot metrics across the bridge, MPS, and the optional sync controller.
    pub fn metrics(&self) -> MpsGmsBridgeMetrics {
        MpsGmsBridgeMetrics {
            bridge_submitted: self.bridge_submitted.load(Ordering::Relaxed),
            cpu_completions_queued: self.cpu_completions_queued.load(Ordering::Relaxed),
            cpu_completions_drained: self.cpu_completions_drained.load(Ordering::Relaxed),
            frame_plans_published: self.frame_plans_published.load(Ordering::Relaxed),
            sealed_frames_pending: self.sealed_frames.len(),
            accumulators_pending: self.pending_frames.len(),
            gpu_plan_queue_depth: self.gpu_plan_queue.len(),
            sync_rejections: self.sync_rejections.load(Ordering::Relaxed),
            mps: self.mps.metrics(),
            sync: self.sync.as_ref().map(|sync| sync.snapshot()),
        }
    }

    fn next_plannable_frame_id(&self) -> Option<BridgeFrameId> {
        self.sealed_frames
            .iter()
            .copied()
            .find(|frame_id| self.pending_frames.contains_key(frame_id))
    }

    fn build_frame_plan(&mut self, frame_id: BridgeFrameId) -> Option<BridgeFramePlan> {
        let accumulator = self.pending_frames.remove(&frame_id)?;
        self.sealed_frames.remove(&frame_id);
        if accumulator.is_empty() {
            return None;
        }

        let single_request = accumulator.to_single_gpu_request();
        let multi_request = accumulator.to_multi_gpu_request(self.config.target_frame_budget_ms);

        let single_gpu_plan = self.gms_dispatcher.plan_dispatch(single_request);
        let multi_gpu_plan = self.multi_gpu_dispatcher.plan_dispatch(multi_request);

        self.ensure_sync_from_plan(&multi_gpu_plan);
        let sync_admission = if let Some(sync) = self.sync.as_mut() {
            let admission = sync.admit_frame(
                frame_id,
                multi_gpu_plan.secondary().is_some(),
                multi_gpu_plan.shared_texture_bridge.is_some(),
            );
            if !admission.accepted {
                self.sync_rejections.fetch_add(1, Ordering::Relaxed);
            }
            Some(admission)
        } else {
            None
        };

        let prefer_map_write_uploads = multi_gpu_plan.assignments.iter().any(|assignment| {
            assignment.zero_copy.prefer_mapped_primary
                || assignment
                    .zero_copy
                    .upload_buffer_usages
                    .contains(wgpu::BufferUsages::MAP_WRITE)
        }) || single_gpu_plan.assignments.iter().any(|assignment| {
            assignment.zero_copy.prefer_mapped_primary
                || assignment
                    .zero_copy
                    .upload_buffer_usages
                    .contains(wgpu::BufferUsages::MAP_WRITE)
        });

        let apple_uma_decision = self
            .sync
            .as_ref()
            .and_then(|sync| sync.snapshot().last_apple_uma_decision);

        Some(BridgeFramePlan {
            frame_id,
            created_at: Instant::now(),
            cpu_completed_tasks: accumulator.completed_tasks,
            cpu_execution_time_ms: accumulator.cpu_execution_ns as f64 / 1_000_000.0,
            single_gpu_plan,
            multi_gpu_plan,
            workload_request: multi_request,
            sync_admitted: sync_admission.map(|a| a.accepted).unwrap_or(false),
            sync_spill_secondary_hint: sync_admission
                .map(|a| a.should_spill_secondary)
                .unwrap_or(false),
            sync_secondary_scale_hint: sync_admission
                .map(|a| a.secondary_scale_hint)
                .unwrap_or(1.0),
            prefer_map_write_uploads,
            apple_uma_decision,
        })
    }

    fn ensure_sync_from_plan(&mut self, plan: &MultiGpuDispatchPlan) {
        let primary = plan.primary().and_then(|lane| {
            self.gpu_inventory
                .adapters
                .iter()
                .find(|g| g.index == lane.adapter_index)
        });
        let secondary = plan.secondary().and_then(|lane| {
            self.gpu_inventory
                .adapters
                .iter()
                .find(|g| g.index == lane.adapter_index)
        });

        let Some(primary) = primary else {
            self.sync = None;
            return;
        };

        // Recreate the synchronizer when topology/policy changes. This keeps the implementation
        // simple and avoids lock contention. Frame tracking is preserved only within one sync instance.
        let recreate = match self.sync.as_ref() {
            None => true,
            Some(sync) => {
                sync.frames_in_flight_limit()
                    != self
                        .config
                        .sync
                        .max_frames_in_flight_override
                        .unwrap_or(plan.sync.frames_in_flight)
                        .max(1)
                    || sync.backend_hint() != detect_expected_sync_backend(primary, secondary)
            }
        };

        if recreate {
            self.sync = Some(MultiGpuFrameSynchronizer::new(
                plan.sync,
                self.config.sync,
                primary,
                secondary,
            ));
        }
    }
}

fn detect_expected_sync_backend(
    primary: &gms::GpuAdapterProfile,
    secondary: Option<&gms::GpuAdapterProfile>,
) -> crate::graphics::multigpu::sync::SyncBackendHint {
    if primary.backend == wgpu::Backend::Metal
        || secondary
            .map(|g| g.backend == wgpu::Backend::Metal)
            .unwrap_or(false)
    {
        crate::graphics::multigpu::sync::SyncBackendHint::MetalSharedEventLike
    } else if primary.backend == wgpu::Backend::Vulkan
        || secondary
            .map(|g| g.backend == wgpu::Backend::Vulkan)
            .unwrap_or(false)
    {
        crate::graphics::multigpu::sync::SyncBackendHint::VulkanTimelineFenceLike
    } else {
        crate::graphics::multigpu::sync::SyncBackendHint::PortableWgpu
    }
}

fn average_bytes_per_job(total_bytes: u64, jobs: u32, fallback: u64) -> u64 {
    if jobs == 0 {
        return fallback;
    }
    (total_bytes / jobs as u64).max(1)
}

fn wrap_bridge_completion_task(
    bridge_submission_id: BridgeSubmissionId,
    descriptor: BridgeTaskDescriptor,
    task: NativeTask,
    completion_queue: Arc<SegQueue<CpuCompletionRecord>>,
    cpu_completions_queued: Arc<AtomicU64>,
) -> NativeTask {
    Box::new(move || {
        let started = Instant::now();
        task();
        let completed_at = Instant::now();
        let cpu_elapsed = completed_at.saturating_duration_since(started);
        completion_queue.push(CpuCompletionRecord {
            submission_id: bridge_submission_id,
            descriptor,
            cpu_elapsed,
            completed_at,
        });
        cpu_completions_queued.fetch_add(1, Ordering::Relaxed);
    })
}

fn effective_task_class(descriptor: BridgeTaskDescriptor) -> TaskClass {
    let base = match descriptor.gpu_kind {
        BridgeGpuTaskKind::TractionVfPhysics | BridgeGpuTaskKind::MassiveParticles => {
            TaskClass::Physics
        }
        BridgeGpuTaskKind::SampledMesh => TaskClass::SampledProcessing,
        BridgeGpuTaskKind::ObjectUpdate => TaskClass::ObjectUpdate,
        BridgeGpuTaskKind::AiMl => TaskClass::AiMl,
        BridgeGpuTaskKind::Ui => TaskClass::Ui,
        BridgeGpuTaskKind::PostFx => TaskClass::PostFx,
        BridgeGpuTaskKind::Custom(class) => class,
    };

    match descriptor.routing {
        BridgeTaskRouting::Auto => base,
        BridgeTaskRouting::ForcePrimaryHeavy => match base {
            // Reclassify latency tasks into heavier throughput lanes so the primary GPU receives
            // them in the planner's asymmetry pass.
            TaskClass::Ui => TaskClass::ObjectUpdate,
            TaskClass::PostFx => TaskClass::SampledProcessing,
            TaskClass::AiMl => TaskClass::SampledProcessing,
            other => other,
        },
        BridgeTaskRouting::ForceSecondaryLatency => match base {
            // Reclassify heavy tasks into latency lanes when the caller explicitly wants helper-GPU
            // overlap. UI vs Post-FX split is chosen by data size (texture-heavy -> Post-FX).
            TaskClass::SampledProcessing | TaskClass::Physics | TaskClass::AiMl => {
                if descriptor.processed_texture_bytes > 0 {
                    TaskClass::PostFx
                } else {
                    TaskClass::Ui
                }
            }
            TaskClass::ObjectUpdate => TaskClass::Ui,
            other => other,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routing_override_can_push_latency_to_heavy_lane() {
        let descriptor = BridgeTaskDescriptor {
            gpu_kind: BridgeGpuTaskKind::Ui,
            routing: BridgeTaskRouting::ForcePrimaryHeavy,
            ..Default::default()
        };
        assert_eq!(effective_task_class(descriptor), TaskClass::ObjectUpdate);
    }

    #[test]
    fn routing_override_can_push_heavy_to_latency_lane() {
        let descriptor = BridgeTaskDescriptor {
            gpu_kind: BridgeGpuTaskKind::TractionVfPhysics,
            routing: BridgeTaskRouting::ForceSecondaryLatency,
            ..Default::default()
        };
        assert!(matches!(
            effective_task_class(descriptor),
            TaskClass::Ui | TaskClass::PostFx
        ));
    }
}
