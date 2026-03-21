//! Custom MPS physics thread pool.
//!
//! This module is intentionally narrow and data-oriented:
//! - workers are pinned to logical CPU ids
//! - jobs are passed through a lock-free MPMC queue
//! - physics frame execution is overlapped with render-side scene building
//! - transform state uses double buffering so render N and physics N+1 never
//!   race on the same write path

use crate::topology::{CpuClass, CpuTopology};
use crate::worker::{
    normalize_worker_launch_for_host, spawn_worker, WorkerLaunchConfig, WorkerSignal,
};
use crossbeam::queue::SegQueue;
use std::io;
use std::ops::Range;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

const NO_ACTIVE_FRAME: u64 = u64::MAX;

/// Per-body transform sample exchanged between render and physics.
#[derive(Debug, Clone, Copy)]
pub struct TransformSample {
    /// Object position in world space.
    pub position: [f32; 3],
    /// Object orientation as a quaternion.
    pub rotation: [f32; 4],
}

impl Default for TransformSample {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
        }
    }
}

struct TransformBuffer {
    position_x: Vec<AtomicU32>,
    position_y: Vec<AtomicU32>,
    position_z: Vec<AtomicU32>,
    rotation_x: Vec<AtomicU32>,
    rotation_y: Vec<AtomicU32>,
    rotation_z: Vec<AtomicU32>,
    rotation_w: Vec<AtomicU32>,
    frame_id: AtomicU64,
}

impl TransformBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            position_x: atomic_f32_lane(capacity, 0.0),
            position_y: atomic_f32_lane(capacity, 0.0),
            position_z: atomic_f32_lane(capacity, 0.0),
            rotation_x: atomic_f32_lane(capacity, 0.0),
            rotation_y: atomic_f32_lane(capacity, 0.0),
            rotation_z: atomic_f32_lane(capacity, 0.0),
            rotation_w: atomic_f32_lane(capacity, 1.0),
            frame_id: AtomicU64::new(0),
        }
    }

    fn write(&self, index: usize, sample: TransformSample) {
        self.position_x[index].store(sample.position[0].to_bits(), Ordering::Release);
        self.position_y[index].store(sample.position[1].to_bits(), Ordering::Release);
        self.position_z[index].store(sample.position[2].to_bits(), Ordering::Release);
        self.rotation_x[index].store(sample.rotation[0].to_bits(), Ordering::Release);
        self.rotation_y[index].store(sample.rotation[1].to_bits(), Ordering::Release);
        self.rotation_z[index].store(sample.rotation[2].to_bits(), Ordering::Release);
        self.rotation_w[index].store(sample.rotation[3].to_bits(), Ordering::Release);
    }

    fn read(&self, index: usize) -> TransformSample {
        TransformSample {
            position: [
                f32::from_bits(self.position_x[index].load(Ordering::Acquire)),
                f32::from_bits(self.position_y[index].load(Ordering::Acquire)),
                f32::from_bits(self.position_z[index].load(Ordering::Acquire)),
            ],
            rotation: [
                f32::from_bits(self.rotation_x[index].load(Ordering::Acquire)),
                f32::from_bits(self.rotation_y[index].load(Ordering::Acquire)),
                f32::from_bits(self.rotation_z[index].load(Ordering::Acquire)),
                f32::from_bits(self.rotation_w[index].load(Ordering::Acquire)),
            ],
        }
    }
}

fn atomic_f32_lane(capacity: usize, initial_value: f32) -> Vec<AtomicU32> {
    let bits = initial_value.to_bits();
    (0..capacity).map(|_| AtomicU32::new(bits)).collect()
}

/// Double-buffered transform storage shared between render and physics.
pub struct DoubleBufferedTransformStorage {
    buffers: [TransformBuffer; 2],
    capacity: usize,
    render_read_slot: AtomicUsize,
    physics_write_slot: AtomicUsize,
    latest_published_frame: AtomicU64,
    active_write_frame: AtomicU64,
}

impl DoubleBufferedTransformStorage {
    /// Allocate the double-buffered transform storage.
    pub fn new(capacity: usize) -> Self {
        Self {
            buffers: [
                TransformBuffer::new(capacity),
                TransformBuffer::new(capacity),
            ],
            capacity,
            render_read_slot: AtomicUsize::new(0),
            physics_write_slot: AtomicUsize::new(1),
            latest_published_frame: AtomicU64::new(0),
            active_write_frame: AtomicU64::new(NO_ACTIVE_FRAME),
        }
    }

    /// Number of transform entries available per slot.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Slot currently read by render-side code.
    pub fn render_read_slot(&self) -> usize {
        self.render_read_slot.load(Ordering::Acquire)
    }

    /// Slot currently written by physics-side code.
    pub fn physics_write_slot(&self) -> usize {
        self.physics_write_slot.load(Ordering::Acquire)
    }

    /// Most recently published physics frame id, if any.
    pub fn latest_published_frame(&self) -> Option<u64> {
        let frame_id = self.latest_published_frame.load(Ordering::Acquire);
        (frame_id != 0).then_some(frame_id)
    }

    /// Mark a frame as the current write target.
    pub fn begin_write_frame(&self, frame_id: u64) -> usize {
        self.active_write_frame.store(frame_id, Ordering::Release);
        self.physics_write_slot()
    }

    /// Write a transform into the active physics slot.
    pub fn write_transform(&self, index: usize, sample: TransformSample) {
        let slot = self.physics_write_slot();
        self.write_transform_to_slot(slot, index, sample);
    }

    /// Write a transform into an explicit slot.
    pub fn write_transform_to_slot(&self, slot: usize, index: usize, sample: TransformSample) {
        if index >= self.capacity || slot > 1 {
            return;
        }
        self.buffers[slot].write(index, sample);
    }

    /// Read a transform from the current render slot.
    pub fn read_render_transform(&self, index: usize) -> TransformSample {
        self.read_transform_from_slot(self.render_read_slot(), index)
    }

    /// Read a transform from an explicit slot.
    pub fn read_transform_from_slot(&self, slot: usize, index: usize) -> TransformSample {
        if index >= self.capacity || slot > 1 {
            return TransformSample::default();
        }
        self.buffers[slot].read(index)
    }

    /// Publish the current write slot and rotate the buffers.
    pub fn publish_completed_write(&self, frame_id: u64) {
        if self.active_write_frame.load(Ordering::Acquire) != frame_id {
            return;
        }

        let previous_render_slot = self.render_read_slot();
        let completed_write_slot = self.physics_write_slot();
        self.buffers[completed_write_slot]
            .frame_id
            .store(frame_id, Ordering::Release);
        self.render_read_slot
            .store(completed_write_slot, Ordering::Release);
        self.physics_write_slot
            .store(previous_render_slot, Ordering::Release);
        self.latest_published_frame
            .store(frame_id, Ordering::Release);
        self.active_write_frame
            .store(NO_ACTIVE_FRAME, Ordering::Release);
    }
}

/// Physics phase handled by the custom thread pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhysicsPhase {
    /// Broadphase overlap generation.
    Broadphase,
    /// Narrowphase contact generation.
    Narrowphase,
    /// Integration and state write-back.
    Integration,
}

impl PhysicsPhase {
    fn next(self) -> Option<Self> {
        match self {
            Self::Broadphase => Some(Self::Narrowphase),
            Self::Narrowphase => Some(Self::Integration),
            Self::Integration => None,
        }
    }
}

/// Per-job context passed to physics phase callbacks.
#[derive(Clone)]
pub struct PhysicsJobContext {
    /// Physics frame id being executed.
    pub frame_id: u64,
    /// Current physics phase.
    pub phase: PhysicsPhase,
    /// Linear work range for the job.
    pub work_range: Range<usize>,
    /// Render slot still visible to the main thread.
    pub render_read_slot: usize,
    /// Physics slot being written by this frame.
    pub physics_write_slot: usize,
    /// Shared transform storage.
    pub transforms: Arc<DoubleBufferedTransformStorage>,
    /// Worker metadata for phase-local specialization.
    pub worker_index: usize,
    /// Logical core backing the current worker.
    pub worker_core_id: usize,
    /// Core class backing the current worker.
    pub worker_class: CpuClass,
}

/// Phase callback function.
pub type PhysicsPhaseHandler = Arc<dyn Fn(&PhysicsJobContext) + Send + Sync + 'static>;

/// Callback set used when a new physics frame is triggered.
#[derive(Clone, Default)]
pub struct PhysicsPhaseCallbacks {
    /// Broadphase callback.
    pub broadphase: Option<PhysicsPhaseHandler>,
    /// Narrowphase callback.
    pub narrowphase: Option<PhysicsPhaseHandler>,
    /// Integration callback.
    pub integration: Option<PhysicsPhaseHandler>,
}

impl PhysicsPhaseCallbacks {
    /// Attach a broadphase callback.
    pub fn with_broadphase(mut self, handler: PhysicsPhaseHandler) -> Self {
        self.broadphase = Some(handler);
        self
    }

    /// Attach a narrowphase callback.
    pub fn with_narrowphase(mut self, handler: PhysicsPhaseHandler) -> Self {
        self.narrowphase = Some(handler);
        self
    }

    /// Attach an integration callback.
    pub fn with_integration(mut self, handler: PhysicsPhaseHandler) -> Self {
        self.integration = Some(handler);
        self
    }

    fn handler_for(&self, phase: PhysicsPhase) -> Option<PhysicsPhaseHandler> {
        match phase {
            PhysicsPhase::Broadphase => self.broadphase.clone(),
            PhysicsPhase::Narrowphase => self.narrowphase.clone(),
            PhysicsPhase::Integration => self.integration.clone(),
        }
    }
}

/// Trigger payload used to kick off a new asynchronous physics frame.
#[derive(Debug, Clone)]
pub struct PhysicsStepTrigger {
    /// Monotonic physics frame id.
    pub frame_id: u64,
    /// Number of broadphase items.
    pub broadphase_items: usize,
    /// Number of narrowphase items.
    pub narrowphase_items: usize,
    /// Number of integration items.
    pub integration_items: usize,
    /// Chunk size used for phase job partitioning.
    pub chunk_size: usize,
}

impl PhysicsStepTrigger {
    /// Build a trigger with the provided frame id and item counts.
    pub fn new(
        frame_id: u64,
        broadphase_items: usize,
        narrowphase_items: usize,
        integration_items: usize,
        chunk_size: usize,
    ) -> Self {
        Self {
            frame_id,
            broadphase_items,
            narrowphase_items,
            integration_items,
            chunk_size,
        }
    }
}

/// Ticket returned immediately after physics N+1 is launched.
#[derive(Debug, Clone, Copy)]
pub struct PhysicsFrameTicket {
    /// Triggered frame id.
    pub frame_id: u64,
    /// Render slot still safe to read while this frame runs.
    pub render_read_slot: usize,
    /// Write slot reserved for the new physics frame.
    pub physics_write_slot: usize,
    /// Planned number of jobs for the frame.
    pub planned_job_count: usize,
}

/// Thread-pool configuration.
#[derive(Debug, Clone)]
pub struct MpsThreadPoolConfig {
    /// Topology snapshot used during worker placement.
    pub topology: CpuTopology,
    /// Logical core ids assigned to workers.
    pub worker_core_ids: Vec<usize>,
    /// Number of transform entries per double-buffered slot.
    pub transform_capacity: usize,
    /// Default chunk size used when a trigger requests zero.
    pub default_chunk_size: usize,
    /// Tight spin count before yielding.
    pub spin_iterations: u32,
    /// Cooperative yields before futex sleep.
    pub yield_iterations: u32,
    /// Nice value for performance workers.
    pub performance_nice: i32,
    /// Nice value for efficient workers.
    pub efficient_nice: i32,
    /// Nice value for unknown workers.
    pub unknown_nice: i32,
}

impl Default for MpsThreadPoolConfig {
    fn default() -> Self {
        let topology = CpuTopology::detect();
        let worker_core_ids = topology.preferred_core_ids();
        Self {
            worker_core_ids,
            topology,
            transform_capacity: 30_000,
            default_chunk_size: 256,
            spin_iterations: 2_048,
            yield_iterations: 64,
            performance_nice: -6,
            efficient_nice: -2,
            unknown_nice: -4,
        }
    }
}

impl MpsThreadPoolConfig {
    fn nice_value_for_class(&self, class: CpuClass) -> i32 {
        match class {
            CpuClass::Performance => self.performance_nice,
            CpuClass::Efficient => self.efficient_nice,
            CpuClass::Unknown => self.unknown_nice,
        }
    }
}

/// Public thread-pool error.
#[derive(Debug)]
pub enum ThreadPoolError {
    /// No worker cores were configured.
    NoWorkersConfigured,
    /// Zero-capacity transform storage is not valid.
    ZeroTransformCapacity,
    /// A physics frame is already in flight.
    FrameAlreadyInFlight { active_frame_id: u64 },
    /// Worker thread creation failed.
    WorkerSpawn(io::Error),
}

/// Public metrics snapshot for thread-pool telemetry.
#[derive(Debug, Clone, Copy, Default)]
pub struct MpsThreadPoolMetrics {
    /// Total worker count.
    pub worker_count: usize,
    /// Jobs currently waiting in the queue.
    pub queued_jobs: u64,
    /// Jobs currently executing or queued across the active frame.
    pub in_flight_jobs: u64,
    /// Completed jobs since startup.
    pub completed_jobs: u64,
    /// Completed physics frames since startup.
    pub completed_frames: u64,
    /// Latest completed frame id.
    pub latest_completed_frame: Option<u64>,
    /// Current in-flight frame id.
    pub active_frame_id: Option<u64>,
}

struct PhysicsJob {
    frame: Arc<PhysicsFrameState>,
    phase: PhysicsPhase,
    work_range: Range<usize>,
}

struct PhysicsFrameState {
    frame_id: u64,
    render_read_slot: usize,
    physics_write_slot: usize,
    chunk_size: usize,
    broadphase_items: usize,
    narrowphase_items: usize,
    integration_items: usize,
    callbacks: PhysicsPhaseCallbacks,
    transforms: Arc<DoubleBufferedTransformStorage>,
    broadphase_remaining: AtomicUsize,
    narrowphase_remaining: AtomicUsize,
    integration_remaining: AtomicUsize,
}

impl PhysicsFrameState {
    fn new(
        trigger: PhysicsStepTrigger,
        callbacks: PhysicsPhaseCallbacks,
        transforms: Arc<DoubleBufferedTransformStorage>,
        render_read_slot: usize,
        physics_write_slot: usize,
        chunk_size: usize,
    ) -> Self {
        Self {
            frame_id: trigger.frame_id,
            render_read_slot,
            physics_write_slot,
            chunk_size: chunk_size.max(1),
            broadphase_items: trigger.broadphase_items,
            narrowphase_items: trigger.narrowphase_items,
            integration_items: trigger.integration_items,
            callbacks,
            transforms,
            broadphase_remaining: AtomicUsize::new(0),
            narrowphase_remaining: AtomicUsize::new(0),
            integration_remaining: AtomicUsize::new(0),
        }
    }

    fn item_count(&self, phase: PhysicsPhase) -> usize {
        match phase {
            PhysicsPhase::Broadphase => self.broadphase_items,
            PhysicsPhase::Narrowphase => self.narrowphase_items,
            PhysicsPhase::Integration => self.integration_items,
        }
    }

    fn remaining_counter(&self, phase: PhysicsPhase) -> &AtomicUsize {
        match phase {
            PhysicsPhase::Broadphase => &self.broadphase_remaining,
            PhysicsPhase::Narrowphase => &self.narrowphase_remaining,
            PhysicsPhase::Integration => &self.integration_remaining,
        }
    }

    fn planned_job_count(&self) -> usize {
        let mut planned = 0;
        for phase in [
            PhysicsPhase::Broadphase,
            PhysicsPhase::Narrowphase,
            PhysicsPhase::Integration,
        ] {
            if self.callbacks.handler_for(phase).is_some() && self.item_count(phase) > 0 {
                planned += self.item_count(phase).div_ceil(self.chunk_size);
            }
        }
        planned
    }
}

/// Custom lock-free thread pool used by MPS physics stages.
pub struct MpsThreadPool {
    topology: CpuTopology,
    worker_core_ids: Vec<usize>,
    transforms: Arc<DoubleBufferedTransformStorage>,
    queue: Arc<SegQueue<PhysicsJob>>,
    signal: Arc<WorkerSignal>,
    queued_jobs: Arc<AtomicU64>,
    in_flight_jobs: Arc<AtomicU64>,
    completed_jobs: Arc<AtomicU64>,
    completed_frames: Arc<AtomicU64>,
    latest_completed_frame: Arc<AtomicU64>,
    active_frame_id: Arc<AtomicU64>,
    default_chunk_size: usize,
    workers: Vec<JoinHandle<()>>,
}

impl MpsThreadPool {
    /// Create and start the custom physics thread pool.
    pub fn new(config: MpsThreadPoolConfig) -> Result<Self, ThreadPoolError> {
        if config.worker_core_ids.is_empty() {
            return Err(ThreadPoolError::NoWorkersConfigured);
        }
        if config.transform_capacity == 0 {
            return Err(ThreadPoolError::ZeroTransformCapacity);
        }

        let topology = config.topology.clone();
        let transforms = Arc::new(DoubleBufferedTransformStorage::new(
            config.transform_capacity,
        ));
        let queue = Arc::new(SegQueue::new());
        let signal = Arc::new(WorkerSignal::default());
        let queued_jobs = Arc::new(AtomicU64::new(0));
        let in_flight_jobs = Arc::new(AtomicU64::new(0));
        let completed_jobs = Arc::new(AtomicU64::new(0));
        let completed_frames = Arc::new(AtomicU64::new(0));
        let latest_completed_frame = Arc::new(AtomicU64::new(0));
        let active_frame_id = Arc::new(AtomicU64::new(NO_ACTIVE_FRAME));

        let mut workers = Vec::with_capacity(config.worker_core_ids.len());
        for (worker_index, &logical_core_id) in config.worker_core_ids.iter().enumerate() {
            let class = topology.class_for_core(logical_core_id);
            let mut launch = WorkerLaunchConfig::new(
                worker_index,
                logical_core_id,
                class,
                format!("mps-physics-worker-{worker_index}-core-{logical_core_id}-{class:?}"),
                config.nice_value_for_class(class),
                config.spin_iterations,
                config.yield_iterations,
            );
            normalize_worker_launch_for_host(&mut launch);

            let worker_queue = Arc::clone(&queue);
            let worker_signal = Arc::clone(&signal);
            let worker_queued_jobs = Arc::clone(&queued_jobs);
            let worker_in_flight_jobs = Arc::clone(&in_flight_jobs);
            let worker_completed_jobs = Arc::clone(&completed_jobs);
            let worker_completed_frames = Arc::clone(&completed_frames);
            let worker_latest_completed_frame = Arc::clone(&latest_completed_frame);
            let worker_active_frame_id = Arc::clone(&active_frame_id);

            let handle = spawn_worker(launch, Arc::clone(&signal), move |launch, signal| {
                physics_worker_loop(
                    launch,
                    signal,
                    worker_queue,
                    worker_queued_jobs,
                    worker_in_flight_jobs,
                    worker_completed_jobs,
                    worker_completed_frames,
                    worker_latest_completed_frame,
                    worker_active_frame_id,
                );
            })
            .map_err(ThreadPoolError::WorkerSpawn)?;

            workers.push(handle);
            worker_signal.wake_one();
        }

        Ok(Self {
            topology,
            worker_core_ids: config.worker_core_ids,
            transforms,
            queue,
            signal,
            queued_jobs,
            in_flight_jobs,
            completed_jobs,
            completed_frames,
            latest_completed_frame,
            active_frame_id,
            default_chunk_size: config.default_chunk_size.max(1),
            workers,
        })
    }

    /// Return the topology snapshot used during pool construction.
    pub fn topology(&self) -> &CpuTopology {
        &self.topology
    }

    /// Return the current render-visible slot.
    pub fn current_render_slot(&self) -> usize {
        self.transforms.render_read_slot()
    }

    /// Return the current physics write slot.
    pub fn current_write_slot(&self) -> usize {
        self.transforms.physics_write_slot()
    }

    /// Return shared access to the transform storage.
    pub fn transform_storage(&self) -> Arc<DoubleBufferedTransformStorage> {
        Arc::clone(&self.transforms)
    }

    /// Latest completed physics frame id, if any.
    pub fn latest_completed_frame(&self) -> Option<u64> {
        let frame_id = self.latest_completed_frame.load(Ordering::Acquire);
        (frame_id != 0).then_some(frame_id)
    }

    /// Trigger asynchronous physics for frame N+1 while render N continues.
    pub fn trigger_next_physics_step(
        &self,
        trigger: PhysicsStepTrigger,
        callbacks: PhysicsPhaseCallbacks,
    ) -> Result<PhysicsFrameTicket, ThreadPoolError> {
        match self.active_frame_id.compare_exchange(
            NO_ACTIVE_FRAME,
            trigger.frame_id,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => {}
            Err(active_frame_id) => {
                return Err(ThreadPoolError::FrameAlreadyInFlight { active_frame_id });
            }
        }

        let render_read_slot = self.transforms.render_read_slot();
        let physics_write_slot = self.transforms.begin_write_frame(trigger.frame_id);
        let frame = Arc::new(PhysicsFrameState::new(
            trigger.clone(),
            callbacks,
            Arc::clone(&self.transforms),
            render_read_slot,
            physics_write_slot,
            if trigger.chunk_size == 0 {
                self.default_chunk_size
            } else {
                trigger.chunk_size
            },
        ));
        let planned_job_count = frame.planned_job_count();

        if planned_job_count == 0 {
            self.complete_frame(&frame);
        } else {
            self.schedule_phase(&frame, PhysicsPhase::Broadphase);
        }

        Ok(PhysicsFrameTicket {
            frame_id: frame.frame_id,
            render_read_slot,
            physics_write_slot,
            planned_job_count,
        })
    }

    /// Wait until a specific frame becomes visible to render code.
    pub fn wait_for_frame(&self, frame_id: u64, timeout: Duration) -> bool {
        let started = Instant::now();
        while started.elapsed() <= timeout {
            if self
                .latest_completed_frame()
                .is_some_and(|completed_frame| completed_frame >= frame_id)
            {
                return true;
            }
            std::hint::spin_loop();
            std::thread::yield_now();
        }
        false
    }

    /// Collect a metrics snapshot.
    pub fn metrics(&self) -> MpsThreadPoolMetrics {
        MpsThreadPoolMetrics {
            worker_count: self.worker_core_ids.len(),
            queued_jobs: self.queued_jobs.load(Ordering::Acquire),
            in_flight_jobs: self.in_flight_jobs.load(Ordering::Acquire),
            completed_jobs: self.completed_jobs.load(Ordering::Acquire),
            completed_frames: self.completed_frames.load(Ordering::Acquire),
            latest_completed_frame: self.latest_completed_frame(),
            active_frame_id: match self.active_frame_id.load(Ordering::Acquire) {
                NO_ACTIVE_FRAME => None,
                frame_id => Some(frame_id),
            },
        }
    }

    fn schedule_phase(&self, frame: &Arc<PhysicsFrameState>, phase: PhysicsPhase) {
        if frame.callbacks.handler_for(phase).is_none() || frame.item_count(phase) == 0 {
            match phase.next() {
                Some(next_phase) => self.schedule_phase(frame, next_phase),
                None => self.complete_frame(frame),
            }
            return;
        }

        let item_count = frame.item_count(phase);
        let chunk_size = frame.chunk_size.max(1);
        let job_count = item_count.div_ceil(chunk_size);
        frame
            .remaining_counter(phase)
            .store(job_count, Ordering::Release);

        self.queued_jobs
            .fetch_add(job_count as u64, Ordering::AcqRel);
        self.in_flight_jobs
            .fetch_add(job_count as u64, Ordering::AcqRel);

        for job_index in 0..job_count {
            let start = job_index * chunk_size;
            let end = (start + chunk_size).min(item_count);
            self.queue.push(PhysicsJob {
                frame: Arc::clone(frame),
                phase,
                work_range: start..end,
            });
        }

        self.signal.wake_all();
    }

    fn complete_frame(&self, frame: &PhysicsFrameState) {
        self.transforms.publish_completed_write(frame.frame_id);
        self.latest_completed_frame
            .store(frame.frame_id, Ordering::Release);
        self.completed_frames.fetch_add(1, Ordering::AcqRel);
        self.active_frame_id
            .compare_exchange(
                frame.frame_id,
                NO_ACTIVE_FRAME,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .ok();
        self.signal.wake_all();
    }
}

impl Drop for MpsThreadPool {
    fn drop(&mut self) {
        self.signal.request_shutdown();
        while let Some(handle) = self.workers.pop() {
            let _ = handle.join();
        }
    }
}

fn physics_worker_loop(
    launch: WorkerLaunchConfig,
    signal: Arc<WorkerSignal>,
    queue: Arc<SegQueue<PhysicsJob>>,
    queued_jobs: Arc<AtomicU64>,
    in_flight_jobs: Arc<AtomicU64>,
    completed_jobs: Arc<AtomicU64>,
    completed_frames: Arc<AtomicU64>,
    latest_completed_frame: Arc<AtomicU64>,
    active_frame_id: Arc<AtomicU64>,
) {
    let mut observed_epoch = signal.observed_epoch();

    loop {
        if let Some(job) = queue.pop() {
            queued_jobs.fetch_sub(1, Ordering::AcqRel);
            execute_job(
                &launch,
                job,
                &in_flight_jobs,
                &completed_jobs,
                &completed_frames,
                &latest_completed_frame,
                &active_frame_id,
                &signal,
                &queue,
                &queued_jobs,
            );
            observed_epoch = signal.observed_epoch();
            continue;
        }

        if signal.is_shutdown_requested()
            && queued_jobs.load(Ordering::Acquire) == 0
            && in_flight_jobs.load(Ordering::Acquire) == 0
        {
            break;
        }

        signal.wait_for_change(
            &mut observed_epoch,
            launch.spin_iterations,
            launch.yield_iterations,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_job(
    launch: &WorkerLaunchConfig,
    job: PhysicsJob,
    in_flight_jobs: &AtomicU64,
    completed_jobs: &AtomicU64,
    completed_frames: &AtomicU64,
    latest_completed_frame: &AtomicU64,
    active_frame_id: &AtomicU64,
    signal: &WorkerSignal,
    queue: &SegQueue<PhysicsJob>,
    queued_jobs: &AtomicU64,
) {
    if let Some(handler) = job.frame.callbacks.handler_for(job.phase) {
        let context = PhysicsJobContext {
            frame_id: job.frame.frame_id,
            phase: job.phase,
            work_range: job.work_range,
            render_read_slot: job.frame.render_read_slot,
            physics_write_slot: job.frame.physics_write_slot,
            transforms: Arc::clone(&job.frame.transforms),
            worker_index: launch.worker_index,
            worker_core_id: launch.logical_core_id,
            worker_class: launch.class,
        };
        handler(&context);
    }

    completed_jobs.fetch_add(1, Ordering::AcqRel);
    in_flight_jobs.fetch_sub(1, Ordering::AcqRel);

    if job
        .frame
        .remaining_counter(job.phase)
        .fetch_sub(1, Ordering::AcqRel)
        == 1
    {
        match job.phase.next() {
            Some(next_phase) => schedule_followup_phase(
                &job.frame,
                next_phase,
                queue,
                queued_jobs,
                in_flight_jobs,
                completed_frames,
                latest_completed_frame,
                active_frame_id,
                signal,
            ),
            None => complete_frame_from_worker(
                &job.frame,
                completed_frames,
                latest_completed_frame,
                active_frame_id,
                signal,
            ),
        }
    }
}

fn schedule_followup_phase(
    frame: &Arc<PhysicsFrameState>,
    phase: PhysicsPhase,
    queue: &SegQueue<PhysicsJob>,
    queued_jobs: &AtomicU64,
    in_flight_jobs: &AtomicU64,
    completed_frames: &AtomicU64,
    latest_completed_frame: &AtomicU64,
    active_frame_id: &AtomicU64,
    signal: &WorkerSignal,
) {
    if frame.callbacks.handler_for(phase).is_none() || frame.item_count(phase) == 0 {
        match phase.next() {
            Some(next_phase) => schedule_followup_phase(
                frame,
                next_phase,
                queue,
                queued_jobs,
                in_flight_jobs,
                completed_frames,
                latest_completed_frame,
                active_frame_id,
                signal,
            ),
            None => {
                complete_frame_from_worker(
                    frame,
                    completed_frames,
                    latest_completed_frame,
                    active_frame_id,
                    signal,
                );
            }
        }
        return;
    }

    let item_count = frame.item_count(phase);
    let job_count = item_count.div_ceil(frame.chunk_size);
    frame
        .remaining_counter(phase)
        .store(job_count, Ordering::Release);
    queued_jobs.fetch_add(job_count as u64, Ordering::AcqRel);
    in_flight_jobs.fetch_add(job_count as u64, Ordering::AcqRel);

    for job_index in 0..job_count {
        let start = job_index * frame.chunk_size;
        let end = (start + frame.chunk_size).min(item_count);
        queue.push(PhysicsJob {
            frame: Arc::clone(frame),
            phase,
            work_range: start..end,
        });
    }

    signal.wake_all();
}

fn complete_frame_from_worker(
    frame: &PhysicsFrameState,
    completed_frames: &AtomicU64,
    latest_completed_frame: &AtomicU64,
    active_frame_id: &AtomicU64,
    signal: &WorkerSignal,
) {
    frame.transforms.publish_completed_write(frame.frame_id);
    latest_completed_frame.store(frame.frame_id, Ordering::Release);
    completed_frames.fetch_add(1, Ordering::AcqRel);
    active_frame_id
        .compare_exchange(
            frame.frame_id,
            NO_ACTIVE_FRAME,
            Ordering::AcqRel,
            Ordering::Acquire,
        )
        .ok();
    signal.wake_all();
}
