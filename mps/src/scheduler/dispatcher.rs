//! Dispatcher for native/WASM jobs and overlapped physics frame execution.
//!
//! The stateless `Dispatcher` continues to execute queued native/WASM payloads,
//! while `TaskDispatcher` provides a bare-metal frame pipeline intended to
//! replace stage/barrier driven ECS scheduling in hot physics paths.

use super::{TaskPayload, WasmTask};
use crate::topology::{CpuClass, CpuTopology};
use crate::worker::{spawn_worker, WorkerLaunchConfig, WorkerSignal};
use crossbeam::queue::{ArrayQueue, SegQueue};
use std::ops::Range;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use wasmer::{imports, Instance, Module, Store, Value};

const NO_ACTIVE_FRAME: u64 = u64::MAX;

/// Dispatcher-level result type.
pub type DispatchResult<T = ()> = Result<T, DispatchError>;

/// Dispatch errors for native/WASM execution.
#[derive(Debug, Clone)]
pub enum DispatchError {
    Compile(String),
    Instantiate(String),
    MissingExport(String),
    Runtime(String),
    QueueSaturated(String),
    FrameAlreadyInFlight(u64),
    WorkerSpawn(String),
}

/// Stateless dispatcher.
///
/// The implementation currently compiles WASM modules on demand.
/// A lock-free module cache can be layered on top in a later phase.
#[derive(Debug, Default)]
pub struct Dispatcher;

impl Dispatcher {
    /// Build a new dispatcher.
    pub fn new() -> Self {
        Self
    }

    /// Execute a queued task payload.
    pub fn execute(&self, payload: TaskPayload) -> DispatchResult {
        match payload {
            TaskPayload::Native(task) => {
                task();
                Ok(())
            }
            TaskPayload::Wasm(wasm_task) => self.execute_wasm(wasm_task),
        }
    }

    fn execute_wasm(&self, wasm_task: WasmTask) -> DispatchResult {
        // Wasmer's default engine provides JIT/AOT capabilities depending on platform.
        let mut store = Store::default();

        let module = Module::new(&store, wasm_task.module_bytes.as_ref())
            .map_err(|err| DispatchError::Compile(err.to_string()))?;

        let import_object = imports! {};
        let instance = Instance::new(&mut store, &module, &import_object)
            .map_err(|err| DispatchError::Instantiate(err.to_string()))?;

        let entrypoint = wasm_task.entrypoint;
        let function = instance
            .exports
            .get_function(&entrypoint)
            .map_err(|_| DispatchError::MissingExport(entrypoint.clone()))?;

        let params: Vec<Value> = wasm_task.args.into_iter().map(Value::I64).collect();
        function
            .call(&mut store, &params)
            .map_err(|err| DispatchError::Runtime(err.to_string()))?;

        Ok(())
    }
}

/// Per-transform sample exchanged between the render thread and physics workers.
#[derive(Debug, Clone, Copy)]
pub struct DispatcherTransformSample {
    /// World-space translation.
    pub position: [f32; 3],
    /// Object orientation in quaternion form.
    pub rotation: [f32; 4],
}

impl Default for DispatcherTransformSample {
    fn default() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
        }
    }
}

struct TransformLaneBuffer {
    position_x: Vec<AtomicU32>,
    position_y: Vec<AtomicU32>,
    position_z: Vec<AtomicU32>,
    rotation_x: Vec<AtomicU32>,
    rotation_y: Vec<AtomicU32>,
    rotation_z: Vec<AtomicU32>,
    rotation_w: Vec<AtomicU32>,
}

impl TransformLaneBuffer {
    fn new(capacity: usize) -> Self {
        Self {
            position_x: atomic_f32_lane(capacity, 0.0),
            position_y: atomic_f32_lane(capacity, 0.0),
            position_z: atomic_f32_lane(capacity, 0.0),
            rotation_x: atomic_f32_lane(capacity, 0.0),
            rotation_y: atomic_f32_lane(capacity, 0.0),
            rotation_z: atomic_f32_lane(capacity, 0.0),
            rotation_w: atomic_f32_lane(capacity, 1.0),
        }
    }

    fn write(&self, index: usize, sample: DispatcherTransformSample) {
        self.position_x[index].store(sample.position[0].to_bits(), Ordering::Release);
        self.position_y[index].store(sample.position[1].to_bits(), Ordering::Release);
        self.position_z[index].store(sample.position[2].to_bits(), Ordering::Release);
        self.rotation_x[index].store(sample.rotation[0].to_bits(), Ordering::Release);
        self.rotation_y[index].store(sample.rotation[1].to_bits(), Ordering::Release);
        self.rotation_z[index].store(sample.rotation[2].to_bits(), Ordering::Release);
        self.rotation_w[index].store(sample.rotation[3].to_bits(), Ordering::Release);
    }

    fn read(&self, index: usize) -> DispatcherTransformSample {
        DispatcherTransformSample {
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

/// Double-buffered transform storage skeleton.
///
/// Render reads slot N while physics writes slot N+1; publication swaps the slots
/// atomically once integration completes.
pub struct DispatcherDoubleBufferedTransforms {
    buffers: [TransformLaneBuffer; 2],
    capacity: usize,
    render_read_slot: AtomicUsize,
    physics_write_slot: AtomicUsize,
    latest_published_frame: AtomicU64,
}

impl DispatcherDoubleBufferedTransforms {
    /// Allocate the fixed-capacity double buffer.
    pub fn new(capacity: usize) -> Self {
        Self {
            buffers: [
                TransformLaneBuffer::new(capacity),
                TransformLaneBuffer::new(capacity),
            ],
            capacity,
            render_read_slot: AtomicUsize::new(0),
            physics_write_slot: AtomicUsize::new(1),
            latest_published_frame: AtomicU64::new(0),
        }
    }

    /// Number of transforms available in each slot.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Slot currently visible to render-side code.
    pub fn render_read_slot(&self) -> usize {
        self.render_read_slot.load(Ordering::Acquire)
    }

    /// Slot currently reserved for the next physics write.
    pub fn physics_write_slot(&self) -> usize {
        self.physics_write_slot.load(Ordering::Acquire)
    }

    /// Latest frame id published into the render-visible slot.
    pub fn latest_published_frame(&self) -> Option<u64> {
        let frame_id = self.latest_published_frame.load(Ordering::Acquire);
        (frame_id != 0).then_some(frame_id)
    }

    /// Write into the active physics slot.
    pub fn write_physics_transform(&self, index: usize, sample: DispatcherTransformSample) {
        self.write_transform_to_slot(self.physics_write_slot(), index, sample);
    }

    /// Write into an explicit slot.
    pub fn write_transform_to_slot(
        &self,
        slot: usize,
        index: usize,
        sample: DispatcherTransformSample,
    ) {
        if slot > 1 || index >= self.capacity {
            return;
        }
        self.buffers[slot].write(index, sample);
    }

    /// Read from the render-visible slot.
    pub fn read_render_transform(&self, index: usize) -> DispatcherTransformSample {
        self.read_transform_from_slot(self.render_read_slot(), index)
    }

    /// Read from an explicit slot.
    pub fn read_transform_from_slot(&self, slot: usize, index: usize) -> DispatcherTransformSample {
        if slot > 1 || index >= self.capacity {
            return DispatcherTransformSample::default();
        }
        self.buffers[slot].read(index)
    }

    fn publish_completed_frame(&self, frame_id: u64) -> usize {
        let previous_render_slot = self.render_read_slot();
        let completed_write_slot = self.physics_write_slot();
        self.render_read_slot
            .store(completed_write_slot, Ordering::Release);
        self.physics_write_slot
            .store(previous_render_slot, Ordering::Release);
        self.latest_published_frame
            .store(frame_id, Ordering::Release);
        completed_write_slot
    }
}

/// Execution phase for overlapped frame dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispatcherPhase {
    /// Main/render thread is building scene data from the previously published slot.
    SceneBuild,
    /// Physics broadphase for frame N+1.
    Broadphase,
    /// Physics narrowphase for frame N+1.
    Narrowphase,
    /// Physics integration/state write-back for frame N+1.
    Integration,
}

impl DispatcherPhase {
    fn next_physics(self) -> Option<Self> {
        match self {
            Self::Broadphase => Some(Self::Narrowphase),
            Self::Narrowphase => Some(Self::Integration),
            Self::Integration | Self::SceneBuild => None,
        }
    }
}

/// Worker-visible context for one dispatched chunk.
#[derive(Clone)]
pub struct DispatcherTaskContext {
    /// Monotonic frame id.
    pub frame_id: u64,
    /// Phase currently being processed.
    pub phase: DispatcherPhase,
    /// Linear chunk range assigned to the worker.
    pub work_range: Range<usize>,
    /// Slot still visible to render thread during this frame.
    pub render_read_slot: usize,
    /// Slot reserved for the active physics write.
    pub physics_write_slot: usize,
    /// Shared double-buffered transform storage.
    pub transforms: Arc<DispatcherDoubleBufferedTransforms>,
    /// Stable worker index.
    pub worker_index: usize,
    /// Logical core backing this worker.
    pub worker_core_id: usize,
    /// Topology class for the worker core.
    pub worker_class: CpuClass,
}

/// Phase handler type used by the dispatcher.
pub type DispatcherPhaseHandler = Arc<dyn Fn(&DispatcherTaskContext) + Send + Sync + 'static>;

/// Callback set for one overlapped physics frame.
#[derive(Clone, Default)]
pub struct DispatcherPhaseCallbacks {
    /// Broadphase chunk callback.
    pub broadphase: Option<DispatcherPhaseHandler>,
    /// Narrowphase chunk callback.
    pub narrowphase: Option<DispatcherPhaseHandler>,
    /// Integration chunk callback.
    pub integration: Option<DispatcherPhaseHandler>,
}

impl DispatcherPhaseCallbacks {
    /// Attach a broadphase callback.
    pub fn with_broadphase(mut self, handler: DispatcherPhaseHandler) -> Self {
        self.broadphase = Some(handler);
        self
    }

    /// Attach a narrowphase callback.
    pub fn with_narrowphase(mut self, handler: DispatcherPhaseHandler) -> Self {
        self.narrowphase = Some(handler);
        self
    }

    /// Attach an integration callback.
    pub fn with_integration(mut self, handler: DispatcherPhaseHandler) -> Self {
        self.integration = Some(handler);
        self
    }

    fn handler_for(&self, phase: DispatcherPhase) -> Option<DispatcherPhaseHandler> {
        match phase {
            DispatcherPhase::Broadphase => self.broadphase.clone(),
            DispatcherPhase::Narrowphase => self.narrowphase.clone(),
            DispatcherPhase::Integration => self.integration.clone(),
            DispatcherPhase::SceneBuild => None,
        }
    }
}

/// Trigger payload used to start physics N+1 while render thread builds scene N.
#[derive(Debug, Clone)]
pub struct PhysicsDispatchTrigger {
    /// Frame id assigned to the physics step.
    pub frame_id: u64,
    /// Broadphase item count.
    pub broadphase_items: usize,
    /// Narrowphase item count.
    pub narrowphase_items: usize,
    /// Integration item count.
    pub integration_items: usize,
    /// Chunk size override for this frame.
    pub chunk_size: usize,
}

impl PhysicsDispatchTrigger {
    /// Build a new frame trigger.
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

/// Ticket returned immediately after physics N+1 is armed.
#[derive(Debug, Clone, Copy)]
pub struct PhysicsDispatchTicket {
    /// Triggered frame id.
    pub frame_id: u64,
    /// Slot still safe for render thread reads.
    pub render_read_slot: usize,
    /// Slot reserved for N+1 physics writes.
    pub physics_write_slot: usize,
    /// Number of planned jobs across all phases.
    pub planned_job_count: usize,
}

/// Scene-build view for frame N while N+1 physics is in flight.
#[derive(Debug, Clone, Copy)]
pub struct SceneBuildTicket {
    /// Logical render frame id the main thread is building.
    pub frame_id: u64,
    /// Slot that remains render-visible during scene build.
    pub render_read_slot: usize,
    /// Latest physics frame already published to render.
    pub latest_published_physics_frame: Option<u64>,
}

/// Reconciled completion returned to the main thread after workers publish a frame.
#[derive(Debug, Clone, Copy)]
pub struct CompletedPhysicsFrame {
    /// Completed physics frame id.
    pub frame_id: u64,
    /// Slot that was visible while the frame was in flight.
    pub render_read_slot: usize,
    /// Slot that became the new render-visible slot when the frame published.
    pub published_render_slot: usize,
    /// Total planned jobs for the frame.
    pub planned_job_count: usize,
}

/// Runtime metrics for the bare-metal dispatcher.
#[derive(Debug, Clone, Copy, Default)]
pub struct TaskDispatcherMetrics {
    /// Worker count owned by the dispatcher.
    pub worker_count: usize,
    /// Jobs currently queued.
    pub queued_jobs: u64,
    /// Jobs currently in flight.
    pub in_flight_jobs: u64,
    /// Jobs completed since startup.
    pub completed_jobs: u64,
    /// Frames triggered since startup.
    pub triggered_frames: u64,
    /// Frames published since startup.
    pub published_frames: u64,
    /// Latest published frame id.
    pub latest_published_frame: Option<u64>,
    /// Active in-flight frame id.
    pub active_frame_id: Option<u64>,
}

/// Bare-metal task dispatcher configuration.
#[derive(Debug, Clone)]
pub struct TaskDispatcherConfig {
    /// CPU topology snapshot used for worker placement.
    pub topology: CpuTopology,
    /// Logical cores used by workers.
    pub worker_core_ids: Vec<usize>,
    /// Bounded queue capacity for chunk jobs.
    pub queue_capacity: usize,
    /// Fixed transform capacity per slot.
    pub transform_capacity: usize,
    /// Default chunk size when the trigger requests zero.
    pub default_chunk_size: usize,
    /// Number of spin iterations before yielding.
    pub spin_iterations: u32,
    /// Number of yields before futex wait.
    pub yield_iterations: u32,
    /// Nice value for performance workers.
    pub performance_nice: i32,
    /// Nice value for efficient workers.
    pub efficient_nice: i32,
    /// Nice value for unknown workers.
    pub unknown_nice: i32,
    /// Use strict one-core affinity on Linux.
    pub strict_affinity: bool,
    /// Attempt `SCHED_FIFO` on Linux for worker threads.
    pub enable_realtime_policy: bool,
}

impl Default for TaskDispatcherConfig {
    fn default() -> Self {
        let topology = CpuTopology::detect();
        Self {
            worker_core_ids: topology.preferred_core_ids(),
            topology,
            queue_capacity: 131_072,
            transform_capacity: 30_000,
            default_chunk_size: 256,
            spin_iterations: 2_048,
            yield_iterations: 64,
            performance_nice: -8,
            efficient_nice: -4,
            unknown_nice: -6,
            strict_affinity: true,
            enable_realtime_policy: true,
        }
    }
}

impl TaskDispatcherConfig {
    fn nice_value_for_class(&self, class: CpuClass) -> i32 {
        match class {
            CpuClass::Performance => self.performance_nice,
            CpuClass::Efficient => self.efficient_nice,
            CpuClass::Unknown => self.unknown_nice,
        }
    }
}

struct DispatchJob {
    frame: Arc<DispatchFrameState>,
    phase: DispatcherPhase,
    work_range: Range<usize>,
}

struct DispatchFrameState {
    frame_id: u64,
    render_read_slot: usize,
    physics_write_slot: usize,
    broadphase_items: usize,
    narrowphase_items: usize,
    integration_items: usize,
    chunk_size: usize,
    callbacks: DispatcherPhaseCallbacks,
    transforms: Arc<DispatcherDoubleBufferedTransforms>,
    planned_job_count: usize,
    broadphase_remaining: AtomicUsize,
    narrowphase_remaining: AtomicUsize,
    integration_remaining: AtomicUsize,
}

impl DispatchFrameState {
    fn new(
        trigger: PhysicsDispatchTrigger,
        callbacks: DispatcherPhaseCallbacks,
        transforms: Arc<DispatcherDoubleBufferedTransforms>,
        render_read_slot: usize,
        physics_write_slot: usize,
        default_chunk_size: usize,
    ) -> Self {
        let chunk_size = if trigger.chunk_size == 0 {
            default_chunk_size.max(1)
        } else {
            trigger.chunk_size.max(1)
        };
        let planned_job_count = phase_job_count(
            trigger.broadphase_items,
            chunk_size,
            callbacks.broadphase.is_some(),
        ) + phase_job_count(
            trigger.narrowphase_items,
            chunk_size,
            callbacks.narrowphase.is_some(),
        ) + phase_job_count(
            trigger.integration_items,
            chunk_size,
            callbacks.integration.is_some(),
        );

        Self {
            frame_id: trigger.frame_id,
            render_read_slot,
            physics_write_slot,
            broadphase_items: trigger.broadphase_items,
            narrowphase_items: trigger.narrowphase_items,
            integration_items: trigger.integration_items,
            chunk_size,
            callbacks,
            transforms,
            planned_job_count,
            broadphase_remaining: AtomicUsize::new(0),
            narrowphase_remaining: AtomicUsize::new(0),
            integration_remaining: AtomicUsize::new(0),
        }
    }

    fn item_count(&self, phase: DispatcherPhase) -> usize {
        match phase {
            DispatcherPhase::Broadphase => self.broadphase_items,
            DispatcherPhase::Narrowphase => self.narrowphase_items,
            DispatcherPhase::Integration => self.integration_items,
            DispatcherPhase::SceneBuild => 0,
        }
    }

    fn remaining_counter(&self, phase: DispatcherPhase) -> &AtomicUsize {
        match phase {
            DispatcherPhase::Broadphase => &self.broadphase_remaining,
            DispatcherPhase::Narrowphase => &self.narrowphase_remaining,
            DispatcherPhase::Integration => &self.integration_remaining,
            DispatcherPhase::SceneBuild => &self.integration_remaining,
        }
    }
}

fn phase_job_count(item_count: usize, chunk_size: usize, enabled: bool) -> usize {
    if enabled && item_count > 0 {
        item_count.div_ceil(chunk_size.max(1))
    } else {
        0
    }
}

/// Bare-metal overlapped dispatcher for Physics N+1 / SceneBuild N.
pub struct TaskDispatcher {
    topology: CpuTopology,
    worker_core_ids: Vec<usize>,
    transforms: Arc<DispatcherDoubleBufferedTransforms>,
    queue: Arc<ArrayQueue<DispatchJob>>,
    completed_frames: Arc<SegQueue<CompletedPhysicsFrame>>,
    signal: Arc<WorkerSignal>,
    queued_jobs: Arc<AtomicU64>,
    in_flight_jobs: Arc<AtomicU64>,
    completed_jobs: Arc<AtomicU64>,
    triggered_frames: Arc<AtomicU64>,
    published_frames: Arc<AtomicU64>,
    latest_published_frame: Arc<AtomicU64>,
    active_frame_id: Arc<AtomicU64>,
    default_chunk_size: usize,
    workers: Vec<JoinHandle<()>>,
}

impl TaskDispatcher {
    /// Create the bare-metal dispatcher and spawn pinned workers.
    pub fn new(config: TaskDispatcherConfig) -> DispatchResult<Self> {
        if config.worker_core_ids.is_empty() {
            return Err(DispatchError::Runtime(
                "task dispatcher requires at least one worker core".to_string(),
            ));
        }
        if config.queue_capacity == 0 || config.transform_capacity == 0 {
            return Err(DispatchError::Runtime(
                "queue_capacity and transform_capacity must be non-zero".to_string(),
            ));
        }

        let topology = config.topology.clone();
        let queue = Arc::new(ArrayQueue::new(config.queue_capacity));
        let completed_frames = Arc::new(SegQueue::new());
        let signal = Arc::new(WorkerSignal::default());
        let transforms = Arc::new(DispatcherDoubleBufferedTransforms::new(
            config.transform_capacity,
        ));
        let queued_jobs = Arc::new(AtomicU64::new(0));
        let in_flight_jobs = Arc::new(AtomicU64::new(0));
        let completed_jobs = Arc::new(AtomicU64::new(0));
        let triggered_frames = Arc::new(AtomicU64::new(0));
        let published_frames = Arc::new(AtomicU64::new(0));
        let latest_published_frame = Arc::new(AtomicU64::new(0));
        let active_frame_id = Arc::new(AtomicU64::new(NO_ACTIVE_FRAME));

        let mut workers = Vec::with_capacity(config.worker_core_ids.len());
        for (worker_index, &core_id) in config.worker_core_ids.iter().enumerate() {
            let class = topology.class_for_core(core_id);
            let mut launch = WorkerLaunchConfig::new(
                worker_index,
                core_id,
                class,
                format!("mps-dispatch-worker-{worker_index}-core-{core_id}-{class:?}"),
                config.nice_value_for_class(class),
                config.spin_iterations,
                config.yield_iterations,
            );
            launch.strict_affinity = config.strict_affinity;
            if !config.enable_realtime_policy {
                launch.realtime_priority = None;
            }

            let handle = spawn_worker(launch, Arc::clone(&signal), {
                let worker_queue = Arc::clone(&queue);
                let worker_completed_frames = Arc::clone(&completed_frames);
                let worker_signal = Arc::clone(&signal);
                let worker_queued_jobs = Arc::clone(&queued_jobs);
                let worker_in_flight_jobs = Arc::clone(&in_flight_jobs);
                let worker_completed_jobs = Arc::clone(&completed_jobs);
                let worker_published_frames = Arc::clone(&published_frames);
                let worker_latest_published_frame = Arc::clone(&latest_published_frame);
                let worker_active_frame_id = Arc::clone(&active_frame_id);
                move |launch, signal| {
                    dispatcher_worker_loop(
                        launch,
                        signal,
                        worker_queue,
                        worker_completed_frames,
                        worker_queued_jobs,
                        worker_in_flight_jobs,
                        worker_completed_jobs,
                        worker_published_frames,
                        worker_latest_published_frame,
                        worker_active_frame_id,
                        worker_signal,
                    );
                }
            })
            .map_err(|err| DispatchError::WorkerSpawn(err.to_string()))?;
            workers.push(handle);
        }

        Ok(Self {
            topology,
            worker_core_ids: config.worker_core_ids,
            transforms,
            queue,
            completed_frames,
            signal,
            queued_jobs,
            in_flight_jobs,
            completed_jobs,
            triggered_frames,
            published_frames,
            latest_published_frame,
            active_frame_id,
            default_chunk_size: config.default_chunk_size.max(1),
            workers,
        })
    }

    /// Access the topology snapshot used for worker placement.
    pub fn topology(&self) -> &CpuTopology {
        &self.topology
    }

    /// Shared transform storage used by render/physics overlap.
    pub fn transforms(&self) -> Arc<DispatcherDoubleBufferedTransforms> {
        Arc::clone(&self.transforms)
    }

    /// Acquire the render-side view for frame N while physics N+1 is in flight.
    pub fn acquire_scene_build(&self, frame_id: u64) -> SceneBuildTicket {
        SceneBuildTicket {
            frame_id,
            render_read_slot: self.transforms.render_read_slot(),
            latest_published_physics_frame: self.latest_published_frame(),
        }
    }

    /// Trigger physics N+1 without blocking the render thread.
    pub fn trigger_next_physics(
        &self,
        trigger: PhysicsDispatchTrigger,
        callbacks: DispatcherPhaseCallbacks,
    ) -> DispatchResult<PhysicsDispatchTicket> {
        match self.active_frame_id.compare_exchange(
            NO_ACTIVE_FRAME,
            trigger.frame_id,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => {}
            Err(active_frame_id) => {
                return Err(DispatchError::FrameAlreadyInFlight(active_frame_id));
            }
        }

        let render_read_slot = self.transforms.render_read_slot();
        let physics_write_slot = self.transforms.physics_write_slot();
        let frame = Arc::new(DispatchFrameState::new(
            trigger.clone(),
            callbacks,
            Arc::clone(&self.transforms),
            render_read_slot,
            physics_write_slot,
            self.default_chunk_size,
        ));
        self.triggered_frames.fetch_add(1, Ordering::AcqRel);

        if frame.planned_job_count == 0 {
            self.publish_completed_frame(&frame);
        } else {
            self.schedule_phase(&frame, DispatcherPhase::Broadphase)?;
        }

        Ok(PhysicsDispatchTicket {
            frame_id: trigger.frame_id,
            render_read_slot,
            physics_write_slot,
            planned_job_count: frame.planned_job_count,
        })
    }

    /// Pop the next completed physics frame that became visible to render.
    pub fn try_pop_completed_frame(&self) -> Option<CompletedPhysicsFrame> {
        self.completed_frames.pop()
    }

    /// Wait for a specific frame to publish, using spin/yield only on the main thread.
    pub fn wait_for_completed_frame(&self, frame_id: u64, timeout: Duration) -> bool {
        let started = Instant::now();
        while started.elapsed() <= timeout {
            if self
                .latest_published_frame()
                .is_some_and(|completed_frame| completed_frame >= frame_id)
            {
                return true;
            }
            std::hint::spin_loop();
            std::thread::yield_now();
        }
        false
    }

    /// Snapshot dispatcher metrics.
    pub fn metrics(&self) -> TaskDispatcherMetrics {
        TaskDispatcherMetrics {
            worker_count: self.worker_core_ids.len(),
            queued_jobs: self.queued_jobs.load(Ordering::Acquire),
            in_flight_jobs: self.in_flight_jobs.load(Ordering::Acquire),
            completed_jobs: self.completed_jobs.load(Ordering::Acquire),
            triggered_frames: self.triggered_frames.load(Ordering::Acquire),
            published_frames: self.published_frames.load(Ordering::Acquire),
            latest_published_frame: self.latest_published_frame(),
            active_frame_id: match self.active_frame_id.load(Ordering::Acquire) {
                NO_ACTIVE_FRAME => None,
                frame_id => Some(frame_id),
            },
        }
    }

    /// Latest frame published to the render-visible slot.
    pub fn latest_published_frame(&self) -> Option<u64> {
        let frame_id = self.latest_published_frame.load(Ordering::Acquire);
        (frame_id != 0).then_some(frame_id)
    }

    fn schedule_phase(
        &self,
        frame: &Arc<DispatchFrameState>,
        phase: DispatcherPhase,
    ) -> DispatchResult {
        if frame.callbacks.handler_for(phase).is_none() || frame.item_count(phase) == 0 {
            match phase.next_physics() {
                Some(next_phase) => self.schedule_phase(frame, next_phase),
                None => {
                    self.publish_completed_frame(frame);
                    Ok(())
                }
            }
        } else {
            schedule_frame_phase(
                frame,
                phase,
                &self.queue,
                &self.queued_jobs,
                &self.in_flight_jobs,
                &self.signal,
            )
        }
    }

    fn publish_completed_frame(&self, frame: &DispatchFrameState) {
        let published_render_slot = frame.transforms.publish_completed_frame(frame.frame_id);
        self.latest_published_frame
            .store(frame.frame_id, Ordering::Release);
        self.published_frames.fetch_add(1, Ordering::AcqRel);
        self.active_frame_id
            .compare_exchange(
                frame.frame_id,
                NO_ACTIVE_FRAME,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .ok();
        self.completed_frames.push(CompletedPhysicsFrame {
            frame_id: frame.frame_id,
            render_read_slot: frame.render_read_slot,
            published_render_slot,
            planned_job_count: frame.planned_job_count,
        });
        self.signal.wake_all();
    }
}

impl Drop for TaskDispatcher {
    fn drop(&mut self) {
        self.signal.request_shutdown();
        while let Some(handle) = self.workers.pop() {
            let _ = handle.join();
        }
    }
}

fn schedule_frame_phase(
    frame: &Arc<DispatchFrameState>,
    phase: DispatcherPhase,
    queue: &ArrayQueue<DispatchJob>,
    queued_jobs: &AtomicU64,
    in_flight_jobs: &AtomicU64,
    signal: &WorkerSignal,
) -> DispatchResult {
    let item_count = frame.item_count(phase);
    let chunk_size = frame.chunk_size.max(1);
    let job_count = item_count.div_ceil(chunk_size);
    frame
        .remaining_counter(phase)
        .store(job_count, Ordering::Release);

    queued_jobs.fetch_add(job_count as u64, Ordering::AcqRel);
    in_flight_jobs.fetch_add(job_count as u64, Ordering::AcqRel);

    for job_index in 0..job_count {
        let start = job_index * chunk_size;
        let end = (start + chunk_size).min(item_count);
        push_dispatch_job(
            queue,
            DispatchJob {
                frame: Arc::clone(frame),
                phase,
                work_range: start..end,
            },
            signal,
        )?;
    }
    signal.wake_all();
    Ok(())
}

fn push_dispatch_job(
    queue: &ArrayQueue<DispatchJob>,
    mut job: DispatchJob,
    signal: &WorkerSignal,
) -> DispatchResult {
    let mut spin_count = 0u32;
    loop {
        match queue.push(job) {
            Ok(()) => return Ok(()),
            Err(returned_job) => {
                job = returned_job;
                spin_count = spin_count.saturating_add(1);
                if spin_count <= 64 {
                    std::hint::spin_loop();
                } else if spin_count <= 256 {
                    std::thread::yield_now();
                } else {
                    return Err(DispatchError::QueueSaturated(
                        "dispatcher queue remained saturated while scheduling phase".to_string(),
                    ));
                }
                signal.wake_one();
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn dispatcher_worker_loop(
    launch: WorkerLaunchConfig,
    signal: Arc<WorkerSignal>,
    queue: Arc<ArrayQueue<DispatchJob>>,
    completed_frames: Arc<SegQueue<CompletedPhysicsFrame>>,
    queued_jobs: Arc<AtomicU64>,
    in_flight_jobs: Arc<AtomicU64>,
    completed_jobs: Arc<AtomicU64>,
    published_frames: Arc<AtomicU64>,
    latest_published_frame: Arc<AtomicU64>,
    active_frame_id: Arc<AtomicU64>,
    wake_signal: Arc<WorkerSignal>,
) {
    let mut observed_epoch = signal.observed_epoch();

    loop {
        if let Some(job) = queue.pop() {
            queued_jobs.fetch_sub(1, Ordering::AcqRel);
            execute_dispatch_job(
                &launch,
                job,
                &queue,
                &completed_frames,
                &queued_jobs,
                &in_flight_jobs,
                &completed_jobs,
                &published_frames,
                &latest_published_frame,
                &active_frame_id,
                &wake_signal,
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
fn execute_dispatch_job(
    launch: &WorkerLaunchConfig,
    job: DispatchJob,
    queue: &ArrayQueue<DispatchJob>,
    completed_frames: &SegQueue<CompletedPhysicsFrame>,
    queued_jobs: &AtomicU64,
    in_flight_jobs: &AtomicU64,
    completed_jobs: &AtomicU64,
    published_frames: &AtomicU64,
    latest_published_frame: &AtomicU64,
    active_frame_id: &AtomicU64,
    signal: &WorkerSignal,
) {
    if let Some(handler) = job.frame.callbacks.handler_for(job.phase) {
        let context = DispatcherTaskContext {
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
        match job.phase.next_physics() {
            Some(next_phase) => {
                let _ = schedule_followup_phase_from_worker(
                    &job.frame,
                    next_phase,
                    queue,
                    queued_jobs,
                    in_flight_jobs,
                    signal,
                    completed_frames,
                    published_frames,
                    latest_published_frame,
                    active_frame_id,
                );
            }
            None => publish_completed_frame_from_worker(
                &job.frame,
                completed_frames,
                published_frames,
                latest_published_frame,
                active_frame_id,
                signal,
            ),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn schedule_followup_phase_from_worker(
    frame: &Arc<DispatchFrameState>,
    phase: DispatcherPhase,
    queue: &ArrayQueue<DispatchJob>,
    queued_jobs: &AtomicU64,
    in_flight_jobs: &AtomicU64,
    signal: &WorkerSignal,
    completed_frames: &SegQueue<CompletedPhysicsFrame>,
    published_frames: &AtomicU64,
    latest_published_frame: &AtomicU64,
    active_frame_id: &AtomicU64,
) -> DispatchResult {
    if frame.callbacks.handler_for(phase).is_none() || frame.item_count(phase) == 0 {
        match phase.next_physics() {
            Some(next_phase) => schedule_followup_phase_from_worker(
                frame,
                next_phase,
                queue,
                queued_jobs,
                in_flight_jobs,
                signal,
                completed_frames,
                published_frames,
                latest_published_frame,
                active_frame_id,
            ),
            None => {
                publish_completed_frame_from_worker(
                    frame,
                    completed_frames,
                    published_frames,
                    latest_published_frame,
                    active_frame_id,
                    signal,
                );
                Ok(())
            }
        }
    } else {
        schedule_frame_phase(frame, phase, queue, queued_jobs, in_flight_jobs, signal)
    }
}

fn publish_completed_frame_from_worker(
    frame: &DispatchFrameState,
    completed_frames: &SegQueue<CompletedPhysicsFrame>,
    published_frames: &AtomicU64,
    latest_published_frame: &AtomicU64,
    active_frame_id: &AtomicU64,
    signal: &WorkerSignal,
) {
    let published_render_slot = frame.transforms.publish_completed_frame(frame.frame_id);
    latest_published_frame.store(frame.frame_id, Ordering::Release);
    published_frames.fetch_add(1, Ordering::AcqRel);
    active_frame_id
        .compare_exchange(
            frame.frame_id,
            NO_ACTIVE_FRAME,
            Ordering::AcqRel,
            Ordering::Acquire,
        )
        .ok();
    completed_frames.push(CompletedPhysicsFrame {
        frame_id: frame.frame_id,
        render_read_slot: frame.render_read_slot,
        published_render_slot,
        planned_job_count: frame.planned_job_count,
    });
    signal.wake_all();
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn trigger_next_physics_publishes_completed_frame() {
        let dispatcher = TaskDispatcher::new(TaskDispatcherConfig {
            worker_core_ids: vec![0],
            transform_capacity: 32,
            queue_capacity: 128,
            enable_realtime_policy: false,
            ..TaskDispatcherConfig::default()
        })
        .expect("dispatcher");

        let broadphase_calls = Arc::new(AtomicUsize::new(0));
        let narrowphase_calls = Arc::new(AtomicUsize::new(0));
        let integration_calls = Arc::new(AtomicUsize::new(0));

        let broadphase_counter = Arc::clone(&broadphase_calls);
        let narrowphase_counter = Arc::clone(&narrowphase_calls);
        let integration_counter = Arc::clone(&integration_calls);

        let callbacks = DispatcherPhaseCallbacks::default()
            .with_broadphase(Arc::new(move |_ctx| {
                broadphase_counter.fetch_add(1, Ordering::AcqRel);
            }))
            .with_narrowphase(Arc::new(move |_ctx| {
                narrowphase_counter.fetch_add(1, Ordering::AcqRel);
            }))
            .with_integration(Arc::new(move |ctx| {
                ctx.transforms.write_physics_transform(
                    ctx.work_range.start,
                    DispatcherTransformSample {
                        position: [1.0, 2.0, 3.0],
                        rotation: [0.0, 0.0, 0.0, 1.0],
                    },
                );
                integration_counter.fetch_add(1, Ordering::AcqRel);
            }));

        let ticket = dispatcher
            .trigger_next_physics(PhysicsDispatchTrigger::new(1, 8, 4, 8, 4), callbacks)
            .expect("trigger");
        assert_eq!(ticket.frame_id, 1);
        assert!(dispatcher.wait_for_completed_frame(1, Duration::from_millis(250)));
        let mut completed = None;
        for _ in 0..128 {
            if let Some(frame) = dispatcher.try_pop_completed_frame() {
                completed = Some(frame);
                break;
            }
            std::thread::yield_now();
        }
        let completed = completed.expect("completed frame");
        assert_eq!(completed.frame_id, 1);
        assert!(broadphase_calls.load(Ordering::Acquire) > 0);
        assert!(narrowphase_calls.load(Ordering::Acquire) > 0);
        assert!(integration_calls.load(Ordering::Acquire) > 0);
        let sample = dispatcher.transforms().read_render_transform(0);
        assert_eq!(sample.position, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn scene_build_ticket_exposes_render_slot_without_blocking() {
        let dispatcher = TaskDispatcher::new(TaskDispatcherConfig {
            worker_core_ids: vec![0],
            transform_capacity: 8,
            queue_capacity: 64,
            enable_realtime_policy: false,
            ..TaskDispatcherConfig::default()
        })
        .expect("dispatcher");

        let ticket = dispatcher.acquire_scene_build(9);
        assert_eq!(ticket.frame_id, 9);
        assert_eq!(ticket.render_read_slot, 0);
        assert_eq!(ticket.latest_published_physics_frame, None);
    }
}
