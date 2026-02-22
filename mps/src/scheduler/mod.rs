//! MPS scheduler.
//!
//! Goals of this implementation:
//! - lock-free queueing and worker coordination
//! - topology-aware priority routing (P-core/E-core)
//! - WASM execution through the dispatcher
//! - make -j$(nproc)-style saturation by default

pub mod dispatcher;
pub mod queue;

use crate::balancer::{CorePreference, LoadBalancer, TaskPriority};
use crate::topology::{CpuClass, CpuTopology};
use crossbeam::utils::Backoff;
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

pub use dispatcher::{DispatchError, DispatchResult, Dispatcher};
pub use queue::{PriorityTaskQueue, QueueDepth};

/// Unique task identifier.
pub type TaskId = u64;

/// Native task payload type.
pub type NativeTask = Box<dyn FnOnce() + Send + 'static>;

/// WASM task payload metadata.
#[derive(Debug, Clone)]
pub struct WasmTask {
    /// Raw WASM module bytes.
    pub module_bytes: Arc<[u8]>,
    /// Exported function name to invoke.
    pub entrypoint: String,
    /// i64 arguments marshalled as WASM `Value::I64`.
    pub args: Vec<i64>,
}

impl WasmTask {
    /// Build a WASM task from module bytes and an entrypoint.
    pub fn new(module_bytes: impl Into<Arc<[u8]>>, entrypoint: impl Into<String>) -> Self {
        Self {
            module_bytes: module_bytes.into(),
            entrypoint: entrypoint.into(),
            args: Vec::new(),
        }
    }

    /// Attach i64 arguments.
    pub fn with_args(mut self, args: Vec<i64>) -> Self {
        self.args = args;
        self
    }
}

/// Executable payload for a scheduled task.
pub enum TaskPayload {
    Native(NativeTask),
    Wasm(WasmTask),
}

/// Full queue envelope used by the scheduler.
pub struct TaskEnvelope {
    pub id: TaskId,
    pub priority: TaskPriority,
    pub preferred_class: CpuClass,
    pub spill_to_any: bool,
    pub payload: TaskPayload,
    pub submitted_at: Instant,
}

impl TaskEnvelope {
    fn new(
        id: TaskId,
        priority: TaskPriority,
        preferred_class: CpuClass,
        spill_to_any: bool,
        payload: TaskPayload,
    ) -> Self {
        Self {
            id,
            priority,
            preferred_class,
            spill_to_any,
            payload,
            submitted_at: Instant::now(),
        }
    }
}

/// Runtime counters for scheduler health checks.
#[derive(Debug, Clone, Copy, Default)]
pub struct SchedulerMetrics {
    pub submitted: u64,
    pub completed: u64,
    pub failed: u64,
    pub queue_depth: QueueDepth,
}

/// Lock-free, topology-aware scheduler for MPS phase 1.
pub struct MpsScheduler {
    topology: CpuTopology,
    balancer: LoadBalancer,
    queue: PriorityTaskQueue,
    shutdown: Arc<AtomicBool>,
    submitted: Arc<AtomicU64>,
    completed: Arc<AtomicU64>,
    failed: Arc<AtomicU64>,
    next_task_id: AtomicU64,
    workers: Vec<JoinHandle<()>>,
}

impl MpsScheduler {
    /// Create a scheduler using detected topology and full logical core parallelism.
    pub fn new() -> Self {
        Self::with_topology(CpuTopology::detect())
    }

    /// Create a scheduler using a caller-provided topology snapshot.
    pub fn with_topology(topology: CpuTopology) -> Self {
        let balancer = LoadBalancer::new(topology.clone());
        let queue = PriorityTaskQueue::new();
        let dispatcher = Arc::new(Dispatcher::new());
        let shutdown = Arc::new(AtomicBool::new(false));
        let submitted = Arc::new(AtomicU64::new(0));
        let completed = Arc::new(AtomicU64::new(0));
        let failed = Arc::new(AtomicU64::new(0));

        let workers = spawn_workers(
            &topology,
            queue.clone(),
            Arc::clone(&dispatcher),
            Arc::clone(&shutdown),
            Arc::clone(&completed),
            Arc::clone(&failed),
        );

        Self {
            topology,
            balancer,
            queue,
            shutdown,
            submitted,
            completed,
            failed,
            next_task_id: AtomicU64::new(1),
            workers,
        }
    }

    /// Read-only access to topology metadata.
    pub fn topology(&self) -> &CpuTopology {
        &self.topology
    }

    /// Submit a native Rust closure.
    pub fn submit_native<F>(
        &self,
        priority: TaskPriority,
        preference: CorePreference,
        task: F,
    ) -> TaskId
    where
        F: FnOnce() + Send + 'static,
    {
        self.submit_native_boxed(priority, preference, Box::new(task))
    }

    /// Submit a boxed native task.
    pub fn submit_native_boxed(
        &self,
        priority: TaskPriority,
        preference: CorePreference,
        task: NativeTask,
    ) -> TaskId {
        self.enqueue_payload(priority, preference, TaskPayload::Native(task))
    }

    /// Submit a WASM task.
    pub fn submit_wasm(
        &self,
        priority: TaskPriority,
        preference: CorePreference,
        task: WasmTask,
    ) -> TaskId {
        self.enqueue_payload(priority, preference, TaskPayload::Wasm(task))
    }

    /// Submit native tasks in parallel using Rayon.
    pub fn submit_batch_native(
        &self,
        priority: TaskPriority,
        preference: CorePreference,
        tasks: Vec<NativeTask>,
    ) -> Vec<TaskId> {
        let queue = self.queue.clone();
        let balancer = self.balancer.clone();
        let submitted = Arc::clone(&self.submitted);
        let next_task_id = &self.next_task_id;

        tasks
            .into_par_iter()
            .map(|task| {
                let id = next_task_id.fetch_add(1, Ordering::Relaxed);
                let decision = balancer.decide(priority, preference, queue.total_len());
                let envelope = TaskEnvelope::new(
                    id,
                    priority,
                    decision.preferred_class,
                    decision.spill_to_any,
                    TaskPayload::Native(task),
                );
                queue.push(envelope);
                submitted.fetch_add(1, Ordering::Relaxed);
                id
            })
            .collect()
    }

    /// Submit WASM tasks in parallel using Rayon.
    pub fn submit_batch_wasm(
        &self,
        priority: TaskPriority,
        preference: CorePreference,
        tasks: Vec<WasmTask>,
    ) -> Vec<TaskId> {
        let queue = self.queue.clone();
        let balancer = self.balancer.clone();
        let submitted = Arc::clone(&self.submitted);
        let next_task_id = &self.next_task_id;

        tasks
            .into_par_iter()
            .map(|task| {
                let id = next_task_id.fetch_add(1, Ordering::Relaxed);
                let decision = balancer.decide(priority, preference, queue.total_len());
                let envelope = TaskEnvelope::new(
                    id,
                    priority,
                    decision.preferred_class,
                    decision.spill_to_any,
                    TaskPayload::Wasm(task),
                );
                queue.push(envelope);
                submitted.fetch_add(1, Ordering::Relaxed);
                id
            })
            .collect()
    }

    /// Wait until the queue is drained or a timeout expires.
    pub fn wait_for_idle(&self, timeout: Duration) -> bool {
        let started = Instant::now();
        while started.elapsed() <= timeout {
            let submitted = self.submitted.load(Ordering::Acquire);
            let finished =
                self.completed.load(Ordering::Acquire) + self.failed.load(Ordering::Acquire);
            if self.queue.is_empty() && finished >= submitted {
                return true;
            }
            thread::sleep(Duration::from_millis(1));
        }
        false
    }

    /// Return runtime counters.
    pub fn metrics(&self) -> SchedulerMetrics {
        SchedulerMetrics {
            submitted: self.submitted.load(Ordering::Acquire),
            completed: self.completed.load(Ordering::Acquire),
            failed: self.failed.load(Ordering::Acquire),
            queue_depth: self.queue.depth_snapshot(),
        }
    }

    /// Get queue depth snapshot directly.
    pub fn queue_depth(&self) -> QueueDepth {
        self.queue.depth_snapshot()
    }

    fn enqueue_payload(
        &self,
        priority: TaskPriority,
        preference: CorePreference,
        payload: TaskPayload,
    ) -> TaskId {
        let id = self.next_task_id.fetch_add(1, Ordering::Relaxed);
        let decision = self
            .balancer
            .decide(priority, preference, self.queue.total_len());
        let envelope = TaskEnvelope::new(
            id,
            priority,
            decision.preferred_class,
            decision.spill_to_any,
            payload,
        );

        self.queue.push(envelope);
        self.submitted.fetch_add(1, Ordering::Relaxed);
        id
    }
}

impl Drop for MpsScheduler {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Release);

        for worker in &self.workers {
            worker.thread().unpark();
        }

        while let Some(worker) = self.workers.pop() {
            let _ = worker.join();
        }
    }
}

fn spawn_workers(
    topology: &CpuTopology,
    queue: PriorityTaskQueue,
    dispatcher: Arc<Dispatcher>,
    shutdown: Arc<AtomicBool>,
    completed: Arc<AtomicU64>,
    failed: Arc<AtomicU64>,
) -> Vec<JoinHandle<()>> {
    let core_ids = topology.preferred_core_ids();
    let mut handles = Vec::with_capacity(core_ids.len());

    for (worker_index, core_id) in core_ids.into_iter().enumerate() {
        let worker_queue = queue.clone();
        let worker_dispatcher = Arc::clone(&dispatcher);
        let worker_shutdown = Arc::clone(&shutdown);
        let worker_completed = Arc::clone(&completed);
        let worker_failed = Arc::clone(&failed);
        let worker_class = topology.class_for_core(core_id);
        let worker_name = format!(
            "mps-worker-{worker_index}-core-{core_id}-{:?}",
            worker_class
        );

        let handle = thread::Builder::new()
            .name(worker_name)
            .spawn(move || {
                worker_loop(
                    worker_class,
                    worker_queue,
                    worker_dispatcher,
                    worker_shutdown,
                    worker_completed,
                    worker_failed,
                );
            })
            .expect("failed to spawn MPS worker thread");

        handles.push(handle);
    }

    handles
}

fn worker_loop(
    worker_class: CpuClass,
    queue: PriorityTaskQueue,
    dispatcher: Arc<Dispatcher>,
    shutdown: Arc<AtomicBool>,
    completed: Arc<AtomicU64>,
    failed: Arc<AtomicU64>,
) {
    let backoff = Backoff::new();

    loop {
        if let Some(task) = queue.pop_for_worker(worker_class) {
            backoff.reset();
            match dispatcher.execute(task.payload) {
                Ok(_) => {
                    completed.fetch_add(1, Ordering::Release);
                }
                Err(_) => {
                    failed.fetch_add(1, Ordering::Release);
                }
            }
            continue;
        }

        if shutdown.load(Ordering::Acquire) && queue.is_empty() {
            break;
        }

        if backoff.is_completed() {
            thread::park_timeout(Duration::from_micros(250));
        } else {
            backoff.snooze();
        }
    }
}
