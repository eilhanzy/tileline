//! Async physics step runner via MPS (Multi-Processing Scaler).
//!
//! The runner now targets the bare-metal `TaskDispatcher` path so runtime
//! overlap semantics match the low-latency MPS frame pipeline. The world stays
//! behind `Arc<Mutex<_>>` for now so the rest of TLApp can keep its current
//! synchronous borrow flow while we migrate internal world phases away from the
//! remaining single-step hot path.

use mps::{
    DispatcherPhaseCallbacks, MpsThreadPoolMetrics, PhysicsDispatchTrigger, TaskDispatcher,
    TaskDispatcherConfig,
};
use paradoxpe::PhysicsWorld;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex, MutexGuard};
use std::time::Duration;

/// Completion handle for an async physics step.
///
/// Call [`wait`] before reading or writing the `PhysicsWorld` again.
///
/// [`wait`]: PhysicsStepToken::wait
pub struct PhysicsStepToken {
    rx: mpsc::Receiver<u32>,
    dispatcher: Arc<TaskDispatcher>,
    frame_id: u64,
    await_publish: bool,
}

impl PhysicsStepToken {
    /// Block until the physics step completes and return the substep count.
    pub fn wait(self) -> u32 {
        let substeps = self.rx.recv().unwrap_or(0);
        if self.await_publish
            && !self
                .dispatcher
                .wait_for_completed_frame(self.frame_id, Duration::from_millis(250))
        {
            eprintln!(
                "[physics mps] frame {} finished stepping but did not publish within 250 ms",
                self.frame_id
            );
        }
        substeps
    }
}

/// Wraps a `PhysicsWorld` for async MPS-based stepping.
///
/// All non-step access happens synchronously via [`borrow`] / [`borrow_mut`].
/// [`step_begin`] triggers a one-frame physics job on the custom MPS pool and
/// returns immediately. The caller must call [`PhysicsStepToken::wait`] before
/// touching the world again.
///
/// [`borrow`]: PhysicsMpsRunner::borrow
/// [`borrow_mut`]: PhysicsMpsRunner::borrow_mut
/// [`step_begin`]: PhysicsMpsRunner::step_begin
pub struct PhysicsMpsRunner {
    world: Arc<Mutex<PhysicsWorld>>,
    dispatcher: Arc<TaskDispatcher>,
    next_frame_id: AtomicU64,
}

impl PhysicsMpsRunner {
    /// Create a runner from an existing world. Spawns the MPS worker pool.
    pub fn new(world: PhysicsWorld) -> Self {
        let mut dispatcher_config = TaskDispatcherConfig::default();
        dispatcher_config.transform_capacity =
            dispatcher_config.transform_capacity.max(world.body_count());
        let dispatcher = Arc::new(
            TaskDispatcher::new(dispatcher_config)
                .expect("failed to create bare-metal MPS task dispatcher"),
        );
        let transforms = dispatcher.transforms();
        world.write_render_transforms_to_dispatcher_storage(
            transforms.as_ref(),
            transforms.render_read_slot(),
        );
        Self {
            world: Arc::new(Mutex::new(world)),
            dispatcher,
            next_frame_id: AtomicU64::new(1),
        }
    }

    /// Borrow the world. Blocks only if a step is currently in flight.
    #[inline]
    pub fn borrow(&self) -> MutexGuard<'_, PhysicsWorld> {
        self.world.lock().unwrap()
    }

    /// Mutably borrow the world. Blocks only if a step is currently in flight.
    #[inline]
    pub fn borrow_mut(&self) -> MutexGuard<'_, PhysicsWorld> {
        self.world.lock().unwrap()
    }

    /// Replace the inner world (e.g. for simulation reset).
    ///
    /// The caller must drain any pending [`PhysicsStepToken`] first.
    pub fn replace(&self, new_world: PhysicsWorld) {
        let mut guard = self.world.lock().unwrap();
        *guard = new_world;
        let transforms = self.dispatcher.transforms();
        guard.write_render_transforms_to_dispatcher_storage(
            transforms.as_ref(),
            transforms.render_read_slot(),
        );
    }

    /// Snapshot runtime-visible metrics from the bare-metal dispatcher.
    pub fn thread_pool_metrics(&self) -> MpsThreadPoolMetrics {
        let metrics = self.dispatcher.metrics();
        MpsThreadPoolMetrics {
            worker_count: metrics.worker_count,
            queued_jobs: metrics.queued_jobs,
            in_flight_jobs: metrics.in_flight_jobs,
            completed_jobs: metrics.completed_jobs,
            completed_frames: metrics.published_frames,
            latest_completed_frame: metrics.latest_published_frame,
            active_frame_id: metrics.active_frame_id,
        }
    }

    /// Submit `world.step(dt)` on the bare-metal MPS dispatcher.
    ///
    /// Returns immediately. Call [`PhysicsStepToken::wait`] before touching
    /// the world again.
    pub fn step_begin(&self, dt: f32) -> PhysicsStepToken {
        let (tx, rx) = mpsc::sync_channel(1);
        let fallback_tx = tx.clone();
        let world = Arc::clone(&self.world);
        let frame_id = self.next_frame_id.fetch_add(1, Ordering::Relaxed);
        let planned_substeps = Arc::new(AtomicU32::new(0));
        let planned_fixed_dt_bits = Arc::new(AtomicU32::new(0));
        let planned_sleep_stride = Arc::new(AtomicUsize::new(1));
        let planned_valid = Arc::new(AtomicBool::new(false));

        let prepare_world = Arc::clone(&world);
        let prepare_substeps = Arc::clone(&planned_substeps);
        let prepare_fixed_dt_bits = Arc::clone(&planned_fixed_dt_bits);
        let prepare_sleep_stride = Arc::clone(&planned_sleep_stride);
        let prepare_valid = Arc::clone(&planned_valid);
        let broadphase = Arc::new(move |ctx: &mps::DispatcherTaskContext| {
            let result = catch_unwind(AssertUnwindSafe(|| {
                let mut world = prepare_world.lock().unwrap();
                if let Some(plan) = world.prepare_step_execution(dt) {
                    prepare_substeps.store(plan.substeps, Ordering::Release);
                    prepare_fixed_dt_bits.store(plan.fixed_dt.to_bits(), Ordering::Release);
                    prepare_sleep_stride.store(plan.sleep_update_stride, Ordering::Release);
                    prepare_valid.store(true, Ordering::Release);
                } else {
                    prepare_valid.store(false, Ordering::Release);
                    prepare_substeps.store(0, Ordering::Release);
                }
            }));
            if result.is_err() {
                eprintln!(
                    "[physics mps] frame {} panicked while preparing step execution plan",
                    ctx.frame_id
                );
                prepare_valid.store(false, Ordering::Release);
                prepare_substeps.store(0, Ordering::Release);
            }
        });

        let integration_world = Arc::clone(&world);
        let integration_substeps = Arc::clone(&planned_substeps);
        let integration_fixed_dt_bits = Arc::clone(&planned_fixed_dt_bits);
        let integration_sleep_stride = Arc::clone(&planned_sleep_stride);
        let integration_valid = Arc::clone(&planned_valid);
        let integration_tx = tx;
        let integration = Arc::new(move |ctx: &mps::DispatcherTaskContext| {
            if !integration_valid.load(Ordering::Acquire) {
                let _ = integration_tx.send(0);
                return;
            }

            let substeps = integration_substeps.load(Ordering::Acquire);
            if substeps == 0 {
                let _ = integration_tx.send(0);
                return;
            }

            let fixed_dt = f32::from_bits(integration_fixed_dt_bits.load(Ordering::Acquire));
            let sleep_update_stride = integration_sleep_stride.load(Ordering::Acquire).max(1);

            let result = catch_unwind(AssertUnwindSafe(|| {
                let mut world = integration_world.lock().unwrap();
                let plan = paradoxpe::PhysicsStepExecutionPlan {
                    substeps,
                    fixed_dt,
                    sleep_update_stride,
                };
                let completed_substeps = world.step_with_execution_plan(plan);
                world.write_render_transforms_to_dispatcher_storage(
                    ctx.transforms.as_ref(),
                    ctx.physics_write_slot,
                );
                completed_substeps
            }));

            match result {
                Ok(completed_substeps) => {
                    let _ = integration_tx.send(completed_substeps);
                }
                Err(_) => {
                    eprintln!(
                        "[physics mps] frame {} panicked during planned step execution; returning 0 substeps",
                        ctx.frame_id
                    );
                    let _ = integration_tx.send(0);
                }
            }
        });

        let callbacks = DispatcherPhaseCallbacks::default()
            .with_broadphase(broadphase)
            .with_integration(integration);
        let trigger = PhysicsDispatchTrigger::new(frame_id, 1, 0, 1, 1);
        let await_publish =
            if let Err(err) = self.dispatcher.trigger_next_physics(trigger, callbacks) {
                eprintln!(
                    "[physics mps] failed to trigger physics frame {}: {:?}",
                    frame_id, err
                );
                let _ = fallback_tx.send(0);
                false
            } else {
                true
            };
        PhysicsStepToken {
            rx,
            dispatcher: Arc::clone(&self.dispatcher),
            frame_id,
            await_publish,
        }
    }
}
