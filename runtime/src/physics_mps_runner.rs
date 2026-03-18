//! Async physics step runner via MPS (Multi-Processing Scaler).
//!
//! Wraps a `PhysicsWorld` in `Arc<Mutex<>>` so that `world.step()` can be
//! submitted as a Critical-priority MPS task. The caller submits the step
//! before the GPU upload and waits for it at the start of the next frame,
//! overlapping ~2 ms of physics with ~9 ms of GPU upload.

use mps::{CorePreference, MpsScheduler, TaskPriority};
use paradoxpe::PhysicsWorld;
use std::sync::{mpsc, Arc, Mutex, MutexGuard};

/// Completion handle for an async physics step.
///
/// Call [`wait`] before reading or writing the `PhysicsWorld` again.
///
/// [`wait`]: PhysicsStepToken::wait
pub struct PhysicsStepToken {
    rx: mpsc::Receiver<u32>,
}

impl PhysicsStepToken {
    /// Block until the physics step completes and return the substep count.
    pub fn wait(self) -> u32 {
        self.rx.recv().unwrap_or(0)
    }
}

/// Wraps a `PhysicsWorld` for async MPS-based stepping.
///
/// All non-step access happens synchronously via [`borrow`] / [`borrow_mut`].
/// [`step_begin`] submits `world.step(dt)` as a Critical-priority MPS task
/// and returns immediately. The caller must call [`PhysicsStepToken::wait`]
/// before touching the world again.
///
/// [`borrow`]: PhysicsMpsRunner::borrow
/// [`borrow_mut`]: PhysicsMpsRunner::borrow_mut
/// [`step_begin`]: PhysicsMpsRunner::step_begin
pub struct PhysicsMpsRunner {
    world: Arc<Mutex<PhysicsWorld>>,
    scheduler: Arc<MpsScheduler>,
}

impl PhysicsMpsRunner {
    /// Create a runner from an existing world. Spawns the MPS worker pool.
    pub fn new(world: PhysicsWorld) -> Self {
        Self {
            world: Arc::new(Mutex::new(world)),
            scheduler: Arc::new(MpsScheduler::new()),
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
        *self.world.lock().unwrap() = new_world;
    }

    /// Submit `world.step(dt)` as a Critical-priority MPS task.
    ///
    /// Returns immediately. Call [`PhysicsStepToken::wait`] before touching
    /// the world again.
    pub fn step_begin(&self, dt: f32) -> PhysicsStepToken {
        let (tx, rx) = mpsc::sync_channel(1);
        let world = Arc::clone(&self.world);
        self.scheduler.submit_native(
            TaskPriority::Critical,
            CorePreference::Performance,
            move || {
                let substeps = world.lock().unwrap().step(dt);
                let _ = tx.send(substeps);
            },
        );
        PhysicsStepToken { rx }
    }
}
