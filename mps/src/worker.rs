//! Worker bootstrap helpers for the custom MPS physics thread pool.
//!
//! The goal of this module is to keep worker threads predictable on Linux:
//! - bind each worker to a specific logical core
//! - apply a per-thread nice value
//! - provide a low-latency wake primitive backed by futex on Linux
//! - avoid `park()` / `unpark()` in the hot path

use crate::topology::CpuClass;
use std::io;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Duration;

/// Linux scheduling policy requested for the worker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkerSchedulingPolicy {
    /// Keep the default time-sharing scheduler.
    Other,
    /// Request `SCHED_FIFO` for deterministic wake latency.
    Fifo,
}

/// Outcome of worker bootstrap system calls.
#[derive(Debug, Clone, Copy, Default)]
pub struct WorkerBootstrapReport {
    /// `true` when the thread affinity syscall succeeded.
    pub affinity_applied: bool,
    /// `true` when the nice priority syscall succeeded.
    pub nice_applied: bool,
    /// `true` when a real-time policy was successfully applied.
    pub realtime_applied: bool,
    /// Last observed Linux errno-style failure code, if any.
    pub last_os_error: Option<i32>,
}

/// Worker launch configuration passed to each spawned thread.
#[derive(Debug, Clone)]
pub struct WorkerLaunchConfig {
    /// Stable worker index inside the pool.
    pub worker_index: usize,
    /// Logical CPU id used for strict affinity binding.
    pub logical_core_id: usize,
    /// Topology class associated with the selected logical core.
    pub class: CpuClass,
    /// Human-readable thread name.
    pub thread_name: String,
    /// Desired Linux nice value for the worker.
    pub nice_value: i32,
    /// Optional Linux real-time priority for `SCHED_FIFO`.
    pub realtime_priority: Option<i32>,
    /// Requested Linux scheduling policy.
    pub scheduling_policy: WorkerSchedulingPolicy,
    /// When true, the worker is pinned to exactly one logical CPU.
    pub strict_affinity: bool,
    /// Number of tight spin iterations before yielding.
    pub spin_iterations: u32,
    /// Number of cooperative yields before the futex wait path.
    pub yield_iterations: u32,
}

impl WorkerLaunchConfig {
    /// Build a new worker launch configuration.
    pub fn new(
        worker_index: usize,
        logical_core_id: usize,
        class: CpuClass,
        thread_name: impl Into<String>,
        nice_value: i32,
        spin_iterations: u32,
        yield_iterations: u32,
    ) -> Self {
        Self {
            worker_index,
            logical_core_id,
            class,
            thread_name: thread_name.into(),
            nice_value,
            realtime_priority: default_realtime_priority_for_class(class),
            scheduling_policy: WorkerSchedulingPolicy::Fifo,
            strict_affinity: true,
            spin_iterations,
            yield_iterations,
        }
    }
}

fn default_realtime_priority_for_class(class: CpuClass) -> Option<i32> {
    #[cfg(target_os = "linux")]
    {
        Some(match class {
            CpuClass::Performance => 72,
            CpuClass::Unknown => 56,
            CpuClass::Efficient => 40,
        })
    }

    #[cfg(not(target_os = "linux"))]
    {
        let _ = class;
        None
    }
}

/// Lightweight worker wake primitive.
///
/// The signal uses an epoch counter so workers can cheaply observe whether
/// something changed before entering a blocking wait. On Linux the blocking
/// path uses futex, which is substantially lighter than `park()` / `unpark()`
/// for repeated micro-wakeups.
#[derive(Debug, Default)]
pub struct WorkerSignal {
    epoch: AtomicU32,
    shutdown_requested: AtomicBool,
}

impl WorkerSignal {
    /// Return the currently observed epoch.
    pub fn observed_epoch(&self) -> u32 {
        self.epoch.load(Ordering::Acquire)
    }

    /// Return whether the worker group is shutting down.
    pub fn is_shutdown_requested(&self) -> bool {
        self.shutdown_requested.load(Ordering::Acquire)
    }

    /// Request shutdown for all workers and wake them immediately.
    pub fn request_shutdown(&self) {
        self.shutdown_requested.store(true, Ordering::Release);
        self.wake_all();
    }

    /// Wake at least one sleeping worker.
    pub fn wake_one(&self) {
        self.epoch.fetch_add(1, Ordering::AcqRel);
        futex_wake(&self.epoch, 1);
    }

    /// Wake all sleeping workers.
    pub fn wake_all(&self) {
        self.epoch.fetch_add(1, Ordering::AcqRel);
        futex_wake(&self.epoch, i32::MAX);
    }

    /// Wait until work becomes visible or shutdown is requested.
    ///
    /// The wait path escalates from spin -> yield -> futex/sleep.
    pub fn wait_for_change(
        &self,
        observed_epoch: &mut u32,
        spin_iterations: u32,
        yield_iterations: u32,
    ) {
        for _ in 0..spin_iterations {
            if self.has_epoch_advanced(*observed_epoch) {
                *observed_epoch = self.observed_epoch();
                return;
            }
            std::hint::spin_loop();
        }

        for _ in 0..yield_iterations {
            if self.has_epoch_advanced(*observed_epoch) {
                *observed_epoch = self.observed_epoch();
                return;
            }
            thread::yield_now();
        }

        let expected = *observed_epoch;
        futex_wait(&self.epoch, expected, Duration::from_micros(250));
        *observed_epoch = self.observed_epoch();
    }

    fn has_epoch_advanced(&self, observed_epoch: u32) -> bool {
        self.is_shutdown_requested() || self.observed_epoch() != observed_epoch
    }
}

/// Spawn a worker thread and apply affinity / priority inside the thread body.
pub fn spawn_worker<F>(
    launch: WorkerLaunchConfig,
    signal: Arc<WorkerSignal>,
    entry: F,
) -> io::Result<JoinHandle<()>>
where
    F: FnOnce(WorkerLaunchConfig, Arc<WorkerSignal>) + Send + 'static,
{
    let thread_name = launch.thread_name.clone();
    thread::Builder::new().name(thread_name).spawn(move || {
        let bootstrap = apply_worker_affinity_and_priority(&launch);
        #[cfg(target_os = "linux")]
        if !bootstrap.affinity_applied
            || (!bootstrap.realtime_applied && launch.realtime_priority.is_some())
        {
            eprintln!(
                "[mps worker] '{}' bootstrap affinity={} rt={} nice={} errno={:?}",
                launch.thread_name,
                bootstrap.affinity_applied,
                bootstrap.realtime_applied,
                bootstrap.nice_applied,
                bootstrap.last_os_error
            );
        }
        entry(launch, signal);
    })
}

/// Apply core affinity and thread priority for the current worker thread.
pub fn apply_worker_affinity_and_priority(launch: &WorkerLaunchConfig) -> WorkerBootstrapReport {
    let mut report = WorkerBootstrapReport::default();
    report.affinity_applied =
        pin_current_thread_to_core(launch.logical_core_id, launch.strict_affinity)
            .inspect_err(|errno| report.last_os_error = Some(*errno))
            .is_ok();
    report.realtime_applied = apply_linux_realtime_policy(launch)
        .inspect_err(|errno| report.last_os_error = Some(*errno))
        .is_ok();
    report.nice_applied = set_current_thread_nice(launch.nice_value)
        .inspect_err(|errno| report.last_os_error = Some(*errno))
        .is_ok();
    report
}

#[cfg(target_os = "linux")]
fn pin_current_thread_to_core(core_id: usize, strict_affinity: bool) -> Result<(), i32> {
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        libc::CPU_SET(core_id, &mut set);
        if !strict_affinity {
            let total = libc::sysconf(libc::_SC_NPROCESSORS_ONLN).max(1) as usize;
            if core_id > 0 {
                libc::CPU_SET(core_id - 1, &mut set);
            }
            if core_id + 1 < total {
                libc::CPU_SET(core_id + 1, &mut set);
            }
        }
        let tid = current_linux_tid();
        let rc = libc::sched_setaffinity(tid, std::mem::size_of::<libc::cpu_set_t>(), &set);
        if rc == 0 {
            Ok(())
        } else {
            Err(errno_code())
        }
    }
}

#[cfg(not(target_os = "linux"))]
fn pin_current_thread_to_core(_core_id: usize, _strict_affinity: bool) -> Result<(), i32> {
    Ok(())
}

#[cfg(target_os = "linux")]
fn set_current_thread_nice(nice_value: i32) -> Result<(), i32> {
    unsafe {
        let tid = current_linux_tid() as u32;
        let rc = libc::setpriority(libc::PRIO_PROCESS, tid, nice_value);
        if rc == 0 {
            Ok(())
        } else {
            Err(errno_code())
        }
    }
}

#[cfg(not(target_os = "linux"))]
fn set_current_thread_nice(_nice_value: i32) -> Result<(), i32> {
    Ok(())
}

#[cfg(target_os = "linux")]
fn apply_linux_realtime_policy(launch: &WorkerLaunchConfig) -> Result<(), i32> {
    let Some(priority) = launch.realtime_priority else {
        return Ok(());
    };
    if !matches!(launch.scheduling_policy, WorkerSchedulingPolicy::Fifo) {
        return Ok(());
    }

    unsafe {
        let tid = current_linux_tid();
        let mut param = libc::sched_param {
            sched_priority: priority.clamp(1, 99),
        };
        let rc = libc::sched_setscheduler(tid, libc::SCHED_FIFO, &mut param);
        if rc == 0 {
            Ok(())
        } else {
            Err(errno_code())
        }
    }
}

#[cfg(not(target_os = "linux"))]
fn apply_linux_realtime_policy(_launch: &WorkerLaunchConfig) -> Result<(), i32> {
    Ok(())
}

#[cfg(target_os = "linux")]
fn current_linux_tid() -> libc::pid_t {
    unsafe { libc::syscall(libc::SYS_gettid) as libc::pid_t }
}

#[cfg(target_os = "linux")]
fn errno_code() -> i32 {
    unsafe { *libc::__errno_location() }
}

#[cfg(target_os = "linux")]
fn futex_wait(epoch: &AtomicU32, expected: u32, timeout: Duration) {
    let timeout_spec = libc::timespec {
        tv_sec: timeout.as_secs() as libc::time_t,
        tv_nsec: timeout.subsec_nanos() as libc::c_long,
    };
    unsafe {
        let _ = libc::syscall(
            libc::SYS_futex,
            epoch as *const AtomicU32 as *const u32,
            libc::FUTEX_WAIT | libc::FUTEX_PRIVATE_FLAG,
            expected,
            &timeout_spec as *const libc::timespec,
        );
    }
}

#[cfg(not(target_os = "linux"))]
fn futex_wait(_epoch: &AtomicU32, _expected: u32, timeout: Duration) {
    thread::sleep(timeout);
}

#[cfg(target_os = "linux")]
fn futex_wake(epoch: &AtomicU32, wake_count: i32) {
    unsafe {
        let _ = libc::syscall(
            libc::SYS_futex,
            epoch as *const AtomicU32 as *const u32,
            libc::FUTEX_WAKE | libc::FUTEX_PRIVATE_FLAG,
            wake_count,
        );
    }
}

#[cfg(not(target_os = "linux"))]
fn futex_wake(_epoch: &AtomicU32, _wake_count: i32) {}
