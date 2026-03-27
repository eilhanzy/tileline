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
use std::sync::OnceLock;
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
    /// Errno returned by the affinity syscall, if any.
    pub affinity_error: Option<i32>,
    /// Errno returned by the nice syscall, if any.
    pub nice_error: Option<i32>,
    /// Errno returned by the real-time scheduler syscall, if any.
    pub realtime_error: Option<i32>,
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
        if should_report_bootstrap_error(&launch, &bootstrap) {
            eprintln!(
                "[mps worker] '{}' bootstrap affinity={} rt={} nice={} affinity_errno={:?} rt_errno={:?} nice_errno={:?}",
                launch.thread_name,
                bootstrap.affinity_applied,
                bootstrap.realtime_applied,
                bootstrap.nice_applied,
                bootstrap.affinity_error,
                bootstrap.realtime_error,
                bootstrap.nice_error,
            );
        }
        entry(launch, signal);
    })
}

/// Apply core affinity and thread priority for the current worker thread.
pub fn apply_worker_affinity_and_priority(launch: &WorkerLaunchConfig) -> WorkerBootstrapReport {
    let mut report = WorkerBootstrapReport::default();
    match pin_current_thread_to_core(launch.logical_core_id, launch.strict_affinity) {
        Ok(()) => report.affinity_applied = true,
        Err(errno) => {
            report.affinity_error = Some(errno);
            report.last_os_error = Some(errno);
        }
    }
    match apply_linux_realtime_policy(launch) {
        Ok(()) => report.realtime_applied = true,
        Err(errno) => {
            report.realtime_error = Some(errno);
            report.last_os_error = Some(errno);
        }
    }
    match set_current_thread_nice(launch.nice_value) {
        Ok(()) => report.nice_applied = true,
        Err(errno) => {
            report.nice_error = Some(errno);
            report.last_os_error = Some(errno);
        }
    }
    report
}

/// Normalize a worker launch request to what the current Linux host is likely
/// allowed to grant without elevated privileges.
pub fn normalize_worker_launch_for_host(launch: &mut WorkerLaunchConfig) {
    #[cfg(target_os = "linux")]
    {
        if privileged_scheduler_uplift_available() {
            return;
        }

        launch.realtime_priority = None;
        launch.scheduling_policy = WorkerSchedulingPolicy::Other;
        if launch.nice_value < 0 {
            launch.nice_value = 0;
        }
    }

    #[cfg(not(target_os = "linux"))]
    {
        let _ = launch;
    }
}

/// Return whether the current process is likely allowed to request privileged
/// scheduler uplift such as `SCHED_FIFO` or negative nice values.
pub fn privileged_scheduler_uplift_available() -> bool {
    #[cfg(target_os = "linux")]
    {
        static CACHED: OnceLock<bool> = OnceLock::new();
        *CACHED.get_or_init(detect_privileged_scheduler_uplift_available)
    }

    #[cfg(not(target_os = "linux"))]
    {
        false
    }
}

#[cfg(target_os = "linux")]
fn detect_privileged_scheduler_uplift_available() -> bool {
    if let Ok(value) = std::env::var("TILELINE_MPS_PRIVILEGED_SCHED") {
        let normalized = value.trim().to_ascii_lowercase();
        if matches!(normalized.as_str(), "1" | "true" | "yes" | "on") {
            return true;
        }
        if matches!(normalized.as_str(), "0" | "false" | "no" | "off") {
            return false;
        }
    }

    unsafe {
        if libc::geteuid() == 0 {
            return true;
        }

        let mut rt_limit: libc::rlimit = std::mem::zeroed();
        if libc::getrlimit(libc::RLIMIT_RTPRIO, &mut rt_limit) == 0 && rt_limit.rlim_cur > 0 {
            return true;
        }

        let mut nice_limit: libc::rlimit = std::mem::zeroed();
        if libc::getrlimit(libc::RLIMIT_NICE, &mut nice_limit) == 0 && nice_limit.rlim_cur > 0 {
            return true;
        }
    }

    false
}

#[cfg(target_os = "linux")]
fn should_report_bootstrap_error(
    launch: &WorkerLaunchConfig,
    bootstrap: &WorkerBootstrapReport,
) -> bool {
    if bootstrap.affinity_error.is_some() {
        return true;
    }

    if let Some(errno) = bootstrap.realtime_error {
        if !is_expected_permission_denial(errno, launch.realtime_priority.is_some()) {
            return true;
        }
    }

    if let Some(errno) = bootstrap.nice_error {
        if !is_expected_permission_denial(errno, launch.nice_value < 0) {
            return true;
        }
    }

    false
}

#[cfg(target_os = "linux")]
fn is_expected_permission_denial(errno: i32, requested_uplift: bool) -> bool {
    requested_uplift && matches!(errno, libc::EPERM | libc::EACCES)
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
