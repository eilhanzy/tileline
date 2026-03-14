//! MAS (Multi Audio Scheduler) runtime scaffolding.
//!
//! MAS bridges audio mixing jobs onto MPS workers so the render thread can stay non-blocking.
//! The implementation intentionally starts small:
//! - lock-free ready queue for mixed blocks
//! - topology-aware MPS submission (P/E core preference)
//! - soft-fail accounting for panics in user mix callbacks

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crossbeam::queue::SegQueue;
use mps::{CorePreference, MpsScheduler, SchedulerMetrics, TaskPriority};

/// Audio job priority class mapped onto MPS scheduling priority.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MasPriority {
    Critical,
    Realtime,
    Normal,
    Background,
}

impl Default for MasPriority {
    fn default() -> Self {
        Self::Realtime
    }
}

/// Preferred CPU core class for an audio task.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MasCoreAffinity {
    Auto,
    Performance,
    Efficient,
}

impl Default for MasCoreAffinity {
    fn default() -> Self {
        Self::Auto
    }
}

/// One mixed audio buffer block produced asynchronously by MAS jobs.
#[derive(Debug, Clone)]
pub struct AudioBufferBlock {
    pub sequence: u64,
    pub sample_rate_hz: u32,
    pub channels: u16,
    /// Interleaved PCM f32 samples.
    pub samples: Vec<f32>,
}

impl AudioBufferBlock {
    /// Build a silent interleaved block.
    pub fn silent(sequence: u64, sample_rate_hz: u32, channels: u16, frames: usize) -> Self {
        let channels = channels.max(1);
        let samples = vec![0.0; frames.saturating_mul(channels as usize)];
        Self {
            sequence,
            sample_rate_hz: sample_rate_hz.max(8_000),
            channels,
            samples,
        }
    }

    /// Number of frames in this block.
    pub fn frame_count(&self) -> usize {
        let channels = self.channels.max(1) as usize;
        self.samples.len() / channels
    }
}

/// MAS configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MasConfig {
    pub sample_rate_hz: u32,
    pub channels: u16,
    /// Soft capacity for completed ready blocks.
    pub ready_queue_soft_capacity: usize,
    pub max_ready_drain_per_tick: usize,
}

impl Default for MasConfig {
    fn default() -> Self {
        Self {
            sample_rate_hz: 48_000,
            channels: 2,
            ready_queue_soft_capacity: 16,
            max_ready_drain_per_tick: 8,
        }
    }
}

/// Submission token for tracking a job.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MasSubmission {
    pub mix_sequence: u64,
    pub mps_task_id: u64,
}

/// Snapshot of MAS counters.
#[derive(Debug, Clone, Copy, Default)]
pub struct MasMetrics {
    pub submitted_jobs: u64,
    pub completed_jobs: u64,
    pub failed_jobs: u64,
    pub dropped_ready_blocks: u64,
    pub ready_blocks: usize,
    pub mps: SchedulerMetrics,
}

/// Lock-free MAS coordinator backed by MPS worker lanes.
pub struct MultiAudioScheduler {
    mps: Arc<MpsScheduler>,
    config: MasConfig,
    next_sequence: AtomicU64,
    submitted_jobs: Arc<AtomicU64>,
    completed_jobs: Arc<AtomicU64>,
    failed_jobs: Arc<AtomicU64>,
    dropped_ready_blocks: Arc<AtomicU64>,
    ready_depth: Arc<AtomicUsize>,
    ready_blocks: Arc<SegQueue<AudioBufferBlock>>,
}

impl MultiAudioScheduler {
    /// Build MAS with a private MPS scheduler instance.
    pub fn new(config: MasConfig) -> Self {
        Self::with_mps(Arc::new(MpsScheduler::new()), config)
    }

    /// Build MAS with a caller-provided shared MPS scheduler.
    pub fn with_mps(mps: Arc<MpsScheduler>, config: MasConfig) -> Self {
        Self {
            mps,
            config,
            next_sequence: AtomicU64::new(1),
            submitted_jobs: Arc::new(AtomicU64::new(0)),
            completed_jobs: Arc::new(AtomicU64::new(0)),
            failed_jobs: Arc::new(AtomicU64::new(0)),
            dropped_ready_blocks: Arc::new(AtomicU64::new(0)),
            ready_depth: Arc::new(AtomicUsize::new(0)),
            ready_blocks: Arc::new(SegQueue::new()),
        }
    }

    /// Returns a shared handle to underlying MPS scheduler.
    pub fn mps(&self) -> &Arc<MpsScheduler> {
        &self.mps
    }

    /// Submit one asynchronous mix job.
    ///
    /// The closure receives a mutable buffer block to fill/transform.
    pub fn submit_mix_block<F>(
        &self,
        priority: MasPriority,
        affinity: MasCoreAffinity,
        frames: usize,
        mix_fn: F,
    ) -> MasSubmission
    where
        F: FnOnce(&mut AudioBufferBlock) + Send + 'static,
    {
        let sequence = self.next_sequence.fetch_add(1, Ordering::Relaxed);
        let mut block = AudioBufferBlock::silent(
            sequence,
            self.config.sample_rate_hz,
            self.config.channels,
            frames,
        );

        let ready_blocks = Arc::clone(&self.ready_blocks);
        let ready_depth = Arc::clone(&self.ready_depth);
        let completed_jobs = Arc::clone(&self.completed_jobs);
        let failed_jobs = Arc::clone(&self.failed_jobs);
        let dropped_ready_blocks = Arc::clone(&self.dropped_ready_blocks);
        let ready_capacity = self.config.ready_queue_soft_capacity.max(1);

        self.submitted_jobs.fetch_add(1, Ordering::Relaxed);
        let mps_task_id =
            self.mps
                .submit_native(map_priority(priority), map_affinity(affinity), move || {
                    let result = catch_unwind(AssertUnwindSafe(|| {
                        mix_fn(&mut block);
                    }));
                    match result {
                        Ok(()) => {
                            let current_depth = ready_depth.load(Ordering::Acquire);
                            if current_depth >= ready_capacity {
                                dropped_ready_blocks.fetch_add(1, Ordering::Relaxed);
                                return;
                            }
                            ready_blocks.push(block);
                            ready_depth.fetch_add(1, Ordering::Release);
                            completed_jobs.fetch_add(1, Ordering::Relaxed);
                        }
                        Err(_) => {
                            failed_jobs.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                });

        MasSubmission {
            mix_sequence: sequence,
            mps_task_id,
        }
    }

    /// Drain completed blocks into `out` with a bounded work budget.
    pub fn drain_ready_blocks(&self, out: &mut Vec<AudioBufferBlock>) -> usize {
        let mut drained = 0usize;
        let budget = self.config.max_ready_drain_per_tick.max(1);
        while drained < budget {
            let Some(block) = self.ready_blocks.pop() else {
                break;
            };
            out.push(block);
            self.ready_depth.fetch_sub(1, Ordering::AcqRel);
            drained += 1;
        }
        drained
    }

    /// Wait until MPS queue is idle.
    pub fn wait_for_idle(&self, timeout: Duration) -> bool {
        self.mps.wait_for_idle(timeout)
    }

    /// Snapshot metrics for telemetry and editor overlays.
    pub fn metrics(&self) -> MasMetrics {
        MasMetrics {
            submitted_jobs: self.submitted_jobs.load(Ordering::Acquire),
            completed_jobs: self.completed_jobs.load(Ordering::Acquire),
            failed_jobs: self.failed_jobs.load(Ordering::Acquire),
            dropped_ready_blocks: self.dropped_ready_blocks.load(Ordering::Acquire),
            ready_blocks: self.ready_depth.load(Ordering::Acquire),
            mps: self.mps.metrics(),
        }
    }
}

#[inline]
fn map_priority(priority: MasPriority) -> TaskPriority {
    match priority {
        MasPriority::Critical => TaskPriority::Critical,
        MasPriority::Realtime => TaskPriority::High,
        MasPriority::Normal => TaskPriority::Normal,
        MasPriority::Background => TaskPriority::Background,
    }
}

#[inline]
fn map_affinity(affinity: MasCoreAffinity) -> CorePreference {
    match affinity {
        MasCoreAffinity::Auto => CorePreference::Auto,
        MasCoreAffinity::Performance => CorePreference::Performance,
        MasCoreAffinity::Efficient => CorePreference::Efficient,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn submits_and_drains_audio_block() {
        let mas = MultiAudioScheduler::new(MasConfig {
            max_ready_drain_per_tick: 4,
            ..MasConfig::default()
        });
        let _submission = mas.submit_mix_block(
            MasPriority::Realtime,
            MasCoreAffinity::Performance,
            128,
            |block| {
                for sample in &mut block.samples {
                    *sample = 0.25;
                }
            },
        );

        assert!(mas.wait_for_idle(Duration::from_millis(250)));

        let mut drained = Vec::new();
        let count = mas.drain_ready_blocks(&mut drained);
        assert_eq!(count, 1);
        assert_eq!(drained[0].frame_count(), 128);
        assert!(drained[0]
            .samples
            .iter()
            .all(|sample| (*sample - 0.25).abs() <= f32::EPSILON));
    }
}
