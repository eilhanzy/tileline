//! MAS (Multi Audio Scheduler) runtime scaffolding.
//!
//! MAS bridges audio mixing jobs onto MPS workers so the render thread can stay non-blocking.
//! The implementation intentionally starts small:
//! - lock-free ready queue for mixed blocks
//! - topology-aware MPS submission (P/E core preference)
//! - soft-fail accounting for panics in user mix callbacks

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;
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

/// Decoded interleaved PCM clip loaded from a `.wav` asset.
#[derive(Debug, Clone)]
pub struct WavClip {
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub samples: Vec<f32>,
}

impl WavClip {
    /// Decode a `.wav` file into normalized interleaved `f32` PCM samples.
    pub fn load_from_path(path: impl AsRef<Path>) -> Result<Self, MasWavError> {
        let mut reader =
            hound::WavReader::open(path.as_ref()).map_err(|err| MasWavError::Decode {
                reason: err.to_string(),
            })?;
        let spec = reader.spec();
        let channels = spec.channels.max(1);
        let sample_rate_hz = spec.sample_rate.max(8_000);
        let mut samples = Vec::new();

        match spec.sample_format {
            hound::SampleFormat::Float => {
                for sample in reader.samples::<f32>() {
                    let value = sample.map_err(|err| MasWavError::Decode {
                        reason: err.to_string(),
                    })?;
                    samples.push(value.clamp(-1.0, 1.0));
                }
            }
            hound::SampleFormat::Int => {
                let bits = spec.bits_per_sample.clamp(1, 32);
                let scale = ((1_i64 << (bits - 1)) as f32).max(1.0);
                for sample in reader.samples::<i32>() {
                    let value = sample.map_err(|err| MasWavError::Decode {
                        reason: err.to_string(),
                    })?;
                    samples.push((value as f32 / scale).clamp(-1.0, 1.0));
                }
            }
        }

        let channels_usize = channels as usize;
        if channels_usize > 0 {
            let aligned = (samples.len() / channels_usize) * channels_usize;
            samples.truncate(aligned);
        }

        Ok(Self {
            sample_rate_hz,
            channels,
            samples,
        })
    }

    #[inline]
    pub fn frame_count(&self) -> usize {
        let channels = self.channels.max(1) as usize;
        self.samples.len() / channels
    }

    /// Mix this clip into one output block with tempo/pitch controls.
    ///
    /// First-pass behavior uses a coupled playback-rate model:
    /// `effective_rate = pitch_ratio * tempo * src_hz/out_hz`.
    pub fn mix_into_block(
        &self,
        cursor: &mut WavPlaybackCursor,
        params: WavPlaybackParams,
        out: &mut AudioBufferBlock,
    ) {
        let src_frames = self.frame_count();
        let dst_frames = out.frame_count();
        if src_frames == 0 || dst_frames == 0 {
            return;
        }
        let src_hz = self.sample_rate_hz.max(1) as f64;
        let dst_hz = out.sample_rate_hz.max(1) as f64;
        let pitch_ratio = 2.0_f64.powf((params.pitch_semitones as f64) / 12.0);
        let tempo = params.tempo.clamp(0.25, 4.0) as f64;
        let rate = (pitch_ratio * tempo * (src_hz / dst_hz)).max(1e-6);
        let gain = params.gain.clamp(0.0, 4.0);
        let dst_channels = out.channels.max(1) as usize;
        let src_channels = self.channels.max(1) as usize;
        let max_frame = src_frames.saturating_sub(1) as f64;
        let src_frames_f64 = src_frames as f64;

        for frame_index in 0..dst_frames {
            let mut src_frame = cursor.frame_position;
            if params.looped {
                src_frame = wrap_frame(src_frame, src_frames_f64);
            } else if src_frame > max_frame {
                break;
            }

            let src_floor = src_frame.floor();
            let next_floor = if src_floor + 1.0 <= max_frame {
                src_floor + 1.0
            } else if params.looped {
                0.0
            } else {
                max_frame
            };
            let alpha = (src_frame - src_floor) as f32;
            let src0 = src_floor as usize;
            let src1 = next_floor as usize;

            for out_channel in 0..dst_channels {
                let src_channel = if src_channels == 1 {
                    0
                } else {
                    out_channel.min(src_channels - 1)
                };
                let sample0 = self.sample_at(src0, src_channel);
                let sample1 = self.sample_at(src1, src_channel);
                let mixed = sample0 + (sample1 - sample0) * alpha;
                let dst_index = frame_index * dst_channels + out_channel;
                out.samples[dst_index] += mixed * gain;
            }

            cursor.frame_position += rate;
            if params.looped {
                cursor.frame_position = wrap_frame(cursor.frame_position, src_frames_f64);
            }
        }
    }

    #[inline]
    fn sample_at(&self, frame_index: usize, channel: usize) -> f32 {
        let channels = self.channels.max(1) as usize;
        let index = frame_index
            .saturating_mul(channels)
            .saturating_add(channel.min(channels.saturating_sub(1)));
        self.samples.get(index).copied().unwrap_or(0.0)
    }
}

#[inline]
fn wrap_frame(frame: f64, frame_count: f64) -> f64 {
    if frame_count <= 0.0 {
        return 0.0;
    }
    frame.rem_euclid(frame_count)
}

/// Playback cursor for one streamed `.wav` clip.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct WavPlaybackCursor {
    pub frame_position: f64,
}

/// Runtime playback controls for a `.wav` scene track.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WavPlaybackParams {
    pub gain: f32,
    /// Semitone shift relative to source pitch. `+12.0` means one octave up.
    pub pitch_semitones: f32,
    /// Playback tempo scalar (`1.0` default).
    pub tempo: f32,
    pub looped: bool,
}

impl Default for WavPlaybackParams {
    fn default() -> Self {
        Self {
            gain: 1.0,
            pitch_semitones: 0.0,
            tempo: 1.0,
            looped: true,
        }
    }
}

/// `.wav` decode/mix error for MAS-facing audio content APIs.
#[derive(Debug, Clone)]
pub enum MasWavError {
    Decode { reason: String },
}

impl std::fmt::Display for MasWavError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Decode { reason } => write!(f, "wav decode error: {reason}"),
        }
    }
}

impl std::error::Error for MasWavError {}

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
    use std::time::{SystemTime, UNIX_EPOCH};

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

    #[test]
    fn loads_wav_clip_and_mixes_with_pitch_and_tempo() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let path = std::env::temp_dir().join(format!("tileline-mas-test-{nonce}.wav"));
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 48_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&path, spec).expect("create wav");
        for i in 0..512 {
            let t = i as f32 / 512.0;
            let s = (t * std::f32::consts::TAU * 3.0).sin();
            let q = (s * i16::MAX as f32) as i16;
            writer.write_sample(q).expect("write sample");
        }
        writer.finalize().expect("finalize wav");

        let clip = WavClip::load_from_path(&path).expect("decode wav");
        assert_eq!(clip.channels, 1);
        assert!(clip.frame_count() >= 512);

        let mut cursor = WavPlaybackCursor::default();
        let mut block = AudioBufferBlock::silent(1, 48_000, 2, 256);
        clip.mix_into_block(
            &mut cursor,
            WavPlaybackParams {
                gain: 0.5,
                pitch_semitones: 4.0,
                tempo: 1.25,
                looped: true,
            },
            &mut block,
        );
        assert!(cursor.frame_position > 0.0);
        assert_eq!(block.channels, 2);
        assert!(block.samples.iter().any(|s| s.abs() > 1e-4));

        let _ = std::fs::remove_file(path);
    }
}
