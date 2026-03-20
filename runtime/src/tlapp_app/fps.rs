use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy)]
pub struct FpsReport {
    pub instant_fps: f32,
    pub ema_fps: f32,
    pub avg_fps: f32,
    pub frame_time_stddev_ms: f32,
}

#[derive(Debug, Clone)]
pub struct FpsTracker {
    pub frame_times: [f32; 120],
    pub cursor: usize,
    pub count: usize,
    pub ema_fps: f32,
    pub last_report_at: Instant,
    pub report_interval: Duration,
}

impl FpsTracker {
    pub fn new(report_interval: Duration) -> Self {
        let now = Instant::now();
        Self {
            frame_times: [1.0 / 60.0; 120],
            cursor: 0,
            count: 0,
            ema_fps: 60.0,
            last_report_at: now,
            report_interval,
        }
    }

    pub fn ema_fps(&self) -> f32 {
        self.ema_fps
    }

    pub fn dynamic_uncapped_fps_hint(&self) -> f32 {
        if self.count == 0 {
            return self.ema_fps.max(90.0);
        }
        let fast_lane_fps = self.fast_percentile_fps(0.20);
        let blended = fast_lane_fps * 0.78 + self.ema_fps * 0.22;
        blended.clamp(45.0, 1_200.0)
    }

    pub fn snapshot(&self) -> FpsReport {
        let count = self.count.max(1).min(self.frame_times.len());
        let latest_index = if self.cursor == 0 {
            count.saturating_sub(1)
        } else {
            self.cursor.saturating_sub(1).min(count.saturating_sub(1))
        };
        let instant_frame = self.frame_times[latest_index].max(1e-6);
        let instant_fps = 1.0 / instant_frame;
        let n = count as f32;
        let avg_frame = self.frame_times.iter().take(count).copied().sum::<f32>() / n;
        let variance = self
            .frame_times
            .iter()
            .take(count)
            .map(|t| {
                let d = *t - avg_frame;
                d * d
            })
            .sum::<f32>()
            / n;
        FpsReport {
            instant_fps,
            ema_fps: self.ema_fps,
            avg_fps: 1.0 / avg_frame.max(1e-6),
            frame_time_stddev_ms: variance.sqrt() * 1_000.0,
        }
    }

    pub fn fast_percentile_fps(&self, quantile: f32) -> f32 {
        let count = self.count.max(1).min(self.frame_times.len());
        let mut scratch = [0.0_f32; 120];
        scratch[..count].copy_from_slice(&self.frame_times[..count]);
        scratch[..count].sort_by(|a, b| a.total_cmp(b));
        let idx = ((count as f32 - 1.0) * quantile.clamp(0.0, 1.0)).round() as usize;
        let frame_time = scratch[idx.min(count.saturating_sub(1))].max(1e-6);
        1.0 / frame_time
    }

    pub fn record(&mut self, now: Instant, frame_time: f32) -> Option<FpsReport> {
        let frame_time = frame_time.clamp(1.0 / 500.0, 0.25);
        self.frame_times[self.cursor] = frame_time;
        self.cursor = (self.cursor + 1) % self.frame_times.len();
        self.count = self.count.saturating_add(1).min(self.frame_times.len());

        let instant_fps = 1.0 / frame_time.max(1e-6);
        // Guard EMA against startup/outlier spikes so scheduler tuning follows real throughput.
        let ema_sample_fps = instant_fps.clamp(1.0, 180.0);
        let alpha = 0.12;
        self.ema_fps += (ema_sample_fps - self.ema_fps) * alpha;

        if now.duration_since(self.last_report_at) < self.report_interval {
            return None;
        }
        self.last_report_at = now;

        let n = self.count.max(1) as f32;
        let avg_frame = self
            .frame_times
            .iter()
            .take(self.count)
            .copied()
            .sum::<f32>()
            / n;
        let variance = self
            .frame_times
            .iter()
            .take(self.count)
            .map(|t| {
                let d = *t - avg_frame;
                d * d
            })
            .sum::<f32>()
            / n;
        let stddev = variance.sqrt();

        Some(FpsReport {
            instant_fps,
            ema_fps: self.ema_fps,
            avg_fps: 1.0 / avg_frame.max(1e-6),
            frame_time_stddev_ms: stddev * 1_000.0,
        })
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct RenderDistanceStats {
    pub culled: usize,
    pub blurred: usize,
}
