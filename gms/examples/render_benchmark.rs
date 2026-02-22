use std::error::Error;
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};

use gms::GpuInventory;
use wgpu::{Color, CompositeAlphaMode, PresentMode, SurfaceError, TextureFormat};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

fn main() -> Result<(), Box<dyn Error>> {
    let event_loop = EventLoop::new()?;
    let mut app = RenderBenchmarkApp::default();
    event_loop.run_app(&mut app)?;
    Ok(())
}

#[derive(Default)]
struct RenderBenchmarkApp {
    runtime: Option<BenchmarkRuntime>,
    exit_requested: bool,
    summary_printed: bool,
}

impl ApplicationHandler for RenderBenchmarkApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.runtime.is_some() {
            return;
        }

        match BenchmarkRuntime::new(event_loop) {
            Ok(runtime) => {
                self.runtime = Some(runtime);
                event_loop.set_control_flow(ControlFlow::Poll);
            }
            Err(err) => {
                eprintln!("Failed to start GMS render benchmark: {err}");
                self.exit_requested = true;
                event_loop.exit();
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(runtime) = self.runtime.as_mut() else {
            return;
        };

        if runtime.window.id() != window_id {
            return;
        }

        match event {
            WindowEvent::CloseRequested => {
                self.exit_requested = true;
                event_loop.exit();
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key,
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => match logical_key.as_ref() {
                Key::Named(NamedKey::Escape) => {
                    self.exit_requested = true;
                    event_loop.exit();
                }
                _ => {}
            },
            WindowEvent::Resized(size) => {
                runtime.resize(size);
            }
            WindowEvent::RedrawRequested => {
                if let Err(render_outcome) = runtime.render_frame() {
                    match render_outcome {
                        RenderOutcome::SurfaceLost | RenderOutcome::Outdated => {
                            runtime.reconfigure_surface();
                        }
                        RenderOutcome::OutOfMemory => {
                            eprintln!("wgpu surface out of memory, exiting benchmark");
                            self.exit_requested = true;
                            event_loop.exit();
                        }
                        RenderOutcome::Timeout | RenderOutcome::Other => {
                            // Skip this frame and continue polling.
                        }
                    }
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        event_loop.set_control_flow(ControlFlow::Poll);

        if let Some(runtime) = self.runtime.as_ref() {
            runtime.window.request_redraw();
        }

        if self.exit_requested {
            event_loop.exit();
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        if self.summary_printed {
            return;
        }

        self.summary_printed = true;
        if let Some(runtime) = self.runtime.take() {
            let summary = runtime.finish();
            print_summary(&summary);
        }
    }
}

struct BenchmarkRuntime {
    window: Arc<Window>,
    renderer: Renderer,
    stats: FrameStats,
    adapter_profile: Option<gms::GpuAdapterProfile>,
    gms_inventory: GpuInventory,
}

impl BenchmarkRuntime {
    fn new(event_loop: &ActiveEventLoop) -> Result<Self, Box<dyn Error>> {
        let window = Arc::new(
            event_loop.create_window(
                Window::default_attributes()
                    .with_title("GMS Render Benchmark | Initializing...")
                    .with_inner_size(PhysicalSize::new(1280, 720)),
            )?,
        );

        let renderer = Renderer::new(Arc::clone(&window))?;
        let gms_inventory = GpuInventory::discover();
        let adapter_profile = match_inventory_profile(&gms_inventory, &renderer.adapter_info);

        let mut stats = FrameStats::new(renderer.size, renderer.present_mode);
        stats.last_title_update = Instant::now();

        let mut runtime = Self {
            window,
            renderer,
            stats,
            adapter_profile,
            gms_inventory,
        };

        runtime.update_title(0.0, 0.0);
        Ok(runtime)
    }

    fn resize(&mut self, size: PhysicalSize<u32>) {
        self.renderer.resize(size);
        self.stats.set_resolution(self.renderer.size);
    }

    fn reconfigure_surface(&mut self) {
        self.renderer.reconfigure();
    }

    fn render_frame(&mut self) -> Result<(), RenderOutcome> {
        let frame_start = Instant::now();
        self.renderer.render()?;

        let timing = self.stats.record_frame(frame_start, Instant::now());
        if timing.should_update_title {
            self.update_title(timing.instant_fps, timing.avg_fps);
        }

        Ok(())
    }

    fn update_title(&mut self, instant_fps: f64, avg_fps: f64) {
        let gms_score = self.adapter_profile.as_ref().map(|p| p.score).unwrap_or(0);
        let title = format!(
            "GMS Render Benchmark | FPS {:.1} | Avg {:.1} | Frames {} | GMS {} | Close/Esc to finish",
            instant_fps,
            avg_fps,
            self.stats.frames,
            gms_score
        );
        self.window.set_title(&title);
    }

    fn finish(self) -> BenchmarkSummary {
        let computed = self.stats.compute_summary();
        let gms_score = self.adapter_profile.as_ref().map(|p| p.score).unwrap_or(0);
        let adapter_name = self
            .adapter_profile
            .as_ref()
            .map(|p| p.name.clone())
            .unwrap_or_else(|| self.renderer.adapter_info.name.clone());

        let score = compute_render_benchmark_score(&computed, gms_score);

        BenchmarkSummary {
            adapter_name,
            backend: format!("{:?}", self.renderer.adapter_info.backend),
            device_type: format!("{:?}", self.renderer.adapter_info.device_type),
            resolution: self.renderer.size,
            present_mode: self.renderer.present_mode,
            total_frames: computed.total_frames,
            elapsed: computed.elapsed,
            avg_fps: computed.avg_fps,
            peak_fps: computed.peak_fps,
            avg_frame_ms: computed.avg_frame_ms,
            p95_frame_ms: computed.p95_frame_ms,
            p99_frame_ms: computed.p99_frame_ms,
            low_1_percent_fps: computed.low_1_percent_fps,
            frame_time_stddev_ms: computed.frame_time_stddev_ms,
            gms_hardware_score: gms_score,
            total_discovered_gpus: self.gms_inventory.adapters.len(),
            render_score: score.score,
            render_tier: score.tier,
            score_breakdown: score,
        }
    }
}

struct Renderer {
    _instance: wgpu::Instance,
    surface: wgpu::Surface<'static>,
    adapter_info: wgpu::AdapterInfo,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: PhysicalSize<u32>,
    present_mode: PresentMode,
    clear_phase: f64,
}

impl Renderer {
    fn new(window: Arc<Window>) -> Result<Self, Box<dyn Error>> {
        let size = window.inner_size();
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(Arc::clone(&window))?;

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))?;

        let adapter_info = adapter.get_info();
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("gms-render-benchmark-device"),
                ..Default::default()
            }))?;

        let capabilities = surface.get_capabilities(&adapter);
        let mut config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .ok_or_else(|| SimpleError("surface is not compatible with adapter".into()))?;

        if let Some(srgb) = capabilities
            .formats
            .iter()
            .copied()
            .find(TextureFormat::is_srgb)
        {
            config.format = srgb;
        }
        config.present_mode = select_present_mode(&capabilities.present_modes);
        config.alpha_mode = capabilities
            .alpha_modes
            .iter()
            .copied()
            .find(|mode| *mode == CompositeAlphaMode::Opaque)
            .unwrap_or(config.alpha_mode);
        config.desired_maximum_frame_latency = 2;
        surface.configure(&device, &config);

        Ok(Self {
            _instance: instance,
            surface,
            adapter_info,
            device,
            queue,
            config: config.clone(),
            size,
            present_mode: config.present_mode,
            clear_phase: 0.0,
        })
    }

    fn resize(&mut self, size: PhysicalSize<u32>) {
        if size.width == 0 || size.height == 0 {
            self.size = size;
            return;
        }

        self.size = size;
        self.config.width = size.width;
        self.config.height = size.height;
        self.surface.configure(&self.device, &self.config);
    }

    fn reconfigure(&mut self) {
        if self.size.width == 0 || self.size.height == 0 {
            return;
        }
        self.surface.configure(&self.device, &self.config);
    }

    fn render(&mut self) -> Result<(), RenderOutcome> {
        if self.size.width == 0 || self.size.height == 0 {
            return Ok(());
        }

        let frame = self
            .surface
            .get_current_texture()
            .map_err(RenderOutcome::from)?;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        self.clear_phase = (self.clear_phase + 0.0125) % std::f64::consts::TAU;
        let clear_color = animated_clear_color(self.clear_phase);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gms-render-benchmark-encoder"),
            });

        {
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("gms-render-benchmark-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear_color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        }

        self.queue.submit(Some(encoder.finish()));
        frame.present();
        Ok(())
    }
}

fn animated_clear_color(phase: f64) -> Color {
    let r = 0.15 + 0.25 * (phase).sin().abs();
    let g = 0.12 + 0.28 * (phase * 1.37).sin().abs();
    let b = 0.18 + 0.30 * (phase * 0.73).cos().abs();
    Color { r, g, b, a: 1.0 }
}

fn select_present_mode(modes: &[PresentMode]) -> PresentMode {
    for preferred in [
        PresentMode::AutoNoVsync,
        PresentMode::Immediate,
        PresentMode::Mailbox,
        PresentMode::AutoVsync,
        PresentMode::Fifo,
    ] {
        if modes.contains(&preferred) {
            return preferred;
        }
    }
    PresentMode::Fifo
}

fn match_inventory_profile(
    inventory: &GpuInventory,
    adapter_info: &wgpu::AdapterInfo,
) -> Option<gms::GpuAdapterProfile> {
    inventory
        .adapters
        .iter()
        .find(|profile| {
            profile.vendor_id == adapter_info.vendor
                && profile.device_id == adapter_info.device
                && profile.backend == adapter_info.backend
                && profile.name == adapter_info.name
        })
        .cloned()
        .or_else(|| {
            inventory
                .adapters
                .iter()
                .find(|profile| {
                    profile.backend == adapter_info.backend
                        && profile.vendor_id == adapter_info.vendor
                        && profile.name == adapter_info.name
                })
                .cloned()
        })
}

#[derive(Debug, Clone, Copy)]
enum RenderOutcome {
    Timeout,
    Outdated,
    SurfaceLost,
    OutOfMemory,
    Other,
}

impl From<SurfaceError> for RenderOutcome {
    fn from(value: SurfaceError) -> Self {
        match value {
            SurfaceError::Timeout => Self::Timeout,
            SurfaceError::Outdated => Self::Outdated,
            SurfaceError::Lost => Self::SurfaceLost,
            SurfaceError::OutOfMemory => Self::OutOfMemory,
            SurfaceError::Other => Self::Other,
        }
    }
}

struct FrameStats {
    benchmark_start: Instant,
    last_frame_presented_at: Option<Instant>,
    last_title_update: Instant,
    frames: u64,
    render_durations_ms: Vec<f64>,
    frame_intervals_ms: Vec<f64>,
    peak_fps: f64,
    resolution: PhysicalSize<u32>,
}

struct FrameTiming {
    instant_fps: f64,
    avg_fps: f64,
    should_update_title: bool,
}

impl FrameStats {
    fn new(resolution: PhysicalSize<u32>, _present_mode: PresentMode) -> Self {
        let now = Instant::now();
        Self {
            benchmark_start: now,
            last_frame_presented_at: None,
            last_title_update: now,
            frames: 0,
            render_durations_ms: Vec::with_capacity(16_384),
            frame_intervals_ms: Vec::with_capacity(16_384),
            peak_fps: 0.0,
            resolution,
        }
    }

    fn set_resolution(&mut self, resolution: PhysicalSize<u32>) {
        self.resolution = resolution;
    }

    fn record_frame(&mut self, frame_start: Instant, frame_end: Instant) -> FrameTiming {
        self.frames = self.frames.saturating_add(1);
        let render_ms = (frame_end - frame_start).as_secs_f64() * 1000.0;
        self.render_durations_ms.push(render_ms);

        let mut instant_fps = 0.0;
        if let Some(last) = self.last_frame_presented_at {
            let interval_ms = (frame_end - last).as_secs_f64() * 1000.0;
            if interval_ms.is_finite() && interval_ms > 0.0 {
                self.frame_intervals_ms.push(interval_ms);
                instant_fps = 1000.0 / interval_ms;
                self.peak_fps = self.peak_fps.max(instant_fps);
            }
        }
        self.last_frame_presented_at = Some(frame_end);

        let elapsed = (frame_end - self.benchmark_start).as_secs_f64().max(1e-9);
        let avg_fps = self.frames as f64 / elapsed;

        let should_update_title =
            frame_end.duration_since(self.last_title_update) >= Duration::from_millis(250);
        if should_update_title {
            self.last_title_update = frame_end;
        }

        FrameTiming {
            instant_fps,
            avg_fps,
            should_update_title,
        }
    }

    fn compute_summary(self) -> ComputedFrameStats {
        let elapsed = self.benchmark_start.elapsed();
        let elapsed_secs = elapsed.as_secs_f64().max(1e-9);
        let avg_fps = self.frames as f64 / elapsed_secs;
        let avg_frame_ms = average(&self.frame_intervals_ms).unwrap_or(0.0);
        let p95_frame_ms = percentile_ms(&self.frame_intervals_ms, 95.0).unwrap_or(avg_frame_ms);
        let p99_frame_ms = percentile_ms(&self.frame_intervals_ms, 99.0).unwrap_or(p95_frame_ms);
        let low_1_percent_fps = if p99_frame_ms > 0.0 {
            1000.0 / p99_frame_ms
        } else {
            0.0
        };
        let frame_time_stddev_ms = stddev(&self.frame_intervals_ms).unwrap_or(0.0);

        ComputedFrameStats {
            total_frames: self.frames,
            elapsed,
            avg_fps,
            peak_fps: self.peak_fps,
            avg_frame_ms,
            p95_frame_ms,
            p99_frame_ms,
            low_1_percent_fps,
            frame_time_stddev_ms,
            resolution: self.resolution,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ComputedFrameStats {
    total_frames: u64,
    elapsed: Duration,
    avg_fps: f64,
    peak_fps: f64,
    avg_frame_ms: f64,
    p95_frame_ms: f64,
    p99_frame_ms: f64,
    low_1_percent_fps: f64,
    frame_time_stddev_ms: f64,
    resolution: PhysicalSize<u32>,
}

#[derive(Debug, Clone, Copy)]
struct RenderScore {
    score: u64,
    tier: &'static str,
    fps_term: f64,
    stability_factor: f64,
    tail_factor: f64,
    resolution_factor: f64,
    gms_factor: f64,
}

fn compute_render_benchmark_score(
    stats: &ComputedFrameStats,
    gms_hardware_score: u64,
) -> RenderScore {
    let pixels = (stats.resolution.width.max(1) as f64) * (stats.resolution.height.max(1) as f64);
    let resolution_factor = (pixels / (1280.0 * 720.0)).sqrt().clamp(0.75, 3.0);

    let avg_ms = stats.avg_frame_ms.max(0.001);
    let p95_ms = stats.p95_frame_ms.max(avg_ms);
    let p99_ms = stats.p99_frame_ms.max(p95_ms);

    let stability_factor = (avg_ms / p95_ms).clamp(0.35, 1.0);
    let tail_factor = (avg_ms / p99_ms).clamp(0.25, 1.0);

    // Small hardware-score boost so a stronger adapter helps, but FPS still dominates.
    let gms_factor = (1.0 + (gms_hardware_score as f64).ln_1p() / 12.0).clamp(1.0, 2.0);

    let fps_term = stats.avg_fps.max(0.0) * 100.0;
    let score = (fps_term * resolution_factor * stability_factor * tail_factor * gms_factor)
        .round()
        .max(0.0) as u64;

    RenderScore {
        score,
        tier: render_score_tier(score),
        fps_term,
        stability_factor,
        tail_factor,
        resolution_factor,
        gms_factor,
    }
}

fn render_score_tier(score: u64) -> &'static str {
    match score {
        60_000.. => "S",
        35_000.. => "A",
        20_000.. => "B",
        10_000.. => "C",
        _ => "D",
    }
}

struct BenchmarkSummary {
    adapter_name: String,
    backend: String,
    device_type: String,
    resolution: PhysicalSize<u32>,
    present_mode: PresentMode,
    total_frames: u64,
    elapsed: Duration,
    avg_fps: f64,
    peak_fps: f64,
    avg_frame_ms: f64,
    p95_frame_ms: f64,
    p99_frame_ms: f64,
    low_1_percent_fps: f64,
    frame_time_stddev_ms: f64,
    gms_hardware_score: u64,
    total_discovered_gpus: usize,
    render_score: u64,
    render_tier: &'static str,
    score_breakdown: RenderScore,
}

fn print_summary(summary: &BenchmarkSummary) {
    println!();
    println!("=== GMS Render Benchmark Summary ===");
    println!(
        "Adapter: {} | Backend: {} | Type: {}",
        summary.adapter_name, summary.backend, summary.device_type
    );
    println!(
        "Resolution: {}x{} | Present mode: {:?}",
        summary.resolution.width, summary.resolution.height, summary.present_mode
    );
    println!(
        "Frames: {} | Elapsed: {:.3}s | Avg FPS: {:.2} | Peak FPS: {:.2}",
        summary.total_frames,
        summary.elapsed.as_secs_f64(),
        summary.avg_fps,
        summary.peak_fps
    );
    println!(
        "Frame time(ms): avg {:.3} | p95 {:.3} | p99 {:.3} | 1% low FPS {:.2} | stddev {:.3}",
        summary.avg_frame_ms,
        summary.p95_frame_ms,
        summary.p99_frame_ms,
        summary.low_1_percent_fps,
        summary.frame_time_stddev_ms
    );
    println!(
        "GMS hardware score: {} | discovered adapters: {}",
        summary.gms_hardware_score, summary.total_discovered_gpus
    );
    println!(
        "Render benchmark score: {} [{}]",
        summary.render_score, summary.render_tier
    );
    println!(
        "Score factors => fps_term {:.1}, resolution {:.3}, stability {:.3}, tail {:.3}, gms {:.3}",
        summary.score_breakdown.fps_term,
        summary.score_breakdown.resolution_factor,
        summary.score_breakdown.stability_factor,
        summary.score_breakdown.tail_factor,
        summary.score_breakdown.gms_factor
    );
}

fn average(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    Some(values.iter().sum::<f64>() / values.len() as f64)
}

fn stddev(values: &[f64]) -> Option<f64> {
    if values.len() < 2 {
        return None;
    }
    let mean = average(values)?;
    let variance = values
        .iter()
        .map(|v| {
            let d = *v - mean;
            d * d
        })
        .sum::<f64>()
        / values.len() as f64;
    Some(variance.sqrt())
}

fn percentile_ms(values: &[f64], percentile: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }

    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let p = percentile.clamp(0.0, 100.0) / 100.0;
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted.get(idx).copied()
}

#[derive(Debug, Clone)]
struct SimpleError(String);

impl fmt::Display for SimpleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for SimpleError {}
