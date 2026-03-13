use std::env;
use std::error::Error;
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};

use mgs::bridge::MpsWorkloadHint;
use mgs::fallback::FallbackLevel;
use mgs::{MgsBridge, MobileGpuProfile, TbdrArchitecture};
use wgpu::{Color, CompositeAlphaMode, PresentMode, SurfaceError, TextureFormat};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

fn main() -> Result<(), Box<dyn Error>> {
    let options = CliOptions::parse_from_env()?;
    let event_loop = EventLoop::new()?;
    let mut app = RenderBenchmarkApp::new(options);
    event_loop.run_app(&mut app)?;
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct CliOptions {
    mode_override: ModeOverride,
    vsync_override: VsyncOverride,
    multi_gpu_override: MultiGpuOverride,
    warmup_duration: Duration,
    sample_duration: Duration,
    resolution: PhysicalSize<u32>,
}

impl Default for CliOptions {
    fn default() -> Self {
        Self {
            mode_override: ModeOverride::Auto,
            vsync_override: VsyncOverride::Auto,
            multi_gpu_override: MultiGpuOverride::Off,
            warmup_duration: Duration::from_secs(2),
            sample_duration: Duration::from_secs(10),
            resolution: PhysicalSize::new(1280, 720),
        }
    }
}

impl CliOptions {
    fn parse_from_env() -> Result<Self, Box<dyn Error>> {
        let mut options = Self::default();
        let mut args = env::args().skip(1);

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--help" | "-h" => {
                    print_usage();
                    std::process::exit(0);
                }
                "--mode" => {
                    let value = next_arg(&mut args, "--mode")?;
                    options.mode_override = ModeOverride::parse(&value)?;
                }
                "--vsync" => {
                    let value = next_arg(&mut args, "--vsync")?;
                    options.vsync_override = VsyncOverride::parse(&value)?;
                }
                "--multi-gpu" => {
                    let value = next_arg(&mut args, "--multi-gpu")?;
                    options.multi_gpu_override = MultiGpuOverride::parse(&value)?;
                }
                "--duration" => {
                    let value = next_arg(&mut args, "--duration")?;
                    options.sample_duration = parse_seconds_arg(&value, "--duration")?;
                }
                "--warmup" => {
                    let value = next_arg(&mut args, "--warmup")?;
                    options.warmup_duration = parse_seconds_arg(&value, "--warmup")?;
                }
                "--resolution" => {
                    let value = next_arg(&mut args, "--resolution")?;
                    options.resolution = parse_resolution_arg(&value)?;
                }
                other => {
                    return Err(Box::new(SimpleError(format!(
                        "unknown argument: {other} (use --help)"
                    ))));
                }
            }
        }

        if options.sample_duration.is_zero() {
            return Err(Box::new(SimpleError(
                "--duration must be greater than 0".to_owned(),
            )));
        }

        Ok(options)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModeOverride {
    Auto,
    Stable,
    Max,
}

impl ModeOverride {
    fn parse(value: &str) -> Result<Self, Box<dyn Error>> {
        match value.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "stable" => Ok(Self::Stable),
            "max" | "throughput" => Ok(Self::Max),
            _ => Err(Box::new(SimpleError(format!(
                "invalid --mode value: {value} (expected auto|stable|max)"
            )))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VsyncOverride {
    Auto,
    On,
    Off,
}

impl VsyncOverride {
    fn parse(value: &str) -> Result<Self, Box<dyn Error>> {
        match value.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "on" | "true" | "1" => Ok(Self::On),
            "off" | "false" | "0" => Ok(Self::Off),
            _ => Err(Box::new(SimpleError(format!(
                "invalid --vsync value: {value} (expected auto|on|off)"
            )))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MultiGpuOverride {
    Auto,
    On,
    Off,
}

impl MultiGpuOverride {
    fn parse(value: &str) -> Result<Self, Box<dyn Error>> {
        match value.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "on" | "true" | "1" => Ok(Self::On),
            "off" | "false" | "0" => Ok(Self::Off),
            _ => Err(Box::new(SimpleError(format!(
                "invalid --multi-gpu value: {value} (expected auto|on|off)"
            )))),
        }
    }
}

fn next_arg(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, Box<dyn Error>> {
    args.next()
        .ok_or_else(|| Box::new(SimpleError(format!("missing value for {flag}"))) as Box<dyn Error>)
}

fn parse_seconds_arg(value: &str, flag: &str) -> Result<Duration, Box<dyn Error>> {
    let seconds = value.parse::<f64>().map_err(|_| {
        Box::new(SimpleError(format!(
            "invalid {flag} value: {value} (expected seconds, e.g. 10 or 2.5)"
        ))) as Box<dyn Error>
    })?;

    if !seconds.is_finite() || seconds < 0.0 {
        return Err(Box::new(SimpleError(format!(
            "invalid {flag} value: {value} (must be >= 0)"
        ))));
    }

    Ok(Duration::from_secs_f64(seconds))
}

fn parse_resolution_arg(value: &str) -> Result<PhysicalSize<u32>, Box<dyn Error>> {
    let normalized = value.to_ascii_lowercase();
    let (w, h) = normalized.split_once('x').ok_or_else(|| {
        Box::new(SimpleError(format!(
            "invalid --resolution value: {value} (expected WxH, e.g. 1280x720)"
        ))) as Box<dyn Error>
    })?;

    let width = w.parse::<u32>().map_err(|_| {
        Box::new(SimpleError(format!(
            "invalid --resolution width: {w} (from {value})"
        ))) as Box<dyn Error>
    })?;
    let height = h.parse::<u32>().map_err(|_| {
        Box::new(SimpleError(format!(
            "invalid --resolution height: {h} (from {value})"
        ))) as Box<dyn Error>
    })?;

    if width == 0 || height == 0 {
        return Err(Box::new(SimpleError(
            "--resolution dimensions must be non-zero".to_owned(),
        )));
    }

    Ok(PhysicalSize::new(width, height))
}

fn print_usage() {
    println!("MGS Render Benchmark");
    println!("Usage: cargo run -p mgs --example render_benchmark -- [options]");
    println!("Options:");
    println!("  --mode auto|stable|max      Frame pacing mode (default: auto)");
    println!("  --vsync auto|on|off         Present mode preference (default: auto)");
    println!("  --multi-gpu auto|on|off     Accepted for CLI parity; ignored by MGS");
    println!("  --warmup <sec>              Warmup duration before sampling (default: 2)");
    println!("  --duration <sec>            Sampling duration before auto-exit (default: 10)");
    println!("  --resolution <WxH>          Window resolution (default: 1280x720)");
    println!("  -h, --help                  Show this help");
}

struct RenderBenchmarkApp {
    options: CliOptions,
    runtime: Option<BenchmarkRuntime>,
    exit_requested: bool,
    summary_printed: bool,
}

impl RenderBenchmarkApp {
    fn new(options: CliOptions) -> Self {
        Self {
            options,
            runtime: None,
            exit_requested: false,
            summary_printed: false,
        }
    }
}

impl ApplicationHandler for RenderBenchmarkApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.runtime.is_some() {
            return;
        }

        match BenchmarkRuntime::new(event_loop, self.options) {
            Ok(runtime) => {
                runtime.window.request_redraw();
                event_loop.set_control_flow(runtime.preferred_control_flow());
                self.runtime = Some(runtime);
            }
            Err(err) => {
                eprintln!("Failed to start MGS render benchmark: {err}");
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
            } => {
                if let Key::Named(NamedKey::Escape) = logical_key.as_ref() {
                    self.exit_requested = true;
                    event_loop.exit();
                }
            }
            WindowEvent::Resized(size) => runtime.resize(size),
            WindowEvent::RedrawRequested => {
                if runtime.use_redraw_chaining() {
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
                            RenderOutcome::Timeout | RenderOutcome::Other => {}
                        }
                    }
                    runtime.window.request_redraw();
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(runtime) = self.runtime.as_mut() {
            event_loop.set_control_flow(runtime.preferred_control_flow());
            if runtime.use_redraw_chaining() {
                if runtime.stats.total_frames == 0 {
                    runtime.window.request_redraw();
                }
            } else {
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
                        RenderOutcome::Timeout | RenderOutcome::Other => {}
                    }
                }
            }

            if runtime.should_auto_exit() {
                self.exit_requested = true;
                event_loop.exit();
            }
        } else {
            event_loop.set_control_flow(ControlFlow::Wait);
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BenchmarkPacingMode {
    Stable,
    MaxThroughput,
}

struct BenchmarkRuntime {
    window: Arc<Window>,
    renderer: Renderer,
    stats: FrameStats,
    pacing_mode: BenchmarkPacingMode,
    options: CliOptions,
    session_start: Instant,
    sample_started_at: Option<Instant>,
    sample_finished: bool,
}

impl BenchmarkRuntime {
    fn new(event_loop: &ActiveEventLoop, options: CliOptions) -> Result<Self, Box<dyn Error>> {
        let window = Arc::new(
            event_loop.create_window(
                Window::default_attributes()
                    .with_title("MGS Render Benchmark | Initializing...")
                    .with_inner_size(options.resolution),
            )?,
        );

        if !matches!(options.multi_gpu_override, MultiGpuOverride::Off) {
            eprintln!(
                "MGS note: --multi-gpu is ignored (serial single-GPU fallback architecture)."
            );
        }

        let renderer = Renderer::new(Arc::clone(&window), options)?;
        let pacing_mode = choose_pacing_mode(options);
        let stats = FrameStats::new(renderer.size, Duration::from_millis(500));
        let session_start = Instant::now();
        let sample_started_at = if options.warmup_duration.is_zero() {
            Some(session_start)
        } else {
            None
        };
        let mut runtime = Self {
            window,
            renderer,
            stats,
            pacing_mode,
            options,
            session_start,
            sample_started_at,
            sample_finished: false,
        };
        runtime.update_title(0.0, 0.0, 0.0, 0.0);
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
        self.update_phase_before_frame(Instant::now());

        let frame_start = Instant::now();
        let work_units = self.renderer.render(&self.window)?;
        let frame_end = Instant::now();
        let timing = self.stats.record_frame(frame_start, frame_end, work_units);
        self.update_phase_after_frame(frame_end);

        if timing.should_update_title {
            self.update_title(
                timing.instant_fps,
                timing.avg_fps,
                timing.instant_wfps,
                timing.avg_wfps,
            );
        }
        Ok(())
    }

    fn update_title(&mut self, instant_fps: f64, avg_fps: f64, instant_wfps: f64, avg_wfps: f64) {
        let phase = self.phase_label();
        let show_throughput = self.renderer.throughput_target_burst > 1
            || matches!(self.pacing_mode, BenchmarkPacingMode::MaxThroughput);
        let title = if show_throughput {
            format!(
                "MGS Render Benchmark [{}] | WFPS {:.1} | WAvg {:.1} | FPS {:.1} | Frames {} | {} {} {} | Close/Esc to finish",
                phase,
                instant_wfps,
                avg_wfps,
                instant_fps,
                self.stats.total_frames,
                self.renderer.profile.family.short_label(),
                self.renderer.profile.gfx_backend.label(),
                self.renderer.last_fallback.label(),
            )
        } else {
            format!(
                "MGS Render Benchmark [{}] | FPS {:.1} | Avg {:.1} | Frames {} | {} {} {} | Close/Esc to finish",
                phase,
                instant_fps,
                avg_fps,
                self.stats.total_frames,
                self.renderer.profile.family.short_label(),
                self.renderer.profile.gfx_backend.label(),
                self.renderer.last_fallback.label(),
            )
        };
        self.window.set_title(&title);
    }

    fn finish(self) -> BenchmarkSummary {
        let computed = self.stats.compute_summary();
        let score = compute_render_score(&computed, &self.renderer.profile);

        BenchmarkSummary {
            adapter_name: self.renderer.adapter_info.name.clone(),
            backend: format!("{:?}", self.renderer.adapter_info.backend),
            device_type: format!("{:?}", self.renderer.adapter_info.device_type),
            profile: self.renderer.profile,
            present_mode: self.renderer.present_mode,
            resolution: self.renderer.size,
            total_frames: computed.total_frames,
            total_work_units: computed.total_work_units,
            elapsed: computed.elapsed,
            avg_fps: computed.avg_fps,
            peak_fps: computed.peak_fps,
            avg_work_fps: computed.avg_work_fps,
            peak_work_fps: computed.peak_work_fps,
            avg_frame_ms: computed.avg_frame_ms,
            p95_frame_ms: computed.p95_frame_ms,
            p99_frame_ms: computed.p99_frame_ms,
            frame_time_stddev_ms: computed.frame_time_stddev_ms,
            low_1_percent_fps: computed.low_1_percent_fps,
            avg_work_ms: computed.avg_work_ms,
            p95_work_ms: computed.p95_work_ms,
            p99_work_ms: computed.p99_work_ms,
            score,
            fallback_counts: self.renderer.fallback_counts,
            last_tile_count: self.renderer.last_tile_count,
            last_memory_pressure: self.renderer.last_memory_pressure,
            mode_override: self.options.mode_override,
            vsync_override: self.options.vsync_override,
            warmup_duration: self.options.warmup_duration,
            sample_duration: self.options.sample_duration,
            pacing_mode: self.pacing_mode,
        }
    }

    fn preferred_control_flow(&self) -> ControlFlow {
        match self.pacing_mode {
            BenchmarkPacingMode::Stable => ControlFlow::Wait,
            BenchmarkPacingMode::MaxThroughput => ControlFlow::Poll,
        }
    }

    fn use_redraw_chaining(&self) -> bool {
        matches!(self.pacing_mode, BenchmarkPacingMode::Stable)
    }

    fn should_auto_exit(&self) -> bool {
        self.sample_finished
    }

    fn update_phase_before_frame(&mut self, now: Instant) {
        if self.sample_finished || self.sample_started_at.is_some() {
            return;
        }
        if now.duration_since(self.session_start) >= self.options.warmup_duration {
            self.sample_started_at = Some(now);
            self.stats.reset_measurement(now);
        }
    }

    fn update_phase_after_frame(&mut self, now: Instant) {
        if self.sample_finished {
            return;
        }
        if let Some(sample_start) = self.sample_started_at {
            if now.duration_since(sample_start) >= self.options.sample_duration {
                self.sample_finished = true;
            }
        }
    }

    fn phase_label(&self) -> &'static str {
        if self.sample_finished {
            "Done"
        } else if self.sample_started_at.is_some() {
            "Sample"
        } else {
            "Warmup"
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
    profile: MobileGpuProfile,
    bridge: MgsBridge,
    throughput_target_burst: u32,
    throughput_targets: Vec<wgpu::Texture>,
    throughput_target_cursor: usize,
    presented_frames: u64,
    last_fallback: FallbackLevel,
    fallback_counts: [u64; 4],
    last_tile_count: u32,
    last_memory_pressure: bool,
}

impl Renderer {
    fn new(window: Arc<Window>, options: CliOptions) -> Result<Self, Box<dyn Error>> {
        let size = window.inner_size();
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(Arc::clone(&window))?;
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))?;

        let adapter_info = adapter.get_info();
        let profile = MobileGpuProfile::detect(&adapter_info.name);
        let tuning = mgs::MgsTuningProfile::from_profile(&profile);
        let bridge = MgsBridge::with_tuning(profile.clone(), tuning);
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("mgs-render-benchmark-device"),
                required_limits: adapter.limits(),
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
        let prefer_stable_present = match options.vsync_override {
            VsyncOverride::On => true,
            VsyncOverride::Off => false,
            VsyncOverride::Auto => {
                matches!(adapter_info.device_type, wgpu::DeviceType::IntegratedGpu)
            }
        };
        config.present_mode = select_present_mode(
            &capabilities.present_modes,
            prefer_stable_present,
            options.vsync_override,
        );
        if matches!(options.vsync_override, VsyncOverride::Off)
            && !present_mode_allows_uncapped(config.present_mode)
        {
            eprintln!(
                "MGS note: --vsync off requested but selected present mode is {:?}. Driver/compositor may still cap present FPS; use WFPS for throughput.",
                config.present_mode
            );
        }
        config.alpha_mode = capabilities
            .alpha_modes
            .iter()
            .copied()
            .find(|mode| *mode == CompositeAlphaMode::Opaque)
            .unwrap_or(config.alpha_mode);
        config.desired_maximum_frame_latency = 3;
        surface.configure(&device, &config);

        let throughput_target_burst = select_throughput_burst(&profile, options);

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
            profile,
            bridge,
            throughput_target_burst,
            throughput_targets: Vec::new(),
            throughput_target_cursor: 0,
            presented_frames: 0,
            last_fallback: FallbackLevel::TbdrOptimized,
            fallback_counts: [0; 4],
            last_tile_count: 0,
            last_memory_pressure: false,
        })
    }

    fn resize(&mut self, size: PhysicalSize<u32>) {
        if size.width == 0 || size.height == 0 {
            self.size = size;
            self.throughput_targets.clear();
            self.throughput_target_cursor = 0;
            return;
        }
        self.size = size;
        self.config.width = size.width;
        self.config.height = size.height;
        self.surface.configure(&self.device, &self.config);
        self.throughput_targets.clear();
        self.throughput_target_cursor = 0;
        self.presented_frames = 0;
    }

    fn reconfigure(&mut self) {
        if self.size.width == 0 || self.size.height == 0 {
            return;
        }
        self.surface.configure(&self.device, &self.config);
    }

    fn render(&mut self, window: &Window) -> Result<u32, RenderOutcome> {
        if self.size.width == 0 || self.size.height == 0 {
            return Ok(1);
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

        let hint = MpsWorkloadHint {
            transfer_size_kb: (((self.size.width as u64 * self.size.height as u64 * 4) / 1024)
                .clamp(64, 8192)) as u32,
            object_count: ((self.size.width as u64 * self.size.height as u64) / 8192)
                .clamp(64, 4096) as u32,
            target_width: self.size.width,
            target_height: self.size.height,
            latency_budget_ms: match self.present_mode {
                PresentMode::Fifo | PresentMode::AutoVsync => 16,
                _ => 1,
            },
        };
        let plan = self.bridge.translate(hint);
        self.last_fallback = plan.resolved_fallback;
        self.fallback_counts[self.last_fallback as usize] =
            self.fallback_counts[self.last_fallback as usize].saturating_add(1);
        self.last_tile_count = plan.tile_plan.tile_count;
        self.last_memory_pressure = plan.memory_pressure;

        let mut work_units = self.derive_work_units_from_plan(&plan);
        if work_units > 1 {
            work_units = startup_ramp(work_units, self.presented_frames);
        }

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("mgs-render-benchmark-encoder"),
            });

        if work_units > 1 {
            self.ensure_throughput_targets(work_units as usize);
            for i in 0..(work_units - 1) {
                let target_view = self.next_throughput_view();
                let color = animated_clear_color(self.clear_phase + f64::from(i) * 0.071);
                let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("mgs-throughput-pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &target_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(color),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
            }
        }

        {
            let _main_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("mgs-main-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
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

        self.queue.submit(std::iter::once(encoder.finish()));
        window.pre_present_notify();
        frame.present();
        self.presented_frames = self.presented_frames.saturating_add(1);

        Ok(work_units.max(1))
    }

    fn ensure_throughput_targets(&mut self, required_work_units: usize) {
        if self.size.width == 0 || self.size.height == 0 {
            self.throughput_targets.clear();
            return;
        }
        let ring_len = (required_work_units.max(2)).min(12);
        if self.throughput_targets.len() >= ring_len {
            return;
        }
        while self.throughput_targets.len() < ring_len {
            self.throughput_targets
                .push(self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("mgs-throughput-target"),
                    size: wgpu::Extent3d {
                        width: self.size.width.max(1),
                        height: self.size.height.max(1),
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: self.config.format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                }));
        }
    }

    fn next_throughput_view(&mut self) -> wgpu::TextureView {
        let idx = self.throughput_target_cursor % self.throughput_targets.len();
        self.throughput_target_cursor = self.throughput_target_cursor.saturating_add(1);
        self.throughput_targets[idx].create_view(&wgpu::TextureViewDescriptor::default())
    }

    fn derive_work_units_from_plan(&self, plan: &mgs::bridge::MgsBridgePlan) -> u32 {
        let mut units = self.throughput_target_burst.max(1);
        if units > 1 {
            let tile_factor = (plan.tile_plan.tile_count / 120).max(1);
            units = units.saturating_mul(tile_factor);
        }
        if plan.memory_pressure {
            units = (units / 2).max(1);
        }
        units.clamp(1, 24)
    }
}

#[derive(Debug, Clone, Copy)]
struct FrameTiming {
    instant_fps: f64,
    avg_fps: f64,
    instant_wfps: f64,
    avg_wfps: f64,
    should_update_title: bool,
}

struct FrameStats {
    resolution: PhysicalSize<u32>,
    total_frames: u64,
    total_work_units: u64,
    sample_start: Instant,
    frame_times_ms: Vec<f64>,
    work_times_ms: Vec<f64>,
    work_fps_samples: Vec<f64>,
    last_frame_end: Option<Instant>,
    fps_window: [f64; 32],
    fps_window_len: usize,
    fps_window_cursor: usize,
    wfps_window: [f64; 32],
    wfps_window_len: usize,
    wfps_window_cursor: usize,
    last_title_update: Instant,
    title_update_interval: Duration,
}

impl FrameStats {
    fn new(resolution: PhysicalSize<u32>, title_update_interval: Duration) -> Self {
        let now = Instant::now();
        Self {
            resolution,
            total_frames: 0,
            total_work_units: 0,
            sample_start: now,
            frame_times_ms: Vec::with_capacity(64 * 1024),
            work_times_ms: Vec::with_capacity(64 * 1024),
            work_fps_samples: Vec::with_capacity(64 * 1024),
            last_frame_end: None,
            fps_window: [0.0; 32],
            fps_window_len: 0,
            fps_window_cursor: 0,
            wfps_window: [0.0; 32],
            wfps_window_len: 0,
            wfps_window_cursor: 0,
            last_title_update: now,
            title_update_interval,
        }
    }

    fn set_resolution(&mut self, resolution: PhysicalSize<u32>) {
        self.resolution = resolution;
    }

    fn reset_measurement(&mut self, now: Instant) {
        self.sample_start = now;
        self.frame_times_ms.clear();
        self.work_times_ms.clear();
        self.work_fps_samples.clear();
        self.total_frames = 0;
        self.total_work_units = 0;
        self.last_frame_end = None;
        self.fps_window = [0.0; 32];
        self.fps_window_len = 0;
        self.fps_window_cursor = 0;
        self.wfps_window = [0.0; 32];
        self.wfps_window_len = 0;
        self.wfps_window_cursor = 0;
        self.last_title_update = now;
    }

    fn record_frame(
        &mut self,
        frame_start: Instant,
        frame_end: Instant,
        work_units: u32,
    ) -> FrameTiming {
        self.total_frames = self.total_frames.saturating_add(1);
        let work_units = work_units.max(1);
        self.total_work_units = self.total_work_units.saturating_add(u64::from(work_units));
        let frame_ms = frame_end.duration_since(frame_start).as_secs_f64() * 1000.0;
        self.frame_times_ms.push(frame_ms);
        self.work_times_ms.push(frame_ms / f64::from(work_units));

        let instant_fps = if let Some(prev_end) = self.last_frame_end {
            let dt = frame_end.duration_since(prev_end).as_secs_f64();
            if dt > 0.0 {
                1.0 / dt
            } else {
                0.0
            }
        } else {
            0.0
        };
        let instant_wfps = instant_fps * f64::from(work_units);
        self.last_frame_end = Some(frame_end);

        self.fps_window[self.fps_window_cursor] = instant_fps;
        self.fps_window_cursor = (self.fps_window_cursor + 1) % self.fps_window.len();
        if self.fps_window_len < self.fps_window.len() {
            self.fps_window_len += 1;
        }
        self.wfps_window[self.wfps_window_cursor] = instant_wfps;
        self.wfps_window_cursor = (self.wfps_window_cursor + 1) % self.wfps_window.len();
        if self.wfps_window_len < self.wfps_window.len() {
            self.wfps_window_len += 1;
        }
        if instant_wfps > 0.0 {
            self.work_fps_samples.push(instant_wfps);
        }

        let avg_fps = if self.fps_window_len == 0 {
            0.0
        } else {
            self.fps_window[..self.fps_window_len].iter().sum::<f64>() / self.fps_window_len as f64
        };
        let avg_wfps = if self.wfps_window_len == 0 {
            0.0
        } else {
            self.wfps_window[..self.wfps_window_len].iter().sum::<f64>()
                / self.wfps_window_len as f64
        };

        let should_update_title =
            frame_end.duration_since(self.last_title_update) >= self.title_update_interval;
        if should_update_title {
            self.last_title_update = frame_end;
        }

        FrameTiming {
            instant_fps,
            avg_fps,
            instant_wfps,
            avg_wfps,
            should_update_title,
        }
    }

    fn compute_summary(&self) -> ComputedStats {
        let elapsed = if self.total_frames == 0 {
            Duration::ZERO
        } else {
            self.last_frame_end
                .map(|end| end.duration_since(self.sample_start))
                .unwrap_or_default()
        };
        let avg_fps = if elapsed.is_zero() {
            0.0
        } else {
            self.total_frames as f64 / elapsed.as_secs_f64()
        };
        let avg_work_fps = if elapsed.is_zero() {
            0.0
        } else {
            self.total_work_units as f64 / elapsed.as_secs_f64()
        };
        let peak_fps = self
            .frame_times_ms
            .iter()
            .copied()
            .filter(|ms| *ms > 0.0)
            .map(|ms| 1000.0 / ms)
            .fold(0.0, f64::max);
        let peak_work_fps = self.work_fps_samples.iter().copied().fold(0.0, f64::max);
        let avg_frame_ms = mean(&self.frame_times_ms);
        let p95_frame_ms = percentile(self.frame_times_ms.clone(), 0.95);
        let p99_frame_ms = percentile(self.frame_times_ms.clone(), 0.99);
        let frame_time_stddev_ms = stddev(&self.frame_times_ms, avg_frame_ms);
        let low_1_percent_fps = if p99_frame_ms > 0.0 {
            1000.0 / p99_frame_ms
        } else {
            0.0
        };

        let avg_work_ms = mean(&self.work_times_ms);
        let p95_work_ms = percentile(self.work_times_ms.clone(), 0.95);
        let p99_work_ms = percentile(self.work_times_ms.clone(), 0.99);

        ComputedStats {
            total_frames: self.total_frames,
            total_work_units: self.total_work_units,
            elapsed,
            avg_fps,
            peak_fps,
            avg_work_fps,
            peak_work_fps,
            avg_frame_ms,
            p95_frame_ms,
            p99_frame_ms,
            frame_time_stddev_ms,
            low_1_percent_fps,
            avg_work_ms,
            p95_work_ms,
            p99_work_ms,
        }
    }
}

#[derive(Debug, Clone)]
struct ComputedStats {
    total_frames: u64,
    total_work_units: u64,
    elapsed: Duration,
    avg_fps: f64,
    peak_fps: f64,
    avg_work_fps: f64,
    peak_work_fps: f64,
    avg_frame_ms: f64,
    p95_frame_ms: f64,
    p99_frame_ms: f64,
    frame_time_stddev_ms: f64,
    low_1_percent_fps: f64,
    avg_work_ms: f64,
    p95_work_ms: f64,
    p99_work_ms: f64,
}

#[derive(Debug, Clone)]
struct RenderScore {
    score: u32,
    tier: &'static str,
    stability: f64,
    work_stability: f64,
}

#[derive(Debug, Clone)]
struct BenchmarkSummary {
    adapter_name: String,
    backend: String,
    device_type: String,
    profile: MobileGpuProfile,
    present_mode: PresentMode,
    resolution: PhysicalSize<u32>,
    total_frames: u64,
    total_work_units: u64,
    elapsed: Duration,
    avg_fps: f64,
    peak_fps: f64,
    avg_work_fps: f64,
    peak_work_fps: f64,
    avg_frame_ms: f64,
    p95_frame_ms: f64,
    p99_frame_ms: f64,
    frame_time_stddev_ms: f64,
    low_1_percent_fps: f64,
    avg_work_ms: f64,
    p95_work_ms: f64,
    p99_work_ms: f64,
    score: RenderScore,
    fallback_counts: [u64; 4],
    last_tile_count: u32,
    last_memory_pressure: bool,
    mode_override: ModeOverride,
    vsync_override: VsyncOverride,
    warmup_duration: Duration,
    sample_duration: Duration,
    pacing_mode: BenchmarkPacingMode,
}

fn choose_pacing_mode(options: CliOptions) -> BenchmarkPacingMode {
    match options.mode_override {
        ModeOverride::Stable => BenchmarkPacingMode::Stable,
        ModeOverride::Max => BenchmarkPacingMode::MaxThroughput,
        ModeOverride::Auto => match options.vsync_override {
            VsyncOverride::On => BenchmarkPacingMode::Stable,
            VsyncOverride::Off => BenchmarkPacingMode::MaxThroughput,
            VsyncOverride::Auto => BenchmarkPacingMode::Stable,
        },
    }
}

fn select_throughput_burst(profile: &MobileGpuProfile, options: CliOptions) -> u32 {
    match options.mode_override {
        ModeOverride::Stable => 1,
        ModeOverride::Max => match profile.architecture {
            TbdrArchitecture::AppleTbdr => 6,
            TbdrArchitecture::MaliTbdr => 4,
            TbdrArchitecture::FlexRender => 8,
            TbdrArchitecture::PowerVrTbdr => 4,
            TbdrArchitecture::Unknown => 3,
        },
        ModeOverride::Auto => {
            if profile.is_mobile_tbdr() {
                3
            } else {
                2
            }
        }
    }
}

fn startup_ramp(target: u32, presented_frames: u64) -> u32 {
    if target <= 1 {
        return target;
    }
    let ramp_frames = 60.0;
    let progress = ((presented_frames.saturating_add(1)) as f64 / ramp_frames).clamp(0.0, 1.0);
    let eased = progress * progress * (3.0 - 2.0 * progress);
    let scaled = 1.0 + (target.saturating_sub(1) as f64) * eased;
    scaled.round().clamp(1.0, target as f64) as u32
}

fn select_present_mode(
    available: &[PresentMode],
    prefer_stable: bool,
    vsync_override: VsyncOverride,
) -> PresentMode {
    let has = |mode: PresentMode| available.contains(&mode);
    match vsync_override {
        VsyncOverride::On => {
            if has(PresentMode::Fifo) {
                PresentMode::Fifo
            } else {
                available.first().copied().unwrap_or(PresentMode::AutoVsync)
            }
        }
        VsyncOverride::Off => {
            if has(PresentMode::Immediate) {
                PresentMode::Immediate
            } else if has(PresentMode::AutoNoVsync) {
                PresentMode::AutoNoVsync
            } else if has(PresentMode::Mailbox) {
                PresentMode::Mailbox
            } else {
                available
                    .first()
                    .copied()
                    .unwrap_or(PresentMode::AutoNoVsync)
            }
        }
        VsyncOverride::Auto => {
            if prefer_stable && has(PresentMode::Fifo) {
                PresentMode::Fifo
            } else if has(PresentMode::AutoVsync) {
                PresentMode::AutoVsync
            } else {
                available.first().copied().unwrap_or(PresentMode::AutoVsync)
            }
        }
    }
}

fn present_mode_allows_uncapped(mode: PresentMode) -> bool {
    matches!(mode, PresentMode::Immediate | PresentMode::AutoNoVsync)
}

fn animated_clear_color(phase: f64) -> Color {
    // Keep the clear pattern intentionally vivid so users can immediately
    // validate that frame presentation is alive on new drivers/devices.
    let r = 0.12 + 0.32 * phase.sin().abs();
    let g = 0.14 + 0.34 * (phase * 1.37).sin().abs();
    let b = 0.16 + 0.36 * (phase * 0.91).cos().abs();
    Color {
        r: r.clamp(0.0, 1.0),
        g: g.clamp(0.0, 1.0),
        b: b.clamp(0.0, 1.0),
        a: 1.0,
    }
}

fn compute_render_score(stats: &ComputedStats, profile: &MobileGpuProfile) -> RenderScore {
    let resolution_pixels = (stats.total_frames.max(1) as f64).max(1.0);
    let resolution_factor = ((resolution_pixels / 60.0).sqrt() / 100.0).clamp(0.75, 1.35);

    let p95 = stats.p95_frame_ms.max(0.001);
    let p99 = stats.p99_frame_ms.max(0.001);
    let tail_penalty = (p99 / p95).clamp(1.0, 4.0);
    let stability = (1.0 / tail_penalty).clamp(0.20, 1.0);

    let wp95 = stats.p95_work_ms.max(0.001);
    let wp99 = stats.p99_work_ms.max(0.001);
    let work_penalty = (wp99 / wp95).clamp(1.0, 4.0);
    let work_stability = (1.0 / work_penalty).clamp(0.20, 1.0);

    let profile_factor = (profile.estimated_cores as f64 * profile.estimated_bandwidth_gbps as f64)
        .sqrt()
        .clamp(2.0, 96.0)
        / 16.0;

    let score_f = stats.avg_fps
        * resolution_factor
        * (0.60 * stability + 0.40 * work_stability)
        * profile_factor;
    let score = score_f.round().max(0.0) as u32;

    let tier = match score {
        0..=2499 => "D",
        2500..=9999 => "C",
        10000..=29999 => "B",
        30000..=89999 => "A",
        _ => "S",
    };

    RenderScore {
        score,
        tier,
        stability,
        work_stability,
    }
}

fn percentile(mut values: Vec<f64>, p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    let idx = ((values.len() - 1) as f64 * p.clamp(0.0, 1.0)).round() as usize;
    values[idx]
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn stddev(values: &[f64], mean: f64) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let var = values
        .iter()
        .map(|v| {
            let d = *v - mean;
            d * d
        })
        .sum::<f64>()
        / values.len() as f64;
    var.sqrt()
}

fn print_summary(summary: &BenchmarkSummary) {
    println!("--- MGS Render Benchmark Summary ---");
    println!("Adapter: {}", summary.adapter_name);
    println!(
        "Backend: {} | Device type: {} | Present mode: {:?}",
        summary.backend, summary.device_type, summary.present_mode
    );
    println!(
        "Profile: family={} arch={:?} gfx={} cores={} bw={:.1} GB/s tile_mem={} KB",
        summary.profile.family.short_label(),
        summary.profile.architecture,
        summary.profile.gfx_backend.label(),
        summary.profile.estimated_cores,
        summary.profile.estimated_bandwidth_gbps,
        summary.profile.tile_memory_kb
    );
    println!(
        "Resolution: {}x{} | Frames: {} | WorkUnits: {} | Elapsed: {:.3}s",
        summary.resolution.width,
        summary.resolution.height,
        summary.total_frames,
        summary.total_work_units,
        summary.elapsed.as_secs_f64()
    );
    println!(
        "FPS: avg={:.2} peak={:.2} low1%={:.2}",
        summary.avg_fps, summary.peak_fps, summary.low_1_percent_fps
    );
    println!(
        "WFPS: avg={:.2} peak={:.2}",
        summary.avg_work_fps, summary.peak_work_fps
    );
    println!(
        "Frame ms: avg={:.3} p95={:.3} p99={:.3} stddev={:.3}",
        summary.avg_frame_ms,
        summary.p95_frame_ms,
        summary.p99_frame_ms,
        summary.frame_time_stddev_ms
    );
    println!(
        "Work ms/unit: avg={:.3} p95={:.3} p99={:.3}",
        summary.avg_work_ms, summary.p95_work_ms, summary.p99_work_ms
    );
    println!(
        "Fallback counts: tbdr={} simple={} immediate={} sw={} | last_tile_count={} | last_memory_pressure={}",
        summary.fallback_counts[FallbackLevel::TbdrOptimized as usize],
        summary.fallback_counts[FallbackLevel::SimpleTile as usize],
        summary.fallback_counts[FallbackLevel::FullscreenImmediate as usize],
        summary.fallback_counts[FallbackLevel::SoftwareRasterize as usize],
        summary.last_tile_count,
        summary.last_memory_pressure
    );
    println!(
        "Score: {} [{}] | stability={:.2}% work_stability={:.2}%",
        summary.score.score,
        summary.score.tier,
        summary.score.stability * 100.0,
        summary.score.work_stability * 100.0
    );
    println!(
        "Mode={:?} | VSync={:?} | Warmup={:.2}s | Sample={:.2}s | Pacing={:?}",
        summary.mode_override,
        summary.vsync_override,
        summary.warmup_duration.as_secs_f64(),
        summary.sample_duration.as_secs_f64(),
        summary.pacing_mode
    );
}

#[derive(Debug)]
enum RenderOutcome {
    SurfaceLost,
    Outdated,
    OutOfMemory,
    Timeout,
    Other,
}

impl From<SurfaceError> for RenderOutcome {
    fn from(value: SurfaceError) -> Self {
        match value {
            SurfaceError::Lost => Self::SurfaceLost,
            SurfaceError::Outdated => Self::Outdated,
            SurfaceError::OutOfMemory => Self::OutOfMemory,
            SurfaceError::Timeout => Self::Timeout,
            SurfaceError::Other => Self::Other,
        }
    }
}

#[derive(Debug, Clone)]
struct SimpleError(String);

impl fmt::Display for SimpleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl Error for SimpleError {}
