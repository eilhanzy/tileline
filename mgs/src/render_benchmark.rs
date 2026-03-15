use std::env;
use std::error::Error;
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::bridge::MpsWorkloadHint;
use crate::fallback::FallbackLevel;
use crate::runtime::{RuntimeMode, RuntimePacingMode, ThroughputMemoryPolicy, VsyncMode};
use crate::{
    is_unified_memory_profile, select_present_mode, select_throughput_burst, startup_ramp,
    AdaptiveBurstController, AggressiveNoVsyncPolicy, MgsBridge, MobileGpuProfile,
};
use wgpu::{Color, CompositeAlphaMode, PresentMode, SurfaceError, TextureFormat};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

/// Run the MGS render benchmark using CLI args from the current process.
pub fn run_from_env() -> Result<(), Box<dyn Error>> {
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

type BenchmarkPacingMode = RuntimePacingMode;

struct BenchmarkRuntime {
    window: Arc<Window>,
    renderer: Renderer,
    stats: FrameStats,
    pacing_mode: BenchmarkPacingMode,
    min_frame_interval: Duration,
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
        let min_frame_interval = crate::recommended_min_frame_interval(
            &renderer.profile,
            renderer.memory_policy.is_uma(),
            renderer.throughput_target_burst,
            pacing_mode,
        );
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
            min_frame_interval,
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
        self.apply_max_throughput_pacing(frame_start, frame_end);
        Ok(())
    }

    fn update_title(&mut self, instant_fps: f64, avg_fps: f64, instant_wfps: f64, avg_wfps: f64) {
        let phase = self.phase_label();
        let show_throughput = self.renderer.throughput_target_burst > 1
            || matches!(self.pacing_mode, BenchmarkPacingMode::MaxThroughput);
        let burst_suffix = if show_throughput {
            format!(" | BurstCap {}", self.renderer.adaptive_burst_cap())
        } else {
            String::new()
        };
        let vsync_suffix = if self.renderer.no_vsync_policy.enabled {
            format!(
                " | AggNoVsync x{}",
                self.renderer.no_vsync_policy.forced_present_interval.max(1)
            )
        } else {
            String::new()
        };
        let title = if show_throughput {
            format!(
                "MGS Render Benchmark [{}] | WFPS {:.1} | WAvg {:.1} | FPS {:.1} | Frames {}{}{} | {} {} {} | Close/Esc to finish",
                phase,
                instant_wfps,
                avg_wfps,
                instant_fps,
                self.stats.total_frames,
                burst_suffix,
                vsync_suffix,
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
        let adaptive_burst_cap = self.renderer.adaptive_burst_cap();

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
            adaptive_burst_cap,
            aggressive_no_vsync_fallback: self.renderer.no_vsync_policy.enabled,
            forced_present_interval: self.renderer.no_vsync_policy.forced_present_interval,
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

    fn apply_max_throughput_pacing(&self, frame_start: Instant, frame_end: Instant) {
        if !matches!(self.pacing_mode, BenchmarkPacingMode::MaxThroughput) {
            return;
        }
        if self.min_frame_interval.is_zero() {
            return;
        }

        let elapsed = frame_end.saturating_duration_since(frame_start);
        if elapsed >= self.min_frame_interval {
            return;
        }

        let wait_until = frame_start + self.min_frame_interval;
        std::thread::yield_now();
        while Instant::now() < wait_until {
            std::hint::spin_loop();
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
    adaptive_burst: AdaptiveBurstController,
    memory_policy: ThroughputMemoryPolicy,
    throughput_targets: Vec<wgpu::Texture>,
    throughput_target_views: Vec<wgpu::TextureView>,
    throughput_target_signature: Option<ThroughputTargetSignature>,
    throughput_target_cursor: usize,
    no_vsync_main_target: Option<wgpu::Texture>,
    no_vsync_main_view: Option<wgpu::TextureView>,
    no_vsync_main_signature: Option<ThroughputTargetSignature>,
    total_frames_submitted: u64,
    presented_frames: u64,
    no_vsync_policy: AggressiveNoVsyncPolicy,
    last_fallback: FallbackLevel,
    fallback_counts: [u64; 4],
    last_tile_count: u32,
    last_memory_pressure: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ThroughputTargetSignature {
    width: u32,
    height: u32,
    format: TextureFormat,
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
        let uma_shared_memory = is_unified_memory_profile(&profile, &adapter_info);
        let memory_policy = ThroughputMemoryPolicy::new(uma_shared_memory);
        let runtime_mode = runtime_mode_from_override(options.mode_override);
        let vsync_mode = runtime_vsync_from_override(options.vsync_override);
        let tuning = crate::MgsTuningProfile::from_profile(&profile);
        let bridge = MgsBridge::with_tuning(profile.clone(), tuning);
        let supported_limits = adapter.limits();
        let (required_limits, limit_clamp_report) =
            clamp_required_limits_to_supported(wgpu::Limits::default(), &supported_limits);
        if limit_clamp_report.any_clamped() {
            eprintln!(
                "MGS note: device limits clamped for adapter '{}' (1d={}, 2d={}, 3d={}, layers={})",
                adapter_info.name,
                required_limits.max_texture_dimension_1d,
                required_limits.max_texture_dimension_2d,
                required_limits.max_texture_dimension_3d,
                required_limits.max_texture_array_layers
            );
        }
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("mgs-render-benchmark-device"),
                required_limits,
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
            vsync_mode,
        );
        let no_vsync_policy = AggressiveNoVsyncPolicy::from_selection(
            vsync_mode,
            config.present_mode,
            uma_shared_memory,
        );
        if matches!(options.vsync_override, VsyncOverride::Off)
            && !crate::present_mode_allows_uncapped(config.present_mode)
        {
            eprintln!(
                "MGS note: --vsync off requested but selected present mode is {:?}. Driver/compositor may still cap present FPS; use WFPS for throughput.",
                config.present_mode
            );
            if no_vsync_policy.enabled {
                eprintln!(
                "MGS note: aggressive no-vsync fallback enabled (present every {} frame) to bypass compositor throttling.",
                    no_vsync_policy.forced_present_interval
                );
            }
        }
        config.alpha_mode = capabilities
            .alpha_modes
            .iter()
            .copied()
            .find(|mode| *mode == CompositeAlphaMode::Opaque)
            .unwrap_or(config.alpha_mode);
        config.desired_maximum_frame_latency = memory_policy.desired_surface_frame_latency();
        surface.configure(&device, &config);

        let throughput_target_burst =
            select_throughput_burst(&profile, runtime_mode, uma_shared_memory);
        let adaptive_burst = AdaptiveBurstController::new(
            throughput_target_burst,
            runtime_mode,
            config.present_mode,
            uma_shared_memory,
        );

        let mut renderer = Self {
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
            adaptive_burst,
            memory_policy,
            throughput_targets: Vec::new(),
            throughput_target_views: Vec::new(),
            throughput_target_signature: None,
            throughput_target_cursor: 0,
            no_vsync_main_target: None,
            no_vsync_main_view: None,
            no_vsync_main_signature: None,
            total_frames_submitted: 0,
            presented_frames: 0,
            no_vsync_policy,
            last_fallback: FallbackLevel::TbdrOptimized,
            fallback_counts: [0; 4],
            last_tile_count: 0,
            last_memory_pressure: false,
        };

        let preallocate = renderer
            .memory_policy
            .initial_target_count(renderer.throughput_target_burst);
        if preallocate > 0 {
            // Unified memory systems can spike on first target growth; preallocate once.
            renderer.ensure_throughput_targets(preallocate);
        }

        Ok(renderer)
    }

    fn resize(&mut self, size: PhysicalSize<u32>) {
        if size.width == 0 || size.height == 0 {
            self.size = size;
            self.throughput_targets.clear();
            self.throughput_target_views.clear();
            self.throughput_target_signature = None;
            self.no_vsync_main_target = None;
            self.no_vsync_main_view = None;
            self.no_vsync_main_signature = None;
            self.throughput_target_cursor = 0;
            self.total_frames_submitted = 0;
            self.presented_frames = 0;
            return;
        }
        self.size = size;
        self.config.width = size.width;
        self.config.height = size.height;
        self.surface.configure(&self.device, &self.config);
        self.throughput_targets.clear();
        self.throughput_target_views.clear();
        self.throughput_target_signature = None;
        self.no_vsync_main_target = None;
        self.no_vsync_main_view = None;
        self.no_vsync_main_signature = None;
        self.throughput_target_cursor = 0;
        self.total_frames_submitted = 0;
        self.presented_frames = 0;
        self.adaptive_burst.reset(self.throughput_target_burst);
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
        let frame_begin = Instant::now();

        self.clear_phase = (self.clear_phase + 0.0125) % std::f64::consts::TAU;
        let clear_color = animated_clear_color(self.clear_phase);
        let should_present = self
            .no_vsync_policy
            .should_present(self.total_frames_submitted);
        let mut present_frame: Option<wgpu::SurfaceTexture> = None;
        let main_view = if should_present {
            let frame = self
                .surface
                .get_current_texture()
                .map_err(RenderOutcome::from)?;
            let view = frame
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());
            present_frame = Some(frame);
            view
        } else {
            self.no_vsync_main_view()
        };

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
        if let Some(keep_len) = self.memory_policy.pressure_trim_keep_len() {
            if plan.memory_pressure {
                self.trim_throughput_targets(keep_len);
            }
        }

        let mut planned_units = self.derive_work_units_from_plan(&plan);
        if planned_units > 1 {
            planned_units = startup_ramp(planned_units, self.total_frames_submitted);
        }
        let work_units = self.adaptive_burst.resolve(planned_units);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("mgs-render-benchmark-encoder"),
            });

        if work_units > 1 {
            self.ensure_throughput_targets(work_units as usize);
            let throughput_store = self.memory_policy.throughput_store_op();
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
                            store: throughput_store,
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
                    view: &main_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear_color),
                        store: if should_present {
                            wgpu::StoreOp::Store
                        } else {
                            wgpu::StoreOp::Discard
                        },
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        if let Some(frame) = present_frame {
            window.pre_present_notify();
            frame.present();
            self.presented_frames = self.presented_frames.saturating_add(1);
        }
        self.total_frames_submitted = self.total_frames_submitted.saturating_add(1);
        let frame_ms = frame_begin.elapsed().as_secs_f64() * 1000.0;
        self.adaptive_burst
            .observe(frame_ms, work_units, planned_units, plan.memory_pressure);

        Ok(work_units.max(1))
    }

    fn ensure_throughput_targets(&mut self, required_work_units: usize) {
        if self.size.width == 0 || self.size.height == 0 {
            self.throughput_targets.clear();
            self.throughput_target_views.clear();
            self.throughput_target_signature = None;
            return;
        }

        let (target_width, target_height) = self.offscreen_target_dimensions();
        let signature = ThroughputTargetSignature {
            width: target_width,
            height: target_height,
            format: self.config.format,
        };
        if self.throughput_target_signature != Some(signature) {
            self.throughput_targets.clear();
            self.throughput_target_views.clear();
            self.throughput_target_signature = Some(signature);
            self.throughput_target_cursor = 0;
        }

        let ring_len = self.desired_throughput_ring_len(required_work_units);
        if self.memory_policy.is_uma() && self.throughput_targets.len() > ring_len {
            self.throughput_targets.truncate(ring_len);
            self.throughput_target_views.truncate(ring_len);
            if ring_len == 0 || self.throughput_target_cursor >= ring_len {
                self.throughput_target_cursor = 0;
            }
        }
        if self.throughput_targets.len() >= ring_len {
            return;
        }
        while self.throughput_targets.len() < ring_len {
            let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("mgs-throughput-target"),
                size: wgpu::Extent3d {
                    width: target_width,
                    height: target_height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.config.format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            self.throughput_targets.push(texture);
            self.throughput_target_views.push(view);
        }
    }

    fn next_throughput_view(&mut self) -> wgpu::TextureView {
        let idx = self.throughput_target_cursor % self.throughput_target_views.len();
        self.throughput_target_cursor = self.throughput_target_cursor.saturating_add(1);
        self.throughput_target_views[idx].clone()
    }

    fn no_vsync_main_view(&mut self) -> wgpu::TextureView {
        let (target_width, target_height) = self.offscreen_target_dimensions();
        let signature = ThroughputTargetSignature {
            width: target_width,
            height: target_height,
            format: self.config.format,
        };
        let must_recreate = self.no_vsync_main_signature != Some(signature)
            || self.no_vsync_main_target.is_none()
            || self.no_vsync_main_view.is_none();
        if must_recreate {
            let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("mgs-no-vsync-main-target"),
                size: wgpu::Extent3d {
                    width: target_width,
                    height: target_height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: self.config.format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            self.no_vsync_main_target = Some(texture);
            self.no_vsync_main_view = Some(view);
            self.no_vsync_main_signature = Some(signature);
        }
        self.no_vsync_main_view
            .as_ref()
            .expect("no-vsync main target view must be initialized")
            .clone()
    }

    fn derive_work_units_from_plan(&self, plan: &crate::bridge::MgsBridgePlan) -> u32 {
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

    fn adaptive_burst_cap(&self) -> u32 {
        self.adaptive_burst.current_cap()
    }

    fn desired_throughput_ring_len(&self, required_work_units: usize) -> usize {
        self.memory_policy.desired_ring_len(required_work_units)
    }

    fn offscreen_target_dimensions(&self) -> (u32, u32) {
        self.memory_policy
            .offscreen_dimensions(self.size.width, self.size.height)
    }

    fn trim_throughput_targets(&mut self, keep_len: usize) {
        let keep_len = keep_len.max(1);
        if self.throughput_targets.len() <= keep_len {
            return;
        }
        self.throughput_targets.truncate(keep_len);
        self.throughput_target_views.truncate(keep_len);
        if self.throughput_target_cursor >= keep_len {
            self.throughput_target_cursor = 0;
        }
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
    adaptive_burst_cap: u32,
    aggressive_no_vsync_fallback: bool,
    forced_present_interval: u32,
}

fn choose_pacing_mode(options: CliOptions) -> BenchmarkPacingMode {
    crate::choose_pacing_mode(
        runtime_mode_from_override(options.mode_override),
        runtime_vsync_from_override(options.vsync_override),
    )
}

fn runtime_mode_from_override(mode: ModeOverride) -> RuntimeMode {
    match mode {
        ModeOverride::Auto => RuntimeMode::Auto,
        ModeOverride::Stable => RuntimeMode::Stable,
        ModeOverride::Max => RuntimeMode::MaxThroughput,
    }
}

fn runtime_vsync_from_override(vsync: VsyncOverride) -> VsyncMode {
    match vsync {
        VsyncOverride::Auto => VsyncMode::Auto,
        VsyncOverride::On => VsyncMode::On,
        VsyncOverride::Off => VsyncMode::Off,
    }
}

fn animated_clear_color(phase: f64) -> Color {
    // Use a full hue wheel instead of abs(sin/cos) channel waves to avoid
    // persistent lavender/purple bias on some driver + swapchain paths.
    let hue = phase.rem_euclid(std::f64::consts::TAU) / std::f64::consts::TAU;
    let (r, g, b) = hsv_to_rgb(hue, 0.78, 0.92);
    Color { r, g, b, a: 1.0 }
}

fn hsv_to_rgb(h: f64, s: f64, v: f64) -> (f64, f64, f64) {
    let h = h.rem_euclid(1.0);
    let s = s.clamp(0.0, 1.0);
    let v = v.clamp(0.0, 1.0);
    if s <= f64::EPSILON {
        return (v, v, v);
    }

    let sector = (h * 6.0).floor();
    let f = h * 6.0 - sector;
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    match (sector as i32).rem_euclid(6) {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
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
        "Mode={:?} | VSync={:?} | Warmup={:.2}s | Sample={:.2}s | Pacing={:?} | BurstCap={} | AggNoVsync={} | PresentEvery={}",
        summary.mode_override,
        summary.vsync_override,
        summary.warmup_duration.as_secs_f64(),
        summary.sample_duration.as_secs_f64(),
        summary.pacing_mode,
        summary.adaptive_burst_cap,
        summary.aggressive_no_vsync_fallback,
        summary.forced_present_interval
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

#[derive(Debug, Clone, Copy, Default)]
struct DeviceLimitClampReport {
    max_texture_dimension_1d: bool,
    max_texture_dimension_2d: bool,
    max_texture_dimension_3d: bool,
    max_texture_array_layers: bool,
}

impl DeviceLimitClampReport {
    #[inline]
    fn any_clamped(self) -> bool {
        self.max_texture_dimension_1d
            || self.max_texture_dimension_2d
            || self.max_texture_dimension_3d
            || self.max_texture_array_layers
    }
}

fn clamp_required_limits_to_supported(
    mut requested: wgpu::Limits,
    supported: &wgpu::Limits,
) -> (wgpu::Limits, DeviceLimitClampReport) {
    let mut report = DeviceLimitClampReport::default();

    if requested.max_texture_dimension_1d > supported.max_texture_dimension_1d {
        requested.max_texture_dimension_1d = supported.max_texture_dimension_1d;
        report.max_texture_dimension_1d = true;
    }
    if requested.max_texture_dimension_2d > supported.max_texture_dimension_2d {
        requested.max_texture_dimension_2d = supported.max_texture_dimension_2d;
        report.max_texture_dimension_2d = true;
    }
    if requested.max_texture_dimension_3d > supported.max_texture_dimension_3d {
        requested.max_texture_dimension_3d = supported.max_texture_dimension_3d;
        report.max_texture_dimension_3d = true;
    }
    if requested.max_texture_array_layers > supported.max_texture_array_layers {
        requested.max_texture_array_layers = supported.max_texture_array_layers;
        report.max_texture_array_layers = true;
    }

    (requested, report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamps_3d_dimension_to_supported_limit() {
        let requested = wgpu::Limits::default();
        let mut supported = wgpu::Limits::default();
        supported.max_texture_dimension_3d = 512;

        let (clamped, report) = clamp_required_limits_to_supported(requested, &supported);
        assert!(report.max_texture_dimension_3d);
        assert_eq!(clamped.max_texture_dimension_3d, 512);
    }

    #[test]
    fn no_clamp_when_supported_limits_are_sufficient() {
        let requested = wgpu::Limits::default();
        let mut supported = requested.clone();
        supported.max_texture_dimension_3d = requested.max_texture_dimension_3d.saturating_add(1);

        let (_clamped, report) = clamp_required_limits_to_supported(requested, &supported);
        assert!(!report.any_clamped());
    }
}
