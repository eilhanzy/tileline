use std::env;
use std::error::Error;
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::{
    safe_default_required_limits_for_adapter, GmsRuntimeTuningProfile, GpuInventory,
    MultiGpuExecutor, MultiGpuExecutorConfig, MultiGpuExecutorSummary, MultiGpuInitPolicy,
    MultiGpuWorkloadRequest,
};
use wgpu::{Color, CompositeAlphaMode, PresentMode, SurfaceError, TextureFormat};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

/// Run the GMS render benchmark from process CLI arguments.
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
    println!("GMS Render Benchmark");
    println!("Usage: cargo run -p gms --example render_benchmark -- [options]");
    println!("Options:");
    println!("  --mode auto|stable|max      Frame pacing mode (default: auto)");
    println!("  --vsync auto|on|off         Present mode preference (default: auto)");
    println!("  --multi-gpu auto|on|off     Enable explicit multi-GPU helper lane (default: off)");
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
                let control_flow = runtime.preferred_control_flow();
                self.runtime = Some(runtime);
                event_loop.set_control_flow(control_flow);
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

                if runtime.use_redraw_chaining() {
                    runtime.window.request_redraw();
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(runtime) = self.runtime.as_ref() {
            event_loop.set_control_flow(runtime.preferred_control_flow());

            if runtime.use_redraw_chaining() {
                if runtime.stats.frames == 0 {
                    runtime.window.request_redraw();
                }
            } else {
                runtime.window.request_redraw();
            }
        } else {
            event_loop.set_control_flow(ControlFlow::Wait);
        }

        if let Some(runtime) = self.runtime.as_ref() {
            if runtime.should_auto_exit() {
                self.exit_requested = true;
                event_loop.exit();
            }
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
    multi_gpu: Option<MultiGpuExecutor>,
    stats: FrameStats,
    adapter_profile: Option<crate::GpuAdapterProfile>,
    gms_inventory: GpuInventory,
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
                    .with_title("GMS Render Benchmark | Initializing...")
                    .with_inner_size(options.resolution),
            )?,
        );

        let renderer = Renderer::new(Arc::clone(&window), options)?;
        let gms_inventory = GpuInventory::discover();
        let adapter_profile = match_inventory_profile(&gms_inventory, &renderer.adapter_info);
        let pacing_mode =
            choose_pacing_mode(adapter_profile.as_ref(), &renderer.adapter_info, options);
        let multi_gpu = if matches!(options.multi_gpu_override, MultiGpuOverride::Off)
            || (matches!(options.multi_gpu_override, MultiGpuOverride::Auto)
                && matches!(options.mode_override, ModeOverride::Stable))
        {
            None
        } else {
            let workload_request = build_multi_gpu_workload_request(&renderer, options);
            let policy = match options.multi_gpu_override {
                MultiGpuOverride::Auto => MultiGpuInitPolicy::Auto,
                MultiGpuOverride::On => MultiGpuInitPolicy::Force,
                MultiGpuOverride::Off => unreachable!(),
            };

            MultiGpuExecutor::try_new(MultiGpuExecutorConfig {
                policy,
                primary_adapter_info: renderer.adapter_info.clone(),
                inventory: gms_inventory.clone(),
                primary_device: renderer.device.clone(),
                frame_width: renderer.size.width,
                frame_height: renderer.size.height,
                secondary_offscreen_format: TextureFormat::Rgba8Unorm,
                primary_work_units_per_present: renderer.work_units_per_present(),
                workload_request,
                auto_min_projected_gain_pct: 5.0,
            })?
        };

        let mut stats = FrameStats::new(
            renderer.size,
            renderer.present_mode,
            frame_stats_timing_capacity(&renderer.runtime_tuning),
            title_update_interval(&renderer.runtime_tuning),
        );
        stats.last_title_update = Instant::now();
        let session_start = Instant::now();
        let sample_started_at = if options.warmup_duration.is_zero() {
            Some(session_start)
        } else {
            None
        };

        let mut runtime = Self {
            window,
            renderer,
            multi_gpu,
            stats,
            adapter_profile,
            gms_inventory,
            pacing_mode,
            options,
            session_start,
            sample_started_at,
            sample_finished: false,
        };

        runtime.update_title(0.0, 0.0);
        Ok(runtime)
    }

    fn resize(&mut self, size: PhysicalSize<u32>) {
        self.renderer.resize(size);
        if let Some(multi_gpu) = self.multi_gpu.as_mut() {
            multi_gpu.resize(size.width, size.height);
        }
        self.stats.set_resolution(self.renderer.size);
    }

    fn reconfigure_surface(&mut self) {
        self.renderer.reconfigure();
    }

    fn render_frame(&mut self) -> Result<(), RenderOutcome> {
        self.update_benchmark_phase_before_frame(Instant::now());

        let frame_start = Instant::now();
        let primary_work_units = self.renderer.render(&self.window)?;
        let secondary_work_units = self
            .multi_gpu
            .as_mut()
            .map(|multi_gpu| multi_gpu.submit_frame())
            .unwrap_or(0);
        let work_units = primary_work_units.saturating_add(secondary_work_units);

        let frame_end = Instant::now();
        let timing = self.stats.record_frame(frame_start, frame_end, work_units);
        self.update_benchmark_phase_after_frame(frame_end);

        if timing.should_update_title {
            self.update_title(timing.display_fps, timing.avg_fps);
        }

        Ok(())
    }

    fn update_title(&mut self, instant_fps: f64, avg_fps: f64) {
        let gms_score = self.adapter_profile.as_ref().map(|p| p.score).unwrap_or(0);
        let compute_unit_suffix = self
            .adapter_profile
            .as_ref()
            .map(compact_compute_unit_title_suffix)
            .unwrap_or_default();
        let phase = self.phase_label();
        let burst = self.renderer.synthetic_work_units_per_present();
        let secondary_burst = self
            .multi_gpu
            .as_ref()
            .map(|multi_gpu| multi_gpu.secondary_work_units_per_present())
            .unwrap_or(0);
        let total_burst = burst.saturating_add(secondary_burst);
        let fps_label = if total_burst > 1 { "WFPS" } else { "FPS" };
        let burst_suffix = if burst > 1 {
            format!(" | Burst x{}", burst)
        } else {
            String::new()
        };
        let multi_gpu_suffix = if secondary_burst > 0 {
            format!(" | MGPU +{}", secondary_burst)
        } else {
            String::new()
        };
        let title = format!(
            "GMS Render Benchmark [{}] | {} {:.1} | Avg {:.1} | Frames {} | GMS {}{}{}{} | Close/Esc to finish",
            phase,
            fps_label,
            instant_fps,
            avg_fps,
            self.stats.frames,
            gms_score,
            compute_unit_suffix,
            burst_suffix,
            multi_gpu_suffix
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

        let score = compute_render_benchmark_score(
            &computed,
            gms_score,
            self.pacing_mode,
            &self.renderer.runtime_tuning,
        );
        let multi_gpu_summary = self.multi_gpu.as_ref().map(|multi_gpu| multi_gpu.summary());
        let (estimated_compute_units, compute_unit_short_label, compute_unit_display_label) = self
            .adapter_profile
            .as_ref()
            .map(|profile| {
                let (short_label, count) = profile.compute_unit_summary();
                (
                    Some(count),
                    Some(short_label),
                    Some(profile.compute_unit_kind.display_label()),
                )
            })
            .unwrap_or((None, None, None));
        let compute_unit_source = self
            .adapter_profile
            .as_ref()
            .map(|profile| profile.compute_unit_source.short_label());
        let compute_unit_probe_note = self
            .adapter_profile
            .as_ref()
            .and_then(|profile| profile.compute_unit_probe_note.clone());
        let arm_shader_core_count = self
            .adapter_profile
            .as_ref()
            .and_then(|profile| profile.arm_shader_core_count);

        BenchmarkSummary {
            adapter_name,
            backend: format!("{:?}", self.renderer.adapter_info.backend),
            device_type: format!("{:?}", self.renderer.adapter_info.device_type),
            estimated_compute_units,
            compute_unit_short_label,
            compute_unit_display_label,
            compute_unit_source,
            compute_unit_probe_note,
            arm_shader_core_count,
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
            avg_work_ms: computed.avg_work_ms,
            p95_work_ms: computed.p95_work_ms,
            p99_work_ms: computed.p99_work_ms,
            work_time_stddev_ms: computed.work_time_stddev_ms,
            gms_hardware_score: gms_score,
            total_discovered_gpus: self.gms_inventory.adapters.len(),
            render_score: score.score,
            render_tier: score.tier,
            score_breakdown: score,
            work_units_per_present: self.renderer.synthetic_work_units_per_present(),
            primary_passes_per_work_unit: self.renderer.passes_per_work_unit(),
            multi_gpu: multi_gpu_summary,
            pacing_mode: self.pacing_mode,
            warmup_duration: self.options.warmup_duration,
            sample_duration: self.options.sample_duration,
            mode_override: self.options.mode_override,
            vsync_override: self.options.vsync_override,
            multi_gpu_override: self.options.multi_gpu_override,
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

    fn update_benchmark_phase_before_frame(&mut self, now: Instant) {
        if self.sample_finished || self.sample_started_at.is_some() {
            return;
        }

        if now.duration_since(self.session_start) >= self.options.warmup_duration {
            self.sample_started_at = Some(now);
            self.stats.reset_measurement(now);
        }
    }

    fn update_benchmark_phase_after_frame(&mut self, now: Instant) {
        if self.sample_finished {
            return;
        }

        if let Some(sample_started_at) = self.sample_started_at {
            if now.duration_since(sample_started_at) >= self.options.sample_duration {
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

fn build_multi_gpu_workload_request(
    renderer: &Renderer,
    options: CliOptions,
) -> MultiGpuWorkloadRequest {
    let pixels = (renderer.size.width.max(1) as u64) * (renderer.size.height.max(1) as u64);
    let target_frame_budget_ms = match options.mode_override {
        ModeOverride::Stable => 16.67,
        ModeOverride::Max => 0.26,
        ModeOverride::Auto => {
            if renderer.synthetic_work_units_per_present() > 1 {
                0.50
            } else {
                8.33
            }
        }
    };

    MultiGpuWorkloadRequest {
        sampled_processing_jobs: (pixels / 2048).clamp(128, 8192) as u32,
        object_updates: (pixels / 1024).clamp(256, 16384) as u32,
        physics_jobs: (pixels / 3072).clamp(64, 4096) as u32,
        ui_jobs: (pixels / 8192).clamp(16, 1024) as u32,
        post_fx_jobs: (pixels / 6144).clamp(32, 2048) as u32,
        bytes_per_sampled_job: 4096,
        bytes_per_object: 256,
        bytes_per_physics_job: 1024,
        bytes_per_ui_job: 512,
        bytes_per_post_fx_job: 1024,
        processed_texture_bytes_per_frame: pixels
            .saturating_mul(4)
            .clamp(512 * 1024, 64 * 1024 * 1024),
        base_workgroup_size: 64,
        target_frame_budget_ms,
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
    throughput_burst: Option<ThroughputBurst>,
    throughput_targets: Vec<wgpu::Texture>,
    throughput_target_cursor: usize,
    primary_stress: PrimaryStressKernel,
    runtime_tuning: GmsRuntimeTuningProfile,
    presented_frames: u64,
}

#[derive(Debug, Clone, Copy)]
struct ThroughputBurst {
    work_units_per_present: u32,
    passes_per_work_unit: u32,
    offscreen_target_ring_len: usize,
}

struct PrimaryStressKernel {
    pipeline: wgpu::ComputePipeline,
    bind_groups: Vec<wgpu::BindGroup>,
    params_buffer: wgpu::Buffer,
    element_count: u32,
    dispatch_groups: u32,
    ring_cursor: usize,
}

impl PrimaryStressKernel {
    fn new(
        device: &wgpu::Device,
        adapter_info: &wgpu::AdapterInfo,
        size: PhysicalSize<u32>,
        ring_len_hint: usize,
    ) -> Self {
        let element_count = select_primary_stress_elements(adapter_info, size);
        let dispatch_groups = ceil_div_u32(element_count.max(1), 128).max(1);
        let ring_len = ring_len_hint.clamp(2, 8);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("gms-primary-stress-shader"),
            source: wgpu::ShaderSource::Wgsl(PRIMARY_STRESS_WGSL.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gms-primary-stress-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("gms-primary-stress-pipeline-layout"),
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("gms-primary-stress-pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gms-primary-stress-params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let payload_bytes = (element_count as u64).saturating_mul(16).max(16);
        let mut bind_groups = Vec::with_capacity(ring_len);
        for idx in 0..ring_len {
            let payload = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gms-primary-stress-payload"),
                size: payload_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gms-primary-stress-bg"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: payload.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });
            let _ = idx;
            bind_groups.push(bind_group);
        }

        Self {
            pipeline,
            bind_groups,
            params_buffer,
            element_count,
            dispatch_groups,
            ring_cursor: 0,
        }
    }

    fn write_params(
        &self,
        queue: &wgpu::Queue,
        phase: f32,
        amplitude: f32,
        len: u32,
        iterations: u32,
    ) {
        let mut bytes = [0u8; 16];
        bytes[0..4].copy_from_slice(&phase.to_bits().to_ne_bytes());
        bytes[4..8].copy_from_slice(&amplitude.to_bits().to_ne_bytes());
        bytes[8..12].copy_from_slice(&len.to_ne_bytes());
        bytes[12..16].copy_from_slice(&iterations.max(1).to_ne_bytes());
        queue.write_buffer(&self.params_buffer, 0, &bytes);
    }

    fn encode_one(&mut self, encoder: &mut wgpu::CommandEncoder) {
        if self.bind_groups.is_empty() {
            return;
        }
        let idx = self.ring_cursor % self.bind_groups.len();
        self.ring_cursor = self.ring_cursor.wrapping_add(1);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gms-primary-stress-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_groups[idx], &[]);
        pass.dispatch_workgroups(self.dispatch_groups, 1, 1);
    }
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
        let runtime_tuning = GmsRuntimeTuningProfile::from_adapter_info(&adapter_info);
        let (required_limits, limit_clamp_report) =
            safe_default_required_limits_for_adapter(&adapter);
        if limit_clamp_report.any_clamped() {
            eprintln!(
                "GMS: clamped requested device limits to adapter support (1D={} 2D={} 3D={} layers={})",
                required_limits.max_texture_dimension_1d,
                required_limits.max_texture_dimension_2d,
                required_limits.max_texture_dimension_3d,
                required_limits.max_texture_array_layers,
            );
        }
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("gms-render-benchmark-device"),
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
            options.vsync_override,
        );
        config.alpha_mode = capabilities
            .alpha_modes
            .iter()
            .copied()
            .find(|mode| *mode == CompositeAlphaMode::Opaque)
            .unwrap_or(config.alpha_mode);
        // Unified-memory systems benefit from a little more frame buffering to reduce
        // transient compositor/present jitter showing up in the benchmark.
        config.desired_maximum_frame_latency = runtime_tuning.recommended_surface_frame_latency;
        surface.configure(&device, &config);

        let throughput_burst =
            select_throughput_burst(&adapter_info, &runtime_tuning, options, config.present_mode);
        let primary_stress = PrimaryStressKernel::new(
            &device,
            &adapter_info,
            size,
            throughput_burst
                .map(|mode| mode.offscreen_target_ring_len)
                .unwrap_or(2),
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
            throughput_burst,
            throughput_targets: Vec::new(),
            throughput_target_cursor: 0,
            primary_stress,
            runtime_tuning,
            presented_frames: 0,
        };

        renderer.prewarm_throughput_resources();
        Ok(renderer)
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
        self.throughput_targets.clear();
        self.throughput_target_cursor = 0;
        self.primary_stress = PrimaryStressKernel::new(
            &self.device,
            &self.adapter_info,
            size,
            self.throughput_burst
                .map(|mode| mode.offscreen_target_ring_len)
                .unwrap_or(2),
        );
        self.presented_frames = 0;
        self.surface.configure(&self.device, &self.config);
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

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gms-render-benchmark-encoder"),
            });

        let work_units_per_present = self.effective_work_units_per_present();
        let passes_per_work_unit = self.passes_per_work_unit();
        let total_stress_passes = work_units_per_present.max(1);
        let synthetic_work_units = total_stress_passes.saturating_mul(passes_per_work_unit);
        self.primary_stress.write_params(
            &self.queue,
            self.clear_phase as f32,
            0.92,
            self.primary_stress.element_count,
            passes_per_work_unit,
        );
        for _ in 0..total_stress_passes {
            self.primary_stress.encode_one(&mut encoder);
        }
        self.clear_phase = (self.clear_phase + 0.0125) % std::f64::consts::TAU;
        encode_clear_pass(&mut encoder, &view, animated_clear_color(self.clear_phase));

        self.queue.submit(Some(encoder.finish()));
        window.pre_present_notify();
        frame.present();
        self.presented_frames = self.presented_frames.saturating_add(1);
        Ok(synthetic_work_units)
    }

    fn work_units_per_present(&self) -> u32 {
        self.throughput_burst
            .map(|mode| mode.work_units_per_present)
            .unwrap_or(1)
    }

    fn passes_per_work_unit(&self) -> u32 {
        self.throughput_burst
            .map(|mode| mode.passes_per_work_unit)
            .unwrap_or(1)
    }

    fn synthetic_work_units_per_present(&self) -> u32 {
        self.work_units_per_present()
            .saturating_mul(self.passes_per_work_unit())
    }

    fn effective_work_units_per_present(&self) -> u32 {
        self.runtime_tuning
            .effective_throughput_work_units_per_present(
                self.work_units_per_present(),
                self.presented_frames,
            )
    }

    fn current_throughput_target(&mut self) -> Option<&wgpu::Texture> {
        if self.throughput_targets.is_empty() {
            return None;
        }

        let index = self.throughput_target_cursor % self.throughput_targets.len();
        self.throughput_target_cursor =
            (self.throughput_target_cursor + 1) % self.throughput_targets.len();
        self.throughput_targets.get(index)
    }

    fn ensure_throughput_target_ring(&mut self) {
        if self.size.width == 0 || self.size.height == 0 {
            self.throughput_targets.clear();
            self.throughput_target_cursor = 0;
            return;
        }

        let desired_ring_len = self
            .throughput_burst
            .map(|mode| mode.offscreen_target_ring_len.max(1))
            .unwrap_or(1);

        let needs_recreate = self.throughput_targets.len() != desired_ring_len
            || self.throughput_targets.iter().any(|texture| {
                let extent = texture.size();
                extent.width != self.size.width
                    || extent.height != self.size.height
                    || texture.format() != self.config.format
            });

        if !needs_recreate {
            return;
        }

        self.throughput_targets.clear();
        self.throughput_targets.reserve(desired_ring_len);
        for slot in 0..desired_ring_len {
            self.throughput_targets
                .push(self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some(match self.runtime_tuning.prefer_unified_memory_tuning {
                        true => "gms-render-benchmark-throughput-target-uma",
                        false => "gms-render-benchmark-throughput-target",
                    }),
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
            let _ = slot;
        }
        self.throughput_target_cursor = 0;
    }

    fn prewarm_throughput_resources(&mut self) {
        if self.work_units_per_present() <= 1
            || self.runtime_tuning.throughput_startup_prewarm_submits == 0
        {
            return;
        }

        self.ensure_throughput_target_ring();
        if self.throughput_targets.is_empty() {
            return;
        }

        let prewarm_submits = self.runtime_tuning.startup_prewarm_submits_for_ring(
            self.work_units_per_present(),
            self.throughput_targets.len(),
        );

        for _ in 0..prewarm_submits {
            let Some(target) = self.current_throughput_target() else {
                break;
            };
            let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("gms-render-benchmark-throughput-prewarm-encoder"),
                });
            for _ in 0..self.passes_per_work_unit() {
                self.clear_phase = (self.clear_phase + 0.00625) % std::f64::consts::TAU;
                encode_clear_pass(
                    &mut encoder,
                    &target_view,
                    animated_clear_color(self.clear_phase),
                );
            }
            self.queue.submit(Some(encoder.finish()));
        }

        let _ = self.device.poll(wgpu::PollType::Poll);
    }
}

fn animated_clear_color(phase: f64) -> Color {
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

fn encode_clear_pass(encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView, color: Color) {
    let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("gms-render-benchmark-pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            depth_slice: None,
            resolve_target: None,
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

fn select_present_mode(
    modes: &[PresentMode],
    prefer_stable: bool,
    vsync_override: VsyncOverride,
) -> PresentMode {
    if matches!(vsync_override, VsyncOverride::On) {
        for preferred in [
            PresentMode::AutoVsync,
            PresentMode::Fifo,
            PresentMode::Mailbox,
        ] {
            if modes.contains(&preferred) {
                return preferred;
            }
        }
    }

    if matches!(vsync_override, VsyncOverride::Off) {
        for preferred in [
            PresentMode::AutoNoVsync,
            PresentMode::Immediate,
            PresentMode::Mailbox,
        ] {
            if modes.contains(&preferred) {
                return preferred;
            }
        }
    }

    let stable_order = [
        PresentMode::AutoVsync,
        PresentMode::Fifo,
        PresentMode::Mailbox,
        PresentMode::AutoNoVsync,
        PresentMode::Immediate,
    ];
    let throughput_order = [
        PresentMode::AutoNoVsync,
        PresentMode::Immediate,
        PresentMode::Mailbox,
        PresentMode::AutoVsync,
        PresentMode::Fifo,
    ];

    for preferred in if prefer_stable {
        stable_order
    } else {
        throughput_order
    } {
        if modes.contains(&preferred) {
            return preferred;
        }
    }
    PresentMode::Fifo
}

fn select_throughput_burst(
    adapter_info: &wgpu::AdapterInfo,
    runtime_tuning: &GmsRuntimeTuningProfile,
    options: CliOptions,
    present_mode: PresentMode,
) -> Option<ThroughputBurst> {
    let mode_allows_burst = match options.mode_override {
        ModeOverride::Stable => false,
        ModeOverride::Max => true,
        ModeOverride::Auto => !matches!(
            adapter_info.device_type,
            wgpu::DeviceType::IntegratedGpu | wgpu::DeviceType::Cpu
        ),
    };

    if !mode_allows_burst || matches!(options.vsync_override, VsyncOverride::On) {
        return None;
    }

    let base_units = match adapter_info.device_type {
        wgpu::DeviceType::DiscreteGpu => 64,
        wgpu::DeviceType::VirtualGpu => 16,
        _ => runtime_tuning.integrated_throughput_burst_work_units,
    };
    let max_units = match adapter_info.device_type {
        wgpu::DeviceType::DiscreteGpu => 768,
        wgpu::DeviceType::VirtualGpu => 128,
        _ => 384,
    };

    let pixels_per_present = (options.resolution.width.max(1) as u64)
        .saturating_mul(options.resolution.height.max(1) as u64)
        .max(1);
    let vsync_locked = matches!(
        present_mode,
        PresentMode::Fifo | PresentMode::FifoRelaxed | PresentMode::AutoVsync
    );
    let target_pixels_per_present = match adapter_info.device_type {
        wgpu::DeviceType::DiscreteGpu => match options.mode_override {
            ModeOverride::Max => 320_000_000u64,
            _ => 220_000_000u64,
        },
        wgpu::DeviceType::VirtualGpu => 90_000_000u64,
        _ => {
            if vsync_locked {
                140_000_000u64
            } else {
                100_000_000u64
            }
        }
    };

    let scaled_units = ((target_pixels_per_present + pixels_per_present - 1) / pixels_per_present)
        .clamp(1, max_units as u64) as u32;
    let work_units_per_present = scaled_units.clamp(base_units.max(1), max_units.max(base_units));
    let passes_per_work_unit =
        select_primary_passes_per_work_unit(adapter_info, options, present_mode);

    Some(ThroughputBurst {
        work_units_per_present,
        passes_per_work_unit,
        offscreen_target_ring_len: runtime_tuning.throughput_offscreen_target_ring_len,
    })
}

fn select_primary_passes_per_work_unit(
    adapter_info: &wgpu::AdapterInfo,
    options: CliOptions,
    present_mode: PresentMode,
) -> u32 {
    let pixels_per_present = (options.resolution.width.max(1) as u64)
        .saturating_mul(options.resolution.height.max(1) as u64)
        .max(1);
    let vsync_locked = matches!(
        present_mode,
        PresentMode::Fifo | PresentMode::FifoRelaxed | PresentMode::AutoVsync
    );

    let (target_pixels_per_wu, max_passes) = match adapter_info.device_type {
        wgpu::DeviceType::DiscreteGpu => (
            match options.mode_override {
                ModeOverride::Max => 6_000_000u64,
                ModeOverride::Stable => 2_000_000u64,
                ModeOverride::Auto => 4_000_000u64,
            },
            12u32,
        ),
        wgpu::DeviceType::VirtualGpu => (3_000_000u64, 8u32),
        _ => (
            if vsync_locked {
                2_500_000u64
            } else {
                2_000_000u64
            },
            10u32,
        ),
    };

    ((target_pixels_per_wu + pixels_per_present - 1) / pixels_per_present)
        .clamp(1, max_passes as u64) as u32
}

fn select_primary_stress_elements(
    adapter_info: &wgpu::AdapterInfo,
    size: PhysicalSize<u32>,
) -> u32 {
    let pixels = (size.width.max(1) as u64)
        .saturating_mul(size.height.max(1) as u64)
        .max(262_144);
    match adapter_info.device_type {
        wgpu::DeviceType::DiscreteGpu => {
            (pixels.saturating_mul(4)).clamp(1_048_576, 8_388_608) as u32
        }
        wgpu::DeviceType::VirtualGpu => (pixels.saturating_mul(2)).clamp(524_288, 2_097_152) as u32,
        _ => (pixels.saturating_mul(2)).clamp(524_288, 4_194_304) as u32,
    }
}

fn ceil_div_u32(numerator: u32, denominator: u32) -> u32 {
    if denominator == 0 {
        0
    } else {
        (numerator.saturating_add(denominator - 1)) / denominator
    }
}

const PRIMARY_STRESS_WGSL: &str = r#"
struct StressParams {
    phase_bits: u32,
    amplitude_bits: u32,
    len: u32,
    iterations: u32,
};

@group(0) @binding(0) var<storage, read_write> payload: array<vec4<f32>>;
@group(0) @binding(1) var<uniform> params: StressParams;

@compute @workgroup_size(128, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.len) {
        return;
    }

    let phase = bitcast<f32>(params.phase_bits);
    let amplitude = bitcast<f32>(params.amplitude_bits);
    let prev = payload[i];
    var acc = prev;
    let iters = max(params.iterations, 1u);
    for (var it: u32 = 0u; it < iters; it = it + 1u) {
        let x = f32(i) * 0.0000152587890625 + phase + f32(it) * 0.03125;
        let wave = sin(x) * cos(x * 1.37);
        let mix_target = vec4<f32>(wave * amplitude, wave * 0.73, wave * 0.41, 1.0);
        acc = mix(acc, mix_target, 0.35);
    }
    payload[i] = acc;
}
"#;

fn frame_stats_timing_capacity(tuning: &GmsRuntimeTuningProfile) -> usize {
    tuning.benchmark_timing_capacity.max(16_384)
}

fn title_update_interval(tuning: &GmsRuntimeTuningProfile) -> Duration {
    Duration::from_millis(tuning.benchmark_title_update_interval_ms.max(100))
}

fn match_inventory_profile(
    inventory: &GpuInventory,
    adapter_info: &wgpu::AdapterInfo,
) -> Option<crate::GpuAdapterProfile> {
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
    title_update_interval: Duration,
    frames: u64,
    render_durations_ms: Vec<f64>,
    frame_intervals_ms: Vec<f64>,
    peak_fps: f64,
    resolution: PhysicalSize<u32>,
}

struct FrameTiming {
    display_fps: f64,
    avg_fps: f64,
    should_update_title: bool,
}

impl FrameStats {
    fn new(
        resolution: PhysicalSize<u32>,
        _present_mode: PresentMode,
        timing_capacity: usize,
        title_update_interval: Duration,
    ) -> Self {
        let now = Instant::now();
        Self {
            benchmark_start: now,
            last_frame_presented_at: None,
            last_title_update: now,
            title_update_interval,
            frames: 0,
            // Avoid vector growth during short benchmarks (especially high-WFPS throughput mode)
            // because realloc spikes can poison p95/p99 stability metrics on unified-memory systems.
            render_durations_ms: Vec::with_capacity(timing_capacity),
            frame_intervals_ms: Vec::with_capacity(timing_capacity),
            peak_fps: 0.0,
            resolution,
        }
    }

    fn set_resolution(&mut self, resolution: PhysicalSize<u32>) {
        self.resolution = resolution;
    }

    fn reset_measurement(&mut self, now: Instant) {
        self.benchmark_start = now;
        self.last_frame_presented_at = None;
        self.last_title_update = now;
        self.frames = 0;
        self.render_durations_ms.clear();
        self.frame_intervals_ms.clear();
        self.peak_fps = 0.0;
    }

    fn record_frame(
        &mut self,
        frame_start: Instant,
        frame_end: Instant,
        work_units: u32,
    ) -> FrameTiming {
        let work_units = work_units.max(1) as u64;
        self.frames = self.frames.saturating_add(work_units);
        let render_ms = (frame_end - frame_start).as_secs_f64() * 1000.0;
        self.render_durations_ms.push(render_ms / work_units as f64);

        let mut instant_fps = 0.0;
        if let Some(last) = self.last_frame_presented_at {
            let interval_ms = (frame_end - last).as_secs_f64() * 1000.0;
            if interval_ms.is_finite() && interval_ms > 0.0 {
                let normalized_interval_ms = interval_ms / work_units as f64;
                self.frame_intervals_ms.push(normalized_interval_ms);
                instant_fps = 1000.0 / normalized_interval_ms;
                self.peak_fps = self.peak_fps.max(instant_fps);
            }
        }
        self.last_frame_presented_at = Some(frame_end);

        let elapsed = (frame_end - self.benchmark_start).as_secs_f64().max(1e-9);
        let avg_fps = self.frames as f64 / elapsed;
        let display_fps = self.smoothed_fps_recent(30).unwrap_or(instant_fps);

        let should_update_title =
            frame_end.duration_since(self.last_title_update) >= self.title_update_interval;
        if should_update_title {
            self.last_title_update = frame_end;
        }

        FrameTiming {
            display_fps,
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
        let avg_work_ms = average(&self.render_durations_ms).unwrap_or(0.0);
        let p95_work_ms = percentile_ms(&self.render_durations_ms, 95.0).unwrap_or(avg_work_ms);
        let p99_work_ms = percentile_ms(&self.render_durations_ms, 99.0).unwrap_or(p95_work_ms);
        let work_time_stddev_ms = stddev(&self.render_durations_ms).unwrap_or(0.0);

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
            avg_work_ms,
            p95_work_ms,
            p99_work_ms,
            work_time_stddev_ms,
            resolution: self.resolution,
        }
    }

    fn smoothed_fps_recent(&self, sample_count: usize) -> Option<f64> {
        if sample_count == 0 || self.frame_intervals_ms.is_empty() {
            return None;
        }

        let len = self.frame_intervals_ms.len();
        let start = len.saturating_sub(sample_count);
        let slice = &self.frame_intervals_ms[start..];
        let avg_interval_ms = slice.iter().sum::<f64>() / slice.len() as f64;

        if avg_interval_ms > 0.0 {
            Some(1000.0 / avg_interval_ms)
        } else {
            None
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
    avg_work_ms: f64,
    p95_work_ms: f64,
    p99_work_ms: f64,
    work_time_stddev_ms: f64,
    resolution: PhysicalSize<u32>,
}

#[derive(Debug, Clone, Copy)]
struct RenderScore {
    score: u64,
    tier: &'static str,
    fps_term: f64,
    stability_factor: f64,
    tail_factor: f64,
    present_stability_factor: f64,
    present_tail_factor: f64,
    work_stability_factor: f64,
    work_tail_factor: f64,
    work_stability_blend: f64,
    resolution_factor: f64,
    gms_factor: f64,
}

fn compute_render_benchmark_score(
    stats: &ComputedFrameStats,
    gms_hardware_score: u64,
    pacing_mode: BenchmarkPacingMode,
    runtime_tuning: &GmsRuntimeTuningProfile,
) -> RenderScore {
    let pixels = (stats.resolution.width.max(1) as f64) * (stats.resolution.height.max(1) as f64);
    let resolution_factor = (pixels / (1280.0 * 720.0)).sqrt().clamp(0.75, 3.0);

    let avg_ms = stats.avg_frame_ms.max(0.001);
    let p95_ms = stats.p95_frame_ms.max(avg_ms);
    let p99_ms = stats.p99_frame_ms.max(p95_ms);
    let avg_work_ms = stats.avg_work_ms.max(0.001);
    let p95_work_ms = stats.p95_work_ms.max(avg_work_ms);
    let p99_work_ms = stats.p99_work_ms.max(p95_work_ms);

    let present_stability_factor = (avg_ms / p95_ms).clamp(0.35, 1.0);
    let present_tail_factor = (avg_ms / p99_ms).clamp(0.25, 1.0);
    let work_stability_factor = (avg_work_ms / p95_work_ms).clamp(0.35, 1.0);
    let work_tail_factor = (avg_work_ms / p99_work_ms).clamp(0.25, 1.0);

    // In max-throughput mode on unified-memory adapters (especially Apple Silicon), present-time
    // intervals include compositor scheduling noise that does not reflect actual GPU work pacing.
    // Blend in per-work-unit render durations so the score tracks engine-side throughput stability
    // while still retaining some penalty for visible present jitter.
    let work_stability_blend = if matches!(pacing_mode, BenchmarkPacingMode::MaxThroughput) {
        runtime_tuning
            .throughput_work_stability_blend
            .clamp(0.0, 1.0)
    } else {
        0.0
    };

    let stability_factor = lerp(
        present_stability_factor,
        work_stability_factor,
        work_stability_blend,
    );
    let tail_factor = lerp(present_tail_factor, work_tail_factor, work_stability_blend);

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
        present_stability_factor,
        present_tail_factor,
        work_stability_factor,
        work_tail_factor,
        work_stability_blend,
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
    estimated_compute_units: Option<u32>,
    compute_unit_short_label: Option<&'static str>,
    compute_unit_display_label: Option<&'static str>,
    compute_unit_source: Option<&'static str>,
    compute_unit_probe_note: Option<String>,
    arm_shader_core_count: Option<u32>,
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
    avg_work_ms: f64,
    p95_work_ms: f64,
    p99_work_ms: f64,
    work_time_stddev_ms: f64,
    gms_hardware_score: u64,
    total_discovered_gpus: usize,
    render_score: u64,
    render_tier: &'static str,
    score_breakdown: RenderScore,
    work_units_per_present: u32,
    primary_passes_per_work_unit: u32,
    multi_gpu: Option<MultiGpuExecutorSummary>,
    pacing_mode: BenchmarkPacingMode,
    warmup_duration: Duration,
    sample_duration: Duration,
    mode_override: ModeOverride,
    vsync_override: VsyncOverride,
    multi_gpu_override: MultiGpuOverride,
}

fn print_summary(summary: &BenchmarkSummary) {
    println!();
    println!("=== GMS Render Benchmark Summary ===");
    println!(
        "Adapter: {} | Backend: {} | Type: {}",
        summary.adapter_name, summary.backend, summary.device_type
    );
    if let (Some(units), Some(short_label), Some(display_label), Some(source)) = (
        summary.estimated_compute_units,
        summary.compute_unit_short_label,
        summary.compute_unit_display_label,
        summary.compute_unit_source,
    ) {
        println!(
            "Estimated {}: {} {} (source: {})",
            display_label, units, short_label, source
        );
        if let Some(arm_shader_cores) = summary.arm_shader_core_count {
            println!("ARM shader cores (aux, not cluster count): {arm_shader_cores}");
        }
        if let Some(note) = summary.compute_unit_probe_note.as_deref() {
            println!("Compute-unit probe note: {note}");
        }
    }
    println!(
        "Resolution: {}x{} | Present mode: {:?}",
        summary.resolution.width, summary.resolution.height, summary.present_mode
    );
    if summary.work_units_per_present > 1 {
        println!(
            "Throughput burst: x{} work units / present | primary passes/WU: {} (WFPS may exceed display refresh)",
            summary.work_units_per_present,
            summary.primary_passes_per_work_unit
        );
    }
    println!(
        "Pacing mode: {:?} | CLI mode: {:?} | CLI vsync: {:?} | CLI multi-gpu: {:?}",
        summary.pacing_mode,
        summary.mode_override,
        summary.vsync_override,
        summary.multi_gpu_override
    );
    println!(
        "Warmup: {:.2}s | Sample: {:.2}s",
        summary.warmup_duration.as_secs_f64(),
        summary.sample_duration.as_secs_f64()
    );
    println!(
        "Frames (work units): {} | Elapsed: {:.3}s | Avg FPS/WFPS: {:.2} | Peak FPS/WFPS: {:.2}",
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
        "Work time / unit(ms): avg {:.3} | p95 {:.3} | p99 {:.3} | stddev {:.3}",
        summary.avg_work_ms, summary.p95_work_ms, summary.p99_work_ms, summary.work_time_stddev_ms
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
    if summary.score_breakdown.work_stability_blend > 0.0 {
        println!(
            "Stability blend => present {:.3}/{:.3}, work {:.3}/{:.3}, work_weight {:.2}",
            summary.score_breakdown.present_stability_factor,
            summary.score_breakdown.present_tail_factor,
            summary.score_breakdown.work_stability_factor,
            summary.score_breakdown.work_tail_factor,
            summary.score_breakdown.work_stability_blend
        );
    }
    if let Some(multi_gpu) = summary.multi_gpu.as_ref() {
        println!(
            "Multi-GPU: {} -> {} ({:?}) | secondary WU/present: {} | passes/WU: {} | total secondary WU: {}",
            multi_gpu.primary_adapter_name,
            multi_gpu.secondary_adapter_name,
            multi_gpu.secondary_memory_topology,
            multi_gpu.secondary_work_units_per_present,
            multi_gpu.secondary_passes_per_work_unit,
            multi_gpu.total_secondary_work_units
        );
        println!(
            "Multi-GPU submissions: {}",
            multi_gpu.total_secondary_submissions
        );
        println!(
            "Multi-GPU planner: single {:.3}ms -> multi {:.3}ms | projected gain: {:.2}% | target>=20%: {}",
            multi_gpu.estimated_single_gpu_frame_ms,
            multi_gpu.estimated_multi_gpu_frame_ms,
            multi_gpu.projected_score_gain_pct,
            multi_gpu.meets_target_gain_20pct
        );
        if multi_gpu.vulkan_version_gate_enabled {
            println!(
                "Vulkan version gate: enabled | primary: {} | secondary: {}",
                multi_gpu
                    .primary_vulkan_api_version
                    .as_deref()
                    .unwrap_or("unknown"),
                multi_gpu
                    .secondary_vulkan_api_version
                    .as_deref()
                    .unwrap_or("unknown")
            );
        }
        println!(
            "Bridge: {:?} | bytes/frame: {} | chunk: {} | sync frames_in_flight: {} | queue waits/polls/timeouts/skips: {}/{}/{}/{}",
            multi_gpu.bridge_kind,
            multi_gpu.bridge_bytes_per_frame,
            multi_gpu.bridge_chunk_bytes,
            multi_gpu.sync_frames_in_flight,
            multi_gpu.sync_queue_waits,
            multi_gpu.sync_queue_polls,
            multi_gpu.sync_queue_wait_timeouts,
            multi_gpu.sync_skipped_submissions
        );
        if multi_gpu.aggressive_integrated_preallocation {
            println!(
                "iGPU stability preallocation: enabled | encoder pool: {} | ring segments: {}",
                multi_gpu.integrated_encoder_pool, multi_gpu.integrated_ring_segments
            );
        }
    }
}

fn average(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    Some(values.iter().sum::<f64>() / values.len() as f64)
}

fn lerp(a: f64, b: f64, t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    a + (b - a) * t
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

#[derive(Debug, Clone, Copy)]
enum BenchmarkPacingMode {
    Stable,
    MaxThroughput,
}

fn choose_pacing_mode(
    profile: Option<&crate::GpuAdapterProfile>,
    adapter_info: &wgpu::AdapterInfo,
    options: CliOptions,
) -> BenchmarkPacingMode {
    match options.mode_override {
        ModeOverride::Stable => return BenchmarkPacingMode::Stable,
        ModeOverride::Max => return BenchmarkPacingMode::MaxThroughput,
        ModeOverride::Auto => {}
    }

    if let Some(profile) = profile {
        if matches!(profile.memory_topology, crate::MemoryTopology::Unified) {
            return BenchmarkPacingMode::Stable;
        }
    }

    if matches!(adapter_info.device_type, wgpu::DeviceType::IntegratedGpu) {
        BenchmarkPacingMode::Stable
    } else {
        BenchmarkPacingMode::MaxThroughput
    }
}

fn compact_compute_unit_title_suffix(profile: &crate::GpuAdapterProfile) -> String {
    if profile.estimated_compute_units == 0 {
        return String::new();
    }

    let (label, count) = profile.compute_unit_summary();
    format!(" | {} {}", label, count)
}

#[derive(Debug, Clone)]
struct SimpleError(String);

impl fmt::Display for SimpleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for SimpleError {}
