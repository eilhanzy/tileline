use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use nalgebra::Vector3;
use paradoxpe::{PhysicsWorld, PhysicsWorldConfig};
use runtime::{
    compile_tlscript_showcase, BounceTankSceneConfig, BounceTankSceneController, DrawPathCompiler,
    RenderSyncMode, TelemetryHudComposer, TelemetryHudSample, TickRatePolicy,
    TlscriptShowcaseConfig, TlscriptShowcaseFrameInput, TlscriptShowcaseProgram,
    TlspriteHotReloadEvent, TlspriteProgram, TlspriteProgramCache, TlspriteWatchReloader,
    WgpuSceneRenderer,
};
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalSize};
use winit::event::{DeviceEvent, ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, KeyCode, NamedKey, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowAttributes, WindowId};

pub fn run_from_env() -> Result<(), Box<dyn Error>> {
    let options = CliOptions::parse_from_env()?;
    let event_loop = EventLoop::new()?;
    let mut app = TlApp::new(options);
    event_loop.run_app(&mut app)?;
    Ok(())
}

#[derive(Debug, Clone)]
struct CliOptions {
    resolution: PhysicalSize<u32>,
    vsync: VsyncMode,
    fps_cap: Option<f32>,
    fps_report_interval: Duration,
    script_path: PathBuf,
    sprite_path: PathBuf,
}

impl Default for CliOptions {
    fn default() -> Self {
        Self {
            resolution: PhysicalSize::new(1280, 720),
            vsync: VsyncMode::Auto,
            fps_cap: Some(60.0),
            fps_report_interval: Duration::from_secs_f32(1.0),
            script_path: PathBuf::from("docs/demos/tlapp/bounce_showcase.tlscript"),
            sprite_path: PathBuf::from("docs/demos/tlapp/bounce_hud.tlsprite"),
        }
    }
}

impl CliOptions {
    fn parse_from_env() -> Result<Self, Box<dyn Error>> {
        let mut options = Self::default();
        let mut args = env::args().skip(1);

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "-h" | "--help" => {
                    print_usage();
                    std::process::exit(0);
                }
                "--resolution" => {
                    let value = next_arg(&mut args, "--resolution")?;
                    options.resolution = parse_resolution(&value)?;
                }
                "--vsync" => {
                    let value = next_arg(&mut args, "--vsync")?;
                    options.vsync = VsyncMode::parse(&value)?;
                }
                "--fps-cap" => {
                    let value = next_arg(&mut args, "--fps-cap")?;
                    options.fps_cap = parse_fps_cap(&value)?;
                }
                "--fps-report" => {
                    let value = next_arg(&mut args, "--fps-report")?;
                    options.fps_report_interval = parse_seconds_arg(&value, "--fps-report")?;
                }
                "--script" => {
                    let value = next_arg(&mut args, "--script")?;
                    options.script_path = PathBuf::from(value);
                }
                "--sprite" => {
                    let value = next_arg(&mut args, "--sprite")?;
                    options.sprite_path = PathBuf::from(value);
                }
                other => {
                    return Err(format!("unknown argument: {other} (use --help)").into());
                }
            }
        }

        if let Some(fps_cap) = options.fps_cap {
            if !(fps_cap.is_finite() && fps_cap > 1.0) {
                return Err("--fps-cap must be > 1.0 or 'off'".into());
            }
        }
        if options.fps_report_interval.is_zero() {
            return Err("--fps-report must be > 0".into());
        }

        Ok(options)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VsyncMode {
    Auto,
    On,
    Off,
}

impl VsyncMode {
    fn parse(value: &str) -> Result<Self, Box<dyn Error>> {
        match value.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "on" | "1" | "true" => Ok(Self::On),
            "off" | "0" | "false" => Ok(Self::Off),
            _ => Err(format!("invalid --vsync value: {value} (expected auto|on|off)").into()),
        }
    }
}

fn next_arg(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, Box<dyn Error>> {
    args.next()
        .ok_or_else(|| format!("missing value for {flag}").into())
}

fn parse_seconds_arg(value: &str, flag: &str) -> Result<Duration, Box<dyn Error>> {
    let seconds = value
        .parse::<f64>()
        .map_err(|_| format!("invalid {flag}: {value} (expected seconds)"))?;
    if !seconds.is_finite() || seconds <= 0.0 {
        return Err(format!("invalid {flag}: {value} (must be > 0)").into());
    }
    Ok(Duration::from_secs_f64(seconds))
}

fn parse_resolution(value: &str) -> Result<PhysicalSize<u32>, Box<dyn Error>> {
    let lower = value.to_ascii_lowercase();
    let (w, h) = lower
        .split_once('x')
        .ok_or_else(|| format!("invalid --resolution: {value} (expected WxH)"))?;
    let width = w
        .parse::<u32>()
        .map_err(|_| format!("invalid width in --resolution: {value}"))?;
    let height = h
        .parse::<u32>()
        .map_err(|_| format!("invalid height in --resolution: {value}"))?;
    if width == 0 || height == 0 {
        return Err("--resolution values must be non-zero".into());
    }
    Ok(PhysicalSize::new(width, height))
}

fn parse_fps_cap(value: &str) -> Result<Option<f32>, Box<dyn Error>> {
    if value.eq_ignore_ascii_case("off") {
        return Ok(None);
    }
    let fps = value
        .parse::<f32>()
        .map_err(|_| format!("invalid --fps-cap value: {value} (expected number or off)"))?;
    if !fps.is_finite() || fps <= 1.0 {
        return Err("--fps-cap must be > 1.0".into());
    }
    Ok(Some(fps))
}

fn print_usage() {
    println!("Tileline TLApp Runtime Demo");
    println!("Usage: cargo run -p runtime --example tlapp -- [options]");
    println!("Options:");
    println!("  --resolution <WxH>        Window size (default: 1280x720)");
    println!("  --vsync auto|on|off       Present mode preference (default: auto)");
    println!("  --fps-cap <N|off>         Frame cap target (default: 60)");
    println!("  --fps-report <sec>        CLI FPS report cadence (default: 1.0)");
    println!(
        "  --script <path>           .tlscript path (default: docs/demos/tlapp/bounce_showcase.tlscript)"
    );
    println!(
        "  --sprite <path>           .tlsprite path (default: docs/demos/tlapp/bounce_hud.tlsprite)"
    );
    println!("  -h, --help                Show help");
}

struct TlApp {
    options: CliOptions,
    runtime: Option<TlAppRuntime>,
    exit_requested: bool,
}

impl TlApp {
    fn new(options: CliOptions) -> Self {
        Self {
            options,
            runtime: None,
            exit_requested: false,
        }
    }
}

impl ApplicationHandler for TlApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.runtime.is_some() {
            return;
        }

        match TlAppRuntime::new(event_loop, self.options.clone()) {
            Ok(runtime) => {
                self.runtime = Some(runtime);
            }
            Err(err) => {
                eprintln!("Failed to start TLApp runtime: {err}");
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
            WindowEvent::KeyboardInput { event, .. } => {
                runtime.on_keyboard_input(&event);
                if matches!(event.logical_key, Key::Named(NamedKey::Escape))
                    && event.state == ElementState::Pressed
                {
                    self.exit_requested = true;
                    event_loop.exit();
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                runtime.on_mouse_button(state, button);
            }
            WindowEvent::Resized(size) => runtime.resize(size),
            WindowEvent::RedrawRequested => {
                if let Err(err) = runtime.render_frame() {
                    eprintln!("Render error: {err}");
                    self.exit_requested = true;
                    event_loop.exit();
                }
            }
            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        if let Some(runtime) = self.runtime.as_mut() {
            runtime.on_device_event(event);
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self.exit_requested {
            event_loop.exit();
            return;
        }

        if let Some(runtime) = self.runtime.as_mut() {
            runtime.schedule_next_redraw(event_loop);
        }
    }
}

struct TlAppRuntime {
    window: Arc<Window>,
    _instance: wgpu::Instance,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: PhysicalSize<u32>,
    world: PhysicsWorld,
    scene: BounceTankSceneController,
    draw_compiler: DrawPathCompiler,
    hud: TelemetryHudComposer,
    renderer: WgpuSceneRenderer,
    camera: FreeCameraController,
    sprite_loader: TlspriteWatchReloader,
    sprite_cache: TlspriteProgramCache,
    force_full_fbx_from_sprite: bool,
    script_program: TlscriptShowcaseProgram<'static>,
    script_last_spawned: usize,
    script_frame_index: u64,
    tick_policy: TickRatePolicy,
    tick_hz: f32,
    max_substeps: u32,
    last_substeps: u32,
    tick_retune_timer: f32,
    frame_started_at: Instant,
    frame_cap_interval: Option<Duration>,
    next_redraw_at: Instant,
    fps_tracker: FpsTracker,
}

impl TlAppRuntime {
    fn new(event_loop: &ActiveEventLoop, options: CliOptions) -> Result<Self, Box<dyn Error>> {
        let window = Arc::new(
            event_loop.create_window(
                WindowAttributes::default()
                    .with_title("Tileline TLApp")
                    .with_inner_size(LogicalSize::new(
                        options.resolution.width as f64,
                        options.resolution.height as f64,
                    )),
            )?,
        );
        let size = window.inner_size();

        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(Arc::clone(&window))?;
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: Some(&surface),
        }))
        .map_err(|err| format!("request_adapter failed: {err}"))?;

        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("tlapp-device"),
                ..Default::default()
            }))?;

        let mut config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .ok_or("surface is not compatible with selected adapter")?;
        config.present_mode = match options.vsync {
            VsyncMode::Auto => wgpu::PresentMode::AutoVsync,
            VsyncMode::On => wgpu::PresentMode::Fifo,
            VsyncMode::Off => wgpu::PresentMode::AutoNoVsync,
        };
        surface.configure(&device, &config);

        let script_source_owned = fs::read_to_string(&options.script_path).map_err(|err| {
            format!(
                "failed to read script '{}': {err}",
                options.script_path.display()
            )
        })?;
        let script_source: &'static str = Box::leak(script_source_owned.into_boxed_str());
        let script_compile =
            compile_tlscript_showcase(script_source, TlscriptShowcaseConfig::default());
        for warning in &script_compile.warnings {
            eprintln!("[tlscript warning] {warning}");
        }
        if !script_compile.errors.is_empty() {
            return Err(format!(
                "failed to compile script '{}': {}",
                options.script_path.display(),
                script_compile.errors.join(" | ")
            )
            .into());
        }
        let script_program = script_compile
            .program
            .expect("showcase script should compile without errors");

        let tick_policy = TickRatePolicy {
            min_tick_hz: 120.0,
            max_tick_hz: 960.0,
            ticks_per_render_frame: 4.0,
            default_tick_hz: 240.0,
        };
        let render_mode = match options.fps_cap {
            Some(fps) => RenderSyncMode::FpsCap { fps },
            None => RenderSyncMode::Uncapped,
        };
        let fixed_dt = tick_policy.resolve_fixed_dt_seconds(render_mode, options.fps_cap);
        let max_substeps = 20;
        let world = PhysicsWorld::new(PhysicsWorldConfig {
            fixed_dt,
            max_substeps,
            ..PhysicsWorldConfig::default()
        });
        let scene = BounceTankSceneController::new(BounceTankSceneConfig {
            target_ball_count: 8_000,
            spawn_per_tick: 280,
            ..BounceTankSceneConfig::default()
        });

        let mut renderer =
            WgpuSceneRenderer::new(&device, &queue, config.format, size.width, size.height);
        let camera = FreeCameraController::default();
        let (eye, target) = camera.eye_target();
        renderer.set_camera_view(&queue, size.width.max(1), size.height.max(1), eye, target);

        let draw_compiler = DrawPathCompiler::new();
        let hud = TelemetryHudComposer::new(Default::default());
        let mut sprite_loader = TlspriteWatchReloader::new(&options.sprite_path);
        let mut sprite_cache = TlspriteProgramCache::new();
        if let Some(warn) = sprite_loader.init_warning() {
            eprintln!("[tlsprite watch] {warn}");
        }
        eprintln!("[tlsprite watch] backend={:?}", sprite_loader.backend());
        let event = sprite_loader.reload_into_cache(&mut sprite_cache);
        print_tlsprite_event("[tlsprite boot]", event);

        let mut scene = scene;
        let mut force_full_fbx_from_sprite = false;
        if let Some(program) = sprite_cache.program_for_path(sprite_loader.path()).cloned() {
            force_full_fbx_from_sprite = program.requires_full_fbx_render();
            scene.set_sprite_program(program.clone());
            bind_renderer_meshes_from_tlsprite(
                &mut renderer,
                &device,
                sprite_loader.path(),
                &program,
            );
        }
        renderer.set_force_full_fbx_sphere(force_full_fbx_from_sprite);

        let now = Instant::now();
        let frame_cap_interval = options
            .fps_cap
            .map(|fps| Duration::from_secs_f32(1.0 / fps.max(1.0)));
        let fps_report_interval = options.fps_report_interval;

        Ok(Self {
            window,
            _instance: instance,
            surface,
            device,
            queue,
            config,
            size,
            world,
            scene,
            draw_compiler,
            hud,
            renderer,
            camera,
            sprite_loader,
            sprite_cache,
            force_full_fbx_from_sprite,
            script_program,
            script_last_spawned: 0,
            script_frame_index: 0,
            tick_policy,
            tick_hz: 1.0 / fixed_dt.max(1e-6),
            max_substeps,
            last_substeps: 0,
            tick_retune_timer: 0.0,
            frame_started_at: now,
            frame_cap_interval,
            next_redraw_at: now,
            fps_tracker: FpsTracker::new(fps_report_interval),
        })
    }

    fn schedule_next_redraw(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(interval) = self.frame_cap_interval {
            let now = Instant::now();
            if now < self.next_redraw_at {
                event_loop.set_control_flow(ControlFlow::WaitUntil(self.next_redraw_at));
                return;
            }
            self.next_redraw_at = now + interval;
        }

        event_loop.set_control_flow(ControlFlow::Poll);
        self.window.request_redraw();
    }

    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }
        self.size = new_size;
        self.config.width = new_size.width;
        self.config.height = new_size.height;
        self.surface.configure(&self.device, &self.config);
        self.renderer.resize(
            &self.device,
            &self.queue,
            new_size.width.max(1),
            new_size.height.max(1),
        );
    }

    fn on_keyboard_input(&mut self, event: &KeyEvent) {
        self.camera.on_keyboard_input(event);
    }

    fn on_mouse_button(&mut self, state: ElementState, button: MouseButton) {
        if button == MouseButton::Right {
            let active = state == ElementState::Pressed;
            self.camera.set_look_active(&self.window, active);
        }
    }

    fn on_device_event(&mut self, event: DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta } = event {
            self.camera.on_mouse_delta(delta.0 as f32, delta.1 as f32);
        }
    }

    fn render_frame(&mut self) -> Result<(), Box<dyn Error>> {
        let frame_begin = Instant::now();
        let dt = (frame_begin - self.frame_started_at)
            .as_secs_f32()
            .clamp(1.0 / 500.0, 1.0 / 20.0);
        self.frame_started_at = frame_begin;

        self.camera.update(dt);
        let (eye, target) = self.camera.eye_target();
        self.renderer.set_camera_view(
            &self.queue,
            self.size.width.max(1),
            self.size.height.max(1),
            eye,
            target,
        );

        let event = self.sprite_loader.reload_into_cache(&mut self.sprite_cache);
        match &event {
            TlspriteHotReloadEvent::Applied { .. } => {
                print_tlsprite_event("[tlsprite reload]", event);
                if let Some(program) = self
                    .sprite_cache
                    .program_for_path(self.sprite_loader.path())
                    .cloned()
                {
                    self.force_full_fbx_from_sprite = program.requires_full_fbx_render();
                    self.scene.set_sprite_program(program.clone());
                    bind_renderer_meshes_from_tlsprite(
                        &mut self.renderer,
                        &self.device,
                        self.sprite_loader.path(),
                        &program,
                    );
                }
            }
            TlspriteHotReloadEvent::Unchanged => {}
            _ => print_tlsprite_event("[tlsprite reload]", event),
        }

        let frame_eval = self
            .script_program
            .evaluate_frame(TlscriptShowcaseFrameInput {
                frame_index: self.script_frame_index,
                live_balls: self.scene.live_ball_count(),
                spawned_this_tick: self.script_last_spawned,
            });

        if let Some(speed) = frame_eval.camera_move_speed {
            self.camera.set_move_speed(speed);
        }
        if let Some(sensitivity) = frame_eval.camera_look_sensitivity {
            self.camera.set_mouse_sensitivity(sensitivity);
        }
        if let Some((camera_eye, camera_target)) = frame_eval.camera_pose {
            self.camera.set_pose(camera_eye, camera_target);
        }

        let live_balls = self.scene.live_ball_count();
        let parallel_ready = frame_eval
            .dispatch_decision
            .as_ref()
            .map(|d| d.is_parallel())
            .unwrap_or(false);
        let force_full_fbx = frame_eval
            .force_full_fbx_sphere
            .unwrap_or(self.force_full_fbx_from_sprite);
        self.renderer.set_force_full_fbx_sphere(force_full_fbx);

        let mut runtime_patch = frame_eval.patch;
        if self.last_substeps + 1 >= self.max_substeps && live_balls > 2_000 {
            runtime_patch.spawn_per_tick =
                Some(runtime_patch.spawn_per_tick.unwrap_or(96).clamp(32, 96));
            runtime_patch.linear_damping =
                Some(runtime_patch.linear_damping.unwrap_or(0.016).max(0.018));
        }
        if !parallel_ready && live_balls > 4_000 {
            runtime_patch.spawn_per_tick =
                Some(runtime_patch.spawn_per_tick.unwrap_or(64).clamp(24, 64));
            runtime_patch.linear_damping =
                Some(runtime_patch.linear_damping.unwrap_or(0.018).max(0.020));
        }

        self.tick_retune_timer -= dt;
        if self.tick_retune_timer <= 0.0 {
            let desired_hz = choose_aggressive_tick_hz(
                self.tick_policy,
                self.fps_tracker.ema_fps().max(1.0),
                parallel_ready,
                self.last_substeps,
                self.max_substeps,
                live_balls,
            );
            self.tick_hz = smooth_tick_hz(self.tick_hz, desired_hz, 0.35);
            self.world
                .set_timestep(1.0 / self.tick_hz.max(60.0), self.max_substeps);
            self.tick_retune_timer = 0.2;
        }

        let _patch_metrics = self
            .scene
            .apply_runtime_patch(&mut self.world, runtime_patch);
        let tick = self.scene.physics_tick(&mut self.world);
        self.script_last_spawned = tick.spawned_this_tick;
        self.script_frame_index = self.script_frame_index.saturating_add(1);

        let substeps = self.world.step(dt);
        self.last_substeps = substeps;

        let mut frame = self
            .scene
            .build_frame_instances(&self.world, Some(self.world.interpolation_alpha()));
        let _hud = self.hud.append_to_sprites(
            TelemetryHudSample {
                fps: self.fps_tracker.ema_fps(),
                frame_time_ms: dt * 1_000.0,
                physics_substeps: substeps,
                live_balls: tick.live_balls,
                draw_calls: frame.opaque_3d.len()
                    + frame.transparent_3d.len()
                    + frame.sprites.len(),
            },
            &mut frame.sprites,
        );
        let draw = self.draw_compiler.compile(&frame);
        let _upload = self
            .renderer
            .upload_draw_frame(&self.device, &self.queue, &draw);

        let output = match self.surface.get_current_texture() {
            Ok(frame) => frame,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                self.surface.configure(&self.device, &self.config);
                return Ok(());
            }
            Err(wgpu::SurfaceError::OutOfMemory) => {
                return Err("wgpu surface out of memory".into());
            }
            Err(wgpu::SurfaceError::Timeout) => {
                return Ok(());
            }
            Err(wgpu::SurfaceError::Other) => {
                return Ok(());
            }
        };

        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("tlapp-encoder"),
            });
        self.renderer.encode(
            &mut encoder,
            &view,
            wgpu::Color {
                r: 0.07,
                g: 0.09,
                b: 0.12,
                a: 1.0,
            },
        );
        self.queue.submit(Some(encoder.finish()));
        output.present();

        let frame_end = Instant::now();
        let frame_time = (frame_end - frame_begin).as_secs_f32();
        let report = self.fps_tracker.record(frame_end, frame_time);
        let title = format!(
            "Tileline TLApp | FPS {:.1} | Frame {:.2} ms | Balls {} | Substeps {}{}",
            self.fps_tracker.ema_fps(),
            frame_time * 1_000.0,
            tick.live_balls,
            substeps,
            self.frame_cap_interval
                .map(|d| format!(" | cap {:.0}", 1.0 / d.as_secs_f32().max(1e-6)))
                .unwrap_or_default()
        );
        self.window.set_title(&title);

        if let Some(report) = report {
            println!(
                "tlapp fps | inst: {:>6.1} | ema: {:>6.1} | avg: {:>6.1} | stddev: {:>5.2} ms | balls: {:>5} | substeps: {}",
                report.instant_fps,
                report.ema_fps,
                report.avg_fps,
                report.frame_time_stddev_ms,
                tick.live_balls,
                substeps
            );
        }

        Ok(())
    }
}

fn bind_renderer_meshes_from_tlsprite(
    renderer: &mut WgpuSceneRenderer,
    device: &wgpu::Device,
    sprite_path: &Path,
    program: &TlspriteProgram,
) {
    for (slot, raw_path) in program.mesh_fbx_bindings() {
        let resolved = resolve_asset_path(sprite_path, raw_path);
        match renderer.bind_fbx_mesh_slot_from_path(device, slot, resolved.as_path()) {
            Ok(()) => {
                println!(
                    "[tlapp fbx] bound slot {} <- {}",
                    slot,
                    resolved.to_string_lossy()
                );
            }
            Err(err) => {
                eprintln!(
                    "[tlapp fbx] failed slot {} for '{}': {}",
                    slot,
                    resolved.to_string_lossy(),
                    err
                );
            }
        }
    }
}

fn resolve_asset_path(base_file: &Path, raw_path: &str) -> PathBuf {
    let raw = PathBuf::from(raw_path);
    if raw.is_absolute() {
        return raw;
    }

    if let Some(parent) = base_file.parent() {
        let candidate = parent.join(&raw);
        if candidate.exists() {
            return candidate;
        }
    }

    raw
}

#[derive(Debug, Clone, Copy, Default)]
struct CameraInputState {
    forward: bool,
    backward: bool,
    left: bool,
    right: bool,
    up: bool,
    down: bool,
    sprint: bool,
}

#[derive(Debug, Clone)]
struct FreeCameraController {
    position: Vector3<f32>,
    yaw_rad: f32,
    pitch_rad: f32,
    move_speed: f32,
    sprint_multiplier: f32,
    mouse_sensitivity: f32,
    look_active: bool,
    input: CameraInputState,
}

impl Default for FreeCameraController {
    fn default() -> Self {
        Self {
            position: Vector3::new(0.0, 12.0, 36.0),
            yaw_rad: 0.0,
            pitch_rad: -0.321_750_55,
            move_speed: 18.0,
            sprint_multiplier: 2.5,
            mouse_sensitivity: 0.0018,
            look_active: false,
            input: CameraInputState::default(),
        }
    }
}

impl FreeCameraController {
    fn on_keyboard_input(&mut self, event: &KeyEvent) {
        let pressed = event.state == ElementState::Pressed;
        match event.physical_key {
            PhysicalKey::Code(KeyCode::KeyW) => self.input.forward = pressed,
            PhysicalKey::Code(KeyCode::KeyS) => self.input.backward = pressed,
            PhysicalKey::Code(KeyCode::KeyA) => self.input.left = pressed,
            PhysicalKey::Code(KeyCode::KeyD) => self.input.right = pressed,
            PhysicalKey::Code(KeyCode::Space) => self.input.up = pressed,
            PhysicalKey::Code(KeyCode::ShiftLeft) | PhysicalKey::Code(KeyCode::ShiftRight) => {
                self.input.down = pressed
            }
            PhysicalKey::Code(KeyCode::ControlLeft) | PhysicalKey::Code(KeyCode::ControlRight) => {
                self.input.sprint = pressed
            }
            _ => {}
        }
    }

    fn set_look_active(&mut self, window: &Window, active: bool) {
        if self.look_active == active {
            return;
        }
        self.look_active = active;

        if active {
            let grab_result = window
                .set_cursor_grab(CursorGrabMode::Locked)
                .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined));
            if let Err(err) = grab_result {
                eprintln!("[camera] cursor grab failed: {err}");
            }
            window.set_cursor_visible(false);
        } else {
            let _ = window.set_cursor_grab(CursorGrabMode::None);
            window.set_cursor_visible(true);
        }
    }

    fn on_mouse_delta(&mut self, dx: f32, dy: f32) {
        if !self.look_active {
            return;
        }
        self.yaw_rad += dx * self.mouse_sensitivity;
        self.pitch_rad -= dy * self.mouse_sensitivity;
        self.pitch_rad = self.pitch_rad.clamp(-1.553_343, 1.553_343);
    }

    fn set_move_speed(&mut self, speed: f32) {
        self.move_speed = speed.clamp(1.0, 200.0);
    }

    fn set_mouse_sensitivity(&mut self, sensitivity: f32) {
        self.mouse_sensitivity = sensitivity.clamp(0.0001, 0.02);
    }

    fn set_pose(&mut self, eye: [f32; 3], target: [f32; 3]) {
        self.position = Vector3::new(eye[0], eye[1], eye[2]);
        let dir = Vector3::new(target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]);
        let len = dir.norm();
        if len <= 1e-5 {
            return;
        }
        let d = dir / len;
        self.pitch_rad = d.y.asin().clamp(-1.553_343, 1.553_343);
        self.yaw_rad = d.x.atan2(-d.z);
    }

    fn update(&mut self, dt: f32) {
        let forward = self.forward_vector();
        let horizontal_forward = Vector3::new(forward.x, 0.0, forward.z);
        let forward_len = horizontal_forward.norm();
        let forward_flat = if forward_len > 1e-5 {
            horizontal_forward / forward_len
        } else {
            Vector3::new(0.0, 0.0, -1.0)
        };
        let right = Vector3::new(-forward_flat.z, 0.0, forward_flat.x);

        let mut move_dir = Vector3::zeros();
        if self.input.forward {
            move_dir += forward_flat;
        }
        if self.input.backward {
            move_dir -= forward_flat;
        }
        if self.input.right {
            move_dir += right;
        }
        if self.input.left {
            move_dir -= right;
        }
        if self.input.up {
            move_dir.y += 1.0;
        }
        if self.input.down {
            move_dir.y -= 1.0;
        }

        let len = move_dir.norm();
        if len > 1e-5 {
            let move_dir = move_dir / len;
            let speed = self.move_speed
                * if self.input.sprint {
                    self.sprint_multiplier
                } else {
                    1.0
                };
            self.position += move_dir * speed * dt.max(0.0);
        }
    }

    fn eye_target(&self) -> ([f32; 3], [f32; 3]) {
        let forward = self.forward_vector();
        let eye = [self.position.x, self.position.y, self.position.z];
        let target = [
            self.position.x + forward.x,
            self.position.y + forward.y,
            self.position.z + forward.z,
        ];
        (eye, target)
    }

    fn forward_vector(&self) -> Vector3<f32> {
        let (sin_yaw, cos_yaw) = self.yaw_rad.sin_cos();
        let (sin_pitch, cos_pitch) = self.pitch_rad.sin_cos();
        Vector3::new(sin_yaw * cos_pitch, sin_pitch, -cos_yaw * cos_pitch)
    }
}

#[derive(Debug, Clone, Copy)]
struct FpsReport {
    instant_fps: f32,
    ema_fps: f32,
    avg_fps: f32,
    frame_time_stddev_ms: f32,
}

#[derive(Debug, Clone)]
struct FpsTracker {
    frame_times: [f32; 120],
    cursor: usize,
    count: usize,
    ema_fps: f32,
    last_report_at: Instant,
    report_interval: Duration,
}

impl FpsTracker {
    fn new(report_interval: Duration) -> Self {
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

    fn ema_fps(&self) -> f32 {
        self.ema_fps
    }

    fn record(&mut self, now: Instant, frame_time: f32) -> Option<FpsReport> {
        let frame_time = frame_time.clamp(1.0 / 500.0, 0.25);
        self.frame_times[self.cursor] = frame_time;
        self.cursor = (self.cursor + 1) % self.frame_times.len();
        self.count = self.count.saturating_add(1).min(self.frame_times.len());

        let instant_fps = 1.0 / frame_time.max(1e-6);
        let alpha = 0.12;
        self.ema_fps += (instant_fps - self.ema_fps) * alpha;

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

fn choose_aggressive_tick_hz(
    policy: TickRatePolicy,
    fps_estimate: f32,
    parallel_ready: bool,
    last_substeps: u32,
    max_substeps: u32,
    live_balls: usize,
) -> f32 {
    let fps = fps_estimate.clamp(20.0, 240.0);
    let mut multiplier = if parallel_ready { 7.0 } else { 5.0 };
    if live_balls > 5_000 {
        multiplier += 1.0;
    }
    if live_balls > 9_000 {
        multiplier += 1.0;
    }

    let mut target_hz = fps * multiplier;
    if last_substeps + 1 >= max_substeps {
        target_hz *= 0.70;
    } else if last_substeps >= max_substeps.saturating_sub(3) {
        target_hz *= 0.85;
    }
    if fps < 45.0 {
        target_hz *= 0.90;
    }

    target_hz.clamp(
        policy.min_tick_hz.max(120.0),
        policy.max_tick_hz.max(policy.min_tick_hz),
    )
}

fn smooth_tick_hz(current_hz: f32, target_hz: f32, alpha: f32) -> f32 {
    let alpha = alpha.clamp(0.0, 1.0);
    current_hz + (target_hz - current_hz) * alpha
}

fn print_tlsprite_event(prefix: &str, event: TlspriteHotReloadEvent) {
    match event {
        TlspriteHotReloadEvent::Unchanged => {}
        TlspriteHotReloadEvent::Applied {
            sprite_count,
            warning_count,
        } => eprintln!("{prefix} applied sprites={sprite_count} warnings={warning_count}"),
        TlspriteHotReloadEvent::Rejected {
            error_count,
            warning_count,
            kept_last_program,
        } => eprintln!(
            "{prefix} rejected errors={error_count} warnings={warning_count} kept_last_program={kept_last_program}"
        ),
        TlspriteHotReloadEvent::SourceError { message } => {
            eprintln!("{prefix} source_error: {message}")
        }
    }
}
