use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::Once;
use std::time::{Duration, Instant};

use crate::{
    compile_tlscript_showcase, BounceTankSceneConfig, BounceTankSceneController, DrawPathCompiler,
    RenderSyncMode, TelemetryHudComposer, TelemetryHudSample, TickRatePolicy,
    TlscriptShowcaseConfig, TlscriptShowcaseControlInput, TlscriptShowcaseFrameInput,
    TlscriptShowcaseProgram, TlspriteHotReloadEvent, TlspriteProgram, TlspriteProgramCache,
    TlspriteWatchReloader, WgpuSceneRenderer,
};
use nalgebra::Vector3;
use paradoxpe::{
    BroadphaseConfig, ContactSolverConfig, NarrowphaseConfig, PhysicsWorld, PhysicsWorldConfig,
};
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalSize};
use winit::event::{DeviceEvent, ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, KeyCode, ModifiersState, NamedKey, PhysicalKey};
use winit::window::{CursorGrabMode, Fullscreen, Window, WindowAttributes, WindowId};

#[cfg(feature = "gamepad")]
use gilrs::{Axis, Button, EventType, GamepadId, Gilrs};

const AUTO_LOW_POLY_BALL_SLOT: u8 = 250;
const DEFAULT_FBX_BALL_SLOT: u8 = 2;
#[cfg(feature = "gamepad")]
const GAMEPAD_DEADZONE: f32 = 0.16;
const GAMEPAD_LOOK_SPEED_RAD: f32 = 2.6;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RuntimeCommand {
    None,
    Exit,
}

pub fn run_from_env() -> Result<(), Box<dyn Error>> {
    configure_parallel_runtime();
    if cfg!(debug_assertions) {
        eprintln!(
            "[perf warning] TLApp is running in debug mode. Use `--release` for meaningful FPS/core utilization."
        );
    }
    let options = CliOptions::parse_from_env()?;
    let event_loop = EventLoop::new()?;
    let mut app = TlApp::new(options);
    event_loop.run_app(&mut app)?;
    Ok(())
}

fn configure_parallel_runtime() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .max(1);
        match rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|idx| format!("tileline-mps-{idx}"))
            .build_global()
        {
            Ok(()) => eprintln!("[mps] rayon global thread pool configured: {threads} threads"),
            Err(err) => eprintln!("[mps] rayon global thread pool unchanged: {err}"),
        }
    });
}

#[derive(Debug, Clone)]
struct CliOptions {
    resolution: PhysicalSize<u32>,
    vsync: VsyncMode,
    fps_cap: Option<f32>,
    tick_profile: TickProfile,
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
            tick_profile: TickProfile::Max,
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
                "--tick-profile" => {
                    let value = next_arg(&mut args, "--tick-profile")?;
                    options.tick_profile = TickProfile::parse(&value)?;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TickProfile {
    Balanced,
    Max,
}

impl TickProfile {
    fn parse(value: &str) -> Result<Self, Box<dyn Error>> {
        match value.to_ascii_lowercase().as_str() {
            "balanced" => Ok(Self::Balanced),
            "max" | "aggressive" => Ok(Self::Max),
            _ => {
                Err(format!("invalid --tick-profile value: {value} (expected balanced|max)").into())
            }
        }
    }
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

fn bootstrap_uncapped_fps_hint(logical_threads: usize) -> f32 {
    // Start from a conservative-but-fast target and let runtime sampling retune to hardware limit.
    (170.0 + logical_threads as f32 * 7.5).clamp(170.0, 420.0)
}

fn print_usage() {
    println!("Tileline TLApp Runtime Demo");
    println!("Usage: cargo run -p runtime --bin tlapp -- [options]");
    println!("Options:");
    println!("  --resolution <WxH>        Window size (default: 1280x720)");
    println!("  --vsync auto|on|off       Present mode preference (default: auto)");
    println!("  --fps-cap <N|off>         Frame cap target (default: 60)");
    println!("  --tick-profile <mode>     Physics tick planner: balanced|max (default: max)");
    println!("  --fps-report <sec>        CLI FPS report cadence (default: 1.0)");
    println!(
        "  --script <path>           .tlscript path (default: docs/demos/tlapp/bounce_showcase.tlscript)"
    );
    println!(
        "  --sprite <path>           .tlsprite path (default: docs/demos/tlapp/bounce_hud.tlsprite)"
    );
    println!("  -h, --help                Show help");
    println!();
    println!("Keyboard:");
    println!("  Move: WASD / Arrows | Up: Space/E | Down: Ctrl/Q | Sprint: Shift");
    println!("  Look: RMB hold + mouse");
    println!("  Combos: Ctrl+Q exit | Ctrl+F fullscreen | Alt+Enter fullscreen");
    println!("          Ctrl+R reset camera | Ctrl+L toggle look lock");
    println!("Gamepad:");
    println!("  Left stick move | Right stick look | D-Pad move");
    println!("  South button mirrors F action | Trigger buttons add vertical move");
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
                if matches!(runtime.on_keyboard_input(&event), RuntimeCommand::Exit) {
                    self.exit_requested = true;
                    event_loop.exit();
                    return;
                }
                if matches!(event.logical_key, Key::Named(NamedKey::Escape))
                    && event.state == ElementState::Pressed
                {
                    self.exit_requested = true;
                    event_loop.exit();
                }
            }
            WindowEvent::ModifiersChanged(modifiers) => {
                runtime.on_modifiers_changed(modifiers.state());
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
    script_key_f_keyboard: bool,
    keyboard_camera: CameraInputState,
    mouse_look_held: bool,
    look_lock_active: bool,
    mouse_look_delta: (f32, f32),
    camera_reset_requested: bool,
    keyboard_modifiers: ModifiersState,
    gamepad: GamepadManager,
    tick_policy: TickRatePolicy,
    tick_profile: TickProfile,
    tick_hz: f32,
    fps_limit_hint: f32,
    uncapped_dynamic_fps_hint: bool,
    mps_logical_threads: usize,
    max_substeps: u32,
    last_substeps: u32,
    tick_retune_timer: f32,
    adaptive_ball_render_limit: Option<usize>,
    adaptive_live_ball_budget: Option<usize>,
    adaptive_low_poly_override: bool,
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

        let logical_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .max(1);
        let physical_threads = num_cpus::get_physical().max(1);
        let uncapped_dynamic_fps_hint =
            options.fps_cap.is_none() && matches!(options.vsync, VsyncMode::Off);
        let display_refresh_hint_hz = window
            .current_monitor()
            .and_then(|monitor| monitor.refresh_rate_millihertz())
            .map(|mhz| (mhz as f32 / 1_000.0).max(24.0));
        let initial_fps_hint = match (options.fps_cap, options.vsync) {
            (Some(fps), _) => fps.max(24.0),
            (None, VsyncMode::Off) => bootstrap_uncapped_fps_hint(logical_threads),
            (None, _) => display_refresh_hint_hz.unwrap_or(60.0),
        };
        let tick_policy = TickRatePolicy {
            min_tick_hz: 30.0,
            max_tick_hz: (480.0 + logical_threads as f32 * 40.0).clamp(480.0, 1_440.0),
            ticks_per_render_frame: 2.6,
            default_tick_hz: (initial_fps_hint * 2.4).clamp(180.0, 420.0),
        };
        let render_mode = match options.fps_cap {
            Some(fps) => RenderSyncMode::FpsCap { fps },
            None => {
                if uncapped_dynamic_fps_hint {
                    RenderSyncMode::Uncapped
                } else {
                    RenderSyncMode::Vsync {
                        display_hz: display_refresh_hint_hz.unwrap_or(60.0),
                    }
                }
            }
        };
        let fixed_dt = tick_policy.resolve_fixed_dt_seconds(render_mode, Some(initial_fps_hint));
        let fps_limit_hint = initial_fps_hint;
        let thread_scale = (logical_threads as f32 / 8.0).clamp(0.75, 4.0);
        // Keep shards coarse enough to avoid scheduler overhead on high-core systems.
        let broadphase_chunk = logical_threads.saturating_mul(16).clamp(128, 512);
        let max_pairs = (96_000.0 * thread_scale).round() as usize;
        let max_manifolds = max_pairs;
        let solver_iterations = if logical_threads >= 20 { 5 } else { 4 };
        let tuned_max_substeps = (12 + logical_threads / 4).clamp(10, 24) as u32;
        let world = PhysicsWorld::new(PhysicsWorldConfig {
            // Keep Earth-like gravity explicit for showcase consistency across presets.
            gravity: Vector3::new(0.0, -9.81, 0.0),
            fixed_dt,
            max_substeps: tuned_max_substeps,
            broadphase: BroadphaseConfig {
                chunk_size: broadphase_chunk,
                max_candidate_pairs: max_pairs,
                shard_pair_reserve: 1_536,
            },
            narrowphase: NarrowphaseConfig {
                max_manifolds,
                ..NarrowphaseConfig::default()
            },
            solver: ContactSolverConfig {
                iterations: solver_iterations,
                baumgarte: 0.32,
                penetration_slop: 0.0015,
                parallel_contact_push_strength: 0.28,
                parallel_contact_push_threshold: 256,
                hard_position_projection_strength: 0.95,
                hard_position_projection_threshold: 128,
                max_projection_per_contact: 0.10,
                ..ContactSolverConfig::default()
            },
            ..PhysicsWorldConfig::default()
        });
        let scene = BounceTankSceneController::new(BounceTankSceneConfig {
            target_ball_count: 8_000,
            // Script can still tune this, but runtime starts from a progressive default.
            spawn_per_tick: (96.0 * thread_scale).round() as usize,
            ball_restitution: 0.74,
            wall_restitution: 0.78,
            linear_damping: 0.012,
            initial_speed_min: 0.35,
            initial_speed_max: 1.25,
            ..BounceTankSceneConfig::default()
        });
        eprintln!(
            "[mps] cpu profile logical={} physical={} thread_scale={:.2} broadphase_chunk={} max_pairs={} solver_iters={} max_substeps={}",
            logical_threads,
            physical_threads,
            thread_scale,
            broadphase_chunk,
            max_pairs,
            solver_iterations,
            tuned_max_substeps
        );

        let mut renderer =
            WgpuSceneRenderer::new(&device, &queue, config.format, size.width, size.height);
        // Always keep a deterministic high-quality FBX-equivalent mesh in slot 2 so script
        // `set_ball_mesh_slot(2)` stays stable even when tlsprite binding fails.
        renderer.bind_builtin_sphere_mesh_slot(&device, DEFAULT_FBX_BALL_SLOT, true);
        renderer.bind_builtin_sphere_mesh_slot(&device, AUTO_LOW_POLY_BALL_SLOT, false);
        let camera = FreeCameraController::default();
        let gamepad = GamepadManager::new();
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
            script_key_f_keyboard: false,
            keyboard_camera: CameraInputState::default(),
            mouse_look_held: false,
            look_lock_active: false,
            mouse_look_delta: (0.0, 0.0),
            camera_reset_requested: false,
            keyboard_modifiers: ModifiersState::empty(),
            gamepad,
            tick_policy,
            tick_profile: options.tick_profile,
            tick_hz: 1.0 / fixed_dt.max(1e-6),
            fps_limit_hint,
            uncapped_dynamic_fps_hint,
            mps_logical_threads: logical_threads,
            max_substeps: tuned_max_substeps,
            last_substeps: 0,
            tick_retune_timer: 0.0,
            adaptive_ball_render_limit: None,
            adaptive_live_ball_budget: None,
            adaptive_low_poly_override: false,
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

    fn on_keyboard_input(&mut self, event: &KeyEvent) -> RuntimeCommand {
        let pressed = event.state == ElementState::Pressed;
        self.update_camera_keyboard_input(event.physical_key, pressed);
        if let PhysicalKey::Code(KeyCode::KeyF) = event.physical_key {
            self.script_key_f_keyboard = pressed;
        }

        if !pressed || event.repeat {
            return RuntimeCommand::None;
        }

        let ctrl = self.keyboard_modifiers.control_key();
        let alt = self.keyboard_modifiers.alt_key();

        if alt && matches!(event.physical_key, PhysicalKey::Code(KeyCode::Enter)) {
            self.toggle_fullscreen();
            return RuntimeCommand::None;
        }

        if ctrl {
            match event.physical_key {
                PhysicalKey::Code(KeyCode::KeyQ) => return RuntimeCommand::Exit,
                PhysicalKey::Code(KeyCode::KeyF) => {
                    self.toggle_fullscreen();
                    return RuntimeCommand::None;
                }
                PhysicalKey::Code(KeyCode::KeyR) => {
                    self.camera_reset_requested = true;
                    return RuntimeCommand::None;
                }
                PhysicalKey::Code(KeyCode::KeyL) => {
                    self.look_lock_active = !self.look_lock_active;
                    return RuntimeCommand::None;
                }
                _ => {}
            }
        }
        RuntimeCommand::None
    }

    fn on_modifiers_changed(&mut self, modifiers: ModifiersState) {
        self.keyboard_modifiers = modifiers;
    }

    fn toggle_fullscreen(&self) {
        let next = if self.window.fullscreen().is_some() {
            None
        } else {
            Some(Fullscreen::Borderless(None))
        };
        self.window.set_fullscreen(next);
    }

    fn on_mouse_button(&mut self, state: ElementState, button: MouseButton) {
        if button == MouseButton::Right {
            self.mouse_look_held = state == ElementState::Pressed;
        }
    }

    fn on_device_event(&mut self, event: DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta } = event {
            self.mouse_look_delta.0 += delta.0 as f32;
            self.mouse_look_delta.1 += delta.1 as f32;
        }
    }

    fn poll_input_devices(&mut self) {
        self.gamepad.poll();
    }

    fn update_camera_keyboard_input(&mut self, key: PhysicalKey, pressed: bool) {
        match key {
            PhysicalKey::Code(KeyCode::KeyW) | PhysicalKey::Code(KeyCode::ArrowUp) => {
                self.keyboard_camera.forward = pressed
            }
            PhysicalKey::Code(KeyCode::KeyS) | PhysicalKey::Code(KeyCode::ArrowDown) => {
                self.keyboard_camera.backward = pressed
            }
            PhysicalKey::Code(KeyCode::KeyA) | PhysicalKey::Code(KeyCode::ArrowLeft) => {
                self.keyboard_camera.left = pressed
            }
            PhysicalKey::Code(KeyCode::KeyD) | PhysicalKey::Code(KeyCode::ArrowRight) => {
                self.keyboard_camera.right = pressed
            }
            PhysicalKey::Code(KeyCode::Space) | PhysicalKey::Code(KeyCode::KeyE) => {
                self.keyboard_camera.up = pressed
            }
            PhysicalKey::Code(KeyCode::ControlLeft)
            | PhysicalKey::Code(KeyCode::ControlRight)
            | PhysicalKey::Code(KeyCode::KeyQ)
            | PhysicalKey::Code(KeyCode::KeyC) => self.keyboard_camera.down = pressed,
            PhysicalKey::Code(KeyCode::ShiftLeft) | PhysicalKey::Code(KeyCode::ShiftRight) => {
                self.keyboard_camera.sprint = pressed
            }
            _ => {}
        }
    }

    fn script_camera_input(&mut self, view_dt: f32) -> TlscriptShowcaseControlInput {
        let gamepad = self.gamepad.camera_state();
        let sensitivity = self.camera.mouse_sensitivity().max(0.0001);
        let pad_look_to_mouse = (GAMEPAD_LOOK_SPEED_RAD * view_dt.max(0.0)) / sensitivity;

        let move_x = (axis_from_bools(self.keyboard_camera.right, self.keyboard_camera.left)
            + gamepad.move_x)
            .clamp(-1.0, 1.0);
        let move_y = (axis_from_bools(self.keyboard_camera.forward, self.keyboard_camera.backward)
            + gamepad.move_y)
            .clamp(-1.0, 1.0);
        let move_z = (axis_from_bools(self.keyboard_camera.up, self.keyboard_camera.down)
            + gamepad.rise
            - gamepad.descend)
            .clamp(-1.0, 1.0);
        let look_dx = self.mouse_look_delta.0 + gamepad.look_x * pad_look_to_mouse;
        let look_dy = self.mouse_look_delta.1 + gamepad.look_y * pad_look_to_mouse;
        self.mouse_look_delta = (0.0, 0.0);

        let look_active = self.look_lock_active
            || self.mouse_look_held
            || gamepad.look_x.abs() > 0.001
            || gamepad.look_y.abs() > 0.001;
        let reset_camera = std::mem::take(&mut self.camera_reset_requested);

        TlscriptShowcaseControlInput {
            move_x,
            move_y,
            move_z,
            look_dx,
            look_dy,
            sprint_down: self.keyboard_camera.sprint || gamepad.sprint,
            look_active,
            reset_camera,
        }
    }

    fn render_frame(&mut self) -> Result<(), Box<dyn Error>> {
        let frame_begin = Instant::now();
        let raw_dt = (frame_begin - self.frame_started_at).as_secs_f32();
        // Keep simulation time real-time (decoupled from render FPS) and only guard against large
        // stalls (alt-tab/debugger) so physics does not enter slow-motion at low FPS.
        let sim_dt = raw_dt.clamp(1.0 / 1_000.0, 0.25);
        // Camera/input smoothing can use a separate clamped delta.
        let view_dt = raw_dt.clamp(1.0 / 500.0, 1.0 / 24.0);
        self.frame_started_at = frame_begin;

        self.poll_input_devices();
        let script_camera_input = self.script_camera_input(view_dt);

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

        let frame_eval = self.script_program.evaluate_frame_with_controls(
            TlscriptShowcaseFrameInput {
                frame_index: self.script_frame_index,
                live_balls: self.scene.live_ball_count(),
                spawned_this_tick: self.script_last_spawned,
                key_f_down: self.script_key_f_keyboard || self.gamepad.action_f_down(),
            },
            script_camera_input,
        );

        if let Some(speed) = frame_eval.camera_move_speed {
            self.camera.set_move_speed(speed);
        }
        if let Some(sensitivity) = frame_eval.camera_look_sensitivity {
            self.camera.set_mouse_sensitivity(sensitivity);
        }
        if let Some(active) = frame_eval.camera_look_active {
            self.camera.set_look_active(&self.window, active);
        }
        if frame_eval.camera_reset_pose {
            self.camera.reset_pose();
        }
        if let Some((camera_eye, camera_target)) = frame_eval.camera_pose {
            self.camera.set_pose(camera_eye, camera_target);
        }
        self.camera.set_script_move_axis(
            frame_eval.camera_move_axis.unwrap_or([0.0, 0.0, 0.0]),
            frame_eval.camera_sprint.unwrap_or(false),
        );
        if let Some([look_dx, look_dy]) = frame_eval.camera_look_delta {
            self.camera.on_mouse_delta(look_dx, look_dy);
        }
        self.camera.update(view_dt);
        let (eye, target) = self.camera.eye_target();
        self.renderer.set_camera_view(
            &self.queue,
            self.size.width.max(1),
            self.size.height.max(1),
            eye,
            target,
        );

        let live_balls = self.scene.live_ball_count();
        let parallel_ready = frame_eval
            .dispatch_decision
            .as_ref()
            .map(|d| d.is_parallel())
            .unwrap_or(false);
        let force_full_fbx = frame_eval
            .force_full_fbx_sphere
            .unwrap_or(self.force_full_fbx_from_sprite);
        let mut runtime_patch = frame_eval.patch;
        let mut load_plan = choose_runtime_load_plan(
            self.fps_tracker.ema_fps(),
            raw_dt * 1_000.0,
            live_balls,
            self.last_substeps,
            self.max_substeps,
            parallel_ready,
            self.mps_logical_threads,
        );
        if matches!(self.tick_profile, TickProfile::Max) {
            load_plan.tick_scale = load_plan.tick_scale.max(0.90);
            let profile_cap = (10_u32 + (self.mps_logical_threads as u32 / 6)).clamp(10, 20);
            load_plan.max_substeps = load_plan.max_substeps.clamp(10, profile_cap);
        }
        self.adaptive_ball_render_limit = load_plan.visible_ball_limit;
        self.adaptive_live_ball_budget = load_plan.live_ball_budget;
        self.adaptive_low_poly_override = load_plan.force_low_poly_ball_mesh;

        if let Some(cap) = self.adaptive_live_ball_budget {
            let target = runtime_patch
                .target_ball_count
                .unwrap_or(self.scene.config().target_ball_count)
                .min(cap);
            runtime_patch.target_ball_count = Some(target);
        }
        runtime_patch.spawn_per_tick = Some(
            runtime_patch
                .spawn_per_tick
                .unwrap_or(load_plan.spawn_per_tick_cap)
                .min(load_plan.spawn_per_tick_cap),
        );

        if self.last_substeps + 1 >= self.max_substeps && live_balls > 1_500 {
            runtime_patch.spawn_per_tick =
                Some(runtime_patch.spawn_per_tick.unwrap_or(96).clamp(32, 96));
            runtime_patch.linear_damping =
                Some(runtime_patch.linear_damping.unwrap_or(0.016).max(0.018));
        }
        if !parallel_ready && live_balls > 2_500 {
            runtime_patch.spawn_per_tick =
                Some(runtime_patch.spawn_per_tick.unwrap_or(64).clamp(24, 64));
            runtime_patch.linear_damping =
                Some(runtime_patch.linear_damping.unwrap_or(0.018).max(0.020));
        }
        if self.adaptive_low_poly_override {
            // Preserve FBX silhouette; shed load via body/render budgets instead of slot swapping.
            runtime_patch.ball_mesh_slot = Some(DEFAULT_FBX_BALL_SLOT);
        }
        self.renderer.set_force_full_fbx_sphere(force_full_fbx);

        if let Some(cap) = self.adaptive_live_ball_budget {
            let _ = self.scene.enforce_live_ball_budget(&mut self.world, cap);
        }

        self.tick_retune_timer -= sim_dt;
        if self.tick_retune_timer <= 0.0 {
            self.max_substeps = load_plan.max_substeps.max(2);
            let mut desired_hz = choose_aggressive_tick_hz(
                self.tick_policy,
                self.tick_profile,
                self.fps_tracker.ema_fps().max(1.0),
                parallel_ready,
                self.last_substeps,
                self.max_substeps,
                live_balls,
                load_plan.tick_scale,
                self.fps_limit_hint,
                self.mps_logical_threads,
            );
            // Avoid fixed-step overload: if tick is too high for current FPS and max_substeps,
            // simulation falls behind (slow-motion). Clamp to catch-up-safe frequency.
            let catch_up_hz =
                (self.fps_tracker.ema_fps().max(1.0) * self.max_substeps as f32 * 0.88)
                    .clamp(24.0, 900.0);
            desired_hz = desired_hz.min(catch_up_hz);
            let ramp_up = desired_hz > self.tick_hz;
            let smoothing = match (self.tick_profile, ramp_up) {
                (TickProfile::Max, true) => 0.86,
                (TickProfile::Max, false) => 0.55,
                (_, true) => 0.72,
                (_, false) => 0.42,
            };
            self.tick_hz = smooth_tick_hz(self.tick_hz, desired_hz, smoothing);
            let hard_floor = match self.tick_profile {
                TickProfile::Balanced => 35.0,
                TickProfile::Max => {
                    let ema_floor = (self.fps_tracker.ema_fps().max(1.0) * 6.0).clamp(45.0, 180.0);
                    let cap_floor = if self.uncapped_dynamic_fps_hint {
                        self.fps_limit_hint * 0.45
                    } else {
                        self.fps_limit_hint * 0.60
                    };
                    ema_floor.min(cap_floor.clamp(45.0, 220.0))
                }
            };
            let floor_hz = hard_floor.min(catch_up_hz * 0.90).max(24.0);
            self.tick_hz = self.tick_hz.max(floor_hz).min(catch_up_hz);
            self.world
                .set_timestep(1.0 / self.tick_hz, self.max_substeps);
            self.tick_retune_timer = if ramp_up { 0.08 } else { 0.16 };
        }

        let _patch_metrics = self
            .scene
            .apply_runtime_patch(&mut self.world, runtime_patch);
        let tick = self.scene.physics_tick(&mut self.world);
        self.script_last_spawned = tick.spawned_this_tick;
        self.script_frame_index = self.script_frame_index.saturating_add(1);

        let substeps = self.world.step(sim_dt);
        self.last_substeps = substeps;
        let _ = self.scene.reconcile_after_step(&mut self.world);

        let mut frame = self.scene.build_frame_instances_with_ball_limit(
            &self.world,
            Some(self.world.interpolation_alpha()),
            self.adaptive_ball_render_limit,
        );
        let visible_ball_count = frame.opaque_3d.len().saturating_sub(12);
        let _hud = self.hud.append_to_sprites(
            TelemetryHudSample {
                fps: self.fps_tracker.ema_fps(),
                frame_time_ms: raw_dt * 1_000.0,
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
        if self.uncapped_dynamic_fps_hint {
            let measured_hint = self.fps_tracker.dynamic_uncapped_fps_hint();
            self.fps_limit_hint =
                smooth_tick_hz(self.fps_limit_hint, measured_hint, 0.22).clamp(48.0, 1_200.0);
        }
        let pacing_suffix = self
            .frame_cap_interval
            .map(|d| format!(" | cap {:.0}", 1.0 / d.as_secs_f32().max(1e-6)))
            .unwrap_or_else(|| {
                if self.uncapped_dynamic_fps_hint {
                    format!(" | target {:.0}", self.fps_limit_hint)
                } else {
                    String::new()
                }
            });
        let title = format!(
            "Tileline TLApp | FPS {:.1} | Frame {:.2} ms | Tick {:.0} Hz | Balls {} (draw {}) | Substeps {}{}{}{}",
            self.fps_tracker.ema_fps(),
            frame_time * 1_000.0,
            self.tick_hz,
            tick.live_balls,
            visible_ball_count,
            substeps,
            pacing_suffix,
            if self.adaptive_low_poly_override {
                " | lowpoly"
            } else {
                ""
            },
            if tick.scattered_this_tick > 0 {
                format!(" | scatter {}", tick.scattered_this_tick)
            } else {
                String::new()
            }
        );
        self.window.set_title(&title);

        if let Some(report) = report {
            let broadphase = self.world.broadphase().stats();
            let narrowphase = self.world.narrowphase().stats();
            println!(
                "tlapp fps | inst: {:>6.1} | ema: {:>6.1} | avg: {:>6.1} | stddev: {:>5.2} ms | balls: {:>5} | draw: {:>5} | substeps: {} | scattered: {:>4} | mps_threads: {} | shards: {} | pairs: {} | manifolds: {}",
                report.instant_fps,
                report.ema_fps,
                report.avg_fps,
                report.frame_time_stddev_ms,
                tick.live_balls,
                visible_ball_count,
                substeps,
                tick.scattered_this_tick,
                self.mps_logical_threads,
                broadphase.shard_count,
                broadphase.candidate_pairs,
                narrowphase.manifolds
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

#[inline]
fn axis_from_bools(positive: bool, negative: bool) -> f32 {
    (positive as i8 - negative as i8) as f32
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

#[derive(Debug, Clone, Copy, Default)]
struct GamepadCameraState {
    move_x: f32,
    move_y: f32,
    look_x: f32,
    look_y: f32,
    rise: f32,
    descend: f32,
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
    gamepad: GamepadCameraState,
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
            gamepad: GamepadCameraState::default(),
        }
    }
}

impl FreeCameraController {
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

    fn mouse_sensitivity(&self) -> f32 {
        self.mouse_sensitivity
    }

    fn set_script_move_axis(&mut self, axis: [f32; 3], sprint: bool) {
        self.gamepad.move_x = axis[0].clamp(-1.0, 1.0);
        self.gamepad.move_y = axis[1].clamp(-1.0, 1.0);
        self.gamepad.rise = axis[2].max(0.0);
        self.gamepad.descend = (-axis[2]).max(0.0);
        self.gamepad.sprint = sprint;
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

    fn reset_pose(&mut self) {
        self.position = Vector3::new(0.0, 12.0, 36.0);
        self.yaw_rad = 0.0;
        self.pitch_rad = -0.321_750_55;
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
        move_dir += right * self.gamepad.move_x;
        move_dir += forward_flat * self.gamepad.move_y;
        move_dir.y += self.gamepad.rise;
        move_dir.y -= self.gamepad.descend;

        let len = move_dir.norm();
        if len > 1e-5 {
            let move_dir = move_dir / len;
            let speed = self.move_speed
                * if self.gamepad.sprint {
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

#[cfg(feature = "gamepad")]
#[derive(Debug, Clone, Copy, Default)]
struct GamepadRawState {
    left_x: f32,
    left_y: f32,
    right_x: f32,
    right_y: f32,
    left_trigger_2: f32,
    right_trigger_2: f32,
    dpad_up: bool,
    dpad_down: bool,
    dpad_left: bool,
    dpad_right: bool,
    south: bool,
    sprint_left_trigger: bool,
    sprint_left_thumb: bool,
}

#[cfg(feature = "gamepad")]
struct GamepadManager {
    gilrs: Option<Gilrs>,
    active_id: Option<GamepadId>,
    raw: GamepadRawState,
}

#[cfg(feature = "gamepad")]
impl GamepadManager {
    fn new() -> Self {
        let mut manager = Self {
            gilrs: None,
            active_id: None,
            raw: GamepadRawState::default(),
        };
        match Gilrs::new() {
            Ok(gilrs) => {
                manager.active_id = gilrs.gamepads().next().map(|(id, _)| id);
                if manager.active_id.is_some() {
                    eprintln!("[input] gamepad support enabled");
                } else {
                    eprintln!("[input] gamepad subsystem ready (no device connected yet)");
                }
                manager.gilrs = Some(gilrs);
            }
            Err(err) => {
                eprintln!("[input] gamepad subsystem unavailable: {err}");
            }
        }
        manager
    }

    fn poll(&mut self) {
        loop {
            let event = match self.gilrs.as_mut().and_then(|g| g.next_event()) {
                Some(event) => event,
                None => break,
            };
            self.active_id = Some(event.id);
            match event.event {
                EventType::Connected => {
                    self.active_id = Some(event.id);
                    self.raw = GamepadRawState::default();
                }
                EventType::Disconnected => {
                    if self.active_id == Some(event.id) {
                        self.active_id = None;
                        self.raw = GamepadRawState::default();
                    }
                }
                EventType::AxisChanged(axis, value, _) => {
                    self.update_axis(axis, value);
                }
                EventType::ButtonPressed(button, _) => {
                    self.set_button(button, true);
                }
                EventType::ButtonReleased(button, _) => {
                    self.set_button(button, false);
                }
                _ => {}
            }
        }
    }

    fn action_f_down(&self) -> bool {
        self.raw.south
    }

    fn camera_state(&self) -> GamepadCameraState {
        let dpad_x = (self.raw.dpad_right as i8 - self.raw.dpad_left as i8) as f32;
        let dpad_y = (self.raw.dpad_up as i8 - self.raw.dpad_down as i8) as f32;
        GamepadCameraState {
            move_x: normalize_axis(self.raw.left_x + dpad_x),
            move_y: normalize_axis(-self.raw.left_y + dpad_y),
            look_x: normalize_axis(self.raw.right_x),
            look_y: normalize_axis(-self.raw.right_y),
            rise: normalize_axis(self.raw.right_trigger_2),
            descend: normalize_axis(self.raw.left_trigger_2),
            sprint: self.raw.sprint_left_trigger || self.raw.sprint_left_thumb,
        }
    }

    fn update_axis(&mut self, axis: Axis, value: f32) {
        let value = normalize_axis(value);
        match axis {
            Axis::LeftStickX => self.raw.left_x = value,
            Axis::LeftStickY => self.raw.left_y = value,
            Axis::RightStickX => self.raw.right_x = value,
            Axis::RightStickY => self.raw.right_y = value,
            Axis::LeftZ => self.raw.left_trigger_2 = value.max(0.0),
            Axis::RightZ => self.raw.right_trigger_2 = value.max(0.0),
            Axis::DPadX => {
                self.raw.dpad_left = value < -0.5;
                self.raw.dpad_right = value > 0.5;
            }
            Axis::DPadY => {
                self.raw.dpad_down = value < -0.5;
                self.raw.dpad_up = value > 0.5;
            }
            _ => {}
        }
    }

    fn set_button(&mut self, button: Button, pressed: bool) {
        match button {
            Button::South => self.raw.south = pressed,
            Button::LeftTrigger => self.raw.sprint_left_trigger = pressed,
            Button::LeftThumb => self.raw.sprint_left_thumb = pressed,
            Button::LeftTrigger2 => self.raw.left_trigger_2 = if pressed { 1.0 } else { 0.0 },
            Button::RightTrigger2 => self.raw.right_trigger_2 = if pressed { 1.0 } else { 0.0 },
            Button::DPadUp => self.raw.dpad_up = pressed,
            Button::DPadDown => self.raw.dpad_down = pressed,
            Button::DPadLeft => self.raw.dpad_left = pressed,
            Button::DPadRight => self.raw.dpad_right = pressed,
            _ => {}
        }
    }
}

#[cfg(not(feature = "gamepad"))]
struct GamepadManager;

#[cfg(not(feature = "gamepad"))]
impl GamepadManager {
    fn new() -> Self {
        eprintln!("[input] gamepad support disabled (runtime feature 'gamepad' is off)");
        Self
    }

    fn poll(&mut self) {}

    fn action_f_down(&self) -> bool {
        false
    }

    fn camera_state(&self) -> GamepadCameraState {
        GamepadCameraState::default()
    }
}

#[cfg(feature = "gamepad")]
#[inline]
fn normalize_axis(value: f32) -> f32 {
    let clamped = value.clamp(-1.0, 1.0);
    if clamped.abs() < GAMEPAD_DEADZONE {
        0.0
    } else {
        clamped
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

    fn dynamic_uncapped_fps_hint(&self) -> f32 {
        if self.count == 0 {
            return self.ema_fps.max(90.0);
        }
        let fast_lane_fps = self.fast_percentile_fps(0.20);
        let blended = fast_lane_fps * 0.78 + self.ema_fps * 0.22;
        blended.clamp(45.0, 1_200.0)
    }

    fn fast_percentile_fps(&self, quantile: f32) -> f32 {
        let count = self.count.max(1).min(self.frame_times.len());
        let mut scratch = [0.0_f32; 120];
        scratch[..count].copy_from_slice(&self.frame_times[..count]);
        scratch[..count].sort_by(|a, b| a.total_cmp(b));
        let idx = ((count as f32 - 1.0) * quantile.clamp(0.0, 1.0)).round() as usize;
        let frame_time = scratch[idx.min(count.saturating_sub(1))].max(1e-6);
        1.0 / frame_time
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
    profile: TickProfile,
    fps_estimate: f32,
    parallel_ready: bool,
    last_substeps: u32,
    max_substeps: u32,
    live_balls: usize,
    tick_scale: f32,
    fps_limit_hint: f32,
    logical_threads: usize,
) -> f32 {
    let fps_limit = fps_limit_hint.clamp(24.0, 1_200.0);
    let fps = fps_estimate.clamp(20.0, fps_limit * 2.0);
    let fps_ratio = (fps / fps_limit).clamp(0.35, 1.25);
    let mut multiplier = match profile {
        TickProfile::Balanced => {
            if parallel_ready {
                2.9
            } else {
                2.3
            }
        }
        TickProfile::Max => {
            if parallel_ready {
                3.4
            } else {
                2.9
            }
        }
    };
    if fps_ratio > 0.95 {
        multiplier += 1.0;
    }
    if fps_ratio > 1.05 {
        multiplier += 0.45;
    }
    if live_balls > 3_500 {
        multiplier += 0.2;
    }
    if live_balls > 5_500 {
        multiplier += 0.25;
    }
    if live_balls > 8_000 {
        multiplier += 0.25;
    }

    let mut target_hz = fps_limit * multiplier * fps_ratio;
    if last_substeps + 1 >= max_substeps {
        target_hz *= match profile {
            TickProfile::Balanced => 0.62,
            TickProfile::Max => 0.78,
        };
    } else if last_substeps >= max_substeps.saturating_sub(3) {
        target_hz *= match profile {
            TickProfile::Balanced => 0.78,
            TickProfile::Max => 0.90,
        };
    }
    if fps < 45.0 {
        target_hz *= match profile {
            TickProfile::Balanced => 0.85,
            TickProfile::Max => 0.93,
        };
    }
    target_hz *= match profile {
        TickProfile::Balanced => tick_scale.clamp(0.50, 1.20),
        TickProfile::Max => tick_scale.clamp(0.85, 1.35),
    };

    let dynamic_min = match profile {
        TickProfile::Balanced => (fps_limit * 0.5).max(policy.min_tick_hz.max(30.0)),
        TickProfile::Max => policy.min_tick_hz.max(45.0),
    };
    let thread_scale = (logical_threads as f32 / 8.0).clamp(0.75, 4.0);
    let ceiling_factor = match profile {
        TickProfile::Balanced => {
            if parallel_ready {
                8.0
            } else {
                6.0
            }
        }
        TickProfile::Max => {
            if parallel_ready {
                12.0
            } else {
                10.0
            }
        }
    };
    let thread_gain = match profile {
        TickProfile::Balanced => 0.8 + thread_scale * 0.4,
        TickProfile::Max => 0.9 + thread_scale * 0.55,
    };
    let dynamic_max =
        (fps_limit * ceiling_factor * thread_gain).min(policy.max_tick_hz.max(dynamic_min));

    target_hz.clamp(dynamic_min, dynamic_max)
}

fn smooth_tick_hz(current_hz: f32, target_hz: f32, alpha: f32) -> f32 {
    let alpha = alpha.clamp(0.0, 1.0);
    current_hz + (target_hz - current_hz) * alpha
}

#[derive(Debug, Clone, Copy)]
struct RuntimeLoadPlan {
    visible_ball_limit: Option<usize>,
    live_ball_budget: Option<usize>,
    spawn_per_tick_cap: usize,
    max_substeps: u32,
    force_low_poly_ball_mesh: bool,
    tick_scale: f32,
}

fn choose_runtime_load_plan(
    ema_fps: f32,
    frame_time_ms: f32,
    live_balls: usize,
    last_substeps: u32,
    max_substeps: u32,
    parallel_ready: bool,
    logical_threads: usize,
) -> RuntimeLoadPlan {
    let mut pressure = 0u32;
    let fps = ema_fps.clamp(1.0, 240.0);
    if fps < 55.0 {
        pressure += 1;
    }
    if fps < 42.0 {
        pressure += 1;
    }
    if fps < 30.0 {
        pressure += 1;
    }
    if fps < 20.0 {
        pressure += 1;
    }
    if frame_time_ms > 20.0 {
        pressure += 1;
    }
    if frame_time_ms > 33.0 {
        pressure += 1;
    }
    if frame_time_ms > 50.0 {
        pressure += 1;
    }
    if last_substeps + 1 >= max_substeps {
        pressure += 2;
    } else if last_substeps >= max_substeps.saturating_sub(2) {
        pressure += 1;
    }
    if !parallel_ready {
        pressure += 1;
    }
    if live_balls > 4_500 {
        pressure += 1;
    }
    if live_balls > 6_500 {
        pressure += 1;
    }

    let thread_scale = (logical_threads as f32 / 8.0).clamp(0.75, 4.0);
    let cap = |base: usize| ((base as f32) * thread_scale).round() as usize;

    match pressure {
        0..=2 => RuntimeLoadPlan {
            visible_ball_limit: None,
            live_ball_budget: None,
            spawn_per_tick_cap: cap(420),
            max_substeps: 14,
            force_low_poly_ball_mesh: false,
            tick_scale: 1.00,
        },
        3..=4 => RuntimeLoadPlan {
            visible_ball_limit: None,
            live_ball_budget: None,
            spawn_per_tick_cap: cap(360),
            max_substeps: 12,
            force_low_poly_ball_mesh: false,
            tick_scale: 0.96,
        },
        5..=6 => RuntimeLoadPlan {
            visible_ball_limit: None,
            live_ball_budget: None,
            spawn_per_tick_cap: cap(260),
            max_substeps: 10,
            force_low_poly_ball_mesh: false,
            tick_scale: 0.82,
        },
        7..=8 => RuntimeLoadPlan {
            visible_ball_limit: None,
            live_ball_budget: None,
            spawn_per_tick_cap: cap(200),
            max_substeps: 10,
            force_low_poly_ball_mesh: false,
            tick_scale: 0.72,
        },
        _ => RuntimeLoadPlan {
            visible_ball_limit: None,
            live_ball_budget: None,
            spawn_per_tick_cap: cap(160),
            max_substeps: 10,
            force_low_poly_ball_mesh: false,
            tick_scale: 0.60,
        },
    }
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
