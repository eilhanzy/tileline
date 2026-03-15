use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::Once;
use std::time::{Duration, Instant};

use crate::{
    app_runner, choose_scheduler_path_for_platform, compile_tljoint_scene_from_path,
    compile_tlpfile_scene_from_path, compile_tlscript_showcase, BounceTankRuntimePatch,
    BounceTankSceneConfig, BounceTankSceneController, DrawPathCompiler, GraphicsSchedulerPath,
    RenderSyncMode, RuntimePlatform, SceneFrameInstances, ScenePrimitive3d, TelemetryHudComposer,
    TelemetryHudSample, TickRatePolicy, TljointDiagnosticLevel, TljointSceneBundle,
    TlpfileDiagnosticLevel, TlpfileGraphicsScheduler, TlscriptCoordinateSpace,
    TlscriptShowcaseConfig, TlscriptShowcaseControlInput, TlscriptShowcaseFrameInput,
    TlscriptShowcaseFrameOutput, TlscriptShowcaseProgram, TlspriteHotReloadEvent, TlspriteProgram,
    TlspriteProgramCache, TlspriteWatchReloader, WgpuSceneRenderer,
};
use gms::safe_default_required_limits_for_adapter;
use nalgebra::Vector3;
use paradoxpe::{
    BroadphaseConfig, ContactSolverConfig, NarrowphaseConfig, PhysicsWorld, PhysicsWorldConfig,
};
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalSize};
use winit::event::{
    DeviceEvent, ElementState, KeyEvent, MouseButton, Touch, TouchPhase, WindowEvent,
};
use winit::event_loop::{ActiveEventLoop, ControlFlow};
use winit::keyboard::{Key, KeyCode, ModifiersState, NamedKey, PhysicalKey};
#[cfg(target_os = "android")]
use winit::platform::android::activity::AndroidApp;
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
    let app = TlApp::new(options, RuntimePlatform::current());
    app_runner::run_app_desktop(app)
}

/// Run TLApp from Android lifecycle entrypoint.
#[cfg(target_os = "android")]
pub fn run_with_android_app(android_app: AndroidApp) -> Result<(), Box<dyn Error>> {
    configure_parallel_runtime();
    let options = CliOptions::default();
    let app = TlApp::new(options, RuntimePlatform::Android);
    app_runner::run_app_android(android_app, app)
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
    render_distance: Option<f32>,
    adaptive_distance: ToggleAuto,
    distance_blur: ToggleAuto,
    fps_report_interval: Duration,
    project_path: Option<PathBuf>,
    joint_path: Option<PathBuf>,
    joint_scene: String,
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
            render_distance: None,
            adaptive_distance: ToggleAuto::Auto,
            distance_blur: ToggleAuto::Auto,
            fps_report_interval: Duration::from_secs_f32(1.0),
            project_path: None,
            joint_path: None,
            joint_scene: "main".to_string(),
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
                "--render-distance" => {
                    let value = next_arg(&mut args, "--render-distance")?;
                    options.render_distance = parse_render_distance(&value)?;
                }
                "--adaptive-distance" => {
                    let value = next_arg(&mut args, "--adaptive-distance")?;
                    options.adaptive_distance = ToggleAuto::parse(&value, "--adaptive-distance")?;
                }
                "--distance-blur" => {
                    let value = next_arg(&mut args, "--distance-blur")?;
                    options.distance_blur = ToggleAuto::parse(&value, "--distance-blur")?;
                }
                "--fps-report" => {
                    let value = next_arg(&mut args, "--fps-report")?;
                    options.fps_report_interval = parse_seconds_arg(&value, "--fps-report")?;
                }
                "--project" => {
                    let value = next_arg(&mut args, "--project")?;
                    options.project_path = Some(PathBuf::from(value));
                }
                "--joint" => {
                    let value = next_arg(&mut args, "--joint")?;
                    options.joint_path = Some(PathBuf::from(value));
                }
                "--scene" => {
                    let value = next_arg(&mut args, "--scene")?;
                    options.joint_scene = value.trim().to_string();
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
        if options.joint_scene.is_empty() {
            return Err("--scene cannot be empty".into());
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ToggleAuto {
    Auto,
    On,
    Off,
}

#[derive(Debug, Clone)]
struct SchedulerResolution {
    selected: GraphicsSchedulerPath,
    fallback_applied: bool,
    reason: String,
}

fn gms_supported_on_platform(platform: RuntimePlatform, adapter_info: &wgpu::AdapterInfo) -> bool {
    match platform {
        RuntimePlatform::Android => {
            matches!(adapter_info.backend, wgpu::Backend::Vulkan)
                && !matches!(adapter_info.device_type, wgpu::DeviceType::Cpu)
        }
        RuntimePlatform::Desktop => true,
    }
}

fn resolve_project_scheduler(
    manifest: TlpfileGraphicsScheduler,
    platform: RuntimePlatform,
    adapter_info: &wgpu::AdapterInfo,
) -> Result<SchedulerResolution, String> {
    match manifest {
        TlpfileGraphicsScheduler::Mgs => Ok(SchedulerResolution {
            selected: GraphicsSchedulerPath::Mgs,
            fallback_applied: false,
            reason: "manifest scheduler=mgs".to_string(),
        }),
        TlpfileGraphicsScheduler::Gms => {
            if gms_supported_on_platform(platform, adapter_info) {
                Ok(SchedulerResolution {
                    selected: GraphicsSchedulerPath::Gms,
                    fallback_applied: false,
                    reason: "manifest scheduler=gms".to_string(),
                })
            } else {
                Err(format!(
                    "manifest scheduler=gms is unsupported on {:?} (backend={:?}, type={:?}); explicit scheduler rejected for fail-soft policy",
                    platform, adapter_info.backend, adapter_info.device_type
                ))
            }
        }
        TlpfileGraphicsScheduler::Auto => {
            let decision = choose_scheduler_path_for_platform(adapter_info, platform);
            Ok(SchedulerResolution {
                selected: decision.path,
                fallback_applied: false,
                reason: format!("manifest scheduler=auto -> {}", decision.reason),
            })
        }
    }
}

struct AdapterBootstrap {
    instance: wgpu::Instance,
    surface: wgpu::Surface<'static>,
    adapter: wgpu::Adapter,
    bootstrap_note: String,
}

fn request_adapter_with_platform_policy(
    window: &Arc<Window>,
    platform: RuntimePlatform,
) -> Result<AdapterBootstrap, Box<dyn Error>> {
    let request_adapter = |instance: &wgpu::Instance,
                           surface: &wgpu::Surface<'static>|
     -> Result<wgpu::Adapter, String> {
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: Some(surface),
        }))
        .map_err(|err| format!("request_adapter failed: {err}"))
    };

    if matches!(platform, RuntimePlatform::Android) {
        let vk_instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });
        let vk_surface = vk_instance.create_surface(Arc::clone(window))?;
        match request_adapter(&vk_instance, &vk_surface) {
            Ok(vk_adapter) => {
                return Ok(AdapterBootstrap {
                    instance: vk_instance,
                    surface: vk_surface,
                    adapter: vk_adapter,
                    bootstrap_note: "vulkan-first path active".to_string(),
                });
            }
            Err(vk_err) => {
                let fallback_instance = wgpu::Instance::default();
                let fallback_surface = fallback_instance.create_surface(Arc::clone(window))?;
                if let Ok(fallback_adapter) = request_adapter(&fallback_instance, &fallback_surface)
                {
                    let info = fallback_adapter.get_info();
                    return Err(build_vulkan_unavailable_message(
                        &vk_err,
                        Some((&info.name, info.backend)),
                    )
                    .into());
                }
                return Err(build_vulkan_unavailable_message(&vk_err, None).into());
            }
        }
    }

    let instance = wgpu::Instance::default();
    let surface = instance.create_surface(Arc::clone(window))?;
    let adapter = request_adapter(&instance, &surface)?;
    Ok(AdapterBootstrap {
        instance,
        surface,
        adapter,
        bootstrap_note: "desktop default adapter path".to_string(),
    })
}

fn build_vulkan_unavailable_message(
    vk_err: &str,
    fallback: Option<(&str, wgpu::Backend)>,
) -> String {
    if let Some((name, backend)) = fallback {
        format!(
            "Vulkan unavailable on Android (probe error: {vk_err}); fallback adapter '{name}' backend={backend:?} was detected, entering fail-soft policy"
        )
    } else {
        format!(
            "Vulkan unavailable on Android (probe error: {vk_err}); no compatible fallback adapter detected"
        )
    }
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

impl ToggleAuto {
    fn parse(value: &str, flag: &str) -> Result<Self, Box<dyn Error>> {
        match value.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "on" | "1" | "true" => Ok(Self::On),
            "off" | "0" | "false" => Ok(Self::Off),
            _ => Err(format!("invalid {flag} value: {value} (expected auto|on|off)").into()),
        }
    }

    fn resolve(self, auto_default: bool) -> bool {
        match self {
            Self::Auto => auto_default,
            Self::On => true,
            Self::Off => false,
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

fn parse_render_distance(value: &str) -> Result<Option<f32>, Box<dyn Error>> {
    if value.eq_ignore_ascii_case("off") {
        return Ok(None);
    }
    let distance = value.parse::<f32>().map_err(|_| {
        format!("invalid --render-distance value: {value} (expected number or off)")
    })?;
    if !distance.is_finite() || distance <= 0.0 {
        return Err("--render-distance must be > 0.0 or 'off'".into());
    }
    Ok(Some(distance))
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
    println!("  --render-distance <N|off> Distance cull radius for 3D balls (default: auto)");
    println!("  --adaptive-distance <mode> Adaptive distance tuning: auto|on|off (default: auto)");
    println!("  --distance-blur <mode>    Distance haze blur: auto|on|off (default: auto)");
    println!("  --fps-report <sec>        CLI FPS report cadence (default: 1.0)");
    println!("  --project <path>          .tlpfile manifest (GMS required for TLApp runtime)");
    println!("  --joint <path>            .tljoint manifest path (overrides --script/--sprite)");
    println!("  --scene <name>            Scene id inside .tlpfile/.tljoint (default: main)");
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
    runtime: Option<TlAppRuntimeState>,
    platform: RuntimePlatform,
    exit_requested: bool,
}

impl TlApp {
    fn new(options: CliOptions, platform: RuntimePlatform) -> Self {
        Self {
            options,
            runtime: None,
            platform,
            exit_requested: false,
        }
    }
}

enum TlAppRuntimeState {
    Ready(TlAppRuntime),
    FailSoft(FailSoftRuntime),
}

impl TlAppRuntimeState {
    fn window_id(&self) -> WindowId {
        match self {
            Self::Ready(runtime) => runtime.window.id(),
            Self::FailSoft(runtime) => runtime.window.id(),
        }
    }

    fn schedule_next_redraw(&mut self, event_loop: &ActiveEventLoop) {
        match self {
            Self::Ready(runtime) => runtime.schedule_next_redraw(event_loop),
            Self::FailSoft(runtime) => runtime.schedule_next_redraw(event_loop),
        }
    }
}

impl ApplicationHandler for TlApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.runtime.is_some() {
            return;
        }

        match TlAppRuntime::new(event_loop, self.options.clone(), self.platform) {
            Ok(runtime) => {
                self.runtime = Some(TlAppRuntimeState::Ready(runtime));
            }
            Err(err) => {
                eprintln!("Failed to start TLApp runtime core, switching to fail-soft mode: {err}");
                let fail_soft =
                    FailSoftRuntime::new(event_loop, self.options.resolution, format!("{err}"));
                match fail_soft {
                    Ok(runtime) => {
                        self.runtime = Some(TlAppRuntimeState::FailSoft(runtime));
                    }
                    Err(inner) => {
                        eprintln!("Failed to initialize TLApp fail-soft runtime: {inner}");
                        self.exit_requested = true;
                        event_loop.exit();
                    }
                }
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(runtime_state) = self.runtime.as_mut() else {
            return;
        };
        if runtime_state.window_id() != window_id {
            return;
        }

        match event {
            WindowEvent::CloseRequested => {
                self.exit_requested = true;
                event_loop.exit();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                match runtime_state {
                    TlAppRuntimeState::Ready(runtime) => {
                        if matches!(runtime.on_keyboard_input(&event), RuntimeCommand::Exit) {
                            self.exit_requested = true;
                            event_loop.exit();
                            return;
                        }
                    }
                    TlAppRuntimeState::FailSoft(runtime) => {
                        if runtime.on_keyboard_input(&event) {
                            self.exit_requested = true;
                            event_loop.exit();
                            return;
                        }
                    }
                }
                if matches!(event.logical_key, Key::Named(NamedKey::Escape))
                    && event.state == ElementState::Pressed
                {
                    self.exit_requested = true;
                    event_loop.exit();
                }
            }
            WindowEvent::ModifiersChanged(modifiers) => {
                if let TlAppRuntimeState::Ready(runtime) = runtime_state {
                    runtime.on_modifiers_changed(modifiers.state());
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if let TlAppRuntimeState::Ready(runtime) = runtime_state {
                    runtime.on_mouse_button(state, button);
                }
            }
            WindowEvent::Touch(touch) => {
                if let TlAppRuntimeState::Ready(runtime) = runtime_state {
                    runtime.on_touch(touch);
                }
            }
            WindowEvent::Resized(size) => match runtime_state {
                TlAppRuntimeState::Ready(runtime) => runtime.resize(size),
                TlAppRuntimeState::FailSoft(runtime) => runtime.resize(size),
            },
            WindowEvent::RedrawRequested => match runtime_state {
                TlAppRuntimeState::Ready(runtime) => {
                    if let Err(err) = runtime.render_frame() {
                        eprintln!("Render error: {err}");
                        self.exit_requested = true;
                        event_loop.exit();
                    }
                }
                TlAppRuntimeState::FailSoft(runtime) => {
                    runtime.render_frame();
                }
            },
            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        if let Some(TlAppRuntimeState::Ready(runtime)) = self.runtime.as_mut() {
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
    sprite_loader: Option<TlspriteWatchReloader>,
    sprite_cache: Option<TlspriteProgramCache>,
    force_full_fbx_from_sprite: bool,
    script_runtime: ScriptRuntime<'static>,
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
    touch_look_id: Option<u64>,
    touch_last_position: Option<(f32, f32)>,
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
    render_distance: Option<f32>,
    render_distance_min: f32,
    render_distance_max: f32,
    adaptive_distance_enabled: bool,
    distance_blur_mode: ToggleAuto,
    distance_blur_enabled: bool,
    last_distance_culled: usize,
    last_distance_blurred: usize,
    last_framebuffer_fill_ratio: f32,
    framebuffer_fill_ema: f32,
    distance_retune_timer: f32,
    frame_time_ema_ms: f32,
    frame_time_jitter_ema_ms: f32,
    frame_time_budget_ms: f32,
    frame_started_at: Instant,
    frame_cap_interval: Option<Duration>,
    next_redraw_at: Instant,
    fps_tracker: FpsTracker,
    scheduler_path: GraphicsSchedulerPath,
    scheduler_reason: String,
    scheduler_fallback_applied: bool,
    adapter_backend: wgpu::Backend,
    adapter_name: String,
    present_mode: wgpu::PresentMode,
    platform: RuntimePlatform,
}

struct FailSoftRuntime {
    window: Arc<Window>,
    reason: String,
    size: PhysicalSize<u32>,
    last_present_note: Instant,
}

impl FailSoftRuntime {
    fn new(
        event_loop: &ActiveEventLoop,
        resolution: PhysicalSize<u32>,
        reason: String,
    ) -> Result<Self, Box<dyn Error>> {
        let window = Arc::new(
            event_loop.create_window(
                WindowAttributes::default()
                    .with_title("Tileline TLApp | Fail-Soft")
                    .with_inner_size(LogicalSize::new(
                        resolution.width as f64,
                        resolution.height as f64,
                    )),
            )?,
        );
        Ok(Self {
            window,
            reason,
            size: resolution,
            last_present_note: Instant::now(),
        })
    }

    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }
        self.size = new_size;
    }

    fn on_keyboard_input(&self, event: &KeyEvent) -> bool {
        let pressed = event.state == ElementState::Pressed && !event.repeat;
        if !pressed {
            return false;
        }
        matches!(event.logical_key, Key::Named(NamedKey::Escape))
            || matches!(event.physical_key, PhysicalKey::Code(KeyCode::KeyQ))
    }

    fn schedule_next_redraw(&mut self, event_loop: &ActiveEventLoop) {
        event_loop.set_control_flow(ControlFlow::WaitUntil(
            Instant::now() + Duration::from_millis(100),
        ));
        self.window.request_redraw();
    }

    fn render_frame(&mut self) {
        if self.last_present_note.elapsed() >= Duration::from_millis(450) {
            let title = format!(
                "Tileline TLApp | Fail-Soft | Unsupported runtime path | {}",
                self.reason
            );
            self.window.set_title(&title);
            self.last_present_note = Instant::now();
        }
    }
}

enum ScriptRuntime<'src> {
    Single(TlscriptShowcaseProgram<'src>),
    Joint(TljointSceneBundle),
    MultiScripts(Vec<TlscriptShowcaseProgram<'src>>),
}

impl<'src> ScriptRuntime<'src> {
    fn evaluate_frame(
        &self,
        input: TlscriptShowcaseFrameInput,
        controls: TlscriptShowcaseControlInput,
    ) -> TlscriptShowcaseFrameOutput {
        match self {
            Self::Single(program) => program.evaluate_frame_with_controls(input, controls),
            Self::Joint(bundle) => bundle.evaluate_frame(input, controls),
            Self::MultiScripts(programs) => {
                let mut merged = empty_showcase_output();
                for (index, program) in programs.iter().enumerate() {
                    let output = program.evaluate_frame_with_controls(input, controls);
                    merge_showcase_output(&mut merged, output, index);
                }
                merged
            }
        }
    }
}

fn empty_showcase_output() -> TlscriptShowcaseFrameOutput {
    TlscriptShowcaseFrameOutput {
        patch: BounceTankRuntimePatch::default(),
        force_full_fbx_sphere: None,
        camera_move_speed: None,
        camera_look_sensitivity: None,
        camera_pose: None,
        camera_coordinate_space: None,
        camera_translate_delta: None,
        camera_rotate_delta_deg: None,
        camera_move_axis: None,
        camera_look_delta: None,
        camera_sprint: None,
        camera_look_active: None,
        camera_reset_pose: false,
        dispatch_decision: None,
        warnings: Vec::new(),
        aborted_early: false,
    }
}

fn merge_showcase_output(
    merged: &mut TlscriptShowcaseFrameOutput,
    mut next: TlscriptShowcaseFrameOutput,
    script_index: usize,
) {
    merge_runtime_patch(&mut merged.patch, next.patch);
    if next.force_full_fbx_sphere.is_some() {
        merged.force_full_fbx_sphere = next.force_full_fbx_sphere;
    }
    if next.camera_move_speed.is_some() {
        merged.camera_move_speed = next.camera_move_speed;
    }
    if next.camera_look_sensitivity.is_some() {
        merged.camera_look_sensitivity = next.camera_look_sensitivity;
    }
    if next.camera_pose.is_some() {
        merged.camera_pose = next.camera_pose;
    }
    if next.camera_coordinate_space.is_some() {
        merged.camera_coordinate_space = next.camera_coordinate_space;
    }
    if next.camera_translate_delta.is_some() {
        merged.camera_translate_delta = next.camera_translate_delta;
    }
    if next.camera_rotate_delta_deg.is_some() {
        merged.camera_rotate_delta_deg = next.camera_rotate_delta_deg;
    }
    if next.camera_move_axis.is_some() {
        merged.camera_move_axis = next.camera_move_axis;
    }
    if next.camera_look_delta.is_some() {
        merged.camera_look_delta = next.camera_look_delta;
    }
    if next.camera_sprint.is_some() {
        merged.camera_sprint = next.camera_sprint;
    }
    if next.camera_look_active.is_some() {
        merged.camera_look_active = next.camera_look_active;
    }
    merged.camera_reset_pose |= next.camera_reset_pose;

    merged.dispatch_decision = match (merged.dispatch_decision, next.dispatch_decision) {
        (None, right) => right,
        (left, None) => left,
        (Some(existing), Some(candidate)) => {
            if candidate.is_parallel() {
                Some(candidate)
            } else {
                Some(existing)
            }
        }
    };

    merged.aborted_early |= next.aborted_early;
    if !next.warnings.is_empty() {
        for warning in next.warnings.drain(..) {
            merged
                .warnings
                .push(format!("script[{script_index}] {warning}"));
        }
    }
}

fn merge_runtime_patch(target: &mut BounceTankRuntimePatch, patch: BounceTankRuntimePatch) {
    if patch.target_ball_count.is_some() {
        target.target_ball_count = patch.target_ball_count;
    }
    if patch.spawn_per_tick.is_some() {
        target.spawn_per_tick = patch.spawn_per_tick;
    }
    if patch.linear_damping.is_some() {
        target.linear_damping = patch.linear_damping;
    }
    if patch.gravity.is_some() {
        target.gravity = patch.gravity;
    }
    if patch.contact_guard.is_some() {
        target.contact_guard = patch.contact_guard;
    }
    if patch.ball_restitution.is_some() {
        target.ball_restitution = patch.ball_restitution;
    }
    if patch.wall_restitution.is_some() {
        target.wall_restitution = patch.wall_restitution;
    }
    if patch.initial_speed_min.is_some() {
        target.initial_speed_min = patch.initial_speed_min;
    }
    if patch.initial_speed_max.is_some() {
        target.initial_speed_max = patch.initial_speed_max;
    }
    if patch.scatter_interval_ticks.is_some() {
        target.scatter_interval_ticks = patch.scatter_interval_ticks;
    }
    if patch.scatter_strength.is_some() {
        target.scatter_strength = patch.scatter_strength;
    }
    if patch.ball_mesh_slot.is_some() {
        target.ball_mesh_slot = patch.ball_mesh_slot;
    }
    if patch.container_mesh_slot.is_some() {
        target.container_mesh_slot = patch.container_mesh_slot;
    }
}

impl TlAppRuntime {
    fn new(
        event_loop: &ActiveEventLoop,
        options: CliOptions,
        platform: RuntimePlatform,
    ) -> Result<Self, Box<dyn Error>> {
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

        let adapter_bootstrap = request_adapter_with_platform_policy(&window, platform)?;
        let instance = adapter_bootstrap.instance;
        let surface = adapter_bootstrap.surface;
        let adapter = adapter_bootstrap.adapter;
        let adapter_info = adapter.get_info();
        eprintln!(
            "[runtime bootstrap] platform={:?} adapter='{}' backend={:?} note={}",
            platform, adapter_info.name, adapter_info.backend, adapter_bootstrap.bootstrap_note
        );

        let (required_limits, limit_clamp_report) =
            safe_default_required_limits_for_adapter(&adapter);
        if limit_clamp_report.any_clamped() {
            eprintln!(
                "[runtime limits] adapter='{}' clamped required limits to supported values (1d={}, 2d={}, 3d={}, layers={})",
                adapter_info.name,
                required_limits.max_texture_dimension_1d,
                required_limits.max_texture_dimension_2d,
                required_limits.max_texture_dimension_3d,
                required_limits.max_texture_array_layers
            );
        }
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("tlapp-device"),
                required_limits,
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
        let selected_present_mode = config.present_mode;
        surface.configure(&device, &config);

        let mut script_runtime = None;
        let mut joint_merged_sprite_program: Option<TlspriteProgram> = None;
        let mut bundle_sprite_root: Option<PathBuf> = None;
        let mut project_scheduler_manifest: Option<TlpfileGraphicsScheduler> = None;
        if let Some(project_path) = &options.project_path {
            let project_compile = compile_tlpfile_scene_from_path(
                project_path,
                Some(&options.joint_scene),
                TlscriptShowcaseConfig::default(),
            );
            for diagnostic in &project_compile.diagnostics {
                match diagnostic.level {
                    TlpfileDiagnosticLevel::Warning => {
                        eprintln!("[tlpfile warning] {}", diagnostic.message);
                    }
                    TlpfileDiagnosticLevel::Error => {
                        eprintln!("[tlpfile error] {}", diagnostic.message);
                    }
                }
            }
            if project_compile.has_errors() {
                return Err(format!(
                    "failed to compile .tlpfile '{}' scene '{}'",
                    project_path.display(),
                    options.joint_scene
                )
                .into());
            }

            let bundle = project_compile.bundle.ok_or_else(|| {
                format!(
                    "no bundle returned for .tlpfile '{}' scene '{}'",
                    project_path.display(),
                    options.joint_scene
                )
            })?;

            project_scheduler_manifest = Some(bundle.scheduler);

            if bundle.scene_name == "main" {
                let main_joint_ok = bundle
                    .selected_joint_path
                    .as_ref()
                    .and_then(|path| path.file_name())
                    .and_then(|name| name.to_str())
                    .map(|name| name == "main.tljoint")
                    .unwrap_or(false);
                if !main_joint_ok {
                    return Err(format!(
                        "scene 'main' in '{}' must resolve to main.tljoint",
                        project_path.display()
                    )
                    .into());
                }
            }

            let scene_name = bundle.scene_name.clone();
            let script_count = bundle.scripts.len();
            let sprite_count = bundle.sprite_count();
            let scheduler_name = bundle.scheduler.as_str();
            let joint_label = bundle
                .selected_joint_path
                .as_ref()
                .map(|path| path.display().to_string())
                .unwrap_or_else(|| String::from("none"));
            bundle_sprite_root = bundle
                .selected_joint_path
                .clone()
                .or_else(|| Some(bundle.project_path.clone()));
            joint_merged_sprite_program = bundle.merged_sprite_program.clone();
            script_runtime = Some(ScriptRuntime::MultiScripts(bundle.scripts));
            eprintln!(
                "[tlpfile] project='{}' scene='{}' scheduler={} scripts={} sprites={} joint={}",
                project_path.display(),
                scene_name,
                scheduler_name,
                script_count,
                sprite_count,
                joint_label
            );
        } else if let Some(joint_path) = &options.joint_path {
            let joint_compile = compile_tljoint_scene_from_path(
                joint_path,
                &options.joint_scene,
                TlscriptShowcaseConfig::default(),
            );
            for diagnostic in &joint_compile.diagnostics {
                match diagnostic.level {
                    TljointDiagnosticLevel::Warning => {
                        eprintln!("[tljoint warning] {}", diagnostic.message);
                    }
                    TljointDiagnosticLevel::Error => {
                        eprintln!("[tljoint error] {}", diagnostic.message);
                    }
                }
            }
            if joint_compile.has_errors() {
                return Err(format!(
                    "failed to compile .tljoint '{}' scene '{}'",
                    joint_path.display(),
                    options.joint_scene
                )
                .into());
            }
            let bundle = joint_compile.bundle.ok_or_else(|| {
                format!(
                    "no bundle returned for .tljoint '{}' scene '{}'",
                    joint_path.display(),
                    options.joint_scene
                )
            })?;
            bundle_sprite_root = Some(joint_path.clone());
            joint_merged_sprite_program = bundle.merged_sprite_program.clone();
            script_runtime = Some(ScriptRuntime::Joint(bundle));
            eprintln!(
                "[tljoint] active manifest='{}' scene='{}'",
                joint_path.display(),
                options.joint_scene
            );
        }
        if script_runtime.is_none() {
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
            script_runtime = Some(ScriptRuntime::Single(script_program));
        }

        let scheduler_resolution = if let Some(manifest_scheduler) = project_scheduler_manifest {
            resolve_project_scheduler(manifest_scheduler, platform, &adapter_info).map_err(
                |reason| {
                    format!(
                        "project scheduler rejected on '{}': {}",
                        adapter_info.name, reason
                    )
                },
            )?
        } else {
            let decision = choose_scheduler_path_for_platform(&adapter_info, platform);
            SchedulerResolution {
                selected: decision.path,
                fallback_applied: false,
                reason: decision.reason,
            }
        };
        eprintln!(
            "[scheduler] platform={:?} path={:?} fallback={} reason={}",
            platform,
            scheduler_resolution.selected,
            scheduler_resolution.fallback_applied,
            scheduler_resolution.reason
        );

        let logical_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .max(1);
        let physical_threads = num_cpus::get_physical().max(1);
        let mgs_like_path = matches!(scheduler_resolution.selected, GraphicsSchedulerPath::Mgs);
        // Treat MGS + ARM/Android as mobile-class scheduling: tighter tick ceilings and smaller
        // chunks help avoid frame-time chopping on heterogeneous SoCs (e.g., RK3588S).
        let mobile_class_tuning = matches!(platform, RuntimePlatform::Android)
            || mgs_like_path
            || cfg!(any(target_arch = "aarch64", target_arch = "arm"));
        let little_core_class = mobile_class_tuning && logical_threads <= 8;
        let adaptive_distance_enabled = options.adaptive_distance.resolve(mobile_class_tuning);
        let mut render_distance = options
            .render_distance
            .or_else(|| mobile_class_tuning.then_some(72.0));
        if render_distance.is_none() && adaptive_distance_enabled {
            render_distance = Some(if mobile_class_tuning { 84.0 } else { 96.0 });
        }
        let mut render_distance_min = 0.0;
        let mut render_distance_max = 0.0;
        if let Some(base) = render_distance {
            render_distance_min = (base * 0.72).clamp(28.0, base);
            render_distance_max = (base * 1.55).clamp(base, 220.0);
        }
        let distance_blur_mode = options.distance_blur;
        let distance_blur_enabled = distance_blur_mode.resolve(mobile_class_tuning);
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
        let initial_fps_hint = if mobile_class_tuning && options.fps_cap.is_none() {
            initial_fps_hint.min(if little_core_class { 96.0 } else { 120.0 })
        } else {
            initial_fps_hint
        };
        let tick_policy = if mobile_class_tuning {
            TickRatePolicy {
                min_tick_hz: 24.0,
                max_tick_hz: if little_core_class { 220.0 } else { 300.0 },
                ticks_per_render_frame: 1.85,
                default_tick_hz: (initial_fps_hint * 1.45).clamp(72.0, 180.0),
            }
        } else {
            TickRatePolicy {
                min_tick_hz: 30.0,
                max_tick_hz: (480.0 + logical_threads as f32 * 40.0).clamp(480.0, 1_440.0),
                ticks_per_render_frame: 2.6,
                default_tick_hz: (initial_fps_hint * 2.4).clamp(180.0, 420.0),
            }
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
        let thread_scale = if mobile_class_tuning {
            (logical_threads as f32 / 6.0).clamp(0.70, 2.20)
        } else {
            (logical_threads as f32 / 8.0).clamp(0.75, 4.0)
        };
        // Smaller mobile shards improve load balancing across big.LITTLE clusters.
        let broadphase_chunk = if little_core_class {
            logical_threads.saturating_mul(8).clamp(48, 96)
        } else if mobile_class_tuning {
            logical_threads.saturating_mul(10).clamp(64, 160)
        } else {
            // Keep shards coarse enough to avoid scheduler overhead on high-core systems.
            logical_threads.saturating_mul(16).clamp(128, 512)
        };
        let max_pairs = if mobile_class_tuning {
            (72_000.0 * thread_scale).round() as usize
        } else {
            (96_000.0 * thread_scale).round() as usize
        };
        let max_manifolds = max_pairs;
        let solver_iterations = if little_core_class {
            4
        } else if logical_threads >= 20 {
            5
        } else {
            4
        };
        let tuned_max_substeps = if mobile_class_tuning {
            (8 + logical_threads / 4).clamp(6, 12) as u32
        } else {
            (12 + logical_threads / 4).clamp(10, 24) as u32
        };
        let world = PhysicsWorld::new(PhysicsWorldConfig {
            // Keep Earth-like gravity explicit for showcase consistency across presets.
            gravity: Vector3::new(0.0, -9.81, 0.0),
            fixed_dt,
            max_substeps: tuned_max_substeps,
            broadphase: BroadphaseConfig {
                chunk_size: broadphase_chunk,
                max_candidate_pairs: max_pairs,
                shard_pair_reserve: if mobile_class_tuning { 768 } else { 1_536 },
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
                parallel_contact_push_threshold: if mobile_class_tuning { 128 } else { 256 },
                hard_position_projection_strength: 0.95,
                hard_position_projection_threshold: if mobile_class_tuning { 96 } else { 128 },
                max_projection_per_contact: if mobile_class_tuning { 0.08 } else { 0.10 },
                ..ContactSolverConfig::default()
            },
            ..PhysicsWorldConfig::default()
        });
        let scene = BounceTankSceneController::new(BounceTankSceneConfig {
            target_ball_count: 8_000,
            // Script can still tune this, but runtime starts from a progressive default.
            spawn_per_tick: if mobile_class_tuning {
                (68.0 * thread_scale).round() as usize
            } else {
                (96.0 * thread_scale).round() as usize
            },
            ball_restitution: 0.74,
            wall_restitution: 0.78,
            linear_damping: 0.012,
            initial_speed_min: 0.35,
            initial_speed_max: 1.25,
            ..BounceTankSceneConfig::default()
        });
        let mut renderer =
            WgpuSceneRenderer::new(&device, &queue, config.format, size.width, size.height);
        // Always keep a deterministic high-quality FBX-equivalent mesh in slot 2 so script
        // `set_ball_mesh_slot(2)` stays stable even when tlsprite binding fails.
        renderer.bind_builtin_sphere_mesh_slot(&device, DEFAULT_FBX_BALL_SLOT, true);
        renderer.bind_builtin_sphere_mesh_slot(&device, AUTO_LOW_POLY_BALL_SLOT, false);
        let camera = FreeCameraController::default();
        let gamepad = GamepadManager::new();
        let (eye, target) = camera.eye_target();
        if let Some(current_distance) = render_distance {
            let ext = scene.config().container_half_extents;
            let tank_radius = (ext[0] * ext[0] + ext[1] * ext[1] + ext[2] * ext[2]).sqrt();
            let camera_to_center = (eye[0] * eye[0] + eye[1] * eye[1] + eye[2] * eye[2]).sqrt();
            let stable_distance_floor = (camera_to_center + tank_radius + 6.0).clamp(40.0, 180.0);
            let guarded_distance = current_distance.max(stable_distance_floor);
            render_distance = Some(guarded_distance);
            render_distance_min = render_distance_min
                .max(stable_distance_floor * 0.86)
                .min(guarded_distance);
            render_distance_max = render_distance_max
                .max(stable_distance_floor * 1.20)
                .max(guarded_distance)
                .clamp(guarded_distance, 260.0);
        }
        renderer.set_camera_view(&queue, size.width.max(1), size.height.max(1), eye, target);
        eprintln!(
            "[mps] cpu profile logical={} physical={} mobile_tuning={} little_core_class={} thread_scale={:.2} broadphase_chunk={} max_pairs={} solver_iters={} max_substeps={} render_distance={:?} adaptive_distance={} distance_blur={:?} ({})",
            logical_threads,
            physical_threads,
            mobile_class_tuning,
            little_core_class,
            thread_scale,
            broadphase_chunk,
            max_pairs,
            solver_iterations,
            tuned_max_substeps,
            render_distance,
            adaptive_distance_enabled,
            distance_blur_mode,
            distance_blur_enabled
        );

        let draw_compiler = DrawPathCompiler::new();
        let hud = TelemetryHudComposer::new(Default::default());
        let mut scene = scene;
        let mut force_full_fbx_from_sprite = false;
        let (sprite_loader, sprite_cache) = if let Some(program) = joint_merged_sprite_program {
            force_full_fbx_from_sprite = program.requires_full_fbx_render();
            scene.set_sprite_program(program.clone());
            if let Some(root_hint) = bundle_sprite_root.as_deref() {
                bind_renderer_meshes_from_tlsprite(&mut renderer, &device, root_hint, &program);
            }
            (None, None)
        } else {
            let mut sprite_loader = TlspriteWatchReloader::new(&options.sprite_path);
            let mut sprite_cache = TlspriteProgramCache::new();
            if let Some(warn) = sprite_loader.init_warning() {
                eprintln!("[tlsprite watch] {warn}");
            }
            eprintln!("[tlsprite watch] backend={:?}", sprite_loader.backend());
            let event = sprite_loader.reload_into_cache(&mut sprite_cache);
            print_tlsprite_event("[tlsprite boot]", event);
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
            (Some(sprite_loader), Some(sprite_cache))
        };
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
            script_runtime: script_runtime.expect("script runtime must be initialized"),
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
            touch_look_id: None,
            touch_last_position: None,
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
            render_distance,
            render_distance_min,
            render_distance_max,
            adaptive_distance_enabled,
            distance_blur_mode,
            distance_blur_enabled,
            last_distance_culled: 0,
            last_distance_blurred: 0,
            last_framebuffer_fill_ratio: 0.0,
            framebuffer_fill_ema: 0.0,
            distance_retune_timer: 0.0,
            frame_time_ema_ms: 0.0,
            frame_time_jitter_ema_ms: 0.0,
            frame_time_budget_ms: (1_000.0 / fps_limit_hint.max(24.0)).clamp(3.0, 41.0),
            frame_started_at: now,
            frame_cap_interval,
            next_redraw_at: now,
            fps_tracker: FpsTracker::new(fps_report_interval),
            scheduler_path: scheduler_resolution.selected,
            scheduler_reason: scheduler_resolution.reason,
            scheduler_fallback_applied: scheduler_resolution.fallback_applied,
            adapter_backend: adapter_info.backend,
            adapter_name: adapter_info.name,
            present_mode: selected_present_mode,
            platform,
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
        } else {
            let mobile_path = matches!(self.platform, RuntimePlatform::Android)
                || matches!(self.scheduler_path, GraphicsSchedulerPath::Mgs);
            if mobile_path {
                // Avoid uncapped busy-spin on mobile/TBDR paths; it can cause whole-system
                // chopping even when the app's own FPS appears acceptable.
                let next = Instant::now() + Duration::from_millis(2);
                self.next_redraw_at = next;
                event_loop.set_control_flow(ControlFlow::WaitUntil(next));
                self.window.request_redraw();
                return;
            }
        }

        event_loop.set_control_flow(ControlFlow::Poll);
        self.window.request_redraw();
    }

    fn retune_render_distance(&mut self, mobile_path: bool) {
        if !self.adaptive_distance_enabled || self.frame_time_ema_ms <= f32::EPSILON {
            return;
        }
        let Some(mut current) = self.render_distance else {
            return;
        };
        if self.render_distance_max <= self.render_distance_min + f32::EPSILON {
            return;
        }

        let target_ms = (1_000.0 / self.fps_limit_hint.max(24.0)).clamp(5.0, 42.0);
        let frame_pressure = (self.frame_time_ema_ms / target_ms).clamp(0.4, 2.5);
        let jitter_pressure =
            (self.frame_time_jitter_ema_ms / (target_ms * 0.5).max(0.5)).clamp(0.0, 2.0);
        let fill_pressure = self.framebuffer_fill_ema.clamp(0.0, 3.0);

        let overload = (frame_pressure - 1.0).max(0.0)
            + (fill_pressure - 1.0).max(0.0) * 0.75
            + jitter_pressure * 0.18;
        let headroom = (1.0 - frame_pressure).max(0.0)
            + (0.85 - fill_pressure).max(0.0) * 0.65
            + (0.12 - jitter_pressure).max(0.0) * 0.40;

        if overload > 0.04 {
            let shrink =
                (0.025 + overload * 0.035).clamp(0.02, if mobile_path { 0.10 } else { 0.07 });
            current *= 1.0 - shrink;
        } else if headroom > 0.10 {
            let grow =
                (0.008 + headroom * 0.025).clamp(0.008, if mobile_path { 0.06 } else { 0.04 });
            current *= 1.0 + grow;
        }

        let snapped = (current * 2.0).round() * 0.5;
        self.render_distance =
            Some(snapped.clamp(self.render_distance_min, self.render_distance_max));
    }

    fn refresh_distance_blur_state(&mut self, mobile_path: bool) {
        self.distance_blur_enabled = match self.distance_blur_mode {
            ToggleAuto::On => true,
            ToggleAuto::Off => false,
            ToggleAuto::Auto => {
                if self.distance_blur_enabled {
                    !(self.framebuffer_fill_ema < 0.72
                        && self.frame_time_ema_ms < self.frame_time_budget_ms * 0.90
                        && !mobile_path)
                } else {
                    self.framebuffer_fill_ema > 0.92
                        || self.frame_time_ema_ms > self.frame_time_budget_ms * 1.05
                        || mobile_path
                }
            }
        };
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

    fn on_touch(&mut self, touch: Touch) {
        let id = touch.id;
        let current = (touch.location.x as f32, touch.location.y as f32);
        match touch.phase {
            TouchPhase::Started => {
                self.touch_look_id = Some(id);
                self.touch_last_position = Some(current);
                self.mouse_look_held = true;
            }
            TouchPhase::Moved => {
                if self.touch_look_id == Some(id) {
                    if let Some(previous) = self.touch_last_position {
                        self.mouse_look_delta.0 += current.0 - previous.0;
                        self.mouse_look_delta.1 += current.1 - previous.1;
                    }
                    self.touch_last_position = Some(current);
                }
            }
            TouchPhase::Ended | TouchPhase::Cancelled => {
                if self.touch_look_id == Some(id) {
                    self.touch_look_id = None;
                    self.touch_last_position = None;
                    self.mouse_look_held = false;
                }
            }
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

        let move_x = merge_axis_inputs(
            axis_from_bools(self.keyboard_camera.right, self.keyboard_camera.left),
            gamepad.move_x,
        );
        let move_y = merge_axis_inputs(
            axis_from_bools(self.keyboard_camera.forward, self.keyboard_camera.backward),
            gamepad.move_y,
        );
        let move_z = merge_axis_inputs(
            axis_from_bools(self.keyboard_camera.up, self.keyboard_camera.down),
            gamepad.rise - gamepad.descend,
        );
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
        let mobile_path = matches!(self.platform, RuntimePlatform::Android)
            || matches!(self.scheduler_path, GraphicsSchedulerPath::Mgs);
        let raw_dt = (frame_begin - self.frame_started_at).as_secs_f32();
        // Keep simulation time real-time (decoupled from render FPS) and only guard against large
        // stalls (alt-tab/debugger) so physics does not enter slow-motion at low FPS.
        let sim_dt = raw_dt.clamp(1.0 / 1_000.0, 0.25);
        // Camera/input smoothing can use a separate clamped delta.
        let view_dt = raw_dt.clamp(1.0 / 500.0, 1.0 / 24.0);
        let frame_ms = raw_dt * 1_000.0;
        if self.frame_time_ema_ms <= f32::EPSILON {
            self.frame_time_ema_ms = frame_ms;
            self.frame_time_jitter_ema_ms = 0.0;
        } else {
            let frame_delta = (frame_ms - self.frame_time_ema_ms).abs();
            self.frame_time_ema_ms += (frame_ms - self.frame_time_ema_ms) * 0.10;
            self.frame_time_jitter_ema_ms += (frame_delta - self.frame_time_jitter_ema_ms) * 0.16;
        }
        self.frame_started_at = frame_begin;

        self.poll_input_devices();
        let script_camera_input = self.script_camera_input(view_dt);

        if let (Some(sprite_loader), Some(sprite_cache)) =
            (self.sprite_loader.as_mut(), self.sprite_cache.as_mut())
        {
            let event = sprite_loader.reload_into_cache(sprite_cache);
            match &event {
                TlspriteHotReloadEvent::Applied { .. } => {
                    print_tlsprite_event("[tlsprite reload]", event);
                    if let Some(program) =
                        sprite_cache.program_for_path(sprite_loader.path()).cloned()
                    {
                        self.force_full_fbx_from_sprite = program.requires_full_fbx_render();
                        self.scene.set_sprite_program(program.clone());
                        bind_renderer_meshes_from_tlsprite(
                            &mut self.renderer,
                            &self.device,
                            sprite_loader.path(),
                            &program,
                        );
                    }
                }
                TlspriteHotReloadEvent::Unchanged => {}
                _ => print_tlsprite_event("[tlsprite reload]", event),
            }
        }

        let frame_eval = self.script_runtime.evaluate_frame(
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
        if let Some(space) = frame_eval.camera_coordinate_space {
            self.camera.set_script_coordinate_space(space);
        }
        if let Some(delta) = frame_eval.camera_translate_delta {
            self.camera.apply_script_translate_delta(delta);
        }
        if let Some(delta_deg) = frame_eval.camera_rotate_delta_deg {
            self.camera.apply_script_rotate_delta_deg(delta_deg);
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
        self.frame_time_budget_ms = (1_000.0 / self.fps_limit_hint.max(24.0)).clamp(3.0, 41.0);
        self.distance_retune_timer -= sim_dt;
        if self.distance_retune_timer <= 0.0 {
            self.retune_render_distance(mobile_path);
            self.distance_retune_timer = if mobile_path { 0.22 } else { 0.28 };
        }
        self.refresh_distance_blur_state(mobile_path);
        let mut load_plan = choose_runtime_load_plan(
            self.fps_tracker.ema_fps(),
            raw_dt * 1_000.0,
            live_balls,
            self.last_substeps,
            self.max_substeps,
            parallel_ready,
            self.mps_logical_threads,
            mobile_path,
            self.framebuffer_fill_ema,
            self.render_distance,
        );
        let moderate_jitter = self.frame_time_ema_ms > self.frame_time_budget_ms * 1.10
            || self.frame_time_jitter_ema_ms > 1.8;
        let severe_jitter = self.frame_time_ema_ms > self.frame_time_budget_ms * 1.25
            || self.frame_time_jitter_ema_ms > 3.2;
        if moderate_jitter {
            load_plan.tick_scale *= 0.82;
            load_plan.max_substeps = load_plan.max_substeps.min(8);
            load_plan.spawn_per_tick_cap = load_plan.spawn_per_tick_cap.min(180);
        }
        if severe_jitter {
            load_plan.tick_scale *= 0.68;
            load_plan.max_substeps = load_plan.max_substeps.min(6);
            load_plan.spawn_per_tick_cap = load_plan.spawn_per_tick_cap.min(120);
        }
        if mobile_path {
            load_plan.tick_scale *= if severe_jitter { 0.70 } else { 0.82 };
            load_plan.max_substeps = load_plan
                .max_substeps
                .min(if severe_jitter { 4 } else { 6 });
            load_plan.spawn_per_tick_cap =
                load_plan
                    .spawn_per_tick_cap
                    .min(if severe_jitter { 96 } else { 112 });
        }
        if matches!(self.tick_profile, TickProfile::Max) {
            let min_tick_scale = if severe_jitter {
                0.62
            } else if moderate_jitter {
                0.74
            } else if mobile_path {
                0.68
            } else {
                0.90
            };
            load_plan.tick_scale = load_plan.tick_scale.max(min_tick_scale);
            let profile_cap = if mobile_path {
                (5_u32 + (self.mps_logical_threads as u32 / 10)).clamp(5, 8)
            } else {
                (10_u32 + (self.mps_logical_threads as u32 / 6)).clamp(10, 20)
            };
            let min_substeps = if mobile_path { 3 } else { 10 };
            load_plan.max_substeps = load_plan
                .max_substeps
                .clamp(min_substeps, profile_cap.max(min_substeps));
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
        let mut force_full_fbx_runtime = force_full_fbx;
        if self.adaptive_low_poly_override {
            // On mobile/TBDR profiles, visual fallback is cheaper than frame-time collapse.
            runtime_patch.ball_mesh_slot = Some(AUTO_LOW_POLY_BALL_SLOT);
            force_full_fbx_runtime = false;
        }
        self.renderer
            .set_force_full_fbx_sphere(force_full_fbx_runtime);

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
                mobile_path,
            );
            if mobile_path {
                let mobile_ceiling = match self.tick_profile {
                    TickProfile::Balanced => 120.0,
                    TickProfile::Max => 160.0,
                };
                desired_hz = desired_hz.min(mobile_ceiling);
            }
            // Avoid fixed-step overload: if tick is too high for current FPS and max_substeps,
            // simulation falls behind (slow-motion). Clamp to catch-up-safe frequency.
            let catch_up_factor = if mobile_path { 0.78 } else { 0.88 };
            let catch_up_hz =
                (self.fps_tracker.ema_fps().max(1.0) * self.max_substeps as f32 * catch_up_factor)
                    .clamp(24.0, 900.0);
            desired_hz = desired_hz.min(catch_up_hz);
            let ramp_up = desired_hz > self.tick_hz;
            let smoothing = if mobile_path {
                match (self.tick_profile, ramp_up) {
                    (TickProfile::Max, true) => 0.48,
                    (TickProfile::Max, false) => 0.30,
                    (_, true) => 0.42,
                    (_, false) => 0.26,
                }
            } else {
                match (self.tick_profile, ramp_up) {
                    (TickProfile::Max, true) => 0.86,
                    (TickProfile::Max, false) => 0.55,
                    (_, true) => 0.72,
                    (_, false) => 0.42,
                }
            };
            self.tick_hz = smooth_tick_hz(self.tick_hz, desired_hz, smoothing);
            let hard_floor = match self.tick_profile {
                TickProfile::Balanced => 35.0,
                TickProfile::Max => {
                    let ema_floor = if mobile_path {
                        (self.fps_tracker.ema_fps().max(1.0) * 1.45).clamp(28.0, 84.0)
                    } else {
                        (self.fps_tracker.ema_fps().max(1.0) * 6.0).clamp(45.0, 180.0)
                    };
                    let cap_floor = if mobile_path {
                        if self.uncapped_dynamic_fps_hint {
                            self.fps_limit_hint * 0.26
                        } else {
                            self.fps_limit_hint * 0.32
                        }
                    } else if self.uncapped_dynamic_fps_hint {
                        self.fps_limit_hint * 0.45
                    } else {
                        self.fps_limit_hint * 0.60
                    };
                    ema_floor.min(cap_floor.clamp(32.0, 220.0))
                }
            };
            let floor_hz = hard_floor.min(catch_up_hz * 0.90).max(24.0);
            self.tick_hz = self.tick_hz.max(floor_hz).min(catch_up_hz);
            self.world
                .set_timestep(1.0 / self.tick_hz, self.max_substeps);
            self.tick_retune_timer = if mobile_path {
                if ramp_up {
                    0.12
                } else {
                    0.08
                }
            } else if ramp_up {
                0.08
            } else {
                0.16
            };
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
        let distance_stats = apply_render_distance_haze(
            &mut frame,
            eye,
            self.render_distance,
            self.distance_blur_enabled,
        );
        self.last_distance_culled = distance_stats.culled;
        self.last_distance_blurred = distance_stats.blurred;
        let fill_ratio = estimate_framebuffer_fill_ratio(
            &frame,
            self.size.width.max(1),
            self.size.height.max(1),
        );
        self.last_framebuffer_fill_ratio = fill_ratio;
        if self.framebuffer_fill_ema <= f32::EPSILON {
            self.framebuffer_fill_ema = fill_ratio;
        } else {
            self.framebuffer_fill_ema += (fill_ratio - self.framebuffer_fill_ema) * 0.18;
        }
        let visible_ball_count = count_rendered_balls(&frame);
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
            let upper_hint = if mobile_path { 144.0 } else { 1_200.0 };
            self.fps_limit_hint =
                smooth_tick_hz(self.fps_limit_hint, measured_hint, 0.22).clamp(48.0, upper_hint);
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
        let scheduler_label = scheduler_path_label(self.scheduler_path);
        let distance_suffix = match self.render_distance {
            Some(distance) => format!(
                " | rd {:.0}m c{} b{} fill {:.2}",
                distance,
                self.last_distance_culled,
                self.last_distance_blurred,
                self.last_framebuffer_fill_ratio
            ),
            None => String::new(),
        };
        let title = format!(
            "Tileline TLApp | FPS {:.1} | Frame {:.2} ms | Tick {:.0} Hz | Balls {} (draw {}) | Substeps {} | {:?} {} {:?}{}{}{}{}{}",
            self.fps_tracker.ema_fps(),
            frame_time * 1_000.0,
            self.tick_hz,
            tick.live_balls,
            visible_ball_count,
            substeps,
            self.adapter_backend,
            scheduler_label,
            self.present_mode,
            if self.scheduler_fallback_applied {
                " | fallback"
            } else {
                ""
            },
            if self.adaptive_low_poly_override {
                " | lowpoly"
            } else {
                ""
            },
            if tick.scattered_this_tick > 0 {
                format!(" | scatter {}", tick.scattered_this_tick)
            } else {
                String::new()
            },
            distance_suffix,
            pacing_suffix,
        );
        self.window.set_title(&title);

        if let Some(report) = report {
            let broadphase = self.world.broadphase().stats();
            let narrowphase = self.world.narrowphase().stats();
            println!(
                "tlapp fps | inst: {:>6.1} | ema: {:>6.1} | avg: {:>6.1} | stddev: {:>5.2} ms | balls: {:>5} | draw: {:>5} | substeps: {} | scattered: {:>4} | rd_culled: {:>4} | rd_blur: {:>4} | fill: {:>4.2} | fill_ema: {:>4.2} | mps_threads: {} | shards: {} | pairs: {} | manifolds: {} | platform: {:?} | backend: {:?} | scheduler: {} | present: {:?} | fallback: {} | adapter: {} | reason: {}",
                report.instant_fps,
                report.ema_fps,
                report.avg_fps,
                report.frame_time_stddev_ms,
                tick.live_balls,
                visible_ball_count,
                substeps,
                tick.scattered_this_tick,
                self.last_distance_culled,
                self.last_distance_blurred,
                self.last_framebuffer_fill_ratio,
                self.framebuffer_fill_ema,
                self.mps_logical_threads,
                broadphase.shard_count,
                broadphase.candidate_pairs,
                narrowphase.manifolds,
                self.platform,
                self.adapter_backend,
                scheduler_label,
                self.present_mode,
                self.scheduler_fallback_applied,
                self.adapter_name,
                self.scheduler_reason
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

#[inline]
fn merge_axis_inputs(primary: f32, secondary: f32) -> f32 {
    (primary + secondary).clamp(-1.0, 1.0)
}

#[inline]
fn scheduler_path_label(path: GraphicsSchedulerPath) -> &'static str {
    match path {
        GraphicsSchedulerPath::Gms => "gms",
        GraphicsSchedulerPath::Mgs => "mgs",
    }
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
    script_coordinate_space: TlscriptCoordinateSpace,
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
            script_coordinate_space: TlscriptCoordinateSpace::World,
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

    fn set_script_coordinate_space(&mut self, space: TlscriptCoordinateSpace) {
        self.script_coordinate_space = space;
    }

    fn apply_script_translate_delta(&mut self, delta: [f32; 3]) {
        let delta = Vector3::new(delta[0], delta[1], delta[2]);
        match self.script_coordinate_space {
            TlscriptCoordinateSpace::World => {
                self.position += delta;
            }
            TlscriptCoordinateSpace::Local => {
                let forward = self.forward_vector();
                let world_up = Vector3::new(0.0, 1.0, 0.0);
                let mut right = forward.cross(&world_up);
                let right_len = right.norm();
                if right_len <= 1e-5 {
                    right = Vector3::new(1.0, 0.0, 0.0);
                } else {
                    right /= right_len;
                }
                let up = right.cross(&forward).normalize();
                self.position += right * delta.x + up * delta.y + forward * delta.z;
            }
        }
    }

    fn apply_script_rotate_delta_deg(&mut self, delta_deg: [f32; 2]) {
        self.yaw_rad += delta_deg[0].to_radians();
        self.pitch_rad = (self.pitch_rad + delta_deg[1].to_radians()).clamp(-1.553_343, 1.553_343);
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
    mobile_path: bool,
) -> f32 {
    let fps_limit = fps_limit_hint.clamp(24.0, 1_200.0);
    let fps = fps_estimate.clamp(20.0, fps_limit * 2.0);
    let fps_ratio = (fps / fps_limit).clamp(0.35, 1.25);
    let mut multiplier = match (profile, mobile_path, parallel_ready) {
        (TickProfile::Balanced, true, true) => 2.0,
        (TickProfile::Balanced, true, false) => 1.7,
        (TickProfile::Balanced, false, true) => 2.9,
        (TickProfile::Balanced, false, false) => 2.3,
        (TickProfile::Max, true, true) => 2.4,
        (TickProfile::Max, true, false) => 2.0,
        (TickProfile::Max, false, true) => 3.4,
        (TickProfile::Max, false, false) => 2.9,
    };
    if fps_ratio > 0.95 {
        multiplier += if mobile_path { 0.35 } else { 1.0 };
    }
    if fps_ratio > 1.05 {
        multiplier += if mobile_path { 0.15 } else { 0.45 };
    }
    if live_balls > 3_500 {
        multiplier += if mobile_path { 0.12 } else { 0.2 };
    }
    if live_balls > 5_500 {
        multiplier += if mobile_path { 0.16 } else { 0.25 };
    }
    if live_balls > 8_000 {
        multiplier += if mobile_path { 0.18 } else { 0.25 };
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
            TickProfile::Balanced => {
                if mobile_path {
                    0.72
                } else {
                    0.85
                }
            }
            TickProfile::Max => {
                if mobile_path {
                    0.82
                } else {
                    0.93
                }
            }
        };
    }
    target_hz *= match profile {
        TickProfile::Balanced => {
            if mobile_path {
                tick_scale.clamp(0.42, 1.05)
            } else {
                tick_scale.clamp(0.50, 1.20)
            }
        }
        TickProfile::Max => {
            if mobile_path {
                tick_scale.clamp(0.60, 1.10)
            } else {
                tick_scale.clamp(0.85, 1.35)
            }
        }
    };

    let dynamic_min = match profile {
        TickProfile::Balanced => {
            if mobile_path {
                policy.min_tick_hz.max(24.0)
            } else {
                (fps_limit * 0.5).max(policy.min_tick_hz.max(30.0))
            }
        }
        TickProfile::Max => {
            if mobile_path {
                policy.min_tick_hz.max(28.0)
            } else {
                policy.min_tick_hz.max(45.0)
            }
        }
    };
    let thread_scale = if mobile_path {
        (logical_threads as f32 / 6.0).clamp(0.70, 2.20)
    } else {
        (logical_threads as f32 / 8.0).clamp(0.75, 4.0)
    };
    let ceiling_factor = match profile {
        TickProfile::Balanced => {
            if mobile_path {
                if parallel_ready {
                    3.2
                } else {
                    2.6
                }
            } else if parallel_ready {
                8.0
            } else {
                6.0
            }
        }
        TickProfile::Max => {
            if mobile_path {
                if parallel_ready {
                    4.0
                } else {
                    3.2
                }
            } else if parallel_ready {
                12.0
            } else {
                10.0
            }
        }
    };
    let thread_gain = match profile {
        TickProfile::Balanced => {
            if mobile_path {
                0.72 + thread_scale * 0.18
            } else {
                0.8 + thread_scale * 0.4
            }
        }
        TickProfile::Max => {
            if mobile_path {
                0.82 + thread_scale * 0.28
            } else {
                0.9 + thread_scale * 0.55
            }
        }
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
    mobile_path: bool,
    framebuffer_fill_ema: f32,
    render_distance: Option<f32>,
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
    if framebuffer_fill_ema > 0.95 {
        pressure += 1;
    }
    if framebuffer_fill_ema > 1.25 {
        pressure += 1;
    }
    if framebuffer_fill_ema > 1.65 {
        pressure += 1;
    }
    if mobile_path {
        // Mobile scheduler should shed load earlier to avoid whole-system chopping.
        if fps < 62.0 {
            pressure += 1;
        }
        if frame_time_ms > 16.8 {
            pressure += 1;
        }
        if last_substeps >= max_substeps.saturating_sub(3) {
            pressure += 1;
        }
        if render_distance.is_none() {
            pressure += 1;
        }
    }

    let thread_scale = if mobile_path {
        (logical_threads as f32 / 6.0).clamp(0.70, 2.20)
    } else {
        (logical_threads as f32 / 8.0).clamp(0.75, 4.0)
    };
    let cap = |base: usize| ((base as f32) * thread_scale).round() as usize;
    let heavy_fill = framebuffer_fill_ema > 1.20;
    let severe_fill = framebuffer_fill_ema > 1.55;

    match pressure {
        0..=2 => RuntimeLoadPlan {
            visible_ball_limit: if mobile_path && heavy_fill && live_balls > 3_600 {
                Some(cap(2_600))
            } else {
                None
            },
            live_ball_budget: if mobile_path { Some(cap(4_800)) } else { None },
            spawn_per_tick_cap: cap(if mobile_path { 96 } else { 420 }),
            max_substeps: if mobile_path { 6 } else { 14 },
            force_low_poly_ball_mesh: false,
            tick_scale: if mobile_path { 0.78 } else { 1.00 },
        },
        3..=4 => RuntimeLoadPlan {
            visible_ball_limit: if mobile_path && heavy_fill && live_balls > 3_200 {
                Some(cap(2_300))
            } else {
                None
            },
            live_ball_budget: if mobile_path { Some(cap(4_000)) } else { None },
            spawn_per_tick_cap: cap(if mobile_path { 72 } else { 360 }),
            max_substeps: if mobile_path { 5 } else { 12 },
            force_low_poly_ball_mesh: false,
            tick_scale: if mobile_path { 0.66 } else { 0.96 },
        },
        5..=6 => RuntimeLoadPlan {
            visible_ball_limit: if mobile_path && (heavy_fill || live_balls > 4_200) {
                Some(cap(2_000))
            } else {
                None
            },
            live_ball_budget: if mobile_path { Some(cap(3_400)) } else { None },
            spawn_per_tick_cap: cap(if mobile_path { 48 } else { 260 }),
            max_substeps: if mobile_path { 4 } else { 10 },
            force_low_poly_ball_mesh: mobile_path && heavy_fill && live_balls > 2_600,
            tick_scale: if mobile_path { 0.56 } else { 0.82 },
        },
        7..=8 => RuntimeLoadPlan {
            visible_ball_limit: if mobile_path && (heavy_fill || live_balls > 3_600) {
                Some(cap(1_700))
            } else {
                None
            },
            live_ball_budget: if mobile_path { Some(cap(2_900)) } else { None },
            spawn_per_tick_cap: cap(if mobile_path { 32 } else { 200 }),
            max_substeps: if mobile_path { 3 } else { 10 },
            force_low_poly_ball_mesh: mobile_path && (heavy_fill || live_balls > 3_200),
            tick_scale: if mobile_path { 0.46 } else { 0.72 },
        },
        _ => RuntimeLoadPlan {
            visible_ball_limit: if mobile_path { Some(cap(1_300)) } else { None },
            live_ball_budget: if mobile_path { Some(cap(2_300)) } else { None },
            spawn_per_tick_cap: cap(if mobile_path { 24 } else { 160 }),
            max_substeps: if mobile_path { 2 } else { 10 },
            force_low_poly_ball_mesh: mobile_path && (severe_fill || live_balls > 3_600),
            tick_scale: if mobile_path { 0.38 } else { 0.60 },
        },
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct RenderDistanceStats {
    culled: usize,
    blurred: usize,
}

fn count_rendered_balls(frame: &SceneFrameInstances) -> usize {
    frame
        .opaque_3d
        .iter()
        .chain(frame.transparent_3d.iter())
        .filter(|instance| !matches!(instance.primitive, ScenePrimitive3d::Box))
        .count()
}

fn estimate_framebuffer_fill_ratio(frame: &SceneFrameInstances, width: u32, height: u32) -> f32 {
    let pixels = (width.max(1) as f32) * (height.max(1) as f32);
    let opaque = frame.opaque_3d.len() as f32;
    let transparent = frame.transparent_3d.len() as f32;
    let sprites = frame.sprites.len() as f32;
    // Heuristic fill estimate: transparent passes are weighted higher due alpha blending and
    // overdraw on tiled mobile GPUs.
    let estimated_fragments = opaque * 560.0 + transparent * 1_140.0 + sprites * 180.0;
    (estimated_fragments / pixels).clamp(0.0, 8.0)
}

fn apply_render_distance_haze(
    frame: &mut SceneFrameInstances,
    camera_eye: [f32; 3],
    render_distance: Option<f32>,
    blur_enabled: bool,
) -> RenderDistanceStats {
    let Some(max_distance) = render_distance.filter(|distance| *distance > 0.25) else {
        return RenderDistanceStats::default();
    };
    let blur_start_ratio = 0.80;
    let blur_start = (max_distance * blur_start_ratio).max(0.25);
    // Keep a large soft margin after render distance to avoid sudden popping/flicker.
    let hard_cull_distance = max_distance * if blur_enabled { 1.42 } else { 1.24 };
    let max_distance_sq = max_distance * max_distance;
    let hard_cull_distance_sq = hard_cull_distance * hard_cull_distance;
    let blur_start_sq = blur_start * blur_start;
    let total_candidates = frame
        .opaque_3d
        .iter()
        .filter(|instance| !matches!(instance.primitive, ScenePrimitive3d::Box))
        .count();
    let min_keep = ((total_candidates as f32) * 0.18)
        .round()
        .clamp(120.0, 640.0) as usize;

    let mut stats = RenderDistanceStats::default();
    let mut next_opaque = Vec::with_capacity(frame.opaque_3d.len());
    let mut blurred = Vec::<(f32, crate::scene::SceneInstance3d)>::new();
    let mut hard_culled = Vec::<(f32, crate::scene::SceneInstance3d)>::new();

    for mut instance in frame.opaque_3d.drain(..) {
        if matches!(instance.primitive, ScenePrimitive3d::Box) {
            next_opaque.push(instance);
            continue;
        }
        let p = instance.transform.translation;
        let dx = p[0] - camera_eye[0];
        let dy = p[1] - camera_eye[1];
        let dz = p[2] - camera_eye[2];
        let distance_sq = dx * dx + dy * dy + dz * dz;

        if distance_sq > hard_cull_distance_sq {
            hard_culled.push((distance_sq, instance));
            continue;
        }
        if blur_enabled && distance_sq > blur_start_sq {
            let distance = distance_sq.sqrt();
            let t = ((distance - blur_start) / (hard_cull_distance - blur_start).max(1e-4))
                .clamp(0.0, 1.0);
            soften_instance_for_distance_haze(&mut instance, t);
            blurred.push((distance_sq, instance));
            stats.blurred = stats.blurred.saturating_add(1);
        } else if distance_sq > max_distance_sq {
            let distance = distance_sq.sqrt();
            let t = ((distance - max_distance) / (hard_cull_distance - max_distance).max(1e-4))
                .clamp(0.0, 1.0);
            soften_instance_for_distance_haze(&mut instance, t * 0.80 + 0.20);
            blurred.push((distance_sq, instance));
            stats.blurred = stats.blurred.saturating_add(1);
        } else {
            next_opaque.push(instance);
        }
    }

    let rendered_count = next_opaque.len() + blurred.len();
    if rendered_count < min_keep && !hard_culled.is_empty() {
        hard_culled.sort_by(|left, right| left.0.total_cmp(&right.0));
        let rescue_count = (min_keep - rendered_count).min(hard_culled.len());
        for (distance_sq, mut instance) in hard_culled.drain(..rescue_count) {
            soften_instance_for_distance_haze(&mut instance, 0.96);
            blurred.push((distance_sq, instance));
            stats.blurred = stats.blurred.saturating_add(1);
        }
    }

    stats.culled = stats.culled.saturating_add(hard_culled.len());
    // Draw farther blurred objects first to reduce alpha-overdraw artifacts.
    blurred.sort_by(|left, right| right.0.total_cmp(&left.0));

    frame.opaque_3d = next_opaque;
    frame
        .transparent_3d
        .extend(blurred.into_iter().map(|(_, instance)| instance));
    stats
}

fn soften_instance_for_distance_haze(instance: &mut crate::scene::SceneInstance3d, t: f32) {
    let t = t.clamp(0.0, 1.0);
    let haze_rgb = [0.58, 0.64, 0.72];
    for (index, haze) in haze_rgb.into_iter().enumerate() {
        let current = instance.material.base_color_rgba[index];
        instance.material.base_color_rgba[index] = current + (haze - current) * (t * 0.72);
    }
    let alpha = instance.material.base_color_rgba[3];
    instance.material.base_color_rgba[3] = (alpha * (1.0 - t * 0.78)).clamp(0.10, 1.0);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn make_adapter_info(
        name: &str,
        backend: wgpu::Backend,
        device_type: wgpu::DeviceType,
    ) -> wgpu::AdapterInfo {
        wgpu::AdapterInfo {
            name: name.to_string(),
            vendor: 0,
            device: 0,
            device_type,
            device_pci_bus_id: String::new(),
            driver: String::new(),
            driver_info: String::new(),
            backend,
            subgroup_min_size: 1,
            subgroup_max_size: 1,
            transient_saves_memory: false,
        }
    }

    #[test]
    fn android_auto_scheduler_prefers_mgs() {
        let info = make_adapter_info(
            "NVIDIA GeForce RTX 5060 Ti",
            wgpu::Backend::Vulkan,
            wgpu::DeviceType::DiscreteGpu,
        );
        let resolved = resolve_project_scheduler(
            TlpfileGraphicsScheduler::Auto,
            RuntimePlatform::Android,
            &info,
        )
        .expect("auto scheduler should resolve");
        assert_eq!(resolved.selected, GraphicsSchedulerPath::Mgs);
    }

    #[test]
    fn android_explicit_gms_requires_supported_backend() {
        let info = make_adapter_info(
            "Mali-G610",
            wgpu::Backend::Gl,
            wgpu::DeviceType::IntegratedGpu,
        );
        let err = resolve_project_scheduler(
            TlpfileGraphicsScheduler::Gms,
            RuntimePlatform::Android,
            &info,
        )
        .expect_err("explicit gms should be rejected on unsupported backend");
        assert!(err.contains("unsupported"));
    }

    #[test]
    fn android_explicit_mgs_is_accepted() {
        let info = make_adapter_info(
            "Adreno 740",
            wgpu::Backend::Vulkan,
            wgpu::DeviceType::IntegratedGpu,
        );
        let resolved = resolve_project_scheduler(
            TlpfileGraphicsScheduler::Mgs,
            RuntimePlatform::Android,
            &info,
        )
        .expect("mgs scheduler should resolve");
        assert_eq!(resolved.selected, GraphicsSchedulerPath::Mgs);
    }

    #[test]
    fn merged_axis_inputs_are_clamped() {
        assert_eq!(merge_axis_inputs(0.9, 0.8), 1.0);
        assert_eq!(merge_axis_inputs(-0.9, -0.8), -1.0);
        assert_eq!(merge_axis_inputs(0.2, -0.1), 0.1);
    }

    #[test]
    fn vulkan_fail_soft_message_contains_backend_diagnostics() {
        let message = build_vulkan_unavailable_message(
            "vk init failed",
            Some(("FallbackGPU", wgpu::Backend::Gl)),
        );
        assert!(message.contains("Vulkan unavailable on Android"));
        assert!(message.contains("FallbackGPU"));
        assert!(message.contains("backend=Gl"));
    }
}
