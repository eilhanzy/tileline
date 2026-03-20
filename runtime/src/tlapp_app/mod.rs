use std::env;
use std::error::Error;
use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::Once;
use std::time::{Duration, Instant};

pub mod cli;
pub mod console;
pub mod camera;
pub mod fps;

mod runtime_init;
mod runtime_render;
mod runtime_input;
mod runtime_console;
mod runtime_scene;

pub use self::cli::{
    CliOptions, VsyncMode, TickProfile, ToggleAuto, SchedulerResolution,
    parse_fps_cap, parse_render_distance, parse_fsr_sharpness, parse_fsr_scale, parse_msaa,
};
pub use self::console::{
    RuntimeConsoleState, ConsoleUiLayout, RuntimeConsoleLogLine, RuntimeConsoleLogLevel,
    RuntimeConsoleLogFilter, RuntimeConsoleEditTarget, RuntimeConsoleTailFollow,
    RuntimeConsoleFileWatch, empty_showcase_output,
};
pub use self::camera::{FreeCameraController, CameraInputState, GamepadManager};
pub use self::fps::{FpsTracker, RenderDistanceStats};


use crate::{
    app_runner, apply_scene_light_overrides, choose_scheduler_path_for_platform,
    clamp_scene_lights_for_camera, compile_tljoint_scene_from_path,
    compile_tlpfile_scene_from_path, compile_tlscript_showcase, resolve_tileline_version_query,
    tileline_version_entries, BounceTankRuntimePatch, BounceTankSceneConfig,
    BounceTankSceneController, BounceTankTickMetrics, DrawPathCompiler, FsrConfig, FsrDynamoConfig,
    FsrMode, FsrQualityPreset, GraphicsSchedulerPath, RayTracingMode, RenderSyncMode,
    RuntimePlatform,
    SceneFrameInstances, ScenePrimitive3d, SpriteInstance, SpriteKind, TelemetryHudComposer,
    TelemetryHudSample, TickRatePolicy, TljointDiagnosticLevel, TljointSceneBundle,
    TlpfileDiagnosticLevel, TlpfileGraphicsScheduler,
    TlscriptShowcaseConfig, TlscriptShowcaseControlInput, TlscriptShowcaseFrameInput,
    TlscriptShowcaseFrameOutput, TlscriptShowcaseProgram, TlspriteHotReloadEvent, TlspriteProgram,
    TlspriteProgramCache, TlspriteWatchReloader, WgpuSceneRenderer, ENGINE_ID, ENGINE_VERSION,
    MAX_SCENE_LIGHTS,
};
use gms::safe_default_required_limits_for_adapter;
use mgs::MobileGpuProfile;
use crate::physics_mps_runner::{PhysicsMpsRunner, PhysicsStepToken};
use nalgebra::Vector3;
use paradoxpe::{
    BroadphaseConfig, ContactSolverConfig, NarrowphaseConfig, PhysicsWorld, PhysicsWorldConfig,
};
use regex::RegexBuilder;
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalSize};
use winit::event::{
    DeviceEvent, ElementState, KeyEvent, MouseButton, MouseScrollDelta, Touch, TouchPhase,
    WindowEvent,
};
use winit::event_loop::{ActiveEventLoop, ControlFlow};
use winit::keyboard::{Key, KeyCode, ModifiersState, NamedKey, PhysicalKey};
#[cfg(target_os = "android")]
use winit::platform::android::activity::AndroidApp;
use winit::window::{Fullscreen, Window, WindowAttributes, WindowId};

const AUTO_LOW_POLY_BALL_SLOT: u8 = 250;
const DEFAULT_FBX_BALL_SLOT: u8 = 2;
const CONSOLE_SCRIPT_INDEX: usize = 9_999;
const CONSOLE_MAX_LOG_LINES: usize = 320;
const CONSOLE_ERROR_BLINK_MS: u128 = 320;
const CONSOLE_WHEEL_PIXELS_PER_LINE: f64 = 24.0;
const CONSOLE_TEXT_SLOT_BASE: u16 = 128;
const CONSOLE_FILE_HEAD_DEFAULT_LINES: usize = 32;
const CONSOLE_FILE_HEAD_MAX_LINES: usize = 256;
const CONSOLE_FILE_LIST_DEFAULT_LIMIT: usize = 48;
const CONSOLE_FILE_LIST_MAX_LIMIT: usize = 256;
const CONSOLE_FILE_READ_DEFAULT_BYTES: usize = 4 * 1024;
const CONSOLE_FILE_READ_MAX_BYTES: usize = 128 * 1024;
const CONSOLE_FILE_TAIL_DEFAULT_LINES: usize = 40;
const CONSOLE_FILE_TAIL_MAX_LINES: usize = 300;
const CONSOLE_FILE_TAIL_DEFAULT_WINDOW_BYTES: usize = 64 * 1024;
const CONSOLE_FILE_TAIL_MAX_WINDOW_BYTES: usize = 512 * 1024;
const CONSOLE_FILE_FIND_DEFAULT_MATCHES: usize = 32;
const CONSOLE_FILE_FIND_MAX_MATCHES: usize = 256;
const CONSOLE_FILE_FIND_DEFAULT_BYTES: usize = 128 * 1024;
const CONSOLE_FILE_FIND_MAX_BYTES: usize = 512 * 1024;
const CONSOLE_FILE_HEAD_MAX_WINDOW_BYTES: usize = 256 * 1024;
const CONSOLE_FILE_TAILF_DEFAULT_POLL_MS: u64 = 220;
const CONSOLE_FILE_TAILF_MAX_LINES_PER_POLL: usize = 64;
const CONSOLE_FILE_WATCH_DEFAULT_POLL_MS: u64 = 350;
const CONSOLE_FILE_GREP_DEFAULT_CONTEXT: usize = 0;
const CONSOLE_FILE_GREP_MAX_CONTEXT: usize = 8;
const CONSOLE_HELP_COMMANDS: &[&str] = &[
    "help",
    "help <file|gfx|sim|script|cam|log>",
    "version [module|all]",
    "status | gfx.status",
    "sim.status | sim.pause | sim.resume | sim.step <n> | sim.reset",
    "scene.reload | sprite.reload | script.reload",
    "perf.snapshot",
    "phys.gravity <x y z>",
    "phys.substeps <auto|n>",
    "cam.speed <v>",
    "cam.sens <v>",
    "cam.reset",
    "log.clear",
    "log.tail <off|n>",
    "log.level <all|info|error>",
    "file.exists <path>",
    "file.head <path> [lines]",
    "file.tail <path> [lines] [max_bytes]",
    "file.find <path> <pattern> [max_matches] [max_bytes]",
    "file.findi <path> <pattern> [max_matches] [max_bytes]",
    "file.findr <path> <regex> [max_matches] [max_bytes]",
    "file.grep <path> <pattern> [context] [max_matches] [max_bytes]",
    "file.tailf <path>|stop [poll_ms] [max_lines]",
    "file.watch <path>|stop [poll_ms]",
    "file.read <path> [max_bytes]",
    "file.list <dir> [limit]",
    "clear | exit",
    "gfx.vsync <auto|on|off>",
    "gfx.fps_cap <off|N>",
    "gfx.rt <off|auto|on>",
    "gfx.fsr <off|auto|on>",
    "gfx.fsr_quality <native|ultra|quality|balanced|performance>",
    "gfx.fsr_sharpness <0..1>",
    "gfx.fsr_scale <auto|0.5..1>",
    "gfx.profile <low|med|high|ultra>",
    "gfx.render_distance <off|N>",
    "gfx.adaptive_distance <auto|on|off>",
    "gfx.distance_blur <auto|on|off>",
    "script.var <name> <expr>",
    "script.unset <name>",
    "script.vars",
    "script.call <fn(args)>",
    "script.exec <stmt>",
    "script.uncall <idx|all>",
    "script.list | script.clear",
];
#[cfg(feature = "gamepad")]
const GAMEPAD_DEADZONE: f32 = 0.16;
const GAMEPAD_LOOK_SPEED_RAD: f32 = 2.6;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RuntimeCommand {
    None,
    Consumed,
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




fn is_valid_tlscript_ident(name: &str) -> bool {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !(first.is_ascii_alphabetic() || first == '_') {
        return false;
    }
    chars.all(|ch| ch.is_ascii_alphanumeric() || ch == '_')
}








fn parse_console_f32_in_range(value: &str, label: &str, min: f32, max: f32) -> Result<f32, String> {
    let parsed = value
        .parse::<f32>()
        .map_err(|_| format!("invalid {label}: {value}"))?;
    if !parsed.is_finite() || parsed < min || parsed > max {
        return Err(format!("{label} must be in [{min}..{max}]"));
    }
    Ok(parsed)
}

fn parse_console_u32_in_range(value: &str, label: &str, min: u32, max: u32) -> Result<u32, String> {
    let parsed = value
        .parse::<u32>()
        .map_err(|_| format!("invalid {label}: {value}"))?;
    if parsed < min || parsed > max {
        return Err(format!("{label} must be in [{min}..{max}]"));
    }
    Ok(parsed)
}

fn parse_console_usize_in_range(
    value: &str,
    label: &str,
    min: usize,
    max: usize,
) -> Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|_| format!("invalid {label}: {value}"))?;
    if parsed < min || parsed > max {
        return Err(format!("{label} must be in [{min}..{max}]"));
    }
    Ok(parsed)
}

fn bootstrap_uncapped_fps_hint(logical_threads: usize) -> f32 {
    // Start from a conservative-but-fast target and let runtime sampling retune to hardware limit.
    (170.0 + logical_threads as f32 * 7.5).clamp(170.0, 420.0)
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

    fn shutdown_now(&mut self, event_loop: &ActiveEventLoop) {
        self.exit_requested = true;
        if let Some(runtime_state) = self.runtime.as_mut() {
            runtime_state.prepare_for_exit();
        }
        // Drop runtime resources before exiting the event loop. This helps avoid
        // backend teardown races on some Vulkan drivers.
        self.runtime.take();
        event_loop.exit();
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

    fn prepare_for_exit(&mut self) {
        if let Self::Ready(runtime) = self {
            runtime.prepare_for_exit();
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

        if let TlAppRuntimeState::Ready(runtime) = runtime_state {
            if runtime.handle_console_window_event(&event) {
                return;
            }
        }

        let mut request_shutdown = false;
        match event {
            WindowEvent::CloseRequested => {
                request_shutdown = true;
            }
            WindowEvent::KeyboardInput { event, .. } => match runtime_state {
                TlAppRuntimeState::Ready(runtime) => match runtime.on_keyboard_input(&event) {
                    RuntimeCommand::Exit => {
                        request_shutdown = true;
                    }
                    RuntimeCommand::Consumed => return,
                    RuntimeCommand::None => {}
                },
                TlAppRuntimeState::FailSoft(runtime) => {
                    if runtime.on_keyboard_input(&event) {
                        request_shutdown = true;
                    }
                }
            },
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
            WindowEvent::CursorMoved { position, .. } => {
                if let TlAppRuntimeState::Ready(runtime) = runtime_state {
                    runtime.on_cursor_moved(position.x as f32, position.y as f32);
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
                        request_shutdown = true;
                    }
                }
                TlAppRuntimeState::FailSoft(runtime) => {
                    runtime.render_frame();
                }
            },
            _ => {}
        }

        if request_shutdown {
            self.shutdown_now(event_loop);
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
            self.shutdown_now(event_loop);
            return;
        }

        if let Some(runtime) = self.runtime.as_mut() {
            runtime.schedule_next_redraw(event_loop);
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(runtime_state) = self.runtime.as_mut() {
            runtime_state.prepare_for_exit();
        }
        self.runtime.take();
    }
}

struct TlAppRuntime {
    cli_options: CliOptions,
    file_io_root: PathBuf,
    window: Arc<Window>,
    _instance: wgpu::Instance,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: PhysicalSize<u32>,
    world: PhysicsMpsRunner,
    /// Pending async physics step submitted before last GPU upload.
    /// Drained at the start of the next frame before building instances.
    physics_token: Option<PhysicsStepToken>,
    /// Metrics from the most recent physics_tick() call (1-frame pipeline lag).
    last_tick: BounceTankTickMetrics,
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
    script_key_g_keyboard: bool,
    console: RuntimeConsoleState,
    console_overlay_sprites: Vec<SpriteInstance>,
    keyboard_camera: CameraInputState,
    mouse_look_held: bool,
    look_lock_active: bool,
    mouse_look_delta: (f32, f32),
    camera_reset_requested: bool,
    keyboard_modifiers: ModifiersState,
    gamepad: GamepadManager,
    touch_look_id: Option<u64>,
    touch_last_position: Option<(f32, f32)>,
    cursor_position: Option<(f32, f32)>,
    tick_policy: TickRatePolicy,
    tick_profile: TickProfile,
    tick_cap: Option<f32>,
    tick_hz: f32,
    fps_limit_hint: f32,
    uncapped_dynamic_fps_hint: bool,
    adaptive_pacer_enabled: bool,
    adaptive_pacer_fps: f32,
    adaptive_pacer_timer: f32,
    mps_logical_threads: usize,
    max_substeps: u32,
    last_substeps: u32,
    manual_max_substeps: Option<u32>,
    simulation_paused: bool,
    simulation_step_budget: u32,
    tick_retune_timer: f32,
    adaptive_ball_render_limit: Option<usize>,
    adaptive_live_ball_budget: Option<usize>,
    adaptive_low_poly_override: bool,
    // True only when MGS path was selected AND the adapter is genuine mobile hardware.
    // Desktop-class adapters (Apple M-series, discrete GPUs) keep this false even when the
    // MGS path is active via TILELINE_SCHEDULER override.
    mgs_is_mobile_hardware: bool,
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
    rt_mode: RayTracingMode,
    fsr_config: FsrConfig,
    /// Dynamo FSR distance thresholds and smoothing parameters.
    fsr_dynamo_config: FsrDynamoConfig,
    /// Current smoothed render scale maintained by Dynamo FSR (starts at 1.0 = native).
    fsr_dynamo_scale: f32,
    shutdown_prepared: bool,
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
        matches!(event.physical_key, PhysicalKey::Code(KeyCode::KeyQ))
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














fn merge_showcase_output(
    merged: &mut TlscriptShowcaseFrameOutput,
    mut next: TlscriptShowcaseFrameOutput,
    script_index: usize,
) {
    merge_runtime_patch(&mut merged.patch, next.patch);
    merge_light_overrides(&mut merged.light_overrides, &next.light_overrides);
    if next.rt_mode.is_some() {
        merged.rt_mode = next.rt_mode;
    }
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

fn merge_light_overrides(
    target: &mut Vec<crate::scene::SceneLightOverride>,
    incoming: &[crate::scene::SceneLightOverride],
) {
    for entry in incoming {
        if let Some(existing) = target.iter_mut().find(|cur| cur.id == entry.id) {
            if entry.enabled.is_some() {
                existing.enabled = entry.enabled;
            }
            if entry.position.is_some() {
                existing.position = entry.position;
            }
            if entry.direction.is_some() {
                existing.direction = entry.direction;
            }
            if entry.color.is_some() {
                existing.color = entry.color;
            }
            if entry.intensity.is_some() {
                existing.intensity = entry.intensity;
            }
            if entry.range.is_some() {
                existing.range = entry.range;
            }
            if entry.inner_cone_deg.is_some() {
                existing.inner_cone_deg = entry.inner_cone_deg;
            }
            if entry.outer_cone_deg.is_some() {
                existing.outer_cone_deg = entry.outer_cone_deg;
            }
            if entry.softness.is_some() {
                existing.softness = entry.softness;
            }
            continue;
        }
        target.push(entry.clone());
    }
    target.sort_by_key(|entry| entry.id);
}

fn merge_runtime_patch(target: &mut BounceTankRuntimePatch, patch: BounceTankRuntimePatch) {
    if patch.target_ball_count.is_some() {
        target.target_ball_count = patch.target_ball_count;
    }
    if patch.spawn_per_tick.is_some() {
        target.spawn_per_tick = patch.spawn_per_tick;
    }
    if patch.container_half_extents.is_some() {
        target.container_half_extents = patch.container_half_extents;
    }
    if patch.wall_thickness.is_some() {
        target.wall_thickness = patch.wall_thickness;
    }
    if patch.container_mesh_scale.is_some() {
        target.container_mesh_scale = patch.container_mesh_scale;
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
    if patch.ball_friction.is_some() {
        target.ball_friction = patch.ball_friction;
    }
    if patch.wall_restitution.is_some() {
        target.wall_restitution = patch.wall_restitution;
    }
    if patch.wall_friction.is_some() {
        target.wall_friction = patch.wall_friction;
    }
    if patch.friction_transition_speed.is_some() {
        target.friction_transition_speed = patch.friction_transition_speed;
    }
    if patch.friction_static_boost.is_some() {
        target.friction_static_boost = patch.friction_static_boost;
    }
    if patch.friction_kinetic_scale.is_some() {
        target.friction_kinetic_scale = patch.friction_kinetic_scale;
    }
    if patch.restitution_velocity_threshold.is_some() {
        target.restitution_velocity_threshold = patch.restitution_velocity_threshold;
    }
    if patch.levitation_height.is_some() {
        target.levitation_height = patch.levitation_height;
    }
    if patch.levitation_strength.is_some() {
        target.levitation_strength = patch.levitation_strength;
    }
    if patch.levitation_damping.is_some() {
        target.levitation_damping = patch.levitation_damping;
    }
    if patch.levitation_max_vertical_speed.is_some() {
        target.levitation_max_vertical_speed = patch.levitation_max_vertical_speed;
    }
    if patch.levitation_reaction_strength.is_some() {
        target.levitation_reaction_strength = patch.levitation_reaction_strength;
    }
    if patch.levitation_reaction_radius.is_some() {
        target.levitation_reaction_radius = patch.levitation_reaction_radius;
    }
    if patch.levitation_reaction_damping.is_some() {
        target.levitation_reaction_damping = patch.levitation_reaction_damping;
    }
    if patch.levitation_lateral_strength.is_some() {
        target.levitation_lateral_strength = patch.levitation_lateral_strength;
    }
    if patch.levitation_lateral_damping.is_some() {
        target.levitation_lateral_damping = patch.levitation_lateral_damping;
    }
    if patch.levitation_lateral_max_horizontal_speed.is_some() {
        target.levitation_lateral_max_horizontal_speed =
            patch.levitation_lateral_max_horizontal_speed;
    }
    if patch.levitation_lateral_wall_push.is_some() {
        target.levitation_lateral_wall_push = patch.levitation_lateral_wall_push;
    }
    if patch.levitation_lateral_frequency.is_some() {
        target.levitation_lateral_frequency = patch.levitation_lateral_frequency;
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
    if patch.virtual_barrier_enabled.is_some() {
        target.virtual_barrier_enabled = patch.virtual_barrier_enabled;
    }
    if patch.speculative_sweep_enabled.is_some() {
        target.speculative_sweep_enabled = patch.speculative_sweep_enabled;
    }
    if patch.speculative_sweep_max_distance.is_some() {
        target.speculative_sweep_max_distance = patch.speculative_sweep_max_distance;
    }
    if patch.speculative_contacts_enabled.is_some() {
        target.speculative_contacts_enabled = patch.speculative_contacts_enabled;
    }
    if patch.speculative_contact_distance.is_some() {
        target.speculative_contact_distance = patch.speculative_contact_distance;
    }
    if patch.speculative_max_prediction_distance.is_some() {
        target.speculative_max_prediction_distance = patch.speculative_max_prediction_distance;
    }
    if patch.ball_mesh_slot.is_some() {
        target.ball_mesh_slot = patch.ball_mesh_slot;
    }
    if patch.container_mesh_slot.is_some() {
        target.container_mesh_slot = patch.container_mesh_slot;
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

impl Drop for TlAppRuntime {
    fn drop(&mut self) {
        self.prepare_for_exit();
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
fn scheduler_path_label(path: GraphicsSchedulerPath) -> &'static str {
    match path {
        GraphicsSchedulerPath::Gms => "gms",
        GraphicsSchedulerPath::Mgs => "mgs",
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

