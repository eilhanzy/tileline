use std::collections::BTreeMap;
use std::env;
use std::error::Error;
use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::Once;
use std::time::{Duration, Instant};

use crate::{
    app_runner, apply_scene_light_overrides, choose_scheduler_path_for_platform,
    clamp_scene_lights_for_camera, compile_tljoint_scene_from_path,
    compile_tlpfile_scene_from_path, compile_tlscript_showcase, resolve_tileline_version_query,
    tileline_version_entries, BounceTankRuntimePatch, BounceTankSceneConfig,
    BounceTankSceneController, BounceTankTickMetrics, DrawPathCompiler, FsrConfig, FsrMode,
    FsrQualityPreset, GraphicsSchedulerPath, RayTracingMode, RenderSyncMode, RuntimePlatform,
    SceneFrameInstances, ScenePrimitive3d, SpriteInstance, SpriteKind, TelemetryHudComposer,
    TelemetryHudSample, TickRatePolicy, TljointDiagnosticLevel, TljointSceneBundle,
    TlpfileDiagnosticLevel, TlpfileGraphicsScheduler, TlscriptCoordinateSpace,
    TlscriptShowcaseConfig, TlscriptShowcaseControlInput, TlscriptShowcaseFrameInput,
    TlscriptShowcaseFrameOutput, TlscriptShowcaseProgram, TlspriteHotReloadEvent, TlspriteProgram,
    TlspriteProgramCache, TlspriteWatchReloader, WgpuSceneRenderer, ENGINE_ID, ENGINE_VERSION,
    MAX_SCENE_LIGHTS,
};
use gms::safe_default_required_limits_for_adapter;
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
use winit::window::{CursorGrabMode, Fullscreen, Window, WindowAttributes, WindowId};

#[cfg(feature = "gamepad")]
use gilrs::{Axis, Button, EventType, GamepadId, Gilrs};

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

#[derive(Debug, Clone)]
struct CliOptions {
    resolution: PhysicalSize<u32>,
    vsync: VsyncMode,
    fps_cap: Option<f32>,
    tick_profile: TickProfile,
    render_distance: Option<f32>,
    adaptive_distance: ToggleAuto,
    distance_blur: ToggleAuto,
    fsr_mode: FsrMode,
    fsr_quality: FsrQualityPreset,
    fsr_sharpness: f32,
    fsr_scale_override: Option<f32>,
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
            fsr_mode: FsrMode::Auto,
            fsr_quality: FsrQualityPreset::Quality,
            fsr_sharpness: 0.35,
            fsr_scale_override: None,
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
                "-V" | "--version" => {
                    print_version_overview();
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
                "--fsr" => {
                    let value = next_arg(&mut args, "--fsr")?;
                    options.fsr_mode =
                        FsrMode::parse(&value).ok_or_else(|| -> Box<dyn Error> {
                            "invalid --fsr value (expected off|auto|on|fsr1)".into()
                        })?;
                }
                "--fsr-quality" => {
                    let value = next_arg(&mut args, "--fsr-quality")?;
                    options.fsr_quality = FsrQualityPreset::parse(&value).ok_or_else(
                        || -> Box<dyn Error> {
                            "invalid --fsr-quality value (expected native|ultra|quality|balanced|performance)"
                                .into()
                        },
                    )?;
                }
                "--fsr-sharpness" => {
                    let value = next_arg(&mut args, "--fsr-sharpness")?;
                    options.fsr_sharpness = parse_fsr_sharpness(&value)?;
                }
                "--fsr-scale" => {
                    let value = next_arg(&mut args, "--fsr-scale")?;
                    options.fsr_scale_override = parse_fsr_scale(&value)?;
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
        if !(0.0..=1.0).contains(&options.fsr_sharpness) {
            return Err("--fsr-sharpness must be in 0.0..=1.0".into());
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

fn parse_fsr_sharpness(value: &str) -> Result<f32, Box<dyn Error>> {
    let parsed = value
        .parse::<f32>()
        .map_err(|_| format!("invalid --fsr-sharpness value: {value}"))?;
    if !parsed.is_finite() || !(0.0..=1.0).contains(&parsed) {
        return Err("--fsr-sharpness must be in 0.0..=1.0".into());
    }
    Ok(parsed)
}

fn parse_fsr_scale(value: &str) -> Result<Option<f32>, Box<dyn Error>> {
    if value.eq_ignore_ascii_case("auto") {
        return Ok(None);
    }
    let parsed = value
        .parse::<f32>()
        .map_err(|_| format!("invalid --fsr-scale value: {value} (expected number or auto)"))?;
    if !parsed.is_finite() || !(0.50..=1.0).contains(&parsed) {
        return Err("--fsr-scale must be in 0.50..=1.0 or auto".into());
    }
    Ok(Some(parsed))
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
    println!("  --fsr off|auto|on         FSR policy mode (default: auto)");
    println!("  --fsr-quality <preset>    FSR quality: native|ultra|quality|balanced|performance");
    println!("  --fsr-sharpness <0..1>    FSR sharpen amount (default: 0.35)");
    println!("  --fsr-scale <0.5..1|auto> Render scale override for FSR mode");
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
    println!("  -V, --version             Show engine + module versions");
    println!("  -h, --help                Show help");
    println!();
    println!("Keyboard:");
    println!("  Move: WASD / Arrows | Up: R/E | Down: F/Q/C | Sprint: Shift");
    println!("  Demo action: G (particle/scatter burst)");
    println!("  Look: RMB hold + mouse");
    println!("  Combos: Ctrl+Q exit | Ctrl+F fullscreen | Alt+Enter fullscreen");
    println!("          Ctrl+F1 / F1 / Ctrl+` in-app CLI console");
    println!("          Ctrl+R reset camera | Ctrl+L toggle look lock");
    println!("Gamepad:");
    println!("  Left stick move | Right stick look | D-Pad move");
    println!("  South button mirrors G action | Trigger buttons add vertical move");
}

fn print_version_overview() {
    println!("Tileline Engine: v{ENGINE_VERSION}");
    for entry in tileline_version_entries() {
        println!("  {:<10} v{}", entry.module, entry.version);
    }
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

#[derive(Debug, Clone)]
struct RuntimeConsoleState {
    open: bool,
    input_line: String,
    history: Vec<String>,
    history_cursor: Option<usize>,
    last_feedback: String,
    log_lines: Vec<RuntimeConsoleLogLine>,
    script_vars: BTreeMap<String, String>,
    script_statements: Vec<String>,
    script_overlay: TlscriptShowcaseFrameOutput,
    edit_target: RuntimeConsoleEditTarget,
    quick_fps_cap: String,
    quick_render_distance: String,
    quick_fsr_sharpness: String,
    log_scroll: usize,
    log_filter: RuntimeConsoleLogFilter,
    log_tail_limit: Option<usize>,
    tail_follow: Option<RuntimeConsoleTailFollow>,
    file_watch: Option<RuntimeConsoleFileWatch>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RuntimeConsoleLogLevel {
    Info,
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RuntimeConsoleLogFilter {
    All,
    Info,
    Error,
}

impl RuntimeConsoleLogFilter {
    fn matches(self, level: RuntimeConsoleLogLevel) -> bool {
        match self {
            Self::All => true,
            Self::Info => matches!(level, RuntimeConsoleLogLevel::Info),
            Self::Error => matches!(level, RuntimeConsoleLogLevel::Error),
        }
    }
}

#[derive(Debug, Clone)]
struct RuntimeConsoleLogLine {
    timestamp: Instant,
    level: RuntimeConsoleLogLevel,
    message: String,
}

#[derive(Debug, Clone)]
struct RuntimeConsoleTailFollow {
    path: PathBuf,
    poll_interval: Duration,
    max_lines_per_poll: usize,
    cursor: u64,
    partial_line: String,
    last_poll: Instant,
    last_error: Option<String>,
}

#[derive(Debug, Clone)]
struct RuntimeConsoleFileWatch {
    path: PathBuf,
    poll_interval: Duration,
    last_modified: Option<std::time::SystemTime>,
    last_len: u64,
    last_poll: Instant,
    last_error: Option<String>,
}

#[derive(Debug, Clone, Copy)]
struct ConsoleUiLayout {
    sx: f32,
    sy: f32,
    text_scale: f32,
    command_center: (f32, f32),
    command_size: (f32, f32),
    send_center: (f32, f32),
    send_size: (f32, f32),
    fps_center: (f32, f32),
    fps_size: (f32, f32),
    distance_center: (f32, f32),
    distance_size: (f32, f32),
    sharpness_center: (f32, f32),
    sharpness_size: (f32, f32),
    apply_center: (f32, f32),
    apply_size: (f32, f32),
}

impl ConsoleUiLayout {
    fn from_size(size: PhysicalSize<u32>) -> Self {
        let width = size.width.max(1) as f32;
        let height = size.height.max(1) as f32;
        // Scale with resolution: smaller windows shrink UI, larger windows can scale up.
        let sx = (width / 1280.0).clamp(0.72, 1.34);
        let sy = (height / 720.0).clamp(0.72, 1.34);
        let text_scale = (sx.min(sy) * 0.98).clamp(0.72, 1.34);
        Self {
            sx,
            sy,
            text_scale,
            command_center: (-0.13 * sx, -0.56 * sy),
            command_size: (1.60 * text_scale, 0.12 * text_scale),
            send_center: (0.81 * sx, -0.56 * sy),
            send_size: (0.24 * text_scale, 0.12 * text_scale),
            fps_center: (0.50 * sx, 0.57 * sy),
            fps_size: (0.84 * text_scale, 0.06 * text_scale),
            distance_center: (0.50 * sx, 0.515 * sy),
            distance_size: (0.84 * text_scale, 0.06 * text_scale),
            sharpness_center: (0.50 * sx, 0.46 * sy),
            sharpness_size: (0.84 * text_scale, 0.06 * text_scale),
            apply_center: (0.52 * sx, 0.32 * sy),
            apply_size: (0.84 * text_scale, 0.06 * text_scale),
        }
    }

    #[inline]
    fn pos(self, x: f32, y: f32, z: f32) -> [f32; 3] {
        [x * self.sx, y * self.sy, z]
    }

    #[inline]
    fn rect_size(self, w: f32, h: f32) -> [f32; 2] {
        [w * self.sx, h * self.sy]
    }

    #[inline]
    fn glyph_size(self, w: f32, h: f32) -> [f32; 2] {
        [w * self.text_scale, h * self.text_scale]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RuntimeConsoleEditTarget {
    Command,
    FpsCap,
    RenderDistance,
    FsrSharpness,
}

impl RuntimeConsoleEditTarget {
    fn label(self) -> &'static str {
        match self {
            Self::Command => "command",
            Self::FpsCap => "fps_cap",
            Self::RenderDistance => "render_distance",
            Self::FsrSharpness => "fsr_sharpness",
        }
    }

    fn next(self) -> Self {
        match self {
            Self::Command => Self::FpsCap,
            Self::FpsCap => Self::RenderDistance,
            Self::RenderDistance => Self::FsrSharpness,
            Self::FsrSharpness => Self::Command,
        }
    }
}

impl Default for RuntimeConsoleState {
    fn default() -> Self {
        Self {
            open: false,
            input_line: String::new(),
            history: Vec::new(),
            history_cursor: None,
            last_feedback: String::new(),
            log_lines: Vec::new(),
            script_vars: BTreeMap::new(),
            script_statements: Vec::new(),
            script_overlay: empty_showcase_output(),
            edit_target: RuntimeConsoleEditTarget::Command,
            quick_fps_cap: "60".to_string(),
            quick_render_distance: "off".to_string(),
            quick_fsr_sharpness: "0.35".to_string(),
            log_scroll: 0,
            log_filter: RuntimeConsoleLogFilter::All,
            log_tail_limit: None,
            tail_follow: None,
            file_watch: None,
        }
    }
}

fn empty_showcase_output() -> TlscriptShowcaseFrameOutput {
    TlscriptShowcaseFrameOutput {
        patch: BounceTankRuntimePatch::default(),
        light_overrides: Vec::new(),
        rt_mode: None,
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

impl TlAppRuntime {
    fn new(
        event_loop: &ActiveEventLoop,
        options: CliOptions,
        platform: RuntimePlatform,
    ) -> Result<Self, Box<dyn Error>> {
        let file_io_root = env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .canonicalize()
            .unwrap_or_else(|_| PathBuf::from("."));
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
        let adaptive_pacer_enabled = mobile_class_tuning
            && options.fps_cap.is_none()
            && matches!(options.vsync, VsyncMode::Off);
        let display_refresh_hint_hz = window
            .current_monitor()
            .and_then(|monitor| monitor.refresh_rate_millihertz())
            .map(|mhz| (mhz as f32 / 1_000.0).max(24.0));
        let mut adaptive_pacer_fps = display_refresh_hint_hz.unwrap_or(60.0).clamp(48.0, 90.0);
        if little_core_class {
            adaptive_pacer_fps = adaptive_pacer_fps.min(72.0);
        }
        let uncapped_dynamic_fps_hint = options.fps_cap.is_none()
            && matches!(options.vsync, VsyncMode::Off)
            && !adaptive_pacer_enabled;
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
                speculative_sweep: true,
                speculative_max_distance: if mobile_class_tuning { 0.75 } else { 1.20 },
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
            ball_friction: 0.28,
            wall_restitution: 0.78,
            wall_friction: 0.20,
            friction_transition_speed: 1.2,
            friction_static_boost: 1.10,
            friction_kinetic_scale: 0.92,
            linear_damping: 0.012,
            initial_speed_min: 0.35,
            initial_speed_max: 1.25,
            ..BounceTankSceneConfig::default()
        });
        let mut renderer = WgpuSceneRenderer::new(
            &device,
            &queue,
            config.format,
            size.width,
            size.height,
            adapter_info.backend,
        );
        renderer.set_ray_tracing_mode(&queue, RayTracingMode::Auto);
        let fsr_config = FsrConfig {
            mode: options.fsr_mode,
            quality: options.fsr_quality,
            sharpness: options.fsr_sharpness,
            render_scale_override: options.fsr_scale_override,
        };
        renderer.set_fsr_config(&queue, fsr_config);
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
        let fsr_status = renderer.fsr_status();
        eprintln!(
            "[mps] cpu profile logical={} physical={} mobile_tuning={} little_core_class={} thread_scale={:.2} broadphase_chunk={} max_pairs={} solver_iters={} max_substeps={} render_distance={:?} adaptive_distance={} distance_blur={:?} ({}) adaptive_pacer={} ({:.0} fps) fsr={:?} active={} scale={:.2} sharpness={:.2} reason={}",
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
            distance_blur_enabled,
            adaptive_pacer_enabled,
            adaptive_pacer_fps,
            options.fsr_mode,
            fsr_status.active,
            fsr_status.render_scale,
            fsr_status.sharpness,
            if fsr_status.reason.is_empty() {
                "none"
            } else {
                fsr_status.reason.as_str()
            }
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
            .map(|fps| 1.0 / fps.max(1.0))
            .or_else(|| adaptive_pacer_enabled.then_some(1.0 / adaptive_pacer_fps.max(1.0)));
        let frame_cap_interval = frame_cap_interval.map(Duration::from_secs_f32);
        let fps_report_interval = options.fps_report_interval;

        let mut runtime = Self {
            cli_options: options.clone(),
            file_io_root,
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
            script_key_g_keyboard: false,
            console: RuntimeConsoleState::default(),
            console_overlay_sprites: Vec::new(),
            keyboard_camera: CameraInputState::default(),
            mouse_look_held: false,
            look_lock_active: false,
            mouse_look_delta: (0.0, 0.0),
            camera_reset_requested: false,
            keyboard_modifiers: ModifiersState::empty(),
            gamepad,
            touch_look_id: None,
            touch_last_position: None,
            cursor_position: None,
            tick_policy,
            tick_profile: options.tick_profile,
            tick_hz: 1.0 / fixed_dt.max(1e-6),
            fps_limit_hint,
            uncapped_dynamic_fps_hint,
            adaptive_pacer_enabled,
            adaptive_pacer_fps,
            adaptive_pacer_timer: 0.0,
            mps_logical_threads: logical_threads,
            max_substeps: tuned_max_substeps,
            last_substeps: 0,
            manual_max_substeps: None,
            simulation_paused: false,
            simulation_step_budget: 0,
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
            rt_mode: RayTracingMode::Auto,
            fsr_config,
            shutdown_prepared: false,
        };
        runtime.sync_console_quick_fields_from_runtime();
        Ok(runtime)
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

    fn prepare_for_exit(&mut self) {
        if self.shutdown_prepared {
            return;
        }
        self.shutdown_prepared = true;
        self.console.open = false;
        self.mouse_look_held = false;
        self.look_lock_active = false;
        self.keyboard_camera = CameraInputState::default();
        self.script_key_f_keyboard = false;
        self.script_key_g_keyboard = false;
        self.console_overlay_sprites.clear();
        self.frame_cap_interval = None;
        self.adaptive_pacer_enabled = false;
        self.queue.submit(std::iter::empty());
        if let Err(err) = self.device.poll(wgpu::PollType::wait_indefinitely()) {
            eprintln!("[shutdown] device poll failed: {err}");
        }
    }

    fn toggle_console(&mut self) {
        self.console.open = !self.console.open;
        self.console.history_cursor = None;
        self.console.edit_target = RuntimeConsoleEditTarget::Command;
        if self.console.open {
            self.keyboard_camera = CameraInputState::default();
            self.mouse_look_held = false;
            self.script_key_f_keyboard = false;
            self.script_key_g_keyboard = false;
            self.cursor_position = None;
            self.camera.set_look_active(&self.window, false);
            self.sync_console_quick_fields_from_runtime();
            self.console_feedback(
                "console opened (Ctrl+F1 toggles, Enter runs command, 'help' lists commands)",
            );
        } else {
            self.console_feedback("console closed");
        }
    }

    fn sync_console_quick_fields_from_runtime(&mut self) {
        self.console.quick_fps_cap = self
            .frame_cap_interval
            .map(|itv| format!("{:.0}", 1.0 / itv.as_secs_f32().max(1e-6)))
            .unwrap_or_else(|| "off".to_string());
        self.console.quick_render_distance = self
            .render_distance
            .map(|value| format!("{value:.1}"))
            .unwrap_or_else(|| "off".to_string());
        self.console.quick_fsr_sharpness = format!("{:.2}", self.fsr_config.sharpness);
    }

    fn classify_console_feedback(message: &str) -> RuntimeConsoleLogLevel {
        let lower = message.to_ascii_lowercase();
        if lower.contains("error")
            || lower.contains("failed")
            || lower.contains("invalid")
            || lower.contains("unknown")
            || lower.contains("usage:")
            || lower.contains("out of range")
        {
            RuntimeConsoleLogLevel::Error
        } else {
            RuntimeConsoleLogLevel::Info
        }
    }

    fn push_console_log(
        &mut self,
        level: RuntimeConsoleLogLevel,
        message: String,
        print_to_stdout: bool,
    ) {
        self.console.last_feedback = message.clone();
        self.console.log_lines.push(RuntimeConsoleLogLine {
            timestamp: Instant::now(),
            level,
            message: message.clone(),
        });
        if self.console.log_lines.len() > CONSOLE_MAX_LOG_LINES {
            let trim = self.console.log_lines.len() - CONSOLE_MAX_LOG_LINES;
            self.console.log_lines.drain(0..trim);
        }
        let max_scroll = self.max_console_log_scroll();
        self.console.log_scroll = self.console.log_scroll.min(max_scroll);
        if print_to_stdout {
            println!("[tlapp console] {message}");
        }
    }

    fn console_feedback(&mut self, message: impl Into<String>) {
        let message = message.into();
        let level = Self::classify_console_feedback(&message);
        self.push_console_log(level, message, true);
    }

    fn submit_console_command(&mut self, command: String) -> RuntimeCommand {
        let command = command.trim().to_string();
        if !command.is_empty() {
            self.console.history.push(command.clone());
            if self.console.history.len() > 128 {
                let trim = self.console.history.len() - 128;
                self.console.history.drain(0..trim);
            }
            self.push_console_log(RuntimeConsoleLogLevel::Info, format!("> {command}"), false);
        }
        self.console.input_line.clear();
        self.console.history_cursor = None;
        self.console.log_scroll = 0;
        if command.is_empty() {
            return RuntimeCommand::Consumed;
        }
        self.execute_console_command(&command)
    }

    fn selected_console_edit_buffer_mut(&mut self) -> &mut String {
        match self.console.edit_target {
            RuntimeConsoleEditTarget::Command => &mut self.console.input_line,
            RuntimeConsoleEditTarget::FpsCap => &mut self.console.quick_fps_cap,
            RuntimeConsoleEditTarget::RenderDistance => &mut self.console.quick_render_distance,
            RuntimeConsoleEditTarget::FsrSharpness => &mut self.console.quick_fsr_sharpness,
        }
    }

    fn apply_active_console_edit_target(&mut self) -> RuntimeCommand {
        match self.console.edit_target {
            RuntimeConsoleEditTarget::Command => {
                self.submit_console_command(self.console.input_line.clone())
            }
            RuntimeConsoleEditTarget::FpsCap => {
                let value = self.console.quick_fps_cap.trim();
                match parse_fps_cap(value) {
                    Ok(cap) => {
                        self.apply_fps_cap_runtime(cap);
                        match cap {
                            Some(fps) => self.console_feedback(format!("fps cap set to {fps:.1}")),
                            None => self.console_feedback("fps cap disabled"),
                        }
                    }
                    Err(err) => self.console_feedback(err.to_string()),
                }
                RuntimeCommand::Consumed
            }
            RuntimeConsoleEditTarget::RenderDistance => {
                let value = self.console.quick_render_distance.trim();
                match parse_render_distance(value) {
                    Ok(distance) => {
                        self.render_distance = distance;
                        if let Some(value) = distance {
                            self.render_distance_min = (value * 0.72).clamp(28.0, value);
                            self.render_distance_max =
                                (value * 1.55).clamp(value, value.max(220.0));
                            self.console_feedback(format!("render distance set to {value:.1}"));
                        } else {
                            self.console_feedback("render distance disabled");
                        }
                    }
                    Err(err) => self.console_feedback(err.to_string()),
                }
                RuntimeCommand::Consumed
            }
            RuntimeConsoleEditTarget::FsrSharpness => {
                let value = self.console.quick_fsr_sharpness.trim();
                match parse_fsr_sharpness(value) {
                    Ok(sharpness) => {
                        self.fsr_config.sharpness = sharpness;
                        self.renderer.set_fsr_config(&self.queue, self.fsr_config);
                        self.console_feedback(format!("fsr sharpness set to {sharpness:.2}"));
                    }
                    Err(err) => self.console_feedback(err.to_string()),
                }
                RuntimeCommand::Consumed
            }
        }
    }

    fn handle_console_window_event(&mut self, event: &WindowEvent) -> bool {
        if !self.console.open {
            return false;
        }

        match event {
            WindowEvent::MouseWheel { delta, .. } => {
                let (delta_y, line_steps) = match delta {
                    MouseScrollDelta::LineDelta(_, y) => {
                        let steps = y.abs().round().max(1.0) as i32;
                        (*y as f64, steps)
                    }
                    MouseScrollDelta::PixelDelta(pixels) => {
                        let steps = (pixels.y.abs() / CONSOLE_WHEEL_PIXELS_PER_LINE)
                            .round()
                            .max(1.0) as i32;
                        (pixels.y, steps)
                    }
                };
                if delta_y > 0.0 {
                    self.scroll_console_logs(line_steps);
                } else if delta_y < 0.0 {
                    self.scroll_console_logs(-line_steps);
                }
                true
            }
            _ => false,
        }
    }

    #[inline]
    fn max_console_log_scroll(&self) -> usize {
        self.visible_console_log_count().saturating_sub(1)
    }

    fn scroll_console_logs(&mut self, delta_lines: i32) {
        if delta_lines == 0 {
            return;
        }
        let max_scroll = self.max_console_log_scroll();
        if delta_lines > 0 {
            self.console.log_scroll = self
                .console
                .log_scroll
                .saturating_add(delta_lines as usize)
                .min(max_scroll);
        } else {
            self.console.log_scroll = self
                .console
                .log_scroll
                .saturating_sub(delta_lines.unsigned_abs() as usize);
        }
    }

    fn visible_console_log_count(&self) -> usize {
        let mut count = self
            .console
            .log_lines
            .iter()
            .filter(|line| self.console.log_filter.matches(line.level))
            .count();
        if let Some(limit) = self.console.log_tail_limit {
            count = count.min(limit);
        }
        count
    }

    fn reset_simulation_state(&mut self) {
        let world_config = self.world.config().clone();
        let scene_config = self.scene.config();
        let sprite_program = self.scene.sprite_program_cloned();
        let ball_mesh_slot = self.scene.ball_mesh_slot();
        let container_mesh_slot = self.scene.container_mesh_slot();
        self.world = PhysicsWorld::new(world_config);
        self.scene = BounceTankSceneController::new(scene_config);
        if let Some(program) = sprite_program {
            self.scene.set_sprite_program(program);
        }
        if ball_mesh_slot.is_some() || container_mesh_slot.is_some() {
            let _ = self.scene.apply_runtime_patch(
                &mut self.world,
                BounceTankRuntimePatch {
                    ball_mesh_slot,
                    container_mesh_slot,
                    ..BounceTankRuntimePatch::default()
                },
            );
        }
        self.script_last_spawned = 0;
        self.script_frame_index = 0;
        self.last_substeps = 0;
        self.simulation_step_budget = 0;
        self.tick_retune_timer = 0.0;
        self.distance_retune_timer = 0.0;
    }

    fn apply_bundle_sprite_program(
        &mut self,
        program: Option<TlspriteProgram>,
        root_hint: Option<&Path>,
    ) {
        if let Some(program) = program {
            self.force_full_fbx_from_sprite = program.requires_full_fbx_render();
            self.scene.set_sprite_program(program.clone());
            if let Some(root) = root_hint {
                bind_renderer_meshes_from_tlsprite(
                    &mut self.renderer,
                    &self.device,
                    root,
                    &program,
                );
            }
        } else {
            self.force_full_fbx_from_sprite = false;
            self.scene.clear_sprite_program();
        }
        self.renderer
            .set_force_full_fbx_sphere(self.force_full_fbx_from_sprite);
    }

    fn reload_sprite_from_watcher(&mut self) -> Result<String, String> {
        let (Some(sprite_loader), Some(sprite_cache)) =
            (self.sprite_loader.as_mut(), self.sprite_cache.as_mut())
        else {
            return Err("sprite watcher is not active in this runtime mode".to_string());
        };
        let event = sprite_loader.reload_into_cache(sprite_cache);
        let event_note = format!("{event:?}");
        print_tlsprite_event("[tlsprite manual reload]", event);
        if let Some(program) = sprite_cache.program_for_path(sprite_loader.path()).cloned() {
            self.force_full_fbx_from_sprite = program.requires_full_fbx_render();
            self.scene.set_sprite_program(program.clone());
            bind_renderer_meshes_from_tlsprite(
                &mut self.renderer,
                &self.device,
                sprite_loader.path(),
                &program,
            );
            self.renderer
                .set_force_full_fbx_sphere(self.force_full_fbx_from_sprite);
            Ok(format!(
                "sprite reloaded from '{}' ({event_note})",
                sprite_loader.path().display()
            ))
        } else {
            Err(format!(
                "sprite reload did not produce a compiled program ({event_note})"
            ))
        }
    }

    fn reload_script_runtime_from_sources(
        &mut self,
        include_bundle_sprite: bool,
    ) -> Result<String, String> {
        if let Some(project_path) = &self.cli_options.project_path {
            let compile = compile_tlpfile_scene_from_path(
                project_path,
                Some(&self.cli_options.joint_scene),
                TlscriptShowcaseConfig::default(),
            );
            let warning_count = compile
                .diagnostics
                .iter()
                .filter(|it| matches!(it.level, TlpfileDiagnosticLevel::Warning))
                .count();
            if compile.has_errors() {
                let first_error = compile
                    .diagnostics
                    .iter()
                    .find(|it| matches!(it.level, TlpfileDiagnosticLevel::Error))
                    .map(|it| it.message.as_str())
                    .unwrap_or("unknown project compile error");
                return Err(format!(
                    "project reload failed for '{}': {}",
                    project_path.display(),
                    first_error
                ));
            }
            let bundle = compile.bundle.ok_or_else(|| {
                format!(
                    "project reload produced no bundle for '{}'",
                    project_path.display()
                )
            })?;
            let scene_name = bundle.scene_name.clone();
            let script_count = bundle.scripts.len();
            let sprite_count = bundle.sprite_count();
            let merged_sprite_program = bundle.merged_sprite_program.clone();
            let sprite_root = bundle
                .selected_joint_path
                .clone()
                .or_else(|| Some(bundle.project_path.clone()));
            self.script_runtime = ScriptRuntime::MultiScripts(bundle.scripts);
            if include_bundle_sprite {
                self.apply_bundle_sprite_program(merged_sprite_program, sprite_root.as_deref());
            }
            return Ok(format!(
                "project reloaded scene='{}' scripts={} sprites={} warnings={}",
                scene_name, script_count, sprite_count, warning_count
            ));
        }

        if let Some(joint_path) = &self.cli_options.joint_path {
            let compile = compile_tljoint_scene_from_path(
                joint_path,
                &self.cli_options.joint_scene,
                TlscriptShowcaseConfig::default(),
            );
            let warning_count = compile
                .diagnostics
                .iter()
                .filter(|it| matches!(it.level, TljointDiagnosticLevel::Warning))
                .count();
            if compile.has_errors() {
                let first_error = compile
                    .diagnostics
                    .iter()
                    .find(|it| matches!(it.level, TljointDiagnosticLevel::Error))
                    .map(|it| it.message.as_str())
                    .unwrap_or("unknown joint compile error");
                return Err(format!(
                    "joint reload failed for '{}': {}",
                    joint_path.display(),
                    first_error
                ));
            }
            let bundle = compile.bundle.ok_or_else(|| {
                format!(
                    "joint reload produced no bundle for '{}'",
                    joint_path.display()
                )
            })?;
            let scene_name = bundle.scene_name.clone();
            let script_count = bundle.scripts.len();
            let sprite_count = bundle.sprite_paths.len();
            let merged_sprite_program = bundle.merged_sprite_program.clone();
            let sprite_root = Some(bundle.manifest_path.clone());
            self.script_runtime = ScriptRuntime::Joint(bundle);
            if include_bundle_sprite {
                self.apply_bundle_sprite_program(merged_sprite_program, sprite_root.as_deref());
            }
            return Ok(format!(
                "joint reloaded scene='{}' scripts={} sprites={} warnings={}",
                scene_name, script_count, sprite_count, warning_count
            ));
        }

        let script_source_owned =
            fs::read_to_string(&self.cli_options.script_path).map_err(|err| {
                format!(
                    "failed to read script '{}': {err}",
                    self.cli_options.script_path.display()
                )
            })?;
        let script_source: &'static str = Box::leak(script_source_owned.into_boxed_str());
        let compile = compile_tlscript_showcase(script_source, TlscriptShowcaseConfig::default());
        if !compile.errors.is_empty() {
            return Err(format!(
                "script reload failed for '{}': {}",
                self.cli_options.script_path.display(),
                compile.errors.join(" | ")
            ));
        }
        let warnings = compile.warnings.len();
        let program = compile
            .program
            .ok_or_else(|| "script reload produced no runnable program".to_string())?;
        self.script_runtime = ScriptRuntime::Single(program);
        Ok(format!(
            "script reloaded '{}' warnings={}",
            self.cli_options.script_path.display(),
            warnings
        ))
    }

    fn apply_gfx_profile(&mut self, profile: &str) -> Result<String, String> {
        let mobile_path = matches!(self.platform, RuntimePlatform::Android)
            || matches!(self.scheduler_path, GraphicsSchedulerPath::Mgs);
        match profile {
            "low" => {
                self.fsr_config.mode = FsrMode::On;
                self.fsr_config.quality = FsrQualityPreset::Performance;
                self.fsr_config.sharpness = 0.42;
                self.fsr_config.render_scale_override = None;
                self.render_distance = Some(if mobile_path { 36.0 } else { 48.0 });
                self.adaptive_distance_enabled = true;
                self.distance_blur_mode = ToggleAuto::On;
                self.tick_profile = TickProfile::Balanced;
            }
            "med" | "medium" => {
                self.fsr_config.mode = FsrMode::Auto;
                self.fsr_config.quality = FsrQualityPreset::Balanced;
                self.fsr_config.sharpness = 0.36;
                self.fsr_config.render_scale_override = None;
                self.render_distance = Some(if mobile_path { 56.0 } else { 76.0 });
                self.adaptive_distance_enabled = true;
                self.distance_blur_mode = ToggleAuto::Auto;
                self.tick_profile = TickProfile::Balanced;
            }
            "high" => {
                self.fsr_config.mode = FsrMode::Auto;
                self.fsr_config.quality = FsrQualityPreset::Quality;
                self.fsr_config.sharpness = 0.33;
                self.fsr_config.render_scale_override = None;
                self.render_distance = Some(if mobile_path { 72.0 } else { 104.0 });
                self.adaptive_distance_enabled = true;
                self.distance_blur_mode = ToggleAuto::Off;
                self.tick_profile = TickProfile::Max;
            }
            "ultra" => {
                self.fsr_config.mode = FsrMode::Off;
                self.fsr_config.quality = FsrQualityPreset::Native;
                self.fsr_config.sharpness = 0.28;
                self.fsr_config.render_scale_override = Some(1.0);
                self.render_distance = None;
                self.adaptive_distance_enabled = false;
                self.distance_blur_mode = ToggleAuto::Off;
                self.tick_profile = TickProfile::Max;
            }
            _ => {
                return Err("usage: gfx.profile <low|med|high|ultra>".to_string());
            }
        }

        if let Some(distance) = self.render_distance {
            self.render_distance_min = (distance * 0.72).clamp(28.0, distance);
            self.render_distance_max = (distance * 1.55).clamp(distance, 260.0);
        } else {
            self.render_distance_min = 0.0;
            self.render_distance_max = 0.0;
        }
        self.distance_blur_enabled = self.distance_blur_mode.resolve(mobile_path);
        self.renderer.set_fsr_config(&self.queue, self.fsr_config);
        self.sync_console_quick_fields_from_runtime();

        let distance_label = self
            .render_distance
            .map(|v| format!("{v:.1}"))
            .unwrap_or_else(|| "off".to_string());
        Ok(format!(
            "gfx profile '{}' applied: fsr={:?}/{:?} sharpness={:.2} distance={} blur={:?} tick_profile={:?}",
            profile,
            self.fsr_config.mode,
            self.fsr_config.quality,
            self.fsr_config.sharpness,
            distance_label,
            self.distance_blur_mode,
            self.tick_profile
        ))
    }

    fn apply_present_mode(&mut self, mode: wgpu::PresentMode) {
        if self.present_mode == mode {
            return;
        }
        self.present_mode = mode;
        self.config.present_mode = mode;
        self.surface.configure(&self.device, &self.config);
    }

    fn apply_fps_cap_runtime(&mut self, cap: Option<f32>) {
        match cap {
            Some(fps) => {
                let clamped = fps.max(24.0);
                self.frame_cap_interval = Some(Duration::from_secs_f32(1.0 / clamped));
                self.fps_limit_hint = clamped;
                self.uncapped_dynamic_fps_hint = false;
                self.adaptive_pacer_enabled = false;
            }
            None => {
                self.frame_cap_interval = None;
                self.uncapped_dynamic_fps_hint = true;
                self.fps_limit_hint = bootstrap_uncapped_fps_hint(self.mps_logical_threads);
                self.adaptive_pacer_enabled = false;
            }
        }
    }

    fn resolve_console_candidate_path(&self, raw_path: &str) -> Result<PathBuf, String> {
        let trimmed = raw_path.trim();
        if trimmed.is_empty() {
            return Err("path must not be empty".to_string());
        }
        let candidate = PathBuf::from(trimmed);
        Ok(if candidate.is_absolute() {
            candidate
        } else {
            self.file_io_root.join(candidate)
        })
    }

    fn resolve_console_existing_path(&self, raw_path: &str, op: &str) -> Result<PathBuf, String> {
        let candidate = self.resolve_console_candidate_path(raw_path)?;
        let canonical = candidate
            .canonicalize()
            .map_err(|err| format!("{op}: failed to access '{raw_path}': {err}"))?;
        if !canonical.starts_with(&self.file_io_root) {
            return Err(format!(
                "{op}: denied (path escapes workspace root '{}')",
                self.file_io_root.display()
            ));
        }
        Ok(canonical)
    }

    fn emit_console_text_lines(&mut self, prefix: &str, text: &str, max_lines: usize) {
        let mut lines = text.lines();
        let mut emitted = 0usize;
        while emitted < max_lines {
            let Some(line) = lines.next() else {
                break;
            };
            self.console_feedback(format!("{prefix}{line}"));
            emitted += 1;
        }
        if emitted == 0 {
            self.console_feedback(format!("{prefix}<empty>"));
        } else if lines.next().is_some() {
            self.console_feedback(format!("{prefix}... output truncated to {max_lines} lines"));
        }
    }

    fn read_file_prefix_bytes(
        &self,
        path: &Path,
        max_bytes: usize,
    ) -> Result<(Vec<u8>, bool), String> {
        let mut file = fs::File::open(path)
            .map_err(|err| format!("failed to open '{}': {err}", path.display()))?;
        let file_len = file.metadata().ok().map(|meta| meta.len());
        let mut buf = vec![0u8; max_bytes];
        let read_len = file
            .read(&mut buf)
            .map_err(|err| format!("failed to read '{}': {err}", path.display()))?;
        buf.truncate(read_len);
        let truncated = file_len.map(|len| len > read_len as u64).unwrap_or(false);
        Ok((buf, truncated))
    }

    fn read_file_tail_window(
        &self,
        path: &Path,
        max_window_bytes: usize,
    ) -> Result<(Vec<u8>, bool), String> {
        let mut file = fs::File::open(path)
            .map_err(|err| format!("failed to open '{}': {err}", path.display()))?;
        let file_len = file
            .metadata()
            .map(|meta| meta.len())
            .map_err(|err| format!("failed to query metadata '{}': {err}", path.display()))?;
        if file_len == 0 {
            return Ok((Vec::new(), false));
        }
        let start_offset = file_len.saturating_sub(max_window_bytes as u64);
        file.seek(SeekFrom::Start(start_offset))
            .map_err(|err| format!("failed to seek '{}': {err}", path.display()))?;
        let mut buf = Vec::with_capacity((file_len - start_offset) as usize);
        file.read_to_end(&mut buf)
            .map_err(|err| format!("failed to read tail '{}': {err}", path.display()))?;
        Ok((buf, start_offset > 0))
    }

    fn run_file_find(
        &mut self,
        path: &Path,
        pattern: &str,
        max_matches: usize,
        max_bytes: usize,
        case_insensitive: bool,
    ) {
        match self.read_file_prefix_bytes(path, max_bytes) {
            Ok((bytes, truncated)) => {
                let text = String::from_utf8_lossy(&bytes);
                let mut total_matches = 0usize;
                let mut shown = 0usize;
                let pattern_cmp = if case_insensitive {
                    pattern.to_ascii_lowercase()
                } else {
                    String::new()
                };
                self.console_feedback(format!(
                    "{} '{}' pattern='{}' scan={} byte(s) max_matches={}{}",
                    if case_insensitive {
                        "file.findi"
                    } else {
                        "file.find"
                    },
                    path.display(),
                    pattern,
                    bytes.len(),
                    max_matches,
                    if truncated {
                        ", truncated scan window"
                    } else {
                        ""
                    }
                ));
                for (line_idx, line) in text.lines().enumerate() {
                    let matched = if case_insensitive {
                        line.to_ascii_lowercase().contains(&pattern_cmp)
                    } else {
                        line.contains(pattern)
                    };
                    if !matched {
                        continue;
                    }
                    total_matches += 1;
                    if shown < max_matches {
                        shown += 1;
                        self.console_feedback(format!("  L{}: {}", line_idx + 1, line));
                    }
                }
                if total_matches == 0 {
                    self.console_feedback("  <no matches>");
                } else if total_matches > max_matches {
                    self.console_feedback(format!(
                        "  ... {} additional match(es) hidden",
                        total_matches - max_matches
                    ));
                }
            }
            Err(err) => self.console_feedback(format!(
                "{}: {err}",
                if case_insensitive {
                    "file.findi"
                } else {
                    "file.find"
                }
            )),
        }
    }

    fn run_file_find_regex(
        &mut self,
        path: &Path,
        pattern: &str,
        max_matches: usize,
        max_bytes: usize,
    ) {
        let regex = match RegexBuilder::new(pattern).size_limit(1_000_000).build() {
            Ok(regex) => regex,
            Err(err) => {
                self.console_feedback(format!("file.findr: invalid regex '{pattern}': {err}"));
                return;
            }
        };
        match self.read_file_prefix_bytes(path, max_bytes) {
            Ok((bytes, truncated)) => {
                let text = String::from_utf8_lossy(&bytes);
                let mut total_matches = 0usize;
                let mut shown = 0usize;
                self.console_feedback(format!(
                    "file.findr '{}' regex='{}' scan={} byte(s) max_matches={}{}",
                    path.display(),
                    pattern,
                    bytes.len(),
                    max_matches,
                    if truncated {
                        ", truncated scan window"
                    } else {
                        ""
                    }
                ));
                for (line_idx, line) in text.lines().enumerate() {
                    if !regex.is_match(line) {
                        continue;
                    }
                    total_matches += 1;
                    if shown < max_matches {
                        shown += 1;
                        self.console_feedback(format!("  L{}: {}", line_idx + 1, line));
                    }
                }
                if total_matches == 0 {
                    self.console_feedback("  <no matches>");
                } else if total_matches > max_matches {
                    self.console_feedback(format!(
                        "  ... {} additional match(es) hidden",
                        total_matches - max_matches
                    ));
                }
            }
            Err(err) => self.console_feedback(format!("file.findr: {err}")),
        }
    }

    fn run_file_grep(
        &mut self,
        path: &Path,
        pattern: &str,
        context: usize,
        max_matches: usize,
        max_bytes: usize,
    ) {
        match self.read_file_prefix_bytes(path, max_bytes) {
            Ok((bytes, truncated)) => {
                let text = String::from_utf8_lossy(&bytes);
                let lines = text.lines().collect::<Vec<_>>();
                let mut total_matches = 0usize;
                let mut shown_matches = 0usize;
                let mut printed = vec![false; lines.len()];
                self.console_feedback(format!(
                    "file.grep '{}' pattern='{}' context={} scan={} byte(s) max_matches={}{}",
                    path.display(),
                    pattern,
                    context,
                    bytes.len(),
                    max_matches,
                    if truncated {
                        ", truncated scan window"
                    } else {
                        ""
                    }
                ));
                for (idx, line) in lines.iter().enumerate() {
                    if !line.contains(pattern) {
                        continue;
                    }
                    total_matches += 1;
                    if shown_matches >= max_matches {
                        continue;
                    }
                    shown_matches += 1;
                    let start = idx.saturating_sub(context);
                    let end = (idx + context).min(lines.len().saturating_sub(1));
                    for i in start..=end {
                        if printed[i] {
                            continue;
                        }
                        printed[i] = true;
                        let marker = if i == idx { ">" } else { " " };
                        self.console_feedback(format!(" {marker} L{}: {}", i + 1, lines[i]));
                    }
                }
                if total_matches == 0 {
                    self.console_feedback("  <no matches>");
                } else if total_matches > max_matches {
                    self.console_feedback(format!(
                        "  ... {} additional match(es) hidden",
                        total_matches - max_matches
                    ));
                }
            }
            Err(err) => self.console_feedback(format!("file.grep: {err}")),
        }
    }

    fn poll_console_tail_follow(&mut self) {
        let (path, cursor, poll_interval, max_lines_per_poll, partial_line, last_poll, last_error) =
            match self.console.tail_follow.as_ref() {
                Some(follow) => (
                    follow.path.clone(),
                    follow.cursor,
                    follow.poll_interval,
                    follow.max_lines_per_poll,
                    follow.partial_line.clone(),
                    follow.last_poll,
                    follow.last_error.clone(),
                ),
                None => return,
            };

        let now = Instant::now();
        if now.saturating_duration_since(last_poll) < poll_interval {
            return;
        }

        let mut next_cursor = cursor;
        let mut next_partial = partial_line;
        let mut next_error: Option<String> = None;

        let metadata = match fs::metadata(&path) {
            Ok(meta) => meta,
            Err(err) => {
                let msg = format!("tailf metadata failed for '{}': {err}", path.display());
                if last_error.as_deref() != Some(msg.as_str()) {
                    self.console_feedback(format!("[tailf] {msg}"));
                }
                if let Some(follow) = self.console.tail_follow.as_mut() {
                    follow.last_poll = now;
                    follow.last_error = Some(msg);
                }
                return;
            }
        };

        if !metadata.is_file() {
            let msg = format!("tailf target is no longer a file: '{}'", path.display());
            if last_error.as_deref() != Some(msg.as_str()) {
                self.console_feedback(format!("[tailf] {msg}"));
            }
            if let Some(follow) = self.console.tail_follow.as_mut() {
                follow.last_poll = now;
                follow.last_error = Some(msg);
            }
            return;
        }

        let file_len = metadata.len();
        if file_len < next_cursor {
            next_cursor = 0;
            next_partial.clear();
            self.console_feedback(format!(
                "[tailf] file was truncated/rotated, rewinding '{}'",
                path.display()
            ));
        }

        let bytes_to_read_u64 =
            (file_len - next_cursor).min(CONSOLE_FILE_TAIL_MAX_WINDOW_BYTES as u64);
        if bytes_to_read_u64 > 0 {
            match fs::File::open(&path) {
                Ok(mut file) => {
                    if let Err(err) = file.seek(SeekFrom::Start(next_cursor)) {
                        let msg = format!("tailf seek failed '{}': {err}", path.display());
                        if last_error.as_deref() != Some(msg.as_str()) {
                            self.console_feedback(format!("[tailf] {msg}"));
                        }
                        next_error = Some(msg);
                    } else {
                        let mut buf = vec![0u8; bytes_to_read_u64 as usize];
                        match file.read_exact(&mut buf) {
                            Ok(()) => {
                                next_cursor = next_cursor.saturating_add(bytes_to_read_u64);
                                let chunk = String::from_utf8_lossy(&buf);
                                next_partial.push_str(&chunk);

                                let mut lines = Vec::new();
                                while let Some(pos) = next_partial.find('\n') {
                                    let mut line = next_partial[..pos].to_string();
                                    if line.ends_with('\r') {
                                        line.pop();
                                    }
                                    lines.push(line);
                                    next_partial.drain(..=pos);
                                }
                                if lines.len() > max_lines_per_poll {
                                    let skipped = lines.len() - max_lines_per_poll;
                                    self.console_feedback(format!(
                                        "[tailf] ... {skipped} line(s) skipped this poll"
                                    ));
                                }
                                let start = lines.len().saturating_sub(max_lines_per_poll);
                                for line in lines.into_iter().skip(start) {
                                    self.console_feedback(format!("[tailf] {line}"));
                                }
                                if next_partial.len() > 8 * 1024 {
                                    let keep_from = next_partial.len().saturating_sub(8 * 1024);
                                    next_partial = next_partial[keep_from..].to_string();
                                }
                            }
                            Err(err) => {
                                let msg = format!("tailf read failed '{}': {err}", path.display());
                                if last_error.as_deref() != Some(msg.as_str()) {
                                    self.console_feedback(format!("[tailf] {msg}"));
                                }
                                next_error = Some(msg);
                            }
                        }
                    }
                }
                Err(err) => {
                    let msg = format!("tailf open failed '{}': {err}", path.display());
                    if last_error.as_deref() != Some(msg.as_str()) {
                        self.console_feedback(format!("[tailf] {msg}"));
                    }
                    next_error = Some(msg);
                }
            }
        }

        if let Some(follow) = self.console.tail_follow.as_mut() {
            follow.cursor = next_cursor;
            follow.partial_line = next_partial;
            follow.last_poll = now;
            follow.last_error = next_error;
        }
    }

    fn poll_console_file_watch(&mut self) {
        let (path, poll_interval, last_modified, last_len, last_poll, last_error) =
            match self.console.file_watch.as_ref() {
                Some(watch) => (
                    watch.path.clone(),
                    watch.poll_interval,
                    watch.last_modified,
                    watch.last_len,
                    watch.last_poll,
                    watch.last_error.clone(),
                ),
                None => return,
            };

        let now = Instant::now();
        if now.saturating_duration_since(last_poll) < poll_interval {
            return;
        }

        let metadata = match fs::metadata(&path) {
            Ok(meta) => meta,
            Err(err) => {
                let msg = format!("watch metadata failed for '{}': {err}", path.display());
                if last_error.as_deref() != Some(msg.as_str()) {
                    self.console_feedback(format!("[watch] {msg}"));
                }
                if let Some(watch) = self.console.file_watch.as_mut() {
                    watch.last_poll = now;
                    watch.last_error = Some(msg);
                }
                return;
            }
        };

        if !metadata.is_file() {
            let msg = format!("watch target is no longer a file: '{}'", path.display());
            if last_error.as_deref() != Some(msg.as_str()) {
                self.console_feedback(format!("[watch] {msg}"));
            }
            if let Some(watch) = self.console.file_watch.as_mut() {
                watch.last_poll = now;
                watch.last_error = Some(msg);
            }
            return;
        }

        let modified = metadata.modified().ok();
        let len = metadata.len();
        let changed = len != last_len || modified != last_modified;
        if changed {
            self.console_feedback(format!(
                "[watch] changed '{}' size={} (delta={:+})",
                path.display(),
                len,
                len as i64 - last_len as i64
            ));
        }

        if let Some(watch) = self.console.file_watch.as_mut() {
            watch.last_modified = modified;
            watch.last_len = len;
            watch.last_poll = now;
            watch.last_error = None;
        }
    }

    fn console_status_line(&self) -> String {
        let fsr = self.renderer.fsr_status();
        let rt = self.renderer.ray_tracing_status();
        let fps_cap = self
            .frame_cap_interval
            .map(|itv| format!("{:.0}", 1.0 / itv.as_secs_f32().max(1e-6)))
            .unwrap_or_else(|| "off".to_string());
        let render_distance = self
            .render_distance
            .map(|v| format!("{v:.1}"))
            .unwrap_or_else(|| "off".to_string());
        let log_tail = self
            .console
            .log_tail_limit
            .map(|n| n.to_string())
            .unwrap_or_else(|| "off".to_string());
        let log_filter = match self.console.log_filter {
            RuntimeConsoleLogFilter::All => "all",
            RuntimeConsoleLogFilter::Info => "info",
            RuntimeConsoleLogFilter::Error => "error",
        };
        let substeps = self
            .manual_max_substeps
            .map(|n| format!("manual:{n}"))
            .unwrap_or_else(|| format!("auto:{}", self.max_substeps));
        let tailf_state = self
            .console
            .tail_follow
            .as_ref()
            .map(|f| f.path.display().to_string())
            .unwrap_or_else(|| "off".to_string());
        let watch_state = self
            .console
            .file_watch
            .as_ref()
            .map(|w| w.path.display().to_string())
            .unwrap_or_else(|| "off".to_string());
        format!(
            "fps_cap={fps_cap} vsync={:?} rt={:?}/{} fsr={:?}/{} scale={:.2} sharpness={:.2} render_distance={} adaptive_distance={:?} distance_blur={:?} sim_paused={} step_budget={} substeps={} log_filter={} log_tail={} tailf={} watch={} script_vars={} script_calls={}",
            self.present_mode,
            self.rt_mode,
            if rt.active { "on" } else { "off" },
            fsr.requested_mode,
            if fsr.active { "on" } else { "off" },
            fsr.render_scale,
            fsr.sharpness,
            render_distance,
            self.adaptive_distance_enabled,
            self.distance_blur_mode,
            self.simulation_paused,
            self.simulation_step_budget,
            substeps,
            log_filter,
            log_tail,
            tailf_state,
            watch_state,
            self.console.script_vars.len(),
            self.console.script_statements.len()
        )
    }

    fn console_title_suffix(&self) -> String {
        if !self.console.open {
            return String::new();
        }
        if self.console.input_line.is_empty() {
            return " | CLI ready".to_string();
        }
        let mut preview = self.console.input_line.clone();
        if preview.chars().count() > 42 {
            preview = preview.chars().take(42).collect::<String>() + "...";
        }
        format!(" | CLI> {preview}")
    }

    fn expand_console_script_vars(&self, statement: &str) -> Result<String, String> {
        let mut out = String::with_capacity(statement.len() + 16);
        let chars = statement.chars().collect::<Vec<_>>();
        let mut i = 0usize;
        while i < chars.len() {
            let ch = chars[i];
            if ch != '$' {
                out.push(ch);
                i += 1;
                continue;
            }
            let start = i + 1;
            let mut end = start;
            while end < chars.len() && (chars[end].is_ascii_alphanumeric() || chars[end] == '_') {
                end += 1;
            }
            if end == start {
                out.push(ch);
                i += 1;
                continue;
            }
            let name = chars[start..end].iter().collect::<String>();
            let Some(value) = self.console.script_vars.get(&name) else {
                return Err(format!("unknown script var '${name}'"));
            };
            out.push_str(value);
            i = end;
        }
        Ok(out)
    }

    fn evaluate_console_statement(
        &self,
        statement: &str,
    ) -> Result<TlscriptShowcaseFrameOutput, String> {
        let source = format!("@export\ndef showcase_tick():\n    {statement}\n");
        let compile = compile_tlscript_showcase(&source, TlscriptShowcaseConfig::default());
        if !compile.errors.is_empty() {
            return Err(compile.errors.join(" | "));
        }
        let Some(program) = compile.program else {
            return Err("script statement produced no runnable program".to_string());
        };
        let mut output = program.evaluate_frame_with_controls(
            TlscriptShowcaseFrameInput {
                frame_index: self.script_frame_index,
                live_balls: self.scene.live_ball_count(),
                spawned_this_tick: self.script_last_spawned,
                key_f_down: self.script_key_f_keyboard || self.gamepad.action_f_down(),
            },
            TlscriptShowcaseControlInput::default(),
        );
        if !compile.warnings.is_empty() {
            output.warnings.extend(
                compile
                    .warnings
                    .into_iter()
                    .map(|warning| format!("compile warning: {warning}")),
            );
        }
        Ok(output)
    }

    fn sanitize_console_overlay_output(output: &mut TlscriptShowcaseFrameOutput) {
        output.camera_translate_delta = None;
        output.camera_rotate_delta_deg = None;
        output.camera_move_axis = None;
        output.camera_look_delta = None;
        output.camera_sprint = None;
        output.camera_look_active = None;
        output.camera_reset_pose = false;
        output.dispatch_decision = None;
        output.aborted_early = false;
    }

    fn normalize_script_call_statement(raw: &str) -> Result<String, String> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err("script.call requires a function call".to_string());
        }
        if trimmed.contains('(') {
            return Ok(trimmed.trim_end_matches(';').to_string());
        }

        let mut tokens = trimmed.split_whitespace();
        let Some(name) = tokens.next() else {
            return Err("script.call requires a function call".to_string());
        };
        if !is_valid_tlscript_ident(name) {
            return Err(format!("invalid function name '{name}'"));
        }
        let args = tokens.collect::<Vec<_>>();
        if args.is_empty() {
            return Ok(format!("{name}()"));
        }
        Ok(format!("{name}({})", args.join(", ")))
    }

    fn rebuild_console_script_overlay(&mut self) -> Result<Vec<String>, String> {
        let mut rebuilt = empty_showcase_output();
        let mut notes = Vec::new();
        for (index, statement_template) in self.console.script_statements.iter().enumerate() {
            let expanded = self
                .expand_console_script_vars(statement_template)
                .map_err(|err| {
                    format!("statement[{index}] '{statement_template}' variable error: {err}")
                })?;
            let mut output = self
                .evaluate_console_statement(&expanded)
                .map_err(|err| format!("statement[{index}] '{expanded}' failed: {err}"))?;
            if !output.warnings.is_empty() {
                notes.push(format!(
                    "statement[{index}] warnings: {}",
                    output.warnings.join(" | ")
                ));
            }
            Self::sanitize_console_overlay_output(&mut output);
            merge_showcase_output(&mut rebuilt, output, CONSOLE_SCRIPT_INDEX);
        }
        rebuilt.warnings.clear();
        self.console.script_overlay = rebuilt;
        Ok(notes)
    }

    fn execute_console_command(&mut self, command: &str) -> RuntimeCommand {
        let trimmed = command.trim();
        if trimmed.is_empty() {
            self.console_feedback("empty command");
            return RuntimeCommand::Consumed;
        }

        let mut parts = trimmed.split_whitespace();
        let Some(head) = parts.next() else {
            self.console_feedback("empty command");
            return RuntimeCommand::Consumed;
        };
        let head_lc = head.to_ascii_lowercase();
        match head_lc.as_str() {
            "help" | "?" => {
                if let Some(topic) = parts.next() {
                    let topic = topic.to_ascii_lowercase();
                    let topic_list = match topic.as_str() {
                        "file" => Some(vec![
                            "file.exists <path>",
                            "file.head <path> [lines]",
                            "file.tail <path> [lines] [max_bytes]",
                            "file.tailf <path>|stop [poll_ms] [max_lines]",
                            "file.watch <path>|stop [poll_ms]",
                            "file.read <path> [max_bytes]",
                            "file.list <dir> [limit]",
                            "file.find <path> <pattern> [max_matches] [max_bytes]",
                            "file.findi <path> <pattern> [max_matches] [max_bytes]",
                            "file.findr <path> <regex> [max_matches] [max_bytes]",
                            "file.grep <path> <pattern> [context] [max_matches] [max_bytes]",
                        ]),
                        "gfx" => Some(vec![
                            "gfx.status",
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
                        ]),
                        "sim" => Some(vec![
                            "sim.status",
                            "sim.pause",
                            "sim.resume",
                            "sim.step <n>",
                            "sim.reset",
                            "scene.reload | script.reload | sprite.reload",
                            "perf.snapshot",
                            "phys.gravity <x y z>",
                            "phys.substeps <auto|n>",
                        ]),
                        "script" => Some(vec![
                            "script.var <name> <expr>",
                            "script.unset <name>",
                            "script.vars",
                            "script.call <fn(args)>",
                            "script.exec <stmt>",
                            "script.uncall <idx|all>",
                            "script.list",
                            "script.clear",
                        ]),
                        "cam" | "camera" => {
                            Some(vec!["cam.speed <v>", "cam.sens <v>", "cam.reset"])
                        }
                        "log" => Some(vec![
                            "log.clear",
                            "log.tail <off|n>",
                            "log.level <all|info|error>",
                        ]),
                        _ => None,
                    };
                    if let Some(commands) = topic_list {
                        self.console_feedback(format!("help {topic}:"));
                        for cmd in commands {
                            self.console_feedback(format!("  {cmd}"));
                        }
                    } else {
                        self.console_feedback(
                            "unknown help topic (use: file|gfx|sim|script|cam|log)",
                        );
                    }
                } else {
                    self.console_feedback(format!(
                        "commands: {}",
                        CONSOLE_HELP_COMMANDS.join(" | ")
                    ));
                }
            }
            "version" | "ver" => {
                let query = parts.next().unwrap_or("all");
                if query.eq_ignore_ascii_case("all") {
                    self.console_feedback(format!("{ENGINE_ID} engine v{ENGINE_VERSION}"));
                    for entry in tileline_version_entries() {
                        self.console_feedback(format!("  {:<10} v{}", entry.module, entry.version));
                    }
                } else if let Some(entry) = resolve_tileline_version_query(query) {
                    self.console_feedback(format!("{} v{}", entry.module, entry.version));
                } else {
                    self.console_feedback(
                        "usage: version [module|all] (module: tileline|runtime|tl-core|mps|gms|mgs|nps|paradoxpe)",
                    );
                }
            }
            "status" | "gfx.status" => {
                self.console_feedback(self.console_status_line());
            }
            "sim.status" => {
                self.console_feedback(format!(
                    "sim paused={} step_budget={} tick_hz={:.1} max_substeps={} manual_substeps={:?}",
                    self.simulation_paused,
                    self.simulation_step_budget,
                    self.tick_hz,
                    self.max_substeps,
                    self.manual_max_substeps
                ));
            }
            "sim.pause" => {
                self.simulation_paused = true;
                self.console_feedback("simulation paused");
            }
            "sim.resume" => {
                self.simulation_paused = false;
                self.simulation_step_budget = 0;
                self.console_feedback("simulation resumed");
            }
            "sim.step" => {
                let steps = match parts.next() {
                    Some(value) => match parse_console_u32_in_range(value, "sim.step", 1, 240) {
                        Ok(parsed) => parsed,
                        Err(err) => {
                            self.console_feedback(err);
                            return RuntimeCommand::Consumed;
                        }
                    },
                    None => 1,
                };
                self.simulation_paused = true;
                self.simulation_step_budget = self.simulation_step_budget.saturating_add(steps);
                self.console_feedback(format!(
                    "simulation step budget increased by {steps} (pending={})",
                    self.simulation_step_budget
                ));
            }
            "sim.reset" => {
                self.reset_simulation_state();
                self.console_feedback("simulation reset (world + scene)");
            }
            "scene.reload" => {
                let mut notes = Vec::new();
                match self.reload_script_runtime_from_sources(true) {
                    Ok(note) => notes.push(note),
                    Err(err) => {
                        self.console_feedback(err);
                        return RuntimeCommand::Consumed;
                    }
                }
                if self.sprite_loader.is_some() {
                    match self.reload_sprite_from_watcher() {
                        Ok(note) => notes.push(note),
                        Err(err) => self.console_feedback(format!("sprite reload warning: {err}")),
                    }
                }
                self.reset_simulation_state();
                self.camera.reset_pose();
                self.console_feedback("scene state reset");
                for note in notes {
                    self.console_feedback(note);
                }
            }
            "script.reload" => match self.reload_script_runtime_from_sources(false) {
                Ok(note) => self.console_feedback(note),
                Err(err) => self.console_feedback(err),
            },
            "sprite.reload" => {
                if self.sprite_loader.is_some() {
                    match self.reload_sprite_from_watcher() {
                        Ok(note) => self.console_feedback(note),
                        Err(err) => self.console_feedback(err),
                    }
                } else if self.cli_options.project_path.is_some()
                    || self.cli_options.joint_path.is_some()
                {
                    match self.reload_script_runtime_from_sources(true) {
                        Ok(note) => self.console_feedback(format!(
                            "sprite refreshed via bundle reload: {note}"
                        )),
                        Err(err) => self.console_feedback(err),
                    }
                } else {
                    self.console_feedback(
                        "sprite.reload is unavailable (no sprite watcher / bundle context)",
                    );
                }
            }
            "perf.snapshot" => {
                let fps = self.fps_tracker.snapshot();
                let fsr = self.renderer.fsr_status();
                let rt = self.renderer.ray_tracing_status();
                let gravity = self.world.gravity();
                self.console_feedback(format!(
                    "perf snapshot | fps inst={:.1} ema={:.1} avg={:.1} stddev={:.2}ms | frame_ema={:.2}ms jitter_ema={:.2}ms | fill={:.2}/{:.2} | balls={} draw_limit={:?} | substeps={}/{} manual={:?} | grav=[{:.2},{:.2},{:.2}] | rt={:?}/{} dyn={} | fsr={:?}/{:?} scale={:.2} sharp={:.2}",
                    fps.instant_fps,
                    fps.ema_fps,
                    fps.avg_fps,
                    fps.frame_time_stddev_ms,
                    self.frame_time_ema_ms,
                    self.frame_time_jitter_ema_ms,
                    self.last_framebuffer_fill_ratio,
                    self.framebuffer_fill_ema,
                    self.scene.live_ball_count(),
                    self.adaptive_ball_render_limit,
                    self.last_substeps,
                    self.max_substeps,
                    self.manual_max_substeps,
                    gravity.x,
                    gravity.y,
                    gravity.z,
                    self.rt_mode,
                    if rt.active { "on" } else { "off" },
                    rt.rt_dynamic_count,
                    fsr.requested_mode,
                    if fsr.active { "on" } else { "off" },
                    fsr.render_scale,
                    fsr.sharpness
                ));
            }
            "phys.gravity" => {
                let Some(x_raw) = parts.next() else {
                    self.console_feedback("usage: phys.gravity <x> <y> <z>");
                    return RuntimeCommand::Consumed;
                };
                let Some(y_raw) = parts.next() else {
                    self.console_feedback("usage: phys.gravity <x> <y> <z>");
                    return RuntimeCommand::Consumed;
                };
                let Some(z_raw) = parts.next() else {
                    self.console_feedback("usage: phys.gravity <x> <y> <z>");
                    return RuntimeCommand::Consumed;
                };
                let x = match parse_console_f32_in_range(x_raw, "phys.gravity.x", -120.0, 120.0) {
                    Ok(v) => v,
                    Err(err) => {
                        self.console_feedback(err);
                        return RuntimeCommand::Consumed;
                    }
                };
                let y = match parse_console_f32_in_range(y_raw, "phys.gravity.y", -120.0, 120.0) {
                    Ok(v) => v,
                    Err(err) => {
                        self.console_feedback(err);
                        return RuntimeCommand::Consumed;
                    }
                };
                let z = match parse_console_f32_in_range(z_raw, "phys.gravity.z", -120.0, 120.0) {
                    Ok(v) => v,
                    Err(err) => {
                        self.console_feedback(err);
                        return RuntimeCommand::Consumed;
                    }
                };
                self.world.set_gravity(Vector3::new(x, y, z));
                self.console_feedback(format!("gravity set to [{x:.3}, {y:.3}, {z:.3}]"));
            }
            "phys.substeps" => {
                let Some(value) = parts.next() else {
                    self.console_feedback("usage: phys.substeps <auto|n>");
                    return RuntimeCommand::Consumed;
                };
                if value.eq_ignore_ascii_case("auto") {
                    self.manual_max_substeps = None;
                    self.tick_retune_timer = 0.0;
                    self.console_feedback("substeps override cleared (auto)");
                } else {
                    let substeps = match parse_console_u32_in_range(value, "phys.substeps", 1, 64) {
                        Ok(v) => v,
                        Err(err) => {
                            self.console_feedback(err);
                            return RuntimeCommand::Consumed;
                        }
                    };
                    self.manual_max_substeps = Some(substeps);
                    self.max_substeps = substeps;
                    self.world
                        .set_timestep(1.0 / self.tick_hz.max(1.0), self.max_substeps);
                    self.console_feedback(format!("manual max_substeps={substeps}"));
                }
            }
            "cam.speed" => {
                let Some(value) = parts.next() else {
                    self.console_feedback("usage: cam.speed <1..200>");
                    return RuntimeCommand::Consumed;
                };
                match parse_console_f32_in_range(value, "cam.speed", 1.0, 200.0) {
                    Ok(speed) => {
                        self.camera.set_move_speed(speed);
                        self.console_feedback(format!("camera speed set to {speed:.2}"));
                    }
                    Err(err) => self.console_feedback(err),
                }
            }
            "cam.sens" => {
                let Some(value) = parts.next() else {
                    self.console_feedback("usage: cam.sens <0.0001..0.02>");
                    return RuntimeCommand::Consumed;
                };
                match parse_console_f32_in_range(value, "cam.sens", 0.0001, 0.02) {
                    Ok(sens) => {
                        self.camera.set_mouse_sensitivity(sens);
                        self.console_feedback(format!("camera sensitivity set to {sens:.5}"));
                    }
                    Err(err) => self.console_feedback(err),
                }
            }
            "cam.reset" => {
                self.camera.reset_pose();
                self.console_feedback("camera pose reset");
            }
            "log.clear" => {
                self.console.log_lines.clear();
                self.console.log_scroll = 0;
                self.console_feedback("log buffer cleared");
            }
            "log.tail" => {
                let Some(value) = parts.next() else {
                    self.console_feedback("usage: log.tail <off|n>");
                    return RuntimeCommand::Consumed;
                };
                if value.eq_ignore_ascii_case("off") {
                    self.console.log_tail_limit = None;
                    self.console.log_scroll = 0;
                    self.console_feedback("log tail disabled");
                } else {
                    match parse_console_u32_in_range(value, "log.tail", 1, 320) {
                        Ok(limit) => {
                            self.console.log_tail_limit = Some(limit as usize);
                            self.console.log_scroll =
                                self.console.log_scroll.min(self.max_console_log_scroll());
                            self.console_feedback(format!("log tail set to last {limit} lines"));
                        }
                        Err(err) => self.console_feedback(err),
                    }
                }
            }
            "log.level" => {
                let Some(value) = parts.next() else {
                    self.console_feedback("usage: log.level <all|info|error>");
                    return RuntimeCommand::Consumed;
                };
                let next = if value.eq_ignore_ascii_case("all") {
                    Some(RuntimeConsoleLogFilter::All)
                } else if value.eq_ignore_ascii_case("info") {
                    Some(RuntimeConsoleLogFilter::Info)
                } else if value.eq_ignore_ascii_case("error") {
                    Some(RuntimeConsoleLogFilter::Error)
                } else {
                    None
                };
                match next {
                    Some(filter) => {
                        self.console.log_filter = filter;
                        self.console.log_scroll =
                            self.console.log_scroll.min(self.max_console_log_scroll());
                        self.console_feedback(format!(
                            "log filter set to {}",
                            value.to_ascii_lowercase()
                        ));
                    }
                    None => self.console_feedback("usage: log.level <all|info|error>"),
                }
            }
            "file.exists" => {
                let Some(path_raw) = parts.next() else {
                    self.console_feedback("usage: file.exists <path>");
                    return RuntimeCommand::Consumed;
                };
                let candidate = match self.resolve_console_candidate_path(path_raw) {
                    Ok(path) => path,
                    Err(err) => {
                        self.console_feedback(err);
                        return RuntimeCommand::Consumed;
                    }
                };
                if !candidate.exists() {
                    self.console_feedback(format!("file.exists '{path_raw}' => false"));
                    return RuntimeCommand::Consumed;
                }
                match candidate.canonicalize() {
                    Ok(canonical) => {
                        if !canonical.starts_with(&self.file_io_root) {
                            self.console_feedback(format!(
                                "file.exists denied: '{}' is outside workspace root '{}'",
                                canonical.display(),
                                self.file_io_root.display()
                            ));
                        } else {
                            self.console_feedback(format!(
                                "file.exists '{path_raw}' => true ({})",
                                canonical.display()
                            ));
                        }
                    }
                    Err(err) => {
                        self.console_feedback(format!("file.exists failed for '{path_raw}': {err}"))
                    }
                }
            }
            "file.head" => {
                let Some(path_raw) = parts.next() else {
                    self.console_feedback("usage: file.head <path> [lines]");
                    return RuntimeCommand::Consumed;
                };
                let line_limit = match parts.next() {
                    Some(raw) => match parse_console_usize_in_range(
                        raw,
                        "file.head.lines",
                        1,
                        CONSOLE_FILE_HEAD_MAX_LINES,
                    ) {
                        Ok(limit) => limit,
                        Err(err) => {
                            self.console_feedback(err);
                            return RuntimeCommand::Consumed;
                        }
                    },
                    None => CONSOLE_FILE_HEAD_DEFAULT_LINES,
                };
                let path = match self.resolve_console_existing_path(path_raw, "file.head") {
                    Ok(path) => path,
                    Err(err) => {
                        self.console_feedback(err);
                        return RuntimeCommand::Consumed;
                    }
                };
                if !path.is_file() {
                    self.console_feedback(format!("file.head: '{}' is not a file", path.display()));
                    return RuntimeCommand::Consumed;
                }
                match self.read_file_prefix_bytes(&path, CONSOLE_FILE_HEAD_MAX_WINDOW_BYTES) {
                    Ok((bytes, truncated)) => {
                        self.console_feedback(format!(
                            "file.head '{}' ({} byte(s) window, lines={line_limit}{})",
                            path.display(),
                            bytes.len(),
                            if truncated { ", truncated window" } else { "" }
                        ));
                        let text = String::from_utf8_lossy(&bytes);
                        self.emit_console_text_lines("  ", &text, line_limit);
                    }
                    Err(err) => self.console_feedback(format!("file.head: {err}")),
                }
            }
            "file.tail" => {
                let Some(path_raw) = parts.next() else {
                    self.console_feedback("usage: file.tail <path> [lines] [max_bytes]");
                    return RuntimeCommand::Consumed;
                };
                let line_limit = match parts.next() {
                    Some(raw) => match parse_console_usize_in_range(
                        raw,
                        "file.tail.lines",
                        1,
                        CONSOLE_FILE_TAIL_MAX_LINES,
                    ) {
                        Ok(limit) => limit,
                        Err(err) => {
                            self.console_feedback(err);
                            return RuntimeCommand::Consumed;
                        }
                    },
                    None => CONSOLE_FILE_TAIL_DEFAULT_LINES,
                };
                let max_window_bytes = match parts.next() {
                    Some(raw) => match parse_console_usize_in_range(
                        raw,
                        "file.tail.max_bytes",
                        64,
                        CONSOLE_FILE_TAIL_MAX_WINDOW_BYTES,
                    ) {
                        Ok(limit) => limit,
                        Err(err) => {
                            self.console_feedback(err);
                            return RuntimeCommand::Consumed;
                        }
                    },
                    None => CONSOLE_FILE_TAIL_DEFAULT_WINDOW_BYTES,
                };
                let path = match self.resolve_console_existing_path(path_raw, "file.tail") {
                    Ok(path) => path,
                    Err(err) => {
                        self.console_feedback(err);
                        return RuntimeCommand::Consumed;
                    }
                };
                if !path.is_file() {
                    self.console_feedback(format!("file.tail: '{}' is not a file", path.display()));
                    return RuntimeCommand::Consumed;
                }
                match self.read_file_tail_window(&path, max_window_bytes) {
                    Ok((bytes, truncated_window)) => {
                        let text = String::from_utf8_lossy(&bytes);
                        let lines = text.lines().collect::<Vec<_>>();
                        self.console_feedback(format!(
                            "file.tail '{}' ({} byte(s) window, lines={line_limit}{})",
                            path.display(),
                            bytes.len(),
                            if truncated_window {
                                ", truncated window"
                            } else {
                                ""
                            }
                        ));
                        if lines.is_empty() {
                            self.console_feedback("  <empty>");
                            return RuntimeCommand::Consumed;
                        }
                        let start = lines.len().saturating_sub(line_limit);
                        for line in lines.iter().skip(start) {
                            self.console_feedback(format!("  {line}"));
                        }
                        if lines.len() > line_limit {
                            self.console_feedback(format!(
                                "  ... showing last {line_limit} line(s) from window"
                            ));
                        }
                    }
                    Err(err) => self.console_feedback(format!("file.tail: {err}")),
                }
            }
            "file.find" | "file.findi" => {
                let case_insensitive = matches!(head_lc.as_str(), "file.findi");
                let Some(path_raw) = parts.next() else {
                    self.console_feedback(
                        "usage: file.find <path> <pattern> [max_matches] [max_bytes]",
                    );
                    return RuntimeCommand::Consumed;
                };
                let Some(pattern) = parts.next() else {
                    self.console_feedback(
                        "usage: file.find <path> <pattern> [max_matches] [max_bytes]",
                    );
                    return RuntimeCommand::Consumed;
                };
                let max_matches = match parts.next() {
                    Some(raw) => match parse_console_usize_in_range(
                        raw,
                        if case_insensitive {
                            "file.findi.max_matches"
                        } else {
                            "file.find.max_matches"
                        },
                        1,
                        CONSOLE_FILE_FIND_MAX_MATCHES,
                    ) {
                        Ok(limit) => limit,
                        Err(err) => {
                            self.console_feedback(err);
                            return RuntimeCommand::Consumed;
                        }
                    },
                    None => CONSOLE_FILE_FIND_DEFAULT_MATCHES,
                };
                let max_bytes = match parts.next() {
                    Some(raw) => match parse_console_usize_in_range(
                        raw,
                        if case_insensitive {
                            "file.findi.max_bytes"
                        } else {
                            "file.find.max_bytes"
                        },
                        64,
                        CONSOLE_FILE_FIND_MAX_BYTES,
                    ) {
                        Ok(limit) => limit,
                        Err(err) => {
                            self.console_feedback(err);
                            return RuntimeCommand::Consumed;
                        }
                    },
                    None => CONSOLE_FILE_FIND_DEFAULT_BYTES,
                };
                let path = match self.resolve_console_existing_path(
                    path_raw,
                    if case_insensitive {
                        "file.findi"
                    } else {
                        "file.find"
                    },
                ) {
                    Ok(path) => path,
                    Err(err) => {
                        self.console_feedback(err);
                        return RuntimeCommand::Consumed;
                    }
                };
                if !path.is_file() {
                    self.console_feedback(format!(
                        "{}: '{}' is not a file",
                        if case_insensitive {
                            "file.findi"
                        } else {
                            "file.find"
                        },
                        path.display()
                    ));
                    return RuntimeCommand::Consumed;
                }
                self.run_file_find(&path, pattern, max_matches, max_bytes, case_insensitive);
            }
            "file.findr" => {
                let Some(path_raw) = parts.next() else {
                    self.console_feedback(
                        "usage: file.findr <path> <regex> [max_matches] [max_bytes]",
                    );
                    return RuntimeCommand::Consumed;
                };
                let Some(pattern) = parts.next() else {
                    self.console_feedback(
                        "usage: file.findr <path> <regex> [max_matches] [max_bytes]",
                    );
                    return RuntimeCommand::Consumed;
                };
                let max_matches = match parts.next() {
                    Some(raw) => match parse_console_usize_in_range(
                        raw,
                        "file.findr.max_matches",
                        1,
                        CONSOLE_FILE_FIND_MAX_MATCHES,
                    ) {
                        Ok(limit) => limit,
                        Err(err) => {
                            self.console_feedback(err);
                            return RuntimeCommand::Consumed;
                        }
                    },
                    None => CONSOLE_FILE_FIND_DEFAULT_MATCHES,
                };
                let max_bytes = match parts.next() {
                    Some(raw) => match parse_console_usize_in_range(
                        raw,
                        "file.findr.max_bytes",
                        64,
                        CONSOLE_FILE_FIND_MAX_BYTES,
                    ) {
                        Ok(limit) => limit,
                        Err(err) => {
                            self.console_feedback(err);
                            return RuntimeCommand::Consumed;
                        }
                    },
                    None => CONSOLE_FILE_FIND_DEFAULT_BYTES,
                };
                let path = match self.resolve_console_existing_path(path_raw, "file.findr") {
                    Ok(path) => path,
                    Err(err) => {
                        self.console_feedback(err);
                        return RuntimeCommand::Consumed;
                    }
                };
                if !path.is_file() {
                    self.console_feedback(format!(
                        "file.findr: '{}' is not a file",
                        path.display()
                    ));
                    return RuntimeCommand::Consumed;
                }
                self.run_file_find_regex(&path, pattern, max_matches, max_bytes);
            }
            "file.grep" => {
                let Some(path_raw) = parts.next() else {
                    self.console_feedback(
                        "usage: file.grep <path> <pattern> [context] [max_matches] [max_bytes]",
                    );
                    return RuntimeCommand::Consumed;
                };
                let Some(pattern) = parts.next() else {
                    self.console_feedback(
                        "usage: file.grep <path> <pattern> [context] [max_matches] [max_bytes]",
                    );
                    return RuntimeCommand::Consumed;
                };
                let context = match parts.next() {
                    Some(raw) => match parse_console_usize_in_range(
                        raw,
                        "file.grep.context",
                        0,
                        CONSOLE_FILE_GREP_MAX_CONTEXT,
                    ) {
                        Ok(v) => v,
                        Err(err) => {
                            self.console_feedback(err);
                            return RuntimeCommand::Consumed;
                        }
                    },
                    None => CONSOLE_FILE_GREP_DEFAULT_CONTEXT,
                };
                let max_matches = match parts.next() {
                    Some(raw) => match parse_console_usize_in_range(
                        raw,
                        "file.grep.max_matches",
                        1,
                        CONSOLE_FILE_FIND_MAX_MATCHES,
                    ) {
                        Ok(v) => v,
                        Err(err) => {
                            self.console_feedback(err);
                            return RuntimeCommand::Consumed;
                        }
                    },
                    None => CONSOLE_FILE_FIND_DEFAULT_MATCHES,
                };
                let max_bytes = match parts.next() {
                    Some(raw) => match parse_console_usize_in_range(
                        raw,
                        "file.grep.max_bytes",
                        64,
                        CONSOLE_FILE_FIND_MAX_BYTES,
                    ) {
                        Ok(v) => v,
                        Err(err) => {
                            self.console_feedback(err);
                            return RuntimeCommand::Consumed;
                        }
                    },
                    None => CONSOLE_FILE_FIND_DEFAULT_BYTES,
                };
                let path = match self.resolve_console_existing_path(path_raw, "file.grep") {
                    Ok(path) => path,
                    Err(err) => {
                        self.console_feedback(err);
                        return RuntimeCommand::Consumed;
                    }
                };
                if !path.is_file() {
                    self.console_feedback(format!("file.grep: '{}' is not a file", path.display()));
                    return RuntimeCommand::Consumed;
                }
                self.run_file_grep(&path, pattern, context, max_matches, max_bytes);
            }
            "file.tailf" => {
                let Some(arg) = parts.next() else {
                    if let Some(follow) = self.console.tail_follow.as_ref() {
                        self.console_feedback(format!(
                            "tailf active path='{}' poll={}ms lines/poll={}",
                            follow.path.display(),
                            follow.poll_interval.as_millis(),
                            follow.max_lines_per_poll
                        ));
                    } else {
                        self.console_feedback(
                            "usage: file.tailf <path>|stop [poll_ms] [max_lines]",
                        );
                    }
                    return RuntimeCommand::Consumed;
                };
                if arg.eq_ignore_ascii_case("stop") || arg.eq_ignore_ascii_case("off") {
                    if let Some(prev) = self.console.tail_follow.take() {
                        self.console_feedback(format!(
                            "tailf stopped for '{}'",
                            prev.path.display()
                        ));
                    } else {
                        self.console_feedback("tailf already inactive");
                    }
                    return RuntimeCommand::Consumed;
                }

                let poll_ms = match parts.next() {
                    Some(raw) => {
                        match parse_console_usize_in_range(raw, "file.tailf.poll_ms", 50, 10_000) {
                            Ok(v) => v as u64,
                            Err(err) => {
                                self.console_feedback(err);
                                return RuntimeCommand::Consumed;
                            }
                        }
                    }
                    None => CONSOLE_FILE_TAILF_DEFAULT_POLL_MS,
                };
                let max_lines_per_poll = match parts.next() {
                    Some(raw) => match parse_console_usize_in_range(
                        raw,
                        "file.tailf.max_lines",
                        1,
                        CONSOLE_FILE_TAILF_MAX_LINES_PER_POLL,
                    ) {
                        Ok(v) => v,
                        Err(err) => {
                            self.console_feedback(err);
                            return RuntimeCommand::Consumed;
                        }
                    },
                    None => CONSOLE_FILE_TAILF_MAX_LINES_PER_POLL / 2,
                };
                let path = match self.resolve_console_existing_path(arg, "file.tailf") {
                    Ok(path) => path,
                    Err(err) => {
                        self.console_feedback(err);
                        return RuntimeCommand::Consumed;
                    }
                };
                if !path.is_file() {
                    self.console_feedback(format!(
                        "file.tailf: '{}' is not a file",
                        path.display()
                    ));
                    return RuntimeCommand::Consumed;
                }
                let start_cursor = match fs::metadata(&path) {
                    Ok(meta) => meta.len(),
                    Err(err) => {
                        self.console_feedback(format!(
                            "file.tailf: failed to read metadata '{}': {err}",
                            path.display()
                        ));
                        return RuntimeCommand::Consumed;
                    }
                };
                self.console.tail_follow = Some(RuntimeConsoleTailFollow {
                    path: path.clone(),
                    poll_interval: Duration::from_millis(poll_ms),
                    max_lines_per_poll,
                    cursor: start_cursor,
                    partial_line: String::new(),
                    last_poll: Instant::now(),
                    last_error: None,
                });
                self.console_feedback(format!(
                    "tailf started for '{}' poll={}ms max_lines={}",
                    path.display(),
                    poll_ms,
                    max_lines_per_poll
                ));
            }
            "file.watch" => {
                let Some(arg) = parts.next() else {
                    if let Some(watch) = self.console.file_watch.as_ref() {
                        self.console_feedback(format!(
                            "watch active path='{}' poll={}ms size={}",
                            watch.path.display(),
                            watch.poll_interval.as_millis(),
                            watch.last_len
                        ));
                    } else {
                        self.console_feedback("usage: file.watch <path>|stop [poll_ms]");
                    }
                    return RuntimeCommand::Consumed;
                };
                if arg.eq_ignore_ascii_case("stop") || arg.eq_ignore_ascii_case("off") {
                    if let Some(prev) = self.console.file_watch.take() {
                        self.console_feedback(format!(
                            "watch stopped for '{}'",
                            prev.path.display()
                        ));
                    } else {
                        self.console_feedback("watch already inactive");
                    }
                    return RuntimeCommand::Consumed;
                }

                let poll_ms = match parts.next() {
                    Some(raw) => {
                        match parse_console_usize_in_range(raw, "file.watch.poll_ms", 50, 10_000) {
                            Ok(v) => v as u64,
                            Err(err) => {
                                self.console_feedback(err);
                                return RuntimeCommand::Consumed;
                            }
                        }
                    }
                    None => CONSOLE_FILE_WATCH_DEFAULT_POLL_MS,
                };
                let path = match self.resolve_console_existing_path(arg, "file.watch") {
                    Ok(path) => path,
                    Err(err) => {
                        self.console_feedback(err);
                        return RuntimeCommand::Consumed;
                    }
                };
                if !path.is_file() {
                    self.console_feedback(format!(
                        "file.watch: '{}' is not a file",
                        path.display()
                    ));
                    return RuntimeCommand::Consumed;
                }
                let (modified, len) = match fs::metadata(&path) {
                    Ok(meta) => (meta.modified().ok(), meta.len()),
                    Err(err) => {
                        self.console_feedback(format!(
                            "file.watch: failed to read metadata '{}': {err}",
                            path.display()
                        ));
                        return RuntimeCommand::Consumed;
                    }
                };
                self.console.file_watch = Some(RuntimeConsoleFileWatch {
                    path: path.clone(),
                    poll_interval: Duration::from_millis(poll_ms),
                    last_modified: modified,
                    last_len: len,
                    last_poll: Instant::now(),
                    last_error: None,
                });
                self.console_feedback(format!(
                    "watch started for '{}' poll={}ms baseline_size={}",
                    path.display(),
                    poll_ms,
                    len
                ));
            }
            "file.read" => {
                let Some(path_raw) = parts.next() else {
                    self.console_feedback("usage: file.read <path> [max_bytes]");
                    return RuntimeCommand::Consumed;
                };
                let max_bytes = match parts.next() {
                    Some(raw) => match parse_console_usize_in_range(
                        raw,
                        "file.read.max_bytes",
                        64,
                        CONSOLE_FILE_READ_MAX_BYTES,
                    ) {
                        Ok(limit) => limit,
                        Err(err) => {
                            self.console_feedback(err);
                            return RuntimeCommand::Consumed;
                        }
                    },
                    None => CONSOLE_FILE_READ_DEFAULT_BYTES,
                };
                let path = match self.resolve_console_existing_path(path_raw, "file.read") {
                    Ok(path) => path,
                    Err(err) => {
                        self.console_feedback(err);
                        return RuntimeCommand::Consumed;
                    }
                };
                if !path.is_file() {
                    self.console_feedback(format!("file.read: '{}' is not a file", path.display()));
                    return RuntimeCommand::Consumed;
                }
                match self.read_file_prefix_bytes(&path, max_bytes) {
                    Ok((buf, truncated)) => {
                        let preview = String::from_utf8_lossy(&buf);
                        self.console_feedback(format!(
                            "file.read '{}' (preview {} byte(s), limit {max_bytes}{})",
                            path.display(),
                            buf.len(),
                            if truncated { ", truncated" } else { "" }
                        ));
                        self.emit_console_text_lines(
                            "  ",
                            &preview,
                            CONSOLE_FILE_HEAD_DEFAULT_LINES,
                        );
                    }
                    Err(err) => self.console_feedback(format!("file.read: {err}")),
                }
            }
            "file.list" => {
                let dir_raw = parts.next().unwrap_or(".");
                let limit = match parts.next() {
                    Some(raw) => match parse_console_usize_in_range(
                        raw,
                        "file.list.limit",
                        1,
                        CONSOLE_FILE_LIST_MAX_LIMIT,
                    ) {
                        Ok(limit) => limit,
                        Err(err) => {
                            self.console_feedback(err);
                            return RuntimeCommand::Consumed;
                        }
                    },
                    None => CONSOLE_FILE_LIST_DEFAULT_LIMIT,
                };
                let dir = match self.resolve_console_existing_path(dir_raw, "file.list") {
                    Ok(path) => path,
                    Err(err) => {
                        self.console_feedback(err);
                        return RuntimeCommand::Consumed;
                    }
                };
                if !dir.is_dir() {
                    self.console_feedback(format!(
                        "file.list: '{}' is not a directory",
                        dir.display()
                    ));
                    return RuntimeCommand::Consumed;
                }
                let mut entries = match fs::read_dir(&dir) {
                    Ok(read_dir) => read_dir
                        .filter_map(|entry| entry.ok())
                        .map(|entry| {
                            let path = entry.path();
                            let name = entry.file_name().to_string_lossy().to_string();
                            let is_dir = path.is_dir();
                            (name, is_dir)
                        })
                        .collect::<Vec<_>>(),
                    Err(err) => {
                        self.console_feedback(format!(
                            "file.list failed for '{}': {err}",
                            dir.display()
                        ));
                        return RuntimeCommand::Consumed;
                    }
                };
                entries.sort_by(|a, b| a.0.cmp(&b.0));
                self.console_feedback(format!(
                    "file.list '{}' total={} showing={}",
                    dir.display(),
                    entries.len(),
                    entries.len().min(limit)
                ));
                for (name, is_dir) in entries.iter().take(limit) {
                    self.console_feedback(format!(
                        "  [{}] {}",
                        if *is_dir { "d" } else { "f" },
                        name
                    ));
                }
                if entries.len() > limit {
                    self.console_feedback(format!(
                        "  ... {} more entrie(s) hidden",
                        entries.len() - limit
                    ));
                }
            }
            "clear" => {
                self.console.last_feedback.clear();
                self.console_feedback("console status cleared");
            }
            "exit" | "quit" => {
                self.console_feedback("exit requested from console");
                return RuntimeCommand::Exit;
            }
            "gfx.vsync" => {
                let Some(value) = parts.next() else {
                    self.console_feedback("usage: gfx.vsync <auto|on|off>");
                    return RuntimeCommand::Consumed;
                };
                match VsyncMode::parse(value) {
                    Ok(mode) => {
                        let present_mode = match mode {
                            VsyncMode::Auto => wgpu::PresentMode::AutoVsync,
                            VsyncMode::On => wgpu::PresentMode::Fifo,
                            VsyncMode::Off => wgpu::PresentMode::AutoNoVsync,
                        };
                        self.apply_present_mode(present_mode);
                        self.console_feedback(format!("vsync set to {mode:?} ({present_mode:?})"));
                    }
                    Err(err) => self.console_feedback(err.to_string()),
                }
            }
            "gfx.fps_cap" => {
                let Some(value) = parts.next() else {
                    self.console_feedback("usage: gfx.fps_cap <off|N>");
                    return RuntimeCommand::Consumed;
                };
                match parse_fps_cap(value) {
                    Ok(cap) => {
                        self.apply_fps_cap_runtime(cap);
                        if let Some(fps) = cap {
                            self.console_feedback(format!("fps cap set to {fps:.1}"));
                        } else {
                            self.console_feedback("fps cap disabled".to_string());
                        }
                    }
                    Err(err) => self.console_feedback(err.to_string()),
                }
            }
            "gfx.rt" => {
                let Some(value) = parts.next() else {
                    self.console_feedback("usage: gfx.rt <off|auto|on>");
                    return RuntimeCommand::Consumed;
                };
                let Some(mode) = RayTracingMode::from_str(value) else {
                    self.console_feedback("invalid gfx.rt value (expected off|auto|on)");
                    return RuntimeCommand::Consumed;
                };
                self.rt_mode = mode;
                self.renderer.set_ray_tracing_mode(&self.queue, mode);
                let status = self.renderer.ray_tracing_status();
                self.console_feedback(format!(
                    "rt set to {:?} (active={}, reason={})",
                    mode,
                    status.active,
                    if status.fallback_reason.is_empty() {
                        "none"
                    } else {
                        status.fallback_reason.as_str()
                    }
                ));
            }
            "gfx.fsr" => {
                let Some(value) = parts.next() else {
                    self.console_feedback("usage: gfx.fsr <off|auto|on>");
                    return RuntimeCommand::Consumed;
                };
                let Some(mode) = FsrMode::parse(value) else {
                    self.console_feedback("invalid gfx.fsr value (expected off|auto|on)");
                    return RuntimeCommand::Consumed;
                };
                self.fsr_config.mode = mode;
                self.renderer.set_fsr_config(&self.queue, self.fsr_config);
                self.console_feedback(format!("fsr mode set to {:?}", mode));
            }
            "gfx.fsr_quality" => {
                let Some(value) = parts.next() else {
                    self.console_feedback(
                        "usage: gfx.fsr_quality <native|ultra|quality|balanced|performance>",
                    );
                    return RuntimeCommand::Consumed;
                };
                let Some(quality) = FsrQualityPreset::parse(value) else {
                    self.console_feedback("invalid gfx.fsr_quality preset");
                    return RuntimeCommand::Consumed;
                };
                self.fsr_config.quality = quality;
                self.renderer.set_fsr_config(&self.queue, self.fsr_config);
                self.console_feedback(format!("fsr quality set to {:?}", quality));
            }
            "gfx.fsr_sharpness" => {
                let Some(value) = parts.next() else {
                    self.console_feedback("usage: gfx.fsr_sharpness <0..1>");
                    return RuntimeCommand::Consumed;
                };
                match parse_fsr_sharpness(value) {
                    Ok(sharpness) => {
                        self.fsr_config.sharpness = sharpness;
                        self.renderer.set_fsr_config(&self.queue, self.fsr_config);
                        self.console_feedback(format!("fsr sharpness set to {sharpness:.2}"));
                    }
                    Err(err) => self.console_feedback(err.to_string()),
                }
            }
            "gfx.fsr_scale" => {
                let Some(value) = parts.next() else {
                    self.console_feedback("usage: gfx.fsr_scale <auto|0.5..1>");
                    return RuntimeCommand::Consumed;
                };
                match parse_fsr_scale(value) {
                    Ok(scale) => {
                        self.fsr_config.render_scale_override = scale;
                        self.renderer.set_fsr_config(&self.queue, self.fsr_config);
                        match scale {
                            Some(v) => self.console_feedback(format!("fsr scale override={v:.2}")),
                            None => self.console_feedback("fsr scale override=auto"),
                        }
                    }
                    Err(err) => self.console_feedback(err.to_string()),
                }
            }
            "gfx.profile" => {
                let Some(value) = parts.next() else {
                    self.console_feedback("usage: gfx.profile <low|med|high|ultra>");
                    return RuntimeCommand::Consumed;
                };
                let profile = value.to_ascii_lowercase();
                match self.apply_gfx_profile(profile.as_str()) {
                    Ok(note) => self.console_feedback(note),
                    Err(err) => self.console_feedback(err),
                }
            }
            "gfx.render_distance" => {
                let Some(value) = parts.next() else {
                    self.console_feedback("usage: gfx.render_distance <off|N>");
                    return RuntimeCommand::Consumed;
                };
                match parse_render_distance(value) {
                    Ok(distance) => {
                        self.render_distance = distance;
                        if let Some(value) = distance {
                            self.render_distance_min = (value * 0.72).clamp(28.0, value);
                            self.render_distance_max =
                                (value * 1.55).clamp(value, value.max(220.0));
                            self.console_feedback(format!("render distance set to {value:.1}"));
                        } else {
                            self.console_feedback("render distance disabled");
                        }
                    }
                    Err(err) => self.console_feedback(err.to_string()),
                }
            }
            "gfx.adaptive_distance" => {
                let Some(value) = parts.next() else {
                    self.console_feedback("usage: gfx.adaptive_distance <auto|on|off>");
                    return RuntimeCommand::Consumed;
                };
                match ToggleAuto::parse(value, "gfx.adaptive_distance") {
                    Ok(mode) => {
                        self.adaptive_distance_enabled = mode.resolve(true);
                        self.console_feedback(format!(
                            "adaptive distance mode={mode:?} enabled={}",
                            self.adaptive_distance_enabled
                        ));
                    }
                    Err(err) => self.console_feedback(err.to_string()),
                }
            }
            "gfx.distance_blur" => {
                let Some(value) = parts.next() else {
                    self.console_feedback("usage: gfx.distance_blur <auto|on|off>");
                    return RuntimeCommand::Consumed;
                };
                match ToggleAuto::parse(value, "gfx.distance_blur") {
                    Ok(mode) => {
                        self.distance_blur_mode = mode;
                        self.distance_blur_enabled = mode.resolve(false);
                        self.console_feedback(format!(
                            "distance blur mode={mode:?} enabled={}",
                            self.distance_blur_enabled
                        ));
                    }
                    Err(err) => self.console_feedback(err.to_string()),
                }
            }
            "script.var" => {
                let mut tokens = trimmed.splitn(3, char::is_whitespace);
                let _ = tokens.next();
                let Some(name) = tokens.next() else {
                    self.console_feedback("usage: script.var <name> <expr>");
                    return RuntimeCommand::Consumed;
                };
                if !is_valid_tlscript_ident(name) {
                    self.console_feedback(format!("invalid variable name '{name}'"));
                    return RuntimeCommand::Consumed;
                }
                let value = tokens.next().unwrap_or("").trim();
                if value.is_empty() {
                    self.console_feedback("usage: script.var <name> <expr>");
                    return RuntimeCommand::Consumed;
                }
                self.console
                    .script_vars
                    .insert(name.to_string(), value.to_string());
                match self.rebuild_console_script_overlay() {
                    Ok(notes) => {
                        self.console_feedback(format!("script var '{name}' set to '{value}'"));
                        for note in notes {
                            self.console_feedback(note);
                        }
                    }
                    Err(err) => self.console_feedback(err),
                }
            }
            "script.unset" => {
                let Some(name) = parts.next() else {
                    self.console_feedback("usage: script.unset <name>");
                    return RuntimeCommand::Consumed;
                };
                self.console.script_vars.remove(name);
                match self.rebuild_console_script_overlay() {
                    Ok(notes) => {
                        self.console_feedback(format!("script var '{name}' removed"));
                        for note in notes {
                            self.console_feedback(note);
                        }
                    }
                    Err(err) => self.console_feedback(err),
                }
            }
            "script.vars" => {
                if self.console.script_vars.is_empty() {
                    self.console_feedback("script vars: <empty>");
                } else {
                    let vars = self
                        .console
                        .script_vars
                        .iter()
                        .map(|(k, v)| format!("{k}={v}"))
                        .collect::<Vec<_>>()
                        .join(", ");
                    self.console_feedback(format!("script vars: {vars}"));
                }
            }
            "script.call" | "script.exec" => {
                let statement_raw = trimmed
                    .split_once(char::is_whitespace)
                    .map(|(_, tail)| tail.trim())
                    .unwrap_or("");
                if statement_raw.is_empty() {
                    self.console_feedback(format!("usage: {head_lc} <statement>"));
                    return RuntimeCommand::Consumed;
                }
                let statement_template = if head_lc == "script.call" {
                    match Self::normalize_script_call_statement(statement_raw) {
                        Ok(v) => v,
                        Err(err) => {
                            self.console_feedback(err);
                            return RuntimeCommand::Consumed;
                        }
                    }
                } else {
                    statement_raw.trim_end_matches(';').to_string()
                };
                self.console
                    .script_statements
                    .push(statement_template.clone());
                match self.rebuild_console_script_overlay() {
                    Ok(notes) => {
                        self.console_feedback(format!(
                            "script statement added [{}]: {}",
                            self.console.script_statements.len() - 1,
                            statement_template
                        ));
                        for note in notes {
                            self.console_feedback(note);
                        }
                    }
                    Err(err) => {
                        let _ = self.console.script_statements.pop();
                        self.console_feedback(err);
                    }
                }
            }
            "script.uncall" => {
                let Some(target) = parts.next() else {
                    self.console_feedback("usage: script.uncall <index|all>");
                    return RuntimeCommand::Consumed;
                };
                if target.eq_ignore_ascii_case("all") {
                    self.console.script_statements.clear();
                    self.console.script_overlay = empty_showcase_output();
                    self.console_feedback("all script statements removed");
                    return RuntimeCommand::Consumed;
                }
                let Ok(index) = target.parse::<usize>() else {
                    self.console_feedback("script.uncall expects numeric index or 'all'");
                    return RuntimeCommand::Consumed;
                };
                if index >= self.console.script_statements.len() {
                    self.console_feedback(format!(
                        "script.uncall index out of range (0..{})",
                        self.console.script_statements.len().saturating_sub(1)
                    ));
                    return RuntimeCommand::Consumed;
                }
                let removed = self.console.script_statements.remove(index);
                match self.rebuild_console_script_overlay() {
                    Ok(notes) => {
                        self.console_feedback(format!(
                            "removed script statement[{index}]: {removed}"
                        ));
                        for note in notes {
                            self.console_feedback(note);
                        }
                    }
                    Err(err) => self.console_feedback(err),
                }
            }
            "script.list" => {
                if self.console.script_statements.is_empty() {
                    self.console_feedback("script statements: <empty>");
                } else {
                    let statements = self.console.script_statements.clone();
                    for (index, statement) in statements.iter().enumerate() {
                        self.console_feedback(format!("[{index}] {statement}"));
                    }
                }
            }
            "script.clear" => {
                self.console.script_vars.clear();
                self.console.script_statements.clear();
                self.console.script_overlay = empty_showcase_output();
                self.console_feedback("script vars and statements cleared");
            }
            _ => {
                self.console_feedback(format!("unknown command '{head}' (run 'help')"));
            }
        }
        RuntimeCommand::Consumed
    }

    fn on_console_keyboard_input(&mut self, event: &KeyEvent) -> RuntimeCommand {
        if event.state != ElementState::Pressed {
            return RuntimeCommand::Consumed;
        }

        match event.physical_key {
            PhysicalKey::Code(KeyCode::Escape) => {
                self.toggle_console();
                return RuntimeCommand::Consumed;
            }
            PhysicalKey::Code(KeyCode::Tab) => {
                self.console.edit_target = self.console.edit_target.next();
                self.console_feedback(format!(
                    "active edit box: {}",
                    self.console.edit_target.label()
                ));
                return RuntimeCommand::Consumed;
            }
            PhysicalKey::Code(KeyCode::Enter) | PhysicalKey::Code(KeyCode::NumpadEnter) => {
                return self.apply_active_console_edit_target();
            }
            PhysicalKey::Code(KeyCode::Backspace) => {
                self.selected_console_edit_buffer_mut().pop();
                return RuntimeCommand::Consumed;
            }
            PhysicalKey::Code(KeyCode::ArrowUp) => {
                if self.console.edit_target != RuntimeConsoleEditTarget::Command {
                    return RuntimeCommand::Consumed;
                }
                if self.console.history.is_empty() {
                    return RuntimeCommand::Consumed;
                }
                let next = match self.console.history_cursor {
                    None => self.console.history.len().saturating_sub(1),
                    Some(cur) => cur.saturating_sub(1),
                };
                self.console.history_cursor = Some(next);
                self.console.input_line = self.console.history[next].clone();
                return RuntimeCommand::Consumed;
            }
            PhysicalKey::Code(KeyCode::ArrowDown) => {
                if self.console.edit_target != RuntimeConsoleEditTarget::Command {
                    return RuntimeCommand::Consumed;
                }
                if self.console.history.is_empty() {
                    return RuntimeCommand::Consumed;
                }
                let Some(cur) = self.console.history_cursor else {
                    return RuntimeCommand::Consumed;
                };
                if cur + 1 >= self.console.history.len() {
                    self.console.history_cursor = None;
                    self.console.input_line.clear();
                } else {
                    let next = cur + 1;
                    self.console.history_cursor = Some(next);
                    self.console.input_line = self.console.history[next].clone();
                }
                return RuntimeCommand::Consumed;
            }
            PhysicalKey::Code(KeyCode::PageUp) => {
                self.scroll_console_logs(4);
                return RuntimeCommand::Consumed;
            }
            PhysicalKey::Code(KeyCode::PageDown) => {
                self.scroll_console_logs(-4);
                return RuntimeCommand::Consumed;
            }
            PhysicalKey::Code(KeyCode::Home) => {
                self.console.log_scroll = self.max_console_log_scroll();
                return RuntimeCommand::Consumed;
            }
            PhysicalKey::Code(KeyCode::End) => {
                self.console.log_scroll = 0;
                return RuntimeCommand::Consumed;
            }
            _ => {}
        }

        if self.keyboard_modifiers.control_key()
            || self.keyboard_modifiers.alt_key()
            || self.keyboard_modifiers.super_key()
        {
            return RuntimeCommand::Consumed;
        }

        if let Some(text) = &event.text {
            for ch in text.chars() {
                if !ch.is_control() {
                    self.selected_console_edit_buffer_mut().push(ch);
                }
            }
        }
        RuntimeCommand::Consumed
    }

    fn console_glyph_slot(ch: char) -> u16 {
        let code = ch as u32;
        if (32..=126).contains(&code) {
            CONSOLE_TEXT_SLOT_BASE + (code as u16 - 32)
        } else {
            CONSOLE_TEXT_SLOT_BASE + (u16::from(b'?') - 32)
        }
    }

    fn push_console_text_line(
        sprites: &mut Vec<SpriteInstance>,
        sprite_id_seed: &mut u64,
        text: &str,
        x: f32,
        y: f32,
        z: f32,
        glyph_size: [f32; 2],
        color: [f32; 4],
        layer: i16,
    ) {
        let mut cursor_x = x;
        for ch in text.chars() {
            if ch == '\t' {
                cursor_x += glyph_size[0] * 4.0;
                continue;
            }
            if ch == ' ' {
                cursor_x += glyph_size[0];
                continue;
            }
            sprites.push(SpriteInstance {
                sprite_id: *sprite_id_seed,
                kind: SpriteKind::Generic,
                position: [cursor_x, y, z],
                size: glyph_size,
                rotation_rad: 0.0,
                color_rgba: color,
                texture_slot: Self::console_glyph_slot(ch),
                layer,
            });
            *sprite_id_seed = sprite_id_seed.saturating_add(1);
            cursor_x += glyph_size[0] * 0.86;
        }
    }

    fn push_console_rect(
        sprites: &mut Vec<SpriteInstance>,
        sprite_id_seed: &mut u64,
        pos: [f32; 3],
        size: [f32; 2],
        color: [f32; 4],
        layer: i16,
    ) {
        sprites.push(SpriteInstance {
            sprite_id: *sprite_id_seed,
            kind: SpriteKind::Hud,
            position: pos,
            size,
            rotation_rad: 0.0,
            color_rgba: color,
            texture_slot: 1,
            layer,
        });
        *sprite_id_seed = sprite_id_seed.saturating_add(1);
    }

    fn append_console_overlay_sprites(
        console: &RuntimeConsoleState,
        layout: ConsoleUiLayout,
        sprites: &mut Vec<SpriteInstance>,
    ) {
        if !console.open {
            return;
        }
        let mut sprite_id_seed = 9_200_000u64 + sprites.len() as u64;

        // Full-screen semi-transparent console shell.
        Self::push_console_rect(
            sprites,
            &mut sprite_id_seed,
            [0.0, 0.0, 0.99],
            [2.0, 2.0],
            [0.02, 0.08, 0.03, 0.80],
            29_000,
        );
        Self::push_console_rect(
            sprites,
            &mut sprite_id_seed,
            layout.pos(0.0, 0.80, 0.98),
            layout.rect_size(1.92, 0.26),
            [0.06, 0.20, 0.09, 0.92],
            29_001,
        );
        Self::push_console_rect(
            sprites,
            &mut sprite_id_seed,
            layout.pos(-0.50, 0.37, 0.98),
            layout.rect_size(0.92, 0.76),
            [0.04, 0.12, 0.05, 0.92],
            29_001,
        );
        Self::push_console_rect(
            sprites,
            &mut sprite_id_seed,
            layout.pos(0.52, 0.37, 0.98),
            layout.rect_size(0.90, 0.76),
            [0.04, 0.12, 0.05, 0.92],
            29_001,
        );
        Self::push_console_rect(
            sprites,
            &mut sprite_id_seed,
            layout.pos(0.0, -0.72, 0.98),
            layout.rect_size(1.92, 0.44),
            [0.03, 0.10, 0.04, 0.94],
            29_001,
        );
        // Input/editable box.
        Self::push_console_rect(
            sprites,
            &mut sprite_id_seed,
            [layout.command_center.0, layout.command_center.1, 0.97],
            [layout.command_size.0, layout.command_size.1],
            [0.02, 0.18, 0.05, 0.96],
            29_002,
        );
        // Send button (right side of command input).
        Self::push_console_rect(
            sprites,
            &mut sprite_id_seed,
            [layout.send_center.0, layout.send_center.1, 0.97],
            [layout.send_size.0, layout.send_size.1],
            [0.04, 0.28, 0.08, 0.98],
            29_003,
        );

        let header_color = [0.74, 1.0, 0.72, 1.0];
        let info_color = [0.62, 0.96, 0.62, 1.0];
        let mut y = 0.90;
        Self::push_console_text_line(
            sprites,
            &mut sprite_id_seed,
            "TLAPP CLI OVERLAY",
            -0.93 * layout.sx,
            y * layout.sy,
            0.96,
            layout.glyph_size(0.028, 0.046),
            header_color,
            29_010,
        );
        y -= 0.065;
        Self::push_console_text_line(
            sprites,
            &mut sprite_id_seed,
            "GREEN TEXT | FULLSCREEN SEMI-TRANSPARENT | ERRORS BLINK RED",
            -0.93 * layout.sx,
            y * layout.sy,
            0.96,
            layout.glyph_size(0.018, 0.032),
            info_color,
            29_010,
        );
        y -= 0.055;
        Self::push_console_text_line(
            sprites,
            &mut sprite_id_seed,
            "HOTKEYS: CTRL+F1/F1 TOGGLE | ENTER RUN | ARROW UP/DOWN HISTORY",
            -0.93 * layout.sx,
            y * layout.sy,
            0.96,
            layout.glyph_size(0.018, 0.032),
            info_color,
            29_010,
        );
        y -= 0.055;
        Self::push_console_text_line(
            sprites,
            &mut sprite_id_seed,
            "LOG SCROLL: WHEEL / PGUP/PGDN | HOME/END",
            -0.93 * layout.sx,
            y * layout.sy,
            0.96,
            layout.glyph_size(0.016, 0.029),
            info_color,
            29_010,
        );
        y -= 0.055;
        Self::push_console_text_line(
            sprites,
            &mut sprite_id_seed,
            &format!(
                "TAB SWITCH BOX | ACTIVE: {}",
                console.edit_target.label().to_uppercase()
            ),
            -0.93 * layout.sx,
            y * layout.sy,
            0.96,
            layout.glyph_size(0.018, 0.032),
            [0.84, 0.98, 0.74, 1.0],
            29_010,
        );

        // Command list box.
        Self::push_console_text_line(
            sprites,
            &mut sprite_id_seed,
            "COMMAND LIST",
            -0.93 * layout.sx,
            0.63 * layout.sy,
            0.96,
            layout.glyph_size(0.02, 0.036),
            header_color,
            29_010,
        );
        let mut cmd_y = 0.58;
        for cmd in CONSOLE_HELP_COMMANDS.iter().take(14) {
            Self::push_console_text_line(
                sprites,
                &mut sprite_id_seed,
                cmd,
                -0.93 * layout.sx,
                cmd_y * layout.sy,
                0.96,
                layout.glyph_size(0.016, 0.029),
                [0.58, 0.92, 0.58, 1.0],
                29_010,
            );
            cmd_y -= 0.044;
        }

        // Quick settings / editable values.
        let quick_left = 0.08;
        Self::push_console_text_line(
            sprites,
            &mut sprite_id_seed,
            "EDIT BOXES",
            quick_left * layout.sx,
            0.63 * layout.sy,
            0.96,
            layout.glyph_size(0.02, 0.036),
            header_color,
            29_010,
        );
        let settings = [
            format!("FPS CAP: {}", console.quick_fps_cap),
            format!("RENDER DISTANCE: {}", console.quick_render_distance),
            format!("FSR SHARPNESS: {}", console.quick_fsr_sharpness),
        ];
        let active_fps = console.edit_target == RuntimeConsoleEditTarget::FpsCap;
        let active_rd = console.edit_target == RuntimeConsoleEditTarget::RenderDistance;
        let active_sharp = console.edit_target == RuntimeConsoleEditTarget::FsrSharpness;
        let mut settings_y = 0.57;
        for (index, line) in settings.into_iter().enumerate() {
            let is_active = match index {
                0 => active_fps,
                1 => active_rd,
                _ => active_sharp,
            };
            Self::push_console_text_line(
                sprites,
                &mut sprite_id_seed,
                &if is_active { format!("> {line}") } else { line },
                quick_left * layout.sx,
                settings_y * layout.sy,
                0.96,
                layout.glyph_size(0.018, 0.032),
                if is_active {
                    [0.90, 1.00, 0.82, 1.0]
                } else {
                    [0.64, 0.98, 0.64, 1.0]
                },
                29_010,
            );
            settings_y -= 0.055;
        }
        Self::push_console_text_line(
            sprites,
            &mut sprite_id_seed,
            "BOX INPUT: TYPE VALUE + COMMAND (GFX.FPS_CAP / GFX.RENDER_DISTANCE ...)",
            quick_left * layout.sx,
            0.39 * layout.sy,
            0.96,
            layout.glyph_size(0.014, 0.026),
            [0.52, 0.86, 0.52, 1.0],
            29_010,
        );
        Self::push_console_rect(
            sprites,
            &mut sprite_id_seed,
            [layout.apply_center.0, layout.apply_center.1, 0.97],
            [layout.apply_size.0, layout.apply_size.1],
            [0.03, 0.22, 0.06, 0.96],
            29_002,
        );
        Self::push_console_text_line(
            sprites,
            &mut sprite_id_seed,
            "[CLICK] APPLY ALL BOXES",
            0.18 * layout.sx,
            0.32 * layout.sy,
            0.96,
            layout.glyph_size(0.016, 0.028),
            [0.84, 1.0, 0.84, 1.0],
            29_011,
        );
        Self::push_console_text_line(
            sprites,
            &mut sprite_id_seed,
            "SEND",
            layout.send_center.0 - layout.send_size.0 * 0.22,
            layout.send_center.1,
            0.96,
            layout.glyph_size(0.020, 0.033),
            [0.90, 1.0, 0.88, 1.0],
            29_021,
        );

        // Live input line.
        let glyph = layout.glyph_size(0.022, 0.038);
        let max_chars = ((layout.command_size.0 * 0.90) / (glyph[0] * 0.86))
            .floor()
            .max(8.0) as usize;
        let mut input_tail = console.input_line.clone();
        let input_len = input_tail.chars().count();
        if input_len > max_chars.saturating_sub(2) {
            let keep = max_chars.saturating_sub(3);
            input_tail = input_tail
                .chars()
                .rev()
                .take(keep)
                .collect::<String>()
                .chars()
                .rev()
                .collect::<String>();
            input_tail = format!("...{input_tail}");
        }
        let input_line = format!("> {}", input_tail);
        let input_active = console.edit_target == RuntimeConsoleEditTarget::Command;
        Self::push_console_text_line(
            sprites,
            &mut sprite_id_seed,
            &input_line,
            layout.command_center.0 - layout.command_size.0 * 0.48,
            layout.command_center.1,
            0.96,
            glyph,
            if input_active {
                [0.90, 1.0, 0.90, 1.0]
            } else {
                [0.66, 0.90, 0.66, 1.0]
            },
            29_020,
        );

        // Output list with blinking red errors.
        Self::push_console_text_line(
            sprites,
            &mut sprite_id_seed,
            "OUTPUT",
            -0.93 * layout.sx,
            -0.66 * layout.sy,
            0.96,
            layout.glyph_size(0.02, 0.036),
            header_color,
            29_010,
        );
        let now = Instant::now();
        let line_step = (0.047 * layout.text_scale).max(0.026);
        let out_top = -0.72f32;
        let out_bottom = -0.90f32;
        let visible_logs = (((out_top - out_bottom) / line_step).floor() as usize).max(3);
        let filtered_logs = console
            .log_lines
            .iter()
            .filter(|line| console.log_filter.matches(line.level))
            .collect::<Vec<_>>();
        let total_logs = filtered_logs.len();
        let tail_start = if let Some(limit) = console.log_tail_limit {
            total_logs.saturating_sub(limit)
        } else {
            0
        };
        let end = total_logs.saturating_sub(console.log_scroll);
        let start = end.saturating_sub(visible_logs).max(tail_start);
        let mut out_y = out_top;
        for line in filtered_logs
            .iter()
            .skip(start)
            .take(end.saturating_sub(start))
        {
            let age_ms = now.saturating_duration_since(line.timestamp).as_millis();
            let color = match line.level {
                RuntimeConsoleLogLevel::Info => [0.70, 0.98, 0.70, 1.0],
                RuntimeConsoleLogLevel::Error => {
                    if ((age_ms / CONSOLE_ERROR_BLINK_MS) % 2) == 0 {
                        [1.0, 0.32, 0.32, 1.0]
                    } else {
                        [0.68, 0.12, 0.12, 1.0]
                    }
                }
            };
            let text = format!("[{:>5}ms] {}", age_ms.min(99_999), line.message);
            Self::push_console_text_line(
                sprites,
                &mut sprite_id_seed,
                &text,
                -0.93 * layout.sx,
                out_y * layout.sy,
                0.96,
                layout.glyph_size(0.016, 0.028),
                color,
                29_010,
            );
            out_y -= line_step;
            if out_y < out_bottom {
                break;
            }
        }
        let filter_label = match console.log_filter {
            RuntimeConsoleLogFilter::All => "all",
            RuntimeConsoleLogFilter::Info => "info",
            RuntimeConsoleLogFilter::Error => "error",
        };
        let tail_label = console
            .log_tail_limit
            .map(|n| n.to_string())
            .unwrap_or_else(|| "off".to_string());
        if console.log_scroll > 0
            || !matches!(console.log_filter, RuntimeConsoleLogFilter::All)
            || console.log_tail_limit.is_some()
        {
            Self::push_console_text_line(
                sprites,
                &mut sprite_id_seed,
                &format!(
                    "SCROLL {} | FILTER {} | TAIL {}",
                    console.log_scroll, filter_label, tail_label
                ),
                0.72 * layout.sx,
                -0.66 * layout.sy,
                0.96,
                layout.glyph_size(0.013, 0.024),
                [0.74, 0.96, 0.74, 1.0],
                29_010,
            );
        }
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

    fn retune_adaptive_pacer(&mut self, dt: f32, mobile_path: bool) {
        if !self.adaptive_pacer_enabled {
            return;
        }

        self.adaptive_pacer_timer -= dt.max(0.0);
        if self.adaptive_pacer_timer > 0.0 {
            return;
        }

        let target_ms = (1_000.0 / self.adaptive_pacer_fps.max(1.0)).clamp(6.0, 42.0);
        let overload = self.frame_time_ema_ms > target_ms * 1.03
            || self.frame_time_jitter_ema_ms > target_ms * 0.20
            || self.framebuffer_fill_ema > 0.95;
        let headroom = self.frame_time_ema_ms < target_ms * 0.82
            && self.frame_time_jitter_ema_ms < target_ms * 0.10
            && self.framebuffer_fill_ema < 0.70;

        if overload {
            self.adaptive_pacer_fps *= 0.93;
            self.adaptive_pacer_timer = 0.18;
        } else if headroom {
            self.adaptive_pacer_fps *= 1.03;
            self.adaptive_pacer_timer = 0.42;
        } else {
            self.adaptive_pacer_timer = 0.28;
        }

        let min_cap = if mobile_path { 45.0 } else { 55.0 };
        let max_cap = if mobile_path { 90.0 } else { 120.0 };
        self.adaptive_pacer_fps = self.adaptive_pacer_fps.clamp(min_cap, max_cap);
        self.frame_cap_interval = Some(Duration::from_secs_f32(
            1.0 / self.adaptive_pacer_fps.max(1.0),
        ));
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
        let ctrl = self.keyboard_modifiers.control_key() || self.keyboard_camera.key_ctrl;
        let alt = self.keyboard_modifiers.alt_key();
        let is_f1 = matches!(event.physical_key, PhysicalKey::Code(KeyCode::F1))
            || matches!(event.logical_key, Key::Named(NamedKey::F1));
        let is_backquote = matches!(event.physical_key, PhysicalKey::Code(KeyCode::Backquote))
            || matches!(&event.logical_key, Key::Character(raw) if raw.as_ref() == "`" || raw.as_ref() == "~");
        let is_key_k = matches!(event.physical_key, PhysicalKey::Code(KeyCode::KeyK));
        let toggle_console =
            pressed && !event.repeat && (is_f1 || (ctrl && (is_backquote || is_key_k)));

        if toggle_console {
            self.toggle_console();
            return RuntimeCommand::Consumed;
        }

        if self.console.open {
            return self.on_console_keyboard_input(event);
        }

        self.update_camera_keyboard_input(event.physical_key, pressed);
        if let PhysicalKey::Code(KeyCode::KeyF) = event.physical_key {
            self.script_key_f_keyboard = pressed;
        }
        if let PhysicalKey::Code(KeyCode::KeyG) = event.physical_key {
            self.script_key_g_keyboard = pressed;
        }

        if !pressed || event.repeat {
            return RuntimeCommand::None;
        }

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

    fn on_cursor_moved(&mut self, x: f32, y: f32) {
        self.cursor_position = Some((x, y));
    }

    fn cursor_ndc(&self) -> Option<(f32, f32)> {
        let (x, y) = self.cursor_position?;
        let width = self.size.width.max(1) as f32;
        let height = self.size.height.max(1) as f32;
        let ndc_x = (x / width) * 2.0 - 1.0;
        let ndc_y = 1.0 - (y / height) * 2.0;
        Some((ndc_x, ndc_y))
    }

    fn point_in_rect(point: (f32, f32), center: (f32, f32), size: (f32, f32)) -> bool {
        let (px, py) = point;
        let (cx, cy) = center;
        let (sx, sy) = size;
        (px >= cx - sx * 0.5)
            && (px <= cx + sx * 0.5)
            && (py >= cy - sy * 0.5)
            && (py <= cy + sy * 0.5)
    }

    fn apply_all_console_quick_boxes(&mut self) {
        let previous_target = self.console.edit_target;
        self.console.edit_target = RuntimeConsoleEditTarget::FpsCap;
        let _ = self.apply_active_console_edit_target();
        self.console.edit_target = RuntimeConsoleEditTarget::RenderDistance;
        let _ = self.apply_active_console_edit_target();
        self.console.edit_target = RuntimeConsoleEditTarget::FsrSharpness;
        let _ = self.apply_active_console_edit_target();
        self.console.edit_target = previous_target;
    }

    fn handle_console_left_click(&mut self) {
        let Some(cursor) = self.cursor_ndc() else {
            return;
        };
        let layout = ConsoleUiLayout::from_size(self.size);
        if Self::point_in_rect(cursor, layout.send_center, layout.send_size) {
            self.console.edit_target = RuntimeConsoleEditTarget::Command;
            let _ = self.submit_console_command(self.console.input_line.clone());
            return;
        }
        if Self::point_in_rect(cursor, layout.command_center, layout.command_size) {
            self.console.edit_target = RuntimeConsoleEditTarget::Command;
            self.console_feedback("active edit box: command");
            return;
        }
        if Self::point_in_rect(cursor, layout.fps_center, layout.fps_size) {
            self.console.edit_target = RuntimeConsoleEditTarget::FpsCap;
            self.console_feedback("active edit box: fps_cap");
            return;
        }
        if Self::point_in_rect(cursor, layout.distance_center, layout.distance_size) {
            self.console.edit_target = RuntimeConsoleEditTarget::RenderDistance;
            self.console_feedback("active edit box: render_distance");
            return;
        }
        if Self::point_in_rect(cursor, layout.sharpness_center, layout.sharpness_size) {
            self.console.edit_target = RuntimeConsoleEditTarget::FsrSharpness;
            self.console_feedback("active edit box: fsr_sharpness");
            return;
        }
        if Self::point_in_rect(cursor, layout.apply_center, layout.apply_size) {
            self.apply_all_console_quick_boxes();
        }
    }

    fn on_mouse_button(&mut self, state: ElementState, button: MouseButton) {
        if self.console.open {
            if button == MouseButton::Left && state == ElementState::Pressed {
                self.handle_console_left_click();
            }
            return;
        }
        if button == MouseButton::Right {
            self.mouse_look_held = state == ElementState::Pressed;
        }
    }

    fn on_touch(&mut self, touch: Touch) {
        if self.console.open {
            return;
        }
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
        if self.console.open {
            return;
        }
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
                if matches!(key, PhysicalKey::Code(KeyCode::KeyW)) {
                    self.keyboard_camera.key_w = pressed;
                }
                if matches!(key, PhysicalKey::Code(KeyCode::ArrowUp)) {
                    self.keyboard_camera.key_up = pressed;
                }
            }
            PhysicalKey::Code(KeyCode::KeyS) | PhysicalKey::Code(KeyCode::ArrowDown) => {
                if matches!(key, PhysicalKey::Code(KeyCode::KeyS)) {
                    self.keyboard_camera.key_s = pressed;
                }
                if matches!(key, PhysicalKey::Code(KeyCode::ArrowDown)) {
                    self.keyboard_camera.key_down = pressed;
                }
            }
            PhysicalKey::Code(KeyCode::KeyA) | PhysicalKey::Code(KeyCode::ArrowLeft) => {
                if matches!(key, PhysicalKey::Code(KeyCode::KeyA)) {
                    self.keyboard_camera.key_a = pressed;
                }
                if matches!(key, PhysicalKey::Code(KeyCode::ArrowLeft)) {
                    self.keyboard_camera.key_left = pressed;
                }
            }
            PhysicalKey::Code(KeyCode::KeyD) | PhysicalKey::Code(KeyCode::ArrowRight) => {
                if matches!(key, PhysicalKey::Code(KeyCode::KeyD)) {
                    self.keyboard_camera.key_d = pressed;
                }
                if matches!(key, PhysicalKey::Code(KeyCode::ArrowRight)) {
                    self.keyboard_camera.key_right = pressed;
                }
            }
            PhysicalKey::Code(KeyCode::Space) | PhysicalKey::Code(KeyCode::KeyE) => {
                if matches!(key, PhysicalKey::Code(KeyCode::Space)) {
                    self.keyboard_camera.key_space = pressed;
                }
                if matches!(key, PhysicalKey::Code(KeyCode::KeyE)) {
                    self.keyboard_camera.key_e = pressed;
                }
            }
            PhysicalKey::Code(KeyCode::ControlLeft) | PhysicalKey::Code(KeyCode::ControlRight) => {
                self.keyboard_camera.key_ctrl = pressed
            }
            PhysicalKey::Code(KeyCode::KeyQ) => self.keyboard_camera.key_q = pressed,
            PhysicalKey::Code(KeyCode::KeyC) => self.keyboard_camera.key_c = pressed,
            PhysicalKey::Code(KeyCode::ShiftLeft) | PhysicalKey::Code(KeyCode::ShiftRight) => {
                self.keyboard_camera.key_shift = pressed
            }
            PhysicalKey::Code(KeyCode::KeyR) => self.keyboard_camera.key_r = pressed,
            PhysicalKey::Code(KeyCode::KeyL) => self.keyboard_camera.key_l = pressed,
            PhysicalKey::Code(KeyCode::AltLeft) | PhysicalKey::Code(KeyCode::AltRight) => {
                self.keyboard_camera.key_alt = pressed
            }
            PhysicalKey::Code(KeyCode::Enter) | PhysicalKey::Code(KeyCode::NumpadEnter) => {
                self.keyboard_camera.key_enter = pressed
            }
            PhysicalKey::Code(KeyCode::Escape) => self.keyboard_camera.key_escape = pressed,
            PhysicalKey::Code(KeyCode::Tab) => self.keyboard_camera.key_tab = pressed,
            _ => {}
        }
    }

    fn script_camera_input(&mut self, view_dt: f32) -> TlscriptShowcaseControlInput {
        if self.console.open {
            self.mouse_look_delta = (0.0, 0.0);
            return TlscriptShowcaseControlInput::default();
        }
        let gamepad = self.gamepad.camera_state();
        let sensitivity = self.camera.mouse_sensitivity().max(0.0001);
        let pad_look_to_mouse = (GAMEPAD_LOOK_SPEED_RAD * view_dt.max(0.0)) / sensitivity;

        // Keyboard layout is script-driven now; keep legacy move_* channels fed from gamepad only.
        let move_x = gamepad.move_x;
        let move_y = gamepad.move_y;
        let move_z = gamepad.rise - gamepad.descend;
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
            sprint_down: self.keyboard_camera.key_shift || gamepad.sprint,
            look_active,
            reset_camera,
            key_w_down: self.keyboard_camera.key_w,
            key_s_down: self.keyboard_camera.key_s,
            key_a_down: self.keyboard_camera.key_a,
            key_d_down: self.keyboard_camera.key_d,
            key_up_down: self.keyboard_camera.key_up,
            key_down_down: self.keyboard_camera.key_down,
            key_left_down: self.keyboard_camera.key_left,
            key_right_down: self.keyboard_camera.key_right,
            key_space_down: self.keyboard_camera.key_space,
            key_ctrl_down: self.keyboard_camera.key_ctrl,
            key_shift_down: self.keyboard_camera.key_shift,
            key_q_down: self.keyboard_camera.key_q,
            key_e_down: self.keyboard_camera.key_e,
            key_c_down: self.keyboard_camera.key_c,
            key_g_down: self.script_key_g_keyboard || self.gamepad.action_f_down(),
            key_r_down: self.keyboard_camera.key_r,
            key_l_down: self.keyboard_camera.key_l,
            key_alt_down: self.keyboard_camera.key_alt,
            key_enter_down: self.keyboard_camera.key_enter,
            key_escape_down: self.keyboard_camera.key_escape,
            key_tab_down: self.keyboard_camera.key_tab,
            mouse_look_down: self.mouse_look_held,
            pad_move_x: gamepad.move_x,
            pad_move_y: gamepad.move_y,
            pad_rise: gamepad.rise,
            pad_descend: gamepad.descend,
            pad_look_x: gamepad.look_x,
            pad_look_y: gamepad.look_y,
            pad_sprint_down: gamepad.sprint,
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
        if self.console.open {
            self.poll_console_tail_follow();
            self.poll_console_file_watch();
        }
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

        let mut frame_eval = self.script_runtime.evaluate_frame(
            TlscriptShowcaseFrameInput {
                frame_index: self.script_frame_index,
                live_balls: self.scene.live_ball_count(),
                spawned_this_tick: self.script_last_spawned,
                key_f_down: self.script_key_f_keyboard || self.gamepad.action_f_down(),
            },
            script_camera_input,
        );
        if !self.console.script_statements.is_empty() {
            merge_showcase_output(
                &mut frame_eval,
                self.console.script_overlay.clone(),
                CONSOLE_SCRIPT_INDEX,
            );
        }

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
        if let Some(mode) = frame_eval.rt_mode {
            self.rt_mode = mode;
            self.renderer.set_ray_tracing_mode(&self.queue, mode);
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
            self.max_substeps = self
                .manual_max_substeps
                .unwrap_or_else(|| load_plan.max_substeps.max(2));
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

        self.script_frame_index = self.script_frame_index.saturating_add(1);
        let allow_physics_step = if self.simulation_paused {
            if self.simulation_step_budget > 0 {
                self.simulation_step_budget = self.simulation_step_budget.saturating_sub(1);
                true
            } else {
                false
            }
        } else {
            true
        };

        let (tick, substeps) = if allow_physics_step {
            let _patch_metrics = self
                .scene
                .apply_runtime_patch(&mut self.world, runtime_patch);
            let tick = self.scene.physics_tick(&mut self.world);
            self.script_last_spawned = tick.spawned_this_tick;
            let step_dt = if self.simulation_paused {
                self.world.config().fixed_dt
            } else {
                sim_dt
            };
            let substeps = self.world.step(step_dt);
            self.last_substeps = substeps;
            let _ = self.scene.reconcile_after_step(&mut self.world);
            (tick, substeps)
        } else {
            self.script_last_spawned = 0;
            self.last_substeps = 0;
            (
                BounceTankTickMetrics {
                    spawned_this_tick: 0,
                    scattered_this_tick: 0,
                    live_balls: self.scene.live_ball_count(),
                    target_balls: self.scene.config().target_ball_count,
                    fully_spawned: self.scene.live_ball_count()
                        >= self.scene.config().target_ball_count,
                },
                0,
            )
        };

        let mut frame = self.scene.build_frame_instances_with_ball_limit(
            &self.world,
            Some(self.world.interpolation_alpha()),
            self.adaptive_ball_render_limit,
        );
        let unknown_light_overrides =
            apply_scene_light_overrides(&mut frame, frame_eval.light_overrides.as_slice());
        let light_pruned = clamp_scene_lights_for_camera(&mut frame, eye, MAX_SCENE_LIGHTS);
        if unknown_light_overrides > 0 && self.script_frame_index % 180 == 0 {
            eprintln!(
                "[tlscript light] {} unknown light id override(s) were ignored",
                unknown_light_overrides
            );
        }
        if light_pruned > 0 && self.script_frame_index % 180 == 0 {
            eprintln!(
                "[scene lights] pruned {} light(s) to MAX_SCENE_LIGHTS={}",
                light_pruned, MAX_SCENE_LIGHTS
            );
        }
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
        self.retune_adaptive_pacer(sim_dt, mobile_path);
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
                rt_mode: self.rt_mode,
                rt_active: self.renderer.ray_tracing_status().active,
                rt_dynamic_count: self.renderer.ray_tracing_status().rt_dynamic_count,
                rt_fallback: !self
                    .renderer
                    .ray_tracing_status()
                    .fallback_reason
                    .is_empty(),
            },
            &mut frame.sprites,
        );
        self.console_overlay_sprites.clear();
        let console_layout = ConsoleUiLayout::from_size(self.size);
        Self::append_console_overlay_sprites(
            &self.console,
            console_layout,
            &mut self.console_overlay_sprites,
        );
        let draw = self.draw_compiler.compile(&frame);
        let upload = self
            .renderer
            .upload_draw_frame(&self.device, &self.queue, &draw);
        self.renderer.upload_overlay_sprites(
            &self.device,
            &self.queue,
            self.console_overlay_sprites.as_slice(),
        );
        let rt_status = self.renderer.ray_tracing_status();
        let fsr_status = self.renderer.fsr_status();

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
        self.renderer.encode_overlay_sprites(&mut encoder, &view);
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
        let console_suffix = self.console_title_suffix();
        let title = format!(
            "Tileline TLApp | FPS {:.1} | Frame {:.2} ms | Tick {:.0} Hz | Balls {} (draw {}) | Lights {} | RT {:?}/{} ({}) | FSR {:?}/{} ({:.2}) | Substeps {} | {:?} {} {:?}{}{}{}{}{}{}",
            self.fps_tracker.ema_fps(),
            frame_time * 1_000.0,
            self.tick_hz,
            tick.live_balls,
            visible_ball_count,
            upload.light_count,
            self.rt_mode,
            if rt_status.active { "on" } else { "off" },
            rt_status.rt_dynamic_count,
            fsr_status.requested_mode,
            if fsr_status.active { "on" } else { "off" },
            fsr_status.render_scale,
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
            console_suffix,
        );
        self.window.set_title(&title);

        if let Some(report) = report {
            let broadphase = self.world.broadphase().stats();
            let narrowphase = self.world.narrowphase().stats();
            println!(
                "tlapp fps | inst: {:>6.1} | ema: {:>6.1} | avg: {:>6.1} | stddev: {:>5.2} ms | balls: {:>5} | draw: {:>5} | lights: {:>2} | substeps: {} | scattered: {:>4} | rd_culled: {:>4} | rd_blur: {:>4} | fill: {:>4.2} | fill_ema: {:>4.2} | rt_mode: {:?} | rt_active: {} | rt_dynamic: {:>4} | rt_reason: {} | fsr_mode: {:?} | fsr_active: {} | fsr_scale: {:>4.2} | fsr_sharpness: {:>4.2} | fsr_reason: {} | mps_threads: {} | shards: {} | pairs: {} | manifolds: {} | platform: {:?} | backend: {:?} | scheduler: {} | present: {:?} | fallback: {} | adapter: {} | reason: {}",
                report.instant_fps,
                report.ema_fps,
                report.avg_fps,
                report.frame_time_stddev_ms,
                tick.live_balls,
                visible_ball_count,
                upload.light_count,
                substeps,
                tick.scattered_this_tick,
                self.last_distance_culled,
                self.last_distance_blurred,
                self.last_framebuffer_fill_ratio,
                self.framebuffer_fill_ema,
                self.rt_mode,
                rt_status.active,
                rt_status.rt_dynamic_count,
                if rt_status.fallback_reason.is_empty() {
                    "none"
                } else {
                    rt_status.fallback_reason.as_str()
                },
                fsr_status.requested_mode,
                fsr_status.active,
                fsr_status.render_scale,
                fsr_status.sharpness,
                if fsr_status.reason.is_empty() {
                    "none"
                } else {
                    fsr_status.reason.as_str()
                },
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

#[derive(Debug, Clone, Copy, Default)]
struct CameraInputState {
    key_w: bool,
    key_s: bool,
    key_a: bool,
    key_d: bool,
    key_up: bool,
    key_down: bool,
    key_left: bool,
    key_right: bool,
    key_space: bool,
    key_ctrl: bool,
    key_shift: bool,
    key_q: bool,
    key_e: bool,
    key_c: bool,
    key_r: bool,
    key_l: bool,
    key_alt: bool,
    key_enter: bool,
    key_escape: bool,
    key_tab: bool,
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

    fn snapshot(&self) -> FpsReport {
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
