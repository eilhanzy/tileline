use std::env;
use std::error::Error;
use std::path::PathBuf;
use std::time::Duration;

use winit::dpi::PhysicalSize;

use crate::{FsrMode, FsrQualityPreset, GraphicsSchedulerPath, DEFAULT_MSAA_SAMPLE_COUNT};

#[derive(Debug, Clone)]
pub struct CliOptions {
    pub resolution: PhysicalSize<u32>,
    pub vsync: VsyncMode,
    pub fps_cap: Option<f32>,
    pub tick_profile: TickProfile,
    pub render_distance: Option<f32>,
    pub adaptive_distance: ToggleAuto,
    pub distance_blur: ToggleAuto,
    pub msaa: u32,
    pub fsr_mode: FsrMode,
    pub fsr_quality: FsrQualityPreset,
    pub fsr_sharpness: f32,
    pub fsr_scale_override: Option<f32>,
    pub fps_report_interval: Duration,
    pub project_path: Option<PathBuf>,
    pub joint_path: Option<PathBuf>,
    pub joint_scene: String,
    pub script_path: PathBuf,
    pub sprite_path: PathBuf,
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
            msaa: DEFAULT_MSAA_SAMPLE_COUNT,
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
    pub fn parse_from_env() -> Result<Self, Box<dyn Error>> {
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
                "--msaa" => {
                    let value = next_arg(&mut args, "--msaa")?;
                    options.msaa = parse_msaa(&value)?;
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
pub enum VsyncMode {
    Auto,
    On,
    Off,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TickProfile {
    Balanced,
    Max,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToggleAuto {
    Auto,
    On,
    Off,
}

#[derive(Debug, Clone)]
pub struct SchedulerResolution {
    pub selected: GraphicsSchedulerPath,
    pub fallback_applied: bool,
    pub reason: String,
}

impl TickProfile {
    pub fn parse(value: &str) -> Result<Self, Box<dyn Error>> {
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
    pub fn parse(value: &str) -> Result<Self, Box<dyn Error>> {
        match value.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "on" | "1" | "true" => Ok(Self::On),
            "off" | "0" | "false" => Ok(Self::Off),
            _ => Err(format!("invalid --vsync value: {value} (expected auto|on|off)").into()),
        }
    }
}

impl ToggleAuto {
    pub fn parse(value: &str, flag: &str) -> Result<Self, Box<dyn Error>> {
        match value.to_ascii_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "on" | "1" | "true" => Ok(Self::On),
            "off" | "0" | "false" => Ok(Self::Off),
            _ => Err(format!("invalid {flag} value: {value} (expected auto|on|off)").into()),
        }
    }

    pub fn resolve(self, auto_default: bool) -> bool {
        match self {
            Self::Auto => auto_default,
            Self::On => true,
            Self::Off => false,
        }
    }
}

pub fn next_arg(
    args: &mut impl Iterator<Item = String>,
    flag: &str,
) -> Result<String, Box<dyn Error>> {
    args.next()
        .ok_or_else(|| format!("missing value for {flag}").into())
}

pub fn parse_seconds_arg(value: &str, flag: &str) -> Result<Duration, Box<dyn Error>> {
    let seconds = value
        .parse::<f64>()
        .map_err(|_| format!("invalid {flag}: {value} (expected seconds)"))?;
    if !seconds.is_finite() || seconds <= 0.0 {
        return Err(format!("invalid {flag}: {value} (must be > 0)").into());
    }
    Ok(Duration::from_secs_f64(seconds))
}

pub fn parse_resolution(value: &str) -> Result<PhysicalSize<u32>, Box<dyn Error>> {
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

pub fn parse_fps_cap(value: &str) -> Result<Option<f32>, Box<dyn Error>> {
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

pub fn parse_render_distance(value: &str) -> Result<Option<f32>, Box<dyn Error>> {
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

pub fn parse_msaa(value: &str) -> Result<u32, Box<dyn Error>> {
    match value.to_ascii_lowercase().as_str() {
        "off" | "1" => Ok(1),
        "2" => Ok(2),
        "4" => Ok(4),
        _ => Err(format!("invalid --msaa value: {value} (expected off|2|4)").into()),
    }
}

pub fn parse_fsr_sharpness(value: &str) -> Result<f32, Box<dyn Error>> {
    let parsed = value
        .parse::<f32>()
        .map_err(|_| format!("invalid --fsr-sharpness value: {value}"))?;
    if !parsed.is_finite() || !(0.0..=1.0).contains(&parsed) {
        return Err("--fsr-sharpness must be in 0.0..=1.0".into());
    }
    Ok(parsed)
}

pub fn parse_fsr_scale(value: &str) -> Result<Option<f32>, Box<dyn Error>> {
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
    println!("  --msaa <off|2|4>          MSAA sample count (default: 4)");
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
    use crate::{tileline_version_entries, ENGINE_VERSION};
    println!("Tileline Engine: v{ENGINE_VERSION}");
    for entry in tileline_version_entries() {
        println!("  {:<10} v{}", entry.module, entry.version);
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
        use crate::{TlpfileGraphicsScheduler, RuntimePlatform, GraphicsSchedulerPath};
        use super::super::resolve_project_scheduler;
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
        use crate::{TlpfileGraphicsScheduler, RuntimePlatform};
        use super::super::resolve_project_scheduler;
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
        use crate::{TlpfileGraphicsScheduler, RuntimePlatform, GraphicsSchedulerPath};
        use super::super::resolve_project_scheduler;
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
        use super::super::build_vulkan_unavailable_message;
        let message = build_vulkan_unavailable_message(
            "vk init failed",
            Some(("FallbackGPU", wgpu::Backend::Gl)),
        );
        assert!(message.contains("Vulkan unavailable on Android"));
        assert!(message.contains("FallbackGPU"));
        assert!(message.contains("backend=Gl"));
    }
}
