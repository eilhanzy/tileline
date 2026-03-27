use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use winit::dpi::PhysicalSize;

use crate::{FsrMode, FsrQualityPreset, GraphicsSchedulerPath, DEFAULT_MSAA_SAMPLE_COUNT};

#[derive(Debug, Clone)]
pub struct CliOptions {
    pub resolution: PhysicalSize<u32>,
    pub vsync: VsyncMode,
    pub fps_cap: Option<f32>,
    pub pipeline_mode: PipelineMode,
    pub tick_profile: TickProfile,
    pub tick_cap: Option<f32>,
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
    pub ini_path: Option<PathBuf>,
    pub pak_path: Option<PathBuf>,
}

impl Default for CliOptions {
    fn default() -> Self {
        Self {
            resolution: PhysicalSize::new(1280, 720),
            vsync: VsyncMode::Auto,
            fps_cap: Some(60.0),
            pipeline_mode: PipelineMode::default_for_build(),
            tick_profile: TickProfile::Max,
            tick_cap: None,
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
            ini_path: None,
            pak_path: None,
        }
    }
}

impl CliOptions {
    pub fn parse_from_env() -> Result<Self, Box<dyn Error>> {
        let args = env::args().skip(1).collect::<Vec<_>>();
        if args
            .iter()
            .any(|arg| matches!(arg.as_str(), "-h" | "--help"))
        {
            print_usage();
            std::process::exit(0);
        }
        if args
            .iter()
            .any(|arg| matches!(arg.as_str(), "-V" | "--version"))
        {
            print_version_overview();
            std::process::exit(0);
        }

        let mut options = Self::default();
        let explicit_ini_path = scan_ini_path_from_args(&args)?;
        let ini_path = explicit_ini_path
            .clone()
            .or_else(|| env::var("TILELINE_INI").ok().map(PathBuf::from));
        if let Some(ini_path) = ini_path {
            let warnings = apply_ini_overrides(&mut options, &ini_path)?;
            options.ini_path = Some(ini_path);
            for warning in warnings {
                eprintln!("[tlapp ini] {warning}");
            }
        }

        // Precedence: CLI > env > ini > defaults.
        // Apply env overrides after INI, then parse CLI for final authority.
        apply_env_overrides(&mut options)?;
        let pipeline_explicit = parse_cli_overrides(&args, &mut options)?;
        if !pipeline_explicit {
            if let Ok(value) = env::var("TILELINE_PIPELINE") {
                options.pipeline_mode = PipelineMode::parse(&value).map_err(|_| {
                    format!("invalid TILELINE_PIPELINE value: {value} (expected parallel|legacy)")
                })?;
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
pub enum PipelineMode {
    Parallel,
    Legacy,
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

impl PipelineMode {
    pub fn default_for_build() -> Self {
        if cfg!(feature = "parallel_pipeline_v2") {
            Self::Parallel
        } else {
            Self::Legacy
        }
    }

    pub fn parse(value: &str) -> Result<Self, Box<dyn Error>> {
        match value.to_ascii_lowercase().as_str() {
            "parallel" | "v2" => Ok(Self::Parallel),
            "legacy" | "v1" => Ok(Self::Legacy),
            _ => {
                Err(format!("invalid --pipeline value: {value} (expected parallel|legacy)").into())
            }
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Parallel => "parallel",
            Self::Legacy => "legacy",
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

pub fn parse_tick_cap(value: &str) -> Result<Option<f32>, Box<dyn Error>> {
    if value.eq_ignore_ascii_case("off") || value.eq_ignore_ascii_case("none") {
        return Ok(None);
    }
    let hz = value
        .parse::<f32>()
        .map_err(|_| format!("invalid --tick-cap value: {value} (expected Hz number or off)"))?;
    if !hz.is_finite() || hz < 24.0 {
        return Err("--tick-cap must be >= 24.0 Hz or 'off'".into());
    }
    Ok(Some(hz))
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

fn scan_ini_path_from_args(args: &[String]) -> Result<Option<PathBuf>, Box<dyn Error>> {
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--ini" => {
                let Some(value) = args.get(i + 1) else {
                    return Err("missing value for --ini".into());
                };
                return Ok(Some(PathBuf::from(value)));
            }
            "-h" | "--help" | "-V" | "--version" => {
                i += 1;
            }
            _ => {
                i += 1;
            }
        }
    }
    Ok(None)
}

fn apply_env_overrides(options: &mut CliOptions) -> Result<(), Box<dyn Error>> {
    if let Ok(value) = env::var("TILELINE_RESOLUTION") {
        options.resolution = parse_resolution(&value)?;
    }
    if let Ok(value) = env::var("TILELINE_VSYNC") {
        options.vsync = VsyncMode::parse(&value)?;
    }
    if let Ok(value) = env::var("TILELINE_FPS_CAP") {
        options.fps_cap = parse_fps_cap(&value)?;
    }
    if let Ok(value) = env::var("TILELINE_TICK_PROFILE") {
        options.tick_profile = TickProfile::parse(&value)?;
    }
    if let Ok(value) = env::var("TILELINE_TICK_CAP") {
        options.tick_cap = parse_tick_cap(&value)?;
    }
    if let Ok(value) = env::var("TILELINE_RENDER_DISTANCE") {
        options.render_distance = parse_render_distance(&value)?;
    }
    if let Ok(value) = env::var("TILELINE_ADAPTIVE_DISTANCE") {
        options.adaptive_distance = ToggleAuto::parse(&value, "TILELINE_ADAPTIVE_DISTANCE")?;
    }
    if let Ok(value) = env::var("TILELINE_DISTANCE_BLUR") {
        options.distance_blur = ToggleAuto::parse(&value, "TILELINE_DISTANCE_BLUR")?;
    }
    if let Ok(value) = env::var("TILELINE_MSAA") {
        options.msaa = parse_msaa(&value)?;
    }
    if let Ok(value) = env::var("TILELINE_FSR_MODE") {
        options.fsr_mode = FsrMode::parse(&value).ok_or_else(|| -> Box<dyn Error> {
            "invalid TILELINE_FSR_MODE value (expected off|auto|on|fsr1)".into()
        })?;
    }
    if let Ok(value) = env::var("TILELINE_FSR_QUALITY") {
        options.fsr_quality =
            FsrQualityPreset::parse(&value).ok_or_else(|| -> Box<dyn Error> {
                "invalid TILELINE_FSR_QUALITY value (expected native|ultra|quality|balanced|performance)"
                    .into()
            })?;
    }
    if let Ok(value) = env::var("TILELINE_FSR_SHARPNESS") {
        options.fsr_sharpness = parse_fsr_sharpness(&value)?;
    }
    if let Ok(value) = env::var("TILELINE_FSR_SCALE") {
        options.fsr_scale_override = parse_fsr_scale(&value)?;
    }
    if let Ok(value) = env::var("TILELINE_FPS_REPORT") {
        options.fps_report_interval = parse_seconds_arg(&value, "TILELINE_FPS_REPORT")?;
    }
    if let Ok(value) = env::var("TILELINE_PROJECT") {
        options.project_path = Some(PathBuf::from(value));
    }
    if let Ok(value) = env::var("TILELINE_JOINT") {
        options.joint_path = Some(PathBuf::from(value));
    }
    if let Ok(value) = env::var("TILELINE_SCENE") {
        options.joint_scene = value.trim().to_string();
    }
    if let Ok(value) = env::var("TILELINE_SCRIPT") {
        options.script_path = PathBuf::from(value);
    }
    if let Ok(value) = env::var("TILELINE_SPRITE") {
        options.sprite_path = PathBuf::from(value);
    }
    if let Ok(value) = env::var("TILELINE_PAK") {
        options.pak_path = Some(PathBuf::from(value));
    }
    Ok(())
}

fn parse_cli_overrides(args: &[String], options: &mut CliOptions) -> Result<bool, Box<dyn Error>> {
    let mut iter = args.iter().cloned();
    let mut pipeline_explicit = false;

    while let Some(arg) = iter.next() {
        match arg.as_str() {
            "-h" | "--help" | "-V" | "--version" => {}
            "--ini" => {
                let value = next_arg(&mut iter, "--ini")?;
                options.ini_path = Some(PathBuf::from(value));
            }
            "--pak" => {
                let value = next_arg(&mut iter, "--pak")?;
                options.pak_path = Some(PathBuf::from(value));
            }
            "--resolution" => {
                let value = next_arg(&mut iter, "--resolution")?;
                options.resolution = parse_resolution(&value)?;
            }
            "--vsync" => {
                let value = next_arg(&mut iter, "--vsync")?;
                options.vsync = VsyncMode::parse(&value)?;
            }
            "--fps-cap" => {
                let value = next_arg(&mut iter, "--fps-cap")?;
                options.fps_cap = parse_fps_cap(&value)?;
            }
            "--pipeline" => {
                let value = next_arg(&mut iter, "--pipeline")?;
                options.pipeline_mode = PipelineMode::parse(&value)?;
                pipeline_explicit = true;
            }
            "--tick-profile" => {
                let value = next_arg(&mut iter, "--tick-profile")?;
                options.tick_profile = TickProfile::parse(&value)?;
            }
            "--tick-cap" => {
                let value = next_arg(&mut iter, "--tick-cap")?;
                options.tick_cap = parse_tick_cap(&value)?;
            }
            "--render-distance" => {
                let value = next_arg(&mut iter, "--render-distance")?;
                options.render_distance = parse_render_distance(&value)?;
            }
            "--adaptive-distance" => {
                let value = next_arg(&mut iter, "--adaptive-distance")?;
                options.adaptive_distance = ToggleAuto::parse(&value, "--adaptive-distance")?;
            }
            "--distance-blur" => {
                let value = next_arg(&mut iter, "--distance-blur")?;
                options.distance_blur = ToggleAuto::parse(&value, "--distance-blur")?;
            }
            "--msaa" => {
                let value = next_arg(&mut iter, "--msaa")?;
                options.msaa = parse_msaa(&value)?;
            }
            "--fsr" => {
                let value = next_arg(&mut iter, "--fsr")?;
                options.fsr_mode = FsrMode::parse(&value).ok_or_else(|| -> Box<dyn Error> {
                    "invalid --fsr value (expected off|auto|on|fsr1)".into()
                })?;
            }
            "--fsr-quality" => {
                let value = next_arg(&mut iter, "--fsr-quality")?;
                options.fsr_quality = FsrQualityPreset::parse(&value).ok_or_else(
                    || -> Box<dyn Error> {
                        "invalid --fsr-quality value (expected native|ultra|quality|balanced|performance)"
                            .into()
                    },
                )?;
            }
            "--fsr-sharpness" => {
                let value = next_arg(&mut iter, "--fsr-sharpness")?;
                options.fsr_sharpness = parse_fsr_sharpness(&value)?;
            }
            "--fsr-scale" => {
                let value = next_arg(&mut iter, "--fsr-scale")?;
                options.fsr_scale_override = parse_fsr_scale(&value)?;
            }
            "--fps-report" => {
                let value = next_arg(&mut iter, "--fps-report")?;
                options.fps_report_interval = parse_seconds_arg(&value, "--fps-report")?;
            }
            "--project" => {
                let value = next_arg(&mut iter, "--project")?;
                options.project_path = Some(PathBuf::from(value));
            }
            "--joint" => {
                let value = next_arg(&mut iter, "--joint")?;
                options.joint_path = Some(PathBuf::from(value));
            }
            "--scene" => {
                let value = next_arg(&mut iter, "--scene")?;
                options.joint_scene = value.trim().to_string();
            }
            "--script" => {
                let value = next_arg(&mut iter, "--script")?;
                options.script_path = PathBuf::from(value);
            }
            "--sprite" => {
                let value = next_arg(&mut iter, "--sprite")?;
                options.sprite_path = PathBuf::from(value);
            }
            other => {
                return Err(format!("unknown argument: {other} (use --help)").into());
            }
        }
    }

    Ok(pipeline_explicit)
}

fn apply_ini_overrides(
    options: &mut CliOptions,
    ini_path: &Path,
) -> Result<Vec<String>, Box<dyn Error>> {
    let source = fs::read_to_string(ini_path).map_err(|err| {
        format!(
            "failed to read ini '{}': {err}",
            ini_path.as_os_str().to_string_lossy()
        )
    })?;
    let mut warnings = Vec::new();
    for (line_no, key, value) in parse_ini_lines(&source, &mut warnings) {
        let normalized = key.to_ascii_lowercase().replace('-', "_");
        match normalized.as_str() {
            "resolution" => options.resolution = parse_resolution(&value)?,
            "vsync" => options.vsync = VsyncMode::parse(&value)?,
            "fps_cap" => options.fps_cap = parse_fps_cap(&value)?,
            "pipeline" => options.pipeline_mode = PipelineMode::parse(&value)?,
            "tick_profile" => options.tick_profile = TickProfile::parse(&value)?,
            "tick_cap" => options.tick_cap = parse_tick_cap(&value)?,
            "render_distance" => options.render_distance = parse_render_distance(&value)?,
            "adaptive_distance" => {
                options.adaptive_distance = ToggleAuto::parse(&value, "adaptive_distance")?
            }
            "distance_blur" => options.distance_blur = ToggleAuto::parse(&value, "distance_blur")?,
            "msaa" => options.msaa = parse_msaa(&value)?,
            "fsr" | "fsr_mode" => {
                options.fsr_mode =
                    FsrMode::parse(&value).ok_or_else(|| -> Box<dyn Error> {
                        "invalid fsr value in ini (expected off|auto|on|fsr1)".into()
                    })?
            }
            "fsr_quality" => {
                options.fsr_quality = FsrQualityPreset::parse(&value).ok_or_else(
                    || -> Box<dyn Error> {
                        "invalid fsr_quality value in ini (expected native|ultra|quality|balanced|performance)"
                            .into()
                    },
                )?
            }
            "fsr_sharpness" => options.fsr_sharpness = parse_fsr_sharpness(&value)?,
            "fsr_scale" => options.fsr_scale_override = parse_fsr_scale(&value)?,
            "fps_report" => options.fps_report_interval = parse_seconds_arg(&value, "fps_report")?,
            "project" | "project_path" => options.project_path = Some(PathBuf::from(value)),
            "joint" | "joint_path" => options.joint_path = Some(PathBuf::from(value)),
            "scene" | "joint_scene" => options.joint_scene = value.trim().to_string(),
            "script" | "script_path" => options.script_path = PathBuf::from(value),
            "sprite" | "sprite_path" => options.sprite_path = PathBuf::from(value),
            "pak" | "pak_path" => options.pak_path = Some(PathBuf::from(value)),
            "ini" | "ini_path" => warnings.push(format!(
                "{}:{}: '{}' key is ignored (ini recursion is not supported)",
                ini_path.display(),
                line_no,
                key
            )),
            _ => warnings.push(format!(
                "{}:{}: unknown ini key '{}', ignoring",
                ini_path.display(),
                line_no,
                key
            )),
        }
    }
    Ok(warnings)
}

fn parse_ini_lines(source: &str, warnings: &mut Vec<String>) -> Vec<(usize, String, String)> {
    let mut parsed = Vec::new();
    for (index, raw_line) in source.lines().enumerate() {
        let line_no = index + 1;
        let trimmed = raw_line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with(';') {
            continue;
        }
        if trimmed.starts_with('[') && trimmed.ends_with(']') && trimmed.len() >= 2 {
            continue;
        }
        let Some((raw_key, raw_value)) = trimmed.split_once('=') else {
            warnings.push(format!("line {}: invalid ini entry '{trimmed}'", line_no));
            continue;
        };
        let key = raw_key.trim();
        if key.is_empty() {
            warnings.push(format!("line {}: empty ini key", line_no));
            continue;
        }
        let value = strip_ini_inline_comment(raw_value.trim());
        if value.is_empty() {
            warnings.push(format!("line {}: empty value for key '{key}'", line_no));
            continue;
        }
        let key = key
            .strip_prefix("tlapp_")
            .or_else(|| key.strip_prefix("runtime_"))
            .unwrap_or(key)
            .to_string();
        parsed.push((line_no, key, value));
    }
    parsed
}

fn strip_ini_inline_comment(raw: &str) -> String {
    let mut out = String::new();
    let mut in_quote = false;
    for ch in raw.chars() {
        match ch {
            '"' => {
                in_quote = !in_quote;
                out.push(ch);
            }
            '#' | ';' if !in_quote => break,
            _ => out.push(ch),
        }
    }
    let trimmed = out.trim();
    if trimmed.len() >= 2 && trimmed.starts_with('"') && trimmed.ends_with('"') {
        trimmed[1..trimmed.len() - 1].to_string()
    } else {
        trimmed.to_string()
    }
}

fn print_usage() {
    println!("Tileline TLApp Runtime Demo");
    println!("Usage: cargo run -p runtime --bin tlapp -- [options]");
    println!("Options:");
    println!("  --resolution <WxH>        Window size (default: 1280x720)");
    println!("  --vsync auto|on|off       Present mode preference (default: auto)");
    println!("  --fps-cap <N|off>         Frame cap target (default: 60)");
    println!(
        "  --pipeline <mode>         Runtime frame pipeline: parallel|legacy (default: parallel)"
    );
    println!("  --tick-profile <mode>     Physics tick planner: balanced|max (default: max)");
    println!("  --tick-cap <Hz|off>       Maximum physics tick Hz (default: off = auto)");
    println!("  --render-distance <N|off> Distance cull radius for 3D balls (default: auto)");
    println!("  --adaptive-distance <mode> Adaptive distance tuning: auto|on|off (default: auto)");
    println!("  --distance-blur <mode>    Distance haze blur: auto|on|off (default: auto)");
    println!("  --msaa <off|2|4>          MSAA sample count (default: 4)");
    println!("  --fsr off|auto|on         FSR policy mode (default: auto)");
    println!("  --fsr-quality <preset>    FSR quality: native|ultra|quality|balanced|performance");
    println!("  --fsr-sharpness <0..1>    FSR sharpen amount (default: 0.35)");
    println!("  --fsr-scale <0.5..1|auto> Render scale override for FSR mode");
    println!("  --fps-report <sec>        CLI FPS report cadence (default: 1.0)");
    println!("  --ini <path>              Startup config (.ini) file");
    println!("  --pak <path>              Asset package (.pak) to mount before scene load");
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
    use std::fs;
    use std::path::Path;

    use super::{apply_ini_overrides, scan_ini_path_from_args, CliOptions};

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
        use super::super::resolve_project_scheduler;
        use crate::{GraphicsSchedulerPath, RuntimePlatform, TlpfileGraphicsScheduler};
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
        use super::super::resolve_project_scheduler;
        use crate::{RuntimePlatform, TlpfileGraphicsScheduler};
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
        use super::super::resolve_project_scheduler;
        use crate::{GraphicsSchedulerPath, RuntimePlatform, TlpfileGraphicsScheduler};
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

    #[test]
    fn scan_ini_path_from_args_detects_flag() {
        let args = vec![
            "--vsync".to_string(),
            "off".to_string(),
            "--ini".to_string(),
            "config/tlapp.ini".to_string(),
        ];
        let ini = scan_ini_path_from_args(&args).expect("scan should succeed");
        assert_eq!(ini.as_deref(), Some(Path::new("config/tlapp.ini")));
    }

    #[test]
    fn ini_overrides_apply_paths_and_graphics_fields() {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let temp_dir =
            std::env::temp_dir().join(format!("tileline-cli-ini-{}-{stamp}", std::process::id(),));
        fs::create_dir_all(&temp_dir).expect("temp dir should be creatable");
        let ini_path = temp_dir.join("tlapp.ini");
        fs::write(
            &ini_path,
            r#"
                [tlapp]
                resolution = 1920x1080
                vsync = off
                fps_cap = off
                fsr = on
                fsr_quality = balanced
                fsr_sharpness = 0.55
                fsr_scale = 0.75
                script = assets/gameplay.tlscript
                sprite = assets/hud.tlsprite
                pak = assets/base.pak
            "#,
        )
        .expect("ini should be writable");

        let mut options = CliOptions::default();
        let warnings = apply_ini_overrides(&mut options, &ini_path).expect("ini should parse");
        assert!(warnings.is_empty(), "unexpected warnings: {warnings:?}");
        assert_eq!(options.resolution.width, 1920);
        assert_eq!(options.resolution.height, 1080);
        assert!(options.fps_cap.is_none());
        assert_eq!(options.fsr_sharpness, 0.55);
        assert_eq!(options.fsr_scale_override, Some(0.75));
        assert_eq!(options.script_path, Path::new("assets/gameplay.tlscript"));
        assert_eq!(options.sprite_path, Path::new("assets/hud.tlsprite"));
        assert_eq!(
            options.pak_path.as_deref(),
            Some(Path::new("assets/base.pak"))
        );

        let _ = fs::remove_file(&ini_path);
        let _ = fs::remove_dir_all(&temp_dir);
    }
}
