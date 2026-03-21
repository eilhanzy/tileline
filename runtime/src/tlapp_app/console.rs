use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Instant;

use winit::dpi::PhysicalSize;

use crate::{BounceTankRuntimePatch, TlscriptShowcaseFrameOutput};

#[derive(Debug, Clone)]
pub struct RuntimeConsoleState {
    pub open: bool,
    pub input_line: String,
    pub history: Vec<String>,
    pub history_cursor: Option<usize>,
    pub last_feedback: String,
    pub log_lines: Vec<RuntimeConsoleLogLine>,
    pub script_vars: BTreeMap<String, String>,
    pub script_statements: Vec<String>,
    pub script_overlay: TlscriptShowcaseFrameOutput,
    pub edit_target: RuntimeConsoleEditTarget,
    pub quick_fps_cap: String,
    pub quick_render_distance: String,
    pub quick_fsr_sharpness: String,
    pub quick_msaa: String,
    pub log_scroll: usize,
    pub log_filter: RuntimeConsoleLogFilter,
    pub log_tail_limit: Option<usize>,
    pub tail_follow: Option<RuntimeConsoleTailFollow>,
    pub file_watch: Option<RuntimeConsoleFileWatch>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeConsoleLogLevel {
    Info,
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeConsoleLogFilter {
    All,
    Info,
    Error,
}

impl RuntimeConsoleLogFilter {
    pub fn matches(self, level: RuntimeConsoleLogLevel) -> bool {
        match self {
            Self::All => true,
            Self::Info => matches!(level, RuntimeConsoleLogLevel::Info),
            Self::Error => matches!(level, RuntimeConsoleLogLevel::Error),
        }
    }
}

#[derive(Debug, Clone)]
pub struct RuntimeConsoleLogLine {
    pub timestamp: Instant,
    pub level: RuntimeConsoleLogLevel,
    pub message: String,
}

#[derive(Debug, Clone)]
pub struct RuntimeConsoleTailFollow {
    pub path: PathBuf,
    pub poll_interval: std::time::Duration,
    pub max_lines_per_poll: usize,
    pub cursor: u64,
    pub partial_line: String,
    pub last_poll: Instant,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone)]
pub struct RuntimeConsoleFileWatch {
    pub path: PathBuf,
    pub poll_interval: std::time::Duration,
    pub last_modified: Option<std::time::SystemTime>,
    pub last_len: u64,
    pub last_poll: Instant,
    pub last_error: Option<String>,
}

#[derive(Debug, Clone, Copy)]
pub struct ConsoleUiLayout {
    pub sx: f32,
    pub sy: f32,
    pub text_scale: f32,
    pub command_center: (f32, f32),
    pub command_size: (f32, f32),
    pub send_center: (f32, f32),
    pub send_size: (f32, f32),
    pub fps_center: (f32, f32),
    pub fps_size: (f32, f32),
    pub distance_center: (f32, f32),
    pub distance_size: (f32, f32),
    pub sharpness_center: (f32, f32),
    pub sharpness_size: (f32, f32),
    pub msaa_center: (f32, f32),
    pub msaa_size: (f32, f32),
    pub apply_center: (f32, f32),
    pub apply_size: (f32, f32),
}

impl ConsoleUiLayout {
    pub fn from_size(size: PhysicalSize<u32>) -> Self {
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
            msaa_center: (0.50 * sx, 0.405 * sy),
            msaa_size: (0.84 * text_scale, 0.06 * text_scale),
            apply_center: (0.52 * sx, 0.32 * sy),
            apply_size: (0.84 * text_scale, 0.06 * text_scale),
        }
    }

    #[inline]
    pub fn pos(self, x: f32, y: f32, z: f32) -> [f32; 3] {
        [x * self.sx, y * self.sy, z]
    }

    #[inline]
    pub fn rect_size(self, w: f32, h: f32) -> [f32; 2] {
        [w * self.sx, h * self.sy]
    }

    #[inline]
    pub fn glyph_size(self, w: f32, h: f32) -> [f32; 2] {
        [w * self.text_scale, h * self.text_scale]
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeConsoleEditTarget {
    Command,
    FpsCap,
    RenderDistance,
    FsrSharpness,
    Msaa,
}

impl RuntimeConsoleEditTarget {
    pub fn label(self) -> &'static str {
        match self {
            Self::Command => "command",
            Self::FpsCap => "fps_cap",
            Self::RenderDistance => "render_distance",
            Self::FsrSharpness => "fsr_sharpness",
            Self::Msaa => "msaa",
        }
    }

    pub fn next(self) -> Self {
        match self {
            Self::Command => Self::FpsCap,
            Self::FpsCap => Self::RenderDistance,
            Self::RenderDistance => Self::FsrSharpness,
            Self::FsrSharpness => Self::Msaa,
            Self::Msaa => Self::Command,
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
            quick_msaa: "4".to_string(),
            log_scroll: 0,
            log_filter: RuntimeConsoleLogFilter::All,
            log_tail_limit: None,
            tail_follow: None,
            file_watch: None,
        }
    }
}

pub fn empty_showcase_output() -> TlscriptShowcaseFrameOutput {
    TlscriptShowcaseFrameOutput {
        patch: BounceTankRuntimePatch::default(),
        light_overrides: Vec::new(),
        tile_mutations: Vec::new(),
        tile_fills: Vec::new(),
        performance_preset: None,
        gfx_profile: None,
        ball_metallic: None,
        ball_roughness: None,
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
