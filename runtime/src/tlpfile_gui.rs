//! General-purpose project GUI for `.tlpfile`-driven workflows.
//!
//! This is the first desktop shell that unifies `.tlscript`, `.tlsprite`, and `.tljoint`
//! composition under one project manifest.

use std::env;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::Instant;

use egui::{Color32, RichText};
use egui_wgpu::wgpu;
use egui_wgpu::{Renderer, RendererOptions, ScreenDescriptor};
use egui_winit::State as EguiWinitState;
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalSize};
use winit::event::WindowEvent;
use winit::event_loop::ActiveEventLoop;
#[cfg(target_os = "android")]
use winit::platform::android::activity::AndroidApp;
use winit::window::{Window, WindowAttributes, WindowId};

use crate::app_runner;
use crate::tlpfile::{
    compile_tlpfile_scene_from_path, load_tlpfile, parse_tlpfile, TlpfileDiagnostic,
    TlpfileDiagnosticLevel, TlpfileGraphicsScheduler, TlpfileParseOutcome, TlpfileProject,
    TlpfileSceneCompileOutcome,
};
use crate::TlscriptShowcaseConfig;

const PROJECT_SCAN_MAX_DEPTH: usize = 5;

#[derive(Debug, Clone)]
struct Cli {
    project_path: PathBuf,
    initial_scene: Option<String>,
    resolution: PhysicalSize<u32>,
    vsync: bool,
}

impl Default for Cli {
    fn default() -> Self {
        Self {
            project_path: PathBuf::from("docs/demos/tlapp/tlapp_project.tlpfile"),
            initial_scene: None,
            resolution: PhysicalSize::new(1280, 820),
            vsync: true,
        }
    }
}

impl Cli {
    fn parse_from_env() -> Result<Self, Box<dyn Error>> {
        let mut cli = Self::default();
        let mut args = env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--project" => {
                    let value = args
                        .next()
                        .ok_or_else(|| String::from("missing value for --project"))?;
                    cli.project_path = PathBuf::from(value);
                }
                "--scene" => {
                    let value = args
                        .next()
                        .ok_or_else(|| String::from("missing value for --scene"))?;
                    cli.initial_scene = Some(value);
                }
                "--resolution" => {
                    let value = args
                        .next()
                        .ok_or_else(|| String::from("missing value for --resolution"))?;
                    cli.resolution = parse_resolution(&value)?;
                }
                "--vsync" => {
                    let value = args
                        .next()
                        .ok_or_else(|| String::from("missing value for --vsync"))?;
                    cli.vsync = parse_vsync(&value)?;
                }
                "-h" | "--help" => {
                    print_usage();
                    std::process::exit(0);
                }
                other => return Err(format!("unknown arg '{other}'").into()),
            }
        }
        Ok(cli)
    }
}

fn parse_resolution(value: &str) -> Result<PhysicalSize<u32>, Box<dyn Error>> {
    let lower = value.to_ascii_lowercase();
    let (w, h) = lower
        .split_once('x')
        .ok_or_else(|| format!("resolution must be WxH, got '{value}'"))?;
    let width = w
        .parse::<u32>()
        .map_err(|_| format!("invalid width in --resolution: '{w}'"))?;
    let height = h
        .parse::<u32>()
        .map_err(|_| format!("invalid height in --resolution: '{h}'"))?;
    if width == 0 || height == 0 {
        return Err("resolution cannot be zero".into());
    }
    Ok(PhysicalSize::new(width, height))
}

fn parse_vsync(value: &str) -> Result<bool, Box<dyn Error>> {
    Ok(match value.to_ascii_lowercase().as_str() {
        "on" | "true" | "1" => true,
        "off" | "false" | "0" => false,
        other => return Err(format!("invalid --vsync value '{other}', use on/off").into()),
    })
}

fn print_usage() {
    println!("Tileline Project GUI (.tlpfile)");
    println!("Usage: cargo run -p runtime --bin tlproject_gui -- [options]");
    println!("Options:");
    println!(
        "  --project <path>     .tlpfile path (default: docs/demos/tlapp/tlapp_project.tlpfile)"
    );
    println!("  --scene <name>       initial scene override");
    println!("  --resolution <WxH>   window resolution (default: 1280x820)");
    println!("  --vsync on|off       present mode policy (default: on)");
    println!("  -h, --help           show help");
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlaybackState {
    Stopped,
    Running,
    Paused,
}

#[derive(Debug, Clone)]
struct PreviewBall {
    pos: [f32; 2],
    vel: [f32; 2],
    radius: f32,
    color: Color32,
}

#[derive(Debug, Clone)]
struct ScenePreviewState {
    playback: PlaybackState,
    light_mode: bool,
    balls: Vec<PreviewBall>,
    last_tick: Instant,
    accumulator: f32,
    tool_anchor: [f32; 2],
    tool_rotation_rad: f32,
}

impl ScenePreviewState {
    fn new() -> Self {
        let mut state = Self {
            playback: PlaybackState::Stopped,
            light_mode: true,
            balls: Vec::new(),
            last_tick: Instant::now(),
            accumulator: 0.0,
            tool_anchor: [0.5, 0.5],
            tool_rotation_rad: 0.0,
        };
        state.reset_balls();
        state
    }

    fn set_light_mode(&mut self, enabled: bool) {
        if self.light_mode == enabled {
            return;
        }
        self.light_mode = enabled;
        self.reset_balls();
    }

    fn start(&mut self) {
        self.playback = PlaybackState::Running;
        self.last_tick = Instant::now();
    }

    fn pause(&mut self) {
        self.playback = PlaybackState::Paused;
    }

    fn stop(&mut self) {
        self.playback = PlaybackState::Stopped;
        self.accumulator = 0.0;
        self.reset_balls();
    }

    fn reset_balls(&mut self) {
        let count = if self.light_mode { 36 } else { 96 };
        let mut balls = Vec::with_capacity(count);
        for idx in 0..count {
            let x = ((idx % 12) as f32 + 0.5) / 12.0;
            let y = ((idx / 12) as f32 + 1.0) / ((count / 12).max(1) as f32 + 2.0);
            let hue = (idx as f32 * 23.0) % 255.0;
            let color = Color32::from_rgb(
                ((80.0 + hue * 0.7) as i32).clamp(0, 255) as u8,
                ((120.0 + hue * 0.4) as i32).clamp(0, 255) as u8,
                ((220.0 - hue * 0.5) as i32).clamp(0, 255) as u8,
            );
            balls.push(PreviewBall {
                pos: [x * 0.8 + 0.1, y * 0.6 + 0.1],
                vel: [0.04 * ((idx % 5) as f32 - 2.0), 0.0],
                radius: if self.light_mode { 0.012 } else { 0.010 },
                color,
            });
        }
        self.balls = balls;
    }

    fn apply_transform_delta(
        &mut self,
        coordinate_space: EditorCoordinateSpace,
        move_delta: [f32; 3],
        rotate_delta_deg: [f32; 2],
    ) {
        let scale = if self.light_mode { 0.04 } else { 0.03 };
        let local_xy = [move_delta[0], -move_delta[1]];
        let world_xy = match coordinate_space {
            EditorCoordinateSpace::World => local_xy,
            EditorCoordinateSpace::Local => {
                let (s, c) = self.tool_rotation_rad.sin_cos();
                [
                    local_xy[0] * c - local_xy[1] * s,
                    local_xy[0] * s + local_xy[1] * c,
                ]
            }
        };
        self.tool_anchor[0] = (self.tool_anchor[0] + world_xy[0] * scale).clamp(0.1, 0.9);
        self.tool_anchor[1] = (self.tool_anchor[1] + world_xy[1] * scale).clamp(0.1, 0.9);
        self.tool_rotation_rad += rotate_delta_deg[0].to_radians();
    }

    fn tick(&mut self) {
        let now = Instant::now();
        let mut dt = (now - self.last_tick).as_secs_f32();
        self.last_tick = now;

        if self.playback != PlaybackState::Running {
            return;
        }

        dt = dt.clamp(0.0, 0.05);
        let step = if self.light_mode {
            1.0 / 30.0
        } else {
            1.0 / 90.0
        };
        self.accumulator += dt;
        let mut iterations = 0usize;
        while self.accumulator >= step && iterations < 8 {
            self.integrate(step);
            self.accumulator -= step;
            iterations += 1;
        }
    }

    fn integrate(&mut self, dt: f32) {
        let gravity = if self.light_mode { 0.55 } else { 0.75 };
        for ball in &mut self.balls {
            ball.vel[1] += gravity * dt;
            ball.pos[0] += ball.vel[0] * dt;
            ball.pos[1] += ball.vel[1] * dt;

            if ball.pos[0] - ball.radius < 0.0 {
                ball.pos[0] = ball.radius;
                ball.vel[0] = ball.vel[0].abs() * 0.85;
            }
            if ball.pos[0] + ball.radius > 1.0 {
                ball.pos[0] = 1.0 - ball.radius;
                ball.vel[0] = -ball.vel[0].abs() * 0.85;
            }
            if ball.pos[1] - ball.radius < 0.0 {
                ball.pos[1] = ball.radius;
                ball.vel[1] = ball.vel[1].abs() * 0.85;
            }
            if ball.pos[1] + ball.radius > 1.0 {
                ball.pos[1] = 1.0 - ball.radius;
                ball.vel[1] = -ball.vel[1].abs() * 0.75;
            }
        }
    }

    fn apply_touch_pan(&mut self, delta_pixels: [f32; 2], viewport: [f32; 2]) {
        let w = viewport[0].max(1.0);
        let h = viewport[1].max(1.0);
        self.tool_anchor[0] = (self.tool_anchor[0] + (delta_pixels[0] / w)).clamp(0.1, 0.9);
        self.tool_anchor[1] = (self.tool_anchor[1] + (delta_pixels[1] / h)).clamp(0.1, 0.9);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProjectFileKind {
    Directory,
    Project,
    Joint,
    Script,
    Sprite,
    Other,
}

impl ProjectFileKind {
    fn icon(self) -> &'static str {
        match self {
            Self::Directory => "DIR",
            Self::Project => "TLP",
            Self::Joint => "JNT",
            Self::Script => "SCR",
            Self::Sprite => "SPR",
            Self::Other => "FIL",
        }
    }
}

#[derive(Debug, Clone)]
struct ProjectFileEntry {
    absolute_path: PathBuf,
    relative_path: String,
    kind: ProjectFileKind,
    depth: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NewAssetKind {
    Tlscript,
    Tlsprite,
    Tljoint,
}

impl NewAssetKind {
    fn label(self) -> &'static str {
        match self {
            Self::Tlscript => ".tlscript",
            Self::Tlsprite => ".tlsprite",
            Self::Tljoint => ".tljoint",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EditorCoordinateSpace {
    World,
    Local,
}

impl EditorCoordinateSpace {
    fn as_tlscript_str(self) -> &'static str {
        match self {
            Self::World => "world",
            Self::Local => "local",
        }
    }
}

#[derive(Debug, Clone)]
struct TransformToolState {
    coordinate_space: EditorCoordinateSpace,
    move_delta: [f32; 3],
    rotate_delta_deg: [f32; 2],
    move_step: f32,
    rotate_step_deg: f32,
}

impl Default for TransformToolState {
    fn default() -> Self {
        Self {
            coordinate_space: EditorCoordinateSpace::World,
            move_delta: [0.0, 0.0, 0.0],
            rotate_delta_deg: [0.0, 0.0],
            move_step: 0.25,
            rotate_step_deg: 5.0,
        }
    }
}

#[derive(Debug)]
struct UiModel {
    project_path: PathBuf,
    project_root: PathBuf,
    selected_scene: String,
    parse: Option<TlpfileParseOutcome>,
    compile: Option<TlpfileSceneCompileOutcome>,
    status: String,
    project_source: String,
    editor_dirty: bool,
    editor_status: String,
    preview: ScenePreviewState,
    scheduler_override: TlpfileGraphicsScheduler,
    project_files: Vec<ProjectFileEntry>,
    selected_file: Option<PathBuf>,
    new_asset_name: String,
    new_asset_kind: NewAssetKind,
    creation_status: String,
    runtime_child: Option<Child>,
    runtime_status: String,
    runtime_last_command: String,
    transform_tool: TransformToolState,
    transform_status: String,
    android_editor_lite: bool,
}

impl UiModel {
    fn new(project_path: PathBuf, scene_override: Option<String>) -> Self {
        let project_root = project_path
            .parent()
            .map_or_else(|| PathBuf::from("."), Path::to_path_buf);
        let mut model = Self {
            project_path,
            project_root,
            selected_scene: scene_override.unwrap_or_default(),
            parse: None,
            compile: None,
            status: String::new(),
            project_source: String::new(),
            editor_dirty: false,
            editor_status: String::new(),
            preview: ScenePreviewState::new(),
            scheduler_override: TlpfileGraphicsScheduler::Auto,
            project_files: Vec::new(),
            selected_file: None,
            new_asset_name: String::from("new_scene"),
            new_asset_kind: NewAssetKind::Tlscript,
            creation_status: String::new(),
            runtime_child: None,
            runtime_status: String::from("runtime idle"),
            runtime_last_command: String::new(),
            transform_tool: TransformToolState::default(),
            transform_status: String::new(),
            android_editor_lite: cfg!(target_os = "android"),
        };
        model.reload_parse();
        model.compile_selected_scene();
        model
    }

    fn reload_parse(&mut self) {
        self.reload_source_from_disk();
        self.refresh_project_files();
        match load_tlpfile(&self.project_path) {
            Ok(outcome) => {
                self.status = format!(
                    "loaded project: parse_errors={} parse_warnings={}",
                    count_diagnostics(&outcome.diagnostics, |level| level
                        == TlpfileDiagnosticLevel::Error),
                    count_diagnostics(&outcome.diagnostics, |level| level
                        == TlpfileDiagnosticLevel::Warning)
                );
                if let Some(project) = outcome.project.as_ref() {
                    if self.selected_scene.is_empty()
                        || project.scene(&self.selected_scene).is_none()
                    {
                        self.selected_scene = project.default_scene.clone();
                    }
                    self.scheduler_override = project.scheduler;
                }
                self.parse = Some(outcome);
            }
            Err(err) => {
                self.status = format!(
                    "failed to load project '{}': {err}",
                    self.project_path.display()
                );
                self.parse = Some(TlpfileParseOutcome {
                    project: None,
                    diagnostics: vec![TlpfileDiagnostic {
                        level: TlpfileDiagnosticLevel::Error,
                        line: 1,
                        message: err,
                    }],
                });
            }
        }
        self.compile = None;
    }

    fn compile_selected_scene(&mut self) {
        if self.editor_dirty {
            if let Err(err) = self.save_source_to_disk() {
                self.status = format!("compile blocked: {err}");
                self.compile = None;
                return;
            }
        }
        if self.selected_scene.is_empty() {
            self.status = "no scene selected".to_string();
            self.compile = None;
            return;
        }
        let outcome = compile_tlpfile_scene_from_path(
            &self.project_path,
            Some(self.selected_scene.as_str()),
            TlscriptShowcaseConfig::default(),
        );
        let errors = count_diagnostics(&outcome.diagnostics, |level| {
            level == TlpfileDiagnosticLevel::Error
        });
        let warnings = count_diagnostics(&outcome.diagnostics, |level| {
            level == TlpfileDiagnosticLevel::Warning
        });
        if let Some(bundle) = &outcome.bundle {
            self.status = format!(
                "scene '{}' compiled | scheduler={} scripts={} sprites={} warnings={warnings}",
                bundle.scene_name,
                bundle.scheduler.as_str(),
                bundle.scripts.len(),
                bundle.sprite_count(),
            );
        } else {
            self.status = format!(
                "scene '{}' failed | errors={errors} warnings={warnings}",
                self.selected_scene
            );
        }
        self.compile = Some(outcome);
    }

    fn project(&self) -> Option<&TlpfileProject> {
        self.parse.as_ref().and_then(|parse| parse.project.as_ref())
    }

    fn reload_source_from_disk(&mut self) {
        match fs::read_to_string(&self.project_path) {
            Ok(source) => {
                self.project_source = source;
                self.editor_dirty = false;
                self.editor_status = "buffer reloaded from disk".to_string();
                self.project_root = self
                    .project_path
                    .parent()
                    .map_or_else(|| PathBuf::from("."), Path::to_path_buf);
            }
            Err(err) => {
                self.project_source.clear();
                self.editor_dirty = false;
                self.editor_status = format!("failed to read source: {err}");
            }
        }
    }

    fn save_source_to_disk(&mut self) -> Result<(), String> {
        fs::write(&self.project_path, &self.project_source)
            .map_err(|err| format!("failed to save '{}': {err}", self.project_path.display()))?;
        self.editor_dirty = false;
        self.editor_status = format!("saved {}", self.project_path.display());
        self.refresh_project_files();
        Ok(())
    }

    fn reparse_from_buffer(&mut self) {
        let outcome = parse_tlpfile(&self.project_source);
        if let Some(project) = outcome.project.as_ref() {
            if self.selected_scene.is_empty() || project.scene(&self.selected_scene).is_none() {
                self.selected_scene = project.default_scene.clone();
            }
            self.scheduler_override = project.scheduler;
        }
        self.parse = Some(outcome);
        self.compile = None;
        self.status = "reparsed in-memory .tlpfile buffer".to_string();
    }

    fn apply_scheduler_override(&mut self) {
        self.project_source =
            upsert_project_scheduler(&self.project_source, self.scheduler_override);
        self.editor_dirty = true;
        self.editor_status = format!(
            "scheduler override set to {} (buffer modified)",
            self.scheduler_override.as_str()
        );
        self.reparse_from_buffer();
    }

    fn refresh_project_files(&mut self) {
        self.project_files.clear();
        if let Err(err) = collect_project_files(
            &self.project_root,
            &self.project_root,
            0,
            &mut self.project_files,
        ) {
            self.creation_status = format!("file explorer refresh failed: {err}");
            return;
        }
        self.project_files
            .sort_by(|a, b| a.relative_path.cmp(&b.relative_path));
    }

    fn create_scene_pack(&mut self) {
        let raw_name = self.new_asset_name.trim();
        if raw_name.is_empty() {
            self.creation_status = "scene pack name cannot be empty".to_string();
            return;
        }
        let scene_name = sanitize_scene_name(raw_name);
        if scene_name.is_empty() {
            self.creation_status = "scene pack name is not valid".to_string();
            return;
        }

        if let Err(err) = fs::create_dir_all(&self.project_root) {
            self.creation_status = format!("failed to create project root: {err}");
            return;
        }

        let joint_name = format!("{scene_name}.tljoint");
        let script_name = format!("{scene_name}.tlscript");
        let sprite_name = format!("{scene_name}_hud.tlsprite");
        let joint_path = self.project_root.join(&joint_name);
        let script_path = self.project_root.join(&script_name);
        let sprite_path = self.project_root.join(&sprite_name);

        if let Err(err) = write_if_missing(&script_path, &default_script_template()) {
            self.creation_status = format!("failed to create script: {err}");
            return;
        }
        if let Err(err) = write_if_missing(&sprite_path, &default_sprite_template()) {
            self.creation_status = format!("failed to create sprite: {err}");
            return;
        }
        let joint_source = format!(
            "tljoint_v1\n\n[scene.{scene_name}]\ntlscripts = {script_name}\ntlsprites = {sprite_name}\n"
        );
        if let Err(err) = write_if_missing(&joint_path, &joint_source) {
            self.creation_status = format!("failed to create joint: {err}");
            return;
        }

        append_scene_section_to_project_source(&mut self.project_source, &scene_name, &joint_name);
        self.editor_dirty = true;
        self.selected_scene = scene_name.clone();
        self.creation_status =
            format!("scene pack created: {joint_name}, {script_name}, {sprite_name}");
        self.editor_status = "scene pack appended to .tlpfile buffer".to_string();
        self.refresh_project_files();
        self.reparse_from_buffer();
    }

    fn create_single_asset(&mut self) {
        let raw_name = self.new_asset_name.trim();
        if raw_name.is_empty() {
            self.creation_status = "asset name cannot be empty".to_string();
            return;
        }
        if let Err(err) = fs::create_dir_all(&self.project_root) {
            self.creation_status = format!("failed to create project root: {err}");
            return;
        }

        let (file_name, template) = match self.new_asset_kind {
            NewAssetKind::Tlscript => {
                (ensure_ext(raw_name, ".tlscript"), default_script_template())
            }
            NewAssetKind::Tlsprite => {
                (ensure_ext(raw_name, ".tlsprite"), default_sprite_template())
            }
            NewAssetKind::Tljoint => (
                ensure_ext(raw_name, ".tljoint"),
                String::from("tljoint_v1\n\n[scene.main]\n"),
            ),
        };
        let path = self.project_root.join(&file_name);
        match write_if_missing(&path, &template) {
            Ok(created) => {
                self.creation_status = if created {
                    format!("created {}", path.display())
                } else {
                    format!("already exists {}", path.display())
                };
                self.selected_file = Some(path);
                self.refresh_project_files();
            }
            Err(err) => self.creation_status = format!("asset creation failed: {err}"),
        }
    }

    fn poll_runtime_process(&mut self) {
        let Some(child) = self.runtime_child.as_mut() else {
            return;
        };
        match child.try_wait() {
            Ok(Some(status)) => {
                self.runtime_status = format!("runtime exited ({status})");
                self.runtime_child = None;
            }
            Ok(None) => {
                self.runtime_status = format!("runtime running (pid {})", child.id());
            }
            Err(err) => {
                self.runtime_status = format!("runtime poll error: {err}");
                self.runtime_child = None;
            }
        }
    }

    fn runtime_scene_name(&self) -> String {
        if self.selected_scene.trim().is_empty() {
            String::from("main")
        } else {
            self.selected_scene.trim().to_string()
        }
    }

    fn launch_runtime(&mut self) {
        self.poll_runtime_process();
        if self.runtime_child.is_some() {
            self.runtime_status = "runtime already running".to_string();
            return;
        }

        let scene = self.runtime_scene_name();
        let mut command = if let Some(binary) = tlapp_binary_candidate() {
            if binary.exists() {
                self.runtime_last_command = format!(
                    "{} --project {} --scene {}",
                    binary.display(),
                    self.project_path.display(),
                    scene
                );
                let mut command = Command::new(binary);
                command
                    .arg("--project")
                    .arg(&self.project_path)
                    .arg("--scene")
                    .arg(&scene);
                command
            } else {
                self.runtime_last_command = format!(
                    "cargo run -p runtime --bin tlapp -- --project {} --scene {}",
                    self.project_path.display(),
                    scene
                );
                let mut command = Command::new("cargo");
                command
                    .arg("run")
                    .arg("-p")
                    .arg("runtime")
                    .arg("--bin")
                    .arg("tlapp")
                    .arg("--")
                    .arg("--project")
                    .arg(&self.project_path)
                    .arg("--scene")
                    .arg(&scene);
                command
            }
        } else {
            self.runtime_last_command = format!(
                "cargo run -p runtime --bin tlapp -- --project {} --scene {}",
                self.project_path.display(),
                scene
            );
            let mut command = Command::new("cargo");
            command
                .arg("run")
                .arg("-p")
                .arg("runtime")
                .arg("--bin")
                .arg("tlapp")
                .arg("--")
                .arg("--project")
                .arg(&self.project_path)
                .arg("--scene")
                .arg(&scene);
            command
        };

        command
            .current_dir(workspace_root_hint())
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        match command.spawn() {
            Ok(child) => {
                self.runtime_status = format!("runtime launched (pid {})", child.id());
                self.runtime_child = Some(child);
            }
            Err(err) => {
                self.runtime_status = format!("runtime launch failed: {err}");
            }
        }
    }

    fn stop_runtime(&mut self) {
        let Some(mut child) = self.runtime_child.take() else {
            self.runtime_status = "runtime is not running".to_string();
            return;
        };
        let _ = child.kill();
        let _ = child.wait();
        self.runtime_status = "runtime stopped".to_string();
    }

    fn apply_transform_to_preview(&mut self) {
        self.preview.apply_transform_delta(
            self.transform_tool.coordinate_space,
            self.transform_tool.move_delta,
            self.transform_tool.rotate_delta_deg,
        );
        self.transform_status = format!(
            "applied transform | space={} move=({:.2},{:.2},{:.2}) rotate=({:.2},{:.2})",
            self.transform_tool.coordinate_space.as_tlscript_str(),
            self.transform_tool.move_delta[0],
            self.transform_tool.move_delta[1],
            self.transform_tool.move_delta[2],
            self.transform_tool.rotate_delta_deg[0],
            self.transform_tool.rotate_delta_deg[1],
        );
    }

    fn append_transform_to_selected_script(&mut self) {
        let Some(script_path) = self.resolve_target_tlscript_path() else {
            self.transform_status =
                "select a .tlscript file in Project Files (or compile a scene first)".to_string();
            return;
        };
        let snippet = format!(
            "\n# Editor transform sync\nset_coordinate_space(\"{}\")\nmove_camera({:.4}, {:.4}, {:.4})\nrotate_camera({:.4}, {:.4})\n",
            self.transform_tool.coordinate_space.as_tlscript_str(),
            self.transform_tool.move_delta[0],
            self.transform_tool.move_delta[1],
            self.transform_tool.move_delta[2],
            self.transform_tool.rotate_delta_deg[0],
            self.transform_tool.rotate_delta_deg[1],
        );
        match fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(&script_path)
        {
            Ok(mut file) => {
                use std::io::Write;
                if let Err(err) = file.write_all(snippet.as_bytes()) {
                    self.transform_status =
                        format!("failed to write script '{}': {err}", script_path.display());
                    return;
                }
                self.transform_status =
                    format!("transform snippet appended to {}", script_path.display());
                self.selected_file = Some(script_path);
                self.refresh_project_files();
            }
            Err(err) => {
                self.transform_status =
                    format!("failed to open script '{}': {err}", script_path.display());
            }
        }
    }

    fn resolve_target_tlscript_path(&self) -> Option<PathBuf> {
        if let Some(path) = self.selected_file.as_ref() {
            if path.extension().and_then(|ext| ext.to_str()) == Some("tlscript") {
                return Some(path.clone());
            }
        }
        self.compile.as_ref().and_then(|compile| {
            compile
                .bundle
                .as_ref()
                .and_then(|bundle| bundle.script_paths.first().cloned())
        })
    }
}

impl Drop for UiModel {
    fn drop(&mut self) {
        if let Some(mut child) = self.runtime_child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

struct GuiRuntime {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    egui_ctx: egui::Context,
    egui_state: EguiWinitState,
    egui_renderer: Renderer,
    model: UiModel,
}

impl GuiRuntime {
    fn new(event_loop: &ActiveEventLoop, cli: &Cli) -> Result<Self, Box<dyn Error>> {
        let window = Arc::new(
            event_loop.create_window(
                WindowAttributes::default()
                    .with_title("Tileline Project GUI")
                    .with_inner_size(LogicalSize::new(
                        cli.resolution.width as f64,
                        cli.resolution.height as f64,
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
        let adapter_info = adapter.get_info();
        let supported_limits = adapter.limits();
        let mut required_limits = wgpu::Limits::default();
        let mut any_clamped = false;
        if required_limits.max_texture_dimension_1d > supported_limits.max_texture_dimension_1d {
            required_limits.max_texture_dimension_1d = supported_limits.max_texture_dimension_1d;
            any_clamped = true;
        }
        if required_limits.max_texture_dimension_2d > supported_limits.max_texture_dimension_2d {
            required_limits.max_texture_dimension_2d = supported_limits.max_texture_dimension_2d;
            any_clamped = true;
        }
        if required_limits.max_texture_dimension_3d > supported_limits.max_texture_dimension_3d {
            required_limits.max_texture_dimension_3d = supported_limits.max_texture_dimension_3d;
            any_clamped = true;
        }
        if required_limits.max_texture_array_layers > supported_limits.max_texture_array_layers {
            required_limits.max_texture_array_layers = supported_limits.max_texture_array_layers;
            any_clamped = true;
        }
        if any_clamped {
            eprintln!(
                "[tlproject-gui limits] adapter='{}' clamped required limits to supported values (1d={}, 2d={}, 3d={}, layers={})",
                adapter_info.name,
                required_limits.max_texture_dimension_1d,
                required_limits.max_texture_dimension_2d,
                required_limits.max_texture_dimension_3d,
                required_limits.max_texture_array_layers
            );
        }
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("tlproject-gui-device"),
                required_limits,
                ..Default::default()
            }))?;
        let mut config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .ok_or("surface is not compatible with selected adapter")?;
        config.present_mode = if cli.vsync {
            wgpu::PresentMode::Fifo
        } else {
            wgpu::PresentMode::AutoNoVsync
        };
        surface.configure(&device, &config);

        let egui_ctx = egui::Context::default();
        apply_lavender_theme(&egui_ctx);
        let egui_state = EguiWinitState::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            window.as_ref(),
            Some(window.scale_factor() as f32),
            window.theme(),
            Some(device.limits().max_texture_dimension_2d as usize),
        );
        let egui_renderer = Renderer::new(&device, config.format, RendererOptions::default());

        Ok(Self {
            window,
            surface,
            device,
            queue,
            config,
            egui_ctx,
            egui_state,
            egui_renderer,
            model: UiModel::new(cli.project_path.clone(), cli.initial_scene.clone()),
        })
    }

    fn resize(&mut self, size: PhysicalSize<u32>) {
        if size.width == 0 || size.height == 0 {
            return;
        }
        self.config.width = size.width;
        self.config.height = size.height;
        self.surface.configure(&self.device, &self.config);
    }

    fn handle_window_event(&mut self, event: &WindowEvent) {
        let _ = self.egui_state.on_window_event(self.window.as_ref(), event);
    }

    fn render(&mut self) -> Result<(), Box<dyn Error>> {
        self.model.poll_runtime_process();
        self.model.preview.tick();
        let raw_input = self.egui_state.take_egui_input(self.window.as_ref());
        let egui_ctx = self.egui_ctx.clone();
        let full_output = egui_ctx.run(raw_input, |ctx| self.draw_ui(ctx));
        self.egui_state
            .handle_platform_output(self.window.as_ref(), full_output.platform_output);

        let pixels_per_point = self.window.scale_factor() as f32;
        let paint_jobs = egui_ctx.tessellate(full_output.shapes, pixels_per_point);
        let screen_descriptor = ScreenDescriptor {
            size_in_pixels: [self.config.width.max(1), self.config.height.max(1)],
            pixels_per_point,
        };

        for (id, image_delta) in &full_output.textures_delta.set {
            self.egui_renderer
                .update_texture(&self.device, &self.queue, *id, image_delta);
        }

        let surface_texture = match self.surface.get_current_texture() {
            Ok(frame) => frame,
            Err(wgpu::SurfaceError::Outdated | wgpu::SurfaceError::Lost) => {
                self.surface.configure(&self.device, &self.config);
                return Ok(());
            }
            Err(wgpu::SurfaceError::OutOfMemory) => {
                return Err("surface out of memory".into());
            }
            Err(wgpu::SurfaceError::Timeout) => {
                return Ok(());
            }
            Err(wgpu::SurfaceError::Other) => {
                return Ok(());
            }
        };

        let view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("tlproject-gui-encoder"),
            });

        let mut user_cmds = self.egui_renderer.update_buffers(
            &self.device,
            &self.queue,
            &mut encoder,
            &paint_jobs,
            &screen_descriptor,
        );

        {
            let render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("tlproject-gui-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.08,
                            g: 0.07,
                            b: 0.12,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            let mut render_pass = render_pass.forget_lifetime();
            self.egui_renderer
                .render(&mut render_pass, &paint_jobs, &screen_descriptor);
        }

        for texture_id in &full_output.textures_delta.free {
            self.egui_renderer.free_texture(texture_id);
        }

        user_cmds.push(encoder.finish());
        self.queue.submit(user_cmds);
        surface_texture.present();

        self.window.request_redraw();
        Ok(())
    }

    fn draw_ui(&mut self, ctx: &egui::Context) {
        let project_meta = self.model.project().map(|project| {
            (
                project.name.clone(),
                project.default_scene.clone(),
                project.scheduler,
                project
                    .scenes
                    .iter()
                    .map(|scene| scene.scene.clone())
                    .collect::<Vec<_>>(),
            )
        });

        egui::TopBottomPanel::top("top_panel").show(ctx, |ui| {
            ui.horizontal_wrapped(|ui| {
                ui.heading("Tileline Project GUI");
                ui.separator();
                ui.label(RichText::new(self.model.project_path.display().to_string()).monospace());

                if icon_button(ui, "R", "Reload").clicked() {
                    self.model.reload_parse();
                }
                if icon_button(ui, "C", "Compile").clicked() {
                    self.model.compile_selected_scene();
                }
                ui.separator();

                if icon_button(ui, ">", "Start").clicked() {
                    self.model.preview.start();
                }
                if icon_button(ui, "||", "Pause").clicked() {
                    self.model.preview.pause();
                }
                if icon_button(ui, "[]", "Stop").clicked() {
                    self.model.preview.stop();
                }

                let mut light_mode = self.model.preview.light_mode;
                if ui.checkbox(&mut light_mode, "Light Mode").changed() {
                    self.model.preview.set_light_mode(light_mode);
                }

                ui.separator();
                let mut scheduler = self.model.scheduler_override;
                egui::ComboBox::from_label("Scheduler")
                    .selected_text(scheduler.as_str())
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut scheduler, TlpfileGraphicsScheduler::Auto, "auto");
                        ui.selectable_value(&mut scheduler, TlpfileGraphicsScheduler::Gms, "gms");
                        ui.selectable_value(&mut scheduler, TlpfileGraphicsScheduler::Mgs, "mgs");
                    });
                if scheduler != self.model.scheduler_override {
                    self.model.scheduler_override = scheduler;
                }
                if icon_button(ui, "AS", "Apply Scheduler").clicked() {
                    self.model.apply_scheduler_override();
                }
                ui.separator();
                if icon_button(ui, "RUN", "Run TLApp").clicked() {
                    self.model.launch_runtime();
                }
                if icon_button(ui, "KILL", "Stop TLApp").clicked() {
                    self.model.stop_runtime();
                }
                ui.separator();
                ui.label(format!("Scene: {}", self.model.selected_scene));
            });
            ui.label(
                RichText::new(self.model.status.clone()).color(Color32::from_rgb(188, 174, 230)),
            );
            if self.model.editor_dirty {
                ui.label(
                    RichText::new(format!("editor: {} (unsaved)", self.model.editor_status))
                        .color(Color32::from_rgb(255, 210, 130)),
                );
            } else {
                ui.label(
                    RichText::new(format!("editor: {}", self.model.editor_status))
                        .color(Color32::from_rgb(166, 214, 255)),
                );
            }
            if !self.model.creation_status.is_empty() {
                ui.label(
                    RichText::new(format!("workspace: {}", self.model.creation_status))
                        .color(Color32::from_rgb(190, 242, 190)),
                );
            }
            ui.label(
                RichText::new(format!("runtime: {}", self.model.runtime_status))
                    .color(Color32::from_rgb(172, 224, 250)),
            );
            if !self.model.runtime_last_command.is_empty() {
                ui.label(
                    RichText::new(format!("runtime cmd: {}", self.model.runtime_last_command))
                        .small()
                        .monospace()
                        .color(Color32::from_rgb(152, 192, 226)),
                );
            }
            if !self.model.transform_status.is_empty() {
                ui.label(
                    RichText::new(format!("transform: {}", self.model.transform_status))
                        .color(Color32::from_rgb(218, 194, 255)),
                );
            }
        });

        if !self.model.android_editor_lite {
            egui::SidePanel::left("scene_panel")
                .resizable(true)
                .default_width(320.0)
                .show(ctx, |ui| {
                ui.heading("Scenes");
                if let Some((_, _, _, scene_names)) = project_meta.as_ref() {
                    egui::ScrollArea::vertical()
                        .max_height(160.0)
                        .show(ui, |ui| {
                            for scene_name in scene_names {
                                let selected = self.model.selected_scene == *scene_name;
                                if ui.selectable_label(selected, scene_name).clicked() {
                                    self.model.selected_scene = scene_name.clone();
                                    self.model.compile_selected_scene();
                                }
                            }
                        });
                } else {
                    ui.label("No parsed project");
                }
                ui.separator();
                ui.label(format!(
                    "Playback: {}",
                    playback_label(self.model.preview.playback)
                ));
                ui.label(format!("Preview balls: {}", self.model.preview.balls.len()));

                ui.separator();
                ui.heading("Project Files");
                let file_entries = self.model.project_files.clone();
                egui::ScrollArea::vertical()
                    .max_height(260.0)
                    .show(ui, |ui| {
                        for entry in file_entries {
                            let indent = "  ".repeat(entry.depth);
                            let label = format!(
                                "{indent}[{}] {}",
                                entry.kind.icon(),
                                entry.relative_path
                            );
                            let selected = self
                                .model
                                .selected_file
                                .as_ref()
                                .map(|path| path == &entry.absolute_path)
                                .unwrap_or(false);
                            if ui.selectable_label(selected, label).clicked() {
                                self.model.selected_file = Some(entry.absolute_path.clone());
                                self.model.creation_status =
                                    format!("selected {}", entry.relative_path);
                            }
                        }
                    });

                ui.separator();
                ui.heading("Create");
                ui.label("Name");
                ui.add(egui::TextEdit::singleline(&mut self.model.new_asset_name));
                egui::ComboBox::from_label("Asset Type")
                    .selected_text(self.model.new_asset_kind.label())
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut self.model.new_asset_kind,
                            NewAssetKind::Tlscript,
                            ".tlscript",
                        );
                        ui.selectable_value(
                            &mut self.model.new_asset_kind,
                            NewAssetKind::Tlsprite,
                            ".tlsprite",
                        );
                        ui.selectable_value(
                            &mut self.model.new_asset_kind,
                            NewAssetKind::Tljoint,
                            ".tljoint",
                        );
                    });
                ui.horizontal_wrapped(|ui| {
                    if icon_button(ui, "+F", "Create File").clicked() {
                        self.model.create_single_asset();
                    }
                    if icon_button(ui, "+S", "Create Scene Pack").clicked() {
                        self.model.create_scene_pack();
                    }
                });
                ui.label(
                    RichText::new(
                        "Scene pack creates [.tlscript + .tlsprite + .tljoint] and appends [scene.*] to .tlpfile.",
                    )
                    .small()
                    .color(Color32::from_rgb(176, 164, 220)),
                );
                });
        }

        egui::SidePanel::right("diagnostics_panel")
            .resizable(true)
            .default_width(380.0)
            .show(ctx, |ui| {
                ui.heading("Diagnostics");
                egui::ScrollArea::both().show(ui, |ui| {
                    if let Some(parse) = &self.model.parse {
                        ui.label(RichText::new("Parse").strong());
                        render_diagnostics(ui, &parse.diagnostics);
                    }
                    if let Some(compile) = &self.model.compile {
                        ui.separator();
                        ui.label(RichText::new("Compile").strong());
                        render_diagnostics(ui, &compile.diagnostics);
                    }
                });
            });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("Scene Summary");
            if let Some((project_name, default_scene, scheduler, _)) = project_meta.as_ref() {
                ui.label(format!("Project: {}", project_name));
                ui.label(format!("Default scene: {}", default_scene));
                ui.label(format!("Active scene: {}", self.model.selected_scene));
                ui.label(format!("Scheduler: {}", scheduler.as_str()));
            } else {
                ui.label("Project parse failed.");
            }
            let runtime_ready = true;
            let runtime_text = if self.model.android_editor_lite {
                "Scene viewer runtime: Android mini-editor mode (preview + diagnostics)"
            } else {
                "Scene viewer runtime: ready"
            };
            let runtime_color = if runtime_ready {
                Color32::from_rgb(178, 232, 178)
            } else {
                Color32::from_rgb(255, 180, 140)
            };
            ui.label(RichText::new(runtime_text).color(runtime_color));
            ui.separator();

            ui.label(RichText::new("Scene View").strong());
            ui.label("Runtime-linked preview lane (light mode keeps editor responsive).");
            let preview_height = if self.model.preview.light_mode {
                240.0
            } else {
                300.0
            };
            let (preview_rect, _) = ui.allocate_exact_size(
                egui::vec2(ui.available_width().max(120.0), preview_height),
                egui::Sense::drag(),
            );
            let preview_response = ui.interact(
                preview_rect,
                ui.id().with("scene_preview_touch"),
                egui::Sense::drag(),
            );
            if preview_response.dragged() {
                let delta = preview_response.drag_delta();
                self.model.preview.apply_touch_pan(
                    [delta.x, delta.y],
                    [preview_rect.width(), preview_rect.height()],
                );
            }
            let painter = ui.painter_at(preview_rect);
            draw_scene_preview(&painter, preview_rect, &self.model.preview);
            ui.separator();

            if !self.model.android_editor_lite {
                ui.label(RichText::new("Transform Tool").strong());
                ui.label("Pre-Beta move/rotate lane with world/local coordinate-space mapping.");
                egui::ComboBox::from_label("Coordinate Space")
                    .selected_text(self.model.transform_tool.coordinate_space.as_tlscript_str())
                    .show_ui(ui, |ui| {
                        ui.selectable_value(
                            &mut self.model.transform_tool.coordinate_space,
                            EditorCoordinateSpace::World,
                            "world",
                        );
                        ui.selectable_value(
                            &mut self.model.transform_tool.coordinate_space,
                            EditorCoordinateSpace::Local,
                            "local",
                        );
                    });
                ui.horizontal_wrapped(|ui| {
                    ui.label("Move Δ");
                    ui.add(
                        egui::DragValue::new(&mut self.model.transform_tool.move_delta[0])
                            .speed(0.05)
                            .prefix("x:"),
                    );
                    ui.add(
                        egui::DragValue::new(&mut self.model.transform_tool.move_delta[1])
                            .speed(0.05)
                            .prefix("y:"),
                    );
                    ui.add(
                        egui::DragValue::new(&mut self.model.transform_tool.move_delta[2])
                            .speed(0.05)
                            .prefix("z:"),
                    );
                });
                ui.horizontal_wrapped(|ui| {
                    ui.label("Rotate Δ");
                    ui.add(
                        egui::DragValue::new(&mut self.model.transform_tool.rotate_delta_deg[0])
                            .speed(0.5)
                            .prefix("yaw:"),
                    );
                    ui.add(
                        egui::DragValue::new(&mut self.model.transform_tool.rotate_delta_deg[1])
                            .speed(0.5)
                            .prefix("pitch:"),
                    );
                });
                ui.horizontal_wrapped(|ui| {
                    ui.label("Nudge");
                    ui.add(
                        egui::DragValue::new(&mut self.model.transform_tool.move_step)
                            .speed(0.01)
                            .range(0.01..=5.0)
                            .prefix("move:"),
                    );
                    ui.add(
                        egui::DragValue::new(&mut self.model.transform_tool.rotate_step_deg)
                            .speed(0.25)
                            .range(0.1..=90.0)
                            .prefix("rot:"),
                    );
                });
                ui.horizontal_wrapped(|ui| {
                    if icon_button(ui, "MX-", "-X").clicked() {
                        self.model.transform_tool.move_delta[0] -=
                            self.model.transform_tool.move_step;
                    }
                    if icon_button(ui, "MX+", "+X").clicked() {
                        self.model.transform_tool.move_delta[0] +=
                            self.model.transform_tool.move_step;
                    }
                    if icon_button(ui, "MY-", "-Y").clicked() {
                        self.model.transform_tool.move_delta[1] -=
                            self.model.transform_tool.move_step;
                    }
                    if icon_button(ui, "MY+", "+Y").clicked() {
                        self.model.transform_tool.move_delta[1] +=
                            self.model.transform_tool.move_step;
                    }
                    if icon_button(ui, "R-", "-Yaw").clicked() {
                        self.model.transform_tool.rotate_delta_deg[0] -=
                            self.model.transform_tool.rotate_step_deg;
                    }
                    if icon_button(ui, "R+", "+Yaw").clicked() {
                        self.model.transform_tool.rotate_delta_deg[0] +=
                            self.model.transform_tool.rotate_step_deg;
                    }
                });
                ui.horizontal_wrapped(|ui| {
                    if icon_button(ui, "AP", "Apply Preview").clicked() {
                        self.model.apply_transform_to_preview();
                    }
                    if icon_button(ui, "TS", "Append .tlscript").clicked() {
                        self.model.append_transform_to_selected_script();
                    }
                    if icon_button(ui, "RST", "Reset Tool").clicked() {
                        self.model.transform_tool = TransformToolState::default();
                        self.model.transform_status = "transform tool reset".to_string();
                    }
                });
                ui.separator();
            }

            if let Some(compile) = &self.model.compile {
                if let Some(bundle) = &compile.bundle {
                    ui.label(format!("Scripts: {}", bundle.scripts.len()));
                    ui.label(format!("Sprites: {}", bundle.sprite_count()));
                    ui.label(format!("Script paths: {}", bundle.script_paths.len()));
                    ui.label(format!("Sprite paths: {}", bundle.sprite_paths.len()));
                    if let Some(path) = &bundle.selected_joint_path {
                        ui.label(format!("Joint: {}", path.display()));
                    } else {
                        ui.label("Joint: none");
                    }

                    ui.separator();
                    ui.label(RichText::new("Resolved Assets").strong());
                    egui::ScrollArea::both().max_height(320.0).show(ui, |ui| {
                        ui.label(
                            RichText::new("Scripts")
                                .monospace()
                                .color(Color32::from_rgb(165, 214, 255)),
                        );
                        for path in &bundle.script_paths {
                            ui.label(RichText::new(path.display().to_string()).monospace());
                        }
                        ui.separator();
                        ui.label(
                            RichText::new("Sprites")
                                .monospace()
                                .color(Color32::from_rgb(189, 255, 189)),
                        );
                        for path in &bundle.sprite_paths {
                            ui.label(RichText::new(path.display().to_string()).monospace());
                        }
                    });
                } else {
                    ui.label("Scene is not compile-ready. Check diagnostics.");
                }
            } else {
                ui.label("Press 'Compile Scene' after loading project.");
            }

            if let Some(path) = self.model.selected_file.as_ref() {
                ui.separator();
                ui.label(RichText::new("Selected File").strong());
                ui.label(RichText::new(path.display().to_string()).monospace());
            }
        });

        if !self.model.android_editor_lite {
            egui::TopBottomPanel::bottom("editor_panel")
                .resizable(true)
                .default_height(230.0)
                .show(ctx, |ui| {
                    ui.horizontal_wrapped(|ui| {
                        ui.label(RichText::new("Mini .tlpfile Editor").strong());
                        ui.separator();
                        if icon_button(ui, "L", "Load").clicked() {
                            self.model.reload_source_from_disk();
                            self.model.reparse_from_buffer();
                        }
                        if icon_button(ui, "S", "Save").clicked() {
                            if let Err(err) = self.model.save_source_to_disk() {
                                self.model.editor_status = err;
                            }
                        }
                        if icon_button(ui, "P", "Parse").clicked() {
                            self.model.reparse_from_buffer();
                        }
                        if icon_button(ui, "SC", "Save+Compile").clicked() {
                            match self.model.save_source_to_disk() {
                                Ok(()) => self.model.compile_selected_scene(),
                                Err(err) => self.model.editor_status = err,
                            }
                        }
                    });

                    egui::ScrollArea::both()
                        .id_salt("tlpfile_editor_scroll")
                        .show(ui, |ui| {
                            let response = ui.add(
                                egui::TextEdit::multiline(&mut self.model.project_source)
                                    .code_editor()
                                    .desired_rows(14)
                                    .desired_width(f32::INFINITY),
                            );
                            if response.changed() {
                                self.model.editor_dirty = true;
                                self.model.editor_status = "buffer modified".to_string();
                            }
                        });
                });
        }
    }
}

struct GuiApp {
    cli: Cli,
    window_id: Option<WindowId>,
    runtime: Option<GuiRuntime>,
}

impl GuiApp {
    fn new(cli: Cli) -> Self {
        Self {
            cli,
            window_id: None,
            runtime: None,
        }
    }
}

impl ApplicationHandler for GuiApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.runtime.is_some() {
            return;
        }
        match GuiRuntime::new(event_loop, &self.cli) {
            Ok(runtime) => {
                self.window_id = Some(runtime.window.id());
                runtime.window.request_redraw();
                self.runtime = Some(runtime);
            }
            Err(err) => {
                eprintln!("failed to initialize project GUI: {err}");
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
        if Some(window_id) != self.window_id {
            return;
        }
        let Some(runtime) = self.runtime.as_mut() else {
            return;
        };

        runtime.handle_window_event(&event);

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => runtime.resize(size),
            WindowEvent::ScaleFactorChanged { .. } => {
                let size = runtime.window.inner_size();
                runtime.resize(size);
            }
            WindowEvent::RedrawRequested => {
                if let Err(err) = runtime.render() {
                    eprintln!("render error: {err}");
                    event_loop.exit();
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(runtime) = self.runtime.as_ref() {
            runtime.window.request_redraw();
        }
    }
}

/// Run the general-purpose project GUI based on `.tlpfile`.
pub fn run_from_env() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse_from_env()?;
    let app = GuiApp::new(cli);
    app_runner::run_app_desktop(app)
}

/// Run project GUI from Android lifecycle entrypoint.
#[cfg(target_os = "android")]
pub fn run_with_android_app(android_app: AndroidApp) -> Result<(), Box<dyn Error>> {
    let cli = Cli::default();
    let app = GuiApp::new(cli);
    app_runner::run_app_android(android_app, app)
}

fn upsert_project_scheduler(source: &str, scheduler: TlpfileGraphicsScheduler) -> String {
    let mut out = String::with_capacity(source.len() + 32);
    let mut in_project = false;
    let mut wrote_scheduler = false;
    let mut saw_project = false;

    for line in source.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            if in_project && !wrote_scheduler {
                out.push_str(&format!("scheduler = {}\n", scheduler.as_str()));
                wrote_scheduler = true;
            }
            let section = &trimmed[1..trimmed.len() - 1];
            in_project = section.eq_ignore_ascii_case("project");
            if in_project {
                saw_project = true;
            }
            out.push_str(line);
            out.push('\n');
            continue;
        }

        if in_project {
            let head = trimmed.split_once('=').map(|(key, _)| key.trim());
            if matches!(head, Some("scheduler")) {
                out.push_str(&format!("scheduler = {}\n", scheduler.as_str()));
                wrote_scheduler = true;
                continue;
            }
        }

        out.push_str(line);
        out.push('\n');
    }

    if saw_project {
        if in_project && !wrote_scheduler {
            out.push_str(&format!("scheduler = {}\n", scheduler.as_str()));
        }
        out
    } else {
        if !out.is_empty() && !out.ends_with('\n') {
            out.push('\n');
        }
        out.push_str("[project]\n");
        out.push_str(&format!("scheduler = {}\n", scheduler.as_str()));
        out
    }
}

fn append_scene_section_to_project_source(source: &mut String, scene_name: &str, joint_name: &str) {
    let section_header = format!("[scene.{scene_name}]");
    if source.contains(&section_header) {
        return;
    }
    if !source.ends_with('\n') {
        source.push('\n');
    }
    source.push('\n');
    source.push_str(&section_header);
    source.push('\n');
    source.push_str(&format!("tljoint = {joint_name}\n"));
    source.push_str(&format!("tljoint_scene = {scene_name}\n"));
}

fn ensure_ext(raw_name: &str, ext: &str) -> String {
    if raw_name.ends_with(ext) {
        raw_name.to_string()
    } else {
        format!("{raw_name}{ext}")
    }
}

fn sanitize_scene_name(raw: &str) -> String {
    raw.chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
                ch
            } else {
                '_'
            }
        })
        .collect::<String>()
        .trim_matches('_')
        .to_string()
}

fn write_if_missing(path: &Path, source: &str) -> Result<bool, String> {
    if path.exists() {
        return Ok(false);
    }
    fs::write(path, source)
        .map_err(|err| format!("failed to write '{}': {err}", path.display()))?;
    Ok(true)
}

fn default_script_template() -> String {
    String::from(
        "@export\n\
def showcase_tick(frame: int, live_balls: int, spawned_this_tick: int, key_f_down: bool, input_move_x: float, input_move_y: float, input_move_z: float, input_look_dx: float, input_look_dy: float, input_sprint_down: bool, input_look_active: bool, input_reset_camera: bool):\n\
    # Runtime-safe default template for new scene scripts.\n\
    set_spawn_per_tick(64)\n",
    )
}

fn default_sprite_template() -> String {
    String::from(
        "tlsprite_v1\n\
\n\
[hud.main]\n\
sprite_id = 1\n\
kind = hud\n\
texture_slot = 0\n\
layer = 100\n\
position = 0.0, 0.0, 0.0\n\
size = 0.2, 0.08\n\
rotation_rad = 0.0\n\
color = 1.0, 1.0, 1.0, 1.0\n",
    )
}

fn classify_project_file_kind(path: &Path, is_dir: bool) -> ProjectFileKind {
    if is_dir {
        return ProjectFileKind::Directory;
    }
    match path.extension().and_then(|ext| ext.to_str()) {
        Some("tlpfile") => ProjectFileKind::Project,
        Some("tljoint") => ProjectFileKind::Joint,
        Some("tlscript") => ProjectFileKind::Script,
        Some("tlsprite") => ProjectFileKind::Sprite,
        _ => ProjectFileKind::Other,
    }
}

fn collect_project_files(
    root: &Path,
    dir: &Path,
    depth: usize,
    out: &mut Vec<ProjectFileEntry>,
) -> std::io::Result<()> {
    if depth > PROJECT_SCAN_MAX_DEPTH {
        return Ok(());
    }
    let mut entries = fs::read_dir(dir)?.collect::<Result<Vec<_>, _>>()?;
    entries.sort_by_key(|entry| entry.path());
    for entry in entries {
        let path = entry.path();
        let file_type = entry.file_type()?;
        let relative = path.strip_prefix(root).map_or_else(
            |_| path.display().to_string(),
            |rel| rel.display().to_string(),
        );
        let kind = classify_project_file_kind(&path, file_type.is_dir());
        out.push(ProjectFileEntry {
            absolute_path: path.clone(),
            relative_path: relative,
            kind,
            depth,
        });
        if file_type.is_dir() {
            collect_project_files(root, &path, depth + 1, out)?;
        }
    }
    Ok(())
}

fn tlapp_binary_candidate() -> Option<PathBuf> {
    let current = std::env::current_exe().ok()?;
    let mut candidate = current.with_file_name("tlapp");
    if cfg!(windows) {
        candidate.set_extension("exe");
    }
    Some(candidate)
}

fn workspace_root_hint() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map_or_else(|| PathBuf::from("."), Path::to_path_buf)
}

fn apply_lavender_theme(ctx: &egui::Context) {
    let mut visuals = egui::Visuals::dark();
    visuals.window_fill = Color32::from_rgb(25, 21, 38);
    visuals.panel_fill = Color32::from_rgb(31, 26, 47);
    visuals.faint_bg_color = Color32::from_rgb(40, 34, 58);
    visuals.widgets.active.bg_fill = Color32::from_rgb(91, 78, 144);
    visuals.widgets.hovered.bg_fill = Color32::from_rgb(73, 63, 113);
    visuals.widgets.inactive.bg_fill = Color32::from_rgb(49, 42, 74);
    visuals.selection.bg_fill = Color32::from_rgb(109, 92, 172);
    ctx.set_visuals(visuals);
}

fn playback_label(state: PlaybackState) -> &'static str {
    match state {
        PlaybackState::Stopped => "stopped",
        PlaybackState::Running => "running",
        PlaybackState::Paused => "paused",
    }
}

fn icon_button(ui: &mut egui::Ui, icon: &str, label: &str) -> egui::Response {
    ui.add(
        egui::Button::new(
            RichText::new(format!("[{icon}] {label}"))
                .monospace()
                .color(Color32::from_rgb(236, 230, 255)),
        )
        .fill(Color32::from_rgb(62, 52, 98)),
    )
}

fn draw_scene_preview(painter: &egui::Painter, rect: egui::Rect, preview: &ScenePreviewState) {
    painter.rect_filled(rect, 8.0, Color32::from_rgb(30, 27, 44));
    painter.rect_stroke(
        rect,
        8.0,
        egui::Stroke::new(1.0, Color32::from_rgb(86, 72, 130)),
        egui::StrokeKind::Outside,
    );

    let sim_rect = rect.shrink2(egui::vec2(10.0, 10.0));
    painter.rect_filled(sim_rect, 6.0, Color32::from_rgb(18, 16, 27));
    painter.rect_stroke(
        sim_rect,
        6.0,
        egui::Stroke::new(1.0, Color32::from_rgb(72, 64, 110)),
        egui::StrokeKind::Outside,
    );

    for ball in &preview.balls {
        let x = sim_rect.left() + ball.pos[0] * sim_rect.width();
        let y = sim_rect.top() + ball.pos[1] * sim_rect.height();
        let radius = ball.radius * sim_rect.width().min(sim_rect.height());
        painter.circle_filled(egui::pos2(x, y), radius.max(2.0), ball.color);
    }

    let anchor = egui::pos2(
        sim_rect.left() + preview.tool_anchor[0] * sim_rect.width(),
        sim_rect.top() + preview.tool_anchor[1] * sim_rect.height(),
    );
    let half_w = sim_rect.width() * 0.06;
    let half_h = sim_rect.height() * 0.035;
    let (s, c) = preview.tool_rotation_rad.sin_cos();
    let rotate = |x: f32, y: f32| egui::vec2(x * c - y * s, x * s + y * c);
    let corners = [
        anchor + rotate(-half_w, -half_h),
        anchor + rotate(half_w, -half_h),
        anchor + rotate(half_w, half_h),
        anchor + rotate(-half_w, half_h),
    ];
    painter.add(egui::Shape::convex_polygon(
        corners.to_vec(),
        Color32::from_rgba_unmultiplied(174, 220, 255, 80),
        egui::Stroke::new(1.5, Color32::from_rgb(174, 220, 255)),
    ));
    let axis_tip = anchor + rotate(half_w * 1.15, 0.0);
    painter.line_segment(
        [anchor, axis_tip],
        egui::Stroke::new(2.0, Color32::from_rgb(255, 205, 120)),
    );
    painter.circle_filled(anchor, 2.5, Color32::from_rgb(255, 205, 120));

    let tag = if preview.light_mode {
        "LIGHT SCENE VIEW"
    } else {
        "SCENE VIEW"
    };
    painter.text(
        sim_rect.left_top() + egui::vec2(8.0, 8.0),
        egui::Align2::LEFT_TOP,
        format!("{tag} | {}", playback_label(preview.playback)),
        egui::FontId::monospace(12.0),
        Color32::from_rgb(185, 173, 232),
    );
    painter.text(
        sim_rect.left_bottom() + egui::vec2(8.0, -8.0),
        egui::Align2::LEFT_BOTTOM,
        format!(
            "Tool pos=({:.2},{:.2}) rot={:.1}deg",
            preview.tool_anchor[0],
            preview.tool_anchor[1],
            preview.tool_rotation_rad.to_degrees()
        ),
        egui::FontId::monospace(11.0),
        Color32::from_rgb(177, 214, 255),
    );
}

fn render_diagnostics(ui: &mut egui::Ui, diagnostics: &[TlpfileDiagnostic]) {
    if diagnostics.is_empty() {
        ui.label(RichText::new("No diagnostics").color(Color32::from_rgb(164, 220, 164)));
        return;
    }
    for diagnostic in diagnostics {
        let color = match diagnostic.level {
            TlpfileDiagnosticLevel::Error => Color32::from_rgb(255, 130, 130),
            TlpfileDiagnosticLevel::Warning => Color32::from_rgb(255, 210, 130),
        };
        ui.label(
            RichText::new(format!(
                "{:?} line {}: {}",
                diagnostic.level, diagnostic.line, diagnostic.message
            ))
            .color(color),
        );
    }
}

fn count_diagnostics<F>(diagnostics: &[TlpfileDiagnostic], predicate: F) -> usize
where
    F: Fn(TlpfileDiagnosticLevel) -> bool,
{
    diagnostics.iter().filter(|d| predicate(d.level)).count()
}
