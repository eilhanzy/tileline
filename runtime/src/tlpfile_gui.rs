//! General-purpose project GUI for `.tlpfile`-driven workflows.
//!
//! This is the first desktop shell that unifies `.tlscript`, `.tlsprite`, and `.tljoint`
//! composition under one project manifest.

use std::env;
use std::error::Error;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use egui::{Color32, RichText};
use egui_wgpu::wgpu;
use egui_wgpu::{Renderer, RendererOptions, ScreenDescriptor};
use egui_winit::State as EguiWinitState;
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalSize};
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

use crate::tlpfile::{
    compile_tlpfile_scene_from_path, load_tlpfile, parse_tlpfile, TlpfileDiagnostic,
    TlpfileDiagnosticLevel, TlpfileParseOutcome, TlpfileProject, TlpfileSceneCompileOutcome,
};
use crate::TlscriptShowcaseConfig;

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
}

impl ScenePreviewState {
    fn new() -> Self {
        let mut state = Self {
            playback: PlaybackState::Stopped,
            light_mode: true,
            balls: Vec::new(),
            last_tick: Instant::now(),
            accumulator: 0.0,
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
}

#[derive(Debug)]
struct UiModel {
    project_path: PathBuf,
    selected_scene: String,
    parse: Option<TlpfileParseOutcome>,
    compile: Option<TlpfileSceneCompileOutcome>,
    status: String,
    project_source: String,
    editor_dirty: bool,
    editor_status: String,
    preview: ScenePreviewState,
}

impl UiModel {
    fn new(project_path: PathBuf, scene_override: Option<String>) -> Self {
        let mut model = Self {
            project_path,
            selected_scene: scene_override.unwrap_or_default(),
            parse: None,
            compile: None,
            status: String::new(),
            project_source: String::new(),
            editor_dirty: false,
            editor_status: String::new(),
            preview: ScenePreviewState::new(),
        };
        model.reload_parse();
        model.compile_selected_scene();
        model
    }

    fn reload_parse(&mut self) {
        self.reload_source_from_disk();
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
                "scene '{}' compiled | scripts={} sprites={} warnings={warnings}",
                bundle.scene_name,
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
        Ok(())
    }

    fn reparse_from_buffer(&mut self) {
        let outcome = parse_tlpfile(&self.project_source);
        if let Some(project) = outcome.project.as_ref() {
            if self.selected_scene.is_empty() || project.scene(&self.selected_scene).is_none() {
                self.selected_scene = project.default_scene.clone();
            }
        }
        self.parse = Some(outcome);
        self.compile = None;
        self.status = "reparsed in-memory .tlpfile buffer".to_string();
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
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("tlproject-gui-device"),
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
        });

        egui::SidePanel::left("scene_panel")
            .resizable(true)
            .default_width(240.0)
            .show(ctx, |ui| {
                ui.heading("Scenes");
                if let Some(project) = self.model.project() {
                    let scene_names = project
                        .scenes
                        .iter()
                        .map(|scene| scene.scene.clone())
                        .collect::<Vec<_>>();
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        for scene_name in scene_names {
                            let selected = self.model.selected_scene == scene_name;
                            if ui.selectable_label(selected, &scene_name).clicked() {
                                self.model.selected_scene = scene_name;
                                self.model.compile_selected_scene();
                            }
                        }
                    });
                    ui.separator();
                    ui.label(format!(
                        "Playback: {}",
                        playback_label(self.model.preview.playback)
                    ));
                    ui.label(format!("Preview balls: {}", self.model.preview.balls.len()));
                } else {
                    ui.label("No parsed project");
                }
            });

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
            if let Some(project) = self.model.project() {
                ui.label(format!("Project: {}", project.name));
                ui.label(format!("Default scene: {}", project.default_scene));
                ui.label(format!("Active scene: {}", self.model.selected_scene));
            } else {
                ui.label("Project parse failed.");
            }
            ui.separator();

            ui.label(RichText::new("Scene View").strong());
            ui.label("GIMP-style lightweight preview lane for rapid iteration.");
            let preview_height = if self.model.preview.light_mode {
                240.0
            } else {
                300.0
            };
            let (preview_rect, _) = ui.allocate_exact_size(
                egui::vec2(ui.available_width().max(120.0), preview_height),
                egui::Sense::hover(),
            );
            let painter = ui.painter_at(preview_rect);
            draw_scene_preview(&painter, preview_rect, &self.model.preview);
            ui.separator();

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
        });

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
    let event_loop = EventLoop::new()?;
    let mut app = GuiApp::new(cli);
    event_loop.run_app(&mut app)?;
    Ok(())
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
