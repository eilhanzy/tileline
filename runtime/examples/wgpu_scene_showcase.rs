use std::error::Error;
use std::sync::Arc;
use std::time::Instant;

use paradoxpe::{PhysicsWorld, PhysicsWorldConfig};
use runtime::{
    BounceTankSceneConfig, BounceTankSceneController, DrawPathCompiler, RenderSyncMode,
    TelemetryHudComposer, TelemetryHudSample, TickRatePolicy, TlspriteHotReloadEvent,
    TlspriteProgramCache, TlspriteWatchReloader, WgpuSceneRenderer,
};
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalSize};
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowAttributes, WindowId};

fn main() -> Result<(), Box<dyn Error>> {
    let event_loop = EventLoop::new()?;
    let mut app = ShowcaseApp::default();
    event_loop.run_app(&mut app)?;
    Ok(())
}

#[derive(Default)]
struct ShowcaseApp {
    runtime: Option<ShowcaseRuntime>,
    exit_requested: bool,
}

impl ApplicationHandler for ShowcaseApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.runtime.is_some() {
            return;
        }
        match ShowcaseRuntime::new(event_loop) {
            Ok(rt) => {
                self.runtime = Some(rt);
            }
            Err(err) => {
                eprintln!("Failed to start runtime wgpu showcase: {err}");
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
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: Key::Named(NamedKey::Escape),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                self.exit_requested = true;
                event_loop.exit();
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

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self.exit_requested {
            event_loop.exit();
            return;
        }
        if let Some(runtime) = self.runtime.as_ref() {
            runtime.window.request_redraw();
        }
    }
}

struct ShowcaseRuntime {
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
    sprite_loader: TlspriteWatchReloader,
    sprite_cache: TlspriteProgramCache,
    frame_started_at: Instant,
    fps_window_started_at: Instant,
    frames_in_window: u32,
    fps_estimate: f32,
}

impl ShowcaseRuntime {
    fn new(event_loop: &ActiveEventLoop) -> Result<Self, Box<dyn Error>> {
        let window = Arc::new(
            event_loop.create_window(
                WindowAttributes::default()
                    .with_title("Tileline Runtime Scene Showcase")
                    .with_inner_size(LogicalSize::new(1280.0, 720.0)),
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
                label: Some("runtime-wgpu-scene-showcase-device"),
                ..Default::default()
            }))?;

        let mut config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .ok_or("surface is not compatible with selected adapter")?;
        config.present_mode = wgpu::PresentMode::AutoNoVsync;
        surface.configure(&device, &config);

        let tick_policy = TickRatePolicy {
            ticks_per_render_frame: 2.0,
            ..TickRatePolicy::default()
        };
        let fixed_dt = tick_policy
            .resolve_fixed_dt_seconds(RenderSyncMode::Vsync { display_hz: 60.0 }, Some(60.0));
        let world = PhysicsWorld::new(PhysicsWorldConfig {
            fixed_dt,
            ..PhysicsWorldConfig::default()
        });
        let scene = BounceTankSceneController::new(BounceTankSceneConfig {
            target_ball_count: 8_000,
            spawn_per_tick: 280,
            ..BounceTankSceneConfig::default()
        });

        let renderer =
            WgpuSceneRenderer::new(&device, &queue, config.format, size.width, size.height);
        let draw_compiler = DrawPathCompiler::new();
        let hud = TelemetryHudComposer::new(Default::default());

        let sprite_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("examples/assets/bounce_hud.tlsprite");
        let mut sprite_loader = TlspriteWatchReloader::new(&sprite_path);
        let mut sprite_cache = TlspriteProgramCache::new();
        if let Some(warn) = sprite_loader.init_warning() {
            eprintln!("[tlsprite watch] {warn}");
        }
        eprintln!("[tlsprite watch] backend={:?}", sprite_loader.backend());
        let event = sprite_loader.reload_into_cache(&mut sprite_cache);
        print_tlsprite_event("[tlsprite boot]", event);
        let mut scene = scene;
        if let Some(program) = sprite_cache.program_for_path(sprite_loader.path()).cloned() {
            scene.set_sprite_program(program);
        }

        let now = Instant::now();
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
            sprite_loader,
            sprite_cache,
            frame_started_at: now,
            fps_window_started_at: now,
            frames_in_window: 0,
            fps_estimate: 60.0,
        })
    }

    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }
        self.size = new_size;
        self.config.width = new_size.width;
        self.config.height = new_size.height;
        self.surface.configure(&self.device, &self.config);
        self.renderer
            .resize(&self.queue, new_size.width.max(1), new_size.height.max(1));
    }

    fn render_frame(&mut self) -> Result<(), Box<dyn Error>> {
        let now = Instant::now();
        let dt = (now - self.frame_started_at).as_secs_f32().max(1.0 / 240.0);
        self.frame_started_at = now;
        self.frames_in_window = self.frames_in_window.saturating_add(1);
        let fps_window = (now - self.fps_window_started_at).as_secs_f32();
        if fps_window >= 0.5 {
            self.fps_estimate = self.frames_in_window as f32 / fps_window.max(1e-6);
            self.frames_in_window = 0;
            self.fps_window_started_at = now;
        }

        let event = self.sprite_loader.reload_into_cache(&mut self.sprite_cache);
        match &event {
            TlspriteHotReloadEvent::Applied { .. } => {
                print_tlsprite_event("[tlsprite reload]", event);
                if let Some(program) = self
                    .sprite_cache
                    .program_for_path(self.sprite_loader.path())
                    .cloned()
                {
                    self.scene.set_sprite_program(program);
                }
            }
            TlspriteHotReloadEvent::Unchanged => {}
            _ => print_tlsprite_event("[tlsprite reload]", event),
        }

        let tick = self.scene.physics_tick(&mut self.world);
        let substeps = self.world.step(dt);
        let mut frame = self
            .scene
            .build_frame_instances(&self.world, Some(self.world.interpolation_alpha()));
        let _hud = self.hud.append_to_sprites(
            TelemetryHudSample {
                fps: self.fps_estimate,
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
                label: Some("runtime-wgpu-scene-showcase-encoder"),
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
        Ok(())
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
