use mgs::MgsBridge;
use paradoxpe::{PhysicsWorld, PhysicsWorldConfig};
use runtime::{
    choose_scheduler_path, estimate_mobile_workload_hint, estimate_scene_workload_requests,
    submit_scene_estimate_to_bridge, BounceTankSceneConfig, BounceTankSceneController,
    DrawPathCompiler, FrameLoopRuntime, FrameLoopRuntimeConfig, GraphicsSchedulerPath,
    MobileSceneWorkloadBridgeConfig, RenderSyncMode, SceneDispatchBridgeConfig,
    SceneWorkloadBridgeConfig, TelemetryHudComposer, TelemetryHudSample, TickRatePolicy,
    TlspriteHotReloadEvent, TlspriteProgramCache, TlspriteWatchReloader,
};
use tl_core::MpsGmsBridgeConfig;
use wgpu::{AdapterInfo, Backend, DeviceType};

fn main() {
    let adapter_info = discover_primary_adapter().unwrap_or_else(default_adapter_info);
    let decision = choose_scheduler_path(&adapter_info);

    println!(
        "[Runtime Auto Scheduler] adapter='{}' backend={:?} type={:?} => path={:?} | reason={}",
        decision.adapter_name,
        decision.backend,
        decision.device_type,
        decision.path,
        decision.reason
    );

    let render_sync_mode = RenderSyncMode::Vsync { display_hz: 60.0 };
    let tick_policy = TickRatePolicy {
        ticks_per_render_frame: 2.0,
        ..TickRatePolicy::default()
    };
    let fixed_dt = tick_policy.resolve_fixed_dt_seconds(render_sync_mode, Some(60.0));
    let render_dt = 1.0 / 60.0;
    let total_frames = 120u64;

    let mut world = PhysicsWorld::new(PhysicsWorldConfig {
        fixed_dt,
        ..PhysicsWorldConfig::default()
    });
    let mut scene = BounceTankSceneController::new(BounceTankSceneConfig {
        target_ball_count: 4_000,
        spawn_per_tick: 220,
        ..BounceTankSceneConfig::default()
    });
    let sprite_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples/assets/bounce_hud.tlsprite");
    let mut sprite_loader = TlspriteWatchReloader::new(&sprite_path);
    let mut sprite_cache = TlspriteProgramCache::new();
    if let Some(warn) = sprite_loader.init_warning() {
        println!("[tlsprite watch] {warn}");
    }
    println!("[tlsprite watch] backend={:?}", sprite_loader.backend());
    print_tlsprite_event(
        "[tlsprite boot]",
        sprite_loader.reload_into_cache(&mut sprite_cache),
    );
    if let Some(program) = sprite_cache.program_for_path(sprite_loader.path()).cloned() {
        scene.set_sprite_program(program);
    }

    match decision.path {
        GraphicsSchedulerPath::Gms => run_gms_path(
            &mut world,
            &mut scene,
            total_frames,
            render_dt,
            SceneWorkloadBridgeConfig::default(),
            &mut sprite_loader,
            &mut sprite_cache,
        ),
        GraphicsSchedulerPath::Mgs => run_mgs_path(
            &mut world,
            &mut scene,
            total_frames,
            render_dt,
            MobileSceneWorkloadBridgeConfig::default(),
            decision.mobile_profile.name.clone(),
            &mut sprite_loader,
            &mut sprite_cache,
        ),
    }
}

fn run_gms_path(
    world: &mut PhysicsWorld,
    scene: &mut BounceTankSceneController,
    total_frames: u64,
    render_dt: f32,
    workload_cfg: SceneWorkloadBridgeConfig,
    sprite_loader: &mut TlspriteWatchReloader,
    sprite_cache: &mut TlspriteProgramCache,
) {
    let mut frame_loop = FrameLoopRuntime::new(
        FrameLoopRuntimeConfig::default(),
        MpsGmsBridgeConfig::default(),
    );
    let mut draw_compiler = DrawPathCompiler::new();
    let telemetry_hud = TelemetryHudComposer::new(Default::default());
    let dispatch_cfg = SceneDispatchBridgeConfig::default();
    let mut submitted_tasks = 0u64;
    let mut published_plans = 0u64;

    for frame_id in 1..=total_frames {
        maybe_reload_tlsprite(scene, sprite_loader, sprite_cache);
        let tick = scene.physics_tick(world);
        let substeps = world.step(render_dt);
        let mut frame = scene.build_frame_instances(world, Some(world.interpolation_alpha()));
        let fps = if render_dt > f32::EPSILON {
            1.0 / render_dt
        } else {
            0.0
        };
        let _hud = telemetry_hud.append_to_sprites(
            TelemetryHudSample {
                fps,
                frame_time_ms: render_dt * 1_000.0,
                physics_substeps: substeps,
                live_balls: tick.live_balls,
                draw_calls: frame.opaque_3d.len()
                    + frame.transparent_3d.len()
                    + frame.sprites.len(),
            },
            &mut frame.sprites,
        );
        let draw = draw_compiler.compile(&frame);
        let estimate =
            estimate_scene_workload_requests(&frame, tick.live_balls, 1280, 720, workload_cfg);
        let dispatch =
            submit_scene_estimate_to_bridge(&mut frame_loop, frame_id, &estimate, dispatch_cfg);
        submitted_tasks = submitted_tasks.saturating_add(dispatch.total_submitted_tasks as u64);

        let _ = frame_loop.tick();
        if frame_loop.pop_next_frame_plan().is_some() {
            published_plans = published_plans.saturating_add(1);
        }

        if frame_id % 30 == 0 || frame_id == total_frames {
            println!(
                "[GMS DrawPath] frame={} draw_calls={} opaque_batches={} transparent_batches={} sprites={}",
                frame_id,
                draw.stats.total_draw_calls,
                draw.stats.opaque_batches,
                draw.stats.transparent_batches,
                draw.stats.sprite_instances
            );
        }
    }

    let _ = frame_loop.wait_for_cpu_idle(std::time::Duration::from_secs(2));
    for _ in 0..12 {
        let _ = frame_loop.tick();
        while frame_loop.pop_next_frame_plan().is_some() {
            published_plans = published_plans.saturating_add(1);
        }
    }

    let metrics = frame_loop.metrics();
    println!(
        "[GMS Path] submitted_tasks={} published_plans={} bridge_published={} queued={}",
        submitted_tasks,
        published_plans,
        metrics.bridge_tick_published_frames,
        metrics.queued_frame_plans
    );
}

fn run_mgs_path(
    world: &mut PhysicsWorld,
    scene: &mut BounceTankSceneController,
    total_frames: u64,
    render_dt: f32,
    mobile_cfg: MobileSceneWorkloadBridgeConfig,
    adapter_name: String,
    sprite_loader: &mut TlspriteWatchReloader,
    sprite_cache: &mut TlspriteProgramCache,
) {
    let bridge = MgsBridge::new(mgs::MobileGpuProfile::detect(&adapter_name));
    let mut draw_compiler = DrawPathCompiler::new();
    let telemetry_hud = TelemetryHudComposer::new(Default::default());
    let mut memory_pressure_frames = 0u64;
    let mut total_tiles = 0u64;
    let mut total_draws = 0u64;

    for frame_id in 0..total_frames {
        maybe_reload_tlsprite(scene, sprite_loader, sprite_cache);
        let tick = scene.physics_tick(world);
        let substeps = world.step(render_dt);
        let mut frame = scene.build_frame_instances(world, Some(world.interpolation_alpha()));
        let fps = if render_dt > f32::EPSILON {
            1.0 / render_dt
        } else {
            0.0
        };
        let _hud = telemetry_hud.append_to_sprites(
            TelemetryHudSample {
                fps,
                frame_time_ms: render_dt * 1_000.0,
                physics_substeps: substeps,
                live_balls: tick.live_balls,
                draw_calls: frame.opaque_3d.len()
                    + frame.transparent_3d.len()
                    + frame.sprites.len(),
            },
            &mut frame.sprites,
        );
        let draw = draw_compiler.compile(&frame);
        let hint = estimate_mobile_workload_hint(&frame, tick.live_balls, 1280, 720, mobile_cfg);
        let plan = bridge.translate(hint);
        if plan.memory_pressure {
            memory_pressure_frames = memory_pressure_frames.saturating_add(1);
        }
        total_tiles = total_tiles.saturating_add(plan.tile_plan.assignments.len() as u64);
        total_draws = total_draws.saturating_add(
            plan.tile_plan
                .assignments
                .iter()
                .map(|a| a.draw_calls as u64)
                .sum::<u64>(),
        );

        if frame_id % 30 == 0 || frame_id + 1 == total_frames {
            println!(
                "frame={:03} hint.transfer_kb={} hint.objects={} fallback={:?} tiles={} mem_pressure={} draw_calls={} sprites={}",
                frame_id,
                hint.transfer_size_kb,
                hint.object_count,
                plan.resolved_fallback,
                plan.tile_plan.assignments.len(),
                plan.memory_pressure,
                draw.stats.total_draw_calls,
                draw.stats.sprite_instances
            );
        }
    }

    println!(
        "[MGS Path] frames={} pressure_frames={} avg_tiles_per_frame={:.2} avg_draws_per_frame={:.2}",
        total_frames,
        memory_pressure_frames,
        total_tiles as f64 / total_frames as f64,
        total_draws as f64 / total_frames as f64
    );
}

fn discover_primary_adapter() -> Option<AdapterInfo> {
    let instance = wgpu::Instance::default();
    let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));
    adapters
        .into_iter()
        .map(|adapter| adapter.get_info())
        .max_by_key(adapter_rank)
}

fn adapter_rank(info: &AdapterInfo) -> u32 {
    let device_score = match info.device_type {
        DeviceType::DiscreteGpu => 400,
        DeviceType::IntegratedGpu => 300,
        DeviceType::VirtualGpu => 200,
        DeviceType::Cpu => 100,
        DeviceType::Other => 50,
    };
    let backend_score = match info.backend {
        Backend::Vulkan => 40,
        Backend::Metal => 35,
        Backend::Dx12 => 30,
        Backend::Gl => 20,
        Backend::BrowserWebGpu => 10,
        Backend::Noop => 0,
    };
    device_score + backend_score
}

fn default_adapter_info() -> AdapterInfo {
    AdapterInfo {
        name: "Unknown Adapter".to_string(),
        vendor: 0,
        device: 0,
        device_type: DeviceType::Other,
        device_pci_bus_id: String::new(),
        driver: String::new(),
        driver_info: String::new(),
        backend: Backend::Noop,
        subgroup_min_size: 1,
        subgroup_max_size: 1,
        transient_saves_memory: false,
    }
}

fn maybe_reload_tlsprite(
    scene: &mut BounceTankSceneController,
    loader: &mut TlspriteWatchReloader,
    cache: &mut TlspriteProgramCache,
) {
    let event = loader.reload_into_cache(cache);
    match &event {
        TlspriteHotReloadEvent::Applied { .. } => {
            print_tlsprite_event("[tlsprite reload]", event);
            if let Some(program) = cache.program_for_path(loader.path()).cloned() {
                scene.set_sprite_program(program);
            }
        }
        TlspriteHotReloadEvent::Unchanged => {}
        _ => print_tlsprite_event("[tlsprite reload]", event),
    }
}

fn print_tlsprite_event(prefix: &str, event: TlspriteHotReloadEvent) {
    match event {
        TlspriteHotReloadEvent::Unchanged => {}
        TlspriteHotReloadEvent::Applied {
            sprite_count,
            warning_count,
        } => println!("{prefix} applied sprites={sprite_count} warnings={warning_count}"),
        TlspriteHotReloadEvent::Rejected {
            error_count,
            warning_count,
            kept_last_program,
        } => println!(
            "{prefix} rejected errors={error_count} warnings={warning_count} kept_last_program={kept_last_program}"
        ),
        TlspriteHotReloadEvent::SourceError { message } => {
            println!("{prefix} source_error: {message}")
        }
    }
}
