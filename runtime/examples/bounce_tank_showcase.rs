use paradoxpe::{PhysicsWorld, PhysicsWorldConfig};
use runtime::{
    compile_tlscript_showcase, estimate_scene_workload_requests, BounceTankSceneConfig,
    BounceTankSceneController, DrawPathCompiler, RenderSyncMode, SceneWorkloadBridgeConfig,
    TelemetryHudComposer, TelemetryHudSample, TickRatePolicy, TlscriptShowcaseConfig,
    TlscriptShowcaseFrameInput, TlspriteHotReloadEvent, TlspriteProgramCache,
    TlspriteWatchReloader,
};

fn main() {
    let render_sync_mode = RenderSyncMode::Vsync { display_hz: 60.0 };
    let measured_render_fps = Some(60.0);
    let tick_policy = TickRatePolicy {
        ticks_per_render_frame: 3.0,
        ..TickRatePolicy::default()
    };
    let fixed_dt = tick_policy.resolve_fixed_dt_seconds(render_sync_mode, measured_render_fps);
    let render_dt = 1.0 / 60.0;

    let script_source = include_str!("assets/bounce_showcase.tlscript");
    let script_compile =
        compile_tlscript_showcase(script_source, TlscriptShowcaseConfig::default());
    for warning in &script_compile.warnings {
        println!("[tlscript warning] {warning}");
    }
    if !script_compile.errors.is_empty() {
        for error in &script_compile.errors {
            eprintln!("[tlscript error] {error}");
        }
        std::process::exit(1);
    }
    let script = script_compile
        .program
        .expect("showcase script must compile without errors");

    let mut world = PhysicsWorld::new(PhysicsWorldConfig {
        fixed_dt,
        ..PhysicsWorldConfig::default()
    });
    let mut scene = BounceTankSceneController::new(BounceTankSceneConfig {
        target_ball_count: 8_000,
        spawn_per_tick: 280,
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
    let workload_cfg = SceneWorkloadBridgeConfig {
        target_frame_budget_ms: 16.67,
        ..SceneWorkloadBridgeConfig::default()
    };
    let mut draw_compiler = DrawPathCompiler::new();
    let telemetry_hud = TelemetryHudComposer::new(Default::default());
    let mut last_spawned = 0usize;

    let total_frames = 60 * 6;
    println!(
        "[Showcase] render_dt={:.4}ms | physics_fixed_dt={:.4}ms | target_frames={} | tlscript_entry={} parallel_contract={}",
        render_dt * 1_000.0,
        fixed_dt * 1_000.0,
        total_frames,
        script.entry_function_name(),
        script.has_parallel_contract()
    );

    for frame_index in 0..total_frames {
        let event = sprite_loader.reload_into_cache(&mut sprite_cache);
        match &event {
            TlspriteHotReloadEvent::Applied { .. } => {
                print_tlsprite_event("[tlsprite reload]", event);
                if let Some(program) = sprite_cache.program_for_path(sprite_loader.path()).cloned()
                {
                    scene.set_sprite_program(program);
                }
            }
            TlspriteHotReloadEvent::Unchanged => {}
            _ => print_tlsprite_event("[tlsprite reload]", event),
        }
        let frame_eval = script.evaluate_frame(TlscriptShowcaseFrameInput {
            frame_index: frame_index as u64,
            live_balls: scene.live_ball_count(),
            spawned_this_tick: last_spawned,
            key_f_down: false,
        });
        let _patch_metrics = scene.apply_runtime_patch(&mut world, frame_eval.patch);
        let tick = scene.physics_tick(&mut world);
        last_spawned = tick.spawned_this_tick;
        let substeps = world.step(render_dt);
        let _ = scene.reconcile_after_step(&mut world);
        let mut frame = scene.build_frame_instances(&world, Some(world.interpolation_alpha()));
        let estimate =
            estimate_scene_workload_requests(&frame, tick.live_balls, 1280, 720, workload_cfg);
        let fps = if render_dt > f32::EPSILON {
            1.0 / render_dt
        } else {
            0.0
        };
        let frame_time_ms = render_dt * 1_000.0;
        let hud = telemetry_hud.append_to_sprites(
            TelemetryHudSample {
                fps,
                frame_time_ms,
                physics_substeps: substeps,
                live_balls: tick.live_balls,
                draw_calls: frame.opaque_3d.len()
                    + frame.transparent_3d.len()
                    + frame.sprites.len(),
            },
            &mut frame.sprites,
        );
        let draw = draw_compiler.compile(&frame);

        if frame_index % 30 == 0 || frame_index + 1 == total_frames {
            let cache_stats = sprite_cache.stats();
            println!(
                "frame={:03} spawn={} live={} substeps={} opaque={} transparent={} sprites={} draw_calls={} hud_sprites={} cache.programs={} cache.bindings={} dispatch_mode={:?} chunk={:?} gms.sampled={} gms.object={} gms.physics={} gms.ui={} gms.postfx={}",
                frame_index,
                tick.spawned_this_tick,
                tick.live_balls,
                substeps,
                draw.stats.opaque_instances,
                draw.stats.transparent_instances,
                draw.stats.sprite_instances,
                draw.stats.total_draw_calls,
                hud.appended_sprites,
                cache_stats.unique_programs,
                cache_stats.path_bindings,
                frame_eval.dispatch_decision.as_ref().map(|d| d.mode),
                frame_eval.dispatch_decision.as_ref().and_then(|d| d.chunk_size),
                estimate.multi_gpu.sampled_processing_jobs,
                estimate.multi_gpu.object_updates,
                estimate.multi_gpu.physics_jobs,
                estimate.multi_gpu.ui_jobs,
                estimate.multi_gpu.post_fx_jobs
            );
            for warning in frame_eval.warnings.iter().take(2) {
                println!("  script warning: {warning}");
            }
        }
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
