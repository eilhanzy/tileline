use paradoxpe::{PhysicsWorld, PhysicsWorldConfig};
use runtime::{
    compile_tlscript_showcase, estimate_scene_workload_requests, BounceTankSceneConfig,
    BounceTankSceneController, RenderSyncMode, SceneWorkloadBridgeConfig, TickRatePolicy,
    TlscriptShowcaseConfig, TlscriptShowcaseFrameInput,
};

fn main() {
    let script = concat!(
        "@export\n",
        "@parallel(domain=\"bodies\", read=\"transform,aabb\", write=\"velocity\", chunk=128, schedule=\"performance\")\n",
        "@deterministic\n",
        "def showcase_tick(frame: int, live_balls: int, spawned_this_tick: int):\n",
        "    let burst: int = 320\n",
        "    if frame < 45:\n",
        "        burst = 420\n",
        "    if live_balls > 3000:\n",
        "        burst = 96\n",
        "    set_spawn_per_tick(burst)\n",
        "    if spawned_this_tick == 0 && live_balls > 1000:\n",
        "        set_linear_damping(0.014)\n",
        "    else:\n",
        "        set_linear_damping(0.008)\n",
        "    if frame % 120 == 0:\n",
        "        set_ball_restitution(0.95)\n",
        "        set_wall_restitution(0.95)\n",
    );

    let compile = compile_tlscript_showcase(script, TlscriptShowcaseConfig::default());
    for warning in &compile.warnings {
        eprintln!("[tlscript warning] {warning}");
    }
    if !compile.errors.is_empty() {
        for err in &compile.errors {
            eprintln!("[tlscript error] {err}");
        }
        std::process::exit(1);
    }
    let program = compile.program.expect("program must exist without errors");

    let tick_policy = TickRatePolicy {
        ticks_per_render_frame: 2.0,
        ..TickRatePolicy::default()
    };
    let fixed_dt = tick_policy
        .resolve_fixed_dt_seconds(RenderSyncMode::Vsync { display_hz: 60.0 }, Some(60.0));
    let render_dt = 1.0 / 60.0;
    let mut world = PhysicsWorld::new(PhysicsWorldConfig {
        fixed_dt,
        ..PhysicsWorldConfig::default()
    });
    let mut scene = BounceTankSceneController::new(BounceTankSceneConfig {
        target_ball_count: 7_500,
        spawn_per_tick: 280,
        ..BounceTankSceneConfig::default()
    });
    let workload_cfg = SceneWorkloadBridgeConfig::default();
    let mut last_spawned = 0usize;

    println!(
        "[tlscript demo] entry={} parallel_contract={} schedule_hint={:?}",
        program.entry_function_name(),
        program.has_parallel_contract(),
        program.parallel_schedule_hint()
    );

    for frame_index in 0..180u64 {
        let frame_eval = program.evaluate_frame(TlscriptShowcaseFrameInput {
            frame_index,
            live_balls: scene.live_ball_count(),
            spawned_this_tick: last_spawned,
        });
        let patch_metrics = scene.apply_runtime_patch(&mut world, frame_eval.patch);
        let tick = scene.physics_tick(&mut world);
        last_spawned = tick.spawned_this_tick;
        let substeps = world.step(render_dt);
        let frame = scene.build_frame_instances(&world, Some(world.interpolation_alpha()));
        let estimate =
            estimate_scene_workload_requests(&frame, tick.live_balls, 1280, 720, workload_cfg);

        if frame_index % 30 == 0 || frame_index == 179 {
            println!(
                "frame={:03} spawn={} live={} substeps={} patch_updated={} retuned={} dispatch_mode={:?} chunk={:?} sampled={} physics={}",
                frame_index,
                tick.spawned_this_tick,
                tick.live_balls,
                substeps,
                patch_metrics.config_updated,
                patch_metrics.dynamic_bodies_retuned,
                frame_eval.dispatch_decision.as_ref().map(|d| d.mode),
                frame_eval.dispatch_decision.as_ref().and_then(|d| d.chunk_size),
                estimate.multi_gpu.sampled_processing_jobs,
                estimate.multi_gpu.physics_jobs
            );
            for warning in frame_eval.warnings.iter().take(2) {
                eprintln!("  script warning: {warning}");
            }
        }
    }
}
