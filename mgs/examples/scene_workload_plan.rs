use mgs::{
    plan_scene_with_bridge, MgsBridge, MobileGpuProfile, MobileSceneSnapshot, MobileSceneTuning,
};

fn main() {
    let profile = MobileGpuProfile::detect("Mali-G78 MC24");
    let bridge = MgsBridge::new(profile.clone());
    let snapshot = MobileSceneSnapshot {
        opaque_instances: 3_000,
        transparent_instances: 240,
        sprite_instances: 48,
        dynamic_body_count: 1_500,
        estimated_contacts: 2_600,
        viewport_width: 1280,
        viewport_height: 720,
        target_latency_ms: 16,
    };

    let plan = plan_scene_with_bridge(&bridge, snapshot, MobileSceneTuning::default());
    let total_draws: u32 = plan
        .tile_plan
        .assignments
        .iter()
        .map(|a| a.draw_calls)
        .sum();
    println!(
        "[MGS Scene Plan] family={:?} arch={:?} fallback={:?} memory_pressure={} tiles={} draws={}",
        profile.family,
        profile.architecture,
        plan.resolved_fallback,
        plan.memory_pressure,
        plan.tile_plan.assignments.len(),
        total_draws
    );
}
