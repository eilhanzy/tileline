//! Runtime bridge between scene payloads and GMS workload estimators.
//!
//! This keeps scene->GPU workload synthesis in `runtime/src`, so demos and main engine loops can
//! use the same mapping logic instead of re-implementing formulas in benchmark binaries.

use gms::{
    estimate_scene_workload, SceneWorkloadEstimate, SceneWorkloadSnapshot, SceneWorkloadTuning,
};

use crate::scene::SceneFrameInstances;

/// Runtime policy for synthesizing scene-level workload snapshots.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SceneWorkloadBridgeConfig {
    /// Estimated contact pairs per dynamic body when an exact contact count is unavailable.
    pub estimated_contacts_per_dynamic_body: f32,
    /// Target frame budget fed into multi-GPU planning.
    pub target_frame_budget_ms: f64,
    /// Estimator tuning constants used by GMS.
    pub tuning: SceneWorkloadTuning,
}

impl Default for SceneWorkloadBridgeConfig {
    fn default() -> Self {
        Self {
            estimated_contacts_per_dynamic_body: 1.6,
            target_frame_budget_ms: 16.67,
            tuning: SceneWorkloadTuning::default(),
        }
    }
}

/// Build a GMS scene snapshot from runtime scene payloads.
pub fn build_scene_workload_snapshot(
    frame: &SceneFrameInstances,
    dynamic_body_count: usize,
    viewport_width: u32,
    viewport_height: u32,
    config: SceneWorkloadBridgeConfig,
) -> SceneWorkloadSnapshot {
    let shadow_casters = frame
        .opaque_3d
        .iter()
        .chain(frame.transparent_3d.iter())
        .filter(|instance| instance.casts_shadow)
        .count();
    let shadow_receivers = frame
        .opaque_3d
        .iter()
        .chain(frame.transparent_3d.iter())
        .filter(|instance| instance.receives_shadow)
        .count();

    let estimated_contacts = ((dynamic_body_count as f32)
        * config.estimated_contacts_per_dynamic_body.max(0.0))
    .round() as usize;

    SceneWorkloadSnapshot {
        opaque_instances: frame.opaque_3d.len(),
        transparent_instances: frame.transparent_3d.len(),
        sprite_instances: frame.sprites.len(),
        shadow_casters,
        shadow_receivers,
        dynamic_body_count,
        estimated_contacts,
        viewport_width,
        viewport_height,
        target_frame_budget_ms: config.target_frame_budget_ms,
    }
}

/// Estimate GMS workload requests directly from runtime scene payloads.
pub fn estimate_scene_workload_requests(
    frame: &SceneFrameInstances,
    dynamic_body_count: usize,
    viewport_width: u32,
    viewport_height: u32,
    config: SceneWorkloadBridgeConfig,
) -> SceneWorkloadEstimate {
    let snapshot = build_scene_workload_snapshot(
        frame,
        dynamic_body_count,
        viewport_width,
        viewport_height,
        config,
    );
    estimate_scene_workload(snapshot, config.tuning)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        SceneFrameInstances, SceneInstance3d, SceneMaterial, ScenePrimitive3d, SceneTransform3d,
    };

    #[test]
    fn scene_workload_snapshot_reflects_shadow_flags() {
        let mut frame = SceneFrameInstances::default();
        frame.opaque_3d.push(SceneInstance3d {
            instance_id: 1,
            primitive: ScenePrimitive3d::Sphere,
            transform: SceneTransform3d::default(),
            material: SceneMaterial::default(),
            casts_shadow: true,
            receives_shadow: true,
        });
        frame.transparent_3d.push(SceneInstance3d {
            instance_id: 2,
            primitive: ScenePrimitive3d::Box,
            transform: SceneTransform3d::default(),
            material: SceneMaterial::default(),
            casts_shadow: false,
            receives_shadow: true,
        });

        let snapshot = build_scene_workload_snapshot(
            &frame,
            300,
            1920,
            1080,
            SceneWorkloadBridgeConfig::default(),
        );
        assert_eq!(snapshot.shadow_casters, 1);
        assert_eq!(snapshot.shadow_receivers, 2);
        assert_eq!(snapshot.dynamic_body_count, 300);
    }

    #[test]
    fn workload_requests_scale_with_dynamic_body_count() {
        let frame = SceneFrameInstances::default();
        let cfg = SceneWorkloadBridgeConfig::default();
        let low = estimate_scene_workload_requests(&frame, 64, 1280, 720, cfg);
        let high = estimate_scene_workload_requests(&frame, 2_048, 1280, 720, cfg);
        assert!(high.multi_gpu.physics_jobs > low.multi_gpu.physics_jobs);
        assert!(high.single_gpu.physics_jobs > low.single_gpu.physics_jobs);
    }
}
