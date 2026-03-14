//! Runtime bridge between scene payloads and MGS workload hints.

use mgs::{estimate_mps_workload_hint, MobileSceneSnapshot, MobileSceneTuning, MpsWorkloadHint};

use crate::scene::SceneFrameInstances;

/// Runtime policy for MGS mobile scene-hint synthesis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MobileSceneWorkloadBridgeConfig {
    /// Estimated contact count multiplier for dynamic bodies.
    pub estimated_contacts_per_dynamic_body: f32,
    /// Target frame latency budget in milliseconds.
    pub target_latency_ms: u32,
    /// Tuning constants forwarded to MGS hint estimation.
    pub tuning: MobileSceneTuning,
}

impl Default for MobileSceneWorkloadBridgeConfig {
    fn default() -> Self {
        Self {
            estimated_contacts_per_dynamic_body: 1.2,
            target_latency_ms: 16,
            tuning: MobileSceneTuning::default(),
        }
    }
}

/// Build an MGS mobile scene snapshot from runtime scene payloads.
pub fn build_mobile_scene_snapshot(
    frame: &SceneFrameInstances,
    dynamic_body_count: usize,
    viewport_width: u32,
    viewport_height: u32,
    config: MobileSceneWorkloadBridgeConfig,
) -> MobileSceneSnapshot {
    let estimated_contacts = ((dynamic_body_count as f32)
        * config.estimated_contacts_per_dynamic_body.max(0.0))
    .round() as usize;
    MobileSceneSnapshot {
        opaque_instances: frame.opaque_3d.len(),
        transparent_instances: frame.transparent_3d.len(),
        sprite_instances: frame.sprites.len(),
        dynamic_body_count,
        estimated_contacts,
        viewport_width,
        viewport_height,
        target_latency_ms: config.target_latency_ms,
    }
}

/// Estimate an MGS bridge hint directly from runtime scene payloads.
pub fn estimate_mobile_workload_hint(
    frame: &SceneFrameInstances,
    dynamic_body_count: usize,
    viewport_width: u32,
    viewport_height: u32,
    config: MobileSceneWorkloadBridgeConfig,
) -> MpsWorkloadHint {
    let snapshot = build_mobile_scene_snapshot(
        frame,
        dynamic_body_count,
        viewport_width,
        viewport_height,
        config,
    );
    estimate_mps_workload_hint(snapshot, config.tuning)
}

#[cfg(test)]
mod tests {
    use crate::SceneFrameInstances;

    use super::*;

    #[test]
    fn mobile_hint_uses_default_latency_when_not_overridden() {
        let frame = SceneFrameInstances::default();
        let hint = estimate_mobile_workload_hint(
            &frame,
            256,
            1280,
            720,
            MobileSceneWorkloadBridgeConfig::default(),
        );
        assert!(hint.transfer_size_kb > 0);
        assert_eq!(hint.latency_budget_ms, 16);
    }
}
