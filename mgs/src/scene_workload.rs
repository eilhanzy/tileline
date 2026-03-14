//! Scene-density to MGS bridge hint estimation.
//!
//! This module provides a runtime-friendly path to map scene payload density into
//! `MpsWorkloadHint`, so MGS planning can be driven from engine state (not benchmark-only logic).

use crate::bridge::{MgsBridgePlan, MpsWorkloadHint};
use crate::MgsBridge;

/// Lightweight mobile scene snapshot used for MGS planning hints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MobileSceneSnapshot {
    /// Number of opaque 3D instances.
    pub opaque_instances: usize,
    /// Number of transparent 3D instances.
    pub transparent_instances: usize,
    /// Number of sprites/overlay elements.
    pub sprite_instances: usize,
    /// Number of dynamic physics bodies.
    pub dynamic_body_count: usize,
    /// Estimated contact count for the frame.
    pub estimated_contacts: usize,
    /// Render width in pixels.
    pub viewport_width: u32,
    /// Render height in pixels.
    pub viewport_height: u32,
    /// Target latency budget in milliseconds (0 means default fallback).
    pub target_latency_ms: u32,
}

/// Tuning constants for mobile scene->hint conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MobileSceneTuning {
    /// Base transfer bytes per visible instance.
    pub transfer_bytes_per_instance: u32,
    /// Additional transfer bytes per dynamic body.
    pub transfer_bytes_per_dynamic_body: u32,
    /// Additional transfer bytes per contact estimate.
    pub transfer_bytes_per_contact: u32,
    /// Minimum transfer size in KB.
    pub min_transfer_kb: u32,
    /// Maximum transfer size in KB.
    pub max_transfer_kb: u32,
    /// Minimum object count passed to MGS planner.
    pub min_object_count: u32,
    /// Fallback latency budget (ms) when snapshot budget is zero.
    pub default_latency_budget_ms: u32,
}

impl Default for MobileSceneTuning {
    fn default() -> Self {
        Self {
            transfer_bytes_per_instance: 320,
            transfer_bytes_per_dynamic_body: 480,
            transfer_bytes_per_contact: 192,
            min_transfer_kb: 64,
            max_transfer_kb: 128 * 1024,
            min_object_count: 16,
            default_latency_budget_ms: 16,
        }
    }
}

/// Convert a mobile scene snapshot into an MGS bridge hint.
pub fn estimate_mps_workload_hint(
    snapshot: MobileSceneSnapshot,
    tuning: MobileSceneTuning,
) -> MpsWorkloadHint {
    let width = snapshot.viewport_width.max(1);
    let height = snapshot.viewport_height.max(1);
    let pixel_count = width as u64 * height as u64;

    let visible_instances = snapshot
        .opaque_instances
        .saturating_add(snapshot.transparent_instances)
        .saturating_add(snapshot.sprite_instances);
    let object_count = visible_instances
        .saturating_add(snapshot.dynamic_body_count / 2)
        .max(tuning.min_object_count as usize)
        .min(u32::MAX as usize) as u32;

    // Transfer estimate is intentionally conservative for mobile: scene payload + body/contact
    // pressure + render target pressure term.
    let transfer_bytes = (visible_instances as u64)
        .saturating_mul(tuning.transfer_bytes_per_instance as u64)
        .saturating_add(
            (snapshot.dynamic_body_count as u64)
                .saturating_mul(tuning.transfer_bytes_per_dynamic_body as u64),
        )
        .saturating_add(
            (snapshot.estimated_contacts as u64)
                .saturating_mul(tuning.transfer_bytes_per_contact as u64),
        )
        .saturating_add((pixel_count / 16).clamp(64 * 1024, 8 * 1024 * 1024));

    let transfer_kb = ((transfer_bytes + 1023) / 1024)
        .clamp(tuning.min_transfer_kb as u64, tuning.max_transfer_kb as u64)
        as u32;
    let latency_budget_ms = if snapshot.target_latency_ms == 0 {
        tuning.default_latency_budget_ms.max(1)
    } else {
        snapshot.target_latency_ms.max(1)
    };

    MpsWorkloadHint {
        transfer_size_kb: transfer_kb,
        object_count,
        target_width: width,
        target_height: height,
        latency_budget_ms,
    }
}

/// Estimate and immediately translate a scene snapshot through an existing MGS bridge.
pub fn plan_scene_with_bridge(
    bridge: &MgsBridge,
    snapshot: MobileSceneSnapshot,
    tuning: MobileSceneTuning,
) -> MgsBridgePlan {
    let hint = estimate_mps_workload_hint(snapshot, tuning);
    bridge.translate(hint)
}

#[cfg(test)]
mod tests {
    use crate::MobileGpuProfile;

    use super::*;

    fn sample_snapshot() -> MobileSceneSnapshot {
        MobileSceneSnapshot {
            opaque_instances: 2_048,
            transparent_instances: 128,
            sprite_instances: 32,
            dynamic_body_count: 1_000,
            estimated_contacts: 2_000,
            viewport_width: 1280,
            viewport_height: 720,
            target_latency_ms: 0,
        }
    }

    #[test]
    fn hint_scales_with_density() {
        let tuning = MobileSceneTuning::default();
        let low = estimate_mps_workload_hint(sample_snapshot(), tuning);
        let mut high_snapshot = sample_snapshot();
        high_snapshot.dynamic_body_count *= 2;
        high_snapshot.estimated_contacts *= 2;
        high_snapshot.opaque_instances *= 2;
        let high = estimate_mps_workload_hint(high_snapshot, tuning);

        assert!(high.transfer_size_kb > low.transfer_size_kb);
        assert!(high.object_count > low.object_count);
    }

    #[test]
    fn scene_can_be_planned_with_bridge() {
        let profile = MobileGpuProfile::detect("Mali-G78");
        let bridge = MgsBridge::new(profile);
        let plan = plan_scene_with_bridge(&bridge, sample_snapshot(), MobileSceneTuning::default());
        assert!(!plan.tile_plan.assignments.is_empty());
    }
}
