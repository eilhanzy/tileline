//! Scene/sprite workload estimation for runtime-facing GMS integration.
//!
//! This module converts renderer-agnostic scene density into GMS planner requests so engine code
//! can stay in `src/` crates instead of benchmark-only paths.

use crate::{MultiGpuWorkloadRequest, WorkloadRequest};

/// Runtime scene snapshot used to derive GMS workload requests.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct SceneWorkloadSnapshot {
    /// Number of opaque 3D instances (meshes/spheres/boxes/etc.).
    pub opaque_instances: usize,
    /// Number of transparent 3D instances.
    pub transparent_instances: usize,
    /// Number of sprite/overlay instances.
    pub sprite_instances: usize,
    /// Number of instances casting shadows.
    pub shadow_casters: usize,
    /// Number of instances receiving shadows.
    pub shadow_receivers: usize,
    /// Number of dynamic physics bodies participating in the scene.
    pub dynamic_body_count: usize,
    /// Estimated broadphase/narrowphase contacts for the frame.
    pub estimated_contacts: usize,
    /// Render viewport width.
    pub viewport_width: u32,
    /// Render viewport height.
    pub viewport_height: u32,
    /// Frame-time budget for multi-GPU planning.
    pub target_frame_budget_ms: f64,
}

/// Tuning constants for scene-to-workload conversion.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SceneWorkloadTuning {
    /// How many pixels are represented by one sampled-processing job baseline.
    pub pixels_per_sampled_job: u64,
    /// Additional sampled jobs per complexity point.
    pub sampled_jobs_per_complexity: f64,
    /// Object jobs per visual instance baseline.
    pub object_jobs_per_visual_instance: f64,
    /// Additional object jobs per dynamic body.
    pub object_jobs_per_dynamic_body: f64,
    /// Physics jobs per dynamic body.
    pub physics_jobs_per_dynamic_body: f64,
    /// Physics jobs per estimated contact.
    pub physics_jobs_per_contact: f64,
    /// AI/ML jobs per dynamic body.
    pub ai_ml_jobs_per_dynamic_body: f64,
    /// UI jobs per sprite.
    pub ui_jobs_per_sprite: f64,
    /// Post-FX jobs per transparent instance.
    pub post_fx_jobs_per_transparent: f64,
    /// Preferred base workgroup size before per-adapter scaling.
    pub base_workgroup_size: u32,
    /// Fallback frame budget when snapshot budget is not valid.
    pub fallback_frame_budget_ms: f64,
}

impl Default for SceneWorkloadTuning {
    fn default() -> Self {
        Self {
            pixels_per_sampled_job: 2_048,
            sampled_jobs_per_complexity: 4.5,
            object_jobs_per_visual_instance: 0.90,
            object_jobs_per_dynamic_body: 1.75,
            physics_jobs_per_dynamic_body: 2.10,
            physics_jobs_per_contact: 1.20,
            ai_ml_jobs_per_dynamic_body: 0.85,
            ui_jobs_per_sprite: 0.25,
            post_fx_jobs_per_transparent: 1.10,
            base_workgroup_size: 64,
            fallback_frame_budget_ms: 16.67,
        }
    }
}

/// Combined single-GPU and multi-GPU workload estimates.
#[derive(Debug, Clone, Copy)]
pub struct SceneWorkloadEstimate {
    /// Single-GPU request used by `GmsDispatcher`.
    pub single_gpu: WorkloadRequest,
    /// Multi-GPU request used by `MultiGpuDispatcher`.
    pub multi_gpu: MultiGpuWorkloadRequest,
    /// Aggregate scene complexity score used during estimation.
    pub complexity_score: f64,
    /// Estimated bytes touched per frame by scene payloads.
    pub estimated_frame_bytes: u64,
}

/// Estimate workload requests from a scene snapshot.
pub fn estimate_scene_workload(
    snapshot: SceneWorkloadSnapshot,
    tuning: SceneWorkloadTuning,
) -> SceneWorkloadEstimate {
    let width = snapshot.viewport_width.max(1) as u64;
    let height = snapshot.viewport_height.max(1) as u64;
    let pixels = width.saturating_mul(height);

    let opaque = snapshot.opaque_instances as f64;
    let transparent = snapshot.transparent_instances as f64;
    let sprites = snapshot.sprite_instances as f64;
    let dynamic_bodies = snapshot.dynamic_body_count as f64;
    let contacts = snapshot.estimated_contacts as f64;
    let shadows = snapshot.shadow_casters as f64 + (snapshot.shadow_receivers as f64 * 0.35);

    // Weighted complexity: transparent and contact-heavy lanes cost more due to blending/overdraw
    // and collision processing pressure.
    let complexity_score = opaque
        + (transparent * 1.45)
        + (sprites * 0.40)
        + (dynamic_bodies * 0.90)
        + (contacts * 1.20)
        + (shadows * 0.55);

    let sampled_processing_jobs = clamp_u32(
        (pixels as f64 / tuning.pixels_per_sampled_job.max(1) as f64)
            + complexity_score * tuning.sampled_jobs_per_complexity,
        64,
        32_768,
    );
    let object_updates = clamp_u32(
        ((opaque + transparent) * tuning.object_jobs_per_visual_instance)
            + dynamic_bodies * tuning.object_jobs_per_dynamic_body
            + shadows * 0.35,
        64,
        65_535,
    );
    let physics_jobs = clamp_u32(
        dynamic_bodies * tuning.physics_jobs_per_dynamic_body
            + contacts * tuning.physics_jobs_per_contact,
        32,
        65_535,
    );
    let ai_ml_jobs = clamp_u32(
        dynamic_bodies * tuning.ai_ml_jobs_per_dynamic_body + contacts * 0.18 + 8.0,
        8,
        32_768,
    );
    let ui_jobs = clamp_u32(sprites * tuning.ui_jobs_per_sprite + 8.0, 8, 8_192);
    let post_fx_jobs = clamp_u32(
        transparent * tuning.post_fx_jobs_per_transparent + (pixels as f64 / 10_240.0),
        16,
        16_384,
    );

    let estimated_frame_bytes = estimate_frame_bytes(snapshot, pixels);
    let sampled_bytes = round_up_u64(
        (estimated_frame_bytes / sampled_processing_jobs.max(1) as u64).clamp(1_024, 16_384),
        256,
    );
    let object_bytes = round_up_u64(
        (estimated_frame_bytes / object_updates.max(1) as u64).clamp(256, 4_096),
        256,
    );
    let physics_bytes = round_up_u64(
        (estimated_frame_bytes / physics_jobs.max(1) as u64).clamp(512, 8_192),
        256,
    );
    let ui_bytes = round_up_u64(
        (estimated_frame_bytes / ui_jobs.max(1) as u64).clamp(256, 2_048),
        256,
    );
    let ai_ml_bytes = round_up_u64(
        (estimated_frame_bytes / ai_ml_jobs.max(1) as u64).clamp(512, 12_288),
        256,
    );
    let post_fx_bytes = round_up_u64(
        (estimated_frame_bytes / post_fx_jobs.max(1) as u64).clamp(512, 6_144),
        256,
    );

    let base_workgroup_size = normalize_base_workgroup_size(tuning.base_workgroup_size);
    let target_frame_budget_ms = sanitize_budget(
        snapshot.target_frame_budget_ms,
        tuning.fallback_frame_budget_ms,
    );

    let single_gpu = WorkloadRequest {
        object_updates: object_updates
            .saturating_add(ui_jobs / 2)
            .saturating_add(ai_ml_jobs / 3),
        physics_jobs: physics_jobs
            .saturating_add(post_fx_jobs / 3)
            .saturating_add(ai_ml_jobs / 2),
        bytes_per_object: object_bytes,
        bytes_per_physics_job: physics_bytes,
        base_workgroup_size,
    };

    let multi_gpu = MultiGpuWorkloadRequest {
        sampled_processing_jobs,
        object_updates,
        physics_jobs,
        ai_ml_jobs,
        ui_jobs,
        post_fx_jobs,
        bytes_per_sampled_job: sampled_bytes,
        bytes_per_object: object_bytes,
        bytes_per_physics_job: physics_bytes,
        bytes_per_ai_ml_job: ai_ml_bytes,
        bytes_per_ui_job: ui_bytes,
        bytes_per_post_fx_job: post_fx_bytes,
        processed_texture_bytes_per_frame: round_up_u64(
            (pixels.saturating_mul(4)).max(estimated_frame_bytes / 4),
            256,
        ),
        base_workgroup_size,
        target_frame_budget_ms,
    };

    SceneWorkloadEstimate {
        single_gpu,
        multi_gpu,
        complexity_score,
        estimated_frame_bytes,
    }
}

fn estimate_frame_bytes(snapshot: SceneWorkloadSnapshot, pixels: u64) -> u64 {
    let base_color_bytes = pixels.saturating_mul(4);
    let depth_bytes = pixels.saturating_mul(4);
    let object_bytes = (snapshot.opaque_instances as u64)
        .saturating_add(snapshot.transparent_instances as u64)
        .saturating_mul(128);
    let sprite_bytes = (snapshot.sprite_instances as u64).saturating_mul(64);
    let physics_bytes = (snapshot.dynamic_body_count as u64)
        .saturating_mul(192)
        .saturating_add((snapshot.estimated_contacts as u64).saturating_mul(96));

    round_up_u64(
        base_color_bytes
            .saturating_add(depth_bytes)
            .saturating_add(object_bytes)
            .saturating_add(sprite_bytes)
            .saturating_add(physics_bytes)
            .clamp(256 * 1024, 256 * 1024 * 1024),
        256,
    )
}

fn sanitize_budget(budget_ms: f64, fallback_ms: f64) -> f64 {
    if budget_ms.is_finite() && budget_ms > 0.10 {
        budget_ms
    } else {
        fallback_ms.max(0.10)
    }
}

fn normalize_base_workgroup_size(raw: u32) -> u32 {
    let clamped = raw.clamp(32, 1024);
    let pow2 = clamped.next_power_of_two();
    if pow2 > 1024 {
        1024
    } else {
        pow2
    }
}

fn clamp_u32(value: f64, min_value: u32, max_value: u32) -> u32 {
    value.round().clamp(min_value as f64, max_value as f64) as u32
}

fn round_up_u64(value: u64, alignment: u64) -> u64 {
    if alignment <= 1 {
        return value;
    }
    let remainder = value % alignment;
    if remainder == 0 {
        value
    } else {
        value.saturating_add(alignment - remainder)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_snapshot() -> SceneWorkloadSnapshot {
        SceneWorkloadSnapshot {
            opaque_instances: 1_024,
            transparent_instances: 16,
            sprite_instances: 32,
            shadow_casters: 960,
            shadow_receivers: 1_040,
            dynamic_body_count: 900,
            estimated_contacts: 2_500,
            viewport_width: 1_280,
            viewport_height: 720,
            target_frame_budget_ms: 16.67,
        }
    }

    #[test]
    fn workload_grows_with_scene_density() {
        let tuning = SceneWorkloadTuning::default();
        let low = estimate_scene_workload(sample_snapshot(), tuning);

        let mut high_snapshot = sample_snapshot();
        high_snapshot.opaque_instances *= 2;
        high_snapshot.dynamic_body_count *= 2;
        high_snapshot.estimated_contacts *= 3;
        let high = estimate_scene_workload(high_snapshot, tuning);

        assert!(high.complexity_score > low.complexity_score);
        assert!(high.multi_gpu.sampled_processing_jobs > low.multi_gpu.sampled_processing_jobs);
        assert!(high.multi_gpu.physics_jobs > low.multi_gpu.physics_jobs);
        assert!(high.multi_gpu.ai_ml_jobs > low.multi_gpu.ai_ml_jobs);
        assert!(high.single_gpu.object_updates >= low.single_gpu.object_updates);
    }

    #[test]
    fn invalid_budget_falls_back_to_tuning_default() {
        let tuning = SceneWorkloadTuning::default();
        let mut snapshot = sample_snapshot();
        snapshot.target_frame_budget_ms = -1.0;
        let estimate = estimate_scene_workload(snapshot, tuning);
        assert!(
            (estimate.multi_gpu.target_frame_budget_ms - tuning.fallback_frame_budget_ms).abs()
                < 1e-6
        );
    }
}
