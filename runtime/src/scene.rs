//! Runtime scene + sprite data model and a pre-alpha bounce showcase controller.
//!
//! This module keeps scene orchestration in `runtime/src` so examples can stay thin:
//! - generic 3D instance payloads (sphere/box primitives + PBR-ish material params)
//! - sprite instance payloads for HUD/overlay composition
//! - `BounceTankSceneController` that builds a transparent 3D container and progressively spawns
//!   thousands of colored balls using ParadoxPE bodies/colliders
//! - render/present vs physics tick-rate policy helper

use std::collections::HashMap;
use std::f32::consts::TAU;

use nalgebra::Vector3;
use paradoxpe::{
    Aabb, BodyDesc, BodyHandle, BodyKind, ColliderDesc, ColliderMaterial, ColliderShape,
    PhysicsWorld,
};
use rayon::prelude::*;

use crate::tlsprite::{TlspriteFrameContext, TlspriteProgram};

const COLLISION_GROUP_BALL: u16 = 1 << 0;
const COLLISION_GROUP_WALL: u16 = 1 << 1;
const SCATTER_MIN_LIVE_BALLS_BASE: usize = 600;
const SCATTER_MIN_AFFECTED: usize = 12;

#[inline]
fn collision_filter_tag(group: u16, mask: u16) -> u32 {
    (group as u32) | ((mask as u32) << 16)
}

/// Lightweight primitive kind for runtime-side scene batching.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScenePrimitive3d {
    Sphere,
    Box,
    /// Custom FBX mesh bound into the renderer slot table.
    Mesh {
        slot: u8,
    },
}

/// Material shading model hint for runtime/renderer integration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShadingModel {
    LitPbr,
    Unlit,
}

/// Compact material payload for renderer-facing scene instances.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SceneMaterial {
    pub base_color_rgba: [f32; 4],
    pub roughness: f32,
    pub metallic: f32,
    pub emissive_rgb: [f32; 3],
    pub shading: ShadingModel,
}

impl Default for SceneMaterial {
    fn default() -> Self {
        Self {
            base_color_rgba: [0.7, 0.7, 0.7, 1.0],
            roughness: 0.6,
            metallic: 0.0,
            emissive_rgb: [0.0, 0.0, 0.0],
            shading: ShadingModel::LitPbr,
        }
    }
}

/// Compact transform payload for runtime/renderer integration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SceneTransform3d {
    pub translation: [f32; 3],
    pub rotation_xyzw: [f32; 4],
    pub scale: [f32; 3],
}

impl Default for SceneTransform3d {
    fn default() -> Self {
        Self {
            translation: [0.0, 0.0, 0.0],
            rotation_xyzw: [0.0, 0.0, 0.0, 1.0],
            scale: [1.0, 1.0, 1.0],
        }
    }
}

/// One 3D render instance emitted by runtime scene management.
#[derive(Debug, Clone, PartialEq)]
pub struct SceneInstance3d {
    pub instance_id: u64,
    pub primitive: ScenePrimitive3d,
    pub transform: SceneTransform3d,
    pub material: SceneMaterial,
    pub casts_shadow: bool,
    pub receives_shadow: bool,
}

/// One sprite instance (HUD/overlay/billboard path).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpriteKind {
    Generic,
    Hud,
    Camera,
    Terrain,
}

impl Default for SpriteKind {
    fn default() -> Self {
        Self::Generic
    }
}

/// One sprite instance (HUD/overlay/billboard path).
#[derive(Debug, Clone, PartialEq)]
pub struct SpriteInstance {
    pub sprite_id: u64,
    pub kind: SpriteKind,
    pub position: [f32; 3],
    pub size: [f32; 2],
    pub rotation_rad: f32,
    pub color_rgba: [f32; 4],
    pub texture_slot: u16,
    pub layer: i16,
}

/// Runtime scene batch for one render frame.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SceneFrameInstances {
    pub opaque_3d: Vec<SceneInstance3d>,
    pub transparent_3d: Vec<SceneInstance3d>,
    pub sprites: Vec<SpriteInstance>,
}

/// Render sync mode used for tick-rate policy decisions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RenderSyncMode {
    /// Render loop is synced to display refresh.
    Vsync { display_hz: f32 },
    /// Render loop is capped by an explicit FPS limit.
    FpsCap { fps: f32 },
    /// Render loop is uncapped (use measured FPS when available).
    Uncapped,
}

/// Runtime policy for decoupling render FPS and physics tick-rate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TickRatePolicy {
    pub min_tick_hz: f32,
    pub max_tick_hz: f32,
    pub ticks_per_render_frame: f32,
    pub default_tick_hz: f32,
}

impl Default for TickRatePolicy {
    fn default() -> Self {
        Self {
            min_tick_hz: 60.0,
            max_tick_hz: 360.0,
            ticks_per_render_frame: 3.0,
            default_tick_hz: 180.0,
        }
    }
}

impl TickRatePolicy {
    /// Resolve physics tick-rate from current render pacing mode and optional measured FPS.
    pub fn resolve_tick_hz(self, mode: RenderSyncMode, measured_render_fps: Option<f32>) -> f32 {
        let render_basis = match mode {
            RenderSyncMode::Vsync { display_hz } => display_hz.max(1.0),
            RenderSyncMode::FpsCap { fps } => fps.max(1.0),
            RenderSyncMode::Uncapped => measured_render_fps
                .filter(|fps| *fps > 1.0)
                .unwrap_or(self.default_tick_hz),
        };
        let target = render_basis * self.ticks_per_render_frame.max(0.25);
        quantize_tick_hz(target).clamp(self.min_tick_hz.max(1.0), self.max_tick_hz.max(1.0))
    }

    /// Resolve fixed-step delta seconds for `PhysicsWorldConfig::fixed_dt`.
    pub fn resolve_fixed_dt_seconds(
        self,
        mode: RenderSyncMode,
        measured_render_fps: Option<f32>,
    ) -> f32 {
        1.0 / self.resolve_tick_hz(mode, measured_render_fps).max(1.0)
    }
}

/// Configuration for the bounce showcase scene.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BounceTankSceneConfig {
    pub target_ball_count: usize,
    pub spawn_per_tick: usize,
    pub container_half_extents: [f32; 3],
    pub wall_thickness: f32,
    pub container_mesh_scale: [f32; 3],
    pub ball_radius_min: f32,
    pub ball_radius_max: f32,
    pub ball_restitution: f32,
    pub ball_friction: f32,
    pub wall_restitution: f32,
    pub wall_friction: f32,
    pub friction_transition_speed: f32,
    pub friction_static_boost: f32,
    pub friction_kinetic_scale: f32,
    pub restitution_velocity_threshold: f32,
    pub levitation_height: f32,
    pub levitation_strength: f32,
    pub levitation_damping: f32,
    pub levitation_max_vertical_speed: f32,
    pub levitation_reaction_strength: f32,
    pub levitation_reaction_radius: f32,
    pub levitation_reaction_damping: f32,
    pub levitation_lateral_strength: f32,
    pub levitation_lateral_damping: f32,
    pub levitation_lateral_max_horizontal_speed: f32,
    pub levitation_lateral_wall_push: f32,
    pub levitation_lateral_frequency: f32,
    pub linear_damping: f32,
    pub initial_speed_min: f32,
    pub initial_speed_max: f32,
    pub scatter_interval_ticks: u64,
    pub scatter_strength: f32,
    pub virtual_barrier_enabled: bool,
    pub seed: u64,
}

impl Default for BounceTankSceneConfig {
    fn default() -> Self {
        Self {
            target_ball_count: 12_000,
            spawn_per_tick: 120,
            container_half_extents: [24.0, 16.0, 24.0],
            wall_thickness: 0.40,
            container_mesh_scale: [1.0, 1.0, 1.0],
            ball_radius_min: 0.10,
            ball_radius_max: 0.24,
            ball_restitution: 0.74,
            ball_friction: 0.28,
            wall_restitution: 0.78,
            wall_friction: 0.20,
            friction_transition_speed: 1.2,
            friction_static_boost: 1.10,
            friction_kinetic_scale: 0.92,
            restitution_velocity_threshold: 0.35,
            levitation_height: 0.0,
            levitation_strength: 0.0,
            levitation_damping: 0.0,
            levitation_max_vertical_speed: 4.5,
            levitation_reaction_strength: 0.0,
            levitation_reaction_radius: 1.2,
            levitation_reaction_damping: 0.0,
            levitation_lateral_strength: 0.0,
            levitation_lateral_damping: 1.2,
            levitation_lateral_max_horizontal_speed: 10.0,
            levitation_lateral_wall_push: 0.0,
            levitation_lateral_frequency: 0.35,
            linear_damping: 0.012,
            initial_speed_min: 0.35,
            initial_speed_max: 1.25,
            scatter_interval_ticks: 420,
            scatter_strength: 0.16,
            virtual_barrier_enabled: true,
            seed: 0x5EED_C0DE_u64,
        }
    }
}

/// Per-physics-tick output from bounce scene orchestration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BounceTankTickMetrics {
    pub spawned_this_tick: usize,
    pub scattered_this_tick: usize,
    pub live_balls: usize,
    pub target_balls: usize,
    pub fully_spawned: bool,
}

/// Runtime-safe patch set for `.tlscript` or gameplay controllers.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct BounceTankRuntimePatch {
    pub target_ball_count: Option<usize>,
    pub spawn_per_tick: Option<usize>,
    pub container_half_extents: Option<[f32; 3]>,
    pub wall_thickness: Option<f32>,
    pub container_mesh_scale: Option<[f32; 3]>,
    pub linear_damping: Option<f32>,
    pub gravity: Option<[f32; 3]>,
    pub contact_guard: Option<f32>,
    pub ball_restitution: Option<f32>,
    pub ball_friction: Option<f32>,
    pub wall_restitution: Option<f32>,
    pub wall_friction: Option<f32>,
    pub friction_transition_speed: Option<f32>,
    pub friction_static_boost: Option<f32>,
    pub friction_kinetic_scale: Option<f32>,
    pub restitution_velocity_threshold: Option<f32>,
    pub levitation_height: Option<f32>,
    pub levitation_strength: Option<f32>,
    pub levitation_damping: Option<f32>,
    pub levitation_max_vertical_speed: Option<f32>,
    pub levitation_reaction_strength: Option<f32>,
    pub levitation_reaction_radius: Option<f32>,
    pub levitation_reaction_damping: Option<f32>,
    pub levitation_lateral_strength: Option<f32>,
    pub levitation_lateral_damping: Option<f32>,
    pub levitation_lateral_max_horizontal_speed: Option<f32>,
    pub levitation_lateral_wall_push: Option<f32>,
    pub levitation_lateral_frequency: Option<f32>,
    pub initial_speed_min: Option<f32>,
    pub initial_speed_max: Option<f32>,
    pub scatter_interval_ticks: Option<u64>,
    pub scatter_strength: Option<f32>,
    pub virtual_barrier_enabled: Option<bool>,
    pub speculative_sweep_enabled: Option<bool>,
    pub speculative_sweep_max_distance: Option<f32>,
    pub speculative_contacts_enabled: Option<bool>,
    pub speculative_contact_distance: Option<f32>,
    pub speculative_max_prediction_distance: Option<f32>,
    pub ball_mesh_slot: Option<u8>,
    pub container_mesh_slot: Option<u8>,
}

/// Result summary for one runtime patch application.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BounceTankPatchMetrics {
    pub config_updated: bool,
    pub dynamic_bodies_retuned: usize,
    pub collider_materials_retuned: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct BallVisual {
    body: BodyHandle,
    radius: f32,
    color: [f32; 4],
    roughness: f32,
    metallic: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct LevitationCellKey {
    x: i32,
    y: i32,
    z: i32,
}

/// Scene controller for a transparent 3D tank that progressively fills with bouncing balls.
///
/// Physics uses ParadoxPE dynamic sphere bodies and static AABB wall colliders.
pub struct BounceTankSceneController {
    config: BounceTankSceneConfig,
    rng_state: u64,
    walls: Vec<BodyHandle>,
    edge_barriers: Vec<BodyHandle>,
    balls: Vec<BallVisual>,
    sprite_program: Option<TlspriteProgram>,
    ball_mesh_slot: Option<u8>,
    container_mesh_slot: Option<u8>,
    last_scatter_tick: u64,
    scatter_face_cursor: u8,
    levitation_body_handles: Vec<BodyHandle>,
    levitation_positions: Vec<[f32; 3]>,
    levitation_velocities: Vec<[f32; 3]>,
    levitation_reaction_cells: HashMap<LevitationCellKey, Vec<usize>>,
    levitation_lateral_phase: f32,
}

impl BounceTankSceneController {
    pub fn new(config: BounceTankSceneConfig) -> Self {
        Self {
            rng_state: config.seed,
            config,
            walls: Vec::with_capacity(6),
            edge_barriers: Vec::with_capacity(12),
            balls: Vec::new(),
            sprite_program: None,
            ball_mesh_slot: None,
            container_mesh_slot: None,
            last_scatter_tick: 0,
            scatter_face_cursor: 0,
            levitation_body_handles: Vec::new(),
            levitation_positions: Vec::new(),
            levitation_velocities: Vec::new(),
            levitation_reaction_cells: HashMap::new(),
            levitation_lateral_phase: 0.0,
        }
    }

    pub fn config(&self) -> BounceTankSceneConfig {
        self.config
    }

    pub fn live_ball_count(&self) -> usize {
        self.balls.len()
    }

    pub fn wall_count(&self) -> usize {
        self.walls.len()
    }

    pub fn edge_barrier_count(&self) -> usize {
        self.edge_barriers.len()
    }

    fn virtual_barrier_enabled(&self) -> bool {
        self.config.virtual_barrier_enabled
    }

    fn set_virtual_barrier_enabled(&mut self, enabled: bool) {
        self.config.virtual_barrier_enabled = enabled;
    }

    /// Override the default HUD sprite emission with a compiled `.tlsprite` program.
    pub fn set_sprite_program(&mut self, program: TlspriteProgram) {
        self.sprite_program = Some(program);
    }

    /// Remove a custom `.tlsprite` program and fall back to the built-in progress sprite.
    pub fn clear_sprite_program(&mut self) {
        self.sprite_program = None;
    }

    pub fn has_sprite_program(&self) -> bool {
        self.sprite_program.is_some()
    }

    /// Apply a bounded runtime patch and propagate dynamic damping changes to existing bodies.
    pub fn apply_runtime_patch(
        &mut self,
        world: &mut PhysicsWorld,
        patch: BounceTankRuntimePatch,
    ) -> BounceTankPatchMetrics {
        let mut updated = false;
        let mut retuned = 0usize;
        let mut collider_materials_retuned = 0usize;
        let mut refresh_ball_material = false;
        let mut refresh_wall_material = false;
        let mut refresh_solver_friction = false;
        let mut refresh_solver_restitution = false;

        if let Some(target) = patch.target_ball_count {
            let clamped = target.clamp(1, 200_000);
            if self.config.target_ball_count != clamped {
                self.config.target_ball_count = clamped;
                updated = true;
            }
        }
        if let Some(spawn) = patch.spawn_per_tick {
            let clamped = spawn.clamp(1, 8_192);
            if self.config.spawn_per_tick != clamped {
                self.config.spawn_per_tick = clamped;
                updated = true;
            }
        }
        let mut rebuild_barriers = false;
        if let Some(ext) = patch.container_half_extents {
            let clamped = [
                ext[0].clamp(0.5, 256.0),
                ext[1].clamp(0.5, 256.0),
                ext[2].clamp(0.5, 256.0),
            ];
            if self.config.container_half_extents != clamped {
                self.config.container_half_extents = clamped;
                rebuild_barriers = true;
                updated = true;
            }
        }
        if let Some(thickness) = patch.wall_thickness {
            let clamped = thickness.clamp(0.01, 8.0);
            if (self.config.wall_thickness - clamped).abs() > f32::EPSILON {
                self.config.wall_thickness = clamped;
                rebuild_barriers = true;
                updated = true;
            }
        }
        if let Some(scale) = patch.container_mesh_scale {
            let clamped = [
                scale[0].clamp(0.05, 64.0),
                scale[1].clamp(0.05, 64.0),
                scale[2].clamp(0.05, 64.0),
            ];
            if self.config.container_mesh_scale != clamped {
                self.config.container_mesh_scale = clamped;
                updated = true;
            }
        }
        if let Some(damping) = patch.linear_damping {
            let clamped = damping.clamp(0.0, 0.95);
            if (self.config.linear_damping - clamped).abs() > f32::EPSILON {
                self.config.linear_damping = clamped;
                retuned = world.set_linear_damping_all_dynamic(clamped);
                updated = true;
            }
        }
        if let Some(gravity) = patch.gravity {
            let clamped = Vector3::new(
                gravity[0].clamp(-120.0, 120.0),
                gravity[1].clamp(-120.0, 120.0),
                gravity[2].clamp(-120.0, 120.0),
            );
            if (world.gravity() - clamped).norm_squared() > 1e-8 {
                world.set_gravity(clamped);
                updated = true;
            }
        }
        if let Some(contact_guard) = patch.contact_guard {
            if world.set_contact_guard(contact_guard.clamp(0.0, 1.0)) {
                updated = true;
            }
        }
        if let Some(restitution) = patch.ball_restitution {
            let clamped = restitution.clamp(0.0, 1.25);
            if (self.config.ball_restitution - clamped).abs() > f32::EPSILON {
                self.config.ball_restitution = clamped;
                updated = true;
                refresh_ball_material = true;
            }
        }
        if let Some(friction) = patch.ball_friction {
            let clamped = friction.clamp(0.0, 2.0);
            if (self.config.ball_friction - clamped).abs() > f32::EPSILON {
                self.config.ball_friction = clamped;
                updated = true;
                refresh_ball_material = true;
            }
        }
        if let Some(restitution) = patch.wall_restitution {
            let clamped = restitution.clamp(0.0, 1.25);
            if (self.config.wall_restitution - clamped).abs() > f32::EPSILON {
                self.config.wall_restitution = clamped;
                updated = true;
                refresh_wall_material = true;
            }
        }
        if let Some(friction) = patch.wall_friction {
            let clamped = friction.clamp(0.0, 2.0);
            if (self.config.wall_friction - clamped).abs() > f32::EPSILON {
                self.config.wall_friction = clamped;
                updated = true;
                refresh_wall_material = true;
            }
        }
        if let Some(transition) = patch.friction_transition_speed {
            let clamped = transition.clamp(0.0, 12.0);
            if (self.config.friction_transition_speed - clamped).abs() > f32::EPSILON {
                self.config.friction_transition_speed = clamped;
                updated = true;
                refresh_solver_friction = true;
            }
        }
        if let Some(static_boost) = patch.friction_static_boost {
            let clamped = static_boost.clamp(0.0, 4.0);
            if (self.config.friction_static_boost - clamped).abs() > f32::EPSILON {
                self.config.friction_static_boost = clamped;
                updated = true;
                refresh_solver_friction = true;
            }
        }
        if let Some(kinetic_scale) = patch.friction_kinetic_scale {
            let clamped = kinetic_scale.clamp(0.0, 4.0);
            if (self.config.friction_kinetic_scale - clamped).abs() > f32::EPSILON {
                self.config.friction_kinetic_scale = clamped;
                updated = true;
                refresh_solver_friction = true;
            }
        }
        if let Some(threshold) = patch.restitution_velocity_threshold {
            let clamped = threshold.clamp(0.0, 6.0);
            if (self.config.restitution_velocity_threshold - clamped).abs() > f32::EPSILON {
                self.config.restitution_velocity_threshold = clamped;
                updated = true;
                refresh_solver_restitution = true;
            }
        }
        if let Some(height) = patch.levitation_height {
            let hy = self.config.container_half_extents[1].max(0.5);
            let clamped = height.clamp(-hy * 0.95, hy * 0.95);
            if (self.config.levitation_height - clamped).abs() > f32::EPSILON {
                self.config.levitation_height = clamped;
                updated = true;
            }
        }
        if let Some(strength) = patch.levitation_strength {
            let clamped = strength.clamp(0.0, 200.0);
            if (self.config.levitation_strength - clamped).abs() > f32::EPSILON {
                self.config.levitation_strength = clamped;
                updated = true;
            }
        }
        if let Some(damping) = patch.levitation_damping {
            let clamped = damping.clamp(0.0, 80.0);
            if (self.config.levitation_damping - clamped).abs() > f32::EPSILON {
                self.config.levitation_damping = clamped;
                updated = true;
            }
        }
        if let Some(max_speed) = patch.levitation_max_vertical_speed {
            let clamped = max_speed.clamp(0.05, 80.0);
            if (self.config.levitation_max_vertical_speed - clamped).abs() > f32::EPSILON {
                self.config.levitation_max_vertical_speed = clamped;
                updated = true;
            }
        }
        if let Some(strength) = patch.levitation_reaction_strength {
            let clamped = strength.clamp(0.0, 80.0);
            if (self.config.levitation_reaction_strength - clamped).abs() > f32::EPSILON {
                self.config.levitation_reaction_strength = clamped;
                updated = true;
            }
        }
        if let Some(radius) = patch.levitation_reaction_radius {
            let clamped = radius.clamp(0.0, 8.0);
            if (self.config.levitation_reaction_radius - clamped).abs() > f32::EPSILON {
                self.config.levitation_reaction_radius = clamped;
                updated = true;
            }
        }
        if let Some(damping) = patch.levitation_reaction_damping {
            let clamped = damping.clamp(0.0, 40.0);
            if (self.config.levitation_reaction_damping - clamped).abs() > f32::EPSILON {
                self.config.levitation_reaction_damping = clamped;
                updated = true;
            }
        }
        if let Some(strength) = patch.levitation_lateral_strength {
            let clamped = strength.clamp(0.0, 120.0);
            if (self.config.levitation_lateral_strength - clamped).abs() > f32::EPSILON {
                self.config.levitation_lateral_strength = clamped;
                updated = true;
            }
        }
        if let Some(damping) = patch.levitation_lateral_damping {
            let clamped = damping.clamp(0.0, 80.0);
            if (self.config.levitation_lateral_damping - clamped).abs() > f32::EPSILON {
                self.config.levitation_lateral_damping = clamped;
                updated = true;
            }
        }
        if let Some(max_speed) = patch.levitation_lateral_max_horizontal_speed {
            let clamped = max_speed.clamp(0.05, 120.0);
            if (self.config.levitation_lateral_max_horizontal_speed - clamped).abs() > f32::EPSILON
            {
                self.config.levitation_lateral_max_horizontal_speed = clamped;
                updated = true;
            }
        }
        if let Some(push) = patch.levitation_lateral_wall_push {
            let clamped = push.clamp(0.0, 240.0);
            if (self.config.levitation_lateral_wall_push - clamped).abs() > f32::EPSILON {
                self.config.levitation_lateral_wall_push = clamped;
                updated = true;
            }
        }
        if let Some(freq) = patch.levitation_lateral_frequency {
            let clamped = freq.clamp(0.0, 20.0);
            if (self.config.levitation_lateral_frequency - clamped).abs() > f32::EPSILON {
                self.config.levitation_lateral_frequency = clamped;
                updated = true;
            }
        }
        if let Some(min_speed) = patch.initial_speed_min {
            let clamped = min_speed.clamp(0.0, 32.0);
            if (self.config.initial_speed_min - clamped).abs() > f32::EPSILON {
                self.config.initial_speed_min = clamped;
                updated = true;
            }
        }
        if let Some(max_speed) = patch.initial_speed_max {
            let clamped = max_speed.clamp(0.0, 64.0);
            if (self.config.initial_speed_max - clamped).abs() > f32::EPSILON {
                self.config.initial_speed_max = clamped;
                updated = true;
            }
        }
        if let Some(scatter_interval_ticks) = patch.scatter_interval_ticks {
            let clamped = scatter_interval_ticks.clamp(1, 5_000);
            if self.config.scatter_interval_ticks != clamped {
                self.config.scatter_interval_ticks = clamped;
                updated = true;
            }
        }
        if let Some(scatter_strength) = patch.scatter_strength {
            let clamped = scatter_strength.clamp(0.0, 1.0);
            if (self.config.scatter_strength - clamped).abs() > f32::EPSILON {
                self.config.scatter_strength = clamped;
                updated = true;
            }
        }
        if let Some(enabled) = patch.virtual_barrier_enabled {
            if self.virtual_barrier_enabled() != enabled {
                self.set_virtual_barrier_enabled(enabled);
                updated = true;
            }
        }
        if patch.speculative_sweep_enabled.is_some()
            || patch.speculative_sweep_max_distance.is_some()
        {
            let current = &world.config().broadphase;
            let enabled = patch
                .speculative_sweep_enabled
                .unwrap_or(current.speculative_sweep);
            let max_distance = patch
                .speculative_sweep_max_distance
                .unwrap_or(current.speculative_max_distance)
                .clamp(0.0, 8.0);
            if world.set_broadphase_speculative_sweep(enabled, max_distance) {
                updated = true;
            }
        }
        if patch.speculative_contacts_enabled.is_some()
            || patch.speculative_contact_distance.is_some()
            || patch.speculative_max_prediction_distance.is_some()
        {
            let current = &world.config().narrowphase;
            let enabled = patch
                .speculative_contacts_enabled
                .unwrap_or(current.speculative_contacts);
            let contact_distance = patch
                .speculative_contact_distance
                .unwrap_or(current.speculative_contact_distance)
                .clamp(0.0, 2.0);
            let max_prediction_distance = patch
                .speculative_max_prediction_distance
                .unwrap_or(current.speculative_max_prediction_distance)
                .clamp(0.0, 8.0)
                .max(contact_distance);
            if world.set_narrowphase_speculative_contacts(
                enabled,
                contact_distance,
                max_prediction_distance,
            ) {
                updated = true;
            }
        }
        if self.config.initial_speed_max < self.config.initial_speed_min {
            self.config.initial_speed_max = self.config.initial_speed_min;
            updated = true;
        }
        if let Some(slot) = patch.ball_mesh_slot {
            if self.ball_mesh_slot != Some(slot) {
                self.ball_mesh_slot = Some(slot);
                updated = true;
            }
        }
        if let Some(slot) = patch.container_mesh_slot {
            if self.container_mesh_slot != Some(slot) {
                self.container_mesh_slot = Some(slot);
                updated = true;
            }
        }
        if rebuild_barriers {
            for body in self.walls.drain(..) {
                let _ = world.destroy_body(body);
            }
            for body in self.edge_barriers.drain(..) {
                let _ = world.destroy_body(body);
            }
        }

        if refresh_ball_material {
            collider_materials_retuned = collider_materials_retuned.saturating_add(
                world.set_collider_material_for_collision_group(
                    COLLISION_GROUP_BALL,
                    ColliderMaterial {
                        restitution: self.config.ball_restitution.max(0.0),
                        friction: self.config.ball_friction.max(0.0),
                    },
                ),
            );
        }
        if refresh_wall_material {
            collider_materials_retuned = collider_materials_retuned.saturating_add(
                world.set_collider_material_for_collision_group(
                    COLLISION_GROUP_WALL,
                    ColliderMaterial {
                        restitution: self.config.wall_restitution.max(0.0),
                        friction: self.config.wall_friction.max(0.0),
                    },
                ),
            );
        }
        if refresh_solver_friction || refresh_solver_restitution {
            let mut solver = world.solver().config().clone();
            if refresh_solver_friction {
                solver.friction_transition_speed = self.config.friction_transition_speed.max(0.0);
                solver.friction_static_boost = self.config.friction_static_boost.max(0.0);
                solver.friction_kinetic_scale = self.config.friction_kinetic_scale.max(0.0);
            }
            if refresh_solver_restitution {
                solver.restitution_velocity_threshold =
                    self.config.restitution_velocity_threshold.max(0.0);
            }
            world.set_solver_config(solver);
        }

        BounceTankPatchMetrics {
            config_updated: updated,
            dynamic_bodies_retuned: retuned,
            collider_materials_retuned,
        }
    }

    /// Ensure static physics walls exist and then spawn a progressive ball batch.
    pub fn physics_tick(&mut self, world: &mut PhysicsWorld) -> BounceTankTickMetrics {
        self.ensure_container_walls(world);
        self.ensure_container_edge_barriers(world);
        self.cull_missing(world);
        if self.virtual_barrier_enabled() {
            let _ = self.recycle_escaped_balls(world);
        }
        let spawned = self.spawn_batch(world);
        let scattered = self.maybe_scatter_burst(world);
        let _ = self.apply_levitation_field(world);
        let _ = self.clamp_ball_velocity_for_collision_safety(world);
        BounceTankTickMetrics {
            spawned_this_tick: spawned,
            scattered_this_tick: scattered,
            live_balls: self.balls.len(),
            target_balls: self.config.target_ball_count,
            fully_spawned: self.balls.len() >= self.config.target_ball_count,
        }
    }

    /// Reconcile escaped dynamic balls after a fixed-step update.
    ///
    /// Call this right after `world.step(...)` to avoid one-frame visual pops where a high-energy
    /// body tunnels outside the prism before the next `physics_tick`.
    pub fn reconcile_after_step(&mut self, world: &mut PhysicsWorld) -> usize {
        let recycled = if self.virtual_barrier_enabled() {
            self.recycle_escaped_balls(world)
        } else {
            // Even with "virtual barrier off", keep a hard safety net for obvious tunnels so
            // dynamic balls do not remain visibly outside the container.
            self.recycle_hard_escaped_balls(world)
        };
        if recycled > 0 {
            // Interpolation uses stored snapshots captured inside `world.step(...)`.
            // If we corrected positions after the step, push one fresh snapshot so render doesn't
            // show an old out-of-bounds pose for a frame.
            // Push twice to replace both (previous, latest) interpolation endpoints.
            world.push_interpolation_snapshot();
            world.push_interpolation_snapshot();
        }
        recycled
    }

    /// Enforce an upper bound for live dynamic balls and destroy overflow bodies immediately.
    ///
    /// This is used by runtime-level performance governors to keep CPU broadphase/narrowphase
    /// cost bounded when scripts request very high object counts.
    pub fn enforce_live_ball_budget(
        &mut self,
        world: &mut PhysicsWorld,
        max_live_balls: usize,
    ) -> usize {
        if self.balls.len() <= max_live_balls {
            return 0;
        }
        let to_remove = self.balls.len() - max_live_balls;
        for _ in 0..to_remove {
            if let Some(ball) = self.balls.pop() {
                let _ = world.destroy_body(ball.body);
            }
        }
        to_remove
    }

    /// Build renderer-facing scene payloads using optional interpolation alpha.
    pub fn build_frame_instances(
        &self,
        world: &PhysicsWorld,
        interpolation_alpha: Option<f32>,
    ) -> SceneFrameInstances {
        self.build_frame_instances_with_ball_limit(world, interpolation_alpha, None)
    }

    /// Build renderer-facing scene payloads with an optional visible-ball cap.
    ///
    /// When `ball_render_limit` is lower than live body count, balls are sampled by stride to keep
    /// draw workload bounded without touching physics ownership/state.
    pub fn build_frame_instances_with_ball_limit(
        &self,
        world: &PhysicsWorld,
        interpolation_alpha: Option<f32>,
        ball_render_limit: Option<usize>,
    ) -> SceneFrameInstances {
        let mut frame = SceneFrameInstances::default();
        if self.container_mesh_slot.is_some() {
            self.append_container_wall_instances(&mut frame.transparent_3d);
        } else {
            frame.transparent_3d.push(self.container_visual_instance());
        }
        self.append_container_edge_instances(&mut frame.opaque_3d);

        let alpha = interpolation_alpha.unwrap_or(1.0).clamp(0.0, 1.0);
        let total_balls = self.balls.len();
        let visible_limit = ball_render_limit.unwrap_or(total_balls).min(total_balls);
        let stride = if visible_limit == 0 || visible_limit >= total_balls {
            1
        } else {
            ((total_balls + visible_limit - 1) / visible_limit).max(1)
        };
        let use_interpolation = alpha > 0.0 && total_balls <= 4_000;
        let primitive = self
            .ball_mesh_slot
            .map(|slot| ScenePrimitive3d::Mesh { slot })
            .unwrap_or(ScenePrimitive3d::Sphere);
        let mut ball_instances = self
            .balls
            .par_iter()
            .enumerate()
            .filter_map(|(index, ball)| {
                if stride > 1 && (index % stride != 0) {
                    return None;
                }
                let pose = if use_interpolation {
                    world
                        .interpolate_body_pose(ball.body, alpha)
                        .map(|interp| (interp.position, interp.rotation))
                        .or_else(|| {
                            world
                                .body(ball.body)
                                .map(|body| (body.position, body.rotation))
                        })
                } else {
                    world
                        .body(ball.body)
                        .map(|body| (body.position, body.rotation))
                };
                let (position, rotation) = pose?;
                let q = rotation.quaternion();
                Some(SceneInstance3d {
                    instance_id: ball.body.raw() as u64,
                    primitive,
                    transform: SceneTransform3d {
                        translation: [position.x, position.y, position.z],
                        rotation_xyzw: [q.i, q.j, q.k, q.w],
                        scale: [ball.radius * 2.0, ball.radius * 2.0, ball.radius * 2.0],
                    },
                    material: SceneMaterial {
                        base_color_rgba: ball.color,
                        roughness: ball.roughness,
                        metallic: ball.metallic,
                        emissive_rgb: [0.0, 0.0, 0.0],
                        shading: ShadingModel::LitPbr,
                    },
                    casts_shadow: true,
                    receives_shadow: true,
                })
            })
            .collect::<Vec<_>>();
        if ball_instances.len() > visible_limit {
            ball_instances.truncate(visible_limit);
        }
        frame.opaque_3d.extend(ball_instances);

        // Simple sprite overlay payload (spawn progress bar), kept renderer-agnostic.
        let progress = if self.config.target_ball_count == 0 {
            1.0
        } else {
            (self.balls.len() as f32 / self.config.target_ball_count as f32).clamp(0.0, 1.0)
        };
        if let Some(program) = &self.sprite_program {
            program.emit_instances(
                TlspriteFrameContext {
                    spawn_progress: progress,
                    live_balls: self.balls.len(),
                    target_balls: self.config.target_ball_count,
                },
                &mut frame.sprites,
            );
        }

        // Built-in fallback overlay (spawn progress bar), kept renderer-agnostic.
        if frame.sprites.is_empty() {
            frame.sprites.push(SpriteInstance {
                sprite_id: 1,
                kind: SpriteKind::Hud,
                position: [-0.86, 0.90, 0.0],
                size: [0.40 * progress.max(0.02), 0.035],
                rotation_rad: 0.0,
                color_rgba: [0.10, 0.84, 0.62, 0.92],
                texture_slot: 0,
                layer: 100,
            });
        }

        frame
    }

    fn cull_missing(&mut self, world: &PhysicsWorld) {
        self.balls.retain(|ball| world.body(ball.body).is_some());
        self.walls.retain(|wall| world.body(*wall).is_some());
        self.edge_barriers
            .retain(|barrier| world.body(*barrier).is_some());
    }

    fn ensure_container_walls(&mut self, world: &mut PhysicsWorld) {
        if self.walls.len() == 6 && self.walls.iter().all(|wall| world.body(*wall).is_some()) {
            return;
        }
        self.walls.clear();

        let hx = self.config.container_half_extents[0].max(0.5);
        let hy = self.config.container_half_extents[1].max(0.5);
        let hz = self.config.container_half_extents[2].max(0.5);
        // Keep physics walls thick enough for stable high-bounce scenes.
        let t = self
            .config
            .wall_thickness
            .max(self.config.ball_radius_max.max(0.02) * 1.35)
            .max(0.02);

        let walls = [
            (
                Vector3::new(hx + t * 0.5, 0.0, 0.0),
                Vector3::new(t * 0.5, hy + t, hz + t),
            ),
            (
                Vector3::new(-hx - t * 0.5, 0.0, 0.0),
                Vector3::new(t * 0.5, hy + t, hz + t),
            ),
            (
                Vector3::new(0.0, hy + t * 0.5, 0.0),
                Vector3::new(hx + t, t * 0.5, hz + t),
            ),
            (
                Vector3::new(0.0, -hy - t * 0.5, 0.0),
                Vector3::new(hx + t, t * 0.5, hz + t),
            ),
            (
                Vector3::new(0.0, 0.0, hz + t * 0.5),
                Vector3::new(hx + t, hy + t, t * 0.5),
            ),
            (
                Vector3::new(0.0, 0.0, -hz - t * 0.5),
                Vector3::new(hx + t, hy + t, t * 0.5),
            ),
        ];

        let wall_material = ColliderMaterial {
            restitution: self.config.wall_restitution.max(0.0),
            friction: self.config.wall_friction.max(0.0),
        };
        for (center, half_extents) in walls {
            let body = world.spawn_body(BodyDesc {
                kind: BodyKind::Static,
                position: center,
                local_bounds: Aabb::from_center_half_extents(Vector3::zeros(), half_extents),
                ..BodyDesc::default()
            });
            let _ = world.spawn_collider(ColliderDesc {
                body: Some(body),
                shape: ColliderShape::Aabb { half_extents },
                material: wall_material,
                is_sensor: false,
                user_tag: collision_filter_tag(COLLISION_GROUP_WALL, COLLISION_GROUP_BALL),
            });
            self.walls.push(body);
        }
    }

    fn ensure_container_edge_barriers(&mut self, world: &mut PhysicsWorld) {
        if self.edge_barriers.len() == 12
            && self
                .edge_barriers
                .iter()
                .all(|barrier| world.body(*barrier).is_some())
        {
            return;
        }
        self.edge_barriers.clear();

        let hx = self.config.container_half_extents[0].max(0.5);
        let hy = self.config.container_half_extents[1].max(0.5);
        let hz = self.config.container_half_extents[2].max(0.5);
        let edge = self.config.wall_thickness.max(0.03) * 0.30;
        let edge_half = edge.max(0.01) * 0.5;

        let x_edges = [
            ([0.0, hy, hz], [hx * 0.5, edge_half, edge_half]),
            ([0.0, hy, -hz], [hx * 0.5, edge_half, edge_half]),
            ([0.0, -hy, hz], [hx * 0.5, edge_half, edge_half]),
            ([0.0, -hy, -hz], [hx * 0.5, edge_half, edge_half]),
        ];
        let y_edges = [
            ([hx, 0.0, hz], [edge_half, hy * 0.5, edge_half]),
            ([hx, 0.0, -hz], [edge_half, hy * 0.5, edge_half]),
            ([-hx, 0.0, hz], [edge_half, hy * 0.5, edge_half]),
            ([-hx, 0.0, -hz], [edge_half, hy * 0.5, edge_half]),
        ];
        let z_edges = [
            ([hx, hy, 0.0], [edge_half, edge_half, hz * 0.5]),
            ([hx, -hy, 0.0], [edge_half, edge_half, hz * 0.5]),
            ([-hx, hy, 0.0], [edge_half, edge_half, hz * 0.5]),
            ([-hx, -hy, 0.0], [edge_half, edge_half, hz * 0.5]),
        ];

        let wall_material = ColliderMaterial {
            restitution: self.config.wall_restitution.max(0.0),
            friction: self.config.wall_friction.max(0.0),
        };

        for (center, half_extents) in x_edges
            .into_iter()
            .chain(y_edges.into_iter())
            .chain(z_edges.into_iter())
        {
            let half_extents = Vector3::new(half_extents[0], half_extents[1], half_extents[2]);
            let body = world.spawn_body(BodyDesc {
                kind: BodyKind::Static,
                position: Vector3::new(center[0], center[1], center[2]),
                local_bounds: Aabb::from_center_half_extents(Vector3::zeros(), half_extents),
                ..BodyDesc::default()
            });
            let _ = world.spawn_collider(ColliderDesc {
                body: Some(body),
                shape: ColliderShape::Aabb { half_extents },
                material: wall_material,
                is_sensor: false,
                user_tag: collision_filter_tag(COLLISION_GROUP_WALL, COLLISION_GROUP_BALL),
            });
            self.edge_barriers.push(body);
        }
    }

    fn spawn_batch(&mut self, world: &mut PhysicsWorld) -> usize {
        if self.balls.len() >= self.config.target_ball_count {
            return 0;
        }
        let live = self.balls.len();
        let remaining = self.config.target_ball_count.saturating_sub(live);
        // Stage spawn pressure so ball count grows progressively toward the target.
        let progress = if self.config.target_ball_count == 0 {
            1.0
        } else {
            (live as f32 / self.config.target_ball_count as f32).clamp(0.0, 1.0)
        };
        let stage_scale = if progress < 0.10 {
            0.22
        } else if progress < 0.25 {
            0.32
        } else if progress < 0.45 {
            0.48
        } else if progress < 0.65 {
            0.64
        } else if progress < 0.80 {
            0.74
        } else if progress < 0.92 {
            0.56
        } else {
            0.30
        };
        let staged_spawn_cap = ((self.config.spawn_per_tick.max(1) as f32) * stage_scale)
            .round()
            .clamp(8.0, 128.0) as usize;
        let to_spawn = remaining.min(staged_spawn_cap.max(1));
        let mut spawned = 0usize;

        let hx = self.config.container_half_extents[0].max(0.5);
        let hy = self.config.container_half_extents[1].max(0.5);
        let hz = self.config.container_half_extents[2].max(0.5);
        let material = ColliderMaterial {
            restitution: self.config.ball_restitution.max(0.0),
            friction: self.config.ball_friction.max(0.0),
        };

        for _ in 0..to_spawn {
            let radius = lerp(
                self.config.ball_radius_min.max(0.02),
                self.config
                    .ball_radius_max
                    .max(self.config.ball_radius_min.max(0.02)),
                self.rand01(),
            );
            let spawn_x = self.rand_range(-(hx - radius) * 0.86, (hx - radius) * 0.86);
            let spawn_z = self.rand_range(-(hz - radius) * 0.86, (hz - radius) * 0.86);
            let spawn_y = self.rand_range(hy * 0.45, hy * 0.88).min(hy - radius);
            let mass = (radius * radius * radius * 14.0).max(0.1);

            let speed = self.rand_range(
                self.config.initial_speed_min.max(0.0),
                self.config
                    .initial_speed_max
                    .max(self.config.initial_speed_min.max(0.0)),
            );
            let theta = self.rand01() * TAU;
            let vz = self.rand_range(-1.0, 1.0);
            let xy = (1.0 - vz * vz).sqrt();
            // Keep launch velocity bounded to configured speed range; avoid speed^2 blow-up.
            let mut launch_dir = Vector3::new(
                xy * theta.cos(),
                -self.rand_range(0.15, 0.55),
                xy * theta.sin(),
            );
            let launch_len = launch_dir.norm();
            if launch_len > 1e-6 {
                launch_dir /= launch_len;
            } else {
                launch_dir = Vector3::new(0.0, -1.0, 0.0);
            }
            let velocity = launch_dir * speed;

            let body = world.spawn_body(BodyDesc {
                kind: BodyKind::Dynamic,
                position: Vector3::new(spawn_x, spawn_y, spawn_z),
                linear_velocity: velocity,
                mass,
                linear_damping: self.config.linear_damping.max(0.0),
                local_bounds: Aabb::from_center_half_extents(
                    Vector3::zeros(),
                    Vector3::repeat(radius),
                ),
                ..BodyDesc::default()
            });
            let collider_ok = world
                .spawn_collider(ColliderDesc {
                    body: Some(body),
                    shape: ColliderShape::Sphere { radius },
                    material,
                    is_sensor: false,
                    user_tag: collision_filter_tag(
                        COLLISION_GROUP_BALL,
                        COLLISION_GROUP_BALL | COLLISION_GROUP_WALL,
                    ),
                })
                .is_some();
            if !collider_ok {
                let _ = world.destroy_body(body);
                continue;
            }

            let color = hsv_to_rgb(self.rand01(), 0.72, 0.96);
            let roughness = self.rand_range(0.16, 0.86);
            let metallic = self.rand_range(0.0, 0.30);
            self.balls.push(BallVisual {
                body,
                radius,
                color: [color[0], color[1], color[2], 1.0],
                roughness,
                metallic,
            });
            spawned += 1;
        }

        spawned
    }

    /// Apply an occasional high-energy burst so the showcase naturally "splashes" without
    /// per-frame randomness overhead.
    fn maybe_scatter_burst(&mut self, world: &mut PhysicsWorld) -> usize {
        let strength = self.config.scatter_strength.clamp(0.0, 1.0);
        if strength <= 0.01 {
            return 0;
        }
        let min_live =
            ((self.config.target_ball_count as f32) * (0.18 + strength * 0.25)).round() as usize;
        if self.balls.len() < min_live.max(SCATTER_MIN_LIVE_BALLS_BASE) {
            return 0;
        }
        let tick = world.fixed_step_clock().tick();
        let interval = self.config.scatter_interval_ticks.max(1);
        if tick == 0 || tick.saturating_sub(self.last_scatter_tick) < interval {
            return 0;
        }

        let live = self.balls.len();
        let fraction = self.rand_range(0.003 + 0.007 * strength, 0.010 + 0.030 * strength);
        let divisor = ((12.0 - strength * 6.0).round() as usize).clamp(6, 12);
        let mut target = ((live as f32) * fraction).round() as usize;
        target = target
            .max(SCATTER_MIN_AFFECTED.min(live))
            .min((live / divisor).max(SCATTER_MIN_AFFECTED));
        if target == 0 {
            return 0;
        }

        let stride = (live / target).max(1);
        let start = ((self.rand01() * stride as f32) as usize).min(stride.saturating_sub(1));
        let base_speed = self
            .config
            .initial_speed_max
            .max(self.config.initial_speed_min)
            .max(0.8);
        let burst_min = base_speed * (1.02 + 0.35 * strength);
        let burst_max = base_speed * (1.25 + 0.95 * strength);

        // Cycle bursts across six axis-aligned face directions so contact pressure reaches all
        // container sides over time instead of continuously biasing toward diagonal corner piles.
        let face_dirs = [
            [1.0_f32, 0.0, 0.0],
            [-1.0_f32, 0.0, 0.0],
            [0.0_f32, 0.0, 1.0],
            [0.0_f32, 0.0, -1.0],
            [0.0_f32, 1.0, 0.0],
            [0.0_f32, -1.0, 0.0],
        ];
        let face = face_dirs[(self.scatter_face_cursor as usize) % face_dirs.len()];
        self.scatter_face_cursor = self.scatter_face_cursor.wrapping_add(1);
        let face_dir = Vector3::new(face[0], face[1], face[2]);
        let lateral_jitter = 0.22 + strength * 0.35;
        let up_bias = self.config.container_half_extents[1] * 0.55;

        let mut scattered = 0usize;
        for i in (start..live).step_by(stride) {
            if scattered >= target {
                break;
            }
            let ball = self.balls[i];
            let Some(body) = world.body(ball.body) else {
                continue;
            };
            let mut dir = face_dir;
            if face_dir.x.abs() < 0.5 {
                dir.x += self.rand_range(-lateral_jitter, lateral_jitter);
            } else {
                dir.x += self.rand_range(-0.12, 0.12);
            }
            if face_dir.z.abs() < 0.5 {
                dir.z += self.rand_range(-lateral_jitter, lateral_jitter);
            } else {
                dir.z += self.rand_range(-0.12, 0.12);
            }
            if face_dir.y.abs() < 0.5 {
                dir.y += self.rand_range(0.05, 0.28 + 0.25 * strength);
            } else {
                dir.y += self.rand_range(-0.10, 0.10);
            }
            if body.position.y < -up_bias {
                dir.y += self.rand_range(0.18, 0.42);
            }
            let len2 = dir.dot(&dir);
            if len2 <= 1e-8 {
                continue;
            }
            let speed = self.rand_range(burst_min, burst_max);
            let velocity = (dir / len2.sqrt()) * speed;
            if world.set_velocity(ball.body, velocity) {
                scattered = scattered.saturating_add(1);
            }
        }

        if scattered > 0 {
            self.last_scatter_tick = tick;
        }
        scattered
    }

    /// Apply spring-damper hover control and optional X/Z momentum glide on dynamic balls.
    ///
    /// The vertical path keeps balls floating while preserving external momentum under stress.
    /// The lateral path adds deterministic X/Z glide plus side-wall steering so balls also move
    /// and rebound across horizontal axes during the showcase.
    fn apply_levitation_field(&mut self, world: &mut PhysicsWorld) -> usize {
        let vertical_enabled = self.config.levitation_strength > 1e-5;
        let lateral_strength = self.config.levitation_lateral_strength.max(0.0);
        let lateral_damping = self.config.levitation_lateral_damping.max(0.0);
        let lateral_max_speed = self
            .config
            .levitation_lateral_max_horizontal_speed
            .max(0.05);
        let lateral_wall_push = self.config.levitation_lateral_wall_push.max(0.0);
        let lateral_frequency = self.config.levitation_lateral_frequency.max(0.0);
        let lateral_enabled = lateral_strength > 1e-5 || lateral_wall_push > 1e-5;
        if !vertical_enabled && !lateral_enabled {
            return 0;
        }

        let dt = world.config().fixed_dt.max(1e-4);
        let hy = self.config.container_half_extents[1].max(0.5);
        let hx = self.config.container_half_extents[0].max(0.5);
        let hz = self.config.container_half_extents[2].max(0.5);
        let target_y = self.config.levitation_height.clamp(-hy * 0.95, hy * 0.95);
        let strength = self.config.levitation_strength.max(0.0);
        let damping = self.config.levitation_damping.max(0.0);
        let max_vy = self.config.levitation_max_vertical_speed.max(0.05);
        let reaction_strength = self.config.levitation_reaction_strength.max(0.0);
        let reaction_radius = self.config.levitation_reaction_radius.max(0.0);
        let reaction_damping = self.config.levitation_reaction_damping.max(0.0);
        let reaction_enabled =
            vertical_enabled && reaction_strength > 1e-5 && reaction_radius > 0.05;
        let reaction_r2 = reaction_radius * reaction_radius;
        let inv_cell = if reaction_enabled {
            1.0 / reaction_radius.max(0.05)
        } else {
            0.0
        };
        let side_wall_band_x = hx * 0.86;
        let side_wall_band_z = hz * 0.86;

        if lateral_enabled {
            self.levitation_lateral_phase =
                (self.levitation_lateral_phase + dt * lateral_frequency * TAU).rem_euclid(TAU);
        }

        self.levitation_body_handles.clear();
        self.levitation_positions.clear();
        self.levitation_velocities.clear();
        self.levitation_body_handles.reserve(self.balls.len());
        self.levitation_positions.reserve(self.balls.len());
        self.levitation_velocities.reserve(self.balls.len());
        for ball in &self.balls {
            let Some(body) = world.body(ball.body) else {
                continue;
            };
            self.levitation_body_handles.push(ball.body);
            self.levitation_positions
                .push([body.position.x, body.position.y, body.position.z]);
            self.levitation_velocities.push([
                body.linear_velocity.x,
                body.linear_velocity.y,
                body.linear_velocity.z,
            ]);
        }
        if self.levitation_body_handles.is_empty() {
            return 0;
        }

        self.levitation_reaction_cells.clear();
        if reaction_enabled {
            self.levitation_reaction_cells
                .reserve(self.levitation_body_handles.len() / 2);
            for (index, pos) in self.levitation_positions.iter().enumerate() {
                let key = LevitationCellKey {
                    x: (pos[0] * inv_cell).floor() as i32,
                    y: (pos[1] * inv_cell).floor() as i32,
                    z: (pos[2] * inv_cell).floor() as i32,
                };
                self.levitation_reaction_cells
                    .entry(key)
                    .or_default()
                    .push(index);
            }
        }

        let mut adjusted = 0usize;
        for i in 0..self.levitation_body_handles.len() {
            let pos = self.levitation_positions[i];
            let mut velocity = self.levitation_velocities[i];

            if vertical_enabled {
                let current_vy = velocity[1];
                let error = target_y - pos[1];
                let mut accel_y = error * strength - current_vy * damping;

                if reaction_enabled {
                    let base = LevitationCellKey {
                        x: (pos[0] * inv_cell).floor() as i32,
                        y: (pos[1] * inv_cell).floor() as i32,
                        z: (pos[2] * inv_cell).floor() as i32,
                    };
                    let mut reaction = 0.0f32;
                    for nx in (base.x - 1)..=(base.x + 1) {
                        for ny in (base.y - 1)..=(base.y + 1) {
                            for nz in (base.z - 1)..=(base.z + 1) {
                                let key = LevitationCellKey {
                                    x: nx,
                                    y: ny,
                                    z: nz,
                                };
                                let Some(indices) = self.levitation_reaction_cells.get(&key) else {
                                    continue;
                                };
                                for &j in indices {
                                    if j == i {
                                        continue;
                                    }
                                    let other = self.levitation_positions[j];
                                    let dx = pos[0] - other[0];
                                    let dy = pos[1] - other[1];
                                    let dz = pos[2] - other[2];
                                    let dist2 = dx * dx + dy * dy + dz * dz;
                                    if dist2 >= reaction_r2 || dist2 <= 1e-8 {
                                        continue;
                                    }
                                    let dist = dist2.sqrt();
                                    let proximity = (1.0 - (dist / reaction_radius)).max(0.0);
                                    reaction += (dy / dist.max(1e-4)) * proximity;
                                }
                            }
                        }
                    }
                    accel_y += reaction * reaction_strength - current_vy * reaction_damping;
                }

                let mut next_vy = current_vy + accel_y * dt;
                if next_vy.abs() > max_vy {
                    if current_vy.abs() <= max_vy {
                        // When inside the configured envelope we keep levitation bounded.
                        next_vy = next_vy.clamp(-max_vy, max_vy);
                    } else {
                        // Preserve external momentum (gravity/scatter bursts) and avoid abrupt cuts.
                        // We only block extra acceleration that would push further in the same direction.
                        let same_sign = next_vy.signum() == current_vy.signum();
                        let speeding_up = next_vy.abs() > current_vy.abs();
                        if same_sign && speeding_up {
                            next_vy = current_vy;
                        }
                    }
                }
                velocity[1] = next_vy;
            }

            if lateral_enabled {
                let current_vx = velocity[0];
                let current_vz = velocity[2];
                let mut accel_x = 0.0f32;
                let mut accel_z = 0.0f32;

                if lateral_strength > 1e-5 {
                    let body_raw = self.levitation_body_handles[i].raw();
                    let phase_offset = (body_raw % 1024) as f32 * (TAU / 1024.0);
                    let angle = self.levitation_lateral_phase + phase_offset;
                    let target_vx = lateral_strength * angle.cos();
                    let target_vz = lateral_strength * angle.sin();
                    accel_x += (target_vx - current_vx) * lateral_damping;
                    accel_z += (target_vz - current_vz) * lateral_damping;
                }

                if lateral_wall_push > 1e-5 {
                    if pos[0].abs() > side_wall_band_x {
                        accel_x += -pos[0].signum() * lateral_wall_push;
                    }
                    if pos[2].abs() > side_wall_band_z {
                        accel_z += -pos[2].signum() * lateral_wall_push;
                    }
                }

                let mut next_vx = current_vx + accel_x * dt;
                let mut next_vz = current_vz + accel_z * dt;
                let speed_sq = next_vx * next_vx + next_vz * next_vz;
                let max_sq = lateral_max_speed * lateral_max_speed;
                if speed_sq > max_sq {
                    let scale = lateral_max_speed / speed_sq.sqrt().max(1e-5);
                    next_vx *= scale;
                    next_vz *= scale;
                }
                velocity[0] = next_vx;
                velocity[2] = next_vz;
            }

            if world.set_velocity(
                self.levitation_body_handles[i],
                Vector3::new(velocity[0], velocity[1], velocity[2]),
            ) {
                adjusted = adjusted.saturating_add(1);
            }
        }
        adjusted
    }

    /// Clamps rare escaped balls back inside the prism instead of deleting them.
    ///
    /// This avoids visible pop-in/pop-out under high-energy bursts while keeping live count stable.
    fn recycle_escaped_balls(&mut self, world: &mut PhysicsWorld) -> usize {
        let hx = self.config.container_half_extents[0].max(0.5);
        let hy = self.config.container_half_extents[1].max(0.5);
        let hz = self.config.container_half_extents[2].max(0.5);
        let wall_bounce = self.config.wall_restitution.clamp(0.45, 0.98);
        let mut recycled = 0usize;

        let mut i = 0usize;
        while i < self.balls.len() {
            let handle = self.balls[i].body;
            let radius = self.balls[i].radius.max(0.02);
            let Some(body) = world.body(handle) else {
                self.balls.swap_remove(i);
                continue;
            };

            let mut position = body.position;
            let mut velocity = body.linear_velocity;
            let limit_x = (hx - radius).max(radius * 0.5);
            let limit_y = (hy - radius).max(radius * 0.5);
            let limit_z = (hz - radius).max(radius * 0.5);
            // Only recycle when the body clearly escaped beyond a tolerance band.
            // If we clamp multiple axes in one shot, escaped bodies collapse into corners.
            let recycle_margin = (self.config.wall_thickness.max(radius * 0.45)).max(0.06);
            let contact_slop = (radius * 0.02).max(0.0025);
            let over_x = (position.x.abs() - limit_x).max(0.0);
            let over_y = (position.y.abs() - limit_y).max(0.0);
            let over_z = (position.z.abs() - limit_z).max(0.0);

            let max_over = over_x.max(over_y.max(over_z));
            if max_over <= recycle_margin {
                i += 1;
                continue;
            }

            let escaped_axes = u8::from(over_x > contact_slop)
                + u8::from(over_y > contact_slop)
                + u8::from(over_z > contact_slop);
            let dominant_only = escaped_axes >= 2 && max_over > recycle_margin * 3.0;
            let inset = (radius * 0.22).clamp(0.02, 0.10);
            let mut corrected_axes = 0u8;
            if dominant_only {
                if over_x >= over_y && over_x >= over_z {
                    if position.x > limit_x {
                        position.x = (limit_x - inset).max(radius * 0.5);
                        velocity.x = -velocity.x.abs() * wall_bounce;
                    } else {
                        position.x = -(limit_x - inset).max(radius * 0.5);
                        velocity.x = velocity.x.abs() * wall_bounce;
                    }
                    velocity.y *= 0.985;
                    velocity.z *= 0.985;
                    corrected_axes = 1;
                } else if over_y >= over_z {
                    if position.y > limit_y {
                        position.y = (limit_y - inset).max(radius * 0.5);
                        velocity.y = -velocity.y.abs() * wall_bounce;
                    } else {
                        position.y = -(limit_y - inset).max(radius * 0.5);
                        velocity.y = velocity.y.abs() * wall_bounce;
                    }
                    velocity.x *= 0.985;
                    velocity.z *= 0.985;
                    corrected_axes = 1;
                } else {
                    if position.z > limit_z {
                        position.z = (limit_z - inset).max(radius * 0.5);
                        velocity.z = -velocity.z.abs() * wall_bounce;
                    } else {
                        position.z = -(limit_z - inset).max(radius * 0.5);
                        velocity.z = velocity.z.abs() * wall_bounce;
                    }
                    velocity.x *= 0.985;
                    velocity.y *= 0.985;
                    corrected_axes = 1;
                }
            } else {
                if position.x > limit_x + contact_slop {
                    position.x = (limit_x - inset).max(radius * 0.5);
                    if velocity.x > 0.0 {
                        velocity.x = -velocity.x * wall_bounce;
                    }
                    corrected_axes = corrected_axes.saturating_add(1);
                } else if position.x < -(limit_x + contact_slop) {
                    position.x = -(limit_x - inset).max(radius * 0.5);
                    if velocity.x < 0.0 {
                        velocity.x = -velocity.x * wall_bounce;
                    }
                    corrected_axes = corrected_axes.saturating_add(1);
                }
                if position.y > limit_y + contact_slop {
                    position.y = (limit_y - inset).max(radius * 0.5);
                    if velocity.y > 0.0 {
                        velocity.y = -velocity.y * wall_bounce;
                    }
                    corrected_axes = corrected_axes.saturating_add(1);
                } else if position.y < -(limit_y + contact_slop) {
                    position.y = -(limit_y - inset).max(radius * 0.5);
                    if velocity.y < 0.0 {
                        velocity.y = -velocity.y * wall_bounce;
                    }
                    corrected_axes = corrected_axes.saturating_add(1);
                }
                if position.z > limit_z + contact_slop {
                    position.z = (limit_z - inset).max(radius * 0.5);
                    if velocity.z > 0.0 {
                        velocity.z = -velocity.z * wall_bounce;
                    }
                    corrected_axes = corrected_axes.saturating_add(1);
                } else if position.z < -(limit_z + contact_slop) {
                    position.z = -(limit_z - inset).max(radius * 0.5);
                    if velocity.z < 0.0 {
                        velocity.z = -velocity.z * wall_bounce;
                    }
                    corrected_axes = corrected_axes.saturating_add(1);
                }
                if corrected_axes > 1 {
                    velocity *= 0.992;
                }
            }
            if corrected_axes == 0 {
                i += 1;
                continue;
            }

            let max_rebound_speed = self.config.initial_speed_max.max(1.0)
                * (3.5 + self.config.scatter_strength.clamp(0.0, 1.0) * 8.0);
            let speed_sq = velocity.norm_squared();
            if speed_sq > max_rebound_speed * max_rebound_speed {
                let inv = max_rebound_speed / speed_sq.sqrt();
                velocity *= inv;
            }
            let _ = world.set_position(handle, position);
            let _ = world.set_velocity(handle, velocity);
            recycled = recycled.saturating_add(1);
            i += 1;
        }

        recycled
    }

    /// Hard safety recycle path used when `virtual_barrier` is disabled.
    ///
    /// This path is intentionally conservative: it only corrects bodies that clearly tunneled
    /// outside the container by a wide margin, so normal wall interaction remains purely physics
    /// driven in the common case.
    fn recycle_hard_escaped_balls(&mut self, world: &mut PhysicsWorld) -> usize {
        let hx = self.config.container_half_extents[0].max(0.5);
        let hy = self.config.container_half_extents[1].max(0.5);
        let hz = self.config.container_half_extents[2].max(0.5);
        let wall_bounce = self.config.wall_restitution.clamp(0.45, 0.98);
        let mut recycled = 0usize;

        let mut i = 0usize;
        while i < self.balls.len() {
            let handle = self.balls[i].body;
            let radius = self.balls[i].radius.max(0.02);
            let Some(body) = world.body(handle) else {
                self.balls.swap_remove(i);
                continue;
            };

            let mut position = body.position;
            let mut velocity = body.linear_velocity;
            let limit_x = (hx - radius).max(radius * 0.5);
            let limit_y = (hy - radius).max(radius * 0.5);
            let limit_z = (hz - radius).max(radius * 0.5);
            let hard_margin = (self.config.wall_thickness.max(radius * 0.45)).max(0.06) * 3.5;
            let over_x = (position.x.abs() - limit_x).max(0.0);
            let over_y = (position.y.abs() - limit_y).max(0.0);
            let over_z = (position.z.abs() - limit_z).max(0.0);
            let max_over = over_x.max(over_y.max(over_z));
            if max_over <= hard_margin {
                i += 1;
                continue;
            }

            let contact_slop = (radius * 0.03).max(0.005);
            let inset = (radius * 0.22).clamp(0.02, 0.12);
            let mut corrected = false;

            if position.x > limit_x + contact_slop {
                position.x = (limit_x - inset).max(radius * 0.5);
                velocity.x = -velocity.x.abs() * wall_bounce;
                corrected = true;
            } else if position.x < -(limit_x + contact_slop) {
                position.x = -(limit_x - inset).max(radius * 0.5);
                velocity.x = velocity.x.abs() * wall_bounce;
                corrected = true;
            }
            if position.y > limit_y + contact_slop {
                position.y = (limit_y - inset).max(radius * 0.5);
                velocity.y = -velocity.y.abs() * wall_bounce;
                corrected = true;
            } else if position.y < -(limit_y + contact_slop) {
                position.y = -(limit_y - inset).max(radius * 0.5);
                velocity.y = velocity.y.abs() * wall_bounce;
                corrected = true;
            }
            if position.z > limit_z + contact_slop {
                position.z = (limit_z - inset).max(radius * 0.5);
                velocity.z = -velocity.z.abs() * wall_bounce;
                corrected = true;
            } else if position.z < -(limit_z + contact_slop) {
                position.z = -(limit_z - inset).max(radius * 0.5);
                velocity.z = velocity.z.abs() * wall_bounce;
                corrected = true;
            }
            if !corrected {
                i += 1;
                continue;
            }

            // Keep post-recycle rebound finite so one escaped body doesn't re-tunnel repeatedly.
            let speed_cap = self.collision_safety_speed_cap(world) * 0.90;
            let speed_sq = velocity.norm_squared();
            let speed_cap_sq = speed_cap * speed_cap;
            if speed_sq > speed_cap_sq {
                velocity *= speed_cap / speed_sq.sqrt().max(1e-5);
            }
            let _ = world.set_position(handle, position);
            let _ = world.set_velocity(handle, velocity);
            recycled = recycled.saturating_add(1);
            i += 1;
        }

        recycled
    }

    /// Clamp rare outlier velocities so one-step motion cannot regularly leap through thin walls.
    ///
    /// This protects against energy spikes from high restitution + burst combinations without
    /// flattening normal motion under typical showcase settings.
    fn clamp_ball_velocity_for_collision_safety(&mut self, world: &mut PhysicsWorld) -> usize {
        let cap = self.collision_safety_speed_cap(world);
        let cap_sq = cap * cap;
        let axis_cap = cap * 0.86;
        let mut clamped = 0usize;

        for ball in &self.balls {
            let Some(body) = world.body(ball.body) else {
                continue;
            };
            let mut velocity = body.linear_velocity;
            let mut changed = false;

            if velocity.x.abs() > axis_cap {
                velocity.x = velocity.x.signum() * axis_cap;
                changed = true;
            }
            if velocity.y.abs() > axis_cap {
                velocity.y = velocity.y.signum() * axis_cap;
                changed = true;
            }
            if velocity.z.abs() > axis_cap {
                velocity.z = velocity.z.signum() * axis_cap;
                changed = true;
            }

            let speed_sq = velocity.norm_squared();
            if speed_sq > cap_sq {
                velocity *= cap / speed_sq.sqrt().max(1e-5);
                changed = true;
            }

            if changed && world.set_velocity(ball.body, velocity) {
                clamped = clamped.saturating_add(1);
            }
        }

        clamped
    }

    #[inline]
    fn collision_safety_speed_cap(&self, world: &PhysicsWorld) -> f32 {
        let dt = world.config().fixed_dt.max(1e-4);
        let radius = self
            .config
            .ball_radius_max
            .max(self.config.ball_radius_min)
            .max(0.02);
        let wall = self
            .config
            .wall_thickness
            .max(self.config.ball_radius_max.max(0.02) * 1.35)
            .max(0.02);
        let tunnel_cap = ((wall + radius * 1.25) / dt) * 0.92;
        let design_cap = self
            .config
            .initial_speed_max
            .max(self.config.initial_speed_min)
            .max(1.0)
            * (8.0 + self.config.scatter_strength.clamp(0.0, 1.0) * 18.0);
        tunnel_cap.max(design_cap).clamp(12.0, 220.0)
    }

    fn container_visual_instance(&self) -> SceneInstance3d {
        SceneInstance3d {
            instance_id: u64::MAX - 1,
            primitive: ScenePrimitive3d::Box,
            transform: SceneTransform3d {
                translation: [0.0, 0.0, 0.0],
                rotation_xyzw: [0.0, 0.0, 0.0, 1.0],
                scale: [
                    self.config.container_half_extents[0] * 2.0,
                    self.config.container_half_extents[1] * 2.0,
                    self.config.container_half_extents[2] * 2.0,
                ],
            },
            material: SceneMaterial {
                base_color_rgba: [0.58, 0.72, 0.98, 0.30],
                roughness: 0.04,
                metallic: 0.0,
                emissive_rgb: [0.05, 0.06, 0.08],
                shading: ShadingModel::LitPbr,
            },
            casts_shadow: false,
            receives_shadow: true,
        }
    }

    /// Adds six wall panel mesh instances (three pairs) so custom FBX walls form a clean box.
    fn append_container_wall_instances(&self, out: &mut Vec<SceneInstance3d>) {
        let Some(slot) = self.container_mesh_slot else {
            return;
        };
        let hx = self.config.container_half_extents[0].max(0.5);
        let hy = self.config.container_half_extents[1].max(0.5);
        let hz = self.config.container_half_extents[2].max(0.5);
        let t = self
            .config
            .wall_thickness
            .max(self.config.ball_radius_max.max(0.02) * 1.35)
            .max(0.02);
        let primitive = ScenePrimitive3d::Mesh { slot };
        let mesh_scale = self.config.container_mesh_scale;
        let faces = [
            (
                [hx + t * 0.5, 0.0, 0.0],
                [t, (hy + t) * 2.0, (hz + t) * 2.0],
            ),
            (
                [-hx - t * 0.5, 0.0, 0.0],
                [t, (hy + t) * 2.0, (hz + t) * 2.0],
            ),
            (
                [0.0, hy + t * 0.5, 0.0],
                [(hx + t) * 2.0, t, (hz + t) * 2.0],
            ),
            (
                [0.0, -hy - t * 0.5, 0.0],
                [(hx + t) * 2.0, t, (hz + t) * 2.0],
            ),
            (
                [0.0, 0.0, hz + t * 0.5],
                [(hx + t) * 2.0, (hy + t) * 2.0, t],
            ),
            (
                [0.0, 0.0, -hz - t * 0.5],
                [(hx + t) * 2.0, (hy + t) * 2.0, t],
            ),
        ];
        for (index, (translation, scale)) in faces.into_iter().enumerate() {
            let shaped_scale = [
                scale[0] * mesh_scale[0],
                scale[1] * mesh_scale[1],
                scale[2] * mesh_scale[2],
            ];
            out.push(SceneInstance3d {
                instance_id: (u64::MAX - 200).saturating_sub(index as u64),
                primitive,
                transform: SceneTransform3d {
                    translation,
                    rotation_xyzw: [0.0, 0.0, 0.0, 1.0],
                    scale: shaped_scale,
                },
                material: SceneMaterial {
                    base_color_rgba: [0.80, 0.89, 1.0, 0.26],
                    roughness: 0.06,
                    metallic: 0.0,
                    emissive_rgb: [0.05, 0.07, 0.10],
                    shading: ShadingModel::LitPbr,
                },
                casts_shadow: false,
                receives_shadow: true,
            });
        }
    }

    /// Adds 12 thin edge prisms so the tank silhouette remains readable from every camera angle.
    fn append_container_edge_instances(&self, out: &mut Vec<SceneInstance3d>) {
        let hx = self.config.container_half_extents[0] * 2.0;
        let hy = self.config.container_half_extents[1] * 2.0;
        let hz = self.config.container_half_extents[2] * 2.0;
        let edge = self.config.wall_thickness.max(0.03) * 0.30;
        let primitive = ScenePrimitive3d::Box;

        // X-axis edges (y/z corners).
        let x_edges = [
            (
                [
                    0.0,
                    self.config.container_half_extents[1],
                    self.config.container_half_extents[2],
                ],
                [hx, edge, edge],
            ),
            (
                [
                    0.0,
                    self.config.container_half_extents[1],
                    -self.config.container_half_extents[2],
                ],
                [hx, edge, edge],
            ),
            (
                [
                    0.0,
                    -self.config.container_half_extents[1],
                    self.config.container_half_extents[2],
                ],
                [hx, edge, edge],
            ),
            (
                [
                    0.0,
                    -self.config.container_half_extents[1],
                    -self.config.container_half_extents[2],
                ],
                [hx, edge, edge],
            ),
        ];
        // Y-axis edges (x/z corners).
        let y_edges = [
            (
                [
                    self.config.container_half_extents[0],
                    0.0,
                    self.config.container_half_extents[2],
                ],
                [edge, hy, edge],
            ),
            (
                [
                    self.config.container_half_extents[0],
                    0.0,
                    -self.config.container_half_extents[2],
                ],
                [edge, hy, edge],
            ),
            (
                [
                    -self.config.container_half_extents[0],
                    0.0,
                    self.config.container_half_extents[2],
                ],
                [edge, hy, edge],
            ),
            (
                [
                    -self.config.container_half_extents[0],
                    0.0,
                    -self.config.container_half_extents[2],
                ],
                [edge, hy, edge],
            ),
        ];
        // Z-axis edges (x/y corners).
        let z_edges = [
            (
                [
                    self.config.container_half_extents[0],
                    self.config.container_half_extents[1],
                    0.0,
                ],
                [edge, edge, hz],
            ),
            (
                [
                    self.config.container_half_extents[0],
                    -self.config.container_half_extents[1],
                    0.0,
                ],
                [edge, edge, hz],
            ),
            (
                [
                    -self.config.container_half_extents[0],
                    self.config.container_half_extents[1],
                    0.0,
                ],
                [edge, edge, hz],
            ),
            (
                [
                    -self.config.container_half_extents[0],
                    -self.config.container_half_extents[1],
                    0.0,
                ],
                [edge, edge, hz],
            ),
        ];

        let mut edge_index: u64 = 0;
        for (translation, scale) in x_edges
            .into_iter()
            .chain(y_edges.into_iter())
            .chain(z_edges.into_iter())
        {
            out.push(SceneInstance3d {
                instance_id: (u64::MAX - 20).saturating_sub(edge_index),
                primitive,
                transform: SceneTransform3d {
                    translation,
                    rotation_xyzw: [0.0, 0.0, 0.0, 1.0],
                    scale,
                },
                material: SceneMaterial {
                    base_color_rgba: [0.90, 0.96, 1.0, 0.56],
                    roughness: 0.06,
                    metallic: 0.0,
                    emissive_rgb: [0.06, 0.09, 0.12],
                    shading: ShadingModel::LitPbr,
                },
                casts_shadow: false,
                receives_shadow: false,
            });
            edge_index = edge_index.saturating_add(1);
        }
    }

    fn rand01(&mut self) -> f32 {
        // xorshift64*
        let mut x = self.rng_state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rng_state = x;
        let value = x.wrapping_mul(0x2545_F491_4F6C_DD1D);
        ((value >> 11) as f64 / ((1u64 << 53) as f64)) as f32
    }

    fn rand_range(&mut self, min: f32, max: f32) -> f32 {
        if (max - min).abs() <= f32::EPSILON {
            min
        } else if max > min {
            min + (max - min) * self.rand01()
        } else {
            max + (min - max) * self.rand01()
        }
    }
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t.clamp(0.0, 1.0)
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    let h = h.rem_euclid(1.0);
    let s = s.clamp(0.0, 1.0);
    let v = v.clamp(0.0, 1.0);
    if s <= f32::EPSILON {
        return [v, v, v];
    }
    let sector = (h * 6.0).floor();
    let f = h * 6.0 - sector;
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));
    match (sector as i32).rem_euclid(6) {
        0 => [v, t, p],
        1 => [q, v, p],
        2 => [p, v, t],
        3 => [p, q, v],
        4 => [t, p, v],
        _ => [v, p, q],
    }
}

fn quantize_tick_hz(target: f32) -> f32 {
    const CANDIDATES: &[f32] = &[
        30.0, 48.0, 60.0, 72.0, 90.0, 100.0, 120.0, 144.0, 165.0, 180.0, 200.0, 240.0, 300.0,
        360.0, 420.0, 480.0, 540.0, 600.0, 720.0, 840.0, 960.0, 1200.0,
    ];
    let target = target.max(1.0);
    CANDIDATES
        .iter()
        .copied()
        .min_by(|a, b| {
            (a - target)
                .abs()
                .partial_cmp(&(b - target).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(target)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile_tlsprite;
    use paradoxpe::PhysicsWorldConfig;

    #[test]
    fn tick_policy_uses_vsync_basis_with_ratio() {
        let policy = TickRatePolicy::default();
        let tick_hz = policy.resolve_tick_hz(RenderSyncMode::Vsync { display_hz: 60.0 }, None);
        assert!(tick_hz >= 100.0);
        assert!(
            (policy.resolve_fixed_dt_seconds(RenderSyncMode::Vsync { display_hz: 60.0 }, None)
                - (1.0 / tick_hz))
                .abs()
                < 1e-6
        );
    }

    #[test]
    fn bounce_scene_spawns_progressively_and_emits_instances() {
        let mut world = PhysicsWorld::new(PhysicsWorldConfig {
            fixed_dt: 1.0 / 120.0,
            ..PhysicsWorldConfig::default()
        });
        let mut scene = BounceTankSceneController::new(BounceTankSceneConfig {
            target_ball_count: 256,
            spawn_per_tick: 64,
            container_half_extents: [8.0, 8.0, 8.0],
            ..BounceTankSceneConfig::default()
        });

        let mut total_spawned = 0usize;
        for _ in 0..8 {
            let metrics = scene.physics_tick(&mut world);
            total_spawned += metrics.spawned_this_tick;
            let _ = world.step(world.config().fixed_dt);
        }

        assert_eq!(scene.wall_count(), 6);
        assert_eq!(scene.edge_barrier_count(), 12);
        assert!(total_spawned > 0);
        assert!(scene.live_ball_count() <= 256);

        let frame = scene.build_frame_instances(&world, Some(0.5));
        assert!(frame.transparent_3d.is_empty());
        assert!(!frame.sprites.is_empty());
        assert!(!frame.opaque_3d.is_empty());
    }

    #[test]
    fn runtime_patch_clamps_values_and_retunes_dynamic_bodies() {
        let mut world = PhysicsWorld::new(PhysicsWorldConfig {
            fixed_dt: 1.0 / 120.0,
            ..PhysicsWorldConfig::default()
        });
        let mut scene = BounceTankSceneController::new(BounceTankSceneConfig {
            target_ball_count: 128,
            spawn_per_tick: 128,
            ..BounceTankSceneConfig::default()
        });
        let _ = scene.physics_tick(&mut world);
        let _ = world.step(1.0 / 120.0);

        let metrics = scene.apply_runtime_patch(
            &mut world,
            BounceTankRuntimePatch {
                spawn_per_tick: Some(0),
                linear_damping: Some(2.0),
                gravity: Some([0.0, -14.5, 0.0]),
                contact_guard: Some(0.92),
                initial_speed_min: Some(3.0),
                initial_speed_max: Some(1.0),
                ball_friction: Some(0.48),
                wall_friction: Some(0.22),
                friction_transition_speed: Some(1.65),
                friction_static_boost: Some(1.28),
                friction_kinetic_scale: Some(0.84),
                restitution_velocity_threshold: Some(0.25),
                ..BounceTankRuntimePatch::default()
            },
        );
        let cfg = scene.config();
        assert_eq!(cfg.spawn_per_tick, 1);
        assert!(cfg.linear_damping <= 0.95);
        assert!(cfg.initial_speed_max >= cfg.initial_speed_min);
        assert!((cfg.ball_friction - 0.48).abs() < 1e-6);
        assert!((cfg.wall_friction - 0.22).abs() < 1e-6);
        assert!((cfg.friction_transition_speed - 1.65).abs() < 1e-6);
        assert!((cfg.friction_static_boost - 1.28).abs() < 1e-6);
        assert!((cfg.friction_kinetic_scale - 0.84).abs() < 1e-6);
        assert!((cfg.restitution_velocity_threshold - 0.25).abs() < 1e-6);
        assert!(world.gravity().y < -9.9);
        assert!(world.solver().config().hard_position_projection_strength > 1.0);
        assert!((world.solver().config().friction_transition_speed - 1.65).abs() < 1e-6);
        assert!((world.solver().config().friction_static_boost - 1.28).abs() < 1e-6);
        assert!((world.solver().config().friction_kinetic_scale - 0.84).abs() < 1e-6);
        assert!((world.solver().config().restitution_velocity_threshold - 0.25).abs() < 1e-6);
        assert!(metrics.config_updated);
        assert!(metrics.collider_materials_retuned > 0);
    }

    #[test]
    fn runtime_patch_can_switch_ball_and_container_mesh_slots() {
        let mut world = PhysicsWorld::new(PhysicsWorldConfig {
            fixed_dt: 1.0 / 120.0,
            ..PhysicsWorldConfig::default()
        });
        let mut scene = BounceTankSceneController::new(BounceTankSceneConfig {
            target_ball_count: 64,
            spawn_per_tick: 64,
            ..BounceTankSceneConfig::default()
        });
        let _ = scene.physics_tick(&mut world);
        let _ = world.step(world.config().fixed_dt);

        let _ = scene.apply_runtime_patch(
            &mut world,
            BounceTankRuntimePatch {
                ball_mesh_slot: Some(5),
                container_mesh_slot: Some(2),
                ..BounceTankRuntimePatch::default()
            },
        );

        let frame = scene.build_frame_instances(&world, Some(1.0));
        assert!(matches!(
            frame.transparent_3d.first().map(|i| i.primitive),
            Some(ScenePrimitive3d::Mesh { slot: 2 })
        ));
        assert!(frame
            .opaque_3d
            .iter()
            .any(|i| matches!(i.primitive, ScenePrimitive3d::Mesh { slot: 5 })));
    }

    #[test]
    fn tlsprite_program_overrides_builtin_progress_sprite() {
        let src = concat!(
            "tlsprite_v1\n",
            "[hud.progress]\n",
            "sprite_id = 77\n",
            "texture_slot = 3\n",
            "layer = 120\n",
            "position = -0.5, 0.9, 0.0\n",
            "size = 0.4, 0.03\n",
            "color = 0.9, 0.2, 0.2, 1.0\n",
            "scale_axis = x\n",
            "scale_source = spawn_progress\n",
            "scale_min = 0.1\n",
            "scale_max = 1.0\n",
        );
        let compile = compile_tlsprite(src);
        assert!(!compile.has_errors());
        let program = compile.program.expect("program");

        let mut world = PhysicsWorld::new(PhysicsWorldConfig {
            fixed_dt: 1.0 / 120.0,
            ..PhysicsWorldConfig::default()
        });
        let mut scene = BounceTankSceneController::new(BounceTankSceneConfig {
            target_ball_count: 200,
            spawn_per_tick: 100,
            ..BounceTankSceneConfig::default()
        });
        scene.set_sprite_program(program);

        let _ = scene.physics_tick(&mut world);
        let _ = world.step(world.config().fixed_dt);
        let frame = scene.build_frame_instances(&world, Some(0.5));
        assert_eq!(frame.sprites.len(), 1);
        assert_eq!(frame.sprites[0].sprite_id, 77);
        assert_eq!(frame.sprites[0].texture_slot, 3);
        assert_eq!(frame.sprites[0].kind, SpriteKind::Hud);
    }

    #[test]
    fn frame_build_respects_visible_ball_cap() {
        let mut world = PhysicsWorld::new(PhysicsWorldConfig {
            fixed_dt: 1.0 / 120.0,
            ..PhysicsWorldConfig::default()
        });
        let mut scene = BounceTankSceneController::new(BounceTankSceneConfig {
            target_ball_count: 300,
            spawn_per_tick: 300,
            ..BounceTankSceneConfig::default()
        });

        let _ = scene.physics_tick(&mut world);
        let _ = world.step(world.config().fixed_dt);
        let full = scene.build_frame_instances(&world, Some(1.0));
        let capped = scene.build_frame_instances_with_ball_limit(&world, Some(1.0), Some(64));

        assert!(full.opaque_3d.len() > capped.opaque_3d.len());
        // 12 entries are always container edge visuals.
        assert!(capped.opaque_3d.len() <= 12 + 64);
    }

    #[test]
    fn enforce_live_ball_budget_removes_overflow_bodies() {
        let mut world = PhysicsWorld::new(PhysicsWorldConfig {
            fixed_dt: 1.0 / 120.0,
            ..PhysicsWorldConfig::default()
        });
        let mut scene = BounceTankSceneController::new(BounceTankSceneConfig {
            target_ball_count: 256,
            spawn_per_tick: 256,
            ..BounceTankSceneConfig::default()
        });

        for _ in 0..8 {
            let _ = scene.physics_tick(&mut world);
            let _ = world.step(world.config().fixed_dt);
            if scene.live_ball_count() > 96 {
                break;
            }
        }
        let removed = scene.enforce_live_ball_budget(&mut world, 96);
        assert!(removed > 0);
        assert!(scene.live_ball_count() <= 96);
    }

    #[test]
    fn physics_tick_recycles_escaped_balls_back_into_container_budget() {
        let mut world = PhysicsWorld::new(PhysicsWorldConfig {
            fixed_dt: 1.0 / 120.0,
            ..PhysicsWorldConfig::default()
        });
        let mut scene = BounceTankSceneController::new(BounceTankSceneConfig {
            target_ball_count: 32,
            spawn_per_tick: 32,
            ..BounceTankSceneConfig::default()
        });
        let _ = scene.physics_tick(&mut world);
        let _ = world.step(world.config().fixed_dt);
        let before = scene.live_ball_count();
        assert!(before > 0);

        // Force one body far outside the prism.
        let escaped = scene.balls[0].body;
        let mut snapshot = world.capture_snapshot();
        if let Some(frame) = snapshot.bodies.iter_mut().find(|b| b.handle == escaped) {
            frame.position.x = 999.0;
        }
        let _ = world.restore_snapshot(&snapshot);

        let _ = scene.physics_tick(&mut world);
        assert!(scene.live_ball_count() <= scene.config().target_ball_count);
    }

    #[test]
    fn recycle_prefers_dominant_axis_and_avoids_corner_snap() {
        let mut world = PhysicsWorld::new(PhysicsWorldConfig {
            fixed_dt: 1.0 / 120.0,
            ..PhysicsWorldConfig::default()
        });
        let mut scene = BounceTankSceneController::new(BounceTankSceneConfig {
            target_ball_count: 16,
            spawn_per_tick: 16,
            container_half_extents: [8.0, 8.0, 8.0],
            ..BounceTankSceneConfig::default()
        });
        let _ = scene.physics_tick(&mut world);
        let _ = world.step(world.config().fixed_dt);
        let escaped = scene.balls[0].body;

        let mut snapshot = world.capture_snapshot();
        if let Some(frame) = snapshot.bodies.iter_mut().find(|b| b.handle == escaped) {
            // Escape in both X and Z, with X as dominant axis.
            frame.position.x = 100.0;
            frame.position.z = 20.0;
        }
        let _ = world.restore_snapshot(&snapshot);
        let _ = scene.physics_tick(&mut world);

        let body = world
            .body(escaped)
            .expect("escaped body should still exist");
        let radius = scene.balls[0].radius.max(0.02);
        let limit_x = (scene.config().container_half_extents[0] - radius).max(radius * 0.5);
        let limit_z = (scene.config().container_half_extents[2] - radius).max(radius * 0.5);
        // X is corrected back inside. Z must not be snapped to boundary in the same pass
        // (avoids immediate corner attractor behavior).
        assert!(body.position.x.abs() <= limit_x + 1e-3);
        assert!((body.position.z.abs() - limit_z).abs() > 0.25);
    }

    #[test]
    fn reconcile_after_step_refreshes_interpolation_for_recycled_bodies() {
        let mut world = PhysicsWorld::new(PhysicsWorldConfig {
            fixed_dt: 1.0 / 120.0,
            ..PhysicsWorldConfig::default()
        });
        let mut scene = BounceTankSceneController::new(BounceTankSceneConfig {
            target_ball_count: 24,
            spawn_per_tick: 24,
            container_half_extents: [8.0, 8.0, 8.0],
            ..BounceTankSceneConfig::default()
        });
        let _ = scene.physics_tick(&mut world);
        let _ = world.step(world.config().fixed_dt);
        let escaped = scene.balls[0].body;

        let mut snapshot = world.capture_snapshot();
        if let Some(frame) = snapshot.bodies.iter_mut().find(|b| b.handle == escaped) {
            frame.position.x = 512.0;
            frame.position.z = 64.0;
        }
        let _ = world.restore_snapshot(&snapshot);
        world.push_interpolation_snapshot();

        let recycled = scene.reconcile_after_step(&mut world);
        assert!(recycled > 0);

        let frame = scene.build_frame_instances(&world, Some(0.5));
        let inst = frame
            .opaque_3d
            .iter()
            .find(|instance| instance.instance_id == escaped.raw() as u64)
            .expect("escaped body instance should be present");
        let radius = scene.balls[0].radius.max(0.02);
        let limit_x = (scene.config().container_half_extents[0] - radius).max(radius * 0.5);
        assert!(inst.transform.translation[0].abs() <= limit_x + 0.2);
    }

    #[test]
    fn reconcile_after_step_hard_safety_recycles_with_virtual_barrier_disabled() {
        let mut world = PhysicsWorld::new(PhysicsWorldConfig {
            fixed_dt: 1.0 / 120.0,
            ..PhysicsWorldConfig::default()
        });
        let mut scene = BounceTankSceneController::new(BounceTankSceneConfig {
            target_ball_count: 24,
            spawn_per_tick: 24,
            container_half_extents: [8.0, 8.0, 8.0],
            virtual_barrier_enabled: false,
            ..BounceTankSceneConfig::default()
        });
        let _ = scene.physics_tick(&mut world);
        let _ = world.step(world.config().fixed_dt);
        let escaped = scene.balls[0].body;

        let mut snapshot = world.capture_snapshot();
        if let Some(frame) = snapshot.bodies.iter_mut().find(|b| b.handle == escaped) {
            frame.position.z = -512.0;
            frame.linear_velocity.z = -256.0;
        }
        let _ = world.restore_snapshot(&snapshot);
        world.push_interpolation_snapshot();

        let recycled = scene.reconcile_after_step(&mut world);
        assert!(recycled > 0);
        let body = world.body(escaped).expect("body should still exist");
        let radius = scene.balls[0].radius.max(0.02);
        let limit_z = (scene.config().container_half_extents[2] - radius).max(radius * 0.5);
        assert!(body.position.z.abs() <= limit_z + 0.2);
    }
}
