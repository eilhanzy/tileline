//! Runtime scene + sprite data model and a pre-alpha bounce showcase controller.
//!
//! This module keeps scene orchestration in `runtime/src` so examples can stay thin:
//! - generic 3D instance payloads (sphere/box primitives + PBR-ish material params)
//! - sprite instance payloads for HUD/overlay composition
//! - `BounceTankSceneController` that builds a transparent 3D container and progressively spawns
//!   thousands of colored balls using ParadoxPE bodies/colliders
//! - render/present vs physics tick-rate policy helper

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
    pub ball_radius_min: f32,
    pub ball_radius_max: f32,
    pub ball_restitution: f32,
    pub ball_friction: f32,
    pub wall_restitution: f32,
    pub wall_friction: f32,
    pub linear_damping: f32,
    pub initial_speed_min: f32,
    pub initial_speed_max: f32,
    pub scatter_interval_ticks: u64,
    pub scatter_strength: f32,
    pub seed: u64,
}

impl Default for BounceTankSceneConfig {
    fn default() -> Self {
        Self {
            target_ball_count: 12_000,
            spawn_per_tick: 120,
            container_half_extents: [24.0, 16.0, 24.0],
            wall_thickness: 0.40,
            ball_radius_min: 0.10,
            ball_radius_max: 0.24,
            ball_restitution: 0.74,
            ball_friction: 0.28,
            wall_restitution: 0.78,
            wall_friction: 0.20,
            linear_damping: 0.012,
            initial_speed_min: 0.35,
            initial_speed_max: 1.25,
            scatter_interval_ticks: 420,
            scatter_strength: 0.16,
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
    pub linear_damping: Option<f32>,
    pub gravity: Option<[f32; 3]>,
    pub contact_guard: Option<f32>,
    pub ball_restitution: Option<f32>,
    pub wall_restitution: Option<f32>,
    pub initial_speed_min: Option<f32>,
    pub initial_speed_max: Option<f32>,
    pub scatter_interval_ticks: Option<u64>,
    pub scatter_strength: Option<f32>,
    pub ball_mesh_slot: Option<u8>,
    pub container_mesh_slot: Option<u8>,
}

/// Result summary for one runtime patch application.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BounceTankPatchMetrics {
    pub config_updated: bool,
    pub dynamic_bodies_retuned: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct BallVisual {
    body: BodyHandle,
    radius: f32,
    color: [f32; 4],
    roughness: f32,
    metallic: f32,
}

/// Scene controller for a transparent 3D tank that progressively fills with bouncing balls.
///
/// Physics uses ParadoxPE dynamic sphere bodies and static AABB wall colliders.
pub struct BounceTankSceneController {
    config: BounceTankSceneConfig,
    rng_state: u64,
    walls: Vec<BodyHandle>,
    balls: Vec<BallVisual>,
    sprite_program: Option<TlspriteProgram>,
    ball_mesh_slot: Option<u8>,
    container_mesh_slot: Option<u8>,
    last_scatter_tick: u64,
}

impl BounceTankSceneController {
    pub fn new(config: BounceTankSceneConfig) -> Self {
        Self {
            rng_state: config.seed,
            config,
            walls: Vec::with_capacity(6),
            balls: Vec::new(),
            sprite_program: None,
            ball_mesh_slot: None,
            container_mesh_slot: None,
            last_scatter_tick: 0,
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
            }
        }
        if let Some(restitution) = patch.wall_restitution {
            let clamped = restitution.clamp(0.0, 1.25);
            if (self.config.wall_restitution - clamped).abs() > f32::EPSILON {
                self.config.wall_restitution = clamped;
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

        BounceTankPatchMetrics {
            config_updated: updated,
            dynamic_bodies_retuned: retuned,
        }
    }

    /// Ensure static physics walls exist and then spawn a progressive ball batch.
    pub fn physics_tick(&mut self, world: &mut PhysicsWorld) -> BounceTankTickMetrics {
        self.ensure_container_walls(world);
        self.cull_missing(world);
        let _ = self.recycle_escaped_balls(world);
        let spawned = self.spawn_batch(world);
        let scattered = self.maybe_scatter_burst(world);
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
        self.recycle_escaped_balls(world)
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
        // Default to edge-only prism rendering; full shell is opt-in via container mesh slot.
        if self.container_mesh_slot.is_some() {
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

        let mut scattered = 0usize;
        for i in (start..live).step_by(stride) {
            if scattered >= target {
                break;
            }
            let ball = self.balls[i];
            let Some(body) = world.body(ball.body) else {
                continue;
            };
            let mut dir = body.position;
            let jitter = 0.08 + strength * 0.18;
            dir.x += self.rand_range(-jitter, jitter);
            dir.y = dir.y.abs() + self.rand_range(0.10, 0.25 + 0.45 * strength);
            dir.z += self.rand_range(-jitter, jitter);
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

            let mut clamped = false;
            let mut position = body.position;
            let mut velocity = body.linear_velocity;
            let limit_x = (hx - radius).max(radius * 0.5);
            let limit_y = (hy - radius).max(radius * 0.5);
            let limit_z = (hz - radius).max(radius * 0.5);

            if position.x > limit_x {
                position.x = limit_x;
                velocity.x = -velocity.x.abs() * wall_bounce;
                clamped = true;
            } else if position.x < -limit_x {
                position.x = -limit_x;
                velocity.x = velocity.x.abs() * wall_bounce;
                clamped = true;
            }

            if position.y > limit_y {
                position.y = limit_y;
                velocity.y = -velocity.y.abs() * wall_bounce;
                clamped = true;
            } else if position.y < -limit_y {
                position.y = -limit_y;
                velocity.y = velocity.y.abs() * wall_bounce;
                clamped = true;
            }

            if position.z > limit_z {
                position.z = limit_z;
                velocity.z = -velocity.z.abs() * wall_bounce;
                clamped = true;
            } else if position.z < -limit_z {
                position.z = -limit_z;
                velocity.z = velocity.z.abs() * wall_bounce;
                clamped = true;
            }

            if clamped {
                let _ = world.set_position(handle, position);
                let _ = world.set_velocity(handle, velocity);
                recycled = recycled.saturating_add(1);
            }
            i += 1;
        }

        recycled
    }

    fn container_visual_instance(&self) -> SceneInstance3d {
        SceneInstance3d {
            instance_id: u64::MAX - 1,
            primitive: self
                .container_mesh_slot
                .map(|slot| ScenePrimitive3d::Mesh { slot })
                .unwrap_or(ScenePrimitive3d::Box),
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

    /// Adds 12 thin edge prisms so the tank silhouette remains readable from every camera angle.
    fn append_container_edge_instances(&self, out: &mut Vec<SceneInstance3d>) {
        let hx = self.config.container_half_extents[0] * 2.0;
        let hy = self.config.container_half_extents[1] * 2.0;
        let hz = self.config.container_half_extents[2] * 2.0;
        let edge = self.config.wall_thickness.max(0.03) * 0.30;

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
                primitive: ScenePrimitive3d::Box,
                transform: SceneTransform3d {
                    translation,
                    rotation_xyzw: [0.0, 0.0, 0.0, 1.0],
                    scale,
                },
                material: SceneMaterial {
                    base_color_rgba: [0.90, 0.96, 1.0, 0.72],
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
                ..BounceTankRuntimePatch::default()
            },
        );
        let cfg = scene.config();
        assert_eq!(cfg.spawn_per_tick, 1);
        assert!(cfg.linear_damping <= 0.95);
        assert!(cfg.initial_speed_max >= cfg.initial_speed_min);
        assert!(world.gravity().y < -9.9);
        assert!(world.solver().config().hard_position_projection_strength > 1.0);
        assert!(metrics.config_updated);
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
}
