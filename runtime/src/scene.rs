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

use crate::tlsprite::{TlspriteFrameContext, TlspriteProgram};

/// Lightweight primitive kind for runtime-side scene batching.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScenePrimitive3d {
    Sphere,
    Box,
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
            min_tick_hz: 30.0,
            max_tick_hz: 300.0,
            ticks_per_render_frame: 2.0,
            default_tick_hz: 120.0,
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
    pub seed: u64,
}

impl Default for BounceTankSceneConfig {
    fn default() -> Self {
        Self {
            target_ball_count: 12_000,
            spawn_per_tick: 240,
            container_half_extents: [24.0, 16.0, 24.0],
            wall_thickness: 0.40,
            ball_radius_min: 0.10,
            ball_radius_max: 0.24,
            ball_restitution: 0.84,
            ball_friction: 0.28,
            wall_restitution: 0.92,
            wall_friction: 0.20,
            linear_damping: 0.006,
            initial_speed_min: 0.2,
            initial_speed_max: 1.8,
            seed: 0x5EED_C0DE_u64,
        }
    }
}

/// Per-physics-tick output from bounce scene orchestration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BounceTankTickMetrics {
    pub spawned_this_tick: usize,
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
    pub ball_restitution: Option<f32>,
    pub wall_restitution: Option<f32>,
    pub initial_speed_min: Option<f32>,
    pub initial_speed_max: Option<f32>,
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
}

impl BounceTankSceneController {
    pub fn new(config: BounceTankSceneConfig) -> Self {
        Self {
            rng_state: config.seed,
            config,
            walls: Vec::with_capacity(6),
            balls: Vec::new(),
            sprite_program: None,
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
        if self.config.initial_speed_max < self.config.initial_speed_min {
            self.config.initial_speed_max = self.config.initial_speed_min;
            updated = true;
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
        let spawned = self.spawn_batch(world);
        BounceTankTickMetrics {
            spawned_this_tick: spawned,
            live_balls: self.balls.len(),
            target_balls: self.config.target_ball_count,
            fully_spawned: self.balls.len() >= self.config.target_ball_count,
        }
    }

    /// Build renderer-facing scene payloads using optional interpolation alpha.
    pub fn build_frame_instances(
        &self,
        world: &PhysicsWorld,
        interpolation_alpha: Option<f32>,
    ) -> SceneFrameInstances {
        let mut frame = SceneFrameInstances::default();
        frame.transparent_3d.push(self.container_visual_instance());

        let alpha = interpolation_alpha.unwrap_or(1.0).clamp(0.0, 1.0);
        for ball in &self.balls {
            let pose = world
                .interpolate_body_pose(ball.body, alpha)
                .map(|interp| (interp.position, interp.rotation))
                .or_else(|| {
                    world
                        .body(ball.body)
                        .map(|body| (body.position, body.rotation))
                });
            let Some((position, rotation)) = pose else {
                continue;
            };

            let q = rotation.quaternion();
            frame.opaque_3d.push(SceneInstance3d {
                instance_id: ball.body.raw() as u64,
                primitive: ScenePrimitive3d::Sphere,
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
            });
        }

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
        let t = self.config.wall_thickness.max(0.02);

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
                user_tag: 0,
            });
            self.walls.push(body);
        }
    }

    fn spawn_batch(&mut self, world: &mut PhysicsWorld) -> usize {
        if self.balls.len() >= self.config.target_ball_count {
            return 0;
        }
        let remaining = self
            .config
            .target_ball_count
            .saturating_sub(self.balls.len());
        let to_spawn = remaining.min(self.config.spawn_per_tick.max(1));
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
            let velocity = Vector3::new(xy * theta.cos(), -speed.abs(), xy * theta.sin()) * speed;

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
                    user_tag: 0,
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
                base_color_rgba: [0.58, 0.72, 0.98, 0.12],
                roughness: 0.04,
                metallic: 0.0,
                emissive_rgb: [0.0, 0.0, 0.0],
                shading: ShadingModel::LitPbr,
            },
            casts_shadow: false,
            receives_shadow: true,
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
        assert_eq!(frame.transparent_3d.len(), 1);
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
                initial_speed_min: Some(3.0),
                initial_speed_max: Some(1.0),
                ..BounceTankRuntimePatch::default()
            },
        );
        let cfg = scene.config();
        assert_eq!(cfg.spawn_per_tick, 1);
        assert!(cfg.linear_damping <= 0.95);
        assert!(cfg.initial_speed_max >= cfg.initial_speed_min);
        assert!(metrics.config_updated);
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
}
