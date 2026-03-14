//! Telemetry HUD sprite overlay composer.
//!
//! This module appends deterministic HUD sprites directly onto `SceneFrameInstances::sprites`.
//! It is renderer-agnostic and can be consumed by either GMS/MGS draw paths.

use crate::scene::SpriteInstance;

/// Input telemetry for HUD composition.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TelemetryHudSample {
    pub fps: f32,
    pub frame_time_ms: f32,
    pub physics_substeps: u32,
    pub live_balls: usize,
    pub draw_calls: usize,
}

/// HUD configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TelemetryHudConfig {
    pub base_sprite_id: u64,
    pub base_layer: i16,
    pub anchor_top_left: [f32; 2],
    pub width: f32,
    pub row_height: f32,
    pub target_fps: f32,
    pub frame_budget_ms: f32,
    pub reference_balls: usize,
    pub reference_draw_calls: usize,
}

impl Default for TelemetryHudConfig {
    fn default() -> Self {
        Self {
            base_sprite_id: 50_000,
            base_layer: 280,
            anchor_top_left: [-0.97, 0.95],
            width: 0.44,
            row_height: 0.028,
            target_fps: 60.0,
            frame_budget_ms: 16.67,
            reference_balls: 12_000,
            reference_draw_calls: 10_000,
        }
    }
}

/// Composition summary for one frame.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TelemetryHudMetrics {
    pub appended_sprites: usize,
}

/// Reusable HUD composer.
#[derive(Debug, Clone, Copy)]
pub struct TelemetryHudComposer {
    config: TelemetryHudConfig,
}

impl TelemetryHudComposer {
    pub fn new(config: TelemetryHudConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> TelemetryHudConfig {
        self.config
    }

    /// Append HUD panel + meters into scene sprite list.
    pub fn append_to_sprites(
        &self,
        sample: TelemetryHudSample,
        sprites: &mut Vec<SpriteInstance>,
    ) -> TelemetryHudMetrics {
        let start_len = sprites.len();
        let cfg = self.config;

        // Panel backdrop.
        sprites.push(SpriteInstance {
            sprite_id: cfg.base_sprite_id,
            position: [
                cfg.anchor_top_left[0] + cfg.width * 0.5,
                cfg.anchor_top_left[1] - 0.075,
                0.0,
            ],
            size: [cfg.width, 0.17],
            rotation_rad: 0.0,
            color_rgba: [0.05, 0.07, 0.10, 0.70],
            texture_slot: 0,
            layer: cfg.base_layer,
        });

        // FPS bar (higher is better).
        let fps_ratio = safe_ratio(sample.fps, cfg.target_fps);
        self.push_meter(
            sprites,
            1,
            1,
            0.0,
            fps_ratio,
            [0.12, 0.84, 0.52, 0.92],
            [0.16, 0.20, 0.16, 0.80],
        );

        // Frame-time budget bar (lower frame_time is better -> invert ratio).
        let time_ok = 1.0 - safe_ratio(sample.frame_time_ms, cfg.frame_budget_ms);
        self.push_meter(
            sprites,
            2,
            2,
            1.0,
            time_ok,
            [0.18, 0.66, 0.92, 0.92],
            [0.16, 0.18, 0.22, 0.80],
        );

        // Physics substep pressure.
        let substep_ratio = safe_ratio(sample.physics_substeps as f32, 6.0);
        self.push_meter(
            sprites,
            3,
            3,
            2.0,
            1.0 - substep_ratio,
            [0.96, 0.72, 0.24, 0.92],
            [0.20, 0.19, 0.14, 0.80],
        );

        // Scene load pressure from balls + draw calls.
        let load = (safe_ratio(sample.live_balls as f32, cfg.reference_balls as f32)
            + safe_ratio(sample.draw_calls as f32, cfg.reference_draw_calls as f32))
            * 0.5;
        self.push_meter(
            sprites,
            4,
            4,
            3.0,
            1.0 - load,
            [0.92, 0.36, 0.28, 0.92],
            [0.25, 0.14, 0.14, 0.80],
        );

        TelemetryHudMetrics {
            appended_sprites: sprites.len().saturating_sub(start_len),
        }
    }

    fn push_meter(
        &self,
        sprites: &mut Vec<SpriteInstance>,
        track_id_offset: u64,
        fill_id_offset: u64,
        row_index: f32,
        fill_ratio: f32,
        fill_color: [f32; 4],
        track_color: [f32; 4],
    ) {
        let cfg = self.config;
        let y = cfg.anchor_top_left[1] - (0.04 + row_index * cfg.row_height);
        let x = cfg.anchor_top_left[0] + 0.03;
        let meter_width = cfg.width - 0.06;
        let fill = fill_ratio.clamp(0.0, 1.0);

        sprites.push(SpriteInstance {
            sprite_id: cfg.base_sprite_id + track_id_offset,
            position: [x + meter_width * 0.5, y, 0.0],
            size: [meter_width, 0.016],
            rotation_rad: 0.0,
            color_rgba: track_color,
            texture_slot: 0,
            layer: cfg.base_layer + 1,
        });

        sprites.push(SpriteInstance {
            sprite_id: cfg.base_sprite_id + fill_id_offset + 100,
            position: [x + meter_width * 0.5 * fill, y, 0.0],
            size: [meter_width * fill.max(0.02), 0.013],
            rotation_rad: 0.0,
            color_rgba: fill_color,
            texture_slot: 0,
            layer: cfg.base_layer + 2,
        });
    }
}

fn safe_ratio(value: f32, reference: f32) -> f32 {
    if reference <= f32::EPSILON {
        0.0
    } else {
        (value / reference).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hud_appends_expected_sprite_count() {
        let mut sprites = Vec::new();
        let hud = TelemetryHudComposer::new(Default::default());
        let m = hud.append_to_sprites(
            TelemetryHudSample {
                fps: 58.0,
                frame_time_ms: 17.2,
                physics_substeps: 2,
                live_balls: 4_000,
                draw_calls: 1_200,
            },
            &mut sprites,
        );
        assert_eq!(m.appended_sprites, 9);
        assert_eq!(sprites.len(), 9);
    }

    #[test]
    fn hud_fill_ratios_are_clamped() {
        let mut sprites = Vec::new();
        let hud = TelemetryHudComposer::new(Default::default());
        let _ = hud.append_to_sprites(
            TelemetryHudSample {
                fps: 10_000.0,
                frame_time_ms: 0.01,
                physics_substeps: 100,
                live_balls: 1_000_000,
                draw_calls: 1_000_000,
            },
            &mut sprites,
        );
        assert!(sprites.iter().all(|s| s.size[0].is_finite()));
    }
}
