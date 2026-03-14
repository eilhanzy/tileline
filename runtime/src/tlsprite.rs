//! `.tlsprite` parser and runtime sprite emission helpers.
//!
//! The format is intentionally small and text-based so toolchains can emit it easily.
//! Parsing is memory-resident (`&str` input), and the output can be reused every frame.
//!
//! Minimal format:
//! ```text
//! tlsprite_v1
//! [progress_bar]
//! sprite_id = 1
//! texture_slot = 0
//! layer = 100
//! position = -0.86, 0.90, 0.0
//! size = 0.40, 0.035
//! rotation_rad = 0.0
//! color = 0.10, 0.84, 0.62, 0.92
//! scale_axis = x
//! scale_source = spawn_progress
//! scale_min = 0.02
//! scale_max = 1.0
//! ```

use std::{
    collections::hash_map::DefaultHasher,
    fs,
    hash::{Hash, Hasher},
    path::{Path, PathBuf},
};

use crate::scene::SpriteInstance;

/// Input bindings consumed while emitting runtime sprite instances.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TlspriteFrameContext {
    pub spawn_progress: f32,
    pub live_balls: usize,
    pub target_balls: usize,
}

impl Default for TlspriteFrameContext {
    fn default() -> Self {
        Self {
            spawn_progress: 1.0,
            live_balls: 0,
            target_balls: 0,
        }
    }
}

/// Runtime diagnostics produced by `.tlsprite` compilation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TlspriteDiagnosticLevel {
    Warning,
    Error,
}

/// One parser/validation diagnostic line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TlspriteDiagnostic {
    pub level: TlspriteDiagnosticLevel,
    pub line: usize,
    pub message: String,
}

/// Soft compile outcome for `.tlsprite`.
#[derive(Debug, Clone)]
pub struct TlspriteCompileOutcome {
    pub program: Option<TlspriteProgram>,
    pub diagnostics: Vec<TlspriteDiagnostic>,
}

impl TlspriteCompileOutcome {
    pub fn has_errors(&self) -> bool {
        self.diagnostics
            .iter()
            .any(|d| d.level == TlspriteDiagnosticLevel::Error)
    }
}

/// Which sprite axis should be dynamically scaled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TlspriteScaleAxis {
    X,
    Y,
}

/// Which runtime signal drives dynamic scaling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TlspriteScaleSource {
    SpawnProgress,
    SpawnRemaining,
    LiveBallRatio,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct TlspriteDynamicScale {
    axis: TlspriteScaleAxis,
    source: TlspriteScaleSource,
    min_factor: f32,
    max_factor: f32,
}

/// One compiled sprite definition.
#[derive(Debug, Clone, PartialEq)]
pub struct TlspriteSpriteDef {
    pub name: String,
    pub sprite: SpriteInstance,
    dynamic_scale: Option<TlspriteDynamicScale>,
}

/// Compiled `.tlsprite` sprite set.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TlspriteProgram {
    sprites: Vec<TlspriteSpriteDef>,
}

impl TlspriteProgram {
    pub fn sprites(&self) -> &[TlspriteSpriteDef] {
        &self.sprites
    }

    /// Emit frame-local sprite instances into `out`.
    pub fn emit_instances(&self, ctx: TlspriteFrameContext, out: &mut Vec<SpriteInstance>) {
        out.reserve(self.sprites.len());
        let spawn_progress = ctx.spawn_progress.clamp(0.0, 1.0);
        let live_ratio = if ctx.target_balls == 0 {
            1.0
        } else {
            (ctx.live_balls as f32 / ctx.target_balls as f32).clamp(0.0, 1.0)
        };

        for def in &self.sprites {
            let mut instance = def.sprite.clone();
            if let Some(scale) = def.dynamic_scale {
                let raw = match scale.source {
                    TlspriteScaleSource::SpawnProgress => spawn_progress,
                    TlspriteScaleSource::SpawnRemaining => 1.0 - spawn_progress,
                    TlspriteScaleSource::LiveBallRatio => live_ratio,
                };
                let factor = raw.clamp(scale.min_factor, scale.max_factor);
                match scale.axis {
                    TlspriteScaleAxis::X => {
                        instance.size[0] = (instance.size[0] * factor).max(0.0001);
                    }
                    TlspriteScaleAxis::Y => {
                        instance.size[1] = (instance.size[1] * factor).max(0.0001);
                    }
                }
            }
            out.push(instance);
        }
    }
}

/// Hot-reload behavior knobs for `.tlsprite` disk sources.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TlspriteHotReloadConfig {
    /// Keep the last valid program if a new edit fails to compile.
    pub keep_last_good_program: bool,
}

impl Default for TlspriteHotReloadConfig {
    fn default() -> Self {
        Self {
            keep_last_good_program: true,
        }
    }
}

/// Result of one hot-reload polling attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TlspriteHotReloadEvent {
    Unchanged,
    Applied {
        sprite_count: usize,
        warning_count: usize,
    },
    Rejected {
        error_count: usize,
        warning_count: usize,
        kept_last_program: bool,
    },
    SourceError {
        message: String,
    },
}

/// Simple hash-based `.tlsprite` file hot-reloader.
#[derive(Debug, Clone)]
pub struct TlspriteHotReloader {
    path: PathBuf,
    config: TlspriteHotReloadConfig,
    source_hash: Option<u64>,
    program: Option<TlspriteProgram>,
    diagnostics: Vec<TlspriteDiagnostic>,
}

impl TlspriteHotReloader {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self::with_config(path, TlspriteHotReloadConfig::default())
    }

    pub fn with_config(path: impl Into<PathBuf>, config: TlspriteHotReloadConfig) -> Self {
        Self {
            path: path.into(),
            config,
            source_hash: None,
            program: None,
            diagnostics: Vec::new(),
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn program(&self) -> Option<&TlspriteProgram> {
        self.program.as_ref()
    }

    pub fn diagnostics(&self) -> &[TlspriteDiagnostic] {
        &self.diagnostics
    }

    /// Read source and recompile when file contents have changed.
    pub fn reload_if_changed(&mut self) -> TlspriteHotReloadEvent {
        let bytes = match fs::read(&self.path) {
            Ok(bytes) => bytes,
            Err(err) => {
                return TlspriteHotReloadEvent::SourceError {
                    message: format!("failed to read '{}': {err}", self.path.display()),
                };
            }
        };

        let hash = hash_bytes(&bytes);
        if self.source_hash == Some(hash) {
            return TlspriteHotReloadEvent::Unchanged;
        }
        self.source_hash = Some(hash);

        let source = match String::from_utf8(bytes) {
            Ok(text) => text,
            Err(err) => {
                return TlspriteHotReloadEvent::SourceError {
                    message: format!("source '{}' is not valid UTF-8: {err}", self.path.display()),
                };
            }
        };
        let outcome = compile_tlsprite(&source);
        let warning_count = outcome
            .diagnostics
            .iter()
            .filter(|d| d.level == TlspriteDiagnosticLevel::Warning)
            .count();
        let error_count = outcome
            .diagnostics
            .iter()
            .filter(|d| d.level == TlspriteDiagnosticLevel::Error)
            .count();
        self.diagnostics = outcome.diagnostics;

        if let Some(program) = outcome.program {
            let sprite_count = program.sprites().len();
            self.program = Some(program);
            return TlspriteHotReloadEvent::Applied {
                sprite_count,
                warning_count,
            };
        }

        if !self.config.keep_last_good_program {
            self.program = None;
        }
        TlspriteHotReloadEvent::Rejected {
            error_count,
            warning_count,
            kept_last_program: self.program.is_some(),
        }
    }
}

/// Compile `.tlsprite` source from memory.
pub fn compile_tlsprite(source: &str) -> TlspriteCompileOutcome {
    let mut diagnostics = Vec::new();
    let mut sprites = Vec::new();
    let mut header_checked = false;
    let mut pending: Option<PendingSprite> = None;

    for (idx, raw_line) in source.lines().enumerate() {
        let line_no = idx + 1;
        let line = strip_comments(raw_line).trim();
        if line.is_empty() {
            continue;
        }

        if !header_checked {
            header_checked = true;
            if line == "tlsprite_v1" {
                continue;
            }
            diagnostics.push(TlspriteDiagnostic {
                level: TlspriteDiagnosticLevel::Warning,
                line: line_no,
                message: "missing 'tlsprite_v1' header, continuing in compatibility mode"
                    .to_string(),
            });
        }

        if let Some(section) = parse_section_name(line) {
            if let Some(curr) = pending.take() {
                if let Some(def) = curr.finish(&mut diagnostics) {
                    sprites.push(def);
                }
            }
            pending = Some(PendingSprite::new(section.to_string(), line_no));
            continue;
        }

        let Some((key, value)) = line.split_once('=') else {
            diagnostics.push(TlspriteDiagnostic {
                level: TlspriteDiagnosticLevel::Error,
                line: line_no,
                message: "expected 'key = value' entry".to_string(),
            });
            continue;
        };
        let key = key.trim();
        let value = value.trim();

        let Some(curr) = pending.as_mut() else {
            diagnostics.push(TlspriteDiagnostic {
                level: TlspriteDiagnosticLevel::Error,
                line: line_no,
                message: "key-value entry outside [section]".to_string(),
            });
            continue;
        };
        curr.apply(key, value, line_no, &mut diagnostics);
    }

    if let Some(curr) = pending.take() {
        if let Some(def) = curr.finish(&mut diagnostics) {
            sprites.push(def);
        }
    }

    if sprites.is_empty() {
        diagnostics.push(TlspriteDiagnostic {
            level: TlspriteDiagnosticLevel::Error,
            line: 0,
            message: "no sprite sections were produced".to_string(),
        });
    }

    let has_errors = diagnostics
        .iter()
        .any(|d| d.level == TlspriteDiagnosticLevel::Error);
    TlspriteCompileOutcome {
        program: (!has_errors).then_some(TlspriteProgram { sprites }),
        diagnostics,
    }
}

fn hash_bytes(bytes: &[u8]) -> u64 {
    let mut hasher = DefaultHasher::new();
    bytes.hash(&mut hasher);
    hasher.finish()
}

fn strip_comments(line: &str) -> &str {
    let mut cut = line.len();
    if let Some(i) = line.find('#') {
        cut = cut.min(i);
    }
    if let Some(i) = line.find("//") {
        cut = cut.min(i);
    }
    &line[..cut]
}

fn parse_section_name(line: &str) -> Option<&str> {
    let trimmed = line.trim();
    if trimmed.len() >= 3 && trimmed.starts_with('[') && trimmed.ends_with(']') {
        let inner = trimmed[1..trimmed.len() - 1].trim();
        if !inner.is_empty() {
            return Some(inner);
        }
    }
    None
}

#[derive(Debug, Clone)]
struct PendingSprite {
    name: String,
    line_started: usize,
    sprite_id: Option<u64>,
    texture_slot: Option<u16>,
    layer: Option<i16>,
    position: Option<[f32; 3]>,
    size: Option<[f32; 2]>,
    rotation_rad: Option<f32>,
    color: Option<[f32; 4]>,
    scale_axis: Option<TlspriteScaleAxis>,
    scale_source: Option<TlspriteScaleSource>,
    scale_min: Option<f32>,
    scale_max: Option<f32>,
}

impl PendingSprite {
    fn new(name: String, line_started: usize) -> Self {
        Self {
            name,
            line_started,
            sprite_id: None,
            texture_slot: None,
            layer: None,
            position: None,
            size: None,
            rotation_rad: None,
            color: None,
            scale_axis: None,
            scale_source: None,
            scale_min: None,
            scale_max: None,
        }
    }

    fn apply(
        &mut self,
        key: &str,
        value: &str,
        line: usize,
        diagnostics: &mut Vec<TlspriteDiagnostic>,
    ) {
        match key {
            "sprite_id" => {
                self.sprite_id = parse_scalar::<u64>(value, line, key, diagnostics);
            }
            "texture_slot" => {
                self.texture_slot = parse_scalar::<u16>(value, line, key, diagnostics);
            }
            "layer" => {
                self.layer = parse_scalar::<i16>(value, line, key, diagnostics);
            }
            "position" => {
                self.position = parse_vec3(value, line, key, diagnostics);
            }
            "size" => {
                self.size = parse_vec2(value, line, key, diagnostics);
            }
            "rotation_rad" => {
                self.rotation_rad = parse_scalar::<f32>(value, line, key, diagnostics);
            }
            "color" => {
                self.color = parse_vec4(value, line, key, diagnostics);
            }
            "scale_axis" => {
                self.scale_axis = parse_scale_axis(value, line, diagnostics);
            }
            "scale_source" => {
                self.scale_source = parse_scale_source(value, line, diagnostics);
            }
            "scale_min" => {
                self.scale_min = parse_scalar::<f32>(value, line, key, diagnostics);
            }
            "scale_max" => {
                self.scale_max = parse_scalar::<f32>(value, line, key, diagnostics);
            }
            _ => diagnostics.push(TlspriteDiagnostic {
                level: TlspriteDiagnosticLevel::Warning,
                line,
                message: format!("unknown key '{key}' ignored"),
            }),
        }
    }

    fn finish(self, diagnostics: &mut Vec<TlspriteDiagnostic>) -> Option<TlspriteSpriteDef> {
        let Some(sprite_id) = self.sprite_id else {
            diagnostics.push(TlspriteDiagnostic {
                level: TlspriteDiagnosticLevel::Error,
                line: self.line_started,
                message: format!(
                    "section '{}' is missing required key 'sprite_id'",
                    self.name
                ),
            });
            return None;
        };

        let mut color = self.color.unwrap_or([1.0, 1.0, 1.0, 1.0]);
        for c in &mut color {
            *c = c.clamp(0.0, 1.0);
        }
        let mut size = self.size.unwrap_or([0.1, 0.1]);
        size[0] = size[0].max(0.001);
        size[1] = size[1].max(0.001);

        let dynamic_scale = if let Some(axis) = self.scale_axis {
            let mut min_factor = self.scale_min.unwrap_or(0.0).clamp(0.0, 4.0);
            let mut max_factor = self.scale_max.unwrap_or(1.0).clamp(0.0, 4.0);
            if max_factor < min_factor {
                std::mem::swap(&mut min_factor, &mut max_factor);
            }
            Some(TlspriteDynamicScale {
                axis,
                source: self
                    .scale_source
                    .unwrap_or(TlspriteScaleSource::SpawnProgress),
                min_factor,
                max_factor,
            })
        } else if self.scale_source.is_some()
            || self.scale_min.is_some()
            || self.scale_max.is_some()
        {
            diagnostics.push(TlspriteDiagnostic {
                level: TlspriteDiagnosticLevel::Warning,
                line: self.line_started,
                message: format!(
                    "section '{}' defines scale_* keys but no 'scale_axis'; dynamic scaling disabled",
                    self.name
                ),
            });
            None
        } else {
            None
        };

        Some(TlspriteSpriteDef {
            name: self.name,
            sprite: SpriteInstance {
                sprite_id,
                position: self.position.unwrap_or([0.0, 0.0, 0.0]),
                size,
                rotation_rad: self.rotation_rad.unwrap_or(0.0),
                color_rgba: color,
                texture_slot: self.texture_slot.unwrap_or(0),
                layer: self.layer.unwrap_or(0),
            },
            dynamic_scale,
        })
    }
}

fn parse_scale_axis(
    value: &str,
    line: usize,
    diagnostics: &mut Vec<TlspriteDiagnostic>,
) -> Option<TlspriteScaleAxis> {
    match value.trim() {
        "x" | "X" => Some(TlspriteScaleAxis::X),
        "y" | "Y" => Some(TlspriteScaleAxis::Y),
        other => {
            diagnostics.push(TlspriteDiagnostic {
                level: TlspriteDiagnosticLevel::Error,
                line,
                message: format!("invalid scale_axis '{other}', expected 'x' or 'y'"),
            });
            None
        }
    }
}

fn parse_scale_source(
    value: &str,
    line: usize,
    diagnostics: &mut Vec<TlspriteDiagnostic>,
) -> Option<TlspriteScaleSource> {
    match value.trim() {
        "spawn_progress" => Some(TlspriteScaleSource::SpawnProgress),
        "spawn_remaining" => Some(TlspriteScaleSource::SpawnRemaining),
        "live_ball_ratio" => Some(TlspriteScaleSource::LiveBallRatio),
        other => {
            diagnostics.push(TlspriteDiagnostic {
                level: TlspriteDiagnosticLevel::Error,
                line,
                message: format!(
                    "invalid scale_source '{other}', expected 'spawn_progress', 'spawn_remaining', or 'live_ball_ratio'"
                ),
            });
            None
        }
    }
}

fn parse_scalar<T: std::str::FromStr>(
    raw: &str,
    line: usize,
    key: &str,
    diagnostics: &mut Vec<TlspriteDiagnostic>,
) -> Option<T> {
    match raw.trim().parse::<T>() {
        Ok(v) => Some(v),
        Err(_) => {
            diagnostics.push(TlspriteDiagnostic {
                level: TlspriteDiagnosticLevel::Error,
                line,
                message: format!("failed to parse key '{key}'"),
            });
            None
        }
    }
}

fn parse_vec2(
    raw: &str,
    line: usize,
    key: &str,
    diagnostics: &mut Vec<TlspriteDiagnostic>,
) -> Option<[f32; 2]> {
    parse_f32_components(raw, 2, line, key, diagnostics).map(|v| [v[0], v[1]])
}

fn parse_vec3(
    raw: &str,
    line: usize,
    key: &str,
    diagnostics: &mut Vec<TlspriteDiagnostic>,
) -> Option<[f32; 3]> {
    parse_f32_components(raw, 3, line, key, diagnostics).map(|v| [v[0], v[1], v[2]])
}

fn parse_vec4(
    raw: &str,
    line: usize,
    key: &str,
    diagnostics: &mut Vec<TlspriteDiagnostic>,
) -> Option<[f32; 4]> {
    parse_f32_components(raw, 4, line, key, diagnostics).map(|v| [v[0], v[1], v[2], v[3]])
}

fn parse_f32_components(
    raw: &str,
    expected: usize,
    line: usize,
    key: &str,
    diagnostics: &mut Vec<TlspriteDiagnostic>,
) -> Option<Vec<f32>> {
    let mut parts = Vec::with_capacity(expected);
    for chunk in raw.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        match chunk.parse::<f32>() {
            Ok(v) => parts.push(v),
            Err(_) => {
                diagnostics.push(TlspriteDiagnostic {
                    level: TlspriteDiagnosticLevel::Error,
                    line,
                    message: format!("failed to parse float component in key '{key}'"),
                });
                return None;
            }
        }
    }
    if parts.len() != expected {
        diagnostics.push(TlspriteDiagnostic {
            level: TlspriteDiagnosticLevel::Error,
            line,
            message: format!("key '{key}' expects {expected} comma-separated float values"),
        });
        return None;
    }
    Some(parts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        fs,
        path::PathBuf,
        time::{SystemTime, UNIX_EPOCH},
    };

    #[test]
    fn parses_tlsprite_and_emits_scaled_sprite() {
        let src = concat!(
            "tlsprite_v1\n",
            "[progress]\n",
            "sprite_id = 7\n",
            "texture_slot = 2\n",
            "layer = 100\n",
            "position = -0.5, 0.8, 0.0\n",
            "size = 0.4, 0.05\n",
            "rotation_rad = 0.0\n",
            "color = 0.2, 0.7, 0.9, 1.0\n",
            "scale_axis = x\n",
            "scale_source = spawn_progress\n",
            "scale_min = 0.1\n",
            "scale_max = 1.0\n",
        );
        let out = compile_tlsprite(src);
        assert!(!out.has_errors());
        let program = out.program.as_ref().expect("program");
        assert_eq!(program.sprites().len(), 1);

        let mut instances = Vec::new();
        program.emit_instances(
            TlspriteFrameContext {
                spawn_progress: 0.5,
                live_balls: 500,
                target_balls: 1_000,
            },
            &mut instances,
        );
        assert_eq!(instances.len(), 1);
        assert!((instances[0].size[0] - 0.2).abs() < 1e-6);
    }

    #[test]
    fn reports_missing_required_sprite_id() {
        let src = concat!("tlsprite_v1\n", "[broken]\n", "size = 0.2, 0.1\n",);
        let out = compile_tlsprite(src);
        assert!(out.has_errors());
        assert!(out.program.is_none());
    }

    #[test]
    fn warns_when_header_is_missing() {
        let src = concat!("[ok]\n", "sprite_id = 1\n");
        let out = compile_tlsprite(src);
        assert!(out.program.is_some());
        assert!(out
            .diagnostics
            .iter()
            .any(|d| d.level == TlspriteDiagnosticLevel::Warning));
    }

    #[test]
    fn hot_reloader_applies_and_tracks_changes() {
        let path = temp_tlsprite_path("reload");
        fs::write(&path, concat!("tlsprite_v1\n", "[s]\n", "sprite_id = 1\n",))
            .expect("write initial");

        let mut loader = TlspriteHotReloader::new(&path);
        let first = loader.reload_if_changed();
        assert!(matches!(first, TlspriteHotReloadEvent::Applied { .. }));
        assert_eq!(loader.program().map(|p| p.sprites().len()), Some(1));
        assert!(matches!(
            loader.reload_if_changed(),
            TlspriteHotReloadEvent::Unchanged
        ));

        fs::write(
            &path,
            concat!("tlsprite_v1\n", "[s2]\n", "sprite_id = 2\n",),
        )
        .expect("write changed");
        let changed = loader.reload_if_changed();
        assert!(matches!(changed, TlspriteHotReloadEvent::Applied { .. }));
        assert_eq!(
            loader
                .program()
                .and_then(|p| p.sprites().first())
                .map(|s| s.sprite.sprite_id),
            Some(2)
        );

        let _ = fs::remove_file(path);
    }

    #[test]
    fn hot_reloader_keeps_last_good_program_on_compile_error() {
        let path = temp_tlsprite_path("fallback");
        fs::write(
            &path,
            concat!("tlsprite_v1\n", "[ok]\n", "sprite_id = 42\n",),
        )
        .expect("write initial");

        let mut loader = TlspriteHotReloader::new(&path);
        let _ = loader.reload_if_changed();
        assert!(loader.program().is_some());

        fs::write(
            &path,
            concat!("tlsprite_v1\n", "[broken]\n", "size = 0.2, 0.1\n",),
        )
        .expect("write broken");
        let event = loader.reload_if_changed();
        assert!(matches!(
            event,
            TlspriteHotReloadEvent::Rejected {
                kept_last_program: true,
                ..
            }
        ));
        assert_eq!(
            loader
                .program()
                .and_then(|p| p.sprites().first())
                .map(|s| s.sprite.sprite_id),
            Some(42)
        );

        let _ = fs::remove_file(path);
    }

    fn temp_tlsprite_path(tag: &str) -> PathBuf {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!(
            "tileline_tlsprite_{tag}_{}_{}.tlsprite",
            std::process::id(),
            ts
        ))
    }
}
