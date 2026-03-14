//! `.tljoint` scene manifest loader for multi-script and multi-sprite composition.
//!
//! Format (`tljoint_v1`) is section based:
//! ```text
//! tljoint_v1
//! [scene.showcase]
//! tlscripts = docs/demos/tlapp/bounce_showcase.tlscript
//! tlsprites = docs/demos/tlapp/bounce_hud.tlsprite
//! ```
//!
//! Supported keys per scene section:
//! - `tlscript` / `script` (single value)
//! - `tlscripts` / `scripts` (comma-separated values)
//! - `tlsprite` / `sprite` (single value)
//! - `tlsprites` / `sprites` (comma-separated values)

use std::fs;
use std::path::{Path, PathBuf};

use crate::scene::BounceTankRuntimePatch;
use crate::tlscript_showcase::{
    compile_tlscript_showcase, TlscriptShowcaseConfig, TlscriptShowcaseControlInput,
    TlscriptShowcaseFrameInput, TlscriptShowcaseFrameOutput, TlscriptShowcaseProgram,
};
use crate::tlsprite::{compile_tlsprite, TlspriteDiagnosticLevel, TlspriteProgram};

/// Diagnostic level for `.tljoint` parsing/compilation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TljointDiagnosticLevel {
    Warning,
    Error,
}

/// One `.tljoint` diagnostic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TljointDiagnostic {
    pub level: TljointDiagnosticLevel,
    pub line: usize,
    pub message: String,
}

/// One scene entry resolved from `.tljoint`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TljointSceneBinding {
    pub scene: String,
    pub tlscripts: Vec<String>,
    pub tlsprites: Vec<String>,
}

/// Parsed `.tljoint` manifest.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TljointManifest {
    pub scenes: Vec<TljointSceneBinding>,
}

impl TljointManifest {
    /// Returns scene binding by exact scene name.
    pub fn scene(&self, name: &str) -> Option<&TljointSceneBinding> {
        self.scenes.iter().find(|scene| scene.scene == name)
    }
}

/// Parse outcome for `.tljoint` source.
#[derive(Debug, Clone)]
pub struct TljointParseOutcome {
    pub manifest: Option<TljointManifest>,
    pub diagnostics: Vec<TljointDiagnostic>,
}

impl TljointParseOutcome {
    pub fn has_errors(&self) -> bool {
        self.diagnostics
            .iter()
            .any(|d| d.level == TljointDiagnosticLevel::Error)
    }
}

/// Runtime-compiled scene bundle from a `.tljoint` scene binding.
#[derive(Debug, Clone)]
pub struct TljointSceneBundle {
    pub manifest_path: PathBuf,
    pub scene_name: String,
    pub script_paths: Vec<PathBuf>,
    pub sprite_paths: Vec<PathBuf>,
    pub scripts: Vec<TlscriptShowcaseProgram<'static>>,
    pub merged_sprite_program: Option<TlspriteProgram>,
}

impl TljointSceneBundle {
    /// Evaluate all scene scripts in declaration order and merge outputs.
    pub fn evaluate_frame(
        &self,
        input: TlscriptShowcaseFrameInput,
        controls: TlscriptShowcaseControlInput,
    ) -> TlscriptShowcaseFrameOutput {
        let mut merged = empty_frame_output();
        for (index, script) in self.scripts.iter().enumerate() {
            let out = script.evaluate_frame_with_controls(input, controls);
            merge_frame_output(&mut merged, out, index);
        }
        merged
    }
}

/// Compile outcome for one `.tljoint` + scene selection.
#[derive(Debug, Clone)]
pub struct TljointSceneCompileOutcome {
    pub bundle: Option<TljointSceneBundle>,
    pub diagnostics: Vec<TljointDiagnostic>,
}

impl TljointSceneCompileOutcome {
    pub fn has_errors(&self) -> bool {
        self.diagnostics
            .iter()
            .any(|d| d.level == TljointDiagnosticLevel::Error)
    }
}

/// Parse `.tljoint` text in-memory.
pub fn parse_tljoint(source: &str) -> TljointParseOutcome {
    let mut diagnostics = Vec::new();
    let mut manifest = TljointManifest::default();
    let mut current_scene: Option<TljointSceneBinding> = None;
    let mut saw_header = false;

    for (line_index, raw_line) in source.lines().enumerate() {
        let line_no = line_index + 1;
        let line = strip_comment(raw_line).trim();
        if line.is_empty() {
            continue;
        }

        if !saw_header {
            saw_header = true;
            if line != "tljoint_v1" {
                diagnostics.push(TljointDiagnostic {
                    level: TljointDiagnosticLevel::Error,
                    line: line_no,
                    message: format!("expected 'tljoint_v1' header, got '{line}'"),
                });
            }
            continue;
        }

        if line.starts_with('[') && line.ends_with(']') {
            if let Some(scene) = current_scene.take() {
                manifest.scenes.push(scene);
            }
            let section = &line[1..line.len() - 1];
            let Some(name) = section.strip_prefix("scene.") else {
                diagnostics.push(TljointDiagnostic {
                    level: TljointDiagnosticLevel::Error,
                    line: line_no,
                    message: format!(
                        "invalid section '{section}', expected '[scene.<name>]' format"
                    ),
                });
                continue;
            };
            if name.trim().is_empty() {
                diagnostics.push(TljointDiagnostic {
                    level: TljointDiagnosticLevel::Error,
                    line: line_no,
                    message: "scene name cannot be empty".to_string(),
                });
                continue;
            }
            current_scene = Some(TljointSceneBinding {
                scene: name.trim().to_string(),
                tlscripts: Vec::new(),
                tlsprites: Vec::new(),
            });
            continue;
        }

        let Some((key, value)) = line.split_once('=') else {
            diagnostics.push(TljointDiagnostic {
                level: TljointDiagnosticLevel::Error,
                line: line_no,
                message: format!("expected key=value, got '{line}'"),
            });
            continue;
        };
        let key = key.trim();
        let value = value.trim();
        let Some(scene) = current_scene.as_mut() else {
            diagnostics.push(TljointDiagnostic {
                level: TljointDiagnosticLevel::Error,
                line: line_no,
                message: "key declared before any [scene.*] section".to_string(),
            });
            continue;
        };

        match key {
            "tlscript" | "script" => {
                push_if_non_empty(&mut scene.tlscripts, value, line_no, &mut diagnostics)
            }
            "tlscripts" | "scripts" => {
                push_csv(&mut scene.tlscripts, value, line_no, &mut diagnostics)
            }
            "tlsprite" | "sprite" => {
                push_if_non_empty(&mut scene.tlsprites, value, line_no, &mut diagnostics)
            }
            "tlsprites" | "sprites" => {
                push_csv(&mut scene.tlsprites, value, line_no, &mut diagnostics)
            }
            other => diagnostics.push(TljointDiagnostic {
                level: TljointDiagnosticLevel::Warning,
                line: line_no,
                message: format!("unknown key '{other}' ignored"),
            }),
        }
    }

    if let Some(scene) = current_scene.take() {
        manifest.scenes.push(scene);
    }

    if !saw_header {
        diagnostics.push(TljointDiagnostic {
            level: TljointDiagnosticLevel::Error,
            line: 1,
            message: "missing 'tljoint_v1' header".to_string(),
        });
    }
    if manifest.scenes.is_empty() {
        diagnostics.push(TljointDiagnostic {
            level: TljointDiagnosticLevel::Error,
            line: 1,
            message: "no [scene.*] sections found".to_string(),
        });
    }
    for scene in &manifest.scenes {
        if scene.tlscripts.is_empty() {
            diagnostics.push(TljointDiagnostic {
                level: TljointDiagnosticLevel::Warning,
                line: 1,
                message: format!("scene '{}' has no tlscripts", scene.scene),
            });
        }
        if scene.tlsprites.is_empty() {
            diagnostics.push(TljointDiagnostic {
                level: TljointDiagnosticLevel::Warning,
                line: 1,
                message: format!("scene '{}' has no tlsprites", scene.scene),
            });
        }
    }

    let manifest = if diagnostics
        .iter()
        .any(|d| d.level == TljointDiagnosticLevel::Error)
    {
        None
    } else {
        Some(manifest)
    };
    TljointParseOutcome {
        manifest,
        diagnostics,
    }
}

/// Read and parse `.tljoint` from disk.
pub fn load_tljoint(path: &Path) -> Result<TljointParseOutcome, String> {
    let source = fs::read_to_string(path)
        .map_err(|err| format!("failed to read '{}': {err}", path.display()))?;
    Ok(parse_tljoint(&source))
}

/// Compile one scene from `.tljoint` manifest path.
pub fn compile_tljoint_scene_from_path(
    manifest_path: &Path,
    scene_name: &str,
    script_config: TlscriptShowcaseConfig,
) -> TljointSceneCompileOutcome {
    let mut diagnostics = Vec::new();
    let parse = match load_tljoint(manifest_path) {
        Ok(parse) => parse,
        Err(err) => {
            diagnostics.push(TljointDiagnostic {
                level: TljointDiagnosticLevel::Error,
                line: 1,
                message: err,
            });
            return TljointSceneCompileOutcome {
                bundle: None,
                diagnostics,
            };
        }
    };
    diagnostics.extend(parse.diagnostics);
    let Some(manifest) = parse.manifest else {
        return TljointSceneCompileOutcome {
            bundle: None,
            diagnostics,
        };
    };
    let Some(scene) = manifest.scene(scene_name) else {
        diagnostics.push(TljointDiagnostic {
            level: TljointDiagnosticLevel::Error,
            line: 1,
            message: format!(
                "scene '{scene_name}' was not found in '{}'",
                manifest_path.display()
            ),
        });
        return TljointSceneCompileOutcome {
            bundle: None,
            diagnostics,
        };
    };

    let base_dir = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    let script_paths = scene
        .tlscripts
        .iter()
        .map(|raw| resolve_path(base_dir, raw))
        .collect::<Vec<_>>();
    let sprite_paths = scene
        .tlsprites
        .iter()
        .map(|raw| resolve_path(base_dir, raw))
        .collect::<Vec<_>>();

    let mut script_programs = Vec::new();
    for script_path in &script_paths {
        let source = match fs::read_to_string(script_path) {
            Ok(source) => source,
            Err(err) => {
                diagnostics.push(TljointDiagnostic {
                    level: TljointDiagnosticLevel::Error,
                    line: 1,
                    message: format!("failed to read tlscript '{}': {err}", script_path.display()),
                });
                continue;
            }
        };
        let leaked: &'static str = Box::leak(source.into_boxed_str());
        let compiled = compile_tlscript_showcase(leaked, script_config.clone());
        for warning in &compiled.warnings {
            diagnostics.push(TljointDiagnostic {
                level: TljointDiagnosticLevel::Warning,
                line: 1,
                message: format!("{}: {warning}", script_path.display()),
            });
        }
        if !compiled.errors.is_empty() {
            for error in compiled.errors {
                diagnostics.push(TljointDiagnostic {
                    level: TljointDiagnosticLevel::Error,
                    line: 1,
                    message: format!("{}: {error}", script_path.display()),
                });
            }
            continue;
        }
        if let Some(program) = compiled.program {
            script_programs.push(program);
        }
    }

    let mut sprite_programs = Vec::new();
    for sprite_path in &sprite_paths {
        let source = match fs::read_to_string(sprite_path) {
            Ok(source) => source,
            Err(err) => {
                diagnostics.push(TljointDiagnostic {
                    level: TljointDiagnosticLevel::Error,
                    line: 1,
                    message: format!("failed to read tlsprite '{}': {err}", sprite_path.display()),
                });
                continue;
            }
        };
        let compiled = compile_tlsprite(&source);
        for diagnostic in &compiled.diagnostics {
            diagnostics.push(TljointDiagnostic {
                level: match diagnostic.level {
                    TlspriteDiagnosticLevel::Warning => TljointDiagnosticLevel::Warning,
                    TlspriteDiagnosticLevel::Error => TljointDiagnosticLevel::Error,
                },
                line: diagnostic.line,
                message: format!("{}: {}", sprite_path.display(), diagnostic.message),
            });
        }
        if let Some(program) = compiled.program {
            sprite_programs.push(program);
        }
    }

    if script_programs.is_empty() {
        diagnostics.push(TljointDiagnostic {
            level: TljointDiagnosticLevel::Error,
            line: 1,
            message: format!("scene '{scene_name}' produced no runnable tlscript programs"),
        });
    }

    let has_errors = diagnostics
        .iter()
        .any(|d| d.level == TljointDiagnosticLevel::Error);
    if has_errors {
        return TljointSceneCompileOutcome {
            bundle: None,
            diagnostics,
        };
    }

    let merged_sprite_program = if sprite_programs.is_empty() {
        None
    } else {
        Some(TlspriteProgram::merge_programs(&sprite_programs))
    };

    TljointSceneCompileOutcome {
        bundle: Some(TljointSceneBundle {
            manifest_path: manifest_path.to_path_buf(),
            scene_name: scene_name.to_string(),
            script_paths,
            sprite_paths,
            scripts: script_programs,
            merged_sprite_program,
        }),
        diagnostics,
    }
}

fn strip_comment(line: &str) -> &str {
    line.split_once('#').map_or(line, |(head, _)| head)
}

fn push_if_non_empty(
    out: &mut Vec<String>,
    value: &str,
    line: usize,
    diagnostics: &mut Vec<TljointDiagnostic>,
) {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        diagnostics.push(TljointDiagnostic {
            level: TljointDiagnosticLevel::Warning,
            line,
            message: "empty value ignored".to_string(),
        });
        return;
    }
    out.push(trimmed.to_string());
}

fn push_csv(
    out: &mut Vec<String>,
    value: &str,
    line: usize,
    diagnostics: &mut Vec<TljointDiagnostic>,
) {
    let mut added = 0usize;
    for part in value.split(',') {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            continue;
        }
        out.push(trimmed.to_string());
        added += 1;
    }
    if added == 0 {
        diagnostics.push(TljointDiagnostic {
            level: TljointDiagnosticLevel::Warning,
            line,
            message: "empty CSV list ignored".to_string(),
        });
    }
}

fn resolve_path(base_dir: &Path, raw: &str) -> PathBuf {
    let path = PathBuf::from(raw);
    if path.is_absolute() {
        path
    } else {
        base_dir.join(path)
    }
}

fn empty_frame_output() -> TlscriptShowcaseFrameOutput {
    TlscriptShowcaseFrameOutput {
        patch: BounceTankRuntimePatch::default(),
        force_full_fbx_sphere: None,
        camera_move_speed: None,
        camera_look_sensitivity: None,
        camera_pose: None,
        camera_coordinate_space: None,
        camera_translate_delta: None,
        camera_rotate_delta_deg: None,
        camera_move_axis: None,
        camera_look_delta: None,
        camera_sprint: None,
        camera_look_active: None,
        camera_reset_pose: false,
        dispatch_decision: None,
        warnings: Vec::new(),
        aborted_early: false,
    }
}

fn merge_frame_output(
    merged: &mut TlscriptShowcaseFrameOutput,
    mut next: TlscriptShowcaseFrameOutput,
    script_index: usize,
) {
    merge_patch(&mut merged.patch, next.patch);
    if next.force_full_fbx_sphere.is_some() {
        merged.force_full_fbx_sphere = next.force_full_fbx_sphere;
    }
    if next.camera_move_speed.is_some() {
        merged.camera_move_speed = next.camera_move_speed;
    }
    if next.camera_look_sensitivity.is_some() {
        merged.camera_look_sensitivity = next.camera_look_sensitivity;
    }
    if next.camera_pose.is_some() {
        merged.camera_pose = next.camera_pose;
    }
    if next.camera_coordinate_space.is_some() {
        merged.camera_coordinate_space = next.camera_coordinate_space;
    }
    if next.camera_translate_delta.is_some() {
        merged.camera_translate_delta = next.camera_translate_delta;
    }
    if next.camera_rotate_delta_deg.is_some() {
        merged.camera_rotate_delta_deg = next.camera_rotate_delta_deg;
    }
    if next.camera_move_axis.is_some() {
        merged.camera_move_axis = next.camera_move_axis;
    }
    if next.camera_look_delta.is_some() {
        merged.camera_look_delta = next.camera_look_delta;
    }
    if next.camera_sprint.is_some() {
        merged.camera_sprint = next.camera_sprint;
    }
    if next.camera_look_active.is_some() {
        merged.camera_look_active = next.camera_look_active;
    }
    merged.camera_reset_pose |= next.camera_reset_pose;

    merged.dispatch_decision = match (merged.dispatch_decision, next.dispatch_decision) {
        (None, right) => right,
        (left, None) => left,
        (Some(existing), Some(candidate)) => {
            if candidate.is_parallel() {
                Some(candidate)
            } else {
                Some(existing)
            }
        }
    };

    merged.aborted_early |= next.aborted_early;
    if !next.warnings.is_empty() {
        for warning in next.warnings.drain(..) {
            merged
                .warnings
                .push(format!("script[{script_index}] {warning}"));
        }
    }
}

fn merge_patch(target: &mut BounceTankRuntimePatch, patch: BounceTankRuntimePatch) {
    if patch.target_ball_count.is_some() {
        target.target_ball_count = patch.target_ball_count;
    }
    if patch.spawn_per_tick.is_some() {
        target.spawn_per_tick = patch.spawn_per_tick;
    }
    if patch.linear_damping.is_some() {
        target.linear_damping = patch.linear_damping;
    }
    if patch.gravity.is_some() {
        target.gravity = patch.gravity;
    }
    if patch.contact_guard.is_some() {
        target.contact_guard = patch.contact_guard;
    }
    if patch.ball_restitution.is_some() {
        target.ball_restitution = patch.ball_restitution;
    }
    if patch.wall_restitution.is_some() {
        target.wall_restitution = patch.wall_restitution;
    }
    if patch.initial_speed_min.is_some() {
        target.initial_speed_min = patch.initial_speed_min;
    }
    if patch.initial_speed_max.is_some() {
        target.initial_speed_max = patch.initial_speed_max;
    }
    if patch.scatter_interval_ticks.is_some() {
        target.scatter_interval_ticks = patch.scatter_interval_ticks;
    }
    if patch.scatter_strength.is_some() {
        target.scatter_strength = patch.scatter_strength;
    }
    if patch.ball_mesh_slot.is_some() {
        target.ball_mesh_slot = patch.ball_mesh_slot;
    }
    if patch.container_mesh_slot.is_some() {
        target.container_mesh_slot = patch.container_mesh_slot;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_dir(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("tileline-{name}-{nonce}"));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    #[test]
    fn parses_multi_scene_manifest() {
        let source = concat!(
            "tljoint_v1\n",
            "[scene.main]\n",
            "tlscripts = a.tlscript, b.tlscript\n",
            "tlsprites = hud_a.tlsprite, hud_b.tlsprite\n",
            "[scene.ui]\n",
            "tlscript = ui.tlscript\n",
            "tlsprite = ui.tlsprite\n",
        );
        let out = parse_tljoint(source);
        assert!(!out.has_errors(), "{:?}", out.diagnostics);
        let manifest = out.manifest.expect("manifest");
        assert_eq!(manifest.scenes.len(), 2);
        assert_eq!(manifest.scenes[0].tlscripts.len(), 2);
        assert_eq!(manifest.scenes[0].tlsprites.len(), 2);
    }

    #[test]
    fn compiles_scene_bundle_and_merges_sprite_programs() {
        let dir = unique_temp_dir("tljoint");
        let joint = dir.join("scene.tljoint");
        let script_a = dir.join("a.tlscript");
        let script_b = dir.join("b.tlscript");
        let sprite_a = dir.join("a.tlsprite");
        let sprite_b = dir.join("b.tlsprite");

        std::fs::write(
            &joint,
            concat!(
                "tljoint_v1\n",
                "[scene.main]\n",
                "tlscripts = a.tlscript, b.tlscript\n",
                "tlsprites = a.tlsprite, b.tlsprite\n",
            ),
        )
        .expect("write joint");
        std::fs::write(
            &script_a,
            concat!(
                "@export\n",
                "def showcase_tick(frame: int, live_balls: int, spawned_this_tick: int, key_f_down: bool, input_move_x: float, input_move_y: float, input_move_z: float, input_look_dx: float, input_look_dy: float, input_sprint_down: bool, input_look_active: bool, input_reset_camera: bool):\n",
                "    set_spawn_per_tick(64)\n",
            ),
        )
        .expect("write script a");
        std::fs::write(
            &script_b,
            concat!(
                "@export\n",
                "def showcase_tick(frame: int, live_balls: int, spawned_this_tick: int, key_f_down: bool, input_move_x: float, input_move_y: float, input_move_z: float, input_look_dx: float, input_look_dy: float, input_sprint_down: bool, input_look_active: bool, input_reset_camera: bool):\n",
                "    set_target_ball_count(1024)\n",
            ),
        )
        .expect("write script b");
        std::fs::write(
            &sprite_a,
            concat!(
                "tlsprite_v1\n",
                "[a]\n",
                "sprite_id = 1\n",
                "kind = hud\n",
                "texture_slot = 0\n",
                "layer = 100\n",
                "position = 0.0, 0.0, 0.0\n",
                "size = 0.1, 0.1\n",
                "rotation_rad = 0.0\n",
                "color = 1.0, 1.0, 1.0, 1.0\n",
            ),
        )
        .expect("write sprite a");
        std::fs::write(
            &sprite_b,
            concat!(
                "tlsprite_v1\n",
                "[b]\n",
                "sprite_id = 2\n",
                "kind = hud\n",
                "texture_slot = 1\n",
                "layer = 101\n",
                "position = 0.0, 0.1, 0.0\n",
                "size = 0.1, 0.1\n",
                "rotation_rad = 0.0\n",
                "color = 1.0, 1.0, 1.0, 1.0\n",
            ),
        )
        .expect("write sprite b");

        let out =
            compile_tljoint_scene_from_path(&joint, "main", TlscriptShowcaseConfig::default());
        assert!(!out.has_errors(), "{:?}", out.diagnostics);
        let bundle = out.bundle.expect("bundle");
        assert_eq!(bundle.scripts.len(), 2);
        let merged_sprite = bundle
            .merged_sprite_program
            .as_ref()
            .expect("merged sprite program");
        assert_eq!(merged_sprite.sprites().len(), 2);

        let frame = bundle.evaluate_frame(
            TlscriptShowcaseFrameInput {
                frame_index: 1,
                live_balls: 0,
                spawned_this_tick: 0,
                key_f_down: false,
            },
            TlscriptShowcaseControlInput::default(),
        );
        assert_eq!(frame.patch.spawn_per_tick, Some(64));
        assert_eq!(frame.patch.target_ball_count, Some(1024));
    }
}
