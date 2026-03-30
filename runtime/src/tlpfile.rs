//! `.tlpfile` project manifest loader and scene compiler.
//!
//! A `.tlpfile` lets runtime tools keep `.tlscript`, `.tlsprite`, and optional `.tljoint`
//! references in one project-level file.
//!
//! Minimal format:
//! ```text
//! tlpfile_v1
//! [project]
//! name = TLApp Demo
//! scheduler = auto
//! default_dimension = 3d
//! default_scene = main
//!
//! [scene.main]
//! dimension = 3d
//! tljoint = main.tljoint
//! tljoint_scene = main
//! tlscripts = bonus_rules.tlscript
//! tlsprites = overlay.tlsprite
//! ```

use std::fs;
use std::path::{Path, PathBuf};

use crate::runtime_bridge::{GmsGuardrailProfile, GmsScalerConfig, GmsScalerMode};
use crate::tljoint::{compile_tljoint_scene_from_path, TljointDiagnosticLevel, TljointSceneBundle};
use crate::tlscript_showcase::{
    compile_tlscript_showcase, TlscriptShowcaseConfig, TlscriptShowcaseProgram,
};
use crate::tlsprite::{
    compile_tlsprite_with_extra_roots, TlspriteDiagnosticLevel, TlspriteProgram,
};

/// Project-level graphics scheduler selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TlpfileGraphicsScheduler {
    /// Platform policy decides the runtime path.
    Auto,
    /// Graphics Multi Scaler path (desktop/high-throughput).
    Gms,
    /// Mobile Graphics Scaler path (mobile fallback).
    Mgs,
}

impl Default for TlpfileGraphicsScheduler {
    fn default() -> Self {
        Self::Auto
    }
}

impl TlpfileGraphicsScheduler {
    fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "auto" => Some(Self::Auto),
            "gms" => Some(Self::Gms),
            "mgs" => Some(Self::Mgs),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Gms => "gms",
            Self::Mgs => "mgs",
        }
    }
}

/// Scene-space dimension selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TlpfileSceneDimension {
    TwoD,
    ThreeD,
}

impl Default for TlpfileSceneDimension {
    fn default() -> Self {
        Self::ThreeD
    }
}

impl TlpfileSceneDimension {
    fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "2d" | "side2d" | "side-view-2d" | "side_view_2d" | "sideview2d" => Some(Self::TwoD),
            "3d" | "spatial3d" | "spatial_3d" | "spatial3" | "spatial" => Some(Self::ThreeD),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::TwoD => "2d",
            Self::ThreeD => "3d",
        }
    }
}

/// Diagnostic level for `.tlpfile` parsing/compilation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TlpfileDiagnosticLevel {
    Warning,
    Error,
}

/// One `.tlpfile` diagnostic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TlpfileDiagnostic {
    pub level: TlpfileDiagnosticLevel,
    pub line: usize,
    pub message: String,
}

/// One scene entry resolved from `.tlpfile`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TlpfileSceneBinding {
    pub scene: String,
    pub dimension: TlpfileSceneDimension,
    pub tljoint: Option<String>,
    pub tljoint_scene: Option<String>,
    pub tlscripts: Vec<String>,
    pub tlsprites: Vec<String>,
}

/// Parsed `.tlpfile` project.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TlpfileProject {
    pub name: String,
    pub scheduler: TlpfileGraphicsScheduler,
    pub gms_scaler: GmsScalerConfig,
    pub default_dimension: TlpfileSceneDimension,
    pub default_scene: String,
    pub scenes: Vec<TlpfileSceneBinding>,
}

impl TlpfileProject {
    pub fn scene(&self, name: &str) -> Option<&TlpfileSceneBinding> {
        self.scenes.iter().find(|scene| scene.scene == name)
    }
}

/// Parse outcome for `.tlpfile`.
#[derive(Debug, Clone)]
pub struct TlpfileParseOutcome {
    pub project: Option<TlpfileProject>,
    pub diagnostics: Vec<TlpfileDiagnostic>,
}

impl TlpfileParseOutcome {
    pub fn has_errors(&self) -> bool {
        self.diagnostics
            .iter()
            .any(|d| d.level == TlpfileDiagnosticLevel::Error)
    }
}

/// Runtime-compiled scene bundle from `.tlpfile`.
#[derive(Debug, Clone)]
pub struct TlpfileSceneBundle {
    pub project_path: PathBuf,
    pub project_name: String,
    pub scheduler: TlpfileGraphicsScheduler,
    pub gms_scaler: GmsScalerConfig,
    pub scene_dimension: TlpfileSceneDimension,
    pub scene_name: String,
    pub selected_joint_path: Option<PathBuf>,
    pub selected_joint_scene: Option<String>,
    pub script_paths: Vec<PathBuf>,
    pub sprite_paths: Vec<PathBuf>,
    pub scripts: Vec<TlscriptShowcaseProgram<'static>>,
    pub merged_sprite_program: Option<TlspriteProgram>,
}

impl TlpfileSceneBundle {
    pub fn sprite_count(&self) -> usize {
        self.merged_sprite_program
            .as_ref()
            .map(|p| p.sprites().len())
            .unwrap_or(0)
    }
}

/// Compile outcome for `.tlpfile` + selected scene.
#[derive(Debug, Clone)]
pub struct TlpfileSceneCompileOutcome {
    pub bundle: Option<TlpfileSceneBundle>,
    pub diagnostics: Vec<TlpfileDiagnostic>,
}

impl TlpfileSceneCompileOutcome {
    pub fn has_errors(&self) -> bool {
        self.diagnostics
            .iter()
            .any(|d| d.level == TlpfileDiagnosticLevel::Error)
    }
}

/// Parse `.tlpfile` text in-memory.
pub fn parse_tlpfile(source: &str) -> TlpfileParseOutcome {
    let mut diagnostics = Vec::new();
    let mut scenes = Vec::new();
    let mut saw_header = false;
    let mut project_name: Option<String> = None;
    let mut project_scheduler: Option<TlpfileGraphicsScheduler> = None;
    let mut project_gms_scaler = GmsScalerConfig::default();
    let mut project_default_dimension: Option<TlpfileSceneDimension> = None;
    let mut project_default_scene: Option<String> = None;
    let mut current_scene: Option<TlpfileSceneBinding> = None;
    let mut in_project_section = false;
    let mut in_gms_scaler_section = false;

    for (line_index, raw_line) in source.lines().enumerate() {
        let line_no = line_index + 1;
        let line = strip_comment(raw_line).trim();
        if line.is_empty() {
            continue;
        }

        if !saw_header {
            saw_header = true;
            if line != "tlpfile_v1" {
                diagnostics.push(TlpfileDiagnostic {
                    level: TlpfileDiagnosticLevel::Error,
                    line: line_no,
                    message: format!("expected 'tlpfile_v1' header, got '{line}'"),
                });
            }
            continue;
        }

        if line.starts_with('[') && line.ends_with(']') {
            if let Some(scene) = current_scene.take() {
                scenes.push(scene);
            }
            in_project_section = false;
            in_gms_scaler_section = false;
            let section = line[1..line.len() - 1].trim();
            if section.eq_ignore_ascii_case("project") {
                in_project_section = true;
                continue;
            }
            if section.eq_ignore_ascii_case("gms_scaler") {
                in_gms_scaler_section = true;
                continue;
            }
            if let Some(name) = section.strip_prefix("scene.") {
                let name = name.trim();
                if name.is_empty() {
                    diagnostics.push(TlpfileDiagnostic {
                        level: TlpfileDiagnosticLevel::Error,
                        line: line_no,
                        message: "scene name cannot be empty".to_string(),
                    });
                    continue;
                }
                current_scene = Some(TlpfileSceneBinding {
                    scene: name.to_string(),
                    dimension: project_default_dimension.unwrap_or_default(),
                    ..TlpfileSceneBinding::default()
                });
                continue;
            }
            diagnostics.push(TlpfileDiagnostic {
                level: TlpfileDiagnosticLevel::Warning,
                line: line_no,
                message: format!(
                    "unknown section '{section}' ignored (expected [project], [gms_scaler], or [scene.<name>])"
                ),
            });
            continue;
        }

        let Some((key, value)) = line.split_once('=') else {
            diagnostics.push(TlpfileDiagnostic {
                level: TlpfileDiagnosticLevel::Error,
                line: line_no,
                message: format!("expected key=value, got '{line}'"),
            });
            continue;
        };
        let key = key.trim();
        let value = value.trim();

        if in_project_section {
            match key {
                "name" => {
                    if value.is_empty() {
                        diagnostics.push(TlpfileDiagnostic {
                            level: TlpfileDiagnosticLevel::Warning,
                            line: line_no,
                            message: "empty project name ignored".to_string(),
                        });
                    } else {
                        project_name = Some(value.to_string());
                    }
                }
                "default_scene" => {
                    if value.is_empty() {
                        diagnostics.push(TlpfileDiagnostic {
                            level: TlpfileDiagnosticLevel::Warning,
                            line: line_no,
                            message: "empty default_scene ignored".to_string(),
                        });
                    } else {
                        project_default_scene = Some(value.to_string());
                    }
                }
                "scheduler" => {
                    if value.is_empty() {
                        diagnostics.push(TlpfileDiagnostic {
                            level: TlpfileDiagnosticLevel::Warning,
                            line: line_no,
                            message: "empty scheduler ignored (expected auto|gms|mgs)".to_string(),
                        });
                    } else if let Some(parsed) = TlpfileGraphicsScheduler::parse(value) {
                        project_scheduler = Some(parsed);
                    } else {
                        diagnostics.push(TlpfileDiagnostic {
                            level: TlpfileDiagnosticLevel::Error,
                            line: line_no,
                            message: format!("invalid scheduler '{value}' (expected auto|gms|mgs)"),
                        });
                    }
                }
                "default_dimension" | "dimension" | "scene_mode" | "mode" => {
                    if value.is_empty() {
                        diagnostics.push(TlpfileDiagnostic {
                            level: TlpfileDiagnosticLevel::Warning,
                            line: line_no,
                            message: "empty default_dimension ignored (expected 2d|3d)".to_string(),
                        });
                    } else if let Some(parsed) = TlpfileSceneDimension::parse(value) {
                        project_default_dimension = Some(parsed);
                    } else {
                        diagnostics.push(TlpfileDiagnostic {
                            level: TlpfileDiagnosticLevel::Error,
                            line: line_no,
                            message: format!(
                                "invalid default_dimension '{value}' (expected 2d|3d)"
                            ),
                        });
                    }
                }
                other => diagnostics.push(TlpfileDiagnostic {
                    level: TlpfileDiagnosticLevel::Warning,
                    line: line_no,
                    message: format!("unknown project key '{other}' ignored"),
                }),
            }
            continue;
        }

        if in_gms_scaler_section {
            match key {
                "mode" => {
                    if value.is_empty() {
                        diagnostics.push(TlpfileDiagnostic {
                            level: TlpfileDiagnosticLevel::Warning,
                            line: line_no,
                            message: "empty gms_scaler.mode ignored (expected adaptive|fixed)"
                                .to_string(),
                        });
                    } else if let Some(parsed) = GmsScalerMode::parse(value) {
                        project_gms_scaler.mode = parsed;
                    } else {
                        diagnostics.push(TlpfileDiagnostic {
                            level: TlpfileDiagnosticLevel::Error,
                            line: line_no,
                            message: format!(
                                "invalid gms_scaler.mode '{value}' (expected adaptive|fixed)"
                            ),
                        });
                    }
                }
                "target_fps" => match value.parse::<u32>() {
                    Ok(v) if (24..=480).contains(&v) => project_gms_scaler.target_fps = v,
                    Ok(_) => diagnostics.push(TlpfileDiagnostic {
                        level: TlpfileDiagnosticLevel::Error,
                        line: line_no,
                        message: "gms_scaler.target_fps must be in 24..=480".to_string(),
                    }),
                    Err(_) => diagnostics.push(TlpfileDiagnostic {
                        level: TlpfileDiagnosticLevel::Error,
                        line: line_no,
                        message: format!("invalid gms_scaler.target_fps '{value}'"),
                    }),
                },
                "min_physics_budget_pct" => match value.parse::<u8>() {
                    Ok(v) if v <= 100 => project_gms_scaler.min_physics_budget_pct = v,
                    Ok(_) => diagnostics.push(TlpfileDiagnostic {
                        level: TlpfileDiagnosticLevel::Error,
                        line: line_no,
                        message: "gms_scaler.min_physics_budget_pct must be in 0..=100"
                            .to_string(),
                    }),
                    Err(_) => diagnostics.push(TlpfileDiagnostic {
                        level: TlpfileDiagnosticLevel::Error,
                        line: line_no,
                        message: format!("invalid gms_scaler.min_physics_budget_pct '{value}'"),
                    }),
                },
                "render_budget_pct" => match value.parse::<u8>() {
                    Ok(v) if v <= 100 => project_gms_scaler.budgets.render_budget_pct = v,
                    Ok(_) => diagnostics.push(TlpfileDiagnostic {
                        level: TlpfileDiagnosticLevel::Error,
                        line: line_no,
                        message: "gms_scaler.render_budget_pct must be in 0..=100".to_string(),
                    }),
                    Err(_) => diagnostics.push(TlpfileDiagnostic {
                        level: TlpfileDiagnosticLevel::Error,
                        line: line_no,
                        message: format!("invalid gms_scaler.render_budget_pct '{value}'"),
                    }),
                },
                "physics_budget_pct" => match value.parse::<u8>() {
                    Ok(v) if v <= 100 => project_gms_scaler.budgets.physics_budget_pct = v,
                    Ok(_) => diagnostics.push(TlpfileDiagnostic {
                        level: TlpfileDiagnosticLevel::Error,
                        line: line_no,
                        message: "gms_scaler.physics_budget_pct must be in 0..=100".to_string(),
                    }),
                    Err(_) => diagnostics.push(TlpfileDiagnostic {
                        level: TlpfileDiagnosticLevel::Error,
                        line: line_no,
                        message: format!("invalid gms_scaler.physics_budget_pct '{value}'"),
                    }),
                },
                "ai_ml_budget_pct" => match value.parse::<u8>() {
                    Ok(v) if v <= 100 => project_gms_scaler.budgets.ai_ml_budget_pct = v,
                    Ok(_) => diagnostics.push(TlpfileDiagnostic {
                        level: TlpfileDiagnosticLevel::Error,
                        line: line_no,
                        message: "gms_scaler.ai_ml_budget_pct must be in 0..=100".to_string(),
                    }),
                    Err(_) => diagnostics.push(TlpfileDiagnostic {
                        level: TlpfileDiagnosticLevel::Error,
                        line: line_no,
                        message: format!("invalid gms_scaler.ai_ml_budget_pct '{value}'"),
                    }),
                },
                "postfx_budget_pct" => match value.parse::<u8>() {
                    Ok(v) if v <= 100 => project_gms_scaler.budgets.postfx_budget_pct = v,
                    Ok(_) => diagnostics.push(TlpfileDiagnostic {
                        level: TlpfileDiagnosticLevel::Error,
                        line: line_no,
                        message: "gms_scaler.postfx_budget_pct must be in 0..=100".to_string(),
                    }),
                    Err(_) => diagnostics.push(TlpfileDiagnostic {
                        level: TlpfileDiagnosticLevel::Error,
                        line: line_no,
                        message: format!("invalid gms_scaler.postfx_budget_pct '{value}'"),
                    }),
                },
                "ui_budget_pct" => match value.parse::<u8>() {
                    Ok(v) if v <= 100 => project_gms_scaler.budgets.ui_budget_pct = v,
                    Ok(_) => diagnostics.push(TlpfileDiagnostic {
                        level: TlpfileDiagnosticLevel::Error,
                        line: line_no,
                        message: "gms_scaler.ui_budget_pct must be in 0..=100".to_string(),
                    }),
                    Err(_) => diagnostics.push(TlpfileDiagnostic {
                        level: TlpfileDiagnosticLevel::Error,
                        line: line_no,
                        message: format!("invalid gms_scaler.ui_budget_pct '{value}'"),
                    }),
                },
                "guardrail" => {
                    if value.is_empty() {
                        diagnostics.push(TlpfileDiagnostic {
                            level: TlpfileDiagnosticLevel::Warning,
                            line: line_no,
                            message: "empty gms_scaler.guardrail ignored".to_string(),
                        });
                    } else if let Some(profile) = GmsGuardrailProfile::parse(value) {
                        project_gms_scaler.guardrail = profile;
                    } else {
                        diagnostics.push(TlpfileDiagnostic {
                            level: TlpfileDiagnosticLevel::Error,
                            line: line_no,
                            message: format!(
                                "invalid gms_scaler.guardrail '{value}' (expected balanced|aggressive|relaxed)"
                            ),
                        });
                    }
                }
                other => diagnostics.push(TlpfileDiagnostic {
                    level: TlpfileDiagnosticLevel::Warning,
                    line: line_no,
                    message: format!("unknown gms_scaler key '{other}' ignored"),
                }),
            }
            continue;
        }

        let Some(scene) = current_scene.as_mut() else {
            diagnostics.push(TlpfileDiagnostic {
                level: TlpfileDiagnosticLevel::Error,
                line: line_no,
                message: "key declared before any [project] or [scene.*] section".to_string(),
            });
            continue;
        };

        match key {
            "tljoint" | "joint" => {
                push_single(&mut scene.tljoint, value, line_no, &mut diagnostics)
            }
            "tljoint_scene" | "joint_scene" => {
                push_single(&mut scene.tljoint_scene, value, line_no, &mut diagnostics)
            }
            "tlscript" | "script" => {
                push_single_to_vec(&mut scene.tlscripts, value, line_no, &mut diagnostics)
            }
            "tlscripts" | "scripts" => {
                push_csv(&mut scene.tlscripts, value, line_no, &mut diagnostics)
            }
            "tlsprite" | "sprite" => {
                push_single_to_vec(&mut scene.tlsprites, value, line_no, &mut diagnostics)
            }
            "tlsprites" | "sprites" => {
                push_csv(&mut scene.tlsprites, value, line_no, &mut diagnostics)
            }
            "dimension" | "scene_mode" | "mode" => {
                if value.is_empty() {
                    diagnostics.push(TlpfileDiagnostic {
                        level: TlpfileDiagnosticLevel::Warning,
                        line: line_no,
                        message: "empty scene dimension ignored (expected 2d|3d)".to_string(),
                    });
                } else if let Some(parsed) = TlpfileSceneDimension::parse(value) {
                    scene.dimension = parsed;
                } else {
                    diagnostics.push(TlpfileDiagnostic {
                        level: TlpfileDiagnosticLevel::Error,
                        line: line_no,
                        message: format!("invalid scene dimension '{value}' (expected 2d|3d)"),
                    });
                }
            }
            other => diagnostics.push(TlpfileDiagnostic {
                level: TlpfileDiagnosticLevel::Warning,
                line: line_no,
                message: format!("unknown scene key '{other}' ignored"),
            }),
        }
    }

    if let Some(scene) = current_scene.take() {
        scenes.push(scene);
    }

    if !saw_header {
        diagnostics.push(TlpfileDiagnostic {
            level: TlpfileDiagnosticLevel::Error,
            line: 1,
            message: "missing 'tlpfile_v1' header".to_string(),
        });
    }
    if scenes.is_empty() {
        diagnostics.push(TlpfileDiagnostic {
            level: TlpfileDiagnosticLevel::Error,
            line: 1,
            message: "no [scene.*] sections found".to_string(),
        });
    }

    let default_scene = if let Some(default_scene) = project_default_scene {
        default_scene
    } else if let Some(first) = scenes.first() {
        diagnostics.push(TlpfileDiagnostic {
            level: TlpfileDiagnosticLevel::Warning,
            line: 1,
            message: format!(
                "project default_scene missing; using first scene '{}'",
                first.scene
            ),
        });
        first.scene.clone()
    } else {
        String::new()
    };

    if !default_scene.is_empty() && !scenes.iter().any(|scene| scene.scene == default_scene) {
        diagnostics.push(TlpfileDiagnostic {
            level: TlpfileDiagnosticLevel::Error,
            line: 1,
            message: format!("default_scene '{default_scene}' does not exist"),
        });
    }

    for scene in &scenes {
        if scene.tljoint.is_none() && scene.tlscripts.is_empty() {
            diagnostics.push(TlpfileDiagnostic {
                level: TlpfileDiagnosticLevel::Warning,
                line: 1,
                message: format!("scene '{}' has no tljoint/tlscript entries", scene.scene),
            });
        }
        if scene.tljoint.is_none() && scene.tlsprites.is_empty() {
            diagnostics.push(TlpfileDiagnostic {
                level: TlpfileDiagnosticLevel::Warning,
                line: 1,
                message: format!("scene '{}' has no tljoint/tlsprite entries", scene.scene),
            });
        }

        if scene.scene == "main" {
            match scene.tljoint.as_deref() {
                Some(path) => {
                    let file_name = Path::new(path).file_name().and_then(|name| name.to_str());
                    if file_name != Some("main.tljoint") {
                        diagnostics.push(TlpfileDiagnostic {
                            level: TlpfileDiagnosticLevel::Error,
                            line: 1,
                            message: format!(
                                "scene 'main' must use tljoint = main.tljoint (got '{path}')"
                            ),
                        });
                    }
                }
                None => diagnostics.push(TlpfileDiagnostic {
                    level: TlpfileDiagnosticLevel::Error,
                    line: 1,
                    message: "scene 'main' must declare tljoint = main.tljoint".to_string(),
                }),
            }
        }
    }

    let has_errors = diagnostics
        .iter()
        .any(|d| d.level == TlpfileDiagnosticLevel::Error);
    let project = if has_errors {
        None
    } else {
        Some(TlpfileProject {
            name: project_name.unwrap_or_else(|| "Tileline Project".to_string()),
            scheduler: project_scheduler.unwrap_or_default(),
            gms_scaler: project_gms_scaler,
            default_dimension: project_default_dimension.unwrap_or_default(),
            default_scene,
            scenes,
        })
    };

    TlpfileParseOutcome {
        project,
        diagnostics,
    }
}

/// Read and parse `.tlpfile` from disk.
pub fn load_tlpfile(path: &Path) -> Result<TlpfileParseOutcome, String> {
    let source = fs::read_to_string(path)
        .map_err(|err| format!("failed to read '{}': {err}", path.display()))?;
    Ok(parse_tlpfile(&source))
}

/// Compile one scene from `.tlpfile` project path.
pub fn compile_tlpfile_scene_from_path(
    project_path: &Path,
    scene_name: Option<&str>,
    script_config: TlscriptShowcaseConfig,
) -> TlpfileSceneCompileOutcome {
    let mut diagnostics = Vec::new();
    let parse = match load_tlpfile(project_path) {
        Ok(parse) => parse,
        Err(err) => {
            diagnostics.push(TlpfileDiagnostic {
                level: TlpfileDiagnosticLevel::Error,
                line: 1,
                message: err,
            });
            return TlpfileSceneCompileOutcome {
                bundle: None,
                diagnostics,
            };
        }
    };
    diagnostics.extend(parse.diagnostics);
    let Some(project) = parse.project else {
        return TlpfileSceneCompileOutcome {
            bundle: None,
            diagnostics,
        };
    };

    let resolved_scene_name = scene_name.unwrap_or(project.default_scene.as_str());
    let Some(scene) = project.scene(resolved_scene_name) else {
        diagnostics.push(TlpfileDiagnostic {
            level: TlpfileDiagnosticLevel::Error,
            line: 1,
            message: format!(
                "scene '{}' was not found in '{}'",
                resolved_scene_name,
                project_path.display()
            ),
        });
        return TlpfileSceneCompileOutcome {
            bundle: None,
            diagnostics,
        };
    };
    let scene_dimension = scene.dimension;

    let base_dir = project_path.parent().unwrap_or_else(|| Path::new("."));
    let mut joint_bundle: Option<TljointSceneBundle> = None;
    let mut selected_joint_path: Option<PathBuf> = None;
    let mut selected_joint_scene: Option<String> = None;

    if let Some(raw_joint_path) = scene.tljoint.as_deref() {
        let joint_path = resolve_path(base_dir, raw_joint_path);
        let joint_scene = scene
            .tljoint_scene
            .as_deref()
            .unwrap_or(resolved_scene_name)
            .to_string();
        let joint_out =
            compile_tljoint_scene_from_path(&joint_path, &joint_scene, script_config.clone());
        for diagnostic in joint_out.diagnostics {
            diagnostics.push(TlpfileDiagnostic {
                level: match diagnostic.level {
                    TljointDiagnosticLevel::Warning => TlpfileDiagnosticLevel::Warning,
                    TljointDiagnosticLevel::Error => TlpfileDiagnosticLevel::Error,
                },
                line: diagnostic.line,
                message: format!("[joint {}] {}", joint_path.display(), diagnostic.message),
            });
        }
        if let Some(bundle) = joint_out.bundle {
            selected_joint_path = Some(joint_path);
            selected_joint_scene = Some(joint_scene);
            joint_bundle = Some(bundle);
        }
    }

    let direct_script_paths = scene
        .tlscripts
        .iter()
        .map(|raw| resolve_path(base_dir, raw))
        .collect::<Vec<_>>();
    let direct_sprite_paths = scene
        .tlsprites
        .iter()
        .map(|raw| resolve_path(base_dir, raw))
        .collect::<Vec<_>>();

    let mut direct_script_programs = Vec::new();
    for script_path in &direct_script_paths {
        let source = match fs::read_to_string(script_path) {
            Ok(source) => source,
            Err(err) => {
                diagnostics.push(TlpfileDiagnostic {
                    level: TlpfileDiagnosticLevel::Error,
                    line: 1,
                    message: format!("failed to read tlscript '{}': {err}", script_path.display()),
                });
                continue;
            }
        };
        let leaked: &'static str = Box::leak(source.into_boxed_str());
        let compiled = compile_tlscript_showcase(leaked, script_config.clone());
        for warning in &compiled.warnings {
            diagnostics.push(TlpfileDiagnostic {
                level: TlpfileDiagnosticLevel::Warning,
                line: 1,
                message: format!("{}: {warning}", script_path.display()),
            });
        }
        if !compiled.errors.is_empty() {
            for error in compiled.errors {
                diagnostics.push(TlpfileDiagnostic {
                    level: TlpfileDiagnosticLevel::Error,
                    line: 1,
                    message: format!("{}: {error}", script_path.display()),
                });
            }
            continue;
        }
        if let Some(program) = compiled.program {
            direct_script_programs.push(program);
        }
    }

    let mut direct_sprite_programs = Vec::new();
    for sprite_path in &direct_sprite_paths {
        let source = match fs::read_to_string(sprite_path) {
            Ok(source) => source,
            Err(err) => {
                diagnostics.push(TlpfileDiagnostic {
                    level: TlpfileDiagnosticLevel::Error,
                    line: 1,
                    message: format!("failed to read tlsprite '{}': {err}", sprite_path.display()),
                });
                continue;
            }
        };

        let mut extra_roots = vec![base_dir.to_path_buf()];
        if let Some(parent) = sprite_path.parent() {
            extra_roots.push(parent.to_path_buf());
        }
        let compiled = compile_tlsprite_with_extra_roots(&source, &extra_roots);
        for diagnostic in compiled.diagnostics {
            diagnostics.push(TlpfileDiagnostic {
                level: match diagnostic.level {
                    TlspriteDiagnosticLevel::Warning => TlpfileDiagnosticLevel::Warning,
                    TlspriteDiagnosticLevel::Error => TlpfileDiagnosticLevel::Error,
                },
                line: diagnostic.line,
                message: format!("{}: {}", sprite_path.display(), diagnostic.message),
            });
        }
        if let Some(program) = compiled.program {
            direct_sprite_programs.push(program);
        }
    }

    let mut scripts = Vec::new();
    let mut script_paths = Vec::new();
    if let Some(bundle) = &joint_bundle {
        scripts.extend(bundle.scripts.iter().cloned());
        script_paths.extend(bundle.script_paths.iter().cloned());
    }
    scripts.extend(direct_script_programs);
    script_paths.extend(direct_script_paths);

    let mut sprite_programs = Vec::new();
    let mut sprite_paths = Vec::new();
    if let Some(bundle) = &joint_bundle {
        if let Some(program) = &bundle.merged_sprite_program {
            sprite_programs.push(program.clone());
        }
        sprite_paths.extend(bundle.sprite_paths.iter().cloned());
    }
    sprite_programs.extend(direct_sprite_programs);
    sprite_paths.extend(direct_sprite_paths);

    if scripts.is_empty() {
        diagnostics.push(TlpfileDiagnostic {
            level: TlpfileDiagnosticLevel::Error,
            line: 1,
            message: format!(
                "scene '{}' produced no runnable tlscript programs",
                resolved_scene_name
            ),
        });
    }

    let merged_sprite_program = if sprite_programs.is_empty() {
        None
    } else {
        Some(TlspriteProgram::merge_programs(&sprite_programs))
    };

    let has_errors = diagnostics
        .iter()
        .any(|d| d.level == TlpfileDiagnosticLevel::Error);
    if has_errors {
        return TlpfileSceneCompileOutcome {
            bundle: None,
            diagnostics,
        };
    }

    TlpfileSceneCompileOutcome {
        bundle: Some(TlpfileSceneBundle {
            project_path: project_path.to_path_buf(),
            project_name: project.name,
            scheduler: project.scheduler,
            gms_scaler: project.gms_scaler,
            scene_dimension,
            scene_name: resolved_scene_name.to_string(),
            selected_joint_path,
            selected_joint_scene,
            script_paths,
            sprite_paths,
            scripts,
            merged_sprite_program,
        }),
        diagnostics,
    }
}

fn strip_comment(line: &str) -> &str {
    line.split_once('#').map_or(line, |(head, _)| head)
}

fn push_single(
    out: &mut Option<String>,
    value: &str,
    line: usize,
    diagnostics: &mut Vec<TlpfileDiagnostic>,
) {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        diagnostics.push(TlpfileDiagnostic {
            level: TlpfileDiagnosticLevel::Warning,
            line,
            message: "empty value ignored".to_string(),
        });
        return;
    }
    *out = Some(trimmed.to_string());
}

fn push_single_to_vec(
    out: &mut Vec<String>,
    value: &str,
    line: usize,
    diagnostics: &mut Vec<TlpfileDiagnostic>,
) {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        diagnostics.push(TlpfileDiagnostic {
            level: TlpfileDiagnosticLevel::Warning,
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
    diagnostics: &mut Vec<TlpfileDiagnostic>,
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
        diagnostics.push(TlpfileDiagnostic {
            level: TlpfileDiagnosticLevel::Warning,
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn unique_temp_dir(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("time")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("tileline-tlpfile-{name}-{nonce}"));
        std::fs::create_dir_all(&dir).expect("create temp dir");
        dir
    }

    #[test]
    fn parses_project_and_scene_bindings() {
        let source = concat!(
            "tlpfile_v1\n",
            "[project]\n",
            "name = Demo Project\n",
            "scheduler = gms\n",
            "[gms_scaler]\n",
            "mode = adaptive\n",
            "target_fps = 75\n",
            "min_physics_budget_pct = 40\n",
            "render_budget_pct = 30\n",
            "physics_budget_pct = 40\n",
            "ai_ml_budget_pct = 20\n",
            "postfx_budget_pct = 10\n",
            "default_dimension = 3d\n",
            "default_scene = main\n",
            "[scene.main]\n",
            "dimension = 2d\n",
            "tljoint = main.tljoint\n",
            "tljoint_scene = main\n",
            "tlscripts = extra.tlscript\n",
            "tlsprites = overlay.tlsprite\n",
        );
        let out = parse_tlpfile(source);
        assert!(!out.has_errors(), "{:?}", out.diagnostics);
        let project = out.project.expect("project");
        assert_eq!(project.name, "Demo Project");
        assert_eq!(project.scheduler, TlpfileGraphicsScheduler::Gms);
        assert_eq!(project.gms_scaler.mode, GmsScalerMode::Adaptive);
        assert_eq!(project.gms_scaler.target_fps, 75);
        assert_eq!(project.gms_scaler.min_physics_budget_pct, 40);
        assert_eq!(project.gms_scaler.budgets.render_budget_pct, 30);
        assert_eq!(project.gms_scaler.budgets.physics_budget_pct, 40);
        assert_eq!(project.gms_scaler.budgets.ai_ml_budget_pct, 20);
        assert_eq!(project.gms_scaler.budgets.postfx_budget_pct, 10);
        assert_eq!(project.default_dimension, TlpfileSceneDimension::ThreeD);
        assert_eq!(project.default_scene, "main");
        assert_eq!(project.scenes.len(), 1);
        assert_eq!(project.scenes[0].dimension, TlpfileSceneDimension::TwoD);
        assert_eq!(project.scenes[0].tljoint.as_deref(), Some("main.tljoint"));
    }

    #[test]
    fn compiles_scene_with_direct_assets() {
        let dir = unique_temp_dir("compile");
        let manifest = dir.join("demo.tlpfile");
        let script = dir.join("demo.tlscript");
        let sprite = dir.join("demo.tlsprite");

        std::fs::write(
            &manifest,
            concat!(
                "tlpfile_v1\n",
                "[project]\n",
                "name = Compile Demo\n",
                "default_scene = demo\n",
                "[scene.demo]\n",
                "tlscript = demo.tlscript\n",
                "tlsprite = demo.tlsprite\n",
            ),
        )
        .expect("write manifest");
        std::fs::write(
            &script,
            concat!(
                "@export\n",
                "def showcase_tick(frame: int, live_balls: int, spawned_this_tick: int, key_f_down: bool, input_move_x: float, input_move_y: float, input_move_z: float, input_look_dx: float, input_look_dy: float, input_sprint_down: bool, input_look_active: bool, input_reset_camera: bool):\n",
                "    set_spawn_per_tick(42)\n",
            ),
        )
        .expect("write script");
        std::fs::write(
            &sprite,
            concat!(
                "tlsprite_v1\n",
                "[hud.demo]\n",
                "sprite_id = 7\n",
                "kind = hud\n",
                "texture_slot = 0\n",
                "layer = 100\n",
                "position = 0.0, 0.0, 0.0\n",
                "size = 0.2, 0.1\n",
                "rotation_rad = 0.0\n",
                "color = 1.0, 1.0, 1.0, 1.0\n",
            ),
        )
        .expect("write sprite");

        let out = compile_tlpfile_scene_from_path(
            &manifest,
            Some("demo"),
            TlscriptShowcaseConfig::default(),
        );
        assert!(!out.has_errors(), "{:?}", out.diagnostics);
        let bundle = out.bundle.expect("bundle");
        assert_eq!(bundle.scripts.len(), 1);
        assert_eq!(bundle.sprite_count(), 1);
    }

    #[test]
    fn parses_project_scheduler_override() {
        let source = concat!(
            "tlpfile_v1\n",
            "[project]\n",
            "name = Scheduler Demo\n",
            "scheduler = mgs\n",
            "default_scene = demo\n",
            "[scene.demo]\n",
            "tlscript = demo.tlscript\n",
        );
        let out = parse_tlpfile(source);
        assert!(!out.has_errors(), "{:?}", out.diagnostics);
        let project = out.project.expect("project");
        assert_eq!(project.scheduler, TlpfileGraphicsScheduler::Mgs);
    }

    #[test]
    fn parses_project_scheduler_auto() {
        let source = concat!(
            "tlpfile_v1\n",
            "[project]\n",
            "name = Scheduler Auto Demo\n",
            "scheduler = auto\n",
            "default_scene = demo\n",
            "[scene.demo]\n",
            "tlscript = demo.tlscript\n",
        );
        let out = parse_tlpfile(source);
        assert!(!out.has_errors(), "{:?}", out.diagnostics);
        let project = out.project.expect("project");
        assert_eq!(project.scheduler, TlpfileGraphicsScheduler::Auto);
    }

    #[test]
    fn uses_project_default_dimension_for_scenes() {
        let source = concat!(
            "tlpfile_v1\n",
            "[project]\n",
            "name = Scene Mode Defaults\n",
            "default_dimension = 2d\n",
            "default_scene = demo\n",
            "[scene.demo]\n",
            "tlscript = demo.tlscript\n",
        );
        let out = parse_tlpfile(source);
        assert!(!out.has_errors(), "{:?}", out.diagnostics);
        let project = out.project.expect("project");
        assert_eq!(project.default_dimension, TlpfileSceneDimension::TwoD);
        assert_eq!(project.scenes[0].dimension, TlpfileSceneDimension::TwoD);
    }

    #[test]
    fn compile_outcome_exposes_scene_dimension() {
        let dir = unique_temp_dir("dimension");
        let manifest = dir.join("demo.tlpfile");
        let script = dir.join("demo.tlscript");
        let sprite = dir.join("demo.tlsprite");

        std::fs::write(
            &manifest,
            concat!(
                "tlpfile_v1\n",
                "[project]\n",
                "name = Dimension Demo\n",
                "default_scene = demo\n",
                "[scene.demo]\n",
                "dimension = 2d\n",
                "tlscript = demo.tlscript\n",
                "tlsprite = demo.tlsprite\n",
            ),
        )
        .expect("write manifest");
        std::fs::write(
            &script,
            concat!(
                "@export\n",
                "def showcase_tick(frame: int, live_balls: int, spawned_this_tick: int, key_f_down: bool, input_move_x: float, input_move_y: float, input_move_z: float, input_look_dx: float, input_look_dy: float, input_sprint_down: bool, input_look_active: bool, input_reset_camera: bool):\n",
                "    set_spawn_per_tick(42)\n",
            ),
        )
        .expect("write script");
        std::fs::write(
            &sprite,
            concat!(
                "tlsprite_v1\n",
                "[hud.demo]\n",
                "sprite_id = 7\n",
                "kind = hud\n",
                "texture_slot = 0\n",
                "layer = 100\n",
                "position = 0.0, 0.0, 0.0\n",
                "size = 0.2, 0.1\n",
                "rotation_rad = 0.0\n",
                "color = 1.0, 1.0, 1.0, 1.0\n",
            ),
        )
        .expect("write sprite");

        let out = compile_tlpfile_scene_from_path(
            &manifest,
            Some("demo"),
            TlscriptShowcaseConfig::default(),
        );
        assert!(!out.has_errors(), "{:?}", out.diagnostics);
        let bundle = out.bundle.expect("bundle");
        assert_eq!(bundle.scene_dimension, TlpfileSceneDimension::TwoD);
        assert_eq!(bundle.gms_scaler.mode, GmsScalerMode::Adaptive);
    }

    #[test]
    fn rejects_main_scene_without_main_joint() {
        let source = concat!(
            "tlpfile_v1\n",
            "[project]\n",
            "name = Main Scene Rule\n",
            "default_scene = main\n",
            "[scene.main]\n",
            "tljoint = bounce_showcase.tljoint\n",
        );
        let out = parse_tlpfile(source);
        assert!(out.has_errors());
    }
}
