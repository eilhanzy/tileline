//! Lightweight `.tlsprite` list-mode editor primitives for Alpha tooling.
//!
//! This module keeps editing logic outside benchmarks/examples so CLI and UI frontends can share
//! the same parse/list/update path.

use std::fmt::Write;
use std::path::Path;

use crate::tlsprite::{
    compile_tlsprite, compile_tlsprite_with_extra_roots, TlspriteCompileOutcome,
    TlspriteDiagnostic, TlspriteDiagnosticLevel, TlspriteProgram, TlspriteSpriteDef,
};
use crate::SpriteKind;

/// Built-in editor theme presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TlspriteEditorTheme {
    /// Dark lavender palette for the Alpha branch.
    DarkLavender,
}

/// RGBA palette used by editor surfaces/widgets.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TlspriteEditorPalette {
    pub window_bg: [f32; 4],
    pub panel_bg: [f32; 4],
    pub panel_alt_bg: [f32; 4],
    pub border: [f32; 4],
    pub text_primary: [f32; 4],
    pub text_muted: [f32; 4],
    pub accent: [f32; 4],
}

impl TlspriteEditorTheme {
    pub fn palette(self) -> TlspriteEditorPalette {
        match self {
            Self::DarkLavender => TlspriteEditorPalette {
                window_bg: [0.102, 0.086, 0.145, 1.0],
                panel_bg: [0.131, 0.109, 0.192, 1.0],
                panel_alt_bg: [0.166, 0.138, 0.238, 1.0],
                border: [0.463, 0.404, 0.639, 1.0],
                text_primary: [0.952, 0.936, 0.995, 1.0],
                text_muted: [0.768, 0.734, 0.891, 1.0],
                accent: [0.620, 0.518, 0.909, 1.0],
            },
        }
    }
}

/// One list row rendered by Alpha `.tlsprite` editor.
#[derive(Debug, Clone, PartialEq)]
pub struct TlspriteListRow {
    pub index: usize,
    pub name: String,
    pub sprite_id: u64,
    pub kind: SpriteKind,
    pub texture_slot: u16,
    pub layer: i16,
    pub position: [f32; 3],
    pub size: [f32; 2],
    pub rotation_rad: f32,
    pub color_rgba: [f32; 4],
    pub fbx_source: Option<String>,
}

impl TlspriteListRow {
    pub fn kind_label(&self) -> &'static str {
        match self.kind {
            SpriteKind::Generic => "generic",
            SpriteKind::Hud => "hud",
            SpriteKind::Camera => "camera",
            SpriteKind::Terrain => "terrain",
            SpriteKind::LightGlow => "light_glow",
        }
    }
}

/// Parsed document model consumed by list-mode editor frontends.
#[derive(Debug, Clone)]
pub struct TlspriteListDocument {
    source: String,
    rows: Vec<TlspriteListRow>,
    diagnostics: Vec<TlspriteDiagnostic>,
}

impl TlspriteListDocument {
    /// Parse and compile a source string into list rows.
    pub fn from_source(source: impl Into<String>) -> Self {
        let source = source.into();
        let outcome = compile_tlsprite(&source);
        Self::from_outcome(source, outcome)
    }

    /// Read source from path and compile into list rows.
    pub fn from_path(path: &Path) -> Result<Self, String> {
        let source = std::fs::read_to_string(path)
            .map_err(|err| format!("failed to read '{}': {err}", path.display()))?;
        let mut roots = Vec::new();
        if let Some(parent) = path.parent() {
            roots.push(parent.to_path_buf());
        }
        let outcome = compile_tlsprite_with_extra_roots(&source, &roots);
        Ok(Self::from_outcome(source, outcome))
    }

    /// Replace source text and recompile.
    pub fn replace_source(&mut self, source: impl Into<String>) {
        let source = source.into();
        let outcome = compile_tlsprite(&source);
        *self = Self::from_outcome(source, outcome);
    }

    /// Returns the original source string.
    pub fn source(&self) -> &str {
        &self.source
    }

    /// Returns list rows for UI/table rendering.
    pub fn rows(&self) -> &[TlspriteListRow] {
        &self.rows
    }

    /// Returns parser diagnostics.
    pub fn diagnostics(&self) -> &[TlspriteDiagnostic] {
        &self.diagnostics
    }

    /// Returns true when diagnostics include at least one error.
    pub fn has_errors(&self) -> bool {
        self.diagnostics
            .iter()
            .any(|d| d.level == TlspriteDiagnosticLevel::Error)
    }

    /// Save source back to disk.
    pub fn save_to_path(&self, path: &Path) -> Result<(), String> {
        std::fs::write(path, &self.source)
            .map_err(|err| format!("failed to write '{}': {err}", path.display()))
    }

    /// Serialize current row view as a markdown table (for docs/console mode).
    pub fn to_markdown_table(&self) -> String {
        let mut out = String::new();
        out.push_str(
            "| # | Section | Kind | Slot | Layer | Pos(x,y,z) | Size(w,h) | FBX |\n\
             |---:|---|---|---:|---:|---|---|---|\n",
        );
        for row in &self.rows {
            let fbx = row.fbx_source.as_deref().unwrap_or("-");
            let _ = writeln!(
                out,
                "| {} | {} | {} | {} | {} | {:.2}, {:.2}, {:.2} | {:.2}, {:.2} | {} |",
                row.index,
                row.name,
                row.kind_label(),
                row.texture_slot,
                row.layer,
                row.position[0],
                row.position[1],
                row.position[2],
                row.size[0],
                row.size[1],
                fbx
            );
        }
        out
    }

    fn from_outcome(source: String, outcome: TlspriteCompileOutcome) -> Self {
        let rows = outcome
            .program
            .as_ref()
            .map(program_to_rows)
            .unwrap_or_default();
        Self {
            source,
            rows,
            diagnostics: outcome.diagnostics,
        }
    }
}

fn program_to_rows(program: &TlspriteProgram) -> Vec<TlspriteListRow> {
    program
        .sprites()
        .iter()
        .enumerate()
        .map(|(index, def)| row_from_sprite(index, def))
        .collect()
}

fn row_from_sprite(index: usize, def: &TlspriteSpriteDef) -> TlspriteListRow {
    TlspriteListRow {
        index,
        name: def.name.clone(),
        sprite_id: def.sprite.sprite_id,
        kind: def.sprite.kind,
        texture_slot: def.sprite.texture_slot,
        layer: def.sprite.layer,
        position: def.sprite.position,
        size: def.sprite.size,
        rotation_rad: def.sprite.rotation_rad,
        color_rgba: def.sprite.color_rgba,
        fbx_source: def.fbx_source.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn parses_rows_and_builds_markdown_table() {
        let source = concat!(
            "tlsprite_v1\n",
            "[camera.icon]\n",
            "sprite_id = 42\n",
            "kind = camera\n",
            "texture_slot = 2\n",
            "layer = 120\n",
            "position = 0.2, 0.3, 0.0\n",
            "size = 0.4, 0.2\n",
            "rotation_rad = 0.0\n",
            "color = 1.0, 1.0, 1.0, 1.0\n",
            "fbx = docs/demos/sphere.fbx\n",
        );

        let document = TlspriteListDocument::from_source(source);
        assert!(!document.has_errors());
        assert_eq!(document.rows().len(), 1);
        assert_eq!(document.rows()[0].kind_label(), "camera");
        assert!(document.to_markdown_table().contains("camera.icon"));
    }

    #[test]
    fn from_path_resolves_relative_fbx_against_file_directory() {
        let file = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../docs/demos/tlapp/bounce_hud.tlsprite");
        let document = TlspriteListDocument::from_path(&file).expect("load tlsprite path");
        assert!(
            !document
                .diagnostics()
                .iter()
                .any(|d| d.message.contains("failed to parse fbx")),
            "relative fbx should resolve from source directory"
        );
    }
}
