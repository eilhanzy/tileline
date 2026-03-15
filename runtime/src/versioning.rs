/// Canonical engine id surfaced in runtime version commands.
pub const ENGINE_ID: &str = "tileline";
/// Runtime crate version is treated as the engine release version.
pub const ENGINE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Canonical runtime module id.
pub const RUNTIME_MODULE_ID: &str = "runtime";
/// Runtime crate version resolved at compile time.
pub const RUNTIME_MODULE_VERSION: &str = env!("CARGO_PKG_VERSION");

/// Single module version entry used by `version` commands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TilelineVersionEntry {
    pub module: &'static str,
    pub version: &'static str,
}

/// Version table for all engine modules linked into runtime.
pub const VERSION_ENTRIES: &[TilelineVersionEntry] = &[
    TilelineVersionEntry {
        module: ENGINE_ID,
        version: ENGINE_VERSION,
    },
    TilelineVersionEntry {
        module: RUNTIME_MODULE_ID,
        version: RUNTIME_MODULE_VERSION,
    },
    TilelineVersionEntry {
        module: tl_core::MODULE_ID,
        version: tl_core::MODULE_VERSION,
    },
    TilelineVersionEntry {
        module: mps::MODULE_ID,
        version: mps::MODULE_VERSION,
    },
    TilelineVersionEntry {
        module: gms::MODULE_ID,
        version: gms::MODULE_VERSION,
    },
    TilelineVersionEntry {
        module: mgs::MODULE_ID,
        version: mgs::MODULE_VERSION,
    },
    TilelineVersionEntry {
        module: nps::MODULE_ID,
        version: nps::MODULE_VERSION,
    },
    TilelineVersionEntry {
        module: paradoxpe::MODULE_ID,
        version: paradoxpe::MODULE_VERSION,
    },
];

/// Returns all known module version entries.
pub fn tileline_version_entries() -> &'static [TilelineVersionEntry] {
    VERSION_ENTRIES
}

fn normalize_query(raw: &str) -> String {
    raw.trim()
        .to_ascii_lowercase()
        .chars()
        .filter(|ch| *ch != '-' && *ch != '_' && !ch.is_whitespace())
        .collect()
}

/// Resolves `version <module>` aliases to a concrete entry.
pub fn resolve_tileline_version_query(query: &str) -> Option<TilelineVersionEntry> {
    let normalized = normalize_query(query);
    if normalized.is_empty() || normalized == "all" {
        return None;
    }

    let key = match normalized.as_str() {
        "engine" | "tileline" => ENGINE_ID,
        "rt" | "runtime" => RUNTIME_MODULE_ID,
        "core" | "tlcore" => tl_core::MODULE_ID,
        "mps" => mps::MODULE_ID,
        "gms" => gms::MODULE_ID,
        "mgs" => mgs::MODULE_ID,
        "nps" => nps::MODULE_ID,
        "paradoxpe" | "ppe" => paradoxpe::MODULE_ID,
        _ => return None,
    };

    VERSION_ENTRIES
        .iter()
        .find(|entry| entry.module == key)
        .copied()
}
