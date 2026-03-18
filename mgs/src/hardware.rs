//! TBDR-aware mobile GPU detection.
//!
//! Detects GPU families using Tile-Based Deferred Rendering (TBDR) architecture
//! and extracts their profiles.
//!
//! Supported families:
//! - **Adreno** (Qualcomm Snapdragon) — Flex Render (hibrit IMR/TBDR)
//! - **Mali** (ARM) — pure TBDR (Bifrost, Valhall, 5th gen)
//! - **PowerVR** (Imagination Technologies) — pure TBDR
//! - **Apple GPU** — TBDR (A/M series SoC)
//!
//! No dependency on `wgpu` or platform APIs; uses name-based
//! heuristic matching.

/// TBDR hardware architecture classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TbdrArchitecture {
    /// Qualcomm Adreno — Flex Render (hibrit IMR/TBDR).
    /// Tile size is usually 32×32 or 64×64 pixels.
    FlexRender,
    /// ARM Mali Bifrost / Valhall / 5th gen — pure TBDR.
    /// Tile size is usually 16×16 pixels.
    MaliTbdr,
    /// Imagination PowerVR — pure TBDR.
    PowerVrTbdr,
    /// Apple GPU (A/M series) — TBDR.
    AppleTbdr,
    /// Unknown or hardware not supporting TBDR.
    Unknown,
}

impl TbdrArchitecture {
    /// Tile size recommendation (square edge in pixels).
    /// Scheduler uses this value as a basis for tile budget calculations.
    pub fn recommended_tile_px(self) -> u32 {
        match self {
            Self::FlexRender => 32,
            Self::MaliTbdr => 16,
            Self::PowerVrTbdr => 32,
            Self::AppleTbdr => 32,
            Self::Unknown => 16,
        }
    }

    /// True when the architecture benefits from explicit tile budget constraints.
    pub fn is_tbdr(self) -> bool {
        !matches!(self, Self::Unknown)
    }
}

/// Mobile GPU family (manufacturer-based classification).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MobileGpuFamily {
    /// Qualcomm Snapdragon SoC — Adreno series.
    Adreno,
    /// ARM — Mali Bifrost, Valhall or 5th gen.
    Mali,
    /// Imagination Technologies — PowerVR series.
    PowerVr,
    /// Apple Silicon — A/M series integrated GPU.
    Apple,
    /// Unidentified or desktop GPU.
    Unknown,
}

impl MobileGpuFamily {
    /// Short label (for log and summary screens).
    pub fn short_label(self) -> &'static str {
        match self {
            Self::Adreno => "Adreno",
            Self::Mali => "Mali",
            Self::PowerVr => "PowerVR",
            Self::Apple => "Apple GPU",
            Self::Unknown => "Unknown",
        }
    }
}

/// Graphics backend classification.
///
/// There are two primary backends on mobile platforms:
/// - **Vulkan** — Android (Adreno, Mali, PowerVR).
/// - **Metal** — iOS / macOS (Apple Silicon, A series).
///
/// Backend directly affects TBDR render pass semantics:
/// Vulkan'da `loadOp`/`storeOp`; Metal'de `MTLLoadAction`/`MTLStoreAction`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GfxBackend {
    /// Vulkan — Android-based mobile devices.
    Vulkan,
    /// Metal — iOS / macOS (Apple Silicon, A series).
    Metal,
    /// Backend unclear or not detected.
    Unknown,
}

impl GfxBackend {
    /// Short label.
    pub fn label(self) -> &'static str {
        match self {
            Self::Vulkan => "Vulkan",
            Self::Metal => "Metal",
            Self::Unknown => "Unknown",
        }
    }

    /// True when backend supports explicit subpass dependencies (Vulkan-only).
    pub fn has_subpass_dependencies(self) -> bool {
        matches!(self, Self::Vulkan)
    }

    /// True when backend supports tile shaders / imageblocks (Metal A11+).
    pub fn has_tile_shaders(self) -> bool {
        matches!(self, Self::Metal)
    }
}

/// Detected mobile GPU profile.
#[derive(Debug, Clone)]
pub struct MobileGpuProfile {
    /// Raw adapter/device name (obtained from platform API).
    pub name: String,
    /// Detected GPU family.
    pub family: MobileGpuFamily,
    /// TBDR architecture.
    pub architecture: TbdrArchitecture,
    /// Graphics backend (Vulkan or Metal).
    pub gfx_backend: GfxBackend,
    /// Estimated shader core / execution unit count.
    pub estimated_cores: u32,
    /// Estimated shared bandwidth (GB/s).
    pub estimated_bandwidth_gbps: f32,
    /// Estimated on-chip tile memory capacity (KB).
    pub tile_memory_kb: u32,
}

impl MobileGpuProfile {
    /// Extracts profile using name-based heuristic matching.
    ///
    /// Pass the platform GPU name converted to lowercase.
    pub fn detect(name: &str) -> Self {
        let n = name.to_ascii_lowercase();
        let family = classify_family(&n);
        let architecture = classify_architecture(family);
        let gfx_backend = infer_backend(family);
        let estimated_cores = estimate_cores(&n, family);
        let estimated_bandwidth_gbps = estimate_bandwidth(&n, family, estimated_cores);
        let tile_memory_kb = estimate_tile_memory_kb(family, estimated_cores);

        Self {
            name: name.to_owned(),
            family,
            architecture,
            gfx_backend,
            estimated_cores,
            estimated_bandwidth_gbps,
            tile_memory_kb,
        }
    }

    /// True when the profile represents a known mobile TBDR target.
    pub fn is_mobile_tbdr(&self) -> bool {
        self.architecture.is_tbdr() && !matches!(self.family, MobileGpuFamily::Unknown)
    }

    /// Returns true if the profile is actually a desktop/laptop class TBDR, like Apple M series.
    /// Runtime can use this flag to route the device to GMS instead of MGS.
    pub fn is_desktop_class(&self) -> bool {
        let n = self.name.to_ascii_lowercase();
        matches!(self.family, MobileGpuFamily::Apple) 
            && (n.contains(" m1") || n.contains(" m2") || n.contains(" m3")
                || n.contains(" m4") || n.contains(" m5") || n.contains(" m6")
                || n.contains(" m7") || n.contains(" m8") || n.contains(" m9")
                || n.starts_with("m1") || n.starts_with("m2") || n.starts_with("m3")
                || n.starts_with("m4") || n.starts_with("m5") || n.starts_with("m6")
                || n.starts_with("m7") || n.starts_with("m8") || n.starts_with("m9"))
    }
}

fn classify_family(name: &str) -> MobileGpuFamily {
    if name.contains("adreno") || name.contains("qualcomm") {
        MobileGpuFamily::Adreno
    } else if name.contains("mali") || name.contains("bifrost") || name.contains("valhall") {
        MobileGpuFamily::Mali
    } else if name.contains("powervr") || name.contains("imagination") {
        MobileGpuFamily::PowerVr
    } else if name.contains("apple")
        || name.contains(" m1")
        || name.contains(" m2")
        || name.contains(" m3")
        || name.contains(" m4")
        || name.contains(" m5")
    {
        MobileGpuFamily::Apple
    } else {
        MobileGpuFamily::Unknown
    }
}

fn infer_backend(family: MobileGpuFamily) -> GfxBackend {
    match family {
        MobileGpuFamily::Apple => GfxBackend::Metal,
        MobileGpuFamily::Adreno | MobileGpuFamily::Mali | MobileGpuFamily::PowerVr => {
            GfxBackend::Vulkan
        }
        MobileGpuFamily::Unknown => GfxBackend::Unknown,
    }
}

fn classify_architecture(family: MobileGpuFamily) -> TbdrArchitecture {
    match family {
        MobileGpuFamily::Adreno => TbdrArchitecture::FlexRender,
        MobileGpuFamily::Mali => TbdrArchitecture::MaliTbdr,
        MobileGpuFamily::PowerVr => TbdrArchitecture::PowerVrTbdr,
        MobileGpuFamily::Apple => TbdrArchitecture::AppleTbdr,
        MobileGpuFamily::Unknown => TbdrArchitecture::Unknown,
    }
}

fn estimate_cores(name: &str, family: MobileGpuFamily) -> u32 {
    match family {
        MobileGpuFamily::Adreno => parse_adreno_cores(name),
        MobileGpuFamily::Mali => parse_mali_cores(name),
        MobileGpuFamily::Apple => parse_apple_cores(name),
        MobileGpuFamily::PowerVr | MobileGpuFamily::Unknown => 4,
    }
}

fn parse_adreno_cores(name: &str) -> u32 {
    // Adreno shader core count lookup (SPs = Shader Processors).
    // Higher models have more SPs.
    const TABLE: &[(&str, u32)] = &[
        ("adreno 750", 12),
        ("adreno 740", 12),
        ("adreno 735", 8),
        ("adreno 732", 6),
        ("adreno 730", 8),
        ("adreno 725", 6),
        ("adreno 720", 6),
        ("adreno 710", 6),
        ("adreno 702", 4),
        ("adreno 700", 4),
        ("adreno 695", 6),
        ("adreno 690", 6),
        ("adreno 685", 4),
        ("adreno 680", 4),
        ("adreno 650", 4),
        ("adreno 640", 2),
        ("adreno 630", 2),
        ("adreno 620", 2),
        ("adreno 619", 2),
        ("adreno 616", 2),
        ("adreno 612", 2),
        ("adreno 610", 1),
        ("adreno 608", 1),
        ("adreno 605", 1),
    ];
    table_lookup(name, TABLE).unwrap_or(2)
}

fn parse_mali_cores(name: &str) -> u32 {
    // Mali shader core estimate (number of execution engines).
    const TABLE: &[(&str, u32)] = &[
        ("mali-g925", 16),
        ("mali-g720", 16),
        ("mali-g715", 14),
        ("mali-g710", 14),
        ("mali-g710 mc10", 10),
        ("mali-g78", 24),
        ("mali-g77", 16),
        ("mali-g76", 12),
        ("mali-g72", 12),
        ("mali-g71", 8),
        ("mali-g68", 6),
        ("mali-g57", 6),
        ("mali-g52", 4),
        ("mali-g51", 4),
        ("mali-g31", 2),
    ];
    table_lookup(name, TABLE).unwrap_or(4)
}

fn parse_apple_cores(name: &str) -> u32 {
    // Apple GPU core count (shader core kümeleri).
    if name.contains("m4 max") {
        40
    } else if name.contains("m4 pro") {
        20
    } else if name.contains("m4") {
        10
    } else if name.contains("m3 max") {
        40
    } else if name.contains("m3 pro") {
        18
    } else if name.contains("m3") {
        10
    } else if name.contains("m2 ultra") {
        60
    } else if name.contains("m2 max") {
        38
    } else if name.contains("m2 pro") {
        19
    } else if name.contains("m2") {
        10
    } else if name.contains("m1 ultra") {
        64
    } else if name.contains("m1 max") {
        32
    } else if name.contains("m1 pro") {
        16
    } else if name.contains("m1") || name.contains("a15") || name.contains("a16") {
        8
    } else if name.contains("a14") || name.contains("a13") {
        6
    } else if name.contains(" m") || name.starts_with("m") {
        10 // Gelecek/bilinmeyen M serisi (M5 vb.) için güvenli masaüstü tabanı
    } else {
        4
    }
}

fn estimate_bandwidth(name: &str, family: MobileGpuFamily, cores: u32) -> f32 {
    let _ = name;
    let core_factor = cores as f32;
    match family {
        // LPDDR5X tabanlı tahmin (tipik Snapdragon amiral değerleri)
        MobileGpuFamily::Adreno => (core_factor * 8.5).clamp(15.0, 120.0),
        // Mali LPDDR5 tahmin
        MobileGpuFamily::Mali => (core_factor * 6.0).clamp(10.0, 80.0),
        // Apple Unified Memory (yüksek bant genişliği)
        MobileGpuFamily::Apple => (core_factor * 3.5).clamp(60.0, 400.0),
        MobileGpuFamily::PowerVr | MobileGpuFamily::Unknown => 15.0,
    }
}

fn estimate_tile_memory_kb(family: MobileGpuFamily, cores: u32) -> u32 {
    // Tile bellek, on-chip SRAM büyüklüğüne dayalı tahmin.
    // Gerçek değerler donanıma ve sürücüye göre değişir.
    let per_core_kb: u32 = match family {
        MobileGpuFamily::Adreno => 32,
        MobileGpuFamily::Mali => 16,
        MobileGpuFamily::Apple => 64,
        MobileGpuFamily::PowerVr => 32,
        MobileGpuFamily::Unknown => 16,
    };
    per_core_kb * cores.max(1)
}

fn table_lookup(name: &str, table: &[(&str, u32)]) -> Option<u32> {
    table
        .iter()
        .find(|(pattern, _)| name.contains(*pattern))
        .map(|(_, v)| *v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_adreno_750_as_flex_render() {
        let p = MobileGpuProfile::detect("Adreno 750");
        assert_eq!(p.family, MobileGpuFamily::Adreno);
        assert_eq!(p.architecture, TbdrArchitecture::FlexRender);
        assert_eq!(p.gfx_backend, GfxBackend::Vulkan);
        assert_eq!(p.estimated_cores, 12);
    }

    #[test]
    fn detects_mali_g78_as_tbdr() {
        let p = MobileGpuProfile::detect("Mali-G78 MC24");
        assert_eq!(p.family, MobileGpuFamily::Mali);
        assert_eq!(p.architecture, TbdrArchitecture::MaliTbdr);
        assert_eq!(p.gfx_backend, GfxBackend::Vulkan);
        assert_eq!(p.estimated_cores, 24);
    }

    #[test]
    fn detects_apple_as_metal_backend() {
        let p = MobileGpuProfile::detect("Apple M2 Pro");
        assert_eq!(p.gfx_backend, GfxBackend::Metal);
        assert!(p.gfx_backend.has_tile_shaders());
        assert!(!p.gfx_backend.has_subpass_dependencies());
    }

    #[test]
    fn unknown_gpu_not_mobile_tbdr() {
        let p = MobileGpuProfile::detect("NVIDIA GeForce RTX 4090");
        assert!(!p.is_mobile_tbdr());
    }
}
