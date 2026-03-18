//! Mobile-specific tuning parameters.
//!
//! Unlike GMS's `GmsRuntimeTuningProfile`, this module:
//! - Models thermal constraints (throttling).
//! - Considers power budget limitations.
//! - Includes render pass and tile flush strategies for TBDR pipeline.
//! - Does not depend on wgpu or any other GPU API.

use crate::hardware::{GfxBackend, MobileGpuFamily, MobileGpuProfile, TbdrArchitecture};

/// Instant thermal / power state of the device.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThermalState {
    /// Normal operation — full performance.
    Nominal,
    /// Slight warming — power limited mode.
    Warm,
    /// Critical — heavy throttling active.
    Hot,
    /// Thermal state unknown.
    Unknown,
}

impl ThermalState {
    /// Performance multiplier based on the current thermal state (0.0 – 1.0).
    pub fn performance_factor(self) -> f32 {
        match self {
            Self::Nominal => 1.00,
            Self::Warm => 0.75,
            Self::Hot => 0.45,
            Self::Unknown => 0.80,
        }
    }
}

/// Render pass strategy: TBDR tile flush timing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderPassStrategy {
    /// Her frame sonunda tek flush — TBDR için ideal.
    SinglePassPerFrame,
    /// Güç tasarrufu: birden fazla pass birleştirilir.
    MergedPasses,
    /// Her tile bitiminde hemen flush — düşük bellekli cihazlar için.
    EagerFlush,
}

// ─── Backend-spesifik render pass ipuçları ────────────────────────────────────

/// Attachment yükleme aksiyonu.
///
/// Vulkan `VkAttachmentLoadOp` ve Metal `MTLLoadAction` ikisine de karşılık gelir.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadAction {
    /// Attachment'ı clear değeriyle doldur.
    /// Vulkan: `LOAD_OP_CLEAR` / Metal: `MTLLoadActionClear`
    Clear,
    /// Önceki içeriği koru (pahalı — TBDR'da kaçın).
    /// Vulkan: `LOAD_OP_LOAD` / Metal: `MTLLoadActionLoad`
    Load,
    /// Önceki içerik önemsiz; GPU serbest bırakabilir.
    /// Vulkan: `LOAD_OP_DONT_CARE` / Metal: `MTLLoadActionDontCare`
    DontCare,
}

/// Attachment depolama aksiyonu.
///
/// TBDR'da `DontCare` kritik: derinlik/stencil tile'dan DRAM'e yazılmaz,
/// bant genişliği büyük ölçüde azalır.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StoreAction {
    /// Sonucu belleğe yaz.
    /// Vulkan: `STORE_OP_STORE` / Metal: `MTLStoreActionStore`
    Store,
    /// Sonucu at; geçici attachment için (depth/stencil genellikle bu).
    /// Vulkan: `STORE_OP_DONT_CARE` / Metal: `MTLStoreActionDontCare`
    DontCare,
    /// Çözümlenmiş multisampled sonucu depola, çözümsüz veriyi at.
    /// Vulkan: `STORE_OP_STORE` + resolve attachment / Metal: `MTLStoreActionMultisampleResolve`
    MultisampleResolve,
}

/// Vulkan render pass için TBDR-aware ipuçları.
///
/// Bu değerler `VkRenderPassCreateInfo` / `VkAttachmentDescription`'ı doldurmak
/// için referans olarak kullanılır; MGS doğrudan Vulkan çağrısı yapmaz.
#[derive(Debug, Clone, Copy)]
pub struct VulkanPassHints {
    /// Renk attachment yükleme aksiyonu.
    pub color_load: LoadAction,
    /// Renk attachment depolama aksiyonu.
    pub color_store: StoreAction,
    /// Derinlik attachment yükleme aksiyonu.
    pub depth_load: LoadAction,
    /// Derinlik attachment depolama aksiyonu.
    /// TBDR bandwidth tasarrufu için genellikle `DontCare`.
    pub depth_store: StoreAction,
    /// Stencil attachment depolama aksiyonu.
    pub stencil_store: StoreAction,
    /// Subpass input attachment kullan (deferred shading için).
    /// `VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT` yerine texture sampling daha ucuz
    /// olmadığı durumlarda true.
    pub prefer_input_attachments: bool,
    /// Önerilen subpass sayısı (1 = tek pass, 2 = geometry + lighting).
    pub recommended_subpass_count: u32,
}

/// Metal render pass için TBDR-aware ipuçları.
///
/// Bu değerler `MTLRenderPassDescriptor` ayarlarına rehberlik eder.
#[derive(Debug, Clone, Copy)]
pub struct MetalPassHints {
    /// Renk attachment yükleme aksiyonu.
    pub color_load: LoadAction,
    /// Renk attachment depolama aksiyonu.
    pub color_store: StoreAction,
    /// Derinlik attachment yükleme aksiyonu.
    pub depth_load: LoadAction,
    /// Derinlik attachment depolama aksiyonu (`DontCare` = memoryless).
    pub depth_store: StoreAction,
    /// Tile shader / imageblock kullan (A11 Bionic veya M serisi gerektirir).
    /// `MTLRenderPipelineDescriptor.tileFunction` etkinleştirme sinyali.
    pub use_tile_shaders: bool,
    /// Geçici depth/stencil buffer'ı `MTLStorageModeMemoryless` olarak oluştur.
    pub memoryless_depth: bool,
}

/// Backend'e özgü render pass ipuçları.
#[derive(Debug, Clone, Copy)]
pub enum BackendRenderHints {
    Vulkan(VulkanPassHints),
    Metal(MetalPassHints),
    /// Backend bilinmiyor — güvenli varsayılanlar kullan.
    Unknown,
}

impl BackendRenderHints {
    /// Verilen backend ve donanım mimarisine göre varsayılan ipuçlarını üretir.
    pub fn from_backend(backend: GfxBackend, arch: TbdrArchitecture) -> Self {
        match backend {
            GfxBackend::Vulkan => Self::Vulkan(vulkan_hints(arch)),
            GfxBackend::Metal => Self::Metal(metal_hints(arch)),
            GfxBackend::Unknown => Self::Unknown,
        }
    }
}

fn vulkan_hints(arch: TbdrArchitecture) -> VulkanPassHints {
    // Storing depth/stencil with DONT_CARE in TBDR Vulkan
    // prevents DRAM writing from on-chip tile memory — major bandwidth savings.
    let prefer_input = arch.is_tbdr(); // input attachment subpass is free in TBDR
    let subpass_count = if arch.is_tbdr() { 2 } else { 1 };
    VulkanPassHints {
        color_load: LoadAction::Clear,
        color_store: StoreAction::Store,
        depth_load: LoadAction::Clear,
        depth_store: StoreAction::DontCare, // ← TBDR critical optimization
        stencil_store: StoreAction::DontCare,
        prefer_input_attachments: prefer_input,
        recommended_subpass_count: subpass_count,
    }
}

fn metal_hints(arch: TbdrArchitecture) -> MetalPassHints {
    // Tile shader support in Metal exists in Apple A11+ and all M series.
    // Memoryless depth completely removes DRAM allocation for depth attachment.
    let tile_shaders = matches!(arch, TbdrArchitecture::AppleTbdr);
    MetalPassHints {
        color_load: LoadAction::Clear,
        color_store: StoreAction::Store,
        depth_load: LoadAction::Clear,
        depth_store: StoreAction::DontCare, // ← combines with memoryless
        use_tile_shaders: tile_shaders,
        memoryless_depth: tile_shaders,
    }
}

/// MGS mobile tuning profile.
#[derive(Debug, Clone)]
pub struct MgsTuningProfile {
    /// Thermal state of the device.
    pub thermal_state: ThermalState,
    /// Are we operating in power saving mode?
    pub power_saving_mode: bool,
    /// Tile size override (hardware recommendation is used if None).
    pub tile_size_override_px: Option<u32>,
    /// Render pass strategy.
    pub render_pass_strategy: RenderPassStrategy,
    /// Maximum draw calls per frame (0 = unlimited).
    pub max_draw_calls_per_frame: u32,
    /// Vertex batch size (optimal vertex group per tile).
    pub vertex_batch_size: u32,
    /// Maximum bandwidth usage per tile (MB).
    pub tile_bandwidth_budget_mb: f32,
    /// Recommended MSAA sample count for the framebuffer (1 = MSAA off).
    pub recommended_msaa_samples: u32,
    /// Backend-specific render pass hints (Vulkan / Metal).
    pub backend_hints: BackendRenderHints,
}

impl MgsTuningProfile {
    /// Derives default tuning parameters from hardware profile.
    pub fn from_profile(profile: &MobileGpuProfile) -> Self {
        let (render_pass_strategy, vertex_batch_size, recommended_msaa_samples) =
            arch_defaults(profile.architecture);

        let tile_bandwidth_budget_mb =
            family_bandwidth_budget(profile.family, profile.estimated_cores);
        let backend_hints =
            BackendRenderHints::from_backend(profile.gfx_backend, profile.architecture);

        let max_draw_calls_per_frame = match profile.family {
            // Keep limit very high for all desktop-class Apple M-series chips
            MobileGpuFamily::Apple if profile.is_desktop_class() => 8192,
            MobileGpuFamily::Apple => 2048, // For A-series chips
            _ => 512, // Standard mobile GPUs
        };

        Self {
            thermal_state: ThermalState::Unknown,
            power_saving_mode: false,
            tile_size_override_px: None,
            render_pass_strategy,
            max_draw_calls_per_frame,
            vertex_batch_size,
            tile_bandwidth_budget_mb,
            recommended_msaa_samples,
            backend_hints,
        }
    }

    /// Updates thermal state and sets parameters accordingly.
    pub fn apply_thermal_state(mut self, state: ThermalState) -> Self {
        self.thermal_state = state;
        let factor = state.performance_factor();
        self.max_draw_calls_per_frame =
            ((self.max_draw_calls_per_frame as f32) * factor).round() as u32;
        self.tile_bandwidth_budget_mb *= factor;
        if matches!(state, ThermalState::Hot) {
            self.power_saving_mode = true;
            self.render_pass_strategy = RenderPassStrategy::MergedPasses;
            self.recommended_msaa_samples = 1;
        }
        self
    }

    /// Effective bandwidth budget per tile (MB).
    pub fn effective_tile_bandwidth_mb(&self) -> f32 {
        self.tile_bandwidth_budget_mb * self.thermal_state.performance_factor()
    }
}

fn arch_defaults(arch: TbdrArchitecture) -> (RenderPassStrategy, u32, u32) {
    match arch {
        TbdrArchitecture::FlexRender => {
            // Adreno — Flex Render sometimes manages multiple passes well.
            (RenderPassStrategy::SinglePassPerFrame, 256, 4)
        }
        TbdrArchitecture::MaliTbdr => {
            // Mali — pure TBDR; single pass is the most efficient way.
            (RenderPassStrategy::SinglePassPerFrame, 128, 4)
        }
        TbdrArchitecture::PowerVrTbdr => (RenderPassStrategy::SinglePassPerFrame, 128, 4),
        TbdrArchitecture::AppleTbdr => {
            // Apple GPU — large tile memory capacity; tolerant to MSAA.
            (RenderPassStrategy::MergedPasses, 512, 4)
        }
        TbdrArchitecture::Unknown => {
            // Unknown: safe, low profile.
            (RenderPassStrategy::EagerFlush, 64, 1)
        }
    }
}

fn family_bandwidth_budget(family: MobileGpuFamily, cores: u32) -> f32 {
    let core_f = cores as f32;
    match family {
        MobileGpuFamily::Adreno => (core_f * 2.5).clamp(8.0, 48.0),
        MobileGpuFamily::Mali => (core_f * 1.5).clamp(4.0, 32.0),
        MobileGpuFamily::Apple => (core_f * 4.0).clamp(32.0, 256.0),
        MobileGpuFamily::PowerVr | MobileGpuFamily::Unknown => 8.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::MobileGpuProfile;

    #[test]
    fn hot_thermal_state_reduces_draw_calls() {
        let profile = MobileGpuProfile::detect("Adreno 750");
        let tuning =
            MgsTuningProfile::from_profile(&profile).apply_thermal_state(ThermalState::Hot);
        // 512 * 0.45 = 230 (including rounding)
        assert!(tuning.max_draw_calls_per_frame < 512);
        assert!(tuning.power_saving_mode);
    }

    #[test]
    fn nominal_thermal_state_keeps_full_budget() {
        let profile = MobileGpuProfile::detect("Mali-G78 MC24");
        let tuning =
            MgsTuningProfile::from_profile(&profile).apply_thermal_state(ThermalState::Nominal);
        assert_eq!(tuning.max_draw_calls_per_frame, 512);
    }

    #[test]
    fn apple_profile_uses_merged_passes() {
        let profile = MobileGpuProfile::detect("Apple M2 Pro");
        let tuning = MgsTuningProfile::from_profile(&profile);
        assert_eq!(
            tuning.render_pass_strategy,
            RenderPassStrategy::MergedPasses
        );
    }

    #[test]
    fn vulkan_hints_use_depth_dont_care() {
        let profile = MobileGpuProfile::detect("Adreno 750");
        let tuning = MgsTuningProfile::from_profile(&profile);
        if let BackendRenderHints::Vulkan(hints) = tuning.backend_hints {
            assert_eq!(hints.depth_store, StoreAction::DontCare);
            assert!(hints.prefer_input_attachments);
            assert_eq!(hints.recommended_subpass_count, 2);
        } else {
            panic!("Adreno için Vulkan backend hints bekleniyor");
        }
    }

    #[test]
    fn metal_hints_enable_tile_shaders_for_apple() {
        let profile = MobileGpuProfile::detect("Apple M3");
        let tuning = MgsTuningProfile::from_profile(&profile);
        if let BackendRenderHints::Metal(hints) = tuning.backend_hints {
            assert!(hints.use_tile_shaders);
            assert!(hints.memoryless_depth);
            assert_eq!(hints.depth_store, StoreAction::DontCare);
        } else {
            panic!("Apple için Metal backend hints bekleniyor");
        }
    }

    #[test]
    fn mali_vulkan_hints_have_two_subpasses() {
        let profile = MobileGpuProfile::detect("Mali-G78 MC24");
        let tuning = MgsTuningProfile::from_profile(&profile);
        if let BackendRenderHints::Vulkan(hints) = tuning.backend_hints {
            assert_eq!(hints.recommended_subpass_count, 2);
        } else {
            panic!("Mali için Vulkan backend hints bekleniyor");
        }
    }
}
