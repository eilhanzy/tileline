//! Mobile Graphics Scheduler (MGS)
//!
//! TBDR-aware, serial-first workload scheduling for mobile GPU platforms.
//! Hedef donanım: Adreno (Qualcomm), Mali (ARM).
//!
//! GMS (Graphics Multi Scaler) ile sıfır bağımlılık — MGS tamamen bağımsızdır.
//! GMS paralel, çok-GPU hesaplama için tasarlanmıştır; MGS ise tek-GPU, serial
//! fallback zinciri ve mobil güç/termal kısıtları için yazılmıştır.

/// Canonical module id used by runtime version commands.
pub const MODULE_ID: &str = "mgs";
/// Crate version resolved at compile time.
pub const MODULE_VERSION: &str = env!("CARGO_PKG_VERSION");

pub mod bridge;
pub mod fallback;
pub mod hardware;
pub mod render_benchmark;
pub mod runtime;
pub mod scene_workload;
pub mod tile_planner;
pub mod tuning;
pub mod zram;

pub use bridge::{MgsBridge, MgsBridgePlan, MpsWorkloadHint};
pub use fallback::FallbackChain;
pub use hardware::{GfxBackend, MobileGpuFamily, MobileGpuProfile, TbdrArchitecture};
pub use runtime::{
    choose_pacing_mode, clamp_required_limits_to_supported, is_unified_memory_profile,
    present_mode_allows_uncapped, recommended_min_frame_interval,
    safe_default_required_limits_for_adapter, select_present_mode, select_throughput_burst,
    startup_ramp, AdaptiveBurstController, AggressiveNoVsyncPolicy, DeviceLimitClampReport,
    RuntimeMode, RuntimePacingMode, ThroughputFramePacer, ThroughputMemoryPolicy, VsyncMode,
};
pub use scene_workload::{
    estimate_mps_workload_hint, plan_scene_with_bridge, MobileSceneSnapshot, MobileSceneTuning,
};
pub use tile_planner::MgsPlanner;
pub use tuning::{
    BackendRenderHints, LoadAction, MetalPassHints, MgsTuningProfile, StoreAction, VulkanPassHints,
};
pub use zram::{MpsZramConfig, MpsZramError, MpsZramSpillOutcome, MpsZramSpillPool, MpsZramStats};
