//! Mobile Graphics Scheduler (MGS)
//!
//! TBDR-aware, serial-first workload scheduling for mobile GPU platforms.
//! Hedef donanım: Adreno (Qualcomm), Mali (ARM).
//!
//! GMS (Graphics Multi Scaler) ile sıfır bağımlılık — MGS tamamen bağımsızdır.
//! GMS paralel, çok-GPU hesaplama için tasarlanmıştır; MGS ise tek-GPU, serial
//! fallback zinciri ve mobil güç/termal kısıtları için yazılmıştır.

pub mod bridge;
pub mod fallback;
pub mod hardware;
pub mod render_benchmark;
pub mod runtime;
pub mod tile_planner;
pub mod tuning;

pub use bridge::MgsBridge;
pub use fallback::FallbackChain;
pub use hardware::{GfxBackend, MobileGpuFamily, MobileGpuProfile, TbdrArchitecture};
pub use runtime::{
    choose_pacing_mode, is_unified_memory_profile, present_mode_allows_uncapped,
    recommended_min_frame_interval, select_present_mode, select_throughput_burst, startup_ramp,
    AdaptiveBurstController, AggressiveNoVsyncPolicy, RuntimeMode, RuntimePacingMode,
    ThroughputMemoryPolicy, VsyncMode,
};
pub use tile_planner::MgsPlanner;
pub use tuning::{
    BackendRenderHints, LoadAction, MetalPassHints, MgsTuningProfile, StoreAction, VulkanPassHints,
};
