//! Graphics Multi Scaler (GMS)
//! GPU-side workload discovery, scoring, and proportional dispatch planning.
//!
//! This crate provides the GPU-side half of Tileline's scaling stack:
//! - adapter inventory and heuristic/native hardware profiling
//! - single- and multi-GPU workload planning
//! - portable helper runtime for explicit secondary-GPU bring-up
//! - UMA/Apple Silicon adaptive buffer regulation
//! - runtime tuning profiles shared by benchmarks and engine runtime code

pub mod adaptive_buffer;
pub mod bridge;
pub mod hardware;
pub mod multi_gpu_runtime;
pub mod render_benchmark;
pub mod tuning;

pub use adaptive_buffer::{
    AdaptiveBuffer, AdaptiveBufferConfig, AdaptiveBufferDecision, AdaptiveBufferMode,
    AdaptiveFrameTelemetry, SharedBufferKey, SharedBufferLease, SharedBufferLockError,
    SharedBufferOwner,
};
pub use bridge::{
    DispatchPlan, GmsDispatcher, GpuWorkAssignment, MultiGpuDispatchPlan, MultiGpuDispatcher,
    MultiGpuLaneAssignment, MultiGpuRole, MultiGpuSyncPlan, MultiGpuWorkloadRequest,
    SharedTextureBridgePlan, SharedTransferKind, SyncEquivalent, TaskClass, WorkloadRequest,
    ZeroCopyBufferPlan,
};
pub use hardware::{
    clamp_required_limits_to_supported, safe_default_required_limits_for_adapter,
    ComputeUnitEstimateSource, ComputeUnitKind, DeviceLimitClampReport, GpuAdapterProfile,
    GpuInventory, GpuScoreBreakdown, MemoryTopology,
};
pub use multi_gpu_runtime::{
    MultiGpuExecutor, MultiGpuExecutorConfig, MultiGpuExecutorSummary, MultiGpuFrameSubmitResult,
    MultiGpuInitPolicy,
};
pub use tuning::GmsRuntimeTuningProfile;
