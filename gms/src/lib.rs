//! Graphics Multi Scaler (GMS)
//! GPU-side workload discovery, scoring, and proportional dispatch planning.

pub mod adaptive_buffer;
pub mod bridge;
pub mod hardware;
pub mod multi_gpu_runtime;
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
    ComputeUnitEstimateSource, ComputeUnitKind, GpuAdapterProfile, GpuInventory, GpuScoreBreakdown,
    MemoryTopology,
};
pub use multi_gpu_runtime::{
    MultiGpuExecutor, MultiGpuExecutorConfig, MultiGpuExecutorSummary, MultiGpuInitPolicy,
};
pub use tuning::GmsRuntimeTuningProfile;
