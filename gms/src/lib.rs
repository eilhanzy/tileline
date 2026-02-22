//! Graphics Multi Scaler (GMS)
//! GPU-side workload discovery, scoring, and proportional dispatch planning.

pub mod bridge;
pub mod hardware;
pub mod multi_gpu_runtime;

pub use bridge::{
    DispatchPlan, GmsDispatcher, GpuWorkAssignment, MultiGpuDispatchPlan, MultiGpuDispatcher,
    MultiGpuLaneAssignment, MultiGpuRole, MultiGpuSyncPlan, MultiGpuWorkloadRequest,
    SharedTextureBridgePlan, SharedTransferKind, SyncEquivalent, TaskClass, WorkloadRequest,
    ZeroCopyBufferPlan,
};
pub use hardware::{
    ComputeUnitKind, GpuAdapterProfile, GpuInventory, GpuScoreBreakdown, MemoryTopology,
};
pub use multi_gpu_runtime::{
    MultiGpuExecutor, MultiGpuExecutorConfig, MultiGpuExecutorSummary, MultiGpuInitPolicy,
};
