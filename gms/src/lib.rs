//! Graphics Multi Scaler (GMS)
//! GPU-side workload discovery, scoring, and proportional dispatch planning.

pub mod bridge;
pub mod hardware;

pub use bridge::{
    DispatchPlan, GmsDispatcher, GpuWorkAssignment, WorkloadRequest, ZeroCopyBufferPlan,
};
pub use hardware::{
    ComputeUnitKind, GpuAdapterProfile, GpuInventory, GpuScoreBreakdown, MemoryTopology,
};
