//! Multi-Processing Scaler (MPS)
//! ---------------------------------
//! This crate contains the first execution layer of the engine:
//! CPU topology detection, priority-aware load balancing, lock-free
//! task queues, and WASM dispatch through Wasmer.

pub mod balancer;
pub mod scheduler;
pub mod topology;
#[cfg(feature = "mobile")]
pub mod mobile_bridge;

pub use balancer::{CorePreference, LoadBalancer, RoutingDecision, TaskPriority};
pub use scheduler::{
    ClassExecutionMetrics, DispatchError, DispatchResult, Dispatcher, MpsScheduler, NativeTask,
    SchedulerMetrics, TaskEnvelope, TaskId, TaskPayload, WasmTask,
};
pub use topology::{CpuClass, CpuCore, CpuTopology};
#[cfg(feature = "mobile")]
pub use mobile_bridge::MpsMobileAdapter;
