//! Multi-Processing Scaler (MPS)
//! ---------------------------------
//! This crate contains the first execution layer of the engine:
//! CPU topology detection, priority-aware load balancing, lock-free
//! task queues, and WASM dispatch through Wasmer.
//!
//! Primary responsibilities:
//! - detect performance/efficient core topology
//! - route tasks by priority and core preference
//! - execute native Rust closures and WASM tasks in memory
//! - expose scheduler metrics for bridge/runtime feedback loops

pub mod balancer;
pub mod scheduler;
pub mod topology;

pub use balancer::{CorePreference, LoadBalancer, RoutingDecision, TaskPriority};
pub use scheduler::{
    ClassExecutionMetrics, DispatchError, DispatchResult, Dispatcher, MpsScheduler, NativeTask,
    SchedulerMetrics, TaskEnvelope, TaskId, TaskPayload, WasmTask,
};
pub use topology::{CpuClass, CpuCore, CpuTopology};
