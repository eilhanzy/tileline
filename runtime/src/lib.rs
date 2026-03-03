//! Tileline runtime integration crate.
//!
//! This crate hosts engine-level orchestration code that wires `tl-core` bridge primitives into a
//! render loop without pushing integration logic into benchmarks/examples.
//!
//! Modules:
//! - `frame_loop`: bridge pumping and frame-plan queue management
//! - `wgpu_render_loop`: canonical `wgpu` submit/present integration hooks

mod frame_loop;
mod network_transport;
mod tlscript_parallel;
mod wgpu_render_loop;

pub use frame_loop::{
    FrameLoopRuntime, FrameLoopRuntimeConfig, FrameLoopRuntimeMetrics, FrameSubmissionRecordResult,
    RuntimeTickResult,
};
pub use network_transport::{
    NetworkPumpResult, NetworkTransportConfig, NetworkTransportMetrics, NetworkTransportRuntime,
    SnapshotCadenceConfig,
};
pub use tlscript_parallel::{
    TlscriptDispatchSubmission, TlscriptMpsDispatchConfig, TlscriptParallelRuntimeCoordinator,
    TlscriptParallelRuntimeMetrics, TlscriptWorkChunk,
};
pub use wgpu_render_loop::{
    FrameExecutionTelemetry, SecondaryHelperSubmitOutcome, WgpuRenderLoopCoordinator,
    WgpuRenderLoopMetrics,
};
