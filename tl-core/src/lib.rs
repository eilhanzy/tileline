//! Tileline core orchestration crate.
//!
//! `tl-core` is the integration boundary between:
//! - `mps` (CPU scheduling / WASM execution)
//! - `gms` (GPU discovery, planning, and multi-GPU heuristics)
//! - runtime/render-loop code that records real `wgpu` submissions and present synchronization
//!
//! The crate exposes:
//! - [`core::bridge`] for MPS<->GMS frame planning
//! - [`graphics::multigpu::sync`] for portable explicit multi-GPU synchronization and UMA hooks

pub mod core;
pub mod graphics;
pub mod tlscript;

pub use core::bridge::{
    BridgeFrameId, BridgeFramePlan, BridgeGpuTaskKind, BridgeMpsSubmission, BridgeSubmitReceipt,
    BridgeTaskDescriptor, BridgeTaskRouting, MpsGmsBridge, MpsGmsBridgeConfig, MpsGmsBridgeMetrics,
};
pub use gms::{AdaptiveBufferDecision, AdaptiveFrameTelemetry};
pub use graphics::multigpu::sync::{
    ComposeBarrierState, GpuQueueLane, MultiGpuFrameSyncConfig, MultiGpuFrameSynchronizer,
    MultiGpuSyncSnapshot, SharedPlacementPolicy, SyncBackendHint,
};
pub use tlscript::{LexError, LexErrorKind, Lexer, Span, Token, TokenKind};
