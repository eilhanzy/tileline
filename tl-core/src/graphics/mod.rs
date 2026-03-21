//! Graphics-facing core abstractions.
//!
//! These modules sit below runtime-facing scene orchestration and above explicit GPU API usage.
//! The current focus is:
//! - portable explicit multi-GPU synchronization used by the MPS<->GMS bridge
//! - a Linux-first raw Vulkan backend skeleton for the `v0.5.0` independence transition

pub mod multigpu;
pub mod vulkan_backend;
