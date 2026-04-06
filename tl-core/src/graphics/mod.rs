//! Graphics-facing core abstractions.
//!
//! These modules sit below runtime-facing scene orchestration and above explicit GPU API usage.
//! The current focus is:
//! - portable explicit multi-GPU synchronization used by the MPS<->GMS bridge
//! - a Linux-first raw Vulkan backend skeleton for the `v0.5.0` independence transition

pub mod frame_snapshot;
#[cfg(target_os = "macos")]
pub mod metal_backend;
#[cfg(target_os = "macos")]
pub mod metal_physics_compute;
pub mod multigpu;
#[cfg(target_os = "linux")]
pub mod vulkan_backend;
#[cfg(target_os = "linux")]
pub mod vulkan_physics_compute;
