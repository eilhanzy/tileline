//! Core orchestration modules for Tileline.
//!
//! The `core` namespace hosts engine-facing coordination logic that is intentionally renderer-
//! agnostic. It includes:
//! - [`bridge`]: MPS<->GMS bridge — transforms CPU task completions into frame-scoped GMS plans.
//! - [`mgs_bridge`]: MPS<->MGS bridge — same lock-free pattern for the mobile GPU path.

pub mod bridge;
pub mod mgs_bridge;
