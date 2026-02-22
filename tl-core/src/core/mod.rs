//! Core orchestration modules for Tileline.
//!
//! The `core` namespace hosts engine-facing coordination logic that is intentionally renderer-
//! agnostic. It currently includes the MPS<->GMS bridge which transforms CPU-side task
//! completions into frame-scoped GPU planning requests.

pub mod bridge;
