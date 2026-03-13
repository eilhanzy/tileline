//! Graphics-facing core abstractions.
//!
//! These modules sit below a concrete renderer implementation and above raw `wgpu` usage. The
//! current focus is portable explicit multi-GPU synchronization used by the MPS<->GMS bridge.

pub mod multigpu;
