//! Multi-GPU graphics support primitives.
//!
//! This namespace contains synchronization and policy helpers used by runtime render loops when
//! coordinating primary/secondary GPU work and host-mediated bridge transfers.

pub mod sync;
