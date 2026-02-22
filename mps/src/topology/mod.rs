//! CPU topology module.
//! Re-exports the detector data types used by the scheduler.

pub mod detector;

pub use detector::{CpuClass, CpuCore, CpuTopology};
