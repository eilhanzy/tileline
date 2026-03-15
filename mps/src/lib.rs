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

/// Canonical module id used by runtime version commands.
pub const MODULE_ID: &str = "mps";
/// Crate version resolved at compile time.
pub const MODULE_VERSION: &str = env!("CARGO_PKG_VERSION");

pub mod balancer;
pub mod key_bruteforce;
pub mod key_decode;
pub mod scheduler;
pub mod topology;

pub use balancer::{CorePreference, LoadBalancer, RoutingDecision, TaskPriority};
pub use key_bruteforce::{
    entropy_check, normalize_candidate_keys, submit_payload_bruteforce_scan,
    submit_pinned_nonce_matrix_scan, BlobTransform, BruteForceFinding, BruteForceInputError,
    BruteForceMatch, BruteForceScanHandle, CipherAlgorithm, EntropyCheck, FindingKind,
    FindingPriority, HeuristicSignal, KeyTransform, MacAlgorithm, MacContextInjection,
    MacKeyTransform, MacPadding, MacScope, MacValidation, BINARY_ENTROPY_THRESHOLD_MILLI,
    BLOB_BYTES, CIPHERTEXT_BYTES, CTR_NONCE_BYTES, HEADER_BYTES, HEURISTIC_SCORE_THRESHOLD,
    MAC_VALIDATION_SCORE_THRESHOLD, TAG_BYTES,
};
pub use key_decode::{
    parse_strict_key_record, Base64Alphabet, DecodedSegment, ParseError, ParsedKeyRecord,
    EXPECTED_RECORD_LENGTHS, KEY_KIND, KEY_PREFIX,
};
pub use scheduler::{
    ClassExecutionMetrics, DispatchError, DispatchResult, Dispatcher, MpsScheduler, NativeTask,
    SchedulerMetrics, TaskEnvelope, TaskId, TaskPayload, WasmTask,
};
pub use topology::{CpuClass, CpuCore, CpuTopology};
