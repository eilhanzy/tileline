//! Runtime-facing parallel dispatch planning for `.tlscript` typed IR.
//!
//! This module provides a small, lock-free metrics-backed planner that decides whether a lowered
//! function should run as a parallel MPS chunked task, a serial MPS task, or on the main thread.
//! It is intentionally conservative: when in doubt it records a fallback reason and keeps behavior
//! deterministic and safe.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use super::typed_ir::{
    IrExecutionPolicy, IrReduceKind, IrScheduleHint, TypedIrExecutionMeta, TypedIrFunction,
};

/// Runtime dispatch mode selected for a script function invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParallelDispatchMode {
    /// Execute on the main/game thread.
    MainThread,
    /// Execute as a single serial task (still offloadable to MPS if desired).
    Serial,
    /// Execute as chunked parallel tasks over MPS workers.
    ParallelChunked,
}

/// Runtime fallback reason when a function does not run in parallel chunked mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParallelRuntimeFallbackReason {
    /// Function is explicitly marked as main-thread-only.
    MainThreadOnlyPolicy,
    /// Function has no validated parallel contract (default serial-safe path).
    MissingParallelContract,
    /// Workload size is zero, so parallel planning is unnecessary.
    EmptyWorkload,
    /// Workload is below the configured threshold for chunking.
    WorkloadTooSmall,
    /// Chunking would result in a single chunk, so serial is cheaper.
    SingleChunkOnly,
    /// Reduction requested without deterministic contract (safety fallback).
    NonDeterministicReduction,
}

/// Planner decision for one function invocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParallelDispatchDecision {
    /// Chosen execution mode.
    pub mode: ParallelDispatchMode,
    /// Chunk size when `mode == ParallelChunked`.
    pub chunk_size: Option<usize>,
    /// Chunk count when `mode == ParallelChunked`.
    pub chunk_count: Option<usize>,
    /// Fallback reason when parallel chunking was not selected.
    pub fallback_reason: Option<ParallelRuntimeFallbackReason>,
    /// MPS core preference hint carried from IR metadata.
    pub schedule_hint: IrScheduleHint,
    /// Deterministic execution requirement.
    pub deterministic: bool,
    /// Optional reduction strategy for merge step.
    pub reduce: Option<IrReduceKind>,
}

impl ParallelDispatchDecision {
    /// `true` if the planner selected chunked parallel execution.
    pub fn is_parallel(&self) -> bool {
        matches!(self.mode, ParallelDispatchMode::ParallelChunked)
    }
}

/// Planner tuning knobs for balancing accessibility (safe defaults) and performance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParallelDispatchPlannerConfig {
    /// Minimum number of work items before chunked parallelization is considered.
    pub min_parallel_items: usize,
    /// Minimum chunk size used when a function does not provide `chunk_hint`.
    pub min_chunk_size: usize,
    /// Maximum number of chunks to emit for a single invocation.
    pub max_chunks: usize,
    /// Default chunk size used when no function chunk hint exists.
    pub default_chunk_size: usize,
}

impl Default for ParallelDispatchPlannerConfig {
    fn default() -> Self {
        Self {
            min_parallel_items: 128,
            min_chunk_size: 64,
            max_chunks: 64,
            default_chunk_size: 256,
        }
    }
}

/// Snapshot of planner decisions and fallback counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ParallelDispatchPlannerMetrics {
    pub planned_parallel: u64,
    pub planned_serial: u64,
    pub planned_main_thread: u64,
    pub fallback_main_thread_only_policy: u64,
    pub fallback_missing_parallel_contract: u64,
    pub fallback_empty_workload: u64,
    pub fallback_workload_too_small: u64,
    pub fallback_single_chunk_only: u64,
    pub fallback_nondeterministic_reduction: u64,
}

#[derive(Debug, Default)]
struct ParallelDispatchPlannerCounters {
    planned_parallel: AtomicU64,
    planned_serial: AtomicU64,
    planned_main_thread: AtomicU64,
    fallback_main_thread_only_policy: AtomicU64,
    fallback_missing_parallel_contract: AtomicU64,
    fallback_empty_workload: AtomicU64,
    fallback_workload_too_small: AtomicU64,
    fallback_single_chunk_only: AtomicU64,
    fallback_nondeterministic_reduction: AtomicU64,
}

impl ParallelDispatchPlannerCounters {
    fn snapshot(&self) -> ParallelDispatchPlannerMetrics {
        ParallelDispatchPlannerMetrics {
            planned_parallel: self.planned_parallel.load(Ordering::Relaxed),
            planned_serial: self.planned_serial.load(Ordering::Relaxed),
            planned_main_thread: self.planned_main_thread.load(Ordering::Relaxed),
            fallback_main_thread_only_policy: self
                .fallback_main_thread_only_policy
                .load(Ordering::Relaxed),
            fallback_missing_parallel_contract: self
                .fallback_missing_parallel_contract
                .load(Ordering::Relaxed),
            fallback_empty_workload: self.fallback_empty_workload.load(Ordering::Relaxed),
            fallback_workload_too_small: self.fallback_workload_too_small.load(Ordering::Relaxed),
            fallback_single_chunk_only: self.fallback_single_chunk_only.load(Ordering::Relaxed),
            fallback_nondeterministic_reduction: self
                .fallback_nondeterministic_reduction
                .load(Ordering::Relaxed),
        }
    }

    fn record(&self, decision: ParallelDispatchDecision) {
        match decision.mode {
            ParallelDispatchMode::ParallelChunked => {
                self.planned_parallel.fetch_add(1, Ordering::Relaxed);
            }
            ParallelDispatchMode::Serial => {
                self.planned_serial.fetch_add(1, Ordering::Relaxed);
            }
            ParallelDispatchMode::MainThread => {
                self.planned_main_thread.fetch_add(1, Ordering::Relaxed);
            }
        }

        if let Some(reason) = decision.fallback_reason {
            let counter = match reason {
                ParallelRuntimeFallbackReason::MainThreadOnlyPolicy => {
                    &self.fallback_main_thread_only_policy
                }
                ParallelRuntimeFallbackReason::MissingParallelContract => {
                    &self.fallback_missing_parallel_contract
                }
                ParallelRuntimeFallbackReason::EmptyWorkload => &self.fallback_empty_workload,
                ParallelRuntimeFallbackReason::WorkloadTooSmall => {
                    &self.fallback_workload_too_small
                }
                ParallelRuntimeFallbackReason::SingleChunkOnly => &self.fallback_single_chunk_only,
                ParallelRuntimeFallbackReason::NonDeterministicReduction => {
                    &self.fallback_nondeterministic_reduction
                }
            };
            counter.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Runtime planner that selects parallel vs serial execution and tracks fallback reasons.
#[derive(Debug, Clone)]
pub struct ParallelDispatchPlanner {
    config: ParallelDispatchPlannerConfig,
    counters: Arc<ParallelDispatchPlannerCounters>,
}

impl ParallelDispatchPlanner {
    /// Create a planner with explicit config.
    pub fn new(config: ParallelDispatchPlannerConfig) -> Self {
        Self {
            config,
            counters: Arc::new(ParallelDispatchPlannerCounters::default()),
        }
    }

    /// Plan dispatch for a typed-IR function based on its execution metadata and workload size.
    pub fn plan_for_function<'src>(
        &self,
        func: &TypedIrFunction<'src>,
        workload_items: usize,
    ) -> ParallelDispatchDecision {
        self.plan_from_meta(func.meta.execution, workload_items)
    }

    /// Plan dispatch directly from execution metadata (useful before full lowering integration).
    pub fn plan_from_meta(
        &self,
        exec: TypedIrExecutionMeta,
        workload_items: usize,
    ) -> ParallelDispatchDecision {
        let decision = self.compute_decision(exec, workload_items);
        self.counters.record(decision);
        decision
    }

    /// Snapshot planner metrics.
    pub fn metrics(&self) -> ParallelDispatchPlannerMetrics {
        self.counters.snapshot()
    }

    fn compute_decision(
        &self,
        exec: TypedIrExecutionMeta,
        workload_items: usize,
    ) -> ParallelDispatchDecision {
        if matches!(exec.policy, IrExecutionPolicy::MainThreadOnly) {
            return ParallelDispatchDecision {
                mode: ParallelDispatchMode::MainThread,
                chunk_size: None,
                chunk_count: None,
                fallback_reason: Some(ParallelRuntimeFallbackReason::MainThreadOnlyPolicy),
                schedule_hint: exec.schedule_hint,
                deterministic: exec.deterministic,
                reduce: exec.reduce,
            };
        }

        if !matches!(exec.policy, IrExecutionPolicy::ParallelSafe) {
            return self
                .serial_decision(exec, ParallelRuntimeFallbackReason::MissingParallelContract);
        }

        if exec.reduce.is_some() && !exec.deterministic {
            return self.serial_decision(
                exec,
                ParallelRuntimeFallbackReason::NonDeterministicReduction,
            );
        }

        if workload_items == 0 {
            return self.serial_decision(exec, ParallelRuntimeFallbackReason::EmptyWorkload);
        }
        if workload_items < self.config.min_parallel_items {
            return self.serial_decision(exec, ParallelRuntimeFallbackReason::WorkloadTooSmall);
        }

        let chunk_size = usize::from(
            exec.chunk_hint
                .unwrap_or(self.config.default_chunk_size as u16),
        )
        .max(self.config.min_chunk_size)
        .max(1);
        let mut chunk_count = workload_items.div_ceil(chunk_size);
        if chunk_count <= 1 {
            return self.serial_decision(exec, ParallelRuntimeFallbackReason::SingleChunkOnly);
        }
        if chunk_count > self.config.max_chunks {
            chunk_count = self.config.max_chunks;
        }
        let adjusted_chunk_size = workload_items.div_ceil(chunk_count).max(1);

        ParallelDispatchDecision {
            mode: ParallelDispatchMode::ParallelChunked,
            chunk_size: Some(adjusted_chunk_size),
            chunk_count: Some(chunk_count),
            fallback_reason: None,
            schedule_hint: exec.schedule_hint,
            deterministic: exec.deterministic,
            reduce: exec.reduce,
        }
    }

    fn serial_decision(
        &self,
        exec: TypedIrExecutionMeta,
        reason: ParallelRuntimeFallbackReason,
    ) -> ParallelDispatchDecision {
        ParallelDispatchDecision {
            mode: ParallelDispatchMode::Serial,
            chunk_size: None,
            chunk_count: None,
            fallback_reason: Some(reason),
            schedule_hint: exec.schedule_hint,
            deterministic: exec.deterministic,
            reduce: exec.reduce,
        }
    }
}

impl Default for ParallelDispatchPlanner {
    fn default() -> Self {
        Self::new(ParallelDispatchPlannerConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tlscript::{
        IrExecutionPolicy, IrReduceKind, IrScheduleHint, TypedIrExecutionMeta, TypedIrFunctionMeta,
    };

    fn meta(policy: IrExecutionPolicy) -> TypedIrExecutionMeta {
        TypedIrExecutionMeta {
            policy,
            deterministic: false,
            schedule_hint: IrScheduleHint::Auto,
            chunk_hint: Some(128),
            reduce: None,
            read_effect_count: 1,
            write_effect_count: 1,
        }
    }

    #[test]
    fn planner_falls_back_without_parallel_contract() {
        let planner = ParallelDispatchPlanner::default();
        let d = planner.plan_from_meta(meta(IrExecutionPolicy::Serial), 1024);
        assert_eq!(d.mode, ParallelDispatchMode::Serial);
        assert_eq!(
            d.fallback_reason,
            Some(ParallelRuntimeFallbackReason::MissingParallelContract)
        );
        let m = planner.metrics();
        assert_eq!(m.planned_serial, 1);
        assert_eq!(m.fallback_missing_parallel_contract, 1);
    }

    #[test]
    fn planner_selects_parallel_for_valid_workload() {
        let planner = ParallelDispatchPlanner::default();
        let d = planner.plan_from_meta(meta(IrExecutionPolicy::ParallelSafe), 2048);
        assert_eq!(d.mode, ParallelDispatchMode::ParallelChunked);
        assert!(d.chunk_count.unwrap() > 1);
        let m = planner.metrics();
        assert_eq!(m.planned_parallel, 1);
    }

    #[test]
    fn planner_records_main_thread_policy_fallback() {
        let planner = ParallelDispatchPlanner::default();
        let d = planner.plan_from_meta(meta(IrExecutionPolicy::MainThreadOnly), 2048);
        assert_eq!(d.mode, ParallelDispatchMode::MainThread);
        assert_eq!(
            d.fallback_reason,
            Some(ParallelRuntimeFallbackReason::MainThreadOnlyPolicy)
        );
        let m = planner.metrics();
        assert_eq!(m.planned_main_thread, 1);
        assert_eq!(m.fallback_main_thread_only_policy, 1);
    }

    #[test]
    fn planner_rejects_nondeterministic_reduction() {
        let planner = ParallelDispatchPlanner::default();
        let mut e = meta(IrExecutionPolicy::ParallelSafe);
        e.reduce = Some(IrReduceKind::Sum);
        e.deterministic = false;
        let d = planner.plan_from_meta(e, 4096);
        assert_eq!(d.mode, ParallelDispatchMode::Serial);
        assert_eq!(
            d.fallback_reason,
            Some(ParallelRuntimeFallbackReason::NonDeterministicReduction)
        );
    }

    #[test]
    fn typed_ir_function_meta_keeps_execution_defaults_serial_safe() {
        let meta = TypedIrFunctionMeta::default();
        assert_eq!(meta.execution.policy, IrExecutionPolicy::Serial);
    }
}
