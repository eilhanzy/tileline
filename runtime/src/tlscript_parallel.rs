//! Runtime-facing `.tlscript` parallel planning and diagnostics glue.
//!
//! This module keeps scripting-parallel ergonomics in `src/` (not examples):
//! - validated dispatch planning via `tl_core::ParallelDispatchPlanner`
//! - developer-facing advisory summaries via `tl_core::ParallelAdvisor`
//! - runtime metrics for serial fallbacks and parallel-ready coverage
//! - MPS submission routing based on validated/annotated typed-IR execution metadata
//!
//! It does not execute WASM script code yet. Instead, it centralizes the planning/diagnostic layer
//! that a future script VM/WASM host runtime can call before submitting work to MPS.

use mps::{CorePreference, MpsScheduler, NativeTask, TaskId, TaskPriority};
use paradoxpe::PhysicsWorld;
use tl_core::{
    IrScheduleHint, Module, ParallelAdvisor, ParallelAdvisorConfig, ParallelAdvisorReport,
    ParallelDispatchDecision, ParallelDispatchMode, ParallelDispatchPlanner,
    ParallelDispatchPlannerConfig, ParallelDispatchPlannerMetrics, ParallelHookOutcome,
    SemanticReport, TypedIrFunction,
};

/// Snapshot of `.tlscript` parallel planning and advisory metrics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TlscriptParallelRuntimeMetrics {
    /// Number of advisor analyses executed.
    pub advisor_runs: u64,
    /// Number of times an advisor report contained at least one serial fallback function.
    pub advisor_reports_with_fallbacks: u64,
    /// Number of suggested parallel contract templates generated across advisor runs.
    pub advisor_suggested_templates: u64,
    /// Last compact advisor summary line (for log panes / HUD overlays / build output).
    pub last_advisor_summary: Option<String>,
    /// Number of dispatch routing decisions that reached the runtime submission helper.
    pub dispatch_route_calls: u64,
    /// Number of dispatch decisions that required main-thread execution.
    pub dispatch_main_thread_required: u64,
    /// Number of serial MPS submissions emitted.
    pub dispatch_serial_submissions: u64,
    /// Number of parallel MPS batches emitted.
    pub dispatch_parallel_batches: u64,
    /// Number of MPS tasks submitted across all parallel batches.
    pub dispatch_parallel_tasks_submitted: u64,
    /// Snapshot of runtime dispatch planner counters and fallback reasons.
    pub planner: ParallelDispatchPlannerMetrics,
}

impl TlscriptParallelRuntimeMetrics {
    /// Compact planner fallback summary line for logs/HUD overlays.
    pub fn planner_fallbacks_line(&self) -> String {
        let p = self.planner;
        format!(
            "tlscript planner: parallel={} serial={} main={} fallbacks[m_policy={} no_contract={} small={} one_chunk={} nondet_reduce={} body_write={}]",
            p.planned_parallel,
            p.planned_serial,
            p.planned_main_thread,
            p.fallback_main_thread_only_policy,
            p.fallback_missing_parallel_contract,
            p.fallback_workload_too_small,
            p.fallback_single_chunk_only,
            p.fallback_nondeterministic_reduction,
            p.fallback_unsupported_body_write_set,
        )
    }

    /// Overlay/log lines combining advisor summary + planner fallback counters + dispatch stats.
    pub fn overlay_lines(&self) -> Vec<String> {
        let mut lines = Vec::with_capacity(4);
        if let Some(summary) = &self.last_advisor_summary {
            lines.push(summary.clone());
        }
        lines.push(self.planner_fallbacks_line());
        lines.push(format!(
            "tlscript dispatch: routes={} main_thread={} serial_submits={} parallel_batches={} parallel_tasks={}",
            self.dispatch_route_calls,
            self.dispatch_main_thread_required,
            self.dispatch_serial_submissions,
            self.dispatch_parallel_batches,
            self.dispatch_parallel_tasks_submitted,
        ));
        lines
    }
}

/// MPS submission tuning for `.tlscript` dispatch decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TlscriptMpsDispatchConfig {
    /// Priority for serial script tasks offloaded to MPS.
    pub serial_priority: TaskPriority,
    /// Priority for parallel chunk batches offloaded to MPS.
    pub parallel_priority: TaskPriority,
}

impl Default for TlscriptMpsDispatchConfig {
    fn default() -> Self {
        Self {
            serial_priority: TaskPriority::High,
            parallel_priority: TaskPriority::High,
        }
    }
}

/// Chunk span in workload-item space used by native task construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TlscriptWorkChunk {
    /// Inclusive start index.
    pub start: usize,
    /// Exclusive end index.
    pub end: usize,
}

/// Result of routing one script invocation into main-thread or MPS submissions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TlscriptDispatchSubmission {
    /// Planner decision that drove routing.
    pub decision: ParallelDispatchDecision,
    /// `true` when the caller must execute on the main thread.
    pub main_thread_required: bool,
    /// Submitted MPS task ids.
    pub mps_task_ids: Vec<TaskId>,
    /// Chunk spans used for submissions (for trace/profiling).
    pub submitted_chunks: Vec<TlscriptWorkChunk>,
}

/// Engine-side coordinator for `.tlscript` parallel planning + accessibility diagnostics.
#[derive(Debug, Clone)]
pub struct TlscriptParallelRuntimeCoordinator {
    advisor: ParallelAdvisor,
    planner: ParallelDispatchPlanner,
    dispatch_config: TlscriptMpsDispatchConfig,
    advisor_runs: u64,
    advisor_reports_with_fallbacks: u64,
    advisor_suggested_templates: u64,
    last_advisor_summary: Option<String>,
    dispatch_route_calls: u64,
    dispatch_main_thread_required: u64,
    dispatch_serial_submissions: u64,
    dispatch_parallel_batches: u64,
    dispatch_parallel_tasks_submitted: u64,
}

impl TlscriptParallelRuntimeCoordinator {
    /// Build a coordinator with explicit advisor + planner configs.
    pub fn new(
        advisor_config: ParallelAdvisorConfig,
        planner_config: ParallelDispatchPlannerConfig,
    ) -> Self {
        Self::with_dispatch_config(
            advisor_config,
            planner_config,
            TlscriptMpsDispatchConfig::default(),
        )
    }

    /// Build a coordinator with explicit advisor/planner and MPS dispatch tuning.
    pub fn with_dispatch_config(
        advisor_config: ParallelAdvisorConfig,
        planner_config: ParallelDispatchPlannerConfig,
        dispatch_config: TlscriptMpsDispatchConfig,
    ) -> Self {
        Self {
            advisor: ParallelAdvisor::new(advisor_config),
            planner: ParallelDispatchPlanner::new(planner_config),
            dispatch_config,
            advisor_runs: 0,
            advisor_reports_with_fallbacks: 0,
            advisor_suggested_templates: 0,
            last_advisor_summary: None,
            dispatch_route_calls: 0,
            dispatch_main_thread_required: 0,
            dispatch_serial_submissions: 0,
            dispatch_parallel_batches: 0,
            dispatch_parallel_tasks_submitted: 0,
        }
    }

    /// Run the advisor on one parsed/validated module and cache a compact summary.
    pub fn analyze_module<'src>(
        &mut self,
        module: &Module<'src>,
        semantic: &SemanticReport<'src>,
        hooks: &ParallelHookOutcome<'src>,
    ) -> ParallelAdvisorReport<'src> {
        let report = self.advisor.analyze(module, semantic, hooks);
        self.advisor_runs = self.advisor_runs.saturating_add(1);
        if report.serial_fallback_count > 0 {
            self.advisor_reports_with_fallbacks =
                self.advisor_reports_with_fallbacks.saturating_add(1);
        }
        let templates = report.suggested_contract_templates();
        self.advisor_suggested_templates = self
            .advisor_suggested_templates
            .saturating_add(templates.len() as u64);
        self.last_advisor_summary = Some(report.summary_line());
        report
    }

    /// Plan one typed-IR function invocation for MPS dispatch.
    pub fn plan_function_dispatch<'src>(
        &self,
        func: &TypedIrFunction<'src>,
        workload_items: usize,
    ) -> ParallelDispatchDecision {
        self.planner.plan_for_function(func, workload_items)
    }

    /// Plan dispatch for ParadoxPE `domain="bodies"` functions from the current world state.
    ///
    /// The workload size is derived from active physics bodies so sleeping/static bodies do not
    /// inflate chunk planning.
    pub fn plan_paradox_body_dispatch<'src>(
        &self,
        func: &TypedIrFunction<'src>,
        world: &PhysicsWorld,
    ) -> ParallelDispatchDecision {
        let workload_items = world.active_parallel_body_count();
        self.planner.plan_for_function(func, workload_items)
    }

    /// Plan and submit native script work to MPS according to the dispatch decision.
    ///
    /// The caller provides a chunk task factory that returns boxed native tasks for `MpsScheduler`.
    /// This is the canonical pre-WASM-host routing layer for future script runtime integration.
    pub fn dispatch_native_chunks_for_function<'src, F>(
        &mut self,
        mps: &MpsScheduler,
        func: &TypedIrFunction<'src>,
        workload_items: usize,
        mut make_task: F,
    ) -> TlscriptDispatchSubmission
    where
        F: FnMut(TlscriptWorkChunk) -> NativeTask,
    {
        let decision = self.plan_function_dispatch(func, workload_items);
        self.dispatch_route_calls = self.dispatch_route_calls.saturating_add(1);
        let core_preference = core_preference_from_schedule_hint(decision.schedule_hint);

        match decision.mode {
            ParallelDispatchMode::MainThread => {
                self.dispatch_main_thread_required =
                    self.dispatch_main_thread_required.saturating_add(1);
                TlscriptDispatchSubmission {
                    decision,
                    main_thread_required: true,
                    mps_task_ids: Vec::new(),
                    submitted_chunks: Vec::new(),
                }
            }
            ParallelDispatchMode::Serial => {
                if workload_items == 0 {
                    return TlscriptDispatchSubmission {
                        decision,
                        main_thread_required: false,
                        mps_task_ids: Vec::new(),
                        submitted_chunks: Vec::new(),
                    };
                }
                let chunk = TlscriptWorkChunk {
                    start: 0,
                    end: workload_items,
                };
                let task_id = mps.submit_native_boxed(
                    self.dispatch_config.serial_priority,
                    core_preference,
                    make_task(chunk),
                );
                self.dispatch_serial_submissions =
                    self.dispatch_serial_submissions.saturating_add(1);
                TlscriptDispatchSubmission {
                    decision,
                    main_thread_required: false,
                    mps_task_ids: vec![task_id],
                    submitted_chunks: vec![chunk],
                }
            }
            ParallelDispatchMode::ParallelChunked => {
                let chunk_size = decision.chunk_size.unwrap_or(workload_items.max(1)).max(1);
                let mut chunks = Vec::new();
                let mut start = 0usize;
                while start < workload_items {
                    let end = (start + chunk_size).min(workload_items);
                    chunks.push(TlscriptWorkChunk { start, end });
                    start = end;
                }
                if chunks.is_empty() && workload_items > 0 {
                    chunks.push(TlscriptWorkChunk {
                        start: 0,
                        end: workload_items,
                    });
                }

                let tasks: Vec<NativeTask> = chunks.iter().copied().map(&mut make_task).collect();
                let task_ids = if tasks.is_empty() {
                    Vec::new()
                } else {
                    mps.submit_batch_native(
                        self.dispatch_config.parallel_priority,
                        core_preference,
                        tasks,
                    )
                };

                self.dispatch_parallel_batches = self.dispatch_parallel_batches.saturating_add(1);
                self.dispatch_parallel_tasks_submitted = self
                    .dispatch_parallel_tasks_submitted
                    .saturating_add(task_ids.len() as u64);

                TlscriptDispatchSubmission {
                    decision,
                    main_thread_required: false,
                    mps_task_ids: task_ids,
                    submitted_chunks: chunks,
                }
            }
        }
    }

    /// Plan and submit native body-domain work sized from a `PhysicsWorld`.
    ///
    /// This is the first ParadoxPE-specific bridge between `.tlscript` parallel contracts and MPS
    /// chunk routing. It keeps physics-domain planning in `src/` without requiring benchmark code.
    pub fn dispatch_native_paradox_body_chunks_for_function<'src, F>(
        &mut self,
        mps: &MpsScheduler,
        func: &TypedIrFunction<'src>,
        world: &PhysicsWorld,
        make_task: F,
    ) -> TlscriptDispatchSubmission
    where
        F: FnMut(TlscriptWorkChunk) -> NativeTask,
    {
        self.dispatch_native_chunks_for_function(
            mps,
            func,
            world.active_parallel_body_count(),
            make_task,
        )
    }

    /// Access the underlying planner for advanced integrations.
    pub fn planner(&self) -> &ParallelDispatchPlanner {
        &self.planner
    }

    /// Mutable access to the underlying planner for advanced integrations.
    pub fn planner_mut(&mut self) -> &mut ParallelDispatchPlanner {
        &mut self.planner
    }

    /// Access the underlying advisor for advanced integrations.
    pub fn advisor(&self) -> &ParallelAdvisor {
        &self.advisor
    }

    /// Mutable access to the underlying advisor for advanced integrations.
    pub fn advisor_mut(&mut self) -> &mut ParallelAdvisor {
        &mut self.advisor
    }

    /// Snapshot runtime-level metrics and fallback counters.
    pub fn metrics(&self) -> TlscriptParallelRuntimeMetrics {
        TlscriptParallelRuntimeMetrics {
            advisor_runs: self.advisor_runs,
            advisor_reports_with_fallbacks: self.advisor_reports_with_fallbacks,
            advisor_suggested_templates: self.advisor_suggested_templates,
            last_advisor_summary: self.last_advisor_summary.clone(),
            dispatch_route_calls: self.dispatch_route_calls,
            dispatch_main_thread_required: self.dispatch_main_thread_required,
            dispatch_serial_submissions: self.dispatch_serial_submissions,
            dispatch_parallel_batches: self.dispatch_parallel_batches,
            dispatch_parallel_tasks_submitted: self.dispatch_parallel_tasks_submitted,
            planner: self.planner.metrics(),
        }
    }

    /// Last advisor summary line, if any analysis has been executed.
    pub fn last_advisor_summary(&self) -> Option<&str> {
        self.last_advisor_summary.as_deref()
    }
}

impl Default for TlscriptParallelRuntimeCoordinator {
    fn default() -> Self {
        Self::new(Default::default(), Default::default())
    }
}

fn core_preference_from_schedule_hint(hint: IrScheduleHint) -> CorePreference {
    match hint {
        IrScheduleHint::Auto => CorePreference::Auto,
        IrScheduleHint::Performance => CorePreference::Performance,
        IrScheduleHint::Efficient => CorePreference::Efficient,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use paradoxpe::{BodyDesc, BodyKind, PhysicsWorld, PhysicsWorldConfig};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::Duration;
    use tl_core::{
        annotate_typed_ir_with_parallel_hooks, lower_to_typed_ir, Lexer, ParallelHookAnalyzer,
        Parser, SemanticAnalyzer,
    };

    fn compile_pipeline(
        src: &str,
    ) -> (
        Module<'_>,
        SemanticReport<'_>,
        ParallelHookOutcome<'_>,
        tl_core::TypedIrModule<'_>,
    ) {
        let mut parser = Parser::new(Lexer::new(src));
        let module = parser.parse_module().expect("parse ok");
        let semantic = SemanticAnalyzer::new(Default::default())
            .analyze(&module)
            .expect("semantic ok");
        let hooks = ParallelHookAnalyzer::new().analyze(&module, &semantic);
        let mut ir = lower_to_typed_ir(&module, &semantic).expect("lower ok");
        annotate_typed_ir_with_parallel_hooks(&mut ir, &hooks);
        (module, semantic, hooks, ir)
    }

    #[test]
    fn coordinator_collects_advisor_summary_and_templates() {
        let (module, semantic, hooks, _ir) = compile_pipeline(concat!(
            "@export\n",
            "def solve_forces(dt: float):\n",
            "    let x: int = 1\n",
        ));
        let mut coord = TlscriptParallelRuntimeCoordinator::default();
        let report = coord.analyze_module(&module, &semantic, &hooks);
        assert_eq!(report.function_count, 1);
        assert!(coord
            .last_advisor_summary()
            .unwrap()
            .contains("serial_fallback=1"));
        let metrics = coord.metrics();
        assert_eq!(metrics.advisor_runs, 1);
        assert_eq!(metrics.advisor_reports_with_fallbacks, 1);
        assert!(metrics.advisor_suggested_templates >= 1);
    }

    #[test]
    fn coordinator_plans_parallel_dispatch_and_records_planner_metrics() {
        let (_module, _semantic, _hooks, ir) = compile_pipeline(concat!(
            "@export\n",
            "@parallel(domain=\"bodies\", read=\"transform\", write=\"velocity\", chunk=128)\n",
            "def solve(dt: float):\n",
            "    let x: int = 1\n",
        ));
        let coord = TlscriptParallelRuntimeCoordinator::default();
        let decision = coord.plan_function_dispatch(&ir.functions[0], 2048);
        assert!(decision.is_parallel());
        let metrics = coord.metrics();
        assert_eq!(metrics.planner.planned_parallel, 1);
    }

    #[test]
    fn dispatch_native_chunks_routes_parallel_work_to_mps() {
        let (_module, _semantic, _hooks, ir) = compile_pipeline(concat!(
            "@export\n",
            "@parallel(domain=\"bodies\", read=\"transform\", write=\"velocity\", chunk=64)\n",
            "def solve(dt: float):\n",
            "    let x: int = 1\n",
        ));
        let scheduler = MpsScheduler::new();
        let mut coord = TlscriptParallelRuntimeCoordinator::default();
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_for_tasks = Arc::clone(&counter);

        let submission = coord.dispatch_native_chunks_for_function(
            &scheduler,
            &ir.functions[0],
            256,
            move |_chunk| {
                let c = Arc::clone(&counter_for_tasks);
                Box::new(move || {
                    c.fetch_add(1, Ordering::Relaxed);
                })
            },
        );

        assert!(submission.decision.is_parallel());
        assert!(!submission.mps_task_ids.is_empty());
        assert_eq!(
            submission.mps_task_ids.len(),
            submission.submitted_chunks.len()
        );
        assert!(scheduler.wait_for_idle(Duration::from_millis(250)));
        assert_eq!(
            counter.load(Ordering::Relaxed),
            submission.submitted_chunks.len()
        );

        let metrics = coord.metrics();
        assert_eq!(metrics.dispatch_parallel_batches, 1);
        assert_eq!(
            metrics.dispatch_parallel_tasks_submitted as usize,
            submission.submitted_chunks.len()
        );
    }

    #[test]
    fn metrics_overlay_lines_include_advisor_and_planner_data() {
        let (module, semantic, hooks, _ir) = compile_pipeline(concat!(
            "@export\n",
            "def solve_forces(dt: float):\n",
            "    let x: int = 1\n",
        ));
        let mut coord = TlscriptParallelRuntimeCoordinator::default();
        let _ = coord.analyze_module(&module, &semantic, &hooks);
        let lines = coord.metrics().overlay_lines();
        assert!(!lines.is_empty());
        assert!(lines.iter().any(|l| l.contains("tlscript planner:")));
    }

    #[test]
    fn paradox_body_dispatch_uses_active_body_count_and_performance_hint() {
        let (_module, _semantic, _hooks, ir) = compile_pipeline(concat!(
            "@export\n",
            "@parallel(domain=\"bodies\", read=\"transform,aabb\", write=\"velocity\", chunk=64)\n",
            "@deterministic\n",
            "def solve_bodies(dt: float):\n",
            "    let x: int = 1\n",
        ));
        let mut world = PhysicsWorld::new(PhysicsWorldConfig::default());
        let _static_body = world.spawn_body(BodyDesc {
            kind: BodyKind::Static,
            ..BodyDesc::default()
        });
        let _dynamic_a = world.spawn_body(BodyDesc::default());
        let _dynamic_b = world.spawn_body(BodyDesc::default());

        let coord = TlscriptParallelRuntimeCoordinator::default();
        let decision = coord.plan_paradox_body_dispatch(&ir.functions[0], &world);
        assert_eq!(decision.schedule_hint, IrScheduleHint::Performance);
        assert_eq!(
            world.active_parallel_body_count(),
            world.body_registry().active_parallel_body_count()
        );
    }

    #[test]
    fn paradox_body_dispatch_routes_chunks_to_mps_from_world_load() {
        let (_module, _semantic, _hooks, ir) = compile_pipeline(concat!(
            "@export\n",
            "@parallel(domain=\"bodies\", read=\"transform,force\", write=\"velocity\", chunk=64)\n",
            "@deterministic\n",
            "def solve_bodies(dt: float):\n",
            "    let x: int = 1\n",
        ));
        let mut world = PhysicsWorld::new(PhysicsWorldConfig::default());
        for _ in 0..192 {
            let _ = world.spawn_body(BodyDesc::default());
        }

        let scheduler = MpsScheduler::new();
        let mut coord = TlscriptParallelRuntimeCoordinator::default();
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_for_tasks = Arc::clone(&counter);

        let submission = coord.dispatch_native_paradox_body_chunks_for_function(
            &scheduler,
            &ir.functions[0],
            &world,
            move |_chunk| {
                let c = Arc::clone(&counter_for_tasks);
                Box::new(move || {
                    c.fetch_add(1, Ordering::Relaxed);
                })
            },
        );

        assert!(submission.decision.is_parallel());
        assert!(scheduler.wait_for_idle(Duration::from_millis(250)));
        assert_eq!(
            counter.load(Ordering::Relaxed),
            submission.submitted_chunks.len()
        );
    }
}
