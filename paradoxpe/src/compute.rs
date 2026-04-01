//! GPU-compute dispatch foundation for ParadoxPE.
//!
//! The first integration slice is intentionally conservative:
//! - world-step hot path can ask an engine-owned compute backend to execute integration
//! - if no backend is present, or the backend declines the request, CPU execution continues
//! - telemetry keeps track of attempted/executed/fallback dispatches so runtime pacing remains
//!   diagnosable while the real Vulkan compute path is built out

use std::borrow::Cow;
use std::ops::Range;

use nalgebra::Vector3;

use crate::storage::BodyRegistry;

/// Runtime policy for ParadoxPE compute offload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PhysicsComputeMode {
    /// Never attempt GPU compute dispatch.
    #[default]
    Disabled,
    /// Attempt GPU compute first, but always fail soft to CPU.
    PreferGpu,
    /// Strongly prefer GPU dispatch; current foundation still fails soft to CPU so runtime
    /// stability is preserved until the native backend is production-ready.
    ForceGpu,
}

impl PhysicsComputeMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::PreferGpu => "prefer_gpu",
            Self::ForceGpu => "force_gpu",
        }
    }
}

/// Physics stages that can eventually be offloaded to an engine-owned GPU compute backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PhysicsComputeStage {
    #[default]
    Integrate,
    Broadphase,
    Narrowphase,
    Solver,
}

impl PhysicsComputeStage {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Integrate => "integrate",
            Self::Broadphase => "broadphase",
            Self::Narrowphase => "narrowphase",
            Self::Solver => "solver",
        }
    }
}

/// Back-end classification for telemetry and future scheduler policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PhysicsComputeBackendKind {
    #[default]
    Unknown,
    Vulkan,
    CpuSimd,
    Custom,
}

/// Compute offload switches exposed through the ParadoxPE world config.
#[derive(Debug, Clone, PartialEq)]
pub struct PhysicsComputeConfig {
    pub mode: PhysicsComputeMode,
    pub min_body_count: usize,
    pub integrate_enabled: bool,
    pub broadphase_enabled: bool,
    pub narrowphase_enabled: bool,
    pub solver_enabled: bool,
}

impl PhysicsComputeConfig {
    pub const fn stage_enabled(&self, stage: PhysicsComputeStage) -> bool {
        match stage {
            PhysicsComputeStage::Integrate => self.integrate_enabled,
            PhysicsComputeStage::Broadphase => self.broadphase_enabled,
            PhysicsComputeStage::Narrowphase => self.narrowphase_enabled,
            PhysicsComputeStage::Solver => self.solver_enabled,
        }
    }
}

impl Default for PhysicsComputeConfig {
    fn default() -> Self {
        Self {
            mode: PhysicsComputeMode::Disabled,
            min_body_count: 2_048,
            integrate_enabled: true,
            broadphase_enabled: false,
            narrowphase_enabled: false,
            solver_enabled: false,
        }
    }
}

/// Summary of one pending compute dispatch attempt.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhysicsComputeDispatchRequest {
    pub stage: PhysicsComputeStage,
    pub substep_index: u32,
    pub total_substeps: u32,
    pub fixed_dt: f32,
    pub body_count: usize,
    pub candidate_pair_count: usize,
    pub manifold_count: usize,
}

/// Outcome of one compute dispatch request.
#[derive(Debug, Clone, PartialEq)]
pub enum PhysicsComputeDispatchResult {
    Executed {
        gpu_time_us: u64,
        workgroups: u32,
    },
    Fallback {
        reason: Cow<'static, str>,
    },
}

/// Frame/step-local compute telemetry exported by `PhysicsWorld`.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PhysicsComputeStats {
    pub mode: PhysicsComputeMode,
    pub backend_name: Option<String>,
    pub backend_kind: PhysicsComputeBackendKind,
    pub attempted_dispatches: u32,
    pub executed_dispatches: u32,
    pub fallback_dispatches: u32,
    pub skipped_dispatches: u32,
    pub last_stage: Option<PhysicsComputeStage>,
    pub last_reason: Option<String>,
    pub last_gpu_time_us: u64,
    pub last_workgroups: u32,
}

impl PhysicsComputeStats {
    pub fn new(
        mode: PhysicsComputeMode,
        backend_name: Option<String>,
        backend_kind: PhysicsComputeBackendKind,
    ) -> Self {
        Self {
            mode,
            backend_name,
            backend_kind,
            ..Self::default()
        }
    }

    pub fn note_skip(&mut self, stage: PhysicsComputeStage, reason: impl Into<String>) {
        self.skipped_dispatches = self.skipped_dispatches.saturating_add(1);
        self.last_stage = Some(stage);
        self.last_reason = Some(reason.into());
    }

    pub fn note_fallback(&mut self, stage: PhysicsComputeStage, reason: impl Into<String>) {
        self.attempted_dispatches = self.attempted_dispatches.saturating_add(1);
        self.fallback_dispatches = self.fallback_dispatches.saturating_add(1);
        self.last_stage = Some(stage);
        self.last_reason = Some(reason.into());
    }

    pub fn note_executed(
        &mut self,
        stage: PhysicsComputeStage,
        gpu_time_us: u64,
        workgroups: u32,
    ) {
        self.attempted_dispatches = self.attempted_dispatches.saturating_add(1);
        self.executed_dispatches = self.executed_dispatches.saturating_add(1);
        self.last_stage = Some(stage);
        self.last_reason = None;
        self.last_gpu_time_us = gpu_time_us;
        self.last_workgroups = workgroups;
    }
}

/// Engine-owned compute backend contract used by ParadoxPE hot phases.
///
/// The first shipping slice only wires `dispatch_integrate` into the world step. Broadphase,
/// narrowphase, and solver are intentionally left as staged follow-up work so the engine can land
/// fail-soft integration before the native Vulkan compute path is fully authored.
pub trait PhysicsComputeBackend: Send + Sync {
    fn backend_name(&self) -> &str;

    fn backend_kind(&self) -> PhysicsComputeBackendKind {
        PhysicsComputeBackendKind::Unknown
    }

    fn supports_stage(&self, stage: PhysicsComputeStage) -> bool;

    fn dispatch_integrate(
        &self,
        request: PhysicsComputeDispatchRequest,
        bodies: &mut BodyRegistry,
        shards: &[Range<usize>],
        gravity: Vector3<f32>,
        fixed_dt: f32,
    ) -> PhysicsComputeDispatchResult;
}
