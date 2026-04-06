//! Metal-backed ParadoxPE compute backend skeleton.
//!
//! v0.5.0 scope intentionally keeps execution fail-soft while exposing deterministic planning
//! and telemetry surfaces for integrate-stage offload.

#![cfg(target_os = "macos")]

use paradoxpe::{
    BodyRegistry, PhysicsComputeBackend, PhysicsComputeBackendKind, PhysicsComputeDispatchRequest,
    PhysicsComputeDispatchResult, PhysicsComputeStage,
};

use super::metal_backend::MetalBackend;

/// Runtime tuning for the Metal physics-compute bridge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetalPhysicsComputeConfig {
    /// Allow the backend to advertise/support integration-stage dispatch.
    pub enable_integrate_stage: bool,
    /// Minimum bodies required before planning any workgroups.
    pub min_body_count: usize,
    /// Preferred threadgroup width for future Metal kernels.
    pub preferred_threads_per_group: u32,
    /// Keep backend in planning/fallback mode until full kernels are wired.
    pub dry_run: bool,
}

impl Default for MetalPhysicsComputeConfig {
    fn default() -> Self {
        Self {
            enable_integrate_stage: true,
            min_body_count: 2_048,
            preferred_threads_per_group: 128,
            dry_run: true,
        }
    }
}

/// Capability summary exported into ParadoxPE scheduling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetalPhysicsComputeCapabilities {
    pub device_name: String,
    pub supports_argument_buffers: bool,
    pub supports_indirect_command_buffers: bool,
    pub integrate_stage_ready: bool,
    pub fallback_reason: Option<String>,
}

/// Deterministic dispatch plan for one physics-compute request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetalPhysicsDispatchPlan {
    pub stage: PhysicsComputeStage,
    pub threads_per_group: u32,
    pub threadgroup_count_x: u32,
}

/// Metal-facing ParadoxPE compute backend skeleton.
#[derive(Debug, Clone)]
pub struct MetalPhysicsComputeBackend {
    config: MetalPhysicsComputeConfig,
    capabilities: MetalPhysicsComputeCapabilities,
}

impl MetalPhysicsComputeBackend {
    pub fn from_backend(backend: &MetalBackend, config: MetalPhysicsComputeConfig) -> Self {
        Self::from_device_name(backend.device_name(), config)
    }

    pub fn from_device_name(device_name: &str, config: MetalPhysicsComputeConfig) -> Self {
        let integrate_stage_ready = config.enable_integrate_stage;
        let fallback_reason = if integrate_stage_ready {
            None
        } else {
            Some("integrate stage disabled in Metal physics compute config".to_string())
        };

        Self {
            config,
            capabilities: MetalPhysicsComputeCapabilities {
                device_name: device_name.to_string(),
                supports_argument_buffers: true,
                supports_indirect_command_buffers: false,
                integrate_stage_ready,
                fallback_reason,
            },
        }
    }

    pub fn config(&self) -> &MetalPhysicsComputeConfig {
        &self.config
    }

    pub fn capabilities(&self) -> &MetalPhysicsComputeCapabilities {
        &self.capabilities
    }

    pub fn build_dispatch_plan(
        &self,
        request: PhysicsComputeDispatchRequest,
    ) -> Option<MetalPhysicsDispatchPlan> {
        if request.stage != PhysicsComputeStage::Integrate {
            return None;
        }
        if request.body_count < self.config.min_body_count {
            return None;
        }
        let threads_per_group = self.config.preferred_threads_per_group.max(32);
        let threadgroup_count_x =
            ((request.body_count as u32).saturating_add(threads_per_group - 1)) / threads_per_group;
        Some(MetalPhysicsDispatchPlan {
            stage: request.stage,
            threads_per_group,
            threadgroup_count_x: threadgroup_count_x.max(1),
        })
    }
}

impl PhysicsComputeBackend for MetalPhysicsComputeBackend {
    fn backend_name(&self) -> &str {
        "metal-physics-compute"
    }

    fn backend_kind(&self) -> PhysicsComputeBackendKind {
        PhysicsComputeBackendKind::Metal
    }

    fn supports_stage(&self, stage: PhysicsComputeStage) -> bool {
        matches!(stage, PhysicsComputeStage::Integrate) && self.capabilities.integrate_stage_ready
    }

    fn dispatch_integrate(
        &self,
        request: PhysicsComputeDispatchRequest,
        _bodies: &mut BodyRegistry,
        _shards: &[std::ops::Range<usize>],
        _gravity: nalgebra::Vector3<f32>,
        _fixed_dt: f32,
    ) -> PhysicsComputeDispatchResult {
        if !self.config.enable_integrate_stage {
            return PhysicsComputeDispatchResult::Fallback {
                reason: "integrate stage disabled in Metal physics compute config".into(),
            };
        }
        if !self.capabilities.integrate_stage_ready {
            return PhysicsComputeDispatchResult::Fallback {
                reason: self
                    .capabilities
                    .fallback_reason
                    .clone()
                    .unwrap_or_else(|| "Metal physics compute backend is not ready".to_string())
                    .into(),
            };
        }
        let Some(plan) = self.build_dispatch_plan(request) else {
            return PhysicsComputeDispatchResult::Fallback {
                reason: format!(
                    "integrate body threshold not reached for Metal compute ({} < {})",
                    request.body_count, self.config.min_body_count
                )
                .into(),
            };
        };

        if self.config.dry_run {
            return PhysicsComputeDispatchResult::Fallback {
                reason: format!(
                    "Metal integrate compute pipeline not compiled yet (planned {} threadgroups on {})",
                    plan.threadgroup_count_x, self.capabilities.device_name
                )
                .into(),
            };
        }

        PhysicsComputeDispatchResult::Fallback {
            reason: format!(
                "Metal integrate dispatch is wired but pipeline upload path is not implemented yet ({})",
                self.capabilities.device_name
            )
            .into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_integrate_dispatch_plan_from_device_name() {
        let backend = MetalPhysicsComputeBackend::from_device_name(
            "Apple M4 Pro",
            MetalPhysicsComputeConfig {
                min_body_count: 64,
                preferred_threads_per_group: 128,
                dry_run: true,
                ..MetalPhysicsComputeConfig::default()
            },
        );

        let plan = backend
            .build_dispatch_plan(PhysicsComputeDispatchRequest {
                stage: PhysicsComputeStage::Integrate,
                substep_index: 0,
                total_substeps: 2,
                fixed_dt: 1.0 / 120.0,
                body_count: 513,
                candidate_pair_count: 0,
                manifold_count: 0,
            })
            .expect("plan");

        assert_eq!(plan.threads_per_group, 128);
        assert_eq!(plan.threadgroup_count_x, 5);
        assert!(backend.supports_stage(PhysicsComputeStage::Integrate));
    }
}
