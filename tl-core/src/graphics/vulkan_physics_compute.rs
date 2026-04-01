//! Vulkan-backed ParadoxPE compute backend skeleton.
//!
//! This module intentionally lands as a real integration boundary before full shader/pipeline
//! authoring:
//! - it converts Vulkan device/profile information into ParadoxPE compute capabilities
//! - it produces deterministic dispatch plans for physics integration work
//! - it implements the `paradoxpe::PhysicsComputeBackend` contract with fail-soft fallback
//! - it keeps the runtime honest about what is and is not GPU-executed today

#![cfg(target_os = "linux")]

use paradoxpe::{
    BodyRegistry, PhysicsComputeBackend, PhysicsComputeBackendKind, PhysicsComputeDispatchRequest,
    PhysicsComputeDispatchResult, PhysicsComputeStage,
};

use super::vulkan_backend::{VulkanBackend, VulkanPhysicalDeviceProfile};

/// Runtime tuning for the Vulkan physics-compute bridge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VulkanPhysicsComputeConfig {
    /// Allow the backend to advertise/support integration-stage dispatch.
    pub enable_integrate_stage: bool,
    /// Minimum bodies required before we even bother planning workgroups.
    pub min_body_count: usize,
    /// Preferred local workgroup width for future compute kernels.
    pub preferred_local_size_x: u32,
    /// Keep the backend in planning/fallback mode until shader/pipeline upload is connected.
    pub dry_run: bool,
}

impl Default for VulkanPhysicsComputeConfig {
    fn default() -> Self {
        Self {
            enable_integrate_stage: true,
            min_body_count: 2_048,
            preferred_local_size_x: 128,
            dry_run: true,
        }
    }
}

/// Capability summary exported from the Vulkan bootstrap path into ParadoxPE scheduling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VulkanPhysicsComputeCapabilities {
    pub device_name: String,
    pub queue_family_index: Option<u32>,
    pub descriptor_indexing_supported: bool,
    pub timeline_semaphore: bool,
    pub buffer_device_address: bool,
    pub dynamic_rendering: bool,
    pub integrate_stage_ready: bool,
    pub fallback_reason: Option<String>,
}

/// Deterministic dispatch plan for one physics-compute request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VulkanPhysicsDispatchPlan {
    pub stage: PhysicsComputeStage,
    pub local_size_x: u32,
    pub workgroup_count_x: u32,
}

/// Vulkan-facing ParadoxPE compute backend skeleton.
#[derive(Debug, Clone)]
pub struct VulkanPhysicsComputeBackend {
    config: VulkanPhysicsComputeConfig,
    capabilities: VulkanPhysicsComputeCapabilities,
}

impl VulkanPhysicsComputeBackend {
    pub fn from_backend(
        backend: &VulkanBackend,
        config: VulkanPhysicsComputeConfig,
    ) -> Self {
        Self::from_device_profile(backend.primary_device_profile(), config)
    }

    pub fn from_device_profile(
        profile: &VulkanPhysicalDeviceProfile,
        config: VulkanPhysicsComputeConfig,
    ) -> Self {
        let queue_family_index = profile.queue_selection.compute_family_index;
        let integrate_stage_ready = config.enable_integrate_stage && queue_family_index.is_some();
        let fallback_reason = if integrate_stage_ready {
            None
        } else if !config.enable_integrate_stage {
            Some("integrate stage disabled in Vulkan physics compute config".to_string())
        } else {
            Some("no Vulkan compute queue family selected for ParadoxPE compute".to_string())
        };
        Self {
            config,
            capabilities: VulkanPhysicsComputeCapabilities {
                device_name: profile.device_name.clone(),
                queue_family_index,
                descriptor_indexing_supported: profile.descriptor_indexing_supported,
                timeline_semaphore: profile.extension_support.timeline_semaphore,
                buffer_device_address: profile.extension_support.buffer_device_address,
                dynamic_rendering: profile.extension_support.dynamic_rendering,
                integrate_stage_ready,
                fallback_reason,
            },
        }
    }

    pub fn config(&self) -> &VulkanPhysicsComputeConfig {
        &self.config
    }

    pub fn capabilities(&self) -> &VulkanPhysicsComputeCapabilities {
        &self.capabilities
    }

    pub fn build_dispatch_plan(
        &self,
        request: PhysicsComputeDispatchRequest,
    ) -> Option<VulkanPhysicsDispatchPlan> {
        if request.stage != PhysicsComputeStage::Integrate {
            return None;
        }
        if request.body_count < self.config.min_body_count {
            return None;
        }
        let local_size_x = self.config.preferred_local_size_x.max(32);
        let workgroup_count_x = ((request.body_count as u32).saturating_add(local_size_x - 1))
            / local_size_x;
        Some(VulkanPhysicsDispatchPlan {
            stage: request.stage,
            local_size_x,
            workgroup_count_x: workgroup_count_x.max(1),
        })
    }
}

impl PhysicsComputeBackend for VulkanPhysicsComputeBackend {
    fn backend_name(&self) -> &str {
        "vulkan-physics-compute"
    }

    fn backend_kind(&self) -> PhysicsComputeBackendKind {
        PhysicsComputeBackendKind::Vulkan
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
                reason: "integrate stage disabled in Vulkan physics compute config".into(),
            };
        }
        if !self.capabilities.integrate_stage_ready {
            return PhysicsComputeDispatchResult::Fallback {
                reason: self
                    .capabilities
                    .fallback_reason
                    .clone()
                    .unwrap_or_else(|| "Vulkan physics compute backend is not ready".to_string())
                    .into(),
            };
        }
        let Some(plan) = self.build_dispatch_plan(request) else {
            return PhysicsComputeDispatchResult::Fallback {
                reason: format!(
                    "integrate body threshold not reached for Vulkan compute ({} < {})",
                    request.body_count, self.config.min_body_count
                )
                .into(),
            };
        };
        if self.config.dry_run {
            return PhysicsComputeDispatchResult::Fallback {
                reason: format!(
                    "Vulkan integrate compute pipeline not compiled yet (planned {} workgroups on {})",
                    plan.workgroup_count_x, self.capabilities.device_name
                )
                .into(),
            };
        }
        PhysicsComputeDispatchResult::Fallback {
            reason: format!(
                "Vulkan integrate dispatch is wired but shader/pipeline upload path is not implemented yet ({})",
                self.capabilities.device_name
            )
            .into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use ash::vk;

    use super::*;
    use crate::graphics::vulkan_backend::{
        VulkanDeviceExtensionSupport, VulkanQueueSelection,
    };

    fn test_profile() -> VulkanPhysicalDeviceProfile {
        VulkanPhysicalDeviceProfile {
            physical_device: vk::PhysicalDevice::null(),
            device_name: "Test Vulkan GPU".to_string(),
            vendor_id: 1,
            device_id: 2,
            device_type: vk::PhysicalDeviceType::DISCRETE_GPU,
            api_version: vk::make_api_version(0, 1, 3, 0),
            queue_selection: VulkanQueueSelection {
                graphics_family_index: 0,
                present_family_index: Some(0),
                compute_family_index: Some(1),
                transfer_family_index: Some(2),
            },
            descriptor_indexing_supported: true,
            extension_support: VulkanDeviceExtensionSupport {
                swapchain: true,
                timeline_semaphore: true,
                external_memory_fd: true,
                external_semaphore_fd: true,
                buffer_device_address: true,
                dynamic_rendering: true,
            },
        }
    }

    #[test]
    fn builds_integrate_dispatch_plan_from_profile() {
        let backend = VulkanPhysicsComputeBackend::from_device_profile(
            &test_profile(),
            VulkanPhysicsComputeConfig {
                min_body_count: 64,
                preferred_local_size_x: 128,
                dry_run: true,
                ..VulkanPhysicsComputeConfig::default()
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
        assert_eq!(plan.local_size_x, 128);
        assert_eq!(plan.workgroup_count_x, 5);
        assert!(backend.supports_stage(PhysicsComputeStage::Integrate));
    }

    #[test]
    fn dry_run_backend_fails_soft_with_clear_reason() {
        let backend = VulkanPhysicsComputeBackend::from_device_profile(
            &test_profile(),
            VulkanPhysicsComputeConfig {
                min_body_count: 1,
                preferred_local_size_x: 64,
                dry_run: true,
                ..VulkanPhysicsComputeConfig::default()
            },
        );
        let result = backend.dispatch_integrate(
            PhysicsComputeDispatchRequest {
                stage: PhysicsComputeStage::Integrate,
                substep_index: 0,
                total_substeps: 1,
                fixed_dt: 1.0 / 120.0,
                body_count: 256,
                candidate_pair_count: 0,
                manifold_count: 0,
            },
            &mut BodyRegistry::default(),
            &[],
            nalgebra::Vector3::new(0.0, -9.81, 0.0),
            1.0 / 120.0,
        );
        match result {
            PhysicsComputeDispatchResult::Fallback { reason } => {
                assert!(reason.contains("not compiled yet"));
            }
            other => panic!("expected fallback, got {other:?}"),
        }
    }
}
