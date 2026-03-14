//! Runtime GPU scheduler-path selection (GMS vs MGS).
//!
//! This policy chooses a default graphics scheduling path from adapter metadata:
//! - mobile TBDR targets (Mali/Adreno/PowerVR/Apple mobile profile) -> MGS
//! - desktop/high-throughput targets -> GMS

use mgs::MobileGpuProfile;

/// Graphics scheduler family selected by runtime policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphicsSchedulerPath {
    /// High-throughput graphics multi-scaler path.
    Gms,
    /// Mobile/serial fallback graphics scheduler path.
    Mgs,
}

/// Runtime scheduler-path decision with explainable metadata.
#[derive(Debug, Clone)]
pub struct GraphicsSchedulerDecision {
    /// Selected scheduler path.
    pub path: GraphicsSchedulerPath,
    /// Adapter name used for the decision.
    pub adapter_name: String,
    /// Adapter backend used for the decision.
    pub backend: wgpu::Backend,
    /// Adapter device type used for the decision.
    pub device_type: wgpu::DeviceType,
    /// Mobile profile inferred from adapter name.
    pub mobile_profile: MobileGpuProfile,
    /// Human-readable policy explanation.
    pub reason: String,
}

/// Select a scheduler path from `wgpu::AdapterInfo`.
pub fn choose_scheduler_path(adapter_info: &wgpu::AdapterInfo) -> GraphicsSchedulerDecision {
    let profile = MobileGpuProfile::detect(&adapter_info.name);
    let is_mobile_tbdr = profile.is_mobile_tbdr();
    let is_integrated_or_mobile = matches!(
        adapter_info.device_type,
        wgpu::DeviceType::IntegratedGpu | wgpu::DeviceType::Other
    );

    let (path, reason) = if is_mobile_tbdr && is_integrated_or_mobile {
        (
            GraphicsSchedulerPath::Mgs,
            format!(
                "mobile TBDR profile detected ({:?}/{:?}) on integrated-like adapter",
                profile.family, profile.architecture
            ),
        )
    } else if is_mobile_tbdr && matches!(adapter_info.backend, wgpu::Backend::Metal) {
        (
            GraphicsSchedulerPath::Mgs,
            format!(
                "Metal mobile/TBDR profile detected ({:?}/{:?})",
                profile.family, profile.architecture
            ),
        )
    } else {
        (
            GraphicsSchedulerPath::Gms,
            format!(
                "non-mobile or throughput-oriented adapter ({:?}, {:?})",
                adapter_info.device_type, adapter_info.backend
            ),
        )
    };

    GraphicsSchedulerDecision {
        path,
        adapter_name: adapter_info.name.clone(),
        backend: adapter_info.backend,
        device_type: adapter_info.device_type,
        mobile_profile: profile,
        reason,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_adapter_info(name: &str, device_type: wgpu::DeviceType) -> wgpu::AdapterInfo {
        wgpu::AdapterInfo {
            name: name.to_string(),
            vendor: 0,
            device: 0,
            device_type,
            device_pci_bus_id: String::new(),
            driver: String::new(),
            driver_info: String::new(),
            backend: wgpu::Backend::Vulkan,
            subgroup_min_size: 1,
            subgroup_max_size: 1,
            transient_saves_memory: false,
        }
    }

    #[test]
    fn chooses_mgs_for_mobile_tbdr_profile() {
        let info = make_adapter_info("Mali-G78 MC24", wgpu::DeviceType::IntegratedGpu);
        let decision = choose_scheduler_path(&info);
        assert_eq!(decision.path, GraphicsSchedulerPath::Mgs);
    }

    #[test]
    fn chooses_gms_for_discrete_desktop_profile() {
        let info = make_adapter_info("NVIDIA GeForce RTX 5060 Ti", wgpu::DeviceType::DiscreteGpu);
        let decision = choose_scheduler_path(&info);
        assert_eq!(decision.path, GraphicsSchedulerPath::Gms);
    }
}
