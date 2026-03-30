//! Runtime GPU scheduler-path selection (GMS vs MGS).
//!
//! This policy chooses a default graphics scheduling path from adapter metadata:
//! - mobile TBDR targets (Mali/Adreno/PowerVR/Apple mobile profile) -> MGS
//! - desktop/high-throughput targets -> GMS

use mgs::MobileGpuProfile;

/// Runtime platform classification used by scheduler policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimePlatform {
    Desktop,
    Android,
}

impl RuntimePlatform {
    /// Build-time platform signal.
    pub fn current() -> Self {
        if cfg!(target_os = "android") {
            Self::Android
        } else {
            Self::Desktop
        }
    }
}

/// Graphics scheduler family selected by runtime policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphicsSchedulerPath {
    /// High-throughput graphics multi-scaler path.
    Gms,
    /// Mobile/serial fallback graphics scheduler path.
    Mgs,
}

/// Backend-agnostic GPU backend classification used by runtime policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeGpuBackend {
    Vulkan,
    Metal,
    Dx12,
    Gl,
    BrowserWebGpu,
    Other,
}

/// Backend-agnostic GPU device-class classification used by runtime policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeGpuDeviceType {
    IntegratedGpu,
    DiscreteGpu,
    VirtualGpu,
    Cpu,
    Other,
}

/// Runtime-owned adapter identity used by scheduler policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeAdapterInfo {
    pub name: String,
    pub backend: RuntimeGpuBackend,
    pub device_type: RuntimeGpuDeviceType,
}

impl RuntimeAdapterInfo {
    /// Build runtime-owned adapter info from a `wgpu` adapter descriptor.
    pub fn from_wgpu(adapter_info: &wgpu::AdapterInfo) -> Self {
        Self {
            name: adapter_info.name.clone(),
            backend: RuntimeGpuBackend::from_wgpu(adapter_info.backend),
            device_type: RuntimeGpuDeviceType::from_wgpu(adapter_info.device_type),
        }
    }
}

impl RuntimeGpuBackend {
    /// Convert from `wgpu::Backend` to runtime-owned backend classification.
    pub fn from_wgpu(backend: wgpu::Backend) -> Self {
        match backend {
            wgpu::Backend::Vulkan => Self::Vulkan,
            wgpu::Backend::Metal => Self::Metal,
            wgpu::Backend::Dx12 => Self::Dx12,
            wgpu::Backend::Gl => Self::Gl,
            wgpu::Backend::BrowserWebGpu => Self::BrowserWebGpu,
            _ => Self::Other,
        }
    }
}

impl RuntimeGpuDeviceType {
    /// Convert from `wgpu::DeviceType` to runtime-owned device classification.
    pub fn from_wgpu(device_type: wgpu::DeviceType) -> Self {
        match device_type {
            wgpu::DeviceType::IntegratedGpu => Self::IntegratedGpu,
            wgpu::DeviceType::DiscreteGpu => Self::DiscreteGpu,
            wgpu::DeviceType::VirtualGpu => Self::VirtualGpu,
            wgpu::DeviceType::Cpu => Self::Cpu,
            wgpu::DeviceType::Other => Self::Other,
        }
    }
}

/// Runtime scheduler-path decision with explainable metadata.
#[derive(Debug, Clone)]
pub struct GraphicsSchedulerDecision {
    /// Selected scheduler path.
    pub path: GraphicsSchedulerPath,
    /// Adapter name used for the decision.
    pub adapter_name: String,
    /// Adapter backend used for the decision.
    pub backend: RuntimeGpuBackend,
    /// Adapter device type used for the decision.
    pub device_type: RuntimeGpuDeviceType,
    /// Mobile profile inferred from adapter name.
    pub mobile_profile: MobileGpuProfile,
    /// Human-readable policy explanation.
    pub reason: String,
}

/// Reads the `TILELINE_SCHEDULER` environment variable to force a scheduler path.
///
/// Values: `"mgs"` forces MGS, `"gms"` forces GMS.
/// If the variable is absent or unrecognised, auto-detection is used.
///
/// Temporary: remove this function and the `if let` block in
/// `choose_scheduler_path_for_platform` once MGS testing is complete.
/// To disable the override without a code change, simply unset `TILELINE_SCHEDULER`.
fn env_scheduler_override() -> Option<GraphicsSchedulerPath> {
    match std::env::var("TILELINE_SCHEDULER").as_deref() {
        Ok("mgs") => Some(GraphicsSchedulerPath::Mgs),
        Ok("gms") => Some(GraphicsSchedulerPath::Gms),
        _ => None,
    }
}

/// Select a scheduler path from `wgpu::AdapterInfo`.
///
/// Respects the `TILELINE_SCHEDULER` env-var override when set.
pub fn choose_scheduler_path(adapter_info: &wgpu::AdapterInfo) -> GraphicsSchedulerDecision {
    choose_scheduler_path_from_adapter(&RuntimeAdapterInfo::from_wgpu(adapter_info))
}

/// Select a scheduler path from `wgpu::AdapterInfo` with explicit platform policy.
///
/// `TILELINE_SCHEDULER` overrides platform policy when set.
pub fn choose_scheduler_path_for_platform(
    adapter_info: &wgpu::AdapterInfo,
    platform: RuntimePlatform,
) -> GraphicsSchedulerDecision {
    choose_scheduler_path_for_platform_from_adapter(
        &RuntimeAdapterInfo::from_wgpu(adapter_info),
        platform,
    )
}

/// Select a scheduler path from backend-neutral runtime adapter info.
///
/// Respects the `TILELINE_SCHEDULER` env-var override when set.
pub fn choose_scheduler_path_from_adapter(
    adapter_info: &RuntimeAdapterInfo,
) -> GraphicsSchedulerDecision {
    choose_scheduler_path_for_platform_from_adapter(adapter_info, RuntimePlatform::current())
}

/// Select a scheduler path from backend-neutral runtime adapter info with explicit platform policy.
///
/// `TILELINE_SCHEDULER` overrides platform policy when set.
pub fn choose_scheduler_path_for_platform_from_adapter(
    adapter_info: &RuntimeAdapterInfo,
    platform: RuntimePlatform,
) -> GraphicsSchedulerDecision {
    // Temporary — remove this block and env_scheduler_override() once MGS testing is done.
    if let Some(forced) = env_scheduler_override() {
        let profile = MobileGpuProfile::detect(&adapter_info.name);
        let reason = format!(
            "TILELINE_SCHEDULER override active: {:?} forced (adapter: {}, platform: {:?})",
            forced, adapter_info.name, platform
        );
        return GraphicsSchedulerDecision {
            path: forced,
            adapter_name: adapter_info.name.clone(),
            backend: adapter_info.backend,
            device_type: adapter_info.device_type,
            mobile_profile: profile,
            reason,
        };
    }

    let profile = MobileGpuProfile::detect(&adapter_info.name);
    let is_mobile_tbdr = profile.is_mobile_tbdr();
    let is_integrated_or_mobile = matches!(
        adapter_info.device_type,
        RuntimeGpuDeviceType::IntegratedGpu | RuntimeGpuDeviceType::Other
    );

    let (path, reason) = if platform == RuntimePlatform::Android {
        (
            GraphicsSchedulerPath::Mgs,
            format!(
                "android auto policy prefers MGS ({:?}/{:?}, {:?}, {:?})",
                profile.family,
                profile.architecture,
                adapter_info.device_type,
                adapter_info.backend
            ),
        )
    } else if profile.is_desktop_class() {
        (
            GraphicsSchedulerPath::Gms,
            format!(
                "Apple M-series desktop-class profile detected ({:?}), routing to GMS throughput path",
                profile.family
            ),
        )
    } else if is_mobile_tbdr && is_integrated_or_mobile {
        (
            GraphicsSchedulerPath::Mgs,
            format!(
                "mobile TBDR profile detected ({:?}/{:?}) on integrated-like adapter",
                profile.family, profile.architecture
            ),
        )
    } else if is_mobile_tbdr && matches!(adapter_info.backend, RuntimeGpuBackend::Metal) {
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
                "desktop auto policy selected throughput path ({:?}, {:?})",
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

    fn make_adapter_info(name: &str, device_type: RuntimeGpuDeviceType) -> RuntimeAdapterInfo {
        RuntimeAdapterInfo {
            name: name.to_string(),
            device_type,
            backend: RuntimeGpuBackend::Vulkan,
        }
    }

    #[test]
    fn chooses_mgs_for_mobile_tbdr_profile() {
        let info = make_adapter_info("Mali-G78 MC24", RuntimeGpuDeviceType::IntegratedGpu);
        let decision = choose_scheduler_path_from_adapter(&info);
        assert_eq!(decision.path, GraphicsSchedulerPath::Mgs);
    }

    #[test]
    fn chooses_gms_for_discrete_desktop_profile() {
        let info = make_adapter_info(
            "NVIDIA GeForce RTX 5060 Ti",
            RuntimeGpuDeviceType::DiscreteGpu,
        );
        let decision =
            choose_scheduler_path_for_platform_from_adapter(&info, RuntimePlatform::Desktop);
        assert_eq!(decision.path, GraphicsSchedulerPath::Gms);
    }

    #[test]
    fn chooses_mgs_for_android_even_on_discrete_label() {
        let info = make_adapter_info(
            "NVIDIA GeForce RTX 5060 Ti",
            RuntimeGpuDeviceType::DiscreteGpu,
        );
        let decision =
            choose_scheduler_path_for_platform_from_adapter(&info, RuntimePlatform::Android);
        assert_eq!(decision.path, GraphicsSchedulerPath::Mgs);
    }

    #[test]
    fn chooses_gms_for_apple_m_series_desktop_class() {
        let mut info = make_adapter_info("Apple M4 Max", RuntimeGpuDeviceType::IntegratedGpu);
        info.backend = RuntimeGpuBackend::Metal;
        let decision =
            choose_scheduler_path_for_platform_from_adapter(&info, RuntimePlatform::Desktop);
        assert_eq!(decision.path, GraphicsSchedulerPath::Gms);
    }
}
