//! GPU hardware discovery and heuristic performance scoring for GMS.
//!
//! The scoring model is intentionally heuristic because `wgpu` does not expose
//! raw CU/SM counts or VRAM size on every backend. We therefore combine:
//! - adapter name parsing (best-effort CU/SM and VRAM hints)
//! - `wgpu::AdapterInfo` device type (discrete/integrated/virtual)
//! - `wgpu` limits/features as capability hints
//! - memory-topology penalties/bonuses

use std::cmp::Ordering;
#[cfg(target_os = "linux")]
use std::path::Path;
use std::process::Command;
use std::sync::OnceLock;

use wgpu::{Adapter, AdapterInfo, Backends, DeviceType, Features, Instance, Limits};

const NVIDIA_VENDOR_ID: u32 = 0x10DE;
const AMD_VENDOR_ID: u32 = 0x1002;
const INTEL_VENDOR_ID: u32 = 0x8086;
const APPLE_VENDOR_ID: u32 = 0x106B;
const ARM_VENDOR_ID: u32 = 0x13B5;

/// GPU compute unit type used for display/debugging.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeUnitKind {
    /// AMD Compute Unit / Work Group Processor equivalent bucket.
    Cu,
    /// NVIDIA Streaming Multiprocessor (SM).
    Sm,
    /// Generic GPU core cluster estimate (Apple/Intel/unknown vendors).
    CoreCluster,
    /// Unknown, fallback heuristic only.
    Unknown,
}

/// Where the CU/SM estimate came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeUnitEstimateSource {
    /// Vendor/OS native tooling probe (best available path when present).
    NativeProbe,
    /// Device-name lookup table maintained by GMS.
    DeviceNameTable,
    /// Coarse fallback derived from `wgpu` limits only.
    DriverLimitsHeuristic,
}

impl ComputeUnitEstimateSource {
    /// Compact source label for summaries.
    pub fn short_label(self) -> &'static str {
        match self {
            Self::NativeProbe => "native",
            Self::DeviceNameTable => "table",
            Self::DriverLimitsHeuristic => "heuristic",
        }
    }
}

impl ComputeUnitKind {
    /// Short label suitable for benchmark headers and compact summaries.
    pub fn short_label(self) -> &'static str {
        match self {
            Self::Cu => "CU",
            Self::Sm => "SM",
            Self::CoreCluster => "CoreCluster",
            Self::Unknown => "Units",
        }
    }

    /// Human-readable description for logs and detailed summaries.
    pub fn display_label(self) -> &'static str {
        match self {
            Self::Cu => "Compute Units (CU)",
            Self::Sm => "Streaming Multiprocessors (SM, CUDA scheduler clusters)",
            Self::CoreCluster => "GPU Core Clusters",
            Self::Unknown => "GPU Compute Units",
        }
    }
}

/// Memory topology that affects latency and transfer behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryTopology {
    /// Discrete GPU with local VRAM (PCIe or external interconnect).
    DiscreteVram,
    /// Integrated/unified memory GPU.
    Unified,
    /// Virtualized GPU path.
    Virtualized,
    /// CPU adapter or software path.
    System,
    /// Unknown or backend-specific behavior.
    Unknown,
}

/// Intermediate score components kept for observability.
#[derive(Debug, Clone, Copy, Default)]
pub struct GpuScoreBreakdown {
    pub cu_term: f64,
    pub bandwidth_term: f64,
    pub vram_term: f64,
    pub limits_term: f64,
    pub feature_term: f64,
    pub memory_topology_factor: f64,
    pub final_score: u64,
}

/// Compact adapter limit snapshot used by the dispatcher.
#[derive(Debug, Clone, Copy)]
pub struct GpuLimitsSummary {
    pub max_buffer_size: u64,
    pub max_storage_buffer_binding_size: u32,
    pub max_compute_invocations_per_workgroup: u32,
    pub max_compute_workgroup_storage_size: u32,
    pub max_compute_workgroups_per_dimension: u32,
}

/// Summary of texture-limit fields clamped during `request_device` limit negotiation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DeviceLimitClampReport {
    pub max_texture_dimension_1d: bool,
    pub max_texture_dimension_2d: bool,
    pub max_texture_dimension_3d: bool,
    pub max_texture_array_layers: bool,
}

impl DeviceLimitClampReport {
    /// True if any tracked limit was reduced to fit the adapter.
    pub fn any_clamped(self) -> bool {
        self.max_texture_dimension_1d
            || self.max_texture_dimension_2d
            || self.max_texture_dimension_3d
            || self.max_texture_array_layers
    }
}

/// Clamp a requested `wgpu::Limits` set to the adapter's supported limits.
///
/// This primarily protects device creation on mobile/embedded stacks (including some Panthor-based
/// Mali/Immortalis systems) where `wgpu::Limits::default()` can request a larger
/// `max_texture_dimension_3d` than the driver advertises.
pub fn clamp_required_limits_to_supported(
    mut requested: Limits,
    supported: &Limits,
) -> (Limits, DeviceLimitClampReport) {
    let mut report = DeviceLimitClampReport::default();

    if requested.max_texture_dimension_1d > supported.max_texture_dimension_1d {
        requested.max_texture_dimension_1d = supported.max_texture_dimension_1d;
        report.max_texture_dimension_1d = true;
    }
    if requested.max_texture_dimension_2d > supported.max_texture_dimension_2d {
        requested.max_texture_dimension_2d = supported.max_texture_dimension_2d;
        report.max_texture_dimension_2d = true;
    }
    if requested.max_texture_dimension_3d > supported.max_texture_dimension_3d {
        requested.max_texture_dimension_3d = supported.max_texture_dimension_3d;
        report.max_texture_dimension_3d = true;
    }
    if requested.max_texture_array_layers > supported.max_texture_array_layers {
        requested.max_texture_array_layers = supported.max_texture_array_layers;
        report.max_texture_array_layers = true;
    }

    (requested, report)
}

/// Conservative, adapter-safe replacement for `wgpu::Limits::default()` device requests.
///
/// It starts from `Limits::default()` and clamps problematic texture dimensions to the adapter's
/// supported values, preserving compatibility while avoiding accidental over-requests on mobile GPUs.
pub fn safe_default_required_limits_for_adapter(
    adapter: &Adapter,
) -> (Limits, DeviceLimitClampReport) {
    let supported = adapter.limits();
    clamp_required_limits_to_supported(Limits::default(), &supported)
}

impl From<Limits> for GpuLimitsSummary {
    fn from(limits: Limits) -> Self {
        Self {
            max_buffer_size: limits.max_buffer_size,
            max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
            max_compute_invocations_per_workgroup: limits.max_compute_invocations_per_workgroup,
            max_compute_workgroup_storage_size: limits.max_compute_workgroup_storage_size,
            max_compute_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
        }
    }
}

/// Heuristic profile used by GMS dispatch planning.
#[derive(Debug, Clone)]
pub struct GpuAdapterProfile {
    pub index: usize,
    pub name: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub backend: wgpu::Backend,
    pub device_type: DeviceType,
    pub memory_topology: MemoryTopology,
    pub compute_unit_kind: ComputeUnitKind,
    pub estimated_compute_units: u32,
    pub compute_unit_source: ComputeUnitEstimateSource,
    /// Optional note describing native-probe success/failure and fallback reason.
    pub compute_unit_probe_note: Option<String>,
    pub estimated_vram_mb: u64,
    pub estimated_bandwidth_gbps: f64,
    pub supports_mappable_primary_buffers: bool,
    pub limits: GpuLimitsSummary,
    pub score: u64,
    pub score_breakdown: GpuScoreBreakdown,
}

impl GpuAdapterProfile {
    /// True when the adapter is a viable GPU target for compute-heavy GMS workloads.
    pub fn is_usable_gpu(&self) -> bool {
        !matches!(self.device_type, DeviceType::Cpu)
            && !matches!(self.memory_topology, MemoryTopology::System)
            && self.score > 0
    }

    /// Relative score ratio vs another adapter score, for debug output and planning.
    pub fn score_ratio_against(&self, other_score: u64) -> f64 {
        if other_score == 0 {
            0.0
        } else {
            self.score as f64 / other_score as f64
        }
    }

    /// Compact `(label, count)` pair for benchmark UIs.
    pub fn compute_unit_summary(&self) -> (&'static str, u32) {
        (
            self.compute_unit_kind.short_label(),
            self.estimated_compute_units,
        )
    }
}

/// Result of adapter discovery and ranking.
#[derive(Debug, Clone, Default)]
pub struct GpuInventory {
    pub adapters: Vec<GpuAdapterProfile>,
}

impl GpuInventory {
    /// Discover all available adapters using `wgpu::Instance::enumerate_adapters`.
    pub fn discover() -> Self {
        let instance = Instance::default();
        Self::discover_with_instance(&instance)
    }

    /// Discover adapters using a caller-provided instance.
    pub fn discover_with_instance(instance: &Instance) -> Self {
        let mut adapters = pollster::block_on(instance.enumerate_adapters(Backends::all()))
            .into_iter()
            .enumerate()
            .map(|(index, adapter)| build_profile(index, &adapter))
            .collect::<Vec<_>>();

        adapters.sort_by(|left, right| {
            right
                .score
                .cmp(&left.score)
                .then_with(|| left.index.cmp(&right.index))
        });

        Self { adapters }
    }

    /// Highest-ranked adapter profile.
    pub fn best(&self) -> Option<&GpuAdapterProfile> {
        self.adapters.first()
    }

    /// Sum of scores across usable GPU adapters.
    pub fn total_usable_score(&self) -> u64 {
        self.adapters
            .iter()
            .filter(|adapter| adapter.is_usable_gpu())
            .map(|adapter| adapter.score)
            .sum()
    }

    /// Return only adapters that should participate in GMS workload dispatch.
    pub fn usable_adapters(&self) -> impl Iterator<Item = &GpuAdapterProfile> {
        self.adapters
            .iter()
            .filter(|adapter| adapter.is_usable_gpu())
    }
}

fn build_profile(index: usize, adapter: &Adapter) -> GpuAdapterProfile {
    let info = adapter.get_info();
    let limits = adapter.limits();
    let features = adapter.features();

    let name_lower = info.name.to_ascii_lowercase();
    let vendor_family = vendor_family(info.vendor, &name_lower);
    let device_type = info.device_type;
    let memory_topology = classify_memory_topology(device_type, vendor_family, &name_lower);

    let (estimated_compute_units, compute_unit_source, compute_unit_probe_note) =
        estimate_compute_units(&info, &limits, &name_lower, vendor_family);
    let compute_unit_kind = estimate_compute_unit_kind(vendor_family);
    let estimated_vram_mb = estimate_vram_mb(&info, &limits, &name_lower, memory_topology);
    let estimated_bandwidth_gbps = estimate_bandwidth_gbps(
        &info,
        &limits,
        estimated_compute_units,
        estimated_vram_mb,
        memory_topology,
    );
    let supports_mappable_primary_buffers = features.contains(Features::MAPPABLE_PRIMARY_BUFFERS);

    let limits_summary = GpuLimitsSummary::from(limits.clone());
    let score_breakdown = score_adapter(
        estimated_compute_units,
        estimated_bandwidth_gbps,
        estimated_vram_mb,
        memory_topology,
        &limits,
        supports_mappable_primary_buffers,
    );

    GpuAdapterProfile {
        index,
        name: info.name,
        vendor_id: info.vendor,
        device_id: info.device,
        backend: info.backend,
        device_type,
        memory_topology,
        compute_unit_kind,
        estimated_compute_units,
        compute_unit_source,
        compute_unit_probe_note,
        estimated_vram_mb,
        estimated_bandwidth_gbps,
        supports_mappable_primary_buffers,
        limits: limits_summary,
        score: score_breakdown.final_score,
        score_breakdown,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VendorFamily {
    Nvidia,
    Amd,
    Intel,
    Apple,
    Arm,
    Other,
}

fn vendor_family(vendor_id: u32, name_lower: &str) -> VendorFamily {
    match vendor_id {
        NVIDIA_VENDOR_ID => VendorFamily::Nvidia,
        AMD_VENDOR_ID => VendorFamily::Amd,
        INTEL_VENDOR_ID => VendorFamily::Intel,
        APPLE_VENDOR_ID => VendorFamily::Apple,
        ARM_VENDOR_ID => VendorFamily::Arm,
        _ => {
            if name_lower.contains("nvidia")
                || name_lower.contains("rtx")
                || name_lower.contains("gtx")
            {
                VendorFamily::Nvidia
            } else if name_lower.contains("radeon")
                || name_lower.contains("rx ")
                || name_lower.contains("amd")
            {
                VendorFamily::Amd
            } else if name_lower.contains("intel")
                || name_lower.contains("arc")
                || name_lower.contains("iris")
            {
                VendorFamily::Intel
            } else if name_lower.contains("apple")
                || name_lower.contains("m1")
                || name_lower.contains("m2")
                || name_lower.contains("m3")
                || name_lower.contains("m4")
            {
                VendorFamily::Apple
            } else if name_lower.contains("immortalis")
                || name_lower.contains("mali")
                || name_lower.contains("arm mali")
            {
                VendorFamily::Arm
            } else {
                VendorFamily::Other
            }
        }
    }
}

fn classify_memory_topology(
    device_type: DeviceType,
    vendor: VendorFamily,
    name_lower: &str,
) -> MemoryTopology {
    match device_type {
        DeviceType::DiscreteGpu => MemoryTopology::DiscreteVram,
        DeviceType::IntegratedGpu => MemoryTopology::Unified,
        DeviceType::VirtualGpu => MemoryTopology::Virtualized,
        DeviceType::Cpu => MemoryTopology::System,
        _ => {
            // Some mobile Vulkan stacks report ARM Mali/Immortalis adapters as `Other`.
            // Treat them as unified-memory GPUs so dispatch/zero-copy heuristics remain sane.
            if matches!(vendor, VendorFamily::Arm)
                || name_lower.contains("immortalis")
                || name_lower.contains("mali")
            {
                MemoryTopology::Unified
            } else {
                MemoryTopology::Unknown
            }
        }
    }
}

fn estimate_compute_unit_kind(vendor: VendorFamily) -> ComputeUnitKind {
    match vendor {
        VendorFamily::Amd => ComputeUnitKind::Cu,
        VendorFamily::Nvidia => ComputeUnitKind::Sm,
        VendorFamily::Intel | VendorFamily::Apple | VendorFamily::Arm => {
            ComputeUnitKind::CoreCluster
        }
        VendorFamily::Other => ComputeUnitKind::Unknown,
    }
}

fn estimate_compute_units(
    info: &AdapterInfo,
    limits: &Limits,
    name_lower: &str,
    vendor: VendorFamily,
) -> (u32, ComputeUnitEstimateSource, Option<String>) {
    let native_probe = probe_native_compute_units(info, vendor, name_lower);
    if let Some(native) = native_probe.units {
        return (
            native.max(1),
            ComputeUnitEstimateSource::NativeProbe,
            native_probe.note,
        );
    }

    if let Some(parsed) = parse_known_compute_units(name_lower, vendor) {
        return (
            parsed.max(1),
            ComputeUnitEstimateSource::DeviceNameTable,
            native_probe.note,
        );
    }

    // Fallback heuristic from compute limits.
    // This is not a real CU/SM count. We derive a coarse estimate from limits so that
    // dispatch weighting remains stable even on drivers/backends that hide hardware details.
    let invocations = limits.max_compute_invocations_per_workgroup.max(64) as f64;
    let workgroups = limits.max_compute_workgroups_per_dimension.max(1) as f64;
    let storage_kb = (limits.max_compute_workgroup_storage_size.max(16_384) as f64) / 1024.0;
    let buffer_gb = (limits.max_buffer_size.max(1) as f64) / (1024.0 * 1024.0 * 1024.0);

    let device_bias = match info.device_type {
        DeviceType::DiscreteGpu => 1.6,
        DeviceType::IntegratedGpu => 1.0,
        DeviceType::VirtualGpu => 0.6,
        DeviceType::Cpu => 0.25,
        _ => 0.8,
    };

    let estimate = ((invocations / 64.0)
        + (workgroups.log2().max(1.0) * 2.0)
        + (storage_kb / 8.0)
        + (buffer_gb.sqrt() * 6.0))
        * device_bias;

    (
        estimate.round().clamp(1.0, 256.0) as u32,
        ComputeUnitEstimateSource::DriverLimitsHeuristic,
        native_probe
            .note
            .or_else(|| Some("device-name table had no match".to_owned())),
    )
}

#[derive(Debug, Clone, Default)]
struct NativeProbeOutcome {
    units: Option<u32>,
    note: Option<String>,
}

impl NativeProbeOutcome {
    fn found(units: u32, provider: &'static str) -> Self {
        Self {
            units: Some(units),
            note: Some(format!("native probe via {provider}")),
        }
    }

    fn failed(note: String) -> Self {
        Self {
            units: None,
            note: Some(note),
        }
    }

    fn none() -> Self {
        Self::default()
    }
}

#[derive(Debug, Clone)]
struct NativeProbeCache {
    tool: &'static str,
    entries: Vec<(String, u32)>,
    load_error: Option<String>,
}

impl NativeProbeCache {
    fn ok(tool: &'static str, entries: Vec<(String, u32)>) -> Self {
        Self {
            tool,
            entries,
            load_error: None,
        }
    }

    fn error(tool: &'static str, load_error: String) -> Self {
        Self {
            tool,
            entries: Vec::new(),
            load_error: Some(load_error),
        }
    }
}

fn probe_native_compute_units(
    info: &AdapterInfo,
    vendor: VendorFamily,
    name_lower: &str,
) -> NativeProbeOutcome {
    match vendor {
        VendorFamily::Nvidia => probe_nvidia_smi_compute_units(name_lower),
        VendorFamily::Amd => probe_rocminfo_compute_units(name_lower),
        VendorFamily::Apple => probe_macos_gpu_core_count(info, name_lower),
        VendorFamily::Arm => probe_arm_compute_units(info, name_lower),
        VendorFamily::Intel | VendorFamily::Other => NativeProbeOutcome::none(),
    }
}

fn probe_arm_compute_units(info: &AdapterInfo, name_lower: &str) -> NativeProbeOutcome {
    let _ = info;
    let vk = probe_vulkaninfo_arm_compute_units(name_lower);
    if vk.units.is_some() {
        return vk;
    }
    let driver = probe_linux_arm_gpu_driver_note(name_lower);
    merge_native_probe_outcomes(vk, driver)
}

#[cfg(target_os = "linux")]
fn probe_linux_arm_gpu_driver_note(name_lower: &str) -> NativeProbeOutcome {
    if !name_lower.contains("mali") && !name_lower.contains("immortalis") {
        return NativeProbeOutcome::none();
    }

    static DRIVER_HINT: OnceLock<Option<&'static str>> = OnceLock::new();
    let hint = DRIVER_HINT.get_or_init(|| {
        if Path::new("/sys/module/panthor").exists() {
            Some("panthor")
        } else if Path::new("/sys/module/panfrost").exists() {
            Some("panfrost")
        } else {
            None
        }
    });

    match hint {
        Some(driver) => NativeProbeOutcome::failed(format!(
            "{driver} driver detected; CU/core-cluster count not exposed via portable probe"
        )),
        None => NativeProbeOutcome::none(),
    }
}

#[cfg(not(target_os = "linux"))]
fn probe_linux_arm_gpu_driver_note(_name_lower: &str) -> NativeProbeOutcome {
    NativeProbeOutcome::none()
}

fn probe_vulkaninfo_arm_compute_units(name_lower: &str) -> NativeProbeOutcome {
    if !name_lower.contains("mali") && !name_lower.contains("immortalis") {
        return NativeProbeOutcome::none();
    }

    static CACHE: OnceLock<NativeProbeCache> = OnceLock::new();
    let entries = CACHE.get_or_init(load_vulkaninfo_arm_cluster_cache);
    match_command_probe(entries, name_lower)
}

fn load_vulkaninfo_arm_cluster_cache() -> NativeProbeCache {
    let output = match Command::new("vulkaninfo").output() {
        Ok(output) => output,
        Err(err) => return NativeProbeCache::error("vulkaninfo", err.to_string()),
    };
    if !output.status.success() {
        return NativeProbeCache::error(
            "vulkaninfo",
            format!(
                "command failed (exit {:?}): {}",
                output.status.code(),
                compact_command_stderr(&output.stderr)
            ),
        );
    }

    let text = String::from_utf8_lossy(&output.stdout);
    let entries = parse_vulkaninfo_arm_cluster_entries(&text);
    if entries.is_empty() {
        NativeProbeCache::error(
            "vulkaninfo",
            "no parseable ARM shader core count entries (e.g. `shaderCoreCountARM`)".to_owned(),
        )
    } else {
        NativeProbeCache::ok("vulkaninfo", entries)
    }
}

fn merge_native_probe_outcomes(
    primary: NativeProbeOutcome,
    secondary: NativeProbeOutcome,
) -> NativeProbeOutcome {
    if primary.units.is_some() {
        return primary;
    }
    if secondary.units.is_some() {
        return secondary;
    }

    let mut notes = Vec::new();
    if let Some(note) = primary.note {
        notes.push(note);
    }
    if let Some(note) = secondary.note {
        notes.push(note);
    }
    NativeProbeOutcome {
        units: None,
        note: if notes.is_empty() {
            None
        } else {
            Some(notes.join("; "))
        },
    }
}

fn probe_nvidia_smi_compute_units(name_lower: &str) -> NativeProbeOutcome {
    static CACHE: OnceLock<NativeProbeCache> = OnceLock::new();
    let entries = CACHE.get_or_init(load_nvidia_smi_sm_cache);
    match_command_probe(entries, name_lower)
}

fn load_nvidia_smi_sm_cache() -> NativeProbeCache {
    let output = match Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,multiprocessor_count",
            "--format=csv,noheader,nounits",
        ])
        .output()
    {
        Ok(output) => output,
        Err(err) => return NativeProbeCache::error("nvidia-smi", err.to_string()),
    };
    if !output.status.success() {
        return NativeProbeCache::error(
            "nvidia-smi",
            format!(
                "query failed (exit {:?}): {}",
                output.status.code(),
                compact_command_stderr(&output.stderr)
            ),
        );
    }

    let entries = String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| {
            let (name, value) = line.split_once(',')?;
            let units = value.trim().parse::<u32>().ok()?;
            Some((normalize_gpu_name(name), units))
        })
        .collect::<Vec<_>>();

    if entries.is_empty() {
        NativeProbeCache::error(
            "nvidia-smi",
            "query returned no parseable `multiprocessor_count` rows".to_owned(),
        )
    } else {
        NativeProbeCache::ok("nvidia-smi", entries)
    }
}

fn probe_rocminfo_compute_units(name_lower: &str) -> NativeProbeOutcome {
    static CACHE: OnceLock<NativeProbeCache> = OnceLock::new();
    let entries = CACHE.get_or_init(load_rocminfo_cu_cache);
    match_command_probe(entries, name_lower)
}

fn load_rocminfo_cu_cache() -> NativeProbeCache {
    let output = match Command::new("rocminfo").output() {
        Ok(output) => output,
        Err(err) => return NativeProbeCache::error("rocminfo", err.to_string()),
    };
    if !output.status.success() {
        return NativeProbeCache::error(
            "rocminfo",
            format!(
                "command failed (exit {:?}): {}",
                output.status.code(),
                compact_command_stderr(&output.stderr)
            ),
        );
    }

    let text = String::from_utf8_lossy(&output.stdout);
    let mut entries = Vec::new();
    let mut current_name: Option<String> = None;
    let mut current_cu: Option<u32> = None;

    for raw_line in text.lines() {
        let line = raw_line.trim();
        if line.is_empty() {
            if let (Some(name), Some(cu)) = (current_name.take(), current_cu.take()) {
                entries.push((normalize_gpu_name(&name), cu));
            }
            continue;
        }

        if let Some((key, value)) = line.split_once(':') {
            let key = key.trim().to_ascii_lowercase();
            let value = value.trim();
            if key == "marketing name" || key == "name" {
                // `rocminfo` includes CPU agents too; we keep the latest name and rely on CU parse
                // + name matching to filter out non-GPU entries.
                current_name = Some(value.to_owned());
            } else if key == "compute unit" {
                current_cu = value.parse::<u32>().ok();
            }
        }
    }

    if let (Some(name), Some(cu)) = (current_name.take(), current_cu.take()) {
        entries.push((normalize_gpu_name(&name), cu));
    }

    if entries.is_empty() {
        NativeProbeCache::error(
            "rocminfo",
            "no parseable GPU compute-unit entries".to_owned(),
        )
    } else {
        NativeProbeCache::ok("rocminfo", entries)
    }
}

#[cfg(target_os = "macos")]
fn probe_macos_gpu_core_count(info: &AdapterInfo, name_lower: &str) -> NativeProbeOutcome {
    let _ = info;
    static CACHE: OnceLock<NativeProbeCache> = OnceLock::new();
    let entries = CACHE.get_or_init(load_system_profiler_gpu_cores_cache);
    match_command_probe(entries, name_lower)
}

#[cfg(not(target_os = "macos"))]
fn probe_macos_gpu_core_count(_info: &AdapterInfo, _name_lower: &str) -> NativeProbeOutcome {
    NativeProbeOutcome::none()
}

#[cfg(target_os = "macos")]
fn load_system_profiler_gpu_cores_cache() -> NativeProbeCache {
    let output = match Command::new("system_profiler")
        .args(["SPDisplaysDataType"])
        .output()
    {
        Ok(output) => output,
        Err(err) => return NativeProbeCache::error("system_profiler", err.to_string()),
    };
    if !output.status.success() {
        return NativeProbeCache::error(
            "system_profiler",
            format!(
                "command failed (exit {:?}): {}",
                output.status.code(),
                compact_command_stderr(&output.stderr)
            ),
        );
    }

    let text = String::from_utf8_lossy(&output.stdout);
    let mut entries = Vec::new();
    let mut current_name: Option<String> = None;
    let mut current_cores: Option<u32> = None;

    for raw_line in text.lines() {
        let line = raw_line.trim();
        if let Some(rest) = line.strip_prefix("Chipset Model:") {
            if let (Some(name), Some(cores)) = (current_name.take(), current_cores.take()) {
                entries.push((normalize_gpu_name(&name), cores));
            }
            current_name = Some(rest.trim().to_owned());
            continue;
        }
        if let Some(rest) = line.strip_prefix("Total Number of Cores:") {
            current_cores = rest.trim().parse::<u32>().ok();
        }
    }

    if let (Some(name), Some(cores)) = (current_name.take(), current_cores.take()) {
        entries.push((normalize_gpu_name(&name), cores));
    }

    if entries.is_empty() {
        NativeProbeCache::error(
            "system_profiler",
            "no parseable `Total Number of Cores` entries".to_owned(),
        )
    } else {
        NativeProbeCache::ok("system_profiler", entries)
    }
}

fn match_command_probe(cache: &NativeProbeCache, name_lower: &str) -> NativeProbeOutcome {
    if let Some(units) = match_probe_entries_by_name(&cache.entries, name_lower) {
        return NativeProbeOutcome::found(units, cache.tool);
    }

    if let Some(error) = cache.load_error.as_ref() {
        return NativeProbeOutcome::failed(format!("{} unavailable: {}", cache.tool, error));
    }

    let normalized_target = normalize_gpu_name(name_lower);
    NativeProbeOutcome::failed(format!(
        "{} returned {} adapter entr{} but no name match for '{}'",
        cache.tool,
        cache.entries.len(),
        if cache.entries.len() == 1 { "y" } else { "ies" },
        normalized_target
    ))
}

fn compact_command_stderr(stderr: &[u8]) -> String {
    let text = String::from_utf8_lossy(stderr);
    let line = text.lines().next().unwrap_or("").trim();
    if line.is_empty() {
        "no stderr output".to_owned()
    } else {
        line.chars().take(220).collect()
    }
}

fn match_probe_entries_by_name(entries: &[(String, u32)], name_lower: &str) -> Option<u32> {
    let normalized_target = normalize_gpu_name(name_lower);
    if normalized_target.is_empty() {
        return None;
    }

    // Prefer the longest matching probe name to avoid generic substring collisions.
    entries
        .iter()
        .filter(|(probe_name, units)| {
            *units > 0
                && !probe_name.is_empty()
                && (normalized_target.contains(probe_name)
                    || probe_name.contains(&normalized_target))
        })
        .max_by_key(|(probe_name, _)| probe_name.len())
        .map(|(_, units)| *units)
}

fn normalize_gpu_name(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    let mut last_space = false;
    for ch in name.chars().flat_map(|c| c.to_lowercase()) {
        let keep = ch.is_ascii_alphanumeric();
        if keep {
            out.push(ch);
            last_space = false;
        } else if !last_space {
            out.push(' ');
            last_space = true;
        }
    }
    out.trim().to_owned()
}

fn parse_vulkaninfo_arm_cluster_entries(text: &str) -> Vec<(String, u32)> {
    let mut entries = Vec::<(String, u32)>::new();
    let mut current_name: Option<String> = None;
    let mut current_units: Option<u32> = None;

    let flush = |entries: &mut Vec<(String, u32)>,
                 current_name: &mut Option<String>,
                 current_units: &mut Option<u32>| {
        let Some(name) = current_name.take() else {
            *current_units = None;
            return;
        };
        let Some(units) = current_units.take() else {
            return;
        };
        if units == 0 {
            return;
        }
        let normalized = normalize_gpu_name(&name);
        if normalized.contains("mali") || normalized.contains("immortalis") {
            entries.push((normalized, units));
        }
    };

    for raw_line in text.lines() {
        let line = raw_line.trim();
        if line.is_empty() {
            flush(&mut entries, &mut current_name, &mut current_units);
            continue;
        }

        if let Some(name) = parse_vulkaninfo_gpu_name_line(line) {
            flush(&mut entries, &mut current_name, &mut current_units);
            current_name = Some(name);
            continue;
        }

        let Some((key, value)) = split_kv_line(line) else {
            continue;
        };

        let key_normalized = normalize_gpu_name(key);
        let key_compact = key_normalized.replace(' ', "");

        if key_compact == "devicename" {
            flush(&mut entries, &mut current_name, &mut current_units);
            current_name = Some(value.trim().to_owned());
            continue;
        }

        // ARM Vulkan property names observed/expected variants:
        // - `shaderCoreCountARM`
        // - `shaderCoreCount`
        // We keep this parser tolerant because `vulkaninfo` formatting varies by version.
        let is_arm_shader_core_count = key_compact.contains("shadercorecount")
            && !key_compact.contains("mask")
            && (!current_name
                .as_deref()
                .map(normalize_gpu_name)
                .unwrap_or_default()
                .is_empty()
                || key_compact.contains("arm"));

        if is_arm_shader_core_count {
            if let Some(units) = parse_first_u32(value) {
                current_units = Some(units);
            }
        }
    }

    flush(&mut entries, &mut current_name, &mut current_units);
    entries
}

fn parse_vulkaninfo_gpu_name_line(line: &str) -> Option<String> {
    if let Some((prefix, rest)) = line.split_once('=') {
        let key = normalize_gpu_name(prefix);
        if key.replace(' ', "") == "devicename" {
            let value = rest.trim();
            if !value.is_empty() {
                return Some(value.to_owned());
            }
        }
    }

    // `vulkaninfo --summary` style line: `GPU id = 0 (Mali-G715 ... )`
    if line.to_ascii_lowercase().starts_with("gpu id") {
        let open = line.find('(')?;
        let close = line.rfind(')')?;
        if close > open + 1 {
            return Some(line[open + 1..close].trim().to_owned());
        }
    }

    None
}

fn split_kv_line(line: &str) -> Option<(&str, &str)> {
    line.split_once('=').or_else(|| line.split_once(':'))
}

fn parse_first_u32(text: &str) -> Option<u32> {
    let mut start = None;
    for (i, ch) in text.char_indices() {
        if ch.is_ascii_digit() {
            start = Some(i);
            break;
        }
    }
    let start = start?;
    let rest = &text[start..];
    let end = rest
        .char_indices()
        .find_map(|(i, ch)| (!ch.is_ascii_digit()).then_some(i))
        .unwrap_or(rest.len());
    rest[..end].parse::<u32>().ok()
}

fn parse_known_compute_units(name_lower: &str, vendor: VendorFamily) -> Option<u32> {
    match vendor {
        VendorFamily::Nvidia => parse_nvidia_sm(name_lower),
        VendorFamily::Amd => parse_amd_cu(name_lower),
        VendorFamily::Apple => parse_apple_gpu_clusters(name_lower),
        VendorFamily::Intel => parse_intel_gpu_clusters(name_lower),
        VendorFamily::Arm => parse_arm_gpu_clusters(name_lower),
        VendorFamily::Other => None,
    }
}

fn parse_arm_gpu_clusters(name: &str) -> Option<u32> {
    // ARM Mali/Immortalis adapters often encode shader core-cluster count as `MC<n>` / `MP<n>`.
    // Examples: `Immortalis-G720 MC12`, `Mali-G78 MP14`.
    parse_arm_cluster_suffix(name, "mc").or_else(|| parse_arm_cluster_suffix(name, "mp"))
}

fn parse_arm_cluster_suffix(name: &str, prefix: &str) -> Option<u32> {
    let chars = name.chars().collect::<Vec<_>>();
    for i in 0..chars.len() {
        if !chars[i].eq_ignore_ascii_case(&prefix.chars().next()?) {
            continue;
        }
        let next = i + 1;
        if next >= chars.len() || !chars[next].eq_ignore_ascii_case(&prefix.chars().nth(1)?) {
            continue;
        }

        let mut j = next + 1;
        // Accept separators like `mc-12`, `mc 12`, `mc12`.
        while j < chars.len() && (chars[j] == '-' || chars[j] == '_' || chars[j].is_whitespace()) {
            j += 1;
        }
        let start = j;
        while j < chars.len() && chars[j].is_ascii_digit() {
            j += 1;
        }
        if start == j {
            continue;
        }
        let digits = chars[start..j].iter().collect::<String>();
        if let Ok(count) = digits.parse::<u32>() {
            if (1..=64).contains(&count) {
                return Some(count);
            }
        }
    }
    None
}

fn parse_nvidia_sm(name: &str) -> Option<u32> {
    // GeForce RTX (desktop-focused) SM coverage from RTX 2000 onward.
    //
    // Ordering matters because we use substring matching:
    // - longer / more specific patterns must appear before base models
    // - e.g. "rtx 4070 ti super" before "rtx 4070 ti" before "rtx 4070"
    //
    // Laptop/OEM variants that report distinct names may still fall back unless explicitly listed.
    const TABLE: &[(&str, u32)] = &[
        // RTX 50 series (validated entries first; add more as device names are verified)
        ("rtx 5060 ti", 36),
        // RTX 40 series
        ("rtx 4090 d", 114),
        ("rtx 4090", 128),
        ("rtx 4080 super", 80),
        ("rtx 4080", 76),
        ("rtx 4070 ti super", 66),
        ("rtx 4070 ti", 60),
        ("rtx 4070 super", 56),
        ("rtx 4070", 46),
        ("rtx 4060 ti", 34),
        ("rtx 4060", 24),
        ("rtx 4050", 20),
        // RTX 30 series
        ("rtx 3090 ti", 84),
        ("rtx 3090", 82),
        ("rtx 3080 ti", 80),
        ("rtx 3080 12gb", 70),
        ("rtx 3080", 68),
        ("rtx 3070 ti", 48),
        ("rtx 3070", 46),
        ("rtx 3060 ti", 38),
        ("rtx 3060", 28),
        ("rtx 3050 6gb", 18),
        ("rtx 3050 ti", 20),
        ("rtx 3050", 20),
        // RTX 20 series / Turing refresh
        ("titan rtx", 72),
        ("rtx 2080 ti", 68),
        ("rtx 2080 super", 48),
        ("rtx 2080", 46),
        ("rtx 2070 super", 40),
        ("rtx 2070", 36),
        ("rtx 2060 super", 34),
        ("rtx 2060", 30),
        ("rtx 2050", 16),
        // Pre-RTX fallback entries kept for older benchmark machines
        ("gtx 1660", 22),
        ("gtx 1650", 14),
    ];
    lookup_table_contains(name, TABLE)
}

fn parse_amd_cu(name: &str) -> Option<u32> {
    // Radeon RX desktop-focused CU coverage from RX 6000 onward.
    //
    // We keep this table explicit because many drivers/backends expose only the marketing name,
    // and CU count is the primary weighting factor in GMS scheduling.
    const TABLE: &[(&str, u32)] = &[
        // RX 7000 series (RDNA3)
        ("rx 7900 xtx", 96),
        ("rx 7900 xt", 84),
        ("rx 7900 gre", 80),
        ("rx 7900m", 72),
        ("rx 7800 xt", 60),
        ("rx 7700 xt", 54),
        ("rx 7600 xt", 32),
        ("rx 7600", 32),
        // RX 6000 series (RDNA2 + refresh)
        ("rx 6950 xt", 80),
        ("rx 6900 xt", 80),
        ("rx 6800 xt", 72),
        ("rx 6800", 60),
        ("rx 6750 xt", 40),
        ("rx 6700 xt", 40),
        ("rx 6700", 36),
        ("rx 6650 xt", 32),
        ("rx 6600 xt", 32),
        ("rx 6600", 28),
        ("rx 6500 xt", 16),
        ("rx 6400", 12),
        ("rx 6300", 8),
    ];
    lookup_table_contains(name, TABLE)
}

fn parse_apple_gpu_clusters(name: &str) -> Option<u32> {
    // Apple adapters often report only SoC names. We approximate cluster/core count buckets.
    if name.contains("m4 max") {
        Some(40)
    } else if name.contains("m4 pro") {
        Some(20)
    } else if name.contains("m4") {
        Some(10)
    } else if name.contains("m3 max") {
        Some(40)
    } else if name.contains("m3 pro") {
        Some(18)
    } else if name.contains("m3") {
        Some(10)
    } else if name.contains("m2 ultra") {
        Some(60)
    } else if name.contains("m2 max") {
        Some(38)
    } else if name.contains("m2 pro") {
        Some(19)
    } else if name.contains("m2") {
        Some(10)
    } else if name.contains("m1 ultra") {
        Some(64)
    } else if name.contains("m1 max") {
        Some(32)
    } else if name.contains("m1 pro") {
        Some(16)
    } else if name.contains("m1") {
        Some(8)
    } else {
        None
    }
}

fn parse_intel_gpu_clusters(name: &str) -> Option<u32> {
    if name.contains("arc a770") {
        Some(32)
    } else if name.contains("arc a750") {
        Some(28)
    } else if name.contains("arc a580") {
        Some(24)
    } else if name.contains("arc") {
        Some(16)
    } else if name.contains("iris") || name.contains("uhd") {
        Some(8)
    } else {
        None
    }
}

fn lookup_table_contains(name: &str, table: &[(&str, u32)]) -> Option<u32> {
    table
        .iter()
        .find(|(pattern, _)| name.contains(*pattern))
        .map(|(_, value)| *value)
}

#[cfg(test)]
mod tests {
    use super::{
        clamp_required_limits_to_supported, classify_memory_topology, merge_native_probe_outcomes,
        parse_amd_cu, parse_arm_gpu_clusters, parse_nvidia_sm, parse_vulkaninfo_arm_cluster_entries,
        vendor_family, MemoryTopology, NativeProbeOutcome, VendorFamily,
    };
    use wgpu::DeviceType;

    #[test]
    fn parses_nvidia_rtx_5060_ti_sm() {
        assert_eq!(parse_nvidia_sm("nvidia geforce rtx 5060 ti"), Some(36));
    }

    #[test]
    fn parses_nvidia_specific_variants_before_base_model() {
        assert_eq!(
            parse_nvidia_sm("nvidia geforce rtx 4070 ti super"),
            Some(66)
        );
        assert_eq!(parse_nvidia_sm("nvidia geforce rtx 4080 super"), Some(80));
        assert_eq!(parse_nvidia_sm("nvidia geforce rtx 3080 12gb"), Some(70));
    }

    #[test]
    fn parses_amd_rx_6000_and_7000_cu_counts() {
        assert_eq!(parse_amd_cu("amd radeon rx 6950 xt"), Some(80));
        assert_eq!(parse_amd_cu("amd radeon rx 6700"), Some(36));
        assert_eq!(parse_amd_cu("amd radeon rx 6400"), Some(12));
        assert_eq!(parse_amd_cu("amd radeon rx 7900 gre"), Some(80));
    }

    #[test]
    fn parses_arm_mali_and_immortalis_mc_mp_cluster_counts() {
        assert_eq!(parse_arm_gpu_clusters("arm immortalis-g720 mc12"), Some(12));
        assert_eq!(parse_arm_gpu_clusters("mali-g78 mp14"), Some(14));
        assert_eq!(parse_arm_gpu_clusters("mali g715 mc-10"), Some(10));
    }

    #[test]
    fn detects_arm_vendor_family_by_id_and_name() {
        assert_eq!(vendor_family(0x13B5, "arm gpu"), VendorFamily::Arm);
        assert_eq!(
            vendor_family(0, "arm immortalis-g715 mc16"),
            VendorFamily::Arm
        );
        assert_eq!(vendor_family(0, "mali-g610"), VendorFamily::Arm);
    }

    #[test]
    fn classifies_arm_other_device_as_unified_memory() {
        assert_eq!(
            classify_memory_topology(DeviceType::Other, VendorFamily::Arm, "mali-g720"),
            MemoryTopology::Unified
        );
    }

    #[test]
    fn parses_vulkaninfo_arm_shader_core_count_entries() {
        let text = r#"
GPU id = 0 (Mali-G715)
VkPhysicalDeviceProperties:
    deviceName        = Mali-G715
VkPhysicalDeviceShaderCorePropertiesARM:
    shaderCoreCountARM = 10

GPU id = 1 (Immortalis-G720 MC12)
VkPhysicalDeviceProperties:
    deviceName = Immortalis-G720 MC12
VkPhysicalDeviceShaderCorePropertiesARM:
    shaderCoreCountARM: 12
"#;
        let entries = parse_vulkaninfo_arm_cluster_entries(text);
        assert!(entries
            .iter()
            .any(|(name, units)| name.contains("mali g715") && *units == 10));
        assert!(entries
            .iter()
            .any(|(name, units)| name.contains("immortalis g720 mc12") && *units == 12));
    }

    #[test]
    fn merges_native_probe_notes_when_both_paths_fail() {
        let out = merge_native_probe_outcomes(
            NativeProbeOutcome::failed("vulkaninfo unavailable".to_string()),
            NativeProbeOutcome::failed("panthor driver detected".to_string()),
        );
        assert!(out.units.is_none());
        let note = out.note.expect("merged note");
        assert!(note.contains("vulkaninfo unavailable"));
        assert!(note.contains("panthor driver detected"));
    }

    #[test]
    fn clamps_default_texture_limits_to_supported_adapter_limits() {
        let requested = wgpu::Limits::default();
        let mut supported = wgpu::Limits::default();
        supported.max_texture_dimension_1d = 4096;
        supported.max_texture_dimension_2d = 4096;
        supported.max_texture_dimension_3d = 512;
        supported.max_texture_array_layers = 128;

        let (clamped, report) = clamp_required_limits_to_supported(requested, &supported);
        assert!(report.any_clamped());
        assert!(report.max_texture_dimension_3d);
        assert_eq!(clamped.max_texture_dimension_3d, 512);
        assert_eq!(clamped.max_texture_array_layers, 128);
    }
}

fn estimate_vram_mb(
    info: &AdapterInfo,
    limits: &Limits,
    name_lower: &str,
    memory_topology: MemoryTopology,
) -> u64 {
    if let Some(name_vram_mb) = parse_name_vram_mb(name_lower) {
        return name_vram_mb.max(512);
    }

    let buffer_based_floor_mb =
        ((limits.max_buffer_size as f64) / (1024.0 * 1024.0)).round() as u64;

    let type_default_mb = match memory_topology {
        MemoryTopology::DiscreteVram => 8 * 1024,
        MemoryTopology::Unified => 4 * 1024,
        MemoryTopology::Virtualized => 2 * 1024,
        MemoryTopology::System => 1024,
        MemoryTopology::Unknown => 2 * 1024,
    };

    let vendor_adjust_mb =
        if matches!(info.device_type, DeviceType::IntegratedGpu) && name_lower.contains("apple") {
            8 * 1024
        } else {
            0
        };

    type_default_mb
        .max(buffer_based_floor_mb)
        .saturating_add(vendor_adjust_mb)
}

fn parse_name_vram_mb(name: &str) -> Option<u64> {
    let chars = name.chars().collect::<Vec<_>>();
    let mut i = 0usize;
    while i < chars.len() {
        if !chars[i].is_ascii_digit() {
            i += 1;
            continue;
        }

        let start = i;
        while i < chars.len() && chars[i].is_ascii_digit() {
            i += 1;
        }

        let number_str = chars[start..i].iter().collect::<String>();
        let value = number_str.parse::<u64>().ok()?;

        let suffix = chars[i..chars.len().min(i + 4)]
            .iter()
            .collect::<String>()
            .to_ascii_lowercase();

        if suffix.starts_with("gb") || suffix.starts_with(" g") {
            if (1..=192).contains(&value) {
                return Some(value * 1024);
            }
        }
    }
    None
}

fn estimate_bandwidth_gbps(
    info: &AdapterInfo,
    limits: &Limits,
    estimated_compute_units: u32,
    estimated_vram_mb: u64,
    memory_topology: MemoryTopology,
) -> f64 {
    let cu = estimated_compute_units.max(1) as f64;
    let vram_gb = (estimated_vram_mb as f64 / 1024.0).max(1.0);
    let buffer_scale = ((limits.max_storage_buffer_binding_size.max(1) as f64)
        / (128.0 * 1024.0 * 1024.0))
        .clamp(0.5, 8.0);

    let device_type_factor = match info.device_type {
        DeviceType::DiscreteGpu => 1.0,
        DeviceType::IntegratedGpu => 0.55,
        DeviceType::VirtualGpu => 0.35,
        DeviceType::Cpu => 0.10,
        _ => 0.40,
    };

    let base_gbps = (cu * 7.5) + (vram_gb * 6.0) + (buffer_scale * 18.0);
    let topology_bias = match memory_topology {
        MemoryTopology::DiscreteVram => 1.15,
        MemoryTopology::Unified => 0.85,
        MemoryTopology::Virtualized => 0.60,
        MemoryTopology::System => 0.25,
        MemoryTopology::Unknown => 0.75,
    };

    (base_gbps * device_type_factor * topology_bias).clamp(10.0, 2000.0)
}

fn score_adapter(
    estimated_compute_units: u32,
    estimated_bandwidth_gbps: f64,
    estimated_vram_mb: u64,
    memory_topology: MemoryTopology,
    limits: &Limits,
    supports_mappable_primary_buffers: bool,
) -> GpuScoreBreakdown {
    let cu = estimated_compute_units.max(1) as f64;
    let bandwidth = estimated_bandwidth_gbps.max(1.0);
    let vram_gb = (estimated_vram_mb as f64 / 1024.0).max(1.0);

    // --- Scoring rationale (English, detailed by request) ---
    // 1) CU/SM count is the primary multiplier because we are targeting per-CU / per-SM
    //    workload partitioning. This term should dominate workgroup distribution.
    let cu_term = cu * 100.0;

    // 2) Memory bandwidth is the next strongest term because graphics/physics updates are
    //    frequently transfer-bound or memory-bound. We normalize by ~25 GB/s chunks.
    let bandwidth_term = (bandwidth / 25.0) * 35.0;

    // 3) VRAM capacity matters for texture-heavy simulations and larger working sets.
    //    We use sqrt() so 16 GB beats 8 GB, but capacity does not overpower CU throughput.
    let vram_term = vram_gb.sqrt() * 25.0;

    // 4) `wgpu` limits are used as a soft capability signal. They do not reveal CU counts,
    //    but higher storage buffer limits and compute invocations generally correlate with
    //    more capable hardware/driver paths.
    let limits_term = ((limits.max_compute_invocations_per_workgroup as f64) / 256.0) * 10.0
        + ((limits.max_storage_buffer_binding_size as f64) / (256.0 * 1024.0 * 1024.0)) * 8.0
        + ((limits.max_compute_workgroup_storage_size as f64) / 32_768.0) * 6.0;

    // 5) Mapping support improves our zero-copy pipeline options. This is a small bonus
    //    because it affects data path quality, not raw shader throughput.
    let feature_term = if supports_mappable_primary_buffers {
        12.0
    } else {
        0.0
    };

    // 6) Memory topology multiplier captures latency and transfer penalties.
    //    Discrete GPUs usually win raw throughput. Integrated GPUs get a smaller penalty to
    //    avoid under-scheduling them when unified memory reduces copy overhead.
    let memory_topology_factor = match memory_topology {
        MemoryTopology::DiscreteVram => 1.00,
        MemoryTopology::Unified => 0.82,
        MemoryTopology::Virtualized => 0.55,
        MemoryTopology::System => 0.20,
        MemoryTopology::Unknown => 0.70,
    };

    let final_score = ((cu_term + bandwidth_term + vram_term + limits_term + feature_term)
        * memory_topology_factor)
        .round()
        .clamp(0.0, u64::MAX as f64) as u64;

    GpuScoreBreakdown {
        cu_term,
        bandwidth_term,
        vram_term,
        limits_term,
        feature_term,
        memory_topology_factor,
        final_score,
    }
}

impl PartialEq for GpuAdapterProfile {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}

impl Eq for GpuAdapterProfile {}

impl PartialOrd for GpuAdapterProfile {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for GpuAdapterProfile {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score
            .cmp(&other.score)
            .then_with(|| self.index.cmp(&other.index))
    }
}
