//! GPU hardware discovery and heuristic performance scoring for GMS.
//!
//! The scoring model is intentionally heuristic because `wgpu` does not expose
//! raw CU/SM counts or VRAM size on every backend. We therefore combine:
//! - adapter name parsing (best-effort CU/SM and VRAM hints)
//! - `wgpu::AdapterInfo` device type (discrete/integrated/virtual)
//! - `wgpu` limits/features as capability hints
//! - memory-topology penalties/bonuses

use std::cmp::Ordering;

use wgpu::{Adapter, AdapterInfo, Backends, DeviceType, Features, Instance, Limits};

const NVIDIA_VENDOR_ID: u32 = 0x10DE;
const AMD_VENDOR_ID: u32 = 0x1002;
const INTEL_VENDOR_ID: u32 = 0x8086;
const APPLE_VENDOR_ID: u32 = 0x106B;

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
    let memory_topology = classify_memory_topology(device_type);

    let estimated_compute_units =
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
    Other,
}

fn vendor_family(vendor_id: u32, name_lower: &str) -> VendorFamily {
    match vendor_id {
        NVIDIA_VENDOR_ID => VendorFamily::Nvidia,
        AMD_VENDOR_ID => VendorFamily::Amd,
        INTEL_VENDOR_ID => VendorFamily::Intel,
        APPLE_VENDOR_ID => VendorFamily::Apple,
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
            } else {
                VendorFamily::Other
            }
        }
    }
}

fn classify_memory_topology(device_type: DeviceType) -> MemoryTopology {
    match device_type {
        DeviceType::DiscreteGpu => MemoryTopology::DiscreteVram,
        DeviceType::IntegratedGpu => MemoryTopology::Unified,
        DeviceType::VirtualGpu => MemoryTopology::Virtualized,
        DeviceType::Cpu => MemoryTopology::System,
        _ => MemoryTopology::Unknown,
    }
}

fn estimate_compute_unit_kind(vendor: VendorFamily) -> ComputeUnitKind {
    match vendor {
        VendorFamily::Amd => ComputeUnitKind::Cu,
        VendorFamily::Nvidia => ComputeUnitKind::Sm,
        VendorFamily::Intel | VendorFamily::Apple => ComputeUnitKind::CoreCluster,
        VendorFamily::Other => ComputeUnitKind::Unknown,
    }
}

fn estimate_compute_units(
    info: &AdapterInfo,
    limits: &Limits,
    name_lower: &str,
    vendor: VendorFamily,
) -> u32 {
    if let Some(parsed) = parse_known_compute_units(name_lower, vendor) {
        return parsed.max(1);
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

    estimate.round().clamp(1.0, 256.0) as u32
}

fn parse_known_compute_units(name_lower: &str, vendor: VendorFamily) -> Option<u32> {
    match vendor {
        VendorFamily::Nvidia => parse_nvidia_sm(name_lower),
        VendorFamily::Amd => parse_amd_cu(name_lower),
        VendorFamily::Apple => parse_apple_gpu_clusters(name_lower),
        VendorFamily::Intel => parse_intel_gpu_clusters(name_lower),
        VendorFamily::Other => None,
    }
}

fn parse_nvidia_sm(name: &str) -> Option<u32> {
    // Common, approximate SM counts (sufficient for scheduling ratios).
    const TABLE: &[(&str, u32)] = &[
        ("rtx 4090", 128),
        ("rtx 4080", 76),
        ("rtx 4070 ti", 60),
        ("rtx 4070", 46),
        ("rtx 4060 ti", 34),
        ("rtx 4060", 24),
        ("rtx 3090", 82),
        ("rtx 3080", 68),
        ("rtx 3070", 46),
        ("rtx 3060", 28),
        ("rtx 2080", 46),
        ("rtx 2070", 36),
        ("rtx 2060", 30),
        ("gtx 1660", 22),
        ("gtx 1650", 14),
    ];
    lookup_table_contains(name, TABLE)
}

fn parse_amd_cu(name: &str) -> Option<u32> {
    // Common RDNA/RDNA2/RDNA3 CU counts (approximate but directionally correct for weighting).
    const TABLE: &[(&str, u32)] = &[
        ("rx 7900 xtx", 96),
        ("rx 7900 xt", 84),
        ("rx 7900 gre", 80),
        ("rx 7800 xt", 60),
        ("rx 7700 xt", 54),
        ("rx 7600", 32),
        ("rx 6900 xt", 80),
        ("rx 6800 xt", 72),
        ("rx 6800", 60),
        ("rx 6700 xt", 40),
        ("rx 6600 xt", 32),
        ("rx 6600", 28),
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
