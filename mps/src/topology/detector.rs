//! CPU core detector for heterogeneous topologies.
//! The detector prioritizes practical, cross-platform behavior:
//! - logical/physical core counts from `num_cpus`
//! - vendor information from `raw_cpuid`
//! - Linux frequency probing for big.LITTLE classification

use std::cmp::Ordering;
#[cfg(target_os = "linux")]
use std::fs;

/// Execution profile of a logical CPU core.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuClass {
    /// High-performance core (P-core).
    Performance,
    /// Energy-efficient core (E-core).
    Efficient,
    /// Unknown class; used as a safe fallback.
    Unknown,
}

/// Runtime information for a single logical core.
#[derive(Debug, Clone)]
pub struct CpuCore {
    /// Logical core identifier.
    pub id: usize,
    /// Classified role of the core.
    pub class: CpuClass,
    /// Maximum core frequency in KHz when available.
    pub max_frequency_khz: Option<u64>,
}

/// Snapshot of CPU topology used by the MPS scheduler.
#[derive(Debug, Clone)]
pub struct CpuTopology {
    /// Number of logical cores reported by the OS.
    pub logical_cores: usize,
    /// Number of physical cores reported by the OS.
    pub physical_cores: usize,
    /// CPU vendor string (if CPUID is available).
    pub vendor: Option<String>,
    /// Whether heterogeneous (big.LITTLE / P+E) behavior is detected.
    pub has_hybrid: bool,
    /// Count of classified high-performance cores.
    pub performance_cores: usize,
    /// Count of classified efficient cores.
    pub efficient_cores: usize,
    /// Per-core classification data.
    pub cores: Vec<CpuCore>,
}

impl CpuTopology {
    /// Detect the host CPU topology.
    pub fn detect() -> Self {
        let logical_cores = num_cpus::get().max(1);
        let physical_cores = num_cpus::get_physical().max(1).min(logical_cores);

        let vendor = detect_vendor();

        let frequency_map = detect_linux_frequencies(logical_cores);
        let classes = classify_cores(logical_cores, frequency_map.as_deref());

        let mut cores = Vec::with_capacity(logical_cores);
        for core_id in 0..logical_cores {
            let max_frequency_khz = frequency_map
                .as_ref()
                .and_then(|map| map.get(core_id))
                .copied()
                .flatten();

            cores.push(CpuCore {
                id: core_id,
                class: classes[core_id],
                max_frequency_khz,
            });
        }

        let performance_cores = cores
            .iter()
            .filter(|core| core.class == CpuClass::Performance)
            .count();
        let efficient_cores = cores
            .iter()
            .filter(|core| core.class == CpuClass::Efficient)
            .count();
        let has_hybrid = performance_cores > 0 && efficient_cores > 0;

        Self {
            logical_cores,
            physical_cores,
            vendor,
            has_hybrid,
            performance_cores,
            efficient_cores,
            cores,
        }
    }

    /// Return core IDs sorted for performance-first scheduling.
    pub fn preferred_core_ids(&self) -> Vec<usize> {
        let mut sorted = self.cores.clone();
        sorted.sort_by(|left, right| compare_core_priority(left, right));
        sorted.into_iter().map(|core| core.id).collect()
    }

    /// Return class information for a logical core ID.
    pub fn class_for_core(&self, core_id: usize) -> CpuClass {
        self.cores
            .iter()
            .find(|core| core.id == core_id)
            .map(|core| core.class)
            .unwrap_or(CpuClass::Unknown)
    }
}

fn compare_core_priority(left: &CpuCore, right: &CpuCore) -> Ordering {
    core_rank(left.class)
        .cmp(&core_rank(right.class))
        .then_with(|| right.max_frequency_khz.cmp(&left.max_frequency_khz))
        .then_with(|| left.id.cmp(&right.id))
}

fn core_rank(class: CpuClass) -> u8 {
    match class {
        CpuClass::Performance => 0,
        CpuClass::Unknown => 1,
        CpuClass::Efficient => 2,
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn detect_vendor() -> Option<String> {
    raw_cpuid::CpuId::new()
        .get_vendor_info()
        .map(|vendor_info| vendor_info.as_str().to_owned())
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn detect_vendor() -> Option<String> {
    None
}

fn classify_cores(logical_cores: usize, frequencies: Option<&[Option<u64>]>) -> Vec<CpuClass> {
    let mut classes = vec![CpuClass::Performance; logical_cores];

    let Some(frequencies) = frequencies else {
        return classes;
    };

    let mut known_frequencies: Vec<u64> = frequencies.iter().copied().flatten().collect();
    known_frequencies.sort_unstable();
    known_frequencies.dedup();

    let Some(min_frequency) = known_frequencies.first().copied() else {
        return classes;
    };
    let Some(max_frequency) = known_frequencies.last().copied() else {
        return classes;
    };

    // We treat the CPU as hybrid only when there is a meaningful
    // separation between the slowest and fastest core frequencies.
    let has_meaningful_gap = max_frequency > (min_frequency.saturating_mul(115) / 100);
    if !has_meaningful_gap {
        return classes;
    }

    let performance_threshold = max_frequency.saturating_mul(90) / 100;

    for (index, frequency) in frequencies.iter().enumerate().take(logical_cores) {
        classes[index] = match frequency {
            Some(freq) if *freq >= performance_threshold => CpuClass::Performance,
            Some(_) => CpuClass::Efficient,
            None => CpuClass::Unknown,
        };
    }

    classes
}

#[cfg(target_os = "linux")]
fn detect_linux_frequencies(logical_cores: usize) -> Option<Vec<Option<u64>>> {
    let mut frequencies = Vec::with_capacity(logical_cores);
    let mut discovered = false;

    for core_id in 0..logical_cores {
        let freq = read_linux_frequency(core_id);
        if freq.is_some() {
            discovered = true;
        }
        frequencies.push(freq);
    }

    discovered.then_some(frequencies)
}

#[cfg(not(target_os = "linux"))]
fn detect_linux_frequencies(_logical_cores: usize) -> Option<Vec<Option<u64>>> {
    None
}

#[cfg(target_os = "linux")]
fn read_linux_frequency(core_id: usize) -> Option<u64> {
    let candidates = ["cpuinfo_max_freq", "scaling_max_freq"];
    let base = format!("/sys/devices/system/cpu/cpu{core_id}/cpufreq");

    for file_name in candidates {
        let full_path = format!("{base}/{file_name}");
        if let Ok(raw) = fs::read_to_string(full_path) {
            let parsed = raw.trim().parse::<u64>().ok();
            if parsed.is_some() {
                return parsed;
            }
        }
    }

    None
}
