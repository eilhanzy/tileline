use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use gms::GmsRuntimeTuningProfile;
use wgpu::{AdapterInfo, Backend, DeviceType};

// ── AdapterInfo yardımcısı ─────────────────────────────────────────────────────

fn make_adapter_info(name: &str, device_type: DeviceType) -> AdapterInfo {
    AdapterInfo {
        name: name.to_string(),
        vendor: 0,
        device: 0,
        device_type,
        device_pci_bus_id: String::new(),
        driver: String::new(),
        driver_info: String::new(),
        backend: Backend::Metal,
        subgroup_min_size: 4,
        subgroup_max_size: 32,
        transient_saves_memory: false,
    }
}

// ── GmsRuntimeTuningProfile::from_adapter_info ────────────────────────────────

fn bench_tuning_from_adapter_info(c: &mut Criterion) {
    let adapters = [
        ("Apple M3 Pro",    make_adapter_info("Apple M3 Pro",    DeviceType::IntegratedGpu)),
        ("Apple M2",        make_adapter_info("Apple M2",        DeviceType::IntegratedGpu)),
        ("NVIDIA RTX 4090", make_adapter_info("NVIDIA RTX 4090", DeviceType::DiscreteGpu)),
        ("AMD RX 7900 XTX", make_adapter_info("AMD RX 7900 XTX", DeviceType::DiscreteGpu)),
        ("Intel UHD 770",   make_adapter_info("Intel UHD 770",   DeviceType::IntegratedGpu)),
    ];

    let mut group = c.benchmark_group("tuning/from_adapter_info");
    for (label, info) in &adapters {
        group.bench_with_input(BenchmarkId::from_parameter(label), info, |b, info| {
            b.iter(|| GmsRuntimeTuningProfile::from_adapter_info(black_box(info)));
        });
    }
    group.finish();
}

// ── effective_throughput_work_units_per_present ───────────────────────────────

fn bench_effective_throughput(c: &mut Criterion) {
    let apple = GmsRuntimeTuningProfile::from_adapter_info(&make_adapter_info(
        "Apple M3",
        DeviceType::IntegratedGpu,
    ));
    let discrete = GmsRuntimeTuningProfile::from_adapter_info(&make_adapter_info(
        "NVIDIA RTX 4090",
        DeviceType::DiscreteGpu,
    ));

    let mut group = c.benchmark_group("tuning/effective_throughput");

    // Ramp aşaması (düşük frame sayısı)
    group.bench_function("apple_ramp_early", |b| {
        b.iter(|| {
            apple.effective_throughput_work_units_per_present(
                black_box(8),
                black_box(5),
            )
        });
    });

    // Tam throughput (yüksek frame sayısı)
    group.bench_function("apple_ramp_steady", |b| {
        b.iter(|| {
            apple.effective_throughput_work_units_per_present(
                black_box(8),
                black_box(9999),
            )
        });
    });

    // Discrete GPU — ramp yok
    group.bench_function("discrete_no_ramp", |b| {
        b.iter(|| {
            discrete.effective_throughput_work_units_per_present(
                black_box(8),
                black_box(50),
            )
        });
    });

    group.finish();
}

// ── startup_prewarm_submits_for_ring ──────────────────────────────────────────

fn bench_prewarm_submits(c: &mut Criterion) {
    let profiles = [
        ("apple",    GmsRuntimeTuningProfile::from_adapter_info(&make_adapter_info("Apple M2", DeviceType::IntegratedGpu))),
        ("discrete", GmsRuntimeTuningProfile::from_adapter_info(&make_adapter_info("NVIDIA RTX 3080", DeviceType::DiscreteGpu))),
        ("intel",    GmsRuntimeTuningProfile::from_adapter_info(&make_adapter_info("Intel UHD 770", DeviceType::IntegratedGpu))),
    ];

    let mut group = c.benchmark_group("tuning/prewarm_submits");
    for (label, profile) in &profiles {
        group.bench_with_input(BenchmarkId::from_parameter(label), profile, |b, p| {
            b.iter(|| p.startup_prewarm_submits_for_ring(black_box(4), black_box(6)));
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_tuning_from_adapter_info,
    bench_effective_throughput,
    bench_prewarm_submits,
);
criterion_main!(benches);
