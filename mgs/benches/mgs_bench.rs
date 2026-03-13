use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use mgs::{MgsBridge, MgsPlanner, MobileGpuProfile};
use mgs::bridge::MpsWorkloadHint;
use mgs::tile_planner::TileWorkloadRequest;
use mgs::tuning::{MgsTuningProfile, ThermalState};

// ── Hardware detect ────────────────────────────────────────────────────────────

fn bench_profile_detect(c: &mut Criterion) {
    let names = [
        "Adreno 750",
        "Mali-G78 MC24",
        "Apple M3",
        "PowerVR GX6650",
        "NVIDIA GeForce RTX 4090", // bilinmeyen → Unknown
    ];

    let mut group = c.benchmark_group("hardware/detect");
    for name in &names {
        group.bench_with_input(BenchmarkId::from_parameter(name), name, |b, &name| {
            b.iter(|| MobileGpuProfile::detect(black_box(name)));
        });
    }
    group.finish();
}

// ── Tuning profile ─────────────────────────────────────────────────────────────

fn bench_tuning_from_profile(c: &mut Criterion) {
    let profiles = [
        MobileGpuProfile::detect("Adreno 750"),
        MobileGpuProfile::detect("Mali-G78 MC24"),
        MobileGpuProfile::detect("Apple M3"),
    ];
    let labels = ["Adreno 750", "Mali-G78 MC24", "Apple M3"];

    let mut group = c.benchmark_group("tuning/from_profile");
    for (profile, label) in profiles.iter().zip(labels.iter()) {
        group.bench_with_input(BenchmarkId::from_parameter(label), profile, |b, p| {
            b.iter(|| MgsTuningProfile::from_profile(black_box(p)));
        });
    }
    group.finish();
}

fn bench_thermal_apply(c: &mut Criterion) {
    let profile = MobileGpuProfile::detect("Adreno 750");
    let states = [
        ThermalState::Nominal,
        ThermalState::Warm,
        ThermalState::Hot,
        ThermalState::Unknown,
    ];
    let labels = ["Nominal", "Warm", "Hot", "Unknown"];

    let mut group = c.benchmark_group("tuning/apply_thermal");
    for (state, label) in states.iter().zip(labels.iter()) {
        group.bench_with_input(BenchmarkId::from_parameter(label), state, |b, &state| {
            b.iter(|| {
                MgsTuningProfile::from_profile(black_box(&profile))
                    .apply_thermal_state(state)
            });
        });
    }
    group.finish();
}

// ── Tile planner ───────────────────────────────────────────────────────────────

fn bench_tile_plan(c: &mut Criterion) {
    struct Case {
        label: &'static str,
        gpu: &'static str,
        width: u32,
        height: u32,
        draws: u32,
    }

    let cases = [
        Case { label: "Adreno-1080p-200draws",  gpu: "Adreno 750",     width: 1920, height: 1080, draws: 200 },
        Case { label: "Mali-1440p-512draws",     gpu: "Mali-G78 MC24",  width: 2560, height: 1440, draws: 512 },
        Case { label: "Apple-4K-1024draws",      gpu: "Apple M3",       width: 3840, height: 2160, draws: 1024 },
        Case { label: "Unknown-720p-32draws",    gpu: "SomeUnknownGPU", width: 1280, height: 720,  draws: 32 },
    ];

    let mut group = c.benchmark_group("tile_planner/plan");
    for case in &cases {
        let planner = MgsPlanner::from_profile(MobileGpuProfile::detect(case.gpu));
        let req = TileWorkloadRequest {
            target_width: case.width,
            target_height: case.height,
            total_draw_calls: case.draws,
            bytes_per_draw_kb: 4,
        };
        group.bench_with_input(BenchmarkId::from_parameter(case.label), &req, |b, req| {
            b.iter(|| planner.plan(black_box(*req)));
        });
    }
    group.finish();
}

// ── Bridge translate ───────────────────────────────────────────────────────────

fn bench_bridge_translate(c: &mut Criterion) {
    let bridges = [
        ("Adreno 750",    MgsBridge::new(MobileGpuProfile::detect("Adreno 750"))),
        ("Mali-G78 MC24", MgsBridge::new(MobileGpuProfile::detect("Mali-G78 MC24"))),
        ("Apple M3",      MgsBridge::new(MobileGpuProfile::detect("Apple M3"))),
    ];

    let hint = MpsWorkloadHint {
        transfer_size_kb: 512,
        object_count: 128,
        target_width: 1920,
        target_height: 1080,
        latency_budget_ms: 0,
    };

    let mut group = c.benchmark_group("bridge/translate");
    for (label, bridge) in &bridges {
        group.bench_with_input(BenchmarkId::from_parameter(label), bridge, |b, br| {
            b.iter(|| br.translate(black_box(hint)));
        });
    }
    group.finish();
}

// ── Fallback stress ────────────────────────────────────────────────────────────

fn bench_bridge_memory_pressure(c: &mut Criterion) {
    // Yüksek bytes_per_draw_kb → memory pressure → fallback zincirini çalıştırır.
    let bridge = MgsBridge::new(MobileGpuProfile::detect("Adreno 610")); // düşük core sayısı
    let stress_hint = MpsWorkloadHint {
        transfer_size_kb: 65535,
        object_count: 1,
        target_width: 1920,
        target_height: 1080,
        latency_budget_ms: 0,
    };

    c.bench_function("bridge/fallback_chain_stress", |b| {
        b.iter(|| bridge.translate(black_box(stress_hint)));
    });
}

criterion_group!(
    benches,
    bench_profile_detect,
    bench_tuning_from_profile,
    bench_thermal_apply,
    bench_tile_plan,
    bench_bridge_translate,
    bench_bridge_memory_pressure,
);
criterion_main!(benches);
