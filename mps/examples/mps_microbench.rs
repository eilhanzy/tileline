use mps::{CorePreference, MpsScheduler, NativeTask, TaskPriority};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy)]
struct MultiCoreScore {
    score: u64,
    tier: &'static str,
    tasks_per_sec: f64,
    work_units_per_sec: f64,
    parallel_efficiency_pct: f64,
    e_core_share_pct: f64,
}

fn main() {
    let scheduler = MpsScheduler::new();
    let topology = scheduler.topology();

    let logical = topology.logical_cores.max(1);
    let total_tasks = logical * 240;

    let critical_count = total_tasks / 10;
    let high_count = total_tasks * 2 / 10;
    let normal_count = total_tasks * 5 / 10;
    let background_count = total_tasks - critical_count - high_count - normal_count;

    println!(
        "Topology => logical: {}, physical: {}, hybrid: {}, P-cores: {}, E-cores: {}",
        topology.logical_cores,
        topology.physical_cores,
        topology.has_hybrid,
        topology.performance_cores,
        topology.efficient_cores
    );
    println!(
        "Task distribution => critical: {critical_count}, high: {high_count}, normal: {normal_count}, background: {background_count}"
    );

    let started = Instant::now();

    let (critical_batch, critical_work_units) = make_work_batch(critical_count, 30_000);
    let (high_batch, high_work_units) = make_work_batch(high_count, 24_000);
    let (normal_batch, normal_work_units) = make_work_batch(normal_count, 18_000);
    let (background_batch, background_work_units) = make_work_batch(background_count, 12_000);

    let _ = scheduler.submit_batch_native(
        TaskPriority::Critical,
        CorePreference::Performance,
        critical_batch,
    );
    let _ =
        scheduler.submit_batch_native(TaskPriority::High, CorePreference::Performance, high_batch);
    let _ = scheduler.submit_batch_native(TaskPriority::Normal, CorePreference::Auto, normal_batch);
    let _ = scheduler.submit_batch_native(
        TaskPriority::Background,
        CorePreference::Efficient,
        background_batch,
    );

    let total_work_units = critical_work_units
        .saturating_add(high_work_units)
        .saturating_add(normal_work_units)
        .saturating_add(background_work_units);

    let idle = scheduler.wait_for_idle(Duration::from_secs(30));
    let elapsed = started.elapsed();
    let metrics = scheduler.metrics();
    let score = compute_multicore_score(&metrics, elapsed, logical, total_work_units);

    println!("Idle reached: {idle}");
    println!("Elapsed: {:.3?}", elapsed);
    println!(
        "Metrics => submitted: {}, completed: {}, failed: {}",
        metrics.submitted, metrics.completed, metrics.failed
    );
    println!("Multicore score => {} [{}]", score.score, score.tier);
    println!(
        "Throughput => tasks/s: {:.0}, work units/s: {:.2}M",
        score.tasks_per_sec,
        score.work_units_per_sec / 1_000_000.0
    );
    println!(
        "Efficiency => parallel: {:.2}%, E-core share: {:.2}%",
        score.parallel_efficiency_pct, score.e_core_share_pct
    );
    println!(
        "Class runtime(ms) => P: total={:.3}, avg={:.6}; E: total={:.3}, avg={:.6}; U: total={:.3}, avg={:.6}",
        metrics.performance.execution_ms(),
        metrics.performance.avg_task_ms(),
        metrics.efficient.execution_ms(),
        metrics.efficient.avg_task_ms(),
        metrics.unknown.execution_ms(),
        metrics.unknown.avg_task_ms()
    );
    println!(
        "Queue depth => performance: {}, efficient: {}, shared: {}, total: {}",
        metrics.queue_depth.performance,
        metrics.queue_depth.efficient,
        metrics.queue_depth.shared,
        metrics.queue_depth.total
    );
}

fn make_work_batch(count: usize, base_iterations: u64) -> (Vec<NativeTask>, u64) {
    let total_iterations = (count as u64)
        .saturating_mul(base_iterations)
        .saturating_add(sum_mod_sequence(count as u64, 97));

    let tasks = (0..count)
        .map(|task_index| {
            Box::new(move || {
                let mut state = (task_index as u64)
                    .wrapping_mul(1_103_515_245)
                    .wrapping_add(12_345);
                let iterations = base_iterations + (task_index as u64 % 97);

                for step in 0..iterations {
                    state = state.rotate_left(5) ^ step.wrapping_mul(0x9E37_79B9_7F4A_7C15);
                    state = state
                        .wrapping_mul(0xBF58_476D_1CE4_E5B9)
                        .wrapping_add(0x94D0_49BB_1331_11EB);
                }

                std::hint::black_box(state);
            }) as NativeTask
        })
        .collect();

    (tasks, total_iterations)
}

fn sum_mod_sequence(count: u64, modulo: u64) -> u64 {
    if modulo == 0 {
        return 0;
    }

    let full_cycles = count / modulo;
    let remainder = count % modulo;
    let cycle_sum = (modulo.saturating_sub(1)).saturating_mul(modulo) / 2;
    let remainder_sum = remainder.saturating_sub(1).saturating_mul(remainder) / 2;

    full_cycles
        .saturating_mul(cycle_sum)
        .saturating_add(remainder_sum)
}

fn compute_multicore_score(
    metrics: &mps::SchedulerMetrics,
    elapsed: Duration,
    logical_cores: usize,
    total_work_units: u64,
) -> MultiCoreScore {
    let elapsed_secs = elapsed.as_secs_f64().max(1e-9);
    let tasks_per_sec = metrics.completed as f64 / elapsed_secs;
    let work_units_per_sec = total_work_units as f64 / elapsed_secs;

    let parallel_efficiency_pct = metrics.parallel_efficiency_pct(elapsed, logical_cores);
    let e_core_share_pct = metrics.e_core_share_pct();

    // Scoring model:
    // - base term: workload throughput
    // - boost term: multicore parallel efficiency
    let base_throughput_score = work_units_per_sec / 10_000.0;
    let efficiency_boost = 0.70 + (parallel_efficiency_pct / 100.0) * 0.30;
    let score = (base_throughput_score * efficiency_boost).round().max(0.0) as u64;

    MultiCoreScore {
        score,
        tier: score_tier(score),
        tasks_per_sec,
        work_units_per_sec,
        parallel_efficiency_pct,
        e_core_share_pct,
    }
}

fn score_tier(score: u64) -> &'static str {
    match score {
        220_000.. => "S",
        160_000.. => "A",
        110_000.. => "B",
        70_000.. => "C",
        _ => "D",
    }
}
