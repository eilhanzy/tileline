use mps::{CorePreference, MpsScheduler, NativeTask, TaskPriority};
use std::time::{Duration, Instant};

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

    let _ = scheduler.submit_batch_native(
        TaskPriority::Critical,
        CorePreference::Performance,
        make_work_batch(critical_count, 30_000),
    );
    let _ = scheduler.submit_batch_native(
        TaskPriority::High,
        CorePreference::Performance,
        make_work_batch(high_count, 24_000),
    );
    let _ = scheduler.submit_batch_native(
        TaskPriority::Normal,
        CorePreference::Auto,
        make_work_batch(normal_count, 18_000),
    );
    let _ = scheduler.submit_batch_native(
        TaskPriority::Background,
        CorePreference::Efficient,
        make_work_batch(background_count, 12_000),
    );

    let idle = scheduler.wait_for_idle(Duration::from_secs(30));
    let elapsed = started.elapsed();
    let metrics = scheduler.metrics();

    println!("Idle reached: {idle}");
    println!("Elapsed: {:.3?}", elapsed);
    println!(
        "Metrics => submitted: {}, completed: {}, failed: {}",
        metrics.submitted, metrics.completed, metrics.failed
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

fn make_work_batch(count: usize, base_iterations: u64) -> Vec<NativeTask> {
    (0..count)
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
        .collect()
}
