use mps::{CorePreference, MpsScheduler, TaskPriority, WasmTask};
use std::sync::Arc;
use std::time::Duration;

// Precompiled module:
// (module (func (export "_start")))
const WASM_NOOP_START_MODULE: &[u8] = &[
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x04, 0x01, 0x60, 0x00, 0x00, 0x03, 0x02,
    0x01, 0x00, 0x07, 0x0a, 0x01, 0x06, 0x5f, 0x73, 0x74, 0x61, 0x72, 0x74, 0x00, 0x00, 0x0a, 0x04,
    0x01, 0x02, 0x00, 0x0b,
];

fn main() {
    let scheduler = MpsScheduler::new();

    let wasm_task = WasmTask::new(Arc::<[u8]>::from(WASM_NOOP_START_MODULE), "_start");
    let wasm_id = scheduler.submit_wasm(TaskPriority::Critical, CorePreference::Auto, wasm_task);

    let native_id = scheduler.submit_native(TaskPriority::High, CorePreference::Auto, || {
        let mut sum = 0_u64;
        for n in 0..10_000 {
            sum = sum.wrapping_add(n);
        }
        std::hint::black_box(sum);
    });

    let idle = scheduler.wait_for_idle(Duration::from_secs(5));
    let metrics = scheduler.metrics();

    println!("Submitted WASM task id: {wasm_id}");
    println!("Submitted native task id: {native_id}");
    println!("Idle reached: {idle}");
    println!(
        "Metrics => submitted: {}, completed: {}, failed: {}",
        metrics.submitted, metrics.completed, metrics.failed
    );
    println!(
        "Queue depth => performance: {}, efficient: {}, shared: {}, total: {}",
        metrics.queue_depth.performance,
        metrics.queue_depth.efficient,
        metrics.queue_depth.shared,
        metrics.queue_depth.total
    );
}
