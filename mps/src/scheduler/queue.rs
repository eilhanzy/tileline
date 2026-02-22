//! Lock-free scheduler queues.
//!
//! We use separate lanes for performance and efficient workers,
//! then allow controlled stealing for spillover.

use super::TaskEnvelope;
use crate::balancer::TaskPriority;
use crate::topology::CpuClass;
use crossbeam::queue::SegQueue;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Queue depth snapshot for diagnostics and runtime tuning.
#[derive(Debug, Clone, Copy, Default)]
pub struct QueueDepth {
    pub performance: usize,
    pub efficient: usize,
    pub shared: usize,
    pub total: usize,
}

/// Lock-free task queue with topology-aware lanes.
#[derive(Clone)]
pub struct PriorityTaskQueue {
    performance: Arc<LaneQueue>,
    efficient: Arc<LaneQueue>,
    shared: Arc<LaneQueue>,
}

impl PriorityTaskQueue {
    /// Create an empty queue.
    pub fn new() -> Self {
        Self {
            performance: Arc::new(LaneQueue::new()),
            efficient: Arc::new(LaneQueue::new()),
            shared: Arc::new(LaneQueue::new()),
        }
    }

    /// Push a task into the lane selected by the load balancer.
    pub fn push(&self, task: TaskEnvelope) {
        match task.preferred_class {
            CpuClass::Performance => self.performance.push(task),
            CpuClass::Efficient => self.efficient.push(task),
            CpuClass::Unknown => self.shared.push(task),
        }
    }

    /// Pop a task for a worker class, with controlled cross-lane stealing.
    pub fn pop_for_worker(&self, worker_class: CpuClass) -> Option<TaskEnvelope> {
        match worker_class {
            CpuClass::Performance => self
                .performance
                .pop_high_to_low()
                .or_else(|| self.shared.pop_high_to_low())
                .or_else(|| self.steal_with_spill_check(&self.efficient, worker_class)),
            CpuClass::Efficient => self
                .efficient
                .pop_background_to_critical()
                .or_else(|| self.shared.pop_high_to_low())
                .or_else(|| self.steal_with_spill_check(&self.performance, worker_class)),
            CpuClass::Unknown => self
                .shared
                .pop_high_to_low()
                .or_else(|| self.performance.pop_high_to_low())
                .or_else(|| self.efficient.pop_background_to_critical()),
        }
    }

    /// Return true when all lanes are empty.
    pub fn is_empty(&self) -> bool {
        self.total_len() == 0
    }

    /// Approximate total queue length.
    pub fn total_len(&self) -> usize {
        self.performance.depth() + self.efficient.depth() + self.shared.depth()
    }

    /// Snapshot lane depths.
    pub fn depth_snapshot(&self) -> QueueDepth {
        let performance = self.performance.depth();
        let efficient = self.efficient.depth();
        let shared = self.shared.depth();
        QueueDepth {
            performance,
            efficient,
            shared,
            total: performance + efficient + shared,
        }
    }

    fn steal_with_spill_check(
        &self,
        source: &LaneQueue,
        worker_class: CpuClass,
    ) -> Option<TaskEnvelope> {
        // Bound retry count to avoid endless churn when only non-spill tasks exist.
        for _ in 0..4 {
            let task = source.pop_high_to_low()?;
            if task.spill_to_any
                || task.preferred_class == CpuClass::Unknown
                || task.preferred_class == worker_class
            {
                return Some(task);
            }

            // Return non-spill tasks to their original lane.
            self.push(task);
        }

        None
    }
}

struct LaneQueue {
    critical: SegQueue<TaskEnvelope>,
    high: SegQueue<TaskEnvelope>,
    normal: SegQueue<TaskEnvelope>,
    background: SegQueue<TaskEnvelope>,
    depth: AtomicUsize,
}

impl LaneQueue {
    fn new() -> Self {
        Self {
            critical: SegQueue::new(),
            high: SegQueue::new(),
            normal: SegQueue::new(),
            background: SegQueue::new(),
            depth: AtomicUsize::new(0),
        }
    }

    fn push(&self, task: TaskEnvelope) {
        match task.priority {
            TaskPriority::Critical => self.critical.push(task),
            TaskPriority::High => self.high.push(task),
            TaskPriority::Normal => self.normal.push(task),
            TaskPriority::Background => self.background.push(task),
        }
        self.depth.fetch_add(1, Ordering::Relaxed);
    }

    fn pop_high_to_low(&self) -> Option<TaskEnvelope> {
        self.pop_in_order([
            TaskPriority::Critical,
            TaskPriority::High,
            TaskPriority::Normal,
            TaskPriority::Background,
        ])
    }

    fn pop_background_to_critical(&self) -> Option<TaskEnvelope> {
        self.pop_in_order([
            TaskPriority::Background,
            TaskPriority::Normal,
            TaskPriority::High,
            TaskPriority::Critical,
        ])
    }

    fn pop_in_order(&self, order: [TaskPriority; 4]) -> Option<TaskEnvelope> {
        for priority in order {
            if let Some(task) = self.pop_priority(priority) {
                return Some(task);
            }
        }
        None
    }

    fn pop_priority(&self, priority: TaskPriority) -> Option<TaskEnvelope> {
        let popped = match priority {
            TaskPriority::Critical => self.critical.pop(),
            TaskPriority::High => self.high.pop(),
            TaskPriority::Normal => self.normal.pop(),
            TaskPriority::Background => self.background.pop(),
        };

        if popped.is_some() {
            self.depth.fetch_sub(1, Ordering::Relaxed);
        }

        popped
    }

    fn depth(&self) -> usize {
        self.depth.load(Ordering::Relaxed)
    }
}
