//! Priority and topology-aware load balancing.
//! This module maps task priority to preferred core classes.

use crate::topology::{CpuClass, CpuTopology};

/// Scheduler priority levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskPriority {
    /// Must be executed on the fastest cores whenever possible.
    Critical,
    /// High-value frame or gameplay work.
    High,
    /// Default gameplay/system work.
    Normal,
    /// Lowest priority background work.
    Background,
}

/// Explicit core preference override.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorePreference {
    /// Let the balancer choose by priority and topology.
    Auto,
    /// Prefer performance cores.
    Performance,
    /// Prefer efficient cores.
    Efficient,
}

/// Result of a routing decision for a task.
#[derive(Debug, Clone, Copy)]
pub struct RoutingDecision {
    /// Initial target lane for the task.
    pub preferred_class: CpuClass,
    /// Whether the task may spill to other lanes under pressure.
    pub spill_to_any: bool,
}

/// Load balancer tuned for big.LITTLE-aware dispatch.
#[derive(Debug, Clone)]
pub struct LoadBalancer {
    topology: CpuTopology,
    high_pressure_threshold: usize,
}

impl LoadBalancer {
    /// Build a new balancer from detected topology.
    pub fn new(topology: CpuTopology) -> Self {
        // Queue pressure threshold scales with core count.
        let high_pressure_threshold = (topology.logical_cores.max(1) * 4).max(8);
        Self {
            topology,
            high_pressure_threshold,
        }
    }

    /// Decide where a task should start and whether it can spill.
    pub fn decide(
        &self,
        priority: TaskPriority,
        preference: CorePreference,
        queued_tasks: usize,
    ) -> RoutingDecision {
        let preferred_class = self.normalize_class(match preference {
            CorePreference::Auto => self.auto_class(priority),
            CorePreference::Performance => CpuClass::Performance,
            CorePreference::Efficient => CpuClass::Efficient,
        });

        let spill_to_any = match priority {
            // Critical tasks stay on preferred cores unless the queue is saturated.
            TaskPriority::Critical => queued_tasks > self.high_pressure_threshold * 2,
            TaskPriority::High => true,
            TaskPriority::Normal => true,
            TaskPriority::Background => true,
        };

        RoutingDecision {
            preferred_class,
            spill_to_any,
        }
    }

    /// Access topology data for diagnostics or external policy.
    pub fn topology(&self) -> &CpuTopology {
        &self.topology
    }

    fn auto_class(&self, priority: TaskPriority) -> CpuClass {
        match priority {
            TaskPriority::Critical | TaskPriority::High => CpuClass::Performance,
            TaskPriority::Normal => CpuClass::Performance,
            TaskPriority::Background => {
                if self.topology.efficient_cores > 0 {
                    CpuClass::Efficient
                } else {
                    CpuClass::Performance
                }
            }
        }
    }

    fn normalize_class(&self, class: CpuClass) -> CpuClass {
        match class {
            CpuClass::Performance if self.topology.performance_cores == 0 => CpuClass::Efficient,
            CpuClass::Efficient if self.topology.efficient_cores == 0 => CpuClass::Performance,
            _ => class,
        }
    }
}
