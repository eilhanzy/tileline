//! Pre-alpha canonical runtime phase ordering.
//!
//! This module freezes the intended frame order for the pre-alpha runtime path:
//!
//! 1. Network
//! 2. Script
//! 3. Physics
//! 4. RenderPlan
//! 5. Present
//!
//! The tracker is render-thread owned and lock-free.

/// Canonical pre-alpha runtime frame phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeFramePhase {
    Network,
    Script,
    Physics,
    RenderPlan,
    Present,
}

/// Fixed pre-alpha phase order.
pub const PRE_ALPHA_PHASE_ORDER: [RuntimeFramePhase; 5] = [
    RuntimeFramePhase::Network,
    RuntimeFramePhase::Script,
    RuntimeFramePhase::Physics,
    RuntimeFramePhase::RenderPlan,
    RuntimeFramePhase::Present,
];

/// Phase-order violation details for diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimePhaseViolation {
    pub frame_seq: u64,
    pub expected: RuntimeFramePhase,
    pub got: RuntimeFramePhase,
    pub expected_index: usize,
}

/// Snapshot metrics for phase-order tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RuntimePhaseOrderMetrics {
    pub frames_started: u64,
    pub frames_completed: u64,
    pub violations: u64,
    pub in_frame: bool,
    pub next_phase_index: usize,
    pub last_violation: Option<RuntimePhaseViolation>,
}

/// Render-thread-owned pre-alpha phase tracker.
#[derive(Debug, Clone, Copy, Default)]
pub struct RuntimePhaseOrderTracker {
    frame_seq: u64,
    in_frame: bool,
    next_phase_index: usize,
    frames_started: u64,
    frames_completed: u64,
    violations: u64,
    last_violation: Option<RuntimePhaseViolation>,
}

impl RuntimePhaseOrderTracker {
    /// Start a new frame sequence. If the previous frame was not completed, record a violation.
    pub fn begin_frame(&mut self) -> u64 {
        if self.in_frame {
            self.violations = self.violations.saturating_add(1);
        }
        self.frame_seq = self.frame_seq.saturating_add(1);
        self.in_frame = true;
        self.next_phase_index = 0;
        self.frames_started = self.frames_started.saturating_add(1);
        self.frame_seq
    }

    /// Enter a phase and validate canonical ordering.
    pub fn enter_phase(&mut self, phase: RuntimeFramePhase) -> Result<(), RuntimePhaseViolation> {
        let expected = PRE_ALPHA_PHASE_ORDER
            .get(self.next_phase_index)
            .copied()
            .unwrap_or(RuntimeFramePhase::Present);
        if !self.in_frame || expected != phase {
            let violation = RuntimePhaseViolation {
                frame_seq: self.frame_seq,
                expected,
                got: phase,
                expected_index: self.next_phase_index,
            };
            self.violations = self.violations.saturating_add(1);
            self.last_violation = Some(violation);
            return Err(violation);
        }
        self.next_phase_index = self.next_phase_index.saturating_add(1);
        Ok(())
    }

    /// Finish the current frame and verify all phases were consumed.
    pub fn finish_frame(&mut self) -> bool {
        let complete = self.in_frame && self.next_phase_index == PRE_ALPHA_PHASE_ORDER.len();
        if complete {
            self.frames_completed = self.frames_completed.saturating_add(1);
        } else {
            self.violations = self.violations.saturating_add(1);
        }
        self.in_frame = false;
        self.next_phase_index = 0;
        complete
    }

    /// Read-only snapshot of tracker counters.
    pub fn metrics(&self) -> RuntimePhaseOrderMetrics {
        RuntimePhaseOrderMetrics {
            frames_started: self.frames_started,
            frames_completed: self.frames_completed,
            violations: self.violations,
            in_frame: self.in_frame,
            next_phase_index: self.next_phase_index,
            last_violation: self.last_violation,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_canonical_pre_alpha_order() {
        let mut tracker = RuntimePhaseOrderTracker::default();
        tracker.begin_frame();
        for phase in PRE_ALPHA_PHASE_ORDER {
            assert!(tracker.enter_phase(phase).is_ok());
        }
        assert!(tracker.finish_frame());
        let metrics = tracker.metrics();
        assert_eq!(metrics.frames_started, 1);
        assert_eq!(metrics.frames_completed, 1);
        assert_eq!(metrics.violations, 0);
    }

    #[test]
    fn reports_violation_for_wrong_order() {
        let mut tracker = RuntimePhaseOrderTracker::default();
        tracker.begin_frame();
        let violation = tracker
            .enter_phase(RuntimeFramePhase::Physics)
            .expect_err("wrong phase order must report violation");
        assert_eq!(violation.expected, RuntimeFramePhase::Network);
        assert_eq!(violation.got, RuntimeFramePhase::Physics);
        assert!(!tracker.finish_frame());
        assert!(tracker.metrics().violations >= 1);
    }
}
