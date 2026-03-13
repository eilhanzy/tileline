//! Canonical NPS channel/tick/snapshot model.
//!
//! This module is intentionally small and data-oriented. It provides a single source of truth for
//! packet-lane semantics before the transport runtime is wired in.
//!
//! The goal is to keep future runtime/UDP integration from scattering policy decisions across
//! `manager.rs`, `reliability.rs`, and gameplay glue.

use crate::packet::{PacketChannel, PayloadKind};

/// Logical traffic lane used for runtime budgeting and diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PacketLane {
    /// High-rate physics transforms and authoritative state deltas.
    PhysicsState,
    /// Player input stream bound to the simulation tick.
    PlayerInput,
    /// Reliable lifecycle and authority events.
    LifecycleEvent,
    /// Low-frequency UI/tooling control traffic.
    UiSync,
    /// Script/plugin coordination traffic.
    ScriptSync,
}

/// Tick domain carried by a packet stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TickScope {
    /// Tied to the fixed simulation tick.
    Simulation,
    /// Tied to lifecycle/control state rather than frame-accurate simulation.
    Control,
    /// Tied only to keepalive/ack cadence.
    Heartbeat,
}

/// Snapshot policy associated with a payload kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SnapshotMode {
    /// No snapshot semantics.
    None,
    /// Quantized state batch for fast physics replication.
    QuantizedTransformBatch,
    /// Deterministic input frame that advances authority simulation.
    InputFrame,
    /// Reliable key event rather than state snapshot.
    LifecycleEvent,
}

/// Canonical packet semantics for one wire packet shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PacketSemantics {
    /// Logical traffic lane.
    pub lane: PacketLane,
    /// Tick domain the payload belongs to.
    pub tick_scope: TickScope,
    /// Snapshot policy for payload interpretation.
    pub snapshot_mode: SnapshotMode,
    /// Whether the stream should be treated as reliable.
    pub reliable: bool,
    /// Whether newer packets supersede older ones.
    pub sequenced: bool,
    /// Relative priority hint for later bandwidth budgeting (`0` lowest, `255` highest).
    pub budget_weight: u8,
}

impl PacketSemantics {
    /// Whether this packet participates directly in simulation state progression.
    pub const fn is_simulation_critical(self) -> bool {
        matches!(
            self.lane,
            PacketLane::PhysicsState | PacketLane::PlayerInput | PacketLane::LifecycleEvent
        )
    }

    /// Whether the payload is expected to carry state snapshots rather than one-shot events.
    pub const fn is_snapshot_like(self) -> bool {
        matches!(
            self.snapshot_mode,
            SnapshotMode::QuantizedTransformBatch | SnapshotMode::InputFrame
        )
    }
}

/// Classify a packet into the canonical NPS lane/tick/snapshot model.
pub const fn packet_semantics(channel: PacketChannel, kind: PayloadKind) -> PacketSemantics {
    match (channel, kind) {
        (PacketChannel::Physics, PayloadKind::TransformBatch) => PacketSemantics {
            lane: PacketLane::PhysicsState,
            tick_scope: TickScope::Simulation,
            snapshot_mode: SnapshotMode::QuantizedTransformBatch,
            reliable: false,
            sequenced: true,
            budget_weight: 255,
        },
        (PacketChannel::Physics, PayloadKind::AuthorityTransfer) => PacketSemantics {
            lane: PacketLane::LifecycleEvent,
            tick_scope: TickScope::Control,
            snapshot_mode: SnapshotMode::LifecycleEvent,
            reliable: true,
            sequenced: false,
            budget_weight: 230,
        },
        (PacketChannel::Input, PayloadKind::InputFrame) => PacketSemantics {
            lane: PacketLane::PlayerInput,
            tick_scope: TickScope::Simulation,
            snapshot_mode: SnapshotMode::InputFrame,
            reliable: false,
            sequenced: true,
            budget_weight: 250,
        },
        (PacketChannel::Input, PayloadKind::AckOnly) => PacketSemantics {
            lane: PacketLane::PlayerInput,
            tick_scope: TickScope::Heartbeat,
            snapshot_mode: SnapshotMode::None,
            reliable: false,
            sequenced: false,
            budget_weight: 64,
        },
        (PacketChannel::Lifecycle, _) | (_, PayloadKind::LifecycleEvent) => PacketSemantics {
            lane: PacketLane::LifecycleEvent,
            tick_scope: TickScope::Control,
            snapshot_mode: SnapshotMode::LifecycleEvent,
            reliable: true,
            sequenced: false,
            budget_weight: 220,
        },
        (PacketChannel::Ui, _) => PacketSemantics {
            lane: PacketLane::UiSync,
            tick_scope: TickScope::Control,
            snapshot_mode: SnapshotMode::None,
            reliable: true,
            sequenced: false,
            budget_weight: 96,
        },
        (PacketChannel::Script, _) => PacketSemantics {
            lane: PacketLane::ScriptSync,
            tick_scope: TickScope::Control,
            snapshot_mode: SnapshotMode::None,
            reliable: true,
            sequenced: false,
            budget_weight: 128,
        },
        (PacketChannel::Physics, _) => PacketSemantics {
            lane: PacketLane::PhysicsState,
            tick_scope: TickScope::Simulation,
            snapshot_mode: SnapshotMode::None,
            reliable: false,
            sequenced: true,
            budget_weight: 200,
        },
        (PacketChannel::Input, _) => PacketSemantics {
            lane: PacketLane::PlayerInput,
            tick_scope: TickScope::Simulation,
            snapshot_mode: SnapshotMode::None,
            reliable: false,
            sequenced: true,
            budget_weight: 180,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn physics_transform_batch_is_fast_unreliable_simulation_traffic() {
        let semantics = packet_semantics(PacketChannel::Physics, PayloadKind::TransformBatch);
        assert_eq!(semantics.lane, PacketLane::PhysicsState);
        assert_eq!(semantics.tick_scope, TickScope::Simulation);
        assert_eq!(
            semantics.snapshot_mode,
            SnapshotMode::QuantizedTransformBatch
        );
        assert!(!semantics.reliable);
        assert!(semantics.sequenced);
        assert!(semantics.is_simulation_critical());
        assert!(semantics.is_snapshot_like());
    }

    #[test]
    fn lifecycle_payloads_are_reliable_control_traffic() {
        let semantics = packet_semantics(PacketChannel::Lifecycle, PayloadKind::LifecycleEvent);
        assert_eq!(semantics.lane, PacketLane::LifecycleEvent);
        assert_eq!(semantics.tick_scope, TickScope::Control);
        assert_eq!(semantics.snapshot_mode, SnapshotMode::LifecycleEvent);
        assert!(semantics.reliable);
        assert!(!semantics.sequenced);
    }

    #[test]
    fn input_frames_are_sequenced_simulation_packets() {
        let semantics = packet_semantics(PacketChannel::Input, PayloadKind::InputFrame);
        assert_eq!(semantics.lane, PacketLane::PlayerInput);
        assert_eq!(semantics.tick_scope, TickScope::Simulation);
        assert_eq!(semantics.snapshot_mode, SnapshotMode::InputFrame);
        assert!(!semantics.reliable);
        assert!(semantics.sequenced);
    }
}
