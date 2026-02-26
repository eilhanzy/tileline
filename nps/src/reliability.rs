//! Lightweight UDP reliability layer for NPS.
//!
//! This module provides per-peer sequence tracking, ACK windows, retransmit bookkeeping, and
//! channel policy selection so critical lifecycle events can be reliable while physics remains
//! fast and unreliable.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use crate::packet::{PacketChannel, PacketFlags, PayloadKind};

/// NPS peer identifier.
pub type PeerId = u16;
/// Networked physics object handle.
pub type NetObjectHandle = u32;

/// Transport reliability mode for a packet stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReliabilityMode {
    /// No retransmit; newer packets may supersede older ones.
    UnreliableSequenced,
    /// Lightweight reliability + ordering over UDP.
    ReliableOrdered,
}

/// Packet send policy derived from payload/channel semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SendPolicy {
    /// Reliability mode.
    pub reliability: ReliabilityMode,
    /// Whether the packet should set the `SEQUENCED` flag.
    pub sequenced: bool,
}

impl SendPolicy {
    /// Derive a policy from channel and payload kind.
    pub fn for_payload(channel: PacketChannel, kind: PayloadKind) -> Self {
        match (channel, kind) {
            (PacketChannel::Physics, _) | (PacketChannel::Input, _) => Self {
                reliability: ReliabilityMode::UnreliableSequenced,
                sequenced: true,
            },
            (PacketChannel::Lifecycle, _) | (_, PayloadKind::LifecycleEvent) => Self {
                reliability: ReliabilityMode::ReliableOrdered,
                sequenced: false,
            },
            _ => Self {
                reliability: ReliabilityMode::ReliableOrdered,
                sequenced: false,
            },
        }
    }

    /// Convert to wire packet flags.
    pub fn wire_flags(self) -> PacketFlags {
        let mut flags = PacketFlags::NONE;
        if matches!(self.reliability, ReliabilityMode::ReliableOrdered) {
            flags = flags.union(PacketFlags::RELIABLE);
        }
        if self.sequenced {
            flags = flags.union(PacketFlags::SEQUENCED);
        }
        flags
    }
}

/// Sliding receive ACK window (`ack` + previous 32 packets).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AckWindow {
    most_recent: Option<u16>,
    mask: u32,
}

impl AckWindow {
    /// Empty ACK window.
    pub const fn new() -> Self {
        Self {
            most_recent: None,
            mask: 0,
        }
    }

    /// Register a received sequence and update the ACK window.
    pub fn observe(&mut self, seq: u16) {
        match self.most_recent {
            None => {
                self.most_recent = Some(seq);
                self.mask = 0;
            }
            Some(current) => {
                if seq == current {
                    return;
                }
                if seq_more_recent(seq, current) {
                    let delta = seq.wrapping_sub(current) as u32;
                    if delta >= 32 {
                        self.mask = 0;
                    } else {
                        self.mask <<= delta;
                        // The previous `most_recent` sequence becomes `ack - delta` after the shift.
                        self.mask |= 1u32 << (delta - 1);
                    }
                    self.most_recent = Some(seq);
                } else {
                    let delta = current.wrapping_sub(seq) as u32;
                    if (1..=32).contains(&delta) {
                        self.mask |= 1u32 << (delta - 1);
                    }
                }
            }
        }
    }

    /// Return `(ack, ack_bits)` for header emission.
    pub fn header_fields(&self) -> (u16, u32) {
        match self.most_recent {
            Some(ack) => (ack, self.mask),
            None => (0, 0),
        }
    }

    /// Check if a sequence is acknowledged by the provided remote ack fields.
    pub fn is_acked_by(seq: u16, remote_ack: u16, remote_ack_bits: u32) -> bool {
        if seq == remote_ack {
            return true;
        }
        if seq_more_recent(seq, remote_ack) {
            return false;
        }
        let delta = remote_ack.wrapping_sub(seq) as u32;
        if !(1..=32).contains(&delta) {
            return false;
        }
        (remote_ack_bits & (1u32 << (delta - 1))) != 0
    }
}

impl Default for AckWindow {
    fn default() -> Self {
        Self::new()
    }
}

/// Reliable packet in-flight metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReliableInFlight {
    /// Wire sequence id used for ACK tracking.
    pub sequence: u16,
    /// Packet channel.
    pub channel: PacketChannel,
    /// Payload kind.
    pub kind: PayloadKind,
    /// Encoded datagram bytes.
    pub datagram: std::sync::Arc<[u8]>,
    /// Number of transmit attempts including the first send.
    pub transmit_count: u8,
    /// When the packet was first sent.
    pub first_sent_at: Instant,
    /// Last time the packet was sent.
    pub last_sent_at: Instant,
}

/// Per-peer reliability and sequence state.
#[derive(Debug)]
pub struct PeerReliabilityState {
    /// Peer id this state belongs to.
    pub peer: PeerId,
    next_sequence: u16,
    recv_ack_window: AckWindow,
    inflight_reliable: VecDeque<ReliableInFlight>,
    retransmit_timeout: Duration,
}

impl PeerReliabilityState {
    /// Create a new peer reliability state.
    pub fn new(peer: PeerId) -> Self {
        Self {
            peer,
            next_sequence: 1,
            recv_ack_window: AckWindow::new(),
            inflight_reliable: VecDeque::new(),
            retransmit_timeout: Duration::from_millis(40),
        }
    }

    /// Allocate the next outbound packet sequence number.
    pub fn next_sequence(&mut self) -> u16 {
        let seq = self.next_sequence;
        self.next_sequence = self.next_sequence.wrapping_add(1).max(1);
        seq
    }

    /// Current ack fields to place in outgoing packets.
    pub fn ack_header_fields(&self) -> (u16, u32) {
        self.recv_ack_window.header_fields()
    }

    /// Register a received packet sequence and process remote acks for local in-flight packets.
    pub fn observe_inbound(&mut self, received_seq: u16, remote_ack: u16, remote_ack_bits: u32) {
        self.recv_ack_window.observe(received_seq);
        self.ack_reliable(remote_ack, remote_ack_bits);
    }

    /// Track a newly-sent reliable datagram for retransmit/ack handling.
    pub fn track_reliable(
        &mut self,
        sequence: u16,
        channel: PacketChannel,
        kind: PayloadKind,
        datagram: std::sync::Arc<[u8]>,
        now: Instant,
    ) {
        self.inflight_reliable.push_back(ReliableInFlight {
            sequence,
            channel,
            kind,
            datagram,
            transmit_count: 1,
            first_sent_at: now,
            last_sent_at: now,
        });
    }

    /// Mark acknowledged reliable packets as complete.
    pub fn ack_reliable(&mut self, remote_ack: u16, remote_ack_bits: u32) {
        self.inflight_reliable
            .retain(|p| !AckWindow::is_acked_by(p.sequence, remote_ack, remote_ack_bits));
    }

    /// Collect reliable packets due for retransmit without blocking.
    pub fn collect_retransmit_due(&mut self, now: Instant) -> Vec<ReliableInFlight> {
        let mut due = Vec::new();
        for packet in &mut self.inflight_reliable {
            if now.duration_since(packet.last_sent_at) >= self.retransmit_timeout {
                packet.last_sent_at = now;
                packet.transmit_count = packet.transmit_count.saturating_add(1);
                due.push(packet.clone());
            }
        }
        due
    }

    /// Number of reliable packets waiting for ACK.
    pub fn inflight_reliable_len(&self) -> usize {
        self.inflight_reliable.len()
    }
}

/// Authority handoff reason codes for deterministic object ownership transfers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum AuthorityTransferReason {
    /// A player grabbed the object with a physgun-like tool.
    PhysgunGrab = 1,
    /// A player released the object and ownership returned to the prior master.
    PhysgunRelease = 2,
    /// Server/system forced a reassignment.
    ServerReassign = 3,
}

/// Result of an authority ownership transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AuthorityTransition {
    /// Object handle.
    pub object: NetObjectHandle,
    /// Previous authoritative owner.
    pub previous_owner: PeerId,
    /// New authoritative owner.
    pub new_owner: PeerId,
    /// Transition reason.
    pub reason: AuthorityTransferReason,
}

/// Deterministic master-slave ownership table for physics objects.
#[derive(Debug, Default)]
pub struct PhysAuthorityTable {
    objects: std::collections::HashMap<NetObjectHandle, AuthorityRecord>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AuthorityRecord {
    master_owner: PeerId,
    current_owner: PeerId,
    physgun_holder: Option<PeerId>,
}

impl PhysAuthorityTable {
    /// Insert or update the baseline master owner for an object.
    pub fn set_master_owner(&mut self, object: NetObjectHandle, owner: PeerId) {
        let record = self.objects.entry(object).or_insert(AuthorityRecord {
            master_owner: owner,
            current_owner: owner,
            physgun_holder: None,
        });
        record.master_owner = owner;
        if record.physgun_holder.is_none() {
            record.current_owner = owner;
        }
    }

    /// Get the current authoritative owner.
    pub fn current_owner(&self, object: NetObjectHandle) -> Option<PeerId> {
        self.objects.get(&object).map(|r| r.current_owner)
    }

    /// Transfer authority to the grabbing peer and remember the master owner.
    pub fn begin_physgun_grab(
        &mut self,
        object: NetObjectHandle,
        grabbing_peer: PeerId,
    ) -> Option<AuthorityTransition> {
        let record = self.objects.get_mut(&object)?;
        if record.current_owner == grabbing_peer && record.physgun_holder == Some(grabbing_peer) {
            return None;
        }
        let previous_owner = record.current_owner;
        record.current_owner = grabbing_peer;
        record.physgun_holder = Some(grabbing_peer);
        Some(AuthorityTransition {
            object,
            previous_owner,
            new_owner: grabbing_peer,
            reason: AuthorityTransferReason::PhysgunGrab,
        })
    }

    /// Release a physgun hold and return ownership to the master owner.
    pub fn end_physgun_grab(
        &mut self,
        object: NetObjectHandle,
        releasing_peer: PeerId,
    ) -> Option<AuthorityTransition> {
        let record = self.objects.get_mut(&object)?;
        if record.physgun_holder != Some(releasing_peer) {
            return None;
        }
        let previous_owner = record.current_owner;
        record.current_owner = record.master_owner;
        record.physgun_holder = None;
        Some(AuthorityTransition {
            object,
            previous_owner,
            new_owner: record.master_owner,
            reason: AuthorityTransferReason::PhysgunRelease,
        })
    }
}

/// Sequence recency comparison with wrap-around handling (standard half-range rule).
#[inline]
pub fn seq_more_recent(a: u16, b: u16) -> bool {
    (a > b && a - b <= 32768) || (b > a && b - a > 32768)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn ack_window_tracks_recent_packets() {
        let mut win = AckWindow::new();
        win.observe(10);
        win.observe(9);
        win.observe(8);
        win.observe(12);
        let (ack, bits) = win.header_fields();
        assert_eq!(ack, 12);
        assert!(AckWindow::is_acked_by(12, ack, bits));
        assert!(AckWindow::is_acked_by(10, ack, bits));
        assert!(AckWindow::is_acked_by(9, ack, bits));
        assert!(AckWindow::is_acked_by(8, ack, bits));
        assert!(!AckWindow::is_acked_by(7, ack, bits));
    }

    #[test]
    fn reliable_packets_are_cleared_by_ack() {
        let now = Instant::now();
        let mut peer = PeerReliabilityState::new(1);
        peer.track_reliable(
            11,
            PacketChannel::Lifecycle,
            PayloadKind::LifecycleEvent,
            Arc::from([1u8, 2, 3].as_slice()),
            now,
        );
        peer.track_reliable(
            12,
            PacketChannel::Lifecycle,
            PayloadKind::LifecycleEvent,
            Arc::from([4u8, 5, 6].as_slice()),
            now,
        );
        assert_eq!(peer.inflight_reliable_len(), 2);
        peer.ack_reliable(12, 1 << 0); // ack 12 and 11
        assert_eq!(peer.inflight_reliable_len(), 0);
    }

    #[test]
    fn physgun_handoff_transfers_and_restores_master_authority() {
        let mut table = PhysAuthorityTable::default();
        table.set_master_owner(100, 1);
        let grab = table.begin_physgun_grab(100, 7).unwrap();
        assert_eq!(grab.previous_owner, 1);
        assert_eq!(table.current_owner(100), Some(7));
        let release = table.end_physgun_grab(100, 7).unwrap();
        assert_eq!(release.new_owner, 1);
        assert_eq!(table.current_owner(100), Some(1));
    }
}
