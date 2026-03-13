//! Runtime-owned NPS transport coordination.
//!
//! This module keeps the actual UDP socket pump in `runtime/src` instead of spreading transport
//! behavior across tests or examples. The design stays non-blocking:
//!
//! - inbound datagrams are drained with `try_recv_from`
//! - outbound datagrams are flushed with `try_send_to`
//! - packet encode/decode stays inside `nps::NetworkPacketManager`
//! - ParadoxPE snapshots are emitted on an explicit simulation-tick cadence
//!
//! The caller owns the outer loop and decides which thread/task pumps the socket.

use std::collections::{HashMap, VecDeque};
use std::io;
use std::io::ErrorKind;
use std::net::SocketAddr;
use std::sync::Arc;

use mps::MpsScheduler;
use nps::{
    packet_semantics, DecodedPacketEvent, EncodedDatagram, NetworkPacketConfig,
    NetworkPacketManager, NetworkPacketMetrics, PacketDecodeFailure, PacketEncodeFailure,
    PacketLane, PeerId, PeerLinkMetrics, NPS_HEADER_BYTES,
};
use paradoxpe::PhysicsWorld;
use tokio::net::{ToSocketAddrs, UdpSocket};

/// Snapshot emission cadence for authoritative physics state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SnapshotCadenceConfig {
    /// Set to `false` to disable automatic ParadoxPE snapshot emission.
    pub enabled: bool,
    /// Emit every `N` simulation ticks (`1` = every tick).
    pub emit_every_ticks: u64,
}

impl Default for SnapshotCadenceConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            emit_every_ticks: 1,
        }
    }
}

/// Runtime transport pump limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NetworkTransportConfig {
    /// Max inbound datagrams to dequeue from the UDP socket per pump.
    pub max_recv_datagrams_per_pump: usize,
    /// Max decode jobs to submit to NPS/MPS per pump.
    pub max_decode_jobs_per_pump: usize,
    /// Max outbound encode jobs to submit per pump.
    pub max_encode_jobs_per_pump: usize,
    /// Max outbound datagrams to try sending per pump.
    pub max_send_datagrams_per_pump: usize,
    /// Max retransmit datagrams to collect per pump.
    pub max_retransmits_per_pump: usize,
    /// Reusable inbound receive buffer size.
    pub recv_buffer_bytes: usize,
    /// Authoritative physics snapshot cadence.
    pub snapshot_cadence: SnapshotCadenceConfig,
}

impl Default for NetworkTransportConfig {
    fn default() -> Self {
        Self {
            max_recv_datagrams_per_pump: 64,
            max_decode_jobs_per_pump: 64,
            max_encode_jobs_per_pump: 64,
            max_send_datagrams_per_pump: 64,
            max_retransmits_per_pump: 32,
            recv_buffer_bytes: 2048,
            snapshot_cadence: SnapshotCadenceConfig::default(),
        }
    }
}

/// Per-pump counters for the transport loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct NetworkPumpResult {
    /// Number of raw UDP datagrams accepted from the socket this pump.
    pub recv_datagrams: usize,
    /// Number of UDP bytes accepted from the socket this pump.
    pub recv_bytes: usize,
    /// Number of inbound datagrams dropped because no peer mapping existed for the source address.
    pub recv_unknown_addr_drops: usize,
    /// Number of decode jobs submitted to NPS/MPS.
    pub decode_jobs_submitted: usize,
    /// Number of encode jobs submitted to NPS/MPS.
    pub encode_jobs_submitted: usize,
    /// Number of encoded datagrams moved into the runtime-owned send queue.
    pub encoded_datagrams_buffered: usize,
    /// Number of retransmit datagrams moved into the runtime-owned send queue.
    pub retransmit_datagrams_buffered: usize,
    /// Number of UDP datagrams sent this pump.
    pub sent_datagrams: usize,
    /// Number of UDP bytes sent this pump.
    pub sent_bytes: usize,
    /// Number of authoritative snapshot packets queued this pump.
    pub queued_snapshot_packets: usize,
    /// Number of datagrams still waiting in the runtime send queue after the pump.
    pub pending_send_queue_len: usize,
}

/// Per-lane traffic counter snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LaneTrafficCounter {
    pub datagrams: u64,
    pub bytes: u64,
}

/// Aggregate traffic counters grouped by canonical NPS lanes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct NetworkLaneMetrics {
    pub physics_state: LaneTrafficCounter,
    pub player_input: LaneTrafficCounter,
    pub lifecycle_event: LaneTrafficCounter,
    pub ui_sync: LaneTrafficCounter,
    pub script_sync: LaneTrafficCounter,
}

impl NetworkLaneMetrics {
    fn record(&mut self, lane: PacketLane, bytes: usize) {
        let counter = match lane {
            PacketLane::PhysicsState => &mut self.physics_state,
            PacketLane::PlayerInput => &mut self.player_input,
            PacketLane::LifecycleEvent => &mut self.lifecycle_event,
            PacketLane::UiSync => &mut self.ui_sync,
            PacketLane::ScriptSync => &mut self.script_sync,
        };
        counter.datagrams = counter.datagrams.saturating_add(1);
        counter.bytes = counter.bytes.saturating_add(bytes as u64);
    }
}

/// Per-peer runtime snapshot of link telemetry.
#[derive(Debug, Clone, PartialEq)]
pub struct NetworkPeerMetrics {
    pub peer: PeerId,
    pub addr: Option<SocketAddr>,
    pub link: PeerLinkMetrics,
}

/// Aggregate runtime transport metrics including underlying NPS counters.
#[derive(Debug, Clone, PartialEq)]
pub struct NetworkTransportMetrics {
    pub pump_calls: u64,
    pub recv_datagrams: u64,
    pub recv_bytes: u64,
    pub recv_unknown_addr_drops: u64,
    pub sent_datagrams: u64,
    pub sent_bytes: u64,
    pub decode_failures: u64,
    pub encode_failures: u64,
    pub decoded_events_drained: u64,
    pub snapshot_emit_calls: u64,
    pub snapshot_queued_packets: u64,
    pub snapshot_skipped_cadence: u64,
    pub snapshot_skipped_duplicate_tick: u64,
    pub last_snapshot_tick: Option<u64>,
    pub known_peers: usize,
    pub pending_send_queue_len: usize,
    pub sent_by_lane: NetworkLaneMetrics,
    pub received_by_lane: NetworkLaneMetrics,
    pub peers: Vec<NetworkPeerMetrics>,
    pub manager: NetworkPacketMetrics,
}

/// Runtime-owned non-blocking UDP transport coordinator for NPS.
pub struct NetworkTransportRuntime {
    config: NetworkTransportConfig,
    manager: NetworkPacketManager,
    peer_addrs: HashMap<PeerId, SocketAddr>,
    peer_by_addr: HashMap<SocketAddr, PeerId>,
    recv_buffer: Vec<u8>,
    pending_send: VecDeque<EncodedDatagram>,
    pump_calls: u64,
    recv_datagrams: u64,
    recv_bytes: u64,
    recv_unknown_addr_drops: u64,
    sent_datagrams: u64,
    sent_bytes: u64,
    decode_failures: u64,
    encode_failures: u64,
    decoded_events_drained: u64,
    snapshot_emit_calls: u64,
    snapshot_queued_packets: u64,
    snapshot_skipped_cadence: u64,
    snapshot_skipped_duplicate_tick: u64,
    last_snapshot_tick: Option<u64>,
    sent_by_lane: NetworkLaneMetrics,
    received_by_lane: NetworkLaneMetrics,
}

impl NetworkTransportRuntime {
    /// Build a runtime transport coordinator without MPS offload.
    pub fn new(config: NetworkTransportConfig, packet_config: NetworkPacketConfig) -> Self {
        Self::with_mps(config, packet_config, None)
    }

    /// Build a runtime transport coordinator with an optional MPS scheduler.
    pub fn with_mps(
        config: NetworkTransportConfig,
        packet_config: NetworkPacketConfig,
        mps: Option<Arc<MpsScheduler>>,
    ) -> Self {
        Self {
            recv_buffer: vec![0u8; config.recv_buffer_bytes.max(512)],
            config,
            manager: NetworkPacketManager::with_mps(packet_config, mps),
            peer_addrs: HashMap::new(),
            peer_by_addr: HashMap::new(),
            pending_send: VecDeque::new(),
            pump_calls: 0,
            recv_datagrams: 0,
            recv_bytes: 0,
            recv_unknown_addr_drops: 0,
            sent_datagrams: 0,
            sent_bytes: 0,
            decode_failures: 0,
            encode_failures: 0,
            decoded_events_drained: 0,
            snapshot_emit_calls: 0,
            snapshot_queued_packets: 0,
            snapshot_skipped_cadence: 0,
            snapshot_skipped_duplicate_tick: 0,
            last_snapshot_tick: None,
            sent_by_lane: NetworkLaneMetrics::default(),
            received_by_lane: NetworkLaneMetrics::default(),
        }
    }

    /// Bind a non-blocking UDP socket suitable for this runtime transport loop.
    pub async fn bind<A: ToSocketAddrs>(addr: A) -> io::Result<UdpSocket> {
        UdpSocket::bind(addr).await
    }

    /// Access the underlying NPS packet manager.
    pub fn manager(&self) -> &NetworkPacketManager {
        &self.manager
    }

    /// Mutable access to the underlying NPS packet manager.
    pub fn manager_mut(&mut self) -> &mut NetworkPacketManager {
        &mut self.manager
    }

    /// Register or update a peer/address mapping used by the UDP loop.
    pub fn register_peer(&mut self, peer: PeerId, addr: SocketAddr) {
        if let Some(old_addr) = self.peer_addrs.insert(peer, addr) {
            self.peer_by_addr.remove(&old_addr);
        }
        self.peer_by_addr.insert(addr, peer);
        self.manager.set_peer_addr(peer, addr);
    }

    /// Remove a peer/address mapping.
    pub fn unregister_peer(&mut self, peer: PeerId) -> Option<SocketAddr> {
        let addr = self.peer_addrs.remove(&peer)?;
        self.peer_by_addr.remove(&addr);
        Some(addr)
    }

    /// Number of datagrams waiting in the runtime-owned send queue.
    pub fn pending_send_queue_len(&self) -> usize {
        self.pending_send.len()
    }

    /// Queue an authoritative ParadoxPE snapshot for all registered peers if the cadence allows it.
    pub fn queue_paradox_snapshot_if_due(&mut self, world: &PhysicsWorld) -> usize {
        self.snapshot_emit_calls = self.snapshot_emit_calls.saturating_add(1);
        let cadence = self.config.snapshot_cadence;
        if !cadence.enabled {
            return 0;
        }

        let tick = world.fixed_step_clock().tick();
        if self.last_snapshot_tick == Some(tick) {
            self.snapshot_skipped_duplicate_tick =
                self.snapshot_skipped_duplicate_tick.saturating_add(1);
            return 0;
        }

        let divisor = cadence.emit_every_ticks.max(1);
        if tick % divisor != 0 {
            self.snapshot_skipped_cadence = self.snapshot_skipped_cadence.saturating_add(1);
            return 0;
        }

        let snapshot = world.capture_snapshot();
        let mut queued = 0usize;
        for &peer in self.peer_addrs.keys() {
            self.manager
                .queue_physics_snapshot(peer, tick as u32, &snapshot);
            queued += 1;
        }

        if queued > 0 {
            self.last_snapshot_tick = Some(tick);
            self.snapshot_queued_packets =
                self.snapshot_queued_packets.saturating_add(queued as u64);
        }
        queued
    }

    /// Non-blocking UDP pump. This method never waits on the socket.
    pub fn pump_nonblocking(&mut self, socket: &UdpSocket) -> io::Result<NetworkPumpResult> {
        self.pump_calls = self.pump_calls.saturating_add(1);
        let mut result = NetworkPumpResult::default();

        let recv = self.recv_from_socket(socket)?;
        result.recv_datagrams = recv.0;
        result.recv_bytes = recv.1;
        result.recv_unknown_addr_drops = recv.2;
        self.recv_datagrams = self
            .recv_datagrams
            .saturating_add(result.recv_datagrams as u64);
        self.recv_bytes = self.recv_bytes.saturating_add(result.recv_bytes as u64);
        self.recv_unknown_addr_drops = self
            .recv_unknown_addr_drops
            .saturating_add(result.recv_unknown_addr_drops as u64);

        result.decode_jobs_submitted = self
            .manager
            .submit_inbound_decode_jobs(self.config.max_decode_jobs_per_pump)
            .len();

        result.encode_jobs_submitted = self
            .manager
            .submit_outbound_encode_jobs(self.config.max_encode_jobs_per_pump)
            .len();

        let encoded = self
            .manager
            .drain_encoded_datagrams(self.config.max_send_datagrams_per_pump.max(1));
        result.encoded_datagrams_buffered = encoded.len();
        self.pending_send.extend(encoded);

        let retransmits = self
            .manager
            .collect_retransmit_datagrams(self.config.max_retransmits_per_pump);
        result.retransmit_datagrams_buffered = retransmits.len();
        self.pending_send.extend(retransmits);

        let send_result = self.flush_send_queue(socket)?;
        result.sent_datagrams = send_result.0;
        result.sent_bytes = send_result.1;
        result.pending_send_queue_len = self.pending_send.len();

        self.sent_datagrams = self
            .sent_datagrams
            .saturating_add(result.sent_datagrams as u64);
        self.sent_bytes = self.sent_bytes.saturating_add(result.sent_bytes as u64);

        Ok(result)
    }

    /// Drain decoded packets from NPS and account them in runtime transport telemetry.
    pub fn drain_decoded_packets(&mut self, max_events: usize) -> Vec<DecodedPacketEvent> {
        let events = self.manager.drain_decoded_packets(max_events);
        for event in &events {
            let semantics = packet_semantics(event.header.channel, event.header.kind);
            let payload_bytes = usize::from(event.header.payload_bits).div_ceil(8);
            self.received_by_lane
                .record(semantics.lane, NPS_HEADER_BYTES + payload_bytes);
        }
        self.decoded_events_drained = self
            .decoded_events_drained
            .saturating_add(events.len() as u64);
        events
    }

    /// Drain NPS decode failures and account them in runtime transport telemetry.
    pub fn drain_decode_failures(&mut self, max_events: usize) -> Vec<PacketDecodeFailure> {
        let failures = self.manager.drain_decode_failures(max_events);
        self.decode_failures = self.decode_failures.saturating_add(failures.len() as u64);
        failures
    }

    /// Drain NPS encode failures and account them in runtime transport telemetry.
    pub fn drain_encode_failures(&mut self, max_events: usize) -> Vec<PacketEncodeFailure> {
        let failures = self.manager.drain_encode_failures(max_events);
        self.encode_failures = self.encode_failures.saturating_add(failures.len() as u64);
        failures
    }

    /// Snapshot transport metrics, including underlying NPS manager counters.
    pub fn metrics(&self) -> NetworkTransportMetrics {
        NetworkTransportMetrics {
            pump_calls: self.pump_calls,
            recv_datagrams: self.recv_datagrams,
            recv_bytes: self.recv_bytes,
            recv_unknown_addr_drops: self.recv_unknown_addr_drops,
            sent_datagrams: self.sent_datagrams,
            sent_bytes: self.sent_bytes,
            decode_failures: self.decode_failures,
            encode_failures: self.encode_failures,
            decoded_events_drained: self.decoded_events_drained,
            snapshot_emit_calls: self.snapshot_emit_calls,
            snapshot_queued_packets: self.snapshot_queued_packets,
            snapshot_skipped_cadence: self.snapshot_skipped_cadence,
            snapshot_skipped_duplicate_tick: self.snapshot_skipped_duplicate_tick,
            last_snapshot_tick: self.last_snapshot_tick,
            known_peers: self.peer_addrs.len(),
            pending_send_queue_len: self.pending_send.len(),
            sent_by_lane: self.sent_by_lane,
            received_by_lane: self.received_by_lane,
            peers: self
                .manager
                .all_peer_link_metrics()
                .into_iter()
                .map(|(peer, link)| NetworkPeerMetrics {
                    peer,
                    addr: self.peer_addrs.get(&peer).copied(),
                    link,
                })
                .collect(),
            manager: self.manager.metrics(),
        }
    }

    fn recv_from_socket(&mut self, socket: &UdpSocket) -> io::Result<(usize, usize, usize)> {
        let mut accepted = 0usize;
        let mut recv_bytes = 0usize;
        let mut unknown = 0usize;
        for _ in 0..self.config.max_recv_datagrams_per_pump.max(1) {
            match socket.try_recv_from(&mut self.recv_buffer) {
                Ok((len, addr)) => {
                    let Some(&peer) = self.peer_by_addr.get(&addr) else {
                        unknown += 1;
                        continue;
                    };
                    recv_bytes += len;
                    accepted += 1;
                    let bytes = Arc::<[u8]>::from(self.recv_buffer[..len].to_vec());
                    self.manager
                        .enqueue_inbound_datagram(peer, Some(addr), bytes);
                }
                Err(err) if err.kind() == ErrorKind::WouldBlock => break,
                Err(err) => return Err(err),
            }
        }
        Ok((accepted, recv_bytes, unknown))
    }

    fn flush_send_queue(&mut self, socket: &UdpSocket) -> io::Result<(usize, usize)> {
        let mut sent_datagrams = 0usize;
        let mut sent_bytes = 0usize;

        for _ in 0..self.config.max_send_datagrams_per_pump.max(1) {
            let Some(datagram) = self.pending_send.pop_front() else {
                break;
            };
            let Some(addr) = datagram.addr else {
                continue;
            };

            match socket.try_send_to(&datagram.bytes, addr) {
                Ok(len) => {
                    let semantics = packet_semantics(datagram.header.channel, datagram.header.kind);
                    self.sent_by_lane.record(semantics.lane, len);
                    sent_datagrams += 1;
                    sent_bytes += len;
                }
                Err(err) if err.kind() == ErrorKind::WouldBlock => {
                    self.pending_send.push_front(datagram);
                    break;
                }
                Err(err) => {
                    self.pending_send.push_front(datagram);
                    return Err(err);
                }
            }
        }

        Ok((sent_datagrams, sent_bytes))
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use nps::{DecodedPayload, InputFrame};
    use paradoxpe::{BodyDesc, BodyKind, PhysicsWorldConfig};
    use tokio::task::yield_now;
    use tokio::time::timeout;

    use super::*;

    #[tokio::test]
    async fn transport_runtime_sends_queued_datagrams_to_registered_peer() {
        let runtime_socket = NetworkTransportRuntime::bind("127.0.0.1:0").await.unwrap();
        let peer_socket = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let peer_addr = peer_socket.local_addr().unwrap();

        let mut runtime = NetworkTransportRuntime::new(
            NetworkTransportConfig::default(),
            NetworkPacketConfig::default(),
        );
        runtime.register_peer(7, peer_addr);
        runtime.manager_mut().queue_ack_only(7, 1);

        let mut sent = 0usize;
        for _ in 0..16 {
            let pump = runtime.pump_nonblocking(&runtime_socket).unwrap();
            assert_eq!(pump.encode_jobs_submitted, 0);
            sent += pump.sent_datagrams;
            if sent > 0 {
                break;
            }
            yield_now().await;
        }
        assert_eq!(sent, 1);

        let mut buf = [0u8; 2048];
        let (len, _) = timeout(Duration::from_secs(1), peer_socket.recv_from(&mut buf))
            .await
            .unwrap()
            .unwrap();
        assert!(len > 0);
        let metrics = runtime.metrics();
        assert_eq!(metrics.sent_by_lane.player_input.datagrams, 1);
    }

    #[tokio::test]
    async fn transport_runtime_receives_and_decodes_input_frames() {
        let runtime_socket = NetworkTransportRuntime::bind("127.0.0.1:0").await.unwrap();
        let sender_socket = UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let sender_addr = sender_socket.local_addr().unwrap();
        let runtime_addr = runtime_socket.local_addr().unwrap();

        let mut runtime = NetworkTransportRuntime::new(
            NetworkTransportConfig::default(),
            NetworkPacketConfig::default(),
        );
        runtime.register_peer(3, sender_addr);

        let mut sender_manager = NetworkPacketManager::new(NetworkPacketConfig::default());
        sender_manager.set_peer_addr(9, runtime_addr);
        sender_manager.queue_input_frame(
            9,
            InputFrame {
                tick: 12,
                player_peer: 3,
                buttons: 0b101,
                move_x_q: 32,
                move_y_q: -64,
                aim_x_q: 6_553,
                aim_y_q: 13_107,
            },
        );
        sender_manager.submit_outbound_encode_jobs(4);
        let outbound = sender_manager.drain_encoded_datagrams(4);
        assert_eq!(outbound.len(), 1);

        sender_socket
            .send_to(&outbound[0].bytes, runtime_addr)
            .await
            .unwrap();

        let mut recv = 0usize;
        for _ in 0..16 {
            let pump = runtime.pump_nonblocking(&runtime_socket).unwrap();
            recv += pump.recv_datagrams;
            if recv > 0 {
                break;
            }
            yield_now().await;
        }
        assert_eq!(recv, 1);
        let decoded = runtime.drain_decoded_packets(4);
        assert_eq!(decoded.len(), 1);
        match &decoded[0].payload {
            DecodedPayload::InputFrame(frame) => {
                assert_eq!(frame.tick, 12);
                assert_eq!(frame.player_peer, 3);
            }
            other => panic!("expected input frame, got {other:?}"),
        }
        let metrics = runtime.metrics();
        assert_eq!(metrics.received_by_lane.player_input.datagrams, 1);
    }

    #[test]
    fn paradox_snapshot_cadence_emits_once_per_tick_boundary() {
        let mut runtime = NetworkTransportRuntime::new(
            NetworkTransportConfig {
                snapshot_cadence: SnapshotCadenceConfig {
                    enabled: true,
                    emit_every_ticks: 2,
                },
                ..NetworkTransportConfig::default()
            },
            NetworkPacketConfig::default(),
        );
        runtime.register_peer(1, "127.0.0.1:32000".parse().unwrap());

        let mut world = PhysicsWorld::new(PhysicsWorldConfig::default());
        let _body = world.spawn_body(BodyDesc {
            kind: BodyKind::Dynamic,
            mass: 1.0,
            ..BodyDesc::default()
        });

        assert_eq!(runtime.queue_paradox_snapshot_if_due(&world), 1);
        assert_eq!(runtime.queue_paradox_snapshot_if_due(&world), 0);

        let _ = world.step(world.config().fixed_dt);
        assert_eq!(runtime.queue_paradox_snapshot_if_due(&world), 0);

        let _ = world.step(world.config().fixed_dt);
        assert_eq!(runtime.queue_paradox_snapshot_if_due(&world), 1);
    }
}
