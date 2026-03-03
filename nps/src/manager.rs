//! Network packet manager for Tileline's NPS layer.
//!
//! The manager is designed to cooperate with MPS:
//! - inbound datagrams are queued lock-free and decoded on MPS worker tasks
//! - outbound packet encoding can be offloaded as MPS tasks
//! - the render/GMS loop never needs to block on networking work
//!
//! Transport I/O (`recv_from`/`send_to`) can live in a runtime crate; this manager focuses on the
//! packet protocol, reliability bookkeeping, and async task integration.

use std::collections::{HashMap, VecDeque};
use std::net::SocketAddr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use crossbeam::queue::SegQueue;
use mps::{CorePreference, MpsScheduler, TaskId, TaskPriority};
use paradoxpe::PhysicsSnapshot;

use crate::packet::{
    decode_payload_owned, encode_authority_transfer, encode_input_frame, encode_lifecycle_event,
    encode_packet, encode_transform_batch, AuthorityTransfer, DecodedPayload, GridQuantization,
    InputFrame, LifecycleEvent, PacketChannel, PacketError, PacketFlags, PacketHeader, PacketView,
    PayloadKind, TransformBatch, TransformSample, NPS_HEADER_BYTES, NPS_MAGIC, NPS_VERSION,
};
use crate::reliability::{
    PeerId, PeerLinkMetrics, PeerReliabilityState, PhysAuthorityTable, SendPolicy,
};

/// Default datagram budget used by NPS manager encode tasks.
pub const DEFAULT_MAX_DATAGRAM_BYTES: usize = 1200;

/// NPS manager configuration.
#[derive(Debug, Clone)]
pub struct NetworkPacketConfig {
    /// Physics grid quantization domain for transform packing.
    pub grid: GridQuantization,
    /// Maximum datagram size (including NPS header).
    pub max_datagram_bytes: usize,
    /// MPS priority for inbound decode work.
    pub decode_priority: TaskPriority,
    /// MPS priority for outbound encode work.
    pub encode_priority: TaskPriority,
    /// Core preference for decode work.
    pub decode_core_preference: CorePreference,
    /// Core preference for encode work.
    pub encode_core_preference: CorePreference,
    /// Velocity magnitude that maps to the `-1..=1` normalized transport domain.
    pub velocity_normalization: f32,
}

impl Default for NetworkPacketConfig {
    fn default() -> Self {
        Self {
            grid: GridQuantization::TILELINE_2048,
            max_datagram_bytes: DEFAULT_MAX_DATAGRAM_BYTES,
            decode_priority: TaskPriority::High,
            encode_priority: TaskPriority::Normal,
            decode_core_preference: CorePreference::Performance,
            encode_core_preference: CorePreference::Auto,
            velocity_normalization: 64.0,
        }
    }
}

/// Inbound raw datagram queued for decode.
#[derive(Debug, Clone)]
pub struct InboundDatagram {
    /// Remote peer id assigned by session layer.
    pub peer: PeerId,
    /// Optional socket address metadata for diagnostics/transport mapping.
    pub addr: Option<SocketAddr>,
    /// Raw UDP datagram bytes.
    pub bytes: Arc<[u8]>,
    /// Receipt timestamp.
    pub received_at: Instant,
}

/// Decoded inbound packet event.
#[derive(Debug, Clone)]
pub struct DecodedPacketEvent {
    /// Peer id associated with the datagram.
    pub peer: PeerId,
    /// Optional socket address metadata.
    pub addr: Option<SocketAddr>,
    /// Parsed packet header.
    pub header: PacketHeader,
    /// Owned decoded payload for cross-thread handoff.
    pub payload: DecodedPayload,
    /// Decode completion timestamp.
    pub decoded_at: Instant,
}

/// Decoding failure returned from worker tasks.
#[derive(Debug, Clone)]
pub struct PacketDecodeFailure {
    /// Peer id associated with the datagram.
    pub peer: PeerId,
    /// Optional source address.
    pub addr: Option<SocketAddr>,
    /// Decode error.
    pub error: PacketError,
    /// When decode failed.
    pub failed_at: Instant,
}

/// Encoded outbound datagram ready for UDP send.
#[derive(Debug, Clone)]
pub struct EncodedDatagram {
    /// Destination peer id.
    pub peer: PeerId,
    /// Optional destination socket address.
    pub addr: Option<SocketAddr>,
    /// Parsed header used to build the datagram.
    pub header: PacketHeader,
    /// Encoded UDP datagram bytes.
    pub bytes: Arc<[u8]>,
    /// Encode completion timestamp.
    pub encoded_at: Instant,
}

/// Encoding failure returned from worker tasks.
#[derive(Debug, Clone)]
pub struct PacketEncodeFailure {
    /// Destination peer id.
    pub peer: PeerId,
    /// Optional destination address.
    pub addr: Option<SocketAddr>,
    /// Encode error.
    pub error: PacketError,
    /// Failure timestamp.
    pub failed_at: Instant,
}

/// Outbound payload queued for bit-packing and header emission.
#[derive(Debug, Clone)]
pub enum OutboundPayload {
    /// Physics transform snapshot batch (bit-packed, unreliable sequenced by default).
    TransformBatch(Vec<TransformSample>),
    /// Deterministic input frame.
    InputFrame(InputFrame),
    /// Reliable lifecycle event.
    LifecycleEvent(LifecycleEvent),
    /// Authority transfer event (physgun ownership handoff).
    AuthorityTransfer(AuthorityTransfer),
    /// ACK-only heartbeat.
    AckOnly,
}

impl OutboundPayload {
    fn kind(&self) -> PayloadKind {
        match self {
            Self::TransformBatch(_) => PayloadKind::TransformBatch,
            Self::InputFrame(_) => PayloadKind::InputFrame,
            Self::LifecycleEvent(_) => PayloadKind::LifecycleEvent,
            Self::AuthorityTransfer(_) => PayloadKind::AuthorityTransfer,
            Self::AckOnly => PayloadKind::AckOnly,
        }
    }
}

/// Outbound packet encode job prepared on the game/network thread.
#[derive(Debug, Clone)]
pub struct OutboundPacketJob {
    /// Destination peer id.
    pub peer: PeerId,
    /// Optional destination socket address.
    pub addr: Option<SocketAddr>,
    /// Wire channel.
    pub channel: PacketChannel,
    /// Packet header prepared with seq/ack metadata.
    pub header: PacketHeader,
    /// Send policy used for retransmit bookkeeping.
    pub policy: SendPolicy,
    /// Payload body to encode.
    pub payload: OutboundPayload,
}

/// Snapshot of manager counters for profiling and frame-budget tuning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct NetworkPacketMetrics {
    pub inbound_enqueued: u64,
    pub decode_submitted: u64,
    pub decoded_ok: u64,
    pub decode_failed: u64,
    pub outbound_queued: u64,
    pub encode_submitted: u64,
    pub encoded_ok: u64,
    pub encode_failed: u64,
    pub retransmits_emitted: u64,
    pub mps_task_failures: u64,
}

#[derive(Default)]
struct MetricsAtomic {
    inbound_enqueued: AtomicU64,
    decode_submitted: AtomicU64,
    decoded_ok: AtomicU64,
    decode_failed: AtomicU64,
    outbound_queued: AtomicU64,
    encode_submitted: AtomicU64,
    encoded_ok: AtomicU64,
    encode_failed: AtomicU64,
    retransmits_emitted: AtomicU64,
    mps_task_failures: AtomicU64,
}

impl MetricsAtomic {
    fn snapshot(&self) -> NetworkPacketMetrics {
        NetworkPacketMetrics {
            inbound_enqueued: self.inbound_enqueued.load(Ordering::Relaxed),
            decode_submitted: self.decode_submitted.load(Ordering::Relaxed),
            decoded_ok: self.decoded_ok.load(Ordering::Relaxed),
            decode_failed: self.decode_failed.load(Ordering::Relaxed),
            outbound_queued: self.outbound_queued.load(Ordering::Relaxed),
            encode_submitted: self.encode_submitted.load(Ordering::Relaxed),
            encoded_ok: self.encoded_ok.load(Ordering::Relaxed),
            encode_failed: self.encode_failed.load(Ordering::Relaxed),
            retransmits_emitted: self.retransmits_emitted.load(Ordering::Relaxed),
            mps_task_failures: self.mps_task_failures.load(Ordering::Relaxed),
        }
    }
}

#[derive(Debug)]
struct PeerState {
    addr: Option<SocketAddr>,
    reliability: PeerReliabilityState,
}

impl PeerState {
    fn new(peer: PeerId, addr: Option<SocketAddr>) -> Self {
        Self {
            addr,
            reliability: PeerReliabilityState::new(peer),
        }
    }
}

/// Low-level packet manager that cooperates with MPS for async decode/encode work.
pub struct NetworkPacketManager {
    config: NetworkPacketConfig,
    mps: Option<Arc<MpsScheduler>>,
    inbound_queue: Arc<SegQueue<InboundDatagram>>,
    decoded_queue: Arc<SegQueue<DecodedPacketEvent>>,
    decode_failures: Arc<SegQueue<PacketDecodeFailure>>,
    encoded_queue: Arc<SegQueue<EncodedDatagram>>,
    encode_failures: Arc<SegQueue<PacketEncodeFailure>>,
    pending_outbound_jobs: VecDeque<OutboundPacketJob>,
    peers: HashMap<PeerId, PeerState>,
    authority: PhysAuthorityTable,
    metrics: Arc<MetricsAtomic>,
}

impl NetworkPacketManager {
    /// Construct a manager without MPS offload (encode/decode runs inline when pumped).
    pub fn new(config: NetworkPacketConfig) -> Self {
        Self::with_mps(config, None)
    }

    /// Construct a manager with optional MPS scheduler integration.
    pub fn with_mps(config: NetworkPacketConfig, mps: Option<Arc<MpsScheduler>>) -> Self {
        Self {
            config,
            mps,
            inbound_queue: Arc::new(SegQueue::new()),
            decoded_queue: Arc::new(SegQueue::new()),
            decode_failures: Arc::new(SegQueue::new()),
            encoded_queue: Arc::new(SegQueue::new()),
            encode_failures: Arc::new(SegQueue::new()),
            pending_outbound_jobs: VecDeque::new(),
            peers: HashMap::new(),
            authority: PhysAuthorityTable::default(),
            metrics: Arc::new(MetricsAtomic::default()),
        }
    }

    /// Attach or replace the MPS scheduler used for async packet processing.
    pub fn set_mps_scheduler(&mut self, scheduler: Option<Arc<MpsScheduler>>) {
        self.mps = scheduler;
    }

    /// Register/update peer address metadata.
    pub fn set_peer_addr(&mut self, peer: PeerId, addr: SocketAddr) {
        self.peer_state_mut(peer, Some(addr));
    }

    /// Queue a raw inbound UDP datagram for decode.
    pub fn enqueue_inbound_datagram(
        &self,
        peer: PeerId,
        addr: Option<SocketAddr>,
        datagram: Arc<[u8]>,
    ) {
        self.inbound_queue.push(InboundDatagram {
            peer,
            addr,
            bytes: datagram,
            received_at: Instant::now(),
        });
        self.metrics
            .inbound_enqueued
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Submit up to `max_jobs` inbound decode tasks to MPS (or decode inline if no scheduler).
    pub fn submit_inbound_decode_jobs(&self, max_jobs: usize) -> Vec<TaskId> {
        let mut task_ids = Vec::new();
        for _ in 0..max_jobs {
            let Some(datagram) = self.inbound_queue.pop() else {
                break;
            };
            if let Some(mps) = &self.mps {
                let decoded_queue = Arc::clone(&self.decoded_queue);
                let decode_failures = Arc::clone(&self.decode_failures);
                let metrics = Arc::clone(&self.metrics);
                let task = move || match decode_inbound_datagram(datagram) {
                    Ok(event) => {
                        decoded_queue.push(event);
                        metrics.decoded_ok.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(err) => {
                        decode_failures.push(err);
                        metrics.decode_failed.fetch_add(1, Ordering::Relaxed);
                    }
                };
                let task_id = mps.submit_native(
                    self.config.decode_priority,
                    self.config.decode_core_preference,
                    task,
                );
                self.metrics
                    .decode_submitted
                    .fetch_add(1, Ordering::Relaxed);
                task_ids.push(task_id);
            } else {
                match decode_inbound_datagram(datagram) {
                    Ok(event) => {
                        self.decoded_queue.push(event);
                        self.metrics.decoded_ok.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(err) => {
                        self.decode_failures.push(err);
                        self.metrics.decode_failed.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        }
        task_ids
    }

    /// Drain decoded packets and update per-peer ACK/reliability state on the caller thread.
    pub fn drain_decoded_packets(&mut self, max_events: usize) -> Vec<DecodedPacketEvent> {
        let mut out = Vec::with_capacity(max_events);
        for _ in 0..max_events {
            let Some(event) = self.decoded_queue.pop() else {
                break;
            };
            let peer = self.peer_state_mut(event.peer, event.addr);
            peer.addr = event.addr.or(peer.addr);
            peer.reliability.observe_inbound(
                event.header.sequence,
                event.header.ack,
                event.header.ack_bits,
                event.decoded_at,
            );
            out.push(event);
        }
        out
    }

    /// Drain decode failures collected by worker tasks.
    pub fn drain_decode_failures(&self, max_events: usize) -> Vec<PacketDecodeFailure> {
        drain_segqueue(&self.decode_failures, max_events)
    }

    /// Queue a bit-packed physics transform batch for outbound encode.
    pub fn queue_physics_transforms(
        &mut self,
        peer: PeerId,
        tick: u32,
        samples: Vec<TransformSample>,
    ) {
        let addr = self.peers.get(&peer).and_then(|p| p.addr);
        self.queue_outbound_payload(
            peer,
            addr,
            PacketChannel::Physics,
            tick,
            OutboundPayload::TransformBatch(samples),
        );
    }

    /// Queue a ParadoxPE world snapshot through the existing transform batch packet path.
    ///
    /// Object ids are the packed ParadoxPE body handles. Owner ids are resolved from the authority
    /// table when available; otherwise they fall back to `0`.
    pub fn queue_physics_snapshot(&mut self, peer: PeerId, tick: u32, snapshot: &PhysicsSnapshot) {
        let normalization = self.config.velocity_normalization.max(1.0);
        let samples = snapshot
            .bodies
            .iter()
            .map(|body| {
                let owner_peer = self.current_object_owner(body.handle.raw()).unwrap_or(0);
                let flags = if body.awake { 0x1 } else { 0x0 };
                TransformSample::quantize(
                    body.handle.raw(),
                    owner_peer,
                    body.position.x,
                    body.position.y,
                    (body.linear_velocity.x / normalization).clamp(-1.0, 1.0),
                    (body.linear_velocity.y / normalization).clamp(-1.0, 1.0),
                    flags,
                    self.config.grid,
                )
            })
            .collect();
        self.queue_physics_transforms(peer, tick, samples);
    }

    /// Queue a deterministic input frame for outbound encode.
    pub fn queue_input_frame(&mut self, peer: PeerId, input: InputFrame) {
        let addr = self.peers.get(&peer).and_then(|p| p.addr);
        self.queue_outbound_payload(
            peer,
            addr,
            PacketChannel::Input,
            input.tick,
            OutboundPayload::InputFrame(input),
        );
    }

    /// Queue a reliable lifecycle event for outbound encode.
    pub fn queue_lifecycle_event(&mut self, peer: PeerId, tick: u32, event: LifecycleEvent) {
        let addr = self.peers.get(&peer).and_then(|p| p.addr);
        self.queue_outbound_payload(
            peer,
            addr,
            PacketChannel::Lifecycle,
            tick,
            OutboundPayload::LifecycleEvent(event),
        );
    }

    /// Queue an authority transfer event (used for physgun ownership handoff replication).
    pub fn queue_authority_transfer_event(
        &mut self,
        peer: PeerId,
        tick: u32,
        transfer: AuthorityTransfer,
    ) {
        let addr = self.peers.get(&peer).and_then(|p| p.addr);
        self.queue_outbound_payload(
            peer,
            addr,
            PacketChannel::Physics,
            tick,
            OutboundPayload::AuthorityTransfer(transfer),
        );
    }

    /// Queue an ACK-only heartbeat packet.
    pub fn queue_ack_only(&mut self, peer: PeerId, tick: u32) {
        let addr = self.peers.get(&peer).and_then(|p| p.addr);
        self.queue_outbound_payload(
            peer,
            addr,
            PacketChannel::Input,
            tick,
            OutboundPayload::AckOnly,
        );
    }

    /// Submit outbound encode jobs to MPS (or encode inline if no scheduler).
    pub fn submit_outbound_encode_jobs(&mut self, max_jobs: usize) -> Vec<TaskId> {
        let mut task_ids = Vec::new();
        for _ in 0..max_jobs {
            let Some(job) = self.pending_outbound_jobs.pop_front() else {
                break;
            };
            if let Some(mps) = &self.mps {
                let encoded_queue = Arc::clone(&self.encoded_queue);
                let encode_failures = Arc::clone(&self.encode_failures);
                let metrics = Arc::clone(&self.metrics);
                let max_len = self.config.max_datagram_bytes;
                let task = move || match encode_outbound_job(job, max_len) {
                    Ok(datagram) => {
                        encoded_queue.push(datagram);
                        metrics.encoded_ok.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(err) => {
                        encode_failures.push(err);
                        metrics.encode_failed.fetch_add(1, Ordering::Relaxed);
                    }
                };
                let task_id = mps.submit_native(
                    self.config.encode_priority,
                    self.config.encode_core_preference,
                    task,
                );
                self.metrics
                    .encode_submitted
                    .fetch_add(1, Ordering::Relaxed);
                task_ids.push(task_id);
            } else {
                match encode_outbound_job(job, self.config.max_datagram_bytes) {
                    Ok(datagram) => {
                        self.encoded_queue.push(datagram);
                        self.metrics.encoded_ok.fetch_add(1, Ordering::Relaxed);
                    }
                    Err(err) => {
                        self.encode_failures.push(err);
                        self.metrics.encode_failed.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        }
        task_ids
    }

    /// Drain encoded datagrams ready for UDP `send_to`.
    pub fn drain_encoded_datagrams(&mut self, max_events: usize) -> Vec<EncodedDatagram> {
        let mut out = Vec::with_capacity(max_events);
        for _ in 0..max_events {
            let Some(datagram) = self.encoded_queue.pop() else {
                break;
            };
            self.track_reliable_after_encode(&datagram);
            out.push(datagram);
        }
        out
    }

    /// Drain outbound encode failures collected by worker tasks.
    pub fn drain_encode_failures(&self, max_events: usize) -> Vec<PacketEncodeFailure> {
        drain_segqueue(&self.encode_failures, max_events)
    }

    /// Emit retransmit datagrams for reliable packets whose timeout elapsed.
    pub fn collect_retransmit_datagrams(&mut self, max_events: usize) -> Vec<EncodedDatagram> {
        let now = Instant::now();
        let mut out = Vec::new();
        for (&peer_id, peer) in &mut self.peers {
            if out.len() >= max_events {
                break;
            }
            let due = peer.reliability.collect_retransmit_due(now);
            for inflight in due {
                if out.len() >= max_events {
                    break;
                }
                let mut bytes = inflight.datagram.to_vec();
                if bytes.len() >= NPS_HEADER_BYTES {
                    bytes[3] |= PacketFlags::RETRANSMIT.0;
                    if let Ok(header) = PacketHeader::decode(&bytes) {
                        out.push(EncodedDatagram {
                            peer: peer_id,
                            addr: peer.addr,
                            header,
                            bytes: Arc::from(bytes.into_boxed_slice()),
                            encoded_at: now,
                        });
                        self.metrics
                            .retransmits_emitted
                            .fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        }
        out
    }

    /// Set the baseline authoritative owner for a physics object.
    pub fn set_object_master_owner(&mut self, object: u32, owner: PeerId) {
        self.authority.set_master_owner(object, owner);
    }

    /// Transfer authority to a player grabbing an object with a physgun-like tool.
    pub fn begin_physgun_grab(
        &mut self,
        object: u32,
        grabbing_peer: PeerId,
    ) -> Option<AuthorityTransfer> {
        let transition = self.authority.begin_physgun_grab(object, grabbing_peer)?;
        Some(AuthorityTransfer {
            object: transition.object,
            previous_owner: transition.previous_owner,
            new_owner: transition.new_owner,
            reason: transition.reason as u8,
        })
    }

    /// End a physgun grab and restore master ownership.
    pub fn end_physgun_grab(
        &mut self,
        object: u32,
        releasing_peer: PeerId,
    ) -> Option<AuthorityTransfer> {
        let transition = self.authority.end_physgun_grab(object, releasing_peer)?;
        Some(AuthorityTransfer {
            object: transition.object,
            previous_owner: transition.previous_owner,
            new_owner: transition.new_owner,
            reason: transition.reason as u8,
        })
    }

    /// Read the current authoritative owner for a physics object.
    pub fn current_object_owner(&self, object: u32) -> Option<PeerId> {
        self.authority.current_owner(object)
    }

    /// Snapshot best-effort link telemetry for one peer.
    pub fn peer_link_metrics(&self, peer: PeerId) -> Option<PeerLinkMetrics> {
        self.peers.get(&peer).map(|peer| peer.reliability.metrics())
    }

    /// Snapshot link telemetry for all known peers.
    pub fn all_peer_link_metrics(&self) -> Vec<(PeerId, PeerLinkMetrics)> {
        let mut out = self
            .peers
            .iter()
            .map(|(&peer_id, peer)| (peer_id, peer.reliability.metrics()))
            .collect::<Vec<_>>();
        out.sort_by_key(|(peer_id, _)| *peer_id);
        out
    }

    /// Snapshot lock-free metrics counters.
    pub fn metrics(&self) -> NetworkPacketMetrics {
        self.metrics.snapshot()
    }

    fn queue_outbound_payload(
        &mut self,
        peer: PeerId,
        addr: Option<SocketAddr>,
        channel: PacketChannel,
        tick: u32,
        payload: OutboundPayload,
    ) {
        let payload_kind = payload.kind();
        let policy = SendPolicy::for_payload(channel, payload_kind);
        let peer_state = self.peer_state_mut(peer, addr);
        let (ack, ack_bits) = peer_state.reliability.ack_header_fields();
        let sequence = peer_state.reliability.next_sequence();

        let mut flags = policy.wire_flags();
        if matches!(payload_kind, PayloadKind::AuthorityTransfer) {
            flags = flags.union(PacketFlags::AUTHORITY_EVENT);
        }

        let header = PacketHeader {
            magic: NPS_MAGIC,
            version: NPS_VERSION,
            flags,
            channel,
            kind: payload_kind,
            sequence,
            ack,
            ack_bits,
            tick,
            payload_bits: 0,
        };

        self.pending_outbound_jobs.push_back(OutboundPacketJob {
            peer,
            addr,
            channel,
            header,
            policy,
            payload,
        });
        self.metrics.outbound_queued.fetch_add(1, Ordering::Relaxed);
    }

    fn peer_state_mut(&mut self, peer: PeerId, addr: Option<SocketAddr>) -> &mut PeerState {
        let state = self
            .peers
            .entry(peer)
            .or_insert_with(|| PeerState::new(peer, addr));
        if addr.is_some() {
            state.addr = addr;
        }
        state
    }

    fn track_reliable_after_encode(&mut self, datagram: &EncodedDatagram) {
        if !datagram.header.flags.contains(PacketFlags::RELIABLE) {
            return;
        }
        let now = datagram.encoded_at;
        let peer = self.peer_state_mut(datagram.peer, datagram.addr);
        peer.reliability.track_reliable(
            datagram.header.sequence,
            datagram.header.channel,
            datagram.header.kind,
            Arc::clone(&datagram.bytes),
            now,
        );
    }
}

fn decode_inbound_datagram(
    datagram: InboundDatagram,
) -> Result<DecodedPacketEvent, PacketDecodeFailure> {
    let decoded_at = Instant::now();
    let view = PacketView::parse(&datagram.bytes).map_err(|error| PacketDecodeFailure {
        peer: datagram.peer,
        addr: datagram.addr,
        error,
        failed_at: decoded_at,
    })?;
    let payload = decode_payload_owned(&view).map_err(|error| PacketDecodeFailure {
        peer: datagram.peer,
        addr: datagram.addr,
        error,
        failed_at: decoded_at,
    })?;
    Ok(DecodedPacketEvent {
        peer: datagram.peer,
        addr: datagram.addr,
        header: view.header,
        payload,
        decoded_at,
    })
}

fn encode_outbound_job(
    job: OutboundPacketJob,
    max_datagram_bytes: usize,
) -> Result<EncodedDatagram, PacketEncodeFailure> {
    let encoded_at = Instant::now();
    let mut buf = vec![0u8; max_datagram_bytes.max(NPS_HEADER_BYTES + 8)];
    let result = encode_packet(job.header, &mut buf, |writer| match &job.payload {
        OutboundPayload::TransformBatch(samples) => {
            encode_transform_batch(writer, &TransformBatch { samples })
        }
        OutboundPayload::InputFrame(input) => encode_input_frame(writer, *input),
        OutboundPayload::LifecycleEvent(event) => encode_lifecycle_event(writer, *event),
        OutboundPayload::AuthorityTransfer(transfer) => {
            encode_authority_transfer(writer, *transfer)
        }
        OutboundPayload::AckOnly => Ok(()),
    });

    match result {
        Ok(datagram_len) => {
            buf.truncate(datagram_len);
            let header = PacketHeader::decode(&buf).map_err(|error| PacketEncodeFailure {
                peer: job.peer,
                addr: job.addr,
                error,
                failed_at: encoded_at,
            })?;
            Ok(EncodedDatagram {
                peer: job.peer,
                addr: job.addr,
                header,
                bytes: Arc::from(buf.into_boxed_slice()),
                encoded_at,
            })
        }
        Err(error) => Err(PacketEncodeFailure {
            peer: job.peer,
            addr: job.addr,
            error,
            failed_at: encoded_at,
        }),
    }
}

fn drain_segqueue<T: Clone>(queue: &SegQueue<T>, max_events: usize) -> Vec<T> {
    let mut out = Vec::with_capacity(max_events);
    for _ in 0..max_events {
        let Some(item) = queue.pop() else {
            break;
        };
        out.push(item);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reliability::AuthorityTransferReason;
    use paradoxpe::{BodyHandle, BodyStateFrame, PhysicsSnapshot};

    #[test]
    fn manager_roundtrips_physics_batch_without_mps() {
        let mut mgr = NetworkPacketManager::new(NetworkPacketConfig::default());
        mgr.set_peer_addr(7, "127.0.0.1:27015".parse().unwrap());

        let sample = TransformSample::quantize(
            100,
            7,
            256.0,
            512.0,
            0.1,
            -0.2,
            3,
            GridQuantization::TILELINE_2048,
        );
        mgr.queue_physics_transforms(7, 33, vec![sample]);
        mgr.submit_outbound_encode_jobs(8);
        let datagrams = mgr.drain_encoded_datagrams(8);
        assert_eq!(datagrams.len(), 1);

        mgr.enqueue_inbound_datagram(7, None, Arc::clone(&datagrams[0].bytes));
        mgr.submit_inbound_decode_jobs(8);
        let decoded = mgr.drain_decoded_packets(8);
        assert_eq!(decoded.len(), 1);
        match &decoded[0].payload {
            DecodedPayload::TransformBatch(samples) => {
                assert_eq!(samples.len(), 1);
                assert_eq!(samples[0].object, 100);
            }
            _ => panic!("expected transform batch"),
        }
    }

    #[test]
    fn manager_tracks_reliable_lifecycle_packets() {
        let mut mgr = NetworkPacketManager::new(NetworkPacketConfig::default());
        mgr.queue_lifecycle_event(
            1,
            90,
            LifecycleEvent {
                opcode: 3,
                event_id: 22,
                object: 42,
                arg: 7,
            },
        );
        mgr.submit_outbound_encode_jobs(4);
        let datagrams = mgr.drain_encoded_datagrams(4);
        assert_eq!(datagrams.len(), 1);
        assert!(datagrams[0].header.flags.contains(PacketFlags::RELIABLE));
        let retransmits = mgr.collect_retransmit_datagrams(4);
        assert!(retransmits.is_empty());
    }

    #[test]
    fn physgun_handoff_api_emits_authority_transfer_payload() {
        let mut mgr = NetworkPacketManager::new(NetworkPacketConfig::default());
        mgr.set_object_master_owner(77, 1);
        let grab = mgr.begin_physgun_grab(77, 9).unwrap();
        assert_eq!(grab.reason, AuthorityTransferReason::PhysgunGrab as u8);
        assert_eq!(mgr.current_object_owner(77), Some(9));
        let release = mgr.end_physgun_grab(77, 9).unwrap();
        assert_eq!(
            release.reason,
            AuthorityTransferReason::PhysgunRelease as u8
        );
        assert_eq!(mgr.current_object_owner(77), Some(1));
    }

    #[test]
    fn manager_queues_paradoxpe_snapshot_as_transform_batch() {
        let mut mgr = NetworkPacketManager::new(NetworkPacketConfig::default());
        mgr.set_peer_addr(4, "127.0.0.1:27016".parse().unwrap());
        let body = BodyHandle::new(2, 1);
        mgr.set_object_master_owner(body.raw(), 4);
        let snapshot = PhysicsSnapshot {
            tick: 55,
            bodies: vec![BodyStateFrame {
                handle: body,
                position: nalgebra::Vector3::new(32.0, 64.0, 0.0),
                rotation: nalgebra::UnitQuaternion::identity(),
                linear_velocity: nalgebra::Vector3::new(8.0, -4.0, 0.0),
                awake: true,
            }],
        };

        mgr.queue_physics_snapshot(4, 55, &snapshot);
        mgr.submit_outbound_encode_jobs(4);
        let datagrams = mgr.drain_encoded_datagrams(4);
        assert_eq!(datagrams.len(), 1);

        mgr.enqueue_inbound_datagram(4, None, Arc::clone(&datagrams[0].bytes));
        mgr.submit_inbound_decode_jobs(4);
        let decoded = mgr.drain_decoded_packets(4);
        match &decoded[0].payload {
            DecodedPayload::TransformBatch(samples) => {
                assert_eq!(samples.len(), 1);
                assert_eq!(samples[0].object, body.raw());
                assert_eq!(samples[0].owner_peer, 4);
                assert_eq!(samples[0].flags & 0x1, 0x1);
            }
            other => panic!("expected transform batch, got {other:?}"),
        }
    }
}
