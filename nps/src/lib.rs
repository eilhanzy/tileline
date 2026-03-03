//! Tileline NPS (Network Packet Scaler)
//! ====================================
//!
//! Low-level UDP packet protocol and packet-processing manager integrated with MPS.
//!
//! Design goals:
//! - bit-packed payloads with zero-copy decode views for low overhead
//! - deterministic ownership handoff for physics objects (physgun-style authority transfer)
//! - lightweight UDP reliability for lifecycle events while keeping physics updates unreliable
//! - non-blocking packet encode/decode offload through the MPS scheduler

pub mod bitpack;
pub mod manager;
pub mod model;
pub mod packet;
pub mod reliability;

pub use bitpack::{BitPackError, BitReader, BitWriter};
pub use manager::{
    DecodedPacketEvent, EncodedDatagram, InboundDatagram, NetworkPacketConfig,
    NetworkPacketManager, NetworkPacketMetrics, OutboundPacketJob, OutboundPayload,
    PacketDecodeFailure, PacketEncodeFailure,
};
pub use model::{packet_semantics, PacketLane, PacketSemantics, SnapshotMode, TickScope};
pub use packet::{
    decode_authority_transfer, decode_input_frame, decode_lifecycle_event, decode_payload_owned,
    decode_transform_batch, dequantize_u16_to_unit, encode_authority_transfer, encode_input_frame,
    encode_lifecycle_event, encode_packet, encode_transform_batch, quantize_unit_to_u16,
    AuthorityTransfer, DecodedPayload, GridQuantization, InputFrame, LifecycleEvent, PacketChannel,
    PacketError, PacketFlags, PacketHeader, PacketView, PayloadKind, TransformBatch,
    TransformSample, NPS_HEADER_BYTES, NPS_MAGIC, NPS_VERSION,
};
pub use reliability::{
    seq_more_recent, AckWindow, AuthorityTransferReason, AuthorityTransition, NetObjectHandle,
    PeerId, PeerReliabilityState, PhysAuthorityTable, ReliabilityMode, ReliableInFlight,
    SendPolicy,
};
