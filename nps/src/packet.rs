//! Low-level NPS UDP packet layout and bit-packed payload codecs.
//!
//! The protocol is designed for low-latency gameplay networking:
//! - fixed-size byte header for fast validation and ACK bookkeeping
//! - bit-packed payload body for transform/input snapshots
//! - quantized grid encoding (`f32` <-> `u16`) to minimize bandwidth

use std::fmt;

use crate::bitpack::{BitPackError, BitReader, BitWriter};

/// NPS protocol magic (`TN` / Tileline Network).
pub const NPS_MAGIC: u16 = 0x544E;
/// Current wire version.
pub const NPS_VERSION: u8 = 1;
/// Header size in bytes.
pub const NPS_HEADER_BYTES: usize = 20;

/// Grid quantization domain for physics coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GridQuantization {
    /// Grid width (e.g. `2048.0`).
    pub width: f32,
    /// Grid height (e.g. `2048.0`).
    pub height: f32,
}

impl GridQuantization {
    /// Canonical Tileline physics grid default used by snapshot packing.
    pub const TILELINE_2048: Self = Self {
        width: 2048.0,
        height: 2048.0,
    };

    /// Quantize a 2D position to `u16` grid coordinates.
    pub fn encode_position(&self, x: f32, y: f32) -> (u16, u16) {
        (
            quantize_unit_to_u16((x / self.width).clamp(0.0, 1.0)),
            quantize_unit_to_u16((y / self.height).clamp(0.0, 1.0)),
        )
    }

    /// Decode a quantized `u16` pair back into world-space coordinates.
    pub fn decode_position(&self, x: u16, y: u16) -> (f32, f32) {
        (
            dequantize_u16_to_unit(x) * self.width,
            dequantize_u16_to_unit(y) * self.height,
        )
    }

    /// Quantize a signed normalized velocity component (`-1..=1`) to `u16`.
    pub fn encode_signed_unit(&self, value: f32) -> u16 {
        quantize_unit_to_u16(((value.clamp(-1.0, 1.0) + 1.0) * 0.5).clamp(0.0, 1.0))
    }

    /// Decode a signed normalized velocity component (`u16` -> `-1..=1`).
    pub fn decode_signed_unit(&self, raw: u16) -> f32 {
        (dequantize_u16_to_unit(raw) * 2.0) - 1.0
    }
}

#[inline]
pub fn quantize_unit_to_u16(value: f32) -> u16 {
    (value.clamp(0.0, 1.0) * u16::MAX as f32).round() as u16
}

#[inline]
pub fn dequantize_u16_to_unit(value: u16) -> f32 {
    value as f32 / u16::MAX as f32
}

/// Wire channel class for packet scheduling and reliability policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PacketChannel {
    /// Fast physics/state snapshots (unreliable sequenced).
    Physics = 0,
    /// Player input stream (unreliable sequenced or lightly reliable).
    Input = 1,
    /// Lifecycle/stateful events (reliable ordered).
    Lifecycle = 2,
    /// UI/chat/tooling events (reliable ordered).
    Ui = 3,
    /// Custom plugin/script channel.
    Script = 4,
}

impl PacketChannel {
    fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Physics),
            1 => Some(Self::Input),
            2 => Some(Self::Lifecycle),
            3 => Some(Self::Ui),
            4 => Some(Self::Script),
            _ => None,
        }
    }
}

/// Wire payload kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PayloadKind {
    /// Packed transform updates for physics objects.
    TransformBatch = 1,
    /// Player input frame for deterministic simulation.
    InputFrame = 2,
    /// Lifecycle event (reliable) such as spawn/despawn or scripted event.
    LifecycleEvent = 3,
    /// Authority transfer event (physgun handoff).
    AuthorityTransfer = 4,
    /// ACK-only keepalive or timing packet.
    AckOnly = 5,
}

impl PayloadKind {
    fn from_u8(value: u8) -> Option<Self> {
        match value {
            1 => Some(Self::TransformBatch),
            2 => Some(Self::InputFrame),
            3 => Some(Self::LifecycleEvent),
            4 => Some(Self::AuthorityTransfer),
            5 => Some(Self::AckOnly),
            _ => None,
        }
    }
}

/// Packet flags (bitfield stored in the wire header).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PacketFlags(pub u8);

impl PacketFlags {
    /// No flags set.
    pub const NONE: Self = Self(0);
    /// Packet carries reliable-ordered semantics.
    pub const RELIABLE: Self = Self(1 << 0);
    /// Packet should be sequenced (newer supersedes older).
    pub const SEQUENCED: Self = Self(1 << 1);
    /// Packet contains an authority handoff state update.
    pub const AUTHORITY_EVENT: Self = Self(1 << 2);
    /// Packet is a retransmit copy.
    pub const RETRANSMIT: Self = Self(1 << 3);

    #[inline]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    #[inline]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

/// Fixed-size wire header (byte-aligned for fast validation and ACK bookkeeping).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PacketHeader {
    /// Protocol magic.
    pub magic: u16,
    /// Protocol version.
    pub version: u8,
    /// Packet flags.
    pub flags: PacketFlags,
    /// Logical channel.
    pub channel: PacketChannel,
    /// Payload kind.
    pub kind: PayloadKind,
    /// Packet sequence number (per-peer send stream).
    pub sequence: u16,
    /// Most recent sequence received from the remote peer.
    pub ack: u16,
    /// 32-bit ACK bitmask for the 32 packets before `ack`.
    pub ack_bits: u32,
    /// Simulation tick associated with the payload.
    pub tick: u32,
    /// Payload length in bits for the packed payload body.
    pub payload_bits: u16,
}

impl PacketHeader {
    /// Encode the header into a fixed 20-byte array.
    pub fn encode(&self) -> [u8; NPS_HEADER_BYTES] {
        let mut out = [0u8; NPS_HEADER_BYTES];
        out[0..2].copy_from_slice(&self.magic.to_le_bytes());
        out[2] = self.version;
        out[3] = self.flags.0;
        out[4] = self.channel as u8;
        out[5] = self.kind as u8;
        out[6..8].copy_from_slice(&self.sequence.to_le_bytes());
        out[8..10].copy_from_slice(&self.ack.to_le_bytes());
        out[10..14].copy_from_slice(&self.ack_bits.to_le_bytes());
        out[14..18].copy_from_slice(&self.tick.to_le_bytes());
        out[18..20].copy_from_slice(&self.payload_bits.to_le_bytes());
        out
    }

    /// Decode a header from a byte slice.
    pub fn decode(bytes: &[u8]) -> Result<Self, PacketError> {
        if bytes.len() < NPS_HEADER_BYTES {
            return Err(PacketError::TruncatedHeader);
        }
        let magic = u16::from_le_bytes([bytes[0], bytes[1]]);
        let version = bytes[2];
        let flags = PacketFlags(bytes[3]);
        let channel = PacketChannel::from_u8(bytes[4]).ok_or(PacketError::InvalidChannel)?;
        let kind = PayloadKind::from_u8(bytes[5]).ok_or(PacketError::InvalidPayloadKind)?;
        let sequence = u16::from_le_bytes([bytes[6], bytes[7]]);
        let ack = u16::from_le_bytes([bytes[8], bytes[9]]);
        let ack_bits = u32::from_le_bytes([bytes[10], bytes[11], bytes[12], bytes[13]]);
        let tick = u32::from_le_bytes([bytes[14], bytes[15], bytes[16], bytes[17]]);
        let payload_bits = u16::from_le_bytes([bytes[18], bytes[19]]);
        let header = Self {
            magic,
            version,
            flags,
            channel,
            kind,
            sequence,
            ack,
            ack_bits,
            tick,
            payload_bits,
        };
        header.validate()?;
        Ok(header)
    }

    /// Validate header invariants before payload decode.
    pub fn validate(&self) -> Result<(), PacketError> {
        if self.magic != NPS_MAGIC {
            return Err(PacketError::BadMagic);
        }
        if self.version != NPS_VERSION {
            return Err(PacketError::UnsupportedVersion(self.version));
        }
        Ok(())
    }
}

/// Borrowing parsed datagram view.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PacketView<'src> {
    /// Parsed wire header.
    pub header: PacketHeader,
    /// Borrowed packed payload bytes.
    pub payload: &'src [u8],
}

impl<'src> PacketView<'src> {
    /// Parse a datagram into header + borrowed payload view.
    pub fn parse(datagram: &'src [u8]) -> Result<Self, PacketError> {
        let header = PacketHeader::decode(datagram)?;
        let payload = datagram
            .get(NPS_HEADER_BYTES..)
            .ok_or(PacketError::TruncatedHeader)?;
        let payload_bytes_needed = (header.payload_bits as usize).div_ceil(8);
        if payload.len() < payload_bytes_needed {
            return Err(PacketError::TruncatedPayload);
        }
        Ok(Self {
            header,
            payload: &payload[..payload_bytes_needed],
        })
    }

    /// Open a bit reader constrained to the payload bit length declared in the header.
    pub fn payload_reader(&self) -> Result<BitReader<'src>, PacketError> {
        BitReader::with_bit_len(self.payload, self.header.payload_bits as usize)
            .map_err(PacketError::BitPack)
    }
}

/// Physics transform update packed for network transport.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TransformSample {
    /// Network object handle.
    pub object: u32,
    /// Authoritative owner peer id.
    pub owner_peer: u16,
    /// Quantized world position x.
    pub pos_x: u16,
    /// Quantized world position y.
    pub pos_y: u16,
    /// Quantized signed velocity x (`-1..=1` -> `u16`).
    pub vel_x: u16,
    /// Quantized signed velocity y (`-1..=1` -> `u16`).
    pub vel_y: u16,
    /// Flags reserved for sleep/awake/contact bits.
    pub flags: u8,
}

impl TransformSample {
    /// Quantize a world-space sample using the provided grid.
    pub fn quantize(
        object: u32,
        owner_peer: u16,
        x: f32,
        y: f32,
        vx_norm: f32,
        vy_norm: f32,
        flags: u8,
        grid: GridQuantization,
    ) -> Self {
        let (pos_x, pos_y) = grid.encode_position(x, y);
        let vel_x = grid.encode_signed_unit(vx_norm);
        let vel_y = grid.encode_signed_unit(vy_norm);
        Self {
            object,
            owner_peer,
            pos_x,
            pos_y,
            vel_x,
            vel_y,
            flags,
        }
    }
}

/// Packed batch of transform samples.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransformBatch<'a> {
    /// Borrowed transform sample slice.
    pub samples: &'a [TransformSample],
}

/// Deterministic input payload for one simulation tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InputFrame {
    /// Owning player/peer id.
    pub player_peer: u16,
    /// Simulation tick.
    pub tick: u32,
    /// Bit-packed button state.
    pub buttons: u16,
    /// Quantized move x axis (`-1..=1` -> `i8`).
    pub move_x_q: i8,
    /// Quantized move y axis (`-1..=1` -> `i8`).
    pub move_y_q: i8,
    /// Quantized aim x (`0..=1` -> `u16`).
    pub aim_x_q: u16,
    /// Quantized aim y (`0..=1` -> `u16`).
    pub aim_y_q: u16,
}

/// Reliable lifecycle event for long-lived gameplay state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LifecycleEvent {
    /// Event opcode (spawn, despawn, state transition, etc).
    pub opcode: u8,
    /// Event stream-local id.
    pub event_id: u16,
    /// Target object handle.
    pub object: u32,
    /// Small payload field (event-specific compact data).
    pub arg: u32,
}

/// Authority transfer message emitted on physgun-like ownership handoff.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AuthorityTransfer {
    /// Network object handle.
    pub object: u32,
    /// Previous owner peer id.
    pub previous_owner: u16,
    /// New owner peer id.
    pub new_owner: u16,
    /// Reason code (`1 = physgun_grab`, `2 = physgun_release`, etc).
    pub reason: u8,
}

/// Owned decoded payload value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodedPayload {
    TransformBatch(Vec<TransformSample>),
    InputFrame(InputFrame),
    LifecycleEvent(LifecycleEvent),
    AuthorityTransfer(AuthorityTransfer),
    AckOnly,
}

/// Packet encode/decode error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PacketError {
    TruncatedHeader,
    TruncatedPayload,
    BadMagic,
    UnsupportedVersion(u8),
    InvalidChannel,
    InvalidPayloadKind,
    InvalidPayloadFormat,
    BufferTooSmall,
    BitPack(BitPackError),
}

impl fmt::Display for PacketError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl std::error::Error for PacketError {}

impl From<BitPackError> for PacketError {
    fn from(value: BitPackError) -> Self {
        Self::BitPack(value)
    }
}

/// Encode a packet header + bit-packed payload into the provided datagram buffer.
pub fn encode_packet<F>(
    header: PacketHeader,
    datagram: &mut [u8],
    encode_payload: F,
) -> Result<usize, PacketError>
where
    F: FnOnce(&mut BitWriter<'_>) -> Result<(), PacketError>,
{
    if datagram.len() < NPS_HEADER_BYTES {
        return Err(PacketError::BufferTooSmall);
    }

    let (header_buf, payload_buf) = datagram.split_at_mut(NPS_HEADER_BYTES);
    let mut writer = BitWriter::new(payload_buf);
    encode_payload(&mut writer)?;
    let payload_bits = writer.bit_len() as u16;
    let payload_bytes = writer.byte_len();

    let header = PacketHeader {
        payload_bits,
        ..header
    };
    let header_bytes = header.encode();
    header_buf.copy_from_slice(&header_bytes);
    Ok(NPS_HEADER_BYTES + payload_bytes)
}

/// Encode a transform batch payload.
pub fn encode_transform_batch(
    writer: &mut BitWriter<'_>,
    batch: &TransformBatch<'_>,
) -> Result<(), PacketError> {
    let count = batch.samples.len();
    if count > u8::MAX as usize {
        return Err(PacketError::InvalidPayloadFormat);
    }
    writer.write_bits(count as u32, 8)?;
    for s in batch.samples {
        writer.write_bits(s.object, 32)?;
        writer.write_bits(u32::from(s.owner_peer), 16)?;
        writer.write_bits(u32::from(s.pos_x), 16)?;
        writer.write_bits(u32::from(s.pos_y), 16)?;
        writer.write_bits(u32::from(s.vel_x), 16)?;
        writer.write_bits(u32::from(s.vel_y), 16)?;
        writer.write_bits(u32::from(s.flags), 8)?;
    }
    Ok(())
}

/// Decode a transform batch payload into owned samples.
pub fn decode_transform_batch(view: &PacketView<'_>) -> Result<Vec<TransformSample>, PacketError> {
    let mut reader = view.payload_reader()?;
    let count = reader.read_bits(8)? as usize;
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        out.push(TransformSample {
            object: reader.read_bits(32)?,
            owner_peer: reader.read_bits(16)? as u16,
            pos_x: reader.read_bits(16)? as u16,
            pos_y: reader.read_bits(16)? as u16,
            vel_x: reader.read_bits(16)? as u16,
            vel_y: reader.read_bits(16)? as u16,
            flags: reader.read_bits(8)? as u8,
        });
    }
    Ok(out)
}

/// Encode an input frame payload.
pub fn encode_input_frame(
    writer: &mut BitWriter<'_>,
    input: InputFrame,
) -> Result<(), PacketError> {
    writer.write_bits(u32::from(input.player_peer), 16)?;
    writer.write_bits(input.tick, 32)?;
    writer.write_bits(u32::from(input.buttons), 16)?;
    writer.write_bits(input.move_x_q as u8 as u32, 8)?;
    writer.write_bits(input.move_y_q as u8 as u32, 8)?;
    writer.write_bits(u32::from(input.aim_x_q), 16)?;
    writer.write_bits(u32::from(input.aim_y_q), 16)?;
    Ok(())
}

/// Decode an input frame payload.
pub fn decode_input_frame(view: &PacketView<'_>) -> Result<InputFrame, PacketError> {
    let mut reader = view.payload_reader()?;
    Ok(InputFrame {
        player_peer: reader.read_bits(16)? as u16,
        tick: reader.read_bits(32)?,
        buttons: reader.read_bits(16)? as u16,
        move_x_q: reader.read_bits(8)? as u8 as i8,
        move_y_q: reader.read_bits(8)? as u8 as i8,
        aim_x_q: reader.read_bits(16)? as u16,
        aim_y_q: reader.read_bits(16)? as u16,
    })
}

/// Encode a lifecycle event payload.
pub fn encode_lifecycle_event(
    writer: &mut BitWriter<'_>,
    event: LifecycleEvent,
) -> Result<(), PacketError> {
    writer.write_bits(u32::from(event.opcode), 8)?;
    writer.write_bits(u32::from(event.event_id), 16)?;
    writer.write_bits(event.object, 32)?;
    writer.write_bits(event.arg, 32)?;
    Ok(())
}

/// Decode a lifecycle event payload.
pub fn decode_lifecycle_event(view: &PacketView<'_>) -> Result<LifecycleEvent, PacketError> {
    let mut reader = view.payload_reader()?;
    Ok(LifecycleEvent {
        opcode: reader.read_bits(8)? as u8,
        event_id: reader.read_bits(16)? as u16,
        object: reader.read_bits(32)?,
        arg: reader.read_bits(32)?,
    })
}

/// Encode an authority transfer payload.
pub fn encode_authority_transfer(
    writer: &mut BitWriter<'_>,
    transfer: AuthorityTransfer,
) -> Result<(), PacketError> {
    writer.write_bits(transfer.object, 32)?;
    writer.write_bits(u32::from(transfer.previous_owner), 16)?;
    writer.write_bits(u32::from(transfer.new_owner), 16)?;
    writer.write_bits(u32::from(transfer.reason), 8)?;
    Ok(())
}

/// Decode an authority transfer payload.
pub fn decode_authority_transfer(view: &PacketView<'_>) -> Result<AuthorityTransfer, PacketError> {
    let mut reader = view.payload_reader()?;
    Ok(AuthorityTransfer {
        object: reader.read_bits(32)?,
        previous_owner: reader.read_bits(16)? as u16,
        new_owner: reader.read_bits(16)? as u16,
        reason: reader.read_bits(8)? as u8,
    })
}

/// Decode a packet payload into an owned value for cross-thread handoff.
pub fn decode_payload_owned(view: &PacketView<'_>) -> Result<DecodedPayload, PacketError> {
    match view.header.kind {
        PayloadKind::TransformBatch => {
            decode_transform_batch(view).map(DecodedPayload::TransformBatch)
        }
        PayloadKind::InputFrame => decode_input_frame(view).map(DecodedPayload::InputFrame),
        PayloadKind::LifecycleEvent => {
            decode_lifecycle_event(view).map(DecodedPayload::LifecycleEvent)
        }
        PayloadKind::AuthorityTransfer => {
            decode_authority_transfer(view).map(DecodedPayload::AuthorityTransfer)
        }
        PayloadKind::AckOnly => Ok(DecodedPayload::AckOnly),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_roundtrip() {
        let header = PacketHeader {
            magic: NPS_MAGIC,
            version: NPS_VERSION,
            flags: PacketFlags::RELIABLE.union(PacketFlags::SEQUENCED),
            channel: PacketChannel::Lifecycle,
            kind: PayloadKind::LifecycleEvent,
            sequence: 17,
            ack: 13,
            ack_bits: 0xa5a5_5a5a,
            tick: 42,
            payload_bits: 96,
        };
        let encoded = header.encode();
        let decoded = PacketHeader::decode(&encoded).unwrap();
        assert_eq!(decoded, header);
    }

    #[test]
    fn transform_batch_roundtrip_uses_quantized_u16_positions() {
        let grid = GridQuantization::TILELINE_2048;
        let samples = vec![
            TransformSample::quantize(11, 2, 128.0, 64.0, 0.3, -0.25, 1, grid),
            TransformSample::quantize(12, 2, 2048.0, 2048.0, 1.0, 1.0, 0, grid),
        ];
        let mut datagram = [0u8; 256];
        let header = PacketHeader {
            magic: NPS_MAGIC,
            version: NPS_VERSION,
            flags: PacketFlags::SEQUENCED,
            channel: PacketChannel::Physics,
            kind: PayloadKind::TransformBatch,
            sequence: 1,
            ack: 0,
            ack_bits: 0,
            tick: 100,
            payload_bits: 0,
        };

        let len = encode_packet(header, &mut datagram, |w| {
            encode_transform_batch(w, &TransformBatch { samples: &samples })
        })
        .unwrap();

        let view = PacketView::parse(&datagram[..len]).unwrap();
        let decoded = decode_transform_batch(&view).unwrap();
        assert_eq!(decoded, samples);

        let (x, y) = grid.decode_position(decoded[0].pos_x, decoded[0].pos_y);
        assert!((x - 128.0).abs() < 1.0);
        assert!((y - 64.0).abs() < 1.0);
    }
}
