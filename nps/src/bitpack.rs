//! Zero-copy friendly bit packing primitives for NPS UDP payloads.
//!
//! The writer emits into a caller-provided byte slice and never allocates. The reader borrows the
//! original datagram slice and decodes bits directly from memory.

use core::fmt;

/// Errors emitted by bit-level encoding/decoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BitPackError {
    /// The destination buffer is too small for the requested write.
    BufferTooSmall,
    /// A read tried to consume more bits than remain in the source slice.
    UnexpectedEof,
    /// A bit width outside the supported `1..=32` range was requested.
    InvalidBitWidth,
}

impl fmt::Display for BitPackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl std::error::Error for BitPackError {}

/// Bit-level writer that packs fields densely into a caller-owned buffer.
pub struct BitWriter<'buf> {
    buf: &'buf mut [u8],
    bit_pos: usize,
}

impl<'buf> BitWriter<'buf> {
    /// Create a new writer that zeroes the destination slice before use.
    pub fn new(buf: &'buf mut [u8]) -> Self {
        buf.fill(0);
        Self { buf, bit_pos: 0 }
    }

    /// Total bits written so far.
    #[inline]
    pub fn bit_len(&self) -> usize {
        self.bit_pos
    }

    /// Total bytes touched by written bits, rounded up.
    #[inline]
    pub fn byte_len(&self) -> usize {
        self.bit_pos.div_ceil(8)
    }

    /// Remaining writable bits in the destination buffer.
    #[inline]
    pub fn remaining_bits(&self) -> usize {
        self.buf
            .len()
            .saturating_mul(8)
            .saturating_sub(self.bit_pos)
    }

    /// Write a raw bit field (`1..=32` bits) in least-significant-bit order.
    pub fn write_bits(&mut self, mut value: u32, bits: u8) -> Result<(), BitPackError> {
        if bits == 0 || bits > 32 {
            return Err(BitPackError::InvalidBitWidth);
        }
        if self.remaining_bits() < bits as usize {
            return Err(BitPackError::BufferTooSmall);
        }

        let mut remaining = bits;
        while remaining > 0 {
            let byte_idx = self.bit_pos / 8;
            let bit_idx = (self.bit_pos % 8) as u8;
            let free_bits = 8 - bit_idx;
            let take = free_bits.min(remaining);
            let mask = ((1u32 << take) - 1) as u8;
            let chunk = (value as u8) & mask;
            self.buf[byte_idx] |= chunk << bit_idx;

            self.bit_pos += take as usize;
            value >>= take;
            remaining -= take;
        }

        Ok(())
    }

    /// Write a boolean field as one bit.
    #[inline]
    pub fn write_bool(&mut self, value: bool) -> Result<(), BitPackError> {
        self.write_bits(u32::from(value), 1)
    }

    /// Write a byte-aligned byte sequence. If the writer is not aligned, bytes are still packed.
    pub fn write_bytes(&mut self, bytes: &[u8]) -> Result<(), BitPackError> {
        for &b in bytes {
            self.write_bits(u32::from(b), 8)?;
        }
        Ok(())
    }
}

/// Bit-level reader borrowing a source slice.
pub struct BitReader<'src> {
    buf: &'src [u8],
    bit_pos: usize,
    bit_len: usize,
}

impl<'src> BitReader<'src> {
    /// Construct a reader over an entire byte slice.
    pub fn new(buf: &'src [u8]) -> Self {
        Self {
            buf,
            bit_pos: 0,
            bit_len: buf.len() * 8,
        }
    }

    /// Construct a reader with an explicit bit length.
    pub fn with_bit_len(buf: &'src [u8], bit_len: usize) -> Result<Self, BitPackError> {
        if bit_len > buf.len() * 8 {
            return Err(BitPackError::UnexpectedEof);
        }
        Ok(Self {
            buf,
            bit_pos: 0,
            bit_len,
        })
    }

    /// Remaining unread bits.
    #[inline]
    pub fn remaining_bits(&self) -> usize {
        self.bit_len.saturating_sub(self.bit_pos)
    }

    /// Read a raw bit field (`1..=32` bits) in least-significant-bit order.
    pub fn read_bits(&mut self, bits: u8) -> Result<u32, BitPackError> {
        if bits == 0 || bits > 32 {
            return Err(BitPackError::InvalidBitWidth);
        }
        if self.remaining_bits() < bits as usize {
            return Err(BitPackError::UnexpectedEof);
        }

        let mut out = 0u32;
        let mut written = 0u8;
        let mut remaining = bits;

        while remaining > 0 {
            let byte_idx = self.bit_pos / 8;
            let bit_idx = (self.bit_pos % 8) as u8;
            let available = 8 - bit_idx;
            let take = available.min(remaining);
            let mask = ((1u16 << take) - 1) as u8;
            let chunk = (self.buf[byte_idx] >> bit_idx) & mask;
            out |= u32::from(chunk) << written;

            self.bit_pos += take as usize;
            written += take;
            remaining -= take;
        }

        Ok(out)
    }

    /// Read a boolean bit.
    #[inline]
    pub fn read_bool(&mut self) -> Result<bool, BitPackError> {
        Ok(self.read_bits(1)? != 0)
    }

    /// Read bytes into a caller-provided destination buffer.
    pub fn read_bytes(&mut self, out: &mut [u8]) -> Result<(), BitPackError> {
        for byte in out {
            *byte = self.read_bits(8)? as u8;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bitpack_roundtrip_crosses_byte_boundaries() {
        let mut buf = [0u8; 8];
        let bit_len = {
            let mut writer = BitWriter::new(&mut buf);
            writer.write_bits(0b101, 3).unwrap();
            writer.write_bits(0x1ff, 9).unwrap();
            writer.write_bool(true).unwrap();
            writer.write_bits(0xdead, 16).unwrap();
            writer.bit_len()
        };

        let mut reader = BitReader::with_bit_len(&buf, bit_len).unwrap();
        assert_eq!(reader.read_bits(3).unwrap(), 0b101);
        assert_eq!(reader.read_bits(9).unwrap(), 0x1ff);
        assert!(reader.read_bool().unwrap());
        assert_eq!(reader.read_bits(16).unwrap(), 0xdead);
        assert_eq!(reader.remaining_bits(), 0);
    }
}
