//! Opaque generational handles for ParadoxPE resources.
//!
//! Handles are packed into a 32-bit integer so they can move cleanly through `.tlscript`, WASM MVP,
//! and engine/network layers without exposing raw pointers.

use std::fmt;

const INDEX_BITS: u32 = 16;
const INDEX_MASK: u32 = (1 << INDEX_BITS) - 1;
const GENERATION_BITS: u32 = 14;
const GENERATION_MASK: u32 = (1 << GENERATION_BITS) - 1;
const KIND_SHIFT: u32 = 30;
const GENERATION_SHIFT: u32 = INDEX_BITS;

/// High-level ParadoxPE handle kind encoded into the opaque 32-bit token.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum HandleKind {
    ContactSnapshot = 0,
    Body = 1,
    Collider = 2,
    Joint = 3,
}

impl HandleKind {
    fn from_bits(bits: u32) -> Option<Self> {
        match bits {
            0 => Some(Self::ContactSnapshot),
            1 => Some(Self::Body),
            2 => Some(Self::Collider),
            3 => Some(Self::Joint),
            _ => None,
        }
    }
}

/// Generic script/runtime-visible ParadoxPE handle.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct PhysicsHandle(u32);

impl PhysicsHandle {
    /// Creates a packed handle from a kind, slot index, and generation.
    pub fn new(kind: HandleKind, index: u16, generation: u16) -> Self {
        let raw = ((kind as u32) << KIND_SHIFT)
            | (((generation as u32) & GENERATION_MASK) << GENERATION_SHIFT)
            | ((index as u32) & INDEX_MASK);
        Self(raw)
    }

    /// Raw packed handle bits, suitable for `.tlscript`/WASM `i32` transport.
    pub fn raw(self) -> u32 {
        self.0
    }

    /// Decoded resource kind.
    pub fn kind(self) -> Option<HandleKind> {
        HandleKind::from_bits(self.0 >> KIND_SHIFT)
    }

    /// Slot index within the resource storage arena.
    pub fn index(self) -> usize {
        (self.0 & INDEX_MASK) as usize
    }

    /// Generation counter for stale-handle rejection.
    pub fn generation(self) -> u16 {
        ((self.0 >> GENERATION_SHIFT) & GENERATION_MASK) as u16
    }
}

impl fmt::Debug for PhysicsHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PhysicsHandle")
            .field("raw", &self.0)
            .field("kind", &self.kind())
            .field("index", &self.index())
            .field("generation", &self.generation())
            .finish()
    }
}

impl From<u32> for PhysicsHandle {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

impl From<PhysicsHandle> for u32 {
    fn from(value: PhysicsHandle) -> Self {
        value.raw()
    }
}

macro_rules! define_typed_handle {
    ($name:ident, $kind:expr) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub struct $name(PhysicsHandle);

        impl $name {
            pub fn new(index: u16, generation: u16) -> Self {
                Self(PhysicsHandle::new($kind, index, generation))
            }

            pub fn raw(self) -> u32 {
                self.0.raw()
            }

            pub fn index(self) -> usize {
                self.0.index()
            }

            pub fn generation(self) -> u16 {
                self.0.generation()
            }

            pub fn erased(self) -> PhysicsHandle {
                self.0
            }
        }

        impl TryFrom<PhysicsHandle> for $name {
            type Error = ();

            fn try_from(value: PhysicsHandle) -> Result<Self, Self::Error> {
                if value.kind() == Some($kind) {
                    Ok(Self(value))
                } else {
                    Err(())
                }
            }
        }

        impl From<$name> for PhysicsHandle {
            fn from(value: $name) -> Self {
                value.0
            }
        }
    };
}

define_typed_handle!(BodyHandle, HandleKind::Body);
define_typed_handle!(ColliderHandle, HandleKind::Collider);
define_typed_handle!(JointHandle, HandleKind::Joint);
define_typed_handle!(ContactHandle, HandleKind::ContactSnapshot);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_handle_roundtrips_kind_index_generation() {
        let handle = BodyHandle::new(17, 9);
        let erased = handle.erased();
        assert_eq!(erased.kind(), Some(HandleKind::Body));
        assert_eq!(erased.index(), 17);
        assert_eq!(erased.generation(), 9);
        assert_eq!(BodyHandle::try_from(erased).unwrap().raw(), handle.raw());
    }
}
