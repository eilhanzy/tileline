//! ParadoxPE physics foundations for Tileline.
//!
//! This crate provides the first engine-facing physics world skeleton used to prepare the
//! `.tlscript`/WASM host ABI and the future ParadoxPE runtime:
//! - typed opaque handles packed into 32-bit script-friendly IDs
//! - a fixed-step physics world with body/collider storage
//! - generic handle release for script/runtime integration
//! - a small host ABI surface that `.tlscript` can target without raw pointers

pub mod abi;
pub mod body;
pub mod handle;
pub mod world;

pub use abi::{
    ParadoxScriptHostAbi, ScriptBodyKind, ScriptColliderShapeKind, HOST_CALLS_ALLOWLIST,
    HOST_CALLS_HANDLE_ACQUIRE, HOST_CALLS_HANDLE_RELEASE, HOST_CALL_APPLY_FORCE,
    HOST_CALL_CONTACT_COUNT, HOST_CALL_CREATE_BODY, HOST_CALL_CREATE_COLLIDER,
    HOST_CALL_DESTROY_BODY, HOST_CALL_DESTROY_COLLIDER, HOST_CALL_QUERY_CONTACTS,
    HOST_CALL_RELEASE_HANDLE, HOST_CALL_SET_VELOCITY, HOST_CALL_SPAWN_BODY,
    HOST_CALL_SPAWN_COLLIDER, HOST_CALL_STEP_WORLD,
};
pub use body::{
    BodyDesc, BodyKind, ColliderDesc, ColliderShape, ColliderShapeKind, ContactPair,
    ContactSnapshot, RigidBody,
};
pub use handle::{BodyHandle, ColliderHandle, ContactHandle, HandleKind, PhysicsHandle};
pub use world::{FixedStepClock, PhysicsWorld, PhysicsWorldConfig};
