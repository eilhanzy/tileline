//! `.tlscript`/WASM host ABI surface for ParadoxPE.
//!
//! The ABI is intentionally pointer-free and MVP-friendly: handles are packed into `u32` tokens,
//! vectors are split into scalar components, and world stepping returns plain counters.

pub use crate::body::{ScriptBodyKind, ScriptColliderShapeKind};

pub const HOST_CALL_SPAWN_BODY: &str = "spawn_body";
pub const HOST_CALL_CREATE_BODY: &str = "create_body";
pub const HOST_CALL_SPAWN_COLLIDER: &str = "spawn_collider";
pub const HOST_CALL_CREATE_COLLIDER: &str = "create_collider";
pub const HOST_CALL_RELEASE_HANDLE: &str = "release_handle";
pub const HOST_CALL_DESTROY_BODY: &str = "destroy_body";
pub const HOST_CALL_DESTROY_COLLIDER: &str = "destroy_collider";
pub const HOST_CALL_APPLY_FORCE: &str = "apply_force";
pub const HOST_CALL_SET_VELOCITY: &str = "set_velocity";
pub const HOST_CALL_QUERY_CONTACTS: &str = "query_contacts";
pub const HOST_CALL_CONTACT_COUNT: &str = "contact_count";
pub const HOST_CALL_STEP_WORLD: &str = "step_world";

pub const HOST_CALLS_HANDLE_ACQUIRE: &[&str] = &[
    HOST_CALL_SPAWN_BODY,
    HOST_CALL_CREATE_BODY,
    HOST_CALL_SPAWN_COLLIDER,
    HOST_CALL_CREATE_COLLIDER,
    HOST_CALL_QUERY_CONTACTS,
];

pub const HOST_CALLS_HANDLE_RELEASE: &[&str] = &[
    HOST_CALL_RELEASE_HANDLE,
    HOST_CALL_DESTROY_BODY,
    HOST_CALL_DESTROY_COLLIDER,
];

pub const HOST_CALLS_ALLOWLIST: &[&str] = &[
    HOST_CALL_APPLY_FORCE,
    HOST_CALL_SET_VELOCITY,
    HOST_CALL_CONTACT_COUNT,
    HOST_CALL_STEP_WORLD,
];

/// Script/WASM-facing ParadoxPE host interface.
///
/// Engines can expose this ABI directly to Wasmer/`wit-bindgen` shims or adapt it into richer
/// physics services internally.
pub trait ParadoxScriptHostAbi {
    fn spawn_body(&mut self, kind: u32, x: f32, y: f32, mass: f32) -> u32;
    fn spawn_collider(&mut self, body: u32, shape: u32, a: f32, b: f32) -> u32;
    fn release_handle(&mut self, handle: u32) -> bool;
    fn apply_force(&mut self, body: u32, x: f32, y: f32) -> bool;
    fn set_velocity(&mut self, body: u32, x: f32, y: f32) -> bool;
    fn query_contacts(&mut self, body: u32) -> u32;
    fn contact_count(&self, contacts: u32) -> u32;
    fn step_world(&mut self, dt: f32) -> u32;
}
