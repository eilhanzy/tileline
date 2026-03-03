//! ParadoxPE physics foundations for Tileline.
//!
//! This crate provides the first engine-facing physics world skeleton used to prepare the
//! `.tlscript`/WASM host ABI and the future ParadoxPE runtime:
//! - typed opaque handles packed into 32-bit script-friendly IDs
//! - a fixed-step physics world with SoA body/collider storage
//! - generic handle release for script/runtime integration
//! - a small host ABI surface that `.tlscript` can target without raw pointers

pub mod abi;
pub mod body;
pub mod broadphase;
pub mod handle;
pub mod joint;
pub mod narrowphase;
pub mod sleep;
pub mod snapshot;
pub mod solver;
pub mod storage;
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
    Aabb, BodyDesc, BodyKind, ColliderDesc, ColliderMaterial, ColliderShape, ColliderShapeKind,
    ContactId, ContactManifold, ContactPair, ContactSnapshot, MaterialCombineRule, RigidBody,
};
pub use broadphase::{BroadphaseConfig, BroadphasePipeline, BroadphaseStats};
pub use handle::{
    BodyHandle, ColliderHandle, ContactHandle, HandleKind, JointHandle, PhysicsHandle,
};
pub use joint::{
    DistanceJoint, DistanceJointDesc, FixedJoint, FixedJointDesc, JointConstraint,
    JointConstraintSolver, JointKind, JointSolverConfig, JointSolverStats,
};
pub use narrowphase::{NarrowphaseConfig, NarrowphasePipeline, NarrowphaseStats};
pub use sleep::{SleepConfig, SleepIslandManager, SleepStats};
pub use snapshot::{
    BodyStateFrame, InterpolatedBodyPose, PhysicsInterpolationBuffer, PhysicsSnapshot,
};
pub use solver::{ContactSolver, ContactSolverConfig, ContactSolverStats};
pub use storage::{BodyReadDomain, BodyRegistry, BodyVelocityWriteDomain};
pub use world::{FixedStepClock, PhysicsWorld, PhysicsWorldConfig};
