//! Fixed-step ParadoxPE world skeleton.
//!
//! This is intentionally conservative: it provides stable handles, deterministic stepping, and a
//! starter contact pipeline without pretending to be a finished solver.

use nalgebra::Vector2;

use crate::abi::ParadoxScriptHostAbi;
use crate::body::{
    BodyDesc, BodyKind, ColliderDesc, ColliderShape, ColliderShapeKind, ContactPair,
    ContactSnapshot, RigidBody,
};
use crate::handle::{BodyHandle, ColliderHandle, ContactHandle, HandleKind, PhysicsHandle};

#[derive(Debug, Clone)]
struct Slot<T> {
    generation: u16,
    value: Option<T>,
}

impl<T> Default for Slot<T> {
    fn default() -> Self {
        Self {
            generation: 1,
            value: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct ColliderRecord {
    handle: ColliderHandle,
    desc: ColliderDesc,
}

/// Fixed-step scheduler helper for engine frame times.
#[derive(Debug, Clone, PartialEq)]
pub struct FixedStepClock {
    fixed_dt: f32,
    max_substeps: u32,
    accumulator: f32,
    tick: u64,
}

impl FixedStepClock {
    pub fn new(fixed_dt: f32, max_substeps: u32) -> Self {
        Self {
            fixed_dt: fixed_dt.max(1e-4),
            max_substeps: max_substeps.max(1),
            accumulator: 0.0,
            tick: 0,
        }
    }

    pub fn fixed_dt(&self) -> f32 {
        self.fixed_dt
    }

    pub fn tick(&self) -> u64 {
        self.tick
    }

    pub fn accumulate(&mut self, dt: f32) -> u32 {
        self.accumulator =
            (self.accumulator + dt.max(0.0)).min(self.fixed_dt * self.max_substeps as f32);
        let mut steps = 0;
        while self.accumulator + f32::EPSILON >= self.fixed_dt && steps < self.max_substeps {
            self.accumulator -= self.fixed_dt;
            self.tick = self.tick.saturating_add(1);
            steps += 1;
        }
        steps
    }
}

/// World configuration for the starter ParadoxPE runtime.
#[derive(Debug, Clone, PartialEq)]
pub struct PhysicsWorldConfig {
    pub gravity: Vector2<f32>,
    pub fixed_dt: f32,
    pub max_substeps: u32,
    pub max_contact_snapshots: usize,
}

impl Default for PhysicsWorldConfig {
    fn default() -> Self {
        Self {
            gravity: Vector2::new(0.0, -9.81),
            fixed_dt: 1.0 / 120.0,
            max_substeps: 8,
            max_contact_snapshots: 256,
        }
    }
}

/// Engine-facing ParadoxPE world skeleton.
#[derive(Debug, Clone)]
pub struct PhysicsWorld {
    config: PhysicsWorldConfig,
    clock: FixedStepClock,
    bodies: Vec<Slot<RigidBody>>,
    free_bodies: Vec<u16>,
    colliders: Vec<Slot<ColliderRecord>>,
    free_colliders: Vec<u16>,
    contact_snapshots: Vec<Slot<ContactSnapshot>>,
    free_contact_snapshots: Vec<u16>,
    contacts: Vec<ContactPair>,
}

impl PhysicsWorld {
    pub fn new(config: PhysicsWorldConfig) -> Self {
        Self {
            clock: FixedStepClock::new(config.fixed_dt, config.max_substeps),
            config,
            bodies: Vec::new(),
            free_bodies: Vec::new(),
            colliders: Vec::new(),
            free_colliders: Vec::new(),
            contact_snapshots: Vec::new(),
            free_contact_snapshots: Vec::new(),
            contacts: Vec::new(),
        }
    }

    pub fn config(&self) -> &PhysicsWorldConfig {
        &self.config
    }

    pub fn fixed_step_clock(&self) -> &FixedStepClock {
        &self.clock
    }

    pub fn spawn_body(&mut self, desc: BodyDesc) -> BodyHandle {
        let (index, generation, slot) = alloc_slot(&mut self.bodies, &mut self.free_bodies);
        let handle = BodyHandle::new(index, generation);
        *slot = Some(RigidBody::from_desc(handle, desc));
        handle
    }

    pub fn body(&self, handle: BodyHandle) -> Option<&RigidBody> {
        get_slot(&self.bodies, handle.erased()).and_then(|slot| slot.value.as_ref())
    }

    pub fn body_mut(&mut self, handle: BodyHandle) -> Option<&mut RigidBody> {
        get_slot_mut(&mut self.bodies, handle.erased()).and_then(|slot| slot.value.as_mut())
    }

    pub fn destroy_body(&mut self, handle: BodyHandle) -> bool {
        let removed = free_slot(&mut self.bodies, &mut self.free_bodies, handle.erased()).is_some();
        if removed {
            for idx in 0..self.colliders.len() {
                let erase = if let Some(record) = self.colliders[idx].value.as_ref() {
                    record.desc.body == Some(handle)
                } else {
                    false
                };
                if erase {
                    let collider_handle =
                        ColliderHandle::new(idx as u16, self.colliders[idx].generation);
                    let _ = self.destroy_collider(collider_handle);
                }
            }
        }
        removed
    }

    pub fn spawn_collider(&mut self, desc: ColliderDesc) -> Option<ColliderHandle> {
        if let Some(body) = desc.body {
            self.body(body)?;
        }
        let (index, generation, slot) = alloc_slot(&mut self.colliders, &mut self.free_colliders);
        let handle = ColliderHandle::new(index, generation);
        *slot = Some(ColliderRecord { handle, desc });
        Some(handle)
    }

    pub fn destroy_collider(&mut self, handle: ColliderHandle) -> bool {
        free_slot(
            &mut self.colliders,
            &mut self.free_colliders,
            handle.erased(),
        )
        .is_some()
    }

    pub fn apply_force(&mut self, body: BodyHandle, force: Vector2<f32>) -> bool {
        let Some(body) = self.body_mut(body) else {
            return false;
        };
        if body.kind != BodyKind::Dynamic {
            return false;
        }
        body.accumulated_force += force;
        body.awake = true;
        true
    }

    pub fn set_velocity(&mut self, body: BodyHandle, velocity: Vector2<f32>) -> bool {
        let Some(body) = self.body_mut(body) else {
            return false;
        };
        if body.kind == BodyKind::Static {
            return false;
        }
        body.velocity = velocity;
        body.awake = true;
        true
    }

    pub fn contacts(&self) -> &[ContactPair] {
        &self.contacts
    }

    pub fn query_contacts(&mut self, body: BodyHandle) -> Option<ContactHandle> {
        self.body(body)?;
        let contacts = self
            .contacts
            .iter()
            .filter(|contact| contact.body_a == Some(body) || contact.body_b == Some(body))
            .cloned()
            .collect::<Vec<_>>();
        if self.contact_snapshots.len() >= self.config.max_contact_snapshots
            && self.free_contact_snapshots.is_empty()
        {
            return None;
        }
        let (index, generation, slot) = alloc_slot(
            &mut self.contact_snapshots,
            &mut self.free_contact_snapshots,
        );
        let handle = ContactHandle::new(index, generation);
        *slot = Some(ContactSnapshot { body, contacts });
        Some(handle)
    }

    pub fn contact_snapshot(&self, handle: ContactHandle) -> Option<&ContactSnapshot> {
        get_slot(&self.contact_snapshots, handle.erased()).and_then(|slot| slot.value.as_ref())
    }

    pub fn contact_count(&self, handle: ContactHandle) -> u32 {
        self.contact_snapshot(handle)
            .map(|snapshot| snapshot.contacts.len() as u32)
            .unwrap_or(0)
    }

    pub fn release_handle(&mut self, handle: PhysicsHandle) -> bool {
        match handle.kind() {
            Some(HandleKind::Body) => BodyHandle::try_from(handle)
                .ok()
                .map(|h| self.destroy_body(h))
                .unwrap_or(false),
            Some(HandleKind::Collider) => ColliderHandle::try_from(handle)
                .ok()
                .map(|h| self.destroy_collider(h))
                .unwrap_or(false),
            Some(HandleKind::ContactSnapshot) => free_slot(
                &mut self.contact_snapshots,
                &mut self.free_contact_snapshots,
                handle,
            )
            .is_some(),
            None => false,
        }
    }

    pub fn step(&mut self, dt: f32) -> u32 {
        let substeps = self.clock.accumulate(dt);
        let fixed_dt = self.clock.fixed_dt();
        for _ in 0..substeps {
            self.integrate_bodies(fixed_dt);
            self.rebuild_contacts();
        }
        substeps
    }

    fn integrate_bodies(&mut self, dt: f32) {
        for slot in &mut self.bodies {
            let Some(body) = slot.value.as_mut() else {
                continue;
            };
            match body.kind {
                BodyKind::Static => continue,
                BodyKind::Kinematic => {
                    body.position += body.velocity * dt;
                }
                BodyKind::Dynamic => {
                    let acceleration =
                        self.config.gravity + body.accumulated_force * body.inverse_mass;
                    body.velocity += acceleration * dt;
                    body.velocity *= 1.0 - body.linear_damping.clamp(0.0, 0.95);
                    body.position += body.velocity * dt;
                    body.accumulated_force = Vector2::zeros();
                }
            }
        }
    }

    fn rebuild_contacts(&mut self) {
        self.contacts.clear();

        let colliders = self
            .colliders
            .iter()
            .filter_map(|slot| slot.value.as_ref())
            .collect::<Vec<_>>();

        for i in 0..colliders.len() {
            for j in (i + 1)..colliders.len() {
                let a = colliders[i];
                let b = colliders[j];
                if let Some(contact) = self.compute_contact(a, b) {
                    self.contacts.push(contact);
                }
            }
        }
    }

    fn compute_contact(&self, a: &ColliderRecord, b: &ColliderRecord) -> Option<ContactPair> {
        let center_a = self.collider_center(a)?;
        let center_b = self.collider_center(b)?;
        let extents_a = a.desc.shape.half_extents();
        let extents_b = b.desc.shape.half_extents();
        let delta = center_b - center_a;
        let overlap_x = extents_a.x + extents_b.x - delta.x.abs();
        let overlap_y = extents_a.y + extents_b.y - delta.y.abs();
        if overlap_x <= 0.0 || overlap_y <= 0.0 {
            return None;
        }

        let (penetration, normal) = if overlap_x < overlap_y {
            (
                overlap_x,
                Vector2::new(if delta.x >= 0.0 { 1.0 } else { -1.0 }, 0.0),
            )
        } else {
            (
                overlap_y,
                Vector2::new(0.0, if delta.y >= 0.0 { 1.0 } else { -1.0 }),
            )
        };

        Some(ContactPair {
            collider_a: a.handle,
            collider_b: b.handle,
            body_a: a.desc.body,
            body_b: b.desc.body,
            normal,
            penetration,
        })
    }

    fn collider_center(&self, collider: &ColliderRecord) -> Option<Vector2<f32>> {
        let body = collider.desc.body?;
        Some(self.body(body)?.position)
    }
}

impl ParadoxScriptHostAbi for PhysicsWorld {
    fn spawn_body(&mut self, kind: u32, x: f32, y: f32, mass: f32) -> u32 {
        let Some(kind) = BodyKind::from_u32(kind) else {
            return 0;
        };
        PhysicsWorld::spawn_body(
            self,
            BodyDesc {
                kind,
                position: Vector2::new(x, y),
                mass,
                ..BodyDesc::default()
            },
        )
        .raw()
    }

    fn spawn_collider(&mut self, body: u32, shape: u32, a: f32, b: f32) -> u32 {
        let Ok(body) = BodyHandle::try_from(PhysicsHandle::from(body)) else {
            return 0;
        };
        let Some(shape) = ColliderShapeKind::from_u32(shape) else {
            return 0;
        };
        PhysicsWorld::spawn_collider(
            self,
            ColliderDesc::attached(body, ColliderShape::from_script(shape, a, b)),
        )
        .map(ColliderHandle::raw)
        .unwrap_or(0)
    }

    fn release_handle(&mut self, handle: u32) -> bool {
        PhysicsWorld::release_handle(self, PhysicsHandle::from(handle))
    }

    fn apply_force(&mut self, body: u32, x: f32, y: f32) -> bool {
        let Ok(body) = BodyHandle::try_from(PhysicsHandle::from(body)) else {
            return false;
        };
        PhysicsWorld::apply_force(self, body, Vector2::new(x, y))
    }

    fn set_velocity(&mut self, body: u32, x: f32, y: f32) -> bool {
        let Ok(body) = BodyHandle::try_from(PhysicsHandle::from(body)) else {
            return false;
        };
        PhysicsWorld::set_velocity(self, body, Vector2::new(x, y))
    }

    fn query_contacts(&mut self, body: u32) -> u32 {
        let Ok(body) = BodyHandle::try_from(PhysicsHandle::from(body)) else {
            return 0;
        };
        PhysicsWorld::query_contacts(self, body)
            .map(ContactHandle::raw)
            .unwrap_or(0)
    }

    fn contact_count(&self, contacts: u32) -> u32 {
        let Ok(contacts) = ContactHandle::try_from(PhysicsHandle::from(contacts)) else {
            return 0;
        };
        PhysicsWorld::contact_count(self, contacts)
    }

    fn step_world(&mut self, dt: f32) -> u32 {
        PhysicsWorld::step(self, dt)
    }
}

fn alloc_slot<'a, T>(
    slots: &'a mut Vec<Slot<T>>,
    free_list: &mut Vec<u16>,
) -> (u16, u16, &'a mut Option<T>) {
    if let Some(index) = free_list.pop() {
        let slot = &mut slots[index as usize];
        let generation = slot.generation;
        return (index, generation, &mut slot.value);
    }
    let index = slots.len() as u16;
    slots.push(Slot::default());
    let slot = &mut slots[index as usize];
    (index, slot.generation, &mut slot.value)
}

fn get_slot<T>(slots: &[Slot<T>], handle: PhysicsHandle) -> Option<&Slot<T>> {
    let slot = slots.get(handle.index())?;
    if slot.generation == handle.generation() && slot.value.is_some() {
        Some(slot)
    } else {
        None
    }
}

fn get_slot_mut<T>(slots: &mut [Slot<T>], handle: PhysicsHandle) -> Option<&mut Slot<T>> {
    let slot = slots.get_mut(handle.index())?;
    if slot.generation == handle.generation() && slot.value.is_some() {
        Some(slot)
    } else {
        None
    }
}

fn free_slot<T>(
    slots: &mut [Slot<T>],
    free_list: &mut Vec<u16>,
    handle: PhysicsHandle,
) -> Option<T> {
    let slot = slots.get_mut(handle.index())?;
    if slot.generation != handle.generation() {
        return None;
    }
    let value = slot.value.take()?;
    slot.generation = slot.generation.wrapping_add(1).max(1);
    free_list.push(handle.index() as u16);
    Some(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::body::{ColliderDesc, ColliderShape};

    #[test]
    fn dynamic_body_integrates_with_force() {
        let mut world = PhysicsWorld::new(PhysicsWorldConfig::default());
        let body = world.spawn_body(BodyDesc::default());
        assert!(world.apply_force(body, Vector2::new(10.0, 0.0)));
        let steps = world.step(1.0 / 60.0);
        assert!(steps > 0);
        let body = world.body(body).unwrap();
        assert!(body.position.x > 0.0);
    }

    #[test]
    fn overlapping_colliders_produce_contact_snapshot() {
        let mut world = PhysicsWorld::new(PhysicsWorldConfig::default());
        let a = world.spawn_body(BodyDesc {
            kind: BodyKind::Static,
            position: Vector2::new(0.0, 0.0),
            ..BodyDesc::default()
        });
        let b = world.spawn_body(BodyDesc {
            kind: BodyKind::Static,
            position: Vector2::new(0.5, 0.0),
            ..BodyDesc::default()
        });
        world
            .spawn_collider(ColliderDesc::attached(
                a,
                ColliderShape::Aabb {
                    half_extents: Vector2::new(1.0, 1.0),
                },
            ))
            .unwrap();
        world
            .spawn_collider(ColliderDesc::attached(
                b,
                ColliderShape::Aabb {
                    half_extents: Vector2::new(1.0, 1.0),
                },
            ))
            .unwrap();
        world.step(1.0 / 60.0);
        let snapshot = world.query_contacts(a).unwrap();
        assert_eq!(world.contact_count(snapshot), 1);
        assert!(world.release_handle(snapshot.erased()));
    }

    #[test]
    fn releasing_body_invalidates_stale_handle() {
        let mut world = PhysicsWorld::new(PhysicsWorldConfig::default());
        let body = world.spawn_body(BodyDesc::default());
        assert!(world.release_handle(body.erased()));
        assert!(world.body(body).is_none());
        assert!(!world.release_handle(body.erased()));
    }
}
