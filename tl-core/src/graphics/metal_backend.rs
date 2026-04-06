//! macOS raw Metal backend MVP for Tileline.
//!
//! The v0.5.0 scope is intentionally constrained:
//! - single GPU only
//! - explicit command queue submission
//! - CPU-side snapshot ring with strict capacity guards
//! - fail-soft telemetry surface for runtime integration

#![cfg(target_os = "macos")]

use std::error::Error;
use std::fmt::{Display, Formatter};
use std::sync::Arc;

use metal::{CommandQueue, Device};
use winit::dpi::PhysicalSize;
use winit::window::Window;

use crate::graphics::frame_snapshot::RenderStateSnapshot;

/// Runtime configuration for the raw Metal backend.
#[derive(Debug, Clone)]
pub struct MetalBackendConfig {
    /// Number of ring slots kept in-flight for snapshot uploads.
    pub frames_in_flight: usize,
    /// Maximum transform records accepted per frame.
    pub max_instances: usize,
}

impl Default for MetalBackendConfig {
    fn default() -> Self {
        Self {
            frames_in_flight: 2,
            max_instances: 32_768,
        }
    }
}

/// Snapshot metadata exposed for debugging / telemetry.
#[derive(Debug, Clone, Copy, Default)]
pub struct MetalSnapshotSlotState {
    pub frame_id: u64,
    pub instance_count: u32,
    pub material_count: u32,
    pub texture_count: u32,
    pub light_count: u32,
    pub byte_len: usize,
}

/// Submit telemetry for one recorded frame.
#[derive(Debug, Clone, Copy, Default)]
pub struct MetalFrameSubmissionTelemetry {
    pub frame_slot: usize,
    pub frame_id: u64,
    pub instance_count: u32,
    pub material_count: u32,
    pub texture_count: u32,
    pub light_count: u32,
}

/// Per-frame command queue execution telemetry.
#[derive(Debug, Clone, Copy, Default)]
pub struct MetalFrameExecutionTelemetry {
    pub submission: MetalFrameSubmissionTelemetry,
    pub snapshot_state: MetalSnapshotSlotState,
    pub command_buffer_submitted: bool,
    pub presented: bool,
    pub primary_submission_serial: u64,
}

#[derive(Debug)]
struct SnapshotSlot {
    instance_capacity: usize,
    material_capacity: usize,
    texture_capacity: usize,
    light_capacity: usize,
    last_state: MetalSnapshotSlotState,
}

/// Errors produced by the raw Metal backend.
#[derive(Debug)]
pub enum MetalBackendError {
    NoMetalDevice,
    InvalidConfig(&'static str),
    SnapshotCapacityExceeded { requested: usize, capacity: usize },
    SnapshotMaterialCapacityExceeded { requested: usize, capacity: usize },
    SnapshotTextureCapacityExceeded { requested: usize, capacity: usize },
    SnapshotLightCapacityExceeded { requested: usize, capacity: usize },
}

impl Display for MetalBackendError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoMetalDevice => write!(f, "no Metal system device found"),
            Self::InvalidConfig(message) => write!(f, "invalid Metal backend config: {message}"),
            Self::SnapshotCapacityExceeded {
                requested,
                capacity,
            } => write!(
                f,
                "snapshot capacity exceeded: requested {requested} instances, capacity {capacity}"
            ),
            Self::SnapshotMaterialCapacityExceeded {
                requested,
                capacity,
            } => write!(
                f,
                "snapshot material capacity exceeded: requested {requested} materials, capacity {capacity}"
            ),
            Self::SnapshotTextureCapacityExceeded {
                requested,
                capacity,
            } => write!(
                f,
                "snapshot texture capacity exceeded: requested {requested} textures, capacity {capacity}"
            ),
            Self::SnapshotLightCapacityExceeded {
                requested,
                capacity,
            } => write!(
                f,
                "snapshot light capacity exceeded: requested {requested} lights, capacity {capacity}"
            ),
        }
    }
}

impl Error for MetalBackendError {}

/// Raw Metal backend MVP used by runtime integration layers.
pub struct MetalBackend {
    config: MetalBackendConfig,
    _window: Arc<Window>,
    _device: Device,
    command_queue: CommandQueue,
    device_name: String,
    frame_slots: Vec<SnapshotSlot>,
    current_frame_slot: usize,
    surface_size: PhysicalSize<u32>,
    primary_submission_serial: u64,
}

impl MetalBackend {
    /// Create a new raw Metal backend bound to the provided window.
    pub fn new(window: Arc<Window>, config: MetalBackendConfig) -> Result<Self, MetalBackendError> {
        if config.frames_in_flight == 0 {
            return Err(MetalBackendError::InvalidConfig(
                "frames_in_flight must be greater than zero",
            ));
        }
        if config.max_instances == 0 {
            return Err(MetalBackendError::InvalidConfig(
                "max_instances must be greater than zero",
            ));
        }

        let device = Device::system_default().ok_or(MetalBackendError::NoMetalDevice)?;
        let command_queue = device.new_command_queue();
        let device_name = device.name().to_owned();

        let max_materials = config.max_instances.max(256);
        let max_textures = config.max_instances.max(128);
        let max_lights = 64;
        let frame_slots = (0..config.frames_in_flight)
            .map(|_| SnapshotSlot {
                instance_capacity: config.max_instances,
                material_capacity: max_materials,
                texture_capacity: max_textures,
                light_capacity: max_lights,
                last_state: MetalSnapshotSlotState::default(),
            })
            .collect::<Vec<_>>();

        Ok(Self {
            config,
            _window: Arc::clone(&window),
            _device: device,
            command_queue,
            device_name,
            frame_slots,
            current_frame_slot: 0,
            surface_size: window.inner_size(),
            primary_submission_serial: 0,
        })
    }

    /// Report the active Metal device name.
    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// Current frame extent in pixels.
    pub fn surface_size(&self) -> PhysicalSize<u32> {
        self.surface_size
    }

    /// Active backend config snapshot.
    pub fn config(&self) -> &MetalBackendConfig {
        &self.config
    }

    /// Update current surface size (resource realloc is intentionally deferred in MVP).
    pub fn resize(&mut self, new_size: PhysicalSize<u32>) -> Result<(), MetalBackendError> {
        if new_size.width == 0 || new_size.height == 0 {
            return Ok(());
        }
        self.surface_size = new_size;
        Ok(())
    }

    /// Write one render-visible snapshot into the selected frame slot.
    pub fn upload_state_snapshot(
        &mut self,
        frame_slot: usize,
        snapshot: RenderStateSnapshot<'_>,
    ) -> Result<MetalSnapshotSlotState, MetalBackendError> {
        let slot = self
            .frame_slots
            .get_mut(frame_slot)
            .ok_or(MetalBackendError::InvalidConfig("frame slot out of range"))?;

        if snapshot.transforms.len() > slot.instance_capacity {
            return Err(MetalBackendError::SnapshotCapacityExceeded {
                requested: snapshot.transforms.len(),
                capacity: slot.instance_capacity,
            });
        }
        if snapshot.materials.len() > slot.material_capacity {
            return Err(MetalBackendError::SnapshotMaterialCapacityExceeded {
                requested: snapshot.materials.len(),
                capacity: slot.material_capacity,
            });
        }
        if snapshot.textures.len() > slot.texture_capacity {
            return Err(MetalBackendError::SnapshotTextureCapacityExceeded {
                requested: snapshot.textures.len(),
                capacity: slot.texture_capacity,
            });
        }
        if snapshot.lights.len() > slot.light_capacity {
            return Err(MetalBackendError::SnapshotLightCapacityExceeded {
                requested: snapshot.lights.len(),
                capacity: slot.light_capacity,
            });
        }

        let byte_len = std::mem::size_of_val(snapshot.transforms)
            + std::mem::size_of_val(snapshot.materials)
            + std::mem::size_of_val(snapshot.textures)
            + std::mem::size_of_val(snapshot.lights);

        slot.last_state = MetalSnapshotSlotState {
            frame_id: snapshot.frame_id,
            instance_count: snapshot.transforms.len() as u32,
            material_count: snapshot.materials.len() as u32,
            texture_count: snapshot.textures.len() as u32,
            light_count: snapshot.lights.len() as u32,
            byte_len,
        };

        Ok(slot.last_state)
    }

    /// Submit one frame worth of work to the Metal command queue.
    pub fn render_n(
        &mut self,
        snapshot: RenderStateSnapshot<'_>,
    ) -> Result<MetalFrameExecutionTelemetry, MetalBackendError> {
        let frame_slot = self.current_frame_slot;
        let snapshot_state = self.upload_state_snapshot(frame_slot, snapshot)?;

        let command_buffer = self.command_queue.new_command_buffer();
        command_buffer.commit();

        self.primary_submission_serial = self.primary_submission_serial.saturating_add(1);
        self.current_frame_slot = (self.current_frame_slot + 1) % self.frame_slots.len();

        let submission = MetalFrameSubmissionTelemetry {
            frame_slot,
            frame_id: snapshot_state.frame_id,
            instance_count: snapshot_state.instance_count,
            material_count: snapshot_state.material_count,
            texture_count: snapshot_state.texture_count,
            light_count: snapshot_state.light_count,
        };

        Ok(MetalFrameExecutionTelemetry {
            submission,
            snapshot_state,
            command_buffer_submitted: true,
            presented: false,
            primary_submission_serial: self.primary_submission_serial,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_valid() {
        let cfg = MetalBackendConfig::default();
        assert!(cfg.frames_in_flight >= 1);
        assert!(cfg.max_instances >= 1);
    }
}
