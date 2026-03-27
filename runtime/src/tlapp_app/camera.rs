use nalgebra::Vector3;
use winit::window::{CursorGrabMode, Window};

use crate::TlscriptCoordinateSpace;

#[derive(Debug, Clone, Copy, Default)]
pub struct CameraInputState {
    pub key_w: bool,
    pub key_s: bool,
    pub key_a: bool,
    pub key_d: bool,
    pub key_up: bool,
    pub key_down: bool,
    pub key_left: bool,
    pub key_right: bool,
    pub key_space: bool,
    pub key_ctrl: bool,
    pub key_shift: bool,
    pub key_q: bool,
    pub key_e: bool,
    pub key_c: bool,
    pub key_r: bool,
    pub key_l: bool,
    pub key_alt: bool,
    pub key_enter: bool,
    pub key_escape: bool,
    pub key_tab: bool,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct GamepadCameraState {
    pub move_x: f32,
    pub move_y: f32,
    pub look_x: f32,
    pub look_y: f32,
    pub rise: f32,
    pub descend: f32,
    pub sprint: bool,
}

#[derive(Debug, Clone)]
pub struct FreeCameraController {
    pub position: Vector3<f32>,
    pub yaw_rad: f32,
    pub pitch_rad: f32,
    pub move_speed: f32,
    pub sprint_multiplier: f32,
    pub mouse_sensitivity: f32,
    pub look_active: bool,
    pub gamepad: GamepadCameraState,
    pub script_coordinate_space: TlscriptCoordinateSpace,
}

impl Default for FreeCameraController {
    fn default() -> Self {
        Self {
            position: Vector3::new(0.0, 12.0, 36.0),
            yaw_rad: 0.0,
            pitch_rad: -0.321_750_55,
            move_speed: 18.0,
            sprint_multiplier: 2.5,
            mouse_sensitivity: 0.0018,
            look_active: false,
            gamepad: GamepadCameraState::default(),
            script_coordinate_space: TlscriptCoordinateSpace::World,
        }
    }
}

impl FreeCameraController {
    pub fn set_look_active(&mut self, window: &Window, active: bool) {
        if self.look_active == active {
            return;
        }
        self.look_active = active;

        if active {
            let grab_result = window
                .set_cursor_grab(CursorGrabMode::Locked)
                .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined));
            if let Err(err) = grab_result {
                eprintln!("[camera] cursor grab failed: {err}");
            }
            window.set_cursor_visible(false);
        } else {
            let _ = window.set_cursor_grab(CursorGrabMode::None);
            window.set_cursor_visible(true);
        }
    }

    pub fn on_mouse_delta(&mut self, dx: f32, dy: f32) {
        if !self.look_active {
            return;
        }
        self.yaw_rad += dx * self.mouse_sensitivity;
        self.pitch_rad -= dy * self.mouse_sensitivity;
        self.pitch_rad = self.pitch_rad.clamp(-1.553_343, 1.553_343);
    }

    pub fn set_move_speed(&mut self, speed: f32) {
        self.move_speed = speed.clamp(1.0, 200.0);
    }

    pub fn mouse_sensitivity(&self) -> f32 {
        self.mouse_sensitivity
    }

    pub fn set_script_move_axis(&mut self, axis: [f32; 3], sprint: bool) {
        self.gamepad.move_x = axis[0].clamp(-1.0, 1.0);
        self.gamepad.move_y = axis[1].clamp(-1.0, 1.0);
        self.gamepad.rise = axis[2].max(0.0);
        self.gamepad.descend = (-axis[2]).max(0.0);
        self.gamepad.sprint = sprint;
    }

    pub fn set_script_coordinate_space(&mut self, space: TlscriptCoordinateSpace) {
        self.script_coordinate_space = space;
    }

    pub fn apply_script_translate_delta(&mut self, delta: [f32; 3]) {
        let delta = Vector3::new(delta[0], delta[1], delta[2]);
        match self.script_coordinate_space {
            TlscriptCoordinateSpace::World => {
                self.position += delta;
            }
            TlscriptCoordinateSpace::Local => {
                let forward = self.forward_vector();
                let world_up = Vector3::new(0.0, 1.0, 0.0);
                let mut right = forward.cross(&world_up);
                let right_len = right.norm();
                if right_len <= 1e-5 {
                    right = Vector3::new(1.0, 0.0, 0.0);
                } else {
                    right /= right_len;
                }
                let up = right.cross(&forward).normalize();
                self.position += right * delta.x + up * delta.y + forward * delta.z;
            }
        }
    }

    pub fn apply_script_rotate_delta_deg(&mut self, delta_deg: [f32; 2]) {
        self.yaw_rad += delta_deg[0].to_radians();
        self.pitch_rad = (self.pitch_rad + delta_deg[1].to_radians()).clamp(-1.553_343, 1.553_343);
    }

    pub fn set_mouse_sensitivity(&mut self, sensitivity: f32) {
        self.mouse_sensitivity = sensitivity.clamp(0.0001, 0.02);
    }

    pub fn set_pose(&mut self, eye: [f32; 3], target: [f32; 3]) {
        self.position = Vector3::new(eye[0], eye[1], eye[2]);
        let dir = Vector3::new(target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]);
        let len = dir.norm();
        if len <= 1e-5 {
            return;
        }
        let d = dir / len;
        self.pitch_rad = d.y.asin().clamp(-1.553_343, 1.553_343);
        self.yaw_rad = d.x.atan2(-d.z);
    }

    pub fn reset_pose(&mut self) {
        self.position = Vector3::new(0.0, 12.0, 36.0);
        self.yaw_rad = 0.0;
        self.pitch_rad = -0.321_750_55;
    }

    pub fn update(&mut self, dt: f32) {
        let forward = self.forward_vector();
        let horizontal_forward = Vector3::new(forward.x, 0.0, forward.z);
        let forward_len = horizontal_forward.norm();
        let forward_flat = if forward_len > 1e-5 {
            horizontal_forward / forward_len
        } else {
            Vector3::new(0.0, 0.0, -1.0)
        };
        let right = Vector3::new(-forward_flat.z, 0.0, forward_flat.x);

        let mut move_dir = Vector3::zeros();
        move_dir += right * self.gamepad.move_x;
        move_dir += forward_flat * self.gamepad.move_y;
        move_dir.y += self.gamepad.rise;
        move_dir.y -= self.gamepad.descend;

        let len = move_dir.norm();
        if len > 1e-5 {
            let move_dir = move_dir / len;
            let speed = self.move_speed
                * if self.gamepad.sprint {
                    self.sprint_multiplier
                } else {
                    1.0
                };
            self.position += move_dir * speed * dt.max(0.0);
        }
    }

    pub fn eye_target(&self) -> ([f32; 3], [f32; 3]) {
        let forward = self.forward_vector();
        let eye = [self.position.x, self.position.y, self.position.z];
        let target = [
            self.position.x + forward.x,
            self.position.y + forward.y,
            self.position.z + forward.z,
        ];
        (eye, target)
    }

    pub fn forward_vector(&self) -> Vector3<f32> {
        let (sin_yaw, cos_yaw) = self.yaw_rad.sin_cos();
        let (sin_pitch, cos_pitch) = self.pitch_rad.sin_cos();
        Vector3::new(sin_yaw * cos_pitch, sin_pitch, -cos_yaw * cos_pitch)
    }
}

#[cfg(feature = "gamepad")]
#[derive(Debug, Clone, Copy, Default)]
pub struct GamepadRawState {
    pub left_x: f32,
    pub left_y: f32,
    pub right_x: f32,
    pub right_y: f32,
    pub left_trigger_2: f32,
    pub right_trigger_2: f32,
    pub dpad_up: bool,
    pub dpad_down: bool,
    pub dpad_left: bool,
    pub dpad_right: bool,
    pub south: bool,
    pub east: bool,
    pub sprint_left_trigger: bool,
    pub sprint_left_thumb: bool,
}

#[cfg(feature = "gamepad")]
use gilrs::{Axis, Button, EventType, GamepadId, Gilrs};

#[cfg(feature = "gamepad")]
pub struct GamepadManager {
    pub gilrs: Option<Gilrs>,
    pub active_id: Option<GamepadId>,
    pub raw: GamepadRawState,
}

#[cfg(feature = "gamepad")]
impl GamepadManager {
    pub fn new() -> Self {
        let mut manager = Self {
            gilrs: None,
            active_id: None,
            raw: GamepadRawState::default(),
        };
        match Gilrs::new() {
            Ok(gilrs) => {
                manager.active_id = gilrs.gamepads().next().map(|(id, _)| id);
                if manager.active_id.is_some() {
                    eprintln!("[input] gamepad support enabled");
                } else {
                    eprintln!("[input] gamepad subsystem ready (no device connected yet)");
                }
                manager.gilrs = Some(gilrs);
            }
            Err(err) => {
                eprintln!("[input] gamepad subsystem unavailable: {err}");
            }
        }
        manager
    }

    pub fn poll(&mut self) {
        loop {
            let event = match self.gilrs.as_mut().and_then(|g| g.next_event()) {
                Some(event) => event,
                None => break,
            };
            self.active_id = Some(event.id);
            match event.event {
                EventType::Connected => {
                    self.active_id = Some(event.id);
                    self.raw = GamepadRawState::default();
                }
                EventType::Disconnected => {
                    if self.active_id == Some(event.id) {
                        self.active_id = None;
                        self.raw = GamepadRawState::default();
                    }
                }
                EventType::AxisChanged(axis, value, _) => {
                    self.update_axis(axis, value);
                }
                EventType::ButtonPressed(button, _) => {
                    self.set_button(button, true);
                }
                EventType::ButtonReleased(button, _) => {
                    self.set_button(button, false);
                }
                _ => {}
            }
        }
    }

    pub fn action_f_down(&self) -> bool {
        self.raw.south
    }

    pub fn action_g_down(&self) -> bool {
        self.raw.east
    }

    pub fn camera_state(&self) -> GamepadCameraState {
        let dpad_x = (self.raw.dpad_right as i8 - self.raw.dpad_left as i8) as f32;
        let dpad_y = (self.raw.dpad_up as i8 - self.raw.dpad_down as i8) as f32;
        GamepadCameraState {
            move_x: normalize_axis(self.raw.left_x + dpad_x),
            move_y: normalize_axis(-self.raw.left_y + dpad_y),
            look_x: normalize_axis(self.raw.right_x),
            look_y: normalize_axis(-self.raw.right_y),
            rise: normalize_axis(self.raw.right_trigger_2),
            descend: normalize_axis(self.raw.left_trigger_2),
            sprint: self.raw.sprint_left_trigger || self.raw.sprint_left_thumb,
        }
    }

    fn update_axis(&mut self, axis: Axis, value: f32) {
        let value = normalize_axis(value);
        match axis {
            Axis::LeftStickX => self.raw.left_x = value,
            Axis::LeftStickY => self.raw.left_y = value,
            Axis::RightStickX => self.raw.right_x = value,
            Axis::RightStickY => self.raw.right_y = value,
            Axis::LeftZ => self.raw.left_trigger_2 = value.max(0.0),
            Axis::RightZ => self.raw.right_trigger_2 = value.max(0.0),
            Axis::DPadX => {
                self.raw.dpad_left = value < -0.5;
                self.raw.dpad_right = value > 0.5;
            }
            Axis::DPadY => {
                self.raw.dpad_down = value < -0.5;
                self.raw.dpad_up = value > 0.5;
            }
            _ => {}
        }
    }

    fn set_button(&mut self, button: Button, pressed: bool) {
        match button {
            Button::South => self.raw.south = pressed,
            Button::East => self.raw.east = pressed,
            Button::LeftTrigger => self.raw.sprint_left_trigger = pressed,
            Button::LeftThumb => self.raw.sprint_left_thumb = pressed,
            Button::LeftTrigger2 => self.raw.left_trigger_2 = if pressed { 1.0 } else { 0.0 },
            Button::RightTrigger2 => self.raw.right_trigger_2 = if pressed { 1.0 } else { 0.0 },
            Button::DPadUp => self.raw.dpad_up = pressed,
            Button::DPadDown => self.raw.dpad_down = pressed,
            Button::DPadLeft => self.raw.dpad_left = pressed,
            Button::DPadRight => self.raw.dpad_right = pressed,
            _ => {}
        }
    }
}

#[cfg(not(feature = "gamepad"))]
pub struct GamepadManager;

#[cfg(not(feature = "gamepad"))]
impl GamepadManager {
    pub fn new() -> Self {
        eprintln!("[input] gamepad support disabled (runtime feature 'gamepad' is off)");
        Self
    }

    pub fn poll(&mut self) {}

    pub fn action_f_down(&self) -> bool {
        false
    }

    pub fn action_g_down(&self) -> bool {
        false
    }

    pub fn camera_state(&self) -> GamepadCameraState {
        GamepadCameraState::default()
    }
}

#[cfg(feature = "gamepad")]
#[inline]
pub fn normalize_axis(value: f32) -> f32 {
    use super::GAMEPAD_DEADZONE;
    let clamped = value.clamp(-1.0, 1.0);
    if clamped.abs() < GAMEPAD_DEADZONE {
        0.0
    } else {
        clamped
    }
}
