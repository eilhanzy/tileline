use super::*;

impl TlAppRuntime {
    pub(super) fn on_keyboard_input(&mut self, event: &KeyEvent) -> RuntimeCommand {
        let pressed = event.state == ElementState::Pressed;
        let ctrl = self.keyboard_modifiers.control_key() || self.keyboard_camera.key_ctrl;
        let alt = self.keyboard_modifiers.alt_key();
        let is_f1 = matches!(event.physical_key, PhysicalKey::Code(KeyCode::F1))
            || matches!(event.logical_key, Key::Named(NamedKey::F1));
        let is_backquote = matches!(event.physical_key, PhysicalKey::Code(KeyCode::Backquote))
            || matches!(&event.logical_key, Key::Character(raw) if raw.as_ref() == "`" || raw.as_ref() == "~");
        let is_key_k = matches!(event.physical_key, PhysicalKey::Code(KeyCode::KeyK));
        let toggle_console =
            pressed && !event.repeat && (is_f1 || (ctrl && (is_backquote || is_key_k)));

        if toggle_console {
            self.toggle_console();
            return RuntimeCommand::Consumed;
        }

        if self.console.open {
            return self.on_console_keyboard_input(event);
        }

        self.update_camera_keyboard_input(event.physical_key, pressed);
        if let PhysicalKey::Code(KeyCode::KeyF) = event.physical_key {
            self.script_key_f_keyboard = pressed;
        }
        if let PhysicalKey::Code(KeyCode::KeyG) = event.physical_key {
            self.script_key_g_keyboard = pressed;
        }

        if !pressed || event.repeat {
            return RuntimeCommand::None;
        }

        if alt && matches!(event.physical_key, PhysicalKey::Code(KeyCode::Enter)) {
            self.toggle_fullscreen();
            return RuntimeCommand::None;
        }

        if ctrl {
            match event.physical_key {
                PhysicalKey::Code(KeyCode::KeyQ) => return RuntimeCommand::Exit,
                PhysicalKey::Code(KeyCode::KeyF) => {
                    self.toggle_fullscreen();
                    return RuntimeCommand::None;
                }
                _ => {}
            }
        }
        RuntimeCommand::None
    }

    pub(super) fn on_modifiers_changed(&mut self, modifiers: ModifiersState) {
        self.keyboard_modifiers = modifiers;
    }

    pub(super) fn toggle_fullscreen(&self) {
        let next = if self.window.fullscreen().is_some() {
            None
        } else {
            Some(Fullscreen::Borderless(None))
        };
        self.window.set_fullscreen(next);
    }

    pub(super) fn on_cursor_moved(&mut self, x: f32, y: f32) {
        self.cursor_position = Some((x, y));
    }

    pub(super) fn cursor_ndc(&self) -> Option<(f32, f32)> {
        let (x, y) = self.cursor_position?;
        let width = self.size.width.max(1) as f32;
        let height = self.size.height.max(1) as f32;
        let ndc_x = (x / width) * 2.0 - 1.0;
        let ndc_y = 1.0 - (y / height) * 2.0;
        Some((ndc_x, ndc_y))
    }

    pub(super) fn on_mouse_button(&mut self, state: ElementState, button: MouseButton) {
        if self.console.open {
            if button == MouseButton::Left && state == ElementState::Pressed {
                self.handle_console_left_click();
            }
            return;
        }
        if button == MouseButton::Right {
            self.mouse_look_held = state == ElementState::Pressed;
        }
    }

    pub(super) fn on_touch(&mut self, touch: Touch) {
        if self.console.open {
            return;
        }
        let id = touch.id;
        let current = (touch.location.x as f32, touch.location.y as f32);
        match touch.phase {
            TouchPhase::Started => {
                self.touch_look_id = Some(id);
                self.touch_last_position = Some(current);
                self.mouse_look_held = true;
            }
            TouchPhase::Moved => {
                if self.touch_look_id == Some(id) {
                    if let Some(previous) = self.touch_last_position {
                        self.mouse_look_delta.0 += current.0 - previous.0;
                        self.mouse_look_delta.1 += current.1 - previous.1;
                    }
                    self.touch_last_position = Some(current);
                }
            }
            TouchPhase::Ended | TouchPhase::Cancelled => {
                if self.touch_look_id == Some(id) {
                    self.touch_look_id = None;
                    self.touch_last_position = None;
                    self.mouse_look_held = false;
                }
            }
        }
    }

    pub(super) fn on_device_event(&mut self, event: DeviceEvent) {
        if self.console.open {
            return;
        }
        if let DeviceEvent::MouseMotion { delta } = event {
            self.mouse_look_delta.0 += delta.0 as f32;
            self.mouse_look_delta.1 += delta.1 as f32;
        }
    }

    pub(super) fn poll_input_devices(&mut self) {
        self.gamepad.poll();
    }

    pub(super) fn update_camera_keyboard_input(&mut self, key: PhysicalKey, pressed: bool) {
        match key {
            PhysicalKey::Code(KeyCode::KeyW) | PhysicalKey::Code(KeyCode::ArrowUp) => {
                if matches!(key, PhysicalKey::Code(KeyCode::KeyW)) {
                    self.keyboard_camera.key_w = pressed;
                }
                if matches!(key, PhysicalKey::Code(KeyCode::ArrowUp)) {
                    self.keyboard_camera.key_up = pressed;
                }
            }
            PhysicalKey::Code(KeyCode::KeyS) | PhysicalKey::Code(KeyCode::ArrowDown) => {
                if matches!(key, PhysicalKey::Code(KeyCode::KeyS)) {
                    self.keyboard_camera.key_s = pressed;
                }
                if matches!(key, PhysicalKey::Code(KeyCode::ArrowDown)) {
                    self.keyboard_camera.key_down = pressed;
                }
            }
            PhysicalKey::Code(KeyCode::KeyA) | PhysicalKey::Code(KeyCode::ArrowLeft) => {
                if matches!(key, PhysicalKey::Code(KeyCode::KeyA)) {
                    self.keyboard_camera.key_a = pressed;
                }
                if matches!(key, PhysicalKey::Code(KeyCode::ArrowLeft)) {
                    self.keyboard_camera.key_left = pressed;
                }
            }
            PhysicalKey::Code(KeyCode::KeyD) | PhysicalKey::Code(KeyCode::ArrowRight) => {
                if matches!(key, PhysicalKey::Code(KeyCode::KeyD)) {
                    self.keyboard_camera.key_d = pressed;
                }
                if matches!(key, PhysicalKey::Code(KeyCode::ArrowRight)) {
                    self.keyboard_camera.key_right = pressed;
                }
            }
            PhysicalKey::Code(KeyCode::Space) | PhysicalKey::Code(KeyCode::KeyE) => {
                if matches!(key, PhysicalKey::Code(KeyCode::Space)) {
                    self.keyboard_camera.key_space = pressed;
                }
                if matches!(key, PhysicalKey::Code(KeyCode::KeyE)) {
                    self.keyboard_camera.key_e = pressed;
                }
            }
            PhysicalKey::Code(KeyCode::ControlLeft) | PhysicalKey::Code(KeyCode::ControlRight) => {
                self.keyboard_camera.key_ctrl = pressed
            }
            PhysicalKey::Code(KeyCode::KeyQ) => self.keyboard_camera.key_q = pressed,
            PhysicalKey::Code(KeyCode::KeyC) => self.keyboard_camera.key_c = pressed,
            PhysicalKey::Code(KeyCode::ShiftLeft) | PhysicalKey::Code(KeyCode::ShiftRight) => {
                self.keyboard_camera.key_shift = pressed
            }
            PhysicalKey::Code(KeyCode::KeyR) => self.keyboard_camera.key_r = pressed,
            PhysicalKey::Code(KeyCode::KeyL) => self.keyboard_camera.key_l = pressed,
            PhysicalKey::Code(KeyCode::AltLeft) | PhysicalKey::Code(KeyCode::AltRight) => {
                self.keyboard_camera.key_alt = pressed
            }
            PhysicalKey::Code(KeyCode::Enter) | PhysicalKey::Code(KeyCode::NumpadEnter) => {
                self.keyboard_camera.key_enter = pressed
            }
            PhysicalKey::Code(KeyCode::Escape) => self.keyboard_camera.key_escape = pressed,
            PhysicalKey::Code(KeyCode::Tab) => self.keyboard_camera.key_tab = pressed,
            _ => {}
        }
    }

    pub(super) fn script_camera_input(&mut self, view_dt: f32) -> TlscriptShowcaseControlInput {
        if self.console.open {
            self.mouse_look_delta = (0.0, 0.0);
            return TlscriptShowcaseControlInput::default();
        }
        let gamepad = self.gamepad.camera_state();
        let sensitivity = self.camera.mouse_sensitivity().max(0.0001);
        let pad_look_to_mouse = (GAMEPAD_LOOK_SPEED_RAD * view_dt.max(0.0)) / sensitivity;

        // Keyboard layout is script-driven now; keep legacy move_* channels fed from gamepad only.
        let move_x = gamepad.move_x;
        let move_y = gamepad.move_y;
        let move_z = gamepad.rise - gamepad.descend;
        let look_dx = self.mouse_look_delta.0 + gamepad.look_x * pad_look_to_mouse;
        let look_dy = self.mouse_look_delta.1 + gamepad.look_y * pad_look_to_mouse;
        self.mouse_look_delta = (0.0, 0.0);

        let look_active = self.look_lock_active
            || self.mouse_look_held
            || gamepad.look_x.abs() > 0.001
            || gamepad.look_y.abs() > 0.001;
        let reset_camera = std::mem::take(&mut self.camera_reset_requested);

        TlscriptShowcaseControlInput {
            move_x,
            move_y,
            move_z,
            look_dx,
            look_dy,
            sprint_down: self.keyboard_camera.key_shift || gamepad.sprint,
            look_active,
            reset_camera,
            key_w_down: self.keyboard_camera.key_w,
            key_s_down: self.keyboard_camera.key_s,
            key_a_down: self.keyboard_camera.key_a,
            key_d_down: self.keyboard_camera.key_d,
            key_up_down: self.keyboard_camera.key_up,
            key_down_down: self.keyboard_camera.key_down,
            key_left_down: self.keyboard_camera.key_left,
            key_right_down: self.keyboard_camera.key_right,
            key_space_down: self.keyboard_camera.key_space,
            key_ctrl_down: self.keyboard_camera.key_ctrl,
            key_shift_down: self.keyboard_camera.key_shift,
            key_q_down: self.keyboard_camera.key_q,
            key_e_down: self.keyboard_camera.key_e,
            key_c_down: self.keyboard_camera.key_c,
            key_g_down: self.script_key_g_keyboard || self.gamepad.action_f_down(),
            key_r_down: self.keyboard_camera.key_r,
            key_l_down: self.keyboard_camera.key_l,
            key_alt_down: self.keyboard_camera.key_alt,
            key_enter_down: self.keyboard_camera.key_enter,
            key_escape_down: self.keyboard_camera.key_escape,
            key_tab_down: self.keyboard_camera.key_tab,
            mouse_look_down: self.mouse_look_held,
            pad_move_x: gamepad.move_x,
            pad_move_y: gamepad.move_y,
            pad_rise: gamepad.rise,
            pad_descend: gamepad.descend,
            pad_look_x: gamepad.look_x,
            pad_look_y: gamepad.look_y,
            pad_sprint_down: gamepad.sprint,
        }
    }
}
