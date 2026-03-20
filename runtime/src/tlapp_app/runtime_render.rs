use super::*;

impl TlAppRuntime {
    pub(super) fn schedule_next_redraw(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(interval) = self.frame_cap_interval {
            let now = Instant::now();
            if now < self.next_redraw_at {
                event_loop.set_control_flow(ControlFlow::WaitUntil(self.next_redraw_at));
                return;
            }
            self.next_redraw_at = now + interval;
        } else {
            let mobile_path = matches!(self.platform, RuntimePlatform::Android)
                || self.mgs_is_mobile_hardware;
            if mobile_path {
                // Avoid uncapped busy-spin on mobile/TBDR paths; it can cause whole-system
                // chopping even when the app's own FPS appears acceptable.
                let next = Instant::now() + Duration::from_millis(2);
                self.next_redraw_at = next;
                event_loop.set_control_flow(ControlFlow::WaitUntil(next));
                self.window.request_redraw();
                return;
            }
        }

        event_loop.set_control_flow(ControlFlow::Poll);
        self.window.request_redraw();
    }

    pub(super) fn apply_present_mode(&mut self, mode: wgpu::PresentMode) {
        if self.present_mode == mode {
            return;
        }
        self.present_mode = mode;
        self.config.present_mode = mode;
        self.surface.configure(&self.device, &self.config);
    }

    pub(super) fn apply_fps_cap_runtime(&mut self, cap: Option<f32>) {
        match cap {
            Some(fps) => {
                let clamped = fps.max(24.0);
                self.frame_cap_interval = Some(Duration::from_secs_f32(1.0 / clamped));
                self.fps_limit_hint = clamped;
                self.uncapped_dynamic_fps_hint = false;
                self.adaptive_pacer_enabled = false;
            }
            None => {
                self.frame_cap_interval = None;
                self.uncapped_dynamic_fps_hint = true;
                self.fps_limit_hint = bootstrap_uncapped_fps_hint(self.mps_logical_threads);
                self.adaptive_pacer_enabled = false;
            }
        }
    }

    pub(super) fn retune_render_distance(&mut self, mobile_path: bool) {
        if !self.adaptive_distance_enabled || self.frame_time_ema_ms <= f32::EPSILON {
            return;
        }
        let Some(mut current) = self.render_distance else {
            return;
        };
        if self.render_distance_max <= self.render_distance_min + f32::EPSILON {
            return;
        }

        let target_ms = (1_000.0 / self.fps_limit_hint.max(24.0)).clamp(5.0, 42.0);
        let frame_pressure = (self.frame_time_ema_ms / target_ms).clamp(0.4, 2.5);
        let jitter_pressure =
            (self.frame_time_jitter_ema_ms / (target_ms * 0.5).max(0.5)).clamp(0.0, 2.0);
        let fill_pressure = self.framebuffer_fill_ema.clamp(0.0, 3.0);

        let overload = (frame_pressure - 1.0).max(0.0)
            + (fill_pressure - 1.0).max(0.0) * 0.75
            + jitter_pressure * 0.18;
        let headroom = (1.0 - frame_pressure).max(0.0)
            + (0.85 - fill_pressure).max(0.0) * 0.65
            + (0.12 - jitter_pressure).max(0.0) * 0.40;

        if overload > 0.04 {
            let shrink =
                (0.025 + overload * 0.035).clamp(0.02, if mobile_path { 0.10 } else { 0.07 });
            current *= 1.0 - shrink;
        } else if headroom > 0.10 {
            let grow =
                (0.008 + headroom * 0.025).clamp(0.008, if mobile_path { 0.06 } else { 0.04 });
            current *= 1.0 + grow;
        }

        let snapped = (current * 2.0).round() * 0.5;
        self.render_distance =
            Some(snapped.clamp(self.render_distance_min, self.render_distance_max));
    }

    pub(super) fn refresh_distance_blur_state(&mut self, mobile_path: bool) {
        self.distance_blur_enabled = match self.distance_blur_mode {
            ToggleAuto::On => true,
            ToggleAuto::Off => false,
            ToggleAuto::Auto => {
                if self.distance_blur_enabled {
                    !(self.framebuffer_fill_ema < 0.72
                        && self.frame_time_ema_ms < self.frame_time_budget_ms * 0.90
                        && !mobile_path)
                } else {
                    self.framebuffer_fill_ema > 0.92
                        || self.frame_time_ema_ms > self.frame_time_budget_ms * 1.05
                        || mobile_path
                }
            }
        };
    }

    pub(super) fn retune_adaptive_pacer(&mut self, dt: f32, mobile_path: bool) {
        if !self.adaptive_pacer_enabled {
            return;
        }

        self.adaptive_pacer_timer -= dt.max(0.0);
        if self.adaptive_pacer_timer > 0.0 {
            return;
        }

        let target_ms = (1_000.0 / self.adaptive_pacer_fps.max(1.0)).clamp(6.0, 42.0);
        let overload = self.frame_time_ema_ms > target_ms * 1.03
            || self.frame_time_jitter_ema_ms > target_ms * 0.20
            || self.framebuffer_fill_ema > 0.95;
        let headroom = self.frame_time_ema_ms < target_ms * 0.82
            && self.frame_time_jitter_ema_ms < target_ms * 0.10
            && self.framebuffer_fill_ema < 0.70;

        if overload {
            self.adaptive_pacer_fps *= 0.93;
            self.adaptive_pacer_timer = 0.18;
        } else if headroom {
            self.adaptive_pacer_fps *= 1.03;
            self.adaptive_pacer_timer = 0.42;
        } else {
            self.adaptive_pacer_timer = 0.28;
        }

        let min_cap = if mobile_path { 45.0 } else { 55.0 };
        let max_cap = if mobile_path { 90.0 } else { 120.0 };
        self.adaptive_pacer_fps = self.adaptive_pacer_fps.clamp(min_cap, max_cap);
        self.frame_cap_interval = Some(Duration::from_secs_f32(
            1.0 / self.adaptive_pacer_fps.max(1.0),
        ));
    }

    pub(super) fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }
        self.size = new_size;
        self.config.width = new_size.width;
        self.config.height = new_size.height;
        self.surface.configure(&self.device, &self.config);
        self.renderer.resize(
            &self.device,
            &self.queue,
            new_size.width.max(1),
            new_size.height.max(1),
        );
    }

    pub(super) fn render_frame(&mut self) -> Result<(), Box<dyn Error>> {
        let frame_begin = Instant::now();
        let mobile_path = matches!(self.platform, RuntimePlatform::Android)
            || self.mgs_is_mobile_hardware;
        let raw_dt = (frame_begin - self.frame_started_at).as_secs_f32();
        // Keep simulation time real-time (decoupled from render FPS) and only guard against large
        // stalls (alt-tab/debugger) so physics does not enter slow-motion at low FPS.
        let sim_dt = raw_dt.clamp(1.0 / 1_000.0, 0.25);
        // Camera/input smoothing can use a separate clamped delta.
        let view_dt = raw_dt.clamp(1.0 / 500.0, 1.0 / 24.0);
        let frame_ms = raw_dt * 1_000.0;
        if self.frame_time_ema_ms <= f32::EPSILON {
            self.frame_time_ema_ms = frame_ms;
            self.frame_time_jitter_ema_ms = 0.0;
        } else {
            let frame_delta = (frame_ms - self.frame_time_ema_ms).abs();
            self.frame_time_ema_ms += (frame_ms - self.frame_time_ema_ms) * 0.10;
            self.frame_time_jitter_ema_ms += (frame_delta - self.frame_time_jitter_ema_ms) * 0.16;
        }
        self.frame_started_at = frame_begin;

        self.poll_input_devices();
        if self.console.open {
            self.poll_console_tail_follow();
            self.poll_console_file_watch();
        }
        let script_camera_input = self.script_camera_input(view_dt);

        if let (Some(sprite_loader), Some(sprite_cache)) =
            (self.sprite_loader.as_mut(), self.sprite_cache.as_mut())
        {
            let event = sprite_loader.reload_into_cache(sprite_cache);
            match &event {
                TlspriteHotReloadEvent::Applied { .. } => {
                    print_tlsprite_event("[tlsprite reload]", event);
                    if let Some(program) =
                        sprite_cache.program_for_path(sprite_loader.path()).cloned()
                    {
                        self.force_full_fbx_from_sprite = program.requires_full_fbx_render();
                        self.scene.set_sprite_program(program.clone());
                        bind_renderer_meshes_from_tlsprite(
                            &mut self.renderer,
                            &self.device,
                            sprite_loader.path(),
                            &program,
                        );
                    }
                }
                TlspriteHotReloadEvent::Unchanged => {}
                _ => print_tlsprite_event("[tlsprite reload]", event),
            }
        }

        let mut frame_eval = self.script_runtime.evaluate_frame(
            TlscriptShowcaseFrameInput {
                frame_index: self.script_frame_index,
                live_balls: self.scene.live_ball_count(),
                spawned_this_tick: self.script_last_spawned,
                key_f_down: self.script_key_f_keyboard || self.gamepad.action_f_down(),
            },
            script_camera_input,
        );
        if !self.console.script_statements.is_empty() {
            merge_showcase_output(
                &mut frame_eval,
                self.console.script_overlay.clone(),
                CONSOLE_SCRIPT_INDEX,
            );
        }

        if let Some(speed) = frame_eval.camera_move_speed {
            self.camera.set_move_speed(speed);
        }
        if let Some(sensitivity) = frame_eval.camera_look_sensitivity {
            self.camera.set_mouse_sensitivity(sensitivity);
        }
        if let Some(active) = frame_eval.camera_look_active {
            self.camera.set_look_active(&self.window, active);
        }
        if frame_eval.camera_reset_pose {
            self.camera.reset_pose();
        }
        if let Some((camera_eye, camera_target)) = frame_eval.camera_pose {
            self.camera.set_pose(camera_eye, camera_target);
        }
        if let Some(space) = frame_eval.camera_coordinate_space {
            self.camera.set_script_coordinate_space(space);
        }
        if let Some(delta) = frame_eval.camera_translate_delta {
            self.camera.apply_script_translate_delta(delta);
        }
        if let Some(delta_deg) = frame_eval.camera_rotate_delta_deg {
            self.camera.apply_script_rotate_delta_deg(delta_deg);
        }
        self.camera.set_script_move_axis(
            frame_eval.camera_move_axis.unwrap_or([0.0, 0.0, 0.0]),
            frame_eval.camera_sprint.unwrap_or(false),
        );
        if let Some([look_dx, look_dy]) = frame_eval.camera_look_delta {
            self.camera.on_mouse_delta(look_dx, look_dy);
        }
        self.camera.update(view_dt);
        let (eye, target) = self.camera.eye_target();
        self.renderer.set_camera_view(
            &self.queue,
            self.size.width.max(1),
            self.size.height.max(1),
            eye,
            target,
        );
        if let Some(mode) = frame_eval.rt_mode {
            self.rt_mode = mode;
            self.renderer.set_ray_tracing_mode(&self.queue, mode);
        }

        let live_balls = self.scene.live_ball_count();
        let parallel_ready = frame_eval
            .dispatch_decision
            .as_ref()
            .map(|d| d.is_parallel())
            .unwrap_or(false);
        let force_full_fbx = frame_eval
            .force_full_fbx_sphere
            .unwrap_or(self.force_full_fbx_from_sprite);
        let mut runtime_patch = frame_eval.patch;
        self.frame_time_budget_ms = (1_000.0 / self.fps_limit_hint.max(24.0)).clamp(3.0, 41.0);
        self.distance_retune_timer -= sim_dt;
        if self.distance_retune_timer <= 0.0 {
            self.retune_render_distance(mobile_path);
            self.distance_retune_timer = if mobile_path { 0.22 } else { 0.28 };
        }
        self.refresh_distance_blur_state(mobile_path);
        let mut load_plan = choose_runtime_load_plan(
            self.fps_tracker.ema_fps(),
            raw_dt * 1_000.0,
            live_balls,
            self.last_substeps,
            self.max_substeps,
            parallel_ready,
            self.mps_logical_threads,
            mobile_path,
            self.framebuffer_fill_ema,
            self.render_distance,
        );
        let moderate_jitter = self.frame_time_ema_ms > self.frame_time_budget_ms * 1.10
            || self.frame_time_jitter_ema_ms > 1.8;
        let severe_jitter = self.frame_time_ema_ms > self.frame_time_budget_ms * 1.25
            || self.frame_time_jitter_ema_ms > 3.2;
        if moderate_jitter {
            load_plan.tick_scale *= 0.82;
            load_plan.max_substeps = load_plan.max_substeps.min(8);
            load_plan.spawn_per_tick_cap = load_plan.spawn_per_tick_cap.min(180);
        }
        if severe_jitter {
            load_plan.tick_scale *= 0.68;
            load_plan.max_substeps = load_plan.max_substeps.min(6);
            load_plan.spawn_per_tick_cap = load_plan.spawn_per_tick_cap.min(120);
        }
        if mobile_path {
            load_plan.tick_scale *= if severe_jitter { 0.70 } else { 0.82 };
            load_plan.max_substeps = load_plan
                .max_substeps
                .min(if severe_jitter { 4 } else { 6 });
            load_plan.spawn_per_tick_cap =
                load_plan
                    .spawn_per_tick_cap
                    .min(if severe_jitter { 96 } else { 112 });
        }
        if matches!(self.tick_profile, TickProfile::Max) {
            let min_tick_scale = if severe_jitter {
                0.62
            } else if moderate_jitter {
                0.74
            } else if mobile_path {
                0.68
            } else {
                0.90
            };
            load_plan.tick_scale = load_plan.tick_scale.max(min_tick_scale);
            let profile_cap = if mobile_path {
                (5_u32 + (self.mps_logical_threads as u32 / 10)).clamp(5, 8)
            } else {
                (10_u32 + (self.mps_logical_threads as u32 / 6)).clamp(10, 20)
            };
            let min_substeps = if mobile_path { 3 } else { 10 };
            load_plan.max_substeps = load_plan
                .max_substeps
                .clamp(min_substeps, profile_cap.max(min_substeps));
        }
        self.adaptive_ball_render_limit = load_plan.visible_ball_limit;
        self.adaptive_live_ball_budget = load_plan.live_ball_budget;
        self.adaptive_low_poly_override = load_plan.force_low_poly_ball_mesh;

        if let Some(cap) = self.adaptive_live_ball_budget {
            let target = runtime_patch
                .target_ball_count
                .unwrap_or(self.scene.config().target_ball_count)
                .min(cap);
            runtime_patch.target_ball_count = Some(target);
        }
        runtime_patch.spawn_per_tick = Some(
            runtime_patch
                .spawn_per_tick
                .unwrap_or(load_plan.spawn_per_tick_cap)
                .min(load_plan.spawn_per_tick_cap),
        );

        if self.last_substeps + 1 >= self.max_substeps && live_balls > 1_500 {
            runtime_patch.spawn_per_tick =
                Some(runtime_patch.spawn_per_tick.unwrap_or(96).clamp(32, 96));
            runtime_patch.linear_damping =
                Some(runtime_patch.linear_damping.unwrap_or(0.016).max(0.018));
        }
        if !parallel_ready && live_balls > 2_500 {
            runtime_patch.spawn_per_tick =
                Some(runtime_patch.spawn_per_tick.unwrap_or(64).clamp(24, 64));
            runtime_patch.linear_damping =
                Some(runtime_patch.linear_damping.unwrap_or(0.018).max(0.020));
        }
        let mut force_full_fbx_runtime = force_full_fbx;
        if self.adaptive_low_poly_override {
            // On mobile/TBDR profiles, visual fallback is cheaper than frame-time collapse.
            runtime_patch.ball_mesh_slot = Some(AUTO_LOW_POLY_BALL_SLOT);
            force_full_fbx_runtime = false;
        }
        self.renderer
            .set_force_full_fbx_sphere(force_full_fbx_runtime);

        if let Some(cap) = self.adaptive_live_ball_budget {
            let _ = self.scene.enforce_live_ball_budget(&mut *self.world.borrow_mut(), cap);
        }

        self.tick_retune_timer -= sim_dt;
        if self.tick_retune_timer <= 0.0 {
            self.max_substeps = self
                .manual_max_substeps
                .unwrap_or_else(|| load_plan.max_substeps.max(2));
            let mut desired_hz = choose_aggressive_tick_hz(
                self.tick_policy,
                self.tick_profile,
                self.fps_tracker.ema_fps().max(1.0),
                parallel_ready,
                self.last_substeps,
                self.max_substeps,
                live_balls,
                load_plan.tick_scale,
                self.fps_limit_hint,
                self.mps_logical_threads,
                mobile_path,
            );
            if mobile_path {
                let mobile_ceiling = match self.tick_profile {
                    TickProfile::Balanced => 120.0,
                    TickProfile::Max => 160.0,
                };
                desired_hz = desired_hz.min(mobile_ceiling);
            }
            // Avoid fixed-step overload: if tick is too high for current FPS and max_substeps,
            // simulation falls behind (slow-motion). Clamp to catch-up-safe frequency.
            let catch_up_factor = if mobile_path { 0.78 } else { 0.88 };
            let catch_up_hz =
                (self.fps_tracker.ema_fps().max(1.0) * self.max_substeps as f32 * catch_up_factor)
                    .clamp(24.0, 900.0);
            desired_hz = desired_hz.min(catch_up_hz);
            let ramp_up = desired_hz > self.tick_hz;
            let smoothing = if mobile_path {
                match (self.tick_profile, ramp_up) {
                    (TickProfile::Max, true) => 0.48,
                    (TickProfile::Max, false) => 0.30,
                    (_, true) => 0.42,
                    (_, false) => 0.26,
                }
            } else {
                match (self.tick_profile, ramp_up) {
                    (TickProfile::Max, true) => 0.86,
                    (TickProfile::Max, false) => 0.55,
                    (_, true) => 0.72,
                    (_, false) => 0.42,
                }
            };
            self.tick_hz = smooth_tick_hz(self.tick_hz, desired_hz, smoothing);
            let hard_floor = match self.tick_profile {
                TickProfile::Balanced => 35.0,
                TickProfile::Max => {
                    let ema_floor = if mobile_path {
                        (self.fps_tracker.ema_fps().max(1.0) * 1.45).clamp(28.0, 84.0)
                    } else {
                        (self.fps_tracker.ema_fps().max(1.0) * 6.0).clamp(45.0, 180.0)
                    };
                    let cap_floor = if mobile_path {
                        if self.uncapped_dynamic_fps_hint {
                            self.fps_limit_hint * 0.26
                        } else {
                            self.fps_limit_hint * 0.32
                        }
                    } else if self.uncapped_dynamic_fps_hint {
                        self.fps_limit_hint * 0.45
                    } else {
                        self.fps_limit_hint * 0.60
                    };
                    ema_floor.min(cap_floor.clamp(32.0, 220.0))
                }
            };
            let floor_hz = hard_floor.min(catch_up_hz * 0.90).max(24.0);
            self.tick_hz = self.tick_hz.max(floor_hz).min(catch_up_hz);
            self.world
                .borrow_mut()
                .set_timestep(1.0 / self.tick_hz, self.max_substeps);
            self.tick_retune_timer = if mobile_path {
                if ramp_up {
                    0.12
                } else {
                    0.08
                }
            } else if ramp_up {
                0.08
            } else {
                0.16
            };
        }

        self.script_frame_index = self.script_frame_index.saturating_add(1);
        let allow_physics_step = if self.simulation_paused {
            if self.simulation_step_budget > 0 {
                self.simulation_step_budget = self.simulation_step_budget.saturating_sub(1);
                true
            } else {
                false
            }
        } else {
            true
        };

        // ── Wait for the physics step submitted during last frame's GPU upload ──
        // On the very first frame physics_token is None, so substeps=0 and we
        // start with the world's initial state (empty scene).  From frame 2
        // onward the token is always present and we block here only for the
        // tail of the step that outlasted the GPU upload window.
        let t_phys_begin = Instant::now();
        let substeps = if let Some(token) = self.physics_token.take() {
            let s = token.wait();
            let _ = self.scene.reconcile_after_step(&mut *self.world.borrow_mut());
            self.last_substeps = s;
            s
        } else {
            self.last_substeps = 0;
            0
        };
        // Tick metrics were captured in the previous frame's physics_tick call.
        let tick = self.last_tick;

        // ── Build frame instances from the just-reconciled physics state ─────
        let t_scene_begin = Instant::now();
        let pre_phys_us = (t_phys_begin - frame_begin).as_micros() as u64;
        let mut frame = {
            let w = self.world.borrow();
            self.scene.build_frame_instances_with_ball_limit(
                &*w,
                Some(w.interpolation_alpha()),
                self.adaptive_ball_render_limit,
            )
        };
        let unknown_light_overrides =
            apply_scene_light_overrides(&mut frame, frame_eval.light_overrides.as_slice());
        // Apply global ball material overrides from tlscript.
        if frame_eval.ball_metallic.is_some() || frame_eval.ball_roughness.is_some() {
            for instance in frame
                .opaque_3d
                .iter_mut()
                .filter(|i| matches!(i.primitive, ScenePrimitive3d::Sphere))
            {
                if let Some(m) = frame_eval.ball_metallic {
                    instance.material.metallic = m.clamp(0.0, 1.0);
                }
                if let Some(r) = frame_eval.ball_roughness {
                    instance.material.roughness = r.clamp(0.0, 1.0);
                }
            }
        }
        // Follow-camera lights: move to camera eye and look along camera forward vector.
        // Offset the light slightly below the eye (like a flashlight held at chest level)
        // so that shadow-casting objects create visible shadows on surfaces behind them.
        // Without the offset, light == camera means every shadow falls exactly behind the
        // caster and is therefore occluded from the camera's view.
        let cam_forward = self.camera.forward_vector();
        let cam_right = cam_forward.cross(&nalgebra::Vector3::y());
        let cam_right = if cam_right.norm() > 1e-4 {
            cam_right.normalize()
        } else {
            nalgebra::Vector3::x()
        };
        let cam_down = cam_forward.cross(&cam_right);
        for light in frame.lights.iter_mut().filter(|l| l.follow_camera) {
            light.position = [
                eye[0] + cam_down.x * 1.5,
                eye[1] + cam_down.y * 1.5,
                eye[2] + cam_down.z * 1.5,
            ];
            light.direction = [cam_forward.x, cam_forward.y, cam_forward.z];
        }
        let light_pruned = clamp_scene_lights_for_camera(&mut frame, eye, MAX_SCENE_LIGHTS);
        if unknown_light_overrides > 0 && self.script_frame_index % 180 == 0 {
            eprintln!(
                "[tlscript light] {} unknown light id override(s) were ignored",
                unknown_light_overrides
            );
        }
        if light_pruned > 0 && self.script_frame_index % 180 == 0 {
            eprintln!(
                "[scene lights] pruned {} light(s) to MAX_SCENE_LIGHTS={}",
                light_pruned, MAX_SCENE_LIGHTS
            );
        }
        // ── Dynamo FSR: update render scale based on nearest visible object ────
        // Runs after frame instances are built so we have the full visible set.
        // Uses the nearest opaque instance as the proximity signal; if the scene
        // is empty or FSR mode is not Dynamo, this block is a no-op.
        if self.fsr_config.mode == FsrMode::Dynamo {
            let nearest_sq = frame
                .opaque_3d
                .iter()
                .map(|inst| {
                    let t = inst.transform.translation;
                    let dx = t[0] - eye[0];
                    let dy = t[1] - eye[1];
                    let dz = t[2] - eye[2];
                    dx * dx + dy * dy + dz * dz
                })
                .fold(f32::MAX, f32::min);
            let nearest_m = if nearest_sq < f32::MAX {
                nearest_sq.sqrt()
            } else {
                self.fsr_dynamo_config.far_m
            };
            let target = self.fsr_dynamo_config.scale_for_distance(nearest_m);
            let s = self.fsr_dynamo_config.smoothing.clamp(0.0, 0.99);
            self.fsr_dynamo_scale = self.fsr_dynamo_scale * s + target * (1.0 - s);
            let dynamo_scale = (self.fsr_dynamo_scale * 100.0).round() / 100.0; // snap to 0.01 grid
            if (dynamo_scale - self.fsr_config.render_scale_override.unwrap_or(-1.0)).abs() > 1e-3
            {
                self.fsr_config.render_scale_override = Some(dynamo_scale);
                self.renderer
                    .set_fsr_config(&self.queue, self.fsr_config);
            }
        }

        let distance_stats = apply_render_distance_haze(
            &mut frame,
            eye,
            self.render_distance,
            self.distance_blur_enabled,
        );
        self.last_distance_culled = distance_stats.culled;
        self.last_distance_blurred = distance_stats.blurred;
        let fill_ratio = estimate_framebuffer_fill_ratio(
            &frame,
            self.size.width.max(1),
            self.size.height.max(1),
        );
        self.last_framebuffer_fill_ratio = fill_ratio;
        if self.framebuffer_fill_ema <= f32::EPSILON {
            self.framebuffer_fill_ema = fill_ratio;
        } else {
            self.framebuffer_fill_ema += (fill_ratio - self.framebuffer_fill_ema) * 0.18;
        }
        self.retune_adaptive_pacer(sim_dt, mobile_path);
        let visible_ball_count = count_rendered_balls(&frame);
        let _hud = self.hud.append_to_sprites(
            TelemetryHudSample {
                fps: self.fps_tracker.ema_fps(),
                frame_time_ms: raw_dt * 1_000.0,
                physics_substeps: substeps,
                live_balls: tick.live_balls,
                draw_calls: frame.opaque_3d.len()
                    + frame.transparent_3d.len()
                    + frame.sprites.len(),
                rt_mode: self.rt_mode,
                rt_active: self.renderer.ray_tracing_status().active,
                rt_dynamic_count: self.renderer.ray_tracing_status().rt_dynamic_count,
                rt_fallback: !self
                    .renderer
                    .ray_tracing_status()
                    .fallback_reason
                    .is_empty(),
            },
            &mut frame.sprites,
        );
        // ── Light glow billboards ──────────────────────────────────────────────
        // For each enabled scene light, project its world position to NDC and
        // emit a LightGlow sprite. These are rendered with additive blending so
        // multiple overlapping glows accumulate naturally.
        for light in frame
            .lights
            .iter()
            .filter(|l| l.enabled && l.glow_enabled && l.intensity > 0.01)
        {
            if let Some(ndc) = self.renderer.world_to_ndc(light.position) {
                // ndc = [x, y, depth]; clip.w approximated from camera distance.
                let cam = self.renderer.camera_eye();
                let dx = light.position[0] - cam[0];
                let dy = light.position[1] - cam[1];
                let dz = light.position[2] - cam[2];
                let dist = (dx * dx + dy * dy + dz * dz).sqrt().max(0.1);
                // World radius: prefer per-light config, fall back to intensity-derived heuristic.
                let world_r = if (light.glow_radius_world - 1.2).abs() > 1e-3 {
                    light.glow_radius_world
                } else {
                    (light.intensity.sqrt() * 0.6 + light.range * 0.04).clamp(0.1, 8.0)
                };
                let half = self.renderer.world_radius_to_ndc_half_size(world_r, dist);
                let intensity_scale =
                    (light.intensity / 8.0).clamp(0.1, 1.5) * light.glow_intensity_scale;
                frame.sprites.push(SpriteInstance {
                    sprite_id: light.id,
                    kind: SpriteKind::LightGlow,
                    position: [ndc[0], ndc[1], ndc[2].clamp(0.0, 1.0)],
                    size: [half * 2.0, half * 2.0],
                    rotation_rad: 0.0,
                    color_rgba: [
                        light.color[0] * intensity_scale,
                        light.color[1] * intensity_scale,
                        light.color[2] * intensity_scale,
                        1.0,
                    ],
                    texture_slot: 0,
                    layer: 100,
                });
            }
        }

        self.console_overlay_sprites.clear();
        let console_layout = ConsoleUiLayout::from_size(self.size);
        Self::append_console_overlay_sprites(
            &self.console,
            console_layout,
            &mut self.console_overlay_sprites,
        );
        let t_compile_begin = Instant::now();
        let scene_us = (t_compile_begin - t_scene_begin).as_micros() as u64;
        let draw = self.draw_compiler.compile(&frame);
        let t_upload_begin = Instant::now();
        let compile_us = (t_upload_begin - t_compile_begin).as_micros() as u64;

        // ── Submit next physics step before GPU upload ─────────────────────────
        // apply_runtime_patch + physics_tick prepare the world for the step,
        // then step_begin submits world.step(dt) to an MPS Critical worker.
        // The step runs concurrently with upload_draw_frame (~9 ms), hiding
        // the ~2 ms physics cost behind the already-serialised GPU work.
        if allow_physics_step {
            {
                let mut w = self.world.borrow_mut();
                let _patch_metrics = self
                    .scene
                    .apply_runtime_patch(&mut *w, runtime_patch);
                self.last_tick = self.scene.physics_tick(&mut *w);
                self.script_last_spawned = self.last_tick.spawned_this_tick;
            }
            let step_dt = if self.simulation_paused {
                self.world.borrow().config().fixed_dt
            } else {
                sim_dt
            };
            self.physics_token = Some(self.world.step_begin(step_dt));
        } else {
            self.script_last_spawned = 0;
            self.last_tick = BounceTankTickMetrics {
                spawned_this_tick: 0,
                scattered_this_tick: 0,
                live_balls: self.scene.live_ball_count(),
                target_balls: self.scene.config().target_ball_count,
                fully_spawned: self.scene.live_ball_count()
                    >= self.scene.config().target_ball_count,
            };
        }

        let upload = self
            .renderer
            .upload_draw_frame(&self.device, &self.queue, &draw);
        self.renderer.upload_overlay_sprites(
            &self.device,
            &self.queue,
            self.console_overlay_sprites.as_slice(),
        );
        let rt_status = self.renderer.ray_tracing_status();
        let fsr_status = self.renderer.fsr_status();

        let output = match self.surface.get_current_texture() {
            Ok(frame) => frame,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                self.surface.configure(&self.device, &self.config);
                return Ok(());
            }
            Err(wgpu::SurfaceError::OutOfMemory) => {
                return Err("wgpu surface out of memory".into());
            }
            Err(wgpu::SurfaceError::Timeout) => {
                return Ok(());
            }
            Err(wgpu::SurfaceError::Other) => {
                return Ok(());
            }
        };

        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("tlapp-encoder"),
            });
        self.renderer.encode(
            &mut encoder,
            &view,
            wgpu::Color {
                r: 0.07,
                g: 0.09,
                b: 0.12,
                a: 1.0,
            },
        );
        self.renderer.encode_overlay_sprites(&mut encoder, &view);
        self.queue.submit(Some(encoder.finish()));
        let t_present_begin = Instant::now();
        let upload_us = (t_present_begin - t_upload_begin).as_micros() as u64;
        output.present();

        let frame_end = Instant::now();
        let present_us = (frame_end - t_present_begin).as_micros() as u64;
        let frame_time = (frame_end - frame_begin).as_secs_f32();
        let report = self.fps_tracker.record(frame_end, frame_time);
        if self.uncapped_dynamic_fps_hint {
            let measured_hint = self.fps_tracker.dynamic_uncapped_fps_hint();
            let upper_hint = if mobile_path { 144.0 } else { 1_200.0 };
            self.fps_limit_hint =
                smooth_tick_hz(self.fps_limit_hint, measured_hint, 0.22).clamp(48.0, upper_hint);
        }
        let pacing_suffix = self
            .frame_cap_interval
            .map(|d| format!(" | cap {:.0}", 1.0 / d.as_secs_f32().max(1e-6)))
            .unwrap_or_else(|| {
                if self.uncapped_dynamic_fps_hint {
                    format!(" | target {:.0}", self.fps_limit_hint)
                } else {
                    String::new()
                }
            });
        let scheduler_label = scheduler_path_label(self.scheduler_path);
        let distance_suffix = match self.render_distance {
            Some(distance) => format!(
                " | rd {:.0}m c{} b{} fill {:.2}",
                distance,
                self.last_distance_culled,
                self.last_distance_blurred,
                self.last_framebuffer_fill_ratio
            ),
            None => String::new(),
        };
        let step_timings = self.world.borrow().last_step_timings;
        let console_suffix = self.console_title_suffix();
        let title = format!(
            "Tileline TLApp | FPS {:.1} | Frame {:.2} ms | Tick {:.0} Hz | Balls {} (draw {}) | Lights {} | RT {:?}/{} ({}) | FSR {:?}/{} ({:.2}) | Substeps {} | Phys {}µs (int {}µs bp {}µs np {}µs sv {}µs sl {}µs) | {:?} {} {:?}{}{}{}{}{}{}",
            self.fps_tracker.ema_fps(),
            frame_time * 1_000.0,
            self.tick_hz,
            tick.live_balls,
            visible_ball_count,
            upload.light_count,
            self.rt_mode,
            if rt_status.active { "on" } else { "off" },
            rt_status.rt_dynamic_count,
            fsr_status.requested_mode,
            if fsr_status.active { "on" } else { "off" },
            fsr_status.render_scale,
            substeps,
            step_timings.total_us(),
            step_timings.integrate_us,
            step_timings.broadphase_us,
            step_timings.narrowphase_us,
            step_timings.solver_us,
            step_timings.sleep_us,
            self.adapter_backend,
            scheduler_label,
            self.present_mode,
            if self.scheduler_fallback_applied {
                " | fallback"
            } else {
                ""
            },
            if self.adaptive_low_poly_override {
                " | lowpoly"
            } else {
                ""
            },
            if tick.scattered_this_tick > 0 {
                format!(" | scatter {}", tick.scattered_this_tick)
            } else {
                String::new()
            },
            distance_suffix,
            pacing_suffix,
            console_suffix,
        );
        self.window.set_title(&title);

        if let Some(report) = report {
            let w = self.world.borrow();
            let broadphase = w.broadphase().stats();
            let narrowphase = w.narrowphase().stats();
            println!(
                "tlapp fps | inst: {:>6.1} | ema: {:>6.1} | avg: {:>6.1} | stddev: {:>5.2} ms | balls: {:>5} | draw: {:>5} | lights: {:>2} | substeps: {} | phys_us: {:>6} | int_us: {:>5} | bp_us: {:>5} | np_us: {:>5} | sv_us: {:>5} | sl_us: {:>5} | snap_us: {:>5} | pre_phys_us: {:>5} | scene_us: {:>5} | compile_us: {:>5} | upload_us: {:>5} | present_us: {:>6} | scattered: {:>4} | rd_culled: {:>4} | rd_blur: {:>4} | fill: {:>4.2} | fill_ema: {:>4.2} | rt_mode: {:?} | rt_active: {} | rt_dynamic: {:>4} | rt_reason: {} | fsr_mode: {:?} | fsr_active: {} | fsr_scale: {:>4.2} | fsr_sharpness: {:>4.2} | fsr_reason: {} | mps_threads: {} | shards: {} | pairs: {} | manifolds: {} | platform: {:?} | backend: {:?} | scheduler: {} | present: {:?} | fallback: {} | adapter: {} | reason: {}",
                report.instant_fps,
                report.ema_fps,
                report.avg_fps,
                report.frame_time_stddev_ms,
                tick.live_balls,
                visible_ball_count,
                upload.light_count,
                substeps,
                step_timings.total_us(),
                step_timings.integrate_us,
                step_timings.broadphase_us,
                step_timings.narrowphase_us,
                step_timings.solver_us,
                step_timings.sleep_us,
                step_timings.snapshot_us,
                pre_phys_us,
                scene_us,
                compile_us,
                upload_us,
                present_us,
                tick.scattered_this_tick,
                self.last_distance_culled,
                self.last_distance_blurred,
                self.last_framebuffer_fill_ratio,
                self.framebuffer_fill_ema,
                self.rt_mode,
                rt_status.active,
                rt_status.rt_dynamic_count,
                if rt_status.fallback_reason.is_empty() {
                    "none"
                } else {
                    rt_status.fallback_reason.as_str()
                },
                fsr_status.requested_mode,
                fsr_status.active,
                fsr_status.render_scale,
                fsr_status.sharpness,
                if fsr_status.reason.is_empty() {
                    "none"
                } else {
                    fsr_status.reason.as_str()
                },
                self.mps_logical_threads,
                broadphase.shard_count,
                broadphase.candidate_pairs,
                narrowphase.manifolds,
                self.platform,
                self.adapter_backend,
                scheduler_label,
                self.present_mode,
                self.scheduler_fallback_applied,
                self.adapter_name,
                self.scheduler_reason
            );
        }

        Ok(())
    }
}
