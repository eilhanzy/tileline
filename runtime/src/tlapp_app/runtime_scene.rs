use super::*;

impl TlAppRuntime {
    pub(super) fn prepare_for_exit(&mut self) {
        if self.shutdown_prepared {
            return;
        }
        self.shutdown_prepared = true;
        // Drain any pending async physics step before shutting down.
        if let Some(token) = self.physics_token.take() {
            token.wait();
        }
        self.console.open = false;
        self.mouse_look_held = false;
        self.look_lock_active = false;
        self.keyboard_camera = CameraInputState::default();
        self.script_key_f_keyboard = false;
        self.script_key_g_keyboard = false;
        self.console_overlay_sprites.clear();
        self.frame_cap_interval = None;
        self.adaptive_pacer_enabled = false;
        self.queue.submit(std::iter::empty());
        if let Err(err) = self.device.poll(wgpu::PollType::wait_indefinitely()) {
            eprintln!("[shutdown] device poll failed: {err}");
        }
    }

    pub(super) fn reset_simulation_state(&mut self) {
        // Drain any in-flight physics step before replacing the world.
        if let Some(token) = self.physics_token.take() {
            token.wait();
        }
        self.last_tick = BounceTankTickMetrics {
            spawned_this_tick: 0,
            scattered_this_tick: 0,
            live_balls: 0,
            target_balls: 0,
            fully_spawned: false,
        };
        let world_config = self.world.borrow().config().clone();
        let scene_config = self.scene.config();
        let sprite_program = self.scene.sprite_program_cloned();
        let ball_mesh_slot = self.scene.ball_mesh_slot();
        let container_mesh_slot = self.scene.container_mesh_slot();
        self.world.replace(PhysicsWorld::new(world_config));
        self.scene = BounceTankSceneController::new(scene_config);
        if let Some(program) = sprite_program {
            self.scene.set_sprite_program(program);
        }
        if ball_mesh_slot.is_some() || container_mesh_slot.is_some() {
            let _ = self.scene.apply_runtime_patch(
                &mut *self.world.borrow_mut(),
                BounceTankRuntimePatch {
                    ball_mesh_slot,
                    container_mesh_slot,
                    ..BounceTankRuntimePatch::default()
                },
            );
        }
        self.script_last_spawned = 0;
        self.script_frame_index = 0;
        self.last_substeps = 0;
        self.simulation_step_budget = 0;
        self.tick_retune_timer = 0.0;
        self.distance_retune_timer = 0.0;
        self.adaptive_load_pressure_ema = 0.0;
        if let Some(bridge) = self.runtime_bridge.as_ref() {
            self.runtime_bridge_metrics = bridge.metrics();
            self.runtime_bridge_telemetry = RuntimeBridgeTelemetry::new(bridge.path());
        } else {
            self.runtime_bridge_telemetry.queued_plan_depth = 0;
            self.runtime_bridge_telemetry.bridge_pump_published = 0;
            self.runtime_bridge_telemetry.bridge_pump_drained = 0;
            self.runtime_bridge_telemetry.physics_lag_frames = 0;
            self.runtime_bridge_telemetry.latest_plan_frame_id = None;
            self.runtime_bridge_telemetry.latest_submission_frame_id = None;
            self.runtime_bridge_telemetry.latest_submission_tasks = 0;
            self.runtime_bridge_telemetry.used_fallback_plan = false;
            self.runtime_bridge_telemetry.latest_plan_kind = "none";
        }
    }

    pub(super) fn apply_bundle_sprite_program(
        &mut self,
        program: Option<TlspriteProgram>,
        root_hint: Option<&Path>,
    ) {
        if let Some(program) = program {
            self.force_full_fbx_from_sprite = program.requires_full_fbx_render();
            self.scene.set_sprite_program(program.clone());
            if let Some(root) = root_hint {
                bind_renderer_meshes_from_tlsprite(
                    &mut self.renderer,
                    &self.device,
                    root,
                    &program,
                );
            }
        } else {
            self.force_full_fbx_from_sprite = false;
            self.scene.clear_sprite_program();
        }
        self.renderer
            .set_force_full_fbx_sphere(self.force_full_fbx_from_sprite);
    }

    pub(super) fn reload_sprite_from_watcher(&mut self) -> Result<String, String> {
        let (Some(sprite_loader), Some(sprite_cache)) =
            (self.sprite_loader.as_mut(), self.sprite_cache.as_mut())
        else {
            return Err("sprite watcher is not active in this runtime mode".to_string());
        };
        let event = sprite_loader.reload_into_cache(sprite_cache);
        let event_note = format!("{event:?}");
        print_tlsprite_event("[tlsprite manual reload]", event);
        if let Some(program) = sprite_cache.program_for_path(sprite_loader.path()).cloned() {
            self.force_full_fbx_from_sprite = program.requires_full_fbx_render();
            self.scene.set_sprite_program(program.clone());
            bind_renderer_meshes_from_tlsprite(
                &mut self.renderer,
                &self.device,
                sprite_loader.path(),
                &program,
            );
            self.renderer
                .set_force_full_fbx_sphere(self.force_full_fbx_from_sprite);
            Ok(format!(
                "sprite reloaded from '{}' ({event_note})",
                sprite_loader.path().display()
            ))
        } else {
            Err(format!(
                "sprite reload did not produce a compiled program ({event_note})"
            ))
        }
    }

    pub(super) fn reload_script_runtime_from_sources(
        &mut self,
        include_bundle_sprite: bool,
    ) -> Result<String, String> {
        if let Some(project_path) = &self.cli_options.project_path {
            let compile = compile_tlpfile_scene_from_path(
                project_path,
                Some(&self.cli_options.joint_scene),
                TlscriptShowcaseConfig::default(),
            );
            let warning_count = compile
                .diagnostics
                .iter()
                .filter(|it| matches!(it.level, TlpfileDiagnosticLevel::Warning))
                .count();
            if compile.has_errors() {
                let first_error = compile
                    .diagnostics
                    .iter()
                    .find(|it| matches!(it.level, TlpfileDiagnosticLevel::Error))
                    .map(|it| it.message.as_str())
                    .unwrap_or("unknown project compile error");
                return Err(format!(
                    "project reload failed for '{}': {}",
                    project_path.display(),
                    first_error
                ));
            }
            let bundle = compile.bundle.ok_or_else(|| {
                format!(
                    "project reload produced no bundle for '{}'",
                    project_path.display()
                )
            })?;
            let scene_name = bundle.scene_name.clone();
            let script_count = bundle.scripts.len();
            let sprite_count = bundle.sprite_count();
            let merged_sprite_program = bundle.merged_sprite_program.clone();
            let sprite_root = bundle
                .selected_joint_path
                .clone()
                .or_else(|| Some(bundle.project_path.clone()));
            self.script_runtime = ScriptRuntime::MultiScripts(bundle.scripts);
            if include_bundle_sprite {
                self.apply_bundle_sprite_program(merged_sprite_program, sprite_root.as_deref());
            }
            return Ok(format!(
                "project reloaded scene='{}' scripts={} sprites={} warnings={}",
                scene_name, script_count, sprite_count, warning_count
            ));
        }

        if let Some(joint_path) = &self.cli_options.joint_path {
            let compile = compile_tljoint_scene_from_path(
                joint_path,
                &self.cli_options.joint_scene,
                TlscriptShowcaseConfig::default(),
            );
            let warning_count = compile
                .diagnostics
                .iter()
                .filter(|it| matches!(it.level, TljointDiagnosticLevel::Warning))
                .count();
            if compile.has_errors() {
                let first_error = compile
                    .diagnostics
                    .iter()
                    .find(|it| matches!(it.level, TljointDiagnosticLevel::Error))
                    .map(|it| it.message.as_str())
                    .unwrap_or("unknown joint compile error");
                return Err(format!(
                    "joint reload failed for '{}': {}",
                    joint_path.display(),
                    first_error
                ));
            }
            let bundle = compile.bundle.ok_or_else(|| {
                format!(
                    "joint reload produced no bundle for '{}'",
                    joint_path.display()
                )
            })?;
            let scene_name = bundle.scene_name.clone();
            let script_count = bundle.scripts.len();
            let sprite_count = bundle.sprite_paths.len();
            let merged_sprite_program = bundle.merged_sprite_program.clone();
            let sprite_root = Some(bundle.manifest_path.clone());
            self.script_runtime = ScriptRuntime::Joint(bundle);
            if include_bundle_sprite {
                self.apply_bundle_sprite_program(merged_sprite_program, sprite_root.as_deref());
            }
            return Ok(format!(
                "joint reloaded scene='{}' scripts={} sprites={} warnings={}",
                scene_name, script_count, sprite_count, warning_count
            ));
        }

        let script_source_owned =
            fs::read_to_string(&self.cli_options.script_path).map_err(|err| {
                format!(
                    "failed to read script '{}': {err}",
                    self.cli_options.script_path.display()
                )
            })?;
        let script_source: &'static str = Box::leak(script_source_owned.into_boxed_str());
        let compile = compile_tlscript_showcase(script_source, TlscriptShowcaseConfig::default());
        if !compile.errors.is_empty() {
            return Err(format!(
                "script reload failed for '{}': {}",
                self.cli_options.script_path.display(),
                compile.errors.join(" | ")
            ));
        }
        let warnings = compile.warnings.len();
        let program = compile
            .program
            .ok_or_else(|| "script reload produced no runnable program".to_string())?;
        self.script_runtime = ScriptRuntime::Single(program);
        Ok(format!(
            "script reloaded '{}' warnings={}",
            self.cli_options.script_path.display(),
            warnings
        ))
    }

    pub(super) fn apply_gfx_profile(&mut self, profile: &str) -> Result<String, String> {
        let mobile_path =
            matches!(self.platform, RuntimePlatform::Android) || self.mgs_is_mobile_hardware;
        match profile {
            "low" => {
                self.fsr_config.mode = FsrMode::On;
                self.fsr_config.quality = FsrQualityPreset::Performance;
                self.fsr_config.sharpness = 0.42;
                self.fsr_config.render_scale_override = None;
                self.render_distance = Some(if mobile_path { 36.0 } else { 48.0 });
                self.adaptive_distance_enabled = true;
                self.distance_blur_mode = ToggleAuto::On;
                self.tick_profile = TickProfile::Balanced;
            }
            "med" | "medium" => {
                self.fsr_config.mode = FsrMode::Auto;
                self.fsr_config.quality = FsrQualityPreset::Balanced;
                self.fsr_config.sharpness = 0.36;
                self.fsr_config.render_scale_override = None;
                self.render_distance = Some(if mobile_path { 56.0 } else { 76.0 });
                self.adaptive_distance_enabled = true;
                self.distance_blur_mode = ToggleAuto::Auto;
                self.tick_profile = TickProfile::Balanced;
            }
            "high" => {
                self.fsr_config.mode = FsrMode::Auto;
                self.fsr_config.quality = FsrQualityPreset::Quality;
                self.fsr_config.sharpness = 0.33;
                self.fsr_config.render_scale_override = None;
                self.render_distance = Some(if mobile_path { 72.0 } else { 104.0 });
                self.adaptive_distance_enabled = true;
                self.distance_blur_mode = ToggleAuto::Off;
                self.tick_profile = TickProfile::Max;
            }
            "ultra" => {
                self.fsr_config.mode = FsrMode::Off;
                self.fsr_config.quality = FsrQualityPreset::Native;
                self.fsr_config.sharpness = 0.28;
                self.fsr_config.render_scale_override = Some(1.0);
                self.render_distance = None;
                self.adaptive_distance_enabled = false;
                self.distance_blur_mode = ToggleAuto::Off;
                self.tick_profile = TickProfile::Max;
            }
            _ => {
                return Err("usage: gfx.profile <low|med|high|ultra>".to_string());
            }
        }

        if let Some(distance) = self.render_distance {
            self.render_distance_min = (distance * 0.72).clamp(28.0, distance);
            self.render_distance_max = (distance * 1.55).clamp(distance, 260.0);
        } else {
            self.render_distance_min = 0.0;
            self.render_distance_max = 0.0;
        }
        self.distance_blur_enabled = self.distance_blur_mode.resolve(mobile_path);
        self.renderer.set_fsr_config(&self.queue, self.fsr_config);
        self.sync_console_quick_fields_from_runtime();

        let distance_label = self
            .render_distance
            .map(|v| format!("{v:.1}"))
            .unwrap_or_else(|| "off".to_string());
        Ok(format!(
            "gfx profile '{}' applied: fsr={:?}/{:?} sharpness={:.2} distance={} blur={:?} tick_profile={:?}",
            profile,
            self.fsr_config.mode,
            self.fsr_config.quality,
            self.fsr_config.sharpness,
            distance_label,
            self.distance_blur_mode,
            self.tick_profile
        ))
    }

    pub(super) fn apply_performance_contract_preset(
        &mut self,
        scenario: PerformanceContractScenario,
    ) -> Result<String, String> {
        let mobile_path =
            matches!(self.platform, RuntimePlatform::Android) || self.mgs_is_mobile_hardware;
        let gfx_profile = scenario.recommended_gfx_profile(mobile_path);
        let spawn_per_tick =
            scenario.recommended_spawn_per_tick(self.mps_logical_threads, mobile_path);
        let target_ball_count = scenario.reference_balls();

        let gfx_note = self.apply_gfx_profile(gfx_profile)?;
        {
            let mut world = self.world.borrow_mut();
            let _ = self.scene.apply_runtime_patch(
                &mut *world,
                BounceTankRuntimePatch {
                    target_ball_count: Some(target_ball_count),
                    spawn_per_tick: Some(spawn_per_tick),
                    ..BounceTankRuntimePatch::default()
                },
            );
        }

        self.manual_max_substeps = None;
        self.tick_retune_timer = 0.0;
        self.distance_retune_timer = 0.0;
        self.adaptive_load_pressure_ema = 0.0;
        self.adaptive_ball_render_limit = None;
        self.adaptive_live_ball_budget = None;
        self.adaptive_low_poly_override = false;
        self.reset_simulation_state();
        self.world
            .borrow_mut()
            .set_timestep(1.0 / self.tick_hz.max(1.0), self.max_substeps);
        self.sync_console_quick_fields_from_runtime();

        Ok(format!(
            "perf preset {} applied: target_balls={} spawn_per_tick={} gfx={} auto_substeps=on reset=yes | {}",
            scenario.label(),
            target_ball_count,
            spawn_per_tick,
            gfx_profile,
            gfx_note
        ))
    }
}
