use super::*;

impl TlAppRuntime {
    pub(super) fn new(
        event_loop: &ActiveEventLoop,
        mut options: CliOptions,
        platform: RuntimePlatform,
    ) -> Result<Self, Box<dyn Error>> {
        let file_io_root = env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .canonicalize()
            .unwrap_or_else(|_| PathBuf::from("."));
        let mut pak_mount_root: Option<PathBuf> = None;
        if let Some(requested_pak_path) = options.pak_path.clone() {
            let resolved_pak_path = resolve_path_from_root(&file_io_root, &requested_pak_path);
            let canonical_pak_path = resolved_pak_path.canonicalize().map_err(|err| {
                format!(
                    "failed to resolve pak '{}': {err}",
                    requested_pak_path.display()
                )
            })?;
            let mount_root = allocate_pak_mount_root(&file_io_root)?;
            let unpack_report =
                unpack_pak(&canonical_pak_path, &mount_root).map_err(|err| -> Box<dyn Error> {
                    let _ = fs::remove_dir_all(&mount_root);
                    format!(
                        "failed to mount pak '{}' into '{}': {err}",
                        canonical_pak_path.display(),
                        mount_root.display()
                    )
                    .into()
                })?;
            options.pak_path = Some(canonical_pak_path.clone());
            remap_cli_paths_from_pak_mount(&mut options, &mount_root);
            eprintln!(
                "[pak] mounted '{}' -> '{}' files={} bytes={}",
                canonical_pak_path.display(),
                mount_root.display(),
                unpack_report.file_count,
                unpack_report.total_payload_bytes
            );
            pak_mount_root = Some(mount_root);
        }
        let window = Arc::new(
            event_loop.create_window(
                WindowAttributes::default()
                    .with_title("Tileline TLApp")
                    .with_inner_size(LogicalSize::new(
                        options.resolution.width as f64,
                        options.resolution.height as f64,
                    )),
            )?,
        );
        let size = window.inner_size();

        let adapter_bootstrap = request_adapter_with_platform_policy(&window, platform)?;
        let instance = adapter_bootstrap.instance;
        let mut surface = adapter_bootstrap.surface;
        let adapter = adapter_bootstrap.adapter;
        let adapter_info = adapter.get_info();
        eprintln!(
            "[runtime bootstrap] platform={:?} adapter='{}' backend={:?} note={}",
            platform, adapter_info.name, adapter_info.backend, adapter_bootstrap.bootstrap_note
        );

        let (mut required_limits, limit_clamp_report) =
            safe_default_required_limits_for_adapter(&adapter);
        if limit_clamp_report.any_clamped() {
            eprintln!(
                "[runtime limits] adapter='{}' clamped required limits to supported values (1d={}, 2d={}, 3d={}, layers={})",
                adapter_info.name,
                required_limits.max_texture_dimension_1d,
                required_limits.max_texture_dimension_2d,
                required_limits.max_texture_dimension_3d,
                required_limits.max_texture_array_layers
            );
        }
        // Request RT feature if the adapter supports it; silently skip on non-RT hardware.
        let enabled_rt_features = wgpu::Features::EXPERIMENTAL_RAY_QUERY & adapter.features();
        eprintln!(
            "[runtime bootstrap] ray_query={} adapter='{}'",
            !enabled_rt_features.is_empty(),
            adapter_info.name,
        );

        // RT requires non-zero acceleration structure limits (all default to 0).
        if !enabled_rt_features.is_empty() {
            required_limits.max_acceleration_structures_per_shader_stage = 1;
            required_limits.max_blas_geometry_count = 1;
            required_limits.max_blas_primitive_count = 1024;
            // 16 384 covers 8 000 balls + tank walls with headroom to spare.
            required_limits.max_tlas_instance_count = 16_384;
        }

        // Only opt into experimental features when the adapter actually supports RT.
        // On non-RT hardware the unsafe token is unnecessary and would be misleading.
        let experimental_features = if enabled_rt_features.is_empty() {
            Default::default()
        } else {
            // SAFETY: EXPERIMENTAL_RAY_QUERY may have implementation bugs; we accept
            // this risk on hardware that reports the feature and fall back to PCF
            // shadow maps when the feature is absent.
            unsafe { wgpu::ExperimentalFeatures::enabled() }
        };
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("tlapp-device"),
                required_features: wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
                    | enabled_rt_features,
                required_limits,
                experimental_features,
                ..Default::default()
            }))?;

        let selected_present_mode = match options.vsync {
            VsyncMode::Auto => wgpu::PresentMode::AutoVsync,
            VsyncMode::On => wgpu::PresentMode::Fifo,
            VsyncMode::Off => wgpu::PresentMode::AutoNoVsync,
        };
        let mut config = if let Some(surface) = surface.as_ref() {
            let mut config = surface
                .get_default_config(&adapter, size.width.max(1), size.height.max(1))
                .ok_or("surface is not compatible with selected adapter")?;
            config.present_mode = selected_present_mode;
            surface.configure(&device, &config);
            Some(config)
        } else {
            None
        };
        let mut ensure_wgpu_surface = || -> Result<wgpu::TextureFormat, Box<dyn Error>> {
            if surface.is_none() {
                let new_surface = instance.create_surface(Arc::clone(&window))?;
                let mut new_config = new_surface
                    .get_default_config(&adapter, size.width.max(1), size.height.max(1))
                    .ok_or("surface is not compatible with selected adapter")?;
                new_config.present_mode = selected_present_mode;
                new_surface.configure(&device, &new_config);
                let format = new_config.format;
                surface = Some(new_surface);
                config = Some(new_config);
                Ok(format)
            } else {
                config
                    .as_ref()
                    .map(|config| config.format)
                    .ok_or_else(|| "wgpu surface exists without config".into())
            }
        };

        let mut script_runtime = None;
        let mut joint_merged_sprite_program: Option<TlspriteProgram> = None;
        let mut bundle_sprite_root: Option<PathBuf> = None;
        let mut project_scheduler_manifest: Option<TlpfileGraphicsScheduler> = None;
        let mut project_scene_dimension_manifest: Option<TlpfileSceneDimension> = None;
        if let Some(project_path) = &options.project_path {
            let project_compile = compile_tlpfile_scene_from_path(
                project_path,
                Some(&options.joint_scene),
                TlscriptShowcaseConfig::default(),
            );
            for diagnostic in &project_compile.diagnostics {
                match diagnostic.level {
                    TlpfileDiagnosticLevel::Warning => {
                        eprintln!("[tlpfile warning] {}", diagnostic.message);
                    }
                    TlpfileDiagnosticLevel::Error => {
                        eprintln!("[tlpfile error] {}", diagnostic.message);
                    }
                }
            }
            if project_compile.has_errors() {
                return Err(format!(
                    "failed to compile .tlpfile '{}' scene '{}'",
                    project_path.display(),
                    options.joint_scene
                )
                .into());
            }

            let bundle = project_compile.bundle.ok_or_else(|| {
                format!(
                    "no bundle returned for .tlpfile '{}' scene '{}'",
                    project_path.display(),
                    options.joint_scene
                )
            })?;

            project_scheduler_manifest = Some(bundle.scheduler);
            project_scene_dimension_manifest = Some(bundle.scene_dimension);

            if bundle.scene_name == "main" {
                let main_joint_ok = bundle
                    .selected_joint_path
                    .as_ref()
                    .and_then(|path| path.file_name())
                    .and_then(|name| name.to_str())
                    .map(|name| name == "main.tljoint")
                    .unwrap_or(false);
                if !main_joint_ok {
                    return Err(format!(
                        "scene 'main' in '{}' must resolve to main.tljoint",
                        project_path.display()
                    )
                    .into());
                }
            }

            let scene_name = bundle.scene_name.clone();
            let script_count = bundle.scripts.len();
            let sprite_count = bundle.sprite_count();
            let scheduler_name = bundle.scheduler.as_str();
            let scene_dimension = bundle.scene_dimension.as_str();
            let joint_label = bundle
                .selected_joint_path
                .as_ref()
                .map(|path| path.display().to_string())
                .unwrap_or_else(|| String::from("none"));
            bundle_sprite_root = bundle
                .selected_joint_path
                .clone()
                .or_else(|| Some(bundle.project_path.clone()));
            joint_merged_sprite_program = bundle.merged_sprite_program.clone();
            script_runtime = Some(ScriptRuntime::MultiScripts(bundle.scripts));
            eprintln!(
                "[tlpfile] project='{}' scene='{}' mode={} scheduler={} scripts={} sprites={} joint={}",
                project_path.display(),
                scene_name,
                scene_dimension,
                scheduler_name,
                script_count,
                sprite_count,
                joint_label
            );
        } else if let Some(joint_path) = &options.joint_path {
            let joint_compile = compile_tljoint_scene_from_path(
                joint_path,
                &options.joint_scene,
                TlscriptShowcaseConfig::default(),
            );
            for diagnostic in &joint_compile.diagnostics {
                match diagnostic.level {
                    TljointDiagnosticLevel::Warning => {
                        eprintln!("[tljoint warning] {}", diagnostic.message);
                    }
                    TljointDiagnosticLevel::Error => {
                        eprintln!("[tljoint error] {}", diagnostic.message);
                    }
                }
            }
            if joint_compile.has_errors() {
                return Err(format!(
                    "failed to compile .tljoint '{}' scene '{}'",
                    joint_path.display(),
                    options.joint_scene
                )
                .into());
            }
            let bundle = joint_compile.bundle.ok_or_else(|| {
                format!(
                    "no bundle returned for .tljoint '{}' scene '{}'",
                    joint_path.display(),
                    options.joint_scene
                )
            })?;
            bundle_sprite_root = Some(joint_path.clone());
            joint_merged_sprite_program = bundle.merged_sprite_program.clone();
            script_runtime = Some(ScriptRuntime::Joint(bundle));
            eprintln!(
                "[tljoint] active manifest='{}' scene='{}'",
                joint_path.display(),
                options.joint_scene
            );
        }
        if script_runtime.is_none() {
            let script_source_owned = fs::read_to_string(&options.script_path).map_err(|err| {
                format!(
                    "failed to read script '{}': {err}",
                    options.script_path.display()
                )
            })?;
            let script_source: &'static str = Box::leak(script_source_owned.into_boxed_str());
            let script_compile =
                compile_tlscript_showcase(script_source, TlscriptShowcaseConfig::default());
            for warning in &script_compile.warnings {
                eprintln!("[tlscript warning] {warning}");
            }
            if !script_compile.errors.is_empty() {
                return Err(format!(
                    "failed to compile script '{}': {}",
                    options.script_path.display(),
                    script_compile.errors.join(" | ")
                )
                .into());
            }
            let script_program = script_compile
                .program
                .expect("showcase script should compile without errors");
            script_runtime = Some(ScriptRuntime::Single(script_program));
        }

        let scheduler_resolution = if let Some(manifest_scheduler) = project_scheduler_manifest {
            resolve_project_scheduler(manifest_scheduler, platform, &adapter_info).map_err(
                |reason| {
                    format!(
                        "project scheduler rejected on '{}': {}",
                        adapter_info.name, reason
                    )
                },
            )?
        } else {
            let decision = choose_scheduler_path_for_platform(&adapter_info, platform);
            SchedulerResolution {
                selected: decision.path,
                fallback_applied: false,
                reason: decision.reason,
            }
        };
        eprintln!(
            "[scheduler] platform={:?} path={:?} fallback={} reason={}",
            platform,
            scheduler_resolution.selected,
            scheduler_resolution.fallback_applied,
            scheduler_resolution.reason
        );
        let mut pipeline_mode = options.pipeline_mode;
        if matches!(pipeline_mode, PipelineMode::Parallel)
            && !cfg!(feature = "parallel_pipeline_v2")
        {
            eprintln!(
                "[pipeline] requested parallel pipeline, but feature 'parallel_pipeline_v2' is disabled at build time; falling back to legacy"
            );
            pipeline_mode = PipelineMode::Legacy;
        }
        let fallback_bridge_path = match scheduler_resolution.selected {
            GraphicsSchedulerPath::Gms => RuntimeBridgePath::GmsPath,
            GraphicsSchedulerPath::Mgs => RuntimeBridgePath::MgsPath,
        };
        let runtime_bridge = if matches!(pipeline_mode, PipelineMode::Parallel) {
            Some(RuntimeBridgeOrchestrator::new_for_scheduler(
                scheduler_resolution.selected,
                &adapter_info.name,
                RuntimeBridgeConfig::default(),
                size.width.max(1),
                size.height.max(1),
            ))
        } else {
            None
        };
        let runtime_bridge_metrics = runtime_bridge
            .as_ref()
            .map(RuntimeBridgeOrchestrator::metrics)
            .unwrap_or(RuntimeBridgeMetrics {
                bridge_path: fallback_bridge_path,
                queued_plan_depth: 0,
                bridge_tick_published_frames: 0,
                bridge_tick_drained_plans: 0,
                frame_plans_popped: 0,
            });
        let runtime_bridge_telemetry =
            RuntimeBridgeTelemetry::new(runtime_bridge_metrics.bridge_path);
        eprintln!(
            "[pipeline] mode={} bridge_path={} enabled={}",
            pipeline_mode.as_str(),
            runtime_bridge_metrics.bridge_path.as_str(),
            runtime_bridge.is_some()
        );

        let logical_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .max(1);
        let physical_threads = num_cpus::get_physical().max(1);
        let mgs_like_path = matches!(scheduler_resolution.selected, GraphicsSchedulerPath::Mgs);
        // Treat MGS + ARM/Android as mobile-class scheduling: tighter tick ceilings and smaller
        // chunks help avoid frame-time chopping on heterogeneous SoCs (e.g., RK3588S).
        // Desktop-class adapters (Apple M-series, discrete GPUs) are excluded even when the MGS
        // path is active via an env-var override, so their full throughput budget is preserved.
        let mgs_is_mobile_hardware =
            mgs_like_path && !MobileGpuProfile::detect(&adapter_info.name).is_desktop_class();
        // Mobile-class CPU tuning: Android, actual mobile MGS hardware, or ARM targets other
        // than macOS (e.g., Raspberry Pi, ARM Linux). macOS on aarch64 is Apple Silicon —
        // those machines are desktop-class and are handled by the Apple Silicon path below.
        let mobile_class_tuning = matches!(platform, RuntimePlatform::Android)
            || mgs_is_mobile_hardware
            || (cfg!(any(target_arch = "aarch64", target_arch = "arm"))
                && cfg!(not(target_os = "macos")));
        let little_core_class = mobile_class_tuning && logical_threads <= 8;
        let adaptive_distance_enabled = options.adaptive_distance.resolve(mobile_class_tuning);
        let mut render_distance = options
            .render_distance
            .or_else(|| mobile_class_tuning.then_some(72.0));
        if render_distance.is_none() && adaptive_distance_enabled {
            render_distance = Some(if mobile_class_tuning { 84.0 } else { 96.0 });
        }
        let mut render_distance_min = 0.0;
        let mut render_distance_max = 0.0;
        if let Some(base) = render_distance {
            render_distance_min = (base * 0.72).clamp(28.0, base);
            render_distance_max = (base * 1.55).clamp(base, 220.0);
        }
        let distance_blur_mode = options.distance_blur;
        let distance_blur_enabled = distance_blur_mode.resolve(mobile_class_tuning);
        let adaptive_pacer_enabled = mobile_class_tuning
            && options.fps_cap.is_none()
            && matches!(options.vsync, VsyncMode::Off);
        let display_refresh_hint_hz = window
            .current_monitor()
            .and_then(|monitor| monitor.refresh_rate_millihertz())
            .map(|mhz| (mhz as f32 / 1_000.0).max(24.0));
        let mut adaptive_pacer_fps = display_refresh_hint_hz.unwrap_or(60.0).clamp(48.0, 90.0);
        if little_core_class {
            adaptive_pacer_fps = adaptive_pacer_fps.min(72.0);
        }
        let uncapped_dynamic_fps_hint = options.fps_cap.is_none()
            && matches!(options.vsync, VsyncMode::Off)
            && !adaptive_pacer_enabled;
        let initial_fps_hint = match (options.fps_cap, options.vsync) {
            (Some(fps), _) => fps.max(24.0),
            (None, VsyncMode::Off) => bootstrap_uncapped_fps_hint(logical_threads),
            (None, _) => display_refresh_hint_hz.unwrap_or(60.0),
        };
        let initial_fps_hint = if mobile_class_tuning && options.fps_cap.is_none() {
            initial_fps_hint.min(if little_core_class { 96.0 } else { 120.0 })
        } else {
            initial_fps_hint
        };
        let tick_policy = if mobile_class_tuning {
            TickRatePolicy {
                min_tick_hz: 24.0,
                max_tick_hz: if little_core_class { 220.0 } else { 300.0 },
                ticks_per_render_frame: 1.85,
                default_tick_hz: (initial_fps_hint * 1.45).clamp(72.0, 180.0),
            }
        } else if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
            // Apple Silicon (M-series): desktop-class CPU but a thermally-bounded SoC.
            // Use an intermediate policy — more aggressive than mobile but capped well
            // below the full workstation ceiling to avoid unnecessary core activity.
            // max_tick_hz of 180 Hz is 3× a 60 fps render target, which is ample for
            // smooth physics; the retune's desktop multipliers are bounded by this cap.
            TickRatePolicy {
                min_tick_hz: 30.0,
                max_tick_hz: (120.0 + logical_threads as f32 * 12.0).clamp(180.0, 300.0),
                ticks_per_render_frame: 2.0,
                default_tick_hz: (initial_fps_hint * 1.8).clamp(90.0, 160.0),
            }
        } else {
            TickRatePolicy {
                min_tick_hz: 30.0,
                max_tick_hz: (480.0 + logical_threads as f32 * 40.0).clamp(480.0, 1_440.0),
                ticks_per_render_frame: 2.6,
                default_tick_hz: (initial_fps_hint * 2.4).clamp(180.0, 420.0),
            }
        };
        let render_mode = match options.fps_cap {
            Some(fps) => RenderSyncMode::FpsCap { fps },
            None => {
                if uncapped_dynamic_fps_hint {
                    RenderSyncMode::Uncapped
                } else {
                    RenderSyncMode::Vsync {
                        display_hz: display_refresh_hint_hz.unwrap_or(60.0),
                    }
                }
            }
        };
        let fixed_dt = tick_policy.resolve_fixed_dt_seconds(render_mode, Some(initial_fps_hint));
        let fps_limit_hint = initial_fps_hint;
        let thread_scale = if mobile_class_tuning {
            (logical_threads as f32 / 6.0).clamp(0.70, 2.20)
        } else {
            (logical_threads as f32 / 8.0).clamp(0.75, 4.0)
        };
        // Smaller mobile shards improve load balancing across big.LITTLE clusters.
        let broadphase_chunk = if little_core_class {
            logical_threads.saturating_mul(8).clamp(48, 96)
        } else if mobile_class_tuning {
            logical_threads.saturating_mul(10).clamp(64, 160)
        } else {
            // Keep shards coarse enough to avoid scheduler overhead on high-core systems.
            logical_threads.saturating_mul(16).clamp(128, 512)
        };
        let max_pairs = if mobile_class_tuning {
            (72_000.0 * thread_scale).round() as usize
        } else {
            (96_000.0 * thread_scale).round() as usize
        };
        let max_manifolds = max_pairs;
        let solver_iterations = if little_core_class {
            4
        } else if logical_threads >= 20 {
            5
        } else {
            4
        };
        let tuned_max_substeps = if mobile_class_tuning {
            (8 + logical_threads / 4).clamp(6, 12) as u32
        } else {
            (12 + logical_threads / 4).clamp(10, 24) as u32
        };
        let world = PhysicsWorld::new(PhysicsWorldConfig {
            // Keep Earth-like gravity explicit for showcase consistency across presets.
            gravity: Vector3::new(0.0, -9.81, 0.0),
            fixed_dt,
            max_substeps: tuned_max_substeps,
            broadphase: BroadphaseConfig {
                chunk_size: broadphase_chunk,
                max_candidate_pairs: max_pairs,
                shard_pair_reserve: if mobile_class_tuning { 768 } else { 1_536 },
                speculative_sweep: true,
                speculative_max_distance: if mobile_class_tuning { 0.75 } else { 1.20 },
            },
            narrowphase: NarrowphaseConfig {
                max_manifolds,
                ..NarrowphaseConfig::default()
            },
            solver: ContactSolverConfig {
                iterations: solver_iterations,
                baumgarte: 0.32,
                penetration_slop: 0.0015,
                parallel_contact_push_strength: 0.28,
                parallel_contact_push_threshold: 16,
                hard_position_projection_strength: 0.95,
                hard_position_projection_threshold: if mobile_class_tuning { 96 } else { 128 },
                max_projection_per_contact: if mobile_class_tuning { 0.08 } else { 0.10 },
                ..ContactSolverConfig::default()
            },
            ..PhysicsWorldConfig::default()
        });
        let initial_scene_mode = project_scene_dimension_manifest
            .map(scene_mode_from_tlpfile_dimension)
            .unwrap_or(RuntimeSceneMode::Spatial3d);
        let scene = BounceTankSceneController::new(BounceTankSceneConfig {
            target_ball_count: 8_000,
            // Script can still tune this, but runtime starts from a progressive default.
            spawn_per_tick: if mobile_class_tuning {
                (68.0 * thread_scale).round() as usize
            } else {
                (96.0 * thread_scale).round() as usize
            },
            ball_restitution: 0.74,
            ball_friction: 0.28,
            wall_restitution: 0.78,
            wall_friction: 0.20,
            friction_transition_speed: 1.2,
            friction_static_boost: 1.10,
            friction_kinetic_scale: 0.92,
            linear_damping: 0.012,
            initial_speed_min: 0.35,
            initial_speed_max: 1.25,
            scene_mode: initial_scene_mode,
            ..BounceTankSceneConfig::default()
        });
        let tile_world_2d = build_default_side_view_tile_world(scene.config());
        let tile_world_frame = tile_world_2d.telemetry_snapshot();
        let fsr_config = FsrConfig {
            mode: options.fsr_mode,
            quality: options.fsr_quality,
            sharpness: options.fsr_sharpness,
            render_scale_override: options.fsr_scale_override,
        };
        let mut renderer = if prefer_vulkan_runtime_renderer() {
            #[cfg(target_os = "linux")]
            {
                let present_mode = match options.vsync {
                    VsyncMode::Auto => tl_core::PresentModePreference::MailboxFirst,
                    VsyncMode::On => tl_core::PresentModePreference::FifoOnly,
                    VsyncMode::Off => tl_core::PresentModePreference::ImmediateFirst,
                };
                let vk_config = VulkanSceneRendererConfig {
                    backend: tl_core::VulkanBackendConfig {
                        present_mode,
                        max_instances: 65_536,
                        ..Default::default()
                    },
                    prefer_secondary_gpu: true,
                };
                match VulkanSceneRenderer::new(window.clone(), vk_config) {
                    Ok(renderer) => {
                        eprintln!("[renderer] using raw Vulkan runtime adapter");
                        TlAppRenderer::Vulkan(renderer)
                    }
                    Err(err) => {
                        eprintln!(
                            "[renderer] raw Vulkan runtime adapter failed ({}), falling back to wgpu",
                            err
                        );
                        let format = ensure_wgpu_surface()?;
                        TlAppRenderer::Wgpu(WgpuSceneRenderer::new(
                            &device,
                            &queue,
                            format,
                            size.width,
                            size.height,
                            adapter_info.backend,
                            options.msaa,
                        ))
                    }
                }
            }
            #[cfg(not(target_os = "linux"))]
            {
                let format = ensure_wgpu_surface()?;
                TlAppRenderer::Wgpu(WgpuSceneRenderer::new(
                    &device,
                    &queue,
                    format,
                    size.width,
                    size.height,
                    adapter_info.backend,
                    options.msaa,
                ))
            }
        } else {
            let format = ensure_wgpu_surface()?;
            TlAppRenderer::Wgpu(WgpuSceneRenderer::new(
                &device,
                &queue,
                format,
                size.width,
                size.height,
                adapter_info.backend,
                options.msaa,
            ))
        };
        renderer.set_ray_tracing_mode(&queue, RayTracingMode::Auto);
        renderer.set_fsr_config(&queue, fsr_config);
        // Always keep a deterministic high-quality FBX-equivalent mesh in slot 2 so script
        // `set_ball_mesh_slot(2)` stays stable even when tlsprite binding fails.
        renderer.bind_builtin_sphere_mesh_slot(&device, DEFAULT_FBX_BALL_SLOT, true);
        renderer.bind_builtin_sphere_mesh_slot(&device, AUTO_LOW_POLY_BALL_SLOT, false);
        let camera = FreeCameraController::default();
        let gamepad = GamepadManager::new();
        let (eye, target) = camera.eye_target();
        if let Some(current_distance) = render_distance {
            let ext = scene.config().container_half_extents;
            let tank_radius = (ext[0] * ext[0] + ext[1] * ext[1] + ext[2] * ext[2]).sqrt();
            let camera_to_center = (eye[0] * eye[0] + eye[1] * eye[1] + eye[2] * eye[2]).sqrt();
            let stable_distance_floor = (camera_to_center + tank_radius + 6.0).clamp(40.0, 180.0);
            let guarded_distance = current_distance.max(stable_distance_floor);
            render_distance = Some(guarded_distance);
            render_distance_min = render_distance_min
                .max(stable_distance_floor * 0.86)
                .min(guarded_distance);
            render_distance_max = render_distance_max
                .max(stable_distance_floor * 1.20)
                .max(guarded_distance)
                .clamp(guarded_distance, 260.0);
        }
        renderer.set_camera_view(&queue, size.width.max(1), size.height.max(1), eye, target);
        let fsr_status = renderer.fsr_status();
        eprintln!(
            "[mps] cpu profile logical={} physical={} mobile_tuning={} little_core_class={} thread_scale={:.2} broadphase_chunk={} max_pairs={} solver_iters={} max_substeps={} render_distance={:?} adaptive_distance={} distance_blur={:?} ({}) adaptive_pacer={} ({:.0} fps) fsr={:?} active={} scale={:.2} sharpness={:.2} reason={}",
            logical_threads,
            physical_threads,
            mobile_class_tuning,
            little_core_class,
            thread_scale,
            broadphase_chunk,
            max_pairs,
            solver_iterations,
            tuned_max_substeps,
            render_distance,
            adaptive_distance_enabled,
            distance_blur_mode,
            distance_blur_enabled,
            adaptive_pacer_enabled,
            adaptive_pacer_fps,
            options.fsr_mode,
            fsr_status.active,
            fsr_status.render_scale,
            fsr_status.sharpness,
            if fsr_status.reason.is_empty() {
                "none"
            } else {
                fsr_status.reason.as_str()
            }
        );

        let draw_compiler = DrawPathCompiler::new();
        let hud = TelemetryHudComposer::new(Default::default());
        let mut scene = scene;
        let mut force_full_fbx_from_sprite = false;
        let (sprite_loader, sprite_cache) = if let Some(program) = joint_merged_sprite_program {
            force_full_fbx_from_sprite = program.requires_full_fbx_render();
            scene.set_sprite_program(program.clone());
            if let Some(root_hint) = bundle_sprite_root.as_deref() {
                bind_renderer_meshes_from_tlsprite(
                    &mut renderer,
                    &device,
                    &queue,
                    root_hint,
                    &program,
                );
            }
            (None, None)
        } else {
            let mut sprite_loader = TlspriteWatchReloader::new(&options.sprite_path);
            let mut sprite_cache = TlspriteProgramCache::new();
            if let Some(warn) = sprite_loader.init_warning() {
                eprintln!("[tlsprite watch] {warn}");
            }
            eprintln!("[tlsprite watch] backend={:?}", sprite_loader.backend());
            let event = sprite_loader.reload_into_cache(&mut sprite_cache);
            print_tlsprite_event("[tlsprite boot]", event);
            for diag in sprite_loader.diagnostics() {
                eprintln!(
                    "[tlsprite diag] {:?} line {}: {}",
                    diag.level, diag.line, diag.message
                );
            }
            if let Some(program) = sprite_cache.program_for_path(sprite_loader.path()).cloned() {
                eprintln!(
                    "[tlsprite boot] loaded sprites={} lights={}",
                    program.sprites().len(),
                    program.lights().len()
                );
                force_full_fbx_from_sprite = program.requires_full_fbx_render();
                scene.set_sprite_program(program.clone());
                bind_renderer_meshes_from_tlsprite(
                    &mut renderer,
                    &device,
                    &queue,
                    sprite_loader.path(),
                    &program,
                );
            } else {
                eprintln!(
                    "[tlsprite boot] WARNING: sprite program not loaded from '{}' — lights will be unavailable",
                    sprite_loader.path().display()
                );
            }
            (Some(sprite_loader), Some(sprite_cache))
        };
        renderer.set_force_full_fbx_sphere(force_full_fbx_from_sprite);

        let now = Instant::now();
        let frame_cap_interval = options
            .fps_cap
            .map(|fps| 1.0 / fps.max(1.0))
            .or_else(|| adaptive_pacer_enabled.then_some(1.0 / adaptive_pacer_fps.max(1.0)));
        let frame_cap_interval = frame_cap_interval.map(Duration::from_secs_f32);
        let fps_report_interval = options.fps_report_interval;

        let mut runtime = Self {
            cli_options: options.clone(),
            file_io_root,
            pak_mount_root,
            window,
            _instance: instance,
            surface,
            device,
            queue,
            config,
            size,
            world: PhysicsMpsRunner::new(world),
            physics_token: None,
            last_tick: BounceTankTickMetrics {
                spawned_this_tick: 0,
                scattered_this_tick: 0,
                live_balls: 0,
                target_balls: 0,
                fully_spawned: false,
            },
            scene,
            tile_world_2d,
            tile_world_frame,
            draw_compiler,
            hud,
            renderer,
            camera,
            sprite_loader,
            sprite_cache,
            force_full_fbx_from_sprite,
            script_runtime: script_runtime.expect("script runtime must be initialized"),
            script_last_spawned: 0,
            script_frame_index: 0,
            script_key_f_keyboard: false,
            script_key_g_keyboard: false,
            console: RuntimeConsoleState::default(),
            console_overlay_sprites: Vec::new(),
            keyboard_camera: CameraInputState::default(),
            mouse_look_held: false,
            look_lock_active: false,
            mouse_look_delta: (0.0, 0.0),
            camera_reset_requested: false,
            keyboard_modifiers: ModifiersState::empty(),
            gamepad,
            touch_look_id: None,
            touch_last_position: None,
            cursor_position: None,
            tick_policy,
            tick_profile: options.tick_profile,
            tick_cap: options.tick_cap,
            tick_hz: 1.0 / fixed_dt.max(1e-6),
            fps_limit_hint,
            uncapped_dynamic_fps_hint,
            adaptive_pacer_enabled,
            adaptive_pacer_fps,
            adaptive_pacer_timer: 0.0,
            mps_logical_threads: logical_threads,
            max_substeps: tuned_max_substeps,
            last_substeps: 0,
            manual_max_substeps: None,
            simulation_paused: false,
            simulation_step_budget: 0,
            tick_retune_timer: 0.0,
            adaptive_ball_render_limit: None,
            adaptive_live_ball_budget: None,
            adaptive_low_poly_override: false,
            mgs_is_mobile_hardware,
            render_distance,
            render_distance_min,
            render_distance_max,
            adaptive_distance_enabled,
            distance_blur_mode,
            distance_blur_enabled,
            last_distance_culled: 0,
            last_distance_blurred: 0,
            last_framebuffer_fill_ratio: 0.0,
            framebuffer_fill_ema: 0.0,
            adaptive_load_pressure_ema: 0.0,
            distance_retune_timer: 0.0,
            frame_time_ema_ms: 0.0,
            frame_time_jitter_ema_ms: 0.0,
            frame_time_budget_ms: (1_000.0 / fps_limit_hint.max(24.0)).clamp(3.0, 41.0),
            frame_started_at: now,
            frame_cap_interval,
            next_redraw_at: now,
            fps_tracker: FpsTracker::new(fps_report_interval),
            scheduler_path: scheduler_resolution.selected,
            scheduler_reason: scheduler_resolution.reason,
            scheduler_fallback_applied: scheduler_resolution.fallback_applied,
            pipeline_mode,
            runtime_bridge,
            runtime_bridge_metrics,
            runtime_bridge_telemetry,
            bridge_frame_counter: 1,
            adapter_backend: adapter_info.backend,
            adapter_name: adapter_info.name,
            present_mode: selected_present_mode,
            platform,
            rt_mode: RayTracingMode::Auto,
            fsr_config,
            fsr_dynamo_config: FsrDynamoConfig::default(),
            fsr_dynamo_scale: 1.0,
            shutdown_prepared: false,
        };
        runtime.sync_console_quick_fields_from_runtime();
        Ok(runtime)
    }
}

fn resolve_path_from_root(root: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        root.join(path)
    }
}

fn allocate_pak_mount_root(file_io_root: &Path) -> Result<PathBuf, Box<dyn Error>> {
    let base_dir = file_io_root.join(".tileline").join("pak_mounts");
    fs::create_dir_all(&base_dir).map_err(|err| {
        format!(
            "failed to create pak mount directory '{}': {err}",
            base_dir.display()
        )
    })?;

    let pid = std::process::id();
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros();
    for attempt in 0..64u32 {
        let candidate = base_dir.join(format!("tlapp-{pid}-{stamp}-{attempt}"));
        if candidate.exists() {
            continue;
        }
        fs::create_dir_all(&candidate).map_err(|err| {
            format!(
                "failed to create pak mount root '{}': {err}",
                candidate.display()
            )
        })?;
        return Ok(candidate);
    }

    Err(format!(
        "failed to allocate unique pak mount root under '{}'",
        base_dir.display()
    )
    .into())
}

fn remap_cli_paths_from_pak_mount(options: &mut CliOptions, mount_root: &Path) {
    let mut remapped = Vec::new();
    if remap_optional_relative_path(&mut options.project_path, mount_root) {
        remapped.push("project");
    }
    if remap_optional_relative_path(&mut options.joint_path, mount_root) {
        remapped.push("joint");
    }
    if remap_relative_path(&mut options.script_path, mount_root) {
        remapped.push("script");
    }
    if remap_relative_path(&mut options.sprite_path, mount_root) {
        remapped.push("sprite");
    }
    if !remapped.is_empty() {
        eprintln!(
            "[pak] resolved runtime content from mounted archive: {}",
            remapped.join(", ")
        );
    }
}

fn remap_optional_relative_path(path: &mut Option<PathBuf>, mount_root: &Path) -> bool {
    let Some(inner) = path.as_mut() else {
        return false;
    };
    remap_relative_path(inner, mount_root)
}

fn remap_relative_path(path: &mut PathBuf, mount_root: &Path) -> bool {
    if path.is_absolute() {
        return false;
    }
    let candidate = mount_root.join(path.as_path());
    if !candidate.exists() {
        return false;
    }
    *path = candidate;
    true
}
