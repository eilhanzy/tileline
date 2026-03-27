//! `.tlsprite` parser and runtime sprite emission helpers.
//!
//! The format is intentionally small and text-based so toolchains can emit it easily.
//! Parsing is memory-resident (`&str` input), and the output can be reused every frame.
//!
//! Minimal format:
//! ```text
//! tlsprite_v1
//! [progress_bar]
//! sprite_id = 1
//! kind = hud
//! texture_slot = 0
//! layer = 100
//! position = -0.86, 0.90, 0.0
//! size = 0.40, 0.035
//! rotation_rad = 0.0
//! color = 0.10, 0.84, 0.62, 0.92
//! fbx = docs/demos/tlapp/sphere.fbx
//! scale_axis = x
//! scale_source = spawn_progress
//! scale_min = 0.02
//! scale_max = 1.0
//! ```

use std::{
    collections::{hash_map::DefaultHasher, HashMap},
    fs,
    hash::{Hash, Hasher},
    io::Cursor,
    path::{Path, PathBuf},
    sync::mpsc::{self, Receiver, TryRecvError},
    time::{Duration, Instant},
};

use fbx::Property as FbxProperty;
use image::{imageops::FilterType, ImageReader};
use notify::{
    Config as NotifyConfig, Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher,
};
use resvg::{tiny_skia, usvg};

use crate::scene::{SceneLight, SceneLightKind, SpriteInstance, SpriteKind};

/// Input bindings consumed while emitting runtime sprite instances.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TlspriteFrameContext {
    pub spawn_progress: f32,
    pub live_balls: usize,
    pub target_balls: usize,
}

impl Default for TlspriteFrameContext {
    fn default() -> Self {
        Self {
            spawn_progress: 1.0,
            live_balls: 0,
            target_balls: 0,
        }
    }
}

/// Runtime diagnostics produced by `.tlsprite` compilation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TlspriteDiagnosticLevel {
    Warning,
    Error,
}

/// One parser/validation diagnostic line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TlspriteDiagnostic {
    pub level: TlspriteDiagnosticLevel,
    pub line: usize,
    pub message: String,
}

/// Soft compile outcome for `.tlsprite`.
#[derive(Debug, Clone)]
pub struct TlspriteCompileOutcome {
    pub program: Option<TlspriteProgram>,
    pub diagnostics: Vec<TlspriteDiagnostic>,
}

impl TlspriteCompileOutcome {
    pub fn has_errors(&self) -> bool {
        self.diagnostics
            .iter()
            .any(|d| d.level == TlspriteDiagnosticLevel::Error)
    }
}

/// Which sprite axis should be dynamically scaled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TlspriteScaleAxis {
    X,
    Y,
}

/// Which runtime signal drives dynamic scaling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TlspriteScaleSource {
    SpawnProgress,
    SpawnRemaining,
    LiveBallRatio,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct TlspriteDynamicScale {
    axis: TlspriteScaleAxis,
    source: TlspriteScaleSource,
    min_factor: f32,
    max_factor: f32,
}

/// One compiled sprite definition.
#[derive(Debug, Clone, PartialEq)]
pub struct TlspriteSpriteDef {
    pub name: String,
    pub sprite: SpriteInstance,
    /// Original FBX path from source (if any). Used by runtime policy hooks.
    pub fbx_source: Option<String>,
    /// Optional sprite texture source (`.png`/`.svg`) from source text.
    pub texture_source: Option<String>,
    dynamic_scale: Option<TlspriteDynamicScale>,
}

/// One compiled light definition emitted from `.tlsprite`.
#[derive(Debug, Clone, PartialEq)]
pub struct TlspriteLightDef {
    pub name: String,
    pub light: SceneLight,
    /// Whether this light definition requests an auto-generated glow billboard.
    /// Mirrors `light.glow_enabled`; exposed here for convenient runtime queries.
    pub glow_enabled: bool,
}

/// Compiled `.tlsprite` sprite set.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TlspriteProgram {
    sprites: Vec<TlspriteSpriteDef>,
    lights: Vec<TlspriteLightDef>,
}

impl TlspriteProgram {
    pub fn sprites(&self) -> &[TlspriteSpriteDef] {
        &self.sprites
    }

    pub fn lights(&self) -> &[TlspriteLightDef] {
        &self.lights
    }

    /// Returns `true` when at least one sprite section requests FBX-derived behavior.
    ///
    /// The renderer uses this as a policy signal to keep full FBX sphere rendering enabled and
    /// avoid adaptive low-poly fallback in showcase paths.
    pub fn requires_full_fbx_render(&self) -> bool {
        self.sprites
            .iter()
            .any(|sprite| sprite.fbx_source.as_deref().is_some())
    }

    /// Returns unique FBX mesh bindings inferred from sprite sections.
    ///
    /// Convention:
    /// - `texture_slot` is reused as mesh slot id for runtime 3D mesh binding.
    /// - first declaration for a slot wins (deterministic by section order).
    pub fn mesh_fbx_bindings(&self) -> Vec<(u8, &str)> {
        let mut out = Vec::new();
        for sprite in &self.sprites {
            let Some(path) = sprite.fbx_source.as_deref() else {
                continue;
            };
            let slot = sprite.sprite.texture_slot.min(u8::MAX as u16) as u8;
            if out.iter().any(|(existing_slot, _)| *existing_slot == slot) {
                continue;
            }
            out.push((slot, path));
        }
        out
    }

    /// Returns unique texture bindings inferred from sprite sections.
    ///
    /// Convention:
    /// - `texture_slot` is used as atlas slot id.
    /// - first declaration for a slot wins (deterministic by section order).
    pub fn sprite_texture_bindings(&self) -> Vec<(u16, &str)> {
        let mut out = Vec::new();
        for sprite in &self.sprites {
            let Some(path) = sprite.texture_source.as_deref() else {
                continue;
            };
            let slot = sprite.sprite.texture_slot;
            if out.iter().any(|(existing_slot, _)| *existing_slot == slot) {
                continue;
            }
            out.push((slot, path));
        }
        out
    }

    /// Emit frame-local sprite instances into `out`.
    pub fn emit_instances(&self, ctx: TlspriteFrameContext, out: &mut Vec<SpriteInstance>) {
        out.reserve(self.sprites.len());
        let spawn_progress = ctx.spawn_progress.clamp(0.0, 1.0);
        let live_ratio = if ctx.target_balls == 0 {
            1.0
        } else {
            (ctx.live_balls as f32 / ctx.target_balls as f32).clamp(0.0, 1.0)
        };

        for def in &self.sprites {
            let mut instance = def.sprite.clone();
            if let Some(scale) = def.dynamic_scale {
                let raw = match scale.source {
                    TlspriteScaleSource::SpawnProgress => spawn_progress,
                    TlspriteScaleSource::SpawnRemaining => 1.0 - spawn_progress,
                    TlspriteScaleSource::LiveBallRatio => live_ratio,
                };
                let factor = raw.clamp(scale.min_factor, scale.max_factor);
                match scale.axis {
                    TlspriteScaleAxis::X => {
                        instance.size[0] = (instance.size[0] * factor).max(0.0001);
                    }
                    TlspriteScaleAxis::Y => {
                        instance.size[1] = (instance.size[1] * factor).max(0.0001);
                    }
                }
            }
            out.push(instance);
        }
    }

    /// Emit frame-local scene lights into `out`.
    pub fn emit_lights(&self, out: &mut Vec<SceneLight>) {
        out.reserve(self.lights.len());
        for def in &self.lights {
            out.push(def.light.clone());
        }
        out.sort_by_key(|light| (light.layer, light.id));
    }

    /// Merge multiple compiled programs into one deterministic emission order.
    ///
    /// Programs are appended in slice order and each program's internal sprite order is preserved.
    /// This enables `.tljoint` scene bundles to compose multiple `.tlsprite` files.
    pub fn merge_programs(programs: &[TlspriteProgram]) -> TlspriteProgram {
        let total = programs.iter().map(|program| program.sprites.len()).sum();
        let total_lights = programs.iter().map(|program| program.lights.len()).sum();
        let mut sprites = Vec::with_capacity(total);
        let mut lights = Vec::with_capacity(total_lights);
        for program in programs {
            sprites.extend(program.sprites.iter().cloned());
            lights.extend(program.lights.iter().cloned());
        }
        TlspriteProgram { sprites, lights }
    }
}

/// Hot-reload behavior knobs for `.tlsprite` disk sources.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TlspriteHotReloadConfig {
    /// Keep the last valid program if a new edit fails to compile.
    pub keep_last_good_program: bool,
}

impl Default for TlspriteHotReloadConfig {
    fn default() -> Self {
        Self {
            keep_last_good_program: true,
        }
    }
}

/// Result of one hot-reload polling attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TlspriteHotReloadEvent {
    Unchanged,
    Applied {
        sprite_count: usize,
        warning_count: usize,
    },
    Rejected {
        error_count: usize,
        warning_count: usize,
        kept_last_program: bool,
    },
    SourceError {
        message: String,
    },
}

/// Simple hash-based `.tlsprite` file hot-reloader.
#[derive(Debug, Clone)]
pub struct TlspriteHotReloader {
    path: PathBuf,
    config: TlspriteHotReloadConfig,
    source_hash: Option<u64>,
    program: Option<TlspriteProgram>,
    diagnostics: Vec<TlspriteDiagnostic>,
}

impl TlspriteHotReloader {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self::with_config(path, TlspriteHotReloadConfig::default())
    }

    pub fn with_config(path: impl Into<PathBuf>, config: TlspriteHotReloadConfig) -> Self {
        Self {
            path: path.into(),
            config,
            source_hash: None,
            program: None,
            diagnostics: Vec::new(),
        }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn program(&self) -> Option<&TlspriteProgram> {
        self.program.as_ref()
    }

    pub fn diagnostics(&self) -> &[TlspriteDiagnostic] {
        &self.diagnostics
    }

    pub fn source_hash(&self) -> Option<u64> {
        self.source_hash
    }

    /// Read source and recompile when file contents have changed.
    pub fn reload_if_changed(&mut self) -> TlspriteHotReloadEvent {
        let bytes = match fs::read(&self.path) {
            Ok(bytes) => bytes,
            Err(err) => {
                return TlspriteHotReloadEvent::SourceError {
                    message: format!("failed to read '{}': {err}", self.path.display()),
                };
            }
        };

        let hash = hash_bytes(&bytes);
        if self.source_hash == Some(hash) {
            return TlspriteHotReloadEvent::Unchanged;
        }
        self.source_hash = Some(hash);

        let source = match String::from_utf8(bytes) {
            Ok(text) => text,
            Err(err) => {
                return TlspriteHotReloadEvent::SourceError {
                    message: format!("source '{}' is not valid UTF-8: {err}", self.path.display()),
                };
            }
        };
        let outcome = compile_tlsprite(&source);
        let warning_count = outcome
            .diagnostics
            .iter()
            .filter(|d| d.level == TlspriteDiagnosticLevel::Warning)
            .count();
        let error_count = outcome
            .diagnostics
            .iter()
            .filter(|d| d.level == TlspriteDiagnosticLevel::Error)
            .count();
        self.diagnostics = outcome.diagnostics;

        if let Some(program) = outcome.program {
            let sprite_count = program.sprites().len();
            self.program = Some(program);
            return TlspriteHotReloadEvent::Applied {
                sprite_count,
                warning_count,
            };
        }

        if !self.config.keep_last_good_program {
            self.program = None;
        }
        TlspriteHotReloadEvent::Rejected {
            error_count,
            warning_count,
            kept_last_program: self.program.is_some(),
        }
    }
}

/// Watch backend used by `TlspriteWatchReloader`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TlspriteWatchBackend {
    Polling,
    Notify,
}

/// File-watch layer configuration (Phase 2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TlspriteWatchConfig {
    /// Try OS watcher backend first; fallback to polling if watcher initialization fails.
    pub prefer_notify_backend: bool,
    /// Minimum interval for safety polling and pure polling fallback.
    pub poll_interval_ms: u64,
}

impl Default for TlspriteWatchConfig {
    fn default() -> Self {
        Self {
            prefer_notify_backend: true,
            poll_interval_ms: 200,
        }
    }
}

enum TlspriteWatchState {
    Polling,
    Notify {
        rx: Receiver<notify::Result<Event>>,
        _watcher: RecommendedWatcher,
        pending_change: bool,
        watched_path: PathBuf,
    },
}

/// Phase-2 `.tlsprite` hot reloader: event-driven backend + polling fallback.
pub struct TlspriteWatchReloader {
    inner: TlspriteHotReloader,
    state: TlspriteWatchState,
    poll_interval: Duration,
    next_poll_at: Instant,
    init_warning: Option<String>,
}

impl TlspriteWatchReloader {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self::with_configs(
            path,
            TlspriteHotReloadConfig::default(),
            TlspriteWatchConfig::default(),
        )
    }

    pub fn with_configs(
        path: impl Into<PathBuf>,
        hot_config: TlspriteHotReloadConfig,
        watch_config: TlspriteWatchConfig,
    ) -> Self {
        let path = path.into();
        let inner = TlspriteHotReloader::with_config(path.clone(), hot_config);
        let poll_interval = Duration::from_millis(watch_config.poll_interval_ms.max(1));
        let mut init_warning = None;
        let state = if watch_config.prefer_notify_backend {
            match build_notify_state(&path) {
                Ok(state) => state,
                Err(err) => {
                    init_warning = Some(format!(
                        "notify backend unavailable for '{}': {err}; using polling fallback",
                        path.display()
                    ));
                    TlspriteWatchState::Polling
                }
            }
        } else {
            TlspriteWatchState::Polling
        };

        Self {
            inner,
            state,
            poll_interval,
            next_poll_at: Instant::now(),
            init_warning,
        }
    }

    pub fn backend(&self) -> TlspriteWatchBackend {
        match self.state {
            TlspriteWatchState::Polling => TlspriteWatchBackend::Polling,
            TlspriteWatchState::Notify { .. } => TlspriteWatchBackend::Notify,
        }
    }

    pub fn init_warning(&self) -> Option<&str> {
        self.init_warning.as_deref()
    }

    pub fn program(&self) -> Option<&TlspriteProgram> {
        self.inner.program()
    }

    pub fn diagnostics(&self) -> &[TlspriteDiagnostic] {
        self.inner.diagnostics()
    }

    /// Reload if notify signaled changes (or polling interval elapsed in fallback mode).
    pub fn reload_if_needed(&mut self) -> TlspriteHotReloadEvent {
        // Boot path: always load once.
        if self.inner.program().is_none() {
            self.next_poll_at = Instant::now() + self.poll_interval;
            return self.inner.reload_if_changed();
        }

        match &mut self.state {
            TlspriteWatchState::Polling => {
                if Instant::now() < self.next_poll_at {
                    return TlspriteHotReloadEvent::Unchanged;
                }
                self.next_poll_at = Instant::now() + self.poll_interval;
                self.inner.reload_if_changed()
            }
            TlspriteWatchState::Notify {
                rx,
                pending_change,
                watched_path,
                ..
            } => {
                let mut changed = *pending_change;
                loop {
                    match rx.try_recv() {
                        Ok(Ok(event)) => {
                            if event_can_change_source(&event.kind)
                                && event_targets_path(&event, watched_path.as_path())
                            {
                                changed = true;
                            }
                        }
                        Ok(Err(err)) => {
                            return TlspriteHotReloadEvent::SourceError {
                                message: format!("notify event error: {err}"),
                            };
                        }
                        Err(TryRecvError::Empty) => break,
                        Err(TryRecvError::Disconnected) => {
                            return TlspriteHotReloadEvent::SourceError {
                                message: "notify channel disconnected".to_string(),
                            };
                        }
                    }
                }

                // Safety poll in case a backend misses edge events.
                if !changed && Instant::now() >= self.next_poll_at {
                    changed = true;
                }
                self.next_poll_at = Instant::now() + self.poll_interval;
                *pending_change = false;

                if changed {
                    self.inner.reload_if_changed()
                } else {
                    TlspriteHotReloadEvent::Unchanged
                }
            }
        }
    }
}

fn build_notify_state(path: &Path) -> notify::Result<TlspriteWatchState> {
    let (tx, rx) = mpsc::channel();
    let mut watcher = RecommendedWatcher::new(
        move |res| {
            let _ = tx.send(res);
        },
        NotifyConfig::default(),
    )?;
    watcher.watch(path, RecursiveMode::NonRecursive)?;
    Ok(TlspriteWatchState::Notify {
        rx,
        _watcher: watcher,
        pending_change: true,
        watched_path: path.to_path_buf(),
    })
}

const TLSPRITE_PACK_MAGIC_V1: &[u8; 8] = b"TLSPK001";
const TLSPRITE_PACK_MAGIC_V2: &[u8; 8] = b"TLSPK002";
const TLSPRITE_PACK_MAGIC_V3: &[u8; 8] = b"TLSPK003";
const TLSPRITE_PACK_MAGIC_V4: &[u8; 8] = b"TLSPK004";
const TLSPRITE_PACK_MAGIC: &[u8; 8] = TLSPRITE_PACK_MAGIC_V4;

/// In-memory binary pack produced from a `.tlsprite` source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TlspritePack {
    pub bytes: Vec<u8>,
    pub source_hash: u64,
    pub sprite_count: usize,
    pub light_count: usize,
}

/// Compile `.tlsprite` into a compact precompiled pack.
pub fn compile_tlsprite_pack(source: &str) -> Result<TlspritePack, TlspriteCompileOutcome> {
    let TlspriteCompileOutcome {
        program,
        diagnostics,
    } = compile_tlsprite(source);
    let program = match program {
        Some(program) => program,
        None => {
            return Err(TlspriteCompileOutcome {
                program: None,
                diagnostics,
            });
        }
    };
    let source_hash = hash_bytes(source.as_bytes());
    let bytes = encode_tlsprite_pack(&program, source_hash);
    Ok(TlspritePack {
        bytes,
        source_hash,
        sprite_count: program.sprites().len(),
        light_count: program.lights().len(),
    })
}

/// Decode a precompiled `.tlsprite` pack.
pub fn load_tlsprite_pack(bytes: &[u8]) -> Result<TlspriteProgram, String> {
    decode_tlsprite_pack(bytes).map(|decoded| decoded.program)
}

#[derive(Debug, Clone)]
struct DecodedTlspritePack {
    program: TlspriteProgram,
    source_hash: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TlspritePackVersion {
    V1,
    V2,
    V3,
    V4,
}

/// Source marker returned by cache loads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TlspriteCacheLoadSource {
    CacheHit,
    CompiledSource,
    LoadedPack,
}

/// Cache load summary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TlspriteCacheLoadOutcome {
    pub source: TlspriteCacheLoadSource,
    pub source_hash: u64,
    pub sprite_count: usize,
}

/// Cache stats for runtime telemetry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TlspriteProgramCacheStats {
    pub unique_programs: usize,
    pub path_bindings: usize,
}

/// Runtime `.tlsprite` program cache with explicit invalidation.
#[derive(Debug, Default)]
pub struct TlspriteProgramCache {
    by_hash: HashMap<u64, TlspriteProgram>,
    path_hash: HashMap<PathBuf, u64>,
    hash_refcount: HashMap<u64, usize>,
}

impl TlspriteProgramCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn load_from_source(
        &mut self,
        path: impl Into<PathBuf>,
        source: &str,
    ) -> Result<TlspriteCacheLoadOutcome, TlspriteCompileOutcome> {
        let path = path.into();
        let source_hash = hash_bytes(source.as_bytes());
        if let Some(out) = self.try_cache_hit(path.as_path(), source_hash) {
            return Ok(out);
        }

        let outcome = compile_tlsprite(source);
        let Some(program) = outcome.program else {
            return Err(outcome);
        };
        Ok(self.bind_program(
            path.as_path(),
            source_hash,
            program,
            TlspriteCacheLoadSource::CompiledSource,
        ))
    }

    pub fn load_from_pack(
        &mut self,
        path: impl Into<PathBuf>,
        bytes: &[u8],
    ) -> Result<TlspriteCacheLoadOutcome, String> {
        let path = path.into();
        let decoded = decode_tlsprite_pack(bytes)?;
        if let Some(out) = self.try_cache_hit(path.as_path(), decoded.source_hash) {
            return Ok(out);
        }
        Ok(self.bind_program(
            path.as_path(),
            decoded.source_hash,
            decoded.program,
            TlspriteCacheLoadSource::LoadedPack,
        ))
    }

    pub fn bind_runtime_program(
        &mut self,
        path: impl Into<PathBuf>,
        source_hash: u64,
        program: TlspriteProgram,
    ) -> TlspriteCacheLoadOutcome {
        let path = path.into();
        self.bind_program(
            path.as_path(),
            source_hash,
            program,
            TlspriteCacheLoadSource::CompiledSource,
        )
    }

    pub fn program_for_path(&self, path: &Path) -> Option<&TlspriteProgram> {
        self.path_hash
            .get(path)
            .and_then(|hash| self.by_hash.get(hash))
    }

    pub fn invalidate_path(&mut self, path: &Path) -> bool {
        let Some(old_hash) = self.path_hash.remove(path) else {
            return false;
        };
        self.decrement_hash_ref(old_hash);
        true
    }

    pub fn invalidate_all(&mut self) {
        self.by_hash.clear();
        self.path_hash.clear();
        self.hash_refcount.clear();
    }

    pub fn stats(&self) -> TlspriteProgramCacheStats {
        TlspriteProgramCacheStats {
            unique_programs: self.by_hash.len(),
            path_bindings: self.path_hash.len(),
        }
    }

    fn try_cache_hit(&mut self, path: &Path, source_hash: u64) -> Option<TlspriteCacheLoadOutcome> {
        if !self.by_hash.contains_key(&source_hash) {
            return None;
        }
        self.rebind_path(path, source_hash);
        let sprite_count = self
            .by_hash
            .get(&source_hash)
            .map(|p| p.sprites().len())
            .unwrap_or(0);
        Some(TlspriteCacheLoadOutcome {
            source: TlspriteCacheLoadSource::CacheHit,
            source_hash,
            sprite_count,
        })
    }

    fn bind_program(
        &mut self,
        path: &Path,
        source_hash: u64,
        program: TlspriteProgram,
        source: TlspriteCacheLoadSource,
    ) -> TlspriteCacheLoadOutcome {
        self.by_hash.insert(source_hash, program);
        self.rebind_path(path, source_hash);
        let sprite_count = self
            .by_hash
            .get(&source_hash)
            .map(|p| p.sprites().len())
            .unwrap_or(0);
        TlspriteCacheLoadOutcome {
            source,
            source_hash,
            sprite_count,
        }
    }

    fn rebind_path(&mut self, path: &Path, new_hash: u64) {
        if let Some(old_hash) = self.path_hash.insert(path.to_path_buf(), new_hash) {
            if old_hash == new_hash {
                return;
            }
            self.decrement_hash_ref(old_hash);
        }
        *self.hash_refcount.entry(new_hash).or_insert(0) += 1;
    }

    fn decrement_hash_ref(&mut self, hash: u64) {
        let should_remove = if let Some(count) = self.hash_refcount.get_mut(&hash) {
            *count = count.saturating_sub(1);
            *count == 0
        } else {
            false
        };
        if should_remove {
            self.hash_refcount.remove(&hash);
            self.by_hash.remove(&hash);
        }
    }
}

impl TlspriteWatchReloader {
    /// Reload and store applied programs in cache keyed by this loader path.
    pub fn reload_into_cache(
        &mut self,
        cache: &mut TlspriteProgramCache,
    ) -> TlspriteHotReloadEvent {
        let event = self.reload_if_needed();
        if matches!(event, TlspriteHotReloadEvent::Applied { .. }) {
            if let (Some(hash), Some(program)) =
                (self.inner.source_hash(), self.inner.program().cloned())
            {
                let _ = cache.bind_runtime_program(self.inner.path().to_path_buf(), hash, program);
            }
        }
        event
    }

    pub fn path(&self) -> &Path {
        self.inner.path()
    }
}

/// Compile `.tlsprite` source from memory.
pub fn compile_tlsprite(source: &str) -> TlspriteCompileOutcome {
    compile_tlsprite_with_extra_roots(source, &[])
}

/// Compile `.tlsprite` source with additional root directories for relative `fbx` paths.
pub fn compile_tlsprite_with_extra_roots(
    source: &str,
    extra_roots: &[PathBuf],
) -> TlspriteCompileOutcome {
    let mut diagnostics = Vec::new();
    let mut sprites = Vec::new();
    let mut lights = Vec::new();
    let mut header_checked = false;
    let mut pending: Option<PendingEntry> = None;

    for (idx, raw_line) in source.lines().enumerate() {
        let line_no = idx + 1;
        let line = strip_comments(raw_line).trim();
        if line.is_empty() {
            continue;
        }

        if !header_checked {
            header_checked = true;
            if line == "tlsprite_v1" {
                continue;
            }
            diagnostics.push(TlspriteDiagnostic {
                level: TlspriteDiagnosticLevel::Warning,
                line: line_no,
                message: "missing 'tlsprite_v1' header, continuing in compatibility mode"
                    .to_string(),
            });
        }

        if let Some(section) = parse_section_name(line) {
            if let Some(curr) = pending.take() {
                curr.finish(&mut diagnostics, extra_roots, &mut sprites, &mut lights);
            }
            pending = Some(PendingEntry::new(section.to_string(), line_no));
            continue;
        }

        let Some((key, value)) = line.split_once('=') else {
            diagnostics.push(TlspriteDiagnostic {
                level: TlspriteDiagnosticLevel::Error,
                line: line_no,
                message: "expected 'key = value' entry".to_string(),
            });
            continue;
        };
        let key = key.trim();
        let value = value.trim();

        let Some(curr) = pending.as_mut() else {
            diagnostics.push(TlspriteDiagnostic {
                level: TlspriteDiagnosticLevel::Error,
                line: line_no,
                message: "key-value entry outside [section]".to_string(),
            });
            continue;
        };
        curr.apply(key, value, line_no, &mut diagnostics);
    }

    if let Some(curr) = pending.take() {
        curr.finish(&mut diagnostics, extra_roots, &mut sprites, &mut lights);
    }

    if sprites.is_empty() && lights.is_empty() {
        diagnostics.push(TlspriteDiagnostic {
            level: TlspriteDiagnosticLevel::Error,
            line: 0,
            message: "no sprite/light sections were produced".to_string(),
        });
    }

    let has_errors = diagnostics
        .iter()
        .any(|d| d.level == TlspriteDiagnosticLevel::Error);
    TlspriteCompileOutcome {
        program: (!has_errors).then_some(TlspriteProgram { sprites, lights }),
        diagnostics,
    }
}

fn hash_bytes(bytes: &[u8]) -> u64 {
    let mut hasher = DefaultHasher::new();
    bytes.hash(&mut hasher);
    hasher.finish()
}

fn encode_tlsprite_pack(program: &TlspriteProgram, source_hash: u64) -> Vec<u8> {
    let mut out =
        Vec::with_capacity(64 + program.sprites().len() * 96 + program.lights().len() * 116);
    out.extend_from_slice(TLSPRITE_PACK_MAGIC);
    out.extend_from_slice(&source_hash.to_le_bytes());
    out.extend_from_slice(&(program.sprites().len() as u32).to_le_bytes());
    out.extend_from_slice(&(program.lights().len() as u32).to_le_bytes());
    for def in program.sprites() {
        write_len_prefixed_str(&mut out, &def.name);
        out.extend_from_slice(&def.sprite.sprite_id.to_le_bytes());
        out.extend_from_slice(&def.sprite.texture_slot.to_le_bytes());
        out.extend_from_slice(&def.sprite.layer.to_le_bytes());
        out.push(sprite_kind_to_tag(def.sprite.kind));
        write_f32s(&mut out, &def.sprite.position);
        write_f32s(&mut out, &def.sprite.size);
        out.extend_from_slice(&def.sprite.rotation_rad.to_le_bytes());
        write_f32s(&mut out, &def.sprite.color_rgba);
        match def.dynamic_scale {
            Some(scale) => {
                out.push(1);
                out.push(match scale.axis {
                    TlspriteScaleAxis::X => 0,
                    TlspriteScaleAxis::Y => 1,
                });
                out.push(match scale.source {
                    TlspriteScaleSource::SpawnProgress => 0,
                    TlspriteScaleSource::SpawnRemaining => 1,
                    TlspriteScaleSource::LiveBallRatio => 2,
                });
                out.extend_from_slice(&scale.min_factor.to_le_bytes());
                out.extend_from_slice(&scale.max_factor.to_le_bytes());
            }
            None => out.push(0),
        }
    }
    for def in program.lights() {
        write_len_prefixed_str(&mut out, &def.name);
        out.extend_from_slice(&def.light.id.to_le_bytes());
        out.push(light_kind_to_tag(def.light.kind));
        out.extend_from_slice(&def.light.layer.to_le_bytes());
        out.push(u8::from(def.light.enabled));
        out.push(u8::from(def.light.casts_shadow));
        write_f32s(&mut out, &def.light.position);
        write_f32s(&mut out, &def.light.direction);
        write_f32s(&mut out, &def.light.color);
        out.extend_from_slice(&def.light.intensity.to_le_bytes());
        out.extend_from_slice(&def.light.range.to_le_bytes());
        out.extend_from_slice(&def.light.inner_cone_deg.to_le_bytes());
        out.extend_from_slice(&def.light.outer_cone_deg.to_le_bytes());
        out.extend_from_slice(&def.light.softness.to_le_bytes());
        out.extend_from_slice(&def.light.specular_strength.to_le_bytes());
        // V4: glow + follow_camera fields
        out.push(u8::from(def.light.glow_enabled));
        out.extend_from_slice(&def.light.glow_radius_world.to_le_bytes());
        out.extend_from_slice(&def.light.glow_intensity_scale.to_le_bytes());
        out.push(u8::from(def.light.follow_camera));
    }
    out
}

fn decode_tlsprite_pack(bytes: &[u8]) -> Result<DecodedTlspritePack, String> {
    let mut rd = ByteReader::new(bytes);
    let version = rd.expect_pack_version()?;
    let source_hash = rd.read_u64()?;
    let sprite_count = rd.read_u32()? as usize;
    let light_count = match version {
        TlspritePackVersion::V3 | TlspritePackVersion::V4 => rd.read_u32()? as usize,
        TlspritePackVersion::V1 | TlspritePackVersion::V2 => 0,
    };
    let mut sprites = Vec::with_capacity(sprite_count);
    for _ in 0..sprite_count {
        let name = rd.read_string()?;
        let sprite_id = rd.read_u64()?;
        let texture_slot = rd.read_u16()?;
        let layer = rd.read_i16()?;
        let kind = match version {
            TlspritePackVersion::V1 => infer_sprite_kind(None, Some(layer)),
            TlspritePackVersion::V2 | TlspritePackVersion::V3 | TlspritePackVersion::V4 => {
                sprite_kind_from_tag(rd.read_u8()?)
                    .ok_or_else(|| "invalid packed sprite kind tag".to_string())?
            }
        };
        let position = rd.read_f32_array::<3>()?;
        let size = rd.read_f32_array::<2>()?;
        let rotation_rad = rd.read_f32()?;
        let color_rgba = rd.read_f32_array::<4>()?;
        let has_scale = rd.read_u8()?;
        let dynamic_scale = if has_scale == 0 {
            None
        } else {
            let axis = match rd.read_u8()? {
                0 => TlspriteScaleAxis::X,
                1 => TlspriteScaleAxis::Y,
                other => return Err(format!("invalid packed scale axis tag {other}")),
            };
            let source = match rd.read_u8()? {
                0 => TlspriteScaleSource::SpawnProgress,
                1 => TlspriteScaleSource::SpawnRemaining,
                2 => TlspriteScaleSource::LiveBallRatio,
                other => return Err(format!("invalid packed scale source tag {other}")),
            };
            let min_factor = rd.read_f32()?;
            let max_factor = rd.read_f32()?;
            Some(TlspriteDynamicScale {
                axis,
                source,
                min_factor,
                max_factor,
            })
        };
        sprites.push(TlspriteSpriteDef {
            name,
            sprite: SpriteInstance {
                sprite_id,
                kind,
                position,
                size,
                rotation_rad,
                color_rgba,
                texture_slot,
                layer,
            },
            fbx_source: None,
            texture_source: None,
            dynamic_scale,
        });
    }
    let mut lights = Vec::with_capacity(light_count);
    for _ in 0..light_count {
        let name = rd.read_string()?;
        let id = rd.read_u64()?;
        let kind = light_kind_from_tag(rd.read_u8()?)
            .ok_or_else(|| "invalid packed light kind tag".to_string())?;
        let layer = rd.read_i16()?;
        let enabled = rd.read_u8()? != 0;
        let casts_shadow = rd.read_u8()? != 0;
        let position = rd.read_f32_array::<3>()?;
        let direction = rd.read_f32_array::<3>()?;
        let color = rd.read_f32_array::<3>()?;
        let intensity = rd.read_f32()?;
        let range = rd.read_f32()?;
        let inner_cone_deg = rd.read_f32()?;
        let outer_cone_deg = rd.read_f32()?;
        let softness = rd.read_f32()?;
        let specular_strength = rd.read_f32()?;
        let (glow_enabled, glow_radius_world, glow_intensity_scale, follow_camera) = match version {
            TlspritePackVersion::V4 => {
                let ge = rd.read_u8()? != 0;
                let gr = rd.read_f32()?;
                let gi = rd.read_f32()?;
                let fc = rd.read_u8()? != 0;
                (ge, gr, gi, fc)
            }
            _ => (true, 1.2_f32, 1.0_f32, false),
        };
        lights.push(TlspriteLightDef {
            name,
            glow_enabled,
            light: SceneLight {
                id,
                enabled,
                kind,
                position,
                direction,
                color,
                intensity,
                range,
                inner_cone_deg,
                outer_cone_deg,
                softness,
                casts_shadow,
                specular_strength,
                layer,
                glow_enabled,
                glow_radius_world,
                glow_intensity_scale,
                follow_camera,
            },
        });
    }
    if rd.remaining() != 0 {
        return Err("packed data contains trailing bytes".to_string());
    }

    Ok(DecodedTlspritePack {
        program: TlspriteProgram { sprites, lights },
        source_hash,
    })
}

fn write_len_prefixed_str(out: &mut Vec<u8>, text: &str) {
    let bytes = text.as_bytes();
    let len = bytes.len().min(u16::MAX as usize) as u16;
    out.extend_from_slice(&len.to_le_bytes());
    out.extend_from_slice(&bytes[..len as usize]);
}

fn write_f32s<const N: usize>(out: &mut Vec<u8>, values: &[f32; N]) {
    for v in values {
        out.extend_from_slice(&v.to_le_bytes());
    }
}

struct ByteReader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> ByteReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.offset)
    }

    fn expect_pack_version(&mut self) -> Result<TlspritePackVersion, String> {
        let got = self.read_bytes(8)?;
        if got == TLSPRITE_PACK_MAGIC_V4 {
            Ok(TlspritePackVersion::V4)
        } else if got == TLSPRITE_PACK_MAGIC_V3 {
            Ok(TlspritePackVersion::V3)
        } else if got == TLSPRITE_PACK_MAGIC_V2 {
            Ok(TlspritePackVersion::V2)
        } else if got == TLSPRITE_PACK_MAGIC_V1 {
            Ok(TlspritePackVersion::V1)
        } else {
            Err("invalid tlsprite pack magic/version".to_string())
        }
    }

    fn read_bytes(&mut self, len: usize) -> Result<&'a [u8], String> {
        if self.remaining() < len {
            return Err("unexpected EOF while decoding tlsprite pack".to_string());
        }
        let start = self.offset;
        self.offset += len;
        Ok(&self.bytes[start..self.offset])
    }

    fn read_u8(&mut self) -> Result<u8, String> {
        Ok(self.read_bytes(1)?[0])
    }

    fn read_u16(&mut self) -> Result<u16, String> {
        let mut buf = [0u8; 2];
        buf.copy_from_slice(self.read_bytes(2)?);
        Ok(u16::from_le_bytes(buf))
    }

    fn read_i16(&mut self) -> Result<i16, String> {
        let mut buf = [0u8; 2];
        buf.copy_from_slice(self.read_bytes(2)?);
        Ok(i16::from_le_bytes(buf))
    }

    fn read_u32(&mut self) -> Result<u32, String> {
        let mut buf = [0u8; 4];
        buf.copy_from_slice(self.read_bytes(4)?);
        Ok(u32::from_le_bytes(buf))
    }

    fn read_u64(&mut self) -> Result<u64, String> {
        let mut buf = [0u8; 8];
        buf.copy_from_slice(self.read_bytes(8)?);
        Ok(u64::from_le_bytes(buf))
    }

    fn read_f32(&mut self) -> Result<f32, String> {
        let mut buf = [0u8; 4];
        buf.copy_from_slice(self.read_bytes(4)?);
        Ok(f32::from_le_bytes(buf))
    }

    fn read_f32_array<const N: usize>(&mut self) -> Result<[f32; N], String> {
        let mut out = [0.0f32; N];
        for slot in &mut out {
            *slot = self.read_f32()?;
        }
        Ok(out)
    }

    fn read_string(&mut self) -> Result<String, String> {
        let len = self.read_u16()? as usize;
        let bytes = self.read_bytes(len)?;
        std::str::from_utf8(bytes)
            .map(|s| s.to_string())
            .map_err(|err| format!("invalid UTF-8 string in pack: {err}"))
    }
}

fn event_can_change_source(kind: &EventKind) -> bool {
    !matches!(kind, EventKind::Access(_))
}

fn event_targets_path(event: &Event, target: &Path) -> bool {
    if event.paths.is_empty() {
        return true;
    }
    let target_file_name = target.file_name();
    event.paths.iter().any(|p| {
        p == target
            || target_file_name
                .zip(p.file_name())
                .map(|(a, b)| a == b)
                .unwrap_or(false)
    })
}

fn strip_comments(line: &str) -> &str {
    let mut cut = line.len();
    if let Some(i) = line.find('#') {
        cut = cut.min(i);
    }
    if let Some(i) = line.find("//") {
        cut = cut.min(i);
    }
    &line[..cut]
}

fn parse_section_name(line: &str) -> Option<&str> {
    let trimmed = line.trim();
    if trimmed.len() >= 3 && trimmed.starts_with('[') && trimmed.ends_with(']') {
        let inner = trimmed[1..trimmed.len() - 1].trim();
        if !inner.is_empty() {
            return Some(inner);
        }
    }
    None
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TlspriteEntryKind {
    Sprite,
    Light,
}

impl Default for TlspriteEntryKind {
    fn default() -> Self {
        Self::Sprite
    }
}

#[derive(Debug, Clone)]
struct PendingEntry {
    name: String,
    line_started: usize,
    entry_kind: TlspriteEntryKind,
    // sprite keys
    sprite_id: Option<u64>,
    kind: Option<SpriteKind>,
    fbx: Option<String>,
    texture: Option<String>,
    texture_slot: Option<u16>,
    layer: Option<i16>,
    position: Option<[f32; 3]>,
    size: Option<[f32; 2]>,
    rotation_rad: Option<f32>,
    color: Option<[f32; 4]>,
    scale_axis: Option<TlspriteScaleAxis>,
    scale_source: Option<TlspriteScaleSource>,
    scale_min: Option<f32>,
    scale_max: Option<f32>,
    // light keys
    light_id: Option<u64>,
    light_type: Option<SceneLightKind>,
    direction: Option<[f32; 3]>,
    intensity: Option<f32>,
    range: Option<f32>,
    inner_cone_deg: Option<f32>,
    outer_cone_deg: Option<f32>,
    softness: Option<f32>,
    casts_shadow: Option<bool>,
    specular_strength: Option<f32>,
    glow: Option<bool>,
    glow_radius: Option<f32>,
    glow_intensity_scale: Option<f32>,
    follow_camera: Option<bool>,
}

impl PendingEntry {
    fn new(name: String, line_started: usize) -> Self {
        Self {
            name,
            line_started,
            entry_kind: TlspriteEntryKind::Sprite,
            sprite_id: None,
            kind: None,
            fbx: None,
            texture: None,
            texture_slot: None,
            layer: None,
            position: None,
            size: None,
            rotation_rad: None,
            color: None,
            scale_axis: None,
            scale_source: None,
            scale_min: None,
            scale_max: None,
            light_id: None,
            light_type: None,
            direction: None,
            intensity: None,
            range: None,
            inner_cone_deg: None,
            outer_cone_deg: None,
            softness: None,
            casts_shadow: None,
            specular_strength: None,
            glow: None,
            glow_radius: None,
            glow_intensity_scale: None,
            follow_camera: None,
        }
    }

    fn apply(
        &mut self,
        key: &str,
        value: &str,
        line: usize,
        diagnostics: &mut Vec<TlspriteDiagnostic>,
    ) {
        match key {
            "entry_kind" => match value.trim().to_ascii_lowercase().as_str() {
                "sprite" => self.entry_kind = TlspriteEntryKind::Sprite,
                "light" => self.entry_kind = TlspriteEntryKind::Light,
                other => diagnostics.push(TlspriteDiagnostic {
                    level: TlspriteDiagnosticLevel::Error,
                    line,
                    message: format!("invalid entry_kind '{other}' (expected sprite|light)"),
                }),
            },
            "sprite_id" => {
                self.sprite_id = parse_scalar::<u64>(value, line, key, diagnostics);
            }
            "kind" => {
                self.kind = parse_sprite_kind(value, line, diagnostics);
            }
            "fbx" => {
                let raw = value.trim();
                if raw.is_empty() {
                    diagnostics.push(TlspriteDiagnostic {
                        level: TlspriteDiagnosticLevel::Warning,
                        line,
                        message: "key 'fbx' is empty and will be ignored".to_string(),
                    });
                } else {
                    self.fbx = Some(raw.to_string());
                }
            }
            "texture" | "sprite_texture" => {
                let raw = value.trim();
                if raw.is_empty() {
                    diagnostics.push(TlspriteDiagnostic {
                        level: TlspriteDiagnosticLevel::Warning,
                        line,
                        message: "key 'texture' is empty and will be ignored".to_string(),
                    });
                } else {
                    self.texture = Some(raw.to_string());
                }
            }
            "texture_slot" => {
                self.texture_slot = parse_scalar::<u16>(value, line, key, diagnostics);
            }
            "layer" => {
                self.layer = parse_scalar::<i16>(value, line, key, diagnostics);
            }
            "position" => {
                self.position = parse_vec3(value, line, key, diagnostics);
            }
            "size" => {
                self.size = parse_vec2(value, line, key, diagnostics);
            }
            "rotation_rad" => {
                self.rotation_rad = parse_scalar::<f32>(value, line, key, diagnostics);
            }
            "color" => {
                self.color = parse_vec4(value, line, key, diagnostics);
            }
            "scale_axis" => {
                self.scale_axis = parse_scale_axis(value, line, diagnostics);
            }
            "scale_source" => {
                self.scale_source = parse_scale_source(value, line, diagnostics);
            }
            "scale_min" => {
                self.scale_min = parse_scalar::<f32>(value, line, key, diagnostics);
            }
            "scale_max" => {
                self.scale_max = parse_scalar::<f32>(value, line, key, diagnostics);
            }
            "light_id" => {
                self.light_id = parse_scalar::<u64>(value, line, key, diagnostics);
            }
            "light_type" => {
                self.light_type = parse_light_kind(value, line, diagnostics);
            }
            "direction" => {
                self.direction = parse_vec3(value, line, key, diagnostics);
            }
            "intensity" => {
                self.intensity = parse_scalar::<f32>(value, line, key, diagnostics);
            }
            "range" => {
                self.range = parse_scalar::<f32>(value, line, key, diagnostics);
            }
            "inner_cone_deg" => {
                self.inner_cone_deg = parse_scalar::<f32>(value, line, key, diagnostics);
            }
            "outer_cone_deg" => {
                self.outer_cone_deg = parse_scalar::<f32>(value, line, key, diagnostics);
            }
            "softness" => {
                self.softness = parse_scalar::<f32>(value, line, key, diagnostics);
            }
            "casts_shadow" => {
                self.casts_shadow = parse_scalar::<bool>(value, line, key, diagnostics);
            }
            "specular_strength" => {
                self.specular_strength = parse_scalar::<f32>(value, line, key, diagnostics);
            }
            "glow" => {
                self.glow = parse_scalar::<bool>(value, line, key, diagnostics);
            }
            "glow_radius" => {
                self.glow_radius = parse_scalar::<f32>(value, line, key, diagnostics);
            }
            "glow_intensity_scale" => {
                self.glow_intensity_scale = parse_scalar::<f32>(value, line, key, diagnostics);
            }
            "follow_camera" => {
                self.follow_camera = parse_scalar::<bool>(value, line, key, diagnostics);
            }
            _ => diagnostics.push(TlspriteDiagnostic {
                level: TlspriteDiagnosticLevel::Warning,
                line,
                message: format!("unknown key '{key}' ignored"),
            }),
        }
    }

    fn finish(
        self,
        diagnostics: &mut Vec<TlspriteDiagnostic>,
        extra_roots: &[PathBuf],
        sprites: &mut Vec<TlspriteSpriteDef>,
        lights: &mut Vec<TlspriteLightDef>,
    ) {
        match self.entry_kind {
            TlspriteEntryKind::Sprite => {
                if let Some(def) = self.finish_sprite(diagnostics, extra_roots) {
                    sprites.push(def);
                }
            }
            TlspriteEntryKind::Light => {
                if let Some(def) = self.finish_light(diagnostics) {
                    lights.push(def);
                }
            }
        }
    }

    fn finish_sprite(
        self,
        diagnostics: &mut Vec<TlspriteDiagnostic>,
        extra_roots: &[PathBuf],
    ) -> Option<TlspriteSpriteDef> {
        let Some(sprite_id) = self.sprite_id else {
            diagnostics.push(TlspriteDiagnostic {
                level: TlspriteDiagnosticLevel::Error,
                line: self.line_started,
                message: format!(
                    "section '{}' is missing required key 'sprite_id'",
                    self.name
                ),
            });
            return None;
        };

        let kind = infer_sprite_kind(self.kind, self.layer);
        let defaults = sprite_kind_defaults(kind);
        let fbx_hints = self.fbx.as_deref().and_then(|raw_path| {
            match infer_sprite_hints_from_fbx_path(raw_path, extra_roots) {
                Ok(hints) => Some(hints),
                Err(err) => {
                    diagnostics.push(TlspriteDiagnostic {
                        level: TlspriteDiagnosticLevel::Warning,
                        line: self.line_started,
                        message: format!(
                            "section '{}' failed to parse fbx '{}': {err}; using sprite defaults",
                            self.name, raw_path
                        ),
                    });
                    None
                }
            }
        });
        let texture_hints = self.texture.as_deref().and_then(|raw_path| {
            match infer_sprite_hints_from_texture_path(raw_path, extra_roots) {
                Ok(hints) => Some(hints),
                Err(err) => {
                    diagnostics.push(TlspriteDiagnostic {
                        level: TlspriteDiagnosticLevel::Warning,
                        line: self.line_started,
                        message: format!(
                            "section '{}' failed to parse texture '{}': {err}; using sprite defaults",
                            self.name, raw_path
                        ),
                    });
                    None
                }
            }
        });

        let mut color = self.color.unwrap_or(defaults.color_rgba);
        for c in &mut color {
            *c = c.clamp(0.0, 1.0);
        }
        let mut size = self.size.unwrap_or_else(|| {
            if let Some(hints) = texture_hints {
                hints.suggested_size(defaults.size)
            } else {
                fbx_hints
                    .map(|h| h.suggested_size(defaults.size))
                    .unwrap_or(defaults.size)
            }
        });
        size[0] = size[0].max(0.001);
        size[1] = size[1].max(0.001);

        let dynamic_scale = if let Some(axis) = self.scale_axis {
            let mut min_factor = self.scale_min.unwrap_or(0.0).clamp(0.0, 4.0);
            let mut max_factor = self.scale_max.unwrap_or(1.0).clamp(0.0, 4.0);
            if max_factor < min_factor {
                std::mem::swap(&mut min_factor, &mut max_factor);
            }
            Some(TlspriteDynamicScale {
                axis,
                source: self
                    .scale_source
                    .unwrap_or(TlspriteScaleSource::SpawnProgress),
                min_factor,
                max_factor,
            })
        } else if self.scale_source.is_some()
            || self.scale_min.is_some()
            || self.scale_max.is_some()
        {
            diagnostics.push(TlspriteDiagnostic {
                level: TlspriteDiagnosticLevel::Warning,
                line: self.line_started,
                message: format!(
                    "section '{}' defines scale_* keys but no 'scale_axis'; dynamic scaling disabled",
                    self.name
                ),
            });
            None
        } else {
            None
        };

        Some(TlspriteSpriteDef {
            name: self.name,
            sprite: SpriteInstance {
                sprite_id,
                kind,
                position: self.position.unwrap_or(defaults.position),
                size,
                rotation_rad: self.rotation_rad.unwrap_or(defaults.rotation_rad),
                color_rgba: color,
                texture_slot: self
                    .texture_slot
                    .or_else(|| texture_hints.map(|h| h.texture_slot))
                    .or_else(|| fbx_hints.map(|h| h.texture_slot))
                    .unwrap_or(defaults.texture_slot),
                layer: self.layer.unwrap_or(defaults.layer),
            },
            fbx_source: self.fbx,
            texture_source: self.texture,
            dynamic_scale,
        })
    }

    fn finish_light(self, diagnostics: &mut Vec<TlspriteDiagnostic>) -> Option<TlspriteLightDef> {
        let Some(light_id) = self.light_id else {
            diagnostics.push(TlspriteDiagnostic {
                level: TlspriteDiagnosticLevel::Error,
                line: self.line_started,
                message: format!("section '{}' is missing required key 'light_id'", self.name),
            });
            return None;
        };
        let kind = self.light_type.unwrap_or(SceneLightKind::Point);
        let position = self.position.unwrap_or([0.0, 4.0, 0.0]);
        let mut direction = self.direction.unwrap_or([0.0, -1.0, 0.0]);
        let mut color = self
            .color
            .map(|rgba| [rgba[0], rgba[1], rgba[2]])
            .unwrap_or([1.0; 3]);
        for c in &mut color {
            *c = c.clamp(0.0, 16.0);
        }
        let intensity = self.intensity.unwrap_or(5.0).clamp(0.0, 1000.0);
        let range = self.range.unwrap_or(30.0).clamp(0.05, 10_000.0);
        let mut inner_cone_deg = self.inner_cone_deg.unwrap_or(20.0).clamp(0.0, 89.0);
        let mut outer_cone_deg = self.outer_cone_deg.unwrap_or(35.0).clamp(0.1, 89.9);
        if outer_cone_deg < inner_cone_deg {
            std::mem::swap(&mut outer_cone_deg, &mut inner_cone_deg);
        }
        let softness = self.softness.unwrap_or(0.35).clamp(0.0, 1.0);
        let casts_shadow = self.casts_shadow.unwrap_or(true);
        let specular_strength = self.specular_strength.unwrap_or(1.0).clamp(0.0, 8.0);
        let glow_enabled = self.glow.unwrap_or(true);
        let glow_radius_world = self.glow_radius.unwrap_or(1.2).clamp(0.05, 50.0);
        let glow_intensity_scale = self.glow_intensity_scale.unwrap_or(1.0).clamp(0.0, 8.0);
        let follow_camera = self.follow_camera.unwrap_or(false);

        let len = (direction[0] * direction[0]
            + direction[1] * direction[1]
            + direction[2] * direction[2])
            .sqrt();
        if len <= 1e-6 {
            direction = [0.0, -1.0, 0.0];
        } else {
            direction[0] /= len;
            direction[1] /= len;
            direction[2] /= len;
        }

        Some(TlspriteLightDef {
            name: self.name,
            glow_enabled,
            light: SceneLight {
                id: light_id,
                enabled: true,
                kind,
                position,
                direction,
                color,
                intensity,
                range,
                inner_cone_deg,
                outer_cone_deg,
                softness,
                casts_shadow,
                specular_strength,
                layer: self.layer.unwrap_or(0),
                glow_enabled,
                glow_radius_world,
                glow_intensity_scale,
                follow_camera,
            },
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct SpriteKindDefaults {
    position: [f32; 3],
    size: [f32; 2],
    rotation_rad: f32,
    color_rgba: [f32; 4],
    texture_slot: u16,
    layer: i16,
}

fn infer_sprite_kind(explicit: Option<SpriteKind>, layer: Option<i16>) -> SpriteKind {
    if let Some(kind) = explicit {
        return kind;
    }
    if layer.unwrap_or(0) >= 100 {
        SpriteKind::Hud
    } else {
        SpriteKind::Generic
    }
}

fn sprite_kind_defaults(kind: SpriteKind) -> SpriteKindDefaults {
    match kind {
        SpriteKind::Generic => SpriteKindDefaults {
            position: [0.0, 0.0, 0.0],
            size: [0.1, 0.1],
            rotation_rad: 0.0,
            color_rgba: [1.0, 1.0, 1.0, 1.0],
            texture_slot: 0,
            layer: 0,
        },
        SpriteKind::Hud => SpriteKindDefaults {
            position: [0.0, 0.0, 0.0],
            size: [0.1, 0.1],
            rotation_rad: 0.0,
            color_rgba: [1.0, 1.0, 1.0, 1.0],
            texture_slot: 0,
            layer: 100,
        },
        SpriteKind::Camera => SpriteKindDefaults {
            position: [0.88, 0.84, 0.0],
            size: [0.11, 0.11],
            rotation_rad: 0.0,
            color_rgba: [0.62, 0.82, 1.0, 0.92],
            texture_slot: 2,
            layer: 150,
        },
        SpriteKind::Terrain => SpriteKindDefaults {
            position: [0.0, -0.92, 0.0],
            size: [1.62, 0.14],
            rotation_rad: 0.0,
            color_rgba: [0.23, 0.70, 0.28, 0.95],
            texture_slot: 3,
            layer: -40,
        },
        SpriteKind::LightGlow => SpriteKindDefaults {
            position: [0.0, 0.0, 0.0],
            size: [0.2, 0.2],
            rotation_rad: 0.0,
            color_rgba: [1.0, 1.0, 0.8, 1.0],
            texture_slot: 0,
            layer: 100,
        },
    }
}

fn sprite_kind_to_tag(kind: SpriteKind) -> u8 {
    match kind {
        SpriteKind::Generic => 0,
        SpriteKind::Hud => 1,
        SpriteKind::Camera => 2,
        SpriteKind::Terrain => 3,
        SpriteKind::LightGlow => 4,
    }
}

fn sprite_kind_from_tag(tag: u8) -> Option<SpriteKind> {
    match tag {
        0 => Some(SpriteKind::Generic),
        1 => Some(SpriteKind::Hud),
        2 => Some(SpriteKind::Camera),
        3 => Some(SpriteKind::Terrain),
        _ => None,
    }
}

fn light_kind_to_tag(kind: SceneLightKind) -> u8 {
    match kind {
        SceneLightKind::Point => 0,
        SceneLightKind::Spot => 1,
    }
}

fn light_kind_from_tag(tag: u8) -> Option<SceneLightKind> {
    match tag {
        0 => Some(SceneLightKind::Point),
        1 => Some(SceneLightKind::Spot),
        _ => None,
    }
}

#[derive(Debug, Clone, Copy)]
struct FbxSpriteHints {
    aspect: f32,
    texture_slot: u16,
}

#[derive(Debug, Clone, Copy)]
struct TextureSpriteHints {
    aspect: f32,
    texture_slot: u16,
}

impl TextureSpriteHints {
    fn suggested_size(self, base: [f32; 2]) -> [f32; 2] {
        let aspect = self.aspect.clamp(0.25, 4.0);
        let mut out = if aspect >= 1.0 {
            [base[0] * aspect, base[1]]
        } else {
            [base[0], base[1] / aspect.max(1e-3)]
        };
        out[0] = out[0].clamp(0.001, 2.0);
        out[1] = out[1].clamp(0.001, 2.0);
        out
    }
}

impl FbxSpriteHints {
    fn suggested_size(self, base: [f32; 2]) -> [f32; 2] {
        let aspect = self.aspect.clamp(0.25, 4.0);
        let mut out = if aspect >= 1.0 {
            [base[0] * aspect, base[1]]
        } else {
            [base[0], base[1] / aspect.max(1e-3)]
        };
        out[0] = out[0].clamp(0.001, 2.0);
        out[1] = out[1].clamp(0.001, 2.0);
        out
    }
}

fn infer_sprite_hints_from_fbx_path(
    raw_path: &str,
    extra_roots: &[PathBuf],
) -> Result<FbxSpriteHints, String> {
    let path = resolve_existing_asset_path(raw_path, extra_roots)?;
    let bytes =
        fs::read(&path).map_err(|err| format!("failed to read '{}': {err}", path.display()))?;
    let file = fbx::File::read_from(Cursor::new(bytes))
        .map_err(|err| format!("failed to parse '{}': {err}", path.display()))?;
    let vertices = first_fbx_mesh_vertices(&file)
        .ok_or_else(|| format!("no mesh vertices found in '{}'", path.display()))?;
    if vertices.len() < 9 || vertices.len() % 3 != 0 {
        return Err("mesh vertices are malformed".to_string());
    }

    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut min_z = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    let mut max_z = f64::NEG_INFINITY;
    for p in vertices.chunks_exact(3) {
        min_x = min_x.min(p[0]);
        min_y = min_y.min(p[1]);
        min_z = min_z.min(p[2]);
        max_x = max_x.max(p[0]);
        max_y = max_y.max(p[1]);
        max_z = max_z.max(p[2]);
    }

    let extent_x = (max_x - min_x).abs() as f32;
    let extent_y = (max_y - min_y).abs() as f32;
    let extent_z = (max_z - min_z).abs() as f32;
    let horizontal_extent = extent_x.max(extent_z).max(1e-5);
    let vertical_extent = extent_y.max(1e-5);
    let aspect = (horizontal_extent / vertical_extent).clamp(0.25, 4.0);

    let mut hasher = DefaultHasher::new();
    path.to_string_lossy().hash(&mut hasher);
    let texture_slot = (hasher.finish() % 4) as u16;

    Ok(FbxSpriteHints {
        aspect,
        texture_slot,
    })
}

fn resolve_existing_asset_path(raw_path: &str, extra_roots: &[PathBuf]) -> Result<PathBuf, String> {
    let candidate = PathBuf::from(raw_path);
    if candidate.is_absolute() {
        if candidate.exists() {
            return Ok(candidate);
        }
        return Err(format!("path '{}' does not exist", candidate.display()));
    }

    let mut candidates = Vec::with_capacity(4);
    candidates.push(candidate.clone());
    if let Ok(cwd) = std::env::current_dir() {
        candidates.push(cwd.join(&candidate));
    }
    for root in extra_roots {
        candidates.push(root.join(&candidate));
    }
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    candidates.push(manifest_dir.join(&candidate));
    candidates.push(manifest_dir.join("..").join(&candidate));

    for path in candidates {
        if path.exists() {
            return Ok(path);
        }
    }
    Err(format!("path '{}' does not exist in known roots", raw_path))
}

fn infer_sprite_hints_from_texture_path(
    raw_path: &str,
    extra_roots: &[PathBuf],
) -> Result<TextureSpriteHints, String> {
    let path = resolve_existing_asset_path(raw_path, extra_roots)?;
    let ext = path
        .extension()
        .and_then(|value| value.to_str())
        .map(|value| value.to_ascii_lowercase())
        .ok_or_else(|| format!("texture '{}' has no extension", path.display()))?;

    let (width, height) = match ext.as_str() {
        "png" => {
            let reader = ImageReader::open(&path)
                .map_err(|err| format!("failed to open png '{}': {err}", path.display()))?
                .with_guessed_format()
                .map_err(|err| {
                    format!("failed to detect image format '{}': {err}", path.display())
                })?;
            if !matches!(reader.format(), Some(image::ImageFormat::Png)) {
                return Err(format!("texture '{}' is not a PNG file", path.display()));
            }
            reader.into_dimensions().map_err(|err| {
                format!("failed to read png dimensions '{}': {err}", path.display())
            })?
        }
        "svg" => {
            let bytes = fs::read(&path)
                .map_err(|err| format!("failed to read svg '{}': {err}", path.display()))?;
            let options = usvg::Options::default();
            let tree = usvg::Tree::from_data(&bytes, &options)
                .map_err(|err| format!("failed to parse svg '{}': {err}", path.display()))?;
            let size = tree.size();
            (
                size.width().round().max(1.0) as u32,
                size.height().round().max(1.0) as u32,
            )
        }
        other => {
            return Err(format!(
                "texture '{}' has unsupported extension '{}'; expected .png or .svg",
                path.display(),
                other
            ));
        }
    };

    let width = width.max(1);
    let height = height.max(1);
    let aspect = (width as f32 / height as f32).clamp(0.25, 4.0);
    let texture_slot = infer_texture_slot_from_path(&path);
    Ok(TextureSpriteHints {
        aspect,
        texture_slot,
    })
}

fn infer_texture_slot_from_path(path: &Path) -> u16 {
    // Reserve [64..127] for external image atlas uploads.
    const IMAGE_SLOT_BASE: u16 = 64;
    const IMAGE_SLOT_COUNT: u16 = 64;
    let mut hasher = DefaultHasher::new();
    path.to_string_lossy().hash(&mut hasher);
    IMAGE_SLOT_BASE + (hasher.finish() % IMAGE_SLOT_COUNT as u64) as u16
}

/// Decode a sprite texture source (`.png` or `.svg`) into RGBA8 pixels.
pub fn decode_sprite_texture_to_rgba(
    path: &Path,
    target_width: u32,
    target_height: u32,
) -> Result<Vec<u8>, String> {
    let ext = path
        .extension()
        .and_then(|value| value.to_str())
        .map(|value| value.to_ascii_lowercase())
        .ok_or_else(|| format!("texture '{}' has no extension", path.display()))?;

    let width = target_width.max(1);
    let height = target_height.max(1);
    match ext.as_str() {
        "png" => {
            let image = ImageReader::open(path)
                .map_err(|err| format!("failed to open png '{}': {err}", path.display()))?
                .with_guessed_format()
                .map_err(|err| {
                    format!("failed to detect image format '{}': {err}", path.display())
                })?
                .decode()
                .map_err(|err| format!("failed to decode png '{}': {err}", path.display()))?;
            let resized = image.resize_exact(width, height, FilterType::Lanczos3);
            Ok(resized.to_rgba8().into_raw())
        }
        "svg" => {
            let bytes = fs::read(path)
                .map_err(|err| format!("failed to read svg '{}': {err}", path.display()))?;
            let options = usvg::Options::default();
            let tree = usvg::Tree::from_data(&bytes, &options)
                .map_err(|err| format!("failed to parse svg '{}': {err}", path.display()))?;
            let mut pixmap = tiny_skia::Pixmap::new(width, height)
                .ok_or_else(|| format!("failed to allocate svg raster target {width}x{height}"))?;
            let size = tree.size();
            let sx = width as f32 / size.width().max(1e-6);
            let sy = height as f32 / size.height().max(1e-6);
            let transform = tiny_skia::Transform::from_scale(sx, sy);
            resvg::render(&tree, transform, &mut pixmap.as_mut());
            Ok(pixmap.take())
        }
        other => Err(format!(
            "unsupported sprite texture extension '{}'; expected .png or .svg",
            other
        )),
    }
}

fn first_fbx_mesh_vertices(file: &fbx::File) -> Option<&[f64]> {
    let objects = file.children.iter().find(|node| node.name == "Objects")?;
    for geometry in objects
        .children
        .iter()
        .filter(|node| node.name == "Geometry")
    {
        let kind = geometry
            .properties
            .get(2)
            .and_then(|prop| match prop {
                FbxProperty::String(text) => Some(text.as_str()),
                _ => None,
            })
            .unwrap_or_default();
        if kind != "Mesh" {
            continue;
        }
        let vertices = geometry
            .children
            .iter()
            .find(|node| node.name == "Vertices")
            .and_then(|node| node.properties.first())
            .and_then(|prop| match prop {
                FbxProperty::F64Array(values) => Some(values.as_slice()),
                _ => None,
            });
        if vertices.is_some() {
            return vertices;
        }
    }
    None
}

fn parse_sprite_kind(
    value: &str,
    line: usize,
    diagnostics: &mut Vec<TlspriteDiagnostic>,
) -> Option<SpriteKind> {
    match value.trim() {
        "generic" => Some(SpriteKind::Generic),
        "hud" | "ui" => Some(SpriteKind::Hud),
        "camera" => Some(SpriteKind::Camera),
        "terrain" => Some(SpriteKind::Terrain),
        "light_glow" | "glow" => Some(SpriteKind::LightGlow),
        other => {
            diagnostics.push(TlspriteDiagnostic {
                level: TlspriteDiagnosticLevel::Error,
                line,
                message: format!(
                    "invalid kind '{other}', expected 'generic', 'hud', 'camera', 'terrain', or 'light_glow'"
                ),
            });
            None
        }
    }
}

fn parse_light_kind(
    value: &str,
    line: usize,
    diagnostics: &mut Vec<TlspriteDiagnostic>,
) -> Option<SceneLightKind> {
    match value.trim() {
        "point" => Some(SceneLightKind::Point),
        "spot" => Some(SceneLightKind::Spot),
        other => {
            diagnostics.push(TlspriteDiagnostic {
                level: TlspriteDiagnosticLevel::Error,
                line,
                message: format!("invalid light_type '{other}', expected 'point' or 'spot'"),
            });
            None
        }
    }
}

fn parse_scale_axis(
    value: &str,
    line: usize,
    diagnostics: &mut Vec<TlspriteDiagnostic>,
) -> Option<TlspriteScaleAxis> {
    match value.trim() {
        "x" | "X" => Some(TlspriteScaleAxis::X),
        "y" | "Y" => Some(TlspriteScaleAxis::Y),
        other => {
            diagnostics.push(TlspriteDiagnostic {
                level: TlspriteDiagnosticLevel::Error,
                line,
                message: format!("invalid scale_axis '{other}', expected 'x' or 'y'"),
            });
            None
        }
    }
}

fn parse_scale_source(
    value: &str,
    line: usize,
    diagnostics: &mut Vec<TlspriteDiagnostic>,
) -> Option<TlspriteScaleSource> {
    match value.trim() {
        "spawn_progress" => Some(TlspriteScaleSource::SpawnProgress),
        "spawn_remaining" => Some(TlspriteScaleSource::SpawnRemaining),
        "live_ball_ratio" => Some(TlspriteScaleSource::LiveBallRatio),
        other => {
            diagnostics.push(TlspriteDiagnostic {
                level: TlspriteDiagnosticLevel::Error,
                line,
                message: format!(
                    "invalid scale_source '{other}', expected 'spawn_progress', 'spawn_remaining', or 'live_ball_ratio'"
                ),
            });
            None
        }
    }
}

fn parse_scalar<T: std::str::FromStr>(
    raw: &str,
    line: usize,
    key: &str,
    diagnostics: &mut Vec<TlspriteDiagnostic>,
) -> Option<T> {
    match raw.trim().parse::<T>() {
        Ok(v) => Some(v),
        Err(_) => {
            diagnostics.push(TlspriteDiagnostic {
                level: TlspriteDiagnosticLevel::Error,
                line,
                message: format!("failed to parse key '{key}'"),
            });
            None
        }
    }
}

fn parse_vec2(
    raw: &str,
    line: usize,
    key: &str,
    diagnostics: &mut Vec<TlspriteDiagnostic>,
) -> Option<[f32; 2]> {
    parse_f32_components(raw, 2, line, key, diagnostics).map(|v| [v[0], v[1]])
}

fn parse_vec3(
    raw: &str,
    line: usize,
    key: &str,
    diagnostics: &mut Vec<TlspriteDiagnostic>,
) -> Option<[f32; 3]> {
    parse_f32_components(raw, 3, line, key, diagnostics).map(|v| [v[0], v[1], v[2]])
}

fn parse_vec4(
    raw: &str,
    line: usize,
    key: &str,
    diagnostics: &mut Vec<TlspriteDiagnostic>,
) -> Option<[f32; 4]> {
    parse_f32_components(raw, 4, line, key, diagnostics).map(|v| [v[0], v[1], v[2], v[3]])
}

fn parse_f32_components(
    raw: &str,
    expected: usize,
    line: usize,
    key: &str,
    diagnostics: &mut Vec<TlspriteDiagnostic>,
) -> Option<Vec<f32>> {
    let mut parts = Vec::with_capacity(expected);
    for chunk in raw.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        match chunk.parse::<f32>() {
            Ok(v) => parts.push(v),
            Err(_) => {
                diagnostics.push(TlspriteDiagnostic {
                    level: TlspriteDiagnosticLevel::Error,
                    line,
                    message: format!("failed to parse float component in key '{key}'"),
                });
                return None;
            }
        }
    }
    if parts.len() != expected {
        diagnostics.push(TlspriteDiagnostic {
            level: TlspriteDiagnosticLevel::Error,
            line,
            message: format!("key '{key}' expects {expected} comma-separated float values"),
        });
        return None;
    }
    Some(parts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageFormat, Rgba, RgbaImage};
    use std::{
        fs,
        path::PathBuf,
        time::{Duration, SystemTime, UNIX_EPOCH},
    };

    #[test]
    fn parses_tlsprite_and_emits_scaled_sprite() {
        let src = concat!(
            "tlsprite_v1\n",
            "[progress]\n",
            "sprite_id = 7\n",
            "kind = hud\n",
            "texture_slot = 2\n",
            "layer = 100\n",
            "position = -0.5, 0.8, 0.0\n",
            "size = 0.4, 0.05\n",
            "rotation_rad = 0.0\n",
            "color = 0.2, 0.7, 0.9, 1.0\n",
            "scale_axis = x\n",
            "scale_source = spawn_progress\n",
            "scale_min = 0.1\n",
            "scale_max = 1.0\n",
        );
        let out = compile_tlsprite(src);
        assert!(!out.has_errors());
        let program = out.program.as_ref().expect("program");
        assert_eq!(program.sprites().len(), 1);

        let mut instances = Vec::new();
        program.emit_instances(
            TlspriteFrameContext {
                spawn_progress: 0.5,
                live_balls: 500,
                target_balls: 1_000,
            },
            &mut instances,
        );
        assert_eq!(instances.len(), 1);
        assert!((instances[0].size[0] - 0.2).abs() < 1e-6);
        assert_eq!(instances[0].kind, SpriteKind::Hud);
    }

    #[test]
    fn parses_camera_and_terrain_kinds_with_defaults() {
        let src = concat!(
            "tlsprite_v1\n",
            "[camera.icon]\n",
            "sprite_id = 12\n",
            "kind = camera\n",
            "[terrain.bar]\n",
            "sprite_id = 13\n",
            "kind = terrain\n",
        );
        let out = compile_tlsprite(src);
        assert!(!out.has_errors());
        let program = out.program.expect("program");
        assert_eq!(program.sprites().len(), 2);
        assert_eq!(program.sprites()[0].sprite.kind, SpriteKind::Camera);
        assert_eq!(program.sprites()[1].sprite.kind, SpriteKind::Terrain);
        assert!(program.sprites()[0].sprite.layer > 0);
        assert!(program.sprites()[1].sprite.layer < 0);
    }

    #[test]
    fn parses_light_entries_with_required_fields() {
        let src = concat!(
            "tlsprite_v1\n",
            "[flash]\n",
            "entry_kind = light\n",
            "light_id = 42\n",
            "light_type = spot\n",
            "position = 0.0, 5.0, 2.0\n",
            "direction = 0.0, -1.0, 0.0\n",
            "color = 1.0, 0.9, 0.8, 1.0\n",
            "intensity = 6.0\n",
            "range = 32.0\n",
            "inner_cone_deg = 20.0\n",
            "outer_cone_deg = 32.0\n",
            "softness = 0.4\n",
            "casts_shadow = true\n",
            "specular_strength = 1.4\n",
        );
        let out = compile_tlsprite(src);
        assert!(!out.has_errors());
        let program = out.program.expect("program");
        assert_eq!(program.lights().len(), 1);
        assert_eq!(program.lights()[0].light.id, 42);
        assert_eq!(program.lights()[0].light.kind, SceneLightKind::Spot);
    }

    #[test]
    fn fbx_key_infers_sprite_hints_when_available() {
        let fbx_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../docs/demos/tlapp/sphere.fbx")
            .canonicalize()
            .expect("sphere.fbx should exist");
        let src = format!(
            "tlsprite_v1\n[mesh.preview]\nsprite_id = 777\nkind = generic\nfbx = {}\n",
            fbx_path.display()
        );
        let out = compile_tlsprite(&src);
        assert!(out.program.is_some());
        assert!(out
            .diagnostics
            .iter()
            .all(|d| d.level != TlspriteDiagnosticLevel::Error));
        let program = out.program.expect("program");
        let sprite = &program.sprites()[0].sprite;
        assert!((0..=3).contains(&sprite.texture_slot));
        assert!(sprite.size[0] > 0.0 && sprite.size[1] > 0.0);
    }

    #[test]
    fn invalid_fbx_path_warns_and_uses_defaults() {
        let src = concat!(
            "tlsprite_v1\n",
            "[mesh.bad]\n",
            "sprite_id = 778\n",
            "kind = generic\n",
            "fbx = /this/path/does/not/exist/model.fbx\n",
        );
        let out = compile_tlsprite(src);
        assert!(out.program.is_some());
        assert!(out
            .diagnostics
            .iter()
            .any(|d| d.level == TlspriteDiagnosticLevel::Warning && d.message.contains("fbx")));
    }

    #[test]
    fn mesh_fbx_bindings_dedup_by_slot() {
        let src = concat!(
            "tlsprite_v1\n",
            "[a]\n",
            "sprite_id = 10\n",
            "texture_slot = 4\n",
            "fbx = docs/demos/tlapp/sphere.fbx\n",
            "[b]\n",
            "sprite_id = 11\n",
            "texture_slot = 4\n",
            "fbx = docs/demos/tlapp/other.fbx\n",
        );
        let out = compile_tlsprite(src);
        let program = out.program.expect("program");
        let bindings = program.mesh_fbx_bindings();
        assert_eq!(bindings.len(), 1);
        assert_eq!(bindings[0].0, 4);
    }

    #[test]
    fn texture_png_key_infers_external_slot_and_aspect() {
        let png_path = temp_asset_path("sprite", "png");
        let image = RgbaImage::from_pixel(64, 32, Rgba([240, 200, 180, 255]));
        image
            .save_with_format(&png_path, ImageFormat::Png)
            .expect("write png");
        let src = format!(
            "tlsprite_v1\n[ui.icon]\nsprite_id = 200\ntexture = {}\n",
            png_path.display()
        );

        let out = compile_tlsprite(&src);
        assert!(out.program.is_some());
        assert!(out
            .diagnostics
            .iter()
            .all(|d| d.level != TlspriteDiagnosticLevel::Error));
        let program = out.program.expect("program");
        let sprite = &program.sprites()[0];
        assert!(sprite.texture_source.is_some());
        assert!((64..128).contains(&sprite.sprite.texture_slot));
        assert!(sprite.sprite.size[0] > sprite.sprite.size[1]);

        let bindings = program.sprite_texture_bindings();
        assert_eq!(bindings.len(), 1);
        assert_eq!(bindings[0].0, sprite.sprite.texture_slot);
        assert_eq!(
            bindings[0].1,
            sprite.texture_source.as_deref().unwrap_or_default()
        );

        let _ = fs::remove_file(png_path);
    }

    #[test]
    fn reports_missing_required_sprite_id() {
        let src = concat!("tlsprite_v1\n", "[broken]\n", "size = 0.2, 0.1\n",);
        let out = compile_tlsprite(src);
        assert!(out.has_errors());
        assert!(out.program.is_none());
    }

    #[test]
    fn warns_when_header_is_missing() {
        let src = concat!("[ok]\n", "sprite_id = 1\n");
        let out = compile_tlsprite(src);
        assert!(out.program.is_some());
        assert!(out
            .diagnostics
            .iter()
            .any(|d| d.level == TlspriteDiagnosticLevel::Warning));
    }

    #[test]
    fn pack_roundtrip_preserves_program_shape() {
        let src = concat!(
            "tlsprite_v1\n",
            "[hud]\n",
            "sprite_id = 88\n",
            "kind = hud\n",
            "texture_slot = 2\n",
            "layer = 12\n",
            "position = 0.1, 0.2, 0.0\n",
            "size = 0.5, 0.04\n",
            "rotation_rad = 0.0\n",
            "color = 0.2, 0.9, 0.3, 0.7\n",
            "scale_axis = x\n",
            "scale_source = spawn_progress\n",
            "scale_min = 0.1\n",
            "scale_max = 0.9\n",
            "[flash]\n",
            "entry_kind = light\n",
            "light_id = 200\n",
            "light_type = point\n",
            "position = 0.0, 3.0, 0.0\n",
            "intensity = 3.5\n",
            "range = 14.0\n",
        );

        let pack = compile_tlsprite_pack(src).expect("pack compile");
        let program = load_tlsprite_pack(&pack.bytes).expect("pack decode");
        assert_eq!(pack.sprite_count, 1);
        assert_eq!(pack.light_count, 1);
        assert_eq!(program.sprites().len(), 1);
        assert_eq!(program.lights().len(), 1);
        assert_eq!(program.sprites()[0].sprite.sprite_id, 88);
        assert_eq!(program.sprites()[0].sprite.kind, SpriteKind::Hud);
    }

    #[test]
    fn decodes_legacy_v1_pack_and_infers_kind() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(TLSPRITE_PACK_MAGIC_V1);
        bytes.extend_from_slice(&7u64.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        write_len_prefixed_str(&mut bytes, "legacy");
        bytes.extend_from_slice(&5u64.to_le_bytes()); // sprite_id
        bytes.extend_from_slice(&0u16.to_le_bytes()); // texture_slot
        bytes.extend_from_slice(&120i16.to_le_bytes()); // layer (HUD-like)
        write_f32s(&mut bytes, &[0.0f32, 0.0, 0.0]); // position
        write_f32s(&mut bytes, &[0.4f32, 0.04]); // size
        bytes.extend_from_slice(&0.0f32.to_le_bytes()); // rotation
        write_f32s(&mut bytes, &[1.0f32, 1.0, 1.0, 1.0]); // color
        bytes.push(0); // no dynamic scale

        let program = load_tlsprite_pack(&bytes).expect("v1 decode");
        assert_eq!(program.sprites().len(), 1);
        assert_eq!(program.sprites()[0].sprite.kind, SpriteKind::Hud);
    }

    #[test]
    fn program_cache_hits_and_invalidation_work() {
        let source = concat!("tlsprite_v1\n", "[a]\n", "sprite_id = 3\n");
        let mut cache = TlspriteProgramCache::new();
        let path = PathBuf::from("/tmp/a.tlsprite");

        let first = cache
            .load_from_source(path.clone(), source)
            .expect("first compile");
        assert_eq!(first.source, TlspriteCacheLoadSource::CompiledSource);
        let second = cache
            .load_from_source(path.clone(), source)
            .expect("cached load");
        assert_eq!(second.source, TlspriteCacheLoadSource::CacheHit);
        assert!(cache.program_for_path(path.as_path()).is_some());

        assert!(cache.invalidate_path(path.as_path()));
        assert!(cache.program_for_path(path.as_path()).is_none());
        assert_eq!(cache.stats().path_bindings, 0);
    }

    #[test]
    fn hot_reloader_applies_and_tracks_changes() {
        let path = temp_tlsprite_path("reload");
        fs::write(&path, concat!("tlsprite_v1\n", "[s]\n", "sprite_id = 1\n",))
            .expect("write initial");

        let mut loader = TlspriteHotReloader::new(&path);
        let first = loader.reload_if_changed();
        assert!(matches!(first, TlspriteHotReloadEvent::Applied { .. }));
        assert_eq!(loader.program().map(|p| p.sprites().len()), Some(1));
        assert!(matches!(
            loader.reload_if_changed(),
            TlspriteHotReloadEvent::Unchanged
        ));

        fs::write(
            &path,
            concat!("tlsprite_v1\n", "[s2]\n", "sprite_id = 2\n",),
        )
        .expect("write changed");
        let changed = loader.reload_if_changed();
        assert!(matches!(changed, TlspriteHotReloadEvent::Applied { .. }));
        assert_eq!(
            loader
                .program()
                .and_then(|p| p.sprites().first())
                .map(|s| s.sprite.sprite_id),
            Some(2)
        );

        let _ = fs::remove_file(path);
    }

    #[test]
    fn hot_reloader_keeps_last_good_program_on_compile_error() {
        let path = temp_tlsprite_path("fallback");
        fs::write(
            &path,
            concat!("tlsprite_v1\n", "[ok]\n", "sprite_id = 42\n",),
        )
        .expect("write initial");

        let mut loader = TlspriteHotReloader::new(&path);
        let _ = loader.reload_if_changed();
        assert!(loader.program().is_some());

        fs::write(
            &path,
            concat!("tlsprite_v1\n", "[broken]\n", "size = 0.2, 0.1\n",),
        )
        .expect("write broken");
        let event = loader.reload_if_changed();
        assert!(matches!(
            event,
            TlspriteHotReloadEvent::Rejected {
                kept_last_program: true,
                ..
            }
        ));
        assert_eq!(
            loader
                .program()
                .and_then(|p| p.sprites().first())
                .map(|s| s.sprite.sprite_id),
            Some(42)
        );

        let _ = fs::remove_file(path);
    }

    #[test]
    fn watch_reloader_polling_backend_applies_updates() {
        let path = temp_tlsprite_path("watch_poll");
        fs::write(
            &path,
            concat!("tlsprite_v1\n", "[hud]\n", "sprite_id = 5\n",),
        )
        .expect("write initial");

        let mut loader = TlspriteWatchReloader::with_configs(
            &path,
            TlspriteHotReloadConfig::default(),
            TlspriteWatchConfig {
                prefer_notify_backend: false,
                poll_interval_ms: 1,
            },
        );
        assert_eq!(loader.backend(), TlspriteWatchBackend::Polling);
        assert!(matches!(
            loader.reload_if_needed(),
            TlspriteHotReloadEvent::Applied { .. }
        ));

        fs::write(
            &path,
            concat!("tlsprite_v1\n", "[hud]\n", "sprite_id = 6\n",),
        )
        .expect("write update");
        std::thread::sleep(Duration::from_millis(2));
        assert!(matches!(
            loader.reload_if_needed(),
            TlspriteHotReloadEvent::Applied { .. }
        ));
        assert_eq!(
            loader
                .program()
                .and_then(|p| p.sprites().first())
                .map(|s| s.sprite.sprite_id),
            Some(6)
        );

        let _ = fs::remove_file(path);
    }

    #[test]
    fn watch_reloader_populates_runtime_cache() {
        let path = temp_tlsprite_path("watch_cache");
        fs::write(
            &path,
            concat!("tlsprite_v1\n", "[hud]\n", "sprite_id = 11\n",),
        )
        .expect("write initial");

        let mut loader = TlspriteWatchReloader::with_configs(
            &path,
            TlspriteHotReloadConfig::default(),
            TlspriteWatchConfig {
                prefer_notify_backend: false,
                poll_interval_ms: 1,
            },
        );
        let mut cache = TlspriteProgramCache::new();
        let event = loader.reload_into_cache(&mut cache);
        assert!(matches!(event, TlspriteHotReloadEvent::Applied { .. }));
        assert!(cache.program_for_path(loader.path()).is_some());

        let _ = fs::remove_file(path);
    }

    fn temp_tlsprite_path(tag: &str) -> PathBuf {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!(
            "tileline_tlsprite_{tag}_{}_{}.tlsprite",
            std::process::id(),
            ts
        ))
    }

    fn temp_asset_path(tag: &str, extension: &str) -> PathBuf {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        std::env::temp_dir().join(format!(
            "tileline_tlsprite_{tag}_{}_{}.{}",
            std::process::id(),
            ts,
            extension
        ))
    }
}
