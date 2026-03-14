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
//! texture_slot = 0
//! layer = 100
//! position = -0.86, 0.90, 0.0
//! size = 0.40, 0.035
//! rotation_rad = 0.0
//! color = 0.10, 0.84, 0.62, 0.92
//! scale_axis = x
//! scale_source = spawn_progress
//! scale_min = 0.02
//! scale_max = 1.0
//! ```

use std::{
    collections::{hash_map::DefaultHasher, HashMap},
    fs,
    hash::{Hash, Hasher},
    path::{Path, PathBuf},
    sync::mpsc::{self, Receiver, TryRecvError},
    time::{Duration, Instant},
};

use notify::{
    Config as NotifyConfig, Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher,
};

use crate::scene::SpriteInstance;

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
    dynamic_scale: Option<TlspriteDynamicScale>,
}

/// Compiled `.tlsprite` sprite set.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TlspriteProgram {
    sprites: Vec<TlspriteSpriteDef>,
}

impl TlspriteProgram {
    pub fn sprites(&self) -> &[TlspriteSpriteDef] {
        &self.sprites
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

const TLSPRITE_PACK_MAGIC: &[u8; 8] = b"TLSPK001";

/// In-memory binary pack produced from a `.tlsprite` source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TlspritePack {
    pub bytes: Vec<u8>,
    pub source_hash: u64,
    pub sprite_count: usize,
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
    let mut diagnostics = Vec::new();
    let mut sprites = Vec::new();
    let mut header_checked = false;
    let mut pending: Option<PendingSprite> = None;

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
                if let Some(def) = curr.finish(&mut diagnostics) {
                    sprites.push(def);
                }
            }
            pending = Some(PendingSprite::new(section.to_string(), line_no));
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
        if let Some(def) = curr.finish(&mut diagnostics) {
            sprites.push(def);
        }
    }

    if sprites.is_empty() {
        diagnostics.push(TlspriteDiagnostic {
            level: TlspriteDiagnosticLevel::Error,
            line: 0,
            message: "no sprite sections were produced".to_string(),
        });
    }

    let has_errors = diagnostics
        .iter()
        .any(|d| d.level == TlspriteDiagnosticLevel::Error);
    TlspriteCompileOutcome {
        program: (!has_errors).then_some(TlspriteProgram { sprites }),
        diagnostics,
    }
}

fn hash_bytes(bytes: &[u8]) -> u64 {
    let mut hasher = DefaultHasher::new();
    bytes.hash(&mut hasher);
    hasher.finish()
}

fn encode_tlsprite_pack(program: &TlspriteProgram, source_hash: u64) -> Vec<u8> {
    let mut out = Vec::with_capacity(64 + program.sprites().len() * 96);
    out.extend_from_slice(TLSPRITE_PACK_MAGIC);
    out.extend_from_slice(&source_hash.to_le_bytes());
    out.extend_from_slice(&(program.sprites().len() as u32).to_le_bytes());
    for def in program.sprites() {
        write_len_prefixed_str(&mut out, &def.name);
        out.extend_from_slice(&def.sprite.sprite_id.to_le_bytes());
        out.extend_from_slice(&def.sprite.texture_slot.to_le_bytes());
        out.extend_from_slice(&def.sprite.layer.to_le_bytes());
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
    out
}

fn decode_tlsprite_pack(bytes: &[u8]) -> Result<DecodedTlspritePack, String> {
    let mut rd = ByteReader::new(bytes);
    rd.expect_magic(TLSPRITE_PACK_MAGIC)?;
    let source_hash = rd.read_u64()?;
    let sprite_count = rd.read_u32()? as usize;
    let mut sprites = Vec::with_capacity(sprite_count);
    for _ in 0..sprite_count {
        let name = rd.read_string()?;
        let sprite_id = rd.read_u64()?;
        let texture_slot = rd.read_u16()?;
        let layer = rd.read_i16()?;
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
                position,
                size,
                rotation_rad,
                color_rgba,
                texture_slot,
                layer,
            },
            dynamic_scale,
        });
    }
    if rd.remaining() != 0 {
        return Err("packed data contains trailing bytes".to_string());
    }

    Ok(DecodedTlspritePack {
        program: TlspriteProgram { sprites },
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

    fn expect_magic(&mut self, magic: &[u8]) -> Result<(), String> {
        let got = self.read_bytes(magic.len())?;
        if got != magic {
            return Err("invalid tlsprite pack magic/version".to_string());
        }
        Ok(())
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

#[derive(Debug, Clone)]
struct PendingSprite {
    name: String,
    line_started: usize,
    sprite_id: Option<u64>,
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
}

impl PendingSprite {
    fn new(name: String, line_started: usize) -> Self {
        Self {
            name,
            line_started,
            sprite_id: None,
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
            "sprite_id" => {
                self.sprite_id = parse_scalar::<u64>(value, line, key, diagnostics);
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
            _ => diagnostics.push(TlspriteDiagnostic {
                level: TlspriteDiagnosticLevel::Warning,
                line,
                message: format!("unknown key '{key}' ignored"),
            }),
        }
    }

    fn finish(self, diagnostics: &mut Vec<TlspriteDiagnostic>) -> Option<TlspriteSpriteDef> {
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

        let mut color = self.color.unwrap_or([1.0, 1.0, 1.0, 1.0]);
        for c in &mut color {
            *c = c.clamp(0.0, 1.0);
        }
        let mut size = self.size.unwrap_or([0.1, 0.1]);
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
                position: self.position.unwrap_or([0.0, 0.0, 0.0]),
                size,
                rotation_rad: self.rotation_rad.unwrap_or(0.0),
                color_rgba: color,
                texture_slot: self.texture_slot.unwrap_or(0),
                layer: self.layer.unwrap_or(0),
            },
            dynamic_scale,
        })
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
        );

        let pack = compile_tlsprite_pack(src).expect("pack compile");
        let program = load_tlsprite_pack(&pack.bytes).expect("pack decode");
        assert_eq!(pack.sprite_count, 1);
        assert_eq!(program.sprites().len(), 1);
        assert_eq!(program.sprites()[0].sprite.sprite_id, 88);
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
}
