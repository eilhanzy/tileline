//! MPS-style compressed spill pool for MGS memory pressure recovery.
//!
//! This module provides an in-process, RAM-backed "zram-like" layer that
//! compresses cold payloads instead of dropping them immediately. It is not an
//! OS swap implementation; it is a deterministic runtime utility designed for
//! engine-managed buffers.
//!
//! Design goals:
//! - avoid blocking locks on the hot path (single-owner pool)
//! - use parallel chunk compression/decompression (scoped threads) for MPS-like scaling
//! - expose deterministic telemetry for fallback tuning and diagnostics

use std::collections::{HashMap, VecDeque};
use std::thread;

use miniz_oxide::deflate::compress_to_vec_zlib;
use miniz_oxide::inflate::decompress_to_vec_zlib;

use crate::hardware::{MobileGpuProfile, TbdrArchitecture};

/// Runtime configuration for the compressed spill pool.
#[derive(Debug, Clone, Copy)]
pub struct MpsZramConfig {
    /// Enables or disables the spill pool.
    pub enabled: bool,
    /// Chunk size used for parallel compression/decompression.
    pub chunk_bytes: usize,
    /// Soft compressed-memory target. The pool tries to stay below this size.
    pub target_compressed_bytes: usize,
    /// Hard compressed-memory limit; evictions are forced above this threshold.
    pub hard_compressed_bytes: usize,
    /// Maximum number of pages retained in the pool.
    pub max_pages: usize,
    /// Small payloads below this size are not compressed.
    pub min_payload_bytes: usize,
    /// DEFLATE level for `miniz_oxide` (0-10).
    pub deflate_level: u8,
    /// Number of logical shards used for parallel chunk operations.
    pub compression_shards: usize,
    /// Maximum bytes that cold (uncompressed) pages may occupy in system RAM.
    /// Pages demoted from the hot zram pool land here before being dropped.
    /// Set to `0` to disable the cold tier entirely.
    pub max_cold_bytes: usize,
}

impl Default for MpsZramConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            chunk_bytes: 32 * 1024,
            target_compressed_bytes: 48 * 1024 * 1024,
            hard_compressed_bytes: 72 * 1024 * 1024,
            max_pages: 256,
            min_payload_bytes: 2 * 1024,
            deflate_level: 6,
            compression_shards: 4,
            // 128 MiB cold tier — stays within typical system RAM headroom without
            // risking OOM on mobile/UMA devices.
            max_cold_bytes: 128 * 1024 * 1024,
        }
    }
}

impl MpsZramConfig {
    /// Returns a disabled config.
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }

    /// Derives a profile-aware default configuration.
    pub fn for_profile(
        profile: &MobileGpuProfile,
        logical_threads: usize,
        uma_shared_memory: bool,
    ) -> Self {
        let mut cfg = Self::default();
        let threads = logical_threads.max(1);
        cfg.compression_shards = threads.clamp(1, 8);

        // Mobile/UMA targets need tighter memory budgets to avoid whole-system
        // contention and chopping under thermal/load pressure.
        if uma_shared_memory || profile.is_mobile_tbdr() {
            cfg.chunk_bytes = 24 * 1024;
            cfg.target_compressed_bytes = 16 * 1024 * 1024;
            cfg.hard_compressed_bytes = 24 * 1024 * 1024;
            cfg.max_pages = 160;
            cfg.deflate_level = 5;
        }

        if matches!(profile.architecture, TbdrArchitecture::AppleTbdr) {
            // Apple UMA responds better to lower latency compression settings.
            cfg.chunk_bytes = 16 * 1024;
            cfg.target_compressed_bytes = cfg.target_compressed_bytes.min(12 * 1024 * 1024);
            cfg.hard_compressed_bytes = cfg.hard_compressed_bytes.min(18 * 1024 * 1024);
            cfg.deflate_level = 4;
            // Tighter cold tier on Apple UMA: shared CPU/GPU memory means cold pages
            // compete with GPU resources. 64 MiB is a reasonable guard.
            cfg.max_cold_bytes = 64 * 1024 * 1024;
        }

        if uma_shared_memory || profile.is_mobile_tbdr() {
            // Cold tier is already reduced for UMA above; non-Apple UMA also caps at 64 MiB.
            cfg.max_cold_bytes = cfg.max_cold_bytes.min(64 * 1024 * 1024);
        }

        cfg
    }
}

/// Spill result metadata for diagnostics.
#[derive(Debug, Clone, Copy, Default)]
pub struct MpsZramSpillOutcome {
    /// True when payload was compressed and inserted.
    pub spilled: bool,
    /// Compressed payload size written to the pool.
    pub compressed_bytes: usize,
    /// Number of entries evicted after insertion.
    pub evicted_pages: usize,
}

/// Pool-level telemetry.
#[derive(Debug, Clone, Copy, Default)]
pub struct MpsZramStats {
    /// Number of pages currently in the hot (compressed) zram tier.
    pub stored_pages: usize,
    pub uncompressed_bytes: usize,
    pub compressed_bytes: usize,
    pub spill_events: u64,
    pub restore_events: u64,
    pub evict_events: u64,
    pub failed_restores: u64,
    /// Number of pages currently in the cold (uncompressed system RAM) tier.
    pub cold_stored_pages: usize,
    /// Total uncompressed bytes held by the cold tier.
    pub cold_bytes: usize,
    /// Pages demoted from hot zram to the cold system RAM tier.
    pub cold_spill_events: u64,
    /// Pages promoted from the cold tier back to the caller.
    pub cold_restore_events: u64,
    /// Pages dropped from the cold tier when its budget was exceeded.
    pub cold_evict_events: u64,
}

impl MpsZramStats {
    /// Compression ratio (compressed / uncompressed).
    pub fn compression_ratio(&self) -> f64 {
        if self.uncompressed_bytes == 0 {
            1.0
        } else {
            self.compressed_bytes as f64 / self.uncompressed_bytes as f64
        }
    }

    /// Approximate memory savings in bytes.
    pub fn saved_bytes(&self) -> isize {
        self.uncompressed_bytes as isize - self.compressed_bytes as isize
    }
}

#[derive(Debug, Clone)]
struct CompressedPage {
    generation: u64,
    uncompressed_len: usize,
    compressed_len: usize,
    chunks: Vec<Vec<u8>>,
}

/// Error during compressed page restore.
#[derive(Debug, Clone)]
pub enum MpsZramError {
    InflateFailed,
}

impl std::fmt::Display for MpsZramError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InflateFailed => write!(f, "failed to inflate compressed zram page"),
        }
    }
}

impl std::error::Error for MpsZramError {}

/// Deterministic two-tier compressed spill pool.
///
/// **Hot tier** — pages are DEFLATE-compressed and kept in process memory.
/// When the hot tier exceeds its budget, pages are demoted to the cold tier
/// instead of being dropped immediately.
///
/// **Cold tier** — pages are stored uncompressed in system RAM up to
/// `max_cold_bytes`. When the cold tier is also over budget, the oldest cold
/// pages are dropped (evicted). This prevents OOM by bounding total usage.
///
/// The pool is designed to be owned by one runtime thread. Internally, it uses
/// parallel chunk processing for compression/decompression but does not require
/// `Arc<Mutex<...>>` for coordination.
#[derive(Debug)]
pub struct MpsZramSpillPool {
    config: MpsZramConfig,
    /// Hot tier: DEFLATE-compressed pages.
    pages: HashMap<u64, CompressedPage>,
    lru: VecDeque<(u64, u64)>,
    next_generation: u64,
    /// Cold tier: uncompressed pages in system RAM, ordered oldest-first.
    cold_pages: HashMap<u64, Vec<u8>>,
    cold_lru: VecDeque<u64>,
    stats: MpsZramStats,
}

impl MpsZramSpillPool {
    /// Creates a new spill pool.
    pub fn new(config: MpsZramConfig) -> Self {
        Self {
            config,
            pages: HashMap::with_capacity(config.max_pages.max(16)),
            lru: VecDeque::with_capacity(config.max_pages.max(16)),
            next_generation: 1,
            cold_pages: HashMap::new(),
            cold_lru: VecDeque::new(),
            stats: MpsZramStats::default(),
        }
    }

    /// Returns the active configuration.
    pub fn config(&self) -> MpsZramConfig {
        self.config
    }

    /// Returns current stats.
    pub fn stats(&self) -> MpsZramStats {
        self.stats
    }

    /// True when pool is enabled.
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Inserts or updates a compressed page.
    pub fn spill(&mut self, page_id: u64, payload: &[u8]) -> MpsZramSpillOutcome {
        if !self.config.enabled || payload.len() < self.config.min_payload_bytes {
            return MpsZramSpillOutcome::default();
        }

        let chunks = self.compress_payload(payload);
        let compressed_len = chunks.iter().map(Vec::len).sum::<usize>();
        if compressed_len == 0 {
            return MpsZramSpillOutcome::default();
        }

        self.remove_page(page_id);

        let generation = self.next_generation;
        self.next_generation = self.next_generation.saturating_add(1);
        let page = CompressedPage {
            generation,
            uncompressed_len: payload.len(),
            compressed_len,
            chunks,
        };

        self.stats.spill_events = self.stats.spill_events.saturating_add(1);
        self.stats.uncompressed_bytes = self.stats.uncompressed_bytes.saturating_add(payload.len());
        self.stats.compressed_bytes = self.stats.compressed_bytes.saturating_add(compressed_len);
        self.pages.insert(page_id, page);
        self.lru.push_back((page_id, generation));
        self.stats.stored_pages = self.pages.len();

        let evicted_pages = self.enforce_budget();
        MpsZramSpillOutcome {
            spilled: true,
            compressed_bytes: compressed_len,
            evicted_pages,
        }
    }

    /// Restores a page. Checks the hot (compressed) tier first, then falls back
    /// to the cold (system RAM) tier. Returns `None` if the page is not found in
    /// either tier.
    pub fn restore(&mut self, page_id: u64) -> Result<Option<Vec<u8>>, MpsZramError> {
        // Hot tier lookup.
        if let Some(page) = self.pages.get(&page_id).cloned() {
            let inflated_chunks = self.decompress_chunks(&page.chunks)?;
            let mut restored = Vec::with_capacity(page.uncompressed_len);
            for chunk in inflated_chunks {
                restored.extend_from_slice(&chunk);
            }
            restored.truncate(page.uncompressed_len);
            self.stats.restore_events = self.stats.restore_events.saturating_add(1);
            self.lru.push_back((page_id, page.generation));
            return Ok(Some(restored));
        }

        // Cold tier fallback.
        if let Some(cold) = self.cold_pages.remove(&page_id) {
            self.stats.cold_bytes = self.stats.cold_bytes.saturating_sub(cold.len());
            self.stats.cold_stored_pages = self.cold_pages.len();
            self.stats.cold_restore_events = self.stats.cold_restore_events.saturating_add(1);
            return Ok(Some(cold));
        }

        Ok(None)
    }

    /// Clears all pages from both hot and cold tiers.
    pub fn clear(&mut self) {
        self.pages.clear();
        self.lru.clear();
        self.cold_pages.clear();
        self.cold_lru.clear();
        self.stats.stored_pages = 0;
        self.stats.uncompressed_bytes = 0;
        self.stats.compressed_bytes = 0;
        self.stats.cold_stored_pages = 0;
        self.stats.cold_bytes = 0;
    }

    fn compress_payload(&self, payload: &[u8]) -> Vec<Vec<u8>> {
        let chunk_bytes = self.config.chunk_bytes.max(4 * 1024);
        let chunks: Vec<&[u8]> = payload.chunks(chunk_bytes).collect();
        let level = self.config.deflate_level.min(10);
        let workers = self.compression_worker_count(chunks.len());
        parallel_chunk_map(&chunks, workers, 2, |chunk| {
            compress_to_vec_zlib(chunk, level)
        })
    }

    fn decompress_chunks(&mut self, chunks: &[Vec<u8>]) -> Result<Vec<Vec<u8>>, MpsZramError> {
        let workers = self.compression_worker_count(chunks.len());
        let decoded = parallel_chunk_map(&chunks, workers, 2, |chunk| {
            decompress_to_vec_zlib(chunk).map_err(|_| MpsZramError::InflateFailed)
        });
        let result = decoded.into_iter().collect::<Result<Vec<_>, _>>();

        if result.is_err() {
            self.stats.failed_restores = self.stats.failed_restores.saturating_add(1);
        }

        result
    }

    #[inline]
    fn compression_worker_count(&self, chunk_count: usize) -> usize {
        let available = thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .max(1);
        self.config
            .compression_shards
            .max(1)
            .min(available)
            .min(chunk_count.max(1))
    }

    fn enforce_budget(&mut self) -> usize {
        let mut evicted = 0usize;
        while self.hot_over_budget() {
            let Some((key, generation)) = self.lru.pop_front() else {
                break;
            };
            let stale = self
                .pages
                .get(&key)
                .map(|page| page.generation != generation)
                .unwrap_or(true);
            if stale {
                continue;
            }
            // Try to restore the compressed page so we can demote it to the cold tier.
            // If cold tier is full or disabled, just drop the page.
            let demoted = if self.config.max_cold_bytes > 0 {
                if let Some(page) = self.pages.get(&key).cloned() {
                    if let Ok(chunks) = self.decompress_chunks(&page.chunks) {
                        let mut raw: Vec<u8> = Vec::with_capacity(page.uncompressed_len);
                        for chunk in chunks {
                            raw.extend_from_slice(&chunk);
                        }
                        raw.truncate(page.uncompressed_len);
                        self.try_demote_to_cold(key, raw)
                    } else {
                        false
                    }
                } else {
                    false
                }
            } else {
                false
            };

            if self.remove_page(key) {
                if demoted {
                    self.stats.cold_spill_events = self.stats.cold_spill_events.saturating_add(1);
                } else {
                    evicted = evicted.saturating_add(1);
                    self.stats.evict_events = self.stats.evict_events.saturating_add(1);
                }
            }
        }
        evicted
    }

    /// Attempts to store a raw page in the cold tier. Returns `true` on success.
    /// Enforces the cold budget by evicting the oldest cold pages first.
    fn try_demote_to_cold(&mut self, key: u64, raw: Vec<u8>) -> bool {
        if self.config.max_cold_bytes == 0 {
            return false;
        }
        // Make room in the cold tier if needed.
        while self.stats.cold_bytes.saturating_add(raw.len()) > self.config.max_cold_bytes {
            let Some(oldest_key) = self.cold_lru.pop_front() else {
                break;
            };
            if let Some(dropped) = self.cold_pages.remove(&oldest_key) {
                self.stats.cold_bytes = self.stats.cold_bytes.saturating_sub(dropped.len());
                self.stats.cold_stored_pages = self.cold_pages.len();
                self.stats.cold_evict_events = self.stats.cold_evict_events.saturating_add(1);
            }
        }
        // If still no room (single page larger than budget), bail.
        if raw.len() > self.config.max_cold_bytes {
            return false;
        }
        self.stats.cold_bytes = self.stats.cold_bytes.saturating_add(raw.len());
        self.cold_pages.insert(key, raw);
        self.cold_lru.push_back(key);
        self.stats.cold_stored_pages = self.cold_pages.len();
        true
    }

    fn hot_over_budget(&self) -> bool {
        self.stats.compressed_bytes > self.config.hard_compressed_bytes
            || self.stats.compressed_bytes > self.config.target_compressed_bytes
            || self.pages.len() > self.config.max_pages
    }

    fn remove_page(&mut self, key: u64) -> bool {
        let Some(page) = self.pages.remove(&key) else {
            return false;
        };
        self.stats.uncompressed_bytes = self
            .stats
            .uncompressed_bytes
            .saturating_sub(page.uncompressed_len);
        self.stats.compressed_bytes = self
            .stats
            .compressed_bytes
            .saturating_sub(page.compressed_len);
        self.stats.stored_pages = self.pages.len();
        true
    }
}

fn parallel_chunk_map<T, R, F>(
    items: &[T],
    workers: usize,
    min_items_per_worker: usize,
    map: F,
) -> Vec<R>
where
    T: Sync,
    R: Send,
    F: Fn(&T) -> R + Sync,
{
    let worker_count = workers.max(1);
    if worker_count <= 1 || items.len() <= 1 || items.len() < min_items_per_worker.max(1) * 2 {
        return items.iter().map(map).collect();
    }

    let chunk_len = items
        .len()
        .div_ceil(worker_count)
        .max(min_items_per_worker.max(1));
    let mut shard_outputs: Vec<Vec<R>> = Vec::new();
    thread::scope(|scope| {
        let mut handles = Vec::new();
        for shard in items.chunks(chunk_len) {
            let map_ref = &map;
            handles.push(scope.spawn(move || {
                let mut local = Vec::with_capacity(shard.len());
                for item in shard {
                    local.push(map_ref(item));
                }
                local
            }));
        }
        shard_outputs.reserve(handles.len());
        for handle in handles {
            shard_outputs.push(handle.join().expect("MGS zram worker panicked"));
        }
    });

    let total = shard_outputs.iter().map(Vec::len).sum::<usize>();
    let mut out = Vec::with_capacity(total);
    for mut shard in shard_outputs {
        out.append(&mut shard);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spill_and_restore_roundtrip() {
        let mut pool = MpsZramSpillPool::new(MpsZramConfig {
            chunk_bytes: 8 * 1024,
            min_payload_bytes: 1,
            ..MpsZramConfig::default()
        });

        let payload = vec![42u8; 64 * 1024];
        let outcome = pool.spill(7, &payload);
        assert!(outcome.spilled);

        let restored = pool
            .restore(7)
            .expect("restore should succeed")
            .expect("page must exist");
        assert_eq!(restored, payload);

        let stats = pool.stats();
        assert_eq!(stats.stored_pages, 1);
        assert!(stats.compressed_bytes > 0);
    }

    #[test]
    fn enforce_budget_demotes_to_cold_then_evicts() {
        // Hot budget: tiny (2 KiB), cold budget: also tiny (32 KiB).
        // With 4× 16 KiB pages, the hot tier overflows and pages demote to cold.
        // When cold also fills up, the oldest cold pages are evicted (dropped).
        let mut pool = MpsZramSpillPool::new(MpsZramConfig {
            chunk_bytes: 4 * 1024,
            min_payload_bytes: 1,
            target_compressed_bytes: 2 * 1024,
            hard_compressed_bytes: 2 * 1024,
            max_pages: 2,
            max_cold_bytes: 32 * 1024,
            ..MpsZramConfig::default()
        });

        for i in 0..4u64 {
            let payload = vec![i as u8; 16 * 1024];
            pool.spill(i, &payload);
        }

        let stats = pool.stats();
        // Hot tier must respect max_pages.
        assert!(stats.stored_pages <= 2);
        // Pages were demoted to cold or evicted — at least one cold spill or hard eviction.
        assert!(
            stats.cold_spill_events >= 1 || stats.evict_events >= 1,
            "expected at least one cold demote or hard eviction"
        );
        // Cold tier must not exceed its budget.
        assert!(stats.cold_bytes <= 32 * 1024);
    }

    #[test]
    fn cold_tier_disabled_causes_hard_eviction() {
        // With max_cold_bytes = 0 the cold tier is off; overflow pages are dropped.
        let mut pool = MpsZramSpillPool::new(MpsZramConfig {
            chunk_bytes: 4 * 1024,
            min_payload_bytes: 1,
            target_compressed_bytes: 2 * 1024,
            hard_compressed_bytes: 2 * 1024,
            max_pages: 2,
            max_cold_bytes: 0,
            ..MpsZramConfig::default()
        });

        for i in 0..4u64 {
            let payload = vec![i as u8; 16 * 1024];
            pool.spill(i, &payload);
        }

        let stats = pool.stats();
        assert!(stats.stored_pages <= 2);
        assert!(stats.evict_events >= 1);
        assert_eq!(stats.cold_stored_pages, 0);
    }

    #[test]
    fn profile_config_is_more_conservative_on_uma() {
        let profile = MobileGpuProfile::detect("Mali-G610");
        let cfg_uma = MpsZramConfig::for_profile(&profile, 8, true);
        let cfg_discrete = MpsZramConfig::for_profile(&profile, 8, false);
        assert!(cfg_uma.target_compressed_bytes <= cfg_discrete.target_compressed_bytes);
    }
}
