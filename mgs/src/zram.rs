//! MPS-style compressed spill pool for MGS memory pressure recovery.
//!
//! This module provides an in-process, RAM-backed "zram-like" layer that
//! compresses cold payloads instead of dropping them immediately. It is not an
//! OS swap implementation; it is a deterministic runtime utility designed for
//! engine-managed buffers.
//!
//! Design goals:
//! - avoid blocking locks on the hot path (single-owner pool)
//! - use parallel chunk compression/decompression (Rayon) for MPS-like scaling
//! - expose deterministic telemetry for fallback tuning and diagnostics

use std::collections::{HashMap, VecDeque};

use miniz_oxide::deflate::compress_to_vec_zlib;
use miniz_oxide::inflate::decompress_to_vec_zlib;
use rayon::prelude::*;

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
    pub stored_pages: usize,
    pub uncompressed_bytes: usize,
    pub compressed_bytes: usize,
    pub spill_events: u64,
    pub restore_events: u64,
    pub evict_events: u64,
    pub failed_restores: u64,
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

/// Deterministic compressed spill pool.
///
/// The pool is designed to be owned by one runtime thread. Internally, it uses
/// parallel chunk processing for compression/decompression but does not require
/// `Arc<Mutex<...>>` for coordination.
#[derive(Debug)]
pub struct MpsZramSpillPool {
    config: MpsZramConfig,
    pages: HashMap<u64, CompressedPage>,
    lru: VecDeque<(u64, u64)>,
    next_generation: u64,
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

    /// Restores a compressed page.
    pub fn restore(&mut self, page_id: u64) -> Result<Option<Vec<u8>>, MpsZramError> {
        let Some(page) = self.pages.get(&page_id).cloned() else {
            return Ok(None);
        };

        let inflated_chunks = self.decompress_chunks(&page.chunks)?;
        let mut restored = Vec::with_capacity(page.uncompressed_len);
        for chunk in inflated_chunks {
            restored.extend_from_slice(&chunk);
        }
        restored.truncate(page.uncompressed_len);

        self.stats.restore_events = self.stats.restore_events.saturating_add(1);
        self.lru.push_back((page_id, page.generation));
        Ok(Some(restored))
    }

    /// Clears all compressed pages.
    pub fn clear(&mut self) {
        self.pages.clear();
        self.lru.clear();
        self.stats.stored_pages = 0;
        self.stats.uncompressed_bytes = 0;
        self.stats.compressed_bytes = 0;
    }

    fn compress_payload(&self, payload: &[u8]) -> Vec<Vec<u8>> {
        let chunk_bytes = self.config.chunk_bytes.max(4 * 1024);
        let chunks: Vec<&[u8]> = payload.chunks(chunk_bytes).collect();
        let level = self.config.deflate_level.min(10);

        if self.config.compression_shards > 1 && chunks.len() > 1 {
            chunks
                .into_par_iter()
                .map(|chunk| compress_to_vec_zlib(chunk, level))
                .collect()
        } else {
            chunks
                .into_iter()
                .map(|chunk| compress_to_vec_zlib(chunk, level))
                .collect()
        }
    }

    fn decompress_chunks(&mut self, chunks: &[Vec<u8>]) -> Result<Vec<Vec<u8>>, MpsZramError> {
        let result: Result<Vec<Vec<u8>>, MpsZramError> = if self.config.compression_shards > 1
            && chunks.len() > 1
        {
            chunks
                .par_iter()
                .map(|chunk| decompress_to_vec_zlib(chunk).map_err(|_| MpsZramError::InflateFailed))
                .collect()
        } else {
            chunks
                .iter()
                .map(|chunk| decompress_to_vec_zlib(chunk).map_err(|_| MpsZramError::InflateFailed))
                .collect()
        };

        if result.is_err() {
            self.stats.failed_restores = self.stats.failed_restores.saturating_add(1);
        }

        result
    }

    fn enforce_budget(&mut self) -> usize {
        let mut evicted = 0usize;
        while self.over_budget() {
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
            if self.remove_page(key) {
                evicted = evicted.saturating_add(1);
                self.stats.evict_events = self.stats.evict_events.saturating_add(1);
            }
        }
        evicted
    }

    fn over_budget(&self) -> bool {
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
    fn enforce_budget_evicts_oldest_pages() {
        let mut pool = MpsZramSpillPool::new(MpsZramConfig {
            chunk_bytes: 4 * 1024,
            min_payload_bytes: 1,
            target_compressed_bytes: 2 * 1024,
            hard_compressed_bytes: 2 * 1024,
            max_pages: 2,
            ..MpsZramConfig::default()
        });

        for i in 0..4u64 {
            let payload = vec![i as u8; 16 * 1024];
            pool.spill(i, &payload);
        }

        let stats = pool.stats();
        assert!(stats.stored_pages <= 2);
        assert!(stats.evict_events >= 1);
    }

    #[test]
    fn profile_config_is_more_conservative_on_uma() {
        let profile = MobileGpuProfile::detect("Mali-G610");
        let cfg_uma = MpsZramConfig::for_profile(&profile, 8, true);
        let cfg_discrete = MpsZramConfig::for_profile(&profile, 8, false);
        assert!(cfg_uma.target_compressed_bytes <= cfg_discrete.target_compressed_bytes);
    }
}
