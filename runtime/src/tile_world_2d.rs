//! Chunked 2D tile world foundation for side-view sandbox scenes.
//!
//! This module provides a deterministic, mutation-friendly storage model with local chunk updates
//! so runtime/editor paths can support dig/place without rebuilding whole maps.

use std::collections::{HashMap, HashSet};

pub const TILE_ID_EMPTY: u16 = 0;

/// Configuration for canonical chunked tile world storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileWorld2dConfig {
    /// Number of tiles per chunk side.
    pub chunk_size: u16,
    /// Default visible-tile emission budget per frame.
    pub max_visible_tiles: usize,
}

impl Default for TileWorld2dConfig {
    fn default() -> Self {
        Self {
            chunk_size: 32,
            max_visible_tiles: 12_000,
        }
    }
}

impl TileWorld2dConfig {
    pub fn sanitized(self) -> Self {
        Self {
            chunk_size: self.chunk_size.clamp(8, 128),
            max_visible_tiles: self.max_visible_tiles.clamp(256, 200_000),
        }
    }
}

/// Integer tile-space coordinate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileCoord2d {
    pub x: i32,
    pub y: i32,
}

impl TileCoord2d {
    pub const fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }
}

/// Chunk-space coordinate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileChunkCoord2d {
    pub x: i32,
    pub y: i32,
}

impl TileChunkCoord2d {
    pub const fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }
}

/// Side-view camera/view rectangle used for visible tile collection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TileView2d {
    pub center: [f32; 2],
    pub half_size: [f32; 2],
    pub zoom: f32,
}

impl Default for TileView2d {
    fn default() -> Self {
        Self {
            center: [0.0, 0.0],
            half_size: [12.0, 8.0],
            zoom: 1.0,
        }
    }
}

/// One visible non-empty tile emitted for frame rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileVisibleInstance2d {
    pub coord: TileCoord2d,
    pub tile_id: u16,
    pub layer: i16,
}

/// Per-frame tile world telemetry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TileWorldFrameTelemetry {
    pub loaded_chunks: usize,
    pub dirty_chunks: usize,
    pub visible_chunks: usize,
    pub visible_tiles: usize,
    pub emitted_tiles: usize,
    pub culled_tiles: usize,
    pub world_revision: u64,
    pub mutation_count: u64,
}

/// Visible tile collection result.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TileVisibleSet2d {
    pub tiles: Vec<TileVisibleInstance2d>,
    pub telemetry: TileWorldFrameTelemetry,
}

/// One local tile mutation request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileMutation2d {
    pub coord: TileCoord2d,
    pub tile_id: u16,
}

impl TileMutation2d {
    pub const fn place(coord: TileCoord2d, tile_id: u16) -> Self {
        Self { coord, tile_id }
    }

    pub const fn dig(coord: TileCoord2d) -> Self {
        Self {
            coord,
            tile_id: TILE_ID_EMPTY,
        }
    }
}

#[derive(Debug, Clone)]
struct TileChunk2d {
    tiles: Vec<u16>,
    non_empty_count: usize,
    revision: u64,
}

impl TileChunk2d {
    fn new(chunk_size: usize) -> Self {
        Self {
            tiles: vec![TILE_ID_EMPTY; chunk_size * chunk_size],
            non_empty_count: 0,
            revision: 0,
        }
    }
}

/// Canonical chunked tile world for side-view runtime scenes.
#[derive(Debug, Clone)]
pub struct ChunkedTileWorld2d {
    config: TileWorld2dConfig,
    chunks: HashMap<TileChunkCoord2d, TileChunk2d>,
    dirty_chunks: HashSet<TileChunkCoord2d>,
    world_revision: u64,
    mutation_count: u64,
    last_frame_telemetry: TileWorldFrameTelemetry,
}

impl ChunkedTileWorld2d {
    pub fn new(config: TileWorld2dConfig) -> Self {
        let config = config.sanitized();
        Self {
            config,
            chunks: HashMap::new(),
            dirty_chunks: HashSet::new(),
            world_revision: 0,
            mutation_count: 0,
            last_frame_telemetry: TileWorldFrameTelemetry::default(),
        }
    }

    pub fn config(&self) -> TileWorld2dConfig {
        self.config
    }

    pub fn loaded_chunk_count(&self) -> usize {
        self.chunks.len()
    }

    pub fn dirty_chunk_count(&self) -> usize {
        self.dirty_chunks.len()
    }

    pub fn world_revision(&self) -> u64 {
        self.world_revision
    }

    pub fn mutation_count(&self) -> u64 {
        self.mutation_count
    }

    pub fn clear(&mut self) {
        self.chunks.clear();
        self.dirty_chunks.clear();
        self.bump_world_revision();
    }

    pub fn tile(&self, coord: TileCoord2d) -> u16 {
        let (chunk_coord, lx, ly) = split_tile_coord(coord, self.config.chunk_size as i32);
        let Some(chunk) = self.chunks.get(&chunk_coord) else {
            return TILE_ID_EMPTY;
        };
        chunk.tiles[tile_index(self.config.chunk_size as usize, lx, ly)]
    }

    /// Set one tile. Returns `true` when the tile value changed.
    pub fn set_tile(&mut self, coord: TileCoord2d, tile_id: u16) -> bool {
        let chunk_size_i32 = self.config.chunk_size as i32;
        let (chunk_coord, lx, ly) = split_tile_coord(coord, chunk_size_i32);
        let chunk_size = self.config.chunk_size as usize;
        if tile_id == TILE_ID_EMPTY {
            let mut remove_chunk = false;
            let changed = if let Some(chunk) = self.chunks.get_mut(&chunk_coord) {
                let idx = tile_index(chunk_size, lx, ly);
                let previous = chunk.tiles[idx];
                if previous == TILE_ID_EMPTY {
                    false
                } else {
                    chunk.tiles[idx] = TILE_ID_EMPTY;
                    chunk.non_empty_count = chunk.non_empty_count.saturating_sub(1);
                    remove_chunk = chunk.non_empty_count == 0;
                    true
                }
            } else {
                false
            };
            if changed {
                self.bump_world_revision();
                if let Some(chunk) = self.chunks.get_mut(&chunk_coord) {
                    chunk.revision = self.world_revision;
                }
                self.dirty_chunks.insert(chunk_coord);
            }
            if remove_chunk {
                self.chunks.remove(&chunk_coord);
            }
            return changed;
        }

        let changed = {
            let chunk = self
                .chunks
                .entry(chunk_coord)
                .or_insert_with(|| TileChunk2d::new(chunk_size));
            let idx = tile_index(chunk_size, lx, ly);
            let previous = chunk.tiles[idx];
            if previous == tile_id {
                false
            } else {
                chunk.tiles[idx] = tile_id;
                if previous == TILE_ID_EMPTY {
                    chunk.non_empty_count = chunk.non_empty_count.saturating_add(1);
                }
                true
            }
        };
        if changed {
            self.bump_world_revision();
            if let Some(chunk) = self.chunks.get_mut(&chunk_coord) {
                chunk.revision = self.world_revision;
            }
            self.dirty_chunks.insert(chunk_coord);
        }
        changed
    }

    pub fn apply_mutation(&mut self, mutation: TileMutation2d) -> bool {
        self.set_tile(mutation.coord, mutation.tile_id)
    }

    pub fn apply_mutations<I>(&mut self, mutations: I) -> usize
    where
        I: IntoIterator<Item = TileMutation2d>,
    {
        let mut changed = 0usize;
        for mutation in mutations {
            if self.apply_mutation(mutation) {
                changed = changed.saturating_add(1);
            }
        }
        changed
    }

    /// Fill an inclusive tile rectangle. Returns number of changed tiles.
    pub fn fill_rect(&mut self, min: TileCoord2d, max: TileCoord2d, tile_id: u16) -> usize {
        let min_x = min.x.min(max.x);
        let max_x = min.x.max(max.x);
        let min_y = min.y.min(max.y);
        let max_y = min.y.max(max.y);
        let mut changed = 0usize;
        for y in min_y..=max_y {
            for x in min_x..=max_x {
                if self.set_tile(TileCoord2d::new(x, y), tile_id) {
                    changed = changed.saturating_add(1);
                }
            }
        }
        changed
    }

    /// Returns and clears dirty chunk list in deterministic order.
    pub fn take_dirty_chunks(&mut self) -> Vec<TileChunkCoord2d> {
        let mut out = self.dirty_chunks.drain().collect::<Vec<_>>();
        out.sort_by_key(|coord| (coord.y, coord.x));
        out
    }

    /// Collect visible non-empty tiles inside the side-view rectangle.
    ///
    /// `max_tiles_override` can cap emission per frame for deterministic performance.
    pub fn collect_visible_tiles(
        &mut self,
        view: TileView2d,
        max_tiles_override: Option<usize>,
    ) -> TileVisibleSet2d {
        let mut result = TileVisibleSet2d::default();

        let zoom = view.zoom.max(0.05);
        let half_x = (view.half_size[0].abs().max(0.5) / zoom).max(0.5);
        let half_y = (view.half_size[1].abs().max(0.5) / zoom).max(0.5);

        let min_x = (view.center[0] - half_x).floor() as i32;
        let max_x = (view.center[0] + half_x).ceil() as i32;
        let min_y = (view.center[1] - half_y).floor() as i32;
        let max_y = (view.center[1] + half_y).ceil() as i32;

        let chunk_size = self.config.chunk_size as i32;
        let chunk_min_x = div_floor(min_x, chunk_size);
        let chunk_max_x = div_floor(max_x, chunk_size);
        let chunk_min_y = div_floor(min_y, chunk_size);
        let chunk_max_y = div_floor(max_y, chunk_size);

        let max_visible = max_tiles_override
            .unwrap_or(self.config.max_visible_tiles)
            .max(1);

        for chunk_y in chunk_min_y..=chunk_max_y {
            for chunk_x in chunk_min_x..=chunk_max_x {
                let chunk_coord = TileChunkCoord2d::new(chunk_x, chunk_y);
                let Some(chunk) = self.chunks.get(&chunk_coord) else {
                    continue;
                };

                result.telemetry.visible_chunks = result.telemetry.visible_chunks.saturating_add(1);
                if chunk.non_empty_count == 0 {
                    continue;
                }

                let origin_x = chunk_x * chunk_size;
                let origin_y = chunk_y * chunk_size;
                let local_min_x = (min_x - origin_x).clamp(0, chunk_size - 1) as usize;
                let local_max_x = (max_x - origin_x).clamp(0, chunk_size - 1) as usize;
                let local_min_y = (min_y - origin_y).clamp(0, chunk_size - 1) as usize;
                let local_max_y = (max_y - origin_y).clamp(0, chunk_size - 1) as usize;

                for ly in local_min_y..=local_max_y {
                    for lx in local_min_x..=local_max_x {
                        let idx = tile_index(self.config.chunk_size as usize, lx, ly);
                        let tile_id = chunk.tiles[idx];
                        if tile_id == TILE_ID_EMPTY {
                            continue;
                        }

                        result.telemetry.visible_tiles =
                            result.telemetry.visible_tiles.saturating_add(1);
                        if result.tiles.len() < max_visible {
                            result.tiles.push(TileVisibleInstance2d {
                                coord: TileCoord2d::new(origin_x + lx as i32, origin_y + ly as i32),
                                tile_id,
                                layer: -220,
                            });
                        }
                    }
                }
            }
        }

        result.telemetry.loaded_chunks = self.chunks.len();
        result.telemetry.dirty_chunks = self.dirty_chunks.len();
        result.telemetry.emitted_tiles = result.tiles.len();
        result.telemetry.culled_tiles = result
            .telemetry
            .visible_tiles
            .saturating_sub(result.telemetry.emitted_tiles);
        result.telemetry.world_revision = self.world_revision;
        result.telemetry.mutation_count = self.mutation_count;

        self.last_frame_telemetry = result.telemetry;
        result
    }

    pub fn telemetry_snapshot(&self) -> TileWorldFrameTelemetry {
        let mut telemetry = self.last_frame_telemetry;
        telemetry.loaded_chunks = self.chunks.len();
        telemetry.dirty_chunks = self.dirty_chunks.len();
        telemetry.world_revision = self.world_revision;
        telemetry.mutation_count = self.mutation_count;
        telemetry
    }

    fn bump_world_revision(&mut self) {
        self.world_revision = self.world_revision.saturating_add(1);
        self.mutation_count = self.mutation_count.saturating_add(1);
    }
}

fn split_tile_coord(coord: TileCoord2d, chunk_size: i32) -> (TileChunkCoord2d, usize, usize) {
    let chunk_x = div_floor(coord.x, chunk_size);
    let chunk_y = div_floor(coord.y, chunk_size);
    let local_x = (coord.x - chunk_x * chunk_size) as usize;
    let local_y = (coord.y - chunk_y * chunk_size) as usize;
    (TileChunkCoord2d::new(chunk_x, chunk_y), local_x, local_y)
}

#[inline]
fn tile_index(chunk_size: usize, x: usize, y: usize) -> usize {
    y * chunk_size + x
}

#[inline]
fn div_floor(value: i32, divisor: i32) -> i32 {
    debug_assert!(divisor > 0);
    let mut q = value / divisor;
    let r = value % divisor;
    if r < 0 {
        q -= 1;
    }
    q
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_get_tile_crosses_chunk_boundaries_and_negative_coords() {
        let mut world = ChunkedTileWorld2d::new(TileWorld2dConfig {
            chunk_size: 8,
            max_visible_tiles: 128,
        });
        assert_eq!(world.tile(TileCoord2d::new(-1, -1)), TILE_ID_EMPTY);

        assert!(world.set_tile(TileCoord2d::new(-1, -1), 7));
        assert!(world.set_tile(TileCoord2d::new(8, 0), 9));
        assert_eq!(world.tile(TileCoord2d::new(-1, -1)), 7);
        assert_eq!(world.tile(TileCoord2d::new(8, 0)), 9);
        assert_eq!(world.loaded_chunk_count(), 2);
    }

    #[test]
    fn fill_rect_updates_only_local_chunks() {
        let mut world = ChunkedTileWorld2d::new(TileWorld2dConfig {
            chunk_size: 8,
            max_visible_tiles: 512,
        });

        let changed = world.fill_rect(TileCoord2d::new(0, 0), TileCoord2d::new(15, 7), 3);
        assert_eq!(changed, 16 * 8);
        assert_eq!(world.loaded_chunk_count(), 2);

        let dirty = world.take_dirty_chunks();
        assert_eq!(dirty.len(), 2);
        assert!(dirty.contains(&TileChunkCoord2d::new(0, 0)));
        assert!(dirty.contains(&TileChunkCoord2d::new(1, 0)));
    }

    #[test]
    fn collect_visible_tiles_respects_budget_and_reports_telemetry() {
        let mut world = ChunkedTileWorld2d::new(TileWorld2dConfig {
            chunk_size: 8,
            max_visible_tiles: 12,
        });
        world.fill_rect(TileCoord2d::new(-8, -4), TileCoord2d::new(8, 4), 1);

        let visible = world.collect_visible_tiles(
            TileView2d {
                center: [0.0, 0.0],
                half_size: [16.0, 8.0],
                zoom: 1.0,
            },
            Some(12),
        );

        assert_eq!(visible.tiles.len(), 12);
        assert!(visible.telemetry.visible_tiles > visible.telemetry.emitted_tiles);
        assert!(visible.telemetry.culled_tiles > 0);
        assert!(visible.telemetry.visible_chunks > 0);
    }

    #[test]
    fn digging_last_tile_drops_empty_chunk() {
        let mut world = ChunkedTileWorld2d::new(TileWorld2dConfig::default());
        let coord = TileCoord2d::new(2, 2);
        assert!(world.set_tile(coord, 11));
        assert_eq!(world.loaded_chunk_count(), 1);

        assert!(world.apply_mutation(TileMutation2d::dig(coord)));
        assert_eq!(world.tile(coord), TILE_ID_EMPTY);
        assert_eq!(world.loaded_chunk_count(), 0);
    }
}
