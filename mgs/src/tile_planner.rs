//! Tile budget tabanlı workload planner.
//!
//! TBDR GPU'larda ekran, sabit boyutlu tile'lara (kareler) bölünür.
//! Her tile, GPU'nun on-chip SRAM'inde tamamen işlenir; DRAM'e erişim
//! yalnızca tile tamamlandığında gerçekleşir. Bu yüzden:
//!
//! - Tile başına vertex/fragment yükü dengeli dağıtılmalıdır.
//! - On-chip bellek kapasitesi aşılmamalıdır.
//! - Tile sayısı, donanımın paralel işleyebileceği tile sayısına
//!   saygı duymalıdır.
//!
//! `MgsPlanner`, bu kısıtları girdi olarak alır ve her tile için
//! `TileAssignment` üretir.

use crate::hardware::MobileGpuProfile;
use crate::tuning::MgsTuningProfile;

/// Tek bir tile için planlanan workload.
#[derive(Debug, Clone, Copy)]
pub struct TileAssignment {
    /// Tile'ın sol-üst köşesi (piksel).
    pub origin_x: u32,
    pub origin_y: u32,
    /// Tile genişliği ve yüksekliği (piksel).
    pub tile_w: u32,
    pub tile_h: u32,
    /// Bu tile'a atanan draw call sayısı.
    pub draw_calls: u32,
    /// Tahmini on-chip bellek kullanımı (KB).
    pub estimated_tile_memory_kb: u32,
}

/// Planlayıcıya iletilen workload isteği.
#[derive(Debug, Clone, Copy)]
pub struct TileWorkloadRequest {
    /// Render hedefinin piksel genişliği.
    pub target_width: u32,
    /// Render hedefinin piksel yüksekliği.
    pub target_height: u32,
    /// Toplam draw call sayısı (tüm tile'lara dağıtılacak).
    pub total_draw_calls: u32,
    /// Her draw call için tahmini fragment bellek (KB).
    pub bytes_per_draw_kb: u32,
}

impl Default for TileWorkloadRequest {
    fn default() -> Self {
        Self {
            target_width: 1920,
            target_height: 1080,
            total_draw_calls: 64,
            bytes_per_draw_kb: 4,
        }
    }
}

/// Planlama sonucu.
#[derive(Debug, Clone)]
pub struct TilePlan {
    /// Her tile için hesaplanan atama.
    pub assignments: Vec<TileAssignment>,
    /// Kullanılan tile boyutu (piksel).
    pub tile_px: u32,
    /// Toplam tile sayısı.
    pub tile_count: u32,
    /// On-chip bellek sınırı aşıldı mı (uyarı).
    pub memory_pressure_detected: bool,
}

impl TilePlan {
    /// En yüksek bellek kullanımlı tile'ı döner (hata ayıklama).
    pub fn peak_tile(&self) -> Option<&TileAssignment> {
        self.assignments
            .iter()
            .max_by_key(|a| a.estimated_tile_memory_kb)
    }
}

/// MGS ana planlayıcısı.
///
/// Donanım profiline ve tuning parametrelerine göre tile bazlı
/// workload planı üretir.
pub struct MgsPlanner {
    profile: MobileGpuProfile,
    tuning: MgsTuningProfile,
}

impl MgsPlanner {
    /// Yeni planlayıcı oluşturur.
    pub fn new(profile: MobileGpuProfile, tuning: MgsTuningProfile) -> Self {
        Self { profile, tuning }
    }

    /// Donanım profilinden otomatik tuning ile planlayıcı oluşturur.
    pub fn from_profile(profile: MobileGpuProfile) -> Self {
        let tuning = MgsTuningProfile::from_profile(&profile);
        Self { profile, tuning }
    }

    /// Workload isteğinden bir `TilePlan` üretir.
    pub fn plan(&self, request: TileWorkloadRequest) -> TilePlan {
        let tile_px = self.effective_tile_px();
        let tiles_x = (request.target_width + tile_px - 1) / tile_px;
        let tiles_y = (request.target_height + tile_px - 1) / tile_px;
        let tile_count = tiles_x * tiles_y;

        let draws_per_tile = if tile_count == 0 {
            request.total_draw_calls
        } else {
            // Eşit dağıtım; kalan draw'lar son tile'a eklenir.
            (request.total_draw_calls + tile_count - 1) / tile_count
        };

        let mut assignments = Vec::with_capacity(tile_count as usize);
        let mut remaining_draws = request.total_draw_calls;
        let mut memory_pressure_detected = false;

        for row in 0..tiles_y {
            for col in 0..tiles_x {
                let assigned = remaining_draws.min(draws_per_tile);
                remaining_draws = remaining_draws.saturating_sub(assigned);

                let estimated_kb = assigned * request.bytes_per_draw_kb;
                if estimated_kb > self.profile.tile_memory_kb {
                    memory_pressure_detected = true;
                }

                assignments.push(TileAssignment {
                    origin_x: col * tile_px,
                    origin_y: row * tile_px,
                    tile_w: tile_px.min(request.target_width.saturating_sub(col * tile_px)),
                    tile_h: tile_px.min(request.target_height.saturating_sub(row * tile_px)),
                    draw_calls: assigned,
                    estimated_tile_memory_kb: estimated_kb,
                });
            }
        }

        TilePlan {
            assignments,
            tile_px,
            tile_count,
            memory_pressure_detected,
        }
    }

    /// Tuning ve donanım mimarisine göre efektif tile boyutunu hesaplar.
    fn effective_tile_px(&self) -> u32 {
        if let Some(override_px) = self.tuning.tile_size_override_px {
            return override_px;
        }
        let arch_default = self.profile.architecture.recommended_tile_px();
        // Güç tasarrufu modunda tile boyutunu küçült.
        if self.tuning.power_saving_mode {
            (arch_default / 2).max(8)
        } else {
            arch_default
        }
    }

    /// Mevcut donanım profili.
    pub fn profile(&self) -> &MobileGpuProfile {
        &self.profile
    }

    /// Mevcut tuning profili.
    pub fn tuning(&self) -> &MgsTuningProfile {
        &self.tuning
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::MobileGpuProfile;

    #[test]
    fn plan_covers_all_draw_calls() {
        let profile = MobileGpuProfile::detect("Adreno 750");
        let planner = MgsPlanner::from_profile(profile);
        let req = TileWorkloadRequest {
            target_width: 1920,
            target_height: 1080,
            total_draw_calls: 200,
            bytes_per_draw_kb: 2,
        };
        let plan = planner.plan(req);
        let total: u32 = plan.assignments.iter().map(|a| a.draw_calls).sum();
        assert_eq!(total, 200);
    }

    #[test]
    fn plan_tile_px_matches_architecture_default() {
        let profile = MobileGpuProfile::detect("Mali-G78 MC24");
        let planner = MgsPlanner::from_profile(profile);
        let plan = planner.plan(TileWorkloadRequest::default());
        // Mali önerilen tile boyutu: 16px
        assert_eq!(plan.tile_px, 16);
    }

    #[test]
    fn memory_pressure_detected_when_over_budget() {
        let profile = MobileGpuProfile::detect("Adreno 610");
        let planner = MgsPlanner::from_profile(profile);
        let req = TileWorkloadRequest {
            total_draw_calls: 1,
            bytes_per_draw_kb: 9999, // tile belleğinden çok fazla
            ..Default::default()
        };
        let plan = planner.plan(req);
        assert!(plan.memory_pressure_detected);
    }
}
