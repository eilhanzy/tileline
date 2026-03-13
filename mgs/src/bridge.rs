//! MPS ↔ MGS köprüsü.
//!
//! Bu modül, MPS (Memory Processing System) ve MGS (Mobile Graphics Scheduler)
//! arasındaki veri aktarım sözleşmelerini tanımlar. Gerçek GPU komutları veya
//! API çağrıları içermez — yalnızca plan ve talep tiplerini taşır.
//!
//! Köprü tek yönlü çalışır: MPS bir `MpsWorkloadHint` gönderir,
//! MGS onu `MgsBridgePlan`'a dönüştürür.

use crate::fallback::{FallbackChain, FallbackLevel, FallbackOutcome};
use crate::hardware::MobileGpuProfile;
use crate::tile_planner::{MgsPlanner, TilePlan, TileWorkloadRequest};
use crate::tuning::MgsTuningProfile;

/// MPS tarafından MGS'e iletilen workload ipucu.
///
/// MPS bellek topolojisini ve veri boyutlarını bilir; MGS ise
/// tile stratejisini ve render pass sırasını planlar.
#[derive(Debug, Clone, Copy)]
pub struct MpsWorkloadHint {
    /// Bellek transfer boyutu (KB) — MGS tile bütçesini etkiler.
    pub transfer_size_kb: u32,
    /// Aktarılacak nesne sayısı — draw call tahminine dönüştürülür.
    pub object_count: u32,
    /// Render hedefi genişliği (piksel).
    pub target_width: u32,
    /// Render hedefi yüksekliği (piksel).
    pub target_height: u32,
    /// MPS'in önerdiği maksimum gecikme (ms); 0 = kısıtlama yok.
    pub latency_budget_ms: u32,
}

impl Default for MpsWorkloadHint {
    fn default() -> Self {
        Self {
            transfer_size_kb: 256,
            object_count: 64,
            target_width: 1920,
            target_height: 1080,
            latency_budget_ms: 0,
        }
    }
}

/// MGS'in MPS'e döndürdüğü köprü planı.
#[derive(Debug, Clone)]
pub struct MgsBridgePlan {
    /// Kullanılan fallback aşaması.
    pub resolved_fallback: FallbackLevel,
    /// Tile bazlı workload planı.
    pub tile_plan: TilePlan,
    /// Bellek baskısı algılandı mı?
    pub memory_pressure: bool,
    /// Güç tasarrufu modunda mı üretildi?
    pub power_saving_active: bool,
}

/// MPS ↔ MGS köprüsü.
///
/// `translate` çağrısı, `MpsWorkloadHint`'i alır, fallback zincirini
/// çalıştırır ve en uygun `MgsBridgePlan`'ı döner.
pub struct MgsBridge {
    planner: MgsPlanner,
    chain: FallbackChain,
}

impl MgsBridge {
    /// Yeni köprü oluşturur.
    pub fn new(profile: MobileGpuProfile) -> Self {
        let chain = FallbackChain::new(profile.clone());
        let planner = MgsPlanner::from_profile(profile);
        Self { planner, chain }
    }

    /// Termal durumu ve tuning'i dışarıdan geçirmek isteyenler için.
    pub fn with_tuning(profile: MobileGpuProfile, tuning: MgsTuningProfile) -> Self {
        let chain = FallbackChain::new(profile.clone());
        let planner = MgsPlanner::new(profile, tuning);
        Self { planner, chain }
    }

    /// `MpsWorkloadHint`'i `MgsBridgePlan`'a çevirir.
    pub fn translate(&self, hint: MpsWorkloadHint) -> MgsBridgePlan {
        let request = hint_to_request(&hint);
        let mut resolved_plan: Option<TilePlan> = None;
        let mut resolved_level = FallbackLevel::TbdrOptimized;

        let report = self.chain.run(|level, _profile| {
            let plan = match level {
                FallbackLevel::TbdrOptimized | FallbackLevel::SimpleTile => {
                    self.planner.plan(request)
                }
                FallbackLevel::FullscreenImmediate => {
                    // Tile olmadan, tüm draw'ları tek "tile"a koy.
                    let fullscreen_req = TileWorkloadRequest {
                        target_width: hint.target_width,
                        target_height: hint.target_height,
                        total_draw_calls: hint.object_count,
                        bytes_per_draw_kb: (hint.transfer_size_kb / hint.object_count.max(1))
                            .max(1),
                    };
                    self.planner.plan(fullscreen_req)
                }
                FallbackLevel::SoftwareRasterize => {
                    // Yazılım render: sıfır tile, minimum workload.
                    let sw_req = TileWorkloadRequest {
                        target_width: hint.target_width,
                        target_height: hint.target_height,
                        total_draw_calls: hint.object_count.min(16),
                        bytes_per_draw_kb: 1,
                    };
                    self.planner.plan(sw_req)
                }
            };

            // Bellek baskısı varsa ve bir sonraki aşama mevcutsa, atla.
            if plan.memory_pressure_detected && level != FallbackLevel::SoftwareRasterize {
                return FallbackOutcome::Skip;
            }

            resolved_plan = Some(plan);
            resolved_level = level;
            FallbackOutcome::Handled
        });

        let tile_plan = resolved_plan.unwrap_or_else(|| self.planner.plan(request));
        let memory_pressure = tile_plan.memory_pressure_detected || !report.success;

        MgsBridgePlan {
            resolved_fallback: resolved_level,
            memory_pressure,
            power_saving_active: self.planner.tuning().power_saving_mode,
            tile_plan,
        }
    }

    /// Donanım profilini döner.
    pub fn profile(&self) -> &MobileGpuProfile {
        self.planner.profile()
    }
}

fn hint_to_request(hint: &MpsWorkloadHint) -> TileWorkloadRequest {
    let bytes_per_draw_kb = if hint.object_count == 0 {
        hint.transfer_size_kb
    } else {
        (hint.transfer_size_kb / hint.object_count).max(1)
    };
    TileWorkloadRequest {
        target_width: hint.target_width,
        target_height: hint.target_height,
        total_draw_calls: hint.object_count,
        bytes_per_draw_kb,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::MobileGpuProfile;

    #[test]
    fn translate_returns_valid_plan_for_adreno() {
        let profile = MobileGpuProfile::detect("Adreno 750");
        let bridge = MgsBridge::new(profile);
        let plan = bridge.translate(MpsWorkloadHint::default());
        assert!(!plan.tile_plan.assignments.is_empty());
    }

    #[test]
    fn translate_covers_all_objects() {
        let profile = MobileGpuProfile::detect("Mali-G78 MC24");
        let bridge = MgsBridge::new(profile);
        let hint = MpsWorkloadHint {
            object_count: 128,
            transfer_size_kb: 512,
            ..Default::default()
        };
        let plan = bridge.translate(hint);
        let total: u32 = plan
            .tile_plan
            .assignments
            .iter()
            .map(|a| a.draw_calls)
            .sum();
        assert_eq!(total, 128);
    }
}
