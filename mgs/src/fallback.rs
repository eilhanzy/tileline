//! Serial fallback zinciri.
//!
//! MGS'in temel felsefesi: her adımda bir üst çözüme geçmek yerine
//! mevcut kapasiteyle en doğru işi yapmak. Fallback zinciri, bir workload
//! için sıralı olarak denenen stratejileri tutar.
//!
//! GMS'in aksine bu zincir tamamen senkron ve tek iş parçacıklıdır.

use crate::hardware::MobileGpuProfile;

/// Tek bir fallback adımının sonucu.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FallbackOutcome {
    /// Bu adım başarıyla tamamlandı — zincir durabilir.
    Handled,
    /// Bu adım başarısız oldu — zincir bir sonraki adıma geçmeli.
    Skip,
    /// Zincirin tüm adımları tükendi; hiçbiri işlemi üstlenemedi.
    Exhausted,
}

/// Fallback zincirinin aşamaları.
///
/// Her aşama bir öncekinden daha düşük kalite veya daha az özellik
/// sunar; karşılığında güç ve bellek kullanımı azalır.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum FallbackLevel {
    /// Tam TBDR-aware tile planner — birincil yol.
    TbdrOptimized = 0,
    /// Tile bütçesi olmadan basit tile çizimi.
    SimpleTile = 1,
    /// Tam ekran, tile olmadan immediate-mode render.
    FullscreenImmediate = 2,
    /// CPU-side yazılım rasterleştirmesi (son çare).
    SoftwareRasterize = 3,
}

impl FallbackLevel {
    /// Bir sonraki daha düşük aşamayı döner; en alt seviyedeyse `None`.
    pub fn next(self) -> Option<Self> {
        match self {
            Self::TbdrOptimized => Some(Self::SimpleTile),
            Self::SimpleTile => Some(Self::FullscreenImmediate),
            Self::FullscreenImmediate => Some(Self::SoftwareRasterize),
            Self::SoftwareRasterize => None,
        }
    }

    /// Kısa açıklama (log ve hata raporları için).
    pub fn label(self) -> &'static str {
        match self {
            Self::TbdrOptimized => "tbdr-optimized",
            Self::SimpleTile => "simple-tile",
            Self::FullscreenImmediate => "fullscreen-immediate",
            Self::SoftwareRasterize => "sw-rasterize",
        }
    }
}

/// Zincir çalıştırıldığında üretilen özet rapor.
#[derive(Debug, Clone)]
pub struct FallbackReport {
    /// Başarıyla tamamlanan aşama.
    pub resolved_level: FallbackLevel,
    /// Denenen toplam adım sayısı (başarısız + başarılı).
    pub steps_attempted: u32,
    /// Zincir tükenmeden başarıyla sonuçlandı mı.
    pub success: bool,
}

/// Serial fallback zinciri.
///
/// `run` çağrısı, callback sırasıyla en uygun aşamadan başlayarak
/// dener; callback `FallbackOutcome::Handled` dönene kadar veya
/// tüm aşamalar tükenene kadar devam eder.
pub struct FallbackChain {
    start: FallbackLevel,
    profile: MobileGpuProfile,
}

impl FallbackChain {
    /// Yeni bir zincir oluşturur.
    ///
    /// `profile` donanım durumuna göre başlangıç aşamasını belirler;
    /// bilinmeyen GPU'lar `SimpleTile`'dan başlar.
    pub fn new(profile: MobileGpuProfile) -> Self {
        let start = if profile.is_mobile_tbdr() {
            FallbackLevel::TbdrOptimized
        } else {
            FallbackLevel::SimpleTile
        };
        Self { start, profile }
    }

    /// Başlangıç aşamasını geçersiz kıl (test veya platforma özgü başlatma için).
    pub fn with_start_level(mut self, level: FallbackLevel) -> Self {
        self.start = level;
        self
    }

    /// Callback `f` ile zinciri sırayla çalıştırır.
    ///
    /// `f`, her aşamada çağrılır ve bir `FallbackOutcome` döner.
    /// `Handled` dönüldüğünde zincir başarıyla tamamlanmış kabul edilir.
    pub fn run<F>(&self, mut f: F) -> FallbackReport
    where
        F: FnMut(FallbackLevel, &MobileGpuProfile) -> FallbackOutcome,
    {
        let mut current = Some(self.start);
        let mut steps = 0u32;

        while let Some(level) = current {
            steps += 1;
            match f(level, &self.profile) {
                FallbackOutcome::Handled => {
                    return FallbackReport {
                        resolved_level: level,
                        steps_attempted: steps,
                        success: true,
                    };
                }
                FallbackOutcome::Skip => {
                    current = level.next();
                }
                FallbackOutcome::Exhausted => break,
            }
        }

        FallbackReport {
            resolved_level: self.start,
            steps_attempted: steps,
            success: false,
        }
    }

    /// Donanım profilini döner.
    pub fn profile(&self) -> &MobileGpuProfile {
        &self.profile
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hardware::MobileGpuProfile;

    #[test]
    fn tbdr_profile_starts_at_optimized_level() {
        let profile = MobileGpuProfile::detect("Adreno 750");
        let chain = FallbackChain::new(profile);
        assert_eq!(chain.start, FallbackLevel::TbdrOptimized);
    }

    #[test]
    fn unknown_profile_starts_at_simple_tile() {
        let profile = MobileGpuProfile::detect("Unknown GPU");
        let chain = FallbackChain::new(profile);
        assert_eq!(chain.start, FallbackLevel::SimpleTile);
    }

    #[test]
    fn chain_resolves_at_first_handled_step() {
        let profile = MobileGpuProfile::detect("Adreno 750");
        let chain = FallbackChain::new(profile);

        let report = chain.run(|level, _| {
            if level == FallbackLevel::SimpleTile {
                FallbackOutcome::Handled
            } else {
                FallbackOutcome::Skip
            }
        });

        assert!(report.success);
        assert_eq!(report.resolved_level, FallbackLevel::SimpleTile);
        assert_eq!(report.steps_attempted, 2);
    }

    #[test]
    fn chain_reports_failure_when_all_skip() {
        let profile = MobileGpuProfile::detect("Adreno 750");
        let chain = FallbackChain::new(profile);
        let report = chain.run(|_, _| FallbackOutcome::Skip);
        assert!(!report.success);
    }
}
