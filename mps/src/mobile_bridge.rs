//! MPS → MGS mobil köprüsü.
//!
//! `mobile` feature flag ile etkinleşir. MPS scheduler iş yükü
//! bilgilerini MGS tile planına dönüştürür.
//!
//! Kullanım:
//! ```ignore
//! let adapter = MpsMobileAdapter::new("Adreno 750");
//! let plan = adapter.plan(64, 256, 1920, 1080);
//! ```

use mgs::bridge::{MgsBridgePlan, MpsWorkloadHint};
use mgs::{MgsBridge, MobileGpuProfile};

/// MPS workload bilgilerini MGS tile planına dönüştüren adaptör.
pub struct MpsMobileAdapter {
    bridge: MgsBridge,
}

impl MpsMobileAdapter {
    /// GPU adından adaptör oluşturur.
    ///
    /// `gpu_name` platform API'sinden (Android `getDeviceName`, iOS `MTLDevice.name` vb.)
    /// alınan ham cihaz adı olmalıdır. Tanınmayan adlar için güvenli varsayılanlar kullanılır.
    pub fn new(gpu_name: &str) -> Self {
        let profile = MobileGpuProfile::detect(gpu_name);
        Self {
            bridge: MgsBridge::new(profile),
        }
    }

    /// MPS scheduler metriklerinden bir MGS tile planı üretir.
    ///
    /// - `task_count`: Kuyruktaki toplam görev sayısı (draw call tahmini).
    /// - `estimated_transfer_kb`: Tüm görevler için toplam bellek transferi (KB).
    /// - `target_width` / `target_height`: Render hedefi çözünürlüğü (piksel).
    pub fn plan(
        &self,
        task_count: u32,
        estimated_transfer_kb: u32,
        target_width: u32,
        target_height: u32,
    ) -> MgsBridgePlan {
        let hint = MpsWorkloadHint {
            transfer_size_kb: estimated_transfer_kb,
            object_count: task_count,
            target_width,
            target_height,
            latency_budget_ms: 0,
        };
        self.bridge.translate(hint)
    }

    /// Gecikme bütçesiyle plan üretir (ms; 0 = kısıtlama yok).
    pub fn plan_with_latency(
        &self,
        task_count: u32,
        estimated_transfer_kb: u32,
        target_width: u32,
        target_height: u32,
        latency_budget_ms: u32,
    ) -> MgsBridgePlan {
        let hint = MpsWorkloadHint {
            transfer_size_kb: estimated_transfer_kb,
            object_count: task_count,
            target_width,
            target_height,
            latency_budget_ms,
        };
        self.bridge.translate(hint)
    }

    /// Altındaki GPU profilini döner.
    pub fn gpu_profile(&self) -> &MobileGpuProfile {
        self.bridge.profile()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adreno_adapter_produces_nonempty_plan() {
        let adapter = MpsMobileAdapter::new("Adreno 750");
        let plan = adapter.plan(128, 512, 1920, 1080);
        assert!(!plan.tile_plan.assignments.is_empty());
    }

    #[test]
    fn mali_adapter_covers_all_tasks() {
        let adapter = MpsMobileAdapter::new("Mali-G78 MC24");
        let plan = adapter.plan(200, 800, 2560, 1440);
        let total: u32 = plan.tile_plan.assignments.iter().map(|a| a.draw_calls).sum();
        assert_eq!(total, 200);
    }

    #[test]
    fn unknown_gpu_uses_safe_defaults() {
        let adapter = MpsMobileAdapter::new("SomeUnknownGPU X1");
        let plan = adapter.plan(32, 128, 1280, 720);
        assert!(!plan.tile_plan.assignments.is_empty());
    }
}
