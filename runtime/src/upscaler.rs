//! Runtime upscaler policy surface for v0.3.0.
//!
//! This module keeps quality/profile decisions in `runtime/src` so both TLApp and editor/runtime
//! tools can share one deterministic policy. The first step exposes FSR-oriented mode selection and
//! render-scale resolution with fail-soft semantics.

/// FSR mode policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FsrMode {
    Off,
    #[default]
    Auto,
    On,
    /// Dynamic mode: render scale is driven by scene object proximity each frame.
    /// The runtime computes a smoothed distance metric and writes it into
    /// `FsrConfig::render_scale_override` before calling `resolve_fsr_status`.
    Dynamo,
}

impl FsrMode {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "off" | "0" | "false" => Some(Self::Off),
            "auto" => Some(Self::Auto),
            "on" | "1" | "true" | "fsr1" => Some(Self::On),
            "dynamo" | "dynamic" => Some(Self::Dynamo),
            _ => None,
        }
    }
}

/// Configuration for `FsrMode::Dynamo` distance-driven upscaling.
///
/// Objects closer than `near_m` are rendered at native resolution (scale = 1.0).
/// Objects farther than `far_m` are rendered at `far_scale`.
/// Scale is linearly interpolated between the two thresholds and smoothed with
/// an exponential moving average to avoid per-frame flickering.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FsrDynamoConfig {
    /// Distance in metres below which FSR is fully disabled (render scale = 1.0).
    pub near_m: f32,
    /// Distance in metres above which FSR reaches its maximum strength (`far_scale`).
    pub far_m: f32,
    /// Render scale applied when all visible objects are at or beyond `far_m`.
    pub far_scale: f32,
    /// EMA smoothing coefficient applied to scale changes each frame (0 = instant, higher = slower).
    /// Typical value: 0.80–0.92.
    pub smoothing: f32,
}

impl Default for FsrDynamoConfig {
    fn default() -> Self {
        Self {
            near_m: 6.0,
            far_m: 28.0,
            far_scale: 0.67, // Quality preset equivalent
            smoothing: 0.88,
        }
    }
}

impl FsrDynamoConfig {
    /// Compute the target render scale for a given nearest-object distance.
    pub fn scale_for_distance(&self, nearest_m: f32) -> f32 {
        let t = ((nearest_m - self.near_m) / (self.far_m - self.near_m).max(1e-3)).clamp(0.0, 1.0);
        1.0 - t * (1.0 - self.far_scale.clamp(0.50, 1.0))
    }
}

/// FSR quality presets mapped to render scale.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FsrQualityPreset {
    Native,
    UltraQuality,
    #[default]
    Quality,
    Balanced,
    Performance,
}

impl FsrQualityPreset {
    pub fn parse(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "native" => Some(Self::Native),
            "ultra" | "ultraquality" | "ultra_quality" => Some(Self::UltraQuality),
            "quality" => Some(Self::Quality),
            "balanced" => Some(Self::Balanced),
            "performance" | "perf" => Some(Self::Performance),
            _ => None,
        }
    }

    pub const fn render_scale(self) -> f32 {
        match self {
            Self::Native => 1.00,
            Self::UltraQuality => 0.77,
            Self::Quality => 0.67,
            Self::Balanced => 0.59,
            Self::Performance => 0.50,
        }
    }
}

/// User-facing upscaler config consumed by renderer policy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FsrConfig {
    pub mode: FsrMode,
    pub quality: FsrQualityPreset,
    pub sharpness: f32,
    pub render_scale_override: Option<f32>,
}

impl Default for FsrConfig {
    fn default() -> Self {
        Self {
            mode: FsrMode::Auto,
            quality: FsrQualityPreset::Quality,
            sharpness: 0.35,
            render_scale_override: None,
        }
    }
}

/// Effective renderer-side FSR status after fail-soft resolution.
#[derive(Debug, Clone, PartialEq)]
pub struct FsrStatus {
    pub requested_mode: FsrMode,
    pub active: bool,
    pub render_scale: f32,
    pub sharpness: f32,
    pub reason: String,
}

/// Resolve runtime FSR policy to an effective status.
pub fn resolve_fsr_status(config: FsrConfig, backend: wgpu::Backend) -> FsrStatus {
    let backend_supported = matches!(
        backend,
        wgpu::Backend::Vulkan | wgpu::Backend::Metal | wgpu::Backend::Dx12
    );
    let requested_scale = config
        .render_scale_override
        .unwrap_or_else(|| config.quality.render_scale())
        .clamp(0.50, 1.00);
    let sharpness = config.sharpness.clamp(0.0, 1.0);

    match config.mode {
        FsrMode::Off => FsrStatus {
            requested_mode: config.mode,
            active: false,
            render_scale: 1.0,
            sharpness,
            reason: "fsr disabled by mode=off".to_string(),
        },
        FsrMode::Auto => FsrStatus {
            requested_mode: config.mode,
            active: backend_supported,
            render_scale: if backend_supported {
                requested_scale
            } else {
                1.0
            },
            sharpness,
            reason: if backend_supported {
                String::new()
            } else {
                "fsr auto fallback: backend not in Vulkan/Metal/DX12 support set".to_string()
            },
        },
        FsrMode::On => FsrStatus {
            requested_mode: config.mode,
            active: backend_supported,
            render_scale: if backend_supported {
                requested_scale
            } else {
                1.0
            },
            sharpness,
            reason: if backend_supported {
                String::new()
            } else {
                "fsr mode=on requested but backend unsupported; fail-soft fallback".to_string()
            },
        },
        // Dynamo: the runtime updates render_scale_override each frame based on
        // nearest-object distance. Resolution falls to native (1.0) when objects are
        // close and rises to the far_scale when the scene is distant.
        // Treated identically to On for backend checks; the scale is fully caller-driven.
        FsrMode::Dynamo => FsrStatus {
            requested_mode: config.mode,
            active: backend_supported,
            render_scale: if backend_supported {
                requested_scale
            } else {
                1.0
            },
            sharpness,
            reason: if backend_supported {
                "dynamo: distance-driven render scale".to_string()
            } else {
                "fsr mode=dynamo requested but backend unsupported; fail-soft fallback".to_string()
            },
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quality_presets_map_to_expected_scales() {
        assert!((FsrQualityPreset::UltraQuality.render_scale() - 0.77).abs() < 1e-6);
        assert!((FsrQualityPreset::Quality.render_scale() - 0.67).abs() < 1e-6);
        assert!((FsrQualityPreset::Balanced.render_scale() - 0.59).abs() < 1e-6);
        assert!((FsrQualityPreset::Performance.render_scale() - 0.50).abs() < 1e-6);
    }

    #[test]
    fn off_mode_forces_native_scale() {
        let status = resolve_fsr_status(
            FsrConfig {
                mode: FsrMode::Off,
                quality: FsrQualityPreset::Performance,
                sharpness: 0.4,
                render_scale_override: Some(0.5),
            },
            wgpu::Backend::Vulkan,
        );
        assert!(!status.active);
        assert!((status.render_scale - 1.0).abs() < 1e-6);
    }

    #[test]
    fn on_mode_fail_softs_on_unsupported_backend() {
        let status = resolve_fsr_status(FsrConfig::default(), wgpu::Backend::Gl);
        assert!(!status.active);
        assert!((status.render_scale - 1.0).abs() < 1e-6);
        assert!(!status.reason.is_empty());
    }
}
