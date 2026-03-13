//! GMS Grafik Benchmark — Ekran Üstü FPS / Frame-Time Göstergesi.
//!
//! [`render_benchmark`] ile birebir aynı wgpu render altyapısını kullanır;
//! ekstra bağımlılık olmaksızın saf wgpu ile oluşturulan bir HUD katmanı ekler.
//!
//! # HUD İçeriği
//!
//! - **Anlık FPS** — renk kodlu (yeşil ≥60, sarı ≥30, kırmızı <30), ekrana yazdırılır
//! - **Ortalama / tepe FPS** — ikinci satır
//! - **p95 / p99 frame-time** ve **σ (standart sapma)**
//! - **Benchmark fazı** rozeti — `WARM` / `SMPL` / `DONE`
//! - **Frame-time çubuk grafik** — son 120 frame, hedef çizgisiyle (16.67 ms = 60 FPS)
//! - **GPU adı** ve **GMS donanım skoru**
//!
//! # Teknik Yaklaşım
//!
//! HUD, sıfır harici bağımlılıkla uygulanmıştır:
//!
//! 1. CPU tarafında hardcoded 5×7 bitmap piksel fontla metin RGBA tamponuna işlenir.
//! 2. Tampon, her frame bir wgpu staging buffer üzerinden GPU tekstürüne yüklenir.
//! 3. Alfa karıştırmalı (SrcAlpha / 1-SrcAlpha) bir quad pipeline, HUD tekstürünü
//!    clear-pass çıktısı üzerine kompozitler.
//!
//! ```text
//! [clear-pass] → renkli arka plan
//! [hud-pass]   → LoadOp::Load + alfa blend → FPS metni + grafikler
//! ```
//!
//! # Kullanım
//!
//! ```text
//! cargo run -p gms --example gms_bench_visual --release -- [options]
//! ```
//!
//! | Bayrak | Değerler | Varsayılan |
//! |--------|----------|------------|
//! | `--mode` | `auto\|stable\|max` | `auto` |
//! | `--vsync` | `auto\|on\|off` | `auto` |
//! | `--multi-gpu` | `auto\|on\|off` | `off` |
//! | `--warmup` | saniye | `2` |
//! | `--duration` | saniye | `10` |
//! | `--resolution` | `WxH` | `1280x720` |

use std::collections::VecDeque;
use std::env;
use std::error::Error;
use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};

use gms::{
    GmsRuntimeTuningProfile, GpuInventory, MultiGpuExecutor, MultiGpuExecutorConfig,
    MultiGpuExecutorSummary, MultiGpuInitPolicy, MultiGpuWorkloadRequest,
};
use wgpu::{Color, CompositeAlphaMode, PresentMode, SurfaceError, TextureFormat};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

// ── Giriş noktası ─────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn Error>> {
    let options = CliOptions::parse_from_env()?;
    let event_loop = EventLoop::new()?;
    let mut app = VisualBenchmarkApp::new(options);
    event_loop.run_app(&mut app)?;
    Ok(())
}

// ── CLI (render_benchmark ile aynı) ──────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
struct CliOptions {
    mode_override: ModeOverride,
    vsync_override: VsyncOverride,
    multi_gpu_override: MultiGpuOverride,
    warmup_duration: Duration,
    sample_duration: Duration,
    resolution: PhysicalSize<u32>,
}

impl Default for CliOptions {
    fn default() -> Self {
        Self {
            mode_override: ModeOverride::Auto,
            vsync_override: VsyncOverride::Auto,
            multi_gpu_override: MultiGpuOverride::Off,
            warmup_duration: Duration::from_secs(2),
            sample_duration: Duration::from_secs(10),
            resolution: PhysicalSize::new(1280, 720),
        }
    }
}

impl CliOptions {
    fn parse_from_env() -> Result<Self, Box<dyn Error>> {
        let mut opts = Self::default();
        let mut args = env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--help" | "-h" => {
                    print_usage();
                    std::process::exit(0);
                }
                "--mode" => {
                    let v = next_arg(&mut args, "--mode")?;
                    opts.mode_override = ModeOverride::parse(&v)?;
                }
                "--vsync" => {
                    let v = next_arg(&mut args, "--vsync")?;
                    opts.vsync_override = VsyncOverride::parse(&v)?;
                }
                "--multi-gpu" => {
                    let v = next_arg(&mut args, "--multi-gpu")?;
                    opts.multi_gpu_override = MultiGpuOverride::parse(&v)?;
                }
                "--duration" => {
                    let v = next_arg(&mut args, "--duration")?;
                    opts.sample_duration = parse_secs(&v, "--duration")?;
                }
                "--warmup" => {
                    let v = next_arg(&mut args, "--warmup")?;
                    opts.warmup_duration = parse_secs(&v, "--warmup")?;
                }
                "--resolution" => {
                    let v = next_arg(&mut args, "--resolution")?;
                    opts.resolution = parse_res(&v)?;
                }
                other => return Err(Box::new(SE(format!("bilinmeyen: {other} (--help)")))),
            }
        }
        if opts.sample_duration.is_zero() {
            return Err(Box::new(SE("--duration > 0 olmalı".into())));
        }
        Ok(opts)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModeOverride {
    Auto,
    Stable,
    Max,
}
impl ModeOverride {
    fn parse(v: &str) -> Result<Self, Box<dyn Error>> {
        Ok(match v.to_ascii_lowercase().as_str() {
            "auto" => Self::Auto,
            "stable" => Self::Stable,
            "max" | "throughput" => Self::Max,
            _ => return Err(Box::new(SE(format!("geçersiz --mode: {v}")))),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VsyncOverride {
    Auto,
    On,
    Off,
}
impl VsyncOverride {
    fn parse(v: &str) -> Result<Self, Box<dyn Error>> {
        Ok(match v.to_ascii_lowercase().as_str() {
            "auto" => Self::Auto,
            "on" | "true" | "1" => Self::On,
            "off" | "false" | "0" => Self::Off,
            _ => return Err(Box::new(SE(format!("geçersiz --vsync: {v}")))),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MultiGpuOverride {
    Auto,
    On,
    Off,
}
impl MultiGpuOverride {
    fn parse(v: &str) -> Result<Self, Box<dyn Error>> {
        Ok(match v.to_ascii_lowercase().as_str() {
            "auto" => Self::Auto,
            "on" | "true" | "1" => Self::On,
            "off" | "false" | "0" => Self::Off,
            _ => return Err(Box::new(SE(format!("geçersiz --multi-gpu: {v}")))),
        })
    }
}

fn next_arg(it: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, Box<dyn Error>> {
    it.next()
        .ok_or_else(|| Box::new(SE(format!("{flag} için değer eksik"))) as _)
}
fn parse_secs(v: &str, flag: &str) -> Result<Duration, Box<dyn Error>> {
    let s = v
        .parse::<f64>()
        .map_err(|_| Box::new(SE(format!("{flag}: saniye girin"))) as Box<dyn Error>)?;
    if !s.is_finite() || s < 0.0 {
        return Err(Box::new(SE(format!("{flag} geçersiz"))));
    }
    Ok(Duration::from_secs_f64(s))
}
fn parse_res(v: &str) -> Result<PhysicalSize<u32>, Box<dyn Error>> {
    let lower = v.to_ascii_lowercase();
    let (w, h) = lower
        .split_once('x')
        .ok_or_else(|| Box::new(SE(format!("çözünürlük formatı WxH: {v}"))) as Box<dyn Error>)?;
    let width = w
        .parse::<u32>()
        .map_err(|_| Box::new(SE(format!("genişlik: {w}"))) as Box<dyn Error>)?;
    let height = h
        .parse::<u32>()
        .map_err(|_| Box::new(SE(format!("yükseklik: {h}"))) as Box<dyn Error>)?;
    if width == 0 || height == 0 {
        return Err(Box::new(SE("boyutlar sıfır olamaz".into())));
    }
    Ok(PhysicalSize::new(width, height))
}
fn print_usage() {
    println!("GMS Grafik Benchmark (ekran üstü FPS)");
    println!("Kullanım: cargo run -p gms --example gms_bench_visual --release -- [options]");
    println!("  --mode auto|stable|max     (varsayılan: auto)");
    println!("  --vsync auto|on|off        (varsayılan: auto)");
    println!("  --multi-gpu auto|on|off    (varsayılan: off)");
    println!("  --warmup <sn>              (varsayılan: 2)");
    println!("  --duration <sn>            (varsayılan: 10)");
    println!("  --resolution <WxH>         (varsayılan: 1280x720)");
}

// ── Bitmap Piksel Font (5×7, sıfır bağımlılık) ────────────────────────────────
//
// Her glyph 7 satır × 5 sütunluk bir `[u8; 7]` dizisidir.
// Her satırda bit-4 = en sol sütun, bit-0 = en sağ sütun.

const GLYPH_W: usize = 5;
const GLYPH_H: usize = 7;

/// Tek bir 5×7 glyph — satır başına 1 bayt (bit-4..0 = sütun 0..4).
type Glyph = [u8; GLYPH_H];

/// Desteklenen karakter kümesi ve pixmap verisi.
///
/// `glyph_for(c)` metodu karaktere karşılık gelen glyph'i döner.
struct Font;

impl Font {
    fn glyph(c: char) -> Glyph {
        match c {
            '0' => [
                0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110,
            ],
            '1' => [
                0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110,
            ],
            '2' => [
                0b01110, 0b10001, 0b00001, 0b00110, 0b01000, 0b10000, 0b11111,
            ],
            '3' => [
                0b01110, 0b10001, 0b00001, 0b00110, 0b00001, 0b10001, 0b01110,
            ],
            '4' => [
                0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010,
            ],
            '5' => [
                0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110,
            ],
            '6' => [
                0b01110, 0b10001, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110,
            ],
            '7' => [
                0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000,
            ],
            '8' => [
                0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110,
            ],
            '9' => [
                0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b10001, 0b01110,
            ],
            '.' => [
                0b00000, 0b00000, 0b00000, 0b00000, 0b00000, 0b01100, 0b01100,
            ],
            ' ' => [0b00000; GLYPH_H],
            '-' => [
                0b00000, 0b00000, 0b00000, 0b11111, 0b00000, 0b00000, 0b00000,
            ],
            // Büyük harfler
            'A' => [
                0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001,
            ],
            'D' => [
                0b11110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b11110,
            ],
            'E' => [
                0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111,
            ],
            'F' => [
                0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000,
            ],
            'G' => [
                0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01110,
            ],
            'M' => [
                0b10001, 0b11011, 0b10101, 0b10001, 0b10001, 0b10001, 0b10001,
            ],
            'N' => [
                0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001,
            ],
            'O' => [
                0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110,
            ],
            'P' => [
                0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000,
            ],
            'R' => [
                0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001,
            ],
            'S' => [
                0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110,
            ],
            'V' => [
                0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b00100,
            ],
            'W' => [
                0b10001, 0b10001, 0b10001, 0b10101, 0b10101, 0b11011, 0b10001,
            ],
            // Küçük harfler
            'm' => [
                0b00000, 0b00000, 0b11010, 0b10101, 0b10101, 0b10001, 0b10001,
            ],
            'p' => [
                0b00000, 0b00000, 0b11110, 0b10001, 0b11110, 0b10000, 0b10000,
            ],
            's' => [
                0b00000, 0b00000, 0b01110, 0b10000, 0b01110, 0b00001, 0b11110,
            ],
            // Semboller
            'σ' => [
                0b00000, 0b01111, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110,
            ],
            'x' => [
                0b00000, 0b00000, 0b10001, 0b01010, 0b00100, 0b01010, 0b10001,
            ],
            // Fallback: boş
            _ => [0b00000; GLYPH_H],
        }
    }

    /// Bir glyph'i RGBA piksel tamponuna `scale` büyütme faktörüyle çizer.
    ///
    /// `buf_w`: tamponun piksel cinsinden genişliği.
    /// `fg`: ön plan rengi [R, G, B, A].
    fn blit(
        buf: &mut [u8],
        buf_w: usize,
        x: usize,
        y: usize,
        glyph: &Glyph,
        scale: usize,
        fg: [u8; 4],
    ) {
        for row in 0..GLYPH_H {
            let bits = glyph[row];
            for col in 0..GLYPH_W {
                if bits & (1 << (GLYPH_W - 1 - col)) != 0 {
                    for sy in 0..scale {
                        for sx in 0..scale {
                            let px = x + col * scale + sx;
                            let py = y + row * scale + sy;
                            let base = (py * buf_w + px) * 4;
                            if base + 3 < buf.len() {
                                buf[base] = fg[0];
                                buf[base + 1] = fg[1];
                                buf[base + 2] = fg[2];
                                buf[base + 3] = fg[3];
                            }
                        }
                    }
                }
            }
        }
    }

    /// Bir string'i satır başına glyph genişliği + 1 piksel boşluk bırakarak çizer.
    fn draw_str(
        buf: &mut [u8],
        buf_w: usize,
        x: usize,
        y: usize,
        text: &str,
        scale: usize,
        fg: [u8; 4],
    ) {
        let advance = (GLYPH_W + 1) * scale;
        for (i, c) in text.chars().enumerate() {
            Self::blit(buf, buf_w, x + i * advance, y, &Self::glyph(c), scale, fg);
        }
    }
}

// ── HUD veri anlık görüntüsü ─────────────────────────────────────────────────

/// Tek frame için HUD render verisi.
struct HudSnapshot {
    /// Smoothed anlık FPS (son 30 frame ağırlıklı ortalaması).
    fps: f64,
    /// Oturum ortalaması FPS.
    avg_fps: f64,
    /// Tepe FPS.
    peak_fps: f64,
    /// Ortalama frame-time (ms).
    avg_ms: f64,
    /// p95 frame-time (ms).
    p95_ms: f64,
    /// p99 frame-time (ms).
    p99_ms: f64,
    /// Frame-time standart sapması (ms).
    stddev_ms: f64,
    /// Toplam frame sayısı.
    frames: u64,
    /// Benchmark fazı etiketi (`WARM` / `SMPL` / `DONE`).
    phase: &'static str,
    /// GPU kısa adı.
    gpu_name: String,
    /// GMS donanım skoru.
    gms_score: u64,
    /// Work-units/present (burst modu; 1 = normal).
    work_units: u32,
    /// Gerçek present mode kısa etiketi ("FIFO", "MAIL", "IMMD", "AUTO").
    present_mode_label: &'static str,
    /// Son ≤120 frame-time (ms, f32) — çubuk grafik için.
    recent_ms: Vec<f32>,
}

// ── HUD Renderer (saf wgpu, sıfır bağımlılık) ────────────────────────────────

/// HUD piksel tamponu boyutları.
const HUD_PX_W: u32 = 280;
const HUD_PX_H: u32 = 168;
const HUD_BYTES: usize = (HUD_PX_W * HUD_PX_H * 4) as usize;

/// HUD'u wgpu ile ekrana çizen renderer.
///
/// İş akışı:
/// 1. `update()` — CPU tamponuna glyph + grafik çizer, staging buffer'a kopyalar.
/// 2. `encode_copy()` — staging buffer'dan GPU tekstürüne kopyalama komutunu encoder'a ekler.
/// 3. `draw()` — alfa-karıştırmalı quad pass ile HUD tekstürünü yüzeye kompozitler.
struct HudRenderer {
    pipeline: wgpu::RenderPipeline,
    /// Quad köşe verisi ([x,y,u,v] × 6 köşe); yüzey boyutu değişince güncellenir.
    vertex_buf: wgpu::Buffer,
    /// CPU → GPU piksel transferi için staging buffer (MAP_WRITE).
    staging_buf: wgpu::Buffer,
    /// GPU tarafındaki HUD tekstürü (Rgba8Unorm).
    texture: wgpu::Texture,
    bind_group: wgpu::BindGroup,
    /// CPU tarafı piksel tamponu; her frame sıfırlanıp yeniden çizilir.
    pixels: Vec<u8>,
    /// Geçerli yüzey boyutu; köşe verisi güncelleme kararı için.
    surface_size: PhysicalSize<u32>,
}

impl HudRenderer {
    /// Render pipeline, tekstür ve staging buffer'ı oluşturur.
    fn new(device: &wgpu::Device, queue: &wgpu::Queue, surface_format: TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("hud-shader"),
            source: wgpu::ShaderSource::Wgsl(HUD_WGSL.into()),
        });

        let texture = create_hud_texture(device);
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("hud-sampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("hud-bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("hud-bg"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        let pll = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("hud-pll"),
            bind_group_layouts: &[&bgl],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("hud-pipeline"),
            layout: Some(&pll),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 16, // [f32; 4] = 16 bytes
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x2],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    // Alfa karıştırma: HUD şeffaf alanları geçirgen yapar.
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let vertex_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hud-vertex-buf"),
            size: (6 * 4 * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hud-staging"),
            size: HUD_BYTES as u64,
            usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::MAP_WRITE,
            mapped_at_creation: false,
        });

        // Başlangıçta tam şeffaf tampon yükle.
        let init_pixels = vec![0u8; HUD_BYTES];
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &init_pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(HUD_PX_W * 4),
                rows_per_image: Some(HUD_PX_H),
            },
            wgpu::Extent3d {
                width: HUD_PX_W,
                height: HUD_PX_H,
                depth_or_array_layers: 1,
            },
        );

        Self {
            pipeline,
            vertex_buf,
            staging_buf,
            texture,
            bind_group,
            pixels: vec![0u8; HUD_BYTES],
            surface_size: PhysicalSize::new(0, 0),
        }
    }

    /// Yüzey boyutu değiştiyse köşe koordinatlarını (NDC) günceller.
    fn update_vertices_if_needed(&mut self, queue: &wgpu::Queue, sw: u32, sh: u32) {
        if self.surface_size.width == sw && self.surface_size.height == sh {
            return;
        }
        self.surface_size = PhysicalSize::new(sw, sh);
        let (sw, sh) = (sw as f32, sh as f32);
        let (qw, qh) = (HUD_PX_W as f32, HUD_PX_H as f32);
        let ox = 8.0_f32; // ekranda X başlangıcı (piksel)
        let oy = 8.0_f32; // ekranda Y başlangıcı (piksel)

        // Ekran koordinatlarını NDC'ye çevirir (Y aşağıya doğru → NDC'de ters).
        let nx = |x: f32| x / sw * 2.0 - 1.0;
        let ny = |y: f32| 1.0 - y / sh * 2.0;

        // Tek quad, 2 üçgen (6 köşe): [x, y, u, v]
        let verts: [[f32; 4]; 6] = [
            [nx(ox), ny(oy), 0.0, 0.0],
            [nx(ox + qw), ny(oy), 1.0, 0.0],
            [nx(ox), ny(oy + qh), 0.0, 1.0],
            [nx(ox + qw), ny(oy), 1.0, 0.0],
            [nx(ox + qw), ny(oy + qh), 1.0, 1.0],
            [nx(ox), ny(oy + qh), 0.0, 1.0],
        ];
        queue.write_buffer(&self.vertex_buf, 0, bytemuck_cast_slice(&verts));
    }

    /// CPU tamponunu `snapshot` verisiyle yeniden çizer ve GPU tekstürüne yükler.
    fn update(&mut self, queue: &wgpu::Queue, snapshot: &HudSnapshot, sw: u32, sh: u32) {
        self.update_vertices_if_needed(queue, sw, sh);
        self.render_to_pixels(snapshot);
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &self.pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(HUD_PX_W * 4),
                rows_per_image: Some(HUD_PX_H),
            },
            wgpu::Extent3d {
                width: HUD_PX_W,
                height: HUD_PX_H,
                depth_or_array_layers: 1,
            },
        );
    }

    /// HUD quad'ını yüzey tekstürü üzerine alfa-karıştırmalı render eder.
    ///
    /// `LoadOp::Load` kullanılır; clear-pass çıktısı korunur.
    fn draw(&self, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("hud-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, self.vertex_buf.slice(..));
        pass.draw(0..6, 0..1);
    }

    /// `self.pixels` tamponunu HUD içeriğiyle doldurur.
    fn render_to_pixels(&mut self, s: &HudSnapshot) {
        // Tampon sıfırla (şeffaf siyah).
        for b in self.pixels.iter_mut() {
            *b = 0;
        }

        let w = HUD_PX_W as usize;
        let buf = &mut self.pixels;

        // ── Arka plan: yarı şeffaf koyu panel ─────────────────────────────────
        fill_rect(
            buf,
            w,
            0,
            0,
            HUD_PX_W as usize,
            HUD_PX_H as usize,
            [10, 12, 20, 200],
        );

        // ── Faz etiketi ─────────────────────────────────────────────────────────
        let phase_color: [u8; 4] = match s.phase {
            "WARM" => [255, 170, 50, 255],
            "SMPL" => [60, 210, 60, 255],
            _ => [100, 180, 255, 255],
        };
        Font::draw_str(buf, w, 6, 4, s.phase, 1, phase_color);
        // Frame sayısı sağda
        let frame_str = format!("{} F", s.frames);
        let chars = frame_str.chars().count();
        let x_right = HUD_PX_W as usize - chars * 7 - 4;
        Font::draw_str(buf, w, x_right, 4, &frame_str, 1, [100, 100, 120, 255]);

        // ── Büyük FPS sayısı (ölçek 3×) ────────────────────────────────────────
        let fps_str = format!("{:.1}", s.fps);
        let fps_color: [u8; 4] = if s.fps >= 60.0 {
            [60, 225, 60, 255]
        } else if s.fps >= 30.0 {
            [255, 200, 50, 255]
        } else {
            [230, 60, 60, 255]
        };
        Font::draw_str(buf, w, 6, 14, &fps_str, 3, fps_color);

        // "FPS" veya "WFPS" etiketi, büyük sayının sağına
        let fps_label = if s.work_units > 1 { "WFPS" } else { "FPS" };
        let fps_right_x = 6 + fps_str.chars().count() * (GLYPH_W + 1) * 3 + 4;
        Font::draw_str(buf, w, fps_right_x, 26, fps_label, 2, [160, 160, 180, 255]);
        if s.work_units > 1 {
            let burst_str = format!("x{}", s.work_units);
            Font::draw_str(buf, w, fps_right_x, 40, &burst_str, 1, [120, 120, 140, 255]);
        }

        // ── Ayırıcı çizgi ──────────────────────────────────────────────────────
        fill_rect(buf, w, 4, 50, HUD_PX_W as usize - 8, 1, [60, 60, 80, 200]);

        // ── İstatistik satırları (ölçek 1×) ────────────────────────────────────
        let y0 = 54_usize;
        let line_h = 10_usize;
        let stats: &[(&str, String)] = &[
            ("AVG", format!("{:.1} FPS", s.avg_fps)),
            ("PEA", format!("{:.1} FPS", s.peak_fps)),
            ("AVG", format!("{:.2}ms", s.avg_ms)),
            ("p95", format!("{:.2}ms", s.p95_ms)),
            ("p99", format!("{:.2}ms", s.p99_ms)),
            ("σ  ", format!("{:.2}ms", s.stddev_ms)),
        ];
        for (i, (label, val)) in stats.iter().enumerate() {
            let y = y0 + i * line_h;
            Font::draw_str(buf, w, 6, y, label, 1, [120, 120, 140, 255]);
            Font::draw_str(buf, w, 30, y, val, 1, [220, 220, 235, 255]);
        }

        // ── Ayırıcı çizgi ──────────────────────────────────────────────────────
        let chart_y = y0 + stats.len() * line_h + 4;
        fill_rect(
            buf,
            w,
            4,
            chart_y - 2,
            HUD_PX_W as usize - 8,
            1,
            [60, 60, 80, 200],
        );

        // ── Frame-time çubuk grafiği ────────────────────────────────────────────
        draw_frame_chart(buf, w, 4, chart_y, &s.recent_ms);

        // ── Alt bilgi: GPU adı + GMS skoru ─────────────────────────────────────
        let info_y = HUD_PX_H as usize - 20;
        fill_rect(
            buf,
            w,
            4,
            info_y - 2,
            HUD_PX_W as usize - 8,
            1,
            [60, 60, 80, 200],
        );
        let gpu_str = truncate(&s.gpu_name, 36);
        Font::draw_str(buf, w, 6, info_y + 1, &gpu_str, 1, [140, 140, 160, 255]);
        let score_str = format!("GMS {}  PM:{}", s.gms_score, s.present_mode_label);
        Font::draw_str(buf, w, 6, info_y + 11, &score_str, 1, [200, 200, 80, 255]);
    }
}

/// GPU'da `HUD_PX_W × HUD_PX_H` Rgba8Unorm tekstürü oluşturur.
fn create_hud_texture(device: &wgpu::Device) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("hud-texture"),
        size: wgpu::Extent3d {
            width: HUD_PX_W,
            height: HUD_PX_H,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

/// RGBA tamponunda dikdörtgen doldurur.
fn fill_rect(buf: &mut [u8], bw: usize, x: usize, y: usize, w: usize, h: usize, color: [u8; 4]) {
    for row in 0..h {
        for col in 0..w {
            let base = ((y + row) * bw + (x + col)) * 4;
            if base + 3 < buf.len() {
                buf[base..base + 4].copy_from_slice(&color);
            }
        }
    }
}

/// Son ≤120 frame-time değerini renkli çubuk grafik olarak çizer.
///
/// Yeşil ≤16.67 ms, sarı ≤33.3 ms, kırmızı >33.3 ms.
/// Yatay beyaz çizgi 16.67 ms (60 FPS) hedefini gösterir.
fn draw_frame_chart(buf: &mut [u8], bw: usize, ox: usize, oy: usize, times: &[f32]) {
    const TARGET_MS: f32 = 16.67;
    const CHART_H: usize = 32;
    const BAR_W: usize = 2;

    // Arka plan
    fill_rect(
        buf,
        bw,
        ox,
        oy,
        HUD_PX_W as usize - ox * 2,
        CHART_H,
        [0, 0, 0, 80],
    );

    let max_ms = times.iter().cloned().fold(TARGET_MS * 1.5, f32::max);

    for (i, &ft) in times.iter().enumerate() {
        let bar_x = ox + i * BAR_W;
        if bar_x + BAR_W > HUD_PX_W as usize {
            break;
        }
        let h = ((ft / max_ms) * CHART_H as f32).clamp(1.0, CHART_H as f32) as usize;
        let color: [u8; 4] = if ft > TARGET_MS * 2.0 {
            [210, 55, 55, 200]
        } else if ft > TARGET_MS {
            [250, 185, 40, 200]
        } else {
            [50, 195, 60, 200]
        };
        fill_rect(
            buf,
            bw,
            bar_x,
            oy + CHART_H - h,
            BAR_W.saturating_sub(0),
            h,
            color,
        );
    }

    // 60 FPS hedef çizgisi
    let target_y =
        oy + CHART_H - ((TARGET_MS / max_ms) * CHART_H as f32).min(CHART_H as f32) as usize;
    fill_rect(
        buf,
        bw,
        ox,
        target_y,
        HUD_PX_W as usize - ox * 2,
        1,
        [200, 200, 200, 80],
    );
}

// ── Güvenli tip dönüşümü (bytemuck yerine kendi implementasyonumuz) ───────────

/// `[[f32; 4]; 6]` dizisini `&[u8]`'e güvenli şekilde dönüştürür.
///
/// `bytemuck` bağımlılığı olmadan, aynı bellek düzenini manüel olarak kullanır.
fn bytemuck_cast_slice(data: &[[f32; 4]; 6]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(
            data.as_ptr() as *const u8,
            data.len() * 4 * std::mem::size_of::<f32>(),
        )
    }
}

fn truncate(s: &str, max: usize) -> String {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= max {
        s.to_owned()
    } else {
        chars[..max - 1].iter().collect::<String>() + ">"
    }
}

// ── HUD WGSL shader ───────────────────────────────────────────────────────────

const HUD_WGSL: &str = r#"
// HUD overlay shader: tekstürlenmiş quad, alfa karıştırma etkin.
//
// Vertex: ekran-piksel koordinatları yerine doğrudan NDC alır (CPU tarafında hesaplanmış).
// Fragment: HUD tekstürünü örnekler; alfa=0 pikseller saydam kalır.

struct VertexOut {
    @builtin(position) pos: vec4<f32>,
    @location(0)        uv:  vec2<f32>,
}

@vertex
fn vs_main(
    @location(0) pos: vec2<f32>,
    @location(1) uv:  vec2<f32>,
) -> VertexOut {
    var out: VertexOut;
    out.pos = vec4<f32>(pos, 0.0, 1.0);
    out.uv  = uv;
    return out;
}

@group(0) @binding(0) var hud_tex: texture_2d<f32>;
@group(0) @binding(1) var hud_smp: sampler;

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    return textureSample(hud_tex, hud_smp, in.uv);
}
"#;

// ── ApplicationHandler ────────────────────────────────────────────────────────

/// winit uygulama yöneticisi.
struct VisualBenchmarkApp {
    options: CliOptions,
    runtime: Option<VisualRuntime>,
    exit_requested: bool,
    summary_printed: bool,
}

impl VisualBenchmarkApp {
    fn new(options: CliOptions) -> Self {
        Self {
            options,
            runtime: None,
            exit_requested: false,
            summary_printed: false,
        }
    }
}

impl ApplicationHandler for VisualBenchmarkApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.runtime.is_some() {
            return;
        }
        match VisualRuntime::new(event_loop, self.options) {
            Ok(rt) => {
                let cf = rt.preferred_control_flow();
                self.runtime = Some(rt);
                event_loop.set_control_flow(cf);
            }
            Err(e) => {
                eprintln!("Başlatılamadı: {e}");
                event_loop.exit();
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(rt) = self.runtime.as_mut() else {
            return;
        };
        if rt.renderer.window.id() != window_id {
            return;
        }

        match event {
            WindowEvent::CloseRequested => {
                self.exit_requested = true;
                event_loop.exit();
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key,
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => {
                if matches!(logical_key.as_ref(), Key::Named(NamedKey::Escape)) {
                    self.exit_requested = true;
                    event_loop.exit();
                }
            }
            WindowEvent::Resized(size) => {
                rt.resize(size);
            }
            WindowEvent::RedrawRequested => {
                if let Err(outcome) = rt.render_frame() {
                    match outcome {
                        RenderOutcome::SurfaceLost | RenderOutcome::Outdated => {
                            rt.renderer.reconfigure();
                        }
                        RenderOutcome::OutOfMemory => {
                            eprintln!("Bellek yetersiz");
                            event_loop.exit();
                        }
                        RenderOutcome::Timeout | RenderOutcome::Other => {}
                    }
                }
                if rt.use_redraw_chaining() {
                    rt.renderer.window.request_redraw();
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if let Some(rt) = self.runtime.as_ref() {
            event_loop.set_control_flow(rt.preferred_control_flow());
            if rt.use_redraw_chaining() {
                if rt.stats.frames == 0 {
                    rt.renderer.window.request_redraw();
                }
            } else {
                rt.renderer.window.request_redraw();
            }
            if rt.should_auto_exit() {
                self.exit_requested = true;
                event_loop.exit();
            }
        } else {
            event_loop.set_control_flow(ControlFlow::Wait);
        }
        if self.exit_requested {
            event_loop.exit();
        }
    }

    fn exiting(&mut self, _: &ActiveEventLoop) {
        if self.summary_printed {
            return;
        }
        self.summary_printed = true;
        if let Some(rt) = self.runtime.take() {
            let summary = rt.finish();
            print_summary(&summary);
        }
    }
}

// ── VisualRuntime ─────────────────────────────────────────────────────────────

/// Benchmark oturumunun tüm çalışma durumu.
struct VisualRuntime {
    renderer: Renderer,
    multi_gpu: Option<MultiGpuExecutor>,
    stats: FrameStats,
    adapter_profile: Option<gms::GpuAdapterProfile>,
    gms_inventory: GpuInventory,
    pacing_mode: BenchmarkPacingMode,
    options: CliOptions,
    session_start: Instant,
    sample_started_at: Option<Instant>,
    sample_finished: bool,
}

impl VisualRuntime {
    fn new(event_loop: &ActiveEventLoop, options: CliOptions) -> Result<Self, Box<dyn Error>> {
        let renderer = Renderer::new(event_loop, options)?;
        let gms_inventory = GpuInventory::discover();
        let adapter_profile = match_inventory_profile(&gms_inventory, &renderer.adapter_info);
        let pacing_mode =
            choose_pacing_mode(adapter_profile.as_ref(), &renderer.adapter_info, options);

        let multi_gpu = if matches!(options.multi_gpu_override, MultiGpuOverride::Off) {
            None
        } else {
            let workload = build_multi_gpu_workload_request(&renderer, options);
            let policy = match options.multi_gpu_override {
                MultiGpuOverride::Auto => MultiGpuInitPolicy::Auto,
                MultiGpuOverride::On => MultiGpuInitPolicy::Force,
                MultiGpuOverride::Off => unreachable!(),
            };
            MultiGpuExecutor::try_new(MultiGpuExecutorConfig {
                policy,
                primary_adapter_info: renderer.adapter_info.clone(),
                inventory: gms_inventory.clone(),
                primary_device: renderer.device.clone(),
                frame_width: renderer.size.width,
                frame_height: renderer.size.height,
                secondary_offscreen_format: TextureFormat::Rgba8Unorm,
                primary_work_units_per_present: renderer.work_units_per_present(),
                workload_request: workload,
                auto_min_projected_gain_pct: 5.0,
            })?
        };

        let timing_cap = renderer
            .runtime_tuning
            .benchmark_timing_capacity
            .max(16_384);
        let title_interval = Duration::from_millis(
            renderer
                .runtime_tuning
                .benchmark_title_update_interval_ms
                .max(100),
        );
        let mut stats = FrameStats::new(renderer.size, timing_cap, title_interval);
        stats.last_title_update = Instant::now();
        let session_start = Instant::now();
        let sample_started_at = if options.warmup_duration.is_zero() {
            Some(session_start)
        } else {
            None
        };

        Ok(Self {
            renderer,
            multi_gpu,
            stats,
            adapter_profile,
            gms_inventory,
            pacing_mode,
            options,
            session_start,
            sample_started_at,
            sample_finished: false,
        })
    }

    fn resize(&mut self, size: PhysicalSize<u32>) {
        self.renderer.resize(size);
        if let Some(mg) = self.multi_gpu.as_mut() {
            mg.resize(size.width, size.height);
        }
        self.stats.resolution = self.renderer.size;
    }

    fn render_frame(&mut self) -> Result<(), RenderOutcome> {
        self.tick_phase(Instant::now());
        let snapshot = self.build_snapshot();

        let frame_start = Instant::now();
        let primary_wu = self.renderer.render(snapshot)?;
        let secondary_wu = self
            .multi_gpu
            .as_mut()
            .map(|mg| mg.submit_frame())
            .unwrap_or(0);
        let work_units = primary_wu.saturating_add(secondary_wu);

        let frame_end = Instant::now();
        let timing = self.stats.record_frame(frame_start, frame_end, work_units);
        self.tick_phase_after(frame_end);

        if timing.should_update_title {
            self.update_title(timing.display_fps, timing.avg_fps);
        }
        Ok(())
    }

    fn build_snapshot(&self) -> HudSnapshot {
        let avg_ms = avg_f64(&self.stats.frame_intervals_ms).unwrap_or(0.0);
        let p95_ms = percentile_ms(&self.stats.frame_intervals_ms, 95.0).unwrap_or(avg_ms);
        let p99_ms = percentile_ms(&self.stats.frame_intervals_ms, 99.0).unwrap_or(p95_ms);
        let stddev_ms = stddev(&self.stats.frame_intervals_ms).unwrap_or(0.0);

        let wu = self.renderer.work_units_per_present();
        let sec_wu = self
            .multi_gpu
            .as_ref()
            .map(|mg| mg.secondary_work_units_per_present())
            .unwrap_or(0);

        let gpu_name = self
            .adapter_profile
            .as_ref()
            .map(|p| truncate(&p.name, 38))
            .unwrap_or_else(|| truncate(&self.renderer.adapter_info.name, 38));

        HudSnapshot {
            fps: self.stats.smoothed_fps_recent(30).unwrap_or(0.0),
            avg_fps: self.stats.avg_fps(),
            peak_fps: self.stats.peak_fps,
            avg_ms,
            p95_ms,
            p99_ms,
            stddev_ms,
            frames: self.stats.frames,
            phase: self.phase_label(),
            gpu_name,
            gms_score: self.adapter_profile.as_ref().map(|p| p.score).unwrap_or(0),
            work_units: wu.saturating_add(sec_wu),
            present_mode_label: present_mode_label(self.renderer.present_mode),
            recent_ms: self.stats.recent_frame_ms.iter().copied().collect(),
        }
    }

    fn update_title(&mut self, instant_fps: f64, avg_fps: f64) {
        let score = self.adapter_profile.as_ref().map(|p| p.score).unwrap_or(0);
        self.renderer.window.set_title(&format!(
            "GMS Grafik Benchmark [{}] | {:.1} FPS (ort {:.1}) | Score {} | Esc",
            self.phase_label(),
            instant_fps,
            avg_fps,
            score
        ));
    }

    fn finish(self) -> BenchmarkSummary {
        let computed = self.stats.compute_summary();
        let gms_score = self.adapter_profile.as_ref().map(|p| p.score).unwrap_or(0);
        let adapter_name = self
            .adapter_profile
            .as_ref()
            .map(|p| p.name.clone())
            .unwrap_or_else(|| self.renderer.adapter_info.name.clone());

        let score = compute_render_benchmark_score(
            &computed,
            gms_score,
            self.pacing_mode,
            &self.renderer.runtime_tuning,
        );
        let (est_cu, cu_short, cu_disp) = self
            .adapter_profile
            .as_ref()
            .map(|p| {
                let (sh, cnt) = p.compute_unit_summary();
                (
                    Some(cnt),
                    Some(sh),
                    Some(p.compute_unit_kind.display_label()),
                )
            })
            .unwrap_or((None, None, None));

        BenchmarkSummary {
            adapter_name,
            backend: format!("{:?}", self.renderer.adapter_info.backend),
            device_type: format!("{:?}", self.renderer.adapter_info.device_type),
            estimated_compute_units: est_cu,
            compute_unit_short_label: cu_short,
            compute_unit_display_label: cu_disp,
            compute_unit_source: self
                .adapter_profile
                .as_ref()
                .map(|p| p.compute_unit_source.short_label()),
            resolution: computed.resolution,
            present_mode: self.renderer.present_mode,
            total_frames: computed.total_frames,
            elapsed: computed.elapsed,
            avg_fps: computed.avg_fps,
            peak_fps: computed.peak_fps,
            avg_frame_ms: computed.avg_frame_ms,
            p95_frame_ms: computed.p95_frame_ms,
            p99_frame_ms: computed.p99_frame_ms,
            low_1_percent_fps: computed.low_1_percent_fps,
            frame_time_stddev_ms: computed.frame_time_stddev_ms,
            avg_work_ms: computed.avg_work_ms,
            p95_work_ms: computed.p95_work_ms,
            p99_work_ms: computed.p99_work_ms,
            work_time_stddev_ms: computed.work_time_stddev_ms,
            gms_hardware_score: gms_score,
            total_discovered_gpus: self.gms_inventory.adapters.len(),
            render_score: score.score,
            render_tier: score.tier,
            score_breakdown: score,
            work_units_per_present: self.renderer.work_units_per_present(),
            multi_gpu: self.multi_gpu.as_ref().map(|mg| mg.summary()),
            pacing_mode: self.pacing_mode,
            warmup_duration: self.options.warmup_duration,
            sample_duration: self.options.sample_duration,
            mode_override: self.options.mode_override,
            vsync_override: self.options.vsync_override,
            multi_gpu_override: self.options.multi_gpu_override,
        }
    }

    fn tick_phase(&mut self, now: Instant) {
        if self.sample_finished || self.sample_started_at.is_some() {
            return;
        }
        if now.duration_since(self.session_start) >= self.options.warmup_duration {
            self.sample_started_at = Some(now);
            self.stats.reset_measurement(now);
        }
    }
    fn tick_phase_after(&mut self, now: Instant) {
        if self.sample_finished {
            return;
        }
        if let Some(s) = self.sample_started_at {
            if now.duration_since(s) >= self.options.sample_duration {
                self.sample_finished = true;
            }
        }
    }

    /// Kısa faz etiketi (4 karakter — bitmap font dostu).
    fn phase_label(&self) -> &'static str {
        if self.sample_finished {
            "DONE"
        } else if self.sample_started_at.is_some() {
            "SMPL"
        } else {
            "WARM"
        }
    }

    fn preferred_control_flow(&self) -> ControlFlow {
        match self.pacing_mode {
            BenchmarkPacingMode::Stable => ControlFlow::Wait,
            BenchmarkPacingMode::MaxThroughput => ControlFlow::Poll,
        }
    }
    fn use_redraw_chaining(&self) -> bool {
        matches!(self.pacing_mode, BenchmarkPacingMode::Stable)
    }
    fn should_auto_exit(&self) -> bool {
        self.sample_finished
    }
}

// ── Renderer ──────────────────────────────────────────────────────────────────

/// wgpu render altyapısı + entegre HUD renderer.
struct Renderer {
    window: Arc<Window>,
    _instance: wgpu::Instance,
    surface: wgpu::Surface<'static>,
    adapter_info: wgpu::AdapterInfo,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: PhysicalSize<u32>,
    present_mode: PresentMode,
    clear_phase: f64,
    throughput_burst: Option<ThroughputBurst>,
    throughput_targets: Vec<wgpu::Texture>,
    throughput_target_cursor: usize,
    runtime_tuning: GmsRuntimeTuningProfile,
    presented_frames: u64,
    /// Ekran üstü HUD katmanı (bitmap font + textured quad pipeline).
    hud: HudRenderer,
}

#[derive(Debug, Clone, Copy)]
struct ThroughputBurst {
    work_units_per_present: u32,
    offscreen_target_ring_len: usize,
}

impl Renderer {
    fn new(event_loop: &ActiveEventLoop, options: CliOptions) -> Result<Self, Box<dyn Error>> {
        let window = Arc::new(
            event_loop.create_window(
                Window::default_attributes()
                    .with_title("GMS Grafik Benchmark | Başlatılıyor...")
                    .with_inner_size(options.resolution),
            )?,
        );

        let size = window.inner_size();
        let instance = wgpu::Instance::default();
        let surface = instance.create_surface(Arc::clone(&window))?;
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))?;

        let adapter_info = adapter.get_info();
        let runtime_tuning = GmsRuntimeTuningProfile::from_adapter_info(&adapter_info);
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("gms-bench-visual-dev"),
                ..Default::default()
            }))?;

        let capabilities = surface.get_capabilities(&adapter);
        let mut config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .ok_or_else(|| SE("yüzey uyumsuz".into()))?;

        if let Some(srgb) = capabilities
            .formats
            .iter()
            .copied()
            .find(TextureFormat::is_srgb)
        {
            config.format = srgb;
        }
        // Benchmark varsayılanı: vsync kapalı (max throughput). Sadece --vsync on açar.
        let prefer_stable = matches!(options.vsync_override, VsyncOverride::On);
        config.present_mode = select_present_mode(
            &capabilities.present_modes,
            prefer_stable,
            options.vsync_override,
        );
        config.alpha_mode = capabilities
            .alpha_modes
            .iter()
            .copied()
            .find(|m| *m == CompositeAlphaMode::Opaque)
            .unwrap_or(config.alpha_mode);
        config.desired_maximum_frame_latency = runtime_tuning.recommended_surface_frame_latency;
        surface.configure(&device, &config);

        let throughput_burst =
            select_throughput_burst(&adapter_info, &runtime_tuning, options, config.present_mode);
        // HUD renderer, yüzey formatı belirlendikten sonra oluşturulur.
        let hud = HudRenderer::new(&device, &queue, config.format);

        let mut renderer = Self {
            window,
            _instance: instance,
            surface,
            adapter_info,
            device,
            queue,
            config: config.clone(),
            size,
            present_mode: config.present_mode,
            clear_phase: 0.0,
            throughput_burst,
            throughput_targets: Vec::new(),
            throughput_target_cursor: 0,
            runtime_tuning,
            presented_frames: 0,
            hud,
        };
        renderer.prewarm_throughput_resources();
        Ok(renderer)
    }

    fn resize(&mut self, size: PhysicalSize<u32>) {
        if size.width == 0 || size.height == 0 {
            self.size = size;
            self.throughput_targets.clear();
            return;
        }
        self.size = size;
        self.config.width = size.width;
        self.config.height = size.height;
        self.throughput_targets.clear();
        self.throughput_target_cursor = 0;
        self.presented_frames = 0;
        self.surface.configure(&self.device, &self.config);
    }

    fn reconfigure(&mut self) {
        if self.size.width == 0 || self.size.height == 0 {
            return;
        }
        self.surface.configure(&self.device, &self.config);
    }

    /// Tek frame: throughput burst + clear pass + HUD overlay.
    fn render(&mut self, snapshot: HudSnapshot) -> Result<u32, RenderOutcome> {
        if self.size.width == 0 || self.size.height == 0 {
            return Ok(1);
        }

        let frame = self
            .surface
            .get_current_texture()
            .map_err(RenderOutcome::from)?;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        self.clear_phase = (self.clear_phase + 0.0125) % std::f64::consts::TAU;
        let clear_color = animated_clear_color(self.clear_phase);
        let work_units = self.effective_work_units_per_present();

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gms-bench-visual-enc"),
            });

        // Offscreen burst pass'lar
        if work_units > 1 {
            self.ensure_throughput_target_ring();
            if let Some(target) = self.current_throughput_target() {
                let tv = target.create_view(&wgpu::TextureViewDescriptor::default());
                for _ in 0..(work_units - 1) {
                    self.clear_phase = (self.clear_phase + 0.0125) % std::f64::consts::TAU;
                    encode_clear_pass(&mut encoder, &tv, animated_clear_color(self.clear_phase));
                }
            }
        }

        // Ana yüzey clear pass
        encode_clear_pass(&mut encoder, &view, clear_color);

        // HUD CPU render + GPU yükleme + kompozitleme
        self.hud
            .update(&self.queue, &snapshot, self.size.width, self.size.height);
        self.hud.draw(&mut encoder, &view);

        self.queue.submit(Some(encoder.finish()));
        self.window.pre_present_notify();
        frame.present();
        self.presented_frames = self.presented_frames.saturating_add(1);
        Ok(work_units)
    }

    fn work_units_per_present(&self) -> u32 {
        self.throughput_burst
            .map(|b| b.work_units_per_present)
            .unwrap_or(1)
    }
    fn effective_work_units_per_present(&self) -> u32 {
        self.runtime_tuning
            .effective_throughput_work_units_per_present(
                self.work_units_per_present(),
                self.presented_frames,
            )
    }
    fn current_throughput_target(&mut self) -> Option<&wgpu::Texture> {
        if self.throughput_targets.is_empty() {
            return None;
        }
        let idx = self.throughput_target_cursor % self.throughput_targets.len();
        self.throughput_target_cursor =
            (self.throughput_target_cursor + 1) % self.throughput_targets.len();
        self.throughput_targets.get(idx)
    }
    fn ensure_throughput_target_ring(&mut self) {
        if self.size.width == 0 || self.size.height == 0 {
            self.throughput_targets.clear();
            return;
        }
        let desired = self
            .throughput_burst
            .map(|b| b.offscreen_target_ring_len.max(1))
            .unwrap_or(1);
        let needs = self.throughput_targets.len() != desired
            || self.throughput_targets.iter().any(|t| {
                let e = t.size();
                e.width != self.size.width || e.height != self.size.height
            });
        if !needs {
            return;
        }
        self.throughput_targets.clear();
        for _ in 0..desired {
            self.throughput_targets
                .push(self.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("gms-bench-visual-throughput"),
                    size: wgpu::Extent3d {
                        width: self.size.width,
                        height: self.size.height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: self.config.format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[],
                }));
        }
        self.throughput_target_cursor = 0;
    }
    fn prewarm_throughput_resources(&mut self) {
        if self.work_units_per_present() <= 1
            || self.runtime_tuning.throughput_startup_prewarm_submits == 0
        {
            return;
        }
        self.ensure_throughput_target_ring();
        if self.throughput_targets.is_empty() {
            return;
        }
        let count = self.runtime_tuning.startup_prewarm_submits_for_ring(
            self.work_units_per_present(),
            self.throughput_targets.len(),
        );
        for _ in 0..count {
            let Some(target) = self.current_throughput_target() else {
                break;
            };
            let tv = target.create_view(&wgpu::TextureViewDescriptor::default());
            let mut enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("gms-bench-visual-prewarm"),
                });
            self.clear_phase = (self.clear_phase + 0.00625) % std::f64::consts::TAU;
            encode_clear_pass(&mut enc, &tv, animated_clear_color(self.clear_phase));
            self.queue.submit(Some(enc.finish()));
        }
        let _ = self.device.poll(wgpu::PollType::Poll);
    }
}

// ── FrameStats ────────────────────────────────────────────────────────────────

/// Frame zamanlama istatistikleri.
///
/// Hem canlı HUD hem de oturum sonu özet için kullanılır.
/// `recent_frame_ms` rolling buffer, son 120 frame-time'ı f32 olarak tutar.
struct FrameStats {
    benchmark_start: Instant,
    last_frame_presented_at: Option<Instant>,
    last_title_update: Instant,
    title_update_interval: Duration,
    frames: u64,
    frame_intervals_ms: Vec<f64>,
    render_durations_ms: Vec<f64>,
    /// Rolling buffer — HUD çubuk grafiği için son 120 frame-time (ms).
    recent_frame_ms: VecDeque<f32>,
    peak_fps: f64,
    resolution: PhysicalSize<u32>,
}

struct FrameTiming {
    display_fps: f64,
    avg_fps: f64,
    should_update_title: bool,
}

impl FrameStats {
    fn new(
        resolution: PhysicalSize<u32>,
        timing_capacity: usize,
        title_update_interval: Duration,
    ) -> Self {
        let now = Instant::now();
        Self {
            benchmark_start: now,
            last_frame_presented_at: None,
            last_title_update: now,
            title_update_interval,
            frames: 0,
            frame_intervals_ms: Vec::with_capacity(timing_capacity),
            render_durations_ms: Vec::with_capacity(timing_capacity),
            recent_frame_ms: VecDeque::with_capacity(120),
            peak_fps: 0.0,
            resolution,
        }
    }
    fn reset_measurement(&mut self, now: Instant) {
        self.benchmark_start = now;
        self.last_frame_presented_at = None;
        self.last_title_update = now;
        self.frames = 0;
        self.frame_intervals_ms.clear();
        self.render_durations_ms.clear();
        self.recent_frame_ms.clear();
        self.peak_fps = 0.0;
    }
    fn record_frame(
        &mut self,
        frame_start: Instant,
        frame_end: Instant,
        work_units: u32,
    ) -> FrameTiming {
        let wu = work_units.max(1) as u64;
        self.frames = self.frames.saturating_add(wu);
        let render_ms = (frame_end - frame_start).as_secs_f64() * 1000.0;
        self.render_durations_ms.push(render_ms / wu as f64);

        let mut instant_fps = 0.0;
        if let Some(last) = self.last_frame_presented_at {
            let interval_ms = (frame_end - last).as_secs_f64() * 1000.0;
            if interval_ms.is_finite() && interval_ms > 0.0 {
                let norm_ms = interval_ms / wu as f64;
                self.frame_intervals_ms.push(norm_ms);
                instant_fps = 1000.0 / norm_ms;
                self.peak_fps = self.peak_fps.max(instant_fps);
                if self.recent_frame_ms.len() >= 120 {
                    self.recent_frame_ms.pop_front();
                }
                self.recent_frame_ms.push_back(norm_ms as f32);
            }
        }
        self.last_frame_presented_at = Some(frame_end);
        let display_fps = self.smoothed_fps_recent(30).unwrap_or(instant_fps);
        let avg_fps = self.avg_fps();
        let should_update_title =
            frame_end.duration_since(self.last_title_update) >= self.title_update_interval;
        if should_update_title {
            self.last_title_update = frame_end;
        }
        FrameTiming {
            display_fps,
            avg_fps,
            should_update_title,
        }
    }
    fn avg_fps(&self) -> f64 {
        self.frames as f64 / self.benchmark_start.elapsed().as_secs_f64().max(1e-9)
    }
    fn smoothed_fps_recent(&self, n: usize) -> Option<f64> {
        if n == 0 || self.frame_intervals_ms.is_empty() {
            return None;
        }
        let len = self.frame_intervals_ms.len();
        let slice = &self.frame_intervals_ms[len.saturating_sub(n)..];
        let avg = slice.iter().sum::<f64>() / slice.len() as f64;
        if avg > 0.0 {
            Some(1000.0 / avg)
        } else {
            None
        }
    }
    fn compute_summary(self) -> ComputedFrameStats {
        let elapsed = self.benchmark_start.elapsed();
        let avg_fps = self.frames as f64 / elapsed.as_secs_f64().max(1e-9);
        let avg_ms = avg_f64(&self.frame_intervals_ms).unwrap_or(0.0);
        let p95_ms = percentile_ms(&self.frame_intervals_ms, 95.0).unwrap_or(avg_ms);
        let p99_ms = percentile_ms(&self.frame_intervals_ms, 99.0).unwrap_or(p95_ms);
        let low1pct = if p99_ms > 0.0 { 1000.0 / p99_ms } else { 0.0 };
        let stddev = stddev(&self.frame_intervals_ms).unwrap_or(0.0);
        let avg_work = avg_f64(&self.render_durations_ms).unwrap_or(0.0);
        let p95_work = percentile_ms(&self.render_durations_ms, 95.0).unwrap_or(avg_work);
        let p99_work = percentile_ms(&self.render_durations_ms, 99.0).unwrap_or(p95_work);
        let wstddev = stddev_fn(&self.render_durations_ms).unwrap_or(0.0);
        ComputedFrameStats {
            total_frames: self.frames,
            elapsed,
            avg_fps,
            peak_fps: self.peak_fps,
            avg_frame_ms: avg_ms,
            p95_frame_ms: p95_ms,
            p99_frame_ms: p99_ms,
            low_1_percent_fps: low1pct,
            frame_time_stddev_ms: stddev,
            avg_work_ms: avg_work,
            p95_work_ms: p95_work,
            p99_work_ms: p99_work,
            work_time_stddev_ms: wstddev,
            resolution: self.resolution,
        }
    }
}

// ── İstatistik yardımcıları ───────────────────────────────────────────────────

fn avg_f64(v: &[f64]) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    Some(v.iter().sum::<f64>() / v.len() as f64)
}
fn stddev(v: &[f64]) -> Option<f64> {
    stddev_fn(v)
}
fn stddev_fn(v: &[f64]) -> Option<f64> {
    if v.len() < 2 {
        return None;
    }
    let m = avg_f64(v)?;
    let var = v
        .iter()
        .map(|x| {
            let d = x - m;
            d * d
        })
        .sum::<f64>()
        / v.len() as f64;
    Some(var.sqrt())
}
fn percentile_ms(v: &[f64], pct: f64) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((s.len() - 1) as f64 * pct.clamp(0.0, 100.0) / 100.0).round() as usize;
    s.get(idx).copied()
}
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t.clamp(0.0, 1.0)
}

// ── Render yardımcıları (render_benchmark ile aynı) ───────────────────────────

fn animated_clear_color(phase: f64) -> Color {
    Color {
        r: 0.15 + 0.25 * phase.sin().abs(),
        g: 0.12 + 0.28 * (phase * 1.37).sin().abs(),
        b: 0.18 + 0.30 * (phase * 0.73).cos().abs(),
        a: 1.0,
    }
}
fn encode_clear_pass(enc: &mut wgpu::CommandEncoder, view: &wgpu::TextureView, color: Color) {
    let _p = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("gms-bench-visual-clear"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(color),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
}
fn present_mode_label(pm: PresentMode) -> &'static str {
    match pm {
        PresentMode::Fifo => "FIFO",
        PresentMode::FifoRelaxed => "FREL",
        PresentMode::Mailbox => "MAIL",
        PresentMode::Immediate => "IMMD",
        PresentMode::AutoVsync => "AVSY",
        PresentMode::AutoNoVsync => "ANSV",
        _ => "????",
    }
}
fn select_present_mode(
    modes: &[PresentMode],
    prefer_stable: bool,
    vsync: VsyncOverride,
) -> PresentMode {
    if matches!(vsync, VsyncOverride::On) {
        for p in [
            PresentMode::AutoVsync,
            PresentMode::Fifo,
            PresentMode::Mailbox,
        ] {
            if modes.contains(&p) {
                return p;
            }
        }
    }
    if matches!(vsync, VsyncOverride::Off) {
        for p in [
            PresentMode::AutoNoVsync,
            PresentMode::Immediate,
            PresentMode::Mailbox,
        ] {
            if modes.contains(&p) {
                return p;
            }
        }
    }
    let stable = [
        PresentMode::AutoVsync,
        PresentMode::Fifo,
        PresentMode::Mailbox,
        PresentMode::AutoNoVsync,
        PresentMode::Immediate,
    ];
    let throughput = [
        PresentMode::AutoNoVsync,
        PresentMode::Immediate,
        PresentMode::Mailbox,
        PresentMode::AutoVsync,
        PresentMode::Fifo,
    ];
    for p in if prefer_stable { stable } else { throughput } {
        if modes.contains(&p) {
            return p;
        }
    }
    PresentMode::Fifo
}
fn select_throughput_burst(
    ai: &wgpu::AdapterInfo,
    rt: &GmsRuntimeTuningProfile,
    opts: CliOptions,
    pm: PresentMode,
) -> Option<ThroughputBurst> {
    let burst_ok = match opts.mode_override {
        ModeOverride::Stable => false,
        ModeOverride::Max => true,
        // macOS/Metal'de Immediate desteklenmez; platform gerçek vsync'i kıramıyorsa
        // throughput burst tek çaredir — CPU tipleri hariç her GPU'ya izin ver.
        ModeOverride::Auto => !matches!(ai.device_type, wgpu::DeviceType::Cpu),
    };
    if !burst_ok || matches!(opts.vsync_override, VsyncOverride::On) {
        return None;
    }
    // Fifo/AutoVsync seçildiyse (vsync kırılamadı) burst'ü etkinleştir.
    let vsync_locked = matches!(
        pm,
        PresentMode::Fifo | PresentMode::FifoRelaxed | PresentMode::AutoVsync
    );
    let wu = match ai.device_type {
        wgpu::DeviceType::DiscreteGpu => 64,
        wgpu::DeviceType::VirtualGpu => 16,
        // IntegratedGpu (Apple Silicon dahil) veya bilinmeyen: vsync kilitliyse daha az burst.
        _ => {
            if vsync_locked {
                rt.integrated_throughput_burst_work_units.max(8)
            } else {
                rt.integrated_throughput_burst_work_units
            }
        }
    };
    Some(ThroughputBurst {
        work_units_per_present: wu,
        offscreen_target_ring_len: rt.throughput_offscreen_target_ring_len,
    })
}
fn build_multi_gpu_workload_request(
    renderer: &Renderer,
    opts: CliOptions,
) -> MultiGpuWorkloadRequest {
    let px = renderer.size.width.max(1) as u64 * renderer.size.height.max(1) as u64;
    let ms = match opts.mode_override {
        ModeOverride::Stable => 16.67,
        ModeOverride::Max => 0.26,
        ModeOverride::Auto => {
            if renderer.work_units_per_present() > 1 {
                0.50
            } else {
                8.33
            }
        }
    };
    MultiGpuWorkloadRequest {
        sampled_processing_jobs: (px / 2048).clamp(128, 8192) as u32,
        object_updates: (px / 1024).clamp(256, 16384) as u32,
        physics_jobs: (px / 3072).clamp(64, 4096) as u32,
        ui_jobs: (px / 8192).clamp(16, 1024) as u32,
        post_fx_jobs: (px / 6144).clamp(32, 2048) as u32,
        bytes_per_sampled_job: 4096,
        bytes_per_object: 256,
        bytes_per_physics_job: 1024,
        bytes_per_ui_job: 512,
        bytes_per_post_fx_job: 1024,
        processed_texture_bytes_per_frame: px.saturating_mul(4).clamp(512 * 1024, 64 * 1024 * 1024),
        base_workgroup_size: 64,
        target_frame_budget_ms: ms,
    }
}
fn match_inventory_profile(
    inv: &GpuInventory,
    ai: &wgpu::AdapterInfo,
) -> Option<gms::GpuAdapterProfile> {
    inv.adapters
        .iter()
        .find(|p| {
            p.vendor_id == ai.vendor
                && p.device_id == ai.device
                && p.backend == ai.backend
                && p.name == ai.name
        })
        .or_else(|| {
            inv.adapters
                .iter()
                .find(|p| p.backend == ai.backend && p.vendor_id == ai.vendor && p.name == ai.name)
        })
        .cloned()
}

// ── Skor hesaplama ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
struct ComputedFrameStats {
    total_frames: u64,
    elapsed: Duration,
    avg_fps: f64,
    peak_fps: f64,
    avg_frame_ms: f64,
    p95_frame_ms: f64,
    p99_frame_ms: f64,
    low_1_percent_fps: f64,
    frame_time_stddev_ms: f64,
    avg_work_ms: f64,
    p95_work_ms: f64,
    p99_work_ms: f64,
    work_time_stddev_ms: f64,
    resolution: PhysicalSize<u32>,
}

/// Benchmark sonu performans skoru ve alt faktörler.
#[derive(Debug, Clone, Copy)]
struct RenderScore {
    score: u64,
    tier: &'static str,
    fps_term: f64,
    stability_factor: f64,
    tail_factor: f64,
    present_stability_factor: f64,
    present_tail_factor: f64,
    work_stability_factor: f64,
    work_tail_factor: f64,
    work_stability_blend: f64,
    resolution_factor: f64,
    gms_factor: f64,
}

fn compute_render_benchmark_score(
    s: &ComputedFrameStats,
    gms_hw: u64,
    mode: BenchmarkPacingMode,
    tuning: &GmsRuntimeTuningProfile,
) -> RenderScore {
    let px = s.resolution.width.max(1) as f64 * s.resolution.height.max(1) as f64;
    let res_f = (px / (1280.0 * 720.0)).sqrt().clamp(0.75, 3.0);
    let avg_ms = s.avg_frame_ms.max(0.001);
    let p95_ms = s.p95_frame_ms.max(avg_ms);
    let p99_ms = s.p99_frame_ms.max(p95_ms);
    let aw = s.avg_work_ms.max(0.001);
    let p95w = s.p95_work_ms.max(aw);
    let p99w = s.p99_work_ms.max(p95w);
    let ps = (avg_ms / p95_ms).clamp(0.35, 1.0);
    let pt = (avg_ms / p99_ms).clamp(0.25, 1.0);
    let ws = (aw / p95w).clamp(0.35, 1.0);
    let wt = (aw / p99w).clamp(0.25, 1.0);
    let blend = if matches!(mode, BenchmarkPacingMode::MaxThroughput) {
        tuning.throughput_work_stability_blend.clamp(0.0, 1.0)
    } else {
        0.0
    };
    let stab = lerp(ps, ws, blend);
    let tail = lerp(pt, wt, blend);
    let gms_f = (1.0 + (gms_hw as f64).ln_1p() / 12.0).clamp(1.0, 2.0);
    let fps_term = s.avg_fps.max(0.0) * 100.0;
    let score = (fps_term * res_f * stab * tail * gms_f).round().max(0.0) as u64;
    RenderScore {
        score,
        tier: score_tier(score),
        fps_term,
        stability_factor: stab,
        tail_factor: tail,
        present_stability_factor: ps,
        present_tail_factor: pt,
        work_stability_factor: ws,
        work_tail_factor: wt,
        work_stability_blend: blend,
        resolution_factor: res_f,
        gms_factor: gms_f,
    }
}
fn score_tier(s: u64) -> &'static str {
    match s {
        60_000.. => "S",
        35_000.. => "A",
        20_000.. => "B",
        10_000.. => "C",
        _ => "D",
    }
}

// ── Pacing modu ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
enum BenchmarkPacingMode {
    Stable,
    MaxThroughput,
}

fn choose_pacing_mode(
    profile: Option<&gms::GpuAdapterProfile>,
    ai: &wgpu::AdapterInfo,
    opts: CliOptions,
) -> BenchmarkPacingMode {
    match opts.mode_override {
        ModeOverride::Stable => return BenchmarkPacingMode::Stable,
        ModeOverride::Max => return BenchmarkPacingMode::MaxThroughput,
        ModeOverride::Auto => {}
    }
    if let Some(p) = profile {
        if matches!(p.memory_topology, gms::MemoryTopology::Unified) {
            return BenchmarkPacingMode::Stable;
        }
    }
    if matches!(ai.device_type, wgpu::DeviceType::IntegratedGpu) {
        BenchmarkPacingMode::Stable
    } else {
        BenchmarkPacingMode::MaxThroughput
    }
}

// ── Özet çıktısı ─────────────────────────────────────────────────────────────

struct BenchmarkSummary {
    adapter_name: String,
    backend: String,
    device_type: String,
    estimated_compute_units: Option<u32>,
    compute_unit_short_label: Option<&'static str>,
    compute_unit_display_label: Option<&'static str>,
    compute_unit_source: Option<&'static str>,
    resolution: PhysicalSize<u32>,
    present_mode: PresentMode,
    total_frames: u64,
    elapsed: Duration,
    avg_fps: f64,
    peak_fps: f64,
    avg_frame_ms: f64,
    p95_frame_ms: f64,
    p99_frame_ms: f64,
    low_1_percent_fps: f64,
    frame_time_stddev_ms: f64,
    avg_work_ms: f64,
    p95_work_ms: f64,
    p99_work_ms: f64,
    work_time_stddev_ms: f64,
    gms_hardware_score: u64,
    total_discovered_gpus: usize,
    render_score: u64,
    render_tier: &'static str,
    score_breakdown: RenderScore,
    work_units_per_present: u32,
    multi_gpu: Option<MultiGpuExecutorSummary>,
    pacing_mode: BenchmarkPacingMode,
    warmup_duration: Duration,
    sample_duration: Duration,
    mode_override: ModeOverride,
    vsync_override: VsyncOverride,
    multi_gpu_override: MultiGpuOverride,
}

fn print_summary(s: &BenchmarkSummary) {
    println!();
    println!("=== GMS Grafik Benchmark Özeti ===");
    println!(
        "Adapter: {} | Backend: {} | Tür: {}",
        s.adapter_name, s.backend, s.device_type
    );
    if let (Some(cu), Some(sl), Some(dl), Some(src)) = (
        s.estimated_compute_units,
        s.compute_unit_short_label,
        s.compute_unit_display_label,
        s.compute_unit_source,
    ) {
        println!("Tahmini {}: {} {} (kaynak: {})", dl, cu, sl, src);
    }
    println!(
        "Çözünürlük: {}x{} | Present: {:?} | Pacing: {:?}",
        s.resolution.width, s.resolution.height, s.present_mode, s.pacing_mode
    );
    if s.work_units_per_present > 1 {
        println!("Throughput burst: ×{} WU/present", s.work_units_per_present);
    }
    println!(
        "Warmup: {:.2}s | Örnek: {:.2}s | Frame: {} | Süre: {:.3}s",
        s.warmup_duration.as_secs_f64(),
        s.sample_duration.as_secs_f64(),
        s.total_frames,
        s.elapsed.as_secs_f64()
    );
    println!(
        "FPS: ort {:.2} | tepe {:.2} | 1% alt {:.2}",
        s.avg_fps, s.peak_fps, s.low_1_percent_fps
    );
    println!(
        "Frame-time(ms): ort {:.3} | p95 {:.3} | p99 {:.3} | σ {:.3}",
        s.avg_frame_ms, s.p95_frame_ms, s.p99_frame_ms, s.frame_time_stddev_ms
    );
    println!(
        "Work-time(ms):  ort {:.3} | p95 {:.3} | p99 {:.3} | σ {:.3}",
        s.avg_work_ms, s.p95_work_ms, s.p99_work_ms, s.work_time_stddev_ms
    );
    println!(
        "GMS HW skoru: {} | keşfedilen GPU: {}",
        s.gms_hardware_score, s.total_discovered_gpus
    );
    println!("Render skoru: {} [{}]", s.render_score, s.render_tier);
    println!(
        "Faktörler => fps {:.1} | rez {:.3} | kararlılık {:.3} | kuyruk {:.3} | gms {:.3}",
        s.score_breakdown.fps_term,
        s.score_breakdown.resolution_factor,
        s.score_breakdown.stability_factor,
        s.score_breakdown.tail_factor,
        s.score_breakdown.gms_factor
    );
    if let Some(mg) = &s.multi_gpu {
        println!(
            "Multi-GPU: {} → {} | WU/present: {} | toplam WU: {}",
            mg.primary_adapter_name,
            mg.secondary_adapter_name,
            mg.secondary_work_units_per_present,
            mg.total_secondary_work_units
        );
        println!(
            "Projeksiyon: tek {:.3}ms → çok {:.3}ms | beklenen kazanç: {:.2}%",
            mg.estimated_single_gpu_frame_ms,
            mg.estimated_multi_gpu_frame_ms,
            mg.projected_score_gain_pct
        );
        if mg.vulkan_version_gate_enabled {
            println!(
                "Vulkan gate: açık | primary {} | secondary {}",
                mg.primary_vulkan_api_version
                    .as_deref()
                    .unwrap_or("unknown"),
                mg.secondary_vulkan_api_version
                    .as_deref()
                    .unwrap_or("unknown")
            );
        }
    }
}

// ── Hata türü ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct SE(String);
impl fmt::Display for SE {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl Error for SE {}

#[derive(Debug, Clone, Copy)]
enum RenderOutcome {
    Timeout,
    Outdated,
    SurfaceLost,
    OutOfMemory,
    Other,
}
impl From<SurfaceError> for RenderOutcome {
    fn from(e: SurfaceError) -> Self {
        match e {
            SurfaceError::Timeout => Self::Timeout,
            SurfaceError::Outdated => Self::Outdated,
            SurfaceError::Lost => Self::SurfaceLost,
            SurfaceError::OutOfMemory => Self::OutOfMemory,
            SurfaceError::Other => Self::Other,
        }
    }
}
