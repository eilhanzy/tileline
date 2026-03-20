struct VSIn {
    @location(0) local_pos: vec2<f32>,
    @location(1) translate_size: vec4<f32>,
    @location(2) rot_z: vec4<f32>,
    @location(3) color: vec4<f32>,
    @location(4) atlas_rect: vec4<f32>,
    @location(5) kind_params: vec4<f32>,
};

struct VSOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) atlas_uv: vec2<f32>,
    @location(3) kind_params: vec2<f32>,
};

@group(0) @binding(0) var sprite_tex: texture_2d<f32>;
@group(0) @binding(1) var sprite_smp: sampler;

struct Lighting {
    light_count: u32,
    rt_mode: u32,
    rt_active: u32,
    rt_dynamic_count: u32,
    rt_dynamic_cap: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(1) @binding(2) var<uniform> u_lighting: Lighting;

@vertex
fn vs_main(input: VSIn) -> VSOut {
    let c = cos(input.rot_z.x);
    let s = sin(input.rot_z.x);
    let scaled = vec2<f32>(
        input.local_pos.x * input.translate_size.z,
        input.local_pos.y * input.translate_size.w
    );
    let rotated = vec2<f32>(
        scaled.x * c - scaled.y * s,
        scaled.x * s + scaled.y * c
    );
    let pos = rotated + input.translate_size.xy;
    let uv = input.local_pos + vec2<f32>(0.5, 0.5);
    let atlas_uv = mix(input.atlas_rect.xy, input.atlas_rect.zw, uv);

    var out: VSOut;
    out.position = vec4<f32>(pos, input.rot_z.y, 1.0);
    out.color = input.color;
    out.uv = uv;
    out.atlas_uv = atlas_uv;
    out.kind_params = input.kind_params.xy;
    return out;
}

fn terrain_style(base_color: vec3<f32>, uv: vec2<f32>, atlas_uv: vec2<f32>, slot: f32) -> vec3<f32> {
    let stripe = 0.5 + 0.5 * sin((atlas_uv.x * 64.0) + slot * 0.37);
    let top = vec3<f32>(base_color.r * 1.06, base_color.g * 1.03, base_color.b * 0.90);
    let bottom = vec3<f32>(base_color.r * 0.85, base_color.g * 0.92, base_color.b * 0.78);
    let grad = mix(bottom, top, uv.y);
    return mix(grad, grad * vec3<f32>(0.72, 0.88, 0.72), stripe * 0.35);
}

/// Procedural radial glow disk for light billboard sprites (kind == 4).
/// Returns rgba: bright core fading to transparent edge, rendered additive.
fn light_glow_style(base_color: vec3<f32>, uv: vec2<f32>) -> vec4<f32> {
    let centered = uv - vec2<f32>(0.5, 0.5);
    let dist = length(centered);

    // Tight bright core — very small, very bright.
    let core = 1.0 - smoothstep(0.0, 0.08, dist);
    // Mid-range soft halo.
    let halo = (1.0 - smoothstep(0.04, 0.38, dist)) * 0.55;
    // Wide outer atmospheric scatter.
    let scatter = (1.0 - smoothstep(0.18, 0.50, dist)) * 0.18;

    let brightness = core + halo + scatter;

    // Subtle chromatic aberration: red channel extends slightly further.
    let red_extra = (1.0 - smoothstep(0.06, 0.22, dist)) * 0.18;
    let color = vec3<f32>(
        base_color.r * brightness + red_extra,
        base_color.g * brightness,
        base_color.b * brightness,
    );

    // Alpha drives the additive accumulation weight.
    return vec4<f32>(color, brightness);
}

fn camera_style(base_color: vec3<f32>, uv: vec2<f32>, atlas_uv: vec2<f32>, slot: f32) -> vec3<f32> {
    let centered = uv - vec2<f32>(0.5, 0.5);
    let dist = length(centered);
    let ring = 1.0 - smoothstep(0.26, 0.43, abs(dist - 0.31));
    let lens = 1.0 - smoothstep(0.07, 0.31, dist);
    let scan = 0.5 + 0.5 * sin((atlas_uv.y * 48.0) + slot * 0.21);
    let ring_color = vec3<f32>(0.95, 0.98, 1.0);
    let lens_color = mix(base_color * vec3<f32>(0.36, 0.52, 0.78), base_color, scan * 0.6);
    return lens_color * (0.55 + lens * 0.45) + ring_color * ring * 0.55;
}

@fragment
fn fs_main(input: VSOut) -> @location(0) vec4<f32> {
    let kind = i32(input.kind_params.x + 0.5);
    let slot = input.kind_params.y;
    let sampled = textureSample(sprite_tex, sprite_smp, input.atlas_uv);
    var color = input.color.rgb * sampled.rgb;
    let alpha = input.color.a * sampled.a;

    // LightGlow (kind == 4): fully procedural radial glow, rendered additive.
    // Returns early so the atlas sample and alpha are overridden completely.
    if kind == 4 {
        return light_glow_style(input.color.rgb, input.uv);
    }

    if kind == 2 {
        color = camera_style(color, input.uv, input.atlas_uv, slot);
    } else if kind == 3 {
        color = terrain_style(color, input.uv, input.atlas_uv, slot);
    } else if kind == 1 {
        let pulse = 0.93 + 0.07 * sin(input.atlas_uv.x * 28.0 + slot * 0.15);
        color = color * pulse;
    } else {
        let grain = 0.98 + 0.02 * sin((input.atlas_uv.x + input.atlas_uv.y) * 32.0 + slot * 0.11);
        color = color * grain;
    }

    // Optional light-aware glow billboard helper for sprite lanes.
    let centered = input.uv - vec2<f32>(0.5, 0.5);
    let radial = 1.0 - smoothstep(0.22, 0.66, length(centered));
    let light_gain = 1.0 + min(f32(u_lighting.light_count) / 16.0, 0.40);
    if kind == 0 || kind == 1 {
        color += vec3<f32>(0.06, 0.09, 0.14) * radial * light_gain;
    }

    return vec4<f32>(color, alpha);
}
