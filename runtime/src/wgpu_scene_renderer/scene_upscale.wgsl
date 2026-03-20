struct UpscaleUniform {
    inv_source_size: vec2<f32>,
    source_uv_scale: vec2<f32>,
    sharpness: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var src_tex: texture_2d<f32>;
@group(0) @binding(1) var src_smp: sampler;
@group(0) @binding(2) var<uniform> u_upscale: UpscaleUniform;

struct VSOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VSOut {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -3.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 3.0,  1.0)
    );
    let pos = positions[vid];
    var out: VSOut;
    out.position = vec4<f32>(pos, 0.0, 1.0);
    out.uv = pos * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
    return out;
}

@fragment
fn fs_main(input: VSOut) -> @location(0) vec4<f32> {
    let uv = clamp(input.uv * u_upscale.source_uv_scale, vec2<f32>(0.0), vec2<f32>(1.0));
    let center = textureSample(src_tex, src_smp, uv);

    // RCAS-like lightweight unsharp mask to preserve contrast after spatial upscale.
    let texel = u_upscale.inv_source_size;
    let s0 = textureSample(src_tex, src_smp, uv + vec2<f32>( texel.x, 0.0));
    let s1 = textureSample(src_tex, src_smp, uv + vec2<f32>(-texel.x, 0.0));
    let s2 = textureSample(src_tex, src_smp, uv + vec2<f32>(0.0,  texel.y));
    let s3 = textureSample(src_tex, src_smp, uv + vec2<f32>(0.0, -texel.y));
    let neighborhood = (s0 + s1 + s2 + s3) * 0.25;
    let sharpened = center.rgb + (center.rgb - neighborhood.rgb) * (u_upscale.sharpness * 1.65);
    return vec4<f32>(max(sharpened, vec3<f32>(0.0)), center.a);
}
