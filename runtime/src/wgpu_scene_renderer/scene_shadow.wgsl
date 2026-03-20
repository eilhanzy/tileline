struct ShadowPassUniform {
    light_view_proj: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> u_shadow_pass: ShadowPassUniform;

struct VSIn {
    @location(0) position: vec3<f32>,
    @location(1) model_col0: vec4<f32>,
    @location(2) model_col1: vec4<f32>,
    @location(3) model_col2: vec4<f32>,
    @location(4) model_col3: vec4<f32>,
    @location(5) _base_color: vec4<f32>,
    @location(6) _material_params: vec4<f32>,
    @location(7) _emissive: vec4<f32>,
};

@vertex
fn vs_main(input: VSIn) -> @builtin(position) vec4<f32> {
    let model = mat4x4<f32>(
        input.model_col0,
        input.model_col1,
        input.model_col2,
        input.model_col3
    );
    let world_pos = model * vec4<f32>(input.position, 1.0);
    return u_shadow_pass.light_view_proj * world_pos;
}
