struct Camera {
    view_proj: mat4x4<f32>,
    camera_eye: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> u_camera: Camera;

struct LightData {
    position_kind: vec4<f32>,
    direction_inner: vec4<f32>,
    color_intensity: vec4<f32>,
    params: vec4<f32>,
    shadow: vec4<f32>,
};

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

@group(0) @binding(1)
var<storage, read> u_lights: array<LightData, 32>;

@group(0) @binding(2)
var<uniform> u_lighting: Lighting;

// ── Shadow map resources ──────────────────────────────────────────────────────
struct ShadowUniform {
    light_view_proj: array<mat4x4<f32>, 4>,
    shadow_light_indices: vec4<i32>,
    shadow_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(3)
var shadow_map: texture_depth_2d_array;
@group(0) @binding(4)
var shadow_smp: sampler_comparison;
@group(0) @binding(5)
var<uniform> u_shadow: ShadowUniform;

struct VSIn {
    @location(0) position: vec3<f32>,
    @location(1) model_col0: vec4<f32>,
    @location(2) model_col1: vec4<f32>,
    @location(3) model_col2: vec4<f32>,
    @location(4) model_col3: vec4<f32>,
    @location(5) base_color: vec4<f32>,
    @location(6) material_params: vec4<f32>,
    @location(7) emissive: vec4<f32>,
};

struct VSOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) emissive: vec3<f32>,
    @location(2) world_pos: vec3<f32>,
    @location(3) local_pos: vec3<f32>,
    @location(4) primitive_code: f32,
};

@vertex
fn vs_main(input: VSIn) -> VSOut {
    let model = mat4x4<f32>(
        input.model_col0,
        input.model_col1,
        input.model_col2,
        input.model_col3
    );
    let world_pos = model * vec4<f32>(input.position, 1.0);

    var out: VSOut;
    out.position = u_camera.view_proj * world_pos;
    out.color = input.base_color;
    out.emissive = input.emissive.xyz;
    out.world_pos = world_pos.xyz;
    out.local_pos = input.position;
    out.primitive_code = input.material_params.w;
    return out;
}

/// 3×3 PCF shadow sample from the shadow map depth array.
/// Returns 1.0 = fully lit, 0.0 = fully in shadow.
///
/// Uses `textureLoad` + manual comparison rather than `textureSampleCompare` so
/// the comparison direction is explicit and portable across Metal / Vulkan / DX12.
/// Depth convention: 0 = near, 1 = far.  A surface is LIT when the stored nearest-
/// occluder depth is ≥ (fragment depth − bias), i.e. nothing is significantly closer
/// to the light than the fragment itself.
fn sample_shadow(slot: i32, world_pos: vec3<f32>) -> f32 {
    let light_space = u_shadow.light_view_proj[slot] * vec4<f32>(world_pos, 1.0);
    if light_space.w <= 0.0 {
        return 1.0;
    }
    let proj = light_space.xyz / light_space.w;
    // proj.z is OpenGL NDC [-1, 1].  The shadow depth pass VS goes through naga's
    // Metal/wgpu z-correction so the depth buffer stores (z_gl + 1) * 0.5 ∈ [0, 1].
    // We must apply the same conversion before comparing against stored depth.
    if proj.z < -1.0 || proj.z > 1.0 {
        return 1.0;
    }
    let wgpu_z = proj.z * 0.5 + 0.5;
    let depth_test = wgpu_z - 0.0005;
    // Convert GL NDC xy [-1,1] → UV [0,1], flipping Y (NDC +y up, UV +y down).
    let uv = proj.xy * vec2<f32>(0.5, -0.5) + vec2<f32>(0.5, 0.5);
    if uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 {
        return 1.0;
    }
    // 3×3 PCF kernel — textureLoad returns raw f32 depth, comparison is explicit.
    // A fragment is lit  when stored_depth >= depth_test (occluder is not closer).
    // A fragment is dark when stored_depth <  depth_test (occluder is closer to light).
    let map_size = 1024;
    var shadow_sum = 0.0;
    for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
        for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
            let px = vec2<i32>(
                clamp(i32(uv.x * f32(map_size)) + dx, 0, map_size - 1),
                clamp(i32(uv.y * f32(map_size)) + dy, 0, map_size - 1),
            );
            let stored = textureLoad(shadow_map, px, slot, 0);
            shadow_sum += select(0.0, 1.0, stored >= depth_test);
        }
    }
    return shadow_sum / 9.0;
}

fn evaluate_light(light: LightData, normal: vec3<f32>, world_pos: vec3<f32>, base_color: vec3<f32>) -> vec3<f32> {
    let to_light = light.position_kind.xyz - world_pos;
    let distance = max(length(to_light), 1e-4);
    let light_dir = to_light / distance;
    let range = max(light.params.x, 1e-3);
    if distance > range {
        return vec3<f32>(0.0);
    }

    // Distance falloff: squared smoothstep edge-rolloff combined with a gentle
    // linear-distance term to prevent the near-source hotspot from blowing out.
    // 1/(1+k*d) with k≈0.10 gives ~50% reduction at 10 units, preserving mid-range
    // luminosity while keeping the light physically believable.
    var attenuation = 1.0 - smoothstep(range * 0.65, range, distance);
    attenuation *= attenuation;
    attenuation *= 1.0 / (1.0 + distance * 0.10);

    if light.position_kind.w > 0.5 {
        let spot_axis = normalize(-light.direction_inner.xyz);
        let cone = dot(spot_axis, light_dir);
        let inner = light.direction_inner.w;
        let outer = light.params.y;
        let cone_factor = smoothstep(outer, inner, cone);
        attenuation *= cone_factor;
    }

    let ndotl = max(dot(normal, light_dir), 0.0);
    if ndotl <= 0.0 || attenuation <= 1e-5 {
        return vec3<f32>(0.0);
    }

    let view_dir = normalize(u_camera.camera_eye.xyz - world_pos);
    let half_dir = normalize(light_dir + view_dir);
    let spec_power = 24.0 + light.params.w * 24.0;
    let specular = pow(max(dot(normal, half_dir), 0.0), spec_power) * light.params.w;

    var shadow_term = 1.0;
    let shadow_slot = i32(round(light.shadow.y));
    if shadow_slot >= 0 && u32(shadow_slot) < u_shadow.shadow_count {
        // Shadow map available for this light — use PCF.
        shadow_term = sample_shadow(shadow_slot, world_pos);
    } else if light.shadow.x > 0.5 {
        // No shadow map slot available — fall back to an analytic soft-shadow approximation
        // that works on all platforms (including Metal). The penumbra floor is derived from
        // the light's softness parameter: a very soft light has a higher ambient floor.
        let penumbra_floor = mix(0.35, 0.80, 1.0 - light.params.z);
        shadow_term = mix(penumbra_floor, 1.0, ndotl);
    }

    let diffuse = base_color * ndotl;
    let spec = vec3<f32>(specular);
    return (diffuse + spec) * light.color_intensity.rgb * light.color_intensity.w * attenuation * shadow_term;
}

@fragment
fn fs_main(input: VSOut, @builtin(front_facing) is_front: bool) -> @location(0) vec4<f32> {
    let dpx = dpdx(input.world_pos);
    let dpy = dpdy(input.world_pos);
    // cross(dpy, dpx) gives outward-facing normals for front faces in wgpu window-space
    // (window y increases downward, so cross(dpx,dpy) produces inward normals — swap to fix).
    var normal = normalize(cross(dpy, dpx));
    if !is_front {
        normal = -normal;
    }

    var lit = input.color.rgb * 0.10;
    let light_count = min(u_lighting.light_count, 32u);
    if light_count == 0u {
        let fallback_dir = normalize(vec3<f32>(0.42, 0.74, 0.52));
        let fallback = max(dot(normal, fallback_dir), 0.0) * 0.76 + 0.24;
        lit += input.color.rgb * fallback;
    } else {
        for (var i: u32 = 0u; i < light_count; i = i + 1u) {
            lit += evaluate_light(u_lights[i], normal, input.world_pos, input.color.rgb);
        }
    }
    lit += input.emissive;
    var alpha = input.color.a;

    // Box primitives get extra edge contrast so the tank keeps a clear prism silhouette.
    if input.primitive_code > 0.5 {
        let edge = max(max(abs(input.local_pos.x), abs(input.local_pos.y)), abs(input.local_pos.z));
        let edge_boost = smoothstep(0.38, 0.50, edge);
        lit += vec3<f32>(0.08, 0.12, 0.18) * edge_boost;
        alpha = clamp(alpha + edge_boost * 0.32, 0.0, 1.0);
    }
    return vec4<f32>(lit, alpha);
}
