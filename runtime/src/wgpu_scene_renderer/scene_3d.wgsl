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
    @location(0) color:          vec4<f32>,
    @location(1) emissive:       vec3<f32>,
    @location(2) world_pos:      vec3<f32>,
    @location(3) local_pos:      vec3<f32>,
    @location(4) primitive_code: f32,
    @location(5) roughness:      f32,
    @location(6) metallic:       f32,
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
    out.position      = u_camera.view_proj * world_pos;
    out.color         = input.base_color;
    out.emissive      = input.emissive.xyz;
    out.world_pos     = world_pos.xyz;
    out.local_pos     = input.position;
    out.primitive_code = input.material_params.w;
    out.roughness     = input.material_params.x;
    out.metallic      = input.material_params.y;
    return out;
}

// ── Procedural environment constants ─────────────────────────────────────────
// Sky / ground colors used for IBL approximation.
const ENV_SKY_COLOR:    vec3<f32> = vec3<f32>(0.55, 0.75, 1.20);
const ENV_GROUND_COLOR: vec3<f32> = vec3<f32>(0.30, 0.22, 0.14);

// ── PBR helpers ──────────────────────────────────────────────────────────────

fn schlick_fresnel(f0: vec3<f32>, cos_theta: f32) -> vec3<f32> {
    return f0 + (vec3<f32>(1.0) - f0) * pow(clamp(1.0 - cos_theta, 0.0, 1.0), 5.0);
}

fn d_ggx(ndoth: f32, roughness: f32) -> f32 {
    let a  = roughness * roughness;
    let a2 = a * a;
    let denom = ndoth * ndoth * (a2 - 1.0) + 1.0;
    return a2 / (3.14159265 * denom * denom + 1e-7);
}

/// Sample the procedural sky-ground environment in direction r.
fn env_sample(r: vec3<f32>) -> vec3<f32> {
    let t = r.y * 0.5 + 0.5;   // 0 = below horizon, 1 = straight up
    return mix(ENV_GROUND_COLOR, ENV_SKY_COLOR, t);
}

// ── Shadow sampling ───────────────────────────────────────────────────────────
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
    // Bias: near=0.5, far≈60.5 → gradient ≈ near*far/((far-near)*z²).
    // At z=5:  gradient ≈ 0.004/unit → bias ≈ 0.025 units (well below ball radius).
    // At z=30: gradient ≈ 0.00056/unit → bias ≈ 0.18 units (still catches small balls).
    let depth_test = wgpu_z - 0.0001;
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

// ── Direct lighting ───────────────────────────────────────────────────────────
fn evaluate_light(
    light:     LightData,
    normal:    vec3<f32>,
    world_pos: vec3<f32>,
    base_color: vec3<f32>,
    view_dir:  vec3<f32>,
    roughness: f32,
    metallic:  f32,
) -> vec3<f32> {
    let to_light  = light.position_kind.xyz - world_pos;
    let distance  = max(length(to_light), 1e-4);
    let light_dir = to_light / distance;
    let range     = max(light.params.x, 1e-3);
    if distance > range {
        return vec3<f32>(0.0);
    }

    // Distance falloff: squared smoothstep edge-rolloff combined with a gentle
    // linear-distance term to prevent the near-source hotspot from blowing out.
    var attenuation = 1.0 - smoothstep(range * 0.65, range, distance);
    attenuation *= attenuation;
    attenuation *= 1.0 / (1.0 + distance * 0.10);

    if light.position_kind.w > 0.5 {
        let spot_axis   = normalize(-light.direction_inner.xyz);
        let cone        = dot(spot_axis, light_dir);
        let inner       = light.direction_inner.w;
        let outer       = light.params.y;
        let cone_factor = smoothstep(outer, inner, cone);
        attenuation    *= cone_factor;
    }

    let ndotl = max(dot(normal, light_dir), 0.0);
    if ndotl <= 0.0 || attenuation <= 1e-5 {
        return vec3<f32>(0.0);
    }

    // ── Simplified Cook-Torrance BRDF ────────────────────────────────────────
    let f0       = mix(vec3<f32>(0.04), base_color, metallic);
    let half_dir = normalize(light_dir + view_dir);
    let ndoth    = max(dot(normal, half_dir), 0.0);
    let ldoth    = max(dot(light_dir, half_dir), 0.0);
    let D        = d_ggx(ndoth, max(roughness, 0.05));
    let F        = schlick_fresnel(f0, ldoth);
    // Simplified visibility: drop G term, approximate 1/(4·ndotl·ndotv) as 0.25.
    let specular = D * F * 0.25 * light.params.w;

    // Metals have no diffuse; kd blends towards zero as metallic → 1.
    let kd      = (vec3<f32>(1.0) - F) * (1.0 - metallic);
    let diffuse = kd * base_color * ndotl;

    // ── Shadow ───────────────────────────────────────────────────────────────
    var shadow_term = 1.0;
    let shadow_slot = i32(round(light.shadow.y));
    if shadow_slot >= 0 && u32(shadow_slot) < u_shadow.shadow_count {
        shadow_term = sample_shadow(shadow_slot, world_pos);
    } else if light.shadow.x > 0.5 {
        let penumbra_floor = mix(0.35, 0.80, 1.0 - light.params.z);
        shadow_term = mix(penumbra_floor, 1.0, ndotl);
    }

    return (diffuse + specular) * light.color_intensity.rgb * light.color_intensity.w * attenuation * shadow_term;
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

    let view_dir = normalize(u_camera.camera_eye.xyz - input.world_pos);

    var lit = input.color.rgb * 0.04;
    let light_count = min(u_lighting.light_count, 32u);
    if light_count == 0u {
        let fallback_dir = normalize(vec3<f32>(0.42, 0.74, 0.52));
        let fallback = max(dot(normal, fallback_dir), 0.0) * 0.76 + 0.24;
        lit += input.color.rgb * fallback;
    } else {
        for (var i: u32 = 0u; i < light_count; i = i + 1u) {
            lit += evaluate_light(
                u_lights[i], normal, input.world_pos, input.color.rgb,
                view_dir, input.roughness, input.metallic,
            );
        }
    }

    // ── Procedural environment IBL ────────────────────────────────────────────
    // Fresnel at viewing angle — determines how much environment is reflected.
    let f0_env   = mix(vec3<f32>(0.04), input.color.rgb, input.metallic);
    let F_env    = schlick_fresnel(f0_env, max(dot(normal, view_dir), 0.0));
    let refl     = reflect(-view_dir, normal);
    let rough_sq = input.roughness * input.roughness;
    // Env specular: glossy/metallic surfaces pick up sky/ground color.
    let env_spec = F_env * env_sample(refl) * (1.0 - rough_sq) * (1.0 - rough_sq);
    // Env diffuse: hemisphere-integrated ambient, suppressed on metals (they have no diffuse).
    let env_diff = env_sample(normal) * input.color.rgb * (1.0 - input.metallic) * 0.12;
    lit += env_spec + env_diff;

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
