#version 460
#extension GL_GOOGLE_include_directive : enable
#include "common.glsl"

// ============================================================================
// particle.vert — point-sprite vertex shader for SPH visualization.
//
// One vertex per particle slot; gl_VertexIndex is 0-based, particle_id is
// 1-based (slot 0 unused). Reads particle state from the same set 0 SSBOs the
// simulator owns — buffer declarations come from common.glsl, so this shader
// uses identical binding numbers to all compute kernels (single source of
// truth). glslc -O DCE-strips the bindings this shader does not reference.
//
// Set 0 buffers actually consumed:
//   binding 0 : position_voxel_id        (xyz + voxel_id_as_float)
//   binding 1 : density_pressure         (ρ, P) — canonical, post scratch→primary
//                                         copy in the simulator's step cmd.
//   binding 3 : velocity_mass            (vxyz, mass)
//   binding 4 : acceleration             (axyz, ω_z) — .w carries vorticity,
//                                         written by force.comp for mode 5.
//   binding 8 : density_gradient_kernel_sum (∇ρ, kernel_sum)
//
// Color modes:
//   0  speed         (length(velocity) → viridis colormap, normalized by scale)
//   1  acceleration  (length(acceleration) → viridis, normalized by scale)
//   2  density       ((rho - rho_0)/rho_0 → diverging blue/white/red)
//   3  voxel_id      (deterministic rainbow hash; debug neighbor-cell sort)
//   4  kernel_sum    ((kernel_sum - 1) → diverging blue/white/red; centered on 1
//                     because Σ V·W ≈ 1 for a well-sampled interior particle)
//   5  vorticity     (ω_z = [∇×v]_z from acceleration.w → filled contour bands,
//                     warm = CCW/positive, cool = CW/negative)
//
// Dead particles (voxel_id == VOXEL_ID_DEAD) are pushed off-screen with size 0.
// ============================================================================

layout(push_constant) uniform PushConstants {
    mat4  view_proj;            // 64 B
    uint  color_mode;           //  4 B   0/1/2/3/4/5
    float velocity_scale;       //  4 B   normalize speed → [0, 1]
    float acceleration_scale;   //  4 B   normalize |a| → [0, 1]
    float density_deviation_scale; // 4 B normalize (rho-rho0)/rho0 → [-1, +1]
    float rest_density;         //  4 B   reference density for mode 2
    float point_size;           //  4 B   gl_PointSize in pixels
    float kernel_sum_scale;     //  4 B   normalize (kernel_sum - 1) → [-1, +1]
    float vorticity_scale;      //  4 B   normalize ω_z → [-1, +1] (mode 5)
} pc;

layout(location = 0) out vec3 frag_color;

// ----- colormap helpers ------------------------------------------------------

// Viridis colormap for unsigned [0, 1] values. Always visibly colored —
// the minimum is a saturated dark purple (not black), so static particles
// remain visible against the dark background.
vec3 colormap_viridis(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 c0 = vec3(0.267, 0.005, 0.329);   // dark purple   (t=0)
    vec3 c1 = vec3(0.190, 0.408, 0.557);   // blue          (t≈0.33)
    vec3 c2 = vec3(0.208, 0.718, 0.473);   // green         (t≈0.66)
    vec3 c3 = vec3(0.992, 0.906, 0.144);   // yellow        (t=1)
    if (t < 0.333) return mix(c0, c1, t / 0.333);
    if (t < 0.666) return mix(c1, c2, (t - 0.333) / 0.333);
    return mix(c2, c3, (t - 0.666) / 0.334);
}

// Classic CFD "jet" colormap for unsigned [0, 1] values: dark blue → blue →
// cyan → green → yellow → red → dark red. Used for speed / acceleration so
// the picture reads like a Fluent / ParaView contour plot.
vec3 colormap_jet(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 c0 = vec3(0.000, 0.000, 0.500);   // dark blue   (t=0)
    vec3 c1 = vec3(0.000, 0.000, 1.000);   // blue        (t=0.125)
    vec3 c2 = vec3(0.000, 1.000, 1.000);   // cyan        (t=0.375)
    vec3 c3 = vec3(1.000, 1.000, 0.000);   // yellow      (t=0.625)
    vec3 c4 = vec3(1.000, 0.000, 0.000);   // red         (t=0.875)
    vec3 c5 = vec3(0.500, 0.000, 0.000);   // dark red    (t=1)
    if (t < 0.125) return mix(c0, c1, t / 0.125);
    if (t < 0.375) return mix(c1, c2, (t - 0.125) / 0.25);
    if (t < 0.625) return mix(c2, c3, (t - 0.375) / 0.25);
    if (t < 0.875) return mix(c3, c4, (t - 0.625) / 0.25);
    return mix(c4, c5, (t - 0.875) / 0.125);
}

// Diverging blue-white-red for signed [-1, +1] values (density deviation).
vec3 colormap_diverging(float t) {
    t = clamp(t, -1.0, 1.0);
    vec3 cold  = vec3(0.230, 0.299, 0.754);
    vec3 white = vec3(0.985, 0.985, 0.985);
    vec3 hot   = vec3(0.706, 0.016, 0.150);
    if (t < 0.0) return mix(white, cold, -t);
    return mix(white, hot, t);
}

// Pseudo-rainbow from uint hash (golden ratio for nice spread).
vec3 colormap_rainbow_hash(uint key) {
    // Hash key into a [0, 1] hue; HSV→RGB with full saturation/value.
    float hue = fract(float(key) * 0.6180339887);
    vec3 k = vec3(5.0, 3.0, 1.0);
    vec3 p = abs(fract(vec3(hue) + k / 6.0) * 6.0 - 3.0);
    return clamp(p - 1.0, 0.0, 1.0);
}

// Vorticity contour map — ported from the legacy OpenGL vorticity shader.
// Input v is the already-scaled ω_z (dimensionless; |v| >= 1 saturates). Warm
// (yellow→dark red) for positive/CCW, cool (light→dark blue) for negative/CW,
// quantized into filled contour bands with subtle in-band gradient, band-edge
// line highlights, and a white flash on the strongest cores.
vec3 colormap_vorticity(float v) {
    const float n_levels = 10.0;                 // number of contour bands
    float level = floor(v * n_levels) / n_levels;
    float next_level = level + 1.0 / n_levels;
    float level_frac = clamp((v - level) / max(next_level - level, 1e-6), 0.0, 1.0);

    float norm_level = clamp(abs(level), 0.0, 1.0);
    vec3 color = (v > 0.0)
        ? mix(vec3(1.0, 0.9, 0.7), vec3(0.8, 0.2, 0.0), norm_level)   // warm
        : mix(vec3(0.7, 0.9, 1.0), vec3(0.0, 0.2, 0.8), norm_level);  // cool

    // Subtle gradient within each contour band.
    color = mix(color * 0.9, color * 1.1, level_frac);

    // Darken the band edges so the contour lines read.
    float line_highlight = 1.0 - smoothstep(0.9, 1.0, level_frac)
                         + smoothstep(0.0, 0.1, level_frac);
    color = mix(color, color * 0.5, line_highlight * 0.3);

    // Strong-vorticity cores flash toward white.
    if (abs(v) > 0.8) {
        color = mix(color, vec3(1.0), 0.2);
    }
    return clamp(color, 0.0, 1.0);
}

void main() {
    uint particle_id = gl_VertexIndex + 1u;
    vec4 pos_vid = position_voxel_id[particle_id];
    uint vid = uint(round(pos_vid.w));

    // Dead / unallocated slot → push off-screen and shrink to nothing.
    if (vid == VOXEL_ID_DEAD) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        gl_PointSize = 0.0;
        frag_color = vec3(0.0);
        return;
    }

    gl_Position = pc.view_proj * vec4(pos_vid.xyz, 1.0);
    gl_PointSize = pc.point_size;

    // ---- per-mode colorization ---------------------------------------------
    if (pc.color_mode == 0u) {
        float speed = length(velocity_mass[particle_id].xyz);
        frag_color = colormap_jet(speed * pc.velocity_scale);
    } else if (pc.color_mode == 1u) {
        float accel_mag = length(acceleration[particle_id].xyz);
        frag_color = colormap_jet(accel_mag * pc.acceleration_scale);
    } else if (pc.color_mode == 2u) {
        // density_pressure is the canonical ρ at binding 1; the simulator's
        // step cmd has already copied scratch → primary by render time, so
        // this read is the freshly-written ρ_{n+1}.
        float rho = density_pressure[particle_id].x;
        float dev = (rho - pc.rest_density) / max(pc.rest_density, 1e-6);
        frag_color = colormap_diverging(dev * pc.density_deviation_scale);
    } else if (pc.color_mode == 3u) {
        // voxel_id rainbow
        frag_color = colormap_rainbow_hash(vid);
    } else if (pc.color_mode == 4u) {
        // kernel_sum, centered on 1.0
        float ks = density_gradient_kernel_sum[particle_id].w;
        float dev = (ks - 1.0) * pc.kernel_sum_scale;
        frag_color = colormap_diverging(dev);
    } else {
        // mode 5: vorticity ω_z, stashed in acceleration.w by force.comp.
        // Boundary/inlet particles never run force.comp so their .w stays 0
        // (neutral band); the fluid shows the rotating structure.
        float vort_z = acceleration[particle_id].w * pc.vorticity_scale;
        frag_color = colormap_vorticity(vort_z);
    }
}
