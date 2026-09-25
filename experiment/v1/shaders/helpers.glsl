// ============================================================================
// helpers.glsl
//
// Shared device-side helper functions for all SPH compute shaders.
// Dependency: MUST be included AFTER common.glsl (reads its spec constants
// and buffer declarations).
//
// Usage:
//     #version 460
//     #extension GL_GOOGLE_include_directive : enable
//     #include "common.glsl"
//     #include "helpers.glsl"
//
// Contents:
//   - Wendland C4 kernel evaluation (value + gradient)
//   - 1-based voxel_id ↔ 3D voxel coord conversions + grid bounds check
//   - Symmetric 3×3 correction_inverse unpacking (2 vec4 → mat3)
// ============================================================================

#ifndef SPH_HELPERS_GLSL_INCLUDED
#define SPH_HELPERS_GLSL_INCLUDED

// ============================================================================
// Wendland C4 kernel and gradient.
//
// Support radius = h, q = r / h ∈ [0, 1). Normalization baked into
// KERNEL_COEFFICIENT / KERNEL_GRADIENT_COEFFICIENT (Python-side precomputed,
// including 1/h^DIM and 1/h^(DIM+1)).
//
//   W(q)   = KERNEL_COEFFICIENT          · (1-q)^6 · (35/3·q² + 6q + 1)
//   dW/dq  = (1-q)^5 · q · (-280/3·q - 56/3)
//   ∇W     = KERNEL_GRADIENT_COEFFICIENT · dW/dq · r̂
// ============================================================================

// Support radius squared. glslang rejects a global `const` built from a spec
// constant, so this is a function; the driver folds it at pipeline creation.
// Neighbour loops compare squared distances against it before taking a sqrt.
float smoothing_length_squared() { return SMOOTHING_LENGTH * SMOOTHING_LENGTH; }

float evaluate_kernel(float distance) {
    float normalized_distance = distance / SMOOTHING_LENGTH;
    if (normalized_distance >= 1.0) return 0.0;

    float one_minus_q    = 1.0 - normalized_distance;
    float one_minus_q_sq = one_minus_q * one_minus_q;
    float one_minus_q_6  = one_minus_q_sq * one_minus_q_sq * one_minus_q_sq;

    // (35/3) · q² + 6q + 1
    float profile_polynomial =
        normalized_distance * ((35.0 / 3.0) * normalized_distance + 6.0) + 1.0;

    return KERNEL_COEFFICIENT * one_minus_q_6 * profile_polynomial;
}

vec3 evaluate_kernel_gradient(vec3 relative_position, float distance) {
    if (distance < 1e-12) return vec3(0.0);
    float normalized_distance = distance / SMOOTHING_LENGTH;
    if (normalized_distance >= 1.0) return vec3(0.0);

    float one_minus_q    = 1.0 - normalized_distance;
    float one_minus_q_sq = one_minus_q * one_minus_q;
    float one_minus_q_4  = one_minus_q_sq * one_minus_q_sq;
    float one_minus_q_5  = one_minus_q_4 * one_minus_q;

    // q · (-280/3 · q + -56/3)
    float derivative_polynomial =
        normalized_distance * ((-280.0 / 3.0) * normalized_distance + (-56.0 / 3.0));

    float gradient_magnitude_scalar =
        KERNEL_GRADIENT_COEFFICIENT * one_minus_q_5 * derivative_polynomial;

    return gradient_magnitude_scalar * (relative_position / distance);
}

// ----------------------------------------------------------------------------
// Unguarded variants for neighbour loops that have ALREADY tested
// 1e-24 <= r^2 < h^2 on the squared distance. The guarded functions above
// re-test q >= 1 and r < 1e-12; once the loop test moved to squared
// distances the compiler can no longer prove those re-tests redundant, and
// the extra compares cost ~6 % of the neighbour-loop time (measured
// 2026-09-25, RTX 4070 Ti SUPER, 3 mm tank). Callers guarantee the range.
// ----------------------------------------------------------------------------
float evaluate_kernel_unguarded(float distance) {
    float normalized_distance = distance / SMOOTHING_LENGTH;
    float one_minus_q    = 1.0 - normalized_distance;
    float one_minus_q_sq = one_minus_q * one_minus_q;
    float one_minus_q_6  = one_minus_q_sq * one_minus_q_sq * one_minus_q_sq;
    float profile_polynomial =
        normalized_distance * ((35.0 / 3.0) * normalized_distance + 6.0) + 1.0;
    return KERNEL_COEFFICIENT * one_minus_q_6 * profile_polynomial;
}

vec3 evaluate_kernel_gradient_unguarded(vec3 relative_position, float distance) {
    float normalized_distance = distance / SMOOTHING_LENGTH;
    float one_minus_q    = 1.0 - normalized_distance;
    float one_minus_q_sq = one_minus_q * one_minus_q;
    float one_minus_q_4  = one_minus_q_sq * one_minus_q_sq;
    float one_minus_q_5  = one_minus_q_4 * one_minus_q;
    float derivative_profile =
        normalized_distance * ((-280.0 / 3.0) * normalized_distance + (-56.0 / 3.0));
    return KERNEL_GRADIENT_COEFFICIENT * one_minus_q_5 * derivative_profile
         * (relative_position / distance);
}

// ============================================================================
// 1-based voxel_id ↔ 3D voxel coord.   (V1: x-slowest encoding)
//
// V1 partitions along X. The voxel_id encoding is "x-slowest" so that
// each x-column of voxels (NY*NZ voxels at a fixed x) occupies a CONTIGUOUS
// block of voxel_id values. This makes:
//   * the leading-ghost and trailing-ghost voxel ranges contiguous segments
//     of the global voxel_id space
//   * "is this voxel my own?" reduce to a single range comparison
//   * predict / update_voxel / defrag dispatch naturally over the own range
//
// Encoding:  voxel_id = y + z*GRID_DIMENSION_Y + x*GRID_DIMENSION_Y*GRID_DIMENSION_Z + 1
// Inverse:   y    = (id-1) % NY
//            z    = ((id-1) / NY) % NZ
//            x    = (id-1) / (NY*NZ)
//
// Slot 0 of every voxel buffer is unused (1-based). Particles store their
// voxel_id in position_voxel_id.w (0 = dead sentinel).
//
// V1 merged-buffer scheme: GRID_DIMENSION_X is the EXTENDED nx — it covers
// own columns plus leading/trailing ghost columns. Ghost particles share
// set 0 / set 1 with own particles; ranges are split by spec const.
// (See LEADING_GHOST_VOXEL_COUNT / TRAILING_GHOST_VOXEL_COUNT in common.glsl.)
// ============================================================================

ivec3 own_coord_of(uint voxel_id) {
    uint zero_based = voxel_id - 1u;
    uint y           = zero_based % GRID_DIMENSION_Y;
    uint after_y     = zero_based / GRID_DIMENSION_Y;
    uint z           = after_y    % GRID_DIMENSION_Z;
    uint x           = after_y    / GRID_DIMENSION_Z;
    return ivec3(x, y, z);
}

uint own_voxel_id_of(ivec3 coord) {
    return uint(coord.y)
         + uint(coord.z) * GRID_DIMENSION_Y
         + uint(coord.x) * GRID_DIMENSION_Y * GRID_DIMENSION_Z
         + 1u;
}

bool in_own_grid(ivec3 coord) {
    return coord.x >= 0 && coord.x < int(GRID_DIMENSION_X)
        && coord.y >= 0 && coord.y < int(GRID_DIMENSION_Y)
        && coord.z >= 0 && coord.z < int(GRID_DIMENSION_Z);
}

// ============================================================================
// V1 own-vs-ghost classification on the merged voxel_id range.
//
// Voxel_id layout in extended grid:
//   [1, M]                          = leading ghost  (peer's data, end GPUs: M=0)
//   [M+1, EXTENDED_TOTAL - N]       = own            (this GPU's particles)
//   [EXTENDED_TOTAL - N + 1, TOTAL] = trailing ghost (peer's data, end GPUs: N=0)
//
// EXTENDED_TOTAL = GRID_DIMENSION_X * GRID_DIMENSION_Y * GRID_DIMENSION_Z.
// M = LEADING_GHOST_VOXEL_COUNT, N = TRAILING_GHOST_VOXEL_COUNT.
// ============================================================================

uint extended_voxel_count() {
    return GRID_DIMENSION_X * GRID_DIMENSION_Y * GRID_DIMENSION_Z;
}

bool is_own_voxel(uint voxel_id) {
    return voxel_id > LEADING_GHOST_VOXEL_COUNT
        && voxel_id <= extended_voxel_count() - TRAILING_GHOST_VOXEL_COUNT;
}

bool is_leading_ghost_voxel(uint voxel_id) {
    return voxel_id >= 1u && voxel_id <= LEADING_GHOST_VOXEL_COUNT;
}

bool is_trailing_ghost_voxel(uint voxel_id) {
    return voxel_id > extended_voxel_count() - TRAILING_GHOST_VOXEL_COUNT
        && voxel_id <= extended_voxel_count();
}

// ============================================================================
// V1 own / ghost pid range helpers (mirrors voxel layout).
//
// Pid layout in set 0:
//   [1, LEADING_GHOST_POOL_SIZE]                            = leading ghost
//   [own_first_pid, own_last_pid]                           = own
//   [own_last_pid+1, own_last_pid+TRAILING_GHOST_POOL_SIZE] = trailing ghost
// ============================================================================

uint own_first_pid() {
    return LEADING_GHOST_POOL_SIZE + 1u;
}

uint own_last_pid() {
    return LEADING_GHOST_POOL_SIZE + OWN_POOL_SIZE;
}

// Row stride of the transposed neighbour list = pool capacity incl. slot 0
// (own + both ghost pools + 1). Must match SphSimulatorV1._build_buffer_specs.
uint neighbor_list_stride() {
    return OWN_POOL_SIZE + LEADING_GHOST_POOL_SIZE + TRAILING_GHOST_POOL_SIZE + 1u;
}

uint leading_ghost_first_pid() { return 1u; }
uint leading_ghost_last_pid()  { return LEADING_GHOST_POOL_SIZE; }

uint trailing_ghost_first_pid() {
    return LEADING_GHOST_POOL_SIZE + OWN_POOL_SIZE + 1u;
}
uint trailing_ghost_last_pid() {
    return LEADING_GHOST_POOL_SIZE + OWN_POOL_SIZE + TRAILING_GHOST_POOL_SIZE;
}

// ============================================================================
// Unpack symmetric 3×3 correction_inverse from 2 vec4.
//
//   Storage: [pid*2]   = (M[0][0], M[1][1], M[2][2], M[0][1])
//            [pid*2+1] = (M[0][2], M[1][2], _, _)
// By symmetry M[1][0] = M[0][1], M[2][0] = M[0][2], M[2][1] = M[1][2].
// ============================================================================

mat3 unpack_correction_inverse(uint particle_id) {
    vec4 a = correction_inverse[particle_id * 2u];
    vec4 b = correction_inverse[particle_id * 2u + 1u];
    return mat3(
        vec3(a.x, a.w, b.x),   // column 0
        vec3(a.w, a.y, b.y),   // column 1
        vec3(b.x, b.y, a.z)    // column 2
    );
}

// ============================================================================
// Rotor helpers (2026-09-25)
// ============================================================================

// BOUNDARY and ROTOR are both walls from the fluid's point of view: static
// density (rest density stored), no PST, solid-solid pairs skipped in the
// continuity equation. Only their motion differs (ROTOR is prescribed).
bool is_solid_kind(uint kind) {
    return kind == MATERIAL_BOUNDARY || kind == MATERIAL_ROTOR;
}

// Rodrigues rotation of `vector` about the unit `axis` by the angle whose
// cosine / sine are given (right-hand rule).
vec3 rotate_about_axis(vec3 vector, vec3 axis, float cos_theta, float sin_theta) {
    return vector * cos_theta
         + cross(axis, vector) * sin_theta
         + axis * dot(axis, vector) * (1.0 - cos_theta);
}

#endif  // SPH_HELPERS_GLSL_INCLUDED
