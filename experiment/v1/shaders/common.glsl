// ============================================================================
// common.glsl
//
// Single source of truth for all SPH compute shaders: specialization constants,
// descriptor set bindings, and scalar constants. Every compute shader in
// shaders/sph/ must include this file and use only the fields it needs. SPIR-V
// optimization (compile with -O) strips unused bindings at compile time, so
// declaring everything here does not force every shader to bind every buffer.
//
// Convention:
//   - Each shader's .comp file starts with:
//         #version 460
//         #extension GL_GOOGLE_include_directive : enable
//         #include "common.glsl"
//   - Shader-specific bindings and local helpers go in the .comp file after
//     the include, before main().
//
// Descriptor set layout:
//   set 0 : own particle SoA (per-particle persistent + scratch fields)
//   set 1 : own voxel cell structures
//   set 2 : ghost particle + ghost voxel structures (V1 multi-GPU; V0-a: empty)
//   set 3 : global status, transport, diagnostics, material parameters
// ============================================================================

#ifndef SPH_COMMON_GLSL_INCLUDED
#define SPH_COMMON_GLSL_INCLUDED

// ============================================================================
// Specialization constants (VkSpecializationInfo)
//
// ID range plan:
//   0  - 9   : global physics scalars + own grid origin  (id=3 reserved)
//   10       : multi-GPU bit-exactness toggle
//   11 - 13  : own grid dimensions
//   14 - 16  : correction regularization tunables
//   17 - 19  : gravity
//   20 - 29  : voxel layout / reserved
//   30 - 33  : dimension + kernel coefficients
//   34 - 39  : reserved for future kernel / simulation options
//   40 - 42  : SPH numerical parameters (ε_h², PST main, PST anti)
//   43 - 46  : algorithm ablation toggles (KCG, density diffusion, PST, prefix-sum defrag)
//   47 - 49  : reserved for future ablation toggles
//   50 - 53  : capacities + workgroup size + pool size
//   54 - 79  : reserved
//   80 - 88  : multi-GPU ghost grid parameters
//   89 - 127 : reserved
//
// Per-material parameters (rest_density, viscosity, eos_constant, radius,
// volume, rotor_angular_velocity) are NOT spec constants — they live in
// MaterialParametersBuffer (set 3 binding 7), indexed by group_id.
// ============================================================================

// --- Global physics scalars ---
layout(constant_id = 0) const float SMOOTHING_LENGTH   = 0.009;
layout(constant_id = 1) const float SPEED_OF_SOUND     = 300.0;   // c0
layout(constant_id = 2) const float DELTA_COEFFICIENT  = 0.1;     // δ-plus diffusion coefficient
// constant_id = 3 reserved (was EPSILON_SHIFT, a legacy unused PST placeholder)
layout(constant_id = 4) const float POWER_PARAMETER    = 7.0;     // EOS γ
layout(constant_id = 5) const float CFL_NUMBER         = 0.15;    // informational; dt computed on CPU
layout(constant_id = 6) const float TIMESTEP           = 4.5e-6;  // dt = CFL · H / c0 = 0.15 · 0.009 / 300

// --- Own grid origin (world-space corner of voxel [0,0,0] on this GPU) ---
layout(constant_id = 7) const float OWN_ORIGIN_X = -1.758406;
layout(constant_id = 8) const float OWN_ORIGIN_Y = -1.758406;
layout(constant_id = 9) const float OWN_ORIGIN_Z =  0.0;

// --- Multi-GPU bit-exactness ---
// V0 (single-GPU): default false — no precise qualifiers required.
// V1+ (multi-GPU ghost integration): Python should override to true.
layout(constant_id = 10) const bool STRICT_BIT_EXACT = false;

// --- Own grid dimensions (in voxels) ---
layout(constant_id = 11) const uint GRID_DIMENSION_X = 128u;
layout(constant_id = 12) const uint GRID_DIMENSION_Y = 128u;
layout(constant_id = 13) const uint GRID_DIMENSION_Z = 1u;        // 1 in 2D

// Total number of voxels in the own grid. Spec-constant-derived constant
// (folded at pipeline creation). Used by per-voxel kernels for bounds checks.
const uint TOTAL_VOXEL_COUNT = GRID_DIMENSION_X * GRID_DIMENSION_Y * GRID_DIMENSION_Z;

// --- Correction (KCG) regularization ---
layout(constant_id = 14) const float REGULARIZATION_XI                    = 0.1;
layout(constant_id = 15) const float REGULARIZATION_DETERMINANT_THRESHOLD = 1e-4;
layout(constant_id = 16) const float REGULARIZATION_MAX_FROBENIUS_NORM    = 10.0;

// --- Gravity ---
layout(constant_id = 17) const float GRAVITY_X = 0.0;
layout(constant_id = 18) const float GRAVITY_Y = 0.0;
layout(constant_id = 19) const float GRAVITY_Z = 0.0;

// --- Voxel layout ---
layout(constant_id = 20) const uint VOXEL_ORDER = 0u;             // 0 = linear z-major; 1 = Morton (future)

// --- Micropolar (V0 reserved, not integrated) ---
layout(constant_id = 21) const float MICROPOLAR_THETA = 2.0;

// --- Dimension handling (2D as degenerate 3D) ---
layout(constant_id = 30) const uint  DIMENSION          = 2u;          // 2 or 3
layout(constant_id = 31) const uint  NEIGHBOR_Z_RANGE   = 0u;          // 0 for 2D, 1 for 3D

// --- Kernel normalization. Python-side precomputed including h^DIM factor. ---
// Wendland C4 (support radius = h, NOT 2h):
//     2D coefficient: 9 / (π · H²)
//     3D coefficient: 495 / (32 · π · H³)
// Profile:    W(q) = KERNEL_COEFFICIENT · (1-q)^6 · (35/3·q² + 6q + 1)    for q ∈ [0, 1]
// Gradient:   ∇W = KERNEL_GRADIENT_COEFFICIENT · (1-q)^5 · q · (-280/3·q - 56/3) · r̂
// Note: KERNEL_GRADIENT_COEFFICIENT = KERNEL_COEFFICIENT / H (the 1/h factor from chain rule).
layout(constant_id = 32) const float KERNEL_COEFFICIENT           = 35367.765;  // 9 / (π · 0.009²)
layout(constant_id = 33) const float KERNEL_GRADIENT_COEFFICIENT  = 3929751.7;  // 9 / (π · 0.009³)

// --- Density diffusion / viscosity division-by-zero guard ---
// Used in δ-SPH density diffusion and artificial-viscosity expressions where a
// 1/(r² + ε_h²) term would otherwise blow up when two particles approach each
// other. Typical value: 0.01 · H² (Antuono et al. δ-SPH).
layout(constant_id = 40) const float EPS_H_SQUARED = 8.1e-7;  // 0.01 · 0.009²

// PST main-shift scale coefficient (Sun 2017 δ-plus empirical constant).
// Enters as:  pst_base_factor = CFL · PST_MAIN_SHIFT_COEFFICIENT · 2·h²
// which then premultiplies BOTH the main and anti accumulators inside the loop.
layout(constant_id = 41) const float PST_MAIN_SHIFT_COEFFICIENT = 0.1;

// PST anti-shift (cohesion) magnitude multiplier on top of the main scale.
// Legacy value 0.005.
layout(constant_id = 42) const float PST_ANTI_SHIFT_COEFFICIENT = 0.005;

// ----- Algorithm ablation toggles (id 43-46) -------------------------------
// All bool spec constants; with glslc -O the dead branches are DCE-removed,
// so disabling a feature actually skips its computation (not just zeroes).
//
// Convention: when a toggle is false, the associated coefficients
// (delta_coefficient, pst_main/anti, regularization tunables) are still
// spec-const-supplied but unused — case.py may pass any value.

// KCG correction: when false, density.comp / force.comp use identity for M⁻¹
// and zero for ∇ρ instead of reading correction.comp's outputs. Used for
// comparing against non-KCG SPH codebases. correction.comp still runs
// (kernel_sum drives PST blend), but its M⁻¹ / ∇ρ are ignored downstream.
layout(constant_id = 43) const bool USE_KCG_CORRECTION = true;

// Density diffusion (the δ term in dρ/dt): when false, density.comp skips the
// δ-SPH continuity diffusion term. Result is vanilla WCSPH continuity.
// DELTA_COEFFICIENT is then unused.
layout(constant_id = 44) const bool USE_DENSITY_DIFFUSION = true;

// Particle shifting technique (PST): when false, force.comp skips the shift
// computation and writes shift = 0. Predict's drift then becomes pure
// x_{n+1} = x_n + v_{n+1/2}·dt with no δ-plus correction.
// PST_MAIN_SHIFT_COEFFICIENT / PST_ANTI_SHIFT_COEFFICIENT are unused.
layout(constant_id = 45) const bool USE_PST = true;

// Defrag base-offset source: when true, defrag.comp reads from
// voxel_base_offset[] (deterministic, voxel-id ordered, requires prefix_sum
// pass to populate it first). When false, defrag.comp uses an atomicAdd on
// defrag_scratch_counter — order is non-deterministic but defrag still
// works standalone (no prefix_sum dependency). Default false; flip to true
// after prefix_sum is verified.
layout(constant_id = 46) const bool USE_PREFIX_SUM_DEFRAG = false;
// ----- end ablation toggles ------------------------------------------------

// --- Capacity / dispatch ---
layout(constant_id = 50) const uint MAX_PARTICLES_PER_VOXEL = 96u;
layout(constant_id = 51) const uint WORKGROUP_SIZE          = 128u;
layout(constant_id = 52) const uint MAX_INCOMING_PER_VOXEL  = 16u;
// OWN_POOL_SIZE = number of OWN particle slots (1..OWN_POOL_SIZE). Buffer is
// sized (OWN_POOL_SIZE + LEADING_GHOST_POOL_SIZE + TRAILING_GHOST_POOL_SIZE + 1)
// under 1-based indexing. predict / correction / density / force pad-check
// against OWN_POOL_SIZE.
layout(constant_id = 53) const uint OWN_POOL_SIZE              = 1000000u;
layout(constant_id = 54) const uint LEADING_GHOST_POOL_SIZE    = 0u;
layout(constant_id = 55) const uint TRAILING_GHOST_POOL_SIZE   = 0u;

// --- Rotor (prescribed rigid-body rotation of MATERIAL_ROTOR particles) ---
// One rotor per case. Axis is a unit vector, pivot a point on the axis, both
// in world coordinates (case.yaml `rotor:` block). The time-dependent part
// (cos/sin of the accumulated angle and the current angular velocity) is NOT
// a spec constant: the CPU writes it every step into RotorStateBuffer
// (set 3 binding 9) because the step command buffer is pre-recorded.
// Rotation sense: right-hand rule about +axis; the signed angular velocity
// comes from the rotor material's rotor_angular_velocity (rad/s).
layout(constant_id = 56) const float ROTOR_AXIS_X  = 0.0;
layout(constant_id = 57) const float ROTOR_AXIS_Y  = 0.0;
layout(constant_id = 58) const float ROTOR_AXIS_Z  = 1.0;
layout(constant_id = 59) const float ROTOR_PIVOT_X = 0.0;
layout(constant_id = 60) const float ROTOR_PIVOT_Y = 0.0;
layout(constant_id = 61) const float ROTOR_PIVOT_Z = 0.0;

// --- Multi-GPU ghost (V1 merged-buffer scheme) ---
// V1 partitions along X. The voxel_id encoding (helpers.glsl) is "x-slowest"
// so that each x-column of voxels is a contiguous voxel_id segment. Ghost
// columns (1 voxel thick on each peer-facing side) are placed at the LEADING
// and TRAILING ends of the extended voxel_id range. Per-GPU spec consts:
//
//   LEADING_GHOST_VOXEL_COUNT   = M = (leading ghost x-thickness) * NY * NZ
//   TRAILING_GHOST_VOXEL_COUNT  = N = (trailing ghost x-thickness) * NY * NZ
//
//                voxel_id = 1 ............ M  M+1 ............. T-N  T-N+1 ........ T
//                          \____________/ \________________/ \________________/
//                          leading ghost          own              trailing ghost
//                                              (extended_voxel_count = T)
//
// Endpoints of a 1D chain set the corresponding ghost count to 0:
//   leftmost  GPU: M = 0,   N = NY*NZ
//   middle    GPU: M = NY*NZ, N = NY*NZ
//   rightmost GPU: M = NY*NZ, N = 0
//
// Same code path on every GPU; only spec const values differ.
//
// V0 / dummy default of 0 makes both branches DCE-friendly when shader is
// run in single-GPU mode.
layout(constant_id = 80) const uint LEADING_GHOST_VOXEL_COUNT  = 0u;
layout(constant_id = 81) const uint TRAILING_GHOST_VOXEL_COUNT = 0u;

// ============================================================================
// Scalar constants (compile-time, shared by all shaders)
// ============================================================================

// --- Material kind tags (stored inside MaterialParameters.kind) ---
const uint MATERIAL_FLUID    = 0u;
const uint MATERIAL_BOUNDARY = 1u;
const uint MATERIAL_INLET    = 2u;
const uint MATERIAL_ROTOR    = 3u;

// --- Dead / sentinel values ---
// Convention: EVERY id in the SPH pipeline is 1-based. 0 is reserved as the
// "unallocated / dead / empty" sentinel across all buffers:
//   - particle_id  ∈ [1, pool_size]      (slot 0 of all particle buffers unused)
//   - voxel_id     ∈ [1, voxel_count]    (slot 0 of all voxel buffers unused)
//   - slot entries ∈ [1, pool_size]      (inside_particle_index stores particle_id directly)
// Zero-init of buffers therefore naturally means "all slots empty / all particles dead".
const uint VOXEL_ID_DEAD         = 0u;
const uint INSIDE_SLOT_EMPTY     = 0u;
const uint PARTICLE_ID_NONE      = 0u;

// ============================================================================
// Descriptor set 0 — Particle SoA (own + ghost merged in V1)
// ----------------------------------------------------------------------------
// Per-particle persistent state. All particles (fluid / boundary / inlet /
// rotor) live in this unified pool, distinguished by material[pid].
//
// V1 PID LAYOUT (mirrors set 1 voxel layout: leading | own | trailing):
//   [1, LEADING_GHOST_POOL_SIZE]                                  = leading ghost
//   [LEADING_GHOST_POOL_SIZE+1, LEADING+OWN]                      = own
//   [LEADING+OWN+1, LEADING+OWN+TRAILING_GHOST_POOL_SIZE]         = trailing ghost
//
// Buffer total size = LEADING_GHOST_POOL_SIZE + OWN_POOL_SIZE +
//                     TRAILING_GHOST_POOL_SIZE + 1 (slot 0 = PARTICLE_ID_NONE).
// End-of-chain GPUs have LEADING = 0 or TRAILING = 0; the corresponding range
// is empty and own collapses to [1, OWN_POOL_SIZE], matching V0.
//
// Active region per sub-pool:
//   Leading / trailing ghost: front-packed by atomic-append in ghost_send;
//     active = first ghost_recv_*_count slots after transport. Remainder is
//     never referenced by inside_particle_index, so stale content is invisible.
//   Own: V0 defrag periodically packs alive particles to the front; between
//     defrags alive particles can sit anywhere in own range, with dead slots
//     marked by voxel_id=0 sentinel.
//
// predict / correction / density / force dispatch over [own_first_pid(),
// own_last_pid()] (see helpers.glsl). ghost_send writes own's boundary into
// the ghost-pid sub-range matching the send direction; transport overwrites
// with peer's data; install_migrations.comp finalizes. Hot-path neighbour
// iteration reads set 0 uniformly — own and ghost neighbours are
// indistinguishable after install.
//
// Path 2 / migration semantics (V1.0a):
// Each ghost-pid slot's position_voxel_id.w encodes its purpose, set by
// sender's ghost_send.comp:
//   * voxel_id in receiver's GHOST voxel range  → REPLICA: used by hot
//     kernels via ghost voxel's inside_particle_index.
//   * voxel_id in receiver's OWN voxel range    → MIGRATION: install_migrations
//     copies the slot's 9 fields into the own pid range (end-allocated from
//     own_last_pid via migration_install_count) and registers the new own_pid
//     in own voxel's inside_particle_index.
//   * voxel_id == 0  → dead/consumed slot (skip). install_migrations sets this
//     after consuming a migration so subsequent passes don't double-install.
//
// Sender packs both replicas and migrations into the same ghost-pid range using
// the same byte layout and the same GHOST_VOXEL_ID_OFFSET_TO_RECEIVER spec
// const (the offset cancellation between own boundary↔peer ghost and own
// ghost↔peer own makes one offset suffice).
// ============================================================================

layout(std430, set = 0, binding = 0) buffer PositionVoxelIdBuffer {
    // (x, y, z, voxel_id_as_float)
    // voxel_id is 1-based; 0 marks dead/uninitialized.
    // Decode: uint voxel_id = uint(round(position_voxel_id[pid].w))
    vec4 position_voxel_id[];
};

layout(std430, set = 0, binding = 1) buffer DensityPressureBuffer {
    // (ρ, P)  canonical density / pressure per particle.
    //
    // density.comp writes new values into `density_pressure_scratch` (binding 2
    // below). The simulator's step cmd then issues vkCmdCopyBuffer
    // scratch → primary inside the same submission so that by force.comp's
    // neighbor loop this binding already holds ρ_{n+1}, P_{n+1}. correction
    // and force only ever read this buffer; they never touch scratch.
    vec2 density_pressure[];
};

layout(std430, set = 0, binding = 2) buffer DensityPressureScratchBuffer {
    // (ρ, P)  transient scratch slot for density.comp's writes.
    //
    // Only density.comp writes here. The simulator's step cmd then copies
    // this back into binding 1 immediately after the dispatch, so scratch
    // contents are stale outside that ~µs window.
    vec2 density_pressure_scratch[];
};

layout(std430, set = 0, binding = 3) buffer VelocityMassBuffer {
    // (vx, vy, vz, mass)
    // Leapfrog scheme: .xyz holds v_{n+1/2} (half-step velocity) between steps.
    // The predict kernel updates it via a full-step kick: v_{n+1/2} = v_{n-1/2} + a_n * dt.
    vec4 velocity_mass[];
};

layout(std430, set = 0, binding = 4) buffer AccelerationBuffer {
    // (ax, ay, az, reserved)
    // w slot reserved (previously used as temperature — now in ExtensionFieldsBuffer).
    vec4 acceleration[];
};

layout(std430, set = 0, binding = 5) buffer ShiftBuffer {
    // (shift_x, shift_y, shift_z, reserved)
    // w slot reserved.
    vec4 shift[];
};

layout(std430, set = 0, binding = 6) buffer MaterialBuffer {
    // group_id indexing into MaterialParametersBuffer (set 3 binding 7)
    uint material[];
};

layout(std430, set = 0, binding = 7) buffer CorrectionInverseBuffer {
    // Symmetric 3×3 M⁻¹, packed into 2 vec4 per particle:
    //   correction_inverse[pid*2]     = (m00, m11, m22, m01)
    //   correction_inverse[pid*2 + 1] = (m02, m12, _, _)
    vec4 correction_inverse[];
};

layout(std430, set = 0, binding = 8) buffer DensityGradientKernelSumBuffer {
    // (∇ρ.x, ∇ρ.y, ∇ρ.z, kernel_sum)
    // density_gradient:  Σ_j V_j · (ρ_j - ρ_i) · ∇W_ij
    // kernel_sum:        Σ_j V_j · W_ij
    vec4 density_gradient_kernel_sum[];
};

layout(std430, set = 0, binding = 9) buffer ExtensionFieldsBuffer {
    // Per-particle scalar fields not core to V0 SPH physics. Reserved slots
    // for future / debug use. V0 pipeline reads and writes nothing here.
    //   .xyz : ROTOR particles only — reference (initial) position used by
    //          predict.comp's rigid-body update (2026-09-25). Uploaded from
    //          the initial positions for every particle; other kinds ignore it.
    //   .w   : reserved (future: species concentration)
    vec4 extension_fields[];
};

// binding 10 reserved for GlobalIdBuffer (FTLE / Lagrangian tracking)

// ============================================================================
// Descriptor set 1 — Voxel cell structures (own + ghost merged in V1)
// ----------------------------------------------------------------------------
// V1 VOXEL_ID LAYOUT (extended grid, x-slowest encoding):
//   [1, M]                            = leading ghost  (M = LEADING_GHOST_VOXEL_COUNT)
//   [M+1, T - N]                      = own
//   [T - N + 1, T]                    = trailing ghost (N = TRAILING_GHOST_VOXEL_COUNT)
// where T = GRID_DIMENSION_X * GRID_DIMENSION_Y * GRID_DIMENSION_Z.
// Per-voxel buffers below are sized (T + 1) in the count dimension.
//
// predict.comp atomic-appends new-voxel arrivals into incoming_particle_index
// for any new voxel_id (own OR ghost — the atomicAdd target is whatever voxel
// the particle just drifted to). update_voxel.comp filters by is_own_voxel
// and only rebuilds own voxel inside lists. ghost voxel incoming_particle_index
// entries are intentionally LEFT for ghost_send.comp, which walks them as the
// source of MIGRATION packets (own particles that drifted across the partition
// during this step's predict). Receiver's install_migrations.comp atomic-
// appends the new own_pid into the receiver's OWN voxel inside_particle_index
// for migration arrivals. Replicas land in receiver's ghost voxel
// inside_particle_index directly via ghost_send.
// Hot-path neighbour iteration reads inside_particle_count uniformly —
// own and ghost voxels look identical once populated.
//
// Slot values inside the flat `*_particle_index` buffers store 1-based
// particle_ids directly (no +1 encoding); 0 is the INSIDE_SLOT_EMPTY sentinel.
// ============================================================================

layout(std430, set = 1, binding = 0) buffer InsideParticleCountBuffer {
    // Indexed directly by 1-based voxel_id. Size = voxel_count + 1, slot 0 unused.
    uint inside_particle_count[];
};

layout(std430, set = 1, binding = 1) buffer IncomingParticleCountBuffer {
    // predict writes atomic-append count here; update_voxel consumes and resets to 0.
    //
    // OVERFLOW contract: on atomic-append overflow (slot ≥ MAX_INCOMING_PER_VOXEL),
    // predict still increments this counter but does NOT write into
    // incoming_particle_index (to avoid OOB). So after predict, this count MAY
    // exceed MAX_INCOMING_PER_VOXEL. update_voxel MUST clamp its iteration via
    // `min(count, MAX_INCOMING_PER_VOXEL)` before reading incoming_particle_index.
    uint incoming_particle_count[];
};

layout(std430, set = 1, binding = 2) buffer InsideParticleIndexBuffer {
    // flat [voxel_id * MAX_PARTICLES_PER_VOXEL + slot]; slot stores 1-based particle_id.
    // Total size = (voxel_count + 1) * MAX_PARTICLES_PER_VOXEL.
    //
    // INVARIANT 1 (contiguous packing): for any voxel, valid particle_ids occupy
    //   slots [0, inside_particle_count[voxel_id]); all trailing slots are 0
    //   (INSIDE_SLOT_EMPTY). update_voxel MUST produce this layout; neighbor loops
    //   rely on it for the early `break` on empty slots.
    //
    // INVARIANT 2 (no inlet/dead): inlet particles and dead particles (voxel_id=0)
    //   never appear here. Enforced by predict.comp (never atomic-appends inlet or
    //   particles whose drift lands outside domain) and by initial voxelization
    //   (skip inlet when populating at t=0). Neighbor loops therefore do not need
    //   to filter neighbors by kind.
    uint inside_particle_index[];
};

layout(std430, set = 1, binding = 3) buffer IncomingParticleIndexBuffer {
    // flat [voxel_id * MAX_INCOMING_PER_VOXEL + slot]; slot stores 1-based particle_id.
    // Same inlet/dead exclusion invariant as inside_particle_index.
    uint incoming_particle_index[];
};

layout(std430, set = 1, binding = 4) buffer VoxelBaseOffsetBuffer {
    // Exclusive prefix sum of inside_particle_count[]:
    //   voxel_base_offset[v] = Σ inside_particle_count[0..v-1]
    //
    // Populated by prefix_sum.comp; consumed by defrag.comp as the
    // deterministic destination-SoA base index for voxel v's particles.
    // When defrag's USE_PREFIX_SUM_DEFRAG spec constant is false, this buffer is
    // declared but unread — DCE removes the load from defrag.spv.
    // Other shaders never reference it, so glslc -O strips it from their
    // SPIR-V entirely.
    //
    // Size = (voxel_count + 1); slot 0 unused for symmetry with other
    // voxel buffers.
    uint voxel_base_offset[];
};
// binding 5 reserved

// ============================================================================
// Descriptor set 2 — UNUSED in V1 merged-buffer scheme.
//
// V1 stores ghost particles inside set 0's high-pid range and ghost voxel
// structures inside set 1's leading/trailing voxel-id range. There is no
// separate ghost SoA. Pipelines that don't need set 2 simply omit it from
// their pipeline_layout.
// ============================================================================

// ============================================================================
// Descriptor set 3 — Global status, transport, material parameters, diagnostics
// ============================================================================

layout(std430, set = 3, binding = 0) buffer GlobalStatusBuffer {
    uint  alive_particle_count;
    float maximum_velocity;
    // Per-kernel inside/incoming-list overflow diagnostics. Each kernel that
    // can drop a particle on a full voxel slot has its own counter + sample-vid
    // pair so root-cause attribution is unambiguous on host-side readback.
    uint  overflow_inside_count;          // update_voxel.comp (own incoming → inside, full)
    uint  overflow_incoming_count;        // predict.comp (atomic-append into incoming, full)
    uint  first_overflow_voxel_inside;    // update_voxel.comp sample vid
    uint  first_overflow_voxel_incoming;  // predict.comp sample vid
    uint  correction_fallback_count;      // particles whose M_inv fell to identity
    uint  overflow_ghost_count;           // ghost_send sender-side ghost-pool overflow
    // V1 ghost transport counters. Zeroed at step start (vkCmdFillBuffer
    // before ghost_send dispatch). ghost_send writes send_*; vkCmdCopyBuffer
    // overwrites recv_* with peer's send_*; install_migrations reads recv_*
    // to know dispatch range.
    uint  ghost_send_leading_count;
    uint  ghost_send_trailing_count;
    uint  ghost_recv_leading_count;
    uint  ghost_recv_trailing_count;
    // V1.0a migration installation (end-allocate, strategy 1):
    //   migration_install_count: atomic counter shared by leading + trailing
    //     install_migrations dispatches. Reset only after each defrag dispatch
    //     (NOT every step). own_pid for slot N = own_last_pid() - N.
    //   overflow_install_tail: bumped when end-allocate would overlap with
    //     post-defrag alive region (own pool tail exhausted before next defrag).
    //   overflow_install_inside: bumped when an arriving migration finds the
    //     receiver own voxel's inside_particle_index already at MAX. The newly
    //     allocated own_pid is rolled back (zeroed) and the migration dropped.
    //   first_overflow_voxel_install: sample vid for the inside-list overflow on
    //     the install path; separate from update_voxel's sample so post-mortem
    //     can distinguish the two failure modes.
    uint  migration_install_count;
    uint  overflow_install_tail;
    uint  overflow_install_inside;
    uint  first_overflow_voxel_install;   // total 16 uint = 64 B, one cache line
};

layout(std430, set = 3, binding = 1) buffer OverflowLogBuffer {
    uint  log_event_count;              // atomic counter, modulo ring size
    uint  reserved_log_pad_0;
    uint  reserved_log_pad_1;
    uint  reserved_log_pad_2;
    uvec4 log_events[];                 // ring buffer: (voxel_id, step, kind, lost_particle_id)
};

// Inlet template: one full particle state per template slot. The inlet spawn
// kernel copies from a slot into a free main-pool slot when triggered.
struct InletTemplateEntry {
    vec4 position_voxel_id;
    vec4 velocity_mass;
    uint material;
    uint reserved_0;
    uint reserved_1;
    uint reserved_2;
};

layout(std430, set = 3, binding = 2) buffer InletTemplateBuffer {
    InletTemplateEntry inlet_template[];
};

layout(std430, set = 3, binding = 3) buffer DispatchIndirectBuffer {
    // Consumed by vkCmdDispatchIndirect. A small updater kernel sets x when
    // alive_particle_count changes.
    uint dispatch_indirect_x;
    uint dispatch_indirect_y;
    uint dispatch_indirect_z;
};

layout(std430, set = 3, binding = 4) buffer GhostOutPacketBuffer {
    // (V1+) boundary particles packed for send to peer GPU
    vec4 ghost_out_packet[];
};

layout(std430, set = 3, binding = 5) buffer GhostInStagingBuffer {
    // (V1+) receive destination from peer GPU, before unpack into set 2
    vec4 ghost_in_staging[];
};

layout(std430, set = 3, binding = 6) buffer DiagnosticBuffer {
    // (optional, debug builds only) curl, FTLE, vorticity for visualization
    vec4 diagnostic[];
};

// ----------------------------------------------------------------------------
// MaterialParametersBuffer — per-group SPH parameters.
// MaterialBuffer[particle] stores a group_id that indexes here.
// Struct is 48 B (12 × 4 B); for 16 groups the entire buffer is 768 B
// and lives in L1 after first read.
// ----------------------------------------------------------------------------
struct MaterialParameters {
    // --- Core (V0 used) -----------------------------------------------------
    uint  kind;                   // MATERIAL_FLUID / BOUNDARY / INLET / ROTOR
    float rest_density;
    float viscosity;
    float eos_constant;           // c0² · rest_density / power_parameter
    float smoothing_length;       // V0: same as global SMOOTHING_LENGTH; future: per-group for multi-res
    float radius;
    float volume;
    float rotor_angular_velocity; // Used by ROTOR kind only; 0 for others. Assumed about z-axis in V0.

    // --- Reserved for __future__ (V0 unused) --------------------------------
    float viscosity_transfer;     // micropolar mass transfer
    float viscosity_rotation;     // micropolar rotation
    uint  reserved_material_0;
    uint  reserved_material_1;
};  // 48 B total

layout(std430, set = 3, binding = 7) buffer MaterialParametersBuffer {
    MaterialParameters material_parameters[];
};

// ----------------------------------------------------------------------------
// RotorStateBuffer — per-step rigid-body state of the (single) rotor, written
// by the CPU (host-visible, coherent) before every step submission.
//   cos_theta / sin_theta : accumulated rotation angle theta(t_{n+1}) about
//                           ROTOR_AXIS through ROTOR_PIVOT (angle computed in
//                           float64 on the CPU from the ramped angular
//                           velocity; never accumulated in float32 on GPU)
//   angular_velocity      : omega(t_{n+1}), signed, rad/s (ramp applied)
//   time                  : t_{n+1}, informational
// predict.comp: ROTOR particles get
//   x_{n+1} = pivot + R(theta) (x_ref - pivot),   v = omega * axis x (x_{n+1} - pivot)
// where x_ref = extension_fields.xyz (uploaded = initial position; carried
// through defrag like every other set-0 field).
// ----------------------------------------------------------------------------
layout(std430, set = 3, binding = 9) buffer RotorStateBuffer {
    float rotor_cos_theta;
    float rotor_sin_theta;
    float rotor_angular_velocity_now;
    float rotor_time;
};

layout(std430, set = 3, binding = 8) buffer DefragScratchCounterBuffer {
    // Single uint, atomic-incremented by defrag.comp when USE_PREFIX_SUM_DEFRAG=false.
    // CPU resets to 0 before each defrag dispatch (vkCmdFillBuffer).
    // When USE_PREFIX_SUM_DEFRAG=true, declared but unwritten — DCE removes it
    // from defrag.spv. Other shaders never reference it; glslc -O strips
    // it from their SPIR-V entirely.
    uint defrag_scratch_counter;
};

#endif  // SPH_COMMON_GLSL_INCLUDED
