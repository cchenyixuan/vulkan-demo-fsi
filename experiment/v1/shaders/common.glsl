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
//   54 - 55  : ghost pool sizes
//   56 - 61  : rotor axis / pivot (2026-09-25)
//   62       : neighbour-list capacity (2026-09-25)
//   63 - 71  : scalar transport (2026-09-27)
//   72 - 79  : reserved
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
layout(constant_id = 14) const float REGULARIZATION_XI                    = 0.01;
layout(constant_id = 15) const float REGULARIZATION_DETERMINANT_THRESHOLD = 1e-4;
layout(constant_id = 16) const float REGULARIZATION_MAX_FROBENIUS_NORM    = 10.0;

// --- Gravity ---
layout(constant_id = 17) const float GRAVITY_X = 0.0;
layout(constant_id = 18) const float GRAVITY_Y = 0.0;
layout(constant_id = 19) const float GRAVITY_Z = 0.0;

// --- Background pressure (2026-09-29) ---
// Constant p_b added to the Tait pressure of every particle (fluid and solid):
//   P = eos_constant ((rho / rho0)^gamma - 1) + p_b.
// A constant has no gradient, so the flow of a CLOSED single-phase domain is
// unchanged in the continuum; numerically p_b keeps the pressure positive
// where it would fall below zero (behind the blades, in vortex cores), so
// force.comp stays on the symmetric, pairwise momentum-conserving form
// (P_i + P_j) instead of the TIC form (P_j - P_i). Must be 0 with a free
// surface. case.yaml: physics.background_pressure (default 0).
layout(constant_id = 34) const float BACKGROUND_PRESSURE = 0.0;

// --- Pressure of the solid particles seen by the fluid (2026-09-29) ---
// SOLID_PRESSURE_MODE (case.yaml numerics.solid_pressure):
//   0 "increment"  : the pressure density.comp stores for the solid particle,
//                    P_s = EOS(rho0 + dt (d rho / dt)_s) (the original code).
//   1 "mirror"     : pairwise mirror. In the pair (fluid i, solid j) the solid
//                    pressure is the fluid particle's own pressure continued to
//                    the solid particle with the body force in the frame of the
//                    solid,
//                        P_j = P_i + rho_i (g - a_j) . (x_j - x_i),
//                    a_j = acceleration of the solid particle (0 for a wall,
//                    centripetal for the rotor). The fluid-solid pair always
//                    uses the symmetric form (P_i + P_j), also for P_i < 0.
//   2 "mirror_tic" : as 1, but the fluid-solid pair follows the TIC switch of
//                    the fluid particle like a fluid-fluid pair.
//   3 "accumulate" : the solid particle integrates its continuity equation in
//                    time like a fluid particle (no reset to rho0), with rho0 as
//                    the lower bound; P_s = EOS(rho_s). The dynamic boundary
//                    condition of Crespo et al. (2007) as in DualSPHysics: the
//                    solid keeps the pressure it has built up, so it carries a
//                    static load.
//   4 "extrapolate": P_s is the pressure of the fluid neighbours continued to
//                    the solid particle (Adami et al. 2012),
//                        P_s = [ sum_f P_f W_sf + (g - a_s) . sum_f rho_f (x_s - x_f) W_sf ]
//                              / sum_f W_sf,
//                    with the fluid pressure of the previous step; the stored
//                    density is the one the EOS gives for P_s.
// Modes 3 and 4 only change what density.comp stores for a solid particle;
// force.comp reads the stored pressure as in mode 0.
// The density diffusion of density.comp is not changed by the mode: a solid
// neighbour keeps rho0 there (see the note in density.comp).
// USE_SOLID_REACTION_FORCE (numerics.solid_reaction_force; forced on by the
// mirror modes): the acceleration written for a ROTOR or BOUNDARY particle is
// the reaction of what its fluid neighbours receive from it,
//     a_j = - sum_i (m_i / m_j) a_(i <- j) + g,
// evaluated with the fluid particle's matrix, density and pressure, so the
// force read back on a solid is exactly minus the force on the fluid.
layout(constant_id = 35) const uint SOLID_PRESSURE_MODE = 0u;
layout(constant_id = 36) const bool USE_SOLID_REACTION_FORCE = false;
// SOLID_PRESSURE_OFFSET p_w (numerics.solid_pressure_offset, Pa; mirror modes
// only): constant added to the mirrored solid pressure,
//     P_j = P_i + rho_i (g - a_j) . (x_j - x_i) + p_w,
// so the fluid-solid pair pressure is 2 P_i + p_w. It acts on fluid-solid pairs
// only: a repulsive layer along every solid surface (the potential
// p_w * (smoothed solid volume fraction)), none of the background pressure's
// effect on fluid-fluid pairs. A mirrored solid has no stiffness of its own;
// without p_w the fluid enters the walls wherever its pressure is negative
// (lid-driven cavity, 2026-09-29).
layout(constant_id = 37) const float SOLID_PRESSURE_OFFSET = 0.0;

// --- Density diffusion: gradient term (2026-09-29) ---
// numerics.density_diffusion_gradient_term. psi_ij of the delta-SPH diffusion
// with the renormalised density gradients (Antuono et al. 2010),
//     psi_ij = (rho_j - rho_i) - 1/2 (<grad rho>_i + <grad rho>_j) . (x_j - x_i),
// which vanishes for a linear density field also where the neighbourhood is
// one-sided. Without it (the V0 simplification, default) the diffusion flattens
// a hydrostatic density profile at every wall.
layout(constant_id = 38) const bool USE_DENSITY_DIFFUSION_GRADIENT_TERM = false;

// --- Particle shift near solids (2026-09-29) ---
// numerics.pst_near_solid: 0 "full" (default), 1 "tangential": the component of
// the shift along the local solid normal (direction of sum_solid V_j grad W_ij)
// is removed for fluid particles with solid neighbours.
layout(constant_id = 39) const uint PST_NEAR_SOLID_MODE = 0u;

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

// Per-step neighbour list (2026-09-25): when true, build_neighbor_list.comp
// runs after update_voxel and fills NeighborCountBuffer / NeighborListBuffer
// (set 1, bindings 5-6); correction / density / force then iterate that list
// instead of scanning the 27 surrounding voxels. When false the kernels use
// the original voxel scan and never touch the list buffers (DCE).
// Default false: measured slower than the voxel scan on sorted particles
// (see log/2026-09-25_neighbor-list.md); kept as an opt-in experiment.
layout(constant_id = 47) const bool USE_NEIGHBOR_LIST = false;

// --- Pair correction of the fluid-fluid forces (2026-09-30) ---
// numerics.pair_correction. force.comp corrects the kernel gradient of
// the pair (i, j) with the matrix of particle i: grad W~_ij = M_i^-1 grad W_ij.
// The force of j on i and the force of i on j then differ by
//     m V (P_i + P_j) / rho (M_i^-1 - M_j^-1) grad W_ij,
// a net force on the pair that grows with the absolute pressure and with the
// disorder of the particles (angular momentum budget of the tank, 2026-09-29:
// with gravity the fluid-fluid pairs remove 37 to 87 % of the rotor's input).
// Modes, for FLUID-FLUID pairs only (B = M^-1):
//   0 "own"    : B_i for everything (the original code).
//   1 "mean"   : grad W~_ij = 1/2 (B_i + B_j) grad W_ij for the pressure and the
//                viscous force, pair volume 2 m / (rho_i + rho_j) in the viscous
//                force. Pressure: (P_i + P_j) 1/2 (B_i + B_j) grad W_ij.
//   2 "reverse": reverse kernel gradient correction (Zhang, Adams, Hu 2025,
//                CMAME 433:117484): pressure (P_i B_j + P_j B_i) grad W_ij
//                    = P_i (B_i + B_j) grad W_ij + (P_j - P_i) B_i grad W_ij,
//                the second part is the first-order consistent gradient of
//                particle i. Pairs in the TIC form (P_j - P_i) keep B_i, which
//                is this second part alone. Viscous force as in mode 1.
// In modes 1 and 2 the pressure forces of a pair in the (P_i + P_j) form are
// equal and opposite, and so are the viscous forces. Fluid-solid pairs, the
// density equation, the vorticity, the shift and the scalars keep B_i.
layout(constant_id = 48) const uint PAIR_CORRECTION_MODE = 0u;
// ----- end ablation toggles ------------------------------------------------

// --- Thin plates wetted on both sides (2026-09-30) ---
// Side-aware mirror treatment of the blades, the Rushton disk and the baffles,
// see thin_plates.glsl. The code exists only in the builds compiled with
// -DWITH_THIN_PLATES=1 (density_plates, force_plates, force_scalar_plates;
// compile_shaders_v1.py); the simulator uses them for cases with a
// `thin_plates:` block. The default builds are byte-identical to those before
// this change. ids 73 and 76 were used by the surface integral form (commit
// 441deb4) and are free again. Builds: correction_plates, density_plates,
// force_plates, force_scalar_plates.
#ifndef WITH_THIN_PLATES
#define WITH_THIN_PLATES 0
#endif
#if WITH_THIN_PLATES
layout(constant_id = 49) const uint  THIN_PLATE_COUNT       = 0u;
layout(constant_id = 72) const uint  THIN_PLATE_GROUP_COUNT = 0u;
// numerics.thin_plate_dashpot: beta of the mirrored wall pressure
//   P_j = P_i + rho_i (g - a_w) . (x_j - x_i) + beta rho_i c0 (v_i - v_w) . n_out
layout(constant_id = 74) const float THIN_PLATE_DASHPOT = 0.0;
// numerics.thin_plate_viscosity: viscous force between a fluid particle and
// the wall dummies of a plate (false: free slip)
layout(constant_id = 75) const bool  USE_THIN_PLATE_VISCOSITY = true;
#endif

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

// --- Neighbour list capacity (entries per particle). Python sets it to the
// closest-packing bound for the support sphere (Case.neighbors_max_estimate,
// 160 at h/dx = 3 in 3D) unless case.yaml overrides. Overflow is counted in
// GlobalStatusBuffer.overflow_neighbor_count and the extra neighbours dropped.
layout(constant_id = 62) const uint MAX_NEIGHBORS = 160u;

// --- Scalar transport (2026-09-27) ------------------------------------------
// Scalar fields (tracer, substrate, oxygen, biomass, ...) carried by FLUID
// particles. The particle motion does the advection; force.comp adds an SPH
// Laplacian diffusion and the δ-plus shift correction; predict.comp applies
// the increment. Fields are packed four per vec4:
//     scalar[pid * SCALAR_VEC4_COUNT + v].c  holds field  4 v + c
// SCALAR_VEC4_COUNT = 0 (default) switches the feature off completely: every
// scalar code path is a zero-trip loop or a branch on a false spec constant,
// removed when the pipeline is specialized, and the scalar buffers exist only
// as 16 B placeholders that no kernel touches. At most MAX_SCALAR_VEC4 vec4.
layout(constant_id = 63) const uint  SCALAR_VEC4_COUNT = 0u;
// Smagorinsky sub-grid diffusivity for the scalars (Smagorinsky 1963):
//     nu_t = (C_s Delta)^2 |S|,   |S| = sqrt(2 S:S),   D_t = nu_t / Sc_t
// S is the rate-of-strain tensor of the M-corrected SPH velocity gradient,
// computed in density.comp. nu_t enters ONLY the scalar diffusion (fields
// whose SGS weight is 1), NOT the momentum equation.
layout(constant_id = 64) const bool  USE_SCALAR_SGS = false;
layout(constant_id = 65) const float SGS_LENGTH_SQUARED = 0.0;          // (C_s Delta)^2 in m^2
layout(constant_id = 66) const float INVERSE_TURBULENT_SCHMIDT = 1.0;   // 1 / Sc_t
// Ablation toggles for the two numerical ingredients of the scalar update:
//   shift correction : C += shift . grad C. The δ-plus shift moves particles
//                      relative to the fluid; without this term the shift
//                      itself would transport scalar artificially.
//   compensated sum  : Kahan-compensated accumulation of the per-step
//                      increment in predict.comp. At the small WCSPH time step
//                      a bulk increment can be below half an ulp of C and would
//                      be rounded away every step in plain float32.
// USE_SCALAR_SHIFT_CORRECTION is OFF by default in case.py (2026-09-27): the
// Taylor term is neither conservative nor bounded at sharp fronts; see
// ScalarsConfig in utils/sph/case.py and log/2026-09-27_scalar-transport.md.
layout(constant_id = 67) const bool  USE_SCALAR_SHIFT_CORRECTION = false;
layout(constant_id = 68) const bool  USE_SCALAR_COMPENSATED_SUM  = true;
// Tracer pulses (ScalarInjectionBuffer) are compiled into predict.comp only
// for cases that declare at least one injection.
layout(constant_id = 69) const bool  USE_SCALAR_INJECTION = false;
// Local bounds limiter, active only together with the shift correction: the
// updated value C_i + ΔC_i is clipped to the range spanned by C_i and its
// FLUID neighbours (discrete maximum principle). The diffusion part alone is
// monotone; the shift correction (a first-order Taylor interpolation without
// limiter) is not, and at a sharp front, e.g. the edge of a tracer pulse, it
// produced values of -0.37 and 1.46 for a 0 / 1 tracer within 1 s in the 4 mm
// tank (2026-09-27). Smooth fields are not affected (a linear field
// interpolated within the kernel stays in range).
layout(constant_id = 70) const bool  USE_SCALAR_BOUNDS_LIMITER = true;
// Number of declared fields (<= 4 * SCALAR_VEC4_COUNT). force.comp multiplies
// the scalar vec4 by scalar_component_mask() so that the unused components of
// the last vec4 are compile-time zeros after specialization and their
// arithmetic and registers disappear (7 % of force's time with one field).
layout(constant_id = 71) const uint  SCALAR_FIELD_COUNT = 0u;

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

// --- Scalar transport capacities (2026-09-27) ---
// Register arrays in force.comp are sized with MAX_SCALAR_VEC4; the loops run
// to SCALAR_VEC4_COUNT (<= MAX_SCALAR_VEC4, checked in Python) so the unused
// array elements disappear after specialization.
const uint MAX_SCALAR_VEC4     = 3u;   // up to 12 scalar fields
const uint MAX_INJECTION_SLOTS = 4u;   // simultaneously active tracer pulses

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
    //   correction_inverse[pid*2 + 1] = (m02, m12, d / tr(M), fluid flag)
    // m.. are the entries of the REGULARIZED inverse (M + ξ I)⁻¹ used by every
    // momentum / density term. The last two slots (2026-09-27) are used only
    // by the scalar transport in force.comp: d / tr(M) with tr(M) the trace of
    // the unregularized M = Σ_j V_j (x_j - x_i) ⊗ ∇W_ij (Laplacian
    // normalisation without the ξ shift), and 1 / 0 for FLUID / other kinds
    // (written only when SCALAR_VEC4_COUNT > 0, else 0).
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
    //   .w   : reserved (scalars live in ScalarBuffer, binding 11)
    vec4 extension_fields[];
};

layout(std430, set = 0, binding = 10) buffer ParticleUidBuffer {
    // Persistent particle identity (2026-09-27): the pid the particle had at
    // upload (1-based). Defrag renumbers pids but carries this field, so a
    // particle's trajectory and scalar history can be followed across defrags
    // (lifelines, FTLE). Never read by the physics kernels.
    uint particle_uid[];
};

layout(std430, set = 0, binding = 11) buffer ScalarBuffer {
    // Scalar fields, SCALAR_VEC4_COUNT vec4 per particle (see scalar_index()).
    // Specific (per unit mass) values: advection leaves a particle's value
    // unchanged, and the conserved amount of field k is sum_i m_i C_ik.
    // Meaningful on FLUID particles only; solids hold 0 and are excluded from
    // every scalar sum (zero-flux walls). Written only by predict.comp (and by
    // the host for initial conditions).
    vec4 scalar[];
};

layout(std430, set = 0, binding = 12) buffer ScalarCompensationBuffer {
    // Kahan compensation of the running sum in ScalarBuffer (the low-order
    // part lost in the last float32 addition, with the sign convention of
    // Kahan 1965). Same layout as ScalarBuffer; predict.comp only.
    vec4 scalar_compensation[];
};

layout(std430, set = 0, binding = 13) buffer ScalarDeltaBuffer {
    // Increment for the next predict.comp, written by force.comp:
    //     dt * (dC/dt)_diffusion  +  shift . grad C
    // Same layout as ScalarBuffer. Kept apart from ScalarBuffer so that force
    // never writes a value that another invocation of the same dispatch reads.
    // Carried through defrag (defrag runs between force and the next predict).
    vec4 scalar_delta[];
};

layout(std430, set = 0, binding = 14) buffer TurbulentViscosityBuffer {
    // Smagorinsky nu_t (m^2/s) of FLUID particles, written by density.comp and
    // read by force.comp in the same step. The kernels would not need it
    // carried through defrag (it is recomputed before force reads it), but it
    // is carried anyway so that a host readback after a defrag stays aligned
    // with the particles (found 2026-09-27: without it the values read after
    // a defrag belong to other particles). Touched only when USE_SCALAR_SGS
    // (a 16 B placeholder otherwise).
    float turbulent_viscosity[];
};

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

// ---- Per-step neighbour list (2026-09-25) ----------------------------------
// Written by build_neighbor_list.comp once per step (after update_voxel) and
// once at bootstrap (after initialize_voxelization); read by correction /
// density / force when USE_NEIGHBOR_LIST. Rebuilt from scratch every step,
// so defrag (which renumbers particles after force) never has to carry it.
layout(std430, set = 1, binding = 5) buffer NeighborCountBuffer {
    // Indexed by 1-based particle_id: number of valid entries in this
    // particle's list, already clamped to MAX_NEIGHBORS. 0 for dead / inlet.
    uint neighbor_count[];
};

layout(std430, set = 1, binding = 6) buffer NeighborListBuffer {
    // TRANSPOSED layout: entry k of particle pid lives at
    //     neighbor_list[k * neighbor_list_stride() + pid]
    // (stride = pool capacity incl. slot 0, see helpers.glsl). At loop index
    // k the consecutive pids of one warp therefore read one contiguous run
    // of uints (coalesced), instead of MAX_NEIGHBORS*4 B apart.
    // Entries are 1-based particle_ids of every particle j != i with
    // 1e-24 <= |x_i - x_j|^2 < h^2, in the same order the 27-voxel scan
    // visits them (voxel z-slowest / x-fastest, slot order inside a voxel),
    // so list-mode and voxel-mode sums agree to rounding.
    uint neighbor_list[];
};

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
    uint  first_overflow_voxel_install;
    // build_neighbor_list.comp (2026-09-25): particles whose true neighbour
    // count exceeded MAX_NEIGHBORS (their extra neighbours were dropped) and
    // a sample pid. Python raises at bootstrap / warns at run end if nonzero.
    uint  overflow_neighbor_count;
    uint  first_overflow_neighbor_pid;
    uint  reserved_status_pad_0;
    uint  reserved_status_pad_1;          // total 20 uint = 80 B
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
#if WITH_THIN_PLATES
    float rotor_angular_acceleration_now;   // d omega / dt at t_{n+1} (the ramp), rad/s^2
#endif
};

// ----------------------------------------------------------------------------
// Scalar transport parameters (2026-09-27).
//   scalar_parameters[2 v]     : molecular diffusivity D_m of fields 4v..4v+3 (m^2/s)
//   scalar_parameters[2 v + 1] : SGS weight of the same fields (1 = the field
//                                also receives nu_t / Sc_t, 0 = it does not)
// A field with D_m = 0 and weight 0 does not diffuse at all (e.g. biomass
// attached to the fluid particles); the harmonic pair mean in force.comp then
// makes its flux exactly zero.
// ----------------------------------------------------------------------------
layout(std430, set = 3, binding = 10) buffer ScalarParametersBuffer {
    vec4 scalar_parameters[];
};

// ----------------------------------------------------------------------------
// Tracer injection slots (2026-09-27). The host decides which pulses are
// active for the coming step, writes them into a host-visible staging buffer,
// and the step command buffer copies them here before predict.comp. predict
// sets field (4 target.y + target.z) to value.x on every FLUID particle whose
// new position lies inside the sphere, for as long as the slot is active.
// ----------------------------------------------------------------------------
struct ScalarInjectionSlot {
    vec4  center_radius_squared;   // xyz = centre (m), w = radius^2 (m^2)
    vec4  value;                   // x = value imposed inside the sphere
    uvec4 target;                  // x = active (0 / 1), y = vec4 index v, z = component c
};  // 48 B

layout(std430, set = 3, binding = 11) buffer ScalarInjectionBuffer {
    ScalarInjectionSlot scalar_injection[];   // MAX_INJECTION_SLOTS entries
};

#if WITH_THIN_PLATES
// ----------------------------------------------------------------------------
// ThinPlateBuffer (2026-09-30): the plates and their bounding groups.
// A group is a sphere in world coordinates holding a run of plates of one
// frame; the sphere of a rotor group is centred on the rotor axis, so it does
// not move. Rotor plates are stored at the rotor angle 0.
// ----------------------------------------------------------------------------
const uint MAX_THIN_PLATE_GROUPS = 8u;

struct ThinPlateGroup {
    vec4  centre_radius;          // bounding sphere
    uvec4 range;                  // x = first plate, y = number of plates, z = frame (0 static, 1 rotor)
};  // 32 B

struct ThinPlate {
    vec4  centre_extent_a;        // xyz centre; w = half length along axis_a (rectangle) or outer radius (annulus)
    vec4  normal_extent_b;        // xyz unit normal; w = half length along axis_b or inner radius
    vec4  axis_a_half_thickness;  // xyz unit in-plane axis a; w = half the plate thickness
    vec4  axis_b_measure;         // xyz unit in-plane axis b; w = area (3D) or length (2D) one plate particle stands for
    uvec4 flags;                  // x = shape (0 rectangle, 1 annulus), y = frame (0 static, 1 rotor)
};  // 80 B

layout(std430, set = 3, binding = 12) buffer ThinPlateBuffer {
    ThinPlateGroup thin_plate_group[MAX_THIN_PLATE_GROUPS];
    ThinPlate      thin_plate[];
};

// Reaction of the thin plates (2026-09-30), written by force.comp, read by the
// CPU. Two vec4 per particle slot:
//   [2 pid]     xyz = force the FLUID particle received from the wall dummies of
//               thin plates in this force evaluation (N, m_i a_i), w = index of
//               the plate + 1 (0: no dummy)
//   [2 pid + 1] xyz = position of the particle at this evaluation
// The load on a plate is minus the sum of these forces; the torque uses the
// positions stored here, so the records are complete in themselves (a defrag
// after the force evaluation does not reorder this buffer).
layout(std430, set = 3, binding = 13) buffer ThinPlateReactionBuffer {
    vec4 thin_plate_reaction[];
};
#endif

layout(std430, set = 3, binding = 8) buffer DefragScratchCounterBuffer {
    // Single uint, atomic-incremented by defrag.comp when USE_PREFIX_SUM_DEFRAG=false.
    // CPU resets to 0 before each defrag dispatch (vkCmdFillBuffer).
    // When USE_PREFIX_SUM_DEFRAG=true, declared but unwritten — DCE removes it
    // from defrag.spv. Other shaders never reference it; glslc -O strips
    // it from their SPIR-V entirely.
    uint defrag_scratch_counter;
};

#endif  // SPH_COMMON_GLSL_INCLUDED
