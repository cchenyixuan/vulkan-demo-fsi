"""
case.py — yaml + obj loader, Case dataclass, spec-constant assembly.

Single CPU-side hub for the V0 SPH pipeline. Everything you need to go from
``cases/<name>/case.yaml`` to a Vulkan-ready description lives here:

  - 5 block dataclasses mirroring the case.yaml top-level sections
    (``PhysicsConfig``, ``NumericsConfig``, ``RegularizationConfig``,
    ``CapacitiesConfig``, ``TimeConfig``). Each owns its intra-block validation.
  - ``MaterialEntry`` / ``ParticleSource`` resolved entities.
  - ``Case`` — the atomic unit consumed downstream. Bundles all blocks + grid
    + materials + particle sources. Owns derived ``@property`` and cross-block
    validation.
  - ``build_specialization_info(case)`` — packs the 41 spec constants into a
    bytes blob (kept in lockstep with ``shaders/sph/common.glsl``).
  - ``load_case(path)`` — single entry point: yaml + obj on disk → Case.
  - Optional blocks: ``rotor:`` (2026-09-25, RotorConfig) and ``scalars:``
    (2026-09-27, ScalarsConfig: fields, sgs, injections, probes).

Stays Vulkan-free. Imports only PyYAML, NumPy, and our two pure utilities
(``utils.sph.obj_loader`` for vertex parsing, ``utils.sph.grid`` for
bbox→origin/dimension derivation). Downstream Vulkan code consumes ``Case``
without touching disk or yaml.

V0 scope:
  - INLET kind is rejected (inlet spawn is V0+ work).
  - Single-resolution: smoothing_length / radius / volume are uniform across
    materials, all derived from case.yaml's physics block.
"""

import math
import pathlib
import struct
from dataclasses import dataclass
from typing import Callable, NamedTuple, Optional

import numpy as np
import yaml

from utils.sph.grid import compute_grid
from utils.sph.obj_loader import load_obj_vertices


# ============================================================================
# Schema versions — bumped on ANY breaking change to the corresponding yaml
# format. The loader rejects mismatches hard; case files must be migrated.
# ============================================================================

CASE_SCHEMA_VERSION = 2          # bumped: physics fields renamed to h / particle_radius
MATERIAL_SCHEMA_VERSION = 1


# ============================================================================
# Material kind tags — must match shaders/sph/common.glsl exactly.
# ============================================================================

KIND_FLUID    = 0
KIND_BOUNDARY = 1
KIND_INLET    = 2
KIND_ROTOR    = 3

_KIND_NAME_TO_CODE = {
    "fluid":    KIND_FLUID,
    "boundary": KIND_BOUNDARY,
    "inlet":    KIND_INLET,
    "rotor":    KIND_ROTOR,
}


# ============================================================================
# Leaf parameter blocks (1:1 with case.yaml top-level sections).
# Each block owns its intra-block validation in __post_init__.
# ============================================================================


def _calibrate_particle_volume(
    h: float,
    particle_radius: float,
    dimension: int,
    lattice: str = "grid",
) -> float:
    """SPH partition-of-unity calibration that targets the GPU's ``kernel_sum``.

    correction.comp accumulates ``Σ_{j != i} V_j · W(r_ij)`` (self EXCLUDED via
    the ``if (neighbor == self) continue;`` line). For an interior particle on
    a regular lattice with spacing dx = 2·particle_radius, we want:

        Σ_{j != i} V_j · W(r_ij) = 1

    With uniform V_j = V, this gives:

        V_calibrated = 1 / Σ_{j != 0} W(r_j)

    where the sum is over all integer-lattice offsets within the kernel
    support, EXCLUDING the (0,0[,0]) self term.

    `lattice` determines which lattice points are enumerated:
      - "grid": Cartesian (square in 2D, simple cubic in 3D), spacing dx
      - "hex":  hexagonal close packing (2D), face-centered cubic (3D),
                nearest-neighbor distance dx

    Match the lattice argument to your initial particle layout so the GPU's
    interior kernel_sum lands on 1.0. (Cartesian Blender export → "grid";
    hex-packed pre-processor → "hex".)
    """
    diameter = 2.0 * particle_radius
    if dimension == 2:
        coefficient = 9.0 / (math.pi * h * h)
    else:
        coefficient = 495.0 / (32.0 * math.pi * h * h * h)

    def kernel_W(distance: float) -> float:
        q = distance / h
        if q >= 1.0:
            return 0.0
        one_minus_q = 1.0 - q
        return (
            coefficient
            * (one_minus_q ** 6)
            * ((35.0 / 3.0) * q * q + 6.0 * q + 1.0)
        )

    if lattice not in ("grid", "hex"):
        raise ValueError(
            f"physics.lattice must be 'grid' or 'hex', got {lattice!r}")

    kernel_sum = 0.0

    if lattice == "grid":
        n_max = int(math.ceil(h / diameter))
        if dimension == 2:
            for i in range(-n_max, n_max + 1):
                for j in range(-n_max, n_max + 1):
                    if i == 0 and j == 0:
                        continue                       # self excluded
                    r = math.sqrt((i * diameter) ** 2 + (j * diameter) ** 2)
                    kernel_sum += kernel_W(r)
        else:
            for i in range(-n_max, n_max + 1):
                for j in range(-n_max, n_max + 1):
                    for k in range(-n_max, n_max + 1):
                        if i == 0 and j == 0 and k == 0:
                            continue
                        r = math.sqrt(
                            (i * diameter) ** 2
                            + (j * diameter) ** 2
                            + (k * diameter) ** 2
                        )
                        kernel_sum += kernel_W(r)

    else:  # lattice == "hex"
        if dimension == 2:
            # Hex 2D primitive vectors:
            #   a1 = (dx, 0)
            #   a2 = (dx/2, dx·√3/2)
            # Each interior particle sees 6 nearest neighbors at dx.
            n_iter = int(math.ceil(h / diameter)) + 2     # +2 buffer for skew
            sqrt3_half = math.sqrt(3.0) * 0.5
            for i in range(-2 * n_iter, 2 * n_iter + 1):
                for j in range(-2 * n_iter, 2 * n_iter + 1):
                    if i == 0 and j == 0:
                        continue
                    x = i * diameter + j * diameter * 0.5
                    y = j * diameter * sqrt3_half
                    r = math.sqrt(x * x + y * y)
                    kernel_sum += kernel_W(r)
        else:
            # FCC 3D: 4 atoms per cubic supercell of side a = dx·√2 — gives
            # 12 nearest neighbors at distance dx for each atom.
            a = diameter * math.sqrt(2.0)
            n_iter = int(math.ceil(h / a)) + 2
            basis = (
                (0.0,     0.0,     0.0),
                (a * 0.5, a * 0.5, 0.0),
                (a * 0.5, 0.0,     a * 0.5),
                (0.0,     a * 0.5, a * 0.5),
            )
            for ci in range(-n_iter, n_iter + 1):
                for cj in range(-n_iter, n_iter + 1):
                    for ck in range(-n_iter, n_iter + 1):
                        for (bx, by, bz) in basis:
                            if (ci == 0 and cj == 0 and ck == 0
                                    and bx == 0.0 and by == 0.0 and bz == 0.0):
                                continue
                            x = ci * a + bx
                            y = cj * a + by
                            z = ck * a + bz
                            r = math.sqrt(x * x + y * y + z * z)
                            kernel_sum += kernel_W(r)

    if kernel_sum <= 0:
        raise ValueError(
            f"calibration failed: kernel_sum={kernel_sum} for "
            f"h={h}, dx={diameter}, dim={dimension}, lattice='{lattice}'")
    return 1.0 / kernel_sum


@dataclass
class PhysicsConfig:
    """Per-case physics parameters.

    Convention used throughout this codebase: ``h`` is the **kernel support
    radius** (Wendland C4 with W(r)=0 for r>=h, normalization 9/(πh²) in 2D).
    This is the "support = h" convention — different from Monaghan-style
    "support = 2h". If your old code used support=2h, divide its h by 2 to
    convert.

    ``particle_radius`` is half the inter-particle spacing — the natural
    radius the SPH community usually quotes. Particle diameter (= old
    `particle_spacing`) is derived as ``2 · particle_radius``.
    """
    dimension: int                                  # 2 or 3
    h: float                                        # kernel support radius (W(r>=h)=0)
    particle_radius: float                          # particle "size"; diameter dx = 2·r
    speed_of_sound: float                           # c0
    power: float                                    # γ in Tait EOS
    cfl: float                                      # for timestep derivation
    gravity: tuple[float, float, float]
    # When True (default), per-material volume is calibrated so that
    # Σ_{j != i} V·W(r_ij) = 1 holds for an interior particle in the chosen
    # lattice arrangement — i.e. the GPU's correction.comp output kernel_sum
    # lands on exactly 1.0 in the bulk. Set False to use V = (2·radius)^DIM
    # (naive) for A/B comparison.
    calibrate_volume: bool = True
    # Lattice arrangement used by the case's particle layout. Affects the
    # calibrated volume (different lattices have different Σ W).
    #   "grid": Cartesian (square in 2D, simple cubic in 3D)  — Blender's
    #           default uniform-grid mesh export falls in here.
    #   "hex":  2D hexagonal close packing / 3D face-centered cubic.
    # Defaults to "grid"; switch to "hex" if your preprocessor emits hex/FCC.
    lattice: str = "grid"
    # Constant p_b added to the Tait pressure (Pa), 2026-09-29. Only for closed
    # domains without a free surface; keeps the pressure positive so that the
    # pressure force stays on the symmetric form. 0 = off.
    background_pressure: float = 0.0
    # Hydrostatic initial density (2026-09-29). A point [x, y, z] on the level
    # where the hydrostatic pressure is zero (the lid of a closed tank, the
    # free surface otherwise). FLUID particles then start with the density of
    #   p(x) = rho0 g . (x - reference),   rho = rho0 (1 + p / B)^(1/gamma)
    # instead of rho0, so that gravity is balanced from the first step.
    # None = off (every particle starts at rho0).
    hydrostatic_reference: Optional[tuple] = None

    def __post_init__(self):
        if self.hydrostatic_reference is not None:
            self.hydrostatic_reference = tuple(float(component) for component in self.hydrostatic_reference)
            if len(self.hydrostatic_reference) != 3:
                raise ValueError("physics.hydrostatic_reference must have 3 components")
        # YAML parses [0, -9.81, 0] as a list; coerce to tuple of floats.
        self.gravity = tuple(float(component) for component in self.gravity)
        if len(self.gravity) != 3:
            raise ValueError(
                f"physics.gravity must have 3 components, got {len(self.gravity)}")
        if self.dimension not in (2, 3):
            raise ValueError(
                f"physics.dimension must be 2 or 3, got {self.dimension}")
        if self.h <= 0:
            raise ValueError(f"physics.h must be > 0, got {self.h}")
        if self.particle_radius <= 0:
            raise ValueError(
                f"physics.particle_radius must be > 0, got {self.particle_radius}")
        if self.speed_of_sound <= 0:
            raise ValueError(
                f"physics.speed_of_sound must be > 0, got {self.speed_of_sound}")
        if self.power <= 0:
            raise ValueError(f"physics.power must be > 0, got {self.power}")
        if self.cfl <= 0:
            raise ValueError(f"physics.cfl must be > 0, got {self.cfl}")
        # Hard math contradiction: kernel support < particle diameter means
        # the nearest neighbor is outside the kernel — SPH cannot operate.
        if 2.0 * self.particle_radius > self.h:
            raise ValueError(
                f"physics: particle diameter (2·particle_radius="
                f"{2.0 * self.particle_radius}) must be <= h ({self.h}); "
                f"otherwise the kernel support contains no neighbors and "
                f"SPH is degenerate.")
        if self.lattice not in ("grid", "hex"):
            raise ValueError(
                f"physics.lattice must be 'grid' or 'hex', got {self.lattice!r}")

    @property
    def particle_diameter(self) -> float:
        """dx = 2 · particle_radius (legacy `particle_spacing`)."""
        return 2.0 * self.particle_radius


@dataclass
class RegularizationConfig:
    xi: float                                       # Tikhonov diagonal add
    det_threshold: float                            # below → fall back to identity M⁻¹
    frobenius_max: float                            # cap |M⁻¹|_F

    def __post_init__(self):
        if self.xi <= 0:
            raise ValueError(f"regularization.xi must be > 0, got {self.xi}")
        if self.det_threshold <= 0:
            raise ValueError(
                f"regularization.det_threshold must be > 0, got {self.det_threshold}")
        if self.frobenius_max <= 0:
            raise ValueError(
                f"regularization.frobenius_max must be > 0, got {self.frobenius_max}")


PAIR_CORRECTION_MODES = {"own": 0, "mean": 1, "reverse": 2}
SOLID_PRESSURE_MODES = {"increment": 0, "mirror": 1, "mirror_tic": 2, "accumulate": 3, "extrapolate": 4}
PST_NEAR_SOLID_MODES = {"full": 0, "tangential": 1}


@dataclass
class NumericsConfig:
    delta_coefficient: float                        # δ in δ-SPH density diffusion
    pst_main: float                                 # PST main-shift scale (Sun 2017)
    pst_anti: float                                 # PST anti-shift (cohesion) multiplier
    regularization: RegularizationConfig
    # Ablation toggle (default keeps δ-plus KCG behavior). When False, density
    # and force shaders use identity for M⁻¹ and zero for ∇ρ — equivalent to
    # plain δ-SPH (Antuono 2012) without kernel-gradient correction. Useful
    # for A/B-testing against legacy non-KCG codebases. correction.comp still
    # runs (kernel_sum is needed for PST blend), only its M⁻¹ / ∇ρ outputs
    # are ignored downstream.
    use_kcg_correction: bool = True
    # δ-SPH continuity diffusion: when False, density.comp skips the δ term and
    # the result reduces to vanilla WCSPH continuity. delta_coefficient is then
    # spec-const-supplied but unread (DCE).
    use_density_diffusion: bool = True
    # Particle Shifting Technique: when False, force.comp skips the shift
    # accumulation loop and writes shift = 0. predict's drift becomes pure
    # x_{n+1} = x_n + v_{n+1/2}·dt. pst_main / pst_anti unread (DCE).
    use_pst: bool = True
    # Defrag base-offset strategy: when True, defrag.comp consumes
    # voxel_base_offset[] (deterministic, voxel-id ordered, requires a
    # prefix_sum pass first). When False, defrag uses an atomicAdd on a
    # scratch counter (non-deterministic order, but contiguous within each
    # voxel; works standalone). Default False until prefix_sum.comp lands.
    use_prefix_sum_defrag: bool = False
    # Per-step neighbour list (2026-09-25): when True, build_neighbor_list.comp
    # runs once per step after update_voxel and stores, for every own
    # particle, the ids of all particles inside the kernel support; the three
    # neighbour kernels (correction / density / force) then iterate that list
    # instead of scanning the 27 surrounding voxels three times. Same pair set
    # and same summation order as the voxel scan, so results match to
    # rounding. When False the kernels use the original 27-voxel scan and the
    # list buffers stay unread (spec-constant DCE).
    # DEFAULT False: measured 1.6x SLOWER on the 3 mm tank (RTX 4070 Ti SUPER,
    # log/2026-09-25_neighbor-list.md) because the voxel scan's candidate
    # reads are warp-broadcast while list entries are per-lane gathers. Kept
    # as an opt-in experiment / reference implementation.
    use_neighbor_list: bool = False
    # Particle defragmentation: rearranges the particle SoA so that, for every
    # voxel V, that voxel's particles occupy a contiguous slot range in the
    # SoA. Restores spatial locality between SoA index and voxel coordinates,
    # which would otherwise degrade as particles drift between voxels over
    # time, thrashing GPU caches.
    #
    # When enabled, defrag runs once at the end of bootstrap (init defrag, to
    # remove dependency on .obj upload order) and then every defrag_cadence
    # steps. Each defrag dispatch is a per-voxel scatter to a scratch buffer,
    # followed by a linear copy back to the primary SoA — set 0 is always the
    # canonical live data, scratch is purely transient.
    defrag_enabled: bool = True
    defrag_cadence: int = 1000
    # Pressure of the solid particles seen by the fluid (2026-09-29):
    #   "increment"  : the stored one-step value EOS(rho0 + dt d rho/dt) (original)
    #   "mirror"     : pairwise mirror P_j = P_i + rho_i (g - a_j).(x_j - x_i),
    #                  fluid-solid pairs always in the symmetric form
    #   "mirror_tic" : the same, fluid-solid pairs follow the TIC switch
    #   "accumulate" : the solid integrates its density in time, lower bound rho0
    #                  (dynamic boundary condition, Crespo et al. 2007)
    #   "extrapolate": the pressure of the fluid neighbours continued to the solid
    #                  particle (Adami et al. 2012)
    solid_pressure: str = "increment"
    # KCG matrices used by the pressure and viscous forces of FLUID-FLUID pairs
    # (2026-09-30, see common.glsl):
    #   "own"    : the matrix of the particle itself (the original code)
    #   "mean"   : the mean of the two matrices, pair forces equal and opposite
    #   "reverse": reverse kernel gradient correction P_i B_j + P_j B_i
    #              (Zhang, Adams, Hu 2025), pair forces equal and opposite
    pair_correction: str = "own"
    # older spelling of pair_correction: mean
    symmetric_pair_correction: bool = False
    # psi_ij of the density diffusion with the renormalised density gradients
    # (vanishes for a linear density field; needed with gravity).
    density_diffusion_gradient_term: bool = False
    # "full" or "tangential": remove the solid-normal component of the shift of
    # fluid particles with solid neighbours.
    pst_near_solid: str = "full"
    # The acceleration of ROTOR / BOUNDARY particles is the reaction of the
    # fluid (exactly minus what the fluid receives). Forced on by the mirror
    # modes, where a solid particle has no pressure of its own.
    solid_reaction_force: bool = False
    # Mirror modes: constant p_w (Pa) added to the mirrored solid pressure, a
    # repulsive layer on fluid-solid pairs only (see common.glsl).
    solid_pressure_offset: float = 0.0

    def __post_init__(self):
        if self.solid_pressure not in SOLID_PRESSURE_MODES:
            raise ValueError(f"numerics.solid_pressure must be one of {list(SOLID_PRESSURE_MODES)}, "
                             f"got {self.solid_pressure!r}")
        if self.solid_pressure in ("mirror", "mirror_tic"):
            self.solid_reaction_force = True
        if self.pair_correction not in PAIR_CORRECTION_MODES:
            raise ValueError(f"numerics.pair_correction must be one of {list(PAIR_CORRECTION_MODES)}, "
                             f"got {self.pair_correction!r}")
        if self.symmetric_pair_correction:
            if self.pair_correction not in ("own", "mean"):
                raise ValueError("numerics.symmetric_pair_correction: true contradicts "
                                 f"pair_correction: {self.pair_correction}")
            self.pair_correction = "mean"
        if self.pst_near_solid not in PST_NEAR_SOLID_MODES:
            raise ValueError(f"numerics.pst_near_solid must be one of {list(PST_NEAR_SOLID_MODES)}, "
                             f"got {self.pst_near_solid!r}")
        if self.delta_coefficient < 0:
            raise ValueError(
                f"numerics.delta_coefficient must be >= 0, got {self.delta_coefficient}")
        if self.pst_main < 0:
            raise ValueError(f"numerics.pst_main must be >= 0, got {self.pst_main}")
        if self.pst_anti < 0:
            raise ValueError(f"numerics.pst_anti must be >= 0, got {self.pst_anti}")
        if self.defrag_cadence <= 0:
            raise ValueError(
                f"numerics.defrag_cadence must be > 0, got {self.defrag_cadence}")


@dataclass
class CapacitiesConfig:
    pool_size: int                                  # max active particles
    max_per_voxel: int                              # MAX_PARTICLES_PER_VOXEL
    max_incoming: int                               # MAX_INCOMING_PER_VOXEL
    workgroup: int                                  # WORKGROUP_SIZE
    # MAX_NEIGHBORS: per-particle neighbour-list capacity. 0 (default) means
    # "closest-packing upper bound for this h/dx" (Case.neighbors_max_estimate);
    # an explicit value must be at least that bound.
    max_neighbors: int = 0

    def __post_init__(self):
        if self.max_neighbors < 0:
            raise ValueError(
                f"capacities.max_neighbors must be >= 0, got {self.max_neighbors}")
        if self.pool_size <= 0:
            raise ValueError(f"capacities.pool_size must be > 0, got {self.pool_size}")
        if self.max_per_voxel <= 0:
            raise ValueError(
                f"capacities.max_per_voxel must be > 0, got {self.max_per_voxel}")
        if self.max_incoming <= 0:
            raise ValueError(
                f"capacities.max_incoming must be > 0, got {self.max_incoming}")
        if self.workgroup <= 0:
            raise ValueError(f"capacities.workgroup must be > 0, got {self.workgroup}")


@dataclass
class TimeConfig:
    """All three budgets are independent; None on a budget disables it."""
    total: Optional[float]                          # physical-time cap (s); None = unlimited
    max_steps: Optional[int]                        # step-count cap; None = unlimited
    output_cadence: Optional[float]                 # snapshot interval (s); None = no snapshots

    def __post_init__(self):
        if self.total is not None and self.total <= 0:
            raise ValueError(f"time.total must be > 0 or None, got {self.total}")
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError(
                f"time.max_steps must be > 0 or None, got {self.max_steps}")
        if self.output_cadence is not None and self.output_cadence <= 0:
            raise ValueError(
                f"time.output_cadence must be > 0 or None, got {self.output_cadence}")

    def is_time_exceeded(self, current_time: float) -> bool:
        return self.total is not None and current_time >= self.total

    def is_step_exceeded(self, current_step: int) -> bool:
        return self.max_steps is not None and current_step >= self.max_steps


# ============================================================================
# Resolved entities
# ============================================================================


@dataclass
class MaterialEntry:
    """One resolved material: library spec + derived values + compact group_id.

    Field order from ``kind`` onward mirrors the ``MaterialParameters`` struct
    in shaders/sph/common.glsl. Buffer upload code can pack each row by reading
    these fields in declared order without re-sorting (12 × 4 B = 48 B / row).
    """
    # --- Python-side metadata (NOT uploaded) -----------------------------
    name: str
    group_id: int                                   # compact 0..N-1, indexes material_parameters[]

    # --- GPU struct fields (in common.glsl order; 48 B total) ------------
    kind: int                                       # uint  (FLUID/BOUNDARY/INLET/ROTOR)
    rest_density: float
    viscosity: float                                # kinematic ν
    eos_constant: float                             # derived: c0² · rest_density / γ
    smoothing_length: float                         # derived: V0 = global h
    radius: float                                   # derived: dx / 2
    volume: float                                   # derived: dx^DIM
    rotor_angular_velocity: float = 0.0
    # Reserved (V0 unused; zero-padded on upload to match the struct's 48 B layout).
    viscosity_transfer: float = 0.0                 # micropolar (V0+)
    viscosity_rotation: float = 0.0                 # micropolar (V0+)
    reserved_material_0: int = 0
    reserved_material_1: int = 0

    # --- Python-side runtime hints (NOT in the GPU struct) ---------------
    # initial_velocity: applied to every particle of this material at upload
    # time (simulator._build_initial_data writes velocity_mass.xyz from this).
    # For BOUNDARY kind, predict.comp skips → the value persists, modelling
    # moving walls (lid-driven cavity top, conveyor belts, ...). For FLUID
    # kind, the value is the IC velocity and predict overwrites it normally.
    initial_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class RotorConfig:
    """Prescribed rigid-body rotation of every MATERIAL_ROTOR particle
    (case.yaml optional block ``rotor:``, added 2026-09-25).

    ``axis``      unit vector of the rotation axis (world coordinates)
    ``pivot``     any point on the axis (world coordinates)
    ``ramp_time`` seconds over which the angular velocity rises linearly from
                  0 to the material's ``rotor_angular_velocity`` (0 = no ramp)

    The signed angular velocity itself lives on the rotor material
    (``rotor_angular_velocity``, rad/s, right-hand rule about ``axis``).
    One rotor per case: all rotor-kind materials must share the same value.
    """
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    pivot: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ramp_time: float = 0.0

    def __post_init__(self):
        axis = np.asarray(self.axis, dtype=np.float64)
        if axis.shape != (3,) or np.linalg.norm(axis) == 0.0:
            raise ValueError(f"rotor.axis must be a non-zero 3-vector, got {self.axis}")
        axis = axis / np.linalg.norm(axis)
        self.axis = (float(axis[0]), float(axis[1]), float(axis[2]))
        pivot = np.asarray(self.pivot, dtype=np.float64)
        if pivot.shape != (3,):
            raise ValueError(f"rotor.pivot must be a 3-vector, got {self.pivot}")
        self.pivot = (float(pivot[0]), float(pivot[1]), float(pivot[2]))
        self.ramp_time = float(self.ramp_time)
        if self.ramp_time < 0.0:
            raise ValueError(f"rotor.ramp_time must be >= 0, got {self.ramp_time}")


@dataclass
class ScalarFieldConfig:
    """One scalar field carried by the FLUID particles (case.yaml
    ``scalars.fields`` entry, added 2026-09-27).

    ``diffusivity`` molecular diffusivity D_m (m^2/s); 0 = no molecular diffusion
    ``turbulent``   True: the field also receives nu_t / Sc_t when
                    ``scalars.sgs.enabled``; False: never (e.g. biomass attached
                    to the particles, which must not be smeared between them)
    ``initial``     value given to every FLUID particle at upload (solids carry 0)

    Values are specific (per unit mass): advection leaves them unchanged and the
    conserved amount is sum_i m_i C_i.
    """
    name: str
    diffusivity: float = 0.0
    turbulent: bool = True
    initial: float = 0.0

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"scalars.fields: every field needs a non-empty name, got {self.name!r}")
        self.diffusivity = float(self.diffusivity)
        self.initial = float(self.initial)
        self.turbulent = bool(self.turbulent)
        if self.diffusivity < 0.0:
            raise ValueError(f"scalars.fields[{self.name}].diffusivity must be >= 0, got {self.diffusivity}")


@dataclass
class ScalarSgsConfig:
    """Smagorinsky sub-grid diffusivity for the scalars (``scalars.sgs``):
        nu_t = (smagorinsky_cs * filter_width)^2 |S|,   D_t = nu_t / turbulent_schmidt
    ``filter_width`` defaults to the particle spacing dx. Used in the scalar
    diffusion only, not in the momentum equation."""
    enabled: bool = False
    smagorinsky_cs: float = 0.1
    filter_width: Optional[float] = None
    turbulent_schmidt: float = 0.7

    def __post_init__(self):
        self.enabled = bool(self.enabled)
        self.smagorinsky_cs = float(self.smagorinsky_cs)
        self.turbulent_schmidt = float(self.turbulent_schmidt)
        if self.smagorinsky_cs <= 0.0:
            raise ValueError(f"scalars.sgs.smagorinsky_cs must be > 0, got {self.smagorinsky_cs}")
        if self.turbulent_schmidt <= 0.0:
            raise ValueError(f"scalars.sgs.turbulent_schmidt must be > 0, got {self.turbulent_schmidt}")
        if self.filter_width is not None:
            self.filter_width = float(self.filter_width)
            if self.filter_width <= 0.0:
                raise ValueError(f"scalars.sgs.filter_width must be > 0, got {self.filter_width}")


@dataclass
class ScalarInjectionConfig:
    """A tracer pulse (``scalars.injections`` entry): while
    start <= t < start + duration, every FLUID particle whose position lies
    inside the sphere (center, radius) gets field ``field`` set to ``value``."""
    field: str
    center: tuple[float, float, float]
    radius: float
    start: float
    duration: float
    value: float = 1.0

    def __post_init__(self):
        center = tuple(float(component) for component in self.center)
        if len(center) != 3:
            raise ValueError(f"scalars.injections: center must have 3 components, got {self.center}")
        self.center = center
        self.radius = float(self.radius)
        self.start = float(self.start)
        self.duration = float(self.duration)
        self.value = float(self.value)
        if self.radius <= 0.0 or self.duration <= 0.0:
            raise ValueError(f"scalars.injections[{self.field}]: radius and duration must be > 0")


@dataclass
class ScalarProbePoint:
    name: str
    position: tuple[float, float, float]

    def __post_init__(self):
        position = tuple(float(component) for component in self.position)
        if len(position) != 3:
            raise ValueError(f"scalars.probes: position of {self.name!r} must have 3 components")
        self.position = position


@dataclass
class ScalarProbesConfig:
    """Probe points for time series (``scalars.probes``). The value is a
    Shepard (kernel-weighted) average over the FLUID particles within
    ``radius`` (default h), see SphSimulatorV1.probe_scalars."""
    points: list[ScalarProbePoint]
    radius: Optional[float] = None

    def __post_init__(self):
        if self.radius is not None:
            self.radius = float(self.radius)
            if self.radius <= 0.0:
                raise ValueError(f"scalars.probes.radius must be > 0, got {self.radius}")


MAX_SCALAR_FIELDS = 12              # 4 * MAX_SCALAR_VEC4 in common.glsl
MAX_INJECTION_SLOTS = 4             # MAX_INJECTION_SLOTS in common.glsl


@dataclass
class ScalarsConfig:
    """Optional case.yaml block ``scalars:`` (scalar transport, 2026-09-27).

    ``shift_correction`` / ``compensated_sum`` / ``bounds_limiter`` are
    ablation toggles (common.glsl ids 67, 68, 70). ``shift_correction`` is
    OFF by default (2026-09-27): the Taylor correction C += shift . grad C is
    not conservative (tracer total +0.45 % in 0.5 s after a pulse in the 4 mm
    tank, +3.5 % with the bounds limiter, which clips undershoots) and not
    bounded without the limiter (-0.37 .. 1.46 for a 0 / 1 tracer). Without it
    the scalar is conserved to round-off and bounded, at the price of the
    shift moving particles relative to the fluid (measured effective
    dispersion ~1e-4 m^2/s at dx = 4 mm). See log/2026-09-27_scalar-transport.md. Fields are packed four per vec4 in
    declaration order: field k lives in vec4 k // 4, component k % 4."""
    fields: list[ScalarFieldConfig]
    sgs: ScalarSgsConfig
    injections: list[ScalarInjectionConfig]
    probes: Optional[ScalarProbesConfig] = None
    shift_correction: bool = False
    compensated_sum: bool = True
    bounds_limiter: bool = True

    def __post_init__(self):
        if not 1 <= len(self.fields) <= MAX_SCALAR_FIELDS:
            raise ValueError(
                f"scalars.fields: 1..{MAX_SCALAR_FIELDS} fields supported, got {len(self.fields)}")
        names = [field.name for field in self.fields]
        if len(set(names)) != len(names):
            raise ValueError(f"scalars.fields: duplicate names in {names}")
        for injection in self.injections:
            if injection.field not in names:
                raise ValueError(
                    f"scalars.injections: unknown field {injection.field!r} (fields: {names})")
        # At most MAX_INJECTION_SLOTS pulses may be active at the same time.
        events = sorted([(i.start, 1) for i in self.injections]
                        + [(i.start + i.duration, -1) for i in self.injections],
                        key=lambda event: (event[0], event[1]))
        active = 0
        for _, change in events:
            active += change
            if active > MAX_INJECTION_SLOTS:
                raise ValueError(
                    f"scalars.injections: more than {MAX_INJECTION_SLOTS} pulses overlap in time")
        self.shift_correction = bool(self.shift_correction)
        self.compensated_sum = bool(self.compensated_sum)
        self.bounds_limiter = bool(self.bounds_limiter)

    @property
    def field_names(self) -> list[str]:
        return [field.name for field in self.fields]

    @property
    def vec4_count(self) -> int:
        return (len(self.fields) + 3) // 4

    def field_location(self, name: str) -> tuple[int, int]:
        """(vec4 index v, component c) of a field."""
        field_index = self.field_names.index(name)
        return field_index // 4, field_index % 4


def _parse_scalars(data: dict, source: str) -> "ScalarsConfig":
    allowed = {"fields", "sgs", "injections", "probes", "shift_correction", "compensated_sum",
               "bounds_limiter"}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"{source}: unknown keys in `scalars:` block: {sorted(unknown)}")
    if not data.get("fields"):
        raise ValueError(f"{source}: `scalars:` block needs a non-empty `fields:` list")
    probes = None
    if data.get("probes") is not None:
        probe_data = dict(data["probes"])
        points = [ScalarProbePoint(**point) for point in probe_data.pop("points", [])]
        probes = ScalarProbesConfig(points=points, **probe_data)
    return ScalarsConfig(
        fields=[ScalarFieldConfig(**field) for field in data["fields"]],
        sgs=ScalarSgsConfig(**(data.get("sgs") or {})),
        injections=[ScalarInjectionConfig(**injection) for injection in (data.get("injections") or [])],
        probes=probes,
        shift_correction=data.get("shift_correction", False),
        compensated_sum=data.get("compensated_sum", True),
        bounds_limiter=data.get("bounds_limiter", True),
    )


@dataclass
class ParticleSource:
    """One obj file's vertices + its material assignment."""
    obj_path: pathlib.Path
    vertices: np.ndarray                            # (N, 3) float32
    material_name: str
    material_group_id: int                          # backfilled at Case construction


# ============================================================================
# Case — atomic unit consumed by Vulkan. yaml + obj → this; this → Vulkan.
# ============================================================================


@dataclass
class Case:
    """Fully resolved case ready for Vulkan upload.

    Holds CPU-side data only; no GPU resources allocated yet. Bundles
    parameters (the four block configs from case.yaml), the grid derived from
    frame.obj's bbox, the resolved materials, and per-obj particle sources.

    Derived quantities that depend ONLY on intrinsic parameters are exposed
    as @property (timestep, kernel coefficients, ...). Cross-block validation
    (max_per_voxel ≥ closest-packing bound) runs in __post_init__.
    """
    # --- Parameters (1:1 with case.yaml top-level blocks) ---------------
    physics: PhysicsConfig
    numerics: NumericsConfig
    capacities: CapacitiesConfig
    time: TimeConfig

    # --- Geometry-derived ----------------------------------------------
    grid: dict                                      # {'origin': (3,), 'dimension': (3,)}

    # --- Resolved -------------------------------------------------------
    materials: list[MaterialEntry]                  # ordered by group_id 0..N-1
    particle_sources: list[ParticleSource]
    case_dir: pathlib.Path                          # for relative path debugging
    rotor: Optional["RotorConfig"] = None           # present iff case.yaml has a `rotor:` block
    scalars: Optional["ScalarsConfig"] = None       # present iff case.yaml has a `scalars:` block

    @property
    def scalar_vec4_count(self) -> int:
        """SCALAR_VEC4_COUNT spec constant: 0 without a `scalars:` block."""
        return 0 if self.scalars is None else self.scalars.vec4_count

    @property
    def sgs_length_squared(self) -> float:
        """(C_s Delta)^2 for the Smagorinsky nu_t (0 when the SGS is off);
        Delta defaults to the particle spacing dx."""
        if self.scalars is None or not self.scalars.sgs.enabled:
            return 0.0
        sgs = self.scalars.sgs
        width = sgs.filter_width if sgs.filter_width is not None else self.physics.particle_diameter
        return (sgs.smagorinsky_cs * width) ** 2

    @property
    def rotor_angular_velocity(self) -> float:
        """Signed angular velocity (rad/s) shared by all rotor-kind materials;
        0.0 when the case has no rotor material."""
        values = {float(m.rotor_angular_velocity) for m in self.materials if m.kind == KIND_ROTOR}
        if not values:
            return 0.0
        if len(values) > 1:
            raise ValueError(
                f"all rotor-kind materials must share one rotor_angular_velocity, got {sorted(values)}")
        return values.pop()

    def __post_init__(self):
        # Cross-block validation: max_per_voxel must accommodate the
        # geometric closest-packing upper bound. With this passing, the GPU
        # initialize_voxelization shader's atomicAdd path provably cannot
        # overflow — no per-particle CPU pre-count is needed.
        max_estimate = self.particles_per_voxel_max_estimate
        if self.capacities.max_per_voxel < max_estimate:
            ratio = self.physics.h / self.physics.particle_diameter
            raise ValueError(
                f"capacities.max_per_voxel ({self.capacities.max_per_voxel}) is "
                f"below the closest-packing upper bound ({max_estimate}) for "
                f"h/dx = {ratio:.3f} in {self.physics.dimension}D. Increase "
                f"capacities.max_per_voxel in case.yaml, or coarsen geometry "
                f"(reduce h/dx ratio).")
        # Neighbour-list capacity: same closest-packing argument, applied to
        # the support sphere (radius h) instead of the voxel cube.
        neighbor_bound = self.neighbors_max_estimate
        if self.capacities.max_neighbors == 0:
            self.capacities.max_neighbors = neighbor_bound
        elif self.capacities.max_neighbors < neighbor_bound:
            raise ValueError(
                f"capacities.max_neighbors ({self.capacities.max_neighbors}) is "
                f"below the closest-packing upper bound ({neighbor_bound}) for the "
                f"support sphere at h/dx = "
                f"{self.physics.h / self.physics.particle_diameter:.3f}. Raise it "
                f"or set 0 to use the bound.")

    # ------------------------------------------------------------------
    # Derived from intrinsic parameters (no geometry / materials needed)
    # ------------------------------------------------------------------

    @property
    def timestep(self) -> float:
        """dt = CFL · h / c0."""
        return self.physics.cfl * self.physics.h / self.physics.speed_of_sound

    @property
    def kernel_coefficient(self) -> float:
        """Wendland C4 normalization (support radius = h, NOT 2h).

        2D:  9   / (π · h²)
        3D:  495 / (32 · π · h³)
        """
        h = self.physics.h
        if self.physics.dimension == 2:
            return 9.0 / (math.pi * h * h)
        return 495.0 / (32.0 * math.pi * h * h * h)

    @property
    def kernel_gradient_coefficient(self) -> float:
        """∇W coefficient = W coefficient / h."""
        return self.kernel_coefficient / self.physics.h

    @property
    def eps_h_squared(self) -> float:
        """Antuono δ-SPH division-by-zero guard for 1/(r² + ε_h²) terms."""
        h = self.physics.h
        return 0.01 * h * h

    @property
    def neighbor_z_range(self) -> int:
        """27-voxel neighbor loop in 3D; collapses to 9 in 2D."""
        return 1 if self.physics.dimension == 3 else 0

    # --- Capacity diagnostics (uniform-spacing sanity estimates) --------
    # Both quantities below assume uniform packing at diameter dx = 2·r.
    # The "estimate" forms are loose; the "max" form is the geometric
    # closest-packing upper bound used to validate capacities.max_per_voxel.

    @property
    def particles_per_voxel_estimate(self) -> float:
        """Expected particle count in a single voxel under uniform spacing.

        2D: (h / dx)²        3D: (h / dx)³     (dx = 2·particle_radius)
        """
        ratio = self.physics.h / self.physics.particle_diameter
        return ratio ** self.physics.dimension

    @property
    def neighbors_in_support_estimate(self) -> float:
        """Expected neighbor count inside the Wendland C4 support (radius = h).

        2D: π · (h/dx)²              3D: (4/3) · π · (h/dx)³

        Stable WCSPH typically wants >= ~30 in 2D, >= ~50 in 3D.
        """
        ratio = self.physics.h / self.physics.particle_diameter
        if self.physics.dimension == 2:
            return math.pi * ratio * ratio
        return (4.0 / 3.0) * math.pi * ratio ** 3

    @property
    def particles_per_voxel_max_estimate(self) -> int:
        """Geometric upper bound on particles centered inside a single voxel,
        assuming particles enforce a minimum separation of ``2·particle_radius``.

        2D hexagonal close packing density:  2 / (√3 · dx²)  ≈ 1.155 / dx²
        3D FCC / HCP close packing density:  √2 / dx³        ≈ 1.414 / dx³

        Multiplied by voxel volume (h² in 2D, h³ in 3D) and ceil-rounded.
        Used as a HARD upper bound for capacities.max_per_voxel.
        """
        ratio = self.physics.h / self.physics.particle_diameter
        if self.physics.dimension == 2:
            density_factor = 2.0 / math.sqrt(3.0)
            return int(math.ceil(density_factor * ratio * ratio))
        density_factor = math.sqrt(2.0)
        return int(math.ceil(density_factor * ratio ** 3))

    @property
    def neighbors_max_estimate(self) -> int:
        """Closest-packing upper bound on particles strictly inside the
        Wendland support (a disc / sphere of radius h) around one particle,
        excluding the particle itself. Used as the hard bound for
        capacities.max_neighbors (MAX_NEIGHBORS), like
        particles_per_voxel_max_estimate is for max_per_voxel.

        2D: 2/(sqrt3 dx^2) * pi h^2        3D: sqrt2/dx^3 * 4/3 pi h^3
        (h/dx = 3 in 3D gives 160; the cubic-lattice count at rest is 113.)
        """
        ratio = self.physics.h / self.physics.particle_diameter
        if self.physics.dimension == 2:
            return int(math.ceil(2.0 / math.sqrt(3.0) * math.pi * ratio * ratio))
        return int(math.ceil(math.sqrt(2.0) * 4.0 / 3.0 * math.pi * ratio ** 3))


# ============================================================================
# Specialization constant assembly
# ----------------------------------------------------------------------------
# Hand-maintained mapping. MUST be kept in lockstep with shaders/sph/common.glsl
# spec constant declarations (matching IDs, types, and intent).
#
# Each row: (constant_id, getter(case) -> python value, struct format).
# Format characters: 'f' = float32, 'I' = uint32, 'i' = int32.
# Boolean spec constants are encoded as 'I' (0 / 1) per VkBool32 convention.
# ============================================================================


_SpecGetter = Callable[[Case], object]
_SpecRow = tuple[int, _SpecGetter, str]


_SPEC_CONSTANT_MAPPING: list[_SpecRow] = [
    # id  getter                                                       fmt
    (0,   lambda case: case.physics.h,                                 'f'),
    (1,   lambda case: case.physics.speed_of_sound,                    'f'),
    (2,   lambda case: case.numerics.delta_coefficient,                'f'),
    # id=3 reserved (was EPSILON_SHIFT, removed)
    (4,   lambda case: case.physics.power,                             'f'),
    (5,   lambda case: case.physics.cfl,                               'f'),
    (6,   lambda case: case.timestep,                                  'f'),  # derived
    (7,   lambda case: case.grid['origin'][0],                         'f'),
    (8,   lambda case: case.grid['origin'][1],                         'f'),
    (9,   lambda case: case.grid['origin'][2],                         'f'),
    (10,  lambda case: 0,                                              'I'),  # STRICT_BIT_EXACT (V0: false)
    (11,  lambda case: case.grid['dimension'][0],                      'I'),
    (12,  lambda case: case.grid['dimension'][1],                      'I'),
    (13,  lambda case: case.grid['dimension'][2],                      'I'),
    (14,  lambda case: case.numerics.regularization.xi,                'f'),
    (15,  lambda case: case.numerics.regularization.det_threshold,     'f'),
    (16,  lambda case: case.numerics.regularization.frobenius_max,     'f'),
    (17,  lambda case: case.physics.gravity[0],                        'f'),
    (18,  lambda case: case.physics.gravity[1],                        'f'),
    (19,  lambda case: case.physics.gravity[2],                        'f'),
    (20,  lambda case: 0,                                              'I'),  # VOXEL_ORDER (V0: linear)
    (21,  lambda case: 2.0,                                            'f'),  # MICROPOLAR_THETA (V0 unused)
    (30,  lambda case: case.physics.dimension,                         'I'),
    (31,  lambda case: case.neighbor_z_range,                          'I'),
    (32,  lambda case: case.kernel_coefficient,                        'f'),
    (33,  lambda case: case.kernel_gradient_coefficient,               'f'),
    (34,  lambda case: case.physics.background_pressure,               'f'),  # BACKGROUND_PRESSURE
    (35,  lambda case: SOLID_PRESSURE_MODES[case.numerics.solid_pressure], 'I'),  # SOLID_PRESSURE_MODE
    (36,  lambda case: 1 if case.numerics.solid_reaction_force else 0, 'I'),  # USE_SOLID_REACTION_FORCE
    (37,  lambda case: case.numerics.solid_pressure_offset,            'f'),  # SOLID_PRESSURE_OFFSET
    (38,  lambda case: 1 if case.numerics.density_diffusion_gradient_term else 0, 'I'),
    (39,  lambda case: PST_NEAR_SOLID_MODES[case.numerics.pst_near_solid], 'I'),  # PST_NEAR_SOLID_MODE
    (40,  lambda case: case.eps_h_squared,                             'f'),
    (41,  lambda case: case.numerics.pst_main,                         'f'),
    (42,  lambda case: case.numerics.pst_anti,                         'f'),
    # Algorithm ablation toggles (id 43-46). bool spec consts; glslc -O DCE
    # removes dead branches when a toggle is false.
    (43,  lambda case: 1 if case.numerics.use_kcg_correction    else 0, 'I'),  # USE_KCG_CORRECTION
    (44,  lambda case: 1 if case.numerics.use_density_diffusion else 0, 'I'),  # USE_DENSITY_DIFFUSION
    (45,  lambda case: 1 if case.numerics.use_pst               else 0, 'I'),  # USE_PST
    (46,  lambda case: 1 if case.numerics.use_prefix_sum_defrag else 0, 'I'),  # USE_PREFIX_SUM_DEFRAG
    (47,  lambda case: 1 if case.numerics.use_neighbor_list     else 0, 'I'),  # USE_NEIGHBOR_LIST
    (48,  lambda case: PAIR_CORRECTION_MODES[case.numerics.pair_correction], 'I'),  # PAIR_CORRECTION_MODE
    (50,  lambda case: case.capacities.max_per_voxel,                  'I'),
    (51,  lambda case: case.capacities.workgroup,                      'I'),
    (52,  lambda case: case.capacities.max_incoming,                   'I'),
    (53,  lambda case: case.capacities.pool_size,                      'I'),
    (62,  lambda case: case.capacities.max_neighbors,                  'I'),  # MAX_NEIGHBORS
    # 63-71 scalar transport (2026-09-27); neutral values without a `scalars:` block.
    (63,  lambda case: case.scalar_vec4_count,                         'I'),  # SCALAR_VEC4_COUNT
    (64,  lambda case: 1 if case.scalars is not None and case.scalars.sgs.enabled else 0, 'I'),
    (65,  lambda case: case.sgs_length_squared,                        'f'),  # (C_s Delta)^2
    (66,  lambda case: (1.0 / case.scalars.sgs.turbulent_schmidt) if case.scalars is not None else 1.0, 'f'),
    (67,  lambda case: 1 if case.scalars is not None and case.scalars.shift_correction else 0, 'I'),
    (68,  lambda case: 1 if case.scalars is None or case.scalars.compensated_sum else 0, 'I'),
    (69,  lambda case: 1 if case.scalars is not None and case.scalars.injections else 0, 'I'),
    (70,  lambda case: 1 if case.scalars is None or case.scalars.bounds_limiter else 0, 'I'),
    (71,  lambda case: 0 if case.scalars is None else len(case.scalars.fields), 'I'),  # SCALAR_FIELD_COUNT
    # 56-61 rotor axis / pivot (world coords); defaults when no rotor block.
    (56,  lambda case: (case.rotor.axis[0]  if case.rotor else 0.0),   'f'),
    (57,  lambda case: (case.rotor.axis[1]  if case.rotor else 0.0),   'f'),
    (58,  lambda case: (case.rotor.axis[2]  if case.rotor else 1.0),   'f'),
    (59,  lambda case: (case.rotor.pivot[0] if case.rotor else 0.0),   'f'),
    (60,  lambda case: (case.rotor.pivot[1] if case.rotor else 0.0),   'f'),
    (61,  lambda case: (case.rotor.pivot[2] if case.rotor else 0.0),   'f'),
    # V0-a ghost grid: all disabled (GHOST_DIMENSION_* = 0 dead-code-eliminates branches)
    (80,  lambda case: 0,                                              'I'),
    (81,  lambda case: 0,                                              'I'),
    (82,  lambda case: 0,                                              'I'),
    (83,  lambda case: 0.0,                                            'f'),
    (84,  lambda case: 0.0,                                            'f'),
    (85,  lambda case: 0.0,                                            'f'),
    (86,  lambda case: 0,                                              'i'),
    (87,  lambda case: 0,                                              'i'),
    (88,  lambda case: 0,                                              'i'),
]


class SpecializationInfo(NamedTuple):
    """Pure-Python spec info; pipelines.py converts to VkSpecializationInfo."""
    map_entries: list[tuple[int, int, int]]         # (constant_id, offset, size)
    data: bytes                                     # packed blob


def build_specialization_info(case: Case) -> SpecializationInfo:
    """Pack all spec constants into a contiguous data blob and return alongside
    map entries (constant_id, offset, size).

    Constant ordering and types must match shaders/sph/common.glsl exactly.
    """
    map_entries: list[tuple[int, int, int]] = []
    data = bytearray()
    for constant_id, getter, fmt in _SPEC_CONSTANT_MAPPING:
        value = getter(case)
        size = struct.calcsize(fmt)
        map_entries.append((constant_id, len(data), size))
        data.extend(struct.pack(fmt, value))
    return SpecializationInfo(map_entries=map_entries, data=bytes(data))


# ============================================================================
# Top-level entry
# ============================================================================


def load_case(case_yaml_path) -> Case:
    """Load and validate a complete case from disk.

    Raises ``ValueError`` on schema mismatch, missing field, or any cross-domain
    contradiction (over-budget particle count, particle outside frame bbox,
    unknown material name, V0-disallowed kind). Raises ``TypeError`` on
    structural mismatch between yaml and dataclass field set (typo / missing
    field) thanks to ``**kwargs`` splat construction.
    """
    case_yaml_path = pathlib.Path(case_yaml_path).resolve()
    case_dir = case_yaml_path.parent

    case_data = yaml.safe_load(case_yaml_path.read_text(encoding="utf-8"))
    _check_schema(case_data, CASE_SCHEMA_VERSION, source=str(case_yaml_path))

    # Build parameter blocks via **kwargs splat. Each yaml block's keys must
    # exactly match the dataclass field names — extras / missing surface as
    # TypeError immediately.
    physics = PhysicsConfig(**case_data["physics"])
    capacities = CapacitiesConfig(**case_data["capacities"])
    time = TimeConfig(**case_data["time"])

    numerics_data = dict(case_data["numerics"])     # copy so we can pop
    regularization = RegularizationConfig(**numerics_data.pop("regularization"))
    numerics = NumericsConfig(regularization=regularization, **numerics_data)

    # Material library
    library_path = (case_dir / case_data["material_library"]).resolve()
    library = _load_material_library(library_path)

    # Particle sources
    particle_sources, used_material_names = _load_particle_sources(
        case_dir, case_data["geometry"])

    # Resolve only used materials; assign compact 0..N-1 group_ids
    materials = _resolve_materials(
        library, used_material_names, physics, library_path)
    name_to_group = {m.name: m.group_id for m in materials}
    for source in particle_sources:
        source.material_group_id = name_to_group[source.material_name]

    # Grid from frame.obj bbox
    frame_path = (case_dir / case_data["geometry"]["frame"]).resolve()
    frame_vertices = load_obj_vertices(frame_path)
    if frame_vertices.shape[0] == 0:
        raise ValueError(f"frame obj {frame_path} contains no vertices")
    frame_min = frame_vertices.min(axis=0)
    frame_max = frame_vertices.max(axis=0)
    if physics.dimension == 2:
        # Tolerate Blender-style frames where the cube has tiny z thickness
        # (a 2D layout exported with a default Cube → ±1 mm extrusion). The
        # 2D solver only uses xy; collapse z to zero before grid derivation
        # and downstream particle-in-bbox check.
        frame_min = frame_min.copy()
        frame_max = frame_max.copy()
        frame_min[2] = 0.0
        frame_max[2] = 0.0
    grid = compute_grid(
        frame_min, frame_max,
        physics.h,
        physics.dimension,
    )

    # Cross-domain validation that needs particle data + frame bbox.
    # (Case.__post_init__ then runs the closest-packing self-check.)
    # Pass the (possibly z-flattened) frame bbox to keep the in-frame check
    # consistent with the grid we just computed.
    _cross_validate(physics, capacities, particle_sources, frame_min, frame_max)

    # Optional rotor block (prescribed rigid rotation of rotor-kind materials).
    rotor = None
    if case_data.get("rotor") is not None:
        rotor = RotorConfig(**case_data["rotor"])
    uses_rotor_material = any(m.kind == KIND_ROTOR for m in materials)
    if uses_rotor_material and rotor is None:
        raise ValueError(
            f"{case_yaml_path}: a rotor-kind material is used but the case has no "
            f"`rotor:` block (axis / pivot / ramp_time)")
    if rotor is not None and not uses_rotor_material:
        raise ValueError(
            f"{case_yaml_path}: `rotor:` block present but no rotor-kind material is used")

    # Optional scalar transport block (2026-09-27).
    scalars = None
    if case_data.get("scalars") is not None:
        scalars = _parse_scalars(case_data["scalars"], source=str(case_yaml_path))
        if not any(m.kind == KIND_FLUID for m in materials):
            raise ValueError(f"{case_yaml_path}: `scalars:` block but no fluid-kind material")

    return Case(
        physics=physics,
        numerics=numerics,
        capacities=capacities,
        time=time,
        grid=grid,
        materials=materials,
        particle_sources=particle_sources,
        case_dir=case_dir,
        rotor=rotor,
        scalars=scalars,
    )


# ============================================================================
# Internal helpers
# ============================================================================


def _check_schema(data, expected_version, source) -> None:
    if not isinstance(data, dict):
        raise ValueError(
            f"{source}: top-level yaml must be a mapping, got {type(data).__name__}")
    actual_version = data.get("schema_version")
    if actual_version is None:
        raise ValueError(f"{source}: missing 'schema_version' field")
    if actual_version != expected_version:
        raise ValueError(
            f"{source}: schema_version={actual_version} does not match "
            f"expected {expected_version}. Migrate the file or pin to an "
            f"older code revision.")


def _load_material_library(library_path) -> dict:
    if not library_path.exists():
        raise FileNotFoundError(f"material library not found: {library_path}")
    data = yaml.safe_load(library_path.read_text(encoding="utf-8"))
    _check_schema(data, MATERIAL_SCHEMA_VERSION, source=str(library_path))
    # Drop schema_version key; remaining top-level keys are material names.
    return {key: value for key, value in data.items() if key != "schema_version"}


def _load_particle_sources(case_dir, geometry_dict):
    """Return (sources, ordered-list-of-unique-material-names)."""
    sources: list[ParticleSource] = []
    used_names: list[str] = []
    for entry in geometry_dict["particles"]:
        obj_path = (case_dir / entry["file"]).resolve()
        vertices = load_obj_vertices(obj_path)
        material_name = entry["material"]
        sources.append(ParticleSource(
            obj_path=obj_path,
            vertices=vertices,
            material_name=material_name,
            material_group_id=-1,                   # backfilled in load_case
        ))
        if material_name not in used_names:
            used_names.append(material_name)        # preserve first-seen order
    return sources, used_names


def _resolve_materials(library, used_names, physics, library_path) -> list[MaterialEntry]:
    """Build MaterialEntry list with compact group_ids and derived values.

    V0 rejects INLET kind explicitly (inlet spawn is V0+).
    """
    materials: list[MaterialEntry] = []
    for group_id, name in enumerate(used_names):
        if name not in library:
            raise ValueError(
                f"material '{name}' referenced in case.yaml not found in "
                f"library {library_path}")
        spec = library[name]
        kind_string = spec.get("kind")
        if kind_string not in _KIND_NAME_TO_CODE:
            raise ValueError(
                f"material '{name}': unknown kind {kind_string!r} (expected one of "
                f"{list(_KIND_NAME_TO_CODE.keys())})")
        kind = _KIND_NAME_TO_CODE[kind_string]
        if kind == KIND_INLET:
            raise ValueError(
                f"material '{name}': kind=inlet not supported in V0 (inlet "
                f"spawn is deferred to V0+). Remove inlet materials from the case.")

        # Derived per-material parameters.
        speed_of_sound = physics.speed_of_sound
        gamma = physics.power
        rest_density = float(spec["rest_density"])
        eos_constant = speed_of_sound * speed_of_sound * rest_density / gamma

        radius = physics.particle_radius
        diameter = 2.0 * radius
        if physics.calibrate_volume:
            volume = _calibrate_particle_volume(
                physics.h, radius, physics.dimension, physics.lattice)
        else:
            volume = diameter ** physics.dimension
        smoothing_length = physics.h

        # Optional initial_velocity (default zero). YAML list → tuple[float×3].
        initial_velocity_raw = spec.get("initial_velocity", [0.0, 0.0, 0.0])
        if len(initial_velocity_raw) != 3:
            raise ValueError(
                f"material '{name}': initial_velocity must have 3 components, "
                f"got {initial_velocity_raw}")
        initial_velocity = tuple(float(component) for component in initial_velocity_raw)

        materials.append(MaterialEntry(
            name=name,
            group_id=group_id,
            kind=kind,
            rest_density=rest_density,
            viscosity=float(spec["viscosity"]),
            eos_constant=eos_constant,
            smoothing_length=smoothing_length,
            radius=radius,
            volume=volume,
            rotor_angular_velocity=float(spec.get("rotor_angular_velocity", 0.0)),
            initial_velocity=initial_velocity,
        ))
    return materials


def _cross_validate(physics, capacities, particle_sources, frame_min, frame_max) -> None:
    # Hard: total active particles must fit pool_size.
    total_active = sum(int(source.vertices.shape[0]) for source in particle_sources)
    if total_active > capacities.pool_size:
        raise ValueError(
            f"total particles ({total_active}) exceeds capacities.pool_size "
            f"({capacities.pool_size}). Increase pool_size or reduce "
            f"particle count.")

    # Hard: every particle must lie inside the frame bbox (else init shader
    # silently kills it via open-boundary semantics, surprising the user).
    for source in particle_sources:
        if source.vertices.shape[0] == 0:
            continue
        ps_min = source.vertices.min(axis=0)
        ps_max = source.vertices.max(axis=0)
        if (ps_min < frame_min).any() or (ps_max > frame_max).any():
            raise ValueError(
                f"{source.obj_path.name}: particles extend outside frame bbox.\n"
                f"  frame bbox: min={frame_min.tolist()} max={frame_max.tolist()}\n"
                f"  particles : min={ps_min.tolist()} max={ps_max.tolist()}")

    # Soft: dx > h/2 means too few neighbors in the kernel support.
    diameter = 2.0 * physics.particle_radius
    if diameter > 0.5 * physics.h:
        ratio = physics.h / diameter
        if physics.dimension == 2:
            neighbors = math.pi * ratio * ratio
        else:
            neighbors = (4.0 / 3.0) * math.pi * ratio ** 3
        print(
            f"[case] WARNING: h/dx = {ratio:.2f} is low. WCSPH stability "
            f"typically wants h/dx >= 2 (recommended 3-4). Estimated neighbors "
            f"in support = {neighbors:.1f}; density / KCG / kernel-sum "
            f"estimators may be noisy.")
