"""
_check_rotor_torque_split.py — where does the rotor torque readback come from?

SphSimulatorV1.readback_rotor_torque() sums m_i (x_i - pivot) x (a_i - g) over
the live ROTOR particles, with a_i the full force.comp acceleration (pressure +
viscosity from ALL neighbours: fluid, other rotor particles and wall). Its
docstring assumes that the rotor-rotor pair forces cancel. They would only if
force.comp's pair force were antisymmetric and central, and it is neither:
  * the kernel gradient is corrected with the SELF particle's KCG matrix only,
    ∇W̃_ij = M_i⁻¹ ∇W_ij, so the forces i<-j and j<-i differ by
    (M_i⁻¹ - M_j⁻¹) ∇W_ij; even with a pair-mean matrix the pair force
    M̄ ∇W_ij is not parallel to x_ij, so equal and opposite pair forces still
    leave a pair torque x_ij × M̄ ∇W_ij;
  * TIC: pressure_combined = (P_i > 0 || near_surface_i) ? P_j + P_i : P_j - P_i,
    i.e. P_j - P_i = (P_i + P_j) - 2 P_i on the antisymmetric branch;
  * V_j = m_i / ρ_j (self mass, neighbour density). With one mass for all
    materials this does NOT break the antisymmetry: V_j / ρ_i = m / (ρ_i ρ_j).

This script re-implements force.comp on the CPU (float64) for every live rotor
particle, validates it against the GPU acceleration and the torque readback,
and splits the axial torque, the axial force and the radial force into fluid /
rotor-internal / wall neighbour contributions, each for the pressure and the
viscous term. The internal pressure part is decomposed exactly into
    non-central part (pair-mean M⁻¹, symmetric P_i + P_j; the force of this part
                      cancels exactly, only a torque can remain)
  + M-asymmetry part (self M_i⁻¹ instead of the pair mean, symmetric P)
  + TIC part         (P_j - P_i instead of P_i + P_j on the antisymmetric branch)
and a raw-∇W symmetric variant is evaluated as a bookkeeping check (its force
and torque must cancel to round-off). Because the TIC term -2 P_i multiplies
Σ_j V_j M_i⁻¹ ∇W_ij, which is ~0 over the full neighbourhood of a fully
supported rotor particle but large over its fluid or its rotor neighbours
alone, TIC moves torque between the "fluid" and the "internal" columns; the
script therefore also evaluates the whole rotor torque with other pressure
pairings (symmetric, antisymmetric, and pair-mean M⁻¹ with P_i + P_j, which is
exactly antisymmetric between rotor and fluid particles and so is the
conservative, action = reaction reference), and the reaction on the fluid,
i.e. the torque the rotor exerts on the fluid particles according to THEIR
force.comp formula. Finally it characterises the solid (wall / rotor)
pressure that density.comp produces:
    P_s = EOS(ρ0 + dt (drift + diffusion)), FLUID neighbours only,
    drift     = Σ_j -ρ_i V_j (v_j - v_i)·∇W̃_ij,
    diffusion = δ h c0 Σ_j 2 (ρ_j - ρ_i) (x_ji·∇W̃_ij) / (r² + ε_h²) V_j,
    V_j = m_j / ρ_j (pre-step ρ_j), ∇W̃_ij = M_i⁻¹ ∇W_ij.

Two phases, because the GPU phase needs the solver's Python env (vulkan) and
the CPU phase needs scipy (cKDTree):

  run      <python with vulkan> _check_rotor_torque_split.py run CASE.yaml
               --steps 4005 8005 12005 --out-dir DIR [--device N] [--repository ROOT]
           Bootstraps the case and steps with step(wait=True). After step s-1 it
           reads density_pressure (the pre-step densities that density.comp of
           step s uses), after step s every buffer force.comp consumed, plus
           readback_rotor_torque(). A requested step s must not be a defrag
           step and must not follow one (defrag renumbers the pool slots).
           Writes DIR/snapshot_<s>.npz (live slots only, JSON parameter block).

  analyze  <python with numpy + scipy> _check_rotor_torque_split.py analyze PATH [PATH ...]
               [--json OUT.json] [--shaft-radius 0.007] [--split-height 0.1]
           PATH = a snapshot .npz or a directory of them. Prints, per snapshot,
           the validation, the split tables, the internal-torque diagnosis and
           the solid-pressure table; --json also writes every number.

Sign conventions: torques about the rotor axis through the pivot (+y through
the origin for the 30 L tank). The torque of the fluid on the rotor is
negative for a positive angular velocity; the power number is reported as
Np = -τ ω / (ρ N³ D⁵) with ρ = 998 kg/m³, N = 200/60 1/s, D = 0.096 m (same
scale as _analyze_power_number.py), so the parts of the torque add up.

Example (2026-09-27, dished 30 L tank, --c0-factor 20, h/dx = 3; the solver env
had no scipy, so the analysis ran in another interpreter):
    python utils/geometry/_demo_stirred_tank_30l.py --dx 0.004 --hdx 3 --c0-factor 20 --out <DIR>/tank4mm --no-preview
    <sph env python> _check_rotor_torque_split.py run <DIR>/tank4mm/case.yaml --steps 4005 8005 12005 --out-dir <DIR>/snapshots4mm
    <python with scipy> _check_rotor_torque_split.py analyze <DIR>/snapshots4mm --json results_4mm.json
Result (4 mm t = 0.35-1.05 s, 3 mm t = 0.26-0.53 s): CPU vs GPU acceleration
within 4e-7 of max |a|, torque within 1.2e-7. The internal (rotor-rotor) part
is 35-44 % of the readback, >= 98.5 % of it from TIC and 0.4-0.5 % of the
readback from M_i != M_j, while the fluid part carries an opposite TIC share
of the same size. The readback exceeds the conservative reference by 4.2-4.9 %.
"""

import argparse
import json
import pathlib
import sys
import time

import numpy as np

# Material kind codes (utils/sph/case.py, common.glsl).
KIND_FLUID = 0
KIND_BOUNDARY = 1
KIND_INLET = 2
KIND_ROTOR = 3
NEIGHBOUR_KIND_LABELS = ((KIND_FLUID, "fluid"), (KIND_ROTOR, "rotor"), (KIND_BOUNDARY, "wall"))

# force.comp: near_surface = kernel_sum < 0.75; Morris coefficient 2 (DIM + 2).
NEAR_SURFACE_KERNEL_SUM = 0.75
# Power number scale, as in experiment/v1/checks/_analyze_power_number.py.
POWER_NUMBER_DENSITY = 998.0
POWER_NUMBER_REVOLUTIONS_PER_SECOND = 200.0 / 60.0
POWER_NUMBER_DIAMETER = 0.096
POWER_NUMBER_SCALE = (POWER_NUMBER_DENSITY * POWER_NUMBER_REVOLUTIONS_PER_SECOND ** 3
                      * POWER_NUMBER_DIAMETER ** 5)


# =============================================================================
# Phase 1: GPU run + snapshot dump (needs the solver env)
# =============================================================================

def resolve_repository_root(explicit_root):
    """Repository root: --repository, else three levels above this file
    (experiment/v1/checks/<this file>)."""
    if explicit_root:
        return pathlib.Path(explicit_root).resolve()
    here = pathlib.Path(__file__).resolve()
    if len(here.parents) > 3:
        candidate = here.parents[3]
        if (candidate / "experiment" / "v1" / "utils" / "simulator_v1.py").exists():
            return candidate
    raise SystemExit("cannot locate the repository root; pass --repository")


def read_raw_buffer(simulator, buffer_name, dtype, components):
    """Copy of one device buffer as an (N, components) array (N = pool + 1)."""
    raw = simulator._readback_buffer(simulator.buffers[buffer_name])
    return np.frombuffer(raw, dtype=dtype).reshape(-1, components).copy()


def json_ready(value):
    """numpy scalars / arrays inside a dict -> plain Python for json.dumps."""
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def capture_snapshot(simulator, case, case_path, previous_density_pressure, output_directory,
                     split_height, shaft_radius):
    """Read back everything force.comp consumed in the step just finished and
    save the live slots to output_directory/snapshot_<step>.npz."""
    positions = simulator.readback_positions()
    live = simulator.live_slot_mask(positions)
    slots = np.nonzero(live)[0]
    velocity_mass = simulator.readback_velocity_mass()
    acceleration = simulator.readback_acceleration()
    material = simulator.readback_material()
    density_pressure = read_raw_buffer(simulator, "density_pressure", np.float32, 2)
    # 32 B per particle: [2 pid] = (m00, m11, m22, m01), [2 pid + 1] = (m02, m12, d/trM, fluid flag)
    correction_inverse = read_raw_buffer(simulator, "correction_inverse", np.float32, 8)
    density_gradient_kernel_sum = read_raw_buffer(simulator, "density_gradient_kernel_sum", np.float32, 4)
    status = simulator.readback_global_status()
    gpu_torque = simulator.readback_rotor_torque(split_height, shaft_radius)
    _, angular_velocity_now = simulator.rotor_angle_and_rate(simulator.simulation_time)

    physics, numerics = case.physics, case.numerics
    parameters = {
        "case": str(case_path),
        "step": int(simulator.step_count),
        "time": float(simulator.simulation_time),
        "rotor_angle": float(simulator.rotor_angle),
        "rotor_angular_velocity_now": float(angular_velocity_now),
        "rotor_axis": list(case.rotor.axis),
        "rotor_pivot": list(case.rotor.pivot),
        "dimension": int(physics.dimension),
        "smoothing_length": float(physics.h),
        "particle_spacing": float(physics.particle_diameter),
        "kernel_coefficient": float(case.kernel_coefficient),
        "kernel_gradient_coefficient": float(case.kernel_gradient_coefficient),
        "eps_h_squared": float(case.eps_h_squared),
        "gravity": list(physics.gravity),
        "timestep": float(case.timestep),
        "speed_of_sound": float(physics.speed_of_sound),
        "power": float(physics.power),
        "cfl": float(physics.cfl),
        "delta_coefficient": float(numerics.delta_coefficient),
        "use_kcg_correction": bool(numerics.use_kcg_correction),
        "use_density_diffusion": bool(numerics.use_density_diffusion),
        "use_neighbor_list": bool(numerics.use_neighbor_list),
        "regularization_xi": float(numerics.regularization.xi),
        "defrag_cadence": int(numerics.defrag_cadence),
        "materials": [
            {"group_id": int(entry.group_id), "name": entry.name, "kind": int(entry.kind),
             "rest_density": float(entry.rest_density), "viscosity": float(entry.viscosity),
             "eos_constant": float(entry.eos_constant), "volume": float(entry.volume),
             "rotor_angular_velocity": float(entry.rotor_angular_velocity)}
            for entry in case.materials],
        "global_status": status,
        "gpu_torque": json_ready(gpu_torque),
        "split_height": float(split_height),
        "shaft_radius": float(shaft_radius),
    }
    path = output_directory / f"snapshot_{simulator.step_count:07d}.npz"
    np.savez(path,
             parameters=np.array(json.dumps(json_ready(parameters))),
             slots=slots.astype(np.int64),
             positions=positions[live],
             velocity_mass=velocity_mass[live],
             acceleration=acceleration[live],
             material=material[live],
             density_pressure=density_pressure[live],
             previous_density_pressure=previous_density_pressure[live],
             correction_inverse=correction_inverse[live],
             density_gradient_kernel_sum=density_gradient_kernel_sum[live])
    return path, status, gpu_torque


def run_snapshots(arguments):
    repository_root = resolve_repository_root(arguments.repository)
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))
    from utils.sph.case import load_case
    from utils.sph.vulkan_context import VulkanContext
    from experiment.v1 import compile_shaders_v1
    from experiment.v1.utils.simulator_v1 import SphSimulatorV1

    compile_shaders_v1.compile_v1_shaders()
    case_path = pathlib.Path(arguments.case).resolve()
    case = load_case(str(case_path))
    if case.rotor is None:
        raise SystemExit("the case has no rotor")
    cadence = case.numerics.defrag_cadence if case.numerics.defrag_enabled else 0
    target_steps = sorted(set(arguments.steps))
    for step in target_steps:
        if step < 2:
            raise SystemExit("requested steps must be >= 2")
        if cadence and (step % cadence == 0 or (step - 1) % cadence == 0):
            raise SystemExit(f"step {step}: a defrag runs after every step divisible by {cadence}; "
                             f"pick steps with step % {cadence} not in (0, 1)")
    output_directory = pathlib.Path(arguments.out_dir).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    context_arguments = dict(application_name="rotor_torque_split", enable_validation=False)
    if arguments.device is not None:
        context_arguments["device_index"] = arguments.device
    with VulkanContext.create(**context_arguments) as context:
        simulator = SphSimulatorV1(context, case)
        try:
            simulator.bootstrap()
            start = time.perf_counter()
            for target in target_steps:
                while simulator.step_count < target - 1:
                    simulator.step(wait=True)
                # ρ_n of every slot: what density.comp of the next step reads.
                previous_density_pressure = read_raw_buffer(simulator, "density_pressure", np.float32, 2)
                simulator.step(wait=True)
                path, status, gpu_torque = capture_snapshot(
                    simulator, case, case_path, previous_density_pressure, output_directory,
                    arguments.split_height, arguments.shaft_radius)
                elapsed = time.perf_counter() - start
                print(f"[torque-split] step={simulator.step_count} t={simulator.simulation_time:.4f}s "
                      f"torque_axis(GPU)={gpu_torque['torque_axis']:.6e} N·m "
                      f"alive={status['alive_particle_count']:,} "
                      f"overflow inside/incoming/fallback/neighbor="
                      f"{status['overflow_inside_count']}/{status['overflow_incoming_count']}/"
                      f"{status['correction_fallback_count']}/{status['overflow_neighbor_count']} "
                      f"({simulator.step_count / elapsed:.1f} steps/s incl. readbacks) -> {path.name}")
        finally:
            simulator.destroy()


# =============================================================================
# Phase 2: CPU re-implementation and analysis (numpy + scipy)
# =============================================================================

def as_float32(value):
    """A case constant as the GPU sees it (float32 specialization constant)."""
    return float(np.float32(value))


def unpack_correction_inverse(packed):
    """(N, 8) float32 -> (N, 3, 3) float64, mirroring helpers.glsl
    unpack_correction_inverse (symmetric by construction)."""
    first = packed[:, 0:4].astype(np.float64)
    second = packed[:, 4:8].astype(np.float64)
    matrices = np.empty((packed.shape[0], 3, 3))
    matrices[:, 0, 0] = first[:, 0]
    matrices[:, 1, 1] = first[:, 1]
    matrices[:, 2, 2] = first[:, 2]
    matrices[:, 0, 1] = matrices[:, 1, 0] = first[:, 3]
    matrices[:, 0, 2] = matrices[:, 2, 0] = second[:, 0]
    matrices[:, 1, 2] = matrices[:, 2, 1] = second[:, 1]
    return matrices


def kernel_value(distance, smoothing_length, kernel_coefficient):
    """Wendland C4, support h: C (1-q)^6 (35/3 q² + 6q + 1)."""
    normalized_distance = distance / smoothing_length
    one_minus_q = 1.0 - normalized_distance
    return (kernel_coefficient * one_minus_q ** 6
            * (normalized_distance * ((35.0 / 3.0) * normalized_distance + 6.0) + 1.0))


def kernel_gradient(position_difference, distance, smoothing_length, gradient_coefficient):
    """helpers.glsl evaluate_kernel_gradient_unguarded:
    ∇W_ij = C_grad (1-q)^5 q (-280/3 q - 56/3) x_ij / r,  C_grad = C / h."""
    normalized_distance = distance / smoothing_length
    one_minus_q = 1.0 - normalized_distance
    derivative_profile = normalized_distance * ((-280.0 / 3.0) * normalized_distance + (-56.0 / 3.0))
    magnitude = gradient_coefficient * one_minus_q ** 5 * derivative_profile
    return (magnitude / distance)[:, None] * position_difference


def neighbour_pairs(query_positions, target_tree, radius, radius_squared_limit, target_positions,
                    chunk_size=40000):
    """All (query q, target t) with 1e-24 <= |x_q - x_t|² < radius_squared_limit
    (the force.comp support test, evaluated in float64). Returns index arrays
    into query_positions and into the target tree's points."""
    from scipy.spatial import cKDTree
    query_parts, target_parts = [], []
    for start in range(0, query_positions.shape[0], chunk_size):
        chunk = query_positions[start:start + chunk_size]
        pairs = cKDTree(chunk).sparse_distance_matrix(target_tree, radius, output_type="ndarray")
        query_index = pairs["i"].astype(np.int64)
        target_index = pairs["j"].astype(np.int64)
        difference = chunk[query_index] - target_positions[target_index]
        distance_squared = np.einsum("ij,ij->i", difference, difference)
        keep = (distance_squared < radius_squared_limit) & (distance_squared >= 1e-24)
        query_parts.append(query_index[keep] + start)
        target_parts.append(target_index[keep])
    return np.concatenate(query_parts), np.concatenate(target_parts)


def sum_per_particle(index, values, count):
    """Σ over pairs of per-pair vectors (P, 3) into (count, 3) by index."""
    return np.stack([np.bincount(index, weights=values[:, component], minlength=count)
                     for component in range(values.shape[1])], axis=1)


class Fields:
    """Live-particle state of one snapshot, float64, plus derived per-particle
    material quantities (force.comp takes ν from the SELF material)."""

    def __init__(self, snapshot, parameters):
        self.positions = snapshot["positions"][:, :3].astype(np.float64)
        self.velocity = snapshot["velocity_mass"][:, :3].astype(np.float64)
        self.mass = snapshot["velocity_mass"][:, 3].astype(np.float64)
        self.gpu_acceleration = snapshot["acceleration"][:, :3].astype(np.float64)
        self.density = snapshot["density_pressure"][:, 0].astype(np.float64)
        self.pressure = snapshot["density_pressure"][:, 1].astype(np.float64)
        self.previous_density = snapshot["previous_density_pressure"][:, 0].astype(np.float64)
        self.correction = unpack_correction_inverse(snapshot["correction_inverse"])
        self.kernel_sum = snapshot["density_gradient_kernel_sum"][:, 3].astype(np.float64)
        self.near_surface = snapshot["density_gradient_kernel_sum"][:, 3] < np.float32(NEAR_SURFACE_KERNEL_SUM)
        group = snapshot["material"].astype(np.int64)
        materials = sorted(parameters["materials"], key=lambda entry: entry["group_id"])
        kind_of_group = np.array([entry["kind"] for entry in materials], dtype=np.int64)
        viscosity_of_group = np.array([as_float32(entry["viscosity"]) for entry in materials])
        rest_density_of_group = np.array([as_float32(entry["rest_density"]) for entry in materials])
        eos_constant_of_group = np.array([as_float32(entry["eos_constant"]) for entry in materials])
        self.kind = kind_of_group[group]
        self.viscosity = viscosity_of_group[group]
        self.rest_density = rest_density_of_group[group]
        self.eos_constant = eos_constant_of_group[group]
        self.count = self.positions.shape[0]


class Constants:
    """Specialization constants of the case, rounded to float32 like on the GPU."""

    def __init__(self, parameters):
        self.dimension = int(parameters["dimension"])
        self.smoothing_length = as_float32(parameters["smoothing_length"])
        self.smoothing_length_squared = float(np.float32(self.smoothing_length) * np.float32(self.smoothing_length))
        self.kernel_coefficient = as_float32(parameters["kernel_coefficient"])
        self.kernel_gradient_coefficient = as_float32(parameters["kernel_gradient_coefficient"])
        self.eps_h_squared = as_float32(parameters["eps_h_squared"])
        self.gravity = np.array([as_float32(value) for value in parameters["gravity"]])
        self.timestep = as_float32(parameters["timestep"])
        self.speed_of_sound = as_float32(parameters["speed_of_sound"])
        self.power = as_float32(parameters["power"])
        self.delta_coefficient = as_float32(parameters["delta_coefficient"])
        self.morris_coefficient = 2.0 * (self.dimension + 2.0)
        self.axis = np.asarray(parameters["rotor_axis"], dtype=np.float64)
        self.axis /= np.linalg.norm(self.axis)
        self.pivot = np.asarray(parameters["rotor_pivot"], dtype=np.float64)
        self.angular_velocity = float(parameters["rotor_angular_velocity_now"])


def momentum_pair_terms(fields, constants, self_index, neighbor_index,
                        matrix_mode="self", pressure_mode="tic"):
    """force.comp pair accelerations of particle self_index due to neighbor_index.

    matrix_mode:   "self"      ∇W̃ = M_self⁻¹ ∇W (force.comp)
                   "pair_mean" ∇W̃ = (M_self⁻¹ + M_neighbour⁻¹)/2 ∇W
                   "identity"  ∇W̃ = ∇W (raw kernel gradient)
    pressure_mode: "tic"           force.comp's TIC switch on the SELF pressure
                   "symmetric"     P_self + P_neighbour for every pair
                   "antisymmetric" P_neighbour - P_self for every pair
    Returns (pressure acceleration, viscous acceleration), each (P, 3)."""
    position_difference = fields.positions[self_index] - fields.positions[neighbor_index]   # x_ij
    distance_squared = np.einsum("ij,ij->i", position_difference, position_difference)
    distance = np.sqrt(distance_squared)
    gradient = kernel_gradient(position_difference, distance, constants.smoothing_length,
                               constants.kernel_gradient_coefficient)
    if matrix_mode == "self":
        corrected = np.einsum("nab,nb->na", fields.correction[self_index], gradient)
    elif matrix_mode == "pair_mean":
        pair_matrix = 0.5 * (fields.correction[self_index] + fields.correction[neighbor_index])
        corrected = np.einsum("nab,nb->na", pair_matrix, gradient)
    elif matrix_mode == "identity":
        corrected = gradient
    else:
        raise ValueError(matrix_mode)

    neighbor_volume = fields.mass[self_index] / fields.density[neighbor_index]        # V_j = m_i / ρ_j
    self_pressure = fields.pressure[self_index]
    neighbor_pressure = fields.pressure[neighbor_index]
    if pressure_mode == "tic":
        symmetric_branch = (self_pressure > 0.0) | fields.near_surface[self_index]
        pressure_combined = np.where(symmetric_branch, neighbor_pressure + self_pressure,
                                     neighbor_pressure - self_pressure)
    elif pressure_mode == "symmetric":
        pressure_combined = neighbor_pressure + self_pressure
    elif pressure_mode == "antisymmetric":
        pressure_combined = neighbor_pressure - self_pressure
    else:
        raise ValueError(pressure_mode)
    pressure_acceleration = -(neighbor_volume * pressure_combined / fields.density[self_index])[:, None] * corrected

    velocity_difference = fields.velocity[self_index] - fields.velocity[neighbor_index]
    viscous_scalar = (constants.morris_coefficient * fields.viscosity[self_index] * neighbor_volume
                      * np.einsum("ij,ij->i", velocity_difference, position_difference)
                      / (distance_squared + constants.eps_h_squared))
    viscous_acceleration = viscous_scalar[:, None] * corrected
    return pressure_acceleration, viscous_acceleration


def resultant(positions, mass, acceleration, constants):
    """Force and torque (about pivot) of per-particle accelerations."""
    force_per_particle = mass[:, None] * acceleration
    torque_vector = np.cross(positions - constants.pivot, force_per_particle).sum(axis=0)
    force = force_per_particle.sum(axis=0)
    axial_force = float(force @ constants.axis)
    return {"torque_axis": float(torque_vector @ constants.axis),
            "force_axis": axial_force,
            "force_radial": float(np.linalg.norm(force - axial_force * constants.axis)),
            "force": force.tolist(),
            "torque": torque_vector.tolist()}


def axial_torque_per_particle(positions, mass, acceleration, constants):
    return np.cross(positions - constants.pivot, mass[:, None] * acceleration) @ constants.axis


def power_number(torque_axis, angular_velocity):
    """Np = -τ ω / (ρ N³ D⁵): positive for a torque that resists the rotation."""
    return -torque_axis * angular_velocity / POWER_NUMBER_SCALE


def fit_line(x, y):
    """Least squares y ≈ a x + b and the correlation coefficient."""
    if x.size < 3 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return float("nan"), float("nan"), float("nan")
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept), float(np.corrcoef(x, y)[0, 1])


def analyze_rotor(fields, constants, shaft_radius, split_height):
    """Torque split of the rotor. Returns a result dict."""
    from scipy.spatial import cKDTree
    rotor = np.nonzero(fields.kind == KIND_ROTOR)[0]
    rotor_positions = fields.positions[rotor]
    rotor_mass = fields.mass[rotor]
    tree_all = cKDTree(fields.positions)
    local_self, neighbor = neighbour_pairs(
        rotor_positions, tree_all, constants.smoothing_length * (1.0 + 1e-6),
        constants.smoothing_length_squared, fields.positions)
    self_global = rotor[local_self]
    not_self = self_global != neighbor
    local_self, neighbor, self_global = local_self[not_self], neighbor[not_self], self_global[not_self]
    neighbor_kind = fields.kind[neighbor]

    pressure_pair, viscous_pair = momentum_pair_terms(fields, constants, self_global, neighbor)
    split_acceleration = {}
    for kind, label in NEIGHBOUR_KIND_LABELS:
        mask = neighbor_kind == kind
        split_acceleration[(label, "pressure")] = sum_per_particle(local_self[mask], pressure_pair[mask], rotor.size)
        split_acceleration[(label, "viscous")] = sum_per_particle(local_self[mask], viscous_pair[mask], rotor.size)
    unexpected = ~np.isin(neighbor_kind, [kind for kind, _ in NEIGHBOUR_KIND_LABELS])
    cpu_acceleration = sum(split_acceleration.values()) + constants.gravity

    # --- Validation against the GPU ------------------------------------------
    gpu_acceleration = fields.gpu_acceleration[rotor]
    error = np.linalg.norm(cpu_acceleration - gpu_acceleration, axis=1)
    magnitude = np.linalg.norm(gpu_acceleration, axis=1)
    significant = magnitude > 1e-3 * magnitude.max()
    validation = {
        "rotor_particles": int(rotor.size),
        "pairs": int(local_self.size),
        "pairs_by_neighbour_kind": {label: int((neighbor_kind == kind).sum()) for kind, label in NEIGHBOUR_KIND_LABELS},
        "pairs_other_kind": int(unexpected.sum()),
        "max_abs_error": float(error.max()),
        "max_gpu_acceleration": float(magnitude.max()),
        "rms_gpu_acceleration": float(np.sqrt(np.mean(magnitude ** 2))),
        "max_error_over_max_acceleration": float(error.max() / magnitude.max()),
        "rms_error_over_rms_acceleration": float(np.sqrt(np.mean(error ** 2)) / np.sqrt(np.mean(magnitude ** 2))),
        "per_particle_relative_error_max": float((error[significant] / magnitude[significant]).max()),
        "per_particle_relative_error_rms": float(np.sqrt(np.mean((error[significant] / magnitude[significant]) ** 2))),
        "per_particle_relative_error_count": int(significant.sum()),
    }
    gravity_free = gpu_acceleration - constants.gravity
    gpu_resultant = resultant(rotor_positions, rotor_mass, gravity_free, constants)
    cpu_resultant = resultant(rotor_positions, rotor_mass, cpu_acceleration - constants.gravity, constants)

    # --- Split table ----------------------------------------------------------
    rows = {}
    for (label, term), acceleration in split_acceleration.items():
        rows[f"{label}:{term}"] = resultant(rotor_positions, rotor_mass, acceleration, constants)
    for label in ("fluid", "rotor", "wall"):
        rows[f"{label}:total"] = resultant(
            rotor_positions, rotor_mass,
            split_acceleration[(label, "pressure")] + split_acceleration[(label, "viscous")], constants)

    # --- Regions (shaft r < shaft_radius, else lower / upper by height) --------
    arm = rotor_positions - constants.pivot
    height = arm @ constants.axis
    radial_distance = np.linalg.norm(arm - np.outer(height, constants.axis), axis=1)
    is_shaft = radial_distance < shaft_radius
    regions = {"shaft": is_shaft,
               "lower": ~is_shaft & (height < split_height),
               "upper": ~is_shaft & (height >= split_height)}
    region_table = {}
    for region_name, region_mask in regions.items():
        entry = {"particles": int(region_mask.sum())}
        for label in ("fluid", "rotor", "wall"):
            for term in ("pressure", "viscous"):
                entry[f"{label}:{term}"] = float(axial_torque_per_particle(
                    rotor_positions[region_mask], rotor_mass[region_mask],
                    split_acceleration[(label, term)][region_mask], constants).sum())
        entry["gpu_total"] = float(axial_torque_per_particle(
            rotor_positions[region_mask], rotor_mass[region_mask], gravity_free[region_mask], constants).sum())
        region_table[region_name] = entry

    # --- Internal (rotor-rotor) pressure decomposition ------------------------
    internal = neighbor_kind == KIND_ROTOR
    internal_self_local = local_self[internal]
    internal_self = self_global[internal]
    internal_neighbor = neighbor[internal]
    variants = {}
    for name, matrix_mode, pressure_mode in (
            ("as_gpu (M_i, TIC)", "self", "tic"),
            ("M_i, symmetric P", "self", "symmetric"),
            ("pair-mean M, TIC", "pair_mean", "tic"),
            ("pair-mean M, symmetric P", "pair_mean", "symmetric"),
            ("raw gradient, symmetric P", "identity", "symmetric")):
        pressure_variant, _ = momentum_pair_terms(fields, constants, internal_self, internal_neighbor,
                                                  matrix_mode, pressure_mode)
        per_particle = sum_per_particle(internal_self_local, pressure_variant, rotor.size)
        variants[name] = resultant(rotor_positions, rotor_mass, per_particle, constants)
        # scale of the individual pair terms, to judge "cancels to round-off"
        pair_forces = fields.mass[internal_self][:, None] * pressure_variant
        variants[name]["sum_abs_pair_force"] = float(np.linalg.norm(pair_forces, axis=1).sum())
        variants[name]["sum_abs_pair_axial_torque"] = float(np.abs(
            np.cross(fields.positions[internal_self] - constants.pivot, pair_forces) @ constants.axis).sum())
    decomposition = {
        "non_central (pair-mean M, symmetric P)": variants["pair-mean M, symmetric P"]["torque_axis"],
        "M_asymmetry (M_i vs pair mean, symmetric P)":
            variants["M_i, symmetric P"]["torque_axis"] - variants["pair-mean M, symmetric P"]["torque_axis"],
        "TIC (P_j - P_i vs P_i + P_j, M_i)":
            variants["as_gpu (M_i, TIC)"]["torque_axis"] - variants["M_i, symmetric P"]["torque_axis"],
    }
    decomposition_force_axis = {
        "non_central": variants["pair-mean M, symmetric P"]["force_axis"],
        "M_asymmetry": variants["M_i, symmetric P"]["force_axis"] - variants["pair-mean M, symmetric P"]["force_axis"],
        "TIC": variants["as_gpu (M_i, TIC)"]["force_axis"] - variants["M_i, symmetric P"]["force_axis"],
    }

    # --- Rotor pressure states -------------------------------------------------
    rotor_pressure = fields.pressure[rotor]
    rotor_near_surface = fields.near_surface[rotor]
    fluid_neighbour_count = np.bincount(local_self[neighbor_kind == KIND_FLUID], minlength=rotor.size)
    antisymmetric = (rotor_pressure <= 0.0) & ~rotor_near_surface
    states = {
        "pressure_positive": int((rotor_pressure > 0).sum()),
        "pressure_zero": int((rotor_pressure == 0).sum()),
        "pressure_negative": int((rotor_pressure < 0).sum()),
        "near_surface": int(rotor_near_surface.sum()),
        "tic_antisymmetric_branch (P<=0 and not near_surface)": int(antisymmetric.sum()),
        "tic_antisymmetric_with_nonzero_pressure": int((antisymmetric & (rotor_pressure != 0)).sum()),
        "with_fluid_neighbours": int((fluid_neighbour_count > 0).sum()),
        "with_fluid_neighbours_and_antisymmetric": int(((fluid_neighbour_count > 0) & antisymmetric).sum()),
        "kernel_sum_min": float(fields.kernel_sum[rotor].min()),
        "kernel_sum_median": float(np.median(fields.kernel_sum[rotor])),
    }

    # --- Internal torque by region, split into the three parts -----------------
    region_internal = {}
    per_variant_particle_torque = {}
    for name, matrix_mode, pressure_mode in (("gpu", "self", "tic"), ("self_symmetric", "self", "symmetric"),
                                             ("mean_symmetric", "pair_mean", "symmetric")):
        pressure_variant, _ = momentum_pair_terms(fields, constants, internal_self, internal_neighbor,
                                                  matrix_mode, pressure_mode)
        per_particle = sum_per_particle(internal_self_local, pressure_variant, rotor.size)
        per_variant_particle_torque[name] = axial_torque_per_particle(rotor_positions, rotor_mass, per_particle, constants)
    for region_name, region_mask in regions.items():
        gpu_part = per_variant_particle_torque["gpu"][region_mask].sum()
        self_symmetric = per_variant_particle_torque["self_symmetric"][region_mask].sum()
        mean_symmetric = per_variant_particle_torque["mean_symmetric"][region_mask].sum()
        region_internal[region_name] = {"internal_pressure": float(gpu_part),
                                        "non_central": float(mean_symmetric),
                                        "M_asymmetry": float(self_symmetric - mean_symmetric),
                                        "TIC": float(gpu_part - self_symmetric)}

    # --- Pressure-form counterfactuals, rotor side, per neighbour kind --------
    # With TIC the -2 P_i part of P_j - P_i multiplies Σ_j V_j M_i⁻¹ ∇W_ij,
    # which is ~0 over the FULL neighbourhood of a fully supported rotor
    # particle but large over the fluid or the rotor neighbours separately:
    # the TIC term moves torque between the "fluid" and "internal" columns.
    # "pair-mean M, symmetric" is exactly antisymmetric between a rotor and a
    # fluid particle of equal mass (V_j / ρ_i = m / (ρ_i ρ_j) is symmetric), so
    # it is the conservative reference that action = reaction applies to.
    pressure_forms = {}
    for form_name, matrix_mode, pressure_mode in (("as GPU (M_i, TIC)", "self", "tic"),
                                                  ("M_i, symmetric", "self", "symmetric"),
                                                  ("M_i, antisymmetric", "self", "antisymmetric"),
                                                  ("pair-mean M, symmetric", "pair_mean", "symmetric")):
        pressure_variant, viscous_variant = momentum_pair_terms(fields, constants, self_global, neighbor,
                                                                matrix_mode, pressure_mode)
        entry = {}
        total = np.zeros((rotor.size, 3))
        for kind, label in NEIGHBOUR_KIND_LABELS:
            mask = neighbor_kind == kind
            per_particle = sum_per_particle(local_self[mask], pressure_variant[mask], rotor.size)
            total += per_particle
            entry[label] = resultant(rotor_positions, rotor_mass, per_particle, constants)
        entry["total"] = resultant(rotor_positions, rotor_mass, total, constants)
        # pressure + viscous (viscous with the same matrix) per region
        total_with_viscous = total + sum_per_particle(local_self, viscous_variant, rotor.size)
        region_torque = axial_torque_per_particle(rotor_positions, rotor_mass, total_with_viscous, constants)
        entry["regions_with_viscous"] = {region_name: float(region_torque[region_mask].sum())
                                         for region_name, region_mask in regions.items()}
        pressure_forms[form_name] = entry

    # --- Reaction: what the fluid particles receive from the rotor ------------
    # Evaluated with the fluid particle as SELF (its M_j, its TIC branch, its ν),
    # sign flipped: the torque the rotor would feel by action = reaction.
    fluid_pairs = neighbor_kind == KIND_FLUID
    fluid_index = neighbor[fluid_pairs]
    rotor_index = self_global[fluid_pairs]
    reaction = {}
    for form_name, matrix_mode, pressure_mode in (("as GPU (M_j, TIC_j)", "self", "tic"),
                                                  ("M_j, symmetric", "self", "symmetric"),
                                                  ("pair-mean M, symmetric", "pair_mean", "symmetric")):
        reaction_pressure, reaction_viscous = momentum_pair_terms(fields, constants, fluid_index, rotor_index,
                                                                  matrix_mode, pressure_mode)
        for term, pair_acceleration in (("pressure", reaction_pressure), ("viscous", reaction_viscous)):
            pair_force = fields.mass[fluid_index][:, None] * pair_acceleration
            torque_vector = np.cross(fields.positions[fluid_index] - constants.pivot, pair_force).sum(axis=0)
            force = pair_force.sum(axis=0)
            axial_force = float(force @ constants.axis)
            reaction[f"{form_name}:{term}"] = {
                "torque_axis": -float(torque_vector @ constants.axis),
                "force_axis": -axial_force,
                "force_radial": float(np.linalg.norm(force - axial_force * constants.axis))}
    # Fluid-part viscous torque with the pair-mean matrix (rotor side), to
    # compare with the pair-mean reaction (both exactly antisymmetric).
    _, viscous_mean = momentum_pair_terms(fields, constants, self_global[fluid_pairs], neighbor[fluid_pairs],
                                          "pair_mean", "symmetric")
    pressure_forms["pair-mean M, symmetric"]["fluid_viscous"] = resultant(
        rotor_positions, rotor_mass, sum_per_particle(local_self[fluid_pairs], viscous_mean, rotor.size), constants)

    return {"validation": validation, "gpu": gpu_resultant, "cpu": cpu_resultant, "rows": rows,
            "regions": region_table, "internal_variants": variants, "internal_decomposition": decomposition,
            "internal_decomposition_force_axis": decomposition_force_axis,
            "internal_by_region": region_internal, "rotor_pressure_states": states,
            "pressure_forms": pressure_forms, "reaction": reaction}


def analyze_solid_pressure(fields, constants, kind, tree_fluid, fluid_indices, chunk_size=30000):
    """density.comp's solid pressure for every live particle of `kind` with at
    least one fluid neighbour: statistics, fits and a CPU reconstruction."""
    solids = np.nonzero(fields.kind == kind)[0]
    count = solids.size
    weight_sum = np.zeros(count)                          # Σ_j W_ij            (fluid j)
    weighted_pressure = np.zeros(count)                   # Σ_j W_ij P_j
    weighted_relative_velocity = np.zeros((count, 3))     # Σ_j W_ij (v_j - v_i)
    gradient_sum = np.zeros((count, 3))                   # Σ_j V_j ∇W_ij (raw), points into the fluid
    laplacian_weight = np.zeros(count)                    # Σ_j 2 (x_ji·∇W̃_ij) / (r² + ε_h²) V_j
    neighbour_count = np.zeros(count, dtype=np.int64)
    drift = {"previous": np.zeros(count), "current": np.zeros(count)}
    diffusion = {"previous": np.zeros(count), "current": np.zeros(count)}
    fluid_positions = fields.positions[fluid_indices]
    for start in range(0, count, chunk_size):
        chunk = solids[start:start + chunk_size]
        local_self, local_fluid = neighbour_pairs(
            fields.positions[chunk], tree_fluid, constants.smoothing_length * (1.0 + 1e-6),
            constants.smoothing_length_squared, fluid_positions)
        self_index = chunk[local_self]
        neighbor_index = fluid_indices[local_fluid]
        slot = local_self + start
        position_difference = fields.positions[self_index] - fields.positions[neighbor_index]   # x_ij
        distance_squared = np.einsum("ij,ij->i", position_difference, position_difference)
        distance = np.sqrt(distance_squared)
        gradient = kernel_gradient(position_difference, distance, constants.smoothing_length,
                                   constants.kernel_gradient_coefficient)
        corrected = np.einsum("nab,nb->na", fields.correction[self_index], gradient)          # ∇W̃_ij = M_i⁻¹ ∇W_ij
        weight = kernel_value(distance, constants.smoothing_length, constants.kernel_coefficient)
        velocity_difference = fields.velocity[neighbor_index] - fields.velocity[self_index]   # v_j - v_i
        neighbour_count += np.bincount(slot, minlength=count)
        weight_sum += np.bincount(slot, weights=weight, minlength=count)
        weighted_pressure += np.bincount(slot, weights=weight * fields.pressure[neighbor_index], minlength=count)
        weighted_relative_velocity += sum_per_particle(slot, weight[:, None] * velocity_difference, count)
        current_volume = fields.mass[neighbor_index] / fields.density[neighbor_index]
        gradient_sum += sum_per_particle(slot, current_volume[:, None] * gradient, count)
        laplacian_weight += np.bincount(
            slot, weights=2.0 * np.einsum("ij,ij->i", -position_difference, corrected)
            / (distance_squared + constants.eps_h_squared) * current_volume, minlength=count)
        # density.comp with the pre-step densities (exact inputs) and with the
        # post-step ones (the approximation available from a single readback).
        for label, self_density, neighbor_density in (
                ("previous", fields.previous_density[self_index], fields.previous_density[neighbor_index]),
                ("current", fields.density[self_index], fields.density[neighbor_index])):
            neighbor_volume = fields.mass[neighbor_index] / neighbor_density
            drift_pair = -self_density * neighbor_volume * np.einsum("ij,ij->i", velocity_difference, corrected)
            diffusion_pair = (constants.delta_coefficient * constants.smoothing_length * constants.speed_of_sound
                              * 2.0 * (neighbor_density - self_density)
                              * np.einsum("ij,ij->i", -position_difference, corrected)
                              / (distance_squared + constants.eps_h_squared) * neighbor_volume)
            drift[label] += np.bincount(slot, weights=drift_pair, minlength=count)
            diffusion[label] += np.bincount(slot, weights=diffusion_pair, minlength=count)
    has_fluid = neighbour_count > 0
    # Outward normal of the solid (pointing into the fluid): ∇W_ij = W'(r) x_ij / r
    # with W' < 0 points from i toward j, so Σ_fluid V_j ∇W_ij points into the fluid.
    normal = gradient_sum / np.maximum(np.linalg.norm(gradient_sum, axis=1), 1e-30)[:, None]

    selected = solids[has_fluid]
    solid_pressure = fields.pressure[selected]
    fluid_mean_pressure = weighted_pressure[has_fluid] / weight_sum[has_fluid]            # P̄_f (Shepard)
    # ū_n = Σ W (v_j - v_i)·n / Σ W  (< 0: fluid approaching the solid)
    mean_normal_velocity = (np.einsum("ij,ij->i", weighted_relative_velocity[has_fluid], normal[has_fluid])
                            / weight_sum[has_fluid])
    rest_density = fields.rest_density[selected]
    eos_constant = fields.eos_constant[selected]
    sound_speed_squared = constants.power * eos_constant / rest_density                   # c0² = γ B / ρ0

    reconstruction = {}
    for label in ("previous", "current"):
        self_density = (fields.previous_density if label == "previous" else fields.density)[selected]
        new_density = self_density + constants.timestep * (drift[label][has_fluid] + diffusion[label][has_fluid])
        reconstructed = eos_constant * ((new_density / rest_density) ** constants.power - 1.0)
        difference = reconstructed - solid_pressure
        reconstruction[label] = {
            "rms_difference_over_rms_pressure": float(np.sqrt(np.mean(difference ** 2)) / np.sqrt(np.mean(solid_pressure ** 2))),
            "max_abs_difference": float(np.abs(difference).max()),
            "correlation": float(np.corrcoef(reconstructed, solid_pressure)[0, 1]),
        }
    # Linearised attribution (exact pre-step inputs): P ≈ c0² dt (drift + diffusion).
    drift_pressure = sound_speed_squared * constants.timestep * drift["previous"][has_fluid]
    diffusion_pressure = sound_speed_squared * constants.timestep * diffusion["previous"][has_fluid]

    slope, intercept, correlation = fit_line(fluid_mean_pressure, solid_pressure)
    threshold = np.percentile(np.abs(fluid_mean_pressure), 75.0)
    top_quartile = np.abs(fluid_mean_pressure) > threshold
    diffusion_slope, diffusion_intercept, diffusion_correlation = fit_line(fluid_mean_pressure, diffusion_pressure)
    approach = -rest_density * constants.speed_of_sound * mean_normal_velocity             # -ρ0 c0 ū_n
    drift_gain = float(np.sum(approach * drift_pressure) / np.sum(approach * approach))  # fit through 0
    drift_correlation = float(np.corrcoef(approach, drift_pressure)[0, 1])
    # Geometric coefficients behind half-space estimates (c0² dt = CFL h c0):
    #   uniform relative velocity u:  P_drift     = -CFL (h |M_i Σ V_j ∇W_ij|) ρ0 c0 (u·Ĝ)
    #   uniform fluid pressure P_f:   P_diffusion ≈  CFL δ (h² Σ 2 (x_ji·∇W̃_ij)/(r²+ε_h²) V_j) P_f
    courant = constants.speed_of_sound * constants.timestep / constants.smoothing_length
    corrected_gradient_sum = np.einsum("nab,nb->na", fields.correction[selected], gradient_sum[has_fluid])
    drift_geometric = courant * constants.smoothing_length * np.linalg.norm(corrected_gradient_sum, axis=1)
    diffusion_geometric = (courant * constants.delta_coefficient * constants.smoothing_length ** 2
                           * laplacian_weight[has_fluid])
    rms = lambda values: float(np.sqrt(np.mean(values ** 2)))
    return {
        "particles": int(count),
        "with_fluid_neighbours": int(has_fluid.sum()),
        "fluid_neighbours_median": float(np.median(neighbour_count[has_fluid])),
        "solid_density_stored_min": float(fields.density[selected].min()),
        "solid_density_stored_max": float(fields.density[selected].max()),
        "solid_pressure_mean": float(solid_pressure.mean()),
        "solid_pressure_rms": rms(solid_pressure),
        "solid_pressure_negative_fraction": float((solid_pressure < 0).mean()),
        "fluid_mean_pressure_mean": float(fluid_mean_pressure.mean()),
        "fluid_mean_pressure_rms": rms(fluid_mean_pressure),
        "fit_slope": slope, "fit_intercept": intercept, "fit_correlation": correlation,
        "median_ratio_top_quartile": float(np.median(solid_pressure[top_quartile] / fluid_mean_pressure[top_quartile])),
        "top_quartile_threshold": float(threshold),
        "reconstruction": reconstruction,
        "drift_pressure_rms": rms(drift_pressure),
        "diffusion_pressure_rms": rms(diffusion_pressure),
        "drift_pressure_mean": float(drift_pressure.mean()),
        "diffusion_pressure_mean": float(diffusion_pressure.mean()),
        "correlation_solid_pressure_drift": float(np.corrcoef(solid_pressure, drift_pressure)[0, 1]),
        "correlation_solid_pressure_diffusion": float(np.corrcoef(solid_pressure, diffusion_pressure)[0, 1]),
        "diffusion_fit_slope": diffusion_slope, "diffusion_fit_intercept": diffusion_intercept,
        "diffusion_fit_correlation": diffusion_correlation,
        "drift_gain_vs_minus_rho_c0_un": drift_gain, "drift_correlation_vs_minus_rho_c0_un": drift_correlation,
        "mean_normal_velocity_rms": rms(mean_normal_velocity),
        "courant": float(courant),
        "drift_geometric_quartiles": np.percentile(drift_geometric, [25, 50, 75]).tolist(),
        "diffusion_geometric_quartiles": np.percentile(diffusion_geometric, [25, 50, 75]).tolist(),
    }


def analyze_snapshot(path, shaft_radius, split_height):
    from scipy.spatial import cKDTree
    archive = np.load(path)
    parameters = json.loads(str(archive["parameters"]))
    snapshot = {name: archive[name] for name in archive.files if name != "parameters"}
    fields = Fields(snapshot, parameters)
    constants = Constants(parameters)
    started = time.perf_counter()
    rotor_result = analyze_rotor(fields, constants, shaft_radius, split_height)
    fluid_indices = np.nonzero(fields.kind == KIND_FLUID)[0]
    tree_fluid = cKDTree(fields.positions[fluid_indices])
    solid_results = {label: analyze_solid_pressure(fields, constants, kind, tree_fluid, fluid_indices)
                     for label, kind in (("rotor", KIND_ROTOR), ("wall", KIND_BOUNDARY))}
    rotor_material = [entry for entry in parameters["materials"] if entry["kind"] == KIND_ROTOR][0]
    fluid_material = [entry for entry in parameters["materials"] if entry["kind"] == KIND_FLUID][0]
    return {
        "snapshot": str(path),
        "case": parameters["case"],
        "step": parameters["step"],
        "time": parameters["time"],
        "angular_velocity": constants.angular_velocity,
        "alive": int(fields.count),
        "global_status": parameters["global_status"],
        "gpu_readback": parameters["gpu_torque"],
        "constants": {
            "smoothing_length": constants.smoothing_length, "particle_spacing": parameters["particle_spacing"],
            "eps_h_squared": constants.eps_h_squared, "kernel_coefficient": constants.kernel_coefficient,
            "kernel_gradient_coefficient": constants.kernel_gradient_coefficient,
            "gravity": constants.gravity.tolist(), "timestep": constants.timestep,
            "speed_of_sound": constants.speed_of_sound, "cfl": parameters["cfl"], "power": constants.power,
            "delta_coefficient": constants.delta_coefficient,
            "eos_constant": as_float32(fluid_material["eos_constant"]),
            "rest_density": fluid_material["rest_density"],
            "rotor_viscosity": rotor_material["viscosity"], "fluid_viscosity": fluid_material["viscosity"],
            "regularization_xi": parameters["regularization_xi"],
            "use_neighbor_list": parameters["use_neighbor_list"],
        },
        "rotor": rotor_result,
        "solid_pressure": solid_results,
        "analysis_seconds": time.perf_counter() - started,
    }


# =============================================================================
# Reporting
# =============================================================================

def print_report(result):
    rotor = result["rotor"]
    validation = rotor["validation"]
    gpu_total = rotor["gpu"]["torque_axis"]
    readback_total = result["gpu_readback"]["torque_axis"]
    status = result["global_status"]
    constants = result["constants"]
    omega = result["angular_velocity"]
    np_of = lambda torque: power_number(torque, omega)
    print("=" * 100)
    print(f"{result['case']}\n  step {result['step']}  t = {result['time']:.4f} s  ω = {omega:.4f} rad/s  "
          f"alive {result['alive']:,}  rotor {validation['rotor_particles']:,}  "
          f"overflow inside/incoming/fallback/neighbor = {status['overflow_inside_count']}/"
          f"{status['overflow_incoming_count']}/{status['correction_fallback_count']}/{status['overflow_neighbor_count']}")
    print(f"  h = {constants['smoothing_length']:.6g} m  dx = {constants['particle_spacing']:.6g} m  "
          f"EPS_H_SQUARED = {constants['eps_h_squared']:.6g} m²  g = {constants['gravity']}  "
          f"ν_rotor = {constants['rotor_viscosity']:.3g} m²/s  neighbour list = {constants['use_neighbor_list']}")
    print(f"  pairs rotor->neighbour: {validation['pairs']:,} "
          f"{validation['pairs_by_neighbour_kind']} other kinds: {validation['pairs_other_kind']}")
    print("-- validation (CPU float64 re-implementation vs GPU float32) --")
    print(f"  |a_cpu - a_gpu|: max {validation['max_abs_error']:.3e} m/s² "
          f"(max |a_gpu| {validation['max_gpu_acceleration']:.3e}, rms |a_gpu| {validation['rms_gpu_acceleration']:.3e})")
    print(f"  max err / max |a| = {validation['max_error_over_max_acceleration']:.2e}   "
          f"rms err / rms |a| = {validation['rms_error_over_rms_acceleration']:.2e}   per particle "
          f"(|a| > 1e-3 max, n = {validation['per_particle_relative_error_count']}): "
          f"max {validation['per_particle_relative_error_max']:.2e}, rms {validation['per_particle_relative_error_rms']:.2e}")
    cpu_total = rotor["cpu"]["torque_axis"]
    print(f"  τ_axis: CPU {cpu_total:.6e}  GPU-acc sum {gpu_total:.6e}  readback_rotor_torque {readback_total:.6e}  "
          f"rel diff CPU vs readback {abs(cpu_total - readback_total) / abs(readback_total):.2e}")
    print("-- split of the rotor torque / force by neighbour kind (N·m, N) --")
    print(f"  {'part':22s} {'τ_axis':>12s} {'% of GPU τ':>10s} {'Np part':>8s} {'F_axis':>11s} {'|F_radial|':>11s}")
    for key in ("fluid:pressure", "fluid:viscous", "fluid:total", "rotor:pressure", "rotor:viscous", "rotor:total",
                "wall:pressure", "wall:viscous", "wall:total"):
        row = rotor["rows"][key]
        print(f"  {key:22s} {row['torque_axis']:12.5e} {100 * row['torque_axis'] / gpu_total:10.3f} "
              f"{np_of(row['torque_axis']):8.4f} {row['force_axis']:11.4e} {row['force_radial']:11.4e}")
    for label, row in (("total CPU", rotor["cpu"]), ("total GPU", rotor["gpu"])):
        print(f"  {label:22s} {row['torque_axis']:12.5e} {100 * row['torque_axis'] / gpu_total:10.3f} "
              f"{np_of(row['torque_axis']):8.4f} {row['force_axis']:11.4e} {row['force_radial']:11.4e}")
    print("-- pressure torque on the rotor by pressure form (rotor side, per neighbour kind) --")
    print(f"  {'form':26s} {'fluid':>12s} {'internal':>12s} {'wall':>12s} {'total p':>12s} {'total p+v':>12s} "
          f"{'% of GPU':>9s} {'Np':>7s} {'F_axis':>10s}")
    viscous_total = sum(rotor["rows"][f"{label}:viscous"]["torque_axis"] for label in ("fluid", "rotor", "wall"))
    for form_name, entry in rotor["pressure_forms"].items():
        total_with_viscous = entry["total"]["torque_axis"] + viscous_total
        print(f"  {form_name:26s} {entry['fluid']['torque_axis']:12.5e} {entry['rotor']['torque_axis']:12.5e} "
              f"{entry['wall']['torque_axis']:12.5e} {entry['total']['torque_axis']:12.5e} {total_with_viscous:12.5e} "
              f"{100 * total_with_viscous / gpu_total:9.3f} {np_of(total_with_viscous):7.4f} "
              f"{entry['total']['force_axis'] + sum(rotor['rows'][f'{label}:viscous']['force_axis'] for label in ('fluid', 'rotor', 'wall')):10.3e}")
    as_gpu, symmetric = rotor["pressure_forms"]["as GPU (M_i, TIC)"], rotor["pressure_forms"]["M_i, symmetric"]
    shifts = {label: as_gpu[label]["torque_axis"] - symmetric[label]["torque_axis"] for label in ("fluid", "rotor", "wall")}
    print(f"  TIC shift (as GPU - M_i symmetric): fluid {shifts['fluid']:12.5e}  internal {shifts['rotor']:12.5e}  "
          f"wall {shifts['wall']:12.5e}  net {sum(shifts.values()):12.5e} ({100 * sum(shifts.values()) / gpu_total:.3f} % of GPU)")
    print("  per region, pressure + viscous (shaft / lower / upper):")
    for form_name, entry in rotor["pressure_forms"].items():
        regions_entry = entry["regions_with_viscous"]
        print(f"    {form_name:26s} " + "  ".join(f"{name} {value:12.5e}" for name, value in regions_entry.items()))
    print("    GPU readback               " + "  ".join(
        f"{name} {result['gpu_readback'][f'torque_axis_{name}']:12.5e}" for name in ("shaft", "lower", "upper")))
    print("-- reaction: minus the torque the rotor exerts on the fluid particles (fluid-side force.comp formula) --")
    forms = list(dict.fromkeys(key.rsplit(":", 1)[0] for key in rotor["reaction"]))   # ordered, unique
    for form_name in forms:
        pressure_row = rotor["reaction"][f"{form_name}:pressure"]
        viscous_row = rotor["reaction"][f"{form_name}:viscous"]
        total = pressure_row["torque_axis"] + viscous_row["torque_axis"]
        print(f"  {form_name:26s} pressure {pressure_row['torque_axis']:12.5e}  viscous {viscous_row['torque_axis']:12.5e}  "
              f"total {total:12.5e}  ({100 * total / gpu_total:8.3f} % of GPU, Np {np_of(total):7.4f})  "
              f"F_axis {pressure_row['force_axis'] + viscous_row['force_axis']:10.3e}")
    reference = rotor["pressure_forms"]["pair-mean M, symmetric"]
    reaction_reference = rotor["reaction"]["pair-mean M, symmetric:pressure"]
    print(f"  check, pair-mean M + symmetric P (exactly antisymmetric pair force): rotor-side fluid part "
          f"τ {reference['fluid']['torque_axis']:.8e} F_axis {reference['fluid']['force_axis']:.9f}")
    print(f"  {'':78s} reaction  τ {reaction_reference['torque_axis']:.8e} F_axis {reaction_reference['force_axis']:.9f}")
    print("  (pressure only; forces equal to round-off, torques differ by the non-central pair torque Σ x_ij × f_ij)")
    print("-- internal (rotor-rotor) pressure: variants --")
    for name, row in rotor["internal_variants"].items():
        print(f"  {name:28s} τ {row['torque_axis']:12.5e}  F_axis {row['force_axis']:11.4e}  |F_radial| {row['force_radial']:10.3e}"
              f"   (Σ|pair τ| {row['sum_abs_pair_axial_torque']:.3e}, Σ|pair F| {row['sum_abs_pair_force']:.3e})")
    print("  exact decomposition of the internal pressure torque:")
    for name, value in rotor["internal_decomposition"].items():
        print(f"    {name:45s} {value:12.5e}  ({100 * value / gpu_total:7.3f} % of GPU τ)")
    print("  internal pressure torque by region (shaft / lower / upper):")
    for region_name, entry in rotor["internal_by_region"].items():
        print(f"    {region_name:6s} total {entry['internal_pressure']:12.5e}  non-central {entry['non_central']:12.5e}  "
              f"M-asym {entry['M_asymmetry']:12.5e}  TIC {entry['TIC']:12.5e}")
    print("  torque by region (fluid p / fluid v / internal p / wall p / GPU total):")
    for region_name, entry in rotor["regions"].items():
        print(f"    {region_name:6s} n={entry['particles']:5d}  {entry['fluid:pressure']:12.5e} {entry['fluid:viscous']:12.5e} "
              f"{entry['rotor:pressure']:12.5e} {entry['wall:pressure']:12.5e} {entry['gpu_total']:12.5e}")
    print(f"  rotor pressure states: {rotor['rotor_pressure_states']}")
    print("-- solid pressure (density.comp: EOS(ρ0 + dt (drift + diffusion)), fluid neighbours only) --")
    print(f"  δ = {constants['delta_coefficient']}  h = {constants['smoothing_length']:.6g}  c0 = {constants['speed_of_sound']:.6g}  "
          f"dt = {constants['timestep']:.6g}  CFL = {constants['cfl']}  B = {constants['eos_constant']:.6g} Pa  γ = {constants['power']}")
    for label, entry in result["solid_pressure"].items():
        print(f"  [{label}] {entry['with_fluid_neighbours']:,} of {entry['particles']:,} have fluid neighbours "
              f"(median {entry['fluid_neighbours_median']:.0f}); stored ρ in [{entry['solid_density_stored_min']}, "
              f"{entry['solid_density_stored_max']}]")
        print(f"     P_s mean {entry['solid_pressure_mean']:.4g} rms {entry['solid_pressure_rms']:.4g} Pa "
              f"(negative {100 * entry['solid_pressure_negative_fraction']:.1f} %);  P̄_f mean {entry['fluid_mean_pressure_mean']:.4g} "
              f"rms {entry['fluid_mean_pressure_rms']:.4g} Pa")
        print(f"     fit P_s ≈ {entry['fit_slope']:.4f} P̄_f + {entry['fit_intercept']:.4g} Pa, r = {entry['fit_correlation']:.4f}; "
              f"median P_s/P̄_f (|P̄_f| > {entry['top_quartile_threshold']:.4g} Pa) = {entry['median_ratio_top_quartile']:.4f}")
        for which, rec in entry["reconstruction"].items():
            print(f"     reconstruction with {which:8s}-step ρ: rms diff / rms P_s = {rec['rms_difference_over_rms_pressure']:.3e}, "
                  f"max |diff| = {rec['max_abs_difference']:.4g} Pa, r = {rec['correlation']:.6f}")
        print(f"     linearised parts: drift rms {entry['drift_pressure_rms']:.4g} Pa (mean {entry['drift_pressure_mean']:.4g}), "
              f"diffusion rms {entry['diffusion_pressure_rms']:.4g} Pa (mean {entry['diffusion_pressure_mean']:.4g}); "
              f"corr(P_s, drift) {entry['correlation_solid_pressure_drift']:.3f}, corr(P_s, diffusion) "
              f"{entry['correlation_solid_pressure_diffusion']:.3f}")
        print(f"     diffusion part ≈ {entry['diffusion_fit_slope']:.4f} P̄_f + {entry['diffusion_fit_intercept']:.4g} "
              f"(r = {entry['diffusion_fit_correlation']:.4f});  drift part ≈ {entry['drift_gain_vs_minus_rho_c0_un']:.4f} "
              f"(-ρ0 c0 ū_n) (r = {entry['drift_correlation_vs_minus_rho_c0_un']:.4f}, rms ū_n {entry['mean_normal_velocity_rms']:.4g} m/s)")
        drift_quartiles = ", ".join(f"{value:.4f}" for value in entry["drift_geometric_quartiles"])
        diffusion_quartiles = ", ".join(f"{value:.4f}" for value in entry["diffusion_geometric_quartiles"])
        print(f"     geometric coefficients (quartiles 25/50/75, CFL = c0 dt / h = {entry['courant']:.4f}): "
              f"drift CFL h|M Σ V ∇W| = [{drift_quartiles}], diffusion CFL δ h² Σ 2 x_ji·∇W̃/(r²+ε²) V = [{diffusion_quartiles}]")
    print(f"  (analysis took {result['analysis_seconds']:.1f} s)")


def print_summary(results):
    """One block per snapshot: the torque / force split, shares and Np."""
    print("=" * 100)
    print("SUMMARY  (τ in mN·m, F in mN; % = share of the GPU readback τ; Np = -τ ω / (ρ N³ D⁵))")
    header = (f"{'case/step':>16s} {'t[s]':>6s} {'τ GPU':>8s} {'Np':>6s} | {'fluid p':>8s} {'fluid v':>7s} "
              f"{'int p':>8s} {'int v':>7s} {'wall p':>7s} {'wall v':>7s} | {'int %':>6s} {'TIC %':>6s} "
              f"{'Masym%':>6s} {'ncen%':>6s} | {'τ cons':>8s} {'Np cons':>7s} {'GPU/cons':>8s}")
    print(header)
    for result in results:
        rotor = result["rotor"]
        rows = rotor["rows"]
        gpu_total = rotor["gpu"]["torque_axis"]
        omega = result["angular_velocity"]
        decomposition = list(rotor["internal_decomposition"].values())   # non-central, M-asym, TIC
        conservative = (rotor["reaction"]["pair-mean M, symmetric:pressure"]["torque_axis"]
                        + rotor["reaction"]["pair-mean M, symmetric:viscous"]["torque_axis"])
        label = f"{pathlib.Path(result['case']).parent.name}/{result['step']}"
        milli = lambda key: 1e3 * rows[key]["torque_axis"]
        print(f"{label:>16s} {result['time']:6.3f} {1e3 * gpu_total:8.3f} {power_number(gpu_total, omega):6.3f} | "
              f"{milli('fluid:pressure'):8.3f} {milli('fluid:viscous'):7.3f} {milli('rotor:pressure'):8.3f} "
              f"{milli('rotor:viscous'):7.1e} {milli('wall:pressure'):7.4f} {milli('wall:viscous'):7.4f} | "
              f"{100 * rows['rotor:total']['torque_axis'] / gpu_total:6.2f} {100 * decomposition[2] / gpu_total:6.2f} "
              f"{100 * decomposition[1] / gpu_total:6.3f} {100 * decomposition[0] / gpu_total:6.3f} | "
              f"{1e3 * conservative:8.3f} {power_number(conservative, omega):7.3f} {gpu_total / conservative:8.4f}")
    print(f"{'':16s} axial force F_y and |F_radial| (mN): fluid / internal / wall / total")
    for result in results:
        rows = result["rotor"]["rows"]
        label = f"{pathlib.Path(result['case']).parent.name}/{result['step']}"
        print(f"{label:>16s} F_y " + " ".join(f"{1e3 * rows[f'{name}:total']['force_axis']:9.3f}" for name in ("fluid", "rotor", "wall"))
              + f" {1e3 * result['rotor']['gpu']['force_axis']:9.3f}   |F_r| "
              + " ".join(f"{1e3 * rows[f'{name}:total']['force_radial']:9.3f}" for name in ("fluid", "rotor", "wall"))
              + f" {1e3 * result['rotor']['gpu']['force_radial']:9.3f}")


def collect_snapshot_paths(paths):
    collected = []
    for item in paths:
        path = pathlib.Path(item)
        if path.is_dir():
            collected.extend(sorted(path.glob("snapshot_*.npz")))
        else:
            collected.append(path)
    return collected


def analyze_snapshots(arguments):
    results = []
    for path in collect_snapshot_paths(arguments.paths):
        result = analyze_snapshot(path, arguments.shaft_radius, arguments.split_height)
        print_report(result)
        results.append(result)
    print_summary(results)
    if arguments.json:
        pathlib.Path(arguments.json).write_text(json.dumps(json_ready(results), indent=1), encoding="utf-8")
        print(f"wrote {arguments.json}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="phase", required=True)
    run_parser = subparsers.add_parser("run", help="GPU run + snapshot dump (solver env)")
    run_parser.add_argument("case")
    run_parser.add_argument("--steps", type=int, nargs="+", required=True)
    run_parser.add_argument("--out-dir", required=True)
    run_parser.add_argument("--device", type=int, default=None)
    run_parser.add_argument("--repository", default=None, help="repository root (default: from this file's location)")
    run_parser.add_argument("--split-height", type=float, default=0.1)
    run_parser.add_argument("--shaft-radius", type=float, default=0.007)
    analyze_parser = subparsers.add_parser("analyze", help="CPU analysis of snapshots (numpy + scipy)")
    analyze_parser.add_argument("paths", nargs="+")
    analyze_parser.add_argument("--json", default=None)
    analyze_parser.add_argument("--split-height", type=float, default=0.1)
    analyze_parser.add_argument("--shaft-radius", type=float, default=0.007)
    arguments = parser.parse_args()
    if arguments.phase == "run":
        run_snapshots(arguments)
    else:
        analyze_snapshots(arguments)


if __name__ == "__main__":
    main()
