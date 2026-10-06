"""_check_thin_plate.py — checks of the thin plate treatment (side-aware mirror, 2026-09-30).

A closed box of fluid with a plane plate of particles on one lattice plane.

  static   T1: the plate spans the whole cross-section and divides the box into two chambers.
           Gravity acts along +x, NORMAL to the plate: the left chamber stands on the plate, the
           right chamber hangs below it. Hydrostatic start in each chamber (pressure 0 at its
           upper end, x smallest). The plate carries the pressure jump rho g L_left between its
           faces. The fluid must stay at rest, the jump must stay, the force read back on the plate
           must be (pressure difference of the fluid next to the faces) * area * mass factor, and
           no particle may change sides.
           (A jump put into a chamber WITHOUT a body force behind it decays in this solver also
           without any plate, because the particle shift is not part of the continuity equation;
           `--jump` adds such a jump to the left chamber for reference.)
  edge     T2: the plate ends inside the fluid (free edges all around). No gravity, uniform
           background pressure `--background`. Nothing may move.
  rotor    T3: a paddle (rectangle through the rotor axis, axis +y) turns in the closed box,
           tip speed `--tip-speed`, background pressure `--background`. There is no reference
           solution; the checks are
             - no particle passes through the paddle (followed in the frame of the rotor),
             - the torque read back on the paddle agrees with what the fluid receives:
               residual = sum_fluid m (x x a) + torque on the paddle + torque on the walls,
               evaluated every `--budget-every` steps and averaged between two reports. The
               walls are ordinary solid particles (pairwise reaction), so the residual holds
               the error of the load on the paddle plus the loss of the fluid-fluid pairs,
             - the torque compared with the paddle of three layers of ordinary solid particles.

  inertia  T4: TWO-DIMENSIONAL. A plate of the width 2a (a line of particles through the
           origin) is spun up about its centre with the constant angular acceleration alpha
           (the ramp of the rotor). In an unbounded ideal fluid the plate needs the torque
               T = I_a alpha,   I_a = rho pi a^4 / 8   per unit depth
           (added moment of inertia of a flat plate turning about its centre). The torque the
           fluid receives from the plate (torque on the fluid + torque on the walls) is fitted
           with  T(t) = I alpha + k (alpha t)^2  (added inertia + drag of the growing speed)
           over the ramp; I / I_a is the result. I_a goes with a^4: the test measures the width
           of the plate the fluid feels. The box is `--inertia-cells` wide (default 9.4 a).

--method plate   the thin plate treatment (case block `thin_plates:`): the neighbours behind the
                 plate are wall dummies with the mirrored state of the fluid particle
--method volume  the plate particles are ordinary solid particles in one layer (the code before
                 2026-09-30), for comparison
--method thick   rotor test: three layers of ordinary solid particles
--method none    edge test: no plate at all (the noise level of the box itself)
All methods start from the same lattice.

Usage (repo root, solver env):
    python experiment/v1/checks/_check_thin_plate.py static --method plate volume
    python experiment/v1/checks/_check_thin_plate.py edge --method plate volume none
    python experiment/v1/checks/_check_thin_plate.py rotor --method plate volume thick
    python experiment/v1/checks/_check_thin_plate.py inertia --method plate volume thick
"""
import argparse
import math
import pathlib
import sys

import numpy as np
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.sph.case import load_case, _calibrate_particle_volume      # noqa: E402
from utils.sph.vulkan_context import VulkanContext                    # noqa: E402
from experiment.v1 import compile_shaders_v1                          # noqa: E402
from experiment.v1.utils.simulator_v1 import SphSimulatorV1           # noqa: E402

GRAVITY = 9.81
REST_DENSITY = 1000.0
POWER = 7.0


def write_points(path, points):
    np.savetxt(path, points, fmt="v %.7f %.7f %.7f", header=f"# {points.shape[0]} particles", comments="")


def box_lattice(cells, dx, hdx):
    """Lattice sites of the box with the inner size cells * dx, centred at the origin:
    (sites inside, sites of the wall shell, all sites, half size)."""
    cells = np.asarray(cells)
    half = 0.5 * cells * dx
    centers = [(np.arange(n) + 0.5) * dx - 0.5 * n * dx for n in cells]
    grid = np.stack(np.meshgrid(*centers, indexing="ij"), axis=-1).reshape(-1, 3)
    layers = int(math.ceil(hdx))
    extended = [(np.arange(-layers, n + layers) + 0.5) * dx - 0.5 * n * dx for n in cells]
    shell = np.stack(np.meshgrid(*extended, indexing="ij"), axis=-1).reshape(-1, 3)
    wall = shell[(np.abs(shell) > half).any(axis=1)]
    return grid, wall, shell, half, centers


def write_frame(path, points, dx):
    lo, hi = points.min(axis=0) - 0.6 * dx, points.max(axis=0) + 0.6 * dx
    with open(path, "w") as handle:
        for x in (lo[0], hi[0]):
            for y in (lo[1], hi[1]):
                for z in (lo[2], hi[2]):
                    handle.write(f"v {x:.7f} {y:.7f} {z:.7f}\n")


def case_dictionary(point_count, dx, hdx, speed_of_sound, gravity_vector, background, dashpot, viscosity,
                    pair_correction, use_pst, particles, solid_pressure="accumulate"):
    pool = int(math.ceil(point_count * 1.15 / 128) * 128)
    bound = int(math.ceil(math.sqrt(2) * hdx ** 3))
    return {
        "schema_version": 2,
        "time": {"total": None, "max_steps": None, "output_cadence": None},
        "physics": {"dimension": 3, "h": hdx * dx, "particle_radius": 0.5 * dx, "lattice": "grid",
                    "calibrate_volume": True, "speed_of_sound": speed_of_sound, "power": POWER,
                    "cfl": 0.15, "gravity": [float(v) for v in gravity_vector],
                    "background_pressure": background},
        "numerics": {"use_density_diffusion": True, "delta_coefficient": 0.1, "use_kcg_correction": True,
                     "regularization": {"xi": 0.01, "det_threshold": 1.0e-4, "frobenius_max": 10.0},
                     "use_pst": use_pst, "pst_main": 0.1, "pst_anti": 0.0005,
                     "defrag_enabled": True, "defrag_cadence": 10, "use_prefix_sum_defrag": False,
                     "solid_pressure": solid_pressure, "solid_reaction_force": True,
                     "density_diffusion_gradient_term": bool(np.any(np.asarray(gravity_vector) != 0.0)),
                     "pair_correction": pair_correction,
                     "thin_plate_dashpot": dashpot, "thin_plate_viscosity": viscosity},
        "capacities": {"pool_size": pool, "max_per_voxel": max(64, int(2 ** math.ceil(math.log2(bound * 1.3)))),
                       "max_incoming": 32, "workgroup": 128},
        "material_library": "materials.yaml",
        "geometry": {"frame": "frame.obj", "particles": particles},
    }


def build_inertia_case(directory, method, arguments):
    """T4, two-dimensional: plate = the lattice line y = 0, |x| < a, rotor axis +z through the origin."""
    directory.mkdir(parents=True, exist_ok=True)
    dx, hdx = arguments.dx, arguments.hdx
    cells = np.asarray(arguments.inertia_cells)
    plate_cells = arguments.inertia_plate_cells
    if cells[1] % 2 != 1 or (cells[0] - plate_cells) % 2 != 0:
        raise SystemExit("inertia test: odd cell count along y, the same parity of box and plate along x")
    half = 0.5 * cells * dx
    centers = [(np.arange(n) + 0.5) * dx - 0.5 * n * dx for n in cells]
    grid = np.stack(np.meshgrid(*centers, indexing="ij"), axis=-1).reshape(-1, 2)
    layers_wall = int(math.ceil(hdx))
    extended = [(np.arange(-layers_wall, n + layers_wall) + 0.5) * dx - 0.5 * n * dx for n in cells]
    shell = np.stack(np.meshgrid(*extended, indexing="ij"), axis=-1).reshape(-1, 2)
    wall = shell[(np.abs(shell) > half).any(axis=1)]
    half_a = 0.5 * plate_cells * dx
    layers = {"plate": 1, "volume": 1, "thick": arguments.thick_layers}[method]
    slab = (np.abs(grid[:, 1]) < 0.5 * layers * dx - 1e-6 * dx) & (np.abs(grid[:, 0]) < half_a)
    plate = grid[slab]
    fluid = grid[~slab]

    def lift(points):
        return np.column_stack([points, np.zeros(points.shape[0])])
    write_points(directory / "fluid.obj", lift(fluid))
    write_points(directory / "wall.obj", lift(wall))
    write_points(directory / "plate.obj", lift(plate))
    outer = half + (layers_wall + 0.6) * dx
    with open(directory / "frame.obj", "w") as handle:
        for x in (-outer[0], outer[0]):
            for y in (-outer[1], outer[1]):
                for z in (-0.5 * dx, 0.5 * dx):
                    handle.write(f"v {x:.7f} {y:.7f} {z:.7f}\n")
    plate_entry = {"file": "plate.obj", "material": "plate"}
    if method == "plate":
        plate_entry["thin_plate"] = "paddle"
    case = case_dictionary(fluid.shape[0] + wall.shape[0] + plate.shape[0], dx, hdx, arguments.speed_of_sound,
                           (0.0, 0.0, 0.0), arguments.background, arguments.dashpot,
                           not arguments.free_slip, arguments.pair_correction, not arguments.no_pst,
                           [{"file": "fluid.obj", "material": "box_fluid"},
                            {"file": "wall.obj", "material": "box_wall"}, plate_entry],
                           solid_pressure=arguments.solid_pressure)
    case["physics"]["dimension"] = 2
    case["capacities"]["max_per_voxel"] = 64
    case["rotor"] = {"axis": [0.0, 0.0, 1.0], "pivot": [0.0, 0.0, 0.0], "ramp_time": arguments.ramp_time}
    if method == "plate":
        case["thin_plates"] = [{"name": "paddle", "shape": "rectangle", "frame": "rotor",
                                "centre": [0.0, 0.0, 0.0], "normal": [0.0, 1.0, 0.0],
                                "axis_a": [1.0, 0.0, 0.0],
                                "extent": [float(half_a - arguments.outline_margin * dx), float(dx)],
                                "thickness": float(arguments.thickness), "point_measure": float(dx)}]
    (directory / "case.yaml").write_text(yaml.safe_dump(case, sort_keys=False), encoding="utf-8")
    angular_velocity = arguments.tip_speed / half_a
    materials = {"schema_version": 1,
                 "box_fluid": {"kind": "fluid", "rest_density": REST_DENSITY, "viscosity": 1.0e-6},
                 "box_wall": {"kind": "boundary", "rest_density": REST_DENSITY, "viscosity": 1.0e-6},
                 "plate": {"kind": "rotor", "rest_density": REST_DENSITY, "viscosity": 1.0e-6,
                           "rotor_angular_velocity": float(angular_velocity)}}
    (directory / "materials.yaml").write_text(yaml.safe_dump(materials, sort_keys=False), encoding="utf-8")
    geometry = {"half": half, "half_a": float(half_a), "layers": layers,
                "angular_velocity": float(angular_velocity),
                "fluid_count": int(fluid.shape[0]), "plate_count": int(plate.shape[0])}
    return directory / "case.yaml", geometry


def run_inertia(method, arguments, out):
    name = f"inertia_{method}{arguments.name_suffix}"
    case_path, geometry = build_inertia_case(out / name, method, arguments)
    case = load_case(str(case_path))
    dx = arguments.dx
    mass_factor = _calibrate_particle_volume(arguments.hdx * dx, 0.5 * dx, 2, "grid") / dx ** 2
    axis = np.array([0.0, 0.0, 1.0])
    half_a = geometry["half_a"]
    acceleration_angular = geometry["angular_velocity"] / arguments.ramp_time
    added_inertia = REST_DENSITY * math.pi * half_a ** 4 / 8.0
    steps = arguments.steps if arguments.steps_given else int(round(arguments.ramp_time / case.timestep))
    print(f"=== {name}: 2D, fluid {geometry['fluid_count']:,}, plate particles {geometry['plate_count']:,} in "
          f"{geometry['layers']} layer(s), a = {half_a * 1e3:.1f} mm = {half_a / dx:.1f} dx, box "
          f"{2e3 * geometry['half'][0]:.0f} x {2e3 * geometry['half'][1]:.0f} mm, alpha = {acceleration_angular:.1f} rad/s^2 "
          f"over {arguments.ramp_time} s, dt = {case.timestep:.3e} s, {steps} steps, mass factor {mass_factor:.4f}; "
          f"I_a alpha = {added_inertia * acceleration_angular * 1e3:.3f} mN m per m")
    with VulkanContext.create(application_name="thin_plate", enable_validation=False) as context:
        simulator = SphSimulatorV1(context, case)
        try:
            simulator.bootstrap()
            fluid_group = [m.group_id for m in case.materials if m.name == "box_fluid"][0]
            wall_group = [m.group_id for m in case.materials if m.name == "box_wall"][0]
            plate_group = [m.group_id for m in case.materials if m.name == "plate"][0]
            uid = simulator.readback_particle_uid()
            positions = simulator.readback_positions()
            material = simulator.readback_material()
            fluid = simulator.live_slot_mask(positions) & (material == fluid_group)
            previous_of_uid = np.full((int(uid.max()) + 1, 3), np.nan)
            previous_of_uid[uid[fluid]] = positions[fluid, :3]
            penetrated = set()
            history = []
            while simulator.step_count < steps:
                simulator.step()
                step = simulator.step_count
                if step % arguments.budget_every != 0 and step % arguments.leak_every != 0:
                    continue
                positions = simulator.readback_positions()
                material = simulator.readback_material()
                live = simulator.live_slot_mask(positions)
                fluid = live & (material == fluid_group)
                x = positions[fluid, :3].astype(np.float64)
                if step % arguments.leak_every == 0:
                    sample_uid = simulator.readback_particle_uid()[fluid]
                    now = rotate_about_axis(x, axis, -simulator.rotor_angle)
                    before = previous_of_uid[sample_uid]
                    crossing = before[:, 1] * now[:, 1] < 0
                    fraction = np.where(crossing, before[:, 1] / np.where(crossing, before[:, 1] - now[:, 1], 1.0), 0.0)
                    point = before + fraction[:, None] * (now - before)
                    inside = crossing & (np.abs(point[:, 0]) < half_a)
                    penetrated.update(int(u) for u in sample_uid[inside])
                    previous_of_uid[sample_uid] = now
                if step % arguments.budget_every == 0:
                    velocity_mass = simulator.readback_velocity_mass()
                    acceleration = simulator.readback_acceleration()

                    def torque_of(mask):
                        a = acceleration[mask, :3].astype(np.float64)
                        m = velocity_mass[mask, 3].astype(np.float64)
                        return float((m * (np.cross(positions[mask, :3].astype(np.float64), a) @ axis)).sum())
                    plate = live & (material == plate_group)
                    wall = live & (material == wall_group)
                    paddle_torque = torque_of(plate)
                    if method == "plate":
                        reactions = simulator.readback_thin_plate_reactions()
                        paddle_torque = float((np.cross(reactions["position"], reactions["force"]) @ axis).sum())
                    history.append((simulator.simulation_time, paddle_torque, torque_of(wall), torque_of(fluid)))
            history = np.asarray(history)
            time = history[:, 0]
            received = (history[:, 2] + history[:, 3]) / mass_factor       # what the fluid receives from the plate
            read = -history[:, 1] / mass_factor
            status = simulator.readback_global_status()
            print(f"   end: alive {status['alive_particle_count']}, overflow {status['overflow_inside_count']}/"
                  f"{status['overflow_incoming_count']}, KCG fallback {status['correction_fallback_count']}, "
                  f"through the plate {len(penetrated)}")
            print("   fit T(t) = I alpha + k (alpha t)^2 over the window        I / I_a (fluid)   I / I_a (read back)   "
                  "k / (rho a^4) (fluid)")
            for start, stop in ((0.1, 0.5), (0.1, 1.0), (0.2, 1.0), (0.3, 1.0)):
                window = (time >= start * arguments.ramp_time) & (time <= stop * arguments.ramp_time)
                if window.sum() < 8:
                    continue
                design = np.column_stack([np.full(window.sum(), acceleration_angular),
                                          (acceleration_angular * time[window]) ** 2])
                fit_fluid = np.linalg.lstsq(design, received[window], rcond=None)[0]
                fit_read = np.linalg.lstsq(design, read[window], rcond=None)[0]
                print(f"   t / ramp = {start:.1f} .. {stop:.1f} ({int(window.sum())} samples)                 "
                      f"{fit_fluid[0] / added_inertia:8.4f}        {fit_read[0] / added_inertia:8.4f}              "
                      f"{fit_fluid[1] / (REST_DENSITY * half_a ** 4):8.4f}")
            quarter = max(1, len(time) // 10)
            print("   torque the fluid receives / (I_a alpha), means of tenths of the run: "
                  + " ".join(f"{received[k * quarter:(k + 1) * quarter].mean() / (added_inertia * acceleration_angular):.3f}"
                             for k in range(10)))
            np.savetxt(out / f"{name}_history.csv", history, delimiter=",",
                       header="time,torque_plate,torque_walls,torque_fluid", comments="")
            if arguments.dump:
                positions = simulator.readback_positions()
                material = simulator.readback_material()
                live = simulator.live_slot_mask(positions)
                fluid = live & (material == fluid_group)
                np.savez(out / f"{name}_final.npz",
                         positions=rotate_about_axis(positions[fluid, :3].astype(np.float64), axis, -simulator.rotor_angle),
                         velocity=rotate_about_axis(simulator.readback_velocity_mass()[fluid, :3].astype(np.float64),
                                                    axis, -simulator.rotor_angle),
                         pressure=simulator.readback_density_pressure()[fluid, 1].astype(np.float64),
                         angle=simulator.rotor_angle, angular_velocity=geometry["angular_velocity"],
                         half_a=half_a, dx=dx, layers=geometry["layers"], time=simulator.simulation_time)
        finally:
            simulator.destroy()


def rotate_about_axis(points, axis, angle):
    """Rodrigues rotation of the rows of `points` about the unit vector `axis` (right-hand rule)."""
    cosine, sine = math.cos(angle), math.sin(angle)
    return (points * cosine + np.cross(axis, points) * sine
            + np.outer(points @ axis, axis) * (1.0 - cosine))


def build_rotor_case(directory, method, arguments):
    """T3: paddle = rectangle in the plane z = 0 through the rotor axis (+y through the origin),
    half length `half_a` along x (the tip radius), half height `half_b` along y."""
    directory.mkdir(parents=True, exist_ok=True)
    dx, hdx = arguments.dx, arguments.hdx
    cells = np.asarray(arguments.rotor_cells)
    plate_cells = arguments.rotor_plate_cells
    if cells[2] % 2 != 1:
        raise SystemExit("rotor test: odd cell count along z (a lattice plane through the axis)")
    if (cells[0] - plate_cells[0]) % 2 != 0 or (cells[1] - plate_cells[1]) % 2 != 0:
        raise SystemExit("rotor test: the cell counts of box and paddle along x and along y must have the "
                         "same parity (the outline lies half a spacing outside the outermost particles)")
    grid, wall, shell, half, centers = box_lattice(cells, dx, hdx)
    half_a = 0.5 * plate_cells[0] * dx
    half_b = 0.5 * plate_cells[1] * dx
    layers = {"plate": 1, "volume": 1, "thick": arguments.thick_layers}[method]
    slab = ((np.abs(grid[:, 2]) < 0.5 * layers * dx - 1e-6 * dx)
            & (np.abs(grid[:, 0]) < half_a) & (np.abs(grid[:, 1]) < half_b))
    plate = grid[slab]
    fluid = grid[~slab]
    write_points(directory / "fluid.obj", fluid)
    write_points(directory / "wall.obj", wall)
    write_points(directory / "plate.obj", plate)
    write_frame(directory / "frame.obj", np.vstack([fluid, wall, plate]), dx)
    plate_entry = {"file": "plate.obj", "material": "plate"}
    if method == "plate":
        plate_entry["thin_plate"] = "paddle"
    case = case_dictionary(fluid.shape[0] + wall.shape[0] + plate.shape[0], dx, hdx, arguments.speed_of_sound,
                           (0.0, -arguments.rotor_gravity, 0.0), arguments.background, arguments.dashpot,
                           not arguments.free_slip, arguments.pair_correction, not arguments.no_pst,
                           [{"file": "fluid.obj", "material": "box_fluid"},
                            {"file": "wall.obj", "material": "box_wall"}, plate_entry],
                           solid_pressure=arguments.solid_pressure)
    if arguments.rotor_gravity > 0.0:
        case["physics"]["hydrostatic_reference"] = [0.0, float(half[1]), 0.0]
    case["rotor"] = {"axis": [0.0, 1.0, 0.0], "pivot": [0.0, 0.0, 0.0], "ramp_time": arguments.ramp_time}
    if method == "plate":
        case["thin_plates"] = [{"name": "paddle", "shape": "rectangle", "frame": "rotor",
                                "centre": [0.0, 0.0, 0.0], "normal": [0.0, 0.0, 1.0],
                                "axis_a": [1.0, 0.0, 0.0],
                                "extent": [float(half_a - arguments.outline_margin * dx),
                                           float(half_b - arguments.outline_margin * dx)],
                                "thickness": float(arguments.thickness), "point_measure": float(dx * dx)}]
    (directory / "case.yaml").write_text(yaml.safe_dump(case, sort_keys=False), encoding="utf-8")
    angular_velocity = arguments.tip_speed / half_a
    materials = {"schema_version": 1,
                 "box_fluid": {"kind": "fluid", "rest_density": REST_DENSITY, "viscosity": 1.0e-6},
                 "box_wall": {"kind": "boundary", "rest_density": REST_DENSITY, "viscosity": 1.0e-6},
                 "plate": {"kind": "rotor", "rest_density": REST_DENSITY, "viscosity": 1.0e-6,
                           "rotor_angular_velocity": float(angular_velocity)}}
    (directory / "materials.yaml").write_text(yaml.safe_dump(materials, sort_keys=False), encoding="utf-8")
    geometry = {"half": half, "half_a": float(half_a), "half_b": float(half_b), "layers": layers,
                "angular_velocity": float(angular_velocity),
                "fluid_count": int(fluid.shape[0]), "plate_count": int(plate.shape[0])}
    return directory / "case.yaml", geometry


def run_rotor(method, arguments, out):
    name = f"rotor_{method}{arguments.name_suffix}"
    case_path, geometry = build_rotor_case(out / name, method, arguments)
    case = load_case(str(case_path))
    dx = arguments.dx
    mass_factor = _calibrate_particle_volume(arguments.hdx * dx, 0.5 * dx, 3, "grid") / dx ** 3
    axis = np.array([0.0, 1.0, 0.0])
    gravity_vector = np.asarray(case.physics.gravity, dtype=np.float64)
    period = 2.0 * math.pi / geometry["angular_velocity"]
    # scale of the torque: rho U^2 * (area of one half of the paddle) * (mean arm) * 2 halves
    torque_scale = (REST_DENSITY * arguments.tip_speed ** 2 * geometry["half_a"] * 2.0 * geometry["half_b"]
                    * 0.5 * geometry["half_a"] * 2.0)
    print(f"=== {name}: fluid {geometry['fluid_count']:,}, paddle particles {geometry['plate_count']:,} in "
          f"{geometry['layers']} layer(s), tip radius {geometry['half_a'] * 1e3:.1f} mm, height "
          f"{2e3 * geometry['half_b']:.1f} mm, omega {geometry['angular_velocity']:.3f} rad/s "
          f"(period {period:.4f} s = {period / case.timestep:.0f} steps), dt = {case.timestep:.3e} s, "
          f"mass factor {mass_factor:.4f}, rho U^2 A r = {torque_scale * 1e3:.3f} mN m")
    with VulkanContext.create(application_name="thin_plate", enable_validation=False) as context:
        simulator = SphSimulatorV1(context, case)
        try:
            simulator.bootstrap()
            fluid_group = [m.group_id for m in case.materials if m.name == "box_fluid"][0]
            wall_group = [m.group_id for m in case.materials if m.name == "box_wall"][0]
            plate_group = [m.group_id for m in case.materials if m.name == "plate"][0]
            uid = simulator.readback_particle_uid()
            positions = simulator.readback_positions()
            material = simulator.readback_material()
            fluid = simulator.live_slot_mask(positions) & (material == fluid_group)
            previous_of_uid = np.full((int(uid.max()) + 1, 3), np.nan)
            previous_of_uid[uid[fluid]] = positions[fluid, :3]          # rotor angle 0
            penetrated = set()
            crossing_records = []
            sums = np.zeros(7)
            samples = 0
            history = []
            report_every = max(1, arguments.steps // arguments.reports)
            print("   step     time   turns    alive   torque on the paddle   on the walls   on the fluid   residual   "
                  "residual / paddle   C read   C fluid   max speed   mean speed   through the paddle   "
                  "p min     p max   paddle off its plane   faces: pressure side   suction side   mid-plane")
            print("   (torques with the mass factor, mN m; C = torque / mass factor / (rho U^2 A r): "
                  "read = minus the torque read back on the paddle, fluid = what the fluid receives from the "
                  "paddle = torque on the fluid + torque on the walls)")
            while simulator.step_count < arguments.steps:
                simulator.step()
                step = simulator.step_count
                need_budget = step % arguments.budget_every == 0
                need_leak = step % arguments.leak_every == 0
                need_report = step % report_every == 0 or step == arguments.steps
                if not (need_budget or need_leak or need_report):
                    continue
                positions = simulator.readback_positions()
                material = simulator.readback_material()
                live = simulator.live_slot_mask(positions)
                fluid = live & (material == fluid_group)
                x = positions[fluid, :3].astype(np.float64)
                if need_leak:
                    sample_uid = simulator.readback_particle_uid()[fluid]
                    now = rotate_about_axis(x, axis, -simulator.rotor_angle)       # frame of the rotor
                    before = previous_of_uid[sample_uid]
                    crossing = before[:, 2] * now[:, 2] < 0
                    fraction = np.where(crossing, before[:, 2] / np.where(crossing, before[:, 2] - now[:, 2], 1.0), 0.0)
                    point = before + fraction[:, None] * (now - before)
                    inside = crossing & (np.abs(point[:, 0]) < geometry["half_a"]) & (np.abs(point[:, 1]) < geometry["half_b"])
                    penetrated.update(int(u) for u in sample_uid[inside])
                    if inside.any():
                        crossing_records.append(np.column_stack([point[inside, 0], point[inside, 1]]))
                    previous_of_uid[sample_uid] = now
                if need_budget or need_report:
                    velocity_mass = simulator.readback_velocity_mass()
                    acceleration = simulator.readback_acceleration()
                    def torque_of(mask, subtract_gravity):
                        a = acceleration[mask, :3].astype(np.float64)
                        if subtract_gravity:
                            a = a - gravity_vector
                        m = velocity_mass[mask, 3].astype(np.float64)
                        return float((m * (np.cross(positions[mask, :3].astype(np.float64), a) @ axis)).sum())
                    plate = live & (material == plate_group)
                    wall = live & (material == wall_group)
                    paddle_torque = torque_of(plate, True)
                    if method == "plate":
                        reactions = simulator.readback_thin_plate_reactions()
                        paddle_torque = float((np.cross(reactions["position"], reactions["force"]) @ axis).sum())
                    # gravity is along the axis: it has no torque about it, also on the fluid
                    terms = np.zeros(7)
                    terms[:3] = paddle_torque, torque_of(wall, True), torque_of(fluid, False)
                    terms[3] = terms[0] + terms[1] + terms[2]
                    # faces (2026-10-06): the +x half of the paddle moves toward -z (axis +y, omega > 0),
                    # so in the frame of the rotor the face at x z < 0 is the pressure side of either half,
                    # x z > 0 the suction side; the mid-plane layer of a solid paddle with an odd number of
                    # layers belongs to neither
                    if method == "plate":
                        face_points = reactions["position"]
                        face_torque = np.cross(face_points, reactions["force"]) @ axis
                    else:
                        face_points = positions[plate, :3].astype(np.float64)
                        face_torque = (velocity_mass[plate, 3].astype(np.float64)
                                       * (np.cross(face_points, acceleration[plate, :3].astype(np.float64) - gravity_vector) @ axis))
                    in_frame = rotate_about_axis(face_points, axis, -simulator.rotor_angle)
                    on_mid_plane = np.abs(in_frame[:, 2]) < 0.25 * dx
                    side = in_frame[:, 0] * in_frame[:, 2]
                    terms[4] = face_torque[(side < 0.0) & ~on_mid_plane].sum()
                    terms[5] = face_torque[(side > 0.0) & ~on_mid_plane].sum()
                    terms[6] = face_torque[on_mid_plane].sum()
                    sums += terms
                    samples += 1
                if not need_report:
                    continue
                mean = sums / max(samples, 1)
                sums[:] = 0.0
                samples = 0
                speed = np.linalg.norm(velocity_mass[fluid, :3].astype(np.float64), axis=1)
                pressure = simulator.readback_density_pressure()[fluid, 1].astype(np.float64)
                plate_in_frame = rotate_about_axis(positions[plate, :3].astype(np.float64), axis, -simulator.rotor_angle)
                off_plane = float(np.abs(plate_in_frame[:, 2]).max()) - 0.5 * (geometry["layers"] - 1) * dx
                status = simulator.readback_global_status()
                history.append((simulator.simulation_time, *mean, pressure.mean()))
                print(f"   {step:6d} {simulator.simulation_time:7.3f}  {simulator.rotor_angle / (2.0 * math.pi):6.2f}  "
                      f"{status['alive_particle_count']:7d}   {mean[0] * 1e3:12.4f} mN m   {mean[1] * 1e3:10.4f}   "
                      f"{mean[2] * 1e3:10.4f}   {mean[3] * 1e3:9.4f}   {mean[3] / mean[0] if mean[0] != 0 else float('nan'):12.4f}   "
                      f"{-mean[0] / mass_factor / torque_scale:7.4f}  {(mean[1] + mean[2]) / mass_factor / torque_scale:7.4f}   "
                      f"{speed.max():8.4f}   {speed.mean():9.5f}   "
                      f"{len(penetrated):10d}       {pressure.min():8.1f}  {pressure.max():8.1f}   {off_plane:.2e}   "
                      f"{mean[4] * 1e3:10.4f}   {mean[5] * 1e3:10.4f}   {mean[6] * 1e3:10.4f}")
            if crossing_records:
                crossings = np.vstack(crossing_records)
                edge_distance = np.minimum(geometry["half_a"] - np.abs(crossings[:, 0]),
                                           geometry["half_b"] - np.abs(crossings[:, 1])) / dx
                print(f"   crossings of the mid-plane inside the outline: {len(crossings)}; distance from the nearest "
                      f"edge in spacings: < 0.5: {int((edge_distance < 0.5).sum())}, 0.5-1: "
                      f"{int(((edge_distance >= 0.5) & (edge_distance < 1)).sum())}, 1-2: "
                      f"{int(((edge_distance >= 1) & (edge_distance < 2)).sum())}, > 2: {int((edge_distance >= 2).sum())}")
            status = simulator.readback_global_status()
            history = np.asarray(history)
            late = history[history[:, 0] > 0.5 * history[-1, 0]]
            print(f"   end: overflow {status['overflow_inside_count']}/{status['overflow_incoming_count']}, "
                  f"KCG fallback {status['correction_fallback_count']}; second half of the run: torque on the paddle "
                  f"{late[:, 1].mean() * 1e3:.4f} mN m (without the mass factor {late[:, 1].mean() / mass_factor * 1e3:.4f}), "
                  f"walls {late[:, 2].mean() * 1e3:.4f}, fluid {late[:, 3].mean() * 1e3:.4f}, residual "
                  f"{late[:, 4].mean() * 1e3:.4f} mN m = {100.0 * late[:, 4].mean() / late[:, 1].mean():.2f} % of the paddle; "
                  f"C read {-late[:, 1].mean() / mass_factor / torque_scale:.4f}, "
                  f"C fluid {(late[:, 2].mean() + late[:, 3].mean()) / mass_factor / torque_scale:.4f}; "
                  f"faces: pressure side {late[:, 5].mean() * 1e3:.4f} mN m, suction side {late[:, 6].mean() * 1e3:.4f}, "
                  f"mid-plane {late[:, 7].mean() * 1e3:.4f}; mean fluid pressure {late[:, 8].mean():.1f} Pa "
                  f"(a uniform pressure p gives each face p H a^2 = {late[:, 8].mean() * 2.0 * geometry['half_b'] * geometry['half_a'] ** 2 * 1e3:.4f} mN m)")
            np.savetxt(out / f"{name}_history.csv", history, delimiter=",",
                       header="time,torque_paddle,torque_walls,torque_fluid,residual,pressure_side,suction_side,mid_plane,"
                              "mean_pressure", comments="")
        finally:
            simulator.destroy()


def build_case(directory, test, method, dx, hdx, cells, thickness, speed_of_sound, gravity, background,
               dashpot, viscosity, pair_correction, plate_cells, use_pst=True):
    """Box with the inner size cells * dx, centred at the origin. The plate lies on the lattice
    plane x = x_plate (a lattice column), normal +x."""
    directory.mkdir(parents=True, exist_ok=True)
    cells = np.asarray(cells)
    grid, wall, shell, half, centers = box_lattice(cells, dx, hdx)
    layers = int(math.ceil(hdx))

    x_plate = centers[0][cells[0] // 2]                     # a lattice column next to the middle
    if test == "static":
        half_b = half[1] + layers * dx                      # through the walls
        half_c = half[2] + layers * dx
    else:
        half_b = 0.5 * plate_cells[0] * dx
        half_c = 0.5 * plate_cells[1] * dx
    # quadrature points: the lattice sites of the plane x = x_plate inside the outline
    plane = shell[np.abs(shell[:, 0] - x_plate) < 1e-6 * dx]
    plate = plane[(np.abs(plane[:, 1]) < half_b) & (np.abs(plane[:, 2]) < half_c)]
    # the plate takes the place of one lattice plane
    clearance = 0.5 * dx
    def in_slab(points):
        return ((np.abs(points[:, 0] - x_plate) < clearance - 1e-6 * dx)
                & (np.abs(points[:, 1]) < half_b) & (np.abs(points[:, 2]) < half_c))
    if method == "none":
        # control: the box without any plate
        if test != "edge":
            raise SystemExit("--method none belongs to the edge test")
        fluid = grid
        plate = plate[:0]
    else:
        fluid = grid[~in_slab(grid)]
        wall = wall[~in_slab(wall)]

    write_points(directory / "fluid.obj", fluid)
    write_points(directory / "wall.obj", wall)
    write_frame(directory / "frame.obj", np.vstack([fluid, wall, plate]), dx)
    particles = [{"file": "fluid.obj", "material": "box_fluid"},
                 {"file": "wall.obj", "material": "box_wall"}]
    if method != "none":
        write_points(directory / "plate.obj", plate)
        plate_entry = {"file": "plate.obj", "material": "plate"}
        if method == "plate":
            plate_entry["thin_plate"] = "divider"
        particles.append(plate_entry)
    case = case_dictionary(fluid.shape[0] + wall.shape[0] + plate.shape[0], dx, hdx, speed_of_sound,
                           (gravity, 0.0, 0.0), background, dashpot, viscosity, pair_correction, use_pst,
                           particles)
    if method == "plate":
        case["thin_plates"] = [{"name": "divider", "shape": "rectangle", "frame": "static",
                                "centre": [float(x_plate), 0.0, 0.0], "normal": [1.0, 0.0, 0.0],
                                "axis_a": [0.0, 1.0, 0.0], "extent": [float(half_b), float(half_c)],
                                "thickness": float(thickness), "point_measure": float(dx * dx)}]
    (directory / "case.yaml").write_text(yaml.safe_dump(case, sort_keys=False), encoding="utf-8")
    materials = {"schema_version": 1,
                 "box_fluid": {"kind": "fluid", "rest_density": REST_DENSITY, "viscosity": 1.0e-6},
                 "box_wall": {"kind": "boundary", "rest_density": REST_DENSITY, "viscosity": 1.0e-6},
                 "plate": {"kind": "boundary", "rest_density": REST_DENSITY, "viscosity": 1.0e-6}}
    (directory / "materials.yaml").write_text(yaml.safe_dump(materials, sort_keys=False), encoding="utf-8")
    geometry = {"x_plate": float(x_plate), "half": half, "half_b": float(half_b), "half_c": float(half_c),
                "face_right": float(x_plate + 0.5 * dx),
                "fluid_count": int(fluid.shape[0]), "plate_count": int(plate.shape[0]),
                "wetted_area": float(4.0 * min(half_b, half[1]) * min(half_c, half[2]))}
    return directory / "case.yaml", geometry


def hydrostatic_pressure(x, geometry, gravity):
    """Pressure of the fluid at rest: gravity along +x, zero at the upper end of each chamber
    (the left wall for the left chamber, the right face of the plate for the right chamber)."""
    upper_end = np.where(x < geometry["x_plate"], -geometry["half"][0], geometry["face_right"])
    return REST_DENSITY * gravity * (x - upper_end)


def initial_density(simulator, case, geometry, gravity, jump, speed_of_sound):
    """Hydrostatic density with the reference at the lid, plus `jump` Pa in the left chamber,
    for the fluid AND the wall particles (a wall that starts at rho0 next to a fluid under
    pressure lets the fluid expand until its own pressure has built up; in a chamber of a few
    thousand particles that alone takes hundreds of Pa). The plate particles keep their marker."""
    positions = simulator._build_initial_data()["position_voxel_id"]
    positions = np.frombuffer(positions, dtype=np.float32).reshape(-1, 4)
    current = np.frombuffer(simulator._build_initial_data()["density_pressure"], dtype=np.float32).reshape(-1, 2).copy()
    material = np.frombuffer(simulator._build_initial_data()["material"], dtype=np.uint32)
    fluid = current[:, 0] > 0.0                       # everything except the plate markers and empty slots
    fluid[0] = False
    eos_constant = speed_of_sound ** 2 * REST_DENSITY / POWER
    pressure = (hydrostatic_pressure(positions[:, 0].astype(np.float64), geometry, gravity)
                + np.where(positions[:, 0] < geometry["x_plate"], jump, 0.0))
    pressure = np.maximum(pressure, 0.0)              # a solid in accumulate mode never goes below rho0
    density = REST_DENSITY * (1.0 + pressure / eos_constant) ** (1.0 / POWER)
    current[fluid, 0] = density[fluid].astype(np.float32)
    simulator._staging_upload(simulator.buffers["density_pressure"], current.tobytes())


def run(test, method, arguments, out):
    name = f"{test}_{method}"
    gravity = arguments.gravity if test == "static" else 0.0
    background = 0.0 if test == "static" else arguments.background
    case_path, geometry = build_case(
        out / name, test, method, arguments.dx, arguments.hdx, arguments.cells, arguments.thickness,
        arguments.speed_of_sound, gravity, background, arguments.dashpot,
        not arguments.free_slip, arguments.pair_correction, arguments.plate_cells,
        use_pst=not arguments.no_pst)
    case = load_case(str(case_path))
    dx = arguments.dx
    jump = arguments.jump if test == "static" else 0.0
    mass_factor = _calibrate_particle_volume(arguments.hdx * dx, 0.5 * dx, 3, "grid") / dx ** 3
    print(f"=== {name}: fluid {geometry['fluid_count']:,}, plate points {geometry['plate_count']:,}, "
          f"plate at x = {geometry['x_plate'] * 1e3:.1f} mm, thickness {arguments.thickness * 1e3:.2f} mm, "
          f"dt = {case.timestep:.3e} s, mass factor {mass_factor:.4f}")
    with VulkanContext.create(application_name="thin_plate", enable_validation=False) as context:
        simulator = SphSimulatorV1(context, case)
        try:
            if test == "static":
                initial_density(simulator, case, geometry, gravity, jump, arguments.speed_of_sound)
            simulator.bootstrap()
            uid = simulator.readback_particle_uid()
            positions = simulator.readback_positions()
            material = simulator.readback_material()
            live = simulator.live_slot_mask(positions)
            fluid_group = [m.group_id for m in case.materials if m.kind == 0][0]
            plate_groups = [m.group_id for m in case.materials if m.name == "plate"]
            plate_group = plate_groups[0] if plate_groups and method != "none" else 0xFFFFFFFF
            side_of_uid = np.zeros(uid.max() + 1, dtype=np.int8)
            fluid = live & (material == fluid_group)
            side_of_uid[uid[fluid]] = np.sign(positions[fluid, 0] - geometry["x_plate"]).astype(np.int8)
            over_plate_of_uid = np.zeros(uid.max() + 1, dtype=bool)
            over_plate_of_uid[uid[fluid]] = ((np.abs(positions[fluid, 1]) < geometry["half_b"])
                                             & (np.abs(positions[fluid, 2]) < geometry["half_c"]))
            report_every = max(1, arguments.steps // arguments.reports)
            # penetration: a particle whose path between two samples crosses the mid-plane of the
            # plate inside its outline. Sampled often enough for the path to be a straight line.
            previous_position_of_uid = np.full((uid.max() + 1, 3), np.nan)
            previous_position_of_uid[uid[fluid]] = positions[fluid, :3]
            penetrated = set()
            penetration_points = []
            print("   step     time    alive   p left    p right  difference   near plate   F_x plate   F_x / (dp A)   "
                  "max speed   mean speed   through the plate   p min     p max   mean speed within 1.5 dx of the plate")
            while simulator.step_count < arguments.steps:
                simulator.step()
                if simulator.step_count % arguments.leak_every == 0:
                    sample_positions = simulator.readback_positions()
                    sample_live = simulator.live_slot_mask(sample_positions) & (simulator.readback_material() == fluid_group)
                    sample_uid = simulator.readback_particle_uid()[sample_live]
                    now = sample_positions[sample_live, :3].astype(np.float64)
                    before = previous_position_of_uid[sample_uid]
                    d_before = before[:, 0] - geometry["x_plate"]
                    d_now = now[:, 0] - geometry["x_plate"]
                    crossing = d_before * d_now < 0
                    fraction = np.where(crossing, d_before / np.where(crossing, d_before - d_now, 1.0), 0.0)
                    point = before + fraction[:, None] * (now - before)
                    inside = crossing & (np.abs(point[:, 1]) < geometry["half_b"]) & (np.abs(point[:, 2]) < geometry["half_c"])
                    penetrated.update(int(u) for u in sample_uid[inside])
                    if inside.any():
                        penetration_points.append(np.column_stack([
                            point[inside, 1], point[inside, 2], d_before[inside], d_now[inside],
                            np.full(int(inside.sum()), simulator.simulation_time)]))
                    previous_position_of_uid[sample_uid] = now
                if (simulator.step_count % report_every != 0 and simulator.step_count != arguments.steps
                        and simulator.step_count not in arguments.report_steps):
                    continue
                positions = simulator.readback_positions()
                material = simulator.readback_material()
                live = simulator.live_slot_mask(positions)
                fluid = live & (material == fluid_group)
                x = positions[fluid, :3].astype(np.float64)
                velocity_mass = simulator.readback_velocity_mass()
                speed = np.linalg.norm(velocity_mass[fluid, :3].astype(np.float64), axis=1)
                pressure = simulator.readback_density_pressure()[fluid, 1].astype(np.float64)
                hydrostatic = hydrostatic_pressure(x[:, 0], geometry, gravity)
                excess = pressure - hydrostatic - case.physics.background_pressure
                left = x[:, 0] < geometry["x_plate"]
                plate = live & (material == plate_group)
                if method == "plate":
                    force = simulator.readback_thin_plate_reactions()["force"].sum(axis=0)
                else:
                    acceleration = simulator.readback_acceleration()[plate, :3].astype(np.float64)
                    mass = velocity_mass[plate, 3].astype(np.float64)
                    force = ((acceleration - np.asarray(case.physics.gravity)) * mass[:, None]).sum(axis=0)
                # pressure difference of the fluid within two spacings of the faces
                distance = x[:, 0] - geometry["x_plate"]
                over = (np.abs(x[:, 1]) < geometry["half_b"]) & (np.abs(x[:, 2]) < geometry["half_c"])
                band = 0.5 * arguments.thickness + 2.0 * dx
                near_left = over & (distance < 0) & (distance > -band)
                near_right = over & (distance > 0) & (distance < band)
                # continued to the faces with the hydrostatic gradient
                face_left = geometry["x_plate"] - (geometry["face_right"] - geometry["x_plate"])
                at_left = pressure[near_left] + REST_DENSITY * gravity * (face_left - x[near_left, 0])
                at_right = pressure[near_right] + REST_DENSITY * gravity * (geometry["face_right"] - x[near_right, 0])
                near_difference = (at_left.mean() - at_right.mean()
                                   if near_left.any() and near_right.any() else float("nan"))
                expected = near_difference * geometry["wetted_area"] * mass_factor
                uid = simulator.readback_particle_uid()[fluid]
                now_side = np.sign(x[:, 0] - geometry["x_plate"])
                crossed = int(((now_side * side_of_uid[uid] < 0) & over_plate_of_uid[uid]
                               & (np.abs(x[:, 1]) < geometry["half_b"]) & (np.abs(x[:, 2]) < geometry["half_c"])).sum())
                status = simulator.readback_global_status()
                ratio = force[0] / expected if abs(expected) > 1e-9 else float("nan")
                # agitation of the fluid next to the plate: within 1.5 spacings of the outline
                in_plane_gap = np.sqrt(np.maximum(np.abs(x[:, 1]) - geometry["half_b"], 0.0) ** 2
                                       + np.maximum(np.abs(x[:, 2]) - geometry["half_c"], 0.0) ** 2)
                next_to_plate = np.sqrt(distance ** 2 + in_plane_gap ** 2) < 1.5 * dx
                near_plate_speed = speed[next_to_plate].mean() if next_to_plate.any() else float("nan")
                if arguments.profile:
                    # profile across the plate, central part of the cross-section
                    central = (np.abs(x[:, 1]) < 0.5 * geometry["half"][1]) & (np.abs(x[:, 2]) < 0.5 * geometry["half"][2])
                    if test == "edge":
                        central = (np.abs(x[:, 1]) < 0.5 * geometry["half_b"]) & (np.abs(x[:, 2]) < 0.5 * geometry["half_c"])
                    velocity = velocity_mass[fluid, :3].astype(np.float64)
                    shift = simulator.readback_shift()[fluid, :3].astype(np.float64)
                    raw = simulator._readback_buffer(simulator.buffers["density_gradient_kernel_sum"])
                    kernel_sum = np.frombuffer(raw, dtype=np.float32).reshape(-1, 4)[fluid, 3].astype(np.float64)
                    stride = 8
                    raw = simulator._readback_buffer(simulator.buffers["correction_inverse"])
                    matrix_xx = np.frombuffer(raw, dtype=np.float32).reshape(-1, stride)[fluid, 0].astype(np.float64)
                    edges = np.arange(-6.0, 6.01, 0.5)
                    print("      x/dx from the mid-plane: count, mean distance, excess pressure, u_x, speed, "
                          "shift_x / dx per step, kernel sum, B_xx")
                    for a, b in zip(edges[:-1], edges[1:]):
                        m = central & (distance / dx >= a) & (distance / dx < b)
                        if m.any():
                            print(f"      [{a:+5.1f}, {b:+5.1f}): {int(m.sum()):4d}  {distance[m].mean() / dx:+7.3f}  "
                                  f"{excess[m].mean():9.2f}  {velocity[m, 0].mean():+9.5f}  {speed[m].mean():8.5f}  "
                                  f"{shift[m, 0].mean() / dx:+.3e}  {kernel_sum[m].mean():.4f}  {matrix_xx[m].mean():.4f}")
                print(f"   {simulator.step_count:6d} {simulator.simulation_time:7.3f}  {status['alive_particle_count']:7d}  "
                      f"{excess[left].mean():8.2f}  {excess[~left].mean():8.2f}  {excess[left].mean() - excess[~left].mean():9.2f}   "
                      f"{near_difference:9.2f}   {force[0]:9.4f}   {ratio:10.4f}     {speed.max():8.5f}   {speed.mean():9.6f}   {len(penetrated):10d}          "
                      f"{excess.min():8.1f}  {excess.max():8.1f}   {near_plate_speed:8.5f}")
            if arguments.dump:
                np.savez(out / f"{name}_final.npz", positions=x, velocity=velocity_mass[fluid, :3],
                         pressure=pressure, uid=uid, x_plate=geometry["x_plate"],
                         half_b=geometry["half_b"], half_c=geometry["half_c"], half=geometry["half"],
                         penetrated=np.asarray(sorted(penetrated), dtype=np.int64))
            if penetration_points:
                crossings = np.vstack(penetration_points)
                edge_distance = np.minimum(geometry["half_b"] - np.abs(crossings[:, 0]),
                                           geometry["half_c"] - np.abs(crossings[:, 1])) / dx
                print(f"   crossings of the mid-plane inside the outline: {len(crossings)}; distance from the nearest "
                      f"edge in spacings: < 0.5: {int((edge_distance < 0.5).sum())}, 0.5-1: "
                      f"{int(((edge_distance >= 0.5) & (edge_distance < 1)).sum())}, 1-2: "
                      f"{int(((edge_distance >= 1) & (edge_distance < 2)).sum())}, > 2: {int((edge_distance >= 2).sum())}; "
                      f"from the left to the right {int((crossings[:, 2] < 0).sum())}, back {int((crossings[:, 2] > 0).sum())}")
            status = simulator.readback_global_status()
            print(f"   end: overflow {status['overflow_inside_count']}/{status['overflow_incoming_count']}, "
                  f"KCG fallback {status['correction_fallback_count']}, F_y, F_z on the plate {force[1]:.4f}, {force[2]:.4f} N; "
                  f"jump {jump:g} Pa on the wetted area would give F_x = {jump * geometry['wetted_area'] * mass_factor:.4f} N")
        finally:
            simulator.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("test", choices=("static", "edge", "rotor", "inertia"))
    parser.add_argument("--inertia-cells", type=int, nargs=2, default=[61, 61],
                        help="inertia test: inner size of the 2D box in cells (y odd)")
    parser.add_argument("--inertia-plate-cells", type=int, default=13,
                        help="inertia test: width of the plate in cells")
    parser.add_argument("--method", nargs="+", default=["plate"], choices=("plate", "volume", "thick", "none"))
    parser.add_argument("--rotor-cells", type=int, nargs=3, default=[31, 18, 31],
                        help="rotor test: inner size of the box in cells (x and z odd)")
    parser.add_argument("--rotor-plate-cells", type=int, nargs=2, default=[13, 6],
                        help="rotor test: paddle size in cells (x, odd; y)")
    parser.add_argument("--tip-speed", type=float, default=1.0, help="rotor test: tip speed, m/s")
    parser.add_argument("--thick-layers", type=int, default=3,
                        help="rotor test, --method thick: layers of ordinary solid particles (odd)")
    parser.add_argument("--name-suffix", default="", help="appended to the names of the output files")
    parser.add_argument("--solid-pressure", default="accumulate",
                        choices=("increment", "mirror", "mirror_tic", "accumulate", "extrapolate"),
                        help="rotor test: numerics.solid_pressure of the ordinary solid particles")
    parser.add_argument("--outline-margin", type=float, default=0.0,
                        help="rotor test, --method plate: the outline given to the solver lies this many "
                             "spacings INSIDE the outline of the paddle (the sheet of particles stays)")
    parser.add_argument("--ramp-time", type=float, default=0.05, help="rotor test: spin-up time, s")
    parser.add_argument("--rotor-gravity", type=float, default=0.0,
                        help="rotor test: gravity along -y (the rotor axis), m/s^2, hydrostatic start")
    parser.add_argument("--budget-every", type=int, default=4,
                        help="rotor test: steps between two evaluations of the torques")
    parser.add_argument("--dx", type=float, default=0.004)
    parser.add_argument("--hdx", type=float, default=3.0)
    parser.add_argument("--cells", type=int, nargs=3, default=[24, 30, 12])
    parser.add_argument("--plate-cells", type=int, nargs=2, default=[14, 6], help="edge test: plate size in cells (y, z)")
    parser.add_argument("--thickness", type=float, default=0.0029, help="plate thickness, m")
    parser.add_argument("--speed-of-sound", type=float, default=20.0)
    parser.add_argument("--jump", type=float, default=0.0,
                        help="static test: extra pressure of the left chamber at the start, Pa")
    parser.add_argument("--background", type=float, default=1000.0,
                        help="edge and rotor test: background pressure, Pa")
    parser.add_argument("--gravity", type=float, default=GRAVITY, help="static test: gravity, m/s^2")
    parser.add_argument("--dashpot", type=float, default=0.0,
                        help="numerics.thin_plate_dashpot (acoustic term of the dummy pressure)")
    parser.add_argument("--free-slip", action="store_true")
    parser.add_argument("--no-pst", action="store_true", help="switch the particle shift off")
    parser.add_argument("--pair-correction", default="reverse", choices=("own", "mean", "reverse"))
    parser.add_argument("--steps", type=int, default=None,
                        help="default 20000; inertia test: the ramp")
    parser.add_argument("--reports", type=int, default=10)
    parser.add_argument("--report-steps", type=int, nargs="*", default=[], help="additional steps to report")
    parser.add_argument("--profile", action="store_true", help="print the profile across the plate at every report")
    parser.add_argument("--leak-every", type=int, default=20, help="steps between two samples of the penetration check")
    parser.add_argument("--dump", action="store_true", help="save the final fluid state")
    parser.add_argument("--out", default="output/thin_plate")
    arguments = parser.parse_args()
    arguments.steps_given = arguments.steps is not None
    if arguments.steps is None:
        arguments.steps = 20000
    compile_shaders_v1.compile_v1_shaders()
    out = pathlib.Path(arguments.out)
    for method in arguments.method:
        if arguments.test == "inertia":
            if method == "none":
                raise SystemExit("--method none belongs to the edge test")
            run_inertia(method, arguments, out)
            continue
        if arguments.test == "rotor":
            if method == "none":
                raise SystemExit("--method none belongs to the edge test")
            run_rotor(method, arguments, out)
            continue
        if method == "thick":
            raise SystemExit("--method thick belongs to the rotor test")
        run(arguments.test, method, arguments, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
