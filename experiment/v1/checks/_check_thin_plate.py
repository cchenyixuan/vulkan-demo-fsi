"""_check_thin_plate.py — checks of the thin plate (normal flux) treatment (2026-09-30).

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

--method plate   the thin plate treatment (case block `thin_plates:`)
--method volume  the plate particles are ordinary solid particles in one layer (the code before
                 2026-09-30), for comparison

Usage (repo root, solver env):
    python experiment/v1/checks/_check_thin_plate.py static --method plate volume
    python experiment/v1/checks/_check_thin_plate.py edge --form difference tic
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


def build_case(directory, test, method, dx, hdx, cells, thickness, speed_of_sound, gravity, background,
               form, dashpot, viscosity, pair_correction, plate_cells, use_pst=True, penalty=0.1):
    """Box with the inner size cells * dx, centred at the origin. The plate lies on the lattice
    plane x = x_plate (a lattice column), normal +x."""
    directory.mkdir(parents=True, exist_ok=True)
    cells = np.asarray(cells)
    half = 0.5 * cells * dx
    centers = [(np.arange(n) + 0.5) * dx - 0.5 * n * dx for n in cells]
    grid = np.stack(np.meshgrid(*centers, indexing="ij"), axis=-1).reshape(-1, 3)
    layers = int(math.ceil(hdx))
    extended = [(np.arange(-layers, n + layers) + 0.5) * dx - 0.5 * n * dx for n in cells]
    shell = np.stack(np.meshgrid(*extended, indexing="ij"), axis=-1).reshape(-1, 3)
    wall = shell[(np.abs(shell) > half).any(axis=1)]

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
    # fluid and wall sites inside the slab of the plate, half a spacing beyond each face
    clearance = 0.5 * thickness + 0.5 * dx if method == "plate" else 0.5 * dx
    def in_slab(points):
        return ((np.abs(points[:, 0] - x_plate) < clearance - 1e-6 * dx)
                & (np.abs(points[:, 1]) < half_b) & (np.abs(points[:, 2]) < half_c))
    fluid = grid[~in_slab(grid)]
    wall = wall[~in_slab(wall)]

    write_points(directory / "fluid.obj", fluid)
    write_points(directory / "wall.obj", wall)
    write_points(directory / "plate.obj", plate)
    points = np.vstack([fluid, wall, plate])
    lo, hi = points.min(axis=0) - 0.6 * dx, points.max(axis=0) + 0.6 * dx
    with open(directory / "frame.obj", "w") as handle:
        for x in (lo[0], hi[0]):
            for y in (lo[1], hi[1]):
                for z in (lo[2], hi[2]):
                    handle.write(f"v {x:.7f} {y:.7f} {z:.7f}\n")
    pool = int(math.ceil(points.shape[0] * 1.15 / 128) * 128)
    bound = int(math.ceil(math.sqrt(2) * hdx ** 3))
    plate_entry = {"file": "plate.obj", "material": "plate"}
    if method == "plate":
        plate_entry["thin_plate"] = "divider"
    case = {
        "schema_version": 2,
        "time": {"total": None, "max_steps": None, "output_cadence": None},
        "physics": {"dimension": 3, "h": hdx * dx, "particle_radius": 0.5 * dx, "lattice": "grid",
                    "calibrate_volume": True, "speed_of_sound": speed_of_sound, "power": POWER,
                    "cfl": 0.15, "gravity": [gravity, 0.0, 0.0], "background_pressure": background},
        "numerics": {"use_density_diffusion": True, "delta_coefficient": 0.1, "use_kcg_correction": True,
                     "regularization": {"xi": 0.01, "det_threshold": 1.0e-4, "frobenius_max": 10.0},
                     "use_pst": use_pst, "pst_main": 0.1, "pst_anti": 0.0005,
                     "defrag_enabled": True, "defrag_cadence": 10, "use_prefix_sum_defrag": False,
                     "solid_pressure": "accumulate", "solid_reaction_force": True,
                     "density_diffusion_gradient_term": gravity > 0.0,
                     "pair_correction": pair_correction,
                     "thin_plate_pressure_form": form, "thin_plate_dashpot": dashpot,
                     "thin_plate_viscosity": viscosity, "thin_plate_penalty": penalty},
        "capacities": {"pool_size": pool, "max_per_voxel": max(64, int(2 ** math.ceil(math.log2(bound * 1.3)))),
                       "max_incoming": 32, "workgroup": 128},
        "material_library": "materials.yaml",
        "geometry": {"frame": "frame.obj",
                     "particles": [{"file": "fluid.obj", "material": "box_fluid"},
                                   {"file": "wall.obj", "material": "box_wall"},
                                   plate_entry]},
    }
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
                "face_right": float(x_plate + (0.5 * thickness if method == "plate" else 0.5 * dx)),
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
    name = f"{test}_{method}" + (f"_{arguments.current_form}" if method == "plate" else "")
    gravity = arguments.gravity if test == "static" else 0.0
    background = 0.0 if test == "static" else arguments.background
    case_path, geometry = build_case(
        out / name, test, method, arguments.dx, arguments.hdx, arguments.cells, arguments.thickness,
        arguments.speed_of_sound, gravity, background, arguments.current_form, arguments.dashpot,
        not arguments.free_slip, arguments.pair_correction, arguments.plate_cells,
        use_pst=not arguments.no_pst, penalty=arguments.penalty)
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
            plate_group = [m.group_id for m in case.materials if m.name == "plate"][0]
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
                  "max speed   mean speed   through the plate   p min     p max")
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
                if arguments.profile:
                    # profile across the plate, central part of the cross-section
                    central = (np.abs(x[:, 1]) < 0.5 * geometry["half"][1]) & (np.abs(x[:, 2]) < 0.5 * geometry["half"][2])
                    if test == "edge":
                        central = (np.abs(x[:, 1]) < 0.5 * geometry["half_b"]) & (np.abs(x[:, 2]) < 0.5 * geometry["half_c"])
                    velocity = velocity_mass[fluid, :3].astype(np.float64)
                    shift = simulator.readback_shift()[fluid, :3].astype(np.float64)
                    raw = simulator._readback_buffer(simulator.buffers["density_gradient_kernel_sum"])
                    kernel_sum = np.frombuffer(raw, dtype=np.float32).reshape(-1, 4)[fluid, 3].astype(np.float64)
                    stride = 12 if case.thin_plate_count > 0 else 8
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
                      f"{excess.min():8.1f}  {excess.max():8.1f}")
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
    parser.add_argument("test", choices=("static", "edge"))
    parser.add_argument("--method", nargs="+", default=["plate"], choices=("plate", "volume"))
    parser.add_argument("--form", nargs="+", default=["difference"], choices=("difference", "tic"))
    parser.add_argument("--dx", type=float, default=0.004)
    parser.add_argument("--hdx", type=float, default=3.0)
    parser.add_argument("--cells", type=int, nargs=3, default=[24, 30, 12])
    parser.add_argument("--plate-cells", type=int, nargs=2, default=[14, 6], help="edge test: plate size in cells (y, z)")
    parser.add_argument("--thickness", type=float, default=0.0029, help="plate thickness, m")
    parser.add_argument("--speed-of-sound", type=float, default=20.0)
    parser.add_argument("--jump", type=float, default=0.0,
                        help="static test: extra pressure of the left chamber at the start, Pa")
    parser.add_argument("--background", type=float, default=1000.0, help="edge test: background pressure, Pa")
    parser.add_argument("--gravity", type=float, default=GRAVITY, help="static test: gravity, m/s^2")
    parser.add_argument("--dashpot", type=float, default=1.0)
    parser.add_argument("--free-slip", action="store_true")
    parser.add_argument("--penalty", type=float, default=0.1, help="numerics.thin_plate_penalty")
    parser.add_argument("--no-pst", action="store_true", help="switch the particle shift off")
    parser.add_argument("--pair-correction", default="reverse", choices=("own", "mean", "reverse"))
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--reports", type=int, default=10)
    parser.add_argument("--report-steps", type=int, nargs="*", default=[], help="additional steps to report")
    parser.add_argument("--profile", action="store_true", help="print the profile across the plate at every report")
    parser.add_argument("--leak-every", type=int, default=20, help="steps between two samples of the penetration check")
    parser.add_argument("--dump", action="store_true", help="save the final fluid state")
    parser.add_argument("--out", default="output/thin_plate")
    arguments = parser.parse_args()
    compile_shaders_v1.compile_v1_shaders()
    out = pathlib.Path(arguments.out)
    for method in arguments.method:
        forms = arguments.form if method == "plate" else ["difference"]
        for form in forms:
            arguments.current_form = form
            run(arguments.test, method, arguments, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
