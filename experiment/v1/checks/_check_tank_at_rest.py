"""
_check_tank_at_rest.py — the 30 L tank with the impeller stopped (2026-10-03).

At rest, with zero gravity and a uniform background pressure, every load on the walls must vanish and
the fluid must stay still. The case is run with the rotor speed set to 0 and the torque about the axis
is reported per part: the cylinder wall, the lid, the floor, the ordinary wall particles at
100 < r < 139.5 mm next to each baffle (brackets, or the baffles themselves when they are lattice
particles or conformal sheets) and, for thin-plate baffles, the reaction of every plate and its normal
force. Also the mean and maximum fluid speed. The last state is written to CASE_DIR/rest_dump.npz.

    geometry CASE_DIR        fluid gap on both sides of every static plate (from fluid.obj)
    run CASE_DIR STEPS       the run described above

Findings (log/2026-10-03_fluent-and-wall-torque.md): the force on thin-plate baffles is proportional to
the background pressure (zero for p_b = 0) and depends on how the plate cuts the lattice; with
p_b = 2000 Pa even a tank made of lattice particles only starts to move after about 0.05 s (the cubic
lattice is an unstable equilibrium under a background pressure), so the flow speed at rest says little
about the plates, the plate forces do.

usage (repo root, solver environment):
    python experiment/v1/checks/_check_tank_at_rest.py run cases/stirred_tank_30l_4mm 4000
"""
import pathlib
import sys

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.sph.case import load_case                  # noqa: E402
from utils.sph.obj_loader import load_obj_vertices    # noqa: E402

MASS_FACTOR = 1.2187                                  # V_p / dx^3 for h/dx = 3
BAFFLE_AZIMUTHS_DEG = (61.45, 181.45, 301.45)
TANK_INNER_RADIUS_CUT = 0.1395                        # same split as _check_blade_leak_tracking.py
LIQUID_TOP = 0.4265


def static_plates(case):
    for plate in case.thin_plates or []:
        if plate.frame != "static":
            continue
        normal = np.asarray(plate.normal, dtype=np.float64)
        axis_a = np.asarray(plate.axis_a, dtype=np.float64)
        yield plate, np.asarray(plate.centre, dtype=np.float64), axis_a, np.cross(normal, axis_a), normal


def geometry(case_directory):
    case = load_case(str(case_directory / "case.yaml"))
    spacing = 2.0 * float(case.physics.particle_radius)
    fluid = load_obj_vertices(case_directory / "fluid.obj").astype(np.float64)
    print(f"{case_directory.name}: dx = {spacing * 1e3:.1f} mm, {fluid.shape[0]:,} fluid particles")
    print("  plate      side  nearest mm   first layer mean mm")
    for plate, centre, axis_a, axis_b, normal in static_plates(case):
        local = fluid - centre
        along_a, along_b, across = local @ axis_a, local @ axis_b, local @ normal
        half_a, half_b = plate.extent
        inside = ((np.abs(along_a) < half_a - spacing) & (np.abs(along_b) < min(half_b, 0.20) - 2 * spacing)
                  & (np.abs(across) < 3.0 * spacing))
        for side, sign in (("+n", 1.0), ("-n", -1.0)):
            distance = sign * across[inside]
            distance = distance[distance > 0]
            first_layer = distance[distance < distance.min() + 0.5 * spacing]
            print(f"  {plate.name:9s}  {side}    {distance.min() * 1e3:6.2f}       {first_layer.mean() * 1e3:6.2f}")


def run(case_directory, steps):
    from utils.sph.vulkan_context import VulkanContext
    from experiment.v1 import compile_shaders_v1
    from experiment.v1.utils.simulator_v1 import SphSimulatorV1

    compile_shaders_v1.compile_v1_shaders()
    case = load_case(str(case_directory / "case.yaml"))
    for material in case.materials:
        material.rotor_angular_velocity = 0.0
    plate_names = [plate.name for plate in (case.thin_plates or [])]
    plate_normals = {plate.name: np.asarray(plate.normal) for plate in (case.thin_plates or [])}
    samples = {}
    with VulkanContext.create(application_name="tank_at_rest", enable_validation=False) as context:
        simulator = SphSimulatorV1(context, case)
        try:
            simulator.bootstrap()
            while simulator.step_count < steps:
                simulator.step()
                if simulator.step_count % 100 or simulator.step_count < 200:
                    continue
                points, forces, plate = simulator.readback_boundary_forces(with_plate_index=True)
                torque = points[:, 2] * forces[:, 0] - points[:, 0] * forces[:, 2]
                radius, height = np.hypot(points[:, 0], points[:, 2]), points[:, 1]
                ordinary = plate < 0
                row = {"cylinder wall": torque[ordinary & (radius > TANK_INNER_RADIUS_CUT) & (height > 0)
                                               & (height < LIQUID_TOP)].sum(),
                       "lid": torque[ordinary & (height >= LIQUID_TOP)].sum(),
                       "floor": torque[ordinary & (height <= 0)].sum(),
                       "all boundary": torque.sum()}
                near_baffle_band = (ordinary & (radius > 0.10) & (radius < TANK_INNER_RADIUS_CUT) & (height > 0)
                                    & (height < LIQUID_TOP))
                azimuth = np.degrees(np.arctan2(points[:, 2], points[:, 0])) % 360.0
                for index, centre in enumerate(BAFFLE_AZIMUTHS_DEG):
                    near = near_baffle_band & (np.abs((azimuth - centre + 180.0) % 360.0 - 180.0) < 20.0)
                    row[f"wall particles at baffle {index + 1}"] = torque[near].sum()
                for index, name in enumerate(plate_names):
                    if name.startswith("baffle"):
                        mask = plate == index
                        row[f"{name} plate torque"] = torque[mask].sum()
                        row[f"{name} plate normal force"] = forces[mask].sum(axis=0) @ plate_normals[name]
                positions = simulator.readback_positions()
                fluid = simulator.live_slot_mask(positions) & (simulator.readback_material() == 0)
                speed = np.linalg.norm(simulator.readback_velocity_mass()[fluid, :3], axis=1)
                row["fluid mean speed"] = speed.mean()
                row["fluid max speed"] = speed.max()
                row["rotor torque"] = simulator.readback_rotor_torque()["torque_axis"]
                samples[simulator.step_count] = row
            positions = simulator.readback_positions()
            live = simulator.live_slot_mask(positions)
            np.savez_compressed(case_directory / "rest_dump.npz", position=positions[live, :3],
                                material=simulator.readback_material()[live],
                                velocity=simulator.readback_velocity_mass()[live, :3],
                                density=simulator.readback_density_pressure()[live, 0],
                                time=simulator.simulation_time)
        finally:
            simulator.destroy()
    timestep = float(case.timestep)
    checkpoints = [s for s in sorted(samples) if s in (200, 500, 1000) or s % 2000 == 0 or s == max(samples)]
    print(f"{case_directory.name}: rotor stopped, {steps} steps (t = {steps * timestep:.3f} s); "
          f"torques mN m and forces N divided by the mass factor {MASS_FACTOR}")
    print("  quantity                              " + "".join(f"{f't={s * timestep:.3f}s':>11s}" for s in checkpoints))
    for key in samples[checkpoints[0]]:
        if "speed" in key:
            scale, unit = 1e3, "mm/s"
        elif "force" in key:
            scale, unit = 1.0 / MASS_FACTOR, "N"
        else:
            scale, unit = 1e3 / MASS_FACTOR, "mN m"
        print(f"  {key + ' [' + unit + ']':37s} " + "".join(f"{samples[s][key] * scale:11.2f}" for s in checkpoints))


if __name__ == "__main__":
    command, directory = sys.argv[1], pathlib.Path(sys.argv[2])
    if command == "geometry":
        geometry(directory)
    else:
        run(directory, int(sys.argv[3]))
