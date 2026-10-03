"""
_check_tank_energy.py — energy and torque log of a 30 L tank run (2026-10-03).

Written to find why the bulk of our 30 L tank holds 1.2 to 1.6 times the kinetic energy of the Fluent
LES (log/2026-10-03_fluent-and-wall-torque.md). Runs a case, stirred or at rest (--rest: rotor speed 0),
and every --every steps writes one CSV row:

  - rotor torque about +y: total, lower (Rushton side, y < 0.1 m) and upper impeller;
  - torque of the fluid on the boundary parts: cylinder wall (ordinary particles r > 139.5 mm,
    0 < y < 426.5 mm), lid, floor, the ordinary wall particles next to each baffle (100 < r < 139.5 mm,
    within 20 degrees of the baffle: brackets, conformal sheet baffles or lattice baffles), every static
    thin plate, and the split of _check_blade_leak_tracking.py (baffle = plates + 100 < r < 139.5 mm);
    with --split-height H (2026-10-03) also the parts below H: below_split_cylinder (r > 139.5 mm, 0 < y < H),
    below_split_baffles (the baffle split, y < H) and below_split_walls (every static boundary record below
    H, floor included), for the angular momentum sink of the region below the Rushton;
  - kinetic energy with the physical mass rho0 dx^3 per zone: Rushton box and PBT box (Fluent's cell
    zones rt_rotorbox / pbt_rotorbox: r < 72 mm, y 18.7..58.5 / 165..225 mm) and the bulk (the rest);
    the bulk energy split into the radial, tangential and axial velocity components; mean speed per
    zone; mean tangential velocity of the bulk; angular momentum of the fluid;
  - fluid density mean / min / max, the fraction of fluid particles with p < 0 and the 1 % pressure
    quantile (with a low background pressure the TIC branch takes over), overflow counters.

At the end the state is written as OUT_final.npz in the format of the production final dumps
(_check_blade_leak_tracking.py --dump), so the other analysis scripts read it.

usage (repo root, solver environment):
    python experiment/v1/checks/_check_tank_energy.py CASE_DIR --steps 91500 --out output/kecause/plates_pb2000
    python experiment/v1/checks/_check_tank_energy.py CASE_DIR --steps 34300 --rest --every 500 --out ...
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# torques are divided by the mass factor V_p / dx^3 that readback_rotor_torque() reports (1.2187 for h/dx = 3,
# 1.083 for 4); 2026-10-03, was the constant 1.2187
BAFFLE_AZIMUTHS_DEG = (61.45, 181.45, 301.45)
INNER_RADIUS_CUT = 0.1395
LIQUID_TOP = 0.4265
RUSHTON_BOX = (0.072, 0.0187, 0.0585)                 # radius, y0, y1
PBT_BOX = (0.072, 0.165, 0.225)


def boundary_torques(simulator, plate_names, mass_factor, split_height=None):
    points, forces, plate = simulator.readback_boundary_forces(with_plate_index=True)
    torque = (points[:, 2] * forces[:, 0] - points[:, 0] * forces[:, 2]) / mass_factor
    radius, height = np.hypot(points[:, 0], points[:, 2]), points[:, 1]
    ordinary = plate < 0
    near_baffle_band = ordinary & (radius > 0.10) & (radius < INNER_RADIUS_CUT) & (height > 0) & (height < LIQUID_TOP)
    row = {"cylinder": torque[ordinary & (radius > INNER_RADIUS_CUT) & (height > 0) & (height < LIQUID_TOP)].sum(),
           "lid": torque[ordinary & (height >= LIQUID_TOP)].sum(),
           "floor": torque[ordinary & (height <= 0)].sum()}
    azimuth = np.degrees(np.arctan2(points[:, 2], points[:, 0])) % 360.0
    for index, centre in enumerate(BAFFLE_AZIMUTHS_DEG):
        near = near_baffle_band & (np.abs((azimuth - centre + 180.0) % 360.0 - 180.0) < 20.0)
        row[f"wall_at_baffle_{index + 1}"] = torque[near].sum()
    for index, name in enumerate(plate_names):
        if name.startswith("baffle"):
            row[f"plate_{name}"] = torque[plate == index].sum()
    csv_baffle = (((radius < INNER_RADIUS_CUT) & (radius > 0.10) & (height > 0.0)) | (plate >= 0))
    row["csv_walls"] = torque.sum()
    row["csv_baffles"] = torque[csv_baffle].sum()
    if split_height is not None:
        below = height < split_height
        row["below_split_cylinder"] = torque[ordinary & (radius > INNER_RADIUS_CUT) & (height > 0) & below].sum()
        row["below_split_baffles"] = torque[csv_baffle & below].sum()
        row["below_split_walls"] = torque[below].sum()
    return row


def fluid_energy(simulator, rest_density, spacing):
    positions = simulator.readback_positions()
    fluid = simulator.live_slot_mask(positions) & (simulator.readback_material() == 0)
    x = positions[fluid, :3].astype(np.float64)
    v = simulator.readback_velocity_mass()[fluid, :3].astype(np.float64)
    density_pressure = simulator.readback_density_pressure()[fluid].astype(np.float64)
    mass = rest_density * spacing ** 3
    radius = np.hypot(x[:, 0], x[:, 2])
    safe = np.maximum(radius, 1e-12)
    u_radial = (v[:, 0] * x[:, 0] + v[:, 2] * x[:, 2]) / safe
    # positive along the +y right-hand rotation (the rotor's sense): v = omega y x r = omega (z, 0, -x)
    # gives v_x z - v_z x = omega r^2 (sign corrected 2026-10-03 after the overnight runs, whose CSV
    # columns u_tangential_bulk and angular_momentum were flipped in place to match)
    u_tangential = (v[:, 0] * x[:, 2] - v[:, 2] * x[:, 0]) / safe
    speed_squared = (v ** 2).sum(axis=1)
    zones = {"rushton": (radius < RUSHTON_BOX[0]) & (x[:, 1] > RUSHTON_BOX[1]) & (x[:, 1] < RUSHTON_BOX[2]),
             "pbt": (radius < PBT_BOX[0]) & (x[:, 1] > PBT_BOX[1]) & (x[:, 1] < PBT_BOX[2])}
    zones["bulk"] = ~(zones["rushton"] | zones["pbt"])
    row = {}
    for name, mask in zones.items():
        row[f"ke_{name}"] = 0.5 * mass * speed_squared[mask].sum()
        row[f"speed_{name}"] = np.sqrt(speed_squared[mask]).mean()
    bulk = zones["bulk"]
    row["ke_bulk_radial"] = 0.5 * mass * (u_radial[bulk] ** 2).sum()
    row["ke_bulk_tangential"] = 0.5 * mass * (u_tangential[bulk] ** 2).sum()
    row["ke_bulk_axial"] = 0.5 * mass * (v[bulk, 1] ** 2).sum()
    row["u_tangential_bulk"] = u_tangential[bulk].mean()
    row["angular_momentum"] = mass * (radius * u_tangential).sum()
    row["ke_total"] = 0.5 * mass * speed_squared.sum()
    density, pressure = density_pressure[:, 0], density_pressure[:, 1]
    row["density_mean"], row["density_min"], row["density_max"] = density.mean(), density.min(), density.max()
    row["negative_pressure_fraction"] = (pressure < 0).mean()
    row["pressure_q01"] = np.percentile(pressure, 1)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("case", help="case directory (case.yaml)")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--time", type=float, default=None, help="physical time to run, s (instead of --steps)")
    parser.add_argument("--every", type=int, default=1000)
    parser.add_argument("--rest", action="store_true", help="rotor speed 0 (tank at rest)")
    parser.add_argument("--out", required=True, help="output prefix: OUT.csv, OUT_final.npz")
    parser.add_argument("--initial-velocity", default=None,
                        help="(n_fluid, 3) .npy of initial fluid velocities in fluid.obj order "
                             "(_map_fluent_velocity.py), written before bootstrap")
    parser.add_argument("--split-height", type=float, default=None,
                        help="m, also log the boundary torques below this height (default: no extra columns)")
    parser.add_argument("--dump-times", type=float, nargs="*", default=[],
                        help="also write OUT_tX.XXX.npz dumps (format of the final dump) at these times, s")
    arguments = parser.parse_args()

    from utils.sph.case import load_case
    from utils.sph.vulkan_context import VulkanContext
    from experiment.v1 import compile_shaders_v1
    from experiment.v1.utils.simulator_v1 import SphSimulatorV1

    compile_shaders_v1.compile_v1_shaders()
    case = load_case(str(pathlib.Path(arguments.case) / "case.yaml"))
    if arguments.rest:
        for material in case.materials:
            material.rotor_angular_velocity = 0.0
    spacing = 2.0 * float(case.physics.particle_radius)
    if (arguments.steps is None) == (arguments.time is None):
        raise SystemExit("give exactly one of --steps and --time")
    steps = arguments.steps if arguments.steps is not None else int(np.ceil(arguments.time / float(case.timestep)))
    rest_density = next(m.rest_density for m in case.materials if m.kind == 0)
    plate_names = [plate.name for plate in (case.thin_plates or [])]
    out = pathlib.Path(arguments.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    log, header = None, None
    started = time.time()
    dump_times = sorted(arguments.dump_times)

    def write_dump(simulator, path):
        np.savez(path, positions=simulator.readback_positions(), material=simulator.readback_material(),
                 velocity_mass=simulator.readback_velocity_mass(), particle_uid=simulator.readback_particle_uid(),
                 status=json.dumps(simulator.readback_global_status()),
                 density_pressure=simulator.readback_density_pressure())

    with VulkanContext.create(application_name="tank_energy", enable_validation=False) as context:
        simulator = SphSimulatorV1(context, case)
        try:
            if arguments.initial_velocity:
                initial = np.load(arguments.initial_velocity)
                material = simulator.readback_material()
                if not np.all(material[1:1 + initial.shape[0]] == 0):
                    raise SystemExit("--initial-velocity: slots 1..n are not all fluid (fluid.obj must come first)")
                simulator.write_initial_velocities(initial, first_slot=1)
                print(f"[tank_energy] initial velocity of {initial.shape[0]:,} fluid particles from {arguments.initial_velocity}")
            simulator.bootstrap()
            while simulator.step_count < steps:
                simulator.step()
                if dump_times and simulator.simulation_time >= dump_times[0]:
                    write_dump(simulator, out.parent / f"{out.name}_t{dump_times.pop(0):.3f}.npz")
                if simulator.step_count % arguments.every:
                    continue
                simulator.begin_readback_cache()
                rotor = simulator.readback_rotor_torque(0.1, 0.007)
                row = {"step": simulator.step_count, "time": simulator.simulation_time,
                       "rotor": rotor["torque_axis"] / rotor["mass_factor"],
                       "rotor_lower": rotor["torque_axis_lower"] / rotor["mass_factor"],
                       "rotor_upper": rotor["torque_axis_upper"] / rotor["mass_factor"]}
                row.update(boundary_torques(simulator, plate_names, rotor["mass_factor"], arguments.split_height))
                row.update(fluid_energy(simulator, rest_density, spacing))
                status = simulator.readback_global_status()
                row["overflow"] = (status["overflow_inside_count"] + status["overflow_incoming_count"]
                                   + status.get("overflow_neighbor_count", 0))
                row["wall_seconds"] = time.time() - started
                simulator.end_readback_cache()
                if log is None:
                    header = list(row)
                    log = open(out.with_suffix(".csv"), "w")
                    log.write(",".join(header) + "\n")
                log.write(",".join(f"{row[key]:.6e}" if isinstance(row[key], float) else str(row[key])
                                   for key in header) + "\n")
                log.flush()
            status = simulator.readback_global_status()
            write_dump(simulator, out.parent / (out.name + "_final.npz"))
        finally:
            if log is not None:
                log.close()
            simulator.destroy()
    print(f"[tank_energy] {steps} steps, t = {steps * float(case.timestep):.3f} s, "
          f"{(time.time() - started) / 60:.1f} min, alive {status['alive_particle_count']:,} -> {out}.csv")


if __name__ == "__main__":
    main()
