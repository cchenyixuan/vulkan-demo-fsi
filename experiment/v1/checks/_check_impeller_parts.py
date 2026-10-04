"""
_check_impeller_parts.py — where the torque sits on our impellers (2026-10-03), to compare with
_analyze_fluent_impeller_parts.py.

Runs a 30 L tank case (optionally from an initial velocity field, _map_fluent_velocity.py) and every --every
steps splits the resistance torque of the rotor:
  thin-plate records (readback_thin_plate_reactions: fluid particle position, force on the plate, plate index),
  with the plate frame rotated to the current rotor angle, R(theta) = [[cos, 0, sin], [0, 1, 0], [-sin, 0, cos]]:
    Rushton blades: front (the fluid particle lies ahead of the blade in the direction of motion e_theta at the
    blade centre) or back, above the disk (y > 39.7 mm), below it (y < 37.2 mm) or level with it;
    Rushton disk: top / bottom (fluid particle above or below the disk mid-plane); PBT blades: front / back;
  lattice rotor particles (force m (a - g)): Rushton hub (r < 12.5 mm, 19 < y < 43.5 mm), bell (y < 19 mm),
    shaft and PBT hub (the rest).
Torque about +y, divided by the mass factor; reported as resistance (positive = against the rotation, which
turns along +y). Averages over the windows given with --windows.

Options of 2026-10-04 (PBT study, log/2026-10-04_fluctuation-scales-and-pbt-bands.md), off by default:
  --pbt-bands E0 E1 ...  the PBT blade records also split by the radius of the fluid particle into the bands
                         [E0, E1), [E1, E2), ... m, front and back ("PBT blade front, r 24.0..32.0 mm"), and
                         the lattice part "shaft and PBT hub" split into "PBT hub and collar (lattice)"
                         (r < 12 mm, 170 < y < 215 mm) and "shaft (lattice)";
  --out PREFIX           window means of every part as PREFIX_parts.json;
  --dump-times T ...     dumps PREFIX_tT.npz in the format of _check_tank_energy.py (needs --out).

usage (repo root, solver environment):
    python experiment/v1/checks/_check_impeller_parts.py CASE_DIR --time 4 --every 200 \
        [--initial-velocity V.npy] --windows 0.3 0.7 0.7 1.5 3 4 \
        [--pbt-bands 0.0122 0.016 0.024 0.032 0.040 0.050 --out PREFIX --dump-times 0.4 0.5]
"""
import argparse
import json
import pathlib
import sys

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DISK_Y0, DISK_Y1, DISK_MID = 0.0372, 0.0397, 0.03845


def rotation(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def split(simulator, case, rotor_groups, pbt_bands=None):
    theta = float(simulator.rotor_angle)
    turn = rotation(theta)
    pivot = np.asarray(case.rotor.pivot, dtype=np.float64)
    parts = {}

    def add(name, value):
        parts[name] = parts.get(name, 0.0) + value

    reactions = simulator.readback_thin_plate_reactions()
    position, force, plate = reactions["position"] - pivot, reactions["force"], reactions["plate"]
    torque = position[:, 2] * force[:, 0] - position[:, 0] * force[:, 2]
    for index, entry in enumerate(case.thin_plates or []):
        if entry.frame != "rotor":
            continue
        mask = plate == index
        if not mask.any():
            continue
        centre = turn @ (np.asarray(entry.centre, dtype=np.float64) - pivot)
        radius = np.hypot(centre[0], centre[2])
        motion = np.array([centre[2], 0.0, -centre[0]]) / max(radius, 1e-12)
        ahead = (position[mask] - centre) @ motion > 0
        y = position[mask, 1] + pivot[1]
        t = torque[mask]
        if entry.name.startswith("rushton_blade"):
            for side, select in (("front", ahead), ("back", ~ahead)):
                add(f"Rushton blade {side}, above disk", t[select & (y > DISK_Y1)].sum())
                add(f"Rushton blade {side}, below disk", t[select & (y < DISK_Y0)].sum())
                add(f"Rushton blade {side}, disk level", t[select & (y >= DISK_Y0) & (y <= DISK_Y1)].sum())
        elif entry.name == "rushton_disk":
            add("Rushton disk top", t[y > DISK_MID].sum())
            add("Rushton disk bottom", t[y <= DISK_MID].sum())
        elif entry.name.startswith("pbt_blade"):
            # side of the pitched plate by its normal, turned to face the motion (2026-10-04; "ahead of the
            # centre" put the back face near the leading edge into the front)
            normal = turn @ np.asarray(entry.normal, dtype=np.float64)
            normal = normal if normal @ motion > 0 else -normal
            ahead = (position[mask] - centre) @ normal > 0
            add("PBT blade front", t[ahead].sum())
            add("PBT blade back", t[~ahead].sum())
            if pbt_bands is not None:
                r = np.hypot(position[mask, 0], position[mask, 2])
                for r0, r1 in zip(pbt_bands[:-1], pbt_bands[1:]):
                    band = (r >= r0) & (r < r1)
                    for side, select in (("front", ahead), ("back", ~ahead)):
                        add(f"PBT blade {side}, r {r0 * 1e3:4.1f}..{r1 * 1e3:4.1f} mm", t[select & band].sum())
        else:
            add(f"plate {entry.name}", t.sum())
    positions = simulator.readback_positions()
    material = simulator.readback_material()
    rotor = simulator.live_slot_mask(positions) & np.isin(material, rotor_groups)
    x = positions[rotor, :3].astype(np.float64) - pivot
    mass = simulator.readback_velocity_mass()[rotor, 3].astype(np.float64)
    acceleration = simulator.readback_acceleration()[rotor, :3].astype(np.float64)
    lattice_force = (acceleration - np.asarray(case.physics.gravity, dtype=np.float64)) * mass[:, None]
    lattice_torque = x[:, 2] * lattice_force[:, 0] - x[:, 0] * lattice_force[:, 2]
    r, y = np.hypot(x[:, 0], x[:, 2]), x[:, 1] + pivot[1]
    hub = (r < 0.0125) & (y > 0.019) & (y < 0.0435)
    bell = y <= 0.019
    add("Rushton hub (lattice)", lattice_torque[hub].sum())
    add("bell (lattice)", lattice_torque[bell].sum())
    if pbt_bands is None:
        add("shaft and PBT hub (lattice)", lattice_torque[~hub & ~bell].sum())
    else:
        pbt_hub = (r < 0.012) & (y > 0.170) & (y < 0.215)
        add("PBT hub and collar (lattice)", lattice_torque[pbt_hub].sum())
        add("shaft (lattice)", lattice_torque[~hub & ~bell & ~pbt_hub].sum())
    return parts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("case")
    parser.add_argument("--time", type=float, required=True)
    parser.add_argument("--every", type=int, default=200)
    parser.add_argument("--initial-velocity", default=None)
    parser.add_argument("--windows", type=float, nargs="+", default=[0.3, 0.7])
    parser.add_argument("--pbt-bands", type=float, nargs="+", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--dump-times", type=float, nargs="*", default=[])
    arguments = parser.parse_args()
    if arguments.dump_times and not arguments.out:
        parser.error("--dump-times needs --out")

    from utils.sph.case import load_case
    from utils.sph.vulkan_context import VulkanContext
    from experiment.v1 import compile_shaders_v1
    from experiment.v1.utils.simulator_v1 import SphSimulatorV1

    compile_shaders_v1.compile_v1_shaders()
    case = load_case(str(pathlib.Path(arguments.case) / "case.yaml"))
    steps = int(np.ceil(arguments.time / float(case.timestep)))
    windows = list(zip(arguments.windows[0::2], arguments.windows[1::2]))
    rotor_groups = np.asarray([m.group_id for m in case.materials if m.kind == 3], dtype=np.uint32)
    samples = []
    dump_times = sorted(arguments.dump_times)
    out = pathlib.Path(arguments.out) if arguments.out else None
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
    with VulkanContext.create(application_name="impeller_parts", enable_validation=False) as context:
        simulator = SphSimulatorV1(context, case)
        try:
            if arguments.initial_velocity:
                simulator.write_initial_velocities(np.load(arguments.initial_velocity), first_slot=1)
            simulator.bootstrap()
            while simulator.step_count < steps:
                simulator.step()
                if dump_times and simulator.simulation_time >= dump_times[0]:
                    np.savez(out.parent / f"{out.name}_t{dump_times.pop(0):.3f}.npz",
                             positions=simulator.readback_positions(), material=simulator.readback_material(),
                             velocity_mass=simulator.readback_velocity_mass(),
                             particle_uid=simulator.readback_particle_uid(),
                             status=json.dumps(simulator.readback_global_status()),
                             density_pressure=simulator.readback_density_pressure())
                if simulator.step_count % arguments.every or simulator.simulation_time < windows[0][0]:
                    continue
                simulator.begin_readback_cache()
                total = simulator.readback_rotor_torque(0.1, 0.007)
                parts = split(simulator, case, rotor_groups, arguments.pbt_bands)
                factor = total["mass_factor"]
                parts = {name: -value / factor for name, value in parts.items()}
                parts["TOTAL (solver, lower + upper + shaft)"] = -total["torque_axis"] / factor
                parts["  of which below y = 0.1 m (Rushton side)"] = -total["torque_axis_lower"] / factor
                simulator.end_readback_cache()
                samples.append((simulator.simulation_time, parts))
        finally:
            simulator.destroy()
    names = list(samples[0][1])
    if out is not None:
        means = {f"{a:g}..{b:g}": {name: float(np.mean([s[name] for t, s in samples if a <= t < b and name in s]))
                                   for name in names} for a, b in windows}
        with open(out.parent / f"{out.name}_parts.json", "w") as handle:
            json.dump({"case": str(arguments.case), "samples": len(samples), "windows": means}, handle, indent=1)
    print(f"{arguments.case}: resistance torque, mN m (positive = against the rotation)")
    print("  part                                         " + "".join(f"{f'{a:g}..{b:g} s':>12s}" for a, b in windows))
    for name in names:
        cells = []
        for a, b in windows:
            values = [s[name] for t, s in samples if a <= t < b and name in s]
            cells.append(f"{np.mean(values) * 1e3:12.2f}" if values else f"{'':>12s}")
        print(f"  {name:44s} " + "".join(cells))


if __name__ == "__main__":
    main()
