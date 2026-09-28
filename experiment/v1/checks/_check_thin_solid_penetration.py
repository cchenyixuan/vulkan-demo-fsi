"""_check_thin_solid_penetration.py — do fluid particles pass through the thin solids
(impeller blades, Rushton disk, baffles) of the 30 L tank? (2026-09-28)

Written for the test of single-layer blades (generator --thin-layers 1 --skin 0).

  run      <solver env>  _check_thin_solid_penetration.py run CASE.yaml --steps N --every K
               [--pair-gap 50] --out-dir DIR
           Runs the case; every K steps it saves a PAIR of snapshots, at step s and
           s + pair-gap (fluid positions + uid, rotor angle), and the torque every
           1000 steps (DIR/torque.csv, per-impeller split).

  analyze  <python with scipy>  _check_thin_solid_penetration.py analyze DIR [--dx 0.003]
           For every pair, in the frame rotating with the rotor:
             * crossings: fluid particles (matched by uid) whose coordinate normal to a
               thin solid's mid-plane changes sign between the two snapshots while they
               are, at both times, within the solid's in-plane extent shrunk by one
               spacing and closer than 3 dx to the mid-plane. pair-gap is chosen so that
               a particle moves about one spacing in between, so it cannot have gone
               around the edge: a sign change is a passage through the sheet.
               On a single-layer sheet this over-counts: the solid particles sit on a
               staircase within +-dx/2 of the mid-plane, and fluid particles resting in
               its notches jitter across the mid-plane without passing through.
             * inside: fluid particles closer than dx/4 to a mid-plane (same in-plane
               restriction), i.e. sitting inside the sheet.
             * through-flow: the mean velocity normal to the sheet, relative to the
               solid, of the fluid within 3.5 dx of the mid-plane (in-plane extent
               shrunk by 1.5 dx), from the displacement between the two snapshots.
               An impermeable sheet gives zero; it is reported in m/s and as a fraction
               of the sheet's own normal speed (blades), separately for the two sides.
             * the smallest fluid-solid centre distance and the number of fluid
               particles closer than 0.5 dx / 0.7 dx to a solid particle.
             * fluid particles outside the tank.
           The rotor particles rotated back by the rotor angle must coincide with their
           initial positions; the largest deviation is printed as a check of the frame.
"""
import argparse
import json
import math
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
for entry in (ROOT, ROOT / "utils" / "geometry"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

POWER_SCALE = 998.0 * (200.0 / 60.0) ** 3 * 0.096 ** 5
OMEGA = 2.0 * math.pi * 200.0 / 60.0


def run(arguments):
    from utils.sph.case import load_case
    from utils.sph.vulkan_context import VulkanContext
    from experiment.v1 import compile_shaders_v1
    from experiment.v1.utils.simulator_v1 import SphSimulatorV1

    compile_shaders_v1.compile_v1_shaders()
    case = load_case(arguments.case)
    out = pathlib.Path(arguments.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rotor_groups = None

    def snapshot(simulator, tag):
        positions = simulator.readback_positions()
        live = simulator.live_slot_mask(positions)
        material = simulator.readback_material()
        np.savez(out / f"snapshot_{simulator.step_count:08d}_{tag}.npz",
                 positions=positions[live, :3], material=material[live],
                 uid=simulator.readback_particle_uid()[live],
                 angle=simulator.rotor_angle, time=simulator.simulation_time, step=simulator.step_count,
                 status=json.dumps(simulator.readback_global_status()))

    with VulkanContext.create(application_name="thin_solid_penetration", enable_validation=False) as context:
        simulator = SphSimulatorV1(context, case)
        try:
            simulator.bootstrap()
            snapshot(simulator, "initial")
            with open(out / "torque.csv", "w") as log:
                log.write("step,time,torque_axis,torque_lower,torque_upper,torque_shaft\n")
                while simulator.step_count < arguments.steps:
                    simulator.step()
                    step = simulator.step_count
                    if step % 1000 == 0:
                        torque = simulator.readback_rotor_torque(0.1, 0.007)
                        log.write(f"{step},{simulator.simulation_time:.6e},{torque['torque_axis']:.6e},"
                                  f"{torque['torque_axis_lower']:.6e},{torque['torque_axis_upper']:.6e},"
                                  f"{torque['torque_axis_shaft']:.6e}\n")
                        log.flush()
                    if step % arguments.every == 0:
                        snapshot(simulator, "a")
                    elif step % arguments.every == arguments.pair_gap and step > arguments.pair_gap:
                        snapshot(simulator, "b")
                        status = simulator.readback_global_status()
                        print(f"[penetration] step {step} t = {simulator.simulation_time:.3f} s alive "
                              f"{status['alive_particle_count']:,} overflow {status['overflow_inside_count']}/"
                              f"{status['overflow_incoming_count']} fallback {status['correction_fallback_count']}", flush=True)
        finally:
            simulator.destroy()


def rotate_back(points, angle):
    """Inverse of predict.comp's rotation about +y by `angle`."""
    cos, sin = math.cos(-angle), math.sin(-angle)
    x, z = points[:, 0], points[:, 2]
    return np.column_stack([x * cos + z * sin, points[:, 1], z * cos - x * sin])


def sheets(dx):
    """Mid-planes of the thin solids in the reference configuration:
    (name, moving, centre, in-plane axis a, in-plane axis b, normal, half extent a, half extent b)
    with the in-plane extent already shrunk by one spacing."""
    import _demo_stirred_tank_30l as gen
    result = []
    for k in range(6):
        blade = gen.RUSHTON_BLADE
        e_r, e_t, e_y = gen.radial_frame(blade["azimuth0_deg"] + 60 * k)
        r0, r1 = blade["radial"]
        result.append((f"Rushton blade {k + 1}", True, e_r * 0.5 * (r0 + r1) + e_y * 0.5 * (blade["y0"] + blade["y1"]),
                       e_r, e_y, e_t, 0.5 * (r1 - r0) - dx, 0.5 * (blade["y1"] - blade["y0"]) - dx))
        blade = gen.PBT_BLADE
        e_r, e_t, e_y = gen.radial_frame(blade["azimuth0_deg"] + 60 * k)
        pitch = math.radians(blade["pitch_deg"])
        e_chord = math.cos(pitch) * e_t - math.sin(pitch) * e_y
        e_norm = math.sin(pitch) * e_t + math.cos(pitch) * e_y
        r0, r1 = blade["radial"]
        result.append((f"PBT blade {k + 1}", True, e_r * 0.5 * (r0 + r1) + e_y * blade["center_y"],
                       e_r, e_chord, e_norm, 0.5 * (r1 - r0) - dx, 0.5 * blade["chord"] - dx))
    for index, azimuth in enumerate(gen.BAFFLE_AZIMUTHS_DEG):
        e_r, e_t, e_y = gen.radial_frame(azimuth)
        r0, r1 = gen.BAFFLE_RADIAL
        y0, y1 = gen.BAFFLE_Y0, gen.LIQUID_HEIGHT
        result.append((f"baffle {index + 1}", False, e_r * 0.5 * (r0 + r1) + e_y * 0.5 * (y0 + y1),
                       e_r, e_y, e_t, 0.5 * (r1 - r0) - dx, 0.5 * (y1 - y0) - dx))
    disk = gen.RUSHTON_DISK
    return result, (0.5 * (disk["y0"] + disk["y1"]), gen.RUSHTON_HUB["radius"] + dx, disk["radius"] - dx), gen


def analyze(arguments):
    from scipy.spatial import cKDTree
    directory = pathlib.Path(arguments.directory)
    dx = arguments.dx
    sheet_list, (disk_y, disk_r0, disk_r1), gen = sheets(dx)
    interior = gen.DishedTankInterior(gen.TANK_RADIUS, gen.LIQUID_HEIGHT, gen.FLOOR_PROFILE)
    initial = np.load(directory / "snapshot_00000000_initial.npz")
    rotor_reference = initial["positions"][initial["material"] == 2].astype(np.float64)
    reference_tree = cKDTree(rotor_reference)
    print(f"{directory}: rotor particles {len(rotor_reference):,}, wall {int((initial['material'] == 1).sum()):,}, "
          f"fluid {int((initial['material'] == 0).sum()):,}")
    totals = {"crossings": 0, "inside": 0}
    per_sheet = {}
    through = {}
    print(f"{'step':>8s} {'t [s]':>7s} {'alive':>10s} {'crossed':>8s} {'inside':>7s} {'min d/dx':>9s} {'<0.5dx':>7s} "
          f"{'<0.7dx':>7s} {'outside':>8s} {'frame err [m]':>13s}")
    for first_path in sorted(directory.glob("snapshot_*_a.npz")):
        step = int(first_path.name.split("_")[1])
        second_candidates = sorted(directory.glob(f"snapshot_{step + arguments.pair_gap:08d}_b.npz"))
        if not second_candidates:
            continue
        first, second = np.load(first_path), np.load(second_candidates[0])
        frames = []
        for archive in (first, second):
            fluid = archive["material"] == 0
            order = np.argsort(archive["uid"][fluid])
            frames.append((archive["uid"][fluid][order], archive["positions"][fluid].astype(np.float64)[order],
                           float(archive["angle"])))
        assert np.array_equal(frames[0][0], frames[1][0]), "fluid uid sets differ between the two snapshots"
        rotor_now = second["positions"][second["material"] == 2].astype(np.float64)
        frame_error = reference_tree.query(rotate_back(rotor_now, float(second["angle"])))[0].max()

        crossed_total, inside_total = 0, 0
        for name, moving, centre, axis_a, axis_b, normal, half_a, half_b in sheet_list:
            coordinates = []
            for _, positions, angle in frames:
                local = (rotate_back(positions, angle) if moving else positions) - centre
                coordinates.append((local @ axis_a, local @ axis_b, local @ normal))
            within = [(np.abs(a) < half_a) & (np.abs(b) < half_b) & (np.abs(n) < 3 * dx) for a, b, n in coordinates]
            crossed = within[0] & within[1] & (coordinates[0][2] * coordinates[1][2] < 0)
            inside = within[1] & (np.abs(coordinates[1][2]) < 0.25 * dx)
            crossed_total += int(crossed.sum()); inside_total += int(inside.sum())
            entry = per_sheet.setdefault(name.rsplit(" ", 1)[0], [0, 0])
            entry[0] += int(crossed.sum()); entry[1] += int(inside.sum())
            # through-flow velocity relative to the sheet
            interval = float(second["time"]) - float(first["time"])
            near = [(np.abs(a) < half_a - 0.5 * dx) & (np.abs(b) < half_b - 0.5 * dx) & (np.abs(n) < 3.5 * dx)
                    for a, b, n in coordinates]
            both = near[0] & near[1]
            relative = (coordinates[1][2][both] - coordinates[0][2][both]) / interval
            side = coordinates[0][2][both] > 0
            radius_mid = np.hypot(centre[0], centre[2])
            speed = 0.0 if not moving else abs(OMEGA * radius_mid * float(np.dot(normal, np.cross([0.0, 1.0, 0.0], centre / max(radius_mid, 1e-9)))))
            flow = through.setdefault(name.rsplit(" ", 1)[0], {"sum": 0.0, "count": 0, "plus": 0.0, "n_plus": 0,
                                                               "minus": 0.0, "n_minus": 0, "speed": speed})
            sign = 1.0
            if moving:
                # orient the normal along the sheet's own motion, so that a positive
                # through-flow means fluid overtaking the sheet from behind
                sign = 1.0 if float(np.dot(normal, np.cross([0.0, 1.0, 0.0], centre))) > 0 else -1.0
            flow["sum"] += sign * relative.sum(); flow["count"] += relative.size
            ahead = side if sign > 0 else ~side
            flow["plus"] += sign * relative[ahead].sum(); flow["n_plus"] += int(ahead.sum())
            flow["minus"] += sign * relative[~ahead].sum(); flow["n_minus"] += int((~ahead).sum())
        heights, radii = [], []
        for _, positions, _ in frames:
            heights.append(positions[:, 1] - disk_y)
            radii.append(np.hypot(positions[:, 0], positions[:, 2]))
        within = [(r > disk_r0) & (r < disk_r1) & (np.abs(y) < 3 * dx) for y, r in zip(heights, radii)]
        crossed = within[0] & within[1] & (heights[0] * heights[1] < 0)
        inside = within[1] & (np.abs(heights[1]) < 0.25 * dx)
        crossed_total += int(crossed.sum()); inside_total += int(inside.sum())
        entry = per_sheet.setdefault("Rushton disk", [0, 0])
        entry[0] += int(crossed.sum()); entry[1] += int(inside.sum())

        solid = second["positions"][second["material"] != 0].astype(np.float64)
        distance = cKDTree(solid).query(frames[1][1])[0]
        outside = int((interior.signed_distance(frames[1][1]) > 0.5 * dx).sum())
        status = json.loads(str(second["status"]))
        totals["crossings"] += crossed_total; totals["inside"] += inside_total
        print(f"{int(second['step']):8d} {float(second['time']):7.3f} {status['alive_particle_count']:10,d} {crossed_total:8d} "
              f"{inside_total:7d} {distance.min() / dx:9.3f} {int((distance < 0.5 * dx).sum()):7d} "
              f"{int((distance < 0.7 * dx).sum()):7d} {outside:8d} {frame_error:13.2e}")
    print("per thin solid, summed over all pairs (crossed / inside): "
          + ", ".join(f"{name} {values[0]} / {values[1]}" for name, values in per_sheet.items()))
    print("through-flow normal to the sheet, relative to the solid, all pairs (positive = along the sheet's motion):")
    for name, flow in through.items():
        if flow["count"] == 0:
            continue
        mean = flow["sum"] / flow["count"]
        ahead = flow["plus"] / max(flow["n_plus"], 1)
        behind = flow["minus"] / max(flow["n_minus"], 1)
        text = (f"   {name}: mean {mean:+.4f} m/s (side ahead {ahead:+.4f}, side behind {behind:+.4f}; "
                f"{flow['count']} particle samples)")
        if flow["speed"] > 0.01:
            text += f"; sheet's normal speed {flow['speed']:.3f} m/s, through-flow / speed = {mean / flow['speed']:+.3f}"
        print(text)
    torque_path = directory / "torque.csv"
    if torque_path.exists():
        data = np.loadtxt(torque_path, delimiter=",", skiprows=1)
        for start, end in arguments.windows:
            window = (data[:, 1] >= start) & (data[:, 1] < end)
            if window.sum() > 2:
                values = np.abs(data[window, 2:5]) * OMEGA / POWER_SCALE
                print(f"Np over {start:g}-{end:g} s: total {values[:, 0].mean():.2f} +- {values[:, 0].std(ddof=1):.2f}, "
                      f"Rushton {values[:, 1].mean():.2f}, PBT {values[:, 2].mean():.2f} ({window.sum()} samples)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    subparsers = parser.add_subparsers(dest="phase", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("case")
    run_parser.add_argument("--steps", type=int, required=True)
    run_parser.add_argument("--every", type=int, default=5000)
    run_parser.add_argument("--pair-gap", type=int, default=50)
    run_parser.add_argument("--out-dir", required=True)
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("directory")
    analyze_parser.add_argument("--dx", type=float, default=0.003)
    analyze_parser.add_argument("--pair-gap", type=int, default=50)
    analyze_parser.add_argument("--windows", type=float, nargs=2, action="append", default=None,
                                metavar=("START", "END"), help="time windows for the power number")
    arguments = parser.parse_args()
    if arguments.phase == "run":
        run(arguments)
    else:
        arguments.windows = arguments.windows or [(1.0, 3.0), (3.0, 6.0), (6.0, 10.0)]
        analyze(arguments)
    return 0


if __name__ == "__main__":
    sys.exit(main())
