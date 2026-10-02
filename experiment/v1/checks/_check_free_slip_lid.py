"""
_check_free_slip_lid.py — check of the free-slip wall option (material flag free_slip,
USE_FREE_SLIP_WALLS, 2026-10-02) in the 1-impeller tank of Haringa (2023).

The same case runs twice, with a no-slip and with a free-slip lid (utils/geometry/_demo_rushton_tank.py
--preset jahoda, with and without --free-slip-lid). Over the last part of the run the torque about the
axis on the impeller, on the lid, on the walls and on the baffles is averaged, and the swirl of the
fluid in the top layers. A free-slip lid takes no shear: its torque must drop to the residue of the
discrete pressure forces (a central pair force on a flat lid has a small horizontal part), while the
swirl under the lid grows.

usage (repo root, solver environment):
    python experiment/v1/checks/_check_free_slip_lid.py [--dx 0.004] [--time 1.2] [--average-from 0.8]
"""
import argparse
import json
import pathlib
import subprocess
import sys

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def axial_torque(points, forces):
    """(r x F)_y about the axis through the origin."""
    return points[:, 2] * forces[:, 0] - points[:, 0] * forces[:, 2]


def run(case_directory, steps, average_from, every, mass_factor):
    from utils.sph.case import load_case
    from utils.sph.vulkan_context import VulkanContext
    from experiment.v1 import compile_shaders_v1
    from experiment.v1.utils.simulator_v1 import SphSimulatorV1

    compile_shaders_v1.compile_v1_shaders()
    case = load_case(str(case_directory / "case.yaml"))
    tank = json.loads((case_directory / "tank.json").read_text())
    top, dx = tank["liquid_height"], tank["dx"]
    rows = []
    with VulkanContext.create(application_name="free_slip_lid", enable_validation=False) as context:
        simulator = SphSimulatorV1(context, case)
        try:
            simulator.bootstrap()
            while simulator.step_count < steps:
                simulator.step()
                if simulator.step_count % every or simulator.simulation_time < average_from:
                    continue
                rotor = simulator.readback_rotor_torque()
                points, forces, plate = simulator.readback_boundary_forces(with_plate_index=True)
                torque = axial_torque(points, forces)
                lid = (plate < 0) & (points[:, 1] > top)
                wall = (plate < 0) & ~lid
                snapshot_positions = simulator.readback_positions()
                live = simulator.live_slot_mask(snapshot_positions)
                fluid = live & np.isin(simulator.readback_material(), simulator.fluid_group_ids())
                x = snapshot_positions[fluid, :3].astype(np.float64)
                v = simulator.readback_velocity_mass()[fluid, :3].astype(np.float64)
                azimuth = np.arctan2(x[:, 2], x[:, 0])
                tangential = -np.sin(azimuth) * v[:, 0] + np.cos(azimuth) * v[:, 2]
                layer = x[:, 1] > top - 2.0 * dx
                rows.append((simulator.simulation_time, rotor["torque_axis"], torque[lid].sum(), torque[wall].sum(),
                             torque[plate >= 0].sum(), float(tangential[layer].mean()),
                             float(tangential[(x[:, 1] > 0.5 * top) & (x[:, 1] < 0.6 * top)].mean())))
        finally:
            simulator.destroy()
    table = np.asarray(rows)
    result = dict(samples=len(rows), time=(float(table[0, 0]), float(table[-1, 0])))
    for column, name in enumerate(("impeller", "lid", "walls", "baffles"), start=1):
        result[name] = float(table[:, column].mean() / mass_factor)
    result["swirl_top_layers"] = float(table[:, 5].mean())
    result["swirl_mid_height"] = float(table[:, 6].mean())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--dx", type=float, default=0.004)
    parser.add_argument("--time", type=float, default=1.2, help="simulated time (s)")
    parser.add_argument("--average-from", type=float, default=0.8)
    parser.add_argument("--every", type=int, default=200)
    parser.add_argument("--mass-factor", type=float, default=1.2187)
    parser.add_argument("--out-dir", default="output/haringa/free_slip_check")
    args = parser.parse_args()
    out = _REPO_ROOT / args.out_dir
    results = {}
    for name, flags in (("no_slip", []), ("free_slip", ["--free-slip-lid"])):
        case_directory = out / f"case_{name}"
        subprocess.run([sys.executable, str(_REPO_ROOT / "utils/geometry/_demo_rushton_tank.py"), "--preset", "jahoda",
                        "--dx", str(args.dx), "--out", str(case_directory.relative_to(_REPO_ROOT))] + flags,
                       check=True, stdout=subprocess.DEVNULL)
        tank = json.loads((case_directory / "tank.json").read_text())
        steps = int(round(args.time / tank["timestep"]))
        results[name] = run(case_directory, steps, args.average_from, args.every, args.mass_factor)
        print(f"{name:9s}: torque about the axis, mN m (readback / {args.mass_factor}): "
              f"impeller {results[name]['impeller'] * 1e3:8.2f}  lid {results[name]['lid'] * 1e3:8.3f}  "
              f"walls {results[name]['walls'] * 1e3:8.2f}  baffles {results[name]['baffles'] * 1e3:8.2f};  "
              f"mean u_theta: top two layers {results[name]['swirl_top_layers']:+.4f} m/s, "
              f"mid height {results[name]['swirl_mid_height']:+.4f} m/s  ({results[name]['samples']} samples, "
              f"t = {results[name]['time'][0]:.2f} .. {results[name]['time'][1]:.2f} s)")
    ratio = abs(results["free_slip"]["lid"]) / max(abs(results["no_slip"]["lid"]), 1e-30)
    print(f"lid torque free slip / no slip = {ratio:.3f}")
    (out / "free_slip_check.json").write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
