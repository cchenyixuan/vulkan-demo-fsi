"""
_analyze_mstar_stats.py — global statistics of the M-Star LBM-LES runs of the 30 L tank (DARUS-5523),
grouped by lattice resolution (2026-10-03).

M-Star wrote no velocity fields for these runs (Slices.txt and Volume.txt are empty, the checkpoints hold
no field), only global time series. Per run, averaged over --window (default 25..75 s):
  out/Stats/Fluid.txt                         kinetic energy and mean speed of all fluid, fluid volume
  out/Stats/MovingBody_Moving Body_1.txt      torque about y on the whole rotor (Rushton + PBT + shaft)
  out/Stats/Probe_Probe_a_origin_k.txt        4 point probes on x = 0, z = 116.4 mm, y 260..410 mm:
                                              mean velocity (radial, tangential, axial)
Lattice spacing from out/Output/MeshInfo.txt.

usage: python experiment/v1/checks/_analyze_mstar_stats.py [REPOSITORY] [--window 25 75]
"""
import argparse
import pathlib
import re

import numpy as np

DEFAULT_REPOSITORY = pathlib.Path(r"D:/CFD/SPH dev/repository/02_simulation_results")


def table(path):
    return np.genfromtxt(path, delimiter="\t", names=True, invalid_raise=False)


def window_mean(data, column, window):
    time = data["Time_s"]
    selected = (time >= window[0]) & (time <= window[1])
    return data[column][selected].mean() if selected.any() else np.nan


def run_summary(run, window):
    stats = run / "out" / "Stats"
    fluid = table(stats / "Fluid.txt")
    body_path = stats / "MovingBody_Moving Body_1.txt"
    body = table(body_path) if body_path.exists() else None
    mesh_path = run / "out" / "Output" / "MeshInfo.txt"
    mesh = mesh_path.read_text().split() if mesh_path.exists() else []
    spacing = float(mesh[-1]) if mesh else np.nan
    probes = []
    for k in range(4):
        path = stats / f"Probe_Probe_a_origin_{k}.txt"
        if not path.exists():
            continue
        probe = table(path)
        x, y, z = (probe[c][0] for c in ("Position_X_m", "Position_Y_m", "Position_Z_m"))
        u = np.array([window_mean(probe, c, window) for c in ("Velocity_X_ms", "Velocity_Y_ms", "Velocity_Z_ms")])
        radius = np.hypot(x, z)
        # +y right-hand rotation (the rotor's sense in our frame): u_theta = (u_x z - u_z x) / r
        probes.append((y, (u[0] * x + u[2] * z) / radius, (u[0] * z - u[2] * x) / radius, u[1]))
    return {"spacing": spacing, "end": fluid["Time_s"][-1],
            "ke": window_mean(fluid, "Kinetic_Energy_J", window), "speed": window_mean(fluid, "Mean_Velocity_ms", window),
            "volume": window_mean(fluid, "Fluid_Volume_m3", window), "torque": window_mean(body, "Torque_Y_Nm", window) if body is not None else np.nan,
            "probes": probes}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("repository", nargs="?", default=str(DEFAULT_REPOSITORY))
    parser.add_argument("--window", type=float, nargs=2, default=(25.0, 75.0))
    arguments = parser.parse_args()
    runs = sorted(p.parents[2] for p in pathlib.Path(arguments.repository).rglob("Stats/Fluid.txt"))
    groups = {}
    for run in runs:
        summary = run_summary(run, arguments.window)
        group = re.search(r"(LX\d+[^/\\]*?)(_raw_results|_mstar)", str(run))
        label = (group.group(1) if group else run.parent.name) + ("_dp" if "_dp" in str(run) else "")
        groups.setdefault(label, []).append((run, summary))
    print(f"window {arguments.window[0]:g}..{arguments.window[1]:g} s; torque about +y on the whole rotor, mN m")
    print("  group                    runs  dx mm   end s   fluid L   KE J            mean |u| m/s    torque mN m")
    for label, items in groups.items():
        values = {key: np.array([s[key] for _, s in items]) for key in ("spacing", "end", "volume", "ke", "speed", "torque")}
        print(f"  {label:24s} {len(items):3d}   {np.nanmean(values['spacing']) * 1e3:5.2f}  {np.nanmin(values['end']):6.1f}   "
              f"{np.nanmean(values['volume']) * 1e3:6.2f}   {np.nanmean(values['ke']):.3f} +- {np.nanstd(values['ke']):.3f}  "
              f"{np.nanmean(values['speed']):.4f}          {np.nanmean(values['torque']) * 1e3:+7.1f} +- {np.nanstd(values['torque']) * 1e3:.1f}")
    print("\n  point probes (x = 0, z = 116.4 mm), mean over the runs of each group: y mm  u_r  u_theta  u_y  m/s")
    for label, items in groups.items():
        probes = [s["probes"] for _, s in items if s["probes"]]
        if not probes:
            continue
        mean = np.nanmean(np.array(probes), axis=0)
        print(f"  {label:24s} " + "   ".join(f"{p[0] * 1e3:.0f}: {p[1]:+.3f} {p[2]:+.3f} {p[3]:+.3f}" for p in mean))


if __name__ == "__main__":
    main()
