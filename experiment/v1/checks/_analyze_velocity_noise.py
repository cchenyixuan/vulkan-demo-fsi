"""
_analyze_velocity_noise.py — particle-scale velocity noise of a dump (2026-10-03, needs scipy: base Anaconda).

For every fluid particle the kernel-smoothed velocity of its neighbourhood (Shepard mean, Wendland C4
with support h = HDX dx, self included) is subtracted:

    v'_i = v_i - sum_j W_ij v_j / sum_j W_ij

and the kinetic energy of v' (physical mass rho0 dx^3) is reported per zone (Fluent's rotor boxes and the
bulk), next to the kinetic energy of v itself. A smooth field gives v' ~ 0 (the Shepard mean is exact for
constants and, with a symmetric neighbourhood, for linear fields); velocity noise that is uncorrelated
from particle to particle stays almost entirely in v'. Structures larger than the kernel do not.

usage: python experiment/v1/checks/_analyze_velocity_noise.py DUMP.npz DX [--hdx 3] [--format final|rest]
  final: dumps of _check_blade_leak_tracking.py --dump / _check_tank_energy.py (positions, velocity_mass)
  rest:  rest_dump.npz of _check_tank_at_rest.py (position, velocity, material)
"""
import argparse

import numpy as np
from scipy.spatial import cKDTree


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dump")
    parser.add_argument("dx", type=float)
    parser.add_argument("--hdx", type=float, default=3.0)
    parser.add_argument("--format", choices=("final", "rest"), default="final")
    arguments = parser.parse_args()
    dump = np.load(arguments.dump)
    if arguments.format == "final":
        live = dump["positions"][:, 3] > 0
        fluid = live & (dump["material"] == 0)
        x = dump["positions"][fluid, :3].astype(np.float64)
        v = dump["velocity_mass"][fluid, :3].astype(np.float64)
    else:
        fluid = dump["material"] == 0
        x = dump["position"][fluid].astype(np.float64)
        v = dump["velocity"][fluid].astype(np.float64)
    support = arguments.hdx * arguments.dx
    mass = 998.0 * arguments.dx ** 3
    tree = cKDTree(x)
    smoothed = np.zeros_like(v)
    for start in range(0, x.shape[0], 40000):
        block = np.arange(start, min(start + 40000, x.shape[0]))
        neighbours = tree.query_ball_point(x[block], support)
        lengths = np.fromiter((len(n) for n in neighbours), dtype=np.int64, count=block.size)
        i = np.repeat(np.arange(block.size), lengths)
        j = np.concatenate([np.asarray(n, dtype=np.int64) for n in neighbours])
        q = np.linalg.norm(x[j] - x[block][i], axis=1) / support
        weight = np.where(q < 1.0, (1.0 - q) ** 6 * (35.0 / 3.0 * q * q + 6.0 * q + 1.0), 0.0)
        total = np.bincount(i, weights=weight, minlength=block.size)
        for axis in range(3):
            smoothed[block, axis] = np.bincount(i, weights=weight * v[j, axis], minlength=block.size) / total
    fluctuation = v - smoothed
    radius = np.hypot(x[:, 0], x[:, 2])
    zones = {"Rushton box": (radius < 0.072) & (x[:, 1] > 0.0187) & (x[:, 1] < 0.0585),
             "PBT box": (radius < 0.072) & (x[:, 1] > 0.165) & (x[:, 1] < 0.225)}
    zones["bulk"] = ~(zones["Rushton box"] | zones["PBT box"])
    zones["tank"] = np.ones(x.shape[0], dtype=bool)
    print(f"{arguments.dump}: {x.shape[0]:,} fluid particles, dx {arguments.dx * 1e3:g} mm, kernel support {support * 1e3:.1f} mm")
    print("  zone          KE of v, J   KE of v', J   share   rms |v'| mm/s")
    for name, mask in zones.items():
        kinetic = 0.5 * mass * (v[mask] ** 2).sum()
        noise = 0.5 * mass * (fluctuation[mask] ** 2).sum()
        print(f"  {name:12s}  {kinetic:9.4f}    {noise:9.4f}    {noise / kinetic:5.3f}   "
              f"{np.sqrt((fluctuation[mask] ** 2).sum(axis=1).mean()) * 1e3:8.1f}")


if __name__ == "__main__":
    main()
