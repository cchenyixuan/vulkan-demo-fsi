"""_analyze_mean_speed.py — domain-averaged speed of a dump, from the particles and
from the velocity field interpolated to a regular grid (2026-09-28).

The mean of |v_i| over the particles is biased high: every particle carries a
velocity fluctuation about the local mean, and |v| is a convex function, so
fluctuations only increase the mean speed (<|U + u'|> ~ U + sigma^2 / U per
transverse component). A grid-based LES reports the speed of its filtered field.
This script makes the comparable SPH quantity:

  v(x_g) = sum_j v_j W(|x_g - x_j|; R) / sum_j W(|x_g - x_j|; R)      (Shepard, FLUID particles)

on a regular grid of spacing g, Wendland C4 with support radius R, and averages
|v(x_g)| over the grid nodes that lie in the fluid (kernel sum at least half of
the full-support value). It reports the result for several filter radii together
with the equivalent box-filter width of the kernel (the box with the same second
moment, width = sqrt(12) * sigma), to be compared with the LES cell size, and the
kinetic energy per unit mass of the raw and of the filtered field.

Reference (Rautenbach et al. 2026, Table 2, rolling mean over 25-75 s of the
domain-averaged speed): M-Star 0.1495 m/s (lattice 0.95 mm), Fluent 0.1566 m/s
(3.4M cells, about 2.1 mm).

Usage (repo root):
    python experiment/v1/checks/_analyze_mean_speed.py DUMP.npz --dx 0.003 [--radius-factors 1.5 2 3]
        [--radius 0.009 ...] [--grid-spacing G] [--fluid-group 0] [--upper-height 0.25]
DUMP.npz is a --dump of _run_v1_headless.py (positions, material, velocity_mass, status).
"""
import argparse
import json
import math
import sys

import numpy as np

REFERENCE = (("M-Star", 0.149463), ("Fluent", 0.156604))


def wendland_c4(q):
    return (1.0 - q) ** 6 * (35.0 / 3.0 * q * q + 6.0 * q + 1.0)


def equivalent_box_width(radius):
    """Width of the box filter with the same per-axis second moment as the 3D
    Wendland C4 kernel of support `radius`."""
    q = np.linspace(0.0, 1.0, 20001)
    weight = wendland_c4(q) * q * q                      # radial measure
    second = np.trapz(weight * q * q, q) / np.trapz(weight, q)     # <r^2> / R^2
    sigma = radius * math.sqrt(second / 3.0)
    return math.sqrt(12.0) * sigma


def load_fluid(path, fluid_group):
    archive = np.load(path)
    status = json.loads(str(archive["status"]))
    alive = int(status["alive_particle_count"])
    positions = archive["positions"]
    live = np.zeros(positions.shape[0], dtype=bool)
    live[1:alive + 1] = True
    live &= positions[:, 3] > 0
    fluid = live & (archive["material"] == fluid_group)
    return (positions[fluid, :3].astype(np.float64),
            archive["velocity_mass"][fluid, :3].astype(np.float64))


def interpolate_to_grid(positions, velocity, spacing, radius):
    """Particle-to-grid scatter of the Shepard sums. Returns node coordinates
    shape, kernel sum and the interpolated velocity (nodes without any
    particle inside the support get velocity 0 and kernel sum 0)."""
    low = positions.min(axis=0) - radius
    high = positions.max(axis=0) + radius
    shape = np.ceil((high - low) / spacing).astype(np.int64) + 2
    total = int(shape[0] * shape[1] * shape[2])
    relative = (positions - low) / spacing
    base = np.floor(relative).astype(np.int64)
    fraction = (relative - base) * spacing                # in [0, spacing)
    reach = int(math.ceil(radius / spacing))
    kernel_sum = np.zeros(total)
    momentum = np.zeros((3, total))
    radius_squared = radius * radius
    offsets = range(-reach, reach + 2)
    for offset_x in offsets:
        delta_x = offset_x * spacing - fraction[:, 0]
        for offset_y in offsets:
            delta_y = offset_y * spacing - fraction[:, 1]
            partial = delta_x * delta_x + delta_y * delta_y
            if partial.min() >= radius_squared:
                continue
            for offset_z in offsets:
                delta_z = offset_z * spacing - fraction[:, 2]
                distance_squared = partial + delta_z * delta_z
                inside = distance_squared < radius_squared
                if not inside.any():
                    continue
                weight = wendland_c4(np.sqrt(distance_squared[inside]) / radius)
                node = ((base[inside, 0] + offset_x) * shape[1] + (base[inside, 1] + offset_y)) * shape[2] \
                    + (base[inside, 2] + offset_z)
                kernel_sum += np.bincount(node, weights=weight, minlength=total)
                for axis in range(3):
                    momentum[axis] += np.bincount(node, weights=weight * velocity[inside, axis], minlength=total)
    filled = kernel_sum > 0.0
    field = np.zeros((total, 3))
    field[filled] = (momentum[:, filled] / kernel_sum[filled]).T
    node_height = low[1] + (np.arange(total) // shape[2] % shape[1]) * spacing
    return kernel_sum, field, node_height


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("dump")
    parser.add_argument("--dx", type=float, required=True, help="particle spacing of the run (m)")
    parser.add_argument("--radius-factors", type=float, nargs="*", default=[1.5, 2.0, 3.0],
                        help="filter support radii in units of dx")
    parser.add_argument("--radius", type=float, nargs="*", default=[], help="additional absolute radii (m)")
    parser.add_argument("--grid-spacing", type=float, default=None, help="grid spacing (default: dx)")
    parser.add_argument("--fluid-group", type=int, default=0)
    parser.add_argument("--upper-height", type=float, default=0.25, help="report y >= this separately (m)")
    arguments = parser.parse_args()

    positions, velocity = load_fluid(arguments.dump, arguments.fluid_group)
    speed = np.linalg.norm(velocity, axis=1)
    upper = positions[:, 1] >= arguments.upper_height
    spacing = arguments.grid_spacing or arguments.dx
    print(f"{arguments.dump}: {positions.shape[0]:,} fluid particles, dx = {arguments.dx * 1e3:g} mm, "
          f"grid spacing {spacing * 1e3:g} mm")
    print(f"  particles:            mean |v| = {speed.mean():.4f} m/s   upper {speed[upper].mean():.4f}   "
          f"kinetic energy <|v|^2>/2 = {0.5 * (speed ** 2).mean():.5f} m^2/s^2")
    radii = [factor * arguments.dx for factor in arguments.radius_factors] + list(arguments.radius)
    for radius in sorted({round(radius, 9) for radius in radii}):
        kernel_sum, field, node_height = interpolate_to_grid(positions, velocity, spacing, radius)
        reference_sum = np.percentile(kernel_sum[kernel_sum > 0.0], 90.0)
        valid = kernel_sum >= 0.5 * reference_sum
        node_speed = np.linalg.norm(field[valid], axis=1)
        node_upper = node_height[valid] >= arguments.upper_height
        print(f"  grid, R = {radius * 1e3:5.2f} mm ({radius / arguments.dx:.2f} dx, box width "
              f"{equivalent_box_width(radius) * 1e3:4.2f} mm): mean |v| = {node_speed.mean():.4f} m/s "
              f"({node_speed.mean() / speed.mean():.3f} of particles)   upper {node_speed[node_upper].mean():.4f}   "
              f"kinetic energy {0.5 * (node_speed ** 2).mean():.5f}   fluid nodes {valid.sum() * spacing ** 3 * 1e3:.2f} L")
    print("  reference (paper Table 2, mean over 25-75 s): "
          + ", ".join(f"{name} {value:.4f} m/s" for name, value in REFERENCE))
    return 0


if __name__ == "__main__":
    sys.exit(main())
