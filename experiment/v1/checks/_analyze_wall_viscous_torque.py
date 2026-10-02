"""
_analyze_wall_viscous_torque.py — viscous part of the fluid torque on the tank walls from a final dump
(2026-10-03, needs scipy: base Anaconda).

Rebuilds the viscous term of force.comp solid_reaction_acceleration() for the ordinary wall particles
(static, velocity 0):

    a_ij = 2 (d + 2) nu V_j  (v_i - v_j) . x_ij / (|x_ij|^2 + eps h^2)  grad_i W(x_ij),   x_ij = x_i - x_j
    V_j  = m_i / rho_j,      force on wall particle j = - sum_i m_i a_ij,   torque about +y

i = fluid, j = wall, Wendland C4 with support h, eps = 0.01. The KCG matrix of the fluid particle is not
in the dump and is taken as the identity (it is close to it next to a wall with full support). The
pressure term is NOT rebuilt: under the 2000 Pa background pressure the radial pressure force on the
cylinder is about 770 N, and the identity in place of the KCG matrix turns a fraction of a per mille of
it into a torque of the order of the whole wall torque. The pressure part follows as the solver's wall
torque minus this viscous part.

usage: python experiment/v1/checks/_analyze_wall_viscous_torque.py DUMP_final.npz DX
"""
import sys

import numpy as np
from scipy.spatial import cKDTree

MASS_FACTOR = 1.2187
KINEMATIC_VISCOSITY = 1.0e-6


def main(dump_path, spacing):
    dump = np.load(dump_path)
    support = 3.0 * spacing
    gradient_coefficient = 495.0 / (32.0 * np.pi * support ** 4)     # Wendland C4, 3D, support h
    positions = dump["positions"]
    live = positions[:, 3] > 0
    material = dump["material"]
    x = positions[:, :3].astype(np.float64)
    velocity = dump["velocity_mass"][:, :3].astype(np.float64)
    mass = dump["velocity_mass"][:, 3].astype(np.float64)
    density = dump["density_pressure"][:, 0].astype(np.float64)
    pressure = dump["density_pressure"][:, 1].astype(np.float64)
    radius = np.hypot(x[:, 0], x[:, 2])
    fluid = live & (material == 0)
    groups = [group for group in np.unique(material[live]) if group != 0]
    wall_group = max(groups, key=lambda group: np.median(radius[live & (material == group)]))
    wall = live & (material == wall_group) & (pressure > -1.0e5)          # thin-plate particles are marked < 0
    fluid_index, wall_index = np.nonzero(fluid)[0], np.nonzero(wall)[0]
    tree = cKDTree(x[fluid_index])
    torque = np.zeros(wall_index.size)
    for start in range(0, wall_index.size, 20000):
        block = wall_index[start:start + 20000]
        neighbours = tree.query_ball_point(x[block], support)
        lengths = np.fromiter((len(n) for n in neighbours), dtype=np.int64, count=block.size)
        if lengths.sum() == 0:
            continue
        j = np.repeat(block, lengths)
        i = fluid_index[np.concatenate([np.asarray(n, dtype=np.int64) for n in neighbours if len(n)])]
        difference = x[i] - x[j]
        distance_squared = (difference ** 2).sum(axis=1)
        distance = np.sqrt(distance_squared)
        keep = (distance_squared > 1e-24) & (distance < support)
        i, j, difference, distance, distance_squared = (i[keep], j[keep], difference[keep], distance[keep],
                                                       distance_squared[keep])
        q = distance / support
        gradient = (gradient_coefficient * (1.0 - q) ** 5 * q * ((-280.0 / 3.0) * q - 56.0 / 3.0)
                    / distance)[:, None] * difference
        acceleration = (10.0 * KINEMATIC_VISCOSITY * (mass[i] / density[j])
                        * ((velocity[i] - velocity[j]) * difference).sum(axis=1)
                        / (distance_squared + 0.01 * support ** 2))[:, None] * gradient
        force = -mass[i][:, None] * acceleration
        torque[start:start + block.size] += np.bincount(
            np.searchsorted(block, j), weights=x[j, 2] * force[:, 0] - x[j, 0] * force[:, 2], minlength=block.size)
    wall_x = x[wall_index]
    wall_radius, wall_height = radius[wall_index], wall_x[:, 1]
    parts = {"cylinder wall (r > 139.5 mm, 0 < y < 426.5 mm)": (wall_radius > 0.1395) & (wall_height > 0) & (wall_height < 0.4265),
             "100 < r < 139.5 mm (brackets, sheet baffles)": (wall_radius > 0.10) & (wall_radius <= 0.1395) & (wall_height > 0) & (wall_height < 0.4265),
             "floor (y <= 0)": wall_height <= 0,
             "lid (y >= 426.5 mm)": wall_height >= 0.4265,
             "all ordinary wall particles": np.ones(wall_index.size, dtype=bool)}
    print(f"{dump_path}: dx {spacing * 1e3:.0f} mm, {fluid.sum():,} fluid, {wall.sum():,} ordinary wall particles")
    print("  viscous torque about +y, mN m (divided by the mass factor)")
    for name, mask in parts.items():
        print(f"  {name:48s} {torque[mask].sum() / MASS_FACTOR * 1e3:8.2f}")


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]))
