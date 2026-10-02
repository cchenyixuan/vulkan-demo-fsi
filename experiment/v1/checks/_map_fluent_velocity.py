"""
_map_fluent_velocity.py — initial fluid velocities of a 30 L tank case taken from a Fluent LES snapshot
(2026-10-03, needs scipy: base Anaconda).

Every particle of CASE_DIR/fluid.obj gets the velocity of the nearest Fluent cell centroid. The Fluent
cells come from an interpolation file (_fluent_snapshot_reports.py journal cells); Fluent y is shifted by
-58.5 mm into the generator frame. Use a snapshot that turns the same way as our rotor (+y right-hand):
the fine-mesh 25 s snapshot and the coarse snapshots do, the fine 26..34 s snapshots are mirrored.
The result is an (n_fluid, 3) float32 .npy in fluid.obj order for _check_tank_energy.py --initial-velocity.

usage: python experiment/v1/checks/_map_fluent_velocity.py CASE_DIR SNAPSHOT.ip OUT.npy
"""
import pathlib
import sys

import numpy as np
from scipy.spatial import cKDTree

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from utils.sph.obj_loader import load_obj_vertices            # noqa: E402
from _fluent_snapshot_reports import read_interpolation_file  # noqa: E402

FLUENT_Y_SHIFT = 0.0585


def main(case_directory, snapshot, out):
    fluid = load_obj_vertices(pathlib.Path(case_directory) / "fluid.obj").astype(np.float64)
    cells = read_interpolation_file(snapshot)
    centroids = np.stack([cells["x"], cells["y"] - FLUENT_Y_SHIFT, cells["z"]], axis=1)
    velocity = np.stack([cells["x-velocity"], cells["y-velocity"], cells["z-velocity"]], axis=1)
    distance, index = cKDTree(centroids).query(fluid)
    mapped = velocity[index].astype(np.float32)
    np.save(out, mapped)
    radius = np.maximum(np.hypot(fluid[:, 0], fluid[:, 2]), 1e-12)
    swirl = (mapped[:, 0] * fluid[:, 2] - mapped[:, 2] * fluid[:, 0]) / radius
    spacing = np.median(np.diff(np.unique(np.round(fluid[:, 1], 6))))
    print(f"{fluid.shape[0]:,} particles from {centroids.shape[0]:,} cells; nearest-cell distance median "
          f"{np.median(distance) * 1e3:.2f} mm, 99 % {np.percentile(distance, 99) * 1e3:.2f} mm, max {distance.max() * 1e3:.2f} mm")
    print(f"mapped field: mean |u| {np.linalg.norm(mapped, axis=1).mean():.4f} m/s, mean u_theta {swirl.mean():+.4f} m/s "
          f"(positive = +y right-hand, the rotor's sense), KE {0.5 * 998.0 * spacing ** 3 * (mapped.astype(np.float64) ** 2).sum():.4f} J "
          f"(dx {spacing * 1e3:.2f} mm) -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
