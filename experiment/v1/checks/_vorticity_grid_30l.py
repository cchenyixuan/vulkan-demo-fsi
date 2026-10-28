"""_vorticity_grid_30l.py — vorticity magnitude of a 30 L tank snapshot on a uniform grid, for 3D viewing
(2026-10-29). SPH dumps and Fluent LES snapshots go through the same steps, so the two can be compared:

1. samples: SPH fluid particles (positions, velocity_mass of the dump) or Fluent cell centroids and velocities
   (interpolation file of _fluent_snapshot_reports.py journal cells). Fluent: y shifted by -58.5 mm into the
   generator frame; a mirrored snapshot (26..34 s fine, negative swirl in the Rushton box) is mirrored back
   (z -> -z); velocities in the rotor zones are turned by the 12 deg lag of the sliding mesh, as in
   _analyze_rushton_lower_flow.py.
2. binning: sums of u and of the sample count on a uniform grid (spacing --cell, default 3 mm) over
   x, z in [-0.147, 0.147] m, y in [-0.0618, 0.4265] m; both sums smoothed with the same Gaussian,
   u = smoothed sum / smoothed count (normalised convolution). Scales sigma, 2, 4, 8 sigma (--sigma, default
   3 mm): each cell takes the finest scale whose kernel holds at least --fill-count samples. SPH (2 mm particles) always
   gets the finest; Fluent's upper bulk cells are over 10 mm, so there the field is as smooth as the mesh is.
3. vorticity omega = curl u by central differences (numpy.gradient); fluid region: SPH cells whose finest
   smoothed count is at least --min-weight of its median, eroded by one cell; Fluent takes the fluid region of an
   SPH grid (--mask-from), since its sparse bulk cells cannot define one. Everything else is masked.
4. output NPZ: log10 |omega| as float16 (NaN = masked), the grid, time / label, and |omega| percentiles.

usage (repo root):
    python experiment/v1/checks/_vorticity_grid_30l.py --sph DUMP.npz --out OUT.npz [--time 20]
    python experiment/v1/checks/_vorticity_grid_30l.py --fluent fine25.ip --mask-from SPH.npz --out OUT.npz [--time 25]
"""
import argparse
import pathlib
import sys

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

FLUENT_Y_SHIFT = 0.0585
HALF_WIDTH = 0.147
Y_LOW, Y_HIGH = -0.0618, 0.4265


def read_interpolation_file(path):
    """Fluent binary interpolation file -> dict of arrays (copy of _fluent_snapshot_reports.read_interpolation_file)"""
    raw = pathlib.Path(path).read_bytes()
    first = raw.index(b"(")
    header = raw[:first].decode().split()
    count, field_count = int(header[2]), int(header[3])
    names = ["x", "y", "z"] + header[4:4 + field_count]
    arrays, position = {}, first
    for name in names:
        position = raw.index(b"(", position)
        arrays[name] = np.frombuffer(raw, dtype="<f8", count=count, offset=position + 1).copy()
        position += 1 + 8 * count
    return arrays


def turn_about_y(vectors, angle):
    """rotate by angle in the sense of +e_theta (right hand about +y), as _analyze_rushton_lower_flow.turn_about_y"""
    c, s = np.cos(angle), np.sin(angle)
    out = vectors.copy()
    out[:, 0] = c * vectors[:, 0] + s * vectors[:, 2]
    out[:, 2] = -s * vectors[:, 0] + c * vectors[:, 2]
    return out


def sph_samples(path):
    dump = np.load(path)
    position, material = dump["positions"], dump["material"]
    fluid = (position[:, 3] > 0) & (material == 0)
    return position[fluid, :3].astype(np.float64), dump["velocity_mass"][fluid, :3].astype(np.float64), {}


def fluent_samples(path, rotor_lag_degrees=12.0):
    cells = read_interpolation_file(path)
    position = np.stack([cells["x"], cells["y"] - FLUENT_Y_SHIFT, cells["z"]], axis=1)
    velocity = np.stack([cells["x-velocity"], cells["y-velocity"], cells["z-velocity"]], axis=1)
    r = np.maximum(np.hypot(position[:, 0], position[:, 2]), 1e-12)
    rotor = (r < 0.072) & (position[:, 1] > 0.0187) & (position[:, 1] < 0.0585)
    swirl = ((velocity[rotor, 0] * position[rotor, 2] - velocity[rotor, 2] * position[rotor, 0]) / r[rotor]).sum()
    mirror = bool(swirl < 0.0)
    if mirror:
        position[:, 2] *= -1.0
        velocity[:, 2] *= -1.0
    r = np.hypot(position[:, 0], position[:, 2])
    y = position[:, 1]
    rotor_zone = (r < 0.0718) & (((y > 0.0187) & (y < 0.0585)) | ((y > 0.165) & (y < 0.225)))
    velocity[rotor_zone] = turn_about_y(velocity[rotor_zone], np.radians(rotor_lag_degrees))
    return position, velocity, {"mirrored": mirror}


def vorticity_grid(position, velocity, cell, sigma, min_weight, fill_count, mask=None):
    edges = [np.arange(-HALF_WIDTH, HALF_WIDTH + 0.5 * cell, cell), np.arange(Y_LOW, Y_HIGH + 0.5 * cell, cell),
             np.arange(-HALF_WIDTH, HALF_WIDTH + 0.5 * cell, cell)]
    shape = tuple(len(edge) - 1 for edge in edges)
    index = [np.digitize(position[:, axis], edges[axis]) - 1 for axis in range(3)]
    inside = np.all([(index[axis] >= 0) & (index[axis] < shape[axis]) for axis in range(3)], axis=0)
    flat = np.ravel_multi_index([index[axis][inside] for axis in range(3)], shape)
    count = np.bincount(flat, minlength=np.prod(shape)).reshape(shape).astype(np.float64)
    totals = [np.bincount(flat, weights=velocity[inside, component], minlength=np.prod(shape)).reshape(shape)
              for component in range(3)]
    # normalised convolution at sigma, 2, 4, 8 sigma: each cell takes the finest scale whose kernel holds at least
    # fill_count samples, N = smoothed count per cell x (2 pi)^1.5 (sigma / cell)^3 (SPH: always the finest;
    # Fluent: the finest its local cell size supports)
    velocity_grid = [np.full(shape, np.nan) for _ in range(3)]
    scale_used = np.zeros(shape, np.uint8)
    finest_count = None
    for level, width in enumerate((sigma / cell, 2 * sigma / cell, 4 * sigma / cell, 8 * sigma / cell), start=1):
        count_smooth = gaussian_filter(count, width, mode="constant")
        if finest_count is None:
            finest_count = count_smooth
        take = (scale_used == 0) & (count_smooth * (2.0 * np.pi) ** 1.5 * width ** 3 >= fill_count)
        for component in range(3):
            smooth = gaussian_filter(totals[component], width, mode="constant") / np.maximum(count_smooth, 1e-30)
            velocity_grid[component][take] = smooth[take]
        scale_used[take] = level
    ux, uy, uz = (np.nan_to_num(component) for component in velocity_grid)
    # axes 0, 1, 2 = x, y, z
    duz_dy = np.gradient(uz, cell, axis=1); duy_dz = np.gradient(uy, cell, axis=2)
    dux_dz = np.gradient(ux, cell, axis=2); duz_dx = np.gradient(uz, cell, axis=0)
    duy_dx = np.gradient(uy, cell, axis=0); dux_dy = np.gradient(ux, cell, axis=1)
    magnitude = np.sqrt((duz_dy - duy_dz) ** 2 + (dux_dz - duz_dx) ** 2 + (duy_dx - dux_dy) ** 2)
    if mask is None:
        valid = finest_count >= min_weight * np.median(finest_count[count > 0])
        # one cell of erosion keeps the one-sided wall gradients out
        valid &= np.all([np.roll(valid, shift, axis) for axis in range(3) for shift in (-1, 1)], axis=0)
    else:
        valid = mask.copy()
    valid &= scale_used > 0
    valid &= np.all([np.roll(scale_used > 0, shift, axis) for axis in range(3) for shift in (-1, 1)], axis=0)
    magnitude[~valid] = np.nan
    return magnitude, edges, scale_used


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sph", help="SPH dump (_check_tank_energy.py --dump-times or --dump)")
    source.add_argument("--fluent", help="Fluent interpolation file with x/y/z-velocity")
    parser.add_argument("--out", required=True)
    parser.add_argument("--time", type=float, default=float("nan"), help="s, stored as metadata")
    parser.add_argument("--label", default=None)
    parser.add_argument("--cell", type=float, default=0.003, help="grid spacing, m")
    parser.add_argument("--sigma", type=float, default=0.003, help="Gaussian smoothing, m")
    parser.add_argument("--min-weight", type=float, default=0.3, help="SPH: mask below this fraction of the median smoothed count")
    parser.add_argument("--fill-count", type=float, default=4.0,
                        help="samples within the kernel needed to use a smoothing scale (Fluent's cells range from 1 mm in "
                             "the rotor boxes to over 10 mm in the upper bulk)")
    parser.add_argument("--mask-from", default=None,
                        help="NPZ of this script whose finite cells define the fluid region (an SPH grid, for Fluent)")
    arguments = parser.parse_args()
    if arguments.sph:
        position, velocity, info = sph_samples(arguments.sph)
        label = arguments.label or pathlib.Path(arguments.sph).stem
    else:
        position, velocity, info = fluent_samples(arguments.fluent)
        label = arguments.label or pathlib.Path(arguments.fluent).stem
    mask = None
    if arguments.mask_from:
        mask = np.isfinite(np.load(arguments.mask_from)["log_vorticity"].astype(np.float32))
    magnitude, edges, scale_used = vorticity_grid(position, velocity, arguments.cell, arguments.sigma, arguments.min_weight,
                                                  arguments.fill_count, mask)
    info["scale_fractions"] = np.array([(scale_used[np.isfinite(magnitude)] == level).mean() for level in (1, 2, 3, 4)])
    finite = magnitude[np.isfinite(magnitude)]
    percentiles = np.percentile(finite, [5, 25, 50, 75, 95, 99, 99.9])
    np.savez_compressed(arguments.out, log_vorticity=np.log10(np.maximum(magnitude, 1e-6)).astype(np.float16),
                        x_edges=edges[0], y_edges=edges[1], z_edges=edges[2], time=arguments.time, label=label,
                        percentiles=percentiles, samples=len(position), **{k: v for k, v in info.items()})
    print(f"{label}: {len(position):,} samples, grid {magnitude.shape}, valid {finite.size:,}; |omega| 1/s "
          f"p5 {percentiles[0]:.2f} p50 {percentiles[2]:.2f} p95 {percentiles[4]:.1f} p99 {percentiles[5]:.1f} "
          f"p99.9 {percentiles[6]:.1f}; smoothing sigma / 2 / 4 / 8 sigma in "
          + " / ".join(f"{100 * value:.0f}" for value in info["scale_fractions"]) + " % of the valid cells"
          + (f"; mirrored {info['mirrored']}" if "mirrored" in info else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
