"""
_analyze_axisymmetric_energy.py — bulk kinetic energy of the 30 L tank split into the azimuthal-mean flow
and the rest, for SPH dumps and Fluent snapshots alike (2026-10-03).

The bulk (the tank outside Fluent's rotor boxes rt_rotorbox / pbt_rotorbox: r < 72 mm, y 18.7..58.5 and
165..225 mm) is divided into r-y bins of 8 mm. In every bin the azimuthal mean of the radial, tangential
and axial velocity is taken; per component

    KE_mean = 1/2 rho sum_bins V_bin <u>_bin^2                     azimuthal-mean flow (circulation, swirl)
    KE_rest = 1/2 rho sum_bins V_bin (<u^2>_bin - <u>_bin^2)        the rest (turbulence, baffle wakes, noise)

SPH: every particle stands for dx^3, V_bin = count dx^3, bin means are particle means.
Fluent: the interpolation files (_fluent_snapshot_reports.py journal cells) carry no cell volumes. Bin means
are cell means (the cells of one 8 mm bin are of similar size; the bulk cells of the fine mesh reach about
8 mm, so 6 mm bins left gaps) and V_bin comes from a reference SPH dump of the same tank (--volumes), so both
sides use the same bin volumes. Fluent y is shifted by -58.5 mm into the generator frame. Check: the fine-mesh
total is 0.516 J against Fluent's own volume integral of 0.494 J (+4 %).
Also printed: flow numbers Q / (N D^3) of the azimuthal-mean field (pumping()) and the azimuthal-mean velocity
of the Rushton jet and the PBT discharge (jets()). --figure draws r-y maps of the azimuthal-mean flow.

usage (base Anaconda or the solver environment):
    python experiment/v1/checks/_analyze_axisymmetric_energy.py --volumes REF_final.npz 0.003 \
        --sph A_final.npz 0.003 [--sph ...] --fluent fine25.ip [--fluent ...]
"""
import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _fluent_snapshot_reports import read_interpolation_file   # noqa: E402

DENSITY = 998.0
BIN = 0.008
R_EDGES = np.arange(0.0, 0.144 + BIN, BIN)
Y_EDGES = np.arange(-0.060, 0.4265 + BIN, BIN)
FLUENT_Y_SHIFT = 0.0585


def bulk_mask(x):
    radius = np.hypot(x[:, 0], x[:, 2])
    in_box = (radius < 0.072) & (((x[:, 1] > 0.0187) & (x[:, 1] < 0.0585)) | ((x[:, 1] > 0.165) & (x[:, 1] < 0.225)))
    return ~in_box


def bin_moments(x, v):
    """per r-y bin: count, mean of (u_r, u_theta, u_y) and mean of their squares"""
    radius = np.maximum(np.hypot(x[:, 0], x[:, 2]), 1e-12)
    components = np.stack([(v[:, 0] * x[:, 0] + v[:, 2] * x[:, 2]) / radius,
                           (v[:, 0] * x[:, 2] - v[:, 2] * x[:, 0]) / radius,
                           v[:, 1]], axis=1)
    r_index = np.clip(np.digitize(radius, R_EDGES) - 1, 0, len(R_EDGES) - 2)
    y_index = np.clip(np.digitize(x[:, 1], Y_EDGES) - 1, 0, len(Y_EDGES) - 2)
    flat = r_index * (len(Y_EDGES) - 1) + y_index
    size = (len(R_EDGES) - 1) * (len(Y_EDGES) - 1)
    count = np.bincount(flat, minlength=size).astype(np.float64)
    safe = np.maximum(count, 1.0)
    mean = np.stack([np.bincount(flat, weights=components[:, k], minlength=size) / safe for k in range(3)], axis=1)
    mean_square = np.stack([np.bincount(flat, weights=components[:, k] ** 2, minlength=size) / safe
                            for k in range(3)], axis=1)
    return count, mean, mean_square


def energies(volume, count, mean, mean_square, minimum=1):
    used = (count >= minimum) & (volume > 0)
    weight = 0.5 * DENSITY * volume[used][:, None]
    total = (weight * mean_square[used]).sum(axis=0)
    azimuthal_mean = (weight * mean[used] ** 2).sum(axis=0)
    return total, azimuthal_mean


def sph_arrays(path):
    dump = np.load(path)
    fluid = (dump["material"] == 0) & (dump["positions"][:, 3] > 0)
    return dump["positions"][fluid, :3].astype(np.float64), dump["velocity_mass"][fluid, :3].astype(np.float64)


def report(label, total, azimuthal_mean):
    rest = total - azimuthal_mean
    print(f"  {label:44s} {total.sum():6.3f} | mean {azimuthal_mean[0]:.3f} {azimuthal_mean[1]:.3f} "
          f"{azimuthal_mean[2]:.3f} = {azimuthal_mean.sum():.3f} | rest {rest[0]:.3f} {rest[1]:.3f} {rest[2]:.3f} "
          f"= {rest.sum():.3f}")
    return total, azimuthal_mean


PUMPING_SCALE = (200.0 / 60.0) * 0.096 ** 3          # N D^3, m^3/s


def pumping(count, mean):
    """flow numbers Q / (N D^3) from the azimuthal-mean field of all fluid bins:
    PBT: downward flux through the bin row just below the PBT box (y ~ 161 mm), r < 48 mm;
    Rushton: outward flux through the bin column at r ~ 72 mm over y 18.7..58.5 mm;
    wall: upward flux through the bin row at y ~ 120 mm, r > 100 mm."""
    shape = (len(R_EDGES) - 1, len(Y_EDGES) - 1)
    valid = count.reshape(shape) >= 1
    u_r = np.where(valid, mean[:, 0].reshape(shape), 0.0)
    u_y = np.where(valid, mean[:, 2].reshape(shape), 0.0)
    r_mid, y_mid = 0.5 * (R_EDGES[1:] + R_EDGES[:-1]), 0.5 * (Y_EDGES[1:] + Y_EDGES[:-1])
    ring = 2.0 * np.pi * r_mid * BIN                      # area of one annulus of a horizontal plane
    row_pbt = np.argmin(np.abs(y_mid - 0.161))
    row_wall = np.argmin(np.abs(y_mid - 0.120))
    column = np.argmin(np.abs(r_mid - 0.072))
    pbt = -(ring * u_y[:, row_pbt])[r_mid < 0.048].sum()
    wall = (ring * u_y[:, row_wall])[r_mid > 0.100].sum()
    band = (y_mid > 0.0187) & (y_mid < 0.0585)
    rushton = (2.0 * np.pi * r_mid[column] * BIN * u_r[column, band]).sum()
    return pbt / PUMPING_SCALE, rushton / PUMPING_SCALE, wall / PUMPING_SCALE


def jets(count, mean):
    """azimuthal-mean (u_r, u_theta, u_y) of the Rushton jet just outside the blade tips (r 48..64 mm,
    y 28..48 mm) and of the PBT discharge just below the PBT (r 16..48 mm, y 152..168 mm), bin means
    weighted with the bin counts"""
    shape = (len(R_EDGES) - 1, len(Y_EDGES) - 1)
    r_mid, y_mid = 0.5 * (R_EDGES[1:] + R_EDGES[:-1]), 0.5 * (Y_EDGES[1:] + Y_EDGES[:-1])
    weights = count.reshape(shape)
    out = []
    for (r0, r1), (y0, y1) in (((0.048, 0.064), (0.028, 0.048)), ((0.016, 0.048), (0.152, 0.168))):
        select = ((r_mid >= r0) & (r_mid <= r1))[:, None] & ((y_mid >= y0) & (y_mid <= y1))[None, :]
        w = weights * select
        out.append([(w * mean[:, k].reshape(shape)).sum() / max(w.sum(), 1.0) for k in range(3)])
    return out


def figure(maps, out):
    """r-y maps of the azimuthal-mean flow: |(u_r, u_y)| with arrows, and u_theta, one row per data set"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    shape = (len(R_EDGES) - 1, len(Y_EDGES) - 1)
    r_mid, y_mid = 0.5 * (R_EDGES[1:] + R_EDGES[:-1]), 0.5 * (Y_EDGES[1:] + Y_EDGES[:-1])
    fig, axes = plt.subplots(2, len(maps), figsize=(3.2 * len(maps), 9.5), squeeze=False)
    for column, (label, (count, mean)) in enumerate(maps.items()):
        valid = (count >= 1).reshape(shape)
        u_r, u_t, u_y = (np.where(valid, mean[:, k].reshape(shape), np.nan) for k in range(3))
        meridional = np.hypot(u_r, u_y)
        image = axes[0, column].pcolormesh(R_EDGES * 1e3, Y_EDGES * 1e3, meridional.T, vmin=0, vmax=0.25, cmap="viridis")
        step = 2
        grid_r, grid_y = np.meshgrid(r_mid[::step] * 1e3, y_mid[::step] * 1e3, indexing="ij")
        axes[0, column].quiver(grid_r, grid_y, u_r[::step, ::step], u_y[::step, ::step], color="w", scale=3.0, width=0.004)
        axes[0, column].set_title(f"{label}\n|(u_r, u_y)| azimuthal mean", fontsize=8)
        axes[0, column].set_aspect("equal")
        swirl = axes[1, column].pcolormesh(R_EDGES * 1e3, Y_EDGES * 1e3, np.abs(u_t).T, vmin=0, vmax=0.25, cmap="magma")
        axes[1, column].set_title("|u_theta| azimuthal mean", fontsize=8)
        axes[1, column].set_aspect("equal")
        for row in (0, 1):
            axes[row, column].set_xlabel("r, mm")
            axes[row, column].set_ylabel("y, mm")
            for y0, y1 in ((18.7, 58.5), (165.0, 225.0)):
                axes[row, column].plot([0, 72, 72, 0], [y0, y0, y1, y1], "w--", linewidth=0.8)
    fig.colorbar(image, ax=axes[0, :].tolist(), shrink=0.6, label="m/s")
    fig.colorbar(swirl, ax=axes[1, :].tolist(), shrink=0.6, label="m/s")
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--volumes", nargs=2, metavar=("REF_DUMP", "DX"), required=True)
    parser.add_argument("--sph", nargs=2, action="append", default=[], metavar=("DUMP", "DX"))
    parser.add_argument("--fluent", action="append", default=[])
    parser.add_argument("--figure", default=None, help="r-y maps of the azimuthal-mean flow (all fluid, rotor boxes included)")
    arguments = parser.parse_args()
    maps = {}

    x, _ = sph_arrays(arguments.volumes[0])
    bulk = bulk_mask(x)
    reference_count, _, _ = bin_moments(x[bulk], np.zeros_like(x[bulk]))
    reference_volume = reference_count * float(arguments.volumes[1]) ** 3
    print(f"bins {BIN * 1e3:.0f} mm; bulk volume from {pathlib.Path(arguments.volumes[0]).name}: "
          f"{reference_volume.sum() * 1e3:.2f} L")
    print("  data set                                     KE bulk | azimuthal mean: radial tangential axial"
          " = sum | rest: radial tangential axial = sum   [J]")
    for path, spacing in arguments.sph:
        x, v = sph_arrays(path)
        bulk = bulk_mask(x)
        count, mean, mean_square = bin_moments(x[bulk], v[bulk])
        label = pathlib.Path(path).name.replace("_final.npz", "").replace("_budget", "")
        report(label, *energies(count * float(spacing) ** 3, count, mean, mean_square))
        all_count, all_mean, _ = bin_moments(x, v)
        maps[label] = (all_count, all_mean * [1.0, np.sign(all_mean[:, 1].sum() or 1.0), 1.0])
    fluent_results, fluent_maps = [], []
    for path in arguments.fluent:
        cells = read_interpolation_file(path)
        x = np.stack([cells["x"], cells["y"] - FLUENT_Y_SHIFT, cells["z"]], axis=1)
        v = np.stack([cells["x-velocity"], cells["y-velocity"], cells["z-velocity"]], axis=1)
        bulk = bulk_mask(x)
        count, mean, mean_square = bin_moments(x[bulk], v[bulk])
        fluent_results.append(report(pathlib.Path(path).stem, *energies(reference_volume, count, mean, mean_square)))
        all_count, all_mean, _ = bin_moments(x, v)
        # mirrored snapshots turn the other way: orient the swirl positive
        fluent_maps.append((all_count, all_mean * [1.0, np.sign(all_mean[:, 1].sum() or 1.0), 1.0]))
    if fluent_results:
        total = np.mean([r[0] for r in fluent_results], axis=0)
        azimuthal_mean = np.mean([r[1] for r in fluent_results], axis=0)
        report(f"Fluent, mean of {len(fluent_results)} snapshots", total, azimuthal_mean)
        count = np.sum([m[0] for m in fluent_maps], axis=0)
        mean = np.sum([m[0][:, None] * m[1] for m in fluent_maps], axis=0) / np.maximum(count, 1.0)[:, None]
        maps = {f"Fluent, {len(fluent_maps)} snapshots": (count, mean), **maps}
    print("\n  flow numbers Q / (N D^3) of the azimuthal-mean field:  PBT down (y 161 mm, r < 48 mm)   "
          "Rushton out (r 72 mm)   wall up (y 120 mm, r > 100 mm)")
    for label, (count, mean) in maps.items():
        pbt, rushton, wall = pumping(count, mean)
        print(f"  {label:44s}  {pbt:6.3f}                          {rushton:6.3f}               {wall:6.3f}")
    print("\n  azimuthal-mean velocity, m/s (u_r, u_theta, u_y):  Rushton jet r 48..64 mm, y 28..48 mm   |   "
          "PBT discharge r 16..48 mm, y 152..168 mm")
    for label, (count, mean) in maps.items():
        rushton, pbt = jets(count, mean)
        print(f"  {label:44s}  {rushton[0]:+.3f} {rushton[1]:+.3f} {rushton[2]:+.3f}   |   "
              f"{pbt[0]:+.3f} {pbt[1]:+.3f} {pbt[2]:+.3f}")
    if arguments.figure:
        figure(maps, arguments.figure)


if __name__ == "__main__":
    main()
