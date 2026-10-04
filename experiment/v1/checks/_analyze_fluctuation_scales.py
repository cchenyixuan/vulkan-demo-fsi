"""
_analyze_fluctuation_scales.py — the non-axisymmetric kinetic energy of the 30 L tank split by scale and region,
Fluent and SPH on the same points with the same filter (2026-10-04, base Anaconda: needs scipy).

Question: our "rest" energy (azimuthal variance in r-y rings, _plot_energy_maps.py) is 1.5 times Fluent's,
mostly above and between the impellers. Is the excess particle-scale noise or flow structure larger than the
kernel?

Sample points: the fluid particles of each SPH dump. Fluent is resampled onto the fluid particles of a
reference SPH dump (--volumes): the value at a point is the mean of the Fluent cells whose centres lie within
the radius of a sphere of volume dx^3 (0.62 dx), or the nearest cell if none (Fluent's bulk cells reach 8 mm).
The velocities of Fluent's rotating zones are first turned by 12 degrees in their sense of rotation
(_analyze_rushton_lower_flow.py). So both sides are sampled at the particle spacing.

Filter: the kernel-smoothed velocity of every point, Shepard mean with the solver's kernel (Wendland C4,
support h = HDX dx, the point itself included):

    v~_i = sum_j W_ij v_j / sum_j W_ij,      v'_i = v_i - v~_i

Per r-y ring (bins of --bin, all fluid, rotor boxes included) and snapshot, energy densities (J/m^3):

    rest        e_n  = 1/2 rho sum_k (<u_k^2> - <u_k>^2)        u_k = u_r, u_theta, u_y of v
    rest > h    e_ns = the same for v~                          structure larger than the kernel
    sub-kernel  e_v  = 1/2 rho <|v'|^2>                         particle-scale part

e_n = e_ns + e_v + 2 <(v~ - <v>) . v'> (the last term is printed as the remainder). Densities are averaged over
the snapshots of a data set and integrated over regions with the ring volumes of the reference dump.

usage:
    python experiment/v1/checks/_analyze_fluctuation_scales.py --volumes REF.npz 0.003 \
        --fluent fine25.ip [--fluent ...] --sph-set LABEL A.npz [B.npz ...] [--sph-set ...] [--figure OUT.png]
"""
import argparse
import pathlib
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _analyze_axisymmetric_energy as axi   # noqa: E402
import _analyze_rushton_lower_flow as lower  # noqa: E402

DENSITY = 998.0
R_BANDS = ((0.0, 0.072, "r < 72"), (0.072, 0.110, "r 72..110"), (0.110, 0.144, "r 110..144"))
Y_BANDS = ((-0.070, 0.0187, "floor, y < 18.7"), (0.0187, 0.0585, "Rushton height"), (0.0585, 0.165, "between"),
           (0.165, 0.225, "PBT height"), (0.225, 0.430, "top, y > 225"))


def shepard(x, v, support):
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
    return smoothed


def fluent_on_points(path, points, dx, rotor_lag_degrees):
    cells = axi.read_interpolation_file(path)
    x = np.stack([cells["x"], cells["y"] - axi.FLUENT_Y_SHIFT, cells["z"]], axis=1)
    v = np.stack([cells["x-velocity"], cells["y-velocity"], cells["z-velocity"]], axis=1)
    rotor = lower.in_fluent_rotor_zones(x)
    radius = np.maximum(np.hypot(x[rotor, 0], x[rotor, 2]), 1e-12)
    sense = np.sign(((v[rotor, 0] * x[rotor, 2] - v[rotor, 2] * x[rotor, 0]) / radius).sum()) or 1.0
    v[rotor] = lower.turn_about_y(v[rotor], sense * np.radians(rotor_lag_degrees))
    tree = cKDTree(x)
    sampled = np.zeros_like(points)
    filled = np.zeros(points.shape[0], dtype=bool)
    reach = dx * (3.0 / (4.0 * np.pi)) ** (1.0 / 3.0)
    for start in range(0, points.shape[0], 100000):
        block = np.arange(start, min(start + 100000, points.shape[0]))
        neighbours = tree.query_ball_point(points[block], reach)
        lengths = np.fromiter((len(n) for n in neighbours), dtype=np.int64, count=block.size)
        i = np.repeat(np.arange(block.size), lengths)
        j = np.concatenate([np.asarray(n, dtype=np.int64) for n in neighbours]) if lengths.sum() else np.zeros(0, np.int64)
        count = np.bincount(i, minlength=block.size)
        for axis in range(3):
            sampled[block, axis] = np.bincount(i, weights=v[j, axis], minlength=block.size) / np.maximum(count, 1)
        filled[block] = count > 0
    if (~filled).any():
        _, nearest = tree.query(points[~filled])
        sampled[~filled] = v[nearest]
    return sampled, float((~filled).mean())


def ring_densities(x, v, smoothed, edges):
    """per ring: e_n, e_ns, e_v, and the cross remainder"""
    count, mean, mean_square = axi.bin_moments(x, v)
    _, mean_s, mean_square_s = axi.bin_moments(x, smoothed)
    radius = np.hypot(x[:, 0], x[:, 2])
    r_index = np.clip(np.digitize(radius, axi.R_EDGES) - 1, 0, len(axi.R_EDGES) - 2)
    y_index = np.clip(np.digitize(x[:, 1], axi.Y_EDGES) - 1, 0, len(axi.Y_EDGES) - 2)
    flat = r_index * (len(axi.Y_EDGES) - 1) + y_index
    fluctuation = ((v - smoothed) ** 2).sum(axis=1)
    sub = np.bincount(flat, weights=fluctuation, minlength=count.size) / np.maximum(count, 1.0)
    e_n = 0.5 * DENSITY * np.clip(mean_square - mean ** 2, 0.0, None).sum(axis=1)
    e_ns = 0.5 * DENSITY * np.clip(mean_square_s - mean_s ** 2, 0.0, None).sum(axis=1)
    e_v = 0.5 * DENSITY * sub
    return count, np.stack([e_n, e_ns, e_v, e_n - e_ns - e_v])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--volumes", nargs=2, required=True, metavar=("REF_DUMP", "DX"))
    parser.add_argument("--fluent", action="append", default=[])
    parser.add_argument("--fluent-rotor-lag", type=float, default=12.0)
    parser.add_argument("--sph-set", nargs="+", action="append", default=[], metavar="LABEL DUMP")
    parser.add_argument("--hdx", type=float, default=3.0)
    parser.add_argument("--bin", type=float, default=0.008)
    parser.add_argument("--min-count", type=int, default=3)
    parser.add_argument("--figure", default=None)
    arguments = parser.parse_args()

    dx = float(arguments.volumes[1])
    support = arguments.hdx * dx
    axi.BIN = arguments.bin
    axi.R_EDGES = np.arange(0.0, 0.144 + axi.BIN, axi.BIN)
    axi.Y_EDGES = np.arange(-0.064, 0.4265 + axi.BIN, axi.BIN)
    reference_x, _ = axi.sph_arrays(arguments.volumes[0])
    reference_count, _, _ = axi.bin_moments(reference_x, np.zeros_like(reference_x))
    volume = reference_count * dx ** 3
    fluid = reference_count > 0

    sets = []
    if arguments.fluent:
        total, used = 0.0, 0.0
        for path in arguments.fluent:
            v, nearest_share = fluent_on_points(path, reference_x, dx, arguments.fluent_rotor_lag)
            count, e = ring_densities(reference_x, v, shepard(reference_x, v, support), axi.R_EDGES)
            ok = count >= arguments.min_count
            total = total + np.where(ok, e, 0.0)
            used = used + ok
            print(f"  {pathlib.Path(path).name}: {nearest_share * 100:.1f} % of the points took the nearest cell")
        sets.append((f"Fluent ({len(arguments.fluent)})", np.where(used > 0, total / np.maximum(used, 1), np.nan)))
    for entry in arguments.sph_set:
        label, paths = entry[0], entry[1:]
        total, used = 0.0, 0.0
        for path in paths:
            x, v = axi.sph_arrays(path)
            count, e = ring_densities(x, v, shepard(x, v, support), axi.R_EDGES)
            ok = count >= arguments.min_count
            total = total + np.where(ok, e, 0.0)
            used = used + ok
        sets.append((f"{label} ({len(paths)})", np.where(used > 0, total / np.maximum(used, 1), np.nan)))
    for _, e in sets:
        e[:, ~fluid] = np.nan

    r_mid = 0.5 * (axi.R_EDGES[1:] + axi.R_EDGES[:-1])
    y_mid = 0.5 * (axi.Y_EDGES[1:] + axi.Y_EDGES[:-1])
    r_grid, y_grid = (a.ravel() for a in np.meshgrid(r_mid, y_mid, indexing="ij"))
    print(f"\nnon-axisymmetric energy, mJ: all = rest of v, >h = rest of the kernel-smoothed v~, <h = sub-kernel "
          f"1/2 rho |v - v~|^2 (h = {support * 1e3:.1f} mm, rings {arguments.bin * 1e3:g} mm)")
    print(f"  {'region':30s}" + "".join(f"{label[:27]:>29s}" for label, _ in sets))
    print(f"  {'':30s}" + "".join(f"{'all     >h     <h':>29s}" for _ in sets))

    def row(name, select):
        cells = []
        for _, e in sets:
            values = [np.nansum(e[k, select] * volume[select]) * 1e3 for k in range(3)]
            cells.append(f"{values[0]:7.1f}{values[1]:7.1f}{values[2]:7.1f}")
        print(f"  {name:30s}" + "".join(f"{c:>29s}" for c in cells))

    for y0, y1, y_name in Y_BANDS:
        row(y_name, (y_grid >= y0) & (y_grid < y1) & fluid)
        for r0, r1, r_name in R_BANDS:
            select = (y_grid >= y0) & (y_grid < y1) & (r_grid >= r0) & (r_grid < r1) & fluid
            if select.any():
                row("   " + r_name, select)
    row("whole tank", fluid)
    remainder = [np.nansum(e[3] * volume) * 1e3 for _, e in sets]
    print("  cross remainder e_n - e_ns - e_v, whole tank, mJ: " + "  ".join(f"{label}: {value:+.1f}"
                                                                         for (label, _), value in zip(sets, remainder)))

    if arguments.figure:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        shape = (len(axi.R_EDGES) - 1, len(axi.Y_EDGES) - 1)
        columns = len(sets) + len(sets) - 1
        fig, axes = plt.subplots(2, columns, figsize=(2.6 * columns + 1.5, 9.0), squeeze=False, layout="constrained")
        titles = ("rest > h, sqrt(2 e_ns / rho)", "sub-kernel, sqrt(2 e_v / rho)")
        image, ratio_image = None, None
        for row_index, k in enumerate((1, 2)):
            for column, (label, e) in enumerate(sets):
                ax = axes[row_index, column]
                image = ax.pcolormesh(axi.R_EDGES * 1e3, axi.Y_EDGES * 1e3, np.sqrt(2.0 * e[k] / DENSITY).reshape(shape).T,
                                      vmin=0.0, vmax=0.2, cmap="magma")
                ax.set_title(f"{label}\n{titles[row_index]}, m/s", fontsize=7)
            for offset, (label, e) in enumerate(sets[1:]):
                ax = axes[row_index, len(sets) + offset]
                with np.errstate(invalid="ignore", divide="ignore"):
                    ratio = (e[k] / sets[0][1][k]).reshape(shape)
                ratio_image = ax.pcolormesh(axi.R_EDGES * 1e3, axi.Y_EDGES * 1e3, np.clip(ratio, 0.125, 8.0).T,
                                            norm=LogNorm(0.125, 8.0), cmap="RdBu_r")
                ax.set_title(f"{label} / {sets[0][0]}\nenergy ratio", fontsize=7)
            for column in range(columns):
                ax = axes[row_index, column]
                for y0, y1 in ((18.7, 58.5), (165.0, 225.0)):
                    ax.plot([0, 72, 72, 0], [y0, y0, y1, y1], color="c", linestyle=":", linewidth=0.8)
                ax.set_aspect("equal")
                ax.set_xlim(0, 144)
                ax.set_ylim(-64, 426.5)
                ax.tick_params(labelsize=6)
        fig.colorbar(image, ax=axes[:, len(sets) - 1].tolist(), shrink=0.5, label="m/s")
        if ratio_image is not None:
            bar = fig.colorbar(ratio_image, ax=axes[:, -1].tolist(), shrink=0.5, label="energy ratio to the first column")
            bar.set_ticks([0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
            bar.set_ticklabels(["1/8", "1/4", "1/2", "1", "2", "4", "8"])
        fig.suptitle(f"non-axisymmetric energy above and below the kernel scale (support {support * 1e3:.0f} mm); "
                     "Fluent resampled to the SPH points", fontsize=8)
        fig.savefig(arguments.figure, dpi=140)
        print(f"wrote {arguments.figure}")


if __name__ == "__main__":
    main()
