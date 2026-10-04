"""
_plot_energy_maps.py — where the kinetic energy of the 30 L tank sits, split as in _analyze_axisymmetric_energy.py,
Fluent snapshots and SPH dumps side by side (2026-10-04, base Anaconda).

The fluid (all of it, rotor boxes included) is binned on an r-y grid of rings (--bin, default 8 mm). In every ring
and every snapshot the azimuthal mean <u> and <u^2> of u_r, u_theta, u_y are taken (cells or particles weighted
equally), and three energy densities, J/m^3:

    swirl       e_s = 1/2 rho <u_theta>^2
    meridional  e_m = 1/2 rho (<u_r>^2 + <u_y>^2)
    rest        e_n = 1/2 rho sum_k (<u_k^2> - <u_k>^2)      (everything that varies with azimuth: turbulence,
                                                              baffle wakes, blade-periodic flow, particle noise)

Each density is averaged over the snapshots of a data set (a ring counts in a snapshot with at least
--min-count samples). The figure shows them as velocities sqrt(2 e / rho), m/s, one column per data set, and
one more column per SPH data set with the energy ratio to the first data set (Fluent) on a log scale. A table
integrates the three energies over regions (r bands x height bands) with the ring volumes of a reference SPH
dump (--volumes), the same volumes for every data set.

Fluent y is shifted by -58.5 mm into the generator frame; the velocities of Fluent's rotating zones are turned
by 12 degrees in their sense of rotation (--fluent-rotor-lag, see _analyze_rushton_lower_flow.py).

usage:
    python experiment/v1/checks/_plot_energy_maps.py OUT.png --volumes REF_final.npz 0.003 \
        --fluent fine25.ip [--fluent ...] --sph-set LABEL A_t14.npz A_t16.npz ... [--sph-set ...]
"""
import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _analyze_axisymmetric_energy as axi   # noqa: E402
import _analyze_rushton_lower_flow as lower  # noqa: E402

DENSITY = 998.0
R_BANDS = ((0.0, 0.072, "r < 72"), (0.072, 0.110, "r 72..110"), (0.110, 0.144, "r 110..144"))
Y_BANDS = ((-0.070, 0.0187, "y < 18.7 (floor)"), (0.0187, 0.0585, "Rushton height"), (0.0585, 0.165, "between"),
           (0.165, 0.225, "PBT height"), (0.225, 0.430, "y > 225 (top)"))


def fluent_arrays(path, rotor_lag_degrees):
    cells = axi.read_interpolation_file(path)
    x = np.stack([cells["x"], cells["y"] - axi.FLUENT_Y_SHIFT, cells["z"]], axis=1)
    v = np.stack([cells["x-velocity"], cells["y-velocity"], cells["z-velocity"]], axis=1)
    rotor = lower.in_fluent_rotor_zones(x)
    radius = np.maximum(np.hypot(x[rotor, 0], x[rotor, 2]), 1e-12)
    sense = np.sign(((v[rotor, 0] * x[rotor, 2] - v[rotor, 2] * x[rotor, 0]) / radius).sum()) or 1.0
    v[rotor] = lower.turn_about_y(v[rotor], sense * np.radians(rotor_lag_degrees))
    return x, v


def densities(snapshots, min_count):
    """snapshot-averaged energy densities (swirl, meridional, rest) per ring, NaN where no snapshot counts"""
    total = np.zeros((3, (len(axi.R_EDGES) - 1) * (len(axi.Y_EDGES) - 1)))
    used = np.zeros(total.shape[1])
    for x, v in snapshots:
        count, mean, mean_square = axi.bin_moments(x, v)
        ok = count >= min_count
        e = 0.5 * DENSITY * np.stack([mean[:, 1] ** 2, mean[:, 0] ** 2 + mean[:, 2] ** 2,
                                      np.clip(mean_square - mean ** 2, 0.0, None).sum(axis=1)])
        total[:, ok] += e[:, ok]
        used += ok
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(used > 0, total / used, np.nan)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out")
    parser.add_argument("--volumes", nargs=2, required=True, metavar=("REF_DUMP", "DX"))
    parser.add_argument("--fluent", action="append", default=[])
    parser.add_argument("--fluent-label", default=None)
    parser.add_argument("--fluent-rotor-lag", type=float, default=12.0)
    parser.add_argument("--sph-set", nargs="+", action="append", default=[], metavar="LABEL DUMP")
    parser.add_argument("--bin", type=float, default=0.008)
    parser.add_argument("--min-count", type=int, default=3)
    arguments = parser.parse_args()

    axi.BIN = arguments.bin
    axi.R_EDGES = np.arange(0.0, 0.144 + axi.BIN, axi.BIN)
    axi.Y_EDGES = np.arange(-0.064, 0.4265 + axi.BIN, axi.BIN)
    shape = (len(axi.R_EDGES) - 1, len(axi.Y_EDGES) - 1)
    reference_x, _ = axi.sph_arrays(arguments.volumes[0])
    reference_count, _, _ = axi.bin_moments(reference_x, np.zeros_like(reference_x))
    volume = reference_count * float(arguments.volumes[1]) ** 3
    fluid = reference_count > 0

    sets = []
    if arguments.fluent:
        label = arguments.fluent_label or f"Fluent, {len(arguments.fluent)} snapshots"
        sets.append((label, densities((fluent_arrays(p, arguments.fluent_rotor_lag) for p in arguments.fluent),
                                      arguments.min_count)))
    for entry in arguments.sph_set:
        label, paths = entry[0], entry[1:]
        sets.append((f"{label} ({len(paths)})", densities((axi.sph_arrays(p) for p in paths), arguments.min_count)))
    for label, e in sets:
        e[:, ~fluid] = np.nan

    # regional integrals with the reference ring volumes (rings without data count as zero)
    r_mid = 0.5 * (axi.R_EDGES[1:] + axi.R_EDGES[:-1])
    y_mid = 0.5 * (axi.Y_EDGES[1:] + axi.Y_EDGES[:-1])
    r_grid, y_grid = np.meshgrid(r_mid, y_mid, indexing="ij")
    r_grid, y_grid = r_grid.ravel(), y_grid.ravel()
    names = ("swirl", "meridional", "rest")
    print(f"energy by region, mJ (rings {arguments.bin * 1e3:g} mm, volumes of {pathlib.Path(arguments.volumes[0]).name})")
    print(f"  {'region':34s} " + "".join(f"{label[:24]:>26s}" for label, _ in sets))
    print(f"  {'':34s} " + "".join(f"{'swirl  merid   rest':>26s}" for _ in sets))
    for y0, y1, y_name in Y_BANDS:
        for r0, r1, r_name in R_BANDS:
            select = (y_grid >= y0) & (y_grid < y1) & (r_grid >= r0) & (r_grid < r1) & fluid
            if not select.any():
                continue
            cells = []
            for _, e in sets:
                values = [np.nansum(e[k, select] * volume[select]) * 1e3 for k in range(3)]
                cells.append(f"{values[0]:8.1f}{values[1]:8.1f}{values[2]:8.1f}  ")
            print(f"  {y_name + ', ' + r_name:34s} " + "".join(f"{c:>26s}" for c in cells))
    totals = []
    for label, e in sets:
        totals.append([np.nansum(e[k] * volume) * 1e3 for k in range(3)])
    print(f"  {'whole tank':34s} " + "".join(f"{t[0]:8.1f}{t[1]:8.1f}{t[2]:8.1f}  ".rjust(26) for t in totals))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    columns = len(sets) + max(len(sets) - 1, 0)
    fig, axes = plt.subplots(3, columns, figsize=(2.6 * columns + 1.5, 13.0), squeeze=False, layout="constrained")
    titles = ("swirl  |<u_θ>|", "meridional  |(<u_r>, <u_y>)|", "rest  sqrt(Σ var u_k)")
    vmax = (0.25, 0.25, 0.25)
    speed_image, ratio_image = None, None
    for row in range(3):
        for column, (label, e) in enumerate(sets):
            ax = axes[row, column]
            speed = np.sqrt(2.0 * e[row] / DENSITY).reshape(shape)
            speed_image = ax.pcolormesh(axi.R_EDGES * 1e3, axi.Y_EDGES * 1e3, speed.T, vmin=0.0, vmax=vmax[row],
                                        cmap="magma")
            ax.set_title(f"{label}\n{titles[row]}, m/s", fontsize=7)
        for offset, (label, e) in enumerate(sets[1:]):
            ax = axes[row, len(sets) + offset]
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio = (e[row] / sets[0][1][row]).reshape(shape)
            ratio_image = ax.pcolormesh(axi.R_EDGES * 1e3, axi.Y_EDGES * 1e3, np.clip(ratio, 0.125, 8.0).T,
                                        norm=LogNorm(0.125, 8.0), cmap="RdBu_r")
            ax.set_title(f"{label} / {sets[0][0]}\n{titles[row].split()[0]} energy ratio", fontsize=7)
        for column in range(columns):
            ax = axes[row, column]
            for y0, y1 in ((18.7, 58.5), (165.0, 225.0)):
                ax.plot([0, 72, 72, 0], [y0, y0, y1, y1], color="c", linestyle=":", linewidth=0.8)
            ax.axvline(114.35, color="w", linewidth=0.5, alpha=0.5)
            ax.set_aspect("equal")
            ax.set_xlim(0, 144)
            ax.set_ylim(-64, 426.5)
            ax.tick_params(labelsize=6)
            if row == 2:
                ax.set_xlabel("r, mm", fontsize=7)
            if column == 0:
                ax.set_ylabel("y, mm", fontsize=7)
    fig.colorbar(speed_image, ax=axes[:, len(sets) - 1].tolist(), shrink=0.4, label="m/s")
    if ratio_image is not None:
        bar = fig.colorbar(ratio_image, ax=axes[:, -1].tolist(), shrink=0.4,
                           label="energy ratio (red: more than the first column)")
        bar.set_ticks([0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
        bar.set_ticklabels(["1/8", "1/4", "1/2", "1", "2", "4", "8"])
    fig.suptitle("kinetic energy of the azimuthal-mean swirl, the azimuthal-mean meridional flow and the rest; "
                 "dotted: Fluent's rotor boxes; thin line: inner edge of the baffles", fontsize=8)
    fig.savefig(arguments.out, dpi=140)
    print(f"wrote {arguments.out}")


if __name__ == "__main__":
    main()
