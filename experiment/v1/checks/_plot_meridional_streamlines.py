"""
_plot_meridional_streamlines.py — streamlines of the azimuthal-mean flow of the 30 L tank, Fluent snapshots and
SPH dumps side by side (2026-10-03, base Anaconda).

The fluid is binned on an r-y grid (--bin, default 6 mm) and the azimuthal mean of u_r, u_theta, u_y is taken
per bin (cells or particles weighted equally; Fluent's interpolation files carry no volumes). Bins without
data inside the fluid are filled from the nearest bin with data; bins that hold no fluid in the reference SPH
dump (solids, outside the tank) are masked. Shown per data set:
  - colour: |u_theta|, the azimuthal-mean swirl;
  - black lines: contours of the Stokes stream function psi(r, y) = int_0^r r' u_y(r', y) dr'. They are the
    streamlines of the azimuthal-mean meridional flow; between two neighbouring lines flows the same volume
    flow 2 pi d(psi), given in the title as a fraction of N D^3 (N = 200 rpm, D = 96 mm);
  - white arrows: direction of (u_r, u_y).
Fluent y is shifted by -58.5 mm into the generator frame; the mirrored fine snapshots are oriented so that the
swirl is positive. psi > 0 (upward flow inside r) is a clockwise cell in this view (r to the right, y up); the
title said the opposite until 2026-10-03. The velocities of Fluent's rotating zones are turned by 12 degrees in
their sense of rotation (--fluent-rotor-lag): the written cells sit one time step ahead of their velocities
(_analyze_rushton_lower_flow.py). psi uses u_y only and is not affected.

usage:
    python experiment/v1/checks/_plot_meridional_streamlines.py OUT.png --volumes REF_final.npz 0.003 \
        --fluent fine25.ip [--fluent ...] --sph LABEL DUMP.npz [--sph LABEL DUMP.npz ...]
"""
import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _analyze_axisymmetric_energy as axi   # noqa: E402
import _analyze_rushton_lower_flow as lower   # noqa: E402

PUMPING_SCALE = (200.0 / 60.0) * 0.096 ** 3


def field(datasets, kind, rotor_lag_degrees=12.0):
    """count-weighted azimuthal mean over several snapshots of one kind"""
    total_count, total_sum = 0.0, 0.0
    for path in datasets:
        if kind == "fluent":
            cells = axi.read_interpolation_file(path)
            x = np.stack([cells["x"], cells["y"] - axi.FLUENT_Y_SHIFT, cells["z"]], axis=1)
            v = np.stack([cells["x-velocity"], cells["y-velocity"], cells["z-velocity"]], axis=1)
            rotor = lower.in_fluent_rotor_zones(x)
            radius = np.maximum(np.hypot(x[rotor, 0], x[rotor, 2]), 1e-12)
            sense = np.sign(((v[rotor, 0] * x[rotor, 2] - v[rotor, 2] * x[rotor, 0]) / radius).sum()) or 1.0
            v[rotor] = lower.turn_about_y(v[rotor], sense * np.radians(rotor_lag_degrees))
        else:
            x, v = axi.sph_arrays(path)
        count, mean, _ = axi.bin_moments(x, v)
        mean = mean * [1.0, np.sign(mean[:, 1].sum() or 1.0), 1.0]
        total_count = total_count + count
        total_sum = total_sum + count[:, None] * mean
    return total_count, total_sum / np.maximum(total_count, 1.0)[:, None]


def fill_and_mask(count, mean, fluid_bins):
    shape = (len(axi.R_EDGES) - 1, len(axi.Y_EDGES) - 1)
    have = (count > 0).reshape(shape)
    fluid = fluid_bins.reshape(shape)
    components = [mean[:, k].reshape(shape).copy() for k in range(3)]
    missing = fluid & ~have
    if missing.any():
        known = np.argwhere(have)
        for i, j in np.argwhere(missing):
            nearest = known[np.argmin((known[:, 0] - i) ** 2 + (known[:, 1] - j) ** 2)]
            for c in components:
                c[i, j] = c[nearest[0], nearest[1]]
    for c in components:
        c[~fluid] = np.nan
    return components


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out")
    parser.add_argument("--volumes", nargs=2, required=True, metavar=("REF_DUMP", "DX"))
    parser.add_argument("--fluent", action="append", default=[])
    parser.add_argument("--fluent-label", default=None)
    parser.add_argument("--fluent-rotor-lag", type=float, default=12.0)
    parser.add_argument("--sph", nargs=2, action="append", default=[], metavar=("LABEL", "DUMP"))
    parser.add_argument("--bin", type=float, default=0.006)
    parser.add_argument("--levels", type=float, default=0.1, help="flow between two stream lines, in N D^3")
    arguments = parser.parse_args()

    axi.BIN = arguments.bin
    axi.R_EDGES = np.arange(0.0, 0.144 + axi.BIN, axi.BIN)
    axi.Y_EDGES = np.arange(-0.060, 0.4265 + axi.BIN, axi.BIN)
    reference_x, _ = axi.sph_arrays(arguments.volumes[0])
    reference_count, _, _ = axi.bin_moments(reference_x, np.zeros_like(reference_x))
    fluid_bins = reference_count > 0

    panels = []
    if arguments.fluent:
        label = arguments.fluent_label or f"Fluent, {len(arguments.fluent)} snapshots"
        panels.append((label, *field(arguments.fluent, "fluent", arguments.fluent_rotor_lag)))
    for label, path in arguments.sph:
        panels.append((label, *field([path], "sph")))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r_mid = 0.5 * (axi.R_EDGES[1:] + axi.R_EDGES[:-1])
    y_mid = 0.5 * (axi.Y_EDGES[1:] + axi.Y_EDGES[:-1])
    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels), 8.6), squeeze=False)
    levels = None
    for column, (label, count, mean) in enumerate(panels):
        u_r, u_t, u_y = fill_and_mask(count, mean, fluid_bins)
        # Stokes stream function from the axis outward; solid bins carry no flow
        integrand = np.nan_to_num(u_y) * r_mid[:, None] * axi.BIN
        psi = np.cumsum(integrand, axis=0) - 0.5 * integrand
        psi_scaled = 2.0 * np.pi * psi / PUMPING_SCALE          # volume flow below r, in N D^3
        psi_scaled[np.isnan(u_y)] = np.nan
        if levels is None:
            span = np.nanmax(np.abs(psi_scaled))
            levels = np.arange(-np.ceil(span / arguments.levels), np.ceil(span / arguments.levels) + 1) * arguments.levels
            levels = levels[levels != 0.0]
        ax = axes[0, column]
        image = ax.pcolormesh(axi.R_EDGES * 1e3, axi.Y_EDGES * 1e3, np.abs(u_t).T, vmin=0, vmax=0.3, cmap="magma")
        ax.contour(r_mid * 1e3, y_mid * 1e3, psi_scaled.T, levels=levels, colors="k", linewidths=0.8,
                   linestyles=np.where(levels > 0, "solid", "dashed"))
        step = 2
        grid_r, grid_y = np.meshgrid(r_mid[::step] * 1e3, y_mid[::step] * 1e3, indexing="ij")
        ax.quiver(grid_r, grid_y, u_r[::step, ::step], u_y[::step, ::step], color="w", scale=4.0, width=0.004, alpha=0.8)
        for y0, y1 in ((18.7, 58.5), (165.0, 225.0)):
            ax.plot([0, 72, 72, 0], [y0, y0, y1, y1], color="c", linestyle=":", linewidth=0.8)
        ax.axvspan(114.35, 138.36, color="w", alpha=0.08)
        ax.set_aspect("equal")
        ax.set_xlim(0, 144)
        ax.set_ylim(-60, 426.5)
        ax.set_xlabel("r, mm")
        ax.set_ylabel("y, mm")
        circulation = np.nanmax(psi_scaled) - np.nanmin(psi_scaled)
        ax.set_title(f"{label}\nloops span {np.nanmin(psi_scaled):+.2f} .. {np.nanmax(psi_scaled):+.2f} N D³", fontsize=8)
    fig.colorbar(image, ax=axes[0, :].tolist(), shrink=0.5, label="|u_θ| azimuthal mean, m/s")
    fig.suptitle(f"azimuthal-mean meridional streamlines (solid: clockwise in this view, dashed: anticlockwise); "
                 f"{arguments.levels:g} N D³ between lines; dotted: Fluent's rotor boxes; light band: baffles", fontsize=8)
    fig.savefig(arguments.out, dpi=140)
    print(f"wrote {arguments.out}")


if __name__ == "__main__":
    main()
