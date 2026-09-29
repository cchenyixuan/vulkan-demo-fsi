"""_plot_vorticity_slices.py — vorticity images from the slab snapshots of
_check_blade_leak_tracking.py run --slice-dir (2026-09-29).

One image per snapshot and per slab:

  vertical slab through the rotor axis (plane x-y, |z| < half width)
      colour = omega_z, the component normal to the plane, as force.comp computes it per particle
      with the KCG-corrected kernel gradient (omega_z = dv/dx - du/dy), averaged per cell
  horizontal slabs (plane x-z at the heights stored in the snapshot)
      colour = omega_y, the component along the rotor axis, omega_y = du/dz - dw/dx, from central
      differences of the cell-averaged velocity (the solver does not store this component)

Cells without particles (solids) are left blank. The colour range is fixed (--limit, 1/s), the same
in every image, so the images can be joined into a film.

Usage:
    python experiment/v1/checks/_plot_vorticity_slices.py SLICE_DIR OUT_DIR [--limit 60] [--cell 1.0]
        [--first 0] [--last 10**9] [--film]
--cell: cell size in particle spacings. --film: also write OUT_DIR/<slab>.gif (10 frames per second).
"""
import argparse
import pathlib
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TANK_RADIUS = 0.144
LIQUID_HEIGHT = 0.4265
FLOOR_BOTTOM = -0.062


def cell_mean(a, b, values, a_edges, b_edges):
    count = np.histogram2d(a, b, bins=(a_edges, b_edges))[0]
    result = []
    for value in values:
        total = np.histogram2d(a, b, bins=(a_edges, b_edges), weights=value)[0]
        with np.errstate(invalid="ignore", divide="ignore"):
            result.append(np.where(count > 0, total / count, np.nan))
    return result


def draw(field, a_edges, b_edges, limit, title, labels, path, figure_size):
    figure, axis = plt.subplots(figsize=figure_size)
    image = axis.pcolormesh(a_edges * 1e3, b_edges * 1e3, field.T, cmap="RdBu_r", vmin=-limit, vmax=limit)
    axis.set_aspect("equal")
    axis.set_xlabel(labels[0])
    axis.set_ylabel(labels[1])
    axis.set_title(title)
    figure.colorbar(image, ax=axis, shrink=0.85, label=labels[2])
    figure.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("slice_dir")
    parser.add_argument("out_dir")
    parser.add_argument("--limit", type=float, default=60.0, help="colour range, 1/s")
    parser.add_argument("--cell", type=float, default=1.0, help="cell size in particle spacings")
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--last", type=int, default=10 ** 9)
    parser.add_argument("--film", action="store_true")
    arguments = parser.parse_args()

    out = pathlib.Path(arguments.out_dir)
    files = sorted(pathlib.Path(arguments.slice_dir).glob("slice_*.npz"))
    files = [f for f in files if arguments.first <= int(f.stem.split("_")[1]) <= arguments.last]
    if not files:
        print("no snapshots in", arguments.slice_dir)
        return 1
    written = {}
    for file in files:
        index = int(file.stem.split("_")[1])
        data = np.load(file)
        x = data["position"].astype(np.float64)
        v = data["velocity"].astype(np.float64)
        omega_z = data["vorticity_z"].astype(np.float64)
        spacing, half_width = float(data["spacing"]), float(data["half_width"])
        time = float(data["time"])
        cell = arguments.cell * spacing
        x_edges = np.arange(-TANK_RADIUS - cell, TANK_RADIUS + 1.001 * cell, cell)
        y_edges = np.arange(FLOOR_BOTTOM - cell, LIQUID_HEIGHT + 1.001 * cell, cell)

        slab = np.abs(x[:, 2]) < half_width
        (field,) = cell_mean(x[slab, 0], x[slab, 1], [omega_z[slab]], x_edges, y_edges)
        directory = out / "vertical"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"vorticity_{index:04d}.png"
        draw(field, x_edges, y_edges, arguments.limit, f"vertical slab z = 0, t = {time:6.2f} s",
             ("x (mm)", "y (mm)", "omega_z (1/s)"), path, (6.0, 8.5))
        written.setdefault("vertical", []).append(path)

        for height in data["heights"]:
            slab = np.abs(x[:, 1] - height) < half_width
            u, w = cell_mean(x[slab, 0], x[slab, 2], [v[slab, 0], v[slab, 2]], x_edges, x_edges)
            # omega_y = du/dz - dw/dx, central differences on the cell centres
            du_dz = np.full_like(u, np.nan)
            dw_dx = np.full_like(w, np.nan)
            du_dz[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2.0 * cell)
            dw_dx[1:-1, :] = (w[2:, :] - w[:-2, :]) / (2.0 * cell)
            name = f"horizontal_y{height * 1e3:03.0f}mm"
            directory = out / name
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"vorticity_{index:04d}.png"
            draw(du_dz - dw_dx, x_edges, x_edges, arguments.limit,
                 f"horizontal slab y = {height * 1e3:.0f} mm, t = {time:6.2f} s",
                 ("x (mm)", "z (mm)", "omega_y (1/s)"), path, (7.0, 6.0))
            written.setdefault(name, []).append(path)
        print(f"snapshot {index:4d}, t = {time:.3f} s")

    if arguments.film:
        from PIL import Image
        for name, paths in written.items():
            frames = [Image.open(path).convert("P", palette=Image.ADAPTIVE) for path in paths]
            target = out / f"{name}.gif"
            frames[0].save(target, save_all=True, append_images=frames[1:], duration=100, loop=0)
            print("wrote", target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
