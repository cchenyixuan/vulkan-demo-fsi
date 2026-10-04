"""
_plot_pbt_blade_frame.py — the flow around the PBT blades in the blade frame, phase-averaged, Fluent and SPH
(2026-10-04, base Anaconda).

log/2026-10-04_fluctuation-scales-and-pbt-bands.md: our PBT takes too much torque at the blade tips because the
fluid there turns less with the blades (u_theta / (omega r) 0.36 against 0.53) and passes through; Fluent's tip
region turns with the blades and passes almost nothing. Where in the passage does that happen?

Blade angle per snapshot: 6-fold circular mean of the blade azimuths phi = atan2(-z, x) (phi grows along the
motion), Fluent from the faces of the wall zone pbt with |n . e_theta| > 0.3 (both blade sides) weighted by
area, SPH from the rotor particles of the PBT blades (r 13..47 mm, y 186..203 mm). psi = phi - blade angle,
folded into one blade pitch [-30, 30) deg; psi > 0 is ahead of the blade centre. Unwrapped cylinder surfaces:
arc length s = r_mid psi (mm) and y, for radial bands (--bands). Per bin (--ds, --dy) the mean of u_theta,
u_y, u_r over all snapshots of a data set (cells / particles weighted equally). Fluent's rotor-zone velocities
are turned by 12 degrees (_analyze_rushton_lower_flow.py); mirrored snapshots are mirrored back (z -> -z).

The blade is drawn at the band's mid radius: its mid-plane meets the cylinder r in the line
y - y_c = s (the blades are pitched 45 deg, leading edge high, chord 24.8 mm), shown over the chord.

Figure: one row per band, one column per data set; colour u_theta / (omega r) (or u_r, u_y with --colour),
arrows the velocity relative to the blade (u_theta - omega r, u_y).

--meridional OUT2.png (2026-10-04): the azimuthal mean around the PBT in the r-y plane (bins --dr, all phases),
colour u_theta / (omega r), arrows (u_r, u_y), and the circulation of the mean (u_r, u_y) around the rectangle
r 40..56 mm, y 182..204 mm (the ring vortex at the blade tips; positive = up at the outer side, in at the top).

usage:
    python experiment/v1/checks/_plot_pbt_blade_frame.py OUT.png --fluent fine25.ip fine25_pbt.csv [--fluent ...] \
        --sph-set LABEL A.npz [B.npz ...] [--sph-set ...] [--bands 0.040 0.044 0.048 0.052] [--colour u_r]
"""
import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _analyze_axisymmetric_energy as axi   # noqa: E402
import _analyze_rushton_lower_flow as lower  # noqa: E402

OMEGA = 2.0 * np.pi * 200.0 / 60.0
PITCH = np.pi / 3.0
BLADE_CENTRE_Y = 0.19465
HALF_CHORD = 0.0124
Y_RANGE = (0.150, 0.230)


def fluent_blade_angle(path, mirror):
    faces = lower.read_csv_columns(path)
    x, y, z = faces["x-coordinate"], faces["y-coordinate"] - axi.FLUENT_Y_SHIFT, faces["z-coordinate"]
    area = np.stack([faces["x-face-area"], faces["y-face-area"], faces["z-face-area"]], axis=1)
    if mirror:
        z = -z
        area[:, 2] *= -1.0
    r = np.maximum(np.hypot(x, z), 1e-12)
    size = np.linalg.norm(area, axis=1)
    n_theta = (area[:, 0] * z - area[:, 2] * x) / (r * np.maximum(size, 1e-30))
    blade = (np.abs(n_theta) > 0.3) & (r > 0.013) & (r < 0.047) & (y > 0.186) & (y < 0.203)
    phi = np.arctan2(-z[blade], x[blade])
    return np.angle((size[blade] * np.exp(6j * phi)).sum()) / 6.0


def fluent_samples(pairs, rotor_lag_degrees):
    for ip_path, pbt_path in pairs:
        cells = axi.read_interpolation_file(ip_path)
        x = np.stack([cells["x"], cells["y"] - axi.FLUENT_Y_SHIFT, cells["z"]], axis=1)
        v = np.stack([cells["x-velocity"], cells["y-velocity"], cells["z-velocity"]], axis=1)
        rotor = lower.in_fluent_rotor_zones(x)
        radius = np.maximum(np.hypot(x[rotor, 0], x[rotor, 2]), 1e-12)
        sense = np.sign(((v[rotor, 0] * x[rotor, 2] - v[rotor, 2] * x[rotor, 0]) / radius).sum()) or 1.0
        mirror = sense < 0
        if mirror:
            x[:, 2] *= -1.0
            v[:, 2] *= -1.0
        v[rotor] = lower.turn_about_y(v[rotor], np.radians(rotor_lag_degrees))
        angle = fluent_blade_angle(pbt_path, mirror)
        print(f"  {pathlib.Path(ip_path).name}: mirrored {mirror}, PBT blade at {np.degrees(angle):+.2f} deg", flush=True)
        yield x, v, angle


def sph_samples(paths):
    for path in paths:
        dump = np.load(path)
        position, material = dump["positions"], dump["material"]
        alive = position[:, 3] > 0
        fluid = alive & (material == 0)
        solid = position[alive & (material != 0), :3].astype(np.float64)
        r, y = np.hypot(solid[:, 0], solid[:, 2]), solid[:, 1]
        blade = (r > 0.013) & (r < 0.047) & (y > 0.186) & (y < 0.203)
        phi = np.arctan2(-solid[blade, 2], solid[blade, 0])
        angle = np.angle(np.exp(6j * phi).sum()) / 6.0
        print(f"  {pathlib.Path(path).name}: {int(blade.sum())} PBT blade particles, blade at {np.degrees(angle):+.2f} deg",
              flush=True)
        yield position[fluid, :3].astype(np.float64), dump["velocity_mass"][fluid, :3].astype(np.float64), angle


def accumulate(samples, bands, s_edges, y_edges):
    shape = (len(bands) - 1, len(s_edges) - 1, len(y_edges) - 1)
    count, sums = np.zeros(shape), np.zeros((3,) + shape)
    snapshots = 0
    for x, v, angle in samples:
        snapshots += 1
        r = np.maximum(np.hypot(x[:, 0], x[:, 2]), 1e-12)
        keep = (r >= bands[0]) & (r < bands[-1]) & (x[:, 1] >= y_edges[0]) & (x[:, 1] < y_edges[-1])
        x, v, r = x[keep], v[keep], r[keep]
        phi = np.arctan2(-x[:, 2], x[:, 0])
        psi = np.mod(phi - angle + PITCH / 2.0, PITCH) - PITCH / 2.0
        band = np.digitize(r, bands) - 1
        r_mid = 0.5 * (bands[band] + bands[band + 1])
        s = r_mid * psi
        i_s = np.clip(np.digitize(s, s_edges) - 1, 0, len(s_edges) - 2)
        i_y = np.clip(np.digitize(x[:, 1], y_edges) - 1, 0, len(y_edges) - 2)
        flat = np.ravel_multi_index((band, i_s, i_y), shape)
        u_theta = (v[:, 0] * x[:, 2] - v[:, 2] * x[:, 0]) / r
        u_r = (v[:, 0] * x[:, 0] + v[:, 2] * x[:, 2]) / r
        size = int(np.prod(shape))
        count += np.bincount(flat, minlength=size).reshape(shape)
        for k, value in enumerate((u_theta, v[:, 1], u_r)):
            sums[k] += np.bincount(flat, weights=value, minlength=size).reshape(shape)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = sums / count
    means[:, count < 3] = np.nan
    return means, count, snapshots


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out")
    parser.add_argument("--fluent", nargs=2, action="append", default=[], metavar=("CELLS_IP", "PBT_CSV"))
    parser.add_argument("--fluent-rotor-lag", type=float, default=12.0)
    parser.add_argument("--sph-set", nargs="+", action="append", default=[], metavar="LABEL DUMP")
    parser.add_argument("--bands", type=float, nargs="+", default=[0.036, 0.040, 0.044, 0.048, 0.052])
    parser.add_argument("--ds", type=float, default=0.003)
    parser.add_argument("--dy", type=float, default=0.003)
    parser.add_argument("--colour", choices=("co-rotation", "u_r", "u_y"), default="co-rotation")
    parser.add_argument("--meridional", default=None)
    parser.add_argument("--dr", type=float, default=0.002)
    arguments = parser.parse_args()

    bands = np.asarray(arguments.bands)
    half_pitch_arc = bands[-1] * PITCH / 2.0
    s_edges = np.arange(-half_pitch_arc, half_pitch_arc + arguments.ds, arguments.ds)
    y_edges = np.arange(Y_RANGE[0], Y_RANGE[1] + 1e-9, arguments.dy)
    sets = []
    if arguments.fluent:
        sets.append((f"Fluent ({len(arguments.fluent)})",) +
                    accumulate(fluent_samples(arguments.fluent, arguments.fluent_rotor_lag), bands, s_edges, y_edges))
    for entry in arguments.sph_set:
        label, paths = entry[0], entry[1:]
        sets.append((f"{label} ({len(paths)})",) + accumulate(sph_samples(paths), bands, s_edges, y_edges))

    # numbers: per band, near the blade (|s| < 12 mm, y within 6 mm of the blade line) and in the rest of the passage
    s_mid = 0.5 * (s_edges[1:] + s_edges[:-1])
    y_mid = 0.5 * (y_edges[1:] + y_edges[:-1])
    grid_s, grid_y = np.meshgrid(s_mid, y_mid, indexing="ij")
    on_blade = (np.abs(grid_s) < HALF_CHORD * np.cos(np.pi / 4)) & (np.abs(grid_y - BLADE_CENTRE_Y - grid_s) < 0.006)
    print("\nmean u_theta / (omega r), u_y, u_r in the blade height (y 186..203 mm), whole pitch / within 6 mm of the blade")
    for b in range(len(bands) - 1):
        r_mid = 0.5 * (bands[b] + bands[b + 1])
        zone = (grid_y > 0.186) & (grid_y < 0.203)
        cells = []
        for label, means, count, _ in sets:
            values = []
            for mask in (zone, zone & on_blade):
                w = count[b] * mask
                total = max(w.sum(), 1.0)
                values.append([np.nansum(means[k, b] * w) / total for k in range(3)])
            cells.append(f"{values[0][0] / (OMEGA * r_mid):6.3f}{values[0][1]:7.3f}{values[0][2]:7.3f} | "
                         f"{values[1][0] / (OMEGA * r_mid):6.3f}{values[1][1]:7.3f}{values[1][2]:7.3f}")
        print(f"  r {bands[b] * 1e3:4.1f}..{bands[b + 1] * 1e3:4.1f} mm  " + "   ".join(
            f"{label}: {c}" for (label, _, _, _), c in zip(sets, cells)))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows, columns = len(bands) - 1, len(sets)
    fig, axes = plt.subplots(rows, columns, figsize=(3.4 * columns + 1.2, 2.9 * rows), squeeze=False, layout="constrained")
    image = None
    for b in range(rows):
        r_mid = 0.5 * (bands[b] + bands[b + 1])
        for c, (label, means, count, snapshots) in enumerate(sets):
            ax = axes[b, c]
            u_theta, u_y, u_r = means[0, b], means[1, b], means[2, b]
            if arguments.colour == "co-rotation":
                field, limits, cmap = u_theta / (OMEGA * r_mid), (0.0, 1.0), "viridis"
            elif arguments.colour == "u_r":
                field, limits, cmap = u_r, (-0.3, 0.3), "RdBu_r"
            else:
                field, limits, cmap = u_y, (-0.5, 0.5), "RdBu_r"
            image = ax.pcolormesh(s_edges * 1e3, y_edges * 1e3, field.T, vmin=limits[0], vmax=limits[1], cmap=cmap)
            step = 2
            gs, gy = np.meshgrid(s_mid[::step] * 1e3, y_mid[::step] * 1e3, indexing="ij")
            ax.quiver(gs, gy, (u_theta - OMEGA * r_mid)[::step, ::step], u_y[::step, ::step], color="w",
                      scale=8.0, width=0.004, alpha=0.9)
            chord = np.linspace(-HALF_CHORD, HALF_CHORD, 2) * np.cos(np.pi / 4)
            for k in (-1, 0, 1):
                ax.plot((chord + k * r_mid * PITCH) * 1e3, (BLADE_CENTRE_Y + chord) * 1e3, color="r", linewidth=2.0)
            ax.set_xlim(s_edges[0] * 1e3, s_edges[-1] * 1e3)
            ax.set_ylim(y_edges[0] * 1e3, y_edges[-1] * 1e3)
            ax.set_aspect("equal")
            ax.tick_params(labelsize=6)
            ax.set_title(f"{label}, r {bands[b] * 1e3:.0f}..{bands[b + 1] * 1e3:.0f} mm", fontsize=7)
            if b == rows - 1:
                ax.set_xlabel("arc length from the blade centre, mm (motion to the right)", fontsize=6)
            if c == 0:
                ax.set_ylabel("y, mm", fontsize=7)
    label = {"co-rotation": "u_θ / (ω r)", "u_r": "u_r, m/s", "u_y": "u_y, m/s"}[arguments.colour]
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.4, label=label)
    fig.suptitle("PBT blade frame, phase-averaged; red: blade mid-plane; arrows: velocity relative to the blade "
                 "(u_θ − ωr, u_y)", fontsize=8)
    fig.savefig(arguments.out, dpi=130)
    print(f"wrote {arguments.out}")
    if arguments.meridional:
        meridional(arguments, bands)


def meridional(arguments, bands):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r_edges = np.arange(0.010, 0.070 + 1e-9, arguments.dr)
    y_edges = np.arange(0.160, 0.226 + 1e-9, arguments.dr)
    shape = (len(r_edges) - 1, len(y_edges) - 1)
    sources = []
    if arguments.fluent:
        sources.append((f"Fluent ({len(arguments.fluent)})", lambda: fluent_samples(arguments.fluent, arguments.fluent_rotor_lag)))
    for entry in arguments.sph_set:
        sources.append((f"{entry[0]} ({len(entry) - 1})", lambda paths=entry[1:]: sph_samples(paths)))
    panels = []
    for label, make in sources:
        count, sums = np.zeros(shape), np.zeros((3,) + shape)
        for x, v, _ in make():
            r = np.maximum(np.hypot(x[:, 0], x[:, 2]), 1e-12)
            keep = (r >= r_edges[0]) & (r < r_edges[-1]) & (x[:, 1] >= y_edges[0]) & (x[:, 1] < y_edges[-1])
            i_r = np.digitize(r[keep], r_edges) - 1
            i_y = np.digitize(x[keep, 1], y_edges) - 1
            flat = np.ravel_multi_index((i_r, i_y), shape)
            xk, vk, rk = x[keep], v[keep], r[keep]
            values = ((vk[:, 0] * xk[:, 2] - vk[:, 2] * xk[:, 0]) / rk, (vk[:, 0] * xk[:, 0] + vk[:, 2] * xk[:, 2]) / rk,
                      vk[:, 1])
            count += np.bincount(flat, minlength=count.size).reshape(shape)
            for k in range(3):
                sums[k] += np.bincount(flat, weights=values[k], minlength=count.size).reshape(shape)
        with np.errstate(invalid="ignore", divide="ignore"):
            means = sums / count
        means[:, count < 3] = np.nan
        panels.append((label, means))
    r_mid = 0.5 * (r_edges[1:] + r_edges[:-1])
    y_mid = 0.5 * (y_edges[1:] + y_edges[:-1])

    def circulation(means, r0=0.040, r1=0.056, y0=0.182, y1=0.204):
        u_r, u_y = means[1], means[2]
        i0, i1 = np.argmin(np.abs(r_mid - r0)), np.argmin(np.abs(r_mid - r1))
        j0, j1 = np.argmin(np.abs(y_mid - y0)), np.argmin(np.abs(y_mid - y1))
        d = arguments.dr
        bottom = np.nansum(u_r[i0:i1, j0]) * d
        right = np.nansum(u_y[i1, j0:j1]) * d
        top = -np.nansum(u_r[i0:i1, j1]) * d
        left = -np.nansum(u_y[i0, j0:j1]) * d
        return bottom + right + top + left

    fig, axes = plt.subplots(1, len(panels), figsize=(3.6 * len(panels) + 1.0, 4.6), squeeze=False, layout="constrained")
    image = None
    print("\nring vortex at the PBT tips: circulation of the azimuthal-mean (u_r, u_y) around r 40..56 mm, "
          "y 182..204 mm, m^2/s (positive: outward below, up outside, inward above, down inside)")
    for c, (label, means) in enumerate(panels):
        ax = axes[0, c]
        image = ax.pcolormesh(r_edges * 1e3, y_edges * 1e3, (means[0] / (OMEGA * r_mid[:, None])).T, vmin=0, vmax=1,
                              cmap="viridis")
        gr, gy = np.meshgrid(r_mid * 1e3, y_mid * 1e3, indexing="ij")
        ax.quiver(gr, gy, means[1], means[2], color="w", scale=5.0, width=0.004)
        ax.add_patch(plt.Rectangle((40, 182), 16, 22, fill=False, edgecolor="c", linestyle=":", linewidth=0.8))
        ax.plot([12.2, 48.2, 48.2, 12.2, 12.2], [185.9, 185.9, 203.4, 203.4, 185.9], color="r", linewidth=0.8)
        ax.set_aspect("equal")
        ax.set_xlabel("r, mm", fontsize=7)
        ax.set_ylabel("y, mm", fontsize=7)
        ax.tick_params(labelsize=6)
        gamma = circulation(means)
        ax.set_title(f"{label}\ncirculation {gamma * 1e3:+.2f} x 1e-3 m^2/s", fontsize=7)
        print(f"  {label:34s} {gamma * 1e3:+.3f} x 1e-3")
    fig.colorbar(image, ax=axes[0, :].tolist(), shrink=0.7, label="u_θ / (ω r)")
    fig.suptitle("PBT, azimuthal mean in the r-y plane; red: swept region of the blades; dotted: circulation contour",
                 fontsize=8)
    fig.savefig(arguments.meridional, dpi=140)
    print(f"wrote {arguments.meridional}")


if __name__ == "__main__":
    main()
