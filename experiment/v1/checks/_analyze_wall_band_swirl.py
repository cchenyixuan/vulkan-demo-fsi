"""_analyze_wall_band_swirl.py — swirl in the wall band of the 30 L tank as a function of the angle from the
nearest baffle, SPH dumps and Fluent snapshots alike (2026-10-13).

Question: our azimuthal-mean swirl in the wall band (r 110..144 mm) is 2-3 times Fluent's between the impellers
and above, although the cylinder brakes about as much in total (ours mostly pressure form drag of the smooth
shell and the baffle brackets, Fluent's viscous). Is the excess uniform around the circumference (near-wall
shear layer under-resolved) or concentrated downstream of the baffles / brackets (distribution of the braking)?

Method: fluid samples in r in [r0, r1], y in a height band; azimuth phi = atan2(z, x) measured from +x toward +z
(generator frame; Fluent y shifted by -58.5 mm). Baffles at 62, 182, 302 deg. The sense of rotation is taken from
the data (sign of the mean u_phi in the Rushton box); psi = angle from the baffle the fluid passed LAST, in the
sense of rotation, in [0, 120): psi -> 0 just downstream of a baffle (its wake), psi -> 120 just upstream of the
next one. Mean u_theta (positive in the sense of rotation) per psi bin and per sub-ring, samples weighted
equally (Fluent cells carry no volume; sub-rings of 10 mm limit the bias of the wall-refined cells).

usage:
    python experiment/v1/checks/_analyze_wall_band_swirl.py OUT_PREFIX --sph LABEL DUMP [--sph ...] --fluent IP [...]
        [--y0 0.0585 --y1 0.165] [--rings 0.110 0.120 0.130 0.144] [--bins 12]
"""
import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _analyze_axisymmetric_energy as axi   # noqa: E402

BAFFLES_DEG = (62.0, 182.0, 302.0)
PERIOD = 120.0


def cylindrical(x, v):
    r = np.hypot(x[:, 0], x[:, 2])
    phi = np.degrees(np.arctan2(x[:, 2], x[:, 0])) % 360.0
    e_r = np.stack([x[:, 0] / np.maximum(r, 1e-12), np.zeros_like(r), x[:, 2] / np.maximum(r, 1e-12)], axis=1)
    e_phi = np.stack([-e_r[:, 2], np.zeros_like(r), e_r[:, 0]], axis=1)      # +x toward +z
    return r, phi, (v * e_phi).sum(axis=1)


def sense_of_rotation(x, v):
    r, phi, u_phi = cylindrical(x, v)
    box = (r < 0.072) & (x[:, 1] > 0.0187) & (x[:, 1] < 0.0585)
    return 1.0 if u_phi[box].mean() > 0 else -1.0


def baffle_angle(phi, sense):
    """angle from the baffle passed last, in the sense of rotation, in [0, PERIOD)"""
    offsets = np.asarray(BAFFLES_DEG) % PERIOD
    if sense > 0:            # fluid moves toward increasing phi: the last baffle is the one below phi
        psi = (phi[:, None] - offsets[None, :]) % PERIOD
    else:                    # fluid moves toward decreasing phi
        psi = (offsets[None, :] - phi[:, None]) % PERIOD
    return psi.min(axis=1)


def profile(x, v, sense, y0, y1, rings, bins):
    r, phi, u_phi = cylindrical(x, v)
    u_theta = sense * u_phi
    psi = baffle_angle(phi, sense)
    edges = np.linspace(0.0, PERIOD, bins + 1)
    out = []
    for r0, r1 in zip(rings[:-1], rings[1:]):
        select = (r >= r0) & (r < r1) & (x[:, 1] >= y0) & (x[:, 1] < y1)
        which = np.clip(np.digitize(psi[select], edges) - 1, 0, bins - 1)
        count = np.bincount(which, minlength=bins)
        mean = np.bincount(which, weights=u_theta[select], minlength=bins) / np.maximum(count, 1)
        out.append((r0, r1, count, mean, u_theta[select].mean() if select.any() else np.nan))
    return edges, out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out")
    parser.add_argument("--sph", nargs=2, action="append", default=[], metavar=("LABEL", "DUMP"))
    parser.add_argument("--fluent", action="append", default=[])
    parser.add_argument("--y0", type=float, default=0.0585)
    parser.add_argument("--y1", type=float, default=0.165)
    parser.add_argument("--rings", type=float, nargs="+", default=[0.110, 0.120, 0.130, 0.144])
    parser.add_argument("--bins", type=int, default=12)
    arguments = parser.parse_args()

    datasets = []
    for label, path in arguments.sph:
        x, v = axi.sph_arrays(path)
        datasets.append((label, x, v))
    for path in arguments.fluent:
        cells = axi.read_interpolation_file(path)
        x = np.stack([cells["x"], cells["y"] - axi.FLUENT_Y_SHIFT, cells["z"]], axis=1)
        v = np.stack([cells["x-velocity"], cells["y-velocity"], cells["z-velocity"]], axis=1)
        datasets.append((f"Fluent {pathlib.Path(path).stem}", x, v))

    results = {}
    for label, x, v in datasets:
        sense = sense_of_rotation(x, v)
        edges, rows = profile(x, v, sense, arguments.y0, arguments.y1, arguments.rings, arguments.bins)
        results[label] = (edges, rows)
        print(f"\n{label}: rotation sense {'+phi' if sense > 0 else '-phi'}, y {arguments.y0 * 1e3:.0f}..{arguments.y1 * 1e3:.0f} mm")
        print("  ring, mm   band mean |  mean u_theta (m/s) per angle from the last baffle (deg, bin centres):")
        print("                       |  " + " ".join(f"{0.5 * (a + b):6.0f}" for a, b in zip(edges[:-1], edges[1:])))
        for r0, r1, count, mean, band_mean in rows:
            print(f"  {r0 * 1e3:3.0f}..{r1 * 1e3:3.0f}   {band_mean:8.4f}  |  " + " ".join(f"{m:6.3f}" for m in mean)
                  + f"   (samples {count.min()}..{count.max()})")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ring_count = len(arguments.rings) - 1
        fig, axes = plt.subplots(1, ring_count, figsize=(5 * ring_count, 4), squeeze=False)
        for column in range(ring_count):
            ax = axes[0, column]
            for label, (edges, rows) in results.items():
                centres = 0.5 * (edges[:-1] + edges[1:])
                ax.plot(centres, rows[column][3], marker="o", label=label)
            r0, r1 = arguments.rings[column], arguments.rings[column + 1]
            ax.set_title(f"r {r0 * 1e3:.0f}..{r1 * 1e3:.0f} mm, y {arguments.y0 * 1e3:.0f}..{arguments.y1 * 1e3:.0f} mm")
            ax.set_xlabel("angle from the last baffle, deg (0 = just downstream)")
            ax.set_ylabel("mean u_theta, m/s")
            ax.grid(alpha=0.3)
        axes[0, 0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(arguments.out + ".png", dpi=130)
        print(f"wrote {arguments.out}.png")
    except Exception as error:      # noqa: BLE001
        print(f"no figure: {error}")


if __name__ == "__main__":
    main()
