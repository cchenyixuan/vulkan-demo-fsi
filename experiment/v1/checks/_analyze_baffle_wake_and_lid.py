"""Two checks of the upper-tank fluctuation excess (y > 225 mm, r > 72 mm) on existing dumps:
 (1) baffle wakes: the non-axisymmetric energy density (ring deviation of v, 8 mm r-y rings) binned by the azimuth
     measured from the nearest baffle in the sense of rotation (0 = at the baffle plane, 0..120 deg period;
     the fluid moves from larger to smaller angle? no: the swirl is +theta, so the wake lies at angles > 0 just
     behind the baffle in the direction of motion ... the baffle is passed by fluid moving along +theta, the wake
     sits DOWNSTREAM = at larger theta than the baffle). Fluent snapshots are mirrored back so that the swirl is
     +theta (as in _analyze_pbt_bands.fluent_snapshots).
 (2) the lid: fluctuation energy density and pressure standard deviation (ring statistics) per 6 mm layer below
     the lid (y 426.5 mm), compared with the layers of the same thickness at mid height.
usage: python baffle_lid.py --sph LABEL DUMP --fluent IP ..."""
import sys

import numpy as np

sys.path.insert(0, "D:/Uni of Auckland Dropbox/Beichen Zhao/CFD Projects/SPH lifeline/vulkan-demo-fsi/experiment/v1/checks")
import _analyze_axisymmetric_energy as axi   # noqa: E402
import _analyze_pbt_bands as pb              # noqa: E402

RHO = 998.0
BAFFLES = np.radians([61.45, 181.45, 301.45])
AZ_EDGES = np.arange(0.0, 120.0 + 1e-9, 10.0)
LID = 0.4265
LAYER = 0.006


def ring_deviation(x, v):
    """per-particle deviation of v from the azimuthal mean of its 8 mm r-y ring (u_r, u_theta, u_y)"""
    r = np.maximum(np.hypot(x[:, 0], x[:, 2]), 1e-12)
    comp = np.stack([(v[:, 0] * x[:, 0] + v[:, 2] * x[:, 2]) / r, (v[:, 0] * x[:, 2] - v[:, 2] * x[:, 0]) / r, v[:, 1]], axis=1)
    ir = np.clip(np.digitize(r, axi.R_EDGES) - 1, 0, len(axi.R_EDGES) - 2)
    iy = np.clip(np.digitize(x[:, 1], axi.Y_EDGES) - 1, 0, len(axi.Y_EDGES) - 2)
    flat = ir * (len(axi.Y_EDGES) - 1) + iy
    size = (len(axi.R_EDGES) - 1) * (len(axi.Y_EDGES) - 1)
    count = np.bincount(flat, minlength=size).astype(float)
    mean = np.stack([np.bincount(flat, weights=comp[:, k], minlength=size) / np.maximum(count, 1) for k in range(3)], axis=1)
    return comp - mean[flat], r


def analyse(label, x, v, p):
    dev, r = ring_deviation(x, v)
    e = 0.5 * RHO * (dev ** 2).sum(axis=1)                      # J/m^3 per particle
    theta = np.arctan2(-x[:, 2], x[:, 0])                        # grows along the rotation (+theta sense of pb)
    # azimuth downstream of the nearest baffle plane (baffle azimuths in the same convention)
    baffle_theta = np.arctan2(-np.sin(BAFFLES), np.cos(BAFFLES))
    rel = np.min(np.mod(theta[:, None] - baffle_theta[None, :], 2 * np.pi), axis=1)
    rel_deg = np.degrees(rel) % 120.0
    top_outer = (x[:, 1] > 0.225) & (x[:, 1] < 0.40) & (r > 0.072) & (r < 0.110)
    print(f"\n{label}")
    print("  (1) upper tank r 72..110 mm, y 225..400 mm: fluctuation energy density J/m^3 by azimuth downstream of a baffle")
    ib = np.digitize(rel_deg[top_outer], AZ_EDGES) - 1
    dens = np.bincount(ib, weights=e[top_outer], minlength=12)[:12] / np.maximum(np.bincount(ib, minlength=12)[:12], 1)
    print("     " + " ".join(f"{a:4.0f}-{b:<3.0f}" for a, b in zip(AZ_EDGES[:-1], AZ_EDGES[1:])))
    print("     " + " ".join(f"{d:8.1f}" for d in dens))
    near_wall = (x[:, 1] > 0.225) & (x[:, 1] < 0.40) & (r > 0.110)
    ib = np.digitize(rel_deg[near_wall], AZ_EDGES) - 1
    dens = np.bincount(ib, weights=e[near_wall], minlength=12)[:12] / np.maximum(np.bincount(ib, minlength=12)[:12], 1)
    print("      r 110..144 mm (baffle band):")
    print("     " + " ".join(f"{d:8.1f}" for d in dens))
    print("  (2) layers below the lid (r < 140 mm): fluctuation energy density J/m^3 and ring pressure std Pa; mid-height reference y 300..330")
    # ring pressure std: deviation from the 8 mm ring mean of the pressure
    ir = np.clip(np.digitize(r, axi.R_EDGES) - 1, 0, len(axi.R_EDGES) - 2)
    iy = np.clip(np.digitize(x[:, 1], axi.Y_EDGES) - 1, 0, len(axi.Y_EDGES) - 2)
    flat = ir * (len(axi.Y_EDGES) - 1) + iy
    size = (len(axi.R_EDGES) - 1) * (len(axi.Y_EDGES) - 1)
    count = np.bincount(flat, minlength=size).astype(float)
    pmean = np.bincount(flat, weights=p, minlength=size) / np.maximum(count, 1)
    pdev = p - pmean[flat]
    inner = r < 0.140
    for k in range(6):
        y1, y0 = LID - k * LAYER, LID - (k + 1) * LAYER
        m = inner & (x[:, 1] >= y0) & (x[:, 1] < y1)
        print(f"     y {y0*1e3:5.1f}..{y1*1e3:5.1f} mm: n {m.sum():7d}  e {e[m].mean():7.1f}  p std {pdev[m].std():7.1f}")
    m = inner & (x[:, 1] >= 0.300) & (x[:, 1] < 0.330)
    print(f"     y 300..330 mm:    n {m.sum():7d}  e {e[m].mean():7.1f}  p std {pdev[m].std():7.1f}")


axi.BIN = 0.008
axi.R_EDGES = np.arange(0.0, 0.144 + axi.BIN, axi.BIN)
axi.Y_EDGES = np.arange(-0.064, 0.4265 + axi.BIN, axi.BIN)
args = sys.argv[1:]
i = 0
while i < len(args):
    if args[i] == "--sph":
        label, path = args[i + 1], args[i + 2]; i += 3
        d = np.load(path)
        f = (d["positions"][:, 3] > 0) & (d["material"] == 0)
        analyse(label, d["positions"][f, :3].astype(float), d["velocity_mass"][f, :3].astype(float), d["density_pressure"][f, 1].astype(float))
    elif args[i] == "--fluent":
        path = args[i + 1]; i += 2
        cells = axi.read_interpolation_file(path)
        x = np.stack([cells["x"], cells["y"] - axi.FLUENT_Y_SHIFT, cells["z"]], axis=1)
        v = np.stack([cells["x-velocity"], cells["y-velocity"], cells["z-velocity"]], axis=1)
        p = cells["pressure"] if "pressure" in cells else np.zeros(len(cells["x"]))
        r = np.maximum(np.hypot(x[:, 0], x[:, 2]), 1e-12)
        sense = np.sign(((v[:, 0] * x[:, 2] - v[:, 2] * x[:, 0]) / r).sum()) or 1.0
        if sense < 0:
            x[:, 2] *= -1.0; v[:, 2] *= -1.0
        analyse("Fluent " + path.split("/")[-1], x, v, p)
