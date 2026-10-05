"""Kinetic energy of the bulk by scale band: Shepard filters of widths W (support radius) give v~_W; the band
between two widths is v~_W1 - v~_W2 (structures of size between W1 and W2); below the smallest width is
v - v~_W0 (sub-kernel / noise); above the largest width the azimuthal mean (rings 8 mm of v~_Wmax) is the mean
flow and the remainder the large-scale non-axisymmetric part. Fluent is resampled onto the 3 mm reference points.
usage: python scale_bands.py OUT.txt --sph LABEL DUMP DX ... --fluent IP ..."""
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, "D:/Uni of Auckland Dropbox/Beichen Zhao/CFD Projects/SPH lifeline/vulkan-demo-fsi/experiment/v1/checks")
import _analyze_axisymmetric_energy as axi            # noqa: E402
import _analyze_fluctuation_scales as scales          # noqa: E402

WIDTHS = [0.006, 0.009, 0.0135, 0.020, 0.030]
RHO = 998.0
REF = "D:/Uni of Auckland Dropbox/Beichen Zhao/CFD Projects/SPH lifeline/vulkan-demo-fsi/output/kecause/s3_plates_pb2000_newpbt_fluentinit_dense_t0.300.npz"


SAMPLES = 120000


def shepard_at(tree, x_all, v_all, queries, support):
    """Shepard mean of v_all at the query points (sources: all particles)"""
    out = np.zeros((queries.shape[0], 3))
    for start in range(0, queries.shape[0], 5000):
        block = queries[start:start + 5000]
        neighbours = tree.query_ball_point(block, support)
        lengths = np.fromiter((len(n) for n in neighbours), dtype=np.int64, count=block.shape[0])
        i = np.repeat(np.arange(block.shape[0]), lengths)
        j = np.concatenate([np.asarray(n, dtype=np.int64) for n in neighbours])
        q = np.linalg.norm(x_all[j] - block[i], axis=1) / support
        w = np.where(q < 1.0, (1.0 - q) ** 6 * (35.0 / 3.0 * q * q + 6.0 * q + 1.0), 0.0)
        total = np.bincount(i, weights=w, minlength=block.shape[0])
        for a in range(3):
            out[start:start + block.shape[0], a] = np.bincount(i, weights=w * v_all[j, a], minlength=block.shape[0]) / total
    return out


def bands(x, v, dx):
    """energies per band from a random sample of bulk particles (the sources are all particles); scaled to the bulk"""
    bulk = np.nonzero(axi.bulk_mask(x))[0]
    rng = np.random.default_rng(1)
    pick = rng.choice(bulk, size=min(SAMPLES, bulk.size), replace=False)
    scale = RHO * dx ** 3 * bulk.size / pick.size
    tree = cKDTree(x)
    filtered = [shepard_at(tree, x, v, x[pick], w) for w in WIDTHS]
    out = {}
    for k, w in enumerate(WIDTHS):                     # energy of the residual below the width (cumulative, monotone)
        out[f"below {w*1e3:g} mm"] = 0.5 * scale * ((v[pick] - filtered[k]) ** 2).sum()
    count, mean, mean_square = axi.bin_moments(x[pick], v[pick])
    total, azimuthal = axi.energies(count * dx ** 3 * bulk.size / pick.size, count, mean, mean_square)
    out["non-axisym of v"] = (total - azimuthal).sum()
    count, mean, mean_square = axi.bin_moments(x[pick], filtered[-1])
    total, azimuthal = axi.energies(count * dx ** 3 * bulk.size / pick.size, count, mean, mean_square)
    out[f"non-axisym of v~ {WIDTHS[-1]*1e3:g} mm"] = (total - azimuthal).sum()
    out["azimuthal mean"] = azimuthal.sum()
    out["bulk total"] = 0.5 * scale * (v[pick] ** 2).sum()
    return out


out_path = sys.argv[1]
args = sys.argv[2:]
results = []
ref_x, _ = axi.sph_arrays(REF)
i = 0
while i < len(args):
    if args[i] == "--sph":
        label, path, dx = args[i + 1], args[i + 2], float(args[i + 3]); i += 4
        x, v = axi.sph_arrays(path)
        t0 = time.time(); results.append((label, bands(x, v, dx))); print(label, f"{time.time() - t0:.0f} s", flush=True)
    elif args[i] == "--fluent":
        path = args[i + 1]; i += 2
        v, _ = scales.fluent_on_points(path, ref_x, 0.003, 12.0)
        t0 = time.time(); results.append(("Fluent " + path.split("/")[-1], bands(ref_x, v, 0.003))); print(path, f"{time.time() - t0:.0f} s", flush=True)
keys = list(results[0][1])
with open(out_path, "w") as handle:
    handle.write("bulk kinetic energy by scale band, mJ\n")
    handle.write(f"{'':34s}" + "".join(f"{k:>28s}" for k in keys) + "\n")
    for label, r in results:
        handle.write(f"{label:34s}" + "".join(f"{r[k]*1e3:28.1f}" for k in keys) + "\n")
print(open(out_path).read())
