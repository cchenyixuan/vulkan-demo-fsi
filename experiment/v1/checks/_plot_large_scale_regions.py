"""Where the large-scale (> 30 mm) non-axisymmetric energy sits: v~_30 = Shepard mean with support 30 mm at a
random sample of fluid particles (all fluid, rotor boxes included), ring (8 mm r-y) variance of v~_30 about
its azimuthal mean = energy of the non-axisymmetric structures larger than ~30 mm; the azimuthal mean of v~_30
is the mean flow. Printed per region (height bands x radius bands), drawn as r-y maps with ratios to Fluent.
usage: python large_scale_regions.py OUT.png --sph LABEL DUMP DX ... --fluent IP ..."""
import sys
import time

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
sys.path.insert(0, "D:/Uni of Auckland Dropbox/Beichen Zhao/CFD Projects/SPH lifeline/vulkan-demo-fsi/experiment/v1/checks")
import _analyze_axisymmetric_energy as axi            # noqa: E402
import _analyze_fluctuation_scales as scales          # noqa: E402
from _analyze_scale_bands import shepard_at                    # noqa: E402

RHO = 998.0
WIDTH = 0.030
SAMPLES = 200000
REF = "D:/Uni of Auckland Dropbox/Beichen Zhao/CFD Projects/SPH lifeline/vulkan-demo-fsi/output/kecause/s3_plates_pb2000_newpbt_fluentinit_dense_t0.300.npz"
R_BANDS = ((0.0, 0.072, "r < 72"), (0.072, 0.110, "r 72..110"), (0.110, 0.144, "r 110..144"))
Y_BANDS = ((-0.070, 0.0187, "floor"), (0.0187, 0.0585, "Rushton height"), (0.0585, 0.165, "between"),
           (0.165, 0.225, "PBT height"), (0.225, 0.430, "top"))


def ring_fields(x, v, dx, n_total):
    """per 8 mm ring: non-axisymmetric energy density of v~_30 and of v, J/m^3, from the sample"""
    rng = np.random.default_rng(2)
    pick = rng.choice(x.shape[0], size=min(SAMPLES, x.shape[0]), replace=False)
    tree = cKDTree(x)
    big = shepard_at(tree, x, v, x[pick], WIDTH)
    out = []
    for field in (big, v[pick]):
        count, mean, mean_square = axi.bin_moments(x[pick], field)
        with np.errstate(invalid="ignore", divide="ignore"):
            density = 0.5 * RHO * np.clip(mean_square - mean ** 2, 0.0, None).sum(axis=1)
            density[count < 5] = np.nan
        out.append(density)
    count, mean, _ = axi.bin_moments(x[pick], big)
    mean_density = 0.5 * RHO * (mean ** 2).sum(axis=1)
    mean_density[count < 5] = np.nan
    return out[0], out[1], mean_density


axi.BIN = 0.008
axi.R_EDGES = np.arange(0.0, 0.144 + axi.BIN, axi.BIN)
axi.Y_EDGES = np.arange(-0.064, 0.4265 + axi.BIN, axi.BIN)
ref_x, _ = axi.sph_arrays(REF)
ref_count, _, _ = axi.bin_moments(ref_x, np.zeros_like(ref_x))
volume = ref_count * 0.003 ** 3
fluid = ref_count > 0
out_path = sys.argv[1]
args = sys.argv[2:]
sets = []
i = 0
fluent_acc = []
while i < len(args):
    if args[i] == "--sph":
        label, path, dx = args[i + 1], args[i + 2], float(args[i + 3]); i += 4
        x, v = axi.sph_arrays(path)
        t0 = time.time(); sets.append((label, ring_fields(x, v, dx, x.shape[0]))); print(label, f"{time.time() - t0:.0f} s", flush=True)
    elif args[i] == "--fluent":
        path = args[i + 1]; i += 2
        v, _ = scales.fluent_on_points(path, ref_x, 0.003, 12.0)
        t0 = time.time(); fluent_acc.append(ring_fields(ref_x, v, 0.003, ref_x.shape[0])); print(path, f"{time.time() - t0:.0f} s", flush=True)
if fluent_acc:
    sets.insert(0, (f"Fluent ({len(fluent_acc)})", tuple(np.nanmean([f[k] for f in fluent_acc], axis=0) for k in range(3))))

r_mid = 0.5 * (axi.R_EDGES[1:] + axi.R_EDGES[:-1])
y_mid = 0.5 * (axi.Y_EDGES[1:] + axi.Y_EDGES[:-1])
r_grid, y_grid = (a.ravel() for a in np.meshgrid(r_mid, y_mid, indexing="ij"))
print("\nnon-axisymmetric energy of structures > 30 mm (v~_30), mJ, by region   [in brackets: of the raw v]")
print(f"  {'region':30s}" + "".join(f"{label[:22]:>24s}" for label, _ in sets))
for y0, y1, y_name in Y_BANDS:
    for r0, r1, r_name in R_BANDS:
        sel = (y_grid >= y0) & (y_grid < y1) & (r_grid >= r0) & (r_grid < r1) & fluid
        cells = [f"{np.nansum(big[sel] * volume[sel])*1e3:8.1f} [{np.nansum(raw[sel] * volume[sel])*1e3:6.1f}]" for _, (big, raw, _) in sets]
        print(f"  {y_name + ', ' + r_name:30s}" + "".join(f"{c:>24s}" for c in cells))
    sel = (y_grid >= y0) & (y_grid < y1) & fluid
    cells = [f"{np.nansum(big[sel] * volume[sel])*1e3:8.1f} [{np.nansum(raw[sel] * volume[sel])*1e3:6.1f}]" for _, (big, raw, _) in sets]
    print(f"  {y_name + ', all r':30s}" + "".join(f"{c:>24s}" for c in cells))
cells = [f"{np.nansum(big[fluid] * volume[fluid])*1e3:8.1f} [{np.nansum(raw[fluid] * volume[fluid])*1e3:6.1f}]" for _, (big, raw, _) in sets]
print(f"  {'whole tank':30s}" + "".join(f"{c:>24s}" for c in cells))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
shape = (len(axi.R_EDGES) - 1, len(axi.Y_EDGES) - 1)
columns = 2 * len(sets) - 1
fig, axes = plt.subplots(1, columns, figsize=(2.4 * columns + 1.5, 6.5), squeeze=False, layout="constrained")
image = ratio_image = None
for c, (label, (big, _, _)) in enumerate(sets):
    ax = axes[0, c]
    image = ax.pcolormesh(axi.R_EDGES * 1e3, axi.Y_EDGES * 1e3, np.sqrt(2.0 * big / RHO).reshape(shape).T, vmin=0, vmax=0.15, cmap="magma")
    ax.set_title(f"{label}\nstructures > 30 mm, sqrt(2e/rho) m/s", fontsize=7)
for k, (label, (big, _, _)) in enumerate(sets[1:]):
    ax = axes[0, len(sets) + k]
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = (big / sets[0][1][0]).reshape(shape)
    ratio_image = ax.pcolormesh(axi.R_EDGES * 1e3, axi.Y_EDGES * 1e3, np.clip(ratio, 0.125, 8).T, norm=LogNorm(0.125, 8), cmap="RdBu_r")
    ax.set_title(f"{label} / {sets[0][0]}\nenergy ratio", fontsize=7)
for c in range(columns):
    ax = axes[0, c]
    for y0, y1 in ((18.7, 58.5), (165.0, 225.0)):
        ax.plot([0, 72, 72, 0], [y0, y0, y1, y1], color="c", linestyle=":", linewidth=0.8)
    ax.axvline(114.35, color="w", linewidth=0.5, alpha=0.5)
    ax.set_aspect("equal"); ax.set_xlim(0, 144); ax.set_ylim(-64, 426.5); ax.tick_params(labelsize=6)
fig.colorbar(image, ax=axes[0, len(sets) - 1], shrink=0.6, label="m/s")
bar = fig.colorbar(ratio_image, ax=axes[0, -1], shrink=0.6, label="energy ratio")
bar.set_ticks([0.125, 0.25, 0.5, 1, 2, 4, 8]); bar.set_ticklabels(["1/8", "1/4", "1/2", "1", "2", "4", "8"])
fig.suptitle("non-axisymmetric energy of structures larger than 30 mm (ring variance of the 30 mm Shepard mean); dotted: rotor boxes", fontsize=8)
fig.savefig(out_path, dpi=140)
print(f"wrote {out_path}")
