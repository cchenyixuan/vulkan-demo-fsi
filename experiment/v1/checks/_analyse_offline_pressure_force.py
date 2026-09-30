"""Recompute the SPH pressure acceleration of sample fluid particles from a dump, with the kernels and the
pair form of force.comp (Wendland C4, support h, KCG matrix with Tikhonov xi, reverse pair correction, TIC),
and compare with rho g and with the acceleration stored in the dump.
usage: offline_pressure_force.py DUMP.npz [y0 y1 in mm] [samples]"""
import sys

import numpy as np
from scipy.spatial import cKDTree

z = np.load(sys.argv[1])
y0, y1 = (float(sys.argv[2]), float(sys.argv[3])) if len(sys.argv) > 3 else (300.0, 400.0)
samples = int(sys.argv[4]) if len(sys.argv) > 4 else 3000
G = np.array([0.0, -9.81, 0.0])
RHO0 = 998.0
XI = float(sys.argv[5]) if len(sys.argv) > 5 else 0.01
pos = z["positions"]; live = pos[:, 3] > 0; live[0] = False
material = z["material"]
x = pos[:, :3].astype(np.float64)
rho = z["density_pressure"][:, 0].astype(np.float64)
p = z["density_pressure"][:, 1].astype(np.float64)
mass = z["velocity_mass"][:, 3].astype(np.float64)
acc = z["acceleration"][:, :3].astype(np.float64)
h = float(z["smoothing_length"]); dx = float(z["spacing"])
C = 495.0 / (32.0 * np.pi * h ** 3)
idx_live = np.flatnonzero(live)
tree = cKDTree(x[idx_live])
fluid = live & (material == 0)
r_cyl = np.hypot(x[:, 0], x[:, 2])
candidates = np.flatnonzero(fluid & (x[:, 1] > y0 * 1e-3) & (x[:, 1] < y1 * 1e-3) & (r_cyl < 0.11))
rng = np.random.default_rng(1)
sample = rng.choice(candidates, size=min(samples, len(candidates)), replace=False)
print(f"dump t {float(z['time']):.3f} s, h {h * 1e3:.1f} mm, mass {mass[sample[0]]:.6e} kg (rho0 dx^3 x {mass[sample[0]] / (RHO0 * dx ** 3):.4f}), "
      f"{len(sample)} sample particles in y {y0:.0f}..{y1:.0f} mm, r < 110 mm")


def grad_w(d, r):
    q = r / h
    return ((C / h) * (1 - q) ** 5 * q * (-280.0 / 3.0 * q - 56.0 / 3.0))[:, None] * (d / r[:, None])   # gradient at i of W(|x_i - x_j|), d = x_i - x_j


results = {"rev_xi": [], "rev_noxi": [], "own_xi": [], "sym_uncorrected": [], "anti_xi": []}
resid = []
kernel_sums = []
B_cache = {}


def matrix(i, xi):
    js = idx_live[tree.query_ball_point(x[i], h)]
    js = js[js != i]
    d = x[i] - x[js]; r = np.linalg.norm(d, axis=1)
    gw = grad_w(d, r)
    V = mass[js] / rho[js]
    A = np.einsum("j,ja,jb->ab", V, -d, gw)     # sum V_j (x_j - x_i) (x) grad W
    A_reg = A + xi * np.eye(3)
    B = np.linalg.inv(A_reg) if abs(np.linalg.det(A_reg)) > 1e-4 else np.eye(3)
    return js, d, r, gw, V, B, A


for i in sample:
    js, d, r, gw, V, B_i, A_i = matrix(i, XI)
    _, _, _, _, _, B_i0, _ = matrix(i, 0.0)
    kernel_sums.append(np.sum(V * C * (1 - r / h) ** 6 * (1 + 6 * r / h + 35.0 / 3.0 * (r / h) ** 2)))
    # neighbours' matrices (needed for the reverse form); fluid neighbours only get the pair correction
    B_j = np.empty((len(js), 3, 3)); B_j0 = np.empty_like(B_j)
    for k, j in enumerate(js):
        if j not in B_cache:
            B_cache[j] = (matrix(j, XI)[5], matrix(j, 0.0)[5])
        B_j[k], B_j0[k] = B_cache[j]
    fluid_j = material[js] == 0
    gw_i = gw @ B_i.T; gw_i0 = gw @ B_i0.T
    gw_j = np.einsum("jab,jb->ja", B_j, gw); gw_j0 = np.einsum("jab,jb->ja", B_j0, gw)
    tic = p[i] < 0.0
    for name, gi, gj in (("rev_xi", gw_i, gw_j), ("rev_noxi", gw_i0, gw_j0)):
        if tic:
            pair = (p[js] - p[i])[:, None] * gi
        else:
            pair = np.where(fluid_j[:, None], p[i] * gj + p[js][:, None] * gi, (p[i] + p[js])[:, None] * gi)
        results[name].append(-(V[:, None] * pair).sum(axis=0) / rho[i])
    pair_own = ((p[js] - p[i]) if tic else (p[js] + p[i]))[:, None] * gw_i
    results["own_xi"].append(-(V[:, None] * pair_own).sum(axis=0) / rho[i])
    results["sym_uncorrected"].append(-(V[:, None] * (p[js] + p[i])[:, None] * gw).sum(axis=0) / rho[i])
    results["anti_xi"].append(-(V[:, None] * (p[js] - p[i])[:, None] * gw_i).sum(axis=0) / rho[i])
    resid.append(-(V[:, None] * gw_i).sum(axis=0) * p[i] / rho[i])        # the p_i sum V B grad W residual of the own form

rho_s = rho[sample]; p_s = p[sample]
print(f"sample: rho {rho_s.mean():.3f}, p {p_s.mean():.1f} Pa ({(p_s < 0).mean() * 100:.1f} % negative), kernel sum {np.mean(kernel_sums):.4f}, "
      f"stored a_y + g {acc[sample, 1].mean() + 9.81:.4f} m/s^2 (median {np.median(acc[sample, 1]) + 9.81:.4f})")
print(f"{'form':18s} {'mean a_y':>9s} {'median':>9s} {'std':>8s}  {'a_y / g - 1':>12s}  {'mean a_x':>9s} {'mean a_z':>9s}")
for name, values in results.items():
    values = np.array(values)
    print(f"{name:18s} {values[:, 1].mean():9.4f} {np.median(values[:, 1]):9.4f} {values[:, 1].std():8.4f}  {values[:, 1].mean() / 9.81 - 1:12.5f}  "
          f"{values[:, 0].mean():9.4f} {values[:, 2].mean():9.4f}")
resid = np.array(resid)
print(f"{'p_i sum V B gradW':18s} {resid[:, 1].mean():9.4f} {np.median(resid[:, 1]):9.4f} {resid[:, 1].std():8.4f}   (residual term of the own form)")
print(f"rho_i g / rho0 g - 1 = {(rho_s.mean() / RHO0 - 1):.5f}   (the pressure gradient of the initial profile is rho0 g; balance needs rho_i g)")
