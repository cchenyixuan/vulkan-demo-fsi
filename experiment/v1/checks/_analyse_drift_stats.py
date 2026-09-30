"""Vertical drift statistics of a tank dump: per slab mean / median v_y, mean a_y (+g), shift_y; (r, y) map of v_y.
usage: drift_stats.py DUMP.npz [g]"""
import sys

import numpy as np

z = np.load(sys.argv[1])
g = float(sys.argv[2]) if len(sys.argv) > 2 else 9.81
pos = z["positions"]; n = int(z["step"]) if False else None
material = z["material"]
live = (pos[:, 3] > 0)
live[0] = False
fluid = live & (material == 0)
x = pos[fluid, :3].astype(np.float64)
v = z["velocity_mass"][fluid, :3].astype(np.float64)
a = z["acceleration"][fluid, :3].astype(np.float64)
shift = z["shift"][fluid, :3].astype(np.float64)
rho, p = z["density_pressure"][fluid, 0].astype(np.float64), z["density_pressure"][fluid, 1].astype(np.float64)
dx = float(z["spacing"]); dt = float(z["time"]) / int(z["step"])
y = x[:, 1]; r = np.hypot(x[:, 0], x[:, 2])
print(f"step {int(z['step'])}, t {float(z['time']):.3f} s, dt {dt:.3e}, fluid {len(x):,}, dx {dx * 1e3:g} mm")
print(f"whole tank: mean v_y {v[:, 1].mean() * 1e3:7.3f} mm/s, median {np.median(v[:, 1]) * 1e3:7.3f}, "
      f"mean a_y + g {a[:, 1].mean() + g:8.4f} m/s^2, mean shift_y {shift[:, 1].mean() / dx * 1e3:6.3f} 1e-3 dx")
print()
print("slabs of 40 mm (mm/s = 1e-3 m/s; a in m/s^2; shift in 1e-3 dx per step; 'v dt' in 1e-3 dx per step):")
print(f"{'y0..y1':>12s} {'n':>7s} {'mean v_y':>9s} {'median':>8s} {'q10':>8s} {'q90':>8s} {'frac<0':>7s} {'v dt':>7s} {'shift_y':>8s} {'a_y+g':>8s} {'a_y+g med':>9s} {'|v| mean':>8s}")
for y0 in np.arange(-60, 430, 40):
    m = (y >= y0 * 1e-3) & (y < (y0 + 40) * 1e-3)
    if m.sum() == 0:
        continue
    vy = v[m, 1]
    print(f"{y0:5.0f}..{y0 + 40:5.0f} {int(m.sum()):7d} {vy.mean() * 1e3:9.3f} {np.median(vy) * 1e3:8.3f} {np.quantile(vy, 0.1) * 1e3:8.2f} "
          f"{np.quantile(vy, 0.9) * 1e3:8.2f} {(vy < 0).mean():7.3f} {vy.mean() * dt / dx * 1e3:7.3f} {shift[m, 1].mean() / dx * 1e3:8.3f} "
          f"{a[m, 1].mean() + g:8.4f} {np.median(a[m, 1]) + g:9.4f} {np.linalg.norm(v[m], axis=1).mean():8.3f}")
print()
print("(r, y) map of mean v_y in mm/s: rows y (40 mm), columns r (24 mm bins, 0..144)")
r_edges = np.arange(0, 145, 24)
print(f"{'y0..y1':>12s} " + " ".join(f"{r0:3.0f}-{r0 + 24:3.0f}" for r0 in r_edges[:-1]))
for y0 in np.arange(-60, 430, 40):
    row = []
    for r0 in r_edges[:-1]:
        m = (y >= y0 * 1e-3) & (y < (y0 + 40) * 1e-3) & (r >= r0 * 1e-3) & (r < (r0 + 24) * 1e-3)
        row.append(f"{v[m, 1].mean() * 1e3:7.2f}" if m.sum() > 50 else "      -")
    print(f"{y0:5.0f}..{y0 + 40:5.0f} " + " ".join(row))
print()
print("(r, y) map of mean a_y + g in m/s^2:")
for y0 in np.arange(-60, 430, 40):
    row = []
    for r0 in r_edges[:-1]:
        m = (y >= y0 * 1e-3) & (y < (y0 + 40) * 1e-3) & (r >= r0 * 1e-3) & (r < (r0 + 24) * 1e-3)
        row.append(f"{a[m, 1].mean() + g:7.3f}" if m.sum() > 50 else "      -")
    print(f"{y0:5.0f}..{y0 + 40:5.0f} " + " ".join(row))
# is the drift carried by a subset? contribution of the 10 % most negative v_y to the slab mean
print()
print("share of the slab's net downward flux carried by the 10 % most negative particles:")
for y0 in np.arange(-60, 430, 80):
    m = (y >= y0 * 1e-3) & (y < (y0 + 80) * 1e-3)
    vy = np.sort(v[m, 1])
    k = len(vy) // 10
    print(f"  {y0:5.0f}..{y0 + 80:5.0f}: sum v_y {vy.sum() * 1e3:10.1f}, lowest 10 % {vy[:k].sum() * 1e3:10.1f}, highest 10 % {vy[-k:].sum() * 1e3:10.1f}, "
          f"middle 80 % {vy[k:-k].sum() * 1e3:10.1f} (mm/s summed)")
