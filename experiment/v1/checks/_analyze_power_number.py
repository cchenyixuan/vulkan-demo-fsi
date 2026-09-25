"""_analyze_power_number.py — power number from torque logs of the 30 L tank runs.

Usage (repo root):
    python experiment/v1/checks/_analyze_power_number.py OUT.png LABEL=torque.csv [LABEL=torque.csv ...]
        [--avg-from 20] [--avg-to 30]

Np = tau * omega / (rho * N^3 * D^5) with rho = 998 kg/m^3, N = 200/60 rev/s,
D = 0.096 m (Rushton diameter; the torque includes both impellers and the
shaft). Prints per-run mean / std over the averaging window and plots Np(t).

When the CSV has the per-impeller columns (torque_lower / torque_upper /
torque_shaft, written with --torque-split-height, 2026-09-26) the script also
prints Np of the Rushton (lower), the PBT (upper) and the shaft, all on the
same D = 0.096 m so that they add up to the total, plus the PBT on its own
diameter D_PBT = 0.0982 m for comparison with single-impeller correlations.
"""
import sys, argparse, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

RHO, N_REV, D, OMEGA = 998.0, 200.0 / 60.0, 0.096, 2 * np.pi * 200.0 / 60.0
SCALE = RHO * N_REV ** 3 * D ** 5
D_PBT = 0.0982                       # PBT tip diameter (2 x 0.0491 m)

parser = argparse.ArgumentParser()
parser.add_argument("out_png"); parser.add_argument("runs", nargs="+")
parser.add_argument("--avg-from", type=float, default=20.0); parser.add_argument("--avg-to", type=float, default=30.0)
args = parser.parse_args()
fig, ax = plt.subplots(figsize=(9, 5))
print(f"{'run':28s} {'t_end':>6s} {'Np mean':>8s} {'Np std':>7s} {'samples':>7s}   window [{args.avg_from}, {args.avg_to}] s")
for spec in args.runs:
    label, path = spec.split("=", 1)
    d = np.genfromtxt(path, delimiter=",", names=True)
    t = d["time"]; tau = d["torque_axis"]
    # torque_axis is the fluid torque on the rotor about +axis; the drive torque
    # is its negative; Np uses the magnitude of the power tau * omega.
    np_t = np.abs(tau) * abs(OMEGA) / SCALE
    ax.plot(t, np_t, lw=0.8, label=label)
    m = (t >= args.avg_from) & (t <= args.avg_to)
    if m.sum() >= 2:
        print(f"{label:28s} {t[-1]:6.1f} {np_t[m].mean():8.3f} {np_t[m].std():7.3f} {m.sum():7d}")
        if "torque_lower" in d.dtype.names:
            parts = {name: np.abs(d[f"torque_{name}"]) * abs(OMEGA) / SCALE for name in ("lower", "upper", "shaft")}
            print(f"{'':28s}   Rushton {parts['lower'][m].mean():.3f} +- {parts['lower'][m].std():.3f}"
                  f"   PBT {parts['upper'][m].mean():.3f} +- {parts['upper'][m].std():.3f}"
                  f" (on D_PBT {parts['upper'][m].mean() * (D / D_PBT) ** 5:.3f})"
                  f"   shaft {parts['shaft'][m].mean():.3f}   (all on D = {D} m)")
    else:
        print(f"{label:28s} {t[-1]:6.1f}   (not yet in window; last Np {np_t[-1]:.3f})")
ax.set_xlabel("t [s]"); ax.set_ylabel("Np = tau*omega / (rho N^3 D^5)"); ax.grid(alpha=.3); ax.legend()
ax.set_title("30 L tank, 200 rpm, Rushton + PBT: instantaneous power number")
fig.tight_layout(); fig.savefig(args.out_png, dpi=110); print("saved", args.out_png)
