"""_analyze_power_number.py — power number from torque logs of the 30 L tank runs.

Usage (repo root):
    python experiment/v1/checks/_analyze_power_number.py OUT.png LABEL=torque.csv [LABEL=torque.csv ...]
        [--avg-from 20] [--avg-to 30]

Np = tau * omega / (rho * N^3 * D^5) with rho = 998 kg/m^3, N = 200/60 rev/s,
D = 0.096 m (Rushton diameter; the torque includes both impellers and the
shaft). Prints per-run mean / std over the averaging window and plots Np(t).
"""
import sys, argparse, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

RHO, N_REV, D, OMEGA = 998.0, 200.0 / 60.0, 0.096, 2 * np.pi * 200.0 / 60.0
SCALE = RHO * N_REV ** 3 * D ** 5

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
    else:
        print(f"{label:28s} {t[-1]:6.1f}   (not yet in window; last Np {np_t[-1]:.3f})")
ax.set_xlabel("t [s]"); ax.set_ylabel("Np = tau*omega / (rho N^3 D^5)"); ax.grid(alpha=.3); ax.legend()
ax.set_title("30 L tank, 200 rpm, Rushton + PBT: instantaneous power number")
fig.tight_layout(); fig.savefig(args.out_png, dpi=110); print("saved", args.out_png)
