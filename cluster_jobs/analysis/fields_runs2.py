# fields_runs.py (Mahuika analysis_tools, 54 m3 own / mean pairs) with the statistics window taken from the environment:
# START (default 120) .. end of the data; 2026-11-08 for the 240..480 s extensions of G24_SL_t20 / t45.
# Snapshot field statistics, lifeline C_s / mu statistics, tank substrate inventory from the probes. Run from the repo root.
# usage: START=240 python fields_runs2.py PRODUCTION_DIR RUN [RUN ...]
import glob, os, sys, csv
import numpy as np
sys.path.insert(0, ".")
from experiment.v1.utils.lifeline_recorder import load_lifelines
P = sys.argv[1]
START = float(os.environ.get("START", "120"))
KS, Y, QMAX, M, ALPHA = 1e-5, 90.0, 8300e-6 / 3600, 222e-6 / 3600, 0.1
MUMAX = Y * (QMAX - M)
for n in sys.argv[2:]:
    print(f"== {n}  (window {START:.0f} s .. end)")
    for path in sorted(glob.glob(f"{P}/{n}_snapshots/*.npz")):
        d = np.load(path); names = [str(x) for x in d["field_names"]]; s = d["scalars"].astype(float); m = d["mass"].astype(float)
        cs = s[:, names.index("substrate")]; mu = s[:, names.index("growth_rate")]
        mean = np.dot(m, cs) / m.sum()
        plume = m[cs > 2 * mean].sum() / m.sum()
        q = cs / (cs + KS)
        starved = m[q < 0.05].sum() / m.sum(); excess = m[q > 0.95].sum() / m.sum()
        supply = QMAX * q; demand = (mu + ALPHA * MUMAX) / Y + M
        limited = m[demand < supply].sum() / m.sum()
        p = np.percentile(cs, [5, 50, 95]); pm = np.percentile(mu * 3600, [5, 50, 95])
        print(f"  t {float(d['time']):6.1f}: C_s mean {mean:.3e} p5/p50/p95 {p[0]:.2e}/{p[1]:.2e}/{p[2]:.2e}  plume {plume*100:.1f} %  "
              f"starved {starved*100:.1f} %  excess {excess*100:.1f} %  demand-limited {limited*100:.1f} %  mu p5/p50/p95 {pm[0]:.3f}/{pm[1]:.3f}/{pm[2]:.3f} 1/h")
    with open(f"{P}/{n}_probes.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    keys = list(rows[0].keys())
    tk = next(k for k in keys if k.strip().lower() in ("time", "t", "time_s"))
    sk = next(k for k in keys if "total" in k and "substrate" in k)
    t = np.array([float(r[tk]) for r in rows]); tot = np.array([float(r[sk]) for r in rows])
    w = t >= START
    print(f"  probes: total substrate {START:.0f}..{t[-1]:.0f} s mean {tot[w].mean():.4e}, min {tot[w].min():.4e}, max {tot[w].max():.4e}, end {tot[-1]:.4e}")
    data = load_lifelines(f"{P}/{n}_lifelines", with_aux=False)
    t = data["time"]; w = t >= START; names = list(data["field_names"])
    cs = data["scalars"][w][:, :, names.index("substrate")].astype(float)
    mu = data["scalars"][w][:, :, names.index("growth_rate")].astype(float) * 3600
    up = data["scalars"][w][:, :, names.index("uptake")].astype(float)
    print(f"  lifelines {START:.0f}..{t[-1]:.0f} s: C_s mean {cs.mean():.3e} p5 {np.percentile(cs, 5):.2e} p50 {np.median(cs):.2e} p95 {np.percentile(cs, 95):.2e} | "
          f"mu mean {mu.mean():.4f} p5 {np.percentile(mu, 5):.4f} p95 {np.percentile(mu, 95):.4f} 1/h | per-lifeline time-mean mu std {mu.mean(axis=0).std():.4f} 1/h | "
          f"uptake field mean {up.mean():.4e}")
