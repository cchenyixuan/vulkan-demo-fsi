"""_compare_knob_runs.py — one-way vs local two-way vs population-average two-way with the uptake-inhibition
knob (2026-10-08, stage 5, cases B1 / 54 m3): what does LOCAL coupling change?

Inputs: run directories written by _run_v1_headless.py (probes.csv, lifelines/, snapshots/), named by the
protocol:  A = one-way (Monod sink; pools from _integrate_ninepool_lifelines.py --ki K, pass --one-pools A=OUT.npz),
C / D = local two-way (uptake_inhibition own), E = two-way with the population-mean X_gly (uptake_inhibition mean).
For every run:
  1. probes.csv: mean C_s (total / fluid mass), cumulative uptake and the uptake rate against time;
  2. snapshots: C_s mean / p95 / max over the fluid particles, the "plume" mass fraction (C_s > --plume x mean),
     X_gly mean / p95 and, for the local runs, the FEEDBACK DEVIATION d_i = f(X_gly,i) / f(mean X_gly) - 1 with
     f = 1 / (1 + X_gly / K_i): the mass fraction with |d| > 0.1 and its mean inside / outside the plume. This is
     the quantity the population-average coupling cannot represent;
  3. lifelines: distributions of C_s, X_gly, mu over [--start, end] and the Haringa 2017 regimes on C_s.
A JSON summary and (with matplotlib) time-series plots go to --out.
usage:
    python experiment/v1/checks/_compare_knob_runs.py --run A=DIR --run C=DIR --run E=DIR [--one-pools A=OUT.npz]
        --ki 49.2 [--start 40] [--plume 2.0] [--ks 9.8e-6] [--out DIR]
"""
import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from experiment.v1.utils.lifeline_recorder import load_lifelines   # noqa: E402
import _compare_lifeline_cells as cells                            # noqa: E402  (regimes, describe)


def read_probes(path):
    with open(path, encoding="utf-8") as handle:
        header = handle.readline().strip().split(",")
    data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    return {name: data[:, index] for index, name in enumerate(header)}


def probe_series(run_dir):
    p = read_probes(pathlib.Path(run_dir) / "probes.csv")
    time = p["time"]
    mass = p["fluid_mass"]
    cs_mean = p["total:substrate"] / mass
    uptake = p["total:uptake"] if "total:uptake" in p else None
    rate = np.gradient(uptake, time) if uptake is not None else None
    return {"time": time, "cs_mean": cs_mean, "uptake_total": uptake, "uptake_rate": rate,
            "feed_total": p.get("total:feed")}


def snapshot_metrics(run_dir, ki, plume_factor):
    rows = []
    for path in sorted(pathlib.Path(run_dir, "snapshots").glob("snapshot_*.npz")):
        with np.load(path) as snap:
            names = [str(n) for n in snap["field_names"]]
            mass = snap["mass"].astype(np.float64)
            w = mass / mass.sum()
            cs = snap["scalars"][:, names.index("substrate")].astype(np.float64)
            row = {"time": float(snap["time"]), "cs_mean": float(np.dot(w, cs)),
                   "cs_p95": float(np.percentile(cs, 95)), "cs_max": float(cs.max())}
            plume = cs > plume_factor * row["cs_mean"]
            row["plume_fraction"] = float(w[plume].sum())
            if "gly" in names:
                gly = np.maximum(snap["scalars"][:, names.index("gly")].astype(np.float64), 0.0)
                row["gly_mean"] = float(np.dot(w, gly))
                row["gly_p95"] = float(np.percentile(gly, 95))
                factor = 1.0 / (1.0 + gly / ki)
                deviation = factor / (1.0 / (1.0 + row["gly_mean"] / ki)) - 1.0
                row["deviation_fraction_10pct"] = float(w[np.abs(deviation) > 0.1].sum())
                row["deviation_mean_in_plume"] = float(np.dot(w[plume], deviation[plume]) / max(w[plume].sum(), 1e-30)) if plume.any() else float("nan")
                row["deviation_mean_outside"] = float(np.dot(w[~plume], deviation[~plume]) / max(w[~plume].sum(), 1e-30))
            if "growth_rate" in names:
                row["mu_mean"] = float(np.dot(w, snap["scalars"][:, names.index("growth_rate")].astype(np.float64)))
            rows.append(row)
    return rows


def lifeline_series(run_dir, start, pools_path=None):
    data = load_lifelines(pathlib.Path(run_dir) / "lifelines", with_aux=False)
    names = data["field_names"]
    time = data["time"].astype(np.float64)
    scalars = data["scalars"].astype(np.float64)
    series = {"cs": scalars[:, :, names.index("substrate")]}
    if "gly" in names:
        series["gly"] = scalars[:, :, names.index("gly")]
    if "growth_rate" in names:
        series["mu"] = scalars[:, :, names.index("growth_rate")]
    if pools_path is not None:
        pools = np.load(pools_path)
        if pools["time"].size != time.size or not np.allclose(pools["time"], time):
            index = np.searchsorted(time, pools["time"])
            time = time[index]
            series = {k: v[index] for k, v in series.items()}
        series["gly"] = pools["gly"]
        series["mu"] = pools["mu"]
    keep = time >= start
    return time[keep], {k: v[keep] for k, v in series.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", action="append", required=True, metavar="NAME=DIR")
    parser.add_argument("--one-pools", action="append", default=[], metavar="NAME=OUT.npz",
                        help="offline pools of a one-way run (_integrate_ninepool_lifelines.py)")
    parser.add_argument("--ki", type=float, required=True, help="K_i of the knob (umol/gdw)")
    parser.add_argument("--start", type=float, default=40.0, help="lifeline statistics from this time (feed start)")
    parser.add_argument("--plume", type=float, default=2.0, help="plume = C_s > this factor x mean")
    parser.add_argument("--ks", type=float, default=9.8e-6)
    parser.add_argument("--high", type=float, default=0.2)
    parser.add_argument("--low", type=float, default=0.05)
    parser.add_argument("--out", default=None, help="directory of the JSON summary and plots")
    arguments = parser.parse_args()
    runs = dict(item.split("=", 1) for item in arguments.run)
    pools = dict(item.split("=", 1) for item in arguments.one_pools)
    out = pathlib.Path(arguments.out) if arguments.out else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
    summary = {}

    print("1. probes: mean C_s (mol/kg), cumulative uptake (mol, whole tank), uptake rate (mol/s, whole tank)")
    probes = {name: probe_series(path) for name, path in runs.items()}
    marks = sorted({round(t, 1) for p in probes.values() for t in np.percentile(p["time"], [0, 25, 50, 75, 100])})
    for name, p in probes.items():
        print(f"   {name}:")
        for t in marks:
            k = int(np.argmin(np.abs(p["time"] - t)))
            line = f"      t {p['time'][k]:7.2f} s  C_s {p['cs_mean'][k]:.4e}"
            if p["uptake_total"] is not None:
                line += f"  uptake {p['uptake_total'][k] / max(1e-30, 1.0):.4e}  rate {p['uptake_rate'][k]:.3e}"
            print(line)
        summary[name] = {"probes": {k: v.tolist() for k, v in p.items() if v is not None}}

    print(f"\n2. snapshots (plume = C_s > {arguments.plume} x mean; deviation = local knob factor / population factor - 1)")
    for name, path in runs.items():
        rows = snapshot_metrics(path, arguments.ki, arguments.plume)
        summary[name]["snapshots"] = rows
        print(f"   {name}:")
        for row in rows:
            line = (f"      t {row['time']:7.2f} s  C_s mean {row['cs_mean']:.3e} p95 {row['cs_p95']:.3e} max {row['cs_max']:.3e}  "
                    f"plume {100 * row['plume_fraction']:5.2f} %")
            if "gly_mean" in row:
                line += (f"  X_gly {row['gly_mean']:6.2f} p95 {row['gly_p95']:6.2f}  |dev|>10% {100 * row['deviation_fraction_10pct']:5.2f} %  "
                         f"dev plume {row['deviation_mean_in_plume']:+.3f} outside {row['deviation_mean_outside']:+.3f}")
            print(line)

    print(f"\n3. lifelines from t = {arguments.start} s: mean, p5, p50, p95 over records and lifelines")
    print(f"{'quantity':8s} {'run':4s} {'mean':>10s} {'p5':>10s} {'p50':>10s} {'p95':>10s}")
    lifelines = {}
    for name, path in runs.items():
        time, series = lifeline_series(path, arguments.start, pools.get(name))
        lifelines[name] = (time, series)
    for key, unit in (("cs", "mol/kg"), ("gly", "umol/gdw"), ("mu", "1/h")):
        for name, (time, series) in lifelines.items():
            if key in series:
                print(f"{key:8s} {name:4s} {cells.describe(series[key])}   {unit}")
    print(f"   regimes on C_s/(C_s + K_s), K_s {arguments.ks:g}: excess > {arguments.high}, low < {arguments.low}")
    for name, (time, series) in lifelines.items():
        fractions, residence = cells.regimes(series["cs"], time, arguments.ks, arguments.high, arguments.low)
        summary[name]["lifelines"] = {"regime_fractions": fractions, "residence": {str(k): v for k, v in residence.items()},
                                      "cs_p95": float(np.nanpercentile(series["cs"], 95)),
                                      "gly_p95": float(np.nanpercentile(series["gly"], 95)) if "gly" in series else None}
        print(f"      {name:4s} fractions E/L/S {fractions[0]:.3f} / {fractions[1]:.3f} / {fractions[2]:.3f}; "
              f"mean residence E {residence[1]:.2f} s, L {residence[2]:.2f} s, S {residence[3]:.2f} s")

    if out is not None:
        (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            figure, axes = plt.subplots(1, 3, figsize=(15, 4))
            for name, p in probes.items():
                axes[0].plot(p["time"], p["cs_mean"] * 1e6, label=name)
                if p["uptake_rate"] is not None:
                    axes[1].plot(p["time"], p["uptake_rate"] * 1e6, label=name)
            for name in runs:
                rows = summary[name]["snapshots"]
                if rows and "gly_mean" in rows[0]:
                    axes[2].plot([r["time"] for r in rows], [r["gly_p95"] for r in rows], "o-", label=f"{name} p95")
                    axes[2].plot([r["time"] for r in rows], [r["gly_mean"] for r in rows], "s--", label=f"{name} mean")
            axes[0].set_ylabel("mean C_s (umol/kg)"); axes[1].set_ylabel("uptake rate, whole tank (umol/s)"); axes[2].set_ylabel("X_gly (umol/gdw)")
            for axis in axes:
                axis.set_xlabel("t (s)"); axis.legend(fontsize=8); axis.grid(alpha=0.3)
            figure.tight_layout()
            figure.savefig(out / "knob_runs.png", dpi=130)
            print(f"\nwrote {out / 'summary.json'} and {out / 'knob_runs.png'}")
        except Exception as error:   # matplotlib missing in the solver environment
            print(f"\nwrote {out / 'summary.json'} (plot skipped: {error})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
