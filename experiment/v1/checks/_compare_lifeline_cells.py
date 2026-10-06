"""_compare_lifeline_cells.py — two-way against one-way lifelines of the 9-pool cell model (2026-10-07, stage 5).

Inputs: a TWO-WAY lifeline directory (the pools were integrated on the GPU and recorded: gly, growth_rate,
pen_capacity, sto) and a ONE-WAY lifeline directory with the OUT.npz of _integrate_ninepool_lifelines.py (the pools
integrated offline along the recorded C_s). For both:
  - distributions (mean, p5, p50, p95 over records and lifelines) of C_s, X_gly, mu, q_p, X_sto;
  - regime analysis of Haringa 2017 on q_s / q_s,max = C_s / (C_s + K_s): regime 1 excess (> --high), 2 limitation,
    3 low (< --low); volume fraction and mean residence time per regime (hysteresis 0.01, 0.36 s smoothing as in
    _analyze_lifelines.regime_analysis);
  - penicillin: mean q_p against the ideally mixed value: the 0-D chemostat of _model_ninepool.py at the same mean
    growth rate (needs scipy; skipped without).
usage:
    python experiment/v1/checks/_compare_lifeline_cells.py --two DIR_TWO --one DIR_ONE --one-pools OUT.npz
        [--start 30] [--ks 9.8e-6] [--high 0.2] [--low 0.05]
"""
import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from experiment.v1.utils.lifeline_recorder import load_lifelines   # noqa: E402
import _analyze_lifelines as life                                  # noqa: E402


def series_two_way(directory, start):
    data = load_lifelines(directory, with_aux=False)
    names = data["field_names"]
    time = data["time"].astype(np.float64)
    keep = time >= start
    scalars = data["scalars"][keep].astype(np.float64)
    get = lambda name: scalars[:, :, names.index(name)]
    return time[keep], dict(cs=get("substrate"), gly=get("gly"), mu=get("growth_rate"), q_p=get("pen_capacity"), sto=get("sto"))


def series_one_way(directory, pools_path, start):
    data = load_lifelines(directory, with_aux=False)
    names = data["field_names"]
    time = data["time"].astype(np.float64)
    pools = np.load(pools_path)
    if pools["time"].size != time.size or not np.allclose(pools["time"], time):
        # the offline integration may have started later (--start); align on time
        index = np.searchsorted(time, pools["time"])
        time = time[index]
        scalars = data["scalars"][index].astype(np.float64)
    else:
        scalars = data["scalars"].astype(np.float64)
    keep = time >= start
    cs = scalars[keep][:, :, names.index("substrate")]
    return time[keep], dict(cs=cs, gly=pools["gly"][keep], mu=pools["mu"][keep], q_p=pools["q_p"][keep], sto=pools["sto"][keep])


def describe(values):
    flat = values[np.isfinite(values)]
    q = np.percentile(flat, [5, 50, 95])
    return f"{flat.mean():10.4g} {q[0]:10.4g} {q[1]:10.4g} {q[2]:10.4g}"


def regimes(cs, time, ks, high, low):
    q = cs / (ks + np.maximum(cs, 0.0))
    smooth = life.moving_average(q, max(1, int(round(0.36 / max(np.median(np.diff(time)), 1e-6)))))
    states = np.zeros_like(smooth, dtype=np.int8)      # 1 excess, 2 limitation, 3 low
    states[:] = 2
    states[smooth > high] = 1
    states[smooth < low] = 3
    fractions = [float((states == s).mean()) for s in (1, 2, 3)]
    # mean residence time per regime from run lengths along each lifeline
    residence = {1: [], 2: [], 3: []}
    dt = float(np.median(np.diff(time)))
    for k in range(states.shape[1]):
        column = states[:, k]
        change = np.nonzero(np.diff(column) != 0)[0] + 1
        bounds = np.concatenate([[0], change, [column.size]])
        for a, b in zip(bounds[:-1], bounds[1:]):
            residence[int(column[a])].append((b - a) * dt)
    return fractions, {s: (float(np.mean(v)) if v else float("nan")) for s, v in residence.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--two", required=True)
    parser.add_argument("--one", required=True)
    parser.add_argument("--one-pools", required=True)
    parser.add_argument("--start", type=float, default=30.0)
    parser.add_argument("--ks", type=float, default=9.8e-6)
    parser.add_argument("--high", type=float, default=0.2)
    parser.add_argument("--low", type=float, default=0.05)
    arguments = parser.parse_args()
    sets = [("two-way", *series_two_way(arguments.two, arguments.start)),
            ("one-way", *series_one_way(arguments.one, arguments.one_pools, arguments.start))]
    print(f"{'quantity':10s} {'set':8s} {'mean':>10s} {'p5':>10s} {'p50':>10s} {'p95':>10s}")
    for key, unit in (("cs", "mol/kg"), ("gly", "umol/gdw"), ("mu", "1/h"), ("q_p", "mol/Cmol/h"), ("sto", "umol/gdw")):
        for label, time, s in sets:
            print(f"{key:10s} {label:8s} {describe(s[key])}   {unit}")
    print(f"\nregimes on q_s/q_s,max = C_s/(C_s + K_s), K_s {arguments.ks:g}: excess > {arguments.high}, low < {arguments.low}")
    for label, time, s in sets:
        fractions, residence = regimes(s["cs"], time, arguments.ks, arguments.high, arguments.low)
        print(f"   {label:8s} fractions E/L/S {fractions[0]:.3f} / {fractions[1]:.3f} / {fractions[2]:.3f}; "
              f"mean residence E {residence[1]:.2f} s, L {residence[2]:.2f} s, S {residence[3]:.2f} s")
    try:
        import _model_ninepool as model
        for label, time, s in sets:
            mu_mean = float(np.nanmean(s["mu"]))
            sol = model.run(max(mu_mean - model.P["vd"], 1e-3), 0.0895, 5e-3, 400.0, model.Y0, method="BDF")   # LSODA can stall at low D
            v, _ = model.rates(sol.y[:9, -1], sol.y[9, -1], sol.y[10, -1])
            print(f"   {label:8s} mean q_p {np.nanmean(s['q_p']):.3e} vs ideally mixed chemostat at the same mu ({mu_mean:.4f} 1/h): "
                  f"{v[7]:.3e} -> {100.0 * (np.nanmean(s['q_p']) / v[7] - 1.0):+.1f} %")
    except Exception as error:   # scipy missing in the solver environment
        print(f"   (ideal-mixing reference skipped: {error})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
