"""_test_integrate_freeze_e11.py — synthetic check of --freeze-e11 in _integrate_ninepool_lifelines.py (2026-10-13).

Synthetic lifelines (fluctuating C_s around 5e-6 mol/kg, 0.03 s records) are integrated with integrate() and stitch()
with the option off and on. On: X_E11 stays at its initial value in every record and in the final stitched state, the
other pools move. Off: X_E11 drifts. With --old OLD.py (a copy of the script before the option, e.g. `git show
d27b0e4:experiment/v1/checks/_integrate_ninepool_lifelines.py`) every output array of the option-off run must be
bit-identical to the old script's.

usage: python experiment/v1/checks/_test_integrate_freeze_e11.py [--old OLD.py]
"""
import argparse
import importlib.util
import pathlib
import sys
import tempfile

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _integrate_ninepool_lifelines as new_module   # noqa: E402


def synthetic(seed=0, n=400, k=6, interval=0.03):
    rng = np.random.default_rng(seed)
    time = 30.0 + interval * np.arange(n)
    # log-normal C_s with a slow drift per lifeline and record-to-record noise: plume visits and starvation
    base = 5e-6 * np.exp(rng.normal(0.0, 0.8, size=(1, k)))
    walk = np.cumsum(rng.normal(0.0, 0.05, size=(n, k)), axis=0)
    cs = base * np.exp(walk + rng.normal(0.0, 0.3, size=(n, k)))
    cs[n // 2:n // 2 + 40, 0] = 0.0                         # a starvation episode
    cpaa = np.full_like(cs, 2.3e-3)
    return time, cs, cpaa


def run(module, p, tmp, tag):
    time, cs, cpaa = synthetic()
    initial = dict(module.INITIAL, xbio=55.0)
    series = module.integrate(time, cs, cpaa, initial, p, 0.03)
    out = pathlib.Path(tmp) / f"stitch_{tag}.npz"
    module.stitch(time, cs, cpaa, initial, p, 0.02, 50, 0.0, 1, 0.03, 0.005, str(out))
    with np.load(out) as data:
        stitched = {key: data[key] for key in data.files}
    return series, stitched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", default=None, help="the script before the option; outputs with the option off must match bitwise")
    arguments = parser.parse_args()
    p = dict(new_module.NINEPOOL_DEFAULTS)
    p_frozen = dict(p, freeze_e11=True)
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        series_off, stitched_off = run(new_module, p, tmp, "off")
        series_on, stitched_on = run(new_module, p_frozen, tmp, "on")
        e11_0 = new_module.INITIAL["e11"]
        # on: E11 constant everywhere, other pools still move
        if not np.all(series_on["e11"] == e11_0):
            failures.append(f"integrate on: e11 moved, range {series_on['e11'].min():.6g}..{series_on['e11'].max():.6g}")
        if not np.all(stitched_on["final_e11"] == e11_0):
            failures.append("stitch on: final e11 moved")
        header = list(stitched_on["header"])
        e11_cols = [header.index(f"e11_{s}") for s in ("mean", "p5", "p50", "p95")]
        if not np.all(stitched_on["table"][:, e11_cols] == e11_0):
            failures.append("stitch on: reported e11 statistics moved")
        if stitched_on["table"][-1, header.index("ref_e11")] != e11_0:
            failures.append("stitch on: reference cell e11 moved")
        if not bool(stitched_on["freeze_e11"]) or bool(stitched_off["freeze_e11"]):
            failures.append("npz flag freeze_e11 wrong")
        if np.ptp(series_on["gly"]) == 0.0 or np.ptp(stitched_on["final_gly"]) == 0.0:
            failures.append("on: the other pools did not move")
        # off: E11 drifts (its time constant ~25 h, so only slightly in 72 s, but monotonically)
        if np.ptp(series_off["e11"]) == 0.0 or np.all(stitched_off["final_e11"] == e11_0):
            failures.append("off: e11 did not move")
        print(f"option on : e11 {series_on['e11'].min():.6g}..{series_on['e11'].max():.6g} (integrate), final e11 "
              f"{stitched_on['final_e11'].min():.6g}..{stitched_on['final_e11'].max():.6g} (stitch); gly final "
              f"{stitched_on['final_gly'].mean():.4f}, q_p {stitched_on['final_qp'].mean():.4e}")
        print(f"option off: e11 {series_off['e11'].min():.6g}..{series_off['e11'].max():.6g} (integrate), final e11 "
              f"{stitched_off['final_e11'].min():.6g}..{stitched_off['final_e11'].max():.6g} (stitch); gly final "
              f"{stitched_off['final_gly'].mean():.4f}, q_p {stitched_off['final_qp'].mean():.4e}")
        if arguments.old:
            spec = importlib.util.spec_from_file_location("old_integrator", arguments.old)
            old_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(old_module)
            series_old, stitched_old = run(old_module, dict(old_module.NINEPOOL_DEFAULTS), tmp, "old")
            for key in series_old:
                if not np.array_equal(series_old[key], series_off[key]):
                    failures.append(f"integrate off vs old: {key} differs")
            for key in stitched_old:
                if key == "header":
                    same = list(stitched_old[key]) == list(stitched_off[key])
                else:
                    same = np.array_equal(stitched_old[key], stitched_off[key])
                if not same:
                    failures.append(f"stitch off vs old: {key} differs")
            extra = set(stitched_off) - set(stitched_old)
            print(f"option off vs old script: {len(series_old)} series and {len(stitched_old)} stitch arrays compared bitwise"
                  + (f"; new keys {sorted(extra)}" if extra else ""))
    for line in failures:
        print("FAIL:", line)
    print("FAILED" if failures else "OK")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
