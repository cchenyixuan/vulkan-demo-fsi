"""_check_ninepool_box.py — verification of the 9-pool cell model on the GPU (2026-10-07, stage 5).

A closed 4 mm box of fluid at rest (as _check_reaction_box.py), 15 scalar fields in 4 vec4:
    C_s, C_PAA, uptake, feed | X_gly, X_AA, X_sto, X_PAA | X_E11, X_E32, X_E4, v33 | x_bio, product, mu
Every particle starts from the chemostat steady state of the 0-D model (D = 0.05 1/h, _model_ninepool.py).

  N1  glucose pulse: C_s set to 350 umol/kg everywhere, 360 s. The particle means must follow an RK4
      integration (own, numpy) of _model_ninepool.rhs with D = 0: C_s, X_gly, X_sto, mu, q_p, x_bio within
      a relative tolerance (forward Euler on the GPU, dt ~ 1e-3 s against a GLY timescale of 10 s).
      All particles stay equal (no transport at rest).
  N2  budgets of N1: sum m (C_s + uptake) constant (no feed); PAA: sum m (C_PAA + x_bio X_PAA 1e-6 + product)
      constant (penicillin takes one PAA); x_bio against the RK4.
  N3  fed box: glucose source sphere in the centre (rate given) from 0.5 s, 60 s: sum m (C_s + uptake - feed)
      constant, fed amount = rate t N_in dx^3 / V, C_s >= 0, pools >= 0.

Usage (repo root, solver env):
    python experiment/v1/checks/_check_ninepool_box.py [--only N1,N2,N3] [--out output/ninepool_checks] [--seconds 360]
"""
import argparse
import json
import math
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _check_reaction_box as box                 # noqa: E402  (build_case, run_box, REST_DENSITY)
import _model_ninepool as model                   # noqa: E402

# chemostat steady state of the 0-D model at D = 0.05 1/h (C_f 0.0895, C_PAA,f 5e-3; _model_ninepool.py)
STEADY = dict(gly=24.6, aa=931.0, sto=2652.0, paa=2.71, e11=0.146, e32=211.9, e4=0.048, v33=4.06e-4,
              Cs=9.66e-6, CPAA=2.17e-3, Cx=6.65)
NAMES = ["substrate", "paa_ext", "uptake", "feed", "gly", "aa", "sto", "paa_pool", "e11", "e32", "e4",
         "pen_capacity", "biomass", "product", "growth_rate"]
COLUMN = {name: index for index, name in enumerate(NAMES)}
REACTION = {"type": "ninepool", "substrate": "substrate", "paa": "paa_ext", "uptake": "uptake", "product": "product",
            "growth_rate": "growth_rate", "biomass": "biomass", "gly": "gly", "aa": "aa", "sto": "sto",
            "paa_pool": "paa_pool", "e11": "e11", "e32": "e32", "e4": "e4", "pen_capacity": "pen_capacity",
            "q_max": 0.0, "k_s": 1.0}


def fields(cs, cpaa, x_bio, state=STEADY):
    initial = {"substrate": cs, "paa_ext": cpaa, "uptake": 0.0, "feed": 0.0, "gly": state["gly"], "aa": state["aa"],
               "sto": state["sto"], "paa_pool": state["paa"], "e11": state["e11"], "e32": state["e32"],
               "e4": state["e4"], "pen_capacity": state["v33"], "biomass": x_bio, "product": 0.0, "growth_rate": 0.0}
    out = []
    for name in NAMES:
        diffusing = name in ("substrate", "paa_ext")
        out.append({"name": name, "diffusivity": 6.0e-10 if diffusing else 0.0, "turbulent": False,
                    "initial": float(initial[name])})
    return out


def rk4(y0, seconds, samples):
    """numpy RK4 of the 0-D model (D = 0, no feed), state order of _model_ninepool.rhs; returns (times_s, Y)"""
    y = np.array(y0, dtype=np.float64)
    out = [y.copy()]
    hours = seconds / 3600.0
    step = 0.01 / 3600.0
    per_sample = max(1, int(round(hours / samples / step)))
    args = (0.0, 0.0, 0.0, lambda t: 0.0)
    t = 0.0
    for _ in range(samples):
        for _ in range(per_sample):
            k1 = np.array(model.rhs(t, y, *args)); k2 = np.array(model.rhs(t + 0.5 * step, y + 0.5 * step * k1, *args))
            k3 = np.array(model.rhs(t + 0.5 * step, y + 0.5 * step * k2, *args)); k4 = np.array(model.rhs(t + step, y + step * k3, *args))
            y = y + step / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4); t += step
        out.append(y.copy())
    return np.linspace(0.0, seconds, samples + 1), np.asarray(out)


def reference_state(cs, cpaa, x_bio, state=STEADY):
    return [state["gly"], state["aa"], state["sto"], state["paa"], state["e11"], state["e32"], state["e4"],
            state["v33"], 0.0, cs, cpaa, x_bio]


def means(series, name):
    return np.array([s["values"][:, COLUMN[name]].mean() for s in series])


def check_n1n2(out, seconds, do_n2):
    cs0, cpaa0, x0 = 350e-6, STEADY["CPAA"], STEADY["Cx"]
    case_path, _ = box.build_case(out / "n1", c0=2.0, fields=fields(cs0, cpaa0, x0), reaction=REACTION)
    from utils.sph.case import load_case
    dt = load_case(str(case_path)).timestep
    steps = int(round(seconds / dt))
    samples = 36
    case, series, status = box.run_box(case_path, steps, samples)
    times = np.array([s["time"] for s in series])
    ref_t, ref = rk4(reference_state(cs0, cpaa0, x0), seconds, samples)
    # reference columns of model.rhs state: 0 gly 1 aa 2 sto 3 paa 4 e11 5 e32 6 e4 7 v33 8 atp 9 Cs 10 CPAA 11 Cx
    pairs = (("substrate", 9), ("gly", 0), ("sto", 2), ("paa_pool", 3), ("pen_capacity", 7), ("biomass", 11), ("paa_ext", 10))
    gpu = {name: means(series, name) for name, _ in pairs}
    mu_gpu = means(series, "growth_rate")
    mu_ref = np.array([model.rates(ref[k, :9], ref[k, 9], ref[k, 10])[0][2] for k in range(ref.shape[0])])
    spread = max(float(series[-1]["values"][:, COLUMN[name]].max() - series[-1]["values"][:, COLUMN[name]].min())
                 / max(abs(gpu[name][-1]), 1e-30) for name, _ in pairs[:5])
    errors = {}
    for name, column in pairs:
        reference = ref[:, column]
        scale = np.maximum(np.abs(reference), 1e-3 * np.abs(reference).max() + 1e-30)
        errors[name] = float(np.abs(gpu[name] - reference).max() / np.abs(reference).max())
    errors["mu"] = float(np.abs(mu_gpu[1:] - mu_ref[1:]).max() / np.abs(mu_ref).max())
    print(f"N1 glucose pulse: dt {dt:.3e} s, {steps} steps, {seconds:.0f} s; GPU particle means vs RK4 of the 0-D model")
    for k in range(0, len(times), max(1, len(times) // 9)):
        print(f"   t {times[k]:6.1f} s: C_s {gpu['substrate'][k]*1e6:7.2f} ({ref[k, 9]*1e6:7.2f}) umol/kg  X_gly {gpu['gly'][k]:6.2f} ({ref[k, 0]:6.2f})  "
              f"X_sto {gpu['sto'][k]:7.1f} ({ref[k, 2]:7.1f})  mu {mu_gpu[k]:.4f} ({mu_ref[k]:.4f})  q_p {gpu['pen_capacity'][k]:.3e} ({ref[k, 7]:.3e})  "
              f"x_bio {gpu['biomass'][k]:.4f} ({ref[k, 11]:.4f})  X_PAA {gpu['paa_pool'][k]:.4f} ({ref[k, 3]:.4f})  C_PAA {gpu['paa_ext'][k]*1e3:.5f} ({ref[k, 10]*1e3:.5f}) mmol/kg")
    worst = max(errors.values())
    passed = worst < 5e-3 and spread < 1e-5
    print("   max |GPU - RK4| / max|RK4|: " + ", ".join(f"{k} {v:.2e}" for k, v in errors.items())
          + f"; particle spread {spread:.1e}; overflow {status['overflow_inside_count']}/{status['overflow_incoming_count']} -> {'PASS' if passed else 'FAIL'}")
    results = [{"name": "N1", "passed": bool(passed), "errors": errors, "spread": float(spread)}]
    if do_n2:
        glucose = np.array([np.sum(s["mass"] * (s["values"][:, COLUMN["substrate"]] + s["values"][:, COLUMN["uptake"]])) for s in series])
        paa = np.array([np.sum(s["mass"] * (s["values"][:, COLUMN["paa_ext"]]
                                             + s["values"][:, COLUMN["biomass"]] * s["values"][:, COLUMN["paa_pool"]] * 1e-6
                                             + s["values"][:, COLUMN["product"]])) for s in series])
        drift_glucose = abs(glucose[-1] - glucose[0]) / abs(glucose[0])
        drift_paa = abs(paa[-1] - paa[0]) / abs(paa[0])
        passed2 = drift_glucose < 1e-6 and drift_paa < 1e-4
        print(f"N2 budgets: sum m (C_s + U) {glucose[0]:.9e} -> {glucose[-1]:.9e} (drift {drift_glucose:.1e}); "
              f"PAA ext + cell + penicillin {paa[0]:.9e} -> {paa[-1]:.9e} (drift {drift_paa:.1e}) -> {'PASS' if passed2 else 'FAIL'}")
        results.append({"name": "N2", "passed": bool(passed2), "drift_glucose": float(drift_glucose), "drift_paa": float(drift_paa)})
    return results


def check_n3(out, seconds):
    radius, rate, dx = 0.015, 2.0e-8, 0.004
    source = {"field": "substrate", "center": [0.0, 0.0, 0.0], "radius": radius, "rate": rate, "start": 0.5, "record": "feed"}
    case_path, fluid = box.build_case(out / "n3", dx=dx, c0=2.0, fields=fields(STEADY["Cs"], STEADY["CPAA"], STEADY["Cx"]),
                                      reaction=REACTION, sources=[source])
    from utils.sph.case import load_case
    dt = load_case(str(case_path)).timestep
    steps = int(round(seconds / dt))
    case, series, status = box.run_box(case_path, steps, 12)
    times = np.array([s["time"] for s in series])
    inside = int((np.linalg.norm(fluid, axis=1) < radius).sum())
    volume = 4.0 / 3.0 * math.pi * radius ** 3
    budget = np.array([np.sum(s["mass"] * (s["values"][:, COLUMN["substrate"]] + s["values"][:, COLUMN["uptake"]] - s["values"][:, COLUMN["feed"]])) for s in series])
    fed = np.array([box.REST_DENSITY * dx ** 3 * s["values"][:, COLUMN["feed"]].sum() for s in series])
    expected = rate * np.maximum(times - 0.5, 0.0) * inside * dx ** 3 / volume
    minima = {name: min(float(s["values"][:, COLUMN[name]].min()) for s in series) for name in ("substrate", "gly", "sto", "paa_pool", "e11")}
    print(f"N3 fed box: dt {dt:.3e} s, {steps} steps, {inside} particles in the source sphere")
    for k in range(0, len(times), max(1, len(times) // 6)):
        print(f"   t {times[k]:6.2f}: fed {fed[k]:.6e} mol (expected {expected[k]:.6e})  budget {budget[k]:.9e}  "
              f"C_s in sphere max {series[k]['values'][np.linalg.norm(series[k]['positions'], axis=1) < radius, COLUMN['substrate']].max()*1e6:.2f} umol/kg")
    drift = abs(budget[-1] - budget[0]) / abs(budget[0])
    fed_error = abs(fed[-1] - expected[-1]) / expected[-1]
    passed = drift < 1e-6 and fed_error < 1e-4 and min(minima.values()) >= 0.0
    print(f"   budget drift {drift:.1e}, fed error {fed_error:.1e}, minima " + ", ".join(f"{k} {v:.2e}" for k, v in minima.items())
          + f" -> {'PASS' if passed else 'FAIL'}")
    return {"name": "N3", "passed": bool(passed), "budget_drift": float(drift), "fed_error": float(fed_error)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--only", default="N1,N2,N3")
    parser.add_argument("--out", default="output/ninepool_checks")
    parser.add_argument("--seconds", type=float, default=360.0, help="N1 duration (default 360 = one feast-famine cycle)")
    arguments = parser.parse_args()
    from experiment.v1 import compile_shaders_v1
    compile_shaders_v1.compile_v1_shaders()
    out = pathlib.Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    only = arguments.only.split(",")
    results = []
    if "N1" in only or "N2" in only:
        results += check_n1n2(out, arguments.seconds, "N2" in only)
    if "N3" in only:
        results.append(check_n3(out, 60.0))
    (out / "summary.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    return 0 if all(result["passed"] for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
