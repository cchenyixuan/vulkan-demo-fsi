"""_integrate_ninepool_lifelines.py — the one-way protocol (Haringa et al. 2018): integrate the 9-pool cell
model of Tang et al. 2017 offline along recorded lifelines, with the recorded extracellular glucose C_s(t) (and
C_PAA(t) when recorded) as the forcing (2026-10-07, stage 5).

Per lifeline the cell state (X_gly, X_AA, X_sto, X_PAA, X_E11, X_E32, X_E4, v33, x_bio) is advanced with forward
Euler, sub-stepping each record interval (~0.03 s) so that the step is <= --step (default 0.01 s). The rates and
parameters are those of experiment/v1/checks/_model_ninepool.py (vectorised copy here), the same as the GPU's
REACTION_MODE 3 (predict.comp), including the algebraic ATP. The cell does not feed back on C_s: that is the
one-way assumption. Output: OUT.npz with time, uid and the series gly, mu, q_p, sto, atp (records x lifelines),
and a summary of the distributions.

Cross-check (--check): on a TWO-WAY run the recorded fields gly / growth_rate / pen_capacity must be reproduced
when the integration starts from the recorded first values (--init-from-record), because the GPU integrated the
same equations with the same C_s; the difference is the time-stepping (GPU dt ~1e-4 s vs 0.01 s here) and the
0.03 s sampling of C_s.

usage:
    python experiment/v1/checks/_integrate_ninepool_lifelines.py DIR --out OUT.npz [--init-from-record] [--check]
        [--step 0.01] [--paa 2.3e-3] [--init gly=20,aa=928,...] [--param qE11max=1.65e-2,...]
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
from utils.sph.case import NINEPOOL_DEFAULTS                       # noqa: E402

INITIAL = dict(gly=20.0, aa=928.0, sto=2650.0, paa=2.7, e11=0.131, e32=142.0, e4=0.034, v33=4.4e-4)   # generator NINEPOOL_INITIAL
STATE = ("gly", "aa", "sto", "paa", "e11", "e32", "e4", "v33", "xbio")


def hill(x, k, n):
    xn = x ** n
    return xn / (k ** n + xn)


def step(state, cs, cpaa, p, dt_h):
    """one forward-Euler step of dt_h hours for arrays of cells; state is a dict of arrays; returns mu, q_p"""
    gly, aa, sto, paa = (np.maximum(state[k], 0.0) for k in ("gly", "aa", "sto", "paa"))
    e11, e32, e4, v33, xbio = (np.maximum(state[k], 0.0) for k in ("e11", "e32", "e4", "v33", "xbio"))
    cs = np.maximum(cs, 0.0)
    atp = p["ATP_A"] * gly ** 3 / (gly ** 3 + p["ATP_B"] ** 3)
    v11 = p["kE11"] * e11 * cs / (cs + p["Ks11"]) / (1.0 + gly / p.get("Ki11", np.inf))   # knob (2026-10-08), inf = off
    v12 = p["v12max"] * hill(gly, p["Kgly12"], 2) * (1.0 - hill(aa, p["KAA12"], 2)) * hill(atp, p["KATP12"], 3)
    v13 = p["v13max"] * hill(gly, p["Kgly13"], 2) * hill(aa, p["KAA13"], 2) * hill(atp, p["KATP13"], 3)
    v21 = p["v21max"] * hill(gly, p["Kgly21"], 3) * (1.0 - hill(atp, p["KATP21"], 4))
    f_ext = 1.0 / (1.0 + 10.0 ** (p["pHext"] - p["pKPAA"]))
    f_int = 1.0 / (1.0 + 10.0 ** (p["pHint"] - p["pKPAA"]))
    v31 = p["kperm31"] * p["acell"] * (cpaa * p["rho"] * f_ext - paa / 2.5 * f_int)
    v32 = e32 * paa * p["Mw"] * 1e-6
    cs41 = cs / (cs + p["Ks41"])
    cs42 = cs / (cs + p["Ks42"])
    v41 = p["k41"] * e4 * cs41 * (1.0 + 2.0 * cs42) * p["Ksto41"] / (sto + p["Ksto41"])
    v42 = p["k42"] * e4 * (1.0 - cs42) * (1.0 + 2.0 * cs41) * hill(sto, p["Ksto42"], 2) * hill(atp, p["KATP42"], 2)
    mu = v13
    scale = 1e6 / p["Mw"] * dt_h
    mu_dt = mu * dt_h
    new = dict(state)
    new["gly"] = gly + scale * (6.0 * v11 - v12 - 0.578 * v13 - v21 - 4.81 * v33 - 1.07 * v41 + v42) - mu_dt * gly
    new["aa"] = aa + scale * (v12 - 0.5 * v13 - 6.25 * v33) - mu_dt * aa
    new["sto"] = sto + scale * (v41 - v42) - mu_dt * sto
    new["paa"] = paa + scale * (v31 - v32 - v33) - mu_dt * paa
    r11 = ((mu + p["mu0"]) / p["k11"]) ** 5
    new["e11"] = e11 + p["qE11max"] * r11 / (1.0 + r11) * dt_h - (mu_dt + p["kdE11"] * dt_h) * e11
    new["e32"] = e32 + (p["alpha32"] + p["beta32"] * mu) * dt_h - (p["kdE32"] * dt_h + mu_dt) * e32
    new["e4"] = e4 + (p["alpha4"] + p["beta4"] * mu) * dt_h - (p["kdE4"] * dt_h + mu_dt) * e4
    new["v33"] = v33 + p["beta33"] * mu / (1.0 + (gly / p["Kgly33"]) ** p["m33"]) * dt_h - (p["kdE33"] * dt_h + mu_dt) * v33
    new["xbio"] = xbio + (mu - p["vd"]) * dt_h * xbio
    return new, mu, v33, atp


def integrate(time, cs, cpaa, initial, p, step_seconds):
    """time (n,), cs / cpaa (n, K); returns dict of (n, K) series for gly, mu, q_p, sto, atp, e11"""
    n, k = cs.shape
    state = {key: np.full(k, float(initial[key])) if np.isscalar(initial[key]) else np.array(initial[key], dtype=float)
             for key in STATE}
    out = {key: np.zeros((n, k)) for key in ("gly", "mu", "q_p", "sto", "atp", "e11")}
    _, mu, qp, atp = step(state, cs[0], cpaa[0], p, 0.0)
    out["gly"][0], out["mu"][0], out["q_p"][0], out["sto"][0], out["atp"][0], out["e11"][0] = state["gly"], mu, qp, state["sto"], atp, state["e11"]
    for i in range(1, n):
        span = time[i] - time[i - 1]
        sub = max(1, int(np.ceil(span / step_seconds)))
        dt_h = span / sub / 3600.0
        for s in range(sub):
            w = (s + 0.5) / sub                                   # C_s interpolated linearly between records
            cs_now = cs[i - 1] * (1.0 - w) + cs[i] * w
            cpaa_now = cpaa[i - 1] * (1.0 - w) + cpaa[i] * w
            state, mu, qp, atp = step(state, cs_now, cpaa_now, p, dt_h)
        out["gly"][i], out["mu"][i], out["q_p"][i], out["sto"][i], out["atp"][i], out["e11"][i] = state["gly"], mu, qp, state["sto"], atp, state["e11"]
    return out


def parse_assignments(text):
    if not text:
        return {}
    return {k.strip(): float(v) for k, v in (item.split("=") for item in text.split(","))}


def describe(label, values):
    flat = values[np.isfinite(values)]
    q = np.percentile(flat, [5, 50, 95])
    print(f"   {label:14s} mean {flat.mean():.4g}  p5 {q[0]:.4g}  p50 {q[1]:.4g}  p95 {q[2]:.4g}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("dir")
    parser.add_argument("--out", required=True)
    parser.add_argument("--step", type=float, default=0.01, help="largest Euler step, s")
    parser.add_argument("--paa", type=float, default=2.3e-3, help="C_PAA when not recorded, mol/kg")
    parser.add_argument("--init", default="", help="initial pools, e.g. gly=20,aa=928 (default: generator values)")
    parser.add_argument("--xbio", type=float, default=55.0, help="initial x_bio, gdw/kg")
    parser.add_argument("--param", default="", help="parameter overrides, e.g. qE11max=1.65e-2")
    parser.add_argument("--ki", type=float, default=None,
                        help="uptake-inhibition knob K_i (umol/gdw, 2026-10-08): v11 /= 1 + X_gly / K_i, with k_E11 scaled by "
                             "(1 + X_gly0 / K_i) so that the chemostat steady state is unchanged (X_gly0 = initial gly)")
    parser.add_argument("--init-from-record", action="store_true", help="start from the recorded pools of the first record")
    parser.add_argument("--check", action="store_true", help="compare with the recorded gly / growth_rate / pen_capacity")
    parser.add_argument("--start", type=float, default=None, help="use records from this time on, s")
    arguments = parser.parse_args()
    data = load_lifelines(arguments.dir, with_aux=False)
    names = data["field_names"]
    time = data["time"].astype(np.float64)
    scalars = data["scalars"].astype(np.float64)
    if arguments.start is not None:
        keep = time >= arguments.start
        time, scalars = time[keep], scalars[keep]
    cs = scalars[:, :, names.index("substrate")]
    cpaa = scalars[:, :, names.index("paa_ext")] if "paa_ext" in names else np.full_like(cs, arguments.paa)
    p = dict(NINEPOOL_DEFAULTS)
    p.update(parse_assignments(arguments.param))
    initial = dict(INITIAL, xbio=arguments.xbio)
    initial.update(parse_assignments(arguments.init))
    if arguments.ki is not None:
        p["Ki11"] = arguments.ki
        p["kE11"] = p["kE11"] * (1.0 + float(initial["gly"]) / arguments.ki)
        print(f"uptake-inhibition knob: Ki11 {arguments.ki:g} umol/gdw, k_E11 scaled to {p['kE11']:.4f} (X_gly0 {initial['gly']:g})")
    if arguments.init_from_record:
        for key, name in (("gly", "gly"), ("aa", "aa"), ("sto", "sto"), ("paa", "paa_pool"), ("e11", "e11"), ("e32", "e32"),
                          ("e4", "e4"), ("v33", "pen_capacity"), ("xbio", "biomass")):
            if name in names:
                initial[key] = scalars[0, :, names.index(name)]
    print(f"{cs.shape[1]} lifelines, {cs.shape[0]} records, {time[0]:.2f}..{time[-1]:.2f} s, Euler step <= {arguments.step} s")
    out = integrate(time, cs, cpaa, initial, p, arguments.step)
    np.savez_compressed(arguments.out, time=time, uid=data["uid"], **out)
    print("distributions over records >= 10 s after the start and all lifelines:")
    late = time >= time[0] + 10.0
    for key in ("gly", "mu", "q_p", "sto", "atp"):
        describe(key, out[key][late])
    if arguments.check:
        for key, name in (("gly", "gly"), ("mu", "growth_rate"), ("q_p", "pen_capacity"), ("sto", "sto")):
            if name not in names:
                continue
            recorded = scalars[:, :, names.index(name)]
            valid = late[:, None] & np.isfinite(recorded) & (recorded != 0.0)
            difference = np.abs(out[key] - recorded)[valid]
            scale = np.abs(recorded[valid]).max()
            print(f"   check {key:5s} vs recorded {name:12s}: max |diff| {difference.max():.3e} ({difference.max() / scale:.2e} of max), "
                  f"rms {np.sqrt((difference ** 2).mean()):.3e}")
    print(f"wrote {arguments.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
