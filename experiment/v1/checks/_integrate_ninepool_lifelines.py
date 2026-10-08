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


def stitch(time, cs, cpaa, initial, p, hours, count, segment_seconds, seed, step_seconds, report_hours, out_path):
    """Haringa et al. 2018 (2026-10-09): long lifelines by joining randomly chosen recorded lifelines (or random
    segments of them) of a statistically steady run, exploiting that the extra-cellular statistics are stationary.
    `count` synthetic cells are integrated on the fly through `hours` hours; a segment is one whole recorded window
    (segment_seconds 0) or a random window of segment_seconds starting at a random record. The reference is the same
    cell driven by the constant population-mean C_s (and C_PAA): the difference is the effect of the fluctuations at
    fixed mean. Reports the population mean / p5 / p50 / p95 of gly, mu, q_p, e11, sto, atp every report_hours and
    saves them (and the final state of every cell) to out_path."""
    rng = np.random.default_rng(seed)
    n_records, n_lifelines = cs.shape
    interval = float(np.median(np.diff(time)))
    seg = n_records if segment_seconds <= 0 else max(2, int(round(segment_seconds / interval)))
    seg = min(seg, n_records)
    sub = max(1, int(np.ceil(interval / step_seconds)))
    dt_h = interval / sub / 3600.0
    total_records = int(round(hours * 3600.0 / interval))
    state = {key: np.full(count, float(initial[key])) for key in STATE}
    reference = {key: np.array([float(initial[key])]) for key in STATE}
    # Ideally mixed reference: a cell at the constant C_s whose Monod saturation equals the population mean of
    # C_s / (C_s + K_s) over the records, i.e. the same mean uptake capacity use (the mean C_s itself is dominated by
    # the feed plume and would overfeed the reference: Jensen with the saturating Monod term).
    saturation = np.nanmean(cs / (cs + p["Ks11"]))
    cs_mean = np.array([p["Ks11"] * saturation / max(1.0 - saturation, 1e-12)])
    cpaa_mean = np.array([np.nanmean(cpaa)])
    report_every = max(1, int(round(report_hours * 3600.0 / interval)))
    rows, labels = [], ("gly", "mu", "q_p", "e11", "sto", "atp")
    print(f"stitching {count} cells x {hours:g} h from {n_lifelines} lifelines of {n_records} records ({interval:.4f} s): "
          f"segments of {seg} records, Euler step {interval / sub:.4f} s, {total_records:,} records in total; "
          f"reference cell at the constant C_s of equal mean saturation {cs_mean[0]:.3e} mol/kg "
          f"(mean saturation {saturation:.3f}; the arithmetic mean C_s is {np.nanmean(cs):.3e})")
    done = 0
    mu = qp = atp = np.zeros(count)
    next_report = report_every
    reported = -1
    while done < total_records:
        length = min(seg, total_records - done)
        k = rng.integers(0, n_lifelines, size=count)
        start = rng.integers(0, n_records - length + 1, size=count) if length < n_records else np.zeros(count, dtype=np.int64)
        rows_index = start[None, :] + np.arange(length)[:, None]
        cs_seg = cs[rows_index, k[None, :]]
        cpaa_seg = cpaa[rows_index, k[None, :]]
        cs_seg = np.where(np.isfinite(cs_seg), cs_seg, cs_mean[0])      # lost particles: mean value
        cpaa_seg = np.where(np.isfinite(cpaa_seg), cpaa_seg, cpaa_mean[0])
        for i in range(length):
            c_next = cs_seg[i]
            c_prev = cs_seg[i - 1] if i > 0 else c_next
            pa_next, pa_prev = cpaa_seg[i], (cpaa_seg[i - 1] if i > 0 else cpaa_seg[i])
            for s in range(sub):
                w = (s + 0.5) / sub
                state, mu, qp, atp = step(state, c_prev * (1.0 - w) + c_next * w, pa_prev * (1.0 - w) + pa_next * w, p, dt_h)
                reference, mu_ref, qp_ref, atp_ref = step(reference, cs_mean, cpaa_mean, p, dt_h)
            done += 1
            if (done >= next_report or done == total_records) and done != reported:
                reported = done
                values = {"gly": state["gly"], "mu": mu, "q_p": qp, "e11": state["e11"], "sto": state["sto"], "atp": atp}
                row = [done * interval / 3600.0]
                for key in labels:
                    q = np.percentile(values[key], [5, 50, 95])
                    row += [float(values[key].mean()), q[0], q[1], q[2]]
                row += [float(reference["gly"][0]), float(mu_ref[0]), float(qp_ref[0]), float(reference["e11"][0])]
                rows.append(row)
                print(f"   t {row[0]:6.2f} h: gly {row[1]:6.2f} [{row[2]:5.2f} {row[3]:5.2f} {row[4]:5.2f}]  mu {row[5]:.4f}  "
                      f"q_p {row[9]:.3e} [{row[10]:.3e} .. {row[12]:.3e}]  e11 {row[13]:.4f}  | reference gly {row[-4]:.2f} "
                      f"mu {row[-3]:.4f} q_p {row[-2]:.3e} e11 {row[-1]:.4f}  -> q_p loss {100 * (row[9] / row[-2] - 1):+.1f} %", flush=True)
                next_report += report_every
    header = ["hours"] + [f"{key}_{stat}" for key in labels for stat in ("mean", "p5", "p50", "p95")] + \
             ["ref_gly", "ref_mu", "ref_qp", "ref_e11"]
    np.savez_compressed(out_path, table=np.array(rows), header=np.array(header),
                        **{f"final_{key}": state[key] for key in STATE}, final_mu=mu, final_qp=qp, final_atp=atp,
                        cs_mean=cs_mean, seed=seed, segment_records=seg, interval=interval)
    print(f"wrote {out_path}")


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
                             "(1 + X_gly,ref / K_i) so that the chemostat steady state is unchanged")
    parser.add_argument("--ki-reference-gly", type=float, default=None,
                        help="X_gly,ref of the k_E11 rescale (default: the initial gly). Pass the STEADY-STATE X_gly (the "
                             "generator's X_gly0, e.g. 24.6) when the integration starts from a non-steady state such as "
                             "the starved pools at the feed start of B1, otherwise k_E11 differs from the GPU case")
    parser.add_argument("--init-from-record", action="store_true", help="start from the recorded pools of the first record")
    parser.add_argument("--prefeed-probes", default=None, metavar="PROBES.CSV",
                        help="pre-lifeline phase (2026-10-08): before the first lifeline record the field is uniform (no "
                             "feed yet or a uniform start), so integrate the initial pools along the probe log's mean C_s "
                             "(total:substrate / fluid_mass) up to the first record; the B1 runs starved their cells in the "
                             "40 s of flow development (X_gly 24.6 -> 7.7) and the offline pools must start from that state")
    parser.add_argument("--prefeed-cs0", type=float, default=None,
                        help="C_s at t = 0 for --prefeed-probes (default: the first probe row)")
    parser.add_argument("--check", action="store_true", help="compare with the recorded gly / growth_rate / pen_capacity")
    parser.add_argument("--start", type=float, default=None, help="use records from this time on, s")
    parser.add_argument("--stitch-hours", type=float, default=None,
                        help="Haringa 2018 stitching (2026-10-09): instead of integrating the recorded lifelines, join randomly "
                             "chosen recorded lifelines into --stitch-count synthetic lifelines of this many hours and integrate "
                             "the pools on the fly (use --start to keep the statistically steady part; --step 0.03 is enough)")
    parser.add_argument("--stitch-count", type=int, default=2500)
    parser.add_argument("--stitch-segment", type=float, default=0.0, help="segment length, s (0 = one whole recorded window)")
    parser.add_argument("--stitch-seed", type=int, default=1)
    parser.add_argument("--stitch-report", type=float, default=1.0, help="report interval, hours")
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
        reference = float(initial["gly"]) if arguments.ki_reference_gly is None else arguments.ki_reference_gly
        p["Ki11"] = arguments.ki
        p["kE11"] = p["kE11"] * (1.0 + reference / arguments.ki)
        print(f"uptake-inhibition knob: Ki11 {arguments.ki:g} umol/gdw, k_E11 scaled to {p['kE11']:.4f} (X_gly,ref {reference:g})")
    if arguments.init_from_record:
        for key, name in (("gly", "gly"), ("aa", "aa"), ("sto", "sto"), ("paa", "paa_pool"), ("e11", "e11"), ("e32", "e32"),
                          ("e4", "e4"), ("v33", "pen_capacity"), ("xbio", "biomass")):
            if name in names:
                initial[key] = scalars[0, :, names.index(name)]
    if arguments.prefeed_probes:
        with open(arguments.prefeed_probes, encoding="utf-8") as handle:
            header = handle.readline().strip().split(",")
        rows = np.loadtxt(arguments.prefeed_probes, delimiter=",", skiprows=1, ndmin=2)
        probe_time = rows[:, header.index("time")]
        probe_cs = rows[:, header.index("total:substrate")] / rows[:, header.index("fluid_mass")]
        before = probe_time < time[0]
        cs0 = probe_cs[before][0] if arguments.prefeed_cs0 is None else arguments.prefeed_cs0
        pre_time = np.concatenate([[0.0], probe_time[before], [time[0]]])
        pre_cs = np.concatenate([[cs0], probe_cs[before], [probe_cs[before][-1]]])
        state = {key: np.array([float(initial[key])]) for key in STATE}
        for i in range(1, pre_time.size):
            span = pre_time[i] - pre_time[i - 1]
            sub = max(1, int(np.ceil(span / arguments.step)))
            dt_h = span / sub / 3600.0
            for s in range(sub):
                w = (s + 0.5) / sub
                state, mu, _, _ = step(state, np.array([pre_cs[i - 1] * (1.0 - w) + pre_cs[i] * w]), np.array([arguments.paa]), p, dt_h)
        for key in STATE:
            initial[key] = float(state[key][0])
        print(f"pre-lifeline phase 0..{time[0]:.2f} s along the probe mean C_s ({before.sum()} rows, C_s {cs0:.3e} -> "
              f"{pre_cs[-1]:.3e}): pools at the first record " + ", ".join(f"{k} {initial[k]:.5g}" for k in STATE)
              + f", mu {float(mu[0]):.4f} 1/h")
    print(f"{cs.shape[1]} lifelines, {cs.shape[0]} records, {time[0]:.2f}..{time[-1]:.2f} s, Euler step <= {arguments.step} s")
    if arguments.stitch_hours:
        stitch(time, cs, cpaa, initial, p, arguments.stitch_hours, arguments.stitch_count, arguments.stitch_segment,
               arguments.stitch_seed, arguments.step, arguments.stitch_report, arguments.out)
        return 0
    out = integrate(time, cs, cpaa, initial, p, arguments.step)
    np.savez_compressed(arguments.out, time=time, uid=data["uid"], **out)
    late = time >= time[0] + 10.0
    if not late.any():          # short test runs
        late[:] = True
    print(f"distributions over records >= {time[late][0] - time[0]:.1f} s after the start and all lifelines:")
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
