"""_integrate_state_limited_lifelines.py — the one-way protocol for the level-2 cell model `state_limited`
(2026-10-28, case 1): integrate the cell state mu offline along recorded lifelines, with the recorded C_s(t) as
the forcing. The cells do not feed back on C_s (one-way). Same equations as the GPU's REACTION_MODE 2
(predict.comp), per gram of biomass:

    supply   q_sup = q_max C_s / (K_s + C_s)
    demand   q_dem = (mu_d + alpha mu_max) / Y + m_s,       mu_max = Y (q_max - m_s)
    uptake   q = min(q_sup, q_dem);  maintenance min(q, m_s);  mu_act = Y (q - maintenance)
    state    mu <- (mu + r mu_act) / (1 + r),  r = dt / tau,  tau = tau_up if mu_act > mu else tau_down

mu_d is the cell's own mu (--demand-state own, the default and the one-way evaluation of individual cells) or the
per-record mean of the recorded growth_rate over the lifelines (--demand-state recorded-mean; reproduces a
`state_coupling: mean` run, whose demand used the biomass-weighted population mean written every 0.5 s). The GPU
supply is linearised implicitly in C_s over its step (dt ~1e-4 s, a correction of order 1e-4); one-way cells do
not deplete C_s, so the continuous rate is used here. C_s is interpolated linearly between records, implicit Euler
steps <= --step.

Cross-check (--check): on a run with the same coupling, the recorded growth_rate must be reproduced when the
integration starts from the recorded first values (--init-from-record); the difference is the time-stepping and the
sampling of C_s (0.1 s at 54 m3).

usage:
    python experiment/v1/checks/_integrate_state_limited_lifelines.py DIR --out OUT.npz [--init-from-record] [--check]
        [--demand-state own|recorded-mean] [--statistics-start 120] [--biomass 10] [--step 0.01] [--record-stride 1]
        [--q-max-umol-per-g-h 8300 --k-s 1e-5 --growth-yield 90 --maintenance-umol-per-g-h 222 --demand-margin 0.1
         --tau-up 20 --tau-down 20 --growth-rate-initial 6.94e-5]
"""
import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiment.v1.utils.lifeline_recorder import load_lifelines   # noqa: E402

PER_HOUR = 3600.0
UMOL_PER_G_H = 1.0e-6 / 3600.0          # umol/gdw/h -> mol/(g s)


def parameters_from(arguments):
    q_max = arguments.q_max_umol_per_g_h * UMOL_PER_G_H
    maintenance = arguments.maintenance_umol_per_g_h * UMOL_PER_G_H
    return dict(q_max=q_max, k_s=arguments.k_s, growth_yield=arguments.growth_yield, maintenance=maintenance,
                demand_margin=arguments.demand_margin, tau_up=arguments.tau_up, tau_down=arguments.tau_down,
                growth_rate_maximum=max(arguments.growth_yield * (q_max - maintenance), 0.0))


def rates(growth_rate, demand_state, substrate, p):
    """supply / demand / uptake per gram of biomass (mol/(g s)) and the realised growth rate (1/s)"""
    substrate = np.maximum(substrate, 0.0)
    supply = p["q_max"] * substrate / (p["k_s"] + substrate)
    demand = (np.maximum(demand_state, 0.0) + p["demand_margin"] * p["growth_rate_maximum"]) / p["growth_yield"] + p["maintenance"]
    uptake = np.minimum(supply, demand)
    growth_actual = p["growth_yield"] * (uptake - np.minimum(uptake, p["maintenance"]))
    return supply, demand, uptake, growth_actual


def relax(growth_rate, growth_actual, step_seconds, p):
    tau = np.where(growth_actual > growth_rate, p["tau_up"], p["tau_down"])
    ratio = step_seconds / tau
    return (growth_rate + ratio * growth_actual) / (1.0 + ratio)


def integrate(time, substrate, growth_rate_initial, demand_mean, p, step_seconds):
    """time (n,), substrate (n, K); demand_mean (n,) or None (own state); returns (n, K) series of mu (1/s),
    uptake q (mol/(g s)) and a demand-limited flag, all at the record times"""
    records, lifelines = substrate.shape
    growth_rate = np.broadcast_to(np.asarray(growth_rate_initial, dtype=np.float64), (lifelines,)).copy()
    out_growth_rate = np.zeros((records, lifelines), np.float32)
    out_uptake = np.zeros((records, lifelines), np.float32)
    out_limited = np.zeros((records, lifelines), bool)

    def store(index):
        state = growth_rate if demand_mean is None else np.full(lifelines, demand_mean[index])
        supply, demand, uptake, _ = rates(growth_rate, state, substrate[index], p)
        out_growth_rate[index], out_uptake[index], out_limited[index] = growth_rate, uptake, demand < supply

    store(0)
    for index in range(1, records):
        span = time[index] - time[index - 1]
        substeps = max(1, int(np.ceil(span / step_seconds)))
        step_now = span / substeps
        for substep in range(substeps):
            weight = (substep + 0.5) / substeps                       # C_s (and the mean state) linear between records
            substrate_now = substrate[index - 1] * (1.0 - weight) + substrate[index] * weight
            if demand_mean is None:
                state = growth_rate
            else:
                state = demand_mean[index - 1] * (1.0 - weight) + demand_mean[index] * weight
            _, _, _, growth_actual = rates(growth_rate, state, substrate_now, p)
            growth_rate = relax(growth_rate, growth_actual, step_now, p)
        store(index)
    return out_growth_rate, out_uptake, out_limited


def describe(label, values, scale=1.0):
    flat = values[np.isfinite(values)].astype(np.float64) * scale
    q = np.percentile(flat, [5, 50, 95])
    print(f"   {label:28s} mean {flat.mean():.4g}  p5 {q[0]:.4g}  p50 {q[1]:.4g}  p95 {q[2]:.4g}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dir")
    parser.add_argument("--out", required=True)
    parser.add_argument("--step", type=float, default=0.01, help="largest implicit Euler step, s")
    parser.add_argument("--record-stride", type=int, default=1, help="use every N-th record (sampling test)")
    parser.add_argument("--statistics-start", type=float, default=120.0, help="records from this time on enter the summary, s")
    parser.add_argument("--demand-state", choices=("own", "recorded-mean"), default="own")
    parser.add_argument("--init-from-record", action="store_true", help="start from the recorded growth_rate of the first record")
    parser.add_argument("--check", action="store_true", help="compare with the recorded growth_rate")
    parser.add_argument("--biomass", type=float, default=10.0, help="g/kg, only to turn the recorded uptake into q")
    parser.add_argument("--q-max-umol-per-g-h", type=float, default=8300.0)
    parser.add_argument("--k-s", type=float, default=1.0e-5, help="mol/kg")
    parser.add_argument("--growth-yield", type=float, default=90.0, help="g/mol")
    parser.add_argument("--maintenance-umol-per-g-h", type=float, default=222.0)
    parser.add_argument("--demand-margin", type=float, default=0.1)
    parser.add_argument("--tau-up", type=float, default=20.0, help="s")
    parser.add_argument("--tau-down", type=float, default=20.0, help="s")
    parser.add_argument("--growth-rate-initial", type=float, default=6.94e-5, help="1/s")
    arguments = parser.parse_args()

    p = parameters_from(arguments)
    data = load_lifelines(arguments.dir, with_aux=False)
    names = data["field_names"]
    time = data["time"].astype(np.float64)[::arguments.record_stride]
    scalars = data["scalars"][::arguments.record_stride]
    substrate = scalars[:, :, names.index("substrate")].astype(np.float64)
    recorded = scalars[:, :, names.index("growth_rate")].astype(np.float64) if "growth_rate" in names else None
    initial = recorded[0] if (arguments.init_from_record and recorded is not None) else arguments.growth_rate_initial
    demand_mean = None
    if arguments.demand_state == "recorded-mean":
        if recorded is None:
            raise SystemExit("--demand-state recorded-mean needs a recorded growth_rate")
        demand_mean = recorded.mean(axis=1)
    print(f"{substrate.shape[1]} lifelines, {substrate.shape[0]} records, {time[0]:.2f}..{time[-1]:.2f} s, implicit Euler step <= "
          f"{arguments.step} s, demand state {arguments.demand_state}, initial mu "
          f"{'from the first record' if np.ndim(initial) else f'{initial * PER_HOUR:.4f} 1/h'}")
    print(f"q_max {arguments.q_max_umol_per_g_h:g} umol/gdw/h, K_s {arguments.k_s:g}, Y {arguments.growth_yield:g} g/mol, "
          f"m_s {arguments.maintenance_umol_per_g_h:g}, alpha {arguments.demand_margin:g}, mu_max {p['growth_rate_maximum'] * PER_HOUR:.4f} 1/h, "
          f"tau up / down {arguments.tau_up:g} / {arguments.tau_down:g} s")
    growth_rate, uptake, limited = integrate(time, substrate, initial, demand_mean, p, arguments.step)
    np.savez_compressed(arguments.out, time=time, uid=data["uid"], growth_rate=growth_rate, uptake=uptake, demand_limited=limited)

    window = time >= arguments.statistics_start
    if not window.any():          # short test runs
        window[:] = True
    to_umol = 1.0 / UMOL_PER_G_H
    print(f"offline cells, records {time[window][0]:.1f}..{time[window][-1]:.1f} s, all lifelines:")
    describe("mu (1/h)", growth_rate[window], PER_HOUR)
    describe("q (umol/gdw/h)", uptake[window], to_umol)
    print(f"   per-lifeline time-mean mu std {growth_rate[window].mean(axis=0).std() * PER_HOUR:.4f} 1/h; "
          f"demand-limited {100 * limited[window].mean():.1f} % of the samples; q below m_s "
          f"{100 * (uptake[window] < p['maintenance']).mean():.1f} %")
    if "uptake" in names and window.sum() > 1:
        cumulative = scalars[:, :, names.index("uptake")].astype(np.float64)
        span = time[window][-1] - time[window][0]
        recorded_q = (cumulative[window][-1] - cumulative[window][0]) / span / arguments.biomass
        print(f"   field uptake (recorded, x_bio {arguments.biomass:g} g/kg): mean {recorded_q.mean() * to_umol:.1f} umol/gdw/h; "
              f"offline cells {uptake[window].mean() * to_umol:.1f}; offline / field {uptake[window].mean() / recorded_q.mean():.4f}")
    if arguments.check and recorded is not None:
        difference = np.abs(growth_rate[window] - recorded[window]) * PER_HOUR
        print(f"   check mu vs recorded growth_rate: max |diff| {difference.max():.3e} 1/h, rms {np.sqrt((difference ** 2).mean()):.3e} 1/h, "
              f"p99 {np.percentile(difference, 99):.3e} 1/h; means {growth_rate[window].mean() * PER_HOUR:.5f} vs "
              f"{recorded[window].mean() * PER_HOUR:.5f} 1/h")
    print(f"wrote {arguments.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
