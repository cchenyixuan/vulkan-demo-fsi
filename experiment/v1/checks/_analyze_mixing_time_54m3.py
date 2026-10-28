"""_analyze_mixing_time_54m3.py (2026-10-29): mixing of the haringa54 tracer pulses at the bottom probe.
usage: python experiment/v1/checks/_analyze_mixing_time_54m3.py PROBES.csv [...]; pulses at 60 s + 6 s k, k = 0..7"""
# tau95 and circulation time of the 8 tracer pulses at the bottom probe (definition of log/2026-10-25_54m3-results...):
# C* = C / (total / fluid_mass); tau95 = first sample after the last one outside [0.95, 1.05] minus the pulse time;
# circulation time = 2 x lag until C* first reaches 0.05
import csv, sys
import numpy as np
for path in sys.argv[1:]:
    rows = list(csv.DictReader(open(path, newline="")))
    t = np.array([float(r["time"]) for r in rows]); mass = np.array([float(r["fluid_mass"]) for r in rows])
    taus, circs = [], []
    for k in range(8):
        t0 = 60.0 + 6.0 * k; f = f"tracer_{k + 1:02d}"
        c = np.array([float(r[f"probe_bottom:{f}"]) for r in rows]); tot = np.array([float(r[f"total:{f}"]) for r in rows])
        w = t >= t0; tt = t[w]; cs = c[w] / np.maximum(tot[w] / mass[w], 1e-300)
        out = np.where((cs < 0.95) | (cs > 1.05))[0]
        if out.size == 0 or out[-1] + 1 >= tt.size: continue
        taus.append(tt[out[-1] + 1] - t0)
        hit = np.where(cs >= 0.05)[0]
        if hit.size: circs.append(2 * (tt[hit[0]] - t0))
    taus, circs = np.array(taus), np.array(circs)
    print(f"{path}: tau95 {taus.mean():.1f} +- {taus.std(ddof=1):.1f} s ({taus.min():.1f}..{taus.max():.1f}, n {taus.size}); circulation {circs.mean():.1f} +- {circs.std(ddof=1):.1f} s")
