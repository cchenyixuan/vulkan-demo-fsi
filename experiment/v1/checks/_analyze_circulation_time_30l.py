"""_analyze_circulation_time_30l.py (2026-11-04): circulation time of the 30 L tracer pulses at the bottom probes,
same definition as the 54 m3 tank (_analyze_mixing_time_54m3.py, Haringa 2023).
usage: python experiment/v1/checks/_analyze_circulation_time_30l.py PROBES.csv [...]; pulses at 25 s + 1 s k, k = 0..9"""
# C* = C / (total / fluid_mass) (exact well-mixed value); circulation time = 2 x lag from the pulse start until C*
# first reaches 0.05; a pulse whose record ends before C* reaches 0.05 is skipped and counted
import csv, sys
import numpy as np
for path in sys.argv[1:]:
    rows = list(csv.DictReader(open(path, newline="")))
    t = np.array([float(r["time"]) for r in rows]); mass = np.array([float(r["fluid_mass"]) for r in rows])
    probes = sorted({key.split(":")[0] for key in rows[0] if key.startswith("probe_bottom")})
    for probe in probes:
        circs, missed = [], 0
        for k in range(10):
            t0 = 25.0 + 1.0 * k; f = f"tracer_{k + 1:02d}"
            c = np.array([float(r[f"{probe}:{f}"]) for r in rows]); tot = np.array([float(r[f"total:{f}"]) for r in rows])
            w = t >= t0; tt = t[w]; cs = c[w] / np.maximum(tot[w] / mass[w], 1e-300)
            hit = np.where(cs >= 0.05)[0]
            if hit.size: circs.append(2 * (tt[hit[0]] - t0))
            else: missed += 1
        circs = np.array(circs)
        print(f"{path} {probe}: circulation {circs.mean():.1f} +- {circs.std(ddof=1):.1f} s "
              f"({circs.min():.1f}..{circs.max():.1f}, n {circs.size}, not reached {missed}); "
              f"lag {circs.mean() / 2:.2f} s")
