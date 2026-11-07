# halves_runs.py (Mahuika analysis_tools) with the window from the environment: START .. END (defaults 120 .. 240), halves at the
# midpoint, plus the two lifeline-population halves; 2026-11-08 for the 240..480 s extensions. Run from the repo root.
# usage: START=240 END=480 python halves_runs2.py PRODUCTION_DIR RUN [RUN ...]
import os, sys
import numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "experiment/v1/checks")
from experiment.v1.utils.lifeline_recorder import load_lifelines
from _analyze_lifelines import regime_analysis
P = sys.argv[1]; KS = 1e-5
START = float(os.environ.get("START", "120")); END = float(os.environ.get("END", "240")); MID = 0.5 * (START + END)
for n in sys.argv[2:]:
    d = load_lifelines(f"{P}/{n}_lifelines", with_aux=False); t = d["time"]; names = list(d["field_names"])
    cs = d["scalars"][:, :, names.index("substrate")].astype(float); q = cs / (cs + KS)
    out = []
    for a, b in ((START, MID), (MID, END + 0.1)):
        w = (t >= a) & (t < b); out.append(regime_analysis(q[w], t[w])["fractions"])
    w = (t >= START) & (t < END + 0.1)
    count = q.shape[1]
    for half in (slice(0, count // 2), slice(count // 2, count)):
        out.append(regime_analysis(q[w][:, half], t[w])["fractions"])
    print(n, f"S/L/E % in {START:.0f}-{MID:.0f}, {MID:.0f}-{END:.0f}, lifelines first half, second half:",
          "; ".join("/".join(f"{100*x:.1f}" for x in f) for f in out))
