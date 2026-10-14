"""C (local two-way) minus E (population-mean two-way), relative, per Da-scan point; parsed from compare_*.txt (section 3)."""
import re, json, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
root = pathlib.Path(__file__).parent
points = [("Da1", "55 g/kg, K_i 49.2"), ("Da1k", "55 g/kg, K_i 24.6"), ("Da6", "220 g/kg, K_i 49.2")]
rows = {}
for name, label in points:
    text = (root / f"compare_{name}.txt").read_text()
    sec = text.split("3. lifelines")[1]
    vals = {}
    for q in ("cs", "gly", "mu"):
        for run in ("C", "E"):
            m = re.search(rf"^{q}\s+{run}\s+([\d.e+-]+)\s+([\d.e+-]+)\s+([\d.e+-]+)\s+([\d.e+-]+)", sec, re.M)
            vals[(q, run)] = [float(x) for x in m.groups()]
    for run in ("C", "E"):
        m = re.search(rf"{run}\s+fractions E/L/S ([\d.]+) / ([\d.]+) / ([\d.]+); mean residence E ([\d.]+) s, L ([\d.]+) s, S ([\d.]+) s", sec)
        vals[("frac", run)] = [float(x) for x in m.groups()]
    s = json.load(open(root / f"compare_{name}" / "summary.json"))
    vals[("uptake", "C")] = s["C"]["probes"]["uptake_total"][-1]; vals[("uptake", "E")] = s["E"]["probes"]["uptake_total"][-1]
    rows[name] = vals
def rel(v, q, i=None):
    c = v[(q, "C")] if i is None else v[(q, "C")][i]; e = v[(q, "E")] if i is None else v[(q, "E")][i]
    return 100.0 * (c / e - 1.0)
series = {"cumulative uptake": lambda v: rel(v, "uptake"), "X_gly mean": lambda v: rel(v, "gly", 0), "X_gly p5": lambda v: rel(v, "gly", 1),
          "X_gly p95": lambda v: rel(v, "gly", 3), "mu mean": lambda v: rel(v, "mu", 0), "mu p95": lambda v: rel(v, "mu", 3),
          "S fraction (points)": lambda v: 100.0 * (v[("frac", "C")][2] - v[("frac", "E")][2]),
          "E fraction (points)": lambda v: 100.0 * (v[("frac", "C")][0] - v[("frac", "E")][0])}
fig, ax = plt.subplots(figsize=(7.5, 4.5))
x = range(len(points))
for key, f in series.items():
    ax.plot(list(x), [f(rows[n]) for n, _ in points], marker="o", label=key)
ax.axhline(0, color="k", lw=0.8); ax.set_xticks(list(x)); ax.set_xticklabels([f"{n}\n{l}" for n, l in points])
ax.set_ylabel("local two-way C relative to population-mean E, %"); ax.grid(alpha=0.3); ax.legend(fontsize=8, ncol=2)
ax.set_title("30 L, 4 mm, 200 s, statistics from 60 s: C minus E per point")
fig.tight_layout(); fig.savefig(root / "da30_c_minus_e.png", dpi=130)
print("| point | " + " | ".join(series) + " |"); print("|---" * (len(series) + 1) + "|")
for n, l in points: print(f"| {n} ({l}) | " + " | ".join(f"{f(rows[n]):+.1f}" for f in series.values()) + " |")
