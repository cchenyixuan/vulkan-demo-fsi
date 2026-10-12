"""_plot_jahoda_discharge_vs_lda.py - our Jahoda-tank Rushton discharge profiles
(H1, 3 mm and 2 mm) against the experimental points of Haringa (2023) Fig. 1
(2026-10-13).

Inputs
  output/haringa/H1/H1_jahoda_{3mm,2mm}_discharge.csv   (_analyze_flow_statistics.py:
      xi = (r - r_tip)/(R - r_tip), U_rad/U_tip, U_tan/U_tip, U_ax/U_tip, k/U_tip^2,
      k_random/U_tip^2 (blade-periodic part removed), eps_sgs/N3D2 (Smagorinsky
      estimate), eps_2nuSS/N3D2 (resolved 2 nu S:S); y = C, between the baffles,
      average of 20-60 s)
  docs/data/haringa2023_fig1_experiments.csv            (digitised markers, see
      _digitize_haringa2023_fig1.py for the sources)
  docs/data/haringa2023_fig1_lbles_360base.csv          (Haringa's LB-LES 360-BASE line)

Outputs (--out-dir, default output/haringa/H1/analysis)
  jahoda_discharge_vs_lda.png     three panels in the style of Fig. 1
  jahoda_axial_velocity.png       U_ax/U_tip and the jet angle atan(U_ax/U_rad)
  jahoda_discharge_vs_lda_table.md / .json   tip values, peaks, per-source ratios

Which of our quantities is compared with which marker:
  U_rad         all sources (time-averaged radial velocity)
  k_random      Haringa 2018 and Wu & Patterson (blade-passage periodicity removed)
  k (total)     Murthy & Joshi (total fluctuation energy; see the log for the caveat)
  eps_sgs       Ducci & Yianneskis and Wu & Patterson; eps_2nuSS is shown as well
                (our dissipation is a Smagorinsky estimate, not a resolved quantity)

Usage (repo root):
  python experiment/v1/checks/_plot_jahoda_discharge_vs_lda.py [--h1-dir DIR] [--out-dir DIR]
"""
import argparse
import csv
import json
import pathlib

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = (("3 mm", "H1_jahoda_3mm_discharge.csv", "tab:blue"), ("2 mm", "H1_jahoda_2mm_discharge.csv", "tab:red"))
PANELS = (
    ("U_rad/U_tip", r"$U_{rad}/U_{tip}$", (0.0, 1.0)),
    ("k_t/U_tip^2", r"$k_t/U_{tip}^2$", (0.0, 0.15)),
    ("eps/N3D2", r"$\varepsilon/N^3D^2$", (0.0, 15.0)),
)
MARKERS = {  # source: (label, marker, facecolor, edgecolor, size)
    "haringa2018": ("Haringa et al. 2018 (LDA)", "s", "black", "black", 6),
    "wu1989": ("Wu & Patterson 1989", "D", "gold", "black", 6),
    "murthy2008": ("Murthy & Joshi 2008", "o", "white", "black", 6),
    "ducci2005_lda": ("Ducci & Yianneskis 2005 LDA", "*", "white", "black", 10),
    "ducci2005_piv": ("Ducci & Yianneskis 2005 PIV", "*", "black", "black", 10),
    "ranade1990": ("Ranade & Joshi 1990 summary", "x", "grey", "grey", 6),
}


def read_csv(path, delimiter=","):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.reader(handle, delimiter=delimiter) if row and not row[0].startswith("#")]
    header = [name.strip() for name in rows[0]]
    return header, rows[1:]


def read_profile(path):
    header, rows = read_csv(path)
    data = np.array([[float(cell) for cell in row] for row in rows])
    return {name: data[:, index] for index, name in enumerate(header)}


def read_experiments(path):
    header, rows = read_csv(path)
    points = {}
    for row in rows:
        record = dict(zip(header, row))
        key = (record["source"], record["quantity"])
        points.setdefault(key, []).append((float(record["xi"]), float(record["value"]),
                                           float(record["err_lo"]) if record["err_lo"] else np.nan,
                                           float(record["err_hi"]) if record["err_hi"] else np.nan,
                                           int(record["series"])))
    return {key: np.array(values) for key, values in points.items()}


def read_lbles(path):
    header, rows = read_csv(path)
    curves = {}
    for quantity, xi, value in rows:
        curves.setdefault(quantity, []).append((float(xi), float(value)))
    return {key: np.array(values) for key, values in curves.items()}


def interpolate(profile, column, xi):
    return np.interp(xi, profile["xi"], profile[column])


def peak(xi, value, xi_max=0.9):
    mask = xi <= xi_max
    index = int(np.argmax(value[mask]))
    return float(xi[mask][index]), float(value[mask][index])


# ---------------------------------------------------------------------------

def plot_profiles(profiles, experiments, lbles, out_path):
    figure, axes = plt.subplots(1, 3, figsize=(12.5, 4.0))
    ours = {
        "U_rad/U_tip": (("U_rad/U_tip", "-", ""),),
        "k_t/U_tip^2": (("k_random/U_tip^2", "-", " $k$ periodic removed"), ("k/U_tip^2", "--", " $k$ total")),
        "eps/N3D2": (("eps_sgs/N3D2", "-", r" $\varepsilon_{SGS}$"), ("eps_2nuSS/N3D2", ":", r" $2\nu S{:}S$")),
    }
    for axis, (quantity, ylabel, ylim) in zip(axes, PANELS):
        for label, _, colour in RUNS:
            profile = profiles[label]
            for column, style, suffix in ours[quantity]:
                axis.plot(profile["xi"], profile[column], style, color=colour, lw=1.6,
                          label=f"SPH {label}{suffix}")
        if quantity in lbles:
            axis.plot(lbles[quantity][:, 0], lbles[quantity][:, 1], "-", color="black", lw=0.9, alpha=0.7,
                      label="Haringa 2023 LB-LES 360-BASE")
        for source, (label, marker, face, edge, size) in MARKERS.items():
            key = (source, quantity)
            if key not in experiments:
                continue
            pts = experiments[key]
            axis.plot(pts[:, 0], pts[:, 1], linestyle="none", marker=marker, markerfacecolor=face,
                      markeredgecolor=edge, markersize=size, label=label, zorder=5)
            if np.isfinite(pts[:, 2]).any():
                axis.errorbar(pts[:, 0], pts[:, 1], yerr=[pts[:, 1] - pts[:, 2], pts[:, 3] - pts[:, 1]],
                              fmt="none", ecolor="black", elinewidth=0.8, capsize=2, zorder=4)
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(*ylim)
        axis.set_xlabel(r"$(r - r_{tip})/(R - r_{tip})$")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=6.5, loc="upper right")
    figure.suptitle("Rushton discharge at y = C between the baffles: SPH (Jahoda tank, 300 rpm, 20-60 s mean) "
                    "vs experiments of Haringa (2023) Fig. 1", fontsize=9.5)
    figure.tight_layout()
    figure.savefig(out_path, dpi=160)
    plt.close(figure)


def plot_axial(profiles, out_path):
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.6))
    for label, _, colour in RUNS:
        profile = profiles[label]
        axes[0].plot(profile["xi"], profile["U_ax/U_tip"], "-", color=colour, lw=1.6, label=f"SPH {label}")
        angle = np.degrees(np.arctan2(profile["U_ax/U_tip"], profile["U_rad/U_tip"]))
        axes[1].plot(profile["xi"], angle, "-", color=colour, lw=1.6, label=f"SPH {label}")
    axes[0].axhline(0.0, color="black", lw=0.8)
    axes[1].axhline(0.0, color="black", lw=0.8)
    axes[0].set_ylabel(r"$U_{ax}/U_{tip}$ (+ up)")
    axes[1].set_ylabel(r"jet angle $\arctan(U_{ax}/U_{rad})$ [deg]")
    axes[1].set_ylim(-30, 30)
    for axis in axes:
        axis.set_xlim(0.0, 1.0)
        axis.set_xlabel(r"$(r - r_{tip})/(R - r_{tip})$")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle("Axial velocity on the disc plane y = C (no experimental profile available in the sources)",
                    fontsize=9.5)
    figure.tight_layout()
    figure.savefig(out_path, dpi=160)
    plt.close(figure)


# ---------------------------------------------------------------------------

def build_table(profiles, experiments, lbles):
    rows = []  # (group, item, value 3 mm, value 2 mm, reference text)

    def fmt(value, digits=3):
        return f"{value:.{digits}f}" if np.isfinite(value) else "n/a"

    # --- tip values
    tip = {label: {column: float(interpolate(profiles[label], column, 0.0))
                   for column in ("U_rad/U_tip", "U_ax/U_tip", "k/U_tip^2", "k_random/U_tip^2", "eps_sgs/N3D2", "eps_2nuSS/N3D2")}
           for label, _, _ in RUNS}
    ranade = experiments[("ranade1990", "U_rad/U_tip")]
    ranade_tip = [float(np.interp(0.0, ranade[ranade[:, 4] == s][:, 0], ranade[ranade[:, 4] == s][:, 1]))
                  for s in (1, 2)]
    haringa_u = experiments[("haringa2018", "U_rad/U_tip")]
    rows.append(("tip (xi = 0)", "U_rad/U_tip", tip["3 mm"]["U_rad/U_tip"], tip["2 mm"]["U_rad/U_tip"],
                 f"Ranade & Joshi band at xi = 0: {ranade_tip[0]:.2f} .. {ranade_tip[1]:.2f}; "
                 f"LB-LES 360-BASE {np.interp(0.0, lbles['U_rad/U_tip'][:, 0], lbles['U_rad/U_tip'][:, 1]):.2f}"))
    xi_h = haringa_u[0, 0]
    rows.append((f"first LDA point (xi = {xi_h:.3f})", "U_rad/U_tip",
                 float(interpolate(profiles["3 mm"], "U_rad/U_tip", xi_h)),
                 float(interpolate(profiles["2 mm"], "U_rad/U_tip", xi_h)),
                 f"Haringa 2018: {haringa_u[0, 1]:.3f}"))
    rows.append(("tip (xi = 0)", "U_ax/U_tip", tip["3 mm"]["U_ax/U_tip"], tip["2 mm"]["U_ax/U_tip"], "no measured profile"))
    for label, _, _ in RUNS:
        tip[label]["angle"] = float(np.degrees(np.arctan2(tip[label]["U_ax/U_tip"], tip[label]["U_rad/U_tip"])))
    rows.append(("tip (xi = 0)", "jet angle [deg, + up]", tip["3 mm"]["angle"], tip["2 mm"]["angle"], "Rushton: 0 expected (symmetric disc)"))
    rows.append(("tip (xi = 0)", "k_random/U_tip^2", tip["3 mm"]["k_random/U_tip^2"], tip["2 mm"]["k_random/U_tip^2"],
                 f"LB-LES 360-BASE k_t at tip {np.interp(0.0, lbles['k_t/U_tip^2'][:, 0], lbles['k_t/U_tip^2'][:, 1]):.3f}"))
    rows.append(("tip (xi = 0)", "k/U_tip^2 (total)", tip["3 mm"]["k/U_tip^2"], tip["2 mm"]["k/U_tip^2"], ""))
    rows.append(("tip (xi = 0)", "eps_SGS/N3D2", tip["3 mm"]["eps_sgs/N3D2"], tip["2 mm"]["eps_sgs/N3D2"],
                 f"LB-LES 360-BASE at tip {np.interp(0.0, lbles['eps/N3D2'][:, 0], lbles['eps/N3D2'][:, 1]):.1f}"))

    # --- peaks
    peaks = {}
    for label, _, _ in RUNS:
        profile = profiles[label]
        peaks[label] = {column: peak(profile["xi"], profile[column]) for column in
                        ("k/U_tip^2", "k_random/U_tip^2", "eps_sgs/N3D2", "eps_2nuSS/N3D2", "U_ax/U_tip")}
        mask = profile["xi"] <= 0.5
        peaks[label]["U_ax mean 0-0.5"] = float(profile["U_ax/U_tip"][mask].mean())
    exp_peaks = {}
    for key, pts in experiments.items():
        index = int(np.argmax(pts[:, 1]))
        exp_peaks[key] = (float(pts[index, 0]), float(pts[index, 1]))
    lb_peaks = {quantity: peak(curve[:, 0], curve[:, 1]) for quantity, curve in lbles.items()}
    rows.append(("k peak", "xi of max k_random", peaks["3 mm"]["k_random/U_tip^2"][0], peaks["2 mm"]["k_random/U_tip^2"][0],
                 f"Haringa 2018 {exp_peaks[('haringa2018', 'k_t/U_tip^2')][0]:.2f}, Wu & Patterson {exp_peaks[('wu1989', 'k_t/U_tip^2')][0]:.2f}, "
                 f"Murthy & Joshi {exp_peaks[('murthy2008', 'k_t/U_tip^2')][0]:.2f} (first point); LB-LES {lb_peaks['k_t/U_tip^2'][0]:.2f}"))
    rows.append(("k peak", "max k_random/U_tip^2", peaks["3 mm"]["k_random/U_tip^2"][1], peaks["2 mm"]["k_random/U_tip^2"][1],
                 f"Haringa 2018 {exp_peaks[('haringa2018', 'k_t/U_tip^2')][1]:.3f}, Wu & Patterson {exp_peaks[('wu1989', 'k_t/U_tip^2')][1]:.3f}"))
    rows.append(("k peak", "xi of max k (total)", peaks["3 mm"]["k/U_tip^2"][0], peaks["2 mm"]["k/U_tip^2"][0], ""))
    rows.append(("k peak", "max k/U_tip^2 (total)", peaks["3 mm"]["k/U_tip^2"][1], peaks["2 mm"]["k/U_tip^2"][1],
                 f"Murthy & Joshi {exp_peaks[('murthy2008', 'k_t/U_tip^2')][1]:.3f}; LB-LES {lb_peaks['k_t/U_tip^2'][1]:.3f}"))
    rows.append(("eps peak", "xi of max eps_SGS", peaks["3 mm"]["eps_sgs/N3D2"][0], peaks["2 mm"]["eps_sgs/N3D2"][0],
                 f"Ducci PIV {exp_peaks[('ducci2005_piv', 'eps/N3D2')][0]:.2f}, Ducci LDA {exp_peaks[('ducci2005_lda', 'eps/N3D2')][0]:.2f}, "
                 f"Wu & Patterson {exp_peaks[('wu1989', 'eps/N3D2')][0]:.2f}; LB-LES {lb_peaks['eps/N3D2'][0]:.2f}"))
    rows.append(("eps peak", "max eps_SGS/N3D2", peaks["3 mm"]["eps_sgs/N3D2"][1], peaks["2 mm"]["eps_sgs/N3D2"][1],
                 f"Ducci PIV {exp_peaks[('ducci2005_piv', 'eps/N3D2')][1]:.1f}, LDA {exp_peaks[('ducci2005_lda', 'eps/N3D2')][1]:.1f}, "
                 f"Wu & Patterson {exp_peaks[('wu1989', 'eps/N3D2')][1]:.1f}; LB-LES {lb_peaks['eps/N3D2'][1]:.1f}"))
    rows.append(("eps peak", "xi of max 2nuS:S", peaks["3 mm"]["eps_2nuSS/N3D2"][0], peaks["2 mm"]["eps_2nuSS/N3D2"][0], "resolved part only"))
    rows.append(("eps peak", "max 2nuS:S/N3D2", peaks["3 mm"]["eps_2nuSS/N3D2"][1], peaks["2 mm"]["eps_2nuSS/N3D2"][1], ""))
    rows.append(("eps at xi = 0.2", "eps_SGS/N3D2", float(interpolate(profiles["3 mm"], "eps_sgs/N3D2", 0.2)),
                 float(interpolate(profiles["2 mm"], "eps_sgs/N3D2", 0.2)), "experiments 5 .. 12 at xi = 0.18 .. 0.2"))
    rows.append(("axial", "max U_ax/U_tip (xi <= 0.9)", peaks["3 mm"]["U_ax/U_tip"][1], peaks["2 mm"]["U_ax/U_tip"][1],
                 f"at xi = {peaks['3 mm']['U_ax/U_tip'][0]:.2f} / {peaks['2 mm']['U_ax/U_tip'][0]:.2f}"))
    rows.append(("axial", "mean U_ax/U_tip, xi 0-0.5", peaks["3 mm"]["U_ax mean 0-0.5"], peaks["2 mm"]["U_ax mean 0-0.5"], ""))

    # --- per-source ratios ours / experiment at the experimental xi
    pairs = (
        ("U_rad/U_tip", "haringa2018", "U_rad/U_tip"),
        ("U_rad/U_tip", "ranade1990", "U_rad/U_tip"),
        ("k_random/U_tip^2", "haringa2018", "k_t/U_tip^2"),
        ("k_random/U_tip^2", "wu1989", "k_t/U_tip^2"),
        ("k/U_tip^2", "murthy2008", "k_t/U_tip^2"),
        ("eps_sgs/N3D2", "ducci2005_piv", "eps/N3D2"),
        ("eps_sgs/N3D2", "ducci2005_lda", "eps/N3D2"),
        ("eps_sgs/N3D2", "wu1989", "eps/N3D2"),
    )
    ratios = []
    for column, source, quantity in pairs:
        pts = experiments[(source, quantity)]
        pts = pts[pts[:, 0] <= 0.7]  # the near-baffle points (xi > 0.7) are not comparable, see the log
        entry = {"ours": column, "source": source, "n": int(len(pts))}
        for label, _, _ in RUNS:
            ours = interpolate(profiles[label], column, pts[:, 0])
            ratio = ours / pts[:, 1]
            entry[label] = {"mean_ratio": float(np.exp(np.mean(np.log(ratio)))), "min_ratio": float(ratio.min()),
                            "max_ratio": float(ratio.max()),
                            "rms_rel_dev": float(np.sqrt(np.mean((ours - pts[:, 1]) ** 2)) / np.mean(pts[:, 1]))}
        ratios.append(entry)

    lines = ["| 组 | 量 | SPH 3 mm | SPH 2 mm | 参照 |", "|---|---|---|---|---|"]
    for group, item, v3, v2, reference in rows:
        lines.append(f"| {group} | {item} | {fmt(v3)} | {fmt(v2)} | {reference} |")
    lines += ["", "| 我们的量 | 实验来源 | 点数 (xi <= 0.7) | 3 mm 几何平均比 (最小..最大) | 2 mm 几何平均比 (最小..最大) | 3 mm RMS 相对偏差 | 2 mm RMS 相对偏差 |",
              "|---|---|---|---|---|---|---|"]
    for entry in ratios:
        a, b = entry["3 mm"], entry["2 mm"]
        lines.append(f"| {entry['ours']} | {entry['source']} | {entry['n']} | {a['mean_ratio']:.2f} ({a['min_ratio']:.2f}..{a['max_ratio']:.2f}) "
                     f"| {b['mean_ratio']:.2f} ({b['min_ratio']:.2f}..{b['max_ratio']:.2f}) | {a['rms_rel_dev']:.2f} | {b['rms_rel_dev']:.2f} |")
    return "\n".join(lines), {"rows": [dict(zip(("group", "item", "sph_3mm", "sph_2mm", "reference"), row)) for row in rows],
                              "ratios": ratios, "tip": tip, "peaks": {k: {c: list(v) if isinstance(v, tuple) else v for c, v in d.items()} for k, d in peaks.items()},
                              "experiment_peaks": {f"{k[0]}:{k[1]}": v for k, v in exp_peaks.items()}, "lbles_peaks": lb_peaks}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--h1-dir", type=pathlib.Path, default=pathlib.Path("output/haringa/H1"))
    parser.add_argument("--data", type=pathlib.Path, default=pathlib.Path("docs/data/haringa2023_fig1_experiments.csv"))
    parser.add_argument("--lbles", type=pathlib.Path, default=pathlib.Path("docs/data/haringa2023_fig1_lbles_360base.csv"))
    parser.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("output/haringa/H1/analysis"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    profiles = {label: read_profile(args.h1_dir / filename) for label, filename, _ in RUNS}
    experiments = read_experiments(args.data)
    lbles = read_lbles(args.lbles)

    plot_profiles(profiles, experiments, lbles, args.out_dir / "jahoda_discharge_vs_lda.png")
    plot_axial(profiles, args.out_dir / "jahoda_axial_velocity.png")
    table, summary = build_table(profiles, experiments, lbles)
    (args.out_dir / "jahoda_discharge_vs_lda_table.md").write_text(table + "\n", encoding="utf-8")
    with open(args.out_dir / "jahoda_discharge_vs_lda_table.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=1)
    print(table)
    print(f"\nwrote figures and tables to {args.out_dir}")


if __name__ == "__main__":
    main()
