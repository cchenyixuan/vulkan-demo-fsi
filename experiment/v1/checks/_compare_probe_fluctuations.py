"""_compare_probe_fluctuations.py - tracer-concentration fluctuations at the two
mixing-time probes of the 30 L tank: our SPH run against the M-Star (LBM-LES)
and Fluent (FV-LES) probe records of the DARUS-5523 dataset (2026-10-13).

For every tracer pulse and probe the window [t0 + 5 s, t0 + 25 s] after the
pulse start t0 is taken, the signal is normalised with its final concentration
C_inf (mean of the last 2 s of the record), the smooth mixing trend is removed
with a centred moving average of 1 s (the trend is computed on the window
extended by half the averaging length, so no edge is lost), and from the
residual c'(t) = C/C_inf - trend the variance (and its square root, the relative
fluctuation level) and a Welch power spectrum (Hann, 4 s segments, 50 %
overlap) are computed.  Spectra are averaged over all pulses of a dataset.

Records
  ours      PROBE_LOG of _run_v1_headless.py (probe_k:tracer_nn columns, 0.02 s);
            pulses at 25, 26, ... 34 s (first non-zero total sets t0)
  M-Star    02_simulation_results/01_mixing_time_results/00_raw_txt/grid_study_200rpm_LX400/
            trial_NN_probe_K_control_volume.txt, columns "dye_nnn Mean [mol/L]",
            0.02 s, 10 dyes per trial injected at 25, 26, ... 34 s, 5 trials
  Fluent    02_simulation_results/06_fluent_results/{fine_mesh,coarse_mesh}/<t>s/ctracer_LES_n.out,
            columns tracer_p1 tracer_p2, one tracer injected at t = <t> s,
            5 realisations per start time, 0.01 s (fine) / 0.005 s (coarse);
            block-averaged to 0.02 s before the analysis so that all records
            share the sampling rate and Nyquist frequency (25 Hz)

Probe definitions differ and this matters for the fluctuation level: the two
M-Star probes are Ø15 x 35 mm control volumes (6.2e-6 m^3, cell means); ours
(SphSimulatorV1.probe_scalars) are Shepard kernel interpolations at the probe
point with a Wendland C4 support of `scalars.probes.radius` (6 mm in P1), i.e.
a weighted sphere of about 0.9e-6 m^3 holding ~110 particles at 2 mm, of which
the kernel weights the central few tens; the Fluent monitor type is not
documented in the dataset.  Besides the variance and spectra the first-arrival
overshoot max(C/C_inf) after each pulse is reported, which does not depend on
the detrending.

Output files carry the window in their name (<label>_w5-25_...).

Usage (repo root):
  python experiment/v1/checks/_compare_probe_fluctuations.py [--ours PROBE_LOG]
      [--dataset "D:/CFD/SPH dev/repository/02_simulation_results"] [--out-dir DIR]
      [--window 5 25] [--trend 1.0] [--segment 4.0]
"""
import argparse
import csv
import json
import pathlib
import re

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import welch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TARGET_DT = 0.02
IMPELLER_FREQUENCY = 200.0 / 60.0  # 1/s
BANDS = ((0.5, 2.0), (2.0, 6.0), (6.0, 15.0), (15.0, 25.0))


# ---------------------------------------------------------------------------
# Readers: each yields (dataset, realisation, probe, pulse_start, time, concentration)
# ---------------------------------------------------------------------------

def read_table(path, delimiter=","):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle, delimiter=delimiter))
    header = [name.strip() for name in rows[0]]
    data = np.array([[float(cell) if cell.strip() not in ("", "nan") else np.nan for cell in row]
                     for row in rows[1:] if len(row) == len(header)])
    return {name: data[:, index] for index, name in enumerate(header)}


def read_ours(path):
    table = read_table(path)
    time = table["time"]
    tracers = sorted({name.split(":")[1] for name in table if name.startswith("probe_1:")})
    for tracer in tracers:
        total = table[f"total:{tracer}"]
        start = float(time[np.argmax(total > 0.0)])
        start = round(start - 0.02, 2)  # the first non-zero sample is the one after the pulse start
        for probe in (1, 2):
            yield ("SPH", path.stem, probe, start, time, table[f"probe_{probe}:{tracer}"])


def read_mstar(directory):
    for probe_1 in sorted(directory.glob("trial_*_probe_1_control_volume.txt")):
        trial = probe_1.name.split("_probe_")[0]
        for probe in (1, 2):
            table = read_table(directory / f"{trial}_probe_{probe}_control_volume.txt", delimiter="\t")
            time = table["Time [s]"]
            dyes = sorted(name for name in table if name.startswith("dye_") and "Mean" in name)
            for index, name in enumerate(dyes):
                yield ("M-Star", trial, probe, 25.0 + index, time, table[name])


def read_fluent(directory, label):
    for start_dir in sorted(directory.glob("*s")):
        match = re.fullmatch(r"(\d+)s", start_dir.name)
        if not match:
            continue
        start = float(match.group(1))
        for path in sorted(start_dir.glob("ctracer_LES_*.out")):
            rows = []
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    parts = line.split()
                    if len(parts) == 4 and not line.startswith(("\"", "(")):
                        rows.append([float(p) for p in parts])
            data = np.array(rows)
            for probe in (1, 2):
                yield (label, f"{start_dir.name}/{path.stem}", probe, start, data[:, 1], data[:, 1 + probe])


def resample(time, value, target_dt=TARGET_DT):
    """Block-average to the target sampling interval (integer factor only)."""
    dt = float(np.median(np.diff(time)))
    factor = int(round(target_dt / dt))
    if factor <= 1:
        return time, value, dt
    n = (len(time) // factor) * factor
    time = time[:n].reshape(-1, factor).mean(axis=1)
    value = value[:n].reshape(-1, factor).mean(axis=1)
    return time, value, dt * factor


# ---------------------------------------------------------------------------
# Analysis of one signal
# ---------------------------------------------------------------------------

def analyse(time, value, start, window, trend_length, segment_length):
    time, value, dt = resample(time, value)
    final = float(value[time >= time[-1] - 2.0].mean())
    if not final > 0.0:
        return None
    c = value / final
    lo, hi = start + window[0], start + window[1]
    margin = 0.5 * trend_length + 2.0 * dt
    extended = (time >= lo - margin) & (time <= hi + margin)
    if extended.sum() < 10 or time[extended][-1] < hi - dt:
        return None
    trend = uniform_filter1d(c[extended], size=max(1, int(round(trend_length / dt))), mode="nearest")
    residual = c[extended] - trend
    inner = (time[extended] >= lo) & (time[extended] <= hi)
    t_win, c_win, trend_win, r_win = time[extended][inner], c[extended][inner], trend[inner], residual[inner]
    fs = 1.0 / dt
    frequency, psd = welch(r_win, fs=fs, window="hann", nperseg=int(round(segment_length * fs)),
                           noverlap=int(round(segment_length * fs)) // 2, detrend=False, scaling="density")
    half = int(len(r_win) // 2)
    after = time >= start
    peak_index = int(np.argmax(c[after]))
    return {"dt": dt, "final": final, "variance": float(np.mean(r_win ** 2)),
            "peak": float(c[after][peak_index]), "peak_time": float(time[after][peak_index] - start),
            "variance_first_half": float(np.mean(r_win[:half] ** 2)), "variance_second_half": float(np.mean(r_win[half:] ** 2)),
            "trend_level_mean": float(trend_win.mean()), "frequency": frequency, "psd": psd,
            "time": t_win, "c": c_win, "trend": trend_win, "residual": r_win}


# ---------------------------------------------------------------------------

def summarise(results, label):
    """Per dataset and probe: count, mean / median variance, sqrt(mean variance), mean PSD, band powers."""
    out = {}
    for probe in (1, 2):
        items = [r for r in results if r["dataset"] == label and r["probe"] == probe]
        if not items:
            continue
        variances = np.array([r["variance"] for r in items])
        psd = np.mean([r["psd"] for r in items], axis=0)
        frequency = items[0]["frequency"]
        bands = {}
        for lo, hi in BANDS:
            mask = (frequency >= lo) & (frequency < hi)
            bands[f"{lo:g}-{hi:g} Hz"] = float(np.trapz(psd[mask], frequency[mask]))
        out[probe] = {"n": len(items), "variance_mean": float(variances.mean()), "variance_median": float(np.median(variances)),
                      "variance_std": float(variances.std(ddof=1)) if len(items) > 1 else float("nan"),
                      "variance_first_half": float(np.mean([r["variance_first_half"] for r in items])),
                      "variance_second_half": float(np.mean([r["variance_second_half"] for r in items])),
                      "sigma": float(np.sqrt(variances.mean())), "frequency": frequency, "psd_mean": psd, "bands": bands,
                      "peak_mean": float(np.mean([r["peak"] for r in items])), "peak_median": float(np.median([r["peak"] for r in items])),
                      "peak_max": float(np.max([r["peak"] for r in items])),
                      "peak_time_mean": float(np.mean([r["peak_time"] for r in items]))}
    return out


def ratio_text(a, b):
    return f"{a / b:.2f}" if b > 0 else "n/a"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ours", type=pathlib.Path, default=pathlib.Path("output/mixing/P1_2mm/mix_2mm_sheets_P1_probes.csv"))
    parser.add_argument("--dataset", type=pathlib.Path, default=pathlib.Path("D:/CFD/SPH dev/repository/02_simulation_results"))
    parser.add_argument("--mstar-grid", default="grid_study_200rpm_LX400")
    parser.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("output/mixing/P1_2mm/analysis"))
    parser.add_argument("--window", type=float, nargs=2, default=(5.0, 25.0), help="s after the pulse start")
    parser.add_argument("--trend", type=float, default=1.0, help="moving-average length (s) of the trend")
    parser.add_argument("--segment", type=float, default=4.0, help="Welch segment length (s)")
    parser.add_argument("--label", default="P1_2mm")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.label = f"{args.label}_w{args.window[0]:g}-{args.window[1]:g}"

    sources = [read_ours(args.ours),
               read_mstar(args.dataset / "01_mixing_time_results" / "00_raw_txt" / args.mstar_grid),
               read_fluent(args.dataset / "06_fluent_results" / "fine_mesh", "Fluent fine"),
               read_fluent(args.dataset / "06_fluent_results" / "coarse_mesh", "Fluent coarse")]
    results, skipped = [], {}
    for source in sources:
        for dataset, realisation, probe, start, time, value in source:
            result = analyse(time, value, start, args.window, args.trend, args.segment)
            if result is None:
                skipped[dataset] = skipped.get(dataset, 0) + 1
                continue
            result.update({"dataset": dataset, "realisation": realisation, "probe": probe, "start": start})
            results.append(result)
    datasets = [d for d in ("SPH", "M-Star", "Fluent fine", "Fluent coarse") if any(r["dataset"] == d for r in results)]
    summaries = {label: summarise(results, label) for label in datasets}
    for label in datasets:
        dts = sorted({round(r["dt"], 4) for r in results if r["dataset"] == label})
        print(f"{label}: {sum(1 for r in results if r['dataset'] == label)} signals (dt after resampling {dts}), "
              f"skipped {skipped.get(label, 0)}")

    # --- per-signal table
    with open(args.out_dir / f"{args.label}_probe_fluctuations_signals.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "realisation", "probe", "pulse_start_s", "C_inf", "variance", "variance_first_half",
                         "variance_second_half", "sigma", "peak_C_over_Cinf", "peak_time_after_pulse_s"])
        for r in results:
            writer.writerow([r["dataset"], r["realisation"], r["probe"], f"{r['start']:.2f}", f"{r['final']:.6e}",
                             f"{r['variance']:.4e}", f"{r['variance_first_half']:.4e}", f"{r['variance_second_half']:.4e}",
                             f"{np.sqrt(r['variance']):.4e}", f"{r['peak']:.4f}", f"{r['peak_time']:.2f}"])

    # --- summary table (markdown)
    w0, w1 = args.window
    lines = [f"窗口 {w0:g} 到 {w1:g} s（脉冲后），趋势 {args.trend:g} s 滑动平均，方差相对 C_inf；sigma = sqrt(平均方差)", "",
             "| 数据集 | 探针 | 信号数 | 平均方差 | 中位方差 | sigma | 前半窗方差 | 后半窗方差 |", "|---|---|---|---|---|---|---|---|"]
    for label in datasets:
        for probe, s in summaries[label].items():
            lines.append(f"| {label} | {probe} | {s['n']} | {s['variance_mean']:.2e} | {s['variance_median']:.2e} | {s['sigma']:.4f} "
                         f"| {s['variance_first_half']:.2e} | {s['variance_second_half']:.2e} |")
    lines += ["", "首次到达峰值 max(C/C_inf)（脉冲后整段记录，不依赖去趋势）", "",
              "| 数据集 | 探针 | 峰值均值 | 峰值中位 | 峰值最大 | 峰值时刻均值 [s] |", "|---|---|---|---|---|---|"]
    for label in datasets:
        for probe, s in summaries[label].items():
            lines.append(f"| {label} | {probe} | {s['peak_mean']:.2f} | {s['peak_median']:.2f} | {s['peak_max']:.2f} | {s['peak_time_mean']:.1f} |")
    lines += ["", "| 比值 | 探针 1 方差比 | 探针 2 方差比 | 探针 1 sigma 比 | 探针 2 sigma 比 |", "|---|---|---|---|---|"]
    ratios = {}
    for label in datasets[1:]:
        entry = {}
        cells = []
        for probe in (1, 2):
            a, b = summaries["SPH"][probe], summaries[label][probe]
            entry[probe] = {"variance_ratio": a["variance_mean"] / b["variance_mean"], "sigma_ratio": a["sigma"] / b["sigma"]}
        cells = [f"{entry[1]['variance_ratio']:.2f}", f"{entry[2]['variance_ratio']:.2f}",
                 f"{entry[1]['sigma_ratio']:.2f}", f"{entry[2]['sigma_ratio']:.2f}"]
        lines.append(f"| SPH / {label} | " + " | ".join(cells) + " |")
        ratios[label] = entry
    lines += ["", "频带功率（谱积分，相对 C_inf 的方差份额）及比值", "",
              "| 数据集 | 探针 | " + " | ".join(f"{lo:g}-{hi:g} Hz" for lo, hi in BANDS) + " |",
              "|---|---|" + "---|" * len(BANDS)]
    for label in datasets:
        for probe, s in summaries[label].items():
            lines.append(f"| {label} | {probe} | " + " | ".join(f"{v:.2e}" for v in s["bands"].values()) + " |")
    for label in datasets[1:]:
        for probe in (1, 2):
            a, b = summaries["SPH"][probe]["bands"], summaries[label][probe]["bands"]
            lines.append(f"| SPH / {label} | {probe} | " + " | ".join(ratio_text(a[k], b[k]) for k in a) + " |")
    table = "\n".join(lines)
    print(table)
    (args.out_dir / f"{args.label}_probe_fluctuations_table.md").write_text(table + "\n", encoding="utf-8")
    with open(args.out_dir / f"{args.label}_probe_fluctuations_summary.json", "w", encoding="utf-8") as handle:
        json.dump({"window": list(args.window), "trend": args.trend, "segment": args.segment,
                   "summary": {label: {probe: {k: v for k, v in s.items() if k not in ("frequency", "psd_mean")}
                                       for probe, s in summaries[label].items()} for label in datasets},
                   "ratios": ratios}, handle, indent=1)

    # --- spectra figure
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True)
    colours = {"SPH": "tab:red", "M-Star": "tab:blue", "Fluent fine": "tab:green", "Fluent coarse": "tab:olive"}
    for axis, probe in zip(axes, (1, 2)):
        for label in datasets:
            s = summaries[label].get(probe)
            if s is None:
                continue
            mask = s["frequency"] > 0
            axis.loglog(s["frequency"][mask], s["psd_mean"][mask], color=colours[label], lw=1.4, label=f"{label} (n = {s['n']})")
        for multiple, text in ((1, "N"), (4, "4N (PBT blades)"), (6, "6N (Rushton blades)")):
            axis.axvline(multiple * IMPELLER_FREQUENCY, color="grey", lw=0.7, ls="--")
            axis.text(multiple * IMPELLER_FREQUENCY, 1e-12, text, rotation=90, fontsize=6.5, color="grey",
                      va="bottom", ha="right")
        axis.set_xlabel("f [Hz]")
        axis.set_title(f"probe {probe}")
        axis.grid(alpha=0.25, which="both")
        axis.legend(fontsize=7.5)
    axes[0].set_ylabel(r"PSD of $c' = C/C_\infty - \overline{C/C_\infty}^{1\,s}$  [1/Hz]")
    figure.suptitle(f"Tracer fluctuation spectra at the probes, {w0:g}-{w1:g} s after each pulse (mean over pulses)", fontsize=10)
    figure.tight_layout()
    figure.savefig(args.out_dir / f"{args.label}_probe_fluctuation_spectra.png", dpi=160)
    plt.close(figure)

    # --- example time series: pulse at 29 s, probe 1, first realisation of each dataset
    figure, axes = plt.subplots(2, 1, figsize=(10.5, 6.0), sharex=True)
    for label in datasets:
        candidates = [r for r in results if r["dataset"] == label and r["probe"] == 1 and abs(r["start"] - 29.0) < 0.5]
        if not candidates:
            continue
        r = candidates[0]
        axes[0].plot(r["time"] - r["start"], r["c"], color=colours[label], lw=0.9, label=f"{label} ({r['realisation']})")
        axes[0].plot(r["time"] - r["start"], r["trend"], color=colours[label], lw=0.9, ls="--", alpha=0.7)
        axes[1].plot(r["time"] - r["start"], r["residual"], color=colours[label], lw=0.8, label=label)
    axes[0].set_ylabel(r"$C/C_\infty$ and 1 s trend")
    axes[1].set_ylabel(r"residual $c'$")
    axes[1].set_xlabel("time after the pulse [s]")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7.5)
    figure.suptitle("Probe 1, pulse at 29 s: one realisation per dataset", fontsize=10)
    figure.tight_layout()
    figure.savefig(args.out_dir / f"{args.label}_probe_fluctuation_example.png", dpi=160)
    plt.close(figure)

    # --- variance distribution
    figure, axis = plt.subplots(figsize=(7.0, 3.8))
    positions, labels_out = [], []
    data = []
    for index, label in enumerate(datasets):
        for probe in (1, 2):
            data.append([r["variance"] for r in results if r["dataset"] == label and r["probe"] == probe])
            positions.append(index * 3 + probe)
            labels_out.append(f"{label}\nprobe {probe}")
    axis.boxplot(data, positions=positions, widths=0.8)
    axis.set_yscale("log")
    axis.set_xticks(positions)
    axis.set_xticklabels(labels_out, fontsize=7)
    axis.set_ylabel("variance of c' per pulse")
    axis.grid(alpha=0.25, which="both", axis="y")
    figure.tight_layout()
    figure.savefig(args.out_dir / f"{args.label}_probe_fluctuation_variance.png", dpi=160)
    plt.close(figure)
    print(f"wrote tables and figures to {args.out_dir}")


if __name__ == "__main__":
    main()
