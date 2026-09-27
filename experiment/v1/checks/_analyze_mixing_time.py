"""_analyze_mixing_time.py — mixing time tau95 of tracer pulses in the 30 L tank,
compared with the Rautenbach et al. (2026) dataset DARUS-5523 (2026-09-27).

tau95 follows the dataset's own post-processing script
(03_codes_for_post/mixing_times_in_simulation_10dyes_control_vol_filer_for_longest.py):

  C*(t) = (C(t) - C(first sample)) / (C(last sample) - C(first sample))
  tau95 = t_s - t_0,  t_0 = start of the pulse, t_s = time of the first sample
          after the LAST sample at t >= t_0 with C* outside [0.95, 1.05] such
          that this sample and the next two are inside the band (the search
          stops 3 samples before the end of the record, as in their loop)
  the reported tau95 of a pulse is the larger of the two probe values
  ("longest"), a probe that never settles is skipped.

With our conserved totals a second normalisation is available: the exact
well-mixed value C_inf = sum_i m_i C_i / sum_i m_i. tau95 is reported with
both. The probe log of _run_v1_headless.py also carries two global measures per
field, the mass-weighted coefficient of variation (cov:) and the mass fraction
within +-5 % of the mean (mixed5:); "global tau95" is the time after which
mixed5 stays >= 0.95. Finally the mean fluid speed is averaged from 25 s to the
end of the run, as Table 2 of the paper does over 25-75 s.

Usage (repo root):
  our run:  python experiment/v1/checks/_analyze_mixing_time.py PROBE_LOG.csv --case CASE.yaml
                [--torque-log TORQUE.csv] [--snapshots DIR] [--slice-field tracer_01]
                [--out-dir DIR] [--label NAME]
  dataset:  python experiment/v1/checks/_analyze_mixing_time.py --mstar PROBE_1.txt PROBE_2.txt
                (one M-Star trial of DARUS-5523, 02_simulation_results/01_mixing_time_results/
                 00_raw_txt/...; pulses at 25, 26, ... 34 s) - reproduces the dataset's values
                 and so checks this implementation.
"""
import argparse
import csv
import json
import math
import pathlib
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Reference data, DARUS-5523 05_final_results_collections/ (tau95 in s):
#   mixing_time_statistics_collection_paper_200rpm.tab  (means, standard deviations)
#   longest_mixing_times_mstar_exper_high_low_res_all_mstar.tab  (all realisations;
#   count, minimum and maximum below computed from it)
# Mean domain-averaged speed: Table 2 of the paper (rolling mean over 25-75 s).
# ---------------------------------------------------------------------------
REFERENCE_TAU95 = (
    # label,                         mean,  std,  count, minimum, maximum
    ("experiment",                   24.91, 2.71, 11, 19.00, 30.00),
    ("M-Star LBM-LES (LX400)",       29.10, 3.86, 50, 20.38, 37.54),
    ("Fluent FV-LES",                29.17, 4.42, 50, 19.90, 36.66),
    ("M-Star coarsest (LX080)",      23.73, 1.84, 60, 19.42, 28.64),
)
REFERENCE_MEAN_SPEED = (("M-Star", 0.149463), ("Fluent", 0.156604))
PROBE_NAMES = ("probe_1", "probe_2")
INJECTION_POINT = (0.0, 0.4055, 0.1164)
PROBE_POINTS = ((-0.1270, 0.4015, -0.0180), (0.0530, 0.4015, -0.1180))
BAND = 0.05
STABLE_SAMPLES = 3


def tau95(time, value, start, final_value=None, band=BAND, stable_samples=STABLE_SAMPLES):
    """Mixing time of one probe signal after a pulse starting at `start`
    (dataset rule, see the module docstring). final_value replaces the last
    sample as C_inf when given. NaN when the signal never settles."""
    first = value[0]
    final = value[-1] if final_value is None else final_value
    if not np.isfinite(final) or final == first:
        return float("nan")
    normalised = (value - first) / (final - first)
    outside = (time >= start) & ((normalised < 1.0 - band) | (normalised > 1.0 + band))
    if not outside.any():
        return float("nan")
    last_outside = int(np.nonzero(outside)[0].max())
    for index in range(last_outside + 1, len(time) - stable_samples):
        window = normalised[index:index + stable_samples]
        if np.all((window >= 1.0 - band) & (window <= 1.0 + band)):
            return float(time[index] - start)
    return float("nan")


def longest(values):
    """The dataset's per-pulse tau95: the larger of the defined probe values."""
    defined = [value for value in values if np.isfinite(value)]
    return max(defined) if defined else float("nan")


def summary(values):
    values = np.asarray([value for value in values if np.isfinite(value)])
    if values.size == 0:
        return {"count": 0, "mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {"count": int(values.size), "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if values.size > 1 else float("nan"),
            "min": float(values.min()), "max": float(values.max())}


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------

def read_csv_columns(path, delimiter=","):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle, delimiter=delimiter))
    header = [name.strip() for name in rows[0]]
    data = np.array([[float(cell) if cell.strip() not in ("", "nan") else np.nan for cell in row]
                     for row in rows[1:] if len(row) == len(header)])
    return {name: data[:, index] for index, name in enumerate(header)}


def pulse_starts(case_path):
    """{field: start of its pulse} from the case's scalars.injections."""
    import yaml
    with open(case_path, encoding="utf-8") as handle:
        case = yaml.safe_load(handle)
    starts = {}
    for injection in case["scalars"]["injections"]:
        starts.setdefault(injection["field"], float(injection["start"]))
    spacing = 2.0 * float(case["physics"]["particle_radius"])
    return starts, spacing


# ---------------------------------------------------------------------------
# Dataset mode: one M-Star trial
# ---------------------------------------------------------------------------

def analyze_mstar(probe_files):
    columns = [read_csv_columns(path, delimiter="\t") for path in probe_files]
    print(f"M-Star trial: {', '.join(str(path) for path in probe_files)}")
    print(f"  record {columns[0]['Time [s]'][0]:.3f} .. {columns[0]['Time [s]'][-1]:.2f} s")
    dye_columns = sorted(name for name in columns[0] if name.startswith("dye_") and "Mean" in name)
    results = []
    for dye_index, name in enumerate(dye_columns):
        start = 25.0 + dye_index
        per_probe = [tau95(column["Time [s]"], column[name], start) for column in columns]
        results.append(longest(per_probe))
        print(f"  {name[:7]}  start {start:4.1f} s   probe 1 {per_probe[0]:6.2f}   probe 2 {per_probe[1]:6.2f}"
              f"   longest {results[-1]:6.2f} s")
    stats = summary(results)
    print(f"  longest tau95: mean {stats['mean']:.2f} s, std {stats['std']:.2f} s, n = {stats['count']}")
    return results


# ---------------------------------------------------------------------------
# Our run
# ---------------------------------------------------------------------------

def analyze_run(arguments):
    log = read_csv_columns(arguments.probe_log)
    starts, spacing = pulse_starts(arguments.case)
    time = log["time"]
    fields = [name.split(":", 1)[1] for name in log if name.startswith("total:")]
    fluid_mass = log["fluid_mass"]
    out_dir = pathlib.Path(arguments.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    label = arguments.label or pathlib.Path(arguments.probe_log).stem

    print(f"{label}: {len(time)} samples, t = {time[0]:.3f} .. {time[-1]:.2f} s, "
          f"sampling {np.median(np.diff(time)):.4f} s, fields {len(fields)}")
    rows = []
    for field in fields:
        start = starts.get(field, float("nan"))
        total = log[f"total:{field}"]
        exact_final = total[-1] / fluid_mass[-1]
        per_probe = [tau95(time, log[f"{probe}:{field}"], start) for probe in PROBE_NAMES]
        per_probe_exact = [tau95(time, log[f"{probe}:{field}"], start, exact_final) for probe in PROBE_NAMES]
        mixed = log.get(f"mixed5:{field}")
        global_tau = float("nan")
        if mixed is not None:
            after = time >= start
            unmixed = after & ~(mixed >= 0.95)
            if after.any() and unmixed.any():
                last = int(np.nonzero(unmixed)[0].max())
                if last + 1 < len(time):
                    global_tau = float(time[last + 1] - start)
        pulse_end_total = total[time >= start + 1.0]
        drift = (float((pulse_end_total[-1] - pulse_end_total[0]) / pulse_end_total[0])
                 if pulse_end_total.size > 1 and pulse_end_total[0] > 0 else float("nan"))
        rows.append({"field": field, "start": start, "tau95_probe_1": per_probe[0], "tau95_probe_2": per_probe[1],
                     "tau95": longest(per_probe), "tau95_exact_probe_1": per_probe_exact[0],
                     "tau95_exact_probe_2": per_probe_exact[1], "tau95_exact": longest(per_probe_exact),
                     "global_tau95": global_tau, "final_cov": float(log[f"cov:{field}"][-1]) if f"cov:{field}" in log else float("nan"),
                     "record_after_start": float(time[-1] - start), "total_drift_after_pulse": drift,
                     "last_over_exact_probe_1": float(log[f"probe_1:{field}"][-1] / exact_final) if exact_final > 0 else float("nan"),
                     "last_over_exact_probe_2": float(log[f"probe_2:{field}"][-1] / exact_final) if exact_final > 0 else float("nan")})

    print(f"  {'field':10s} {'start':>6s} {'rec.':>6s} | {'P1':>6s} {'P2':>6s} {'tau95':>6s} | "
          f"{'P1 ex':>6s} {'P2 ex':>6s} {'ex':>6s} | {'global':>6s} {'CoV end':>8s} {'drift':>9s} {'P1/Cinf':>7s} {'P2/Cinf':>7s}")
    for row in rows:
        print(f"  {row['field']:10s} {row['start']:6.2f} {row['record_after_start']:6.2f} | "
              f"{row['tau95_probe_1']:6.2f} {row['tau95_probe_2']:6.2f} {row['tau95']:6.2f} | "
              f"{row['tau95_exact_probe_1']:6.2f} {row['tau95_exact_probe_2']:6.2f} {row['tau95_exact']:6.2f} | "
              f"{row['global_tau95']:6.2f} {row['final_cov']:8.4f} {row['total_drift_after_pulse']:9.2e} "
              f"{row['last_over_exact_probe_1']:7.3f} {row['last_over_exact_probe_2']:7.3f}")
    stats = summary([row["tau95"] for row in rows])
    stats_exact = summary([row["tau95_exact"] for row in rows])
    stats_global = summary([row["global_tau95"] for row in rows])
    print(f"  tau95 (dataset rule, longest probe): mean {stats['mean']:.2f} s, std {stats['std']:.2f} s, "
          f"n = {stats['count']}, range {stats['min']:.2f} .. {stats['max']:.2f} s")
    print(f"  tau95 (exact C_inf):                 mean {stats_exact['mean']:.2f} s, std {stats_exact['std']:.2f} s, n = {stats_exact['count']}")
    print(f"  global tau95 (95 % of mass within 5 %): mean {stats_global['mean']:.2f} s, n = {stats_global['count']}")
    for name, mean, std, count, minimum, maximum in REFERENCE_TAU95:
        print(f"  reference {name:26s} {mean:6.2f} +- {std:4.2f} s (n = {count}, {minimum:.1f} .. {maximum:.1f} s)")

    speed = {}
    if "mean_speed" in log:
        window = time >= 25.0
        if window.any():
            speed = {"window_start": 25.0, "window_end": float(time[window][-1]),
                     "mean_speed": float(np.nanmean(log["mean_speed"][window]))}
            print(f"  mean fluid speed over {speed['window_start']:.0f} .. {speed['window_end']:.1f} s: "
                  f"{speed['mean_speed']:.4f} m/s (paper Table 2, 25 .. 75 s: "
                  + ", ".join(f"{name} {value:.4f}" for name, value in REFERENCE_MEAN_SPEED) + ")")

    power = {}
    if arguments.torque_log:
        torque = read_csv_columns(arguments.torque_log)
        window = torque["time"] >= 20.0
        if window.sum() > 2:
            scale = 998.0 * (200.0 / 60.0) ** 3 * 0.096 ** 5
            omega = 2.0 * math.pi * 200.0 / 60.0
            power_number = np.abs(torque["torque_axis"][window]) * omega / scale
            power = {"window_start": 20.0, "power_number": float(power_number.mean()),
                     "power_number_std": float(power_number.std(ddof=1)), "samples": int(window.sum())}
            print(f"  power number (t >= 20 s, readback torque): {power['power_number']:.2f} +- "
                  f"{power['power_number_std']:.2f} ({power['samples']} samples)")

    result = {"label": label, "probe_log": str(arguments.probe_log), "case": str(arguments.case), "pulses": rows,
              "tau95": stats, "tau95_exact": stats_exact, "global_tau95": stats_global, "mean_speed": speed,
              "power": power, "reference": [dict(zip(("label", "mean", "std", "count", "min", "max"), entry))
                                            for entry in REFERENCE_TAU95]}
    (out_dir / f"{label}_mixing.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    plot_run(log, rows, starts, fields, out_dir, label)
    if arguments.snapshots:
        plot_snapshots(pathlib.Path(arguments.snapshots), spacing, arguments.slice_field, out_dir, label)
    print(f"  wrote {out_dir / (label + '_mixing.json')} and figures")
    return result


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_run(log, rows, starts, fields, out_dir, label):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    time = log["time"]

    # 1. normalised probe signals, one panel per pulse
    count = len(fields)
    columns = min(count, 5)
    lines = int(math.ceil(count / columns))
    figure, axes = plt.subplots(lines, columns, figsize=(3.6 * columns, 2.8 * lines), squeeze=False, sharey=True)
    for index, (field, row) in enumerate(zip(fields, rows)):
        axis = axes[index // columns][index % columns]
        start = starts.get(field, float("nan"))
        for probe, colour in zip(PROBE_NAMES, ("tab:cyan", "tab:purple")):
            value = log[f"{probe}:{field}"]
            final = value[-1]
            if final != value[0]:
                axis.plot(time - start, (value - value[0]) / (final - value[0]), color=colour, lw=1.0, label=probe)
        axis.axhspan(1.0 - BAND, 1.0 + BAND, color="0.85", zorder=0)
        if np.isfinite(row["tau95"]):
            axis.axvline(row["tau95"], color="tab:red", ls="--", lw=1.0)
        axis.set_xlim(-1.0, max(1.0, time[-1] - start))
        axis.set_ylim(-0.1, 2.0)
        axis.set_title(f"{field}: tau95 = {row['tau95']:.1f} s", fontsize=9)
        axis.set_xlabel("t - t_inj [s]", fontsize=8)
        if index % columns == 0:
            axis.set_ylabel("C* (last sample = 1)", fontsize=8)
        if index == 0:
            axis.legend(fontsize=7)
    for index in range(count, lines * columns):
        axes[index // columns][index % columns].axis("off")
    figure.suptitle(f"{label}: probe signals (dataset normalisation)")
    figure.tight_layout()
    figure.savefig(out_dir / f"{label}_probe_signals.png", dpi=130)
    plt.close(figure)

    # 2. tau95 against the reference distributions
    figure, axis = plt.subplots(figsize=(7.5, 4.0))
    names = [entry[0] for entry in REFERENCE_TAU95] + [f"this run ({label})"]
    for position, (name, mean, std, count_reference, minimum, maximum) in enumerate(REFERENCE_TAU95):
        axis.plot([position, position], [minimum, maximum], color="0.6", lw=6, solid_capstyle="butt")
        axis.errorbar(position, mean, yerr=std, fmt="o", color="k", capsize=4)
    ours = [row["tau95"] for row in rows if np.isfinite(row["tau95"])]
    position = len(REFERENCE_TAU95)
    if ours:
        axis.scatter(np.full(len(ours), position) + np.linspace(-0.12, 0.12, len(ours)), ours, color="tab:red", zorder=3)
        if len(ours) > 1:
            axis.errorbar(position + 0.25, np.mean(ours), yerr=np.std(ours, ddof=1), fmt="s", color="tab:red", capsize=4)
    axis.set_xticks(range(len(names)))
    axis.set_xticklabels(names, rotation=15, fontsize=8)
    axis.set_ylabel("tau95 [s]")
    axis.set_title("mixing time: range (bar), mean +- std (marker)")
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(out_dir / f"{label}_tau95_vs_reference.png", dpi=130)
    plt.close(figure)

    # 3. global mixing indicators and mean speed
    figure, (left, middle, right) = plt.subplots(1, 3, figsize=(14, 3.8))
    for field in fields:
        start = starts.get(field, float("nan"))
        if f"cov:{field}" in log:
            left.semilogy(time - start, log[f"cov:{field}"], lw=0.8)
            middle.plot(time - start, log[f"mixed5:{field}"], lw=0.8)
    left.axhline(BAND, color="k", ls=":", lw=0.8)
    left.set_xlabel("t - t_inj [s]"); left.set_ylabel("CoV (mass-weighted)"); left.grid(alpha=0.3)
    middle.axhline(0.95, color="k", ls=":", lw=0.8)
    middle.set_xlabel("t - t_inj [s]"); middle.set_ylabel("mass fraction within +-5 %"); middle.grid(alpha=0.3)
    if "mean_speed" in log:
        right.plot(time, log["mean_speed"], lw=0.8, color="tab:blue", label="mean fluid speed")
        window = time >= 25.0
        if window.any():
            rolling = np.cumsum(log["mean_speed"][window]) / np.arange(1, window.sum() + 1)
            right.plot(time[window], rolling, color="tab:red", lw=1.2, label="rolling mean from 25 s")
        for (name, value), style in zip(REFERENCE_MEAN_SPEED, ("--", ":")):
            right.axhline(value, color="k", ls=style, lw=0.8, label=f"{name} (Table 2)")
        right.set_xlabel("t [s]"); right.set_ylabel("speed [m/s]"); right.legend(fontsize=7); right.grid(alpha=0.3)
    figure.suptitle(f"{label}: global mixing and mean speed")
    figure.tight_layout()
    figure.savefig(out_dir / f"{label}_global_mixing.png", dpi=130)
    plt.close(figure)


def plot_snapshots(snapshot_dir, spacing, slice_field, out_dir, label):
    """Two slices per snapshot, coloured by log10(C / C_mean): the vertical plane
    x = 0 (through the axis and the injection point) and the horizontal plane at
    the probe height y = 0.4015 m."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths = sorted(snapshot_dir.glob("snapshot_*.npz"))
    if not paths:
        print(f"  no snapshots in {snapshot_dir}")
        return
    figure, axes = plt.subplots(2, len(paths), figsize=(2.6 * len(paths), 6.6), squeeze=False)
    half = 0.75 * spacing
    image = None
    for column, path in enumerate(paths):
        archive = np.load(path)
        names = [str(name) for name in archive["field_names"]]
        field = slice_field if slice_field in names else names[0]
        values = archive["scalars"][:, names.index(field)].astype(np.float64)
        mass = archive["mass"].astype(np.float64)
        positions = archive["positions"].astype(np.float64)
        mean = float(np.dot(mass, values) / mass.sum())
        with np.errstate(divide="ignore"):
            level = np.log10(np.maximum(values, 1e-30) / mean) if mean > 0 else np.zeros_like(values)
        level = np.clip(level, -2.0, 2.0)
        vertical = np.abs(positions[:, 0]) < half
        horizontal = np.abs(positions[:, 1] - PROBE_POINTS[0][1]) < half
        top, bottom = axes[0][column], axes[1][column]
        image = top.scatter(positions[vertical, 2], positions[vertical, 1], c=level[vertical], s=1.0,
                            cmap="RdBu_r", vmin=-2, vmax=2, linewidths=0)
        top.plot(INJECTION_POINT[2], INJECTION_POINT[1], "k+", ms=8)
        top.set_aspect("equal"); top.set_xticks([]); top.set_yticks([])
        top.set_title(f"{field}\nt = {float(archive['time']):.2f} s", fontsize=8)
        bottom.scatter(positions[horizontal, 0], positions[horizontal, 2], c=level[horizontal], s=1.0,
                       cmap="RdBu_r", vmin=-2, vmax=2, linewidths=0)
        for x, _, z in PROBE_POINTS:
            bottom.plot(x, z, "kx", ms=6)
        bottom.plot(INJECTION_POINT[0], INJECTION_POINT[2], "k+", ms=8)
        bottom.set_aspect("equal"); bottom.set_xticks([]); bottom.set_yticks([])
    axes[0][0].set_ylabel("x = 0 plane (z right, y up)", fontsize=8)
    axes[1][0].set_ylabel("y = 0.4015 m plane (x right, z up)", fontsize=8)
    if image is not None:
        colour_bar = figure.colorbar(image, ax=axes, shrink=0.6, location="right")
        colour_bar.set_label("log10(C / C_mean)")
    figure.suptitle(f"{label}: tracer slices (+ injection, x probes)")
    figure.savefig(out_dir / f"{label}_slices.png", dpi=140, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("probe_log", nargs="?", help="probe CSV of _run_v1_headless.py (--probe-log)")
    parser.add_argument("--case", help="case.yaml of the run (pulse start times, particle spacing)")
    parser.add_argument("--torque-log", default=None, help="torque CSV of the same run (power number from 20 s)")
    parser.add_argument("--snapshots", default=None, help="directory of --scalar-snapshot-times output")
    parser.add_argument("--slice-field", default="tracer_01")
    parser.add_argument("--out-dir", default="output/mixing")
    parser.add_argument("--label", default=None)
    parser.add_argument("--mstar", nargs=2, metavar=("PROBE_1", "PROBE_2"),
                        help="analyse one M-Star trial of the dataset instead of our run")
    arguments = parser.parse_args()
    if arguments.mstar:
        analyze_mstar(arguments.mstar)
        return 0
    if not arguments.probe_log or not arguments.case:
        parser.error("give PROBE_LOG and --case, or --mstar PROBE_1 PROBE_2")
    analyze_run(arguments)
    return 0


if __name__ == "__main__":
    sys.exit(main())
