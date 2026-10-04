"""
_compare_wall_torques.py — flow-induced wall torques of several tank cases in the same flow (2026-10-04).

Written for the baffle study of log/2026-10-04_baffle-representation-resolution.md: the same Fluent field is
put on cases that differ in resolution or baffle representation (_check_tank_energy.py --initial-velocity),
and each case is also run at rest. Under a background pressure the discretised walls carry static torques
at rest (log/2026-10-04_baffle-drag-diagnosis.md: thin plates -20..-32 mN m, the lattice brackets next to
them +29); they cancel in the sum over all walls, not per class. So for every class

    flow-induced torque = mean over --window of the stirred run - mean over --rest-window of the run at rest

and the sum over all walls (csv_walls) is the number that does not depend on how the static part is split.

    python experiment/v1/checks/_compare_wall_torques.py \
        --case "thin plates" 3 output/kecause/rest_A.csv output/kecause/A_fluentinit.csv \
        --case "2-layer sheets" 3 output/kecause/rest_B.csv output/kecause/B_fluentinit.csv \
        [--window 0.3 0.7] [--rest-window 0.02 0.21] [--figure out.png]

--case FAMILY DX_MM REST_CSV RUN_CSV: cases of one family are joined by a line in the figure (torque against
the particle spacing). Torques in mN m, braking positive; impeller torques as the fluid receives them.
The +- columns are the standard error of the window mean, estimated from the means of --block long blocks:
sample standard deviation of the block means / sqrt(number of blocks), as if the blocks were independent.
"""
import argparse
import pathlib

import numpy as np

# Fluent fine mesh, 25..34 s (log/2026-10-03_fluent-and-wall-torque.md, _analyze_fluent_impeller_parts.py)
FLUENT = {"baffles": 69.7, "cylinder": 5.6, "floor": 2.46, "rushton": 60.7, "pbt": 15.0, "angular_momentum": 229.0}
FLUENT["walls"] = FLUENT["baffles"] + FLUENT["cylinder"] + FLUENT["floor"]


def window_means(path, start, end):
    data = np.genfromtxt(path, delimiter=",", names=True)
    inside = (data["time"] >= start) & (data["time"] < end)
    if not inside.any():
        return None, 0
    return {name: float(data[name][inside].mean()) * 1e3 for name in data.dtype.names}, int(inside.sum())


def flow_induced(rest, run):
    plates = sum(run[name] - rest[name] for name in run if name.startswith("plate_baffle"))
    walls_next_to_baffles = sum(run[name] - rest[name] for name in run if name.startswith("wall_at_baffle"))
    row = {"baffles": plates + walls_next_to_baffles, "plates": plates, "walls_next_to_baffles": walls_next_to_baffles,
           "rushton": -run["rotor_lower"], "pbt": -run["rotor_upper"],
           "floor": run["floor"] - rest["floor"], "cylinder": run["cylinder"] - rest["cylinder"],
           "lid": run["lid"] - rest["lid"], "walls": run["csv_walls"] - rest["csv_walls"],
           "walls_at_rest": rest["csv_walls"], "angular_momentum": run["angular_momentum"]}
    if "below_split_walls" in run and "below_split_walls" in rest:
        row["below_split"] = run["below_split_walls"] - rest["below_split_walls"]
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", nargs=4, action="append", required=True,
                        metavar=("FAMILY", "DX_MM", "REST_CSV", "RUN_CSV"))
    parser.add_argument("--window", type=float, nargs=2, default=(0.3, 0.7))
    parser.add_argument("--rest-window", type=float, nargs=2, default=(0.02, 0.21))
    parser.add_argument("--block", type=float, default=0.1, help="block length for the standard error, s")
    parser.add_argument("--figure", default=None)
    args = parser.parse_args()

    rows = []
    for family, spacing, rest_path, run_path in args.case:
        label = f"{float(spacing):g} mm {family}"
        if not (pathlib.Path(rest_path).exists() and pathlib.Path(run_path).exists()):
            print(f"  {label}: not run yet")
            continue
        rest, rest_samples = window_means(rest_path, *args.rest_window)
        run, run_samples = window_means(run_path, *args.window)
        if rest is None or run is None:
            print(f"  {label}: window not reached yet")
            continue
        row = flow_induced(rest, run)
        row.update(family=family, spacing=float(spacing), label=label, samples=(rest_samples, run_samples))
        blocks = []
        for start in np.arange(args.window[0], args.window[1] - 1e-9, args.block):
            block, _ = window_means(run_path, start, min(start + args.block, args.window[1]))
            if block is not None:
                blocks.append(flow_induced(rest, block))
        for key in ("baffles", "walls"):
            values = np.array([block[key] for block in blocks])
            row[key + "_error"] = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else float("nan")
        rows.append(row)

    print(f"Flow-induced torques, mN m: mean over {args.window[0]:g}..{args.window[1]:g} s of the stirred run minus "
          f"the mean over {args.rest_window[0]:g}..{args.rest_window[1]:g} s at rest; braking positive")
    header = (f"  {'case':24s} {'baffles':>7s} {'+-':>4s} {'plates':>7s} {'next to':>8s} {'cylinder':>9s} {'floor':>6s} "
              f"{'lid':>6s} {'all walls':>10s} {'+-':>4s} {'at rest':>8s} {'below':>6s} {'Rushton':>8s} {'PBT':>6s} {'L':>5s} "
              f"{'samples':>8s}")
    print(header)
    for row in rows:
        print(f"  {row['label']:24s} {row['baffles']:7.1f} {row['baffles_error']:4.1f} {row['plates']:7.1f} "
              f"{row['walls_next_to_baffles']:8.1f} {row['cylinder']:9.1f} {row['floor']:6.2f} {row['lid']:6.2f} "
              f"{row['walls']:10.1f} {row['walls_error']:4.1f} {row['walls_at_rest']:8.1f} "
              f"{row.get('below_split', float('nan')):6.1f} {row['rushton']:8.1f} {row['pbt']:6.1f} "
              f"{row['angular_momentum']:5.0f} {row['samples'][0]:3d}/{row['samples'][1]:<3d}")
    print(f"  {'Fluent fine 25..34 s':24s} {FLUENT['baffles']:7.1f} {'':4s} {'':7s} {'':8s} {FLUENT['cylinder']:9.1f} "
          f"{FLUENT['floor']:6.2f} {'':6s} {FLUENT['walls']:10.1f} {'':4s} {'':8s} {'':6s} {FLUENT['rushton']:8.1f} "
          f"{FLUENT['pbt']:6.1f} {FLUENT['angular_momentum']:5.0f}")
    print("  (baffles = plates + the ordinary wall particles next to them; below = every wall record below the "
          "split height; L = fluid angular momentum, g m^2/s, window mean)")

    if args.figure and rows:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharex=True)
        families = list(dict.fromkeys(row["family"] for row in rows))
        markers = ["o", "s", "^", "D", "v"]
        for index, family in enumerate(families):
            members = sorted((row for row in rows if row["family"] == family), key=lambda row: row["spacing"])
            spacing = [row["spacing"] for row in members]
            for axis, key in zip(axes, ("baffles", "walls")):
                axis.errorbar(spacing, [row[key] for row in members], yerr=[row[key + "_error"] for row in members],
                              marker=markers[index % len(markers)], capsize=3, label=family)
        for axis, key, title in zip(axes, ("baffles", "walls"),
                                    ("baffles with their brackets, flow-induced", "all static walls, flow-induced")):
            axis.axhline(FLUENT[key], color="k", linestyle="--", linewidth=1.0, label="Fluent LES")
            axis.set_title(title, fontsize=10)
            axis.set_xlabel("particle spacing, mm")
            axis.set_ylabel("braking torque, mN m")
            axis.set_ylim(0.0, 1.15 * max(FLUENT["walls"], max(row["walls"] for row in rows)))
            axis.grid(alpha=0.3)
        axes[0].invert_xaxis()                   # shared x axis: finer spacing to the right, once for both
        axes[0].legend(fontsize=8, loc="lower right")
        figure.suptitle(f"30 L tank in Fluent's 25 s field, {args.window[0]:g}..{args.window[1]:g} s after the start, "
                        f"static torques at rest subtracted", fontsize=10)
        figure.tight_layout()
        figure.savefig(args.figure, dpi=150)
        print(f"figure -> {args.figure}")


if __name__ == "__main__":
    main()
