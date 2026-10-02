"""
_analyze_tank_energy.py — tables and a figure from the _check_tank_energy.py logs (2026-10-03).

    python experiment/v1/checks/_analyze_tank_energy.py output/kecause [--window 4 6] [--figure out.png]

rest*: the tank at rest. Kinetic energy and bulk mean speed over time, density drift, the load on the
baffles. s3* (or any other stirred run): means over the window: rotor torque and power, zone kinetic
energy (Fluent's rotor boxes and the bulk; physical mass rho0 dx^3) and the bulk energy per velocity
component, KE/P, the wall torque split, the load of each baffle (plate + the ordinary wall particles
next to it), pressure and density range. Reference: Fluent fine mesh 25..34 s and the production runs
(total kinetic energy from their budget logs, divided by the mass factor).
"""
import argparse
import pathlib

import numpy as np

OMEGA = 2.0 * np.pi * 200.0 / 60.0
MASS_FACTOR = 1.2187
FLUENT = {"ke_rushton": 0.0948, "ke_pbt": 0.0563, "ke_bulk": 0.4937, "ke_total": 0.6449, "speed_bulk": 0.1465,
          "rotor": -75.75e-3, "power": 1.586,
          # bulk_liquid per velocity component and |mean u_theta| (_fluent_snapshot_reports.py components)
          "ke_bulk_radial": 0.0975, "ke_bulk_tangential": 0.2305, "ke_bulk_axial": 0.1658, "u_tangential_bulk": 0.0828}
# u_tangential_bulk and angular_momentum: positive along the rotor rotation (the overnight CSV files of
# 2026-10-03 were written with the opposite sign and flipped in place after the runs).
TANGENTIAL_SIGN = 1.0
PRODUCTION = {  # name: (torque csv, budget csv), relative to the repo output directory
    "production 3 mm plates, 6 s run": ("thin_plate/paratera/pb2000/plates_true_pb2000_budget_torque.csv",
                                        "thin_plate/paratera/pb2000/plates_true_pb2000_budget_budget.csv"),
    "production 3 mm plates, 30 s run": ("thin_plate/paratera/pb2000/plates_true_pb2000_30s_budget_torque.csv",
                                         "thin_plate/paratera/pb2000/plates_true_pb2000_30s_budget_budget.csv"),
    "production 3 mm sheets, 30 s run": ("thin_plate/paratera/pb2000/conf1_true_pb2000_30s_budget_torque.csv",
                                         "thin_plate/paratera/pb2000/conf1_true_pb2000_30s_budget_budget.csv"),
}


def load(path):
    data = np.genfromtxt(path, delimiter=",", names=True)
    return data if data.size > 1 else None


def rest_table(directory):
    runs = sorted(directory.glob("rest*.csv"))
    if not runs:
        return
    print("\nTank at rest: total kinetic energy, mJ / bulk mean speed, mm/s")
    checkpoints = (0.1, 0.25, 0.5, 1.0, 2.0, 3.0)
    print("  run                        " + "".join(f"{f't={t:g} s':>16s}" for t in checkpoints)
          + "   density drift   baffles, mN m")
    for path in runs:
        data = load(path)
        if data is None:
            continue
        cells = []
        for t in checkpoints:
            index = np.argmin(np.abs(data["time"] - t))
            if abs(data["time"][index] - t) > 0.05:
                cells.append(f"{'':>16s}")
                continue
            cells.append(f"{data['ke_total'][index] * 1e3:8.1f} / {data['speed_bulk'][index] * 1e3:5.1f}")
        last = data["time"] >= data["time"][-1] - 1.0
        drift = data["density_mean"][-1] - data["density_mean"][0]
        baffles = data["csv_baffles"][last].mean() * 1e3
        print(f"  {path.stem:26s} " + "".join(cells) + f"   {drift:+8.3f}       {baffles:+7.1f}")


def window_mean(data, column, window):
    selected = (data["time"] >= window[0]) & (data["time"] < window[1])
    return data[column][selected].mean() if selected.any() else np.nan


def stirred_table(directory, window, repo_output):
    runs = [p for p in sorted(directory.glob("*.csv")) if not p.stem.startswith("rest")]
    rows = []
    for path in runs:
        data = load(path)
        if data is None or data["time"][-1] < window[0] + 0.2:
            print(f"  ({path.stem}: only up to t = {data['time'][-1] if data is not None else 0:.2f} s, skipped)")
            continue
        mean = lambda column: window_mean(data, column, window)
        power = -mean("rotor") * OMEGA
        row = {"run": path.stem, "rotor": mean("rotor"), "lower": mean("rotor_lower"), "upper": mean("rotor_upper"),
               "power": power}
        for column in ("ke_rushton", "ke_pbt", "ke_bulk", "ke_total", "ke_bulk_radial", "ke_bulk_tangential",
                       "ke_bulk_axial", "speed_rushton", "speed_pbt", "speed_bulk", "u_tangential_bulk",
                       "csv_walls", "csv_baffles", "cylinder", "lid", "floor", "negative_pressure_fraction",
                       "pressure_q01", "density_min", "density_max"):
            row[column] = mean(column)
        for k in (1, 2, 3):
            plate = mean(f"plate_baffle_{k}") if f"plate_baffle_{k}" in data.dtype.names else 0.0
            row[f"baffle_{k}"] = plate + mean(f"wall_at_baffle_{k}")
        row["end"] = data["time"][-1]
        rows.append(row)
    if not rows:
        return
    print(f"\nStirred runs, means over t = {window[0]:g}..{window[1]:g} s (torques mN m, energies J, speeds m/s)")
    print("  run                          rotor  Rushton   PBT   power W  KE Rt   KE PBT  KE bulk  bulk/Fluent  KE total  KE/P s")
    print(f"  {'Fluent fine, 25..34 s':28s} {FLUENT['rotor'] * 1e3:6.1f}   -60.7  -15.0   {FLUENT['power']:6.3f}  "
          f"{FLUENT['ke_rushton']:.3f}  {FLUENT['ke_pbt']:.3f}   {FLUENT['ke_bulk']:.3f}      1.00      {FLUENT['ke_total']:.3f}   "
          f"{FLUENT['ke_total'] / FLUENT['power']:.3f}")
    for row in rows:
        print(f"  {row['run']:28s} {row['rotor'] * 1e3:6.1f}  {row['lower'] * 1e3:6.1f} {row['upper'] * 1e3:6.1f}   "
              f"{row['power']:6.3f}  {row['ke_rushton']:.3f}  {row['ke_pbt']:.3f}   {row['ke_bulk']:.3f}      "
              f"{row['ke_bulk'] / FLUENT['ke_bulk']:.2f}      {row['ke_total']:.3f}   {row['ke_total'] / row['power']:.3f}")
    print("\n  run                          bulk |u|  bulk u_theta  bulk KE: radial  tangential  axial   (ratio to Fluent)")
    print(f"  {'Fluent fine, 25..34 s':28s}  {FLUENT['speed_bulk']:.3f}     {FLUENT['u_tangential_bulk']:+.3f}        "
          f"{FLUENT['ke_bulk_radial']:.3f}       {FLUENT['ke_bulk_tangential']:.3f}    {FLUENT['ke_bulk_axial']:.3f}")
    for row in rows:
        ratios = " / ".join(f"{row[c] / FLUENT[c]:.2f}" for c in ("ke_bulk_radial", "ke_bulk_tangential", "ke_bulk_axial"))
        print(f"  {row['run']:28s}  {row['speed_bulk']:.3f}     {TANGENTIAL_SIGN * row['u_tangential_bulk']:+.3f}        "
              f"{row['ke_bulk_radial']:.3f}       {row['ke_bulk_tangential']:.3f}    {row['ke_bulk_axial']:.3f}   ({ratios})")
    print("\n  run                          walls  baffles (csv)  cylinder   baffle 1  baffle 2  baffle 3   p<0 share   p 1 %   density min..max")
    for row in rows:
        print(f"  {row['run']:28s} {row['csv_walls'] * 1e3:6.1f}  {row['csv_baffles'] * 1e3:8.1f}     "
              f"{row['cylinder'] * 1e3:6.1f}    {row['baffle_1'] * 1e3:6.1f}    {row['baffle_2'] * 1e3:6.1f}    "
              f"{row['baffle_3'] * 1e3:6.1f}    {row['negative_pressure_fraction']:.4f}   {row['pressure_q01']:7.0f}   "
              f"{row['density_min']:.1f}..{row['density_max']:.1f}")
    print("\n  production runs, same window (total KE from the budget log / mass factor):")
    for name, (torque_file, budget_file) in PRODUCTION.items():
        torque_path, budget_path = repo_output / torque_file, repo_output / budget_file
        if not (torque_path.exists() and budget_path.exists()):
            continue
        torque, budget = load(torque_path), load(budget_path)
        rotor = window_mean(torque, "torque_axis", window) / MASS_FACTOR
        kinetic = window_mean(budget, "kinetic_energy", window) / MASS_FACTOR
        print(f"  {name:34s} rotor {rotor * 1e3:6.1f}  KE total {kinetic:.3f}  KE/P {kinetic / (-rotor * OMEGA):.3f}")


def figure(directory, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for path in sorted(directory.glob("rest*.csv")):
        data = load(path)
        if data is not None:
            axes[0].semilogy(data["time"], data["ke_total"] * 1e3, label=path.stem)
    axes[0].set_xlabel("t, s")
    axes[0].set_ylabel("kinetic energy at rest, mJ")
    axes[0].legend(fontsize=8)
    axes[0].set_title("4 mm tank at rest")
    for path in sorted(directory.glob("*.csv")):
        if path.stem.startswith("rest"):
            continue
        data = load(path)
        if data is not None:
            axes[1].plot(data["time"], data["ke_bulk"], label=path.stem)
    axes[1].axhline(FLUENT["ke_bulk"], color="k", linestyle="--", label="Fluent fine, 25..34 s")
    axes[1].set_xlabel("t, s")
    axes[1].set_ylabel("bulk kinetic energy, J")
    axes[1].legend(fontsize=8)
    axes[1].set_title("3 mm stirred")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"\nwrote {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory")
    parser.add_argument("--window", type=float, nargs=2, default=(4.0, 6.0))
    parser.add_argument("--figure", default=None)
    arguments = parser.parse_args()
    directory = pathlib.Path(arguments.directory)
    repo_output = pathlib.Path(__file__).resolve().parents[3] / "output"
    rest_table(directory)
    stirred_table(directory, arguments.window, repo_output)
    if arguments.figure:
        figure(directory, arguments.figure)


if __name__ == "__main__":
    main()
