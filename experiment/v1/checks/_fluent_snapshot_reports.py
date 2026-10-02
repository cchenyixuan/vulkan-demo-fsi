"""Reports from the Fluent LES snapshots of the 30 L tank (Rautenbach et al. 2026, DARUS-5523), 2026-10-03.

The dataset has one case + data file per second, 25..34 s, for a fine and a coarse mesh. This script writes
Fluent TUI journals that read every snapshot and print
  torques: the moment about +y of every wall zone (pressure, viscous, total)
  zones:   per cell zone (rt_rotorbox, pbt_rotorbox, bulk_liquid) the volume, the volume-averaged |u|,
           the kinetic energy and two dissipation integrals:
           eps    = integral of mu_eff |S|^2  (Fluent's strain-rate-mag = sqrt(2 S:S), so this is 2 mu_eff S:S)
           eps_lam = integral of mu    |S|^2  (molecular part)
and parses the transcripts.

    python _fluent_snapshot_reports.py journal torques torques.jou      # then, in an empty directory:
    "<ANSYS>/v251/fluent/ntbin/win64/fluent.exe" 3ddp -g -t8 -wait -i torques.jou
    python _fluent_snapshot_reports.py parse torques torques.trn

Notes (2026-10-02/03): start every journal with /file/set-batch-options no yes yes no (exit on error),
otherwise an error leaves Fluent waiting for input. The fine-mesh snapshots 26..34 s are a mirrored
setup (every moment flips sign): magnitudes are compared. Fluent y = generator y + 58.5 mm.
Results: log/2026-10-03_fluent-and-wall-torque.md."""
import pathlib
import re
import sys

import numpy as np

DATA = pathlib.Path(r"D:/CFD/SPH dev/repository/02_simulation_results/05_fluent_seed_simulations")
SNAPSHOTS = {"fine": DATA / "fine_mesh_start_files", "coarse": DATA / "coarse_mesh_startfiles" / "start_files"}
WALL_ZONES = ("rt", "pbt", "stator", "walls", "baffles", "top_wall", "probe_1", "probe_2")
CELL_ZONES = ("rt_rotorbox", "pbt_rotorbox", "bulk_liquid")
OMEGA = 2.0 * np.pi * 200.0 / 60.0


def case_files():
    for label, folder in SNAPSHOTS.items():
        for second in range(25, 35):
            cases = sorted((folder / f"{second}s").glob("*.cas"))
            if cases:
                yield label, second, cases[0]


def journal(kind, out):
    out = pathlib.Path(out)
    lines = ["/file/set-batch-options no yes yes no",
             f'/file/start-transcript "{out.with_suffix(".trn").resolve().as_posix()}"']
    for label, second, case in case_files():
        lines.append(f'/file/read-case-data "{case.as_posix()}"')
        lines.append(f'(display "MARK {label} {second}")')
        if kind == "torques":
            lines += [f"/report/forces/wall-moments no {zone} () 0 0 0 0 1 0 no" for zone in WALL_ZONES]
        else:
            zones = " ".join(CELL_ZONES)
            lines += ['/define/custom-field-functions/define "kedens" "0.5*density*velocity_magnitude^2"',
                      '/define/custom-field-functions/define "epseff" "sgs_viscosity_eff*strain_rate_mag^2"',
                      '/define/custom-field-functions/define "epslam" "viscosity_lam*strain_rate_mag^2"',
                      f"/report/volume-integrals/volume {zones} () no",
                      f"/report/volume-integrals/volume-avg {zones} () velocity-magnitude no",
                      f"/report/volume-integrals/volume-integral {zones} () kedens no",
                      f"/report/volume-integrals/volume-integral {zones} () epseff no",
                      f"/report/volume-integrals/volume-integral {zones} () epslam no"]
    lines += ["/file/stop-transcript", "/exit yes"]
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out} ({len(lines)} lines)")


def blocks(text):
    # transcripts of 2026-10-02 used the markers MARKER (torques) and ZMARK (zones)
    parts = re.split(r"(?:MARKER|ZMARK|MARK) (\w+) (\d+)", text)
    for index in range(1, len(parts), 3):
        yield parts[index], int(parts[index + 1]), parts[index + 2]


def parse_torques(text):
    rows = {}
    for label, second, body in blocks(text):
        torques = {}
        for match in re.finditer(r"Moment Axis \(0 1 0\)\s*\n.*?\nZone.*?\n(\S+)\s+(\S+)\s+(\S+)\s+(\S+)", body):
            torques[match.group(1)] = tuple(map(float, match.group(2, 3, 4)))
        rows[(label, second)] = torques
    for label in SNAPSHOTS:
        seconds = sorted(s for (l, s) in rows if l == label)
        if not seconds:
            continue
        print(f"\n{label} mesh: total moment about +y, mN m; every snapshot oriented so that the impeller load is"
              " negative (the 26..34 s fine snapshots are mirrored)")
        print("   t   " + "".join(f"{zone:>10s}" for zone in WALL_ZONES) + "    rt+pbt  static")
        table = []
        for second in seconds:
            torques = rows[(label, second)]
            orientation = -np.sign(torques["rt"][2])
            values = [orientation * torques.get(zone, (np.nan,) * 3)[2] * 1e3 for zone in WALL_ZONES]
            rotor, static = values[0] + values[1], np.nansum(values[2:])
            table.append(values + [rotor, static])
            print(f"  {second:2d} s " + "".join(f"{value:10.2f}" for value in values) + f"  {rotor:8.2f} {static:7.2f}")
        table = np.asarray(table)
        print("  mean " + "".join(f"{value:10.2f}" for value in np.nanmean(table, axis=0)[:-2])
              + f"  {np.nanmean(table[:, -2]):8.2f} {np.nanmean(table[:, -1]):7.2f}")
        print("  std  " + "".join(f"{value:10.2f}" for value in np.nanstd(table, axis=0, ddof=1)[:-2]))
        pressure = np.mean([abs(rows[(label, s)]["walls"][0]) for s in seconds]) * 1e3
        viscous = np.mean([abs(rows[(label, s)]["walls"][1]) for s in seconds]) * 1e3
        print(f"  tank walls: pressure part {pressure:.2f}, viscous part {viscous:.2f} mN m (magnitudes)")


def parse_zones(text):
    quantities = ("volume", "speed", "ke", "eps", "eps_lam")
    results = {}
    for label, second, body in blocks(text):
        reports = re.split(r"\n> /report/volume-integrals/", body)[1:]
        values = {}
        for quantity, report in zip(quantities, reports):
            for zone in CELL_ZONES + ("Net",):
                match = re.search(rf"^\s*{zone}\s+([-0-9.eE+]+)\s*$", report, re.M)
                values[(quantity, zone)] = float(match.group(1)) if match else np.nan
        results[(label, second)] = values
    for label in SNAPSHOTS:
        seconds = sorted(s for (l, s) in results if l == label)
        if not seconds:
            continue
        print(f"\n{label} mesh, mean over {len(seconds)} snapshots ({seconds[0]}..{seconds[-1]} s)")
        print("  zone           volume L   mean |u| m/s   KE J      eps W   eps molecular W")
        for zone in CELL_ZONES + ("Net",):
            row = [np.nanmean([results[(label, s)][(q, zone)] for s in seconds]) for q in quantities]
            print(f"  {zone:13s}  {row[0] * 1e3:8.3f}   {row[1]:8.4f}     {row[2]:8.4f}  {row[3]:8.4f}  {row[4]:8.4f}")


if __name__ == "__main__":
    command, kind, path = sys.argv[1:4]
    if command == "journal":
        journal(kind, path)
    else:
        text = pathlib.Path(path).read_text(encoding="utf-8", errors="replace")
        (parse_torques if kind == "torques" else parse_zones)(text)
