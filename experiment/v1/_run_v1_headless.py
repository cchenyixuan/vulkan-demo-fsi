"""
_run_v1_headless.py — run the single-GPU V1 simulator without a window.

Runs N steps, prints the alive particle count, the overflow counters and the
achieved steps per second, and can dump the final particle positions to an
.npz for comparison against another build. Useful as a quick benchmark and
as a regression check while developing boundary conditions / FSI.

Usage (run from repo root):
    python experiment/v1/_run_v1_headless.py [case] [--max-steps N] [--dump PATH]

Options:
    case                 case yaml path (default: cavity 1M)
    --device N           physical device index (default: auto-pick)
    --max-steps N        number of steps to run after bootstrap (default 1000)
    --validation         enable the Vulkan validation layer (slower)
    --dump PATH          save final positions + global status to PATH (.npz)
                         (plus scalars / particle uid when present)
    --torque-every N     rotor cases: torque every N steps (--torque-log, --torque-split-*)
    --probe-every N      scalar cases: probe values + conserved totals every N steps (--probe-log);
                         the CSV also holds the fluid mass, the mean fluid speed and, per field,
                         the mass-weighted coefficient of variation (cov:) and the mass fraction
                         within +-5 % of the mean (mixed5:)
    --scalar-snapshot-times T [T ...]
                         scalar cases: save FLUID positions, uid and fields at the first step
                         with t >= T (--scalar-snapshot-dir, --scalar-snapshot-fields)
    --lifeline-dir DIR   lifelines (2026-10-01): record a fixed random sample of FLUID particles
                         (persistent uid) every --lifeline-every steps from --lifeline-start on:
                         positions, optionally --lifeline-fields, and with --lifeline-aux the
                         velocity and shift every --lifeline-aux-every records
                         (experiment/v1/utils/lifeline_recorder.py)
    --flow-statistics CONFIG.json
                         time-averaged velocity statistics at fixed points (2026-10-02, Haringa 2023 H1):
                         every --flow-statistics-every steps from --flow-statistics-start on, into
                         --flow-statistics-out (experiment/v1/utils/flow_statistics.py)

Note: the solver is not bit-reproducible run to run (voxel incoming lists are
filled by atomics), so compare dumps against the run-to-run noise of an
unchanged build, and compare alive counts exactly.
"""

import argparse
import hashlib
import json
import pathlib
import sys
import time

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

from utils.sph.case import load_case, NINEPOOL_PARAMETER_ORDER
from utils.sph.vulkan_context import VulkanContext

from experiment.v1 import compile_shaders_v1
from experiment.v1.utils.simulator_v1 import SphSimulatorV1
from experiment.v1.utils.lifeline_recorder import LifelineRecorder
from experiment.v1.utils.flow_statistics import FlowStatisticsSampler


DEFAULT_CASE = "cases/lid_driven_cavity_2d/case.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Headless single-GPU V1 SPH run (bench / regression dump).")
    parser.add_argument("case", nargs="?", default=DEFAULT_CASE,
                        help=f"case yaml path (default: {DEFAULT_CASE})")
    parser.add_argument("--device", type=int, default=None,
                        help="physical device index (default: auto-pick)")
    parser.add_argument("--max-steps", type=int, default=1000,
                        help="steps to run after bootstrap (default 1000)")
    parser.add_argument("--validation", action="store_true",
                        help="enable Vulkan validation layer (slower)")
    parser.add_argument("--dump", type=str, default=None, metavar="PATH",
                        help="save final positions + global status to this .npz")
    parser.add_argument("--torque-every", type=int, default=0, metavar="N",
                        help="rotor cases: read back the hydrodynamic torque every N steps")
    parser.add_argument("--torque-log", type=str, default=None, metavar="PATH",
                        help="append torque samples as CSV (step,time,angle,torque_axis,fx,fy,fz"
                             "[,torque_lower,torque_upper,torque_shaft])")
    parser.add_argument("--torque-split-height", type=float, default=None, metavar="H",
                        help="per-impeller torque: rotor particles above/below H (m along the rotor "
                             "axis from the pivot) count as upper/lower impeller")
    parser.add_argument("--torque-shaft-radius", type=float, default=None, metavar="R",
                        help="per-impeller torque: rotor particles closer than R (m) to the axis "
                             "count as shaft (needs --torque-split-height)")
    parser.add_argument("--probe-every", type=int, default=0, metavar="N",
                        help="scalar cases: sample the `scalars.probes` points and the conserved "
                             "totals every N steps")
    parser.add_argument("--probe-log", type=str, default=None, metavar="PATH",
                        help="append the probe samples as CSV (step,time,<probe>:<field>...,"
                             "total:<field>...,fluid_mass,mean_speed,cov:<field>...,mixed5:<field>...)")
    parser.add_argument("--scalar-snapshot-times", type=float, nargs="+", default=None, metavar="T",
                        help="scalar cases: save the FLUID particles' positions, uid and fields at "
                             "the first step with simulation time >= T (s), one .npz per time")
    parser.add_argument("--scalar-snapshot-dir", type=str, default=None, metavar="DIR",
                        help="output directory of --scalar-snapshot-times (snapshot_<step>.npz)")
    parser.add_argument("--scalar-snapshot-fields", type=str, nargs="+", default=None, metavar="FIELD",
                        help="fields stored in the snapshots (default: all)")
    parser.add_argument("--lifeline-dir", type=str, default=None, metavar="DIR",
                        help="record lifelines of a sample of FLUID particles into DIR (2026-10-01)")
    parser.add_argument("--lifeline-count", type=int, default=10000, metavar="K",
                        help="number of sampled fluid particles (default 10000)")
    parser.add_argument("--lifeline-every", type=int, default=0, metavar="N",
                        help="record every N steps (default: the step count closest to 0.03 s)")
    parser.add_argument("--lifeline-start", type=float, default=0.0, metavar="T",
                        help="first record at simulation time >= T (s); the sample is drawn then")
    parser.add_argument("--lifeline-seed", type=int, default=1, help="seed of the random sample")
    parser.add_argument("--lifeline-chunk-seconds", type=float, default=10.0,
                        help="flow time per output file (default 10 s)")
    parser.add_argument("--lifeline-fields", type=str, nargs="+", default=None, metavar="FIELD",
                        help="scalar fields recorded along the lifelines (default: none)")
    parser.add_argument("--lifeline-aux", action="store_true",
                        help="also record velocity and particle shift every --lifeline-aux-every records")
    parser.add_argument("--lifeline-aux-every", type=int, default=10, metavar="M")
    parser.add_argument("--lifeline-release-sphere", type=float, nargs=4, default=None,
                        metavar=("X", "Y", "Z", "R"),
                        help="draw the sample only from fluid particles inside this sphere (m) at the start")
    parser.add_argument("--flow-statistics", type=str, default=None, metavar="CONFIG",
                        help="sample points (flow_statistics.json of _demo_rushton_tank.py) for time-averaged "
                             "velocity statistics (2026-10-02)")
    parser.add_argument("--flow-statistics-out", type=str, default=None, metavar="PATH",
                        help="output .npz of --flow-statistics (default: flow_statistics.npz next to the config)")
    parser.add_argument("--flow-statistics-every", type=int, default=0, metavar="N",
                        help="sample every N steps (default: the step count closest to 0.02 s)")
    parser.add_argument("--flow-statistics-start", type=float, default=0.0, metavar="T",
                        help="first sample at simulation time >= T (s)")
    parser.add_argument("--flow-statistics-flush", type=int, default=250, metavar="M",
                        help="write the accumulators (and a copy) every M samples")
    parser.add_argument("--population-mean-gly-every", type=float, default=0.0, metavar="T",
                        help="9-pool cases with `uptake_inhibition: mean` (2026-10-08): every T s of flow time write the "
                             "mass-weighted mean X_gly of the fluid particles into the reaction parameter glyMean11 "
                             "(Haringa 2018's population-average coupling)")
    # checkpoint / resume (2026-10-08)
    parser.add_argument("--checkpoint-dir", type=str, default=None, metavar="DIR",
                        help="write checkpoints (checkpoint_latest.npz, the previous one kept as "
                             "checkpoint_previous.npz) into DIR: every --checkpoint-wall-minutes of wall-clock "
                             "time and/or every --checkpoint-every s of flow time, and at the end of the run")
    parser.add_argument("--checkpoint-wall-minutes", type=float, default=0.0, metavar="M",
                        help="checkpoint every M minutes of wall-clock time (0 = off)")
    parser.add_argument("--checkpoint-every", type=float, default=0.0, metavar="T",
                        help="checkpoint every T s of flow time (0 = off)")
    parser.add_argument("--resume", type=str, default=None, metavar="PATH",
                        help="continue from a checkpoint instead of bootstrapping ('latest' = "
                             "checkpoint_latest.npz in --checkpoint-dir). Same case and the same sampling "
                             "options as the interrupted run; --max-steps is the total step count. Lifeline "
                             "chunks, probe and torque rows written after the checkpoint are discarded, "
                             "snapshots already taken are skipped. Not supported with --flow-statistics.")
    return parser.parse_args()


def truncate_csv_after(path, time_limit: float, time_column: int = 1) -> int:
    """Drop the rows of an append-mode CSV log whose time column is > time_limit
    (rows a crashed run wrote after the checkpoint). Returns the number removed."""
    csv_path = pathlib.Path(path)
    if not csv_path.exists():
        return 0
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    kept = [lines[0]] if lines else []
    removed = 0
    for line in lines[1:]:
        try:
            if float(line.split(",")[time_column]) > time_limit + 1e-9:
                removed += 1
                continue
        except (IndexError, ValueError):
            pass
        kept.append(line)
    if removed:
        csv_path.write_text("\n".join(kept) + "\n", encoding="utf-8")
    return removed


def mixing_metrics(snapshot: dict, mean_value: np.ndarray):
    """Mixing state of every field over the live FLUID particles (2026-09-27):
    the mass-weighted coefficient of variation about the conserved mean,
        CoV = sqrt( sum_i m_i (C_i / C_mean - 1)^2 / sum_i m_i ),
    and the mass fraction with |C_i / C_mean - 1| <= 0.05. Both are NaN for a
    field that holds no tracer yet (mean 0)."""
    weight = snapshot["mass"] / snapshot["mass"].sum()
    count = mean_value.shape[0]
    coefficient_of_variation = np.full(count, np.nan)
    mixed_fraction = np.full(count, np.nan)
    for field_index in range(count):
        if mean_value[field_index] > 0.0:
            deviation = snapshot["scalars"][:, field_index] / mean_value[field_index] - 1.0
            coefficient_of_variation[field_index] = float(np.sqrt(np.dot(weight, deviation * deviation)))
            mixed_fraction[field_index] = float(weight[np.abs(deviation) <= 0.05].sum())
    return coefficient_of_variation, mixed_fraction


def main() -> None:
    args = parse_args()

    compile_shaders_v1.compile_v1_shaders()

    case = load_case(args.case)
    expected_alive = sum(source.vertices.shape[0] for source in case.particle_sources)
    print(f"\n[v1-headless] loaded {args.case}")
    print(f"[v1-headless]   active particles: {expected_alive:,}")
    print(f"[v1-headless]   validation={'ON' if args.validation else 'OFF'}  "
          f"max_steps={args.max_steps}")

    create_kwargs = dict(
        application_name="sph_v1_headless",
        enable_validation=args.validation,
    )
    if args.device is not None:
        create_kwargs["device_index"] = args.device

    # checkpoint / resume (2026-10-08)
    case_sha1 = hashlib.sha1(pathlib.Path(args.case).read_bytes()).hexdigest()
    checkpoint_dir = pathlib.Path(args.checkpoint_dir) if args.checkpoint_dir else None
    resume_path = None
    if args.resume is not None:
        if args.resume == "latest":
            if checkpoint_dir is None:
                raise SystemExit("--resume latest needs --checkpoint-dir")
            resume_path = checkpoint_dir / "checkpoint_latest.npz"
        else:
            resume_path = pathlib.Path(args.resume)
        if not resume_path.exists():
            raise SystemExit(f"--resume: {resume_path} does not exist")
        if args.flow_statistics is not None:
            raise SystemExit("--resume is not supported together with --flow-statistics")
    if checkpoint_dir is not None and args.checkpoint_wall_minutes <= 0 and args.checkpoint_every <= 0:
        print("[v1-headless] --checkpoint-dir without --checkpoint-wall-minutes / --checkpoint-every: "
              "only the final checkpoint will be written")

    with VulkanContext.create(**create_kwargs) as ctx:
        sim = SphSimulatorV1(ctx, case)
        try:
            resumed = None
            if resume_path is not None:
                resumed = sim.restore_checkpoint(resume_path)
                if resumed.get("case_sha1") not in (None, case_sha1):
                    raise SystemExit(f"--resume: the checkpoint was written from a different case file "
                                     f"({resumed.get('case_path')}, sha1 {resumed.get('case_sha1')[:12]}); "
                                     f"this case has sha1 {case_sha1[:12]}")
                resume_time = sim.simulation_time
                print(f"[v1-headless] resumed at step {sim.step_count} t={resume_time:.6f} s from {resume_path}")
            else:
                sim.bootstrap()
            start = time.perf_counter()
            steps_at_start = sim.step_count
            sampling = (args.torque_every > 0 or args.probe_every > 0 or bool(args.scalar_snapshot_times)
                        or args.lifeline_dir is not None or args.flow_statistics is not None
                        or checkpoint_dir is not None or args.population_mean_gly_every > 0)

            # population-mean X_gly for the uptake-inhibition knob (2026-10-08)
            mean_gly = None
            reaction = case.scalars.reactions[0] if case.scalars is not None and case.scalars.reactions else None
            if reaction is not None and reaction.type == "ninepool" and reaction.uptake_inhibition == "mean":
                if args.population_mean_gly_every <= 0:
                    raise SystemExit("this case has `uptake_inhibition: mean`: pass --population-mean-gly-every T")
                mean_gly = {"every": max(1, round(args.population_mean_gly_every / case.timestep)),
                            "index": NINEPOOL_PARAMETER_ORDER.index("glyMean11"),
                            "field": case.scalars.field_names.index(reaction.gly), "count": 0}
            elif args.population_mean_gly_every > 0:
                raise SystemExit("--population-mean-gly-every needs a ninepool case with `uptake_inhibition: mean`")

            def update_mean_gly() -> None:
                snapshot = sim.scalar_snapshot()
                value = float(np.dot(snapshot["mass"], snapshot["scalars"][:, mean_gly["field"]]) / snapshot["mass"].sum())
                sim.update_reaction_parameter(mean_gly["index"], value)
                if mean_gly["count"] % 50 == 0:
                    print(f"[v1-headless] step={sim.step_count} t={sim.simulation_time:.4f}s population-mean X_gly {value:.4f}")
                mean_gly["count"] += 1

            if mean_gly is not None:
                update_mean_gly()
                print(f"[v1-headless] population-mean X_gly written every {mean_gly['every']} steps "
                      f"= {mean_gly['every'] * case.timestep:.4f} s")
            statistics = None
            if args.flow_statistics is not None:
                every = (args.flow_statistics_every if args.flow_statistics_every > 0
                         else max(1, round(0.02 / case.timestep)))
                out = args.flow_statistics_out or str(pathlib.Path(args.flow_statistics).with_name("flow_statistics.npz"))
                statistics = FlowStatisticsSampler(sim, args.flow_statistics, out, every=every,
                                                   start_time=args.flow_statistics_start,
                                                   flush_every=args.flow_statistics_flush)
            recorder = None
            if args.lifeline_dir is not None:
                every = args.lifeline_every if args.lifeline_every > 0 else max(1, round(0.03 / case.timestep))
                recorder = LifelineRecorder(
                    sim, args.lifeline_dir, count=args.lifeline_count, every=every,
                    start_time=args.lifeline_start, seed=args.lifeline_seed,
                    chunk_records=max(1, round(args.lifeline_chunk_seconds / (every * case.timestep))),
                    fields=args.lifeline_fields, aux=args.lifeline_aux, aux_every=args.lifeline_aux_every,
                    release_sphere=args.lifeline_release_sphere)
                if resumed is not None:
                    if resumed.get("lifeline") is None:
                        raise SystemExit("--resume: the checkpoint has no lifeline state but --lifeline-dir is given")
                    recorder.restore(resumed["lifeline"])
            elif resumed is not None and resumed.get("lifeline") is not None:
                raise SystemExit("--resume: the checkpoint has lifeline state; pass the same --lifeline-* options")
            if resumed is not None:
                for log_path in (args.torque_log, args.probe_log):
                    if log_path:
                        removed = truncate_csv_after(log_path, resume_time)
                        if removed:
                            print(f"[v1-headless] {log_path}: removed {removed} row(s) after t={resume_time:.6f} s")
            if args.torque_every > 0 and case.rotor is None:
                raise SystemExit("--torque-every needs a case with a rotor")
            if args.probe_every > 0 and (case.scalars is None or case.scalars.probes is None):
                raise SystemExit("--probe-every needs a case with a `scalars.probes` block")
            split = args.torque_split_height is not None
            if split and args.torque_shaft_radius is None:
                raise SystemExit("--torque-split-height needs --torque-shaft-radius")

            torque_log = None
            if args.torque_every > 0 and args.torque_log:
                torque_log = open(args.torque_log, "a")
                if torque_log.tell() == 0:
                    torque_log.write("step,time,angle,torque_axis,fx,fy,fz"
                                     + (",torque_lower,torque_upper,torque_shaft" if split else "") + "\n")
            probe_log = None
            if args.probe_every > 0:
                probe_names = [point.name for point in case.scalars.probes.points]
                probe_points = [point.position for point in case.scalars.probes.points]
                field_names = case.scalars.field_names
                if args.probe_log:
                    probe_log = open(args.probe_log, "a")
                    if probe_log.tell() == 0:
                        columns = ["step", "time"]
                        columns += [f"{p}:{f}" for p in probe_names for f in field_names]
                        columns += [f"total:{f}" for f in field_names]
                        columns += ["fluid_mass", "mean_speed"]
                        columns += [f"cov:{f}" for f in field_names]
                        columns += [f"mixed5:{f}" for f in field_names]
                        probe_log.write(",".join(columns) + "\n")

            # Field snapshots at given times (2026-09-27, mixing-time runs).
            snapshot_times = sorted(args.scalar_snapshot_times or [])
            if snapshot_times:
                if case.scalars is None:
                    raise SystemExit("--scalar-snapshot-times needs a case with a `scalars:` block")
                if not args.scalar_snapshot_dir:
                    raise SystemExit("--scalar-snapshot-times needs --scalar-snapshot-dir")
                snapshot_dir = pathlib.Path(args.scalar_snapshot_dir)
                snapshot_dir.mkdir(parents=True, exist_ok=True)
                all_field_names = case.scalars.field_names
                snapshot_field_names = list(args.scalar_snapshot_fields or all_field_names)
                unknown = [name for name in snapshot_field_names if name not in all_field_names]
                if unknown:
                    raise SystemExit(f"--scalar-snapshot-fields: unknown field(s) {unknown}")
                snapshot_field_index = [all_field_names.index(name) for name in snapshot_field_names]
            half_step = 0.5 * case.timestep
            if resumed is not None and snapshot_times:
                done = [t for t in snapshot_times if t - half_step <= resume_time]
                snapshot_times = [t for t in snapshot_times if t - half_step > resume_time]
                if done:
                    print(f"[v1-headless] resume: {len(done)} snapshot time(s) already taken, skipped")

            # checkpoint bookkeeping (2026-10-08)
            last_checkpoint_wall = time.perf_counter()
            next_checkpoint_time = (sim.simulation_time + args.checkpoint_every) if args.checkpoint_every > 0 else None

            def write_checkpoint(reason: str) -> None:
                nonlocal last_checkpoint_wall, next_checkpoint_time
                for handle in (torque_log, probe_log):
                    if handle is not None:
                        handle.flush()
                extra = {"case_path": str(args.case), "case_sha1": case_sha1,
                         "lifeline": recorder.checkpoint_state() if recorder is not None else None}
                latest = checkpoint_dir / "checkpoint_latest.npz"
                previous = checkpoint_dir / "checkpoint_previous.npz"
                began = time.perf_counter()
                written = sim.write_checkpoint(checkpoint_dir / "checkpoint_new.npz", extra=extra)
                if latest.exists():
                    latest.replace(previous)
                written.replace(latest)
                size_mb = latest.stat().st_size / (1024 * 1024)
                print(f"[v1-headless] checkpoint ({reason}): step={sim.step_count} t={sim.simulation_time:.4f} s "
                      f"{size_mb:.0f} MB in {time.perf_counter() - began:.1f} s -> {latest}", flush=True)
                last_checkpoint_wall = time.perf_counter()
                if next_checkpoint_time is not None:
                    while next_checkpoint_time - half_step <= sim.simulation_time:
                        next_checkpoint_time += args.checkpoint_every

            split_reported = False
            while sampling and sim.step_count < args.max_steps:
                sim.step()
                if args.torque_every > 0 and sim.step_count % args.torque_every == 0:
                    torque = sim.readback_rotor_torque(args.torque_split_height,
                                                       args.torque_shaft_radius)
                    if split and not split_reported:
                        print(f"[v1-headless] torque split: lower={torque['rotor_particle_count_lower']:,} "
                              f"upper={torque['rotor_particle_count_upper']:,} "
                              f"shaft={torque['rotor_particle_count_shaft']:,} rotor particles")
                        split_reported = True
                    fx, fy, fz = torque["force"]
                    print(f"[v1-headless] step={torque['step']} t={torque['time']:.4f}s "
                          f"angle={torque['rotor_angle']:.3f}rad "
                          f"torque_axis={torque['torque_axis']:.6e} N·m "
                          f"force=({fx:.3e}, {fy:.3e}, {fz:.3e}) N")
                    if torque_log is not None:
                        line = (f"{torque['step']},{torque['time']:.6e},{torque['rotor_angle']:.6e},"
                                f"{torque['torque_axis']:.6e},{fx:.6e},{fy:.6e},{fz:.6e}")
                        if split:
                            line += (f",{torque['torque_axis_lower']:.6e},{torque['torque_axis_upper']:.6e},"
                                     f"{torque['torque_axis_shaft']:.6e}")
                        torque_log.write(line + "\n")
                        torque_log.flush()
                if args.probe_every > 0 and sim.step_count % args.probe_every == 0:
                    snapshot = sim.scalar_snapshot()
                    values = sim.probe_scalars(probe_points, snapshot=snapshot)
                    totals = sim.scalar_totals(snapshot)
                    print(f"[v1-headless] step={sim.step_count} t={sim.simulation_time:.4f}s probes="
                          + " ".join(f"{name}:{values[p, 0]:.4e}" for p, name in enumerate(probe_names))
                          + f" total:{field_names[0]}={totals[0]:.6e}")
                    if probe_log is not None:
                        row = [f"{sim.step_count}", f"{sim.simulation_time:.6e}"]
                        row += [f"{values[p, k]:.6e}" for p in range(len(probe_names))
                                for k in range(len(field_names))]
                        fluid_mass = float(snapshot["mass"].sum())
                        mean_speed = float(np.linalg.norm(snapshot["velocity"], axis=1).mean())
                        coefficient_of_variation, mixed_fraction = mixing_metrics(snapshot, totals / fluid_mass)
                        row += [f"{total:.9e}" for total in totals]
                        row += [f"{fluid_mass:.9e}", f"{mean_speed:.6e}"]
                        row += [f"{value:.6e}" for value in coefficient_of_variation]
                        row += [f"{value:.6e}" for value in mixed_fraction]
                        probe_log.write(",".join(row) + "\n")
                        probe_log.flush()
                if snapshot_times and sim.simulation_time >= snapshot_times[0] - half_step:
                    field_snapshot = sim.scalar_snapshot()
                    snapshot_path = snapshot_dir / f"snapshot_{sim.step_count:08d}.npz"
                    np.savez(snapshot_path, time=sim.simulation_time, step=sim.step_count,
                             positions=field_snapshot["positions"].astype(np.float32),
                             mass=field_snapshot["mass"].astype(np.float32),
                             uid=field_snapshot["uid"],
                             scalars=field_snapshot["scalars"][:, snapshot_field_index].astype(np.float32),
                             field_names=np.array(snapshot_field_names))
                    print(f"[v1-headless] step={sim.step_count} t={sim.simulation_time:.4f}s "
                          f"scalar snapshot -> {snapshot_path}")
                    while snapshot_times and sim.simulation_time >= snapshot_times[0] - half_step:
                        snapshot_times.pop(0)
                if recorder is not None and recorder.due():
                    recorder.record()
                if statistics is not None and statistics.due():
                    statistics.sample()
                if mean_gly is not None and sim.step_count % mean_gly["every"] == 0:
                    update_mean_gly()
                if checkpoint_dir is not None and sim.step_count < args.max_steps:
                    if args.checkpoint_wall_minutes > 0 and \
                            time.perf_counter() - last_checkpoint_wall >= 60.0 * args.checkpoint_wall_minutes:
                        write_checkpoint("wall clock")
                    elif next_checkpoint_time is not None and sim.simulation_time >= next_checkpoint_time - half_step:
                        write_checkpoint("flow time")
            if checkpoint_dir is not None:
                write_checkpoint("end of run")
            if recorder is not None:
                recorder.close()
            if statistics is not None:
                statistics.close()
            for handle in (torque_log, probe_log):
                if handle is not None:
                    handle.close()
            if not sampling:
                sim.run_until(max_steps=args.max_steps)
            elapsed = time.perf_counter() - start
            status = sim.readback_global_status()
            positions = sim.readback_positions() if args.dump else None
            material = sim.readback_material() if args.dump else None
            velocity_mass = sim.readback_velocity_mass() if args.dump else None
            scalars = sim.readback_scalars() if args.dump and case.scalar_vec4_count > 0 else None
            particle_uid = sim.readback_particle_uid() if args.dump else None
            turbulent_viscosity = (sim.readback_turbulent_viscosity()
                                   if args.dump and case.scalars is not None and case.scalars.sgs.enabled
                                   else None)
        finally:
            sim.destroy()

    steps_per_second = (sim.step_count - steps_at_start) / elapsed if elapsed > 0 else float("nan")
    print(f"[v1-headless] final: step={sim.step_count} "
          f"alive={status['alive_particle_count']:,} (expected {expected_alive:,})  "
          f"{steps_per_second:.1f} steps/s")
    print(f"[v1-headless]   overflow_inside={status['overflow_inside_count']} "
          f"overflow_incoming={status['overflow_incoming_count']} "
          f"correction_fallback={status['correction_fallback_count']} "
          f"overflow_neighbor={status['overflow_neighbor_count']}")
    if status["overflow_neighbor_count"] != 0:
        print(f"[v1-headless] WARNING: {status['overflow_neighbor_count']} particles exceeded "
              f"capacities.max_neighbors (first pid {status['first_overflow_neighbor_pid']}); "
              f"their extra neighbours were dropped", file=sys.stderr)
    if status["alive_particle_count"] != expected_alive:
        print("[v1-headless] WARNING: alive count differs from the loaded particle count",
              file=sys.stderr)

    if args.dump:
        extra = {}
        if scalars is not None:
            extra["scalars"] = scalars
            extra["scalar_names"] = np.array(case.scalars.field_names)
        if turbulent_viscosity is not None:
            extra["turbulent_viscosity"] = turbulent_viscosity
        np.savez(args.dump, positions=positions, material=material, velocity_mass=velocity_mass,
                 particle_uid=particle_uid, status=json.dumps(status), **extra)
        print(f"[v1-headless] dumped positions + status to {args.dump}")


if __name__ == "__main__":
    main()
