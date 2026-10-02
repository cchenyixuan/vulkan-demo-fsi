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
import json
import pathlib
import sys
import time

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

from utils.sph.case import load_case
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
    return parser.parse_args()


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

    with VulkanContext.create(**create_kwargs) as ctx:
        sim = SphSimulatorV1(ctx, case)
        try:
            sim.bootstrap()
            start = time.perf_counter()
            sampling = (args.torque_every > 0 or args.probe_every > 0 or bool(args.scalar_snapshot_times)
                        or args.lifeline_dir is not None or args.flow_statistics is not None)
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

    steps_per_second = sim.step_count / elapsed if elapsed > 0 else float("nan")
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
