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
                        help="append torque samples as CSV (step,time,angle,torque_axis,fx,fy,fz)")
    return parser.parse_args()


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
            if args.torque_every > 0:
                if case.rotor is None:
                    raise SystemExit("--torque-every needs a case with a rotor")
                log_handle = open(args.torque_log, "a") if args.torque_log else None
                if log_handle is not None and log_handle.tell() == 0:
                    log_handle.write("step,time,angle,torque_axis,fx,fy,fz\n")
                while sim.step_count < args.max_steps:
                    sim.step()
                    if sim.step_count % args.torque_every == 0:
                        torque = sim.readback_rotor_torque()
                        fx, fy, fz = torque["force"]
                        print(f"[v1-headless] step={torque['step']} t={torque['time']:.4f}s "
                              f"angle={torque['rotor_angle']:.3f}rad "
                              f"torque_axis={torque['torque_axis']:.6e} N·m "
                              f"force=({fx:.3e}, {fy:.3e}, {fz:.3e}) N")
                        if log_handle is not None:
                            log_handle.write(f"{torque['step']},{torque['time']:.6e},{torque['rotor_angle']:.6e},"
                                             f"{torque['torque_axis']:.6e},{fx:.6e},{fy:.6e},{fz:.6e}\n")
                            log_handle.flush()
                if log_handle is not None:
                    log_handle.close()
            else:
                sim.run_until(max_steps=args.max_steps)
            elapsed = time.perf_counter() - start
            status = sim.readback_global_status()
            positions = sim.readback_positions() if args.dump else None
            material = sim.readback_material() if args.dump else None
            velocity_mass = sim.readback_velocity_mass() if args.dump else None
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
        np.savez(args.dump, positions=positions, material=material, velocity_mass=velocity_mass,
                 status=json.dumps(status))
        print(f"[v1-headless] dumped positions + status to {args.dump}")


if __name__ == "__main__":
    main()
