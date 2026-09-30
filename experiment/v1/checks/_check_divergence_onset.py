"""_check_divergence_onset.py — where does a run start to diverge? (2026-09-30)

The 2 mm tank runs of 2026-09-30 diverged after 3 to 4 s: the smallest density fell from 997 to below
900 kg/m^3 within 3000 steps, the largest rose, then the particles left the domain. The logs of the
blade run hold only the extremes of the whole tank. This script runs a case and

  - every `--check-every` steps prints the smallest and the largest density of the live particles with
    the position, the distance from the rotor axis, the kind and the speed of the two particles,
  - from the first check at which a density leaves [`--density-low`, `--density-high`] it checks every
    `--dump-every` steps and writes the particles within `--radius` kernel radii of the two extreme
    particles to DIR/onset_NNN.npz (position, velocity, density, pressure, material, uid, acceleration,
    shift), `--dumps` times, and stops.

The extremes are those of the fluid particles unless `--all-kinds` is given. The limits must lie outside
the range of a healthy run. 2 mm tank, 1 g, c0 = 20.57 m/s: fluid 997.3 to about 1012, floor particles
below the rotor bell up to 1020 during the start (run of 2026-09-30 that triggered at step 15,000
because of them).

Usage (repo root, solver env):
    python experiment/v1/checks/_check_divergence_onset.py CASE.yaml --max-steps 100000 \
        --check-every 500 --check-start 50000 --density-low 993 --density-high 1030 --out DIR
"""
import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

KIND_NAMES = {0: "fluid", 1: "boundary", 2: "inlet", 3: "rotor"}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("case")
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--check-every", type=int, default=500)
    parser.add_argument("--check-start", type=int, default=0, help="first step at which the densities are checked")
    parser.add_argument("--density-low", type=float, default=990.0)
    parser.add_argument("--density-high", type=float, default=1030.0)
    parser.add_argument("--all-kinds", action="store_true",
                        help="extremes over all particles; default: fluid particles only (the density of a solid "
                             "particle follows from its extrapolated pressure, 1020 kg/m^3 on the floor below the "
                             "rotor bell of the tank is the healthy state)")
    parser.add_argument("--dump-every", type=int, default=250)
    parser.add_argument("--dumps", type=int, default=8)
    parser.add_argument("--radius", type=float, default=6.0, help="radius of the dumped balls in kernel radii")
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()

    from utils.sph.case import load_case
    from utils.sph.vulkan_context import VulkanContext
    from experiment.v1 import compile_shaders_v1
    from experiment.v1.utils.simulator_v1 import SphSimulatorV1

    compile_shaders_v1.compile_v1_shaders()
    case = load_case(arguments.case)
    out = pathlib.Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    kind_of_group = np.asarray([int(m.kind) for m in case.materials])
    axis = np.asarray(case.rotor.axis, dtype=np.float64) if case.rotor is not None else np.array([0.0, 1.0, 0.0])
    axis = axis / np.linalg.norm(axis)
    pivot = np.asarray(case.rotor.pivot, dtype=np.float64) if case.rotor is not None else np.zeros(3)
    ball = arguments.radius * float(case.physics.h)

    with VulkanContext.create(application_name="divergence_onset", enable_validation=False) as context:
        simulator = SphSimulatorV1(context, case)
        try:
            simulator.bootstrap()
            triggered = False
            dump_index = 0
            next_check = max(arguments.check_start, arguments.check_every)
            print("   step     time    alive  fallback   density min  at x, y, z (mm)            r (mm)  kind      speed   pressure "
                  "| density max  at x, y, z (mm)            r (mm)  kind      speed   pressure")
            while simulator.step_count < arguments.max_steps:
                simulator.step()
                step = simulator.step_count
                if step < next_check:
                    continue
                next_check = step + (arguments.dump_every if triggered else arguments.check_every)
                simulator.begin_readback_cache()
                positions = simulator.readback_positions()
                live = simulator.live_slot_mask(positions)
                density_pressure = simulator.readback_density_pressure()
                material = simulator.readback_material()
                velocity_mass = simulator.readback_velocity_mass()
                status = simulator.readback_global_status()
                # plate particles of the thin plate treatment carry a marker in the pressure slot
                ordinary = live & (density_pressure[:, 1] > -1.0e8)
                if not arguments.all_kinds:
                    ordinary &= kind_of_group[material] == 0
                density = np.where(ordinary, density_pressure[:, 0], np.nan)
                if not np.isfinite(density[ordinary]).all():
                    print(f"   {step:6d} {simulator.simulation_time:8.4f}  non-finite densities: "
                          f"{int((~np.isfinite(density[ordinary])).sum())} particles, alive {status['alive_particle_count']}")
                    break
                low, high = int(np.nanargmin(density)), int(np.nanargmax(density))

                def describe(index):
                    x = positions[index, :3].astype(np.float64)
                    arm = x - pivot
                    radial = np.linalg.norm(arm - (arm @ axis) * axis)
                    speed = float(np.linalg.norm(velocity_mass[index, :3]))
                    kind = KIND_NAMES.get(int(kind_of_group[material[index]]), "?")
                    return (f"{density[index]:9.2f}   {x[0] * 1e3:8.2f} {x[1] * 1e3:8.2f} {x[2] * 1e3:8.2f}   {radial * 1e3:7.2f}  "
                            f"{kind:8s} {speed:7.3f}  {density_pressure[index, 1]:9.1f}")
                print(f"   {step:6d} {simulator.simulation_time:8.4f}  {status['alive_particle_count']:7d}  "
                      f"{status['correction_fallback_count']:8d}   {describe(low)} |  {describe(high)}", flush=True)
                if not triggered and (density[low] < arguments.density_low or density[high] > arguments.density_high):
                    triggered = True
                    next_check = step + arguments.dump_every
                    print(f"   density outside [{arguments.density_low}, {arguments.density_high}]: "
                          f"dumps every {arguments.dump_every} steps from here", flush=True)
                if triggered:
                    centres = positions[[low, high], :3].astype(np.float64)
                    selected = np.zeros(positions.shape[0], dtype=bool)
                    for centre in centres:
                        selected |= live & (np.linalg.norm(positions[:, :3] - centre, axis=1) < ball)
                    np.savez_compressed(
                        out / f"onset_{dump_index:03d}.npz",
                        step=step, time=simulator.simulation_time, rotor_angle=simulator.rotor_angle,
                        position=positions[selected, :3], velocity=velocity_mass[selected, :3],
                        density=density_pressure[selected, 0], pressure=density_pressure[selected, 1],
                        material=material[selected], kind=kind_of_group[material[selected]],
                        uid=simulator.readback_particle_uid()[selected],
                        acceleration=simulator.readback_acceleration()[selected, :3],
                        shift=simulator.readback_shift()[selected, :3],
                        centres=centres, spacing=2.0 * float(case.physics.particle_radius),
                        smoothing_length=float(case.physics.h))
                    print(f"      dump {dump_index}: {int(selected.sum()):,} particles -> onset_{dump_index:03d}.npz", flush=True)
                    dump_index += 1
                    if dump_index >= arguments.dumps:
                        break
                simulator.end_readback_cache()
            status = simulator.readback_global_status()
            print(f"   end: step {simulator.step_count}, alive {status['alive_particle_count']}, overflow "
                  f"{status['overflow_inside_count']}/{status['overflow_incoming_count']}, "
                  f"KCG fallback {status['correction_fallback_count']}")
        finally:
            simulator.destroy()
    return 0


if __name__ == "__main__":
    sys.exit(main())
