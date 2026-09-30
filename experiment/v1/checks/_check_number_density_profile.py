"""_check_number_density_profile.py — does the particle number density of a case drift away from its
SPH density? (2026-09-30)

Runs a case and prints every `--report` steps, for horizontal slabs of the fluid,

  - the fluid count of the slab relative to its count at step 0 (per cent),
  - the mean SPH density of the slab relative to its mean at step 0 (per cent),

together with the smallest and largest fluid density, the largest fluid speed and the KCG fallback
count. In a consistent run the two percentages of a slab move together; in the 30 L tank at 1 g
the count drifted (floor 104 %, below the lid 85 % after 6 s at 3 mm) while the SPH density stayed
hydrostatic (log/2026-09-30_tank-2mm-divergence.md).

The slabs are `--slab` particle spacings thick and sit at the floor, at 1/4, 1/2 and 3/4 of the fluid
height, and at the lid (the two topmost slabs). The fluid height is taken from the initial positions.

Usage (repo root, solver env):
    python experiment/v1/checks/_check_number_density_profile.py CASE.yaml --max-steps N --report M [--slab 2]
"""
import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("case")
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--report", type=int, default=2000)
    parser.add_argument("--slab", type=float, default=2.0, help="slab thickness in particle spacings")
    parser.add_argument("--dump", default=None, help="write the final state to this .npz")
    arguments = parser.parse_args()

    from utils.sph.case import load_case
    from utils.sph.vulkan_context import VulkanContext
    from experiment.v1 import compile_shaders_v1
    from experiment.v1.utils.simulator_v1 import SphSimulatorV1

    compile_shaders_v1.compile_v1_shaders()
    case = load_case(arguments.case)
    dx = 2.0 * float(case.physics.particle_radius)
    thickness = arguments.slab * dx

    with VulkanContext.create(application_name="number_density_profile", enable_validation=False) as context:
        simulator = SphSimulatorV1(context, case)
        try:
            simulator.bootstrap()
            positions = simulator.readback_positions()
            live = simulator.live_slot_mask(positions)
            fluid = live & (simulator.readback_material() == 0)
            y = positions[fluid, 1].astype(np.float64)
            y_low, y_high = y.min() - 0.5 * dx, y.max() + 0.5 * dx
            height = y_high - y_low
            bands = [("floor", y_low, y_low + thickness),
                     ("1/4", y_low + 0.25 * height - 0.5 * thickness, y_low + 0.25 * height + 0.5 * thickness),
                     ("1/2", y_low + 0.5 * height - 0.5 * thickness, y_low + 0.5 * height + 0.5 * thickness),
                     ("3/4", y_low + 0.75 * height - 0.5 * thickness, y_low + 0.75 * height + 0.5 * thickness),
                     ("lid-1", y_high - 2.0 * thickness, y_high - thickness),
                     ("lid", y_high - thickness, y_high)]
            print(f"fluid {int(fluid.sum()):,} particles, y {y_low:.4f} .. {y_high:.4f} m, slabs {thickness * 1e3:.1f} mm: "
                  + ", ".join(f"{name} {y0 * 1e3:.0f}..{y1 * 1e3:.0f}" for name, y0, y1 in bands))
            reference_count = None
            reference_density = None
            print(f"{'step':>7s} {'t, s':>7s} | count % of step 0: " + " ".join(f"{name:>6s}" for name, _, _ in bands)
                  + " | density % of step 0: " + " ".join(f"{name:>6s}" for name, _, _ in bands)
                  + f" | {'rho min':>8s} {'rho max':>8s} {'|v| max':>7s} {'fallbk':>6s} | mean dy per step, 1e-3 dx: v dt / shift per slab | mean a_y - g_y per slab, m/s^2",
                  flush=True)

            def report():
                nonlocal reference_count, reference_density
                simulator.begin_readback_cache()
                positions = simulator.readback_positions()
                live = simulator.live_slot_mask(positions)
                fluid = live & (simulator.readback_material() == 0)
                y = positions[:, 1].astype(np.float64)
                density_pressure = simulator.readback_density_pressure()
                velocity = simulator.readback_velocity_mass()[:, :3].astype(np.float64)
                speed = np.linalg.norm(velocity[fluid], axis=1)
                shift = simulator.readback_shift()[:, :3].astype(np.float64)
                acceleration = simulator.readback_acceleration()[:, :3].astype(np.float64)
                gravity_y = float(case.physics.gravity[1])
                timestep = float(simulator.simulation_time / max(simulator.step_count, 1)) if simulator.step_count else 0.0
                status = simulator.readback_global_status()
                counts, densities, drifts = [], [], []
                for _, y0, y1 in bands:
                    inside = fluid & (y >= y0) & (y < y1)
                    counts.append(int(inside.sum()))
                    densities.append(float(density_pressure[inside, 0].mean()) if inside.any() else float("nan"))
                    # mean vertical displacement per step of the slab's particles, in 1e-3 dx: by the velocity and by the shift
                    drifts.append((1e3 * velocity[inside, 1].mean() * timestep / dx, 1e3 * shift[inside, 1].mean() / dx,
                                   acceleration[inside, 1].mean() - gravity_y)
                                  if inside.any() else (float("nan"), float("nan"), float("nan")))
                if reference_count is None:
                    reference_count, reference_density = counts, densities
                print(f"{simulator.step_count:7d} {simulator.simulation_time:7.3f} | "
                      + " ".join(f"{100.0 * c / r:6.1f}" for c, r in zip(counts, reference_count))
                      + " |                     " + " ".join(f"{100.0 * d / r:6.2f}" for d, r in zip(densities, reference_density))
                      + f" | {density_pressure[fluid, 0].min():8.2f} {density_pressure[fluid, 0].max():8.2f} "
                      f"{speed.max():7.3f} {status['correction_fallback_count']:6d} | "
                      + " ".join(f"{a:6.2f}/{b:5.2f}" for a, b, _ in drifts)
                      + " | " + " ".join(f"{c:7.4f}" for _, _, c in drifts), flush=True)
                simulator.end_readback_cache()
                return np.isfinite(density_pressure[fluid, 0]).all()

            report()
            while simulator.step_count < arguments.max_steps:
                simulator.step()
                if simulator.step_count % arguments.report == 0:
                    if not report():
                        print("non-finite densities, stopping")
                        break
            status = simulator.readback_global_status()
            print(f"end: step {simulator.step_count}, alive {status['alive_particle_count']:,}, overflow "
                  f"{status['overflow_inside_count']}/{status['overflow_incoming_count']}, KCG fallback "
                  f"{status['correction_fallback_count']}")
            if arguments.dump:
                np.savez_compressed(arguments.dump, positions=simulator.readback_positions(),
                                    material=simulator.readback_material(),
                                    density_pressure=simulator.readback_density_pressure(),
                                    velocity_mass=simulator.readback_velocity_mass(),
                                    acceleration=simulator.readback_acceleration(),
                                    shift=simulator.readback_shift(), uid=simulator.readback_particle_uid(),
                                    step=simulator.step_count, time=simulator.simulation_time,
                                    smoothing_length=float(case.physics.h), spacing=dx)
        finally:
            simulator.destroy()
    return 0


if __name__ == "__main__":
    sys.exit(main())
