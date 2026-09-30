"""_check_hydrostatic_drift.py — does a column of fluid at rest keep its particle number density? (2026-09-30)

In the 30 L tank runs at 1 g the SPH density stayed hydrostatic (1007 at the floor, 998 below the lid)
while the number of particles per lattice volume drifted: after 6 s at 3 mm the floor slabs held 104 %
of the lattice count, the slabs below the lid 85 %; without gravity every slab stayed at 95 to 105 %.
At 2 mm the layer below the lid emptied, the pressure there fell below -2 kPa and the run diverged
after 3 s (log/2026-09-30_tank-2mm-divergence.md). This check runs a closed box of fluid at rest under
gravity (hydrostatic initial density, zero pressure at the lid, the tank's settings) and prints every
`--report` steps

  - the fluid count of the slabs at the floor, mid-height and below the lid as a percentage of the
    lattice count,
  - the smallest and the largest fluid density, the median pressure of the top and the bottom slab,
  - the largest fluid speed and the mean of the particle shift of the step.

Variants: --no-diffusion, --gradient-term, --no-pst, --background PB, --solid-pressure MODE,
--pair-correction MODE, --delta D, --c0 C.

Usage (repo root, solver env):
    python experiment/v1/checks/_check_hydrostatic_drift.py --dx 0.004 --steps 40000 --report 4000 --out DIR
"""
import argparse
import math
import pathlib
import sys

import numpy as np
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GRAVITY = 9.81
REST_DENSITY = 1000.0


def write_points(path, points):
    np.savetxt(path, points, fmt="v %.7f %.7f %.7f", header=f"# {points.shape[0]} particles", comments="")


def build_case(directory, arguments):
    directory.mkdir(parents=True, exist_ok=True)
    dx, hdx = arguments.dx, arguments.hdx
    width_cells, height_cells = arguments.width_cells, arguments.height_cells
    half_width = 0.5 * width_cells * dx
    height = height_cells * dx
    horizontal = (np.arange(width_cells) + 0.5) * dx - half_width
    vertical = (np.arange(height_cells) + 0.5) * dx
    grid = np.stack(np.meshgrid(horizontal, vertical, horizontal, indexing="ij"), axis=-1).reshape(-1, 3)
    layers = int(math.ceil(hdx))
    extended_h = (np.arange(-layers, width_cells + layers) + 0.5) * dx - half_width
    extended_v = (np.arange(-layers, height_cells + layers) + 0.5) * dx
    shell = np.stack(np.meshgrid(extended_h, extended_v, extended_h, indexing="ij"), axis=-1).reshape(-1, 3)
    outside = (np.abs(shell[:, 0]) > half_width) | (np.abs(shell[:, 2]) > half_width) | (shell[:, 1] < 0.0) | (shell[:, 1] > height)
    wall = shell[outside]
    write_points(directory / "fluid.obj", grid)
    write_points(directory / "wall.obj", wall)
    points = np.vstack([grid, wall])
    lo, hi = points.min(axis=0) - 0.6 * dx, points.max(axis=0) + 0.6 * dx
    with open(directory / "frame.obj", "w") as handle:
        for x in (lo[0], hi[0]):
            for y in (lo[1], hi[1]):
                for z in (lo[2], hi[2]):
                    handle.write(f"v {x:.7f} {y:.7f} {z:.7f}\n")
    pool = int(math.ceil(points.shape[0] * 1.15 / 128) * 128)
    bound = int(math.ceil(math.sqrt(2) * hdx ** 3))
    physics = {"dimension": 3, "h": hdx * dx, "particle_radius": 0.5 * dx, "lattice": "grid",
               "calibrate_volume": True, "speed_of_sound": arguments.c0, "power": 7,
               "cfl": 0.15, "hydrostatic_reference": [0.0, float(height), 0.0], "gravity": [0.0, -GRAVITY, 0.0]}
    if arguments.background != 0.0:
        physics["background_pressure"] = arguments.background
    numerics = {"use_density_diffusion": not arguments.no_diffusion, "delta_coefficient": arguments.delta,
                "use_kcg_correction": True,
                "regularization": {"xi": 0.01, "det_threshold": 1.0e-4, "frobenius_max": 10.0},
                "solid_pressure": arguments.solid_pressure, "solid_reaction_force": True,
                "pair_correction": arguments.pair_correction,
                "density_diffusion_gradient_term": arguments.gradient_term,
                "use_pst": not arguments.no_pst, "pst_main": 0.1, "pst_anti": 0.0005,
                "defrag_enabled": True, "defrag_cadence": 10, "use_prefix_sum_defrag": False}
    case = {"schema_version": 2,
            "time": {"total": None, "max_steps": None, "output_cadence": None},
            "physics": physics, "numerics": numerics,
            "capacities": {"pool_size": pool, "max_per_voxel": max(64, int(2 ** math.ceil(math.log2(bound * 1.3)))),
                           "max_incoming": 32, "workgroup": 128},
            "material_library": "materials.yaml",
            "geometry": {"frame": "frame.obj",
                         "particles": [{"file": "fluid.obj", "material": "box_fluid"},
                                       {"file": "wall.obj", "material": "box_wall"}]}}
    (directory / "case.yaml").write_text(yaml.safe_dump(case, sort_keys=False), encoding="utf-8")
    materials = {"schema_version": 1,
                 "box_fluid": {"kind": "fluid", "rest_density": REST_DENSITY, "viscosity": 1.0e-6},
                 "box_wall": {"kind": "boundary", "rest_density": REST_DENSITY, "viscosity": 1.0e-6}}
    (directory / "materials.yaml").write_text(yaml.safe_dump(materials, sort_keys=False), encoding="utf-8")
    return directory / "case.yaml", grid.shape[0], height


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--dx", type=float, default=0.004)
    parser.add_argument("--hdx", type=float, default=3.0)
    parser.add_argument("--width-cells", type=int, default=20)
    parser.add_argument("--height-cells", type=int, default=60)
    parser.add_argument("--c0", type=float, default=15.4,
                        help="speed of sound; 15.4 m/s with a 0.24 m column gives rho g H / (rho c0^2) = 0.99 %%, "
                             "the tank's value (0.4265 m, 20.57 m/s)")
    parser.add_argument("--steps", type=int, default=40000)
    parser.add_argument("--report", type=int, default=4000)
    parser.add_argument("--out", default="output/hydrostatic_drift/base")
    parser.add_argument("--no-diffusion", action="store_true")
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--gradient-term", action="store_true")
    parser.add_argument("--no-pst", action="store_true")
    parser.add_argument("--background", type=float, default=0.0, help="physics.background_pressure, Pa")
    parser.add_argument("--solid-pressure", default="accumulate")
    parser.add_argument("--pair-correction", default="reverse")
    parser.add_argument("--dump", action="store_true", help="write DIR/final.npz")
    arguments = parser.parse_args()

    from utils.sph.case import load_case
    from utils.sph.vulkan_context import VulkanContext
    from experiment.v1 import compile_shaders_v1
    from experiment.v1.utils.simulator_v1 import SphSimulatorV1

    compile_shaders_v1.compile_v1_shaders()
    out = pathlib.Path(arguments.out)
    case_path, fluid_count, height = build_case(out, arguments)
    case = load_case(str(case_path))
    dx = arguments.dx
    slab = 2.0 * dx
    lattice_per_slab = arguments.width_cells ** 2 * 2
    bands = [("floor", 0.0, slab), ("floor+", slab, 2 * slab), ("quarter", 0.25 * height - slab, 0.25 * height + slab),
             ("middle", 0.5 * height - slab, 0.5 * height + slab), ("3/4", 0.75 * height - slab, 0.75 * height + slab),
             ("lid-", height - 2 * slab, height - slab), ("lid", height - slab, height)]
    print(f"column {arguments.width_cells}x{arguments.height_cells}x{arguments.width_cells} cells, dx {dx * 1e3:g} mm, "
          f"h/dx {arguments.hdx:g}, H {height:.3f} m, c0 {arguments.c0:g} m/s, rho g H {REST_DENSITY * GRAVITY * height:.0f} Pa "
          f"= {REST_DENSITY * GRAVITY * height / (REST_DENSITY * arguments.c0 ** 2) * 100:.2f} % of rho c0^2; "
          f"fluid {fluid_count:,}; diffusion {'off' if arguments.no_diffusion else f'delta {arguments.delta:g}'}"
          f"{' + gradient term' if arguments.gradient_term else ''}, pst {'off' if arguments.no_pst else 'on'}, "
          f"background {arguments.background:g} Pa, solid pressure {arguments.solid_pressure}, "
          f"pair correction {arguments.pair_correction}")
    header = f"{'step':>7s} {'t, s':>7s} " + " ".join(f"{name:>8s}" for name, _, _ in bands) \
        + f" {'rho min':>8s} {'rho max':>8s} {'p lid':>8s} {'p floor':>8s} {'|v| max':>8s} {'shift/dx':>9s} {'fallback':>8s}"
    print("fluid count of the slabs (2 dx thick) as % of the lattice count; p = median pressure of the slab, Pa")
    print(header, flush=True)

    def report(simulator):
        simulator.begin_readback_cache()
        positions = simulator.readback_positions()
        live = simulator.live_slot_mask(positions)
        material = simulator.readback_material()
        fluid = live & (material == 0)
        y = positions[:, 1].astype(np.float64)
        density_pressure = simulator.readback_density_pressure()
        speed = np.linalg.norm(simulator.readback_velocity_mass()[:, :3].astype(np.float64), axis=1)
        shift = np.linalg.norm(simulator.readback_shift()[:, :3].astype(np.float64), axis=1)
        status = simulator.readback_global_status()
        counts = []
        medians = {}
        for name, y0, y1 in bands:
            inside = fluid & (y >= y0) & (y < y1)
            counts.append(inside.sum() / lattice_per_slab * 100.0 * slab / (y1 - y0))
            medians[name] = float(np.median(density_pressure[inside, 1])) if inside.any() else float("nan")
        line = (f"{simulator.step_count:7d} {simulator.simulation_time:7.3f} " + " ".join(f"{c:7.1f}%" for c in counts)
                + f" {density_pressure[fluid, 0].min():8.2f} {density_pressure[fluid, 0].max():8.2f} "
                f"{medians['lid']:8.1f} {medians['floor']:8.1f} {speed[fluid].max():8.4f} {shift[fluid].mean() / dx:9.5f} "
                f"{status['correction_fallback_count']:8d}")
        simulator.end_readback_cache()
        print(line, flush=True)
        return status

    with VulkanContext.create(application_name="hydrostatic_drift", enable_validation=False) as context:
        simulator = SphSimulatorV1(context, case)
        try:
            simulator.bootstrap()
            report(simulator)
            while simulator.step_count < arguments.steps:
                simulator.step()
                if simulator.step_count % arguments.report == 0:
                    status = report(simulator)
                    if not np.isfinite(simulator.readback_density_pressure()[1:status["alive_particle_count"] + 1, 0]).all():
                        print("non-finite densities, stopping")
                        break
            status = simulator.readback_global_status()
            print(f"end: step {simulator.step_count}, alive {status['alive_particle_count']:,}, overflow "
                  f"{status['overflow_inside_count']}/{status['overflow_incoming_count']}, KCG fallback "
                  f"{status['correction_fallback_count']}")
            if arguments.dump:
                positions = simulator.readback_positions()
                np.savez_compressed(out / "final.npz", positions=positions, material=simulator.readback_material(),
                                    density_pressure=simulator.readback_density_pressure(),
                                    velocity_mass=simulator.readback_velocity_mass(), step=simulator.step_count,
                                    time=simulator.simulation_time)
        finally:
            simulator.destroy()
    return 0


if __name__ == "__main__":
    sys.exit(main())
