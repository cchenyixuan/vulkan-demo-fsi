"""_check_buoyancy_force.py — is the force read back for a solid body (readback_rotor_torque)
correctly scaled? Hydrostatic test: buoyancy of a block at rest (2026-09-29).

A closed box of fluid at rest under gravity holds a cubic block of ROTOR particles with zero
angular velocity. After the pressure has settled, the fluid force on the block must be the
buoyancy

    F_y = rho0 g V_block,     V_block = N_block dx^3   (the block occupies N lattice cells).

readback_rotor_torque() sums m_i (a_i - g) over the block's particles with the particle mass
m = rho0 V_p, where V_p is the CALIBRATED particle volume of utils/sph/case.py
(_calibrate_particle_volume: V_p = 1 / sum_{j != i} W_ij on the lattice, i.e. the neighbour
sum without the self term equals 1). V_p is larger than dx^3: 1.438 dx^3 for h/dx = 2.5,
1.219 dx^3 for h/dx = 3, 1.083 dx^3 for h/dx = 4. The accelerations are normalised by the KCG
matrix and do not depend on that factor, but a force m a does. This check measures the ratio

    F_y (read back) / (rho0 g N_block dx^3)

for several h/dx. A ratio equal to V_p / dx^3 means that every force, torque and power
number read back so far carries that factor.

Usage (repo root, solver env):
    python experiment/v1/checks/_check_buoyancy_force.py [--hdx 2.5 3 4] [--dx 0.004] [--out output/buoyancy]
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

from utils.sph.case import load_case, _calibrate_particle_volume      # noqa: E402
from utils.sph.vulkan_context import VulkanContext                    # noqa: E402
from experiment.v1 import compile_shaders_v1                          # noqa: E402
from experiment.v1.utils.simulator_v1 import SphSimulatorV1           # noqa: E402

GRAVITY = 9.81
REST_DENSITY = 1000.0


def write_points(path, points):
    np.savetxt(path, points, fmt="v %.7f %.7f %.7f", header=f"# {points.shape[0]} particles", comments="")


def build_case(directory, dx, hdx, cells=30, block_cells=8, speed_of_sound=12.0):
    directory.mkdir(parents=True, exist_ok=True)
    half = 0.5 * cells * dx
    centers = (np.arange(cells) + 0.5) * dx - half
    grid = np.stack(np.meshgrid(centers, centers, centers, indexing="ij"), axis=-1).reshape(-1, 3)
    block_half = 0.5 * block_cells * dx
    in_block = (np.abs(grid) < block_half).all(axis=1)
    block, fluid = grid[in_block], grid[~in_block]
    layers = int(math.ceil(hdx))
    extended = (np.arange(-layers, cells + layers) + 0.5) * dx - half
    shell = np.stack(np.meshgrid(extended, extended, extended, indexing="ij"), axis=-1).reshape(-1, 3)
    wall = shell[(np.abs(shell) > half).any(axis=1)]
    write_points(directory / "fluid.obj", fluid)
    write_points(directory / "wall.obj", wall)
    write_points(directory / "block.obj", block)
    points = np.vstack([fluid, wall, block])
    lo, hi = points.min(axis=0) - 0.6 * dx, points.max(axis=0) + 0.6 * dx
    with open(directory / "frame.obj", "w") as handle:
        for x in (lo[0], hi[0]):
            for y in (lo[1], hi[1]):
                for z in (lo[2], hi[2]):
                    handle.write(f"v {x:.7f} {y:.7f} {z:.7f}\n")
    pool = int(math.ceil(points.shape[0] * 1.15 / 128) * 128)
    bound = int(math.ceil(math.sqrt(2) * hdx ** 3))
    case = {
        "schema_version": 2,
        "time": {"total": None, "max_steps": None, "output_cadence": None},
        "physics": {"dimension": 3, "h": hdx * dx, "particle_radius": 0.5 * dx, "lattice": "grid",
                    "calibrate_volume": True, "speed_of_sound": speed_of_sound, "power": 7,
                    "cfl": 0.15, "gravity": [0.0, -GRAVITY, 0.0]},
        "numerics": {"use_density_diffusion": True, "delta_coefficient": 0.1, "use_kcg_correction": True,
                     "regularization": {"xi": 0.01, "det_threshold": 1.0e-4, "frobenius_max": 10.0},
                     "use_pst": True, "pst_main": 0.1, "pst_anti": 0.0005,
                     "defrag_enabled": False, "defrag_cadence": 1000, "use_prefix_sum_defrag": False},
        "capacities": {"pool_size": pool, "max_per_voxel": max(64, int(2 ** math.ceil(math.log2(bound * 1.3)))),
                       "max_incoming": 32, "workgroup": 128},
        "material_library": "materials.yaml",
        "rotor": {"axis": [0.0, 1.0, 0.0], "pivot": [0.0, 0.0, 0.0], "ramp_time": 0.0},
        "geometry": {"frame": "frame.obj",
                     "particles": [{"file": "fluid.obj", "material": "box_fluid"},
                                   {"file": "wall.obj", "material": "box_wall"},
                                   {"file": "block.obj", "material": "block"}]},
    }
    (directory / "case.yaml").write_text(yaml.safe_dump(case, sort_keys=False), encoding="utf-8")
    materials = {"schema_version": 1,
                 "box_fluid": {"kind": "fluid", "rest_density": REST_DENSITY, "viscosity": 1.0e-6},
                 "box_wall": {"kind": "boundary", "rest_density": REST_DENSITY, "viscosity": 1.0e-6},
                 "block": {"kind": "rotor", "rest_density": REST_DENSITY, "viscosity": 1.0e-6,
                           "rotor_angular_velocity": 0.0}}
    (directory / "materials.yaml").write_text(yaml.safe_dump(materials, sort_keys=False), encoding="utf-8")
    return directory / "case.yaml", block.shape[0], fluid.shape[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--hdx", type=float, nargs="+", default=[2.5, 3.0, 4.0])
    parser.add_argument("--dx", type=float, default=0.004)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--out", default="output/buoyancy")
    arguments = parser.parse_args()
    compile_shaders_v1.compile_v1_shaders()
    out = pathlib.Path(arguments.out)
    print("h/dx   V_p/dx^3   block particles   F_y read back [N]   rho0 g N dx^3 [N]   ratio   std of samples   mass read back / (rho0 dx^3)")
    for hdx in arguments.hdx:
        case_path, block_count, fluid_count = build_case(out / f"hdx_{hdx:g}", arguments.dx, hdx)
        case = load_case(str(case_path))
        samples = []
        with VulkanContext.create(application_name="buoyancy", enable_validation=False) as context:
            simulator = SphSimulatorV1(context, case)
            try:
                simulator.bootstrap()
                while simulator.step_count < arguments.steps:
                    simulator.step()
                    if simulator.step_count > arguments.steps // 2 and simulator.step_count % 20 == 0:
                        samples.append(simulator.readback_rotor_torque()["force"][1])
                status = simulator.readback_global_status()
                mass = float(simulator.readback_velocity_mass()[1, 3])
            finally:
                simulator.destroy()
        samples = np.array(samples)
        expected = REST_DENSITY * GRAVITY * block_count * arguments.dx ** 3
        calibrated = _calibrate_particle_volume(hdx * arguments.dx, 0.5 * arguments.dx, 3, "grid") / arguments.dx ** 3
        print(f"{hdx:4g}   {calibrated:8.4f}   {block_count:15d}   {samples.mean():18.4f}   {expected:17.4f}   "
              f"{samples.mean() / expected:5.3f}   {samples.std():14.4f}   {mass / (REST_DENSITY * arguments.dx ** 3):.4f}"
              f"      (alive {status['alive_particle_count']:,}, overflow {status['overflow_inside_count']}/"
              f"{status['overflow_incoming_count']}, t = {simulator.simulation_time:.2f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
