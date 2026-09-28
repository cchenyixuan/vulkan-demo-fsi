"""_check_effective_viscosity.py — effective viscosity of the solver from the spin-up of
the 2D Taylor-Couette flow, for several KCG regularisation values xi (2026-09-28).

The steady Couette profile u = A r + B / r does not depend on the viscosity, so the
steady check (_check_taylor_couette.py) cannot see an error in the amplitude of the
viscous operator. The spin-up does: the azimuthal velocity obeys

    du/dt = nu (d2u/dr2 + (1/r) du/dr - u / r^2),   u(r_i, t) = omega(t) r_i,   u(r_o, t) = 0,

with the rotor's linear ramp omega(t). This script runs cases/taylor_couette_2d for
each xi (numerics.regularization.xi, the only change), records the binned profile of
the fluid particles at several times, solves the equation above for a range of nu and
reports the nu that fits all recorded profiles best, as a ratio to the material value.

Lattice prediction (_check_operator_consistency.py, 2D, h/dx = 5, eta^2 = 0.01 h^2):
    xi = 0.1: 0.840    xi = 0.01: 0.907    xi = 0.001: 0.914    (M^-1 exact: 0.915)

Usage (repo root, solver env):
    python experiment/v1/checks/_check_effective_viscosity.py [--xi 0.1 0.01 0.001]
        [--times 0.2 0.4 0.7 1.0 1.5] [--out output/xi_eval]
"""
import argparse
import pathlib
import re
import shutil
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.sph.case import load_case                                  # noqa: E402
from utils.sph.vulkan_context import VulkanContext                    # noqa: E402
from experiment.v1 import compile_shaders_v1                          # noqa: E402
from experiment.v1.utils.simulator_v1 import SphSimulatorV1           # noqa: E402

INNER_RADIUS, OUTER_RADIUS = 0.05, 0.10
BIN_EDGES = np.linspace(INNER_RADIUS, OUTER_RADIUS, 26)


def binned_profile(simulator):
    positions = simulator.readback_positions()
    live = simulator.live_slot_mask(positions)
    fluid = live & (simulator.readback_material() == 0)
    x = positions[fluid, :2].astype(np.float64)
    v = simulator.readback_velocity_mass()[fluid, :2].astype(np.float64)
    radius = np.hypot(x[:, 0], x[:, 1])
    tangential = (-x[:, 1] * v[:, 0] + x[:, 0] * v[:, 1]) / radius
    index = np.digitize(radius, BIN_EDGES)
    centres, values = [], []
    for bin_index in range(1, len(BIN_EDGES)):
        mask = index == bin_index
        if mask.sum() >= 5:
            centres.append(radius[mask].mean())
            values.append(tangential[mask].mean())
    return np.array(centres), np.array(values)


def run_case(case_path, times):
    case = load_case(str(case_path))
    profiles = []
    with VulkanContext.create(application_name="effective_viscosity", enable_validation=False) as context:
        simulator = SphSimulatorV1(context, case)
        try:
            simulator.bootstrap()
            for target in sorted(times):
                while simulator.simulation_time < target - 0.5 * case.timestep:
                    simulator.step()
                centres, values = binned_profile(simulator)
                profiles.append((simulator.simulation_time, centres, values))
            status = simulator.readback_global_status()
            ramp = lambda t: simulator.rotor_angle_and_rate(t)[1]            # noqa: E731
            omega = [ramp(t) for t in np.linspace(0.0, max(times), 2001)]
        finally:
            simulator.destroy()
    return case, profiles, status, np.array(omega)


def solve_spin_up(viscosity, omega, end_time, nodes=201):
    """Crank-Nicolson solution on a uniform radial grid; returns (times, r, u[time, r])."""
    radius = np.linspace(INNER_RADIUS, OUTER_RADIUS, nodes)
    spacing = radius[1] - radius[0]
    steps = len(omega) - 1
    timestep = end_time / steps
    operator = np.zeros((nodes, nodes))
    for k in range(1, nodes - 1):
        operator[k, k - 1] = 1.0 / spacing ** 2 - 0.5 / (radius[k] * spacing)
        operator[k, k] = -2.0 / spacing ** 2 - 1.0 / radius[k] ** 2
        operator[k, k + 1] = 1.0 / spacing ** 2 + 0.5 / (radius[k] * spacing)
    identity = np.eye(nodes)
    left = identity - 0.5 * timestep * viscosity * operator
    right = identity + 0.5 * timestep * viscosity * operator
    left[0, :], left[-1, :] = 0.0, 0.0
    left[0, 0], left[-1, -1] = 1.0, 1.0
    left_inverse = np.linalg.inv(left)
    solution = np.zeros((steps + 1, nodes))
    for n in range(steps):
        rhs = right @ solution[n]
        rhs[0], rhs[-1] = omega[n + 1] * INNER_RADIUS, 0.0
        solution[n + 1] = left_inverse @ rhs
    return np.linspace(0.0, end_time, steps + 1), radius, solution


def misfit(viscosity, omega, end_time, profiles):
    times, radius, solution = solve_spin_up(viscosity, omega, end_time)
    total, count = 0.0, 0
    for time, centres, values in profiles:
        row = solution[int(round(time / end_time * (len(times) - 1)))]
        total += ((np.interp(centres, radius, row) - values) ** 2).sum()
        count += len(values)
    return np.sqrt(total / count)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--xi", type=float, nargs="+", default=[0.1, 0.01, 0.001])
    parser.add_argument("--times", type=float, nargs="+", default=[0.2, 0.4, 0.7, 1.0, 1.5])
    parser.add_argument("--out", default="output/xi_eval")
    parser.add_argument("--case-dir", default="cases/taylor_couette_2d")
    arguments = parser.parse_args()
    compile_shaders_v1.compile_v1_shaders()
    out = pathlib.Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    end_time = max(arguments.times)
    results = []
    for xi in arguments.xi:
        case_dir = out / f"couette_xi_{xi:g}"
        if case_dir.exists():
            shutil.rmtree(case_dir)
        shutil.copytree(arguments.case_dir, case_dir)
        text = (case_dir / "case.yaml").read_text(encoding="utf-8")
        text, replaced = re.subn(r"(\n\s*xi:\s*)[0-9.eE+-]+", rf"\g<1>{xi:g}", text)
        assert replaced == 1, "numerics.regularization.xi not found in case.yaml"
        (case_dir / "case.yaml").write_text(text, encoding="utf-8")
        case, profiles, status, omega = run_case(case_dir / "case.yaml", arguments.times)
        viscosity = [material.viscosity for material in case.materials if material.kind == 0][0]
        ratios = np.linspace(0.60, 1.20, 61)
        errors = np.array([misfit(ratio * viscosity, omega, end_time, profiles) for ratio in ratios])
        best = int(np.argmin(errors))
        fine = np.linspace(ratios[max(best - 1, 0)], ratios[min(best + 1, len(ratios) - 1)], 41)
        fine_errors = np.array([misfit(ratio * viscosity, omega, end_time, profiles) for ratio in fine])
        ratio = float(fine[int(np.argmin(fine_errors))])
        inner_speed = abs(omega[-1]) * INNER_RADIUS
        print(f"xi = {xi:g}: effective viscosity / nu = {ratio:.3f}   (rms misfit {fine_errors.min() / inner_speed * 100:.2f} % of u_inner; "
              f"with nu itself {misfit(viscosity, omega, end_time, profiles) / inner_speed * 100:.2f} %)   "
              f"alive {status['alive_particle_count']:,}, fallback {status['correction_fallback_count']}, "
              f"overflow {status['overflow_inside_count']}/{status['overflow_incoming_count']}")
        results.append((xi, ratio))
    print("lattice prediction (2D, h/dx = 5, eta^2 = 0.01 h^2): xi 0.1 -> 0.840, 0.01 -> 0.907, 0.001 -> 0.914")
    return 0


if __name__ == "__main__":
    sys.exit(main())
