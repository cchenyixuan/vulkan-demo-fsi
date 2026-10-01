"""_check_reaction_box.py — verification of the Monod uptake and the continuous source (2026-10-01, stage 3).

A closed 3D box of fluid at rest (simple cubic lattice, wall shell, no gravity, no rotor) carries four
scalar fields in one vec4: substrate C, biomass X, cumulative uptake U, cumulative feed F.

  R1  batch, no growth: uniform C0, X. The GPU must reproduce the solver's own update
          C_{n+1} = C_n / (1 + q_max X dt / (K_s + C_n))
      iterated in double precision (relative difference < 2e-4, float32 accumulation); all particles stay
      equal; sum (C + U) is conserved. Reported, not tested: the distance of that update to the exact
      solution K_s ln(C0 / C) + (C0 - C) = q_max X t, a first-order time-discretisation error that grows
      like (lambda t)(lambda dt) / 2 with lambda = q_max X / K_s once C << K_s.
  R2  batch with growth (dimensionless parameters): dX/dt = Y q X, dC/dt = -q X; compared with an RK4
      integration of the ODE pair; X + Y C is conserved.
  R3  fed box: Haringa 2023 kinetics, a source sphere in the centre (rate given), reaction on.
      sum m (C + U - F) must stay at its initial value; the fed amount sum rho0 dx^3 F must equal
      rate t N_in dx^3 / V_sphere (N_in particles in the sphere, fixed at rest); C >= 0 everywhere.

Usage (repo root, solver env):
    python experiment/v1/checks/_check_reaction_box.py [--only R1,R2,R3] [--out output/reaction_checks]
"""
import argparse
import json
import math
import pathlib
import sys

import numpy as np
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REST_DENSITY = 1000.0
Q_MAX_HARINGA = 1600e-6 / 3600.0      # mol / (g s)
K_S_HARINGA = 7.8e-6                  # mol / kg
X_HARINGA = 55.0                      # g / kg


def write_points(path, points):
    np.savetxt(path, points, fmt="v %.7f %.7f %.7f", header=f"# {points.shape[0]} particles", comments="")


def build_case(directory, *, dx=0.004, hdx=3.0, cells=16, c0=15.0, fields, reaction, sources=()):
    directory.mkdir(parents=True, exist_ok=True)
    half = 0.5 * cells * dx
    centres = (np.arange(cells) + 0.5) * dx - half
    fluid = np.stack(np.meshgrid(centres, centres, centres, indexing="ij"), axis=-1).reshape(-1, 3)
    layers = int(math.ceil(hdx))
    extended = (np.arange(-layers, cells + layers) + 0.5) * dx - half
    shell = np.stack(np.meshgrid(extended, extended, extended, indexing="ij"), axis=-1).reshape(-1, 3)
    wall = shell[(np.abs(shell) > half).any(axis=1)]
    write_points(directory / "fluid.obj", fluid)
    write_points(directory / "wall.obj", wall)
    points = np.vstack([fluid, wall])
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
                    "calibrate_volume": True, "speed_of_sound": c0, "power": 7, "cfl": 0.15,
                    "gravity": [0.0, 0.0, 0.0]},
        "numerics": {"use_density_diffusion": True, "delta_coefficient": 0.1, "use_kcg_correction": True,
                     "regularization": {"xi": 0.01, "det_threshold": 1.0e-4, "frobenius_max": 10.0},
                     "use_pst": True, "pst_main": 0.1, "pst_anti": 0.0005,
                     "defrag_enabled": True, "defrag_cadence": 10, "use_prefix_sum_defrag": False},
        "capacities": {"pool_size": pool, "max_per_voxel": max(64, int(2 ** math.ceil(math.log2(bound * 1.3)))),
                       "max_incoming": 32, "workgroup": 128},
        "material_library": "materials.yaml",
        "geometry": {"frame": "frame.obj",
                     "particles": [{"file": "fluid.obj", "material": "box_fluid"},
                                   {"file": "wall.obj", "material": "box_wall"}]},
        "scalars": {"fields": fields, "sgs": {"enabled": False}, "reactions": [reaction],
                    "sources": list(sources)},
    }
    (directory / "case.yaml").write_text(yaml.safe_dump(case, sort_keys=False), encoding="utf-8")
    materials = {"schema_version": 1,
                 "box_fluid": {"kind": "fluid", "rest_density": REST_DENSITY, "viscosity": 1.0e-6},
                 "box_wall": {"kind": "boundary", "rest_density": REST_DENSITY, "viscosity": 1.0e-6}}
    (directory / "materials.yaml").write_text(yaml.safe_dump(materials, sort_keys=False), encoding="utf-8")
    return directory / "case.yaml", fluid


def run_box(case_path, steps, samples):
    from utils.sph.case import load_case
    from utils.sph.vulkan_context import VulkanContext
    from experiment.v1.utils.simulator_v1 import SphSimulatorV1
    case = load_case(str(case_path))
    every = max(1, steps // samples)
    series = []
    with VulkanContext.create(application_name="reaction_box", enable_validation=False) as context:
        simulator = SphSimulatorV1(context, case)
        try:
            simulator.bootstrap()
            def sample():
                snapshot = simulator.scalar_snapshot()
                series.append({"time": simulator.simulation_time, "values": snapshot["scalars"].copy(),
                               "mass": snapshot["mass"].copy(), "positions": snapshot["positions"].copy()})
            sample()
            while simulator.step_count < steps:
                simulator.step()
                if simulator.step_count % every == 0:
                    sample()
            status = simulator.readback_global_status()
        finally:
            simulator.destroy()
    return case, series, status


def exact_batch(c0, q_x, k_s, times):
    """Solve k_s ln(c0 / c) + (c0 - c) = q_x t for c by Newton iteration."""
    result = []
    c = c0
    for t in times:
        target = q_x * t
        for _ in range(100):
            f = k_s * math.log(c0 / c) + (c0 - c) - target
            derivative = -k_s / c - 1.0
            step = f / derivative
            c_new = c - step
            if c_new <= 0.0:
                c_new = 0.5 * c
            if abs(c_new - c) < 1e-15 * max(c, 1e-300):
                c = c_new
                break
            c = c_new
        result.append(c)
    return np.asarray(result)


def rk4_growth(c0, x0, q_max, k_s, growth_yield, times):
    def rhs(state):
        c, x = state
        q = q_max * max(c, 0.0) / (k_s + max(c, 0.0))
        return np.array([-q * x, growth_yield * q * x])
    state = np.array([c0, x0], dtype=np.float64)
    out = [state.copy()]
    for t0, t1 in zip(times[:-1], times[1:]):
        steps = max(1, int(math.ceil((t1 - t0) / 1e-3)))
        h = (t1 - t0) / steps
        for _ in range(steps):
            k1 = rhs(state); k2 = rhs(state + 0.5 * h * k1); k3 = rhs(state + 0.5 * h * k2); k4 = rhs(state + h * k3)
            state = state + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        out.append(state.copy())
    return np.asarray(out)


FIELDS_FOUR = lambda c0, x0: [
    {"name": "substrate", "diffusivity": 6.0e-10, "turbulent": False, "initial": c0},
    {"name": "biomass", "diffusivity": 0.0, "turbulent": False, "initial": x0},
    {"name": "uptake", "diffusivity": 0.0, "turbulent": False, "initial": 0.0},
    {"name": "feed", "diffusivity": 0.0, "turbulent": False, "initial": 0.0}]


def check_r1(out):
    c0 = 10.0 * K_S_HARINGA
    reaction = {"type": "monod", "substrate": "substrate", "biomass": "biomass", "uptake": "uptake",
                "q_max": Q_MAX_HARINGA, "k_s": K_S_HARINGA, "yield": 0.0}
    case_path, _ = build_case(out / "r1", fields=FIELDS_FOUR(c0, X_HARINGA), reaction=reaction)
    from utils.sph.case import load_case
    dt = load_case(str(case_path)).timestep
    steps = int(round(6.0 / dt))
    case, series, status = run_box(case_path, steps, 60)
    times = np.array([s["time"] for s in series])
    mean_c = np.array([s["values"][:, 0].mean() for s in series])
    spread = np.array([s["values"][:, 0].max() - s["values"][:, 0].min() for s in series])
    conserved = np.array([(s["values"][:, 0] + s["values"][:, 2]).mean() for s in series])
    exact = exact_batch(c0, Q_MAX_HARINGA * X_HARINGA, K_S_HARINGA, times)
    relative = (mean_c - exact) / exact
    # the solver's own recurrence in double precision, sampled at the recorded steps
    k = Q_MAX_HARINGA * X_HARINGA
    recorded_steps = np.rint(times / dt).astype(np.int64)
    recurrence = np.empty(times.size)
    value = c0
    step = 0
    for index, target in enumerate(recorded_steps):
        while step < target:
            value = value / (1.0 + k * dt / (K_S_HARINGA + value))
            step += 1
        recurrence[index] = value
    implementation = (mean_c - recurrence) / recurrence
    # the linearised implicit update has a first-order splitting error ~ dt / t_reaction
    t_reaction = (K_S_HARINGA + c0) / (Q_MAX_HARINGA * X_HARINGA)
    print(f"R1 batch: dt {dt:.3e} s, {steps} steps, C0 = {c0:.3e}, t_reaction {t_reaction:.3f} s")
    for index in range(0, len(times), max(1, len(times) // 8)):
        print(f"   t {times[index]:6.3f} s: C {mean_c[index]:.6e}  recurrence {recurrence[index]:.6e} "
              f"({implementation[index]:+.1e})  exact {exact[index]:.6e} ({relative[index]:+.1e})  "
              f"spread {spread[index]:.1e}  C+U {conserved[index]:.9e}")
    print(f"   alive {status['alive_particle_count']}, overflow {status['overflow_inside_count']}/{status['overflow_incoming_count']}")
    select = mean_c > 1e-3 * c0
    good = np.abs(implementation[select]).max()
    discretisation = np.abs(relative[select]).max()
    drift = abs(conserved[-1] - conserved[0]) / conserved[0]
    passed = good < 2e-4 and drift < 1e-5 and spread.max() < 1e-6 * c0
    lam = k / K_S_HARINGA
    print(f"   while C > 1e-3 C0: max |GPU / recurrence - 1| {good:.2e}; max |recurrence / exact - 1| {discretisation:.2e} "
          f"(first order: lambda dt = {lam * dt:.1e} per step); C+U drift {drift:.1e}; max spread {spread.max():.1e} "
          f"-> {'PASS' if passed else 'FAIL'}")
    return {"name": "R1", "passed": bool(passed), "implementation_error": float(good),
            "discretisation_error": float(discretisation), "conserved_drift": float(drift)}


def check_r2(out):
    c0, x0, q_max, k_s, growth_yield = 1.0, 0.1, 2.0, 0.2, 0.5
    reaction = {"type": "monod", "substrate": "substrate", "biomass": "biomass", "uptake": "uptake",
                "q_max": q_max, "k_s": k_s, "yield": growth_yield}
    case_path, _ = build_case(out / "r2", fields=FIELDS_FOUR(c0, x0), reaction=reaction)
    from utils.sph.case import load_case
    dt = load_case(str(case_path)).timestep
    steps = int(round(8.0 / dt))
    case, series, status = run_box(case_path, steps, 40)
    times = np.array([s["time"] for s in series])
    mean_c = np.array([s["values"][:, 0].mean() for s in series])
    mean_x = np.array([s["values"][:, 1].mean() for s in series])
    reference = rk4_growth(c0, x0, q_max, k_s, growth_yield, times)
    invariant = mean_x + growth_yield * mean_c
    error_c = np.abs(mean_c - reference[:, 0]).max()
    error_x = np.abs(mean_x - reference[:, 1]).max()
    print(f"R2 growth: dt {dt:.3e} s, {steps} steps")
    for index in range(0, len(times), max(1, len(times) // 8)):
        print(f"   t {times[index]:6.3f}: C {mean_c[index]:.5f} (RK4 {reference[index, 0]:.5f})  X {mean_x[index]:.5f} "
              f"(RK4 {reference[index, 1]:.5f})  X + Y C {invariant[index]:.6f}")
    drift = abs(invariant[-1] - invariant[0]) / invariant[0]
    passed = error_c < 2e-3 and error_x < 2e-3 and drift < 1e-5
    print(f"   max |C - RK4| {error_c:.2e}, max |X - RK4| {error_x:.2e}, X + Y C drift {drift:.1e} -> {'PASS' if passed else 'FAIL'}")
    return {"name": "R2", "passed": bool(passed), "error_c": float(error_c), "error_x": float(error_x)}


def check_r3(out):
    c0 = 10.0 * K_S_HARINGA
    radius = 0.015
    rate = 2.0e-9                      # mol / s into a 14 mL sphere
    reaction = {"type": "monod", "substrate": "substrate", "biomass": "biomass", "uptake": "uptake",
                "q_max": Q_MAX_HARINGA, "k_s": K_S_HARINGA, "yield": 0.0}
    source = {"field": "substrate", "center": [0.0, 0.0, 0.0], "radius": radius, "rate": rate,
              "start": 0.5, "record": "feed"}
    dx = 0.004
    case_path, fluid = build_case(out / "r3", dx=dx, fields=FIELDS_FOUR(c0, X_HARINGA), reaction=reaction, sources=[source])
    from utils.sph.case import load_case
    dt = load_case(str(case_path)).timestep
    steps = int(round(4.0 / dt))
    case, series, status = run_box(case_path, steps, 40)
    times = np.array([s["time"] for s in series])
    inside = int((np.linalg.norm(fluid, axis=1) < radius).sum())
    volume = 4.0 / 3.0 * math.pi * radius ** 3
    budget = np.array([np.sum(s["mass"] * (s["values"][:, 0] + s["values"][:, 2] - s["values"][:, 3])) for s in series])
    fed = np.array([REST_DENSITY * dx ** 3 * s["values"][:, 3].sum() for s in series])
    expected = rate * np.maximum(times - 0.5, 0.0) * inside * dx ** 3 / volume
    minimum = np.array([s["values"][:, 0].min() for s in series])
    print(f"R3 fed box: dt {dt:.3e} s, {steps} steps, {inside} particles in the source sphere "
          f"(N_in dx^3 / V = {inside * dx ** 3 / volume:.4f})")
    for index in range(0, len(times), max(1, len(times) // 8)):
        print(f"   t {times[index]:6.3f}: fed {fed[index]:.6e} mol (expected {expected[index]:.6e})  "
              f"budget sum m (C + U - F) {budget[index]:.9e}  min C {minimum[index]:.2e}")
    drift = abs(budget[-1] - budget[0]) / abs(budget[0])
    fed_error = abs(fed[-1] - expected[-1]) / expected[-1]
    passed = drift < 1e-5 and fed_error < 1e-4 and minimum.min() >= 0.0
    print(f"   budget drift {drift:.1e}, fed error {fed_error:.1e}, min C {minimum.min():.2e} -> {'PASS' if passed else 'FAIL'}")
    return {"name": "R3", "passed": bool(passed), "budget_drift": float(drift), "fed_error": float(fed_error)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--only", default="R1,R2,R3")
    parser.add_argument("--out", default="output/reaction_checks")
    arguments = parser.parse_args()
    from experiment.v1 import compile_shaders_v1
    compile_shaders_v1.compile_v1_shaders()
    out = pathlib.Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for name, function in (("R1", check_r1), ("R2", check_r2), ("R3", check_r3)):
        if name in arguments.only.split(","):
            results.append(function(out))
    (out / "summary.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    return 0 if all(result["passed"] for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
