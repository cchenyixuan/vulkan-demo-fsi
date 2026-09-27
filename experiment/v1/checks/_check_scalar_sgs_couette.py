"""_check_scalar_sgs_couette.py — check of the Smagorinsky nu_t (2026-09-27).

Runs the 2D Taylor-Couette validation case (cases/taylor_couette_2d, inner
ROTOR Omega = 2 rad/s, r_i = 0.05, r_o = 0.10, validated to 0.25 % in
log/2026-09-25_rotor-motion.md) with a `scalars:` block whose SGS is on, and
compares the nu_t that density.comp computes with

    nu_t = (C_s Delta)^2 |S|,   |S| = sqrt(2 S:S) = | r d(u_theta / r) / dr |

evaluated (a) from the analytic steady profile u_theta = A r + B / r, for which
|S| = 2 |B| / r^2, and (b) from the SPH velocity itself (binned profile,
finite-difference derivative), which separates the gradient operator from the
flow error. The case geometry (.obj) is copied from cases/taylor_couette_2d.

Usage (repo root):
    python experiment/v1/checks/_check_scalar_sgs_couette.py [--steps 40000] [--out output/scalar_checks]
"""
import argparse
import json
import pathlib
import shutil
import sys

import numpy as np
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib                                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402

from utils.sph.case import load_case                                 # noqa: E402
from utils.sph.vulkan_context import VulkanContext                   # noqa: E402
from experiment.v1 import compile_shaders_v1                         # noqa: E402
from experiment.v1.utils.simulator_v1 import SphSimulatorV1          # noqa: E402

INNER_RADIUS, OUTER_RADIUS, OMEGA = 0.05, 0.10, 2.0
SMAGORINSKY_CS = 0.1


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--steps", type=int, default=40000)
    parser.add_argument("--out", default="output/scalar_checks")
    args = parser.parse_args()
    out = (ROOT / args.out).resolve()
    work = out / "cases" / "couette_sgs"
    work.mkdir(parents=True, exist_ok=True)
    source = ROOT / "cases" / "taylor_couette_2d"
    for name in ("fluid.obj", "wall.obj", "rotor.obj", "frame.obj", "materials.yaml"):
        shutil.copy(source / name, work / name)
    case_data = yaml.safe_load((source / "case.yaml").read_text(encoding="utf-8"))
    case_data["scalars"] = {
        "fields": [{"name": "tracer", "diffusivity": 0.0, "turbulent": True}],
        "sgs": {"enabled": True, "smagorinsky_cs": SMAGORINSKY_CS, "turbulent_schmidt": 0.7},
    }
    (work / "case.yaml").write_text(yaml.safe_dump(case_data, sort_keys=False), encoding="utf-8")

    compile_shaders_v1.compile_v1_shaders()
    case = load_case(work / "case.yaml")
    filter_width = case.physics.particle_diameter
    with VulkanContext.create(application_name="scalar_sgs_couette", enable_validation=False) as ctx:
        with SphSimulatorV1(ctx, case) as sim:
            sim.bootstrap()
            for _ in range(args.steps):
                sim.step()
            positions = sim.readback_positions()
            live = sim.live_slot_mask(positions)
            material = sim.readback_material()
            velocity = sim.readback_velocity_mass()[:, :3].astype(np.float64)
            nu_t = sim.readback_turbulent_viscosity().astype(np.float64)
            fluid = live & np.isin(material, np.asarray(sim.fluid_group_ids(), dtype=np.uint32))
            t_end = sim.simulation_time

    x = positions[fluid, :2].astype(np.float64)
    v = velocity[fluid, :2]
    nu = nu_t[fluid]
    r = np.hypot(x[:, 0], x[:, 1])
    tangential = np.column_stack([-x[:, 1], x[:, 0]]) / r[:, None]
    u_theta = (v * tangential).sum(axis=1)
    A = OMEGA * INNER_RADIUS ** 2 / (INNER_RADIUS ** 2 - OUTER_RADIUS ** 2)
    B = -A * OUTER_RADIUS ** 2
    length_squared = (SMAGORINSKY_CS * filter_width) ** 2

    edges = np.linspace(INNER_RADIUS, OUTER_RADIUS, 26)
    index = np.digitize(r, edges)
    rows = []
    for b in range(1, len(edges)):
        sel = index == b
        if sel.sum() < 5:
            continue
        rows.append((r[sel].mean(), nu[sel].mean(), nu[sel].std(), (u_theta[sel] / r[sel]).mean()))
    rc, nu_mean, nu_std, angular = map(np.array, zip(*rows))
    nu_analytic = length_squared * 2 * abs(B) / rc ** 2
    # |S| from the SPH profile: r d(u/r)/dr by central differences of the binned angular velocity
    strain_from_sph = np.abs(rc * np.gradient(angular, rc))
    nu_from_sph_profile = length_squared * strain_from_sph
    # exclude the first / last bin (one kernel from the walls, where the fluid-only
    # SPH sum is kernel-deficient and the wall particles' velocity enters)
    interior = (rc > INNER_RADIUS + case.physics.h) & (rc < OUTER_RADIUS - case.physics.h)
    ratio_analytic = nu_mean[interior] / nu_analytic[interior]
    ratio_profile = nu_mean[interior] / nu_from_sph_profile[interior]
    result = {
        "steps": args.steps, "t_end": t_end, "filter_width": filter_width, "cs": SMAGORINSKY_CS,
        "bins": [{"r": float(a), "nu_t_mean": float(b), "nu_t_std": float(c), "nu_t_analytic": float(d),
                  "nu_t_from_sph_profile": float(e)}
                 for a, b, c, d, e in zip(rc, nu_mean, nu_std, nu_analytic, nu_from_sph_profile)],
        "interior_ratio_to_analytic_mean": float(ratio_analytic.mean()),
        "interior_ratio_to_analytic_range": [float(ratio_analytic.min()), float(ratio_analytic.max())],
        "interior_ratio_to_sph_profile_mean": float(ratio_profile.mean()),
        "interior_ratio_to_sph_profile_range": [float(ratio_profile.min()), float(ratio_profile.max())],
    }
    (out / "t6_sgs_couette.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "bins"}, indent=1))
    print(f"{'r':>7} {'nu_t SPH':>11} {'std':>9} {'analytic':>11} {'from SPH u':>11}")
    for row in result["bins"]:
        print(f"{row['r']:7.4f} {row['nu_t_mean']:11.4e} {row['nu_t_std']:9.2e} "
              f"{row['nu_t_analytic']:11.4e} {row['nu_t_from_sph_profile']:11.4e}")

    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    rr = np.linspace(INNER_RADIUS, OUTER_RADIUS, 200)
    ax.plot(rr * 1e3, length_squared * 2 * abs(B) / rr ** 2 * 1e8, "k-", label="(C_s dx)^2 2|B|/r^2 (analytic)")
    ax.errorbar(rc * 1e3, nu_mean * 1e8, yerr=nu_std * 1e8, fmt="o", ms=3, label="density.comp nu_t (bin mean ± std)")
    ax.plot(rc * 1e3, nu_from_sph_profile * 1e8, "s", mfc="none", ms=4, label="(C_s dx)^2 |r d(u/r)/dr| of SPH u")
    ax.set_xlabel("r (mm)"); ax.set_ylabel("nu_t (1e-8 m^2/s)"); ax.legend(fontsize=8)
    ax.set_title(f"T6 Smagorinsky nu_t, 2D Taylor-Couette, t = {t_end:.2f} s")
    fig.tight_layout(); fig.savefig(out / "t6_sgs_couette.png", dpi=120)
    print("saved", out / "t6_sgs_couette.png")


if __name__ == "__main__":
    main()
