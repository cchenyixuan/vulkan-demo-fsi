"""_check_scalar_box.py — validation of the scalar transport (2026-09-27).

Builds small closed 3D boxes of fluid at rest (simple cubic lattice, wall
shell of ceil(h/dx) layers, no gravity) on the fly and checks:

  T1 diffusion     Gaussian blob with constant D. The per-axis variance must
                   grow as sigma^2(t) = sigma_0^2 + 2 D t, the peak must decay
                   as (sigma_0^2 / sigma^2)^(3/2), and the profile must follow
                   the analytic Gaussian. Defrag every 10 steps, so the scalar
                   buffers are also carried through defrag during the test.
  T2 conservation  sum_i m_i C_i of every field (Kahan-compensated values,
                   float64 sum) stays at its initial value (same run as T1).
  T3 no diffusion  a second field with D = 0 is left unchanged (same run).
  T4 compensation  a bump of 0.1 on a background of 1 with a diffusivity so
                   small that every per-step increment is below half an ulp
                   of C: plain float32 accumulation must stall completely,
                   the compensated sum must follow the analytic decay. A third
                   variant switches the shift correction off to attribute the
                   small drift of the total (the shift term is not conservative).
  T5 shift         jittered lattice (particles relax, the δ-plus shift is
                   active), linear field C = 1 + 10 x with D = 0. With the
                   shift correction each interior particle must satisfy
                   C(t) = C(0) + 10 * (sum of its applied shifts in x); without
                   it C stays at C(0) and the whole shift displacement is error.

Usage (repo root):
    python experiment/v1/checks/_check_scalar_box.py [--out output/scalar_checks] [--only T1,T4,T5]
Writes summary.json and figures to --out. See log/2026-09-27_scalar-transport.md.
"""
import argparse
import json
import math
import pathlib
import sys
import time

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


# ----------------------------------------------------------------------------
# Case construction
# ----------------------------------------------------------------------------

def write_points(path, points):
    np.savetxt(path, points, fmt="v %.7f %.7f %.7f", header=f"# {points.shape[0]} particles", comments="")


def write_frame(path, lo, hi):
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    with open(path, "w") as handle:
        for v in [(x0, y0, z0), (x0, y1, z0), (x1, y0, z0), (x1, y1, z0),
                  (x0, y0, z1), (x0, y1, z1), (x1, y0, z1), (x1, y1, z1)]:
            handle.write(f"v {v[0]:.7f} {v[1]:.7f} {v[2]:.7f}\n")


def build_box(directory, *, half_length, dx, hdx, speed_of_sound, scalars,
              jitter=0.0, seed=0, defrag_cadence=None):
    """Closed cube [-L, L]^3 of fluid at rest. Returns the case.yaml path."""
    directory = pathlib.Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    n = int(round(2 * half_length / dx))
    centers = (np.arange(n) + 0.5) * dx - half_length
    grid = np.stack(np.meshgrid(centers, centers, centers, indexing="ij"), axis=-1).reshape(-1, 3)
    fluid = grid.copy()
    if jitter > 0.0:
        fluid += np.random.default_rng(seed).uniform(-jitter, jitter, fluid.shape) * dx
    layers = int(math.ceil(hdx))
    extended = (np.arange(-layers, n + layers) + 0.5) * dx - half_length
    shell = np.stack(np.meshgrid(extended, extended, extended, indexing="ij"), axis=-1).reshape(-1, 3)
    wall = shell[(np.abs(shell) > half_length).any(axis=1)]
    write_points(directory / "fluid.obj", fluid)
    write_points(directory / "wall.obj", wall)
    points = np.vstack([fluid, wall])
    write_frame(directory / "frame.obj", points.min(axis=0) - 0.6 * dx, points.max(axis=0) + 0.6 * dx)
    total = points.shape[0]
    pool = int(math.ceil(total * 1.15 / 128) * 128)
    case = {
        "schema_version": 2,
        "time": {"total": None, "max_steps": None, "output_cadence": None},
        "physics": {"dimension": 3, "h": hdx * dx, "particle_radius": 0.5 * dx, "lattice": "grid",
                    "calibrate_volume": True, "speed_of_sound": speed_of_sound, "power": 7,
                    "cfl": 0.15, "gravity": [0.0, 0.0, 0.0]},
        "numerics": {"use_density_diffusion": True, "delta_coefficient": 0.1, "use_kcg_correction": True,
                     "regularization": {"xi": 0.1, "det_threshold": 1.0e-4, "frobenius_max": 10.0},
                     "use_pst": True, "pst_main": 0.1, "pst_anti": 0.0005,
                     "defrag_enabled": defrag_cadence is not None,
                     "defrag_cadence": defrag_cadence or 1000, "use_prefix_sum_defrag": False},
        "capacities": {"pool_size": pool, "max_per_voxel": 64, "max_incoming": 32, "workgroup": 128},
        "material_library": "materials.yaml",
        "geometry": {"frame": "frame.obj",
                     "particles": [{"file": "fluid.obj", "material": "box_fluid"},
                                   {"file": "wall.obj", "material": "box_wall"}]},
        "scalars": scalars,
    }
    (directory / "case.yaml").write_text(yaml.safe_dump(case, sort_keys=False), encoding="utf-8")
    materials = {"schema_version": 1,
                 "box_fluid": {"kind": "fluid", "rest_density": 1000.0, "viscosity": 1.0e-6},
                 "box_wall": {"kind": "boundary", "rest_density": 1000.0, "viscosity": 1.0e-6}}
    (directory / "materials.yaml").write_text(yaml.safe_dump(materials, sort_keys=False), encoding="utf-8")
    return directory / "case.yaml", fluid.shape[0], wall.shape[0]


def initial_field(sim, function):
    """Evaluate function(positions (N,3)) -> (N, n_fields) on every slot, zero on
    non-fluid slots, and upload it (before bootstrap)."""
    positions = sim.readback_positions()[:, :3].astype(np.float64)
    material = sim.readback_material()
    fluid = np.isin(material, np.asarray(sim.fluid_group_ids(), dtype=np.uint32))
    fluid[0] = False
    values = np.zeros((positions.shape[0], len(sim.case.scalars.fields)))
    values[fluid] = function(positions[fluid])
    sim.write_scalars(values)


def run_steps(sim, n):
    for _ in range(n):
        sim.step()


# ----------------------------------------------------------------------------
# T1 - T3: Gaussian diffusion, conservation, non-diffusing field
# ----------------------------------------------------------------------------

def test_diffusion(ctx, work, out):
    half, dx, hdx, c0 = 0.05, 0.002, 3.0, 1.0
    sigma0, D = 0.008, 3.0e-5
    scalars = {"fields": [{"name": "gauss_D", "diffusivity": D, "turbulent": False},
                          {"name": "gauss_static", "diffusivity": 0.0, "turbulent": False}]}
    case_path, n_fluid, n_wall = build_box(work / "t1_diffusion", half_length=half, dx=dx, hdx=hdx,
                                           speed_of_sound=c0, scalars=scalars, defrag_cadence=10)
    case = load_case(case_path)
    dt = case.timestep
    sample_times = [0.0, 0.5, 1.0, 1.5, 2.0]
    sample_steps = [int(round(t / dt)) for t in sample_times]
    gauss = lambda x: np.exp(-(x * x).sum(axis=1) / (2 * sigma0 ** 2))
    rows, profiles = [], []
    with SphSimulatorV1(ctx, case) as sim:
        initial_field(sim, lambda x: np.column_stack([gauss(x), gauss(x)]))
        sim.bootstrap()
        reference = None
        for target in sample_steps:
            run_steps(sim, target - sim.step_count)
            snap = sim.scalar_snapshot()
            x, m, C, uid = snap["positions"], snap["mass"], snap["scalars"], snap["uid"]
            t = sim.simulation_time
            M0 = (m * C[:, 0]).sum()
            variance = (m * C[:, 0] * (x * x).sum(axis=1)).sum() / (3.0 * M0)
            sigma2 = sigma0 ** 2 + 2 * D * t
            analytic = (sigma0 ** 2 / sigma2) ** 1.5 * np.exp(-(x * x).sum(axis=1) / (2 * sigma2))
            order = np.argsort(uid)
            if reference is None:
                reference = {"uid": uid[order], "static": C[order, 1], "totals": sim.scalar_totals(snap),
                             "variance": variance}
            static_change = np.abs(C[order, 1] - reference["static"]).max()
            assert np.array_equal(uid[order], reference["uid"])
            totals = sim.scalar_totals(snap)
            rows.append({
                "t": t, "step": sim.step_count,
                "variance_growth": variance - reference["variance"], "variance_growth_exact": 2 * D * t,
                "peak": float(C[:, 0].max()), "peak_exact": (sigma0 ** 2 / sigma2) ** 1.5,
                "rms_error_over_peak": float(np.sqrt(((C[:, 0] - analytic) ** 2).mean()) / analytic.max()),
                "total_rel_change_D": float(totals[0] / reference["totals"][0] - 1.0),
                "total_rel_change_static": float(totals[1] / reference["totals"][1] - 1.0),
                "static_max_abs_change": float(static_change),
                "centroid": [float(v) for v in (m[:, None] * C[:, :1] * x).sum(axis=0) / M0],
            })
            r = np.sqrt((x * x).sum(axis=1))
            profiles.append((t, r, C[:, 0], sigma2))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for t, r, c, sigma2 in profiles:
        sel = r < 0.045
        axes[0].plot(r[sel] * 1e3, c[sel], ".", ms=1.0, alpha=0.4)
        rr = np.linspace(0, 0.045, 200)
        axes[0].plot(rr * 1e3, (sigma0 ** 2 / sigma2) ** 1.5 * np.exp(-rr ** 2 / (2 * sigma2)), "k-", lw=0.8)
    axes[0].set_xlabel("r (mm)"); axes[0].set_ylabel("C"); axes[0].set_title("T1 Gaussian: SPH (dots) vs analytic (lines)")
    ts = [row["t"] for row in rows]
    axes[1].plot(ts, [row["variance_growth"] * 1e6 for row in rows], "o", label="SPH")
    axes[1].plot(ts, [row["variance_growth_exact"] * 1e6 for row in rows], "k-", label="2 D t")
    axes[1].set_xlabel("t (s)"); axes[1].set_ylabel("sigma^2(t) - sigma^2(0)  (mm^2)"); axes[1].legend()
    axes[1].set_title(f"variance growth, D = {D:g} m^2/s")
    fig.tight_layout(); fig.savefig(out / "t1_diffusion.png", dpi=120); plt.close(fig)
    return {"n_fluid": n_fluid, "n_wall": n_wall, "dt": dt, "D": D, "sigma0": sigma0, "samples": rows}


# ----------------------------------------------------------------------------
# T4: compensated summation
# ----------------------------------------------------------------------------

def test_compensation(ctx, work, out, steps):
    half, dx, hdx, c0 = 0.015, 0.002, 3.0, 1.0
    sigma0, amplitude, D = 0.004, 0.1, 2.0e-9
    results = {}
    variants = (("compensated", True, True), ("plain_float32", False, True),
                ("compensated_no_shift_correction", True, False))
    for label, compensated, shift_correction in variants:
        scalars = {"fields": [{"name": "bump", "diffusivity": D, "turbulent": False}],
                   "compensated_sum": compensated, "shift_correction": shift_correction}
        case_path, n_fluid, _ = build_box(work / f"t4_{label}", half_length=half, dx=dx, hdx=hdx,
                                          speed_of_sound=c0, scalars=scalars)
        case = load_case(case_path)
        with SphSimulatorV1(ctx, case) as sim:
            initial_field(sim, lambda x: (1.0 + amplitude * np.exp(-(x * x).sum(axis=1) / (2 * sigma0 ** 2)))[:, None])
            sim.bootstrap()
            snap0 = sim.scalar_snapshot()
            delta = sim.readback_scalar_delta()
            fluid_delta = np.abs(delta[:, 0]).max()
            start = time.perf_counter()
            run_steps(sim, steps)
            elapsed = time.perf_counter() - start
            snap = sim.scalar_snapshot()
        excess0 = snap0["scalars"][:, 0] - 1.0
        excess = snap["scalars"][:, 0] - 1.0
        x0, x = snap0["positions"], snap["positions"]
        variance0 = (excess0 * (x0 * x0).sum(axis=1)).sum() / (3 * excess0.sum())
        variance = (excess * (x * x).sum(axis=1)).sum() / (3 * excess.sum())
        t = steps * case.timestep
        results[label] = {
            "t": t, "steps": steps, "wall_seconds": elapsed,
            "max_first_increment": float(fluid_delta),
            "half_ulp_at_1": float(np.spacing(np.float32(1.0)) / 2),
            "peak_excess_ratio": float(excess.max() / excess0.max()),
            "variance_growth": float(variance - variance0),
            "variance_growth_exact": 2 * D * t,
            "total_excess_rel_change": float(excess.sum() / excess0.sum() - 1.0),
        }
        sigma2 = sigma0 ** 2 + 2 * D * t
        results[label]["peak_excess_ratio_exact"] = (sigma0 ** 2 / sigma2) ** 1.5
    return results


# ----------------------------------------------------------------------------
# T5: shift correction
# ----------------------------------------------------------------------------

def test_shift_correction(ctx, work, out, steps):
    half, dx, hdx, c0, slope = 0.02, 0.002, 3.0, 1.0, 10.0
    h = hdx * dx
    results = {}
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for label, correction in (("with_correction", True), ("without_correction", False)):
        scalars = {"fields": [{"name": "linear", "diffusivity": 0.0, "turbulent": False}],
                   "shift_correction": correction}
        case_path, n_fluid, _ = build_box(work / f"t5_{label}", half_length=half, dx=dx, hdx=hdx,
                                          speed_of_sound=c0, scalars=scalars, jitter=0.25, seed=7)
        case = load_case(case_path)
        with SphSimulatorV1(ctx, case) as sim:
            initial_field(sim, lambda x: (1.0 + slope * x[:, 0])[:, None])
            sim.bootstrap()
            C0 = sim.readback_scalars()[:, 0].astype(np.float64)
            applied_shift = np.zeros(C0.shape[0])
            # shift written by force at step n is applied by predict at step n + 1
            applied_shift += sim.readback_shift()[:, 0]
            for step in range(steps):
                sim.step()
                if step < steps - 1:
                    applied_shift += sim.readback_shift()[:, 0]
            positions = sim.readback_positions()
            live = sim.live_slot_mask(positions)
            material = sim.readback_material()
            C = sim.readback_scalars()[:, 0].astype(np.float64) - sim.readback_scalar_compensation()[:, 0]
            fluid = live & np.isin(material, np.asarray(sim.fluid_group_ids(), dtype=np.uint32))
        interior = fluid & (np.abs(positions[:, :3]) < half - h).all(axis=1)
        expected_change = slope * applied_shift[interior]
        residual = C[interior] - C0[interior] - expected_change
        results[label] = {
            "steps": steps, "interior_particles": int(interior.sum()),
            "rms_shift_x_over_dx": float(np.sqrt((applied_shift[interior] ** 2).mean()) / dx),
            "rms_expected_change": float(np.sqrt((expected_change ** 2).mean())),
            "rms_residual": float(np.sqrt((residual ** 2).mean())),
            "max_abs_residual": float(np.abs(residual).max()),
        }
        results[label]["residual_over_expected"] = results[label]["rms_residual"] / results[label]["rms_expected_change"]
        ax.plot(expected_change, C[interior] - C0[interior], ".", ms=2, label=label)
    lim = np.abs(ax.get_xlim()).max()
    ax.plot([-lim, lim], [-lim, lim], "k-", lw=0.8, label="C - C0 = 10 * sum(shift_x)")
    ax.set_xlabel("10 * accumulated shift_x"); ax.set_ylabel("C(t) - C(0)"); ax.legend(fontsize=8)
    ax.set_title("T5 shift correction, jittered lattice, D = 0")
    fig.tight_layout(); fig.savefig(out / "t5_shift_correction.png", dpi=120); plt.close(fig)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="output/scalar_checks")
    parser.add_argument("--only", default="T1,T4,T5")
    parser.add_argument("--t4-steps", type=int, default=100000)
    parser.add_argument("--t5-steps", type=int, default=300)
    args = parser.parse_args()
    out = (ROOT / args.out).resolve()
    work = out / "cases"
    out.mkdir(parents=True, exist_ok=True)
    only = {item.strip().upper() for item in args.only.split(",")}
    compile_shaders_v1.compile_v1_shaders()
    summary = {}
    with VulkanContext.create(application_name="scalar_checks", enable_validation=False) as ctx:
        if "T1" in only:
            summary["T1_T2_T3_diffusion"] = test_diffusion(ctx, work, out)
            print(json.dumps(summary["T1_T2_T3_diffusion"], indent=1))
        if "T4" in only:
            summary["T4_compensation"] = test_compensation(ctx, work, out, args.t4_steps)
            print(json.dumps(summary["T4_compensation"], indent=1))
        if "T5" in only:
            summary["T5_shift_correction"] = test_shift_correction(ctx, work, out, args.t5_steps)
            print(json.dumps(summary["T5_shift_correction"], indent=1))
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print("wrote", out / "summary.json")


if __name__ == "__main__":
    main()
