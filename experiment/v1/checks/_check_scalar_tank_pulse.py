"""_check_scalar_tank_pulse.py — tracer pulse in the stirred tank (2026-09-27).

End-to-end check of the scalar transport in the real flow: injection, probes,
Smagorinsky SGS, defrag every 10 steps, rotor. Runs a tank case generated with
    python utils/geometry/_demo_stirred_tank_30l.py --dx 0.004 --c0-factor 20 \\
        --tracers 2 --injection-start 0.3 --injection-interval 0.3 \\
        --injection-duration 0.2 --sgs --out <BASE> --no-preview
in three variants of the numerical toggles and records, every --sample
steps, the conserved total sum_i m_i C_i and the range [min, max] of every
tracer over the FLUID particles:

  no_shift_correction (default)  the configuration used unless a case opts in
  shift_correction + limiter     Taylor shift correction, bounds limiter on
  shift_correction, no limiter   Taylor shift correction, bounds limiter off

Expected: without the shift correction the total is constant after each
pulse (the diffusion is conservative) and the values stay in [0, 1]; with it
the total drifts (the Taylor term is not conservative) and, without the
limiter, the values overshoot at the pulse edge.

Usage (repo root):
    python experiment/v1/checks/_check_scalar_tank_pulse.py BASE_CASE_DIR [--steps 12000] [--sample 100]
        [--out output/scalar_checks]
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

VARIANTS = (("no_shift_correction (default)", False, True),
            ("shift_correction + limiter", True, True),
            ("shift_correction, no limiter", True, False))


def make_variant(base: pathlib.Path, target: pathlib.Path, shift_correction: bool, limiter: bool) -> pathlib.Path:
    target.mkdir(parents=True, exist_ok=True)
    for name in ("fluid.obj", "wall.obj", "rotor.obj", "frame.obj", "materials.yaml"):
        shutil.copy(base / name, target / name)
    data = yaml.safe_load((base / "case.yaml").read_text(encoding="utf-8"))
    data["scalars"]["shift_correction"] = shift_correction
    data["scalars"]["bounds_limiter"] = limiter
    (target / "case.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return target / "case.yaml"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("base")
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--sample", type=int, default=100)
    parser.add_argument("--out", default="output/scalar_checks")
    args = parser.parse_args()
    base = pathlib.Path(args.base).resolve()
    out = (ROOT / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    compile_shaders_v1.compile_v1_shaders()
    results = {}
    with VulkanContext.create(application_name="scalar_tank_pulse", enable_validation=False) as ctx:
        for label, shift_correction, limiter in VARIANTS:
            directory = "t7_" + label.split(" (")[0].replace(" + ", "_").replace(", ", "_").replace(" ", "_")
            case = load_case(make_variant(base, out / "cases" / directory, shift_correction, limiter))
            names = case.scalars.field_names
            series = {"t": [], "total": [], "min": [], "max": []}
            with SphSimulatorV1(ctx, case) as sim:
                sim.bootstrap()
                while sim.step_count < args.steps:
                    sim.step()
                    if sim.step_count % args.sample == 0:
                        snap = sim.scalar_snapshot()
                        values = snap["scalars"]
                        series["t"].append(sim.simulation_time)
                        series["total"].append(sim.scalar_totals(snap).tolist())
                        series["min"].append(values.min(axis=0).tolist())
                        series["max"].append(values.max(axis=0).tolist())
                status = sim.readback_global_status()
            t = np.array(series["t"]); total = np.array(series["total"])
            summary = {}
            for index, (name, injection) in enumerate(zip(names, case.scalars.injections)):
                end = injection.start + injection.duration
                after = t > end + 2 * args.sample * case.timestep
                if after.sum() >= 2:
                    drift = (total[after, index][-1] - total[after, index][0]) / total[after, index][0]
                    span = t[after][-1] - t[after][0]
                else:
                    drift, span = float("nan"), 0.0
                summary[name] = {
                    "total_after_pulse_rel_change": float(drift), "over_seconds": float(span),
                    "min": float(np.min(np.array(series["min"])[:, index])),
                    "max": float(np.max(np.array(series["max"])[:, index])),
                }
            results[label] = {"shift_correction": shift_correction, "bounds_limiter": limiter,
                              "steps": sim.step_count, "t_end": sim.simulation_time,
                              "overflow": [status["overflow_inside_count"], status["overflow_incoming_count"],
                                           status["correction_fallback_count"]],
                              "fields": summary, "series": series}
            print(label, json.dumps(summary, indent=1))
    (out / "t7_tank_pulse.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for label, result in results.items():
        t = np.array(result["series"]["t"]); total = np.array(result["series"]["total"])
        axes[0].plot(t, total[:, 0] / total[-1, 0], label=label)
        axes[1].plot(t, np.array(result["series"]["max"])[:, 0], label=f"{label} max")
        axes[1].plot(t, np.array(result["series"]["min"])[:, 0], "--", label=f"{label} min")
    axes[0].set_xlabel("t (s)"); axes[0].set_ylabel("total / final total (tracer_01)"); axes[0].legend(fontsize=8)
    axes[0].set_ylim(0.97, 1.01)
    axes[1].set_xlabel("t (s)"); axes[1].set_ylabel("tracer_01 range over fluid"); axes[1].legend(fontsize=7)
    fig.suptitle("T7 tracer pulse in the 4 mm tank (pulse 0.3-0.5 s)")
    fig.tight_layout(); fig.savefig(out / "t7_tank_pulse.png", dpi=120)
    print("saved", out / "t7_tank_pulse.png")


if __name__ == "__main__":
    main()
