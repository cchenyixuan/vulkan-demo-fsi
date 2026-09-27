"""_check_shift_dispersion.py — how far does the δ-plus shift move particles
relative to the fluid? (2026-09-27)

Without the scalar shift correction a particle keeps its scalar value while the
shift displaces it relative to the material flow. This script measures that
displacement in a running case: it follows a random sample of FLUID particles
(by persistent uid, so defrag does not matter) over --window steps, sums the
shift vectors that predict applies to each of them, and reports the mean
squared accumulated shift MSD(L) for lags L. For a random-walk-like process
MSD(L) = 6 D_shift L dt (3D), which gives an effective "shift diffusivity" to
compare with the molecular / SGS diffusivities; a linear growth of sqrt(MSD)
with L instead indicates a persistent drift.

Usage (repo root):
    python experiment/v1/checks/_check_shift_dispersion.py CASE.yaml [--spinup 3000] [--window 400]
        [--sample 20000] [--out output/scalar_checks]
"""
import argparse
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.sph.case import load_case                                 # noqa: E402
from utils.sph.vulkan_context import VulkanContext                   # noqa: E402
from experiment.v1 import compile_shaders_v1                         # noqa: E402
from experiment.v1.utils.simulator_v1 import SphSimulatorV1          # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("case")
    parser.add_argument("--spinup", type=int, default=3000)
    parser.add_argument("--window", type=int, default=400)
    parser.add_argument("--sample", type=int, default=20000)
    parser.add_argument("--out", default="output/scalar_checks")
    args = parser.parse_args()
    compile_shaders_v1.compile_v1_shaders()
    case = load_case(args.case)
    dt = case.timestep
    rng = np.random.default_rng(1)
    with VulkanContext.create(application_name="shift_dispersion", enable_validation=False) as ctx:
        with SphSimulatorV1(ctx, case) as sim:
            sim.bootstrap()
            for _ in range(args.spinup):
                sim.step()
            uid = sim.readback_particle_uid()
            positions = sim.readback_positions()
            live = sim.live_slot_mask(positions)
            material = sim.readback_material()
            fluid = live & np.isin(material, np.asarray(sim.fluid_group_ids(), dtype=np.uint32))
            chosen = rng.choice(uid[fluid], size=min(args.sample, int(fluid.sum())), replace=False)
            chosen.sort()
            shifts = np.zeros((args.window, chosen.size, 3))
            alive_at_start = int(sim.readback_global_status()["alive_particle_count"])
            for k in range(args.window):
                # shift written by the last force pass = the one the next predict applies.
                # No particle dies in a closed tank, so after any defrag the live slots
                # are exactly [1, alive]; checked at the end.
                uid = sim.readback_particle_uid()
                shift = sim.readback_shift()[:, :3].astype(np.float64)
                slot_of_uid = np.full(int(uid.max()) + 1, -1, dtype=np.int64)
                slot_of_uid[uid[1:alive_at_start + 1]] = np.arange(1, alive_at_start + 1)
                slots = slot_of_uid[chosen]
                shifts[k] = shift[slots]
                sim.step()
            assert int(sim.readback_global_status()["alive_particle_count"]) == alive_at_start
    accumulated = np.cumsum(shifts, axis=0)
    lags = np.unique(np.geomspace(1, args.window, 16).astype(int))
    rows = []
    for lag in lags:
        msd = (accumulated[lag - 1] ** 2).sum(axis=1).mean()
        rows.append({"lag_steps": int(lag), "lag_seconds": lag * dt, "rms_displacement": float(np.sqrt(msd)),
                     "D_shift_estimate": float(msd / (6 * lag * dt))})
    per_step = np.sqrt((shifts ** 2).sum(axis=2))
    result = {"case": str(args.case), "dt": dt, "spinup_steps": args.spinup, "sampled_particles": int(chosen.size),
              "per_step_shift_rms": float(np.sqrt((per_step ** 2).mean())),
              "per_step_shift_rms_over_dx": float(np.sqrt((per_step ** 2).mean()) / case.physics.particle_diameter),
              "lags": rows}
    out = (ROOT / args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "shift_dispersion.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
