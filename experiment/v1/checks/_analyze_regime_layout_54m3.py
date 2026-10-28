"""_analyze_regime_layout_54m3.py — where the regimes sit in the 54 m3 haringa54 tank and how fast the flow
exchanges liquid between the impeller compartments, from lifelines with positions (2026-10-29).

Written to locate the difference to Haringa (2023, LB-LES): starvation 48.5 % there, 41.0 % in H54 with the same
sink. Haringa attributes his RANS / LB-LES difference to large-scale axial transport: in LB-LES the top-impeller
discharge stream is on average in limitation, in RANS in starvation (his Fig. 3, Table 5).

1. Regime of every record (Haringa's definition: q = C_s / (C_s + K_s), 0.36 s centred moving average, S < 0.05,
   E > 0.95, hysteresis 0.01), counted by where the sample is: height slabs, inner / outer radius, and zones
   (top / bottom impeller discharge, feed region, floor).
2. Time-averaged field map as in Haringa's Fig. 3: C_s averaged over the window per azimuthally averaged (r, y) cell,
   regime of q(mean C_s); volume fractions of the three regimes. Saved as NPZ for plotting.
3. Exchange through horizontal planes (default y = 2.4 m between the impellers and 5.9 m above the top impeller):
   crossings with a hysteresis band of +-MARGIN, the exchange flow Q = (crossings per direction) / span * V_sample,
   V_sample = fluid volume / lifelines; and the mean residence per visit below the plane (complete visits only).

usage (repo root):
    python experiment/v1/checks/_analyze_regime_layout_54m3.py DIR --k-s 7.8e-6 [--start 120] [--planes 2.4 5.9]
        [--margin 0.1] [--out PREFIX]
"""
import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from experiment.v1.utils.lifeline_recorder import load_lifelines   # noqa: E402
from _analyze_lifelines import moving_average, regime_states, two_state, segments   # noqa: E402

LIQUID_HEIGHT = 7.7
TANK_RADIUS = 1.5
IMPELLER_RADIUS = 0.65
IMPELLER_HEIGHTS = {"bottom": 0.9, "top": 3.9}
NAMES = ("S", "L", "E")


def fractions(states, select):
    chosen = states[select]
    if chosen.size == 0:
        return (np.nan, np.nan, np.nan), 0
    return tuple(100.0 * (chosen == value).mean() for value in (0, 1, 2)), chosen.size


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dir")
    parser.add_argument("--k-s", type=float, required=True, help="mol/kg, of the run's sink")
    parser.add_argument("--start", type=float, default=120.0, help="s")
    parser.add_argument("--window", type=float, default=0.36, help="moving average, s")
    parser.add_argument("--planes", type=float, nargs="*", default=[2.4, 5.9], help="exchange planes, m")
    parser.add_argument("--margin", type=float, default=0.1, help="hysteresis half-band of a plane crossing, m")
    parser.add_argument("--cell", type=float, default=0.1, help="(r, y) cell of the time-averaged map, m")
    parser.add_argument("--out", default=None, help="prefix of the NPZ written")
    arguments = parser.parse_args()

    data = load_lifelines(arguments.dir, with_aux=False)
    meta = data["meta"]
    time = data["time"].astype(np.float64)
    names = data["field_names"]
    concentration = data["scalars"][:, :, names.index("substrate")].astype(np.float64)
    position = data["position"].astype(np.float64)
    interval = float(np.median(np.diff(time)))
    q = concentration / (concentration + arguments.k_s)
    states = regime_states(moving_average(q, max(1, round(arguments.window / interval))))
    keep = time >= arguments.start
    time, concentration, position, states = time[keep], concentration[keep], position[keep], states[keep]
    height = position[..., 1]
    radius = np.hypot(position[..., 0], position[..., 2])
    count = concentration.shape[1]
    span = float(time[-1] - time[0])
    fluid_volume = float(meta["fluid_candidates"]) * float(meta["particle_spacing_m"]) ** 3
    sample_volume = fluid_volume / count
    print(f"{arguments.dir}: {count:,} lifelines, {time.size} records, {time[0]:.1f}..{time[-1]:.1f} s; K_s {arguments.k_s:g}; "
          f"fluid volume {fluid_volume:.2f} m3, {sample_volume * 1e3:.2f} L per lifeline")
    total, _ = fractions(states, np.ones(states.shape, bool))
    print(f"all samples: S / L / E {total[0]:.1f} / {total[1]:.1f} / {total[2]:.1f} %")

    print("\n1. regime of the samples by position (S / L / E %, share of the samples)")
    edges = np.linspace(0.0, LIQUID_HEIGHT, 12)
    print("   height slab, m        all r                   r < 0.65 m              r >= 0.65 m")
    for low, high in zip(edges[:-1][::-1], edges[1:][::-1]):
        slab = (height >= low) & (height < high)
        line = f"   {low:4.2f}..{high:4.2f}"
        for select in (slab, slab & (radius < IMPELLER_RADIUS), slab & (radius >= IMPELLER_RADIUS)):
            value, size = fractions(states, select)
            line += f"   {value[0]:5.1f} / {value[1]:5.1f} / {value[2]:5.1f} ({100 * size / states.size:4.1f})"
        print(line)
    zones = {
        "top impeller discharge (|y-3.9|<0.3, r>0.65)": (np.abs(height - IMPELLER_HEIGHTS["top"]) < 0.3) & (radius > IMPELLER_RADIUS),
        "top impeller swept (|y-3.9|<0.3, r<0.65)": (np.abs(height - IMPELLER_HEIGHTS["top"]) < 0.3) & (radius <= IMPELLER_RADIUS),
        "bottom impeller discharge (|y-0.9|<0.3, r>0.65)": (np.abs(height - IMPELLER_HEIGHTS["bottom"]) < 0.3) & (radius > IMPELLER_RADIUS),
        "between impellers (1.5 < y < 3.3)": (height > 1.5) & (height < 3.3),
        "upper loop (3.3 < y < 5.9)": (height > 3.3) & (height < 5.9),
        "top region (y > 5.9)": height > 5.9,
        "below the bottom impeller (y < 0.6)": height < 0.6,
    }
    print("   zones:")
    for label, select in zones.items():
        value, size = fractions(states, select)
        print(f"   {label:50s} {value[0]:5.1f} / {value[1]:5.1f} / {value[2]:5.1f} %  ({100 * size / states.size:4.1f} % of the samples)")

    print(f"\n2. time-averaged field (azimuthally averaged {arguments.cell:g} m (r, y) cells), regime of q(<C_s>):")
    radius_bins = np.arange(0.0, TANK_RADIUS + arguments.cell, arguments.cell)
    height_bins = np.arange(0.0, LIQUID_HEIGHT + arguments.cell, arguments.cell)
    flat_r, flat_y, flat_c = radius.ravel(), height.ravel(), concentration.ravel()
    sums, _, _ = np.histogram2d(flat_r, flat_y, bins=(radius_bins, height_bins), weights=flat_c)
    counts, _, _ = np.histogram2d(flat_r, flat_y, bins=(radius_bins, height_bins))
    mean_field = np.where(counts >= 20, sums / np.maximum(counts, 1), np.nan)
    q_mean = mean_field / (mean_field + arguments.k_s)
    ring_volume = np.pi * (radius_bins[1:] ** 2 - radius_bins[:-1] ** 2)[:, None] * np.diff(height_bins)[None, :]
    valid = np.isfinite(q_mean)
    weights = np.where(valid, ring_volume, 0.0)
    map_fractions = [100.0 * weights[valid & condition].sum() / weights.sum()
                     for condition in (q_mean < 0.05, (q_mean >= 0.05) & (q_mean <= 0.95), q_mean > 0.95)]
    print(f"   volume fractions S / L / E {map_fractions[0]:.1f} / {map_fractions[1]:.1f} / {map_fractions[2]:.1f} % "
          f"({valid.sum()} of {valid.size} cells with >= 20 samples)")
    coarse_r = [0.0, 0.3, 0.65, 1.0, 1.5]
    coarse_y = np.arange(0.0, LIQUID_HEIGHT + 0.01, 0.7)
    print("   coarse map of the regime of q(<C_s>) (rows: height from the top, columns: r bands 0-0.3, 0.3-0.65, 0.65-1.0, 1.0-1.5 m):")
    for low, high in zip(coarse_y[:-1][::-1], coarse_y[1:][::-1]):
        row = f"   {low:4.1f}..{high:4.1f} m:"
        for r_low, r_high in zip(coarse_r[:-1], coarse_r[1:]):
            select = (flat_r >= r_low) & (flat_r < r_high) & (flat_y >= low) & (flat_y < high)
            value = flat_c[select].mean() if select.any() else np.nan
            ratio = value / (value + arguments.k_s)
            row += f"  {NAMES[0 if ratio < 0.05 else 2 if ratio > 0.95 else 1]} {ratio:5.3f}"
        print(row)

    print(f"\n3. exchange through horizontal planes (hysteresis +-{arguments.margin:g} m, complete visits only)")
    for plane in arguments.planes:
        above = two_state(height > plane + arguments.margin, height < plane - arguments.margin, height[0] > plane)
        changes = np.diff(above.astype(np.int8), axis=0)
        upward, downward = int((changes == 1).sum()), int((changes == -1).sum())
        flow = 0.5 * (upward + downward) / span * sample_volume
        below_fraction = 100.0 * (~above).mean()
        run = segments(above.astype(np.int8), time)
        complete = ~run["truncated"]
        stay_below = (run["end"] - run["start"])[complete & (run["state"] == 0)]
        stay_above = (run["end"] - run["start"])[complete & (run["state"] == 1)]
        print(f"   y = {plane:.2f} m: up {upward:,} / down {downward:,} crossings, exchange flow {flow:.3f} m3/s; "
              f"{below_fraction:.1f} % of the samples below; visit below mean {stay_below.mean():.2f} s (median "
              f"{np.median(stay_below):.2f}, n {stay_below.size:,}), above mean {stay_above.mean():.2f} s; "
              f"V_below / Q = {below_fraction / 100 * fluid_volume / max(flow, 1e-12):.1f} s")

    if arguments.out:
        np.savez_compressed(f"{arguments.out}.npz", radius_edges=radius_bins, height_edges=height_bins,
                            mean_substrate=mean_field, samples=counts, k_s=arguments.k_s)
        print(f"\nwrote {arguments.out}.npz")
    return 0


if __name__ == "__main__":
    sys.exit(main())
