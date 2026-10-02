"""
_analyze_flow_statistics.py — H1 of the Haringa (2023) reproduction (2026-10-02): impeller
discharge profiles, pumping number, mean flow pattern and power number of the 1-impeller tank,
from the accumulators of experiment/v1/utils/flow_statistics.py and the torque log.

Discharge (Haringa 2023 Figure 1): on the lines y = C midway between the baffles, against
xi = (r - r_tip) / (R - r_tip), averaged over the four lines (their spread is printed):
    U_rad / U_tip                 time-averaged radial velocity
    k / U_tip^2                   resolved turbulent kinetic energy 1/2 <u'_i u'_i>, total, and
                                  without the periodic blade-passage part (phase bins):
                                  k_random = k - 1/2 sum_b w_b |<u>_b - <u>|^2
    epsilon / (N^3 D^2)           2 nu <S:S> (resolved, molecular viscosity) and the Smagorinsky
                                  form (C_s Delta)^2 <|S|^3> (what an LES with C_s reports as SGS
                                  dissipation; Delta = --delta, default the particle spacing)
Pumping number: Fl = Q / (N D^3), Q the outflow sum <u_r>^+ dA over the cylinder just outside the
blade tips (r_tip + dx) between C - W and C + W (W the blade height); the net flow is printed too.
Power number from the torque log: Po = 2 pi N tau / (rho N^3 D^5), tau the readback divided by the
mass factor (1.2187 for h/dx = 3).

usage (repo root):
    python experiment/v1/checks/_analyze_flow_statistics.py STATS.npz --tank CASE/tank.json
        [--torque TORQUE.csv --torque-from 20] [--out-dir DIR] [--label NAME] [--cs 0.1] [--delta D]
"""
import argparse
import json
import math
import pathlib

import numpy as np

# Haringa (2023) Table 3, 1-impeller tank, torque power number of the LB-LES cases, and the
# experimental range for Rushton turbines the paper quotes.
REFERENCE_POWER = (("LB-LES NX180 2D sheets", 3.94), ("LB-LES NX360 2D sheets", 4.14),
                   ("LB-LES range (all 10 cases)", "3.69 .. 4.44"), ("experiments (Rushton)", "4.6 .. 6.0"))


def moments(stats, name):
    samples = stats[name + "/samples"]
    mean = stats[name + "/sum_velocity"] / samples[:, None]
    product = stats[name + "/sum_product"] / samples[:, None]
    variance = product[:, :3] - mean ** 2
    return samples, mean, variance


def kinetic_energy_random(stats, name, mean):
    """k without the periodic part: subtract 1/2 sum_b w_b |<u>_b - <u>|^2."""
    if name + "/phase_samples" not in stats:
        return None
    counts = stats[name + "/phase_samples"]
    weight = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1.0)
    phase_mean = stats[name + "/phase_velocity"] / np.maximum(counts, 1.0)[:, :, None]
    periodic = 0.5 * np.sum(weight[:, :, None] * (phase_mean - mean[:, None, :]) ** 2, axis=(1, 2))
    return periodic


def strain_measures(stats, name, viscosity):
    samples = stats[name + "/gradient_samples"]
    squared = stats[name + "/sum_strain_squared"] / samples
    cubed = stats[name + "/sum_strain_cubed"] / samples
    mean_strain = stats[name + "/sum_strain"] / samples[:, None]
    mean_squared = np.sum(mean_strain[:, :3] ** 2, axis=1) + 2.0 * np.sum(mean_strain[:, 3:] ** 2, axis=1)
    return dict(resolved=2.0 * viscosity * squared, resolved_fluctuating=2.0 * viscosity * (squared - mean_squared),
                strain_cubed=cubed, strain_squared=squared)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("stats")
    parser.add_argument("--tank", required=True, help="tank.json of the case")
    parser.add_argument("--torque", default=None, help="torque CSV (_run_v1_headless.py --torque-log)")
    parser.add_argument("--torque-from", type=float, default=20.0, help="average the torque from this time (s)")
    parser.add_argument("--mass-factor", type=float, default=1.2187)
    parser.add_argument("--cs", type=float, default=0.1, help="Smagorinsky constant of the epsilon estimate")
    parser.add_argument("--delta", type=float, default=None, help="filter width of the estimate (default dx)")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--label", default=None)
    args = parser.parse_args()

    stats = dict(np.load(args.stats))
    meta = json.loads(str(stats.pop("meta")))
    tank = json.loads(pathlib.Path(args.tank).read_text(encoding="utf-8"))
    impeller = tank["impellers"][0]
    diameter, speed = impeller["diameter"], tank["rpm"] / 60.0
    tip_speed, radius, tip = tank["tip_speed"], tank["tank_radius"], impeller["tip_radius"]
    dx = tank["dx"]
    delta = args.delta if args.delta is not None else dx
    viscosity = meta["viscosity"]
    epsilon_scale = speed ** 3 * diameter ** 2
    label = args.label or pathlib.Path(args.stats).stem
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else pathlib.Path(args.stats).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{label}: {meta['samples']} samples, t = {meta['first_time']:.2f} .. {meta['last_time']:.2f} s, "
          f"kernel radius {meta['kernel_radius'] * 1e3:.2f} mm, dx {dx * 1e3:g} mm; U_tip {tip_speed:.4f} m/s, "
          f"N {speed:g} 1/s, D {diameter:.4f} m")

    # --- discharge profiles ------------------------------------------------------------------
    lines = [entry for entry in meta["sets"] if entry["name"].startswith("discharge_")]
    coordinate = np.asarray(lines[0]["coordinate"])
    per_line = dict(radial=[], tangential=[], axial=[], k=[], k_random=[], epsilon_resolved=[], epsilon_sgs=[],
                    epsilon_fluctuating=[])
    for entry in lines:
        name = entry["name"]
        _, mean, variance = moments(stats, name)
        k = 0.5 * variance.sum(axis=1)
        periodic = kinetic_energy_random(stats, name, mean)
        strain = strain_measures(stats, name, viscosity)
        per_line["radial"].append(mean[:, 0] / tip_speed)
        per_line["tangential"].append(mean[:, 1] / tip_speed)
        per_line["axial"].append(mean[:, 2] / tip_speed)
        per_line["k"].append(k / tip_speed ** 2)
        per_line["k_random"].append((k - periodic) / tip_speed ** 2 if periodic is not None else k * np.nan)
        per_line["epsilon_resolved"].append(strain["resolved"] / epsilon_scale)
        per_line["epsilon_fluctuating"].append(strain["resolved_fluctuating"] / epsilon_scale)
        per_line["epsilon_sgs"].append((args.cs * delta) ** 2 * strain["strain_cubed"] / epsilon_scale)
    profile = {key: np.nanmean(np.asarray(value), axis=0) for key, value in per_line.items()}
    spread = {key: np.nanstd(np.asarray(value), axis=0) for key, value in per_line.items()}
    print("  discharge at y = C between the baffles, mean of the four lines (std between lines):")
    print("     xi   U_rad/U_tip   U_tan/U_tip   k/U_tip^2   k_rand/U_tip^2   eps_SGS/N3D2   2nuSS/N3D2")
    for index in range(0, coordinate.size, 4):
        print(f"   {coordinate[index]:5.3f}   {profile['radial'][index]:6.3f} ({spread['radial'][index]:5.3f})"
              f"   {profile['tangential'][index]:6.3f}   {profile['k'][index]:8.4f}   {profile['k_random'][index]:8.4f}"
              f"        {profile['epsilon_sgs'][index]:8.3f}     {profile['epsilon_resolved'][index]:8.4f}")

    # --- pumping number --------------------------------------------------------------------------
    cylinder = next(entry for entry in meta["sets"] if entry["name"] == "tip_cylinder")
    _, mean, _ = moments(stats, "tip_cylinder")
    heights = np.asarray(cylinder["heights"])
    azimuths = np.asarray(cylinder["azimuths_deg"])
    radial = mean[:, 0].reshape(heights.size, azimuths.size)
    area = cylinder["radius"] * math.radians(azimuths[1] - azimuths[0]) * (heights[1] - heights[0])
    outflow = float(np.nansum(np.clip(radial, 0.0, None)) * area)
    net = float(np.nansum(radial) * area)
    flow_number = outflow / (speed * diameter ** 3)
    print(f"  pumping: r = {cylinder['radius'] * 1e3:.1f} mm, outflow Q = {outflow * 1e3:.3f} L/s, Fl = {flow_number:.3f} "
          f"(net {net * 1e3:.3f} L/s); max <u_r>/U_tip {np.nanmax(radial) / tip_speed:.3f} at "
          f"y - C = {(heights[np.nanargmax(np.nanmax(radial, axis=1))] - impeller['center_height']) * 1e3:+.1f} mm")

    # --- power number ------------------------------------------------------------------------------
    power = {}
    if args.torque:
        torque = np.genfromtxt(args.torque, delimiter=",", names=True)
        window = torque["time"] >= args.torque_from
        if window.sum() > 2:
            value = np.abs(torque["torque_axis"][window]) / args.mass_factor
            power_number = value * 2.0 * math.pi * speed / (1000.0 * speed ** 3 * diameter ** 5)
            power = dict(window_start=args.torque_from, torque=float(value.mean()), power_number=float(power_number.mean()),
                         power_number_std=float(power_number.std(ddof=1)), samples=int(window.sum()))
            halves = np.array_split(power_number, 2)
            print(f"  power number (t >= {args.torque_from:g} s): Po = {power['power_number']:.3f} +- "
                  f"{power['power_number_std']:.3f} (torque {power['torque'] * 1e3:.1f} mN m, {power['samples']} samples; "
                  f"halves {halves[0].mean():.3f} / {halves[1].mean():.3f})")
    for name, value in REFERENCE_POWER:
        print(f"    reference Po {name}: {value}")

    # --- outputs -------------------------------------------------------------------------------------
    table = np.column_stack([coordinate] + [profile[key] for key in ("radial", "tangential", "axial", "k", "k_random",
                                                                      "epsilon_sgs", "epsilon_resolved")])
    np.savetxt(out_dir / f"{label}_discharge.csv", table, delimiter=",", fmt="%.6e",
               header="xi,U_rad/U_tip,U_tan/U_tip,U_ax/U_tip,k/U_tip^2,k_random/U_tip^2,eps_sgs/N3D2,eps_2nuSS/N3D2",
               comments="")
    summary = dict(label=label, samples=meta["samples"], first_time=meta["first_time"], last_time=meta["last_time"],
                   flow_number=flow_number, outflow=outflow, net_flow=net, power=power, delta=delta, cs=args.cs,
                   peak_radial=float(np.nanmax(profile["radial"])),
                   peak_k=float(np.nanmax(profile["k"])), peak_epsilon_sgs=float(np.nanmax(profile["epsilon_sgs"])))
    (out_dir / f"{label}_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    plot(out_dir, label, coordinate, profile, spread, stats, meta, tank)
    print(f"  wrote {out_dir / (label + '_discharge.csv')}, _summary.json and figures")


def plot(out_dir, label, coordinate, profile, spread, stats, meta, tank):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    for axis, keys, title in ((axes[0], (("radial", "U_rad"), ("tangential", "U_tan"), ("axial", "U_ax")), "mean velocity / U_tip"),
                              (axes[1], (("k", "k total"), ("k_random", "k without blade passage")), "k / U_tip^2"),
                              (axes[2], (("epsilon_sgs", "(C_s dx)^2 <|S|^3>"), ("epsilon_resolved", "2 nu <S:S>")),
                               "epsilon / (N^3 D^2)")):
        for key, name in keys:
            axis.plot(coordinate, profile[key], label=name)
            axis.fill_between(coordinate, profile[key] - spread[key], profile[key] + spread[key], alpha=0.2)
        axis.set_xlabel("(r - r_tip) / (R - r_tip)")
        axis.set_title(title)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
    figure.suptitle(f"{label}: discharge at y = C between the baffles, t = {meta['first_time']:.1f} .. {meta['last_time']:.1f} s")
    figure.tight_layout()
    figure.savefig(out_dir / f"{label}_discharge.png", dpi=130)
    plt.close(figure)

    planes = [entry for entry in meta["sets"] if entry["name"].startswith("plane_")]
    if not planes:
        return
    figure, axes = plt.subplots(1, len(planes), figsize=(5.5 * len(planes), 8))
    axes = np.atleast_1d(axes)
    for axis, entry in zip(axes, planes):
        _, mean, variance = moments(stats, entry["name"])
        r_values, y_values = np.asarray(entry["r_values"]), np.asarray(entry["y_values"])
        shape = (r_values.size, y_values.size)
        u_r = mean[:, 0].reshape(shape)
        u_y = mean[:, 2].reshape(shape)
        k = 0.5 * variance.sum(axis=1).reshape(shape)
        image = axis.pcolormesh(r_values, y_values, (k / tank["tip_speed"] ** 2).T, shading="auto", cmap="viridis")
        axis.streamplot(r_values, y_values, u_r.T, u_y.T, color="w", density=1.4, linewidth=0.7)
        axis.set_aspect("equal")
        axis.set_xlabel("r [m]"); axis.set_ylabel("y [m]")
        axis.set_title(f"{entry['name']}: mean (u_r, u_y), colour k / U_tip^2")
        figure.colorbar(image, ax=axis, shrink=0.6)
    figure.tight_layout()
    figure.savefig(out_dir / f"{label}_planes.png", dpi=130)
    plt.close(figure)


if __name__ == "__main__":
    main()
