"""
_analyze_flow_statistics.py — H1 of the Haringa (2023) reproduction (2026-10-02): impeller
discharge profiles, pumping number, mean flow pattern and power number of the 1-impeller tank,
from the accumulators of experiment/v1/utils/flow_statistics.py and the torque log; since
2026-10-14 also the 2-impeller tank of Haringa, Vandewijer and Mudde (2018) (preset jahoda2).

Discharge (Haringa 2023 Figure 1), per impeller: on the lines y = C midway between the baffles,
against xi = (r - r_tip) / (R - r_tip), averaged over the four lines (their spread is printed):
    U_rad / U_tip                 time-averaged radial velocity
    k / U_tip^2                   resolved turbulent kinetic energy 1/2 <u'_i u'_i>, total, and
                                  without the periodic blade-passage part (phase bins):
                                  k_random = k - 1/2 sum_b w_b |<u>_b - <u>|^2
    epsilon / (N^3 D^2)           2 nu <S:S> (resolved, molecular viscosity) and the Smagorinsky
                                  form (C_s Delta)^2 <|S|^3> (what an LES with C_s reports as SGS
                                  dissipation; Delta = --delta, default the particle spacing)
Pumping number, per impeller: Fl = Q / (N D^3), Q the outflow sum <u_r>^+ dA over the cylinder just
outside the blade tips (r_tip + dx) between C - W and C + W (W the blade height); the net flow is
printed too.
Power number from the torque log: Po = 2 pi N tau / (rho N^3 D^5), tau the readback divided by the
mass factor (1.2187 for h/dx = 3); with the split columns of --torque-split-height also per impeller.

2-impeller tank (sets of the jahoda2 preset, Haringa et al. 2018 section 3.2, Figures 2 and 3):
    lda_outflow_<impeller>_<az>   U_rad / U_tip, k* / U_tip^2 (total 1/2 <u'_i u'_i>) and k_rand (blade
                                  passage removed) at r = 50 .. 102 mm at the disk height, mean of the
                                  four baffle planes
    midplane_<az>                 the 7 x 11 grid of the mid-compartment plane (y_c = 5T/6): tables of
                                  U_rad / U_tip, U_ax / U_tip, u_rad,rms / U_tip, u_ax,rms / U_tip and
                                  k2* / U_tip^2 = 3/4 (u'_rad^2 + u'_ax^2) / U_tip^2 (the paper's
                                  two-component isotropy estimate; the tangential component is printed
                                  separately), rows r/R, columns y/T, mean of the four baffle planes
    midplane_disk                 axial exchange flow Q_ax = -sum min(<u_y>, 0) dA over the polar grid of
                                  the mid plane (the paper's iso-clip of the downward flow; the upward
                                  sum is printed as a check), Q_ax / (N D^3), and Q_ax over the lower
                                  impeller's pumping Q

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


def impeller_of(entry, tank):
    """The tank.json impeller a set belongs to (`impeller` of the set, else the first)."""
    name = entry.get("impeller")
    if name is None:
        return tank["impellers"][0]
    return next(impeller for impeller in tank["impellers"] if impeller["name"] == name)


def discharge_profiles(stats, lines, tip_speed, epsilon_scale, viscosity, cs, delta):
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
        per_line["epsilon_sgs"].append((cs * delta) ** 2 * strain["strain_cubed"] / epsilon_scale)
    profile = {key: np.nanmean(np.asarray(value), axis=0) for key, value in per_line.items()}
    spread = {key: np.nanstd(np.asarray(value), axis=0) for key, value in per_line.items()}
    return profile, spread


def pumping(stats, cylinder, speed, diameter, tip_speed, center_height):
    _, mean, _ = moments(stats, cylinder["name"])
    heights = np.asarray(cylinder["heights"])
    azimuths = np.asarray(cylinder["azimuths_deg"])
    radial = mean[:, 0].reshape(heights.size, azimuths.size)
    area = cylinder["radius"] * math.radians(azimuths[1] - azimuths[0]) * (heights[1] - heights[0])
    outflow = float(np.nansum(np.clip(radial, 0.0, None)) * area)
    net = float(np.nansum(radial) * area)
    flow_number = outflow / (speed * diameter ** 3)
    peak_row = heights[np.nanargmax(np.nanmax(radial, axis=1))]
    print(f"  pumping ({cylinder['name']}): r = {cylinder['radius'] * 1e3:.1f} mm, outflow Q = {outflow * 1e3:.3f} L/s, "
          f"Fl = {flow_number:.3f} (net {net * 1e3:.3f} L/s); max <u_r>/U_tip {np.nanmax(radial) / tip_speed:.3f} at "
          f"y - C = {(peak_row - center_height) * 1e3:+.1f} mm")
    return dict(flow_number=flow_number, outflow=outflow, net_flow=net)


def power_numbers(args, speed, diameter, tank):
    """Po from the torque log: total, and per impeller from the split columns when present."""
    if not args.torque:
        return {}
    torque = np.genfromtxt(args.torque, delimiter=",", names=True)
    window = torque["time"] >= args.torque_from
    if window.sum() <= 2:
        return {}
    columns = [("total", "torque_axis")]
    if "torque_lower" in torque.dtype.names:
        columns += [("lower", "torque_lower"), ("upper", "torque_upper"), ("shaft", "torque_shaft")]
    power = {}
    for label, column in columns:
        value = np.abs(torque[column][window]) / args.mass_factor
        power_number = value * 2.0 * math.pi * speed / (1000.0 * speed ** 3 * diameter ** 5)
        halves = np.array_split(power_number, 2)
        power[label] = dict(window_start=args.torque_from, torque=float(value.mean()),
                            power_number=float(power_number.mean()), power_number_std=float(power_number.std(ddof=1)),
                            samples=int(window.sum()))
        print(f"  power number {label} (t >= {args.torque_from:g} s): Po = {power[label]['power_number']:.3f} +- "
              f"{power[label]['power_number_std']:.3f} (torque {power[label]['torque'] * 1e3:.1f} mN m, "
              f"{window.sum()} samples; halves {halves[0].mean():.3f} / {halves[1].mean():.3f})")
    if len(tank["impellers"]) > 1 and "total" in power:
        print(f"    (two impellers of the same D: Po per impeller = total / 2 = {power['total']['power_number'] / 2:.3f})")
    return power


def lda_outflow(stats, meta, tank, tip_speed, out_dir, label):
    """U_rad, k* and k_rand at the LDA outflow radii, mean of the baffle planes, per impeller."""
    sets = [entry for entry in meta["sets"] if entry["name"].startswith("lda_outflow_")]
    if not sets:
        return {}
    result = {}
    for impeller in tank["impellers"]:
        own = [entry for entry in sets if entry["impeller"] == impeller["name"]]
        if not own:
            continue
        radii = np.asarray(own[0]["radii"])
        rows = dict(radial=[], axial=[], k=[], k_random=[])
        for entry in own:
            _, mean, variance = moments(stats, entry["name"])
            k = 0.5 * variance.sum(axis=1)
            periodic = kinetic_energy_random(stats, entry["name"], mean)
            rows["radial"].append(mean[:, 0] / tip_speed)
            rows["axial"].append(mean[:, 2] / tip_speed)
            rows["k"].append(k / tip_speed ** 2)
            rows["k_random"].append((k - periodic) / tip_speed ** 2)
        table = {key: np.nanmean(np.asarray(value), axis=0) for key, value in rows.items()}
        spread = {key: np.nanstd(np.asarray(value), axis=0) for key, value in rows.items()}
        xi = (radii - impeller["tip_radius"]) / (tank["tank_radius"] - impeller["tip_radius"])
        print(f"  LDA outflow points, {impeller['name']} impeller, disk height, mean of {len(own)} baffle planes "
              f"(std between planes):")
        print("     r [mm]     xi   U_rad/U_tip        U_ax/U_tip   k*/U_tip^2   k_rand/U_tip^2")
        for index in range(radii.size):
            print(f"     {radii[index] * 1e3:5.1f}  {xi[index]:5.3f}   {table['radial'][index]:6.3f} ({spread['radial'][index]:5.3f})"
                  f"   {table['axial'][index]:6.3f}   {table['k'][index]:8.4f}   {table['k_random'][index]:8.4f}")
        np.savetxt(out_dir / f"{label}_lda_outflow_{impeller['name']}.csv",
                   np.column_stack([radii, xi, table["radial"], table["axial"], table["k"], table["k_random"]]),
                   delimiter=",", fmt="%.6e", header="r,xi,U_rad/U_tip,U_ax/U_tip,k/U_tip^2,k_random/U_tip^2", comments="")
        result[impeller["name"]] = {key: value.tolist() for key, value in table.items()}
    return result


def print_grid(title, r_over_R, y_over_T, values, fmt="{:7.3f}"):
    print(f"  {title}: rows r/R, columns y/T")
    print("     r/R  " + "".join(f"{y:8.3f}" for y in y_over_T))
    for row, r in enumerate(r_over_R):
        print(f"   {r:5.3f}  " + "".join(fmt.format(values[row, column]) for column in range(len(y_over_T))))


def midplane_tables(stats, meta, tank, tip_speed, out_dir, label):
    """7 x 11 tables of the mid-compartment plane, mean of the baffle planes."""
    sets = [entry for entry in meta["sets"] if entry["name"].startswith("midplane_") and entry["name"] != "midplane_disk"]
    if not sets:
        return {}
    r_over_R, y_over_T = np.asarray(sets[0]["r_over_R"]), np.asarray(sets[0]["y_over_T"])
    shape = (r_over_R.size, y_over_T.size)
    planes = dict(radial=[], tangential=[], axial=[], rms_radial=[], rms_tangential=[], rms_axial=[])
    for entry in sets:
        _, mean, variance = moments(stats, entry["name"])
        r_index, y_index = np.asarray(entry["r_index"]), np.asarray(entry["y_index"])
        for key, column in (("radial", 0), ("tangential", 1), ("axial", 2)):
            grid = np.full(shape, np.nan)
            grid[r_index, y_index] = mean[:, column] / tip_speed
            planes[key].append(grid)
            grid = np.full(shape, np.nan)
            grid[r_index, y_index] = np.sqrt(np.maximum(variance[:, column], 0.0)) / tip_speed
            planes["rms_" + key].append(grid)
    table = {key: np.nanmean(np.asarray(value), axis=0) for key, value in planes.items()}
    table["k2"] = 0.75 * (table["rms_radial"] ** 2 + table["rms_axial"] ** 2)
    table["k3"] = 0.5 * (table["rms_radial"] ** 2 + table["rms_tangential"] ** 2 + table["rms_axial"] ** 2)
    y_c = sets[0]["midplane_height"]
    print(f"  mid-compartment plane y_c = {y_c * 1e3:.1f} mm (y/T = 0), mean of {len(sets)} baffle planes:")
    print_grid("U_rad / U_tip", r_over_R, y_over_T, table["radial"])
    print_grid("U_ax / U_tip", r_over_R, y_over_T, table["axial"])
    print_grid("U_tan / U_tip", r_over_R, y_over_T, table["tangential"])
    print_grid("u_rad,rms / U_tip", r_over_R, y_over_T, table["rms_radial"])
    print_grid("u_ax,rms / U_tip", r_over_R, y_over_T, table["rms_axial"])
    print_grid("u_tan,rms / U_tip", r_over_R, y_over_T, table["rms_tangential"])
    print_grid("1000 k2* / U_tip^2 = 750 (u'_rad^2 + u'_ax^2) / U_tip^2 (paper's estimate)", r_over_R, y_over_T,
               1e3 * table["k2"], fmt="{:8.2f}")
    print_grid("1000 k3* / U_tip^2 = 500 sum_i u'_i^2 / U_tip^2 (three components)", r_over_R, y_over_T,
               1e3 * table["k3"], fmt="{:8.2f}")
    with open(out_dir / f"{label}_midplane.csv", "w", encoding="utf-8") as handle:
        handle.write("r/R,y/T,U_rad/U_tip,U_tan/U_tip,U_ax/U_tip,u_rad_rms/U_tip,u_tan_rms/U_tip,u_ax_rms/U_tip,"
                     "k2/U_tip^2,k3/U_tip^2\n")
        for row, r in enumerate(r_over_R):
            for column, y in enumerate(y_over_T):
                handle.write(f"{r:.3f},{y:.3f}," + ",".join(
                    f"{table[key][row, column]:.6e}" for key in ("radial", "tangential", "axial", "rms_radial",
                                                                 "rms_tangential", "rms_axial", "k2", "k3")) + "\n")
    return dict(r_over_R=r_over_R.tolist(), y_over_T=y_over_T.tolist(),
                **{key: value.tolist() for key, value in table.items()})


def midplane_exchange(stats, meta, tank, speed, diameter, lower_pumping):
    """Q_ax from the polar grid of the mid plane: minus the sum of the downward <u_y> dA."""
    disk = next((entry for entry in meta["sets"] if entry["name"] == "midplane_disk"), None)
    if disk is None:
        return {}
    _, mean, _ = moments(stats, "midplane_disk")
    radii, counts, width = np.asarray(disk["ring_radii"]), np.asarray(disk["ring_counts"]), disk["ring_width"]
    area = np.repeat(radii * width * 2.0 * math.pi / counts, counts)
    axial = mean[:, 2]
    downward = max(-float(np.nansum(np.clip(axial, None, 0.0) * area)), 0.0)
    upward = float(np.nansum(np.clip(axial, 0.0, None) * area))
    net = float(np.nansum(axial * area))
    result = dict(downward=downward, upward=upward, net=net, flow_number=downward / (speed * diameter ** 3),
                  over_pumping=(downward / lower_pumping if lower_pumping else float("nan")))
    print(f"  mid-plane exchange (y = {disk['midplane_height'] * 1e3:.1f} mm, {len(radii)} rings, {axial.size} points, "
          f"{int(np.isnan(axial).sum())} without particles): Q_ax (down) = {downward * 1e3:.3f} L/s, up {upward * 1e3:.3f} L/s, "
          f"net {net * 1e3:+.4f} L/s; Q_ax / (N D^3) = {result['flow_number']:.3f}; Q_ax / Q_pump,lower = "
          f"{result['over_pumping']:.3f}")
    return result


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
    first = tank["impellers"][0]
    diameter, speed = first["diameter"], tank["rpm"] / 60.0
    tip_speed, radius = tank["tip_speed"], tank["tank_radius"]
    dx = tank["dx"]
    delta = args.delta if args.delta is not None else dx
    viscosity = meta["viscosity"]
    epsilon_scale = speed ** 3 * diameter ** 2
    label = args.label or pathlib.Path(args.stats).stem
    out_dir = pathlib.Path(args.out_dir) if args.out_dir else pathlib.Path(args.stats).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{label}: {meta['samples']} samples, t = {meta['first_time']:.2f} .. {meta['last_time']:.2f} s, "
          f"kernel radius {meta['kernel_radius'] * 1e3:.2f} mm, dx {dx * 1e3:g} mm; U_tip {tip_speed:.4f} m/s, "
          f"N {speed:g} 1/s, D {diameter:.4f} m, {len(tank['impellers'])} impeller(s)")

    # --- discharge profiles, per impeller ---------------------------------------------------------
    lines = [entry for entry in meta["sets"] if entry["name"].startswith("discharge_")]
    coordinate = np.asarray(lines[0]["coordinate"])
    profiles = {}
    for impeller in tank["impellers"]:
        own = [entry for entry in lines if impeller_of(entry, tank)["name"] == impeller["name"]]
        if not own:
            continue
        profile, spread = discharge_profiles(stats, own, tip_speed, epsilon_scale, viscosity, args.cs, delta)
        profiles[impeller["name"]] = (profile, spread)
        where = f"y = C" if len(tank["impellers"]) == 1 else f"{impeller['name']} impeller, y = {impeller['center_height'] * 1e3:.1f} mm"
        print(f"  discharge at {where} between the baffles, mean of the four lines (std between lines):")
        print("     xi   U_rad/U_tip   U_tan/U_tip   k/U_tip^2   k_rand/U_tip^2   eps_SGS/N3D2   2nuSS/N3D2")
        for index in range(0, coordinate.size, 4):
            print(f"   {coordinate[index]:5.3f}   {profile['radial'][index]:6.3f} ({spread['radial'][index]:5.3f})"
                  f"   {profile['tangential'][index]:6.3f}   {profile['k'][index]:8.4f}   {profile['k_random'][index]:8.4f}"
                  f"        {profile['epsilon_sgs'][index]:8.3f}     {profile['epsilon_resolved'][index]:8.4f}")

    # --- pumping number, per impeller --------------------------------------------------------------
    pumping_numbers = {}
    for impeller in tank["impellers"]:
        for cylinder in meta["sets"]:
            if cylinder["name"].startswith("tip_cylinder") and impeller_of(cylinder, tank)["name"] == impeller["name"]:
                pumping_numbers[impeller["name"]] = pumping(stats, cylinder, speed, impeller["diameter"], tip_speed,
                                                            impeller["center_height"])

    # --- power number ------------------------------------------------------------------------------
    power = power_numbers(args, speed, diameter, tank)
    for name, value in REFERENCE_POWER:
        print(f"    reference Po {name}: {value}")

    # --- 2-impeller tank: LDA outflow points, mid plane, exchange flow ----------------------------
    outflow_points = lda_outflow(stats, meta, tank, tip_speed, out_dir, label)
    midplane = midplane_tables(stats, meta, tank, tip_speed, out_dir, label)
    lower_pumping = pumping_numbers.get(first["name"], {}).get("outflow")
    exchange = midplane_exchange(stats, meta, tank, speed, diameter, lower_pumping)

    # --- outputs -------------------------------------------------------------------------------------
    for index, impeller in enumerate(tank["impellers"]):
        if impeller["name"] not in profiles:
            continue
        profile, _ = profiles[impeller["name"]]
        table = np.column_stack([coordinate] + [profile[key] for key in ("radial", "tangential", "axial", "k", "k_random",
                                                                          "epsilon_sgs", "epsilon_resolved")])
        suffix = "" if index == 0 else f"_{impeller['name']}"
        np.savetxt(out_dir / f"{label}_discharge{suffix}.csv", table, delimiter=",", fmt="%.6e",
                   header="xi,U_rad/U_tip,U_tan/U_tip,U_ax/U_tip,k/U_tip^2,k_random/U_tip^2,eps_sgs/N3D2,eps_2nuSS/N3D2",
                   comments="")
    first_profile = profiles[first["name"]][0]
    first_pumping = pumping_numbers.get(first["name"], {})
    summary = dict(label=label, samples=meta["samples"], first_time=meta["first_time"], last_time=meta["last_time"],
                   flow_number=first_pumping.get("flow_number"), outflow=first_pumping.get("outflow"),
                   net_flow=first_pumping.get("net_flow"), power=power.get("total", power), delta=delta, cs=args.cs,
                   peak_radial=float(np.nanmax(first_profile["radial"])),
                   peak_k=float(np.nanmax(first_profile["k"])), peak_epsilon_sgs=float(np.nanmax(first_profile["epsilon_sgs"])))
    if len(tank["impellers"]) > 1:
        summary["impellers"] = {name: dict(pumping=pumping_numbers.get(name), peak_radial=float(np.nanmax(profile["radial"])),
                                           peak_k=float(np.nanmax(profile["k"])))
                                for name, (profile, _) in profiles.items()}
        summary["power_split"] = power
    if outflow_points:
        summary["lda_outflow"] = outflow_points
    if exchange:
        summary["midplane_exchange"] = exchange
    (out_dir / f"{label}_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    plot(out_dir, label, coordinate, profiles, stats, meta, tank, midplane)
    print(f"  wrote {out_dir / (label + '_discharge.csv')}, _summary.json and figures")


def plot(out_dir, label, coordinate, profiles, stats, meta, tank, midplane):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for index, (name, (profile, spread)) in enumerate(profiles.items()):
        figure, axes = plt.subplots(1, 3, figsize=(15, 4.3))
        for axis, keys, title in ((axes[0], (("radial", "U_rad"), ("tangential", "U_tan"), ("axial", "U_ax")), "mean velocity / U_tip"),
                                  (axes[1], (("k", "k total"), ("k_random", "k without blade passage")), "k / U_tip^2"),
                                  (axes[2], (("epsilon_sgs", "(C_s dx)^2 <|S|^3>"), ("epsilon_resolved", "2 nu <S:S>")),
                                   "epsilon / (N^3 D^2)")):
            for key, legend in keys:
                axis.plot(coordinate, profile[key], label=legend)
                axis.fill_between(coordinate, profile[key] - spread[key], profile[key] + spread[key], alpha=0.2)
            axis.set_xlabel("(r - r_tip) / (R - r_tip)")
            axis.set_title(title)
            axis.grid(alpha=0.3)
            axis.legend(fontsize=8)
        where = "y = C" if len(profiles) == 1 else f"{name} impeller"
        figure.suptitle(f"{label}: discharge at {where} between the baffles, t = {meta['first_time']:.1f} .. {meta['last_time']:.1f} s")
        figure.tight_layout()
        suffix = "" if index == 0 else f"_{name}"
        figure.savefig(out_dir / f"{label}_discharge{suffix}.png", dpi=130)
        plt.close(figure)

    planes = [entry for entry in meta["sets"] if entry["name"].startswith("plane_")]
    if planes:
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

    if midplane:
        # axial profiles per radial position as Figure 3 of Haringa et al. (2018): U_rad, U_ax, k2*
        r_over_R, y_over_T = np.asarray(midplane["r_over_R"]), np.asarray(midplane["y_over_T"])
        figure, axes = plt.subplots(1, 3, figsize=(15, 4.6))
        for axis, key, title in ((axes[0], "radial", "U_rad / U_tip"), (axes[1], "axial", "U_ax / U_tip"),
                                 (axes[2], "k2", "k2* / U_tip^2 = 3/4 (u'_rad^2 + u'_ax^2) / U_tip^2")):
            values = np.asarray(midplane[key])
            for row, r in enumerate(r_over_R):
                axis.plot(y_over_T, values[row], marker="o", markersize=3, label=f"r/R = {r:.3f}")
            axis.set_xlabel("y / T (0 = 5T/6)")
            axis.set_title(title)
            axis.grid(alpha=0.3)
        axes[0].legend(fontsize=7)
        figure.suptitle(f"{label}: mid-compartment plane, baffle planes, t = {meta['first_time']:.1f} .. {meta['last_time']:.1f} s")
        figure.tight_layout()
        figure.savefig(out_dir / f"{label}_midplane.png", dpi=130)
        plt.close(figure)


if __name__ == "__main__":
    main()
