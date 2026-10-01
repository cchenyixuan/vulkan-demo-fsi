"""_analyze_lifelines.py — analysis of recorded lifelines (2026-10-01, stage 3).

Lifelines are written by experiment/v1/utils/lifeline_recorder.py (headless runner --lifeline-dir):
a fixed random sample of FLUID particles, identified by the persistent uid, recorded every ~0.03 s.

Subcommands
  integrity DIR       sample size, lost particles, displacement per record against the stored speed
                      (a uid mix-up shows as a jump of many particle spacings)
  passive DIR         passive-particle statistics in the 30 L tank (coordinates of
                      utils/geometry/_demo_stirred_tank_30l.py):
                        - zone and height-slab occupancy over time against the first record
                          (a uniform sample of an incompressible fluid must stay uniform)
                        - pumping through the impellers counted from the trajectories:
                          Rushton: particles leaving the swept cylinder radially through the blade band;
                          PBT: particles leaving the swept cylinder through its bottom (down-pumping);
                          Q = event rate x (fluid particles / sample size) x dx^3, Fl = Q / (N D^3)
                        - circulation times: intervals between successive pumping events of a particle
                        - exchange between the upper and lower compartments (plane between the
                          impellers, hysteresis): flow, residence-time distributions, radius of the
                          upward and downward crossings
                        - with --aux records: particle shift per step against the advective step
  tracer DIR --case CASE.yaml
                      lifelines of the tracer pulses (fields recorded with --lifeline-fields):
                      per-particle settling time (last exit from +-5 % of the mean), arrival and
                      peak, and the Lagrangian mixed fraction against time
  regime DIR --field F [--monod QMAX KS] | --q-field F
                      regime and arc analysis of Haringa 2023 (Eng. Life Sci. 23:e2100159, sec. 2.3.4):
                      q/q_max smoothed over 0.36 s; starvation S < 0.05, excess E > 0.95, limitation L;
                      transitions need to pass the boundary by 0.01; residence times per pattern
                      LEL, LSL, ELE, ELS, SLE, SLS; arcs at q/q_max = 0.05 (duration, maximum)
  selftest            regime / arc analysis of a synthetic telegraph signal with known residence times

Usage (repo root):
    python experiment/v1/checks/_analyze_lifelines.py integrity DIR
    python experiment/v1/checks/_analyze_lifelines.py passive DIR [--out OUT] [--split-height 0.117]
    python experiment/v1/checks/_analyze_lifelines.py tracer DIR --case CASE.yaml [--out OUT]
    python experiment/v1/checks/_analyze_lifelines.py selftest [--out OUT]
"""
import argparse
import json
import math
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.v1.utils.lifeline_recorder import load_lifelines  # noqa: E402

# 30 L tank, generator frame (utils/geometry/_demo_stirred_tank_30l.py): +y up, liquid top 0.4265 m
TANK = dict(radius=0.144, top=0.4265, floor=-0.0618, speed_rpm=200.0, diameter=0.096,
            rushton=dict(r_tip=0.048, y0=0.0288, y1=0.048),
            pbt=dict(r_tip=0.0491, y0=0.1850, y1=0.2043),
            zones={"Rushton": (0.072, 0.0187, 0.0585), "PBT": (0.072, 0.1645, 0.2243)},
            split=0.117)


def radius_of(position):
    return np.hypot(position[..., 0], position[..., 2])


# ---------------------------------------------------------------------------------------------
# event and segment helpers (vectorised over the particles, loop over the records)
# ---------------------------------------------------------------------------------------------
def armed_events(arm, fire, disarm, times):
    """Per particle: becomes armed where `arm`, fires an event (and disarms) at the first later record
    where `fire`, disarms without event where `disarm`. Returns particle index and time of the events."""
    state = arm[0].copy()
    event_particle, event_time = [], []
    for index in range(1, arm.shape[0]):
        fired = state & fire[index]
        if fired.any():
            hit = np.flatnonzero(fired)
            event_particle.append(hit)
            event_time.append(np.full(hit.size, times[index]))
        state = (state & ~fired & ~disarm[index]) | arm[index]
    if not event_particle:
        return np.zeros(0, dtype=np.int64), np.zeros(0)
    return np.concatenate(event_particle), np.concatenate(event_time)


def intervals(event_particle, event_time):
    """Time between successive events of the same particle."""
    if event_particle.size < 2:
        return np.zeros(0)
    order = np.lexsort((event_time, event_particle))
    particle, time = event_particle[order], event_time[order]
    same = particle[1:] == particle[:-1]
    return (time[1:] - time[:-1])[same]


def two_state(upper, lower, initial):
    """Hysteresis state: True above (`upper`), False below (`lower`), unchanged in between."""
    states = np.empty(upper.shape, dtype=bool)
    state = initial.copy()
    for index in range(upper.shape[0]):
        state = (state | upper[index]) & ~lower[index]
        states[index] = state
    return states


def segments(states, times):
    """Run-length segments of an integer state array (n, K) along time.
    Returns dict of arrays: particle, state, start, end, previous, next, truncated (first or last)."""
    n, count = states.shape
    change = np.zeros((n, count), dtype=bool)
    change[0] = True
    change[1:] = states[1:] != states[:-1]
    out = {key: [] for key in ("particle", "state", "start", "end", "start_index", "end_index",
                               "previous", "next", "truncated")}
    for particle in range(count):
        starts = np.flatnonzero(change[:, particle])
        ends = np.append(starts[1:], n)
        values = states[starts, particle]
        m = starts.size
        out["particle"].append(np.full(m, particle))
        out["state"].append(values)
        out["start"].append(times[starts])
        out["end"].append(np.where(ends < n, times[np.minimum(ends, n - 1)], times[-1]))
        out["start_index"].append(starts)
        out["end_index"].append(ends)
        previous = np.full(m, -1); previous[1:] = values[:-1]
        following = np.full(m, -1); following[:-1] = values[1:]
        out["previous"].append(previous)
        out["next"].append(following)
        truncated = np.zeros(m, dtype=bool); truncated[0] = True; truncated[-1] = True
        out["truncated"].append(truncated)
    return {key: np.concatenate(value) for key, value in out.items()}


def describe(values, unit="s"):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return "n = 0"
    return (f"n = {values.size:,}, mean {values.mean():.3f} {unit}, median {np.median(values):.3f}, "
            f"q10 {np.quantile(values, 0.1):.3f}, q90 {np.quantile(values, 0.9):.3f}, max {values.max():.3f}")


# ---------------------------------------------------------------------------------------------
def command_integrity(arguments):
    data = load_lifelines(arguments.dir)
    meta, position, time = data["meta"], data["position"], data["time"]
    dx = float(meta["particle_spacing_m"])
    n, count, _ = position.shape
    lost = ~np.isfinite(position[..., 0])
    print(f"{arguments.dir}: {count:,} particles, {n:,} records, t = {time[0]:.3f} .. {time[-1]:.3f} s, "
          f"interval {np.median(np.diff(time)):.4f} s")
    print(f"  uids unique: {np.unique(data['uid']).size == count}; lost samples {int(lost.sum())} "
          f"({lost.any(axis=0).sum()} particles ever lost)")
    step = np.linalg.norm(np.diff(position, axis=0), axis=2) / dx
    finite = np.isfinite(step)
    print(f"  displacement per record, particle spacings: median {np.nanmedian(step):.3f}, "
          f"q99.9 {np.nanquantile(step[finite], 0.999):.3f}, max {np.nanmax(step):.3f}")
    if "velocity" in data:
        speed = np.linalg.norm(data["velocity"], axis=2)
        interval = float(np.median(np.diff(time)))
        print(f"  stored speed at the aux records: max {np.nanmax(speed):.3f} m/s -> at most "
              f"{np.nanmax(speed) * interval / dx:.2f} spacings per record (plus the shift)")
    jumps = step > arguments.jump
    print(f"  records with a jump > {arguments.jump} spacings: {int(jumps.sum())}")
    return 0 if not jumps.any() and not lost.any() else 1


# ---------------------------------------------------------------------------------------------
def command_passive(arguments):
    data = load_lifelines(arguments.dir)
    meta, position, time = data["meta"], data["position"], data["time"]
    dx = float(meta["particle_spacing_m"])
    n, count, _ = position.shape
    fluid_total = int(meta["fluid_candidates"])
    volume_per_sample = fluid_total * dx ** 3 / count
    span = float(time[-1] - time[0])
    x, y, r = position[..., 0], position[..., 1], radius_of(position)
    rate = TANK["speed_rpm"] / 60.0
    nd3 = rate * TANK["diameter"] ** 3
    result = {"particles": count, "records": n, "span_s": span, "fluid_particles": fluid_total, "dx": dx}
    print(f"{arguments.dir}: {count:,} lifelines over {span:.1f} s ({n} records), each stands for "
          f"{volume_per_sample * 1e6:.2f} mL of fluid")

    # occupancy against the first record
    print("\nzone occupancy (fraction of the sample), first record / time mean / min .. max over records:")
    result["occupancy"] = {}
    for name, (r_max, y_low, y_high) in TANK["zones"].items():
        inside = (r < r_max) & (y > y_low) & (y < y_high)
        fraction = inside.mean(axis=1)
        result["occupancy"][name] = [float(fraction[0]), float(fraction.mean()), float(fraction.min()), float(fraction.max())]
        print(f"  {name:8s} {fraction[0]:.4f} / {fraction.mean():.4f} / {fraction.min():.4f} .. {fraction.max():.4f}"
              f"   (sampling noise of one record {math.sqrt(fraction[0] * (1 - fraction[0]) / count):.4f})")
    edges = np.linspace(TANK["floor"], TANK["top"], 11)
    print("  height slabs (m): first record / time mean")
    slabs = []
    for low, high in zip(edges[:-1], edges[1:]):
        inside = (y >= low) & (y < high)
        fraction = inside.mean(axis=1)
        slabs.append([float(low), float(high), float(fraction[0]), float(fraction.mean())])
        print(f"    {low:7.3f} .. {high:6.3f}: {fraction[0]:.4f} / {fraction.mean():.4f} ({fraction.mean() / max(fraction[0], 1e-9):.3f})")
    result["slabs"] = slabs

    margin = arguments.margin * dx
    # Rushton: radial discharge through the blade band
    rushton = TANK["rushton"]
    band = (y > rushton["y0"] - margin) & (y < rushton["y1"] + margin)
    wide_band = (y > rushton["y0"] - 2 * margin) & (y < rushton["y1"] + 2 * margin)
    arm = (r < rushton["r_tip"] - margin) & band
    fire = (r > rushton["r_tip"] + margin) & wide_band
    disarm = ~wide_band & (r < rushton["r_tip"] + margin)
    rushton_particle, rushton_time = armed_events(arm, fire, disarm, time)
    # PBT: down-pumping through the bottom of the swept cylinder
    pbt = TANK["pbt"]
    inside_pbt = (r < pbt["r_tip"] - margin) & (y > pbt["y0"] - margin) & (y < pbt["y1"] + margin)
    fire_pbt = (y < pbt["y0"] - 2 * margin) & (r < pbt["r_tip"] + 2 * margin)
    disarm_pbt = (y > pbt["y1"] + 2 * margin) | (r > pbt["r_tip"] + 2 * margin)
    pbt_particle, pbt_time = armed_events(inside_pbt, fire_pbt, disarm_pbt, time)
    result["pumping"] = {}
    print(f"\npumping counted from the trajectories (margin {arguments.margin:g} dx; N D^3 = {nd3 * 1e3:.3f} L/s):")
    for name, particle, event_time in (("Rushton radial", rushton_particle, rushton_time),
                                       ("PBT downward", pbt_particle, pbt_time)):
        flow = particle.size / span * volume_per_sample
        circulation = intervals(particle, event_time)
        result["pumping"][name] = {"events": int(particle.size), "flow_m3_s": flow, "flow_number": flow / nd3,
                                   "circulation_mean_s": float(circulation.mean()) if circulation.size else None,
                                   "circulation_median_s": float(np.median(circulation)) if circulation.size else None}
        print(f"  {name:15s}: {particle.size:,} events, Q = {flow * 1e3:.3f} L/s, Fl = {flow / nd3:.3f}, "
              f"V/Q = {fluid_total * dx ** 3 / max(flow, 1e-12):.2f} s")
        print(f"     circulation time (between successive events of a particle): {describe(circulation)}")

    # compartments
    split = arguments.split_height
    upper = y > split + margin
    lower = y < split - margin
    initial = y[0] > split
    above = two_state(upper, lower, initial)
    crossing = above[1:] != above[:-1]
    record, particle = np.nonzero(crossing)
    direction_up = above[1:][record, particle]
    crossing_r = r[1:][record, particle]
    down_events = int((~direction_up).sum())
    up_events = int(direction_up.sum())
    exchange = down_events / span * volume_per_sample
    states = above.astype(np.int8)
    segment = segments(states, time)
    complete = ~segment["truncated"]
    duration = segment["end"] - segment["start"]
    upper_residence = duration[complete & (segment["state"] == 1)]
    lower_residence = duration[complete & (segment["state"] == 0)]
    fraction_upper = float(above.mean())
    print(f"\ncompartments split at y = {split:.3f} m (hysteresis +-{arguments.margin:g} dx): upper holds "
          f"{fraction_upper:.3f} of the sample")
    print(f"  crossings: {down_events:,} down, {up_events:,} up; exchange flow {exchange * 1e3:.3f} L/s "
          f"(Fl {exchange / nd3:.3f}); compartment turnover V_upper/Q = {fraction_upper * fluid_total * dx ** 3 / max(exchange, 1e-12):.2f} s, "
          f"V_lower/Q = {(1 - fraction_upper) * fluid_total * dx ** 3 / max(exchange, 1e-12):.2f} s")
    print(f"  residence in the upper compartment: {describe(upper_residence)}")
    print(f"  residence in the lower compartment: {describe(lower_residence)}")
    radius_edges = np.linspace(0.0, TANK["radius"], 7)
    print("  radius of the crossings (mm): share of the downward / upward crossings per radial band")
    down_hist = np.histogram(crossing_r[~direction_up], bins=radius_edges)[0] / max(down_events, 1)
    up_hist = np.histogram(crossing_r[direction_up], bins=radius_edges)[0] / max(up_events, 1)
    area = np.pi * (radius_edges[1:] ** 2 - radius_edges[:-1] ** 2) / (np.pi * TANK["radius"] ** 2)
    for low, high, down_share, up_share, area_share in zip(radius_edges[:-1], radius_edges[1:], down_hist, up_hist, area):
        print(f"    {low * 1e3:5.1f} .. {high * 1e3:5.1f}: down {down_share:.3f}   up {up_share:.3f}   (area {area_share:.3f})")
    result["compartments"] = {"split_m": split, "upper_fraction": fraction_upper, "down": down_events, "up": up_events,
                              "exchange_m3_s": exchange, "exchange_flow_number": exchange / nd3,
                              "upper_residence_mean_s": float(upper_residence.mean()) if upper_residence.size else None,
                              "lower_residence_mean_s": float(lower_residence.mean()) if lower_residence.size else None,
                              "down_radius_share": down_hist.tolist(), "up_radius_share": up_hist.tolist(),
                              "radius_edges_m": radius_edges.tolist()}

    if "shift" in data and "velocity" in data:
        timestep = float(meta["timestep_s"])
        shift = np.linalg.norm(data["shift"], axis=2)
        advective = np.linalg.norm(data["velocity"], axis=2) * timestep
        ratio = shift / np.maximum(advective, 1e-12)
        print(f"\nparticle shift of one step against v dt (aux records, {shift.size:,} samples):")
        print(f"  |shift| / dx: median {np.nanmedian(shift) / dx:.2e}, q90 {np.nanquantile(shift, 0.9) / dx:.2e}; "
              f"|v| dt / dx: median {np.nanmedian(advective) / dx:.2e}; ratio median {np.nanmedian(ratio):.3f}, q90 {np.nanquantile(ratio, 0.9):.3f}")
        print(f"  upper bound of the shift diffusivity (uncorrelated steps) <|shift|^2> / (6 dt) = "
              f"{np.nanmean(shift ** 2) / (6 * timestep):.2e} m^2/s")
        result["shift"] = {"median_over_dx": float(np.nanmedian(shift) / dx), "ratio_median": float(np.nanmedian(ratio)),
                           "diffusivity_upper_bound": float(np.nanmean(shift ** 2) / (6 * timestep))}

    out = pathlib.Path(arguments.out) if arguments.out else pathlib.Path(arguments.dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "passive_summary.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
        bins = np.linspace(0, 40, 81)
        for name, particle, event_time, color in (("Rushton", rushton_particle, rushton_time, "#1f77b4"),
                                                  ("PBT", pbt_particle, pbt_time, "#ff7f0e")):
            circulation = intervals(particle, event_time)
            if circulation.size:
                axes[0].hist(circulation, bins=bins, histtype="step", density=True, color=color, label=f"{name}, mean {circulation.mean():.1f} s")
        axes[0].set_xlabel("circulation time, s"); axes[0].set_ylabel("probability density"); axes[0].legend(); axes[0].set_yscale("log")
        for values, label, color in ((upper_residence, "upper compartment", "#2ca02c"), (lower_residence, "lower compartment", "#d62728")):
            if values.size:
                axes[1].hist(values, bins=bins, histtype="step", density=True, color=color, label=f"{label}, mean {values.mean():.1f} s")
        axes[1].set_xlabel("residence time, s"); axes[1].legend(); axes[1].set_yscale("log")
        centres = 0.5 * (radius_edges[1:] + radius_edges[:-1]) * 1e3
        width = (radius_edges[1] - radius_edges[0]) * 1e3 * 0.4
        axes[2].bar(centres - width / 2, down_hist, width=width, label="downward", color="#9467bd")
        axes[2].bar(centres + width / 2, up_hist, width=width, label="upward", color="#8c564b")
        axes[2].plot(centres, area, "k--", label="area share")
        axes[2].set_xlabel("radius of the crossing, mm"); axes[2].set_ylabel("share of crossings"); axes[2].legend()
        axes[2].set_title(f"plane y = {split:.3f} m")
        fig.tight_layout(); fig.savefig(out / "passive_summary.png", dpi=120)
        print(f"\nwrote {out / 'passive_summary.json'} and passive_summary.png")
    except ImportError:
        print(f"\nwrote {out / 'passive_summary.json'} (no matplotlib)")
    return 0


# ---------------------------------------------------------------------------------------------
def command_tracer(arguments):
    import yaml
    data = load_lifelines(arguments.dir)
    if "scalars" not in data:
        raise SystemExit("no scalar fields in these lifelines (record with --lifeline-fields)")
    case = yaml.safe_load(pathlib.Path(arguments.case).read_text(encoding="utf-8"))
    injections = {item["field"]: item for item in case["scalars"].get("injections", [])}
    time, values, names = data["time"], data["scalars"].astype(np.float64), data["field_names"]
    print(f"{arguments.dir}: {values.shape[1]:,} lifelines, {time.size} records, fields {names}")
    rows = []
    for field_index, name in enumerate(names):
        if name not in injections:
            continue
        start = float(injections[name]["start"]); end = start + float(injections[name]["duration"])
        after = time >= end
        if after.sum() < 10:
            continue
        signal = values[:, :, field_index]
        mean = np.nanmean(signal[-1])          # equal particle masses: sample mean = conserved mean
        relative = signal / mean
        outside = np.abs(relative - 1.0) > 0.05
        mixed = (~outside).mean(axis=1)
        # per particle: last record outside the band
        last_outside = np.where(outside.any(axis=0), time[(outside.shape[0] - 1) - np.argmax(outside[::-1], axis=0)], np.nan)
        settle = last_outside - start
        arrival_index = np.argmax(relative >= 0.5, axis=0)
        arrived = (relative >= 0.5).any(axis=0)
        arrival = np.where(arrived, time[arrival_index] - start, np.nan)
        peak = np.nanmax(np.where(time[:, None] >= start, relative, np.nan), axis=0)
        stays = np.flatnonzero(mixed >= 0.95)
        lagrangian_tau95 = np.nan
        if stays.size:
            below = np.flatnonzero(mixed < 0.95)
            last_below = below[below < time.size][-1] if below.size else -1
            if last_below + 1 < time.size:
                lagrangian_tau95 = time[last_below + 1] - start
        rows.append((name, start, lagrangian_tau95, np.nanmedian(settle), np.nanquantile(settle, 0.95), np.nanmax(settle),
                     np.nanmedian(arrival), np.nanmedian(peak)))
    print(f"{'field':>10s} {'start':>6s} {'tau95 (95 % of sample)':>23s} {'settle median':>13s} {'q95':>7s} {'max':>7s} "
          f"{'arrival C/Cinf>=0.5 median':>27s} {'peak median':>11s}")
    for row in rows:
        print(f"{row[0]:>10s} {row[1]:6.2f} {row[2]:23.2f} {row[3]:13.2f} {row[4]:7.2f} {row[5]:7.2f} {row[6]:27.2f} {row[7]:11.2f}")
    if rows:
        tau = np.array([row[2] for row in rows])
        print(f"Lagrangian tau95: mean {np.nanmean(tau):.2f} s, std {np.nanstd(tau):.2f} s over {np.isfinite(tau).sum()} pulses")
    out = pathlib.Path(arguments.out) if arguments.out else pathlib.Path(arguments.dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tracer_summary.json").write_text(json.dumps(
        [dict(zip(("field", "start", "tau95_lagrangian", "settle_median", "settle_q95", "settle_max", "arrival_median", "peak_median"),
                  [row[0]] + [float(v) for v in row[1:]])) for row in rows], indent=1), encoding="utf-8")
    return 0


# ---------------------------------------------------------------------------------------------
def moving_average(signal, window):
    """Centred moving average over `window` records along axis 0 (edges: shorter windows)."""
    if window <= 1:
        return signal.copy()
    filled = np.where(np.isfinite(signal), signal, 0.0)
    weight = np.isfinite(signal).astype(np.float64)
    cumulative = np.concatenate([np.zeros((1,) + signal.shape[1:]), np.cumsum(filled, axis=0)])
    cumulative_weight = np.concatenate([np.zeros((1,) + signal.shape[1:]), np.cumsum(weight, axis=0)])
    n = signal.shape[0]
    half = window // 2
    low = np.clip(np.arange(n) - half, 0, n)
    high = np.clip(np.arange(n) - half + window, 0, n)
    total = cumulative[high] - cumulative[low]
    count = cumulative_weight[high] - cumulative_weight[low]
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def regime_states(q, low=0.05, high=0.95, hysteresis=0.01):
    """0 = starvation, 1 = limitation, 2 = excess, with the boundary hysteresis of Haringa 2023."""
    states = np.empty(q.shape, dtype=np.int8)
    state = np.where(q[0] < low, 0, np.where(q[0] > high, 2, 1)).astype(np.int8)
    for index in range(q.shape[0]):
        value = q[index]
        to_excess = value > high + hysteresis
        to_starvation = value < low - hysteresis
        from_excess = (state == 2) & (value < high - hysteresis)
        from_starvation = (state == 0) & (value > low + hysteresis)
        state = np.where(to_excess, 2, np.where(to_starvation, 0,
                         np.where(from_excess | from_starvation, 1, state))).astype(np.int8)
        states[index] = state
    return states


def regime_analysis(q, time, window_seconds=0.36):
    interval = float(np.median(np.diff(time)))
    smooth = moving_average(q, max(1, round(window_seconds / interval))) if window_seconds > 0 else q
    states = regime_states(smooth)
    fractions = [float((states == value).mean()) for value in (0, 1, 2)]
    segment = segments(states, time)
    complete = ~segment["truncated"]
    duration = segment["end"] - segment["start"]
    letters = np.array(["S", "L", "E"])
    patterns = {}
    for pattern in ("LEL", "LSL", "ELE", "ELS", "SLE", "SLS"):
        code = [{"S": 0, "L": 1, "E": 2}[c] for c in pattern]
        select = complete & (segment["previous"] == code[0]) & (segment["state"] == code[1]) & (segment["next"] == code[2])
        patterns[pattern] = duration[select]
    # arcs at 0.05
    arc_states = two_state(smooth > 0.05 + 0.01, smooth < 0.05 - 0.01, smooth[0] > 0.05).astype(np.int8)
    arc = segments(arc_states, time)
    arc_complete = ~arc["truncated"]
    arc_duration = arc["end"] - arc["start"]
    arc_peak = []
    above = arc_complete & (arc["state"] == 1)
    for particle, i0, i1 in zip(arc["particle"][above], arc["start_index"][above], arc["end_index"][above]):
        arc_peak.append(np.nanmax(smooth[i0:max(i1, i0 + 1), particle]))
    return {"fractions": fractions, "patterns": patterns, "letters": letters,
            "arc_above": arc_duration[above], "arc_above_peak": np.asarray(arc_peak),
            "arc_below": arc_duration[arc_complete & (arc["state"] == 0)]}


def print_regime(result, label):
    fractions = result["fractions"]
    print(f"{label}: time fractions  E {fractions[2] * 100:.1f} %  L {fractions[1] * 100:.1f} %  S {fractions[0] * 100:.1f} %")
    for pattern, values in result["patterns"].items():
        print(f"  {pattern}: {describe(values)}")
    print(f"  arcs above 0.05: {describe(result['arc_above'])}; below: {describe(result['arc_below'])}")


def command_regime(arguments):
    data = load_lifelines(arguments.dir)
    names = data.get("field_names", [])
    time = data["time"]
    if arguments.q_field:
        q = data["scalars"][:, :, names.index(arguments.q_field)].astype(np.float64)
    else:
        concentration = data["scalars"][:, :, names.index(arguments.field)].astype(np.float64)
        q_max, k_s = arguments.monod
        q = concentration / (k_s + np.maximum(concentration, 0.0))   # q / q_max
    result = regime_analysis(q, time)
    print_regime(result, arguments.dir)
    return 0


def command_selftest(arguments):
    """Synthetic lifelines: S -> L, E -> L, L -> E with probability 0.4, else L -> S; dwell times
    0.5 s + exponential with means 4 (S), 5 (L), 3 (E) s; q = 0.005 / U(0.2, 0.8) / 0.995 plus noise.
    The segmentation and pattern logic must recover the generator's residence times without the
    moving average (pass / fail at 2 %); the 0.36 s filter of Haringa 2023 is then applied and its
    bias on such step-like signals is reported (not a failure: it is a property of the method)."""
    rng = np.random.default_rng(7)
    interval, span, count = 0.03, 900.0, 600
    time = np.arange(0.0, span, interval)
    q = np.empty((time.size, count))
    true_patterns = {key: [] for key in ("LEL", "LSL", "ELE", "ELS", "SLE", "SLS")}
    total = {"S": 0.0, "L": 0.0, "E": 0.0}
    mean = {"S": 4.0, "L": 5.0, "E": 3.0}
    for particle in range(count):
        state = rng.choice(["S", "L", "E"])
        t = 0.0
        history = []
        while t < span:
            dwell = 0.5 + rng.exponential(mean[state] - 0.5)
            level = {"S": 0.005, "E": 0.995, "L": rng.uniform(0.2, 0.8)}[state]
            history.append((state, t, t + dwell, level))
            t += dwell
            state = "L" if state in ("S", "E") else ("E" if rng.random() < 0.4 else "S")
        ends = np.array([item[2] for item in history])
        levels = np.array([item[3] for item in history])
        q[:, particle] = levels[np.searchsorted(ends, time, side="right")]
        for index in range(1, len(history) - 1):
            state, start, end, _ = history[index]
            total[state] += end - start
            pattern = history[index - 1][0] + state + history[index + 1][0]
            if pattern in true_patterns:
                true_patterns[pattern].append(end - start)
    q = np.clip(q + rng.normal(0.0, 0.003, q.shape), 0.0, 1.0)
    share = {k: v / sum(total.values()) for k, v in total.items()}
    print(f"truth: time fractions E {share['E'] * 100:.1f} %  L {share['L'] * 100:.1f} %  S {share['S'] * 100:.1f} %")
    failures = 0
    for window, label in ((0.0, "no filter"), (0.36, "0.36 s moving average")):
        result = regime_analysis(q, time, window_seconds=window)
        print_regime(result, f"synthetic, {label}")
        for pattern, values in true_patterns.items():
            found = result["patterns"][pattern]
            error = (found.mean() - np.mean(values)) / np.mean(values)
            if window == 0.0:
                failures += abs(error) > 0.02
            print(f"    {pattern}: true mean {np.mean(values):.3f} s (n {len(values)}), found {found.mean():.3f} s "
                  f"(n {found.size}), {error * 100:+.1f} %")
    print("PASS (segmentation and patterns within 2 % without the filter)" if failures == 0
          else f"FAIL ({failures} patterns off by more than 2 % without the filter)")
    return 0 if failures == 0 else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("integrity"); p.add_argument("dir"); p.add_argument("--jump", type=float, default=20.0)
    p = sub.add_parser("passive"); p.add_argument("dir"); p.add_argument("--out", default=None)
    p.add_argument("--split-height", type=float, default=TANK["split"])
    p.add_argument("--margin", type=float, default=1.0, help="hysteresis margin in particle spacings")
    p = sub.add_parser("tracer"); p.add_argument("dir"); p.add_argument("--case", required=True); p.add_argument("--out", default=None)
    p = sub.add_parser("regime"); p.add_argument("dir"); p.add_argument("--field", default=None)
    p.add_argument("--q-field", default=None); p.add_argument("--monod", type=float, nargs=2, default=(1.0, 7.8e-6), metavar=("QMAX", "KS"))
    p = sub.add_parser("selftest"); p.add_argument("--out", default=None)
    arguments = parser.parse_args()
    return {"integrity": command_integrity, "passive": command_passive, "tracer": command_tracer,
            "regime": command_regime, "selftest": command_selftest}[arguments.command](arguments)


if __name__ == "__main__":
    sys.exit(main())
