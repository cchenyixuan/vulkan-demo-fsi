"""
_analyze_pbt_bands.py — the PBT torque by radius against the flow the blades see, Fluent and SPH (2026-10-04).

Our PBT takes 20-36 % more torque than Fluent's in the same flow, and more with finer particles, unlike every
other thin plate. Two explanations: the fluid in the swept region turns less with the blades (larger relative
velocity), or the pitched thin plate gives more force for the same relative flow. Per radial band:

  torque     Fluent: wall zone pbt per snapshot (_fluent_snapshot_reports.py style ASCII export: face centre,
             area vector from the fluid into the solid, pressure, wall shear); force p A + tau |A|, torque about
             +y = r F_theta, the sign set so that the resistance is positive; blade faces (|n . e_theta| > 0.3,
             r > 8 mm) are front (pressure side, rotation * n_theta < 0) or back, the rest (sleeve, edges, hub)
             apart. SPH: _check_impeller_parts.py --pbt-bands ... --out PREFIX (PREFIX_parts.json), window
             0.3..0.7 s, front + back per band (blades of ordinary particles, generator --conformal-pbt: the
             net per band, by the particles' own radius), lattice hub and collar apart. Front and back each carry the
             reference pressure (Fluent's gauge reference, our background pressure) times their area; only the
             net of a band is physical.
  flow       azimuthal means in the band (cells / particles weighted equally; Fluent's rotor-zone velocities
             turned by 12 degrees, see _analyze_rushton_lower_flow.py; mirrored Fluent snapshots mirrored back):
             blade zone y 186..203 mm, above the blades y 205..213 mm, below y 176..184 mm; u_theta, u_y, u_r.
  derived    U = omega r_mid, W_t = U - <u_theta> (blade zone), W_y = <u_y>, relative flow angle below the
             horizontal phi = atan(-W_y / W_t), angle of attack alpha = 45 deg - phi, and the coefficient
             C = net torque / (1/2 rho (W_t^2 + W_y^2) r_mid A), A = 6 blades x chord 24.8 mm x the band's part
             of the blade span: Fluent 7.56..48.0 mm (blades down to the sleeve), SPH --span (default
             12.2..48.2 mm, the generator's thin plates; with --fluent-hubs 7.56..48.2 mm).

Equal C with a different W or alpha: the flow differs (explanation 1). Different C at equal W and alpha: the
blade model differs (explanation 2).

usage (base Anaconda):
    python experiment/v1/checks/_analyze_pbt_bands.py --fluent fine25_pbt.csv fine25.ip [--fluent ...] \
        --sph LABEL PREFIX_parts.json DUMP.npz [DUMP.npz ...] [--sph ...] [--edges E0 E1 ...] [--span A B]
    (PREFIX_parts.json "-": flow only, no torque)
"""
import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _analyze_axisymmetric_energy as axi   # noqa: E402
import _analyze_rushton_lower_flow as lower  # noqa: E402

DENSITY = 998.0
OMEGA = 2.0 * np.pi * 200.0 / 60.0
EDGES = np.array([0.0, 0.0122, 0.016, 0.024, 0.032, 0.040, 0.050, 0.060])
FLUENT_SPAN = (0.00756, 0.0480)
CHORD = 0.0248
ZONES = {"blade zone": (0.186, 0.203), "above": (0.205, 0.213), "below": (0.176, 0.184)}


def band_names():
    return [f"r {a * 1e3:4.1f}..{b * 1e3:4.1f} mm" for a, b in zip(EDGES[:-1], EDGES[1:])]


def fluent_torque(path):
    with open(path) as handle:
        names = [name.strip() for name in handle.readline().split(",")]
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    c = {name: data[:, k] for k, name in enumerate(names)}
    x, z = c["x-coordinate"], c["z-coordinate"]
    area = np.stack([c["x-face-area"], c["y-face-area"], c["z-face-area"]], axis=1)
    size = np.linalg.norm(area, axis=1)
    normal = area / np.maximum(size, 1e-30)[:, None]
    force = c["pressure"][:, None] * area + np.stack([c["x-wall-shear"], c["y-wall-shear"], c["z-wall-shear"]], axis=1) * size[:, None]
    r = np.maximum(np.hypot(x, z), 1e-12)
    e_theta = np.stack([z / r, np.zeros_like(r), -x / r], axis=1)
    torque = r * (force * e_theta).sum(axis=1)
    rotation = -np.sign(torque.sum()) or 1.0
    resistance = -rotation * torque
    n_theta = (normal * e_theta).sum(axis=1)
    blade = (np.abs(n_theta) > 0.3) & (r > 0.008)
    front = blade & (rotation * n_theta < 0)
    out = {}
    index = np.clip(np.digitize(r, EDGES) - 1, 0, len(EDGES) - 2)
    for side, mask in (("front", front), ("back", blade & ~front)):
        out[side] = np.bincount(index[mask], weights=resistance[mask], minlength=len(EDGES) - 1)
    out["other"] = resistance[~blade].sum()
    out["total"] = resistance.sum()
    return out


def sph_torque(path):
    if path == "-":                      # flow only (a run without the band split)
        nothing = np.full(len(EDGES) - 1, np.nan)
        return {"front": nothing, "back": nothing.copy(), "other": np.nan, "total": np.nan}
    window = json.load(open(path))["windows"]
    parts = window["0.3..0.7"] if "0.3..0.7" in window else next(iter(window.values()))
    out = {"front": np.zeros(len(EDGES) - 1), "back": np.zeros(len(EDGES) - 1)}
    for k, (a, b) in enumerate(zip(EDGES[:-1], EDGES[1:])):
        band = f"r {a * 1e3:4.1f}..{b * 1e3:4.1f} mm"
        if f"PBT solid blade, {band}" in parts:          # blades of ordinary particles (--conformal-pbt): net only
            out["front"][k] = parts[f"PBT solid blade, {band}"]
            continue
        for side in ("front", "back"):
            out[side][k] = parts[f"PBT blade {side}, {band}"]
    out["other"] = parts.get("PBT hub and collar (lattice)", np.nan)
    out["total"] = out["front"].sum() + out["back"].sum() + out["other"]
    return out


def band_flow(snapshots):
    """count-weighted azimuthal means of u_theta, u_y, u_r per band and zone, over the snapshots"""
    sums = {zone: np.zeros((4, len(EDGES) - 1)) for zone in ZONES}
    for x, v in snapshots:
        r = np.maximum(np.hypot(x[:, 0], x[:, 2]), 1e-12)
        u_theta = (v[:, 0] * x[:, 2] - v[:, 2] * x[:, 0]) / r        # along the rotation (blades move along +e_theta)
        u_r = (v[:, 0] * x[:, 0] + v[:, 2] * x[:, 2]) / r
        index = np.clip(np.digitize(r, EDGES) - 1, 0, len(EDGES) - 2)
        for zone, (y0, y1) in ZONES.items():
            m = (x[:, 1] >= y0) & (x[:, 1] < y1) & (r < EDGES[-1])
            sums[zone][0] += np.bincount(index[m], minlength=len(EDGES) - 1)
            sums[zone][1] += np.bincount(index[m], weights=u_theta[m], minlength=len(EDGES) - 1)
            sums[zone][2] += np.bincount(index[m], weights=v[m, 1], minlength=len(EDGES) - 1)
            sums[zone][3] += np.bincount(index[m], weights=u_r[m], minlength=len(EDGES) - 1)
    return {zone: {"u_theta": s[1] / np.maximum(s[0], 1), "u_y": s[2] / np.maximum(s[0], 1),
                   "u_r": s[3] / np.maximum(s[0], 1), "count": s[0]} for zone, s in sums.items()}


def fluent_snapshots(paths):
    for path in paths:
        cells = axi.read_interpolation_file(path)
        x = np.stack([cells["x"], cells["y"] - axi.FLUENT_Y_SHIFT, cells["z"]], axis=1)
        v = np.stack([cells["x-velocity"], cells["y-velocity"], cells["z-velocity"]], axis=1)
        rotor = lower.in_fluent_rotor_zones(x)
        radius = np.maximum(np.hypot(x[rotor, 0], x[rotor, 2]), 1e-12)
        sense = np.sign(((v[rotor, 0] * x[rotor, 2] - v[rotor, 2] * x[rotor, 0]) / radius).sum()) or 1.0
        v[rotor] = lower.turn_about_y(v[rotor], sense * np.radians(12.0))
        if sense < 0:                       # mirrored snapshot: mirror it back so that it turns along +e_theta
            x[:, 2] *= -1.0
            v[:, 2] *= -1.0
        yield x, v


def main():
    global EDGES
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fluent", nargs=2, action="append", default=[], metavar=("PBT_CSV", "CELLS_IP"))
    parser.add_argument("--sph", nargs="+", action="append", default=[], metavar="LABEL PARTS_JSON DUMP")
    parser.add_argument("--span", type=float, nargs=2, default=(0.0122, 0.0482), help="SPH blade span, m")
    parser.add_argument("--edges", type=float, nargs="+", default=None, help="band edges, m (as in the runs)")
    arguments = parser.parse_args()
    if arguments.edges:
        EDGES = np.asarray(arguments.edges)

    sets = []
    if arguments.fluent:
        torques = [fluent_torque(csv) for csv, _ in arguments.fluent]
        torque = {k: np.mean([t[k] for t in torques], axis=0) for k in torques[0]}
        flow = band_flow(fluent_snapshots([ip for _, ip in arguments.fluent]))
        sets.append((f"Fluent ({len(arguments.fluent)})", torque, flow, FLUENT_SPAN))
    for entry in arguments.sph:
        label, parts, dumps = entry[0], entry[1], entry[2:]
        flow = band_flow(axi.sph_arrays(p) for p in dumps)
        sets.append((f"{label} ({len(dumps)} dumps)", sph_torque(parts), flow, tuple(arguments.span)))

    r_mid = 0.5 * (EDGES[1:] + EDGES[:-1])

    def blade_area(span):
        return 6.0 * CHORD * np.clip(np.minimum(EDGES[1:], span[1]) - np.maximum(EDGES[:-1], span[0]), 0.0, None)

    print("PBT resistance torque by radius, mN m: front (pressure side) / back / net")
    print("  (front and back each carry the reference pressure times their area; only the net is physical)")
    print(f"  {'band':18s}" + "".join(f"{s[0][:30]:>32s}" for s in sets))
    for k, name in enumerate(band_names()):
        print(f"  {name:18s}" + "".join(f"{t['front'][k] * 1e3:10.2f}{t['back'][k] * 1e3:10.2f}"
                                        f"{(t['front'][k] + t['back'][k]) * 1e3:10.2f}  " for _, t, _, _ in sets))
    print(f"  {'blades':18s}" + "".join(f"{t['front'].sum() * 1e3:10.2f}{t['back'].sum() * 1e3:10.2f}"
                                       f"{(t['front'].sum() + t['back'].sum()) * 1e3:10.2f}  " for _, t, _, _ in sets))
    print(f"  {'hub, sleeve, edges':18s}" + "".join(f"{'':20s}{t['other'] * 1e3:10.2f}  " for _, t, _, _ in sets))
    print(f"  {'total':18s}" + "".join(f"{'':20s}{t['total'] * 1e3:10.2f}  " for _, t, _, _ in sets))

    for zone in ZONES:
        print(f"\nazimuthal means, {zone} (y {ZONES[zone][0] * 1e3:.0f}..{ZONES[zone][1] * 1e3:.0f} mm): "
              "u_theta / (omega r), u_y, u_r m/s")
        for k, name in enumerate(band_names()):
            cells = []
            for _, _, flow, _ in sets:
                f = flow[zone]
                cells.append(f"{f['u_theta'][k] / (OMEGA * r_mid[k]):8.3f}{f['u_y'][k]:8.3f}{f['u_r'][k]:8.3f}"
                             if f["count"][k] > 0 else f"{'':24s}")
            print(f"  {name:18s}" + "".join(f"{c:>32s}" for c in cells))

    print("\nblade zone: relative velocity W (m/s), relative flow angle phi and angle of attack alpha (deg), "
          "coefficient C = net torque / (1/2 rho W^2 r A)")
    print(f"  {'band':18s}" + "".join(f"{s[0][:30]:>36s}" for s in sets))
    print(f"  {'':18s}" + "".join(f"{'W     phi   alpha       C':>36s}" for _ in sets))
    for k, name in enumerate(band_names()):
        cells = []
        for _, t, flow, span in sets:
            area = blade_area(span)[k]
            u_theta, u_y = flow["blade zone"]["u_theta"][k], flow["blade zone"]["u_y"][k]
            w_t, w_y = OMEGA * r_mid[k] - u_theta, u_y
            w2 = w_t ** 2 + w_y ** 2
            phi = np.degrees(np.arctan2(-w_y, w_t))
            c = (t["front"][k] + t["back"][k]) / (0.5 * DENSITY * w2 * r_mid[k] * area) if area > 0 else np.nan
            cells.append(f"{np.sqrt(w2):6.3f}{phi:7.1f}{45.0 - phi:7.1f}{c:9.3f}")
        print(f"  {name:18s}" + "".join(f"{c:>36s}" for c in cells))


if __name__ == "__main__":
    main()
