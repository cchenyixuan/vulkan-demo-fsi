"""
_analyze_fluent_impeller_parts.py — where the torque sits on Fluent's Rushton (zone rt), 2026-10-03.

Input: ASCII exports of the wall zone rt per snapshot (face values, x/y/z-wall-shear, pressure, x/y/z-face-area),
written like _analyze_fluent_wall_bands.py's inputs. Per face: force p A + tau_w |A| (A points from the fluid
into the solid), torque about +y = r F_theta, e_theta = (z, 0, -x) / r. Faces are classified by position and
normal (generator frame, Fluent y - 58.5 mm; Rushton blades r 24..48 mm, y 28.8..48.0 mm; disk r < 32 mm,
y 37.2..39.7 mm; hub r 10.2 mm):
  blade faces (|n . e_theta| > 0.8): front (pushing, n . v < 0) or back, above the disk (y > 39.7 mm),
  below it (y < 37.2 mm) or level with it; blade edges; disk top / bottom / rim; hub; the rest (shaft, sleeve).
Each snapshot is oriented by its rotation sense (the mirrored fine snapshots turn at -200 rpm) so that the
resistance of the fluid is positive.

usage: python experiment/v1/checks/_analyze_fluent_impeller_parts.py fine25_rt.csv [fine26_rt.csv ...]
"""
import sys

import numpy as np

FLUENT_Y_SHIFT = 0.0585
DISK_Y0, DISK_Y1, DISK_R = 0.0372, 0.0397, 0.032


def load(path):
    with open(path) as handle:
        names = [name.strip() for name in handle.readline().split(",")]
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    return {name: data[:, k] for k, name in enumerate(names)}


def classify(columns):
    x, y, z = columns["x-coordinate"], columns["y-coordinate"] - FLUENT_Y_SHIFT, columns["z-coordinate"]
    area_vector = np.stack([columns["x-face-area"], columns["y-face-area"], columns["z-face-area"]], axis=1)
    area = np.linalg.norm(area_vector, axis=1)
    normal = area_vector / np.maximum(area, 1e-30)[:, None]
    force = columns["pressure"][:, None] * area_vector + \
        np.stack([columns["x-wall-shear"], columns["y-wall-shear"], columns["z-wall-shear"]], axis=1) * area[:, None]
    shear_only = np.stack([columns["x-wall-shear"], columns["y-wall-shear"], columns["z-wall-shear"]], axis=1) * area[:, None]
    r = np.maximum(np.hypot(x, z), 1e-12)
    e_theta = np.stack([z / r, np.zeros_like(r), -x / r], axis=1)
    e_r = np.stack([x / r, np.zeros_like(r), z / r], axis=1)
    torque = r * (force * e_theta).sum(axis=1)
    torque_shear = r * (shear_only * e_theta).sum(axis=1)
    rotation = -np.sign(torque.sum()) or 1.0          # +1: the impeller turns along +e_theta
    n_theta, n_r, n_y = (normal * e_theta).sum(axis=1), (normal * e_r).sum(axis=1), normal[:, 1]
    classes = np.full(x.shape[0], "other", dtype=object)
    blade_face = (np.abs(n_theta) > 0.8) & (r > 0.023) & (r < 0.0485) & (y > 0.028) & (y < 0.0485)
    front = blade_face & (rotation * n_theta < 0)
    for side, mask in (("front", front), ("back", blade_face & ~front)):
        classes[mask & (y > DISK_Y1)] = f"blade {side}, above disk"
        classes[mask & (y < DISK_Y0)] = f"blade {side}, below disk"
        classes[mask & (y >= DISK_Y0) & (y <= DISK_Y1)] = f"blade {side}, disk level"
    edge = ~blade_face & (r > 0.0235) & (r < 0.0485) & (y > 0.028) & (y < 0.0485) & ((np.abs(n_r) > 0.8) | (np.abs(n_y) > 0.8)) & (r > DISK_R + 0.0005)
    classes[edge] = "blade edges (tip, top, bottom)"
    disk = ~blade_face & ~edge & (r < DISK_R + 0.0005) & (y > DISK_Y0 - 0.0008) & (y < DISK_Y1 + 0.0008) & (r > 0.0105)
    classes[disk & (np.abs(n_y) > 0.8) & (n_y < 0)] = "disk top"
    classes[disk & (np.abs(n_y) > 0.8) & (n_y > 0)] = "disk bottom"
    classes[disk & (np.abs(n_r) > 0.8)] = "disk rim"
    hub = (classes == "other") & (r < 0.0125) & (y > 0.019) & (y < 0.0435)
    classes[hub] = "hub"
    return classes, -rotation * torque, -rotation * torque_shear


def main(paths):
    totals = {}
    for path in paths:
        classes, torque, torque_shear = classify(load(path))
        for name in np.unique(classes):
            mask = classes == name
            entry = totals.setdefault(name, [])
            entry.append((torque[mask].sum(), torque_shear[mask].sum(), mask.sum()))
        totals.setdefault("TOTAL", []).append((torque.sum(), torque_shear.sum(), len(torque)))
    print(f"{len(paths)} snapshots; resistance torque on the Rushton zone rt, mN m (positive = against the rotation)")
    print("  part                                 total   of which shear   faces")
    order = sorted(totals, key=lambda name: (name == "TOTAL", name))
    for name in order:
        values = np.array(totals[name])
        print(f"  {name:34s} {values[:, 0].mean() * 1e3:7.2f}   {values[:, 1].mean() * 1e3:7.2f}       {values[:, 2].mean():7.0f}")


if __name__ == "__main__":
    main(sys.argv[1:])
