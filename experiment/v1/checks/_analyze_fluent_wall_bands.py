"""
_analyze_fluent_wall_bands.py — torque of the fluid on the Fluent tank walls split by part and height
(2026-10-03, base Anaconda).

Input: ASCII exports of a wall zone per snapshot (face values) with x/y/z-wall-shear, pressure and
x/y/z-face-area, e.g. written by
    /file/export/ascii "fineNN_walls.csv" walls () yes x-wall-shear y-wall-shear z-wall-shear pressure
                       x-face-area y-face-area z-face-area () yes
Per face: shear force tau_w |A|, pressure force p A; torque about +y (z F_x - x F_z). Each snapshot is oriented
so that the total viscous torque is positive (the mirrored fine snapshots 26..34 s turn the other way).
Fluent y is shifted by -58.5 mm into the generator frame. Two splits:
  generator-like: "floor" = y <= 0 (dished bottom and the lowest 12 mm of the cylinder), cylinder bands for
                  r > 139.5 mm and 0 < y < 426.5 mm, the same classes as _check_tank_energy.py;
  geometric:      dished bottom (r < 142.5 mm) and the cylinder (r >= 142.5 mm) by height.

usage: python experiment/v1/checks/_analyze_fluent_wall_bands.py fine25_walls.csv [fine26_walls.csv ...]
"""
import sys

import numpy as np

FLUENT_Y_SHIFT = 0.0585


def load(path):
    with open(path) as handle:
        names = [name.strip() for name in handle.readline().split(",")]
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    return {name: data[:, k] for k, name in enumerate(names)}


def face_torques(columns):
    x, y, z = columns["x-coordinate"], columns["y-coordinate"] - FLUENT_Y_SHIFT, columns["z-coordinate"]
    area_vector = np.stack([columns["x-face-area"], columns["y-face-area"], columns["z-face-area"]], axis=1)
    area = np.linalg.norm(area_vector, axis=1)
    shear = np.stack([columns["x-wall-shear"], columns["y-wall-shear"], columns["z-wall-shear"]], axis=1) * area[:, None]
    pressure = columns["pressure"][:, None] * area_vector
    torque = lambda force: z * force[:, 0] - x * force[:, 2]
    return x, y, z, torque(shear), torque(pressure)


BANDS = {"generator-like split": [
             ("floor, y <= 0", lambda r, y: y <= 0.0),
             ("cylinder 0 < y < 1 mm", lambda r, y: (r > 0.1395) & (y > 0.0) & (y < 0.001)),
             ("cylinder 1 .. 60 mm", lambda r, y: (r > 0.1395) & (y >= 0.001) & (y < 0.060)),
             ("cylinder 60 .. 250 mm", lambda r, y: (r > 0.1395) & (y >= 0.060) & (y < 0.250)),
             ("cylinder 250 .. 426.5 mm", lambda r, y: (r > 0.1395) & (y >= 0.250) & (y < 0.4265))],
         "geometric split": [
             ("dished bottom, r < 142.5 mm", lambda r, y: r < 0.1425),
             ("cylinder below the baffles, y < 1 mm", lambda r, y: (r >= 0.1425) & (y < 0.001)),
             ("cylinder 1 .. 60 mm", lambda r, y: (r >= 0.1425) & (y >= 0.001) & (y < 0.060)),
             ("cylinder 60 .. 250 mm", lambda r, y: (r >= 0.1425) & (y >= 0.060) & (y < 0.250)),
             ("cylinder above 250 mm", lambda r, y: (r >= 0.1425) & (y >= 0.250))]}


def main(paths):
    rows = []
    for path in paths:
        x, y, z, viscous, pressure = face_torques(load(path))
        sign = np.sign(viscous.sum()) or 1.0
        r = np.hypot(x, z)
        row = {"total viscous": sign * viscous.sum(), "total pressure": sign * pressure.sum()}
        for split, bands in BANDS.items():
            for name, select in bands:
                mask = select(r, y)
                row[(split, name, "viscous")] = sign * viscous[mask].sum()
                row[(split, name, "pressure")] = sign * pressure[mask].sum()
        rows.append(row)
    mean = lambda key: 1e3 * np.mean([row[key] for row in rows])
    print(f"{len(rows)} snapshots; torque about +y in mN m, positive = the wall takes angular momentum out of the swirl")
    print(f"  whole zone: viscous {mean('total viscous'):7.2f}, pressure {mean('total pressure'):7.2f}")
    for split, bands in BANDS.items():
        print(f"  {split}:                               viscous  pressure   total")
        for name, _ in bands:
            v, p = mean((split, name, "viscous")), mean((split, name, "pressure"))
            print(f"    {name:40s} {v:7.2f}  {p:7.2f}  {v + p:7.2f}")


if __name__ == "__main__":
    main(sys.argv[1:])
