"""
_demo_stirred_tank_30l.py — particle case for the 30 L stirred tank of
Rautenbach et al. (2026), Comput. Chem. Eng., dataset DARUS-5523.

All dimensions were measured from the dataset's STL files
(00_simulation_for_mstar_video/StaticBody.stl, Moving Body_1.stl) on
2026-09-25; see log/ for the measurement notes. Axis convention follows the
STL: y is the tank axis (up). The origin is where the dished bottom meets
the cylindrical wall's y = 0 level; the dish itself lies below y = 0
(corrected 2026-09-26, the first version modelled a flat floor at y = 0).

Geometry (metres):
    tank        inner radius 0.144, liquid height 0.4265 (the M-Star run uses a
                flat lid there, no free surface)
    bottom      dished (torispherical): FLOOR_PROFILE below, measured from
                StaticBody.stl; -0.0618 at r = 0.02 (crown radius ~0.27 m),
                -0.0333 at r = 0.126, knuckle up to -0.012 at the wall r = 0.144
    bearing     static boss r 0.0145 on the dish centre, top at y = -0.023
    baffles     3 (120 deg apart, azimuths 62/182/302 deg from +x toward +z),
                radial 0.1254..0.1384, thickness 2.65e-3, y 0.010 .. lid
    shaft       radius 0.004, through floor and lid
    Rushton     hub r 0.0102 y 0.0211..0.0414; disk r 0.032 y 0.0372..0.0397;
                6 blades radial 0.024..0.048, y 0.0288..0.048, thickness 2.2e-3,
                azimuths 23.1 + 60 k deg
    PBT         hub r 0.0109 y 0.180..0.2093; 6 blades pitched 45 deg, radial
                0.0122..0.0491, chord 0.0273, thickness 2.5e-3, centre y 0.19465,
                azimuths 24.3 + 60 k deg, y decreases toward +theta
    probes      2 square rods 15 mm x 15 mm, y 0.406..0.456, at
                (x, z) = (-0.127, -0.018) and (0.053, -0.118)

Every solid thinner than THIN_LAYERS particle spacings is thickened to that
many layers (blades, disk, baffles), otherwise fluid particles pass through.

Materials: fluid / wall / rotor come from the case-local materials.yaml that
this script also writes (rotor = shaft + both impellers, one rigid body).

Usage (repo root):
    python utils/geometry/_demo_stirred_tank_30l.py --dx 0.003 --hdx 3
    python utils/geometry/_demo_stirred_tank_30l.py --dx 0.002 --hdx 3 --out cases/stirred_tank_30l_2mm
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.geometry import LATTICE_GRID, tile_bounding_box          # noqa: E402
from utils.geometry.region import Region, Box, Cylinder, Union, Intersection, Difference      # noqa: E402

# ----------------------------------------------------------------------------
# Measured dimensions (m)
# ----------------------------------------------------------------------------
TANK_RADIUS = 0.144
LIQUID_HEIGHT = 0.4265
TANK_WALL_TOP = 0.5

# Dished bottom (2026-09-26): inner floor height y(r), measured from
# StaticBody.stl (max vertex height per radial band, away from the baffles).
# Piecewise-linear in r; the centre (r < 0.02) is covered by the bearing boss.
FLOOR_PROFILE = np.array([
    (0.0000, -0.0620), (0.0225, -0.0618), (0.0325, -0.0606), (0.0425, -0.0592),
    (0.0525, -0.0579), (0.0625, -0.0554), (0.0725, -0.0529), (0.0825, -0.0509),
    (0.0925, -0.0470), (0.1025, -0.0435), (0.1125, -0.0402), (0.1210, -0.0366),
    (0.1262, -0.0333), (0.1312, -0.0308), (0.1362, -0.0256), (0.1388, -0.0236),
    (0.1412, -0.0169), (0.1440, -0.0120),
])
FLOOR_BOTTOM = float(FLOOR_PROFILE[:, 1].min())
BEARING_BOSS = dict(radius=0.0145, y1=-0.023)
# Baffles, measured 2026-09-29 on sections of the dataset's static body
# (30L_saticbody.stl, mm, same origin): three plates 24.0 mm wide (T / 12) from
# r = 114.35 to 138.36 mm, 2.55-2.63 mm thick, in planes through the axis at
# 61.45 / 181.45 / 301.45 deg, from y = 1 mm to above the liquid level; each is
# fixed to the wall by a bracket at its lower end (y = 1.0 .. 20.7 mm): 9.2 mm
# wide from r = 128 mm, widening with a fillet to 30.6 mm at the wall. The
# bracket is modelled by three boxes (BAFFLE_FOOT: r0, r1, width).
# Until 2026-09-29 the generator used r = 125.4 .. 138.4 mm (13 mm wide, 54 % of
# the true width), 62 / 182 / 302 deg, 2.65 mm, from y = 10 mm, no block:
# LEGACY_BAFFLES, generator flag --legacy-baffles. Every tank run before that
# date has the narrow baffles.
BAFFLE_Y0 = 0.001
BAFFLE_AZIMUTHS_DEG = (61.45, 181.45, 301.45)
BAFFLE_RADIAL = (0.11435, 0.13836)
BAFFLE_THICKNESS = 2.6e-3
BAFFLE_FOOT = dict(y0=0.001, y1=0.0207,
                   parts=((0.1280, 0.1370, 0.0096), (0.1370, 0.1395, 0.0180), (0.1395, 0.1445, 0.0306)))
LEGACY_BAFFLES = dict(y0=0.010, azimuths_deg=(62.0, 182.0, 302.0), radial=(0.1254, 0.1384), thickness=2.65e-3)

SHAFT_RADIUS = 0.004

# Rotating bell at the lower end of the shaft, measured 2026-09-29 on sections of the dataset's
# 'Moving Body_1.stl': a body of revolution that covers the static bearing. Outer profile (y, r):
# sleeve of radius 8 mm from y = 25 down to 14 mm, cone to r = 21 mm at y = -26 mm, cylinder of
# radius 21 mm down to y = -57 mm, chamfered rim, lower end at y = -58.3 mm (3.7 mm above the
# floor). It is hollow (the static bearing sits inside, clearance below 1 mm); here it is a solid
# of revolution, the static bearing boss inside it is dropped. Until 2026-09-29 the rotor had only
# the shaft of radius 4 mm here: generator flag --no-rotor-bell.
ROTOR_BELL_PROFILE = np.array([
    (-0.0583, 0.0206), (-0.0570, 0.0210), (-0.0260, 0.0210), (0.0140, 0.0080), (0.0250, 0.0080),
])

RUSHTON_HUB = dict(radius=0.0102, y0=0.0211, y1=0.0414)
RUSHTON_DISK = dict(radius=0.032, y0=0.0372, y1=0.0397)
RUSHTON_BLADE = dict(radial=(0.024, 0.048), y0=0.0288, y1=0.048, thickness=2.2e-3, azimuth0_deg=23.1)

# PBT corrected 2026-10-02 (sections of 'Moving Body_1.stl', log/2026-10-02_pbt-geometry.md): the blades are
# flat 2.5 mm plates at 45 deg whose mid-surface chord is 24.8 mm. Until then the chord was 27.3 mm =
# the blades' total height 19.3 mm / sin 45 deg, which counts the thickness (2.5 mm cos 45 deg = 1.8 mm
# of the height) as chord: the thin-plate blades were 10 % too wide. The hub is r 10.87 mm only over
# the blade band y 184.98 .. 204.25 mm, with a collar of r 7.5 mm from 180.0 to 209.3 mm; until then
# one cylinder r 10.9 mm from 180.0 to 209.3 mm. --legacy-pbt restores the old values.
PBT_HUB = dict(radius=0.01087, y0=0.18498, y1=0.20425)
PBT_COLLAR = dict(radius=0.0075, y0=0.1800, y1=0.2093)
LEGACY_PBT_HUB = dict(radius=0.0109, y0=0.180, y1=0.2093)
LEGACY_PBT_CHORD = 0.0273
PBT_BLADE = dict(radial=(0.0122, 0.0491), chord=0.0248, thickness=2.5e-3,
                 center_y=0.19465, pitch_deg=45.0, azimuth0_deg=24.3)

# The hubs of the Fluent LES mesh (2026-10-03, wall zones rt / pbt / stator of the fine 25 s snapshot, Fluent
# y - 58.5 mm; log/2026-10-03_smooth-walls.md) are thinner than those of the M-Star STL above: the Rushton hub
# is the r 7.97 mm sleeve from the bell (y 18.6 mm) to the disk (37.1 mm), only the shaft above the disk; the PBT
# sits on a sleeve r 7.56 mm from y 175.4 to 207.0 mm and its blades reach in to it (pitched faces from r 7.8 mm)
# and out to r 48.0 mm (faces up to 47.96 mm). Rushton blades and disk, bell and shaft agree with the STL.
# Generator flag --fluent-hubs.
FLUENT_RUSHTON_HUB = dict(radius=0.00797, y0=0.0186, y1=0.0372)
FLUENT_PBT_HUB = dict(radius=0.00756, y0=0.1754, y1=0.2070)
FLUENT_PBT_RADIAL = (0.00756, 0.0480)

# Blade tips measured on the dataset's 'Moving Body_1.stl' (2026-09-28): both blades end in a
# straight cut. Rushton: cut at 48.0 mm from the axis, corners at 48.0 mm. PBT: cut at about
# 48.2 mm along the blade centre line, its corners (the blade is 19 mm wide seen from above)
# at the largest radius 49.1 mm. PBT_BLADE["radial"][1] = 49.1 mm is that largest radius, so
# the box-shaped model blade is 0.9 mm too long and its corners reach 50.0 mm (50.7 mm when
# thickened to 9 mm). --clip-tips uses the true length and cuts the blades at the true radius.
RUSHTON_TIP_RADIUS = 0.0480
PBT_TIP_RADIUS = 0.0491
PBT_TRUE_LENGTH = 0.0482

PROBE_HALF = 0.0075
PROBE_Y = (0.406, 0.456)
PROBE_CENTERS = ((-0.127, -0.018), (0.053, -0.118))

# Measurement and tracer-injection points of Rautenbach et al. (2026), Table 1
# (same frame as the dataset STL: y up, origin on the axis at y = 0). The
# measurement points sit 4.5 mm below the probe rods above. Injection volume in
# the experiments: 5 mL, i.e. a sphere of radius (3 * 5e-6 / (4 pi))^(1/3).
PROBE_POINTS = (("probe_1", (-0.1270, 0.4015, -0.0180)),
                ("probe_2", ( 0.0530, 0.4015, -0.1180)))
INJECTION_POINT = (0.0, 0.4055, 0.1164)
INJECTION_RADIUS = (3.0 * 5.0e-6 / (4.0 * math.pi)) ** (1.0 / 3.0)   # 0.0106 m

IMPELLER_RPM = 200.0                      # M-Star: -200 rpm about +y = CCW seen from above (see comment in CASE_YAML)
TIP_SPEED = math.pi * 2 * 0.0491 * IMPELLER_RPM / 60.0


# ----------------------------------------------------------------------------
# Extra region: liquid volume of the dished tank (2026-09-26)
# ----------------------------------------------------------------------------
class RevolvedProfile(Region):
    """Solid of revolution about the y axis: r < radius(y) for y inside the profile's range,
    radius(y) piecewise linear through the points (y, r). Approximate signed distance (max of
    the one-sided distances, the radial one projected on the local surface normal)."""

    def __init__(self, profile):
        self.profile = np.asarray(profile, dtype=np.float64)
        self.dimension = 3

    def signed_distance(self, points):
        pts = np.asarray(points, dtype=np.float64)
        r = np.hypot(pts[:, 0], pts[:, 2])
        y = pts[:, 1]
        radius = np.interp(y, self.profile[:, 0], self.profile[:, 1])
        slope = np.interp(y, self.profile[:, 0], np.gradient(self.profile[:, 1], self.profile[:, 0]))
        d_side = (r - radius) / np.sqrt(1.0 + slope ** 2)
        return np.maximum(np.maximum(d_side, self.profile[0, 0] - y), y - self.profile[-1, 0])

    def bounds(self):
        radius = self.profile[:, 1].max()
        return (np.array([-radius, self.profile[0, 0], -radius]), np.array([radius, self.profile[-1, 0], radius]))


class DishedTankInterior(Region):
    """Liquid volume: r < radius, floor(r) < y < top, with floor(r) the
    piecewise-linear FLOOR_PROFILE. Approximate SDF = max of the three
    one-sided distances, the floor term projected on the local floor normal
    (exact on the flat and gently sloped parts, conservative in corners),
    which is all the generator needs (half-spacing skins, shell thickness)."""

    def __init__(self, radius, top, profile):
        self.radius = float(radius)
        self.top = float(top)
        self.profile = np.asarray(profile, dtype=np.float64)
        self.dimension = 3

    def floor(self, r):
        return np.interp(r, self.profile[:, 0], self.profile[:, 1])

    def signed_distance(self, points):
        pts = np.asarray(points, dtype=np.float64)
        r = np.hypot(pts[:, 0], pts[:, 2])
        y = pts[:, 1]
        slope = np.gradient(self.profile[:, 1], self.profile[:, 0])
        local_slope = np.interp(r, self.profile[:, 0], slope)
        d_floor = (self.floor(r) - y) / np.sqrt(1.0 + local_slope ** 2)
        d_side = r - self.radius
        d_top = y - self.top
        return np.maximum(np.maximum(d_side, d_top), d_floor)

    def bounds(self):
        lo = np.array([-self.radius, self.profile[:, 1].min(), -self.radius])
        hi = np.array([self.radius, self.top, self.radius])
        return lo, hi

    def volume(self):
        r = np.linspace(0.0, self.radius, 4001)
        depth = self.top - self.floor(r)
        return float(np.trapezoid(2.0 * np.pi * r * depth, r))


# ----------------------------------------------------------------------------
# Extra region: a box in a rotated frame (radial / in-blade / normal axes)
# ----------------------------------------------------------------------------
class OrientedBox(Region):
    """Box with centre ``center`` and orthonormal axes rows ``axes`` (3x3);
    half-extents ``half`` along those axes. Exact SDF."""

    def __init__(self, center, axes, half):
        self.center = np.asarray(center, dtype=np.float64)
        self.axes = np.asarray(axes, dtype=np.float64)
        self.half = np.asarray(half, dtype=np.float64)
        self.dimension = 3

    def signed_distance(self, points):
        local = (np.asarray(points, dtype=np.float64) - self.center) @ self.axes.T
        q = np.abs(local) - self.half
        outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
        inside = np.minimum(np.max(q, axis=1), 0.0)
        return outside + inside

    def bounds(self):
        corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * self.half
        world = corners @ self.axes + self.center
        return world.min(axis=0), world.max(axis=0)


def radial_frame(azimuth_deg):
    """Unit vectors (e_r, e_theta, e_y) at azimuth measured from +x toward +z."""
    a = math.radians(azimuth_deg)
    e_r = np.array([math.cos(a), 0.0, math.sin(a)])
    e_t = np.array([-math.sin(a), 0.0, math.cos(a)])
    e_y = np.array([0.0, 1.0, 0.0])
    return e_r, e_t, e_y


def y_cylinder(radius, y0, y1):
    return Cylinder([0.0, y0, 0.0], [0.0, y1 - y0, 0.0], radius)


def radial_slab(azimuth_deg, r0, r1, y0, y1, thickness):
    """Vertical plate in the plane containing the axis (baffle, Rushton blade)."""
    e_r, e_t, e_y = radial_frame(azimuth_deg)
    center = e_r * 0.5 * (r0 + r1) + e_y * 0.5 * (y0 + y1)
    return OrientedBox(center, np.stack([e_r, e_y, e_t]),
                       [0.5 * (r1 - r0), 0.5 * (y1 - y0), 0.5 * thickness])


def pitched_blade(azimuth_deg, r0, r1, chord, thickness, center_y, pitch_deg):
    """PBT blade: plate rotated about the radial axis so that y decreases toward +theta."""
    e_r, e_t, e_y = radial_frame(azimuth_deg)
    p = math.radians(pitch_deg)
    e_chord = math.cos(p) * e_t - math.sin(p) * e_y      # in-blade width direction
    e_norm = math.sin(p) * e_t + math.cos(p) * e_y       # blade normal
    center = e_r * 0.5 * (r0 + r1) + e_y * center_y
    return OrientedBox(center, np.stack([e_r, e_chord, e_norm]),
                       [0.5 * (r1 - r0), 0.5 * chord, 0.5 * thickness])


def blade_frames(impellers="both"):
    """The twelve blades as plane rectangles with their TRUE outline:
    (name, centre, in-plane axis a (radial), in-plane axis b, normal, half extent a, half extent b).
    The PBT uses its true length (PBT_TRUE_LENGTH, straight cut), so its corners lie at the
    true tip radius without any clipping."""
    frames = []
    for k in range(6):
        if impellers in ("both", "rushton"):
            e_r, e_t, e_y = radial_frame(RUSHTON_BLADE["azimuth0_deg"] + 60 * k)
            r0, r1 = RUSHTON_BLADE["radial"]
            y0, y1 = RUSHTON_BLADE["y0"], RUSHTON_BLADE["y1"]
            frames.append((f"rushton_{k + 1}", e_r * 0.5 * (r0 + r1) + e_y * 0.5 * (y0 + y1),
                           e_r, e_y, e_t, 0.5 * (r1 - r0), 0.5 * (y1 - y0)))
        if impellers in ("both", "pbt"):
            e_r, e_t, e_y = radial_frame(PBT_BLADE["azimuth0_deg"] + 60 * k)
            pitch = math.radians(PBT_BLADE["pitch_deg"])
            e_chord = math.cos(pitch) * e_t - math.sin(pitch) * e_y
            e_norm = math.sin(pitch) * e_t + math.cos(pitch) * e_y
            r0, r1 = PBT_BLADE["radial"][0], PBT_TRUE_LENGTH
            frames.append((f"pbt_{k + 1}", e_r * 0.5 * (r0 + r1) + e_y * PBT_BLADE["center_y"],
                           e_r, e_chord, e_norm, 0.5 * (r1 - r0), 0.5 * PBT_BLADE["chord"]))
    return frames


def conformal_blades(dx, layers, impellers="both"):
    """Blade particles on a plane grid in each blade's own frame (2026-09-28).

    Taking the blades from the global lattice gives a staircase wherever a blade is not
    parallel to the lattice; a single-layer blade leaks at every step of that staircase and
    nowhere else (log/2026-09-28_single-layer-blades.md). Here each blade is `layers` flat
    layers of particles, spacing <= dx in the two in-plane directions (the number of cells is
    rounded up, so the outline is exact and the sheet slightly denser than the lattice), the
    outermost centres half a cell inside the true outline.
    Returns (points, boxes): the particle positions and one OrientedBox per blade with the
    true outline and thickness layers * dx; lattice sites inside a box are dropped."""
    points, boxes = [], []
    for _, centre, axis_a, axis_b, normal, half_a, half_b in blade_frames(impellers):
        cells_a = max(1, int(math.ceil(2.0 * half_a / dx - 1e-9)))
        cells_b = max(1, int(math.ceil(2.0 * half_b / dx - 1e-9)))
        a = -half_a + (np.arange(cells_a) + 0.5) * (2.0 * half_a / cells_a)
        b = -half_b + (np.arange(cells_b) + 0.5) * (2.0 * half_b / cells_b)
        n = (np.arange(layers) - 0.5 * (layers - 1)) * dx
        grid_a, grid_b, grid_n = np.meshgrid(a, b, n, indexing="ij")
        points.append(centre + grid_a.reshape(-1, 1) * axis_a + grid_b.reshape(-1, 1) * axis_b
                      + grid_n.reshape(-1, 1) * normal)
        boxes.append(OrientedBox(centre, np.stack([axis_a, axis_b, normal]), [half_a, half_b, 0.5 * layers * dx]))
    return np.vstack(points), boxes


def conformal_baffles(dx, layers, top):
    """Baffle plates as plane grids of particles in the baffles' own frames (2026-09-29), like
    conformal_blades(): `layers` flat layers, spacing <= dx in the plate, true outline from
    BAFFLE_RADIAL and BAFFLE_Y0 up to `top` (the liquid surface; the lid shell closes the tank
    above it). Two of the three baffles cross the lattice obliquely; taken from the lattice a
    single-layer baffle is a staircase whose neighbouring particles touch only at corners.
    Returns (points, boxes) as conformal_blades()."""
    points, boxes = [], []
    for azimuth in BAFFLE_AZIMUTHS_DEG:
        e_r, e_t, e_y = radial_frame(azimuth)
        r0, r1 = BAFFLE_RADIAL
        half_a, half_b = 0.5 * (r1 - r0), 0.5 * (top - BAFFLE_Y0)
        centre = e_r * 0.5 * (r0 + r1) + e_y * 0.5 * (BAFFLE_Y0 + top)
        cells_a = max(1, int(math.ceil(2.0 * half_a / dx - 1e-9)))
        cells_b = max(1, int(math.ceil(2.0 * half_b / dx - 1e-9)))
        a = -half_a + (np.arange(cells_a) + 0.5) * (2.0 * half_a / cells_a)
        b = -half_b + (np.arange(cells_b) + 0.5) * (2.0 * half_b / cells_b)
        n = (np.arange(layers) - 0.5 * (layers - 1)) * dx
        grid_a, grid_b, grid_n = np.meshgrid(a, b, n, indexing="ij")
        points.append(centre + grid_a.reshape(-1, 1) * e_r + grid_b.reshape(-1, 1) * e_y
                      + grid_n.reshape(-1, 1) * e_t)
        boxes.append(OrientedBox(centre, np.stack([e_r, e_y, e_t]), [half_a, half_b, 0.5 * layers * dx]))
    return np.vstack(points), boxes


def points_closer_than(queries, others, distance):
    """For every query point: does a point of `others` lie closer than `distance`? Cell lists on a grid of
    spacing `distance` (numpy only, the solver environment has no scipy): a neighbour that close lies in one
    of the 27 cells around the query's cell."""
    bias = 1 << 20

    def cell_ids(cells):
        cells = cells + bias
        return (cells[:, 0] << 42) | (cells[:, 1] << 21) | cells[:, 2]

    other_ids = cell_ids(np.floor(others / distance).astype(np.int64))
    order = np.argsort(other_ids, kind="stable")
    sorted_ids, sorted_points = other_ids[order], others[order]
    query_cells = np.floor(queries / distance).astype(np.int64)
    found = np.zeros(queries.shape[0], dtype=bool)
    for offset in np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1], indexing="ij")).reshape(3, -1).T:
        ids = cell_ids(query_cells + offset)
        slot = np.searchsorted(sorted_ids, ids, side="left")
        stop = np.searchsorted(sorted_ids, ids, side="right")
        while True:
            index = np.nonzero((slot < stop) & ~found)[0]
            if index.size == 0:
                break
            gap = sorted_points[slot[index]] - queries[index]
            found[index[np.einsum("ij,ij->i", gap, gap) < distance ** 2]] = True
            slot[index] += 1
    return found


def smooth_tank_shell(dx, layers):
    """Shell of the cylinder wall and of the dished floor as particle layers that follow the surfaces
    (2026-10-03, --smooth-walls). Taken from the lattice, the shell of a curved wall is a staircase; at 3 mm
    its steps brake the swirl by pressure (form drag) about twice as hard as Fluent's smooth wall
    (log/2026-10-03_fluent-and-wall-torque.md). Layer k = 0 .. layers - 1 lies (k + 0.5) dx outside the
    surface along its normal, as the lattice shell's first layer does on a flat wall:
      cylinder r = TANK_RADIUS: rings on the lattice planes y = j dx from (layers + 0.5) dx below the floor
        rim up to the top of the lid shell;
      floor y = floor(r) (FLOOR_PROFILE): the profile's normal is averaged over about dx of arc (the profile
        is piecewise linear), each offset profile is resampled at about dx along its arc, and each sample
        becomes a ring about the axis (one particle on the axis);
    ring particles about dx apart (count = round(2 pi r / dx)), every other ring turned by half a spacing.
    At the rim the floor follows the knuckle: floor particles closer than 0.6 dx to a cylinder particle above
    the rim are dropped, cylinder particles below the rim closer than 0.6 dx to a remaining floor particle too
    (cylinder first, a first version, left a pocket in the knuckle where fluid sat 0.6 mm outside the floor)."""
    top = LIQUID_HEIGHT + layers * dx
    rim = FLOOR_PROFILE[-1, 1]
    rows = np.arange(math.floor((rim - (layers + 0.5) * dx) / dx), math.floor(top / dx + 1e-9) + 1) * dx
    rings = []
    for k in range(layers):
        radius = TANK_RADIUS + (k + 0.5) * dx
        count = int(round(2.0 * math.pi * radius / dx))
        for index, y in enumerate(rows):
            angle = (np.arange(count) + 0.5 * (index % 2)) * (2.0 * math.pi / count)
            rings.append(np.column_stack([radius * np.cos(angle), np.full(count, y), radius * np.sin(angle)]))
    cylinder = np.vstack(rings)

    dense_r = np.linspace(0.0, TANK_RADIUS, 4001)
    dense_y = np.interp(dense_r, FLOOR_PROFILE[:, 0], FLOOR_PROFILE[:, 1])
    tangent = np.stack([np.gradient(dense_r), np.gradient(dense_y)], axis=1)
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    arc = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(dense_r), np.diff(dense_y)))])
    window = max(1, int(round(dx / (arc[-1] / (arc.size - 1)))))
    kernel = np.ones(window) / window
    tangent = np.stack([np.convolve(np.pad(tangent[:, axis], (window // 2, window - 1 - window // 2), mode="edge"),
                                    kernel, mode="valid") for axis in range(2)], axis=1)
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    normal = np.stack([tangent[:, 1], -tangent[:, 0]], axis=1)            # out of the liquid: down on the crown
    rings = []
    for k in range(layers):
        offset_r = dense_r + (k + 0.5) * dx * normal[:, 0]
        offset_y = dense_y + (k + 0.5) * dx * normal[:, 1]
        offset_arc = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(offset_r), np.diff(offset_y)))])
        samples = max(1, int(round(offset_arc[-1] / dx)))
        targets = np.arange(samples + 1) * (offset_arc[-1] / samples)
        for index, (r, y) in enumerate(zip(np.interp(targets, offset_arc, offset_r),
                                           np.interp(targets, offset_arc, offset_y))):
            if r < 0.25 * dx:
                rings.append(np.array([[0.0, y, 0.0]]))
                continue
            count = max(3, int(round(2.0 * math.pi * r / dx)))
            angle = (np.arange(count) + 0.5 * (index % 2)) * (2.0 * math.pi / count)
            rings.append(np.column_stack([r * np.cos(angle), np.full(count, y), r * np.sin(angle)]))
    floor = np.vstack(rings)
    below_rim = cylinder[:, 1] < rim - 1e-9
    floor = floor[~points_closer_than(floor, cylinder[~below_rim], 0.6 * dx)]
    lower = cylinder[below_rim]
    lower = lower[~points_closer_than(lower, floor, 0.6 * dx)]
    return np.vstack([cylinder[~below_rim], lower, floor])


def thin_plate_sheets(dx, impellers, top, lid, legacy_baffles, disk_inner_radius,
                      with_impeller=True, with_baffles=True, with_disk=True):
    """The thin plates of the tank (2026-09-30, case block `thin_plates:`): the blades, the
    Rushton disk and the baffle plates, each ONE layer of particles on its mid-plane.
    with_impeller / with_baffles = False leave out the blades and the disk / the baffles
    (2026-10-03, mixed representations, see --thin-plates).

    Returns a list of dicts: name, shape, frame, centre, normal, axis_a, extent (the analytic
    outline the solver uses to decide what lies behind the plate), thickness (true, for the
    record), points (the particles) and box (a Region: lattice sites inside it are dropped).
    Blades and baffles: plane grids as conformal_blades() / conformal_baffles(), the outermost
    centres half a cell inside the true outline. The outline of a baffle continues `lid` into
    the lid shell above the liquid (`top`), its particles end at the liquid surface.
    Disk: rings of particles between `disk_inner_radius` (the surface of the hub particles) and
    the rim; its outline is the annulus from the hub radius to the rim."""
    plates = []
    for name, centre, axis_a, axis_b, normal, half_a, half_b in (blade_frames(impellers) if with_impeller else ()):
        cells_a = max(1, int(math.ceil(2.0 * half_a / dx - 1e-9)))
        cells_b = max(1, int(math.ceil(2.0 * half_b / dx - 1e-9)))
        a = -half_a + (np.arange(cells_a) + 0.5) * (2.0 * half_a / cells_a)
        b = -half_b + (np.arange(cells_b) + 0.5) * (2.0 * half_b / cells_b)
        grid_a, grid_b = np.meshgrid(a, b, indexing="ij")
        points = centre + grid_a.reshape(-1, 1) * axis_a + grid_b.reshape(-1, 1) * axis_b
        thickness = RUSHTON_BLADE["thickness"] if name.startswith("rushton") else PBT_BLADE["thickness"]
        plates.append(dict(name=name.replace("_", "_blade_"), shape="rectangle", frame="rotor", centre=centre,
                           normal=normal, axis_a=axis_a, extent=(half_a, half_b), thickness=thickness,
                           points=points, measure=(2.0 * half_a / cells_a) * (2.0 * half_b / cells_b),
                           box=OrientedBox(centre, np.stack([axis_a, axis_b, normal]), [half_a, half_b, 0.5 * dx])))
    if with_impeller and with_disk and impellers in ("both", "rushton"):
        disk_mid = 0.5 * (RUSHTON_DISK["y0"] + RUSHTON_DISK["y1"])
        outer = RUSHTON_DISK["radius"]
        rings = max(1, int(math.ceil((outer - disk_inner_radius) / dx - 1e-9)))
        ring_width = (outer - disk_inner_radius) / rings
        points = []
        for ring in range(rings):
            radius = disk_inner_radius + (ring + 0.5) * ring_width
            count = max(6, int(round(2.0 * math.pi * radius / dx)))
            angle = (np.arange(count) + 0.5 * (ring % 2)) * (2.0 * math.pi / count)
            points.append(np.column_stack([radius * np.cos(angle), np.full(count, disk_mid), radius * np.sin(angle)]))
        points = np.vstack(points)
        centre = np.array([0.0, disk_mid, 0.0])
        plates.append(dict(name="rushton_disk", shape="annulus", frame="rotor", centre=centre,
                           normal=np.array([0.0, 1.0, 0.0]), axis_a=np.array([1.0, 0.0, 0.0]),
                           extent=(outer, RUSHTON_HUB["radius"]),
                           thickness=RUSHTON_DISK["y1"] - RUSHTON_DISK["y0"], points=points,
                           measure=math.pi * (outer ** 2 - disk_inner_radius ** 2) / points.shape[0],
                           box=Difference(y_cylinder(outer, disk_mid - 0.5 * dx, disk_mid + 0.5 * dx),
                                          y_cylinder(disk_inner_radius, disk_mid - dx, disk_mid + dx))))
    if legacy_baffles:
        azimuths, radial = LEGACY_BAFFLES["azimuths_deg"], LEGACY_BAFFLES["radial"]
        bottom, thickness = LEGACY_BAFFLES["y0"], LEGACY_BAFFLES["thickness"]
    else:
        azimuths, radial, bottom, thickness = BAFFLE_AZIMUTHS_DEG, BAFFLE_RADIAL, BAFFLE_Y0, BAFFLE_THICKNESS
    for index, azimuth in enumerate(azimuths if with_baffles else ()):
        e_r, e_t, e_y = radial_frame(azimuth)
        r0, r1 = radial
        half_a, half_b = 0.5 * (r1 - r0), 0.5 * (top - bottom)
        centre = e_r * 0.5 * (r0 + r1) + e_y * 0.5 * (bottom + top)
        cells_a = max(1, int(math.ceil(2.0 * half_a / dx - 1e-9)))
        cells_b = max(1, int(math.ceil(2.0 * half_b / dx - 1e-9)))
        a = -half_a + (np.arange(cells_a) + 0.5) * (2.0 * half_a / cells_a)
        b = -half_b + (np.arange(cells_b) + 0.5) * (2.0 * half_b / cells_b)
        grid_a, grid_b = np.meshgrid(a, b, indexing="ij")
        points = centre + grid_a.reshape(-1, 1) * e_r + grid_b.reshape(-1, 1) * e_y
        plates.append(dict(name=f"baffle_{index + 1}", shape="rectangle", frame="static",
                           centre=centre + e_y * 0.5 * lid, normal=e_t, axis_a=e_r,
                           extent=(half_a, half_b + 0.5 * lid), thickness=thickness, points=points,
                           measure=(2.0 * half_a / cells_a) * (2.0 * half_b / cells_b),
                           box=OrientedBox(centre, np.stack([e_r, e_y, e_t]), [half_a, half_b, 0.5 * dx])))
    return plates


MAX_NEAR_THIN_PLATES = 4          # shaders/thin_plates.glsl


def count_near_plates(points, plates, support_radius):
    """For every point the number of plates that come closer than the support radius (the
    solver keeps at most MAX_NEAR_THIN_PLATES of them per particle)."""
    count = np.zeros(points.shape[0], dtype=np.int32)
    for plate in plates:
        relative = points - plate["centre"]
        normal = np.asarray(plate["normal"])
        axis_a = np.asarray(plate["axis_a"])
        axis_b = np.cross(normal, axis_a)
        distance_normal = relative @ normal
        a, b = relative @ axis_a, relative @ axis_b
        if plate["shape"] == "annulus":
            radius = np.hypot(a, b)
            gap = np.maximum(np.maximum(radius - plate["extent"][0], plate["extent"][1] - radius), 0.0)
        else:
            gap = np.hypot(np.maximum(np.abs(a) - plate["extent"][0], 0.0),
                           np.maximum(np.abs(b) - plate["extent"][1], 0.0))
        count += (distance_normal ** 2 + gap ** 2 < support_radius ** 2)
    return count


def thin_plates_block(plates) -> str:
    """`thin_plates:` block of case.yaml."""
    def vector(values):
        return "[" + ", ".join(f"{float(v):.9g}" for v in values) + "]"
    lines = ["", "# Thin plates wetted on both sides (2026-09-30, shaders/thin_plates.glsl): one layer of particles",
             "# on the mid-plane; for a fluid particle everything behind a plate is a dummy of the wall.",
             "# Rotor plates are given at the rotor angle 0. thickness (true) and point_measure (area per",
             "# particle) are for the record, the solver does not use them.",
             "thin_plates:"]
    for plate in plates:
        lines.append(f"  - {{name: {plate['name']}, shape: {plate['shape']}, frame: {plate['frame']}, "
                     f"centre: {vector(plate['centre'])}, normal: {vector(plate['normal'])}, "
                     f"axis_a: {vector(plate['axis_a'])}, extent: {vector(plate['extent'])}, "
                     f"thickness: {plate['thickness']:.6g}, point_measure: {plate['measure']:.6g}}}")
    return "\n".join(lines) + "\n"


def build_solids(thin, shaft_y0, shaft_y1, top_y, impellers="both", clip_tips=False, with_blades=True,
                 legacy_baffles=False, with_baffle_plates=True, with_bell=True, with_disk=True, with_shaft=True,
                 disk_thickness=None):
    """Return (rotor_region, wall_solid_region). ``thin`` = minimum thickness.
    ``impellers`` = "both" | "rushton" | "pbt": which impellers (hub + blades,
    and the disk for the Rushton) are kept on the full-length shaft; used for
    the single-impeller control runs of 2026-09-26.
    ``disk_thickness`` overrides the disk's max(true thickness, thin) (2026-10-06, --solid-disk)."""
    keep_rushton = impellers in ("both", "rushton")
    keep_pbt = impellers in ("both", "pbt")
    t_baffle = max(LEGACY_BAFFLES["thickness"] if legacy_baffles else BAFFLE_THICKNESS, thin)
    t_rblade = max(RUSHTON_BLADE["thickness"], thin)
    t_disk = max(RUSHTON_DISK["y1"] - RUSHTON_DISK["y0"], thin) if disk_thickness is None else disk_thickness
    t_pblade = max(PBT_BLADE["thickness"], thin)

    disk_mid = 0.5 * (RUSHTON_DISK["y0"] + RUSHTON_DISK["y1"])
    # with_shaft=False (2026-10-03, diagnostic --no-shaft): no shaft cylinder; hubs, collar and bell stay
    # (the rotor moves kinematically, its parts need not touch)
    rotor_parts = [y_cylinder(SHAFT_RADIUS, shaft_y0, shaft_y1)] if with_shaft else []
    if with_bell:
        rotor_parts.append(RevolvedProfile(ROTOR_BELL_PROFILE))
    if keep_rushton:
        rotor_parts.append(y_cylinder(RUSHTON_HUB["radius"], RUSHTON_HUB["y0"], RUSHTON_HUB["y1"]))
        if with_disk:
            rotor_parts.append(y_cylinder(RUSHTON_DISK["radius"], disk_mid - 0.5 * t_disk, disk_mid + 0.5 * t_disk))
    if keep_pbt:
        rotor_parts.append(y_cylinder(PBT_HUB["radius"], PBT_HUB["y0"], PBT_HUB["y1"]))
        if PBT_HUB is not LEGACY_PBT_HUB:
            rotor_parts.append(y_cylinder(PBT_COLLAR["radius"], PBT_COLLAR["y0"], PBT_COLLAR["y1"]))
    for k in range(6 if with_blades else 0):
        if keep_rushton:
            blade = radial_slab(RUSHTON_BLADE["azimuth0_deg"] + 60 * k,
                                *RUSHTON_BLADE["radial"], RUSHTON_BLADE["y0"],
                                RUSHTON_BLADE["y1"], t_rblade)
            if clip_tips:
                blade = Intersection(blade, y_cylinder(RUSHTON_TIP_RADIUS, shaft_y0, shaft_y1))
            rotor_parts.append(blade)
        if keep_pbt:
            outer = PBT_TRUE_LENGTH if clip_tips else PBT_BLADE["radial"][1]
            blade = pitched_blade(PBT_BLADE["azimuth0_deg"] + 60 * k,
                                  PBT_BLADE["radial"][0], outer, PBT_BLADE["chord"], t_pblade,
                                  PBT_BLADE["center_y"], PBT_BLADE["pitch_deg"])
            if clip_tips:
                blade = Intersection(blade, y_cylinder(PBT_TIP_RADIUS, shaft_y0, shaft_y1))
            rotor_parts.append(blade)
    wall_parts = [y_cylinder(BEARING_BOSS["radius"], FLOOR_BOTTOM - 0.02, BEARING_BOSS["y1"])]
    if legacy_baffles:
        for az in LEGACY_BAFFLES["azimuths_deg"] if with_baffle_plates else ():
            wall_parts.append(radial_slab(az, *LEGACY_BAFFLES["radial"], LEGACY_BAFFLES["y0"], top_y, t_baffle))
    else:
        for az in BAFFLE_AZIMUTHS_DEG:
            if with_baffle_plates:
                wall_parts.append(radial_slab(az, *BAFFLE_RADIAL, BAFFLE_Y0, top_y, t_baffle))
            for r0, r1, width in BAFFLE_FOOT["parts"]:
                wall_parts.append(radial_slab(az, r0, r1, BAFFLE_FOOT["y0"], BAFFLE_FOOT["y1"], width))
    for cx, cz in PROBE_CENTERS:
        wall_parts.append(Box([cx - PROBE_HALF, PROBE_Y[0], cz - PROBE_HALF],
                              [cx + PROBE_HALF, top_y, cz + PROBE_HALF]))
    return Union(*rotor_parts), Union(*wall_parts)


CASE_YAML = """\
schema_version: 2

# 30 L stirred tank (Rautenbach et al. 2026 dataset geometry), 3D.
# GENERATED by utils/geometry/_demo_stirred_tank_30l.py.
#   dx = {dx:.6f} m, h = {h:.6f} m (h/dx = {hdx}), thin parts >= {thin_layers} layers
#   fluid {n_fluid:,} / wall {n_wall:,} / rotor {n_rotor:,}
# Flat lid at y = {liquid_height} (no free surface), gravity off: with a closed
# single-phase tank gravity only adds a hydrostatic offset, so c0 is set from
# the impeller tip speed only ({c0_factor:g} * {tip_speed:.3f} m/s).
# Rotor: shaft + both impellers rotate rigidly about +y through the origin
# (predict.comp ROTOR branch, 2026-09-25). The dataset's M-Star input.xml has
# freq = -200 rpm, rotationAxis (0,1,0). M-Star's sign convention is the mixing
# one, NOT the right-hand rule: "a positive RPM implies clockwise motion as
# viewed when looking into the direction of gravity" (docs.mstarcfd.com, Moving
# Bodies). So -200 rpm = counter-clockwise seen from above = POSITIVE angular
# velocity about +y in the right-hand sense: the blades move from +x toward -z
# (toward -theta). The CAD's PBT blades have y decreasing toward +theta, so
# their leading edge is the high edge and the PBT pumps DOWN. (2026-09-26 fix:
# the sign was negative before, which made the PBT pump UP; measured +0.24 m/s
# axial through the PBT in the c0 = 20 U_tip series, see log.)

time:
  total: null
  max_steps: null
  output_cadence: null

physics:
  dimension: 3
  h: {h:.6f}
  particle_radius: {radius:.6f}
  lattice: grid
  calibrate_volume: true
  speed_of_sound: {c0:.3f}
  power: 7
  cfl: {cfl:g}
  gravity: [0.0, {gravity_y:.3f}, 0.0]

numerics:
  use_density_diffusion: true
  delta_coefficient: 0.1
  use_kcg_correction: true
  regularization:
    xi: {xi:g}      # default 0.01; 0.1 until 2026-09-28: it made (M + xi I)^-1 about 8 % too small (log/2026-09-28_kcg-xi-0p01.md)
    det_threshold: 1.0e-4
    frobenius_max: 10.0
  use_pst: true
  pst_main: 0.1
  pst_anti: 0.0005
  defrag_enabled: true
  defrag_cadence: 10
  use_prefix_sum_defrag: false

capacities:
  pool_size: {pool_size}
  max_per_voxel: {max_per_voxel}
  max_incoming: {max_incoming}
  workgroup: 128

material_library: materials.yaml

rotor:
  axis: [0.0, 1.0, 0.0]
  pivot: [0.0, 0.0, 0.0]
  ramp_time: {ramp_time:.4f}      # s, linear spin-up (M-Star used 0.0092 s)

geometry:
  frame: frame.obj
  particles:
    - {{file: fluid.obj, material: tank_water}}
    - {{file: wall.obj,  material: tank_wall}}
    - {{file: rotor.obj, material: impeller}}
"""

MATERIALS_YAML = """\
schema_version: 1

# Case-local material library for the 30 L stirred tank.
tank_water:
  kind: fluid
  rest_density: 998.0
  viscosity: {viscosity:.6e}

tank_wall:
  kind: boundary
  rest_density: 998.0
  viscosity: {viscosity:.6e}

impeller:
  kind: rotor
  rest_density: 998.0
  viscosity: {viscosity:.6e}
  rotor_angular_velocity: {omega:.5f}   # rad/s, right-hand sign about +y (= M-Star -200 rpm, CCW from above, PBT down-pumping)
"""


def write_obj(path, points):
    np.savetxt(path, points, fmt="v %.6f %.6f %.6f", header=f"# {points.shape[0]} particles", comments="")


def write_frame_obj(path, lo, hi):
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    vertices = [(x0, y0, z0), (x0, y1, z0), (x1, y0, z0), (x1, y1, z0),
                (x0, y0, z1), (x0, y1, z1), (x1, y0, z1), (x1, y1, z1)]
    faces = [(1, 2, 4), (1, 4, 3), (5, 7, 8), (5, 8, 6), (1, 5, 6), (1, 6, 2),
             (3, 4, 8), (3, 8, 7), (1, 3, 7), (1, 7, 5), (2, 6, 8), (2, 8, 4)]
    with open(path, "w") as handle:
        handle.write("# stirred tank frame bbox\n")
        for v in vertices:
            handle.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for f in faces:
            handle.write(f"f {f[0]} {f[1]} {f[2]}\n")


def write_preview(path, fluid, wall, rotor, dx):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.5))

    def section(axis, mask_fn, cols, title):
        for cloud, color, size in ((fluid, "tab:blue", 0.3), (wall, "tab:green", 1.0), (rotor, "tab:red", 1.5)):
            m = mask_fn(cloud)
            axis.scatter(cloud[m, cols[0]], cloud[m, cols[1]], s=size, c=color)
        axis.set_aspect("equal")
        axis.set_title(title)

    section(axes[0], lambda c: np.abs(c[:, 2]) < 0.6 * dx, (0, 1), "vertical section z = 0 (x-y)")
    section(axes[1], lambda c: np.abs(c[:, 1] - 0.037) < 0.6 * dx, (0, 2), "horizontal section y = 0.037 (Rushton)")
    section(axes[2], lambda c: np.abs(c[:, 1] - 0.195) < 0.6 * dx, (0, 2), "horizontal section y = 0.195 (PBT)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# Monod kinetics of Haringa (2023, Eng. Life Sci. 23:e2100159; P. chrysogenum, 54 m3):
# q_max = 1600 umol / (g h), K_s = 7.8 umol / kg, C_x = 55 g / kg, initial C_s = 10 K_s.
HARINGA_Q_MAX_UMOL_PER_G_H = 1600.0
HARINGA_K_S = 7.8e-6
HARINGA_BIOMASS = 55.0


def scalars_block(args, h) -> str:
    """`scalars:` block (2026-09-27, extended 2026-10-01).
    Substrate setup (--substrate): fields substrate, biomass, uptake, feed (one vec4; mol/kg, g/kg),
    Monod uptake unless --no-uptake, a continuous feed sphere (default: the paper's injection point).
    Mixing-time runs (--tracers N): one tracer field per pulse (so the pulses do not contaminate
    each other's t95), injection at the paper's point. Probes at the paper's two measurement
    points (Shepard radius = h)."""
    lines = ["", "# Scalar transport (2026-09-27): tracer pulses at the injection point of",
             "# Rautenbach et al. (2026) Table 1, probes at its two measurement points.",
             "# Substrate setup (2026-10-01, stage 3): Monod uptake by biomass on the fluid particles,",
             "# continuous feed (log/2026-10-01_reaction-and-feed.md).",
             "scalars:", "  fields:"]
    sgs_flag = "true" if args.sgs else "false"
    if args.substrate:
        initial = args.substrate_initial if args.substrate_initial is not None else 10.0 * args.k_s
        lines.append(f"    - {{name: substrate, diffusivity: {args.substrate_diffusivity:.3e}, turbulent: true, "
                     f"initial: {initial:.6e}}}")
        lines.append(f"    - {{name: biomass, diffusivity: 0.0, turbulent: false, initial: {args.biomass:.6e}}}")
        lines.append("    - {name: uptake, diffusivity: 0.0, turbulent: false, initial: 0.0}")
        lines.append("    - {name: feed, diffusivity: 0.0, turbulent: false, initial: 0.0}")
        if args.state_limited:
            # level-2 cell model (2026-10-06, stage 5): state vec4 after the substrate vec4
            q_max = args.q_max_umol_per_g_h * 1e-6 / 3600.0
            maintenance = args.maintenance_umol_per_g_h * 1e-6 / 3600.0
            growth_rate_initial = (args.growth_rate_initial if args.growth_rate_initial is not None
                                   else max(args.growth_yield * (q_max - maintenance), 0.0))
            lines.append(f"    - {{name: growth_rate, diffusivity: 0.0, turbulent: false, initial: {growth_rate_initial:.6e}}}")
            lines.append("    - {name: product, diffusivity: 0.0, turbulent: false, initial: 0.0}")
            lines.append("    - {name: maintenance, diffusivity: 0.0, turbulent: false, initial: 0.0}")
    for index in range(args.tracers):
        lines.append(f"    - {{name: tracer_{index + 1:02d}, diffusivity: {args.tracer_diffusivity:.3e}, "
                     f"turbulent: true, initial: 0.0}}")
    lines += ["  sgs:", f"    enabled: {sgs_flag}", "    smagorinsky_cs: 0.1",
              "    turbulent_schmidt: 0.7"]
    if args.substrate:
        q_max = args.q_max_umol_per_g_h * 1e-6 / 3600.0
        if args.state_limited:
            maintenance = args.maintenance_umol_per_g_h * 1e-6 / 3600.0
            p0, p1, p2 = args.product_rate
            lines += ["  reactions:",
                      f"    - {{type: state_limited, substrate: substrate, biomass: biomass, uptake: uptake, "
                      f"growth_rate: growth_rate, product: product, maintenance: maintenance, "
                      f"q_max: {q_max:.6e}, k_s: {args.k_s:.6e}, yield: {args.growth_yield:.6e}, "
                      f"maintenance_rate: {maintenance:.6e}, demand_margin: {args.demand_margin:.6e}, "
                      f"tau_up: {args.tau_up:.6e}, tau_down: {args.tau_down:.6e}, "
                      f"product_rate: [{p0:.6e}, {p1:.6e}, {p2:.6e}]}}"]
        elif not args.no_uptake:
            lines += ["  reactions:",
                      f"    - {{type: monod, substrate: substrate, biomass: biomass, uptake: uptake, "
                      f"q_max: {q_max:.6e}, k_s: {args.k_s:.6e}, yield: {args.growth_yield:.6e}}}"]
        center = args.feed_center if args.feed_center is not None else INJECTION_POINT
        stop = "" if args.feed_stop is None else f", stop: {args.feed_stop:.4f}"
        lines += ["  sources:",
                  f"    - {{field: substrate, center: [{center[0]}, {center[1]}, {center[2]}], radius: {args.feed_radius:.4f}, "
                  f"rate: {args.feed_rate:.6e}, start: {args.feed_start:.4f}{stop}, record: feed}}"]
    lines.append("  injections:" if args.tracers > 0 else "  injections: []")
    for index in range(args.tracers):
        start = args.injection_start + index * args.injection_interval
        lines.append(f"    - {{field: tracer_{index + 1:02d}, center: [{INJECTION_POINT[0]}, {INJECTION_POINT[1]}, "
                     f"{INJECTION_POINT[2]}], radius: {args.injection_radius:.4f}, start: {start:.4f}, "
                     f"duration: {args.injection_duration:.4f}, value: 1.0}}")
    lines += ["  probes:", f"    radius: {h:.6f}", "    points:"]
    for name, (x, y, z) in PROBE_POINTS:
        lines.append(f"      - {{name: {name}, position: [{x}, {y}, {z}]}}")
    if args.shift_correction:
        # Taylor shift correction of the scalars (default off, see
        # log/2026-09-27_scalar-transport.md 4.6); the bounds limiter keeps its default (on).
        lines.append("  shift_correction: true")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dx", type=float, default=0.003, help="particle spacing (m)")
    parser.add_argument("--hdx", type=float, default=3.0, help="h/dx (kernel support radius in spacings; non-integer allowed, e.g. 2.5)")
    parser.add_argument("--thin-layers", type=int, default=3, help="minimum layers across thin solids")
    parser.add_argument("--clip-tips", action="store_true",
                        help="blades with the true PBT length (48.2 mm) and cut at the true tip radius "
                             "(Rushton 48.0 mm, PBT 49.1 mm) instead of the plain boxes")
    parser.add_argument("--xi", type=float, default=0.01, help="KCG regularisation numerics.regularization.xi")
    parser.add_argument("--viscosity", type=float, default=1.0e-6,
                        help="kinematic viscosity written for all three materials (m^2/s). The Morris viscous "
                             "term is 0.841 of the exact operator on the h/dx = 3 lattice with xi = 0.01 "
                             "(_check_operator_consistency.py), so 1.0e-6 / 0.8408 = 1.18934e-6 gives an "
                             "effective viscosity of 1.0e-6")
    parser.add_argument("--no-rotor-bell", action="store_true",
                        help="rotor without the rotating bell at the lower end of the shaft (as until 2026-09-29)")
    parser.add_argument("--no-shaft", action="store_true",
                        help="diagnostic (2026-10-03): leave out the shaft cylinder (r = 4 mm, a 3 x 3 particle "
                             "column at 3 mm that stirs like a paddle); hubs, collar and bell stay. Not physical")
    parser.add_argument("--conformal-baffles", action="store_true",
                        help="baffle plates on plane grids in the baffles' own frames (--thin-layers flat "
                             "layers), like --conformal-blades; not with --legacy-baffles")
    parser.add_argument("--legacy-pbt", action="store_true",
                        help="PBT as before 2026-10-02: chord 27.3 mm (10 %% too wide) and a hub r 10.9 mm over 180 .. 209.3 mm")
    parser.add_argument("--legacy-baffles", action="store_true",
                        help="baffles as until 2026-09-29: 13 mm wide (r = 125.4 .. 138.4 mm), 62 / 182 / 302 deg, "
                             "from y = 10 mm, no block at the lower end; the true baffles are 24 mm wide")
    parser.add_argument("--conformal-blades", action="store_true",
                        help="blade particles on plane grids in the blades' own frames (--thin-layers flat "
                             "layers, true outline) instead of lattice sites; avoids the staircase at which "
                             "single-layer blades leak")
    parser.add_argument("--thin-plates", action="store_true",
                        help="blades, Rushton disk and baffle plates as thin plates (case block "
                             "`thin_plates:`, one layer of particles each, shaders/thin_plates.glsl); needs "
                             "--solid-reaction-force; replaces --conformal-blades / --conformal-baffles. "
                             "Mixed (2026-10-03): with --conformal-baffles only the blades and the disk are "
                             "plates (baffles: conformal sheets); with --conformal-blades only the baffles are "
                             "plates (blades: conformal sheets, disk: lattice, as without --thin-plates)")
    parser.add_argument("--conformal-pbt", type=int, default=None, metavar="LAYERS",
                        help="with --thin-plates (2026-10-04): only the PBT blades as conformal sheets of LAYERS "
                             "flat layers (a solid outer edge), the Rushton blades and disk stay thin plates, the "
                             "baffles follow --conformal-baffles / --thin-layers; to test whether the PBT tip flow "
                             "depends on the thin-plate edge")
    parser.add_argument("--solid-disk", type=int, default=None, metavar="LAYERS",
                        help="with --thin-plates: the Rushton disk as a lattice cylinder of rotor particles "
                             "LAYERS spacings thick instead of a thin-plate annulus; the blades stay thin plates "
                             "(2026-10-06, test of the jet tilt)")
    parser.add_argument("--thin-plate-dashpot", type=float, default=None,
                        help="numerics.thin_plate_dashpot (default of the solver: 0)")
    parser.add_argument("--skin", type=float, default=0.5,
                        help="rotor / baffle / probe sites are claimed up to this many spacings outside "
                             "the solid surface (default 0.5; 0 = site centre inside the solid)")
    parser.add_argument("--border", type=int, default=None, help="wall shell layers (default = hdx)")
    parser.add_argument("--fluent-hubs", action="store_true",
                        help="hubs and PBT blade span of the Fluent LES mesh (FLUENT_RUSHTON_HUB, FLUENT_PBT_HUB, "
                             "FLUENT_PBT_RADIAL) instead of the M-Star STL's")
    parser.add_argument("--pbt-tip-radius", type=float, default=None, metavar="R",
                        help="diagnostic (2026-10-05): cut the PBT blades at radius R (m) instead of the true tip "
                             "(thin plates: PBT_TRUE_LENGTH 48.2 mm; lattice blades: 49.1 mm). A shorter blade mimics "
                             "the tip unloading Fluent shows and our thin plates lack; 45.2 mm brings the PBT torque "
                             "to about Fluent's 15 mN m. Not physical")
    parser.add_argument("--smooth-walls", action="store_true",
                        help="shell of the cylinder and the dished floor as layers that follow the surfaces "
                             "(smooth_tank_shell) instead of lattice sites (a staircase); the flat lid stays on the lattice")
    parser.add_argument("--out", default="cases/stirred_tank_30l")
    parser.add_argument("--max-per-voxel", type=int, default=None)
    parser.add_argument("--max-incoming", type=int, default=32)
    parser.add_argument("--ramp-time", type=float, default=0.05, help="rotor spin-up time (s)")
    parser.add_argument("--c0-factor", type=float, default=10.0, help="speed of sound = factor * tip speed")
    parser.add_argument("--cfl", type=float, default=0.15,
                        help="physics.cfl: dt = cfl * h / c0 with h the kernel support radius (0.15 with "
                             "h = 3 dx is 0.30 in terms of half the support radius)")
    parser.add_argument("--background-pressure", type=float, default=0.0,
                        help="constant added to the Tait pressure, Pa (physics.background_pressure); keeps the "
                             "pressure behind the blades positive. About rho * U_tip^2 = 1000 Pa. 0 = off")
    parser.add_argument("--symmetric-pair-correction", action="store_true",
                        help="older spelling of --pair-correction mean (writes symmetric_pair_correction: true)")
    parser.add_argument("--momentum-sgs", type=float, default=None, metavar="CS",
                        help="numerics.momentum_sgs with Smagorinsky C_s (2026-10-01); see --momentum-sgs-width")
    parser.add_argument("--momentum-sgs-width", default="dx",
                        help="filter width of the momentum SGS: 'dx', '2dx', 'h' or metres (default dx)")
    parser.add_argument("--momentum-sgs-wall-damping", action="store_true",
                        help="numerics.momentum_sgs_wall_damping (2026-10-05): mixing length limited by 0.41 x the "
                             "distance to the nearest solid particle or plate dummy (Smagorinsky-Lilly wall limit)")
    parser.add_argument("--shift-transport", choices=("none", "density", "momentum", "both"), default="none",
                        help="numerics.shift_transport: transport terms of the particle shift in the continuity "
                             "and momentum equations (2026-09-30)")
    parser.add_argument("--no-solid-density-floor", action="store_true",
                        help="numerics.solid_density_floor: false (accumulate walls may fall below rho0; needed "
                             "with --gravity --hydrostatic so that the lid holds the top fluid layers, 2026-09-30)")
    parser.add_argument("--pair-correction", choices=("own", "mean", "reverse"), default="own",
                        help="numerics.pair_correction, KCG matrices of fluid-fluid pairs: own (original), "
                             "mean of the two, or reverse (P_i B_j + P_j B_i, Zhang, Adams, Hu 2025)")
    parser.add_argument("--diffusion-gradient-term", action="store_true",
                        help="numerics.density_diffusion_gradient_term (needed with gravity)")
    parser.add_argument("--pst-near-solid", choices=("full", "tangential"), default="full",
                        help="numerics.pst_near_solid")
    parser.add_argument("--pst-main", type=float, default=None,
                        help="numerics.pst_main, the main coefficient of the particle shift (default of the case "
                             "0.1; 0.05 cut the sub-kernel noise by 25 %% in the 0.7 s test of 2026-10-04)")
    parser.add_argument("--solid-pressure", choices=("increment", "mirror", "mirror_tic", "accumulate", "extrapolate"),
                        default="increment",
                        help="numerics.solid_pressure: pressure of the solid particles seen by the fluid "
                             "(increment = original one-step value, mirror = pairwise mirror)")
    parser.add_argument("--solid-pressure-offset", type=float, default=0.0,
                        help="numerics.solid_pressure_offset p_w (Pa), mirror modes: repulsive layer on "
                             "fluid-solid pairs, pair pressure 2 P_i + p_w")
    parser.add_argument("--solid-reaction-force", action="store_true",
                        help="numerics.solid_reaction_force: forces on rotor and walls as the reaction of the "
                             "fluid (always on with the mirror modes)")
    parser.add_argument("--hydrostatic", action="store_true",
                        help="with --gravity: the fluid starts with the hydrostatic density, zero pressure at the lid "
                             "(physics.hydrostatic_reference)")
    parser.add_argument("--gravity", type=float, default=0.0,
                        help="gravity magnitude along -y (0 = off; with gravity on use --c0-factor 20 so that c0 >= 10*sqrt(g*H))")
    parser.add_argument("--impellers", choices=("both", "rushton", "pbt"), default="both",
                        help="keep both impellers (default) or only one of them on the full shaft "
                             "(single-impeller control runs)")
    parser.add_argument("--tracers", type=int, default=0,
                        help="scalar transport: number of tracer fields, one pulse each (0 = no "
                             "`scalars:` block; the paper used 10 pulses 1 s apart from t = 25 s)")
    parser.add_argument("--injection-start", type=float, default=25.0,
                        help="start of the first tracer pulse (s)")
    parser.add_argument("--injection-interval", type=float, default=1.0,
                        help="time between the starts of consecutive pulses (s)")
    parser.add_argument("--injection-duration", type=float, default=1.0, help="pulse length (s)")
    parser.add_argument("--injection-radius", type=float, default=INJECTION_RADIUS,
                        help="radius of the injection sphere (m), default = 5 mL sphere")
    parser.add_argument("--tracer-diffusivity", type=float, default=1.0e-9,
                        help="molecular diffusivity of the tracers (m^2/s)")
    parser.add_argument("--sgs", action="store_true",
                        help="enable the Smagorinsky sub-grid diffusivity for the tracers")
    parser.add_argument("--shift-correction", action="store_true",
                        help="interpolate the tracers along the particle shift (scalars.shift_correction; "
                             "default off: the tracers move with the particles)")
    parser.add_argument("--substrate", action="store_true",
                        help="stage 3 (2026-10-01): fields substrate, biomass, uptake, feed with Monod uptake and a "
                             "continuous feed sphere (see the --q-max ... --feed-* options)")
    parser.add_argument("--q-max-umol-per-g-h", type=float, default=HARINGA_Q_MAX_UMOL_PER_G_H,
                        help="Monod maximum uptake rate, umol / (g h) (default Haringa 2023: 1600)")
    parser.add_argument("--k-s", type=float, default=HARINGA_K_S, help="Monod half-saturation, mol / kg (7.8e-6)")
    parser.add_argument("--biomass", type=float, default=HARINGA_BIOMASS, help="biomass, g / kg (55)")
    parser.add_argument("--growth-yield", type=float, default=0.0, help="g biomass / mol substrate (0: no growth)")
    parser.add_argument("--substrate-initial", type=float, default=None, help="initial substrate, mol / kg (10 K_s)")
    parser.add_argument("--substrate-diffusivity", type=float, default=6.0e-10, help="m^2 / s (glucose in water)")
    parser.add_argument("--no-uptake", action="store_true", help="substrate setup without the Monod sink (pure tracer feed)")
    parser.add_argument("--state-limited", action="store_true",
                        help="stage 5 (2026-10-06): level-2 cell model instead of Monod; adds the fields growth_rate, "
                             "product, maintenance (needs --substrate and --growth-yield > 0; docs/stage5_pichia_level2_design)")
    parser.add_argument("--maintenance-umol-per-g-h", type=float, default=0.0, help="level 2: maintenance uptake m_s, umol / (g h)")
    parser.add_argument("--demand-margin", type=float, default=0.1, help="level 2: alpha, margin of the demand above mu")
    parser.add_argument("--tau-up", type=float, default=60.0, help="level 2: relaxation time of mu upwards, s")
    parser.add_argument("--tau-down", type=float, default=60.0, help="level 2: relaxation time of mu downwards, s")
    parser.add_argument("--product-rate", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("P0", "P1", "P2"),
                        help="level 2: q_p(mu) = p0 + p1 mu + p2 mu^2, g product / (g biomass s), /(g), s/g")
    parser.add_argument("--growth-rate-initial", type=float, default=None,
                        help="level 2: initial mu, 1/s (default mu_max = yield (q_max - m_s))")
    parser.add_argument("--feed-rate", type=float, default=2.0e-4,
                        help="feed, mol / s (default 2.0e-4: mean q / q_max about 0.28 in 29.5 L, as in Haringa 2023)")
    parser.add_argument("--feed-center", type=float, nargs=3, default=None, help="feed sphere centre (default: injection point)")
    parser.add_argument("--feed-radius", type=float, default=0.02, help="feed sphere radius, m (0.02)")
    parser.add_argument("--feed-start", type=float, default=0.0, help="feed start, s")
    parser.add_argument("--feed-stop", type=float, default=None, help="feed stop, s (default: never)")
    parser.add_argument("--no-preview", action="store_true")
    args = parser.parse_args()
    global PBT_HUB, PBT_COLLAR, RUSHTON_HUB, PBT_TRUE_LENGTH, PBT_TIP_RADIUS
    if args.pbt_tip_radius is not None:
        # shortened PBT blades (diagnostic): the thin plates end at R, the lattice blades (with or without
        # --clip-tips) are cut at R as well
        PBT_TRUE_LENGTH = PBT_TIP_RADIUS = args.pbt_tip_radius
        PBT_BLADE["radial"] = (PBT_BLADE["radial"][0], args.pbt_tip_radius)
        print(f"PBT blades cut at r = {args.pbt_tip_radius * 1e3:.1f} mm (diagnostic --pbt-tip-radius)")
    if args.legacy_pbt:
        PBT_HUB = LEGACY_PBT_HUB
        PBT_BLADE["chord"] = LEGACY_PBT_CHORD
        print("legacy PBT: chord 27.3 mm, hub r 10.9 mm over y 0.180 .. 0.2093")
    if args.fluent_hubs:
        if args.legacy_pbt:
            parser.error("--fluent-hubs cannot be combined with --legacy-pbt")
        # the collar is replaced by the Fluent sleeve (build_solids adds PBT_COLLAR as well, the same cylinder)
        RUSHTON_HUB, PBT_HUB, PBT_COLLAR = FLUENT_RUSHTON_HUB, FLUENT_PBT_HUB, FLUENT_PBT_HUB
        PBT_BLADE["radial"] = FLUENT_PBT_RADIAL
        print("Fluent hubs: Rushton hub r 7.97 mm y 18.6 .. 37.2 mm, PBT sleeve r 7.56 mm y 175.4 .. 207.0 mm, "
              "PBT blades r 7.56 .. 48.0 mm")

    dx = args.dx
    h = args.hdx * dx
    border = args.border if args.border is not None else int(math.ceil(args.hdx))   # wall shell layers (int)
    thin = args.thin_layers * dx
    shell = border * dx
    top_y = LIQUID_HEIGHT + shell                # top of lid shell

    # Lattice sites over the whole frame (anchored at the origin by tile_bounding_box).
    lo = np.array([-(TANK_RADIUS + shell), FLOOR_BOTTOM - shell, -(TANK_RADIUS + shell)])
    hi = np.array([TANK_RADIUS + shell, top_y, TANK_RADIUS + shell])
    sites = tile_bounding_box(lo - 0.5 * dx, hi + 0.5 * dx, dx, LATTICE_GRID, 3)
    print(f"dx={dx:.4e} h={h:.4e} (h/dx={args.hdx}) border={border} thin>={args.thin_layers} layers -> {sites.shape[0]:,} lattice sites")

    if args.thin_plates:
        if args.conformal_blades and args.conformal_baffles:
            parser.error("--thin-plates with both --conformal-blades and --conformal-baffles leaves no plate")
        if not (args.solid_reaction_force or args.solid_pressure != "increment"):
            parser.error("--thin-plates needs --solid-reaction-force (the load on the plates is a reaction)")
    if args.conformal_pbt is not None and not (args.thin_plates and not args.conformal_blades and args.impellers == "both"):
        parser.error("--conformal-pbt needs --thin-plates, both impellers and no --conformal-blades")
    if args.solid_disk is not None and not (args.thin_plates and not args.conformal_blades):
        parser.error("--solid-disk needs --thin-plates and no --conformal-blades (it replaces the thin-plate disk)")
    # Which parts are thin plates (2026-10-03): all of them with --thin-plates alone; with
    # --conformal-baffles only the impeller (blades and disk), with --conformal-blades only the baffles.
    impeller_plates = args.thin_plates and not args.conformal_blades
    baffle_plates = args.thin_plates and not args.conformal_baffles
    interior = DishedTankInterior(TANK_RADIUS, LIQUID_HEIGHT, FLOOR_PROFILE)
    rotor_region, wall_solid = build_solids(thin, FLOOR_BOTTOM - shell, top_y, top_y, impellers=args.impellers,
                                            clip_tips=args.clip_tips,
                                            with_blades=not (args.conformal_blades or impeller_plates),
                                            legacy_baffles=args.legacy_baffles,
                                            with_baffle_plates=not (args.conformal_baffles or baffle_plates),
                                            with_bell=not args.no_rotor_bell,
                                            with_disk=not impeller_plates or args.solid_disk is not None,
                                            with_shaft=not args.no_shaft,
                                            disk_thickness=None if args.solid_disk is None else args.solid_disk * dx)
    if args.impellers != "both":
        print(f"impellers={args.impellers} (single-impeller control)")

    sdf_interior = interior.signed_distance(sites)
    sdf_rotor = rotor_region.signed_distance(sites)
    sdf_wall_solid = wall_solid.signed_distance(sites)

    # Solids claim their interior plus a skin of `--skin` spacings (default 0.5).
    # The skin moves the effective solid surface outward by that amount, i.e. every
    # solid dimension grows by 2 * skin * dx (2026-09-28: measured on the 3 mm case,
    # blades 12.5 mm instead of the nominal 9 mm, Rushton tip particles at 49.2 mm
    # instead of 48.0). It is not needed to keep fluid and solid sites apart: they
    # share one lattice, so their centres are at least dx apart for any skin.
    # --skin 0 claims a site only when its centre lies inside the solid.
    skin = args.skin * dx
    is_rotor = sdf_rotor <= skin
    blade_points = np.zeros((0, 3))
    if args.conformal_blades or args.conformal_pbt is not None:
        # Blades: plane sheets of particles in the blade frames; the lattice sites inside the
        # sheets' boxes (true outline, thickness thin_layers * dx) are dropped, whatever they
        # would have been, and so are the lattice rotor sites (hub, disk) that would sit closer
        # than 0.6 dx to a blade particle.
        if args.conformal_pbt is not None:
            blade_points, blade_boxes = conformal_blades(dx, args.conformal_pbt, "pbt")
        else:
            blade_points, blade_boxes = conformal_blades(dx, args.thin_layers, args.impellers)
        in_blade = Union(*blade_boxes).signed_distance(sites) <= 0.0
        near = np.nonzero(is_rotor & ~in_blade)[0]
        if near.size:
            difference = sites[near][:, None, :] - blade_points[None, :, :]
            too_close = (np.einsum("ijk,ijk->ij", difference, difference) < (0.6 * dx) ** 2).any(axis=1)
            is_rotor[near[too_close]] = False
            in_blade[near[too_close]] = True
        is_rotor &= ~in_blade
    else:
        in_blade = np.zeros(sites.shape[0], dtype=bool)
    baffle_points = np.zeros((0, 3))
    if args.conformal_baffles:
        if args.legacy_baffles:
            parser.error("--conformal-baffles cannot be combined with --legacy-baffles")
        # Baffle plates: plane sheets in the baffle frames. The lattice sites inside the sheets'
        # boxes are dropped, and so is every other lattice site (fluid, bracket, shell) that
        # would sit closer than 0.6 dx to a baffle particle.
        baffle_points, baffle_boxes = conformal_baffles(dx, args.thin_layers, LIQUID_HEIGHT)
        sdf_baffle = Union(*baffle_boxes).signed_distance(sites)
        in_baffle = sdf_baffle <= 0.0
        near = np.nonzero(~in_baffle & (sdf_baffle < dx))[0]
        if near.size:
            too_close = np.zeros(near.size, dtype=bool)
            for start in range(0, near.size, 2000):
                block = sites[near[start:start + 2000]]
                difference = block[:, None, :] - baffle_points[None, :, :]
                too_close[start:start + 2000] = (np.einsum("ijk,ijk->ij", difference, difference)
                                                 < (0.6 * dx) ** 2).any(axis=1)
            in_baffle[near[too_close]] = True
        in_blade = in_blade | in_baffle          # from here on: "replaced by a conformal sheet"
    plates = []
    if args.thin_plates:
        # Thin plates: one layer of particles each. Lattice sites inside the slab of a plate (one
        # spacing thick) are dropped, and so is every other lattice site closer than 0.6 dx to a
        # plate particle. Where two plates meet (disk and blades) the particles of the later
        # plate that come closer than 0.6 dx to those of an earlier one are dropped.
        plates = thin_plate_sheets(dx, "rushton" if args.conformal_pbt is not None else args.impellers, LIQUID_HEIGHT, shell,
                                   args.legacy_baffles,
                                   RUSHTON_HUB["radius"] + skin,
                                   with_impeller=impeller_plates, with_baffles=baffle_plates,
                                   with_disk=args.solid_disk is None)
        kept = np.zeros((0, 3))
        for plate in plates:
            points = plate["points"]
            if kept.shape[0]:
                difference = points[:, None, :] - kept[None, :, :]
                points = points[(np.einsum("ijk,ijk->ij", difference, difference) >= (0.6 * dx) ** 2).all(axis=1)]
            plate["points"] = points
            kept = np.vstack([kept, points])
        sdf_plates = Union(*[plate["box"] for plate in plates]).signed_distance(sites)
        in_plate = sdf_plates <= 0.0
        near = np.nonzero(~in_plate & (sdf_plates < 2.0 * dx))[0]
        too_close = np.zeros(near.size, dtype=bool)
        for start in range(0, near.size, 2000):
            block = sites[near[start:start + 2000]]
            difference = block[:, None, :] - kept[None, :, :]
            too_close[start:start + 2000] = (np.einsum("ijk,ijk->ij", difference, difference)
                                             < (0.6 * dx) ** 2).any(axis=1)
        in_plate[near[too_close]] = True
        is_rotor &= ~in_plate
        in_blade = in_blade | in_plate
        print(f"thin plates: {len(plates)} plates, {kept.shape[0]:,} particles, "
              f"{int(in_plate.sum()):,} lattice sites dropped")
    is_wall_solid = (sdf_wall_solid <= skin) & ~is_rotor & ~in_blade
    is_fluid = (sdf_interior <= -0.5 * dx) & ~is_rotor & ~is_wall_solid & ~in_blade
    # Everything else inside the frame that is not liquid = tank shell (walls, floor, lid).
    is_shell = ~is_fluid & ~is_rotor & ~is_wall_solid & ~in_blade & (sdf_interior > -0.5 * dx)
    # Keep only shell sites within `border` layers of the liquid surface (drop far corners).
    is_shell &= sdf_interior <= shell + 0.5 * dx
    shell_points = np.zeros((0, 3))
    if args.smooth_walls:
        # --smooth-walls (2026-10-03): the cylinder and the dished floor get a shell that follows the surfaces.
        # Of the lattice shell only the flat lid is kept (y above the liquid, r < TANK_RADIUS + 0.5 dx), all of
        # it: shell particles closer than 0.6 dx to a lid site are dropped instead (thinning the lid at the
        # corner left an annular gap through which fluid climbed out, 12 particles in 0.2 s at rest). Shell
        # particles inside the rotor are dropped. Inside a static lattice solid (bearing boss, baffle brackets)
        # the solid stays and a shell particle is kept only where it fills a notch of the solid's staircase:
        # within one spacing of the solid's surface and no site of the solid closer than 0.6 dx (2026-10-04;
        # dropping all of them left the notches open: at 2 mm fluid from under the rotating bell went down
        # between the bearing boss and the floor shell and out of the domain, 54 particles in 0.2 s at rest,
        # 979 in 0.7 s stirred, and at two baffle brackets fluid sat in notches at the cylinder, r 145 mm).
        # Every other lattice site (static solids, fluid) closer than 0.6 dx to a shell particle is dropped.
        shell_points = smooth_tank_shell(dx, border)
        depth = wall_solid.signed_distance(shell_points)
        fills_notch = (depth > -dx) & ~points_closer_than(shell_points, sites[is_wall_solid], 0.6 * dx)
        outside_rotor = rotor_region.signed_distance(shell_points) > 0.0
        notch_count = int((fills_notch & (depth <= 0.0) & outside_rotor).sum())
        shell_points = shell_points[((depth > 0.0) | fills_notch) & outside_rotor]
        radius_sites = np.hypot(sites[:, 0], sites[:, 2])
        is_shell &= (sites[:, 1] > LIQUID_HEIGHT - 0.5 * dx) & (radius_sites < TANK_RADIUS + 0.5 * dx)
        shell_points = shell_points[~points_closer_than(shell_points, sites[is_shell], 0.6 * dx)]
        candidates = np.nonzero((is_wall_solid | is_fluid) & (sdf_interior < (border + 1) * dx))[0]
        too_close = candidates[points_closer_than(sites[candidates], shell_points, 0.6 * dx)]
        dropped_fluid = int(is_fluid[too_close].sum())
        is_shell[too_close] = False
        is_wall_solid[too_close] = False
        is_fluid[too_close] = False
        print(f"smooth walls: {shell_points.shape[0]:,} shell particles in {border} layers "
              f"({notch_count} of them in notches of lattice solids); "
              f"{too_close.size:,} lattice sites closer than 0.6 dx dropped, {dropped_fluid} of them fluid")

    fluid = sites[is_fluid]
    if plates:
        near_count = count_near_plates(fluid, plates, h)
        print(f"thin plates: {int((near_count > 0).sum()):,} fluid particles ({100.0 * (near_count > 0).mean():.2f} %) "
              f"have a plate inside their support; at most {int(near_count.max())} plates at once "
              f"(the solver keeps {MAX_NEAR_THIN_PLATES})")
        if near_count.max() > MAX_NEAR_THIN_PLATES:
            parser.error("more plates inside one support than the solver keeps (MAX_NEAR_THIN_PLATES)")
    wall = np.vstack([sites[is_wall_solid | is_shell], baffle_points, shell_points])
    if args.conformal_baffles:
        print(f"conformal baffles: {baffle_points.shape[0]:,} baffle particles in {args.thin_layers} layer(s)")
    rotor = np.vstack([sites[is_rotor], blade_points])
    if args.conformal_blades or args.conformal_pbt is not None:
        layers = args.conformal_pbt if args.conformal_pbt is not None else args.thin_layers
        print(f"conformal blades: {blade_points.shape[0]:,} blade particles in {layers} layer(s), "
              f"{int(in_blade.sum()):,} lattice sites dropped")
    n_plate_rotor = sum(plate["points"].shape[0] for plate in plates if plate["frame"] == "rotor")
    n_plate_static = sum(plate["points"].shape[0] for plate in plates if plate["frame"] == "static")
    n_fluid, n_wall, n_rotor = fluid.shape[0], wall.shape[0] + n_plate_static, rotor.shape[0] + n_plate_rotor
    total = n_fluid + n_wall + n_rotor
    pool_size = int(math.ceil(total * 1.15 / 128) * 128)

    # Voxel capacity: closest-packing bound for h^3 cube on a grid lattice.
    bound = int(math.ceil(math.sqrt(2) * args.hdx ** 3))
    max_per_voxel = args.max_per_voxel or max(64, int(2 ** math.ceil(math.log2(bound * 1.3))))
    c0 = args.c0_factor * TIP_SPEED
    omega = +2 * math.pi * IMPELLER_RPM / 60.0          # right-hand sign about +y; see CASE_YAML comment (M-Star -200 rpm = CCW from above)

    liquid_volume = interior.volume()
    print(f"fluid={n_fluid:,} wall={n_wall:,} rotor={n_rotor:,} total={total:,} pool={pool_size:,}")
    print(f"fluid volume check: {n_fluid * dx ** 3 * 1e3:.2f} L of particles vs {liquid_volume * 1e3:.2f} L "
          f"dished tank (minus solids); Rushton clearance {0.5 * (RUSHTON_BLADE['y0'] + RUSHTON_BLADE['y1']) - FLOOR_BOTTOM:.4f} m")
    print(f"max_per_voxel={max_per_voxel} (bound {bound}), c0={c0:.2f} m/s, tip speed {TIP_SPEED:.3f} m/s, "
          f"dt = {args.cfl * h / c0:.3e} s")

    out = _REPO_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    write_obj(out / "fluid.obj", fluid)
    write_obj(out / "wall.obj", wall)
    write_obj(out / "rotor.obj", rotor)
    for plate in plates:
        write_obj(out / f"plate_{plate['name']}.obj", plate["points"])
    all_points = np.vstack([fluid, wall, rotor] + [plate["points"] for plate in plates])
    write_frame_obj(out / "frame.obj", all_points.min(axis=0) - 0.6 * dx, all_points.max(axis=0) + 0.6 * dx)
    case_text = CASE_YAML.format(
        dx=dx, h=h, hdx=args.hdx, thin_layers=args.thin_layers, n_fluid=n_fluid, n_wall=n_wall,
        n_rotor=n_rotor, liquid_height=LIQUID_HEIGHT, tip_speed=TIP_SPEED, c0_factor=args.c0_factor, radius=0.5 * dx, c0=c0,
        pool_size=pool_size, max_per_voxel=max_per_voxel, max_incoming=args.max_incoming,
        ramp_time=args.ramp_time, gravity_y=-abs(args.gravity), xi=args.xi, cfl=args.cfl)
    if args.background_pressure != 0.0:
        # written only when used, so that the files of all other cases stay as they were
        assert case_text.count("  gravity: [") == 1
        case_text = case_text.replace("  gravity: [", f"  background_pressure: {args.background_pressure:g}\n  gravity: [")
    if args.solid_pressure != "increment" or args.solid_reaction_force:
        assert case_text.count("  use_pst: true") == 1
        case_text = case_text.replace("  use_pst: true", f"  solid_pressure: {args.solid_pressure}\n"
                                      f"  solid_reaction_force: true\n"
                                      f"  solid_pressure_offset: {args.solid_pressure_offset:g}\n  use_pst: true")
    if args.pst_main is not None:
        assert case_text.count("  pst_main: 0.1\n") == 1
        case_text = case_text.replace("  pst_main: 0.1\n", f"  pst_main: {args.pst_main:g}\n")
    if args.symmetric_pair_correction:
        assert case_text.count("  use_pst: true") == 1
        case_text = case_text.replace("  use_pst: true", "  symmetric_pair_correction: true\n  use_pst: true")
    if args.momentum_sgs is not None:
        width_text = args.momentum_sgs_width
        width = {"dx": args.dx, "2dx": 2.0 * args.dx, "h": args.hdx * args.dx}.get(width_text)
        if width is None:
            width = float(width_text)
        damping = "  momentum_sgs_wall_damping: true\n" if args.momentum_sgs_wall_damping else ""
        case_text = case_text.replace(
            "  use_pst: true",
            f"  momentum_sgs: true\n  momentum_sgs_cs: {args.momentum_sgs:g}\n  momentum_sgs_filter_width: {width:.6g}\n"
            f"{damping}  use_pst: true")
    if args.shift_transport != "none":
        case_text = case_text.replace("  use_pst: true", f"  shift_transport: {args.shift_transport}\n  use_pst: true")
    if args.no_solid_density_floor:
        case_text = case_text.replace("  use_pst: true", "  solid_density_floor: false\n  use_pst: true")
    if args.pair_correction != "own":
        assert not args.symmetric_pair_correction, "use --pair-correction alone"
        assert case_text.count("  use_pst: true") == 1
        case_text = case_text.replace("  use_pst: true", f"  pair_correction: {args.pair_correction}\n  use_pst: true")
    if args.diffusion_gradient_term:
        assert case_text.count("  use_pst: true") == 1
        case_text = case_text.replace("  use_pst: true", "  density_diffusion_gradient_term: true\n  use_pst: true")
    if args.pst_near_solid != "full":
        assert case_text.count("  use_pst: true") == 1
        case_text = case_text.replace("  use_pst: true", f"  pst_near_solid: {args.pst_near_solid}\n  use_pst: true")
    if args.hydrostatic:
        if args.gravity == 0.0:
            parser.error("--hydrostatic needs --gravity")
        assert case_text.count("  gravity: [") == 1
        case_text = case_text.replace("  gravity: [", f"  hydrostatic_reference: [0.0, {LIQUID_HEIGHT}, 0.0]\n  gravity: [")
    if args.thin_plates:
        assert case_text.count("    - {file: rotor.obj, material: impeller}\n") == 1
        entries = "".join(
            f"    - {{file: plate_{plate['name']}.obj, "
            f"material: {'impeller' if plate['frame'] == 'rotor' else 'tank_wall'}, thin_plate: {plate['name']}}}\n"
            for plate in plates)
        case_text = case_text.replace("    - {file: rotor.obj, material: impeller}\n",
                                      "    - {file: rotor.obj, material: impeller}\n" + entries)
        case_text += thin_plates_block(plates)
        if args.thin_plate_dashpot is not None:
            assert case_text.count("  use_pst: true") == 1
            case_text = case_text.replace("  use_pst: true",
                                          f"  thin_plate_dashpot: {args.thin_plate_dashpot:g}\n  use_pst: true")
    (out / "case.yaml").write_text(case_text, encoding="utf-8")
    (out / "materials.yaml").write_text(MATERIALS_YAML.format(omega=omega, viscosity=args.viscosity), encoding="utf-8")
    if args.state_limited and (not args.substrate or args.no_uptake or args.growth_yield <= 0.0):
        parser.error("--state-limited needs --substrate, no --no-uptake and --growth-yield > 0")
    if args.tracers > 0 or args.substrate:
        substrate_fields = (7 if args.state_limited else 4) if args.substrate else 0
        if args.tracers + substrate_fields > 12:
            parser.error("at most 12 scalar fields: --substrate uses 4 (7 with --state-limited), so --tracers <= 8 (5)")
        with open(out / "case.yaml", "a", encoding="utf-8") as handle:
            handle.write(scalars_block(args, h))
        print(f"scalars: {args.tracers} tracer(s), pulses from t = {args.injection_start} s every "
              f"{args.injection_interval} s, radius {args.injection_radius:.4f} m, sgs={'on' if args.sgs else 'off'}")
        if args.substrate:
            print(f"substrate: Monod q_max {args.q_max_umol_per_g_h:g} umol/(g h), K_s {args.k_s:.2e} mol/kg, "
                  f"X {args.biomass:g} g/kg, uptake {'off' if args.no_uptake else 'on'}; feed {args.feed_rate:.3e} mol/s "
                  f"from t = {args.feed_start:g} s, sphere r = {args.feed_radius:g} m")
    print(f"wrote fluid.obj wall.obj rotor.obj frame.obj case.yaml materials.yaml -> {out}")
    if not args.no_preview:
        write_preview(out / "split_preview.png", fluid, wall, rotor, dx)
        print("wrote split_preview.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
