"""
_demo_stirred_tank_30l.py — particle case for the 30 L stirred tank of
Rautenbach et al. (2026), Comput. Chem. Eng., dataset DARUS-5523.

All dimensions were measured from the dataset's STL files
(00_simulation_for_mstar_video/StaticBody.stl, Moving Body_1.stl) on
2026-09-25; see log/ for the measurement notes. Axis convention follows the
STL: y is the tank axis (up), origin at the centre of the flat tank floor.

Geometry (metres):
    tank        inner radius 0.144, flat floor at y = 0, liquid height 0.4265
                (the M-Star run uses a flat lid there, no free surface)
    baffles     3 (120 deg apart, azimuths 62/182/302 deg from +x toward +z),
                radial 0.1254..0.1384, thickness 2.65e-3, full height
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
from utils.geometry.region import Region, Box, Cylinder, Union      # noqa: E402

# ----------------------------------------------------------------------------
# Measured dimensions (m)
# ----------------------------------------------------------------------------
TANK_RADIUS = 0.144
LIQUID_HEIGHT = 0.4265
TANK_WALL_TOP = 0.5

BAFFLE_AZIMUTHS_DEG = (62.0, 182.0, 302.0)
BAFFLE_RADIAL = (0.1254, 0.1384)
BAFFLE_THICKNESS = 2.65e-3

SHAFT_RADIUS = 0.004

RUSHTON_HUB = dict(radius=0.0102, y0=0.0211, y1=0.0414)
RUSHTON_DISK = dict(radius=0.032, y0=0.0372, y1=0.0397)
RUSHTON_BLADE = dict(radial=(0.024, 0.048), y0=0.0288, y1=0.048, thickness=2.2e-3, azimuth0_deg=23.1)

PBT_HUB = dict(radius=0.0109, y0=0.180, y1=0.2093)
PBT_BLADE = dict(radial=(0.0122, 0.0491), chord=0.0273, thickness=2.5e-3,
                 center_y=0.19465, pitch_deg=45.0, azimuth0_deg=24.3)

PROBE_HALF = 0.0075
PROBE_Y = (0.406, 0.456)
PROBE_CENTERS = ((-0.127, -0.018), (0.053, -0.118))

IMPELLER_RPM = 200.0                      # M-Star: -200 rpm about +y
TIP_SPEED = math.pi * 2 * 0.0491 * IMPELLER_RPM / 60.0


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


def build_solids(thin, shaft_y0, shaft_y1, top_y):
    """Return (rotor_region, wall_solid_region). ``thin`` = minimum thickness."""
    t_baffle = max(BAFFLE_THICKNESS, thin)
    t_rblade = max(RUSHTON_BLADE["thickness"], thin)
    t_disk = max(RUSHTON_DISK["y1"] - RUSHTON_DISK["y0"], thin)
    t_pblade = max(PBT_BLADE["thickness"], thin)

    disk_mid = 0.5 * (RUSHTON_DISK["y0"] + RUSHTON_DISK["y1"])
    rotor_parts = [
        y_cylinder(SHAFT_RADIUS, shaft_y0, shaft_y1),
        y_cylinder(RUSHTON_HUB["radius"], RUSHTON_HUB["y0"], RUSHTON_HUB["y1"]),
        y_cylinder(RUSHTON_DISK["radius"], disk_mid - 0.5 * t_disk, disk_mid + 0.5 * t_disk),
        y_cylinder(PBT_HUB["radius"], PBT_HUB["y0"], PBT_HUB["y1"]),
    ]
    for k in range(6):
        rotor_parts.append(radial_slab(RUSHTON_BLADE["azimuth0_deg"] + 60 * k,
                                       *RUSHTON_BLADE["radial"], RUSHTON_BLADE["y0"],
                                       RUSHTON_BLADE["y1"], t_rblade))
        rotor_parts.append(pitched_blade(PBT_BLADE["azimuth0_deg"] + 60 * k,
                                         *PBT_BLADE["radial"], PBT_BLADE["chord"], t_pblade,
                                         PBT_BLADE["center_y"], PBT_BLADE["pitch_deg"]))
    wall_parts = []
    for az in BAFFLE_AZIMUTHS_DEG:
        wall_parts.append(radial_slab(az, *BAFFLE_RADIAL, 0.0, top_y, t_baffle))
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
# the impeller tip speed only (10 * {tip_speed:.3f} m/s).
# Rotor: shaft + both impellers rotate rigidly about +y through the origin
# (predict.comp ROTOR branch, 2026-09-25). M-Star's "-200 rpm about +y" is
# a NEGATIVE angular velocity in the right-hand sense, i.e. the blades move
# from +x toward +z; the PBT (y decreasing toward +theta) then pumps DOWN.

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
  cfl: 0.15
  gravity: [0.0, {gravity_y:.3f}, 0.0]

numerics:
  use_density_diffusion: true
  delta_coefficient: 0.1
  use_kcg_correction: true
  regularization:
    xi: 0.1
    det_threshold: 1.0e-4
    frobenius_max: 10.0
  use_pst: true
  pst_main: 0.1
  pst_anti: 0.0005
  defrag_enabled: true
  defrag_cadence: 1000
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
  viscosity: 1.0e-6

tank_wall:
  kind: boundary
  rest_density: 998.0
  viscosity: 1.0e-6

impeller:
  kind: rotor
  rest_density: 998.0
  viscosity: 1.0e-6
  rotor_angular_velocity: {omega:.5f}   # rad/s, signed: -200 rpm about +y (right-hand rule)
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dx", type=float, default=0.003, help="particle spacing (m)")
    parser.add_argument("--hdx", type=int, default=3, help="h/dx (kernel support radius in spacings)")
    parser.add_argument("--thin-layers", type=int, default=3, help="minimum layers across thin solids")
    parser.add_argument("--border", type=int, default=None, help="wall shell layers (default = hdx)")
    parser.add_argument("--out", default="cases/stirred_tank_30l")
    parser.add_argument("--max-per-voxel", type=int, default=None)
    parser.add_argument("--max-incoming", type=int, default=32)
    parser.add_argument("--ramp-time", type=float, default=0.05, help="rotor spin-up time (s)")
    parser.add_argument("--c0-factor", type=float, default=10.0, help="speed of sound = factor * tip speed")
    parser.add_argument("--gravity", type=float, default=0.0,
                        help="gravity magnitude along -y (0 = off; with gravity on use --c0-factor 20 so that c0 >= 10*sqrt(g*H))")
    parser.add_argument("--no-preview", action="store_true")
    args = parser.parse_args()

    dx = args.dx
    h = args.hdx * dx
    border = args.border if args.border is not None else args.hdx
    thin = args.thin_layers * dx
    shell = border * dx
    top_y = LIQUID_HEIGHT + shell                # top of lid shell

    # Lattice sites over the whole frame (anchored at the origin by tile_bounding_box).
    lo = np.array([-(TANK_RADIUS + shell), -shell, -(TANK_RADIUS + shell)])
    hi = np.array([TANK_RADIUS + shell, top_y, TANK_RADIUS + shell])
    sites = tile_bounding_box(lo - 0.5 * dx, hi + 0.5 * dx, dx, LATTICE_GRID, 3)
    print(f"dx={dx:.4e} h={h:.4e} (h/dx={args.hdx}) border={border} thin>={args.thin_layers} layers -> {sites.shape[0]:,} lattice sites")

    interior = y_cylinder(TANK_RADIUS, 0.0, LIQUID_HEIGHT)
    rotor_region, wall_solid = build_solids(thin, -shell, top_y, top_y)

    sdf_interior = interior.signed_distance(sites)
    sdf_rotor = rotor_region.signed_distance(sites)
    sdf_wall_solid = wall_solid.signed_distance(sites)

    # Solids claim their interior plus a half-spacing skin so no fluid site sits
    # closer than ~0.5 dx to a solid surface.
    is_rotor = sdf_rotor <= 0.5 * dx
    is_wall_solid = (sdf_wall_solid <= 0.5 * dx) & ~is_rotor
    is_fluid = (sdf_interior <= -0.5 * dx) & ~is_rotor & ~is_wall_solid
    # Everything else inside the frame that is not liquid = tank shell (walls, floor, lid).
    is_shell = ~is_fluid & ~is_rotor & ~is_wall_solid & (sdf_interior > -0.5 * dx)
    # Keep only shell sites within `border` layers of the liquid surface (drop far corners).
    is_shell &= sdf_interior <= shell + 0.5 * dx

    fluid = sites[is_fluid]
    wall = sites[is_wall_solid | is_shell]
    rotor = sites[is_rotor]
    n_fluid, n_wall, n_rotor = fluid.shape[0], wall.shape[0], rotor.shape[0]
    total = n_fluid + n_wall + n_rotor
    pool_size = int(math.ceil(total * 1.15 / 128) * 128)

    # Voxel capacity: closest-packing bound for h^3 cube on a grid lattice.
    bound = int(math.ceil(math.sqrt(2) * args.hdx ** 3))
    max_per_voxel = args.max_per_voxel or max(64, int(2 ** math.ceil(math.log2(bound * 1.3))))
    c0 = args.c0_factor * TIP_SPEED
    omega = -2 * math.pi * IMPELLER_RPM / 60.0          # signed: -200 rpm about +y

    liquid_volume = math.pi * TANK_RADIUS ** 2 * LIQUID_HEIGHT
    print(f"fluid={n_fluid:,} wall={n_wall:,} rotor={n_rotor:,} total={total:,} pool={pool_size:,}")
    print(f"fluid volume check: {n_fluid * dx ** 3 * 1e3:.2f} L of particles vs {liquid_volume * 1e3:.2f} L cylinder")
    print(f"max_per_voxel={max_per_voxel} (bound {bound}), c0={c0:.2f} m/s, tip speed {TIP_SPEED:.3f} m/s, "
          f"dt = {0.15 * h / c0:.3e} s")

    out = _REPO_ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    write_obj(out / "fluid.obj", fluid)
    write_obj(out / "wall.obj", wall)
    write_obj(out / "rotor.obj", rotor)
    all_points = np.vstack([fluid, wall, rotor])
    write_frame_obj(out / "frame.obj", all_points.min(axis=0) - 0.6 * dx, all_points.max(axis=0) + 0.6 * dx)
    (out / "case.yaml").write_text(CASE_YAML.format(
        dx=dx, h=h, hdx=args.hdx, thin_layers=args.thin_layers, n_fluid=n_fluid, n_wall=n_wall,
        n_rotor=n_rotor, liquid_height=LIQUID_HEIGHT, tip_speed=TIP_SPEED, radius=0.5 * dx, c0=c0,
        pool_size=pool_size, max_per_voxel=max_per_voxel, max_incoming=args.max_incoming,
        ramp_time=args.ramp_time, gravity_y=-abs(args.gravity)), encoding="utf-8")
    (out / "materials.yaml").write_text(MATERIALS_YAML.format(omega=omega), encoding="utf-8")
    print(f"wrote fluid.obj wall.obj rotor.obj frame.obj case.yaml materials.yaml -> {out}")
    if not args.no_preview:
        write_preview(out / "split_preview.png", fluid, wall, rotor, dx)
        print("wrote split_preview.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
