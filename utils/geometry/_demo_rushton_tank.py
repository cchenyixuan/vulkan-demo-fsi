"""
_demo_rushton_tank.py — particle cases for the two tanks of Haringa (2023), Eng. Life Sci.
23:e2100159 (2026-10-02, reproduction H1 to H3, docs/haringa2023_reproduction_design_2026-10-02.md).

Presets (section 2.2 and 2.3 of the paper):
    jahoda      1-impeller tank of Jahoda et al. (2007): flat bottom, T = 0.29 m, H = T, 6-blade
                Rushton D = T/3 at clearance C = T/3, four baffles T/10 over the full height,
                N = 300 rpm. H1: discharge profiles (U_rad, k, epsilon) and power number.
    haringa54   54 m3 penicillin fermentor: flat bottom, T = 3 m, H = 7.7 m, lower 8-blade
                Rushton at C = 0.9 m, upper 6-blade Rushton 3.0 m higher, both D = 1.3 m, shaft
                0.27 m, four baffles T/10 over the full height, N = 98 rpm. H2: mixing time,
                H3: lifelines.
The paper does not give the impeller details; the standard Rushton proportions are used:
blade length D/4 (radial D/4 .. D/2), blade height D/5 centred on the disk, disk diameter 3D/4.
The 1-impeller shaft (not given) is 12 mm. No hub: the disk starts at the shaft.

Internals: the blades and the disks are thin plates (the paper's "2D sheet bodies"): one layer of
particles on their mid-plane (case block `thin_plates:`). The baffles are by default solid, three
layers of wall particles centred on the baffle plane (6 mm at dx = 2 mm; the paper's 3D bodies are
5 mm thick): thin-plate baffles (--baffles plates) let the fluid at rest drift in density, see
below. The top is a flat lid at the liquid height (no free surface, as in the paper);
--free-slip-lid makes it a no-shear wall like the paper's top (material flag free_slip). Fluid
rho = 1000 kg/m3, nu = 1e-6 m2/s, no gravity, background pressure p_b = factor * rho * U_tip^2
(factor 2 = the 2000 Pa of the 30 L runs in tip-speed units), c0 = 20 U_tip, accumulate walls
with reaction force, pair correction reverse: the standard setup of the 30 L validation.

Frame: y is the tank axis (up), the bottom at y = 0, the lid at y = H. Azimuth theta from +x
toward +z (radial_frame of the 30 L generator); the baffles sit at theta = 0, 90, 180, 270 deg,
so theta = 0 is the paper's baffle plane.

Lattice and plates (2026-10-02, log/2026-10-02_haringa-h1-setup.md): the lattice is shifted by half
a spacing in y only, so that the bottom and the lid lie half a spacing outside the first fluid
layer. A plane plate must REPLACE a lattice layer: the correction matrix and the kernel sum count
the plate particles as neighbours, the force and the continuity equation count them as wall
dummies, so plate plus fluid must form one regular point set. The baffle planes x = 0 and z = 0
are lattice planes, and the disks are snapped to the nearest lattice layer y = (k + 1/2) dx
(printed; C = T/3 = 96.67 mm becomes 97 mm at dx = 2 mm). Findings of the static checks (rotor at
rest, 4 mm): with the plates between two layers (--lattice-offset xyz) the fluid compresses by 1 %
in 0.3 s; with thin-plate baffles on lattice planes it still drifts by 10 to 25 kg/m3 per second
and accelerates, for every variant tried (outline continued into the shells or not, baffles at
general angles, detached from wall and floor, the 30 L wall convention, the 30 L c0 and p_b);
without p_b nothing moves, without baffle plates nothing drifts. The wall dummies of a plate take
no part in the density diffusion, so unlike a wall a plate does not pull the density back to rho0.
Whether it runs away depends on the plate area (both faces) over the wall area: 0.17 (four baffles)
runs away, 0.08 (two baffles, or four of half width) and 0.03 (impeller plates only) do not drift;
the 30 L tank with all its plates (0.16) drifts by about 1 kg/m3 per second at rest, slowing down.
Solid baffles do not drift (mean density constant to 0.03 kg/m3 over 1.8 s rotating).
Particles of a wall shell lying exactly on a plate's plane inside its outline have an ambiguous
side (static tank, unshifted planes, outline continued into the shells: +-0.3 N m per baffle);
the frame vectors of the baffles are cleaned of rounding residues (cos 90 deg = 6e-17).

Besides case.yaml and materials.yaml the script writes tank.json (geometry for the analysis
scripts) and, with --flow-statistics, flow_statistics.json (sample points of
experiment/v1/utils/flow_statistics.py).

Usage (repo root):
    python utils/geometry/_demo_rushton_tank.py --preset jahoda --dx 0.002 --free-slip-lid --flow-statistics
    python utils/geometry/_demo_rushton_tank.py --preset haringa54 --dx 0.025 --free-slip-lid --tracers 8
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.geometry import LATTICE_GRID, tile_bounding_box          # noqa: E402
from utils.geometry.region import Cylinder, Difference, Union      # noqa: E402
from utils.geometry._demo_stirred_tank_30l import (                 # noqa: E402
    MAX_NEAR_THIN_PLATES, OrientedBox, count_near_plates, radial_frame, thin_plates_block,
    write_frame_obj, write_obj, y_cylinder)

# Monod kinetics and feed of Haringa (2023): q_max 1600 umol/(g h), K_s 7.8 umol/kg, X 55 g/kg,
# initial substrate 10 K_s, feed 0.37 mol/s into the tracer injection sphere (54 m3 tank).
HARINGA_Q_MAX_UMOL_PER_G_H = 1600.0
HARINGA_K_S = 7.8e-6
HARINGA_BIOMASS = 55.0
HARINGA_FEED_RATE = 0.37

PRESETS = {
    "jahoda": dict(
        title="Haringa (2023) 1-impeller tank, geometry of Jahoda et al. (2007)",
        tank_diameter=0.29, liquid_height=0.29, baffle_count=4, baffle_width=0.029, baffle_azimuth0_deg=0.0,
        shaft_diameter=0.012, rpm=300.0, ramp_time=0.1,
        impellers=(dict(name="rushton", blades=6, diameter=0.29 / 3.0, center_height=0.29 / 3.0, azimuth0_deg=15.0),),
        injection=None, probes=(), release_sphere=None,
    ),
    "haringa54": dict(
        title="Haringa (2023) 54 m3 penicillin fermentor",
        tank_diameter=3.0, liquid_height=7.7, baffle_count=4, baffle_width=0.3, baffle_azimuth0_deg=0.0,
        shaft_diameter=0.27, rpm=98.0, ramp_time=0.5,
        impellers=(dict(name="lower", blades=8, diameter=1.3, center_height=0.9, azimuth0_deg=11.25),
                   dict(name="upper", blades=6, diameter=1.3, center_height=3.9, azimuth0_deg=15.0)),
        # tracer injection and feed: sphere of 0.4 m diameter at y = 7.35 m, r = 0.8 m in the baffle
        # plane theta = 0; probe at the bottom, y = 0.25 m, r = 0.75 m, theta = 180 deg; parcels released
        # in a sphere of 0.2 m diameter at y = 3 m, r = 0.5 m, theta = 0
        injection=dict(center=(0.8, 7.35, 0.0), radius=0.2),
        probes=(("probe_bottom", (-0.75, 0.25, 0.0)),),
        release_sphere=dict(center=(0.5, 3.0, 0.0), radius=0.1),
    ),
}


def closer_than(points, reference, distance):
    """Mask of the points that lie closer than `distance` to any reference point (cKDTree when
    scipy is available, blocks of brute force otherwise)."""
    mask = np.zeros(points.shape[0], dtype=bool)
    if points.shape[0] == 0 or reference.shape[0] == 0:
        return mask
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        cKDTree = None
    if cKDTree is not None:
        nearest, _ = cKDTree(reference).query(points, k=1, distance_upper_bound=distance)
        return nearest < distance
    for start in range(0, points.shape[0], 500):
        block = points[start:start + 500]
        for first in range(0, reference.shape[0], 4000):
            difference = block[:, None, :] - reference[None, first:first + 4000, :]
            mask[start:start + 500] |= (np.einsum("ijk,ijk->ij", difference, difference) < distance ** 2).any(axis=1)
    return mask


def rushton_dimensions(impeller):
    """Standard Rushton proportions: blades radial D/4 .. D/2, height D/5 centred on the disk,
    disk radius 3D/8."""
    diameter = impeller["diameter"]
    return dict(blade_inner=0.25 * diameter, blade_outer=0.5 * diameter, blade_height=0.2 * diameter,
                disk_radius=0.375 * diameter, tip_radius=0.5 * diameter)


def plane_grid(centre, axis_a, axis_b, half_a, half_b, dx):
    """Particles of a plane rectangle: spacing <= dx, the outermost centres half a cell inside the
    outline (as conformal_blades() of the 30 L generator). Returns (points, area per point)."""
    cells_a = max(1, int(math.ceil(2.0 * half_a / dx - 1e-9)))
    cells_b = max(1, int(math.ceil(2.0 * half_b / dx - 1e-9)))
    a = -half_a + (np.arange(cells_a) + 0.5) * (2.0 * half_a / cells_a)
    b = -half_b + (np.arange(cells_b) + 0.5) * (2.0 * half_b / cells_b)
    grid_a, grid_b = np.meshgrid(a, b, indexing="ij")
    points = centre + grid_a.reshape(-1, 1) * axis_a + grid_b.reshape(-1, 1) * axis_b
    return points, (2.0 * half_a / cells_a) * (2.0 * half_b / cells_b)


def tank_plates(preset, dx, shell, shaft_radius, baffle_extension="all", baffle_gap=0.0, baffle_bottom=0.0):
    """Thin plates (dicts as thin_plate_sheets() of the 30 L generator): per impeller the blades
    (rotor frame, at rotor angle 0) and the disk; the baffles (static), whose outline continues
    `shell` into the floor, the lid and the wall, their particles covering the liquid only."""
    plates = []
    for impeller in preset["impellers"]:
        size = rushton_dimensions(impeller)
        height = impeller["center_height"]
        for blade in range(impeller["blades"]):
            e_r, e_t, e_y = radial_frame(impeller["azimuth0_deg"] + blade * 360.0 / impeller["blades"])
            half_a = 0.5 * (size["blade_outer"] - size["blade_inner"])
            half_b = 0.5 * size["blade_height"]
            centre = e_r * 0.5 * (size["blade_inner"] + size["blade_outer"]) + e_y * height
            points, measure = plane_grid(centre, e_r, e_y, half_a, half_b, dx)
            plates.append(dict(name=f"{impeller['name']}_blade_{blade + 1}", shape="rectangle", frame="rotor",
                               centre=centre, normal=e_t, axis_a=e_r, extent=(half_a, half_b), thickness=0.0,
                               points=points, measure=measure,
                               box=OrientedBox(centre, np.stack([e_r, e_y, e_t]), [half_a, half_b, 0.5 * dx])))
        outer, inner = size["disk_radius"], shaft_radius
        rings = max(1, int(math.ceil((outer - inner) / dx - 1e-9)))
        ring_width = (outer - inner) / rings
        points = []
        for ring in range(rings):
            radius = inner + (ring + 0.5) * ring_width
            count = max(6, int(round(2.0 * math.pi * radius / dx)))
            angle = (np.arange(count) + 0.5 * (ring % 2)) * (2.0 * math.pi / count)
            points.append(np.column_stack([radius * np.cos(angle), np.full(count, height), radius * np.sin(angle)]))
        points = np.vstack(points)
        plates.append(dict(name=f"{impeller['name']}_disk", shape="annulus", frame="rotor",
                           centre=np.array([0.0, height, 0.0]), normal=np.array([0.0, 1.0, 0.0]),
                           axis_a=np.array([1.0, 0.0, 0.0]), extent=(outer, inner), thickness=0.0, points=points,
                           measure=math.pi * (outer ** 2 - inner ** 2) / points.shape[0],
                           box=Difference(y_cylinder(outer, height - 0.5 * dx, height + 0.5 * dx),
                                          y_cylinder(inner, height - dx, height + dx))))
    radius, top = 0.5 * preset["tank_diameter"], preset["liquid_height"]
    for index in range(preset["baffle_count"]):
        # exact zeros in the frame (cos 90 deg = 6e-17): a particle on the plane is then exactly on it
        e_r, e_t, e_y = (np.where(np.abs(axis) < 1e-12, 0.0, axis)
                         for axis in radial_frame(preset["baffle_azimuth0_deg"] + index * 360.0 / preset["baffle_count"]))
        r1 = radius - baffle_gap
        r0 = r1 - preset["baffle_width"]
        half_a, half_b = 0.5 * (r1 - r0), 0.5 * (top - baffle_bottom)
        centre = e_r * 0.5 * (r0 + r1) + e_y * 0.5 * (top + baffle_bottom)
        if baffle_gap > 0.0 or baffle_bottom > 0.0:
            baffle_extension = "lid" if baffle_extension == "all" else baffle_extension
        points, measure = plane_grid(centre, e_r, e_y, half_a, half_b, dx)
        radial_extension = shell if baffle_extension == "all" else 0.0
        bottom_extension = shell if baffle_extension == "all" else 0.0
        top_extension = 0.0 if baffle_extension == "none" else shell
        plates.append(dict(name=f"baffle_{index + 1}", shape="rectangle", frame="static",
                           centre=centre + e_r * 0.5 * radial_extension + e_y * 0.5 * (top_extension - bottom_extension),
                           normal=e_t, axis_a=e_r,
                           extent=(half_a + 0.5 * radial_extension, half_b + 0.5 * (top_extension + bottom_extension)),
                           thickness=0.0, points=points,
                           measure=measure,
                           box=OrientedBox(centre, np.stack([e_r, e_y, e_t]), [half_a, half_b, 0.5 * dx])))
    return plates


def flow_statistics_sets(preset, dx):
    """Sample points of the flow statistics (experiment/v1/utils/flow_statistics.py) for the
    1-impeller tank, Haringa (2023) Figure 1: the impeller discharge at y = C on the four lines
    midway between the baffles, from the blade tip r_tip to the wall R, at (r - r_tip) / (R - r_tip)
    = 0, 0.025, ..., 1 (blade-phase resolved, with velocity gradients); a cylinder just outside the
    blade tips over twice the blade height (pumping number); two r-y planes midway between the
    baffles at spacing 2 dx (mean flow pattern)."""
    radius, top = 0.5 * preset["tank_diameter"], preset["liquid_height"]
    impeller = preset["impellers"][0]
    size = rushton_dimensions(impeller)
    height = impeller["center_height"]
    mid_baffle = [preset["baffle_azimuth0_deg"] + (index + 0.5) * 360.0 / preset["baffle_count"]
                  for index in range(preset["baffle_count"])]
    sets = []
    fraction = np.linspace(0.0, 1.0, 41)
    for azimuth in mid_baffle:
        e_r, _, _ = radial_frame(azimuth)
        r = size["tip_radius"] + fraction * (radius - size["tip_radius"])
        points = r[:, None] * e_r + np.array([0.0, height, 0.0])
        sets.append(dict(name=f"discharge_{azimuth:05.1f}", points=points.tolist(), gradient=True, phase=True,
                         coordinate=fraction.tolist(), coordinate_name="(r - r_tip) / (R - r_tip)"))
    heights = height + np.arange(-size["blade_height"], size["blade_height"] + 0.5 * dx, 0.5 * dx)
    azimuths = np.arange(72) * 5.0
    tip = size["tip_radius"] + dx
    points = np.array([[tip * math.cos(math.radians(a)), y, tip * math.sin(math.radians(a))]
                       for y in heights for a in azimuths])
    sets.append(dict(name="tip_cylinder", points=points.tolist(), gradient=False, phase=False,
                     radius=tip, heights=heights.tolist(), azimuths_deg=azimuths.tolist()))
    spacing = 2.0 * dx
    r_values = np.arange(0.5 * spacing, radius, spacing)
    y_values = np.arange(0.5 * spacing, top, spacing)
    for azimuth in mid_baffle[:2]:
        e_r, _, _ = radial_frame(azimuth)
        grid_r, grid_y = np.meshgrid(r_values, y_values, indexing="ij")
        points = grid_r.reshape(-1, 1) * e_r + np.column_stack([np.zeros(grid_y.size), grid_y.ravel(),
                                                                np.zeros(grid_y.size)])
        sets.append(dict(name=f"plane_{azimuth:05.1f}", points=points.tolist(), gradient=False, phase=False,
                         r_values=r_values.tolist(), y_values=y_values.tolist()))
    return sets


CASE_YAML = """\
schema_version: 2

# {title} (Haringa 2023, Eng. Life Sci. 23:e2100159), 3D.
# GENERATED by utils/geometry/_demo_rushton_tank.py --preset {preset}.
#   T = {tank_diameter} m, H = {liquid_height} m, N = {rpm:g} rpm, U_tip = {tip_speed:.4f} m/s
#   dx = {dx:.6f} m, h = {h:.6f} m (h/dx = {hdx}), c0 = {c0_factor:g} U_tip, p_b = {pressure_factor:g} rho U_tip^2
#   fluid {n_fluid:,} / wall {n_wall:,} / lid {n_lid:,} / rotor {n_rotor:,} (thin plates included)
# Flat lid at y = H ({lid_kind}), no gravity. Rotor (shaft, disks, blades) about +y through the
# origin, positive angular velocity (the blades move from +x toward -z).

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
  speed_of_sound: {c0:.4f}
  power: 7
  cfl: {cfl:g}
  background_pressure: {background_pressure:.6g}
  gravity: [0.0, 0.0, 0.0]

numerics:
  use_density_diffusion: true
  delta_coefficient: 0.1
  use_kcg_correction: true
  regularization:
    xi: 0.01
    det_threshold: 1.0e-4
    frobenius_max: 10.0
  solid_pressure: accumulate
  solid_reaction_force: true
  solid_pressure_offset: 0
  pair_correction: reverse
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
  ramp_time: {ramp_time:.4f}

geometry:
  frame: frame.obj
  particles:
    - {{file: fluid.obj, material: tank_water}}
    - {{file: wall.obj,  material: tank_wall}}
    - {{file: lid.obj,   material: tank_lid}}
    - {{file: rotor.obj, material: impeller}}
"""

MATERIALS_YAML = """\
schema_version: 1

# Case-local material library (utils/geometry/_demo_rushton_tank.py).
tank_water:
  kind: fluid
  rest_density: 1000.0
  viscosity: {viscosity:.6e}

tank_wall:
  kind: boundary
  rest_density: 1000.0
  viscosity: {viscosity:.6e}

tank_lid:
  kind: boundary
  rest_density: 1000.0
  viscosity: {viscosity:.6e}
  free_slip: {free_slip}

impeller:
  kind: rotor
  rest_density: 1000.0
  viscosity: {viscosity:.6e}
  rotor_angular_velocity: {omega:.6f}   # rad/s about +y
"""


def scalars_block(args, preset, h) -> str:
    """`scalars:` block: tracer pulses (--tracers) and the substrate setup (--substrate) at the
    preset's injection sphere, probes at the preset's probe points (Shepard radius = h)."""
    injection = preset["injection"]
    lines = ["", "# Scalars (Haringa 2023 section 2.3): tracer pulses and the glucose feed in the injection",
             "# sphere, Monod uptake by biomass on the fluid particles.", "scalars:", "  fields:"]
    if args.substrate:
        initial = 10.0 * args.k_s
        lines.append(f"    - {{name: substrate, diffusivity: {args.substrate_diffusivity:.3e}, turbulent: true, "
                     f"initial: {initial:.6e}}}")
        lines.append(f"    - {{name: biomass, diffusivity: 0.0, turbulent: false, initial: {args.biomass:.6e}}}")
        lines.append("    - {name: uptake, diffusivity: 0.0, turbulent: false, initial: 0.0}")
        lines.append("    - {name: feed, diffusivity: 0.0, turbulent: false, initial: 0.0}")
    for index in range(args.tracers):
        lines.append(f"    - {{name: tracer_{index + 1:02d}, diffusivity: {args.tracer_diffusivity:.3e}, "
                     f"turbulent: true, initial: 0.0}}")
    lines += ["  sgs:", f"    enabled: {'true' if args.sgs else 'false'}", "    smagorinsky_cs: 0.1",
              "    turbulent_schmidt: 0.7"]
    centre = injection["center"]
    if args.substrate:
        q_max = HARINGA_Q_MAX_UMOL_PER_G_H * 1e-6 / 3600.0
        lines += ["  reactions:",
                  f"    - {{type: monod, substrate: substrate, biomass: biomass, uptake: uptake, "
                  f"q_max: {q_max:.6e}, k_s: {args.k_s:.6e}, yield: 0.0}}",
                  "  sources:",
                  f"    - {{field: substrate, center: [{centre[0]}, {centre[1]}, {centre[2]}], "
                  f"radius: {injection['radius']:.4f}, rate: {args.feed_rate:.6e}, start: {args.feed_start:.4f}, "
                  f"record: feed}}"]
    lines.append("  injections:" if args.tracers > 0 else "  injections: []")
    for index in range(args.tracers):
        start = args.injection_start + index * args.injection_interval
        lines.append(f"    - {{field: tracer_{index + 1:02d}, center: [{centre[0]}, {centre[1]}, {centre[2]}], "
                     f"radius: {injection['radius']:.4f}, start: {start:.4f}, duration: {args.injection_duration:.4f}, "
                     f"value: 1.0}}")
    lines += ["  probes:", f"    radius: {h:.6f}", "    points:"]
    for name, (x, y, z) in preset["probes"]:
        lines.append(f"      - {{name: {name}, position: [{x}, {y}, {z}]}}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--preset", choices=sorted(PRESETS), required=True)
    parser.add_argument("--dx", type=float, required=True, help="particle spacing (m)")
    parser.add_argument("--hdx", type=float, default=3.0, help="h/dx (default 3)")
    parser.add_argument("--c0-factor", type=float, default=20.0, help="speed of sound = factor * tip speed (20)")
    parser.add_argument("--cfl", type=float, default=0.15)
    parser.add_argument("--background-pressure-factor", type=float, default=2.0,
                        help="p_b = factor * rho * U_tip^2 (default 2: 2000 Pa in the 30 L tank)")
    parser.add_argument("--viscosity", type=float, default=1.0e-6, help="kinematic viscosity (m2/s)")
    parser.add_argument("--free-slip-lid", action="store_true", help="no-shear lid (the paper's top)")
    parser.add_argument("--border", type=int, default=None, help="wall shell layers (default = hdx)")
    parser.add_argument("--max-incoming", type=int, default=32)
    parser.add_argument("--ramp-time", type=float, default=None, help="rotor spin-up time (s, preset default)")
    parser.add_argument("--flow-statistics", action="store_true",
                        help="write flow_statistics.json (jahoda: discharge lines, tip cylinder, planes)")
    parser.add_argument("--tracers", type=int, default=0, help="tracer pulses in the injection sphere")
    parser.add_argument("--injection-start", type=float, default=60.0, help="first pulse (s, the paper: 60)")
    parser.add_argument("--injection-interval", type=float, default=6.0, help="time between pulses (s)")
    parser.add_argument("--injection-duration", type=float, default=0.05,
                        help="pulse length (s; the paper's injection is instantaneous)")
    parser.add_argument("--tracer-diffusivity", type=float, default=6.0e-10, help="m2/s (the paper: 6e-10)")
    parser.add_argument("--sgs", action="store_true", help="Smagorinsky diffusion of the scalars")
    parser.add_argument("--substrate", action="store_true", help="glucose feed with Monod uptake (Haringa 2023)")
    parser.add_argument("--k-s", type=float, default=HARINGA_K_S, help="Monod half-saturation, mol/kg")
    parser.add_argument("--biomass", type=float, default=HARINGA_BIOMASS, help="g/kg")
    parser.add_argument("--substrate-diffusivity", type=float, default=6.0e-10)
    parser.add_argument("--feed-rate", type=float, default=HARINGA_FEED_RATE, help="mol/s (0.37)")
    parser.add_argument("--feed-start", type=float, default=0.0, help="s")
    parser.add_argument("--out", default=None, help="output directory (default cases/rushton_<preset>_<dx mm>mm)")
    parser.add_argument("--wall-convention", choices=("surface", "30l"), default="surface",
                        help="diagnostics: '30l' = fluid where sdf <= -dx/2 and four shell layers, as the 30 L generator")
    parser.add_argument("--lattice-offset", choices=("y", "xyz", "none"), default="y",
                        help="diagnostics: 'xyz' shifts the lattice in x and z too, the plates then lie between layers")
    parser.add_argument("--rpm", type=float, default=None, help="diagnostics: override the preset's speed (0 = at rest)")
    parser.add_argument("--no-snap-disks", action="store_true", help="diagnostics: keep the nominal disk heights")
    parser.add_argument("--no-plates", action="store_true", help="diagnostics: no blades, disks or baffles (shaft only)")
    parser.add_argument("--baffle-extension", choices=("all", "lid", "none"), default="none",
                        help="diagnostics: how far the baffle outline continues into the shells")
    parser.add_argument("--only-plates", default=None, help="diagnostics: keep only plates whose name contains this")
    parser.add_argument("--baffles", choices=("solid", "plates"), default="solid",
                        help="solid: baffles of --baffle-layers layers of wall particles (default); plates: thin plates "
                             "(the fluid at rest drifts in density, see the module docstring)")
    parser.add_argument("--baffle-layers", type=int, default=3, help="layers of a solid baffle (odd, centred on its plane)")
    parser.add_argument("--baffle-gap", type=float, default=0.0, help="diagnostics: gap between baffle and wall (m)")
    parser.add_argument("--baffle-bottom", type=float, default=0.0, help="diagnostics: height of the baffles' lower edge (m)")
    parser.add_argument("--baffle-count", type=int, default=None, help="diagnostics: number of baffles")
    parser.add_argument("--baffle-width", type=float, default=None, help="diagnostics: baffle width (m)")
    parser.add_argument("--baffle-offset-deg", type=float, default=0.0, help="diagnostics: rotate the baffles")
    args = parser.parse_args()

    preset = json.loads(json.dumps(PRESETS[args.preset]))      # a copy: the disk heights are snapped below
    if args.rpm is not None:
        preset["rpm"] = args.rpm
    preset["baffle_azimuth0_deg"] += args.baffle_offset_deg
    if args.baffle_count is not None:
        preset["baffle_count"] = args.baffle_count
    if args.baffle_width is not None:
        preset["baffle_width"] = args.baffle_width
    dx, h = args.dx, args.hdx * args.dx
    border = args.border if args.border is not None else int(math.ceil(args.hdx))
    shell = border * dx
    radius, top = 0.5 * preset["tank_diameter"], preset["liquid_height"]
    shaft_radius = 0.5 * preset["shaft_diameter"]
    if abs(top / dx - round(top / dx)) > 1e-6:
        print(f"note: H / dx = {top / dx:.3f} is not an integer, the lid is not half a spacing above a fluid layer")
    # c0 and p_b follow the preset's speed also when --rpm overrides it (a tank at rest keeps them)
    tip_speed = math.pi * max(impeller["diameter"] for impeller in preset["impellers"]) * PRESETS[args.preset]["rpm"] / 60.0
    omega = 2.0 * math.pi * preset["rpm"] / 60.0
    c0 = args.c0_factor * tip_speed
    background_pressure = args.background_pressure_factor * 1000.0 * tip_speed ** 2
    if (args.tracers > 0 or args.substrate) and preset["injection"] is None:
        parser.error(f"preset {args.preset} has no injection point")
    if args.tracers + (4 if args.substrate else 0) > 12:
        parser.error("at most 12 scalar fields: --substrate uses 4, so --tracers <= 8")

    # Lattice: integer multiples of dx shifted by dx / 2 in x, y and z; disks on mid-planes.
    lo = np.array([-(radius + shell), -shell, -(radius + shell)])
    hi = np.array([radius + shell, top + shell, radius + shell])
    shift = {"xyz": np.full(3, 0.5 * dx), "y": np.array([0.0, 0.5 * dx, 0.0]), "none": np.zeros(3)}[args.lattice_offset]
    sites = tile_bounding_box(lo - dx, hi + dx, dx, LATTICE_GRID, 3) + shift
    sites = sites[np.all((sites >= lo - 0.5 * dx) & (sites <= hi + 0.5 * dx), axis=1)]
    for impeller in preset["impellers"]:
        if args.no_snap_disks:
            snapped = impeller["center_height"]
        elif args.lattice_offset in ("xyz", "none"):
            snapped = round(impeller["center_height"] / dx) * dx                     # between (xyz) or on (none) layers
        else:
            snapped = (math.floor(impeller["center_height"] / dx) + 0.5) * dx          # on a layer
        if abs(snapped - impeller["center_height"]) > 1e-12:
            print(f"impeller {impeller['name']}: disk height {impeller['center_height']:.5f} m snapped to "
                  f"{snapped:.5f} m ({(snapped - impeller['center_height']) * 1e3:+.2f} mm)")
        impeller["center_height"] = snapped
    print(f"preset {args.preset}: T = {preset['tank_diameter']} m, H = {top} m, dx = {dx:.4g} m, h/dx = {args.hdx}, "
          f"{sites.shape[0]:,} lattice sites")

    interior = Cylinder([0.0, 0.0, 0.0], [0.0, top, 0.0], radius)
    lowest = min(impeller["center_height"] - 0.5 * rushton_dimensions(impeller)["blade_height"]
                 for impeller in preset["impellers"])
    shaft = y_cylinder(shaft_radius, lowest, top + shell)
    sdf_interior = interior.signed_distance(sites)
    is_rotor = shaft.signed_distance(sites) <= 0.0

    # Thin plates: lattice sites strictly inside a plate's slab (half a spacing on each side of the
    # mid-plane) and every other site closer than 0.6 dx to a plate particle are dropped; where plates
    # meet (disk and blades) the particles of the later plate closer than 0.6 dx to an earlier one are
    # dropped (as in the 30 L generator, which also drops sites on the slab surface).
    plates = [] if args.no_plates else tank_plates(preset, dx, shell, shaft_radius, args.baffle_extension, args.baffle_gap,
                                                    args.baffle_bottom)
    if args.baffles == "solid":
        plates = [plate for plate in plates if not plate["name"].startswith("baffle")]
    if args.only_plates:
        plates = [plate for plate in plates if args.only_plates in plate["name"]]
    kept = np.zeros((0, 3))
    for plate in plates:
        points = plate["points"]
        points = points[~closer_than(points, kept, 0.6 * dx)]
        plate["points"] = points
        kept = np.vstack([kept, points])
    sdf_plates = (Union(*[plate["box"] for plate in plates]).signed_distance(sites) if plates
                  else np.full(sites.shape[0], np.inf))
    in_plate = sdf_plates < -1e-9 * dx
    near = np.nonzero(~in_plate & (sdf_plates < 2.0 * dx))[0]
    in_plate[near[closer_than(sites[near], kept, 0.6 * dx)]] = True
    is_rotor &= ~in_plate
    is_baffle = np.zeros(sites.shape[0], dtype=bool)
    if args.baffles == "solid":
        if args.baffle_layers % 2 == 0:
            parser.error("--baffle-layers must be odd (the baffle plane is a lattice plane)")
        boxes = []
        for index in range(preset["baffle_count"]):
            e_r, e_t, e_y = (np.where(np.abs(axis) < 1e-12, 0.0, axis) for axis in
                             radial_frame(preset["baffle_azimuth0_deg"] + index * 360.0 / preset["baffle_count"]))
            r0 = radius - preset["baffle_width"]
            centre = e_r * 0.5 * (r0 + radius + shell) + e_y * 0.5 * top
            boxes.append(OrientedBox(centre, np.stack([e_r, e_y, e_t]),
                                     [0.5 * (radius + shell - r0), 0.5 * top + shell, 0.5 * args.baffle_layers * dx]))
        is_baffle = (Union(*boxes).signed_distance(sites) < -1e-9 * dx) & ~is_rotor & ~in_plate
        print(f"solid baffles: {args.baffle_layers} layers ({args.baffle_layers * dx * 1e3:.1f} mm), "
              f"{int((is_baffle & (sdf_interior < 0.0)).sum()):,} particles inside the tank")
    print(f"thin plates: {len(plates)} plates, {kept.shape[0]:,} particles, {int(in_plate.sum()):,} lattice sites dropped")

    # Fluid: lattice sites inside the tank; shell: `border` layers outside. Bottom and lid lie half a
    # spacing from the nearest layers (lattice shifted by dx / 2 in y), and on the cylinder the
    # mean boundary between fluid and wall sites is the tank radius itself (the 30 L generator puts
    # the walls half a spacing further in: fluid where sdf <= -dx / 2).
    if args.wall_convention == "30l":
        is_fluid = (sdf_interior <= -0.5 * dx) & ~is_rotor & ~in_plate & ~is_baffle
        is_shell = (~is_fluid & ~is_rotor & ~in_plate & (sdf_interior > -0.5 * dx)
                    & (sdf_interior <= shell + 0.5 * dx))
    else:
        is_fluid = (sdf_interior < 0.0) & ~is_rotor & ~in_plate & ~is_baffle
        is_shell = ~is_fluid & ~is_rotor & ~in_plate & (sdf_interior >= 0.0) & (sdf_interior < shell)
    is_shell |= is_baffle & (sdf_interior < 0.0)
    is_lid = is_shell & (sites[:, 1] > top)
    is_wall = is_shell & ~is_lid
    fluid, wall, lid, rotor = sites[is_fluid], sites[is_wall], sites[is_lid], sites[is_rotor]

    near_count = count_near_plates(fluid, plates, h) if plates else np.zeros(1, dtype=np.int32)
    print(f"thin plates: {int((near_count > 0).sum()):,} fluid particles ({100.0 * (near_count > 0).mean():.2f} %) "
          f"have a plate inside their support; at most {int(near_count.max())} plates at once "
          f"(the solver keeps {MAX_NEAR_THIN_PLATES})")
    if near_count.max() > MAX_NEAR_THIN_PLATES:
        parser.error("more plates inside one support than the solver keeps (MAX_NEAR_THIN_PLATES)")

    n_plate_rotor = sum(plate["points"].shape[0] for plate in plates if plate["frame"] == "rotor")
    n_plate_static = sum(plate["points"].shape[0] for plate in plates if plate["frame"] == "static")
    n_fluid, n_wall, n_lid = fluid.shape[0], wall.shape[0] + n_plate_static, lid.shape[0]
    n_rotor = rotor.shape[0] + n_plate_rotor
    total = n_fluid + n_wall + n_lid + n_rotor
    pool_size = int(math.ceil(total * 1.15 / 128) * 128)
    bound = int(math.ceil(math.sqrt(2) * args.hdx ** 3))
    max_per_voxel = max(64, int(2 ** math.ceil(math.log2(bound * 1.3))))
    volume = math.pi * radius ** 2 * top
    timestep = args.cfl * h / c0
    print(f"fluid={n_fluid:,} wall={n_wall:,} lid={n_lid:,} rotor={n_rotor:,} total={total:,} pool={pool_size:,}")
    print(f"fluid volume check: {n_fluid * dx ** 3:.6g} m3 of lattice cells vs {volume:.6g} m3 cylinder")
    print(f"U_tip = {tip_speed:.4f} m/s, c0 = {c0:.3f} m/s, p_b = {background_pressure:.1f} Pa, dt = {timestep:.4e} s, "
          f"{1.0 / timestep:,.0f} steps per second of flow")

    out = _REPO_ROOT / (args.out or f"cases/rushton_{args.preset}_{dx * 1e3:g}mm")
    out.mkdir(parents=True, exist_ok=True)
    write_obj(out / "fluid.obj", fluid)
    write_obj(out / "wall.obj", wall)
    write_obj(out / "lid.obj", lid)
    write_obj(out / "rotor.obj", rotor)
    for plate in plates:
        write_obj(out / f"plate_{plate['name']}.obj", plate["points"])
    all_points = np.vstack([fluid, wall, lid, rotor] + [plate["points"] for plate in plates])
    write_frame_obj(out / "frame.obj", all_points.min(axis=0) - 0.6 * dx, all_points.max(axis=0) + 0.6 * dx)

    ramp_time = args.ramp_time if args.ramp_time is not None else preset["ramp_time"]
    case_text = CASE_YAML.format(
        title=preset["title"], preset=args.preset, tank_diameter=preset["tank_diameter"], liquid_height=top,
        rpm=preset["rpm"], tip_speed=tip_speed, dx=dx, h=h, hdx=args.hdx, c0_factor=args.c0_factor,
        pressure_factor=args.background_pressure_factor, n_fluid=n_fluid, n_wall=n_wall, n_lid=n_lid,
        n_rotor=n_rotor, lid_kind="free slip" if args.free_slip_lid else "no slip", radius=0.5 * dx, c0=c0,
        cfl=args.cfl, background_pressure=background_pressure, pool_size=pool_size, max_per_voxel=max_per_voxel,
        max_incoming=args.max_incoming, ramp_time=ramp_time)
    entries = "".join(
        f"    - {{file: plate_{plate['name']}.obj, "
        f"material: {'impeller' if plate['frame'] == 'rotor' else 'tank_wall'}, thin_plate: {plate['name']}}}\n"
        for plate in plates)
    case_text = case_text.replace("    - {file: rotor.obj, material: impeller}\n",
                                  "    - {file: rotor.obj, material: impeller}\n" + entries)
    if plates:
        case_text += thin_plates_block(plates)
    if args.tracers > 0 or args.substrate:
        case_text += scalars_block(args, preset, h)
    (out / "case.yaml").write_text(case_text, encoding="utf-8")
    (out / "materials.yaml").write_text(MATERIALS_YAML.format(
        viscosity=args.viscosity, omega=omega, free_slip="true" if args.free_slip_lid else "false"), encoding="utf-8")

    tank = dict(preset=args.preset, title=preset["title"], tank_radius=radius, liquid_height=top,
                rpm=preset["rpm"], omega=omega, tip_speed=tip_speed, dx=dx, hdx=args.hdx, c0=c0,
                background_pressure=background_pressure, timestep=timestep, shaft_radius=shaft_radius,
                free_slip_lid=bool(args.free_slip_lid), baffle_count=preset["baffle_count"],
                baffle_width=preset["baffle_width"], baffle_azimuth0_deg=preset["baffle_azimuth0_deg"],
                impellers=[dict(impeller, **rushton_dimensions(impeller)) for impeller in preset["impellers"]],
                injection=preset["injection"], probes=[dict(name=name, position=list(position))
                                                       for name, position in preset["probes"]],
                release_sphere=preset["release_sphere"], fluid_particles=n_fluid, volume=volume)
    (out / "tank.json").write_text(json.dumps(tank, indent=1), encoding="utf-8")
    if args.flow_statistics:
        if len(preset["impellers"]) != 1:
            parser.error("--flow-statistics is defined for the 1-impeller tank")
        impeller = preset["impellers"][0]
        statistics = dict(kernel_radius=2.0 * dx, axis=[0.0, 1.0, 0.0], pivot=[0.0, 0.0, 0.0],
                          blades=impeller["blades"], blade_azimuth0_deg=impeller["azimuth0_deg"], phase_bins=30,
                          viscosity=args.viscosity, sets=flow_statistics_sets(preset, dx))
        (out / "flow_statistics.json").write_text(json.dumps(statistics), encoding="utf-8")
        print(f"flow statistics: {sum(len(s['points']) for s in statistics['sets']):,} points in "
              f"{len(statistics['sets'])} sets")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
