"""
_analyze_rushton_lower_flow.py — the flow around the Rushton turbine, in particular below its disk and around
the lower blade halves: Fluent snapshots vs SPH dumps (2026-10-03, base Anaconda: numpy, matplotlib).

Frame. Generator frame (Fluent y - 58.5 mm). Every data set is oriented so that the impeller turns along
+e_theta = (z, 0, -x) / r, right hand about +y as in our cases: a Fluent snapshot whose Rushton zone swirls the
other way (the mirrored fine snapshots 26..34 s) is mirrored, z -> -z and w -> -w, its rt faces too.
phi = atan2(-z, x) grows in the direction of rotation; psi = phi - phi_blade, wrapped to [-30, 30) deg, is the
angle from the nearest Rushton blade (psi > 0: ahead of the pushing face, psi < 0: in its wake). phi_blade is
the six-fold circular mean of the blade faces (Fluent: rt faces with |n . e_theta| > 0.8) or of the blade plate
particles (SPH: solid particles at r 24..48.5 mm, y 28.5..48.3 mm, away from the disk).

Rotating zones. In the interpolation files (and the wall exports) of these sliding-mesh snapshots the cells of
rt_rotorbox and pbt_rotorbox sit one time step ahead of their velocities: the mesh has already been turned by
Omega dt = 12 deg (dt 0.01 s) in the direction of rotation. Seen at the written positions (fine 25, 26, 31 s),
the fluid next to the hub (r 10.2..10.5 mm, y 22..30 mm) has atan(u_r / u_theta) = +11.5..+20.4 deg, the radial
volume flow jumps by -0.20..-0.25 N D^3 across the sliding interface (r 71.8 mm) and the control volume below
loses -0.19..-0.24 N D^3 (in 0.55, out 0.80). Turning the velocity vectors of the rotor-zone cells by 12 deg in
the direction of rotation (--fluent-rotor-lag) gives -0.5..+8.4 deg, -0.03..-0.06 and -0.02..+0.01 N D^3
(15 deg: -3.5..+5.4 deg, -0.01..+0.02, +0.04..+0.06). Rotor zones by position: r < 71.8 mm and
18.7 < y < 58.5 mm (rt) or 165 < y < 225 mm (pbt), generator frame.

Weights. SPH particles dx^3. Fluent cells: their volumes, from a third interpolation file per snapshot. Fluent's
interpolate/write-data refuses custom fields and user memory, and the patch menu has no x-velocity, so the cell
volume in mm^3 is patched into the pressure and written:
    /define/custom-field-functions/define "cellvolmm" "cell_volume * 1e9"
    /solve/patch rt_rotorbox pbt_rotorbox bulk_liquid () () pressure yes cellvolmm
    /file/interpolate/write-data "fineNN_volume.ip" (rt_rotorbox pbt_rotorbox bulk_liquid) pressure ()
The written pressure is shifted by the value at the reference location (one constant per file); the constant
follows from the total volume, /report/volume-integrals/volume: 0.029953756 m^3 for the mesh of fine 25 s
(3408405 cells), 0.029952864 m^3 for the mirrored mesh of 26..34 s (3408218 cells), FLUENT_TOTAL_VOLUME. The
smallest cell then comes out at 1.3e-4 mm^3 and the median at 1.000 mm^3. The cell order of an interpolation
file changes from one .cas to the next, so the volume file must come from the same snapshot; x and y are
checked cell by cell.

Output, averaged over the samples of a data set:
  1. control volume around the Rushton, r < 52 mm, 25 < y < 52 mm (blades r 24..48 mm, y 28.8..48.0 mm; disk
     r < 32 mm, y 37.2..39.7 mm; top and bottom faces split at the disk rim): volume flow and angular momentum
     flow rho u_n r u_theta through its faces from the samples in slabs of +-3 mm about each face,
     sum(V u_n ...) / 6 mm. The products are taken per sample, so the angular momentum flow includes the
     turbulent and blade-periodic parts; its net outflow balances the torque of the blades up to the storage,
     the viscous stresses on the faces and the hub and shaft;
  2. the region below, y < 25 mm: up and down flow and angular momentum flow through the bin row y 24..27 mm,
     mean swirl. Outside Fluent's rotor zones the cells are several mm and layered in y, so a thin slab catches
     whole layers or none (slab volume / geometric volume 0.03..4 for +-1 mm, 0.2..1.4 for +-3 mm, net flow through
     a whole-tank plane up to 0.3 N D^3). Plane flows outside the control volume are therefore taken as the
     volume-weighted bin mean times the fluid area of the bin, pi (r2^2 - r1^2) f, with the fluid fraction f of
     each 3 mm bin from an SPH reference dump (--reference, default the first SPH dump); so is the stream
     function of the figure. Inside the rotor zones Fluent's cells are 1 mm and the slabs of item 1 hold
     0.97..1.00 of their geometric volume (the hub excepted);
  3. swirl inside the blade passages relative to the blade speed, lower and upper half; the extrema of the
     stream function below y 25 mm; the feed from the PBT: downward volume flow through the disks r < 32, 52,
     72 mm above the PBT (y 230, 260 mm) and between the impellers, mean swirl in the rings r < 32, 32..52,
     52..72 mm there, radial outflow through r = 72 mm between them (bin means as in item 2);
  4. figures OUT_meridional.png (Stokes stream function from the bin means as in item 2, u_theta, lower tank;
     psi > 0, i.e. upward flow inside r and downward outside, is a clockwise cell with r to the right, y up) and
     OUT_blade_<quantity>.png around a blade in the rotating frame (slabs below the blades, lower half, upper
     half): u_r with the relative velocity (u_r, u_theta - Omega r), u_y, pressure minus its slab mean.

usage:
    python experiment/v1/checks/_analyze_rushton_lower_flow.py OUT_PREFIX \
        --fluent FINE25.ip FINE25_RT.csv FINE25_VOLUME.ip [--fluent ...] \
        --sph LABEL DUMP.npz [DUMP.npz ...] [--sph ...] [--dx 0.003] [--reference DUMP.npz]
"""
import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _analyze_axisymmetric_energy as axi   # noqa: E402  (read_interpolation_file)

FLUENT_Y_SHIFT = 0.0585
FLUENT_TOTAL_VOLUME = {3408405: 0.029953756, 3408218: 0.029952864}   # m^3 by cell count (see the docstring)
DENSITY = 998.0
OMEGA = 200.0 / 60.0 * 2.0 * np.pi
PUMPING_SCALE = (200.0 / 60.0) * 0.096 ** 3          # N D^3, m^3/s
DISK_Y0, DISK_Y1, DISK_R, HUB_R = 0.0372, 0.0397, 0.032, 0.0102
BLADE_Y0, BLADE_Y1, BLADE_R0, BLADE_R1 = 0.0288, 0.0480, 0.024, 0.048
CV_R, CV_Y0, CV_Y1, HALF_SLAB = 0.052, 0.025, 0.052, 0.003
BLADE_SLABS = (("below the blades, y 20.4..28.8 mm", 0.0204, 0.0288),
               ("lower blade half, y 28.8..37.2 mm", 0.0288, 0.0372),
               ("upper blade half, y 39.7..48.0 mm", 0.0397, 0.0480))
MERIDIONAL_BIN = 0.003
R_EDGES = np.arange(0.0, 0.144 + 1e-9, MERIDIONAL_BIN)
Y_EDGES = np.arange(-0.060, 0.426 + 1e-9, MERIDIONAL_BIN)
BLADE_R_EDGES = np.arange(0.0, 0.072 + 1e-9, 0.003)
BLADE_PSI_EDGES = np.radians(np.arange(-30.0, 30.0 + 1e-9, 3.0))


class Sample:
    """one snapshot in the oriented frame: cylindrical velocity components, volumes, pressure, blade angle"""

    def __init__(self, position, velocity, volume, pressure, blade_angle):
        self.r = np.maximum(np.hypot(position[:, 0], position[:, 2]), 1e-12)
        self.y = position[:, 1]
        self.u_r = (velocity[:, 0] * position[:, 0] + velocity[:, 2] * position[:, 2]) / self.r
        self.u_t = (velocity[:, 0] * position[:, 2] - velocity[:, 2] * position[:, 0]) / self.r
        self.u_y = velocity[:, 1]
        self.volume = volume
        self.pressure = pressure
        self.blade_angle = blade_angle
        phi = np.arctan2(-position[:, 2], position[:, 0])
        self.psi = np.mod(phi - blade_angle + np.pi / 6.0, np.pi / 3.0) - np.pi / 6.0


def read_csv_columns(path):
    with open(path) as handle:
        names = [name.strip() for name in handle.readline().split(",")]
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    return {name: data[:, k] for k, name in enumerate(names)}


def fluent_blade_angle(rt_path, mirror):
    faces = read_csv_columns(rt_path)
    x, y, z = faces["x-coordinate"], faces["y-coordinate"] - FLUENT_Y_SHIFT, faces["z-coordinate"]
    area = np.stack([faces["x-face-area"], faces["y-face-area"], faces["z-face-area"]], axis=1)
    if mirror:
        z = -z
        area[:, 2] *= -1.0
    r = np.maximum(np.hypot(x, z), 1e-12)
    size = np.linalg.norm(area, axis=1)
    n_theta = (area[:, 0] * z - area[:, 2] * x) / (r * np.maximum(size, 1e-30))
    blade = (np.abs(n_theta) > 0.8) & (r > 0.0235) & (r < 0.0485) & (y > 0.0285) & (y < 0.0485)
    phi = np.arctan2(-z[blade], x[blade])
    return np.angle((size[blade] * np.exp(6j * phi)).sum()) / 6.0


def in_fluent_rotor_zones(position):
    r = np.hypot(position[:, 0], position[:, 2])
    y = position[:, 1]
    return (r < 0.0718) & (((y > 0.0187) & (y < 0.0585)) | ((y > 0.165) & (y < 0.225)))


def turn_about_y(vectors, angle):
    """rotate by angle in the sense of +e_theta (right hand about +y): (1, 0, 0) -> (cos, 0, -sin)"""
    c, s = np.cos(angle), np.sin(angle)
    out = vectors.copy()
    out[:, 0] = c * vectors[:, 0] + s * vectors[:, 2]
    out[:, 2] = -s * vectors[:, 0] + c * vectors[:, 2]
    return out


def fluent_samples(triples, rotor_lag_degrees=12.0):
    for ip_path, rt_path, volume_path in triples:
        cells = axi.read_interpolation_file(ip_path)
        volume_cells = axi.read_interpolation_file(volume_path)
        if len(volume_cells["x"]) != len(cells["x"]) or not (
                np.array_equal(volume_cells["x"], cells["x"]) and np.array_equal(volume_cells["y"], cells["y"])):
            raise ValueError(f"{volume_path} does not hold the cells of {ip_path} in the same order")
        written = volume_cells["pressure"] * 1e-9
        volume = written + (FLUENT_TOTAL_VOLUME[len(written)] - written.sum()) / len(written)
        if volume.min() < -1e-14:
            raise ValueError(f"{volume_path}: negative cell volume {volume.min():.3e} m^3 after the shift")
        volume = np.maximum(volume, 0.0)
        position = np.stack([cells["x"], cells["y"] - FLUENT_Y_SHIFT, cells["z"]], axis=1)
        velocity = np.stack([cells["x-velocity"], cells["y-velocity"], cells["z-velocity"]], axis=1)
        r = np.maximum(np.hypot(position[:, 0], position[:, 2]), 1e-12)
        rotor = (r < 0.072) & (position[:, 1] > 0.0187) & (position[:, 1] < 0.0585)
        swirl = ((velocity[rotor, 0] * position[rotor, 2] - velocity[rotor, 2] * position[rotor, 0]) / r[rotor]).sum()
        mirror = swirl < 0.0
        if mirror:
            position[:, 2] *= -1.0
            velocity[:, 2] *= -1.0
        rotor_zone = in_fluent_rotor_zones(position)
        velocity[rotor_zone] = turn_about_y(velocity[rotor_zone], np.radians(rotor_lag_degrees))
        sample = Sample(position, velocity, volume, cells.get("pressure"), fluent_blade_angle(rt_path, mirror))
        print(f"  {pathlib.Path(ip_path).name}: {len(r)} cells, smallest {volume.min() * 1e9:.4f} mm^3, mirrored {mirror}, "
              f"blade at {np.degrees(sample.blade_angle):+.2f} deg", flush=True)
        yield sample


def sph_samples(paths, dx):
    for path in paths:
        dump = np.load(path)
        position, material = dump["positions"], dump["material"]
        alive = position[:, 3] > 0
        fluid = alive & (material == 0)
        solid = position[alive & (material != 0), :3].astype(np.float64)
        r, y = np.hypot(solid[:, 0], solid[:, 2]), solid[:, 1]
        blade = (r > 0.0235) & (r < 0.0485) & (y > 0.0285) & (y < 0.0483) & \
            ((np.abs(y - 0.5 * (DISK_Y0 + DISK_Y1)) > 0.0016) | (r > DISK_R + 0.0005))
        phi = np.arctan2(-solid[blade, 2], solid[blade, 0])
        sample = Sample(position[fluid, :3].astype(np.float64), dump["velocity_mass"][fluid, :3].astype(np.float64),
                        np.full(int(fluid.sum()), dx ** 3), dump["density_pressure"][fluid, 1].astype(np.float64),
                        np.angle(np.exp(6j * phi).sum()) / 6.0)
        print(f"  {pathlib.Path(path).name}: {int(fluid.sum())} fluid particles, {int(blade.sum())} blade particles, "
              f"blade at {np.degrees(sample.blade_angle):+.2f} deg", flush=True)
        yield sample


def fluid_fraction(path, dx):
    """fraction of each meridional bin occupied by fluid particles in an SPH dump (generator geometry)"""
    sample = next(sph_samples([path], dx))
    nr, ny = len(R_EDGES) - 1, len(Y_EDGES) - 1
    index_r = np.clip(np.digitize(sample.r, R_EDGES) - 1, 0, nr - 1)
    index_y = np.digitize(sample.y, Y_EDGES) - 1
    inside = (index_y >= 0) & (index_y < ny) & (sample.r < R_EDGES[-1])
    volume = binned(index_r, np.clip(index_y, 0, ny - 1), nr, ny, sample.volume, inside)
    bin_volume = np.pi * (R_EDGES[1:] ** 2 - R_EDGES[:-1] ** 2)[:, None] * MERIDIONAL_BIN
    return np.minimum(volume / bin_volume, 1.0)


FLOW_NAMES = ("in through the top (y 52 mm), r < 32 mm", "in through the top, r 32..52 mm",
              "in through the bottom (y 25 mm), r < 32 mm", "in through the bottom, r 32..52 mm",
              "out through r 52 mm, below the disk", "out through r 52 mm, disk level",
              "out through r 52 mm, above the disk")


def surface_flows(s):
    """(volume flow, angular momentum flow) through the faces of the control volume, oriented as named"""
    out = {}

    def plane(name, y0, r0, r1, sign):
        m = (np.abs(s.y - y0) < HALF_SLAB) & (s.r >= r0) & (s.r < r1)
        flow = sign * s.volume[m] * s.u_y[m] / (2.0 * HALF_SLAB)
        out[name] = (flow.sum(), DENSITY * (flow * s.r[m] * s.u_t[m]).sum())

    def cylinder(name, r0, y0, y1):
        m = (np.abs(s.r - r0) < HALF_SLAB) & (s.y >= y0) & (s.y < y1)
        flow = s.volume[m] * s.u_r[m] / (2.0 * HALF_SLAB)
        out[name] = (flow.sum(), DENSITY * (flow * s.r[m] * s.u_t[m]).sum())

    plane(FLOW_NAMES[0], CV_Y1, 0.0, DISK_R, -1.0)
    plane(FLOW_NAMES[1], CV_Y1, DISK_R, CV_R, -1.0)
    plane(FLOW_NAMES[2], CV_Y0, 0.0, DISK_R, 1.0)
    plane(FLOW_NAMES[3], CV_Y0, DISK_R, CV_R, 1.0)
    cylinder(FLOW_NAMES[4], CV_R, CV_Y0, DISK_Y0)
    cylinder(FLOW_NAMES[5], CV_R, DISK_Y0, DISK_Y1)
    cylinder(FLOW_NAMES[6], CV_R, DISK_Y1, CV_Y1)
    return out


def scalars(s):
    out = {}
    below = s.y < CV_Y0
    under = below & (s.y > 0.0) & (s.r < CV_R)
    out["swirl below y 25 mm, all r, m/s"] = (s.volume[below] * s.u_t[below]).sum() / s.volume[below].sum()
    out["swirl below the Rushton, r < 52 mm, 0 < y < 25 mm, m/s"] = \
        (s.volume[under] * s.u_t[under]).sum() / s.volume[under].sum()
    out["angular momentum below y 25 mm, g m^2/s"] = 1e3 * DENSITY * (s.volume[below] * s.r[below] * s.u_t[below]).sum()
    disk = (np.abs(s.y - 0.5 * (DISK_Y0 + DISK_Y1)) < 0.001) & (s.r > HUB_R + 0.001) & (s.r < DISK_R - 0.001)
    out["fluid volume within 1 mm of the disk mid-plane / slab volume"] = \
        s.volume[disk].sum() / (np.pi * ((DISK_R - 0.001) ** 2 - (HUB_R + 0.001) ** 2) * 0.002)
    for name, y0, y1 in (("lower", BLADE_Y0, DISK_Y0), ("upper", DISK_Y1, BLADE_Y1)):
        m = (s.r > BLADE_R0) & (s.r < BLADE_R1) & (s.y > y0) & (s.y < y1)
        out[f"swirl / blade speed in the {name} passages"] = \
            (s.volume[m] * s.u_t[m]).sum() / (s.volume[m] * OMEGA * s.r[m]).sum()
        out[f"radial velocity in the {name} passages, m/s"] = (s.volume[m] * s.u_r[m]).sum() / s.volume[m].sum()
        out[f"axial velocity in the {name} passages, m/s"] = (s.volume[m] * s.u_y[m]).sum() / s.volume[m].sum()
    return out


def binned(index_r, index_c, size_r, size_c, weights, mask):
    flat = index_r[mask] * size_c + index_c[mask]
    return np.bincount(flat, weights=weights[mask], minlength=size_r * size_c).reshape(size_r, size_c)


class Statistics:
    def __init__(self, label):
        self.label = label
        self.count = 0
        self.flows, self.scalars = [], []
        self.meridional = np.zeros((5, len(R_EDGES) - 1, len(Y_EDGES) - 1))   # V, V u_r, V u_t, V u_y, V u_y r u_t
        self.blade = np.zeros((len(BLADE_SLABS), 6, len(BLADE_R_EDGES) - 1, len(BLADE_PSI_EDGES) - 1))

    def add(self, s):
        self.count += 1
        self.flows.append(surface_flows(s))
        self.scalars.append(scalars(s))
        nr, ny = len(R_EDGES) - 1, len(Y_EDGES) - 1
        index_r = np.clip(np.digitize(s.r, R_EDGES) - 1, 0, nr - 1)
        index_y = np.digitize(s.y, Y_EDGES) - 1
        inside = (index_y >= 0) & (index_y < ny) & (s.r < R_EDGES[-1])
        index_y = np.clip(index_y, 0, ny - 1)
        for k, weight in enumerate((s.volume, s.volume * s.u_r, s.volume * s.u_t, s.volume * s.u_y,
                                    s.volume * s.u_y * s.r * s.u_t)):
            self.meridional[k] += binned(index_r, index_y, nr, ny, weight, inside)
        br, bp = len(BLADE_R_EDGES) - 1, len(BLADE_PSI_EDGES) - 1
        index_br = np.digitize(s.r, BLADE_R_EDGES) - 1
        index_bp = np.clip(np.digitize(s.psi, BLADE_PSI_EDGES) - 1, 0, bp - 1)
        for k, (_, y0, y1) in enumerate(BLADE_SLABS):
            m = (s.y >= y0) & (s.y < y1) & (index_br >= 0) & (index_br < br)
            weights = [s.volume, s.volume * s.u_r, s.volume * s.u_t, s.volume * s.u_y]
            if s.pressure is not None:
                p_mean = (s.volume[m] * s.pressure[m]).sum() / s.volume[m].sum()
                weights += [s.volume * (s.pressure - p_mean), s.volume]
            else:
                weights += [np.zeros_like(s.volume), np.zeros_like(s.volume)]
            for q, weight in enumerate(weights):
                self.blade[k, q] += binned(np.clip(index_br, 0, br - 1), index_bp, br, bp, weight, m)

    def mean_flows(self):
        return {name: np.mean([f[name] for f in self.flows], axis=0) for name in self.flows[0]}

    def mean_scalars(self):
        return {name: np.mean([f[name] for f in self.scalars]) for name in self.scalars[0]}

    def bin_means(self, fraction):
        """volume-weighted bin means of u_r, u_t, u_y, u_y r u_t; empty bins that hold fluid in the reference are
        filled from the nearest bin with data, solid bins (fraction < 0.05) are NaN"""
        have = self.meridional[0] > 0
        with np.errstate(invalid="ignore", divide="ignore"):
            means = self.meridional[1:] / self.meridional[0]
        missing = (fraction >= 0.05) & ~have
        if missing.any():
            known = np.argwhere(have)
            for i, j in np.argwhere(missing):
                nearest = known[np.argmin((known[:, 0] - i) ** 2 + (known[:, 1] - j) ** 2)]
                means[:, i, j] = means[:, nearest[0], nearest[1]]
        means[:, fraction < 0.05] = np.nan
        return means

    def plane_flows(self, fraction, row):
        """per r bin: (volume flow up, angular momentum flow up) through the bin row, from the bin means"""
        means = np.nan_to_num(self.bin_means(fraction))
        area = np.pi * (R_EDGES[1:] ** 2 - R_EDGES[:-1] ** 2) * fraction[:, row]
        return area * means[2, :, row], DENSITY * area * means[3, :, row]


def report(statistics, fraction):
    print("\n1. control volume around the Rushton, r < 52 mm, 25 < y < 52 mm")
    print("   volume flow in N D^3 (N D^3 = 2.95 L/s), angular momentum flow in mN m")
    header = "".join(f"{s.label[:30]:>32s}" for s in statistics)
    print(f"   {'':44s}{header}")
    flows = [s.mean_flows() for s in statistics]
    for name in flows[0]:
        print(f"   {name:44s}" + "".join(f"     {f[name][0] / PUMPING_SCALE:+7.3f} {f[name][1] * 1e3:+8.2f}    "
                                         for f in flows))
    print(f"   {'balance (in - out), volume flow':44s}" + "".join(
        f"     {(sum(f[n][0] for n in FLOW_NAMES[:4]) - sum(f[n][0] for n in FLOW_NAMES[4:])) / PUMPING_SCALE:+7.3f}"
        f"{'':13s}" for f in flows))
    print(f"   {'torque from the angular momentum budget':44s}" + "".join(
        f"{'':17s}{(sum(f[n][1] for n in FLOW_NAMES[4:]) - sum(f[n][1] for n in FLOW_NAMES[:4])) * 1e3:+8.2f}    "
        for f in flows))
    print("\n2./3. swirl and passages")
    values = [s.mean_scalars() for s in statistics]
    for name in values[0]:
        print(f"   {name:58s}" + "".join(f"{v[name]:+10.3f}" for v in values))
    r_mid = 0.5 * (R_EDGES[1:] + R_EDGES[:-1])
    y_mid = 0.5 * (Y_EDGES[1:] + Y_EDGES[:-1])
    row = int(np.digitize(CV_Y0, Y_EDGES) - 1)
    print(f"\n   region below: flows through the bin row y {Y_EDGES[row] * 1e3:.0f}..{Y_EDGES[row + 1] * 1e3:.0f} mm "
          f"(bin means x fluid area), N D^3 and mN m (positive into the region below)")
    for s in statistics:
        q, angular = s.plane_flows(fraction, row)
        q, angular = -q / PUMPING_SCALE, -angular * 1e3
        inner = r_mid < CV_R
        psi = -np.cumsum(q)
        print(f"   {s.label:40s} down {q[q > 0].sum():6.3f}  up {q[q < 0].sum():7.3f}  net {q.sum():+6.3f}  "
              f"angular momentum in: r < 52 mm {angular[inner].sum():+6.2f}, r > 52 mm {angular[~inner].sum():+6.2f}\n"
              f"   {'':40s} stream function at this row: max {psi.max():+6.3f} at r {r_mid[np.argmax(psi)] * 1e3:4.0f} mm, "
              f"min {psi.min():+6.3f} at r {r_mid[np.argmin(psi)] * 1e3:4.0f} mm")
    print("\n   cells of the stream function below y 25 mm (bin means x fluid area), N D^3: clockwise max / anticlockwise min")
    for s in statistics:
        psi = stream_function(s.bin_means(fraction), fraction)
        region = (y_mid < CV_Y0)[None, :] & (fraction >= 0.3)
        values = np.where(region, psi, np.nan)
        i, j = np.unravel_index(np.nanargmax(values), values.shape)
        k, m = np.unravel_index(np.nanargmin(values), values.shape)
        print(f"   {s.label:40s} {values[i, j]:+6.3f} at r {r_mid[i] * 1e3:4.0f}, y {y_mid[j] * 1e3:4.0f} mm   "
              f"{values[k, m]:+6.3f} at r {r_mid[k] * 1e3:4.0f}, y {y_mid[m] * 1e3:4.0f} mm")
    print("\n   feed from the PBT: downward volume flow through the disk r < R at height y (bin means x fluid area), N D^3")
    for height in (0.260, 0.230, 0.161, 0.130, 0.100, 0.070, 0.052):
        row_h = int(np.digitize(height, Y_EDGES) - 1)
        cells = []
        for s in statistics:
            q = -s.plane_flows(fraction, row_h)[0] / PUMPING_SCALE
            cells.append(" ".join(f"{q[r_mid < radius].sum():+.3f}" for radius in (0.032, 0.052, 0.072)))
        print(f"   y {Y_EDGES[row_h] * 1e3:5.0f}..{Y_EDGES[row_h + 1] * 1e3:3.0f} mm, r < 32 / 52 / 72 mm: " +
              "".join(f"{c:>26s}" for c in cells))
    print("   mean swirl u_theta in the rings r < 32 / 32..52 / 52..72 mm (volume-weighted bin means), m/s")
    for height in (0.260, 0.230, 0.161, 0.130, 0.100, 0.070, 0.052):
        row_h = int(np.digitize(height, Y_EDGES) - 1)
        cells = []
        for s in statistics:
            means = np.nan_to_num(s.bin_means(fraction))
            weight = np.pi * (R_EDGES[1:] ** 2 - R_EDGES[:-1] ** 2) * fraction[:, row_h]
            rings = [(r_mid < 0.032), (r_mid >= 0.032) & (r_mid < 0.052), (r_mid >= 0.052) & (r_mid < 0.072)]
            cells.append(" ".join(f"{(weight * means[1, :, row_h])[m].sum() / weight[m].sum():+.3f}" for m in rings))
        print(f"   y {Y_EDGES[row_h] * 1e3:5.0f}..{Y_EDGES[row_h + 1] * 1e3:3.0f} mm: " +
              "".join(f"{c:>26s}" for c in cells))
    column_72 = int(np.digitize(0.072, R_EDGES) - 1)
    print(f"   radial outflow through r {R_EDGES[column_72] * 1e3:.0f}..{R_EDGES[column_72 + 1] * 1e3:.0f} mm "
          f"(bin means x fluid area), N D^3:")
    for y0, y1 in ((0.055, 0.100), (0.100, 0.130), (0.130, 0.160)):
        rows = (y_mid > y0) & (y_mid < y1)
        cells = []
        for s in statistics:
            means = np.nan_to_num(s.bin_means(fraction))
            area = 2.0 * np.pi * r_mid[column_72] * MERIDIONAL_BIN * fraction[column_72, rows]
            cells.append(f"{(area * means[0, column_72, rows]).sum() / PUMPING_SCALE:+.3f}")
        print(f"   y {y0 * 1e3:3.0f}..{y1 * 1e3:3.0f} mm: " + "".join(f"{c:>26s}" for c in cells))
    print("\n   azimuthal-mean jet at r 52 mm (bins 3 mm): u_r / u_theta / u_y, m/s")
    column = np.argmin(np.abs(r_mid - CV_R))
    for row in np.nonzero((y_mid > 0.012) & (y_mid < 0.066))[0]:
        cells = []
        for s in statistics:
            v = s.meridional[:, column, row]
            cells.append(f"   {v[1] / v[0]:+.2f} {v[2] / v[0]:+.2f} {v[3] / v[0]:+.2f}" if v[0] > 0 else " " * 18)
        print(f"   y {y_mid[row] * 1e3:5.1f}" + "".join(f"{c:>32s}" for c in cells))


def stream_function(means, fraction):
    """volume flow upward through the disk of radius r at the bin's height, at the bin centre, N D^3"""
    flow = np.nan_to_num(means[2]) * np.pi * (R_EDGES[1:] ** 2 - R_EDGES[:-1] ** 2)[:, None] * fraction
    return (np.cumsum(flow, axis=0) - 0.5 * flow) / PUMPING_SCALE


def plot_meridional(statistics, fraction, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(statistics), figsize=(2.9 * len(statistics) + 1.2, 4.9), squeeze=False,
                             constrained_layout=True)
    r_mid = 0.5 * (R_EDGES[1:] + R_EDGES[:-1])
    y_mid = 0.5 * (Y_EDGES[1:] + Y_EDGES[:-1])
    levels = np.arange(-1.5, 1.51, 0.05)
    levels = levels[np.abs(levels) > 1e-9]
    for column, s in enumerate(statistics):
        ax = axes[0, column]
        mean = s.bin_means(fraction)
        mask = fraction < 0.3
        u_t = np.ma.array(mean[1], mask=mask)
        psi_mid = stream_function(mean, fraction)
        image = ax.pcolormesh(R_EDGES * 1e3, Y_EDGES * 1e3, u_t.T, vmin=0.0, vmax=0.5, cmap="viridis")
        ax.contour(r_mid * 1e3, y_mid * 1e3, np.ma.array(psi_mid, mask=mask).T, levels=levels, colors="k",
                   linewidths=0.7, linestyles=np.where(levels > 0, "solid", "dashed"))
        step = 2
        grid_r, grid_y = np.meshgrid(r_mid[::step] * 1e3, y_mid[::step] * 1e3, indexing="ij")
        ax.quiver(grid_r, grid_y, np.ma.array(mean[0], mask=mask)[::step, ::step],
                  np.ma.array(mean[2], mask=mask)[::step, ::step], color="w", scale=6.0, width=0.003, alpha=0.9)
        for r0, r1, y0, y1 in ((BLADE_R0, BLADE_R1, BLADE_Y0, BLADE_Y1),):
            ax.plot(np.array([r0, r1, r1, r0, r0]) * 1e3, np.array([y0, y0, y1, y1, y0]) * 1e3, color="r", linewidth=0.8)
        ax.plot([HUB_R * 1e3, DISK_R * 1e3], [38.45, 38.45], color="r", linewidth=2.0)
        ax.plot(np.array([0, CV_R, CV_R, 0]) * 1e3, np.array([CV_Y0, CV_Y0, CV_Y1, CV_Y1]) * 1e3, color="c",
                linestyle=":", linewidth=0.8)
        ax.set_aspect("equal")
        ax.set_xlim(0, 144)
        ax.set_ylim(-60, 120)
        ax.set_xlabel("r, mm")
        ax.set_ylabel("y, mm")
        q = s.plane_flows(fraction, int(np.digitize(CV_Y0, Y_EDGES) - 1))[0] / PUMPING_SCALE
        ax.set_title(f"{s.label}\nthrough y 24..27 mm: up {q[q > 0].sum():.2f}, down {q[q < 0].sum():.2f} N D³", fontsize=7)
        ax.tick_params(labelsize=7)
        if column:
            ax.set_ylabel("")
    fig.colorbar(image, ax=axes[0, :].tolist(), shrink=0.7, label="u_θ azimuthal mean, m/s")
    fig.suptitle("lower tank, azimuthal mean. Lines: stream function, 0.05 N D³ apart (solid: clockwise in this "
                 "view, dashed: anticlockwise).\nArrows: (u_r, u_y). Red: Rushton blades and disk. Dotted: control "
                 "volume r < 52 mm, 25 < y < 52 mm.", fontsize=7)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def plot_blade_frame(statistics, quantity, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    settings = {"u_r": (1, "u_r, m/s", -0.8, 0.8, "RdBu_r"), "u_y": (3, "u_y, m/s", -0.5, 0.5, "RdBu_r"),
                "pressure": (4, "p − slab mean, Pa", -300.0, 300.0, "RdBu_r")}
    index, label, vmin, vmax, cmap = settings[quantity]
    rows = [s for s in statistics if quantity != "pressure" or s.blade[:, 5].sum() > 0]
    if not rows:
        return
    fig, axes = plt.subplots(len(rows), len(BLADE_SLABS), figsize=(4.4 * len(BLADE_SLABS), 3.0 * len(rows)),
                             squeeze=False)
    psi_mid = 0.5 * (BLADE_PSI_EDGES[1:] + BLADE_PSI_EDGES[:-1])
    r_mid = 0.5 * (BLADE_R_EDGES[1:] + BLADE_R_EDGES[:-1])
    edge_r, edge_psi = np.meshgrid(BLADE_R_EDGES, BLADE_PSI_EDGES, indexing="ij")
    mid_r, mid_psi = np.meshgrid(r_mid, psi_mid, indexing="ij")
    for row, s in enumerate(rows):
        for column, (title, _, _) in enumerate(BLADE_SLABS):
            ax = axes[row, column]
            sums = s.blade[column]
            weight = sums[5] if quantity == "pressure" else sums[0]
            with np.errstate(invalid="ignore", divide="ignore"):
                value = np.ma.masked_invalid(sums[index] / weight)
                u_r = sums[1] / sums[0]
                w_rel = sums[2] / sums[0] - OMEGA * mid_r
            image = ax.pcolormesh(edge_r * np.cos(edge_psi) * 1e3, edge_r * np.sin(edge_psi) * 1e3, value,
                                  vmin=vmin, vmax=vmax, cmap=cmap, shading="flat")
            if quantity == "u_r":
                step = (slice(None, None, 2), slice(None, None, 2))
                u_x = u_r * np.cos(mid_psi) - w_rel * np.sin(mid_psi)
                u_z = u_r * np.sin(mid_psi) + w_rel * np.cos(mid_psi)
                ax.quiver((mid_r * np.cos(mid_psi) * 1e3)[step], (mid_r * np.sin(mid_psi) * 1e3)[step],
                          np.ma.masked_invalid(u_x)[step], np.ma.masked_invalid(u_z)[step], scale=8.0, width=0.004)
            style = "--" if column == 0 else "-"
            ax.plot([BLADE_R0 * 1e3, BLADE_R1 * 1e3], [0, 0], color="k", linewidth=2.5, linestyle=style)
            arc = np.radians(np.linspace(-30, 30, 61))
            ax.plot(HUB_R * 1e3 * np.cos(arc), HUB_R * 1e3 * np.sin(arc), color="k", linewidth=0.8)
            ax.set_aspect("equal")
            ax.set_xlim(0, 73)
            ax.set_ylim(-37, 37)
            ax.set_title(f"{s.label}\n{title}", fontsize=7)
            ax.tick_params(labelsize=6)
    fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.6, label=label)
    fig.suptitle(f"around a Rushton blade (black, at angle 0), rotating frame; the blade moves up in this view, "
                 f"its wake is below; {label}" + ("; arrows: velocity relative to the blade" if quantity == "u_r" else ""),
                 fontsize=8)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", help="output prefix of the figures")
    parser.add_argument("--fluent", nargs=3, action="append", default=[], metavar=("IP", "RT_CSV", "VOLUME_IP"))
    parser.add_argument("--fluent-label", default=None)
    parser.add_argument("--fluent-rotor-lag", type=float, default=12.0,
                        help="deg, turn the velocity of the rotor-zone cells by this angle (see the docstring)")
    parser.add_argument("--sph", nargs="+", action="append", default=[], metavar="LABEL DUMP")
    parser.add_argument("--dx", type=float, default=0.003)
    parser.add_argument("--reference", default=None, help="SPH dump for the fluid fraction of the bins")
    parser.add_argument("--no-blade-figures", action="store_true")
    arguments = parser.parse_args()

    statistics = []
    if arguments.fluent:
        label = arguments.fluent_label or f"Fluent fine, {len(arguments.fluent)} snapshots"
        print(label)
        entry = Statistics(label)
        for sample in fluent_samples(arguments.fluent, arguments.fluent_rotor_lag):
            entry.add(sample)
        statistics.append(entry)
    for label, *paths in arguments.sph:
        print(label)
        entry = Statistics(label)
        for sample in sph_samples(paths, arguments.dx):
            entry.add(sample)
        statistics.append(entry)
    reference = arguments.reference or (arguments.sph[0][1] if arguments.sph else None)
    if reference is None:
        parser.error("--reference is needed without --sph")
    print("fluid fraction of the bins from the reference dump")
    fraction = fluid_fraction(reference, arguments.dx)
    report(statistics, fraction)
    plot_meridional(statistics, fraction, f"{arguments.out}_meridional.png")
    if not arguments.no_blade_figures:
        for quantity in ("u_r", "u_y", "pressure"):
            plot_blade_frame(statistics, quantity, f"{arguments.out}_blade_{quantity}.png")


if __name__ == "__main__":
    main()
