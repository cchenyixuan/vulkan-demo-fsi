"""
_analyze_blade_pressure_profiles.py — pressure on the faces of the thin-plate blades, where the torque sits on
the blade and how that moves with the resolution (2026-10-13, log/2026-10-13_blade-pressure-profiles.md).

Two sources of the face load, both folded onto one blade (the six blades of an impeller averaged), split into
the pressure side (front, the face that moves into the fluid) and the back, and binned on the blade's own
lattice (cells of about one spacing along the two in-plane axes of the plate):

  records   the thin-plate reaction records of a live run (readback_thin_plate_reactions, saved by
            _check_impeller_parts.py --reaction-times): fluid particle position x_i, force f_i ON the plate,
            plate index. Side of a record by the sign of (x_i - c) . n_front, exactly the rule of
            _check_impeller_parts.py (Rushton: ahead of / behind the blade along its motion; PBT: the normal
            turned toward the motion), which is the x z < 0 rule of _check_thin_plate.py rotor for a blade through
            the axis. The normal force of a record, -f_i . n_face (n_face = outward normal of that face, positive
            for a pressure that pushes on the face), is spread over the cells within the kernel support of x_i
            with the weights W(|x_i - cell|) / sum W: the record is the sum over the dummies the particle saw,
            and those sit in the plate within h of it. Face pressure of a cell = normal force / (cell area x
            mass factor). The torque of the face, of the radial bands and of the whole blade uses the record
            positions and the full force (what _check_impeller_parts.py reports), so the face sums must agree
            with the parts tables.
  dump      a dump of the run (_check_tank_energy.py / _check_impeller_parts.py --dump-times: positions,
            velocity_mass, material, density_pressure; the thin-plate particles carry -(index + 1) 1e9 in the
            pressure slot). The reaction of a fluid particle i on the dummies behind a plate is, to leading
            order, -2 P_i V_i sum_j V_j grad W_ij over the dummy half space = 2 P_i V_i x (integral of W over
            the plate area within the support of i) along the normal (divergence theorem; the one-sided tangential
            gradient of the dummy pressure is odd and drops out of the normal force), so the offline estimate of
            the normal force on cell k is
                F_k = sum_i 2 P_i V_i W(|x_i - cell_k|) A_k           (i on that side, |delta_i| < h)
            and the face pressure p_k = F_k / (A_k x mass factor). For a uniform fluid layer this gives p_k = P
            exactly (sum_i V_i W over a half space = 1/2 x mass factor). The one-sided Shepard interpolant
            sum P_i V_i W / sum V_i W of the same particles is kept as well ("pressure next to the face").
            The viscous part of the reaction and the clipping of the dummy region at the plate edges (dummies
            are neighbours whose segment to i crosses the plate INSIDE the outline) are not in the estimate; the
            records of the same step calibrate it.

For every impeller kind the script writes the cell maps of both faces (Pa, and C_p = (p - p_b) / (1/2 rho
(omega r)^2) with r the radius of the cell), chordwise and spanwise profiles (mean over the middle third of the
other coordinate), the torque of radial bands (Rushton: 6 bands of 4 mm from the root; PBT: the bands of
_analyze_pbt_bands.py) and the face totals, averaged over all dumps / record files given.

usage (base Anaconda: numpy, scipy, matplotlib, pyyaml):
    analyze one resolution (dumps and/or records; several files are averaged):
        python experiment/v1/checks/_analyze_blade_pressure_profiles.py analyze --case output/kecause/cases/s3_sheet2baf_smooth \\
            --label s3 --spacing 0.003 --out output/kecause/blade_profiles \\
            --dump output/kecause/s3_sheet2baf_smooth_fluentinit_final.npz \\
            --records output/kecause/blade_profiles/s3_sheet2baf_smooth_fluentinit_reactions_t*.npz
    figures and the markdown tables of several resolutions:
        python experiment/v1/checks/_analyze_blade_pressure_profiles.py plot --out output/kecause/blade_profiles --labels s3 s2 s15 s1
"""
import argparse
import glob
import json
import pathlib
import sys

import numpy as np

RHO0 = 998.0
THIN_PLATE_MARKER = 1.0e9
RUSHTON_BANDS = [0.024, 0.028, 0.032, 0.036, 0.040, 0.044, 0.048, 0.060]       # 6 bands of 4 mm + beyond the tip
PBT_BANDS = [0.0, 0.0122, 0.016, 0.024, 0.032, 0.040, 0.050, 0.060]           # _check_impeller_parts.py --pbt-bands
KINDS = ("rushton", "pbt")


# ----------------------------------------------------------------------------------------------- geometry
def rotation(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def wendland_c4(r, h):
    """3D Wendland C4 of the solver (support h)."""
    q = np.minimum(r / h, 1.0)
    return 495.0 / (32.0 * np.pi * h ** 3) * (1.0 - q) ** 6 * (1.0 + 6.0 * q + 35.0 / 3.0 * q ** 2)


class Plate:
    """A rotor thin plate of the case at rotor angle 0: centre c, normal n, in-plane axes a, b = n x a, half
    extents (ea, eb). kind rushton / pbt / disk / other; n_front points into the fluid on the pressure side
    (the normal turned toward the motion of the plate centre)."""

    def __init__(self, entry, omega_sign=1.0):
        self.name = entry["name"]
        self.shape = entry["shape"]
        self.centre = np.asarray(entry["centre"], dtype=np.float64)
        self.normal = np.asarray(entry["normal"], dtype=np.float64)
        self.axis_a = np.asarray(entry["axis_a"], dtype=np.float64)
        self.axis_b = np.cross(self.normal, self.axis_a)
        self.extent = tuple(float(v) for v in entry["extent"])
        if self.name.startswith("rushton_blade"):
            self.kind = "rushton"
        elif self.name.startswith("pbt_blade"):
            self.kind = "pbt"
        elif self.name == "rushton_disk":
            self.kind = "disk"
        else:
            self.kind = "other"
        radius = np.hypot(self.centre[0], self.centre[2])
        motion = omega_sign * np.array([self.centre[2], 0.0, -self.centre[0]]) / max(radius, 1e-12)
        self.n_front = self.normal if self.normal @ motion >= 0.0 else -self.normal
        # in-plane coordinates: u along axis_a, v along axis_b; the one closer to radial is the span
        radial = np.array([self.centre[0], 0.0, self.centre[2]]) / max(radius, 1e-12)
        self.span_axis = "a" if abs(self.axis_a @ radial) >= abs(self.axis_b @ radial) else "b"

    def rotated(self, theta):
        if getattr(self, "frame", "rotor") != "rotor":
            return (self.centre, self.n_front, self.axis_a, self.axis_b)
        turn = rotation(theta)
        return (turn @ self.centre, turn @ self.n_front, turn @ self.axis_a, turn @ self.axis_b)


def load_case(case_dir):
    import yaml
    case = yaml.safe_load((pathlib.Path(case_dir) / "case.yaml").read_text(encoding="utf-8"))
    materials = yaml.safe_load((pathlib.Path(case_dir) / case.get("material_library", "materials.yaml")).read_text(encoding="utf-8"))
    omega = None
    for name, entry in materials.items():
        if isinstance(entry, dict) and entry.get("kind") == "rotor":
            omega = float(entry["rotor_angular_velocity"])
    dx = 2.0 * float(case["physics"]["particle_radius"])
    info = {"dx": dx, "h": float(case["physics"]["h"]), "omega": omega,
            "p_background": float(case["physics"].get("background_pressure", 0.0)),
            "pivot": np.asarray(case.get("rotor", {}).get("pivot", [0.0, 0.0, 0.0]), dtype=np.float64),
            "ramp_time": float(case.get("rotor", {}).get("ramp_time", 0.0))}
    # buffer order of utils/sph/case.py build_thin_plate_groups (the plate index of the records and of the dump
    # marker): static plates first, then the rotor plates by height along the axis (stable for equal heights)
    entries = case.get("thin_plates", [])
    static = [entry for entry in entries if entry.get("frame") != "rotor"]
    moving = [entry for entry in entries if entry.get("frame") == "rotor"]
    axis = np.asarray(case.get("rotor", {}).get("axis", [0.0, 1.0, 0.0]), dtype=np.float64)
    height = [float((np.asarray(entry["centre"], dtype=np.float64) - info["pivot"]) @ axis) for entry in moving]
    moving = [moving[int(k)] for k in np.argsort(height, kind="stable")]
    plates = [Plate(entry, np.sign(omega)) for entry in static + moving]
    for plate, entry in zip(plates, static + moving):
        plate.frame = entry.get("frame", "static")
        if plate.frame != "rotor":
            plate.kind = "other"
    return info, plates


class CellGrid:
    """Cells on one blade of a kind: cell size about dx along the two in-plane axes (a, b), area a_cell x b_cell;
    the same grid for the six blades of the impeller. r and y of every cell are those of the first blade."""

    def __init__(self, plate, dx):
        self.kind = plate.kind
        ea, eb = plate.extent
        self.na, self.nb = max(int(round(2.0 * ea / dx)), 1), max(int(round(2.0 * eb / dx)), 1)
        self.da, self.db = 2.0 * ea / self.na, 2.0 * eb / self.nb
        self.a = -ea + (np.arange(self.na) + 0.5) * self.da
        self.b = -eb + (np.arange(self.nb) + 0.5) * self.db
        self.area = self.da * self.db
        A, B = np.meshgrid(self.a, self.b, indexing="ij")
        points = plate.centre + A[..., None] * plate.axis_a + B[..., None] * plate.axis_b
        self.r = np.hypot(points[..., 0], points[..., 2])
        self.y = points[..., 1]
        self.span_axis = plate.span_axis

    def sites(self, centre, axis_a, axis_b):
        A, B = np.meshgrid(self.a, self.b, indexing="ij")
        return (centre + A[..., None] * axis_a + B[..., None] * axis_b).reshape(-1, 3)


# ----------------------------------------------------------------------------------------------- accumulation
class FaceLoad:
    """Sums for one impeller kind: per face the normal force of every cell (N, with the mass factor), the face
    totals (force vector, torque about +y from the record positions / cell sites), and the band torques."""

    def __init__(self, grid, bands, blade_count=1):
        self.grid = grid
        self.bands = list(bands)
        self.blade_count = blade_count
        self.samples = 0
        self.cell_force = {face: np.zeros((grid.na, grid.nb)) for face in ("front", "back")}
        self.cell_force_shepard_num = {face: np.zeros((grid.na, grid.nb)) for face in ("front", "back")}
        self.cell_force_shepard_den = {face: np.zeros((grid.na, grid.nb)) for face in ("front", "back")}
        self.force = {face: np.zeros(3) for face in ("front", "back")}
        self.torque = {face: 0.0 for face in ("front", "back")}
        self.torque_normal = {face: 0.0 for face in ("front", "back")}
        self.band_torque = {face: np.zeros(len(bands) - 1) for face in ("front", "back")}
        self.band_torque_cells = {face: np.zeros(len(bands) - 1) for face in ("front", "back")}
        self.record_count = {face: 0 for face in ("front", "back")}

    def finish(self, mass_factor):
        n = max(self.samples, 1) * self.blade_count       # cell maps: one blade (the blades of the kind averaged)
        area = self.grid.area
        out = {"na": self.grid.na, "nb": self.grid.nb, "a": self.grid.a, "b": self.grid.b, "r": self.grid.r, "y": self.grid.y,
               "area": area, "span_axis": self.grid.span_axis, "bands": np.asarray(self.bands), "samples": self.samples}
        for face in ("front", "back"):
            out[f"p_{face}"] = self.cell_force[face] / (n * area * mass_factor)
            with np.errstate(invalid="ignore", divide="ignore"):
                out[f"p_shepard_{face}"] = np.where(self.cell_force_shepard_den[face] > 0,
                                                    self.cell_force_shepard_num[face] / self.cell_force_shepard_den[face], np.nan)
            n_total = max(self.samples, 1)                   # totals: all blades of the kind (parts convention)
            out[f"force_{face}"] = self.force[face] / (n_total * mass_factor)
            out[f"torque_{face}"] = self.torque[face] / (n_total * mass_factor)
            out[f"torque_normal_{face}"] = self.torque_normal[face] / (n_total * mass_factor)
            out[f"band_torque_{face}"] = self.band_torque[face] / (n_total * mass_factor)
            out[f"band_torque_cells_{face}"] = self.band_torque_cells[face] / (n_total * mass_factor)
            out[f"records_{face}"] = self.record_count[face] / n_total
        return out


def resistance_torque(points, forces, pivot):
    """-(x - pivot) x f . y: positive = against a rotation with omega > 0 about +y (parts convention)."""
    d = points - pivot
    return -(d[:, 2] * forces[:, 0] - d[:, 0] * forces[:, 2])


def match_plates(plates, position, plate_index, theta):
    """plate of every record index by geometry (the buffer order of the plates sorts equal heights in an order
    this script does not reproduce): the centroid of the records of an index is matched to the nearest rotated
    plate centre. Returns {index: plate}."""
    mapping = {}
    rotated = [plate.rotated(theta)[0] for plate in plates]
    for index in np.unique(plate_index):
        centroid = position[plate_index == index].mean(axis=0)
        distances = [float(np.linalg.norm(centroid - c)) for c in rotated]
        best = int(np.argmin(distances))
        plate = plates[best]
        if distances[best] > 0.3 * float(np.hypot(*plate.extent)) + 0.003:
            print(f"   WARNING: records of index {index}: centroid {distances[best] * 1e3:.1f} mm from the nearest plate "
                  f"({plate.name})", file=sys.stderr)
        if plate in mapping.values():
            print(f"   WARNING: plate {plate.name} matched twice", file=sys.stderr)
        mapping[int(index)] = plate
    return mapping


def assign_records(plates, position, plate_index, theta, h, split):
    """Plate (index into `plates`) of every record. split = "label": the plate index of the record (the plate
    of the first dummy the particle met, what _check_impeller_parts.py splits by). split = "geometry": a record
    labelled with the Rushton disk that lies in the region of a Rushton blade (|delta| < h, inside the outline
    widened by h/2) goes to that blade (the nearest by |delta|). The blades are 19.2 mm tall with the disk in
    the middle, so every fluid particle next to a blade is also within h of the disk and its label falls on
    the disk whenever a disk dummy came first in its neighbour list (2026-10-13: the whole torque of the
    disk-labelled records sits in the blade regions, log/2026-10-13_blade-pressure-profiles.md)."""
    mapping = match_plates(plates, position, plate_index, theta)
    assigned = np.full(len(plate_index), -1, dtype=np.int64)
    for index, plate in mapping.items():
        assigned[plate_index == index] = plates.index(plate)
    moved = 0
    if split == "geometry":
        disk = [plates.index(plate) for plate in plates if plate.kind == "disk"]
        on_disk = np.isin(assigned, disk)
        best_delta = np.full(len(plate_index), np.inf)
        best_plate = np.full(len(plate_index), -1, dtype=np.int64)
        for plate in plates:
            if plate.kind != "rushton":
                continue
            centre, n_front, axis_a, axis_b = plate.rotated(theta)
            d = position - centre
            delta = np.abs(d @ n_front)
            ea, eb = plate.extent
            inside = on_disk & (delta < h) & (np.abs(d @ axis_a) < ea + 0.5 * h) & (np.abs(d @ axis_b) < eb + 0.5 * h) & (delta < best_delta)
            best_delta[inside] = delta[inside]
            best_plate[inside] = plates.index(plate)
        move = best_plate >= 0
        assigned[move] = best_plate[move]
        moved = int(move.sum())
    return assigned, moved


def accumulate_records(loads, plates, grid_of, position, force, plate_index, theta, h, pivot, split="geometry"):
    """One record file into the loads (per kind)."""
    assigned, moved = assign_records(plates, position, plate_index, theta, h, split)
    if moved:
        print(f"   {moved} disk-labelled records reassigned to the Rushton blades (geometry split)")
    for plate in plates:
        if plate.kind not in loads:
            continue
        mask = assigned == plates.index(plate)
        if not mask.any():
            continue
        x, f = position[mask], force[mask]
        centre, n_front, axis_a, axis_b = plate.rotated(theta)
        load, grid = loads[plate.kind], grid_of[plate.kind]
        # a record of this plate lies within h of its plane (safety against a wrong plate match)
        d = x - centre
        ea, eb = plate.extent
        keep = (np.abs(d @ n_front) < h) & (np.abs(d @ axis_a) < ea + h) & (np.abs(d @ axis_b) < eb + h)
        if not keep.all():
            print(f"   WARNING: {int((~keep).sum())} records of {plate.name} farther than h from the plate dropped", file=sys.stderr)
            x, f = x[keep], f[keep]
        sites = grid.sites(centre, axis_a, axis_b)
        delta = (x - centre) @ n_front
        r = np.hypot(x[:, 0] - pivot[0], x[:, 2] - pivot[2])
        tau = resistance_torque(x, f, pivot)
        for face, select in (("front", delta > 0.0), ("back", delta <= 0.0)):
            if not select.any():
                continue
            n_face = n_front if face == "front" else -n_front
            normal_force = -(f[select] @ n_face)                       # > 0: pressure pushing on this face
            # spread over the cells within h (kernel weights, normalised per record)
            dist = np.linalg.norm(x[select][:, None, :] - sites[None, :, :], axis=2)
            w = wendland_c4(dist, h)
            w[dist >= h] = 0.0
            total = w.sum(axis=1)
            has = total > 0
            w[has] /= total[has][:, None]
            cell = (normal_force[:, None] * w).sum(axis=0).reshape(grid.na, grid.nb)
            load.cell_force[face] += cell
            load.force[face] += f[select].sum(axis=0)
            load.torque[face] += tau[select].sum()
            # torque of the normal part only, at the record positions
            f_normal = -normal_force[:, None] * n_face[None, :]
            load.torque_normal[face] += resistance_torque(x[select], f_normal, pivot).sum()
            load.band_torque[face] += np.histogram(r[select], bins=load.bands, weights=tau[select])[0]
            # the same torque from the cells: normal force of the cell at the cell site
            site_torque = resistance_torque(sites, -cell.reshape(-1)[:, None] * n_face[None, :], pivot)
            load.band_torque_cells[face] += np.histogram(grid.r.reshape(-1), bins=load.bands, weights=site_torque)[0]
            load.record_count[face] += int(select.sum())


def wendland_c4_gradient(d, r, h):
    """gradient at x_i of W(|x_i - x_j|), d = x_i - x_j (n, 3), r = |d| (n,)."""
    q = r / h
    inside = q < 1.0
    factor = np.zeros_like(r)
    factor[inside] = (495.0 / (32.0 * np.pi * h ** 4)) * (1.0 - q[inside]) ** 5 * q[inside] * (-280.0 / 3.0 * q[inside] - 56.0 / 3.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        unit = np.where(r[:, None] > 0, d / np.maximum(r, 1e-30)[:, None], 0.0)
    return factor[:, None] * unit


def accumulate_dump(loads, plates, grid_of, x_all, pressure_all, volume_all, is_fluid, plate_of_particle, theta, h, pivot,
                    xi=0.01, det_threshold=1.0e-4, chunk=60000):
    """One dump into the loads: the pressure part of the thin-plate reaction rebuilt from the dump with the
    dummy rule of thin_plates.glsl. For a fluid particle i within h of a rotor plate, its neighbours j within h
    are dummies of that plate when j is a particle of the plate, or when the segment (x_i, x_j) crosses the
    plate plane inside the outline. Pair force on i (pressure term of force.comp with the dummy pressure
    P_j = P_i, the tangential one-sided gradient dropped; TIC: nothing for P_i < 0):
        f_ij = -2 P_i V_i V_j B_i grad W_ij,      B_i = (A_i + xi I)^-1,  A_i = sum_j V_j (x_j - x_i) (x) grad W_ij
    over ALL neighbours (fluid, wall, rotor, plate particles), as correction.comp does. The reaction -f_ij is
    put on the plate at the crossing point (plate particle: its position) -> cell; the torque uses x_i (as the
    records do). x_all, pressure_all, volume_all, is_fluid, plate_of_particle (-1, or the index into `plates`)
    describe the live particles near the impellers."""
    from scipy.spatial import cKDTree

    tree = cKDTree(x_all)
    rotated = {id(plate): plate.rotated(theta) for plate in plates}
    # fluid particles within h of any plate of interest (and within the outline + h)
    candidate = np.zeros(len(x_all), dtype=bool)
    for plate in plates:
        if plate.kind not in loads:
            continue
        centre, n_front, axis_a, axis_b = rotated[id(plate)]
        d = x_all - centre
        ea, eb = plate.extent
        candidate |= is_fluid & (np.abs(d @ n_front) < h) & (np.abs(d @ axis_a) < ea + h) & (np.abs(d @ axis_b) < eb + h)
    fluid_index = np.flatnonzero(candidate)
    for begin in range(0, len(fluid_index), chunk):
        ids = fluid_index[begin:begin + chunk]
        neighbours = tree.query_ball_point(x_all[ids], h)
        counts = np.fromiter((len(n) for n in neighbours), dtype=np.int64, count=len(neighbours))
        i_of_pair = np.repeat(np.arange(len(ids)), counts)
        j_of_pair = np.fromiter((j for n in neighbours for j in n), dtype=np.int64, count=int(counts.sum()))
        xi_pair, xj_pair = x_all[ids[i_of_pair]], x_all[j_of_pair]
        d = xi_pair - xj_pair
        r = np.linalg.norm(d, axis=1)
        self_pair = j_of_pair == ids[i_of_pair]
        grad = wendland_c4_gradient(d, r, h)
        grad[self_pair] = 0.0
        vj = volume_all[j_of_pair]
        # KCG matrix of every i from all neighbours
        outer = (vj[:, None, None] * (-d)[:, :, None] * grad[:, None, :])          # V_j (x_j - x_i) (x) grad W
        A = np.zeros((len(ids), 3, 3))
        np.add.at(A, i_of_pair, outer)
        A_reg = A + xi * np.eye(3)
        det = np.linalg.det(A_reg)
        regular = np.abs(det) > det_threshold
        B = np.tile(np.eye(3), (len(ids), 1, 1))
        B[regular] = np.linalg.inv(A_reg[regular])
        corrected = np.einsum("pab,pb->pa", B[i_of_pair], grad)                       # B_i grad W_ij
        pi, vi = pressure_all[ids[i_of_pair]], volume_all[ids[i_of_pair]]
        pair_force = -(2.0 * np.where(pi > 0.0, pi, 0.0) * vi * vj)[:, None] * corrected   # on i, if j is a dummy
        for plate in plates:
            if plate.kind not in loads:
                continue
            load, grid = loads[plate.kind], grid_of[plate.kind]
            centre, n_front, axis_a, axis_b = rotated[id(plate)]
            ea, eb = plate.extent
            delta_i = (xi_pair - centre) @ n_front
            delta_j = (xj_pair - centre) @ n_front
            on_plate = plate_of_particle[j_of_pair] == plates.index(plate)
            crossing = (delta_i * delta_j < 0.0) & ~on_plate
            # crossing point of the segment with the plane (plate particles: their own position)
            with np.errstate(invalid="ignore", divide="ignore"):
                t = np.where(crossing, delta_i / np.where(delta_i - delta_j != 0, delta_i - delta_j, 1.0), 0.0)
            point = np.where(on_plate[:, None], xj_pair, xi_pair + t[:, None] * (xj_pair - xi_pair))
            u, v = (point - centre) @ axis_a, (point - centre) @ axis_b
            if plate.shape == "annulus":
                rr = np.hypot(u, v)
                inside = (rr <= ea) & (rr >= eb)
            else:
                inside = (np.abs(u) <= ea) & (np.abs(v) <= eb)
            dummy = ((crossing & inside) | on_plate) & (np.abs(delta_i) < h) & ~self_pair
            if not dummy.any() or plate.shape == "annulus":
                continue
            sites = grid.sites(centre, axis_a, axis_b)
            for face, select in (("front", dummy & (delta_i > 0.0)), ("back", dummy & (delta_i <= 0.0))):
                if not select.any():
                    continue
                n_face = n_front if face == "front" else -n_front
                reaction = -pair_force[select]                                            # on the plate
                normal_force = -(reaction @ n_face)                                      # > 0: pushing on the face
                ia = np.clip(np.floor((u[select] + ea) / grid.da).astype(np.int64), 0, grid.na - 1)
                ib = np.clip(np.floor((v[select] + eb) / grid.db).astype(np.int64), 0, grid.nb - 1)
                cell = np.zeros((grid.na, grid.nb))
                np.add.at(cell, (ia, ib), normal_force)
                load.cell_force[face] += cell
                load.force[face] += reaction.sum(axis=0)
                tau = resistance_torque(xi_pair[select], reaction, pivot)
                load.torque[face] += tau.sum()
                load.torque_normal[face] += resistance_torque(xi_pair[select], -normal_force[:, None] * n_face[None, :], pivot).sum()
                r_i = np.hypot(xi_pair[select][:, 0] - pivot[0], xi_pair[select][:, 2] - pivot[2])
                load.band_torque[face] += np.histogram(r_i, bins=load.bands, weights=tau)[0]
                site_torque = resistance_torque(sites, -cell.reshape(-1)[:, None] * n_face[None, :], pivot)
                load.band_torque_cells[face] += np.histogram(grid.r.reshape(-1), bins=load.bands, weights=site_torque)[0]
                load.record_count[face] += int(np.unique(i_of_pair[select]).size)
        # one-sided Shepard pressure next to each face (fluid particles of that side within h, kernel weights)
        xi_chunk = x_all[ids]
        for plate in plates:
            if plate.kind not in loads or plate.shape == "annulus":
                continue
            load, grid = loads[plate.kind], grid_of[plate.kind]
            centre, n_front, axis_a, axis_b = rotated[id(plate)]
            dd = xi_chunk - centre
            delta = dd @ n_front
            ea, eb = plate.extent
            near = (np.abs(delta) < h) & (np.abs(dd @ axis_a) < ea + h) & (np.abs(dd @ axis_b) < eb + h)
            if not near.any():
                continue
            sites = grid.sites(centre, axis_a, axis_b)
            for face, select in (("front", near & (delta > 0.0)), ("back", near & (delta <= 0.0))):
                if not select.any():
                    continue
                dist = np.linalg.norm(xi_chunk[select][:, None, :] - sites[None, :, :], axis=2)
                w = wendland_c4(dist, h)
                w[dist >= h] = 0.0
                pv = pressure_all[ids][select] * volume_all[ids][select]
                load.cell_force_shepard_num[face] += (pv[:, None] * w).sum(axis=0).reshape(grid.na, grid.nb)
                load.cell_force_shepard_den[face] += (volume_all[ids][select][:, None] * w).sum(axis=0).reshape(grid.na, grid.nb)


# ----------------------------------------------------------------------------------------------- inputs
def rotor_angle_of_dump(dump, plates, live, pressure_slot):
    """Rotor angle from the thin-plate particles of the dump (marker -(index + 1) 1e9 in the pressure slot):
    the centroid of every marked plate is matched to the rotated case centres; returns (theta, mapping check)."""
    marked = live & (pressure_slot < -1e8)
    index = np.rint(-pressure_slot[marked] / THIN_PLATE_MARKER).astype(int) - 1
    points = dump["positions"][marked, :3].astype(np.float64)
    centroids = {int(k): points[index == k].mean(axis=0) for k in np.unique(index)}
    # the angle from the plate with the largest radius (most sensitive): try all, take the median
    angles = []
    for k, c in centroids.items():
        r = np.hypot(c[0], c[2])
        if r < 0.015:
            continue
        candidates = [p for p in plates if abs(np.hypot(p.centre[0], p.centre[2]) - r) < 0.004 and abs(p.centre[1] - c[1]) < 0.004]
        if not candidates:
            continue
        for p in candidates:
            angles.append((np.arctan2(c[0], c[2]) - np.arctan2(p.centre[0], p.centre[2])) % (2.0 * np.pi))
    angles = np.asarray(angles)
    # cluster on the circle: take the angle with most support within 1 degree
    best, support = None, -1
    for a in angles:
        n = int((np.abs((angles - a + np.pi) % (2.0 * np.pi) - np.pi) < np.radians(1.0)).sum())
        if n > support:
            best, support = a, n
    close = np.abs((angles - best + np.pi) % (2.0 * np.pi) - np.pi) < np.radians(1.0)
    theta = float(best + np.mean((angles[close] - best + np.pi) % (2.0 * np.pi) - np.pi))
    return theta, support, len(angles)


def analyze(arguments):
    info, plates = load_case(arguments.case)
    dx, h, pivot = arguments.spacing or info["dx"], info["h"], info["pivot"]
    out = pathlib.Path(arguments.out)
    out.mkdir(parents=True, exist_ok=True)
    grid_of = {}
    for plate in plates:
        if plate.kind in KINDS and plate.kind not in grid_of:
            grid_of[plate.kind] = CellGrid(plate, dx)
    bands_of = {"rushton": RUSHTON_BANDS, "pbt": PBT_BANDS}
    blade_count = {kind: sum(1 for plate in plates if plate.kind == kind) for kind in grid_of}
    results = {}
    log = {"case": str(arguments.case), "dx": dx, "h": h, "omega": info["omega"], "p_background": info["p_background"],
           "dumps": [], "records": []}

    dump_files = [f for pattern in arguments.dump for f in sorted(glob.glob(pattern))]
    record_files = [f for pattern in arguments.records for f in sorted(glob.glob(pattern))]
    if not dump_files and not record_files:
        raise SystemExit("nothing to analyze: give --dump and/or --records")

    if dump_files:
        loads = {kind: FaceLoad(grid_of[kind], bands_of[kind], blade_count[kind]) for kind in grid_of}
        mass_factor = None
        for file in dump_files:
            dump = np.load(file)
            positions = dump["positions"]
            live = positions[:, 3] > 0
            live[0] = False
            density_pressure = dump["density_pressure"]
            theta, support, total = rotor_angle_of_dump(dump, plates, live, density_pressure[:, 1])
            material = dump["material"]
            x_all = positions[:, :3]
            r_all = np.hypot(x_all[:, 0] - pivot[0], x_all[:, 2] - pivot[2])
            y_all = x_all[:, 1]
            box = np.zeros(live.shape, dtype=bool)
            for plate in plates:
                if plate.kind not in grid_of:
                    continue
                ea, eb = plate.extent
                reach = float(np.hypot(ea, eb)) + h
                rc = np.hypot(plate.centre[0], plate.centre[2])
                box |= (r_all < rc + reach) & (np.abs(y_all - plate.centre[1]) < reach)
            near = live & box
            x = x_all[near].astype(np.float64)
            pressure = density_pressure[near, 1].astype(np.float64)
            density = density_pressure[near, 0].astype(np.float64)
            mass = dump["velocity_mass"][near, 3].astype(np.float64)
            is_fluid = material[near] == 0
            marked = pressure < -1e8
            marker_index = np.rint(-pressure[marked] / THIN_PLATE_MARKER).astype(np.int64) - 1
            pressure[marked] = 0.0
            density[marked] = RHO0
            volume = mass / density
            mass_factor = float(np.median(mass[is_fluid]) / (RHO0 * dx ** 3))
            # plate particle -> plate (geometric match of the marker groups, like the records)
            plate_of_particle = np.full(len(x), -1, dtype=np.int64)
            mapping = match_plates(plates, x[marked], marker_index, theta)
            plate_of_particle[np.flatnonzero(marked)] = [plates.index(mapping[int(k)]) for k in marker_index]
            accumulate_dump(loads, plates, grid_of, x, pressure, volume, is_fluid, plate_of_particle, theta, h, pivot)
            for load in loads.values():
                load.samples += 1
            log["dumps"].append({"file": file, "rotor_angle": theta, "angle_support": f"{support}/{total}",
                                 "particles_near_impellers": int(near.sum()), "mass_factor": mass_factor})
            print(f"dump {file}: rotor angle {theta:.4f} rad ({support}/{total} plates agree), {int(near.sum()):,} "
                  f"particles near the impellers ({int(is_fluid.sum()):,} fluid), mass factor {mass_factor:.4f}")
        results["dump"] = {kind: load.finish(mass_factor) for kind, load in loads.items()}

    if record_files:
        loads = {kind: FaceLoad(grid_of[kind], bands_of[kind], blade_count[kind]) for kind in grid_of}
        mass_factor = None
        for file in record_files:
            rec = np.load(file)
            theta = float(rec["rotor_angle"])
            mass_factor = float(rec["mass_factor"])
            accumulate_records(loads, plates, grid_of, rec["position"].astype(np.float64), rec["force"].astype(np.float64),
                               rec["plate"], theta, h, pivot, arguments.records_split)
            for load in loads.values():
                load.samples += 1
            log["records"].append({"file": file, "time": float(rec["time"]), "rotor_angle": theta, "mass_factor": mass_factor,
                                   "count": int(rec["plate"].shape[0])})
            print(f"records {file}: t {float(rec['time']):.3f} s, rotor angle {theta:.4f} rad, {int(rec['plate'].shape[0]):,} records, "
                  f"mass factor {mass_factor:.4f}")
        results["records" if arguments.records_split == "geometry" else "records_label"] = {kind: load.finish(mass_factor) for kind, load in loads.items()}

    # ---- save
    payload = {"label": arguments.label, "dx": dx, "h": h, "omega": info["omega"], "p_background": info["p_background"],
               "sources": json.dumps(log)}
    for source, kinds in results.items():
        for kind, data in kinds.items():
            for key, value in data.items():
                payload[f"{source}/{kind}/{key}"] = value
    np.savez(out / f"{arguments.label}_profiles.npz", **payload)
    summary = {"label": arguments.label, "dx": dx, "h": h, "sources": log}
    for source, kinds in results.items():
        summary[source] = {}
        for kind, data in kinds.items():
            summary[source][kind] = {
                "torque_front_mNm": 1e3 * float(data["torque_front"]), "torque_back_mNm": 1e3 * float(data["torque_back"]),
                "torque_normal_front_mNm": 1e3 * float(data["torque_normal_front"]),
                "torque_normal_back_mNm": 1e3 * float(data["torque_normal_back"]),
                "force_front_N": data["force_front"].tolist(), "force_back_N": data["force_back"].tolist(),
                "bands_mm": (1e3 * data["bands"]).tolist(),
                "band_torque_front_mNm": (1e3 * data["band_torque_front"]).tolist(),
                "band_torque_back_mNm": (1e3 * data["band_torque_back"]).tolist(),
                "band_torque_cells_front_mNm": (1e3 * data["band_torque_cells_front"]).tolist(),
                "band_torque_cells_back_mNm": (1e3 * data["band_torque_cells_back"]).tolist(),
                "mean_p_front_Pa": float(np.nanmean(data["p_front"])), "mean_p_back_Pa": float(np.nanmean(data["p_back"])),
                "records_per_sample_front": float(data["records_front"]), "records_per_sample_back": float(data["records_back"]),
                "samples": int(data["samples"])}
    (out / f"{arguments.label}_profiles.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print_summary(summary)


def print_summary(summary):
    print(f"\n{summary['label']}: dx {summary['dx'] * 1e3:g} mm, resistance torque mN m (positive = against the rotation)")
    for source in ("records", "records_label", "dump"):
        if source not in summary:
            continue
        for kind in KINDS:
            if kind not in summary[source]:
                continue
            d = summary[source][kind]
            print(f"  {source:13s} {kind:8s} front {d['torque_front_mNm']:8.2f}  back {d['torque_back_mNm']:8.2f}  "
                  f"net {d['torque_front_mNm'] + d['torque_back_mNm']:7.2f}   (normal part only: front "
                  f"{d['torque_normal_front_mNm']:7.2f}, back {d['torque_normal_back_mNm']:7.2f})   "
                  f"mean p front {d['mean_p_front_Pa']:7.1f} back {d['mean_p_back_Pa']:7.1f} Pa")
            edges = d["bands_mm"]
            cells = "  ".join(f"[{edges[i]:.0f},{edges[i + 1]:.0f}) {d['band_torque_front_mNm'][i] + d['band_torque_back_mNm'][i]:6.2f}"
                              for i in range(len(edges) - 1))
            print(f"           bands net: {cells}")


# ----------------------------------------------------------------------------------------------- plots
def profile(data, face, along, kind):
    """1D profile of the face pressure along the span (vs r) or along the chord, mean over the middle third of
    the other in-plane coordinate. Returns (coordinate, p (Pa), C_p, r of the cells)."""
    p = data[f"p_{face}"]
    r, y = data["r"], data["y"]
    span_is_a = data["span_axis"] == "a"
    if (along == "span") == span_is_a:
        # profile along axis a, average over the middle third of b
        nb = p.shape[1]
        lo, hi = nb // 3, nb - nb // 3
        if hi <= lo:
            lo, hi = 0, nb
        pm, rm, ym = p[:, lo:hi].mean(axis=1), r[:, lo:hi].mean(axis=1), y[:, lo:hi].mean(axis=1)
    else:
        na = p.shape[0]
        lo, hi = na // 3, na - na // 3
        if hi <= lo:
            lo, hi = 0, na
        pm, rm, ym = p[lo:hi, :].mean(axis=0), r[lo:hi, :].mean(axis=0), y[lo:hi, :].mean(axis=0)
    coordinate = rm if along == "span" else ym
    return coordinate, pm, rm


def cp_of(p, r, omega, p_background):
    return (p - p_background) / (0.5 * RHO0 * (omega * r) ** 2)


def order_estimate(h, values):
    """Order of the change with the spacing from the three finest levels: v(h) = v_inf + C h^p, so
    (v(h1) - v(h2)) / (v(h2) - v(h3)) = (h1^p - h2^p) / (h2^p - h3^p) with h1 > h2 > h3, solved for p by
    bisection in (0.05, 6). Returns (p, v_inf) or (nan, nan) when the sequence is not monotone over the
    three levels (then no order can be read off)."""
    order = np.argsort(-np.asarray(h, dtype=np.float64))
    h = np.asarray(h, dtype=np.float64)[order][-3:]
    v = np.asarray(values, dtype=np.float64)[order][-3:]
    if len(h) < 3:
        return float("nan"), float("nan")
    d12, d23 = v[0] - v[1], v[1] - v[2]
    if d12 * d23 <= 0.0 or abs(d23) < 1e-12:
        return float("nan"), float("nan")
    target = d12 / d23

    def ratio(p):
        return (h[0] ** p - h[1] ** p) / (h[1] ** p - h[2] ** p)

    lo, hi = 0.05, 6.0
    if not (min(ratio(lo), ratio(hi)) <= target <= max(ratio(lo), ratio(hi))):
        return float("nan"), float("nan")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if (ratio(mid) - target) * (ratio(lo) - target) <= 0.0:
            hi = mid
        else:
            lo = mid
    p_order = 0.5 * (lo + hi)
    c = d23 / (h[1] ** p_order - h[2] ** p_order)
    return float(p_order), float(v[2] - c * h[2] ** p_order)


def plot(arguments):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors

    out = pathlib.Path(arguments.out)
    runs = []
    for label in arguments.labels:
        file = out / f"{label}_profiles.npz"
        if not file.exists():
            print(f"missing {file}", file=sys.stderr)
            continue
        z = np.load(file, allow_pickle=False)
        data = {}
        for key in z.files:
            if "/" in key:
                source, kind, name = key.split("/", 2)
                data.setdefault(source, {}).setdefault(kind, {})[name] = z[key]
        for source in data:
            for kind in data[source]:
                data[source][kind]["span_axis"] = str(data[source][kind]["span_axis"])
        runs.append({"label": label, "dx": float(z["dx"]), "omega": float(z["omega"]), "p_background": float(z["p_background"]),
                     "data": data})
    runs.sort(key=lambda run: -run["dx"])
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    sources = [s for s in ("dump", "records") if any(s in run["data"] for run in runs)]
    lines = []     # markdown tables

    # ---- 1D profiles: rows span / chord, columns front / back / front - back; C_p and Pa
    for source in sources:
        for kind in KINDS:
            for unit in ("cp", "pa"):
                fig, axes = plt.subplots(2, 3, figsize=(15, 7.5))
                for row, along in enumerate(("span", "chord")):
                    for column, face in enumerate(("front", "back", "difference")):
                        ax = axes[row, column]
                        for k, run in enumerate(runs):
                            if source not in run["data"] or kind not in run["data"][source]:
                                continue
                            data = run["data"][source][kind]
                            if face == "difference":
                                coordinate, p_front, r = profile(data, "front", along, kind)
                                _, p_back, _ = profile(data, "back", along, kind)
                                value = p_front - p_back
                                if unit == "cp":
                                    value = value / (0.5 * RHO0 * (run["omega"] * r) ** 2)
                            else:
                                coordinate, p, r = profile(data, face, along, kind)
                                value = cp_of(p, r, run["omega"], run["p_background"]) if unit == "cp" else p - run["p_background"]
                            ax.plot(coordinate * 1e3, value, "o-", ms=3, color=palette[k % len(palette)], label=f"{run['dx'] * 1e3:g} mm")
                        ax.set_xlabel("r of the cell, mm" if along == "span" else "y of the cell, mm")
                        if face == "difference":
                            ax.set_ylabel("(p_front - p_back) / (1/2 rho (omega r)^2)" if unit == "cp" else "p_front - p_back, Pa")
                        else:
                            ax.set_ylabel("C_p = (p - p_b) / (1/2 rho (omega r)^2)" if unit == "cp" else "p - p_b, Pa")
                        where = "spanwise (middle third of the chord)" if along == "span" else "chordwise (middle third of the span)"
                        ax.set_title(f"{kind} {face if face != 'difference' else 'front - back'}, {where}", fontsize=9)
                        ax.grid(alpha=0.3)
                        ax.legend(fontsize=8)
                what = "reaction records" if source == "records" else "near-face pressure rebuilt from the dump"
                fig.suptitle(f"{kind}: face pressure from the {source} ({what}), {'coefficient' if unit == 'cp' else 'Pa above the background'}", fontsize=10)
                fig.tight_layout()
                fig.savefig(out / f"profiles_{kind}_{source}{'' if unit == 'cp' else '_pa'}.png", dpi=130)
                plt.close(fig)

    # ---- 2D maps of C_p (one row per resolution, front / back / difference)
    for source in sources:
        for kind in KINDS:
            available = [run for run in runs if source in run["data"] and kind in run["data"][source]]
            if not available:
                continue
            fig, axes = plt.subplots(len(available), 3, figsize=(12, 3.2 * len(available)), squeeze=False)
            all_cp = []
            for run in available:
                d = run["data"][source][kind]
                for face in ("front", "back"):
                    all_cp.append(cp_of(d[f"p_{face}"], d["r"], run["omega"], run["p_background"]))
            vmax = float(np.nanpercentile(np.abs(np.concatenate([c.ravel() for c in all_cp])), 99))
            for row, run in enumerate(available):
                d = run["data"][source][kind]
                a, b = d["a"] * 1e3, d["b"] * 1e3
                cps = {face: cp_of(d[f"p_{face}"], d["r"], run["omega"], run["p_background"]) for face in ("front", "back")}
                for column, (name, field) in enumerate((("front", cps["front"]), ("back", cps["back"]), ("front - back", cps["front"] - cps["back"]))):
                    ax = axes[row, column]
                    extent = [a[0] - 0.5 * (a[1] - a[0]) if len(a) > 1 else a[0] - 1, a[-1] + 0.5 * (a[1] - a[0]) if len(a) > 1 else a[0] + 1,
                              b[0] - 0.5 * (b[1] - b[0]) if len(b) > 1 else b[0] - 1, b[-1] + 0.5 * (b[1] - b[0]) if len(b) > 1 else b[0] + 1]
                    image = ax.imshow(field.T, origin="lower", extent=extent, aspect="equal", cmap="RdBu_r",
                                      norm=colors.TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax))
                    ax.set_title(f"{run['dx'] * 1e3:g} mm, {name}", fontsize=9)
                    ax.set_xlabel("axis a, mm" + (" (span, radial)" if d["span_axis"] == "a" else " (chord)"))
                    ax.set_ylabel("axis b, mm" + (" (span, radial)" if d["span_axis"] == "b" else " (chord)"))
                    fig.colorbar(image, ax=ax, fraction=0.04)
            fig.suptitle(f"{kind}: C_p on the blade from the {source}", fontsize=10)
            fig.tight_layout()
            fig.savefig(out / f"maps_{kind}_{source}.png", dpi=120)
            plt.close(fig)

    # ---- band torques and face totals: figure + tables
    for source in sources:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for column, kind in enumerate(KINDS):
            ax = axes[column]
            for k, run in enumerate(runs):
                if source not in run["data"] or kind not in run["data"][source]:
                    continue
                d = run["data"][source][kind]
                edges = d["bands"] * 1e3
                centres = 0.5 * (edges[:-1] + edges[1:])
                net = 1e3 * (d["band_torque_front"] + d["band_torque_back"])
                ax.plot(centres, net, "o-", color=palette[k % len(palette)], label=f"{run['dx'] * 1e3:g} mm net")
                ax.plot(centres, 1e3 * d["band_torque_front"], "^--", ms=4, color=palette[k % len(palette)], alpha=0.6)
                ax.plot(centres, -1e3 * d["band_torque_back"], "v:", ms=4, color=palette[k % len(palette)], alpha=0.6)
            ax.set_xlabel("radial band (centre), mm")
            ax.set_ylabel("resistance torque of the band, mN m (6 blades)")
            ax.set_title(f"{kind}: net (solid), front (dashed, ^), -back (dotted, v)", fontsize=9)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        fig.suptitle(f"torque by radial band from the {source}", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / f"bands_{source}.png", dpi=130)
        plt.close(fig)

        for kind in KINDS:
            available = [run for run in runs if source in run["data"] and kind in run["data"][source]]
            if not available:
                continue
            hs = [run["dx"] for run in available]
            lines.append(f"\n### {source}: {kind}, resistance torque mN m (6 blades)\n")
            header = ("| part | " + " | ".join(f"{run['dx'] * 1e3:g} mm" for run in available)
                      + " | order p (three finest levels, v = v_inf + C h^p) | v_inf |")
            lines.append(header)
            lines.append("| --- | " + " | ".join("---" for _ in available) + " | --- | --- |")
            rows = []
            for face in ("front", "back"):
                rows.append((f"{face} face", [1e3 * float(run["data"][source][kind][f"torque_{face}"]) for run in available]))
            rows.append(("net (front + back)", [1e3 * float(run["data"][source][kind]["torque_front"] + run["data"][source][kind]["torque_back"]) for run in available]))
            if source == "records":
                for face in ("front", "back"):
                    rows.append((f"{face} face, normal force only", [1e3 * float(run["data"][source][kind][f"torque_normal_{face}"]) for run in available]))
            edges = available[0]["data"][source][kind]["bands"] * 1e3
            for i in range(len(edges) - 1):
                for face in ("front", "back", "net"):
                    values = []
                    for run in available:
                        d = run["data"][source][kind]
                        if face == "net":
                            values.append(1e3 * float(d["band_torque_front"][i] + d["band_torque_back"][i]))
                        else:
                            values.append(1e3 * float(d[f"band_torque_{face}"][i]))
                    rows.append((f"band {edges[i]:.1f} to {edges[i + 1]:.1f} mm, {face}", values))
            for name, values in rows:
                p, v_inf = order_estimate(hs, values)
                cell = f"{p:.2f} | {v_inf:.2f}" if np.isfinite(p) else "n/a (not monotone) | n/a"
                lines.append(f"| {name} | " + " | ".join(f"{v:.2f}" for v in values) + f" | {cell} |")
            # mean C_p of the faces and of the edge cells vs interior
            lines.append("")
            lines.append(f"mean (p - p_b) over the face, Pa, and over the outer cells (tip and edges, one cell wide) vs the interior:")
            lines.append("| face | " + " | ".join(f"{run['dx'] * 1e3:g} mm whole / edge / interior" for run in available) + " |")
            lines.append("| --- | " + " | ".join("---" for _ in available) + " |")
            for face in ("front", "back"):
                cells = []
                for run in available:
                    d = run["data"][source][kind]
                    p = d[f"p_{face}"] - run["p_background"]
                    edge = np.ones_like(p, dtype=bool)
                    if p.shape[0] > 2 and p.shape[1] > 2:
                        edge[1:-1, 1:-1] = False
                    cells.append(f"{np.nanmean(p):.0f} / {np.nanmean(p[edge]):.0f} / {np.nanmean(p[~edge]) if (~edge).any() else float('nan'):.0f}")
                lines.append(f"| {face} | " + " | ".join(cells) + " |")

    (out / "tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nfigures and tables in {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("--case", required=True, help="case directory (case.yaml, materials.yaml)")
    a.add_argument("--label", required=True)
    a.add_argument("--spacing", type=float, default=None, help="dx (default: from case.yaml)")
    a.add_argument("--out", required=True)
    a.add_argument("--dump", nargs="*", default=[], help="dump files (globs), averaged")
    a.add_argument("--records", nargs="*", default=[], help="reaction record files (globs), averaged")
    a.add_argument("--records-split", choices=("geometry", "label"), default="geometry",
                   help="plate of a record: by geometry (disk-labelled records in a blade region go to the blade; "
                        "default) or by its label (the parts tables)")
    a.set_defaults(function=analyze)
    p = sub.add_parser("plot")
    p.add_argument("--out", required=True)
    p.add_argument("--labels", nargs="+", required=True)
    p.set_defaults(function=plot)
    arguments = parser.parse_args()
    arguments.function(arguments)


if __name__ == "__main__":
    main()
