"""
flow_statistics.py — time-averaged velocity statistics at fixed points (2026-10-02, Haringa 2023 H1).

Every `every` steps from `start_time` on, the velocity of the live FLUID particles is interpolated to
the sample points of a configuration file (written by utils/geometry/_demo_rushton_tank.py
--flow-statistics) with a Shepard-normalised Wendland C4 kernel of support R:

    u(x) = sum_j u_j W(|x - x_j|; R) / sum_j W(|x - x_j|; R)

and, for the sets marked `gradient`, value and gradient together by a first-order moving least
squares fit with the same weights (exact for a linear field, also where the support is cut by a wall):

    minimise sum_j W_j |u_j - a - B (x_j - x)|^2   ->   u(x) = a,   grad u = B.

The velocity is taken in the cylindrical components of each point (radial, tangential, axial; the
azimuth from +x toward +z, the axis +y through the origin). Accumulated per point, in float64:
the number of samples, sum u, sum u u (rr, tt, yy, rt, ry, ty); with `gradient` also sum S:S,
sum |S|^3 (|S| = sqrt(2 S:S), S the strain rate) and sum S (xx, yy, zz, xy, xz, yz); with `phase`
the same velocity moments per blade-phase bin. The blade phase of a point at azimuth phi is
    psi = (phi - phi_blade0 + theta) mod (2 pi / blades),
theta the rotor angle (positive about +y, the blades move toward -phi): psi = 0 when a blade passes
the point, psi grows with the time since. From the sums the analysis
(experiment/v1/checks/_analyze_flow_statistics.py) forms the mean velocity, the resolved turbulent
kinetic energy (total and without the periodic blade-passage part) and dissipation estimates
2 nu <S:S> and (C_s Delta)^2 <|S|^3> (Smagorinsky form, as LES codes report epsilon).

The accumulators are written to `out_path` (.npz) every `flush_every` samples and at the end;
every flush also keeps a copy <stem>_sNNNNN.npz (NNNNN = samples so far) to judge convergence.
"""

from __future__ import annotations

import itertools
import json
import math
import pathlib

import numpy as np

VELOCITY_COMPONENTS = ("radial", "tangential", "axial")
PRODUCT_PAIRS = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))      # rr, tt, yy, rt, ry, ty
STRAIN_PAIRS = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))       # xx, yy, zz, xy, xz, yz


def wendland_c4(q):
    return (1.0 - q) ** 6 * (35.0 / 3.0 * q * q + 6.0 * q + 1.0)


def pairs_within(points, particles, radius):
    """(point index, particle index) of every pair closer than `radius`: a cell list of cell size
    `radius`, the 27 cells around each point, ragged ranges expanded with numpy."""
    if points.shape[0] == 0 or particles.shape[0] == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    lower = np.minimum(points.min(axis=0), particles.min(axis=0)) - radius
    particle_cell = np.floor((particles - lower) / radius).astype(np.int64)
    point_cell = np.floor((points - lower) / radius).astype(np.int64)
    shape = np.maximum(particle_cell.max(axis=0), point_cell.max(axis=0)) + 2

    def cell_key(cells):
        return (cells[:, 0] * shape[1] + cells[:, 1]) * shape[2] + cells[:, 2]

    order = np.argsort(cell_key(particle_cell), kind="stable")
    sorted_key = cell_key(particle_cell)[order]
    starts, lengths, owners = [], [], []
    point_index = np.arange(points.shape[0], dtype=np.int64)
    for offset in itertools.product((-1, 0, 1), repeat=3):
        key = cell_key(point_cell + np.asarray(offset, dtype=np.int64))
        begin = np.searchsorted(sorted_key, key, side="left")
        end = np.searchsorted(sorted_key, key, side="right")
        starts.append(begin)
        lengths.append(end - begin)
        owners.append(point_index)
    starts, lengths, owners = np.concatenate(starts), np.concatenate(lengths), np.concatenate(owners)
    total = int(lengths.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    exclusive = np.cumsum(lengths) - lengths
    flat = np.repeat(starts - exclusive, lengths) + np.arange(total, dtype=np.int64)
    candidate_point = np.repeat(owners, lengths)
    candidate_particle = order[flat]
    offset = particles[candidate_particle] - points[candidate_point]
    inside = np.einsum("ij,ij->i", offset, offset) < radius * radius
    return candidate_point[inside], candidate_particle[inside]


class PointSet:
    """Sample points of one set and their accumulators."""

    def __init__(self, entry, phase_bins):
        self.name = entry["name"]
        self.points = np.asarray(entry["points"], dtype=np.float64)
        self.gradient = bool(entry.get("gradient", False))
        self.phase = bool(entry.get("phase", False))
        self.extra = {key: value for key, value in entry.items()
                      if key not in ("name", "points", "gradient", "phase")}
        azimuth = np.arctan2(self.points[:, 2], self.points[:, 0])
        self.azimuth = azimuth
        # rows of the cylindrical basis at each point: e_r, e_theta, e_y (azimuth from +x toward +z)
        self.basis = np.zeros((self.points.shape[0], 3, 3))
        self.basis[:, 0, 0], self.basis[:, 0, 2] = np.cos(azimuth), np.sin(azimuth)
        self.basis[:, 1, 0], self.basis[:, 1, 2] = -np.sin(azimuth), np.cos(azimuth)
        self.basis[:, 2, 1] = 1.0
        count = self.points.shape[0]
        self.samples = np.zeros(count)
        self.sum_velocity = np.zeros((count, 3))
        self.sum_product = np.zeros((count, 6))
        if self.gradient:
            self.gradient_samples = np.zeros(count)
            self.sum_strain_squared = np.zeros(count)
            self.sum_strain_cubed = np.zeros(count)
            self.sum_strain = np.zeros((count, 6))
        if self.phase:
            self.phase_bins = phase_bins
            self.phase_samples = np.zeros((count, phase_bins))
            self.phase_velocity = np.zeros((count, phase_bins, 3))
            self.phase_product = np.zeros((count, phase_bins, 6))

    def bounds(self, margin):
        return self.points.min(axis=0) - margin, self.points.max(axis=0) + margin

    def accumulate(self, positions, velocity, radius, phase_angle=None, blades=1):
        """One sample: positions, velocity (n, 3) of the FLUID particles near the set."""
        point, particle = pairs_within(self.points, positions, radius)
        count = self.points.shape[0]
        offset = positions[particle] - self.points[point]                    # x_j - x
        distance = np.sqrt(np.einsum("ij,ij->i", offset, offset))
        q = np.minimum(distance / radius, 1.0)
        weight = wendland_c4(q)
        weight_sum = np.bincount(point, weights=weight, minlength=count)
        valid = weight_sum > 0.0
        mean = np.zeros((count, 3))
        for component in range(3):
            mean[:, component] = np.bincount(point, weights=weight * velocity[particle, component], minlength=count)
        mean[valid] /= weight_sum[valid, None]
        cylindrical = np.einsum("pij,pj->pi", self.basis, mean)
        self.samples[valid] += 1.0
        self.sum_velocity[valid] += cylindrical[valid]
        product = np.stack([cylindrical[:, a] * cylindrical[:, b] for a, b in PRODUCT_PAIRS], axis=1)
        self.sum_product[valid] += product[valid]
        if self.phase and phase_angle is not None:
            period = 2.0 * math.pi / blades
            psi = np.mod(self.azimuth + phase_angle, period)
            bins = np.minimum((psi / period * self.phase_bins).astype(np.int64), self.phase_bins - 1)
            rows = np.nonzero(valid)[0]
            self.phase_samples[rows, bins[rows]] += 1.0
            self.phase_velocity[rows, bins[rows]] += cylindrical[rows]
            self.phase_product[rows, bins[rows]] += product[rows]
        if self.gradient:
            # first-order moving least squares: [1, r]^T W [1, r] (a; B^T) = [1, r]^T W u, r = (x_j - x) / R
            basis = np.column_stack([np.ones(offset.shape[0]), offset / radius])
            moment = np.zeros((count, 4, 4))
            right = np.zeros((count, 4, 3))
            for a in range(4):
                for b in range(a, 4):
                    moment[:, a, b] = np.bincount(point, weights=weight * basis[:, a] * basis[:, b], minlength=count)
                    moment[:, b, a] = moment[:, a, b]
                for c in range(3):
                    right[:, a, c] = np.bincount(point, weights=weight * basis[:, a] * velocity[particle, c],
                                                 minlength=count)
            # well-posed only with neighbours on all sides: reject a near-singular moment matrix
            eigen = np.linalg.eigvalsh(moment)
            usable = valid & (eigen[:, 0] > 1e-4 * np.maximum(eigen[:, -1], 1e-300))
            solution = np.full((count, 4, 3), np.nan)
            solution[usable] = np.linalg.solve(moment[usable], right[usable])
            gradient = np.transpose(solution[:, 1:, :], (0, 2, 1)) / radius        # (p, a, b) = d u_a / d x_b
            strain = 0.5 * (gradient + np.transpose(gradient, (0, 2, 1)))
            strain_squared = np.einsum("pab,pab->p", strain, strain)
            magnitude = np.sqrt(2.0 * strain_squared)
            rows = usable
            self.gradient_samples[rows] += 1.0
            self.sum_strain_squared[rows] += strain_squared[rows]
            self.sum_strain_cubed[rows] += magnitude[rows] ** 3
            self.sum_strain[rows] += np.stack([strain[rows, a, b] for a, b in STRAIN_PAIRS], axis=1)

    def arrays(self):
        prefix = self.name + "/"
        result = {prefix + "points": self.points, prefix + "samples": self.samples,
                  prefix + "sum_velocity": self.sum_velocity, prefix + "sum_product": self.sum_product}
        if self.gradient:
            result.update({prefix + "gradient_samples": self.gradient_samples,
                           prefix + "sum_strain_squared": self.sum_strain_squared,
                           prefix + "sum_strain_cubed": self.sum_strain_cubed,
                           prefix + "sum_strain": self.sum_strain})
        if self.phase:
            result.update({prefix + "phase_samples": self.phase_samples,
                           prefix + "phase_velocity": self.phase_velocity,
                           prefix + "phase_product": self.phase_product})
        return result


class FlowStatisticsSampler:
    """Hooks for the headless runner: due(), sample(), close()."""

    def __init__(self, simulator, config_path, out_path, every, start_time=0.0, flush_every=250):
        config = json.loads(pathlib.Path(config_path).read_text(encoding="utf-8"))
        self.simulator = simulator
        self.config = config
        self.out_path = pathlib.Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.every = int(every)
        self.start_time = float(start_time)
        self.flush_every = int(flush_every)
        self.radius = float(config["kernel_radius"])
        self.blades = int(config.get("blades", 1))
        self.blade_azimuth0 = math.radians(float(config.get("blade_azimuth0_deg", 0.0)))
        self.sets = [PointSet(entry, int(config.get("phase_bins", 30))) for entry in config["sets"]]
        lower = np.min([s.bounds(self.radius)[0] for s in self.sets], axis=0)
        upper = np.max([s.bounds(self.radius)[1] for s in self.sets], axis=0)
        self.box = (lower, upper)
        self.samples = 0
        self.first_time = None
        self.last_time = None
        self.fluid_groups = np.asarray(simulator.fluid_group_ids(), dtype=np.uint32)
        print(f"[flow-statistics] {sum(s.points.shape[0] for s in self.sets):,} points in {len(self.sets)} sets, "
              f"kernel radius {self.radius:.4g} m, every {self.every} steps from t = {self.start_time:g} s -> {self.out_path}")

    def due(self) -> bool:
        simulator = self.simulator
        return (simulator.step_count % self.every == 0
                and simulator.simulation_time >= self.start_time - 0.5 * simulator.case.timestep)

    def sample(self) -> None:
        simulator = self.simulator
        positions = simulator.readback_positions()
        live = simulator.live_slot_mask(positions)
        material = simulator.readback_material()
        fluid = live & np.isin(material, self.fluid_groups)
        xyz = positions[:, :3]
        inside = fluid & np.all((xyz >= self.box[0]) & (xyz <= self.box[1]), axis=1)
        particle_positions = xyz[inside].astype(np.float64)
        particle_velocity = simulator.readback_velocity_mass()[inside, :3].astype(np.float64)
        # phase: blade k sits at azimuth phi_k0 - theta; psi = phi - phi_blade0 + theta (mod period)
        phase_angle = float(simulator.rotor_angle) - self.blade_azimuth0
        for point_set in self.sets:
            lower, upper = point_set.bounds(self.radius)
            near = np.all((particle_positions >= lower) & (particle_positions <= upper), axis=1)
            point_set.accumulate(particle_positions[near], particle_velocity[near], self.radius,
                                 phase_angle=phase_angle, blades=self.blades)
        self.samples += 1
        if self.first_time is None:
            self.first_time = float(simulator.simulation_time)
        self.last_time = float(simulator.simulation_time)
        if self.samples % self.flush_every == 0:
            self.flush(keep_copy=True)

    def flush(self, keep_copy=False) -> None:
        arrays = {}
        for point_set in self.sets:
            arrays.update(point_set.arrays())
        meta = dict(samples=self.samples, first_time=self.first_time, last_time=self.last_time, every=self.every,
                    timestep=float(self.simulator.case.timestep), kernel_radius=self.radius, blades=self.blades,
                    blade_azimuth0_deg=math.degrees(self.blade_azimuth0),
                    viscosity=float(self.config.get("viscosity", 1.0e-6)),
                    sets=[dict(name=s.name, gradient=s.gradient, phase=s.phase, **s.extra) for s in self.sets])
        arrays["meta"] = np.array(json.dumps(meta))
        np.savez(self.out_path, **arrays)
        if keep_copy:
            np.savez(self.out_path.with_name(f"{self.out_path.stem}_s{self.samples:05d}.npz"), **arrays)

    def close(self) -> None:
        self.flush(keep_copy=False)
        print(f"[flow-statistics] {self.samples} samples, t = {self.first_time} .. {self.last_time} s -> {self.out_path}")
