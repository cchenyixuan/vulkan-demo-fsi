"""
_check_flow_statistics.py — checks of experiment/v1/utils/flow_statistics.py on synthetic particle
fields, no GPU (2026-10-02).

    F1  linear field u = b + A x on a jittered lattice: the moving least squares gradient equals A
        to rounding, inside the cloud and at points whose support is cut by a plane "wall"; the
        Shepard mean is within the jitter error; S:S and |S|^3 as from A.
    F2  blade-periodic field u_r = U0 + U1 cos(blades (phi + theta)) sampled at many rotor angles:
        the phase-binned means reproduce the periodic part, the total variance is U1^2 / 2 and the
        variance left inside the bins is the bin-width error only.
    F3  pairs_within against brute force.

usage (repo root): python experiment/v1/checks/_check_flow_statistics.py
"""
import json
import math
import pathlib
import sys
import tempfile

import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiment.v1.utils.flow_statistics import FlowStatisticsSampler, PointSet, pairs_within   # noqa: E402


class FakeCase:
    timestep = 1.0e-3


class FakeSimulator:
    """The part of SphSimulatorV1 the sampler reads."""

    def __init__(self, positions, velocity):
        self.case = FakeCase()
        self.step_count = 0
        self.simulation_time = 0.0
        self.rotor_angle = 0.0
        self.set_state(positions, velocity)

    def set_state(self, positions, velocity):
        count = positions.shape[0]
        self.positions = np.zeros((count + 1, 4), dtype=np.float32)
        self.positions[1:, :3] = positions
        self.positions[1:, 3] = 1.0
        self.velocity = np.zeros((count + 1, 4), dtype=np.float32)
        self.velocity[1:, :3] = velocity
        self.material = np.zeros(count + 1, dtype=np.uint32)

    def readback_positions(self):
        return self.positions

    def readback_velocity_mass(self):
        return self.velocity

    def readback_material(self):
        return self.material

    def live_slot_mask(self, positions):
        mask = positions[:, 3] > 0
        mask[0] = False
        return mask

    def fluid_group_ids(self):
        return [0]


def lattice(spacing, half, jitter, rng):
    axis = np.arange(-half, half + 0.5 * spacing, spacing)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    return grid + rng.uniform(-jitter, jitter, grid.shape)


def check_linear(rng):
    spacing, radius = 0.002, 0.004
    particles = lattice(spacing, 0.03, 0.15 * spacing, rng)
    wall = particles[:, 1] > -0.01                       # particles only above y = -0.01: a cut support
    particles = particles[wall]
    gradient = rng.normal(0.0, 5.0, (3, 3))
    offset_velocity = np.array([0.3, -0.2, 0.1])
    velocity = offset_velocity + particles @ gradient.T
    points = np.array([[0.0, 0.0, 0.0], [0.005, 0.004, -0.003], [0.0, -0.01, 0.0], [0.01, -0.009, 0.004]])
    point_set = PointSet(dict(name="test", points=points.tolist(), gradient=True, phase=False), 30)
    point_set.accumulate(particles, velocity, radius)
    strain = 0.5 * (gradient + gradient.T)
    expected_squared = float(np.sum(strain * strain))
    measured_squared = point_set.sum_strain_squared / point_set.gradient_samples
    error_squared = np.max(np.abs(measured_squared / expected_squared - 1.0))
    expected_cubed = math.sqrt(2.0 * expected_squared) ** 3
    error_cubed = np.max(np.abs(point_set.sum_strain_cubed / point_set.gradient_samples / expected_cubed - 1.0))
    exact_mean = offset_velocity + points @ gradient.T
    basis = point_set.basis
    exact_cylindrical = np.einsum("pij,pj->pi", basis, exact_mean)
    mean_error = np.max(np.abs(point_set.sum_velocity - exact_cylindrical)) / np.max(np.abs(exact_cylindrical))
    print(f"F1 linear field: S:S relative error {error_squared:.2e}, |S|^3 {error_cubed:.2e} "
          f"(4 points, 2 of them 0 and 1 mm from the cut), Shepard mean relative error {mean_error:.2e}")
    return error_squared < 1e-9 and error_cubed < 1e-9 and mean_error < 0.05


def check_phase(rng):
    blades, bins = 6, 30
    spacing = 0.002
    particles = lattice(spacing, 0.02, 0.0, rng)
    particles = particles[np.hypot(particles[:, 0], particles[:, 2]) > 0.004]
    phi = np.arctan2(particles[:, 2], particles[:, 0])
    e_r = np.column_stack([np.cos(phi), np.zeros_like(phi), np.sin(phi)])
    points = np.array([[0.012 * math.cos(a), 0.0, 0.012 * math.sin(a)] for a in np.radians([10.0, 47.0, 200.0])])
    config = dict(kernel_radius=0.0025, blades=blades, blade_azimuth0_deg=0.0, phase_bins=bins, viscosity=1e-6,
                  sets=[dict(name="ring", points=points.tolist(), gradient=False, phase=True)])
    directory = pathlib.Path(tempfile.mkdtemp())
    (directory / "config.json").write_text(json.dumps(config))
    simulator = FakeSimulator(particles, np.zeros_like(particles))
    sampler = FlowStatisticsSampler(simulator, directory / "config.json", directory / "out.npz", every=1, flush_every=10 ** 9)
    base, amplitude = 0.5, 0.2
    for theta in rng.uniform(0.0, 2.0 * math.pi, 3000):
        simulator.rotor_angle = theta
        u_r = base + amplitude * np.cos(blades * (phi + theta))
        simulator.velocity[1:, :3] = (u_r[:, None] * e_r).astype(np.float32)
        simulator.step_count += 1
        sampler.sample()
    ring = sampler.sets[0]
    mean = ring.sum_velocity[:, 0] / ring.samples
    variance = ring.sum_product[:, 0] / ring.samples - mean ** 2
    phase_mean = ring.phase_velocity[:, :, 0] / ring.phase_samples
    phase_variance = ring.phase_product[:, :, 0] / ring.phase_samples - phase_mean ** 2
    within = np.sum(ring.phase_samples * phase_variance, axis=1) / ring.samples
    centres = (np.arange(bins) + 0.5) * (2.0 * math.pi / blades / bins)
    # Shepard averaging over the kernel smears the azimuthal cosine a little: compare the shape
    expected = np.cos(blades * centres)
    fitted = [np.polyfit(expected, phase_mean[p] - mean[p], 1)[0] for p in range(points.shape[0])]
    correlation = min(np.corrcoef(expected, phase_mean[p])[0, 1] for p in range(points.shape[0]))
    bin_error = (amplitude * blades * (2.0 * math.pi / blades / bins)) ** 2 / 12.0 / 2.0
    print(f"F2 blade-periodic field: mean {mean.round(4)} (expected {base}), total variance "
          f"{variance.round(5)} (expected <= {amplitude ** 2 / 2:.5f}), amplitude ratio {np.round(fitted, 3)}, "
          f"phase correlation {correlation:.5f}, variance inside the bins {within.round(6)} "
          f"(bin-width error ~{bin_error:.6f})")
    # the Shepard kernel and the fixed particle positions smear the azimuthal cosine by a few per cent
    return (np.all(np.abs(mean - base) < 0.01) and correlation > 0.99
            and np.all(np.abs(np.asarray(fitted) / amplitude - 1.0) < 0.1)
            and np.all(within < 3.0 * bin_error + 1e-6))


def check_pairs(rng):
    particles = rng.uniform(-1.0, 1.0, (4000, 3))
    points = rng.uniform(-1.2, 1.2, (300, 3))
    radius = 0.17
    point, particle = pairs_within(points, particles, radius)
    found = set(zip(point.tolist(), particle.tolist()))
    difference = points[:, None, :] - particles[None, :, :]
    brute = np.argwhere(np.einsum("ijk,ijk->ij", difference, difference) < radius * radius)
    expected = set(map(tuple, brute.tolist()))
    print(f"F3 pairs_within: {len(found)} pairs, brute force {len(expected)}, equal {found == expected}")
    return found == expected


def main():
    rng = np.random.default_rng(7)
    results = [check_linear(rng), check_phase(rng), check_pairs(rng)]
    print("all passed" if all(results) else f"FAILED: {results}")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
