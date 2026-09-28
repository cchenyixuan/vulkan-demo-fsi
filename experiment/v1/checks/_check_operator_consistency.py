"""_check_operator_consistency.py — how exact are the SPH operators on a regular lattice? (2026-09-27)

Evaluates, at an interior particle of an ideal simple cubic (3D) / square (2D)
lattice with the calibrated particle volume of utils/sph/case.py, the discrete
operators used by the V1 kernels on fields whose exact result is known:

  pressure gradient   -(1/rho) grad P  with the symmetric sum  sum_j V_j (P_i + P_j) gradW~_ij,  P = x   (exact 1)
  Morris viscosity    sum_j V_j 2(d+2) (v_i - v_j).x_ij / (r^2 + eta^2) gradW~_ij,  v = (y^2, 0, 0)  (exact lap v_x = 2)
  delta-diffusion /   sum_j V_j 2 (f_j - f_i) x_ji.gradW~_ij / (r^2 + eta^2),  f = |x|^2           (exact 2 d)
  scalar Brookshaw

for the gradient variants gradW~ = (M + xi I)^-1 gradW (what correction.comp stores; xi = 0.01 since 2026-09-28, 0.1 before),
M^-1 gradW (unregularized) and gradW (raw), and eta^2 = 0.01 h^2 (EPS_H_SQUARED, h = support
radius), 0.01 dx^2 and 0. M = sum_j V_j (x_j - x_i) (x) gradW_ij. Results are printed as the
ratio to the exact value. The scalar transport of 2026-09-27 uses the trace normalisation
lambda = d / tr M instead of M^-1 (identical on these isotropic lattices) with eta^2 = 0.01 dx^2.

Usage (repo root):
    python experiment/v1/checks/_check_operator_consistency.py [--xi 0.1]
See log/2026-09-27_scalar-transport-review.md sections 2 and 4.
"""
import argparse
import math
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.sph.case import _calibrate_particle_volume      # noqa: E402


def lattice_neighbours(dimension: int, h: float, dx: float) -> np.ndarray:
    n = int(math.ceil(h / dx)) + 1
    axis = np.arange(-n, n + 1) * dx
    if dimension == 3:
        points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
    else:
        points = np.stack(np.meshgrid(axis, axis, indexing="ij"), -1).reshape(-1, 2)
        points = np.column_stack([points, np.zeros(len(points))])
    distance = np.linalg.norm(points, axis=1)
    return points[(distance > 0) & (distance < h)]


def evaluate(dimension: int, h_over_dx: float, xi: float) -> None:
    dx = 1.0
    h = h_over_dx * dx
    volume = _calibrate_particle_volume(h, 0.5 * dx, dimension, "grid")
    coefficient = 9.0 / (math.pi * h * h) if dimension == 2 else 495.0 / (32.0 * math.pi * h ** 3)
    gradient_coefficient = coefficient / h

    def kernel_derivative(r):
        q = r / h
        return gradient_coefficient * (1 - q) ** 5 * q * (-280.0 / 3.0 * q - 56.0 / 3.0)

    X = lattice_neighbours(dimension, h, dx)               # x_j - x_i with x_i = 0
    r = np.linalg.norm(X, axis=1)
    x_ij = -X                                              # x_i - x_j
    grad = kernel_derivative(r)[:, None] * x_ij / r[:, None]   # grad_i W_ij
    M = (volume * (X[:, :, None] * grad[:, None, :])).sum(axis=0)
    regularised = M.copy()
    exact = M.copy()
    for axis_index in range(dimension):
        regularised[axis_index, axis_index] += xi
    if dimension == 2:
        regularised[2, 2] = 1.0
        exact[2, 2] = 1.0
    variants = {"(M + xi I)^-1  [correction.comp]": np.linalg.inv(regularised),
                "M^-1           [unregularized]": np.linalg.inv(exact),
                "I              [raw gradW]": np.eye(3)}
    epsilons = {"0.01 h^2 [EPS_H_SQUARED]": 0.01 * h * h, "0.01 dx^2": 0.01 * dx * dx, "0": 0.0}

    print(f"\n=== {dimension}D, h/dx = {h_over_dx:g}: V = {volume / dx ** dimension:.4f} dx^{dimension}, "
          f"M_xx = {M[0, 0]:.4f}, (M_xx + xi)^-1 / M_xx^-1 = {M[0, 0] / (M[0, 0] + xi):.4f}")
    print("ratio to the exact result (1 = exact)")
    for label, inverse in variants.items():
        corrected = grad @ inverse.T
        pressure = (volume * (X @ np.array([1.0, 0.0, 0.0]))[:, None] * corrected).sum(axis=0)[0]
        print(f"  {label}  pressure gradient (P = x): {pressure:.4f}")
        for eps_label, eps in epsilons.items():
            v_j = np.column_stack([X[:, 1] ** 2, np.zeros(len(X)), np.zeros(len(X))])
            viscous = ((volume * 2 * (dimension + 2) * ((-v_j) * x_ij).sum(axis=1) / (r ** 2 + eps))[:, None]
                       * corrected).sum(axis=0)[0] / 2.0
            laplacian = (volume * 2 * r ** 2 * ((-x_ij) * corrected).sum(axis=1) / (r ** 2 + eps)).sum() / (2 * dimension)
            print(f"      eta^2 = {eps_label:26s} Morris viscosity {viscous:.4f}   "
                  f"delta-diffusion / Brookshaw {laplacian:.4f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--xi", type=float, default=0.01)
    args = parser.parse_args()
    evaluate(3, 3.0, args.xi)       # 30 L tank cases (h/dx = 3)
    evaluate(2, 5.0, args.xi)       # 2D cavity / Taylor-Couette cases (h/dx = 5)


if __name__ == "__main__":
    main()
