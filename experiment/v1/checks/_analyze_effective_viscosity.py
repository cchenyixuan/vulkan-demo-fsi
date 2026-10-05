"""Effective viscosity of a tank dump from the energy budget: the input power P (rotor torque x omega, from the CSV
window) is dissipated by the fluid; with the resolved strain rate S of the kernel-smoothed velocity v~,
    eps_resolved = 2 nu <S:S> V,   nu_eff = P / (rho sum_i V_i 2 S_i:S_i)
gives the viscosity the resolved field would need to dissipate P. Also the Smagorinsky nu_t the model would
give for Delta = dx, 2dx, h at C_s 0.2 on the same S, volume-averaged and in the two rotor boxes.
Strain rate: SPH gradient of v~ with a renormalised (KCG-like) matrix, Wendland C4, support h, scipy KD-tree.
usage: python effective_viscosity.py DUMP.npz DX TORQUE_mNm"""
import sys

import numpy as np
from scipy.spatial import cKDTree

path, dx, torque = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]) * 1e-3
support = 3.0 * dx
rho = 998.0
omega = 2.0 * np.pi * 200.0 / 60.0
d = np.load(path)
live = d["positions"][:, 3] > 0
fluid = live & (d["material"] == 0)
x_all = d["positions"][live, :3].astype(np.float64)
v_all = d["velocity_mass"][live, :3].astype(np.float64)
is_fluid = (d["material"][live] == 0)
tree = cKDTree(x_all)


def kernel(q):
    return np.where(q < 1.0, (1.0 - q) ** 6 * (35.0 / 3.0 * q * q + 6.0 * q + 1.0), 0.0)


def kernel_gradient_factor(q):
    # dW/dq / q for the Wendland C4 shape (1-q)^6 (35/3 q^2 + 6 q + 1): dW/dq = -(56/3) q (1-q)^5 (5 q + 1)
    return np.where(q < 1.0, -(56.0 / 3.0) * (1.0 - q) ** 5 * (5.0 * q + 1.0), 0.0)


# pass 1: Shepard-smoothed velocity of every live particle (solids keep their own velocity: no-slip walls)
smoothed = v_all.copy()
fluid_index = np.nonzero(is_fluid)[0]
for start in range(0, fluid_index.size, 30000):
    block = fluid_index[start:start + 30000]
    neighbours = tree.query_ball_point(x_all[block], support)
    lengths = np.fromiter((len(n) for n in neighbours), dtype=np.int64, count=block.size)
    i = np.repeat(np.arange(block.size), lengths)
    j = np.concatenate([np.asarray(n, dtype=np.int64) for n in neighbours])
    q = np.linalg.norm(x_all[j] - x_all[block][i], axis=1) / support
    w = kernel(q)
    total = np.bincount(i, weights=w, minlength=block.size)
    for a in range(3):
        smoothed[block, a] = np.bincount(i, weights=w * v_all[j, a], minlength=block.size) / total

# pass 2: renormalised gradient of v~ at the fluid particles
volume = dx ** 3
ss = np.zeros(fluid_index.size)            # S:S per fluid particle
for start in range(0, fluid_index.size, 20000):
    block = fluid_index[start:start + 20000]
    neighbours = tree.query_ball_point(x_all[block], support)
    lengths = np.fromiter((len(n) for n in neighbours), dtype=np.int64, count=block.size)
    i = np.repeat(np.arange(block.size), lengths)
    j = np.concatenate([np.asarray(n, dtype=np.int64) for n in neighbours])
    r = x_all[j] - x_all[block][i]
    dist = np.linalg.norm(r, axis=1)
    q = dist / support
    grad = r * (kernel_gradient_factor(q) / (support ** 2))[:, None] * volume     # V grad W (direction r_ij / |r| * dW/dr)
    # renormalisation matrix M = sum_j r_ji (x) V grad W ; gradient = M^-1 sum_j (v_j - v_i) (x) V grad W
    M = np.zeros((block.size, 3, 3))
    G = np.zeros((block.size, 3, 3))
    dv = smoothed[j] - smoothed[block][i]
    for a in range(3):
        for b in range(3):
            M[:, a, b] = np.bincount(i, weights=r[:, a] * grad[:, b], minlength=block.size)
            G[:, a, b] = np.bincount(i, weights=dv[:, a] * grad[:, b], minlength=block.size)
    Minv = np.linalg.inv(M + 1e-3 * np.eye(3)[None] * np.abs(np.trace(M, axis1=1, axis2=2))[:, None, None] / 3.0)
    gradv = np.einsum("nab,ncb->nac", G, Minv)          # dv_a/dx_c
    S = 0.5 * (gradv + np.transpose(gradv, (0, 2, 1)))
    ss[start:start + block.size] = np.einsum("nab,nab->n", S, S)

power = torque * omega
x_f = x_all[fluid_index]
r_f = np.hypot(x_f[:, 0], x_f[:, 2])
rushton = (r_f < 0.072) & (x_f[:, 1] > 0.0187) & (x_f[:, 1] < 0.0585)
pbt = (r_f < 0.072) & (x_f[:, 1] > 0.165) & (x_f[:, 1] < 0.225)
bulk = ~(rushton | pbt)
eps_coefficient = 2.0 * rho * volume * ss.sum()            # eps = nu * this
nu_eff = power / eps_coefficient
print(f"{path}: {fluid_index.size:,} fluid particles, dx {dx*1e3:g} mm, support {support*1e3:g} mm")
print(f"input power {power:.3f} W; resolved 2 S:S volume integral {eps_coefficient / rho:.4e} m^3/s^2")
print(f"nu_eff = P / (rho int 2 S:S) = {nu_eff:.3e} m^2/s = {nu_eff / 1e-6:.0f} x molecular")
print(f"dissipation of the resolved field with molecular nu: {1e-6 * eps_coefficient:.4f} W ({100 * 1e-6 * eps_coefficient / power:.1f} % of P)")
Sabs = np.sqrt(2.0 * ss)
for label, m in (("whole", np.ones_like(bulk)), ("Rushton box", rushton), ("PBT box", pbt), ("bulk", bulk)):
    print(f"  {label:12s} |S| mean {Sabs[m].mean():7.1f} 1/s, rms {np.sqrt((Sabs[m]**2).mean()):7.1f}; share of 2S:S {ss[m].sum() / ss.sum():.2f}; "
          + "  ".join(f"nu_t(C_s 0.2, Delta {name}) = {((0.2 * delta) ** 2 * Sabs[m]).mean() / 1e-6:6.1f} x nu" for name, delta in (("dx", dx), ("2dx", 2 * dx), ("h", support))))
