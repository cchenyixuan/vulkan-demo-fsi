"""Lei, Rotboll, Jorgensen 2001 (J. Biotechnol. 88:205) biochemically structured S. cerevisiae model, 0-D.
States: s_glu, s_pyr, s_acetald, s_acetate, s_EtOH (g/L), x (g/L), X_a, X_Acdh (g/g). Rates r_i in 1/h (Table 5),
mass balances Table 6, parameters Table 7. Reproduces Fig. 3 (chemostat, S_f 15 g/L) and Fig. 5 (batch 15 g/L).
usage: python experiment/v1/checks/_model_lei2001.py OUT.png   (base Anaconda: scipy, matplotlib)
Reference implementation for the stage-5 level-2 model choice (2026-10-06); transcription validated against the
paper: D_crit 0.38 1/h, washout ~0.48, ethanol ~4 g/L at D 0.45, batch glucose gone at 18 h, ethanol peak 5.4 g/L."""
import sys

import numpy as np
from scipy.integrate import solve_ivp

P = dict(k1h=0.584, K1h=0.0116, k1l=1.43, K1l=0.94, k1e=47.1, K1e=0.12, K1i=14.2,
         k2=0.501, K2=2.0e-5, K2i=0.101, k3=5.81, K3=5.0e-7, k4=4.80, K4=2.64e-4,
         k5=0.0104, K5=0.0102, k5e=0.775, K5e=0.10, K5i=440.0, k6=2.82, K6=0.034, k6r=0.0125, K6e=0.057,
         k7=1.203, K7=0.0101, k8=0.589, k9=0.008, K9=1.0e-6, k9e=0.0751, K9i=13.0, K9e=25.0, k9c=3.99e-3,
         k10=0.392, K10=2.3e-3, k10e=3.39e-3, K10e=1.8e-3, k11=0.02)


def rates(state, p=P):
    glu, pyr, ald, ace, eth, x, xa, xacdh = [max(v, 0.0) for v in state]
    r1 = (p["k1l"] * glu / (glu + p["K1l"]) + p["k1h"] * glu / (glu + p["K1h"])
          + p["k1e"] * glu / (glu * (p["K1i"] * ald + 1.0) + p["K1e"]) * ald) * xa
    r2 = p["k2"] * pyr / (pyr + p["K2"]) / (p["K2i"] * glu + 1.0) * xa
    r3 = p["k3"] * pyr ** 4 / (pyr ** 4 + p["K3"]) * xa
    r4 = p["k4"] * ald / (ald + p["K4"]) * xa * xacdh
    r5 = (p["k5"] * ace / (ace + p["K5"]) + p["k5e"] * ace / (ace + p["K5e"]) / (1.0 + p["K5i"] * glu)) * xa
    r6 = p["k6"] * (ald - p["k6r"] * eth) / (ald + p["K6"] + p["K6e"] * eth) * xa
    r7 = p["k7"] * glu / (glu + p["K7"]) * xa
    r8 = p["k8"] * ace / (ace + p["K5e"]) / (1.0 + p["K5i"] * glu) * xa
    r9 = ((p["k9"] * glu / (glu + p["K9"]) + p["k9e"] * eth / (eth + p["K9e"])) / (p["K9i"] * glu + 1.0)
          + p["k9c"] * glu / (glu + p["K9"])) * xa
    r10 = (p["k10"] * glu / (glu + p["K10"]) + p["k10e"] * eth / (eth + p["K10e"])) * xa
    r11 = p["k11"] * xacdh
    return np.array([r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11])


def rhs(t, state, D, s_f):
    glu, pyr, ald, ace, eth, x, xa, xacdh = state
    r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11 = rates(state)
    mu = 0.732 * r7 + 0.619 * r8
    return [-(r1 + r7) * x + (s_f - glu) * D,
            (0.978 * r1 - r2 - r3) * x - pyr * D,
            (0.5 * r3 - r4 - r6) * x - ald * D,
            (1.363 * r4 - r5 - r8) * x - ace * D,
            1.045 * r6 * x - eth * D,
            (mu - D) * x,
            mu - r9 - r10 - mu * xa,
            r9 - r11 - mu * xacdh]


def gas(state):
    r = rates(state)
    q_o2 = 1000.0 / 32.0 * (0.178 * r[0] + 0.908 * r[1] + 0.363 * r[3] + 1.066 * r[4] - 0.363 * r[5] + 0.063 * r[6] + 0.214 * r[7])
    q_co2 = 1000.0 / 44.01 * (1.499 * r[1] + 0.5 * r[2] + 1.466 * r[4] + 0.127 * r[6] + 0.325 * r[7])
    return q_o2, q_co2


def chemostat(D, s_f=15.0, hours=600.0, start=None):
    y0 = start if start is not None else [0.05, 0.01, 0.01, 0.05, 0.1, 7.0, 0.3, 0.02]
    sol = solve_ivp(rhs, (0.0, hours), y0, args=(D, s_f), method="LSODA", rtol=1e-8, atol=1e-12)
    return sol.y[:, -1]


def batch(hours=35.0, s0=15.0):
    y0 = [s0, 0.0, 0.0, 0.0, 0.0, 0.002, 0.1, 0.0075]
    sol = solve_ivp(rhs, (0.0, hours), y0, args=(0.0, 0.0), method="LSODA", rtol=1e-8, atol=1e-12,
                    t_eval=np.linspace(0.0, hours, 351))
    return sol


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = sys.argv[1]
    dilution = np.concatenate([np.linspace(0.1, 0.34, 7), np.linspace(0.35, 0.47, 13)])
    states = []
    previous = None
    for D in dilution:                                  # continuation from low to high D (lower branch)
        previous = chemostat(D, start=previous)
        states.append(previous.copy())
    states = np.array(states)
    gases = np.array([gas(s) for s in states])
    print("D      glu     pyr    acetald  acetate  EtOH    x      Xa     XAcdh   qO2   qCO2")
    for D, s, g in zip(dilution, states, gases):
        print(f"{D:5.3f} {s[0]:7.4f} {s[1]:7.4f} {s[2]:7.4f} {s[3]:7.4f} {s[4]:6.3f} {s[5]:6.3f} {s[6]:6.3f} {s[7]:7.4f} {g[0]:5.1f} {g[1]:5.1f}")
    b = batch()
    cer = np.array([gas(b.y[:, k])[1] * b.y[5, k] for k in range(b.t.size)])
    print("batch: glucose gone at t = %.1f h; ethanol max %.2f g/L at %.1f h; biomass end %.2f g/L" % (
        b.t[np.argmax(b.y[0] < 0.05)], b.y[4].max(), b.t[np.argmax(b.y[4])], b.y[5, -1]))
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ax, k, name, ylim in zip(axes[0], (0, 5, 4, None), ("glucose g/L", "biomass g/L", "ethanol g/L", "qO2 / qCO2 mmol/g/h"), ((0, 0.3), (0, 8), (0, 5), (0, 30))):
        if k is None:
            ax.plot(dilution, gases[:, 0], "o-", label="qO2"); ax.plot(dilution, gases[:, 1], "v-", label="qCO2"); ax.legend()
        else:
            ax.plot(dilution, states[:, k], "o-")
        ax.set_title("chemostat: " + name); ax.set_xlabel("D, 1/h"); ax.set_xlim(0.1, 0.47); ax.set_ylim(*ylim)
    for ax, k, name in zip(axes[1], (0, 4, 5, None), ("glucose g/L", "ethanol g/L", "biomass g/L", "CER mmol/L/h")):
        ax.plot(b.t, cer if k is None else b.y[k]); ax.set_title("batch: " + name); ax.set_xlabel("t, h")
    fig.tight_layout(); fig.savefig(out, dpi=110)
    print("wrote", out)
