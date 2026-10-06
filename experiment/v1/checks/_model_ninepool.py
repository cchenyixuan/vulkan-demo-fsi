"""Tang et al. 2017 9-pool model of P. chrysogenum, as summarised in Haringa et al. 2018 supplementary A
(tables A1-A3, pool balance eq. 1, ATP quasi-steady patch), 0-D.
Pools (umol/gdw): X_gly, X_AA, X_sto, X_PAA (+ X_ATP algebraic or dynamic); enzyme pools (-): X_E11, X_E32, X_E4;
q_p capacity v33 (mol/Cmol/h). Extracellular: C_s, C_PAA (mol/kg), C_x (gdw/kg). Rates v in mol/Cmol_x/h.
usage: python experiment/v1/checks/_model_ninepool.py OUT.png   (base Anaconda: scipy, matplotlib)
Reference implementation for the stage-5 level-2 model (2026-10-06). Three departures from the printed tables, each
needed to reproduce the paper: v31 sign (import driven by the extracellular excess), pH_int 7.20 (printed 2.20),
q_E11,max 1.65e-2 (printed 6.5e-2; calibrated to the stated q_s,max). D <= 0.01 1/h stalls the integrator (not needed)."""
import sys

import numpy as np
# scipy only for run() (the solver environment has none; _check_ninepool_box.py uses rhs() with its own RK4)

# qE11max: table A3 prints 6.5e-2; 1.65e-2 reproduces the paper's q_s,max 1.13 mmol/gdw/h at mu 0.033 and C_s ~1e-5 at D 0.05
P = dict(qE11max=1.65e-2, mu0=5.5e-2, k11=0.10, kdE11=1.46e-2, kE11=0.26, Ks11=9.8e-6,
         v12max=0.18, Kgly12=31.38, KAA12=870.23, KATP12=2.01,
         v13max=0.32, Kgly13=38.54, KAA13=757.81, KATP13=1.95,
         v21max=0.35, Kgly21=25.64, KATP21=6.01, mATP22=3.3e-2,
         kperm31=1.62e-2, acell=56.0, alpha32=0.0, beta32=1.56e3, kdE32=0.35,
         beta33=6.5e-4, kdE33=1.47e-2, Kgly33=30.76, m33=6.0,
         alpha4=8.01e-4, beta4=0.289, kdE4=0.29,
         k41=1.01, Ks41=1e-8, Ksto41=4.25e3, k42=3.99, Ks42=1e-4, Ksto42=7.99e3, KATP42=6.48,
         pHint=7.20, pHext=6.50, pKPAA=4.31, vd=5e-3, Mw=28.05, rho=1000.0)
# stoichiometry (rows gly, AA, sto, ATP, PAA; columns v11 v12 v13 v21 v22 v31 v32 v33 v41 v42)
S = np.array([[6, -1, -0.578, -1, 0, 0, 0, -4.81, -1.07, 1],
              [0, 1, -0.5, 0, 0, 0, 0, -6.25, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 1, -1],
              [-2, -0.65, -1.037, 4.43, -1, 0, -2, -8, -0.167, -0.167],
              [0, 0, 0, 0, 0, 1, -1, -1, 0, 0]], dtype=float)
ATP_A, ATP_B = 8.5, 10.5           # X_ATP = A X_gly^3 / (X_gly^3 + B^3) (Haringa 2018 supp. A, used values)


def hill(x, k, n):
    return x ** n / (k ** n + x ** n)


def rates(X, Cs, CPAA, p=P, atp_dynamic=False):
    gly, aa, sto, paa, e11, e32, e4, v33 = X[:8]
    atp = X[8] if atp_dynamic else ATP_A * gly ** 3 / (gly ** 3 + ATP_B ** 3)
    gly, aa, sto, paa, atp = (max(v, 0.0) for v in (gly, aa, sto, paa, atp))
    Cs = max(Cs, 0.0)
    v11 = p["kE11"] * e11 * Cs / (Cs + p["Ks11"])
    v12 = p["v12max"] * hill(gly, p["Kgly12"], 2) * (1.0 - hill(aa, p["KAA12"], 2)) * hill(atp, p["KATP12"], 3)
    v13 = p["v13max"] * hill(gly, p["Kgly13"], 2) * hill(aa, p["KAA13"], 2) * hill(atp, p["KATP13"], 3)
    v21 = p["v21max"] * hill(gly, p["Kgly21"], 3) * (1.0 - hill(atp, p["KATP21"], 4))
    v22 = p["mATP22"]
    # passive import of undissociated PAA, driven by the extracellular excess (sign as in Deshmukh 2015;
    # table A2 prints the two terms the other way round, which makes the pool run away)
    v31 = p["kperm31"] * p["acell"] * (CPAA * p["rho"] / (1.0 + 10.0 ** (p["pHext"] - p["pKPAA"]))
                                       - paa / 2.5 / (1.0 + 10.0 ** (p["pHint"] - p["pKPAA"])))
    v32 = e32 * paa * p["Mw"] * 1e-6
    v41 = p["k41"] * e4 * Cs / (Cs + p["Ks41"]) * (1.0 + 2.0 * Cs / (Cs + p["Ks42"])) * p["Ksto41"] / (sto + p["Ksto41"])
    v42 = (p["k42"] * e4 * p["Ks42"] / (Cs + p["Ks42"]) * (1.0 + 2.0 * Cs / (Cs + p["Ks41"]))
           * hill(sto, p["Ksto42"], 2) * hill(atp, p["KATP42"], 2))
    return np.array([v11, v12, v13, v21, v22, v31, v32, v33, v41, v42]), atp


def rhs(t, y, D, Cs_f, CPAA_f, feed_on, p=P, atp_dynamic=False):
    X = y[:9]; Cs, CPAA, Cx = y[9], y[10], y[11]
    v, atp = rates(X, Cs, CPAA, p, atp_dynamic)
    mu = v[2]
    # S rows: gly, AA, sto, ATP, PAA (table A1 order); state order gly, aa, sto, paa, ..., atp
    pools = 1e6 / p["Mw"] * (S @ v) - mu * np.array([X[0], X[1], X[2], atp, X[3]])
    gly, aa, sto, paa, e11, e32, e4, v33 = X[:8]
    de11 = p["qE11max"] * hill(mu + p["mu0"], p["k11"], 5) - (mu + p["kdE11"]) * e11
    de32 = p["alpha32"] + p["beta32"] * mu - p["kdE32"] * e32 - mu * e32
    de4 = p["alpha4"] + p["beta4"] * mu - p["kdE4"] * e4 - mu * e4
    dv33 = p["beta33"] * mu / (1.0 + (gly / p["Kgly33"]) ** p["m33"]) - (p["kdE33"] + mu) * v33
    datp = pools[3] if atp_dynamic else 0.0           # ATP row (bug until 2026-10-07: rows 3 and 4 were swapped)
    f = feed_on(t)
    Cx_cmol = Cx / p["Mw"]                                     # Cmol/kg
    dCs = -v[0] * Cx_cmol + D * (f * Cs_f - Cs)                # feed_on scales the inflow concentration
    dCPAA = (v[6] - v[5]) * Cx_cmol + D * (f * CPAA_f - CPAA)
    dCx = (mu - p["vd"] - D) * Cx
    return [pools[0], pools[1], pools[2], pools[4], de11, de32, de4, dv33, datp, dCs, dCPAA, dCx]


def run(D, Cs_f, CPAA_f, hours, y0, feed_on=lambda t: 1.0, t_eval=None, atp_dynamic=False, max_step=np.inf,
        method="LSODA"):
    from scipy.integrate import solve_ivp
    return solve_ivp(rhs, (0.0, hours), y0, args=(D, Cs_f, CPAA_f, feed_on, P, atp_dynamic), method=method,
                     rtol=1e-7, atol=[1e-6] * 9 + [1e-12, 1e-12, 1e-6], t_eval=t_eval, max_step=max_step)


Y0 = [30.0, 900.0, 3000.0, 2.0, 0.5, 20.0, 0.1, 5e-4, 7.0, 1e-5, 1e-3, 5.6]


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = sys.argv[1]
    # chemostat steady states: feed 0.0895 mol/kg glucose (Haringa supp. E), 1 mmol/kg PAA
    Ds = np.array([0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12])
    rows = []
    y = Y0
    for D in Ds:
        sol = run(D, 0.0895, 5e-3, 400.0, y); y = sol.y[:, -1]
        v, atp = rates(y[:9], y[9], y[10])
        rows.append((D, y, v, atp))
        print(f"D {D:5.3f}: mu {v[2]:.4f}  q_s {v[0]:.4f}  q_p {v[7]:.2e}  Cs {y[9]:.2e}  Cx {y[11]:.2f}  gly {y[0]:.1f}  AA {y[1]:.0f}  sto {y[2]:.0f}  PAA {y[3]:.2f}  ATP {atp:.2f}  E11 {y[4]:.3f}  E4 {y[6]:.3f}")
    # feast-famine (de Jonge 2011): D 0.05 1/h, 360 s cycles, feed only during the first 36 s at 10x
    y_ss = run(0.05, 0.0833, 5e-3, 400.0, Y0).y[:, -1]
    cycle = 360.0 / 3600.0
    feed = lambda t: 10.0 if (t % cycle) < 0.1 * cycle else 0.0
    t_eval = np.linspace(0.0, 20 * cycle, 4001)
    ff = run(0.05, 0.0833, 5e-3, 20 * cycle, y_ss, feed_on=feed, t_eval=t_eval, max_step=cycle / 200)
    last = t_eval >= 19 * cycle
    tt = (t_eval[last] - 19 * cycle) * 3600.0
    gly = ff.y[0, last]; sto = ff.y[2, last]; cs = ff.y[9, last]
    atp = ATP_A * gly ** 3 / (gly ** 3 + ATP_B ** 3)
    vv = np.array([rates(ff.y[:9, k], ff.y[9, k], ff.y[10, k])[0] for k in np.nonzero(last)[0]])
    print(f"feast-famine cycle 20: X_gly {gly.min():.1f} .. {gly.max():.1f}, X_ATP {atp.min():.2f} .. {atp.max():.2f}, "
          f"Cs max {cs.max()*1e6:.0f} umol/kg, Cs < 1 umol/kg from t = {tt[np.argmax(cs < 1e-6)]:.0f} s, "
          f"q_s {vv[:, 0].min():.3f} .. {vv[:, 0].max():.3f}, mu {vv[:, 2].min():.3f} .. {vv[:, 2].max():.3f}")
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    mus = [r[2][2] for r in rows]
    for ax, (name, values) in zip(axes[0], (("q_s mol/Cmol/h", [r[2][0] for r in rows]), ("q_p mol/Cmol/h", [r[2][7] for r in rows]),
                                             ("X_gly umol/gdw", [r[1][0] for r in rows]), ("X_E11", [r[1][4] for r in rows]))):
        ax.plot(mus, values, "o-"); ax.set_xlabel("mu, 1/h"); ax.set_title("chemostat: " + name)
    for ax, (name, values) in zip(axes[1], (("C_s umol/kg", cs * 1e6), ("X_gly umol/gdw", gly), ("X_ATP umol/gdw", atp), ("X_sto umol/gdw", sto))):
        ax.plot(tt, values); ax.set_xlabel("t in cycle, s"); ax.set_title("feast-famine: " + name)
    fig.tight_layout(); fig.savefig(out, dpi=110); print("wrote", out)
