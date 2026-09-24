"""_check_taylor_couette.py — compare the steady 2D Taylor-Couette profile with the analytic solution.

Usage (repo root):
    python experiment/v1/checks/_check_taylor_couette.py DUMP.npz OUT.png
DUMP.npz from _run_v1_headless.py cases/taylor_couette_2d/case.yaml --max-steps 40000 --dump DUMP.npz
See log/2026-09-25_rotor-motion.md section 4.3 for the expected output."""
import numpy as np, sys, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
d=np.load(sys.argv[1]); P=d["positions"]; M=d["material"]; V=d["velocity_mass"]
alive=P[:,3]>0; alive[0]=False
fluid=alive&(M==0)          # first particle source = couette_fluid -> group 0
x=P[fluid,:3].astype(np.float64); v=V[fluid,:3].astype(np.float64)
r=np.hypot(x[:,0],x[:,1]); et=np.column_stack([-x[:,1],x[:,0]])/r[:,None]
u_t=(v[:,:2]*et).sum(1); u_r=(v[:,:2]*(x[:,:2]/r[:,None])).sum(1)
ri,ro,Om=0.05,0.10,2.0
A=Om*ri**2/(ri**2-ro**2); B=-A*ro**2
ana=lambda rr: A*rr+B/rr
bins=np.linspace(ri,ro,26); idx=np.digitize(r,bins)
rc=[];um=[];us=[];ur=[]
for b in range(1,len(bins)):
    m=idx==b
    if m.sum()<5: continue
    rc.append(r[m].mean()); um.append(u_t[m].mean()); us.append(u_t[m].std()); ur.append(np.abs(u_r[m]).mean())
rc,um,us,ur=map(np.array,(rc,um,us,ur))
err=um-ana(rc); ui=Om*ri
print("fluid particles:",len(x))
print("bin   r[m]    u_theta_sph   u_theta_exact   err      err/u_inner   std_in_bin   |u_r|")
for a,b,c,e,s,q in zip(rc,um,ana(rc),err,us,ur): print("     %.4f   %.5f      %.5f      %+.2e   %+.3f%%     %.1e      %.1e"%(a,b,c,e,100*e/ui,s,q))
print("max |err| / u_inner = %.3f %%,  rms = %.3f %%"%(100*np.abs(err).max()/ui,100*np.sqrt((err**2).mean())/ui))
fig,ax=plt.subplots(figsize=(7,5)); rr=np.linspace(ri,ro,200)
ax.plot(rr*1e3,ana(rr),"k-",label="analytic  A r + B/r")
ax.errorbar(rc*1e3,um,yerr=us,fmt="o",ms=4,label="SPH (bin mean ± std), t = 7.5 s")
ax.set_xlabel("r [mm]"); ax.set_ylabel("u_theta [m/s]"); ax.set_title("2D Taylor-Couette, inner ROTOR Omega=2 rad/s, Re=5, dx=1 mm, h/dx=5"); ax.legend(); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(sys.argv[2],dpi=110); print("saved",sys.argv[2])
