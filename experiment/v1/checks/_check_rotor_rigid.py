"""_check_rotor_rigid.py — rigid-body check for the ROTOR branch of predict.comp.

Usage (repo root):
    python experiment/v1/checks/_check_rotor_rigid.py DUMP.npz ROTOR.obj ROTOR_GROUP_ID THETA OMEGA
DUMP.npz comes from _run_v1_headless.py --dump (positions, material, velocity_mass);
ROTOR.obj is the initial rotor point cloud; THETA/OMEGA are the angle (rad) and
angular velocity (rad/s) at the dump time (printed by --torque-every).
Rotation axis is assumed +y through the origin (30 L tank convention).
See log/2026-09-25_rotor-motion.md section 4.2 for the expected output."""
import numpy as np, sys, json
from scipy.spatial import cKDTree
d=np.load(sys.argv[1]); P=d["positions"]; M=d["material"]; V=d["velocity_mass"]
alive=P[:,3]>0; alive[0]=False
rotor_gid=int(sys.argv[3]); theta=float(sys.argv[4]); omega=float(sys.argv[5])
sel=alive&(M==rotor_gid)
x=P[sel,:3].astype(np.float64); v=V[sel,:3].astype(np.float64)
ref=np.loadtxt(sys.argv[2],comments="#",usecols=(1,2,3))
print("rotor particles: final %d, initial %d"%(len(x),len(ref)))
# rotate initial by theta about +y through origin
c,s=np.cos(theta),np.sin(theta)
R=np.array([[c,0,s],[0,1,0],[-s,0,c]])   # right-hand rotation about +y
rot=ref@R.T
tree=cKDTree(rot); dist,idx=tree.query(x)
print("nearest rotated-initial point: max %.3e  mean %.3e m  (dx=3e-3)"%(dist.max(),dist.mean()))
print("bijective match: %s (unique idx %d)"%(len(np.unique(idx))==len(x),len(np.unique(idx))))
r0=np.hypot(ref[:,0],ref[:,2]); r1=np.hypot(x[:,0],x[:,2])
print("radius multiset: max |sorted r1 - sorted r0| = %.3e m"%np.abs(np.sort(r1)-np.sort(r0)).max())
print("y multiset:      max |sorted y1 - sorted y0| = %.3e m"%np.abs(np.sort(x[:,1])-np.sort(ref[:,1])).max())
# velocity check: v should equal omega * (axis x (x - pivot)) with axis=+y
axis=np.array([0,1,0.]); v_exp=omega*np.cross(axis,x)
print("velocity vs omega x r: max |dv| %.3e m/s, max |v| %.3f m/s (tip expected %.3f)"%(np.abs(v-v_exp).max(),np.linalg.norm(v,axis=1).max(),abs(omega)*0.0491))
print("status:",json.loads(str(d["status"])))
