"""Plot ccx first-mode deformation magnitude (rigid motion removed) for a design.
usage: mode_plot.py <solid|lat_wX> r_c mesh out.png"""
import sys, numpy as np
from common import hf, lattice, E
sys.path.insert(0, E); import verify_ccx as V
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.tri as mtri
name, rc, ms, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
rounding = sys.argv[5] if len(sys.argv) > 5 else "smooth"
p = hf.Problem(hf.GIRDER, hf.H_FINE)
d = np.ones((p.nely, p.nelx), bool) if name == "solid" else lattice(p, w=float(name.split("w")[1]))
r = V.verify(d, kind="phi", r_c=rc, mesh_size=ms, threads=4, nev=1, return_mesh=True, rounding=rounding)
X, T, m = r["X"], r["T"], r["mode_mag"]
tri = mtri.Triangulation(X[:, 0], X[:, 1], T[:, :3])
fig, ax = plt.subplots(figsize=(12, 6.5))
tc = ax.tripcolor(tri, m / m.max(), shading="gouraud", cmap="viridis")
ax.plot(*r["mode_xy"], "r+", ms=14)
ax.set_aspect("equal"); ax.set_title(f"{name} r_c={rc} mesh={ms}: BLF={r['blf']:.4f}, mode max at {np.round(r['mode_xy'], 2)}")
plt.colorbar(tc, ax=ax, shrink=0.6); plt.tight_layout(); plt.savefig(out, dpi=90)
print(r["blf"], r["mode_xy"], r["mode_pr"])
