"""Zoom on the extracted geometry around a point: geom_zoom.py w r_c x y half."""
import sys, numpy as np
from common import hf, lattice, E
sys.path.insert(0, E); import verify_ccx as V
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
w, rc, x0, y0, hw = map(float, sys.argv[1:6])
p = hf.Problem(hf.GIRDER, hf.H_FINE)
phi = lattice(p, w=w); s, _ = p.clean(phi)
g, xs, ys, info = V.boundary_field(phi, "phi", r_c=rc)
lp = V.extract_loops(g, xs, ys)
fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(s, origin="lower", extent=[0, 100, 0, 50], cmap="Greys", alpha=0.4)
for P in [lp["outer"]] + lp["holes"]:
    ax.plot(*np.r_[P, P[:1]].T, "r-", lw=0.8)
    R = V.ear_resample(P, 0.5, V._corner_marks(P, 2 * info["hs"]))
    ax.plot(*np.r_[R, R[:1]].T, "b.-", lw=0.5, ms=3)
ax.set_xlim(x0 - hw, x0 + hw); ax.set_ylim(y0 - hw, y0 + hw); ax.set_aspect("equal")
plt.savefig(sys.argv[6] if len(sys.argv) > 6 else "geom_zoom.png", dpi=100)
