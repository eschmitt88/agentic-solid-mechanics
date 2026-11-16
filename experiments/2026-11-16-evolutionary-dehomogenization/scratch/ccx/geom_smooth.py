"""Plot smoothed (rounding='smooth') resampled contours vs raw for lattices and a real design."""
import sys, numpy as np
from common import hf, lattice, E
sys.path.insert(0, E); import verify_ccx as V
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
p = hf.Problem(hf.GIRDER, hf.H_FINE)
cases = [("lat w0.15", lattice(p, w=0.15), "phi", (-0.5, 12, -0.5, 8)), ("lat w0.30", lattice(p, w=0.30), "phi", (-0.5, 12, -0.5, 8)),
         ("real_0", np.load("real_0.npz")["level"], "level", (60, 80, 20, 35))]
fig, axs = plt.subplots(len(cases), 3, figsize=(16, 4.2 * len(cases)))
for i, (name, d, kind, lim) in enumerate(cases):
    g, xs, ys, info = V.boundary_field(d, kind)
    lp = V.extract_loops(g, xs, ys)
    for j, rc in enumerate((0.0, 0.5, 1.0)):
        ax = axs[i, j]
        for k, P in enumerate([lp["outer"]] + lp["holes"]):
            ax.plot(*np.r_[P, P[:1]].T, color="0.6", lw=0.6)
            if k == 0:
                P = V._snap(P, hf.GIRDER, 2.5 * info["hs"])
            onrect = (P[:, 0] == 0) | (P[:, 0] == 100) | (P[:, 1] == 0) | (P[:, 1] == 50)
            S = V.smooth_loop(P, rc, onrect if k == 0 else None)
            ax.plot(*np.r_[S, S[:1]].T, "r-", lw=0.8)
        ax.set_xlim(lim[0], lim[1]); ax.set_ylim(lim[2], lim[3]); ax.set_aspect("equal"); ax.set_title(f"{name} smooth r_c={rc}")
plt.tight_layout(); plt.savefig("geom_smooth.png", dpi=80)
