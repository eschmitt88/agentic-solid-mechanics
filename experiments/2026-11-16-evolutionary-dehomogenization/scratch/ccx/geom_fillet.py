"""Plot fillet_field (union of solid and its smoothed copy) contours vs raw."""
import sys, time, numpy as np
from common import hf, lattice, E
sys.path.insert(0, E); import verify_ccx as V
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
p = hf.Problem(hf.GIRDER, hf.H_FINE)
cases = [("lat w0.15", lattice(p, w=0.15), "phi", (-0.5, 12, -0.5, 8)), ("lat w0.30", lattice(p, w=0.30), "phi", (-0.5, 12, -0.5, 8)),
         ("real_0", np.load("real_0.npz")["level"], "level", (60, 80, 20, 35))]
fig, axs = plt.subplots(len(cases), 3, figsize=(16, 4.2 * len(cases)))
for i, (name, d, kind, lim) in enumerate(cases):
    g0, xs, ys, info = V.boundary_field(d, kind)
    raw = V.extract_loops(g0, xs, ys)
    for j, rc in enumerate((0.0, 0.5, 1.0)):
        t = time.time()
        g = V.fillet_field(g0, xs, ys, info, rc) if rc > 0 else g0
        lp = V.extract_loops(g, xs, ys)
        ax = axs[i, j]
        for P in [raw["outer"]] + raw["holes"]:
            ax.plot(*np.r_[P, P[:1]].T, color="0.6", lw=0.6)
        for P in [lp["outer"]] + lp["holes"]:
            ax.plot(*np.r_[P, P[:1]].T, "r-", lw=0.8)
        ax.set_xlim(lim[0], lim[1]); ax.set_ylim(lim[2], lim[3]); ax.set_aspect("equal")
        ax.set_title(f"{name} fillet r_c={rc}: holes {len(raw['holes'])}->{len(lp['holes'])} ({time.time()-t:.1f}s)")
plt.tight_layout(); plt.savefig("geom_fillet.png", dpi=80)
