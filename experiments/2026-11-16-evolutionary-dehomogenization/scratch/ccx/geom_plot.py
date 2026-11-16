"""Plot extracted/resampled loops of lattice designs for r_c in {0, 0.5, 1.0}."""
import sys, time, numpy as np
from common import hf, lattice, lattice_level, E
sys.path.insert(0, E); import verify_ccx as V
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
p = hf.Problem(hf.GIRDER, hf.H_FINE)
fig, axs = plt.subplots(3, 2, figsize=(14, 10))
for j, w in enumerate((0.3, 0.15)):
    phi = lattice(p, w=w)
    for i, rc in enumerate((0.0, 0.5, 1.0)):
        g, xs, ys, info = V.boundary_field(phi, "phi", r_c=rc)
        lp = V.extract_loops(g, xs, ys)
        ax = axs[i, j]
        for P in [lp["outer"]] + lp["holes"]:
            ax.plot(*np.r_[P, P[:1]].T, "k-", lw=0.6)
        ax.set_aspect("equal"); ax.set_xlim(-0.5, 20); ax.set_ylim(-0.5, 6); ax.axis("off")
        ax.set_title(f"w={w} r_c={rc}: holes={len(lp['holes'])} dropped={lp['n_dropped_loops']}")
        print(w, rc, len(lp["holes"]), lp["n_dropped_loops"], flush=True)
plt.tight_layout(); plt.savefig("geom_lattice_zoom.png", dpi=110)
