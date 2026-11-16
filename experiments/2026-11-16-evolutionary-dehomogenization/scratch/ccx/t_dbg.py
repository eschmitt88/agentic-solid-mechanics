import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + "/../.."); sys.path.insert(0, HERE)
import verify_ccx as V
from t_conv import tri6_rect
W = os.path.join(HERE, "work")
cfg = dict(Lx=10.0, Ly=1.0, pad_w=5.0, k_line=1.0, P=1.0)
X, T = tri6_rect(10.0, 1.0, 40, 4)
deck, meta = V.build_deck(X, T, cfg, 1.0, buckling=False)
open(os.path.join(W, "j.inp"), "w").write(deck); V.run_ccx("j", W, 8)
b = [b for b in V.parse_frd(os.path.join(W, "j.frd")) if b["name"] == "DISP"][0]
nn = len(X); U = np.zeros((3 * nn + 200, 3)); U[b["nodes"]] = b["values"][:, :3]
for yv in (0, 0.25, 0.5, 0.75, 1.0):
    s = np.isclose(X[:, 1], yv)
    print(yv, "uy z0", U[:nn][s, 1].mean().round(4), "uy zt", U[nn:2*nn][s, 1].mean().round(4), "ux range", U[:nn][s,0].min().round(4), U[:nn][s,0].max().round(4), "uz0", U[:nn][s,2].mean().round(4))
print("ky bottom nodes", meta['ky'][np.isclose(X[:,1],0)][:6], "sum", meta['ky'].sum(), meta['kx'].sum())
print("F sum", meta['Fv'].sum(), "F first", list(zip(meta['Fn'][:8], meta['Fv'][:8].round(5))))
