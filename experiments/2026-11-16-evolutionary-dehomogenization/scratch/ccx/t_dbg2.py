import os, sys, re, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + "/../.."); sys.path.insert(0, HERE)
import verify_ccx as V
from t_conv import tri6_rect
W = os.path.join(HERE, "work")
cfg = dict(Lx=10.0, Ly=1.0, pad_w=5.0, k_line=1.0, P=1.0)
X, T = tri6_rect(10.0, 1.0, 20, 2)
nn = len(X)
deck, meta = V.build_deck(X, T, cfg, 1.0, buckling=False)
# variant A: replace springs by uy=0 on bottom (z=0,z=t) + ux=0 at one node
head, rest = deck.split("*ELEMENT, TYPE=SPRING1", 1)
steps = rest[rest.index("*STEP"):]
bot = np.where(np.isclose(X[:, 1], 0))[0]
bc = "*NSET, NSET=NB\n" + "".join(f"{n+1}\n{n+nn+1}\n" for n in bot) + "*BOUNDARY\nNB,2,2\n" + f"{bot[0]+1},1,1\n{bot[0]+nn+1},1,1\n"
for name, d in (("fixed", head + bc + steps),):
    open(os.path.join(W, "k.inp"), "w").write(d); V.run_ccx("k", W, 8)
    b = [b for b in V.parse_frd(os.path.join(W, "k.frd")) if b["name"] == "DISP"][0]
    U = np.zeros((3 * nn + 200, 3)); U[b["nodes"]] = b["values"][:, :3]
    for yv in (0, 0.5, 1.0):
        s = np.isclose(X[:, 1], yv)
        print(name, yv, "uy", U[:nn][s, 1].mean().round(4), "uz z0", U[:nn][s, 2].mean().round(4), "uz zt", U[nn:2*nn][s, 2].mean().round(4))
