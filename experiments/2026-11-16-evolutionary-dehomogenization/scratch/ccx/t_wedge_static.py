"""Static checks of the one-layer C3D15 model (verify_ccx.build_deck) on structured tri6 meshes."""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE + "/../.."); sys.path.insert(0, HERE)
import verify_ccx as V
from t_conv import tri6_rect
W = os.path.join(HERE, "work"); os.makedirs(W, exist_ok=True)
def solve(X, T, cfg):
    deck, meta = V.build_deck(X, T, cfg, cfg["P"], buckling=False)
    open(os.path.join(W, "j.inp"), "w").write(deck)
    rc, log = V.run_ccx("j", W, 8)
    b = [b for b in V.parse_frd(os.path.join(W, "j.frd")) if b["name"] == "DISP"][0]
    nn = meta["nn2"]; U = np.zeros((3 * nn + 1, 3)); U[b["nodes"]] = b["values"][:, :3]
    F = np.zeros(len(U)); F[meta["Fn"]] = meta["Fv"]
    return -(F * U[:, 1]).sum(), U[:nn], meta
# strip 10 x 1, springs under whole bottom (pad_w = Lx) -> left x+y over [0,5], right y over [5,10]
cfg = dict(Lx=10.0, Ly=1.0, pad_w=5.0, k_line=1.0, P=1.0)
for n in (1, 2, 4):
    X, T = tri6_rect(10.0, 1.0, 20 * n, 2 * n)
    C, U, meta = solve(X, T, cfg)
    print(f"strip n={n}: C={C:.6f} (expect 0.1+0.1=0.2 for uniform; ux free) uy_top={U[X[:,1]==1,1].mean():.6f} spring_sum={-(meta['ky']*U[:,1]).sum():.6f}")
# deep beam: same girder dims, no duct, structured mesh
import hf
cfgG = dict(hf.GIRDER)
for n in (1, 2):
    X, T = tri6_rect(100.0, 50.0, 100 * n, 50 * n)
    C, U, meta = solve(X, T, cfgG)
    print(f"girder-no-duct structured tri6 h={1/n}: C={C:.5f}")
for h in (0.5, 0.25, 0.125):
    p = hf.Problem(dict(hf.GIRDER, ducts=[], liners=[]), h); print("hf no duct", h, p.evaluate(np.ones((p.nely, p.nelx), bool))["compliance"])
