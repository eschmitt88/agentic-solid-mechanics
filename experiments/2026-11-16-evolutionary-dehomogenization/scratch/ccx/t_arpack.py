"""Buckle-step cost vs ARPACK settings and load scaling (solid girder, mesh 1.0)."""
import sys, time, re, os, numpy as np
from common import E
sys.path.insert(0, E); import verify_ccx as V
d = np.ones((200, 400), bool)
orig = V.build_deck
for label, s_buck, line in [("s=1e-3 tol1e-6 ncv def", 1e-3, "1, 1e-6"), ("s=1e-3 tol1e-3 ncv 20", 1e-3, "1, 1e-3, 20"),
                            ("s=0.5*BLF tol1e-3", 0.5 * 2.2528, "1, 1e-3"), ("s=0.85*BLF tol1e-3", 0.85 * 2.2528, "1, 1e-3"),
                            ("s=0.85*BLF tol1e-3 ncv12", 0.85 * 2.2528, "1, 1e-3, 12")]:
    def bd(*a, **k):
        deck, meta = orig(*a, **k)
        return re.sub(r"\*BUCKLE\n[^\n]+\n", f"*BUCKLE\n{line}\n", deck), meta
    V.build_deck = bd
    r = V.verify(d, kind="phi", r_c=0.0, mesh_size=1.0, threads=8, s_buck=s_buck, nev=1)
    print(label, "blf", r.get("blf"), "fac", r.get("ccx_factors"), "t_ccx", round(r["t_ccx"], 1), flush=True)
