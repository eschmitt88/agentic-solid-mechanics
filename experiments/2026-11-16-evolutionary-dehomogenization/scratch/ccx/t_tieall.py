"""Tie the in-plane dofs of every z-column (plane-stress kinematics by *EQUATION): cost and result."""
import sys, time, re, os, numpy as np
from common import E
sys.path.insert(0, E); import verify_ccx as V
d = np.ones((200, 400), bool)
orig = V.build_deck
def bd(X, T, *a, **k):
    deck, meta = orig(X, T, *a, **k)
    nn = len(X); corners = np.unique(T[:, :3]); midmap = -np.ones(nn, int); midmap[corners] = np.arange(len(corners))
    tied = set(np.where((meta["kx"] > 0) | (meta["ky"] > 0))[0])
    eq = []
    for n in range(nn):
        if n in tied: continue
        for o in [n + nn] + ([2 * nn + midmap[n]] if midmap[n] >= 0 else []):
            for dd in (1, 2):
                eq += ["*EQUATION", "2", f"{o+1},{dd},1.0,{n+1},{dd},-1.0"]
    deck = deck.replace("*STEP\n", "\n".join(eq) + "\n*STEP\n", 1)
    return deck, meta
for tie in (False, True):
    V.build_deck = bd if tie else orig
    r = V.verify(d, kind="phi", r_c=0.0, mesh_size=1.0, threads=8, nev=1)
    print("tie all" if tie else "faces free", "C", r.get("compliance"), "blf", r.get("blf"), "t_ccx", round(r["t_ccx"], 1), flush=True)
