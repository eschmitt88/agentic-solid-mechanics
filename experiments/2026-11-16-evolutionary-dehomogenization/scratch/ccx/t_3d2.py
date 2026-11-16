"""3D wedge: (a) static step before *BUCKLE (preload?), (b) solver choice, (c) strip plane-stress check."""
import os, sys, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t_conv import tri6_rect, run, dat_factors
from t_3d import wedge_deck
P, T = tri6_rect(4.0, 100.0, 8, 100)
s, nn, corners, mids = wedge_deck(P, T)
bot = np.where(P[:, 1] < 1e-9)[0]; top = np.where(P[:, 1] > 100 - 1e-9)[0]
botall = list(bot + 1) + list(bot + nn + 1) + [2 * nn + k + 1 for k, c in enumerate(corners) if P[c, 1] < 1e-9]
s += "*NSET, NSET=BOT\n" + "".join(f"{n}\n" for n in botall) + "*BOUNDARY\nBOT,1,2\n"
topall = list(top + 1) + list(top + nn + 1)
Pcr = np.pi ** 2 * (4 ** 3 / 12) / (4 * 100 ** 2)
cl = lambda f, op="": f"*CLOAD{op}\n" + "".join(f"{n},2,{-Pcr*f/len(topall):.12g}\n" for n in topall)
for solver in ("SPOOLES", "PARDISO"):
    t = time.time(); out = run("c3", s + f"*STEP\n*STATIC, SOLVER={solver}\n" + cl(0.5) + "*END STEP\n*STEP\n*BUCKLE, SOLVER={solver}\n3\n" + cl(0.01, ", OP=NEW") + "*END STEP\n")
    print(solver, "static(0.5Pcr)+buckle(0.01Pcr):", np.round(np.array(dat_factors("c3")) * 0.01, 4), f"{time.time()-t:.2f}s", [l for l in out.splitlines() if "ERROR" in l or "solver" in l.lower()][:3])
out = run("c3", s + "*STEP\n*BUCKLE\n3\n" + cl(0.01) + "*END STEP\n")
print("buckle only (0.01Pcr):", np.round(np.array(dat_factors("c3")) * 0.01, 4))
