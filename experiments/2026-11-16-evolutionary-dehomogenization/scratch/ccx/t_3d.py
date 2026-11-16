"""Column buckling with extruded one-layer C3D15 wedges (uz fixed at mid-thickness nodes)."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t_conv import tri6_rect, run, dat_factors
def wedge_deck(P, T, t=1.0):
    nn = len(P); corners = np.unique(T[:, :3]); midmap = -np.ones(nn, int); midmap[corners] = np.arange(len(corners))
    nod = "*NODE, NSET=NALL\n"
    nod += "".join(f"{i+1},{x:.12g},{y:.12g},0\n" for i, (x, y) in enumerate(P))
    nod += "".join(f"{nn+i+1},{x:.12g},{y:.12g},{t:.12g}\n" for i, (x, y) in enumerate(P))
    nod += "".join(f"{2*nn+k+1},{P[c,0]:.12g},{P[c,1]:.12g},{t/2:.12g}\n" for k, c in enumerate(corners))
    el = "*ELEMENT, TYPE=C3D15, ELSET=EALL\n"
    for e, tr in enumerate(T):
        b = tr + 1; tp = tr + nn + 1; m = midmap[tr[:3]] + 2 * nn + 1
        ids = [b[0], b[1], b[2], tp[0], tp[1], tp[2], b[3], b[4], b[5], tp[3], tp[4], tp[5], m[0], m[1], m[2]]
        el += f"{e+1}," + ",".join(map(str, ids[:10])) + ",\n" + ",".join(map(str, ids[10:])) + "\n"
    mids = 2 * nn + np.arange(len(corners)) + 1
    s = nod + el + "*NSET, NSET=NMID\n" + "".join(f"{n}\n" for n in mids)
    s += "*MATERIAL, NAME=M\n*ELASTIC\n1.0, 0.3\n*SOLID SECTION, ELSET=EALL, MATERIAL=M\n*BOUNDARY\nNMID,3,3\n"
    return s, nn, corners, mids
if __name__ == "__main__":
    P, T = tri6_rect(4.0, 100.0, 8, 100)
    s, nn, corners, mids = wedge_deck(P, T)
    bot = np.where(P[:, 1] < 1e-9)[0]; top = np.where(P[:, 1] > 100 - 1e-9)[0]
    botall = list(bot + 1) + list(bot + nn + 1) + [2 * nn + k + 1 for k, c in enumerate(corners) if P[c, 1] < 1e-9]
    s += "*NSET, NSET=BOT\n" + "".join(f"{n}\n" for n in botall) + "*BOUNDARY\nBOT,1,2\n"
    topall = list(top + 1) + list(top + nn + 1)
    Pcr = np.pi ** 2 * (4 ** 3 / 12) / (4 * 100 ** 2)
    for f in (1.0, 0.2):
        cl = "*CLOAD\n" + "".join(f"{n},2,{-Pcr*f/len(topall):.12g}\n" for n in topall)
        run("col3", s + "*STEP\n*BUCKLE\n4\n" + cl + "*END STEP\n")
        print(f"load {f}*Pcr: factors", np.round(dat_factors("col3"), 4), "expect", 1 / f)
