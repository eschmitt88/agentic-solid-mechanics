"""Sign convention: two independent cantilevers, A compressed (lam=+a), B in tension (lam=-b)."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t_conv import tri6_rect, run, dat_factors
from t_3d import wedge_deck
P1, T1 = tri6_rect(4.0, 100.0, 4, 50)
P = np.r_[P1, P1 + [10.0, 0]]; T = np.r_[T1, T1 + len(P1)]
s, nn, corners, mids = wedge_deck(P, T)
Pcr = np.pi ** 2 * (4 ** 3 / 12) / (4 * 100 ** 2)
bot = np.where(P[:, 1] < 1e-9)[0]
botall = list(bot + 1) + list(bot + nn + 1) + [2 * nn + k + 1 for k, c in enumerate(corners) if P[c, 1] < 1e-9]
s += "*NSET, NSET=BOT\n" + "".join(f"{n}\n" for n in botall) + "*BOUNDARY\nBOT,1,2\n"
for la, lb in [(10, 1), (10, 2), (3, 0.5), (0.3, 0.1)]:
    cl = "*CLOAD\n"
    for col, lam in ((0, la), (1, -lb)):
        top = np.where((P[:, 1] > 100 - 1e-9) & ((P[:, 0] > 5) == bool(col)))[0]
        ta = list(top + 1) + list(top + nn + 1)
        cl += "".join(f"{n},2,{-Pcr/lam/len(ta):.12g}\n" for n in ta)
    run("sign", s + "*STEP\n*BUCKLE\n4\n" + cl + "*END STEP\n")
    print(f"A=+{la} B=-{lb}:", np.round(dat_factors("sign"), 4))
