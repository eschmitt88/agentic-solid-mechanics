"""Euler column with ALL z-columns tied in-plane (*EQUATION): buckling still right?"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t_conv import tri6_rect, run, dat_factors
from t_3d import wedge_deck
P, T = tri6_rect(4.0, 100.0, 8, 100)
s, nn, corners, mids = wedge_deck(P, T)
cidx = -np.ones(nn, int); cidx[corners] = np.arange(len(corners))
eq = ""
for n in range(nn):
    for other in ([n + nn] + ([2 * nn + cidx[n]] if cidx[n] >= 0 else [])):
        for d in (1, 2):
            eq += f"*EQUATION\n2\n{other+1},{d},1.0,{n+1},{d},-1.0\n"
bot = np.where(P[:, 1] < 1e-9)[0]; top = np.where(P[:, 1] > 100 - 1e-9)[0]
s += eq + "*NSET, NSET=BOT\n" + "".join(f"{n+1}\n" for n in bot) + "*BOUNDARY\nBOT,1,2\n"
Pcr = np.pi ** 2 * (4 ** 3 / 12) / (4 * 100 ** 2)
cl = "*CLOAD\n" + "".join(f"{n+1},2,{-Pcr*0.01/len(top):.12g}\n" for n in top)
run("ceqa", s + "*STEP\n*BUCKLE\n3\n" + cl + "*END STEP\n")
print("all tied: factors*0.01 / Euler", np.round(np.array(dat_factors("ceqa")) * 0.01, 4))
