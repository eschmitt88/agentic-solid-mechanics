import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t_sign
from t_sign import *
for la, lb, nf in [(0.3, None, 4), (0.3, 0.1, 10), (0.3, 1.0, 4), (0.05, None, 4), (0.05, 0.02, 10)]:
    cl = "*CLOAD\n"
    for col, lam in ((0, la), (1, None if lb is None else -lb)):
        if lam is None: continue
        top = np.where((P[:, 1] > 100 - 1e-9) & ((P[:, 0] > 5) == bool(col)))[0]
        ta = list(top + 1) + list(top + nn + 1)
        cl += "".join(f"{n},2,{-Pcr/lam/len(ta):.12g}\n" for n in ta)
    run("sign", s + f"*STEP\n*BUCKLE\n{nf}\n" + cl + "*END STEP\n")
    print(f"A=+{la} B={lb} nf={nf}:", np.round(dat_factors("sign"), 4))
