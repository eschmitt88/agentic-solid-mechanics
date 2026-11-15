"""Verification of pf3d (hybrid, GPU-PCG) against the 2D solver: SENT tension as a plane-strain slab."""
import sys, time, json, numpy as np
import pf3d
from pf import graded_1d, LAM, MU, GC
l, h = 0.015, float(sys.argv[1]) if len(sys.argv) > 1 else 0.00375
E = MU * (3 * LAM + 2 * MU) / (LAM + MU); nu = LAM / (2 * (LAM + MU))
xs = graded_1d(0, 1, 0.5 - 2 * l - 0.01, 1.0, h, 0.04, must=(0.5,))
nb = int(round(max(4 * l, 0.03) / h))
zs = graded_1d(0, 1, 0.5 - (nb + 0.5) * h, 0.5 + (nb + 0.5) * h, h, 0.04)
ys = np.array([0.0, h])
P = pf3d.PF3D(xs, ys, zs, E, nu, GC, l, remove_fn=lambda x, y, z: (x < 0.5) & (np.abs(z - 0.5) < 0.5 * h), kres=1e-6)
bot = P.nodes_where(lambda x, y, z: z < 1e-9); top = P.nodes_where(lambda x, y, z: z > 1 - 1e-9)
alln = np.nonzero(P.active)[0]
P.fix(alln, 1)                      # plane strain
for c in (0, 2):
    P.fix(bot, c); P.fix(top, c)
us = np.concatenate([np.arange(1, 51) * 1e-4, 5e-3 + np.arange(1, 151) * 1e-5])
rec = []
t0 = time.time()
for n, ub in enumerate(us):
    uD = np.zeros(3 * P.nn); uD[3 * top + 2] = ub
    info = P.step(uD, max_stag=300, stag_tol=1e-3)
    F = P.reaction(top, 2) / h
    rec.append((ub, F, info["nstag"]))
    if n % 10 == 0 or info["nstag"] > 5:
        print(f"step {n} u={ub:.5e} F={F:.5f} nstag={info['nstag']} cg={info['cg_u']} dmax={float(P.d.max()):.3f} t={time.time()-t0:.0f}s", flush=True)
    if F < 0.02 and n > 20:
        break
rec = np.array(rec); i = rec[:, 1].argmax()
print(f"RESULT pf3d-hybrid SENT tension l={l} h={h}: peak {rec[i,1]:.4f} kN at u={rec[i,0]:.5e}")
np.savetxt(f"results/verify3d_sent_h{h}.csv", rec, header="u_mm F_kN nstag", fmt="%.6e")
