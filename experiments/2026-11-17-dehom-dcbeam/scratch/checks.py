"""Sanity checks for the DCBEAM problem: LF vs HF solid compliance, Timoshenko, masks."""
import os, sys, json, time
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import hf, lf

out = {}
pc = hf.Problem(hf.PROBLEM, hf.HC); pf = hf.Problem(hf.PROBLEM, hf.H_FINE)
out["grids"] = dict(coarse=[pc.nelx, pc.nely], fine=[pf.nelx, pf.nely])
out["coarse_counts"] = dict(solid=int(pc.solid_fixed.sum()), design=int(pc.design.sum()), load_block=int(pc.skin.sum()))
out["fine_counts"] = dict(solid=int(pf.solid_fixed.sum()), design=int(pf.design.sum()), load_block=int(pf.skin.sum()))
out["F_sum"] = [float(pc.F.sum()), float(pf.F.sum())]
out["n_fixed_dofs"] = [len(pc.fixed), len(pf.fixed)]
t = time.time(); J_lf = lf.lf_solid_compliance(); t_lf = time.time() - t
# HF on the coarse grid too (same Q4 solver, isotropic) to separate discretisation from LF model
ec = pc.evaluate(np.ones((pc.nely, pc.nelx), bool))
t = time.time(); ef = pf.evaluate(np.ones((pf.nely, pf.nelx), bool)); t_hf = time.time() - t
out["solid"] = dict(C_lf_coarse=J_lf, C_q4_coarse=ec["compliance"], C_hf_fine=ef["compliance"],
                    rel_lf_vs_hf=J_lf / ef["compliance"] - 1, umax_coarse=ec["umax"], umax_fine=ef["umax"],
                    t_lf_s=t_lf, t_hf_s=t_hf)
# Timoshenko clamped-clamped, central point load, L = 150, depth 40, unit thickness, plane stress E = 1
E, nu, L, d, P = 1.0, 0.3, 150.0, 40.0, 1.0
I = d ** 3 / 12; A = d; G = E / (2 * (1 + nu)); k = 5 / 6
db = P * L ** 3 / (192 * E * I); ds = P * L / (4 * k * G * A)
u = ef["u"]; uy = u[1::2].reshape(pf.nelx + 1, pf.nely + 1)   # [ix, iy]
ixm = pf.nelx // 2
out["timoshenko"] = dict(bending=db, shear=ds, total=db + ds, fe_umax=ef["umax"],
                         fe_mid_top=-float(uy[ixm, -1]), fe_mid_axis=-float(uy[ixm, pf.nely // 2]),
                         fe_mid_bottom=-float(uy[ixm, 0]),
                         ratio_umax=ef["umax"] / (db + ds), ratio_axis=-float(uy[ixm, pf.nely // 2]) / (db + ds))
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "checks.json"), "w"), indent=1)
