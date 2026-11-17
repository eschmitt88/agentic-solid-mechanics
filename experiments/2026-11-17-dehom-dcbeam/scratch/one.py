import os, sys, time
os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, hf, lf, dehom
r = lf.lf_optimise(0.35, 0.10, **lf.GEN0_SETTINGS)
print("LF", r["compliance_lf"], r["vol_lf"], r["n_iter"], round(r["wall_s"], 1))
t = time.time(); d = dehom.dehomogenise(r["w1"], r["w2"], r["theta"]); td = time.time() - t
P = hf.Problem(hf.PROBLEM, hf.H_FINE)
t = time.time(); e = P.evaluate(d["phi"]); te = time.time() - t
print("HF", e["vf"], e["connected"], e["compliance"], e["umax"], round(td, 2), round(te, 2), d["info"])
np.savez("/tmp/dcb_one.npz", phi=d["phi"], s=e["s"], w1=r["w1"], w2=r["w2"], th=r["theta"], rho=r["rho"])
