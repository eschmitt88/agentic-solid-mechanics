"""LABELLED SANITY CHECK (not a result): recover a known design from inside the parameter space."""
import json, sys, numpy as np
from scipy.optimize import least_squares
import design, fem

p_true = np.array([4.5, 0.6, -0.4, 0.0, np.log(1.3), 0.15, -0.10, 0.0, 0.0])
S = fem.Solver(nx=40, ny=2, Lh=design.LH)
F, _, _ = S.path(p_true, design.DS)
tgt = F * design.KPA
out = dict(p_true=p_true.tolist(), target_sigma=tgt.tolist(), runs={})
for name, p0 in [("arch6", fem.cosine_params(6.0, 1.5)), ("arch3", fem.cosine_params(3.0, 1.2))]:
    prob = design.Problem(tgt, S, log=sys.stdout)
    r = least_squares(prob.fun, p0, jac=prob.jac, bounds=(design.LB, design.UB), method="trf", max_nfev=60, xtol=1e-8, ftol=1e-12, gtol=1e-12)
    _, sig, _, _ = prob.evaluate(r.x)
    out["runs"][name] = dict(p=r.x.tolist(), nrmse=design.nrmse(sig, tgt), max_abs_param_err=float(np.max(np.abs(r.x - p_true))), nfev=len(prob.hist), sigma=sig.tolist())
    print(name, out["runs"][name]["nrmse"], out["runs"][name]["max_abs_param_err"], np.round(r.x, 3), flush=True)
    json.dump(out, open("results/sanity_recovery.json", "w"), indent=1)
