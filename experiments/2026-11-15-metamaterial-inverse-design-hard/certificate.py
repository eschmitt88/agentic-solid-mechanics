"""Infeasibility certificate for a target curve of ANY hyperelastic cell that is stress-free at rest.

Stored energy per unit cell volume W(eps) = int_0^eps sigma d eps must be >= 0 at every strain.  The smallest NRMSE of
any sampled curve whose running trapezoid integral is non-negative is a lower bound on what any design can reach
(under the sampled-curve discretisation).  Quadratic programme: min ||s - s*||^2  s.t.  C s >= 0.
"""
import json, sys
import numpy as np
from scipy.optimize import LinearConstraint, minimize
import design

out = {}
for t in ("H1", "H2", "H4", "T2"):
    tgt = design.TARGETS[t](design.EPS)
    n, h = len(tgt), 0.01
    C = np.zeros((n, n))  # running trapezoid integral from eps = 0 (sigma(0) = 0) to eps_k
    for k in range(n):
        C[k, :k] = h
        C[k, k] = h / 2
    r = minimize(lambda s: 0.5 * np.sum((s - tgt) ** 2), np.maximum(tgt, 0), jac=lambda s: s - tgt, method="SLSQP",
                 constraints=[dict(type="ineq", fun=lambda s: C @ s, jac=lambda s: C)], options=dict(maxiter=500, ftol=1e-14))
    lb = design.nrmse(r.x, tgt)
    out[t] = dict(lower_bound_nrmse=lb, tolerance=design.SPEC[t]["tol"], min_running_integral_of_target=float((C @ tgt).min()),
                  certified_infeasible=bool(lb > design.SPEC[t]["tol"]), closest_admissible_curve=r.x.tolist(), qp_success=bool(r.success))
    print(t, "lower bound NRMSE", round(lb, 4), "tol", design.SPEC[t]["tol"], "min running integral of target", round(float((C @ tgt).min()), 4), "infeasible" if lb > design.SPEC[t]["tol"] else "no certificate")
json.dump(out, open("results/certificate.json", "w"), indent=1)
