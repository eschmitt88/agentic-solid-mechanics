"""Which part of the fabrication-error model breaks the T2 design?  Pass fraction of the nominal optimum when only one
group of parameters is perturbed (same 64 draws, other components zeroed).  Diagnostic, not pre-registered."""
import json
import numpy as np
import design, fem

t = "T2"
p = np.array(json.load(open("results/designs/T2_nominal.json"))["p"])
prob = design.Problem(t)
offs = design.fab_samples(64, 2026, 4, 4)
groups = {"centre_line_only (0.05 mm)": slice(0, 4), "thickness_scale_only (3 %)": slice(4, 5), "thickness_modes_only (0.02)": slice(5, 9)}
out = {}
for name, sl in groups.items():
    res = []
    for off in offs:
        o = np.zeros(9); o[sl] = off[sl]
        g = design.grade(t, prob.curve(p + o)[0])
        res.append(dict(nrmse=g["nrmse"], passed=g["passed"], band=g["bands"][0]["max_dev"]))
    e = np.array([r["nrmse"] for r in res])
    out[name] = dict(pass_fraction=float(np.mean([r["passed"] for r in res])), median_nrmse=float(np.median(e)), p90_nrmse=float(np.quantile(e, 0.9)),
                     fail_by_nrmse=int(np.sum(e > 0.05)), fail_by_band=int(np.sum([r["band"] > 0.06 for r in res])))
    print(name, out[name], flush=True)
# local sensitivity: d(sigma)/d(log t0) relative to sigma (3 = pure bending scaling)
sig, zs, _ = prob.curve(p)
_, J = prob.S.jacobian(p, prob.DS, zs)
ratio = (J[:, 4] * design.KPA) / sig
out["dlog_sigma_dlog_t0"] = dict(median=float(np.median(ratio)), min=float(ratio.min()), max=float(ratio.max()))
print(out["dlog_sigma_dlog_t0"])
json.dump(out, open("results/robust_decompose_T2.json", "w"), indent=1)
