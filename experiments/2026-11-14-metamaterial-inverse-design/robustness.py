"""Geometric-tolerance sensitivity of the final designs (design solver; reported as a caveat, not pass/fail).

16 random perturbations per design: centre-line modes +-0.05 mm (1 sigma), thickness scale 3 %, thickness modes 0.02.
"""
import json, sys, numpy as np, design, fem
T = sys.argv[1]
rng = np.random.default_rng(42)
p = np.array(json.load(open(f"results/design_{T}.json"))["p"])
tgt = design.TARGETS[T](design.EPS)
S = fem.Solver(nx=40, ny=2, Lh=design.LH)
sd = np.array([0.05] * fem.NK + [0.03] + [0.02] * fem.NM)
out = dict(sd=sd.tolist(), samples=[])
for k in range(16):
    q = p + sd * rng.standard_normal(len(p))
    F, _, infos = S.path(q, design.DS)
    sig = F * design.KPA
    out["samples"].append(dict(p=q.tolist(), nrmse=design.nrmse(sig, tgt) if np.all(np.isfinite(sig)) else None, sigma=sig.tolist(),
                               switched=int(sum(i["switched"] for i in infos))))
    print(T, k, out["samples"][-1]["nrmse"], flush=True)
e = np.array([s["nrmse"] for s in out["samples"] if s["nrmse"] is not None])
out.update(median=float(np.median(e)), max=float(e.max()), n_ok=int(len(e)), frac_within_tol=float(np.mean(e <= (0.08 if T == "T3" else 0.05))))
json.dump(out, open(f"results/robustness_{T}.json", "w"), indent=1)
print(T, out["median"], out["max"], out["frac_within_tol"])
