"""Sensitivity of the development-plate scores to the fixed elastic constants.  One process per setting,
because the constants are baked into the jit-compiled kernels at first trace (the first attempt, changing them
inside one process, silently changed nothing)."""
import sys, json, numpy as np
import fem
E, nu = float(sys.argv[1]), float(sys.argv[2])
fem.set_elastic(E, nu)
import specimen
fz = json.load(open("results/frozen_models.json")); p = np.array(fz["models"][fz["primary"]]["p"])
S = specimen.Specimen("O5-4", verbose=False)
m = S.metrics(S.simulate(p))
f = "results/elastic_sensitivity_O5-4.json"
try: res = json.load(open(f))
except Exception: res = {}
res[f"E{E / 1e3:.0f}_nu{nu}"] = m
json.dump(res, open(f, "w"), indent=1)
print(E, nu, {a: round(b, 5) for a, b in m.items()})
