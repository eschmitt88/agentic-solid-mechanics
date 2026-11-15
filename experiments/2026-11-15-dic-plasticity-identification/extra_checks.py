"""Supporting number: plastic-strain range seen by the calibration plates (primary model, station 40)."""
import json, numpy as np
import specimen, fem
fz = json.load(open("results/frozen_models.json")); p = np.array(fz["models"][fz["primary"]]["p"])
out = {}
al = []
for s in ["O1-3", "O14-3"]:
    S = specimen.Specimen(s); sim = S.simulate(p, want_u=True)
    a = np.asarray(sim["u"][40][1][1]).ravel(); w = np.asarray(S.mesh.wdet).ravel()
    o = np.argsort(a); cw = np.cumsum(w[o]) / w.sum()
    out[s] = {"alpha_median": float(a[o][np.searchsorted(cw, 0.5)]), "alpha95": float(a[o][np.searchsorted(cw, 0.95)]), "alpha_max": float(a.max())}
    al.append(out[s]["alpha95"])
out["alpha95_calibration"] = float(np.mean(al))
json.dump(out, open("results/alpha_seen.json", "w"), indent=1); print(out)
# elastic-constant sensitivity lives in elastic_sensitivity.py (needs one process per setting)
