"""Apply the pre-registered selection rule on the development specimen and freeze the models.
Writes results/frozen_models.json; score.py heldout refuses to run without it."""
import json, datetime, numpy as np
dev = json.load(open("results/dev_scores.json"))["O5-4"]["models"]
NPAR = {"M1": 3, "M2": 4, "M3": 6, "M4": 7, "M4b": 9}
S = {}
for k, m in dev.items():
    S[k] = m["force_nrmse"] / 0.05 + np.mean([m["strain_relL2_st13"], m["strain_relL2_st27"], m["strain_relL2_st40"]]) / 0.15
forms = [k for k in NPAR if k in S]
best = min(S[k] for k in forms)
elig = [k for k in forms if S[k] <= 1.10 * best]
primary = min(elig, key=lambda k: NPAR[k])
vm = [k for k in ("M1", "M2", "M3") if k in S]
bestvm = min(S[k] for k in vm)
vm_pick = min([k for k in vm if S[k] <= 1.10 * bestvm], key=lambda k: NPAR[k])
hard_of = {"M4": "M2", "M4b": "M3"}.get(primary, primary)
models = {}
for k in forms:
    f = json.load(open(f"results/fits/{k}.json"))
    ok = {a: b for a, b in f["fits"].items() if "cost" in b}
    b = ok[min(ok, key=lambda a: ok[a]["cost"])]
    models[k] = {"p": b["p"], "J_calibration": b["cost"], "n_par": NPAR[k], "dev_S": S[k], "kind": "full-field FEMU (O1-3 + O14-3)"}
base = json.load(open("results/baseline.json"))["fits"]
for k in ("M1", "M2", "M3"):
    models[f"uniaxial-{k}"] = {"p": base[k]["p"], "dev_S": S[f"uniaxial-{k}"], "kind": "uniaxial dog bones only (BT3, CT3, CT4), von Mises"}
models["null-elastic"] = {"p": [300.0, 0, 1, 0, 0, 1, 1, 0.5, 3], "elastic": True, "kind": "null model: material kept elastic"}
out = {"frozen_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "primary": primary, "baseline": f"uniaxial-{hard_of}",
       "von_mises_for_calculix": vm_pick, "selection": {"S": S, "best_S": best, "eligible_within_10pct": elig}, "models": models,
       "param_names": ["s0", "Q1", "b1", "H", "Q2", "b2", "a", "h", "c"], "elastic": {"E": 193000.0, "nu": 0.29}}
json.dump(out, open("results/frozen_models.json", "w"), indent=1)
print(json.dumps({k: out[k] for k in ("frozen_utc", "primary", "baseline", "von_mises_for_calculix", "selection")}, indent=1))
