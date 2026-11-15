"""POST HOC (after the held-out pass; not part of any graded check): what do the slow rolling-direction dog bones
(AR3, BR3 -- never used for fitting) say about the Hill coefficient `a` identified from transverse-loaded plates?"""
import json, numpy as np, baseline
fz = json.load(open("results/frozen_models.json")); p = fz["models"][fz["primary"]]["p"]
g = np.array([0.02, 0.05, 0.1, 0.2, 0.3, 0.4])
T = np.mean([np.interp(g, *baseline.curve(n)[2:4]) for n in ["BT3", "CT3", "CT4"]], 0)
Rl = np.mean([np.interp(g, *baseline.curve(n)[2:4]) for n in ["AR3", "BR3"]], 0)
out = {"plastic_strain": g.tolist(), "true_stress_transverse_MPa": T.tolist(), "true_stress_rolling_MPa": Rl.tolist(),
       "measured_ratio_rolling_over_transverse": (Rl / T).tolist(), "M4_predicted_ratio": float(1 / np.sqrt(p[6])),
       "M4_r_value_transverse": float(p[7] / (1 - p[7]))}
json.dump(out, open("results/posthoc_rolling_dogbones.json", "w"), indent=1)
print(json.dumps(out, indent=1))
