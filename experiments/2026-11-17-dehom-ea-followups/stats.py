"""Addendum B item 4: paired-seed statistics for d = PCA - FULL on every metric in results/decomposition.json
(+ the beam's), with t-intervals (95 %) and an exact two-sided sign-flip p-value."""
import itertools, json, sys
import numpy as np
from scipy import stats

rows = json.load(open("results/decomposition.json"))
try:
    rows += json.load(open("../2026-11-17-dehom-dcbeam/results/decomposition.json"))
except FileNotFoundError:
    pass
blocks = {"S gen 300": ("stiff", 300, [0, 1, 2]), "S gen 100": ("stiff", 100, [0, 1, 2, 3, 4]),
          "B gen 100": ("buck", 100, [0, 1, 2, 3, 4]), "beam gen 100": ("umax", 100, [0, 1, 2])}
metrics = ["ho_front", "ho_all", "ho_all_inrng", "att_inrng"]
out = {}
for b, (obj, gen, seeds) in blocks.items():
    get = lambda arm, s: next((r for r in rows if r["run"] == f"{obj}_{arm}_s{s}" and r["gen"] == gen), None)
    pairs = [(get("PCA", s), get("FULL", s)) for s in seeds]
    pairs = [(p, f) for p, f in pairs if p and f]
    if len(pairs) < 2:
        continue
    out[b] = {"n": len(pairs)}
    for m in metrics:
        d = np.array([(p[m] - f[m]) * (100 if m != "att_inrng" else 1) for p, f in pairs])
        n = len(d); mean = d.mean(); se = d.std(ddof=1) / np.sqrt(n); t = stats.t.ppf(0.975, n - 1)
        flips = [np.mean(d * np.array(sg)) for sg in itertools.product([-1, 1], repeat=n)]
        p = np.mean(np.abs(flips) >= abs(mean) - 1e-12)
        out[b][m] = dict(d=[round(x, 3) for x in d], mean=round(mean, 3), ci95=[round(mean - t * se, 3), round(mean + t * se, 3)], p_signflip=round(p, 3))
    print(b, json.dumps(out[b]))
json.dump(out, open("results/paired_stats.json", "w"), indent=1)
