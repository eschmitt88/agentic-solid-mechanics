"""Continuity check (trial 14 scratch/chaos.py adapted to dehom-dcbeam): perturb gen-0 designs'
PCA coordinates by eps * N(0, I), decode, de-homogenise, re-evaluate on the HF grid.
Pass: eps = 1e-12 changes no compliance (trial 14 after the ea.decode fix: 0 %)."""
import os, sys, json
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("JAX_PLATFORMS", "cpu")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE); os.chdir(HERE)
import numpy as np
import dehom, hf, run_arm as ra
e0 = np.load('results/gen0/eval14_stiff.npz'); pca = ra.load_pca(); Z0 = e0['Z0']
P = hf.Problem(hf.PROBLEM, hf.H_FINE)
rng = np.random.default_rng(1)
idx = rng.choice(len(Z0), int(os.environ.get("NCHAOS", 40)), replace=False)
EPSS = (0.0, 1e-12, 1e-6, 1e-3)
rows = []
for i in idx:
    res = []
    for eps in EPSS:
        z = Z0[i] + eps * rng.standard_normal(Z0.shape[1]) if eps else Z0[i]
        w1, w2, th = ra.fields_from_features(pca.inverse(z)); d = dehom.dehomogenise(w1, w2, th)
        e = P.evaluate(d['phi'])
        res.append((eps, float(e['compliance']), float(e['umax']), int(d['phi'].sum())))
    rows.append(dict(i=int(i), res=res))
    print(i, [(r[0], round(r[1], 4), r[3]) for r in res], flush=True)
summ = {}
for k, eps in enumerate(EPSS[1:], start=1):
    fin = [r for r in rows if np.isfinite(r['res'][0][1]) and np.isfinite(r['res'][k][1])]
    rel = np.array([abs(r['res'][k][1] / r['res'][0][1] - 1) for r in fin])
    pix = np.array([abs(r['res'][k][3] - r['res'][0][3]) for r in rows])
    conn_flip = int(sum(np.isfinite(r['res'][0][1]) != np.isfinite(r['res'][k][1]) for r in rows))
    summ[str(eps)] = dict(any_pixel=float(np.mean(pix > 0)), dC_gt_1pc=float(np.mean(rel > 0.01)),
                          dC_gt_10pc=float(np.mean(rel > 0.1)), median_dC=float(np.median(rel)),
                          max_dC=float(rel.max()), connectivity_flips=conn_flip, n=len(rows))
    print(f"eps {eps:g}: {summ[str(eps)]}")
json.dump(dict(rows=rows, summary=summ), open('scratch/chaos.json', 'w'), indent=1)
