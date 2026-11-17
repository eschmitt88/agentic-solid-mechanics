"""How sensitive is the evaluator to infinitesimal perturbations of the design coordinates?"""
import os, sys, json, numpy as np
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, '.')
import dehom, hf, run_arm as ra
e0 = np.load('results/gen0/eval_stiff.npz'); pca = ra.load_pca(); Z0 = e0['Z0']
P = hf.Problem(hf.GIRDER, hf.H_FINE)
rng = np.random.default_rng(1)
idx = rng.choice(len(Z0), int(os.environ.get("NCHAOS", 60)), replace=False)
rows = []
for i in idx:
    res = []
    for eps in (0.0, 1e-12, 1e-9, 1e-6, 1e-3):  # noqa
        z = Z0[i] + eps * rng.standard_normal(Z0.shape[1]) if eps else Z0[i]
        w1, w2, th = ra.fields_from_features(pca.inverse(z)); d = dehom.dehomogenise(w1, w2, th)
        e = P.evaluate(d['phi'])
        res.append((eps, float(e['compliance']), int(d['phi'].sum())))
    rows.append(dict(i=int(i), res=res))
    print(i, [(r[0], round(r[1], 4), r[2]) for r in res], flush=True)
json.dump(rows, open(f"scratch/chaos_{os.environ.get('DEHOM_ORDER','width')}.json", 'w'))
for k, eps in enumerate((1e-12, 1e-9, 1e-6, 1e-3), start=1):
    rel = np.array([abs(r['res'][k][1] / r['res'][0][1] - 1) for r in rows])
    pix = np.array([abs(r['res'][k][2] - r['res'][0][2]) for r in rows])
    print(f"eps {eps:g}: designs with any pixel change {np.mean(pix > 0):.2f}; |dC|/C > 1 %: {np.mean(rel > 0.01):.2f}, > 10 %: {np.mean(rel > 0.1):.2f}; median {np.median(rel):.4f}")
