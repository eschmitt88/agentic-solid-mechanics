"""Fable's gate diagnostic: is the discontinuity in 808's algorithm or in our port/encoding?
Rows: raw LF fields perturbed directly | PCA coordinates perturbed + decode.
Cols: 808-faithful dehom | our default dehom. eps = 1e-12, static compliance."""
import os, sys, json, numpy as np
os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, '.')
import dehom, hf, run_arm as ra
lf = np.load('results/gen0/lf_designs.npz'); e0 = np.load('results/gen0/eval_stiff.npz'); pca = ra.load_pca()
P = hf.Problem(hf.GIRDER, hf.H_FINE); rng = np.random.default_rng(7); DES = ra.DES
idx = rng.choice(480, int(os.environ.get("N2X2", 30)), replace=False); EPS = 1e-12
def fields(route, i, eps):
    if route == "raw":
        w1, w2, th = (lf[k][i].copy() for k in ("w1", "w2", "theta"))
        if eps:
            for a in (w1, w2, th):
                a[DES] += eps * rng.standard_normal(DES.sum())
        return w1, w2, th
    z = e0['Z0'][i] + (eps * rng.standard_normal(e0['Z0'].shape[1]) if eps else 0)
    return ra.fields_from_features(pca.inverse(z))
out = {}
for route in ("raw", "pca"):
    for mode in ("808", "ours"):
        rows = []
        for i in idx:
            r = []
            for eps in (0.0, EPS):
                w1, w2, th = fields(route, i, eps)
                d = dehom.dehomogenise(w1, w2, th, faithful=(mode == "808"), return_fields=True)
                r.append((d, P.evaluate(d['phi'])['compliance']))
            (a, ca), (b, cb) = r
            dN = float(np.abs(a['N'] - b['N']).max()); dW = float(max(np.abs(a['w_coarse'][k] - b['w_coarse'][k]).max() for k in range(2)))
            dP = float(np.abs(np.angle(np.exp(1j * (np.concatenate([np.ravel(p) for p in a['pshift']]) - np.concatenate([np.ravel(p) for p in b['pshift']]))))).max())
            dth_in = float(np.abs(np.angle(np.exp(2j * (fields(route, i, 0.0)[2] - w1 * 0 - th)))).max()) if False else 0.0
            rows.append(dict(i=int(i), pix=int((a['phi'] != b['phi']).sum()), dC=abs(cb / ca - 1), dN=dN, dW=dW, dPhase=dP))
        k = f"{route}|{mode}"
        out[k] = rows
        pix = np.array([r['pix'] for r in rows]); dC = np.array([r['dC'] for r in rows])
        print(f"{k:9s} pixel-changed {np.mean(pix > 0):.2f} | median dC {np.median(dC):.4f} | dC>1% {np.mean(dC > .01):.2f} | "
              f"cleanup changed (dN>1e-6) {np.mean([r['dN'] > 1e-6 for r in rows]):.2f} | widths swapped (dW>1e-6) {np.mean([r['dW'] > 1e-6 for r in rows]):.2f} | "
              f"phase changed (>1e-3) {np.mean([r['dPhase'] > 1e-3 for r in rows]):.2f}", flush=True)
json.dump(out, open('scratch/diag2x2.json', 'w'))
