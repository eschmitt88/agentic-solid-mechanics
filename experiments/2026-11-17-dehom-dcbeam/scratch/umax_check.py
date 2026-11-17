"""Where is max |u| on gen-0 designs with high umax / C? Solid-node vs all-node umax."""
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE); os.chdir(HERE)
import numpy as np, hf, dehom, run_arm as ra
a = np.load('results/gen0/eval14_stiff.npz'); pca = ra.load_pca()
r = a['rec_umax'] / a['rec_compliance']
P = hf.Problem(hf.PROBLEM, hf.H_FINE)
ny, nx = P.nely, P.nelx
for i in list(np.argsort(-r)[:4]) + [int(np.argsort(-r)[240])]:
    w1, w2, th = ra.fields_from_features(pca.inverse(a['Z0'][i])); d = dehom.dehomogenise(w1, w2, th)
    e = P.evaluate(d['phi']); u = e['u']; s = e['s']
    mag = np.hypot(u[0::2], u[1::2]).reshape(nx + 1, ny + 1)      # [ix, iy]
    sn = np.zeros((nx + 1, ny + 1), bool)
    st = s.T                                                        # [ix, iy]
    sn[:-1, :-1] |= st; sn[1:, :-1] |= st; sn[1:, 1:] |= st; sn[:-1, 1:] |= st
    k = np.unravel_index(mag.argmax(), mag.shape)
    print(i, f"U/C {r[i]:.2f} C {e['compliance']:.2f} umax_all {mag.max():.2f} umax_solid {mag[sn].max():.2f} "
          f"argmax node x={k[0]*hf.H_FINE:.1f} y={k[1]*hf.H_FINE:.1f} solid_node={sn[k]}")

# render the worst outlier with |u| overlay
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
i = int(np.argsort(-r)[0])
w1, w2, th = ra.fields_from_features(pca.inverse(a['Z0'][i])); d = dehom.dehomogenise(w1, w2, th)
e = P.evaluate(d['phi']); u = e['u']
mag = np.hypot(u[0::2], u[1::2]).reshape(nx + 1, ny + 1)
fig, ax = plt.subplots(figsize=(10, 3))
el = 0.25 * (mag[:-1, :-1] + mag[1:, :-1] + mag[1:, 1:] + mag[:-1, 1:]).T
ax.imshow(np.where(e['s'], el, np.nan), origin='lower', extent=[0, 150, 0, 40], cmap='viridis')
ax.set_title(f"gen-0 #{i} (PCA-reconstructed): |u| on solid, C={e['compliance']:.1f}, umax={e['umax']:.1f}", fontsize=9)
fig.colorbar(ax.images[0], ax=ax, fraction=0.02); fig.tight_layout(); fig.savefig('scratch/umax_outlier.png', dpi=110)
