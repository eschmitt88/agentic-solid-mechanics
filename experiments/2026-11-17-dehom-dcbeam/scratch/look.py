"""Render a few gen-0 LF designs (w1 + w2 - w1 w2, theta) and their de-homogenised HF
geometry (after island removal) to scratch/look.png. Also the SIMP designs if present."""
import os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import hf, dehom

lf = np.load(os.path.join(HERE, "results/gen0/lf_designs.npz"))
P = hf.Problem(hf.PROBLEM, hf.H_FINE)
idx = [0, 80, 159, 160 + 40, 160 + 120, 320 + 0, 320 + 80, 320 + 159]
simp_f = os.path.join(HERE, "results/simp/simp_designs.npz")
simp = np.load(simp_f) if os.path.exists(simp_f) else None
nrow = len(idx) + (1 if simp is not None else 0)
fig, ax = plt.subplots(nrow, 2, figsize=(13, 1.85 * nrow), gridspec_kw=dict(width_ratios=[1, 1]))
ext = [0, hf.PROBLEM["Lx"], 0, hf.PROBLEM["Ly"]]
for r, i in enumerate(idx):
    w1, w2, th = lf["w1"][i], lf["w2"][i], lf["theta"][i]
    d = dehom.dehomogenise(w1, w2, th)
    e = P.evaluate(d["phi"])
    dens = w1 + w2 - w1 * w2
    ax[r, 0].imshow(dens, origin="lower", extent=ext, cmap="Greys", vmin=0, vmax=1)
    yy, xx = np.mgrid[0:dens.shape[0], 0:dens.shape[1]]
    q = dens.ravel() > 0.05
    ax[r, 0].quiver((xx.ravel()[q] + .5) * hf.HC, (yy.ravel()[q] + .5) * hf.HC, np.cos(th).ravel()[q],
                    np.sin(th).ravel()[q], color="tab:red", scale=90, width=0.0015, headwidth=0, pivot="mid")
    ax[r, 0].set_title(f"#{i} LF  V0={lf['V0'][i]:.3f} Lmin={lf['Lmin'][i]:.2f}  C_LF={lf['compliance_lf'][i]:.2f}", fontsize=8)
    img = np.where(e["s"], 0.0, 1.0)
    img = np.stack([img] * 3, -1); img[P.solid_fixed] = (0.35, 0.45, 0.8)
    ax[r, 1].imshow(img, origin="lower", extent=ext, interpolation="nearest")
    ax[r, 1].set_title(f"HF  vf={e['vf']:.3f}  conn={e['connected']}  C={e['compliance']:.2f}  umax={e['umax']:.2f}", fontsize=8)
if simp is not None:
    r = len(idx)
    for c, k in enumerate((0, 3)):
        e = P.evaluate(simp["phi"][k])
        img = np.where(e["s"], 0.0, 1.0); img = np.stack([img] * 3, -1); img[P.solid_fixed] = (0.35, 0.45, 0.8)
        ax[r, c].imshow(img, origin="lower", extent=ext, interpolation="nearest")
        ax[r, c].set_title(f"SIMP vf={simp['vf'][k]:.2f}: vf={e['vf']:.3f} C={e['compliance']:.2f} umax={e['umax']:.2f}", fontsize=8)
for a in ax.ravel():
    a.set_xticks([]); a.set_yticks([])
fig.tight_layout()
fig.savefig(os.path.join(HERE, "scratch/look.png"), dpi=110)
print("wrote scratch/look.png")
