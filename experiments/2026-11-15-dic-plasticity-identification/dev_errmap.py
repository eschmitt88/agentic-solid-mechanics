"""Development diagnostic on a CALIBRATION specimen: where does the strain misfit live?"""
import sys, json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import specimen, data
from scipy.spatial import cKDTree
from plotlib import *
spec = sys.argv[1]
p = np.array(json.load(open("results/baseline.json"))["fits"]["M2"]["p"])
S = specimen.Specimen(spec)
sim = S.simulate(p); simE = S.simulate(p, elastic=True)
print("metrics M2-uniaxial", {k: round(v, 4) for k, v in S.metrics(sim).items()})
print("metrics null-elastic", {k: round(v, 4) for k, v in S.metrics(simE).items()})
# distance of each DIC point to the specimen boundary (outer loop + holes, nominal)
bo, bh = data.load_geometry(data.geom_key(spec))
bpts = np.concatenate([bo] + bh)
dist = cKDTree(bpts).query(S.pts)[0]
dcut = np.minimum(S.pts[:, 1] - S.ylo, S.yhi - S.pts[:, 1])
w = np.array([1, 1, 2.0])[:, None]
for i, j in enumerate(data.FIELD_ST):
    Em, Es = S.E_dic[i], sim["E"][j]
    def rl(mask): return np.sqrt((w * (Es - Em)[:, mask] ** 2).sum() / (w * Em[:, mask] ** 2).sum())
    print(f"st{j}: all {rl(np.ones(len(dist), bool)):.4f} | >1.5mm from edges {rl(dist > 1.5):.4f} | >3mm {rl(dist > 3):.4f} | >3mm & >8mm from cut lines {rl((dist > 3) & (dcut > 8)):.4f} | "
          f"per comp Exx {np.sqrt(((Es-Em)[0]**2).sum()/(Em[0]**2).sum()):.3f} Eyy {np.sqrt(((Es-Em)[1]**2).sum()/(Em[1]**2).sum()):.3f} Exy {np.sqrt(((Es-Em)[2]**2).sum()/(Em[2]**2).sum()):.3f}"
          f" | mean Eyy dic {Em[1].mean():.4f} fe {Es[1].mean():.4f} | mean Exx dic {Em[0].mean():.4f} fe {Es[0].mean():.4f}")
fig, axs = plt.subplots(2, 3, figsize=(10, 12))
for r, c in enumerate([1, 0]):
    Em, Es = S.E_dic[2][c], sim["E"][40][c]
    v = np.percentile(np.abs(Em), 99.5)
    for ax, val, cm, lim in ((axs[r, 0], Em, SEQ if c == 1 else DIV, (0, v) if c == 1 else (-v, v)), (axs[r, 1], Es, SEQ if c == 1 else DIV, (0, v) if c == 1 else (-v, v)), (axs[r, 2], Es - Em, DIV, (-0.25 * v, 0.25 * v))):
        sc = ax.scatter(S.pts[:, 0], S.pts[:, 1], c=val, cmap=cm, vmin=lim[0], vmax=lim[1], s=0.5, marker="s", linewidths=0, rasterized=True)
        ax.set_aspect("equal"); fig.colorbar(sc, ax=ax, orientation="horizontal", pad=0.03)
fig.savefig(f"results/cache/dev_errmap_{spec}.png", dpi=80)
