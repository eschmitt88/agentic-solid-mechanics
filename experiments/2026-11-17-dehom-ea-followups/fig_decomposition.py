"""Paper Figure 3: how the hypervolume gain shrinks as each confound is removed (results/decomposition.json).
Dataviz reference palette, slots 1-2 (validated in trial 13); single axis per panel; direct labels."""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COL = {"FULL": "#2a78d6", "PCA": "#eb6834"}
LAB = {"FULL": "paper's method (VAE crossover)", "PCA": "linear crossover in PCA space"}
STAGES = ["nominal", "ho_front", "ho_all", "ho_all_inrng"]
XT = ["nominal\n(search's own\nrealisation)", "held-out\nphases", "+ fair\nbaseline", "+ inside gen-0\nvolume range"]
rows = json.load(open("results/decomposition.json"))
blocks = [("Girder, stiffness (300 generations)", "stiff", 300, [0, 1, 2]),
          ("Girder, buckling (100 generations)", "buck", 100, [0, 1, 2, 3, 4]),
          ("Paper's double-clamped beam (100 generations)", "umax", 100, [0, 1, 2])]
fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.6), sharey=True)
x = np.arange(len(STAGES))
for ax, (title, obj, gen, seeds) in zip(axs, blocks):
    ax.set_facecolor(SURF)
    for arm in ("FULL", "PCA"):
        R = [r for r in rows if r["run"] in [f"{obj}_{arm}_s{s}" for s in seeds] and r["gen"] == gen]
        if not R:
            continue
        Y = np.array([[100 * r[s] for s in STAGES] for r in R])
        for y in Y:
            ax.plot(x, y, color=COL[arm], lw=0.9, alpha=0.35)
        m = Y.mean(0)
        ax.plot(x, m, color=COL[arm], lw=2, marker="o", ms=8, mec=SURF, mew=2, label=f"{LAB[arm]} (n = {len(R)})")
        ax.annotate(f"{m[-1]:.1f}%", (x[-1], m[-1]), xytext=(8, 6 if arm == "PCA" else -10), textcoords="offset points",
                    fontsize=9, color=INK)
        ax.annotate(f"{m[0]:.0f}%", (x[0], m[0]), xytext=(-30, 0), textcoords="offset points", fontsize=9, color=INK, va="center")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(XT, fontsize=8.5, color=INK2)
    ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(colors=INK2, labelsize=9)
    ax.set_xlim(-0.6, len(STAGES) - 0.3)
axs[0].set_ylabel("hypervolume gain over gen 0 (%)", color=INK2)
leg = axs[0].legend(frameon=False, fontsize=8.5, loc="upper right")
fig.patch.set_facecolor(SURF)
fig.suptitle("The reported kind of gain, one confound at a time", x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
fig.text(0.01, 0.005, "Thin lines: individual search seeds. Held-out: mean over 10 phasor realisations never used in search. "
         "Fair baseline: all 480 initial designs, not their nominal front.", fontsize=8, color=INK2)
fig.tight_layout(rect=(0, 0.03, 1, 0.95))
out = Path("../../docs/paper/figs/fig3_decomposition.png"); fig.savefig(out, dpi=170); print(out)
