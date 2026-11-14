"""Cell schematic and the three target curves."""
import json
import numpy as np
from plotlib import *
import design, fem

p = np.array(json.load(open("results/design_T3.json"))["p"])
fig, ax = plt.subplots(figsize=(6.4, 3.9), dpi=160)
draw_cell(ax, p, color=BLUE)
xm = design.LH + design.WS / 2
ax.set_ylim(-19.5, 15); ax.set_xlim(-8, 2 * xm + 8)
kw = dict(fontsize=8.5, color=INK)
ax.annotate("", (xm, 9.6), (xm, 14.5), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.6))
ax.text(xm + 1.5, 12.2, "force F, travel d", **kw)
ax.text(-2.5, -17.6, "stiff wall", ha="center", **kw); ax.text(2 * xm + 2.5, -17.6, "stiff wall", ha="center", **kw)
ax.text(xm, -0.5, "stiff\nshuttle", ha="center", va="top", **kw)
ax.annotate("elastomer beam:\nshape + thickness\nare the 9 design\nparameters", (13, 2.6), (6, -12), fontsize=8.5, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
ax.annotate("", (-5, -19), (2 * xm + 5, -19), arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1)); ax.text(xm, -18.6, "cell width W = 70 mm", ha="center", va="bottom", fontsize=8, color=MUTED)
ax.annotate("", (2 * xm + 7, -15.5), (2 * xm + 7, 9.5), arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1)); ax.text(2 * xm + 7.6, -3, "H = 25 mm", rotation=90, va="center", fontsize=8, color=MUTED)
ax.set_title("One unit cell.  stress = F / W,  strain = d / H", fontsize=10)
fig.tight_layout(); fig.savefig("results/cell_schematic.png"); plt.close(fig)

e = np.linspace(0, 0.5, 201)
fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=160)
for T, col, ls, lab in [("T1", BLUE, "-", "T1 stiffening"), ("T2", ORANGE, "--", "T2 plateau"), ("T3", AQUA, "-.", "T3 snap-through")]:
    y = design.TARGETS[T](e); ax.plot(e, y, ls, color=col, lw=2.2, label=lab)
    ax.text(e[-1] + 0.005, y[-1], lab.split()[0], color=INK, fontsize=8.5, va="center")
ax.set_xlim(0, 0.54); ax.set_xlabel("compressive strain"); ax.set_ylabel("stress (kPa)"); ax.set_title("The three prescribed target curves"); ax.legend(fontsize=8.5)
fig.tight_layout(); fig.savefig("results/targets.png"); plt.close(fig)
print("ok")
