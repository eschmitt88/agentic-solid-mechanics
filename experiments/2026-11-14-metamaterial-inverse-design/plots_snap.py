"""Figure: why the bistability optimiser stalled (non-smooth ridge at the onset of a secondary snap)."""
import json
import numpy as np
from plotlib import *

g = json.load(open("results/bistable_snap_diagnostic.json"))
d = np.array(g["d"]); m = (d >= 6.0) & (d <= 9.0)
fig, axs = plt.subplots(1, 2, figsize=(10.4, 4.2), dpi=150)
a = axs[0]
for key, col, ls, lab in [("0.0", BLUE, "-", "stalled design"), ("0.3", AQUA, "--", "30 % of the proposed step"), ("1.0", ORANGE, "-.", "full proposed step")]:
    a.plot(d[m], np.array(g["curves"][key]["F"])[m] * 1e3, ls, color=col, lw=2, label=lab)
c0 = g["curves"]["0.0"]
a.annotate("secondary snap:\nforce jumps here", (c0["snap_d"], np.interp(c0["snap_d"], d, c0["F"]) * 1e3), (7.55, 1.5), fontsize=8.5, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
a.set_xlabel("shuttle travel d (mm)"); a.set_ylabel("force (mN per mm depth)"); a.set_title("Force on the negative branch, 0.02 mm resolution"); a.legend(fontsize=8, loc="lower left")
a = axs[1]
al = np.array([0.0, 0.3, 1.0])
names = ["peak force", "second position", "return barrier"]; cols = [BLUE, AQUA, ORANGE]; lss = ["-", "--", "-."]
for k in range(3):
    act = [100 * g["curves"][str(x)]["rel_res"][k] for x in al]
    a.plot(al, act, lss[k], marker="o", color=cols[k], lw=2, ms=6, label=f"{names[k]} — actual")
    a.plot([0, 1], [100 * g["res0"][k], 100 * g["predicted_res"][k]], ":", color=cols[k], lw=1.3)
a.axhline(0, color=MUTED, lw=0.8)
a.text(0.52, -3.0, "dotted = what the gradient predicts\n(all three errors → 0)", fontsize=8.5, color=MUTED)
a.set_xlabel("fraction of the proposed step"); a.set_ylabel("error vs target (%)"); a.set_title("Gradient prediction vs what happens"); a.legend(fontsize=8, loc="lower left")
fig.tight_layout(); fig.savefig("results/bistable_stall_diagnostic.png"); plt.close(fig)
print("ok")
