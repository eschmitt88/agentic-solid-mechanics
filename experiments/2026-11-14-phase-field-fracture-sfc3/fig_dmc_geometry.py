"""Schematic of the calibration and challenge beams (front view + view from below)."""
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from analyze_pf import C
from dmc import GEOMS, W, Hh, L, XS
names = {"HC": "HC: centre notch", "HB": "HB: 9.5 mm off centre", "HA": "HA: 19 mm off centre", "H45": "H45: rotated 45°", "CH": "CHALLENGE: off centre,\nskewed 26.6°, depth 5.1→2.5 mm"}
fig, axs = plt.subplots(2, 5, figsize=(16, 3.6), gridspec_kw=dict(height_ratios=[25.4, 12.7]))
for i, (g, lab) in enumerate(names.items()):
    q = GEOMS[g]; ax = axs[0, i]
    ax.add_patch(plt.Rectangle((-L / 2, -Hh), L, Hh, fc="#eef1f5", ec="#555", lw=1))
    ax.add_patch(plt.Polygon([[q["xf"] - .5, -Hh], [q["xf"] - .5, -Hh + q["df"]], [q["xf"] + .5, -Hh + q["df"]], [q["xf"] + .5, -Hh]], fc=C["orange"], ec=C["orange"]))
    if q["xb"] != q["xf"] or q["db"] != q["df"]:
        ax.add_patch(plt.Polygon([[q["xb"] - .5, -Hh], [q["xb"] - .5, -Hh + q["db"]], [q["xb"] + .5, -Hh + q["db"]], [q["xb"] + .5, -Hh]], fc="none", ec=C["orange"], ls="--"))
    ax.plot([0], [1.2], "v", color=C["ink"], ms=8, clip_on=False); ax.plot([-XS, XS], [-Hh - 1.2] * 2, "^", color=C["ink"], ms=8, clip_on=False)
    ax.set_xlim(-40, 40); ax.set_ylim(-Hh - 3, 3); ax.set_aspect("equal"); ax.axis("off"); ax.set_title(lab, fontsize=9.5, color="#b3360b" if g == "CH" else C["ink"])
    ax = axs[1, i]
    ax.add_patch(plt.Rectangle((-L / 2, -W / 2), L, W, fc="#eef1f5", ec="#555", lw=1))
    ax.plot([q["xf"], q["xb"]], [-W / 2, W / 2], color=C["orange"], lw=3, solid_capstyle="butt")
    ax.set_xlim(-40, 40); ax.set_ylim(-W / 2 - 1, W / 2 + 1); ax.set_aspect("equal"); ax.axis("off")
axs[0, 0].text(-40, 0, "front", fontsize=8, color=C["mute"], va="bottom"); axs[1, 0].text(-40, W / 2 + 0.6, "from below", fontsize=8, color=C["mute"], va="bottom")
fig.tight_layout(); fig.savefig("results/fig_dmc_geometry.png", dpi=130, bbox_inches="tight")
