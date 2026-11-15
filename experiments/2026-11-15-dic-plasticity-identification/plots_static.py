"""Figures that need no solver output: specimen outlines + split, dog-bone curves with baseline fits."""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import data, baseline, femu
from plotlib import *

def fig_specimens():
    items = [("O1", "O1-3", "calibration"), ("O14", "O14-3", "calibration"), ("O5", "O5-4", "development"),
             ("O15", "O15-3", "held out"), ("X", "XT10 / XR5", "held out")]
    col = {"calibration": C_FULL, "development": C_MUTED, "held out": C_BASE}
    fig, axs = plt.subplots(1, 5, figsize=(7.2, 4.3), gridspec_kw={"width_ratios": [60, 60, 60, 60, 71]})
    for ax, (k, name, role) in zip(axs, items):
        bo, bh = data.load_geometry(k)
        ax.fill(bo[:, 0], bo[:, 1], color=col[role], alpha=0.16, lw=0)
        ax.plot(bo[:, 0], bo[:, 1], color=col[role], lw=1.6)
        for h in bh:
            ax.fill(h[:, 0], h[:, 1], color=SURFACE, lw=0); ax.plot(h[:, 0], h[:, 1], color=col[role], lw=1.6)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"{name}\n{role}", fontsize=9, color=INK)
    fig.suptitle("Five plate shapes cut from one 1.5 mm 304L sheet, all pulled vertically", fontsize=10, color=INK, y=0.99)
    fig.savefig("results/fig_specimens.png", dpi=170, facecolor=SURFACE, bbox_inches="tight"); plt.close(fig)

def fig_dogbone():
    b = json.load(open("results/baseline.json"))
    fig, ax = plt.subplots(figsize=(6.4, 4.2)); style(ax)
    for i, n in enumerate(["BT3", "CT3", "CT4"]):
        e, s_eng, ep, s = baseline.curve(n)
        ax.plot(ep, s, color=INK, lw=1.2, alpha=0.55, label="three dog-bone tests (measured)" if i == 0 else None)
    a = np.linspace(0, 0.9, 300)
    ax.plot(a, femu.sigy(b["fits"]["M2"]["p"], a), color=C_BASE, lw=2, label="fit: Voce + linear (uniaxial baseline)")
    ax.plot(a, femu.sigy(b["fits"]["M1"]["p"], a), color=C_BASE, lw=2, ls=(0, (4, 2)), label="fit: Voce only")
    ax.axvline(b["ep_max"], color=C_MUTED, lw=1, ls=":")
    ax.text(b["ep_max"] + 0.01, 380, "end of uniform\nstretching:\nno data beyond", fontsize=8, color=INK2, va="bottom")
    ax.set_xlabel("plastic strain (true)"); ax.set_ylabel("true stress, MPa"); ax.set_xlim(0, 0.9); ax.set_ylim(250, 1600)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Plain tension test: the baseline's only information", fontsize=10, color=INK, loc="left")
    fig.savefig("results/fig_dogbone.png", dpi=170, facecolor=SURFACE, bbox_inches="tight"); plt.close(fig)

if __name__ == "__main__":
    fig_specimens(); fig_dogbone()
