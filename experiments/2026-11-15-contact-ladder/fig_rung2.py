"""Rung 2 figures."""
import json, os, glob
import numpy as np
import plotstyle
from matplotlib.collections import PolyCollection
plt = plotstyle.setup(); C = plotstyle.C
R = "results/"
import rung2_mindlin_own as M2
import rung2_ironing_own as RI

# ---------------- Cattaneo-Mindlin
def mindlin(own_npz, ccx_npz, out):
    fig, axs = plt.subplots(3, 1, figsize=(6.6, 11.5))
    d = np.load(own_npz); c = np.load(ccx_npz) if ccx_npz and os.path.exists(ccx_npz) else None
    for ax, qr in zip(axs, M2.QS):
        xx = np.linspace(-1.05, 1.05, 800) * M2.A_T
        cth, qth = M2.theory(xx, qr)
        ax.plot(xx / M2.A_T, qth / (M2.MU * M2.P0), color=C["ref"], lw=2.5, label="Cattaneo–Mindlin (closed form)")
        ax.plot(xx / M2.A_T, np.sqrt(np.clip(1 - (xx / M2.A_T) ** 2, 0, None)), color=C["mute"], lw=1, ls="--", label="friction limit f·p")
        o = np.argsort(d["x"]); ax.plot(d["x"][o] / M2.A_T, np.abs(d[f"q_{qr}"][o]) / (M2.MU * M2.P0), "o", ms=3.2, color=C["blue"], label="own solver")
        if c is not None:
            o = np.argsort(c["x"]); ax.plot(c["x"][o] / M2.A_T, c[f"q_{qr}"][o] / (M2.MU * M2.P0), "s", ms=3.2, mfc="none", color=C["orange"], label="CalculiX")
        ax.axvspan(-cth / M2.A_T, cth / M2.A_T, color="#e9f1fb", zorder=0)
        ax.set_title(f"sideways load = {qr:g} × sliding load"); ax.set_xlim(-1.15, 1.15); ax.set_ylim(0, 1.1)
    [a.set_ylabel("friction traction / (f · Hertz peak)") for a in axs]; axs[0].legend(fontsize=9.5, loc="upper center")
    axs[0].text(0, 0.05, "stuck zone", ha="center", fontsize=10.5, color="#52514e"); axs[2].set_xlabel("position / contact half-width")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)

# ---------------- ironing histories
def ironing(out, own_files, ccx_files):
    fig, axs = plt.subplots(2, 1, figsize=(6.6, 10.0))
    band = {"v": [(1.0, 2.9e8, 3.3e8), (1.5, 3.60e8, 4.18e8)], "t": [(1.0, 0.60e8, 0.69e8), (1.5, 1.45e8, 1.95e8)]}
    for ax, key, comp, ttl in ((axs[0], "v", "Ry", "Vertical force"), (axs[1], "t", "Rx", "Tangential (pulling) force")):
        for t0, lo, hi in band[key]:
            ax.fill_between([t0 - 0.06, t0 + 0.06], lo / 1e8, hi / 1e8, color="#d9d8d2", zorder=0)
        ax.plot([], [], color="#d9d8d2", lw=8, label="range of four published curves")
        for f, lab, col, ls in own_files:
            if not os.path.exists(R + f): continue
            h = json.load(open(R + f))["hist"]
            ax.plot([q["t"] for q in h], [abs(q[comp]) / 1e8 for q in h], color=col, ls=ls, lw=2, label=lab)
        for f, lab, col, ls in ccx_files:
            if not os.path.exists(R + f): continue
            r = json.load(open(R + f)); h = [q for q in r["hist"] if q["t"] > 0.2]
            ax.plot([q["t"] for q in h], [abs(q[comp]) / 1e8 for q in h], color=col, ls=ls, lw=1.4, label=lab)
            if not r["finished"]:
                ax.plot(h[-1]["t"], abs(h[-1][comp]) / 1e8, "x", color=C["bad"], ms=10, mew=2.5)
        ax.set_xlabel("time (0–1 press down, 1–2 slide)"); ax.set_ylabel("force / 1e8"); ax.set_title(ttl); ax.set_xlim(0.15, 2.02)
    # Code_Aster table
    tt = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.01]
    axs[0].plot(tt, [0.9454, 1.3456, 1.7840, 2.2557, 2.7555, 3.2990, 3.4230], "k+", ms=8, label="Code_Aster table")
    axs[1].plot(tt, [0.1836, 0.2668, 0.3599, 0.4608, 0.5689, 0.6902, 0.8744], "k+", ms=8, label="Code_Aster table")
    axs[1].legend(fontsize=9, loc="upper left"); axs[1].set_ylim(0, 3.1)
    axs[1].plot([], [], "x", color=C["bad"], ms=9, mew=2.5)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)

def ironing_mesh(npz, out):
    d = np.load(npz); x = d["X"] + d["u"].reshape(-1, 2)
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    ax.add_collection(PolyCollection(x[d["Q"]], facecolors=[("#cfe0f5" if b == 0 else "#f6cdbf") for b in d["body"]], edgecolors="#52514e", linewidths=0.25))
    ax.set_aspect("equal"); ax.set_xlim(-0.2, 12.4); ax.set_ylim(-4.1, 1.4); ax.grid(False); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_title("Ironing at mid-stroke (own solver)", fontsize=11)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)

def ccx_bars(out, rows):
    fig, ax = plt.subplots(figsize=(6.6, 6.4))
    labs, vals, cols = [], [], []
    for lab, f, col in rows:
        if not os.path.exists(R + f): continue
        r = json.load(open(R + f))
        if "hist" in r and r.get("t_reached", 2.0) >= 1.6 or ("h" in r and "n2" in r):
            s = RI.summary(r["hist"]); vals.append(s["Ft_mid"] / 1e8); labs.append(lab); cols.append(col)
        else:
            vals.append(0.0); labs.append(lab + f" FAILS t={r['t_reached']:.2f}"); cols.append(C["bad"])
    y = np.arange(len(labs))[::-1]
    ax.barh(y, vals, color=cols, height=0.62)
    for yi, v in zip(y, vals):
        ax.text(v + 0.02 if v > 0 else 0.02, yi, f"{v:.2f}" if v > 0 else "no result", va="center", fontsize=10)
    ax.axvspan(1.45, 1.95, color="#d9d8d2", zorder=0); ax.text(1.46, len(labs) - 0.45, "published range", fontsize=9.5, color="#52514e")
    ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=9); ax.set_xlabel("tangential force during sliding / 1e8"); ax.set_xlim(0, 2.2); ax.grid(axis="y", visible=False)
    ax.set_title("Same problem: CalculiX answer by setting", fontsize=12)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)

if __name__ == "__main__":
    lv = 3 if os.path.exists(R + "r2_mindlin_L3_nq15_own.npz") else 2
    mindlin(R + f"r2_mindlin_L{lv}_nq15_own.npz", R + "r2_mindlin_SURFACETOSURFACE_L2.npz", R + "r2_mindlin.png")
    ironing(R + "r2_ironing_forces.png",
            [("r2_ironing_own_h0.05.json", "own solver, h = 0.05", C["blue"], "-"), ("r2_ironing_own_h0.1_half.json", "own solver, h = 0.1", C["aqua"], "-")],
            [("r2_ironing_ccx_iron_SURFACETOSURFACE_h0.1.json", "CalculiX, increments 0.005 (fails)", C["bad"], "-"),
             ("r2_ironing_ccx_iron_SURFACETOSURFACE_h0.1_dm0.001.json", "CalculiX, increments 0.001", C["orange"], "-"),
             ("r2_ironing_ccx_iron_SURFACETOSURFACE_h0.1_dm0.0005.json", "CalculiX, increments 0.0005", C["yellow"], "-"),
             ("r2_ironing_ccx_iron_SURFACETOSURFACE_h0.1_k5.json", "CalculiX, penalty 10× softer", C["violet"], "--")])
    f = "r2_ironing_own_h0.05_t1.5.npz" if os.path.exists(R + "r2_ironing_own_h0.05_t1.5.npz") else "r2_ironing_own_h0.1_t1.5.npz"
    ironing_mesh(R + f, R + "r2_ironing_mesh.png")
    ccx_bars(R + "r2_ccx_settings.png", [
        ("own solver, h = 0.1", "r2_ironing_own_h0.1_half.json", C["blue"]),
        ("ccx h=0.2, K=50E/h,\nsteps 0.005", "r2_ironing_ccx_iron_SURFACETOSURFACE_h0.2_probe.json", C["orange"]),
        ("ccx h=0.2, K=5E/h,\nsteps 0.005", "r2_ironing_ccx_iron_SURFACETOSURFACE_h0.2_k5.json", C["orange"]),
        ("ccx h=0.2, K=500E/h,\nsteps 0.005", "r2_ironing_ccx_iron_SURFACETOSURFACE_h0.2_k500.json", C["orange"]),
        ("ccx h=0.1, K=50E/h,\nsteps 0.005", "r2_ironing_ccx_iron_SURFACETOSURFACE_h0.1.json", C["orange"]),
        ("ccx h=0.1, K=5E/h,\nsteps 0.005", "r2_ironing_ccx_iron_SURFACETOSURFACE_h0.1_k5.json", C["orange"]),
        ("ccx h=0.1, K=50E/h,\nsteps 0.001", "r2_ironing_ccx_iron_SURFACETOSURFACE_h0.1_dm0.001.json", C["orange"]),
        ("ccx h=0.1, K=50E/h,\nsteps 0.0005", "r2_ironing_ccx_iron_SURFACETOSURFACE_h0.1_dm0.0005.json", C["orange"]),
        ("ccx h=0.1,\nnode-to-surface", "r2_ironing_ccx_iron_NODETOSURFACE_h0.1.json", C["orange"]),
        ("ccx h=0.05, K=50E/h,\nsteps 0.005", "r2_ironing_ccx_iron_SURFACETOSURFACE_h0.05_graded.json", C["orange"]),
        ("ccx h=0.05, K=50E/h,\nsteps 0.001", "r2_ironing_ccx_iron_SURFACETOSURFACE_h0.05_dm0.001.json", C["orange"])])
    print("ok")
