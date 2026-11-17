"""[dehom-dcbeam: copied unchanged from trial 14; GIRDER-specific, NOT adapted to hf.PROBLEM = DCBEAM.]
Figures for trial 13 (results/fig_*.png). Palette and mark rules: dataviz skill reference
palette (categorical slots in fixed order, validated light-mode), single axis per chart,
2 px lines, direct end labels, recessive grid. Run after analyze.py.
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ea  # noqa
import analyze as A  # noqa
import run_arm as ra  # noqa

RES = HERE / "results"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# categorical slots, fixed order: entity -> slot never changes between charts
COL = {"FULL": "#2a78d6", "NOVAE": "#eb6834", "NOMUT": "#1baf7a", "NOSEL": "#eda100", "MORELF": "#e87ba4",
       "gen0": "#52514e", "SIMP": "#6250d6"}
LAB = {"FULL": "FULL (paper's method)", "NOVAE": "no VAE (PCA-space crossover)", "NOMUT": "no mutation",
       "NOSEL": "no selection", "MORELF": "more LF designs, no EA", "gen0": "gen 0 (LF designs)", "SIMP": "SIMP"}


def style(ax, title, xl, yl):
    ax.set_facecolor(SURF)
    ax.figure.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_title(title, loc="left", color=INK, fontsize=11.5, fontweight="bold")
    ax.set_xlabel(xl, color=INK2, fontsize=9.5)
    ax.set_ylabel(yl, color=INK2, fontsize=9.5)


def hv_curve(obj, p, setup):
    """Archive HV(r=1.1) after each generation, recomputed from archive.npz (gen <= last)."""
    F0, cv0, norm, hv = setup
    a = np.load(p / "archive.npz")
    gmax = int(a["gen"].max())
    xs, ys = [], []
    for g in range(0, gmax + 1):
        m = a["gen"] <= g
        xs.append(int(m.sum())); ys.append(hv(a["F"][m], a["CV"][m]))
    return np.array(xs), np.array(ys)


def morelf_curve(obj, setup, kind="raw", step=550):
    F0, cv0, norm, hv = setup
    f = RES / "morelf" / f"eval_{obj}.npz"
    if not f.exists() or f"{kind}_vf" not in np.load(f).files:
        return None
    base = A.gen0_objs(obj, "raw")
    d = np.load(f)
    ex = [dict(vf=d[f"{kind}_vf"][i], connected=bool(d[f"{kind}_connected"][i]), compliance=d[f"{kind}_compliance"][i],
               blf=d[f"{kind}_blf"][i]) for i in range(len(d[f"{kind}_vf"]))]
    F, cv = ra.objectives(base + ex, obj, A.cref(obj))
    xs = np.arange(len(base), len(F) + 1, step)
    return xs, np.array([hv(F[:n], cv[:n]) for n in xs])


def fig_hv(obj):
    setup = A.setup(obj)
    F0, cv0, norm, hv = setup
    h0 = hv(F0, cv0)
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ends = []
    for arm in ("FULL", "NOVAE", "NOMUT", "NOSEL"):
        for i, p in enumerate(A.arm_runs(obj, arm)):
            x, y = hv_curve(obj, p, setup)
            g = 100 * (y / h0 - 1)
            ax.plot(x, g, color=COL[arm], lw=2 if i == 0 else 1.2, alpha=1 if i == 0 else 0.55,
                    label=LAB[arm] if i == 0 else None)
            if i == 0:
                ends.append((x[-1], g[-1], arm))
    m = morelf_curve(obj, setup)
    if m is not None:
        g = 100 * (m[1] / h0 - 1)
        ax.plot(m[0], g, color=COL["MORELF"], lw=2, ls="--", label=LAB["MORELF"])
        ends.append((m[0][-1], g[-1], "MORELF"))
    ax.axhline(0, color=INK2, lw=0.8)
    for x, y, arm in ends:
        ax.annotate(f"{y:+.0f}%", (x, y), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8.5, color=INK)
    ttl = "Stiffness" if obj == "stiff" else "Buckling (stiffness-constrained)"
    style(ax, f"{ttl}: hypervolume gain over gen 0 vs. HF evaluations",
          "HF evaluations (480 = gen 0; 110 per generation)", "HV gain vs gen 0, r = 1.1 (%)")
    leg = ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    for t in leg.get_texts():
        t.set_color(INK)
    fig.text(0.01, 0.005, "Thick line = seed 0, thin = seeds 1-2. Dashed: same budget spent on extra LF designs.",
             fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(RES / f"fig_hv_{obj}.png", dpi=160)
    plt.close(fig)


def front(F, cv):
    ok = cv <= 0
    Fk = F[ok]
    r = ea.nondominated_sort(Fk)
    P = Fk[r == 0]
    return P[np.argsort(P[:, 0])]


def fig_fronts(obj):
    setup = A.setup(obj)
    F0, cv0, norm, hv = setup
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    sign = 1 if obj == "stiff" else -1
    P0 = front(F0, cv0)
    ax.step(P0[:, 0], sign * P0[:, 1], where="post", color=COL["gen0"], lw=2, label=LAB["gen0"])
    p = RES / "arms" / f"{obj}_FULL_s0" / "archive.npz"
    if p.exists():
        a = np.load(p); P = front(a["F"], a["CV"])
        ax.step(P[:, 0], sign * P[:, 1], where="post", color=COL["FULL"], lw=2, label=LAB["FULL"] + ", seed 0")
    f = RES / "morelf" / f"eval_{obj}.npz"
    if f.exists():
        base = A.gen0_objs(obj, "raw"); d = np.load(f)
        ex = [dict(vf=d["raw_vf"][i], connected=bool(d["raw_connected"][i]), compliance=d["raw_compliance"][i],
                   blf=d["raw_blf"][i]) for i in range(len(d["raw_vf"]))]
        Fm, cvm = ra.objectives(base + ex, obj, A.cref(obj))
        P = front(Fm, cvm)
        ax.step(P[:, 0], sign * P[:, 1], where="post", color=COL["MORELF"], lw=2, ls="--", label=LAB["MORELF"])
    s = json.load(open(RES / "simp" / "simp_eval.json"))["designs"]
    xs = [d["vf"] for d in s]; ys = [d["compliance"] if obj == "stiff" else d["blf"] for d in s]
    ax.plot(xs, ys, "o", ms=8, color=COL["SIMP"], mec=SURF, mew=2, label=LAB["SIMP"] + " (solid, gradient)")
    if obj == "stiff":
        ax.set_yscale("log")
        style(ax, "Stiffness: Pareto fronts (lower-left is better)", "volume fraction of design region",
              "compliance (log)")
        ax.set_xlim(0.2, 0.6)
    else:
        style(ax, "Buckling: Pareto fronts (upper-left is better)", "volume fraction of design region",
              "first buckling load factor")
        ax.set_xlim(0.2, 0.6)
    leg = ax.legend(frameon=False, fontsize=8.5)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.tight_layout()
    fig.savefig(RES / f"fig_fronts_{obj}.png", dpi=160)
    plt.close(fig)


def fig_calibration():
    g = json.load(open(RES / "grade_ccx.json"))["rows"]
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.2))
    for ax, key, ttl, log in ((axs[0], "compliance", "Compliance", True), (axs[1], "blf", "Buckling load factor", False)):
        x = np.array([r["ccx"][key] for r in g]); y = np.array([r["pixel"][key] for r in g])
        lo, hi = min(x.min(), y.min()) * 0.9, max(x.max(), y.max()) * 1.1
        ax.plot([lo, hi], [lo, hi], color=INK2, lw=1)
        ax.plot(x, y, "o", ms=8, color=COL["FULL"], mec=SURF, mew=2)
        bad = np.argmax(np.abs(np.log(y / x)))
        ax.annotate(g[bad]["key"].replace("|", " ").replace("buck", "B").replace("stiff", "S"), (x[bad], y[bad]),
                    xytext=(6, -12), textcoords="offset points", fontsize=8, color=INK)
        if log:
            ax.set_xscale("log"); ax.set_yscale("log")
        style(ax, ttl, "CalculiX, body-fitted (grading)", "pixel grid (in the loop)")
    fig.suptitle("Evaluator calibration on 24 front designs (line = perfect agreement)", x=0.01, ha="left",
                 fontsize=11.5, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(RES / "fig_calibration.png", dpi=160)
    plt.close(fig)


def fig_gallery():
    import dehom, hf
    pca = ra.load_pca()
    P = hf.Problem(hf.GIRDER, hf.H_FINE)
    rows = []
    for obj in ("stiff", "buck"):
        for run in ("gen0", f"{obj}_FULL_s0"):
            from rereal import front_Z
            Z, F = front_Z(obj, run)
            i = int(np.argmin(np.abs(F[:, 0] - 0.35)))
            w1, w2, th = ra.fields_from_features(pca.inverse(Z[i]))
            d = dehom.dehomogenise(w1, w2, th)
            s, _ = P.clean(d["phi"])
            rows.append((obj, run, s, F[i]))
    sp = json.load(open(RES / "simp" / "simp_eval.json"))
    simp = np.load(RES / "simp" / "simp_designs.npz")
    k = int(np.argmin(np.abs(simp["vf"] - 0.35)))
    fig, axs = plt.subplots(3, 2, figsize=(10, 7.6))
    for ax, (obj, run, s, f) in zip(axs.ravel()[:4], rows):
        ax.imshow(s, origin="lower", cmap="gray_r", extent=[0, 100, 0, 50], interpolation="nearest")
        val = f"C = {f[1]:.2f}" if obj == "stiff" else f"BLF = {-f[1]:.3f}"
        ax.set_title(f"{'stiffness' if obj == 'stiff' else 'buckling'} · {'gen 0' if run == 'gen0' else 'FULL seed 0'}"
                     f" · vf {f[0]:.3f} · {val}", fontsize=9.5, color=INK, loc="left")
        ax.axis("off")
    ax = axs[2, 0]
    ax.imshow(simp["phi"][k], origin="lower", cmap="gray_r", extent=[0, 100, 0, 50], interpolation="nearest")
    dd = sp["designs"][k]
    ax.set_title(f"SIMP · vf {dd['vf']:.2f} · C = {dd['compliance']:.2f} · BLF = {dd['blf']:.3f}", fontsize=9.5,
                 color=INK, loc="left"); ax.axis("off")
    axs[2, 1].axis("off")
    axs[2, 1].text(0, 0.5, "Girder 100 × 50 mm: spring bearings under\nthe bottom pads, uniform load on the top deck,\n"
                   "Ø12 mm duct with liner at quarter span.\nDesigns nearest vf = 0.35 on each front.",
                   fontsize=9.5, color=INK2, va="center")
    fig.patch.set_facecolor(SURF)
    fig.tight_layout()
    fig.savefig(RES / "fig_gallery.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    for obj in ("stiff", "buck"):
        fig_hv(obj); fig_fronts(obj)
    fig_calibration()
    fig_gallery()
    print("figures written")
