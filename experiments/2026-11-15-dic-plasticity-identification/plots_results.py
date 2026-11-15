"""Result figures: measured vs predicted strain fields, force curves, hardening curves, identifiability."""
import os, sys, json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import data, femu
from plotlib import *

FD = "results/cache/fields"
ST_LABEL = {0: "30 %", 1: "60 %", 2: "90 %"}

def _outline(ax, spec, ylo, yhi):
    bo, bh = data.load_geometry(data.geom_key(spec))
    ax.plot(bo[:, 0], bo[:, 1], color=C_MUTED, lw=0.7)
    for h in bh:
        ax.plot(h[:, 0], h[:, 1], color=C_MUTED, lw=0.7)
    ax.set_ylim(ylo - 4, yhi + 4); ax.set_aspect("equal"); ax.axis("off")

def fig_fields(spec, model, info, out, st=2, comp=1, mesh="cal_n1", title=None, model_label="model"):
    D = np.load(f"{FD}/{spec}__DIC.npz"); M = np.load(f"{FD}/{spec}__{model}__{mesh}.npz")
    pts = D["pts"]; Em = D["E"][st, comp]; Es = M["E"][st, comp]
    vmax = float(np.percentile(Em, 99.5)); dv = 0.25 * vmax
    w = (pts[:, 0].max() - pts[:, 0].min()); hgt = info["yhi"] - info["ylo"] + 8
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 7.2 / 3 * hgt / (w + 14) + 1.3))
    kw = dict(s=0.5, marker="s", linewidths=0, rasterized=True)
    for ax, val, cm, lim, ttl in ((axs[0], Em, SEQ, (0, vmax), "measured (DIC)"), (axs[1], Es, SEQ, (0, vmax), model_label),
                                  (axs[2], Es - Em, DIV, (-dv, dv), "difference")):
        sc = ax.scatter(pts[:, 0], pts[:, 1], c=val, cmap=cm, vmin=lim[0], vmax=lim[1], **kw)
        _outline(ax, spec, info["ylo"], info["yhi"]); ax.set_title(ttl, fontsize=9, color=INK)
        cb = fig.colorbar(sc, ax=ax, orientation="horizontal", pad=0.02, fraction=0.05, aspect=18)
        cb.ax.tick_params(labelsize=7); cb.outline.set_visible(False)
    comp_name = ["sideways strain Exx", "stretch along the pull, Eyy", "shear strain Exy"][comp]
    fig.suptitle(title or f"{spec}: {comp_name} at {ST_LABEL[st]} of the way to peak load", fontsize=10, color=INK, y=1.0)
    fig.savefig(out, dpi=170, facecolor=SURFACE, bbox_inches="tight"); plt.close(fig)

def fig_force(spec, res, models, out, title=None):
    """models: list of (key, label, colour, linestyle)."""
    info = res["info"]; d = np.load(os.path.join(data.CACHE, f"{spec}.npz"))
    fig, ax = plt.subplots(figsize=(4.6, 3.7)); style(ax)
    ok = np.isfinite(d["ext"]) & np.isfinite(d["force"])
    ax.plot(d["ext"][ok], d["force"][ok] / 1e3, color=INK, lw=1.3, alpha=0.75, label="measured")
    x = np.array(info["ext_st"])
    for key, label, col, ls in models:
        m = res["models"].get(key)
        if not m or "fail" in m: continue
        ax.plot(x[1:], np.array(m["F"])[1:] / 1e3, color=col, lw=1.8, ls=ls, label=label)
    ax.axvspan(0.9 * info["xF"], d["ext"][ok].max(), color=GRID, alpha=0.6, lw=0)
    ax.text(0.9 * info["xF"] + 0.15, 0.42 * info["Fmax"] / 1e3, "not\npredicted\n(no tearing\nin model)", fontsize=8, color=INK2, va="bottom")
    ax.set_xlabel("grip extension, mm"); ax.set_ylabel("force, kN"); ax.set_xlim(0, None); ax.set_ylim(0, None)
    ax.legend(frameon=False, fontsize=9, loc="lower center")
    ax.set_title(title or f"{spec}: load against extension", fontsize=10.5, color=INK, loc="left")
    fig.savefig(out, dpi=170, facecolor=SURFACE, bbox_inches="tight"); plt.close(fig)

def fig_hardening(curves, out, alpha_seen=None, ep_uni=None, title="Identified hardening curves"):
    """curves: list of (label, p, colour, ls)."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2)); style(ax)
    a = np.linspace(0, 0.9, 300)
    for label, p, col, ls in curves:
        ax.plot(a, femu.sigy(p, a), color=col, lw=2, ls=ls, label=label)
    if ep_uni:
        ax.axvline(ep_uni, color=C_MUTED, lw=1, ls=":"); ax.text(ep_uni + 0.01, 300, "tension test\nends here", fontsize=7.5, color=INK2)
    if alpha_seen:
        ax.axvline(alpha_seen, color=C_MUTED, lw=1, ls="--"); ax.text(alpha_seen + 0.01, 420, "95 % of plate material\nstays below this", fontsize=7.5, color=INK2)
    ax.set_xlabel("plastic strain (true)"); ax.set_ylabel("yield stress, MPa"); ax.set_xlim(0, 0.9)
    ax.legend(frameon=False, fontsize=8, loc="lower right"); ax.set_title(title, fontsize=10, color=INK, loc="left")
    fig.savefig(out, dpi=170, facecolor=SURFACE, bbox_inches="tight"); plt.close(fig)
