"""All figures of trial 12, generated from results/*.json.  usage: python plots.py [name ...]  (default: every figure
whose inputs exist)."""
import glob
import json
import os
import sys

import numpy as np

import design
import snap
from plotlib import *  # noqa: F401,F403

R = "results/"
J = lambda f: json.load(open(R + f))
ex = lambda f: os.path.exists(R + f)
arr = lambda v: np.array([np.nan if x is None else x for x in v], float)
ARMS = ["B0-legacy", "B0", "A", "A+O", "A+O2", "AM+O2"]
ARM_LABEL = {"B0-legacy": "trial 9 as run\n(old continuation)", "B0": "trial 9 gradient,\nfixed continuation", "B0+O": "+ restarts",
             "A": "fold-aware\ngradient", "A+O": "fold-aware\n+ restarts", "A+O2": "… quicker\nrestarts", "AM+O2": "… + bifurcation\nmargin"}


def band_boxes(ax, tname):
    for lo, hi, lvl, hw in design.SPEC[tname]["bands"]:
        ax.fill_between([lo, hi], lvl - hw, lvl + hw, color=MUTED, alpha=0.13, lw=0)


def fig_targets():
    fig, axs = plt.subplots(1, 4, figsize=(11, 2.9))
    e = np.linspace(0, 0.5, 301)
    for ax, t, ttl in zip(axs, ["H1", "H2", "H4", "T2"], ["H1 foam plateau (Gibson–Ashby)", "H2 two-step staircase (stack)", "H4 expected infeasible", "R robust design on trial 9's T2"]):
        ax.plot(e, design.TARGETS[t](e), color=INK, lw=2)
        band_boxes(ax, t)
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_title(ttl, fontsize=9.5)
        ax.set_xlabel("strain")
    axs[0].set_ylabel("stress (kPa)")
    fig.tight_layout()
    fig.savefig(R + "targets.png", dpi=150)
    plt.close(fig)


def fig_design(tname, vname, title):
    """Shape + target vs design solver vs CalculiX ladder."""
    v = J(f"verify_{vname}.json")
    sp = design.SPEC[tname]
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(10.5, 3.6 if sp["ncell"] == 1 else 4.6), gridspec_kw=dict(width_ratios=[1, 1.35]))
    draw_design(a0, v["p"], sp["nk"], sp["ncell"])
    a0.set_title(title, fontsize=10)
    e = np.array(v["eps"])
    band_boxes(a1, tname)
    a1.plot(e, v["target_sigma"], color=INK, lw=1.4, ls=(0, (4, 2)), label="target")
    a1.plot(e, v["jax_40x2"], color=BLUE, lw=2, label=f"design solver (NRMSE {v['jax_40x2_grade']['nrmse']:.3f})")
    g = v.get("graded_mesh")
    if g:
        a1.plot(e, arr(v["ccx"][g]["sigma"]), color=ORANGE, lw=0, marker="o", ms=3.6, label=f"CalculiX {g.replace('t', 't_min/')} (NRMSE {v['verdict']['nrmse']:.3f})")
    else:
        for k, r in v["ccx"].items():
            a1.plot(e, arr(r["sigma"]), color=ORANGE, lw=0, marker="o", ms=3, label=f"CalculiX {k} (aborted at {r['d_reached']:.1f} mm)")
    a1.axhline(0, color=MUTED, lw=0.6)
    a1.set_xlabel("strain")
    a1.set_ylabel("stress (kPa)")
    a1.set_title(f"{v['verdict']['verdict']} — tolerance NRMSE ≤ {sp['tol']}" + (", grey = band" if sp["bands"] else ""), fontsize=10)
    a1.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(R + f"design_{vname}.png", dpi=150)
    plt.close(fig)


def fig_opt_starts():
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.1))
    for ax, t in zip(axs, ["H1", "H2", "H4"]):
        files = sorted(glob.glob(R + f"opt/{t}__*.json"))
        names, vals = [], []
        for f in files:
            r = json.load(open(f))
            names.append(r["start"])
            vals.append(r["nrmse"])
        if not vals:
            ax.set_title(f"{t}: not run", fontsize=9.5)
            continue
        y = np.arange(len(names))
        ax.barh(y, vals, color=BLUE, height=0.55)
        for yi, v in zip(y, vals):
            ax.text(v, yi, f" {v:.3f}", va="center", fontsize=8.5)
        ax.axvline(design.SPEC[t]["tol"], color=INK, lw=1.2, ls=(0, (4, 2)))
        ax.text(design.SPEC[t]["tol"], len(names) - 0.45, " tolerance", fontsize=8, color=MUTED)
        ax.set_yticks(y, names)
        ax.set_xlim(0, max(vals) * 1.25)
        ax.set_title(f"{t}: optimiser loss per start (NRMSE)", fontsize=9.5)
        ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(R + "opt_starts.png", dpi=150)
    plt.close(fig)


def fig_certificate():
    c = J("certificate.json")["H4"]
    e = design.EPS
    tgt = design.TARGETS["H4"](e)
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(10, 3.2))
    a0.plot(e, tgt, color=INK, lw=1.4, ls=(0, (4, 2)), label="H4 target")
    a0.plot(e, c["closest_admissible_curve"], color=VIOLET, label=f"closest curve any elastic cell could have\n(NRMSE {c['lower_bound_nrmse']:.3f})")
    if ex("verify_H4.json"):
        v = J("verify_H4.json")
        a0.plot(e, v["jax_40x2"], color=BLUE, label=f"best design found (NRMSE {v['jax_40x2_grade']['nrmse']:.3f})")
    a0.axhline(0, color=MUTED, lw=0.6)
    a0.set_xlabel("strain"); a0.set_ylabel("stress (kPa)"); a0.legend(fontsize=8.5); a0.set_title("H4: target vs what energy allows", fontsize=10)
    W = np.concatenate([[0], np.cumsum(0.5 * (np.r_[0, tgt][1:] + np.r_[0, tgt][:-1]) * 0.01)]) * 25.0 * 70.0 / 1000
    a1.plot(np.r_[0, e], W, color=INK)
    a1.fill_between(np.r_[0, e], W, 0, where=W < 0, color=ORANGE, alpha=0.3, lw=0)
    a1.axhline(0, color=MUTED, lw=0.6)
    a1.set_xlabel("strain"); a1.set_ylabel("stored energy the target implies (N·mm/mm)"); a1.set_title("negative stored energy = impossible", fontsize=10)
    fig.tight_layout()
    fig.savefig(R + "certificate_H4.png", dpi=150)
    plt.close(fig)


def fig_robust(tname="T2"):
    r = J(f"robust_{tname}.json")
    sp = design.SPEC[tname]
    e = design.EPS
    fig, axs = plt.subplots(1, 3, figsize=(11.5, 3.5), gridspec_kw=dict(width_ratios=[1, 1, 1.05]))
    for ax, key, col, ttl in [(axs[0], "nominal_design", BLUE, "nominal optimum"), (axs[1], "robust_design", AQUA, "robust optimum")]:
        band_boxes(ax, tname)
        for s in r[key]["copies"]["samples"]:
            ax.plot(e, s["sigma"], color=col, lw=0.6, alpha=0.35)
        ax.plot(e, design.TARGETS[tname](e), color=INK, lw=1.4, ls=(0, (4, 2)))
        ax.set_ylim(0, 1.25)
        ax.set_xlabel("strain")
        ax.set_title(f"{ttl}: 64 perturbed copies\n{100*r[key]['copies']['pass_fraction']:.0f} % inside tolerance (design solver)", fontsize=9.5)
    axs[0].set_ylabel("stress (kPa)")
    ax = axs[2]
    rows = []
    for key, col, lab in [("nominal", BLUE, "nominal"), ("robust", AQUA, "robust")]:
        ds = arr([s["nrmse"] for s in r[key + "_design"]["copies"]["samples"]])
        cc = [json.load(open(f)) for f in sorted(glob.glob(R + f"robust_ccx/{tname}_{key}_*.json"))]
        rows.append((lab + "\ndesign solver, 64", ds, col, "o"))
        if cc:
            rows.append((lab + f"\nCalculiX, {len(cc)}", arr([c["nrmse"] for c in cc]), col, "s"))
    rng = np.random.default_rng(0)
    for i, (lab, v, col, mk) in enumerate(rows):
        v2 = np.where(np.isfinite(v), v, 0.2)
        ax.plot(np.clip(v2, 0, 0.2), i + 0.5 * (rng.random(len(v)) - 0.5), mk, color=col, ms=4, alpha=0.6, mec=SURF, mew=0.5)
    ax.axvline(sp["tol"], color=INK, lw=1.2, ls=(0, (4, 2)))
    ax.set_yticks(range(len(rows)), [r_[0] for r_ in rows], fontsize=8.5)
    ax.set_xlabel("error of each copy (dashed = tolerance)")
    ax.set_title("error of perturbed copies", fontsize=9.5)
    ax.grid(axis="y", visible=False)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(R + f"robust_{tname}.png", dpi=150)
    plt.close(fig)
    # shapes
    fig, ax = plt.subplots(figsize=(6.5, 2.9))
    draw_design(ax, r["p_nominal"], sp["nk"], sp["ncell"], color=BLUE, alpha=0.55, label="nominal optimum")
    draw_design(ax, r["p_robust"], sp["nk"], sp["ncell"], color=AQUA, fill=False, frame=False, lw=1.6, label="robust optimum (outline)")
    ax.legend(fontsize=8.5, loc="lower center", ncol=2)
    fig.tight_layout()
    fig.savefig(R + f"robust_{tname}_shapes.png", dpi=150)
    plt.close(fig)


def snap_table():
    starts = [s for s in snap.snap_starts().keys() if not s.startswith('fresh_')]
    T = {}
    for a in ARMS:
        for s in starts:
            f = R + f"snap/{a}__{s}.json"
            if os.path.exists(f):
                T[(a, s)] = json.load(open(f))
    return starts, T


def fig_snap():
    starts, T = snap_table()
    arms = [a for a in ARMS if any((a, s) in T for s in starts)]
    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    for i, s in enumerate(starts):
        for j, a in enumerate(arms):
            r = T.get((a, s))
            if r is None:
                ax.text(j, i, "not run", ha="center", va="center", fontsize=8, color=MUTED)
                continue
            ok = r["success"]
            rel = r["score"]["rel_err"]
            txt = "not bistable" if rel is None else f"{100*np.max(np.abs(rel)):.1f} %"
            ax.add_patch(plt.Rectangle((j - 0.46, i - 0.42), 0.92, 0.84, color=AQUA if ok else ORANGE, alpha=0.85 if ok else 0.28, lw=0))
            ax.text(j, i, ("pass " if ok else "miss ") + txt + f"\n{r['nfev']} evals", ha="center", va="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(arms)), [ARM_LABEL[a] + f"\n{sum(T[(a, s)]['success'] for s in starts if (a, s) in T)}/{sum((a, s) in T for s in starts)}" for a in arms], fontsize=8.5)
    ax.set_yticks(range(len(starts)), starts, fontsize=8.5)
    ax.set_xlim(-0.5, len(arms) - 0.5); ax.set_ylim(len(starts) - 0.5, -0.5)
    ax.xaxis.tick_top()
    ax.grid(False)
    for sp_ in ax.spines.values():
        sp_.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Snap test: largest feature error of the final design (success = all three ≤ 1 %)", fontsize=10, pad=58)
    fig.tight_layout()
    fig.savefig(R + "snap_matrix.png", dpi=150)
    plt.close(fig)
    # convergence traces
    fig, axs = plt.subplots(1, len(arms), figsize=(2.5 * len(arms), 3.0), sharey=True)
    for ax, a in zip(np.atleast_1d(axs), arms):
        for s in starts:
            r = T.get((a, s))
            if r is None:
                continue
            err = [np.max(np.abs(np.array(h["feat"]) / snap.TGT - 1)) if h["feat"] else np.nan for h in r["history"]]
            best = np.fmin.accumulate(np.where(np.isfinite(err), err, np.inf))
            ax.plot(np.arange(1, len(best) + 1), best, color=AQUA if r["success"] else ORANGE, lw=1.1, alpha=0.8)
        ax.axhline(0.01, color=INK, lw=1, ls=(0, (4, 2)))
        ax.set_yscale("log"); ax.set_ylim(1e-4, 3)
        ax.set_title(ARM_LABEL[a].replace("\n", " "), fontsize=8.5)
        ax.set_xlabel("evaluations")
    np.atleast_1d(axs)[0].set_ylabel("best largest feature error so far")
    fig.tight_layout()
    fig.savefig(R + "snap_traces.png", dpi=150)
    plt.close(fig)


def fig_gradcheck():
    g = J("gradcheck_fold.json")
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(10, 3.4))
    an, fd, anE, fdE, fdJ, anJ = [], [], [], [], [], []
    for name, rec in g.items():
        for k, v in rec.get("fd_dds", {}).items():
            if k.endswith("1e-05") and np.isfinite(v["fd"]):
                an.append(v["an"]); fd.append(v["fd"])
        for k, v in rec["fd_feat"].items():
            if "fd" in v and k.endswith("2e-05"):
                (anE if v["rel"][2] < 1e-3 else anJ).append(v["an"][2])
                (fdE if v["rel"][2] < 1e-3 else fdJ).append(v["fd"][2])
    a0.plot(an, fd, "o", color=BLUE, ms=7, mec=SURF)
    lim = 1.1 * max(np.abs(an + fd))
    a0.plot([-lim, lim], [-lim, lim], color=MUTED, lw=0.8)
    a0.set_xlabel("fold formula  dd_s/dp  (mm per unit parameter)"); a0.set_ylabel("finite difference of the re-solved fold")
    worst = max(abs(f / a - 1) for a, f in zip(an, fd))
    a0.set_title(f"snap position sensitivity: worst mismatch {worst:.0e}", fontsize=10)
    a1.plot(anE, fdE, "o", color=AQUA, ms=7, mec=SURF, label="function continuous: matches")
    lim = 1.15 * max(np.abs(anE + fdE + anJ))
    if anJ:
        a1.plot(anJ, np.clip(fdJ, -2.7 * lim, 2.7 * lim), "s", color=ORANGE, ms=7, mec=SURF, label="probe straddles a branch flip: finite\ndifference = jump / 2h (off scale, clipped)")
    a1.plot([-lim, lim], [-lim, lim], color=MUTED, lw=0.8)
    a1.set_ylim(-3 * lim, 3 * lim) if anJ else None
    a1.set_xlabel("fold-aware gradient  dE_back/dp"); a1.set_ylabel("finite difference, whole pipeline re-run")
    a1.set_title("return-barrier gradient vs independent re-run", fontsize=10)
    a1.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(R + "gradcheck_fold.png", dpi=150)
    plt.close(fig)


def fig_branchflip():
    b = J("branch_flip.json")
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(10, 3.4))
    h = np.array(b["h"])
    a0.plot(h * 1e4, np.array(b["E_new"]) * 1e3, "o-", color=AQUA, ms=5, label="fixed continuation")
    a0.plot(h * 1e4, np.array(b["E_legacy"]) * 1e3, "s-", color=ORANGE, ms=5, lw=1, label="trial 9 continuation")
    a0.set_xlabel("change of one thickness parameter (×1e-4)"); a0.set_ylabel("return barrier E_back (mN·mm/mm)")
    a0.set_title("the objective jumps where the path flips branch", fontsize=10); a0.legend(fontsize=8.5)
    d = np.array(b["d"])
    a1.plot(d, (np.array(b["F_plus"]) - np.array(b["F_minus"])) * 1e3, color=VIOLET)
    a1.axhline(0, color=MUTED, lw=0.6)
    a1.set_xlabel("shuttle travel (mm)"); a1.set_ylabel("force difference (mN per mm depth)")
    a1.set_title("two designs 2e-4 apart: different branch from 3.4 to 7.4 mm", fontsize=10)
    fig.tight_layout()
    fig.savefig(R + "branch_flip.png", dpi=150)
    plt.close(fig)


def fig_bistable(vname="bistable", title="bistable design, fold-aware arm (sits on the bifurcation)"):
    v = J(f"verify_{vname}.json")
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(10.5, 3.5), gridspec_kw=dict(width_ratios=[1, 1.4]))
    draw_design(a0, v["p"], 4, 1)
    a0.set_title(title, fontsize=9.5)
    d = np.array(v["d"])
    a1.plot(d, np.array(v["jax_40x2"]["F"]) * 1e3, color=BLUE, label="design solver")
    for k, r in v["ccx"].items():
        if r["completed"] and k == v.get("graded_mesh"):
            a1.plot(d, arr(r["F"]) * 1e3, "o", color=ORANGE, ms=3, label=f"CalculiX {k.replace('t', 't_min/')}")
        elif not r["completed"]:
            a1.plot(d, arr(r["F"]) * 1e3, "x", color=YELLOW, ms=4, label=f"CalculiX {k.replace('t', 't_min/')} aborted at {r['d_reached']:.2f} mm")
    a1.axhline(0, color=MUTED, lw=0.6)
    a1.axhline(40, color=MUTED, lw=0.8, ls=":"); a1.axvline(10, color=MUTED, lw=0.8, ls=":")
    a1.set_xlabel("shuttle travel (mm)"); a1.set_ylabel("force (mN per mm depth)")
    a1.set_title(f"{v['verdict']['verdict']} — targets: peak 40 mN/mm, second rest position 10 mm", fontsize=10)
    a1.legend(fontsize=8.5)
    fig.tight_layout()
    fig.savefig(R + f"{vname}.png", dpi=150)
    plt.close(fig)


def fig_relaxed():
    files = sorted(glob.glob(R + "relaxed_*.json"))
    files = [f for f in files if json.load(open(f))["kind"] == "curve"] + [f for f in files if json.load(open(f))["kind"] != "curve"]
    n = len(files)
    nr = (n + 2) // 3
    fig, axs = plt.subplots(nr, 3, figsize=(10.5, 3.3 * nr), squeeze=False)
    nice = {"H1": "H1 foam plateau", "T2_nominal": "trial 9's plateau design", "T2_robust": "robust plateau design", "bistable": "bistable (on buckling surface)", "bistable_margin": "bistable (margin variant)"}
    for ax in axs.ravel()[n:]:
        ax.axis("off")
    for ax, f in zip(axs.ravel(), files):
        r = json.load(open(f))
        g = np.array(r["grid"])
        cur = r["kind"] == "curve"
        x = g / g[-1] * 0.5 if cur else g
        sc = 1.0 if cur else 1e3
        for key, col, lab, ls in [("ps", BLUE, "plane strain, rigid frame", "-"), ("3d", ORANGE, "3D, 10 mm deep", "-"), ("ef", VIOLET, "elastic frame", (0, (4, 2)))]:
            if key in r and r[key].get("y"):
                ax.plot(x, arr(r[key]["y"]) * sc, color=col, ls=ls, lw=1.8, label=lab)
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set_title(nice.get(r["name"], r["name"]), fontsize=9.5)
        ax.set_xlabel("strain" if cur else "travel (mm)")
        ax.set_ylabel("stress (kPa)" if cur else "force (mN/mm)")
    axs[0][0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(R + "relaxed.png", dpi=150)
    plt.close(fig)


def fig_schematic():
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    import fem
    draw_design(axs[0], fem.cosine_params(5.0, 1.6, 6, 6), 6, 1)
    axs[0].annotate("", xy=(30, 0.5), xytext=(30, 9.5), arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.4))
    axs[0].text(33, 7.5, "push", fontsize=9)
    axs[0].set_title("one cell: rubber beam between stiff walls", fontsize=10)
    p2 = np.concatenate([fem.cosine_params(4.0, 1.2, 6, 6), fem.cosine_params(6.0, 2.0, 6, 6)])
    draw_design(axs[1], p2, 6, 2)
    axs[1].set_title("two-cell stack (H2): cell B rides on cell A's shuttle", fontsize=10)
    axs[1].text(-4, -14, "A", fontsize=10, color=INK); axs[1].text(-4, 11, "B", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(R + "cell_schematic.png", dpi=150)
    plt.close(fig)


FIGS = {
    "targets": (fig_targets, []), "schematic": (fig_schematic, []), "opt_starts": (fig_opt_starts, []), "certificate": (fig_certificate, ["certificate.json"]),
    "H1": (lambda: fig_design("H1", "H1", "H1 design (13 parameters)"), ["verify_H1.json"]),
    "H2": (lambda: fig_design("H2", "H2", "H2 design (two cells, 26 parameters)"), ["verify_H2.json"]),
    "H4": (lambda: fig_design("H4", "H4", "H4: best design found"), ["verify_H4.json"]),
    "T2rob": (lambda: fig_design("T2", "T2_robust", "T2 robust design"), ["verify_T2_robust.json"]),
    "robust": (fig_robust, ["robust_T2.json"]), "snap": (fig_snap, []), "gradcheck": (fig_gradcheck, ["gradcheck_fold.json"]),
    "branchflip": (fig_branchflip, ["branch_flip.json"]), "bistable": (fig_bistable, ["verify_bistable.json"]), "relaxed": (fig_relaxed, []),
    "bistable_margin": (lambda: fig_bistable("bistable_margin", "bistable design, margin arm (kept off the bifurcation)"), ["verify_bistable_margin.json"]),
}

if __name__ == "__main__":
    for name in (sys.argv[1:] or FIGS):
        fn, need = FIGS[name]
        if all(ex(f) for f in need):
            try:
                fn()
                print("ok", name)
            except Exception as e:
                import traceback
                print("FAILED", name, repr(e)); traceback.print_exc()
        else:
            print("skip", name, "(inputs missing)")
