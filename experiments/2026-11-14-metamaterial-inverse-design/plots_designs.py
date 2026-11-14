"""Figures for rung 1: curves (target vs design solver vs CalculiX), shapes, optimisation history, mesh convergence."""
import json
import numpy as np
from plotlib import *
import design, fem

TITLE = {"T1": "T1 — monotonic stiffening (J-curve)", "T2": "T2 — plateau", "T3": "T3 — snap-through (non-monotonic)"}
S = fem.Solver(nx=40, ny=2, Lh=design.LH)
e = design.EPS
for T in ("T1", "T2", "T3"):
    v = json.load(open(f"results/verify_{T}.json")); o = json.load(open(f"results/opt_{T}.json")); dsn = json.load(open(f"results/design_{T}.json"))
    tgt = np.array(v["target_sigma"]); p = np.array(dsn["p"]); p0 = np.array(dsn["p0"])
    sig0 = np.array(o[dsn["start"]]["history"][0].get("sigma", [])) if False else None
    F0, _, _ = S.path(p0, design.DS); sig0 = F0 * design.KPA
    # ---- curves
    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=160)
    ax.plot(e, sig0, color=FRAME, lw=1.6, label="starting design (design solver)")
    ax.plot(e, tgt, color=INK, lw=1.4, ls="--", label="target", zorder=5)
    ax.plot(e, v["jax_40x2"], color=BLUE, lw=2.6, label="optimised — design solver (JAX)")
    ax.plot(e[::2], np.array(v["ccx_t3"])[::2], "o", color=ORANGE, ms=5, mec=SURF, mew=0.8, label="optimised — CalculiX re-solve", zorder=6)
    ax.set_xlim(0, 0.5); ax.set_ylim(min(0, tgt.min()) - 0.03, tgt.max() * 1.12)
    ax.set_xlabel("compressive strain  d / H"); ax.set_ylabel("stress  F / W  (kPa)")
    vd = v["verdict"]
    ax.set_title(f"{TITLE[T]}\nCalculiX error {100*vd['nrmse']:.2f} % of range (tolerance {100*vd['tol']:.0f} %) — {'PASS' if vd['pass'] else 'FAIL'}", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(f"results/curves_{T}.png"); plt.close(fig)
    # ---- shapes: start, final, final deformed at 3 strains
    _, states, _ = deformed_states(p, [2.5, 7.5, 12.5], S)
    fig, axs = plt.subplots(5, 1, figsize=(5.2, 9.6), dpi=160)
    draw_cell(axs[0], p0, color=FRAME); axs[0].set_title("starting design", fontsize=9, loc="left")
    draw_cell(axs[1], p, color=BLUE); axs[1].set_title("optimised design (undeformed)", fontsize=9, loc="left")
    for a, d in zip(axs[2:], (2.5, 7.5, 12.5)):
        draw_cell(a, p, d=d, u_nodes=states[d], S=S, color=BLUE)
        a.set_title(f"compressed to strain {d/design.H_CELL:.1f}  (shuttle down {d:g} mm)", fontsize=9, loc="left")
    fig.suptitle(TITLE[T], fontsize=10, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(); fig.savefig(f"results/shapes_{T}.png"); plt.close(fig)

# ---- optimisation history
fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.6), dpi=150, sharey=True)
cols = {"sag3": VIOLET, "flat": YELLOW, "arch3": AQUA, "arch6": BLUE}; lss = {"sag3": ":", "flat": "-.", "arch3": "--", "arch6": "-"}
for a, T in zip(axs, ("T1", "T2", "T3")):
    o = json.load(open(f"results/opt_{T}.json"))
    for s, r in o.items():
        h = np.minimum.accumulate([x["nrmse"] for x in r["history"]])
        a.semilogy(np.arange(1, len(h) + 1), h, lss[s], color=cols[s], lw=1.8, label=f"start: {s}")
    tol = 0.08 if T == "T3" else 0.05
    a.axhline(tol, color=MUTED, lw=1, ls=(0, (1, 2))); a.text(60, tol * 1.12, "tolerance", fontsize=7.5, color=MUTED, ha="right")
    a.set_title(TITLE[T].split(" — ")[0] + " — " + TITLE[T].split(" — ")[1], fontsize=9.5); a.set_xlabel("solver evaluations")
axs[0].set_ylabel("best curve error so far (NRMSE)"); axs[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig("results/opt_history.png"); plt.close(fig)

# ---- mesh convergence
fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.6), dpi=150, sharey=True)
for a, T in zip(axs, ("T1", "T2", "T3")):
    v = json.load(open(f"results/verify_{T}.json")); tgt = np.array(v["target_sigma"]); rng = tgt.max() - tgt.min()
    ref = np.array(v["ccx_t6"], float)
    for key, lab, col, ls in [("jax_40x2", "JAX 40×2 (design mesh)", BLUE, "-"), ("jax_80x2", "JAX 80×2", AQUA, "--"), ("jax_80x4", "JAX 80×4", VIOLET, "-."), ("ccx_t3", "CalculiX t/3 (verifier)", ORANGE, ":")]:
        a.plot(e, 100 * (np.array(v[key], float) - ref) / rng, ls, color=col, lw=1.6, label=lab)
    a.axhline(0, color=MUTED, lw=0.8); a.set_title(TITLE[T].split(" — ")[0] + ": difference to finest CalculiX mesh", fontsize=9); a.set_xlabel("strain")
axs[0].set_ylabel("% of target range"); axs[0].legend(fontsize=7.5)
fig.tight_layout(); fig.savefig("results/mesh_convergence.png"); plt.close(fig)
print("ok")
