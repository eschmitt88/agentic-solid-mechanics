"""Figures for rung 2 (programmed bistability)."""
import json
import numpy as np
from plotlib import *
import bistable, design, fem

v = json.load(open("results/verify_bistable.json")); dsn = json.load(open("results/design_bistable.json"))
p = np.array(v["p"]); d = np.array(v["d"]); tg = v["target"]
Fj = np.array(v["jax_40x2"]["F"]); fj = v["jax_40x2"]["features"]
_c = v["ccx_t3"] if v["ccx_t3"]["F"] else (v.get("extra", {}).get("ccx_t4") or v["ccx_t6"])
Fc = np.array(_c["F"]) if _c["F"] else None; fc = _c["features"]
fig, axs = plt.subplots(2, 1, figsize=(6.4, 7.4), dpi=160, sharex=True)
a = axs[0]
a.axhline(0, color=MUTED, lw=0.8)
a.plot(np.r_[0, d], np.r_[0, Fj] * 1e3, color=BLUE, lw=2.6, label="design solver, pushing down")
bd = np.array(v["jax_40x2_backward"]["d"]); bF = np.array(v["jax_40x2_backward"]["F"])
a.plot(bd, bF * 1e3, color=AQUA, lw=1.4, ls="--", label="design solver, coming back up")
if Fc is not None:
    a.plot(d[::3], Fc[::3] * 1e3, "o", color=ORANGE, ms=4.5, mec=SURF, mew=0.7, label="CalculiX re-solve")
a.plot([0, 3.5], [tg["F_pk"] * 1e3] * 2, color=INK, lw=1.2, ls=":"); a.text(3.7, tg["F_pk"] * 1e3, "target peak force", fontsize=8, va="center")
a.plot([tg["d_2"]], [0], "D", color=INK, ms=7, mfc="none", mew=1.4); a.annotate("target second\nstable position", (tg["d_2"], 0), (tg["d_2"] - 3.6, 14), fontsize=8, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
ref = fc or fj
if ref:
    a.plot([0, ref["d_2"]], [0, 0], "o", color=INK, ms=6); 
    a.fill_between(np.r_[0, d], np.r_[0, Fc if Fc is not None else Fj] * 1e3, 0, where=(np.r_[0, d] >= ref["d_u"]) & (np.r_[0, d] <= ref["d_2"]), color=ORANGE, alpha=0.18, lw=0)
    a.text(0.5 * (ref["d_u"] + ref["d_2"]), -6.5, "return\nbarrier", fontsize=8, ha="center", color=MUTED)
a.set_ylim(min(-30, Fj.min() * 1e3 * 1.2), max(45, tg["F_pk"] * 1.25e3)); a.set_ylabel("force (mN per mm depth)")
a.set_title("Bistable cell: force through the full loop"); a.legend(fontsize=8, loc="lower left")
a = axs[1]
U = np.r_[0, np.cumsum(0.5 * (np.r_[0, Fj][1:] + np.r_[0, Fj][:-1]) * np.diff(np.r_[0, d]))]
a.plot(np.r_[0, d], U * 1e3, color=BLUE, lw=2.4)
if ref:
    for dd, lab in ((0.0, "stable 1"), (ref["d_2"], "stable 2")):
        a.plot([dd], [np.interp(dd, np.r_[0, d], U) * 1e3], "o", color=INK, ms=6); a.annotate(lab, (dd, np.interp(dd, np.r_[0, d], U) * 1e3), (dd + 0.3, np.interp(dd, np.r_[0, d], U) * 1e3 - 22), fontsize=8)
a.set_ylim(-30, max(U.max() * 1e3 * 0.6, 150)); a.set_xlim(0, 13)
a.set_xlabel("shuttle travel d (mm)"); a.set_ylabel("stored energy (µJ per mm depth)"); a.set_title("Energy landscape: two wells")
fig.tight_layout(); fig.savefig("results/bistable_curves.png"); plt.close(fig)

S = fem.Solver(nx=40, ny=2, Lh=design.LH)
dl = [ref["d_u"], ref["d_2"]] if ref else [5.0, 10.0]
_, states, _ = deformed_states(p, dl, S)
fig, axs = plt.subplots(4, 1, figsize=(5.2, 7.8), dpi=160)
draw_cell(axs[0], np.array(dsn["p0"]), color=FRAME); axs[0].set_title("starting design of the final attempt", fontsize=9, loc="left")
draw_cell(axs[1], p, color=BLUE); axs[1].set_title("optimised — stable state 1 (as made)", fontsize=9, loc="left")
draw_cell(axs[2], p, d=dl[0], u_nodes=states[float(sorted(dl)[0])], S=S, color=BLUE); axs[2].set_title(f"tipping point (unstable), d = {dl[0]:.1f} mm", fontsize=9, loc="left")
draw_cell(axs[3], p, d=dl[1], u_nodes=states[float(sorted(dl)[1])], S=S, color=BLUE); axs[3].set_title(f"stable state 2, d = {dl[1]:.1f} mm (holds with no force)", fontsize=9, loc="left")
fig.tight_layout(); fig.savefig("results/bistable_shapes.png"); plt.close(fig)
print("ok")
