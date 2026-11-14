"""Figures for rung 0 (Qiu validation) and the gradient check. Reads results/*.json, *.npz only."""
import json
import numpy as np
from plotlib import *
import fem, qiu

# ---------------- rung 0
z = np.load("results/rung0_curves.npz")
v = json.load(open("results/rung0_qiu_validation.json"))
l, t, h = 100.0, 1.0, 6.0
c = fem.E_PS * t**3 / 12 * h / l**3
D, Fs = qiu.series_curve(h / t)
ref = v["qiu_eq_2_47_2_48"]
fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=150)
ax.axhline(0, color=MUTED, lw=0.8)
ax.plot(z["ccx_perfect_d"], z["ccx_perfect_F"] * 1e3, color=ORANGE, lw=1.4, ls=":", label="CalculiX, perfect geometry: stays on the unstable branch, then fails")
ax.plot(D * h, Fs * c * 1e3, color=INK, lw=1.3, ls="--", label="Qiu theory, modal series at Q = 6")
ax.plot(z["jax_100x2_d"], z["jax_100x2_F"] * 1e3, color=BLUE, lw=2.4, label="design solver (JAX, 100×2 Q9)")
ax.plot(z["ccx_imp_lc_t5_d"], z["ccx_imp_lc_t5_F"] * 1e3, color=ORANGE, lw=1.4, label="CalculiX CPE6, 1 %-of-thickness imperfection")
ax.plot([ref["d_top"], ref["d_bot"]], [ref["f_top"] * 1e3, -ref["f_bot"] * 1e3], "s", color=INK, ms=8, mfc="none", mew=1.5, label="Qiu closed form (740 / 370 · EIh/l³, large-Q limit)")
ax.set_ylim(-0.95, 2.4); ax.set_xlim(0, 12.9)
ax.set_xlabel("centre displacement d (mm)"); ax.set_ylabel("force (mN per mm depth)")
ax.set_title("Rung 0 — cosine bistable beam (l = 100, t = 1, h = 6 mm)")
ax.legend(fontsize=7.6, loc="upper center")
fig.tight_layout(); fig.savefig("results/rung0_qiu_validation.png"); plt.close(fig)

# ---------------- gradient check
g = json.load(open("results/gradcheck.json"))
f = g["fine"]; d = np.array(f["d"]); J = np.array(f["J"]); dc = f["d_c"]; F = np.array(f["F"]); lam = np.array(f["lam_min"])
near = np.abs(d - dc) < 0.35
fig, axs = plt.subplots(2, 2, figsize=(9.6, 7.2), dpi=150)
a = axs[0, 0]
a.plot(d[near], F[near] * 1e3, "-o", ms=3.5, color=BLUE); a.axvline(dc, color=MUTED, ls=":", lw=1)
a.set_title("Force has a kink at the bifurcation"); a.set_xlabel("shuttle displacement d (mm)"); a.set_ylabel("force (mN/mm)")
a.annotate("mode-3 buckling\nstarts here", (dc, F[near].max() * 1e3), xytext=(dc + 0.06, F[near].min() * 1e3 + 0.25), fontsize=8, color=MUTED)
a = axs[0, 1]
a.plot(d[near], lam[near] * 1e6, "-o", ms=3.5, color=VIOLET); a.axvline(dc, color=MUTED, ls=":", lw=1); a.axhline(0, color=MUTED, lw=0.8)
a.set_title("Tangent stiffness goes singular there"); a.set_xlabel("shuttle displacement d (mm)"); a.set_ylabel("lowest tangent eigenvalue (×10⁻⁶)")
a = axs[1, 0]
names = {0: ("apex height a1", BLUE, "-"), 4: ("thickness log t0", AQUA, "--"), 6: ("thickness mode b2", YELLOW, "-."), 1: ("symmetry-breaking a2", ORANGE, ":")}
for j, (nm, col, ls) in names.items():
    fd = np.array(f["fd"][str(j)]["1e-06"])
    err = np.abs(fd - J[:, j]) / np.max(np.abs(J[:, j])) + 1e-10
    a.semilogy(d[near] - dc, err[near], ls, marker="o", ms=3, color=col, lw=1.5, label=nm)
a.axvline(0, color=MUTED, ls=":", lw=1); a.set_ylim(1e-10, 1)
a.set_title("Adjoint vs finite difference (same branch)"); a.set_xlabel("d − d_c (mm)"); a.set_ylabel("|adjoint − FD| / max|adjoint|")
a.legend(fontsize=7.5, ncol=2, loc="upper left")
a = axs[1, 1]
m = d > dc - 0.31
a.plot(d[m] - dc, J[m, 1], "-o", ms=3, color=ORANGE, label="adjoint on the branch the solver took")
a.plot(d[m] - dc, np.array(f["fd_restart"]["1"])[m], "--s", ms=3, color=INK, lw=1.2, label="central FD when the path is re-run from zero")
a.set_xscale("symlog", linthresh=0.4); a.set_xlim(-0.32, 4.5)
a.set_title("dF/da2: diverges, and a re-run sees zero"); a.set_xlabel("d − d_c (mm)"); a.set_ylabel("dF/da2 (N/mm per mm)")
a.legend(fontsize=7.5, loc="lower right")
fig.tight_layout(); fig.savefig("results/gradcheck.png"); plt.close(fig)
print("ok")
