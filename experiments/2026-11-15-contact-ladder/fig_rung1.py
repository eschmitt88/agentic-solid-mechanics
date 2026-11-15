"""Rung 1 figures."""
import json, os
import numpy as np
import plotstyle
plt = plotstyle.setup(); C = plotstyle.C
R = "results/"
j = lambda f: json.load(open(R + f))
cyl, sph, scal, cx, po, pc = j("r1_cyl_own.json"), j("r1_sphere_own.json"), j("r1_sphere_scaling_own.json"), j("r1_hertz_ccx.json"), j("r1_patch_own.json"), j("r1_patch_ccx.json")

# --- pressure distributions
fig, axs = plt.subplots(2, 1, figsize=(6.6, 9.0))
for ax, tag, ref, ccxf, ttl in ((axs[0], "cyl", cyl[-1]["ref"], "r1_hertz_2d_SURFACETOSURFACE_L3.npz", "Cylinder on a flat (2D)"),
                                (axs[1], "sphere", sph[-1]["ref"], "r1_hertz_ax_SURFACETOSURFACE_L3.npz", "Sphere on a flat (axisymmetric)")):
    a, p0 = ref["a"], ref["p0"]
    xx = np.linspace(0, 1, 400)
    ax.plot(xx, np.sqrt(1 - xx ** 2), color=C["ref"], lw=2.5, label="Hertz (closed form)")
    d = np.load(R + f"r1_{tag}_L3_own.npz"); x, p = np.abs(d["x"]), d["p"]; o = np.argsort(x)
    ax.plot(x[o][::3] / a, p[o][::3] / p0, "o", ms=4, color=C["blue"], label="own solver")
    d = np.load(R + ccxf)
    ax.plot(d["x"][::3] / a, d["p"][::3] / p0, "s", ms=4, mfc="none", color=C["orange"], label="CalculiX (surface-to-surface)")
    ax.set_xlim(0, 1.25); ax.set_ylim(0, 1.3); ax.set_xlabel("distance from centre / contact half-width"); ax.set_title(ttl)
[a.set_ylabel("contact pressure / Hertz peak") for a in axs]; axs[0].legend(loc="lower left")
axs[1].annotate("CalculiX output is wrong on the\nfirst nodes next to the axis\n(0, 4.9, 0.2, 1.2 × Hertz)", xy=(0.03, 1.22), xytext=(0.3, 1.02), fontsize=10,
                arrowprops=dict(arrowstyle="->", color=C["mute"]), color="#52514e")
fig.tight_layout(); fig.savefig(R + "r1_hertz_pressure.png"); plt.close(fig)

# --- convergence + finite size
fig, axs = plt.subplots(2, 1, figsize=(6.6, 9.0))
h = [c["h_c"] / c["ref"]["a"] for c in cyl]
axs[0].loglog(h, [abs(c["err_p0"]) * 100 for c in cyl], "o-", color=C["blue"], label="peak pressure, own")
axs[0].loglog(h, [abs(c["err_a"]) * 100 for c in cyl], "s-", color=C["aqua"], label="contact half-width, own")
axs[0].loglog(h, [c["l2"] * 100 for c in cyl], "^-", color=C["violet"], label="pressure distribution (L2), own")
cc = [c for c in cx if c.get("ctype") == "SURFACE TO SURFACE" and c.get("kfac") == 50.0 and abs(c["ref"]["a"] - 0.05) < 1e-9 and c["ref"]["F"] > 1][:4]
axs[0].loglog([c["h_c"] / 0.05 for c in cc], [abs(c["err_p0"]) * 100 for c in cc], "o--", mfc="none", color=C["orange"], label="peak pressure, CalculiX")
axs[0].set_xlabel("contact element size / contact half-width"); axs[0].set_ylabel("error vs Hertz (%)"); axs[0].set_title("2D Hertz: mesh convergence"); axs[0].legend(fontsize=9)
ar = [s["a_over_R"] for s in scal]
axs[1].plot(ar, [s["err_F"] * 100 for s in scal], "o-", color=C["blue"], label="own solver (force at given approach)")
axs[1].plot(ar, [100 * 4 / (3 * np.pi) * a * 1.5 for a in ar], "--", color=C["mute"], label="half-space estimate (2/π)·a/R")
axs[1].axhline(3, color=C["bad"], lw=1); axs[1].text(0.012, 3.15, "pre-registered tolerance 3 %", color=C["bad"], fontsize=10)
axs[1].axvline(0.05, color=C["mute"], lw=0.8, ls=":"); axs[1].text(0.052, 0.3, "graded case", fontsize=10, color="#52514e")
axs[1].set_xlabel("contact radius / sphere radius"); axs[1].set_ylabel("force above Hertz (%)"); axs[1].set_title("Sphere: the 'error' is the finite sphere"); axs[1].legend(fontsize=9, loc="upper left")
fig.tight_layout(); fig.savefig(R + "r1_convergence.png"); plt.close(fig)

# --- patch test
fig, ax = plt.subplots(figsize=(6.6, 4.4))
ng = sorted(po["ngp_sweep"], key=int)
ax.semilogy([int(n) for n in ng], [po["ngp_sweep"][n]["syy_dev"] * 100 for n in ng], "o-", color=C["blue"], label="own solver, by integration points per edge")
for c, col, lab in zip(pc, (C["orange"], C["bad"], None), ("CalculiX surface-to-surface", "CalculiX node-to-surface", None)):
    if lab and c.get("finished"):
        v = max(c["syy_dev"], c["p_dev"]) * 100
        ax.axhline(max(v, 1e-3), color=col, ls="--", label=f"{lab} ({v:.2g} %)")
ax.axhline(1, color=C["ref"], lw=1); ax.text(12, 1.15, "tolerance 1 %", fontsize=10)
ax.set_xlabel("integration points per slave edge (own solver)"); ax.set_ylabel("max stress deviation (%)"); ax.set_title("Contact patch test, non-matching meshes"); ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(R + "r1_patch.png"); plt.close(fig)
print("ok")
