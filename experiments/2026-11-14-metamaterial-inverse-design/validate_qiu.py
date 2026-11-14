"""Rung 0: validate the JAX solver on Qiu's cosine bistable beam (mode 2 constrained)."""
import json, sys, time
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import fem, qiu, ccx

def pts(d, F, h):
    """f_top, f_bot, d_top, d_bot, d_mid, d_end from a sampled curve (parabolic refinement of extrema)."""
    def refine(i):
        if 0 < i < len(d) - 1:
            c = np.polyfit(d[i-1:i+2], F[i-1:i+2], 2)
            x = -c[1] / (2 * c[0]); return x, np.polyval(c, x)
        return d[i], F[i]
    it = np.argmax(np.where(d < h, F, -np.inf)); ib = np.argmin(np.where(d > h, F, np.inf))
    dt, ft = refine(it); db, fb = refine(ib)
    s = np.sign(F); zc = np.where(s[:-1] * s[1:] < 0)[0]
    z = [d[i] - F[i] * (d[i+1] - d[i]) / (F[i+1] - F[i]) for i in zc]
    return dict(f_top=ft, f_bot=-fb, d_top=dt, d_bot=db, d_mid=z[0] if z else np.nan, d_end=z[1] if len(z) > 1 else np.nan)

def grid(h):
    return np.unique(np.round(np.concatenate([np.arange(0.05, 0.45 * h, 0.05), np.arange(0.45 * h, 1.7 * h, 0.125),
                                              np.arange(1.7 * h, 2.15 * h + 1e-9, 0.05)]), 6))

out = {}
l, t, h = 100.0, 1.0, 6.0
c = fem.E_PS * t**3 / 12 * h / l**3
ds = grid(h)
ref = qiu.approx_points(fem.E_PS, t**3 / 12, h, l)
D, Fs = qiu.series_curve(h / t)
ser = pts(D * h, Fs * c, h)
out["case"] = dict(l=l, t=t, h=h, Q=h / t, E_ps=fem.E_PS)
out["qiu_eq_2_47_2_48"] = ref
out["qiu_series_Q6"] = ser
curves = {}
p0 = fem.cosine_params(h, t)
pim = p0.copy(); pim[1] = 0.01
for name, (nx, ny), p in [("jax_25x1", (25, 1), p0), ("jax_50x2", (50, 2), p0), ("jax_100x2", (100, 2), p0), ("jax_100x3", (100, 3), p0),
                          ("jax_100x2_imp", (100, 2), pim)]:
    t0 = time.time()
    S = fem.Solver(nx=nx, ny=ny, Lh=l / 2)
    F, us, infos = S.path(p, ds)
    r = pts(ds, F, h)
    r.update(ndof=int(S.mesh["nfree"]), seconds=time.time() - t0, n_switched=int(sum(i["switched"] for i in infos)),
             all_stable=bool(all(not i.get("unstable", False) for i in infos)))
    out[name] = r; curves[name] = (ds, F)
    print(name, r, flush=True)
for name, lc in [("ccx_imp_lc_t3", t / 3), ("ccx_imp_lc_t5", t / 5)]:
    d, F, info = ccx.run("rung0_" + name, pim, Lh=l / 2, ws=0.0, dmax=ds[-1], ninc=int(round(ds[-1] / 0.05)), lc=lc)
    r = pts(d, F, h); r.update(info); out[name] = r; curves[name] = (d, F)
    print(name, r, flush=True)
d, F, info = ccx.run("rung0_ccx_perfect", p0, Lh=l / 2, ws=0.0, dmax=ds[-1], ninc=int(round(ds[-1] / 0.05)), lc=t / 3)
curves["ccx_perfect"] = (d, F); out["ccx_perfect"] = dict(info, f_max=float(F.max()), d_last=float(d[-1]))

# like-for-like curve error JAX vs CalculiX (same imperfect geometry)
dj, Fj = curves["jax_100x2_imp"]; dc, Fc = curves["ccx_imp_lc_t5"]
Fci = np.interp(dj, dc, Fc)
out["nrmse_jax_vs_ccx_imperfect"] = float(np.sqrt(np.mean((Fj - Fci) ** 2)) / (Fci.max() - Fci.min()))
best = out["jax_100x2"]
out["rel_err_vs_eq_2_47"] = {k: float(best[k] / ref[k] - 1) for k in ("f_top", "f_bot")}
out["abs_err_over_h_vs_eq_2_48"] = {k: float((best[k] - ref[k]) / h) for k in ("d_top", "d_bot", "d_end", "d_mid")}
out["rel_err_vs_series"] = {k: float(best[k] / ser[k] - 1) for k in ("f_top", "f_bot")}
out["abs_err_over_h_vs_series"] = {k: float((best[k] - ser[k]) / h) for k in ("d_top", "d_bot", "d_mid", "d_end")}
json.dump(out, open("results/rung0_qiu_validation.json", "w"), indent=1, default=float)
np.savez("results/rung0_curves.npz", **{k + "_d": v[0] for k, v in curves.items()}, **{k + "_F": v[1] for k, v in curves.items()})

fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
ax.axhline(0, color="#999", lw=0.6)
ax.plot(D * h, Fs * c * 1e3, color="k", lw=1.2, ls="--", label="Qiu modal series (thesis eqs 2.23–2.36)")
ax.plot(*[curves["jax_100x2"][0], curves["jax_100x2"][1] * 1e3], color="#0969da", lw=2, label="JAX solver (Q9, 100×2)")
ax.plot(curves["ccx_imp_lc_t5"][0], curves["ccx_imp_lc_t5"][1] * 1e3, color="#cf222e", lw=1.2, label="CalculiX CPE6, 1 % t imperfection")
ax.plot(curves["ccx_perfect"][0], curves["ccx_perfect"][1] * 1e3, color="#cf222e", lw=1, ls=":", label="CalculiX, perfect geometry (stays on unstable branch)")
ax.plot([ref["d_top"], ref["d_bot"]], [ref["f_top"] * 1e3, -ref["f_bot"] * 1e3], "s", color="k", ms=6, mfc="none", label="Qiu closed form 740 / 370 EIh/l³")
ax.set_ylim(-0.9, 2.0); ax.set_xlim(0, ds[-1])
ax.set_xlabel("centre displacement d (mm)"); ax.set_ylabel("force (mN per mm depth)")
ax.set_title("Rung 0 — cosine bistable beam, l = 100, t = 1, h = 6 mm (Q = 6)", fontsize=10)
ax.legend(fontsize=7.5, frameon=False); fig.tight_layout(); fig.savefig("results/rung0_qiu_validation.png")
print(json.dumps({k: out[k] for k in out if k.startswith(("rel", "abs", "nrmse"))}, indent=1))
