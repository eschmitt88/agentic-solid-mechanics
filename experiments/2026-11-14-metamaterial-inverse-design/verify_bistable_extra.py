"""Rung 2 follow-up after the t_min/3 CalculiX run aborted at d = 5.93 mm:
(a) what the design solver sees there; (b) CalculiX t_min/3 with a +-0.01 mm symmetry-breaking bump (rung-0 remedy),
(c) CalculiX t_min/4, (d) release from just above state 2 with smaller offsets."""
import json, numpy as np, jax.numpy as jnp
import bistable, ccx, design, fem

v = json.load(open("results/verify_bistable.json")); p = np.array(v["p"]); DS = bistable.DS
tgt = np.array([bistable.F_PK, bistable.D_2, bistable.E_BACK])
def feats(F):
    idx = bistable.indices(F)
    if idx is None: return None
    f, du = bistable.features(np.asarray(F), idx, DS)
    return dict(F_pk=float(f[0]), d_2=float(f[1]), E_back=float(f[2]), d_u=float(du), rel_err=((f / tgt) - 1).tolist(), within_5pct=bool(np.all(np.abs(f / tgt - 1) <= 0.05)))
ex = {}
S = fem.Solver(nx=40, ny=2, Lh=design.LH)
dfine = np.round(np.arange(5.5, 6.6001, 0.02), 6)
F, us, infos = S.path(p, np.concatenate([DS[DS < 5.5], dfine]))
X = S.coords(jnp.asarray(p)); n0 = int(np.sum(DS < 5.5))
lam = [float(S.min_eig(u, float(d), X, 1)[0]) for u, d in zip(us[n0:], dfine)]
k = int(np.argmin(lam))
ex["design_solver_near_abort"] = dict(d=dfine.tolist(), F=F[n0:].tolist(), lam_min=lam, d_at_min_lam=float(dfine[k]), min_lam=lam[k],
                                      refined=[(float(d), i.get("refined")) for d, i in zip(dfine, infos[n0:]) if i.get("refined")],
                                      max_abs_dF_step=float(np.max(np.abs(np.diff(F[n0:])))))
print({k_: ex["design_solver_near_abort"][k_] for k_ in ("d_at_min_lam", "min_lam", "refined", "max_abs_dF_step")}, flush=True)
tmin = float(np.min(np.asarray(fem.thickness(jnp.asarray(p), jnp.linspace(0, 1, 201)))))
for key, q, div in [("ccx_t3_imp_plus", p + np.eye(9)[1] * 0.01, 3), ("ccx_t3_imp_minus", p - np.eye(9)[1] * 0.01, 3), ("ccx_t4", p, 4)]:
    d, Fc, info = ccx.run(f"bistable_{key}", q, Lh=design.LH, ws=design.WS, dmax=13.0, ninc=104, lc=tmin / div)
    Fi = np.interp(DS, d, Fc) if info["completed"] else None
    ex[key] = dict(info=info, F=None if Fi is None else Fi.tolist(), features=None if Fi is None else feats(Fi), d_reached=float(d[-1]) if len(d) else 0.0)
    print(key, info["completed"], ex[key]["d_reached"], ex[key]["features"], flush=True)
d2 = v["ccx_t6"]["features"]["d_2"]
for nm, dh in (("above_state_2_plus0.15", d2 + 0.15), ("above_state_2_plus0.3", d2 + 0.3)):
    r = ccx.run_release("bistable_release_" + nm, p, float(dh))
    v["ccx_release"][nm] = dict(d_hold=float(dh), d_release=r["d_release"], eigvals=r["eigvals"], all_eig_positive=bool(len(r["eigvals"]) > 0 and min(r["eigvals"]) > 0), completed=r["completed"])
    print(nm, v["ccx_release"][nm], flush=True)
v["extra"] = ex
json.dump(v, open("results/verify_bistable.json", "w"), indent=1)
