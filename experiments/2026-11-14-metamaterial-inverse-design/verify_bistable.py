"""Rung 2 verification: full force-displacement loop (forward + backward), both stable states, CalculiX re-solve + release test."""
import json, sys
import numpy as np, jax.numpy as jnp
import bistable, ccx, design, fem

src = sys.argv[1] if len(sys.argv) > 1 else "results/design_bistable.json"
p = np.array(json.load(open(src))["p"])
tgt = np.array([bistable.F_PK, bistable.D_2, bistable.E_BACK])
DS = bistable.DS
out = dict(p=p.tolist(), target=dict(F_pk=tgt[0], d_2=tgt[1], E_back=tgt[2]), d=DS.tolist())

def feats(F, d=DS):
    idx = bistable.indices(F)
    if idx is None:
        return None
    f, du = bistable.features(np.asarray(F), idx, d)
    return dict(F_pk=float(f[0]), d_2=float(f[1]), E_back=float(f[2]), d_u=float(du), rel_err=((f / tgt) - 1).tolist(),
                within_5pct=bool(np.all(np.abs(f / tgt - 1) <= 0.05)))

# design solver: forward sweep, then backward sweep from the end state (full loop)
for key, (nx, ny) in [("jax_40x2", (40, 2)), ("jax_80x2", (80, 2))]:
    S = fem.Solver(nx=nx, ny=ny, Lh=design.LH)
    F, us, infos = S.path(p, DS)
    out[key] = dict(F=F.tolist(), features=feats(F), switched=int(sum(i["switched"] for i in infos)),
                    bad=int(sum((not i["converged"]) or i.get("unstable", False) for i in infos)))
    print(key, out[key]["features"], flush=True)
    if key == "jax_40x2":
        Fb, usb, infb = S.path(p, DS[::-1][1:], u0=us[-1], d_start=float(DS[-1]))
        out["jax_40x2_backward"] = dict(d=DS[::-1][1:].tolist(), F=Fb.tolist(), max_abs_diff_vs_forward=float(np.max(np.abs(Fb[::-1] - F[:-1]))))
        # stability of the two force-free states: displacement-controlled tangent PD (checked every step) and dF/dd > 0
        f = out[key]["features"]
        if f:
            X = S.coords(jnp.asarray(p))
            st = {}
            for nm, d0 in (("state_1", 0.0), ("state_2", f["d_2"])):
                dd = np.array([max(d0 - 0.02, 1e-4), d0 + 0.02]) if d0 > 0 else np.array([1e-4, 0.02])
                i0 = int(np.argmin(np.abs(DS - d0))) if d0 > 0 else 0
                Fl, ul, il = S.path(p, dd, u0=us[i0], d_start=float(DS[i0])) if d0 > 0 else S.path(p, dd)
                st[nm] = dict(d=d0, dF_dd=float((Fl[1] - Fl[0]) / (dd[1] - dd[0])), lam_min_disp_ctrl=float(S.min_eig(ul[-1], float(dd[-1]), X, 1)[0]))
                st[nm]["stable"] = bool(st[nm]["dF_dd"] > 0 and st[nm]["lam_min_disp_ctrl"] > 0)
            out["jax_stability"] = st
            print(st, flush=True)

# CalculiX displacement control, two meshes
tmin = float(np.min(np.asarray(fem.thickness(jnp.asarray(p), jnp.linspace(0, 1, 201)))))
for key, div in [("ccx_t3", 3), ("ccx_t6", 6)]:
    d, F, info = ccx.run(f"bistable_{key}", p, Lh=design.LH, ws=design.WS, dmax=13.0, ninc=104, lc=tmin / div)
    Fi = np.interp(DS, d, F) if info["completed"] else None
    out[key] = dict(info=info, F=None if Fi is None else Fi.tolist(), features=None if Fi is None else feats(Fi), d_reached=float(d[-1]) if len(d) else 0.0)
    print(key, info["completed"], out[key]["features"], flush=True)

# CalculiX: release tests around the second state and near the first
f = out["ccx_t3"]["features"] or out["jax_40x2"]["features"]
if f:
    rel = {}
    for nm, dh in [("above_state_2", f["d_2"] + 0.6), ("below_state_2", f["d_u"] + 0.6 * (f["d_2"] - f["d_u"])), ("near_state_1", 0.5 * f["d_u"] if f["d_u"] < 3 else 1.0)]:
        r = ccx.run_release(f"bistable_release_{nm}", p, float(dh))
        rel[nm] = dict(d_hold=float(dh), d_release=r["d_release"], eigvals=r["eigvals"], all_eig_positive=bool(len(r["eigvals"]) > 0 and min(r["eigvals"]) > 0), completed=r["completed"])
        print(nm, rel[nm], flush=True)
    out["ccx_release"] = rel
json.dump(out, open("results/verify_bistable.json", "w"), indent=1)
