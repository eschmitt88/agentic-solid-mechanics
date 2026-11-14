"""Why the bistability optimiser stalls: the Gauss-Newton step from the stalled design improves F_pk and d_2 as predicted
but moves E_back the wrong way.  Resolve F(d) finely before/after the step to see what the pointwise gradient misses."""
import json, numpy as np, jax.numpy as jnp
import bistable, design, fem

r = json.load(open("results/opt_bistable_attempt5_B.json"))["h6_t1.9_a2"]
p = np.array(r["p"])
P = bistable.Problem(log=open("/dev/null", "w"))
r0 = P.fun(p); J = P.jac(p)
dp = -np.linalg.solve(J.T @ J + 1e-2 * np.eye(9), J.T @ r0)
pred = (r0 + J @ dp)[:3]
S = P.S
dfine = np.round(np.concatenate([np.arange(0.25, 5.0, 0.25), np.arange(5.0, 9.5001, 0.02), np.arange(9.75, 13.01, 0.25)]), 6)
out = dict(p=p.tolist(), dp=dp.tolist(), res0=r0[:3].tolist(), predicted_res=pred.tolist(), d=dfine.tolist(), curves={})
for a in (0.0, 0.3, 1.0):
    q = p + a * dp
    F, us, infos = S.path(q, dfine)
    Fc, _, _ = S.path(q, bistable.DS)
    idx = bistable.indices(Fc); f = bistable.features(Fc, idx)[0]
    jumps = np.abs(np.diff(F)) / np.diff(dfine)
    k = int(np.argmax(np.where((dfine[1:] > 5) & (dfine[1:] < 9.5), jumps, 0)))
    # sensitivities at this design: pointwise dF/d(alpha) along dp, by implicit differentiation
    _, Jf = S.jacobian(q, dfine, us)
    out["curves"][str(a)] = dict(F=F.tolist(), features=f.tolist(), rel_res=(f / P.tgt - 1).tolist(), snap_d=float(dfine[k + 1]), snap_dF=float(F[k + 1] - F[k]),
                                 max_slope=float(jumps[k]), dF_dalpha=(Jf @ dp).tolist(), switched=[float(d) for d, i in zip(dfine, infos) if i["switched"]],
                                 jump_flags=[(float(d), float(i["jump"])) for d, i in zip(dfine, infos) if "jump" in i])
    print(a, "features", np.round(f, 5), "rel res", np.round(f / P.tgt - 1, 4), "steepest point d=", dfine[k + 1], "dF=", F[k + 1] - F[k], "slope", jumps[k], out["curves"][str(a)]["jump_flags"][:3], flush=True)
print("predicted residual after full step", np.round(pred, 4))
json.dump(out, open("results/bistable_snap_diagnostic.json", "w"), indent=1)
