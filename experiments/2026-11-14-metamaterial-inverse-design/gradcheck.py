"""Gradient check: implicit (adjoint) dF/dp vs central finite differences, and where it breaks near the bifurcation.

Test design: perfect cosine arch in the design cell (h = 6 mm, t = 1.5 mm, Q = 4) -> symmetric pitchfork (mode 3).
"""
import json, numpy as np, jax, jax.numpy as jnp
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import fem

S = fem.Solver(nx=40, ny=2, Lh=28.0)
p0 = fem.cosine_params(6.0, 1.5)
out = {}

# 1. coarse scan to locate the bifurcation (first d where the lowest tangent eigenvalue of the symmetric path -> 0)
ds = np.round(np.arange(0.05, 4.0001, 0.05), 6)
F, us, infos = S.path(p0, ds)
X = S.coords(jnp.asarray(p0))
lam = np.array([S.min_eig(u, d, X, 1)[0] for u, d in zip(us, ds)])
sw = [i for i, inf in enumerate(infos) if inf["switched"]]
# kink = max of second difference of F
k = int(np.argmax(np.abs(np.diff(F, 2)))) + 1
dc0 = ds[k]
out["coarse"] = dict(d=ds.tolist(), F=F.tolist(), lam_min=lam.tolist(), switched_idx=sw, d_kink=float(dc0))
print("kink near", dc0, "switched", sw, flush=True)

# 2. fine grid around the kink
dfine = np.round(np.concatenate([np.arange(dc0 - 0.30, dc0 + 0.3001, 0.02), [dc0 + 0.5, dc0 + 1.0, dc0 + 2.0, dc0 + 4.0]]), 6)
dfine = dfine[dfine > 0.02]
def path(p):
    return S.path(p, dfine)
F0, us0, inf0 = path(p0)
us0 = [S.minimise(u, float(d), X, tol=1e-11)[0] for u, d in zip(us0, dfine)]
lam0 = np.array([S.min_eig(u, d, X, 1)[0] for u, d in zip(us0, dfine)])
_, J = S.jacobian(p0, dfine, us0)
# refine d_c: linear extrapolation of lam_min to zero from the post-buckled side is unreliable; use the kink in F
kf = int(np.argmax(np.abs(np.diff(F0[: len(dfine) - 4], 2)))) + 1
dc = float(dfine[kf])
res = dict(d=dfine.tolist(), F=F0.tolist(), lam_min=lam0.tolist(), d_c=dc, J=J.tolist(), fd={})
names = {0: "a1 (apex height)", 4: "log t0 (thickness)", 6: "b2 (thickness mode)", 1: "a2 (imperfection mode)"}
for j in names:
    res["fd"][j] = {}
    for h in (1e-3, 1e-4, 1e-5, 1e-6):
        pp = p0.copy(); pp[j] += h; pm = p0.copy(); pm[j] -= h
        fd = (S.resolve(pp, dfine, us0) - S.resolve(pm, dfine, us0)) / (2 * h)   # same-branch FD
        res["fd"][j][h] = fd.tolist()
        if h == 1e-4:
            res.setdefault("fd_restart", {})[j] = ((path(pp)[0] - path(pm)[0]) / (2 * h)).tolist()  # what a re-run sees
        err = np.abs(fd - J[:, j]) / np.max(np.abs(J[:, j]))
        print(names[j], "h", h, "max rel err far (|d-dc|>0.1):", err[np.abs(dfine - dc) > 0.1].max(), "near:", err[np.abs(dfine - dc) <= 0.1].max(), flush=True)
out["fine"] = res

# 3. cusp: F as a function of the imperfection parameter a2 at fixed d just past d_c
a2 = np.concatenate([-np.logspace(-0.5, -5, 19), [0.0], np.logspace(-5, -0.5, 19)]); a2.sort()
dprobe = [dc + 0.1, dc + 1.0]
cusp = {str(dp): [] for dp in dprobe}
for a in a2:
    p = p0.copy(); p[1] = a
    Fp, usp, _ = S.path(p, np.array(dprobe))
    _, Jp = S.jacobian(p, np.array(dprobe), usp)
    for dp, f, jj in zip(dprobe, Fp, Jp):
        cusp[str(dp)].append((float(a), float(f), float(jj[1])))
out["cusp"] = cusp
json.dump(out, open("results/gradcheck.json", "w"), indent=1)

