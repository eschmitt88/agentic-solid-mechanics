"""Rung 1, own solver: contact patch test, 2D Hertz (cylinder), axisymmetric Hertz (sphere)."""
import sys, json, time, os
import numpy as np
from contact2d import Model, ContactPair
import meshes as M

HERE = os.path.dirname(os.path.abspath(__file__)); WORK = os.path.join(HERE, "results", "work")
E, NU, R, A_T = 1000.0, 0.3, 1.0, 0.05


def edge_from(x, p):
    """midpoint between last loaded and first unloaded contact point (sorted by x >= 0)."""
    o = np.argsort(x); x, p = x[o], p[o]
    on = np.where(p > 0)[0]
    last = on.max()
    return 0.5 * (x[last] + x[last + 1]), x[last + 1] - x[last]


def patch(E_up=1000.0, nlow=(7, 4), nup=(5, 3), p_app=1.0, verbose=False, ngp=6):
    Xl, ql = M.structured_rect(0, 2, 0, 1, *nlow)
    Xu, qu = M.structured_rect(0, 2, 1, 2, *nup)
    X, Q, off, body = M.merge([(Xl, ql), (Xu, qu)])
    m = Model(X, Q, np.where(body == 0, E, E_up), NU)
    edges, _, _ = M.boundary_edges(Q)
    mid = 0.5 * (X[edges[:, 0]] + X[edges[:, 1]])
    isup = edges[:, 0] >= off[1]
    slave = edges[isup & (np.abs(mid[:, 1] - 1) < 1e-9)]
    master = M.chain(edges[~isup & (np.abs(mid[:, 1] - 1) < 1e-9)][:, ::-1])  # ordered +x
    h = 2 / max(nlow[0], nup[0])
    cp = ContactPair(slave, master, X, mu=0.0, epsN=10 * E / h, nsign=+1.0, ngp=ngp)
    m.pairs.append(cp)
    for n in np.where(np.abs(X[:, 1]) < 1e-12)[0]:
        m.fixed[2 * n + 1] = 0.0
    m.fixed[2 * int(np.where((np.abs(X[:, 0]) < 1e-12) & (np.abs(X[:, 1]) < 1e-12))[0][0])] = 0.0
    m.fixed[2 * int(off[1] + np.where((np.abs(Xu[:, 0]) < 1e-12) & (np.abs(Xu[:, 1] - 2) < 1e-12))[0][0])] = 0.0
    f = np.zeros(m.ndof)
    top = edges[isup & (np.abs(mid[:, 1] - 2) < 1e-9)]
    for a, b in top:
        L = abs(X[a, 0] - X[b, 0]); f[2 * a + 1] -= 0.5 * p_app * L; f[2 * b + 1] -= 0.5 * p_app * L
    ok, info = m.solve_step(fext=f, pen_tol=1e-10, verbose=verbose)
    t = cp.tractions(); s = m.stress()
    return dict(converged=bool(ok), info=info, p_dev=float(np.max(np.abs(t["p"] - p_app)) / p_app),
                syy_dev=float(np.max(np.abs(s[:, :, 1, 1] + p_app)) / p_app),
                sxx_max=float(np.max(np.abs(s[:, :, 0, 0])) / p_app), sxy_max=float(np.max(np.abs(s[:, :, 0, 1])) / p_app))


def hertz(level, axisym, tag, verbose=False, A_T=A_T, save=True):
    h_c = A_T / (8 * 2 ** level)
    geo = M.half_disc_geo(R, h_c, 3 * A_T, 0.08, 0.7, quarter=axisym)
    X, Q, _ = M.run_gmsh(geo, WORK, f"hertz_{'ax' if axisym else '2d'}_L{level}_a{A_T}")
    nn = len(X)
    X = np.vstack([X, [[-0.5, 0.0], [0.5, 0.0]]])   # rigid flat: two fixed nodes
    m = Model(X, Q, E, NU, axisym=axisym)
    edges, _, _ = M.boundary_edges(Q)
    mid = 0.5 * (X[edges[:, 0]] + X[edges[:, 1]])
    arc = edges[(mid[:, 1] < 0.3 * R) & (np.abs(np.hypot(mid[:, 0], mid[:, 1] - R) - R) < 1e-3 * R) & (np.abs(mid[:, 0]) < 6 * A_T)
                & (np.abs(mid[:, 0]) > 1e-9)]   # (the last test drops the edge lying on the symmetry axis)
    cp = ContactPair(arc, np.array([nn, nn + 1]), X, mu=0.0, epsN=20 * E / h_c, nsign=+1.0, axisym=axisym)
    m.pairs.append(cp)
    for d in (2 * nn, 2 * nn + 1, 2 * nn + 2, 2 * nn + 3):
        m.fixed[d] = 0.0
    top = np.where(np.abs(X[:nn, 1] - R) < 1e-9)[0]
    Es = E / (1 - NU ** 2)
    t0 = time.time()
    if axisym:
        d = A_T ** 2 / R
        for n in np.where(np.abs(X[:nn, 0]) < 1e-12)[0]:
            m.fixed[2 * n] = 0.0
        nst = 4
        for k in range(1, nst + 1):
            ok, info = m.solve_step(fixed_values={2 * n + 1: -d * k / nst for n in top}, pen_tol=1e-7 * d, verbose=verbose)
            assert ok, info
        F = float(np.sum(m.reaction[2 * top + 1]))   # reaction on the moved face
        F = abs(F)
        F_H = 4 / 3 * Es * np.sqrt(R) * d ** 1.5
        a_H = np.sqrt(R * d); p0_H = 3 * F_H / (2 * np.pi * a_H ** 2)
        ref = dict(F=F_H, a=a_H, p0=p0_H, d=d)
    else:
        P = np.pi * Es * A_T ** 2 / (4 * R)
        rep = m.tie(2 * top + 1)
        c = top[np.argmin(np.abs(X[top, 0]))]
        m.fixed[2 * c] = 0.0
        f = np.zeros(m.ndof)
        # establish contact under displacement control first (a free body resting on a
        # single touching point has a singular tangent), then switch to force control
        ok, info = m.solve_step(fixed_values={rep: -0.2 * A_T ** 2 / R}, pen_tol=1e-7 * A_T ** 2 / R)
        assert ok, info
        del m.fixed[rep]
        nst = 4
        for k in range(1, nst + 1):
            f[rep] = -P * k / nst
            ok, info = m.solve_step(fext=f.copy(), pen_tol=1e-7 * A_T ** 2 / R, verbose=verbose)
            assert ok, info
        F = P
        ref = dict(F=P, a=A_T, p0=2 * P / (np.pi * A_T), d=None)
    t = cp.tractions()
    x, p = t["X"][:, 0], t["p"]
    Fc = float(np.sum(p * t["area"]))
    xa, pa = np.abs(x), p
    a_fe, res = edge_from(xa, pa)
    # peak pressure: mean of the contact points nearest the centre (within 0.1 a)
    near = xa < 0.1 * ref["a"]
    p0_fe = float(np.mean(pa[near])) if near.any() else float(pa[np.argmin(xa)])
    pH = ref["p0"] * np.sqrt(np.clip(1 - (xa / ref["a"]) ** 2, 0, None))
    sel = (pa > 0) | (pH > 0)
    l2 = float(np.sqrt(np.sum((pa[sel] - pH[sel]) ** 2) / np.sum(pH[sel] ** 2)))
    out = dict(level=level, h_c=h_c, nodes=int(nn), elements=int(len(Q)), F=F, F_contact=Fc, a=float(a_fe), a_resolution=float(res),
               p0=p0_fe, p0_max=float(pa.max()), l2=l2, ref=ref, approach=float(-m.u[2 * top[0] + 1]),
               err_F=F / ref["F"] - 1, err_a=a_fe / ref["a"] - 1, err_p0=p0_fe / ref["p0"] - 1, seconds=time.time() - t0)
    if save:
        np.savez(os.path.join(HERE, "results", f"r1_{tag}_L{level}_own.npz"), x=x, p=p, X=X[:nn], Q=Q, u=m.u[:2 * nn])
    return out


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "sphere_scaling":
        res = []
        for at in (0.0125, 0.025, 0.05, 0.1):
            r = hertz(2, True, "scal", A_T=at, save=False); r["a_over_R"] = at; res.append(r)
            print(at, r["err_F"], r["err_p0"], r["err_a"], flush=True)
        json.dump(res, open(os.path.join(HERE, "results", "r1_sphere_scaling_own.json"), "w"), indent=1)
    elif what == "patch":
        res = dict(same_material=patch(), stiff_upper=patch(E_up=10 * E), fine=patch(nlow=(13, 6), nup=(17, 5)),
                   matching=patch(nlow=(6, 4), nup=(6, 3)), ngp_sweep={n: patch(ngp=n) for n in (2, 3, 4, 6, 10, 20)})
        print(json.dumps(res, indent=1)); json.dump(res, open(os.path.join(HERE, "results", "r1_patch_own.json"), "w"), indent=1)
    else:
        axisym = what == "sphere"
        tag = "sphere" if axisym else "cyl"
        res = []
        for lv in [int(v) for v in sys.argv[2:]]:
            r = hertz(lv, axisym, tag); res.append(r)
            print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items() if k != "ref"}, flush=True)
        json.dump(res, open(os.path.join(HERE, "results", f"r1_{tag}_own.json"), "w"), indent=1)
