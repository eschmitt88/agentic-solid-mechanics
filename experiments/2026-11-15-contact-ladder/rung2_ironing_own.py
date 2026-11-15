"""Rung 2, own solver: shallow ironing (Code_Aster SSNP153 set-up)."""
import sys, json, os, time
import numpy as np
from contact2d import Model, ContactPair
import meshes as M

HERE = os.path.dirname(os.path.abspath(__file__))
E_IND, E_SLAB, NU, MU = 68.96e8, 6.896e8, 0.32, 0.3
D1, D2, RAD, H1, H2, GAP, LEN = 0.2, 1.2, 0.75, 0.95, 4.0, 0.2, 12.0


def ironing_mesh(h, ind_ratio=2):
    nx, ny = int(round(LEN / h)), int(round(H2 / h))
    Xs, Qs = M.structured_rect(0, LEN, -H2, 0, nx, ny)
    ni = int(round(D2 / h)) * ind_ratio; nj = max(4, int(round(H1 / h)) * ind_ratio // 2)
    Xi, Qi = M.structured_rect(0, 1, 0, 1, ni, nj)
    x = D1 + D2 * Xi[:, 0]
    yc = GAP + RAD                                    # centre of the bottom arc
    yb = yc - np.sqrt(RAD ** 2 - (x - (D1 + D2 / 2)) ** 2)
    ytop = GAP + H1
    Xi = np.stack([x, yb + (ytop - yb) * Xi[:, 1]], axis=1)
    X, Q, off, body = M.merge([(Xs, Qs), (Xi, Qi)])
    edges, el, fa = M.boundary_edges(Q); mid = 0.5 * (X[edges[:, 0]] + X[edges[:, 1]])
    isind = edges[:, 0] >= off[1]
    slab_top = (~isind) & (np.abs(mid[:, 1]) < 1e-9)
    ind_surf = isind & (mid[:, 1] < ytop - 1e-9)       # both side walls and the arc
    top = off[1] + np.where(np.abs(Xi[:, 1] - ytop) < 1e-9)[0]
    bottom = np.where(np.abs(X[:, 1] + H2) < 1e-9)[0]
    return dict(X=X, Q=Q, off=off, body=body, edges=edges, el=el, fa=fa, slab_top=slab_top, ind_surf=ind_surf, top=top, bottom=bottom, h=h)


def run(h=0.1, n1=50, n2=200, nh_form="ccx", eps_fac=5.0, mu=MU, tag="", verbose=True, ngp=6):
    g = ironing_mesh(h); X, Q = g["X"], g["Q"]
    m = Model(X, Q, np.where(g["body"] == 0, E_SLAB, E_IND), NU, kind="nh", nh_form=nh_form)
    slave = g["edges"][g["slab_top"]]
    master = M.chain(g["edges"][g["ind_surf"]])       # CCW boundary order: down the left wall, along the arc, up the right wall
    cp = ContactPair(slave, master, X, mu=mu, epsN=eps_fac * E_SLAB / h, nsign=-1.0, ngp=ngp)
    m.pairs.append(cp)
    for n in g["bottom"]:
        m.fixed[2 * n] = 0.0; m.fixed[2 * n + 1] = 0.0
    top = g["top"]
    tol = dict(pen_tol=2e-4 * h, slip_tol=2e-4 * h, max_aug=60, max_newton=40)
    hist = []; t0 = time.time()

    def bc(t):
        uy = -min(t, 1.0); ux = 10.0 * max(t - 1.0, 0.0)
        d = {}
        for n in top:
            d[2 * n] = ux; d[2 * n + 1] = uy
        return d

    t = 0.2 - 1e-9 if True else 0.0      # nothing happens before the gap closes at t = 0.2
    m.solve_step(fixed_values=bc(t), **tol)
    sched = [(1.0, (1.0 - 0.2) / n1), (2.0, 1.0 / n2)]
    nfail = 0
    for t_end, dt0 in sched:
        dt = dt0
        while t < t_end - 1e-12:
            dt_try = min(dt, t_end - t)
            snap = m.snapshot()
            try:
                ok, info = m.solve_step(fixed_values=bc(t + dt_try), **tol)
            except Exception as e:
                ok, info = False, dict(reason=repr(e)[:80])
            if not ok:
                m.restore(snap); nfail += 1; dt = dt_try / 2
                if verbose: print(f"  cutback at t={t:.4f}: {info} -> dt={dt:.2e}", flush=True)
                if dt < dt0 / 64:
                    json.dump(dict(hist=hist, failed_at=t, info=str(info)), open(os.path.join(HERE, "results", f"r2_ironing_own{tag}.json"), "w"))
                    raise RuntimeError(f"step size underflow at t={t}")
                continue
            t += dt_try
            dt = min(dt0, dt * 1.5)
            Rx = float(np.sum(m.reaction[2 * top])); Ry = float(np.sum(m.reaction[2 * top + 1]))
            tr = cp.tractions()
            hist.append(dict(t=t, Rx=Rx, Ry=Ry, newton=info["newton"], aug=info["aug"],
                             Fn_contact=float(np.sum(tr["p"] * tr["area"])), Ft_contact=float(np.sum(tr["q"] * tr["area"]))))
            if verbose and (len(hist) % 10 == 0):
                print(f"t={t:.3f} Rx={Rx:.4e} Ry={Ry:.4e} newton={info['newton']} aug={info['aug']} [{time.time()-t0:.0f}s]", flush=True)
            if abs(t - 1.0) < 1e-9 or abs(t - 1.5) < 1e-9 or abs(t - 2.0) < 1e-9:
                np.savez(os.path.join(HERE, "results", f"r2_ironing_own{tag}_t{t:.1f}.npz"), X=X, Q=Q, u=m.u, body=g["body"], sn=cp.snodes, p=tr["p"], q=tr["q"], W=tr["area"], Rx=Rx, Ry=Ry)
    out = dict(h=h, n1=n1, n2=n2, nh_form=nh_form, eps_fac=eps_fac, mu=mu, nodes=len(X), elements=len(Q), cutbacks=nfail, seconds=time.time() - t0, hist=hist)
    json.dump(out, open(os.path.join(HERE, "results", f"r2_ironing_own{tag}.json"), "w"))
    return out


def summary(hist):
    t = np.array([h["t"] for h in hist]); Rx = np.abs([h["Rx"] for h in hist]); Ry = np.abs([h["Ry"] for h in hist])
    f = lambda a, tt: float(np.interp(tt, t, a))
    w = (t >= 1.4) & (t <= 1.6)
    return dict(Fv_t1=f(Ry, 1.0), Ft_t1=f(Rx, 1.0), Fv_mid=float(Ry[w].mean()), Ft_mid=float(Rx[w].mean()),
                Ft_osc=float((Rx[w].max() - Rx[w].min()) / Rx[w].mean()), ratio_mid=float(Rx[w].mean() / Ry[w].mean()))


if __name__ == "__main__":
    kw = dict(a.split("=") for a in sys.argv[1:])
    conv = dict(h=float, n1=int, n2=int, eps_fac=float, mu=float, nh_form=str, tag=str, ngp=int)
    kw = {k: conv[k](v) for k, v in kw.items()}
    out = run(**kw)
    print(json.dumps(summary(out["hist"])), "cutbacks", out["cutbacks"], "seconds", out["seconds"])
