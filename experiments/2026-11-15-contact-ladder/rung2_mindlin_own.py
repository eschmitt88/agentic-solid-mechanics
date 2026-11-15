"""Rung 2a, own solver: Cattaneo-Mindlin partial slip, cylinder on a block of the same material."""
import sys, json, os, time
import numpy as np
from contact2d import Model, ContactPair
import meshes as M

HERE = os.path.dirname(os.path.abspath(__file__)); WORK = os.path.join(HERE, "results", "work")
E, NU, R, A_T, MU = 1000.0, 0.3, 1.0, 0.05, 0.3
ES = E / (2 * (1 - NU ** 2))
P = np.pi * ES * A_T ** 2 / (4 * R)
P0 = 2 * P / (np.pi * A_T)
QS = (0.25, 0.5, 0.75)
BLOCK_H = float(os.environ.get("BLOCK_H", "0.6"))   # master (block) element size relative to the slave; 1.3 gave oscillating pressures
EPS_FAC = 2.0   # penalty = EPS_FAC*E/h; with 20 the first Newton steps diverged (geometric stiffness of the over-penalised first guess)


def build_mesh(level):
    h_c = A_T / (8 * 2 ** level)
    X1, Q1, _ = M.run_gmsh(M.half_disc_geo(R, h_c, 3 * A_T, 0.08, 0.7), WORK, f"hertz_2d_L{level}_a{A_T}")
    X2, Q2, _ = M.run_gmsh(M.block_geo(3 * R, 1.5 * R, BLOCK_H * h_c, 3 * A_T, 0.1, 0.9), WORK, f"block_L{level}_{BLOCK_H}")
    X, Q, off, body = M.merge([(X1, Q1), (X2, Q2)])
    edges, el, fa = M.boundary_edges(Q); mid = 0.5 * (X[edges[:, 0]] + X[edges[:, 1]])
    isblk = edges[:, 0] >= off[1]
    sl = (~isblk) & (mid[:, 1] < 0.3 * R) & (np.abs(np.hypot(mid[:, 0], mid[:, 1] - R) - R) < 1e-3 * R) & (np.abs(mid[:, 0]) < 2.0 * A_T)
    ma = isblk & (np.abs(mid[:, 1]) < 1e-12) & (np.abs(mid[:, 0]) < 3.0 * A_T)
    return dict(X=X, Q=Q, off=off, body=body, edges=edges, el=el, fa=fa, sl=sl, ma=ma, h_c=h_c)


def theory(x, qr):
    c = A_T * np.sqrt(1 - qr)
    q = MU * P0 * (np.sqrt(np.clip(1 - (x / A_T) ** 2, 0, None)) - (c / A_T) * np.sqrt(np.clip(1 - (x / c) ** 2, 0, None)))
    return c, q


def measure(x, p, q, qr):
    o = np.argsort(x); x, p, q = x[o], p[o], np.abs(q[o])
    c_t, q_t = theory(x, qr)
    on = p > 0
    slip = on & (q >= 0.99 * MU * p)
    stick = on & ~slip
    # stick edges: midpoint between last stick node and the neighbouring slip node, both sides
    i = np.where(stick)[0]
    if len(i) == 0:
        c_fe = 0.0
    else:
        cl = 0.5 * (x[i.min()] + x[i.min() - 1]); cr = 0.5 * (x[i.max()] + x[i.max() + 1])
        c_fe = 0.5 * (cr - cl)
    q0 = float(np.interp(0.0, x, q))
    l2 = float(np.sqrt(np.sum((q[on] - q_t[on]) ** 2) / np.sum(q_t[on] ** 2)))
    return dict(Q_over_fP=qr, c=float(c_fe), c_ref=float(c_t), c_err_over_a=float((c_fe - c_t) / A_T), q0=q0,
                q0_ref=float(MU * P0 * (1 - c_t / A_T)), q0_err=float(q0 / (MU * P0 * (1 - c_t / A_T)) - 1), l2=l2)


def run(level, nq=15, verbose=False):
    g = build_mesh(level); X, Q, off = g["X"], g["Q"], g["off"]
    m = Model(X, Q, E, NU)
    slave = g["edges"][g["sl"]]
    master = M.chain(g["edges"][g["ma"]][:, ::-1])       # block top, ordered in +x
    cp = ContactPair(slave, master, X, mu=MU, epsN=EPS_FAC * E / g["h_c"], nsign=+1.0)
    m.pairs.append(cp)
    for n in np.where(np.abs(X[:, 1] + 1.5 * R) < 1e-9)[0]:
        m.fixed[2 * n] = 0.0; m.fixed[2 * n + 1] = 0.0
    top = np.where((np.abs(X[:, 1] - R) < 1e-9) & (np.arange(len(X)) < off[1]))[0]
    ry = m.tie(2 * top + 1); rx = m.tie(2 * top)
    d0 = A_T ** 2 / R
    tol = dict(pen_tol=1e-5 * d0, slip_tol=1e-5 * d0, max_aug=80)   # 1e-7*d0 needed >40 Uzawa sweeps per step
    t0 = time.time()
    ok, info = m.solve_step(fixed_values={ry: -0.05 * d0, rx: 0.0}, **tol); assert ok, info
    del m.fixed[ry]            # normal load under force control; u_x of the top stays prescribed
    f = np.zeros(m.ndof)
    nload = 8 * 2 ** max(0, level - 1)     # the active set moves about one node per Newton iteration:
    for k in range(1, nload + 1):          # finer meshes need proportionally smaller load steps (level 3 failed with 8)
        f[ry] = -P * k / nload
        ok, info = m.solve_step(fext=f.copy(), max_newton=200, **tol); assert ok, info
    t = cp.tractions()
    out = dict(level=level, h_c=g["h_c"], nodes=len(X), nq=nq, normal=dict(p0=float(np.interp(0, *sort2(t["X"][:, 0], t["p"]))), p0_ref=P0,
               q_max_after_normal=float(np.abs(t["q"]).max())), loads=[])
    prof = dict(x=t["X"][:, 0], p=t["p"].copy())
    # Tangential loading under displacement control (a force-controlled body has a singular
    # tangent whenever every contact node slips during the Newton iterations).  The top
    # displacement is adjusted by a secant rule until the transmitted force hits each target.
    Qof = lambda: float(np.sum(cp.tractions()["q"] * cp.tractions()["area"]))
    ux, Qprev, dux = 0.0, 0.0, 2e-6
    sgn = None
    for k in range(1, nq + 1):
        qr = 0.75 * k / nq; Qt = qr * MU * P
        for trial in range(40):
            snap = m.snapshot()
            ok, info = m.solve_step(fixed_values={rx: ux + dux}, **tol)
            if not ok:
                m.restore(snap); dux *= 0.5; continue
            Qn = Qof()
            if sgn is None:
                sgn = np.sign(Qn) if Qn != 0 else 1.0
            Qn *= sgn
            if abs(Qn - Qt) < 2e-4 * MU * P:
                ux += dux; slope = (Qn - Qprev) / dux; Qprev = Qn; dux = (0.75 * MU * P / nq) / slope
                break
            slope = (Qn - Qprev) / dux
            if Qn > Qt:          # overshoot: go back and aim again (monotonic loading must be kept)
                m.restore(snap); dux = (Qt - Qprev) / slope
            else:
                ux += dux; Qprev = Qn; dux = (Qt - Qn) / slope
        else:
            raise RuntimeError("target not reached")
        if verbose: print(qr, Qn / (MU * P), ux, flush=True)
        for target in QS:
            if abs(qr - target) < 1e-9:
                t = cp.tractions()
                r = measure(t["X"][:, 0], t["p"], t["q"], Qn / (MU * P)); r["target"] = target
                r["ux_top"] = float(ux); out["loads"].append(r); prof[f"q_{target}"] = t["q"].copy(); prof[f"p_{target}"] = t["p"].copy()
    out["seconds"] = time.time() - t0
    np.savez(os.path.join(HERE, "results", f"r2_mindlin_L{level}_nq{nq}_own.npz"), **prof)
    return out


def sort2(x, y):
    o = np.argsort(x); return x[o], y[o]


if __name__ == "__main__":
    res = []
    for a in sys.argv[1:]:
        lv, nq = (int(v) for v in a.split(":"))
        r = run(lv, nq); res.append(r)
        print(json.dumps(r, indent=None)[:1500], flush=True)
        json.dump(res, open(os.path.join(HERE, "results", f"r2_mindlin_own_{'_'.join(sys.argv[1:]).replace(':','-')}.json"), "w"), indent=1)
