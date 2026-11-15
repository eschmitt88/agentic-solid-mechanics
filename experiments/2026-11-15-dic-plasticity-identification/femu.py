"""Finite-element model updating: Levenberg-Marquardt on the pre-registered objective with exact
sensitivities; one worker process per specimen.  Also Gauss-Newton identifiability analysis."""
import os, sys, json, time, numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
PN = ["s0", "Q1", "b1", "H", "Q2", "b2", "a", "h", "c"]
FORMS = {
    "M1": {"free": [0, 1, 2], "fixed": {3: 0.0, 4: 0.0, 5: 1.0, 6: 1.0, 7: 0.5, 8: 3.0}},
    "M2": {"free": [0, 1, 2, 3], "fixed": {4: 0.0, 5: 1.0, 6: 1.0, 7: 0.5, 8: 3.0}},
    "M3": {"free": [0, 1, 2, 3, 4, 5], "fixed": {6: 1.0, 7: 0.5, 8: 3.0}},
    "M4": {"free": [0, 1, 2, 3, 6, 7, 8], "fixed": {4: 0.0, 5: 1.0}},
    "M4b": {"free": [0, 1, 2, 3, 4, 5, 6, 7, 8], "fixed": {}},
}
SIG_F, SIG_E = 0.02, 0.10

_S = {}


def _init(threads):
    os.environ["MKL_NUM_THREADS"] = str(threads); os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    sys.path.insert(0, HERE)


def _eval(spec, kw, p, sens_idx, opts):
    import specimen, data
    key = (spec, json.dumps(kw, sort_keys=True))
    if key not in _S:
        _S[key] = specimen.Specimen(spec, verbose=False, **kw)
    S = _S[key]
    t0 = time.time()
    try:
        sim = S.simulate(np.array(p), sens_idx=sens_idx, **opts)
    except Exception as e:  # solver failure -> infinite cost
        return {"fail": repr(e)}
    r, J = residual(S, sim, len(sens_idx) if sens_idx else 0)
    return {"r": r, "J": J, "metrics": S.metrics(sim), "F": sim["F"], "stats": sim["stats"], "time": time.time() - t0,
            "nF": int(S.force_ok.sum())}


def residual(S, sim, nk):
    import data
    ok = S.force_ok
    Fm = S.F_meas[ok]
    sF = SIG_F * Fm.mean() * np.sqrt(ok.sum())
    r = [(sim["F"][ok] - Fm) / sF]
    J = [sim["dF"][ok] / sF] if nk else []
    nst = len(data.FIELD_ST)
    for i, j in enumerate(data.FIELD_ST):
        Em = S.E_dic[i]
        sE = SIG_E * np.sqrt((Em ** 2).mean()) * np.sqrt(Em.size * nst)
        r.append(((sim["E"][j] - Em) / sE).ravel())
        if nk:
            J.append((sim["dE"][j] / sE).reshape(-1, nk))
    return np.concatenate(r), (np.concatenate(J, 0) if nk else None)


class Pool:
    def __init__(self, specs, threads=3):
        """specs: list of (name, kwargs for Specimen)."""
        self.specs = specs
        ctx = mp.get_context("spawn")
        self.ex = [ProcessPoolExecutor(1, mp_context=ctx, initializer=_init, initargs=(threads,)) for _ in specs]

    def evaluate(self, p, sens_idx=None, opts=None):
        futs = [ex.submit(_eval, s, kw, list(map(float, p)), sens_idx, opts or {}) for ex, (s, kw) in zip(self.ex, self.specs)]
        return [f.result() for f in futs]

    def close(self):
        for ex in self.ex:
            ex.shutdown(cancel_futures=True)


def full_p(form, x):
    """x = log of the free parameters -> full 9-vector."""
    p = np.zeros(9)
    for i, v in FORMS[form]["fixed"].items():
        p[i] = v
    p[FORMS[form]["free"]] = np.exp(x)
    return p


ALPHA_NODES = {3: [0.0, 0.1, 0.5], 4: [0.0, 0.05, 0.2, 0.6], 6: [0.0, 0.02, 0.08, 0.2, 0.4, 0.7]}


class CurveCoords:
    """Re-parametrisation used only inside the optimiser: the hardening parameters are replaced by the
    values of sigma_y at fixed plastic strains (log), Hill coefficients stay as logs.  In these coordinates
    the objective is far closer to quadratic than in (Q, b, H), which trade off along a curved valley."""

    def __init__(self, form):
        self.form = form
        self.free = FORMS[form]["free"]
        self.hard = [k for k, i in enumerate(self.free) if i < 6]
        self.nodes = np.array(ALPHA_NODES[len(self.hard)])

    def z(self, x):
        p = full_p(self.form, x)
        z = np.array(x, float).copy()
        z[self.hard] = np.log(sigy(p, self.nodes))
        return z

    def dzdx(self, x, h=1e-6):
        n = len(x); D = np.zeros((n, n))
        for k in range(n):
            e = np.zeros(n); e[k] = h
            D[:, k] = (self.z(x + e) - self.z(x - e)) / (2 * h)
        return D

    def x_of_z(self, z, x0):
        from scipy.optimize import root
        r = root(lambda x: self.z(x) - z, x0, method="hybr", tol=1e-12)
        if not r.success or np.max(np.abs(self.z(r.x) - z)) > 1e-8 or not np.all(np.isfinite(r.x)):
            return None
        return r.x


def lm_fit(pool, form, p0_free, max_iter=30, lam0=1e-2, log=print, xtol=1e-4, ftol=2e-3, patience=2, reparam=True):
    free = FORMS[form]["free"]
    x = np.log(np.array(p0_free, float))
    hist = []
    cc = CurveCoords(form)
    nev = [0]

    def ev(x):
        p = full_p(form, x)
        nev[0] += 1
        res = pool.evaluate(p, sens_idx=free)
        if any("fail" in r for r in res):
            return None
        r = np.concatenate([q["r"] for q in res])
        J = np.concatenate([q["J"] for q in res], 0) * p[free][None, :]  # d/dlog p
        return r, J, res

    out = ev(x)
    if out is None:
        raise RuntimeError("start point failed")
    r, J, res = out
    cost = float(r @ r)
    lam = lam0
    slow = 0
    log(f"  it 0 cost {cost:.5f} p={np.round(full_p(form, x)[free], 4).tolist()} " +
        " ".join(f"[F {q['metrics']['force_nrmse']:.4f} E {q['metrics'].get('strain_relL2_st40', np.nan):.4f}]" for q in res))
    for it in range(1, max_iter + 1):
        D = cc.dzdx(x) if reparam else None
        use_z = reparam and np.all(np.isfinite(D)) and np.linalg.cond(D) < 1e6
        Jz = J @ np.linalg.inv(D) if use_z else J
        g = Jz.T @ r; H = Jz.T @ Jz
        improved = False
        for _ in range(8):
            try:
                dz = np.linalg.solve(H + lam * np.diag(np.diag(H)) + 1e-12 * np.eye(len(x)), -g)
            except np.linalg.LinAlgError:
                lam *= 10; continue
            dz = np.clip(dz, -1.0, 1.0)
            x_try = cc.x_of_z(cc.z(x) + dz, x) if use_z else x + dz
            if x_try is None or np.max(np.abs(x_try - x)) > 3.0:
                lam *= 5; continue
            out = ev(x_try)
            if out is not None and float(out[0] @ out[0]) < cost:
                improved = True
                break
            lam *= 5
        if not improved:
            log(f"  it {it}: no improving step (lam={lam:.1e}) -> stop")
            break
        r_n, J_n, res_n = out
        cost_n = float(r_n @ r_n)
        rel = (cost - cost_n) / cost
        dx = x_try - x
        x, r, J, res, cost = x_try, r_n, J_n, res_n, cost_n
        lam = max(lam / 5, 1e-7)
        hist.append({"it": it, "cost": cost, "p": full_p(form, x).tolist()})
        log(f"  it {it} cost {cost:.5f} (rel drop {rel:.1e}, |dx|={np.abs(dx).max():.1e}, lam={lam:.0e}) p={np.round(full_p(form, x)[free], 4).tolist()}")
        slow = slow + 1 if rel < ftol else 0
        if slow >= patience or np.abs(dx).max() < xtol:
            log(f"  stop: relative cost drop < {ftol:g} for {patience} consecutive iterations (or step < xtol)")
            break
    ident = identifiability(J, r, free)
    D = cc.dzdx(x)
    if np.all(np.isfinite(D)) and np.linalg.cond(D) < 1e10:
        Jz = J @ np.linalg.inv(D)
        ident["curve_coords"] = dict(identifiability(Jz, r, free), nodes=cc.nodes.tolist(),
                                     names=[f"sigy({a:g})" for a in cc.nodes] + [PN[i] for i in free if i >= 6])
    return {"form": form, "p": full_p(form, x).tolist(), "cost": cost, "x": x.tolist(), "hist": hist, "n_eval": nev[0],
            "metrics": {s[0]: q["metrics"] for s, q in zip(pool.specs, res)}, "ident": ident}


def identifiability(J, r, free):
    """Gauss-Newton analysis in log-parameters."""
    H = J.T @ J
    n, k = J.shape
    w, V = np.linalg.eigh(H)
    # residual-scaled covariance is meaningless here (model error >> noise, residuals correlated), so report
    # (a) eigen-structure, (b) correlation, (c) the relative parameter change that raises the cost by 1 %.
    cost = float(r @ r)
    try:
        C = np.linalg.inv(H)
        sd = np.sqrt(np.diag(C)); corr = C / np.outer(sd, sd)
        dlog_1pct = np.sqrt(0.01 * cost * np.diag(C))  # profile (others re-optimised) change in log p for +1 % cost
    except np.linalg.LinAlgError:
        corr = np.full((k, k), np.nan); dlog_1pct = np.full(k, np.nan)
    dlog_1pct_fixed = np.sqrt(0.01 * cost / np.diag(H))  # others held fixed
    return {"params": [PN[i] for i in free], "eig": w.tolist(), "eigvec": V.tolist(), "cond": float(w[-1] / max(w[0], 1e-300)),
            "corr": corr.tolist(), "dlog_1pct_profile": dlog_1pct.tolist(), "dlog_1pct_fixed": dlog_1pct_fixed.tolist()}


def sigy(p, a):
    p = np.asarray(p); a = np.asarray(a)
    return p[0] + p[1] * (1 - np.exp(-p[2] * a)) + p[3] * a + p[4] * (1 - np.exp(-p[5] * a))
