"""Past the snap: bistability features as state functions with fold-point sensitivities (arm A), trial 9's
grid features (arm B0), and an automatic outer loop (arms +O).  See PREREGISTRATION.md, "Snap test".

Arm A.  Along a displacement-controlled path the internal state can snap at a fold point d_s (tangent singular,
null vector phi).  An integral of the force over the path is then NOT differentiable state-by-state: its
derivative has the extra term  [F] * dd_s/dp  that the per-state implicit gradient cannot see.  Here
    E_back = -int_{d_u}^{d_2} F dd = Pi(d_u) - Pi(d_2) - sum_snaps (Pi^- - Pi^+)
is evaluated from energies at equilibria (no quadrature across the snap) and differentiated as
    dPi/dp = partial_p Pi                      (envelope theorem; F = 0 at d_u and d_2)
    d(Pi^- - Pi^+)/dp = partial_p(Pi^- - Pi^+) + (F^- - F^+) dd_s/dp
    dd_s/dp = -(phi^T partial_p R) / (phi^T partial_d R)     (augmented limit-point system, first order only)
"""
from __future__ import annotations

import json
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

import design
import fem

F_PK, D_2, E_BACK = 0.040, 10.0, 0.045
TGT = np.array([F_PK, D_2, E_BACK])
DS = np.round(np.arange(1, 105) * 0.125, 6)
NK = NM = 4
PEN, PEN_JAC, LB, UB = design.make_penalty(NK, NM, 1)


# ---- trial 9's feature extractor, verbatim (the common scorer and arm B0) ----------------------------------
def indices(F):
    F0 = np.concatenate([[0.0], F])
    down = [i for i in range(1, len(F0) - 1) if F0[i] > 0 >= F0[i + 1]]
    if not down:
        return None
    iu = down[0]
    up = [i for i in range(iu + 1, len(F0) - 1) if F0[i] < 0 <= F0[i + 1]]
    if not up:
        return None
    i2 = up[0]
    ipk = int(np.argmax(F0[: iu + 1]))
    ipk = min(max(ipk, 1), len(F0) - 2)
    return ipk, iu, i2


def features(F, idx, d=None):
    xp = jnp if isinstance(F, jnp.ndarray) else np
    d = DS if d is None else d
    F0 = xp.concatenate([xp.zeros(1), F])
    d0 = np.concatenate([[0.0], d])
    ipk, iu, i2 = idx
    a, b, c = F0[ipk - 1], F0[ipk], F0[ipk + 1]
    den = a - 2 * b + c
    fpk = b - (c - a) ** 2 / (8 * den)
    du = d0[iu] + (d0[iu + 1] - d0[iu]) * F0[iu] / (F0[iu] - F0[iu + 1])
    d2 = d0[i2] + (d0[i2 + 1] - d0[i2]) * F0[i2] / (F0[i2] - F0[i2 + 1])
    seg = 0.5 * (F0[iu + 1: i2] + F0[iu + 2: i2 + 1]) * (d0[iu + 2: i2 + 1] - d0[iu + 1: i2])
    E = -(0.5 * F0[iu + 1] * (d0[iu + 1] - du) + xp.sum(seg) + 0.5 * F0[i2] * (d2 - d0[i2]))
    return xp.stack([fpk, d2, E]) if xp is jnp else np.array([fpk, d2, E]), du


def score(S, p):
    """Common scorer: trial 9's grid features at design p."""
    F, zs, infos = S.path(p, DS)
    if not np.all(np.isfinite(F)):
        return dict(feat=None, rel_err=None, success=False, bistable=False)
    idx = indices(F)
    if idx is None:
        return dict(feat=None, rel_err=None, success=False, bistable=False, F=F.tolist())
    f = features(F, idx)[0]
    rel = f / TGT - 1
    return dict(feat=f.tolist(), rel_err=rel.tolist(), success=bool(np.all(np.abs(rel) <= 0.01)), bistable=True, F=F.tolist(),
                switched=int(sum(i["switched"] for i in infos)))


# ---- fold tools ---------------------------------------------------------------------------------------------
class Folds:
    def __init__(self, S):
        self.S = S
        res = S.residual

        def border(z, d, X, phi, rhs_top, rhs_bot):
            K = S.assemble(z, d, X)[1]
            c = jax.jvp(lambda dd: res(z, dd, X), (d,), (1.0,))[1]
            A = jnp.block([[K, c[:, None]], [phi[None, :], jnp.zeros((1, 1))]])
            return jnp.linalg.solve(A, jnp.concatenate([rhs_top, jnp.reshape(rhs_bot, (1,))]))

        self._border = jax.jit(border)

    def arc_state(self, X, z0, d0, phi, tau, tol=1e-10, maxit=25):
        """Solve R(z, d) = 0, phi.(z - z0) = tau  (pseudo-arclength; regular at a fold).  Returns z, d, dd/dtau."""
        z, d = z0 + tau * phi, jnp.asarray(d0)
        for _ in range(maxit):
            r = self.S.residual(z, d, X)
            g = float(phi @ (z - z0)) - tau
            if float(jnp.max(jnp.abs(r))) < tol and abs(g) < 1e-12:
                break
            dx = self._border(z, d, X, phi, -r, -g)
            z, d = z + dx[:-1], d + dx[-1]
        ok = float(jnp.max(jnp.abs(self.S.residual(z, d, X)))) < 1e-8
        t = self._border(z, d, X, phi, jnp.zeros_like(z), 1.0)
        return z, float(d), float(t[-1]), ok

    def bracket(self, X, ev, levels=6):
        """Shrink a jumping sub-step [d0, d1] by bisection, keeping one state on each side of the jump."""
        d0, d1, z0, z1 = ev["d0"], ev["d1"], ev["z0"], ev["z1"]
        for _ in range(levels):
            full = float(jnp.max(jnp.abs(z1 - z0)))
            if full < 1e-6:
                break
            dm = 0.5 * (d0 + d1)
            zm, info = self.S.solve_step(z0, dm, X)
            if not info["converged"]:
                break
            if float(jnp.max(jnp.abs(zm - z0))) < 0.5 * full:
                d0, z0 = dm, zm
            else:
                d1, z1 = dm, zm
        return d0, z0, d1, z1

    def locate(self, p, X, ev, jump_tol=0.02):
        """Classify a jumping sub-step and, if it is a snap, solve for the fold point.

        Returns None for a steep but continuous passage; otherwise a dict with d_s, the fold state, the landed
        state at d_s, forces and energies on both sides, and dd_s/dp."""
        S = self.S
        d0, z0, d1, z1 = self.bracket(X, ev)
        if float(jnp.max(jnp.abs(z1 - z0))) < jump_tol:
            return None
        w, V = S._eigh(S.stiffness(z0, d0, X))
        phi = V[:, 0]
        out = dict(lam_pre=float(w[0]), bracket=[d0, d1], exact=False)
        # orient phi so that d increases with tau on the pre-fold side, then secant on dd/dtau = 0
        _, _, dd0, ok = self.arc_state(X, z0, d0, phi, 0.0)
        if dd0 < 0:
            phi, dd0 = -phi, -dd0
        zs, ds = z0, d0
        try:
            # near a fold d_s - d = a tau'^2  ->  tau to the fold ~ 2 (d_s - d) / (dd/dtau);  d_s - d0 <= bracket width
            t_prev, g_prev = 0.0, dd0
            t = 2.0 * max(d1 - d0, 1e-7) / dd0
            for _ in range(12):
                zt, dt, g, ok = self.arc_state(X, z0, d0, phi, t)
                if not ok:
                    raise RuntimeError("arc solve failed")
                if abs(g) < 1e-9 * max(abs(dd0), 1e-30) or abs(t - t_prev) < 1e-13:
                    break
                t_new = t - g * (t - t_prev) / (g - g_prev)
                t_prev, g_prev, t = t, g, t_new
            if abs(g) < 1e-5 * abs(dd0) and abs(dt - d0) < 10 * (d1 - d0) + 1e-6:
                zs, ds = zt, dt
                out["exact"] = True
                out["ddtau_rel"] = abs(g) / abs(dd0)
        except Exception as e:  # keep the bracket state as the fold estimate
            out["error"] = repr(e)
        w2, V2 = S._eigh(S.stiffness(zs, ds, X))
        phi = V2[:, 0]
        zl, info = S.minimise(z1, ds, X, tol=1e-10)  # landed state at exactly d_s
        pj = jnp.asarray(p)
        gp, gd = S._dphiR(pj, ds, zs, phi)
        Em, gEm = S.energy_grad(p, zs, ds)
        Ep, gEp = S.energy_grad(p, zl, ds)
        Fm, Fp = 2 * float(S.reaction(zs, ds, X)), 2 * float(S.reaction(zl, ds, X))
        out.update(d_s=float(ds), z_s=zs, z_l=zl, lam_fold=float(w2[0]), dds_dp=-np.asarray(gp) / float(gd), phiRd=float(gd),
                   F_minus=Fm, F_plus=Fp, loss=Em - Ep, dloss_dp=(gEm - gEp) + (Fm - Fp) * (-np.asarray(gp) / float(gd)),
                   landed_ok=bool(info["converged"]), jump=float(jnp.max(jnp.abs(zl - zs))))
        return out

    def refold(self, p, X, f, back=2e-4):
        """Re-solve an already located fold at a nearby design (for finite-difference checks): arclength secant from f's
        state, started `back` mm before the old fold (must exceed the fold's shift)."""
        S = self.S
        w, V = S._eigh(S.stiffness(f["z_s"], f["d_s"], X))
        phi = V[:, 0]
        d0 = f["d_s"] - back
        z0, _ = S.minimise(f["z_s"], d0, X, tol=1e-11)  # pre-fold branch, safely before the shifted fold
        _, _, dd0, _ = self.arc_state(X, z0, d0, phi, 0.0)
        if dd0 < 0:
            phi, dd0 = -phi, -dd0
        t_prev, g_prev, t = 0.0, dd0, 2.0 * back / dd0
        for _ in range(20):
            zt, dt, g, ok = self.arc_state(X, z0, d0, phi, t)
            if abs(g) < 1e-10 * abs(dd0):
                break
            t_new = t - g * (t - t_prev) / (g - g_prev)
            t_prev, g_prev, t = t, g, t_new
        zl, _ = S.minimise(f["z_l"], dt, X, tol=1e-11)
        return dict(d_s=dt, loss=2 * float(S.energy(zt, dt, X)) - 2 * float(S.energy(zl, dt, X)))


# ---- problems -------------------------------------------------------------------------------------------------
class Base:
    def __init__(self, S=None, log=sys.stdout):
        self.S = S or fem.Solver(nx=40, ny=2, Lh=design.LH, nk=NK, nm=NM)
        self.cache, self.hist, self.log, self.npath = None, [], log, 0

    def fun(self, p):
        c = self.evaluate(p)
        r = (c["feat"] / TGT - 1.0) if c["feat"] is not None else np.full(3, 30.0)
        return np.concatenate([r, np.asarray(PEN(jnp.asarray(p)))])

    def jac(self, p):
        c = self.evaluate(p)
        return np.vstack([c["dfeat"] / TGT[:, None], np.asarray(PEN_JAC(jnp.asarray(p)))])

    def _log(self, c):
        self.hist.append(dict(p=c["p"].tolist(), feat=None if c["feat"] is None else c["feat"].tolist(), switched=c["switched"], nsnap=c.get("nsnap", 0)))
        f = None if c["feat"] is None else np.round(c["feat"], 5).tolist()
        print(f"  eval {len(self.hist):3d} feat={f} sw={c['switched']} snaps={c.get('nsnap', 0)} p={np.round(c['p'], 3).tolist()}", file=self.log, flush=True)


class GridProblem(Base):
    """Arm B0: trial 9's bistable.Problem."""

    def evaluate(self, p):
        p = np.asarray(p, float)
        if self.cache is not None and np.array_equal(self.cache["p"], p):
            return self.cache
        F, zs, infos = self.S.path(p, DS)
        self.npath += 1
        ok = bool(np.all(np.isfinite(F)))
        idx = indices(F) if ok else None
        c = dict(p=p.copy(), feat=None, dfeat=np.zeros((3, len(p))), switched=int(sum(i["switched"] for i in infos)))
        if idx:
            c["feat"] = features(F, idx)[0]
            _, J = self.S.jacobian(p, DS, zs)
            c["dfeat"] = np.asarray(jax.jacobian(lambda f: features(f, idx)[0])(jnp.asarray(F))) @ J
        self.cache = c
        self._log(c)
        return c


class FoldProblem(Base):
    """Arm A: features as state functions, fold points as explicit unknowns."""

    def __init__(self, S=None, log=sys.stdout):
        super().__init__(S, log)
        self.folds = Folds(self.S)

    def probe(self, p, X, d, z0):
        S = self.S
        z, info = S.minimise(z0, float(d), X, tol=1e-10)
        K = S.stiffness(z, float(d), X)
        F, (g, gD) = S._dF(jnp.asarray(p), float(d), z, K)
        return dict(d=float(d), z=z, F=2 * float(F), dFdp=2 * np.asarray(g), dFdD=2 * float(gD), ok=info["converged"])

    def root(self, p, X, key, a, b, tol, maxit=30):
        """Illinois regula falsi on probe[key] between two probed states a, b (opposite signs)."""
        fa, fb = a[key], b[key]
        if fa * fb > 0:
            return a if abs(fa) < abs(fb) else b
        side = 0
        for _ in range(maxit):
            d = (a["d"] * fb - b["d"] * fa) / (fb - fa)
            near = a if abs(d - a["d"]) < abs(d - b["d"]) else b
            c = self.probe(p, X, d, near["z"])
            fc = c[key]
            if abs(fc) < tol or abs(b["d"] - a["d"]) < 1e-9:
                return c
            if fc * fb < 0:
                a, fa = b, fb
                side = 0
            else:
                if side == 1:
                    fa *= 0.5
                side = 1
            b, fb = c, fc
        return c

    def state_features(self, p):
        """Returns dict(feat, dfeat, snaps, ...) or feat None when not bistable."""
        S = self.S
        p = np.asarray(p, float)
        X = S.coords(jnp.asarray(p))
        events = []
        F, zs, infos = S.path(p, DS, events=events)
        self.npath += 1
        self._last = (X, zs)
        out = dict(p=p.copy(), feat=None, dfeat=np.zeros((3, len(p))), switched=int(sum(i["switched"] for i in infos)), nsnap=0, F=F)
        if not np.all(np.isfinite(F)):
            return out
        idx = indices(F)
        if idx is None:
            return out
        ipk, iu, i2 = idx
        d0 = np.concatenate([[0.0], DS])
        st = lambda i: self.probe(p, X, d0[i], zs[i - 1])  # state at grid index i >= 1
        # d_u, d_2: zero crossings of F
        pu = self.root(p, X, "F", st(iu), st(iu + 1), 1e-11) if iu >= 1 else st(1)
        p2 = self.root(p, X, "F", st(i2), st(i2 + 1), 1e-11)
        # F_pk: zero of dF/dD around the grid maximum (a kink, where dF/dD jumps, converges to the kink)
        a, b = st(max(ipk - 1, 1)), st(ipk + 1)
        ppk = self.root(p, X, "dFdD", a, b, 1e-9) if a["dFdD"] * b["dFdD"] < 0 else st(ipk)
        Eu, gEu = S.energy_grad(p, pu["z"], pu["d"])
        E2, gE2 = S.energy_grad(p, p2["z"], p2["d"])
        E, gE = Eu - E2, gEu - gE2
        snaps = []
        for ev in events:
            if ev["d0"] >= pu["d"] - 1e-9 and ev["d1"] <= p2["d"] + 1e-9:
                f = self.folds.locate(p, X, ev)
                if f is not None:
                    snaps.append(f)
                    E -= f["loss"]
                    gE = gE - f["dloss_dp"]
        out.update(feat=np.array([ppk["F"], p2["d"], E]), dfeat=np.vstack([ppk["dFdp"], -p2["dFdp"] / p2["dFdD"], gE]), nsnap=len(snaps),
                   snaps=snaps, d_u=pu["d"], d_pk=ppk["d"], pk_dFdD=ppk["dFdD"], E_endpoints=Eu - E2, n_events=len(events))
        return out

    def evaluate(self, p):
        p = np.asarray(p, float)
        if self.cache is not None and np.array_equal(self.cache["p"], p):
            return self.cache
        try:
            c = self.state_features(p)
        except Exception as e:
            print("  state_features failed:", repr(e), file=self.log, flush=True)
            c = dict(p=p.copy(), feat=None, dfeat=np.zeros((3, len(p))), switched=-1)
        self.cache = c
        self._log(c)
        return c


class MarginProblem(FoldProblem):
    """Arm AM (post-hoc, added 19:00): fold-aware features plus a fourth residual that keeps the design away from the
    pitchfork / fold surface where the loading path flips branch:
        rho = min_d lambda_min(d) / lambda_min(d = 0.125 mm)  >=  RHO_C ,   residual = max(0, 1 - rho / RHO_C).
    lambda_min is the lowest eigenvalue of the displacement-controlled tangent along the path up to the second rest
    position; d lambda / dp = d/dp [phi^T K(z(p), p) phi] with z(p) implicit (third derivatives of the energy by AD)."""
    RHO_C = 0.15

    def __init__(self, S=None, log=sys.stdout):
        super().__init__(S, log)
        Sv = self.S

        def lam_fn(p, z, d, phi, K):
            X = Sv.coords(p)
            zi = z - jnp.linalg.solve(K, Sv.residual(z, d, X))
            return jax.jvp(lambda zz: jnp.dot(phi, Sv.residual(zz, d, X)), (zi,), (phi,))[1]

        self._dlam = jax.jit(jax.value_and_grad(lam_fn))

    def lam(self, p, X, z, d):
        w, V = self.S._eigh(self.S.stiffness(z, float(d), X))
        l, g = self._dlam(jnp.asarray(p), z, float(d), V[:, 0], self.S.stiffness(z, float(d), X))
        return float(l), np.asarray(g)

    def margin(self, p, i_end):
        X, zs = self._last
        S = self.S
        ks = list(range(0, i_end, 4))
        lam = np.array([float(S._eigh(S.stiffness(zs[k], float(DS[k]), X))[0][0]) for k in ks])
        k = ks[int(np.argmin(lam))]
        cand = [kk for kk in (k - 2, k - 1, k, k + 1, k + 2) if 0 <= kk < i_end]
        lams = [float(S._eigh(S.stiffness(zs[kk], float(DS[kk]), X))[0][0]) for kk in cand]
        k = cand[int(np.argmin(lams))]
        lmin, gmin = self.lam(p, X, zs[k], DS[k])
        l0, g0 = self.lam(p, X, zs[0], DS[0])
        rho = lmin / l0
        return rho, gmin / l0 - lmin * g0 / l0**2, float(DS[k])

    def state_features(self, p):
        c = super().state_features(p)
        c["rho"], c["drho"] = None, np.zeros(len(c["p"]))
        if c["feat"] is not None:
            i2 = indices(c["F"])[2]
            c["rho"], c["drho"], c["d_rho"] = self.margin(c["p"], i2)
        return c

    def fun(self, p):
        c = self.evaluate(p)
        r = (c["feat"] / TGT - 1.0) if c["feat"] is not None else np.full(3, 30.0)
        m = max(0.0, 1.0 - c["rho"] / self.RHO_C) if c.get("rho") is not None else 0.0
        return np.concatenate([r, [m], np.asarray(PEN(jnp.asarray(p)))])

    def jac(self, p):
        c = self.evaluate(p)
        act = c.get("rho") is not None and c["rho"] < self.RHO_C
        row = -c["drho"] / self.RHO_C if act else np.zeros(len(c["p"]))
        return np.vstack([c["dfeat"] / TGT[:, None], row[None, :], np.asarray(PEN_JAC(jnp.asarray(p)))])

    def _log(self, c):
        super()._log(c)
        self.hist[-1]["rho"] = c.get("rho")
        if c.get("rho") is not None:
            print(f"       margin rho={c['rho']:.3f} at d={c.get('d_rho', 0):.2f}", file=self.log, flush=True)


# ---- drivers --------------------------------------------------------------------------------------------------
def trf(prob, p0, budget, xtol=1e-8):
    if budget < 2:
        return np.asarray(p0), "no budget"
    n0 = len(prob.hist)
    try:
        r = least_squares(prob.fun, p0, jac=prob.jac, bounds=(LB, UB), method="trf", max_nfev=budget, xtol=xtol, ftol=1e-10, gtol=1e-10)
        return r.x, r.message
    except Exception as e:
        good = [h for h in prob.hist[n0:] if h["feat"]]
        p = np.array(min(good, key=lambda h: np.sum((np.array(h["feat"]) / TGT - 1) ** 2))["p"]) if good else np.asarray(p0)
        return p, repr(e)


def best_seen(prob):
    good = [h for h in prob.hist if h["feat"]]
    if not good:
        return None
    return np.array(min(good, key=lambda h: np.max(np.abs(np.array(h["feat"]) / TGT - 1)))["p"])


def run(arm, name, p0, budget=60, log=sys.stdout):
    """arm in {B0, A, B0+O, A+O, B0-legacy}.  B0-legacy = trial 9's continuation as well.  Budget counts every design evaluation (one path solve each)."""
    t0 = time.time()
    arm_full = arm
    if "@" in arm:  # e.g. AM+O2@120: same arm with a different evaluation budget (exploratory)
        arm, budget = arm.split("@")[0], int(arm.split("@")[1])
    S = fem.Solver(nx=40, ny=2, Lh=design.LH, nk=NK, nm=NM, legacy=arm.endswith("legacy"))
    prob = (MarginProblem if arm.startswith("AM") else FoldProblem if arm.startswith("A") else GridProblem)(S=S, log=log)
    outer = arm.endswith("+O") or arm.endswith("+O2")
    # +O2 (added 17:45, after reading the first A+O trajectories; NOT pre-registered): identical to +O except that trf
    # gives up on a stall sooner (xtol 1e-4 instead of trial 9's 1e-8), leaving budget for the restarts
    xtol = 1e-4 if arm.endswith("+O2") else 1e-8
    rng = np.random.default_rng(12)  # fixed seed for the outer loop's perturbations
    p = np.clip(np.asarray(p0, float), LB, UB)
    events = []
    left = lambda: budget - len(prob.hist)
    err = lambda c: np.inf if c["feat"] is None else float(np.max(np.abs(c["feat"] / TGT - 1)))
    if outer:
        for k in range(3):  # (1) bifurcation escape
            c = prob.evaluate(p)
            if c["switched"] == 0:
                break
            xi = rng.standard_normal(len(p))
            p = np.clip(p + 0.05 * xi / np.linalg.norm(xi) * np.sqrt(len(p)) / 3.0, LB, UB)
            events.append(dict(kind="bifurcation_escape", at_eval=len(prob.hist)))
    restarts = 0
    while True:
        p, msg = trf(prob, p, left(), xtol)
        c = prob.evaluate(p) if (prob.cache is not None and np.array_equal(prob.cache["p"], p)) or left() > 0 else None
        if not outer or c is None or err(c) <= 0.01 or restarts >= 4 or left() < 5:
            break
        restarts += 1
        cands = []
        if c["feat"] is not None:  # (2) damped Gauss-Newton step across whatever stopped trf
            r0, J = prob.fun(p), prob.jac(p)
            dp = -np.linalg.solve(J.T @ J + 1e-2 * np.eye(len(p)), J.T @ r0)
            for a in (1.0, 0.5, 2.0):
                q = np.clip(p + a * dp, LB, UB)
                cq = prob.evaluate(q)
                if cq["feat"] is not None:
                    cands.append((err(cq), a, q))
        if cands:
            cands.sort(key=lambda t: t[0])
            events.append(dict(kind="gn_restart", at_eval=len(prob.hist), fraction=cands[0][1], err_before=err(c), err_after=cands[0][0]))
            p = cands[0][2]
        else:
            xi = rng.standard_normal(len(p))
            p = np.clip(p + 0.1 * xi / np.linalg.norm(xi) * np.sqrt(len(p)) / 3.0, LB, UB)
            events.append(dict(kind="random_restart", at_eval=len(prob.hist)))
    pb = best_seen(prob)
    p_final = pb if pb is not None else p
    sc = score(fem.Solver(nx=40, ny=2, Lh=design.LH, nk=NK, nm=NM) if S.legacy else S, p_final)  # one scorer for all arms
    res = dict(arm=arm_full, budget=budget, start=name, p0=np.asarray(p0).tolist(), p=np.asarray(p_final).tolist(), message=str(msg), nfev=len(prob.hist), npath=prob.npath,
               seconds=time.time() - t0, outer_events=events, score=sc, success=sc["success"],
               history=[dict(feat=h["feat"], switched=h["switched"], nsnap=h["nsnap"], p=h["p"], rho=h.get("rho")) for h in prob.hist])
    print(f"[{arm}] {name}: success={sc['success']} rel_err={None if sc['rel_err'] is None else np.round(sc['rel_err'], 4).tolist()} nfev={res['nfev']} {res['seconds']:.0f}s", file=log, flush=True)
    return res


def snap_starts(fresh_only=False):
    T9 = "../2026-11-14-metamaterial-inverse-design/results/"

    def cos(h, t, a2=0.0):
        p = fem.cosine_params(h, t)
        p[1] = a2
        return p

    def hinge(h, t0, b2, a2=0.2):
        p = cos(h, t0, a2)
        p[NK + 2] = b2
        return p

    cont = np.array(json.load(open(T9 + "opt_bistable_attempt5_A.json"))["continue_attempt3_best"]["p0"])
    fresh = {}
    rng = np.random.default_rng(2027)  # fresh starts for the post-hoc arm, drawn before that arm was run on anything
    for k in range(6):
        h, t, a2, b2 = rng.uniform(4.5, 6.5), rng.uniform(1.2, 2.1), rng.choice([-1, 1]) * rng.uniform(0.1, 0.4), rng.uniform(-0.5, 0.1)
        fresh[f"fresh_{k}"] = hinge(h, t, b2, a2)
    if fresh_only:
        return fresh
    return {**fresh, "cos_h5_t1.6": cos(5.0, 1.6), "cos_h6_t1.4": cos(6.0, 1.4), "cos_h4.5_t1.2": cos(4.5, 1.2),
            "h5_t1.6_a2": cos(5.0, 1.6, 0.3), "h5.5_t2.0_a2": cos(5.5, 2.0, 0.3), "h6_t1.4_a2": cos(6.0, 1.4, -0.3),
            "hinge_h5_t1.1_b-0.8": hinge(5.0, 1.1, -0.8), "continue_attempt3_best": cont, "hinge_h5.5_t1.4_b-0.7": hinge(5.5, 1.4, -0.7),
            "h6_t1.9_a2": cos(6.0, 1.9, 0.3), "hinge_h5_t1.5_b-0.4": hinge(5.0, 1.5, -0.4)}


if __name__ == "__main__":
    arm, name = sys.argv[1], sys.argv[2]
    res = run(arm, name, snap_starts()[name])
    json.dump(res, open(f"results/snap/{arm}__{name}.json", "w"), indent=1)
