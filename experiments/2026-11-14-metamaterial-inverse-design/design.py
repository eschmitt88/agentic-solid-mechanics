"""Rung 1/2: gradient-based inverse design of the curved-beam cell.

Targets are the pre-registered curves (PREREGISTRATION.md).  Optimiser:
bounded Levenberg-Marquardt / trust-region least squares (scipy `trf`) on the
50 stress residuals, with the exact Jacobian dF_k/dp from implicit
differentiation of the converged equilibria (fem.Solver.jacobian).
"""
from __future__ import annotations

import json
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import least_squares

import fem

W_CELL, H_CELL, LH, WS = 70.0, 25.0, 28.0, 4.0
EPS = np.round(np.arange(1, 51) * 0.01, 4)
DS = EPS * H_CELL
KPA = 1000.0 / W_CELL  # F [N/mm] -> sigma [kPa]
T_MIN, T_MAX, Y_MAX = 0.4, 3.0, 9.0


def target_T1(eps):
    a = 2.0 / (np.exp(2.5) - 1.0)
    return a * (np.exp(5.0 * eps) - 1.0)


_T2 = PchipInterpolator([0, 0.05, 0.10, 0.35, 0.42, 0.50], [0, 0.30, 0.40, 0.40, 0.55, 1.00])
_T3 = PchipInterpolator([0, 0.07, 0.32, 0.50], [0, 0.60, 0.15, 0.90])
TARGETS = {"T1": target_T1, "T2": lambda e: _T2(e), "T3": lambda e: _T3(e)}


def nrmse(sig, tgt):
    return float(np.sqrt(np.mean((sig - tgt) ** 2)) / (tgt.max() - tgt.min()))


# bound penalties (smooth in p): thickness and centre-line limits sampled along the beam
_s = jnp.linspace(0, 1, 81)


SLOPE_MAX, KT_MAX = 1.0, 0.8  # added 21:12 after the first T2/T3 attempts folded the mesh (see log.md)


def _penalty(p):
    t = fem.thickness(p, _s)
    y, dydx = fem.centerline(p, _s, LH)
    d2 = jax.vmap(jax.grad(lambda s: fem.centerline(p, s, LH)[1]))(_s) / LH
    kappa = jnp.abs(d2) / (1 + dydx**2) ** 1.5
    h = lambda x: jnp.maximum(x, 0.0)
    return 30.0 * jnp.concatenate([h(t - T_MAX), h(T_MIN - t) / T_MIN, h(jnp.abs(y) - Y_MAX), h(jnp.abs(dydx) - SLOPE_MAX), h(kappa * t - KT_MAX)])


_pen = jax.jit(_penalty)
_pen_jac = jax.jit(jax.jacfwd(_penalty))

LB = np.array([-Y_MAX] * fem.NK + [np.log(T_MIN)] + [-1.0] * fem.NM)
UB = np.array([Y_MAX] * fem.NK + [np.log(T_MAX)] + [1.0] * fem.NM)


class Problem:
    def __init__(self, tgt_sigma, solver=None, log=None):
        self.S = solver or fem.Solver(nx=40, ny=2, Lh=LH)
        self.tgt = np.asarray(tgt_sigma)
        self.rng = self.tgt.max() - self.tgt.min()
        self.cache = None
        self.hist = []
        self.log = log

    def evaluate(self, p):
        p = np.asarray(p, float)
        if self.cache is not None and np.array_equal(self.cache[0], p):
            return self.cache
        t0 = time.time()
        F, us, infos = self.S.path(p, DS)
        sig = F * KPA
        if not np.all(np.isfinite(sig)):
            sig = np.where(np.isfinite(sig), sig, 1e3)  # failed solve: huge residual, step gets rejected
        bad = sum((not i["converged"]) or i.get("unstable", False) for i in infos)
        sw = sum(i["switched"] for i in infos)
        self.cache = (p.copy(), sig, us, infos)
        rec = dict(p=p.tolist(), nrmse=nrmse(sig, self.tgt), bad=int(bad), switched=int(sw), sec=time.time() - t0, sigma=sig.tolist())
        self.hist.append(rec)
        if self.log:
            print(f"  eval {len(self.hist):3d} nrmse={rec['nrmse']:.4f} bad={bad} sw={sw} {rec['sec']:.1f}s p={np.round(p, 3).tolist()}", file=self.log, flush=True)
        return self.cache

    def fun(self, p):
        _, sig, _, _ = self.evaluate(p)
        r = (sig - self.tgt) / self.rng / np.sqrt(len(sig))  # sum r^2 = NRMSE^2
        return np.concatenate([r, np.asarray(_pen(jnp.asarray(p)))])

    def jac(self, p):
        _, sig, us, _ = self.evaluate(p)
        _, J = self.S.jacobian(p, DS, us)
        J = J * KPA / self.rng / np.sqrt(len(sig))
        return np.vstack([J, np.asarray(_pen_jac(jnp.asarray(p)))])


def start_points():
    """Generic starts spanning the family: sagging, straight, shallow arch, deep arch."""
    P = {}
    for name, h, t in [("sag3", -3.0, 1.2), ("flat", -0.3, 1.2), ("arch3", 3.0, 1.2), ("arch6", 6.0, 1.5)]:
        P[name] = fem.cosine_params(h, t)
    return P


def optimise(tname, starts=None, max_nfev=60, out=None, log=sys.stdout):
    tgt = TARGETS[tname](EPS)
    S = fem.Solver(nx=40, ny=2, Lh=LH)
    res = {}
    for sname, p0 in (starts or start_points()).items():
        print(f"[{tname}] start {sname}", file=log, flush=True)
        prob = Problem(tgt, S, log=log)
        t0 = time.time()
        try:
            r = least_squares(prob.fun, p0, jac=prob.jac, bounds=(LB, UB), method="trf", x_scale=1.0, max_nfev=max_nfev,
                              xtol=1e-6, ftol=1e-7, gtol=1e-8)
            p = r.x
            status = int(r.status)
            msg = r.message
        except Exception as e:  # keep the failure record
            best = min(prob.hist, key=lambda h: h["nrmse"]) if prob.hist else None
            p = np.array(best["p"]) if best else p0
            status, msg = -99, repr(e)
        _, sig, _, infos = prob.evaluate(p)
        res[sname] = dict(p0=np.asarray(p0).tolist(), p=np.asarray(p).tolist(), nrmse=nrmse(sig, tgt), status=status, message=msg,
                          nfev=len(prob.hist), seconds=time.time() - t0, sigma=sig.tolist(),
                          history=[dict(nrmse=h["nrmse"], bad=h["bad"], switched=h["switched"], p=h["p"]) for h in prob.hist],
                          final_bad=int(sum((not i["converged"]) or i.get("unstable", False) for i in infos)),
                          final_switched=int(sum(i["switched"] for i in infos)))
        print(f"[{tname}] {sname}: nrmse={res[sname]['nrmse']:.4f} nfev={res[sname]['nfev']} status={status} {msg}", file=log, flush=True)
        if out:
            json.dump(res, open(out, "w"), indent=1)
    return res


if __name__ == "__main__":
    t = sys.argv[1]
    optimise(t, out=f"results/opt_{t}.json")
