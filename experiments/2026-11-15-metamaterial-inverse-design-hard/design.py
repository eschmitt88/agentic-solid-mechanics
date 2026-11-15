"""Targets, validity penalties and the curve-matching least-squares problem (nominal and robust) for trial 12.

Optimiser as in trial 9: scipy bounded trust-region least squares (`trf`) with the exact Jacobian from implicit
differentiation.  Robust variant: the residual vector is stacked over a fixed set of fabrication perturbations
(sample-average approximation with common random numbers).
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
KPA = 1000.0 / W_CELL
T_MIN, T_MAX, Y_MAX, SLOPE_MAX, KT_MAX = 0.4, 3.0, 9.0, 1.0, 0.8

# ---- targets (PREREGISTRATION.md) -------------------------------------------------------------------------
E_S_KPA, RHO = 2800.0, 0.055
GA_E, GA_SIG = E_S_KPA * RHO**2, 0.05 * E_S_KPA * RHO**2  # 8.47 kPa, 0.4235 kPa
_T2 = PchipInterpolator([0, 0.05, 0.10, 0.35, 0.42, 0.50], [0, 0.30, 0.40, 0.40, 0.55, 1.00])
_H2 = PchipInterpolator([0, 0.04, 0.08, 0.22, 0.27, 0.31, 0.45, 0.50], [0, 0.28, 0.35, 0.35, 0.55, 0.70, 0.70, 0.90])
_H4 = PchipInterpolator([0, 0.04, 0.12, 0.40, 0.50], [0, 0.20, -0.40, -0.40, 0.30])
TARGETS = {"T2": lambda e: _T2(e), "H1": lambda e: np.minimum(GA_E * e, GA_SIG), "H2": lambda e: _H2(e), "H4": lambda e: _H4(e)}
SPEC = {  # family, ncell, nk, nm, NRMSE tolerance, bands [(lo, hi, level, halfwidth)]
    "T2": dict(ncell=1, nk=4, nm=4, tol=0.05, bands=[(0.12, 0.33, 0.40, 0.06)]),
    "H1": dict(ncell=1, nk=6, nm=6, tol=0.03, bands=[(0.08, 0.50, GA_SIG, 0.05 * GA_SIG)]),
    "H2": dict(ncell=2, nk=6, nm=6, tol=0.04, bands=[(0.10, 0.20, 0.35, 0.028), (0.33, 0.43, 0.70, 0.056)]),
    "H4": dict(ncell=1, nk=6, nm=6, tol=0.05, bands=[]),
}


def nrmse(sig, tgt):
    return float(np.sqrt(np.mean((sig - tgt) ** 2)) / (tgt.max() - tgt.min()))


def grade(tname, sig):
    """Pre-registered pass rule applied to a sampled curve (any solver)."""
    sp, tgt = SPEC[tname], TARGETS[tname](EPS)
    sig = np.asarray(sig, float)
    if not np.all(np.isfinite(sig)):
        return dict(nrmse=None, bands=[], passed=False, note="incomplete curve")
    out = dict(nrmse=nrmse(sig, tgt), tol=sp["tol"], bands=[])
    ok = out["nrmse"] <= sp["tol"]
    for lo, hi, lvl, hw in sp["bands"]:
        m = (EPS >= lo - 1e-9) & (EPS <= hi + 1e-9)
        dev = float(np.max(np.abs(sig[m] - lvl)))
        out["bands"].append(dict(lo=lo, hi=hi, level=lvl, max_dev=dev, tol=hw, ok=bool(dev <= hw)))
        ok = ok and dev <= hw
    out["passed"] = bool(ok)
    return out


def travel_grid(ncell):
    return EPS * H_CELL * ncell


# ---- validity penalty (per cell) ---------------------------------------------------------------------------
_s = jnp.linspace(0, 1, 81)


def make_penalty(nk, nm, ncell):
    npc = nk + 1 + nm

    def one(p):
        t = fem.thickness(p, _s, nk)
        y, dydx = fem.centerline(p, _s, LH, nk)
        d2 = jax.vmap(jax.grad(lambda s: fem.centerline(p, s, LH, nk)[1]))(_s) / LH
        kappa = jnp.abs(d2) / (1 + dydx**2) ** 1.5
        h = lambda x: jnp.maximum(x, 0.0)
        return 30.0 * jnp.concatenate([h(t - T_MAX), h(T_MIN - t) / T_MIN, h(jnp.abs(y) - Y_MAX), h(jnp.abs(dydx) - SLOPE_MAX), h(kappa * t - KT_MAX)])

    pen = lambda p: jnp.concatenate([one(q) for q in jnp.reshape(p, (ncell, npc))])
    lb = np.tile(np.array([-Y_MAX] * nk + [np.log(T_MIN)] + [-1.0] * nm), ncell)
    ub = np.tile(np.array([Y_MAX] * nk + [np.log(T_MAX)] + [1.0] * nm), ncell)
    return jax.jit(pen), jax.jit(jax.jacfwd(pen)), lb, ub


def fab_sd(nk, nm, ncell=1):
    """Fabrication-error model: 0.05 mm on the first 4 centre-line modes, 3 % thickness, 0.02 on the first 4 thickness modes."""
    sd = np.zeros(nk + 1 + nm)
    sd[:4] = 0.05
    sd[nk] = 0.03
    sd[nk + 1: nk + 5] = 0.02
    return np.tile(sd, ncell)


def fab_samples(n, seed, nk, nm, ncell=1):
    rng = np.random.default_rng(seed)
    sd = fab_sd(nk, nm, ncell)
    return sd * rng.standard_normal((n, len(sd)))


class Problem:
    def __init__(self, tname, S=None, offsets=None, log=None):
        sp = SPEC[tname]
        self.tname, self.sp = tname, sp
        self.S = S or fem.Solver(nx=40, ny=2, Lh=LH, nk=sp["nk"], nm=sp["nm"], ncell=sp["ncell"])
        self.DS = travel_grid(sp["ncell"])
        self.tgt = TARGETS[tname](EPS)
        self.rng = self.tgt.max() - self.tgt.min()
        self.pen, self.pen_jac, self.lb, self.ub = make_penalty(sp["nk"], sp["nm"], sp["ncell"])
        self.offsets = [np.zeros(len(self.lb))] if offsets is None else [np.zeros(len(self.lb))] + list(offsets)
        self.cache, self.hist, self.log = None, [], log

    def curve(self, p):
        F, zs, infos = self.S.path(p, self.DS)
        sig = F * KPA
        bad = int(sum((not i["converged"]) or i.get("unstable", False) for i in infos))
        return sig, zs, dict(bad=bad, switched=int(sum(i["switched"] for i in infos)), jumps=int(sum("jump" in i for i in infos)))

    def evaluate(self, p):
        p = np.asarray(p, float)
        if self.cache is not None and np.array_equal(self.cache[0], p):
            return self.cache
        t0 = time.time()
        sigs, states, metas = [], [], []
        for off in self.offsets:
            sig, zs, meta = self.curve(p + off)
            sigs.append(np.where(np.isfinite(sig), sig, 1e3))
            states.append(zs)
            metas.append(meta)
        self.cache = (p.copy(), sigs, states)
        errs = [nrmse(s, self.tgt) for s in sigs]
        rec = dict(p=p.tolist(), nrmse=errs[0], mean_sq=float(np.mean(np.square(errs))), bad=metas[0]["bad"], switched=metas[0]["switched"],
                   jumps=metas[0]["jumps"], sec=time.time() - t0)
        self.hist.append(rec)
        if self.log:
            print(f"  eval {len(self.hist):3d} nrmse={rec['nrmse']:.4f} rms_over_copies={np.sqrt(rec['mean_sq']):.4f} bad={rec['bad']} sw={rec['switched']} "
                  f"jumps={rec['jumps']} {rec['sec']:.1f}s", file=self.log, flush=True)
        return self.cache

    def fun(self, p):
        _, sigs, _ = self.evaluate(p)
        n, k = len(self.tgt), len(self.offsets)
        r = np.concatenate([(s - self.tgt) / self.rng / np.sqrt(n * k) for s in sigs])
        return np.concatenate([r, np.asarray(self.pen(jnp.asarray(p)))])

    def jac(self, p):
        p = np.asarray(p, float)
        _, sigs, states = self.evaluate(p)
        n, k = len(self.tgt), len(self.offsets)
        rows = []
        for off, zs in zip(self.offsets, states):
            _, J = self.S.jacobian(p + off, self.DS, zs)
            rows.append(J * KPA / self.rng / np.sqrt(n * k))
        return np.vstack(rows + [np.asarray(self.pen_jac(jnp.asarray(p)))])


def cos(h, t, nk, nm, a2=0.0):
    p = fem.cosine_params(h, t, nk, nm)
    p[1] = a2
    return p


def starts(tname):
    sp = SPEC[tname]
    nk, nm = sp["nk"], sp["nm"]
    if sp["ncell"] == 1:
        return {"sag3": cos(-3, 1.2, nk, nm), "flat": cos(-0.3, 1.2, nk, nm), "arch3": cos(3, 1.2, nk, nm), "arch6": cos(6, 1.5, nk, nm),
                "arch8": cos(8, 1.5, nk, nm), "wavy": cos(6, 2.0, nk, nm, a2=-2.0)}
    pair = lambda a, b: np.concatenate([cos(*a, nk, nm), cos(*b, nk, nm)])
    return {"A4_B6": pair((4, 1.2), (6, 2.0)), "A3_B5": pair((3, 1.0), (5, 1.8)), "A6_B6": pair((6, 1.5), (6, 2.2)), "Aflat_B6": pair((-0.3, 1.2), (6, 1.5))}


def run(tname, sname, p0, offsets=None, max_nfev=80, log=sys.stdout, S=None, prob=None):
    prob = prob or Problem(tname, S=S, offsets=offsets, log=log)
    t0 = time.time()
    print(f"[{tname}] start {sname}", file=log, flush=True)
    try:
        r = least_squares(prob.fun, p0, jac=prob.jac, bounds=(prob.lb, prob.ub), method="trf", x_scale=1.0, max_nfev=max_nfev,
                          xtol=1e-6, ftol=1e-7, gtol=1e-8)
        p, status, msg = r.x, int(r.status), r.message
    except Exception as e:
        key = "mean_sq" if offsets is not None else "nrmse"
        p = np.array(min(prob.hist, key=lambda h: h[key])["p"]) if prob.hist else np.asarray(p0)
        status, msg = -99, repr(e)
    f, J = prob.fun(p), prob.jac(p)
    g = J.T @ f
    # projected gradient (bounds): stationarity measure of the smooth problem at the end point
    free = ~(((p <= prob.lb + 1e-9) & (g > 0)) | ((p >= prob.ub - 1e-9) & (g < 0)))
    sig = prob.cache[1][0]
    res = dict(target=tname, start=sname, p0=np.asarray(p0).tolist(), p=np.asarray(p).tolist(), nrmse=nrmse(sig, prob.tgt),
               rms_over_copies=float(np.sqrt(np.mean([nrmse(s, prob.tgt) ** 2 for s in prob.cache[1]]))), status=status, message=msg,
               nfev=len(prob.hist), seconds=time.time() - t0, sigma=np.asarray(sig).tolist(), penalty=float(np.sum(np.asarray(prob.pen(jnp.asarray(p))) ** 2)),
               proj_grad_norm=float(np.linalg.norm(g[free])), loss=float(0.5 * f @ f), design_grade=grade(tname, sig),
               history=[dict(nrmse=h["nrmse"], mean_sq=h["mean_sq"], bad=h["bad"], switched=h["switched"], jumps=h["jumps"]) for h in prob.hist])
    print(f"[{tname}] {sname}: nrmse={res['nrmse']:.4f} nfev={res['nfev']} status={status} {msg} |proj grad|={res['proj_grad_norm']:.2e}", file=log, flush=True)
    return res


if __name__ == "__main__":
    tname, sname = sys.argv[1], sys.argv[2]
    res = run(tname, sname, starts(tname)[sname])
    json.dump(res, open(f"results/opt/{tname}__{sname}.json", "w"), indent=1)
