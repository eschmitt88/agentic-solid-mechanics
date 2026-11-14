"""Rung 2: programmed bistability — prescribed snap-through peak force, second stable position and return barrier."""
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
DS = np.round(np.arange(1, 105) * 0.125, 6)  # 0.125 .. 13 mm


def indices(F):
    """Sample indices that define the features (non-differentiable choices). None if the curve is not bistable."""
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
    """[F_pk, d_2, E_back] as differentiable functions of the sampled forces (indices frozen)."""
    xp = jnp if isinstance(F, jnp.ndarray) else np
    d = DS if d is None else d
    F0 = xp.concatenate([xp.zeros(1), F])
    d0 = np.concatenate([[0.0], d])
    ipk, iu, i2 = idx
    a, b, c = F0[ipk - 1], F0[ipk], F0[ipk + 1]
    den = a - 2 * b + c
    fpk = b - (c - a) ** 2 / (8 * den)  # vertex of the parabola through 3 equally spaced samples
    du = d0[iu] + (d0[iu + 1] - d0[iu]) * F0[iu] / (F0[iu] - F0[iu + 1])
    d2 = d0[i2] + (d0[i2 + 1] - d0[i2]) * F0[i2] / (F0[i2] - F0[i2 + 1])
    seg = 0.5 * (F0[iu + 1 : i2] + F0[iu + 2 : i2 + 1]) * (d0[iu + 2 : i2 + 1] - d0[iu + 1 : i2])
    E = -(0.5 * F0[iu + 1] * (d0[iu + 1] - du) + xp.sum(seg) + 0.5 * F0[i2] * (d2 - d0[i2]))
    return xp.stack([fpk, d2, E]) if xp is jnp else np.array([fpk, d2, E]), du


class Problem:
    def __init__(self, log=sys.stdout):
        self.S = fem.Solver(nx=40, ny=2, Lh=design.LH)
        self.cache = None
        self.hist = []
        self.log = log
        self.tgt = np.array([F_PK, D_2, E_BACK])

    def evaluate(self, p):
        p = np.asarray(p, float)
        if self.cache is not None and np.array_equal(self.cache[0], p):
            return self.cache
        F, us, infos = self.S.path(p, DS)
        ok = bool(np.all(np.isfinite(F)))
        idx = indices(F) if ok else None
        feat = features(F, idx)[0] if idx else None
        self.cache = (p.copy(), F, us, idx, feat)
        self.hist.append(dict(p=p.tolist(), feat=None if feat is None else feat.tolist(), bistable=idx is not None,
                              switched=int(sum(i["switched"] for i in infos))))
        print(f"  eval {len(self.hist):3d} feat={None if feat is None else np.round(feat, 5).tolist()} p={np.round(p, 3).tolist()}", file=self.log, flush=True)
        return self.cache

    def fun(self, p):
        _, F, us, idx, feat = self.evaluate(p)
        r = (feat / self.tgt - 1.0) if idx else np.full(3, 30.0)
        return np.concatenate([r, np.asarray(design._pen(jnp.asarray(p)))])

    def jac(self, p):
        _, F, us, idx, feat = self.evaluate(p)
        _, J = self.S.jacobian(p, DS, us)
        dfeat = np.asarray(jax.jacobian(lambda f: features(f, idx)[0])(jnp.asarray(F)))
        return np.vstack([(dfeat @ J) / self.tgt[:, None], np.asarray(design._pen_jac(jnp.asarray(p)))])


def optimise(starts, out="results/opt_bistable.json", max_nfev=40, log=sys.stdout):
    res = {}
    for name, p0 in starts.items():
        print(f"[bistable] start {name}", file=log, flush=True)
        prob = Problem(log)
        t0 = time.time()
        try:
            r = least_squares(prob.fun, p0, jac=prob.jac, bounds=(design.LB, design.UB), method="trf", max_nfev=max_nfev, xtol=1e-8, ftol=1e-10, gtol=1e-10)
            p, msg = r.x, r.message
        except Exception as e:
            good = [h for h in prob.hist if h["feat"]]
            p = np.array(min(good, key=lambda h: np.sum((np.array(h["feat"]) / prob.tgt - 1) ** 2))["p"]) if good else np.asarray(p0)
            msg = repr(e)
        _, F, us, idx, feat = prob.evaluate(p)
        res[name] = dict(p0=np.asarray(p0).tolist(), p=np.asarray(p).tolist(), feat=None if feat is None else feat.tolist(), target=prob.tgt.tolist(),
                         rel_err=None if feat is None else (feat / prob.tgt - 1).tolist(), message=msg, nfev=len(prob.hist),
                         seconds=time.time() - t0, F=F.tolist(), d=DS.tolist(), history=prob.hist)
        print(f"[bistable] {name}: feat={feat} msg={msg}", file=log, flush=True)
        json.dump(res, open(out, "w"), indent=1)
    return res


if __name__ == "__main__":
    # attempt 5 (after adaptive step bisection made evaluations repeatable; attempts 1-4 are in log.md and
    # results/opt_bistable_attempt*.json)
    def cos(h, t, a2=0.3):
        p = fem.cosine_params(h, t)
        p[1] = a2
        return p

    def hinge(h, t0, b2, a2=0.2):
        p = cos(h, t0, a2)
        p[fem.NK + 2] = b2
        return p

    best3 = json.load(open("results/opt_bistable_attempt3_stalled.json"))["h5.5_t2.0_a2"]["p"]
    sets = {"A": {"continue_attempt3_best": np.array(best3), "hinge_h5.5_t1.4_b-0.7": hinge(5.5, 1.4, -0.7)},
            "B": {"h6_t1.9_a2": cos(6.0, 1.9), "hinge_h5_t1.5_b-0.4": hinge(5.0, 1.5, -0.4)}}
    if sys.argv[1] == "C":
        # attempt 6: start on the far side of the non-smooth ridge (stalled design + its Gauss-Newton step, snap_diagnostic.py)
        dg = json.load(open("results/bistable_snap_diagnostic.json"))
        sets["C"] = {"across_ridge": np.array(dg["p"]) + np.array(dg["dp"])}
    which = sys.argv[1]
    optimise(sets[which], out=f"results/opt_bistable_attempt{6 if which == 'C' else 5}_{which}.json", max_nfev=60)
