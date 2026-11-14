"""Arc-length (pseudo-arclength) continuation with limit-point passing,
bifurcation detection from the tangent stiffness, and branch switching.

Solves G(u, lam) = 0 for a curve of solutions (u(s), lam(s)).

The caller supplies
    G(u, lam)      -> residual vector (n,)
    K(u, lam)      -> tangent stiffness dG/du (n, n), dense
    Glam(u, lam)   -> dG/dlam (n,)   (for dead loading: -q)
    monitor(u, lam, K) -> 1-D integer array of "stability counts".  Default:
        [number of negative eigenvalues of K].  A change of any entry between
        two converged points flags a critical point, which is then located by
        bisection on the arc-length step.

Dense linear algebra (numpy) on purpose: the problems in rung 1 and the 1-D
shell model in rung 3 have a few hundred to a few thousand unknowns, and
inertia counts need a factorisation/eigen-solve anyway.

Everything here is plain numpy; the JAX part (energy -> gradient -> Hessian by
automatic differentiation) lives in the problem files.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def neg_count(K: np.ndarray) -> np.ndarray:
    """Number of negative eigenvalues of the symmetric matrix K (its inertia)."""
    w = np.linalg.eigvalsh(0.5 * (K + K.T))
    return np.array([int(np.sum(w < 0.0))])


@dataclass
class Point:
    u: np.ndarray
    lam: float
    tu: np.ndarray          # unit tangent, displacement part
    tlam: float             # unit tangent, load part
    mon: np.ndarray         # stability counts
    s: float = 0.0          # accumulated arc length
    iters: int = 0


@dataclass
class Critical:
    kind: str               # "limit" | "bifurcation" | "monitor"
    u: np.ndarray
    lam: float
    mon_before: np.ndarray
    mon_after: np.ndarray
    phi: np.ndarray | None = None   # null vector of K (bifurcation only)
    s: float = 0.0
    info: dict = field(default_factory=dict)


class Continuation:
    def __init__(self, G, K, Glam, monitor=None, psi=1.0, uscale=1.0,
                 tol=1e-9, maxit=25, verbose=False):
        self.G, self.K, self.Glam = G, K, Glam
        self.monitor = monitor or (lambda u, lam, K: neg_count(K))
        self.psi = psi            # weight of lam in the arc-length metric
        self.uscale = uscale      # displacement scale in the metric
        self.tol, self.maxit, self.verbose = tol, maxit, verbose
        self.nsolve = 0

    # -- metric --------------------------------------------------------
    def _dot(self, au, al, bu, bl):
        return float(au @ bu) / self.uscale**2 + self.psi**2 * al * bl

    def tangent(self, u, lam, K=None, ref=None):
        """Unit tangent of the solution curve at (u, lam)."""
        K = self.K(u, lam) if K is None else K
        gl = self.Glam(u, lam)
        try:
            b = np.linalg.solve(K, -gl)
            tu, tl = b, 1.0
            if not np.all(np.isfinite(b)):
                raise np.linalg.LinAlgError
        except np.linalg.LinAlgError:
            # exactly singular K: tangent is the null vector, dlam = 0
            w, V = np.linalg.eigh(0.5 * (K + K.T))
            tu, tl = V[:, np.argmin(np.abs(w))], 0.0
        nrm = np.sqrt(self._dot(tu, tl, tu, tl))
        tu, tl = tu / nrm, tl / nrm
        if ref is not None and self._dot(tu, tl, ref[0], ref[1]) < 0:
            tu, tl = -tu, -tl
        return tu, tl

    # -- corrector ------------------------------------------------------
    def correct(self, u, lam, nu, nl, u0, lam0, ds_target):
        """Newton on [G(u,lam)=0 ; n.(x - x0) = ds_target] (bordered solve)."""
        dnorm = np.inf
        for it in range(self.maxit):
            g = self.G(u, lam)
            c = self._dot(nu, nl, u - u0, lam - lam0) - ds_target
            gn = np.linalg.norm(g)
            c_ok = abs(c) < 1e-10 * max(1.0, abs(ds_target))
            # converged on the residual (relative to the force scale), or
            # stagnated at round-off: Newton update negligible and residual small
            if c_ok and (gn <= self.tol * self.fscale or
                         (dnorm < 1e-12 and gn <= 1e4 * self.tol * self.fscale)):
                return u, lam, it, True
            K = self.K(u, lam)
            gl = self.Glam(u, lam)
            try:
                ab = np.linalg.solve(K, np.column_stack([-g, -gl]))
            except np.linalg.LinAlgError:
                return u, lam, it, False
            self.nsolve += 1
            a, b = ab[:, 0], ab[:, 1]
            dl = -(c + float(nu @ a) / self.uscale**2) / (self.psi**2 * nl + float(nu @ b) / self.uscale**2)
            du = a + dl * b
            if not (np.all(np.isfinite(du)) and np.isfinite(dl)):
                return u, lam, it, False
            u, lam = u + du, lam + dl
            dnorm = np.sqrt(self._dot(du, dl, du, dl))
        return u, lam, self.maxit, False

    def start(self, u0, lam0, direction=+1.0, fscale=1.0):
        self.fscale = fscale
        K = self.K(u0, lam0)
        tu, tl = self.tangent(u0, lam0, K)
        if tl * direction < 0:
            tu, tl = -tu, -tl
        return Point(u0.copy(), float(lam0), tu, tl, self.monitor(u0, lam0, K))

    def step(self, p: Point, ds: float):
        up, lp = p.u + ds * p.tu, p.lam + ds * p.tlam
        u, lam, it, ok = self.correct(up, lp, p.tu, p.tlam, p.u, p.lam, ds)
        if not ok:
            return None
        K = self.K(u, lam)
        tu, tl = self.tangent(u, lam, K, ref=(p.tu, p.tlam))
        # Near a bifurcation K is almost singular and the tangent solve
        # amplifies round-off along the null vector.  Guard: if the computed
        # tangent disagrees with the secant from the previous point, use the
        # secant (the path itself is smooth).
        su, sl = u - p.u, lam - p.lam
        sn = np.sqrt(self._dot(su, sl, su, sl))
        if sn > 0 and self._dot(tu, tl, su, sl) / sn < 0.7:
            tu, tl = su / sn, sl / sn
        return Point(u, lam, tu, tl, self.monitor(u, lam, K), p.s + ds, it)

    # -- locating critical points ----------------------------------------
    def _bisect(self, p: Point, ds: float, changed, nbis=48):
        """Find the largest step a in [0, ds] from p for which `changed(q)` is
        False.  Returns (point just before, point just after)."""
        lo, hi = 0.0, ds
        plo, phi = p, self.step(p, ds)
        for _ in range(nbis):
            mid = 0.5 * (lo + hi)
            q = self.step(p, mid)
            if q is None:
                break
            if changed(q):
                hi, phi = mid, q
            else:
                lo, plo = mid, q
            if hi - lo < 1e-13 * max(1.0, abs(ds)):
                break
        return plo, phi

    def locate(self, p: Point, q: Point):
        """Classify and locate what happened between converged points p, q."""
        out = []
        ds = q.s - p.s
        limit = p.tlam * q.tlam < 0
        dmon = q.mon - p.mon
        if limit:
            a, b = self._bisect(p, ds, lambda r: r.tlam * p.tlam < 0)
            out.append(Critical("limit", b.u, 0.5 * (a.lam + b.lam), a.mon, b.mon, s=b.s,
                                info={"lam_bracket": (a.lam, b.lam)}))
        # eigenvalue crossings of K itself (entry 0) not explained by the limit point
        n_bif = abs(int(dmon[0])) - (1 if limit else 0)
        if n_bif > 0:
            a, b = self._bisect(p, ds, lambda r: r.mon[0] != p.mon[0] and not
                                (limit and r.tlam * p.tlam < 0 and abs(r.mon[0] - p.mon[0]) == 1))
            Kb = self.K(b.u, b.lam)
            w, V = np.linalg.eigh(0.5 * (Kb + Kb.T))
            j = int(np.argmin(np.abs(w)))
            # primary-path tangent with the critical mode deflated out
            rhs = -self.Glam(b.u, b.lam)
            keep = np.arange(len(w)) != j
            tprim = V[:, keep] @ ((V[:, keep].T @ rhs) / w[keep])
            out.append(Critical("bifurcation", b.u, 0.5 * (a.lam + b.lam), a.mon, b.mon,
                                phi=V[:, j], s=b.s,
                                info={"lam_bracket": (a.lam, b.lam), "eig": float(w[j]),
                                      "tu": tprim, "tlam": 1.0,
                                      "phi_dot_q": float(V[:, j] @ rhs) / (np.linalg.norm(rhs) + 1e-300)}))
        # other monitors (e.g. per-harmonic stiffness in the shell model)
        for k in range(1, len(dmon)):
            if dmon[k] != 0:
                a, b = self._bisect(p, ds, lambda r, k=k: r.mon[k] != p.mon[k])
                out.append(Critical("monitor", b.u, 0.5 * (a.lam + b.lam), a.mon, b.mon, s=b.s,
                                    info={"index": k, "lam_bracket": (a.lam, b.lam)}))
        return out

    # -- driver -----------------------------------------------------------
    def run(self, p: Point, ds: float, nsteps: int, ds_min=1e-8, ds_max=None,
            stop=None, locate=True, it_target=5):
        path, crit = [p], []
        ds_max = ds_max or 20 * ds
        for _ in range(nsteps):
            q = None
            while q is None:
                q = self.step(p, ds)
                if q is None:
                    ds *= 0.5
                    if ds < ds_min:
                        if self.verbose:
                            print("step size underflow at lam=%.6g" % p.lam)
                        return path, crit
            if locate and (np.any(q.mon != p.mon) or p.tlam * q.tlam < 0):
                found = self.locate(p, q)
                crit.extend(found)
                if self.verbose:
                    for c in found:
                        print(f"  critical point: {c.kind} at lam={c.lam:.8g} mon {c.mon_before}->{c.mon_after}")
            path.append(q)
            p = q
            if stop is not None and stop(q):
                break
            if q.iters <= it_target - 2:
                ds = min(ds * 1.5, ds_max)
            elif q.iters > it_target + 2:
                ds *= 0.6
        return path, crit

    def switch_branch(self, c: Critical, eps: float, sign=+1.0):
        """Step off the primary path at bifurcation point c along its null
        vector phi (made orthogonal to the primary tangent), correct with the
        amplitude along phi held fixed, and return a Point on the new branch."""
        phi = c.phi.copy()
        tu = c.info["tu"]
        phi -= (phi @ tu) / (tu @ tu) * tu
        phi /= np.linalg.norm(phi)
        phi *= sign
        u0, lam0 = c.u, c.lam
        up = u0 + eps * phi
        # constraint normal: pure displacement direction phi, so that
        # _dot(nu, 0, du, 0) == phi . du  (amplitude along phi held at eps)
        nu = phi * self.uscale**2
        target = self._dot(nu, 0.0, up - u0, 0.0)
        u, lam, it, ok = self.correct(up, lam0, nu, 0.0, u0, lam0, target)
        if not ok:
            return None
        K = self.K(u, lam)
        ref_u, ref_l = u - u0, lam - lam0
        tu2, tl2 = self.tangent(u, lam, K, ref=(ref_u, ref_l))
        return Point(u, lam, tu2, tl2, self.monitor(u, lam, K), c.s + eps, it)


def load_control_newton(G, K, lam_values, u0, tol=1e-9, maxit=50, fscale=1.0):
    """Plain load-controlled Newton-Raphson, for the 'what goes wrong' demo.
    Returns list of dicts (lam, u, converged, iters)."""
    out, u = [], u0.copy()
    for lam in lam_values:
        v, ok = u.copy(), False
        for it in range(maxit):
            g = G(v, lam)
            if np.linalg.norm(g) <= tol * max(1.0, abs(lam)) * fscale:
                ok = True
                break
            try:
                dv = np.linalg.solve(K(v, lam), -g)
            except np.linalg.LinAlgError:
                break
            if not np.all(np.isfinite(dv)) or np.linalg.norm(dv) > 1e12:
                break
            v = v + dv
        out.append({"lam": float(lam), "u": v.copy() if ok else None, "converged": ok, "iters": it})
        if ok:
            u = v
    return out
