"""2D contact solver written for trial 10 (JAX, float64, CPU).

Solids: 4-node quads, plane strain or axisymmetric; small-strain linear
elasticity or finite-strain neo-Hookean.  Only the strain energy is
hand-written; forces and tangents come from jax.grad / jax.hessian.

Contact: integral ("Gauss-point-to-segment") formulation.  Contact
tractions are evaluated at Gauss points of the slave surface, each point
is projected onto the closest segment of the master polyline.  Normal
contact and Coulomb friction are enforced by an augmented Lagrangian with
Uzawa updates (penalty regularisation inside a Newton solve, multipliers
updated outside until gap and stick-slip error are below tolerance), so the
converged answer does not depend on the penalty value.  The friction
residual is not derivable from a potential; its (non-symmetric) tangent is
obtained with jax.jacfwd per contact point.
"""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
# XLA's CPU thread pool ignores OMP_NUM_THREADS (it grabbed 7+ cores) and, when pinned with
# taskset, deadlocked on the 60k-element vmap.  One XLA thread; the sparse solver keeps
# MKL_NUM_THREADS.
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
import numpy as np
import scipy.sparse as sp
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)
try:
    import pypardiso
    def _solve(A, b):
        return pypardiso.spsolve(A.tocsr(), b)
except Exception:  # pragma: no cover
    from scipy.sparse.linalg import spsolve as _sps
    def _solve(A, b):
        return _sps(A.tocsc(), b)

_GP = np.array([-1.0, 1.0]) / np.sqrt(3.0)


def _shape(xi, eta):
    N = 0.25 * jnp.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta), (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])
    dN = 0.25 * jnp.array([[-(1 - eta), -(1 - xi)], [(1 - eta), -(1 + xi)], [(1 + eta), (1 + xi)], [-(1 + eta), (1 - xi)]])
    return N, dN


def make_element_energy(kind, axisym=False, nh_form="ccx"):
    """Return W_el(u_el(4,2), X_el(4,2), E, nu) -> scalar strain energy."""
    def w_el(u, X, E, nu):
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))
        K = E / (3 * (1 - 2 * nu))
        W = 0.0
        for xi in _GP:
            for eta in _GP:
                N, dN = _shape(xi, eta)
                J = X.T @ dN
                detJ = J[0, 0] * J[1, 1] - J[0, 1] * J[1, 0]
                dNdX = dN @ jnp.linalg.inv(J)
                H = u.T @ dNdX  # displacement gradient
                if axisym:
                    r = N @ X[:, 0]
                    ur = N @ u[:, 0]
                    hoop = ur / r
                    wt = detJ * 2 * jnp.pi * r
                else:
                    hoop = 0.0
                    wt = detJ
                if kind == "linear":
                    e = 0.5 * (H + H.T)
                    tr = e[0, 0] + e[1, 1] + hoop
                    psi = 0.5 * lam * tr ** 2 + mu * (jnp.sum(e * e) + hoop ** 2)
                else:  # neo-Hookean, plane strain (F33 = 1)
                    F = jnp.eye(2) + H
                    Jd = F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]
                    I1 = jnp.sum(F * F) + 1.0
                    if nh_form == "ccx":   # C10 (I1bar - 3) + (J-1)^2 / D1, C10 = mu/2, D1 = 2/K
                        psi = 0.5 * mu * (I1 * Jd ** (-2.0 / 3.0) - 3.0) + 0.5 * K * (Jd - 1.0) ** 2
                    else:                  # mu/2 (I1-3) - mu ln J + lam/2 (ln J)^2
                        lj = jnp.log(Jd)
                        psi = 0.5 * mu * (I1 - 3.0) - mu * lj + 0.5 * lam * lj ** 2
                W = W + psi * wt
        return W
    return w_el


def _contact_forces(xsurf, s1, s2, m1, m2, a1, a2, Ns, wt, Winv, hist, lamN, lamT, fstick, epsN, epsT, mu, nsign, nsl):
    """Global contact force vector on the surface nodes (nsurf,2) and nodal state.
    Nodal ("mortar-like") multipliers on the slave nodes, tractions interpolated
    with the slave shape functions and integrated at fixed slave Gauss points."""
    x = xsurf.reshape(-1, 2)
    xs1, xs2, xm1, xm2 = x[s1], x[s2], x[m1], x[m2]
    xs = Ns[:, :1] * xs1 + Ns[:, 1:] * xs2
    d = xm2 - xm1
    L = jnp.sqrt(jnp.sum(d * d, axis=1))
    t = d / L[:, None]
    n = nsign * jnp.stack([-t[:, 1], t[:, 0]], axis=1)
    xi = jnp.clip(jnp.sum((xs - xm1) * t, axis=1) / L, 0.0, 1.0)
    xp = xm1 + xi[:, None] * d
    g = jnp.sum((xs - xp) * n, axis=1)
    # slip since the last converged step, measured along the current master surface.
    # hist = (mode, xi0, o1, o2, mid): mode 0 no history, 1 same segment, 2 moved forward,
    # 3 moved backward; o1,o2 nodes of the old segment; mid = length of segments in between.
    mode, xi0, o1, o2, mid = hist
    d0 = x[o2] - x[o1]
    L0 = jnp.sqrt(jnp.sum(d0 * d0, axis=1))
    gT = jnp.where(mode == 1, (xi - xi0) * L,
         jnp.where(mode == 2, xi * L + mid + (1 - xi0) * L0,
         jnp.where(mode == 3, -((1 - xi) * L + mid + xi0 * L0), 0.0)))
    # weighted nodal averages
    gA = (jnp.zeros(nsl).at[a1].add(wt * Ns[:, 0] * g).at[a2].add(wt * Ns[:, 1] * g)) * Winv
    gTA = (jnp.zeros(nsl).at[a1].add(wt * Ns[:, 0] * gT).at[a2].add(wt * Ns[:, 1] * gT)) * Winv
    arg = lamN - epsN * gA
    pA = jnp.where(arg > 0, arg, 0.0)
    lim = mu * pA
    ttr = lamT + epsT * gTA
    # explicit return map (jnp.clip has a spurious unit derivative when lim = 0 = ttr,
    # which put tangential stiffness on every node that is not in contact)
    # fstick: nodes whose slip direction reversed since the previous Newton iteration are
    # linearised on the stick branch (used for the tangent only, never for the residual)
    tA = jnp.where((jnp.abs(ttr) < lim) | (fstick > 0), ttr, lim * jnp.sign(ttr))
    p = Ns[:, 0] * pA[a1] + Ns[:, 1] * pA[a2]
    tt = Ns[:, 0] * tA[a1] + Ns[:, 1] * tA[a2]
    fs = (p[:, None] * n - tt[:, None] * t) * wt[:, None]
    F = jnp.zeros_like(x)
    F = F.at[s1].add(Ns[:, :1] * fs).at[s2].add(Ns[:, 1:] * fs)
    F = F.at[m1].add(-(1 - xi)[:, None] * fs).at[m2].add(-xi[:, None] * fs)
    return F.reshape(-1), (gA, gTA, pA, tA, g, xi, p, tt)


_cf = jax.jit(_contact_forces, static_argnames=("nsl",))
_cf_jac = jax.jit(jax.jacfwd(lambda *a, **k: _contact_forces(*a, **k)[0]), static_argnames=("nsl",))


class ContactPair:
    """slave_edges (ns,2) node ids; master_chain (nm+1,) node ids in order.
    nsign = +1 if the outward master normal is the left normal of the chain direction."""
    def __init__(self, slave_edges, master_chain, X, mu=0.0, epsN=None, epsT=None, nsign=1.0, ngp=6, axisym=False):
        self.se = np.asarray(slave_edges)
        self.mc = np.asarray(master_chain)
        self.mu, self.epsN, self.epsT, self.nsign = mu, epsN, epsN if epsT is None else epsT, nsign
        gp, gw = np.polynomial.legendre.leggauss(ngp)
        ns = len(self.se)
        self.edge = np.repeat(np.arange(ns), ngp)
        s = np.tile(0.5 * (gp + 1), ns)
        self.Ns = np.stack([1 - s, s], axis=1)
        Lref = np.linalg.norm(X[self.se[:, 1]] - X[self.se[:, 0]], axis=1)
        self.wt = np.repeat(Lref, ngp) * np.tile(0.5 * gw, ns)
        self.Xgp = self.Ns[:, :1] * X[self.se[self.edge, 0]] + self.Ns[:, 1:] * X[self.se[self.edge, 1]]
        if axisym:
            self.wt = self.wt * 2 * np.pi * self.Xgp[:, 0]
        # local numbering of the surface nodes: slave nodes first, then master nodes
        self.snodes = np.unique(self.se)
        self.surf = np.concatenate([self.snodes, self.mc])
        loc = {n: i for i, n in enumerate(self.snodes)}
        self.a1 = np.array([loc[n] for n in self.se[self.edge, 0]]); self.a2 = np.array([loc[n] for n in self.se[self.edge, 1]])
        self.nsl = len(self.snodes)
        W = np.zeros(self.nsl)
        np.add.at(W, self.a1, self.wt * self.Ns[:, 0]); np.add.at(W, self.a2, self.wt * self.Ns[:, 1])
        self.W = W; self.Winv = 1.0 / np.where(W > 0, W, 1.0)   # W = 0 only for a node on the symmetry axis
        self.Xn = X[self.snodes]
        self.dofs = (2 * self.surf[:, None] + np.arange(2)[None, :]).reshape(-1)
        n = len(self.wt)
        self.lamN = np.zeros(self.nsl); self.lamT = np.zeros(self.nsl)
        self.seg0 = -np.ones(n, int); self.xi0 = np.zeros(n)   # stick point at last converged step
        self.state = None

    def search(self, x):
        xs = self.Ns[:, :1] * x[self.se[self.edge, 0]] + self.Ns[:, 1:] * x[self.se[self.edge, 1]]
        a = x[self.mc[:-1]]; d = x[self.mc[1:]] - a
        L2 = np.sum(d * d, axis=1)
        best = np.zeros(len(xs), int)
        CH = 4000
        for i in range(0, len(xs), CH):
            r = xs[i:i + CH, None, :] - a[None]
            xi = np.clip(np.sum(r * d[None], axis=2) / L2[None], 0, 1)
            q = r - xi[..., None] * d[None]
            best[i:i + CH] = np.argmin(np.sum(q * q, axis=2), axis=1)
        return best, xs

    def evaluate(self, x, tangent=True):
        seg, xs = self.search(x)
        xm1, xm2 = x[self.mc[seg]], x[self.mc[seg + 1]]
        d = xm2 - xm1; L = np.linalg.norm(d, axis=1)
        xi = np.clip(np.sum((xs - xm1) * d, axis=1) / L ** 2, 0, 1)
        nrm = self.nsign * np.stack([-d[:, 1], d[:, 0]], axis=1) / L[:, None]
        gnow = np.sum((xs - (xm1 + xi[:, None] * d)) * nrm, axis=1)
        # first touch inside this step: freeze the stick reference where it touched
        # (tried: freeze the stick reference at the Newton iterate where a point first touches.
        #  The iterate can be far from equilibrium, the frozen offset then drives a slip
        #  flip-flop.  Points without committed history therefore carry no friction during
        #  the step in which they first touch; they get their reference at commit.)
        Lall = np.linalg.norm(x[self.mc[1:]] - x[self.mc[:-1]], axis=1)
        cum = np.concatenate([[0.0], np.cumsum(Lall)])
        s0 = np.maximum(self.seg0, 0)
        mode = np.where(self.seg0 < 0, 0, np.where(seg == s0, 1, np.where(seg > s0, 2, 3)))
        lo, hi = np.minimum(seg, s0), np.maximum(seg, s0)
        mid = np.where(mode >= 2, cum[hi] - cum[lo + 1], 0.0)
        hist = (mode, self.xi0, self.nsl + s0, self.nsl + s0 + 1, mid)
        nsl = self.nsl
        pre = (x[self.surf].reshape(-1), self.a1, self.a2, nsl + seg, nsl + seg + 1, self.a1, self.a2, self.Ns, self.wt,
               self.Winv, hist, self.lamN, self.lamT)
        post = (self.epsN, self.epsT, self.mu, self.nsign)
        F, st = _cf(*pre, np.zeros(nsl), *post, nsl=nsl)
        self.state = [np.asarray(v) for v in st]; self.seg = seg
        f = np.zeros(x.size)
        np.add.at(f, self.dofs, np.asarray(F))
        if not tangent:
            return f, None
        pA, tA = self.state[2], self.state[3]
        sgn = np.where((pA > 0) & (np.abs(tA) >= self.mu * pA * (1 - 1e-12)), np.sign(tA), 0.0)
        flip = (sgn * getattr(self, "prev_sign", np.zeros(nsl)) < 0).astype(float)
        self.prev_sign = sgn
        J = np.asarray(_cf_jac(*pre, flip, *post, nsl=nsl))
        nz = np.nonzero(J)
        K = sp.coo_matrix((J[nz], (self.dofs[nz[0]], self.dofs[nz[1]])), shape=(x.size, x.size)).tocsr()
        return f, K

    def augment(self):
        """Uzawa update; returns (max mean penetration of active nodes, max slip of sticking nodes)."""
        gA, gTA, pA, tA = self.state[:4]
        active = pA > 0
        pen = np.max(np.abs(gA[active])) if active.any() else 0.0
        stick = active & (np.abs(tA) < self.mu * pA * (1 - 1e-12))
        slip = np.max(np.abs(gTA[stick])) if stick.any() else 0.0
        self.lamN = pA.copy(); self.lamT = tA.copy()
        return pen, slip

    def commit(self):
        gA, gTA, pA, tA, g, xi, p, tt = self.state
        active = p > 0
        self.seg0 = np.where(active, self.seg, -1)
        self.xi0 = xi.copy()
        self.lamN = pA.copy(); self.lamT = tA.copy()

    def tractions(self):
        gA, gTA, pA, tA, g, xi, p, tt = self.state
        return dict(X=self.Xn, p=pA, q=tA, g=gA, area=self.W, Xgp=self.Xgp, p_gp=p, q_gp=tt, w_gp=self.wt)


class Model:
    def __init__(self, X, quads, E, nu, kind="linear", axisym=False, nh_form="ccx"):
        self.X = np.asarray(X, float); self.quads = np.asarray(quads)
        ne = len(self.quads)
        self.E = np.broadcast_to(np.asarray(E, float), (ne,)).copy()
        self.nu = np.broadcast_to(np.asarray(nu, float), (ne,)).copy()
        w = make_element_energy(kind, axisym, nh_form)
        self._g = jax.jit(jax.vmap(jax.grad(w), in_axes=(0, 0, 0, 0)))
        self._h = jax.jit(jax.vmap(jax.hessian(w), in_axes=(0, 0, 0, 0)))
        self.ndof = 2 * len(self.X)
        ed = (2 * self.quads[:, :, None] + np.arange(2)[None, None, :]).reshape(ne, 8)
        self._ed = ed
        self._r = np.repeat(ed, 8, axis=1).reshape(-1)
        self._c = np.tile(ed, (1, 8)).reshape(-1)
        self.pairs = []
        self.fixed = {}          # dof -> value
        self.rep = np.arange(self.ndof)   # tie map
        self.fext = np.zeros(self.ndof)
        self.u = np.zeros(self.ndof)
        self.linear = (kind == "linear")
        self._Klin = None

    def internal(self, u, tangent=True):
        U = u.reshape(-1, 2)[self.quads]
        Xe = self.X[self.quads]
        g = np.asarray(self._g(U, Xe, self.E, self.nu)).reshape(len(self.quads), 8)
        f = np.zeros(self.ndof)
        np.add.at(f, self._ed.reshape(-1), g.reshape(-1))
        if not tangent:
            return f, None
        if self.linear and self._Klin is not None:
            return f, self._Klin
        h = np.asarray(self._h(U, Xe, self.E, self.nu)).reshape(-1)
        K = sp.coo_matrix((h, (self._r, self._c)), shape=(self.ndof, self.ndof)).tocsr()
        if self.linear:
            self._Klin = K
        return f, K

    def tie(self, dofs):
        dofs = np.asarray(dofs)
        self.rep[dofs] = dofs[0]
        return dofs[0]

    def _reduction(self):
        fixed = np.array(sorted(self.fixed), int)
        isfix = np.zeros(self.ndof, bool); isfix[fixed] = True
        indep = np.where((self.rep == np.arange(self.ndof)) & ~isfix)[0]
        col = -np.ones(self.ndof, int); col[indep] = np.arange(len(indep))
        rows = np.where(col[self.rep] >= 0)[0]
        T = sp.coo_matrix((np.ones(len(rows)), (rows, col[self.rep[rows]])), shape=(self.ndof, len(indep))).tocsr()
        return T, fixed

    def residual(self, u, tangent=True):
        f, K = self.internal(u, tangent)
        x = self.X + u.reshape(-1, 2)
        self._fscale = max(np.linalg.norm(f), np.linalg.norm(self.fext))
        for cp in self.pairs:
            fc, Kc = cp.evaluate(x, tangent)
            self._fscale = max(self._fscale, np.linalg.norm(fc))
            f = f - fc
            if tangent:
                K = K - Kc
        return f - self.fext, K

    def solve_step(self, fixed_values=None, fext=None, newton_tol=1e-8, max_newton=80, max_aug=40,
                   pen_tol=None, slip_tol=None, verbose=False, line_search=False):
        """One load step.  Returns (converged, info)."""
        if fixed_values is not None:
            self.fixed.update(fixed_values)
        if fext is not None:
            self.fext = fext
        T, fixed = self._reduction()
        u = self.u.copy()
        # prescribed-displacement increment, applied through the linearisation in the first
        # Newton iteration (writing it straight into u inverts the elements next to the
        # boundary when the increment exceeds their size)
        target = u.copy()
        target[fixed] = np.array([self.fixed[d] for d in fixed])
        target = np.where(self.rep != np.arange(self.ndof), target[self.rep], target)
        isdir = np.zeros(self.ndof, bool); isdir[fixed] = True
        isdir = isdir | isdir[self.rep]
        dUf = np.where(isdir, target - u, 0.0)
        pending = bool(np.any(dUf != 0.0))
        nit_total = 0
        for ia in range(max_aug):
            rn0 = None
            for it in range(max_newton):
                R, K = self.residual(u)
                Rr = T.T @ R
                fscale = max(self._fscale, 1e-30)   # largest of |f_int|, |f_ext|, |f_contact|
                ra = np.linalg.norm(Rr)
                rn = ra / fscale
                if verbose:
                    print(f"    aug {ia} it {it} |R|/f = {rn:.3e}")
                if not np.isfinite(rn):
                    return False, dict(reason="nan")
                if not pending and ((rn < newton_tol and it > 0) or rn < newton_tol * 1e-3):
                    break
                Kr = (T.T @ K @ T).tocsr()
                if pending:
                    du = T @ _solve(Kr, -(Rr + T.T @ (K @ dUf))) + dUf
                    pending = False
                else:
                    du = T @ _solve(Kr, -Rr)
                if not np.all(np.isfinite(du)):
                    return False, dict(reason="singular")
                step = 1.0
                if line_search and it > 0:
                    for _ in range(5):
                        Rt, _ = self.residual(u + step * du, tangent=False)
                        if np.linalg.norm(T.T @ Rt) < (1 - 1e-4 * step) * ra:
                            break
                        step *= 0.5
                u = u + step * du
                nit_total += 1
            else:
                return False, dict(reason="newton", rn=rn, aug=ia)
            if not self.pairs:
                break
            done = True
            for cp in self.pairs:
                pen, slip = cp.augment()
                if verbose:
                    print(f"  aug {ia}: pen {pen:.3e} stick-slip {slip:.3e}")
                if (pen_tol is not None and pen > pen_tol) or (slip_tol is not None and slip > slip_tol):
                    done = False
            if pen_tol is None and slip_tol is None:
                break   # pure penalty
            if done:
                break
        else:
            return False, dict(reason="augmentation")
        self.u = u
        R, _ = self.residual(u, tangent=False)
        for cp in self.pairs:
            cp.commit()
        self.reaction = R
        return True, dict(newton=nit_total, aug=ia + 1)

    def snapshot(self):
        st = [(cp.lamN.copy(), cp.lamT.copy(), cp.seg0.copy(), cp.xi0.copy()) for cp in self.pairs]
        return self.u.copy(), dict(self.fixed), self.fext.copy(), st

    def restore(self, snap):
        self.u, self.fixed, self.fext = snap[0].copy(), dict(snap[1]), snap[2].copy()
        for cp, s in zip(self.pairs, snap[3]):
            cp.lamN, cp.lamT, cp.seg0, cp.xi0 = [a.copy() for a in s]

    def stress(self, u=None):
        """Cauchy/small-strain stress at element centres (linear, plane strain)."""
        u = self.u if u is None else u
        U = u.reshape(-1, 2)[self.quads]; Xe = self.X[self.quads]
        out = []
        for xi in _GP:
            for eta in _GP:
                N, dN = (np.asarray(a) for a in _shape(xi, eta))
                J = np.einsum("eni,nj->eij", Xe, dN)
                dNdX = np.einsum("nj,ejk->enk", dN, np.linalg.inv(J))
                H = np.einsum("eni,enj->eij", U, dNdX)
                e = 0.5 * (H + H.transpose(0, 2, 1))
                lam = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu)); mu = self.E / (2 * (1 + self.nu))
                tr = e[:, 0, 0] + e[:, 1, 1]
                s = 2 * mu[:, None, None] * e + (lam * tr)[:, None, None] * np.eye(2)[None]
                out.append(s)
        return np.stack(out, axis=1)   # (ne, 4, 2, 2)
