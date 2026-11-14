"""Differentiable finite-strain plane-strain FEM for a shape-parameterised curved beam.

Half-beam model (symmetric about mid-span): wall end clamped, shuttle end held
vertical (u_x = 0) and pushed down by d (u_y = -d).  Q9 (biquadratic) elements on
a body-fitted mapped mesh whose node coordinates are a smooth function of the
design parameters, so everything is differentiable with respect to shape.

Material: CalculiX-form compressible neo-Hooke
    U = C10 (J^-2/3 I1 - 3) + (1/D1) (J - 1)^2 ,   plane strain (F33 = 1).

Forward solve: displacement-controlled continuation; each step is an
energy-minimising Newton (Cholesky, eigenvalue shift when indefinite, Armijo
backtracking) followed by a stability check; an unstable converged state is
kicked along its lowest mode (branch switching).

Sensitivities: the response is path-independent (hyperelastic), so the reaction
F_k = dPi/dd at each converged state is differentiated by the implicit-function
theorem at that state alone: u(p) = u* - K^-1 R(u*, p) with K frozen has the
exact first derivative, and reverse-mode AD through it is the adjoint method.
"""
from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

C10 = 0.5
D1 = 3.0 / 7.0  # K = 2/D1 = 4.6667 MPa -> nu = 0.4 with mu = 1
MU = 2 * C10
KBULK = 2 / D1
E_SMALL = 9 * KBULK * MU / (3 * KBULK + MU)
NU_SMALL = (3 * KBULK - 2 * MU) / (2 * (3 * KBULK + MU))
E_PS = E_SMALL / (1 - NU_SMALL**2)

NK = 4  # centre-line modes
NM = 4  # thickness modes (plus t0)
NP = NK + 1 + NM


# ----------------------------------------------------------------------------
# geometry: parameters -> boundary / mesh
# p = [a_1..a_NK, log t0, b_1..b_NM]
# ----------------------------------------------------------------------------
def centerline(p, s, Lh):
    """s in [0,1] along the half-span (x = s*Lh, full span l = 2*Lh)."""
    k = jnp.arange(1, NK + 1)
    a = p[:NK]
    y = jnp.sum(a * (1 - jnp.cos(jnp.pi * k * s[..., None])) / 2, axis=-1)
    dyds = jnp.sum(a * (jnp.pi * k) * jnp.sin(jnp.pi * k * s[..., None]) / 2, axis=-1)
    return y, dyds / Lh


def thickness(p, s):
    m = jnp.arange(1, NM + 1)
    b = p[NK + 1 :]
    return jnp.exp(p[NK] + jnp.sum(b * jnp.cos(jnp.pi * m * s[..., None]), axis=-1))


def surface_points(p, s, eta, Lh):
    """Point at fraction eta in [-1/2, 1/2] of the thickness, offset along the normal."""
    y, dydx = centerline(p, s, Lh)
    t = thickness(p, s)
    th = jnp.arctan(dydx)
    return jnp.stack([s * Lh - eta * t * jnp.sin(th), y + eta * t * jnp.cos(th)], axis=-1)


def make_mesh(nx, ny):
    """Topology of an nx-by-ny Q9 grid.  Node (i, j): i along span, j through thickness."""
    ni, nj = 2 * nx + 1, 2 * ny + 1
    idx = np.arange(ni * nj).reshape(ni, nj)
    conn = []
    for ex in range(nx):
        for ey in range(ny):
            conn.append(idx[2 * ex : 2 * ex + 3, 2 * ey : 2 * ey + 3].reshape(-1))
    conn = np.array(conn)
    s = np.repeat(np.linspace(0, 1, ni), nj)
    eta = np.tile(np.linspace(-0.5, 0.5, nj), ni)
    nn = ni * nj
    fixed = np.zeros((nn, 2), bool)
    fixed[idx[0, :], :] = True  # wall
    fixed[idx[-1, :], :] = True  # shuttle face
    free = np.where(~fixed.reshape(-1))[0]
    shut = idx[-1, :]
    edofs = np.stack([2 * conn, 2 * conn + 1], axis=-1).reshape(len(conn), -1)
    return dict(nx=nx, ny=ny, ni=ni, nj=nj, conn=jnp.array(conn), s=jnp.array(s), eta=jnp.array(eta),
                nn=nn, free=jnp.array(free), shut=jnp.array(shut), edofs=jnp.array(edofs), nfree=len(free))


def node_coords(p, mesh, Lh):
    return surface_points(p, mesh["s"], mesh["eta"], Lh)


# ----------------------------------------------------------------------------
# element
# ----------------------------------------------------------------------------
_g = np.sqrt(0.6)
_GP = np.array([-_g, 0.0, _g])
_GW = np.array([5 / 9, 8 / 9, 5 / 9])


def _l1d(x):
    return np.array([x * (x - 1) / 2, (1 - x) * (1 + x), x * (x + 1) / 2]), np.array([x - 0.5, -2 * x, x + 0.5])


_dN = []
_W = []
for _a, _wa in zip(_GP, _GW):
    for _b, _wb in zip(_GP, _GW):
        Na, dNa = _l1d(_a)
        Nb, dNb = _l1d(_b)
        # node ordering in conn: index = 3*ia + jb  (ia along span, jb through thickness)
        dxi = np.outer(dNa, Nb).reshape(-1)
        deta = np.outer(Na, dNb).reshape(-1)
        _dN.append(np.stack([dxi, deta], axis=-1))
        _W.append(_wa * _wb)
DN = jnp.array(np.array(_dN))  # (9 gp, 9 nodes, 2)
GW = jnp.array(np.array(_W))


def psi(F2):
    """Strain energy density, plane strain, F2 is the 2x2 in-plane deformation gradient."""
    J = F2[0, 0] * F2[1, 1] - F2[0, 1] * F2[1, 0]
    I1 = jnp.sum(F2 * F2) + 1.0
    return C10 * (J ** (-2.0 / 3.0) * I1 - 3.0) + (J - 1.0) ** 2 / D1


def elem_energy(ue, Xe):
    """ue (18,), Xe (9,2)."""
    u = ue.reshape(9, 2)

    def gp(dN, w):
        Jm = Xe.T @ dN  # dX/dxi
        detJ = Jm[0, 0] * Jm[1, 1] - Jm[0, 1] * Jm[1, 0]
        dNdX = dN @ jnp.linalg.inv(Jm)
        F = jnp.eye(2) + u.T @ dNdX
        return psi(F) * detJ * w

    return jnp.sum(jax.vmap(gp)(DN, GW))


_elem_grad = jax.grad(elem_energy)
_elem_hess = jax.hessian(elem_energy)


def full_u(uf, d, mesh):
    u = jnp.zeros(2 * mesh["nn"])
    u = u.at[mesh["free"]].set(uf)
    u = u.at[2 * mesh["shut"] + 1].set(-d)
    return u


def energy(uf, d, X, mesh):
    u = full_u(uf, d, mesh)
    return jnp.sum(jax.vmap(elem_energy)(u[mesh["edofs"]], X[mesh["conn"]]))


def residual(uf, d, X, mesh):
    u = full_u(uf, d, mesh)
    ge = jax.vmap(_elem_grad)(u[mesh["edofs"]], X[mesh["conn"]])
    g = jnp.zeros(2 * mesh["nn"]).at[mesh["edofs"].reshape(-1)].add(ge.reshape(-1))
    return g[mesh["free"]]


def stiffness(uf, d, X, mesh):
    u = full_u(uf, d, mesh)
    ke = jax.vmap(_elem_hess)(u[mesh["edofs"]], X[mesh["conn"]])
    n = 2 * mesh["nn"]
    ed = mesh["edofs"]
    rows = jnp.repeat(ed[:, :, None], 18, axis=2).reshape(-1)
    cols = jnp.repeat(ed[:, None, :], 18, axis=1).reshape(-1)
    K = jnp.zeros((n, n)).at[rows, cols].add(ke.reshape(-1))
    return K[mesh["free"]][:, mesh["free"]]


def reaction(uf, d, X, mesh):
    """Downward force on the half-beam shuttle face = dPi/dd."""
    return jax.grad(energy, argnums=1)(uf, d, X, mesh)


# ----------------------------------------------------------------------------
# solver
# ----------------------------------------------------------------------------
class Solver:
    def __init__(self, nx=40, ny=2, Lh=28.0):
        self.mesh = make_mesh(nx, ny)
        self.Lh = Lh
        m = self.mesh
        self.energy = jax.jit(lambda uf, d, X: energy(uf, d, X, m))
        self.residual = jax.jit(lambda uf, d, X: residual(uf, d, X, m))
        self.stiffness = jax.jit(lambda uf, d, X: stiffness(uf, d, X, m))
        self.reaction = jax.jit(lambda uf, d, X: reaction(uf, d, X, m))
        self.coords = jax.jit(lambda p: node_coords(p, m, Lh))
        self._chol = jax.jit(jnp.linalg.cholesky)
        self._eigh = jax.jit(jnp.linalg.eigh)

        def newton_dir(K, r):
            L = jnp.linalg.cholesky(K)
            ok = jnp.all(jnp.isfinite(L))
            du = -jax.scipy.linalg.cho_solve((L, True), r)
            return du, ok

        self._newton_dir = jax.jit(newton_dir)

        def sfn_dir(K, r):
            w, V = jnp.linalg.eigh(K)
            lam = jnp.maximum(jnp.abs(w), 1e-7 * jnp.max(jnp.abs(w)))
            return -V @ ((V.T @ r) / lam)

        self._sfn_dir = jax.jit(sfn_dir)

        def F_implicit(p, ustar, d):
            X = node_coords(p, m, Lh)
            K = jax.lax.stop_gradient(stiffness(ustar, d, X, m))
            r = residual(ustar, d, X, m)
            u = ustar - jnp.linalg.solve(K, r)
            return reaction(u, d, X, m)

        self._F_implicit = F_implicit
        self._dFdp = jax.jit(jax.value_and_grad(F_implicit))

    # -- one equilibrium solve at fixed d (energy-minimising Newton) ----------
    def minimise(self, uf, d, X, tol=1e-9, maxit=60):
        """Energy-minimising Newton.  Indefinite tangent -> add a growing diagonal shift (no eigen-solve)."""
        info = dict(iters=0, shifted=0, converged=False)
        E = float(self.energy(uf, d, X))
        if not np.isfinite(E):
            info["res"] = np.inf
            return uf, info
        for it in range(maxit):
            r = self.residual(uf, d, X)
            rn = float(jnp.max(jnp.abs(r)))
            if rn < tol:
                break
            K = self.stiffness(uf, d, X)
            du, ok = self._newton_dir(K, r)
            if not bool(ok):
                # indefinite tangent: saddle-free Newton step (|eigenvalues|), which moves off the saddle along
                # the negative-curvature mode instead of creeping (a plain diagonal shift needed > 40 iterations)
                info["shifted"] += 1
                du = self._sfn_dir(K, r)
            slope = float(r @ du)
            a = 1.0
            ok_ls = False
            for _ in range(14):
                E2 = float(self.energy(uf + a * du, d, X))
                if np.isfinite(E2) and E2 <= E + 1e-4 * a * slope + 1e-14 * abs(E):
                    ok_ls = True
                    break
                a *= 0.5
            if not ok_ls:
                # near convergence the energy decrease drops below round-off; accept the full Newton
                # step if it reduces the residual, otherwise give up
                r2 = float(jnp.max(jnp.abs(self.residual(uf + du, d, X))))
                if r2 < rn:
                    a, E2 = 1.0, float(self.energy(uf + du, d, X))
                else:
                    break
            uf = uf + a * du
            E = E2
            info["iters"] = it + 1
        info["res"] = float(jnp.max(jnp.abs(self.residual(uf, d, X))))
        info["converged"] = bool(info["res"] < max(tol, 1e-8))
        return uf, info

    def stable(self, uf, d, X):
        L = self._chol(self.stiffness(uf, d, X))
        return bool(jnp.all(jnp.isfinite(L)))

    def solve_step(self, uf, d, X, kick=0.05):
        uf, info = self.minimise(uf, d, X)
        info["switched"] = False
        if not self.stable(uf, d, X):
            w, V = self._eigh(self.stiffness(uf, d, X))
            v = V[:, 0] / jnp.max(jnp.abs(V[:, 0]))
            cands = []
            for sgn in (+1.0, -1.0):
                u2, i2 = self.minimise(uf + sgn * kick * v, d, X)
                if i2["converged"] and self.stable(u2, d, X):
                    cands.append((float(self.energy(u2, d, X)), sgn, u2, i2))
            if cands:
                cands.sort(key=lambda c: c[0])
                _, sgn, uf, i2 = cands[0]
                info.update(i2)
                info["switched"] = True
                info["kick_sign"] = sgn
                info["n_branches"] = len(cands)
                if len(cands) == 2:
                    info["dE_branches"] = cands[1][0] - cands[0][0]
            else:
                info["unstable"] = True
        return uf, info

    # -- continuation -----------------------------------------------------------
    def _advance(self, uf, d0, d1, X, depth, jump_ratio, max_depth):
        """Go from a converged state at d0 to d1.  If the solution moved much more than the load step warrants
        (a jump to another branch, or a rapidly growing buckle), bisect the step so the connected path is followed."""
        u1, info = self.solve_step(uf, d1, X)
        jump = float(jnp.max(jnp.abs(u1 - uf)))
        if depth < max_depth and ((not info["converged"]) or jump > jump_ratio * abs(d1 - d0)):
            mid = 0.5 * (d0 + d1)
            um, im = self._advance(uf, d0, mid, X, depth + 1, jump_ratio, max_depth)
            if not im["converged"]:
                return um, im
            u1, info = self._advance(um, mid, d1, X, depth + 1, jump_ratio, max_depth)
            info = dict(info, refined=max(depth + 1, im.get("refined", 0), info.get("refined", 0)))
        elif depth == max_depth and jump > jump_ratio * abs(d1 - d0):
            info = dict(info, jump=jump, jump_step=abs(d1 - d0))
        return u1, info

    def path(self, p, ds, u0=None, verbose=False, stop_on_fail=True, d_start=None, jump_ratio=4.0, max_depth=5):
        """ds: sequence of shuttle displacements. Returns F (full beam = 2 halves), states, per-step info.

        Base sub-steps are <= 0.3 mm; a sub-step is bisected (up to 2^max_depth) when max|du| > jump_ratio*|dd|.
        """
        X = self.coords(jnp.asarray(p))
        uf = jnp.zeros(self.mesh["nfree"]) if u0 is None else u0
        Fs, us, infos = [], [], []
        dprev = 0.0 if u0 is None else d_start
        for d in ds:
            d = float(d)
            if dprev is None:
                uf, info = self.solve_step(uf, d, X)
            else:
                n = max(1, int(np.ceil(abs(d - dprev) / 0.3)))
                grid = np.linspace(dprev, d, n + 1)
                acc = {}
                for da, db in zip(grid[:-1], grid[1:]):
                    uf, info = self._advance(uf, float(da), float(db), X, 0, jump_ratio, max_depth)
                    for k in ("refined", "jump", "jump_step"):
                        if k in info:
                            acc[k] = max(acc.get(k, 0), info[k])
                    if info.get("switched"):
                        acc["switched"] = True
                    if not info["converged"]:
                        break
                info = dict(info, **acc)
            dprev = d
            if not info["converged"] and stop_on_fail:
                n_left = len(ds) - len(Fs)
                Fs += [np.nan] * n_left
                us += [uf] * n_left
                infos += [dict(info, aborted=True)] * n_left
                break
            F = 2 * float(self.reaction(uf, d, X))
            Fs.append(F)
            us.append(uf)
            infos.append(info)
            if verbose:
                print(f"d={d:7.3f} F={F:+.6e} it={info['iters']} sh={info['shifted']} sw={info['switched']} res={info['res']:.1e} ref={info.get('refined', 0)}", flush=True)
        return np.array(Fs), us, infos

    # -- sensitivities ------------------------------------------------------------
    def jacobian(self, p, ds, us):
        """dF_k/dp (full beam) by implicit differentiation at each converged state."""
        p = jnp.asarray(p)
        J = []
        Fs = []
        for d, u in zip(ds, us):
            F, g = self._dFdp(p, u, float(d))
            J.append(2 * np.asarray(g))
            Fs.append(2 * float(F))
        return np.array(Fs), np.array(J)

    def resolve(self, p, ds, us, tol=1e-11):
        """Re-converge at a perturbed design from given states (stays on the same branch); returns F (full beam)."""
        X = self.coords(jnp.asarray(p))
        out = []
        for d, u in zip(ds, us):
            u2, _ = self.minimise(u, float(d), X, tol=tol)
            out.append(2 * float(self.reaction(u2, float(d), X)))
        return np.array(out)

    def min_eig(self, uf, d, X, k=3):
        w, _ = self._eigh(self.stiffness(uf, d, X))
        return np.asarray(w[:k])


def cosine_params(h, t):
    p = np.zeros(NP)
    p[0] = h
    p[NK] = np.log(t)
    return p
