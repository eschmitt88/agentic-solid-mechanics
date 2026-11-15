"""Differentiable finite-strain plane-strain FEM for shape-parameterised curved beams (trial 12).

Built on trial 9's `fem.py` (experiments/2026-11-14-metamaterial-inverse-design); same element, material,
Newton and continuation.  New here:

* the number of shape modes is a solver argument (nk centre-line modes, nm thickness modes);
* a series STACK of ncell cells: cell i's shuttle carries cell i+1's frame, the top shuttle is pushed by the
  total travel D.  Unknowns z = [free nodal dofs of every cell, d_1 .. d_{ncell-1}], d_ncell = D - sum(d_i);
  one total potential energy, so the energy-minimising Newton, the stability check and the implicit
  sensitivities carry over unchanged (ncell = 1 reproduces trial 9 exactly);
* derivatives with respect to the load parameter D and of the energy (for fold-point sensitivities, snap.py).

Half-beam model per cell (symmetric about mid-span): wall end clamped, shuttle end held vertical (u_x = 0)
and pushed down by d (u_y = -d).  Q9 elements on a body-fitted mapped mesh that is a smooth function of the
design parameters.  Material: CalculiX-form compressible neo-Hooke, plane strain.
All forces returned by Solver are for the FULL beam (two halves) per mm depth.
"""
from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

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


# ----------------------------------------------------------------------------
# geometry: parameters -> boundary / mesh.   p = [a_1..a_nk, log t0, b_1..b_nm]
# ----------------------------------------------------------------------------
def centerline(p, s, Lh, nk):
    k = jnp.arange(1, nk + 1)
    a = p[:nk]
    y = jnp.sum(a * (1 - jnp.cos(jnp.pi * k * s[..., None])) / 2, axis=-1)
    dyds = jnp.sum(a * (jnp.pi * k) * jnp.sin(jnp.pi * k * s[..., None]) / 2, axis=-1)
    return y, dyds / Lh


def thickness(p, s, nk):
    b = p[nk + 1:]
    m = jnp.arange(1, b.shape[0] + 1)
    return jnp.exp(p[nk] + jnp.sum(b * jnp.cos(jnp.pi * m * s[..., None]), axis=-1))


def surface_points(p, s, eta, Lh, nk):
    y, dydx = centerline(p, s, Lh, nk)
    t = thickness(p, s, nk)
    th = jnp.arctan(dydx)
    return jnp.stack([s * Lh - eta * t * jnp.sin(th), y + eta * t * jnp.cos(th)], axis=-1)


def make_mesh(nx, ny):
    ni, nj = 2 * nx + 1, 2 * ny + 1
    idx = np.arange(ni * nj).reshape(ni, nj)
    conn = []
    for ex in range(nx):
        for ey in range(ny):
            conn.append(idx[2 * ex: 2 * ex + 3, 2 * ey: 2 * ey + 3].reshape(-1))
    conn = np.array(conn)
    s = np.repeat(np.linspace(0, 1, ni), nj)
    eta = np.tile(np.linspace(-0.5, 0.5, nj), ni)
    nn = ni * nj
    fixed = np.zeros((nn, 2), bool)
    fixed[idx[0, :], :] = True
    fixed[idx[-1, :], :] = True
    free = np.where(~fixed.reshape(-1))[0]
    shut = idx[-1, :]
    edofs = np.stack([2 * conn, 2 * conn + 1], axis=-1).reshape(len(conn), -1)
    return dict(nx=nx, ny=ny, ni=ni, nj=nj, conn=jnp.array(conn), s=jnp.array(s), eta=jnp.array(eta), nn=nn,
                free=jnp.array(free), shut=jnp.array(shut), sd=jnp.array(2 * shut + 1), edofs=jnp.array(edofs), nfree=len(free))


# ----------------------------------------------------------------------------
# element
# ----------------------------------------------------------------------------
_g = np.sqrt(0.6)
_GP = np.array([-_g, 0.0, _g])
_GW = np.array([5 / 9, 8 / 9, 5 / 9])


def _l1d(x):
    return np.array([x * (x - 1) / 2, (1 - x) * (1 + x), x * (x + 1) / 2]), np.array([x - 0.5, -2 * x, x + 0.5])


_dN, _W = [], []
for _a, _wa in zip(_GP, _GW):
    for _b, _wb in zip(_GP, _GW):
        Na, dNa = _l1d(_a)
        Nb, dNb = _l1d(_b)
        _dN.append(np.stack([np.outer(dNa, Nb).reshape(-1), np.outer(Na, dNb).reshape(-1)], axis=-1))
        _W.append(_wa * _wb)
DN = jnp.array(np.array(_dN))
GW = jnp.array(np.array(_W))


def psi(F2):
    J = F2[0, 0] * F2[1, 1] - F2[0, 1] * F2[1, 0]
    I1 = jnp.sum(F2 * F2) + 1.0
    return C10 * (J ** (-2.0 / 3.0) * I1 - 3.0) + (J - 1.0) ** 2 / D1


def elem_energy(ue, Xe):
    u = ue.reshape(9, 2)

    def gp(dN, w):
        Jm = Xe.T @ dN
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
    return u.at[mesh["sd"]].set(-d)


def cell_energy(u, X, mesh):
    return jnp.sum(jax.vmap(elem_energy)(u[mesh["edofs"]], X[mesh["conn"]]))


def cell_gK(u, X, mesh):
    n = 2 * mesh["nn"]
    ed = mesh["edofs"]
    ge = jax.vmap(_elem_grad)(u[ed], X[mesh["conn"]])
    g = jnp.zeros(n).at[ed.reshape(-1)].add(ge.reshape(-1))
    ke = jax.vmap(_elem_hess)(u[ed], X[mesh["conn"]])
    rows = jnp.repeat(ed[:, :, None], 18, axis=2).reshape(-1)
    cols = jnp.repeat(ed[:, None, :], 18, axis=1).reshape(-1)
    K = jnp.zeros((n, n)).at[rows, cols].add(ke.reshape(-1))
    return g, K


class Solver:
    def __init__(self, nx=40, ny=2, Lh=28.0, nk=4, nm=4, ncell=1, legacy=False):
        # legacy=True: trial 9's continuation exactly (no tangent predictor, bisection only on a large jump)
        self.legacy = legacy
        self.mesh = m = make_mesh(nx, ny)
        self.Lh, self.nk, self.nm, self.ncell = Lh, nk, nm, ncell
        self.npc = nk + 1 + nm  # parameters per cell
        nf, nc = m["nfree"], ncell
        self.nz = nc * nf + nc - 1
        free, sd = m["free"], m["sd"]

        def coords(p):  # (ncell, nn, 2)
            P = jnp.reshape(p, (nc, self.npc))
            return jax.vmap(lambda q: surface_points(q, m["s"], m["eta"], Lh, nk))(P)

        def unpack(z, D):
            ufs = z[: nc * nf].reshape(nc, nf)
            dint = z[nc * nf:]
            ds = jnp.concatenate([dint, jnp.reshape(D - jnp.sum(dint), (1,))])
            return ufs, ds

        def energy(z, D, Xs):  # HALF beams
            ufs, ds = unpack(z, D)
            U = jax.vmap(lambda uf, d: full_u(uf, d, m))(ufs, ds)
            return jnp.sum(jax.vmap(lambda u, X: cell_energy(u, X, m))(U, Xs))

        residual = jax.grad(energy, argnums=0)
        reaction = jax.grad(energy, argnums=1)

        def assemble(z, D, Xs):
            ufs, ds = unpack(z, D)
            U = jax.vmap(lambda uf, d: full_u(uf, d, m))(ufs, ds)
            g, K = jax.vmap(lambda u, X: cell_gK(u, X, m))(U, Xs)
            f = -jnp.sum(g[:, sd], axis=1)  # dE_i/dd_i
            c = -jnp.sum(K[:, :, sd], axis=2)  # d g_i / d d_i
            k = -jnp.sum(c[:, sd], axis=1)  # d2E_i/dd_i^2
            Kff = jax.scipy.linalg.block_diag(*[K[i][free][:, free] for i in range(nc)])
            ru = g[:, free].reshape(-1)
            if nc == 1:
                return ru, Kff
            rd = f[:-1] - f[-1]
            Kud = jnp.zeros((nc * nf, nc - 1))
            for j in range(nc - 1):
                Kud = Kud.at[j * nf: (j + 1) * nf, j].set(c[j][free])
                Kud = Kud.at[(nc - 1) * nf:, j].set(-c[-1][free])
            Kdd = jnp.diag(k[:-1]) + k[-1]
            return jnp.concatenate([ru, rd]), jnp.block([[Kff, Kud], [Kud.T, Kdd]])

        self.coords = jax.jit(coords)
        self.unpack = unpack
        self.energy = jax.jit(energy)
        self.residual = jax.jit(residual)
        self.reaction = jax.jit(reaction)
        self.assemble = jax.jit(assemble)
        self.stiffness = jax.jit(lambda z, D, Xs: assemble(z, D, Xs)[1])
        self._chol = jax.jit(jnp.linalg.cholesky)
        self._eigh = jax.jit(jnp.linalg.eigh)

        def newton_dir(K, r):
            L = jnp.linalg.cholesky(K)
            ok = jnp.all(jnp.isfinite(L))
            return -jax.scipy.linalg.cho_solve((L, True), r), ok

        self._newton_dir = jax.jit(newton_dir)

        def sfn_dir(K, r):
            w, V = jnp.linalg.eigh(K)
            lam = jnp.maximum(jnp.abs(w), 1e-7 * jnp.max(jnp.abs(w)))
            return -V @ ((V.T @ r) / lam)

        self._sfn_dir = jax.jit(sfn_dir)

        def tangent(z, D, Xs):  # dz/dD along the equilibrium path
            K = assemble(z, D, Xs)[1]
            c = jax.jvp(lambda dd: residual(z, dd, Xs), (D,), (1.0,))[1]
            return -jnp.linalg.solve(K, c)

        self._tangent = jax.jit(tangent)

        # implicit function: z(p, D) = z* - K^-1 R(z*, p, D) has the exact first derivative at a regular state
        def z_implicit(p, D, zstar, Kinv_r):
            Xs = coords(p)
            return zstar - Kinv_r(residual(zstar, D, Xs))

        def F_implicit(p, D, zstar, K):
            Xs = coords(p)
            z = zstar - jnp.linalg.solve(K, residual(zstar, D, Xs))
            return reaction(z, D, Xs)

        self._dF = jax.jit(jax.value_and_grad(F_implicit, argnums=(0, 1)))
        # energy at an equilibrium: dPi/dp = partial_p Pi (envelope theorem), dPi/dD = F
        self._dPi = jax.jit(jax.value_and_grad(lambda p, z, D: energy(z, D, coords(p))))
        # fold sensitivity pieces: phi . R(z, D, p)  differentiated w.r.t. p and D
        self._dphiR = jax.jit(jax.grad(lambda p, D, z, phi: jnp.dot(phi, residual(z, D, coords(p))), argnums=(0, 1)))

    # -- one equilibrium solve at fixed D (energy-minimising Newton) ----------
    def minimise(self, z, D, X, tol=1e-9, maxit=60):
        info = dict(iters=0, shifted=0, converged=False)
        E = float(self.energy(z, D, X))
        if not np.isfinite(E):
            info["res"] = np.inf
            return z, info
        for it in range(maxit):
            r, K = self.assemble(z, D, X)
            rn = float(jnp.max(jnp.abs(r)))
            if rn < tol:
                break
            du, ok = self._newton_dir(K, r)
            if not bool(ok):
                info["shifted"] += 1
                du = self._sfn_dir(K, r)
            slope = float(r @ du)
            a = 1.0
            ok_ls = False
            for _ in range(14):
                E2 = float(self.energy(z + a * du, D, X))
                if np.isfinite(E2) and E2 <= E + 1e-4 * a * slope + 1e-14 * abs(E):
                    ok_ls = True
                    break
                a *= 0.5
            if not ok_ls:
                r2 = float(jnp.max(jnp.abs(self.residual(z + du, D, X))))
                if r2 < rn:
                    a, E2 = 1.0, float(self.energy(z + du, D, X))
                else:
                    break
            z = z + a * du
            E = E2
            info["iters"] = it + 1
        info["res"] = float(jnp.max(jnp.abs(self.residual(z, D, X))))
        info["converged"] = bool(info["res"] < max(tol, 1e-8))
        return z, info

    def stable(self, z, D, X):
        L = self._chol(self.stiffness(z, D, X))
        return bool(jnp.all(jnp.isfinite(L)))

    def solve_step(self, z, D, X, kick=0.05):
        z, info = self.minimise(z, D, X)
        info["switched"] = False
        if not self.stable(z, D, X):
            w, V = self._eigh(self.stiffness(z, D, X))
            v = V[:, 0] / jnp.max(jnp.abs(V[:, 0]))
            cands = []
            for sgn in (+1.0, -1.0):
                z2, i2 = self.minimise(z + sgn * kick * v, D, X)
                if i2["converged"] and self.stable(z2, D, X):
                    cands.append((float(self.energy(z2, D, X)), sgn, z2, i2))
            if cands:
                cands.sort(key=lambda c: c[0])
                _, sgn, z, i2 = cands[0]
                info.update(i2)
                info["switched"] = True
                info["kick_sign"] = sgn
                info["n_branches"] = len(cands)
            else:
                info["unstable"] = True
        return z, info

    # -- continuation -----------------------------------------------------------
    def _advance(self, z, d0, d1, X, depth, jump_ratio, max_depth, events):
        zp = z
        if not self.legacy:
            # tangent predictor keeps Newton in the basin of the connected branch (skipped where the tangent blows up)
            zd = self._tangent(z, jnp.asarray(d0), X)
            m = float(jnp.max(jnp.abs(zd))) * abs(d1 - d0)
            if np.isfinite(m) and m < 0.5:
                zp = z + zd * (d1 - d0)
        z1, info = self.solve_step(zp, d1, X)
        jump = float(jnp.max(jnp.abs(z1 - z)))
        # an indefinite tangent met during the solve means the iterate left the convex neighbourhood of the branch:
        # the landing branch is then decided by round-off, so the step is bisected (new in trial 12)
        left_basin = (not self.legacy) and (info["shifted"] > 0 or info["switched"])
        if depth < max_depth and ((not info["converged"]) or jump > jump_ratio * abs(d1 - d0) or left_basin):
            mid = 0.5 * (d0 + d1)
            zm, im = self._advance(z, d0, mid, X, depth + 1, jump_ratio, max_depth, events)
            if not im["converged"]:
                return zm, im
            z1, info = self._advance(zm, mid, d1, X, depth + 1, jump_ratio, max_depth, events)
            info = dict(info, refined=max(depth + 1, im.get("refined", 0), info.get("refined", 0)))
        elif depth == max_depth and (jump > jump_ratio * abs(d1 - d0) or left_basin):
            info = dict(info, jump=jump, jump_step=abs(d1 - d0))
            if events is not None and info["converged"]:
                events.append(dict(d0=d0, d1=d1, z0=z, z1=z1, jump=jump))
        return z1, info

    def path(self, p, ds, u0=None, verbose=False, stop_on_fail=True, d_start=None, jump_ratio=4.0, max_depth=5, events=None,
             max_sub=0.3):
        """ds: sequence of total travels.  Returns F (full beams), states z, per-step info.
        `events` (optional list) collects every sub-step that still jumps at the finest bisection level:
        candidates for a snap (fold) of the internal state; see snap.py."""
        X = self.coords(jnp.asarray(p))
        z = jnp.zeros(self.nz) if u0 is None else u0
        Fs, zs, infos = [], [], []
        dprev = 0.0 if u0 is None else d_start
        for d in ds:
            d = float(d)
            n = max(1, int(np.ceil(abs(d - dprev) / max_sub)))
            grid = np.linspace(dprev, d, n + 1)
            acc = {}
            for da, db in zip(grid[:-1], grid[1:]):
                z, info = self._advance(z, float(da), float(db), X, 0, jump_ratio, max_depth, events)
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
                zs += [z] * n_left
                infos += [dict(info, aborted=True)] * n_left
                break
            F = 2 * float(self.reaction(z, d, X))
            Fs.append(F)
            zs.append(z)
            infos.append(info)
            if verbose:
                print(f"d={d:7.3f} F={F:+.6e} it={info['iters']} sh={info['shifted']} sw={info['switched']} res={info['res']:.1e} ref={info.get('refined', 0)}", flush=True)
        return np.array(Fs), zs, infos

    # -- sensitivities ------------------------------------------------------------
    def jacobian(self, p, ds, zs, with_dD=False):
        """dF_k/dp (full beams) by implicit differentiation at each converged state; optionally dF_k/dD."""
        p = jnp.asarray(p)
        X = self.coords(p)
        J, Fs, JD = [], [], []
        for d, z in zip(ds, zs):
            K = self.stiffness(z, float(d), X)
            F, (g, gD) = self._dF(p, float(d), z, K)
            J.append(2 * np.asarray(g))
            JD.append(2 * float(gD))
            Fs.append(2 * float(F))
        if with_dD:
            return np.array(Fs), np.array(J), np.array(JD)
        return np.array(Fs), np.array(J)

    def energy_grad(self, p, z, d):
        """Pi (full beams) at an equilibrium state and its partial derivative w.r.t. p at fixed d."""
        E, g = self._dPi(jnp.asarray(p), z, float(d))
        return 2 * float(E), 2 * np.asarray(g)

    def resolve(self, p, ds, zs, tol=1e-11):
        X = self.coords(jnp.asarray(p))
        out = []
        for d, z in zip(ds, zs):
            z2, _ = self.minimise(z, float(d), X, tol=tol)
            out.append(2 * float(self.reaction(z2, float(d), X)))
        return np.array(out)

    def min_eig(self, z, d, X, k=3):
        w, _ = self._eigh(self.stiffness(z, d, X))
        return np.asarray(w[:k])

    def cell_travels(self, z, D):
        return np.asarray(self.unpack(z, D)[1])


def cosine_params(h, t, nk=4, nm=4):
    p = np.zeros(nk + 1 + nm)
    p[0] = h
    p[nk] = np.log(t)
    return p


def embed(p9, nk, nm, nk0=4, nm0=4):
    """Embed a trial-9 parameter vector (4 + 1 + 4) into a larger (nk + 1 + nm) family."""
    p9 = np.asarray(p9, float)
    p = np.zeros(nk + 1 + nm)
    p[:nk0] = p9[:nk0]
    p[nk] = p9[nk0]
    p[nk + 1: nk + 1 + nm0] = p9[nk0 + 1:]
    return p
