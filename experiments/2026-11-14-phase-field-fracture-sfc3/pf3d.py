"""3D phase-field fracture on tensor-product hex8 grids, GPU matrix-free (JAX).

"Hybrid" AT2 formulation (isotropic degradation in the momentum balance -> linear u-problem;
crack driving force = tensile part of the spectral split, history-field irreversibility):

    div[ ((1-d)^2 + k) C : eps(u) ] = 0
    (Gc/l + 2H) d - Gc l lap(d) = 2H,   H = max_t psi_plus(eps),
    psi_plus = lam/2 <tr eps>_+^2 + mu sum_i <eps_i>_+^2

Both linear systems are solved by Jacobi-preconditioned CG on the GPU. Notches are removed elements.
Degradation and H are element-wise constant (centroid strain, mean nodal d).
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import time, json
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

_g = 1.0 / np.sqrt(3.0)
_SGN = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)


def _B_at(xi, h):
    """Strain-displacement matrix (6,24) of a brick with sizes h=(hx,hy,hz) at local point xi."""
    dN = 0.125 * jnp.asarray(_SGN) * jnp.stack([
        (1 + _SGN[:, 1] * xi[1]) * (1 + _SGN[:, 2] * xi[2]),
        (1 + _SGN[:, 0] * xi[0]) * (1 + _SGN[:, 2] * xi[2]),
        (1 + _SGN[:, 0] * xi[0]) * (1 + _SGN[:, 1] * xi[1])], 1)        # dN/dxi (8,3)
    dNx = dN * (2.0 / h)[None, :]
    B = jnp.zeros((6, 24))
    for a in range(8):
        B = B.at[0, 3 * a].set(dNx[a, 0]); B = B.at[1, 3 * a + 1].set(dNx[a, 1]); B = B.at[2, 3 * a + 2].set(dNx[a, 2])
        B = B.at[3, 3 * a].set(dNx[a, 1]); B = B.at[3, 3 * a + 1].set(dNx[a, 0])
        B = B.at[4, 3 * a + 1].set(dNx[a, 2]); B = B.at[4, 3 * a + 2].set(dNx[a, 1])
        B = B.at[5, 3 * a].set(dNx[a, 2]); B = B.at[5, 3 * a + 2].set(dNx[a, 0])
    return B, dNx


def _N_at(xi):
    return 0.125 * jnp.prod(1 + jnp.asarray(_SGN) * xi[None, :], 1)


def _elem_mats(h, lam, mu):
    C = jnp.array([[lam + 2 * mu, lam, lam, 0, 0, 0], [lam, lam + 2 * mu, lam, 0, 0, 0], [lam, lam, lam + 2 * mu, 0, 0, 0],
                   [0, 0, 0, mu, 0, 0], [0, 0, 0, 0, mu, 0], [0, 0, 0, 0, 0, mu]])
    vol = h[0] * h[1] * h[2]
    K = jnp.zeros((24, 24)); M = jnp.zeros((8, 8)); L = jnp.zeros((8, 8))
    for s in _SGN:
        xi = jnp.asarray(s) * _g
        B, dNx = _B_at(xi, h)
        N = _N_at(xi)
        K = K + B.T @ C @ B * vol / 8
        M = M + jnp.outer(N, N) * vol / 8
        L = L + dNx @ dNx.T * vol / 8
    Bc, _ = _B_at(jnp.zeros(3), h)
    return K, M, L, Bc


class PF3D:
    def __init__(self, xs, ys, zs, E, nu, Gc, l, remove_fn=None, shield_fn=None, kres=1e-5, verbose=True):
        self.xs, self.ys, self.zs = map(np.asarray, (xs, ys, zs))
        nx, ny, nz = len(xs), len(ys), len(zs)
        self.shape = (nx, ny, nz)
        lam = E * nu / ((1 + nu) * (1 - 2 * nu)); mu = E / (2 * (1 + nu))
        self.lam, self.mu, self.Gc, self.l, self.kres = lam, mu, Gc, l, kres
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        self.coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
        nid = np.arange(nx * ny * nz).reshape(nx, ny, nz)
        self.nid = nid
        I, J, K = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), np.arange(nz - 1), indexing="ij")
        I, J, K = I.ravel(), J.ravel(), K.ravel()
        conn = np.stack([nid[I, J, K], nid[I + 1, J, K], nid[I + 1, J + 1, K], nid[I, J + 1, K],
                         nid[I, J, K + 1], nid[I + 1, J, K + 1], nid[I + 1, J + 1, K + 1], nid[I, J + 1, K + 1]], 1)
        cen = self.coords[conn].mean(1)
        hs = np.stack([np.diff(xs)[I], np.diff(ys)[J], np.diff(zs)[K]], 1)
        keep = np.ones(len(conn), bool) if remove_fn is None else ~remove_fn(cen[:, 0], cen[:, 1], cen[:, 2])
        self.elem_ijk = np.stack([I, J, K], 1)[keep]
        conn, cen, hs = conn[keep], cen[keep], hs[keep]
        self.conn_np, self.cen, self.hs = conn, cen, hs
        self.ne, self.nn = len(conn), nx * ny * nz
        active = np.zeros(self.nn, bool); active[conn.ravel()] = True
        self.active = active
        # element matrices (unique sizes only, then gather)
        hu, inv = np.unique(np.round(hs, 12), axis=0, return_inverse=True)
        inv = inv.ravel()
        # sort elements by size-type so that element matrices are applied per type (no per-element 24x24 storage)
        order = np.argsort(inv, kind="stable")
        conn, cen, hs, inv = conn[order], cen[order], hs[order], inv[order]
        self.elem_ijk = self.elem_ijk[order]
        self.conn_np, self.cen, self.hs = conn, cen, hs
        bounds = np.searchsorted(inv, np.arange(len(hu) + 1))
        self.groups = [(int(bounds[t]), int(bounds[t + 1])) for t in range(len(hu))]
        self.K0, self.M0, self.L0, self.Bc = jax.vmap(lambda h: _elem_mats(h, lam, mu))(jnp.asarray(hu))
        self.conn = jnp.asarray(conn)
        self.edof = jnp.asarray((3 * conn[:, :, None] + np.arange(3)[None, None, :]).reshape(self.ne, 24))
        self.shield = jnp.asarray(np.zeros(self.ne, bool) if shield_fn is None else shield_fn(cen[:, 0], cen[:, 1], cen[:, 2]))
        self.u = jnp.zeros(3 * self.nn); self.d = jnp.zeros(self.nn); self.H = jnp.zeros(self.ne)
        self.fixed_mask = np.zeros(3 * self.nn, bool)
        self.fixed_mask[np.repeat(~active, 3)] = True
        self._build()
        if verbose:
            print(f"[pf3d] grid {nx}x{ny}x{nz}, elements {self.ne}, dofs {3*int(active.sum())}, "
                  f"unique element sizes {len(hu)}", flush=True)

    def fix(self, nodes, comp):
        self.fixed_mask[3 * np.asarray(nodes) + comp] = True

    def nodes_where(self, fn):
        c = self.coords
        return np.nonzero(fn(c[:, 0], c[:, 1], c[:, 2]) & self.active)[0]

    def _build(self):
        ndof, nn, ne = 3 * self.nn, self.nn, self.ne
        K0, M0, L0, Bc, edof, conn = self.K0, self.M0, self.L0, self.Bc, self.edof, self.conn
        lam, mu, Gc, l, kres = self.lam, self.mu, self.Gc, self.l, self.kres
        groups = self.groups
        def apply(mats, xe):          # per-type element matrices applied to element vectors (ne, n) -> (ne, m)
            return jnp.concatenate([xe[a:b] @ mats[t].T for t, (a, b) in enumerate(groups)], 0)
        def expand(vals):             # per-type row (ntypes, n) -> (ne, n)
            return jnp.concatenate([jnp.broadcast_to(vals[t], (b - a,) + vals[t].shape) for t, (a, b) in enumerate(groups)], 0)
        dK = expand(jnp.diagonal(K0, axis1=1, axis2=2)); dM = expand(jnp.diagonal(M0, axis1=1, axis2=2)); dL = expand(jnp.diagonal(L0, axis1=1, axis2=2))
        Mrow = expand(M0.sum(2))

        def gdeg(d):
            dc = jnp.clip(d[conn].mean(1), 0.0, 1.0)
            return (1 - dc) ** 2 + kres

        def Ku(u, g):
            return jax.ops.segment_sum((g[:, None] * apply(K0, u[edof])).ravel(), edof.ravel(), ndof)

        def pcg(matvec, b, x0, Minv, rtol, maxit):
            r = b - matvec(x0); z = Minv * r; p = z; rz = jnp.vdot(r, z)
            bn = jnp.linalg.norm(b) + 1e-300
            def cond(s):
                x, r, p, rz, k = s
                return (jnp.linalg.norm(r) > rtol * bn) & (k < maxit)
            def body(s):
                x, r, p, rz, k = s
                Ap = matvec(p); a = rz / jnp.vdot(p, Ap)
                x = x + a * p; r = r - a * Ap; z = Minv * r; rz2 = jnp.vdot(r, z)
                p = z + (rz2 / rz) * p
                return (x, r, p, rz2, k + 1)
            x, r, p, rz, k = jax.lax.while_loop(cond, body, (x0, r, p, rz, 0))
            return x, k, jnp.linalg.norm(r) / bn

        @jax.jit
        def solve_u(u0, uD, free, d, rtol, maxit):
            g = gdeg(d)
            diag = jax.ops.segment_sum((g[:, None] * dK).ravel(), edof.ravel(), ndof)
            Minv = jnp.where(free & (diag > 0), 1.0 / jnp.where(diag > 0, diag, 1.0), 0.0)
            mv = lambda x: jnp.where(free, Ku(x, g), 0.0)
            b = jnp.where(free, -Ku(uD, g), 0.0)
            x, k, res = pcg(mv, b, jnp.where(free, u0, 0.0), Minv, rtol, maxit)
            u = x + uD
            return u, Ku(u, g), k, res

        @jax.jit
        def psi_plus(u):
            eps = apply(Bc, u[edof])
            T = jnp.stack([jnp.stack([eps[:, 0], eps[:, 3] / 2, eps[:, 5] / 2], 1),
                           jnp.stack([eps[:, 3] / 2, eps[:, 1], eps[:, 4] / 2], 1),
                           jnp.stack([eps[:, 5] / 2, eps[:, 4] / 2, eps[:, 2]], 1)], 1)
            ev = jnp.linalg.eigvalsh(T)
            tr = ev.sum(1)
            return 0.5 * lam * jnp.maximum(tr, 0) ** 2 + mu * jnp.sum(jnp.maximum(ev, 0) ** 2, 1)

        @jax.jit
        def solve_d(d0, H, rtol, maxit):
            a = Gc / l + 2 * H
            def mv(x):
                xe = x[conn]
                ye = a[:, None] * apply(M0, xe) + Gc * l * apply(L0, xe)
                return jax.ops.segment_sum(ye.ravel(), conn.ravel(), nn)
            diag = jax.ops.segment_sum((a[:, None] * dM + Gc * l * dL).ravel(), conn.ravel(), nn)
            b = jax.ops.segment_sum((2 * H[:, None] * Mrow).ravel(), conn.ravel(), nn)
            Minv = jnp.where(diag > 0, 1.0 / jnp.where(diag > 0, diag, 1.0), 0.0)
            x, k, res = pcg(mv, b, d0, Minv, rtol, maxit)
            return x, k, res

        @jax.jit
        def energies(u, d):
            g = gdeg(d)
            el = 0.5 * jnp.vdot(u, Ku(u, g))
            de = d[conn]
            ga = jnp.sum(de * apply(M0, de)) / (2 * l) + 0.5 * l * jnp.sum(de * apply(L0, de))
            return el, ga

        self._solve_u, self._psi_plus, self._solve_d, self._energies = solve_u, psi_plus, solve_d, energies

    def step(self, uD_np, max_stag=50, stag_tol=2e-3, rtol=1e-7, maxit=40000, irreversible=True):
        """One load step with prescribed dof values uD_np (full vector; only fixed dofs used)."""
        free = jnp.asarray(~self.fixed_mask)
        uD = jnp.asarray(np.where(self.fixed_mask, uD_np, 0.0))
        Hn = self.H
        its_u = 0
        for k in range(max_stag):
            self.u, self.fint, ku, ru = self._solve_u(self.u, uD, free, self.d, rtol, maxit)
            its_u += int(ku)
            pp = jnp.where(self.shield, 0.0, self._psi_plus(self.u))
            Hnew = jnp.maximum(Hn, pp) if irreversible else pp
            dnew, kd, rd = self._solve_d(self.d, Hnew, 1e-8, 5000)
            dd = float(jnp.max(jnp.abs(dnew - self.d)))
            self.d = dnew
            if dd < stag_tol:
                break
        if k > 0:   # make force consistent with final d
            self.u, self.fint, ku, ru = self._solve_u(self.u, uD, free, self.d, rtol, maxit)
            its_u += int(ku)
        self.H = Hnew
        return dict(nstag=k + 1, cg_u=its_u, res_u=float(ru), dd=dd)

    def reaction(self, nodes, comp):
        return float(jnp.sum(self.fint[3 * jnp.asarray(nodes) + comp]))

    def energies(self):
        el, ga = self._energies(self.u, self.d)
        return float(el), float(ga)

    def d_grid(self):
        d = np.array(self.d).reshape(self.shape)
        d[~self.active.reshape(self.shape)] = np.nan
        return d
