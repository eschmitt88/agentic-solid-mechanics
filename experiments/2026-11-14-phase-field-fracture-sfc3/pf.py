"""Phase-field brittle fracture (AT2, Miehe 2010 style) on graded Q4 meshes.

Hand-written: JAX (GPU) for element residuals / tangents by autodiff, PARDISO for
the sparse solves. Plane strain, small strain, unit thickness.

  psi(eps, d) = ((1-d)^2 + k) psi_plus(eps) + psi_minus(eps)
  phase field: (Gc/l + 2H) d - Gc l lap(d) = 2H,   H = max_t psi_plus  (irreversibility)

Energy splits: 'spectral' (Miehe), 'amor' (volumetric-deviatoric), 'none' (isotropic).
"""
import os
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")
import time, json
import numpy as np
import scipy.sparse as sp
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import pypardiso

LAM, MU, GC = 121.15, 80.77, 2.7e-3   # kN/mm^2, kN/mm^2, kN/mm
KRES = 1e-7


# ---------------------------------------------------------------- mesh
def graded_1d(lo, hi, fine_lo, fine_hi, h, hmax, ratio=1.25, must=()):
    """Grid on [lo,hi]: spacing h in [fine_lo,fine_hi], geometric growth outside. `must` points are grid lines."""
    def fill(a, b):  # fine region, exact division
        n = max(1, int(round((b - a) / h)))
        return list(np.linspace(a, b, n + 1))
    pts = sorted(set([fine_lo, fine_hi] + [m for m in must if fine_lo < m < fine_hi]))
    xs = []
    for a, b in zip(pts[:-1], pts[1:]):
        seg = fill(a, b)
        xs += seg if not xs else seg[1:]
    def grow(start, end, sign):
        out, x, s = [], start, h
        L = abs(end - start)
        if L < 1e-12:
            return out
        # geometric sizes capped at hmax, then rescale to fit exactly
        sizes = []
        tot = 0.0
        while tot < L - 1e-12:
            s = min(s * ratio, hmax)
            sizes.append(s); tot += s
        sizes = np.array(sizes) * (L / tot)
        for s in sizes:
            x = x + sign * s
            out.append(x)
        out[-1] = end
        return out
    left = grow(fine_lo, lo, -1.0)[::-1]
    right = grow(fine_hi, hi, +1.0)
    return np.array(left + xs + right)


def make_mesh(case, h, hmax=0.04, band=None, l=0.015):
    """Unit square with a slit y=0.5, 0<=x<=0.5 (duplicated nodes). Returns dict."""
    if case == "tension":
        band = band or max(4 * l, 0.03)
        x = graded_1d(0, 1, 0.5 - 2 * l - 0.01, 1.0, h, hmax, must=(0.5,))
        y = graded_1d(0, 1, 0.5 - band, 0.5 + band, h, hmax, must=(0.5,))
    elif case == "shear":
        x = graded_1d(0, 1, 0.5 - 2 * l - 0.01, 1.0, h, hmax, must=(0.5,))
        y = graded_1d(0, 1, 0.0, 0.5 + 2 * l + 0.01, h, hmax, must=(0.5,))
    elif case == "uniform":
        n = int(round(1 / h)); x = np.linspace(0, 1, n + 1); y = x.copy()
    else:
        raise ValueError(case)
    nx, ny = len(x), len(y)
    X, Y = np.meshgrid(x, y, indexing="ij")
    coords = np.stack([X.ravel(), Y.ravel()], 1)
    nid = np.arange(nx * ny).reshape(nx, ny)
    jmid = int(np.argmin(np.abs(y - 0.5))); assert abs(y[jmid] - 0.5) < 1e-12
    itip = int(np.argmin(np.abs(x - 0.5))); assert abs(x[itip] - 0.5) < 1e-12
    # duplicate nodes on the slit (x<0.5, y=0.5) for elements above
    dup = np.arange(itip) + nx * ny
    coords = np.vstack([coords, np.stack([x[:itip], np.full(itip, 0.5)], 1)])
    I, J = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), indexing="ij")
    I, J = I.ravel(), J.ravel()
    n0, n1, n2, n3 = nid[I, J], nid[I + 1, J], nid[I + 1, J + 1], nid[I, J + 1]
    above = J == jmid
    m0 = above & (I < itip); n0 = np.where(m0, dup[np.minimum(I, itip - 1)], n0)
    m1 = above & (I + 1 < itip); n1 = np.where(m1, dup[np.minimum(I + 1, itip - 1)], n1)
    conn = np.stack([n0, n1, n2, n3], 1)
    return dict(coords=coords, conn=conn, x=x, y=y, nx=nx, ny=ny, nid=nid, dup=dup, h=h)


# ---------------------------------------------------------------- element kernels
_g = 1.0 / np.sqrt(3.0)
GP = np.array([[-_g, -_g], [_g, -_g], [_g, _g], [-_g, _g]])


def shape(xi, eta):
    N = 0.25 * jnp.array([(1 - xi) * (1 - eta), (1 + xi) * (1 - eta), (1 + xi) * (1 + eta), (1 - xi) * (1 + eta)])
    dN = 0.25 * jnp.array([[-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)],
                           [-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)]])
    return N, dN


def elem_geom(xe):
    """xe (4,2) -> N (4gp,4), dNdx (4gp,2,4), wdet (4gp)"""
    Ns, Bs, ws = [], [], []
    for xi, eta in GP:
        N, dN = shape(xi, eta)
        Jm = dN @ xe
        det = Jm[0, 0] * Jm[1, 1] - Jm[0, 1] * Jm[1, 0]
        inv = jnp.array([[Jm[1, 1], -Jm[0, 1]], [-Jm[1, 0], Jm[0, 0]]]) / det
        Ns.append(N); Bs.append(inv @ dN); ws.append(det)
    return jnp.stack(Ns), jnp.stack(Bs), jnp.stack(ws)


def psi_split(eps, split, lam=LAM, mu=MU):
    """eps = (exx, eyy, gxy) plane strain. returns psi_plus, psi_minus."""
    exx, eyy, gxy = eps[0], eps[1], eps[2]
    tr = exx + eyy
    if split == "none":
        return 0.5 * lam * tr**2 + mu * (exx**2 + eyy**2 + 0.5 * gxy**2), 0.0 * tr
    if split == "amor":
        K = lam + 2.0 * mu / 3.0
        devdev = exx**2 + eyy**2 + 0.5 * gxy**2 - tr**2 / 3.0
        trp, trn = jnp.maximum(tr, 0.0), jnp.minimum(tr, 0.0)
        return 0.5 * K * trp**2 + mu * devdev, 0.5 * K * trn**2
    m = 0.5 * tr
    r = jnp.sqrt((0.5 * (exx - eyy))**2 + (0.5 * gxy)**2 + 1e-12 * 1e-12)
    e1, e2 = m + r, m - r
    pos = lambda a: jnp.maximum(a, 0.0)
    neg = lambda a: jnp.minimum(a, 0.0)
    pp = 0.5 * lam * pos(tr)**2 + mu * (pos(e1)**2 + pos(e2)**2)
    pn = 0.5 * lam * neg(tr)**2 + mu * (neg(e1)**2 + neg(e2)**2)
    return pp, pn


def make_kernels(split):
    def strains(ue, B):  # ue (8,), B (4gp,2,4)
        u = ue.reshape(4, 2)
        exx = B[:, 0, :] @ u[:, 0]; eyy = B[:, 1, :] @ u[:, 1]
        gxy = B[:, 1, :] @ u[:, 0] + B[:, 0, :] @ u[:, 1]
        return jnp.stack([exx, eyy, gxy], 1)

    def psi_plus_gp(ue, B):
        return jax.vmap(lambda e: psi_split(e, split)[0])(strains(ue, B))

    def energy_u(ue, de, N, B, w):
        eps = strains(ue, B)
        pp, pn = jax.vmap(lambda e: psi_split(e, split))(eps)
        d = N @ de
        return jnp.sum(w * (((1 - d)**2 + KRES) * pp + pn))

    res_u = jax.grad(energy_u)
    tan_u = jax.jacfwd(res_u)

    def d_system(Hgp, N, B, w, l, gc):
        # K (4,4), f (4,)
        K = jnp.einsum("g,ga,gb->ab", w * (gc / l + 2 * Hgp), N, N) + gc * l * jnp.einsum("g,gia,gib->ab", w, B, B)
        f = jnp.einsum("g,ga->a", w * 2 * Hgp, N)
        return K, f

    def surf_energy(de, N, B, w, l):
        d = N @ de
        gd = jnp.einsum("gia,a->gi", B, de)
        return jnp.sum(w * (d**2 / (2 * l) + 0.5 * l * jnp.sum(gd**2, 1)))

    return dict(psi_plus_gp=jax.jit(jax.vmap(psi_plus_gp)),
                energy_u=jax.jit(lambda U, D, N, B, w: jnp.sum(jax.vmap(energy_u)(U, D, N, B, w))),
                res_u=jax.jit(jax.vmap(res_u)), tan_u=jax.jit(jax.vmap(tan_u)),
                d_system=jax.jit(jax.vmap(d_system, in_axes=(0, 0, 0, 0, None, None))),
                surf=jax.jit(lambda D, N, B, w, l: jnp.sum(jax.vmap(surf_energy, in_axes=(0, 0, 0, 0, None))(D, N, B, w, l))))


class CSRMap:
    """Precomputed map from element-matrix entries to CSR slots of the (free x free) matrix."""
    def __init__(self, edof, free_id, n_free):
        ne, nd = edof.shape
        r = free_id[np.repeat(edof, nd, axis=1)].ravel()
        c = free_id[np.tile(edof, (1, nd))].ravel()
        keep = (r >= 0) & (c >= 0)
        key = r[keep].astype(np.int64) * n_free + c[keep]
        uniq, inv = np.unique(key, return_inverse=True)
        self.keep = jnp.asarray(np.nonzero(keep)[0])
        self.inv = jnp.asarray(inv)
        self.nnz = len(uniq)
        rows = (uniq // n_free).astype(np.int32)
        self.indices = (uniq % n_free).astype(np.int32)
        self.indptr = np.zeros(n_free + 1, np.int32)
        np.add.at(self.indptr, rows + 1, 1)
        self.indptr = np.cumsum(self.indptr).astype(np.int32)
        self.n = n_free
        self._asm = jax.jit(lambda vals: jax.ops.segment_sum(vals.ravel()[self.keep], self.inv, self.nnz))

    def matrix(self, vals):
        data = np.asarray(self._asm(vals))
        return sp.csr_matrix((data, self.indices, self.indptr), shape=(self.n, self.n))


class Problem:
    def __init__(self, mesh, case, l, split="spectral", irreversible=True, gc=GC, side_uy_fixed=False):
        self.mesh, self.case, self.l, self.gc, self.split, self.irrev = mesh, case, l, gc, split, irreversible
        coords, conn = mesh["coords"], mesh["conn"]
        self.nn, self.ne = len(coords), len(conn)
        self.kern = make_kernels(split)
        xe = jnp.asarray(coords[conn])
        self.N, self.B, self.w = jax.jit(jax.vmap(elem_geom))(xe)
        self.conn = jnp.asarray(conn)
        edof = np.stack([2 * conn, 2 * conn + 1], 2).reshape(self.ne, 8)
        self.edof = jnp.asarray(edof)
        tol = 1e-10
        top = np.nonzero(coords[:, 1] > 1 - tol)[0]; bot = np.nonzero(coords[:, 1] < tol)[0]
        self.top = top
        fixed = {}
        for n in bot:
            fixed[2 * n] = 0.0; fixed[2 * n + 1] = 0.0
        self.load_dofs = 2 * top + (1 if case == "tension" else 0)
        other = 2 * top + (0 if case == "tension" else 1)
        for dd in other:
            fixed[dd] = 0.0          # top edge: transverse component held (as in Zhang/Zhou/Hirshikesh)
        if side_uy_fixed:
            side = np.nonzero((coords[:, 0] < tol) | (coords[:, 0] > 1 - tol))[0]
            for n in side:
                fixed[2 * n + 1] = 0.0
        for dd in self.load_dofs:
            fixed[dd] = 0.0
        self.fixed = np.array(sorted(fixed))
        ndof = 2 * self.nn
        free_id = -np.ones(ndof, np.int64)
        free = np.setdiff1d(np.arange(ndof), self.fixed)
        free_id[free] = np.arange(len(free))
        self.free = free
        self.umap = CSRMap(edof, free_id, len(free))
        self.dmap = CSRMap(np.asarray(conn), np.arange(self.nn), self.nn)
        self.u = np.zeros(ndof); self.d = np.zeros(self.nn)
        self.H = jnp.zeros((self.ne, 4))
        self._res = jax.jit(lambda u, d: jax.ops.segment_sum(
            self.kern["res_u"](u[self.edof], d[self.conn], self.N, self.B, self.w).ravel(), self.edof.ravel(), ndof))
        self._tan = jax.jit(lambda u, d: self.kern["tan_u"](u[self.edof], d[self.conn], self.N, self.B, self.w))
        self._dsys = jax.jit(lambda H: self._dsys_impl(H))
        self.n_usolve = 0; self.n_dsolve = 0

    def _dsys_impl(self, H):
        K, f = self.kern["d_system"](H, self.N, self.B, self.w, self.l, self.gc)
        F = jax.ops.segment_sum(f.ravel(), self.conn.ravel(), self.nn)
        return K, F

    def solve_u(self, ubar, tol=1e-7, maxit=25):
        u = self.u.copy()
        old = u[self.load_dofs[0]]
        if abs(old) > 0:      # proportional predictor
            u *= ubar / old
        u[self.load_dofs] = ubar
        d = jnp.asarray(self.d)
        fscale = None
        for it in range(maxit):
            R = np.asarray(self._res(jnp.asarray(u), d))
            rf = R[self.free]
            fscale = max(np.linalg.norm(R[self.fixed]), 1e-30)
            rn = np.linalg.norm(rf) / fscale
            if rn < tol:
                break
            K = self.umap.matrix(self._tan(jnp.asarray(u), d))
            du = pypardiso.spsolve(K, -rf)
            self.n_usolve += 1
            u[self.free] += du
        else:
            print(f"   [warn] u-Newton not converged: rel res {rn:.2e}")
        self.u = u
        self.R = R
        return it

    def reaction(self):
        return float(np.sum(self.R[self.load_dofs]))

    def solve_d(self, Hn):
        pp = self.kern["psi_plus_gp"](jnp.asarray(self.u)[self.edof], self.B)
        H = jnp.maximum(Hn, pp) if self.irrev else pp
        K, F = self._dsys(H)
        A = self.dmap.matrix(K)
        dnew = pypardiso.spsolve(A, np.asarray(F))
        self.n_dsolve += 1
        return dnew, H

    def energies(self):
        U = jnp.asarray(self.u)[self.edof]; D = jnp.asarray(self.d)[self.conn]
        el = float(self.kern["energy_u"](U, D, self.N, self.B, self.w))
        ga = float(self.kern["surf"](D, self.N, self.B, self.w, self.l))
        return el, ga


def run(case, l, h, steps, split="spectral", irreversible=True, max_stag=1, stag_tol=1e-4,
        tag="run", outdir="results/runs", snap_at=(), hmax=0.04, stop_frac=0.01, verbose_every=50,
        side_uy_fixed=False, mesh_case=None):
    """steps: array of prescribed displacements (monotone). max_stag=1 -> Miehe single-pass staggered."""
    os.makedirs(outdir, exist_ok=True)
    mesh = make_mesh(mesh_case or case, h, hmax=hmax, l=l)
    P = Problem(mesh, case, l, split=split, irreversible=irreversible, side_uy_fixed=side_uy_fixed)
    print(f"[{tag}] case={case} l={l} h={h} (l/h={l/h:.2f}) elems={P.ne} dofs={2*P.nn} split={split} "
          f"irrev={irreversible} max_stag={max_stag} steps={len(steps)}", flush=True)
    rec = dict(u=[], F=[], el=[], gamma=[], nstag=[], dmax=[])
    snaps = {}
    Hn = P.H
    Fpeak = 0.0
    t0 = time.time()
    snap_at = list(snap_at)
    for n, ub in enumerate(steps):
        nst = 0
        for k in range(max_stag):
            P.solve_u(ub)
            dnew, Hnew = P.solve_d(Hn)
            dd = float(np.max(np.abs(dnew - P.d)))
            P.d = dnew
            nst += 1
            if max_stag > 1 and dd < stag_tol:
                break
        if max_stag > 1:   # report force consistent with final d
            P.solve_u(ub)
        Hn = Hnew
        F = P.reaction()
        el, ga = P.energies()
        rec["u"].append(float(ub)); rec["F"].append(F); rec["el"].append(el); rec["gamma"].append(ga)
        rec["nstag"].append(nst); rec["dmax"].append(float(P.d.max()))
        Fpeak = max(Fpeak, F)
        while snap_at and ub >= snap_at[0] - 1e-15:
            snaps[f"d_at_{snap_at[0]:.6f}"] = P.d.copy(); snap_at.pop(0)
        if n % verbose_every == 0 or n == len(steps) - 1:
            print(f"[{tag}] step {n:5d} u={ub:.6e} F={F:.5f} dmax={P.d.max():.4f} gamma={ga:.4f} nstag={nst} "
                  f"t={time.time()-t0:.0f}s", flush=True)
        if not np.isfinite(F):
            print(f"[{tag}] non-finite force, abort"); break
        if stop_frac and F < stop_frac * Fpeak and ub > 0.5 * steps[-1] * 0 and n > 10 and Fpeak > 0.1:
            print(f"[{tag}] load dropped below {stop_frac:.0%} of peak at u={ub:.6e}; stop", flush=True)
            break
    wall = time.time() - t0
    out = os.path.join(outdir, tag + ".npz")
    np.savez_compressed(out, coords=mesh["coords"], conn=mesh["conn"], d=P.d, uvec=P.u,
                        meta=json.dumps(dict(case=case, l=l, h=h, split=split, irreversible=irreversible,
                                             max_stag=max_stag, stag_tol=stag_tol, ne=P.ne, ndof=2 * P.nn,
                                             wall_s=wall, n_usolve=P.n_usolve, n_dsolve=P.n_dsolve,
                                             side_uy_fixed=side_uy_fixed, nx=mesh["nx"], ny=mesh["ny"])),
                        **{k: np.array(v) for k, v in rec.items()}, **snaps)
    i = int(np.argmax(rec["F"]))
    print(f"[{tag}] DONE peak F={rec['F'][i]:.5f} at u={rec['u'][i]:.6e}; wall {wall:.0f}s; "
          f"u-solves {P.n_usolve}, d-solves {P.n_dsolve}", flush=True)
    return out


def tension_steps(du1=1e-5, u_switch=5e-3, du2=1e-6, u_end=7.0e-3):
    a = np.arange(1, int(round(u_switch / du1)) + 1) * du1
    b = u_switch + np.arange(1, int(round((u_end - u_switch) / du2)) + 1) * du2
    return np.concatenate([a, b])


def shear_steps(du=1e-5, u_end=0.02, du1=None, u_switch=None):
    if du1 is None:
        return np.arange(1, int(round(u_end / du)) + 1) * du
    a = np.arange(1, int(round(u_switch / du1)) + 1) * du1
    b = u_switch + np.arange(1, int(round((u_end - u_switch) / du)) + 1) * du
    return np.concatenate([a, b])


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True); ap.add_argument("--l", type=float, required=True)
    ap.add_argument("--h", type=float, required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--split", default="spectral"); ap.add_argument("--no-irrev", action="store_true")
    ap.add_argument("--max-stag", type=int, default=1); ap.add_argument("--stag-tol", type=float, default=1e-4)
    ap.add_argument("--du1", type=float, default=None); ap.add_argument("--du2", type=float, default=None)
    ap.add_argument("--u-switch", type=float, default=None); ap.add_argument("--u-end", type=float, default=None)
    ap.add_argument("--side-uy-fixed", action="store_true")
    a = ap.parse_args()
    if a.case == "tension":
        steps = tension_steps(a.du1 or 1e-5, a.u_switch or 5e-3, a.du2 or 1e-6, a.u_end or 7.0e-3)
        snaps = (5.0e-3, 5.5e-3, 5.7e-3, 5.9e-3, 6.2e-3)
    else:
        if a.du1:
            steps = shear_steps(a.du2 or 1e-5, a.u_end or 0.02, a.du1, a.u_switch)
        else:
            steps = shear_steps(a.du2 or 1e-5, a.u_end or 0.02)
        snaps = (0.008, 0.010, 0.011, 0.012, 0.013, 0.014, 0.016, 0.018)
    run(a.case, a.l, a.h, steps, split=a.split, irreversible=not a.no_irrev, max_stag=a.max_stag,
        stag_tol=a.stag_tol, tag=a.tag, snap_at=snaps, side_uy_fixed=a.side_uy_fixed)
