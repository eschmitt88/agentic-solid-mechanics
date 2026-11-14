"""Rung 2 — Biot surface instability of a compressed incompressible
neo-Hookean block (plane strain), detected from the tangent stiffness.

Model.  Reference cell [0, W] x [-H, 0], periodic in x (one cell = one
wavelength of the longest admitted mode), free top surface, frictionless
rigid base.  Incompressible neo-Hookean, mixed displacement-pressure
(Taylor-Hood Q2/Q1) Lagrangian
        L(v, p) = int  mu/2 (F:F - 2) - p (det F - 1)  dA ,
        F = diag(lam, 1/lam) + grad v .
The homogeneous compression F0 = diag(lam, 1/lam), p0 = mu/lam^2 is an exact
equilibrium of the discrete problem for every lam (checked: residual at
round-off), so the fundamental path is known and the question is purely
where the tangent stiffness on that path loses positive definiteness on the
space of admissible (incompressibility-preserving, periodic) perturbations.

Detection.  Hessian of L by JAX autodiff, element by element; smallest
eigenvalue theta_min(lam) of the constrained problem
        [K  B^T] [v]           [I 0] [v]
        [B   0 ] [q]  = theta  [0 0] [q]
tracked as lam decreases from the stable side; the critical stretch is the
first zero crossing (Brent).  No deformed mesh is inspected.

Independent check.  The finite-depth dispersion relation (4x4 determinant)
derived from the incremental equations of Cao & Hutchinson 2012 eq. (2.6)
with their top-surface conditions and a frictionless base.
"""
import json
import pathlib
import time

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg as sla
import scipy.optimize as sopt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

jax.config.update("jax_enable_x64", True)

from plotstyle import C, setup

HERE = pathlib.Path(__file__).parent
RES = HERE / "results"
MU = 1.0

# ---------------------------------------------------------------- analytics
R_BIOT = float(np.real([z for z in np.roots([1, -3, -1, -1]) if abs(z.imag) < 1e-12 and z.real > 1][0]))
LAM_BIOT = R_BIOT ** -0.5          # lam1/lam2 = 1/r, lam2 = 1/lam1  ->  lam1 = r^(-1/2)
EPS_BIOT = 1.0 - LAM_BIOT


def dispersion_det(r, kH):
    """det of the 4x4 boundary-condition matrix for a layer of reference depth
    H (k = reference wavenumber).  Unknowns G_s for s in (1, r, -1, -r),
    U2 = sum G_s exp(k s x2) cos(k x1); from eq. (2.6): F_s = -s G_s / r,
    h_s = s G_s (s^2 - 1)/r^2.  Rows scaled by exp to stay O(1)."""
    S = np.array([1.0, r, -1.0, -r])
    top1 = -(S ** 2 / r + r)                       # U1,2 + r U2,1 = 0
    top2 = -2 * S + S * (S ** 2 - 1) / r ** 2      # 2 r U1,1 + lam1 Q = 0
    e = np.exp(-kH * S - kH * r)                   # common factor exp(-kH r) keeps entries bounded
    bot1 = e                                       # U2 = 0 at x2 = -H
    bot2 = -(S ** 2) / r * e                       # U1,2 = 0 at x2 = -H
    return np.linalg.det(np.array([top1, top2, bot1, bot2]))


def critical_r_finite_depth(kH):
    """Smallest r > 1 (first instability when compressing) with det = 0."""
    rs = np.linspace(1.05, 12.0, 4000)
    d = np.array([dispersion_det(r, kH) for r in rs])
    idx = np.where(d[:-1] * d[1:] < 0)[0]
    return sopt.brentq(dispersion_det, rs[idx[0]], rs[idx[0] + 1], args=(kH,), xtol=1e-14, rtol=1e-14)


# ---------------------------------------------------------------- FE
def shape_q2_1d(x):
    return jnp.array([0.5 * x * (x - 1), 1 - x * x, 0.5 * x * (x + 1)]), jnp.array([x - 0.5, -2 * x, x + 0.5])


def shape_q1_1d(x):
    return jnp.array([0.5 * (1 - x), 0.5 * (1 + x)])


GP, GW = np.polynomial.legendre.leggauss(3)


def elem_lagrangian(z, xe, lam):
    """z = (18 displacement dofs [node a: vx, vy], 4 pressure dofs)."""
    ve, pe = z[:18].reshape(9, 2), z[18:]
    tot = 0.0
    for i in range(3):
        for j in range(3):
            Nx, dNx = shape_q2_1d(GP[i])
            Ny, dNy = shape_q2_1d(GP[j])
            dN = jnp.stack([jnp.outer(Ny, dNx).ravel(), jnp.outer(dNy, Nx).ravel()], axis=1)   # (9,2), node = ix + 3 iy
            Jm = xe.T @ dN                                   # dX/dxi
            dNX = dN @ jnp.linalg.inv(Jm)
            gradv = ve.T @ dNX
            F = jnp.diag(jnp.array([lam, 1.0 / lam])) + gradv
            Np = jnp.outer(shape_q1_1d(GP[j]), shape_q1_1d(GP[i])).ravel()
            p = Np @ pe
            detF = F[0, 0] * F[1, 1] - F[0, 1] * F[1, 0]
            W = 0.5 * MU * (jnp.sum(F * F) - 2.0) - p * (detF - 1.0)
            tot = tot + GW[i] * GW[j] * jnp.linalg.det(Jm) * W
    return tot


elem_hess = jax.jit(jax.vmap(jax.hessian(elem_lagrangian), in_axes=(None, 0, None)))
elem_grad = jax.jit(jax.vmap(jax.grad(elem_lagrangian), in_axes=(None, 0, None)))


class Mesh:
    def __init__(self, nx, ny, W=1.0, H=2.0, grade=3.0):
        self.nx, self.ny, self.W, self.H = nx, ny, W, H
        mx, my = 2 * nx, 2 * ny + 1
        eta = np.linspace(0, 1, my)
        g = np.expm1(grade * eta) / np.expm1(grade) if grade > 0 else eta
        # mid-side nodes at the geometric middle of each element (straight Q2 geometry)
        yv = -H * g
        yv[1::2] = 0.5 * (yv[0:-1:2] + yv[2::2])
        xs = np.arange(mx) * W / mx
        self.X = np.array([[xs[i], yv[j]] for j in range(my) for i in range(mx)])
        nid = lambda i, j: (i % mx) + mx * j
        pid = lambda i, j: ((i // 2) % nx) + nx * (j // 2)
        conn, pconn, xe = [], [], []
        for ej in range(ny):
            for ei in range(nx):
                i0, j0 = 2 * ei, 2 * ej
                # local order ix + 3*iy with iy increasing upward (y increasing) -> j decreasing
                nodes = [nid(i0 + a, j0 + 2 - b) for b in range(3) for a in range(3)]
                conn.append(nodes)
                pconn.append([pid(i0 + 2 * a, j0 + 2 - 2 * b) for b in range(2) for a in range(2)])
                xloc = np.array([[xs[i0] + a * W / mx, yv[j0 + 2 - b]] for b in range(3) for a in range(3)])
                xe.append(xloc)
        self.conn, self.pconn, self.xe = np.array(conn), np.array(pconn), np.array(xe)
        self.nu, self.np_ = 2 * mx * my, nx * (ny + 1)
        bottom = [nid(i, my - 1) for i in range(mx)]
        fixed = [2 * n + 1 for n in bottom] + [2 * bottom[0]]       # v_y = 0 on base; one v_x pinned (rigid shift)
        self.free_u = np.setdiff1d(np.arange(self.nu), fixed)
        self.top = [nid(i, 0) for i in range(mx)]
        edof = np.concatenate([np.stack([2 * self.conn, 2 * self.conn + 1], axis=2).reshape(len(conn), 18),
                               self.nu + self.pconn], axis=1)
        self.edof = edof
        self.h_top = float(yv[0] - yv[2])

    def hessian(self, lam):
        z0 = jnp.concatenate([jnp.zeros(18), jnp.full(4, MU / lam ** 2)])
        He = np.asarray(elem_hess(z0, jnp.asarray(self.xe), lam))
        n = self.nu + self.np_
        I = np.repeat(self.edof, 22, axis=1).ravel()
        J = np.tile(self.edof, (1, 22)).ravel()
        A = sp.coo_matrix((He.ravel(), (I, J)), shape=(n, n)).tocsr()
        keep = np.concatenate([self.free_u, self.nu + np.arange(self.np_)])
        return A[keep][:, keep], len(self.free_u)

    def residual_norm(self, lam):
        z0 = jnp.concatenate([jnp.zeros(18), jnp.full(4, MU / lam ** 2)])
        ge = np.asarray(elem_grad(z0, jnp.asarray(self.xe), lam))
        g = np.zeros(self.nu + self.np_)
        np.add.at(g, self.edof.ravel(), ge.ravel())
        keep = np.concatenate([self.free_u, self.nu + np.arange(self.np_)])
        return float(np.max(np.abs(g[keep])))


def theta_min_sparse(A, nfree, k=4, sigma=-0.02):
    n = A.shape[0]
    M = sp.diags(np.concatenate([np.ones(nfree), np.zeros(n - nfree)])).tocsc()
    vals, vecs = spla.eigsh(A.tocsc(), k=k, M=M, sigma=sigma, which="LM")
    o = np.argsort(vals)
    return vals[o], vecs[:nfree, o]


def theta_dense(A, nfree):
    """Reference implementation: eigenvalues of Z^T K Z, Z = null space of B.
    Also returns the inertia-based count of negative eigenvalues."""
    Ad = A.toarray()
    K, B = Ad[:nfree, :nfree], Ad[nfree:, :nfree]
    Z = sla.null_space(B)
    w = np.linalg.eigvalsh(Z.T @ K @ Z)
    return w


def harmonic_of_mode(mesh, vec):
    """Dominant Fourier harmonic of the surface-normal displacement."""
    full = np.zeros(mesh.nu)
    full[mesh.free_u] = vec
    vy = full[[2 * n + 1 for n in mesh.top]]
    a = np.abs(np.fft.rfft(vy))
    return int(np.argmax(a[1:]) + 1), full


def critical_stretch(mesh, lam_start=0.70, dlam=0.01, verbose=False):
    """March down in stretch from the stable side; first zero of theta_min."""
    f = lambda lam: theta_min_sparse(*mesh.hessian(lam))[0][0]
    lam, th = lam_start, f(lam_start)
    assert th > 0, "start state must be stable"
    trace = [(lam, th)]
    while True:
        lam2 = lam - dlam
        th2 = f(lam2)
        trace.append((lam2, th2))
        if th2 <= 0:
            break
        lam, th = lam2, th2
        if lam < 0.40:
            raise RuntimeError("no instability found")
    lam_c = sopt.brentq(f, lam2, lam, xtol=1e-12, rtol=1e-12)
    vals, vecs = theta_min_sparse(*mesh.hessian(lam_c))
    n_h, full = harmonic_of_mode(mesh, vecs[:, 0])
    return lam_c, trace, n_h, vals, full



# ---------------------------------------------------------------- second discretisation
# Stream-function formulation: v = (psi_,2 , -r psi_,1) with r = lam2/lam1 satisfies the
# linearised incompressibility constraint r v1,1 + v2,2 = 0 *pointwise*.  Bicubic Hermite
# (Bogner-Fox-Schmit) rectangles are C1, so the discretisation is conforming and the
# stability test is a Rayleigh-Ritz one: no pressure field, no weakly enforced constraint.
G4, W4 = np.polynomial.legendre.leggauss(4)


def herm(s, h):
    H = jnp.array([1 - 3 * s ** 2 + 2 * s ** 3, h * (s - 2 * s ** 2 + s ** 3), 3 * s ** 2 - 2 * s ** 3, h * (-s ** 2 + s ** 3)])
    d1 = jnp.array([(-6 * s + 6 * s ** 2) / h, 1 - 4 * s + 3 * s ** 2, (6 * s - 6 * s ** 2) / h, -2 * s + 3 * s ** 2])
    d2 = jnp.array([(-6 + 12 * s) / h ** 2, (-4 + 6 * s) / h, (6 - 12 * s) / h ** 2, (-2 + 6 * s) / h])
    return H, d1, d2


def elem_energy_psi(c, hx, hy, lam):
    """c[4*a + t]: corner a = ax + 2*ay, dof type t = (psi, psi_x, psi_y, psi_xy)."""
    r = 1.0 / lam ** 2
    p0 = MU / lam ** 2
    # coefficient tensor Cij multiplying Hx[i] Hy[j],  i = 2*ax + sx, j = 2*ay + sy, t = sx + 2*sy
    Cm = jnp.zeros((4, 4))
    for ax in range(2):
        for ay in range(2):
            for sx in range(2):
                for sy in range(2):
                    Cm = Cm.at[2 * ax + sx, 2 * ay + sy].set(c[4 * (ax + 2 * ay) + sx + 2 * sy])
    tot = 0.0
    for i in range(4):
        for j in range(4):
            Hx, dHx, ddHx = herm(0.5 * (G4[i] + 1), hx)
            Hy, dHy, ddHy = herm(0.5 * (G4[j] + 1), hy)
            pxx, pxy, pyy = ddHx @ Cm @ Hy, dHx @ Cm @ dHy, Hx @ Cm @ ddHy
            F11, F12, F21, F22 = lam + pxy, pyy, -r * pxx, 1.0 / lam - r * pxy
            Wd = 0.5 * MU * (F11 ** 2 + F12 ** 2 + F21 ** 2 + F22 ** 2 - 2.0) - p0 * (F11 * F22 - F12 * F21 - 1.0)
            tot = tot + 0.25 * W4[i] * W4[j] * hx * hy * Wd
    return tot


psi_hess = jax.jit(jax.vmap(jax.hessian(elem_energy_psi), in_axes=(None, None, 0, None)))
psi_grad = jax.jit(jax.vmap(jax.grad(elem_energy_psi), in_axes=(None, None, 0, None)))


class StreamMesh:
    def __init__(self, nx, ny, W=1.0, H=2.0, grade=3.0):
        self.nx, self.ny, self.W, self.H = nx, ny, W, H
        eta = np.linspace(0, 1, ny + 1)
        g = np.expm1(grade * eta) / np.expm1(grade) if grade > 0 else eta
        self.yv = -H * g                                  # row 0 = free surface
        self.hx = W / nx
        self.hy = self.yv[:-1] - self.yv[1:]              # element row heights (row e between node rows e, e+1)
        nid = lambda i, j: (i % nx) + nx * j
        edof = []
        for e in range(ny):
            for i in range(nx):
                corners = [nid(i, e + 1), nid(i + 1, e + 1), nid(i, e), nid(i + 1, e)]   # (ax, ay), ay=1 is the upper row
                edof.append([4 * n + t for n in corners for t in range(4)])
        self.edof = np.array(edof)
        self.erow = np.repeat(np.arange(ny), nx)
        self.n = 4 * nx * (ny + 1)
        bottom = [nid(i, ny) for i in range(nx)]
        fixed = [4 * n for n in bottom] + [4 * n + 1 for n in bottom] + [4 * bottom[0] + 2]
        self.free = np.setdiff1d(np.arange(self.n), fixed)
        self.band = 8 * nx + 8

    def stiffness(self, lam):
        He_row = np.asarray(psi_hess(jnp.zeros(16), self.hx, jnp.asarray(self.hy), lam))    # one per element row
        He = He_row[self.erow]
        I = np.repeat(self.edof, 16, axis=1).ravel()
        J = np.tile(self.edof, (1, 16)).ravel()
        K = sp.coo_matrix((He.ravel(), (I, J)), shape=(self.n, self.n)).tocsr()
        return K[self.free][:, self.free]

    def residual_norm(self, lam):
        ge = np.asarray(psi_grad(jnp.zeros(16), self.hx, jnp.asarray(self.hy), lam))[self.erow]
        g = np.zeros(self.n)
        np.add.at(g, self.edof.ravel(), ge.ravel())
        return float(np.max(np.abs(g[self.free])))

    def is_pd(self, lam):
        """Rigorous test: banded Cholesky succeeds iff K is positive definite."""
        K = self.stiffness(lam).tocoo()
        m = K.row <= K.col
        b = int(np.max(K.col[m] - K.row[m]))
        ab = np.zeros((b + 1, K.shape[0]))
        np.add.at(ab, (b + K.row[m] - K.col[m], K.col[m]), K.data[m])
        try:
            sla.cholesky_banded(ab, lower=False, check_finite=False)
            return True
        except np.linalg.LinAlgError:
            return False

    def critical(self, lam_hi=0.98, step=0.02, tol=1e-11):
        lam = lam_hi
        assert self.is_pd(lam), "start state must be stable"
        scan = [(lam, True)]
        while True:
            l2 = lam - step
            ok = self.is_pd(l2)
            scan.append((l2, ok))
            if not ok:
                break
            lam = l2
            if lam < 0.3:
                raise RuntimeError("no loss of stability found")
        lo, hi = l2, lam
        while hi - lo > tol:
            mid = 0.5 * (lo + hi)
            if self.is_pd(mid):
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi), scan

    def mode(self, lam):
        K = self.stiffness(lam).tocsc()
        vals, vecs = spla.eigsh(K, k=4, sigma=-1e-6, which="LM")
        o = np.argsort(vals)
        full = np.zeros(self.n)
        full[self.free] = vecs[:, o[0]]
        return vals[o], full

    def surface_harmonic(self, full, lam):
        # v2 = -r psi_,1 on the top row
        px = full[[4 * i + 1 for i in range(self.nx)]]
        a = np.abs(np.fft.rfft(px))
        return int(np.argmax(a[1:]) + 1) if len(a) > 1 else 0


def th_inertia_onset(mesh, lams):
    """Taylor-Hood: number of negative eigenvalues of the constrained stiffness
    (= negative eigenvalues of the saddle matrix minus n_p) on a list of stretches."""
    out = []
    for lam in lams:
        A, nf = mesh.hessian(lam)
        w = np.linalg.eigvalsh(A.toarray())
        out.append((float(lam), int(np.sum(w < 0)) - mesh.np_))
    return out


def main():
    out = {"reference": {"r": R_BIOT, "lam_W": LAM_BIOT, "eps_W": EPS_BIOT,
                         "check_poly": float(R_BIOT ** 3 - 3 * R_BIOT ** 2 - R_BIOT - 1)}}
    kHs = [0.5, 1, 2, 3, np.pi, 2 * np.pi, 4 * np.pi]
    out["finite_depth_analytic"] = [{"kH": float(k), "eps_c": 1 - critical_r_finite_depth(k) ** -0.5} for k in kHs]
    print("Biot:", out["reference"])

    # ===== attempt 1: Taylor-Hood Q2/Q1 mixed elements =====================
    th = {"naive_shift_invert": [], "inertia": []}
    for nx, ny in [(4, 6), (8, 12), (16, 24), (32, 48)]:
        mesh = Mesh(nx, ny)
        try:
            lam_c, trace, n_h, vals, full = critical_stretch(mesh)
            th["naive_shift_invert"].append({"nx": nx, "ny": ny, "eps_c": 1 - lam_c, "harmonic": n_h,
                                             "rel_err_vs_biot": (1 - lam_c) / EPS_BIOT - 1})
            if nx == 16:
                np.save(RES / "r2_th_spurious_mode_16x24.npy", full)
        except AssertionError as e:
            th["naive_shift_invert"].append({"nx": nx, "ny": ny, "eps_c": None, "note": str(e)})
        print("TH naive", th["naive_shift_invert"][-1])
    lams = [0.995, 0.99, 0.98, 0.97, 0.96, 0.95, 0.9, 0.8, 0.7, 0.6, 0.55]
    for nx, ny in [(4, 6), (8, 12), (16, 24)]:
        mesh = Mesh(nx, ny)
        cnt = th_inertia_onset(mesh, lams)
        first = next((1 - l for l, c in cnt if c > 0), None)
        th["inertia"].append({"nx": nx, "ny": ny, "n_negative_by_stretch": cnt, "first_unstable_strain_on_grid": first})
        print("TH inertia", th["inertia"][-1])
    m0 = Mesh(4, 6)
    th["equilibrium_residual_base_state"] = {str(l): m0.residual_norm(l) for l in (0.9, 0.6, 0.5)}
    out["taylor_hood"] = th

    # ===== attempt 2: stream function, C1 Hermite ===========================
    conv = []
    for nx, ny in [(2, 3), (4, 6), (8, 12), (16, 24), (32, 48), (64, 96)]:
        t = time.time()
        sm = StreamMesh(nx, ny)
        lam_c, scan = sm.critical()
        vals, full = sm.mode(lam_c)
        conv.append({"nx": nx, "ny": ny, "n_dof": int(len(sm.free)), "lam_c": lam_c, "eps_c": 1 - lam_c,
                     "rel_err_vs_biot": (1 - lam_c) / EPS_BIOT - 1, "harmonic": sm.surface_harmonic(full, lam_c),
                     "lowest_eigs_at_crit": vals.tolist(), "n_pd_tests": len(scan), "seconds": time.time() - t,
                     "residual_base_state": sm.residual_norm(lam_c)})
        print("stream", conv[-1])
        if nx == 16:
            np.save(RES / "r2_mode_stream_16x24.npy", full)
    out["stream_function"] = {"mesh_convergence": conv}
    e = [abs(c["rel_err_vs_biot"]) for c in conv]
    out["stream_function"]["monotone"] = bool(all(e[i + 1] < e[i] for i in range(len(e) - 1)))
    out["stream_function"]["observed_order_last"] = float(np.log2(e[-2] / e[-1])) if e[-1] > 0 else None
    # depth study at nx = 16 against the analytic finite-depth dispersion relation
    depth = []
    for H, ny, grade in [(0.125, 4, 0.0), (0.25, 6, 0.0), (0.5, 10, 1.0), (1.0, 16, 2.0), (2.0, 24, 3.0)]:
        sm = StreamMesh(16, ny, H=H, grade=grade)
        lam_c, _ = sm.critical()
        vals, full = sm.mode(lam_c)
        n_h = sm.surface_harmonic(full, lam_c)
        # minimum over the harmonics the cell admits (n = 1..8) of the analytic threshold
        an = {n: 1 - critical_r_finite_depth(2 * np.pi * n * H) ** -0.5 for n in range(1, 9)}
        depth.append({"H_over_W": H, "eps_c_fe": 1 - lam_c, "harmonic_fe": n_h,
                      "eps_c_analytic_same_harmonic": an[n_h] if n_h in an else None,
                      "eps_c_analytic_n1": an[1]})
        print("depth", depth[-1])
    out["stream_function"]["depth_study"] = depth
    (RES / "r2_biot.json").write_text(json.dumps(out, indent=1, default=float))
    figures(out)


def figures(out=None):
    out = out or json.loads((RES / "r2_biot.json").read_text())
    plt = setup()
    conv = out["stream_function"]["mesh_convergence"]
    th = out["taylor_hood"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    ax = axes[0]
    nxs = [c["nx"] for c in conv]
    ax.semilogx(nxs, [c["eps_c"] for c in conv], marker="o", color=C["blue"], label="exactly incompressible elements (stream function)")
    tn = [t for t in th["naive_shift_invert"] if t["eps_c"] is not None]
    ax.semilogx([t["nx"] for t in tn], [t["eps_c"] for t in tn], marker="s", ls=":", color=C["orange"],
                label="mixed Q2/Q1 elements, eigen-solver near zero (wrong)")
    ti = [t for t in th["inertia"] if t["first_unstable_strain_on_grid"] is not None]
    ax.semilogx([t["nx"] for t in ti], [t["first_unstable_strain_on_grid"] for t in ti], marker="x", ls=":", color=C["bad"],
                label="mixed Q2/Q1 elements, full inertia count (spurious)")
    ax.axhline(out["reference"]["eps_W"], color=C["ref"], ls="--", lw=1.2, label="Biot threshold 0.4563")
    ax.set_xscale("log", base=2); ax.set_xticks(nxs); ax.set_xticklabels([str(n) for n in nxs])
    ax.set_xlabel("elements per wavelength"); ax.set_ylabel("detected critical compressive strain")
    ax.set_ylim(0, 0.6); ax.set_title("Detected threshold vs mesh", loc="left"); ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.33), fontsize=9)
    ax = axes[1]
    er = np.abs([c["rel_err_vs_biot"] for c in conv])
    ax.loglog(nxs, er, marker="o", color=C["blue"], label="stream-function elements")
    ax.axhline(0.01, color=C["ref"], ls="--", lw=1); ax.text(nxs[0], 0.013, "1 % tolerance", fontsize=9.5)
    ax.set_xscale("log", base=2); ax.set_xticks(nxs); ax.set_xticklabels([str(n) for n in nxs])
    ax.set_xlabel("elements per wavelength"); ax.set_ylabel("relative error vs Biot")
    ax.set_title("Mesh convergence of the threshold", loc="left"); ax.legend(loc="upper right")
    fig.savefig(RES / "r2_biot.png")

    # mode shapes: physical (stream) and spurious (Taylor-Hood)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9))
    sm = StreamMesh(16, 24)
    c16 = [c for c in conv if c["nx"] == 16][0]
    lam_c = c16["lam_c"]
    full = np.load(RES / "r2_mode_stream_16x24.npy").reshape(-1, 4)
    r = 1 / lam_c ** 2
    v1, v2 = full[:, 2], -r * full[:, 1]
    sc = 0.06 / np.max(np.abs(v2))
    ax = axes[0]
    xs = np.arange(sm.nx) * sm.hx
    for rep in range(2):
        X = np.tile(xs, sm.ny + 1) + rep
        Y = np.repeat(sm.yv, sm.nx)
        x = (X * lam_c + sc * v1).reshape(sm.ny + 1, sm.nx); y = (Y / lam_c + sc * v2).reshape(sm.ny + 1, sm.nx)
        for j in range(sm.ny + 1):
            ax.plot(np.append(x[j], x[j, 0] + lam_c), np.append(y[j], y[j, 0]), color=C["blue"] if j == 0 else C["mute"], lw=2 if j == 0 else 0.5)
        for i in range(sm.nx):
            ax.plot(x[:, i], y[:, i], color=C["mute"], lw=0.5)
    ax.set_ylim(-1.3, 0.2); ax.set_aspect("equal"); ax.grid(False)
    ax.set_title(f"Physical wrinkle mode, strain {1 - lam_c:.4f}", loc="left", fontsize=11.5)
    ax = axes[1]
    try:
        mesh = Mesh(16, 24)
        V = np.load(RES / "r2_th_spurious_mode_16x24.npy").reshape(-1, 2)
        t16 = [t for t in th["naive_shift_invert"] if t["nx"] == 16][0]
        lc = 1 - t16["eps_c"]
        V = V / np.max(np.abs(V)) * 0.02
        mx, my = 2 * mesh.nx, 2 * mesh.ny + 1
        x = (mesh.X[:, 0] * lc + V[:, 0]).reshape(my, mx); y = (mesh.X[:, 1] / lc + V[:, 1]).reshape(my, mx)
        for j in range(0, my):
            ax.plot(np.append(x[j], x[j, 0] + lc), np.append(y[j], y[j, 0]), color=C["orange"] if j == 0 else C["mute"], lw=2 if j == 0 else 0.4)
        ax.set_ylim(-0.45, 0.08); ax.set_xlim(0, lc); ax.set_aspect("equal"); ax.grid(False)
        ax.set_title(f"Spurious mesh-scale mode of mixed elements, strain {t16['eps_c']:.3f}", loc="left", fontsize=11.5)
    except Exception as ex:   # figure is illustrative only
        ax.text(0.1, 0.5, f"(not available: {ex})")
    fig.savefig(RES / "r2_biot_mode.png")
    print("figures done")


if __name__ == "__main__":
    import sys
    figures() if "--figures" in sys.argv else main()
