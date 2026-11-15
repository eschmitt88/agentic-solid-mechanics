"""Differentiable plane-stress finite-strain elastoplastic FE solver (JAX, float64).

Kinematics / constitutive framework
-----------------------------------
* 6-node triangles (T6), 3 Gauss points, total-Lagrangian.
* Logarithmic-strain framework (Miehe, Apel & Lambrecht 2002 form): E = 1/2 ln C
  (in-plane 2x2 block; the thickness direction is a principal direction and the
  plane-stress condition T33 = 0 is enforced inside the small-strain-format return
  map), additive split E = Ee + Ep, stress T conjugate to E, P = T : dE/dF by
  reverse-mode AD.  For coaxial loading this coincides with multiplicative
  (Simo) plasticity with Hencky elasticity.
* Yield: quadratic (Hill-48 family, plane stress)
      phi^2 = a T11^2 - 2 h T11 T22 + T22^2 + c T12^2
  with uniaxial yield along y (the loading direction) equal to sigma_y(alpha).
  von Mises is a = 1, h = 1/2, c = 3.
* Isotropic hardening
      sigma_y = s0 + Q1 (1 - exp(-b1 a)) + H a + Q2 (1 - exp(-b2 a)).
* Parameter vector p = [s0, Q1, b1, H, Q2, b2, a, h, c]  (MPa, -, ...).
  Elastic constants E, nu are fixed (module-level, set via set_elastic()).

Differentiation
---------------
Direct (forward) sensitivities: at each converged step the linearised residual
gives  K du/dp = -(dR/ds_n ds_n/dp + dR/dp)  and the state tangent is pushed
through the return map with jax.jvp.  The scalar return-map root is differentiated
implicitly (one stop-gradient Newton polish step).
"""
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np
import jax
import jax.numpy as jnp
import scipy.sparse as sp

jax.config.update("jax_enable_x64", True)

try:
    import pypardiso

    def spsolve_factory(K):
        return lambda b: pypardiso.spsolve(K, b)
except Exception:  # pragma: no cover
    from scipy.sparse.linalg import splu

    def spsolve_factory(K):
        lu = splu(K.tocsc())
        return lu.solve

PNAMES = ["s0", "Q1", "b1", "H", "Q2", "b2", "a", "h", "c"]
P_DEFAULT = np.array([300.0, 500.0, 5.0, 0.0, 0.0, 1.0, 1.0, 0.5, 3.0])
ELASTIC = {"E": 193000.0, "nu": 0.29}


def set_elastic(E, nu):
    ELASTIC["E"] = float(E)
    ELASTIC["nu"] = float(nu)


# ----------------------------------------------------------------------------
# 2x2 SPD matrix logarithm, smooth at repeated eigenvalues
# ----------------------------------------------------------------------------
def _sqrtm2(C):
    s = jnp.sqrt(C[0, 0] * C[1, 1] - C[0, 1] * C[1, 0])
    t = jnp.sqrt(C[0, 0] + C[1, 1] + 2.0 * s)
    return (C + s * jnp.eye(2)) / t


def _inv2(A):
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    return jnp.array([[A[1, 1], -A[0, 1]], [-A[1, 0], A[0, 0]]]) / det


def logm2(C):
    A = C
    for _ in range(4):
        A = _sqrtm2(A)
    I = jnp.eye(2)
    Z = (A - I) @ _inv2(A + I)
    Z2 = Z @ Z
    Z4 = Z2 @ Z2
    L = 2.0 * Z @ (I + Z2 / 3.0 + Z4 / 5.0 + Z4 @ Z2 / 7.0 + Z4 @ Z4 / 9.0)
    return 16.0 * L


def log_strain(F):
    """E = 1/2 ln(F^T F) as (E11, E22, E12)."""
    L = 0.5 * logm2(F.T @ F)
    return jnp.array([L[0, 0], L[1, 1], 0.5 * (L[0, 1] + L[1, 0])])


# ----------------------------------------------------------------------------
# constitutive update (small-strain format in log-strain space, plane stress)
# ----------------------------------------------------------------------------
def sigma_y(alpha, p):
    return p[0] + p[1] * (1.0 - jnp.exp(-p[2] * alpha)) + p[3] * alpha + p[4] * (1.0 - jnp.exp(-p[5] * alpha))


def _phi(T, p):
    return jnp.sqrt(p[6] * T[0] ** 2 - 2.0 * p[7] * T[0] * T[1] + T[1] ** 2 + p[8] * T[2] ** 2 + 1e-30)


def _T_of_mu(mu, ee, p):
    E, nu = ELASTIC["E"], ELASTIC["nu"]
    # (S + mu Pm) T = ee,   ee = (e11, e22, e12 tensor)
    m11 = 1.0 / E + mu * p[6]
    m12 = -nu / E - mu * p[7]
    m22 = 1.0 / E + mu
    det = m11 * m22 - m12 * m12
    T11 = (m22 * ee[0] - m12 * ee[1]) / det
    T22 = (-m12 * ee[0] + m11 * ee[1]) / det
    T12 = ee[2] / ((1.0 + nu) / E + 0.5 * mu * p[8])
    return jnp.array([T11, T22, T12])


def stress_update(Ev, Ep, alpha, p):
    """Return (T, Ep_new, alpha_new, g_resid)."""
    ee = Ev - Ep
    T_tr = _T_of_mu(0.0, ee, p)
    phi_tr = _phi(T_tr, p)
    sy0 = sigma_y(alpha, p)
    plastic = phi_tr > sy0

    def g(mu):
        ph = _phi(_T_of_mu(mu, ee, p), p)
        return ph - sigma_y(alpha + mu * ph, p)

    dg = jax.grad(g)

    def body(i, st):
        mu, lo, hi = st
        gv = g(mu)
        lo = jnp.where(gv > 0, mu, lo)
        hi = jnp.where(gv <= 0, mu, hi)
        d = dg(mu)
        mu_n = mu - gv / jnp.where(d < 0, d, -1.0)
        bad = (mu_n <= lo) | (mu_n >= hi) | ~jnp.isfinite(mu_n)
        mid = jnp.where(jnp.isfinite(hi), 0.5 * (lo + hi), 2.0 * mu + 1e-7)
        mu_n = jnp.where(bad, mid, mu_n)
        return (mu_n, lo, hi)

    mu0 = jnp.zeros(())
    mu, _, _ = jax.lax.fori_loop(0, 25, body, (mu0, jnp.zeros(()), jnp.inf * jnp.ones(())))
    mu = jax.lax.stop_gradient(jnp.where(plastic, mu, 0.0))
    # implicit-differentiation polish step
    d = jax.lax.stop_gradient(dg(mu))
    gv = g(mu)
    mu_p = mu - gv / jnp.where(d < 0, d, -1.0)
    mu_f = jnp.where(plastic, mu_p, 0.0)
    T = _T_of_mu(mu_f, ee, p)
    ph = _phi(T, p)
    dgam = mu_f * ph
    # flow direction * dgam = mu * Pm' T   (tensor components)
    dEp = mu_f * jnp.array([p[6] * T[0] - p[7] * T[1], T[1] - p[7] * T[0], 0.5 * p[8] * T[2]])
    resid = jnp.where(plastic, jnp.abs(gv) / sy0, 0.0)
    return T, Ep + dEp, alpha + dgam, resid


def gp_update(F, Ep, alpha, p):
    """First Piola stress (2x2) and updated state at one Gauss point."""
    Ev, vjp = jax.vjp(log_strain, F)
    T, Ep_n, al_n, resid = stress_update(Ev, Ep, alpha, p)
    (P,) = vjp(jnp.array([T[0], T[1], 2.0 * T[2]]))
    return P, Ep_n, al_n, resid, T


# ----------------------------------------------------------------------------
# T6 element
# ----------------------------------------------------------------------------
_GP = np.array([[1 / 6, 1 / 6], [2 / 3, 1 / 6], [1 / 6, 2 / 3]])
_GW = np.array([1 / 6, 1 / 6, 1 / 6])


def _t6_shape(xi, eta):
    z = 1 - xi - eta
    N = np.array([z * (2 * z - 1), xi * (2 * xi - 1), eta * (2 * eta - 1), 4 * z * xi, 4 * xi * eta, 4 * eta * z])
    dN = np.array(
        [
            [-(4 * z - 1), -(4 * z - 1)],
            [4 * xi - 1, 0],
            [0, 4 * eta - 1],
            [4 * (z - xi), -4 * xi],
            [4 * eta, 4 * xi],
            [-4 * eta, 4 * (z - eta)],
        ]
    )
    return N, dN


def t6_shape_np(xi, eta):
    return _t6_shape(xi, eta)


class Mesh:
    """T6 mesh with precomputed gradients.  nodes (nn,2), elems (ne,6) (gmsh order)."""

    def __init__(self, nodes, elems, thickness):
        self.nodes = np.asarray(nodes, float)
        self.elems = np.asarray(elems, np.int64)
        self.t = float(thickness)
        nn, ne = len(self.nodes), len(self.elems)
        self.nn, self.ne = nn, ne
        dNdX = np.zeros((ne, 3, 6, 2))
        wdet = np.zeros((ne, 3))
        Xe = self.nodes[self.elems]  # ne,6,2
        for g in range(3):
            _, dN = _t6_shape(*_GP[g])
            J = np.einsum("eai,aj->eij", Xe, dN)  # dX_i/dxi_j
            det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
            Jinv = np.linalg.inv(J)
            dNdX[:, g] = np.einsum("aj,eji->eai", dN, Jinv)
            wdet[:, g] = det * _GW[g] * self.t
        if (wdet <= 0).any():
            raise ValueError("inverted elements")
        self.dNdX = jnp.asarray(dNdX)
        self.wdet = jnp.asarray(wdet)
        self.gp_xy = np.stack([np.einsum("a,eai->ei", _t6_shape(*_GP[g])[0], Xe) for g in range(3)], 1)
        dof = np.stack([2 * self.elems, 2 * self.elems + 1], -1).reshape(ne, 12)
        self.edof = dof
        self.edof_j = jnp.asarray(dof)
        self.rows = np.repeat(dof, 12, axis=1).ravel()
        self.cols = np.tile(dof, (1, 12)).ravel()
        self.ndof = 2 * nn

    def zero_state(self):
        return (jnp.zeros((self.ne, 3, 3)), jnp.zeros((self.ne, 3)))


def _elem_res(ue, dNdX, wdet, Ep, alpha, p):
    """ue (12,) -> re (12,), new state."""
    u = ue.reshape(6, 2)

    def one(dN, w, Ep_g, al_g):
        F = jnp.eye(2) + u.T @ dN
        P, Ep_n, al_n, resid, T = gp_update(F, Ep_g, al_g, p)
        r = (dN @ P.T) * w  # (6,2): r_ai = dN_aj P_ij
        return r, Ep_n, al_n, resid

    r, Ep_n, al_n, resid = jax.vmap(one)(dNdX, wdet, Ep, alpha)
    return r.sum(0).reshape(12), (Ep_n, al_n, resid)


def _elem_res_only(ue, dNdX, wdet, Ep, alpha, p):
    return _elem_res(ue, dNdX, wdet, Ep, alpha, p)[0]


@jax.jit
def _assemble(u, edof, dNdX, wdet, Ep, alpha, p):
    ue = u[edof]
    re, (Ep_n, al_n, resid) = jax.vmap(_elem_res, in_axes=(0, 0, 0, 0, 0, None))(ue, dNdX, wdet, Ep, alpha, p)
    Ke = jax.vmap(jax.jacfwd(_elem_res_only), in_axes=(0, 0, 0, 0, 0, None))(ue, dNdX, wdet, Ep, alpha, p)
    R = jnp.zeros(u.shape[0]).at[edof.ravel()].add(re.ravel())
    return R, Ke, Ep_n, al_n, resid.max()


@jax.jit
def _resid_state(u, edof, dNdX, wdet, Ep, alpha, p):
    ue = u[edof]
    re, (Ep_n, al_n, resid) = jax.vmap(_elem_res, in_axes=(0, 0, 0, 0, 0, None))(ue, dNdX, wdet, Ep, alpha, p)
    R = jnp.zeros(u.shape[0]).at[edof.ravel()].add(re.ravel())
    return R, Ep_n, al_n


@jax.jit
def _jvp_all(u, edof, dNdX, wdet, Ep, alpha, p, du, dEp, dal, dp):
    """Batched JVP of (R, Ep_new, alpha_new) w.r.t. (u, Ep, alpha, p).
    du (ndof,k), dEp (ne,3,3,k), dal (ne,3,k), dp (np,k)."""

    def f(u_, Ep_, al_, p_):
        return _resid_state(u_, edof, dNdX, wdet, Ep_, al_, p_)

    def one(du_, dEp_, dal_, dp_):
        return jax.jvp(f, (u, Ep, alpha, p), (du_, dEp_, dal_, dp_))[1]

    return jax.vmap(one, in_axes=(-1, -1, -1, -1), out_axes=-1)(du, dEp, dal, dp)


class Lin:
    """Pardiso with one symbolic analysis for a fixed sparsity pattern (free-free block of K)."""

    def __init__(self, n, rows, cols):
        key = rows.astype(np.int64) * n + cols
        uniq, inv = np.unique(key, return_inverse=True)
        self.inv = inv
        self.nnz = len(uniq)
        r = uniq // n
        self.indices = (uniq % n).astype(np.int32)
        self.indptr = np.zeros(n + 1, np.int32)
        np.cumsum(np.bincount(r, minlength=n), out=self.indptr[1:])
        self.n = n
        self.s = pypardiso.PyPardisoSolver()
        self.A = None
        self.analysed = False

    def matrix(self, vals):
        data = np.bincount(self.inv, weights=vals, minlength=self.nnz)
        return sp.csr_matrix((data, self.indices, self.indptr), shape=(self.n, self.n))

    def factor(self, vals):
        self.A = self.matrix(vals)
        self.s.set_iparm(12, 0)
        self.s.set_phase(22 if self.analysed else 12)
        self.s._call_pardiso(self.A, np.zeros((self.n, 1), order="F"))
        self.analysed = True

    def solve(self, b):
        self.s.set_phase(33)
        return self.s._call_pardiso(self.A, np.asfortranarray(b, dtype=np.float64))

    def free(self):
        self.s.free_memory(everything=True)


class Solver:
    """Displacement-driven quasi-static solve with optional direct sensitivities.

    presc: integer dof indices with prescribed displacement.
    """

    def __init__(self, mesh, presc, verbose=False):
        self.m = mesh
        self.presc = np.asarray(presc)
        free = np.ones(mesh.ndof, bool)
        free[self.presc] = False
        self.free = np.where(free)[0]
        self.verbose = verbose
        self.stats = {"newton_its": 0, "cuts": 0, "max_rm_resid": 0.0, "steps": 0}
        # free-free block pattern, and free-prescribed block as a sparse operator pattern
        fmap = -np.ones(mesh.ndof, np.int64); fmap[self.free] = np.arange(len(self.free))
        pmap = -np.ones(mesh.ndof, np.int64); pmap[self.presc] = np.arange(len(self.presc))
        r, c = mesh.rows, mesh.cols
        self.ff = np.where((fmap[r] >= 0) & (fmap[c] >= 0))[0]
        self.fp = np.where((fmap[r] >= 0) & (pmap[c] >= 0))[0]
        self.fp_r, self.fp_c = fmap[r[self.fp]], pmap[c[self.fp]]
        if not hasattr(mesh, "_lin_cache"):
            mesh._lin_cache = {}
        key = hash(self.presc.tobytes())
        if key not in mesh._lin_cache:
            mesh._lin_cache[key] = Lin(len(self.free), fmap[r[self.ff]], fmap[c[self.ff]])
        self.lin = mesh._lin_cache[key]

    def _factor(self, Ke):
        """factorise K_ff and return K_fp as a sparse matrix"""
        v = np.asarray(Ke).ravel()
        self._last_vals = v[self.ff]
        self.lin.factor(self._last_vals)
        return sp.coo_matrix((v[self.fp], (self.fp_r, self.fp_c)), shape=(len(self.free), len(self.presc))).tocsr()

    def _newton(self, u0, ubar, state, p, Kfp=None, tol=1e-8, maxit=25):
        """Kfp not None means self.lin holds the factorised tangent of the last converged state."""
        m = self.m
        u = np.array(u0)
        Ep, al = state
        if Kfp is None:
            _, Ke0, _, _, _ = _assemble(jnp.asarray(u), m.edof_j, m.dNdX, m.wdet, Ep, al, p)
            Kfp = self._factor(Ke0)
        # linearised predictor for the boundary increment
        dub = ubar - u[self.presc]
        u[self.free] += self.lin.solve(-(Kfp @ dub))
        u[self.presc] = ubar
        for it in range(maxit):
            R, Ke, Ep_n, al_n, rmres = _assemble(jnp.asarray(u), m.edof_j, m.dNdX, m.wdet, Ep, al, p)
            R = np.asarray(R)
            rn = np.linalg.norm(R[self.free])
            fscale = np.linalg.norm(R[self.presc]) + 1e-9
            if not np.isfinite(rn):
                return None
            if self.verbose:
                print(f"      it {it} |R_f|={rn:.3e} rel={rn / fscale:.2e}")
            Kfp_new = self._factor(Ke)
            if rn <= tol * fscale and it > 0:
                self.stats["newton_its"] += it
                self.stats["max_rm_resid"] = max(self.stats["max_rm_resid"], float(rmres))
                return u, (Ep_n, al_n), R, Kfp_new
            du = self.lin.solve(-R[self.free])
            if not np.all(np.isfinite(du)):
                return None
            u[self.free] += du
        return None

    def run(self, ubar_fn, n_stations, p, sens_idx=None, observe=None, max_cuts=6):
        """ubar_fn(s) for continuous station coordinate s in [0, n_stations-1].
        sens_idx: list of parameter indices to differentiate (or None).
        observe(k, u, R, du, dR, state): called at each integer station k >= 1.
        """
        m = self.m
        p = jnp.asarray(p, dtype=jnp.float64)
        state = m.zero_state()
        u = np.zeros(m.ndof)
        nk = 0 if sens_idx is None else len(sens_idx)
        if nk:
            du = np.zeros((m.ndof, nk))
            dEp = jnp.zeros((m.ne, 3, 3, nk))
            dal = jnp.zeros((m.ne, 3, nk))
            dp = np.zeros((len(PNAMES), nk))
            for j, i in enumerate(sens_idx):
                dp[i, j] = 1.0
            dp = jnp.asarray(dp)
        s = 0.0
        Kfp_last = None
        for k in range(1, n_stations):
            target = float(k)
            ds = target - s
            cuts = 0
            while s < target - 1e-12:
                ds = min(ds, target - s)
                s_new = s + ds
                ubar = ubar_fn(s_new)
                out = self._newton(u, ubar, state, p, Kfp=Kfp_last)
                if out is None:
                    cuts += 1
                    self.stats["cuts"] += 1
                    if cuts > max_cuts:
                        raise RuntimeError(f"step failed at s={s:.3f}")
                    ds *= 0.5
                    if Kfp_last is not None:  # restore the factorisation of the last converged tangent
                        self.lin.factor(self._conv_vals)
                    continue
                u_new, state_new, R, Kfp_last = out
                self._conv_vals = self._last_vals
                if nk:
                    Ep, al = state
                    dR0, _, _ = _jvp_all(jnp.asarray(u_new), m.edof_j, m.dNdX, m.wdet, Ep, al, p,
                                         jnp.zeros((m.ndof, nk)), dEp, dal, dp)
                    dR0 = np.asarray(dR0)
                    du = np.zeros((m.ndof, nk))
                    du[self.free] = self.lin.solve(-dR0[self.free])
                    dR, dEp_n, dal_n = _jvp_all(jnp.asarray(u_new), m.edof_j, m.dNdX, m.wdet, Ep, al, p,
                                                jnp.asarray(du), dEp, dal, dp)
                    dEp, dal = dEp_n, dal_n
                    dR = np.asarray(dR)
                else:
                    dR = None
                u, state = u_new, state_new
                s = s_new
                self.stats["steps"] += 1
            if observe is not None:
                observe(k, u, R, du if nk else None, dR, state)
        return u, state
