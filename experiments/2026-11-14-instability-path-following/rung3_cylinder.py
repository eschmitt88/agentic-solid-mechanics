"""Rung 3 — imperfection sensitivity of an axially compressed thin cylinder.

Own solver: Donnell (DMV) shell kinematics, Hermite-cubic finite elements
along the axis, Fourier harmonics around the circumference.

  displacement   u = u0(x) + un(x) cos(n th),  v = vn(x) sin(n th),  w = w0(x) + wn(x) cos(n th)
  imperfection   wb(x) = xi t cos(q0 x / R)        (axisymmetric, stress-free)
  strains        ex  = u,x + w,x^2/2 + wb,x w,x
                 eth = v,th/R + w/R + (w,th/R)^2/2
                 gam = u,th/R + v,x + (w,x + wb,x) w,th/R
                 kx = -w,xx   kth = -w,thth/R^2   kxth = -2 w,xth/R

Method.  (1) Follow the nonlinear *axisymmetric* equilibrium path of the
imperfect shell under axial load with the arc-length solver of rung 1.
(2) At every converged point test, for each circumferential harmonic n, whether
the tangent stiffness K_n (Hessian of the energy with respect to the harmonic
amplitudes, by JAX autodiff; exact theta-integration by an 8-point rule per
period) is still positive definite.  (3) The first load at which some K_n
becomes singular is the bifurcation load; located by bisection on the
arc-length step.  No deformed shape is inspected.
"""
import json
import pathlib
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg as sla
import scipy.optimize as sopt

jax.config.update("jax_enable_x64", True)

from pathfollow import Continuation, neg_count
from plotstyle import C, setup

HERE = pathlib.Path(__file__).parent
RES = HERE / "results"

# geometry / material (config.yaml: rung3)
R, T, E, NU = 200.0, 1.0, 70000.0, 0.3
CC = np.sqrt(3 * (1 - NU ** 2))
Q0 = (12 * (1 - NU ** 2)) ** 0.25 * np.sqrt(R / T)
ELL = 2 * np.pi * R / Q0                      # wavelength of the classical axisymmetric mode
SIG_CL = E * T / (R * CC)
N_CL = SIG_CL * T                             # classical axial force per unit circumference
CM = E * T / (1 - NU ** 2)
DB = E * T ** 3 / (12 * (1 - NU ** 2))

GX, WX = np.polynomial.legendre.leggauss(4)
PH = 2 * np.pi * np.arange(8) / 8             # phases n*theta of the theta-quadrature
COS, SIN = jnp.asarray(np.cos(PH)), jnp.asarray(np.sin(PH))


def herm(s, h):
    H = jnp.array([1 - 3 * s ** 2 + 2 * s ** 3, h * (s - 2 * s ** 2 + s ** 3), 3 * s ** 2 - 2 * s ** 3, h * (-s ** 2 + s ** 3)])
    d1 = jnp.array([(-6 * s + 6 * s ** 2) / h, 1 - 4 * s + 3 * s ** 2, (6 * s - 6 * s ** 2) / h, -2 * s + 3 * s ** 2])
    d2 = jnp.array([(-6 + 12 * s) / h ** 2, (-4 + 6 * s) / h, (6 - 12 * s) / h ** 2, (-2 + 6 * s) / h])
    return H, d1, d2


def elem_energy(q0e, qne, h, x0, xi, n):
    """Strain energy of one element per unit circumference (theta-average).
    q0e = [u,u',w,w'] x 2 nodes (axisymmetric);  qne = [u,u',v,v',w,w'] x 2 nodes (harmonic n)."""
    u0c, w0c = q0e.reshape(2, 4)[:, 0:2].ravel(), q0e.reshape(2, 4)[:, 2:4].ravel()
    qn = qne.reshape(2, 6)
    unc, vnc, wnc = qn[:, 0:2].ravel(), qn[:, 2:4].ravel(), qn[:, 4:6].ravel()
    tot = 0.0
    for g in range(4):
        s = 0.5 * (GX[g] + 1)
        H, d1, d2 = herm(s, h)
        x = x0 + s * h
        wbx = -xi * T * (Q0 / R) * jnp.sin(Q0 * x / R)
        u0x, w0, w0x, w0xx = d1 @ u0c, H @ w0c, d1 @ w0c, d2 @ w0c
        un, unx, vn, vnx = H @ unc, d1 @ unc, H @ vnc, d1 @ vnc
        wn, wnx, wnxx = H @ wnc, d1 @ wnc, d2 @ wnc
        # fields at the 8 theta points
        wx = w0x + wnx * COS
        wth = -n * wn * SIN
        ex = u0x + unx * COS + 0.5 * wx ** 2 + wbx * wx
        eth = (n * vn * COS + w0 + wn * COS) / R + 0.5 * (wth / R) ** 2
        gam = -n * un * SIN / R + vnx * SIN + (wx + wbx) * wth / R
        kx = -(w0xx + wnxx * COS)
        kth = n ** 2 * wn * COS / R ** 2
        kxth = 2 * n * wnx * SIN / R
        dens = (0.5 * CM * (ex ** 2 + eth ** 2 + 2 * NU * ex * eth + 0.5 * (1 - NU) * gam ** 2)
                + 0.5 * DB * (kx ** 2 + kth ** 2 + 2 * NU * kx * kth + 0.5 * (1 - NU) * kxth ** 2))
        tot = tot + 0.5 * WX[g] * h * jnp.mean(dens)
    return tot


class Shell:
    def __init__(self, nel, L, xi, bc):
        self.nel, self.L, self.xi, self.bc = nel, L, xi, bc
        self.h = L / nel
        nn = nel + 1
        self.x0 = jnp.asarray(np.arange(nel) * self.h)
        # axisymmetric dofs: 4 per node [u, u', w, w']
        fixed0 = [0]                                   # u(0) = 0
        if bc == "cell":
            fixed0 += [3, 4 * (nn - 1) + 3]            # w' = 0 at both ends (symmetry)
        elif bc == "ss":
            fixed0 += [2, 4 * (nn - 1) + 2]            # w = 0 at both ends
        self.free0 = np.setdiff1d(np.arange(4 * nn), fixed0)
        self.edof0 = np.array([[4 * e + k for k in range(8)] for e in range(nel)])
        # harmonic dofs: 6 per node [u, u', v, v', w, w']
        ends = [0, nn - 1]
        if bc == "cell":
            fixedn = [6 * a + k for a in ends for k in (0, 3, 5)]     # un = 0, vn' = 0, wn' = 0
        else:
            fixedn = [6 * a + k for a in ends for k in (0, 2, 4)]     # un = vn = wn = 0
        self.freen = np.setdiff1d(np.arange(6 * nn), fixedn)
        self.edofn = np.array([[6 * e + k for k in range(12)] for e in range(nel)])
        self.n0, self.nn_ = len(self.free0), nn
        free0j, edof0j = jnp.asarray(self.free0), jnp.asarray(self.edof0)
        h, x0, xi_ = self.h, self.x0, xi

        def energy0(qf):
            q = jnp.zeros(4 * nn).at[free0j].set(qf)
            return jnp.sum(jax.vmap(lambda qe, x: elem_energy(qe, jnp.zeros(12), h, x, xi_, 1.0))(q[edof0j], x0))

        self._f0 = jax.jit(jax.grad(energy0))
        self._K0 = jax.jit(jax.hessian(energy0))
        hess_n = jax.hessian(elem_energy, argnums=1)

        def Kn_elems(qf, n):
            q = jnp.zeros(4 * nn).at[free0j].set(qf)
            return jax.vmap(lambda qe, x: hess_n(qe, jnp.zeros(12), h, x, xi_, n))(q[edof0j], x0)

        self._Kn_e = jax.jit(Kn_elems)
        self._dKn_e = jax.jit(lambda qf, dq, n: jax.jvp(lambda z: Kn_elems(z, n), (qf,), (dq,))[1])
        qv = np.zeros(4 * nn)
        qv[4 * (nn - 1)] = -N_CL                          # compressive end load, lam = N / N_cl
        self.q = qv[self.free0]
        self.iend = int(np.where(self.free0 == 4 * (nn - 1))[0][0])
        self.G = lambda u, lam: np.asarray(self._f0(jnp.asarray(u))) - lam * self.q
        self.K = lambda u, lam: np.asarray(self._K0(jnp.asarray(u)))
        self.Glam = lambda u, lam: -self.q
        II = np.repeat(self.edofn, 12, axis=1).ravel()
        JJ = np.tile(self.edofn, (1, 12)).ravel()
        self._IJ = (II, JJ)

    def _assemble(self, Ke):
        A = np.zeros((6 * self.nn_, 6 * self.nn_))
        np.add.at(A, self._IJ, np.asarray(Ke).ravel())
        return A[np.ix_(self.freen, self.freen)]

    def Kn(self, u, n):
        return self._assemble(self._Kn_e(jnp.asarray(u), float(n)))

    def dKn(self, u, du, n):
        return self._assemble(self._dKn_e(jnp.asarray(u), jnp.asarray(du), float(n)))

    def unstable_harmonics(self, u, ns):
        bad = []
        for n in ns:
            try:
                np.linalg.cholesky(self.Kn(u, n))
            except np.linalg.LinAlgError:
                bad.append(n)
        return bad


def linear_buckling(nel, L, bc, ns):
    """Classical linear eigenvalue analysis of the perfect shell: K_n(0) + lam dK_n/dlam."""
    sh = Shell(nel, L, 0.0, bc)
    u0 = np.zeros(sh.n0)
    du = np.linalg.solve(sh.K(u0, 0.0), sh.q)             # linear pre-buckling state per unit lam
    out = {}
    for n in ns:
        A = sh.Kn(u0, n)
        B = -sh.dKn(u0, du, n)
        mu = sla.eigh(B, A, eigvals_only=True)            # B x = mu A x ; lam = 1/mu_max
        out[n] = float(1.0 / mu[-1]) if mu[-1] > 0 else np.inf
    return out


def bifurcation_load(nel, L, xi, bc, ns, ds=0.02, lam_max=1.3, verbose=False):
    sh = Shell(nel, L, xi, bc)
    mon = lambda u, lam, K: np.array([neg_count(K)[0], len(sh.unstable_harmonics(u, ns))])
    cont = Continuation(sh.G, sh.K, sh.Glam, monitor=mon, psi=1.0, uscale=T * np.sqrt(sh.n0), tol=1e-9)
    p = cont.start(np.zeros(sh.n0), 0.0, fscale=N_CL)
    path, crit = [p], []
    for _ in range(4000):
        path2, crit = cont.run(path[-1], ds, 1, ds_max=ds)
        path.append(path2[-1])
        if crit or path[-1].lam > lam_max:
            break
    res = {"xi": xi, "bc": bc, "nel": nel, "n_path": len(path)}
    lam_path = [float(q.lam) for q in path]
    short = [float(-q.u[sh.iend] / L) for q in path]
    if not crit:
        res.update({"lam_c": None})
        return res, lam_path, short, None
    c = crit[0]
    lo, hi = c.info["lam_bracket"]
    res.update({"kind": c.kind if c.kind != "monitor" else "bifurcation into harmonic n", "lam_c": float(c.lam),
                "lam_bracket": [float(lo), float(hi)], "end_shortening_c": float(-c.u[sh.iend] / L)})
    bad = sh.unstable_harmonics(c.u, ns)
    res["n_crit"] = bad
    mode = None
    if bad:
        w, V = np.linalg.eigh(sh.Kn(c.u, bad[0]))
        full = np.zeros(6 * sh.nn_)
        full[sh.freen] = V[:, 0]
        mode = full.reshape(-1, 6)[:, 4]
        res["lowest_eig_Kn"] = float(w[0])
    # lowest eigenvalue of every K_n at the critical state (how close the competitors are)
    return res, lam_path, short, (np.asarray(c.u), mode, sh)


# ---------------------------------------------------------------- reference formulas
def koiter_P2(sig, xi, g):
    """Hutchinson 2010 eq. (13)-(15), cylinder (beta = 0), uniaxial compression (r = 0)."""
    a = CC * xi * g ** 2 / (1 - sig)
    b1 = 0.5 * (0.25 + g ** 2) ** -2 * (-0.25 + a)
    b2 = -0.5 * (2.25 + g ** 2) ** -2 * a
    return (0.25 + g ** 2) ** 2 + 4 * (b1 * (0.25 + g ** 2)) ** 2 + 4 * (b2 * (2.25 + g ** 2)) ** 2 - 0.5 * sig - a * sig


def koiter_upper_bound(xi):
    """min over gamma of the smallest root sigma in (0,1) of P2 = 0."""
    if xi == 0:
        return 1.0, 0.5
    best = (2.0, None)
    sg = np.linspace(1e-4, 1 - 1e-6, 4000)
    for g in np.linspace(0.05, 2.0, 400):
        v = koiter_P2(sg, xi, g)
        idx = np.where(v[:-1] * v[1:] < 0)[0]
        if len(idx):
            r = sopt.brentq(koiter_P2, sg[idx[0]], sg[idx[0] + 1], args=(xi, g))
            if r < best[0]:
                best = (r, g)
    # polish gamma
    f = lambda g: _root(xi, g)
    o = sopt.minimize_scalar(f, bounds=(max(0.02, best[1] - 0.02), best[1] + 0.02), method="bounded", options={"xatol": 1e-9})
    return (float(o.fun), float(o.x)) if o.fun < best[0] else best


def _root(xi, g):
    sg = np.linspace(1e-4, 1 - 1e-6, 4000)
    v = koiter_P2(sg, xi, g)
    idx = np.where(v[:-1] * v[1:] < 0)[0]
    return sopt.brentq(koiter_P2, sg[idx[0]], sg[idx[0] + 1], args=(xi, g)) if len(idx) else 2.0


def bh_longwave(xi):
    """Budiansky & Hutchinson 1972 eq. (8): k -> 0 limit curve.  gamma = sqrt(12(1-nu^2)) xi."""
    g = np.sqrt(12 * (1 - NU ** 2)) * xi
    f = lambda l: g ** 4 * (1 - 6 * l - 4 * l ** 2) + 4 * g ** 2 * (1 - l) ** 2 * (2 - 6 * l - 5 * l ** 2) + 16 * (1 - l) ** 7 * (1 + l)
    ls = np.linspace(1e-4, 1 - 1e-9, 20000)
    v = f(ls)
    idx = np.where(v[:-1] * v[1:] < 0)[0]
    return float(sopt.brentq(f, ls[idx[-1]], ls[idx[-1] + 1])) if len(idx) else float("nan")


def nasa_sp8007(r_over_t):
    return 1 - 0.901 * (1 - np.exp(-np.sqrt(r_over_t) / 16.0))


def koiter_asymptotic_memory(xi):
    """(1-rho)^2 = (3c/2) xi rho  -- the simple form I remembered; small-xi limit only."""
    a = 1.5 * CC * xi
    return float(((2 + a) - np.sqrt((2 + a) ** 2 - 4)) / 2)


XIS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
NS = list(range(1, 41))


def main():
    out = {"params": {"R": R, "t": T, "E": E, "nu": NU, "q0": Q0, "ell": ELL, "L": 4 * ELL, "sigma_cl": SIG_CL,
                      "N_cl": N_CL, "P_cl": 2 * np.pi * R * N_CL, "nasa_gamma": float(nasa_sp8007(R / T))}}
    L = 4 * ELL
    # sanity of the reference formula: xi -> 0 gives 1 at gamma = 1/2
    out["koiter_check"] = {"xi=1e-6": koiter_upper_bound(1e-6), "xi=1": koiter_upper_bound(1.0)}
    print(out["params"]); print(out["koiter_check"])

    # ---- 3a: classical load by linear eigenvalue analysis, mesh convergence
    lin = []
    for per in (4, 8, 16, 32):
        t0 = time.time()
        lb = linear_buckling(4 * per, L, "cell", NS)
        nmin = min(lb, key=lb.get)
        lbs = linear_buckling(4 * per, L, "ss", NS)
        nmins = min(lbs, key=lbs.get)
        lin.append({"el_per_wavelength": per, "lam_cell": lb[nmin], "n_cell": nmin, "lam_ss": lbs[nmins], "n_ss": nmins,
                    "lam_by_n_cell": lb if per == 16 else None, "seconds": time.time() - t0})
        print("linear", {k: v for k, v in lin[-1].items() if k != "lam_by_n_cell"})
    out["linear_buckling"] = lin

    # ---- 3c: knockdown curve, Koiter cell
    kd = []
    for xi in XIS:
        ref, gk = koiter_upper_bound(xi)
        row = {"xi": xi, "koiter_upper_bound": ref, "koiter_gamma": gk, "koiter_n": gk * Q0, "bh_longwave": bh_longwave(xi),
               "koiter_asymptotic_from_memory": koiter_asymptotic_memory(xi), "by_mesh": []}
        for per in (8, 16, 32):
            t0 = time.time()
            res, lam_path, short, extra = bifurcation_load(4 * per, L, xi, "cell", NS)
            res["seconds"] = time.time() - t0
            row["by_mesh"].append(res)
            if per == 32:
                row["rho"] = res["lam_c"]; row["n_crit"] = res.get("n_crit"); row["kind"] = res.get("kind")
                row["abs_diff"] = None if res["lam_c"] is None else abs(res["lam_c"] - ref)
                if extra is not None and extra[1] is not None:
                    np.savetxt(RES / f"r3_mode_cell_xi{xi}.csv", np.column_stack([np.linspace(0, L, len(extra[1])), extra[1]]),
                               delimiter=",", header="x,wn", comments="")
        print("knockdown cell", {k: v for k, v in row.items() if k != "by_mesh"}, [m["lam_c"] for m in row["by_mesh"]])
        kd.append(row)
    out["knockdown_cell"] = kd

    # ---- longer cells (not graded): do long-wave modes lower the load, as Budiansky & Hutchinson say?
    longc = []
    for m in (8, 16):
        for xi in (0.1, 0.3, 1.0):
            res, *_ = bifurcation_load(8 * m, m * ELL, xi, "cell", NS)
            longc.append({"L_over_ell": m, "xi": xi, "rho": res["lam_c"], "n_crit": res.get("n_crit")})
            print("long cell", longc[-1])
    out["knockdown_long_cell"] = longc

    # ---- realistic simply supported ends (used for the CalculiX comparison); all n and even n only
    ss = []
    for xi in [0.0] + XIS:
        res, lam_path, short, extra = bifurcation_load(64, L, xi, "ss", NS)
        res_even, *_ = bifurcation_load(64, L, xi, "ss", [n for n in NS if n % 2 == 0])
        res32, *_ = bifurcation_load(128, L, xi, "ss", NS)
        ss.append({"xi": xi, "rho": res["lam_c"], "n_crit": res.get("n_crit"), "kind": res.get("kind"),
                   "rho_even_n": res_even["lam_c"], "n_crit_even": res_even.get("n_crit"),
                   "rho_fine_mesh": res32["lam_c"], "end_shortening_c": res.get("end_shortening_c")})
        np.savetxt(RES / f"r3_path_ss_xi{xi}.csv", np.column_stack([short, lam_path]), delimiter=",",
                   header="end_shortening_over_L,N_over_Ncl", comments="")
        print("ss", ss[-1])
    out["knockdown_ss"] = ss
    (RES / "r3_cylinder.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
