"""Low-fidelity (LF) homogenisation-based topology optimisation (trial 13/14;
dehom-dcbeam copy).

Coarse Q4 grid (hc = 2.5 mm; 60 x 16 for hf.PROBLEM = DCBEAM, 40 x 20 for the
trial-13 GIRDER) of the problem in hf.py: same geometry, same coarse non-design
masks, same supports (springs or fixed dofs) and load (hf.Problem(PROBLEM, HC)
is reused for dof numbering, supports and F). dehom-dcbeam change: fixed dofs
are eliminated exactly in the banded solve (rows/columns zeroed, unit
diagonal, zero load), so u = 0 there. Material per element (paper Eq. 31):

    C~ = T(theta)^T CH(mu1 rho_bar, mu2 rho_bar) T(theta)

with CH from homog.py and T the ENGINEERING-shear strain transformation
(eps_local = T eps_global, Voigt (11, 22, 12), gamma = 2 eps12):

    T = [[c^2, s^2, c s], [s^2, c^2, -c s], [-2 c s, 2 c s, c^2 - s^2]]

This is deHomTop808's T (its Eq. 3). Xu et al.'s printed Eq. 3,
[[c^2, s^2, 2cs], [s^2, c^2, -2cs], [-cs, cs, c^2 - s^2]], is the STRESS
transformation; T^T C T with it is not a stiffness rotation under engineering
shear (see `rotation_check`). Bar 1 runs along theta, bar 2 along theta + pi/2.

Formulation choices that are OURS (from the technical review), not the paper's:
  * theta is a gradient design variable in [-4 pi, 4 pi] (paper Eq. 33 bounds),
    initialised from the principal-stress directions of a solid isotropic solve
    with deHomTop808's seam fix (+pi on non-positive angles); not filtered.
  * mu1, mu2 in [Lmin, 1]; rho in [0, 1] is a void indicator: density filter
    (linear hat, rmin = 2 hc default) -> Heaviside projection (eta = 0.5) with
    beta continuation 1 -> 16 -> rho_bar (three-field scheme). Effective
    widths w_i = mu_i rho_bar. Non-design solid: w = 1; non-design void: w = 0.
  * Objective (deHomTop808 Eq. 10): gamma0 J + gamma_S mean_design(rho_bar),
    gamma0 = V0 / J_solid, gamma_S = 1/20 (non-uniqueness penalty).
  * Volume V = mean over design elements of (w1 + w2 - w1 w2) <= V0.
  * Optimiser: Svanberg MMA (mmasub/subsolv port below) with per-iteration
    move limits [0.2, 0.2, 0.05 rad, 0.2] for (mu1, mu2, theta, rho), as in
    deHomTop808, and asymptote restart at each beta update.
  * Sensitivities: exact adjoint for the (self-adjoint) compliance,
    dJ/dC~_e = -u_e^T K0_ij u_e, chained through the material map by JAX vjp.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
from scipy.linalg import solveh_banded

import hf
import homog

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

NU = 0.3
GAMMA_S = 1.0 / 20.0
ETA = 0.5
MOVE = (0.2, 0.2, 0.05, 0.2)          # mu1, mu2, theta, rho
THETA_BOUND = 4 * np.pi


# ----------------------------------------------------------------------------
# Coarse FE model
# ----------------------------------------------------------------------------
def q4_K0(nu=NU):
    """K0[i, j] (3, 3, 8, 8): element stiffness KE = sum_ij C_ij K0[i, j] for a
    unit square Q4 (2 x 2 Gauss), node order as hf.q4_matrices."""
    xi_n = np.array([-1, 1, 1, -1.0]); eta_n = np.array([-1, -1, 1, 1.0])
    g = 1 / np.sqrt(3); K0 = np.zeros((3, 3, 8, 8))
    for xi in (-g, g):
        for eta in (-g, g):
            dx = 0.25 * xi_n * (1 + eta_n * eta) * 2.0
            dy = 0.25 * eta_n * (1 + xi_n * xi) * 2.0
            B = np.zeros((3, 8))
            B[0, 0::2] = dx; B[1, 1::2] = dy; B[2, 0::2] = dy; B[2, 1::2] = dx
            K0 += 0.25 * np.einsum("ia,jb->ijab", B, B)
    return K0


class Coarse:
    """Coarse grid model: dofs, supports and load from hf.Problem(PROBLEM, HC)."""

    def __init__(self, cfg=None, hc=hf.HC):
        cfg = hf.PROBLEM if cfg is None else cfg
        p = hf.Problem(cfg, hc)
        self.p = p
        self.nelx, self.nely = p.nelx, p.nely
        self.shape = (p.nely, p.nelx)
        self.nel = p.nelx * p.nely
        self.edof = p.edof
        self.ndof = p.ndof
        self.F = p.F.copy()
        self.kdiag = p.kdiag.copy()
        isfix = np.zeros(self.ndof, bool); isfix[p.fixed] = True
        self.fixed = p.fixed
        self.F[isfix] = 0.0
        self.kdiag[isfix] += 1.0                         # unit diagonal on eliminated dofs
        cm = hf.coarse_masks(cfg, hc)
        for k in ("solid_fixed", "void_fixed", "design"):
            assert np.array_equal(cm[k], getattr(p, k)), k
        self.solid = p.solid_fixed.ravel(); self.void = p.void_fixed.ravel(); self.design = p.design.ravel()
        self.di = np.flatnonzero(self.design)
        self.nd = len(self.di)
        self.K0 = q4_K0()
        self.Bc = p.Bc / hc
        # banded (upper) storage index map for solveh_banded
        iK = np.repeat(self.edof, 8, axis=1); jK = np.tile(self.edof, (1, 8))
        up = iK <= jK
        self.u = int((jK - iK)[up].max())
        self._up = up
        self._keep = (~(isfix[iK] | isfix[jK]))[up].astype(float)   # zero fixed rows/cols
        self._flat = ((self.u + iK - jK) * self.ndof + jK)[up]
        self._diag_flat = self.u * self.ndof + np.arange(self.ndof)

    def solve(self, Ce):
        """Ce (nel, 3, 3) -> nodal displacement u (ndof,)."""
        KE = np.einsum("eij,ijab->eab", Ce, self.K0)
        vals = KE.reshape(self.nel, 64)[self._up.reshape(self.nel, 64)] * self._keep
        ab = np.bincount(self._flat, weights=vals, minlength=(self.u + 1) * self.ndof)
        ab[self._diag_flat] += self.kdiag
        return solveh_banded(ab.reshape(self.u + 1, self.ndof), self.F, lower=False, check_finite=False)

    def dJdC(self, u):
        """Compliance sensitivity wrt the element tensors, (nel, 3, 3)."""
        ue = u[self.edof]
        return -np.einsum("ea,ijab,eb->eij", ue, self.K0, ue)

    def element_centre(self, u):
        ux = u[0::2][self.edof[:, 0::2] // 2].mean(1); uy = u[1::2][self.edof[:, 1::2] // 2].mean(1)
        return ux.reshape(self.shape), uy.reshape(self.shape)

    def filter_matrix(self, rmin):
        """Row-normalised linear hat filter over the full grid (element units)."""
        iy, ix = np.divmod(np.arange(self.nel), self.nelx)
        d = np.hypot(ix[:, None] - ix[None], iy[:, None] - iy[None])
        H = np.maximum(0.0, rmin - d)
        return H / H.sum(1, keepdims=True)


_COARSE = None


def coarse():
    global _COARSE
    if _COARSE is None:
        _COARSE = Coarse()
    return _COARSE


# ----------------------------------------------------------------------------
# Material map
# ----------------------------------------------------------------------------
def T_strain(theta):
    """Engineering-shear strain transformation (..., 3, 3), eps_loc = T eps_glob."""
    c = jnp.cos(theta); s = jnp.sin(theta)
    return jnp.stack([jnp.stack([c * c, s * s, c * s], -1),
                      jnp.stack([s * s, c * c, -c * s], -1),
                      jnp.stack([-2 * c * s, 2 * c * s, c * c - s * s], -1)], -2)


def T_paper_eq3(theta):
    """Xu et al. Eq. 3 as printed (stress transformation) -- for the check only."""
    c = jnp.cos(theta); s = jnp.sin(theta)
    return jnp.stack([jnp.stack([c * c, s * s, 2 * c * s], -1),
                      jnp.stack([s * s, c * c, -2 * c * s], -1),
                      jnp.stack([-c * s, c * s, c * c - s * s], -1)], -2)


def rotated_C(w1, w2, theta):
    """Global-frame tensor C~ = T^T CH(w1, w2) T (paper Eq. 31), (..., 3, 3)."""
    T = T_strain(theta)
    return jnp.einsum("...ki,...kl,...lj->...ij", T, homog.CH(w1, w2), T)


def rotation_check(w=0.3, thetas=np.linspace(-np.pi, np.pi, 25)):
    """Single bar (CH(w, 0)) rotated by theta vs the analytic rotated uniaxial
    stiffness E w n(x)n(x)n(x)n, n = (cos, sin), engineering-shear Voigt.
    Returns max abs error for our T and for the paper's printed Eq. 3."""
    c = np.cos(thetas); s = np.sin(thetas)
    A = w * np.stack([np.stack([c ** 4, c * c * s * s, c ** 3 * s], -1),
                      np.stack([c * c * s * s, s ** 4, c * s ** 3], -1),
                      np.stack([c ** 3 * s, c * s ** 3, c * c * s * s], -1)], -2)
    C0 = homog.CH(jnp.full_like(thetas, w), jnp.zeros_like(thetas))
    ours = np.asarray(jnp.einsum("...ki,...kl,...lj->...ij", T_strain(thetas), C0, T_strain(thetas)))
    Tp = T_paper_eq3(thetas)
    paper = np.asarray(jnp.einsum("...ki,...kl,...lj->...ij", Tp, C0, Tp))
    # invariance: rotated solid stays isotropic
    Cs = np.asarray(rotated_C(jnp.ones_like(thetas), jnp.ones_like(thetas), thetas))
    return dict(max_err_ours=float(np.abs(ours - A).max()), max_err_paper_eq3=float(np.abs(paper - A).max()),
                scale=w, iso_invariance_err=float(np.abs(Cs - homog.c_iso()).max()))


def heaviside(x, beta, eta=ETA):
    return (jnp.tanh(beta * eta) + jnp.tanh(beta * (x - eta))) / (jnp.tanh(beta * eta) + jnp.tanh(beta * (1 - eta)))


def make_fields(cm: Coarse, rmin, rmin_mu=None):
    """Returns jitted (fields, grads): x = [mu1, mu2, theta, rho] (4, nd).
    rmin_mu (optional, off by default = our brief): linear-hat filter of mu1, mu2
    over design elements (deHomTop808 filters its widths); keeps [Lmin, 1]."""
    Hf = jnp.asarray(cm.filter_matrix(rmin))
    if rmin_mu:
        Hm = cm.filter_matrix(rmin_mu)[np.ix_(cm.di, cm.di)]
        Hm = jnp.asarray(Hm / Hm.sum(1, keepdims=True))
    else:
        Hm = None
    di = jnp.asarray(cm.di)
    solid = jnp.asarray(cm.solid.astype(float)); void = jnp.asarray(cm.void)
    nd = cm.nd

    def fields(x, beta, theta_fill, etas):
        """etas = (eta_C, eta_V): projection threshold used for the stiffness
        (compliance) and for the volume / penalty. three-field: (0.5, 0.5);
        robust (deHomTop808): eroded 0.55 for C, dilated 0.45 for V and S."""
        mu1, mu2, th, rho = x[0], x[1], x[2], x[3]
        if Hm is not None:
            mu1 = Hm @ mu1; mu2 = Hm @ mu2
        r_full = solid.at[di].set(rho)                 # fixed solid = 1, fixed void = 0
        rt = (Hf @ r_full)[di]
        rbC = heaviside(rt, beta, etas[0]); rbV = heaviside(rt, beta, etas[1]); rb = heaviside(rt, beta, ETA)
        w1 = solid.at[di].set(mu1 * rbC); w2 = solid.at[di].set(mu2 * rbC)
        theta = theta_fill.at[di].set(th)
        C = rotated_C(w1, w2, theta)
        vol = lambda r: jnp.mean(mu1 * r + mu2 * r - mu1 * mu2 * r * r)
        V = vol(rbV); S = jnp.mean(rbV); Vb = vol(rb)
        wb1 = solid.at[di].set(mu1 * rb); wb2 = solid.at[di].set(mu2 * rb)
        return C, V, S, Vb, wb1, wb2, theta, rb, rt, mu1, mu2

    @jax.jit
    def fwd(x, beta, theta_fill, etas):
        return fields(x, beta, theta_fill, etas)

    @jax.jit
    def bwd(x, beta, theta_fill, etas, gC, gammaS, invV0):
        f = lambda x_: fields(x_, beta, theta_fill, etas)[:3]
        _, vjp = jax.vjp(f, x)
        dobj = vjp((gC, jnp.zeros(()), gammaS))[0]
        dcon = vjp((jnp.zeros_like(gC), invV0, jnp.zeros(())))[0]
        return dobj, dcon

    return fwd, bwd


# ----------------------------------------------------------------------------
# MMA (Svanberg 1987/2002, mmasub + subsolv; port of the standard Matlab code)
# ----------------------------------------------------------------------------
class MMA:
    def __init__(self, n, m=1, asyinit=0.5, asyincr=1.2, asydecr=0.7, albefa=0.1, raa0=1e-5):
        self.n, self.m = n, m
        self.asyinit, self.asyincr, self.asydecr, self.albefa, self.raa0 = asyinit, asyincr, asydecr, albefa, raa0
        self.a0 = 1.0; self.a = np.zeros(m); self.c = 1000.0 * np.ones(m); self.d = np.ones(m)
        self.restart()

    def restart(self):
        self.k = 0; self.xold1 = None; self.xold2 = None; self.low = None; self.upp = None

    def update(self, x, xmin, xmax, f0, df0, fval, dfdx):
        n, m = self.n, self.m
        self.k += 1
        xmm = np.maximum(xmax - xmin, 1e-12)
        if self.k <= 2:
            low = x - self.asyinit * xmm; upp = x + self.asyinit * xmm
        else:
            zz = (x - self.xold1) * (self.xold1 - self.xold2)
            fac = np.ones(n); fac[zz > 0] = self.asyincr; fac[zz < 0] = self.asydecr
            low = x - fac * (self.xold1 - self.low); upp = x + fac * (self.upp - self.xold1)
            low = np.clip(low, x - 10 * xmm, x - 0.01 * xmm)
            upp = np.clip(upp, x + 0.01 * xmm, x + 10 * xmm)
        # xmin/xmax are already the move-limited box for this iteration
        alfa = np.maximum(low + self.albefa * (x - low), xmin)
        beta = np.minimum(upp - self.albefa * (upp - x), xmax)
        ux1 = upp - x; xl1 = x - low; ux2 = ux1 ** 2; xl2 = xl1 ** 2
        uxinv = 1 / ux1; xlinv = 1 / xl1
        p0 = np.maximum(df0, 0); q0 = np.maximum(-df0, 0)
        pq0 = 0.001 * (p0 + q0) + self.raa0 / xmm
        p0 = (p0 + pq0) * ux2; q0 = (q0 + pq0) * xl2
        P = np.maximum(dfdx, 0); Q = np.maximum(-dfdx, 0)
        PQ = 0.001 * (P + Q) + self.raa0 / xmm[None]
        P = (P + PQ) * ux2[None]; Q = (Q + PQ) * xl2[None]
        b = P @ uxinv + Q @ xlinv - fval
        xnew = self._subsolv(low, upp, alfa, beta, p0, q0, P, Q, b)
        self.xold2 = self.xold1 if self.xold1 is not None else x.copy()
        self.xold1 = x.copy(); self.low = low; self.upp = upp
        return xnew

    def _subsolv(self, low, upp, alfa, beta, p0, q0, P, Q, b):
        n, m = self.n, self.m
        a0, a, c, d = self.a0, self.a, self.c, self.d
        een = np.ones(n); eem = np.ones(m)
        epsi = 1.0
        x = 0.5 * (alfa + beta); y = eem.copy(); z = 1.0; lam = eem.copy()
        xsi = np.maximum(een / (x - alfa), 1); eta = np.maximum(een / (beta - x), 1)
        mu = np.maximum(eem, 0.5 * c); zet = 1.0; s = eem.copy()
        while epsi > 1e-8:
            epsvecn = epsi * een; epsvecm = epsi * eem
            for _ in range(200):
                ux1 = upp - x; xl1 = x - low; ux2 = ux1 ** 2; xl2 = xl1 ** 2
                uxinv1 = 1 / ux1; xlinv1 = 1 / xl1
                plam = p0 + P.T @ lam; qlam = q0 + Q.T @ lam
                gvec = P @ uxinv1 + Q @ xlinv1
                dpsidx = plam / ux2 - qlam / xl2
                rex = dpsidx - xsi + eta; rey = c + d * y - mu - lam; rez = a0 - zet - a @ lam
                relam = gvec - a * z - y + s - b
                rexsi = xsi * (x - alfa) - epsvecn; reeta = eta * (beta - x) - epsvecn
                remu = mu * y - epsvecm; rezet = zet * z - epsi; res = lam * s - epsvecm
                residu = np.concatenate([rex, rey, [rez], relam, rexsi, reeta, remu, [rezet], res])
                residunorm = np.sqrt(residu @ residu); residumax = np.abs(residu).max()
                if residumax <= 0.9 * epsi:
                    break
                ux3 = ux1 * ux2; xl3 = xl1 * xl2
                uxinv2 = 1 / ux2; xlinv2 = 1 / xl2
                GG = P * uxinv2[None] - Q * xlinv2[None]
                dpsidx = plam / ux2 - qlam / xl2
                delx = dpsidx - epsvecn / (x - alfa) + epsvecn / (beta - x)
                dely = c + d * y - lam - epsvecm / y
                delz = a0 - a @ lam - epsi / z
                dellam = gvec - a * z - y - b + epsvecm / lam
                diagx = 2 * (plam / ux3 + qlam / xl3) + xsi / (x - alfa) + eta / (beta - x)
                diagxinv = 1 / diagx
                diagy = d + mu / y; diagyinv = 1 / diagy
                diaglam = s / lam; diaglamyi = diaglam + diagyinv
                # m < n: solve the (m+1) system
                blam = dellam + dely / diagy - GG @ (delx / diagx)
                bb = np.concatenate([blam, [delz]])
                Alam = np.diag(diaglamyi) + (GG * diagxinv[None]) @ GG.T
                AA = np.block([[Alam, a[:, None]], [a[None, :], np.array([[-zet / z]])]])
                solut = np.linalg.solve(AA, bb)
                dlam = solut[:m]; dz = solut[m]
                dx = -delx / diagx - (GG.T @ dlam) / diagx
                dy = -dely / diagy + dlam / diagy
                dxsi = -xsi + epsvecn / (x - alfa) - (xsi * dx) / (x - alfa)
                deta = -eta + epsvecn / (beta - x) + (eta * dx) / (beta - x)
                dmu = -mu + epsvecm / y - (mu * dy) / y
                dzet = -zet + epsi / z - zet * dz / z
                ds = -s + epsvecm / lam - (s * dlam) / lam
                xx = np.concatenate([y, [z], lam, xsi, eta, mu, [zet], s])
                dxx = np.concatenate([dy, [dz], dlam, dxsi, deta, dmu, [dzet], ds])
                stepxx = -1.01 * dxx / xx
                stmxx = stepxx.max()
                stepalfa = -1.01 * dx / (x - alfa); stmalfa = stepalfa.max()
                stepbeta = 1.01 * dx / (beta - x); stmbeta = stepbeta.max()
                steg = 1 / max(stmalfa, stmbeta, stmxx, 1.0)
                xold, yold, zold, lamold = x, y, z, lam
                xsiold, etaold, muold, zetold, sold = xsi, eta, mu, zet, s
                resinew = 2 * residunorm
                for _ in range(50):
                    x = xold + steg * dx; y = yold + steg * dy; z = zold + steg * dz
                    lam = lamold + steg * dlam; xsi = xsiold + steg * dxsi; eta = etaold + steg * deta
                    mu = muold + steg * dmu; zet = zetold + steg * dzet; s = sold + steg * ds
                    ux1 = upp - x; xl1 = x - low; ux2 = ux1 ** 2; xl2 = xl1 ** 2
                    plam = p0 + P.T @ lam; qlam = q0 + Q.T @ lam
                    gvec = P @ (1 / ux1) + Q @ (1 / xl1)
                    dpsidx = plam / ux2 - qlam / xl2
                    rex = dpsidx - xsi + eta; rey = c + d * y - mu - lam; rez = a0 - zet - a @ lam
                    relam = gvec - a * z - y + s - b
                    rexsi = xsi * (x - alfa) - epsvecn; reeta = eta * (beta - x) - epsvecn
                    remu = mu * y - epsvecm; rezet = zet * z - epsi; res = lam * s - epsvecm
                    residu = np.concatenate([rex, rey, [rez], relam, rexsi, reeta, remu, [rezet], res])
                    resinew = np.sqrt(residu @ residu)
                    if resinew < residunorm:
                        break
                    steg *= 0.5
            epsi *= 0.1
        return x


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------
_CACHE = {}


def _setup(rmin, rmin_mu=None):
    cm = coarse()
    key = ("fields", float(rmin), rmin_mu)
    if key not in _CACHE:
        _CACHE[key] = make_fields(cm, rmin, rmin_mu)
    return cm, _CACHE[key]


def _solid_state():
    """Solid isotropic solve (design + non-design solid, void_fixed void):
    compliance, principal-stress angle (deHomTop808 seam fix), u."""
    if "solid" in _CACHE:
        return _CACHE["solid"]
    cm = coarse()
    w = np.where(cm.void, 0.0, 1.0)
    C = np.asarray(rotated_C(jnp.asarray(w), jnp.asarray(w), jnp.zeros(cm.nel)))
    u = cm.solve(C)
    J = float(cm.F @ u)
    eps = u[cm.edof] @ cm.Bc.T                         # (nel, 3) centroid strain
    sig = np.einsum("eij,ej->ei", C, eps)
    a = 0.5 * np.arctan2(2 * sig[:, 2], sig[:, 0] - sig[:, 1])
    a[~(a > 0)] += np.pi                               # deHomTop808 line 26
    _CACHE["solid"] = (J, a, u)
    return _CACHE["solid"]


def lf_solid_compliance():
    return _solid_state()[0]


def _apply_masks(w1, w2, theta):
    cm = coarse()
    w1 = np.asarray(w1, float).ravel().copy(); w2 = np.asarray(w2, float).ravel().copy()
    th = np.asarray(theta, float).ravel()
    w1[cm.solid] = w2[cm.solid] = 1.0
    w1[cm.void] = w2[cm.void] = 0.0
    return w1, w2, th


def _solve_design(w1, w2, theta):
    cm = coarse()
    w1, w2, th = _apply_masks(w1, w2, theta)
    C = np.asarray(_rotated_jit(jnp.asarray(w1), jnp.asarray(w2), jnp.asarray(th)))
    return cm, cm.solve(C)


_rotated_jit = jax.jit(rotated_C)


def lf_displacement(w1, w2, theta):
    """Element-centre displacement (ux, uy), each (nely, nelx), of the homogenised
    design under the design load (non-design masks enforced)."""
    cm, u = _solve_design(w1, w2, theta)
    return cm.element_centre(u)


def lf_compliance(w1, w2, theta):
    cm, u = _solve_design(w1, w2, theta)
    return float(cm.F @ u)


def lf_optimise(V0, Lmin, seed=0, rmin=None, theta_jitter=0.0, max_iter=300, beta_max=16.0,
                beta_every=40, tol=1e-3, move=MOVE, gamma_s=GAMMA_S, init_jitter=0.0,
                projection="three_field", asyincr=None, rmin_mu=None, verbose=False):
    """Homogenisation-based compliance TO on the coarse grid (formulation: see
    module docstring). Parametric for the control arm:
      seed          RNG seed for the jitters below;
      rmin          density-filter radius in coarse elements (default 2, i.e. 2 hc);
      theta_jitter  std (rad) of Gaussian noise added to the principal-stress start;
      init_jitter   std of Gaussian noise added to the initial mu (clipped to bounds);
      projection    "three_field" (our brief; default) or "robust" (deHomTop808:
                    eroded eta = 0.55 for the stiffness, dilated eta = 0.45 for
                    volume and penalty, dilated target rescaled every 20 its so the
                    blueprint (eta = 0.5) volume meets V0). Measured: three-field
                    leaves rho_bar grey (55-62 % of design elements in (0.05, 0.95))
                    because w = mu rho_bar is non-unique; robust gives crisp rho_bar;
      rmin_mu       optional width filter radius (elements); default None = no
                    filter on mu (our brief). Unfiltered mu gives checkerboard
                    widths in some lattice regions at high V0;
      asyincr       MMA asymptote expansion; default 1.05 for three-field (Svanberg's
                    1.2 gives periodic 2-3 % compliance overshoot bursts there),
                    1.2 for robust.
    beta: 1, 2, 4, ..., beta_max; doubled every `beta_every` iterations or when
    max |dx| < tol; stop when beta = beta_max and max |dx| < tol, or at max_iter.
    Returns dict with w1, w2, theta, mu1, mu2, rho (= rho_bar), each (nely, nelx),
    compliance_lf (pure J at the final design), vol_lf, history, timing."""
    t0 = time.time()
    rmin = 2.0 if rmin is None else float(rmin)
    cm, (fwd, bwd) = _setup(rmin, rmin_mu)
    J_solid, a0, _ = _solid_state()
    rng = np.random.default_rng(seed)
    nd = cm.nd; di = cm.di
    mu0 = float(np.clip(1 - np.sqrt(1 - V0), Lmin, 1.0))     # deHomTop808 line 30
    mu1 = np.full(nd, mu0); mu2 = np.full(nd, mu0)
    if init_jitter > 0:
        mu1 = np.clip(mu1 + init_jitter * rng.standard_normal(nd), Lmin, 1)
        mu2 = np.clip(mu2 + init_jitter * rng.standard_normal(nd), Lmin, 1)
    th0 = a0[di] + (theta_jitter * rng.standard_normal(nd) if theta_jitter > 0 else 0.0)
    th0 = np.clip(th0, -THETA_BOUND, THETA_BOUND)
    x = np.stack([mu1, mu2, th0, np.ones(nd)])
    lb = np.stack([np.full(nd, Lmin), np.full(nd, Lmin), np.full(nd, -THETA_BOUND), np.zeros(nd)])
    ub = np.stack([np.ones(nd), np.ones(nd), np.full(nd, THETA_BOUND), np.ones(nd)])
    mv = np.asarray(move, float)[:, None] * np.ones((1, nd))
    theta_fill = jnp.asarray(a0)                       # non-design theta = PS start (smooth field)
    gamma0 = V0 / J_solid
    if asyincr is None:
        asyincr = 1.05 if projection == "three_field" else 1.2
    mma = MMA(4 * nd, 1, asyincr=asyincr)
    etas = jnp.asarray((0.55, 0.45) if projection == "robust" else (ETA, ETA))
    V0d = V0                                            # constraint target on the volume-projection field
    beta = 1.0; stage_it = 0
    hist = []
    for it in range(1, max_iter + 1):
        stage_it += 1
        xj = jnp.asarray(x)
        C, V, S, Vb, *_ = fwd(xj, beta, theta_fill, etas)
        if projection == "robust" and it % 20 == 0:          # deHomTop808 line 104
            V0d = float(V) / float(Vb) * V0
        u = cm.solve(np.asarray(C))
        J = float(cm.F @ u)
        gC = jnp.asarray(cm.dJdC(u) * gamma0)
        dobj, dcon = bwd(xj, beta, theta_fill, etas, gC, gamma_s, 1.0 / V0d)
        obj = gamma0 * J + gamma_s * float(S)
        g = float(V) / V0d - 1.0
        xmin = np.maximum(lb, x - mv); xmax = np.minimum(ub, x + mv)
        xn = mma.update(x.ravel(), xmin.ravel(), xmax.ravel(), obj, np.asarray(dobj).ravel(), np.array([g]),
                        np.asarray(dcon).ravel()[None]).reshape(4, nd)
        change = float(np.abs(xn - x).max())
        dth = float(np.abs(xn[2] - x[2]).max())
        hist.append(dict(it=it, beta=beta, J=J, obj=obj, V=float(V), Vb=float(Vb), S=float(S), g=g,
                         change=change, dtheta=dth))
        if verbose and (it % 10 == 0 or it == 1):
            print(f"it {it:3d} beta {beta:4.1f} J {J:.4f} obj {obj:.4f} V {float(V):.4f} change {change:.4f}")
        x = xn
        if beta >= beta_max and change < tol and stage_it > 5:
            break
        if beta < beta_max and (stage_it >= beta_every or (change < tol and stage_it > 5)):
            beta = min(2 * beta, beta_max); stage_it = 0; mma.restart()
    # final state at the final x
    xj = jnp.asarray(x)
    # reported design = blueprint (eta = 0.5) projection; compliance of that design
    _, _, S, V, w1, w2, theta, rb, rt, mu1p, mu2p = fwd(xj, beta, theta_fill, etas)
    J = lf_compliance(np.asarray(w1), np.asarray(w2), np.asarray(theta))
    shp = cm.shape
    mu1f = np.where(cm.solid, 1.0, 0.0); mu1f[di] = np.asarray(mu1p)
    mu2f = np.where(cm.solid, 1.0, 0.0); mu2f[di] = np.asarray(mu2p)
    rhof = np.where(cm.solid, 1.0, 0.0); rhof[di] = np.asarray(rb)
    return dict(w1=np.asarray(w1).reshape(shp), w2=np.asarray(w2).reshape(shp),
                theta=np.asarray(theta).reshape(shp), mu1=mu1f.reshape(shp), mu2=mu2f.reshape(shp),
                rho=rhof.reshape(shp), compliance_lf=J, vol_lf=float(V), S=float(S), beta=beta,
                theta0=np.asarray(theta_fill.at[di].set(th0)).reshape(shp),
                history=hist, n_iter=len(hist), wall_s=time.time() - t0,
                params=dict(V0=V0, Lmin=Lmin, seed=seed, rmin=rmin, theta_jitter=theta_jitter,
                            init_jitter=init_jitter, max_iter=max_iter, beta_max=beta_max,
                            projection=projection, asyincr=asyincr, rmin_mu=rmin_mu,
                            beta_every=beta_every, tol=tol, move=list(move), gamma_s=gamma_s))


# ----------------------------------------------------------------------------
# Initial population sweep (480 designs)
# ----------------------------------------------------------------------------
def _worker_init():
    os.environ["JAX_PLATFORMS"] = "cpu"


def _sweep_one(args):
    V0, Lmin, kw = args
    r = lf_optimise(V0, Lmin, **kw)
    return {k: r[k] for k in ("w1", "w2", "theta", "mu1", "mu2", "rho", "compliance_lf", "vol_lf",
                              "n_iter", "wall_s", "S")} | {"hist_J": np.array([h["J"] for h in r["history"]])}


def sweep(out=None, workers=8, n_v=160, **kw):
    """V0 = linspace(0.25, 0.50, n_v) x Lmin in {0.10, 0.15, 0.20} -> gen0/lf_designs.npz."""
    import multiprocessing as mp
    from pathlib import Path
    out = Path(out or Path(__file__).resolve().parent / "results" / "gen0" / "lf_designs.npz")
    V0s = np.linspace(0.25, 0.50, n_v)
    tasks = [(float(v), float(L), kw) for L in (0.10, 0.15, 0.20) for v in V0s]
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_worker_init) as pool:
        res = pool.map(_sweep_one, tasks, chunksize=1)
    wall = time.time() - t0
    stack = lambda k: np.stack([r[k] for r in res])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **{k: stack(k) for k in ("w1", "w2", "theta", "mu1", "mu2", "rho")},
                        V0=np.array([t[0] for t in tasks]), Lmin=np.array([t[1] for t in tasks]),
                        compliance_lf=stack("compliance_lf"), vol_lf=stack("vol_lf"),
                        n_iter=stack("n_iter"), wall_s=stack("wall_s"), S=stack("S"),
                        hist_J_final100=np.stack([np.pad(r["hist_J"][-100:], (100 - len(r["hist_J"][-100:]), 0),
                                                          constant_values=np.nan) for r in res]),
                        settings=json.dumps(dict(kw, rmin=kw.get("rmin", 2.0), max_iter=300)))
    return dict(wall_total_s=wall, per_design_s=[r["wall_s"] for r in res], out=str(out), results=res, tasks=tasks)


# Canonical gen-0 settings (orchestrator decision 05:40 UTC, see log.md): deHomTop808's
# robust projection (crisp void indicator, Lmin acts as a real minimum width) and a
# width filter (removes checkerboard widths for ~1 % compliance). Every arm, including
# the MORELF control, generates LF designs with these settings.
GEN0_SETTINGS = dict(projection="robust", rmin_mu=1.5)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "gen0":
        from pathlib import Path
        r = sweep(out=Path(__file__).resolve().parent / "results" / "gen0" / "lf_designs.npz",
                  workers=int(os.environ.get("LF_WORKERS", 4)),
                  **GEN0_SETTINGS)
        print(f"gen0 sweep wall {r['wall_total_s']:.1f}s, per design mean {np.mean(r['per_design_s']):.2f}s")
        sys.exit(0)
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    if len(sys.argv) > 1 and sys.argv[1] == "sweep":
        proj = sys.argv[2] if len(sys.argv) > 2 else "three_field"
        from pathlib import Path
        name = "lf_designs.npz" if proj == "three_field" else f"lf_designs_{proj}.npz"
        r = sweep(out=Path(__file__).resolve().parent / "results" / "gen0" / name, workers=8, projection=proj)
        print(f"sweep wall {r['wall_total_s']:.1f}s, per design mean {np.mean(r['per_design_s']):.2f}s "
              f"max {np.max(r['per_design_s']):.2f}s -> {r['out']}")
    else:
        print("solid LF compliance", lf_solid_compliance())
        print("rotation check", rotation_check())
        r = lf_optimise(0.35, 0.10, verbose=True)
        print(f"J {r['compliance_lf']:.4f} V {r['vol_lf']:.4f} iters {r['n_iter']} wall {r['wall_s']:.1f}s")
