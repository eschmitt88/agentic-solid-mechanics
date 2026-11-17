"""High-fidelity pixel evaluator (in-loop HF model) for trial 13.

A binary field phi[iy, ix] (iy = 0 at the bottom, y up; one Q4 plane-stress
element per pixel) is evaluated for compliance, max |u| and the first linear
buckling load factor. Void pixels get E = E_VOID; their geometric stiffness is
zeroed so void cannot produce spurious buckling modes. Solid islands that are
not 4-connected to the loaded skin are removed before analysis.

Conventions shared by every module of this trial:
  arrays are [iy, ix], iy = 0 at y = 0; physical units mm; E = 1, nu = 0.3.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy import ndimage
import pypardiso

E_VOID = 1e-6
NU = 0.3


def q4_matrices(nu=NU):
    """Unit-E plane-stress Q4 stiffness (8x8), geometric-stiffness kernels
    Mxx, Myy, Mxy (8x8, multiply by sigma components) and the centroid
    strain-displacement matrix B_c (3x8, times 1/h). Square element of side h;
    KE and the M kernels are h-independent in 2D with unit thickness.
    Node order: (0,0) (1,0) (1,1) (0,1), dofs [u1 v1 u2 v2 ...]."""
    D = 1.0 / (1 - nu ** 2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
    xi_n = np.array([-1, 1, 1, -1.0])
    eta_n = np.array([-1, -1, 1, 1.0])
    g = 1 / np.sqrt(3)
    KE = np.zeros((8, 8)); Mxx = np.zeros((8, 8)); Myy = np.zeros((8, 8)); Mxy = np.zeros((8, 8))

    def dN(xi, eta):
        # derivatives wrt physical x, y for an element of side 2 in (xi,eta) mapped to side 1
        dxi = 0.25 * xi_n * (1 + eta_n * eta)
        deta = 0.25 * eta_n * (1 + xi_n * xi)
        return dxi * 2.0, deta * 2.0          # d/dx = 2 d/dxi for unit side

    for xi in (-g, g):
        for eta in (-g, g):
            dx, dy = dN(xi, eta)
            B = np.zeros((3, 8))
            B[0, 0::2] = dx; B[1, 1::2] = dy; B[2, 0::2] = dy; B[2, 1::2] = dx
            w = 0.25                            # |J| * weight for unit square
            KE += w * B.T @ D @ B
            Gx = np.zeros((2, 8)); Gy = np.zeros((2, 8))
            Gx[0, 0::2] = dx; Gx[1, 1::2] = dx
            Gy[0, 0::2] = dy; Gy[1, 1::2] = dy
            Mxx += w * Gx.T @ Gx
            Myy += w * Gy.T @ Gy
            Mxy += w * (Gx.T @ Gy + Gy.T @ Gx)
    dx, dy = dN(0.0, 0.0)
    Bc = np.zeros((3, 8))
    Bc[0, 0::2] = dx; Bc[1, 1::2] = dy; Bc[2, 0::2] = dy; Bc[2, 1::2] = dx
    return KE, Mxx, Myy, Mxy, Bc, D


class Problem:
    """Geometry + boundary conditions on a pixel grid.

    masks (all [iy, ix] bool): solid_fixed (non-design solid), void_fixed
    (non-design void), design = ~(solid_fixed | void_fixed).
    """

    def __init__(self, cfg: dict, h: float):
        self.cfg = cfg
        self.h = h
        Lx, Ly = cfg["Lx"], cfg["Ly"]
        self.nelx, self.nely = int(round(Lx / h)), int(round(Ly / h))
        nx, ny = self.nelx, self.nely
        xc = (np.arange(nx) + 0.5) * h
        yc = (np.arange(ny) + 0.5) * h
        X, Y = np.meshgrid(xc, yc)                      # [iy, ix]
        self.Xc, self.Yc = X, Y
        m = _masks(cfg, X, Y)
        self.skin = m["skin"]
        self.pad_mask = m["pads"]
        self.solid_fixed = m["solid_fixed"]
        self.void_fixed = m["void_fixed"]
        self.design = m["design"]
        pw = cfg["pad_w"]

        # node ids: node (ix, iy) -> ix * (ny+1) + iy
        nid = lambda ix, iy: ix * (ny + 1) + iy
        ex, ey = np.meshgrid(np.arange(nx), np.arange(ny))   # [iy, ix]
        n1 = nid(ex, ey); n2 = nid(ex + 1, ey); n3 = nid(ex + 1, ey + 1); n4 = nid(ex, ey + 1)
        conn = np.stack([n1, n2, n3, n4], -1).reshape(-1, 4)      # element order = [iy, ix].ravel()
        self.edof = np.stack([2 * conn, 2 * conn + 1], -1).reshape(-1, 8)
        self.ndof = 2 * (nx + 1) * (ny + 1)
        self.iK = np.repeat(self.edof, 8, axis=1).ravel()
        self.jK = np.tile(self.edof, (1, 8)).ravel()
        self.KE, self.Mxx, self.Myy, self.Mxy, self.Bc, self.D = q4_matrices()

        # supports: distributed elastic springs (Winkler) under the pads.
        # Fixed nodes give a singular traction at the pad edge and a spurious
        # corner buckling mode whose load factor -> 0 as h -> 0 (see log.md);
        # springs keep the traction bounded. Left pad: x and y springs (pin),
        # right pad: y springs (roller). k_line per mm of pad, E = 1 units.
        xs = np.arange(nx + 1) * h
        trib = np.full(nx + 1, h); trib[0] = trib[-1] = h / 2
        kl = cfg.get("k_line", 10.0)
        kdiag = np.zeros(self.ndof)
        for i in range(nx + 1):
            n = nid(i, 0)
            if xs[i] <= pw + 1e-9:
                w = trib[i] if xs[i] < pw - 1e-9 else h / 2
                kdiag[2 * n] += kl * w; kdiag[2 * n + 1] += kl * w
            elif xs[i] >= Lx - pw - 1e-9:
                w = trib[i] if xs[i] > Lx - pw + 1e-9 else h / 2
                kdiag[2 * n + 1] += kl * w
        self.kdiag = kdiag
        self.fixed = np.array([], dtype=int)
        self.free = np.arange(self.ndof)
        # uniform downward line load on the top edge, total P
        F = np.zeros(self.ndof)
        top = [nid(i, ny) for i in range(nx + 1)]
        w = np.full(nx + 1, 1.0); w[0] = w[-1] = 0.5
        w *= cfg["P"] / w.sum()
        F[2 * np.array(top) + 1] = -w
        self.F = F

    # ------------------------------------------------------------------
    def clean(self, phi: np.ndarray):
        """Apply non-design masks and drop solid islands not 4-connected to
        the loaded skin. Returns (phi_clean, connected_flag)."""
        s = phi.astype(bool).copy()
        s[self.solid_fixed] = True
        s[self.void_fixed] = False
        lab, n = ndimage.label(s)                         # 4-connectivity default
        keep = np.unique(lab[self.skin]); keep = keep[keep > 0]
        s = np.isin(lab, keep)
        # both pads must be in the skin's component
        pads = self.solid_fixed & ~self.skin
        connected = bool(s[pads].all())
        return s, connected

    def volume_fraction(self, s):
        return float(s[self.design].mean()) if self.design.any() else 1.0

    def _filter_stress(self, sig, s, rf):
        from scipy.signal import fftconvolve
        R = int(np.ceil(rf / self.h))
        yy, xx = np.mgrid[-R:R + 1, -R:R + 1]
        ker = ((xx * self.h) ** 2 + (yy * self.h) ** 2 <= rf ** 2).astype(float)
        m = s.astype(float)
        den = fftconvolve(m, ker, mode="same")
        out = np.empty_like(sig)
        for k in range(3):
            num = fftconvolve(sig[:, k].reshape(s.shape) * m, ker, mode="same")
            out[:, k] = (num / np.maximum(den, 1e-12)).ravel()
        return out

    def assemble(self, Ee):
        sK = (self.KE.ravel()[None, :] * Ee[:, None]).ravel()
        K = sp.csr_matrix((sK, (self.iK, self.jK)), shape=(self.ndof, self.ndof))
        return K + sp.diags(self.kdiag)

    def evaluate(self, phi: np.ndarray, buckling: bool = False, nev: int = 2) -> dict:
        s, connected = self.clean(phi)
        vf = self.volume_fraction(s)
        out = {"vf": vf, "connected": connected}
        if not connected:
            out.update(compliance=np.inf, umax=np.inf, blf=0.0)
            return out
        Ee = np.where(s.ravel(), 1.0, E_VOID)
        Ee[self.pad_mask.ravel()] = self.cfg.get("pad_E", 1.0)
        K = self.assemble(Ee)
        fr = self.free
        Kff = K[fr][:, fr].tocsr()
        solver = pypardiso.PyPardisoSolver()
        try:
            solver.set_matrix_type(2)                         # real SPD: upper triangle only
            Kup = sp.triu(Kff, format="csr"); Kup.sort_indices()
            # Factorise once, then solve-only (phase 33). pypardiso's solve() without a prior
            # factorize() refactorised on every eigsh iteration: identical numbers, 5-10x slower
            # buckling (trial 14 log, 16:20 UTC).
            solver.factorize(Kup)

            def fsolve(b):
                solver.set_phase(33)
                return solver._call_pardiso(Kup, np.ascontiguousarray(b, dtype=np.float64))

            uf = fsolve(self.F[fr])
            u = np.zeros(self.ndof); u[fr] = uf
            out["compliance"] = float(self.F @ u)
            uy = u[1::2]; ux = u[0::2]
            out["umax"] = float(np.sqrt(ux ** 2 + uy ** 2).max())
            if buckling:
                ue = u[self.edof]                                  # (nel, 8)
                sig = (Ee[:, None] * (ue @ (self.D @ (self.Bc / self.h)).T))   # (nel, 3)
                solid = s.ravel()
                rf = self.cfg.get("kg_filter_r", 0.0)
                if rf > 0:
                    # Stress regularisation for the geometric stiffness: average the
                    # element stress over a disc of physical radius rf, solid pixels
                    # only (normalised convolution). Kills corner-singularity modes,
                    # keeps the section-mean axial stress that drives member buckling.
                    sig = self._filter_stress(sig, s, rf)
                sx, sy, txy = (sig[:, k] * solid for k in range(3))
                kg = (sx[:, None] * self.Mxx.ravel()[None] + sy[:, None] * self.Myy.ravel()[None]
                      + txy[:, None] * self.Mxy.ravel()[None]).ravel()
                KG = sp.csr_matrix((kg, (self.iK, self.jK)), shape=(self.ndof, self.ndof))[fr][:, fr].tocsr()
                # (K + lam KG) psi = 0  <=>  (-KG) psi = (1/lam) K psi ; largest mu = 1/lam1
                Minv = sla.LinearOperator(Kff.shape, matvec=fsolve, dtype=float)
                mu = sla.eigsh(-KG, k=nev, M=Kff, Minv=Minv, which="LA", return_eigenvectors=False,
                               tol=self.cfg.get("eig_tol", 1e-6), ncv=self.cfg.get("eig_ncv", None), maxiter=5000)
                mu = np.sort(mu)[::-1]
                pos = mu[mu > 0]
                out["blf"] = float(1.0 / pos[0]) if len(pos) else np.inf
                out["blf_modes"] = [float(1 / m) for m in pos]

        finally:
            # MKL keeps the factorisation until phase -1: without this each call leaked
            # ~190 MB and 16 workers OOM-killed the box (log 15:03 UTC).
            solver.free_memory(everything=True)
        out["u"] = u
        out["s"] = s
        return out


# Final problem definition (trial 13). Snapped to the 2.5 mm coarse grid
# (40 x 20 coarse elements, hc = 2.5 mm). Fine HF grid h = 0.25 mm (400 x 200).
GIRDER = dict(
    Lx=100.0, Ly=50.0,
    skin=2.5,                 # non-design solid deck on top, loaded
    pad_w=5.0, pad_t=2.5,     # non-design solid bearing pads, bottom corners
    P=1.0,                    # total uniformly distributed downward load on the top edge
    k_line=1.0,               # bearing springs per mm of pad (E = 1 units); left x+y, right y
    ducts=[(25.0, 22.5, 6.0)],            # passive void (cx, cy, r): service duct at quarter span
    liners=[(25.0, 22.5, 6.0, 8.5)],      # non-design solid liner annulus (cx, cy, r_in, r_out)
    kg_filter_r=1.0,          # stress-filter radius for the geometric stiffness (mm)
    eig_tol=1e-3, eig_ncv=12,
)
HC = 2.5
H_FINE = 0.25


def coarse_masks(cfg=GIRDER, hc=HC):
    """Non-design masks on the coarse grid, by element-centre test."""
    nx, ny = int(round(cfg["Lx"] / hc)), int(round(cfg["Ly"] / hc))
    X, Y = np.meshgrid((np.arange(nx) + 0.5) * hc, (np.arange(ny) + 0.5) * hc)
    return _masks(cfg, X, Y)


def _masks(cfg, X, Y):
    Lx, Ly = cfg["Lx"], cfg["Ly"]
    skin = Y >= Ly - cfg["skin"]
    pw, pt = cfg["pad_w"], cfg["pad_t"]
    pads = ((X <= pw) | (X >= Lx - pw)) & (Y <= pt)
    duct = np.zeros_like(skin)
    for (cx, cy, r) in cfg.get("ducts", []):
        duct |= (X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2
    liner = np.zeros_like(skin)
    for (cx, cy, ri, ro) in cfg.get("liners", []):
        d2 = (X - cx) ** 2 + (Y - cy) ** 2
        liner |= (d2 > ri ** 2) & (d2 <= ro ** 2)
    solid = skin | pads | liner
    void = duct & ~solid
    return dict(skin=skin, pads=pads, solid_fixed=solid, void_fixed=void, design=~(solid | void))
