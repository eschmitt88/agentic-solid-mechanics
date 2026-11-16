"""Shared helpers for the verify_ccx calibration checks (designs + hf mode location)."""
import os, sys
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import pypardiso

E = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, E)
import hf  # noqa: E402
import verify_ccx  # noqa: E402,F401


def lattice(p, lam=7.5, w=0.3, ang=np.pi / 4):
    """45-degree cross lattice on the grid of Problem p (as scratch/filter_conv.py)."""
    X, Y = p.Xc, p.Yc
    n1 = np.array([np.cos(ang), np.sin(ang)]); n2 = np.array([-np.sin(ang), np.cos(ang)])
    t1 = ((X * n1[0] + Y * n1[1]) / lam) % 1.0; t2 = ((X * n2[0] + Y * n2[1]) / lam) % 1.0
    return (np.abs(t1 - 0.5) < w / 2) | (np.abs(t2 - 0.5) < w / 2)


def lattice_level(p, lam=7.5, w=0.3, ang=np.pi / 4):
    """Continuous level (positive = solid, mm) of the same lattice: max over the two bar families
    of (half width - distance to the bar centreline)."""
    X, Y = p.Xc, p.Yc
    out = -np.inf
    for a in (ang, ang + np.pi / 2):
        n = np.array([np.cos(a), np.sin(a)])
        t = ((X * n[0] + Y * n[1]) / lam) % 1.0
        out = np.maximum(out, (w / 2 - np.abs(t - 0.5)) * lam)
    return out


def hf_eval_mode(p, phi):
    """hf.Problem.evaluate(buckling=True) with the first mode shape: returns blf, compliance,
    mode argmax (x, y) and participation ratio, using exactly hf's K, K_sigma (filtered)."""
    s, connected = p.clean(phi)
    Ee = np.where(s.ravel(), 1.0, hf.E_VOID)
    Ee[p.pad_mask.ravel()] = p.cfg.get("pad_E", 1.0)
    K = p.assemble(Ee).tocsr()
    Kup = sp.triu(K, format="csr"); Kup.sort_indices()
    sol = pypardiso.PyPardisoSolver(); sol.set_matrix_type(2)
    u = sol.solve(Kup, p.F)
    C = float(p.F @ u)
    ue = u[p.edof]
    sig = Ee[:, None] * (ue @ (p.D @ (p.Bc / p.h)).T)
    rf = p.cfg.get("kg_filter_r", 0.0)
    if rf > 0:
        sig = p._filter_stress(sig, s, rf)
    sig = sig * s.ravel()[:, None]
    kg = (sig[:, 0:1] * p.Mxx.ravel()[None] + sig[:, 1:2] * p.Myy.ravel()[None]
          + sig[:, 2:3] * p.Mxy.ravel()[None]).ravel()
    KG = sp.csr_matrix((kg, (p.iK, p.jK)), shape=(p.ndof, p.ndof))
    Minv = sla.LinearOperator(K.shape, matvec=lambda x: sol.solve(Kup, x), dtype=float)
    mu, V = sla.eigsh(-KG, k=2, M=K, Minv=Minv, which="LA", tol=1e-4, ncv=12, maxiter=5000)
    i = np.argmax(mu)
    v = V[:, i]
    nsolid = np.zeros(p.ndof // 2, bool)
    nsolid[(p.edof[s.ravel()][:, 0::2] // 2).ravel()] = True     # nodes of solid pixels only
    ixn, iyn = np.divmod(np.arange(p.ndof // 2), p.nely + 1)
    Xn = np.c_[ixn * p.h, iyn * p.h]
    from verify_ccx import mode_deformation
    mag = np.zeros(p.ndof // 2)
    mag[nsolid] = mode_deformation(Xn[nsolid], np.c_[v[0::2], v[1::2]][nsolid])
    n = np.argmax(mag)
    ix, iy = divmod(n, p.nely + 1)
    w = mag[nsolid] ** 2
    sol.free_memory(everything=True)
    return dict(blf=float(1 / mu[i]), compliance=C, mode_xy=[ix * p.h, iy * p.h],
                mode_pr=float(w.sum() ** 2 / (len(w) * (w ** 2).sum())), vf=p.volume_fraction(s))
