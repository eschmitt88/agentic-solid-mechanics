"""Homogenised stiffness of the cross cell, CH(w1, w2), for trial 13.

Cell: unit square, two orthogonal grid-aligned bars crossing at the centre,
bar 1 along local x with width fraction w1, bar 2 along local y with width
fraction w2. Base E = 1, nu = 0.3, plane stress; void E = 1e-9.
Voigt order (11, 22, 12) with ENGINEERING shear (gamma = 2 eps12), so
sigma = CH @ [eps11, eps22, gamma12].

Pipeline (run `python homog.py` to rebuild results/homog_coeffs.json):
  1. Numerical homogenisation, periodic, Andreassen & Andreasen (2014) energy
     formulation, N x N bilinear Q4 pixels (N = 200), on a 21 x 21 grid of
     (w1, w2) in [0, 1]^2 including 0 and 1, plus an off-grid validation set.
  2. Fit of the four orthotropic components C11, C22, C12, C33 (C13 = C23 = 0
     by mirror symmetry; checked numerically). Paper Eq. 4 (bivariate
     least-squares polynomial in 2w - 1) is tried first; if it misses the
     acceptance criteria the fallback is a tensor-product cubic spline
     (interpolating, not-a-knot), stored as per-cell bicubic coefficients so
     that evaluation is plain JAX arithmetic.
     The data has a genuine jump at w = 1: for any w_i < 1 the cell still has
     a slit that cuts the other bar's load path (C11(0.1, 0.95) = 0.28 vs
     C_iso11 = 1.099 at w2 = 1), so no smooth fit is both exact at w = 1 and
     accurate just below it. Ours: fit the INTERIOR data w in [0, W_EDGE]^2
     (W_EDGE = 0.95, 20 x 20 nodes) and blend linearly to C_iso over
     w_i in [W_EDGE, 1]:
        C = C_iso + (1 - a1)(1 - a2) (C_int(min(w1, W_EDGE), min(w2, W_EDGE)) - C_iso),
        a_i = clip((w_i - W_EDGE) / (1 - W_EDGE), 0, 1).
     Exact solid at w_i = 1; the ramp band is a regularisation, not data.
  3. CH(w1, w2) = sym(fit) + PD guard, + 1e-6 * C_iso floor.
     PD guard (ours): eigen-clamp at 1e-6; the clamp correction is held as a
     constant in the gradient (jax.lax.stop_gradient) so degenerate
     eigenvalues cannot produce NaN gradients. Where the clamp is inactive
     (checked on a 201 x 201 grid) this is exact.

`CH` is vectorised over leading dims and JAX-differentiable.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
COEFF_PATH = HERE / "results" / "homog_coeffs.json"

E_SOLID = 1.0
E_VOID_CELL = 1e-9
NU = 0.3
N_PIX = 200
N_GRID = 21
W_EDGE = 0.95
EIG_CLAMP = 1e-6
ISO_FLOOR = 1e-6


def c_iso(E=E_SOLID, nu=NU):
    return E / (1 - nu ** 2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])


# ----------------------------------------------------------------------------
# 1. Numerical homogenisation (numpy / scipy; offline)
# ----------------------------------------------------------------------------
def cell_mask(w1, w2, n=N_PIX):
    """Solid mask [iy, ix] of the cross cell on an n x n pixel grid. Bar widths
    are rounded to an even pixel count so the bars stay centred."""
    c = (np.arange(n) + 0.5) / n - 0.5
    k1 = 2 * int(round(w1 * n / 2)); k2 = 2 * int(round(w2 * n / 2))
    Y, X = np.meshgrid(c, c, indexing="ij")
    bar1 = np.abs(Y) < k1 / (2 * n) - 1e-12        # along x: |y| < w1/2
    bar2 = np.abs(X) < k2 / (2 * n) - 1e-12        # along y: |x| < w2/2
    return bar1 | bar2


def _q4_unit_ke(nu=NU):
    """Unit-E plane-stress Q4 stiffness for a square element (size independent)."""
    D = 1.0 / (1 - nu ** 2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
    xi_n = np.array([-1, 1, 1, -1.0]); eta_n = np.array([-1, -1, 1, 1.0])
    g = 1 / np.sqrt(3); KE = np.zeros((8, 8))
    for xi in (-g, g):
        for eta in (-g, g):
            dx = 0.25 * xi_n * (1 + eta_n * eta) * 2.0
            dy = 0.25 * eta_n * (1 + xi_n * xi) * 2.0
            B = np.zeros((3, 8))
            B[0, 0::2] = dx; B[1, 1::2] = dy; B[2, 0::2] = dy; B[2, 1::2] = dx
            KE += 0.25 * B.T @ D @ B
    return KE


def homogenise_mask(solid, E=E_SOLID, Emin=E_VOID_CELL, nu=NU):
    """Periodic homogenisation of a pixel cell (unit square). Returns 3x3 CH
    (Voigt, engineering shear)."""
    import scipy.sparse as sp
    import pypardiso
    n = solid.shape[0]
    h = 1.0 / n
    KE = _q4_unit_ke(nu)
    # periodic node ids: node (i, j) with i = x index, j = y index, mod n
    nid = lambda i, j: (i % n) * n + (j % n)
    ey, ex = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")    # [iy, ix]
    conn = np.stack([nid(ex, ey), nid(ex + 1, ey), nid(ex + 1, ey + 1), nid(ex, ey + 1)], -1).reshape(-1, 4)
    edof = np.stack([2 * conn, 2 * conn + 1], -1).reshape(-1, 8)
    Ee = np.where(solid.ravel(), E, Emin)
    ndof = 2 * n * n
    iK = np.repeat(edof, 8, axis=1).ravel(); jK = np.tile(edof, (1, 8)).ravel()
    K = sp.csr_matrix(((KE.ravel()[None] * Ee[:, None]).ravel(), (iK, jK)), shape=(ndof, ndof))
    # macroscopic unit strains -> nodal displacement of an element, u0 = eps . x
    xl = np.array([0, h, h, 0.0]); yl = np.array([0, 0, h, h])
    U0 = np.zeros((3, 8))
    U0[0, 0::2] = xl                              # eps11 = 1
    U0[1, 1::2] = yl                              # eps22 = 1
    U0[2, 0::2] = 0.5 * yl; U0[2, 1::2] = 0.5 * xl   # gamma12 = 1 (eps12 = 1/2)
    fe = -(KE @ U0.T)                             # (8, 3)
    F = np.zeros((ndof, 3))
    for k in range(3):
        np.add.at(F[:, k], edof.ravel(), (Ee[:, None] * fe[:, k][None]).ravel())
    free = np.arange(2, ndof)                     # fix node 0 (removes rigid translation)
    Kf = K[free][:, free].tocsr()
    solver = pypardiso.PyPardisoSolver()
    solver.set_matrix_type(2)
    Kup = sp.triu(Kf, format="csr"); Kup.sort_indices()
    chi = np.zeros((ndof, 3))
    chi[free] = solver.solve(Kup, F[free])
    solver.free_memory(everything=True)
    ue = chi[edof] + U0.T[None]                   # (nel, 8, 3)
    CH = np.einsum("eai,ab,ebj,e->ij", ue, KE, ue, Ee)   # / |Y| with |Y| = 1
    return 0.5 * (CH + CH.T)


def homogenise(w1, w2, n=N_PIX):
    return homogenise_mask(cell_mask(w1, w2, n))


def _hom_worker(args):
    w1, w2, n = args
    return homogenise(w1, w2, n)


def compute_dataset(points, n=N_PIX, workers=8):
    """CH for a list of (w1, w2) points. Uses the w1 <-> w2 symmetry
    (CH(w2, w1) = P CH(w1, w2) P with P swapping 11 <-> 22) to halve the work."""
    from concurrent.futures import ProcessPoolExecutor
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    pts = [tuple(map(float, p)) for p in points]
    keys = {}
    todo = []
    for p in pts:
        k = (min(p), max(p))
        if k not in keys:
            keys[k] = len(todo); todo.append(k)
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_hom_worker, [(a, b, n) for a, b in todo], chunksize=4))
    P = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1.0]])
    out = []
    for p in pts:
        C = res[keys[(min(p), max(p))]]
        if p[0] > p[1]:                           # stored as (small, large) -> swap roles
            C = P @ C @ P
        out.append(C)
    return np.array(out)


# ----------------------------------------------------------------------------
# 2. Fits
# ----------------------------------------------------------------------------
COMP = ("C11", "C22", "C12", "C33")
COMP_IJ = ((0, 0), (1, 1), (0, 1), (2, 2))


def _vander(w1, w2, deg):
    t1 = 2 * np.asarray(w1) - 1; t2 = 2 * np.asarray(w2) - 1
    return np.stack([t1 ** q * t2 ** r for q in range(deg + 1) for r in range(deg + 1)], -1)


def fit_poly(W1, W2, vals, deg):
    """Paper Eq. 4: least squares sum_{q,r<=deg} a_qr (2w1-1)^q (2w2-1)^r."""
    A = _vander(W1.ravel(), W2.ravel(), deg)
    coef, *_ = np.linalg.lstsq(A, vals.ravel(), rcond=None)
    return coef


def eval_poly_np(coef, w1, w2, deg):
    return _vander(w1, w2, deg) @ coef


def fit_spline(grid, vals):
    """Tensor-product cubic interpolating spline (not-a-knot) on a uniform grid,
    converted to per-cell bicubic coefficients c[i, j, a, b] so that on cell
    (i, j) with local s = (w1 - g_i)/dg, t = (w2 - g_j)/dg:
        f = sum_ab c[i, j, a, b] s^a t^b."""
    from scipy.interpolate import RectBivariateSpline
    spl = RectBivariateSpline(grid, grid, vals, kx=3, ky=3, s=0)
    m = len(grid) - 1
    dg = grid[1] - grid[0]
    loc = np.array([0, 1 / 3, 2 / 3, 1.0])
    V = np.vander(loc, 4, increasing=True)        # V[p, a] = loc_p^a
    Vi = np.linalg.inv(V)
    c = np.zeros((m, m, 4, 4))
    for i in range(m):
        for j in range(m):
            F = spl(grid[i] + loc * dg, grid[j] + loc * dg)    # F[p, q]
            c[i, j] = Vi @ F @ Vi.T
    return c


def eval_spline_np(c, w1, w2, dg):
    n_cells = c.shape[0]
    x1 = np.clip(np.asarray(w1, float) / dg, 0, n_cells); x2 = np.clip(np.asarray(w2, float) / dg, 0, n_cells)
    i = np.clip(np.floor(x1).astype(int), 0, n_cells - 1)
    j = np.clip(np.floor(x2).astype(int), 0, n_cells - 1)
    s = x1 - i; t = x2 - j
    S = np.stack([s ** a for a in range(4)], -1); T = np.stack([t ** b for b in range(4)], -1)
    return np.einsum("...a,...ab,...b->...", S, c[i, j], T)


def ramp_np(f_int, w1, w2, w_edge=W_EDGE):
    """Blend interior components (..., 4) to C_iso over w_i in [w_edge, 1]."""
    ci = _comp_vals(c_iso())
    a1 = np.clip((np.asarray(w1) - w_edge) / (1 - w_edge), 0, 1)
    a2 = np.clip((np.asarray(w2) - w_edge) / (1 - w_edge), 0, 1)
    return ci + ((1 - a1) * (1 - a2))[..., None] * (f_int - ci)


# ----------------------------------------------------------------------------
# 3. JAX evaluation
# ----------------------------------------------------------------------------
_MODEL = None


def load_model(path=COEFF_PATH):
    global _MODEL
    if _MODEL is None:
        with open(path) as f:
            d = json.load(f)
        _MODEL = d["model"]
        if _MODEL["kind"] == "poly":
            _MODEL["_coef"] = np.array([d["model"]["coef"][k] for k in COMP])      # (4, ncoef)
        else:
            _MODEL["_coef"] = np.array([d["model"]["coef"][k] for k in COMP])      # (4, m, m, 4, 4)
    return _MODEL


def _fit_components_jax(w1, w2, model):
    import jax.numpy as jnp
    we = model.get("w_edge", 1.0)
    w1 = jnp.clip(w1, 0.0, 1.0); w2 = jnp.clip(w2, 0.0, 1.0)
    f = _interior_jax(jnp.minimum(w1, we), jnp.minimum(w2, we), model)
    if we >= 1.0:
        return f
    ci = jnp.asarray(_comp_vals(c_iso()))
    a1 = jnp.clip((w1 - we) / (1 - we), 0.0, 1.0); a2 = jnp.clip((w2 - we) / (1 - we), 0.0, 1.0)
    return ci + ((1 - a1) * (1 - a2))[..., None] * (f - ci)


def _interior_jax(w1, w2, model):
    import jax.numpy as jnp
    coef = jnp.asarray(model["_coef"])
    if model["kind"] == "poly":
        deg = model["degree"]
        t1 = 2 * w1 - 1; t2 = 2 * w2 - 1
        P1 = jnp.stack([t1 ** q for q in range(deg + 1)], -1)
        P2 = jnp.stack([t2 ** r for r in range(deg + 1)], -1)
        cq = coef.reshape(4, deg + 1, deg + 1)
        return jnp.einsum("...q,kqr,...r->...k", P1, cq, P2)
    m = model["n_cells"]; dg = model["dg"]
    x1 = jnp.clip(w1 / dg, 0.0, m); x2 = jnp.clip(w2 / dg, 0.0, m)
    i = jnp.clip(jnp.floor(x1).astype(jnp.int32), 0, m - 1)
    j = jnp.clip(jnp.floor(x2).astype(jnp.int32), 0, m - 1)
    s = x1 - i; t = x2 - j
    S = jnp.stack([s ** a for a in range(4)], -1); T = jnp.stack([t ** b for b in range(4)], -1)
    cij = jnp.moveaxis(coef[:, i, j], 0, -3)         # (..., 4comp, 4, 4)
    return jnp.einsum("...a,...kab,...b->...k", S, cij, T)


def CH_raw(w1, w2, model=None):
    """Symmetric fitted tensor without the PD guard/floor, (..., 3, 3)."""
    import jax.numpy as jnp
    model = model or load_model()
    f = _fit_components_jax(jnp.asarray(w1, dtype=float), jnp.asarray(w2, dtype=float), model)
    z = jnp.zeros_like(f[..., 0])
    C11, C22, C12, C33 = f[..., 0], f[..., 1], f[..., 2], f[..., 3]
    return jnp.stack([jnp.stack([C11, C12, z], -1), jnp.stack([C12, C22, z], -1),
                      jnp.stack([z, z, C33], -1)], -2)


def CH(w1, w2, model=None):
    """Local-frame homogenised tensor (..., 3, 3), Voigt (11, 22, 12), engineering
    shear. PD-projected (eigen-clamp >= 1e-6, clamp held constant in the
    gradient) plus a 1e-6 * C_iso floor. JAX-differentiable in w1, w2."""
    import jax
    import jax.numpy as jnp
    C = CH_raw(w1, w2, model)
    lam, V = jnp.linalg.eigh(jax.lax.stop_gradient(C))
    corr = jnp.einsum("...ik,...k,...jk->...ij", V, jnp.maximum(lam, EIG_CLAMP) - lam, V)
    return C + jax.lax.stop_gradient(corr) + ISO_FLOOR * jnp.asarray(c_iso())


# ----------------------------------------------------------------------------
# Build + checks
# ----------------------------------------------------------------------------
def _comp_vals(Cs):
    return np.stack([Cs[..., i, j] for (i, j) in COMP_IJ], -1)


def _errors(pred, data, W1, W2, wmin=0.1):
    """Error metrics on points with w1 >= wmin and w2 >= wmin.
    rel: |pred - data| / |data| pointwise (C11, C22, C33);
    C12 is also reported relative to sqrt(C11 C22) because it is small and the
    pointwise ratio is ill-conditioned."""
    sel = (W1 >= wmin - 1e-12) & (W2 >= wmin - 1e-12)
    out = {}
    for k, name in enumerate(COMP):
        d = data[..., k][sel]; p = pred[..., k][sel]
        rel = np.abs(p - d) / np.maximum(np.abs(d), 1e-30)
        out[name] = dict(max_rel=float(rel.max()), mean_rel=float(rel.mean()),
                         argmax=[float(W1[sel][rel.argmax()]), float(W2[sel][rel.argmax()])])
    sc = np.sqrt(np.abs(data[..., 0] * data[..., 1]))[sel]
    out["C12"]["max_rel_to_sqrtC11C22"] = float((np.abs(pred[..., 2] - data[..., 2])[sel] / sc).max())
    return out


def _dataset(workers, n, cache):
    """Homogenisation data on the 21 x 21 grid + validation set; reuses the raw
    data stored in an existing coefficient file if its settings match."""
    grid = np.linspace(0, 1, N_GRID)
    vg = np.array([0.03, 0.07, 0.12, 0.18, 0.23, 0.27, 0.33, 0.41, 0.48, 0.56, 0.64, 0.72, 0.81, 0.88,
                   0.92, 0.94, 0.96, 0.98])           # 0.01 lattice -> even pixel counts at n = 200
    if cache.exists():
        d = json.load(open(cache))
        if d["report"].get("n_pix") == n and d["validation"]["widths"] == vg.tolist():
            dg = np.stack([np.array(d["data"][k]) for k in COMP], -1)
            dv = np.stack([np.array(d["validation"][k]) for k in COMP], -1)
            return grid, vg, dg, dv, d["report"]["t_homogenisation_s"], d["report"]["max_abs_C13_C23"]
    W1, W2 = np.meshgrid(grid, grid, indexing="ij"); V1, V2 = np.meshgrid(vg, vg, indexing="ij")
    t0 = time.time()
    allC = compute_dataset(np.concatenate([np.stack([W1.ravel(), W2.ravel()], -1),
                                           np.stack([V1.ravel(), V2.ravel()], -1)]), n=n, workers=workers)
    t_hom = time.time() - t0
    Cg = allC[:W1.size].reshape(N_GRID, N_GRID, 3, 3); Cv = allC[W1.size:].reshape(len(vg), len(vg), 3, 3)
    asym = float(max(np.abs(allC[..., 0, 2]).max(), np.abs(allC[..., 1, 2]).max()))
    return grid, vg, _comp_vals(Cg), _comp_vals(Cv), t_hom, asym


def build(workers=8, n=N_PIX, out=COEFF_PATH, force_kind=None, poly_degrees=(5, 6, 7, 8, 9, 10, 12)):
    grid, vg, dg, dv, t_hom, asym = _dataset(workers, n, out)
    W1, W2 = np.meshgrid(grid, grid, indexing="ij"); V1, V2 = np.meshgrid(vg, vg, indexing="ij")
    ni = int(round(W_EDGE / (grid[1] - grid[0]))) + 1          # interior nodes 0 .. W_EDGE
    gi = grid[:ni]; I1, I2 = W1[:ni, :ni], W2[:ni, :ni]; di = dg[:ni, :ni]
    interior_v = (V1 <= W_EDGE + 1e-12) & (V2 <= W_EDGE + 1e-12)
    band_v = ~interior_v & (V1 < 1) & (V2 < 1)
    report = dict(n_pix=n, n_grid=N_GRID, w_edge=W_EDGE, t_homogenisation_s=t_hom, max_abs_C13_C23=asym,
                  error_note="max_rel = max |fit-data|/|data| over validation points with "
                             "0.1 <= w1, w2 <= W_EDGE (interior); band = points with a w in (W_EDGE, 1)")

    def score(pred_v, pred_g=None):
        r = dict(val_interior=_errors(np.where(interior_v[..., None], pred_v, dv), dv, V1, V2),
                 val_band_max_rel={c: float((np.abs(pred_v - dv)[..., k][band_v & (V1 >= .1) & (V2 >= .1)]
                                             / np.abs(dv[..., k][band_v & (V1 >= .1) & (V2 >= .1)])).max())
                                   for k, c in enumerate(COMP)})
        if pred_g is not None:
            r["grid_interior"] = _errors(pred_g, di, I1, I2)
        return r

    fits = {}
    # (a) paper Eq. 4 as written: least squares on the full 21 x 21 grid, no ramp
    for deg in poly_degrees:
        co = [fit_poly(W1, W2, dg[..., k], deg) for k in range(4)]
        pv = np.stack([eval_poly_np(c, V1, V2, deg) for c in co], -1)
        pg = np.stack([eval_poly_np(c, I1, I2, deg) for c in co], -1)
        fits[f"poly{deg}_full"] = dict(coefs=co, deg=deg, edge=1.0, score=score(pv, pg))
    # (b) Eq. 4 on interior data + ramp to C_iso
    for deg in poly_degrees:
        co = [fit_poly(I1, I2, di[..., k], deg) for k in range(4)]
        ev = lambda A, B: np.stack([eval_poly_np(c, np.minimum(A, W_EDGE), np.minimum(B, W_EDGE), deg) for c in co], -1)
        pv = ramp_np(ev(V1, V2), V1, V2); pg = ev(I1, I2)
        fits[f"poly{deg}_ramp"] = dict(coefs=co, deg=deg, edge=W_EDGE, score=score(pv, pg))
    # (c) tensor-product cubic spline on interior data + ramp
    sc = [fit_spline(gi, di[..., k]) for k in range(4)]
    dgs = float(grid[1] - grid[0])
    ev = lambda A, B: np.stack([eval_spline_np(c, np.minimum(A, W_EDGE), np.minimum(B, W_EDGE), dgs) for c in sc], -1)
    fits["spline_ramp"] = dict(coefs=sc, edge=W_EDGE, score=score(ramp_np(ev(V1, V2), V1, V2)))
    report["fits"] = {k: v["score"] for k, v in fits.items()}

    def ok(sc_):
        e = sc_["val_interior"]
        return all(e[c]["max_rel"] < 0.02 for c in COMP)

    choice = force_kind
    if choice is None:
        cands = [f"poly{d}_ramp" for d in poly_degrees if d <= 8] + ["spline_ramp"]
        choice = next(c for c in cands if ok(fits[c]["score"]) or c == "spline_ramp")
    f = fits[choice]
    if choice.startswith("poly"):
        model = dict(kind="poly", degree=f["deg"], w_edge=f["edge"],
                     variables="(2w1-1)^q (2w2-1)^r, q,r<=degree (paper Eq. 4)",
                     coef={k: f["coefs"][i].tolist() for i, k in enumerate(COMP)})
    else:
        model = dict(kind="spline", n_cells=ni - 1, dg=dgs, w_edge=W_EDGE,
                     note="per-cell bicubic c[i][j][a][b]; s = w1/dg - i, t = w2/dg - j; f = sum c s^a t^b",
                     coef={k: f["coefs"][i].tolist() for i, k in enumerate(COMP)})
    report["chosen"] = choice
    report["chosen_meets_interior_2pct"] = ok(f["score"])
    res = dict(model=model, report=report,
               data=dict(grid=grid.tolist(), **{k: dg[..., i].tolist() for i, k in enumerate(COMP)}),
               validation=dict(widths=vg.tolist(), **{k: dv[..., i].tolist() for i, k in enumerate(COMP)}))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(res, fh)
    global _MODEL
    _MODEL = None
    return res


def checks(path=COEFF_PATH):
    """Acceptance checks on the cached model; returns a dict of numbers."""
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    with open(path) as f:
        d = json.load(f)
    model = load_model(path)
    Ci = c_iso()
    out = dict(chosen=d["report"]["chosen"])
    # solid at w = 1 on either bar
    s = np.linspace(0, 1, 101)
    Cw1 = np.asarray(CH(np.ones_like(s), s)); Cw2 = np.asarray(CH(s, np.ones_like(s)))
    idx = [(0, 0), (1, 1), (0, 1), (2, 2)]
    err = max(max(abs(C[:, i, j] - Ci[i, j]).max() / abs(Ci[i, j]) for (i, j) in idx) for C in (Cw1, Cw2))
    out["solid_w1_or_w2_eq_1_max_rel_err"] = float(err)
    # PD on 201 x 201, before and after projection
    g = np.linspace(0, 1, 201); G1, G2 = np.meshgrid(g, g, indexing="ij")
    lam_raw = np.linalg.eigvalsh(np.asarray(CH_raw(G1, G2)))
    lam = np.linalg.eigvalsh(np.asarray(CH(G1, G2)))
    out["min_eig_raw_201"] = float(lam_raw.min())
    out["clamp_active_points_201"] = int((lam_raw < EIG_CLAMP).any(-1).sum())
    out["min_eig_projected_201"] = float(lam.min())
    out["pd_after_projection"] = bool(lam.min() > 0)
    # sample values
    out["samples"] = {}
    for p in [(0, 0), (0.1, 0.1), (0.3, 0.1), (0.1, 0.3), (0.5, 0.5), (1, 0), (1, 0.3), (0.3, 1), (1, 1)]:
        out["samples"][str(p)] = np.round(np.asarray(CH(*map(float, p))), 6).tolist()
    # gradient sanity (finite, matches FD)
    f = lambda a, b: jnp.sum(CH(a, b) * jnp.arange(1, 10.0).reshape(3, 3))
    gr = jax.grad(f, argnums=(0, 1))(0.33, 0.21)
    e = 1e-6
    fd = ((f(0.33 + e, 0.21) - f(0.33 - e, 0.21)) / (2 * e), (f(0.33, 0.21 + e) - f(0.33, 0.21 - e)) / (2 * e))
    out["grad_vs_fd"] = [float(gr[0]), float(fd[0]), float(gr[1]), float(fd[1])]
    out["grad_finite_at_00"] = bool(np.all(np.isfinite(jax.grad(f, argnums=(0, 1))(0.0, 0.0))))
    return out


if __name__ == "__main__":
    kind = sys.argv[1] if len(sys.argv) > 1 else None
    r = build(force_kind=kind)
    rep = r["report"]
    print(f"homogenisation: {rep['t_homogenisation_s']:.1f}s, |C13|,|C23| max {rep['max_abs_C13_C23']:.2e}")
    for name, sc_ in rep["fits"].items():
        e = sc_["val_interior"]
        print(f"{name:16s} val-interior max rel %: " + " ".join(f"{c} {100*e[c]['max_rel']:7.2f}" for c in COMP)
              + f" | C12/sqrt {100*e['C12']['max_rel_to_sqrtC11C22']:.2f} | band " +
              " ".join(f"{100*v:.0f}" for v in sc_["val_band_max_rel"].values()))
    print("chosen:", rep["chosen"], "meets 2% interior:", rep["chosen_meets_interior_2pct"])
    print(json.dumps(checks(), indent=1))
