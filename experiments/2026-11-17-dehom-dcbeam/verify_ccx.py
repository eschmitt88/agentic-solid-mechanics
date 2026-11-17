"""[dehom-dcbeam: copied unchanged from trial 14; GIRDER-specific, NOT adapted to hf.PROBLEM = DCBEAM.]
Independent body-fitted verification oracle (grading only, never in the loop).

design (fine 200 x 400 field, [iy, ix], iy = 0 at y = 0) -> pixel clean exactly as
hf.Problem.clean (masks + drop solid islands not 4-connected to the skin) ->
supersampled continuous boundary field (bilinear interpolation of the pixel field,
analytic non-design masks, optional re-entrant-corner rounding of scale r_c, see
verify()) -> marching-squares contours (skimage) -> outer loop / hole
classification by containment -> equal-arc-length resampling (Xu et al. Eqs. 15-18,
piecewise between sharp corners; eps_shape / eps_acc of Eqs. 19-20) -> gmsh plane
surface with holes, quadratic triangles, min boundary spacing = Delta s ->
CalculiX 2.23: static compliance + linear buckling.

CalculiX modelling notes (see scratch/ccx/t_*.py):
  * Plane-stress CPS6 + *BUCKLE in ccx 2.23 gives wrong buckling factors (Euler
    column: 731x the Euler load); CPS6 static is fine. So the triangle mesh is
    extruded into ONE layer of C3D15 wedges, thickness t = 1, with u_z = 0 on the
    mid-thickness (vertical-edge midside) nodes and the in-plane displacements of
    every node column tied by *EQUATION (plane-stress kinematics, faces free in z).
    Euler column check: 1.0012 x Euler (hf Q4 h=0.25: 1.001).
  * ccx *BUCKLE only returns factors > 1 (shift at 1; negative factors are never
    returned). The buckle step therefore carries the design load scaled by
    s_buck (1e-3), and BLF = s_buck * factor_1. Checked on a two-column model
    with one column in tension.
  * Supports: distributed SPRING1 springs to ground under the pads (left pad x+y,
    right pad y), k_line per mm, lumped per node with the consistent quadratic-edge
    weights (L/6, 2L/3, L/6) on the z = 0 node of each tied column.
  * Load: total P downward, uniform along the top edge y = Ly, consistent nodal
    loads (L/6, 2L/3, L/6) per quadratic edge.
Material E = 1, nu = 0.3, unit thickness (hf.GIRDER).

Grading defaults (calibration in scratch/ccx/, report to the orchestrator 2026-11-16):
r_c = 0 (no rounding: the paper's marching-squares geometry), mesh_size = Delta s =
0.5 mm. On three dehomogenised gen-0 designs the BLF changes <= 2 % and the
compliance <= 1 % from mesh 0.5 to 0.25; 20-45 s per design (3 threads).
Rounding (r_c > 0) is available but changes the design (BLF +10..35 %, compliance
-3..-17 % on those designs), so it is a sensitivity check, not the default.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
import uuid

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

import hf

HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "results", "ccx_work")
MAMBA_ENV = os.path.expanduser("~/projects/.micromamba/envs/solidmech/bin")
CCX = os.path.join(MAMBA_ENV, "ccx")
GMSH = os.path.join(MAMBA_ENV, "gmsh")
NU = 0.3

_PROB = {}


def _problem(cfg, h):
    key = (id(cfg), h)
    if key not in _PROB:
        _PROB[key] = hf.Problem(cfg, h)
    return _PROB[key]


# ----------------------------------------------------------------------------
# 1. boundary field on a supersampled grid
# ----------------------------------------------------------------------------
def _sdf_box(X, Y, x0, x1, y0, y1):
    """Signed distance (positive inside) of an axis-aligned box (exact)."""
    dx = np.maximum(x0 - X, X - x1); dy = np.maximum(y0 - Y, Y - y1)
    out = np.hypot(np.maximum(dx, 0), np.maximum(dy, 0))
    ins = np.minimum(np.maximum(dx, dy), 0)
    return -(out + ins)


def _binary_sdf(B, hs):
    """Signed distance (positive inside) of a binary set sampled at cell centres."""
    din = ndimage.distance_transform_edt(B)
    dout = ndimage.distance_transform_edt(~B)
    return np.where(B, din - 0.5, -(dout - 0.5)) * hs


def boundary_field(design, kind="phi", cfg=hf.GIRDER, h=hf.H_FINE, r_c=0.0, ss=4, pad=1.0):
    """Continuous field g (positive = solid, ~signed distance in mm) on a supersampled
    grid; returns (g, xs, ys, info). Masks + island removal exactly as hf.clean."""
    p = _problem(cfg, h)
    design = np.asarray(design)
    if kind == "phi":
        phi = design.astype(bool)
    elif kind == "level":
        phi = design >= 0                         # dehom contract: phi == (level >= 0)
    else:
        raise ValueError(kind)
    s, connected = p.clean(phi)
    if kind == "phi":
        gpix = (s.astype(float) - 0.5) * h
    else:
        L = design.astype(float)
        gy, gx = np.gradient(L, h)
        gpix = np.clip(L / np.maximum(np.hypot(gx, gy), 1e-12), -2 * h, 2 * h)
        gpix[~s] = np.minimum(gpix[~s], -0.5 * h)       # removed islands, masked void
        gpix[s] = np.maximum(gpix[s], 1e-3 * h)         # masked solid stays solid at its centre
    Lx, Ly = cfg["Lx"], cfg["Ly"]
    hs = h / ss
    nx = int(round((Lx + 2 * pad) / hs)); ny = int(round((Ly + 2 * pad) / hs))
    xs = -pad + (np.arange(nx) + 0.5) * hs; ys = -pad + (np.arange(ny) + 0.5) * hs
    X, Y = np.meshgrid(xs, ys)
    g = ndimage.map_coordinates(gpix, [Y / h - 0.5, X / h - 0.5], order=1, mode="nearest")
    # analytic non-design geometry
    rect = _sdf_box(X, Y, 0, Lx, 0, Ly)
    solid = Y - (Ly - cfg["skin"])
    pw, pt = cfg["pad_w"], cfg["pad_t"]
    solid = np.maximum(solid, _sdf_box(X, Y, -1, pw, -1, pt))
    solid = np.maximum(solid, _sdf_box(X, Y, Lx - pw, Lx + 1, -1, pt))
    notduct = np.full_like(X, np.inf)
    for (cx, cy, ri, ro) in cfg.get("liners", []):
        r = np.hypot(X - cx, Y - cy)
        solid = np.maximum(solid, np.minimum(r - ri, ro - r))
    for (cx, cy, r0) in cfg.get("ducts", []):
        notduct = np.minimum(notduct, np.hypot(X - cx, Y - cy) - r0)
    g = np.maximum(g, solid)
    g = np.minimum(np.minimum(g, notduct), rect)
    if r_c > 0:
        # morphological closing (dilate then erode) by a disc of radius r_c:
        # fills re-entrant corners with fillets of radius r_c (and slots < 2 r_c).
        B = g > 0
        D1 = ndimage.distance_transform_edt(~B) * hs <= r_c
        gcl = _binary_sdf(D1, hs) - r_c
        g = np.maximum(g, gcl)
        g = np.minimum(np.minimum(g, notduct), rect)
    info = dict(connected_pix=connected, vf_pix=p.volume_fraction(s), hs=hs, rect=rect, notduct=notduct)
    return g, xs, ys, info


def _raster_loops(loops, xs, ys):
    """even-odd fill of closed polygons on the supersampled grid (cell centres),
    vectorised scanline: crossings of every edge with every row, sorted per row,
    filled between consecutive pairs."""
    hs = xs[1] - xs[0]
    ny, nx = len(ys), len(xs)
    E = np.vstack([np.c_[P, np.roll(P, -1, 0)] for P in loops])
    x0 = (E[:, 0] - xs[0]) / hs; y0 = (E[:, 1] - ys[0]) / hs
    x1 = (E[:, 2] - xs[0]) / hs; y1 = (E[:, 3] - ys[0]) / hs
    lo = np.minimum(y0, y1); hi = np.maximum(y0, y1)
    r0 = np.ceil(lo).astype(int); r1 = np.ceil(hi).astype(int)          # rows r with lo <= r < hi
    cnt = np.maximum(r1 - r0, 0)
    e = np.repeat(np.arange(len(E)), cnt)
    r = r0[e] + (np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt))
    t = (r - y0[e]) / (y1[e] - y0[e])
    xc = x0[e] + t * (x1[e] - x0[e])
    o = np.lexsort((xc, r)); r = r[o]; xc = xc[o]
    D = np.zeros((ny, nx + 1), int)
    ra, xa, xb = r[0::2], xc[0::2], xc[1::2]
    ca = np.clip(np.ceil(xa).astype(int), 0, nx); cb = np.clip(np.floor(xb).astype(int) + 1, 0, nx)
    ok = (ra >= 0) & (ra < ny) & (cb > ca)
    np.add.at(D, (ra[ok], ca[ok]), 1); np.add.at(D, (ra[ok], cb[ok]), -1)
    return np.cumsum(D, axis=1)[:, :nx] > 0


def fillet_field(g, xs, ys, info, r_c, cfg=hf.GIRDER):
    """Re-entrant-corner rounding without bridging: smooth every contour of g along its
    arc length (smooth_loop, std r_c; points on the domain rectangle fixed) and take the
    union of the smoothed solid with the original one. Re-entrant corners get a fillet
    (material added), convex corners and bar tips stay as they are, and two separate
    boundary pieces never merge (a slot only gets shorter by ~r_c at its end; small
    holes shrink). Returns the new field (positive = solid)."""
    lp = extract_loops(g, xs, ys, cfg, keep_all=True)
    Lx, Ly = cfg["Lx"], cfg["Ly"]
    S = []
    for P in lp:
        P = _snap(P, cfg, 2.5 * info["hs"])
        onrect = (P[:, 0] == 0) | (P[:, 0] == Lx) | (P[:, 1] == 0) | (P[:, 1] == Ly)
        S.append(smooth_loop(P, r_c, onrect if onrect.any() else None))
    Bs = _raster_loops(S, xs, ys)
    gs = _binary_sdf(Bs, info["hs"])
    out = np.maximum(g, gs)
    return np.minimum(np.minimum(out, info["notduct"]), info["rect"])


# ----------------------------------------------------------------------------
# 2. contours, classification, resampling
# ----------------------------------------------------------------------------
def _area(P):
    x, y = P[:, 0], P[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def _inside(P, Q):
    """points Q inside closed polygon P (even-odd, vectorised over Q)."""
    from matplotlib.path import Path
    return Path(P).contains_points(Q)


def extract_loops(g, xs, ys, cfg=hf.GIRDER, keep_all=False):
    from skimage.measure import find_contours
    hs = xs[1] - xs[0]
    cs = find_contours(g, 0.0, fully_connected="low")
    loops = []
    for c in cs:
        P = np.c_[xs[0] + c[:, 1] * hs, ys[0] + c[:, 0] * hs]
        if np.allclose(P[0], P[-1]):
            P = P[:-1]
        if len(P) >= 3 and abs(_area(P)) > 1e-6:
            loops.append(P)
    if keep_all:
        return loops
    n = len(loops)
    areas = np.array([abs(_area(P)) for P in loops])
    order = np.argsort(-areas)
    loops = [loops[i] for i in order]; areas = areas[order]
    # containment: parent = smallest loop containing a vertex of this loop
    probe = np.array([P[0] for P in loops])
    depth = np.zeros(n, int); parent = -np.ones(n, int)
    for j in range(n):
        ins = _inside(loops[j], probe)
        ins[: j + 1] = False                       # only smaller loops can be inside j
        for i in np.where(ins)[0]:
            depth[i] += 1
            parent[i] = j                          # loops sorted by area: last hit = smallest
    Lx, Ly = cfg["Lx"], cfg["Ly"]
    skin_pt = np.array([[Lx / 2, Ly - cfg["skin"] / 2]])
    main = [i for i in range(n) if depth[i] % 2 == 0 and _inside(loops[i], skin_pt)[0]]
    if not main:
        return None
    # the skin point lies in exactly one solid region: the outer loop with no hole covering it
    m = max(main, key=lambda i: depth[i])
    holes = [i for i in range(n) if parent[i] == m and depth[i] == depth[m] + 1]
    dropped = n - 1 - len(holes)
    return dict(outer=loops[m], holes=[loops[i] for i in holes], n_loops=n, n_dropped_loops=dropped)


def _snap(P, cfg, tol):
    """Snap near-edge coordinates onto the domain rectangle and replace the corner
    chamfers of the rectangle by the exact corners."""
    Lx, Ly = cfg["Lx"], cfg["Ly"]
    P = P.copy()
    for k, v in ((0, 0.0), (0, Lx), (1, 0.0), (1, Ly)):
        P[np.abs(P[:, k] - v) < 1e-7, k] = v
    corners = np.array([[0, 0], [Lx, 0], [Lx, Ly], [0, Ly]], float)
    keep = np.ones(len(P), bool); ins = {}
    for c in corners:
        d = np.hypot(*(P - c).T)
        near = np.where(d < tol)[0]
        if len(near):
            keep[near] = False
            ins[near[np.argmin(d[near])]] = c
    out = []
    for i in range(len(P)):
        if i in ins:
            out.append(ins[i])
        elif keep[i]:
            out.append(P[i])
    return np.array(out)


def _insert_on_bottom(P, xcut):
    """Insert vertex (xcut, 0) where a bottom-edge (y = 0) segment straddles x = xcut."""
    out = []; marks = []
    n = len(P)
    for i in range(n):
        a, b = P[i], P[(i + 1) % n]
        out.append(a); marks.append(False)
        if a[1] == 0.0 and b[1] == 0.0:
            for xc in xcut:
                if (a[0] - xc) * (b[0] - xc) < 0:
                    out.append(np.array([xc, 0.0])); marks.append(True)
    return np.array(out), np.array(marks)


def _corner_marks(P, win, ang_deg=35.0):
    """Vertices where the polyline turns by more than ang_deg over arc +-win."""
    n = len(P)
    seg = np.hypot(*(np.roll(P, -1, 0) - P).T)
    s = np.r_[0, np.cumsum(seg)]; L = s[-1]
    s = s[:-1]
    # positions at arc distance -win and +win (periodic)
    def at(t):
        t = np.mod(t, L)
        j = np.searchsorted(s, t, side="right") - 1
        f = (t - s[j]) / np.maximum(seg[j], 1e-15)
        return P[j] + f[:, None] * (P[(j + 1) % n] - P[j])
    A = at(s - win); B = at(s + win)
    u = P - A; v = B - P
    cosang = np.sum(u * v, 1) / np.maximum(np.hypot(*u.T) * np.hypot(*v.T), 1e-15)
    ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    cand = ang > ang_deg
    marks = np.zeros(n, bool)
    # keep local maxima within the window
    k = max(1, int(np.ceil(win / max(np.median(seg), 1e-12))))
    for i in np.where(cand)[0]:
        idx = np.arange(i - k, i + k + 1) % n
        if ang[i] >= ang[idx].max():
            marks[i] = True
    return marks


def _uniform(Q, delta):
    """resample an open polyline Q (endpoints kept) at ~uniform arc spacing delta."""
    seg = np.hypot(*np.diff(Q, axis=0).T)
    s = np.r_[0, np.cumsum(seg)]
    n = max(2, int(np.ceil(s[-1] / delta)) + 1)
    t = np.linspace(0, s[-1], n)
    return np.c_[np.interp(t, s, Q[:, 0]), np.interp(t, s, Q[:, 1])]


def smooth_loop(P, sigma, fixed=None, delta=0.03):
    """Corner rounding without topology change: Gaussian smoothing of the closed
    contour's coordinates along arc length (std sigma, mm). Vertices flagged in
    `fixed` (on the domain rectangle) stay put; free runs between them are smoothed
    with odd reflection about their end points, so a straight edge through a fixed
    point stays straight and the end point does not move. A 90-degree corner becomes
    a smooth bend of curvature radius ~ sigma; straight edges are unchanged; a circle
    of radius R shrinks by ~ sigma^2 / 2R. Unlike a morphological closing this
    never fills slots or holes."""
    from scipy.ndimage import gaussian_filter1d
    if sigma <= 0:
        return P
    k = sigma / delta
    if fixed is None or not fixed.any():
        Q = _uniform(np.r_[P, P[:1]], delta)[:-1]
        return np.c_[gaussian_filter1d(Q[:, 0], k, mode="wrap"), gaussian_filter1d(Q[:, 1], k, mode="wrap")]
    n = len(P)
    start = int(np.where(fixed)[0][0])
    P = np.roll(P, -start, 0); fixed = np.roll(fixed, -start)
    idx = list(np.where(fixed)[0]) + [n]
    out = []
    for a, b in zip(idx[:-1], idx[1:]):
        if b == a + 1:                                   # consecutive fixed points: keep the segment
            out.append(P[a:a + 1]); continue
        Q = _uniform(P[np.r_[a:b, b % n] % n], delta)
        m = min(len(Q) - 2, int(4 * k) + 1)
        if m < 1:
            out.append(Q[:-1]); continue
        A, B = Q[0], Q[-1]
        Z = np.r_[2 * A - Q[1:m + 1][::-1], Q, 2 * B - Q[-m - 1:-1][::-1]]
        Zs = np.c_[gaussian_filter1d(Z[:, 0], k, mode="nearest"), gaussian_filter1d(Z[:, 1], k, mode="nearest")]
        Qs = Zs[m:m + len(Q)]
        Qs[0] = A; Qs[-1] = B
        out.append(Qs[:-1])
    return np.vstack(out)


def ear_resample(P, ds, marks=None):
    """Equal-arc-length resampling (Xu et al. Eqs. 15-18) of a closed polyline,
    piecewise between marked vertices (exact corners kept). Returns P~."""
    n = len(P)
    if marks is None or not marks.any():
        marks = np.zeros(n, bool); marks[0] = True
    idx = np.where(marks)[0]
    out = []
    for a, b in zip(idx, np.r_[idx[1:], idx[0] + n]):
        Q = P[np.arange(a, b + 1) % n]
        seg = np.hypot(*np.diff(Q, axis=0).T)
        s = np.r_[0, np.cumsum(seg)]                          # Eq. 15
        L = s[-1]
        if L <= 0:
            continue
        nseg = max(1, int(round(L / ds)))
        dsk = L / nseg                                        # Eq. 16 (N_ear - 1 = nseg)
        t = np.arange(nseg) * dsk                             # endpoint b is the next piece's start
        j = np.clip(np.searchsorted(s, t, side="right") - 1, 0, len(seg) - 1)
        f = (t - s[j]) / np.maximum(seg[j], 1e-15)
        out.append(Q[j] + f[:, None] * (Q[j + 1] - Q[j]))     # Eq. 17
    R = np.vstack(out)
    # drop near-duplicates
    d = np.hypot(*(np.roll(R, -1, 0) - R).T)
    R = R[d > 1e-9]
    return R


def _seg_dist(points, R):
    """distance of each point to the closed polyline R (nearest-vertex neighbourhood)."""
    tree = cKDTree(R)
    _, k = tree.query(points, k=min(4, len(R)))
    k = np.atleast_2d(k)
    best = np.full(len(points), np.inf)
    n = len(R)
    for col in range(k.shape[1]):
        for off in (-1, 0):
            a = R[(k[:, col] + off) % n]; b = R[(k[:, col] + off + 1) % n]
            ab = b - a; ap = points - a
            t = np.clip(np.sum(ap * ab, 1) / np.maximum(np.sum(ab * ab, 1), 1e-30), 0, 1)
            best = np.minimum(best, np.hypot(*(ap - t[:, None] * ab).T))
    return best


def shape_error(P, R):
    """Eq. 19 (point-set Hausdorff) and polyline (point-to-segment) Hausdorff."""
    tP, tR = cKDTree(P), cKDTree(R)
    e19 = max(tP.query(R)[0].max(), tR.query(P)[0].max())
    eseg = max(_seg_dist(P, R).max(), _seg_dist(R, P).max())
    return e19, eseg


# ----------------------------------------------------------------------------
# 3. gmsh
# ----------------------------------------------------------------------------
def write_geo(path, loops, lc, ds):
    pid = 0; lid = 0; s = []
    loop_ids = []
    for k, P in enumerate(loops):
        first = pid + 1
        for (x, y) in P:
            pid += 1; s.append(f"Point({pid}) = {{{x:.10g}, {y:.10g}, 0, {lc:.6g}}};")
        lines = []
        for i in range(len(P)):
            a = first + i; b = first + (i + 1) % len(P)
            lid += 1; s.append(f"Line({lid}) = {{{a}, {b}}};"); lines.append(lid)
        s.append(f"Curve Loop({k+1}) = {{{', '.join(map(str, lines))}}};")
        loop_ids.append(k + 1)
    s.append(f"Plane Surface(1) = {{{', '.join(map(str, loop_ids))}}};")
    s.append("Physical Surface(1) = {1};")
    s += [f"Mesh.MeshSizeMax = {lc:.6g};", f"Mesh.MeshSizeMin = {min(ds, lc) * 0.5:.6g};",
          "Mesh.MeshSizeExtendFromBoundary = 0;", "Mesh.MeshSizeFromCurvature = 0;",
          "Mesh.Algorithm = 6;", "Mesh.ElementOrder = 2;", "Mesh.SecondOrderLinear = 1;",
          "Mesh.HighOrderOptimize = 0;", "Mesh.Optimize = 1;", "Mesh.MshFileVersion = 2.2;"]
    open(path, "w").write("\n".join(s) + "\n")


def run_gmsh(geo, msh, threads=4, timeout=600):
    r = subprocess.run([GMSH, geo, "-2", "-nt", str(threads), "-o", msh, "-format", "msh22", "-v", "2"],
                       capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout + r.stderr


def read_msh(msh):
    import meshio
    m = meshio.read(msh)
    T = np.vstack([c.data for c in m.cells if c.type == "triangle6"])
    X = m.points[:, :2]
    used = np.unique(T)
    remap = -np.ones(len(X), int); remap[used] = np.arange(len(used))
    X = X[used]; T = remap[T]
    a, b, c = X[T[:, 0]], X[T[:, 1]], X[T[:, 2]]
    ar = 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    neg = ar < 0
    T[neg] = T[neg][:, [0, 2, 1, 5, 4, 3]]
    return X, T, np.abs(ar)


# ----------------------------------------------------------------------------
# 4. CalculiX deck (one layer of C3D15 wedges)
# ----------------------------------------------------------------------------
def _boundary_edges(T):
    E = np.r_[T[:, [0, 1, 3]], T[:, [1, 2, 4]], T[:, [2, 0, 5]]]
    key = np.sort(E[:, :2], 1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    return E[cnt[inv.ravel()] == 1]


def build_deck(X, T, cfg, P=1.0, s_buck=1e-3, nev=3, t=1.0, buckling=True):
    nn = len(X)
    corners = np.unique(T[:, :3])
    midmap = -np.ones(nn, int); midmap[corners] = np.arange(len(corners))
    nmid = 2 * nn + np.arange(len(corners))                     # 0-based ids of mid-thickness nodes
    L = ["*HEADING", "verify_ccx: body-fitted C3D15 one-layer plane-stress model"]
    L.append("*NODE, NSET=NALL")
    L += [f"{i+1},{x:.12g},{y:.12g},0" for i, (x, y) in enumerate(X)]
    L += [f"{nn+i+1},{x:.12g},{y:.12g},{t:.12g}" for i, (x, y) in enumerate(X)]
    L += [f"{2*nn+k+1},{X[c,0]:.12g},{X[c,1]:.12g},{t/2:.12g}" for k, c in enumerate(corners)]
    L.append("*ELEMENT, TYPE=C3D15, ELSET=EALL")
    for e, tr in enumerate(T):
        b = tr + 1; tp = tr + nn + 1; m = midmap[tr[:3]] + 2 * nn + 1
        ids = [b[0], b[1], b[2], tp[0], tp[1], tp[2], b[3], b[4], b[5], tp[3], tp[4], tp[5], m[0], m[1], m[2]]
        L.append(f"{e+1}," + ",".join(map(str, ids[:10])) + ",")
        L.append(",".join(map(str, ids[10:])))
    L.append("*NSET, NSET=NMID")
    L += [",".join(str(n + 1) for n in nmid[i:i + 16]) for i in range(0, len(nmid), 16)]
    L += ["*MATERIAL, NAME=MAT", "*ELASTIC", f"1.0, {NU}", "*SOLID SECTION, ELSET=EALL, MATERIAL=MAT",
          "*BOUNDARY", "NMID,3,3"]
    Lx, Ly = cfg["Lx"], cfg["Ly"]; pw = cfg["pad_w"]; kl = cfg.get("k_line", 1.0)
    taper = cfg.get("k_taper", 0.0)          # 0 = abrupt pad edge, as hf.py
    be = _boundary_edges(T)
    ya = X[be[:, 0], 1]; yb = X[be[:, 1], 1]
    # --- springs: Simpson weights along each bottom-pad edge
    kx = np.zeros(nn); ky = np.zeros(nn)
    bot = be[(ya == 0.0) & (yb == 0.0)]
    for a, b, m in bot:
        xa, xb = X[a, 0], X[b, 0]; le = abs(xb - xa); xm = 0.5 * (xa + xb)
        w = np.array([le / 6, le / 6, 2 * le / 3])
        if taper > 0:                       # optional soft pad edge: k ramps 0 -> k_line over `taper` mm
            xx = X[[a, b, m], 0]
            d_in = np.where(xm < pw, pw - xx, xx - (Lx - pw))
            w = w * np.clip(d_in / taper, 0, 1)
        if xm < pw:
            kx[[a, b, m]] += kl * w; ky[[a, b, m]] += kl * w
        elif xm > Lx - pw:
            ky[[a, b, m]] += kl * w
    # Plane-stress kinematics: the in-plane displacement of every node column
    # (z = 0, t/2, t at in-plane corners; z = 0, t at midsides) is tied to the z = 0
    # node by *EQUATION; u_z stays free on the faces (thickness change, sigma_zz = 0).
    # Without the ties the one-layer model has through-thickness modes that plane
    # stress does not (solid girder, mesh 1.0: BLF 2.253 untied vs 2.347 tied = the
    # untied model's 2nd mode) and lumped face springs bulge the face.
    # Springs and loads then act on the z = 0 node with 2D consistent edge weights.
    corner_set = np.zeros(nn, bool); corner_set[corners] = True
    eid = len(T)
    sp = []
    for n in range(nn):
        others = (n + nn, 2 * nn + midmap[n]) if corner_set[n] else (n + nn,)
        for o in others:
            sp += ["*EQUATION", "2", f"{o+1},1,1.0,{n+1},1,-1.0", "*EQUATION", "2", f"{o+1},2,1.0,{n+1},2,-1.0"]
    for dof, kk in ((1, kx), (2, ky)):
        for n in np.where(kk > 0)[0]:
            eid += 1
            sp += [f"*ELEMENT, TYPE=SPRING1, ELSET=S{dof}_{n+1}", f"{eid},{n+1}",
                   f"*SPRING, ELSET=S{dof}_{n+1}", f"{dof}", f"{kk[n]:.12g}"]
    L += sp
    # --- top load: uniform line load q = P / Lx, consistent quadratic-edge weights
    top = be[(ya == Ly) & (yb == Ly)]
    q = P / Lx
    F = {}
    def add(n, v):
        F[n] = F.get(n, 0.0) + v
    for a, b, m in top:
        le = abs(X[b, 0] - X[a, 0])
        add(a, q * le / 6); add(b, q * le / 6); add(m, 2 * q * le / 3)
    Fn = np.array(sorted(F)); Fv = np.array([F[n] for n in Fn])
    cl = lambda sc, op="": [f"*CLOAD{op}"] + [f"{n+1},2,{-sc*v:.14g}" for n, v in zip(Fn, Fv)]
    L += ["*STEP", "*STATIC"] + cl(1.0) + ["*NODE FILE", "U", "*END STEP"]
    if buckling:
        L += ["*STEP", "*BUCKLE", f"{nev}, 1e-5"] + cl(s_buck, ", OP=NEW") + ["*NODE FILE", "U", "*END STEP"]
    meta = dict(nn2=nn, Fn=Fn, Fv=Fv, kx=kx, ky=ky, n_top_edges=len(top), n_bot_edges=len(bot),
                top_len=float(np.abs(X[top[:, 1], 0] - X[top[:, 0], 0]).sum()))
    return "\n".join(L) + "\n", meta


def parse_frd(path):
    """list of dict(time, name, nodes (0-based), values) per result block."""
    out = []; cur = None; t = None
    with open(path) as f:
        for line in f:
            k = line[:5]
            if k == "  100":
                t = float(line[12:24])
            elif k == " -4  ":
                cur = dict(time=t, name=line[5:13].strip(), nodes=[], values=[])
                out.append(cur)
            elif line[:3] == " -1":
                if cur is not None:
                    cur["nodes"].append(int(line[3:13]))
                    vals = line[13:].rstrip("\n")
                    cur["values"].append([float(vals[i:i + 12]) for i in range(0, len(vals), 12)])
            elif line[:3] == " -3":
                if cur is not None:
                    cur["nodes"] = np.array(cur["nodes"]) - 1; cur["values"] = np.array(cur["values"])
                cur = None
    return out


def parse_buckling_factors(path):
    txt = open(path).read()
    m = re.search(r"B U C K L I N G   F A C T O R   O U T P U T(.*?)(?:\n\s*\n\s*\n|\Z)", txt, re.S)
    if not m:
        return []
    return [float(v) for v in re.findall(r"^\s+\d+\s+(\S+)\s*$", m.group(1), re.M)]


def run_ccx(job, workdir, threads=4, timeout=1800):
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), CCX_NPROC_STIFFNESS=str(threads),
               CCX_NPROC_EQUATION_SOLVER=str(threads))
    r = subprocess.run([CCX, job], cwd=workdir, env=env, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout + r.stderr


# ----------------------------------------------------------------------------
# 5. driver
# ----------------------------------------------------------------------------
def verify(design, kind="level", r_c=0.0, mesh_size=0.5, ds=None, buckling=True, cfg=hf.GIRDER,
           h=hf.H_FINE, ss=4, nev=3, s_buck=1e-3, threads=4, keep=False, tag=None, return_mesh=False,
           rounding="fillet", static_r_c=0.0):
    """Body-fitted CalculiX verification of a fine design field (200 x 400, [iy, ix]).

    kind: "phi" (bool, solid = True) or "level" (float, solid = level >= 0, dehom.py).
    r_c: corner-rounding scale (mm) of the model used for buckling.
         rounding="fillet" (default): union of the solid with its arc-length-smoothed
         copy (fillet_field): re-entrant corners get fillets of radius ~r_c, convex
         corners, bar tips, slots and holes are kept (no bridging).
         rounding="smooth": the smoothed contours alone (also rounds convex corners).
         rounding="close": morphological closing, disc radius r_c (bridges slots and
         fills holes narrower than 2 r_c: too invasive on real designs; record only).
    static_r_c: if not None and != r_c (or rounding="close"), compliance and umax come
         from a separate static-only model with this rounding (default 0: the
         unrounded contour, as in the paper); the rounded model's compliance is then
         reported as compliance_rc. Fillets at bar joints stiffen bending-dominated
         lattices strongly (45-degree lattice w=0.3: -20 % compliance at r_c=0.5).
    mesh_size: max triangle size (mm); ds: boundary arc-length interval (default mesh_size).
    Returns dict: compliance (F.u), umax, blf, blf_modes, mode_xy, mode_pr, n_elements, n_nodes
    (2D tri6 mesh; n_nodes_3d, n_dof), eps_shape (Eq. 19), eps_seg, eps_acc (Eq. 20, %),
    vf_ccx, vf_pix, spring_sum_y/x, load_sum, wall times, ok flags."""
    t0 = time.time()
    ds = mesh_size if ds is None else ds
    kw = dict(kind=kind, mesh_size=mesh_size, ds=ds, cfg=cfg, h=h, ss=ss, threads=threads, keep=keep,
              rounding=rounding, static_r_c=None)
    if static_r_c is not None and r_c > 0 and abs(static_r_c - r_c) > 1e-12:
        if buckling:
            rb = verify(design, r_c=r_c, buckling=True, nev=nev, s_buck=s_buck, return_mesh=return_mesh, **kw)
        else:
            rb = {}
        rs = verify(design, r_c=static_r_c, buckling=False, **kw)
        out = dict(rb) if buckling else dict(rs)
        out["compliance_rc"] = rb.get("compliance")
        for k in ("compliance", "umax", "umax_xy", "vf_ccx", "eps_shape", "eps_seg", "eps_acc", "n_holes"):
            out[k] = rs.get(k)
        out["static_r_c"] = static_r_c; out["r_c"] = r_c
        out["vf_ccx_rc"] = rb.get("vf_ccx")
        out["n_elements_static"] = rs.get("n_elements")
        out["ok"] = bool(rs.get("ok")) and (bool(rb.get("ok")) if buckling else True)
        out["wall"] = time.time() - t0
        return out
    out = dict(kind=kind, r_c=r_c, rounding=rounding, mesh_size=mesh_size, ds=ds, ok=False)
    g, xs, ys, info = boundary_field(design, kind, cfg, h, r_c if rounding == "close" else 0.0, ss)
    if rounding == "fillet" and r_c > 0 and info["connected_pix"]:
        g = fillet_field(g, xs, ys, info, r_c, cfg)
    out.update(connected_pix=info["connected_pix"], vf_pix=info["vf_pix"])
    if not info["connected_pix"]:
        out.update(compliance=np.inf, umax=np.inf, blf=0.0, reason="pixel design not connected")
        return out
    lp = extract_loops(g, xs, ys, cfg)
    if lp is None:
        out.update(reason="no solid region contains the skin"); return out
    Lx, Ly = cfg["Lx"], cfg["Ly"]; pw, pt = cfg["pad_w"], cfg["pad_t"]
    hs = info["hs"]
    loops_raw = [lp["outer"]] + lp["holes"]
    # connectivity of pads to the meshed region
    pads_pt = np.array([[pw / 2, pt / 2], [Lx - pw / 2, pt / 2]])
    con = _inside(lp["outer"], pads_pt)
    for Hh in lp["holes"]:
        con &= ~_inside(Hh, pads_pt)
    out["connected"] = bool(con.all())
    out.update(n_loops=lp["n_loops"], n_holes=len(lp["holes"]), n_dropped_loops=lp["n_dropped_loops"])
    pre = []                                   # contour loops after snapping / smoothing, with corner marks
    for k, P in enumerate(loops_raw):
        if k == 0:
            P = _snap(P, cfg, 2.5 * hs)
        if rounding == "smooth" and r_c > 0:
            onrect = (P[:, 0] == 0) | (P[:, 0] == Lx) | (P[:, 1] == 0) | (P[:, 1] == Ly)
            P = smooth_loop(P, r_c, onrect if k == 0 else None)
        if k == 0:
            P, ins = _insert_on_bottom(P, [pw, Lx - pw])
            marks = _corner_marks(P, 2 * hs) | ins
            for c in ([0, 0], [Lx, 0], [Lx, Ly], [0, Ly]):
                marks |= np.all(P == c, axis=1)
        else:
            marks = _corner_marks(P, 2 * hs)
        pre.append((P, marks))

    def resample(dsk):
        loops = []; e19 = 0.0; eseg = 0.0; Ltot = 0.0; eacc = 100.0
        for k, (P, marks) in enumerate(pre):
            R = ear_resample(P, dsk, marks)
            if k == 0:
                R = _snap(R, cfg, 1e-9)
            if len(R) < 3:
                continue
            if (_area(R) > 0) != (k == 0):          # outer CCW, holes CW
                R = R[::-1]
            a, b = shape_error(P, R)
            Lk = np.hypot(*(np.roll(P, -1, 0) - P).T).sum()
            e19 = max(e19, a); eseg = max(eseg, b); Ltot += Lk
            eacc = min(eacc, (1 - a / Lk) * 100)
            loops.append(R)
        # eps_acc: Eq. 20 per loop, worst loop (small holes dominate); eps_acc_total: eps_shape / total length
        return loops, dict(eps_shape=float(e19), eps_seg=float(eseg), eps_acc=float(eacc),
                           eps_acc_total=float((1 - e19 / Ltot) * 100), contour_len=float(Ltot),
                           n_boundary_pts=int(sum(len(R) for R in loops)))

    t1 = time.time()
    os.makedirs(WORK, exist_ok=True)
    job = tag or f"v{uuid.uuid4().hex[:10]}"
    wd = os.path.join(WORK, job); os.makedirs(wd, exist_ok=True)
    try:
        # Delta s = ds first; if gmsh fails or the mesh area does not match the polygon
        # area (chords of the resampled boundary crossing in a narrow gap), retry with
        # a finer boundary (ds/2, ds/4). out["ds_used"] records what was meshed.
        for attempt, dsk in enumerate((ds, ds / 2, ds / 4)):
            loops, st = resample(dsk)
            out.update(st)
            area_poly = abs(_area(loops[0])) - sum(abs(_area(R)) for R in loops[1:])
            write_geo(os.path.join(wd, "m.geo"), loops, mesh_size, dsk)
            rc, log = run_gmsh(os.path.join(wd, "m.geo"), os.path.join(wd, "m.msh"), threads)
            try:
                X, T, ar = read_msh(os.path.join(wd, "m.msh"))
                good = rc == 0 and len(T) > 0 and abs(ar.sum() - area_poly) < 1e-6 * area_poly + 1e-6
            except Exception:
                good = False
            out["ds_used"] = dsk; out["mesh_attempts"] = attempt + 1
            if good:
                break
        if not good:
            out.update(reason="gmsh failed", gmsh_log=log[-2000:]); return out
        out.update(n_elements=int(len(T)), n_nodes=int(len(X)), mesh_area=float(ar.sum()),
                   poly_area=float(area_poly))
        out["mesh_ok"] = True
        A_fixed = _fixed_area(cfg)
        A_design = Lx * Ly - A_fixed - sum(np.pi * r ** 2 for (_, _, r) in cfg.get("ducts", []))
        out["vf_ccx"] = float((ar.sum() - A_fixed) / A_design)
        t2 = time.time()
        deck, meta = build_deck(X, T, cfg, cfg["P"], s_buck, nev, buckling=buckling)
        open(os.path.join(wd, "job.inp"), "w").write(deck)
        rc, log = run_ccx("job", wd, threads)
        t3 = time.time()
        if "Job finished" not in log:
            out.update(reason="ccx failed", ccx_log=log[-3000:]); return out
        nn = meta["nn2"]
        blocks = [b for b in parse_frd(os.path.join(wd, "job.frd")) if b["name"] == "DISP"]
        U = np.zeros((3 * nn + 1, 3))
        b0 = blocks[0]; U[b0["nodes"]] = b0["values"][:, :3]
        u = U[:nn, :2]
        F = np.zeros(len(U)); F[meta["Fn"]] = meta["Fv"]
        out["compliance"] = float(-(F * U[:len(F), 1]).sum())      # F is downward: F_y = -Fv
        out["umax"] = float(np.hypot(u[:, 0], u[:, 1]).max())
        out["umax_xy"] = X[np.argmax(np.hypot(u[:, 0], u[:, 1]))].tolist()
        out["spring_sum_y"] = float(-(meta["ky"] * U[:nn, 1]).sum())
        out["spring_sum_x"] = float(-(meta["kx"] * U[:nn, 0]).sum())
        out["load_sum"] = float(meta["Fv"].sum()); out["top_len"] = meta["top_len"]
        out["uz_asym"] = float(np.abs(U[:nn, 2] + U[nn:2 * nn, 2]).max())
        out["n_nodes_3d"] = int(2 * nn + len(np.unique(T[:, :3])))
        if buckling:
            fac = parse_buckling_factors(os.path.join(wd, "job.dat"))
            out["ccx_factors"] = fac
            if fac:
                out["blf"] = float(s_buck * fac[0])
                out["blf_modes"] = [float(s_buck * f) for f in fac]
                out["blf_ok"] = bool(fac[0] > 1.5)          # factors < 1 are never returned by ccx
                mb = [b for b in blocks[1:] if b["time"] is not None and abs(b["time"] - fac[0]) <= 1e-4 * fac[0]]
                if mb:
                    V = np.zeros((3 * nn + 1, 3)); V[mb[0]["nodes"]] = mb[0]["values"][:, :3]
                    v = mode_deformation(X, V[:nn, :2])
                    out["mode_xy"] = X[np.argmax(v)].tolist()
                    w = v ** 2
                    out["mode_pr"] = float(w.sum() ** 2 / (len(w) * (w ** 2).sum()))   # participation ratio
                    if return_mesh:
                        out["mode_mag"] = v
            else:
                out["blf"] = np.nan; out["blf_ok"] = False
        out["ok"] = bool(out["mesh_ok"] and out["connected"] and abs(out["spring_sum_y"] - cfg["P"]) < 1e-3)
        out["t_geom"] = t1 - t0; out["t_mesh"] = t2 - t1; out["t_ccx"] = t3 - t2
        if return_mesh:
            out["X"] = X; out["T"] = T; out["u"] = u; out["loops"] = loops
    finally:
        out["wall"] = time.time() - t0
        if not keep:
            shutil.rmtree(wd, ignore_errors=True)
    return out


def mode_deformation(X, v):
    """|mode displacement| after removing the least-squares rigid motion (translation +
    rotation): on soft springs the raw mode carries rigid motion that hides where it
    deforms. Returns per-node magnitude; argmax = mode location, participation ratio
    of its square ~ 1/(fraction of nodes involved)."""
    xc = X - X.mean(0)
    A = np.zeros((2 * len(X), 3))
    A[0::2, 0] = 1; A[1::2, 1] = 1; A[0::2, 2] = -xc[:, 1]; A[1::2, 2] = xc[:, 0]
    c, *_ = np.linalg.lstsq(A, v.ravel(), rcond=None)
    r = (v.ravel() - A @ c).reshape(-1, 2)
    return np.hypot(r[:, 0], r[:, 1])


def _fixed_area(cfg):
    Lx = cfg["Lx"]
    A = Lx * cfg["skin"] + 2 * cfg["pad_w"] * cfg["pad_t"]
    A += sum(np.pi * (ro ** 2 - ri ** 2) for (_, _, ri, ro) in cfg.get("liners", []))
    return A


if __name__ == "__main__":
    import json, sys
    p = _problem(hf.GIRDER, hf.H_FINE)
    r = verify(np.ones((p.nely, p.nelx), bool), kind="phi", r_c=0.0, mesh_size=float(sys.argv[1]) if len(sys.argv) > 1 else 1.0)
    print(json.dumps({k: v for k, v in r.items() if not isinstance(v, np.ndarray)}, indent=1, default=float))
