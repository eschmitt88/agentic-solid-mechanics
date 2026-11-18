"""Phasor-based de-homogenisation for trial 13: a Python port of deHomTop808.

Reference: Woldseth, Sigmund & Jensen, "An 808 line phasor-based
dehomogenisation Matlab code for multi-scale topology optimisation",
SMO 67:205 (2024), code `raw/repos/deHomTop808/deHomTop808.m` (MIT).
Method: Woldseth, Baerentzen & Sigmund, CMAME 418:116551 (2024). The
pipeline follows Xu et al. (Eng. Struct. 2027, Sec. 2.2.1, Eqs. 8-11, Fig. 4).

    X = (w1, w2, theta) on the 40 x 20 coarse grid (hc = 2.5 mm)
      -> per-layer orientation clean-up       (808 filterVectorField)
      -> phase alignment of phasor kernels    (808 phaseAlignment, 20 sweeps)
      -> phasor sampling with anisotropic Gaussian kernels and the integrated
         sampling filter, on an intermediate grid T_i (808 phasorSampling)
      -> cubic interpolation of the complex field to the 400 x 200 fine grid
      -> sawtooth Arg(G) -> triangular wave tau in [0, 1]
      -> threshold each layer by its width (tau >= 1 - w), union of layers,
         solid where w >= 0.99, absent where w < W_VOID = 0.05
      -> non-design masks.

Conventions (INTERFACES.md): arrays are [iy, ix], iy = 0 at y = 0 (y up),
physical units mm, angles CCW from +x. Bar 1 runs ALONG theta, so its wave
normal is theta + pi/2; bar 2 runs along theta + pi/2, so its wave normal is
theta (808 line 120: N = [cos(a+pi/2), sin(a+pi/2)], [cos a, sin a]).

Constants follow the .m source, not only the paper text. Where they differ:
  * Sampling filter: the .m code computes
        exp(-s*a*b/(a+b) - pi^2 |L|^2/(a+b) + 2i a/(a+b) L.(x-xe) + i(2 pi w n.(x-xe) + phi_e))
    with a = 0.9 b (808 l. 313-314, 754-756). The CMAME paper Eq. 14 has
    b^2 Delta^2 in place of a*b*s. We follow the code.
  * Kernel bandwidth b = -ln(0.1)/(2 hc^2) (808 l. 307), anisotropy
    r = (0.5 tangential, 2 normal) for kernels and alignment, alignment
    neighbourhood radius^2 R = max((1.1 lambda)^2, 4 hc^2), cut-off
    s < -ln(1e-3)/b, intermediate-grid factor ceil(hc / (0.1 lambda)) = 4.
  * 808 passes (Nx, -Ny) because its MATLAB image rows run downward. Our grid
    is y-up, so we pass (Nx, +Ny).

OUR CHOICES (deviations from 808 / Xu, each deliberate):
  1. Frequency: 808 sets omega = wMin/(dMin hc). We impose lambda = 7.5 mm
     (INTERFACES) directly: omega = 1/lambda. wMin := W_VOID = 0.05.
  2. Orientation clean-up is run once on the frame (layer-1 normal) with
     the union of the two layers' active kernels. Layer 2 is then rebuilt
     as the layer-1 normal rotated by -90 deg. 808 runs the same algorithm
     independently per layer; for an orthogonal pair the two runs make the
     same choices unless the active sets differ. 808's candidate set
     includes +-90 deg rotations ("layer jumps"). Where one is chosen, we
     also SWAP w1 and w2 at that element, so the physical bar keeps its
     width across the jump. 808 rotates the direction without swapping the
     width.
  3. Phase alignment: same weights, neighbourhood, Gauss-Seidel update and
     20 sweeps as 808. Two things differ.
     (a) 808 sorts kernels by (distance to boundary-aligned kernels,
     width descending) and starts boundary-aligned kernels at -pi/2. That
     information comes from addBoundary, part of the boundary layer, which
     is off here. Our sweep order is width descending, ties in MATLAB
     linear-index order. All initial phases are 0.
     (b) first_sweep="flood" (default): sweep 0 visits kernels in a
     priority flood (always the widest kernel adjacent to the
     already-visited set) and uses only already-visited neighbours.
     Sweeps 1..19 are plain 808 sweeps. Reason (scratch/dehom/
     compare_first_sweep.txt): with all phases at 0 and the plain 808
     sweep, uniform fields whose wave normal is along x come out tilted
     0.8 deg, with period 7.42 instead of 7.5 and visible jogs. A linear
     phase twist along the bar direction is a fixed point of the 808
     update (the neighbour prediction only encodes n.d), so a twist left
     by the first sweep never decays. The flood removes it (spacing
     7.500 +- 0.000) and lowers the median orientation deviation on the
     girder field from 5.1 to 4.5 deg. first_sweep="808" restores the
     plain behaviour; "visited" keeps 808's order with the visited-only
     first sweep.
     phase_seed != None: each layer gets a uniform random global initial
     phase (the flood start takes it, so the pattern translates), and ties
     in the order are broken randomly.
  4. Fine-grid wave: 808 interpolates the complex field cubically to a
     second intermediate grid T_if, used only for branch closure. It then
     interpolates the triangular wave *linearly* to the final grid. With
     branch closure off we interpolate the complex field cubically (Keys
     cubic convolution, a = -0.5, as MATLAB griddedInterpolant 'cubic')
     straight to the fine grid. The sawtooth and triangle are formed there.
  5. Width rules (INTERFACES): coarse w >= 0.99 -> 1 (solid), w < W_VOID -> 0
     (absent). The rest follows 808 interpThickness: w_f = linear interp of
     (indicator >= 0.25) * max(w, wMin). On the fine grid a layer is also cut
     where w_f < W_VOID. 808 lets bars taper to zero width there; the cut
     keeps them from tapering into sub-pixel slivers.
  6. Not ported: branch closure (closeBranches / pinch), the synthetic
     boundary layer (addBoundary, shell wave), the domain cut-field and
     removeIslands2D. None of these is in Xu's Eqs. 8-11. Island removal
     is hf.Problem.clean's job.
  7. Masks: coarse w1 = w2 = 1 in hf.coarse_masks()['solid_fixed'] and 0 in
     ['void_fixed']. The fine output is forced to the exact pixel masks of
     hf.Problem(GIRDER, H_FINE). In `level`, the masks enter through analytic
     signed distances (skin, pads, liner annulus, duct circle) scaled by
     2/lambda (the slope of tau). Their 0-level is then the true geometry,
     for body-fitted meshing. At pixel centres these agree in sign with the
     pixel masks.

The two sequential Gauss-Seidel loops (orientation clean-up, phase alignment)
are scalar Python over precomputed neighbour lists. They are deliberately
not JAX: on this box JAX defaults to the GPU and preallocates most of its
memory in every process that touches it. Everything else is vectorised numpy.
No dependencies beyond numpy and scipy (hf).
"""
from __future__ import annotations

import time
from functools import lru_cache

import numpy as np

import hf

# --------------------------------------------------------------------------- constants
HC = hf.HC                    # coarse element size, mm
H = hf.H_FINE                 # fine pixel size, mm
LAMBDA = 3.0 * HC             # phasor period, mm (INTERFACES)
W_VOID = 0.05                 # thinner bars are absent (also 808 wMin)
W_SOLID = 0.99                # w >= this is solid
CFG = hf.GIRDER
NXC, NYC = int(round(CFG["Lx"] / HC)), int(round(CFG["Ly"] / HC))
NXF, NYF = int(round(CFG["Lx"] / H)), int(round(CFG["Ly"] / H))
NE = NXC * NYC

OMEGA = 1.0 / LAMBDA                                  # 808 grid.omega (1/length)
SC = int(np.ceil(OMEGA * HC / 0.1))                   # 808 l. 279: intermediate factor (=4)
HI = HC / SC                                          # intermediate pixel, mm
NXI, NYI = NXC * SC, NYC * SC
BETA = -np.log(1e-1) / (2 * HC ** 2)                  # 808 l. 307 kernel bandwidth
ALPHA = 0.9 * BETA                                    # 808 l. 313 sampling-filter bandwidth
R_ALIGN = max((1.1 / OMEGA) ** 2, 4 * HC ** 2)        # 808 l. 308 neighbourhood radius^2
RX, RY = 0.5, 2.0                                     # 808 l. 311 kernel anisotropy (tangential, normal)
RRX, RRY = 0.5, 2.0                                   # alignment anisotropy
CUTOFF = -np.log(1e-3) / BETA                         # 808 l. 312
BPAINV = 1.0 / (BETA + ALPHA)
ABPAINV = ALPHA * BPAINV
BABPAINV = BETA * ABPAINV
ALIGN_ITERS = 20                                      # 808 l. 37 default
FILTER_MAX_IT = 50                                    # 808 l. 669
ALIGN_PD = int(np.ceil(1 / min(RRX, RRY)))            # 808 l. 342: 5x5 window
SWX = int(np.ceil(np.sqrt(2 * CUTOFF) / 2 / min(RX, RY) / HI))   # 808 l. 344 stencil half width
UNSAMPLED = 1.0 - 4j * np.pi                          # 808 l. 739 value outside sampling region


# --------------------------------------------------------------------------- interpolation matrices
def _lin_matrix(n_src, h_src, x_tgt):
    """1-D linear interpolation from cell centres (i+0.5)h_src, linear
    extrapolation outside (griddedInterpolant 'linear' default)."""
    t = x_tgt / h_src - 0.5
    i0 = np.clip(np.floor(t).astype(int), 0, n_src - 2)
    f = t - i0
    M = np.zeros((len(x_tgt), n_src))
    r = np.arange(len(x_tgt))
    M[r, i0] = 1 - f
    M[r, i0 + 1] += f
    return M


def _nearest_index(n_src, h_src, x_tgt):
    return np.clip(np.floor(x_tgt / h_src).astype(int), 0, n_src - 1)


def _keys(s):
    s = np.abs(s)
    return np.where(s <= 1, 1.5 * s ** 3 - 2.5 * s ** 2 + 1,
                    np.where(s <= 2, -0.5 * s ** 3 + 2.5 * s ** 2 - 4 * s + 2, 0.0))


def _cubic_matrix(n_src, h_src, x_tgt):
    """1-D Keys cubic convolution (a = -0.5) from cell centres, with Keys'
    boundary ghosts f[-1] = 3f0 - 3f1 + f2 (and recursively f[-2])."""
    P = np.zeros((n_src + 4, n_src))
    P[2:-2] = np.eye(n_src)
    g1 = np.zeros(n_src); g1[:3] = [3, -3, 1]                    # f[-1]
    P[1] = g1
    P[0] = 3 * g1 - 3 * P[2] + P[3]                               # f[-2] = 3f[-1] - 3f0 + f1
    e1 = np.zeros(n_src); e1[-3:] = [1, -3, 3]                   # f[n]
    P[-2] = e1
    P[-1] = 3 * e1 - 3 * P[-3] + P[-4]
    t = x_tgt / h_src - 0.5
    i = np.floor(t).astype(int)
    f = t - i
    K = np.zeros((len(x_tgt), n_src + 4))
    r = np.arange(len(x_tgt))
    for o, d in zip((-1, 0, 1, 2), (f + 1, f, 1 - f, 2 - f)):
        K[r, i + o + 2] += _keys(d)
    return K @ P


@lru_cache(maxsize=1)
def _grid():
    g = {}
    xc = (np.arange(NXC) + 0.5) * HC; yc = (np.arange(NYC) + 0.5) * HC
    xi = (np.arange(NXI) + 0.5) * HI; yi = (np.arange(NYI) + 0.5) * HI
    xf = (np.arange(NXF) + 0.5) * H; yf = (np.arange(NYF) + 0.5) * H
    g["Xc"], g["Yc"] = np.meshgrid(xc, yc)
    g["Xf"], g["Yf"] = np.meshgrid(xf, yf)
    # separable interpolation: field_t = Ay @ F @ Ax.T
    g["c2i"] = (_lin_matrix(NYC, HC, yi), _lin_matrix(NXC, HC, xi))
    g["c2f"] = (_lin_matrix(NYC, HC, yf), _lin_matrix(NXC, HC, xf))
    g["c2i_nn"] = (_nearest_index(NYC, HC, yi), _nearest_index(NXC, HC, xi))
    g["i2f"] = (_cubic_matrix(NYI, HI, yf), _cubic_matrix(NXI, HI, xf))

    # MATLAB linear-index order of coarse elements (column-major, row 1 = top
    # = our iy = NYC-1); used for Gauss-Seidel sweep order and tie-breaks.
    iy, ix = np.divmod(np.arange(NE), NXC)                 # our flat index = iy*NXC + ix
    g["mat_rank"] = ix * NYC + (NYC - 1 - iy)
    g["mat_order"] = np.argsort(g["mat_rank"], kind="stable")

    # 3x3 neighbourhood (808 filter1), pad index NE
    def window(pd):
        off = [(dy, dx) for dx in range(-pd, pd + 1) for dy in range(-pd, pd + 1)]
        idx = np.full((NE, len(off)), NE, dtype=np.int32)
        for m, (dy, dx) in enumerate(off):
            jy, jx = iy + dy, ix + dx
            ok = (jy >= 0) & (jy < NYC) & (jx >= 0) & (jx < NXC)
            idx[ok, m] = (jy * NXC + jx)[ok]
        return idx, np.array(off, dtype=float)
    g["nb3"], _ = window(1)
    g["nb5"], off5 = window(ALIGN_PD)
    # d = x_centre - x_neighbour = -(offset)*hc   (808 xldista)
    g["d5x"] = np.broadcast_to(-off5[:, 1] * HC, (NE, len(off5))).copy()
    g["d5y"] = np.broadcast_to(-off5[:, 0] * HC, (NE, len(off5))).copy()

    # phasor-sampling stencil (808 prepPhasorKernels): (2 SWX + 2)^2 points
    ms_x = np.floor((ix + 0.5) * SC + 0.5).astype(int) - 1      # 0-based centre index
    ms_y = np.floor((iy + 0.5) * SC + 0.5).astype(int) - 1
    o = np.arange(-SWX, SWX + 2)
    g["st_jx"] = ms_x[:, None, None] + o[None, None, :]          # (NE, 1, n)
    g["st_jy"] = ms_y[:, None, None] + o[None, :, None]          # (NE, n, 1)

    # fine masks (exact) and analytic signed distances for `level`
    prob_f = hf.Problem(CFG, H)
    g["solid_f"], g["void_f"] = prob_f.solid_fixed.copy(), prob_f.void_fixed.copy()
    cm = hf.coarse_masks()
    g["solid_c"], g["void_c"] = cm["solid_fixed"], cm["void_fixed"]
    g["sd_solid"], g["sd_void"] = _mask_sdf(CFG, g["Xf"], g["Yf"])
    return g


def _mask_sdf(cfg, X, Y):
    """Signed distances (mm) to the non-design regions. sd_solid > 0 inside
    non-design solid (skin, pads, liner), sd_void > 0 inside the duct. Domain
    edges are not material boundaries, so pads and skin extend outward."""
    Lx, Ly = cfg["Lx"], cfg["Ly"]
    sd = Y - (Ly - cfg["skin"])                                     # skin
    pw, pt = cfg["pad_w"], cfg["pad_t"]

    def box_sdf(ex, ey):           # ex, ey: signed distances inside along each axis
        out = np.hypot(np.minimum(ex, 0), np.minimum(ey, 0))
        return np.where((ex >= 0) & (ey >= 0), np.minimum(ex, ey), -out)
    sd = np.maximum(sd, box_sdf(pw - X, pt - Y))                    # left pad
    sd = np.maximum(sd, box_sdf(X - (Lx - pw), pt - Y))             # right pad
    for (cx, cy, ri, ro) in cfg.get("liners", []):
        r = np.hypot(X - cx, Y - cy)
        sd = np.maximum(sd, np.minimum(r - ri, ro - r))
    sdv = np.full_like(X, -np.inf)
    for (cx, cy, r0) in cfg.get("ducts", []):
        sdv = np.maximum(sdv, r0 - np.hypot(X - cx, Y - cy))
    return sd, sdv


# --------------------------------------------------------------------------- sequential loops
# Both 808 loops are Gauss-Seidel: each update reads values written earlier in
# the same sweep, so they cannot be vectorised without changing the result.
# They are written as scalar Python over precomputed neighbour lists, which
# runs them in ~0.1 s per call, in float64, with no JAX/GPU dependency.

def _filter_frame(nx, ny, act, nb, order, max_it=FILTER_MAX_IT):
    """808 filterVectorField (l. 658-688), one layer. nx, ny (NE,) unit
    orientation; act (NE,) bool; nb (NE, 9) 3x3 neighbourhood with pad index
    NE; order: sweep order. Candidate score of c in {n, -n, R90 n, -R90 n}:
    mean over fixed neighbours of (c.n_m + 1)/2 = (A + c.S)/(2 nrm), with A
    the number of fixed neighbours and S the sum of their orientations."""
    nxp = np.append(nx, 0.0); nyp = np.append(ny, 0.0)
    fdot0 = np.mean(nxp[:NE, None] * nxp[nb] + nyp[:NE, None] * nyp[nb], axis=1)
    fixed = np.append((fdot0 > 0.975) & act, False)
    nfix0 = int(fixed.sum())
    X = nxp.tolist(); Y = nyp.tolist()
    nbl = nb.tolist(); order = [int(k) for k in order]
    it = 0
    while it < max_it:
        fs = fixed.tolist()                                   # activep: fixed at sweep start
        nrm = np.maximum(fixed[nb].sum(1), 1).tolist()
        fixed_new = fs[:]
        ch = False
        for k in order:
            if fs[k]:
                continue
            A = 0; Sx = 0.0; Sy = 0.0
            for m in nbl[k]:
                if fs[m]:
                    A += 1; Sx += X[m]; Sy += Y[m]
            cx, cy = X[k], Y[k]
            d0 = cx * Sx + cy * Sy                            # candidate n
            d2 = -cy * Sx + cx * Sy                           # candidate R90 n = (-ny, nx)
            v = (A + d0, A - d0, A + d2, A - d2)
            u = v.index(max(v))                               # first max (MATLAB find(..,1,'first'))
            fd = v[u] / (2 * nrm[k])
            if u == 1:
                X[k], Y[k] = -cx, -cy
            elif u == 2:
                X[k], Y[k] = -cy, cx
            elif u == 3:
                X[k], Y[k] = cy, -cx
            nf = fd > 0.975
            fixed_new[k] = nf
            ch = ch or u != 0 or nf
        fixed = np.array(fixed_new)
        it += 1
        # 808: continue while it < 50 and sum(fixed) > nfix0. We also stop once a
        # sweep changes nothing (the following sweeps would be identical).
        if int(fixed.sum()) == nfix0 or not ch:
            break
    return np.array(X[:NE]), np.array(Y[:NE]), it


def _align_loop(prep, first_visited, n_iter=ALIGN_ITERS):
    """808 phaseAlignment inner loop (l. 720-733) on unit complex phases
    u = exp(i phi): a neighbour with sign +1 contributes W u_f, sign -1
    contributes W exp(i(pi - phi_f)) = -W conj(u_f), sign 0 contributes W i.
    Sweep 0 visits order0, sweeps 1.. visit order. If first_visited, sweep 0
    only uses neighbours already updated in that sweep (a kernel with none
    keeps its initial phase)."""
    n = prep["n_act"]
    if n == 0:
        return np.zeros(NE)
    W = (prep["wr"] + 1j * prep["wi"])
    sg, nbr = prep["sg"], prep["nbr"]
    P, M, Z = {}, {}, {}
    act = prep["order"][:n]
    for j in act.tolist():
        ok = nbr[j] < NE
        f, wj, s = nbr[j][ok], W[j][ok], sg[j][ok]
        P[j] = list(zip(f[s > 0].tolist(), wj[s > 0].tolist()))
        M[j] = list(zip(f[s < 0].tolist(), wj[s < 0].tolist()))
        Z[j] = list(zip(f[s == 0].tolist(), (1j * wj[s == 0]).tolist()))
    u = np.exp(1j * prep["phi0"].astype(float)).tolist()
    vis = [False] * (NE + 1)
    for sweep in range(n_iter):
        order = (prep["order0"] if sweep == 0 else prep["order"])[:n].tolist()
        masked = first_visited and sweep == 0
        for j in order:
            z = 0j
            if masked:
                for f, w in P[j]:
                    if vis[f]:
                        z += w * u[f]
                for f, w in M[j]:
                    if vis[f]:
                        z -= w * u[f].conjugate()
                for f, w in Z[j]:
                    if vis[f]:
                        z += w
                if z != 0:
                    u[j] = z / abs(z)
                vis[j] = True
            else:
                for f, w in P[j]:
                    z += w * u[f]
                for f, w in M[j]:
                    z -= w * u[f].conjugate()
                for f, w in Z[j]:
                    z += w
                u[j] = z / abs(z) if z != 0 else 1.0 + 0j
    return np.angle(np.array(u[:NE]))


# --------------------------------------------------------------------------- pipeline pieces
def _box3_replicate(a):
    """808 getIndicator: 3x3 mean with replicate padding."""
    p = np.pad(a.astype(float), 1, mode="edge")
    return sum(p[dy:dy + a.shape[0], dx:dx + a.shape[1]] for dy in range(3) for dx in range(3)) / 9.0


def _interp(mats, F):
    Ay, Ax = mats
    return Ay @ F @ Ax.T


def _clean_frame(n1x, n1y, act_union, g):
    ox, oy, it = _filter_frame(n1x.ravel(), n1y.ravel(), act_union.ravel(), g["nb3"], g["mat_order"])
    return ox.reshape(NYC, NXC), oy.reshape(NYC, NXC), it


DEFAULT_PHASE = 0.3
import os as _os
ORDER_MODE = _os.environ.get("DEHOM_ORDER", "width")
PHASE_MODE = _os.environ.get("DEHOM_PHASE_MODE", "full")


def _align_prep(nx, ny, act, w, g, rng, first_sweep="flood"):
    """Precompute 808 phaseAlignment weights (l. 693-717) and the update
    order for one layer."""
    nxf, nyf = nx.ravel(), ny.ravel()
    actf = act.ravel()
    nb = g["nb5"]
    valid_win = nb < NE
    nbc = np.where(valid_win, nb, 0)
    dx, dy = g["d5x"], g["d5y"]
    cx, cy = nxf[:, None], nyf[:, None]                        # centre orientation
    idn = (RRX * (dx * cy - dy * cx) ** 2 + RRY * (dx * cx + dy * cy) ** 2 <= R_ALIGN) \
        & valid_win & actf[nbc]
    fx, fy = nxf[nbc], nyf[nbc]                                # neighbour orientation
    sgn = np.where(idn, np.sign(fx * cx + fy * cy), 0.0)
    gd = np.exp(-(dx ** 2 + dy ** 2) / R_ALIGN)
    ph = sgn * 2 * np.pi * OMEGA * (fx * dx + fy * dy)
    wr = np.where(idn, gd * np.cos(ph), 0.0)
    wi = np.where(idn, gd * np.sin(ph), 0.0)
    nbr = np.where(idn, nb, NE).astype(np.int32)

    aind = np.flatnonzero(actf)
    if rng is None:
        tie = g["mat_rank"][aind]
        # Orchestrator fix: a non-zero deterministic global phase. With 0, axis-aligned
        # bars whose 15*w is an integer put both edges on pixel centres and lose a
        # row (w = 0.2 bars came out 15 % thin, builder report). 0.3 rad shifts edges
        # 0.43 px off the pixel grid.
        phi0_val = DEFAULT_PHASE
    elif PHASE_MODE == "global":
        # Addendum B: translation-only realisation — fixed raster tie order, random global phase
        # per layer (the alignment then shifts every phase by the same amount: a pure translation).
        tie = g["mat_rank"][aind]
        phi0_val = rng.uniform(-np.pi, np.pi)
    else:
        tie = rng.permutation(aind.size)
        phi0_val = rng.uniform(-np.pi, np.pi)
    if ORDER_MODE == "fixed":
        # Trial 14: fixed raster order, independent of the widths. Width-descending order
        # makes the alignment sweep order (and so the phase field) jump under 1e-12
        # perturbations of the design (scratch/chaos.py).
        order = aind[np.argsort(tie, kind="stable")]
    else:
        order = aind[np.lexsort((tie, -w.ravel()[aind]))]      # width descending, then tie
    order0 = _flood_order(order, idn, nb) if first_sweep == "flood" else order
    full = lambda o: np.concatenate([o, np.zeros(NE - o.size, int)]).astype(np.int32)
    phi0 = np.zeros(NE + 1, np.float32); phi0[:NE] = phi0_val
    return dict(wr=wr, wi=wi, sg=sgn, nbr=nbr, order0=full(order0), order=full(order),
                n_act=order.size, phi0=phi0)


def _flood_order(order, idn, nb):
    """Priority flood: start at order[0]; repeatedly take the highest-priority
    (position in `order`) kernel adjacent to the visited set; restart at the
    best unvisited kernel when a component is exhausted."""
    import heapq
    rank = np.full(NE, -1); rank[order] = np.arange(order.size)
    adj = [[] for _ in range(NE)]
    jj, mm = np.nonzero(idn)
    ff = nb[jj, mm]
    for a, b in zip(jj.tolist(), ff.tolist()):
        if a != b:
            adj[a].append(b); adj[b].append(a)
    seen = np.zeros(NE, bool); out = []; heap = []; nxt = 0
    rk = rank.tolist()
    while len(out) < order.size:
        if not heap:
            while seen[order[nxt]]:
                nxt += 1
            heapq.heappush(heap, rk[order[nxt]]); seen[order[nxt]] = True
        r = heapq.heappop(heap); j = order[r]; out.append(j)
        for f in adj[j]:
            if rk[f] >= 0 and not seen[f]:
                seen[f] = True; heapq.heappush(heap, rk[f])
    return np.array(out)


def _align(preps, first_sweep="flood"):
    return [_align_loop(p, first_sweep != "808") for p in preps]


def _sample_layer(nx, ny, act, pshift, inx, iny, unsampled, g):
    """808 phasorSampling (l. 737-760) for one layer on T_i, vectorised over
    all active kernels and their stencil."""
    G = np.where(unsampled, UNSAMPLED, 0.0 + 0.0j).ravel()
    aind = np.flatnonzero(act.ravel())
    if aind.size == 0:
        return G.reshape(NYI, NXI)
    jx = g["st_jx"][aind]; jy = g["st_jy"][aind]              # (A,1,n), (A,n,1)
    Xc, Yc = g["Xc"].ravel()[aind], g["Yc"].ravel()[aind]
    dxs = (jx + 0.5) * HI - Xc[:, None, None]                  # sample - centre
    dys = (jy + 0.5) * HI - Yc[:, None, None]
    ex, ey = nx.ravel()[aind][:, None, None], ny.ravel()[aind][:, None, None]
    xdv, ydv = dxs * ex, dys * ey
    s = RX * (dxs * ey - dys * ex) ** 2 + RY * (xdv + ydv) ** 2
    ok = (s < CUTOFF) & (jx >= 0) & (jx < NXI) & (jy >= 0) & (jy < NYI)
    ka, ya, xa = np.nonzero(ok)
    jxs = jx[ka, 0, xa]; jys = jy[ka, ya, 0]
    q = jys * NXI + jxs
    dxv, dyv = dxs[ka, 0, xa], dys[ka, ya, 0]
    exv, eyv = ex[ka, 0, 0], ey[ka, 0, 0]
    sv = s[ka, ya, xa]
    djx = OMEGA * (exv - inx.ravel()[q])
    djy = OMEGA * (eyv - iny.ravel()[q])
    mag = np.exp(-sv * BABPAINV - np.pi ** 2 * (djx ** 2 + djy ** 2) * BPAINV)
    arg = 2 * (np.pi * OMEGA * (dxv * exv + dyv * eyv) + (djx * dxv + djy * dyv) * ABPAINV) \
        + pshift.ravel()[aind][ka]
    n = NXI * NYI
    G = G + np.bincount(q, mag * np.cos(arg), minlength=n) + 1j * np.bincount(q, mag * np.sin(arg), minlength=n)
    return G.reshape(NYI, NXI)


# --------------------------------------------------------------------------- main entry
def dehomogenise(w1, w2, theta, phase_seed=None, *, apply_masks=True, return_fields=False,
                 first_sweep="flood", faithful=False):
    """De-homogenise a coarse rank-2 design into a fine binary field.

    Parameters
    ----------
    w1, w2 : (20, 40) arrays in [0, 1]. Realised width (fraction of lambda)
        of bar 1 (runs along theta) and bar 2 (runs along theta + pi/2).
    theta : (20, 40) array, bar-1 direction (rad, CCW from +x, any branch).
    phase_seed : None for the deterministic default. An int randomises the
        global initial phase of each layer and the alignment tie-break order,
        giving a different but equally valid realisation.
    apply_masks : keyword-only, default True. False skips the non-design
        masks (used by the method tests on uniform fields).
    return_fields : keyword-only. True also returns the intermediate fields
        (tau, w_f per layer, cleaned orientations) for diagnostics.
    first_sweep : keyword-only, "flood" (default), "visited" or "808". The
        first phase-alignment sweep; see module docstring, choice 3.

    Returns
    -------
    dict with
      phi   : (200, 400) bool, solid = True, masks applied, before island removal.
      level : (200, 400) float, continuous, positive = solid; phi == (level >= 0).
              Layer part max_k (tau_k - (1 - w_f,k)), cut where w_f,k < W_VOID;
              solid part max_k w_f,k - 0.99; masks as signed distances x 2/lambda.
      timing: dict of stage times (s).
      info  : kernel counts, swaps, filter sweeps.
    """
    T0 = time.perf_counter(); tm = {}
    g = _grid()
    w = np.stack([np.asarray(w1, float), np.asarray(w2, float)]).copy()
    th = np.asarray(theta, float)
    assert w.shape == (2, NYC, NXC) and th.shape == (NYC, NXC), "coarse fields must be (20, 40)"
    if not (np.isfinite(w).all() and np.isfinite(th).all()):
        raise ValueError("non-finite input")
    w = np.clip(w, 0.0, 1.0)
    if apply_masks:
        w[:, g["solid_c"]] = 1.0
        w[:, g["void_c"]] = 0.0
    w[w >= W_SOLID] = 1.0                                       # choice 5
    w[w < W_VOID] = 0.0
    wmin = W_VOID

    # --- orientations (808 l. 120) and frame clean-up (choice 2)
    n1x, n1y = np.cos(th + np.pi / 2), np.sin(th + np.pi / 2)
    act = (w >= wmin * 0.95) & (w < 1 - 1e-3)                  # 808 l. 359
    if faithful:
        # Trial 14 "808-faithful" mode: each layer's normal field cleaned on its own
        # (808 l. 663-687), no width swap, plain first alignment sweep.
        N = np.empty((2, 2, NYC, NXC)); f_it = 0
        for k, (nx0, ny0) in enumerate(((n1x, n1y), (n1y, -n1x))):
            ox, oy, it_k = _filter_frame(nx0.ravel(), ny0.ravel(), act[k].ravel(), g["nb3"], g["mat_order"])
            nr = np.hypot(ox, oy)
            N[k, 0], N[k, 1] = (ox / nr).reshape(NYC, NXC), (oy / nr).reshape(NYC, NXC)
            f_it = max(f_it, it_k)
        swap = np.zeros((NYC, NXC), bool)
        first_sweep = "808"
    else:
        c1x, c1y, f_it = _clean_frame(n1x, n1y, act[0] | act[1], g)
        nrm = np.hypot(c1x, c1y); c1x /= nrm; c1y /= nrm
        swap = np.abs(c1x * n1x + c1y * n1y) < 0.5             # a +-90 deg layer jump was chosen
        if swap.any():
            w[:, swap] = w[::-1][:, swap]
            act = (w >= wmin * 0.95) & (w < 1 - 1e-3)
        N = np.stack([np.stack([c1x, c1y]), np.stack([c1y, -c1x])])   # (layer, comp, NYC, NXC)
    tm["filter"] = time.perf_counter() - T0

    # --- indicators, orientation and thickness interpolation (808 l. 612-656)
    t = time.perf_counter()
    iny_nn, inx_nn = g["c2i_nn"]
    layers = []
    for k in range(2):
        ind_c = _box3_replicate(w[k] >= wmin * 0.95)
        ind_i = _interp(g["c2i"], ind_c)
        inx = _interp(g["c2i"], N[k, 0]); iny = _interp(g["c2i"], N[k, 1])
        bad = inx ** 2 + iny ** 2 <= 0.975
        if bad.any():
            by, bx = np.nonzero(bad)
            inx[bad] = N[k, 0][iny_nn[by], inx_nn[bx]]
            iny[bad] = N[k, 1][iny_nn[by], inx_nn[bx]]
        wc = (ind_c >= 0.25) * np.maximum(w[k], wmin)
        w_i = _interp(g["c2i"], wc)
        w_f = np.clip(_interp(g["c2f"], wc), 0.0, 1.0)
        region = ((w_i < 1 - 1e-3) * ind_i) >= 0.01            # 808 l. 368-370
        layers.append(dict(inx=inx, iny=iny, w_f=w_f, region=region))
    tm["interp"] = time.perf_counter() - t

    # --- phase alignment (808 l. 690-735)
    t = time.perf_counter()
    rng = None if phase_seed is None else np.random.default_rng(phase_seed)
    psh = _align([_align_prep(N[k, 0], N[k, 1], act[k], w[k], g, rng, first_sweep) for k in range(2)],
                 first_sweep)
    tm["align"] = time.perf_counter() - t

    # --- sampling on T_i, cubic to fine grid, sawtooth -> triangle, threshold
    t = time.perf_counter()
    level = np.full((NYF, NXF), -np.inf)
    wfmax = np.zeros((NYF, NXF))
    taus = []
    for k in range(2):
        L = layers[k]
        Gi = _sample_layer(N[k, 0], N[k, 1], act[k], psh[k].reshape(NYC, NXC),
                           L["inx"], L["iny"], ~L["region"], g)
        Gf = _interp(g["i2f"], Gi)
        tau = np.arcsin(np.sin(np.angle(Gf))) / np.pi + 0.5      # 808 l. 402-404, 390
        lk = np.minimum(tau - (1.0 - L["w_f"]), L["w_f"] - W_VOID)   # choice 5 cut
        level = np.maximum(level, lk)
        wfmax = np.maximum(wfmax, L["w_f"])
        taus.append(tau)
    level = np.maximum(level, wfmax - W_SOLID)                  # 808 l. 388 solid part
    tm["sample"] = time.perf_counter() - t

    if apply_masks:
        gs = 2.0 / LAMBDA
        level = np.maximum(level, g["sd_solid"] * gs)
        level = np.minimum(level, -g["sd_void"] * gs)
    phi = level >= 0
    if apply_masks:
        phi[g["solid_f"]] = True
        phi[g["void_f"]] = False
    tm["total"] = time.perf_counter() - T0
    out = dict(phi=phi, level=level, timing=tm,
               info=dict(n_active=[int(a.sum()) for a in act], n_swapped=int(swap.sum()),
                         filter_sweeps=f_it))
    if return_fields:
        out.update(tau=taus, w_f=[L["w_f"] for L in layers], N=N, pshift=psh, w_coarse=w)
    return out
