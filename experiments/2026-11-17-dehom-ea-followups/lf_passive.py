"""LF optimiser with passive solid / void boxes (problem-grain pilot, DESIGN.md C).

`lf_optimise_passive` is lf.lf_optimise (same formulation, same MMA, same defaults) plus one
knob: up to a few axis-aligned rectangles in mm, each forced SOLID or VOID on the coarse grid.
With no boxes it delegates to lf.lf_optimise, so the no-box path is bit-identical to gen 0.

Passive elements are imposed through the variable bounds (lf.py is not edited):
  void box  : rho in [0, 1e-3]                (starts at 0)
  solid box : rho, mu1, mu2 in [1 - 1e-3, 1]   (start at 1)
A gap of 1e-3 (not lb = ub) because the MMA subproblem divides by (x - alfa) and (beta - x).
The density filter still blurs rho across the box edge by rmin, so a box acts on its
interior; boxes overlapping non-design elements only touch their design elements.
Element membership = coarse element centre inside the box. Boxes never change JIT shapes
(only bound arrays), so they add no compilation.
"""
from __future__ import annotations

import time

import numpy as np
import jax.numpy as jnp

import lf

BOX_GAP = 1e-3


def box_masks(boxes, cm=None):
    """boxes: list of dict(kind='solid'|'void', x0, y0, x1, y1) in mm (y up).
    -> (solid, void) bool arrays over design elements (cm.nd,)."""
    cm = cm or lf.coarse()
    hc = lf.hf.HC
    iy, ix = np.divmod(cm.di, cm.nelx)
    xc, yc = (ix + 0.5) * hc, (iy + 0.5) * hc
    solid = np.zeros(cm.nd, bool); void = np.zeros(cm.nd, bool)
    for b in boxes or []:
        inside = (xc >= b["x0"]) & (xc <= b["x1"]) & (yc >= b["y0"]) & (yc <= b["y1"])
        (solid if b["kind"] == "solid" else void)[inside] = True
    void &= ~solid                                         # overlap: solid wins
    return solid, void


def lf_optimise_passive(V0, Lmin, boxes=None, seed=0, rmin=None, theta_jitter=0.0, max_iter=300,
                        beta_max=16.0, beta_every=40, tol=1e-3, move=lf.MOVE, gamma_s=lf.GAMMA_S,
                        init_jitter=0.0, projection="three_field", asyincr=None, rmin_mu=None):
    """lf.lf_optimise with passive boxes (see module docstring). Same return dict, plus
    params['boxes'] and n_solid_box / n_void_box (design elements forced)."""
    if not boxes:
        r = lf.lf_optimise(V0, Lmin, seed=seed, rmin=rmin, theta_jitter=theta_jitter, max_iter=max_iter,
                           beta_max=beta_max, beta_every=beta_every, tol=tol, move=move, gamma_s=gamma_s,
                           init_jitter=init_jitter, projection=projection, asyincr=asyincr, rmin_mu=rmin_mu)
        r["params"]["boxes"] = []
        r["n_solid_box"] = r["n_void_box"] = 0
        return r
    # ---- copy of lf.lf_optimise with the bound hook (marked PASSIVE) ----
    t0 = time.time()
    rmin = 2.0 if rmin is None else float(rmin)
    cm, (fwd, bwd) = lf._setup(rmin, rmin_mu)
    J_solid, a0, _ = lf._solid_state()
    rng = np.random.default_rng(seed)
    nd = cm.nd; di = cm.di
    mu0 = float(np.clip(1 - np.sqrt(1 - V0), Lmin, 1.0))
    mu1 = np.full(nd, mu0); mu2 = np.full(nd, mu0)
    if init_jitter > 0:
        mu1 = np.clip(mu1 + init_jitter * rng.standard_normal(nd), Lmin, 1)
        mu2 = np.clip(mu2 + init_jitter * rng.standard_normal(nd), Lmin, 1)
    th0 = a0[di] + (theta_jitter * rng.standard_normal(nd) if theta_jitter > 0 else 0.0)
    th0 = np.clip(th0, -lf.THETA_BOUND, lf.THETA_BOUND)
    x = np.stack([mu1, mu2, th0, np.ones(nd)])
    lb = np.stack([np.full(nd, Lmin), np.full(nd, Lmin), np.full(nd, -lf.THETA_BOUND), np.zeros(nd)])
    ub = np.stack([np.ones(nd), np.ones(nd), np.full(nd, lf.THETA_BOUND), np.ones(nd)])
    # PASSIVE
    sb, vb = box_masks(boxes, cm)
    lb[3, vb] = 0.0; ub[3, vb] = BOX_GAP; x[3, vb] = 0.0
    for k in (0, 1, 3):
        lb[k, sb] = 1.0 - BOX_GAP; ub[k, sb] = 1.0; x[k, sb] = 1.0
    # /PASSIVE
    mv = np.asarray(move, float)[:, None] * np.ones((1, nd))
    theta_fill = jnp.asarray(a0)
    gamma0 = V0 / J_solid
    if asyincr is None:
        asyincr = 1.05 if projection == "three_field" else 1.2
    mma = lf.MMA(4 * nd, 1, asyincr=asyincr)
    etas = jnp.asarray((0.55, 0.45) if projection == "robust" else (lf.ETA, lf.ETA))
    V0d = V0
    beta = 1.0; stage_it = 0
    hist = []
    for it in range(1, max_iter + 1):
        stage_it += 1
        xj = jnp.asarray(x)
        C, V, S, Vb, *_ = fwd(xj, beta, theta_fill, etas)
        if projection == "robust" and it % 20 == 0:
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
        hist.append(dict(it=it, beta=beta, J=J, obj=obj, V=float(V), Vb=float(Vb), S=float(S), g=g, change=change))
        x = xn
        if beta >= beta_max and change < tol and stage_it > 5:
            break
        if beta < beta_max and (stage_it >= beta_every or (change < tol and stage_it > 5)):
            beta = min(2 * beta, beta_max); stage_it = 0; mma.restart()
    xj = jnp.asarray(x)
    _, _, S, V, w1, w2, theta, rb, rt, mu1p, mu2p = fwd(xj, beta, theta_fill, etas)
    J = lf.lf_compliance(np.asarray(w1), np.asarray(w2), np.asarray(theta))
    shp = cm.shape
    mu1f = np.where(cm.solid, 1.0, 0.0); mu1f[di] = np.asarray(mu1p)
    mu2f = np.where(cm.solid, 1.0, 0.0); mu2f[di] = np.asarray(mu2p)
    rhof = np.where(cm.solid, 1.0, 0.0); rhof[di] = np.asarray(rb)
    return dict(w1=np.asarray(w1).reshape(shp), w2=np.asarray(w2).reshape(shp),
                theta=np.asarray(theta).reshape(shp), mu1=mu1f.reshape(shp), mu2=mu2f.reshape(shp),
                rho=rhof.reshape(shp), compliance_lf=J, vol_lf=float(V), S=float(S), beta=beta,
                history=hist, n_iter=len(hist), wall_s=time.time() - t0,
                n_solid_box=int(sb.sum()), n_void_box=int(vb.sum()),
                params=dict(V0=V0, Lmin=Lmin, seed=seed, rmin=rmin, theta_jitter=theta_jitter,
                            init_jitter=init_jitter, max_iter=max_iter, beta_max=beta_max,
                            projection=projection, asyincr=asyincr, rmin_mu=rmin_mu, boxes=list(boxes)))
