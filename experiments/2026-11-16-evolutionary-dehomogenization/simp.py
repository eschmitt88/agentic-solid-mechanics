"""SIMP baseline (paper Dsg.4) on the FINE HF grid, trial 13.

Andreassen et al. (2011) 88-line style: minimum compliance, modified SIMP
E = Emin + x^p (E0 - Emin) with p = 3, Emin = 1e-9, density filter (linear hat,
ft = 2), optimality-criteria update (move 0.2, eta = 1/2). Same problem as hf.py:
400 x 200 Q4 pixels (h = 0.25 mm), hf.Problem's non-design masks (solid_fixed
held at 1, void_fixed at 0, both enter the filter at those values), the same
distributed bearing springs and top-edge load. The volume fraction is that of
the design region.

Our choices (the paper does not give its SIMP settings): filter radius
rmin = 1.5 mm (6 px); stop at max |dx| < 0.01 or 300 iterations; the grey
result is thresholded to binary by keeping the k = round(vf * n_design)
highest-density design pixels (volume preserved exactly before island removal).

Run `python simp.py` -> results/simp/simp_designs.npz (phi (6, 200, 400) bool,
vf, plus x_grey) and results/simp/simp_eval.json (hf.Problem.evaluate with
buckling=True on each binary design).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy import ndimage

import hf

HERE = Path(__file__).resolve().parent
VFS = (0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
PENAL = 3.0
EMIN = 1e-9
RMIN_MM = 1.5


def simp_optimise(vf, rmin_mm=RMIN_MM, penal=PENAL, max_iter=300, tol=0.01, move=0.2, verbose=False):
    import pypardiso
    t0 = time.time()
    p = hf.Problem(hf.GIRDER, hf.H_FINE)
    shp = (p.nely, p.nelx)
    design = p.design; solid = p.solid_fixed
    R = rmin_mm / p.h
    k = int(np.ceil(R)) - 1
    yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
    ker = np.maximum(0.0, R - np.hypot(xx, yy))
    filt = lambda a: ndimage.convolve(a, ker, mode="constant", cval=0.0)
    Hs = filt(np.ones(shp))
    x = np.where(solid, 1.0, 0.0); x[design] = vf
    pad = p.pad_mask
    pad_E = p.cfg.get("pad_E", 1.0)
    hist = []
    solver = pypardiso.PyPardisoSolver(); solver.set_matrix_type(2)
    for it in range(1, max_iter + 1):
        xPhys = filt(x) / Hs
        xPhys[solid] = 1.0; xPhys[p.void_fixed] = 0.0
        Ee = EMIN + xPhys.ravel() ** penal * (1 - EMIN)
        Ee[pad.ravel()] = pad_E
        K = p.assemble(Ee)
        Kup = sp.triu(K, format="csr"); Kup.sort_indices()
        u = solver.solve(Kup, p.F)
        ue = u[p.edof]
        ce = np.einsum("ea,ab,eb->e", ue, p.KE, ue).reshape(shp)
        c = float(p.F @ u)
        dc = -penal * (1 - EMIN) * xPhys ** (penal - 1) * ce
        dc[~design] = 0.0
        dv = np.where(design, 1.0, 0.0)
        dc = filt(dc / Hs); dv = filt(dv / Hs)
        # OC with bisection on the design-region volume of xPhys
        l1, l2 = 0.0, 1e9
        xd = x[design]; dcd = np.minimum(dc[design], -1e-30); dvd = np.maximum(dv[design], 1e-30)
        while (l2 - l1) / (l1 + l2) > 1e-4:
            lm = 0.5 * (l1 + l2)
            xn = np.clip(xd * np.sqrt(-dcd / dvd / lm), np.maximum(0, xd - move), np.minimum(1, xd + move))
            xt = x.copy(); xt[design] = xn
            xP = filt(xt) / Hs; xP[solid] = 1.0; xP[p.void_fixed] = 0.0
            if xP[design].mean() > vf:
                l1 = lm
            else:
                l2 = lm
        change = float(np.abs(xn - xd).max())
        x[design] = xn
        hist.append(dict(it=it, c=c, vol=float(xP[design].mean()), change=change))
        if verbose and it % 20 == 0:
            print(f"vf {vf} it {it} c {c:.4f} vol {xP[design].mean():.4f} change {change:.4f}", flush=True)
        if change < tol:
            break
    solver.free_memory(everything=True)
    xPhys = filt(x) / Hs; xPhys[solid] = 1.0; xPhys[p.void_fixed] = 0.0
    # volume-preserving threshold on the design region
    nd = int(design.sum()); kk = int(round(vf * nd))
    vals = xPhys[design]
    order = np.argsort(-vals, kind="stable")
    keep = np.zeros(nd, bool); keep[order[:kk]] = True
    phi = solid.copy(); phi[design] = keep
    grey = float(np.mean((vals > 0.05) & (vals < 0.95)))
    return dict(phi=phi, x=xPhys, compliance_grey=hist[-1]["c"], n_iter=len(hist), history=hist,
                wall_s=time.time() - t0, threshold=float(vals[order[kk - 1]]), greyness=grey)


def _run(vf):
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    r = simp_optimise(vf, verbose=True)
    p = hf.Problem(hf.GIRDER, hf.H_FINE)
    t = time.time()
    ev = p.evaluate(r["phi"], buckling=True)
    t_ev = time.time() - t
    ev = {k: v for k, v in ev.items() if k not in ("u", "s")}
    return r, ev, t_ev


def main(workers=6):
    from concurrent.futures import ProcessPoolExecutor
    out = HERE / "results" / "simp"; out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_run, VFS))
    np.savez_compressed(out / "simp_designs.npz", phi=np.stack([r["phi"] for r, _, _ in res]).astype(bool),
                        vf=np.array(VFS), x_grey=np.stack([r["x"] for r, _, _ in res]).astype(np.float32))
    js = dict(settings=dict(penal=PENAL, Emin=EMIN, rmin_mm=RMIN_MM, filter="density (ft=2), linear hat",
                            update="OC move 0.2", stop="max|dx|<0.01 or 300 it", grid=[400, 200],
                            threshold="volume-preserving top-k on design region"),
              designs=[])
    for vf, (r, ev, t_ev) in zip(VFS, res):
        js["designs"].append(dict(vf_target=vf, **{k: (float(v) if np.isscalar(v) else v) for k, v in ev.items()},
                                  compliance_grey_simp=r["compliance_grey"], n_iter=r["n_iter"],
                                  greyness_before_threshold=r["greyness"], threshold=r["threshold"],
                                  wall_opt_s=r["wall_s"], wall_eval_s=t_ev))
    js["wall_total_s"] = time.time() - t0
    with open(out / "simp_eval.json", "w") as f:
        json.dump(js, f, indent=1)
    return js


if __name__ == "__main__":
    js = main()
    for d in js["designs"]:
        print(f"vf {d['vf_target']:.2f}: vf_eval {d['vf']:.4f} conn {d['connected']} C {d['compliance']:.4f} "
              f"(grey {d['compliance_grey_simp']:.4f}) umax {d['umax']:.4f} BLF {d['blf']:.5f} it {d['n_iter']} "
              f"grey {d['greyness_before_threshold']:.3f} t {d['wall_opt_s']:.0f}+{d['wall_eval_s']:.0f}s")
    print("total", js["wall_total_s"])
