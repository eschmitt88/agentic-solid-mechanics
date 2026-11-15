"""Relaxed-idealisation re-verification: the same design re-solved in CalculiX with one shared assumption dropped.

  ps  : plane strain, rigid frame (reference; same in-plane mesh size t_min/2.5 as the relaxed runs)
  3d  : 10 mm deep 3D body, free side faces (C3D15 wedges, 4 layers over the half depth)  -> drops plane strain
  ef  : wall posts and shuttle meshed in the same elastomer (posts held on the lattice symmetry plane and at the
        foot, shuttle pushed on its top face)                                               -> drops the rigid frame
usage: python verify_relaxed.py curve <target> <name> <design.json> | bistable <name> <design.json>
"""
import json, sys, time
import numpy as np
import ccx, design, snap
from verify import bfeat

kind = sys.argv[1]
if kind == "curve":
    tname, name, src = sys.argv[2:5]
    sp = design.SPEC[tname]
    nk, ncell, grid, ninc, kpa = sp["nk"], sp["ncell"], design.travel_grid(sp["ncell"]), 100, design.KPA
    tgt = design.TARGETS[tname](design.EPS)
    rng = tgt.max() - tgt.min()
else:
    name, src = sys.argv[2:4]
    nk, ncell, grid, ninc, kpa = 4, 1, snap.DS, 104, 1.0
p = np.array(json.load(open(src))["p"])
out = dict(name=name, kind=kind, p=p.tolist(), grid=grid.tolist(), depth_mm=10.0)
variants = [("ps", {}), ("3d", dict(depth3d=10.0, layers=4))] + ([("ef", dict(elastic_frame=True))] if ncell == 1 else [])
for key, kw in variants:
    t0 = time.time()
    try:
        d, F, info = ccx.run(f"relax_{name}_{key}", p, nk, ncell=ncell, dmax=float(grid[-1]), ninc=ninc, lc_div=2.5, threads=2, **kw)
        y = ccx.sample(d, F, grid, info["completed"]) * kpa
        out[key] = dict(y=[None if not np.isfinite(v) else float(v) for v in y], completed=info["completed"], d_reached=info["d_reached"], nnodes=info["nnodes"], sec=time.time() - t0)
    except Exception as e:
        out[key] = dict(y=None, completed=False, error=repr(e))
    print(name, key, out[key].get("completed"), out[key].get("d_reached"), f"{time.time()-t0:.0f}s", flush=True)
ref = np.array([np.nan if v is None else v for v in out["ps"]["y"]])
for key, _ in variants:
    r = out[key]
    if r["y"] is None:
        continue
    y = np.array([np.nan if v is None else v for v in r["y"]])
    m = np.isfinite(y) & np.isfinite(ref)
    if kind == "curve":
        r["grade"] = design.grade(tname, y)
        if key != "ps" and m.any():
            r["shift_nrmse_vs_plane_strain"] = float(np.sqrt(np.mean((y[m] - ref[m]) ** 2)) / rng)
            big = m & (np.abs(ref) > 0.25 * np.nanmax(np.abs(ref)))
            r["median_force_ratio"] = float(np.median(y[big] / ref[big]))
    else:
        r["features"] = bfeat(r["y"])
        if key != "ps" and r["features"] and out["ps"].get("features"):
            a, b = r["features"], out["ps"]["features"]
            r["feature_change_vs_plane_strain"] = {k: a[k] / b[k] - 1 for k in ("F_pk", "d_2", "E_back")}
    print(name, key, {k: v for k, v in r.items() if k not in ("y",)}, flush=True)
json.dump(out, open(f"results/relaxed_{name}.json", "w"), indent=1)
