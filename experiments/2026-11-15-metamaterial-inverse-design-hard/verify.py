"""Independent verification with the pre-registered mesh ladder (t_min/3, /4, /6), graded on the finest that completes.

usage: python verify.py curve <target> <name> <design.json>      (design.json has "p")
       python verify.py bistable <name> <design.json>
"""
import json, sys, time
import numpy as np
import ccx, design, fem, snap

LADDER = (3.0, 4.0, 6.0)


def ladder(name, p, nk, ncell, dmax, ninc, grid, threads=2, extra=()):
    runs = {}
    for div in LADDER + tuple(extra):
        t0 = time.time()
        d, F, info = ccx.run(f"{name}_t{div:g}", p, nk, ncell=ncell, dmax=dmax, ninc=ninc, lc_div=div, threads=threads)
        runs[f"t{div:g}"] = dict(F=[None if not np.isfinite(v) else float(v) for v in ccx.sample(d, F, grid, info["completed"])], completed=info["completed"],
                                 d_reached=info["d_reached"], nnodes=info["nnodes"], lc=info["lc"], sec=time.time() - t0, d_cellA=info.get("d_cellA"))
        print(name, div, "completed", info["completed"], "reached", info["d_reached"], f"{time.time()-t0:.0f}s", flush=True)
    done = [k for k in (f"t{d:g}" for d in LADDER) if runs[k]["completed"]]
    graded = done[-1] if done else None
    imp = {}
    if graded is None:  # pre-registered fallback: +-0.01 mm a_2 imperfection at t_min/4, graded on the worse
        for sgn in (+1, -1):
            d, F, info = ccx.run(f"{name}_t4_imp{'p' if sgn > 0 else 'm'}", p, nk, ncell=ncell, dmax=dmax, ninc=ninc, lc_div=4.0, imperfection=0.01 * sgn, threads=threads)
            imp["p" if sgn > 0 else "m"] = dict(F=[None if not np.isfinite(v) else float(v) for v in ccx.sample(d, F, grid, info["completed"])],
                                                completed=info["completed"], d_reached=info["d_reached"])
            print(name, "imperfection", sgn, info["completed"], flush=True)
    return runs, graded, done, imp


def verify_curve(tname, name, p):
    sp = design.SPEC[tname]
    grid = design.travel_grid(sp["ncell"])
    tgt = design.TARGETS[tname](design.EPS)
    rng = tgt.max() - tgt.min()
    out = dict(name=name, target=tname, p=np.asarray(p).tolist(), eps=design.EPS.tolist(), target_sigma=tgt.tolist())
    # stack: the monolithic 40x2 solve doubles as the cross-check of stack.py; the 80x2 monolithic solve (3161 dense dof) is skipped for cost
    for key, (nx, ny) in [("jax_40x2", (40, 2))] + ([("jax_80x2", (80, 2))] if sp["ncell"] == 1 else []):
        S = fem.Solver(nx=nx, ny=ny, Lh=design.LH, nk=sp["nk"], nm=sp["nm"], ncell=sp["ncell"])
        F, zs, infos = S.path(p, grid)
        out[key] = (F * design.KPA).tolist()
        out[key + "_grade"] = design.grade(tname, F * design.KPA)
        if key == "jax_40x2" and sp["ncell"] > 1:
            out["jax_d_cellA"] = [float(S.cell_travels(z, D)[0]) for z, D in zip(zs, grid)]
        print(name, key, out[key + "_grade"]["nrmse"], flush=True)
    runs, graded, done, imp = ladder(name, p, sp["nk"], sp["ncell"], float(grid[-1]), 100, grid)
    for k, r in runs.items():
        r["sigma"] = [None if v is None else v * design.KPA for v in r.pop("F")]
        r["grade"] = design.grade(tname, np.array([np.nan if v is None else v for v in r["sigma"]]))
    out.update(ccx=runs, graded_mesh=graded, completed_meshes=done)
    gap = lambda a, b: float(np.sqrt(np.mean((np.array(a) - np.array(b)) ** 2)) / rng)
    if graded:
        out["verdict"] = dict(runs[graded]["grade"], verdict="PASS" if runs[graded]["grade"]["passed"] else "MISS", mesh=graded)
        out["solver_gap_jax40x2_vs_ccx"] = gap(out["jax_40x2"], runs[graded]["sigma"])
        out["ladder_spread"] = max([gap(runs[a]["sigma"], runs[graded]["sigma"]) for a in done] + [0.0])
        out["ladder_spread_flag"] = bool(out["ladder_spread"] > 0.01)
    else:
        for k, r in imp.items():
            r["sigma"] = [None if v is None else v * design.KPA for v in r.pop("F")]
            r["grade"] = design.grade(tname, np.array([np.nan if v is None else v for v in r["sigma"]]))
        out["imperfection_runs"] = imp
        if imp and all(r["completed"] for r in imp.values()):
            worst = max(imp.values(), key=lambda r: r["grade"]["nrmse"])
            ok = all(r["grade"]["passed"] for r in imp.values())
            out["verdict"] = dict(worst["grade"], verdict="PASS" if ok else "MISS", mesh="t4 with +-0.01 mm imperfection (worse of two)")
        else:
            out["verdict"] = dict(verdict="FAIL (unverifiable)", passed=False, nrmse=None)
    out["mesh_conv_jax_40x2_vs_80x2"] = gap(out["jax_40x2"], out["jax_80x2"]) if "jax_80x2" in out and all(np.isfinite(out["jax_80x2"])) else None
    json.dump(out, open(f"results/verify_{name}.json", "w"), indent=1)
    print(name, "VERDICT", out["verdict"]["verdict"], out["verdict"].get("nrmse"), "graded on", graded, flush=True)
    return out


def bfeat(F):
    F = np.array([np.nan if v is None else v for v in F])
    if not np.all(np.isfinite(F)):
        return None
    idx = snap.indices(F)
    if idx is None:
        return None
    f, du = snap.features(F, idx)
    rel = f / snap.TGT - 1
    return dict(F_pk=float(f[0]), d_2=float(f[1]), E_back=float(f[2]), d_u=float(du), rel_err=rel.tolist(), within_5pct=bool(np.all(np.abs(rel) <= 0.05)))


def verify_bistable(name, p):
    out = dict(name=name, p=np.asarray(p).tolist(), d=snap.DS.tolist(), target=snap.TGT.tolist())
    S = fem.Solver(nx=40, ny=2, Lh=design.LH)
    F, zs, infos = S.path(p, snap.DS)
    out["jax_40x2"] = dict(F=F.tolist(), features=bfeat(F))
    Fb, _, _ = S.path(p, snap.DS[::-1][1:], u0=zs[-1], d_start=float(snap.DS[-1]))
    out["jax_backward_max_abs_diff"] = float(np.max(np.abs(Fb[::-1] - F[:-1])))
    S2 = fem.Solver(nx=80, ny=2, Lh=design.LH)
    F2, _, _ = S2.path(p, snap.DS)
    out["jax_80x2"] = dict(F=F2.tolist(), features=bfeat(F2))
    print(name, "jax", out["jax_40x2"]["features"], flush=True)
    runs, graded, done, imp = ladder(name, p, 4, 1, 13.0, 104, snap.DS)
    for r in list(runs.values()) + list(imp.values()):
        r["features"] = bfeat(r["F"])
    out.update(ccx=runs, graded_mesh=graded, completed_meshes=done, imperfection_runs=imp)
    if graded:
        f = runs[graded]["features"]
        out["verdict"] = dict(verdict="PASS" if f and f["within_5pct"] else "MISS", mesh=graded, features=f)
    elif imp and all(r["completed"] for r in imp.values()):
        fs = [r["features"] for r in imp.values()]
        ok = all(f and f["within_5pct"] for f in fs)
        out["verdict"] = dict(verdict="PASS" if ok else "MISS", mesh="t4 with +-0.01 mm imperfection (both signs)", features=fs)
    else:
        out["verdict"] = dict(verdict="FAIL (unverifiable)")
    json.dump(out, open(f"results/verify_{name}.json", "w"), indent=1)
    print(name, "VERDICT", out["verdict"], flush=True)
    return out


if __name__ == "__main__":
    if sys.argv[1] == "curve":
        verify_curve(sys.argv[2], sys.argv[3], np.array(json.load(open(sys.argv[4]))["p"]))
    else:
        verify_bistable(sys.argv[2], np.array(json.load(open(sys.argv[3]))["p"]))
