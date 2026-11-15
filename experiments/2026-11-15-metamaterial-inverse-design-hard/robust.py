"""Robust design under the pre-registered fabrication-error model.

usage: python robust.py opt <target> <nominal.json>         -> results/robust_<target>.json  (design + design-solver pass fractions)
       python robust.py ccx <target> <which> <i0> <i1>      -> results/robust_ccx/<target>_<which>_<k>.json for samples i0..i1-1
"""
import json, sys, time
import numpy as np
import design, fem

N_OPT, SEED_OPT, N_EVAL, SEED_EVAL, N_CCX = 8, 1, 64, 2026, 24


def eval_copies(tname, p, S=None):
    sp = design.SPEC[tname]
    prob = design.Problem(tname, S=S)
    offs = design.fab_samples(N_EVAL, SEED_EVAL, sp["nk"], sp["nm"], sp["ncell"])
    out = []
    for k, off in enumerate(offs):
        sig, _, meta = prob.curve(np.asarray(p) + off)
        g = design.grade(tname, sig)
        out.append(dict(k=k, nrmse=g["nrmse"], passed=g["passed"], bands=[b["max_dev"] for b in g["bands"]], sigma=np.asarray(sig).tolist(), **meta))
    e = np.array([o["nrmse"] if o["nrmse"] is not None else np.inf for o in out])
    return dict(samples=out, pass_fraction=float(np.mean([o["passed"] for o in out])), median_nrmse=float(np.median(e)), max_nrmse=float(e.max()),
                p90_nrmse=float(np.quantile(e, 0.9)))


if __name__ == "__main__":
    mode, tname = sys.argv[1], sys.argv[2]
    sp = design.SPEC[tname]
    if mode == "opt":
        p_nom = np.array(json.load(open(sys.argv[3]))["p"])
        S = fem.Solver(nx=40, ny=2, Lh=design.LH, nk=sp["nk"], nm=sp["nm"], ncell=sp["ncell"])
        offs = design.fab_samples(N_OPT, SEED_OPT, sp["nk"], sp["nm"], sp["ncell"])
        # default (pre-registered): start from the nominal optimum.  Optional 4th argument: a generic start name
        # (added 17:25 after the pre-registered start turned out to be a local robust optimum; see log.md)
        sname = sys.argv[4] if len(sys.argv) > 4 else None
        p0 = design.starts(tname)[sname] if sname else p_nom
        res = design.run(tname, f"robust_from_{sname or 'nominal'}", p0, offsets=offs, max_nfev=80, S=S)
        out = dict(target=tname, p_nominal=p_nom.tolist(), p_robust=res["p"], opt=res, start=sname or "nominal")
        nom_curve = design.Problem(tname, S=S).curve(p_nom)[0]
        out["nominal_design"] = dict(grade=design.grade(tname, nom_curve), copies=eval_copies(tname, p_nom, S))
        out["robust_design"] = dict(grade=res["design_grade"], copies=eval_copies(tname, res["p"], S))
        print("nominal: own nrmse", out["nominal_design"]["grade"]["nrmse"], "pass fraction", out["nominal_design"]["copies"]["pass_fraction"])
        print("robust : own nrmse", out["robust_design"]["grade"]["nrmse"], "pass fraction", out["robust_design"]["copies"]["pass_fraction"])
        json.dump(out, open(f"results/robust_{tname}{'_from_' + sname if sname else ''}.json", "w"), indent=1)
    else:
        import ccx
        which, i0, i1 = sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
        r = json.load(open(f"results/robust_{tname}.json"))  # the selected robust design (see select step in log.md)
        p = np.array(r["p_" + which])
        offs = design.fab_samples(N_EVAL, SEED_EVAL, sp["nk"], sp["nm"], sp["ncell"])
        grid = design.travel_grid(sp["ncell"])
        for k in range(i0, i1):
            rec = dict(k=k, tries=[])
            for div in (3.0, 4.0):  # pre-registered: t_min/3, fall back to t_min/4 on abort; abort on both = failed copy
                t0 = time.time()
                try:
                    d, F, info = ccx.run(f"rob_{tname}_{which}_{k}", p + offs[k], sp["nk"], ncell=sp["ncell"], dmax=float(grid[-1]), lc_div=div, threads=1)
                    sig = ccx.sample(d, F, grid, info["completed"]) * design.KPA
                    g = design.grade(tname, sig)
                    rec["tries"].append(dict(lc_div=div, completed=info["completed"], d_reached=info["d_reached"], sec=time.time() - t0))
                    if info["completed"]:
                        rec.update(nrmse=g["nrmse"], passed=g["passed"], bands=[b["max_dev"] for b in g["bands"]], sigma=sig.tolist(), lc_div=div)
                        break
                except Exception as e:
                    rec["tries"].append(dict(lc_div=div, error=repr(e)))
            rec.setdefault("passed", False)
            rec.setdefault("nrmse", None)
            json.dump(rec, open(f"results/robust_ccx/{tname}_{which}_{k}.json", "w"))
            print(which, k, rec["nrmse"], rec["passed"], flush=True)
