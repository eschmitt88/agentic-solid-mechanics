"""Rung 1, CalculiX: patch test and Hertz with several contact formulations."""
import sys, json, os
import numpy as np
import meshes as M, ccx
from rung1_own import E, NU, R, A_T, edge_from

HERE = os.path.dirname(os.path.abspath(__file__))


def contact_cards(ctype, K, mu=0.0, stick=None, slave="SSLAVE", master="SMASTER"):
    s = "*SURFACE INTERACTION, NAME=SI\n"
    if ctype in ("MORTAR", "LINMORTAR", "PGLINMORTAR"):
        s += "*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=HARD\n"
    else:
        s += f"*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=LINEAR\n{K:.6g}\n"
    if mu > 0:
        s += f"*FRICTION\n{mu},{stick:.6g}\n"
    s += f"*CONTACT PAIR, INTERACTION=SI, TYPE={ctype}\n{slave},{master}\n"
    return s


def patch(ctype, kfac=50.0, tag="", nlow=(7, 4), nup=(5, 3), p_app=1.0):
    Xl, ql = M.structured_rect(0, 2, 0, 1, *nlow); Xu, qu = M.structured_rect(0, 2, 1, 2, *nup)
    X, Q, off, body = M.merge([(Xl, ql), (Xu, qu)])
    edges, el, fa = M.boundary_edges(Q); mid = 0.5 * (X[edges[:, 0]] + X[edges[:, 1]])
    isup = edges[:, 0] >= off[1]
    sl = isup & (np.abs(mid[:, 1] - 1) < 1e-9); ma = ~isup & (np.abs(mid[:, 1] - 1) < 1e-9); top = isup & (np.abs(mid[:, 1] - 2) < 1e-9)
    h = 2 / max(nlow[0], nup[0]); K = kfac * E / h
    job = f"patch_{ctype.replace(' ', '')}{tag}"
    deck = "*HEADING\npatch\n" + ccx.nodes_block(X) + ccx.elements_block(Q, "CPE4", "EALL")
    deck += ccx.surface_block("SSLAVE", el[sl], fa[sl]) + ccx.surface_block("SMASTER", el[ma], fa[ma]) + ccx.surface_block("STOP", el[top], fa[top])
    deck += ccx.nset("NBOT", np.where(np.abs(X[:, 1]) < 1e-12)[0])
    pin1 = int(np.where((np.abs(X[:, 0]) < 1e-12) & (np.abs(X[:, 1]) < 1e-12))[0][0]) + 1
    pin2 = int(off[1] + np.where((np.abs(Xu[:, 0]) < 1e-12) & (np.abs(Xu[:, 1] - 2) < 1e-12))[0][0]) + 1
    deck += f"*MATERIAL, NAME=M\n*ELASTIC\n{E},{NU}\n*SOLID SECTION, ELSET=EALL, MATERIAL=M\n"
    deck += contact_cards(ctype, K)
    deck += f"*BOUNDARY\nNBOT,2,2\n{pin1},1,1\n{pin2},1,1\n"
    deck += "*STEP, INC=100\n*STATIC\n0.5,1.0\n*DLOAD\n" + "".join(f"{e+1},P{f+1},{p_app}\n" for e, f in zip(el[top], fa[top]))
    deck += "*EL PRINT, ELSET=EALL\nS\n*CONTACT FILE\nCSTR\n*NODE FILE\nU\n*END STEP\n"
    open(os.path.join(ccx.WORK, job + ".inp"), "w").write(deck)
    rc, out, sec = ccx.run(job)
    info = ccx.summarize_log(out); res = dict(ctype=ctype, kfac=kfac, seconds=sec, **info)
    if not info["finished"]:
        return res
    # integration point stresses from .dat
    syy = []
    for l in open(os.path.join(ccx.WORK, job + ".dat")):
        if "stresses" in l:
            syy = []          # keep only the last increment
        p = l.split()
        if len(p) == 8:
            try:
                syy.append(float(p[3]))
            except ValueError:
                pass
    syy = np.array(syy)
    blocks = [b for b in ccx.parse_frd(os.path.join(ccx.WORK, job + ".frd")) if b["name"].startswith("CONTACT")]
    res["syy_dev"] = float(np.max(np.abs(syy + p_app)) / p_app)
    if blocks:
        b = blocks[-1]; sn = np.unique(edges[sl]); idx = np.searchsorted(b["nodes"], sn)
        cp = b["values"][idx, 3]
        res["p_dev"] = float(np.max(np.abs(cp - p_app)) / p_app); res["p_nodes"] = [round(float(v), 4) for v in cp]
    return res


def hertz(level, axisym, ctype, kfac=50.0, tag="", order=1, a_t=A_T):
    h_c = a_t / (8 * 2 ** level)
    geo = M.half_disc_geo(R, h_c, 3 * a_t, 0.08, 0.7, quarter=axisym)
    X1, Q1, _ = M.run_gmsh(geo, os.path.join(HERE, "results", "work"), f"hertz_{'ax' if axisym else '2d'}_L{level}_a{a_t}")
    # "rigid" flat: a strip of very stiff, fully clamped elements, mesh deliberately non-matching
    hx = 1.37 * h_c
    nx = int(np.ceil((6 * a_t if not axisym else 3 * a_t) / hx))
    X2, Q2 = M.structured_rect(0.0 if axisym else -3 * a_t, 3 * a_t, -4 * hx, 0.0, nx, 2)
    X, Q, off, body = M.merge([(X1, Q1), (X2, Q2)])
    edges, el, fa = M.boundary_edges(Q); mid = 0.5 * (X[edges[:, 0]] + X[edges[:, 1]])
    isflat = edges[:, 0] >= off[1]
    sl = (~isflat) & (mid[:, 1] < 0.3 * R) & (np.abs(np.hypot(mid[:, 0], mid[:, 1] - R) - R) < 1e-3 * R) & (np.abs(mid[:, 0]) < 2.5 * a_t) & (np.abs(mid[:, 0]) > 1e-9)
    ma = isflat & (np.abs(mid[:, 1]) < 1e-12)
    et = "CAX4" if axisym else "CPE4"
    K = kfac * E / h_c
    job = f"hertz_{'ax' if axisym else '2d'}_{ctype.replace(' ', '')}_L{level}{tag}"
    n1 = off[1]
    top = np.where(np.abs(X[:n1, 1] - R) < 1e-9)[0]
    deck = "*HEADING\nhertz\n" + ccx.nodes_block(X) + ccx.elements_block(Q[body == 0], et, "EBODY") + ccx.elements_block(Q[body == 1], et, "EFLAT", start=int((body == 0).sum()))
    deck += ccx.surface_block("SSLAVE", el[sl], fa[sl]) + ccx.surface_block("SMASTER", el[ma], fa[ma])
    deck += ccx.nset("NFLAT", np.arange(n1, len(X))) + ccx.nset("NTOP", top)
    deck += f"*MATERIAL, NAME=M\n*ELASTIC\n{E},{NU}\n*MATERIAL, NAME=MR\n*ELASTIC\n{E*1e6},{NU}\n*SOLID SECTION, ELSET=EBODY, MATERIAL=M\n*SOLID SECTION, ELSET=EFLAT, MATERIAL=MR\n"
    deck += contact_cards(ctype, K)
    Es = E / (1 - NU ** 2)
    if axisym:
        d = a_t ** 2 / R
        axis = np.where(np.abs(X[:n1, 0]) < 1e-12)[0]
        deck += ccx.nset("NAXIS", axis) + "*BOUNDARY\nNFLAT,1,2\nNAXIS,1,1\n"
        deck += f"*STEP, INC=1000\n*STATIC\n0.25,1.0\n*BOUNDARY\nNTOP,2,2,{-d}\n"
        F_H = 4 / 3 * Es * np.sqrt(R) * d ** 1.5
        ref = dict(F=F_H, a=np.sqrt(R * d), p0=3 * F_H / (2 * np.pi * R * d))
    else:
        P = np.pi * Es * a_t ** 2 / (4 * R)
        rn = int(top[np.argmin(np.abs(X[top, 0]))]) + 1
        deck += "*EQUATION\n" + "".join(f"2\n{n+1},2,1.0,{rn},2,-1.0\n" for n in top if n + 1 != rn)
        deck += f"*BOUNDARY\nNFLAT,1,2\n{rn},1,1\n"
        deck += f"*STEP, INC=1000\n*STATIC\n1.0,1.0\n*BOUNDARY\n{rn},2,2,{-0.2*a_t**2/R}\n*END STEP\n"
        deck += f"*STEP, INC=1000\n*STATIC\n0.25,1.0\n*BOUNDARY, OP=NEW\nNFLAT,1,2\n{rn},1,1\n*CLOAD\n{rn},2,{-P}\n"
        ref = dict(F=P, a=a_t, p0=2 * P / (np.pi * a_t))
    deck += "*NODE PRINT, NSET=NTOP, TOTALS=ONLY\nRF\n*NODE PRINT, NSET=NFLAT, TOTALS=ONLY\nRF\n*CONTACT FILE\nCSTR\n*NODE FILE\nU\n*EL FILE\nS\n*END STEP\n"
    open(os.path.join(ccx.WORK, job + ".inp"), "w").write(deck)
    rc, out, sec = ccx.run(job)
    info = ccx.summarize_log(out)
    res = dict(ctype=ctype, kfac=kfac, level=level, h_c=h_c, nodes=len(X), seconds=sec, ref=ref, **info)
    if not info["finished"]:
        return res
    tot = ccx.parse_dat_totals(os.path.join(ccx.WORK, job + ".dat"))
    res["RF_flat"] = [t for t in tot if t[0] == "NFLAT"][-1][2][1]
    # CalculiX expands axisymmetric elements into a 2-degree wedge: forces are per 1/180 of the circumference
    res["F"] = abs(res["RF_flat"]) * (180.0 if axisym else 1.0); res["err_F"] = res["F"] / ref["F"] - 1
    b = [b for b in ccx.parse_frd(os.path.join(ccx.WORK, job + ".frd")) if b["name"].startswith("CONTACT")][-1]
    sn = np.unique(edges[sl]); idx = np.searchsorted(b["nodes"], sn)
    if ctype == "NODE TO SURFACE":
        # node-to-surface writes its contact output on internal nodes of the expanded 3D mesh;
        # use minus the extrapolated nodal stress s_yy at the slave nodes instead
        sb = [q for q in ccx.parse_frd(os.path.join(ccx.WORK, job + ".frd")) if q["name"] == "STRESS"][-1]
        cp = np.maximum(-sb["values"][np.searchsorted(sb["nodes"], sn), 1], 0.0); res["pressure_source"] = "-s_yy nodal"
    else:
        cp = b["values"][idx, 3]
    x = np.abs(X[sn, 0]); o = np.argsort(x); x, cp = x[o], cp[o]
    res["F_from_pressure"] = float(np.trapezoid(cp * (2 * np.pi * x if axisym else 1.0), x)) * (1 if axisym else 1)
    a_fe, reso = edge_from(x, cp)
    near = (x < 0.1 * ref["a"]) & ((x > 1e-12) | (not axisym))   # the axis node itself is excluded (see log)
    res["p_axis_node"] = float(cp[0])
    res.update(a=float(a_fe), a_resolution=float(reso), p0=float(np.mean(cp[near])), p0_max=float(cp.max()))
    pH = ref["p0"] * np.sqrt(np.clip(1 - (x / ref["a"]) ** 2, 0, None)); sel = (cp > 0) | (pH > 0)
    res["l2"] = float(np.sqrt(np.sum((cp[sel] - pH[sel]) ** 2) / np.sum(pH[sel] ** 2)))
    res["err_a"] = res["a"] / ref["a"] - 1; res["err_p0"] = res["p0"] / ref["p0"] - 1
    np.savez(os.path.join(HERE, "results", f"r1_{job}.npz"), x=x, p=cp)
    return res


if __name__ == "__main__":
    os.makedirs(ccx.WORK, exist_ok=True)
    what = sys.argv[1]
    if what == "patch":
        res = [patch(ct) for ct in ("SURFACE TO SURFACE", "NODE TO SURFACE", "MORTAR")]
        for r in res: print(r)
        json.dump(res, open(os.path.join(HERE, "results", "r1_patch_ccx.json"), "w"), indent=1)
    elif what == "sweep":
        res = []
        def go(*a, **k):
            try:
                r = hertz(*a, **k)
            except Exception as e:
                r = dict(args=str(a) + str(k), exception=repr(e))
            res.append(r); print({k2: v for k2, v in r.items() if k2 not in ("ref",)}, flush=True)
            json.dump(res, open(os.path.join(HERE, "results", "r1_hertz_ccx.json"), "w"), indent=1)
        for ax in (False, True):
            for lv in (0, 1, 2, 3):
                go(lv, ax, "SURFACE TO SURFACE")                      # graded configuration at level 3
            for kf in (0.5, 5.0, 500.0, 5000.0):
                go(2, ax, "SURFACE TO SURFACE", kfac=kf, tag=f"_k{kf}")
            for kf in (5.0, 50.0, 500.0):
                go(2, ax, "NODE TO SURFACE", kfac=kf, tag=f"_k{kf}")
    elif what == "one":
        r = hertz(int(sys.argv[3]), sys.argv[2] == "sphere", sys.argv[4], kfac=float(sys.argv[5]) if len(sys.argv) > 5 else 50.0)
        print({k: v for k, v in r.items() if k != "ref"})
