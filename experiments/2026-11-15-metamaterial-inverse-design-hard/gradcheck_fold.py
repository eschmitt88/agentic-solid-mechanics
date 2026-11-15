"""Gradient check of the fold-point sensitivities (pre-registered, run before the snap test).

For two of trial 9's stalled designs: dd_s/dp (augmented limit-point formula) and dE_back/dp (energy end points +
fold term) against central finite differences.  FD of d_s re-solves the fold at p +- h by arclength secant; FD of
E_back re-runs the whole pipeline from scratch at p +- h (path, crossings, fold location, energies).
Also recorded: what trial 9's per-state gradient of the grid/trapezoid E_back says for the same components.
"""
import json, sys
import numpy as np, jax.numpy as jnp
import snap

T9 = "../2026-11-14-metamaterial-inverse-design/results/"
designs = {"attempt5_h6_t1.9_stalled": json.load(open(T9 + "opt_bistable_attempt5_B.json"))["h6_t1.9_a2"]["p"],
           "attempt3_h5.5_t2.0_stalled": json.load(open(T9 + "opt_bistable_attempt3_stalled.json"))["h5.5_t2.0_a2"]["p"],
           "attempt6_final_design": json.load(open(T9 + "opt_bistable_attempt6_C.json"))["across_ridge"]["p"]}
P = snap.FoldProblem(log=open("/dev/null", "w"))
G = snap.GridProblem(S=P.S, log=open("/dev/null", "w"))
out = {}
for name, p in designs.items():
    p = np.array(p)
    c = P.state_features(p)
    g = G.evaluate(p)
    rec = dict(p=p.tolist(), feat=c["feat"].tolist(), grid_feat=g["feat"].tolist(), nsnap=c["nsnap"], E_endpoints=c["E_endpoints"],
               dE_fold_aware=c["dfeat"][2].tolist(), dE_grid_per_state=g["dfeat"][2].tolist(), dFpk=c["dfeat"][0].tolist(), dd2=c["dfeat"][1].tolist())
    print(name, "feat", c["feat"], "snaps", c["nsnap"], flush=True)
    comps = list(np.argsort(-np.abs(c["dfeat"][2]))[:3])
    if c["nsnap"]:
        f = c["snaps"][0]
        rec["fold"] = dict(d_s=f["d_s"], lam_fold=f["lam_fold"], loss=f["loss"], F_minus=f["F_minus"], F_plus=f["F_plus"], dds_dp=f["dds_dp"].tolist(), exact=f["exact"])
        comps_s = list(np.argsort(-np.abs(f["dds_dp"]))[:3])
        rec["fd_dds"] = {}
        for j in comps_s:
            for h in (1e-4, 1e-5):
                pp, pm = p.copy(), p.copy()
                pp[j] += h; pm[j] -= h
                back = max(2e-4, 4 * abs(f["dds_dp"][j]) * h)
                a = P.folds.refold(pp, P.S.coords(jnp.asarray(pp)), f, back)
                b = P.folds.refold(pm, P.S.coords(jnp.asarray(pm)), f, back)
                fd = (a["d_s"] - b["d_s"]) / (2 * h)
                fdl = (a["loss"] - b["loss"]) / (2 * h)
                rec["fd_dds"][f"{j}_{h}"] = dict(fd=fd, an=float(f["dds_dp"][j]), rel=abs(fd / f["dds_dp"][j] - 1), fd_loss=fdl, an_loss=float(f["dloss_dp"][j]), rel_loss=abs(fdl / f["dloss_dp"][j] - 1))
                print("  dd_s/dp", j, h, fd, f["dds_dp"][j], "rel", abs(fd / f["dds_dp"][j] - 1), "| dloss/dp", fdl, f["dloss_dp"][j], flush=True)
    rec["fd_feat"] = {}
    for j in comps:
        for h in (1e-4, 2e-5):
            pp, pm = p.copy(), p.copy()
            pp[j] += h; pm[j] -= h
            cp, cm = P.state_features(pp), P.state_features(pm)
            if cp["feat"] is None or cm["feat"] is None:
                rec["fd_feat"][f"{j}_{h}"] = dict(note="perturbed design not bistable / path failed")
                print("  comp", j, "h", h, "perturbed design returned no features", flush=True)
                continue
            fd = (cp["feat"] - cm["feat"]) / (2 * h)
            gp, gm = G.evaluate(pp), G.evaluate(pm)
            fdg = (gp["feat"] - gm["feat"]) / (2 * h) if gp["feat"] is not None and gm["feat"] is not None else np.full(3, np.nan)
            rel = np.abs(fd / c["dfeat"][:, j] - 1)
            rec["fd_feat"][f"{j}_{h}"] = dict(fd=fd.tolist(), an=c["dfeat"][:, j].tolist(), rel=rel.tolist(), nsnap=[cp["nsnap"], cm["nsnap"]],
                                              fd_grid=fdg.tolist(), an_grid=g["dfeat"][:, j].tolist())
            print(f"  comp {j} h {h}: [F_pk,d_2,E] FD {fd}  fold-aware {c['dfeat'][:, j]}  rel {rel}", flush=True)
            print(f"            trial-9 grid: FD of grid feature {fdg}  per-state gradient {g['dfeat'][:, j]}", flush=True)
    out[name] = rec
json.dump(out, open("results/gradcheck_fold.json", "w"), indent=1)
