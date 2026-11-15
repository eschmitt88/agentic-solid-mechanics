"""Tabulate the snap test (results/snap/*.json -> results/snap_summary.json).

Arms A and B0 are derived from A+O and B0+O: identical code until the first outer-loop event, so their final design is the
best design seen before that event (re-scored here with the common scorer).  If a run had no outer event the derived arm
is the run itself."""
import glob, json, os
import numpy as np
import design, fem, snap

S = fem.Solver(nx=40, ny=2, Lh=design.LH)
PRE = [s for s in snap.snap_starts() if not s.startswith("fresh_")]
FRESH = [s for s in snap.snap_starts() if s.startswith("fresh_")]
SYM = ["cos_h5_t1.6", "cos_h6_t1.4", "cos_h4.5_t1.2"]
runs = {}
for f in sorted(glob.glob("results/snap/*.json")):
    r = json.load(open(f))
    runs[(r["arm"], r["start"])] = r
for (arm, start), r in list(runs.items()):
    if arm in ("A+O", "B0+O"):
        base = arm[:-2]
        if (base, start) in runs:
            continue
        ev = r["outer_events"]
        if not ev:
            d = dict(r, arm=base, derived_from=arm, note="no outer event: identical run")
        else:
            cut = ev[0]["at_eval"] - (3 if ev[0]["kind"] == "gn_restart" else 0)
            hist = r["history"][:cut]
            good = [h for h in hist if h["feat"]]
            if good:
                p = np.array(min(good, key=lambda h: np.max(np.abs(np.array(h["feat"]) / snap.TGT - 1)))["p"])
                sc = snap.score(S, p)
            else:
                p, sc = np.array(r["p0"]), dict(feat=None, rel_err=None, success=False, bistable=False)
            d = dict(arm=base, start=start, p=p.tolist(), nfev=cut, score=sc, success=sc["success"], derived_from=arm, history=hist, outer_events=[])
        runs[(base, start)] = d
        json.dump(d, open(f"results/snap/{base}__{start}.json", "w"), indent=1)
arms = ["B0-legacy", "B0", "B0+O", "A", "A+O", "A+O2", "AM+O2", "AM+O2@120"]
summ = dict(arms={}, per_run={})
for a in arms:
    for group, names in (("preregistered", PRE), ("fresh", FRESH)):
        rr = [runs[(a, s)] for s in names if (a, s) in runs]
        if not rr:
            continue
        summ["arms"].setdefault(a, {})[group] = dict(
            n=len(rr), success=int(sum(r["success"] for r in rr)), median_nfev=float(np.median([r["nfev"] for r in rr])),
            success_symmetric=int(sum(runs[(a, s)]["success"] for s in names if s in SYM and (a, s) in runs)), n_symmetric=int(sum(s in SYM and (a, s) in runs for s in names)),
            within_5pct=int(sum(r["score"]["rel_err"] is not None and max(np.abs(r["score"]["rel_err"])) <= 0.05 for r in rr)),
            restarts=int(sum(len(r.get("outer_events", [])) for r in rr)))
    for s in PRE + FRESH:
        if (a, s) in runs:
            r = runs[(a, s)]
            rel = r["score"]["rel_err"]
            summ["per_run"][f"{a}__{s}"] = dict(success=r["success"], max_rel_err=None if rel is None else float(np.max(np.abs(rel))), rel_err=rel, nfev=r["nfev"],
                                                outer_events=[e["kind"] for e in r.get("outer_events", [])], derived_from=r.get("derived_from"), seconds=r.get("seconds"))
json.dump(summ, open("results/snap_summary.json", "w"), indent=1)
for a, g in summ["arms"].items():
    print(a, g)
