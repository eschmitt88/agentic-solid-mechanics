"""Grade trial 13 against PREREGISTRATION.md (H1-H6; H7 in grade_ccx.py).

  python analyze.py            -> results/grades.json, metrics.json, printed table
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ea  # noqa
import run_arm as ra  # noqa

RES = HERE / "results"
R11, R20 = np.array([1.1, 1.1]), np.array([2.0, 2.0])


def gen0_objs(obj, kind="rec"):
    d = np.load(RES / "gen0" / f"eval_{obj}.npz")
    recs = [dict(vf=d[f"{kind}_vf"][i], connected=bool(d[f"{kind}_connected"][i]),
                 compliance=d[f"{kind}_compliance"][i], blf=d[f"{kind}_blf"][i]) for i in range(len(d["Z0"]))]
    return recs


def cref(obj):
    if obj != "buck":
        return None
    d = np.load(RES / "gen0" / "eval_buck.npz")
    ok = d["rec_connected"].astype(bool)
    return ra.compliance_reference(d["rec_vf"][ok], d["rec_compliance"][ok])


def setup(obj):
    F0, cv0 = ra.objectives(gen0_objs(obj), obj, cref(obj))
    norm = ea.Normaliser(F0[cv0 <= 0])
    hv = lambda F, cv, r=R11: ea.hv2d(norm(F[cv <= 0]), r)
    return F0, cv0, norm, hv


def arm_runs(obj, arm):
    return sorted(p for p in (RES / "arms").glob(f"{obj}_{arm}_s*") if (p / "archive.npz").exists())


def final_hv(obj, arm):
    F0, cv0, norm, hv = setup(obj)
    h0 = hv(F0, cv0); h0b = hv(F0, cv0, R20)
    out = {}
    for p in arm_runs(obj, arm):
        a = np.load(p / "archive.npz")
        hist = [json.loads(l) for l in open(p / "history.jsonl")]
        out[p.name] = dict(gens=hist[-1]["gen"], n_eval=int(len(a["F"])), hv11=hv(a["F"], a["CV"]),
                           hv20=hv(a["F"], a["CV"], R20), gain11=hv(a["F"], a["CV"]) / h0 - 1,
                           gain20=hv(a["F"], a["CV"], R20) / h0b - 1)
    return out, h0, h0b


def morelf_hv(obj, n_eval_match, kind="raw"):
    f = RES / "morelf" / f"eval_{obj}.npz"
    if not f.exists():
        return None
    F0, cv0, norm, hv = setup(obj)
    base = gen0_objs(obj, "raw" if kind == "raw" else "rec")
    d = np.load(f)
    n_extra = n_eval_match - len(base)
    ex = [dict(vf=d[f"{kind}_vf"][i], connected=bool(d[f"{kind}_connected"][i]), compliance=d[f"{kind}_compliance"][i],
               blf=d[f"{kind}_blf"][i]) for i in range(min(n_extra, len(d[f"{kind}_vf"])))]
    F, cv = ra.objectives(base + ex, obj, cref(obj))
    return dict(n_eval=len(base) + len(ex), hv11=hv(F, cv), hv20=hv(F, cv, R20), gain11=hv(F, cv) / hv(F0, cv0) - 1)


def h5_simp():
    s = json.load(open(RES / "simp" / "simp_eval.json"))["designs"]
    p = RES / "arms" / "buck_FULL_s0" / "archive.npz"
    if not p.exists():
        return None
    a = np.load(p)
    rows, wins = [], 0
    for d in s:
        if round(d["vf"], 2) not in (0.30, 0.35, 0.40, 0.45):
            continue
        m = (np.abs(a["vf"] - d["vf"]) <= 0.01) & (a["CV"] <= 0)
        if not m.any():
            rows.append(dict(vf=d["vf"], simp_blf=d["blf"], full_blf=None)); continue
        i = np.where(m)[0][np.argmax(a["blf"][m])]
        win = a["blf"][i] > d["blf"]; wins += int(win)
        rows.append(dict(vf=d["vf"], simp_blf=d["blf"], simp_C=d["compliance"], full_blf=float(a["blf"][i]),
                         full_vf=float(a["vf"][i]), full_C=float(a["compliance"][i]),
                         C_ratio=float(a["compliance"][i] / d["compliance"]), win=bool(win)))
    return dict(rows=rows, wins=wins, verdict="PASS" if wins >= 3 else "MISS")


def h6(obj, runs):
    F0, cv0, norm, hv = setup(obj)
    cr = cref(obj)

    def averaged(f):
        d = np.load(f)
        vf = d["vf"].mean(1); C = np.where(d["connected"].all(1), d["compliance"].mean(1), np.inf)
        if obj == "stiff":
            F = np.stack([vf, np.where(np.isfinite(C), C, 1e30)], 1); cv = np.where(np.isfinite(C), 0.0, 1e3)
        else:
            blf = d["blf"].mean(1)
            F = np.stack([vf, np.where(np.isfinite(C), -blf, 1e30)], 1)
            cv = np.where(np.isfinite(C), np.maximum(0, C / cr(vf) - 1), 1e3)
        return F, cv, d["Fnom"]

    g0 = RES / "rereal" / f"{obj}_gen0.npz"
    if not g0.exists():
        return None
    Fg, cvg, Fgn = averaged(g0)
    out = {}
    for run in runs:
        f = RES / "rereal" / f"{obj}_{run}.npz"
        if not f.exists():
            continue
        Fr, cvr, Frn = averaged(f)
        nominal = ea.hv2d(norm(np.concatenate([Fgn, Frn])), R11) / ea.hv2d(norm(Fgn), R11) - 1
        avg = hv(np.concatenate([Fg, Fr]), np.concatenate([cvg, cvr])) / hv(Fg, cvg) - 1
        out[run] = dict(nominal_gain=nominal, averaged_gain=avg, retained=avg / nominal if nominal > 0 else None)
    if out:
        ret = [v["retained"] for v in out.values() if v["retained"] is not None]
        out["mean_retained"] = float(np.mean(ret)) if ret else None
        out["verdict"] = "PASS" if ret and np.mean(ret) >= 0.5 else "MISS"
    return out


def main():
    G = {}
    for obj in ("stiff", "buck"):
        G[obj] = {}
        for arm in ("FULL", "NOMUT", "NOVAE", "NOSEL"):
            r, h0, h0b = final_hv(obj, arm)
            if r:
                G[obj][arm] = r
        G[obj]["gen0_hv11"], G[obj]["gen0_hv20"] = h0, h0b
        if "FULL" in G[obj]:
            n = max(v["n_eval"] for v in G[obj]["FULL"].values())
            G[obj]["MORELF_raw"] = morelf_hv(obj, n, "raw")
            G[obj]["MORELF_proj"] = morelf_hv(obj, n, "proj")
    V = {}
    for obj in ("stiff", "buck"):
        full = G[obj].get("FULL", {})
        gains = [v["gain11"] for v in full.values()]
        if len(gains) == 3:
            V[f"H1_{obj}"] = dict(gains=gains, mean=float(np.mean(gains)),
                                  verdict="PASS" if np.mean(gains) >= 0.10 and min(gains) > 0 else "MISS")
        ml = G[obj].get("MORELF_raw")
        if ml and len(gains) == 3:
            hvs = [v["hv11"] for v in full.values()]
            V[f"H2a_{obj}"] = dict(full=hvs, morelf=ml["hv11"], morelf_proj=(G[obj].get("MORELF_proj") or {}).get("hv11"),
                                   verdict="EA > more LF" if min(hvs) > ml["hv11"] else
                                   ("more LF >= EA" if max(hvs) <= ml["hv11"] else "no difference"))
        nosel = G[obj].get("NOSEL", {})
        if nosel and full:
            pairs = [(full[k]["hv11"], nosel[k.replace("FULL", "NOSEL")]["hv11"]) for k in full
                     if k.replace("FULL", "NOSEL") in nosel]
            V[f"H2b_{obj}"] = dict(pairs=pairs, verdict="PASS" if pairs and all(a > b for a, b in pairs) else "MISS")
    for arm, H in (("NOMUT", "H3"), ("NOVAE", "H4")):
        o = G["stiff"].get(arm, {})
        full = G["stiff"].get("FULL", {})
        if len(o) == 3 and len(full) == 3:
            dlt = np.mean([v["gain11"] for v in full.values()]) - np.mean([v["gain11"] for v in o.values()])
            V[H] = dict(diff_pp=100 * dlt, verdict="PASS" if dlt >= 0.02 else "MISS")
    V["H5"] = h5_simp()
    for obj in ("stiff", "buck"):
        V[f"H6_{obj}"] = h6(obj, [p.name for p in arm_runs(obj, "FULL")])
    out = dict(arms=G, verdicts=V)
    (RES / "grades.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(V, indent=1, default=float))
    return out


if __name__ == "__main__":
    main()
