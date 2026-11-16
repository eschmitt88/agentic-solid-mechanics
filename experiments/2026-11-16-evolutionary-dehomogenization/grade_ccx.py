"""H7: calibrate the in-loop pixel evaluator against body-fitted CalculiX.

  python -u grade_ccx.py run --obj stiff|buck --runs gen0,FULL_s0,FULL_s1 --par 3
  python grade_ccx.py grade

Design set (pre-registered "≥ 24 designs, gen-0 and final fronts, S and B, spanning vf"):
for each objective and each run in {gen0, FULL_s0, FULL_s1}, the feasible front member
nearest each of vf* = 0.30, 0.35, 0.40, 0.45 -> 2 x 3 x 4 = 24 designs. Each is
re-dehomogenised (deterministic), scored by hf.py (pixel) and verify_ccx (r_c = 0,
mesh 0.5 mm, the builder's recommendation fixed before grading), contour of `level`.
"""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ea  # noqa
import run_arm as ra  # noqa
from rereal import front_Z  # noqa

RES = HERE / "results"
OUT = RES / "ccx_grade"
VSTAR = (0.30, 0.35, 0.40, 0.45)


def _one(task):
    import dehom, hf, verify_ccx
    key, Fd = task
    w1, w2, th = ra.fields_from_features(Fd)
    d = dehom.dehomogenise(w1, w2, th)
    P = hf.Problem(hf.GIRDER, hf.H_FINE)
    px = P.evaluate(d["phi"], buckling=True)
    cx = verify_ccx.verify(d["level"], kind="level", r_c=0.0, mesh_size=0.5, buckling=True, threads=3, tag=key)
    keep = lambda r: {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                      for k, v in r.items() if k in ("vf", "connected", "compliance", "umax", "blf", "vf_ccx", "vf_pix",
                                                     "ok", "mesh_ok", "blf_ok", "mode_xy", "n_elements", "wall")}
    return dict(key=key, pixel=keep(px), ccx=keep(cx))


def run(obj, runs, par):
    import multiprocessing as mp
    OUT.mkdir(parents=True, exist_ok=True)
    pca = ra.load_pca()
    tasks = []
    for run_name in runs:
        rn = run_name if run_name == "gen0" else f"{obj}_{run_name}"
        Z, F = front_Z(obj, rn)
        for v in VSTAR:
            i = int(np.argmin(np.abs(F[:, 0] - v)))
            key = f"{obj}|{run_name}|{v:.2f}"
            if (OUT / f"{key.replace('|', '_')}.json").exists():
                continue
            tasks.append((key, pca.inverse(Z[i])))
    with mp.get_context("spawn").Pool(par) as pool:
        for r in pool.imap_unordered(_one, tasks):
            (OUT / f"{r['key'].replace('|', '_')}.json").write_text(json.dumps(r, default=str))
            print(r["key"], r["pixel"].get("compliance"), r["ccx"].get("compliance"), r["pixel"].get("blf"),
                  r["ccx"].get("blf"), flush=True)


def grade():
    from scipy.stats import spearmanr
    rows = [json.loads(p.read_text()) for p in sorted(OUT.glob("*.json"))]
    ok = [r for r in rows if r["ccx"].get("ok") and r["pixel"].get("connected")]
    Cp = np.array([r["pixel"]["compliance"] for r in ok]); Cc = np.array([r["ccx"]["compliance"] for r in ok])
    Bp = np.array([r["pixel"]["blf"] for r in ok]); Bc = np.array([r["ccx"]["blf"] for r in ok])
    rC, rB = Cp / Cc, Bp / Bc
    res = dict(n=len(rows), n_ok=len(ok),
               spearman_C=float(spearmanr(Cp, Cc)[0]), spearman_BLF=float(spearmanr(Bp, Bc)[0]),
               ratio_C_mean=float(rC.mean()), ratio_C_cv=float(rC.std() / rC.mean()),
               ratio_BLF_mean=float(rB.mean()), ratio_BLF_cv=float(rB.std() / rB.mean()))
    # front dominance at matched volumes, pixel vs ccx
    dom = {}
    by = {r["key"]: r for r in ok}
    for obj in ("stiff", "buck"):
        for run in ("FULL_s0", "FULL_s1"):
            cnt = {"pixel": 0, "ccx": 0, "n": 0}
            for v in VSTAR:
                a, b = by.get(f"{obj}|{run}|{v:.2f}"), by.get(f"{obj}|gen0|{v:.2f}")
                if not (a and b):
                    continue
                cnt["n"] += 1
                for src in ("pixel", "ccx"):
                    vfa = a[src]["vf_ccx"] if src == "ccx" else a[src]["vf"]
                    vfb = b[src]["vf_ccx"] if src == "ccx" else b[src]["vf"]
                    if obj == "stiff":
                        fa, fb = a[src]["compliance"], b[src]["compliance"]
                    else:
                        fa, fb = -a[src]["blf"], -b[src]["blf"]
                    if vfa <= vfb + 1e-9 and fa <= fb and (vfa < vfb or fa < fb):
                        cnt[src] += 1
            dom[f"{obj}|{run}"] = dict(cnt, agree=(cnt["pixel"] >= 3) == (cnt["ccx"] >= 3))
    res["dominance"] = dom
    res["verdict"] = "PASS" if (len(rows) >= 24 and res["spearman_C"] >= 0.9 and res["spearman_BLF"] >= 0.9
                                and res["ratio_C_cv"] <= 0.10 and res["ratio_BLF_cv"] <= 0.20
                                and all(d["agree"] for d in dom.values())) else "MISS"
    (RES / "grade_ccx.json").write_text(json.dumps(dict(summary=res, rows=rows), indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "grade"])
    ap.add_argument("--obj")
    ap.add_argument("--runs", default="gen0,FULL_s0,FULL_s1")
    ap.add_argument("--par", type=int, default=3)
    a = ap.parse_args()
    run(a.obj, a.runs.split(","), a.par) if a.cmd == "run" else grade()
