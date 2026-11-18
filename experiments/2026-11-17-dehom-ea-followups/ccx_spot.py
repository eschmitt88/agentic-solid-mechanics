"""Addendum B item 5: CalculiX (body-fitted) spot-check of trial-14 front designs on the girder.

  python -u ccx_spot.py run --par 3          # 15 designs per objective: 5 each from FULL s0, PCA s0, gen-0 (all-480 held-out front)
  python ccx_spot.py grade

Each design: default realisation, pixel evaluator (hf.py, solid-only) vs verify_ccx (contour of `level`,
r_c = 0, mesh 0.5 mm — trial 13's settings). Reports Spearman rho, ratio spread, and flags designs whose
CalculiX compliance exceeds the pixel value by > 30 % (likely pixel-connectivity artefacts).
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
import heldout as H  # noqa

OUT = HERE / "results" / "ccx_spot"


def pick(obj, src, n=5):
    if src == "gen0":
        d = np.load(HERE / f"results/heldout/{obj}__gen0all.npz")
        F, conn = H.averaged_objectives(d, obj)
        Z = d["Z"][conn]
    else:
        d = np.load(HERE / f"results/heldout/{obj}_{src}__archive.npz")
        F, conn = H.averaged_objectives(d, obj)
        Z = d["Z"][conn]
    fr = np.where(ea.nondominated_sort(F) == 0)[0]
    fr = fr[np.argsort(F[fr, 0])]
    sel = fr[np.linspace(0, len(fr) - 1, n).round().astype(int)]
    return [(f"{obj}|{src}|{i}", Z[i]) for i in sel]


def _one(task):
    import dehom, hf, verify_ccx
    key, z = task
    pca = ra.load_pca()
    w1, w2, th = ra.fields_from_features(pca.inverse(z))
    d = dehom.dehomogenise(w1, w2, th)
    px = hf.Problem(hf.GIRDER, hf.H_FINE).evaluate(d["phi"], buckling=True)
    cx = verify_ccx.verify(d["level"], kind="level", r_c=0.0, mesh_size=0.5, buckling=True, threads=3, tag=key.replace("|", "_"))
    f = lambda r: {k: float(r[k]) for k in ("vf", "compliance", "blf") if k in r and np.isscalar(r[k])}
    return dict(key=key, pixel=f(px), ccx=f(cx), ccx_ok=bool(cx.get("ok")))


def run(par):
    import multiprocessing as mp
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = [t for obj in ("stiff", "buck") for src in ("FULL_s0", "PCA_s0", "gen0") for t in pick(obj, src)]
    tasks = [t for t in tasks if not (OUT / f"{t[0].replace('|', '_')}.json").exists()]
    with mp.get_context("spawn").Pool(par) as pool:
        for r in pool.imap_unordered(_one, tasks):
            (OUT / f"{r['key'].replace('|', '_')}.json").write_text(json.dumps(r))
            print(r["key"], r["pixel"].get("compliance"), r["ccx"].get("compliance"), flush=True)


def grade():
    from scipy.stats import spearmanr
    rows = [json.loads(p.read_text()) for p in sorted(OUT.glob("*.json"))]
    ok = [r for r in rows if r["ccx_ok"]]
    res = {}
    for key in ("compliance", "blf"):
        p = np.array([r["pixel"][key] for r in ok]); c = np.array([r["ccx"][key] for r in ok]); q = p / c
        res[key] = dict(n=len(ok), spearman=float(spearmanr(p, c)[0]), ratio_median=float(np.median(q)),
                        ratio_cv=float(q.std() / q.mean()))
    res["artefacts_C_ccx_over_pixel_gt_1.3"] = [r["key"] for r in ok if r["ccx"]["compliance"] / r["pixel"]["compliance"] > 1.3]
    by = {}
    for r in ok:
        src = r["key"].split("|")[1]
        by.setdefault(src, []).append(r["ccx"]["compliance"] / r["pixel"]["compliance"])
    res["ccx_over_pixel_C_median_by_source"] = {k: float(np.median(v)) for k, v in by.items()}
    (HERE / "results" / "ccx_spot.json").write_text(json.dumps(dict(summary=res, rows=rows), indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "grade"])
    ap.add_argument("--par", type=int, default=3)
    a = ap.parse_args()
    run(a.par) if a.cmd == "run" else grade()
