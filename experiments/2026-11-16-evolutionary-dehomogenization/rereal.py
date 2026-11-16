"""H6 winner's-curse test: re-evaluate fronts with K random phasor phase seeds.

  python -u rereal.py --obj stiff|buck --runs stiff_FULL_s0,... --k 5 --workers 10

For gen 0 and each listed run, take the feasible non-dominated set of the archive
(gen 0: of the gen-0 evaluation), re-evaluate every member with phase seeds
1000..1000+K-1 and store all objective samples. Grading is in analyze.py.
"""
from __future__ import annotations

import argparse, json, os, sys
from pathlib import Path

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ea  # noqa
import run_arm as ra  # noqa

RES = HERE / "results"


def front_Z(obj, run):
    if run == "gen0":
        Z0, rec = ra.load_gen0(obj)
        F, cv = ra.objectives(rec, obj, cref(obj))
        Z = Z0
    else:
        a = np.load(RES / "arms" / run / "archive.npz")
        Z, F, cv = a["Z"], a["F"], a["CV"]
    ok = np.where(cv <= 0)[0]
    r = ea.nondominated_sort(F[ok])
    idx = ok[r == 0]
    return Z[idx], F[idx]


def cref(obj):
    if obj != "buck":
        return None
    d = np.load(RES / "gen0" / "eval_buck.npz")
    ok = d["rec_connected"].astype(bool)
    return ra.compliance_reference(d["rec_vf"][ok], d["rec_compliance"][ok])


def main(a):
    pca = ra.load_pca()
    out = RES / "rereal"; out.mkdir(parents=True, exist_ok=True)
    with ra.make_pool(a.workers) as pool:
        for run in ["gen0"] + a.runs.split(","):
            f = out / f"{a.obj}_{run}.npz"
            if f.exists():
                continue
            Z, Fnom = front_Z(a.obj, run)
            Fd = pca.inverse(Z)
            tasks, seeds = [], []
            for i in range(len(Z)):
                for k in range(a.k):
                    tasks.append(Fd[i]); seeds.append(1000 + k)
            recs = ra.evaluate_many(pool, tasks, a.obj == "buck", phase_seeds=seeds)
            arr = lambda key: np.array([r[key] for r in recs]).reshape(len(Z), a.k)
            np.savez(f, Z=Z, Fnom=Fnom, vf=arr("vf"), compliance=arr("compliance"), blf=arr("blf"),
                     connected=arr("connected"))
            print(json.dumps(dict(run=run, n_front=len(Z), k=a.k)), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--workers", type=int, default=10)
    main(ap.parse_args())
