"""Held-out grading for trial 14 (the primary metric of every hypothesis).

  python -u heldout.py --obj stiff|buck --runs stiff_PCA_s0,stiff_FULL_s0 [--archive archive.npz] --workers 12
  python -u heldout.py --obj stiff|buck --gen0                         # the shared baseline front

For a run, take the feasible non-dominated set of its archive (under the objective it was searched
on), re-evaluate every member under the 10 held-out phasor seeds HELDOUT_SEEDS (never used in any
search), and store all samples. Realisation-averaged objectives: mean vf, geometric-mean compliance,
geometric-mean BLF; a design with any disconnected realisation is infeasible. No stiffness constraint
in the held-out buckling HV (pre-registered; see trial-13 log 20:45 for why).

Held-out HV gain of a run = HV(gen-0 front ∪ run front) / HV(gen-0 front) − 1, both held-out,
normalised by the shared nominal gen-0 normaliser (run_arm.shared_normaliser), r = (1.1, 1.1).
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
OUT = RES / "heldout"
HELDOUT_SEEDS = list(range(5000, 5010))
R11 = np.array([1.1, 1.1])


def run_front(obj, run, archive="archive.npz"):
    a = np.load(RES / "arms" / run / archive)
    args = json.load(open(RES / "arms" / run / "args.json"))
    enc = ra.ARM_ENC.get(args["arm"], "inv5")
    ok = np.where(a["CV"] <= 0)[0]
    idx = ok[ea.nondominated_sort(a["F"][ok]) == 0]
    return a["Z"][idx], a["F"][idx], enc


def gen0_front(obj):
    Z0, rec = ra.load_gen0(obj, "inv5")
    F, cv = ra.objectives(rec, obj, ra.cref_for(obj))
    ok = np.where(cv <= 0)[0]
    idx = ok[ea.nondominated_sort(F[ok]) == 0]
    return Z0[idx], F[idx], "inv5"


def evaluate_front(pool, Z, enc, obj):
    pca = ra.load_pca(enc)
    Fd = pca.inverse(Z)
    res = pool.map(ra._evaluate, [(f, enc, obj == "buck", HELDOUT_SEEDS) for f in Fd], chunksize=1)
    arr = lambda k: np.array([[e[k] for e in r["evals"]] for r in res], float)
    return dict(vf=arr("vf"), compliance=arr("compliance"), blf=arr("blf"), connected=arr("connected"))


def averaged_objectives(d, obj):
    conn = d["connected"].all(1)
    vf = d["vf"].mean(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        C = np.where(conn, np.exp(np.log(d["compliance"]).mean(1)), np.inf)
        B = np.where(conn, np.exp(np.log(np.maximum(d["blf"], 1e-12)).mean(1)), 0.0)
    F = np.stack([vf, C if obj == "stiff" else -B], 1)
    return F[conn], conn


def heldout_gain(obj, run_file, gen0_file):
    norm = ra.shared_normaliser(obj)
    g = np.load(gen0_file); r = np.load(run_file)
    Fg, _ = averaged_objectives(g, obj); Fr, _ = averaged_objectives(r, obj)
    h0 = ea.hv2d(norm(Fg), R11)
    h1 = ea.hv2d(norm(np.concatenate([Fg, Fr])), R11)
    return dict(hv_gen0=h0, hv_run=h1, gain=h1 / h0 - 1, n_front=int(len(r["vf"])))


def main(a):
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = []
    if a.gen0:
        Z, F, enc = gen0_front(a.obj)
        jobs.append((OUT / f"{a.obj}__gen0.npz", Z, F, enc))
    for run in (a.runs.split(",") if a.runs else []):
        Z, F, enc = run_front(a.obj, run, a.archive)
        tag = a.archive.replace(".npz", "")
        jobs.append((OUT / f"{run}__{tag}.npz", Z, F, enc))
    with ra.make_pool(a.workers) as pool:
        for f, Z, F, enc in jobs:
            if f.exists():
                continue
            d = evaluate_front(pool, Z, enc, a.obj)
            np.savez(f, Z=Z, Fnom=F, **d)
            print(json.dumps(dict(file=f.name, n_front=len(Z))), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", required=True)
    ap.add_argument("--runs", default="")
    ap.add_argument("--archive", default="archive.npz")
    ap.add_argument("--gen0", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    main(ap.parse_args())
