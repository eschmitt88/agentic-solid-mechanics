"""Addendum B extension control: does the EA's low-volume front beat simply running the LF optimiser at
lower volume targets? 240 LF designs at V0 = linspace(0.10, 0.25, 80) x Lmin in {0.10, 0.15, 0.20}
(lf.GEN0_SETTINGS), PCA-reconstructed like every EA design, scored on the 10 held-out seeds.

  python -u extension.py generate --workers 8
  python -u extension.py evaluate --obj stiff|buck --workers 8
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = HERE / "results" / "extension"


def _gen(task):
    import lf
    V0, Lmin = task
    r = lf.lf_optimise(V0, Lmin, **lf.GEN0_SETTINGS)
    return {k: r[k] for k in ("w1", "w2", "theta", "compliance_lf", "vol_lf")}


def generate(workers):
    import multiprocessing as mp
    tasks = [(float(v), L) for L in (0.10, 0.15, 0.20) for v in np.linspace(0.10, 0.25, 80)]
    OUT.mkdir(parents=True, exist_ok=True)
    t = time.time()
    with mp.get_context("spawn").Pool(workers, maxtasksperchild=25) as pool:
        res = pool.map(_gen, tasks, chunksize=1)
    np.savez_compressed(OUT / "lf_low.npz", **{k: np.stack([r[k] for r in res]) for k in res[0]},
                        V0=np.array([t_[0] for t_ in tasks]), Lmin=np.array([t_[1] for t_ in tasks]))
    print(f"generated {len(tasks)} in {time.time() - t:.0f}s", flush=True)


def evaluate(obj, workers):
    import run_arm as ra, heldout as H
    d = np.load(OUT / "lf_low.npz")
    pca = ra.load_pca()
    Z = pca.transform(ra.features(d["w1"], d["w2"], d["theta"]))
    with ra.make_pool(workers) as pool:
        res = H.evaluate_front(pool, Z, "inv5", obj)
    np.savez(OUT / f"heldout_{obj}.npz", Z=Z, Fnom=np.zeros((len(Z), 2)), **res)
    print(f"evaluated {len(Z)} x {len(H.HELDOUT_SEEDS)} ({obj})", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["generate", "evaluate"])
    ap.add_argument("--obj", default="stiff")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    generate(a.workers) if a.cmd == "generate" else evaluate(a.obj, a.workers)
