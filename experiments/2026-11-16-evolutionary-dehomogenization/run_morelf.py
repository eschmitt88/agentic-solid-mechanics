"""MORELF control (matched HF budget, no evolutionary search): generate extra LF
designs with varied LF parameters, evaluate them in the same HF pipeline.

  python -u run_morelf.py generate --n 11000 --workers 16      # LF designs -> results/morelf/lf_extra.npz
  python -u run_morelf.py evaluate --obj stiff|buck --n N --workers 16

Diversity sources (OURS, from the Fable review): V0 ~ U(0.25, 0.50), Lmin in
{0.10, 0.15, 0.20}, rmin in {1.5, 2.0, 2.5, 3.0} coarse elements (quantised: each new
rmin JIT-compiles new filter functions and a continuous draw OOM-killed the first run), theta_jitter ~ U(0, 0.3) rad,
init_jitter ~ U(0, 0.1), MMA seed. All other LF settings = lf.GEN0_SETTINGS.
Each design is evaluated raw (primary: the practical alternative to the EA) and
PCA-projected (secondary: same representation as the EA's designs).
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
OUT = HERE / "results" / "morelf"


def _gen_one(p):
    import lf
    r = lf.lf_optimise(p["V0"], p["Lmin"], seed=p["seed"], rmin=p["rmin"], theta_jitter=p["theta_jitter"],
                       init_jitter=p["init_jitter"], **lf.GEN0_SETTINGS)
    return {k: r[k] for k in ("w1", "w2", "theta", "compliance_lf", "vol_lf")}


def generate(n, workers, seed=12345):
    import multiprocessing as mp
    rng = np.random.default_rng(seed)
    ps = [dict(V0=float(rng.uniform(0.25, 0.50)), Lmin=float(rng.choice([0.10, 0.15, 0.20])),
               rmin=float(rng.choice([1.5, 2.0, 2.5, 3.0])), theta_jitter=float(rng.uniform(0, 0.3)),
               init_jitter=float(rng.uniform(0, 0.1)), seed=int(i)) for i in range(n)]
    OUT.mkdir(parents=True, exist_ok=True)
    t = time.time()
    with mp.get_context("spawn").Pool(workers, maxtasksperchild=25) as pool:
        res = pool.map(_gen_one, ps, chunksize=1)
    np.savez_compressed(OUT / "lf_extra.npz", **{k: np.stack([r[k] for r in res]) for k in res[0]},
                        params=json.dumps(ps))
    print(f"generated {n} in {time.time() - t:.0f}s", flush=True)


def evaluate(obj, n, workers, kinds=("raw", "proj")):
    import run_arm as ra
    d = np.load(OUT / "lf_extra.npz")
    F = ra.features(d["w1"][:n], d["w2"][:n], d["theta"][:n])
    pca = ra.load_pca()
    Fp = pca.inverse(pca.transform(F))
    buck = obj == "buck"
    t = time.time()
    res = {}
    with ra.make_pool(workers) as pool:
        if "raw" in kinds:
            res["raw"] = ra.evaluate_many(pool, list(F), buck)
        if "proj" in kinds:
            res["proj"] = ra.evaluate_many(pool, list(Fp), buck)
    np.savez_compressed(OUT / f"eval_{obj}.npz", **{f"{kind}_{k}": np.array([r[k] for r in recs])
                                                     for kind, recs in res.items() for k in recs[0]})
    print(f"evaluated {n} x2 ({obj}) in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["generate", "evaluate"])
    ap.add_argument("--obj", choices=["stiff", "buck"])
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--kinds", default="raw,proj")
    a = ap.parse_args()
    generate(a.n, a.workers) if a.cmd == "generate" else evaluate(a.obj, a.n, a.workers, tuple(a.kinds.split(",")))
