"""Buckling-evaluation throughput vs number of pool workers (decides the trial-14 compute plan)."""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import run_arm as ra  # noqa: E402

if __name__ == "__main__":
    Z0, _ = ra.load_gen0("buck")
    pca = ra.load_pca()
    Fr = pca.inverse(Z0[:48])
    for w in (4, 8, 16, 24):
        with ra.make_pool(w) as pool:
            pool.map(ra._evaluate, [(Fr[0], "inv5", True, [None])] * w)
            t = time.time()
            r = ra.evaluate_many(pool, list(Fr), "inv5", True)
            dt = time.time() - t
        print(f"workers {w:2d}: {48 / dt:.2f} buckling evals/s, mean per-eval {np.mean([x['t'] for x in r]):.2f}s", flush=True)
