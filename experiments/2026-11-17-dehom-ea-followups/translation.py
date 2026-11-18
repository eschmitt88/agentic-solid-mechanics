"""Addendum B item: translation-only realisations (only the global phasor phase varies). Re-evaluates the
held-out fronts of gen 0 (all-480), FULL s0 and PCA s0 (S and B) under 10 such seeds and reports the
per-design optimism of the default realisation, next to the full re-realisation numbers."""
import os, sys, json
os.environ["DEHOM_PHASE_MODE"] = "global"
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ea, run_arm as ra, heldout as H

def front_of(f, obj):
    d = np.load(f); F, conn = H.averaged_objectives(d, obj)
    Z = d["Z"][conn]; fr = ea.nondominated_sort(F) == 0
    return Z[fr], F[fr]

def optimism(d, obj, nominal):
    conn = d["connected"].all(1)
    if obj == "stiff":
        avg = np.exp(np.log(d["compliance"]).mean(1)); r = avg / nominal
    else:
        avg = np.exp(np.log(np.maximum(d["blf"], 1e-12)).mean(1)); r = nominal / avg
    return float(np.median(r[conn])), float(np.percentile(r[conn], 90)), float(np.mean(~conn))

if __name__ == "__main__":
    out = {}
    with ra.make_pool(int(sys.argv[1]) if len(sys.argv) > 1 else 6) as pool:
        for obj in ("stiff", "buck"):
            for src in ("gen0all", "FULL_s0", "PCA_s0"):
                f = HERE / ("results/heldout/" + (f"{obj}__gen0all.npz" if src == "gen0all" else f"{obj}_{src}__archive.npz"))
                Z, F = front_of(f, obj)
                nom = [ra._evaluate((ra.load_pca().inverse(z), "inv5", obj == "buck", [None]))["evals"][0] for z in Z[:0]]
                res = pool.map(ra._evaluate, [(ra.load_pca().inverse(z), "inv5", obj == "buck", [None] + H.HELDOUT_SEEDS) for z in Z], chunksize=1)
                arr = lambda k: np.array([[e[k] for e in r["evals"]] for r in res], float)
                nominal = arr("compliance")[:, 0] if obj == "stiff" else arr("blf")[:, 0]
                d = dict(compliance=arr("compliance")[:, 1:], blf=arr("blf")[:, 1:], connected=arr("connected")[:, 1:])
                out[f"{obj}|{src}"] = dict(n=len(Z), optimism_median_p90_disc=optimism(d, obj, nominal))
                print(f"{obj}|{src}", out[f"{obj}|{src}"], flush=True)
    json.dump(out, open(HERE / "results" / "translation_only.json", "w"), indent=1)
