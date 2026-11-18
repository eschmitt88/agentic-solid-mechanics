"""Interim/final table: held-out vs nominal HV gain for every graded front (trial 14 and 14b)."""
import sys, json, glob, os
from pathlib import Path
import numpy as np

def table(E):
    E = Path(E).resolve()
    sys.path.insert(0, str(E))
    for m in ("heldout", "run_arm", "ea", "hf", "dehom"):
        sys.modules.pop(m, None)
    import heldout as H, ea, run_arm as ra
    rows = []
    for f in sorted(glob.glob(str(E / "results/heldout/*__archive*.npz"))):
        name = Path(f).stem; run, tag = name.split("__")
        obj = json.load(open(E / "results/arms" / run / "args.json"))["obj"]
        r = H.heldout_gain(obj, f, E / f"results/heldout/{obj}__gen0.npz")
        a = np.load(E / "results/arms" / run / f"{tag}.npz"); norm = ra.shared_normaliser(obj)
        _, rec0 = ra.load_gen0(obj); F0, cv0 = ra.objectives(rec0, obj, ra.cref_for(obj))
        h0 = ea.hv2d(norm(F0[cv0 <= 0]), H.R11)
        nom = ea.hv2d(norm(a["F"][a["CV"] <= 0]), H.R11) / h0 - 1
        rows.append(dict(run=run, snapshot=tag, gen=int(a["gen"].max()), heldout=r["gain"], nominal=nom, n_front=r["n_front"]))
    sys.path.remove(str(E))
    return rows

if __name__ == "__main__":
    out = []
    for E in sys.argv[1:] or ["."]:
        for r in table(E):
            out.append(r)
            print(f"{r['run']:22s} gen {r['gen']:3d}  held-out {100*r['heldout']:6.1f} %   nominal {100*r['nominal']:6.1f} %   ratio {r['heldout']/r['nominal'] if r['nominal'] else float('nan'):.2f}   n {r['n_front']}")
    json.dump(out, open("results/heldout_summary.json", "w"), indent=1, default=float)
