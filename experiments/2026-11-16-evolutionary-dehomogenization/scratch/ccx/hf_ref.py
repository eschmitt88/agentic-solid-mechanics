"""hf.py reference numbers (compliance, BLF with GIRDER's kg_filter_r, mode location) for the
calibration designs at h = 0.25 and 0.125. Writes scratch/ccx/hf_ref.json."""
import json, os, time, numpy as np
from common import hf, lattice, hf_eval_mode
HERE = os.path.dirname(os.path.abspath(__file__))
out = {}
for h in (0.25, 0.125):
    p = hf.Problem(hf.GIRDER, h)
    for name, phi in (("solid", np.ones((p.nely, p.nelx), bool)),
                      ("lat_w0.30", lattice(p, w=0.30)), ("lat_w0.15", lattice(p, w=0.15))):
        t = time.time(); r = hf_eval_mode(p, phi); r["t"] = time.time() - t
        out[f"{name}@{h}"] = r
        print(name, h, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
json.dump(out, open(os.path.join(HERE, "hf_ref.json"), "w"), indent=1)
