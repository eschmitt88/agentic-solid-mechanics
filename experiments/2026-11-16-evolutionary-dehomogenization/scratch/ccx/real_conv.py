"""Mesh convergence of the ccx BLF on saved dehomogenised gen-0 designs (real_<i>.npz, level path),
fillet rounding r_c x mesh; appends to real_conv.jsonl.  usage: real_conv.py idx,.. r_c,.. mesh,.."""
import json, os, sys, numpy as np
from common import E
sys.path.insert(0, E); import verify_ccx as V
HERE = os.path.dirname(os.path.abspath(__file__))
idx = sys.argv[1].split(","); rcs = [float(v) for v in sys.argv[2].split(",")]; meshes = [float(v) for v in sys.argv[3].split(",")]
for i in idx:
    lev = np.load(os.path.join(HERE, f"real_{i}.npz"))["level"]
    for rc in rcs:
        for ms in meshes:
            r = V.verify(lev, kind="level", r_c=rc, mesh_size=ms, threads=3, nev=2, rounding="fillet", static_r_c=None)
            rec = {k: v for k, v in r.items() if not isinstance(v, np.ndarray) and k not in ("gmsh_log",)}
            rec["idx"] = i
            print(f"real_{i} r_c={rc} mesh={ms} C={r.get('compliance'):.3f} BLF={r.get('blf'):.4f} mode@{np.round(r.get('mode_xy', [np.nan]*2), 2)} "
                  f"vf={r.get('vf_ccx'):.4f} nel={r.get('n_elements')} t={r['wall']:.0f}", flush=True)
            with open(os.path.join(HERE, "real_conv.jsonl"), "a") as f:
                f.write(json.dumps(rec, default=float) + "\n")
