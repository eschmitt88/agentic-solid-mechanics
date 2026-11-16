"""CalculiX calibration runs: design x r_c x mesh. Appends one JSON line per run to
scratch/ccx/ccx_checks.jsonl.  usage: ccx_checks.py <design> <r_c,...> <mesh,...> [kind]"""
import json, os, sys, time, numpy as np
from common import hf, lattice, lattice_level, E
sys.path.insert(0, E); import verify_ccx as V
HERE = os.path.dirname(os.path.abspath(__file__))
name = sys.argv[1]; rcs = [float(v) for v in sys.argv[2].split(",")]; meshes = [float(v) for v in sys.argv[3].split(",")]
kind = sys.argv[4] if len(sys.argv) > 4 else "phi"
rounding = sys.argv[5] if len(sys.argv) > 5 else "fillet"
OUT = sys.argv[6] if len(sys.argv) > 6 else "ccx_checks_fillet.jsonl"
p = hf.Problem(hf.GIRDER, hf.H_FINE)
if name == "solid":
    d = np.ones((p.nely, p.nelx), bool)
else:
    w = float(name.split("w")[1])
    d = lattice(p, w=w) if kind == "phi" else lattice_level(p, w=w)
for rc in rcs:
    for ms in meshes:
        r = V.verify(d, kind=kind, r_c=rc, mesh_size=ms, threads=4, nev=2, rounding=rounding, static_r_c=None)
        rec = {k: v for k, v in r.items() if not isinstance(v, np.ndarray) and k not in ("gmsh_log",)}
        rec["design"] = name; rec["rounding"] = rounding
        print(json.dumps({k: rec.get(k) for k in ["design", "kind", "r_c", "mesh_size", "compliance", "blf", "blf_modes", "mode_xy",
                                                   "mode_pr", "vf_ccx", "vf_pix", "n_elements", "eps_shape", "eps_seg", "eps_acc",
                                                   "ok", "t_ccx", "wall", "reason"]}, default=float), flush=True)
        with open(os.path.join(HERE, OUT), "a") as f:
            f.write(json.dumps(rec, default=float) + "\n")
