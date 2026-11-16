"""verify_ccx on dehomogenised gen-0 designs (level path): hf vs ccx + wall time.
usage: real_designs.py r_c mesh idx,idx,..."""
import json, os, sys, time, numpy as np
from common import hf, E, hf_eval_mode
sys.path.insert(0, E)
import verify_ccx as V
import dehom
HERE = os.path.dirname(os.path.abspath(__file__))
rc = float(sys.argv[1]); ms = float(sys.argv[2]); idx = [int(v) for v in sys.argv[3].split(",")]
D = np.load(os.path.join(E, "results", "gen0", sys.argv[4] if len(sys.argv) > 4 else "lf_designs_robust.npz"))
p = hf.Problem(hf.GIRDER, hf.H_FINE)
for i in idx:
    t = time.time(); dh = dehom.dehomogenise(D["w1"][i], D["w2"][i], D["theta"][i]); td = time.time() - t
    np.savez_compressed(os.path.join(HERE, f"real_{i}.npz"), level=dh["level"], phi=dh["phi"])
    rh = hf_eval_mode(p, dh["phi"])
    r = V.verify(dh["level"], kind="level", r_c=rc, mesh_size=ms, threads=4, nev=2)
    rec = dict(idx=i, V0=float(D["V0"][i]), Lmin=float(D["Lmin"][i]), t_dehom=td, hf=rh,
               ccx={k: v for k, v in r.items() if not isinstance(v, np.ndarray) and k not in ("gmsh_log",)})
    print(json.dumps(dict(idx=i, V0=rec["V0"], Lmin=rec["Lmin"], hf_C=rh["compliance"], hf_blf=rh["blf"], hf_mode=rh["mode_xy"],
                          vf_hf=rh["vf"], ccx_C=r.get("compliance"), ccx_blf=r.get("blf"), ccx_mode=r.get("mode_xy"),
                          vf_ccx=r.get("vf_ccx"), nel=r.get("n_elements"), ok=r.get("ok"), wall=r.get("wall"),
                          reason=r.get("reason")), default=float), flush=True)
    with open(os.path.join(HERE, "real_designs.jsonl"), "a") as f:
        f.write(json.dumps(rec, default=float) + "\n")
