"""Solid girder and lattice w0.30 (fillet 0.5): BLF vs mesh with a soft pad edge (k_taper mm)."""
import json, sys, numpy as np
from common import hf, lattice, E
sys.path.insert(0, E); import verify_ccx as V
p = hf.Problem(hf.GIRDER, hf.H_FINE)
tap = float(sys.argv[1])
cfg = dict(hf.GIRDER, k_taper=tap)
for name, d, rc in (("solid", np.ones((p.nely, p.nelx), bool), 0.0), ("lat_w0.30", lattice(p, w=0.3), 0.5)):
    for ms in (1.0, 0.5, 0.25):
        if name == "solid" and ms == 0.25: continue
        r = V.verify(d, kind="phi", r_c=rc, mesh_size=ms, cfg=cfg, threads=2, nev=2, static_r_c=None)
        print(json.dumps(dict(taper=tap, design=name, r_c=rc, mesh=ms, C=r["compliance"], blf=r["blf"], modes=r["blf_modes"],
                              mode_xy=r.get("mode_xy"), ssum=r["spring_sum_y"], wall=r["wall"])), flush=True)
