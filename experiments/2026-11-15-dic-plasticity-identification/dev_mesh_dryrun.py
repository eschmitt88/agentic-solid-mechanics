"""Dry-run of the mesher on the held-out OUTLINES only (nominal geometry files; no held-out test data is read).
Cut lines are guessed from the camera field of view quoted in the paper."""
import numpy as np, data, fem
from meshutil import mesh_outline, clip_loop_y, clean_loop
for key, ylo, yhi in [("X", 38.0, 206.0), ("O15", 37.0, 138.0)]:
    bo, bh = data.load_geometry(key)
    outer = clip_loop_y(clean_loop(bo), ylo, yhi)
    for h, hf in [(2.5, 1.0), (1.25, 0.5)]:
        nodes, el = mesh_outline(outer, bh, h, hf, workdir="results/cache/mesh", name=f"dryrun_{key}_{h}")
        m = fem.Mesh(nodes, el, 1.5)
        area = float(np.asarray(m.wdet).sum() / 1.5)
        print(key, h, hf, "nodes", m.nn, "elems", m.ne, "area %.1f mm2" % area, "bottom nodes", (np.abs(nodes[:, 1] - ylo) < 1e-6).sum(), "top", (np.abs(nodes[:, 1] - yhi) < 1e-6).sum())
