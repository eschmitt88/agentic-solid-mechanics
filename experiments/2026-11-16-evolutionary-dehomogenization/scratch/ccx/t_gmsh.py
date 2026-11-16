"""gmsh element-size check on the lattice geometry (no ccx)."""
import sys, os, numpy as np, re
from common import hf, lattice, E
sys.path.insert(0, E); import verify_ccx as V
W = os.path.join(os.path.dirname(os.path.abspath(__file__)), "work")
p = hf.Problem(hf.GIRDER, hf.H_FINE); phi = lattice(p, w=0.3)
g, xs, ys, info = V.boundary_field(phi, "phi", r_c=0.0)
lp = V.extract_loops(g, xs, ys)
for ms in (1.0, 0.5):
    loops = []
    for k, P in enumerate([lp["outer"]] + lp["holes"]):
        marks = V._corner_marks(P, 2 * info["hs"])
        R = V.ear_resample(P, ms, marks); loops.append(R)
    seg = np.concatenate([np.hypot(*(np.roll(R, -1, 0) - R).T) for R in loops])
    nmark = sum(V._corner_marks(P, 2 * info["hs"]).sum() for P in [lp["outer"]] + lp["holes"])
    for variant in ("base", "nomin", "alg5", "noext"):
        V.write_geo(os.path.join(W, "g.geo"), loops, ms, ms)
        s = open(os.path.join(W, "g.geo")).read()
        if variant == "nomin": s = re.sub(r"Mesh.MeshSizeMin = [^;]+;", "", s)
        if variant == "alg5": s = s.replace("Mesh.Algorithm = 6;", "Mesh.Algorithm = 5;")
        if variant == "noext": s = s.replace("Mesh.MeshSizeExtendFromBoundary = 1;", "Mesh.MeshSizeExtendFromBoundary = 0;")
        open(os.path.join(W, "g.geo"), "w").write(s)
        rc, log = V.run_gmsh(os.path.join(W, "g.geo"), os.path.join(W, "g.msh"), 4)
        X, T, ar = V.read_msh(os.path.join(W, "g.msh"))
        el = np.hypot(*(X[T[:, 1]] - X[T[:, 0]]).T)
        print(ms, variant, "marks", nmark, "seg median/min", np.median(seg).round(3), seg.min().round(4), "n_el", len(T), "edge median", np.median(el).round(3), "p10", np.percentile(el, 10).round(3))
