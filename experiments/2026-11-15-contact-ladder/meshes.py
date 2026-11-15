"""Mesh helpers: gmsh CLI (all-quad), boundary extraction, CalculiX deck pieces."""
import os, subprocess, tempfile
import numpy as np
import meshio

GMSH = [os.path.expanduser("~/.local/bin/micromamba"), "run", "-n", "solidmech", "gmsh"]
ENV = dict(os.environ, MAMBA_ROOT_PREFIX=os.path.expanduser("~/projects/.micromamba"))


def run_gmsh(geo_text, workdir, name, order=1):
    os.makedirs(workdir, exist_ok=True)
    geo = os.path.join(workdir, name + ".geo"); msh = os.path.join(workdir, name + ".msh")
    open(geo, "w").write(geo_text)
    r = subprocess.run(GMSH + [geo, "-2", "-order", str(order), "-format", "msh22", "-o", msh, "-nt", "4"],
                       env=ENV, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(msh):
        raise RuntimeError(r.stdout[-2000:] + r.stderr[-2000:])
    m = meshio.read(msh)
    X = m.points[:, :2]
    quads = np.vstack([c.data for c in m.cells if c.type == "quad"])
    # physical surface tag per quad
    tags = np.concatenate([d for c, d in zip(m.cells, m.cell_data["gmsh:physical"]) if c.type == "quad"])
    used = np.unique(quads)
    remap = -np.ones(len(X), int); remap[used] = np.arange(len(used))
    X = X[used]; quads = remap[quads]
    # make all elements counter-clockwise
    a = X[quads]
    area = 0.5 * np.sum(a[:, :, 0] * np.roll(a[:, :, 1], -1, axis=1) - np.roll(a[:, :, 0], -1, axis=1) * a[:, :, 1], axis=1)
    quads[area < 0] = quads[area < 0][:, ::-1]
    return X, quads, tags


QUAD_OPTS = """
Mesh.Algorithm = 8; Mesh.RecombineAll = 1; Mesh.RecombinationAlgorithm = 2;
Mesh.SubdivisionAlgorithm = 1; Mesh.Smoothing = 10;
"""


def refine_field(px, py, h_c, r_in, h_far, r_out, fid=1):
    return f"""
Point(9000+{fid}) = {{{px}, {py}, 0, {h_c}}};
Field[{fid}] = Distance; Field[{fid}].PointsList = {{9000+{fid}}};
Field[{fid}+50] = Threshold; Field[{fid}+50].InField = {fid}; Field[{fid}+50].SizeMin = {2*h_c}; Field[{fid}+50].SizeMax = {2*h_far};
Field[{fid}+50].DistMin = {r_in}; Field[{fid}+50].DistMax = {r_out};
"""


def half_disc_geo(R, h_c, r_in, h_far, r_out, tag=1, y0=0.0, quarter=False):
    """half disc, lowest point at (0, y0), flat face on top at y = y0 + R.
    quarter=True: right half only (axis of symmetry x = 0)."""
    if quarter:
        return f"""
Point(1) = {{0, {y0}, 0}}; Point(2) = {{{R}, {y0+R}, 0}}; Point(3) = {{0, {y0+R}, 0}};
Circle(1) = {{1, 3, 2}}; Line(2) = {{2, 3}}; Line(3) = {{3, 1}};
Curve Loop(1) = {{1, 2, 3}}; Plane Surface(1) = {{1}}; Physical Surface({tag}) = {{1}};
""" + refine_field(0, y0, h_c, r_in, h_far, r_out) + "Background Field = 51;\nMesh.MeshSizeExtendFromBoundary = 0; Mesh.MeshSizeFromPoints = 0; Mesh.MeshSizeFromCurvature = 0;\n" + QUAD_OPTS
    return f"""
Point(1) = {{0, {y0}, 0}}; Point(2) = {{{R}, {y0+R}, 0}}; Point(3) = {{0, {y0+R}, 0}}; Point(4) = {{{-R}, {y0+R}, 0}};
Circle(1) = {{4, 3, 1}}; Circle(2) = {{1, 3, 2}}; Line(3) = {{2, 3}}; Line(4) = {{3, 4}};
Curve Loop(1) = {{1, 2, 3, 4}}; Plane Surface(1) = {{1}}; Physical Surface({tag}) = {{1}};
""" + refine_field(0, y0, h_c, r_in, h_far, r_out) + "Background Field = 51;\nMesh.MeshSizeExtendFromBoundary = 0; Mesh.MeshSizeFromPoints = 0; Mesh.MeshSizeFromCurvature = 0;\n" + QUAD_OPTS


def block_geo(W, H, h_c, r_in, h_far, r_out, tag=2):
    """block [-W/2, W/2] x [-H, 0], refined near the origin."""
    return f"""
Point(1) = {{{-W/2}, {-H}, 0}}; Point(2) = {{{W/2}, {-H}, 0}}; Point(3) = {{{W/2}, 0, 0}}; Point(4) = {{0, 0, 0}}; Point(5) = {{{-W/2}, 0, 0}};
Line(1) = {{1, 2}}; Line(2) = {{2, 3}}; Line(3) = {{3, 4}}; Line(4) = {{4, 5}}; Line(5) = {{5, 1}};
Curve Loop(1) = {{1, 2, 3, 4, 5}}; Plane Surface(1) = {{1}}; Physical Surface({tag}) = {{1}};
""" + refine_field(0, 0, h_c, r_in, h_far, r_out) + "Background Field = 51;\nMesh.MeshSizeExtendFromBoundary = 0; Mesh.MeshSizeFromPoints = 0; Mesh.MeshSizeFromCurvature = 0;\n" + QUAD_OPTS


def boundary_edges(quads):
    """edges on the boundary, oriented as in their (CCW) element: body on the left.
    returns (edges (n,2), element index, local face index 0..3)"""
    e = np.stack([quads, np.roll(quads, -1, axis=1)], axis=2).reshape(-1, 2)
    el = np.repeat(np.arange(len(quads)), 4); fa = np.tile(np.arange(4), len(quads))
    key = np.sort(e, axis=1)
    _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    b = cnt[inv.ravel()] == 1
    return e[b], el[b], fa[b]


def chain(edges):
    """order a set of connected edges into a node chain following edge orientation."""
    nxt = {a: b for a, b in edges}
    starts = set(nxt) - set(nxt.values())
    assert len(starts) == 1, f"edges do not form one open chain ({len(starts)} starts)"
    n = starts.pop(); out = [n]
    while n in nxt:
        n = nxt[n]; out.append(n)
    return np.array(out)


def structured_rect(x0, x1, y0, y1, nx, ny, gx=None, gy=None):
    xs = np.linspace(x0, x1, nx + 1) if gx is None else gx
    ys = np.linspace(y0, y1, ny + 1) if gy is None else gy
    nx, ny = len(xs) - 1, len(ys) - 1
    X = np.stack(np.meshgrid(xs, ys, indexing="xy"), axis=-1).reshape(-1, 2)
    i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="xy")
    n0 = (j * (nx + 1) + i).ravel()
    quads = np.stack([n0, n0 + 1, n0 + nx + 2, n0 + nx + 1], axis=1)
    return X, quads


def merge(parts):
    """parts: list of (X, quads) -> X, quads, list of node offsets, element body id"""
    Xs, Qs, off, body = [], [], [], []
    n = 0
    for k, (X, q) in enumerate(parts):
        Xs.append(X); Qs.append(q + n); off.append(n); body.append(np.full(len(q), k)); n += len(X)
    return np.vstack(Xs), np.vstack(Qs), off, np.concatenate(body)
