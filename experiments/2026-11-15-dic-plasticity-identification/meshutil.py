"""Mesh helpers: structured T6 rectangle, gmsh-CLI mesher for a polygon outline with holes."""
import os, subprocess, tempfile
import numpy as np

GMSH = ["env", "MAMBA_ROOT_PREFIX=" + os.path.expanduser("~/projects/.micromamba"),
        os.path.expanduser("~/.local/bin/micromamba"), "run", "-n", "solidmech", "gmsh"]


def rect_t6(Lx, Ly, nx, ny, x0=0.0, y0=0.0):
    xs = np.linspace(0, Lx, 2 * nx + 1) + x0
    ys = np.linspace(0, Ly, 2 * ny + 1) + y0
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    nid = np.arange(X.size).reshape(X.shape)
    nodes = np.stack([X.ravel(), Y.ravel()], 1)
    el = []
    for i in range(nx):
        for j in range(ny):
            a, b = 2 * i, 2 * j
            n = lambda di, dj: nid[a + di, b + dj]
            el.append([n(0, 0), n(2, 0), n(2, 2), n(1, 0), n(2, 1), n(1, 1)])
            el.append([n(0, 0), n(2, 2), n(0, 2), n(1, 1), n(1, 2), n(0, 1)])
    return nodes, np.array(el)


def _split_corners(pts, ang_deg=25.0, bend_tol=2e-3):
    """Break points of a closed polygon: sharp corners plus straight<->curved transitions.
    Returns (break indices, is_bend per vertex)."""
    d1 = pts - np.roll(pts, 1, 0)
    d2 = np.roll(pts, -1, 0) - pts
    a1 = np.arctan2(d1[:, 1], d1[:, 0])
    a2 = np.arctan2(d2[:, 1], d2[:, 0])
    turn = np.abs((a2 - a1 + np.pi) % (2 * np.pi) - np.pi)
    bend = turn > bend_tol
    corner = turn > np.deg2rad(ang_deg)
    trans = bend & (~np.roll(bend, 1) | ~np.roll(bend, -1))  # first/last bend vertex of a curved run
    return np.where(corner | trans)[0], bend


def clean_loop(pts, tol=1e-6):
    pts = np.asarray(pts, float)
    if np.linalg.norm(pts[0] - pts[-1]) < tol:
        pts = pts[:-1]
    keep = [0]
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i] - pts[keep[-1]]) > tol:
            keep.append(i)
    pts = pts[keep]
    # remove zero-area spikes (the nominal outlines overlap themselves by ~0.1 mm at the loop seam)
    changed = True
    while changed:
        d1 = pts - np.roll(pts, 1, 0); d2 = np.roll(pts, -1, 0) - pts
        c = (d1 * d2).sum(1) / (np.linalg.norm(d1, axis=1) * np.linalg.norm(d2, axis=1) + 1e-300)
        spike = c < -0.99
        changed = spike.any()
        if changed:
            pts = pts[~spike]
    return pts


def clip_loop_y(pts, ylo, yhi):
    """Sutherland-Hodgman clip of a closed polygon to ylo <= y <= yhi."""
    def clip(poly, yc, keep_above):
        out = []
        n = len(poly)
        for i in range(n):
            a, b = poly[i], poly[(i + 1) % n]
            ina = (a[1] >= yc) if keep_above else (a[1] <= yc)
            inb = (b[1] >= yc) if keep_above else (b[1] <= yc)
            if ina:
                out.append(a)
            if ina != inb:
                t = (yc - a[1]) / (b[1] - a[1])
                out.append(a + t * (b - a))
        return np.array(out)
    return clean_loop(clip(clip(np.asarray(pts, float), ylo, True), yhi, False))


def mesh_outline(outer, holes, h, h_fine=None, d_fine=4.0, workdir=None, name="mesh", order=2):
    """Mesh polygon `outer` minus `holes` with T6 triangles via the gmsh CLI.
    Smooth runs of points become splines, corners stay sharp.  h: far-field size,
    h_fine: size on curved (non-straight) boundary parts and holes, grading over d_fine.
    Returns nodes (nn,2), elems (ne,6)."""
    import meshio
    h_fine = h if h_fine is None else h_fine
    workdir = workdir or tempfile.mkdtemp(prefix="gmsh_")
    os.makedirs(workdir, exist_ok=True)
    geo = [f'SetFactory("Built-in");']
    pid = [0]; cid = [0]
    loops = []; curved = []

    def add_loop(pts):
        pts = clean_loop(pts)
        corners, bend = _split_corners(pts)
        n = len(pts)
        ids = []
        for q in pts:
            pid[0] += 1
            geo.append(f"Point({pid[0]}) = {{{q[0]:.9g}, {q[1]:.9g}, 0, {h:.6g}}};")
            ids.append(pid[0])
        curves = []
        if len(corners) < 2:
            cs = [0, n // 4, n // 2, 3 * n // 4]
        else:
            cs = list(corners)
        for a, b in zip(cs, cs[1:] + [cs[0] + n]):
            inner = [i % n for i in range(a + 1, b)]
            cid[0] += 1
            if len(inner) == 0 or not bend[inner].any():
                geo.append(f"Line({cid[0]}) = {{{ids[a % n]}, {ids[b % n]}}};")
            else:
                seg = [ids[i % n] for i in range(a, b + 1)]
                geo.append(f"Spline({cid[0]}) = {{{', '.join(map(str, seg))}}};")
                curved.append(cid[0])
            curves.append(cid[0])
        cid[0] += 1
        geo.append(f"Curve Loop({cid[0]}) = {{{', '.join(map(str, curves))}}};")
        loops.append(cid[0])

    add_loop(outer)
    for hh in holes:
        add_loop(hh)
    geo.append(f"Plane Surface(1) = {{{', '.join(map(str, loops))}}};")
    if curved and h_fine < h:
        geo.append(f"Field[1] = Distance; Field[1].CurvesList = {{{', '.join(map(str, curved))}}}; Field[1].Sampling = 200;")
        geo.append(f"Field[2] = Threshold; Field[2].InField = 1; Field[2].SizeMin = {h_fine}; Field[2].SizeMax = {h}; "
                   f"Field[2].DistMin = {0.5 * d_fine}; Field[2].DistMax = {3 * d_fine};")
        geo.append("Background Field = 2;")
    geo.append("Mesh.MeshSizeExtendFromBoundary = 0; Mesh.MeshSizeFromPoints = 0; Mesh.MeshSizeFromCurvature = 0;")
    geo.append(f"Mesh.MeshSizeMax = {h}; Mesh.Algorithm = 6; Mesh.ElementOrder = {order}; Mesh.SecondOrderLinear = 0;")
    geo.append("Mesh.HighOrderOptimize = 1; Mesh.MshFileVersion = 2.2;")
    gp = os.path.join(workdir, name + ".geo")
    mp = os.path.join(workdir, name + ".msh")
    if not os.path.exists(mp):  # meshes are cached: every run of a given specimen/size uses the same file
        tmp = os.path.join(workdir, f"{name}.{os.getpid()}.tmp")
        open(tmp + ".geo", "w").write("\n".join(geo) + "\n")
        r = subprocess.run(GMSH + [tmp + ".geo", "-2", "-format", "msh2", "-o", tmp + ".msh", "-v", "2"], capture_output=True, text=True)
        if not os.path.exists(tmp + ".msh"):
            raise RuntimeError(r.stdout + r.stderr)
        os.replace(tmp + ".geo", gp)
        os.replace(tmp + ".msh", mp)
    msh = meshio.read(mp)
    tri = [c.data for c in msh.cells if c.type == "triangle6"][0]
    nodes = msh.points[:, :2]
    used = np.unique(tri)
    remap = -np.ones(len(nodes), int); remap[used] = np.arange(len(used))
    nodes = nodes[used]; tri = remap[tri]
    # make orientation positive
    a = nodes[tri[:, 0]]; b = nodes[tri[:, 1]]; c = nodes[tri[:, 2]]
    area = 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    neg = area < 0
    if neg.any():
        t = tri.copy()
        t[neg] = tri[neg][:, [0, 2, 1, 5, 4, 3]]
        tri = t
    return nodes, tri
