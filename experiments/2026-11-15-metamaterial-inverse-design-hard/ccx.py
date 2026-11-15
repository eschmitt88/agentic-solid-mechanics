"""Independent verifier: body-fitted Gmsh remesh + CalculiX 2.23 (NLGEOM, neo-Hooke).  Built on trial 9's ccx.py.

New in trial 12:
* any number of shape modes, and a series stack of cells (cell i's shuttle carries cell i+1's walls through
  *EQUATION ties to a reference node; only the top shuttle is displacement-controlled);
* `depth3d`: the same outline extruded to a 3D body of finite depth (C3D15 wedges, symmetry at mid-depth, free
  side face) instead of plane strain — the relaxed-idealisation check;
* `elastic_frame`: wall posts and shuttle meshed in the same elastomer instead of rigid kinematics;
* `imperfection`: adds a small a_2 to every cell (pre-registered fallback when a run aborts).
The full beam is meshed (both halves independently, unstructured); no symmetry in the plane is imposed.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import numpy as np

import fem

MM = os.path.expanduser("~/.local/bin/micromamba")
HERE = Path(__file__).resolve().parent
H_CELL = 25.0


def env(threads):
    return dict(os.environ, MAMBA_ROOT_PREFIX=os.path.expanduser("~/projects/.micromamba"), OMP_NUM_THREADS=str(threads))


def boundary(p, Lh, nk, n=241):
    import jax.numpy as jnp

    s = jnp.linspace(0, 1, n)
    bot = np.asarray(fem.surface_points(jnp.asarray(p), s, -0.5 * jnp.ones(n), Lh, nk))
    top = np.asarray(fem.surface_points(jnp.asarray(p), s, 0.5 * jnp.ones(n), Lh, nk))
    return bot, top


def tmin_of(P, nk):
    import jax.numpy as jnp

    return float(min(np.min(np.asarray(fem.thickness(jnp.asarray(q), jnp.linspace(0, 1, 201), nk))) for q in P))


def write_geo(path, P, Lh, ws, lc, nk, depth3d=None, layers=4, elastic_frame=False, post_w=2.5):
    """P: list of per-cell parameter vectors.  Cell c is drawn at y offset c*H_CELL (purely for separation)."""
    xm = Lh + ws / 2
    L = [f"lc = {lc};"]
    npt, nln, nsurf = [0], [0], [0]

    def pts(arr):
        ids = []
        for x, y in arr:
            npt[0] += 1
            L.append(f"Point({npt[0]}) = {{{x:.12g}, {y:.12g}, 0, lc}};")
            ids.append(npt[0])
        return ids

    def spline(ids):
        nln[0] += 1
        L.append(f"Spline({nln[0]}) = {{{','.join(map(str, ids))}}};")
        return nln[0]

    def line(a, b):
        nln[0] += 1
        L.append(f"Line({nln[0]}) = {{{a}, {b}}};")
        return nln[0]

    def surface(loop):
        nsurf[0] += 1
        L.append(f"Curve Loop({nsurf[0]}) = {{{','.join(map(str, loop))}}};")
        L.append(f"Plane Surface({nsurf[0]}) = {{{nsurf[0]}}};")
        return nsurf[0]

    for c, p in enumerate(P):
        bot, top = boundary(p, Lh, nk)
        y0 = c * H_CELL
        ysh = float(bot[-1, 1] + top[-1, 1]) / 2  # centre-line height at the shuttle
        for side in (0, 1):
            B, T = bot.copy(), top.copy()
            B[:, 1] += y0
            T[:, 1] += y0
            if side == 1:
                B[:, 0] = 2 * xm - B[:, 0]
                T[:, 0] = 2 * xm - T[:, 0]
            ib, it = pts(B), pts(T)
            surface([spline(ib), line(ib[-1], it[-1]), spline(it[::-1]), line(it[0], ib[0])])
            if elastic_frame:
                # wall post: half width post_w (symmetry plane of the lattice at its far face), 10 mm above / 15 mm below
                sx = -1 if side == 0 else 1
                xw = 0.0 if side == 0 else 2 * xm
                q = pts([(xw + sx * post_w, y0 - 15.0), (xw, y0 - 15.0), (xw, y0 + 10.0), (xw + sx * post_w, y0 + 10.0)])
                surface([line(q[0], q[1]), line(q[1], ib[0]), line(ib[0], it[0]), line(it[0], q[2]), line(q[2], q[3]), line(q[3], q[0])])
        if elastic_frame:
            # shuttle block between the two beam end faces, from 2 mm below the beam end to 9.5 mm above (pushed on its top)
            bl, tl = (Lh, y0 + bot[-1, 1]), (Lh, y0 + top[-1, 1])
            q = pts([(Lh, y0 + ysh - 2.0 - 1.0), (Lh + ws, y0 + ysh - 2.0 - 1.0), (Lh + ws, y0 + bot[-1, 1]), (Lh + ws, y0 + top[-1, 1]),
                     (Lh + ws, y0 + ysh + 9.5), (Lh, y0 + ysh + 9.5), tl, bl])
            surface([line(q[i], q[(i + 1) % 8]) for i in range(8)])
    L.append("Coherence;")
    L.append("Mesh.ElementOrder = 2; Mesh.SecondOrderLinear = 0; Mesh.Algorithm = 6;")
    L.append("Mesh.CharacteristicLengthMax = lc; Mesh.CharacteristicLengthMin = lc/2; Mesh.MshFileVersion = 2.2;")
    if depth3d:
        L.append("Mesh.SecondOrderIncomplete = 1;")
        L.append(f"Extrude {{0, 0, {depth3d / 2:.10g}}} {{ Surface{{:}}; Layers{{{layers}}}; Recombine; }}")
    Path(path).write_text("\n".join(L) + "\n")


def read_msh2(path, etype):
    """Minimal ASCII .msh 2.2 reader (meshio has no 15-node wedge).  Returns points (0-based rows) and connectivity."""
    lines = Path(path).read_text().split("\n")
    i = lines.index("$Nodes")
    n = int(lines[i + 1])
    arr = np.array([l.split() for l in lines[i + 2: i + 2 + n]], float)
    tags = arr[:, 0].astype(int)
    pts = np.zeros((tags.max() + 1, 3))
    pts[tags] = arr[:, 1:4]
    j = lines.index("$Elements")
    ne = int(lines[j + 1])
    els = []
    for l in lines[j + 2: j + 2 + ne]:
        t = l.split()
        if int(t[1]) == etype:
            els.append([int(v) for v in t[3 + int(t[2]):]])
    return pts, np.array(els, int)


_W15 = [0, 1, 2, 3, 4, 5, 6, 9, 7, 12, 14, 13, 8, 10, 11]  # gmsh wedge15 -> CalculiX C3D15


def mesh_to_inp(msh, inp, P, Lh, ws, dmax, ninc, depth3d=None, elastic_frame=False, post_w=2.5, tol=1e-6):
    three = depth3d is not None
    allpts, el = read_msh2(msh, 18 if three else 9)  # gmsh type 18 = 15-node prism, 9 = 6-node triangle
    used = np.unique(el)
    remap = -np.ones(len(allpts), int)
    remap[used] = np.arange(len(used)) + 1
    pts = allpts[used]
    el = remap[el]
    if three:
        el = el[:, _W15]
    else:
        a, b, c = pts[el[:, 0] - 1], pts[el[:, 1] - 1], pts[el[:, 2] - 1]
        neg = 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])) < 0
        el[neg] = el[neg][:, [0, 2, 1, 5, 4, 3]]
    xm = Lh + ws / 2
    x, y = pts[:, 0], pts[:, 1]
    ncell = len(P)
    cell_of = np.clip(np.floor((y + 15.5) / H_CELL).astype(int), 0, ncell - 1)
    if elastic_frame:
        # frame is elastic: hold the far face of each wall post (lattice symmetry plane: u_x = 0) and its foot (u_y),
        # push the top face of the shuttle block
        wallx = (np.abs(x + post_w) < tol) | (np.abs(x - (2 * xm + post_w)) < tol)
        ylocal = y - cell_of * H_CELL
        foot = (np.abs(ylocal + 15.0) < tol) & ((x < tol) | (x > 2 * xm - tol))
        shut = np.zeros(len(x), bool)
        for c in range(ncell):
            top_y = np.max(y[(cell_of == c) & (x > Lh + tol) & (x < Lh + ws - tol)])
            shut |= (cell_of == c) & (np.abs(y - top_y) < tol) & (x > Lh - tol) & (x < Lh + ws + tol)
        wall = foot
    else:
        wall = (np.abs(x) < tol) | (np.abs(x - 2 * xm) < tol)
        wallx = wall
        shut = (np.abs(x - Lh) < tol) | (np.abs(x - (Lh + ws)) < tol)
    out = ["*NODE, NSET=NALL"]
    out += [f"{i+1}, {q[0]:.12g}, {q[1]:.12g}, {q[2] if three else 0.:.12g}" for i, q in enumerate(pts)]
    nn = len(pts)
    refs = [nn + 1 + c for c in range(ncell - 1)]  # reference node for the shuttle of cells 0..ncell-2
    out += [f"{r}, {xm:.6g}, {c * H_CELL:.6g}, 0." for c, r in enumerate(refs)]
    out.append(f"*ELEMENT, TYPE={'C3D15' if three else 'CPE6'}, ELSET=EALL")
    out += [f"{i+1}, " + ", ".join(map(str, t)) for i, t in enumerate(el)]

    def nset(name, ids):
        o = [f"*NSET, NSET={name}"]
        ids = list(ids)
        for k in range(0, len(ids), 12):
            o.append(", ".join(map(str, ids[k: k + 12])))
        return o

    ids = lambda mask: np.where(mask)[0] + 1
    top = ncell - 1
    out += nset("WALL0", ids(wall & (cell_of == 0))) + nset("WALLX", ids(wallx)) + nset("SHUTALL", ids(shut)) + nset("SHUTTOP", ids(shut & (cell_of == top)))
    if three:
        out += nset("ZSYM", ids(np.abs(pts[:, 2]) < tol))
    eq = []
    for c in range(ncell - 1):
        for n in ids(shut & (cell_of == c)):  # shuttle of cell c moves with its reference node
            eq += ["2", f"{n}, 2, 1., {refs[c]}, 2, -1."]
        for n in ids(wall & (cell_of == c + 1)):  # walls of cell c+1 ride on the shuttle of cell c
            eq += ["2", f"{n}, 2, 1., {refs[c]}, 2, -1."]
    if eq:
        out += ["*EQUATION"] + eq
    out += ["*BOUNDARY", "WALL0, 2, 2, 0.", "WALLX, 1, 1, 0.", "SHUTALL, 1, 1, 0."]
    if three:
        out += ["ZSYM, 3, 3, 0.", "WALLX, 3, 3, 0.", "SHUTALL, 3, 3, 0."]
    for r in refs:
        out += [f"{r}, 1, 1, 0."] + ([f"{r}, 3, 3, 0."] if three else [])
    out += ["*MATERIAL, NAME=NH", "*HYPERELASTIC, NEO HOOKE", f"{fem.C10:.10g}, {fem.D1:.10g}"]
    out += ["*SOLID SECTION, ELSET=EALL, MATERIAL=NH"] + ([] if three else ["1.0"])
    out += ["*STEP, NLGEOM, INC=100000", "*STATIC", f"{1/ninc:.8g}, 1., 1e-8, {1/ninc:.8g}", "*BOUNDARY", f"SHUTTOP, 2, 2, {-dmax:.10g}"]
    out += ["*NODE PRINT, NSET=SHUTTOP, TOTALS=ONLY", "RF"]
    if refs:
        out += nset("REFS", refs) + ["*NODE PRINT, NSET=REFS", "U"]
    out += ["*END STEP"]
    Path(inp).write_text("\n".join(out) + "\n")
    return dict(nnodes=nn, nelem=len(el), nshut=int(shut.sum()), nwall=int(wall.sum()))


NK_HINT = [4]


def parse_dat(dat, dmax):
    txt = Path(dat).read_text()
    ts, fy = [], []
    for mt in re.finditer(r"total force.*?time\s+([0-9.Ee+-]+)\s*\n\s*\n?\s*([-0-9.Ee+]+)\s+([-0-9.Ee+]+)\s+([-0-9.Ee+]+)", txt):
        ts.append(float(mt.group(1)))
        fy.append(float(mt.group(3)))
    uref = [float(v) for v in re.findall(r"displacements \(vx,vy,vz\) for set REFS and time\s+[0-9.Ee+-]+\s*\n\s*\n\s*\d+\s+[-0-9.Ee+]+\s+([-0-9.Ee+]+)", txt)]
    return np.array(ts) * dmax, -np.array(fy), -np.array(uref)


def run(name, p, nk, ncell=1, Lh=28.0, ws=4.0, dmax=12.5, ninc=100, lc_div=3.0, depth3d=None, layers=4, elastic_frame=False,
        imperfection=0.0, threads=2, workdir=None, keep=False):
    """Returns d, F (per mm depth, full beam), info.  lc = t_min / lc_div."""
    wd = Path(workdir) if workdir else HERE / "results" / "ccx" / name
    wd.mkdir(parents=True, exist_ok=True)
    P = [np.array(q, float) for q in np.reshape(np.asarray(p, float), (ncell, -1))]
    for q in P:
        q[1] += imperfection
    NK_HINT[0] = nk
    tmin = tmin_of(P, nk)
    lc = tmin / lc_div
    geo, msh, inp = wd / "job.geo", wd / "job.msh", wd / "job.inp"
    for f in (msh, wd / "job.dat"):
        if f.exists():
            f.unlink()
    write_geo(geo, P, Lh, ws, lc, nk, depth3d=depth3d, layers=layers, elastic_frame=elastic_frame)
    r = subprocess.run([MM, "run", "-n", "solidmech", "gmsh", "-3" if depth3d else "-2", str(geo), "-o", str(msh), "-format", "msh2"],
                       capture_output=True, text=True, env=env(threads))
    if not msh.exists():
        raise RuntimeError("gmsh failed: " + r.stdout[-2000:] + r.stderr[-2000:])
    info = mesh_to_inp(msh, inp, P, Lh, ws, dmax, ninc, depth3d=depth3d, elastic_frame=elastic_frame)
    r = subprocess.run([MM, "run", "-n", "solidmech", "ccx", "job"], cwd=wd, capture_output=True, text=True, env=env(threads))
    (wd / "ccx.stdout.txt").write_text(r.stdout[-6000:])
    d, F, dref = parse_dat(wd / "job.dat", dmax)
    if depth3d:
        F = 2 * F / depth3d  # half depth modelled -> per mm depth
    info.update(lc=lc, tmin=tmin, completed=bool(len(d) and abs(d[-1] - dmax) < 1e-6 * dmax), n_out=len(d), d_reached=float(d[-1]) if len(d) else 0.0)
    for ext in ["frd", "msh", "12d", "cvg", "sta"] + ([] if keep else ["inp", "geo", "dat"]):
        f = wd / f"job.{ext}"
        if f.exists():
            f.unlink()
    np.savetxt(wd / "curve.csv", np.c_[d, F], delimiter=",", header="d_mm,F_N_per_mm", comments="")
    if len(dref):
        info["d_cellA"] = dref.tolist()
    return d, F, info


def sample(d, F, grid, completed):
    out = np.full(len(grid), np.nan)
    if len(d):
        ok = grid <= d[-1] + 1e-9
        out[ok] = np.interp(grid[ok], np.r_[0.0, d], np.r_[0.0, F])
    return out
