"""Independent verifier: body-fitted Gmsh remesh + CalculiX 2.23 (NLGEOM, neo-Hooke, CPE6).

The full beam is meshed (both halves, independently, unstructured 6-node
triangles); no symmetry is imposed.  The stiff shuttle is represented by its
kinematics: the two beam faces that bond to it are displaced vertically with no
rotation; `lateral='fixed'` also holds u_x = 0 there (pre-registered verifier),
`lateral='free'` ties all shuttle-face nodes to one free lateral translation.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import numpy as np

import fem

MM = os.path.expanduser("~/.local/bin/micromamba")
ENV = dict(os.environ, MAMBA_ROOT_PREFIX=os.path.expanduser("~/projects/.micromamba"), OMP_NUM_THREADS="4")
HERE = Path(__file__).resolve().parent


def boundary(p, Lh, n=241):
    import jax.numpy as jnp

    s = jnp.linspace(0, 1, n)
    bot = np.asarray(fem.surface_points(jnp.asarray(p), s, -0.5 * jnp.ones(n), Lh))
    top = np.asarray(fem.surface_points(jnp.asarray(p), s, 0.5 * jnp.ones(n), Lh))
    return bot, top


def write_geo(path, p, Lh, ws, lc):
    """Two half-beams: left one from the wall (x=0) to the shuttle face (x=Lh); right one mirrored."""
    bot, top = boundary(p, Lh)
    xm = Lh + ws / 2
    lines = [f"lc = {lc};"]
    pid = [0]

    def add_pts(P):
        ids = []
        for x, y in P:
            pid[0] += 1
            lines.append(f"Point({pid[0]}) = {{{x:.12g}, {y:.12g}, 0, lc}};")
            ids.append(pid[0])
        return ids

    lid = [0]
    surf = []
    for side in (0, 1):
        B, T = bot.copy(), top.copy()
        if side == 1:
            B[:, 0] = 2 * xm - B[:, 0]
            T[:, 0] = 2 * xm - T[:, 0]
        ib, it = add_pts(B), add_pts(T)
        l0 = lid[0]
        lines.append(f"Spline({l0+1}) = {{{','.join(map(str, ib))}}};")
        lines.append(f"Line({l0+2}) = {{{ib[-1]}, {it[-1]}}};")
        lines.append(f"Spline({l0+3}) = {{{','.join(map(str, it[::-1]))}}};")
        lines.append(f"Line({l0+4}) = {{{it[0]}, {ib[0]}}};")
        lines.append(f"Curve Loop({side+1}) = {{{l0+1},{l0+2},{l0+3},{l0+4}}};")
        lines.append(f"Plane Surface({side+1}) = {{{side+1}}};")
        lid[0] += 4
        surf.append(side + 1)
    lines.append('Physical Surface("beam") = {1,2};')
    lines.append("Mesh.ElementOrder = 2; Mesh.SecondOrderLinear = 0; Mesh.Algorithm = 6;")
    lines.append("Mesh.CharacteristicLengthMax = lc; Mesh.CharacteristicLengthMin = lc/2;")
    lines.append("Mesh.MshFileVersion = 2.2;")
    Path(path).write_text("\n".join(lines) + "\n")


def mesh_to_inp(msh, inp, Lh, ws, dmax, ninc, lateral="fixed", material=(fem.C10, fem.D1), tol=1e-6, release_at=None, fx=0.0):
    import meshio

    m = meshio.read(msh)
    pts = m.points[:, :2]
    tri = np.vstack([c.data for c in m.cells if c.type == "triangle6"])
    used = np.unique(tri)
    remap = -np.ones(len(pts), int)
    remap[used] = np.arange(len(used)) + 1
    pts = pts[used]
    tri = remap[tri]
    # make all triangles counter-clockwise (the mirrored half can come out either way)
    a, b, c = pts[tri[:, 0] - 1], pts[tri[:, 1] - 1], pts[tri[:, 2] - 1]
    area = 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    neg = area < 0
    tri[neg] = tri[neg][:, [0, 2, 1, 5, 4, 3]]
    xm = Lh + ws / 2
    x = pts[:, 0]
    wall = np.where((np.abs(x) < tol) | (np.abs(x - 2 * xm) < tol))[0] + 1
    shut = np.where((np.abs(x - Lh) < tol) | (np.abs(x - (Lh + ws)) < tol))[0] + 1
    out = ["*NODE, NSET=NALL"]
    out += [f"{i+1}, {p[0]:.12g}, {p[1]:.12g}, 0." for i, p in enumerate(pts)]
    out.append("*ELEMENT, TYPE=CPE6, ELSET=EALL")
    out += [f"{i+1}, " + ", ".join(map(str, t)) for i, t in enumerate(tri)]

    def nset(name, ids):
        o = [f"*NSET, NSET={name}"]
        for k in range(0, len(ids), 12):
            o.append(", ".join(map(str, ids[k : k + 12])))
        return o

    out += nset("WALL", wall) + nset("SHUT", shut)
    if lateral == "free":
        # tie u_x of all shuttle-face nodes to the first one (rigid lateral translation, free)
        m0 = shut[0]
        out.append("*EQUATION")
        for n in shut[1:]:
            out += ["2", f"{n}, 1, 1., {m0}, 1, -1."]
    out += ["*BOUNDARY", "WALL, 1, 2, 0."]
    out += ["*MATERIAL, NAME=NH", "*HYPERELASTIC, NEO HOOKE", f"{material[0]:.10g}, {material[1]:.10g}"]
    out += ["*SOLID SECTION, ELSET=EALL, MATERIAL=NH", "1.0"]
    out += ["*STEP, NLGEOM, INC=100000", "*STATIC", f"{1/ninc:.8g}, 1., 1e-8, {1/ninc:.8g}", "*BOUNDARY"]
    if lateral == "fixed":
        out.append("SHUT, 1, 1, 0.")
    out.append(f"SHUT, 2, 2, {-dmax:.10g}")
    if lateral == "free" and fx:
        # constant sideways dead load on the (laterally free) shuttle: a sway perturbation
        out += ["*AMPLITUDE, NAME=CONST", "0., 1., 1., 1.", "*CLOAD, AMPLITUDE=CONST", f"{shut[0]}, 1, {fx:.10g}"]
    out += ["*NODE PRINT, NSET=SHUT, TOTALS=ONLY", "RF"]
    if lateral == "free":
        out += [f"*NSET, NSET=M0\n{shut[0]}", "*NODE PRINT, NSET=M0", "U"]
    out += ["*NODE FILE", "U", "*END STEP"]
    if release_at is not None:
        # second step: shuttle force-free in y (kept rigid: all shuttle nodes share u_y)
        raise NotImplementedError
    Path(inp).write_text("\n".join(out) + "\n")
    return dict(nnodes=len(pts), nelem=len(tri), nshut=len(shut), nwall=len(wall))


def parse_dat(dat, dmax):
    txt = Path(dat).read_text()
    ts, fy, fx = [], [], []
    for mt in re.finditer(r"total force.*?time\s+([0-9.Ee+-]+)\s*\n\s*\n?\s*([-0-9.Ee+]+)\s+([-0-9.Ee+]+)\s+([-0-9.Ee+]+)", txt):
        ts.append(float(mt.group(1)))
        fx.append(float(mt.group(2)))
        fy.append(float(mt.group(3)))
    ts = np.array(ts)
    return ts * dmax, -np.array(fy), np.array(fx)


def run(name, p, Lh=28.0, ws=4.0, dmax=12.5, ninc=100, lc=None, lateral="fixed", keep_frd=False, workdir=None, fx=0.0):
    wd = Path(workdir) if workdir else HERE / "results" / "ccx" / name
    wd.mkdir(parents=True, exist_ok=True)
    import jax.numpy as jnp

    tmin = float(np.min(np.asarray(fem.thickness(jnp.asarray(p), jnp.linspace(0, 1, 201)))))
    if lc is None:
        lc = tmin / 3
    geo, msh, inp = wd / "job.geo", wd / "job.msh", wd / "job.inp"
    write_geo(geo, p, Lh, ws, lc)
    r = subprocess.run([MM, "run", "-n", "solidmech", "gmsh", "-2", str(geo), "-o", str(msh), "-format", "msh2"],
                       capture_output=True, text=True, env=ENV)
    if not msh.exists():
        raise RuntimeError("gmsh failed: " + r.stdout[-2000:] + r.stderr[-2000:])
    info = mesh_to_inp(msh, inp, Lh, ws, dmax, ninc, lateral=lateral, fx=fx)
    r = subprocess.run([MM, "run", "-n", "solidmech", "ccx", "job"], cwd=wd, capture_output=True, text=True, env=ENV)
    (wd / "ccx.stdout.txt").write_text(r.stdout[-20000:])
    d, F, Fx = parse_dat(wd / "job.dat", dmax)
    if lateral == "free":
        ux = [float(x) for x in re.findall(r"displacements \(vx,vy,vz\) for set M0 and time\s+[0-9.Ee+-]+\s*\n\s*\n\s*\d+\s+([-0-9.Ee+]+)", (wd / "job.dat").read_text())]
        info["ux"] = ux
        info["max_abs_ux"] = float(np.max(np.abs(ux))) if ux else None
    info.update(lc=lc, tmin=tmin, completed=bool(len(d) and abs(d[-1] - dmax) < 1e-6 * dmax), n_out=len(d))
    # remove bulky solver output by exact path
    for ext in ([] if keep_frd else ["frd"]) + ["msh", "12d", "cvg", "sta"]:
        f = wd / f"job.{ext}"
        if f.exists():
            f.unlink()
    np.savetxt(wd / "curve.csv", np.c_[d, F, Fx], delimiter=",", header="d_mm,F_N_per_mm,Fx", comments="")
    return d, F, info


def run_release(name, p, d_hold, Lh=28.0, ws=4.0, lc=None, nfreq=4, workdir=None):
    """Bistability check in CalculiX.

    Step 1: push the (rigid, non-rotating) shuttle to d_hold under displacement control.
    Step 2: release it (vertical force = 0) and let the static solve find the nearby equilibrium.
    Step 3: *FREQUENCY about that released, deformed state: all eigenvalues > 0 <=> stable.
    Returns dict(d_release, freqs, eigvals, completed).
    """
    import jax.numpy as jnp
    import meshio

    wd = Path(workdir) if workdir else HERE / "results" / "ccx" / name
    wd.mkdir(parents=True, exist_ok=True)
    tmin = float(np.min(np.asarray(fem.thickness(jnp.asarray(p), jnp.linspace(0, 1, 201)))))
    lc = lc or tmin / 3
    geo, msh, inp = wd / "job.geo", wd / "job.msh", wd / "job.inp"
    write_geo(geo, p, Lh, ws, lc)
    subprocess.run([MM, "run", "-n", "solidmech", "gmsh", "-2", str(geo), "-o", str(msh), "-format", "msh2"], capture_output=True, text=True, env=ENV)
    mesh_to_inp(msh, inp, Lh, ws, d_hold, 100)
    txt = inp.read_text()
    head = txt[: txt.index("*BOUNDARY")]
    nn = int(re.findall(r"^(\d+),", head[: head.index("*ELEMENT")], flags=re.M)[-1])
    ref = nn + 1
    shut_block = head[head.index("*NSET, NSET=SHUT") :].split("\n", 1)[1]
    shut = [int(x) for x in re.findall(r"\d+", shut_block.split("*")[0])]
    head = head.replace("*ELEMENT, TYPE=CPE6", f"{ref}, {Lh + ws / 2:.6g}, 0., 0.\n*ELEMENT, TYPE=CPE6", 1)
    out = [head.rstrip("\n"), f"*NSET, NSET=REF\n{ref}", "*EQUATION"]
    for n in shut:
        out += ["2", f"{n}, 2, 1., {ref}, 2, -1."]
    out += ["*BOUNDARY", "WALL, 1, 2, 0.", "SHUT, 1, 1, 0.", f"{ref}, 1, 1, 0."]
    out += ["*MATERIAL, NAME=NH", "*HYPERELASTIC, NEO HOOKE", f"{fem.C10:.10g}, {fem.D1:.10g}", "*DENSITY", "1.1e-9",
            "*SOLID SECTION, ELSET=EALL, MATERIAL=NH", "1.0"]
    out += ["*STEP, NLGEOM, INC=100000", "*STATIC", "0.01, 1., 1e-8, 0.01", "*BOUNDARY", f"{ref}, 2, 2, {-d_hold:.10g}",
            "*NODE PRINT, NSET=REF", "U, RF", "*END STEP"]
    out += ["*STEP, NLGEOM, INC=100000", "*STATIC", "0.05, 1., 1e-8, 0.05", "*BOUNDARY, OP=NEW", "WALL, 1, 2, 0.", "SHUT, 1, 1, 0.",
            f"{ref}, 1, 1, 0.", "*NODE PRINT, NSET=REF", "U", "*END STEP"]
    out += ["*STEP, PERTURBATION", "*FREQUENCY", f"{nfreq}", "*END STEP"]
    inp.write_text("\n".join(out) + "\n")
    r = subprocess.run([MM, "run", "-n", "solidmech", "ccx", "job"], cwd=wd, capture_output=True, text=True, env=ENV)
    (wd / "ccx.stdout.txt").write_text(r.stdout[-20000:])
    dat = (wd / "job.dat").read_text()
    us = [(float(a), float(b)) for a, b in re.findall(r"displacements \(vx,vy,vz\) for set REF and time\s+([0-9.Ee+-]+)\s*\n\s*\n\s*\d+\s+[-0-9.Ee+]+\s+([-0-9.Ee+]+)", dat)]
    us = [u for u in us if u[0] <= 2.0 + 1e-9]
    eig = re.findall(r"^\s+\d+\s+([-0-9.E+]+)\s+([-0-9.E+]+)\s+([-0-9.E+]+)\s+([-0-9.E+]+)\s*$", dat, flags=re.M)
    res = dict(d_hold=d_hold, u_history=us, d_release=-us[-1][1] if us else None,
               eigvals=[float(e[0]) for e in eig], freqs_hz=[float(e[2]) for e in eig],
               completed="increment size smaller" not in r.stdout and "ERROR" not in r.stdout[-3000:])
    for ext in ["frd", "msh", "12d", "cvg", "sta", "eig"]:
        f = wd / f"job.{ext}"
        if f.exists():
            f.unlink()
    return res
