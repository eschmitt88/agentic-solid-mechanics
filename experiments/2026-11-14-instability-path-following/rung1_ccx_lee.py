"""Rung 1f — independent check of the Lee-frame limit load with CalculiX.

CalculiX 2.23 has no arc-length (Riks) procedure, so the check is run under
displacement control: the vertical displacement of the load point is
prescribed and the reaction force is read back.  That reaches the first
*load* limit point (the load maximum comes before the displacement turning
point), which is the quantity being cross-checked.

B32R beams (expanded internally to C3D20R solids), NLGEOM, nu = 0.
"""
import json
import os
import pathlib
import subprocess
import sys

import numpy as np

HERE = pathlib.Path(__file__).parent
WORK = HERE / "results" / "ccx_work"
L, A_, E = 120.0, 24.0, 720.0
EI = E * 2.0


def run_ccx(job, cwd, threads=4):
    env = dict(os.environ, MAMBA_ROOT_PREFIX=os.path.expanduser("~/projects/.micromamba"),
               OMP_NUM_THREADS=str(threads))
    exe = os.path.expanduser("~/.local/bin/micromamba")
    r = subprocess.run([exe, "run", "-n", "solidmech", "ccx", job], cwd=cwd, env=env,
                       capture_output=True, text=True)
    (pathlib.Path(cwd) / f"{job}.log").write_text(r.stdout + r.stderr)
    return r.returncode


def deck(nel_seg, vmax, ninc):
    """nel_seg quadratic elements per 24-unit segment."""
    n = 5 * nel_seg
    pts = []
    for i in range(2 * n + 1):                      # column, 2n+1 nodes
        pts.append((0.0, L * i / (2 * n)))
    for i in range(1, 2 * n + 1):                   # beam
        pts.append((L * i / (2 * n), L))
    lines = ["*NODE, NSET=NALL"]
    for k, (x, y) in enumerate(pts):
        lines.append(f"{k + 1}, {x:.10g}, {y:.10g}, 0.")
    lines.append("*ELEMENT, TYPE=B32R, ELSET=EALL")
    for e in range(2 * n):
        lines.append(f"{e + 1}, {2 * e + 1}, {2 * e + 2}, {2 * e + 3}")
    iload = 2 * n + 1 + 2 * nel_seg                # node at x = 24 on the beam
    assert abs(pts[iload - 1][0] - A_) < 1e-9
    iend = len(pts)
    lines += [
        "*MATERIAL, NAME=EL", "*ELASTIC", f"{E}, 0.0",
        "*BEAM SECTION, ELSET=EALL, MATERIAL=EL, SECTION=RECT", "3., 2.", "0., 0., 1.",
        "*BOUNDARY", "1, 1, 3", f"{iend}, 1, 3", "NALL, 3, 3", "NALL, 4, 5",
        f"*NSET, NSET=NLOAD", f"{iload}",
        "*STEP, NLGEOM, INC=100000", "*STATIC", f"{1.0 / ninc}, 1., 1e-8, {1.0 / ninc}",
        "*BOUNDARY", f"{iload}, 2, 2, {-vmax}",
        "*NODE PRINT, NSET=NLOAD, TOTALS=NO", "U, RF", "*END STEP",
    ]
    return "\n".join(lines) + "\n", iload


def parse(dat):
    """Return arrays (uy, RFy) per increment from the .dat file."""
    U, RF, mode = [], [], None
    for line in open(dat):
        s = line.strip()
        if s.startswith("displacements"):
            mode = "U"
        elif s.startswith("forces"):
            mode = "RF"
        elif s and s[0].isdigit() and mode:
            p = s.split()
            (U if mode == "U" else RF).append(float(p[2]))
            mode = None
    n = min(len(U), len(RF))
    return np.array(U[:n]), np.array(RF[:n])


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    out = {}
    for nel_seg in (2, 4, 8):
        job = f"lee_b32r_{nel_seg}"
        txt, iload = deck(nel_seg, vmax=0.46 * L, ninc=230)
        (WORK / f"{job}.inp").write_text(txt)
        rc = run_ccx(job, WORK)
        U, RF = parse(WORK / f"{job}.dat")
        P = -RF * L ** 2 / EI            # downward applied load, normalised
        v = -U / L
        i = int(np.argmax(P))
        # parabolic refinement of the maximum
        if 0 < i < len(P) - 1:
            a, b, c = P[i - 1], P[i], P[i + 1]
            Pmax = b - 0.125 * (c - a) ** 2 / (a - 2 * b + c)
        else:
            Pmax = P[i]
        out[job] = {"n_quadratic_elements": 10 * nel_seg, "rc": rc, "n_inc": len(P), "Pmax": float(Pmax),
                    "v_at_max": float(v[i]), "v_last": float(v[-1]), "max_is_interior": bool(0 < i < len(P) - 1)}
        np.savetxt(HERE / "results" / f"r1_ccx_{job}.csv", np.column_stack([v, P]), delimiter=",",
                   header="uy_over_L,P_L2_over_EI", comments="")
        print(job, out[job])
    (HERE / "results" / "r1_ccx_lee.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
