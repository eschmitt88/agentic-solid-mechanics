"""Rung 3 — CalculiX shell model of the same cylinder (independent solver).

Quarter cylinder (symmetry planes theta = 0 and 90 deg, so even circumferential
wave numbers only), S8R shells, simply supported ends with rigid platens
(radial and circumferential displacement fixed, uniform axial displacement,
rotation free).

  buckle  : linear eigenvalue analysis (*BUCKLE) of the perfect shell
  collapse: geometrically nonlinear static analysis (NLGEOM) of the imperfect
            shell under prescribed end shortening; collapse load = maximum
            of the axial reaction.  CalculiX has no arc-length method, so the
            run simply stops converging once the shell snaps; the increments
            are cut automatically on the way to the peak.

usage: rung3_ccx.py buckle | collapse <xi> [per] | all
"""
import json
import os
import pathlib
import subprocess
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).parent
WORK = HERE / "results" / "ccx_work"
RES = HERE / "results"

R, T, E, NU = 200.0, 1.0, 70000.0, 0.3
CC = np.sqrt(3 * (1 - NU ** 2))
Q0 = (12 * (1 - NU ** 2)) ** 0.25 * np.sqrt(R / T)
ELL = 2 * np.pi * R / Q0
L = 4 * ELL
N_CL = E * T ** 2 / (R * CC)
P_CL = 2 * np.pi * R * N_CL
TRIGGER = 1e-3                      # non-axisymmetric seed amplitude / t


def run_ccx(job, threads=int(os.environ.get("CCX_THREADS", "6"))):
    env = dict(os.environ, MAMBA_ROOT_PREFIX=os.path.expanduser("~/projects/.micromamba"),
               OMP_NUM_THREADS=str(threads))
    exe = os.path.expanduser("~/.local/bin/micromamba")
    t0 = time.time()
    r = subprocess.run([exe, "run", "-n", "solidmech", "ccx", job], cwd=WORK, env=env, capture_output=True, text=True)
    (WORK / f"{job}.log").write_text(r.stdout + r.stderr)
    return r.returncode, time.time() - t0


FULL = bool(os.environ.get("CYL_FULL"))      # full 360 deg model (no symmetry planes, no rotational constraints)


def mesh(per, xi, trigger):
    """per = S8R elements per imperfection wavelength (axially); same size circumferentially."""
    nz = 4 * per
    arc = 2 * np.pi if FULL else 0.5 * np.pi
    nth = int(round((arc * R) / (ELL / per) / 2)) * 2
    nti = 2 * nth if FULL else 2 * nth + 1                    # number of theta grid lines
    ids = -np.ones((2 * nth + 1, 2 * nz + 1), int)
    nodes, k = [], 0
    rng_n = list(range(6, 19)) if FULL else [6, 8, 10, 12, 14, 16, 18]
    for j in range(2 * nz + 1):
        z = L * j / (2 * nz)
        for i in range(nti):
            if i % 2 == 1 and j % 2 == 1:
                continue
            th = arc * i / (2 * nth)
            seed = sum(np.cos(n * th) * (np.sin(np.pi * z / L) + 0.7 * np.sin(2 * np.pi * z / L) + 0.5 * np.sin(4 * np.pi * z / L)
                                         + 0.5 * np.sin(8 * np.pi * z / L)) for n in rng_n) / (2.7 * len(rng_n))
            r = R + xi * T * np.cos(Q0 * z / R) + trigger * T * seed
            k += 1
            ids[i, j] = k
            nodes.append((k, r * np.cos(th), r * np.sin(th), z))
    if FULL:
        ids[2 * nth, :] = ids[0, :]                           # periodic wrap
    elems, e = [], 0
    for j in range(nz):
        for i in range(nth):
            a, b = 2 * i, 2 * j
            e += 1
            elems.append((e, ids[a, b], ids[a + 2, b], ids[a + 2, b + 2], ids[a, b + 2],
                          ids[a + 1, b], ids[a + 2, b + 1], ids[a + 1, b + 2], ids[a, b + 1]))
    return nodes, elems, ids, nth, nz


def deck(per, xi, kind, trigger=TRIGGER, dmax=None, dinc=0.02):
    nodes, elems, ids, nth, nz = mesh(per, xi, trigger if kind == "collapse" else 0.0)
    out = ["*NODE, NSET=NALL"] + [f"{k}, {x:.9f}, {y:.9f}, {z:.9f}" for k, x, y, z in nodes]
    out += ["*ELEMENT, TYPE=S8R, ELSET=EALL"] + [", ".join(map(str, el)) for el in elems]
    nti = 2 * nth if FULL else 2 * nth + 1
    bot = [ids[i, 0] for i in range(nti)]
    top = [ids[i, 2 * nz] for i in range(nti)]

    def nset(name, lst):
        o = [f"*NSET, NSET={name}"]
        for q in range(0, len(lst), 12):
            o.append(", ".join(map(str, lst[q:q + 12])))
        return o
    out += nset("NBOT", bot) + nset("NTOP", top)
    out += ["*NSET, NSET=NENDS", "NBOT, NTOP", "*TRANSFORM, NSET=NENDS, TYPE=C", "0., 0., 0., 0., 0., 1."]
    out += ["*MATERIAL, NAME=AL", "*ELASTIC", f"{E}, {NU}", "*SHELL SECTION, ELSET=EALL, MATERIAL=AL", f"{T}"]
    out += ["*BOUNDARY", "NBOT, 1, 3", "NTOP, 1, 2"]
    if not FULL:
        s0 = [ids[0, j] for j in range(1, 2 * nz)]                 # theta = 0  (y = 0 plane), ends excluded
        s90 = [ids[2 * nth, j] for j in range(1, 2 * nz)]          # theta = 90 (x = 0 plane)
        out[-3:-3] = nset("NS0", s0) + nset("NS90", s90)
        out += ["NS0, 2, 2", "NS90, 1, 1"]
        if not os.environ.get("CYL_NO_ROT_BC"):      # diagnostic switch: drop the rotational symmetry conditions
            out += ["NS0, 4, 4", "NS0, 6, 6", "NS90, 5, 5", "NS90, 6, 6"]
    if kind == "buckle":
        # unit compressive line load N = 1 per unit circumference as consistent nodal forces
        le = ((2 * np.pi if FULL else 0.5 * np.pi) * R) / nth
        out += ["*STEP", "*BUCKLE", "6", "*CLOAD"]
        for i, n in enumerate(top):
            if i % 2 == 1:
                f = 2 * le / 3
            else:
                f = le / 3 if (FULL or 0 < i < 2 * nth) else le / 6
            out.append(f"{n}, 3, {-f:.9f}")
        out += ["*NODE FILE", "U", "*END STEP"]
    else:
        out += ["*STEP, NLGEOM, INC=2000", "*STATIC", f"{dinc}, 1., 1e-7, {dinc}",
                "*BOUNDARY", f"NTOP, 3, 3, {-dmax:.9f}",
                "*NODE PRINT, NSET=NTOP, TOTALS=ONLY", "RF", "*END STEP"]
    return "\n".join(out) + "\n", len(elems), len(nodes)


def parse_totals(dat):
    t, f = [], []
    lines = open(dat).read().splitlines()
    for i, s in enumerate(lines):
        if s.strip().startswith("total force"):
            t.append(float(s.split()[-1]))
            j = i + 1
            while not lines[j].strip():
                j += 1
            f.append(float(lines[j].split()[2]))
    return np.array(t), np.array(f)


def buckle(per):
    job = f"cyl_buckle_{per}"
    txt, ne, nn = deck(per, 0.0, "buckle")
    (WORK / f"{job}.inp").write_text(txt)
    rc, sec = run_ccx(job)
    fac = []
    grab = False
    for s in open(WORK / f"{job}.dat"):
        p = s.split()
        if "B U C K L I N G" in s:
            grab = True
        elif grab and len(p) == 2 and p[0].isdigit():
            fac.append(float(p[1]))
    return {"per": per, "n_elements": ne, "n_nodes": nn, "rc": rc, "seconds": sec, "factors": fac,
            "N_cr_over_N_cl": [x / N_CL for x in fac]}


def collapse(xi, per=8, rho_guess=1.0, trigger=TRIGGER, tag="", dinc=0.02):
    job = f"cyl_nl_xi{xi}_p{per}{tag}"
    d_cl = L * T / (R * CC)                       # end shortening of the perfect shell at the classical load
    dmax = 1.25 * rho_guess * d_cl
    txt, ne, nn = deck(per, xi, "collapse", trigger=trigger, dmax=dmax, dinc=dinc)
    (WORK / f"{job}.inp").write_text(txt)
    rc, sec = run_ccx(job)
    t, f = parse_totals(WORK / f"{job}.dat")
    P = (1 if FULL else 4) * np.abs(f) / P_CL
    short = t * dmax / d_cl
    np.savetxt(RES / f"r3_ccx_xi{xi}_p{per}{tag}.csv", np.column_stack([short, P]), delimiter=",",
               header="end_shortening_over_classical,P_over_Pcl", comments="")
    i = int(np.argmax(P)) if len(P) else -1
    return {"xi": xi, "per": per, "trigger": trigger, "n_elements": ne, "n_nodes": nn, "rc": rc, "seconds": sec,
            "n_increments": int(len(P)), "P_max_over_Pcl": float(P[i]) if len(P) else None,
            "shortening_at_max": float(short[i]) if len(P) else None, "last_time": float(t[-1]) if len(t) else None,
            "max_is_last_increment": bool(i == len(P) - 1), "completed_step": bool(len(t) and t[-1] >= 1 - 1e-9)}


if __name__ == "__main__":
    WORK.mkdir(parents=True, exist_ok=True)
    cmd = sys.argv[1]
    if cmd == "buckle":
        pers = [int(a) for a in sys.argv[2:]] or [6, 8, 12]
        res = [buckle(p) for p in pers]
        for r in res:
            print(r)
        (RES / "r3_ccx_buckle.json").write_text(json.dumps(res, indent=1))
    elif cmd == "collapse":
        xi = float(sys.argv[2]); per = int(sys.argv[3]) if len(sys.argv) > 3 else 8
        rho = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
        trig = float(sys.argv[5]) if len(sys.argv) > 5 else TRIGGER
        tag = sys.argv[6] if len(sys.argv) > 6 else ""
        dinc = float(sys.argv[7]) if len(sys.argv) > 7 else 0.02
        r = collapse(xi, per, rho, trig, tag, dinc)
        r["dinc"] = dinc
        print(r)
        (RES / f"r3_ccx_collapse_xi{xi}_p{per}{tag}.json").write_text(json.dumps(r, indent=1))
