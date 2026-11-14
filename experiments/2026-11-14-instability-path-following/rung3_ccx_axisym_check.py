"""Third opinion on the axisymmetric pre-buckling stiffness of the imperfect
cylinder: CalculiX axisymmetric *continuum* elements (CAX8), i.e. no shell
theory at all, against (a) the own Donnell solver and (b) the CalculiX S8R
shell model.  Linear static, simply supported ends.  Output: axial secant
stiffness relative to the perfect membrane value E t / L."""
import json
import pathlib
import sys

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)
from rung3_ccx import CC, E, L, NU, P_CL, Q0, R, T, WORK, parse_totals, run_ccx
from rung3_cylinder import ELL, N_CL, Shell

HERE = pathlib.Path(__file__).parent


def cax8(xi, nz_per=16, nt=2):
    nz = 4 * nz_per
    ids = -np.ones((2 * nt + 1, 2 * nz + 1), int)
    out, k = ["*NODE, NSET=NALL"], 0
    for j in range(2 * nz + 1):
        z = L * j / (2 * nz)
        rm = R + xi * T * np.cos(Q0 * z / R)
        sl = -xi * T * Q0 / R * np.sin(Q0 * z / R)
        n = np.array([1.0, -sl]) / np.hypot(1.0, sl)
        for i in range(2 * nt + 1):
            if i % 2 == 1 and j % 2 == 1:
                continue
            s = T * (i / (2 * nt) - 0.5)
            k += 1
            ids[i, j] = k
            out.append(f"{k}, {rm + s * n[0]:.10f}, {z + s * n[1]:.10f}, 0.")
    out.append("*ELEMENT, TYPE=CAX8, ELSET=EALL")
    e = 0
    for j in range(nz):
        for i in range(nt):
            a, b = 2 * i, 2 * j
            e += 1
            out.append(", ".join(map(str, (e, ids[a, b], ids[a + 2, b], ids[a + 2, b + 2], ids[a, b + 2],
                                           ids[a + 1, b], ids[a + 2, b + 1], ids[a + 1, b + 2], ids[a, b + 1]))))
    bot, top = ids[nt, 0], ids[nt, 2 * nz]
    d = 0.05 * L * T / (R * CC)
    out += ["*MATERIAL, NAME=AL", "*ELASTIC", f"{E}, {NU}", "*SOLID SECTION, ELSET=EALL, MATERIAL=AL",
            "*NSET, NSET=NTOP", f"{top}", "*BOUNDARY", f"{bot}, 1, 2", f"{top}, 1, 1",
            "*STEP", "*STATIC", "*BOUNDARY", f"{top}, 2, 2, {-d:.10f}",
            "*NODE PRINT, NSET=NTOP, TOTALS=ONLY", "RF", "*END STEP"]
    job = f"cax8_xi{xi}_{nz_per}_{nt}"
    (WORK / f"{job}.inp").write_text("\n".join(out) + "\n")
    run_ccx(job)
    # axial direction is y for axisymmetric elements -> second force component
    lines = (WORK / f"{job}.dat").read_text().splitlines()
    i = max(k for k, s_ in enumerate(lines) if s_.strip().startswith("total force"))
    fy = float(next(s_ for s_ in lines[i + 1:] if s_.strip()).split()[1])
    return abs(fy), d


def cax8_stiffness(xi, nz_per=16, nt=2):
    fy, d = cax8(xi, nz_per, nt)
    f0, _ = cax8(0.0, nz_per, nt)
    # normalise by the perfect-shell run on the same mesh (removes the sector-angle convention);
    # the perfect simply supported shell itself is 1.0051 x membrane (own solver), applied below
    return fy / f0


def _unused():
    P = 0.0
    return P / P_CL / 0.05


def own_linear(xi, bc="ss"):
    sh = Shell(64, L, xi, bc)
    du = np.linalg.solve(sh.K(np.zeros(sh.n0), 0.0), sh.q)          # displacement per unit N/N_cl
    short = -du[sh.iend] / (L * T / (R * CC))
    return 1.0 / short


if __name__ == "__main__":
    WORK.mkdir(parents=True, exist_ok=True)
    res = {}
    for xi in (0.0, 0.1, 0.3, 1.0):
        res[str(xi)] = {"own_donnell_ss": own_linear(xi), "own_donnell_cell": own_linear(xi, "cell"),
                        "koiter_infinite_shell_formula": 1 / (1 + CC ** 2 * xi ** 2),
                        "ccx_cax8_16x2_rel_to_perfect": cax8_stiffness(xi, 16, 2),
                        "ccx_cax8_32x4_rel_to_perfect": cax8_stiffness(xi, 32, 4)}
        res[str(xi)]["own_donnell_ss_rel_to_perfect"] = res[str(xi)]["own_donnell_ss"] / own_linear(0.0)
        print(xi, res[str(xi)])
    (HERE / "results" / "r3_axisym_stiffness_check.json").write_text(json.dumps(res, indent=1))
