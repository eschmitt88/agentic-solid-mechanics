"""Cantilever operator-baseline harness for CalculiX (trial 1).

Provides, in one module:
  * analytical()      — Euler-Bernoulli closed-form reference.
  * generate_inp()    — structured C3D20R (quadratic hex) cantilever deck.
  * run_ccx()         — invoke CalculiX (via micromamba env) on a deck.
  * parse_displacement / parse_stress — read ccx .dat output.
  * grade()           — compare a solved result to the analytical reference.

x runs along the length L (fixed at x=0), y along the width b, z along the
height h. A downward (-z) tip load P is applied at the free end x=L.

Pure stdlib + numpy so it runs under the project `uv` venv; ccx is shelled out
to the micromamba `solidmech` env.
"""
from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path


# --------------------------------------------------------------------------- #
# Problem spec + analytical reference
# --------------------------------------------------------------------------- #
@dataclass
class Spec:
    length_m: float = 1.0
    width_m: float = 0.05
    height_m: float = 0.10
    youngs_modulus_pa: float = 210.0e9
    poisson_ratio: float = 0.30
    tip_load_n: float = 1000.0


def analytical(spec: Spec) -> dict:
    """Euler-Bernoulli end-loaded cantilever (rectangular section)."""
    I = spec.width_m * spec.height_m**3 / 12.0
    c = spec.height_m / 2.0
    delta = spec.tip_load_n * spec.length_m**3 / (3.0 * spec.youngs_modulus_pa * I)
    sigma = spec.tip_load_n * spec.length_m * c / I
    return {"I_m4": I, "tip_deflection_m": delta, "max_bending_stress_pa": sigma}


# --------------------------------------------------------------------------- #
# Structured C3D20R mesh + Abaqus/CalculiX .inp generation
# --------------------------------------------------------------------------- #
# Local node offsets (half-grid units, 0/1/2) for the 20 nodes of a C3D20
# element, in Abaqus ordering (8 corners, 12 mid-edges).
_C3D20_OFFSETS = [
    (0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0),          # 1-4  bottom corners
    (0, 0, 2), (2, 0, 2), (2, 2, 2), (0, 2, 2),          # 5-8  top corners
    (1, 0, 0), (2, 1, 0), (1, 2, 0), (0, 1, 0),          # 9-12 bottom edges
    (1, 0, 2), (2, 1, 2), (1, 2, 2), (0, 1, 2),          # 13-16 top edges
    (0, 0, 1), (2, 0, 1), (2, 2, 1), (0, 2, 1),          # 17-20 vertical edges
]


def _mesh_dims(n_through_height: int) -> tuple[int, int, int]:
    """(nx, ny, nz) element counts from one knob: elements through the height."""
    nz = max(1, n_through_height)
    nx = max(2, round(2.5 * nz))      # along length
    ny = max(1, round(0.5 * nz))      # along width
    return nx, ny, nz


def generate_inp(spec: Spec, n_through_height: int, path: Path) -> dict:
    """Write a CalculiX C3D20R cantilever deck. Returns mesh metadata."""
    nx, ny, nz = _mesh_dims(n_through_height)
    HX, HY, HZ = 2 * nx, 2 * ny, 2 * nz

    # Used half-grid nodes: corners (#odd==0) and mid-edges (#odd==1).
    def used(I, J, K):
        return (I % 2) + (J % 2) + (K % 2) <= 1

    node_id: dict[tuple[int, int, int], int] = {}
    coords: list[tuple[int, float, float, float]] = []
    nid = 0
    for K in range(HZ + 1):
        for J in range(HY + 1):
            for I in range(HX + 1):
                if not used(I, J, K):
                    continue
                nid += 1
                node_id[(I, J, K)] = nid
                x = I / HX * spec.length_m
                y = J / HY * spec.width_m
                z = K / HZ * spec.height_m
                coords.append((nid, x, y, z))

    elements: list[tuple[int, list[int]]] = []
    eid = 0
    for ez in range(nz):
        for ey in range(ny):
            for ex in range(nx):
                base = (2 * ex, 2 * ey, 2 * ez)
                conn = [node_id[(base[0] + dx, base[1] + dy, base[2] + dz)]
                        for (dx, dy, dz) in _C3D20_OFFSETS]
                eid += 1
                elements.append((eid, conn))

    fixed = [nid for (I, J, K), nid in node_id.items() if I == 0]
    tip = [nid for (I, J, K), nid in node_id.items() if I == HX]
    per_node_load = -spec.tip_load_n / len(tip)   # downward (-z), DOF 3

    lines: list[str] = ["*HEADING", "cantilever C3D20R agentic baseline"]
    lines.append("*NODE, NSET=NALL")
    lines += [f"{i}, {x:.10e}, {y:.10e}, {z:.10e}" for (i, x, y, z) in coords]
    lines.append("*ELEMENT, TYPE=C3D20R, ELSET=EALL")
    for (i, conn) in elements:
        lines.append(f"{i}, " + ", ".join(str(c) for c in conn[:15]))
        lines.append("      " + ", ".join(str(c) for c in conn[15:]))
    lines.append("*NSET, NSET=NFIX")
    lines += _chunk(fixed)
    lines.append("*NSET, NSET=NTIP")
    lines += _chunk(tip)
    lines += [
        "*MATERIAL, NAME=STEEL",
        "*ELASTIC",
        f"{spec.youngs_modulus_pa:.6e}, {spec.poisson_ratio}",
        "*SOLID SECTION, ELSET=EALL, MATERIAL=STEEL",
        "*BOUNDARY",
        "NFIX, 1, 3",
        "*STEP",
        "*STATIC",
        "*CLOAD",
    ]
    lines += [f"{i}, 3, {per_node_load:.10e}" for i in tip]
    lines += [
        "*NODE PRINT, NSET=NTIP",
        "U",
        "*EL PRINT, ELSET=EALL",
        "S",
        "*END STEP",
        "",
    ]
    path.write_text("\n".join(lines))
    return {"nx": nx, "ny": ny, "nz": nz, "n_nodes": len(coords),
            "n_elements": len(elements), "n_tip_nodes": len(tip)}


def _chunk(ids: list[int], per_line: int = 8) -> list[str]:
    return [", ".join(str(x) for x in ids[i:i + per_line])
            for i in range(0, len(ids), per_line)]


# --------------------------------------------------------------------------- #
# Run + parse
# --------------------------------------------------------------------------- #
def run_ccx(job_noext: Path, env: str = "solidmech", ccx_bin: str = "ccx",
            timeout: int = 1800) -> subprocess.CompletedProcess:
    """Run `ccx <job>` inside the micromamba env, cwd = job's directory."""
    return subprocess.run(
        ["micromamba", "run", "-n", env, ccx_bin, job_noext.name],
        cwd=job_noext.parent, capture_output=True, text=True, timeout=timeout,
    )


def _data_rows(dat_text: str, header_substr: str):
    """Yield numeric rows from the .dat block whose header contains header_substr."""
    rows, capturing = [], False
    for raw in dat_text.splitlines():
        line = raw.strip()
        if header_substr in line.lower():
            capturing, rows = True, []
            continue
        if capturing:
            parts = line.split()
            if not parts:
                if rows:
                    break
                continue
            try:
                rows.append([float(p) for p in parts])
            except ValueError:
                if rows:
                    break
    return rows


def parse_displacement(dat_path: Path) -> float:
    """Max |U_z| (vertical deflection magnitude) over the printed node set."""
    rows = _data_rows(dat_path.read_text(), "displacements")
    return max(abs(r[3]) for r in rows)   # cols: node, u1, u2, u3


def parse_stress(dat_path: Path) -> float:
    """Max von Mises over integration-point stresses (sxx syy szz sxy sxz syz)."""
    rows = _data_rows(dat_path.read_text(), "stresses")
    best = 0.0
    for r in rows:
        sxx, syy, szz, sxy, sxz, syz = r[2:8]
        vm = math.sqrt(0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
                       + 3.0 * (sxy ** 2 + sxz ** 2 + syz ** 2))
        best = max(best, vm)
    return best


# --------------------------------------------------------------------------- #
# Grade
# --------------------------------------------------------------------------- #
def grade(spec: Spec, fem_deflection_m: float, fem_vm_stress_pa: float | None,
          tol_deflection_rel: float = 0.05, tol_stress_rel: float = 0.10) -> dict:
    ref = analytical(spec)
    d_err = abs(fem_deflection_m - ref["tip_deflection_m"]) / ref["tip_deflection_m"]
    out = {
        "analytical": ref,
        "fem_tip_deflection_m": fem_deflection_m,
        "deflection_rel_err": d_err,
        "deflection_pass": d_err <= tol_deflection_rel,
    }
    if fem_vm_stress_pa is not None:
        s_err = abs(fem_vm_stress_pa - ref["max_bending_stress_pa"]) / ref["max_bending_stress_pa"]
        out["fem_max_vm_stress_pa"] = fem_vm_stress_pa
        out["stress_rel_err"] = s_err
        out["stress_pass"] = s_err <= tol_stress_rel
    return out


if __name__ == "__main__":
    import sys
    s = Spec()
    print(json.dumps({"spec": asdict(s), "analytical": analytical(s)}, indent=2))
