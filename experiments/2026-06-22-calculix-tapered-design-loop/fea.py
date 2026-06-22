"""FEA core for the TAPERED cantilever design loop (CalculiX C3D20R).

Cross-section: constant width b, height varying linearly from h_root at the
clamp (x=0) to h_tip at the free end (x=L). A tapered beam has no elementary
tip-deflection formula, so FE is genuinely required in the design loop (unlike
the prismatic trial 2, where the beam formula nearly sufficed).

x = length (fixed at x=0), y = width, z = height, downward (-z) tip load P.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class TaperedBeam:
    length_m: float
    width_m: float           # b (design var)
    h_root_m: float          # height at clamp (design var)
    h_tip_m: float           # height at free end (design var)
    youngs_modulus_pa: float
    poisson_ratio: float
    density_kg_m3: float
    tip_load_n: float

    def h_at(self, x):
        return self.h_root_m + (self.h_tip_m - self.h_root_m) * (x / self.length_m)

    def mass_kg(self) -> float:
        # volume = b * integral of h(x) dx = b * L * (h_root + h_tip)/2
        return self.density_kg_m3 * self.width_m * self.length_m * (self.h_root_m + self.h_tip_m) / 2.0

    def max_nominal_stress_pa(self, n: int = 400) -> float:
        """max over x of nominal bending stress 6 P (L-x) / (b h(x)^2)."""
        x = np.linspace(0.0, self.length_m, n, endpoint=False)
        sigma = 6.0 * self.tip_load_n * (self.length_m - x) / (self.width_m * self.h_at(x) ** 2)
        return float(np.max(sigma))

    def eb_deflection_integral_m(self, n: int = 2000) -> float:
        """Euler-Bernoulli tip deflection by unit-load integral (neglects shear):
        delta = integral_0^L P (L-x)^2 / (E I(x)) dx, I(x) = b h(x)^3 / 12."""
        x = np.linspace(0.0, self.length_m, n)
        I = self.width_m * self.h_at(x) ** 3 / 12.0
        integrand = self.tip_load_n * (self.length_m - x) ** 2 / (self.youngs_modulus_pa * I)
        return float(np.trapezoid(integrand, x))


_C3D20_OFFSETS = [
    (0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0),
    (0, 0, 2), (2, 0, 2), (2, 2, 2), (0, 2, 2),
    (1, 0, 0), (2, 1, 0), (1, 2, 0), (0, 1, 0),
    (1, 0, 2), (2, 1, 2), (1, 2, 2), (0, 1, 2),
    (0, 0, 1), (2, 0, 1), (2, 2, 1), (0, 2, 1),
]


def _mesh_dims(n_through_height: int) -> tuple[int, int, int]:
    nz = max(1, n_through_height)
    nx = max(2, round(2.5 * nz))
    ny = max(1, round(0.5 * nz))
    return nx, ny, nz


def generate_inp(beam: TaperedBeam, n_through_height: int, path: Path) -> dict:
    nx, ny, nz = _mesh_dims(n_through_height)
    HX, HY, HZ = 2 * nx, 2 * ny, 2 * nz

    def used(I, J, K):
        return (I % 2) + (J % 2) + (K % 2) <= 1

    node_id, coords, nid = {}, [], 0
    for K in range(HZ + 1):
        for J in range(HY + 1):
            for I in range(HX + 1):
                if not used(I, J, K):
                    continue
                nid += 1
                node_id[(I, J, K)] = nid
                x = I / HX * beam.length_m
                h_local = beam.h_at(x)               # <-- taper enters here
                y = J / HY * beam.width_m
                z = K / HZ * h_local
                coords.append((nid, x, y, z))

    elements, eid = [], 0
    for ez in range(nz):
        for ey in range(ny):
            for ex in range(nx):
                base = (2 * ex, 2 * ey, 2 * ez)
                conn = [node_id[(base[0] + dx, base[1] + dy, base[2] + dz)]
                        for (dx, dy, dz) in _C3D20_OFFSETS]
                eid += 1
                elements.append((eid, conn))

    fixed = [n for (I, J, K), n in node_id.items() if I == 0]
    tip = [n for (I, J, K), n in node_id.items() if I == HX]
    per_node = -beam.tip_load_n / len(tip)

    L = ["*HEADING", "tapered cantilever design-loop", "*NODE, NSET=NALL"]
    L += [f"{i}, {x:.10e}, {y:.10e}, {z:.10e}" for (i, x, y, z) in coords]
    L.append("*ELEMENT, TYPE=C3D20R, ELSET=EALL")
    for (i, conn) in elements:
        L.append(f"{i}, " + ", ".join(map(str, conn[:15])))
        L.append("      " + ", ".join(map(str, conn[15:])))
    L.append("*NSET, NSET=NFIX")
    L += _chunk(fixed)
    L.append("*NSET, NSET=NTIP")
    L += _chunk(tip)
    L += ["*MATERIAL, NAME=STEEL", "*ELASTIC",
          f"{beam.youngs_modulus_pa:.6e}, {beam.poisson_ratio}",
          "*SOLID SECTION, ELSET=EALL, MATERIAL=STEEL",
          "*BOUNDARY", "NFIX, 1, 3", "*STEP", "*STATIC", "*CLOAD"]
    L += [f"{i}, 3, {per_node:.10e}" for i in tip]
    L += ["*NODE PRINT, NSET=NTIP", "U", "*END STEP", ""]
    path.write_text("\n".join(L))
    return {"nx": nx, "ny": ny, "nz": nz,
            "n_nodes": len(coords), "n_elements": len(elements)}


def _chunk(ids, per_line=8):
    return [", ".join(map(str, ids[i:i + per_line])) for i in range(0, len(ids), per_line)]


def run_ccx(job_noext: Path, env="solidmech", ccx_bin="ccx", timeout=900):
    return subprocess.run(["micromamba", "run", "-n", env, ccx_bin, job_noext.name],
                          cwd=job_noext.parent, capture_output=True, text=True, timeout=timeout)


def parse_displacement(dat_path: Path) -> float:
    rows, capturing = [], False
    for raw in dat_path.read_text().splitlines():
        line = raw.strip()
        if "displacements" in line.lower():
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
    return max(abs(r[3]) for r in rows)


def fe_deflection(beam: TaperedBeam, n: int, workdir: Path, tag="eval",
                  env="solidmech", ccx_bin="ccx") -> float:
    workdir.mkdir(parents=True, exist_ok=True)
    job = workdir / tag
    generate_inp(beam, n, job.with_suffix(".inp"))
    proc = run_ccx(job, env=env, ccx_bin=ccx_bin)
    dat = job.with_suffix(".dat")
    if proc.returncode != 0 or not dat.exists():
        raise RuntimeError(f"ccx failed rc={proc.returncode}: {proc.stderr[-300:]}")
    return parse_displacement(dat)
