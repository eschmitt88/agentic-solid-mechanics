"""FEA core for the cantilever design-loop trial (CalculiX C3D20R).

Self-contained (experiments are self-contained per the project convention): the
structured C3D20R deck generator + ccx runner + .dat parser are the same proven
code as trial 1's cantilever.py, plus design helpers (mass, nominal stress).

x = length (fixed at x=0), y = width b, z = height h, downward (-z) tip load P.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Beam:
    length_m: float
    width_m: float          # b (design var)
    height_m: float         # h (design var)
    youngs_modulus_pa: float
    poisson_ratio: float
    density_kg_m3: float
    tip_load_n: float

    @property
    def I_m4(self) -> float:
        return self.width_m * self.height_m**3 / 12.0

    def eb_deflection_m(self) -> float:
        """Euler-Bernoulli tip deflection (neglects shear)."""
        return self.tip_load_n * self.length_m**3 / (3.0 * self.youngs_modulus_pa * self.I_m4)

    def nominal_bending_stress_pa(self) -> float:
        """Closed-form max bending stress 6PL/(b h^2) at the fixed end.

        We grade the stress constraint on THIS (not the FE peak): trial 1 showed
        the FE von Mises peak at a fully-encastre face is a mesh-dependent
        singularity, ill-posed as a constraint. The nominal bending stress is
        the well-defined engineering quantity.
        """
        return 6.0 * self.tip_load_n * self.length_m / (self.width_m * self.height_m**2)

    def mass_kg(self) -> float:
        return self.density_kg_m3 * self.length_m * self.width_m * self.height_m


# Local node offsets (half-grid units 0/1/2) for the 20 nodes of a C3D20
# element in Abaqus ordering (8 corners, 12 mid-edges).
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


def generate_inp(beam: Beam, n_through_height: int, path: Path) -> dict:
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
                coords.append((nid, I / HX * beam.length_m,
                               J / HY * beam.width_m, K / HZ * beam.height_m))

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

    L = ["*HEADING", "cantilever design-loop", "*NODE, NSET=NALL"]
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


def fe_deflection(beam: Beam, n: int, workdir: Path, tag: str = "eval",
                  env="solidmech", ccx_bin="ccx") -> float:
    """Mesh, solve, and return the FE tip deflection (|U_z| max) for this beam."""
    workdir.mkdir(parents=True, exist_ok=True)
    job = workdir / f"{tag}"
    generate_inp(beam, n, job.with_suffix(".inp"))
    proc = run_ccx(job, env=env, ccx_bin=ccx_bin)
    dat = job.with_suffix(".dat")
    if proc.returncode != 0 or not dat.exists():
        raise RuntimeError(f"ccx failed rc={proc.returncode}: {proc.stderr[-300:]}")
    return parse_displacement(dat)
