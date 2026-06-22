"""verify3d.py — correctness checks for the 3D differentiable FEM.

(0) Element gold: the 24x24 B8 hex stiffness is symmetric, PSD, with exactly 6
    zero eigenvalues (3D rigid-body modes) and zero force under rigid translation.
(1) Forward-model gold: uniform-density compliance = tip deflection converges to
    3D cantilever beam theory (Euler-Bernoulli) as the mesh is refined through
    the thickness — the residual at coarse meshes is the known shear-locking of
    fully-integrated trilinear hex, and it shrinks monotonically.

Run (GPU auto): ~/projects/research/agentic-solid-mechanics/.venv/bin/python verify3d.py
"""
from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import jaxfem3d as j3

HERE = Path(__file__).parent
RESULTS = HERE / "results"
NU = 0.3


def element_gold() -> dict:
    KE = j3._hex_KE(NU)
    w = np.linalg.eigvalsh(KE)
    tx = np.zeros(24); tx[0::3] = 1.0
    return {
        "symmetric": bool(np.allclose(KE, KE.T)),
        "n_zero_eigs": int(np.sum(np.abs(w) < 1e-9)),
        "rigid_translation_max_force": float(np.max(np.abs(KE @ tx))),
        "min_eig": float(w.min()),
        "pass": bool(np.allclose(KE, KE.T) and np.sum(np.abs(w) < 1e-9) == 6
                     and np.max(np.abs(KE @ tx)) < 1e-9),
    }


def beam_convergence() -> dict:
    """L/h=10 cantilever, refine elements-through-thickness; FEM -> EB."""
    rows = []
    for nx, ny, nz in [(20, 2, 2), (40, 4, 4), (60, 6, 6), (80, 8, 8)]:
        p = j3.TopOpt3D(nx, ny, nz, 0.3, penal=1.0, rmin=1.5)
        C = float(p.compliance(jnp.ones(p.nelem)))
        L, h, b, E = float(nx), float(ny), float(nz), 1.0
        I = b * h ** 3 / 12.0
        d_eb = L ** 3 / (3 * E * I)
        rows.append({"ny_through_thickness": ny, "ndof": int(p.ndof),
                     "fem_tip_deflection": C, "euler_bernoulli": d_eb,
                     "err_vs_eb_pct": 100 * (C - d_eb) / d_eb})
    errs = [abs(r["err_vs_eb_pct"]) for r in rows]
    monotone = all(errs[i] > errs[i + 1] for i in range(len(errs) - 1))
    return {"rows": rows, "monotone_convergence_to_EB": monotone,
            "finest_err_vs_eb_pct": rows[-1]["err_vs_eb_pct"],
            "note": "residual at coarse meshes = shear locking of fully-integrated "
                    "trilinear B8; converges to beam theory under refinement.",
            "pass": monotone and abs(rows[-1]["err_vs_eb_pct"]) < 1.0}


def main():
    import jax
    print("device:", jax.devices()[0])
    elem = element_gold()
    print("element gold:", {k: elem[k] for k in ("symmetric", "n_zero_eigs",
          "rigid_translation_max_force", "pass")})
    beam = beam_convergence()
    print("beam convergence (FEM -> EB under through-thickness refinement):")
    for r in beam["rows"]:
        print(f"  ny={r['ny_through_thickness']} ndof={r['ndof']:6d} "
              f"err_vs_EB={r['err_vs_eb_pct']:+.2f}%")
    print(f"  monotone={beam['monotone_convergence_to_EB']} "
          f"finest={beam['finest_err_vs_eb_pct']:+.2f}% -> {'PASS' if beam['pass'] else 'FAIL'}")
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "verification3d.json").write_text(
        json.dumps({"element_gold": elem, "beam_convergence": beam}, indent=2))
    print("Wrote results/verification3d.json")


if __name__ == "__main__":
    main()
