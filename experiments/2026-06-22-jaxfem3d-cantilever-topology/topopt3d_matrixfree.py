"""Large-grid 3D topology optimization with the matrix-free solver.

Runs the same Optimality-Criteria loop as topopt3d_reference.py, but on the
matrix-free solver (jaxfem3d_mf) at a grid whose dense stiffness matrix would not
fit in GPU memory — demonstrating the full optimization (not just a single solve)
at scale. Saves the density field + metrics.

Run (GPU): ~/projects/research/agentic-solid-mechanics/.venv/bin/python topopt3d_matrixfree.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import jax
import numpy as np

import jaxfem3d_mf as mf
import topopt3d_reference as tr

HERE = Path(__file__).parent
RESULTS = HERE / "results"

NX, NY, NZ = 48, 16, 16
VOLFRAC, PENAL, RMIN = 0.30, 3.0, 1.5


def main():
    prob = mf.TopOpt3DMatrixFree(NX, NY, NZ, VOLFRAC, penal=PENAL, rmin=RMIN,
                                 cg_tol=1e-6, cg_maxiter=2000)
    dense_gb = (prob.ndof ** 2) * 8 / 1e9
    print(f"matrix-free 3D topology opt: {NX}x{NY}x{NZ} = {prob.nelem} elems, "
          f"ndof={prob.ndof} (a dense stiffness matrix would be {dense_gb:.0f} GB), "
          f"device={jax.devices()[0]}")
    t0 = time.time()
    res = tr.run(prob, {"optim": {"max_iter": 40, "move": 0.2, "tol_change": 0.01}})
    dt = time.time() - t0

    rho3d = np.asarray(res["rho"]).reshape(NX, NY, NZ)
    np.save(RESULTS / "matrixfree_density.npy", rho3d)
    out = {"grid": [NX, NY, NZ], "ndof": int(prob.ndof),
           "dense_matrix_gb": round(dense_gb, 1),
           "final_compliance": res["final_compliance"],
           "final_volume": res["final_volume"],
           "n_iterations": len(res["history"]), "wall_s": round(dt, 1),
           "solver": "matrix-free CG + Jacobi, conv density filter"}
    (RESULTS / "matrixfree_topopt.json").write_text(json.dumps(out, indent=2))
    print(f"\nfinal compliance {res['final_compliance']:.3f} at vol "
          f"{res['final_volume']:.3f}, {len(res['history'])} iters, {dt:.1f}s wall")
    print("Wrote results/matrixfree_density.npy, matrixfree_topopt.json")


if __name__ == "__main__":
    main()
