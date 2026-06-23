"""High-resolution 3D topology optimization with the matrix-free solver — how
dense a structure can we optimize in roughly a 2-minute GPU budget.

Usage: highres_topopt.py NX NY NZ ITERS [RMIN] [VOLFRAC]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import jax
import numpy as np

import jaxfem3d_mf as mf
import topopt3d_reference as tr

HERE = Path(__file__).parent
RESULTS = HERE / "results"


def main():
    nx, ny, nz, iters = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    rmin = float(sys.argv[5]) if len(sys.argv) > 5 else 2.0
    volfrac = float(sys.argv[6]) if len(sys.argv) > 6 else 0.30
    prob = mf.TopOpt3DMatrixFree(nx, ny, nz, volfrac, penal=3.0, rmin=rmin,
                                 cg_tol=1e-6, cg_maxiter=3000)
    dense_gb = (prob.ndof ** 2) * 8 / 1e9
    print(f"{nx}x{ny}x{nz} = {prob.nelem:,} cells, ndof={prob.ndof:,}; a dense stiffness "
          f"matrix would be {dense_gb:,.0f} GB. device={jax.devices()[0]}", flush=True)
    t0 = time.time()
    res = tr.run(prob, {"optim": {"max_iter": iters, "move": 0.2, "tol_change": 0.005}})
    dt = time.time() - t0
    rho3d = np.asarray(res["rho"]).reshape(nx, ny, nz)
    np.save(RESULTS / "highres_density.npy", rho3d)
    out = {"grid": [nx, ny, nz], "nelem": int(prob.nelem), "ndof": int(prob.ndof),
           "rmin": rmin, "dense_matrix_gb": round(dense_gb, 1),
           "final_compliance": res["final_compliance"], "final_volume": res["final_volume"],
           "n_iterations": len(res["history"]), "wall_s": round(dt, 1),
           "sec_per_iter": round(dt / len(res["history"]), 3)}
    (RESULTS / "highres_topopt.json").write_text(json.dumps(out, indent=2))
    print(f"\nDONE: C={res['final_compliance']:.3f} vol={res['final_volume']:.3f} "
          f"{len(res['history'])} iters in {dt:.1f}s ({out['sec_per_iter']}s/iter)", flush=True)


if __name__ == "__main__":
    main()
