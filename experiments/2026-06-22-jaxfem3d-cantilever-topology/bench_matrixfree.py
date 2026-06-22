"""bench_matrixfree.py — validate and benchmark the matrix-free (sparse) solver.

Three checks:
  (1) Correctness: matrix-free compliance == dense compliance on a small grid,
      for both uniform and random densities.
  (2) Differentiability preserved: jax.grad(compliance) matches between the
      dense and matrix-free solvers (same sensitivities).
  (3) Scaling: solve a grid far beyond the dense solver's memory limit, and
      time matrix-free vs dense where both fit.

Run (GPU): ~/projects/research/agentic-solid-mechanics/.venv/bin/python bench_matrixfree.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import jaxfem3d as j3
import jaxfem3d_mf as mf

jax.config.update("jax_enable_x64", True)
HERE = Path(__file__).parent
RESULTS = HERE / "results"


def _block(dense, ndof):
    """Dense memory estimate (GB) for the free-free stiffness block."""
    return (ndof ** 2) * 8 / 1e9


def main():
    print("device:", jax.devices()[0])
    out = {"device": str(jax.devices()[0])}

    # (1) correctness + (2) differentiability on a small grid both solvers handle
    nx, ny, nz = 16, 6, 6
    pd = j3.TopOpt3D(nx, ny, nz, 0.4, penal=3.0, rmin=1.5)
    pm = mf.TopOpt3DMatrixFree(nx, ny, nz, 0.4, penal=3.0, rmin=1.5)
    key = jax.random.PRNGKey(0)
    rho_rand = jax.random.uniform(key, (pd.nelem,), minval=0.2, maxval=1.0)
    corr = []
    for name, rho in [("uniform", jnp.full(pd.nelem, 0.4)), ("random", rho_rand)]:
        cd = float(pd.compliance(rho)); cm = float(pm.compliance(rho))
        rel = abs(cm - cd) / abs(cd)
        corr.append({"case": name, "dense": cd, "matrix_free": cm, "rel_diff": rel})
        print(f"(1) {name:7s}  dense={cd:.6f}  matrix-free={cm:.6f}  rel.diff={rel:.2e}")
    gd = np.asarray(jax.grad(pd.compliance)(rho_rand))
    gm = np.asarray(jax.grad(pm.compliance)(rho_rand))
    grad_rel = float(np.linalg.norm(gm - gd) / np.linalg.norm(gd))
    print(f"(2) gradient rel.diff (dense vs matrix-free) = {grad_rel:.2e}")
    out["correctness"] = corr
    out["gradient_rel_diff"] = grad_rel
    out["correctness_pass"] = all(c["rel_diff"] < 1e-5 for c in corr) and grad_rel < 1e-5

    # (3) scaling — time both where they fit, then a grid only matrix-free can do
    timings = []
    for nx, ny, nz in [(16, 6, 6), (24, 8, 8), (40, 12, 12)]:
        pm2 = mf.TopOpt3DMatrixFree(nx, ny, nz, 0.4, penal=3.0, rmin=1.5)
        rho = jnp.full(pm2.nelem, 0.4)
        c = float(pm2.compliance(rho))           # compile
        t0 = time.time(); c = float(pm2.compliance(rho)); dt_mf = time.time() - t0
        row = {"grid": [nx, ny, nz], "ndof": int(pm2.ndof),
               "dense_block_gb": round(_block(True, pm2.ndof), 2),
               "matrix_free_s": round(dt_mf, 3), "compliance": c}
        # dense, only if the matrix plausibly fits (< 6 GB)
        if _block(True, pm2.ndof) < 6.0:
            pd2 = j3.TopOpt3D(nx, ny, nz, 0.4, penal=3.0, rmin=1.5)
            _ = float(pd2.compliance(rho))
            t0 = time.time(); _ = float(pd2.compliance(rho)); row["dense_s"] = round(time.time() - t0, 3)
        else:
            row["dense_s"] = None  # would exceed memory
        timings.append(row)
        print(f"(3) {nx}x{ny}x{nz} ndof={pm2.ndof:7d} dense-matrix={row['dense_block_gb']:6.2f}GB "
              f"mf={row['matrix_free_s']:.3f}s dense={row['dense_s']}")
    out["scaling"] = timings

    # a grid whose dense matrix is far out of reach (~100+ GB)
    nx, ny, nz = 64, 24, 24
    big = mf.TopOpt3DMatrixFree(nx, ny, nz, 0.4, penal=3.0, rmin=1.5)
    rho = jnp.full(big.nelem, 0.4)
    t0 = time.time(); cbig = float(big.compliance(rho)); dt = time.time() - t0
    print(f"(3) BIG {nx}x{ny}x{nz} ndof={big.ndof} "
          f"(dense matrix would be {_block(True, big.ndof):.0f} GB — impossible) "
          f"matrix-free solved in {dt:.2f}s, C={cbig:.3f}")
    out["big_grid"] = {"grid": [nx, ny, nz], "ndof": int(big.ndof),
                       "dense_block_gb": round(_block(True, big.ndof), 1),
                       "matrix_free_s": round(dt, 3), "compliance": cbig}

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "matrixfree_bench.json").write_text(json.dumps(out, indent=2))
    print("\nWrote results/matrixfree_bench.json  | correctness_pass =", out["correctness_pass"])


if __name__ == "__main__":
    main()
