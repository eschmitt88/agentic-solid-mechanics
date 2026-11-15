#!/bin/bash
# Remaining shear runs, relaunched with 3 threads to stay inside the 12-core budget.
cd "$(dirname "$0")"
export XLA_PYTHON_CLIENT_PREALLOCATE=false MKL_NUM_THREADS=3 OMP_NUM_THREADS=3
PY="uv run python -u pf.py --case shear --side-uy-fixed"
$PY --l 0.0075 --h 0.001875 --tag s_l0075_h4
$PY --l 0.015 --h 0.00375 --du2 1e-4 --max-stag 3000 --stag-tol 1e-5 --tag s_l015_h4_stag_tol1e-5
$PY --l 0.015 --h 0.001875 --du1 1e-4 --u-switch 0.006 --du2 1e-5 --u-end 0.0125 --tag s_l015_h8
$PY --l 0.015 --h 0.00375 --du1 1e-4 --u-switch 0.006 --du2 2.5e-6 --u-end 0.0125 --tag s_l015_h4_du2.5e-6
echo BATCH_SHEAR_B_DONE
