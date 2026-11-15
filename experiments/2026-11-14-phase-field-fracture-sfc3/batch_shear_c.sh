#!/bin/bash
# shear step-convergence extras (l=0.015, h=l/4), coarse elastic phase then fine steps
cd "$(dirname "$0")"
export XLA_PYTHON_CLIENT_PREALLOCATE=false MKL_NUM_THREADS=3 OMP_NUM_THREADS=3
PY="uv run python -u pf.py --case shear --side-uy-fixed --l 0.015 --h 0.00375 --du1 1e-4 --u-switch 0.006"
$PY --du2 2.5e-6 --u-end 0.0125 --tag s_l015_h4_du2.5e-6
$PY --du2 1e-5 --u-end 0.0115 --max-stag 100 --stag-tol 1e-4 --tag s_l015_h4_stag100
echo BATCH_SHEAR_C_DONE
