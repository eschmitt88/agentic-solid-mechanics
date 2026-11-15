#!/bin/bash
# Tension series: mesh convergence (single-pass, Miehe steps), step study, staggered-iteration study.
cd "$(dirname "$0")"
export XLA_PYTHON_CLIENT_PREALLOCATE=false MKL_NUM_THREADS=5 OMP_NUM_THREADS=5
PY="uv run python -u pf.py --case tension"
$PY --l 0.015 --h 0.0075    --tag t_l015_h2
$PY --l 0.015 --h 0.00375   --tag t_l015_h4
$PY --l 0.015 --h 0.001875  --tag t_l015_h8
$PY --l 0.0075 --h 0.00375  --tag t_l0075_h2
$PY --l 0.0075 --h 0.001875 --tag t_l0075_h4
$PY --l 0.0075 --h 0.0009375 --tag t_l0075_h8
# load-step study at l=0.015, h=l/4 (single pass)
$PY --l 0.015 --h 0.00375 --du2 1e-5   --tag t_l015_h4_du1e-5
$PY --l 0.015 --h 0.00375 --du1 1e-4 --du2 1e-5 --tag t_l015_h4_molnar_steps
$PY --l 0.015 --h 0.00375 --du2 2.5e-7 --tag t_l015_h4_du2.5e-7
# staggered iterated to tolerance (du=1e-5 throughout from 5e-3; tol on max|delta d|)
$PY --l 0.015 --h 0.00375 --du2 1e-5 --max-stag 3000 --stag-tol 1e-3 --tag t_l015_h4_stag_tol1e-3
$PY --l 0.015 --h 0.00375 --du2 1e-5 --max-stag 3000 --stag-tol 1e-4 --tag t_l015_h4_stag_tol1e-4
$PY --l 0.015 --h 0.00375 --du2 1e-5 --max-stag 5000 --stag-tol 1e-5 --tag t_l015_h4_stag_tol1e-5
# finest meshes last
$PY --l 0.015 --h 0.0009375 --tag t_l015_h16
$PY --l 0.0075 --h 0.00046875 --tag t_l0075_h16
echo BATCH_TENSION_DONE
