#!/bin/bash
# Shear series. Headline BCs: bottom fixed, top u_x prescribed + u_y=0, left/right edges u_y=0
# (the BC set whose elastic stiffness, 63.5 kN/mm, matches the digitized Miehe curve; see log 20:46).
cd "$(dirname "$0")"
export XLA_PYTHON_CLIENT_PREALLOCATE=false MKL_NUM_THREADS=6 OMP_NUM_THREADS=6
PY="uv run python -u pf.py --case shear"
S="--side-uy-fixed"
$PY $S --l 0.015 --h 0.0075    --tag s_l015_h2
$PY $S --l 0.015 --h 0.00375   --tag s_l015_h4
$PY $S --l 0.0075 --h 0.00375  --tag s_l0075_h2
# ablations / traps at l=0.015, h=l/4
$PY    --l 0.015 --h 0.00375 --tag s_l015_h4_freesides
$PY $S --l 0.015 --h 0.00375 --split none --tag s_l015_h4_nosplit
$PY $S --l 0.015 --h 0.00375 --split amor --tag s_l015_h4_amor
$PY $S --l 0.015 --h 0.00375 --no-irrev   --tag s_l015_h4_noirrev
$PY $S --l 0.015 --h 0.00375 --du2 4e-5 --tag s_l015_h4_du4e-5
$PY $S --l 0.015 --h 0.00375 --du2 2.5e-6 --tag s_l015_h4_du2.5e-6
$PY $S --l 0.015 --h 0.00375 --du2 1e-4 --max-stag 3000 --stag-tol 1e-5 --tag s_l015_h4_stag_tol1e-5
$PY $S --l 0.0075 --h 0.001875 --tag s_l0075_h4
$PY $S --l 0.015 --h 0.001875  --tag s_l015_h8
$PY $S --l 0.0075 --h 0.0009375 --tag s_l0075_h8
echo BATCH_SHEAR_DONE
