#!/bin/bash
# Finest shear mesh (l=0.0075, h=l/8). Coarse steps (1e-4) in the elastic phase up to u=0.006, then the
# headline 1e-5 step; stops at u=0.0125 (past the peak). Equivalence of this schedule checked on
# s_l015_h4_coarse_start (identical peak to 4 digits).
cd "$(dirname "$0")"
export XLA_PYTHON_CLIENT_PREALLOCATE=false MKL_NUM_THREADS=3 OMP_NUM_THREADS=3
uv run python -u pf.py --case shear --side-uy-fixed --l 0.0075 --h 0.0009375 --du1 1e-4 --u-switch 0.006 --du2 1e-5 --u-end 0.0125 --tag s_l0075_h8
