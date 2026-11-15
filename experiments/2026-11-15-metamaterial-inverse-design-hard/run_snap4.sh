#!/bin/bash
# Exploratory: the margin arm with twice the budget (120 evaluations) on the six pre-registered starts where it had not
# converged inside 60.  Answers "out of budget, or stalled?".  4 workers on cores 16-19.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cpu
for s in h5.5_t2.0_a2 cos_h5_t1.6 hinge_h5_t1.5_b-0.4 hinge_h5.5_t1.4_b-0.7 hinge_h5_t1.1_b-0.8 h5_t1.6_a2; do echo "AM+O2@120 $s"; done |
  xargs -P 4 -L 1 --process-slot-var=SLOT bash -c 'taskset -c $((16+SLOT)) uv run python -u snap.py "$0" "$1" > "results/logs/snap_$0__$1.log" 2>&1; echo "done $0 $1 rc=$?"'
echo ALL_DONE
