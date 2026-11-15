#!/bin/bash
# Snap test, third queue: the eigenvalue-margin arm AM+O2 (post-hoc) first, then what the second queue had not reached.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cpu
STARTS="cos_h5_t1.6 cos_h6_t1.4 cos_h4.5_t1.2 h5_t1.6_a2 h5.5_t2.0_a2 h6_t1.4_a2 hinge_h5_t1.1_b-0.8 continue_attempt3_best hinge_h5.5_t1.4_b-0.7 h6_t1.9_a2 hinge_h5_t1.5_b-0.4"
FRESH="fresh_0 fresh_1 fresh_2 fresh_3 fresh_4 fresh_5"
{ for s in $STARTS; do echo "AM+O2 $s"; done; for s in $STARTS; do echo "B0-legacy $s"; done; for arm in AM+O2 A+O2 B0-legacy; do for s in $FRESH; do echo "$arm $s"; done; done; for s in $STARTS; do echo "B0+O $s"; done; } |
  while read arm s; do [ -f "results/snap/${arm}__${s}.json" ] || echo "$arm $s"; done |
  xargs -P ${NW:-8} -L 1 --process-slot-var=SLOT bash -c 'taskset -c $((${C0:-22}+SLOT)) uv run python -u snap.py "$0" "$1" > "results/logs/snap_$0__$1.log" 2>&1; echo "done $0 $1 rc=$?"'
echo ALL_DONE
