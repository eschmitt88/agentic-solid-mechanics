#!/bin/bash
# Snap test: 5 arms x 11 pre-registered starts, 8 workers, each pinned to its own core (22-29); pinning all to a shared 8-core set left ~50 threads per process fighting (20 s per evaluation).
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cpu
STARTS="cos_h5_t1.6 cos_h6_t1.4 cos_h4.5_t1.2 h5_t1.6_a2 h5.5_t2.0_a2 h6_t1.4_a2 hinge_h5_t1.1_b-0.8 continue_attempt3_best hinge_h5.5_t1.4_b-0.7 h6_t1.9_a2 hinge_h5_t1.5_b-0.4"
for arm in A+O B0 A B0+O B0-legacy; do for s in $STARTS; do echo "$arm $s"; done; done |
  xargs -P 8 -L 1 --process-slot-var=SLOT bash -c 'taskset -c $((22+SLOT)) uv run python -u snap.py "$0" "$1" > "results/logs/snap_$0__$1.log" 2>&1; echo "done $0 $1 rc=$?"'
echo ALL_DONE
