#!/bin/bash
# Snap test, second queue (after the A+O batch): A+O2 (post-hoc) and B0-legacy on the 11 pre-registered starts, then both
# on 6 fresh starts, then B0+O on the pre-registered starts.  Arms A and B0 are the first phase of A+O / B0+O (same code path until the first
# outer-loop event; the bifurcation escape never fires because no start's path reports a branch switch) and are read
# from those runs' histories instead of being run again (snap_analyse.py).
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cpu
STARTS="cos_h5_t1.6 cos_h6_t1.4 cos_h4.5_t1.2 h5_t1.6_a2 h5.5_t2.0_a2 h6_t1.4_a2 hinge_h5_t1.1_b-0.8 continue_attempt3_best hinge_h5.5_t1.4_b-0.7 h6_t1.9_a2 hinge_h5_t1.5_b-0.4"
FRESH="fresh_0 fresh_1 fresh_2 fresh_3 fresh_4 fresh_5"
{ for arm in A+O2 B0-legacy; do for s in $STARTS; do echo "$arm $s"; done; done; for arm in A+O2 B0-legacy; do for s in $FRESH; do echo "$arm $s"; done; done; for s in $STARTS; do echo "B0+O $s"; done; } |
  xargs -P ${NW:-8} -L 1 --process-slot-var=SLOT bash -c 'taskset -c $((${C0:-22}+SLOT)) uv run python -u snap.py "$0" "$1" > "results/logs/snap_$0__$1.log" 2>&1; echo "done $0 $1 rc=$?"'
echo ALL_DONE
