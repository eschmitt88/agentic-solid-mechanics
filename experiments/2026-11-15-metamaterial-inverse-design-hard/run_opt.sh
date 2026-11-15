#!/bin/bash
# Nominal optimisations of H1, H4 (6 starts each) and H2 (4 stack starts): 2 workers.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cpu
{ for s in arch6 arch8 wavy arch3 flat sag3; do echo "H1 $s"; echo "H4 $s"; done; for s in A4_B6 A3_B5 A6_B6 Aflat_B6; do echo "H2 $s"; done; } |
  xargs -P 2 -L 1 --process-slot-var=SLOT bash -c 'taskset -c $((18+SLOT)) uv run python -u design.py "$0" "$1" > "results/logs/opt_$0__$1.log" 2>&1; echo "done $0 $1 rc=$?"'
echo ALL_DONE
