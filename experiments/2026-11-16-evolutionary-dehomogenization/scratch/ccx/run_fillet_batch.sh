#!/bin/bash
# fillet-rounding calibration batch (memory-capped by the caller)
cd "$(dirname "$0")"
PY=/mnt/projects/research/agentic-solid-mechanics/.venv/bin/python
for d in lat_w0.30 lat_w0.15; do
  OMP_NUM_THREADS=4 $PY ccx_checks.py $d 0.5,1.0 1.0,0.5,0.25 phi fillet
done
OMP_NUM_THREADS=4 $PY ccx_checks.py solid 0 0.25 phi fillet
