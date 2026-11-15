#!/bin/bash
cd "$(dirname "$0")"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
while pgrep -f "dmc_queue2.sh" > /dev/null; do sleep 5; done
DMC_STOP=0.02 uv run python -u dmc_run3d.py 1.0 0.5 1 0.0506 HC,HB,HA _slab_full
DMC_STOP=0.02 uv run python -u dmc_run3d.py 1.0 0.5 26 0.0506 H45 _full
echo QUEUE3_DONE
