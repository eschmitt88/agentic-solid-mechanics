#!/bin/bash
cd "$(dirname "$0")"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
while pgrep -f "tag memtest" > /dev/null || pgrep -f "dmc_queue3.sh" > /dev/null; do sleep 5; done
uv run python -u dmc_run3d.py 1.0 0.35 36 0.0506 H45,CH
uv run python -u dmc_run3d.py 0.6 0.3 42 0.0506 HC,H45,CH
uv run python -u dmc_run3d.py 0.6 0.3 1 0.0506 HB,HA _slab
uv run python -u dmc_run3d.py 1.0 0.35 1 0.0506 HB,HA _slab
echo QUEUE2B_DONE
