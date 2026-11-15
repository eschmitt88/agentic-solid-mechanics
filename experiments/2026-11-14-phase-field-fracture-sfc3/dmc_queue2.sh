#!/bin/bash
cd "$(dirname "$0")"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# wait for the first 3D batch
while pgrep -f "dmc_run3d.py 1.0 0.5 26 0.0506 HC,H45,CH,HB,HA" > /dev/null; do sleep 5; done
DMC_STOP=0.02 uv run python -u dmc_run3d.py 1.0 0.5 26 0.0506 CH _full
uv run python -u dmc_run3d.py 1.0 0.35 36 0.0506 HC,H45,CH
uv run python -u dmc_run3d.py 0.6 0.3 42 0.0506 HC,H45,CH
uv run python -u dmc_run3d.py 0.6 0.3 1 0.0506 HB,HA _slab
uv run python -u dmc_run3d.py 1.0 0.35 1 0.0506 HB,HA _slab
echo QUEUE2_DONE
