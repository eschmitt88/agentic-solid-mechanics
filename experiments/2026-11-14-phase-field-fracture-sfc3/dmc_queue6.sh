#!/bin/bash
cd "$(dirname "$0")"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
while ps -u $USER -o args | grep -v grep | grep -q "dmc_queue5.sh"; do sleep 10; done
uv run python -u dmc_run3d.py 0.6 0.3 42 0.0506 CH
echo QUEUE6_DONE
