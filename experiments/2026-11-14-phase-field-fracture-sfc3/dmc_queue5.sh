#!/bin/bash
# challenge geometry on the finer mesh, run through the post-peak tail for the crack path
cd "$(dirname "$0")"
export XLA_PYTHON_CLIENT_PREALLOCATE=false DMC_STOP=0.03 DMC_POSTPEAK_STEPS=40
while ps -u $USER -o args | grep -v grep | grep -q "dmc_queue2b.sh"; do sleep 10; done
uv run python -u dmc_run3d.py 1.0 0.35 36 0.0506 CH _full
echo QUEUE5_DONE
