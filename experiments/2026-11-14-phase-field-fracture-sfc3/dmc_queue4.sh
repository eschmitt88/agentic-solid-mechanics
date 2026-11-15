#!/bin/bash
# crack-path mesh study on the off-centre calibration beams (slab), run until 40 steps after the load drop
cd "$(dirname "$0")"
export XLA_PYTHON_CLIENT_PREALLOCATE=false DMC_STOP=0.03 DMC_POSTPEAK_STEPS=40
uv run python -u dmc_run3d.py 1.0 0.25 1 0.0506 HB,HA _slab_full
uv run python -u dmc_run3d.py 1.0 0.125 1 0.0506 HB,HA _slab_full
uv run python -u dmc_run3d.py 0.5 0.125 1 0.0506 HB,HA _slab_full
echo QUEUE4_DONE
