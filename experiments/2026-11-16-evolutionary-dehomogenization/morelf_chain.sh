#!/usr/bin/env bash
# Launch the two MORELF evaluations as separate attributed jobs once generation has written its file.
set -u
cd "$(dirname "$0")"
J=~/claude-system/coordinator/.venv/bin/claude-coordinator-jobs
PY=../../.venv/bin/python
L=results/logs
while [ ! -f results/morelf/lf_extra.npz ]; do sleep 60; done
sleep 30
$J run --no-gate --name t13-morelf-evalS --cores 10 --ram-gb 12 --hours 1.5 --class batch --log $L/morelf_evalS.log -- \
  systemd-run --user --scope -q -p MemoryMax=24G -p MemorySwapMax=0 $PY -u run_morelf.py evaluate --obj stiff --n 11000 --workers 10 >> $L/morelf_evalS.log 2>&1
$J run --no-gate --name t13-morelf-evalB --cores 10 --ram-gb 12 --hours 3 --class batch --log $L/morelf_evalB.log -- \
  systemd-run --user --scope -q -p MemoryMax=24G -p MemorySwapMax=0 $PY -u run_morelf.py evaluate --obj buck --n 3300 --workers 10 --kinds raw >> $L/morelf_evalB.log 2>&1
echo "$(date -u +%FT%TZ) morelf evals launched" >> $L/campaign3.log
