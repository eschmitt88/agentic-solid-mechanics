#!/usr/bin/env bash
# Optional strengthening (Fable consult 03:40): beam FULL x 3 seeds x 300 generations, the number directly
# comparable to the paper's 15.1 %. Seeds s10-s12 (fresh; 100-generation runs keep their own dirs).
# Starts only after the 06:45-11:15 scheduled-job window; at most 3 t14 arm jobs at once.
set -u
cd "$(dirname "$0")"
J=~/claude-system/coordinator/.venv/bin/claude-coordinator-jobs
PY=../../.venv/bin/python
L=results/logs
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
running() { systemctl --user list-units 'job-t14-*' 'job-t14b-*' --state=active --no-legend | wc -l; }
while [ "$(date -u +%Y%m%d%H%M)" -lt 202611181115 ]; do sleep 60; done
for s in 10 11 12; do
  grep -q " launched umax_FULL_s$s\$" $L/campaign.log 2>/dev/null && continue
  while [ "$(running)" -ge 3 ]; do sleep 30; done
  $J run --no-gate --name "t14b-umax_FULL_s$s" --cores 8 --ram-gb 9 --hours 2 --class batch --log "$L/umax_FULL_s$s.log" -- \
    systemd-run --user --scope -q -p MemoryMax=16G -p MemorySwapMax=0 $PY -u run_arm.py --obj umax --arm FULL --seed $s --gens 300 --workers 8 >> "$L/umax_FULL_s$s.log" 2>&1
  echo "$(date -u +%FT%TZ) launched umax_FULL_s$s" >> $L/campaign.log
  sleep 20
done
