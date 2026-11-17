#!/usr/bin/env bash
# Beam (trial 14b, H8): FULL and PCA x 3 seeds x 100 generations on (vf, umax). Shares the box with
# trial 14's campaign: launches only while all t14* jobs < 4 (and 1 between 07:30 and 11:15 UTC).
set -u
cd "$(dirname "$0")"
J=~/claude-system/coordinator/.venv/bin/claude-coordinator-jobs
PY=../../.venv/bin/python
L=results/logs; mkdir -p $L
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
running() { systemctl --user list-units 'job-t14*' --state=active --no-legend | wc -l; }
maxj() { h=$(date -u +%H%M); if [ "$h" -ge 0730 ] && [ "$h" -lt 1115 ]; then echo 1; else echo 4; fi; }
launch() {
  local name=$1 hours=$2; shift 2
  while [ "$(running)" -ge "$(maxj)" ]; do sleep 30; done
  $J run --no-gate --name "t14b-$name" --cores 8 --ram-gb 9 --hours "$hours" --class batch --log "$L/$name.log" -- \
    systemd-run --user --scope -q -p MemoryMax=16G -p MemorySwapMax=0 "$@" >> "$L/$name.log" 2>&1
  echo "$(date -u +%FT%TZ) launched $name" >> $L/campaign.log
  sleep 20
}
for s in 0 1 2; do
  launch umax_FULL_s$s 1.5 $PY -u run_arm.py --obj umax --arm FULL --seed $s --gens 100 --workers 8
  launch umax_PCA_s$s 1.5 $PY -u run_arm.py --obj umax --arm PCA --seed $s --gens 100 --workers 8
done
while [ "$(systemctl --user list-units 'job-t14b-*' --state=active --no-legend | wc -l)" -gt 0 ]; do sleep 60; done
echo "$(date -u +%FT%TZ) beam campaign finished" >> $L/campaign.log
