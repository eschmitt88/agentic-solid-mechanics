#!/usr/bin/env bash
# Trial 14 campaign (PREREGISTRATION.md order). The box is memory-bandwidth bound, so 3 jobs x 8 workers.
# Attributed via the coordinator (--no-gate: this driver throttles), each job memory-capped.
# 07:30-11:15 UTC: at most 1 job (scheduled jobs need ~40-69 GB RAM then).
set -u
cd "$(dirname "$0")"
J=~/claude-system/coordinator/.venv/bin/claude-coordinator-jobs
PY=../../.venv/bin/python
L=results/logs; mkdir -p $L
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
running() { systemctl --user list-units 'job-t14-*' --state=active --no-legend | wc -l; }
maxj() { h=$(date -u +%H%M); if [ "$h" -ge 0730 ] && [ "$h" -lt 1115 ]; then echo 1; else echo 3; fi; }
launch() {  # name hours cmd...
  local name=$1 hours=$2; shift 2
  while [ "$(running)" -ge "$(maxj)" ]; do sleep 30; done
  $J run --no-gate --name "t14-$name" --cores 8 --ram-gb 9 --hours "$hours" --class batch --log "$L/$name.log" -- \
    systemd-run --user --scope -q -p MemoryMax=16G -p MemorySwapMax=0 "$@" >> "$L/$name.log" 2>&1
  echo "$(date -u +%FT%TZ) launched $name" >> $L/campaign.log
  sleep 20
}
A="$PY -u run_arm.py --workers 8"
for s in 0 1 2; do
  launch stiff_FULL_s$s 1.5 $A --obj stiff --arm FULL --seed $s --gens 300
  launch stiff_PCA_s$s 1.5 $A --obj stiff --arm PCA --seed $s --gens 300
done
for s in 0 1 2; do
  launch buck_FULL_s$s 2.5 $A --obj buck --arm FULL --seed $s --gens 100
  launch buck_PCA_s$s 2.5 $A --obj buck --arm PCA --seed $s --gens 100
done
for s in 3 4; do
  launch stiff_FULL_s$s 0.6 $A --obj stiff --arm FULL --seed $s --gens 100
  launch stiff_PCA_s$s 0.6 $A --obj stiff --arm PCA --seed $s --gens 100
done
launch stiff_LOTTERY_s0 0.5 $A --obj stiff --arm LOTTERY --seed 0 --gens 100
for s in 0 1; do for arm in VAE32 VAE64 RAWTH; do
  launch stiff_${arm}_s$s 0.6 $A --obj stiff --arm $arm --seed $s --gens 100
done; done
for s in 0 1; do launch stiff_ROB3_s$s 1.5 $A --obj stiff --arm ROB3 --seed $s --gens 100; done
launch pg_stiff_RAND_s0 1.0 $PY -u run_pg.py --obj stiff --op RAND --seed 0 --gens 100 --workers 8
launch pg_stiff_LLM_s0 1.0 $PY -u run_pg.py --obj stiff --op LLM --seed 0 --gens 100 --workers 8
for s in 0 1; do launch stiff_PCANOV_s$s 1.5 $A --obj stiff --arm PCANOV --seed $s --gens 300; done
while [ "$(running)" -gt 0 ]; do sleep 60; done
echo "$(date -u +%FT%TZ) campaign finished" >> $L/campaign.log
