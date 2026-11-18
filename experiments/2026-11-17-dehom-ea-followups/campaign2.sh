#!/usr/bin/env bash
# Trial 14 campaign, round 2 (after the 03:40 UTC framing consult; PREREGISTRATION Addendum A):
# PCANOV, VAE64 s1+ and the problem-grain pilot are cut; ROB3 s0-s1 and buckling s3-s4 are added.
set -u
cd "$(dirname "$0")"
J=~/claude-system/coordinator/.venv/bin/claude-coordinator-jobs
PY=../../.venv/bin/python
L=results/logs
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
running() { systemctl --user list-units 'job-t14-*' 'job-t14b-*' --state=active --no-legend | wc -l; }
maxj() { h=$(date -u +%H%M); if [ "$h" -ge 0730 ] && [ "$h" -lt 1115 ]; then echo 1; else echo 3; fi; }
launch() {
  local name=$1 hours=$2; shift 2
  grep -q " launched $name\$" $L/campaign.log 2>/dev/null && return 0
  while [ "$(running)" -ge "$(maxj)" ]; do sleep 30; done
  $J run --no-gate --name "t14-$name" --cores 8 --ram-gb 9 --hours "$hours" --class batch --log "$L/$name.log" -- \
    systemd-run --user --scope -q -p MemoryMax=16G -p MemorySwapMax=0 "$@" >> "$L/$name.log" 2>&1
  echo "$(date -u +%FT%TZ) launched $name" >> $L/campaign.log
  sleep 20
}
A="$PY -u run_arm.py --workers 8"
for s in 0 1; do launch stiff_ROB3_s$s 1.5 $A --obj stiff --arm ROB3 --seed $s --gens 100; done
for s in 3 4; do
  launch buck_FULL_s$s 2.5 $A --obj buck --arm FULL --seed $s --gens 100
  launch buck_PCA_s$s 2.5 $A --obj buck --arm PCA --seed $s --gens 100
done
while [ "$(running)" -gt 0 ]; do sleep 60; done
echo "$(date -u +%FT%TZ) campaign2 finished" >> $L/campaign.log
