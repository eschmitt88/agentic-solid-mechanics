#!/usr/bin/env bash
# Trial 13 campaign, round 2 (replaces campaign.sh's queue, whose coordinator-gated units ran only
# two at a time). Box forecast is empty of heavy jobs until 09:40 UTC tomorrow, so jobs launch with
# --no-gate; this driver itself caps concurrency at 3 x 10 cores and waits on each unit.
# Order = pre-registered priority; NOVAE/NOMUT seeds 1-2 last (pre-registered sacrifice order).
set -u
cd "$(dirname "$0")"
J=~/claude-system/coordinator/.venv/bin/claude-coordinator-jobs
PY=../../.venv/bin/python
L=results/logs
MAXJ=3
running() { systemctl --user list-units 'job-t13*' --state=active --no-legend | wc -l; }
launch() {  # name hours cmd...
  local name=$1 hours=$2; shift 2
  while [ "$(running)" -ge "$MAXJ" ]; do sleep 30; done
  $J run --no-gate --name "t13-$name" --cores 10 --ram-gb 12 --hours "$hours" --class batch --log "$L/$name.log" -- \
    systemd-run --user --scope -q -p MemoryMax=20G -p MemorySwapMax=0 "$@" >> "$L/$name.log" 2>&1
  echo "$(date -u +%FT%TZ) launched $name" >> $L/campaign2.log
  sleep 20
}
launch stiff_FULL_s2 1.0 $PY -u run_arm.py --obj stiff --arm FULL --seed 2 --gens 100 --workers 10
launch buck_FULL_s0 3.0 $PY -u run_arm.py --obj buck --arm FULL --seed 0 --gens 60 --workers 10
launch buck_FULL_s1 3.0 $PY -u run_arm.py --obj buck --arm FULL --seed 1 --gens 60 --workers 10
launch buck_FULL_s2 3.0 $PY -u run_arm.py --obj buck --arm FULL --seed 2 --gens 60 --workers 10
launch buck_NOSEL_s0 3.0 $PY -u run_arm.py --obj buck --arm NOSEL --seed 0 --gens 60 --workers 10
launch morelf 3.0 bash -c "$PY -u run_morelf.py generate --n 11000 --workers 10 && $PY -u run_morelf.py evaluate --obj stiff --n 11000 --workers 10 && $PY -u run_morelf.py evaluate --obj buck --n 6600 --workers 10"
launch stiff_NOSEL_s0 1.0 $PY -u run_arm.py --obj stiff --arm NOSEL --seed 0 --gens 100 --workers 10
launch stiff_NOMUT_s0 1.0 $PY -u run_arm.py --obj stiff --arm NOMUT --seed 0 --gens 100 --workers 10
launch stiff_NOVAE_s0 1.0 $PY -u run_arm.py --obj stiff --arm NOVAE --seed 0 --gens 100 --workers 10
launch stiff_NOSEL_s1 1.0 $PY -u run_arm.py --obj stiff --arm NOSEL --seed 1 --gens 100 --workers 10
launch stiff_NOSEL_s2 1.0 $PY -u run_arm.py --obj stiff --arm NOSEL --seed 2 --gens 100 --workers 10
launch stiff_NOMUT_s1 1.0 $PY -u run_arm.py --obj stiff --arm NOMUT --seed 1 --gens 100 --workers 10
launch stiff_NOMUT_s2 1.0 $PY -u run_arm.py --obj stiff --arm NOMUT --seed 2 --gens 100 --workers 10
launch stiff_NOVAE_s1 1.0 $PY -u run_arm.py --obj stiff --arm NOVAE --seed 1 --gens 100 --workers 10
launch stiff_NOVAE_s2 1.0 $PY -u run_arm.py --obj stiff --arm NOVAE --seed 2 --gens 100 --workers 10
while [ "$(running)" -gt 0 ]; do sleep 60; done
echo "$(date -u +%FT%TZ) campaign2 finished" >> $L/campaign2.log
