#!/usr/bin/env bash
# Trial 13 campaign, round 3: PREREGISTRATION.md Addendum A (B arms 30 generations).
# Same throttling as campaign2.sh (3 x 10 cores, --no-gate, attributed, memory-capped).
set -u
cd "$(dirname "$0")"
J=~/claude-system/coordinator/.venv/bin/claude-coordinator-jobs
PY=../../.venv/bin/python
L=results/logs
MAXJ=3
running() { systemctl --user list-units 'job-t13*' --state=active --no-legend | wc -l; }
launch() {
  local name=$1 hours=$2; shift 2
  while [ "$(running)" -ge "$MAXJ" ]; do sleep 30; done
  $J run --no-gate --name "t13-$name" --cores 10 --ram-gb 12 --hours "$hours" --class batch --log "$L/$name.log" -- \
    systemd-run --user --scope -q -p MemoryMax=20G -p MemorySwapMax=0 "$@" >> "$L/$name.log" 2>&1
  echo "$(date -u +%FT%TZ) launched $name" >> $L/campaign3.log
  sleep 20
}
# watcher: stop the 60-gen B runs once their gen-30 archive snapshot exists (history reaches gen 31)
(
  for s in 0 1; do :; done
  while :; do
    left=0
    for s in 0 1; do
      u=$(systemctl --user list-units "job-t13-buck_FULL_s$s-*" --state=active --no-legend | awk '{print $1}')
      [ -z "$u" ] && continue
      left=1
      g=$(tail -1 results/arms/buck_FULL_s$s/history.jsonl | python3 -c 'import sys,json; print(json.loads(sys.stdin.read())["gen"])')
      if [ "$g" -ge 31 ]; then systemctl --user stop "$u"; echo "$(date -u +%FT%TZ) stopped buck_FULL_s$s at gen $g (gen-30 snapshot)" >> $L/campaign3.log; fi
    done
    [ $left -eq 0 ] && break
    sleep 60
  done
) &
launch buck_FULL_s2 3.0 $PY -u run_arm.py --obj buck --arm FULL --seed 2 --gens 30 --workers 10
launch stiff_NOSEL_s0 1.0 $PY -u run_arm.py --obj stiff --arm NOSEL --seed 0 --gens 100 --workers 10
launch morelf 4.0 bash -c "$PY -u run_morelf.py generate --n 11000 --workers 10 && $PY -u run_morelf.py evaluate --obj stiff --n 11000 --workers 10 && $PY -u run_morelf.py evaluate --obj buck --n 3300 --workers 10 --kinds raw"
launch buck_NOSEL_s0 3.0 $PY -u run_arm.py --obj buck --arm NOSEL --seed 0 --gens 30 --workers 10
launch stiff_NOMUT_s0 1.0 $PY -u run_arm.py --obj stiff --arm NOMUT --seed 0 --gens 100 --workers 10
launch stiff_NOVAE_s0 1.0 $PY -u run_arm.py --obj stiff --arm NOVAE --seed 0 --gens 100 --workers 10
launch stiff_NOSEL_s1 1.0 $PY -u run_arm.py --obj stiff --arm NOSEL --seed 1 --gens 100 --workers 10
launch stiff_NOSEL_s2 1.0 $PY -u run_arm.py --obj stiff --arm NOSEL --seed 2 --gens 100 --workers 10
launch stiff_NOMUT_s1 1.0 $PY -u run_arm.py --obj stiff --arm NOMUT --seed 1 --gens 100 --workers 10
launch stiff_NOMUT_s2 1.0 $PY -u run_arm.py --obj stiff --arm NOMUT --seed 2 --gens 100 --workers 10
launch stiff_NOVAE_s1 1.0 $PY -u run_arm.py --obj stiff --arm NOVAE --seed 1 --gens 100 --workers 10
launch stiff_NOVAE_s2 1.0 $PY -u run_arm.py --obj stiff --arm NOVAE --seed 2 --gens 100 --workers 10
wait
while [ "$(running)" -gt 0 ]; do sleep 60; done
echo "$(date -u +%FT%TZ) campaign3 finished" >> $L/campaign3.log
