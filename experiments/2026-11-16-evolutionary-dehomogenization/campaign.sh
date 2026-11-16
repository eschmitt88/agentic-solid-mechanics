#!/usr/bin/env bash
# Trial 13 campaign: pre-registered arms in priority order, 3 concurrent jobs x 10 workers.
# Each job is attributed via the coordinator and memory-capped (see log.md 15:10 UTC).
set -u
cd "$(dirname "$0")"
J=~/claude-system/coordinator/.venv/bin/claude-coordinator-jobs
PY=../../.venv/bin/python
L=results/logs
# wait for the shared gen-0 caches
while [ ! -f results/gen0/eval_buck.json ] || [ ! -f results/gen0/eval_stiff.json ]; do sleep 30; done
job() {  # name hours cmd...
  local name=$1 hours=$2; shift 2
  $J run --name "t13-$name" --cores 10 --ram-gb 12 --hours "$hours" --class batch --log "$L/$name.log" -- \
    systemd-run --user --scope -q -p MemoryMax=20G -p MemorySwapMax=0 "$@" >> "$L/$name.log" 2>&1
  echo "$(date -u +%FT%TZ) done $name rc=$?" >> $L/campaign.log
}
export -f job; export J PY L
cat > $L/queue.txt <<Q
stiff_FULL_s0 1.0 $PY -u run_arm.py --obj stiff --arm FULL --seed 0 --gens 100 --workers 10
stiff_FULL_s1 1.0 $PY -u run_arm.py --obj stiff --arm FULL --seed 1 --gens 100 --workers 10
stiff_FULL_s2 1.0 $PY -u run_arm.py --obj stiff --arm FULL --seed 2 --gens 100 --workers 10
buck_FULL_s0 1.5 $PY -u run_arm.py --obj buck --arm FULL --seed 0 --gens 60 --workers 10
buck_FULL_s1 1.5 $PY -u run_arm.py --obj buck --arm FULL --seed 1 --gens 60 --workers 10
buck_FULL_s2 1.5 $PY -u run_arm.py --obj buck --arm FULL --seed 2 --gens 60 --workers 10
buck_NOSEL_s0 1.5 $PY -u run_arm.py --obj buck --arm NOSEL --seed 0 --gens 60 --workers 10
morelf 2.5 bash -c "$PY -u run_morelf.py generate --n 11000 --workers 10 && $PY -u run_morelf.py evaluate --obj stiff --n 11000 --workers 10 && $PY -u run_morelf.py evaluate --obj buck --n 6600 --workers 10"
stiff_NOSEL_s0 1.0 $PY -u run_arm.py --obj stiff --arm NOSEL --seed 0 --gens 100 --workers 10
stiff_NOMUT_s0 1.0 $PY -u run_arm.py --obj stiff --arm NOMUT --seed 0 --gens 100 --workers 10
stiff_NOVAE_s0 1.0 $PY -u run_arm.py --obj stiff --arm NOVAE --seed 0 --gens 100 --workers 10
stiff_NOSEL_s1 1.0 $PY -u run_arm.py --obj stiff --arm NOSEL --seed 1 --gens 100 --workers 10
stiff_NOSEL_s2 1.0 $PY -u run_arm.py --obj stiff --arm NOSEL --seed 2 --gens 100 --workers 10
stiff_NOMUT_s1 1.0 $PY -u run_arm.py --obj stiff --arm NOMUT --seed 1 --gens 100 --workers 10
stiff_NOMUT_s2 1.0 $PY -u run_arm.py --obj stiff --arm NOMUT --seed 2 --gens 100 --workers 10
stiff_NOVAE_s1 1.0 $PY -u run_arm.py --obj stiff --arm NOVAE --seed 1 --gens 100 --workers 10
stiff_NOVAE_s2 1.0 $PY -u run_arm.py --obj stiff --arm NOVAE --seed 2 --gens 100 --workers 10
Q
# one line per job: name hours command...
while IFS= read -r line; do printf '%s\0' "$line"; done < $L/queue.txt | \
  xargs -0 -n 1 -P 3 bash -c 'eval "job $1"' _
echo "$(date -u +%FT%TZ) campaign finished" >> $L/campaign.log
