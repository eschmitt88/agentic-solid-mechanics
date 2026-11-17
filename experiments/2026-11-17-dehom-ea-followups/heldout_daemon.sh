#!/usr/bin/env bash
# Grade finished arms on the held-out seeds as they appear (final archive + gen-100 snapshot).
# A run is finished when its history reaches args.json "gens". Runs in both trial 14 and 14b.
set -u
cd "$(dirname "$0")"
PY=$(pwd)/../../.venv/bin/python
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"
while :; do
  did=0
  for E in . ../2026-11-17-dehom-dcbeam; do
    for d in $E/results/arms/*/; do
      run=$(basename $d); [ -f $d/args.json ] || continue
      obj=$(python3 -c "import json; print(json.load(open('$d/args.json'))['obj'])")
      gens=$(python3 -c "import json; print(json.load(open('$d/args.json'))['gens'])")
      g=$(tail -1 $d/history.jsonl 2>/dev/null | python3 -c 'import json,sys; print(json.loads(sys.stdin.read())["gen"])' 2>/dev/null || echo 0)
      for tag in archive_g100 archive; do
        [ -f $d/$tag.npz ] || continue
        [ "$tag" = archive ] && [ "$g" -lt "$gens" ] && continue
        [ -f $E/results/heldout/${run}__${tag}.npz ] && continue
        sleep 20   # let the final save finish
        (cd $E && systemd-run --user --scope -q -p MemoryMax=10G -p MemorySwapMax=0 $PY -u heldout.py --obj $obj --runs $run --archive $tag.npz --workers 4) >> results/logs/heldout_daemon.log 2>&1
        did=1
      done
    done
  done
  [ $did -eq 0 ] && sleep 120
  [ -f results/logs/heldout_daemon.stop ] && break
done
