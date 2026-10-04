#!/bin/bash
# Runs every method on every dataset (original + perturbed), one after another. Safe to stop and re-run: finished logs are skipped.
#   bash run_all.sh smoke         # ~3 logs incl. the longest one, all methods: check speed and context before the full run
#   bash run_all.sh               # everything
#   SAMPLE=50 bash run_all.sh     # same fixed random 50 logs in every dataset
# Several workers (e.g. one per GPU, each with its own OLLAMA_URL) can share the job list: give them the same CLAIMS folder
# (empty at the start) and each job is run by one worker only. DEADLINE (Unix time) stops a worker cleanly before a time limit.
# Run it inside tmux or with nohup so it survives a dropped SSH connection. Progress: tail -f logs/*.log
set -uo pipefail
cd "$(dirname "$0")"
PY=${PY:-venv/bin/python}
mkdir -p logs

if [ "${1:-}" = "smoke" ]; then
  # Separate results folder (so a --sample run is not mixed with these logs), but the answers are cached and reused by the full run.
  # AG 80 is the longest Algorithm-Generated log and the fewest characters per token; HC 30 is the longest log overall
  "$PY" run.py --data "Who&When/Algorithm-Generated" --method all_at_once,step_by_step,binary_search,dcfa --files 1.json,80.json --out results_smoke || exit 1
  "$PY" run.py --data "Who&When/Hand-Crafted" --method all_at_once,step_by_step,binary_search,dcfa --files 30.json --out results_smoke || exit 1
  for f in results_smoke/*/*/*.jsonl; do echo "== $f"; "$PY" -c "
import json, sys
for r in map(json.loads, open(sys.argv[1])):
    print(f\"  {r['file']:8} {r.get('seconds', '?'):>7}s  step={r['step']}  {r.get('error', '')}{r.get('trace', {}).get('fallback', '')}\")" "$f"; done
  exit 0
fi

SAMPLE_ARG=${SAMPLE:+--sample $SAMPLE}
WORKER=${WORKER:-}
DATASETS=("Who&When/Algorithm-Generated" "Who&When/Hand-Crafted" perturbed/*)
# Cheap baselines on everything first, so a stopped run still leaves complete baseline tables; DCFA last
for methods in all_at_once,step_by_step,binary_search dcfa; do
  for d in "${DATASETS[@]}"; do
    name=$(basename "$d")
    if [ -n "${CLAIMS:-}" ]; then
      mkdir -p "$CLAIMS"
      mkdir "$CLAIMS/$name.$methods" 2>/dev/null || continue  # another worker has this job
    fi
    limit=()
    if [ -n "${DEADLINE:-}" ]; then
      left=$(( DEADLINE - $(date +%s) ))
      if [ "$left" -lt 120 ]; then echo "$(date '+%F %T') ${WORKER}time is up; stopping" | tee -a logs/progress.txt; exit 0; fi
      # SIGINT makes run.py stop and keep every finished log; the unfinished one is redone next time
      limit=(timeout -s INT -k 120 "$left")
    fi
    echo "$(date '+%F %T') ${WORKER}$name $methods" | tee -a logs/progress.txt
    ${limit[@]+"${limit[@]}"} "$PY" run.py --data "$d" --method "$methods" $SAMPLE_ARG >> "logs/$name.log" 2>&1
    if [ $? -ne 0 ]; then
      if [ -n "${DEADLINE:-}" ] && [ $(( DEADLINE - $(date +%s) )) -lt 120 ]; then
        echo "$(date '+%F %T') ${WORKER}time is up; stopped during $name $methods (continues next run)" | tee -a logs/progress.txt; exit 0
      fi
      echo "${WORKER}Stopped on $name (Ollama unreachable?). Re-run to continue." | tee -a logs/progress.txt; exit 1
    fi
  done
done
echo "$(date '+%F %T') ${WORKER}all done" | tee -a logs/progress.txt
