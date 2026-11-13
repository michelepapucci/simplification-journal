#!/usr/bin/env bash
# Train and evaluate the v7 "targets" model (dataset v7, data/v7/) with the v1 recipe and evaluation,
# target dropout off (the data has one example per token plus a combined one),
# on GPU0 only, pinned to CPUs 9-25. Same steps as run_variant.sh; the model is called v7_targets
# (runs/v7_targets, outputs/v7_targets) and is evaluated on the same requests as the v1 and v2 models
# (the v7 test sets are identical to v1's).
#   bash data/v1/experiments/run_v7_targets.sh
set -euo pipefail
NAME=v7_targets
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
PY="$ROOT/../.venv/bin/python"
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$ROOT/envs/pydeps"
RUN="taskset --cpu-list 9-25"
mkdir -p "$HERE/runs/$NAME"
cd "$HERE"
if ! mkdir "runs/$NAME/started" 2>/dev/null; then
  echo "[$(date '+%F %T')] $NAME already started elsewhere, skipping"; exit 0
fi
log() { echo "[$(date '+%F %T')] $*" | tee -a "runs/$NAME/pipeline.log"; }

log "train";      $RUN "$PY" train.py --variant targets --data "$ROOT/data/v7/training_set_v7.zip" --out "runs/$NAME" --target-dropout 0 > "runs/$NAME/train.log" 2>&1
log "generate";   $RUN "$PY" generate.py --variant targets --adapter "runs/$NAME/adapter" --out "outputs/$NAME" > "runs/$NAME/generate.log" 2>&1
log "meaning";    $RUN "$PY" meaning.py --model "$NAME"               > "runs/$NAME/meaning.log" 2>&1
log "readit+profiling"; $RUN "$PY" score_texts.py --model "$NAME"     > "runs/$NAME/score_texts.log" 2>&1
log "parse check"; $RUN "$PY" score_texts.py --model "$NAME" --repair --skip profiling > "runs/$NAME/repair.log" 2>&1
log "analyze";    $RUN "$PY" analyze.py                               > "runs/$NAME/analyze.log" 2>&1
log "done"
