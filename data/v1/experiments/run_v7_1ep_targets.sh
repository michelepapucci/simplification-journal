#!/usr/bin/env bash
# Dataset v7 trained for one epoch instead of three (v7_targets overfits: its validation loss is lowest at step
# 2,500 of 12,534). Validation loss every 250 steps; the adapter at the lowest one (runs/<name>/adapter_best) is the
# model that is evaluated, exactly as the other models, and then put through the interpretability experiments
# (interp/, as for v5_targets). GPU0 only, CPUs 9-25.
#   bash data/v1/experiments/run_v7_1ep_targets.sh
set -euo pipefail
NAME=v7_1ep_targets
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
ADAPTER="runs/$NAME/adapter_best"

log "train";      $RUN "$PY" train.py --variant targets --data "$ROOT/data/v7/training_set_v7.zip" --out "runs/$NAME" \
                    --target-dropout 0 --epochs 1 --eval-steps 250 --keep-best > "runs/$NAME/train.log" 2>&1
log "best: $(cat "$ADAPTER/best.json")"
log "generate";   $RUN "$PY" generate.py --variant targets --adapter "$ADAPTER" --out "outputs/$NAME" > "runs/$NAME/generate.log" 2>&1
log "interp generate"; mkdir -p "outputs/$NAME/interp"
                  $RUN "$PY" interp/generate.py --adapter "$ADAPTER" --out "outputs/$NAME/interp" > "outputs/$NAME/interp/generate.log" 2>&1
log "meaning";    $RUN "$PY" meaning.py --model "$NAME"               > "runs/$NAME/meaning.log" 2>&1
log "readit+profiling"; $RUN "$PY" score_texts.py --model "$NAME"     > "runs/$NAME/score_texts.log" 2>&1
log "parse check"; $RUN "$PY" score_texts.py --model "$NAME" --repair --skip profiling > "runs/$NAME/repair.log" 2>&1
log "analyze";    $RUN "$PY" analyze.py                               > "runs/$NAME/analyze.log" 2>&1
log "interp score"; $RUN "$PY" interp/score.py --out "outputs/$NAME/interp" --reuse outputs/v5_targets/interp > "outputs/$NAME/interp/score.log" 2>&1
log "interp analyze"; $RUN "$PY" interp/analyze.py --model "$NAME"   > "outputs/$NAME/interp/analyze.log" 2>&1
log "done"
