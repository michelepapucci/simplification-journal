#!/usr/bin/env bash
# Train and evaluate one v1 variant on GPU0 only, pinned to CPUs 9-25.
#   bash data/v1/experiments/run_variant.sh targets          # or source_targets
# Steps: train -> generate the main (one-dimension) and joint (three-dimension) requests -> meaning
# preservation -> READ-IT (reproduced IMPaCTS scorer) + profiling -> analysis.
# Logs go to data/v1/experiments/runs/<variant>/*.log. A variant that has already been started
# (runs/<variant>/started exists) is skipped, so two launchers cannot run it twice.
set -euo pipefail
VARIANT="$1"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
PY="$ROOT/../.venv/bin/python"
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$ROOT/envs/pydeps"
RUN="taskset --cpu-list 9-25"
mkdir -p "$HERE/runs/$VARIANT"
cd "$HERE"
if ! mkdir "runs/$VARIANT/started" 2>/dev/null; then
  echo "[$(date '+%F %T')] $VARIANT already started elsewhere, skipping"; exit 0
fi
log() { echo "[$(date '+%F %T')] $*" | tee -a "runs/$VARIANT/pipeline.log"; }

log "train";      $RUN "$PY" train.py --variant "$VARIANT"            > "runs/$VARIANT/train.log" 2>&1
log "generate";   $RUN "$PY" generate.py --variant "$VARIANT"         > "runs/$VARIANT/generate.log" 2>&1
log "meaning";    $RUN "$PY" meaning.py --model "$VARIANT"            > "runs/$VARIANT/meaning.log" 2>&1
log "readit+profiling"; $RUN "$PY" score_texts.py --model "$VARIANT"  > "runs/$VARIANT/score_texts.log" 2>&1
log "analyze";    $RUN "$PY" analyze.py                               > "runs/$VARIANT/analyze.log" 2>&1
log "done"
