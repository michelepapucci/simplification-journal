#!/usr/bin/env bash
# The three g1 models, one after the other on GPU0 (CPUs 9-25): g1 (plain), g1_balanced (levels 0-70 shown equally
# often), g1_guided (10% "no level" token, then classifier-free guidance). Each model is trained, then generates the
# test requests; READ-IT and profiling run in the background, one scoring job at a time (flock), while the next
# model trains. Extra steps: best-of-n samples for g1; for g1_guided, a guidance sweep on the dev requests, the
# choice of w (choose_guidance.py), and the test requests at that w. analyze.py runs at the end.
#   setsid nohup bash experiments/run_g1.sh > experiments/runs/pipeline.out 2>&1 < /dev/null &
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PY="$ROOT/../.venv/bin/python"
export CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$ROOT/envs/pydeps" TOKENIZERS_PARALLELISM=false
RUN="taskset --cpu-list 9-25"
cd "$HERE"
mkdir -p runs outputs results
LOG="runs/pipeline.log"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
score() {  # score <model> <sets...>: READ-IT (+ profiling unless samples) and parse check, serialised by a lock
  local m="$1"; shift
  ( flock 9
    log "score $m $*"
    $RUN "$PY" score_texts.py --model "$m" --sets "$@" >> "runs/$m/score_texts.log" 2>&1
    $RUN "$PY" score_texts.py --model "$m" --sets "$@" --repair >> "runs/$m/score_texts.log" 2>&1
    log "scored $m $*"
  ) 9> runs/scoring.lock
}
train_and_generate() {  # <model> <train options...>
  local m="$1"; shift
  mkdir -p "runs/$m"
  log "train $m $*"
  $RUN "$PY" train.py --name "$m" "$@" > "runs/$m/train.log" 2>&1
  log "generate $m test"
  $RUN "$PY" generate.py --model "$m" --set test > "runs/$m/generate.log" 2>&1
  $RUN "$PY" meaning.py --model "$m" --sets test > "runs/$m/meaning.log" 2>&1
}

score_originals() { ( flock 9; log "score originals"; $RUN "$PY" score_texts.py --originals > runs/originals.log 2>&1; log "scored originals" ) 9> runs/scoring.lock; }
score_originals &

train_and_generate g1
score g1 test &
log "generate g1 samples"
$RUN "$PY" generate.py --model g1 --set test --samples 3 >> runs/g1/generate.log 2>&1
$RUN "$PY" meaning.py --model g1 --sets samples-test >> runs/g1/meaning.log 2>&1
( score g1 samples-test ) &

train_and_generate g1_balanced --balance
score g1_balanced test &

train_and_generate g1_guided --null-share 0.1
score g1_guided test &
for w in 1 1.5 2 3; do
  log "generate g1_guided dev w=$w"
  $RUN "$PY" generate.py --model g1_guided --set dev --guidance "$w" >> runs/g1_guided/generate.log 2>&1
done
$RUN "$PY" meaning.py --model g1_guided --sets dev-w1 dev-w1.5 dev-w2 dev-w3 >> runs/g1_guided/meaning.log 2>&1
score g1_guided dev-w1 dev-w1.5 dev-w2 dev-w3
W=$($RUN "$PY" choose_guidance.py --model g1_guided | tail -1)
log "chosen guidance w=$W"
if [ "$W" != "1" ]; then
  $RUN "$PY" generate.py --model g1_guided --set test --guidance "$W" >> runs/g1_guided/generate.log 2>&1
  $RUN "$PY" meaning.py --model g1_guided --sets "test-w$W" >> runs/g1_guided/meaning.log 2>&1
  score g1_guided "test-w$W"
fi
wait
log "analyze"
$RUN "$PY" analyze.py > runs/analyze.log 2>&1 || log "analyze failed (see runs/analyze.log)"
log "done"
