#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../envs/pydeps TOKENIZERS_PARALLELISM=false
R="taskset --cpu-list 9-25 ../../.venv/bin/python"
until [ -f outputs/g1_guided_patch/patching.json ]; do sleep 15; done
echo "[$(date '+%T')] score patched outputs"
$R score_texts.py --model g1_guided_patch > runs/interp/score_patch.log 2>&1
$R score_texts.py --model g1_guided_patch --repair >> runs/interp/score_patch.log 2>&1
until grep -q "^outputs:" runs/interp/vz.log; do sleep 15; done
echo "[$(date '+%T')] analyze"
$R interp/analyze_vz.py > runs/interp/analyze_vz.log 2>&1
echo "[$(date '+%T')] done"
