#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../envs/pydeps TOKENIZERS_PARALLELISM=false
R="taskset --cpu-list 9-25 ../../.venv/bin/python"
echo "[$(date '+%T')] lexicon"; $R interp/lexicon.py > runs/interp/lexicon.log 2>&1
echo "[$(date '+%T')] logits g1_guided"; $R interp/logits.py --model g1_guided > runs/interp/logits_g1_guided.log 2>&1
echo "[$(date '+%T')] logits g1"; $R interp/logits.py --model g1 > runs/interp/logits_g1.log 2>&1
echo "[$(date '+%T')] analyze"; $R interp/analyze.py > runs/interp/analyze.log 2>&1
echo "[$(date '+%T')] done"
