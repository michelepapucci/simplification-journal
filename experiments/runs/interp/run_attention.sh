#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../envs/pydeps TOKENIZERS_PARALLELISM=false
R="taskset --cpu-list 9-25 ../../.venv/bin/python"
echo "[$(date '+%T')] attention g1_guided"; $R interp/attention.py --model g1_guided --conditions all > runs/interp/attention_g1_guided.log 2>&1
echo "[$(date '+%T')] attention g1"; $R interp/attention.py --model g1 --conditions originals@40 references > runs/interp/attention_g1.log 2>&1
echo "[$(date '+%T')] done"
