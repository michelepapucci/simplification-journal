#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES=0 PYTHONPATH=../envs/pydeps TOKENIZERS_PARALLELISM=false
R="taskset --cpu-list 9-25 ../../.venv/bin/python"
echo "[$(date '+%T')] classes g1_guided"; $R interp/classes.py --model g1_guided > runs/interp/classes_g1_guided.log 2>&1
echo "[$(date '+%T')] classes g1"; $R interp/classes.py --model g1 > runs/interp/classes_g1.log 2>&1
echo "[$(date '+%T')] profile references"; $R ../tools/profiling/profile_texts.py ../data/g1/test_references_g1.csv --column simplification --out outputs/interp/profiling-references.csv > runs/interp/profile_refs.log 2>&1
echo "[$(date '+%T')] done"
