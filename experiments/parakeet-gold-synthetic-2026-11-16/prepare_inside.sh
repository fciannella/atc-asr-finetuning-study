#!/usr/bin/env bash
set -Eeuo pipefail
r=/lustre/fsw/portfolios/llmservice/users/fciannella/ATC_ELDA/asr-pilot/atc-ablation-v1/experiments/parakeet-gold-synthetic-2026-11-16
export PYTHONPATH="$r:/opt/prompt_model/NeMo"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
python "$r/runtime.py"
python "$r/prepare.py"
