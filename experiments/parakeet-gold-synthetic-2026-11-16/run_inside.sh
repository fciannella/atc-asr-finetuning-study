#!/usr/bin/env bash
set -Eeuo pipefail
r=/lustre/fsw/portfolios/llmservice/users/fciannella/ATC_ELDA/asr-pilot/atc-ablation-v1/experiments/parakeet-gold-synthetic-2026-11-16
export PYTHONPATH="$r:/opt/prompt_model/NeMo"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 MKL_THREADING_LAYER=GNU MKL_SERVICE_FORCE_INTEL=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_DYNAMIC=FALSE MKL_DYNAMIC=FALSE
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false
cd "$r"
python verify.py
case "$1" in
 baseline) python evaluate.py base ;;
 smoke)
  env -u SLURM_NTASKS -u SLURM_NTASKS_PER_NODE -u SLURM_STEP_NUM_TASKS -u SLURM_PROCID -u SLURM_LOCALID -u SLURM_NODEID -u SLURM_GTIDS torchrun --standalone --nproc_per_node=8 train.py p_g2 --smoke --end 20
  env -u SLURM_NTASKS -u SLURM_NTASKS_PER_NODE -u SLURM_STEP_NUM_TASKS -u SLURM_PROCID -u SLURM_LOCALID -u SLURM_NODEID -u SLURM_GTIDS torchrun --standalone --nproc_per_node=8 train.py p_g2 --smoke --end 40 --resume 20
  ;;
 train)
  test -f arms/p_g2/smoke/complete-40.json
  env -u SLURM_NTASKS -u SLURM_NTASKS_PER_NODE -u SLURM_STEP_NUM_TASKS -u SLURM_PROCID -u SLURM_LOCALID -u SLURM_NODEID -u SLURM_GTIDS torchrun --standalone --nproc_per_node=8 train.py p_g2
  ;;
 evaluate)
  test -f arms/p_g2/train/complete-2000.json
  for step in 250 500 1000 1500 2000; do python evaluate.py p_g2 --step "$step"; done
  python select.py
  ;;
 *) exit 2 ;;
esac
