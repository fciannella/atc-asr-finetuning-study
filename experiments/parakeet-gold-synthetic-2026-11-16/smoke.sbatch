#!/usr/bin/env bash
#SBATCH --account=nemotron_speechprod_asr
#SBATCH --partition=interactive_singlenode
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=64
#SBATCH --mem=600G
#SBATCH --time=01:00:00
#SBATCH --exclude=batch-block5-00642
#SBATCH --job-name=atc-pg2-smoke
#SBATCH --output=/lustre/fsw/portfolios/llmservice/users/fciannella/ATC_ELDA/asr-pilot/atc-ablation-v1/experiments/parakeet-gold-synthetic-2026-11-16/smoke-%j.log
set -Eeuo pipefail
u=/lustre/fsw/portfolios/llmservice/users/fciannella
r=$u/ATC_ELDA/asr-pilot/atc-ablation-v1/experiments/parakeet-gold-synthetic-2026-11-16
srun --container-image="$u/containers/nemotron-asr-finetune-v0.8.sqsh" --container-mounts=/lustre/fsw/portfolios/llmservice:/lustre/fsw/portfolios/llmservice bash "$r/run_inside.sh" smoke
