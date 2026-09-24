#!/bin/bash
# P(rho) isotherm for CH4 (default 450 K, 128 molecules) on one A100 via
# scripts/eos_scan.py: chained fixed-volume NVT at 13 densities, 0.200-0.500 g/cm3.
#
# Submit from the repo root:
#   sbatch hpc/young/submit_eos_scan.sh [model] [model_path|-] [extra eos_scan.py args...]
# e.g.
#   sbatch hpc/young/submit_eos_scan.sh
#   sbatch hpc/young/submit_eos_scan.sh off24-medium /path/to/MACE-OFF24_medium.model
# Resubmitting skips densities that already finished.
#SBATCH --job-name=ch4_eos_scan
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1,tmpfs:20G
#SBATCH --time=24:00:00
#SBATCH --output=logs/eos_scan_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

MODEL=${1:-off23-medium}
MODEL_PATH=${2:--}
shift $(( $# < 2 ? $# : 2 ))
TEMPERATURE_K=450
OUTDIR="results/eos_${TEMPERATURE_K}K/${MODEL}"

module purge
module load ucl-stack/2025-12
# shellcheck disable=SC1091
source "$HOME/mace-ch4-venv/bin/activate"

mkdir -p "$OUTDIR" logs
nvidia-smi

MODEL_PATH_ARGS=()
if [ "$MODEL_PATH" != "-" ]; then MODEL_PATH_ARGS=(--model-path "$MODEL_PATH"); fi

python scripts/eos_scan.py \
  --temperature-K "$TEMPERATURE_K" \
  --n-molecules 128 \
  --model "$MODEL" \
  "${MODEL_PATH_ARGS[@]}" \
  --device cuda \
  --outdir "$OUTDIR" \
  "$@"
