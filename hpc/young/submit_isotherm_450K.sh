#!/bin/bash
# 450 K isotherm, 128 CH4: one array task per pressure, each running
# scripts/run_state_point.py (build -> NPT -> NVT -> NVE -> D_PBC) on one A100.
#
# Submit from the repo root:
#   sbatch hpc/young/submit_isotherm_450K.sh [model] [model_path]
# e.g.
#   sbatch hpc/young/submit_isotherm_450K.sh off23-medium
#   sbatch hpc/young/submit_isotherm_450K.sh off24-medium /path/to/MACE-OFF24_medium.model
# Rerun a single pressure with --array=<index> (0-4 = 0.1-0.5 GPa); finished
# stages are skipped, an unfinished NVE restarts from its beginning.
#SBATCH --job-name=ch4_450K
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1,tmpfs:20G
#SBATCH --time=48:00:00
#SBATCH --output=logs/isotherm_450K_%A_%a.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

MODEL=${1:-off23-medium}
MODEL_PATH=${2:-}
PRESSURES=(0.1 0.2 0.3 0.4 0.5)
PRESSURE_GPA=${PRESSURES[$SLURM_ARRAY_TASK_ID]}
OUTDIR="results/isotherm_450K/${MODEL}/P${PRESSURE_GPA}GPa"

module purge
module load ucl-stack/2025-12
# shellcheck disable=SC1091
source "$HOME/mace-ch4-venv/bin/activate"

mkdir -p "$OUTDIR" logs
nvidia-smi

python scripts/run_state_point.py \
  --pressure-GPa "$PRESSURE_GPA" \
  --temperature-K 450 \
  --n-molecules 128 \
  --model "$MODEL" \
  ${MODEL_PATH:+--model-path "$MODEL_PATH"} \
  --device cuda \
  --seed "$SLURM_ARRAY_TASK_ID" \
  --outdir "$OUTDIR"
