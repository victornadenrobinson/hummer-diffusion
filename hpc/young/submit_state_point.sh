#!/bin/bash
# One CH4 state point (default 450 K, 128 molecules) on one A100:
# scripts/run_state_point.py volume -> NVT -> NVE -> D_PBC.
#
# Submit from the repo root, one job per pressure:
#   sbatch hpc/young/submit_state_point.sh <pressure_GPa> [model] [model_path|-] [extra run_state_point.py args...]
# e.g.
#   sbatch hpc/young/submit_state_point.sh 0.2
#   sbatch hpc/young/submit_state_point.sh 0.3 off24-medium /path/to/MACE-OFF24_medium.model
#   # volume from an EOS scan instead of NPT:
#   sbatch hpc/young/submit_state_point.sh 0.2 off23-medium - \
#       --eos-fit results/eos_450K/off23-medium/eos_fit.json
#   for p in 0.1 0.2 0.3 0.4 0.5; do sbatch hpc/young/submit_state_point.sh "$p"; done
# Resubmitting the same pressure skips finished stages; an unfinished NVE
# restarts from its beginning.
#SBATCH --job-name=ch4_state_point
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1,tmpfs:20G
#SBATCH --time=48:00:00
#SBATCH --output=logs/state_point_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

PRESSURE_GPA=${1:?usage: submit_state_point.sh <pressure_GPa> [model] [model_path|-] [extra args...]}
MODEL=${2:-off23-medium}
MODEL_PATH=${3:--}
shift $(( $# < 3 ? $# : 3 ))
TEMPERATURE_K=450
OUTDIR="results/T${TEMPERATURE_K}K/${MODEL}/P${PRESSURE_GPA}GPa"

module purge
module load ucl-stack/2025-12
# shellcheck disable=SC1091
source "$HOME/mace-ch4-venv/bin/activate"

mkdir -p "$OUTDIR" logs
nvidia-smi

MODEL_PATH_ARGS=()
if [ "$MODEL_PATH" != "-" ]; then MODEL_PATH_ARGS=(--model-path "$MODEL_PATH"); fi

python scripts/run_state_point.py \
  --pressure-GPa "$PRESSURE_GPA" \
  --temperature-K "$TEMPERATURE_K" \
  --n-molecules 128 \
  --model "$MODEL" \
  "${MODEL_PATH_ARGS[@]}" \
  --device cuda \
  --outdir "$OUTDIR" \
  "$@"
