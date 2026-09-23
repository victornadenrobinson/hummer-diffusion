#!/bin/bash
# Full build -> NPT -> NVT -> NVE production pipeline for one (T, P, N) state
# point on a single Young A100. Self-resubmits if not finished within one
# job's 48h wall-clock limit (run_production.py checkpoints, so this is safe).
#
# Submit from the repo root:
#   sbatch hpc/young/submit_production.sh <n_molecules> <temperature_K> <pressure_GPa> <total_steps>
# e.g.
#   sbatch hpc/young/submit_production.sh 128 400 0.2 2000000
#SBATCH --job-name=mace_production
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1,tmpfs:20G
#SBATCH --time=48:00:00
#SBATCH --output=logs/production_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# Fixed path (not $0) so self-resubmission works even if Slurm copies the
# script elsewhere to run it.
SELF="hpc/young/submit_production.sh"

N_MOLECULES=${1:-128}
TEMPERATURE_K=${2:-400}
PRESSURE_GPA=${3:-0.2}
TOTAL_STEPS=${4:-2000000}
TIMESTEP_FS=0.5
STATE_DIR="results/T${TEMPERATURE_K}K-P${PRESSURE_GPA}GPa/n${N_MOLECULES}"

module purge
module load ucl-stack/2025-12
# shellcheck disable=SC1091
source "$HOME/mace-ch4-venv/bin/activate"

mkdir -p "$STATE_DIR" logs results

if [ ! -f "$STATE_DIR/nvt_equilibrated.xyz" ]; then
  python scripts/build_initial_config.py \
    --n-molecules "$N_MOLECULES" --pressure-GPa "$PRESSURE_GPA" --seed 0 \
    --output "$STATE_DIR/initial.xyz"
  python scripts/equilibrate_npt.py \
    --input "$STATE_DIR/initial.xyz" --output "$STATE_DIR/npt_equilibrated.xyz" \
    --temperature-K "$TEMPERATURE_K" --pressure-GPa "$PRESSURE_GPA" \
    --device cuda --timestep-fs "$TIMESTEP_FS" \
    --log "$STATE_DIR/npt_diagnostics.npz"
  python scripts/equilibrate_nvt.py \
    --input "$STATE_DIR/npt_equilibrated.xyz" --output "$STATE_DIR/nvt_equilibrated.xyz" \
    --temperature-K "$TEMPERATURE_K" --pressure-GPa "$PRESSURE_GPA" \
    --device cuda --timestep-fs "$TIMESTEP_FS" \
    --eos-log "results/eos_results.json"
fi

is_done() {
  [ -f "$STATE_DIR/checkpoint.npz" ] || { echo False; return; }
  python -c "import numpy as np; print(bool(np.load('$STATE_DIR/checkpoint.npz')['done']))"
}

if [ "$(is_done)" = "True" ]; then
  echo "Already done."
else
  python scripts/run_production.py \
    --input "$STATE_DIR/nvt_equilibrated.xyz" \
    --checkpoint "$STATE_DIR/checkpoint.npz" \
    --com-output "$STATE_DIR/com_trajectory.npz" \
    --device cuda \
    --timestep-fs "$TIMESTEP_FS" \
    --total-steps "$TOTAL_STEPS" \
    --max-wall-seconds 168000 # ~46.7h, buffer under the 48h cap

  if [ "$(is_done)" = "False" ]; then
    echo "Not finished within this job's walltime; resubmitting."
    sbatch "$SELF" "$N_MOLECULES" "$TEMPERATURE_K" "$PRESSURE_GPA" "$TOTAL_STEPS"
  fi
fi
