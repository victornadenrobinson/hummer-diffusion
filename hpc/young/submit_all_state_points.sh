#!/bin/bash
# Submit the finite-size pair (N=64, N=128) for the first state point (400K, 0.2GPa).
# Run from the repo root: bash hpc/young/submit_all_state_points.sh
set -euo pipefail

TOTAL_STEPS=${1:-2000000}

sbatch hpc/young/submit_production.sh 64 400 0.2 "$TOTAL_STEPS"
sbatch hpc/young/submit_production.sh 128 400 0.2 "$TOTAL_STEPS"
