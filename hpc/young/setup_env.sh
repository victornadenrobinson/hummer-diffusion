#!/bin/bash
# One-time environment setup on UCL's Young (Young-ng / Slurm portion).
# Run this on a LOGIN node, not inside a job: bash hpc/young/setup_env.sh
set -euo pipefail

module purge
module load ucl-stack/2025-12
module load default-modules/2025-12
# Module name/version for Python varies by stack update; check `module avail python`
# if this fails and adjust.
module load python/3.11 2>/dev/null || module load python3 2>/dev/null || true

ENV_DIR="$HOME/mace-ch4-venv"
python3 -m venv "$ENV_DIR"
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"

pip install --upgrade pip
# GPU nodes have a 12.2 driver (CUDA 12 minor-version compatible); the
# default PyPI torch wheel bundles its own CUDA runtime, so no system CUDA
# module is needed just to run (only to build custom CUDA extensions).
pip install torch
pip install ase numpy scipy matplotlib mace-torch
pip install -e "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" --no-deps

echo "Environment ready at $ENV_DIR"
echo "Activate it in job scripts with: source $ENV_DIR/bin/activate"
