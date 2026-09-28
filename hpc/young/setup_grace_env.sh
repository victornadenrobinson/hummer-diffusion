#!/bin/bash
# One-time GRACE-OFF environment on Young-ng. Run on a LOGIN node:
#   bash hpc/young/setup_grace_env.sh
# Separate venv from the MACE/PyTorch one: tensorflow[and-cuda] pulls its own
# nvidia-* CUDA wheels, which can clash with PyTorch's pinned ones.
set -euo pipefail

ENV_DIR="$HOME/grace-venv"
MODELS_DIR="$HOME/grace-off"

module purge
module load ucl-stack/2025-12
module load python/3.11 2>/dev/null || module load python3 2>/dev/null || true

python3 -m venv "$ENV_DIR"
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"
pip install --upgrade pip
# tensorpotential 0.6.1 allows tensorflow<=2.20; 2.19 is what was tested.
# If the GPU check fails with CUDA/PTX/driver errors (node driver is CUDA 12.2),
# try an older TensorFlow, e.g. "tensorflow[and-cuda]==2.16.*".
pip install "tensorflow[and-cuda]==2.19.*" "tf_keras==2.19.*" "tensorpotential==0.6.1" ase

# Models are plain files in the repo (no Git LFS), ~100 MB for all of them.
if [ ! -d "$MODELS_DIR" ]; then
  git clone --depth 1 https://github.com/heid-lab/grace-off "$MODELS_DIR"
fi

python -c "import tensorflow as tf, tensorpotential; print('tensorflow', tf.__version__)"
echo "GRACE-OFF env ready: source $ENV_DIR/bin/activate"
echo "2-layer medium (published GRACE-OFF 2L-M): $MODELS_DIR/models/2l/b_off_medium/seed/1/saved_model  (float64)"
echo "                                            $MODELS_DIR/models/2l/b_off_medium/seed/1/casted_model (float32)"
echo "Next, on a GPU node: python exploratory/grace_check.py"
