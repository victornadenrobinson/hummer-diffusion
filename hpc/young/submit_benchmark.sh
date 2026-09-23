#!/bin/bash
# Throughput benchmark on a Young GPU node (single A100).
# Submit from the repo root: sbatch hpc/young/submit_benchmark.sh
#SBATCH --job-name=mace_benchmark
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1,tmpfs:20G
#SBATCH --time=02:00:00
#SBATCH --output=logs/benchmark_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

module purge
module load ucl-stack/2025-12

# shellcheck disable=SC1091
source "$HOME/mace-ch4-venv/bin/activate"

mkdir -p benchmarks logs
nvidia-smi

for n in 16 32 64 128 256; do
  python scripts/benchmark_throughput.py \
    --n-molecules "$n" \
    --model off23-medium \
    --device cuda \
    --n-warmup-steps 20 \
    --n-timed-steps 100 \
    --runner-label young-a100 \
    --out benchmarks/results_young.json
done

python scripts/plot_benchmarks.py --results benchmarks/results_young.json --out benchmarks/throughput_young.png
