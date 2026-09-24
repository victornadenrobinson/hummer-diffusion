#!/bin/bash -l
# One CH4 state point on one Young-ng GPU: NPT -> NVT -> NVE -> D.
#   sbatch run_statepoint.sbatch 0.2                # MACE-OFF23
#   sbatch run_statepoint.sbatch 0.3 mace_off24
#   for p in 0.1 0.2 0.3 0.4 0.5; do sbatch run_statepoint.sbatch $p; done
#
# Resource lines mirror the interactive session that works:
#   srun --partition=intergpu --qos=intergpu --account=allusers --gres=gpu:1 ...
# intergpu is the INTERACTIVE partition/QOS and probably has a short time
# limit. Before relying on this, check the batch GPU partition/QOS with
#   sinfo -s ; sacctmgr show assoc user=$USER format=account,partition,qos%40
# and edit the two lines marked CHECK.
#SBATCH --job-name=ch4_sp
#SBATCH --account=allusers
#SBATCH --partition=intergpu          # CHECK: batch GPU partition
#SBATCH --qos=intergpu                # CHECK: batch GPU QOS
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00               # defaults = 542k steps; ~7.5 h at 20 steps/s
#SBATCH --output=ch4_sp_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

PRESSURE=${1:?usage: sbatch run_statepoint.sbatch <pressure_GPa> [mace_off23|mace_off24] [extra args]}
MODEL=${2:-mace_off23}
shift $(( $# < 2 ? $# : 2 ))

# Activate the same Python environment you use interactively, e.g.:
# module load ...
# source ~/your-venv/bin/activate
export SHARED_POTENTIALS_DIR=${SHARED_POTENTIALS_DIR:-/lustre/scratch/mmm0037/methane/first-test-md-x12t-005}

nvidia-smi
python -u run_statepoint.py --pressure "$PRESSURE" --model "$MODEL" "$@"
