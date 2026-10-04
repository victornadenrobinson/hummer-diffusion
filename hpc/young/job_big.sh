#!/bin/bash
# One big GRACE-OFF NVE run (nve_diffusion.py) that resubmits itself until it finishes.
# Submit via submit_mix_grid.sh, or directly:
#   FRAME=start_N512.extxyz N=512 RHO=1.0331 SEED=1 NVE_PS=400 sbatch job_big.sh
# Env: FRAME N RHO (required); SEED (1) NVE_PS (100) TAG ('') XARGS ('') MAX_RESUBMITS (8)
#      DIR (default nve3_grace_mix_N${N}_rho${RHO}_s${SEED}${TAG}: unique per state point and seed)
#SBATCH --job-name=big
#SBATCH --partition=gpu
#SBATCH --qos=freegpu
#SBATCH --account=allusers
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
set -o pipefail
cd "$SLURM_SUBMIT_DIR" || exit 1
SELF="$SLURM_SUBMIT_DIR/job_big.sh"
: "${N:?set N}" "${RHO:?set RHO}"
[ -f "$FRAME" ] || { echo "missing FRAME=$FRAME"; exit 1; }
SEED=${SEED:-1}; RESUBMIT=${RESUBMIT:-0}; MAX_RESUBMITS=${MAX_RESUBMITS:-8}
d=${DIR:-nve3_grace_mix_N${N}_rho${RHO}_s${SEED}${TAG:-}}
mkdir -p "$d"

# Only one live job per directory: refuse if another queued/running job owns it,
# then clear nve_diffusion.py's own lock (a job killed by timeout leaves it behind).
owner="$d/slurm_job.id"
if [ -f "$owner" ]; then
  other=$(cat "$owner")
  if [ "$other" != "$SLURM_JOB_ID" ] && squeue -h -j "$other" 2>/dev/null | grep -q .; then
    echo "!!! $d is owned by live job $other: not running"; exit 1
  fi
fi
echo "$SLURM_JOB_ID" > "$owner"
rm -f "$d/run.lock"

source ~/grace-venv/bin/activate          # same env as your interactive GRACE jobs (add any module lines you use there)
export G=$HOME/work/methane/grace-off/grace-off/models/2l/b_off_medium/seed/1/saved_model
export OMP_NUM_THREADS=4 TF_FORCE_GPU_ALLOW_GROWTH=true
echo "=== $SLURM_JOB_ID $d N=$N rho=$RHO seed=$SEED nve=${NVE_PS:-100} ps, try $RESUBMIT on $(hostname) $(date)"
# 11.5 h of the 12 h limit; SIGTERM, then SIGKILL 120 s later. nve_diffusion.py checkpoints every --ckpt-ps.
timeout -k 120 41400 python -u nve_diffusion.py --start-frame "$FRAME" --rho "$RHO" \
   --pressure-label "N$N" --model "grace:$G" --outdir "$d" --equil-ps 20 --nve-ps "${NVE_PS:-100}" \
   --ckpt-ps 5 --seed "$SEED" ${XARGS:-} >> "$d.log" 2>&1
rc=$?
echo "=== python exited $rc at $(date)"

if [ -f "$d/summary.json" ]; then echo finished; rm -f "$owner"; exit 0; fi
# Resubmit only when stopped by the wall-time guard (timeout: 124 after SIGTERM, 137 after SIGKILL).
# Anything else is a crash: resubmitting would just repeat it. Fix, then sbatch again by hand.
if [ $rc -ne 124 ] && [ $rc -ne 137 ]; then
  echo "!!! python failed (exit $rc), not a timeout: not resubmitting; see $d.log"; rm -f "$owner"; exit 1
fi
if [ "$RESUBMIT" -lt "$MAX_RESUBMITS" ]; then
  new=$(sbatch --parsable --export=ALL,RESUBMIT=$((RESUBMIT+1)),DIR="$d" -J "$SLURM_JOB_NAME" "$SELF") \
    && { echo "resubmitted as $new"; echo "$new" > "$owner"; } || echo "!!! resubmit failed"
else
  echo "!!! MAX_RESUBMITS ($MAX_RESUBMITS) reached"; rm -f "$owner"; exit 1
fi
