#!/bin/bash
# Submit a grid of big CH4/H2O mixture NVE runs (job_big.sh), one self-resubmitting chain each.
# Run from the directory holding job_big.sh, nve_diffusion.py and the start frames:
#   bash submit_mix_grid.sh              # submit
#   DRY=1 bash submit_mix_grid.sh        # just print what would be submitted
# Edit the lists below. FRAME_N<N> must be a start frame with N molecules at the mixture composition
# (nve_diffusion.py rescales it to each --rho).
set -euo pipefail

NS=(512)                                              # 512 and/or 1024
RHOS=(0.9311 0.9500 0.9850 1.0331 1.0570 1.0795 1.0960 1.1109)   # g/cm3, the existing mixture densities
SEEDS=(1 2)
NVE_PS=${NVE_PS:-400}
MAX_RESUBMITS=${MAX_RESUBMITS:-8}
declare -A FRAME=([512]=${FRAME_N512:-start_mix_N512.extxyz} [1024]=${FRAME_N1024:-start_mix_N1024.extxyz})

n=0
for N in "${NS[@]}"; do
  [ -f "${FRAME[$N]}" ] || { echo "missing start frame for N=$N: ${FRAME[$N]} (set FRAME_N$N=...)"; exit 1; }
  for RHO in "${RHOS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      d=nve3_grace_mix_N${N}_rho${RHO}_s${SEED}
      if [ -f "$d/summary.json" ]; then echo "done    $d"; continue; fi
      if [ -f "$d/slurm_job.id" ] && squeue -h -j "$(cat "$d/slurm_job.id")" 2>/dev/null | grep -q .; then
        echo "queued  $d (job $(cat "$d/slurm_job.id"))"; continue
      fi
      cmd=(sbatch --parsable -J "m${N}_${RHO}_s${SEED}"
           --export=ALL,FRAME=${FRAME[$N]},N=$N,RHO=$RHO,SEED=$SEED,NVE_PS=$NVE_PS,MAX_RESUBMITS=$MAX_RESUBMITS,DIR=$d
           job_big.sh)
      if [ -n "${DRY:-}" ]; then echo "would: ${cmd[*]}"; else
        mkdir -p "$d"; id=$("${cmd[@]}"); echo "$id" > "$d/slurm_job.id"; echo "submitted $d as $id"; fi
      n=$((n + 1))
    done
  done
done
echo "$n chain(s) ${DRY:+would be }submitted"
