#!/bin/bash
# EXAMPLE SLURM launcher: the resources below are those of the machine the runs were made on (dgx01: one node of 224 CPUs,
# 1900 GB and 8 GPUs in a partition named `gpu`). Adapt --partition, CPUs and memory to your site. Submit from the repository
# root, one array task per case of configs/experiments/frontends.toml ([cases.<set>], sorted: common.sh config_cases; FE_SETS
# selects, comma-separated); the runner (configs/reproduce.toml stages frontends_*) gives the resources and the array:
#   STAGE=prepare sbatch --array=0-<n-1> experiments/scripts/slurm/frontends.sbatch                       (8 CPUs: the decode, the images' threads)
#   STAGE=detect  sbatch --array=0-<n-1> --gres=gpu:1 --cpus-per-task=4 --dependency=aftercorr:<prepare job> …   (camera_calibration's GPU)
#   STAGE=init    sbatch --array=0-<n-1> --cpus-per-task=2 --dependency=aftercorr:<detect job> …        (the fits: single-threaded)
# The environment is precompiled once beforehand (setup.sbatch); no task compiles.
# Logs go to ./logs/, which must exist before submission.
# Data paths come from the environment: VGTK_DATA_ROOT, VGTK_RESULTS_ROOT (common.sh); the container image is an input of tier
# unpublished, its root the site's (configs/reproduce.toml [sites.<site>.tiers], VGTK_SITE).
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8480M
#SBATCH --time=06:00:00
#SBATCH --job-name=rc-frontends
#SBATCH --output=logs/%x_%A_%a.log
#SBATCH --error=logs/%x_%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
mkdir -p logs
source experiments/scripts/slurm/common.sh

# =============================================================================
# frontends.sbatch — one case of the front ends (scripts/frontends.jl --case=<set> --stage=$STAGE): its decode (prepare, CPU), its
# Schöps run (detect, the GPU the tool needs), its index fits (init, CPU) or its comparisons with the starts of record (evaluate, CPU). Task i runs set i of the selection, resolved from the
# config at run time. JULIA_NUM_THREADS = the task's CPUs (no front-end run is keyed by it). The GPU's use is sampled every second
# (nvidia-smi) into logs/<job>_gpu.csv when the task has one. Never --force.
# =============================================================================
: "${STAGE:?set STAGE (prepare, detect, init or evaluate)}"
mapfile -t ROWS < <(frontends_cases)
i=${SLURM_ARRAY_TASK_ID:?an array task: sbatch --array=0-<n-1>}
[ "$i" -lt "${#ROWS[@]}" ] || { echo "task $i: the selection has ${#ROWS[@]} set(s): ${ROWS[*]}"; exit 2; }
SET=${ROWS[$i]%% *}
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-28}
echo "[$(date +%T)] frontends set=$SET stage=$STAGE (task $i of ${#ROWS[@]})  gpu=${CUDA_VISIBLE_DEVICES:-}  threads=$JULIA_NUM_THREADS"
echo "[$(date +%T)] data=$VGTK_DATA_ROOT  results=$VGTK_RESULTS_ROOT  commit=$(git rev-parse HEAD)"
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader -l 1 > "logs/${SLURM_JOB_NAME}_${SLURM_ARRAY_JOB_ID}_${i}_gpu.csv" &
  SAMPLER=$!; trap 'kill $SAMPLER 2>/dev/null || true' EXIT
fi
"${TIMEV[@]}" "$JULIA" --startup-file=no "${HEAPHINT[@]}" --threads="$JULIA_NUM_THREADS" --project=experiments experiments/scripts/frontends.jl --case="$SET" --stage="$STAGE"
