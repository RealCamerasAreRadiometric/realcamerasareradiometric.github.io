#!/bin/bash
# EXAMPLE SLURM launcher: the resources below are those of the machine the runs were made on (dgx01: one node of 224 CPUs,
# 1900 GB and 8 GPUs in a partition named `gpu`). Adapt --partition, CPUs and memory to your site. Submit from the repository
# root (`bash experiments/scripts/reproduce/reproduce_results_dgx.sh --only=distance_maps`, or `sbatch --cpus-per-task=8
# --mem-per-cpu=8480M experiments/scripts/slurm/distance_maps.sbatch`); logs go to ./logs/, which must exist before submission.
# Data paths come from the environment: VGTK_DATA_ROOT, VGTK_RESULTS_ROOT (common.sh).
#SBATCH --partition=gpu
#SBATCH --time=00:30:00
#SBATCH --job-name=rc-distance-maps
#SBATCH --output=logs/%x_%A.log
#SBATCH --error=logs/%x_%A.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
mkdir -p logs
source experiments/scripts/slurm/common.sh

# =============================================================================
# distance_maps.sbatch — scripts/distance_maps.jl on its config of record: one task, every printed board's edge distance map (stage
# prepare: the generator's sdf.f32 + sdf.json as an IO raster record; src/distance_maps.jl), each step produced or loaded by its key —
# once, before the stages whose cases read them through [runs] (a case array's tasks then only load them). CPU: a conversion of two
# files, no device work. RESOURCES: configs/reproduce.toml [[stage]] cpus_per_task and mem_per_cpu, the one source; a launcher
# submitted by hand states them on its sbatch line.
# =============================================================================

export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
echo "[$(date +%T)] distance_maps  threads=$JULIA_NUM_THREADS  data=$VGTK_DATA_ROOT  results=$VGTK_RESULTS_ROOT  commit=$(git rev-parse HEAD)"
exec "$JULIA" --startup-file=no --project=experiments experiments/scripts/distance_maps.jl \
    --config=experiments/configs/experiments/distance_maps.toml --stage=prepare
