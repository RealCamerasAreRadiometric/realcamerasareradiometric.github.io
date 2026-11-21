#!/bin/bash
# EXAMPLE SLURM launcher: the resources below are those of the machine the runs were made on (dgx01: one node of 224 CPUs,
# 1900 GB and 8 GPUs in a partition named `gpu`). Adapt --partition, CPUs and memory to your site. Submit from the repository
# root (`bash experiments/scripts/reproduce/reproduce_results_dgx.sh --only=board_warping_lenses`, or
# `BOARD_WARPING_CONFIG=<config> sbatch --cpus-per-task=8 --mem-per-cpu=8480M experiments/scripts/slurm/board_warping.sbatch`); logs go
# to ./logs/, which must exist before submission.
# Data paths come from the environment: VGTK_DATA_ROOT, VGTK_RESULTS_ROOT (common.sh).
#SBATCH --partition=gpu
#SBATCH --time=00:10:00
#SBATCH --job-name=rc-board-warping
#SBATCH --output=logs/%x_%A.log
#SBATCH --error=logs/%x_%A.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
mkdir -p logs
source experiments/scripts/slurm/common.sh

# =============================================================================
# board_warping.sbatch — scripts/board_warping.jl on one of its configs of record (BOARD_WARPING_CONFIG: configs/supplement/
# board_warping_lenses.toml, the stage board_warping_lenses of configs/reproduce.toml; or configs/supplement/board_warping.toml):
# one task, stage figures — each case's height field from its refinement's calibrated target, the tables and the figure into
# plots/. After the refinement array of the config's [upstream]: the task only reads its refine products.
#
# CPU only: the calibrated targets are records; a refinement that is missing is its own driver's GPU run, named with its command.
# THREADS: JULIA_NUM_THREADS = the CPUs, the same count as the refinement array's tasks: the thread count keys every refinement,
# and the stage finds each by its key, so the two must agree.
# RESOURCES: configs/reproduce.toml [[stage]] cpus_per_task and mem_per_cpu, the one source; a launcher submitted by hand states
# them on its sbatch line.
# =============================================================================

: "${BOARD_WARPING_CONFIG:?set BOARD_WARPING_CONFIG (a config of record of scripts/board_warping.jl, from the repository root)}"
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
echo "[$(date +%T)] board_warping  config=$BOARD_WARPING_CONFIG  threads=$JULIA_NUM_THREADS  data=$VGTK_DATA_ROOT  results=$VGTK_RESULTS_ROOT  commit=$(git rev-parse HEAD)"
exec "$JULIA" --startup-file=no --project=experiments experiments/scripts/board_warping.jl --config="$BOARD_WARPING_CONFIG" --stage=figures
