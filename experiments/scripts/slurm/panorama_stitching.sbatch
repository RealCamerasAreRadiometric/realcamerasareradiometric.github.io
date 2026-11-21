#!/bin/bash
# EXAMPLE SLURM launcher: the resources below are those of the machine the runs were made on (one node of 224 CPUs,
# 1900 GB and 8 GPUs in a partition named `gpu`). Adapt --partition, CPUs and memory to your site. Submit through the runner
# (`bash experiments/scripts/reproduce/reproduce_results_dgx.sh --only=panorama_stitching`: the stage panorama_stitching, its
# phases, the array over the lenses and the resources of configs/reproduce.toml); logs go to ./logs/.
# Data paths come from the environment: VGTK_DATA_ROOT, VGTK_RESULTS_ROOT (common.sh).
#SBATCH --partition=gpu
#SBATCH --output=logs/%x_%A_%a.log
#SBATCH --error=logs/%x_%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
mkdir -p logs
source experiments/scripts/slurm/common.sh

# =============================================================================
# panorama_stitching.sbatch — scripts/panorama_stitching.jl of one lens per array task: task i runs the i-th set of PS_SETS (default
# fx3_14mm,fx3_fisheye,fx3_tamron), config configs/experiments/panorama_stitching_<set>.toml, its stages PS_STAGES
# (configs/reproduce.toml phases: refine,evaluate,rooftop on the CPU — the f/4 refinement and the sweep's chain read by their keys,
# the learned V's export — then panoramas in a GPU job). The same keyed products scripts/panorama_figure.jl reads,
# which then reuses them.
# Never --force. THREADS: JULIA_NUM_THREADS = the task's CPUs (the launch's one count).
# =============================================================================

: "${PS_STAGES:?set PS_STAGES (panorama_stitching.jl stages, comma-separated: configs/reproduce.toml)}"
IFS=, read -ra SET_LIST <<< "${PS_SETS:-fx3_14mm,fx3_fisheye,fx3_tamron}"
I=${SLURM_ARRAY_TASK_ID:?an array task: the runner (scripts/reproduce), or sbatch --array=0-<n-1>}
[ "$I" -lt "${#SET_LIST[@]}" ] || { echo "task $I: PS_SETS has ${#SET_LIST[@]} set(s): ${SET_LIST[*]}"; exit 2; }
CFG=experiments/configs/experiments/panorama_stitching_${SET_LIST[$I]}.toml
[ -f "$CFG" ] || { echo "no config of record $CFG"; exit 2; }
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
echo "[$(date +%T)] set=${SET_LIST[$I]} (task $I) stages=$PS_STAGES  gpu=${CUDA_VISIBLE_DEVICES:-}  threads=$JULIA_NUM_THREADS"
echo "[$(date +%T)] data=$VGTK_DATA_ROOT  results=$VGTK_RESULTS_ROOT  commit=$(git rev-parse HEAD)"
exec "${TIMEV[@]}" "$JULIA" --startup-file=no "${HEAPHINT[@]}" --project=experiments experiments/scripts/panorama_stitching.jl --config="$CFG" --stage="$PS_STAGES"
