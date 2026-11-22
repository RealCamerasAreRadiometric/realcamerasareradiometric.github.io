#!/bin/bash
# EXAMPLE SLURM launcher: the resources below are those of the machine the runs were made on (dgx01: one node of 224 CPUs,
# 1900 GB and 8 GPUs in a partition named `gpu`). Adapt --partition, CPUs and memory to your site. Submit from the repository
# root (`bash experiments/scripts/reproduce/reproduce_results_vcai.sh --only=image_regions`, or `sbatch --gres=gpu:1
# --cpus-per-task=8 --mem-per-cpu=8480M experiments/scripts/slurm/image_regions.sbatch`); logs go to ./logs/, which must exist before submission.
# Data paths come from the environment: VGTK_DATA_ROOT, VGTK_RESULTS_ROOT (common.sh).
#SBATCH --partition=gpu
#SBATCH --time=00:30:00
#SBATCH --job-name=rc-image-regions
#SBATCH --output=logs/%x_%A.log
#SBATCH --error=logs/%x_%A.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
mkdir -p logs
source experiments/scripts/slurm/common.sh

# =============================================================================
# image_regions.sbatch — scripts/image_regions.jl on its config of record: one task, every capture set's measured image region (stage
# detect: RadiometricCamera's measure_image_region, Features' detect_image_circle on the device; src/image_region.jl), each step
# produced or loaded by its key. DEVICE: Core's default_device() — the runner's gres makes the allocated GPU device 0 of the job
# (CUDA_VISIBLE_DEVICES); nothing pins the CPU. RESOURCES: the CPUs, the memory per CPU (8480M, the node ratio) and the GPU are the
# runner's — configs/reproduce.toml [[stage]] cpus_per_task, mem_per_cpu and gpu, the one source; a launcher submitted by hand states
# them on its sbatch line.
# =============================================================================

export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
echo "[$(date +%T)] image_regions  threads=$JULIA_NUM_THREADS  data=$VGTK_DATA_ROOT  results=$VGTK_RESULTS_ROOT  commit=$(git rev-parse HEAD)"
exec "$JULIA" --startup-file=no --project=experiments experiments/scripts/image_regions.jl \
    --config=experiments/configs/experiments/image_regions.toml --stage=detect
