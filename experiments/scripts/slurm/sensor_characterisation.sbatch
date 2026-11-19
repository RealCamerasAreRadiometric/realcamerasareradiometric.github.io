#!/bin/bash
# EXAMPLE SLURM launcher: the resources below are those of the machine the runs were made on (dgx01: one node of 224 CPUs,
# 1900 GB and 8 GPUs in a partition named `gpu`). Adapt --partition, CPUs and memory to your site. Submit from the repository
# root (`sbatch experiments/scripts/slurm/sensor_characterisation.sbatch`; or `bash experiments/scripts/reproduce/reproduce_results_dgx.sh
# --only=sensor_characterisation`); logs go to ./logs/, which must exist before submission.
# Data paths come from the environment: VGTK_DATA_ROOT, VGTK_RESULTS_ROOT (common.sh) and VGTK_DATA_ROOT_UNPUBLISHED (the inputs of
# tier "unpublished": the runner sets it from configs/reproduce.toml's site tiers).
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8480M
#SBATCH --time=01:00:00
#SBATCH --job-name=rc-sensor-char
#SBATCH --output=logs/%x_%A.log
#SBATCH --error=logs/%x_%A.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
mkdir -p logs
source experiments/scripts/slurm/common.sh

# =============================================================================
# sensor_characterisation.sbatch — scripts/sensor_characterisation.jl on its config of record: one task, the characterisation by
# Core's estimators (stage evaluate; their kernels run on the frames' device, the decoded mosaics' — the host today: no GPU requested,
# none pinned). MEMORY: --mem-per-cpu=8480M is the
# node ratio (1900 GB / 224 CPU); 8 CPUs hold 68 GB, the largest burst (50 frames, 1.2 GB of codes) and the E4 series (111 frames,
# 2.7 GB) many times over.
# =============================================================================

export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
echo "[$(date +%T)] sensor_characterisation  threads=$JULIA_NUM_THREADS  data=$VGTK_DATA_ROOT  results=$VGTK_RESULTS_ROOT  commit=$(git rev-parse HEAD)"
exec "$JULIA" --startup-file=no --project=experiments experiments/scripts/sensor_characterisation.jl \
    --config=experiments/configs/experiments/sensor_characterisation.toml --stage=evaluate
