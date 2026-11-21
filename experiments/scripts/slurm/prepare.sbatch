#!/bin/bash
# EXAMPLE SLURM launcher: the resources below are those of the machine the runs were made on (dgx01: one node of 224 CPUs,
# 1900 GB and 8 GPUs in a partition named `gpu`). Adapt --partition, CPUs and memory to your site. Submit from the repository
# root (`EXPERIMENT=<name> sbatch experiments/scripts/slurm/prepare.sbatch`, or `bash experiments/scripts/reproduce/reproduce_results_dgx.sh`); logs
# go to ./logs/, which must exist before submission.
# Data paths come from the environment: VGTK_DATA_ROOT, VGTK_RESULTS_ROOT (common.sh).
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --job-name=rc-prepare
#SBATCH --output=logs/rc_prepare_%x_%j.log
#SBATCH --error=logs/rc_prepare_%x_%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
mkdir -p logs
source experiments/scripts/slurm/common.sh

# =============================================================================
# prepare.sbatch — the shared stages of a per-case experiment, once, before its array: `--stage=prepare,detect` of EXPERIMENT
# (radiometric_refinement, radiometric_refinement_vignetting or radiometric_refinement_board_warping) on its config of record. prepare decodes the raw photographs of every [prepare.<set>]
# (Features' load_raw, LibRaw_jll) into the set's keyed run; detect checks the
# inputs of record. Every case of the array reads the same set, so the decode runs here once instead of racing in each task; the
# array depends on this job (--dependency=afterok).
#
# CPU only: the decode is LibRaw's, and detect only checks the inputs of record; no GPU is held through CPU work (dgx01's GPU split).
# THREADS: JULIA_NUM_THREADS = the CPUs, the same count as the array's tasks: the thread
# count keys the prepare run, and a case finds the set's run by its key (src/drivers.jl prepare_run), so the two must agree.
# RESOURCES: the CPUs, the memory per CPU (8480M, the node ratio) and the GPU are the runner's — configs/reproduce.toml [[stage]]
# cpus_per_task, mem_per_cpu and gpu, the one source; a launcher submitted by hand states them on its sbatch line.
# =============================================================================

: "${EXPERIMENT:?set EXPERIMENT (radiometric_refinement, radiometric_refinement_vignetting or radiometric_refinement_board_warping)}"
case $EXPERIMENT in radiometric_refinement|radiometric_refinement_vignetting|radiometric_refinement_board_warping) ;; *) echo "EXPERIMENT=$EXPERIMENT has no prepare stage here"; exit 2 ;; esac
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

echo "[$(date +%T)] $EXPERIMENT: prepare, detect  threads=$JULIA_NUM_THREADS  data=$VGTK_DATA_ROOT  results=$VGTK_RESULTS_ROOT  commit=$(git rev-parse HEAD)"
exec "$JULIA" --startup-file=no "${HEAPHINT[@]}" --project=experiments "$(experiment_driver "$EXPERIMENT")" \
    --config="$(experiment_config "$EXPERIMENT")" --stage=prepare,detect
