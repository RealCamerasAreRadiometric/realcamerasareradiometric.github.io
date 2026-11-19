#!/bin/bash
# EXAMPLE SLURM launcher: the resources below are those of the machine the runs were made on (dgx01: one node of 224 CPUs,
# 1900 GB and 8 GPUs in a partition named `gpu`). Adapt --partition, CPUs and memory to your site. Submit from the repository
# root, once the arrays have ended (the runner chains it after them: configs/reproduce.toml) (`sbatch experiments/scripts/slurm/regenerate_paper.sbatch`, or with
# `--dependency=afterany:<the last array>`); logs go to ./logs/, which must exist before submission.
# Data paths come from the environment: VGTK_DATA_ROOT, VGTK_RESULTS_ROOT (common.sh).
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --job-name=rc-regenerate-paper
#SBATCH --output=logs/rc_regenerate_paper_%j.log
#SBATCH --error=logs/rc_regenerate_paper_%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
mkdir -p logs
source experiments/scripts/slurm/common.sh

# =============================================================================
# regenerate_paper.sbatch — scripts/regenerate_paper.jl --gpu on one GPU, for a machine where every run goes through Slurm: the
# pending stages of the drivers of configs/figures/numbers.toml on the runs the arrays made (the chrome scores, the tables and their
# macros; the focus surfaces; the texture's figure), then numbers, verify and deploy. A workstation runs the same
# command directly (REPRODUCE.md).
#
# STAGES: REGENERATE_STAGES (comma-separated stage names of scripts/regenerate_paper.jl --plan) when set, else every stage.
# REGENERATE_RESULTS_BRANCH (set by the one-command reproduction, configs/reproduce.toml): --results-branch, nothing deployed into
# paper/; the renders stay in plotsdir() for the results-branch stage that follows.
# deploy copies a render into ../paper/fig only when it is identical to the submitted figure or new; a differing one is refused
# without its config's sign-off, and a refused stage ends the job with an error after everything before it is done — name the
# stages up to `verify` to end on the report (plots/verify.tsv) instead.
#
# GPU: the runner's gres; the regeneration checks the free memory of device 0, the job's allocated GPU ([gpu] min_free_gib).
# THREADS: JULIA_NUM_THREADS = the task's CPUs. The thread count keys every run, so it must be the arrays' count: the drivers
# find the arrays' runs by their keys (the launch's one count, configs/reproduce.toml).
# RESOURCES: the CPUs, the memory per CPU (8480M, the node ratio) and the GPU are the runner's — configs/reproduce.toml [[stage]]
# cpus_per_task, mem_per_cpu and gpu, the one source; a launcher submitted by hand states them on its sbatch line.
# of the sweep (≈ 190 GB).
# =============================================================================

export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
# --gpu only when the job holds a GPU (CUDA_VISIBLE_DEVICES): configs/reproduce.toml runs it on the CPU once every GPU stage of the
# drivers it runs is a stage of its own upstream (the chrome detect); a GPU stage still pending then stops the
# regeneration and names --gpu, never runs on the host
GPUFLAG=(); [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && GPUFLAG=(--gpu)
echo "[$(date +%T)] regenerate_paper ${GPUFLAG[*]:-(CPU)}  stages=${REGENERATE_STAGES:-all}  gpu=${CUDA_VISIBLE_DEVICES:-}  threads=$JULIA_NUM_THREADS"
echo "[$(date +%T)] data=$VGTK_DATA_ROOT  results=$VGTK_RESULTS_ROOT  commit=$(git rev-parse HEAD)"
exec "$JULIA" --startup-file=no --project=experiments experiments/scripts/regenerate_paper.jl "${GPUFLAG[@]}" ${REGENERATE_STAGES:+--stage=$REGENERATE_STAGES} \
    ${REGENERATE_RESULTS_BRANCH:+--results-branch}
