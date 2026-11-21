#!/bin/bash
# EXAMPLE SLURM launcher: the resources below are those of the machine the runs were made on (dgx01: one node of 224 CPUs,
# 1900 GB and 8 GPUs in a partition named `gpu`). Adapt --partition, CPUs and memory to your site. Submit from the repository
# root (`bash experiments/scripts/reproduce/reproduce_results_dgx.sh`: the runner submits prepare.sbatch first where the experiment
# has one, and gives EXPERIMENT, STAGES, --array, --job-name and --time from configs/reproduce.toml); by hand
# `EXPERIMENT=<name> STAGES=<stages> sbatch --array=0-<n-1> --job-name=rc-<name> --time=<hh:mm:ss> experiments/scripts/slurm/case_task.sbatch`.
# Logs go to ./logs/, which must exist before submission.
# Data paths come from the environment: VGTK_DATA_ROOT, VGTK_RESULTS_ROOT (common.sh).
#SBATCH --partition=gpu
#SBATCH --output=logs/%x_%A_%a.log
#SBATCH --error=logs/%x_%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
mkdir -p logs
source experiments/scripts/slurm/common.sh

# =============================================================================
# case_task.sbatch — one array task of a per-case experiment (common.sh experiment_driver, experiment_config, experiment_cases):
# EXPERIMENT one of radiometric_refinement, radiometric_refinement_vignetting, radiometric_refinement_board_warping, convergence, detector_bias; STAGES its driver's stages of the one case (configs/reproduce.toml: start,refine for the
# refinements). One GPU per task.
#
# Task i runs case i of the selection, resolved from the config at run time (common.sh run_case_task): the experiment's selection
# variable (CASES, CHROME_CASES, VIG_CASES, BW_CASES, DETECTOR_BIAS_CASES; names, comma-separated) when set, else every case, in the
# driver's order. A further case becomes a [cases.<name>] of the config, never an edit here.
#
# Never --force: a complete keyed run under the results root is reused. Never prepare or detect where the experiment has a
# prepare stage (prepare.sbatch, once before the array; the array depends on it), and never figures: the figures and numbers.tex
# are the owner's to regenerate after review. The convergence experiment is the exception by design: its task produces its
# case's upstreams (its refinement, its chrome set's detect and index: the upstream drivers' own keyed products, reused when complete) and
# its own figure, which no paper asset reads yet (configs/reproduce.toml stage convergence). The detector bias's task runs its case's start, refinement and
# evaluation; its figure of both cases is the driver's stage figures, run afterwards on a CPU. The refine stage of the refinements refuses to run while a config's [pending] is
# not empty (src/radiometric_refinement.jl refuse_pending).
#
# GPU: the runner's gres makes the allocated GPU device 0 of the job, the pipeline's default device.
# TIME: the runner's --time is the largest selected [pipeline.lm] max_time + 3 h (configs/reproduce.toml time_margin_s); the
# refinement writes a checkpoint per iteration.
# THREADS: JULIA_NUM_THREADS = the task's CPUs, the count prepare.sbatch decodes at: the thread count keys every run (the
# prepare run a case reads, and the closed-form start sums over thread chunks), so another --cpus-per-task is another run key.
# RESOURCES: the CPUs, the memory per CPU (8480M, the node ratio) and the GPU are the runner's — configs/reproduce.toml [[stage]]
# cpus_per_task, mem_per_cpu and gpu, the one source; a launcher submitted by hand states them on its sbatch line.
# =============================================================================

: "${EXPERIMENT:?set EXPERIMENT (radiometric_refinement, radiometric_refinement_vignetting, radiometric_refinement_board_warping, convergence, detector_bias)}"
: "${STAGES:?set STAGES (the driver stages of the one case, comma-separated: configs/reproduce.toml)}"
run_case_task "$EXPERIMENT" "$STAGES"
