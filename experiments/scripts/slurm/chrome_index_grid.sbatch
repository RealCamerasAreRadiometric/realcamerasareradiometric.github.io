#!/bin/bash
# EXAMPLE SLURM launcher: the resources below are those of the machine the runs were made on (dgx01: one node of 224 CPUs,
# 1900 GB and 8 GPUs in a partition named `gpu`). Adapt --partition, CPUs and memory to your site. Submit from the repository
# root (`bash experiments/scripts/reproduce/reproduce_results_vcai.sh --only=chrome_index_grid`, or `sbatch` it); logs go to ./logs/,
# which must exist before submission.
# Data paths come from the environment: VGTK_DATA_ROOT, VGTK_RESULTS_ROOT (common.sh).
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --job-name=rc-chrome-index-grid
#SBATCH --output=logs/rc_chrome_index_grid_%A_%a.log
#SBATCH --error=logs/rc_chrome_index_grid_%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
mkdir -p logs
source experiments/scripts/slurm/common.sh

# =============================================================================
# chrome_index_grid.sbatch — the launcher of the stage chrome_index_grid of configs/reproduce.toml: the chrome index stage's
# comparison grid (scripts/chrome_index_grid.jl, config of record configs/experiments/chrome_index_grid.toml) in three phases, each
# a keyed run under the results root, reused when complete (no --force):
#   (submitted)        detect: every set's prepare and detect, on the GPU; then it submits the two phases below
#   GRID_PHASE=task    an array, one task per [grid] set (--set=<the SLURM_ARRAY_TASK_ID-th set>, listed here by Python's
#                      tomllib), each the set's index and evaluate stages under every variant in one Julia (the code compiled
#                      once per set: per (set, variant) the compile dominated, 940 s for 18 tasks against 1014 s serial), on the
#                      CPU (no GPU)
#   GRID_PHASE=table   after every task (afterok): the table (--stage=evaluate)
#
# GPU: the detect stage runs on the GPU of the configs of record ([detect] device = "gpu"), the allocated GPU device 0 of the job;
# the index and evaluate stages run on the CPU.
# THREADS: JULIA_NUM_THREADS = the task's CPUs: the thread count keys every run ([execution] threads).
# RESOURCES: the CPUs, the memory per CPU (8480M, the node ratio) and the GPU are the runner's — configs/reproduce.toml [[stage]]
# cpus_per_task, mem_per_cpu and gpu, the one source; a launcher submitted by hand states them on its sbatch line.
# =============================================================================

export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
CONFIG=experiments/configs/experiments/chrome_index_grid.toml
SELF=experiments/scripts/slurm/chrome_index_grid.sbatch
PHASE=${GRID_PHASE:-detect}
echo "[$(date +%T)] chrome_index_grid $PHASE task=${SLURM_ARRAY_TASK_ID:-} threads=$JULIA_NUM_THREADS gpu=${CUDA_VISIBLE_DEVICES:-} data=$VGTK_DATA_ROOT results=$VGTK_RESULTS_ROOT commit=$(git rev-parse HEAD)"
grid() { "$JULIA" --startup-file=no --project=experiments experiments/scripts/chrome_index_grid.jl --config="$CONFIG" "$@"; }
case $PHASE in
  detect)
    grid --stage=detect
    n=$(python3 -I -c '
import sys, tomllib
with open(sys.argv[1], "rb") as f: c = tomllib.load(f)
print(len(c["grid"]["sets"]))' "$CONFIG")
    # WAVE: Eval's runner `array.from` = a config key path ([grid] sets); then the task and table phases are graph stages and this
    # self-submission is deleted.
    # the CPU phases at this job's own sizes (the runner's: SLURM_CPUS_PER_TASK, SLURM_MEM_PER_CPU), with no GPU. The runner cannot
    # array a stage over a config list other than [cases] (Eval's stage_items), so this launcher still submits them itself.
    R=(--cpus-per-task="${SLURM_CPUS_PER_TASK:?}" --mem-per-cpu="${SLURM_MEM_PER_CPU:?}M" --gres=none)
    A=$(sbatch --parsable --array=0-$((n - 1)) "${R[@]}" --export=ALL,GRID_PHASE=task "$SELF")
    T=$(sbatch --parsable "${R[@]}" --dependency=afterok:${A%%;*} --export=ALL,GRID_PHASE=table "$SELF")
    echo "[$(date +%T)] submitted the $n set tasks ($A) and the table ($T)" ;;
  task)  grid_set=$(python3 -I -c '
import sys, tomllib
with open(sys.argv[1], "rb") as f: print(tomllib.load(f)["grid"]["sets"][int(sys.argv[2])])' "$CONFIG" "${SLURM_ARRAY_TASK_ID:?the task phase is an array}")
         grid --stage=index --set="$grid_set" ;;
  table) grid --stage=evaluate ;;
  *) echo "GRID_PHASE=$PHASE: detect | task | table"; exit 2 ;;
esac
