#!/bin/bash
# EXAMPLE SLURM launcher: the resources below are those of the machine the runs were made on (one node of 224 CPUs,
# 1900 GB and 8 GPUs in a partition named `gpu`). Adapt --partition, CPUs and memory to your site. Submit from the repository
# root (`bash experiments/scripts/reproduce/reproduce_results_vcai.sh --only=panorama_vignetting`, or
# `sbatch --array=0-<n-1> experiments/scripts/slurm/panorama_vignetting.sbatch`); logs go to ./logs/, which must exist before submission.
# Data paths come from the environment: VGTK_DATA_ROOT, VGTK_RESULTS_ROOT (common.sh).
#SBATCH --partition=gpu
#SBATCH --time=01:00:00
#SBATCH --array=0-2
#SBATCH --job-name=rc-panorama-vignetting
#SBATCH --output=logs/rc_panorama_vignetting_%A_%a.log
#SBATCH --error=logs/rc_panorama_vignetting_%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
mkdir -p logs
source experiments/scripts/slurm/common.sh

# =============================================================================
# panorama_vignetting.sbatch — the sweep's chain of one lens per array task (scripts/panorama_vignetting.jl, its one stage
# `evaluate`), one GPU each. Task i runs the i-th set of SETS (default fx3_14mm,fx3_fisheye,fx3_tamron), whose config of record is
# configs/experiments/panorama_vignetting_<set>.toml.
#
# DEVICE: the config's [example.compute] device must be "cuda" (it keys the run, so this launcher refuses any other value rather
# than override it), and PANORAMA_VIGNETTING_DEVICE=cuda is exported so an inherited "cpu" override of the example
# (PanoramaStitching examples/panorama_vignetting/run.jl) cannot move the run to the host. The runner's gres makes the allocated GPU
# device 0 of the job (CUDA_VISIBLE_DEVICES).
# THREADS: JULIA_NUM_THREADS = the task's CPUs, and the thread count keys the run (src/drivers.jl driver_product): the launch's one
# count (configs/reproduce.toml) — scripts/panorama_stitching.jl finds this run by its key, at its own thread count, so the counts
# must agree.
# RESOURCES: the CPUs, the memory per CPU (8480M, the node ratio) and the GPU are the runner's — configs/reproduce.toml [[stage]]
# cpus_per_task, mem_per_cpu and gpu, the one source; a launcher submitted by hand states them on its sbatch line.
# =============================================================================

IFS=, read -ra SET_LIST <<< "$PANORAMA_SETS"
I=${SLURM_ARRAY_TASK_ID:?an array task: the runner (scripts/reproduce), or sbatch --array=0-<n-1>}
[ "$I" -lt "${#SET_LIST[@]}" ] || { echo "task $I: SETS has ${#SET_LIST[@]} set(s): ${SET_LIST[*]}"; exit 2; }
SET=${SET_LIST[$I]}
CFG=experiments/configs/experiments/panorama_vignetting_$SET.toml
[ -f "$CFG" ] || { echo "no config of record $CFG"; exit 2; }
python3 -I -c '
import sys, tomllib
with open(sys.argv[1], "rb") as f: d = tomllib.load(f)["example"]["compute"]["device"]
d == "cuda" or sys.exit(f"{sys.argv[1]} [example.compute] device = \"{d}\": this launcher runs the GPU configs only")' "$CFG"
export PANORAMA_VIGNETTING_DEVICE=cuda
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

echo "[$(date +%T)] set=$SET (task $I)  gpu=${CUDA_VISIBLE_DEVICES:-}  threads=$JULIA_NUM_THREADS"
echo "[$(date +%T)] data=$VGTK_DATA_ROOT  results=$VGTK_RESULTS_ROOT  commit=$(git rev-parse HEAD)"
exec "$JULIA" --startup-file=no --project=experiments experiments/scripts/panorama_vignetting.jl --config="$CFG" --stage=evaluate
