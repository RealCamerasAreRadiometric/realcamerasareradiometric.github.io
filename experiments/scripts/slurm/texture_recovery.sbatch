#!/bin/bash
# EXAMPLE SLURM launcher: the resources below are those of the machine the runs were made on (dgx01: one node of 224 CPUs,
# 1900 GB and 8 GPUs in a partition named `gpu`). Adapt --partition, CPUs and memory to your site. Submit from the repository
# root (`bash experiments/scripts/reproduce/reproduce_results_vcai.sh`, or `sbatch --array=0-<n-1> experiments/scripts/slurm/texture_recovery.sbatch`);
# logs go to ./logs/, which must exist before submission.
# Data paths come from the environment: VGTK_DATA_ROOT, VGTK_RESULTS_ROOT (common.sh).
#SBATCH --partition=gpu
#SBATCH --time=04:00:00
#SBATCH --array=0-0
#SBATCH --job-name=rc-texture-recovery
#SBATCH --output=logs/rc_texture_recovery_%A_%a.log
#SBATCH --error=logs/rc_texture_recovery_%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
mkdir -p logs
source experiments/scripts/slurm/common.sh

# =============================================================================
# texture_recovery.sbatch — the texture test of the defocus experiment (scripts/texture_recovery.jl), one array task per texture set:
# its stages prepare (the texture photographs decoded, Features' load_raw), detect (BlobBoards' correspondences of record present), init (the
# poses), refine (the page's texture under both blur models, on the GPU), folds (the split's other rotations, [folds]; GPU), then on the
# host evaluate (the scores), fold_report ([folds]), compare ([compare]: the page runs as IO records and rasters) and report (summary.md),
# so the one command writes the scores and the summary. Task i runs the i-th set of TEXTURE_SETS (default 14mm_small), whose config of
# record is configs/experiments/texture_recovery_<set>.toml. The figures (CPU) are scripts/regenerate_paper.jl's.
#
# The refinement the camera is held fixed by ([upstream] of the config: a case of radiometric_refinement.toml) must exist in the results
# tree: radiometric_refinement.sbatch (CHROME_CASES=<case>) before this array.
# GPU: the runner's gres makes the allocated GPU device 0 of the job. TIME: the decode of ≈ 100 photographs (minutes), the poses (minutes),
# the two page solves (≈ 3 min each on an A100 at 500 iterations × 12 weights) per fold of [folds] (5); 4 h is a wide margin.
# THREADS: JULIA_NUM_THREADS = the task's CPUs, and the thread count keys every run (src/drivers.jl driver_product): the same count as the
# other launchers, so regenerate_paper finds these runs by their keys.
# RESOURCES: the CPUs, the memory per CPU (8480M, the node ratio) and the GPU are the runner's — configs/reproduce.toml [[stage]]
# cpus_per_task, mem_per_cpu and gpu, the one source; a launcher submitted by hand states them on its sbatch line.
# =============================================================================

IFS=, read -ra SET_LIST <<< "${TEXTURE_SETS:-14mm_small}"
I=${SLURM_ARRAY_TASK_ID:?an array task: the runner (scripts/reproduce), or sbatch --array=0-<n-1>}
[ "$I" -lt "${#SET_LIST[@]}" ] || { echo "task $I: TEXTURE_SETS has ${#SET_LIST[@]} set(s): ${SET_LIST[*]}"; exit 2; }
SET=${SET_LIST[$I]}
CFG=experiments/configs/experiments/texture_recovery_$SET.toml
[ -f "$CFG" ] || { echo "no config of record $CFG"; exit 2; }
export JULIA_NUM_THREADS=8   # 8 whatever its CPUs (28 for the memory, reproduce.toml): the thread count keys the refinement it reads (run of record 2026-11-16: 28 threads looked for another key)

echo "[$(date +%T)] set=$SET (task $I)  gpu=${CUDA_VISIBLE_DEVICES:-}  threads=$JULIA_NUM_THREADS"
echo "[$(date +%T)] data=$VGTK_DATA_ROOT  results=$VGTK_RESULTS_ROOT  commit=$(git rev-parse HEAD)"
# TEXTURE_STAGES: the stages of this job (configs/reproduce.toml phases: the CPU decode and poses, the GPU page solves, the CPU scores),
# by default every stage through report in one job
exec "$JULIA" --startup-file=no --project=experiments experiments/scripts/texture_recovery.jl --config="$CFG" --stage="${TEXTURE_STAGES:-prepare,detect,sheet,init,refine,folds,evaluate,fold_report,compare,report}"
