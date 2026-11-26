#!/bin/bash
# EXAMPLE SLURM launcher: the resources below are those of the machine the runs were made on (dgx01: one node of 224 CPUs,
# 1900 GB and 8 GPUs in a partition named `gpu`). Adapt --partition, CPUs and memory to your site. Submit through the runner
# (`bash experiments/scripts/reproduce/reproduce_results_vcai.sh --only=warping_ablation`: its phases, the array over the modes, the
# resources of configs/reproduce.toml); logs go to ./logs/, which must exist before submission.
# Data paths come from the environment: VGTK_DATA_ROOT, VGTK_RESULTS_ROOT (common.sh).
#SBATCH --partition=gpu
#SBATCH --output=logs/%x_%A_%a.log
#SBATCH --error=logs/%x_%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
mkdir -p logs
source experiments/scripts/slurm/common.sh

# =============================================================================
# warping_ablation.sbatch — scripts/warping_ablation.jl on its config of record, one phase of the stage warping_ablation
# (configs/reproduce.toml; VGTK_PHASE):
#   upstream  one task (CPU): every chrome set's upstreams (stage upstream), before any refinement: the chrome detect and index are
#             the stage chrome_evaluation's (upstream of this one), reused by their keys
#   refine    array task i (GPU): shard i of WARP_SHARDS (k/n, configs/reproduce.toml items): every n-th refinement of the selected
#             modes (warping_ablation_modes: WARP_MODES, else every [modes.<name>]) from the k-th, one after another on the task's
#             one GPU (stage refine --mode=<modes> --shard=k/n), so n tasks run every refinement once
#   score     one task (CPU): the scorings, the record and the figure (stages evaluate, record, figure)
# Never --force: a complete keyed run is reused (mode "both" IS the refinements of record, by their keys).
# THREADS: JULIA_NUM_THREADS = the task's CPUs (the thread count keys every run).
# =============================================================================

export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
cfg=experiments/configs/studies/warping_ablation.toml
case ${VGTK_PHASE:?a phase of the stage warping_ablation: upstream, refine or score} in
  upstream) args=(--stage=upstream) ;;
  refine)   modes=$(warping_ablation_modes "$cfg" | paste -sd,)
            IFS=, read -ra shards <<< "${WARP_SHARDS:-0/1}"   # the refine phase's shards, k/n (configs/reproduce.toml items; by hand: one shard, every refinement)
            i=${SLURM_ARRAY_TASK_ID:?an array task: the runner (scripts/reproduce), or sbatch --array=0-<n-1>}
            [ "$i" -lt "${#shards[@]}" ] || { echo "task $i: WARP_SHARDS has ${#shards[@]} shard(s)"; exit 2; }
            args=(--stage=refine "--mode=$modes" "--shard=${shards[$i]}") ;;
  score)    args=(--stage=evaluate,record,figure) ;;
  *) echo "unknown phase $VGTK_PHASE"; exit 2 ;;
esac
echo "[$(date +%T)] warping_ablation $VGTK_PHASE ${args[*]}  gpu=${CUDA_VISIBLE_DEVICES:-}  threads=$JULIA_NUM_THREADS  data=$VGTK_DATA_ROOT  results=$VGTK_RESULTS_ROOT  commit=$(git rev-parse HEAD)"
exec "${TIMEV[@]}" "$JULIA" --startup-file=no "${HEAPHINT[@]}" --project=experiments experiments/scripts/warping_ablation.jl --config="$cfg" "${args[@]}"
