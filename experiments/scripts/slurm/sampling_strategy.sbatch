#!/bin/bash
# EXAMPLE SLURM launcher of the sampling-strategy experiment (scripts/sampling_strategy.jl; configs/studies/sampling_strategy.toml):
# the resources are the runner's (configs/reproduce.toml [[stage]] sampling_strategy_*). Submit from the repository root; logs go to
# ./logs/. Data paths come from the environment: VGTK_DATA_ROOT, VGTK_RESULTS_ROOT (common.sh).
#SBATCH --partition=gpu
#SBATCH --output=logs/%x_%A_%a.log
#SBATCH --error=logs/%x_%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
mkdir -p logs
source experiments/scripts/slurm/common.sh

# =============================================================================
# sampling_strategy.sbatch — one task of the experiment:
#   STAGES=configs,prepare            once, before the arms (one process: the arms never produce a shared upstream twice)
#   STAGES=refine, an array: task i runs pack i of ARM_PACKS (comma-separated packs, each "case:variant …"; default the four packs of
#                  configs/reproduce.toml), its arms PACKED on the task's one GPU, one process each (2–3 per GPU); ARMS="…" runs one pack by hand
#   STAGES=evaluate,texture,compare,record,figure   the scoring, once the arms are refined
# THREADS: JULIA_NUM_THREADS = 8 in every task, whatever its CPUs: the thread count keys every refinement (src/drivers.jl
# execution_threads), so every stage that reads them by key must run at the one count.
# Never --force: a complete keyed run under the results root is reused.
# =============================================================================
: "${STAGES:?set STAGES (configs,prepare | refine | evaluate,texture,compare,record,figure)}"
export JULIA_NUM_THREADS=8
J=("$JULIA" --startup-file=no --project=experiments experiments/scripts/sampling_strategy.jl --config="experiments/configs/experiments/${CONFIG:-sampling_strategy}.toml")
if [ -z "${ARMS:-}" ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  IFS=, read -ra PACKS <<< "${ARM_PACKS:-fisheye_f4:uniform fisheye_f4:old037 fisheye_f4:ecdf_a1,fisheye_f4:ecdf_a2 fisheye_f4:addon,texcal_14mm:uniform texcal_14mm:old037 texcal_14mm:ecdf_a1,texcal_14mm:ecdf_a2 texcal_14mm:addon}"
  [ "$SLURM_ARRAY_TASK_ID" -lt "${#PACKS[@]}" ] || { echo "task $SLURM_ARRAY_TASK_ID: ARM_PACKS has ${#PACKS[@]} packs"; exit 2; }
  ARMS=${PACKS[$SLURM_ARRAY_TASK_ID]}
fi
echo "[$(date +%T)] stages=$STAGES arms=${ARMS:-}  gpu=${CUDA_VISIBLE_DEVICES:-}  data=$VGTK_DATA_ROOT  results=$VGTK_RESULTS_ROOT  commit=$(git rev-parse HEAD)"
if [ -z "${ARMS:-}" ]; then
  exec "${J[@]}" --stage="$STAGES"
fi
pids=(); rc=0
for a in $ARMS; do IFS=: read -r c v <<< "$a"
  "${J[@]}" --case="$c" --variant="$v" --stage="$STAGES" > "logs/sampling_strategy_${c}_${v}_${SLURM_JOB_ID}.log" 2>&1 & pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || rc=1; done
exit $rc
