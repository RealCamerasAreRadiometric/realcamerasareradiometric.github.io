#!/bin/bash
# EXAMPLE SLURM launcher: the resources below are those of the machine the runs were made on (dgx01: one node of 224 CPUs,
# 1900 GB and 8 GPUs in a partition named `gpu`). Adapt --partition, CPUs and memory to your site. Submit through the runner
# (`bash experiments/scripts/reproduce/reproduce_results_dgx.sh`: the stages chrome_evaluation and chrome_evaluation_score, their
# phases, the array over the chrome sets and the resources of configs/reproduce.toml); logs go to ./logs/.
# Data paths come from the environment: VGTK_DATA_ROOT, VGTK_RESULTS_ROOT (common.sh).
#SBATCH --partition=gpu
#SBATCH --output=logs/%x_%A_%a.log
#SBATCH --error=logs/%x_%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
mkdir -p logs
source experiments/scripts/slurm/common.sh

# =============================================================================
# chrome_evaluation.sbatch — scripts/chrome_evaluation.jl on ONE chrome set of the paper per array task: task i runs the i-th set of
# CHROME_SETS (the keys of configs/experiments/chrome_evaluation_paper.toml [chrome.<key>], sorted; the runner sets them), on that
# set's config of record ([chrome.<key>] config), its stages STAGES (configs/reproduce.toml: prepare on the CPU, detect on the GPU,
# index on the CPU — the phases of the stage chrome_evaluation — and evaluate on the CPU once the refinements it scores are made,
# the stage chrome_evaluation_score). The same keyed products chrome_evaluation_paper.jl's sets stage runs in the regeneration,
# which then reuses them; a further chrome set is a [chrome.<key>] of the paper config, never an edit here.
# Never --force. THREADS: JULIA_NUM_THREADS = the task's CPUs (the launch's one count: it keys the runs).
# =============================================================================

: "${STAGES:?set STAGES (chrome_evaluation.jl stages, comma-separated: configs/reproduce.toml)}"
PAPER=experiments/configs/experiments/chrome_evaluation_paper.toml
mapfile -t ROWS < <(python3 -I -c '
import sys, tomllib
with open(sys.argv[1], "rb") as f: chrome = tomllib.load(f)["chrome"]
sel = sys.argv[2]; names = sorted(sel.split(",")) if sel else sorted(chrome)
bad = sorted(set(names) - set(chrome))
if bad: sys.exit(f"unknown chrome set(s) {bad}; the paper config has {sorted(chrome)}")
for n in names: print(n, chrome[n]["config"])' "$PAPER" "${CHROME_SETS:-}")
I=${SLURM_ARRAY_TASK_ID:?an array task: the runner (scripts/reproduce), or sbatch --array=0-<n-1>}
[ "$I" -lt "${#ROWS[@]}" ] || { echo "task $I: CHROME_SETS has ${#ROWS[@]} set(s)"; exit 2; }
SET=${ROWS[$I]%% *}; CFG=experiments/${ROWS[$I]#* }
[ -f "$CFG" ] || { echo "no config of record $CFG (chrome set $SET)"; exit 2; }
export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
echo "[$(date +%T)] chrome set=$SET (task $I) config=$CFG stages=$STAGES  gpu=${CUDA_VISIBLE_DEVICES:-}  threads=$JULIA_NUM_THREADS"
echo "[$(date +%T)] data=$VGTK_DATA_ROOT  results=$VGTK_RESULTS_ROOT  commit=$(git rev-parse HEAD)"
exec "${TIMEV[@]}" "$JULIA" --startup-file=no "${HEAPHINT[@]}" --project=experiments experiments/scripts/chrome_evaluation.jl --config="$CFG" --stage="$STAGES"
