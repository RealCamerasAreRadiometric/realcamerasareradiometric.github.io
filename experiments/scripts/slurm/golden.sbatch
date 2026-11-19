#!/bin/bash
# EXAMPLE SLURM launcher: the resources below are those of the machine the runs were made on (dgx01: one node of 224 CPUs,
# 1900 GB and 8 GPUs in a partition named `gpu`). Adapt --partition, CPUs and memory to your site. Submit from the repository
# root (`sbatch experiments/scripts/slurm/golden.sbatch`: an array, one task per case of the default GOLDEN_CASES;
# `sbatch --array=0-<n-1> …` for n other GOLDEN_CASES), or `bash experiments/scripts/reproduce/reproduce_results_dgx.sh`); logs go to ./logs/,
# which must exist before submission.
# Data paths come from the environment: VGTK_DATA_ROOT, VGTK_RESULTS_ROOT (common.sh).
#SBATCH --partition=gpu
#SBATCH --time=03:00:00
#SBATCH --array=0-0
#SBATCH --job-name=rc-golden
#SBATCH --output=logs/rc_golden_%A_%a.log
#SBATCH --error=logs/rc_golden_%A_%a.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
mkdir -p logs
source experiments/scripts/slurm/common.sh

# =============================================================================
# golden.sbatch — the CPU golden calibration cases (GOLDEN_CASES, default siemens_coarse_full), one array task per case:
# task SLURM_ARRAY_TASK_ID runs case SLURM_ARRAY_TASK_ID of them (the default --array=0-0 is the default case; the reproduction
# runner's array over the stage's items, which key its run, gives its own), each in a Julia of its own, through the paper project's
# driver on its config of record, compared with experiments/test/pipeline/golden/<case>/.
#
# The same chain as the package's golden gate, experiments/test/pipeline/run_pipeline_checks.sh: scripts/calibrate.jl
# --config=configs/calibration/<case>.toml --stage=refine into a results root of its own (here
# $VGTK_RESULTS_ROOT/golden-<array job id>/<case>/, fresh, so no --force), then experiments/test/pipeline/compare_outputs.jl (the package's
# environment) on the run's keyed directory. on the task's GPU (the device of record, RECORDED.toml), JULIA_NUM_THREADS=8 as the goldens were recorded
# (golden/RECORDED.toml julia_num_threads): the closed-form start sums over thread chunks, so the thread count is part of the
# arithmetic.
#
# THE BOUND. compare_outputs.jl compares bit for bit only in the recorded environment (Julia 1.13, the recorded Manifest sha256) and
# within 1e-9 elsewhere, which these two cases are documented to miss in another environment by up to 1.3e-5 relative against the
# recorded goldens. This job keeps compare_outputs.jl's report as it is (every line in
# the log) and holds its summary to that documented bound, GOLDEN_TOL (default 1.3e-5): every file's max relative difference, every
# pose file's max translation (relative) and max rotation (in radians, the relative difference of a unit rotation), and every
# coordinate difference's max absolute, within it; any structural difference (a missing file or field, a count, a NaN) fails.
# A case fails when its driver fails, its run directory is not single, or the report is beyond the bound.
#
# RESOURCES: the CPUs, the memory per CPU (8480M, the node ratio) and the GPU are the runner's — configs/reproduce.toml [[stage]]
# cpus_per_task, mem_per_cpu and gpu, the one source; a launcher submitted by hand states them on its sbatch line.
# =============================================================================

export JULIA_NUM_THREADS=8
# the device of record (the goldens' RECORDED.toml device): the configs' compute.backend "gpu" on this task's GPU
export PIPELINE_DEVICE=$("$JULIA" --startup-file=no --project=RadiometricCamera -e 'import CUDA; print(CUDA.functional() ? "gpu" : "cpu")' 2>/dev/null || echo cpu)
TOL=${GOLDEN_TOL:-1.3e-5}
ROOT=$VGTK_RESULTS_ROOT/golden-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-$(date +%H%M%S)}}
IFS=, read -ra ALL_CASES <<< "$GOLDEN_CASES"
I=${SLURM_ARRAY_TASK_ID:?an array task: sbatch --array=0-<n-1> (the default --array=0-1 is the default GOLDEN_CASES)}
[ "$I" -lt "${#ALL_CASES[@]}" ] || { echo "task $I: GOLDEN_CASES has ${#ALL_CASES[@]} case(s): ${ALL_CASES[*]}"; exit 2; }
CASE_LIST=("${ALL_CASES[$I]}")
echo "[$(date +%T)] cases=${CASE_LIST[*]}  bound=$TOL  data=$VGTK_DATA_ROOT  results=$ROOT  commit=$(git rev-parse HEAD)"

# compare_outputs.jl's report (stdin) against the bound: prints each line beyond it, exits 1 on any
bound() {
  awk -v tol="$TOL" '
    function beyond(v, what) { if (v + 0 > tol + 0) { print "    BEYOND " tol " (" what "): " $0; bad = 1 } }
    /MISSING output|different fields|pose count|: missing$|: count [^ ]+ ≠ golden|NaN mismatch/ { print "    STRUCTURAL: " $0; bad = 1 }
    /compared [0-9]+ quantities/ { seen = 1 }
    / max relative / { v = $0; sub(/.* max relative /, "", v); sub(/ .*/, "", v); beyond(v, "relative") }
    / max translation / { v = $0; sub(/.* max translation /, "", v); sub(/ .*/, "", v); beyond(v, "translation, relative") }
    / max rotation / { v = $0; sub(/.* max rotation /, "", v); sub(/°.*/, "", v); beyond(v * 3.141592653589793 / 180, "rotation, rad") }
    /coordinate differences, max absolute / { v = $0; sub(/.*coordinate differences, max absolute /, "", v); sub(/[^0-9.eE+-].*/, "", v); beyond(v, "coordinate, absolute") }
    END { if (!seen) { print "    no summary line: compare_outputs.jl did not finish"; bad = 1 } exit bad }'
}

status=0
for c in "${CASE_LIST[@]}"; do
  CONFIG=experiments/configs/calibration/$c.toml
  GOLDEN=experiments/test/pipeline/golden/$c
  [ -f "$CONFIG" ] && [ -d "$GOLDEN" ] || { echo "unknown case $c (no $CONFIG or $GOLDEN)"; status=1; continue; }
  O=$ROOT/$c; mkdir -p "$O"
  echo "[$(date +%T)] === $c"
  if ! VGTK_RESULTS_ROOT="$O" "$JULIA" --startup-file=no --project=experiments experiments/scripts/calibrate.jl \
         --config="$CONFIG" --stage=start,refine > "$O/run.log" 2>&1; then
    echo "    FAILED: scripts/calibrate.jl ($O/run.log)"; tail -n 15 "$O/run.log" | sed 's/^/    /'; status=1; continue
  fi
  D=("$O"/calibrate/refine/*)                      # the refinement's product: <results_root()>/calibrate/refine/<key>
  [ ${#D[@]} -eq 1 ] && [ -d "${D[0]}" ] || { echo "    FAILED: no single keyed run directory under $O"; status=1; continue; }
  echo "    run directory ${D[0]}"
  "$JULIA" --startup-file=no --project=RadiometricCamera experiments/test/pipeline/compare_outputs.jl "$GOLDEN" "${D[0]}" > "$O/compare.log" 2>&1 || true
  cat "$O/compare.log"
  if bound < "$O/compare.log"; then
    echo "    $c: within $TOL"
  else
    echo "    $c: BEYOND $TOL"; status=1
  fi
done
echo "[$(date +%T)] golden: $([ $status = 0 ] && echo "every case within $TOL" || echo FAIL) ($ROOT)"
exit $status
