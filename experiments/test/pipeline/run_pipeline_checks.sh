#!/usr/bin/env bash
# PIPELINE CHECKS: the refinement driver bin/refine.jl end to end, run by the paper project's scripts/calibrate.jl, on the published
# case data (the tier rc_calibration of experiments/configs/data_artifacts.toml: its entries rc_golden_<date> and rc_targets_<date>; CPU, a few minutes each), against golden outputs recorded
# from this branch (experiments/test/pipeline/golden/<case>/).
# The package tests check the numerical core on small fixed scenes; these check the front-ends, the initialisation, the masks, the closed-form start, the LM drivers and the output files.
#   siemens_coarse_full  Schöps central-generic run on the synthetic Siemens board (2 views) → the 35 × 25 grid with the board warping h, the
#                        material, V and the lighting free (per-tile LM), the printed-edge albedo
# Usage: bash experiments/test/pipeline/run_pipeline_checks.sh [--record] [case …]      exit code 0 = every case matches
#   --record   write the golden outputs from the CURRENT code and golden/RECORDED.toml (Julia, Manifest sha256, …) — only on purpose; say why
#              in the commit. compare_outputs.jl is bit-exact in the recorded environment and a tolerance comparison elsewhere.
# THE GATE OF RECORD is the exact comparison on Julia 1.13 (golden/RECORDED.toml), the stack's one Julia.
# Each case runs through the paper project's driver on its config of record: experiments/scripts/calibrate.jl
# --config=experiments/configs/calibration/<case>.toml --stage=start,refine --force, which runs bin/refine.jl --stage=start, then --start=<it> (the package's environment) on
# the config's [pipeline] table into the refinement's product <results_root()>/calibrate/refine/<key>/ (its start in calibrate/start/<key>/); that directory is compared. The
# driver's trees are bound here: data_root() is VGTK_DATA_ROOT when set, else experiments/data/exp_raw (the author's link or the
# artifact view of scripts/fetch_data.jl --only=rc_golden_<date>,rc_targets_<date>: experiments/data/README.md) — the cases' inputs, which the
# configs' paths are relative to; results_root() (VGTK_RESULTS_ROOT) is experiments/test/pipeline/work/run-<time>/<case>/ (git-ignored).
# WRITE POLICY: no driver writes under data_root(). A listing of the data root (every file, link and directory, with its size and
# modification time) is taken before and after the cases and must not change (a new or touched file there fails the gate).
# JULIA_NUM_THREADS is fixed at 8 and the driver hands its thread count to bin/refine.jl: the closed-form start sums over thread
# chunks, so the thread count is part of the arithmetic (it keys the run and is recorded in its provenance). The comparison runs in the package's environment (--project=RadiometricCamera).
# (Each case has ONE config, experiments/configs/calibration/<case>.toml; the package's options tests read its [pipeline] table. A run's
# own pipeline_config.toml, written by the driver into its keyed directory, is the table bin/refine.jl takes by hand.)
set -u
RC=$(cd "$(dirname "$0")/../../.." && pwd); cd "$RC"
M=experiments/test/pipeline
RECORD=0; CASES=()
for a in "$@"; do case $a in --record) RECORD=1 ;; *) CASES+=("$a") ;; esac; done
[ ${#CASES[@]} -eq 0 ] && CASES=(siemens_coarse_full)
RUN=$M/work/run-$(date +%Y%m%d-%H%M%S); mkdir -p "$RUN"
export JULIA_NUM_THREADS=8
# THE DEVICE: the configs' compute.backend ("gpu") on the job's GPU, the device of record (owner, 2026-11-13); recorded in
# golden/RECORDED.toml, and compare_outputs.jl compares bit for bit only on the recorded device
DEVICE=$(julia --startup-file=no --project=RadiometricCamera -e 'import CUDA; print(CUDA.functional() ? "gpu" : "cpu")' 2>/dev/null || echo cpu)
export PIPELINE_DEVICE=$DEVICE
echo "device: $DEVICE (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset})"
export VGTK_DATA_ROOT=${VGTK_DATA_ROOT:-$RC/experiments/data/exp_raw}
for f in geometric_accuracy/data/target/siemens_1m/board/sdf.f32 geometric_accuracy/detector_bias/data/fisheye_full/schoeps \
         geometric_accuracy/detector_bias/data/fisheye_full/gt/images_linear; do
  [ -e "$VGTK_DATA_ROOT/$f" ] || { echo "the case data are not under the data root $VGTK_DATA_ROOT (missing $f): bind VGTK_DATA_ROOT or" \
                                        "julia --project=experiments experiments/scripts/fetch_data.jl --only=<the rc_golden and rc_targets entries>"; exit 2; }
done
snapshot() { find -L "$VGTK_DATA_ROOT" -printf '%P %y %s %T@\n' | sort; }
BEFORE=$(snapshot)
JL=(julia --project=RadiometricCamera)
PJ=(julia --project=experiments)
status=0
for c in "${CASES[@]}"; do
  O=$RC/$RUN/$c; mkdir -p "$O"
  CONFIG=experiments/configs/calibration/$c.toml          # the case's config of record
  [ -f "$CONFIG" ] || { echo "unknown case $c (no $CONFIG)"; exit 2; }
  echo "[$(date +%H:%M:%S)] === $c"
  if VGTK_RESULTS_ROOT="$O" "${PJ[@]}" experiments/scripts/calibrate.jl --config="$CONFIG" --stage=start,refine --force > "$O/run.log" 2>&1; then
    D=("$O"/calibrate/refine/*)                        # the refinement's product, <results_root()>/calibrate/refine/<key> (fresh: exactly one)
    [ ${#D[@]} -eq 1 ] && [ -d "${D[0]}" ] || { echo "    FAILED: no single keyed run directory under $O"; status=1; continue; }
    D=${D[0]}; echo "    run directory ${D#$RC/}"
    grep -E "SELF-CHECK|LM:|written to|self-check only" "$D/refine.log" | cut -c1-160 | sed 's/^/    /'
    if [ $RECORD = 1 ]; then
      mkdir -p "$M/golden/$c"; for f in "$D"/*.json; do cp "$f" "$M/golden/$c/"; done; echo "    golden recorded ($(ls "$M/golden/$c" | wc -l) files)"
    else
      "${JL[@]}" "$M/compare_outputs.jl" "$M/golden/$c" "$D" || status=1
    fi
  else
    echo "    FAILED: scripts/calibrate.jl ($O/run.log)"; tail -n 5 "$O/run.log" | sed 's/^/    /'; status=1
  fi
done
if [ "$(snapshot)" != "$BEFORE" ]; then                 # the write policy: nothing new under data_root()
  echo "    FAILED: the data root changed during the run ($VGTK_DATA_ROOT):"; diff <(echo "$BEFORE") <(snapshot) | head -20 | sed 's/^/    /'; status=1
else
  echo "    write policy: data_root() unchanged ($VGTK_DATA_ROOT, $(echo "$BEFORE" | wc -l) entries)"
fi
[ $RECORD = 1 ] && { "${JL[@]}" "$M/record_environment.jl" "$M/golden" "$JULIA_NUM_THREADS" "$DEVICE" || status=1; }
echo "pipeline checks: $([ $status = 0 ] && echo OK || echo FAILED) ($RUN)"
exit $status
