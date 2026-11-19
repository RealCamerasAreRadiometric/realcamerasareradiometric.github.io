# common.sh — the environment every launcher of this directory binds, sourced from the repository root
# (`source experiments/scripts/slurm/common.sh`; Slurm copies a batch script into its spool, so a launcher finds this file through
# SLURM_SUBMIT_DIR, the repository root it was submitted from, never through $0).
#
# The data and results trees come from the environment, never from a launcher or a driver (experiments/data/README.md):
#   VGTK_DATA_ROOT     the inputs, read-only (data_root()); required
#   VGTK_RESULTS_ROOT  the results root (results_root()); required. The runner (scripts/reproduce) sets it from the site
#                      (configs/reproduce.toml [sites.<site>] results_root); a launcher submitted by hand names it. Never a
#                      directory named by a date: a run is found by its key (<results>/<experiment>/<stage>-<key>), and a
#                      complete keyed run is reused, so every launch shares the one results root
#   JULIA              the julia binary (default: julia on PATH)
# No launcher gives --force: a run whose keyed directory is complete is reused, never overwritten. The results root may not be the
# project's own experiments/data/exp_pro, the tree that holds the runs of record.

JULIA=${JULIA:-julia}
# a task's peak memory, CPU time and wall time at the end of its log (GNU time -v), where the machine has it
TIMEV=(); [ -x /usr/bin/time ] && TIMEV=(/usr/bin/time -v)
# the GC's heap target: 80 % of the task's memory. Julia otherwise sizes it from the node's memory (1.9 TB on dgx01), so a long loop of
# large temporaries (a set's RAW decodes: ≈ 55 MB per photograph) is not collected and grows into the cgroup's OOM killer
HEAPHINT=(); [ -n "${SLURM_MEM_PER_CPU:-}" ] && HEAPHINT=("--heap-size-hint=$(( SLURM_MEM_PER_CPU * ${SLURM_CPUS_PER_TASK:-1} * 8 / 10 ))M")
command -v "$JULIA" > /dev/null || { echo "no julia binary \"$JULIA\" (set JULIA, or put julia on PATH)"; exit 2; }
: "${VGTK_DATA_ROOT:?set VGTK_DATA_ROOT to the data root (experiments/data/README.md)}"
[ -d "$VGTK_DATA_ROOT" ] || { echo "VGTK_DATA_ROOT=$VGTK_DATA_ROOT is not a directory"; exit 2; }
export VGTK_DATA_ROOT
: "${VGTK_RESULTS_ROOT:?set VGTK_RESULTS_ROOT to the results root (configs/reproduce.toml [sites.<site>] results_root)}"
mkdir -p "$VGTK_RESULTS_ROOT"
if [ -e experiments/data/exp_pro ] && [ "$(readlink -f "$VGTK_RESULTS_ROOT")" = "$(readlink -f experiments/data/exp_pro)" ]; then
  echo "VGTK_RESULTS_ROOT=$VGTK_RESULTS_ROOT is experiments/data/exp_pro, the runs of record: set RESULTS_ROOT to another tree"; exit 2
fi
export VGTK_RESULTS_ROOT

# config_cases <config> <selection> [<kind>]: the cases of a driver's config a launch runs, one line "<case> <max_time s>" each, in
# the driver's order (case_names: sorted by name). <selection> is a comma-separated list of names, or empty for every case of the
# config (of kind <kind> when given). The config is the only list, so adding a case to it never means editing a launcher. A case
# without a [pipeline] LM budget (a front end's set) lists max_time 0. Read by Python's tomllib (python3 -I: no cold Julia start for a TOML).
config_cases() {
  python3 -I -c '
import sys, tomllib
with open(sys.argv[1], "rb") as f: cases = tomllib.load(f)["cases"]
sel = sys.argv[2]; kind = sys.argv[3] if len(sys.argv) > 3 else ""
names = sorted(n for n, t in cases.items() if not kind or t["kind"] == kind) if not sel else sorted(sel.split(","))
bad = sorted(set(names) - set(cases))
if bad: sys.exit(f"unknown case(s) {bad}; the config has {sorted(cases)}")
if not names: sys.exit("no case selected")
for n in names: print(n, round(cases[n].get("pipeline", {}).get("lm", {}).get("max_time", 0)))' "$@"
}

# The per-case experiments, by the name the launchers and configs/reproduce.toml use: the driver, its config of record, and the case
# selection (an environment variable, empty = the default):
#   radiometric_refinement             CHROME_CASES  default: every case
#   radiometric_refinement_vignetting  VIG_CASES     default: every case (scripts/radiometric_refinement.jl on its own config: the f/4 runs)
experiment_driver() { case $1 in radiometric_refinement_vignetting) echo "experiments/scripts/radiometric_refinement.jl" ;; *) echo "experiments/scripts/$1.jl" ;; esac; }
experiment_config() { echo "experiments/configs/experiments/$1.toml"; }
experiment_cases() {
  case $1 in
    radiometric_refinement) config_cases "$(experiment_config "$1")" "${CHROME_CASES:-}" ;;
    radiometric_refinement_vignetting) config_cases "$(experiment_config "$1")" "${VIG_CASES:-}" ;;
    convergence)       config_cases "$(experiment_config "$1")" "${CONVERGENCE_CASES:-}" ;;
    *) echo "unknown per-case experiment $1" >&2; return 2 ;;
  esac
}

# run_case_task <experiment> <stages>: the body of a per-case array task — case SLURM_ARRAY_TASK_ID of the experiment's selection,
# its driver on the config of record, `--case=<case> --stage=<stages>`, JULIA_NUM_THREADS = the task's CPUs (the thread count keys
# every run, the prepare run a case reads included: prepare.sbatch runs at the same count). Never --force. REPEAT=N (a timing
# measurement) hands --repeat=N to the driver: bin/refine.jl runs the refinement N times in one process (cold, then warm).
run_case_task() {
  local exp=$1 stages=$2 rows i
  mapfile -t rows < <(experiment_cases "$exp")
  i=${SLURM_ARRAY_TASK_ID:?an array task: the runner (scripts/reproduce), or sbatch --array=0-<n-1>}
  [ "$i" -lt "${#rows[@]}" ] || { echo "task $i: the $exp selection has ${#rows[@]} case(s): ${rows[*]}"; exit 2; }
  local case=${rows[$i]%% *}
  export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK:-28}
  echo "[$(date +%T)] $exp case=$case (task $i of ${#rows[@]}; max_time ${rows[$i]#* } s)  stages=$stages  gpu=${CUDA_VISIBLE_DEVICES:-}  threads=$JULIA_NUM_THREADS"
  echo "[$(date +%T)] data=$VGTK_DATA_ROOT  results=$VGTK_RESULTS_ROOT  commit=$(git rev-parse HEAD)"
  # the GPU's use, sampled every second for the task's timing (logs/<job>_<array job>_<task>_gpu.csv; Slurm ends it with the task)
  [ -z "${CUDA_VISIBLE_DEVICES:-}" ] || nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader -l 1 \
      > "logs/${SLURM_JOB_NAME:-task}_${SLURM_ARRAY_JOB_ID:-0}_${i}_gpu.csv" 2>/dev/null &
  exec "${TIMEV[@]}" "$JULIA" --startup-file=no "${HEAPHINT[@]}" --project=experiments "$(experiment_driver "$exp")" \
      --config="$(experiment_config "$exp")" --case="$case" --stage="$stages" ${REPEAT:+--repeat=$REPEAT}
}

# The warping ablation's modes a launch runs (configs/studies/warping_ablation.toml [modes.<name>], sorted by name): WARP_MODES
# (comma-separated), else every mode.
warping_ablation_modes() {
  python3 -I -c '
import sys, tomllib
with open(sys.argv[1], "rb") as f: modes = sorted(tomllib.load(f)["modes"])
sel = sys.argv[2]; names = sorted(sel.split(",")) if sel else modes
bad = sorted(set(names) - set(modes))
if bad: sys.exit(f"unknown mode(s) {bad}; the config has {modes}")
print("\n".join(names))' "$1" "${WARP_MODES:-}"
}

# The front ends' cases a launch runs (configs/experiments/frontends.toml [cases.<set>]): FE_SETS (comma-separated), else every case.
frontends_cases() { config_cases experiments/configs/experiments/frontends.toml "${FE_SETS:-}"; }

# The golden calibration cases (configs/calibration/<case>.toml; experiments/test/pipeline/golden/<case>/): GOLDEN_CASES, comma-separated.
GOLDEN_CASES=${GOLDEN_CASES:-siemens_coarse_full}   # blob_kb and the non-central fisheye_nc_selfcheck are deleted (2026-11-14)
