# =============================================================================
# The experiments' tests, as ReTestItems test items. Run from the repository root:
#   julia --project=experiments experiments/test/runtests.jl                  (or `make check` in ../paper/)
#   julia --project=experiments experiments/test/runtests.jl 'drivers|data'   # the items whose name matches
#   RETESTITEMS_NWORKERS=0 → serial, in-process                         # for debugging
#
# Every `*_tests.jl` file here holds `@testitem`s and `@testsetup`s only; the first is paper_project_tests.jl, the paper-project
# conventions (VisualGeometryToolkitEval PaperProjects): layout, data policy, the REPRODUCE.md index, configs of record, provenance
# and the data catalog. Each item is evaluated in its own module on a pool of worker processes: items share no bindings and run in
# no fixed order, so an item includes and imports everything it uses (the project's src/ files after src/experiments.jl). The shared
# fixtures are the `@testsetup` modules beside the items that take them, each evaluated once per worker.
#
# A paper project is not a package (experiments has no module to load), so every item sets `default_imports = false`, and run as
# this file ReTestItems takes the active project (--project=experiments) as the test environment instead of building one.
#
# Tags:
#   :data   reads the inputs under data_root(); SKIPPED — a recorded skip, not an error — where no data root is bound
#           (VGTK_DATA_ROOT, or the project's data/exp_raw link)
#   :gpu    the device items (texture_recovery's device page operators), on ONE worker with CUDA_VISIBLE_DEVICES=0, after the pool
#           has exited — two processes on one card is a false-failure machine — and only when the package's GPU decision holds
#           (test/gpu_decision.jl: a functional CUDA device with 20 GiB free, RC_TEST_GPU ≠ 0); a CPU-only run never starts one.
# Every run: testitem_timeout = 3600 s (RadiometricCamera's setting).
# =============================================================================
using ReTestItems

const NAME = isempty(ARGS) ? nothing : Regex(join(ARGS, "|"))
const NWORKERS = parse(Int, get(ENV, "RETESTITEMS_NWORKERS", string(max(1, Sys.CPU_THREADS ÷ 2))))
const ONE = NWORKERS == 0 ? 0 : 1          # the GPU run: one worker (serial mode: in-process)

# One run: the package's test/run_tier.jl (shared with test/runtests.jl); only a selection that matches no item in any run is an error.
include(joinpath(@__DIR__, "..", "..", "RadiometricCamera", "test", "run_tier.jl"))
const RAN = Ref(false)

run_tier(ti -> !(:gpu in ti.tags), @__DIR__, NAME, RAN; nworkers = NWORKERS, nworker_threads = 2)

# the GPU run only on a functional GPU with room (the package's one decision, test/gpu_decision.jl: RC_TEST_GPU=0, no functional
# CUDA device or < 20 GiB free skip it, said by an @info); a CPU-only run never starts a `:gpu` item
include(joinpath(@__DIR__, "..", "..", "RadiometricCamera", "test", "gpu_decision.jl"))
if rc_gpu_tests()
    withenv("CUDA_VISIBLE_DEVICES" => "0") do
        run_tier(ti -> :gpu in ti.tags, @__DIR__, NAME, RAN; nworkers = ONE, nworker_threads = 2)
    end
end

RAN[] || error("no test item matches $(repr(NAME))")
