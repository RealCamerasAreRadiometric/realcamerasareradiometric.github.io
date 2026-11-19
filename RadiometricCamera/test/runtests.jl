# =============================================================================
# RadiometricCamera test suite — ReTestItems
# =============================================================================
#
# Every `*_tests.jl` file under test/ is discovered automatically (test/unit/, test/regression/). Each holds `@testitem`s, and each
# item is evaluated in its own module on a pool of worker processes: items share no bindings and run in no fixed order, so an item
# imports and defines everything it uses. Shared fixtures are `@testsetup` modules: RCFixtures and RCGPU in
# test/unit/rc_testsetups.jl, and the scene of a file whose device part is an item of its own in that file. Files without the suffix
# (unit/fixture_board.jl, regression/golden_scenes.jl, gpu_decision.jl, pipeline/,
# integration_blobboards/) are never discovered — an item that needs one includes it.
#
#   Pkg.test()                                   # every item: the CPU pool, then the GPU items, then the goldens
#   Pkg.test(; test_args = ["focus|grid"])       # items whose name matches, in each of the three runs
#   RETESTITEMS_NWORKERS=0 → serial, in-process  # for debugging (every run, with this process's threads)
#   RC_TEST_GPU=0 → no GPU run                  # also skipped without a functional CUDA device or with < 20 GiB free
#
# Three runs, by tag:
#   (untagged)  the CPU unit items, on the pool: RETESTITEMS_NWORKERS workers (default one per two logical CPUs), 2 threads each
#   :gpu        the device items, on ONE worker with 8 threads, after the pool has exited: two processes on one card is a
#               false-failure machine; only when the GPU decision (test/gpu_decision.jl) holds — taken once, here, and
#               handed to the worker (RC_GPU_DECIDED=1), whose items error if the GPU turns out unusable there
#   :golden     the golden regression, on ONE worker with 8 threads: the closed-form radiometry sums over thread chunks, so the
#               thread count is part of the arithmetic; 8 is the pipeline runner's JULIA_NUM_THREADS (run_pipeline_checks.sh). The
#               goldens (rtol 1e-10) also succeed on 1 thread, as the serial mode runs them
# Every run: testitem_timeout = 3600 s (BlobBoards' setting; the longest item takes ≈ 12 min warm on 1.13).
#
# Workers start from `Base.julia_cmd()`, so Pkg.test's `--check-bounds=yes` and `julia_args` reach them.
# =============================================================================

# RC_TEST_PRECOMPILE_ONLY (the release gate's no-compile-on-a-GPU-slot rule; never set in a test run):
#   "check" — BEFORE loading anything: every [deps] package of this environment (Pkg.test's sandbox: the package and its test
#             target) has a valid cache for this process's flags (Pkg.test's --check-bounds=yes); exit 1 naming the stale ones,
#             else 0. A tier's fail-fast: it compiles nothing.
#   "1"     — load the test deps, the package and its CUDA extension (its trigger, CUDA), so their caches exist for these flags,
#             and exit 0 before any item is selected. The gate's prep runs it once per tree.
# Neither mode reaches item selection, so the guard below (a pattern that matches no item is an error) is never bypassed.
import TOML                                # a dependency of the package: in Pkg.test's sandbox (the "check" mode reads the project)
const PRECOMPILE_ONLY = get(ENV, "RC_TEST_PRECOMPILE_ONLY", "")
if PRECOMPILE_ONLY == "check"
    let t = TOML.parsefile(Base.active_project())
        ids = [Base.PkgId(Base.UUID(u), n) for (n, u) in get(t, "deps", Dict{String, Any}())]
        stale = [id.name for id in ids if Base.locate_package(id) !== nothing && !Base.in_sysimage(id) && !Base.isprecompiled(id)]
        println("RC_TEST_PRECOMPILE_ONLY=check: ", Base.CacheFlags(), ": ", isempty(stale) ? "every cache valid ($(length(ids)) packages)" : "STALE: " * join(stale, ", "))
        exit(isempty(stale) ? 0 : 1)
    end
end
using ReTestItems
using RadiometricCamera
if PRECOMPILE_ONLY == "1"
    import CUDA                            # the extension's trigger: RadiometricCameraCUDAExt is compiled for these flags too
    println("RC_TEST_PRECOMPILE_ONLY=1: the test environment is loaded and cached for ", Base.CacheFlags(), "; no item runs")
    exit(0)
end
isempty(PRECOMPILE_ONLY) || error("RC_TEST_PRECOMPILE_ONLY: \"check\" or \"1\", not $(repr(PRECOMPILE_ONLY))")

const NAME = isempty(ARGS) ? nothing : Regex(join(ARGS, "|"))
const NWORKERS = parse(Int, get(ENV, "RETESTITEMS_NWORKERS", string(max(1, Sys.CPU_THREADS ÷ 2))))
const ONE = NWORKERS == 0 ? 0 : 1          # the GPU and golden runs: one worker (serial mode: in-process)

# One run: run_tier.jl (shared with experiments/test/runtests.jl); only a pattern that matches no item in any run is an error.
include("run_tier.jl")
const RAN = Ref(false)

run_tier(ti -> !(:gpu in ti.tags) && !(:golden in ti.tags), RadiometricCamera, NAME, RAN; nworkers = NWORKERS)

include("gpu_decision.jl")
if rc_gpu_tests()                          # decided once, here; the worker takes the verdict (and fails loudly if the GPU is gone)
    ENV["RC_GPU_DECIDED"] = "1"
    run_tier(ti -> :gpu in ti.tags, RadiometricCamera, NAME, RAN; nworkers = ONE, nworker_threads = 8)
end

run_tier(ti -> :golden in ti.tags, RadiometricCamera, NAME, RAN; nworkers = ONE, nworker_threads = 8)

RAN[] || error("no test item matches $(repr(NAME))")
