# The GPU decision of the test suite, in one place (not a test file: ReTestItems never discovers it). The GPU items (tag `:gpu`: the
# device parts of gpu_tests.jl, edge_tests.jl, height_tests.jl, grid_tiles_tests.jl and
# jacobian_richardson_tests.jl) need a functional GPU with room to spare — they allocate per-view device caches and failed with
# "Out of GPU memory" next to running experiments (2026-10-23). RC_TEST_GPU=0 skips them explicitly; on a CPU-only machine they are
# skipped and the suite succeeds on the CPU paths.
#
# test/runtests.jl decides ONCE (`rc_gpu_tests()`), and when it runs the GPU run it hands its verdict to the GPU worker as
# RC_GPU_DECIDED=1. The worker's RCGPU setup takes that verdict and only re-checks that the GPU is still usable
# (`rc_gpu_problem()`): if it is not, every GPU item errors with the reason — the GPU run never quietly records zero tests.
import CUDA

const RC_GPU_MIN_FREE = 20 * 2^30

rc_gpu_free_GiB() = CUDA.functional() ? round(CUDA.free_memory() / 2^30; digits = 1) : 0.0

"""The runner's decision: run the GPU run? `false` (with an `@info` saying why) without a functional GPU, with RC_TEST_GPU=0 or with
less than 20 GiB free."""
function rc_gpu_tests()
    ok = CUDA.functional() && get(ENV, "RC_TEST_GPU", "1") != "0" && CUDA.free_memory() >= RC_GPU_MIN_FREE
    ok || @info "GPU test sets skipped" functional = CUDA.functional() free_GiB = rc_gpu_free_GiB() RC_TEST_GPU = get(ENV, "RC_TEST_GPU", "1")
    return ok
end

"""On the GPU worker: `nothing` when the runner decided to run the GPU items (RC_GPU_DECIDED=1) and the GPU is still functional with
20 GiB free, else the reason the GPU items cannot run."""
function rc_gpu_problem()
    get(ENV, "RC_GPU_DECIDED", "0") == "1" ||
        return "the GPU items run only in test/runtests.jl's GPU run (RC_GPU_DECIDED=1 is not set)"
    CUDA.functional() || return "the runner decided to run the GPU items, but CUDA is not functional on the GPU worker"
    free = CUDA.free_memory()
    free >= RC_GPU_MIN_FREE ||
        return "the runner decided to run the GPU items, but the GPU worker sees only $(round(free / 2^30; digits = 1)) GiB free (< 20 GiB)"
    return nothing
end
