# Shared test fixtures of the unit items. ReTestItems discovers `@testsetup` modules in `*_testsetups.jl` files; an item takes one
# with `setup = [Name]` and sees what it exports. A setup is evaluated once per worker, the first time an item there asks for it.
# (A file's own scene, shared by its CPU item and its GPU item, is a setup in that file: GridTilesScene, …)

# The package alias and the 256² BlobBoards fixture board (test/unit/fixture_board.jl), which runtests.jl used to define in Main.
@testsetup module RCFixtures
    using RadiometricCamera
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitIO as VgtkIO
    const RC = RadiometricCamera
    include(joinpath(@__DIR__, "fixture_board.jl"))
    include(joinpath(@__DIR__, "fixture_camera.jl"))       # cam, camera_of, pinhole_θ
    """`has_lens(c, T)`: `c` is a pixel camera over a Core lens of type `T` (the family a test means)."""
    has_lens(c, ::Type{T}) where {T} = c isa VgtkCore.BoundedCamera{<:VgtkCore.CameraModel} && VgtkCore.lens(c) isa T
    """`free_names(block)`: the free parameters of a lens block (Core's restricted `ParameterManifold`), by Core's names."""
    free_names(block) = Tuple(VgtkCore.parameter_names(VgtkCore.model(block))[i] for i in block.idx)
    """`block_camera(block, bound)`: the pixel camera of a lens block (its model), bounded like `bound`."""
    block_camera(block, bound) = VgtkCore.with_model(bound, VgtkCore.model(block))
    using VisualGeometryToolkitCore: EnvironmentLight, SphericalHarmonicMap, coefficients, with_coefficients, max_degree, sh_count   # the light (Core's)
    export RC, VgtkCore, VgtkIO, FIXTURE_DIR, FIXTURE_BASE, fixture_paths, fixture_board, cam, camera_of, pinhole_θ, saddle, updated_vars, unit_increment, has_lens, free_names, block_camera,
           EnvironmentLight, SphericalHarmonicMap, coefficients, with_coefficients, max_degree, sh_count
end

# The GPU verdict on the GPU worker (test/gpu_decision.jl): test/runtests.jl decides once whether the GPU run runs and hands the
# verdict over as RC_GPU_DECIDED=1; this setup, evaluated once per worker before any of its items allocates, checks that the GPU is
# still usable. Every `:gpu` item starts with `require_gpu()`, which errors with the reason when it is not — never a silent skip.
@testsetup module RCGPU
    import CUDA
    include(joinpath(@__DIR__, "..", "gpu_decision.jl"))
    const RC_GPU_PROBLEM = rc_gpu_problem()
    const RC_GPU_TESTS = RC_GPU_PROBLEM === nothing
    require_gpu() = RC_GPU_TESTS || error(RC_GPU_PROBLEM)
    # The batched per-view device evaluation and the per-tile one compute the same per-pixel values, and the parity items assert them
    # bitwise — which holds on Julia 1.13, and on 1.12 without bounds checks. Julia 1.12 under --check-bounds=yes (the mode Pkg.test
    # forces) rounds the two paths' Float32 Jacobians differently: measured 2026-10-28 on the grid-tile scene, 19661 of 439788 entries,
    # at most 3.7e-7 of a column's norm (1.6e-9 absolute); the residuals stay bitwise. There the parity is to Float32 rounding, 1e-5 of
    # each column (or vector) norm.
    const DEVICE_BITWISE = VERSION >= v"1.13" || Base.JLOptions().check_bounds != 1
    """`device_parity(a, b)`: batched and per-tile device results agree — bitwise where the compiler makes them so (`DEVICE_BITWISE`),
    else to Float32 rounding: Base's `isapprox` at rtol 1e-5 per column (or vector)."""
    device_parity(a::AbstractVector, b::AbstractVector; rtol = 1e-5) = DEVICE_BITWISE ? a == b : isapprox(a, b; rtol)
    device_parity(a::AbstractMatrix, b::AbstractMatrix; rtol = 1e-5) =
        DEVICE_BITWISE ? a == b : size(a) == size(b) && all(c -> isapprox(view(a, :, c), view(b, :, c); rtol), axes(b, 2))
    export RC_GPU_TESTS, require_gpu, device_parity, DEVICE_BITWISE
end
