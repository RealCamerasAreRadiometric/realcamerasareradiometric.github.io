# The grid lenses are VisualGeometryToolkitCore's SplineLens cameras (Task 6a; Core 0.9's fold-in of the generic models): the chief
# direction a pixel reads through Core's lattice against the evaluation of the former `GridLens` (its bicubic chart code, kept here as
# the oracle), and a per-tile window against the full camera.
@testitem "grid lens = Core's SplineLens camera (former evaluation, windows)" begin
    using RadiometricCamera: grid_dims
    using StaticArrays, LinearAlgebra, Random
    import VisualGeometryToolkitCore as VgtkCore
    const RC = RadiometricCamera

    # the former GridLens evaluation (src/camera/models/grid_lens.jl before Task 6a): node g (0-based) at cell coordinate t = g − 1,
    # grid laid [g_x, g_y], cell clamped to [0, C − 1] and u to [0, 1], weights ((1−u)³, 3u³−6u²+4, −3u³+3u²+3u+1, u³)/6
    function former_axis(t, C)
        c = clamp(floor(Int, t), 0, C - 1); u = clamp(t - c, zero(t), one(t))
        return c, SVector((1 - u)^3, 3u^3 - 6u^2 + 4, -3u^3 + 3u^2 + 3u + 1, u^3) / 6
    end
    function former_π⁻¹(Gxy, 𝐩, W, H)                                # 𝐩 in COLMAP pixels of the W × H image the lattice spans
        Cx, Cy = size(Gxy, 1) - 3, size(Gxy, 2) - 3
        cx, wx = former_axis(𝐩[1] / W * Cx, Cx); cy, wy = former_axis(𝐩[2] / H * Cy, Cy)
        acc = zero(SVector{3, Float64})
        for b in 1:4, a in 1:4
            acc += (wx[a] * wy[b]) * Gxy[cx + a, cy + b]
        end
        return acc
    end

    rng = Xoshiro(61)
    W, H, GX, GY = 1200, 800, 35, 25
    Gxy = [normalize(SVector(0.05 * (i - GX / 2), 0.05 * (j - GY / 2), 1.0)) * (0.9 + 0.2 * rand(rng)) for i in 1:GX, j in 1:GY]
    K = VgtkCore.LogicalIntrinsics(VgtkCore.CameraCalibrationMat(SMatrix{3, 3, Float64}(900, 0, 0, 0, 900, 0, 610, 395, 1)))
    g = VgtkCore.BoundedCamera(VgtkCore.CameraModel(VgtkCore.SplineLens, permutedims(Gxy), VgtkCore.Size2(W, H); intrinsics = K), VgtkCore.Size2(W, H))
    @test g isa VgtkCore.BoundedCamera{<:VgtkCore.CameraModel{<:VgtkCore.AbstractIntrinsics, <:VgtkCore.SplineLens}}
    @test grid_dims(g) == (GX, GY) && RC._direction_node(g, 3, 7) == Gxy[4, 8]
    ps = [SVector(W * rand(rng), H * rand(rng)) for _ in 1:5000]
    # Core's pixel → lattice coordinate path rounds differently from the former's: equal to a few ulps
    worst = maximum(p -> norm(RC.chief_ray(g, p) - former_π⁻¹(Gxy, p, W, H)) / norm(former_π⁻¹(Gxy, p, W, H)), ps)
    @test worst < 1e-14
    # a tile window (Core's `window`: a camera of the block) evaluates as the full camera wherever its controls cover the support,
    # to the rounding of the lattice coordinate (Core's own statement of a window); its cells are the full lattice's, shifted by its
    # first control
    rows, cols = 6:15, 11:20                                         # y, x
    w = VgtkCore.window(VgtkCore.model(g), rows, cols)
    n = 0
    for p in ps
        cx, _, cy, _ = RC._grid_support(g, p)
        (first(cols) <= cx + 1 && cx + 4 <= last(cols) && first(rows) <= cy + 1 && cy + 4 <= last(rows)) || continue
        n += 1
        wx, _, wy, _ = RC._grid_support(w, p)
        @test (wx + first(cols) - 1, wy + first(rows) - 1) == (cx, cy)
        @test RC.chief_ray(w, p) ≈ RC.chief_ray(g, p) rtol = 1e-14
    end
    @test n > 100
    # Float32 (the device precision) stays Float32
    @test RC.chief_ray(VgtkCore.convert_eltype(Float32, g), SVector{2, Float32}(720.0, 240.0)) isa SVector{3, Float32}
end

# The start perturbation (`grid.start_perturbation`, perturb_start / perturb_directions in per_tile_smoothness.jl): every direction control
# rotated on its sphere by Core's control chart, deterministic from the seed, σ / f radians RMS per tangent axis.
@testitem "grid start perturbation: reproducible, norm-keeping, σ / f per tangent axis" setup = [RCFixtures] begin
    using RadiometricCamera: grid_model
    using StaticArrays, LinearAlgebra, Statistics, Random
    const StableRNG = RC.StableRNG
    W, H = 520, 348
    f = 416.0
    pin = cam(VgtkCore.Lens(VgtkCore.Perspective()), f, (260.0, 174.0), W, H)
    circle = VgtkCore.Ellipse(VgtkCore.Point2{Float64}(260.0, 174.0), 150.0, 150.0, 0.0)
    g = grid_model(pin, 35, 25; coverage = circle)
    dirs(l) = VgtkCore.directions(VgtkCore.model(l))
    D = dirs(g)
    p = RC.StartPerturbationOptions(; sigma_px = 0.5, seed = 7)
    g1 = RC.perturb_start(g, p, f)
    # reproducible from the seed; another seed draws another start; absent = the start itself
    @test dirs(RC.perturb_start(g, p, f)) == dirs(g1)
    @test dirs(RC.perturb_start(g, RC.StartPerturbationOptions(; sigma_px = 0.5, seed = 8), f)) != dirs(g1)
    @test RC.perturb_start(g, nothing, f) === g
    D1 = dirs(g1)
    # the lattice, every norm, the lens coverage and the K kept
    @test size(D1) == size(D) && RC.grid_dims(g1) == RC.grid_dims(g) == (35, 25)
    @test all(norm.(D1) .≈ norm.(D))
    @test VgtkCore.lens_coverage(g1) == VgtkCore.lens_coverage(g) == circle
    @test VgtkCore.calibration_matrix(VgtkCore.model(g1)) == VgtkCore.calibration_matrix(VgtkCore.model(g)) == VgtkCore.calibration_matrix(VgtkCore.model(pin))
    # each control moved by the chart by (σ / f) z along its tangents (t₁, t₂), z the StableRNG(seed) draws in lattice order
    # (g_x fastest): the tangent coordinates of the perturbed direction recover (σ / f) z exactly
    s = p.sigma_px / f
    rng = StableRNG(p.seed)
    z = Matrix{SVector{2, Float64}}(undef, size(D))
    for gy in axes(D, 1), gx in axes(D, 2)
        z[gy, gx] = SVector(randn(rng), randn(rng))
    end
    c = map(D, D1) do d, d1
        d̂ = normalize(d); y = normalize(d1); t1, t2 = VgtkCore.direction_tangents(d̂)
        SVector(dot(y, t1), dot(y, t2)) / dot(y, d̂)
    end
    @test all(isapprox.(c, s .* z; atol = 1e-12))
    # its size: σ / f RMS per tangent axis, √2 σ / f RMS in angle, within sampling error (875 nodes: the mean square's relative
    # standard error ≈ 1/√875 ≈ 3.4 %; the bound is 10 % on the RMS)
    rms_axis = sqrt(mean(x -> sum(abs2, x) / 2, c))
    @test isapprox(rms_axis, s; rtol = 0.1)
    θ = [VgtkCore.angle_between(d, d1) for (d, d1) in zip(D, D1)]
    @test isapprox(sqrt(mean(abs2, θ)), sqrt(2) * s; rtol = 0.1)
    @test maximum(θ) < 6s
    # σ = 0: the start's directions
    @test all(dirs(RC.perturb_start(g, RC.StartPerturbationOptions(; sigma_px = 0.0, seed = 7), f)) .≈ D)
    # a parametric lens has no grid to perturb; a negative σ is refused
    @test_throws ArgumentError RC.perturb_start(pin, p, f)
    @test_throws ArgumentError RC.perturb_directions(StableRNG(1), g, -0.5, f)
end
