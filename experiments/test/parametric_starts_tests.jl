# The parametric starts (src/parametric_starts.jl, fitted by the front ends' stage: configs/experiments/frontends.toml [fits.<name>]):
# every parametric start of the Siemens-star refinements is a [runs] reference to a fit of the stage on the same Schöps run the case's
# front end reads; on a synthetic board the Kannala–Brandt and the Double Sphere camera are recovered to the detection noise from the
# nominal start, and the record round-trips through RadiometricCamera's reader.
@testitem "parametric starts: the refinement cases start from the front ends' fits" default_imports = false begin
    using Test
    using DrWatson
    using TOML: TOML
    include(srcdir("experiments.jl"))
    include(srcdir("boards.jl"))
    include(srcdir("parametric_starts.jl"))

    fe = TOML.parsefile(projectdir("configs", "experiments", "frontends.toml"))
    rc = TOML.parsefile(projectdir("configs", "experiments", "radiometric_refinement.toml"))
    lensrun(v) = (m = match(r"^lens:run:([A-Za-z0-9_]+)$", v); m === nothing ? nothing : m.captures[1])
    cases = [(n, c) for (n, c) in rc["cases"] if c["kind"] == "lens" && lensrun(c["pipeline"]["io"]["init"]) !== nothing]
    @test Set(n for (n, _) in cases) == Set(["si30_kb_warped", "si30_ds_warped", "si17ta_ds_warped", "si30fe_kb_warped", "si30fe_ds_warped"])   # (si27fe_ds_warped: the excluded set, configs/excluded.toml; si30fe: 2026-11-16/17)
    for (n, c) in cases
        r = rc["runs"][lensrun(c["pipeline"]["io"]["init"])]
        @test r["config"] == "configs/experiments/frontends.toml" && startswith(r["run"], "fits.") && r["path"] == "camera.json"
        fit = fe["fits"][split(r["run"], '.')[2]]
        @test fit["model"] in START_MODELS
        # Known limitation: the fisheye 30 cm fits state f_mm = 8.0 (the nominal) while their cases state 7.8057 (the configs of record;
        # changing either re-keys products of the run of record, so it is marked, not fixed)
        if startswith(n, "si30fe_")
            @test_broken fit["f_mm"] == c["pipeline"]["camera"]["f_mm"]
            @test fit["sensor_mm"] == c["pipeline"]["camera"]["sensor_mm"]
        else
            @test fit["f_mm"] == c["pipeline"]["camera"]["f_mm"] && fit["sensor_mm"] == c["pipeline"]["camera"]["sensor_mm"]
        end
        # the same detections the case's front end reads: its Schöps run is the fit's
        s = rc["runs"][match(r"^run:([A-Za-z0-9_]+)$", c["pipeline"]["frontend"]["schoeps"]).captures[1]]
        @test s["run"] == "schoeps." * fit["detections"]
    end
    @test_throws ArgumentError nominal_camera("opencv14", 100, 80, 14.0, 35.6)
end

@testitem "parametric starts: a synthetic Kannala–Brandt and Double Sphere camera recovered" default_imports = false begin
    using Test
    using DrWatson
    using StaticArrays: SVector, SMatrix
    using LinearAlgebra: normalize
    using StableRNGs: StableRNG
    import VisualGeometryToolkitCore as VgtkCore
    import RadiometricCamera as RC
    include(srcdir("experiments.jl"))
    include(srcdir("boards.jl"))
    include(srcdir("parametric_starts.jl"))

    W, H = 4256, 2848
    K = SMatrix{3, 3, Float64}(1750.0, 0, 0, 0, 1750.0, 0, 2130.0, 1420.0, 1)
    X = [SVector(x, y, 0.0) for x in range(0.03, 0.97; length = 30) for y in range(0.03, 0.97; length = 30)]   # a 1 m board of 900 stars
    rng = StableRNG(3)
    function pose(rng)                                                  # the camera 0.5–1 m in front of the board, tilted ≤ 20°
        ax = normalize(randn(rng, SVector{3, Float64})); R = VgtkCore.exp_so3(0.35 * rand(rng) * ax)
        c = SVector(0.5 + 0.2 * randn(rng), 0.5 + 0.2 * randn(rng), -(0.5 + 0.5 * rand(rng)))
        return VgtkCore.EuclideanMap(R, -R * c)
    end
    poses = [pose(rng) for _ in 1:12]
    function observe(cam, σ, rng)                                       # every star in the frame, the detection noise σ px
        views = Pair{String, Vector{Pair{SVector{3, Float64}, SVector{2, Float64}}}}[]
        for (k, p) in enumerate(poses)
            cs = Pair{SVector{3, Float64}, SVector{2, Float64}}[]
            for Xk in X
                q = VgtkCore.project(cam, p(Xk))
                all(isfinite, q) && 0 < q[1] < W && 0 < q[2] < H && push!(cs, Xk => q + σ * randn(rng, SVector{2, Float64}))
            end
            push!(views, "V$k" => cs)
        end
        return views
    end
    kb = VgtkCore.CameraModel(K; lens = VgtkCore.KannalaBrandt(0.24, 0.18, -0.08, 0.17))
    ds = VgtkCore.CameraModel(K; lens = VgtkCore.Lens(VgtkCore.DoubleSphere(-0.3, 0.4)))
    for (model, truth) in (("kannala_brandt", kb), ("double_sphere", ds))
        views = observe(truth, 0.1, StableRNG(5))
        @test length(views) == 12 && all(v -> length(v[2]) > 400, views)
        r = _fit_start(model, views, W, H, 14.6776, 35.6; scoring = Dict{String, Any}("type" => "msac", "threshold" => 8.0), max_iter = 100, rng = StableRNG(1))
        @test VgtkCore.successful_retcode(r.fit) && r.n_dropped == 0 && length(r.names) == 12 && r.names == ["V$k" for k in 1:12]
        @test 0.12 < r.fit.rms < 0.16                                       # √2 · σ: the rms of the 2-vector residuals at σ = 0.1 px, not more
        θ, θtrue = VgtkCore.parameter_vector(r.fit.u.camera), VgtkCore.parameter_vector(truth)
        @test VgtkCore.parameter_names(r.fit.u.camera) == VgtkCore.parameter_names(truth)
        @test abs(θ[1] - θtrue[1]) < 1.0 && abs(θ[4] - θtrue[4]) < 0.5 && abs(θ[5] - θtrue[5]) < 0.5   # f, cx, cy [px]
        @test all(abs.(θ[6:end] .- θtrue[6:end]) .< 0.005)               # the lens coefficients
        @test all(0.05 .< r.view_rms .< 0.2) && length(r.view_rms) == 12
        @test r.n_correspondences == sum(length(v[2]) for v in views)
        # the record a refinement starts from: RadiometricCamera reads back the same camera over the image
        p = write_start(joinpath(mktempdir(), "start.json"), r.fit.u.camera, W, H; lens_coverage = VgtkCore.Rect2{Float64}(0, 0, W, H), provenance = (; model = "test"))
        l = RC.load_parametric_lens(p)
        @test VgtkCore.parameter_vector(VgtkCore.model(l)) == θ && VgtkCore.extent(l) == VgtkCore.Size2(W, H)
        # the set's measured image region bounds it (src/image_region.jl), stated by the caller: no default
        e = VgtkCore.Ellipse(VgtkCore.Point2(W / 2, H / 2), 0.45W, 0.4W, 0.0)
        @test VgtkCore.lens_coverage(RC.load_parametric_lens(write_start(joinpath(mktempdir(), "s.json"), r.fit.u.camera, W, H; lens_coverage = e, provenance = (; model = "test")))) == e
        @test_throws UndefKeywordError write_start(joinpath(mktempdir(), "s.json"), r.fit.u.camera, W, H; provenance = (; model = "test"))
        @test start_parameters(r.fit.u.camera)["f"] == θ[1]
    end
    # the pose starts alone: every view converges through the nominal camera; a view without a pose is left out
    cam0 = nominal_camera("kannala_brandt", W, H, 14.6776, 35.6)
    views = observe(kb, 0.1, StableRNG(5)); push!(views, "none" => views[1][2][1:2])
    ps, kept = pose_starts(cam0, views; scoring = Dict{String, Any}("type" => "msac", "threshold" => 8.0), rng = StableRNG(1))
    @test kept == 1:12 && length(ps) == 12
end
