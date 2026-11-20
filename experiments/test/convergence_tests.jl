# The convergence experiment (scripts/convergence.jl, configs/experiments/convergence.toml, src/convergence.jl): the config of record
# loads and pairs each case with its chrome set; its record round-trips through VisualGeometryToolkitIO and its figure is the
# dual-axis pgfplots fragment of src/figure_style.jl; on a tiny synthetic refinement (RadiometricCamera's test scene, 2 LM iterations,
# a checkpoint each) the bearing series has one entry per iteration and the start, and its iteration 0 is the start's bearing error
# (the start record's be0, in degrees). No real run: the refinements need the GPU and the published data.
@testitem "convergence: the config, the record through IO, the figure" default_imports = false begin
    using Test
    using DrWatson
    using TOML: TOML
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitIO as VgtkIO
    include(srcdir("experiments.jl"))
    include(srcdir("drivers.jl"))
    include(srcdir("radiometric_refinement.jl"))
    include(srcdir("chrome_evaluation.jl"))
    include(srcdir("figure_style.jl"))
    include(srcdir("convergence.jl"))

    path = projectdir("configs", "experiments", "convergence.toml")
    cfg = load_convergence_config(path)
    @test sort(collect(keys(cfg["cases"]))) == ["ag10fe_spline_warped", "ag30_spline_warped", "si30_spline_warped"]   # ag30: the right panel of fig:convergence-rms
    @test cfg["convergence"]["stride"] == 1

    @testset "a case that is not the chrome set's, or a stride that is not a positive integer, is refused" begin
        bad(f) = (c = TOML.parsefile(path); f(c); p = joinpath(mktempdir(), "convergence.toml"); open(io -> TOML.print(io, c), p, "w"); p)
        @test_throws ArgumentError load_convergence_config(bad(c -> c["convergence"]["stride"] = 0))
        @test_throws ArgumentError load_convergence_config(bad(c -> c["cases"]["si30_spline_warped"]["chrome"] = c["cases"]["ag10fe_spline_warped"]["chrome"]))
        @test_throws ArgumentError load_convergence_config(bad(c -> c["cases"]["no_such_case"] = Dict("chrome" => "x", "label" => "y")))
        @test_throws ArgumentError load_convergence_config(bad(c -> c["figure"]["colour"] = "red"))
    end

    @testset "the products: keyed by the refinement, the chrome index and the protocol, the held-out variant on" begin
        p = convergence_product(cfg, "ag10fe_spline_warped")
        @test p.name == "convergence" && p.output == CONVERGENCE_RECORD_FILE
        rc = convergence_refinements(cfg); cc = convergence_chrome(cfg, "ag10fe_spline_warped")
        up = p.record["upstream"]
        @test up["refine"] == refine_product(rc, "ag10fe_spline_warped").key && up["index"] == index_product(cc).key
        @test p.record["chrome"]["holdout"] == "on" && !any(k -> haskey(p.record["chrome"], k), ("models", "bootstrap", "comparisons"))   # keyed by what it reads
        @test convergence_figure_product(cfg, "ag10fe_spline_warped").record["upstream"]["convergence"] == p.key
        @test p.key != convergence_product(cfg, "si30_spline_warped").key
    end

    @testset "iterations: 0, the stride's multiples and always the last checkpoint" begin
        dir = mktempdir()
        for i in 1:7
            touch(joinpath(dir, "grid_it$i.json")); i == 7 || touch(joinpath(dir, "state_it$i.json"))
        end
        @test convergence_iterations(dir, 1) == 0:6
        @test convergence_iterations(dir, 4) == [0, 4, 6]
        @test_throws ArgumentError convergence_iterations(mktempdir(), 1)
    end

    @testset "the record round-trips through IO; the figure is the dual-axis fragment" begin
        its = [0, 1, 2]
        bearings = [(; iteration = i, median_deg = 0.02 * (1 + i), p95_deg = 0.05 * (1 + i), n = 1000 - i, rel_rms = 0.4 - 0.01i) for i in its]
        v(m) = (; median = m, rms = 1.2 .* m, p95 = 2.0 .* m, n = [500, 500, 500], photos = [9, 9, 9])
        th = (; motion_heldout = v([1.0, 0.8, 0.85]), motion = v([0.9, 0.7, 0.72]))
        rec = convergence_record(; case = "ag10fe_spline_warped", label = "fisheye", iterations = its, bearings, thorlabs = th,
                                 keys = (; refine = "r0", start = "s0", detect = "d0", index = "i0", chrome_config = "c.toml"))
        @test rec.thorlabs_lowest == (; iteration = 1, median_px = 0.8)
        f = joinpath(mktempdir(), CONVERGENCE_RECORD_FILE)
        VgtkIO.write_record(f, rec)
        r = load_convergence_record(f)
        @test r.iterations == its && r.case == rec.case && r.headline == "motion_heldout"
        @test r.bearing_divergence_deg.median == rec.bearing_divergence_deg.median && r.bearing_divergence_deg.p95 == rec.bearing_divergence_deg.p95
        @test r.bearing_divergence_deg.n == rec.bearing_divergence_deg.n && r.rel_rms == rec.rel_rms
        @test r.thorlabs_px.motion_heldout == th.motion_heldout && r.thorlabs_px.motion == th.motion
        @test r.thorlabs_lowest == rec.thorlabs_lowest && r.runs["refine"] == "r0"
        out = mktempdir()
        w = write_convergence_figure(out, r, cfg["figure"])
        tex = read(w.fragment, String)
        @test occursin("axis y line*=left", tex) && occursin("axis y line*=right, axis x line=none", tex)
        @test count("\\begin{axis}", tex) == 2 && occursin("(2,0.85000)", tex) && occursin("(0,0.02000)", tex)
        @test all(k -> occursin("rc $k", tex), ("divergence median", "divergence p95", "thorlabs heldout", "thorlabs in-sample"))
        @test !occursin("width=", tex) && !occursin("height=", tex)      # the layout is the paper's
        sty = read(w.styles, String)
        @test occursin("\\pgfplotsset{rc divergence median/.style=", sty) && occursin("rc axis/.style", sty) && !occursin("width=", sty)
        @test occursin("(0.800\\,px) at iteration 1.", read(w.caption, String))   # the caption snippet: the headline's lowest
        @test isfile(w.preview) && occursin("convergence_ag10fe_spline_warped", read(w.preview, String))
    end
end

@testitem "convergence: a tiny synthetic refinement's bearing series" default_imports = false begin
    using Test
    using DrWatson
    using StaticArrays: SMatrix, SVector
    using LinearAlgebra: normalize
    using Statistics: median
    using PNGFiles: PNGFiles
    using ColorTypes: ColorTypes, RGB
    const N0f16 = ColorTypes.FixedPointNumbers.N0f16                     # the 16-bit linear photographs (ColorTypes' fixed-point numbers)
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: exp_so3, EnvironmentLight, SphericalHarmonicMap
    include(srcdir("experiments.jl"))
    include(srcdir("drivers.jl"))
    include(srcdir("radiometric_refinement.jl"))
    include(srcdir("chrome_evaluation.jl"))
    include(srcdir("figure_style.jl"))
    include(srcdir("convergence.jl"))
    const RC = RadiometricCamera
    rct = joinpath(pkgdir(RC), "test", "unit")
    include(joinpath(rct, "fixture_board.jl")); include(joinpath(rct, "fixture_camera.jl"))   # RadiometricCamera's test scene (its checkpoint item's)

    tex = fixture_board()
    board = RC.PrintedTarget([-1.0, -1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], tex; facing = [0.0, 0.0, 1.5])
    W = H = 24
    scene = RC.Scene(board, W, H)
    glens = RC.grid_model(cam(VgtkCore.Lens(VgtkCore.DoubleSphere(0.25, 0.4)), 30.912, (12.24, 11.52), W, H), 10, 10)
    poses = [VgtkCore.EuclideanMap(SMatrix{3, 3}(exp_so3(SVector(π, 0.0, 0.0)) * exp_so3(SVector(0.15 * cos(k), 0.12 * sin(k), 0.1 * k))),
                                   SVector{3}(SVector(0.3 * cos(k), 0.3 * sin(k), 1.4 + 0.1 * k)); check = false) for k in 1:2]
    focus = RC.Focus(RC.FocusSpline(VgtkCore.grid_nodes(3, VgtkCore.Rect(0, 0, W, H))), fill(1 / 1.5, 9))
    vig = RC.gauge_center_value(VgtkCore.TensorProductVignetting((4, 4), VgtkCore.Rect(0, 0, W, H); degree = 2))
    camera = RC.RadiometricCameraModel(glens, focus, 0.0386, vig)
    b = RC.𝔟(board); a = zeros(3, 9); a[:, 1] .= (2.0, 1.9, 1.8)
    light = EnvironmentLight(SphericalHarmonicMap(a, 2), b, RC.sphere_radius(b, [p.t for p in poses]))
    mat = RC.Material(; 𝛂_ink = 0.12, 𝛂_noink = 0.35, 𝛈_ink = 1.65, 𝛈_noink = 1.33, μ = 0.4)
    rs = RC.RenderSettings(; n_𝐩 = 2, n_ρ = 2, n_φ = 6, N = 4)
    masks = [RC.compute_mask(camera, p, scene, rs; δ = 0.02, backend = RC.KernelAbstractions.CPU()) for p in poses]
    imgs = [RC.render_view(k, camera, poses[k], mat, light, scene, rs, masks[k]) for k in 1:2]
    # the detections: board points seen through the start camera, each image point moved by a fixed sub-pixel offset (a detector's
    # bias), so the start's bearing error is not zero
    detections(k) = [(p .+ SVector(0.3 * cospi(i / 7), 0.3 * sinpi(i / 5)), X) for (i, X) in enumerate(SVector(x, y, 0.0) for x in -0.5:0.1:0.5 for y in -0.5:0.1:0.5)
                     for p in (VgtkCore.project(glens, poses[k].R' * (X - poses[k].t)),) if all(isfinite, p) && all(0 .< p .< W)]
    mktempdir() do dir
        photos = [joinpath(dir, "V0$k.png") for k in 1:2]
        for k in 1:2
            PNGFiles.save(photos[k], [RGB{N0f16}(clamp.(imgs[k][i, j, :], 0, 1)...) for i in 1:H, j in 1:W])
        end
        out = joinpath(dir, "out"); sdir = joinpath(dir, "start")
        opts = RC.pipeline_options(Dict("io" => Dict("out" => out, "checkpoint_each" => "on", "channel" => "RGB"),
                                        "lm" => Dict("max_iter" => 2), "masks" => Dict("samples_per_view" => 10^9, "seed" => 1),
                                        "rendering" => Dict("N" => 4, "n_rho" => 2, "n_phi" => 6, "n_p" => 2)))
        obs = [detections(k) for k in 1:2]
        @test all(o -> length(o) > 20, obs)
        # the start's initialization region: the whole 24 × 24 image (a stored grid start is defined on its region; every detection inside)
        g = RC.StartGeometry(glens, poses, obs, board; names = basename.(photos), init_region = VgtkCore.Rect2{Float64}(0, 0, W, H))
        so = RC.start_options(opts)
        s = RC.start_calibration(g, photos; options = RC.pipeline_options(merge(RC.config_dict(so), Dict("io" => merge(RC.config_dict(so)["io"], Dict("out" => out))))),
                                 device = RC.KernelAbstractions.CPU())
        RC.save_start(sdir, s, s.start, so)                              # the start record: iteration 0
        fit = RC.refine_calibration!(s; options = opts)
        n = VgtkCore.iterations(fit)
        @test n == 2
        its = convergence_iterations(out, 1)
        series = bearing_series(sdir, out, its; geometry = g)
        @test length(series) == n + 1 && [x.iteration for x in series] == 0:n   # one entry per LM iteration, and the start
        start = RC.load_refinement_start(sdir)
        @test series[1].median_deg ≈ rad2deg(median(start.be0) / start.fpx) rtol = 1e-9   # iteration 0: the start's bearing error
        @test series[1].n == length(start.be0)
        @test series[1].rel_rms == start.costs.linear[2] && all(x -> isfinite(x.rel_rms) && x.rel_rms > 0, series)
        @test all(x -> x.p95_deg >= x.median_deg > 0, series)
        @test iteration_models(sdir, out, its) == [("it0", "grid:" * joinpath(sdir, "grid.json")), ("it1", "grid:" * joinpath(out, "grid_it1.json")),
                                                   ("it2", "grid:" * joinpath(out, "grid_it2.json"))]
    end
end

# The paper's figure (fig:convergence-rms): src/convergence_panels.jl writes the submitted panels and frame (T. Harder's rms_figure.py)
# from the medians they plot — every line but the generator line, the coordinates to 6 significant digits.
@testitem "convergence: the paper's panels, from their medians, as in the paper" default_imports = false begin
    using Test
    using DrWatson
    include(srcdir("convergence_panels.jl"))
    F = projectdir("test", "fixtures", "convergence_panels")
    rows = [split(l, '\t') for l in eachline(joinpath(F, "paper_conv.tsv")) if !startswith(l, "#")]
    hdr = popfirst!(rows); col(n) = findfirst(==(n), hdr)
    same(a, b) = (la = split(a, '\n'); lb = split(b, '\n'); length(la) == length(lb) && la[2:end] == lb[2:end])
    for (case, title, target) in (("si30_spline_warped", "Sigma 14 mm lens, Siemens stars, Spline", "siemens"),
                                  ("ag30_spline_warped", "Sigma 14 mm lens, AprilGrid, Spline", "aprilgrid"))
        r = sort([x for x in rows if x[1] == case]; by = x -> parse(Int, x[col("it")]))
        it = [parse(Int, x[col("it")]) for x in r]
        @test it == 0:25
        det = [parse(Float64, x[col("det_median_px")]) for x in r]; chr = [parse(Float64, x[col("chrome_median_px")]) for x in r]
        for narrow in (false, true)
            @test same(convergence_panel_tex(case, title, target, it, det, chr; narrow), read(joinpath(F, "convergence-rms", "$case$(narrow ? "-narrow" : "").tex"), String))
        end
    end
    @test same(convergence_figure_tex("si30_spline_warped", "ag30_spline_warped"), read(joinpath(F, "convergence-rms.tex"), String))
    @test panel_yaxis([0.582, 0.900719, 0.366]) == "ymin=0.2, ymax=1, ytick distance=0.2" && g6(0.37419) == "0.37419" && g6(0.5) == "0.5"
end
