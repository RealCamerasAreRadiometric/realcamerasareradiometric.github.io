# The texture test of the blur model (scripts/texture_recovery.jl, configs/experiments/texture_recovery_<set>.toml, src/texture_recovery.jl):
# the configs of record load and a wrong one is refused; the page solve's pieces (the bilinear texels, Core's operator and CGLS, the penalty, the roles
# and the cells, the scores); a sheet's poses from its blobs (BlobBoards' find_boards on its pose arm, on a synthetic photograph through a
# spline camera); a synthetic sheet rendered through the package's own quadrature and recovered
# end to end (mask, sample, rows, solve, report, figures) on the CPU; the device operators against the host's (a GPU item). The real stages
# need the photographs, the board's design and the refinement's state (REPRODUCE.md); they are not this suite.

@testitem "texture_recovery: the configs of record load, a wrong one is refused" default_imports = false begin
    using Test
    using DrWatson
    using TOML: TOML
    include(srcdir("experiments.jl"))
    include(srcdir("drivers.jl"))
    include(srcdir("radiometric_refinement.jl"))
    include(srcdir("texture_recovery.jl"))
    for set in ("14mm_small",)
        c = load_texture_recovery_config(projectdir("configs", "experiments", "texture_recovery_$set.toml"))
        @test c["upstream"]["case"] == "texcal_14mm_spline_warped" && length(c["prepare"]) == 1 && c["texture"]["split_every"] == 5
        @test all(in(keys(c["figures"])), ["texture_page", "texture_stacked"]) && c["figures"]["texture_stacked"]["asset"] == "texture-learned.png"
    end
    good = TOML.parsefile(projectdir("configs", "experiments", "texture_recovery_14mm_small.toml")); dir = mktempdir()
    function refused(edit!)
        c = deepcopy(good); edit!(c)
        p = joinpath(dir, "texture_recovery_test.toml")
        open(io -> TOML.print(io, c), p, "w")
        return try load_texture_recovery_config(p); nothing catch err; err end
    end
    @test refused(identity) === nothing
    for k in TEXTURE_KEYS
        e = refused(c -> delete!(c["texture"], k)); @test e isa ArgumentError && occursin(k, e.msg)
    end
    e = refused(c -> c["texture"]["tiles"] = 3);                              @test e isa ArgumentError && occursin("tiles", e.msg)
    e = refused(c -> c["texture"]["split_every"] = 2);                        @test e isa ArgumentError && occursin("split_every", e.msg)
    # the optional tables: [folds] (the rotations of the split) and [compare] (page runs by their refine keys)
    @test refused(c -> delete!(c, "folds")) === nothing && refused(c -> delete!(c, "compare")) === nothing
    e = refused(c -> c["folds"] = Dict("rotations" => 9));                     @test e isa ArgumentError && occursin("rotations", e.msg)
    e = refused(c -> c["folds"] = Dict("rotations" => 5, "every" => 5));       @test e isa ArgumentError && occursin("every", e.msg)
    cmp = Dict("runs" => Dict("new" => "c6276681a0d215b8"), "this" => "b", "reference" => "new", "crop_mm" => [28.0, 239.0, 60.0, 33.0])
    @test refused(c -> c["compare"] = cmp) === nothing
    e = refused(c -> c["compare"] = merge(cmp, Dict("reference" => "x")));     @test e isa ArgumentError && occursin("reference", e.msg)
    e = refused(c -> c["compare"] = merge(cmp, Dict("runs" => Dict("new" => "short"))));   @test e isa ArgumentError && occursin("runs", e.msg)
    e = refused(c -> c["compare"] = merge(cmp, Dict("crop_mm" => [1.0, 2.0])));  @test e isa ArgumentError && occursin("crop_mm", e.msg)
    e = refused(c -> c["texture"]["pixel_fraction"] = 1.5);                   @test e isa ArgumentError && occursin("pixel_fraction", e.msg)
    # the CGLS's stops: [cgls.stops] (version 2); a version-1 file's [texture] cgls_max_iter / cgls_rtol read through the converter
    @test cgls_stops(load_texture_recovery_config(projectdir("configs", "experiments", "texture_recovery_14mm_small.toml"))) === VgtkCore.ToleranceStops(; max_iter = 500, rtol = 1.0e-6)
    v1 = refused(c -> (delete!(c, "config_version"); c["texture"]["cgls_max_iter"] = 7; c["texture"]["cgls_rtol"] = 1.0e-5; delete!(c, "cgls")))
    @test v1 === nothing
    e = refused(c -> c["cgls"]["stops"]["rtol"] = 1.0);                      @test e isa ArgumentError && occursin("rtol", e.msg)
    e = refused(c -> c["cgls"]["stops"]["max_iter"] = 0);                    @test e isa ArgumentError && occursin("max_iter", e.msg)
    e = refused(c -> c["texture"]["cgls_rtol"] = 1.0e-6);                    @test e isa ArgumentError
    e = refused(c -> c["texture"]["regularization_weights"] = [0.1, 0.1]);    @test e isa ArgumentError && occursin("regularization_weights", e.msg)
    e = refused(c -> c["texture"]["regularization_weights"] = Float64[]);     @test e isa ArgumentError && occursin("regularization_weights", e.msg)
    e = refused(c -> c["texture"]["centre_rho"] = 0.7);                       @test e isa ArgumentError && occursin("centre_rho", e.msg)
    e = refused(c -> c["texture"]["min_pixels"] = -1);                        @test e isa ArgumentError && occursin("min_pixels", e.msg)
    e = refused(c -> c["texture"]["min_pixels"] = true);                      @test e isa ArgumentError && occursin("min_pixels", e.msg)
    for (k, v) in (("min_blobs", 8), ("outlier_iqr", 6.0), ("refits", 2), ("detections", "blobboard_a4_6eb"))   # retired: unknown keys (Core check_keys lists the allowed)
        e = refused(c -> c["poses"][k] = v); @test e isa ArgumentError && occursin("unknown key", e.msg) && occursin(k, e.msg)
    end
    for k in TEXTURE_POSES_KEYS
        e = refused(c -> delete!(c["poses"], k)); @test e isa ArgumentError && occursin(k, e.msg)
    end
    e = refused(c -> c["poses"]["images"] = "blobboard_a4_6eb");              @test e isa ArgumentError && occursin("images", e.msg)
    e = refused(c -> c["poses"]["find_boards"]["geometry"]["type"] = "radial_homography"); @test e isa ArgumentError && occursin("pose arm", e.msg)
    e = refused(c -> c["upstream"]["case"] = "nope");                         @test e isa ArgumentError && occursin("nope", e.msg)
    e = refused(c -> c["upstream"]["images"] = "fx3_14mm_siemens_20261028");  @test e isa ArgumentError && occursin("images", e.msg)
    e = refused(c -> c["report"]["bootstrap"] = 0);                           @test e isa ArgumentError && occursin("bootstrap", e.msg)
    e = refused(c -> c["figures"]["texture_stacked"]["kind"] = "zoom");         @test e isa ArgumentError && occursin("zoom", e.msg)
    e = refused(c -> c["figures"]["texture_stacked"]["labels"] = ["a"]);        @test e isa ArgumentError && occursin("labels", e.msg)
    e = refused(c -> c["figures"]["texture_stacked"]["variants"] = ["design"]); @test e isa ArgumentError && occursin("variants", e.msg)
    e = refused(c -> c["figures"]["texture_stacked"]["block_mm"] = [1.0, 2.0, 3.0]); @test e isa ArgumentError && occursin("block_mm", e.msg)
    e = refused(c -> c["figures"]["texture_stacked"]["lines_per_band"] = 0);    @test e isa ArgumentError && occursin("lines_per_band", e.msg)
    e = refused(c -> c["figures"]["texture_stacked"]["white"] = 1.5);           @test e isa ArgumentError && occursin("white", e.msg)
    e = refused(c -> delete!(c["figures"]["texture_stacked"], "photograph"));   @test e isa ArgumentError && occursin("photograph", e.msg)
    e = refused(c -> delete!(c["figures"]["texture_page"], "error_range"));   @test e isa ArgumentError && occursin("error_range", e.msg)
    e = refused(c -> c["prepare"]["again"] = deepcopy(first(values(c["prepare"]))));   @test e isa ArgumentError && occursin("one [prepare", e.msg)
    e = refused(c -> c["render"]["significant_digits"] = 0);                  @test e isa ArgumentError && occursin("significant_digits", e.msg)
end

@testitem "texture_recovery: the page solve's pieces (texels, Core's operator and CGLS, the penalty, the roles and cells, the scores)" default_imports = false begin
    using Test
    using DrWatson
    using LinearAlgebra: norm, I, Diagonal
    using Statistics: mean
    using StableRNGs: StableRNG
    include(srcdir("experiments.jl"))
    include(srcdir("drivers.jl"))
    include(srcdir("radiometric_refinement.jl"))
    include(srcdir("texture_recovery.jl"))
    rng = StableRNG(7)
    # the bilinear texel weights: a partition of unity strictly inside the grid, nothing outside
    TW, TH = 24, 30
    w = texel_weights((0.3, 0.4), TW, TH)
    @test w !== nothing && sum(last, w) ≈ 1 && all(1 <= t <= TW * TH for (t, _) in w)
    @test texel_weights((0.0, 0.4), TW, TH) === nothing && texel_weights((0.3, 0.999), TW, TH) === nothing
    @test texel_weights((0.5 / TW, 0.5 / TH), TW, TH) == ((1, 1.0), (2, 0.0), (TW + 1, 0.0), (TW + 2, 0.0))   # texel (1, 1)'s centre
    # Core's page operator: the first differences, horizontal pairs first, then vertical (texel (a, b) at (b − 1) TW + a)
    x = randn(rng, TW * TH)
    D = VgtkCore.difference_rows((TW, TH), 1); X = reshape(x, TW, TH); g = D * x
    # (the penalty reads ‖D x‖², so the rows' sign is Core's: x_i − x_{i+1})
    @test size(D, 1) == (TW - 1) * TH + TW * (TH - 1) && -g[1:(TW - 1) * TH] ≈ vec(X[2:end, :] .- X[1:end - 1, :]) && -g[(TW - 1) * TH + 1:end] ≈ vec(X[:, 2:end] .- X[:, 1:end - 1])
    # Core's CGLS recovers a known texture from bilinear rows (the whole grid seen), and the Tikhonov solve equals the dense normal equations
    xt = rand(rng, TW * TH); R = Vector{Vector{Tuple{Int, Float64}}}()
    for _ in 1:6000
        tw = texel_weights(((0.5 + rand(rng) * (TW - 1.01)) / TW, (0.5 + rand(rng) * (TH - 1.01)) / TH), TW, TH); tw === nothing || push!(R, collect(tw))
    end
    A = sparse(reduce(vcat, [fill(q, length(r)) for (q, r) in enumerate(R)]), reduce(vcat, [first.(r) for r in R]), reduce(vcat, [last.(r) for r in R]), length(R), TW * TH)
    b = A * xt
    op = VgtkCore.PenalizedOperator(A, D; device = VgtkCore.KA.CPU())
    m̄ = VgtkCore.mean_positive_eigenvalue(D); scale = VgtkCore.penalty_scale(sum(abs2, nonzeros(A)), TW * TH, m̄); β = 0.3
    sw = page_solves(op, b, fill(0.5, TW * TH); scale, weights = [β, 0.01, 1.0], stops = VgtkCore.ToleranceStops(; rtol = 1e-12, max_iter = 5000))
    @test VgtkCore.successful_retcode(sw.main) && maximum(abs.(Array(sw.main.u) .- xt)) < 1e-6
    xd = (Matrix(A' * A) + β * scale * Matrix(D' * D)) \ (A' * b)
    @test VgtkCore.successful_retcode(sw.reg[1]) && maximum(abs.(Array(sw.reg[1].u) .- xd)) < 1e-6
    @test norm(D * Array(sw.reg[3].u)) < norm(D * Array(sw.reg[2].u))                    # the heavier weight is the smoother texture
    cap = page_solves(op, b, fill(0.5, TW * TH); scale, weights = Float64[], stops = VgtkCore.ToleranceStops(; rtol = 1e-14, max_iter = 3))
    @test !VgtkCore.successful_retcode(cap.main) && cap.main.stats.nsteps == 3
    # the roles by index, the cells and their coverage
    @test split_roles(10, 5) == [:val, :fit, :fit, :fit, :test, :val, :fit, :fit, :fit, :test]
    @test_throws ArgumentError split_roles(10, 2)
    # the rotation of the split: fold f shifts every role by f photographs; fold 0 the split of record; the folds 0 … 4 put every
    # photograph once in the test and once in the validation set
    @test split_roles(10, 5; fold = 0) == split_roles(10, 5)
    @test split_roles(10, 5; fold = 1) == [:test, :val, :fit, :fit, :fit, :test, :val, :fit, :fit, :fit]
    @test all(k -> count(f -> split_roles(10, 5; fold = f)[k] == :test, 0:4) == 1 && count(f -> split_roles(10, 5; fold = f)[k] == :val, 0:4) == 1, 1:10)
    @test_throws ArgumentError split_roles(10, 5; fold = 5)
    # the rows' weights by their type (RC's weighting): every photograph unweighted → Unweighted; 1/σ vectors → one per row in rows order;
    # a mix, or a non-positive weight, refused
    g = Dict(1 => [1, 2], 2 => [3])
    @test row_weights([RC.Unweighted(), RC.Unweighted()], g, 3) === RC.Unweighted()
    @test row_weights([[2.0, 3.0], [4.0]], g, 3) == [2.0, 3.0, 4.0]
    @test_throws ArgumentError row_weights(Any[RC.Unweighted(), [4.0]], g, 3)
    @test_throws ArgumentError row_weights([[2.0, 0.0], [4.0]], g, 3)
    @test occursin("as measured", weighting_record(RC.Unweighted())) && occursin("1/σ", weighting_record([1.0]))
    @test sensor_record_sha256(RC.SensorRecord(repeat("ab", 32))) == repeat("ab", 32) && sensor_record_sha256(RC.NoSensorRecord()) === nothing
    @test cell_of(0.0, 0.0, 20.0) == (1, 1) && cell_of(20.0, 30.0, 20.0) == (2, 2) && cell_of(19.9, 50.9, 20.0) == (1, 3)
    samples = [[(1.0, 1.0, 5.0, 5.0), (2.0, 1.0, 30.0, 5.0), (3.0, 1.0, 31.0, 6.0)], [(1.0, 1.0, 5.0, 55.0)], NTuple{4, Float64}[]]
    cnt = cell_coverage(samples, 2, 3, 20.0)
    @test cnt[1, 1, 1] == 1 && cnt[2, 1, 1] == 2 && cnt[1, 3, 2] == 1 && sum(cnt) == 4
    roles = split_roles(10, 5); cnt = zeros(Int, 2, 2, 10)
    cnt[1, 1, :] .= 3000; cnt[2, 1, 1:9] .= 3000; cnt[1, 2, [2, 3, 4, 7]] .= 3000
    @test select_cells(cnt, roles; min_fit_views = 4, min_val_views = 1, min_test_views = 1, min_pixels = 2000) == [(1, 1), (2, 1)]   # (1, 2): fit photographs only
    @test select_cells(cnt, roles; min_fit_views = 4, min_val_views = 0, min_test_views = 0, min_pixels = 2000) == [(1, 1), (2, 1), (1, 2)]
    # the scores: the affine map, the block means, the texture against a design, the cells' texels and blocks
    d = rand(rng, 500); af = affine_fit(2.0 .* d .+ 0.1, d)
    @test af.a ≈ 2.0 && isapprox(af.b, 0.1; atol = 1e-12) && af.rms < 1e-10
    B = block_means(reshape(1.0:16.0, 4, 4), 2)
    @test size(B) == (2, 2) && B[1, 1] == mean([1, 2, 5, 6]) && B[2, 2] == mean([11, 12, 15, 16])
    @test nanbox([1.0 NaN; NaN NaN], 2) == fill(1.0, 1, 1) && isnan(nanbox([NaN NaN; NaN NaN], 2)[1])
    des = rand(rng, 40, 60); sel = trues(40, 60); sel[1:10, :] .= false
    ts = texture_scores(1.5 .* des .+ 0.2, sel, des, 4)
    @test ts.a ≈ 1.5
    @test isapprox(ts.b, 0.2; atol = 1e-10)
    @test ts.rms_texel < 1e-10 && ts.rms_block < 1e-10 && ts.n_texels == 30 * 60
    @test all(isnan, ts.blocks[1:3, :]) && all(isfinite, ts.blocks[4:end, :]) && ts.n_blocks == 7 * 15   # block 3 (texels 9–12) straddles the unseen rows
    noisy = texture_scores(des .+ 0.1 .* randn(rng, 40, 60), sel, des, 4)
    @test 0.08 < noisy.rms_texel < 0.12 && noisy.rms_block < noisy.rms_texel                 # the block mean averages the noise
    cm = cells_mask([(1, 1), (2, 2)], 40, 60, 1.0, 20.0)
    @test count(cm) == 2 * 400 && cm[1, 1] && cm[21, 21] && !cm[21, 1] && !cm[1, 21]
    r = cell_blocks(2, 2, 4, 1.0, 20.0, 10, 15)
    @test r == (6:10, 6:10)
end

@testitem "texture_recovery: a sheet's poses from its blobs (read_sheet, board_finder, board_page_map, sheet_poses)" default_imports = false begin
    using Test
    using DrWatson
    using LinearAlgebra: norm
    using StaticArrays: SVector, SMatrix
    using ColorTypes: Gray, N0f8
    import PNGFiles
    include(srcdir("experiments.jl"))
    include(srcdir("drivers.jl"))
    include(srcdir("radiometric_refinement.jl"))
    include(srcdir("texture_recovery.jl"))
    import VisualGeometryToolkitCore as VgtkCore
    import VisualRadiometryToolkitCore as VrtkCore
    import VisualGeometryToolkitEstimation as VgtkEstimation
    import BlobBoards
    Rxyz(a, b, c) = SMatrix{3, 3, Float64}(VgtkCore.Rotations.RotXYZ(a, b, c))
    CT = BlobBoards.CoordinateTransformations                                        # BlobBoards' frame maps (AffineMap)
    # BlobBoards' example board (its artifact) and its print raster (box-averaged ×8, ≈ 150 dpi); a camera with a strong division lens,
    # and its SplineLens BoundedCamera (Estimation's CameraRegression over the sensor): the camera held fixed is the spline
    bjson = joinpath(BlobBoards.example_data(:boards), "blob_board_028_fd896.json")
    board = BlobBoards.load_board(bjson)
    raster = BlobBoards.board_raster(board)
    k = 8; r8 = [sum(Float32(raster[(i - 1) * k + a, (j - 1) * k + b]) for a in 1:k, b in 1:k) / k^2 for i in 1:(size(raster, 1) ÷ k), j in 1:(size(raster, 2) ÷ k)]
    world_to_r8 = CT.AffineMap(SMatrix{2, 2, Float64}(1 / k, 0, 0, 1 / k), SVector(0.0, 0.0)) ∘ CT.AffineMap(board.bundle, :world, :pixel; density = board.density)
    W, H = 1200, 900
    model = VgtkCore.CameraModel(VgtkCore.LogicalIntrinsics(VgtkCore.CameraCalibrationMat(900.0VgtkCore.px, [600.0VgtkCore.px, 450.0VgtkCore.px])), VgtkCore.Division(-0.2, -0.05))
    spline = VgtkCore.BoundedCamera(VgtkCore.solve(VgtkCore.EstimationProblem(VgtkEstimation.CameraRegressionProblem(VgtkCore.SplineLens, VgtkCore.BoundedCamera(model, VgtkCore.Size2(W, H)),
                                                                                  VgtkCore.Rect(0.0, 0.0, Float64(W), Float64(H)))), VgtkEstimation.CameraRegression(; tolerance = 1.0e-2)).u, VgtkCore.Size2(W, H))
    # a photograph of the board (board world → camera, mm; the camera on the printed side) through the spline: every pixel's ray onto
    # the board's plane, the raster sampled bilinearly
    function photograph(R, t)
        img = fill(Gray{N0f8}(1.0), H, W); o = -(R' * t)
        for i in 1:H, j in 1:W
            d = VgtkCore.backproject(spline, SVector(j - 0.5, i - 0.5)); all(isfinite, d) || continue
            dw = R' * SVector{3, Float64}(d); λ = -o[3] / dw[3]; λ > 0 || continue
            X = o + λ * dw; q = world_to_r8(SVector(X[1], X[2])); x, y = q[1] + 0.5, q[2] + 0.5; c0, r0 = floor(Int, x), floor(Int, y)
            (1 <= r0 < size(r8, 1) && 1 <= c0 < size(r8, 2)) || continue
            a, b = x - c0, y - r0
            img[i, j] = Gray{N0f8}(clamp((1 - a) * (1 - b) * r8[r0, c0] + a * (1 - b) * r8[r0, c0 + 1] + (1 - a) * b * r8[r0 + 1, c0] + a * b * r8[r0 + 1, c0 + 1], 0, 1))
        end
        return img
    end
    dir = mktempdir()
    truth = [(Rxyz(π, 0.0, 0.0) * Rxyz(0.25, -0.2, 0.1), SVector(5.0, -3.0, 190.0)), (Rxyz(π, 0.0, 0.0) * Rxyz(-0.15, 0.2, -0.3), SVector(-8.0, 4.0, 210.0))]
    photos = [(name, joinpath(dir, name)) for name in ("P01.png", "P02.png", "P03.png")]
    for (k, (R, t)) in enumerate(truth); PNGFiles.save(photos[k][2], photograph(R, t)); end
    PNGFiles.save(photos[3][2], fill(Gray{N0f8}(1.0), H, W))                       # no board: left out
    # the sheet: the board's canvas on a page with a margin (the page's blob table in mm from its top-left corner, y down)
    margin = SVector(30.0, 40.0); C = BlobBoards.blobs(board.bundle, :canvas; units = BlobBoards.mm)
    sheetdir = joinpath(dir, "sheet"); mkpath(sheetdir)
    PNGFiles.save(joinpath(sheetdir, "texture.png"), fill(Gray{N0f8}(1.0), 230, 210))     # the page's image, 1 mm texels
    blobs = Dict(j => (c = VgtkCore.centroid(d); SVector(VgtkCore.ustrip(BlobBoards.mm, c[1]) + margin[1], VgtkCore.ustrip(BlobBoards.mm, c[2]) + margin[2]))
                 for (j, d) in enumerate(C))
    save_sheet_target(sheetdir; page_mm = (210.0, 230.0), blobs_mm = blobs)                 # the sheet through IO's target record
    sheet = read_sheet(sheetdir)
    @test sheet.page_w ≈ 0.21 && sheet.page_h ≈ 0.23 && length(sheet.blobs) == length(C) && sheet.texel_mm ≈ 1.0
    @test maximum(j -> maximum(abs.(sheet.blobs[j] .- blobs[j])), keys(blobs)) < 1e-9     # the blobs back (mm, decimal-exact lengths)
    # the board's frame on the page: the y flip (π about x) and the canvas centre plus the margin, every blob on its place
    page = board_page_map(board.bundle, sheet)
    @test maximum(page.residuals_mm) < 1e-9
    @test VgtkCore.rotation(page.map) ≈ Rxyz(π, 0.0, 0.0) atol = 1e-12
    # BlobBoards' pose arm, configured by its own [find_boards] table; a non-pose geometry and a bare model are refused
    table = Dict{String, Any}("geometry" => Dict{String, Any}("type" => "pose", "min_inliers" => 6,
                                                            "ransac" => Dict{String, Any}("scoring" => Dict{String, Any}("type" => "student_t_uniform", "outlier_halfwidth" => 5.0),
                                                                                          "local_optimization_strategy" => Dict{String, Any}("type" => "student_t_uniform"))))
    cpu = BlobBoards.KernelAbstractions.CPU()
    finder = board_finder(bjson, spline, table; gallery = joinpath(dir, "gallery.jld2"), device = cpu)
    @test finder.config.geometry isa BlobBoards.PoseGeometry && finder.config.geometry.camera === spline && finder.board.uid == board.uid
    @test_throws ArgumentError board_finder(bjson, spline, Dict{String, Any}("geometry" => Dict{String, Any}("type" => "homography")); gallery = joinpath(dir, "g2.jld2"), device = cpu)
    @test_throws ArgumentError board_finder(bjson, model, table; gallery = joinpath(dir, "g3.jld2"), device = cpu)
    left = String[]
    views, allr, XY, nan = sheet_poses(finder, photos; page = page.map, log = s -> push!(left, s))
    @test length(views) == 2 && length(left) == 1 && occursin("P03", left[1]) && nan == 0
    for (v, (R, t)) in zip(views, truth)
        # the truth on the page, metres: board → camera after page⁻¹, by hand (X_c = R X_w + t, X_w = page⁻¹(1000 X_p))
        P = page.map; Rp = R * SMatrix{3, 3, Float64}(VgtkCore.rotation(P))'; tp = (t - Rp * SVector{3, Float64}(VgtkCore.translation(P))) / 1000
        @test maximum(abs.(v.R - Rp)) < 1e-4 && norm(v.t - tp) < 1e-5                   # 0.1 mrad, 10 µm at 0.2 m
        @test v.blobs >= 100 && v.median_px < 0.1 && v.bearing_median_mrad < 0.2
    end
    @test length(XY) == length(allr) && maximum(allr) < 1.0
    p = write_poses(joinpath(dir, "poses.json"), views, allr, XY, nan; W, H, centre_rho = 0.4, edge_rho = 0.6, label = "test")
    back = read_poses(p)
    @test length(back) == 2 && back[1][1] == "P01.png" && back[1][2] ≈ views[1].R && back[1][3] ≈ views[1].t
    @test VgtkIO.read_record(p; allownan = true).all.n == length(allr)
end

@testitem "texture_recovery: a synthetic sheet through the renderer's quadrature, recovered end to end" default_imports = false begin
    using Test
    using DrWatson
    using LinearAlgebra: norm, normalize
    using Statistics: mean, median
    using StaticArrays: SVector, SMatrix
    using StableRNGs: StableRNG
    using Printf: @sprintf
    import PNGFiles
    using ColorTypes: RGB, N0f16
    include(srcdir("experiments.jl"))
    include(srcdir("drivers.jl"))
    include(srcdir("radiometric_refinement.jl"))
    include(srcdir("texture_recovery.jl"))
    import VisualGeometryToolkitCore as VgtkCore
    import VisualRadiometryToolkitCore as VrtkCore
    rng = StableRNG(3)
    # a camera: a Double Sphere pixel camera (600 px focal length on 240 × 180), a focus surface at 0.45 m, an aperture of 4 mm, a flat
    # vignetting, the pixel-centre quadrature
    W, H = 240, 180
    cam0 = VgtkCore.BoundedCamera(VgtkCore.CameraModel(VgtkCore.LogicalIntrinsics(VgtkCore.CameraCalibrationMat(SMatrix{3, 3}(600.0, 0, 0, 0, 600.0, 0, 120.0, 90.0, 1))),
                                                       VgtkCore.Lens(VgtkCore.DoubleSphere(0.2, 0.3))), VgtkCore.Size2(W, H))
    focus0 = RC.Focus(RC.FocusSpline(VgtkCore.grid_nodes(3, VgtkCore.Rect(0, 0, W, H))), fill(1 / 0.45, 9))
    vig = VrtkCore.TensorProductVignetting((3, 3), VgtkCore.Rect(0, 0, W, H); degree = 2)
    camera = RC.RadiometricCameraModel(cam0, focus0, 0.004, vig)
    sett = RC.RenderSettings(; n_𝐩 = 1, n_ρ = 2, n_φ = 6)
    # the sheet: 60 × 84 mm, 1 mm texels, a design of smooth blobs and sharp bars; a flat target radiometry for the edge parameters
    dir = mktempdir(); TW, TH = 60, 84
    design = [0.35 + 0.3 * sin(0.21 * i) * cos(0.17 * j) + 0.25 * ((i ÷ 6 + j ÷ 9) % 2) for i in 1:TW, j in 1:TH]
    sheetdir = joinpath(dir, "sheet"); mkpath(sheetdir)
    PNGFiles.save(joinpath(sheetdir, "texture.png"), [RGB{N0f16}(design[i, j], design[i, j], design[i, j]) for j in 1:TH, i in 1:TW])
    save_sheet_target(sheetdir; page_mm = (60.0, 84.0), blobs_mm = Dict(1 => SVector(5.0, 5.0)))
    sheet = read_sheet(sheetdir)
    rad = RC.TargetRadiometry(RC.Material(; roughness_ink = 0.12, roughness_paper = 0.35, refractive_index_ink = 1.65, refractive_index_paper = 1.33, diffuse_weight = 0.4), RC.TargetState(; edge = [0.021, 0.034]))
    target = RC.RadiometricTarget(nothing, zero(SVector{RC.HEIGHT_NC, Float64}), 1.0, rad)
    # five photographs (roles val, fit, fit, fit, test) of the sheet from 0.30–0.42 m (the sheet ≈ 100–130 px wide, defocused: the focus
    # is at 0.45 m), the camera on the printed side; world → camera
    Rx(a) = SMatrix{3, 3}(1.0, 0, 0, 0, cos(a), sin(a), 0, -sin(a), cos(a)); Ry(a) = SMatrix{3, 3}(cos(a), 0, -sin(a), 0, 1.0, 0, sin(a), 0, cos(a))
    Cs = ((0.03, 0.042, -0.33), (0.025, 0.045, -0.30), (0.035, 0.04, -0.38), (0.03, 0.04, -0.42), (0.028, 0.044, -0.35)); angs = ((0.06, -0.03), (0.0, 0.05), (-0.04, 0.03), (0.03, 0.02), (0.0, 0.0))
    poses = [(@sprintf("S%02d.png", k), Rx(angs[k][1]) * Ry(angs[k][2]), -(Rx(angs[k][1]) * Ry(angs[k][2]) * SVector(Cs[k]...))) for k in 1:5]
    imgdir = joinpath(dir, "photos"); mkpath(imgdir)
    # blank photographs until the rows render them below: sample_pixels reads each photograph by its file (here decoded PNGs)
    foreach(p -> PNGFiles.save(joinpath(imgdir, p[1]), zeros(RGB{N0f16}, H, W)), poses)
    # the camera held fixed is a complete calibration (decoded renders, no sensor record); the photographs go through RC's entry point
    cc = RC.CalibratedCamera(camera, RC.ObservationModel(RC.NoSensorRecord(), RC.Decoded(false), :RGB), sett)
    ctx = sheet_scene(cc, target, sheet, joinpath(sheetdir, "texture.png"), imgdir, poses, W, H)
    @test ctx.TW == TW && ctx.TH == TH && ctx.tex_mm ≈ 1.0 && size(ctx.design) == (TW, TH) && maximum(abs.(ctx.design .- design)) < 2e-5
    @test ctx.photos isa RC.PhotoSet && ctx.photos.observation == cc.observation && ctx.sett == sett
    @test_throws ArgumentError sheet_scene(cc, target, merge(sheet, (; texel_mm = 0.5)), joinpath(sheetdir, "texture.png"), imgdir, poses, W, H)
    # the entry point refuses what the camera was not calibrated under: a characterisation for a camera of decoded images; a camera
    # calibrated on RAW photosites under a record given these decoded photographs
    @test_throws "calibrated without a sensor characterisation" sheet_scene(cc, target, sheet, joinpath(sheetdir, "texture.png"), imgdir, poses, W, H;
                                                                          characterization = joinpath(sheetdir, "texture.png"))
    raw_cc = RC.CalibratedCamera(camera, RC.ObservationModel(RC.SensorRecord(repeat("ab", 32)), RC.Photosites(), :G), sett)
    @test_throws "calibrated on RAW photographs" sheet_scene(raw_cc, target, sheet, joinpath(sheetdir, "texture.png"), imgdir, poses, W, H;
                                                             characterization = joinpath(sheetdir, "texture.png"))
    # texture recovery reads its photographs through the entry point alone: no second decode path in its source
    src = read(srcdir("texture_recovery.jl"), String)
    @test !occursin("photo_observations(", src) && !occursin("load_photo(", src) && occursin("RC.observe(ctx.photos", src)
    # the pixel mask (the package's compute_mask inside a circle) and the sample
    circle = VgtkCore.Ellipse(VgtkCore.Point2(120.0, 90.0), 140.0, 140.0, 0.0)
    ext = VgtkCore.pixel_mask(circle, W, H)
    @test size(ext) == (H, W) && ext[90, 120] && !ext[1, 1]
    s1 = sample_pixels(ctx, 1, ext; fraction = 1.0, seed = 1, backend = VgtkCore.KA.CPU())
    @test s1.masked > 500 && length(s1.px) == s1.masked && all(0 <= p[3] <= 60 && 0 <= p[4] <= 84 for p in s1.px)
    s2 = sample_pixels(ctx, 1, ext; fraction = 0.3, seed = 1, backend = VgtkCore.KA.CPU())
    @test 0.2 * s1.masked < length(s2.px) < 0.4 * s1.masked
    # the footprint (one row of page_rows, the KernelAbstractions kernel on the CPU backend): the pinhole's is the chief ray's bilinear
    # texels times V (the pixel-centre quadrature); the learned one spreads, sums to V
    u, v = Int.(s1.px[length(s1.px) ÷ 2][1:2]); Vuv = ctx.camera.vignetting(RC.sensor_position(u, v))
    row(camera) = (R = page_rows(ctx, [(1, u, v)], camera, (; all = [true]), Dict(1 => [1]); device = VgtkCore.KA.CPU()); (R.kept[1], R.blocks.all.A[1, :]))
    okp, rp = row(variant_camera(ctx, "pinhole"; pinhole_r = 1e-9))
    o, d = RC.world_ray(ctx.camera, ctx.poses[1], RC.sensor_position(u, v), SVector(0.0, 0.0, 0.0)); sr, 𝐱 = RC.intersect_board(ctx.rays, o, d)
    tw = Dict(texel_weights(RC.chart(ctx.bd, 𝐱), TW, TH))
    @test okp && sum(rp) ≈ Vuv && all(isapprox(Vuv * get(tw, t, 0.0), rp[t]; atol = 1e-9) for t in findall(!iszero, rp))
    okl, rl = row(variant_camera(ctx, "learned"; pinhole_r = 1e-9))
    @test okl && sum(rl) ≈ Vuv && nnz(rl) > nnz(rp)
    @test_throws ArgumentError variant_camera(ctx, "cone"; pinhole_r = 1e-9)
    # the photographs rendered by the rows themselves: I = V · A x — every sample of every photograph
    samples = [sample_pixels(ctx, k, ext; fraction = 1.0, seed = 11, backend = VgtkCore.KA.CPU()).px for k in 1:5]
    rows_px = [(k, Int(p[1]), Int(p[2])) for k in 1:5 for p in samples[k]]
    R = page_rows(ctx, rows_px, ctx.camera, (; all = trues(length(rows_px))), rows_by_photograph(rows_px); device = VgtkCore.KA.CPU())
    A = R.blocks.all.A
    @test all(R.kept) && R.blocks.all.rows == eachindex(rows_px) && size(A) == (length(rows_px), TW * TH)
    I_true = A * vec(design)
    # the operators built where the rows are computed (page_footprints, texture_operator: the rows selected, weighted and transposed on
    # the device) are Core's operators of the host matrix, array for array; a selection keeps its order
    P = page_footprints(ctx, rows_px, ctx.camera, rows_by_photograph(rows_px); device = VgtkCore.KA.CPU())
    Fsel = findall(isodd, eachindex(rows_px)); wsel = [1.0 + 0.1 * (q % 7) for q in Fsel]; Dp = VgtkCore.difference_rows((TW, TH), 1)
    same(a, b) = size(a) == size(b) && a.rowptr == b.rowptr && a.colval == b.colval && a.nzval == b.nzval
    for (w, Ah) in ((wsel, RC._whiten(wsel, A[Fsel, :])), (RC.Unweighted(), A[Fsel, :]))
        opd = texture_operator(P.A, Fsel, w, Dp; device = VgtkCore.KA.CPU()); oph = VgtkCore.PenalizedOperator(Ah, Dp; device = VgtkCore.KA.CPU())
        @test same(opd.A, oph.A) && same(opd.At, oph.At) && same(opd.D, oph.D) && same(opd.Dt, oph.Dt)
    end
    @test all(P.kept) && host_sparse(P.A) == A && host_sparse(csr_rows(P.A, reverse(Fsel); device = VgtkCore.KA.CPU())) == A[reverse(Fsel), :]
    @test host_sparse(csr_transpose(P.A; device = VgtkCore.KA.CPU())) == sparse(transpose(A))
    shuffled = reverse(rows_px)                                         # rows_px in any order: the rows follow it
    @test host_sparse(page_footprints(ctx, shuffled, ctx.camera, rows_by_photograph(shuffled); device = VgtkCore.KA.CPU()).A) == A[end:-1:1, :]
    for k in 1:5
        img = zeros(H, W, 3)
        for (q, (kk, u, v)) in enumerate(rows_px); kk == k && (img[v, u, :] .= I_true[q]); end
        PNGFiles.save(joinpath(imgdir, poses[k][1]), [RGB{N0f16}(img[i, j, 1], img[i, j, 2], img[i, j, 3]) for i in 1:H, j in 1:W])
    end
    # the refine stage on the CPU operators: the learned model recovers the design where seen; the pinhole, blurred photographs read as a
    # pinhole, does worse; then the report, the textures and the figures
    roles = split_roles(5, 5); cells = select_cells(cell_coverage(samples, 3, 5, 20.0), roles; min_fit_views = 1, min_val_views = 1, min_test_views = 1, min_pixels = 20)
    @test !isempty(cells)
    settings = Dict{String, Any}("regularization_weights" => [1e-3, 1e-1], "cell_mm" => 20.0, "centre_rho" => 0.4, "edge_rho" => 0.6,
                                 "pinhole_r_m" => 1e-9, "start" => 0.5)
    stops = VgtkCore.ToleranceStops(; rtol = 1e-8, max_iter = 3000)      # the page's CGLS stops ([cgls.stops])
    out = joinpath(dir, "refine"); logs = String[]
    # each photograph's values at its sampled pixels: read once, by its type (a decoded image: its linear green), the rendered I
    photo_samples = [sample_pixels(ctx, k, ext; fraction = 1.0, seed = 11, backend = VgtkCore.KA.CPU()) for k in 1:5]
    photo_values = [r.values for r in photo_samples]; photo_weights = [r.weights for r in photo_samples]
    @test maximum(abs.(reduce(vcat, photo_values) .- I_true)) <= 1 / 65535   # each value within the 16-bit PNG's quantisation
    @test all(w -> w === RC.Unweighted(), photo_weights)                      # decoded photographs: rows as measured (RC's masked_observations)
    run_page(ctx, samples, roles, cells, circle, out; settings, stops, device = VgtkCore.KA.CPU(), values = photo_values, weights = photo_weights, log = s -> push!(logs, s))
    @test occursin("as measured", String(VgtkIO.read_record(joinpath(out, "page.json"); allownan = true).weighting))
    # whitened rows: a constant 1/σ scales the rows, the values and (through penalty_scale) λ alike, so the texture is the unweighted one;
    # the record states the weighting
    outw = joinpath(dir, "refine_weighted")
    run_page(ctx, samples, roles, cells, circle, outw; settings, stops, device = VgtkCore.KA.CPU(), values = photo_values,
             weights = [fill(2.0, length(v)) for v in photo_values], log = _ -> nothing)
    tex(d) = reshape(Float64.(reinterpret(Float32, read(page_files(d, "learned").reg))), :)
    @test maximum(abs.(tex(outw) .- tex(out))) < 1e-5
    @test occursin("1/σ", String(VgtkIO.read_record(joinpath(outw, "page.json"); allownan = true).weighting))
    @test page_complete(out)
    meta = VgtkIO.read_record(joinpath(out, "page.json"); allownan = true)
    @test Int(meta.TW) == TW && length(meta.cells) == length(cells) && Int(meta.rows_sampled) == length(rows_px) && meta.image_circle.semi_axes[1] == 140.0
    recL = VgtkIO.read_record(page_files(out, "learned").record; allownan = true); recP = VgtkIO.read_record(page_files(out, "pinhole").record; allownan = true)
    relL = sqrt(Float64(recL.solutions[1].test_sse) / Float64(recL.test_sumsq)); relP = sqrt(Float64(recP.solutions[1].test_sse) / Float64(recP.test_sumsq))
    @test relL < 1e-3 && relP > 10 * relL                                # the true blur model predicts the test photograph (to the PNG's quantisation); the pinhole does not
    xL = reshape(Float64.(reinterpret(Float32, read(page_files(out, "learned").texture))), TW, TH)
    seenL = reshape(reinterpret(UInt8, read(page_files(out, "learned").seen)) .!= 0, TW, TH)
    @test count(seenL) > 0.5 * TW * TH && maximum(abs.(xL[seenL] .- design[seenL])) < 2e-3
    rdir = joinpath(dir, "evaluate"); mkpath(rdir)
    rep, textures = report_page(out; settings = Dict{String, Any}("block_mm" => 4.0, "bootstrap" => 200, "seed" => 1), log = s -> push!(logs, s))
    @test rep["chosen"]["learned"]["rms_texel"] < 0.01 && rep["chosen"]["pinhole"]["rms_texel"] > 5 * rep["chosen"]["learned"]["rms_texel"]
    @test rep["chosen"]["learned"]["weight"] in rep["weights"] && length(rep["curves"]["learned"]["val_rel_rms"]) == 3
    @test rep["paired"]["rms_block_mean_diff"] < 0 && rep["paired"]["learned_lower_cells"] == rep["paired"]["cells"] == length(cells)
    lo, hi = rep["paired"]["rms_block_diff_ci95"]; @test lo <= rep["paired"]["rms_block_mean_diff"] <= hi      # Eval's bootstrap_ci over the cells
    @test all(isfinite, textures["learned"][seenL .& cells_mask(cells, TW, TH, 1.0, 20.0)])
    write_textures(rdir, textures, design; tex_mm = 1.0)
    # the comparison of page runs: one scorer, IO records and raster crops (read back bit for bit), the difference to the reference
    cdir = joinpath(dir, "compare"); rset = Dict{String, Any}("block_mm" => 4.0, "bootstrap" => 200, "seed" => 1)
    ch = compare_page_runs(["u" => out, "w" => outw], cdir; settings = rset, crop_mm = [20.0, 30.0, 20.0, 12.0], reference = "u", log = _ -> nothing)
    xs, ys = crop_texels([20.0, 30.0, 20.0, 12.0], 1.0, TW, TH)
    @test xs == 21:40 && ys == 31:42
    cu = VgtkIO.load_raster(joinpath(cdir, "crop_u_learned.json"))
    @test isequal(cu.values, Float32.(permutedims(textures["learned"][xs, ys]))) && cu.frame.chart == :page
    @test cu.frame.area.origin[1] ≈ 0.020 && cu.frame.area.origin[2] ≈ (84 - 42) / 1000 && cu.frame.area.widths[2] ≈ 0.012   # the design chart, y up, metres
    @test isequal(VgtkIO.load_raster(joinpath(cdir, "crop_design.json")).values, Float32.(permutedims(page_design(out, TW, TH)[xs, ys])))   # the page run's design
    dd = filter(isfinite, VgtkIO.load_raster(joinpath(cdir, "diff_w_minus_u_learned.json")).values); @test !isempty(dd) && maximum(abs, dd) < 1e-4
    cr = VgtkIO.read_record(joinpath(cdir, "chosen.json"); allownan = true); cv = VgtkIO.read_record(joinpath(cdir, "curves.json"); allownan = true)
    @test length(cr.runs) == 2 && cr.runs[1].name == "u" && Float64(cr.runs[1].chosen.learned.test_rel_rms) == rep["chosen"]["learned"]["test_rel_rms"]
    @test length(cv.runs[1].weights) == 3 && length(cv.runs[1].curves.pinhole.val_rel_rms) == 3
    @test_throws ArgumentError compare_page_runs(["u" => out, "w" => outw], cdir; settings = rset, crop_mm = [20.0, 30.0, 20.0, 12.0], reference = "x")
    @test_throws ArgumentError crop_texels([50.0, 30.0, 20.0, 12.0], 1.0, TW, TH)
    # the rotation of the split: fold 1's page run beside fold 0's; the report of both
    out1 = joinpath(dir, "refine_fold1")
    run_page(ctx, samples, split_roles(5, 5; fold = 1), cells, circle, out1; settings, stops, device = VgtkCore.KA.CPU(), values = photo_values, weights = photo_weights, log = _ -> nothing)
    fo = fold_report([out, out1], joinpath(dir, "folds"); settings = rset, log = _ -> nothing)
    @test fo["folds"] == 2 && length(fo["paired_at_own_weight"]["values"]) == 2 && fo["models"]["learned"]["chosen_weight_mean_curve"] in fo["weights"]
    @test fo["models"]["learned"]["mean_val_rel_rms"] ≈ [(a + b) / 2 for (a, b) in zip(fo["models"]["learned"]["val_rel_rms_per_fold"]...)]
    @test isfile(joinpath(dir, "folds", "folds.json"))
    # the summary: every number from the records it is given, the definitions from the scorer's settings
    tset = Dict{String, Any}("split_every" => 5, "pixel_fraction" => 1.0, "min_fit_views" => 1, "min_val_views" => 1, "min_test_views" => 1, "min_pixels" => 20,
                             "centre_rho" => 0.4, "edge_rho" => 0.6)
    sp = write_texture_summary(joinpath(dir, "report", "summary.md"); experiment = "texture_recovery", texture = tset, report = rep,
                               chosen = VgtkIO.read_record(joinpath(cdir, "chosen.json"); allownan = true), folds = VgtkIO.read_record(joinpath(dir, "folds", "folds.json"); allownan = true),
                               products = Dict("evaluate" => rdir, "compare" => cdir), run_keys = Dict("evaluate" => "k1", "compare" => "k2"), rc_version = "0.11.1")
    md = read(sp, String)
    @test occursin("√(Σ (y − ŷ)² / Σ y²)", md) && occursin("k % 5 == 0", md) && occursin("RadiometricCamera 0.11.1", md) && occursin("crop_u_learned.json", md)
    @test occursin(@sprintf("%.4f", rep["chosen"]["learned"]["test_rel_rms"]), md) && occursin(@sprintf("%+.4f", rep["paired"]["rms_block_mean_diff"]), md)
    @test occursin("The rotation of the split (2 folds)", md)
    @test occursin("key `k2`", md)
    X, tm = read_texture(rdir, "learned")
    @test size(X) == (TW, TH) && Float64(tm.tex_mm) == 1.0 && isfile(joinpath(rdir, "texture_learned.png")) && isfile(joinpath(rdir, "design.png"))
    fig = figure_texture_closeup(rdir, rep; render = Dict("asset" => "closeup.png", "size_in" => [3.0, 1.2], "dpi" => 100, "crop_mm" => [20.0, 30.0, 20.0],
                                                           "titles" => ["design", "learned", "pinhole"]), dir = joinpath(dir, "plots"))
    @test isfile(fig)
    fig2 = figure_texture_page(rdir, rep; render = Dict("asset" => "page.png", "size_in" => [7.0, 5.0], "dpi" => 60, "downsample" => 2, "error_range" => 0.5, "title" => "t"), dir = joinpath(dir, "plots"))
    @test isfile(fig2)
    @test_throws ArgumentError figure_texture_closeup(rdir, rep; render = Dict("asset" => "c.png", "size_in" => [3.0, 1.2], "dpi" => 100, "crop_mm" => [50.0, 30.0, 20.0],
                                                                                 "titles" => ["design", "learned", "pinhole"]), dir = joinpath(dir, "plots"))
end

@testitem "texture_recovery: the page operators on the device are the host's" default_imports = false tags = [:gpu] skip = !(
        get(ENV, "CUDA_VISIBLE_DEVICES", "0") != "" &&       # a run that hides the GPUs (the CPU tiers) has none, whatever nvidia-smi lists
        Sys.which("nvidia-smi") !== nothing && success(pipeline(`nvidia-smi -L`; stdout = devnull, stderr = devnull))) begin
    using Test
    using DrWatson
    using StableRNGs: StableRNG
    include(srcdir("experiments.jl"))
    include(srcdir("drivers.jl"))
    include(srcdir("radiometric_refinement.jl"))
    include(srcdir("texture_recovery.jl"))
    include(srcdir("texture_gpu.jl"))
    rng = StableRNG(5); TW, TH = 24, 30
    xt = rand(rng, TW * TH); R = Vector{Vector{Tuple{Int, Float64}}}()
    for _ in 1:6000
        tw = texel_weights(((0.5 + rand(rng) * (TW - 1.01)) / TW, (0.5 + rand(rng) * (TH - 1.01)) / TH), TW, TH); tw === nothing || push!(R, collect(tw))
    end
    A = sparse(reduce(vcat, [fill(q, length(r)) for (q, r) in enumerate(R)]), reduce(vcat, [first.(r) for r in R]), reduce(vcat, [last.(r) for r in R]), length(R), TW * TH)
    D = VgtkCore.difference_rows((TW, TH), 1); b = A * xt .+ 0.01 .* randn(rng, size(A, 1))
    scale = VgtkCore.penalty_scale(sum(abs2, nonzeros(A)), TW * TH, VgtkCore.mean_positive_eigenvalue(D))
    cpu = VgtkCore.PenalizedOperator(A, D; device = VgtkCore.KA.CPU()); gpu = VgtkCore.PenalizedOperator(A, D; device = RadiometricCamera.rc_backend("gpu"; inline = false, fallback = false))
    sc = page_solves(cpu, b, fill(0.5, TW * TH); scale, weights = [0.01, 1.0], stops = VgtkCore.ToleranceStops(; rtol = 1e-9, max_iter = 3000))
    sg = page_solves(gpu, b, fill(0.5, TW * TH); scale, weights = [0.01, 1.0], stops = VgtkCore.ToleranceStops(; rtol = 1e-9, max_iter = 3000))
    @test maximum(abs.(Array(sc.main.u) .- Array(sg.main.u))) < 1e-6 && abs(sc.main.stats.nsteps - sg.main.stats.nsteps) <= 2
    for k in 1:2
        @test maximum(abs.(Array(sc.reg[k].u) .- Array(sg.reg[k].u))) < 1e-6
    end
    sg2 = page_solves(gpu, b, fill(0.5, TW * TH); scale, weights = [0.01, 1.0], stops = VgtkCore.ToleranceStops(; rtol = 1e-9, max_iter = 3000))
    @test Array(sg2.main.u) == Array(sg.main.u)                                    # deterministic on the device
    # the operators' construction on the device (the rows selected, weighted, transposed there: csr_rows, csr_transpose, texture_operator)
    # gives the arrays Core uploads from the host matrix
    dev = gpu.device; sel = findall(q -> q % 3 != 0, 1:size(A, 1)); w = [1.0 + 0.01 * (q % 11) for q in sel]
    same(a, b) = size(a) == size(b) && Array(a.rowptr) == Array(b.rowptr) && Array(a.colval) == Array(b.colval) && Array(a.nzval) == Array(b.nzval)
    @test same(csr_transpose(gpu.A; device = dev), gpu.At) && same(csr_rows(gpu.A, sel; device = dev), VgtkCore.CSRMatrix(A[sel, :]; device = dev))
    opd = texture_operator(gpu.A, sel, w, D; device = dev); oph = VgtkCore.PenalizedOperator(RadiometricCamera._whiten(w, A[sel, :]), D; device = dev)
    @test same(opd.A, oph.A) && same(opd.At, oph.At) && same(opd.D, oph.D) && same(opd.Dt, oph.Dt)
    @test host_sparse(csr_rows(gpu.A, reverse(sel); device = dev)) == A[reverse(sel), :]
    @test TextureGPU.gpu_name() isa String
end

@testitem "texture_recovery: the stacked text figure (text_bands, page_photograph, figure_texture_stacked)" default_imports = false begin
    using Test
    using DrWatson
    using StaticArrays: SVector, SMatrix
    import PNGFiles
    include(srcdir("experiments.jl"))
    include(srcdir("drivers.jl"))
    include(srcdir("radiometric_refinement.jl"))
    include(srcdir("texture_recovery.jl"))
    import VisualGeometryToolkitCore as VgtkCore
    import VisualRadiometryToolkitCore as VrtkCore
    # a block of "text": white with 13 lines of ink 5 rows high and 3 blank rows between them (line l: rows 3 + 8(l − 1) … 7 + 8(l − 1))
    lined(TW, TH, n) = (B = ones(TW, TH); for l in 1:n, j in (3 + 8 * (l - 1)):(7 + 8 * (l - 1)), i in 3:(TW - 2 - l); B[i, j] = 0.1; end; B)
    block = lined(40, 110, 13)
    # bands of three lines: a cut in the middle of the gap after lines 3, 6, 9, 12 (rows 24–26 → 25, …); the first band begins half the
    # first gap (3 rows → 1) above line 1
    @test text_bands(block; lines_per_band = 3, bands = 4, white = 0.85) == [(2, 25), (26, 49), (50, 73), (74, 97)]
    @test text_bands(block; lines_per_band = 2, bands = 3, white = 0.85) == [(2, 17), (18, 33), (34, 49)]
    @test_throws ArgumentError text_bands(block; lines_per_band = 7, bands = 2, white = 0.85)      # 14 lines + 1 needed, the block holds 13
    @test_throws ArgumentError text_bands(ones(40, 110); lines_per_band = 1, bands = 1, white = 0.85)   # no ink: no line

    # a photograph on the page's texels: a Double Sphere camera (600 px on 240 × 180) 0.35 m in front of a 60 × 84 mm page of 1 mm
    # texels, slightly turned, the whole page in the image; the image affine in the pixel position, so a texel's value is that function
    # at its centre's projection
    W, H = 240, 180
    cam = VgtkCore.BoundedCamera(VgtkCore.CameraModel(VgtkCore.LogicalIntrinsics(VgtkCore.CameraCalibrationMat(SMatrix{3, 3}(600.0, 0, 0, 0, 600.0, 0, 120.0, 90.0, 1))),
                                                      VgtkCore.Lens(VgtkCore.DoubleSphere(0.2, 0.3))), VgtkCore.Size2(W, H))
    value(q) = 0.2 + 0.002 * q[1] + 0.001 * q[2]
    img = [value((u - 0.5, v - 0.5)) for v in 1:H, u in 1:W]
    Rx(a) = SMatrix{3, 3}(1.0, 0, 0, 0, cos(a), sin(a), 0, -sin(a), cos(a)); Ry(a) = SMatrix{3, 3}(cos(a), 0, -sin(a), 0, 1.0, 0, sin(a), 0, cos(a))
    R = Rx(0.01) * Ry(-0.03); t = -(R * SVector(0.03, 0.042, -0.35))                    # page → camera, the camera on the printed side
    P = page_photograph(img, cam, R, t; TW = 60, TH = 84, tex_mm = 1.0, coarse = 4)
    @test size(P) == (60, 84) && all(isfinite, P)
    centre(i, j) = SVector(1e-3 * (i - 0.5), 1e-3 * (j - 0.5), 0.0)
    @test maximum(abs(P[i, j] - value(VgtkCore.project(cam, R * centre(i, j) + t))) for i in 1:60, j in 1:84) < 1e-4
    # from 0.12 m the page overflows the image: the texels no pixel sees are NaN, the others as before
    t2 = -(R * SVector(0.03, 0.042, -0.12)); P2 = page_photograph(img, cam, R, t2; TW = 60, TH = 84, tex_mm = 1.0, coarse = 4)
    @test any(isnan, P2) && count(isfinite, P2) > 500
    @test maximum(abs(P2[i, j] - value(VgtkCore.project(cam, R * centre(i, j) + t2))) for i in 1:60, j in 1:84 if isfinite(P2[i, j])) < 1e-4
    # a page behind the camera is in no photograph
    @test_throws ArgumentError page_photograph(img, cam, SMatrix{3, 3}(1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0), SVector(-0.03, -0.042, -0.33); TW = 60, TH = 84, tex_mm = 1.0)

    # the figure: a 48 × 120 page whose text block (40 × 110 texels from (4, 4) mm) is stacked from the design, the photograph and the
    # two textures, three lines each; the photograph at half the design's level comes back at the design's
    dir = mktempdir(); rdir = joinpath(dir, "evaluate"); mkpath(rdir)
    design = ones(48, 120); design[5:44, 5:114] .= block
    write_textures(rdir, Dict("learned" => 0.9 .* design, "pinhole" => 0.8 .* design), design; tex_mm = 1.0)
    r = Dict{String, Any}("asset" => "stacked.png", "size_in" => [2.0, 4.0], "dpi" => 100, "block_mm" => [4.0, 4.0, 40.0, 110.0], "lines_per_band" => 3, "white" => 0.85,
                          "label_fraction" => 0.2, "variants" => ["pinhole", "learned"], "labels" => ["ground\ntruth", "training\nphotograph", "pinhole", "ours"])
    rep = Dict{String, Any}("tex_mm" => 1.0)
    out = figure_texture_stacked(rdir, rep; render = r, dir = joinpath(dir, "plots"), photograph = 0.5 .* design)
    @test isfile(out) && (img = PNGFiles.load(out); (size(img, 2), size(img, 1))) == (200, 400)
    @test_throws ArgumentError figure_texture_stacked(rdir, rep; render = r, dir = joinpath(dir, "plots"), photograph = ones(10, 10))            # not the page's texels
    @test_throws ArgumentError figure_texture_stacked(rdir, rep; render = r, dir = joinpath(dir, "plots"), photograph = fill(NaN, 48, 120))      # shows nothing
    @test_throws ArgumentError figure_texture_stacked(rdir, rep; render = merge(r, Dict{String, Any}("block_mm" => [20.0, 4.0, 40.0, 110.0])), dir = joinpath(dir, "plots"),
                                                      photograph = 0.5 .* design)                                                                 # the block leaves the page
    @test_throws ArgumentError figure_texture_stacked(rdir, rep; render = merge(r, Dict{String, Any}("lines_per_band" => 4)), dir = joinpath(dir, "plots"),
                                                      photograph = 0.5 .* design)                                                                 # 17 lines needed, 13 printed
end
