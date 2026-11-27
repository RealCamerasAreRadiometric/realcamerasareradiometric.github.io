# The material start of a refinement (src/material_grid.jl; stage material_grid of scripts/radiometric_refinement.jl, its child
# scripts/material_grid.jl): the config's grid and its candidates, what a grid table refuses, the keys of the stage and of the
# refinement that reads it; and, on a tiny synthetic set, the search itself — a candidate's solves are the start's own, the candidate
# the photographs were rendered with is the one kept, and the record written is the start record of that material; and the candidates
# in parts (the driver's --grid-parts): the parts' rows are the search's number for number, and the search from them is the search.
# No run of a case of record: those need the GPU and the published data.

@testitem "material grid: the configs' grid, its candidates, the keys of the stage" default_imports = false begin
    using Test
    using DrWatson
    using TOML: TOML
    include(srcdir("experiments.jl"))
    include(srcdir("drivers.jl"))
    include(srcdir("radiometric_refinement.jl"))

    rpath = projectdir("configs", "experiments", "radiometric_refinement.toml")
    rc = load_radiometric_refinement_config(rpath)
    case = "ag10fe_bc5_warped"

    @testset "the grid of record: a coarse uniform grid over the physical range, a candidate at each cell's midpoint" begin
        g = rc["material_grid"]
        @test g == Dict{String, Any}("alpha_range" => [0.0, 1.0], "alpha_cells" => 3, "eta_range" => [1.0, 3.0], "eta_cells" => 3)
        @test cell_midpoints(g["alpha_range"], g["alpha_cells"]) ≈ [1 / 6, 1 / 2, 5 / 6]
        @test cell_midpoints(g["eta_range"], g["eta_cells"]) ≈ [4 / 3, 2, 8 / 3]
        @test cell_midpoints([0.2, 1.0], 1) ≈ [0.6] && cell_midpoints([1, 3], 4) == [1.25, 1.75, 2.25, 2.75]
        c = material_candidates(g)
        @test length(c) == 81 && allunique(c) && keys(first(c)) == Symbol.(MATERIAL_START_KEYS)
        @test all(x -> 0 < x.alpha_ink <= 1 && 0 < x.alpha_noink <= 1 && x.eta_ink > 1 && x.eta_noink > 1, c)
        @test c[1].alpha_ink ≈ 1 / 6 && c[2].alpha_ink ≈ 1 / 2 && c[2].alpha_noink ≈ 1 / 6 && c[4].alpha_noink ≈ 1 / 2   # the ink's α runs fastest
        @test c[81] == (; alpha_ink = c[3].alpha_ink, alpha_noink = c[9].alpha_noink, eta_ink = c[27].eta_ink, eta_noink = c[81].eta_noink)
        # the vignetting and the board-warping refinements read the same grid (their [material_grid] source): one place
        for f in ("radiometric_refinement_vignetting.toml", "radiometric_refinement_board_warping.toml")
            p = projectdir("configs", "experiments", f)
            @test TOML.parsefile(p)["material_grid"] == Dict{String, Any}("source" => "configs/experiments/radiometric_refinement.toml")
            @test load_radiometric_refinement_config(p)["material_grid"] == g
        end
        # a study's override config is its base with its cases: the base's grid
        @test load_radiometric_refinement_config(projectdir("configs", "studies", "radiometric_refinement_smoothness_si30fe.toml"))["material_grid"] == g
    end

    @testset "a grid table refuses what is not a grid" begin
        good = Dict{String, Any}("alpha_range" => [0.0, 1.0], "alpha_cells" => 3, "eta_range" => [1.0, 3.0], "eta_cells" => 3)
        refused(edit!) = (t = deepcopy(good); edit!(t); try check_material_grid(t, "the table"); nothing catch err; err end)
        @test refused(identity) === nothing
        e = refused(t -> delete!(t, "eta_cells"));               @test e isa ArgumentError && occursin("eta_cells", e.msg)
        e = refused(t -> t["alphas"] = [0.1]);                   @test e isa ArgumentError && occursin("alphas", e.msg)
        e = refused(t -> t["alpha_range"] = [0.0, 1.5]);         @test e isa ArgumentError && occursin("alpha_range", e.msg)
        e = refused(t -> t["alpha_range"] = [0.5, 0.5]);         @test e isa ArgumentError && occursin("alpha_range", e.msg)
        e = refused(t -> t["alpha_range"] = [0.0, 0.5, 1.0]);    @test e isa ArgumentError && occursin("alpha_range", e.msg)
        e = refused(t -> t["eta_range"] = [0.9, 3.0]);           @test e isa ArgumentError && occursin("eta_range", e.msg)
        e = refused(t -> t["eta_range"] = [3.0, 1.0]);           @test e isa ArgumentError && occursin("eta_range", e.msg)
        e = refused(t -> t["alpha_cells"] = 0);                  @test e isa ArgumentError && occursin("alpha_cells", e.msg)
        e = refused(t -> t["eta_cells"] = 2.5);                  @test e isa ArgumentError && occursin("eta_cells", e.msg)
        @test refused(t -> t["eta_range"] = [1, 1.6]) === nothing && refused(t -> t["alpha_cells"] = 1) === nothing
        # the config's table: its own keys, or `source` alone naming a config that states them; a config without it is refused
        msg(c) = try material_grid_table(c, "c.toml"); "" catch err; err isa ArgumentError ? err.msg : rethrow() end
        @test material_grid_table(Dict{String, Any}("material_grid" => good), "c.toml") == good
        @test occursin("no [material_grid] table", msg(Dict{String, Any}()))
        @test occursin("nothing beside it", msg(Dict{String, Any}("material_grid" => merge(good, Dict("source" => "configs/experiments/radiometric_refinement.toml")))))
        @test occursin("states no [material_grid] of its own", msg(Dict{String, Any}("material_grid" => Dict("source" => "configs/experiments/detector_bias.toml"))))
        @test occursin("states no [material_grid] of its own", msg(Dict{String, Any}("material_grid" => Dict("source" => "configs/experiments/radiometric_refinement_vignetting.toml"))))
        dir = mktempdir(); raw = TOML.parsefile(rpath)
        function loaded(edit!)
            c = deepcopy(raw); edit!(c)
            p = joinpath(dir, "radiometric_refinement.toml")
            open(io -> TOML.print(io, c), p, "w")
            return try load_radiometric_refinement_config(p) catch err; err end
        end
        e = loaded(c -> delete!(c, "material_grid"));            @test e isa ArgumentError && occursin("no [material_grid] table", e.msg)
        e = loaded(c -> c["material_grid"]["eta_cells"] = -1);   @test e isa ArgumentError && occursin("eta_cells", e.msg)
        @test loaded(c -> c["material_grid"]["alpha_cells"] = 5)["material_grid"]["alpha_cells"] == 5
    end

    @testset "the keys: start → material grid → refinement" begin
        s, g, r = start_product(rc, case), material_grid_product(rc, case), refine_product(rc, case)
        @test g.name == "material_grid" && g.output == RadiometricCamera.START_RECORD_FILE          # a start record: what bin/refine.jl --start reads
        @test g.record["material_grid"] == Dict{String, Any}(case => rc["material_grid"]) && g.record["upstream"] == Dict{String, Any}("start" => s.key)
        @test g.record["record"] == EXPERIMENT_RECORD_SCHEMAS.material_grid && g.record["start_record"] == RadiometricCamera.RECORD_SCHEMAS.refinement_start
        @test r.record["upstream"] == Dict{String, Any}("material_grid" => g.key)                   # the refinement reads the grid's record, not the start's
        @test dirname(product_dir(g)) == joinpath(results_root(), "radiometric_refinement", "material_grid")
        # another grid: another material start and another refinement, the same start
        c2 = deepcopy(rc); c2["material_grid"]["alpha_cells"] = 4
        @test start_product(c2, case).key == s.key && material_grid_product(c2, case).key != g.key && refine_product(c2, case).key != r.key
        # a start setting (the case's own [material]): all three; a refinement-only setting: the refinement alone
        c2 = deepcopy(rc); c2["cases"][case]["pipeline"]["material"]["mu"] = 0.3
        @test start_product(c2, case).key != s.key && material_grid_product(c2, case).key != g.key && refine_product(c2, case).key != r.key
        c2 = deepcopy(rc); c2["cases"][case]["pipeline"]["lm"]["max_iter"] = 7
        @test start_product(c2, case).key == s.key && material_grid_product(c2, case).key == g.key && refine_product(c2, case).key != r.key
        # the grid as its config's `source` states it keys as the grid stated: the resolved values key, whichever file holds them
        rv = load_radiometric_refinement_config(projectdir("configs", "experiments", "radiometric_refinement_vignetting.toml"))
        @test material_grid_product(rv, "f4_tamron_spline_warped").record["material_grid"]["f4_tamron_spline_warped"] == rc["material_grid"]
        # a config that states no grid (another driver's: the detector bias) refines from its start, keyed as before
        c2 = deepcopy(rc); delete!(c2, "material_grid")
        @test refine_product(c2, case).record["upstream"] == Dict{String, Any}("start" => s.key)
        # the stage's child reads the start's own settings: the table of the start, never a refinement-only key
        st = start_settings(rc["cases"][case])
        @test !haskey(st["pipeline"], "lm") && st["pipeline"]["material"] == rc["cases"][case]["pipeline"]["material"]
    end

    @testset "the candidates in parts: every candidate once, the option, the keys" begin
        @test part_candidates(81, 1, 3) == 1:3:81 && part_candidates(81, 3, 3) == 3:3:81 && part_candidates(16, 2, 5) == [2, 7, 12]
        for (count, parts) in ((81, 3), (81, 4), (16, 3), (5, 5))
            shares = [collect(part_candidates(count, k, parts)) for k in 1:parts]
            @test sort(reduce(vcat, shares)) == 1:count && maximum(length, shares) - minimum(length, shares) <= 1
        end
        for (k, n) in ((0, 3), (4, 3), (1, 82))
            e = try part_candidates(81, k, n); nothing catch err; err end
            @test e isa ArgumentError && occursin("part $k of $n", e.msg)
        end
        @test material_grid_parts(String[]) === nothing && material_grid_parts(["--stage=material_grid", "--grid-parts=3"]) == 3
        for bad in ("1", "0", "-2", "x", "2.5")
            e = try material_grid_parts(["--grid-parts=$bad"]); nothing catch err; err end
            @test e isa ArgumentError && occursin("--grid-parts=$bad", e.msg)
        end
        @test lowest_cost([(; cost = 3.0), (; cost = 1.0), (; cost = 1.0), (; cost = Inf), (; cost = NaN)]) == 2   # of equal ones the first
        @test lowest_cost([(; cost = NaN), (; cost = Inf)]) === nothing && lowest_cost(NamedTuple[]) === nothing
        # a part's product: keyed as the material start (the grid, the start's key) and by the part; the material start's key is no part's
        g = material_grid_product(rc, case); p = material_grid_part_product(rc, case, 2, 3)
        @test p.name == "material_grid_part" && p.output == "material_grid_part.json" == basename(material_grid_part_record("x"))
        @test p.record["part"] == [2, 3] && p.record["record"] == EXPERIMENT_RECORD_SCHEMAS.material_grid_part
        @test p.record["material_grid"] == g.record["material_grid"] && p.record["upstream"] == g.record["upstream"]
        @test dirname(product_dir(p)) == joinpath(results_root(), "radiometric_refinement", "material_grid_part")
        @test allunique([g.key; [material_grid_part_product(rc, case, k, n).key for n in (2, 3) for k in 1:n]])
        c2 = deepcopy(rc); c2["material_grid"]["alpha_cells"] = 4
        @test material_grid_part_product(c2, case, 2, 3).key != p.key
    end

    @testset "the stage and its child in the drivers" begin
        drv = read(projectdir("scripts", "radiometric_refinement.jl"), String)
        @test occursin("Stage(:material_grid,", drv) && occursin(":material_grid => [CFG[\"cases\"][c] for c in CASES]", drv)   # a GPU stage
        @test findfirst("Stage(:start,", drv).start < findfirst("Stage(:material_grid,", drv).start < findfirst("Stage(:refine,", drv).start
        # the parts of the candidates: a GPU stage before the stage that reads them, its items a case's part each, split by --shard
        @test findfirst("Stage(:start,", drv).start < findfirst("Stage(:material_grid_part,", drv).start < findfirst("Stage(:material_grid,", drv).start
        @test occursin(":material_grid_part => [CFG[\"cases\"][c] for c in CASES]", drv) && occursin("const GRID_PARTS = material_grid_parts()", drv)
        @test occursin("cli_shard(ARGS, [(c, k) for c in CASES for k in 1:something(GRID_PARTS, 0)])", drv)
        @test !occursin("start_with_refine", drv)               # the start is its own child: the material grid reads its record
        child = read(projectdir("scripts", "material_grid.jl"), String)
        @test occursin("load_calibration_start(START; options = OPTIONS, device = DEVICE)", child) && occursin("save_calibration_start(", child)
        @test occursin("cli_option(ARGS, \"part\", \"\")", child) && occursin("cli_option(ARGS, \"parts\", \"\")", child)
        @test findfirst("read_material_grid_parts(PARTS, GRID)", child).start < findfirst("load_calibration_start(", child).start   # refused before the start is loaded
        src = read(projectdir("src", "drivers.jl"), String)
        @test occursin("\"prepare,start,material_grid,refine\"", src)   # another driver's refinement upstream runs the stage too
    end
end

@testitem "material grid: a candidate's solves are the start's own; the search keeps the rendered material; its start record" default_imports = false begin
    using Test
    using DrWatson
    using StaticArrays: SMatrix, SVector
    using PNGFiles: PNGFiles
    using ColorTypes: ColorTypes, RGB
    const N0f16 = ColorTypes.FixedPointNumbers.N0f16                     # the 16-bit linear photographs (ColorTypes' fixed-point numbers)
    import VisualGeometryToolkitCore as VgtkCore
    import VisualRadiometryToolkitCore as VrtkCore
    import VisualGeometryToolkitIO as VgtkIO
    using VisualGeometryToolkitCore: exp_so3
    using VisualRadiometryToolkitCore: EnvironmentLight, SphericalHarmonicMap
    include(srcdir("experiments.jl"))
    include(srcdir("drivers.jl"))
    const RC = RadiometricCamera
    rct = joinpath(pkgdir(RC), "test", "unit")
    include(joinpath(rct, "fixture_board.jl")); include(joinpath(rct, "fixture_camera.jl"))   # RadiometricCamera's test scene (its checkpoint item's)

    # the grid: two cells per quantity, sixteen candidates; the photographs are rendered with one of them (the truth), at the μ the
    # case's [material] starts from: a single sweep of the blocks leaves μ near its start, so that is the μ the solves can meet
    grid = Dict{String, Any}("alpha_range" => [0.0, 0.5], "alpha_cells" => 2, "eta_range" => [1.0, 2.0], "eta_cells" => 2)
    truth = (; alpha_ink = 0.125, alpha_noink = 0.375, eta_ink = 1.75, eta_noink = 1.25)
    candidates = material_candidates(grid)
    @test length(candidates) == 16 && count(==(truth), candidates) == 1
    μ = 0.5
    mat = VrtkCore.Material(; roughness_ink = truth.alpha_ink, roughness_paper = truth.alpha_noink, refractive_index_ink = truth.eta_ink,
                            refractive_index_paper = truth.eta_noink, diffuse_weight = μ)

    tex = fixture_board()
    board = RC.TexturedTarget([-1.0, -1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], tex; facing = [0.0, 0.0, 1.5])
    W = H = 24
    glens = RC.grid_model(cam(VgtkCore.Lens(VgtkCore.DoubleSphere(0.25, 0.4)), 30.912, (12.24, 11.52), W, H), 10, 10)
    poses = [VgtkCore.EuclideanMap(SMatrix{3, 3}(exp_so3(SVector(π, 0.0, 0.0)) * exp_so3(SVector(0.15 * cos(k), 0.12 * sin(k), 0.1 * k))),
                                   SVector{3}(SVector(0.3 * cos(k), 0.3 * sin(k), 1.4 + 0.1 * k)); check = false) for k in 1:2]
    detections(k) = [(p, X) for X in (SVector(x, y, 0.0) for x in -0.5:0.1:0.5 for y in -0.5:0.1:0.5)
                     for p in (VgtkCore.project(glens, poses[k].R' * (X - poses[k].t)),) if all(isfinite, p) && all(0 .< p .< W)]
    a = zeros(3, 9); a[:, 1] .= (2.0, 1.9, 1.8); a[:, 3] .= (0.6, 0.5, 0.4); a[:, 4] .= (0.3, 0.35, 0.4)   # a lighting with a direction
    save_photos(files, imgs) = (gain = 0.8 / maximum(maximum, imgs);     # the flux is linear in the lighting: no photograph is clipped
                                foreach(k -> PNGFiles.save(files[k], [RGB{N0f16}(clamp.(gain .* imgs[k][i, j, :], 0, 1)...) for i in 1:H, j in 1:W]), eachindex(files)))

    mktempdir() do dir
        # the case's own [material]: another material than the photographs' (the start stage's, the record's comparison row)
        table = Dict{String, Any}("io" => Dict{String, Any}("channel" => "RGB"), "masks" => Dict{String, Any}("samples_per_view" => 10^9, "seed" => 1),
                                  "rendering" => Dict{String, Any}("N" => 4, "n_rho" => 2, "n_phi" => 6, "n_p" => 2),
                                  "material" => Dict{String, Any}("alpha_ink" => 0.3, "alpha_noink" => 0.2, "eta_ink" => 1.5, "eta_noink" => 1.6, "mu" => μ))
        run_options(t, out) = RC.pipeline_options(merge(t, Dict{String, Any}("io" => merge(t["io"], Dict{String, Any}("out" => out)))))
        opts = run_options(table, joinpath(dir, "start"))
        nodes(s) = Matrix{Float64}(VrtkCore.control_points(s.camera.vignetting)); sh(s) = Matrix{Float64}(VrtkCore.coefficients(s.light))
        geometry(files) = RC.StartGeometry(glens, poses, [detections(k) for k in 1:2], board; names = basename.(files), init_region = VgtkCore.Rect2{Float64}(0, 0, W, H))
        started(o, files) = RC.start_calibration(geometry(files), files; options = o, device = RC.KernelAbstractions.CPU())
        # the photographs: rendered through the START's own camera (its focus plane and aperture, V flat), poses, masks and quadrature
        # with the truth's material, so the start's model holds them exactly at the truth. A first start, on stand-in photographs
        # of the scene, gives that camera
        focus = RC.Focus(RC.FocusSpline(VgtkCore.grid_nodes(3, VgtkCore.Rect(0, 0, W, H))), fill(1 / 1.5, 9))
        camera0 = RC.RadiometricCameraModel(glens, focus, 0.0386, RC.gauge_center_value(VrtkCore.TensorProductVignetting((4, 4), VgtkCore.Rect(0, 0, W, H); degree = 2)))
        rs = RC.RenderSettings(; n_𝐩 = 2, n_ρ = 2, n_φ = 6, N = 4)
        b = RC.𝔟(board); light0 = EnvironmentLight(SphericalHarmonicMap(a, 2), b, RC.sphere_radius(b, [p.t for p in poses]))
        standin = [joinpath(dir, "standin", "V0$k.png") for k in 1:2]; mkpath(dirname(first(standin)))
        save_photos(standin, [RC.render_view(k, camera0, poses[k], mat, light0, board, rs, RC.compute_mask(camera0, poses[k], board, rs; δ = 0.02, backend = RC.KernelAbstractions.CPU())) for k in 1:2])
        s0 = started(run_options(table, joinpath(dir, "standin_start")), standin)
        v0 = s0.camera.vignetting
        camera = VrtkCore.with_vignetting(s0.camera, VrtkCore.with_control_points(v0, ones(size(VrtkCore.control_points(v0)))))
        photos = [joinpath(dir, "V0$k.png") for k in 1:2]
        save_photos(photos, [RC.render_view(k, camera, s0.poses[k], mat, VrtkCore.with_coefficients(s0.light, a), s0.board, s0.settings, s0.masks[k]) for k in 1:2])
        s = started(opts, photos)
        @test s.masks == s0.masks
        own = s.start.costs; μ_own = s.start.μ_lin; P_own = nodes(s); a_own = sh(s)

        @testset "a candidate's solves are the start's own: at the start's material, the start's state and costs" begin
            r = material_sweep!(s, opts, (; alpha_ink = 0.3, alpha_noink = 0.2, eta_ink = 1.5, eta_noink = 1.6))
            @test collect(r.costs.init) ≈ collect(own.init) rtol = 1e-10
            @test collect(r.costs.linear) ≈ collect(own.linear) rtol = 1e-10
            @test r.mu ≈ μ_own rtol = 1e-10
            @test nodes(s) ≈ P_own rtol = 1e-10
            @test sh(s) ≈ a_own rtol = 1e-10
            @test r.costs.linear[1] <= r.costs.init[1] && isempty(r.vignetting_gcv)   # exact least squares on the same pixels: the cost cannot rise
            # the state a candidate is solved from is not read: from another material's solved state, the same result
            material_sweep!(s, opts, truth)
            r2 = material_sweep!(s, opts, (; alpha_ink = 0.3, alpha_noink = 0.2, eta_ink = 1.5, eta_noink = 1.6))
            @test collect(r2.costs.linear) ≈ collect(own.linear) rtol = 1e-10
            @test r2.mu ≈ μ_own rtol = 1e-10
            @test nodes(s) ≈ P_own rtol = 1e-10
        end

        log = IOBuffer()
        result = material_grid_search!(s, opts, grid; io = log)
        chosen = result.candidates[result.chosen]
        @testset "the search: every candidate in its order, the lowest cost kept — the material the photographs were rendered with" begin
            @test [(; c.alpha_ink, c.alpha_noink, c.eta_ink, c.eta_noink) for c in result.candidates] == candidates
            @test result.chosen == findfirst(==(truth), candidates) == argmin([c.cost for c in result.candidates])
            @test all(c -> c.cost <= c.cost_init && isfinite(c.rel_rms) && 0 <= c.mu <= 1, result.candidates)
            # at the truth the start's model holds the photographs to their 16-bit rounding; every other candidate is far from it
            @test chosen.rel_rms < 1e-3 && chosen.rel_rms < 0.1 * minimum(c.rel_rms for (i, c) in enumerate(result.candidates) if i != result.chosen)
            @test chosen.mu ≈ μ atol = 1e-3
            # the comparison row: the case's own [material] with the start record's costs
            @test result.start == (; alpha_ink = 0.3, alpha_noink = 0.2, eta_ink = 1.5, eta_noink = 1.6, mu = μ_own, cost_init = own.init[1],
                                     rel_rms_init = own.init[2], cost = own.linear[1], rel_rms = own.linear[2])
            @test chosen.cost < result.start.cost
            # the state is left at the chosen candidate, and its start's record of the costs and μ is the candidate's
            @test (s.mat.roughness_ink, s.mat.roughness_paper, s.mat.diffuse_weight) == (truth.alpha_ink, truth.alpha_noink, chosen.mu)
            @test s.start.costs.linear == (chosen.cost, chosen.rel_rms) && s.start.costs.init == (chosen.cost_init, chosen.rel_rms_init) && s.start.μ_lin == chosen.mu
            @test first(VgtkCore.cost(s)) ≈ chosen.cost rtol = 1e-10
            out = String(take!(log))
            @test occursin("material grid: 16 candidates", out) && count("  *\n", out) >= 1 && occursin("material start: candidate $(result.chosen),", out)
        end

        chosen_options = material_start_options(opts, chosen)
        @testset "the chosen material's start: what the start computes for that material" begin
            @test chosen_options.material.alpha_ink === truth.alpha_ink && chosen_options.material.eta_noink === truth.eta_noink
            @test chosen_options.material.mu == μ && chosen_options.rendering == opts.rendering && chosen_options.masks == opts.masks   # μ and all else: the case's
            s1 = started(run_options(RC.config_dict(chosen_options), joinpath(dir, "start_chosen")), photos)
            @test collect(s1.start.costs.linear) ≈ collect(s.start.costs.linear) rtol = 1e-10
            @test collect(s1.start.costs.init) ≈ collect(s.start.costs.init) rtol = 1e-10
            @test s1.start.μ_lin ≈ s.start.μ_lin rtol = 1e-10
            @test nodes(s1) ≈ nodes(s) rtol = 1e-10
            @test sh(s1) ≈ sh(s) rtol = 1e-10
            @test s1.masks == s.masks
        end

        @testset "the stage's records: material_grid.json and the start record of the chosen material" begin
            gdir = joinpath(dir, "material_grid")
            mkpath(gdir)
            path = write_material_grid(gdir, grid, result)
            @test path == material_grid_record(gdir) == joinpath(gdir, "material_grid.json")
            rec = VgtkIO.read_record(path)
            @test rec["schema"] == EXPERIMENT_RECORD_SCHEMAS.material_grid && rec["grid"]["alpha_cells"] == 2 && rec["grid"]["eta_range"] == [1.0, 2.0]
            @test length(rec["candidates"]) == 16 && rec["chosen"]["index"] == result.chosen && rec["chosen"]["cost"] == chosen.cost
            @test rec["candidates"][result.chosen]["rel_rms"] == chosen.rel_rms && rec["start"]["alpha_ink"] == 0.3 && rec["start"]["cost"] == own.linear[1]
            # the refinement's table: the chosen α and η, bit for bit; μ and every other key the case's
            full = merge(table, Dict{String, Any}("lm" => Dict{String, Any}("max_iter" => 3)))
            t2 = with_material_start(full, gdir)
            @test all(k -> t2["material"][k] === getproperty(truth, Symbol(k)), MATERIAL_START_KEYS) && t2["material"]["mu"] == μ
            @test full["material"]["alpha_ink"] == 0.3 && delete!(deepcopy(t2), "material") == delete!(deepcopy(full), "material")   # a copy; nothing else moved
            RC.save_calibration_start(gdir, s; options = chosen_options)
            st = RC.load_refinement_start(gdir)
            @test st.costs == s.start.costs && st.μ_lin == chosen.mu && st.masks == s.masks
            @test st.options == RC.start_options(chosen_options) && st.options.material.alpha_noink === truth.alpha_noink
            m = RC.material(st.state.target.radiometry)
            @test (m.roughness_ink, m.roughness_paper, m.diffuse_weight) == (truth.alpha_ink, truth.alpha_noink, chosen.mu)
            # bin/refine.jl's check of a start: the refinement's table at the chosen material is this record's; the case's own is refused
            @test RC.check_start_options(st, run_options(t2, joinpath(dir, "refine"))) === nothing
            e = try RC.check_start_options(st, run_options(full, joinpath(dir, "refine"))); nothing catch err; err end
            @test e isa ArgumentError && occursin("material.alpha_ink", e.msg) && occursin("other start options", e.msg)
        end

        @testset "the candidates in parts: the parts' rows are the search's, and the search from them is the search" begin
            s2 = started(opts, photos)                               # the start again, as the stage's child loads it for every part
            @test s2.start.costs == own && s2.start.μ_lin == μ_own
            pdirs = [joinpath(dir, "part_$k") for k in 1:3]; foreach(mkpath, pdirs)
            for k in 1:3
                ix = part_candidates(length(candidates), k, 3)
                swept = material_sweeps!(s2, opts, candidates, ix; io = devnull)
                @test swept.rows == result.candidates[ix]            # number for number, whatever was solved before: what the split rests on
                @test swept.best.index == ix[argmin([r.cost for r in swept.rows])]
                path = write_material_grid_part(pdirs[k], grid, k, 3, ix, swept.rows)
                rec = VgtkIO.read_record(path)
                @test path == material_grid_part_record(pdirs[k]) && rec["schema"] == EXPERIMENT_RECORD_SCHEMAS.material_grid_part
                @test (rec["part"], rec["parts"]) == (k, 3) && rec["grid"]["eta_cells"] == 2 && [c["index"] for c in rec["candidates"]] == collect(ix)
                @test rec["candidates"][1]["cost"] == swept.rows[1].cost && rec["candidates"][end]["alpha_ink"] == candidates[last(ix)].alpha_ink
            end
            @test s2.start.costs == own                              # a part's solves leave the start's record of its costs alone
            rows = read_material_grid_parts(pdirs, grid)
            @test rows == result.candidates && rows isa Vector{NamedTuple} && keys(first(rows)) == keys(first(result.candidates))
            @test lowest_cost(rows) == result.chosen
            log2 = IOBuffer()
            r2 = material_grid_search!(s2, opts, grid; io = log2, rows)
            @test r2 == result                                       # the stage's record: the start's row, every candidate's, the chosen one
            material(x) = (x.mat.roughness_ink, x.mat.roughness_paper, x.mat.refractive_index_ink, x.mat.refractive_index_paper, x.mat.diffuse_weight)
            @test material(s2) == material(s) && nodes(s2) == nodes(s) && sh(s2) == sh(s)   # the state the start record holds: the search's
            @test s2.start.costs == s.start.costs && s2.start.μ_lin == s.start.μ_lin
            out = String(take!(log2))
            @test occursin("the solves run at the chosen candidate alone", out) && count("  *\n", out) == 1 && occursin("material start: candidate $(result.chosen),", out)
            # what the parts' records refuse: a part missing, a part twice, the parts of another grid
            msg(f) = try f(); "" catch err; err isa ArgumentError ? err.msg : rethrow() end
            @test occursin("hold no row of the candidates 3, 6, 9, 12, 15", msg(() -> read_material_grid_parts(pdirs[1:2], grid)))
            @test occursin("has a row of another part already", msg(() -> read_material_grid_parts([pdirs; pdirs[2]], grid)))
            @test occursin("is not a candidate of this grid", msg(() -> read_material_grid_parts(pdirs, merge(grid, Dict{String, Any}("alpha_cells" => 3)))))
            # what the search refuses: rows that are not this start's solves (the chosen one's cost, one unit in the last place off), too few rows
            off = [i == result.chosen ? (; r..., cost = prevfloat(r.cost)) : r for (i, r) in enumerate(rows)]
            @test occursin("candidate $(result.chosen)'s solves give", msg(() -> material_grid_search!(s2, opts, grid; io = devnull, rows = off)))
            @test occursin("15 rows for its 16 candidates", msg(() -> material_grid_search!(s2, opts, grid; io = devnull, rows = rows[1:15])))
        end

        @testset "a start whose grid smoothness weight was chosen by GCV is refused" begin
            s.start = (; s.start..., grid_gcv = (; β = 0.1))
            e = try material_grid_search!(s, opts, grid; io = devnull); nothing catch err; err end
            @test e isa ArgumentError && occursin("GCV", e.msg)
            e = try material_sweeps!(s, opts, candidates, 1:1; io = devnull); nothing catch err; err end   # a part of the candidates: before its solves
            @test e isa ArgumentError && occursin("GCV", e.msg)
        end
    end
end
