# The experiment drivers (scripts/radiometric_refinement.jl and the others of DRIVERS), their configs of record
# (configs/experiments/): every config loads and plans, a config refuses what is not its own, a Git LFS pointer is not data, the RAW
# decoder writes the samples it is given, the ported evaluation reproduces the stored runs
# (fixtures/evaluation/, cut from them by make_fixtures.jl), and a driver writes only under results_root() (and its assets under
# plotsdir()). The full runs need the GPU and the published data; they are not this suite (REPRODUCE.md).

# The drivers, their stages and a driver's run in a fresh process: the helpers of both items below.
@testsetup module DriverRuns
    using DrWatson

    const DRIVERS = ("radiometric_refinement", "focus_surfaces", "texture_recovery", "frontends", "sensor_characterisation",
                     "image_regions", "distance_maps", "timing")
    const DRIVER_STAGE_NAMES = Dict(
        "radiometric_refinement" => ("prepare", "detect", "start", "refine"),
        "focus_surfaces" => ("panels", "figures"),
        "texture_recovery" => ("prepare", "detect", "sheet", "init", "refine", "folds", "evaluate", "fold_report", "compare", "report", "figures"),
        "frontends" => ("prepare", "detect", "init", "evaluate"),
        "sensor_characterisation" => ("evaluate",),
        "image_regions" => ("detect",),
        "distance_maps" => ("prepare",),
        "timing" => ("start", "time", "record"))
    experiment_config(d) = d == "texture_recovery" ? projectdir("configs", "experiments", "texture_recovery_14mm_small.toml") :
                           d == "timing" ? projectdir("configs", "supplement", "timing.toml") : projectdir("configs", "experiments", "$d.toml")
    const DRIVER_THREADS = "2,1"

    # A driver in a fresh process with both trees bound; (succeeded, output).
    function run_driver_script(d, args...; data, results, config = experiment_config(d))
        cmd = `$(Base.julia_cmd()) --threads=$DRIVER_THREADS --project=$(projectdir()) $(projectdir("scripts", "$d.jl")) --config=$config $(collect(String, args))`
        out = IOBuffer()
        ok = success(pipeline(addenv(cmd, "VGTK_DATA_ROOT" => data, "VGTK_RESULTS_ROOT" => results);
                              stdout = out, stderr = out))
        return ok, String(take!(out))
    end
    tree_state(dir) = Dict(relpath(joinpath(r, f), dir) => mtime(joinpath(r, f)) for (r, _, fs) in walkdir(dir) for f in fs)
end

@testitem "experiment drivers: --plan lists each driver's stages" default_imports = false setup = [DriverRuns] begin
    using Test
    using DrWatson: projectdir
    using .DriverRuns: DRIVERS, DRIVER_STAGE_NAMES, run_driver_script

    @testset "--plan of $d lists its stages" for d in DRIVERS
        ok, out = run_driver_script(d, "--plan"; data = mktempdir(), results = mktempdir())
        ok || @info out
        @test ok
        for st in DRIVER_STAGE_NAMES[d]
            @test occursin(Regex("^\\s+$st\\s+(check|\\d+/\\d+ produced)", "m"), out)
        end
        @test occursin("plan only", out)
    end
    # the chrome evaluation on its paper config (configs/experiments/chrome_evaluation_paper.toml): the paper stages
    @testset "--plan of chrome_evaluation on its paper config lists the paper stages" begin
        ok, out = run_driver_script("chrome_evaluation_paper", "--plan"; data = mktempdir(), results = mktempdir(),
                                    config = projectdir("configs", "experiments", "chrome_evaluation_paper.toml"))
        ok || @info out
        @test ok
        for st in ("sets", "figures")
            @test occursin(Regex("^\\s+$st\\s+\\d+/\\d+ produced", "m"), out)
        end
        @test occursin("plan only", out)
    end
end

@testitem "experiment drivers" default_imports = false setup = [DriverRuns] begin
    using Test
    using DrWatson
    using TOML: TOML
    import PNGFiles
    using StaticArrays: SVector, SMatrix
    using LinearAlgebra: normalize
    using Statistics: median
    using StableRNGs: StableRNG
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: thrown_message
    import VisualGeometryToolkitIO as VgtkIO
    using ColorTypes: Gray, N0f8
    using JpegTurbo: jpeg_encode
    include(srcdir("experiments.jl"))
    include(srcdir("drivers.jl"))
    include(srcdir("evaluation.jl"))
    include(srcdir("radiometric_refinement.jl"))
    using .Evaluation: reprojection_vectors, truepose_vectors, truepose_scores
    using .DriverRuns: DRIVERS, experiment_config, DRIVER_THREADS, run_driver_script, tree_state

    @testset "--force re-produces a start whose stage runs (code review 0.11.0, finding 1)" begin
        withenv("VGTK_RESULTS_ROOT" => mktempdir(), "VGTK_SCRATCH" => "1") do
            st = Product("d", "start", Dict("n" => 1); output = "start.json")
            @test !start_reused(st; argv = String[])                                   # not produced: produced with its refinement
            produce(d -> (write(joinpath(d, "start.json"), "{}"); nothing), st; driver = "d", config_source = "c")
            @test start_reused(st; argv = String[]) && start_reused(st; argv = ["--stage=refine"])
            @test !start_reused(st; argv = ["--force"]) && !start_reused(st; argv = ["--stage=start,refine", "--force"])   # forced: again
            @test start_reused(st; argv = ["--stage=refine", "--force"])               # the refinement alone forced: from its start
            @test stage_runs("start"; argv = String[]) && !stage_runs("start"; argv = ["--stage=refine"])
        end
    end

    @testset "a consumer produces its missing upstream under its own key, never refuses it (with_upstream, 2026-11-15)" begin
        withenv("VGTK_RESULTS_ROOT" => mktempdir(), "VGTK_SCRATCH" => "1") do
            made = Ref(0)
            up = Product("d", "refine", Dict("n" => 1); output = "page.json")
            make_up() = produce(d -> (made[] += 1; write(joinpath(d, "page.json"), "{}"); nothing), up; driver = "d", config_source = "c")
            down = Product("d", "report", Dict("n" => 1, "upstream" => Dict("refine" => up.key)); output = "summary.md")
            make_down() = (with_upstream(make_up, down); produce(d -> (write(joinpath(d, "summary.md"), read(product_output(up), String)); nothing), down;
                                                                driver = "d", config_source = "c"))
            make_down()
            @test produced(up) && produced(down) && made[] == 1                    # the upstream made first, under its own key
            rm(product_dir(up); recursive = true); rm(product_dir(down); recursive = true)
            make_down()
            @test produced(up) && produced(down) && made[] == 2                    # deleted: re-produced, not refused
            rm(product_dir(up); recursive = true)
            make_down()
            @test !produced(up) && made[] == 2                                     # a produced consumer reads nothing new
        end
        # the texture stages read their upstream through it (the report through produce_upstream, the one consumer path): no "run its
        # driver first" refusal left in the driver
        tex = read(projectdir("scripts", "texture_recovery.jl"), String)
        @test count("with_upstream(", tex) == 3 && occursin("produce_upstream(texture_product()", tex) && !occursin("require_produced", tex)
    end

    @testset "the one-child path resolves its inputs once (code review 0.11.0, finding 7)" begin
        src = read(projectdir("src", "drivers.jl"), String)
        body = src[findfirst("function run_refine(", src).start:findfirst("\"\"\"\n    start_with_refine(", src).start]
        @test count("pipeline_input_paths(", body) == 1 && count("resolve_pipeline(", body) == 1
        cal = read(projectdir("scripts", "calibrate.jl"), String)
        rb = cal[findfirst("refine(p) = produce(p", cal).start:findfirst("evaluate(p) = produce(p", cal).start]
        @test count("prepare()", rb) == 1                                               # the case's inputs checked once
    end

    @testset "the configs of record" begin
        # the paper's experiments: calibration accuracy (radiometric_refinement, its chrome evaluations chrome_evaluation_*, the figures
        # focus_surfaces) and the f/4 refinements (radiometric_refinement_vignetting); the macro defocus (2026-11-14), the synthetic
        # detector bias and the panorama chains (rc-0116, 2026-11-18) left the project with their configs
        lenses = ("14mm", "fisheye", "tamron")
        @test sort(readdir(projectdir("configs", "experiments"))) ==
              sort(vcat(["radiometric_refinement.toml", "radiometric_refinement_vignetting.toml",
                         "focus_surfaces.toml", "texture_recovery_14mm_small.toml", "chrome_evaluation_fx3_14mm_central30cm.toml",
                         "chrome_evaluation_fx3_fisheye_30cm_si.toml",   # the fisheye 30 cm set (2026-11-16: in place of the excluded 27.10 set)
                         "chrome_evaluation_paper.toml",   # the chrome evaluation's paper stages: the sets, their tables and figures
                         "frontends.toml",
                         "chrome_index_grid.toml",   # the chrome index stage's comparison grid (a measurement: scripts/chrome_index_grid.jl)
                         "sensor_characterisation.toml",   # the FX3's sensor characterisation (the refinements' sensor model record; D12)
                         "image_regions.toml",   # the capture sets' image-region steps (scripts/image_regions.jl; 2026-11-15)
                         "distance_maps.toml",   # the printed boards' edge distance maps, read through [runs] (scripts/distance_maps.jl; 2026-11-16)
                         "convergence.toml",   # the bias of Schöps's method per LM iteration (scripts/convergence.jl; 2026-11-15)
                         ],
                        # ONE chrome evaluation per chrome set, its models a list (2026-11-12; the 14 mm central_30cm's above)
                        ["chrome_evaluation_fx3_$s.toml" for s in ("fisheye_10cm_ag", "fisheye_2710_si", "tamron_17cm_si", "tamron_20261108_ag", "tamron_largeaperture", "zoom48_f10")]))
        # the studies (ideas tried, each with the decision it backed) and the supplement's own experiments: their groups (rc-0116)
        @test sort(readdir(projectdir("configs", "studies"))) ==
              ["radiometric_refinement_ablation_lens_coverage_scale.toml", "radiometric_refinement_smoothness_si30fe.toml", "sampling_strategy.toml",
               "sampling_strategy_fe10_count.toml", "sampling_strategy_fe10_kb16384.toml", "sampling_strategy_kb_count.toml", "sampling_strategy_smoke.toml",
               "warping_ablation.toml"]
        @test sort(readdir(projectdir("configs", "supplement"))) == ["timing.toml", "timing_smoke.toml"]
        # every case's [pipeline] table reads as PipelineOptions, and its paths stay inside data_root()
        for f in ("radiometric_refinement.toml", "radiometric_refinement_vignetting.toml")
            cfg = TOML.parsefile(projectdir("configs", "experiments", f))
            for (c, t) in get(cfg, "cases", Dict()), (tb, kv) in get(t, "pipeline", Dict()), (k, v) in kv
                v isa AbstractString || continue
                @test !isabspath(replace(v, r"^(lens|grid):" => ""))
            end
            for (id, t) in cfg["inputs"]
                @test t["tier"] in INPUT_TIERS && t["stage"] in DRIVER_STAGES && !isabspath(t["path"])
            end
        end
        # spot checks: the real-data budget of the runs of record (the 03.11 settings)
        rc = load_radiometric_refinement_config(experiment_config("radiometric_refinement"))
        o = RadiometricCamera.pipeline_options(rc["cases"]["si30_bc5_warped"]["pipeline"])
        @test o.io.channel == "G" && o.masks.samples_per_view == 8192 && o.lm.max_iter == 25 && o.rendering.N == 16 && o.rendering.hammersley
        # the LM cap of every refinement case (owner 2026-11-14): 50 for the fisheye cases (still improving at 25), 25 for the others
        # (the 14 mm flattened by ~15; was 48 / 50)
        isfisheye(n) = occursin(r"^(ag10fe_|si27fe_|si30fe_)|fisheye", n)   # si30fe: the fisheye 30 cm set (2026-11-16)
        for x in ("radiometric_refinement", "radiometric_refinement_vignetting"), (n, t) in load_radiometric_refinement_config(experiment_config(x))["cases"]
            @test RadiometricCamera.pipeline_options(t["pipeline"]).lm.max_iter == (isfisheye(n) ? 50 : 25)
        end
        # the non-central spline is out of the paper (owner, 2026-11-11): no case, no chrome evaluation, no table row and no input of it
        # among the configs of record (the package's non-central model and its golden case are removed too, 2026-11-14)
        @test !any(c -> endswith(c, "_nc") || !isempty(get(get(rc["cases"][c]["pipeline"], "frontend", Dict()), "noncentral", "")), keys(rc["cases"]))
        @test !any(id -> occursin("noncentral", id), keys(rc["inputs"]))
        @test !any(f -> occursin(r"_nc\.toml$", f), readdir(projectdir("configs", "experiments")))
        gf = TOML.parsefile(projectdir("configs", "experiments", "chrome_evaluation_paper.toml"))
        @test !any(k -> endswith(k, "_nc"), keys(gf["chrome"]))
        @test !any(r -> occursin("non-central", r["model"]) || endswith(r["chrome"], "_nc"), vcat((t["rows"] for t in values(gf["tables"]))...))
        rv = load_radiometric_refinement_config(projectdir("configs", "experiments", "radiometric_refinement_vignetting.toml"))
        @test sort(collect(keys(rv["cases"]))) == ["f4_$(l)_spline_warped" for l in lenses]
        @test RadiometricCamera.pipeline_options(rv["cases"]["f4_tamron_spline_warped"]["pipeline"]).camera.fnumber == 2.8
        # the board model is named in every case (owner ruling 2026-11-11), and every case's target is a print whose in-plane aspect and
        # skew are estimated (owner ruling 2026-11-13): [warping] inplane "aspect_skew" from the design as printed, not held, the
        # heights as each case states them. The former <base>_warped_print_error cases (radiometric_refinement_warping.toml) were
        # these cases verbatim once the in-plane block was on everywhere: deleted, no copy left.
        @test all(c -> endswith(c, "_warped"), keys(rc["cases"])) && all(c -> endswith(c, "_warped"), keys(rv["cases"]))
        @test !isfile(projectdir("configs", "experiments", "radiometric_refinement_warping.toml"))
        # The global models of the refinements of record (kind = "lens") the heights alone (owner decision 2026-11-16: "heights only for
        # the global models", the warping ablation's `heights` mode): inplane "none", no inplane_start stated.
        for f in ("radiometric_refinement.toml", "radiometric_refinement_vignetting.toml")
            for (n, t) in TOML.parsefile(projectdir("configs", "experiments", f))["cases"]
                w = t["pipeline"]["warping"]
                o = RadiometricCamera.pipeline_options(t["pipeline"])
                if f == "radiometric_refinement.toml" && t["kind"] == "lens"
                    @test w["inplane"] == "none" && !haskey(w, "inplane_start") && w["degree"] == 1
                    @test o.warping.inplane == "none"
                else
                    @test w["inplane"] == "aspect_skew" && w["inplane_start"] == [0.0, 0.0]
                    @test o.warping.inplane == "aspect_skew" && !("inplane" in o.fixed) && RadiometricCamera._surface_free(o, :inplane)
                end
            end
        end
        # the macro defocus left the project (2026-11-14): no driver, no config, no stage of the reproduction
        @test !isfile(projectdir("scripts", "macro_defocus.jl")) && !isfile(projectdir("configs", "experiments", "macro_defocus.toml"))
        @test !occursin("macro_defocus", read(projectdir("configs", "reproduce.toml"), String))
    end

    @testset "a config refuses what is not its own" begin
        dir = mktempdir()
        good = TOML.parsefile(experiment_config("radiometric_refinement"))
        function refused(edit!)
            cfg = deepcopy(good); edit!(cfg)
            path = joinpath(dir, "radiometric_refinement.toml")
            open(io -> TOML.print(io, cfg), path, "w")
            return try load_radiometric_refinement_config(path); nothing catch err; err end
        end
        raw = "raw_14mm_siemens_central30cm"; c = "si30_bc5_warped"
        @test refused(identity) === nothing
        e = refused(c -> c["evaluat"] = Dict());                              @test e isa ArgumentError && occursin("evaluat", e.msg)
        e = refused(c -> c["inputs"][raw]["tier"] = "rc_raw");                @test e isa ArgumentError && occursin("rc_raw", e.msg)
        e = refused(c -> c["inputs"][raw]["stage"] = "render");               @test e isa ArgumentError && occursin("render", e.msg)
        e = refused(c -> c["inputs"][raw]["path"] = "/abs/raw");              @test e isa ArgumentError && occursin("data_root", e.msg)
        e = refused(c -> c["inputs"][raw]["path"] = "a/../../raw");           @test e isa ArgumentError && occursin("data_root", e.msg)
        e = refused(c -> delete!(c["inputs"][raw], "provenance"));            @test e isa ArgumentError && occursin("provenance", e.msg)
        e = refused(x -> x["cases"][c]["pipeline"]["lm"]["fto"] = 1);         @test e isa ArgumentError && occursin("fto", e.msg) && occursin(c, e.msg)
        e = refused(x -> x["cases"][c]["pipeline"]["io"]["out"] = "o");       @test e isa ArgumentError && occursin("io.out", e.msg)
        e = refused(x -> x["cases"][c]["image"] = "x");                       @test e isa ArgumentError && occursin("image", e.msg)
        e = refused(x -> x["cases"][c]["images"] = "nope");                   @test e isa ArgumentError && occursin("prepare.nope", e.msg)
        e = refused(x -> x["experiment"]["name"] = "other");                  @test e isa ArgumentError && occursin("radiometric_refinement", e.msg)
        # the run keys: an input's provenance text never re-keys a run; a setting and the thread count do
        k(cfg) = refine_product(cfg, c).key
        c2 = deepcopy(good); c2["inputs"][raw]["provenance"] = "edited"
        @test k(c2) == k(good)
        c2 = deepcopy(good); c2["cases"][c]["pipeline"]["lm"]["max_iter"] = 13
        @test k(c2) != k(good) && start_product(c2, c).key == start_product(good, c).key   # a refine-only setting: the start is the same
        c2 = deepcopy(good); c2["prepare"]["fx3_14mm_siemens_central30cm"]["gain"] = 1.0   # its photographs' run moves, so does the case's
        @test k(c2) != k(good)
        # a run is keyed by the inputs its table reads, not by every input of the config
        other = "aprilgrid400_14mm_ag400_green_bc5_fit_json"                 # ag30_bc5_warped's start, which si30_bc5_warped does not read
        c2 = deepcopy(good); c2["inputs"][other]["path"] = "elsewhere/bc5_fit.json"
        @test k(c2) == k(good) && refine_product(c2, "ag30_bc5_warped").key != refine_product(good, "ag30_bc5_warped").key
        # its front end's run (a [runs] entry of frontends.toml) keys it by that run's key: another Schöps run re-keys it
        c2 = deepcopy(good); c2["runs"]["fx3_14mm_central30cm_schoeps_green"]["run"] = "schoeps.tamron17_siemens_w80"
        @test k(c2) != k(good)
        @test with_execution(Dict("a" => 1))["execution"]["threads"] == execution_threads()
        @test run_key(with_execution(Dict("a" => 1))) != run_key(merge(Dict{String,Any}("a" => 1), Dict("execution" => Dict("threads" => "64,1"))))
    end

    @testset "inputs: a Git LFS pointer is not data" begin
        dir = mktempdir()
        write(joinpath(dir, "real.json"), "{}")
        write(joinpath(dir, "stub.png"), "version https://git-lfs.github.com/spec/v1\noid sha256:00\nsize 1\n")
        @test is_lfs_pointer(joinpath(dir, "stub.png")) && !is_lfs_pointer(joinpath(dir, "real.json"))
        @test require_data(joinpath(dir, "real.json"); what = "x", provenance = "p") == joinpath(dir, "real.json")
        @test_throws ErrorException require_data(joinpath(dir, "stub.png"); what = "x", provenance = "p")
        @test_throws ErrorException require_data(dir; what = "a directory holding a pointer", provenance = "p")
        @test_throws ErrorException require_data(joinpath(dir, "absent"); what = "x", provenance = "p")
    end

    @testset "the RAW decoder in Julia (bin2, raw_green): Features' raw module" begin
        # a 4 × 4 RGGB mosaic, black 512, white 16383: photosite (1, 2) (a green) at the white level, gain 2 clips the bright ones
        raw = UInt16[512 16383 600 9000; 700 800 1200 512; 4000 5000 512 16000; 6000 7000 8000 9000]
        colors = UInt8[0 1 0 1; 3 2 3 2; 0 1 0 1; 3 2 3 2]            # the photosites' colours (Features derives them from the pattern)
        m = VgtkCore.RawMosaic(raw, UInt8[0 1; 3 2], (512.0, 512.0, 512.0, 512.0), 16383.0, (0, 0), (4, 4),
                                   (; datetime_original = nothing, subsec = nothing, exposure_s = nothing, fnumber = nothing, focal_mm = nothing,
                                      iso = nothing, model = nothing, daylight_wb = nothing, linear_max = nothing, decoder = "test", libraw_version = ""))
        lin(v) = (v - 512) / (16383 - 512)
        dir = mktempdir(); dirs = (linear = mkpath(joinpath(dir, "l")), srgb = mkpath(joinpath(dir, "s")))
        # mode "raw_green" (owner 2026-11-16): Features' bilinear green on the raw grid — a green photosite its own value (a saturated
        # one censored at 1), a red or blue one the mean of its green 4-neighbours — at full size, written × gain, clamped, 16 bit
        img = RadiometricCamera.raw_green_image(m; sensor_model = VgtkCore.SensorModel(), device = RadiometricCamera.rc_backend("cpu"))
        isgreen(i, j) = colors[i, j] in (0x01, 0x03)
        val(i, j) = raw[i, j] >= 16383 ? 1.0 : lin(raw[i, j])
        nb(i, j) = [(i + di, j + dj) for (di, dj) in ((-1, 0), (1, 0), (0, -1), (0, 1)) if 1 <= i + di <= 4 && 1 <= j + dj <= 4 && isgreen(i + di, j + dj)]
        want = [isgreen(i, j) ? val(i, j) : sum(c -> val(c...), nb(i, j)) / length(nb(i, j)) for i in 1:4, j in 1:4]
        @test size(img) == (4, 4) && eltype(img) == Float32 && all(isapprox.(img, want; atol = 1e-6))
        rec = write_raw_green(img, 2.0, "g.png", dirs)
        g = reinterpret.(UInt16, PNGFiles.load(joinpath(dirs.linear, "g.png")))
        @test g == round.(UInt16, clamp.(2.0 .* Float64.(img), 0, 1) .* 65535) && rec["size"] == [4, 4] && rec["no_green_pixels"] == 0
        @test rec["clipped_frac"] == count(>(1), 2.0 .* Float64.(img)) / 16
        # its frame: the raw grid itself (the identity onto the visible raw's COLMAP pixels), Core's raw_frame_map of the mosaic
        @test decoded_frame("raw_green").record == frame_record(m) && decoded_frame("raw_green").record["scale"] == 1.0
        @test decoded_frame("raw_green").map(SVector(3.25, 7.5)) == SVector(3.25, 7.5)
        @test occursin("raw_green", thrown_message(() -> raw_mode("raw_green")))                 # no Core LinearRaw mode
        @test occursin("retired", thrown_message(() -> check_raw_mode("green", false, "x")))     # the green cells: retired for geometry
        @test occursin("sRGB", thrown_message(() -> check_raw_mode("raw_green", true, "x")))
        rec = write_decoded(VgtkCore.LinearRaw(m; mode = :bin2, gain = 1.0), "b.png", dirs)
        b = PNGFiles.load(joinpath(dirs.linear, "b.png"))
        @test size(b) == (2, 2) && reinterpret(UInt16, b[1, 1].r) == round(UInt16, lin(512) * 65535) &&
              reinterpret(UInt16, b[1, 1].g) == round(UInt16, (lin(16383) + lin(700)) / 2 * 65535) && reinterpret(UInt16, b[1, 1].b) == round(UInt16, lin(800) * 65535)
        @test isfile(joinpath(dirs.srgb, "b.png")) && rec["size"] == [2, 2]
        @test_throws ArgumentError raw_mode("fullres")                     # fullres (a demosaic) is retired
        e = try check_raw_mode("fullres", false, "x"); nothing catch err; err end; @test e isa ArgumentError && occursin("fullres", e.msg) && occursin("raw_green", e.msg)
        # a photosites set is read, not decoded: gain and srgb refused by name; a decoding states them
        @test check_prepare_table(Dict("raw" => "r", "mode" => "photosites"), "x") === nothing
        e = try check_prepare_table(Dict("raw" => "r", "mode" => "photosites", "gain" => "auto"), "x"); nothing catch err; err end
        @test e isa ArgumentError && occursin("gain", e.msg)
        @test_throws ArgumentError check_prepare_table(Dict("raw" => "r", "mode" => "raw_green"), "x")
        @test check_prepare_table(Dict("raw" => "r", "mode" => "raw_green", "gain" => "auto", "srgb" => "off"), "x") === nothing
        @test_throws ArgumentError raw_mode("photosites")                  # not a decoding
    end

    @testset "the keys of a start and a refinement: each keyed by only what it reads" begin
        db = load_radiometric_refinement_config(experiment_config("radiometric_refinement"))
        c = "si30_spline_warped"                                   # a Schöps-started spline: the case's own start state
        ki(cfg) = start_product(cfg, c).key; kc(cfg) = refine_product(cfg, c).key
        c2 = deepcopy(db); c2["cases"][c]["pipeline"]["lm"]["max_iter"] = 3      # a refinement setting: the start is the same product
        @test ki(c2) == ki(db) && kc(c2) != kc(db)
        for k in RadiometricCamera.REFINE_ONLY_KEYS                              # every refine-only key, RadiometricCamera's declaration
            tb, key = split(k, '.')
            c2 = deepcopy(db); t = get!(c2["cases"][c]["pipeline"], tb, Dict{String, Any}())
            v = getfield(getfield(RadiometricCamera.pipeline_options(db["cases"][c]["pipeline"]), Symbol(tb)), Symbol(key))
            t[key] = v isa Bool ? !v : v + 7                                     # another value than the case's
            @test ki(c2) == ki(db) && kc(c2) != kc(db)
        end
        c2 = deepcopy(db); c2["cases"][c]["pipeline"]["masks"]["gradient_bias"] = 0.5     # a start setting: both move
        @test ki(c2) != ki(db) && kc(c2) != kc(db)
        # the start record's format keys every start: a start of an earlier record is another product, regenerated
        @test start_product(db, c).record["start_record"] == RadiometricCamera.RECORD_SCHEMAS.refinement_start
        @test start_product(db, "si30_bc5_warped").record["start_record"] == RadiometricCamera.RECORD_SCHEMAS.refinement_start
        # the start's kind is RadiometricCamera's one decision (`start_is_grid`, taken by prepare_start): the drivers test no io.init
        # prefix of their own (RC 0.9.0 review, finding 6)
        @test !occursin(r"startswith\([^)]*\"lens:\"", read(projectdir("src", "drivers.jl"), String))
        # start_product reads the start's settings once (review finding 9: one deepcopy and check per plan, not three)
        let src = read(projectdir("src", "drivers.jl"), String), body = src[findfirst("function start_product(", src).start:end]
            body = body[1:findfirst("\nend\n", body).stop]
            @test count("start_settings(t)", body) == 1
        end
        @test !haskey(start_settings(db["cases"][c]), "start_from_init") && !haskey(start_settings(db["cases"][c]), "kind")
        @test !haskey(get(start_settings(db["cases"][c])["pipeline"], "lm", Dict()), "max_iter")
        @test haskey(refine_settings(db["cases"][c])["pipeline"], "lm") && keys(refine_settings(db["cases"][c])["pipeline"]["lm"]) ⊆ Set(["max_iter", "max_time"])
        @test RadiometricCamera.start_pipeline_table(db["cases"][c]["pipeline"]) == start_settings(db["cases"][c])["pipeline"] &&
              last(RadiometricCamera.split_refine_only(db["cases"][c]["pipeline"])) == refine_settings(db["cases"][c])["pipeline"]
        # the start's key reads what RC's start reads (code review 0.11.0, finding 5): no output key (io.out, lm.log) keys a start
        let t = merge(deepcopy(db["cases"][c]), Dict("pipeline" => merge(deepcopy(db["cases"][c]["pipeline"]),
                                                     Dict("io" => merge(get(db["cases"][c]["pipeline"], "io", Dict()), Dict("out" => "/x")), "lm" => Dict("log" => "lm.tsv", "max_iter" => 3)))))
            @test !haskey(start_settings(t)["pipeline"]["io"], "out") && !haskey(get(start_settings(t)["pipeline"], "lm", Dict()), "log")
            @test RadiometricCamera.pipeline_options(start_settings(t)["pipeline"]) == RadiometricCamera.start_options(RadiometricCamera.pipeline_options(t["pipeline"]))
        end
        @test resolve_pipeline(db, start_settings(db["cases"][c]); produce = false)["io"] isa AbstractDict      # the start table resolves without an init's grid
        @test all(id -> db["inputs"][id]["stage"] != "evaluate", table_inputs(db, resolve_pipeline(db, start_settings(db["cases"][c]); produce = false)))
        @test require_stage_inputs(db, "start") isa Vector{String}             # typed, for produce's inputs
    end

    # The ported evaluation against the STORED runs of record (fixtures/evaluation/, kilobytes cut from them by make_fixtures.jl:
    # two views of the Siemens ground truth, the fine learned V), at the agreement the full comparison showed
    # (≤ 1.4e-12 px; the former night-stars rotation ≤ 2e-12 px: its residual's Jacobian is now differenced in pixels, not in the former
    # chart; the scale searches of the variants ≤ 1.5e-4 px: Core's Kabsch rounding amplified by the search).
    @testset "the ported evaluation against the stored runs" begin
        F = projectdir("test", "fixtures", "evaluation"); T = mktempdir()
        # read strictly: the fixtures and this item's own outputs hold no NaN, and their integers (degree, step, size, rows, cols)
        # are used as integers, which allownan would read as Float64s
        rj(p) = VgtkIO.read_record(p)
        gt = joinpath(F, "gt", "poses.json"); truth = "lens:" * joinpath(F, "truth_lens.json")
        # reprojection_vectors: a Kannala–Brandt lens (Core's project) and a grid lens (Core's project of its generic model), both
        # through Core's projection (RadiometricCamera's _project_from)
        for (tag, kind) in (("kb", "lens"), ("grid", "grid"))
            reprojection_vectors(gt, "$kind:" * joinpath(F, "$(tag)_lens.json"), joinpath(F, "$(tag)_poses.json"), joinpath(T, "$tag.json"))
            O = rj(joinpath(T, "$tag.json")); e = rj(joinpath(F, "expected_reprojection.json"))[tag]
            at = Dict((String(O["view"][i]), Int(O["id"][i])) => i for i in eachindex(O["dx"]))
            i = [at[(String(e["view"][j]), Int(e["id"][j]))] for j in eachindex(e["dx"])]
            @test length(i) == length(O["dx"]) == 34
            @test maximum(hypot.(O["dx"][i] .- e["dx"], O["dy"][i] .- e["dy"])) <= 1.4e-12
        end
        # the renders' own frame (colmap px): the truth lens maps the true board points under the true poses to the stored true pixels
        tv = truepose_vectors(gt, truth, joinpath(T, "tv.json"); gauge = false)
        O = rj(joinpath(T, "tv.json"))
        @test tv.n == 34 && maximum(hypot.(O["dx"], O["dy"])) <= 1.4e-12
        @test truepose_scores(gt, truth, truth).raw_median == 0
        # truepose_scores' raw error = the lens's direction at the true pixel against the TRUE pose's direction
        sk = truepose_scores(gt, truth, "lens:" * joinpath(F, "kb_lens.json"))
        J = rj(gt)
        f = Float64(VgtkCore.model(RadiometricCamera.load_parametric_lens(joinpath(F, "gt", "camera.json"))).intrinsics.K[1, 1])
        kb = RadiometricCamera.load_lens("lens:" * joinpath(F, "kb_lens.json"))
        X = Dict(Int(c["id"]) => SVector{3}(Float64.(c["X_m"])...) for c in J["board"]["corners"])
        tr = VgtkIO.load(VgtkCore.Trajectory, joinpath(F, "gt", "trajectory.json"))
        pose = Dict(tr.images[k] => VgtkCore.ustrip(VgtkCore.m, tr.poses[k]) for k in keys(tr.poses))          # world → camera
        e = [f * VgtkCore.angle_between(normalize(pose[v["name"]](X[Int(cr["id"])])),
                 normalize(VgtkCore.backproject(kb, SVector{2}(Float64.(cr["px_colmap"])...))))
             for v in J["views"] for cr in v["corners"]]
        @test abs(median(e) - sk.raw_median) <= 1.4e-12 && sk.n == 34
    end

    @testset "the data policy: a driver writes only under results_root() and its assets under plotsdir()" begin
        data, results = mktempdir(), mktempdir()
        # a data root holding exactly the input the Thorlabs driver's detect stage reads: one small photograph (a JPEG of bright
        # dots on a dark ground), and a config whose image list is that photograph. (Before, the panorama driver's lens
        # stage was this test's run; that stage went with the binned rooftop chain.)
        cfg = TOML.parsefile(projectdir("test", "fixtures", "thorlabs", "chrome_evaluation.toml"))   # the HX400V set's config: JPEG photographs
        photos = joinpath(data, cfg["inputs"]["photos"]["path"]); mkpath(photos)
        img = [Gray{N0f8}(0.1 + 0.8 * (hypot(mod(x, 16) - 8, mod(y, 16) - 8) < 3)) for y in 1:160, x in 1:192]   # SFOP's 3 octaves
        write(joinpath(photos, "tiny.jpg"), jpeg_encode(img))
        cfg["detect"]["photographs"] = ["tiny.jpg"]; cfg["detect"]["excluded"] = Dict{String,Any}()
        cfg["index"]["excluded"] = Dict{String,Any}()                    # its exclusions name photographs of the detect stage
        cdir = mktempdir(); config = joinpath(cdir, "chrome_evaluation.toml"); open(io -> TOML.print(io, cfg), config, "w")
        data_before = tree_state(data)
        project_before = read(`git -C $(projectdir()) status --porcelain --untracked-files=all -- .`, String)
        plots_before = isdir(plotsdir()) ? Set(readdir(plotsdir())) : Set{String}()
        ok, out = run_driver_script("chrome_evaluation", "--stage=detect"; data, results, config)
        ok || @info out
        @test ok
        @test tree_state(data) == data_before                                                  # nothing written under data_root()
        made = collect(keys(tree_state(results)))
        @test all(startswith("chrome_evaluation/detect/"), made) && any(endswith("summary.tsv"), made) && any(endswith("_provenance.toml"), made)
        @test (isdir(plotsdir()) ? Set(readdir(plotsdir())) : Set{String}()) == plots_before    # no asset from this stage
        @test read(`git -C $(projectdir()) status --porcelain --untracked-files=all -- .`, String) == project_before
        runs = joinpath(results, "chrome_evaluation", "detect")                            # driver_product: <results>/<experiment>/<stage>/<key>
        run = joinpath(runs, only(readdir(runs)))
        @test TOML.parsefile(joinpath(run, "_provenance.toml"))["device"] isa String           # the device the run computed on is in the provenance
        eff = TOML.parsefile(joinpath(run, "_config.toml"))
        @test !haskey(eff, "execution")                                                   # detect is thread-invariant: the thread count keys it not
    end
end

@testitem "experiment drivers: a raw input's selection (frames | exclude)" default_imports = false begin
    # A config states the subset of a published folder it uses beside the folder's path (INPUT_SELECTION_KEYS, 2026-11-10: the
    # Tamron 17 cm Siemens folder's three stray 14 mm photographs, every second frame of the Tamron panorama): the loader checks the
    # keys, the selection keys the runs, and `selected_raw_dir` hands the decoders a folder of links to the selected files.
    using Test
    using DrWatson
    using TOML: TOML
    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitIO as VgtkIO
    include(srcdir("experiments.jl"))
    include(srcdir("drivers.jl"))
    @test INPUT_SELECTION_KEYS == ("frames", "exclude")
    base = Dict{String,Any}("tier" => "rc_raw_photos", "path" => "paper-data/x/y", "stage" => "prepare", "provenance" => "p")
    @test input_record(base) == Dict("tier" => "rc_raw_photos", "path" => "paper-data/x/y")                       # no selection: as before
    @test input_record(merge(base, Dict("frames" => ["a.ARW", "b.ARW"])))["frames"] == ["a.ARW", "b.ARW"]           # a selection keys the run
    @test input_record(merge(base, Dict("exclude" => ["c.ARW"])))["exclude"] == ["c.ARW"]
    mktempdir() do dir
        # the loader: one of the two keys at most, each a non-empty array of distinct file names
        write_cfg(extra) = (p = joinpath(dir, "radiometric_refinement_t.toml");
            open(p, "w") do io
                println(io, "[experiment]\nname = \"radiometric_refinement\"\n[inputs.raw]\ntier = \"rc_raw_photos\"\npath = \"paper-data/x/y\"\nstage = \"prepare\"\nprovenance = \"p\"")
                println(io, extra)
            end; p)
        ok(extra) = load_driver_config(write_cfg(extra), "radiometric_refinement"; tables = ("experiment", "inputs"))
        @test ok("frames = [\"a.ARW\", \"b.ARW\"]")["inputs"]["raw"]["frames"] == ["a.ARW", "b.ARW"]
        @test ok("exclude = [\"c.ARW\"]")["inputs"]["raw"]["exclude"] == ["c.ARW"]
        @test_throws ArgumentError ok("frames = [\"a.ARW\"]\nexclude = [\"c.ARW\"]")
        @test_throws ArgumentError ok("frames = []")
        @test_throws ArgumentError ok("frames = [\"a.ARW\", \"a.ARW\"]")
        @test_throws ArgumentError ok("exclude = [\"sub/c.ARW\"]")
        @test_throws ArgumentError ok("every = 2")
        # selected_raw_dir on a folder of five raw files, under a data root of its own
        data = mkpath(joinpath(dir, "data", "paper-data", "x", "y")); run = mkpath(joinpath(dir, "results", "prepare-k"))
        for n in ("DSC00001.ARW", "DSC00002.ARW", "DSC00003.ARW", "DSC00004.ARW", "DSC00005.ARW", "notes.txt"); write(joinpath(data, n), n); end
        withenv("VGTK_DATA_ROOT" => joinpath(dir, "data")) do
            cfg(extra) = load_driver_config(write_cfg(extra), "radiometric_refinement"; tables = ("experiment", "inputs"))
            @test selected_raw_dir(cfg(""), "raw", run) == joinpath(dir, "data", "paper-data", "x", "y")           # no selection: the folder itself
            f = selected_raw_dir(cfg("frames = [\"DSC00002.ARW\", \"DSC00004.ARW\"]"), "raw", run)
            @test f == joinpath(run, "raw_frames") && sort(readdir(f)) == ["DSC00002.ARW", "DSC00004.ARW"] && all(islink, readdir(f; join = true))
            @test read(joinpath(f, "DSC00002.ARW"), String) == "DSC00002.ARW"
            @test selected_raw_dir(cfg("frames = [\"DSC00002.ARW\", \"DSC00004.ARW\"]"), "raw", run) == f                      # idempotent
            run2 = mkpath(joinpath(dir, "results", "prepare-k2"))
            e = selected_raw_dir(cfg("exclude = [\"DSC00005.ARW\"]"), "raw", run2)
            @test sort(readdir(e)) == ["DSC00001.ARW", "DSC00002.ARW", "DSC00003.ARW", "DSC00004.ARW"]                     # the raw files less one (notes.txt is no raw file)
            @test_throws ErrorException selected_raw_dir(cfg("frames = [\"DSC00009.ARW\"]"), "raw", mkpath(joinpath(dir, "results", "k3")))   # a name not in the folder
            @test_throws ErrorException selected_raw_dir(cfg("exclude = [\"DSC00009.ARW\"]"), "raw", mkpath(joinpath(dir, "results", "k4")))
            @test_throws ErrorException selected_raw_dir(cfg("exclude = [\"DSC00001.ARW\", \"DSC00002.ARW\", \"DSC00003.ARW\", \"DSC00004.ARW\", \"DSC00005.ARW\"]"), "raw", mkpath(joinpath(dir, "results", "k5")))
        end
    end
    # the configs of record state their selections: the Tamron 17 cm Siemens folder less its three stray 14 mm photographs (owner, 2026-11-14)
    rc = TOML.parsefile(projectdir("configs", "experiments", "radiometric_refinement.toml"))
    @test rc["inputs"]["raw_tamron_siemens_17cm"]["exclude"] == ["DSC08194.ARW", "DSC08195.ARW", "DSC08196.ARW"]
    # every raw-photo input of every config of record is a folder of the published set (paper-data/…)
    for f in readdir(projectdir("configs", "experiments"); join = true)
        for (id, t) in get(TOML.parsefile(f), "inputs", Dict{String, Any}())   # (an experiment of overrides: the base's inputs)
            t["tier"] == "rc_raw_photos" || continue
            @test startswith(t["path"], "paper-data/")
        end
    end
end

# decode_raw decodes the photographs concurrently, one per thread (each load_raw its own LibRaw handle): its files are the serial
# decode's, byte for byte — each photograph through the same steps one after another (load_raw_mosaic, LinearRaw, write_decoded).
@testitem "experiment drivers: the threaded RAW decode is the serial one, byte for byte" default_imports = false tags = [:data] skip = !(
        haskey(ENV, "VGTK_DATA_ROOT") || (p = joinpath(dirname(Base.active_project()), "data", "exp_raw"); islink(p) || isdir(p))) begin
    using Test
    using DrWatson
    include(srcdir("experiments.jl"))
    include(srcdir("drivers.jl"))
    src = joinpath(data_root(), "paper-data", "defocus", "zoom-lens", "siemens-stars")
    if !isdir(src)
        @test_skip "the zoom set is not under data_root()"
    else
        sel = mktempdir()
        foreach(f -> symlink(f, joinpath(sel, basename(f))), first(raw_files(src), 2 * max(Threads.nthreads(), 2)))
        out = decode_raw(sel, mktempdir(), "t"; gain = 4.0, mode = "bin2", srgb = false)
        ref = (linear = mktempdir(), srgb = nothing)
        for f in raw_files(sel)
            name = first(splitext(basename(f))) * ".png"
            write_decoded(VgtkCore.LinearRaw(VgtkFeatures.load_raw_mosaic(f); mode = raw_mode("bin2"), gain = 4.0), name, ref)
            @test read(joinpath(out, name)) == read(joinpath(ref.linear, name))
        end
        @test sort(filter(endswith(".png"), readdir(out))) == sort(readdir(ref.linear))
    end
end

@testitem "cli_shard: n array tasks run every item once" default_imports = false begin
    using Test
    using DrWatson
    include(srcdir("experiments.jl"))
    include(srcdir("drivers.jl"))
    items = ["a", "b", "c", "d", "e"]
    @test cli_shard(String[], items) == items
    @test cli_shard(["--shard=0/2"], items) == ["a", "c", "e"] && cli_shard(["--shard=1/2"], items) == ["b", "d"]
    @test sort(reduce(vcat, [cli_shard(["--shard=$k/3"], items) for k in 0:2])) == items
    @test cli_shard(["--shard=6/7"], items) == String[]                      # a shard beyond the items: none
    @test_throws ArgumentError cli_shard(["--shard=2/2"], items)
    @test_throws ArgumentError cli_shard(["--shard=1"], items)
end

@testitem "the geometry image: one for every detector — the front ends' decode is the chrome detect's image (owner 2026-11-16)" default_imports = false begin
    using Test
    using DrWatson
    using TOML: TOML
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: thrown_message
    import VisualGeometryToolkitFeatures as VgtkFeatures
    include(srcdir("experiments.jl"))
    include(srcdir("drivers.jl"))
    include(srcdir("boards.jl"))
    include(srcdir("parametric_starts.jl"))
    include(srcdir("evaluation.jl"))
    include(srcdir("frontends.jl"))
    include(srcdir("radiometric_refinement.jl")); include(srcdir("chrome_evaluation.jl"))
    # a 6 × 8 RGGB mosaic with a defect and a saturated green, under a sensor model with that defect: the decode's image (RadiometricCamera's
    # raw_green_image, what decode_raw writes for mode raw_green) is Features' bilinear green, the plane its detect_sfop(m::RawMosaic) and
    # seeded detect_foerstner(m, seeds) detect on — the chrome detect's image — value for value
    raw = UInt16[512 + 37 * ((3i + 5j) % 11) * 40 for i in 1:6, j in 1:8]; raw[2, 3] = 16383
    mo = VgtkCore.RawMosaic(raw, UInt8[0 1; 3 2], (512.0, 512.0, 512.0, 512.0), 16383.0, (0, 0), (6, 8),
                           (; datetime_original = nothing, subsec = nothing, exposure_s = nothing, fnumber = nothing, focal_mm = nothing,
                              iso = nothing, model = nothing, daylight_wb = nothing, linear_max = nothing, decoder = "test", libraw_version = ""))
    cpu = RadiometricCamera.rc_backend("cpu")
    for model in (VgtkCore.SensorModel(),)
        img = RadiometricCamera.raw_green_image(mo; sensor_model = model, device = cpu)
        @test isequal(img, Array(VgtkFeatures._bilinear_green(Float32, cpu, mo, model).value))
    end
    feat = read(joinpath(pkgdir(VgtkFeatures), "src", "raw_green.jl"), String)
    @test count("_bilinear_green(Float32, device, m, sensor_model)", feat) == 1        # detect_sfop(m::RawMosaic)'s image: the same call
    drv = read(srcdir("drivers.jl"), String); che = read(srcdir("chrome_evaluation.jl"), String); fe = read(srcdir("frontends.jl"), String)
    @test occursin("Base.invokelatest(RadiometricCamera.raw_green_image, Float32, m; sensor_model = models[i], device)", drv)   # the decode
    @test occursin("raw_sensor_models(files, characterisation;", drv) && occursin("raw_sensor_models(files, spec;", che)   # one model resolution
    @test occursin("mode = t[\"image\"]", fe) && occursin("characterisation = config_input(cfg, t[\"sensor_characterisation\"])", fe)
    @test RadiometricCamera.GEOMETRY_IMAGES == ("raw_green", "luma") && DETECT_IMAGES === RadiometricCamera.GEOMETRY_IMAGES   # one vocabulary
    @test "raw_green" in RAW_MODES && !("green" in RAW_MODES)

    # the source lock: no geometry stage names the green cells (Features' :green_cells, Core's LinearRaw{:green_cells}) — the RC
    # package's src and the paper's src and scripts; the only mentions left are the refusals of the retired name, as a string
    roots = [joinpath(pkgdir(RadiometricCamera), "src"), srcdir(), projectdir("scripts")]
    files = [joinpath(d, f) for r in roots for (d, _, fs) in walkdir(r) for f in fs if endswith(f, ".jl")]
    offenders = [f for f in files if occursin(r":green_cells|LinearRaw\{:green_cells\}|mode = \"green\"", read(f, String))]
    @test isempty(offenders)
    # bin2 (the cell grid) is no geometry image: its one consumer, the per-colour panorama vignetting (a documented deviation of 0.11.5),
    # left the project with the panorama chains (rc-0116) — no config of record decodes it
    cfgs = [joinpath(d, f) for (d, _, fs) in walkdir(projectdir("configs")) for f in fs if endswith(f, ".toml")]
    bin2 = sort([basename(f) for f in cfgs if occursin(r"(?m)^mode\s*=\s*\"bin2\"", read(f, String))])
    @test isempty(bin2)

    # the key is the config's statement: every case of the config of record states it, read with no default — a missing one is an
    # error naming the key, and another image (the decoded images' luma, the retired green cells) is refused
    path = projectdir("configs", "experiments", "frontends.toml")
    cfg = TOML.parsefile(path)
    @test all(t -> t["image"] == "raw_green" && t["sensor_characterisation"] == "fx3_sensor_characterisation", values(cfg["cases"]))
    @test frontend_decode_product(cfg, "zoom48_f10_siemens").record["frame_map"] == decoded_frame("raw_green").record
    edited(f) = (c = deepcopy(TOML.parsefile(path)); f(c["cases"]["zoom48_f10_siemens"]); p = joinpath(mktempdir(), "frontends.toml");
                 open(io -> TOML.print(io, c), p, "w"); p)
    @test occursin("image", thrown_message(() -> load_frontends_config(edited(t -> delete!(t, "image")))))
    @test occursin("retired", thrown_message(() -> load_frontends_config(edited(t -> t["image"] = "green_cells"))))
    @test occursin("raw_green", thrown_message(() -> load_frontends_config(edited(t -> t["image"] = "luma"))))
    @test occursin("sensor_characterisation", thrown_message(() -> load_frontends_config(edited(t -> delete!(t, "sensor_characterisation")))))
    @test frontend_decode_product(cfg, "zoom48_f10_siemens").record["cases"]["zoom48_f10_siemens"]["image"] == "raw_green"   # keyed by its image
end
