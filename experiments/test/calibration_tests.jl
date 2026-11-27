# The calibration cases' configs of record (configs/calibration/) and their driver, scripts/calibrate.jl: the configs load and
# refuse what they must, they are the package's pipeline-check configs, the driver plans its four stages, and it writes only under
# results_root(). The pipeline runs themselves are the package's golden gate (experiments/test/pipeline/run_pipeline_checks.sh, which runs
# `scripts/calibrate.jl --stage=refine --force` on each case), not this suite: they take minutes and need the case data.
@testitem "calibration cases" default_imports = false begin
    using Test
    using DrWatson
    using TOML: TOML
    include(srcdir("experiments.jl"))
    include(srcdir("drivers.jl"))
    include(srcdir("calibration.jl"))

    const CASES = ("siemens_coarse_full",)
    calibration_config(c) = projectdir("configs", "calibration", "$c.toml")

    # The thread count the driver runs under in these tests: it keys the run (src/drivers.jl `with_execution`).
    const TEST_THREADS = "2,1"
    # A run of the driver in a fresh process with both trees bound to scratch directories; (succeeded, output).
    function run_calibrate(args...; data, results)
        cmd = `$(Base.julia_cmd()) --threads=$TEST_THREADS --project=$(projectdir()) $(projectdir("scripts", "calibrate.jl")) $(collect(String, args))`
        out = IOBuffer()
        ok = success(pipeline(addenv(cmd, "VGTK_DATA_ROOT" => data, "VGTK_RESULTS_ROOT" => results);
                              stdout = out, stderr = out))
        return ok, String(take!(out))
    end

    # Every file under `dir`, relative, with its modification time.
    tree_state(dir) = Dict(relpath(joinpath(r, f), dir) => mtime(joinpath(r, f)) for (r, _, fs) in walkdir(dir) for f in fs)

    @testset "the configs of record load through pipeline_options" begin
        for c in CASES
            cfg, run, settings = load_calibration_config(calibration_config(c))
            @test run isa RadiometricCamera.RunOptions && settings isa RadiometricCamera.CalibrationSettings
            @test (run, settings) == pipeline_options(cfg["pipeline"])
            @test cfg["experiment"]["case"] == c
            @test isempty(run.io.out)
        end
        @test sort(readdir(projectdir("configs", "calibration"))) == sort(["$c.toml" for c in CASES])
    end

    @testset "one config per golden case: no mirror in the package (experiments/test/pipeline/configs is gone)" begin
        @test !isdir(joinpath(pkgdir(RadiometricCamera), "..", "experiments", "test", "pipeline", "configs"))
    end

    @testset "the input keys are every path option" begin
        # (that every string option is an input, an output or no path is RadiometricCamera's own test, pipeline_options_tests.jl)
        @test PIPELINE_PATH_KEYS === PIPELINE_INPUT_KEYS                   # the drivers' and the calibration's: the package's one list
        cfg, _, _ = load_calibration_config(calibration_config("siemens_coarse_full"))
        @test pipeline_inputs(cfg) == ["io.images" => "geometric_accuracy/detector_bias/data/fisheye_full/gt/images_linear",
                                       "frontend.schoeps" => "geometric_accuracy/detector_bias/data/fisheye_full/schoeps",
                                       "frontend.texture" => "geometric_accuracy/data/target/siemens_1m/board/texture.png",
                                       "frontend.texture_features" => "geometric_accuracy/data/target/siemens_1m/board/texture_features.bin"]   # edge.sdf: run:siemens_1m_distance_map, a [runs] file (no input under data_root())
        @test table_runs(cfg["pipeline"]) == ["siemens_1m_distance_map"] && cfg["pipeline"]["edge"]["sdf"] == "run:siemens_1m_distance_map"
        # an empty `camera.sensor_characterization` names none: no input path (RadiometricCamera's `names_input_path`)
        cfg, _, _ = load_calibration_config(calibration_config("siemens_coarse_full"))
        c2 = deepcopy(cfg); get!(c2["pipeline"], "camera", Dict{String, Any}())["sensor_characterization"] = ""
        @test pipeline_inputs(c2) == pipeline_inputs(cfg)
        c2["pipeline"]["camera"]["sensor_characterization"] = "sensor/fx3.json"
        @test ("camera.sensor_characterization" => "sensor/fx3.json") in pipeline_inputs(c2)
    end

    @testset "a config of record refuses what is not its own" begin
        dir = mktempdir()
        good = TOML.parsefile(calibration_config("siemens_coarse_full"))
        function refused(edit!, name = "siemens_coarse_full")
            cfg = deepcopy(good); edit!(cfg)
            path = joinpath(dir, "$name.toml")
            open(io -> TOML.print(io, cfg), path, "w")
            return try load_calibration_config(path); nothing catch err; err end
        end
        @test refused(identity) === nothing
        e = refused(c -> c["experiment"]["datset"] = "x");  @test e isa ArgumentError && occursin("datset", e.msg)
        e = refused(c -> c["pipeline"]["lm"]["fto"] = 1);   @test e isa ArgumentError && occursin("fto", e.msg) && occursin("max_iter", e.msg)
        e = refused(c -> c["renderr"] = Dict("a" => 1));    @test e isa ArgumentError && occursin("renderr", e.msg)
        e = refused(c -> c["pipeline"]["io"]["out"] = "o"); @test e isa ArgumentError && occursin("io.out", e.msg)
        e = refused(c -> c["pipeline"]["io"]["save_init"] = "s.json"); @test e isa ArgumentError && occursin("io.save_init", e.msg)
        e = refused(c -> c["pipeline"]["lm"] = merge(c["pipeline"]["lm"], Dict("log" => "lm.tsv"))); @test e isa ArgumentError && occursin("lm.log", e.msg)
        e = refused(c -> delete!(c["render"], "title"));    @test e isa ArgumentError && occursin("title", e.msg)
        e = refused(identity, "other");                     @test e isa ArgumentError && occursin("file's name", e.msg)
        # [render] does not key the run; every other table does
        cfg = deepcopy(good); cfg["render"]["significant_digits"] = 2
        @test run_key(cfg) == run_key(good)
        cfg = deepcopy(good); cfg["pipeline"]["masks"]["samples_per_view"] = 3000
        @test run_key(cfg) != run_key(good)
    end

    @testset "--plan lists the four stages ($c)" for c in CASES
        ok, out = run_calibrate("--config=$(calibration_config(c))", "--plan"; data = mktempdir(), results = mktempdir())
        ok || @info out
        @test ok
        for st in ("prepare", "start", "refine", "evaluate", "figures")
            @test occursin(Regex("^\\s+$st\\s+(check|\\d+/\\d+ produced)", "m"), out)
        end
        @test occursin("plan only", out)
    end

    @testset "the driver writes only under results_root()" begin
        data, results = mktempdir(), mktempdir()
        cfg, _, _ = load_calibration_config(calibration_config("siemens_coarse_full"))
        key = calibration_products(cfg; threads = TEST_THREADS).refine.key          # the driver's products (src/calibration.jl) at the test's threads
        run_dir = joinpath(results, "calibrate", "refine", key)                 # driver_product: <results>/<experiment>/<stage>/<key>
        mkpath(run_dir)
        golden = joinpath(pkgdir(RadiometricCamera), "..", "experiments", "test", "pipeline", "golden", "siemens_coarse_full")
        for f in readdir(golden); cp(joinpath(golden, f), joinpath(run_dir, f)); end      # the refine stage's output, as a run writes it
        project_before = read(`git -C $(projectdir()) status --porcelain --untracked-files=all -- .`, String)
        ok, out = run_calibrate("--config=$(calibration_config("siemens_coarse_full"))", "--stage=evaluate,figures"; data, results)
        ok || @info out
        @test ok
        @test isempty(readdir(data))
        # the refinement's files, and the evaluate and figures products with their records (each keyed by the one before it)
        ev = only(readdir(joinpath(results, "calibrate", "evaluate"))); fg = only(readdir(joinpath(results, "calibrate", "figures")))
        @test sort(collect(keys(tree_state(results)))) ==
              sort([[joinpath("calibrate", "refine", key, f) for f in readdir(golden)];
                    [joinpath("calibrate", "evaluate", ev, f) for f in ("metrics.toml", "_config.toml", "_provenance.toml")];
                    [joinpath("calibrate", "figures", fg, f) for f in ("summary.tex", "_config.toml", "_provenance.toml")]])
        @test read(`git -C $(projectdir()) status --porcelain --untracked-files=all -- .`, String) == project_before
        @test TOML.parsefile(joinpath(results, "calibrate", "evaluate", ev, "metrics.toml")) == RadiometricCamera.load_pipeline_metrics(run_dir)
        summary = read(joinpath(results, "calibrate", "figures", fg, "summary.tex"), String)
        @test occursin("\\begin{tabular}", summary) && occursin("cost.after\\_refinement.rel\\_rms", summary)
    end
end

@testitem "the golden case's config of record loads; its record reads back (owner D15: the pipeline golden is the experiments' gate)" default_imports = false begin
    using Test, TOML, RadiometricCamera
    const RC = RadiometricCamera
    import VisualGeometryToolkitCore as VgtkCore
    root = normpath(pkgdir(RadiometricCamera), "..")   # the repository: the package and the experiments side by side
    _, o = RC.pipeline_options(TOML.parsefile(joinpath(root, "experiments", "configs", "calibration", "siemens_coarse_full.toml"))["pipeline"])
    @test o isa RC.CalibrationSettings && o.warping.inplane == "aspect_skew" && !("inplane" in o.fixed) && o.fixed == ["aperture"]
    @test RC.load_run_parameters(joinpath(root, "experiments", "test", "pipeline", "golden", "siemens_coarse_full", "parameters.json"))["frontend"] == "schoeps"
    # the paper's parametric cases build their problem without a printed-edge block (the edge is never a parameter)
    rr = TOML.parsefile(joinpath(root, "experiments", "configs", "experiments", "radiometric_refinement.toml"))
    for c in ("si30_ocv14_warped", "si30_kb_warped", "ag30_ocv14_warped", "ag30_kb_warped")
        _, oc = RC.pipeline_options(rr["cases"][c]["pipeline"])
        @test !isempty(oc.edge.sdf) && !(:edge in RC.refinement_structure(oc, RC.PerViewRoute()).kw.fixed)
    end
    @test isempty(VgtkCore.config_tag_violations((RC.FRONTEND_TAGS, RC.VIGNETTING_CONFIG_TAGS, RC.BACKEND_TAGS),
                                                    [joinpath(root, "experiments", "configs", "calibration")]))
end

@testitem "the configs of record: every switch is \"on\" | \"off\" (RC 0.11.4); a TOML boolean is data or another package's key" default_imports = false begin
    using Test, TOML, RadiometricCamera
    const RC = RadiometricCamera
    root = normpath(pkgdir(RadiometricCamera), "..")   # the repository: the package and the experiments side by side
    # the booleans that stay booleans, each with its reason (the survey of 2026-11-15, ledgered): a fact, or a key another package reads
    data = [r"artifacts\.[^.]+\.required",                                     # a fact; VisualGeometryToolkitEval DataArtifacts' catalog
            r"artifacts\.[^.]+\.retired",                                      # a fact of the catalog (src/data.jl entries_of_record): a superseded tier
            r"stage\[\d+\]\.(default|gpu|results_branch)",                     # VisualGeometryToolkitEval Reproduce's stage graph
            r"stage\[\d+\]\.phases\[\d+\]\.(gpu|per_item)",                   # (the same schema: a stage's phases, the warping ablation's)
            r"results_branch\.assets\[\d+\]\.submitted_only",                  # (the same schema)
            r"lenses\.[^.]+\.circle",                                           # a fact: the lens's image does not fill the sensor
            r"require\[\d+\]\.optional"]                                       # a fact of the run of record's required experiment (absent is no error)
    dirs = [joinpath(root, "experiments", "configs"), joinpath(root, "experiments", "test", "fixtures"), joinpath(root, "RadiometricCamera", "configs")]
    @test isempty(RC.boolean_switch_violations(dirs; data))
    # the lint catches a boolean switch where a config of record would carry one
    dir = mktempdir(); write(joinpath(dir, "x.toml"), "[evaluate]\nholdout = true\n[pipeline.io]\nlinearize = false\n")
    @test length(RC.boolean_switch_violations([dir]; data)) == 2
    # every [pipeline] table of the configs of record loads through the one reader (its switches "on" | "off")
    pipes(x, p, out) = x isa AbstractDict ? (foreach(((k, v),) -> k == "pipeline" && v isa AbstractDict ? push!(out, "$p.pipeline" => v) : pipes(v, "$p.$k", out), x); out) : out
    n = 0
    for (r, _, fs) in walkdir(joinpath(root, "experiments", "configs")), f in fs
        endswith(f, ".toml") || continue
        for (p, t) in pipes(TOML.parsefile(joinpath(r, f)), f, Pair{String, Any}[])
            n += 1
            @test RC.pipeline_options(t) isa Tuple{RC.RunOptions, RC.CalibrationSettings}
        end
    end
    @test n > 30                                                                  # 38 at 2026-11-15
end
