# The sampling-strategy experiment (scripts/sampling_strategy.jl, configs/studies/sampling_strategy.toml): the variants are
# expand_overrides of the cases of record (nothing else changed), the refinement configs load with the owner's loader, the lattice
# guard refuses a subdivided or oversized start, the chrome and texture configs are the configs of record with the variants as their
# models / upstream, and the figure and table writers render the record. The driver is run in a module of its own (--plan: nothing run).

@testsetup module SamplingStrategy
    using DrWatson
    using VisualGeometryToolkitEval.Experiments: run_isolated
    const CONFIG = projectdir("configs", "studies", "sampling_strategy.toml")
    "The driver's module on `config` (its plan only)."
    driver(config = CONFIG; args = String[]) = redirect_stdout(devnull) do
        run_isolated(projectdir("scripts", "sampling_strategy.jl"); args = ["--plan", "--config=$config", args...])
    end
    export driver, CONFIG
end

@testitem "sampling strategy: the config of record, its variants as overrides of the cases of record" default_imports = false setup = [SamplingStrategy] begin
    using Test
    using DrWatson
    m = driver()
    SS = m.SS
    @test SS["variants"] == ["uniform", "old037", "ecdf_a1", "ecdf_a2", "addon"] && SS["reference"] == "uniform" && SS["headline"] == ["ecdf_a1", "ecdf_a2"]
    @test occursin("provisional", SS["label"])
    for c in SS["cases"]
        r = Base.invokelatest(m.refinement_config, c)
        rec = m.inherit_kinds!(m.TOML.parsefile(projectdir(m.CFG["cases"][c]["refinement"])), m.CFG["cases"][c]["refinement"])   # the cases as the driver reads them ([kinds] inherited)
        base = rec["cases"][m.CFG["cases"][c]["case"]]
        for v in SS["variants"]
            t = r.cfg["cases"][r.names[v]]
            ov = Base.invokelatest(m.variant_overrides, c, v)
            for (k, x) in ov
                tb, key = split(k, '.')
                @test t["pipeline"][tb][key] == x                                    # the variant's keys set
            end
            # nothing else changed: the table minus the overridden keys is the case of record's
            strip(tb) = Dict(k => (k == "pipeline" ? Dict(a => (a == "masks" ? filter(((q, _),) -> !("masks.$q" in keys(ov)), b) : b) for (a, b) in tb[k]) : tb[k]) for k in keys(tb))
            @test strip(t) == strip(base)
        end
        @test r.cfg["cases"][r.names["addon"]]["pipeline"]["masks"]["addon_samples"] == base["pipeline"]["masks"]["samples_per_view"] ÷ 2
        @test r.cfg["cases"][r.names["old037"]]["pipeline"]["masks"]["edge_fraction"] == 0.37
        mktempdir() do d                                                           # the owner's loader accepts the file
            f = joinpath(d, "radiometric_refinement_ss_$c.toml")
            open(io -> m.TOML.print(io, r.cfg; sorted = true), f, "w")
            rc = Base.invokelatest(m.load_radiometric_refinement_config, f)
            @test all(n -> haskey(rc["cases"], n), values(r.names))
            o = Base.invokelatest(m.pipeline_options, rc["cases"][r.names["ecdf_a2"]]["pipeline"]).masks
            @test o.gradient_bias == 2.0 && o.edge_fraction == 0.0 && o.addon_samples == 0 && o.edge_detector == m.RadiometricCamera.GaussianDerivativeEdges(1.5)
        end
    end
    @test length(unique(Base.invokelatest(m.refinement_config, "fisheye_f4").names[v] for v in SS["variants"])) == 5   # five distinct runs
end

@testitem "sampling strategy: unknown or inapplicable keys are errors; a smoke config is its base with the LM capped" default_imports = false setup = [SamplingStrategy] begin
    using Test
    using DrWatson
    m = driver()
    load = p -> Base.invokelatest(m.load_sampling_strategy_config, p)
    mktempdir() do d
        bad(extra) = (f = joinpath(d, "sampling_strategy_bad.toml"); write(f, read(CONFIG, String) * "\n" * extra); f)
        @test_throws ArgumentError load(bad("[cases.texcal_14mm.oops]\nx = 1\n"))
        @test_throws ArgumentError load(bad("[variants.extra]\n\"masks.gradient_bias\" = 3.0\n"))           # not in the variants list
        f = joinpath(d, "sampling_strategy_lat.toml"); write(f, replace(read(CONFIG, String), "max_nodes = 900 }" => "max_nodes = 0 }"))
        @test_throws ArgumentError load(f)
        f = joinpath(d, "sampling_strategy_dome.toml"); write(f, replace(read(CONFIG, String), r"^dome_tolerance = .*$"m => ""))
        e = try load(f); nothing catch err; err end                                  # the dome criterion is the config's: no code default
        @test e isa ArgumentError && occursin("dome_tolerance", e.msg)
    end
    @test load(CONFIG)["sampling_strategy"]["dome_tolerance"] == 0.12
    sm = load(projectdir("configs", "studies", "sampling_strategy_smoke.toml"))
    @test sm["sampling_strategy"]["lm_max_iter"] == 2 && sm["variants"] == load(CONFIG)["variants"]
    ms = driver(projectdir("configs", "studies", "sampling_strategy_smoke.toml"))
    @test Base.invokelatest(ms.variant_overrides, "fisheye_f4", "uniform")["lm.max_iter"] == 2
    @test Base.invokelatest(ms.refinement_config, "fisheye_f4").names["uniform"] != Base.invokelatest(m.refinement_config, "fisheye_f4").names["uniform"]   # never shadows production
end

@testitem "sampling strategy: the lattice guard refuses a subdivided or oversized start" default_imports = false setup = [SamplingStrategy] begin
    using Test
    using DrWatson
    import VisualGeometryToolkitIO as VgtkIO
    m = driver()
    start(g, sub) = (d = mktempdir(); VgtkIO.write_record(joinpath(d, "start.json"), Dict("schema" => m.RadiometricCamera.RECORD_SCHEMAS.refinement_start,
                                                         "resample" => Dict("grid_size" => g, "subdivision" => sub, "lattice_spacing" => [163.7, 164.5]))); d)
    ok = Base.invokelatest(m.lattice_guard, "fisheye_f4", "uniform", start([20, 20], [1, 1]))
    @test ok.nodes == 400 && occursin("grid: 400 nodes (20 × 20), m = 1 × 1", ok.line)
    @test_throws "REFUSED (the slow lattice)" Base.invokelatest(m.lattice_guard, "fisheye_f4", "uniform", start([65, 65], [4, 4]))
    @test_throws "exceeds" Base.invokelatest(m.lattice_guard, "texcal_14mm", "uniform", start([40, 30], [1, 1]))
end

@testitem "sampling strategy: the chrome and texture configs are the configs of record with the variants" default_imports = false setup = [SamplingStrategy] begin
    using Test
    using DrWatson
    m = driver()
    Core.eval(m, :(produced_arms(c, as) = as))                                      # every arm as if refined
    mktempdir() do d
        for c in m.SS["cases"]                                                     # the refinement configs as the configs stage writes them
            open(io -> m.TOML.print(io, Base.invokelatest(m.refinement_config, c).cfg; sorted = true), joinpath(d, "radiometric_refinement_ss_$c.toml"), "w")
        end
        Core.eval(m, :(refinement_path(c) = joinpath($d, "radiometric_refinement_ss_$c.toml")))
        for c in m.SS["cases"], k in m.EVAL_KINDS
            cf = Base.invokelatest(m.chrome_config, c, k)
            rec = m.chrome_record(c)
            @test cf["evaluate"]["holdout"] == rec["evaluate"]["holdout"] == "on" && cf["evaluate"]["bootstrap"] == rec["evaluate"]["bootstrap"]
            @test [x["label"] for x in cf["evaluate"]["models"]][1:5] == m.SS["variants"]
            @test all(x -> x["reference"] == "uniform", cf["evaluate"]["comparisons"][1:4])
            f = joinpath(d, "chrome_evaluation_ss_$(c)_$k.toml"); open(io -> m.TOML.print(io, cf; sorted = true), f, "w")
            Base.invokelatest(m.load_chrome_evaluation_config, f)                    # the owner's loader accepts it
        end
        t = Base.invokelatest(m.texture_config, "texcal_14mm", "ecdf_a1")
        @test t["upstream"]["case"] == Base.invokelatest(m.refinement_config, "texcal_14mm").names["ecdf_a1"] && !haskey(t, "folds") && !haskey(t, "compare")
    end
end

@testitem "sampling strategy: the figure and the table render the record" default_imports = false setup = [SamplingStrategy] begin
    using Test
    using DrWatson
    m = driver()
    pr(ref, d) = Dict("model" => "x", "reference" => ref, "variant" => "motion_heldout", "region" => "all", "stat" => "median_px",
                      "difference" => string(d), "lo" => string(d - 0.01), "hi" => string(d + 0.01), "photos" => "150", "dots" => "36000")
    rows = [Dict{String, Any}("case" => c, "arm" => v, "kind" => "variant", "chrome_variants" => Dict{String, Any}("paired" => [pr("uniform", 0.01 * i)]),
                              "focus" => Dict{String, Any}("dome" => true, "max_dev" => 0.1), "refine" => Dict{String, Any}("s_per_iteration" => 88.0),
                              "texture" => c == "texcal_14mm" ? Dict{String, Any}("minus_reference" => Dict("learned" => Dict("rms_block_mean_diff" => -0.002, "rms_block_diff_ci95" => [-0.004, 0.0]))) : nothing)
            for c in m.SS["cases"] for (i, v) in enumerate(m.SS["variants"])]
    mktempdir() do d
        f = Base.invokelatest(m.write_figure, joinpath(d, "fig.tex"), rows); s = read(f, String)
        @test occursin("groupplot", s) && occursin("rc ours", s) && occursin("ecdfa1", s) && occursin("provisional", s)
        t = Base.invokelatest(m.write_table, joinpath(d, "tab.tex"), rows); s = read(t, String)
        @test count("\\\\", s) >= 10 && occursin("ecdf\\_a2", s) && occursin("[-0.004, +0.000]", s)
    end
end
