# The reproduce index and the generated assets after the drivers (src/assets.jl, around VisualGeometryToolkitEval's Reproduce
# verify/deploy/gather/uncited and Experiments run_isolated): every asset the paper reads is a row of REPRODUCE.md naming its
# generator, config of record and data tiers; every figure config of record loads and names a submitted figure; deploy never
# replaces a submitted figure without sign-off; scripts/numbers.jl gathers the fragments, deploys numbers.tex under the same
# sign-off and every macro it defines is cited by the prose; the drivers declare their GPU stages; regenerate_paper.jl --plan runs.
@testitem "reproduce index and generated assets" default_imports = false begin
    using Test
    using DrWatson
    using TOML: TOML
    using VisualGeometryToolkitEval: Reproduce
    using VisualGeometryToolkitEval.Reproduce: verify_assets, deploy_assets, assets_deployed, uncited_macros, VERIFY_REPORT
    include(srcdir("experiments.jl"))
    include(srcdir("drivers.jl"))
    include(srcdir("assets.jl"))

    const PAPERS = ["../paper/main.tex"]
    const FIGURE_FIXTURES = projectdir("test", "fixtures", "figure_configs")   # the figure-config machinery's fixtures

    # Macros of ../paper/latex/numbers.tex the prose does not cite (yet), each with its reason: `name without backslash => reason`.
    # The texture-recovery macros (scripts/texture_recovery.jl's figures stage on the 14 mm small-defocus set, deployed 2026-11-10)
    # wait for the defocus section's prose, which the author writes; the list is the macros' names so that a macro that vanishes
    # from numbers.tex or is cited later is noticed here.
    const TEXTURE_UNCITED = [
        "textureFourteenSmallBlockMm", "textureFourteenSmallCells", "textureFourteenSmallFitPhotographs",
        "textureFourteenSmallLearnedTestCentreRelRms", "textureFourteenSmallLearnedTestEdgeRelRms", "textureFourteenSmallLearnedTestRelRms",
        "textureFourteenSmallLearnedTextureRmsBlock", "textureFourteenSmallLearnedTextureRmsTexel", "textureFourteenSmallLearnedWeight",
        "textureFourteenSmallPairedLearnedLower", "textureFourteenSmallPairedTestDiff", "textureFourteenSmallPairedTextureDiff",
        "textureFourteenSmallPairedTextureDiffCiHigh", "textureFourteenSmallPairedTextureDiffCiLow", "textureFourteenSmallPhotographs",
        "textureFourteenSmallPinholeTestCentreRelRms", "textureFourteenSmallPinholeTestEdgeRelRms", "textureFourteenSmallPinholeTestRelRms",
        "textureFourteenSmallPinholeTextureRmsBlock", "textureFourteenSmallPinholeTextureRmsTexel", "textureFourteenSmallPinholeWeight",
        "textureFourteenSmallTestPhotographs",
    ]
    const UNCITED_ALLOW = Dict{String,String}(n => "the defocus (texture) section is being written; its prose cites it (2026-11-10)" for n in TEXTURE_UNCITED)

    @testset "the reproduce index, configs of record required" begin
        v = Reproduce.reproduce_index_violations(projectdir(); papers = PAPERS, require_config = true, search_dirs = PAPER_SEARCH_DIRS)
        # the manuscript's own findings, deferred with their reasons (test/fixtures/paper_allow.toml; paper_project_tests.jl reads it too)
        allow = TOML.parsefile(projectdir("test", "fixtures", "paper_allow.toml"))["allow"]
        allowed(x) = any(a -> string(x.kind) == a["rule"] && startswith(x.location, a["file"] * ":") && occursin(a["match"], x.subject * " — " * x.message), allow)
        v = filter(!allowed, v)
        foreach(x -> @info(sprint(show, x)), v)
        @test isempty(v)
    end

    @testset "every index row names its generator, config and data tiers" begin
        tiers = r"`rc_(raw_photos|synthetic|baselines|calibration)`"
        for e in Reproduce.read_reproduce_index(projectdir("REPRODUCE.md"))
            e.kind === :generated || continue
            @test !isempty(e.generators) && !isempty(e.configs)
            @test occursin(tiers, e.data)
        end
    end

    @testset "every generator declares what it writes and reads" begin
        gens = unique(vcat((e.generators for e in Reproduce.read_reproduce_index(projectdir("REPRODUCE.md")))...))
        for g in gens
            @test !isempty(Reproduce.declared_outputs(projectdir(g)))
            @test !isempty(Reproduce.declared_inputs(projectdir(g)))
        end
    end

    @testset "the figure configs of record" begin
        @test isempty(figure_configs())                                 # the paper's figures and tables are external or numbers.tex
        # the machinery on its fixtures (the configs that deployed the focus surfaces, the texture figure and two tables until v0.11.6)
        paths = figure_configs(FIGURE_FIXTURES)
        @test length(paths) == 4
        for p in paths
            cfg = load_figure_config(p)
            @test replace_submitted(cfg) === false          # nothing is signed off
            @test cfg["experiment"]["name"] == first(splitext(figure_asset(cfg)))
            @test asset_kind(cfg)[1] === (endswith(figure_asset(cfg), ".tex") ? :table : :figure)
        end
        @test_throws ArgumentError asset_kind(Dict("render" => Dict("asset" => "plots.x")))
        # an unknown key and a missing one are refused
        cfg = TOML.parsefile(first(paths))
        dir = mktempdir()
        bad = joinpath(dir, "bad.toml")
        open(io -> TOML.print(io, merge(cfg, Dict("deploy" => Dict("replace_submitted" => "off", "force" => true)))), bad, "w")
        @test_throws ArgumentError load_figure_config(bad)
        open(io -> TOML.print(io, merge(cfg, Dict("deploy" => Dict{String,Any}()))), bad, "w")
        @test_throws ArgumentError load_figure_config(bad)
        open(io -> TOML.print(io, merge(cfg, Dict("deploy" => Dict("replace_submitted" => "yes")))), bad, "w")
        @test_throws ArgumentError load_figure_config(bad)
        # the scripts' own configs: one list of drivers, the numbers config's, which the regeneration reads
        numbers = load_config_tables(projectdir("configs", "figures", "numbers.toml"), NUMBERS_CONFIG_KEYS)
        regen = load_config_tables(projectdir("configs", "figures", "regenerate_paper.toml"), REGENERATE_CONFIG_KEYS)
        @test driver_configs(regen) == numbers["runs"]["experiments"]
        @test !haskey(regen["runs"], "experiments")
        @test replace_submitted(numbers) === false                  # the submitted numbers are not signed off either
        for c in numbers["runs"]["experiments"]
            @test isfile(driver_script(projectdir(c)))
        end
    end

    @testset "verify and deploy: a submitted figure is replaced only with sign-off" begin
        figdir = mktempdir()
        outdir = mktempdir()
        name = "focus-surfaces.png"
        write(joinpath(figdir, name), rand(StableRNG(1), UInt8, 4096))   # a submitted figure (its bytes only matter)
        submitted = read(joinpath(figdir, name))
        quietly(f) = redirect_stdout(f, devnull)

        # identical: verified identical, deployed as a no-op
        cp(joinpath(figdir, name), joinpath(outdir, name))
        @test only(quietly(() -> verify_assets(outdir, figdir, [name]))).identical
        @test isempty(quietly(() -> deploy_assets(outdir, figdir, [name])))
        @test assets_deployed(outdir, figdir, [name])

        # differing: refused without sign-off (the submitted bytes untouched), deployed with it
        write(joinpath(outdir, name), vcat(submitted, UInt8[0x00]))
        err = try
            quietly(() -> deploy_assets(outdir, figdir, [name])); nothing
        catch e
            e
        end
        @test err isa ErrorException && occursin(joinpath(outdir, VERIFY_REPORT), err.msg) && occursin("replace_submitted", err.msg)
        @test read(joinpath(figdir, name)) == submitted
        @test quietly(() -> deploy_assets(outdir, figdir, [name]; replace_submitted = true)) == [name]
        @test read(joinpath(figdir, name)) == read(joinpath(outdir, name))
    end

    @testset "deploy of every figure: one report with every row, each figure signed off by its own config" begin
        figdir, outdir = mktempdir(), mktempdir()
        every = [(cfg = load_figure_config(p), path = p) for p in figure_configs(FIGURE_FIXTURES)]
        figures = [f for f in every if asset_kind(f.cfg)[1] === :figure]
        assets = [figure_asset(f.cfg) for f in figures]
        foreach(((i, a),) -> write(joinpath(figdir, a), rand(StableRNG(i), UInt8, 4096)), enumerate(assets))   # the submitted figures
        for a in assets
            cp(joinpath(figdir, a), joinpath(outdir, a))
        end
        a1, a2 = assets[1], assets[2]
        write(joinpath(outdir, a1), vcat(read(joinpath(figdir, a1)), UInt8[0]))   # two renders differ from the submitted figures
        write(joinpath(outdir, a2), vcat(read(joinpath(figdir, a2)), UInt8[0]))
        quietly(f) = redirect_stdout(f, devnull)
        report = joinpath(outdir, VERIFY_REPORT)
        rows() = [first(split(l, '\t')) for l in readlines(report)[2:end]]
        err = try
            quietly(() -> deploy_figures(outdir, figdir, figures)); nothing
        catch e
            e
        end
        @test err isa ErrorException && occursin(report, err.msg)
        @test occursin(relpath(figures[1].path, projectdir()), err.msg) && occursin(relpath(figures[2].path, projectdir()), err.msg)
        @test rows() == assets                                                   # every figure's row, not the last call's
        # the first signed off in its config: still refused for the second, nothing copied, the report still whole
        signed1 = [i == 1 ? (cfg = merge(f.cfg, Dict("deploy" => Dict("replace_submitted" => "on"))), path = f.path) : f
                   for (i, f) in enumerate(figures)]
        @test_throws ErrorException quietly(() -> deploy_figures(outdir, figdir, signed1))
        @test read(joinpath(figdir, a1)) != read(joinpath(outdir, a1)) && rows() == assets
        # both signed off: both deployed
        signed12 = [i <= 2 ? (cfg = merge(f.cfg, Dict("deploy" => Dict("replace_submitted" => "on"))), path = f.path) : f
                    for (i, f) in enumerate(figures)]
        @test quietly(() -> deploy_figures(outdir, figdir, signed12)) == [a1, a2]
        @test assets_deployed(outdir, figdir, assets) && rows() == assets
        # the tables: deployed into their own directory from their own render directory, with a report of their own; refused without it
        tables = [f for f in every if asset_kind(f.cfg)[1] === :table]
        @test !isempty(tables)
        latexdir, latex_outdir = mktempdir(), mktempdir()
        for f in tables
            a = figure_asset(f.cfg); write(joinpath(latexdir, a), "% the submitted $a\n"); cp(joinpath(latexdir, a), joinpath(latex_outdir, a))
        end
        @test_throws ArgumentError quietly(() -> deploy_figures(outdir, figdir, tables))
        @test isempty(quietly(() -> deploy_figures(outdir, figdir, tables; latex_outdir, latexdir)))   # identical: nothing copied
        @test [first(split(l, '\t')) for l in readlines(joinpath(latex_outdir, VERIFY_REPORT))[2:end]] == [figure_asset(f.cfg) for f in tables]
    end

    @testset "scripts/numbers.jl end to end: fragments → a keyed run → numbers.tex, signed off, every macro cited" begin
        tmp = mktempdir()
        results = mkpath(joinpath(tmp, "results"))
        experiments = load_config_tables(projectdir("configs", "figures", "numbers.toml"), NUMBERS_CONFIG_KEYS)["runs"]["experiments"]
        # one fake fragment per driver, in a figures run of its own, as the drivers' figures stages write them
        frags = [joinpath(results, "figures-$i", "numbers.tex") for i in eachindex(experiments)]
        # each fragment written as a driver writes it: Eval's number macros (`PaperNumbers`, `add_number!`, `write_numbers`), its value typeset text
        fragment(path, name, value) = (mkpath(dirname(path));
            Reproduce.write_numbers(path, Reproduce.add_number!(Reproduce.PaperNumbers(; driver = "test_driver", config = "test", run_key = "0", commit = "0"), name, value)))
        for (i, f) in enumerate(frags)
            fragment(f, "fakeNumber$(('A':'Z')[i])", "$(i).5")
        end
        # a paper laid out as this one: main.tex in paper/src, built from paper/, reading src/latex/numbers and its prose
        latexdir = mkpath(joinpath(tmp, "paper", "src", "latex"))
        mkpath(joinpath(tmp, "paper", "src", "sec"))
        main = joinpath(tmp, "paper", "src", "main.tex")
        write(main, "\\input{src/latex/numbers}\n\\input{src/sec/results}\n")
        write(joinpath(tmp, "paper", "src", "sec", "results.tex"),                                          # every driver's macro cited but the last's
              "errors of " * join(["\\fakeNumber$(c)" for c in ('A':'Z')[1:(length(experiments) - 1)]], " and ") * " px\n")
        write(joinpath(latexdir, "numbers.tex"), "% generated by scripts/regenerate_paper.jl — do not edit\n")   # the submitted stub
        config(signed) = (p = joinpath(tmp, "numbers_$(signed).toml");
                          cfg = TOML.parsefile(projectdir("configs", "figures", "numbers.toml"));
                          cfg["deploy"]["replace_submitted"] = signed ? "on" : "off";   # a switch (RC 0.11.4)
                          open(io -> TOML.print(io, cfg), p, "w"); p)
        gather(signed) = withenv("VGTK_RESULTS_ROOT" => results) do
            m = redirect_stdout(() -> run_isolated(scriptsdir("numbers.jl"); args = ["--plan", "--config=$(config(signed))"]), devnull)
            redirect_stdout(() -> Base.invokelatest(Base.invokelatest(getfield, m, :gather), frags; latexdir), devnull)
        end

        # unsigned: the gathered numbers differ from the submitted stub, so the deploy is refused and the paper's copy untouched
        stub = read(joinpath(latexdir, "numbers.tex"))
        @test_throws ErrorException gather(false)
        @test read(joinpath(latexdir, "numbers.tex")) == stub
        # signed off: deployed
        out = gather(true)
        @test out == joinpath(latexdir, "numbers.tex")
        lines = readlines(out)
        @test filter(l -> !startswith(l, "%"), lines) == ["\\newcommand{\\fakeNumber$(c)}{$(i).5}" for (i, c) in enumerate(('A':'Z')[eachindex(frags)])]
        @test !occursin(tmp, read(out, String))                                         # the header names no machine path
        # the keyed run of record holds the gathered file and its records
        nruns = joinpath(results, "numbers", "numbers")                                # driver_product: <results>/<experiment>/<stage>/<key>
        runs() = readdir(nruns)
        @test length(runs()) == 1 && all(f -> isfile(joinpath(nruns, only(runs()), f)), ("numbers.tex", "_config.toml", "_provenance.toml"))
        # the output goes through the citation check: the one macro the prose does not cite is found
        @test uncited_macros(out, [main]; root = tmp, search_dirs = ["paper"]) == ["fakeNumber$(('A':'Z')[length(experiments)])"]
        # a fragment that changes a number: refused again without sign-off
        fragment(frags[1], "fakeNumberA", "9.5")
        @test_throws ErrorException gather(false)
        @test occursin("{1.5}", read(out, String))
        @test length(runs()) == 2                       # the fragment's numbers key the run: rewritten in place, a new run
        @test any(r -> occursin("{9.5}", read(joinpath(nruns, r, "numbers.tex"), String)), runs())
    end

    @testset "the paper's numbers.tex: every macro cited by the prose, or allowed with a reason" begin
        # the paper's tables carry the defocus and convergence-count numbers themselves (tab:defocus, Tab. 2; T. Harder's
        # table generators, REPRODUCE.md), so its prose cites none of the defocus and nonConverged macros numbers.tex still defines
        for n in first.(Reproduce.tex_macro_values(paper_numbers_path()))
            (startswith(n, "defocus") || startswith(n, "nonConverged")) &&
                (UNCITED_ALLOW[n] = "the paper's tables carry this number (tab:defocus, tab:refinement-chrome); its prose cites no macro of them")
        end
        for n in ("timingMaxS", "timingMinS", "timingWarmupS")    # Tab. A6 (runtime-table.tex, scripts/timing.jl) carries them
            UNCITED_ALLOW[n] = "the runtime table (tab:supp-runtime) carries this number; the prose of Sec. A.14 cites the mean and std"
        end
        v = uncited_macros(paper_numbers_path(), projectdir.(PAPERS); root = projectdir(), search_dirs = PAPER_SEARCH_DIRS,
                           allow = UNCITED_ALLOW)
        foreach(x -> @info("uncited number macro: " * x), v)
        @test isempty(v)
    end

    @testset "the GPU gate: declared stages, reused without --gpu, refused without the memory" begin
        ran = Ref(false)
        withenv("VGTK_RESULTS_ROOT" => mktempdir(), "VGTK_SCRATCH" => "1") do
            pending = Product("d", "refine", Dict("n" => 1); output = "out.txt")
            done = Product("d", "refine", Dict("n" => 2); output = "out.txt")
            produce(d -> (write(joinpath(d, "out.txt"), "x"); nothing), done; driver = "d", config_source = "c")
            s = Stage(:refine, "refine"; items = () -> [pending, done], product = identity, run = p -> (ran[] = true; nothing))
            h(gpu, gpu_stages) = host_stage(s; driver = "d", gpu_stages, gpu)
            @test_throws r"--gpu" h(false, [:refine]).run(pending)
            h(false, [:refine]).run(done)                                         # produced: reused without --gpu
            @test !ran[]
            h(true, [:refine]).run(pending)
            @test ran[]
            ran[] = false
            h(false, [:start]).run(pending)
            @test ran[]
            @test host_stage(Stage(:c, "c"; check = () -> (ran[] = false; nothing)); driver = "d", gpu_stages = [:c], gpu = false).items === nothing
        end

        # a stage is a GPU stage by the compute backend of the [pipeline] tables it runs, not by its name
        cpu = Dict("pipeline" => Dict("compute" => Dict("backend" => "cpu")))
        gpu = Dict("pipeline" => Dict("compute" => Dict("backend" => "gpu")))
        @test runs_on_gpu(gpu) && !runs_on_gpu(cpu) && runs_on_gpu(Dict("pipeline" => Dict()))   # the package's default: gpu
        @test gpu_stages(:start => [cpu], :refine => [cpu, gpu], :evaluate => []) == [:refine]
        expected = Dict("texture_recovery" => [:refine, :folds],         # the page solves (the split of record and its rotations) on one GPU
                        "chrome_evaluation_paper" => [:sets],             # the paper config's sets: each set's detect on the GPU ([detect] device = "gpu")
                        "focus_surfaces" => Symbol[])                  # the focus surfaces read the refinements' records
        for c in load_config_tables(projectdir("configs", "figures", "numbers.toml"), NUMBERS_CONFIG_KEYS)["runs"]["experiments"]
            name = TOML.parsefile(projectdir(c))["experiment"]["name"]
            m = withenv("VGTK_DATA_ROOT" => mktempdir(), "VGTK_RESULTS_ROOT" => mktempdir()) do
                redirect_stdout(() -> run_isolated(driver_script(projectdir(c)); args = ["--plan", "--config=$(projectdir(c))"]), devnull)
            end
            @test driver_gpu_stages(m, name) == expected[name]
        end
        @test_throws ErrorException driver_gpu_stages(Module(), "bare")

        # --gpu needs [gpu] min_free_gib on GPU 0
        @test require_gpu_memory(20; free_mib = 21 * 1024) == 21 * 1024
        @test_throws r"8\.0 GiB free" require_gpu_memory(20; free_mib = 8 * 1024)
    end

    @testset "regenerate_paper.jl --plan" begin
        cmd = `$(Base.julia_cmd()) --startup-file=no --project=$(projectdir()) $(projectdir("scripts", "regenerate_paper.jl")) --plan`
        out = IOBuffer()
        ok = success(pipeline(addenv(cmd, "VGTK_DATA_ROOT" => mktempdir(), "VGTK_RESULTS_ROOT" => mktempdir()); stdout = out, stderr = out))
        txt = String(take!(out))
        ok || @info txt
        @test ok
        for st in ("chrome_evaluation_paper", "focus_surfaces", "texture_recovery_14mm_small", "numbers", "verify", "deploy")
            @test occursin(Regex("^\\s+$st\\s+(check|\\d+/\\d+ produced)", "m"), txt)
        end
        @test occursin("plan only", txt)
    end
end

# The one command (configs/reproduce.toml on VisualGeometryToolkitEval's runner, scripts/reproduce/): the stage graph of record
# and its --only subsets; the dry run on the B200 server's site (Slurm) and on the desktop (local), whose commands are the launchers of
# scripts/slurm/ and nothing else; the missing-input report as data; the launchers' case selections are the graph's.
@testitem "the one command: the stage graph of record on a Slurm site and the desktop" default_imports = false begin
    using Test
    using DrWatson
    using TOML: TOML
    using VisualGeometryToolkitEval.Reproduce: load_stage_graph, select_stages, missing_inputs, reproduce_main, load_site, site_root

    graph = projectdir("configs", "reproduce.toml")
    g = load_stage_graph(graph)
    root = dirname(projectdir())                                   # the repository root: where the jobs are submitted from
    @test g.name == "experiments" && normpath(g.workdir) == normpath(root * "/") && g.julia_project == "experiments"
    # the front ends run before the refinements' prepare: its detect reads the front-end runs (Schöps and the fits) that
    # frontends_init produces (configs/reproduce.toml, found by a fresh --only=texture_recovery run, 2026-11-15)
    # the image regions (src/image_region.jl) are measured before every stage that reads them (2026-11-15)
    # the distance maps (src/distance_maps.jl) are produced once before every stage whose cases read them (2026-11-16)
    paper = ["setup", "distance_maps", "image_regions", "frontends_prepare", "frontends_detect", "frontends_init",
             "radiometric_refinement_prepare", "radiometric_refinement", "texture_recovery", "radiometric_refinement_vignetting_prepare",
             "radiometric_refinement_vignetting", "chrome_evaluation", "chrome_evaluation_score", "regenerate_paper",
             "results_branch"]
    env0 = filter(p -> !(first(p) in ("VGTK_DATA_ROOT", "VGTK_RESULTS_ROOT", "CASES", "CHROME_CASES", "VIG_CASES",
                                      "TEXTURE_SETS", "SETS", "CONVERGENCE_CASES", "WARP_MODES", "VGTK_DATA_ROOT_UNPUBLISHED", "VGTK_SITE", "VGTK_ARTIFACT_STORE")), Dict(ENV))   # hermetic: no machine tier root

    @testset "--only: a subset with its prerequisites, or alone" begin
        @test select_stages(g) == paper                            # no flags: every paper stage, nothing else
        @test select_stages(g; only = ["texture_recovery"]) == ["setup", "distance_maps", "image_regions", "frontends_prepare", "frontends_detect", "frontends_init", "radiometric_refinement_prepare", "radiometric_refinement", "texture_recovery"]
        @test select_stages(g; only = ["texture_recovery"], deps = false) == ["texture_recovery"]
        @test select_stages(g; only = ["regenerate_paper"]) == paper[1:end-1]
        @test select_stages(g; only = ["golden"]) == ["setup", "distance_maps", "golden"]
        @test "paper_pdf" in select_stages(g; only = ["paper_pdf"]) && !("paper_pdf" in select_stages(g))
        # the convergence experiment (scripts/convergence.jl): only when named; its task produces its case's refinement, so it runs after
        # the refinements' prepare (and the front ends it reads) and the image regions, never after the whole refinement array
        @test !("convergence" in select_stages(g)) && select_stages(g; only = ["convergence"]) ==
              ["setup", "distance_maps", "image_regions", "frontends_prepare", "frontends_detect", "frontends_init", "radiometric_refinement_prepare", "convergence"]
        # the warping ablation (scripts/warping_ablation.jl): only when named, after the same prerequisites (its mode "both" reuses the refinements of record by their keys)
        # (and after the refinements of record and the chrome sets' detect and index it shares: they are made once, upstream)
        @test !("warping_ablation" in select_stages(g)) && select_stages(g; only = ["warping_ablation"]) ==
              ["setup", "distance_maps", "image_regions", "frontends_prepare", "frontends_detect", "frontends_init", "radiometric_refinement_prepare",
               "radiometric_refinement", "chrome_evaluation", "warping_ablation"]
    end

    # every launcher of scripts/slurm/ is a stage's, the chrome index grid's too (a stage since 2026-11-12, its private three-job
    # launcher deleted): one path runs everything, and the old ones are gone
    launchers = sort([joinpath("experiments/scripts/slurm", f) for f in readdir(projectdir("scripts", "slurm")) if endswith(f, ".sbatch")])
    @test sort(unique(filter(!isempty, [s.sbatch for s in g.stages]))) == launchers
    @test !isfile(projectdir("scripts", "slurm", "submit_all.sh")) && !isfile(projectdir("scripts", "slurm", "submit_chrome_index_grid.sh"))
    @test !("chrome_index_grid" in select_stages(g)) && select_stages(g; only = ["chrome_index_grid"]) == ["setup", "chrome_index_grid"]
    # the case selections the graph exports are the ones the launchers read (common.sh, texture_recovery.sbatch)
    common = join(read(projectdir("scripts", "slurm", f), String) for f in ("common.sh", "texture_recovery.sbatch", "sampling_strategy.sbatch",
                  "chrome_evaluation.sbatch", "warping_ablation.sbatch"))
    for s in g.stages
        isempty(s.select_env) || @test occursin("\${$(s.select_env):-", common)
    end
    # --only=texture_recovery writes the scores and the summary: the launcher runs every texture stage through report (2026-11-15)
    @test occursin("--stage=\"\${TEXTURE_STAGES:-prepare,detect,sheet,init,refine,folds,evaluate,fold_report,compare,report}\"", read(projectdir("scripts", "slurm", "texture_recovery.sbatch"), String))
    # and its phases together run every one of them, the page solves alone on the GPU
    tex = only(filter(x -> x.name == "texture_recovery", g.stages))
    @test join([ph["env"]["TEXTURE_STAGES"] for ph in tex.phases], ",") == "prepare,detect,sheet,init,refine,folds,evaluate,fold_report,compare,report"
    @test [ph["gpu"] for ph in tex.phases] == [false, true, false]

    # the GPU split (the B200 server, 2026-11-13): GPU tasks never compile and never hold a GPU through CPU work. setup (the one precompile) and
    # every prepare (the RAW decode) are CPU stages; every GPU stage runs after setup, transitively, so it only loads
    stage(n) = only(filter(x -> x.name == n, g.stages))
    upstream(n) = setdiff(Set(select_stages(g; only = [n])), [n])       # every stage n runs after, transitively (Eval's select_stages)
    @test !stage("setup").gpu && isempty(stage("setup").gres)
    for s in g.stages
        endswith(s.name, "_prepare") && @test !s.gpu && isempty(s.gres)
        (s.gpu || !isempty(s.gres)) && @test "setup" in upstream(s.name)
    end

    # the shipped Slurm profile (VisualGeometryToolkitEval 0.11 ships vcai and desktop)
    slurm_site = "vcai"
    dry(site, more...) = (io = IOBuffer(); code = reproduce_main(["--graph=$graph", "--site=$site", "--dry-run", more...]; io, env = env0);
                          (code, String(take!(io))))
    data, results = mktempdir(), mktempdir()

    @testset "the dry run on the Slurm site: sbatch of the existing launchers, chained with afterok" begin
        code, out = dry(slurm_site, "--data-root=$data", "--results-root=$results")
        @test code == 0 && occursin("reproduce experiments on $slurm_site (slurm): $(length(paper)) stage(s)", out)
        submits = [strip(l) for l in split(out, '\n') if startswith(strip(l), "submit")]
        @test length(submits) >= length(paper)                       # a stage with phases submits one job per phase (and the precompile)
        used = sort(unique([m.match for l in submits for m in eachmatch(r"experiments/scripts/slurm/[a-z_]+\.sbatch", l)]))
        @test all(f -> isfile(joinpath(root, f)), used)
        @test used == sort(unique(filter(!isempty, [s.sbatch for s in g.stages if s.name in paper])))
        @test all(l -> occursin("--exclude=athena", l), submits)          # the vcai profile's sbatch options
        # each stage's own CPUs and memory (configs/reproduce.toml, over the site's 28): one thread count, 8, for every keyed stage
        @test all(l -> occursin("--mem-per-cpu=8480M", l), submits)
        for name in paper
            s = only(filter(x -> x.name == name, g.stages))
            @test s.mem_per_cpu == "8480M" && occursin("--cpus-per-task=$(s.cpus_per_task)", out)
            # the front ends size their own (6/4/2); the texture's poses phase needs 237 GB, so 28 CPUs at the node's ratio, its threads
            # held at 8 by its launcher (texture_recovery.sbatch: the thread count keys the refinement it reads; 2026-11-16)
            name in ("setup", "results_branch", "texture_recovery") || startswith(name, "frontends_") || @test s.cpus_per_task == 8
        end
        chrome = sort(collect(keys(TOML.parsefile(projectdir("configs", "experiments", "radiometric_refinement.toml"))["cases"])))
        @test occursin("CHROME_CASES=$(join(chrome, ",")) EXPERIMENT=radiometric_refinement STAGES=start,refine sbatch --parsable --dependency=afterok:\$radiometric_refinement_prepare", out)
        @test occursin("--array=0-$(length(chrome) - 1) --time=", out)
        @test occursin("EXPERIMENT=radiometric_refinement sbatch --parsable --dependency=afterok:\$setup", out)
        @test occursin("sbatch --parsable --dependency=afterok:\$radiometric_refinement:\$texture_recovery:\$radiometric_refinement_vignetting:\$chrome_evaluation_score", out)
        @test occursin("REGENERATE_RESULTS_BRANCH=1 sbatch", out)
        @test occursin("TEXTURE_SETS=14mm_small TEXTURE_STAGES=", out)
        @test occursin("results_branch_of_run", out) && occursin("--dependency=afterok:\$regenerate_paper", out)
        @test occursin("driver   julia --project=experiments experiments/scripts/radiometric_refinement.jl --config=experiments/configs/experiments/radiometric_refinement.toml", out)
        @test !isdir(joinpath(results, "_reproduce"))                   # a dry run writes nothing
        # a site profile's own roots when none is given: the shipped vcai profile's [storage] (machine paths live in the site
        # profiles alone; the project's graph names none)
        code, out = dry("vcai")
        vcai = load_site("vcai")
        @test occursin("data     $(site_root(vcai, "data_root", g.name))", out) &&
              occursin("results  $(site_root(vcai, "results_root", g.name))", out)
        @test !haskey(g.site_roots, "vcai")
    end

    @testset "the dry run on the desktop: the same launchers under bash, device 0 only" begin
        code, out = dry("desktop", "--only=texture_recovery")
        @test code == 0 && occursin("reproduce experiments on desktop (local): 9 stage(s)", out)   # with the front ends the prepare waits on and the distance maps
        runs = [strip(l) for l in split(out, '\n') if startswith(strip(l), "run ")]
        @test length(runs) == 9 && all(l -> occursin("SLURM_SUBMIT_DIR=$(normpath(root * "/"))", l), runs)
        @test endswith(runs[1], "bash experiments/scripts/slurm/setup.sbatch") && !occursin(r"CUDA_VISIBLE_DEVICES=[0-9]", runs[1])   # setup: CPU only (GPU tasks never compile)
        prep = only(filter(l -> occursin("EXPERIMENT=radiometric_refinement", l) && occursin("prepare", l), runs))
        @test !occursin(r"CUDA_VISIBLE_DEVICES=[0-9]", prep)   # the decode: CPU
        @test all(l -> !occursin(r"CUDA_VISIBLE_DEVICES=[1-9]", l), runs)
        @test occursin("data     $(projectdir("data", "exp_raw"))", out) && occursin("results  $(projectdir("data", "results"))", out)
    end

    @testset "the missing inputs: reported as data before anything runs" begin
        rows = missing_inputs(g, ["radiometric_refinement"]; data_root = data, env = env0)          # an empty data root: every input
        cfg = TOML.parsefile(projectdir("configs", "experiments", "radiometric_refinement.toml"))
        @test sort([r.path for r in rows]) == sort(unique([t["path"] for t in values(cfg["inputs"])]))
        @test all(r -> r.used_by == ["radiometric_refinement (experiments/configs/experiments/radiometric_refinement.toml)"], rows)
        # each row names its input's tier (Eval decorates the reserved tier "unpublished" — the FX3 record — as never fetched)
        @test all(r -> first(split(r.tier)) == cfg["inputs"][first(k for (k, t) in cfg["inputs"] if t["path"] == r.path)]["tier"], rows)
        first_input = first(sort([t["path"] for t in values(cfg["inputs"])]))
        mkpath(joinpath(data, first_input))                                                  # one there: no longer reported
        @test !any(r -> r.path == first_input, missing_inputs(g, ["radiometric_refinement"]; data_root = data, env = env0))
        # the regeneration's drivers and the chrome evaluations they run are checked too
        regen = missing_inputs(g, ["regenerate_paper"]; data_root = data, env = env0)
        @test any(r -> any(occursin("chrome_evaluation_fx3_14mm_central30cm.toml, via experiments/configs/experiments/chrome_evaluation_paper.toml", u) for u in r.used_by), regen)
        # the command line: nothing runs, exit 3, the report
        io = IOBuffer()
        @test reproduce_main(["--graph=$graph", "--site=desktop", "--only=radiometric_refinement", "--data-root=$data", "--results-root=$results"];
                             io, env = env0) == 3
        out = String(take!(io))
        @test occursin("path\ttier\tused by", out) && occursin("nothing ran", out) && !isdir(joinpath(results, "_reproduce"))
    end
end
