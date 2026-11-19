# The run of record's launch (scripts/run_of_record.jl, src/run_of_record.jl, configs/run_of_record.toml,
# scripts/slurm/run_of_record.sh): its graph construction on the stage graph of record — the stages it selects, the job names, the
# paper-first / longest-first submission order, the packing (one GPU per GPU task, no GPU in a CPU phase, 8480M per CPU), the shared
# products made once upstream of their readers, the schedule and its wall-clock, the risks, the required experiments and the
# Manifest's registered packages. Nothing is submitted: the runner's plan on temporary roots.
@testitem "run of record: the graph, the jobs, the schedule, the risks" default_imports = false begin
    using Test
    using DrWatson
    using TOML: TOML
    using VisualGeometryToolkitEval.Reproduce: load_stage_graph, load_site, plan_reproduction, select_stages
    include(srcdir("run_of_record.jl"))

    cfg = load_ror_config(projectdir("configs", "run_of_record.toml"))
    g = load_stage_graph(projectdir("configs", "reproduce.toml"))
    stage(n) = only(filter(s -> s.name == n, g.stages))
    env0 = filter(p -> !(first(p) in ("VGTK_DATA_ROOT", "VGTK_RESULTS_ROOT", "CASES", "CHROME_CASES", "VIG_CASES", "TEXTURE_SETS", "SETS",
                                      "WARP_MODES", "WARP_SHARDS", "VIG_AGREE_SHARDS", "CHROME_SETS", "PS_SETS", "ARM_PACKS",
                                      "VGTK_DATA_ROOT_UNPUBLISHED", "VGTK_SCRATCH")), Dict(ENV))

    @testset "the config: its keys, its stages in the graph, every estimate in minutes" begin
        bad(f) = (c = TOML.parsefile(projectdir("configs", "run_of_record.toml")); f(c); p = joinpath(mktempdir(), "ror.toml");
                  open(io -> TOML.print(io, c), p, "w"); p)
        @test_throws ArgumentError load_ror_config(bad(c -> c["launch"]["colour"] = "red"))
        @test_throws ArgumentError load_ror_config(bad(c -> c["estimate"]["setup"] = "15 min"))
        @test_throws ArgumentError load_ror_config(bad(c -> push!(c["require"], Dict("what" => "no stage"))))
        sel = ror_stages(g, cfg)
        @test isempty(sel.dropped) && sel.stages == cfg["launch"]["stages"]
        # every stage the paper needs (the graph's default ones) and the front ends; the studies (configs/studies/) and the
        # out-of-scope ones never (rc-0116: the run of record runs the main group)
        @test issubset(select_stages(g), select_stages(g; only = sel.stages))
        @test issubset(["frontends_evaluate", "texture_recovery", "chrome_evaluation_score"], sel.stages)
        @test isempty(intersect(["warping_ablation", "sampling_strategy_upstreams", "sampling_strategy", "sampling_strategy_score"],
                                select_stages(g; only = sel.stages)))
        @test isempty(intersect(["golden", "convergence", "chrome_index_grid", "paper_pdf"], select_stages(g; only = sel.stages)))
        # a launch stage the graph lacks is refused (no stage is optional)
        @test_throws ArgumentError ror_stages(_with(g; stages = filter(s -> s.name != "texture_recovery", g.stages)), cfg)
        @test ror_pattern(cfg["launch"]["results_root"], "0.11.5") == "/raid/data/jpr/rc-results-0.11.5"
    end

    @testset "job names: ror-<driver>-<stage>[-<phase>]" begin
        @test ror_job_base(stage("radiometric_refinement_vignetting")) == "radiometric_refinement-vignetting"
        @test ror_job_base(stage("radiometric_refinement")) == "radiometric_refinement-start+refine"
        @test ror_job_base(stage("chrome_evaluation_score")) == "chrome_evaluation-score"
        @test ror_job_base(stage("frontends_detect")) == "frontends-detect"
        @test ror_job_base(stage("setup")) == "env-setup" && ror_job_base(stage("results_branch")) == "paper-results_branch"
        # never one of the gate-first watcher's experiment names (anchored ^(samp|warp|fe30w|sw-|pano|dotstat|vfin|es-|arm|exp)): never held
        @test !occursin(r"^(samp|warp|fe30w|sw-|pano|dotstat|vfin|es-|arm|exp)", ROR_PREFIX)
    end

    @testset "the order: a dependency order, the paper first, longest remaining chain first" begin
        rg = ror_graph(g, cfg)
        order = [s.name for s in rg.stages]
        @test sort(order) == sort([s.name for s in g.stages])
        pos = Dict(n => i for (i, n) in enumerate(order))
        @test all(pos[a] < pos[s.name] for s in rg.stages for a in s.after)
        paper = Set(select_stages(g))
        @test maximum(pos[n] for n in paper) < minimum(pos[n] for n in order if !(n in paper) && n in ror_stages(g, cfg).stages)
        @test all(s -> startswith(s.job_name, ror_job_base(s)), rg.stages)
    end

    rg = ror_graph(g, cfg)
    site = load_site("dgx01")
    plan = plan_reproduction(rg, site; only = ror_stages(g, cfg).stages, data_root = mktempdir(), results_root = mktempdir(),
                             job_name_prefix = ROR_PREFIX, env = env0)
    jobs = ror_jobs(plan, cfg)

    @testset "the jobs: packing and naming" begin
        @test all(j -> startswith(j.name, "ror-"), jobs)
        @test all(c -> "--mem-per-cpu=8480M" in c, [[st.command for st in plan.stages]; [p.command for st in plan.stages for p in st.phases]])
        # one GPU per GPU task, never a GPU in a CPU phase
        for st in plan.stages, p in [[q.spec for q in st.phases]; st.spec]
            gpu = p.gpu || !isempty(p.gres)
            line = join(p === st.spec ? st.command : only(q.command for q in st.phases if q.spec === p), " ")
            @test occursin("--gres=gpu:1", line) == gpu
            occursin("--gres=gpu:", line) && @test !occursin(r"--gres=gpu:[2-9]", line)
        end
        byname = Dict(j.name => j for j in jobs)
        @test !byname["ror-texture_recovery-run-poses"].gpu && byname["ror-texture_recovery-run-solve"].gpu && !byname["ror-texture_recovery-run"].gpu
        @test !byname["ror-chrome_evaluation-run-decode"].gpu && byname["ror-chrome_evaluation-run-detect"].gpu && !byname["ror-chrome_evaluation-run"].gpu
        @test !byname["ror-regenerate_paper-run"].gpu
        @test all(j -> j.phase != "precompile" || !j.gpu, jobs)        # the runner's precompile: CPU only, so a GPU task only loads
        # the per-item phases follow item by item (aftercorr)
        @test byname["ror-chrome_evaluation-run-detect"].per_item && byname["ror-chrome_evaluation-run"].per_item
    end

    @testset "nothing twice: a shared product is made upstream of every reader" begin
        upstream(n) = setdiff(Set(select_stages(g; only = [n])), [n])   # every stage n runs after, transitively (Eval's select_stages)
        @test issubset(["radiometric_refinement", "chrome_evaluation"], upstream("warping_ablation"))                 # mode both; the detect and index
        @test issubset(["radiometric_refinement", "radiometric_refinement_vignetting"], upstream("sampling_strategy"))  # an arm of record
        @test issubset(["chrome_evaluation", "texture_recovery"], upstream("sampling_strategy_score"))
        @test issubset(["frontends_init", "radiometric_refinement_prepare"], upstream("sampling_strategy_upstreams"))
        @test issubset(["chrome_evaluation_score"], upstream("regenerate_paper"))
        @test issubset(["radiometric_refinement", "radiometric_refinement_vignetting"], upstream("chrome_evaluation_score"))
    end

    @testset "the schedule: a list schedule on the node" begin
        J(name, gpu, cpus, tasks, minutes; after = Int[], per_item = false) =
            (; stage = name, phase = "", name, gpu, cpus, tasks, items = String[], minutes = Float64(minutes), timed = true, limit = 0,
             per_item, after)
        # 3 GPU tasks of 10 min on 2 GPUs: 20 min; a CPU job beside them does not wait
        s = ror_schedule([J("a", true, 8, 3, 10), J("b", false, 8, 1, 5)]; gpus = 2, cpus = 224)
        @test s.makespan == 20 && s.finish[2] == 5 && s.gpu_minutes == 30
        # a chain: afterok on the whole array, then a per-item phase follows each task
        s = ror_schedule([J("a", true, 8, 2, 10), J("b", false, 8, 2, 5; after = [1], per_item = true), J("c", false, 2, 1, 1; after = [2])];
                         gpus = 1, cpus = 224)
        @test s.start[2] == 10 && s.finish[2] == 25 && s.makespan == 26
        # CPUs bound too: 4 tasks of 100 CPUs on 224 run two at a time
        @test ror_schedule([J("a", false, 100, 4, 10)]; gpus = 8, cpus = 224).makespan == 20
        @test_throws ArgumentError ror_schedule([J("a", false, 300, 1, 10)]; cpus = 224)
        # the launch: every task scheduled, no GPU idle while GPU work waits past the paper
        sched = ror_schedule(jobs; gpus = 8, cpus = 224)
        @test sched.makespan >= sched.gpu_minutes / 8 && sched.busy_gpu > 0.5
        @test all(isfinite, sched.finish)
    end

    @testset "the risks, the required experiments, the report" begin
        J(limit, minutes; timed = true) = (; stage = "x", phase = "", name = "ror-x", gpu = true, cpus = 8, tasks = 1, items = String[],
                                              minutes = Float64(minutes), timed, limit, per_item = false, after = Int[])
        r = ror_risks([J(60, 90), J(60, 50), J(60, 10; timed = false)], nothing, cfg)
        @test [x.level for x in r] == ["EXCEEDS", "TIGHT", "UNMEASURED"]
        # no measured job of the launch is projected past its limit (the slow-run rule: limits ≈ 2× the measured runtime)
        @test !any(x -> x.level == "EXCEEDS", ror_risks(jobs, nothing, cfg))
        req = Dict("require" => [Dict("what" => "w", "stage" => "radiometric_refinement", "item" => "^sitla_spline_warped\$"),
                                 Dict("what" => "absent", "stage" => "radiometric_refinement", "item" => "^nope\$"),
                                 Dict("what" => "gone", "stage" => "no_such_stage", "optional" => true)])
        @test ror_missing(plan, req) == ["absent: no item of stage radiometric_refinement matches /^nope\$/ (items: $(join(only(st.items for st in plan.stages if st.spec.name == "radiometric_refinement"), ", ")))"]
        io = IOBuffer(); ror_report(io, plan, jobs, ror_schedule(jobs), cfg); out = String(take!(io))
        @test occursin("projected wall-clock at full capacity (8 GPUs, 224 CPUs)", out) && occursin("GPU jobs ", out) && occursin("tasks: 0=", out)
    end

    @testset "the Manifest: registered packages only, the project's own inside the tree" begin
        tree = mktempdir()
        write(joinpath(tree, "Manifest.toml"), """
            julia_version = "1.13.3"
            manifest_format = "2.0"
            [[deps.RadiometricCamera]]
            path = "."
            uuid = "7c1f0a52-9d3e-4b6a-9f1e-3d2a8c5b7e01"
            [[deps.VisualGeometryToolkitCore]]
            path = "/elsewhere/VisualGeometryToolkitCore"
            uuid = "96820d6b-1dad-4e77-ad4a-8276f2ec1ce7"
            [[deps.Foo]]
            repo-url = "https://example.invalid/Foo.jl"
            uuid = "00000000-0000-0000-0000-000000000001"
            [[deps.Bar]]
            git-tree-sha1 = "0000000000000000000000000000000000000000"
            uuid = "00000000-0000-0000-0000-000000000002"
            version = "1.0.0"
            """)
        @test ror_manifest_leaks(joinpath(tree, "Manifest.toml"), tree) ==
              ["Foo: repo-url https://example.invalid/Foo.jl", "VisualGeometryToolkitCore: path /elsewhere/VisualGeometryToolkitCore (outside the tree)"]
    end

    @testset "the launcher: the registry first, the tree, the runner; no experiment before the version is registered" begin
        sh = read(projectdir("scripts", "slurm", "run_of_record.sh"), String)
        @test occursin("Pkg.Registry.update()", sh) && occursin("is_tracking_path || d.is_tracking_repo", sh)
        @test occursin("--gres=none", sh) && occursin("--mem-per-cpu=8480M", sh)
        @test occursin("git-tree-sha1", sh) && occursin("v\$V:RadiometricCamera", sh) &&                       # the subdir registration: the package directory's tree
              occursin("-- RadiometricCamera/src RadiometricCamera/bin RadiometricCamera/ext RadiometricCamera/Project.toml", sh)
        @test occursin("experiments/scripts/run_of_record.jl", sh) && occursin("SCRATCH", sh)
    end
end
