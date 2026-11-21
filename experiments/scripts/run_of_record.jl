# =============================================================================
# run_of_record.jl — the run of record's launch (src/run_of_record.jl) through VisualGeometryToolkitEval's runner: the stages of
# configs/run_of_record.toml (and what they run after) of the stage graph of record, configs/reproduce.toml, on the site --site names,
# into the release's results root, its jobs `ror-<driver>-<stage>[-<phase>]`, the GPU queue longest-first. Run from a bare julia
# by experiments/scripts/slurm/run_of_record.sh (which first checks the registered version and makes its tree), from the tree's root:
#
#   julia --project=experiments experiments/scripts/run_of_record.jl --version=<v> [--plan] --site=<site> [--results-root=<dir>]
#         [--config=experiments/configs/run_of_record.toml] [--graph=experiments/configs/reproduce.toml] [--scratch]
#
#   --plan     the runner's job graph (stages, items, dependencies, sbatch lines, missing inputs) and the run of record's plan: the
#              jobs with their GPU and CPU counts, the runtime estimates against the limits, the projected wall-clock at full
#              capacity, the jobs at risk and the required experiments the launch lacks; nothing is submitted
#   --scratch  a dry run on a scratch tree (the launcher's --dry-run=<rev>: temporary [sources], VGTK_SCRATCH=1); only with --plan
#   (none)     the launch: refused (nothing submitted) when a required experiment is missing (exit 5), an input is missing (exit
#              3), the instantiated Manifest holds an unregistered package (exit 6), or the code is uncommitted (exit 2)
#
# It writes no result of its own: the runner writes the launch's manifest, the drivers their keyed runs.
# =============================================================================
const ROOT = pwd()
const EVAL = let src = Base.find_package("VisualGeometryToolkitEval")
    src === nothing && (println("run_of_record: VisualGeometryToolkitEval is not in the experiments environment: instantiate it first"); exit(2))
    dirname(src)
end
include(joinpath(dirname(Base.find_package("VisualGeometryToolkitCore")), "base", "canonical.jl"))   # the run keys' encoding
include(joinpath(EVAL, "cli.jl"))
include(joinpath(EVAL, "reproduce", "runner.jl"))
include(joinpath(ROOT, "experiments", "src", "run_of_record.jl"))

function main(args)
    val(k, d = nothing) = something(_cli_value(args, k), Some(d))
    version = val("version")
    version === nothing && (println("usage: run_of_record.jl --version=<rc version> [--plan] [--scratch] [--site=…] [--results-root=…]"); return 2)
    plan_only, scratch = "--plan" in args, "--scratch" in args
    scratch && !plan_only && (println("run_of_record: --scratch is for the dry run only (--plan): a run of record runs the registered tree"); return 2)
    cfg = load_ror_config(val("config", "experiments/configs/run_of_record.toml"))
    g = load_stage_graph(val("graph", "experiments/configs/reproduce.toml"))
    sitename = val("site")
    sitename === nothing && (println("run_of_record: name the site: --site=<site> (VisualGeometryToolkitEval configs/sites/; run_of_record.sh passes VGTK_SITE)"); return 2)
    site = load_site(sitename; dirs = isempty(g.sites_dir) ? String[] : [g.sites_dir])
    launch = cfg["launch"]
    r = val("results-root")      # given (the launcher passes it): no results base is resolved
    results = r !== nothing ? r : joinpath(ror_results_base(g, site), ror_pattern(launch["results_dir"], version))
    sel = ror_stages(g, cfg)
    isempty(sel.dropped) || println("run of record: stage(s) absent from the graph, optional, left out: $(join(sel.dropped, ", "))")
    rg = ror_graph(g, cfg)
    # a scratch launch (VGTK_SCRATCH=1, set by the launcher's --dry-run) never keys a run of record: the runner reads it
    plan = plan_reproduction(rg, site; only = sel.stages, results_root = results, job_name_prefix = ROR_PREFIX)
    unpub = NamedTuple[]
    miss = missing_inputs(rg, [st.spec.name for st in plan.stages]; data_root = plan.data_root, site = site.name, unpublished = unpub)
    m = get(cfg, "machine", Dict{String, Any}())
    jobs = ror_jobs(plan, cfg)
    sched = ror_schedule(jobs; gpus = get(m, "gpus", 8), cpus = get(m, "cpus", 224))
    lacking = ror_missing(plan, cfg)
    leaks = isfile(joinpath(ROOT, "Manifest.toml")) ? ror_manifest_leaks(joinpath(ROOT, "Manifest.toml"), ROOT) : ["no Manifest.toml (not instantiated)"]
    if plan_only
        scratch && println("== SCRATCH DRY RUN (temporary [sources]: never a run of record; VGTK_SCRATCH=1)")
        println("== RadiometricCamera $version: tree $ROOT at $(plan.commit)$(plan.dirty ? " (dirty)" : ""), results $results")
        print_plan(stdout, plan; missing = miss, unpublished = unpub)
        ror_report(stdout, plan, jobs, sched, cfg; missing = lacking)
        println("\n== the Manifest: ", isempty(leaks) ? "every package registered (RadiometricCamera the tree's own)" :
                "$(length(leaks)) unregistered" * (scratch ? " (SCRATCH, the dry run's temporary [sources])" : ""))
        foreach(l -> println("  ", l), leaks)
        println("\ndry run: nothing submitted")
        return 0
    end
    isempty(lacking) || (println("run of record REFUSED: required experiment(s) missing:"); foreach(x -> println("  ", x), lacking); return 5)
    isempty(leaks) || (println("run of record REFUSED: the Manifest holds unregistered packages:"); foreach(l -> println("  ", l), leaks); return 6)
    if !isempty(miss)
        print_missing(stdout, miss; data_root = plan.data_root, data_env = rg.data_env)
        println("run of record REFUSED: inputs missing, nothing submitted"); return 3
    end
    println("run of record: RadiometricCamera $version, $(plan.commit), manifest $(plan.manifest); projected $(_hm(sched.makespan)) at full capacity")
    submit_slurm!(plan)
    println("manifest $(plan.manifest)")
    return 0
end

exit(try
    main(ARGS)
catch e
    e isa ArgumentError ? (println("run_of_record: ", e.msg); 2) : e isa ErrorException ? (println("run_of_record: ", e.msg); 4) : rethrow()
end)
