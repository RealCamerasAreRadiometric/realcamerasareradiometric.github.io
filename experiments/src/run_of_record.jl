# =============================================================================
# run_of_record.jl — the run of record of the paper on a Slurm site (a site profile with backend slurm): every stage of configs/reproduce.toml the paper needs (the stages
# configs/run_of_record.toml lists, and what they run after), submitted by VisualGeometryToolkitEval's runner (Reproduce) as ONE
# launch, its jobs named `ror-<driver>-<stage>[-<phase>]`, and the GPU queue ordered longest-first; and its plan: the job graph with
# its GPU and CPU counts, each job's runtime estimate (the config's, from the last good runs) against its Slurm limit, and the
# launch's projected wall-clock at the machine's full capacity (a list schedule of the job graph on its GPUs and CPUs).
#
# DEPENDENCY-FREE (Base and the stdlibs TOML and Dates), like the runner it builds on: scripts/run_of_record.jl includes the runner
# (Eval's src/cli.jl and src/reproduce/runner.jl, by path) and then this file, from a bare julia; the tests include it the same way.
# Nothing here plans or submits a stage itself: the stage graph, the items, the keys, the dependencies and the sbatch lines are the
# runner's (`plan_reproduction`, `submit_slurm!`), so a run of record is a launch like any other, reused by its keys.
# =============================================================================
using TOML: TOML

"The job-name prefix of the run of record (not in the gate-first watcher's list of experiment names: never held)."
const ROR_PREFIX = "ror-"

const _ROR_KEYS = Dict("launch" => ("stages", "optional", "results_dir", "tree_dir", "repository", "registry"), "machine" => ("gpus", "cpus"),
                       "estimate" => nothing, "require" => nothing, "risk" => ("tight_fraction", "untimed"))

"""
    load_ror_config(path) -> Dict

The run of record's config (configs/run_of_record.toml): `[launch] stages` (the stages the launch names, with what they run after),
`results_dir` and `tree_dir` (patterns with `{version}`, under the machine's results base: [`ror_results_base`](@ref)), `[machine]` gpus and cpus (the node the plan schedules on), `[estimate]`
(`"<stage>"` or `"<stage>:<phase>"` = minutes per task; `default` and `precompile`), `[[require]]` (`what`, `stage`, and an optional
`item` regex: an experiment the paper needs, which the launch must hold) and `[risk] tight_fraction`. Unknown tables or keys are refused.
"""
function load_ror_config(path::AbstractString)
    d = TOML.parsefile(path)
    for (k, v) in d
        haskey(_ROR_KEYS, k) || throw(ArgumentError("$path: unknown table [$k] (known: $(join(sort(collect(keys(_ROR_KEYS))), ", ")))"))
        allowed = _ROR_KEYS[k]
        allowed === nothing && continue
        for kk in keys(v)
            kk in allowed || throw(ArgumentError("$path: unknown key $k.$kk (known: $(join(allowed, ", ")))"))
        end
    end
    for r in get(d, "require", Any[])
        haskey(r, "what") && haskey(r, "stage") || throw(ArgumentError("$path: a [[require]] needs what and stage"))
    end
    for (k, v) in get(d, "estimate", Dict{String, Any}())
        v isa Real && v >= 0 || throw(ArgumentError("$path: estimate.\"$k\" = $(repr(v)) is not minutes"))
    end
    return d
end

ror_pattern(p::AbstractString, version::AbstractString) = replace(p, "{version}" => version)

"""
    ror_results_base(g, site; env = ENV) -> String

The machine's results base, under which a release's results directory and tree are made: the project's results root on the site,
as the runner resolves it (`resolve_roots`: the graph's results variable, VGTK_RESULTS_ROOT, else the site profile's `[storage]
results_root`); an error when nothing names one (a profile without [storage]: set the variable). Machine paths are the site
profiles' alone.
"""
function ror_results_base(g, site; env = ENV)
    r = last(resolve_roots(g, site; env))
    isempty(r) && throw(ArgumentError("site $(site.name) names no [storage] results_root: set $(g.results_env)"))
    return r
end

# A copy of an immutable struct with some fields replaced (the runner's StageSpec and StageGraph: by field name, so a field
# the runner adds later is carried over).
_with(x; kw...) = typeof(x)((haskey(kw, f) ? kw[f] : getfield(x, f) for f in fieldnames(typeof(x)))...)

"""
    ror_job_base(s::StageSpec) -> String

The name of a stage's job in the run of record, before the prefix: `<driver>-<stage>`. The driver is the script's basename (`env`
for a stage without one, `paper` for the results branch), shortened to the words it shares with the stage's name
(`chrome_evaluation_score` of chrome_evaluation_paper.jl is `chrome_evaluation-score`); the stage is the rest of the stage's name,
else its driver stages (STAGES or STAGE, `+`-joined, up to three), else `run`. The runner names a phase's job `<this>-<phase>`, and
an array task is its case (the plan lists task i = item i).
"""
function ror_job_base(s)
    isempty(s.driver) && return (s.results_branch ? "paper" : "env") * "-" * s.name
    dw, nw = split(first(splitext(basename(s.driver))), '_'), split(s.name, '_')
    k = 0
    while k < min(length(dw), length(nw)) && dw[k + 1] == nw[k + 1]
        k += 1
    end
    k == 0 && return join(dw, '_') * "-" * s.name
    rest = join(nw[(k + 1):end], '_')
    if isempty(rest)
        st = split(get(s.env, "STAGES", get(s.env, "STAGE", "")), ','; keepempty = false)
        rest = 1 <= length(st) <= 3 ? join(st, '+') : "run"
    end
    return join(dw[1:k], '_') * "-" * rest
end

# The estimate of a stage's job (its last phase: the stage's own name) or of one of its phases, minutes per task.
function ror_estimate(cfg, stage::AbstractString, phase::AbstractString = "")
    e = get(cfg, "estimate", Dict{String, Any}())
    phase == "precompile" && return (Float64(get(e, "$stage:precompile", get(e, "precompile", 2))), haskey(e, "precompile"))
    key = isempty(phase) ? stage : "$stage:$phase"
    assumed = key in get(get(cfg, "risk", Dict{String, Any}()), "untimed", String[])
    haskey(e, key) && return (Float64(e[key]), !assumed)
    return (Float64(get(e, "default", 60)), false)
end

# Minutes of a Slurm time "[D-]HH:MM:SS" | "MM" (0: none).
function ror_minutes(t::AbstractString)
    isempty(t) && return 0
    d, rest = occursin('-', t) ? (parse(Int, split(t, '-')[1]), split(t, '-')[2]) : (0, t)
    p = parse.(Int, split(rest, ':'))
    m = length(p) == 3 ? p[1] * 60 + p[2] + p[3] / 60 : length(p) == 2 ? p[1] + p[2] / 60 : p[1]
    return round(Int, d * 1440 + m)
end

"The duration of one task of a stage, all its phases in turn: the estimate the graph's longest-first order and the plan read."
ror_stage_minutes(cfg, s) = isempty(s.phases) ? ror_estimate(cfg, s.name)[1] :
    sum(ror_estimate(cfg, s.name, i == length(s.phases) ? "" : String(ph["name"]))[1] for (i, ph) in enumerate(s.phases))

"""
    ror_graph(g, cfg) -> StageGraph

The run of record's view of the stage graph `g`: every job named [`ror_job_base`](@ref), and the stages in a dependency order
that puts the paper first and then the longest remaining chain first (each stage's own estimate plus the longest chain after it),
so the runner submits them in that order: the stages the paper's figures, tables and numbers need (the graph's default stages and
everything they run after) before the experiments no paper asset reads, each tier longest-first. the site's jobs share one priority (multifactor with every weight 0: `--nice` changes nothing), so Slurm starts
the eligible jobs in submission order, and the longest first keeps the tail short. The graph's keys, items and commands are
otherwise the graph's own.
"""
function ror_graph(g, cfg)
    stages = [_with(s; job_name = ror_job_base(s)) for s in g.stages]
    names = [s.name for s in stages]
    succ = Dict(n => String[] for n in names)
    for s in stages, a in s.after
        push!(succ[a], s.name)
    end
    own = Dict(s.name => ror_stage_minutes(cfg, s) for s in stages)
    level = Dict{String, Float64}()
    bottom(n) = get!(() -> own[n] + maximum((bottom(t) for t in succ[n]); init = 0.0), level, n)
    foreach(bottom, names)
    paper = Set(select_stages(g))                        # the default stages and their prerequisites
    out, left = eltype(stages)[], copy(stages)
    while !isempty(left)
        ready = findall(s -> all(a -> any(o -> o.name == a, out), s.after), left)
        isempty(ready) && throw(ArgumentError("stage graph: a dependency cycle"))
        i = ready[argmax([(left[k].name in paper, level[left[k].name], -k) for k in ready])]
        push!(out, popat!(left, i))
    end
    return _with(g; stages = out)
end

"""
    ror_jobs(plan, cfg) -> Vector{NamedTuple}

The jobs of a planned launch (the runner's `ReproducePlan`), in submission order, each `(; stage, phase, name, gpu, cpus, tasks,
items, minutes, timed, limit, per_item, after)`: a stage's phases (the runner's precompile first, on Slurm) and its own job, the
job name with the launch's prefix, its device and CPUs per task, its tasks (an array's items, or one), the estimate per task and
whether the config measured it, its `--time` in minutes, whether it runs per item after the phase before it (`aftercorr`), and the
jobs it waits on (the phase before it; for a stage's first job, the last job of each stage it runs after). A stage the launch
reuses (completed earlier) has no jobs.
"""
function ror_jobs(plan, cfg)
    jobs = NamedTuple[]
    last_of = Dict{String, Int}()
    prefix = ROR_PREFIX
    for st in plan.stages
        st.reuse == "completed" && continue
        specs = [[(p.name, p.spec) for p in st.phases]; [("", st.spec)]]
        prev = 0
        for (k, (ph, s)) in enumerate(specs)
            per_item = s.is_array
            tasks = per_item ? length(st.items) : 1
            est, timed = ror_estimate(cfg, st.spec.name, ph)
            limit = per_item && st.minutes > 0 && isempty(ph) ? st.minutes : ror_minutes(s.time)
            after = k == 1 ? [last_of[a] for a in st.after if haskey(last_of, a)] : [prev]
            name = prefix * (isempty(ph) ? st.spec.job_name : "$(st.spec.job_name)-$ph")
            push!(jobs, (; stage = st.spec.name, phase = ph, name, gpu = s.gpu || !isempty(s.gres),
                         cpus = s.cpus_per_task > 0 ? s.cpus_per_task : plan.site.cpus_per_task, tasks,
                         items = per_item ? st.items : String[], minutes = est, timed, limit,
                         per_item = per_item && k > 1 && specs[k - 1][2].is_array, after))
            prev = length(jobs)
        end
        last_of[st.spec.name] = prev
    end
    return jobs
end

"""
    ror_schedule(jobs; gpus = 8, cpus = 224) -> NamedTuple

A list schedule of the job graph on one node of `gpus` GPUs and `cpus` CPUs, as the B200 server's backfill runs it: at every event, each
eligible task in submission order starts when its GPU (one per GPU task) and CPUs are free; a task is eligible once every job it
waits on has ended (task k of a per-item phase once task k of the phase before it has). Returns `(; makespan, start, finish,
gpu_minutes, cpu_minutes, busy_gpu)`: minutes from the launch, per job its first start and last finish, the GPU- and CPU-minutes,
and the GPUs' mean occupancy.
"""
function ror_schedule(jobs; gpus::Integer = 8, cpus::Integer = 224)
    n = length(jobs)
    tstart = [fill(Inf, j.tasks) for j in jobs]
    tend = [fill(Inf, j.tasks) for j in jobs]
    done(j) = all(isfinite, tend[j])
    jobend(j) = maximum(tend[j])
    free_gpu, free_cpu, now = gpus, cpus, 0.0
    running = Tuple{Float64, Int, Int}[]              # (end, job, task)
    left = sum(j.tasks for j in jobs; init = 0)
    for j in jobs
        j.cpus <= cpus && (!j.gpu || gpus >= 1) || throw(ArgumentError("job $(j.name): $(j.cpus) CPUs, gpu=$(j.gpu) never fits the node"))
    end
    while left > 0
        started = true
        while started
            started = false
            for (ji, j) in enumerate(jobs), t in 1:j.tasks
                isinf(tstart[ji][t]) || continue
                ready = j.per_item ? (tend[only(j.after)][t] <= now) : all(a -> done(a) && jobend(a) <= now, j.after)
                ready || continue
                (j.gpu && free_gpu < 1 || j.cpus > free_cpu) && continue
                tstart[ji][t] = now
                push!(running, (now + j.minutes, ji, t))
                free_cpu -= j.cpus; j.gpu && (free_gpu -= 1)
                started = true
            end
        end
        isempty(running) && throw(ArgumentError("the job graph never finishes (a job waits on one that cannot start)"))
        sort!(running)
        e, ji, t = popfirst!(running)
        now = e
        tend[ji][t] = e
        free_cpu += jobs[ji].cpus; jobs[ji].gpu && (free_gpu += 1)
        left -= 1
        while !isempty(running) && first(running)[1] <= now      # every task ending at the same minute
            e, ji, t = popfirst!(running)
            tend[ji][t] = e
            free_cpu += jobs[ji].cpus; jobs[ji].gpu && (free_gpu += 1)
            left -= 1
        end
    end
    makespan = maximum((maximum(t; init = 0.0) for t in tend); init = 0.0)
    gpu_minutes = sum((j.gpu ? j.tasks * j.minutes : 0.0) for j in jobs; init = 0.0)
    cpu_minutes = sum((j.tasks * j.minutes * j.cpus) for j in jobs; init = 0.0)
    return (; makespan, start = [minimum(t; init = Inf) for t in tstart], finish = [maximum(t; init = 0.0) for t in tend],
            gpu_minutes, cpu_minutes, busy_gpu = makespan > 0 ? gpu_minutes / (gpus * makespan) : 0.0)
end

"""
    ror_risks(jobs, sched, cfg) -> Vector{NamedTuple}

The jobs projected to exceed their Slurm limit or to come within `[risk] tight_fraction` of it (the slow-run rule: a run near its
limit is a defect to act on before it times out), and those whose estimate is unmeasured: `(; name, level, why)`, level
`"EXCEEDS"`, `"TIGHT"` or `"UNMEASURED"`.
"""
function ror_risks(jobs, sched, cfg)
    tight = Float64(get(get(cfg, "risk", Dict{String, Any}()), "tight_fraction", 0.75))
    out = NamedTuple[]
    for j in jobs
        if j.limit > 0 && j.minutes > j.limit
            push!(out, (; name = j.name, level = "EXCEEDS", why = "estimate $(_hm(j.minutes)) per task > its limit $(_hm(j.limit))"))
        elseif j.limit > 0 && j.minutes > tight * j.limit
            push!(out, (; name = j.name, level = "TIGHT", why = "estimate $(_hm(j.minutes)) per task is > $(round(Int, 100tight)) % of its limit $(_hm(j.limit))"))
        end
        j.timed || j.phase == "precompile" || push!(out, (; name = j.name, level = "UNMEASURED", why = "no measured run: $(_hm(j.minutes)) assumed per task"))
    end
    return out
end

"""
    ror_stages(g, cfg) -> (; stages, dropped)

The stages the launch names (`[launch] stages`), those of them the graph `g` has; a stage of `[launch] optional` the graph lacks is
`dropped` (an experiment the release may leave out: the sampling-strategy record), any other missing stage an `ArgumentError`.
"""
function ror_stages(g, cfg)
    l = get(cfg, "launch", Dict{String, Any}())
    want, optional = String.(get(l, "stages", String[])), Set(String.(get(l, "optional", String[])))
    have = Set(s.name for s in g.stages)
    bad = [n for n in want if !(n in have) && !(n in optional)]
    isempty(bad) || throw(ArgumentError("run of record: stage(s) $(join(bad, ", ")) not in the graph $(g.path)"))
    return (; stages = [n for n in want if n in have], dropped = [n for n in want if !(n in have)])
end

"""
    ror_missing(plan, cfg) -> Vector{String}

The experiments of `[[require]]` the launch does not hold: a stage it does not run (nor reuse), or no item of it matching the
requirement's `item` regex (the fisheye 30 cm set before the schoeps-window merge, say). A requirement with `optional = true`
whose stage the graph lacks is not missing (the release left that experiment out).
"""
function ror_missing(plan, cfg)
    out = String[]
    for r in get(cfg, "require", Any[])
        i = findfirst(st -> st.spec.name == r["stage"], plan.stages)
        if i === nothing
            get(r, "optional", false) && !any(s -> s.name == r["stage"], plan.graph.stages) && continue
            push!(out, "$(r["what"]): stage $(r["stage"]) not in the launch")
        elseif haskey(r, "item") && !any(it -> occursin(Regex(r["item"]), it), plan.stages[i].items)
            push!(out, "$(r["what"]): no item of stage $(r["stage"]) matches /$(r["item"])/ (items: $(join(plan.stages[i].items, ", ")))")
        end
    end
    return out
end

_hm(m) = (m = round(Int, m); m >= 60 ? "$(m ÷ 60)h$(lpad(m % 60, 2, '0'))" : "$(m)m")

"""
    ror_report(io, plan, jobs, sched, cfg; missing = ror_missing(plan, cfg))

The plan of the run of record: per job its name, device, CPUs per task, tasks (and an array's items: task i is item i), estimate
per task, limit and projected window; the totals (GPU jobs and tasks, CPU jobs and tasks, GPU- and CPU-hours); the projected
wall-clock at full capacity with the GPUs' occupancy; the jobs at risk; and the required experiments the launch lacks.
"""
function ror_report(io::IO, plan, jobs, sched, cfg; missing = ror_missing(plan, cfg))
    m = get(cfg, "machine", Dict{String, Any}())
    println(io, "\n== run of record: the job graph ($(length(jobs)) jobs, submission order: the paper first, then the longest remaining chain first)")
    println(io, rpad("job", 66), rpad("dev", 5), lpad("cpu", 4), lpad("tasks", 6), lpad("est/task", 9), lpad("limit", 7), "  window (from launch)")
    for (k, j) in enumerate(jobs)
        flag = j.limit > 0 && j.minutes > j.limit ? "  !! EXCEEDS" : j.timed || j.phase == "precompile" ? "" : "  (unmeasured)"
        println(io, rpad(j.name, 65), " ", rpad(j.gpu ? "GPU" : "CPU", 5), lpad(j.cpus, 4), lpad(j.tasks, 6), lpad(_hm(j.minutes), 9),
                lpad(j.limit > 0 ? _hm(j.limit) : "-", 7), "  ", _hm(sched.start[k]), " → ", _hm(sched.finish[k]), flag)
        shown = k > 1 && jobs[k - 1].stage == j.stage && !isempty(jobs[k - 1].items) && jobs[k - 1].phase != "precompile"
        isempty(j.items) || j.phase == "precompile" || shown ||
            println(io, " "^6, "tasks: ", join(("$(i - 1)=$(it)" for (i, it) in enumerate(j.items)), " "))
    end
    gj = filter(j -> j.gpu, jobs); cj = filter(j -> !j.gpu, jobs)
    println(io, "\n== totals")
    println(io, "GPU jobs $(length(gj)) ($(sum(j.tasks for j in gj; init = 0)) tasks, one GPU each), ",
            "CPU jobs $(length(cj)) ($(sum(j.tasks for j in cj; init = 0)) tasks, --gres none)")
    println(io, "GPU-hours $(round(sched.gpu_minutes / 60; digits = 1)), CPU-hours $(round(sched.cpu_minutes / 60; digits = 1))")
    println(io, "projected wall-clock at full capacity ($(get(m, "gpus", 8)) GPUs, $(get(m, "cpus", 224)) CPUs): $(_hm(sched.makespan)) ",
            "(GPU occupancy $(round(Int, 100 * sched.busy_gpu)) %)")
    k = argmax(sched.finish)
    println(io, "the tail: $(jobs[k].name), ending at $(_hm(sched.finish[k]))")
    risks = ror_risks(jobs, sched, cfg)
    println(io, "\n== at risk ($(count(r -> r.level != "UNMEASURED", risks)) over or near their limit, $(count(r -> r.level == "UNMEASURED", risks)) unmeasured)")
    for r in sort(risks; by = r -> r.level)
        println(io, rpad(r.level, 11), rpad(r.name, 65), " ", r.why)
    end
    println(io, "\n== required experiments: ", isempty(missing) ? "all in the launch" : "$(length(missing)) MISSING")
    foreach(x -> println(io, "MISSING    ", x), missing)
    return risks
end

"""
    ror_manifest_leaks(manifest, tree) -> Vector{String}

The entries of an instantiated `Manifest.toml` that are not registered versions: a `path` outside `tree` (a dev path, a scratch
`[sources]`) or a `repo-url` (a package pinned to a git branch). A run of record resolves every package but the project's own
from the registries; its own (RadiometricCamera, `path` inside the tree) is the registered version's code, which the launcher
checks against the registry's tree hash.
"""
function ror_manifest_leaks(manifest::AbstractString, tree::AbstractString)
    d = TOML.parsefile(manifest)
    root = normpath(abspath(tree)) * "/"
    out = String[]
    for (name, entries) in get(d, "deps", Dict{String, Any}()), e in entries
        if haskey(e, "path")
            p = normpath(joinpath(dirname(abspath(manifest)), e["path"]))
            startswith(p * "/", root) || push!(out, "$name: path $(e["path"]) (outside the tree)")
        end
        haskey(e, "repo-url") && push!(out, "$name: repo-url $(e["repo-url"])")
    end
    return sort(out)
end
