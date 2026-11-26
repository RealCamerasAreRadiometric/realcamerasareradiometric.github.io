#!/usr/bin/env julia
# reproduce: reads configs/experiments/chrome_index_grid.toml
# provenance: every product writes _config.toml and _provenance.toml (VisualGeometryToolkitEval produce) beside its results.
# =============================================================================
# scripts/chrome_index_grid.jl — the chrome index stage's comparison grid: every chrome set × every variant of the index stage's
# RANSAC scoring of configs/experiments/chrome_index_grid.toml (each variant a scorer table and an index local-optimization table of
# the current schema), against its baseline. A measurement, not a figure: the configs of record (chrome_evaluation_fx3_*.toml)
# state one variant. A stage of configs/reproduce.toml (`chrome_index_grid`, default = false).
#
#   julia --project=experiments experiments/scripts/chrome_index_grid.jl [--config=<toml>] [--plan] [--stage=detect,index,evaluate] [--set=<set>] [--force]
#
# STAGES (each run keyed by its config record and the keys of the runs it reads; <results>/<experiment>/<stage>/<key>/)
#   detect    scripts/chrome_evaluation.jl's prepare and detect stages on every set's configs of record (their detect runs do not
#             depend on the variant, so every task of the set shares them; GPU)
#   index     per task (a set under a variant): every config of the set rewritten with the variant's [index] scoring and
#             [index.local_optimization_strategy] (<task run>/configs/), its index and evaluate stages run in this process
#             (src/chrome_evaluation.jl register_photos and evaluate_models: their own keyed runs, reused when complete) — the
#             index stage counts every local optimization (Estimation's CountedLO), so each photograph is registered once:
#             <task run>/rows.json — per photograph its registration (inliers, matched, rms, the detections assigned, the
#             seconds), the index's wall time, every LO call (the index run's photos.json), and every evaluate run's
#             rotation-centre rows with their refine's iterations and retcode. Every variant of a set runs in the one process (the
#             code compiled once per set); `--set=<set>` runs that set's tasks alone (chrome_index_grid.sbatch's array, one task
#             per set)
#   evaluate  the table run: table.md and table.json from every task's rows — per set and variant the registrations, the wall
#             time, the LO iterations and retcodes, the assignment differences from the baseline, the evaluate rows, and the
#             evaluate rows of each variant's registrations of the baseline's evaluated photographs (scored here: common/)
#
# Data: VGTK_DATA_ROOT and VGTK_RESULTS_ROOT bind data_root() and results_root() (REPRODUCE.md). The index and evaluate stages run
# on the CPU; the detect stage on the GPU of the configs of record ([detect] device).
# =============================================================================
using DrWatson
@quickactivate "experiments"
include(srcdir("experiments.jl"))
include(srcdir("drivers.jl"))
include(srcdir("radiometric_refinement.jl"))
include(srcdir("chrome_evaluation.jl"))
using VisualGeometryToolkitEval.Experiments: run_isolated
using Statistics: median, mean
using Printf: @sprintf

const DRIVER = "scripts/chrome_index_grid.jl"
const CFG_PATH = abspath(cli_option(ARGS, "config", projectdir("configs", "experiments", "chrome_index_grid.toml")))   # default: the config of record
const CFG = let c = load_driver_config(CFG_PATH, "chrome_index_grid"; tables = ("experiment", "inputs", "grid", "variants"))
    isempty(c["inputs"]) || throw(ArgumentError("$CFG_PATH [inputs]: the grid reads the chrome sets' configs, it has no input of its own"))
    check_required_keys(c["grid"], ("sets", "baseline", "models"), "$CFG_PATH [grid]")
    c["grid"]["models"] in ("inputs", "all") || throw(ArgumentError("$CFG_PATH [grid] models: \"$(c["grid"]["models"])\" is not inputs or all"))
    for (n, v) in c["variants"]
        check_required_keys(v, ("scoring", "local_optimization_strategy"), "$CFG_PATH [variants.$n]")
    end
    haskey(c["variants"], c["grid"]["baseline"]) || throw(ArgumentError("$CFG_PATH [grid] baseline: no variant $(c["grid"]["baseline"])"))
    c
end
const SETS = String.(CFG["grid"]["sets"])
const VARIANTS = sort(collect(keys(CFG["variants"])))
const TASKS = [(s, v) for s in SETS for v in VARIANTS]
"The tasks `--set=<set>` selects (that set's, every variant in this process; every task without it)."
const TASK_SELECTION = let set = cli_option(ARGS, "set", "")
    isempty(set) || set in SETS || throw(ArgumentError("--set=$set: not a [grid] set ($(join(SETS, ", ")))"))
    isempty(set) ? TASKS : [(s, v) for (s, v) in TASKS if s == set]
end
const CHROME_DRIVER = projectdir("scripts", "chrome_evaluation.jl")

"The config of record of chrome set `set` (its file name; one per set, its models a list), as a one-element list."
set_config(set) = "chrome_evaluation_fx3_$set.toml"

"""Config `f` of record with variant `v`'s [index] tables (each variant config is checked as the driver checks it, so a variant of a
retired schema is refused by name: `index_settings`). With `[grid] models = "inputs"` its [evaluate] scores the models of record
inputs alone (the starts), not the refinement cases."""
const VARIANT_CFGS = Dict{Tuple{String, String}, Dict{String, Any}}()
variant_config(v, f) = get!(() -> _variant_config(v, f), VARIANT_CFGS, (String(v), String(f)))   # memoised: read and checked once
function _variant_config(v, f)
    c = load_chrome_evaluation_config(projectdir("configs", "experiments", f))
    c["index"]["scoring"] = deepcopy(CFG["variants"][v]["scoring"])
    c["index"]["local_optimization_strategy"] = deepcopy(CFG["variants"][v]["local_optimization_strategy"])
    CFG["grid"]["models"] == "inputs" && haskey(c, "evaluate") && filter!(m -> haskey(m, "input"), c["evaluate"]["models"])
    index_settings(c, "$CFG_PATH [variants.$v] on $f")
    return c
end

"""
    task_product(set, v) -> Product

The product of task (`set`, `v`) (its output rows.json): keyed by the variant's tables, `[grid] models` and the keys of the index
and evaluate products of every config of the set under the variant (`upstream`; chrome_evaluation.jl's own products).
"""
const TASK_PRODUCTS = Dict{Tuple{String, String}, Product}()
task_product(set, v) = get!(() -> _task_product(set, v), TASK_PRODUCTS, (String(set), String(v)))   # memoised: every stage's plan reads it
function _task_product(set, v)
    up = Dict{String, Any}()
    let f = set_config(set)
        c = variant_config(v, f)
        up[splitext(f)[1]] = merge(Dict{String, Any}("index" => index_product(c).key),
                                   haskey(c, "evaluate") ? Dict{String, Any}("evaluate" => evaluate_product(c).key) : Dict{String, Any}())
    end
    return driver_product("index", Dict{String, Any}("experiment" => CFG["experiment"], "set" => set, "variant" => v,
                                                     "variant_tables" => CFG["variants"][v], "models" => CFG["grid"]["models"], "upstream" => up);
                          output = "rows.json")
end

"The table product (its output table.json): keyed by `[grid]` and the keys of every task's product."
table_product() = driver_product("evaluate", Dict{String, Any}("experiment" => CFG["experiment"], "grid" => CFG["grid"],
                                                               "upstream" => Dict{String, Any}("$s/$v" => task_product(s, v).key for (s, v) in TASKS));
                                 output = "table.json")

"Run scripts/chrome_evaluation.jl's `stages` on the config `path`, in a module of its own (the detect stage: once per set)."
chrome_stages(path, stages) = run_isolated(CHROME_DRIVER; args = ["--config=$path", "--stage=$stages"])

"The rotation-centre rows of the evaluate run `dir`: per model its summary columns, the refine's iterations, retcode and loss width, the start's translation residual, its photographs."
function rotation_center_rows(dir)
    out = Dict{String, Any}[]
    for row in read_summary(joinpath(dir, "summary.tsv"))                 # the evaluate run's table (VisualGeometryToolkitEval's)
        row["variant"] == "motion" || continue
        lab = String(row["model"])
        rec = read_chrome_evaluation(joinpath(dir, "eval_$lab.json"))     # v2.0 (a v1.0 record through its version step)
        f = rec["fit"]["in_sample"]
        push!(out, Dict{String, Any}("model" => lab, (k => v for (k, v) in row if !(k in ("model", "variant")))...,
                                     "its" => f["stats"]["nsteps"], "retcode" => string(f["retcode"]), "loss_width_px" => f["loss_width_px"],
                                     "translation_residual_mm" => median(rec["motion"]["summary"]["start_translation_residual_mm"]),
                                     "photos_list" => [p["photo"] for p in rec["motion"]["per_photo"]]))
    end
    return out
end

# ---- detect -------------------------------------------------------------------------------------------------------------------
"The detect product of a set's config of record (scripts/chrome_evaluation.jl's)."
set_detect_product(set) = detect_product(load_chrome_evaluation_config(projectdir("configs", "experiments", set_config(set))))
"The set's prepare and detect stages, by its driver."
detect_set_stages(set) = chrome_stages(projectdir("configs", "experiments", set_config(set)), "prepare,detect")

# ---- index: the tasks -----------------------------------------------------------------------------------------------------------
task(p) = produce(p; product_records(DRIVER, CFG_PATH)...) do dir
    set, v = p.record["set"], p.record["variant"]
    println(">> task $set under $v → $dir"); flush(stdout)
    fs = [set_config(set)]
    evals = Dict{String, Any}()
    runs = String[]
    index = nothing
    for f in fs
        cfg = variant_config(v, f)
        cp = joinpath(mkpath(joinpath(dir, "configs")), f)
        open(io -> TOML.print(io, cfg), cp, "w")
        r = register_photos(cfg, cp; driver = DRIVER)              # the index stage, registered once (its photos.json: the counted rows)
        idir = r.dir
        index === nothing && (index = (; r.rows, r.wall, dir = idir))
        push!(runs, idir)
        if haskey(cfg, "evaluate")
            e = evaluate_models(cfg, cp; driver = DRIVER)          # produced or loaded
            push!(runs, e)
            evals[f] = Dict("dir" => e, "index" => idir, "detect" => product_dir(detect_product(cfg)), "config" => cp, "rows" => rotation_center_rows(e))
        end
    end
    rec = Dict{String, Any}("set" => set, "variant" => v, "variant_tables" => CFG["variants"][v], "index_run" => index.dir,
                            "wall_s" => index.wall, "threads" => execution_threads(), "photos" => index.rows, "evaluate" => evals)
    VgtkIO.write_record(joinpath(dir, "rows.json"), merge(Dict{String, Any}("schema" => EXPERIMENT_RECORD_SCHEMAS.chrome_index_grid_rows), rec); allownan = true)                         # VisualGeometryToolkitIO's record writer
    println(">> task $set under $v → ", joinpath(dir, "rows.json"))
    Produced(; inputs = runs)
end

# ---- the table ------------------------------------------------------------------------------------------------------------------
_q(x) = x isa Real ? @sprintf("%.4g", x) : string(x)

"Variant `r`'s evaluate run of config `f` scored on the photographs the baseline `b` scored (its registrations of them alone)."
function common_eval(out, set, v, f, r, b)
    e, eb = r["evaluate"][f], b["evaluate"][f]
    names = unique(reduce(vcat, [x["photos_list"] for x in eb["rows"]]; init = String[]))
    idir = joinpath(out, "common", set, v, splitext(f)[1], "index"); rm(idir; recursive = true, force = true)
    foreach(d -> mkpath(joinpath(idir, d)), ("registration", "points", "poses"))
    for n in names, (d, ext) in (("registration", ".tsv"), ("registration", "_scale.json"), ("points", ".json"), ("poses", ".json"))
        src = joinpath(e["index"], d, n * ext)
        isfile(src) && symlink(src, joinpath(idir, d, n * ext))
    end
    edir = joinpath(out, "common", set, v, splitext(f)[1], "eval")
    evaluate_models(load_chrome_evaluation_config(e["config"]), e["config"]; driver = DRIVER, index_dir = idir, out = edir)
    return rotation_center_rows(edir)
end

table(p) = produce(p; product_records(DRIVER, CFG_PATH)..., inputs = [product_dir(task_product(s, v)) for (s, v) in TASKS]) do out
    tasks = [product_dir(task_product(s, v)) for (s, v) in TASKS]
    R = Dict((r["set"], r["variant"]) => r for r in (VgtkIO.read_record(joinpath(d, "rows.json"); allownan = true) for d in tasks if isfile(joinpath(d, "rows.json"))))
    base = CFG["grid"]["baseline"]
    md = String[]
    push!(md, "# The chrome index grid ($(CFG_PATH))", "", "Baseline: `$base`. Per set and variant; missing tasks are listed.", "")
    missing_tasks = [(s, v) for (s, v) in TASKS if !haskey(R, (s, v))]
    isempty(missing_tasks) || push!(md, "Missing: " * join(("$s/$v" for (s, v) in missing_tasks), ", "), "")
    push!(md, "## The index", "", "| set | variant | registered | inliers / matched per photo | rms px (median) | index wall s | LO calls per photo | iterations: histogram | retcodes | assignment vs baseline |",
          "|---|---|---|---|---|---|---|---|---|---|")
    summary = Dict{String, Any}[]
    for (s, v) in TASKS
        haskey(R, (s, v)) || continue
        r = R[(s, v)]; ph = r["photos"]; ok = filter(p -> p["ok"], ph)
        calls = reduce(vcat, [p["lo"] for p in ph]; init = Any[])
        its = [Int(c[1]) for c in calls]                                   # (a record's numbers read back as Float64)
        hist = join(("$k:$(count(==(k), its))" for k in sort(unique(its))), " ")
        retcodes = join(("$t:$(count(c -> c[2] == t, calls))" for t in sort(unique(String[c[2] for c in calls]))), " ")
        diff = "—"
        if haskey(R, (s, base)) && v != base
            bp = Dict(p["photo"] => p for p in R[(s, base)]["photos"])
            changed = 0; gained = 0; lost = 0; newly = 0; dropped = 0
            for p in ph
                q = bp[p["photo"]]
                p["ok"] && !q["ok"] && (newly += 1); !p["ok"] && q["ok"] && (dropped += 1)
                (p["ok"] && q["ok"]) || continue
                a, b = Set(p["assigned"]), Set(q["assigned"])
                a == b && continue
                changed += 1; gained += length(setdiff(a, b)); lost += length(setdiff(b, a))
            end
            diff = "$changed photos differ (+$gained −$lost dots); +$newly / −$dropped registered"
        end
        push!(md, "| $s | $v | $(length(ok))/$(length(ph)) | $(_q(mean(p["inliers"] for p in ok))) / $(_q(mean(p["matched"] for p in ok))) | " *
                  "$(_q(median(p["rms"] for p in ok))) | $(_q(r["wall_s"])) | $(_q(length(calls) / length(ph))) | $hist | $retcodes | $diff |")
        push!(summary, Dict("set" => s, "variant" => v, "registered" => length(ok), "photos" => length(ph), "wall_s" => r["wall_s"],
                            "iterations" => its, "retcodes" => [c[2] for c in calls]))
    end
    push!(md, "", "## The evaluation (the rotation-centre rows): every model, on the variant's own photographs, and on the baseline's (→)", "",
          "| set | config | model | variant | photos | dots | median | mean | p95 | rms | its (retcode) | translation residual mm | on the baseline's photos: photos, median, mean, p95, rms |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for (s, v) in TASKS
        haskey(R, (s, v)) || continue
        r = R[(s, v)]
        for f in sort(collect(keys(r["evaluate"])))
            common = (v != base && haskey(R, (s, base))) ? common_eval(out, s, v, f, r, R[(s, base)]) : nothing
            for x in r["evaluate"][f]["rows"]
                c = common === nothing ? nothing : findfirst(y -> y["model"] == x["model"], common)
                cs = c === nothing ? "—" : (y = common[c]; "$(y["photos"]), $(_q(y["median_px"])), $(_q(y["mean_px"])), $(_q(y["p95_px"])), $(_q(y["rms_px"]))")
                push!(md, "| $s | $(splitext(f)[1]) | $(x["model"]) | $v | $(Int(x["photos"])) | $(Int(x["dots"])) | $(_q(x["median_px"])) | $(_q(x["mean_px"])) | " *
                          "$(_q(x["p95_px"])) | $(_q(x["rms_px"])) | $(Int(x["its"])) ($(x["retcode"])) | $(_q(x["translation_residual_mm"])) | $cs |")
            end
        end
    end
    write(joinpath(out, "table.md"), join(md, "\n") * "\n")
    VgtkIO.write_record(joinpath(out, "table.json"), (; schema = EXPERIMENT_RECORD_SCHEMAS.chrome_index_grid_table, rows = summary); allownan = true)
    println(">> table → ", joinpath(out, "table.md"))
    nothing
end

run_stages("chrome_index_grid", [
    Stage(:detect, "every set's prepare and detect stages (scripts/chrome_evaluation.jl; GPU)"; items = () -> SETS, product = set_detect_product, run = detect_set_stages),
    Stage(:index, "every task: the set's index and evaluate stages under the variant, the counted registrations (rows.json)";
          items = () -> TASK_SELECTION, product = ((s, v),) -> task_product(s, v), run = ((s, v),) -> task(task_product(s, v))),
    Stage(:evaluate, "the table of every task against the baseline (table.md, table.json)"; items = () -> [table_product()], product = identity, run = table),
])
