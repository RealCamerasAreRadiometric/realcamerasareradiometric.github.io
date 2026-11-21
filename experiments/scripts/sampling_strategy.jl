#!/usr/bin/env julia
# reproduce: reads configs/studies/sampling_strategy.toml
# provenance: every product writes _config.toml and _provenance.toml (VisualGeometryToolkitEval produce) beside its results.
# =============================================================================
# scripts/sampling_strategy.jl — the sampling-strategy experiment (config of record configs/studies/sampling_strategy.toml): per
# case of record and variant of the LM's sample draw (RadiometricCamera's draw_samples: uniform, the rule of 2026-10-31, the ECDF draw
# at α = 1 and 2, the uniform draw plus an edge-biased add-on) a refinement — an OVERRIDE of the case of record (src/drivers.jl
# expand_overrides) — scored on the lens's Thorlabs chrome photographs (the chrome evaluation of record's protocol and its paired
# photograph bootstrap) and, for the 14 mm, by the texture the camera recovers (the texture experiment's CV pipeline and compare).
#
# Every step is its OWNER's (Eval's run_isolated: the driver of record in a module of its own, on a config this driver writes into
# a keyed product of its own, generated from the config of record of this experiment and the owners' configs of record alone):
#   configs   per case the refinement config of the variants: the case of record's table with each variant's overrides
#             (expand_overrides, one value each; the cases named as expand_overrides names them) and the context cases of record
#             verbatim — refinement_<case>.toml (the refinement driver's config: radiometric_refinement_ss_<case>.toml)   (CPU)
#   prepare   per case its photographs' prepare and detect (scripts/radiometric_refinement.jl) and its image region
#             (scripts/image_regions.jl), once before the arms (the arms run packed and never produce a shared upstream twice) (GPU)
#   refine    per case and variant (and context case): the refinement driver's start, then THE LATTICE GUARD (the start record's
#             resample: its subdivision must be the case's, its lattice at most its max_nodes; refused otherwise — nothing runs on
#             the slow lattice), then the refinement driver's refine; "grid: N nodes, m = …" logged and recorded            (GPU)
#   the scoring, texture, record and figure stages follow (this file; they read the refinements by their keys)
#
#   julia --project=experiments experiments/scripts/sampling_strategy.jl [--config=<toml>] [--plan] [--stage=NAME[,NAME]] [--case=a] [--variant=a,b] [--force]
#
# --case / --variant select the arms (a packed GPU job runs one arm per process). The thread count keys every refinement (src/drivers.jl
# execution_threads): every stage of one production runs at the same JULIA_NUM_THREADS.
# The experiment's own code lives in THIS file, not in src/: the project's src keys every product of every driver (CODE_VERSION), so a
# change to the scoring or the figure never re-keys the refinements it reads (its two record schemas are src/record_schemas.jl's, the
# project's one table).
# =============================================================================
using DrWatson
@quickactivate "experiments"
include(srcdir("experiments.jl"))
include(srcdir("drivers.jl"))
include(srcdir("radiometric_refinement.jl"))
include(srcdir("chrome_evaluation.jl"))
include(srcdir("figure_style.jl"))
include(srcdir("texture_recovery.jl"))                                       # its record reader `plain` (the compare's records)
using VisualGeometryToolkitEval.Experiments: run_isolated
using Statistics: median, mean, quantile
import PNGFiles
using StaticArrays: SVector
import TOML
using Printf: @printf, @sprintf

const DRIVER = "scripts/sampling_strategy.jl"

# ---- the config of record ---------------------------------------------------------------------------------------------------
const SS_TABLES = ("experiment", "inputs", "sampling_strategy", "variants", "cases", "runtime", "figure")
const SS_KEYS = ("cases", "variants", "reference", "headline", "label", "dome_tolerance", "lm_max_iter")
const SS_REQUIRED = ("cases", "variants", "reference", "headline", "label", "dome_tolerance")
const SS_CASE_KEYS = ("refinement", "case", "chrome", "context", "texture", "texture_crop_mm", "lattice", "variants", "lattice_override")
const SS_CASE_REQUIRED = ("refinement", "case", "chrome", "context")   # `lattice`: a spline case's (its start lays a lattice); a parametric case states none
const SS_LATTICE_KEYS = ("subdivision", "max_nodes")

"Deep merge of TOML tables: `b`'s keys over `a`'s, tables merged recursively (a smoke config over its base)."
_merge(a::AbstractDict, b::AbstractDict) = (o = Dict{String, Any}(a); for (k, v) in b; o[k] = v isa AbstractDict && get(o, k, nothing) isa AbstractDict ? _merge(o[k], v) : v; end; o)

"""
    load_sampling_strategy_config(path) -> Dict

The config of record (or a config whose `[experiment] base` names it, its tables merged over the base's: a smoke run), every table
and key checked: an unknown or inapplicable key is an error.
"""
function load_sampling_strategy_config(path::AbstractString)
    raw = TOML.parsefile(path)
    e = get(raw, "experiment", Dict{String, Any}())
    if haskey(e, "base")
        base = load_sampling_strategy_config(projectdir(e["base"]))
        raw = _merge(base, Dict{String, Any}(k => v for (k, v) in raw if k != "experiment"))
        raw["experiment"] = base["experiment"]
    end
    VgtkCore.check_keys(raw, SS_TABLES, path)
    VgtkCore.check_keys(raw["experiment"], ("name",), "$path [experiment]")
    raw["experiment"]["name"] == "sampling_strategy" || throw(ArgumentError("$path: [experiment] name must be \"sampling_strategy\""))
    s = raw["sampling_strategy"]; w = "$path [sampling_strategy]"
    VgtkCore.check_keys(s, SS_KEYS, w); foreach(k -> VgtkCore.required_key(s, k, w), SS_REQUIRED)
    all(v -> haskey(raw["variants"], v), s["variants"]) && Set(s["variants"]) == Set(keys(raw["variants"])) ||
        throw(ArgumentError("$w variants: exactly the [variants.<name>] tables"))
    s["reference"] in s["variants"] && all(in(s["variants"]), s["headline"]) || throw(ArgumentError("$w: reference and headline are variants"))
    s["dome_tolerance"] isa Real && 0 < s["dome_tolerance"] < 1 || throw(ArgumentError("$w dome_tolerance: a fraction in (0, 1) (the dome's ± deviation from the mean focus distance)"))
    haskey(s, "lm_max_iter") && !(s["lm_max_iter"] isa Integer && s["lm_max_iter"] >= 1) && throw(ArgumentError("$w lm_max_iter: an integer ≥ 1 (a smoke run's LM cap)"))
    for (v, t) in raw["variants"]
        t isa AbstractDict && !isempty(t) && all(k -> occursin(r"^[a-z_]+\.[a-z_]+$", k), keys(t)) ||
            throw(ArgumentError("$path [variants.$v]: `\"<table>.<key>\" = value`, the [pipeline] keys the variant sets"))
    end
    all(c -> haskey(raw["cases"], c), s["cases"]) || throw(ArgumentError("$w cases: each a [cases.<name>]"))
    for (c, t) in raw["cases"]
        wc = "$path [cases.$c]"
        VgtkCore.check_keys(t, SS_CASE_KEYS, wc); foreach(k -> VgtkCore.required_key(t, k, wc), SS_CASE_REQUIRED)
        for k in ("refinement", "chrome", "texture")
            haskey(t, k) && (isfile(projectdir(t[k])) || throw(ArgumentError("$wc $k: no config $(t[k])")))
        end
        haskey(t, "texture") == haskey(t, "texture_crop_mm") ||
            throw(ArgumentError("$wc texture_crop_mm: stated exactly when the case has a texture config (the crop of its [compare])"))
        haskey(t, "texture_crop_mm") && !(t["texture_crop_mm"] isa AbstractVector && length(t["texture_crop_mm"]) == 4 && all(x -> x isa Real, t["texture_crop_mm"])) &&
            throw(ArgumentError("$wc texture_crop_mm: [x0, y0, width, height] in mm"))
        haskey(t, "lattice") && check_lattice(t["lattice"], "$wc lattice")
        for (v, lo) in get(t, "lattice_override", Dict{String, Any}())
            v in s["variants"] || v in t["context"] || throw(ArgumentError("$wc lattice_override.$v: not a variant or a context model of the case"))
            check_lattice(lo, "$wc lattice_override.$v")
        end
        for (v, o) in get(t, "variants", Dict{String, Any}())
            v in s["variants"] || throw(ArgumentError("$wc variants.$v: not a variant"))
            all(k -> occursin(r"^[a-z_]+\.[a-z_]+$", k), keys(o)) || throw(ArgumentError("$wc variants.$v: `\"<table>.<key>\" = value`"))
        end
    end
    VgtkCore.check_keys(raw["runtime"], ("reference_s_per_iteration", "regression_factor"), "$path [runtime]")
    VgtkCore.check_keys(raw["figure"], ("asset", "table"), "$path [figure]")
    return raw
end
function check_lattice(l, where)
    l isa AbstractDict || throw(ArgumentError("$where: {subdivision = [m_x, m_y], max_nodes = N}"))
    VgtkCore.check_keys(l, SS_LATTICE_KEYS, where); foreach(k -> VgtkCore.required_key(l, k, where), SS_LATTICE_KEYS)
    l["subdivision"] isa AbstractVector && length(l["subdivision"]) == 2 && all(x -> x isa Integer && x >= 1, l["subdivision"]) &&
        l["max_nodes"] isa Integer && l["max_nodes"] >= 1 || throw(ArgumentError("$where: subdivision two integers ≥ 1, max_nodes an integer ≥ 1"))
end

const CFG_PATH = abspath(cli_option(ARGS, "config", projectdir("configs", "studies", "sampling_strategy.toml")))   # default: the config of record
const CFG = load_sampling_strategy_config(CFG_PATH)
const SS = CFG["sampling_strategy"]
const CASES = cli_subset(ARGS, "case", SS["cases"]; what = "case")
"The arms --variant selects (variants or context labels; none: every arm)."
const ARM_SELECTION = (v = cli_option(ARGS, "variant", ""); isempty(v) ? nothing : String.(split(v, ',')))
const VARIANTS = ARM_SELECTION === nothing ? SS["variants"] : filter(in(ARM_SELECTION), SS["variants"])
const LABEL = SS["label"]

# ---- the variants as overrides of the cases of record ------------------------------------------------------------------------
"The [pipeline] keys variant `v` sets on case `c`: the variant's, the case's own for it, and a smoke run's LM cap."
function variant_overrides(c::AbstractString, v::AbstractString)
    o = merge(Dict{String, Any}(CFG["variants"][v]), Dict{String, Any}(get(get(CFG["cases"][c], "variants", Dict()), v, Dict())))
    haskey(SS, "lm_max_iter") && (o["lm.max_iter"] = SS["lm_max_iter"])
    return o
end

"The chrome config of record of case `c`, parsed (its models name the context)."
const CHROME_RECORDS = Dict{String, Any}()   # the chrome configs of record, parsed once per case
chrome_record(c) = get!(() -> TOML.parsefile(projectdir(CFG["cases"][c]["chrome"])), CHROME_RECORDS, c)
"The context models of case `c`: the chrome config of record's `[evaluate] models` entries named by the case's `context`."
function context_models(c)
    ms = chrome_record(c)["evaluate"]["models"]
    out = map(CFG["cases"][c]["context"]) do l
        i = findfirst(m -> m["label"] == l, ms)
        i === nothing && throw(ArgumentError("$CFG_PATH [cases.$c] context: no model $l in $(CFG["cases"][c]["chrome"])"))
        ms[i]
    end
    return out
end

"""
    refinement_config(c) -> (; cfg, names)

The refinement config of case `c`: its config of record (`refinement`) with `cases` the variants of `VARIANTS`' superset (every variant,
so one file serves every arm) — each the case of record with the variant's overrides, through expand_overrides (one value each: the
case's own name when the overrides are its own values, else `<case>__<key>=<value>…`) — and the context's case models, verbatim
(under a smoke run with its LM cap). `names`: variant → case name, and each context case's own.
"""
function refinement_config(c::AbstractString)
    t = CFG["cases"][c]; bpath = t["refinement"]
    base = inherit_kinds!(TOML.parsefile(projectdir(bpath)), bpath)
    names = Dict{String, String}(); cases = Dict{String, Any}(); locations = Dict{String, Any}()
    expand(case, ov) = (e = expand_overrides(Dict{String, Any}("experiment" => Dict{String, Any}("name" => base["experiment"]["name"], "base" => bpath, "cases" => [case]),
                                                              "overrides" => Dict{String, Any}(k => [x] for (k, x) in ov)), CFG_PATH);
                        merge!(locations, get(e, "_locations", Dict{String, Any}()));    # an arm lies in the study's directory (`variant_location`)
                        only(e["cases"]))
    for v in SS["variants"]
        n, tb = expand(t["case"], variant_overrides(c, v))
        names[v] = n; cases[n] = tb
    end
    for m in context_models(c)
        haskey(m, "case") || continue
        haskey(base["cases"], m["case"]) || throw(ArgumentError("$CFG_PATH [cases.$c] context $(m["label"]): case $(m["case"]) is not in $bpath"))
        n, tb = haskey(SS, "lm_max_iter") ? expand(m["case"], Dict{String, Any}("lm.max_iter" => SS["lm_max_iter"])) : (m["case"], base["cases"][m["case"]])
        names[m["label"]] = n; cases[n] = tb
    end
    out = deepcopy(base); out["cases"] = cases
    isempty(locations) || (out["_locations"] = locations)
    # the front-end runs and the photograph sets the arms read, alone: the refinement driver checks (and its prepare decodes) every
    # [runs] and [prepare] entry of its config, and a set or run no arm reads is not this experiment's upstream. Neither keys a run
    # (a start is keyed by its own table's inputs and runs: `start_product`)
    used = Set(id for tb in values(cases) for id in table_runs(tb["pipeline"]))
    haskey(out, "runs") && (out["runs"] = Dict{String, Any}(id => r for (id, r) in out["runs"] if id in used))
    sets = Set(String(tb["images"]) for tb in values(cases))
    out["prepare"] = Dict{String, Any}(k => v for (k, v) in out["prepare"] if k in sets)
    return (; cfg = out, names)
end

"The configs of record this experiment's configs are generated from (path => sha256): its key."
source_files() = sort(unique([CFG_PATH; [projectdir(CFG["cases"][c][k]) for c in SS["cases"] for k in ("refinement", "chrome", "texture") if haskey(CFG["cases"][c], k)]]))
sources_record() = Dict{String, Any}(relpath(f, projectdir()) => open(VgtkCore.sha256_hex, f) for f in source_files() if f != CFG_PATH)

configs_product() = driver_product("configs", Dict{String, Any}("experiment" => CFG["experiment"], "sampling_strategy" => SS, "variants" => CFG["variants"],
                                                                "cases" => Dict(c => CFG["cases"][c] for c in SS["cases"]), "sources" => sources_record(),
                                                                "generator" => "pruned runs and prepare sets (2026-11-15)");
                                   output = "configs.json", threads = nothing)
"The refinement driver's config of case `c` (in the configs product)."
refinement_path(c) = joinpath(product_dir(configs_product()), "radiometric_refinement_ss_$c.toml")

function run_configs(p)
    produce(p; product_records(DRIVER, CFG_PATH)...) do dir
        rec = Dict{String, Any}("label" => LABEL, "cases" => Dict{String, Any}())
        for c in SS["cases"]
            r = refinement_config(c)
            f = joinpath(dir, "radiometric_refinement_ss_$c.toml")
            open(io -> TOML.print(io, r.cfg; sorted = true), f, "w")
            load_radiometric_refinement_config(f)                              # the owner's loader accepts it
            rec["cases"][c] = Dict{String, Any}("refinement" => basename(f), "names" => r.names)
            println(">> configs: $c → ", basename(f), ": ", join(("$v = $(r.names[v])" for v in SS["variants"]), "; "))
        end
        VgtkIO.write_record(joinpath(dir, "configs.json"), merge(Dict{String, Any}("schema" => EXPERIMENT_RECORD_SCHEMAS.sampling_strategy_configs), rec))
        Produced(; inputs = source_files())
    end
end

"The refinement config of case `c` as its driver loads it (once), and the case name of `arm` (a variant or a context label)."
const LOADED = Dict{String, Any}()
loaded_refinement(c) = get!(() -> load_radiometric_refinement_config(refinement_path(c)), LOADED, c)
const NAMES = Dict{String, Dict{String, String}}()
arm_case(c, arm) = get!(() -> refinement_config(c).names, NAMES, c)[arm]          # the names expand_overrides gives (configs.json records them)
"The arms of case `c`: the selected variants and, without --variant, the context's case models (their labels)."
function arms(c)
    all_ = [SS["variants"]; [m["label"] for m in context_models(c) if haskey(m, "case")]]
    ARM_SELECTION === nothing && return all_
    bad = setdiff(ARM_SELECTION, union(all_, (a for c2 in CASES for a in [SS["variants"]; [m["label"] for m in context_models(c2) if haskey(m, "case")]])))
    isempty(bad) || throw(ArgumentError("--variant: no arm $(join(bad, ", ")) (the variants and each case's context case models)"))
    return filter(in(ARM_SELECTION), all_)
end

# ---- the lattice guard -------------------------------------------------------------------------------------------------------
"The lattice the arm's start laid (start.json resample): its grid size, node count and subdivision."
function start_lattice(dir::AbstractString)
    r = VgtkIO.read_record(joinpath(dir, RadiometricCamera.START_RECORD_FILE))
    rs = get(r, :resample, nothing)
    rs === nothing && return nothing
    g = Int.(collect(rs.grid_size)); m = Int.(collect(rs.subdivision))
    return (; grid_size = g, nodes = prod(g), subdivision = m, lattice_spacing = Float64.(collect(rs.lattice_spacing)))
end
"""
    lattice_guard(c, arm, dir) -> NamedTuple

The guard before every refinement (owner rule 2026-11-15): the arm's start lattice (`start_lattice`) must be subdivided exactly as the
case's `lattice.subdivision` and hold at most `lattice.max_nodes` nodes (the variant's `lattice_override` where stated); an error
otherwise — the refinement is not run. A parametric case states no `lattice`, and its start must lay none (the guard accepts it, with a note); a spline case's start without a resample is refused.
"""
function lattice_guard(c, arm, dir)
    L = start_lattice(dir)
    if !haskey(CFG["cases"][c], "lattice")         # a parametric case (no lattice stated): its start must lay none either
        L === nothing || throw(ArgumentError("case $c arm $arm: the start lays a lattice but the case states no `lattice` (a spline case states its own)"))
        line = "parametric start: no lattice (the guard reads a spline start's)"
        println(">> $c $arm: $line")
        return (; line)
    end
    l = get(get(CFG["cases"][c], "lattice_override", Dict()), arm, CFG["cases"][c]["lattice"])
    L === nothing && throw(ArgumentError("case $c arm $arm: the start $dir records no resample (no Schöps lattice): the guard cannot read its lattice"))
    msg = @sprintf("grid: %d nodes (%d × %d), m = %d × %d, cell %.1f × %.1f px", L.nodes, L.grid_size..., L.subdivision..., L.lattice_spacing...)
    println(">> $c $arm: $msg")
    L.subdivision == l["subdivision"] || throw(ArgumentError("case $c arm $arm REFUSED (the slow lattice): $msg; the case allows m = $(join(l["subdivision"], " × "))"))
    L.nodes <= l["max_nodes"] || throw(ArgumentError("case $c arm $arm REFUSED: $msg exceeds the case's $(l["max_nodes"]) nodes"))
    return (; L..., line = msg)
end

# ---- prepare and refine: the owner's driver ------------------------------------------------------------------------------------
const RR_DRIVER = projectdir("scripts", "radiometric_refinement.jl")
rr(c, args...) = run_isolated(RR_DRIVER; args = ["--config=$(refinement_path(c))", args...])

"""
    produce_runs(cfg)

Every `[runs]` entry of `cfg` (a front-end run of scripts/frontends.jl: a decode, a Schöps run or a fit) produced — or loaded, by its
key — by its OWNER before the driver that reads it: `config_source` (drivers.jl), through the run's resolver and `produce_upstream`
(the one path from a consumer to an upstream product).
"""
produce_runs(cfg) = foreach(id -> config_source(cfg, id), sort(collect(keys(get(cfg, "runs", Dict{String, Any}())))))

"What every arm of case `c` reads, produced once before the arms: the prepare and detect of its photographs and its image region."
function run_prepare_case(c)
    produce_upstream(configs_product(), "the configs"; driver_run(DRIVER, CFG_PATH, "configs")...)
    rc = loaded_refinement(c)
    for t in values(rc["cases"])                                     # the image regions the cases read (their own step, on demand)
        lens_coverage_record(rc, t["pipeline"]["camera"]["lens_coverage"]; what = "case $c")
    end
    produce_runs(rc)
    rr(c, "--stage=prepare,detect")
    return nothing
end
"The prepare step of case `c` (its arms' upstreams produced: `run_prepare_case`), keyed by the case's refinement config."
prepared_product(c) = driver_product("prepared", Dict{String, Any}("experiment" => CFG["experiment"], "case" => c, "upstream" => Dict("configs" => configs_product().key));
                                     output = "prepared", threads = nothing)
run_prepared(c) = produce(prepared_product(c); product_records(DRIVER, CFG_PATH)...) do dir
    run_prepare_case(c); write(joinpath(dir, "prepared"), ""); Produced(; inputs = [refinement_path(c)])
end

"""The refinement of `arm` of case `c`: the start (the owner's start stage), the lattice guard on it, then the owner's refine."""
function run_arm((c, arm))
    rc = loaded_refinement(c); n = arm_case(c, arm)
    st = start_product(rc, n)
    g = lattice_guard(c, arm, produce_upstream(st, "the start of $c $arm"; driver_run("scripts/radiometric_refinement.jl", refinement_path(c), "start"; case = n)...,
                                               gpu = true))
    open(io -> println(io, g.line), joinpath(product_dir(st), "lattice_guard.txt"), "w")
    rr(c, "--case=$n", "--stage=refine")
    return nothing
end

arm_items() = [(c, a) for c in CASES for a in arms(c)]
arm_product((c, a)) = refine_product(loaded_refinement(c), arm_case(c, a))
"The produced arms of case `c` among `as` (the variants, or the variants and the context)."
produced_arms(c, as) = [a for a in as if produced(arm_product((c, a)))]

# ---- evaluate: the chrome evaluation of record on the arms -----------------------------------------------------------------------
const CHROME_DRIVER = projectdir("scripts", "chrome_evaluation.jl")
const EVAL_KINDS = ("variants", "context")

# The chrome model kind of an arm: its refinement case's `kind` ("grid" a spline case, "lens" a parametric one).
arm_kind(c, v) = String(loaded_refinement(c)["cases"][arm_case(c, v)]["kind"])

"""
    chrome_config(c, kind) -> Union{Nothing, Dict}

The chrome evaluation config of case `c`: its config of record with `[upstream] radiometric_refinement` the case's refinement config and
`[evaluate] models` the PRODUCED variants (`kind = "variants"`: the head-to-head on their own common dots) or the produced variants and
the context's models (`"context"`), every other key of record kept (holdout, the bootstrap, the regions, the index). Comparisons:
every variant − the reference; in the context also the reference and the headline variants − each context model. `nothing` while
the reference arm is not produced (nothing to compare with).
"""
function chrome_config(c::AbstractString, kind::AbstractString)
    ref = SS["reference"]
    vs = produced_arms(c, SS["variants"]); ref in vs || return nothing
    rec = deepcopy(chrome_record(c))                                  # the cached config of record, never edited
    rec["upstream"] = Dict{String, Any}("radiometric_refinement" => refinement_path(c))
    models = Any[Dict{String, Any}("label" => v, "kind" => arm_kind(c, v), "case" => arm_case(c, v)) for v in vs]
    cmp = Any[Dict{String, Any}("model" => v, "reference" => ref) for v in vs if v != ref]
    if kind == "context"
        for m in context_models(c)
            haskey(m, "case") && !produced(arm_product((c, m["label"]))) && continue
            push!(models, haskey(m, "case") ? merge(Dict{String, Any}(m), Dict{String, Any}("case" => arm_case(c, m["label"]))) : Dict{String, Any}(m))
            append!(cmp, [Dict{String, Any}("model" => v, "reference" => m["label"]) for v in unique([ref; filter(in(vs), SS["headline"])])])
        end
    end
    rec["evaluate"]["models"] = models; rec["evaluate"]["comparisons"] = cmp
    # the front-end runs this evaluation reads, alone (the owner's loader checks every [runs] product): the index camera and the models' inputs
    refs = Set{String}(String(v) for tb in ("detect", "index") if haskey(rec, tb) for v in values(rec[tb]) if v isa AbstractString)
    union!(refs, (String(m["input"]) for m in models if haskey(m, "input")), (String(v) for (k, v) in rec["evaluate"] if v isa AbstractString))
    haskey(rec, "runs") && (rec["runs"] = Dict{String, Any}(id => r for (id, r) in rec["runs"] if id in refs))
    return rec
end
chrome_path(c, kind) = joinpath(product_dir(chrome_configs_product(c, kind)), "chrome_evaluation_ss_$(c)_$kind.toml")
chrome_configs_product(c, kind) = driver_product("chrome_configs", Dict{String, Any}("experiment" => CFG["experiment"], "case" => c, "kind" => kind,
                                                                                     "config" => chrome_config(c, kind), "upstream" => Dict("configs" => configs_product().key));
                                                 output = "chrome_evaluation_ss_$(c)_$kind.toml", threads = nothing)
"The chrome config of `(c, kind)` written (its product), loaded by the owner's loader."
function chrome_loaded(c, kind)
    p = chrome_configs_product(c, kind)
    produced(p) || produce(p; product_records(DRIVER, CFG_PATH)...) do dir
        open(io -> TOML.print(io, chrome_config(c, kind); sorted = true), joinpath(dir, "chrome_evaluation_ss_$(c)_$kind.toml"), "w")
        Produced(; inputs = [projectdir(CFG["cases"][c]["chrome"]), refinement_path(c)])
    end
    return load_chrome_evaluation_config(chrome_path(c, kind))
end
eval_items() = [(c, k) for c in CASES for k in EVAL_KINDS if chrome_config(c, k) !== nothing]
eval_product((c, k)) = evaluate_product(chrome_loaded(c, k))
run_eval((c, k)) = (produce_runs(chrome_loaded(c, k)); run_isolated(CHROME_DRIVER; args = ["--config=$(chrome_path(c, k))", "--stage=prepare,detect,index,evaluate"]); nothing)

# ---- texture: the texture experiment on each variant's camera (the 14 mm) ------------------------------------------------------
const TEXTURE_DRIVER = projectdir("scripts", "texture_recovery.jl")
const TEXTURE_STAGES = "prepare,detect,sheet,init,refine,evaluate"
"""The texture config of variant `v` of case `c`: the texture config of record with `[upstream]` the variant's refinement (its
config and case) and without `[folds]` and `[compare]` (one split of record; the comparison is the experiment's, `compare_config`)."""
function texture_config(c, v)
    t = TOML.parsefile(projectdir(CFG["cases"][c]["texture"]))
    t["upstream"]["refinement"] = refinement_path(c); t["upstream"]["case"] = arm_case(c, v)
    delete!(t, "folds"); delete!(t, "compare")
    return t
end
texture_configs_product(c) = driver_product("texture_configs", Dict{String, Any}("experiment" => CFG["experiment"], "case" => c,
                                                                                 "configs" => Dict(v => texture_config(c, v) for v in SS["variants"]),
                                                                                 "upstream" => Dict("configs" => configs_product().key)); output = "done", threads = nothing)
texture_path(c, v) = joinpath(product_dir(texture_configs_product(c)), "texture_recovery_ss_$(c)_$v.toml")
function texture_configs(c)
    p = texture_configs_product(c)
    produced(p) || produce(p; product_records(DRIVER, CFG_PATH)...) do dir
        for v in SS["variants"]; open(io -> TOML.print(io, texture_config(c, v); sorted = true), joinpath(dir, "texture_recovery_ss_$(c)_$v.toml"), "w"); end
        write(joinpath(dir, "done"), "")
        Produced(; inputs = [projectdir(CFG["cases"][c]["texture"]), refinement_path(c)])
    end
    return nothing
end
"The texture driver's module on `path` (its plan: nothing run), for its products."
const TEXTURE_MODULES = Dict{String, Module}()
texture_module(path) = get!(() -> run_isolated(TEXTURE_DRIVER; args = ["--plan", "--config=$path"]), TEXTURE_MODULES, path)
tex_cases() = [c for c in CASES if haskey(CFG["cases"][c], "texture")]
texture_items() = [(c, v) for c in tex_cases() for v in VARIANTS if produced(arm_product((c, v)))]
texture_product((c, v)) = (texture_configs(c); Base.invokelatest(texture_module(texture_path(c, v)).report_product))
run_texture((c, v)) = (texture_configs(c); run_isolated(TEXTURE_DRIVER; args = ["--config=$(texture_path(c, v))", "--stage=$TEXTURE_STAGES"]); nothing)

"""The comparison config of case `c`: the reference variant's texture config with `[compare]`: every other variant whose page run is
produced by its refine key, `this` the reference, the crop of record (`[cases.<c>] texture_crop_mm`)."""
function compare_config(c)
    texture_configs(c); ref = SS["reference"]
    t = texture_config(c, ref)
    runs = Dict{String, Any}()
    for v in SS["variants"]
        v == ref && continue
        m = texture_module(texture_path(c, v)); tp = Base.invokelatest(m.texture_product)
        produced(tp) && (runs[v] = tp.key)
    end
    t["compare"] = Dict{String, Any}("runs" => runs, "this" => ref, "reference" => ref, "crop_mm" => CFG["cases"][c]["texture_crop_mm"])
    return t
end
compare_configs_product(c) = driver_product("compare_configs", Dict{String, Any}("experiment" => CFG["experiment"], "case" => c, "config" => compare_config(c));
                                            output = "texture_recovery_ss_$(c)_compare.toml", threads = nothing)
function compare_path(c)
    p = compare_configs_product(c)
    produced(p) || produce(p; product_records(DRIVER, CFG_PATH)...) do dir
        open(io -> TOML.print(io, compare_config(c); sorted = true), joinpath(dir, "texture_recovery_ss_$(c)_compare.toml"), "w")
        Produced(; inputs = [texture_path(c, SS["reference"])])
    end
    return joinpath(product_dir(p), "texture_recovery_ss_$(c)_compare.toml")
end
compare_items() = [c for c in tex_cases() if length(produced_arms(c, SS["variants"])) >= 2 && SS["reference"] in produced_arms(c, SS["variants"]) &&
                                                !isempty(compare_config(c)["compare"]["runs"])]
compare_product(c) = Base.invokelatest(texture_module(compare_path(c)).compare_product)
run_compare(c) = (run_isolated(TEXTURE_DRIVER; args = ["--config=$(compare_path(c))", "--stage=compare"]); nothing)

# ---- the record --------------------------------------------------------------------------------------------------------------
"The focus surface of a refinement (its state's, state.json: the sensor-grid thin plate spline's nodal values): node distances (m), their range, the max deviation from the mean, dome (every node
finite, its distance within ± `[sampling_strategy] dome_tolerance` of their mean), the bending energy."
function focus_metrics(dir)
    f = joinpath(dir, "state.json"); isfile(f) || return nothing
    focus = RadiometricCamera.load_state(f).camera.focus
    t = Float64.(collect(focus.t̂_i)); d = [x > 0 && isfinite(x) ? 1 / x : Inf for x in t]
    fin = filter(isfinite, d); μ = isempty(fin) ? NaN : mean(fin)
    dev = isempty(fin) ? NaN : maximum(abs.(fin ./ μ .- 1))
    wh = Float64.(collect(VgtkIO.read_record(joinpath(dir, "parameters.json"); allownan = true).image_size))
    hd = hypot(wh...) / 2                                            # the sensor's half-diagonal (px): the nodes in its units, so the energy compares across cases
    nodes = [SVector{2, Float64}(Float64.(collect(n)) ./ hd) for n in focus.nodes]
    be = try VgtkCore.bending_energy(VgtkCore.ThinPlateSplineSystem(nodes), t) catch; NaN end
    return Dict{String, Any}("distance_m" => d, "min_m" => isempty(fin) ? NaN : minimum(fin), "max_m" => isempty(fin) ? NaN : maximum(fin),
                             "max_dev" => dev, "dome" => length(fin) == length(d) && dev <= SS["dome_tolerance"], "runaway" => length(fin) < length(d) || any(<=(0), t),
                             "bending_energy" => be, "bending_energy_units" => "(1/m)² over the nodes in half-diagonals", "nodes" => length(d))
end

"The samples per lattice node of a start: each sample's nearest node of the start lattice (state.json's grid), over every view."
function coverage_metrics(sdir)
    st = RadiometricCamera.load_state(joinpath(sdir, "state.json")); m = VgtkCore.model(st.camera.geometric)
    g = Tuple(RadiometricCamera.grid_dims(st.camera.geometric))
    p11 = VgtkCore.node_position(m, CartesianIndex(1, 1)); p22 = VgtkCore.node_position(m, CartesianIndex(2, 2))
    h = Float64.(p22 .- p11)
    counts = zeros(Int, g)
    for f in readdir(joinpath(sdir, RadiometricCamera.START_MASKS_DIR); join = true)
        M = PNGFiles.load(f)
        for I in findall(x -> Float64(x) > 0.5, M)
            v, u = Tuple(I)
            i = round(Int, (u - 1 - p11[1]) / h[1]) + 1; j = round(Int, (v - 1 - p11[2]) / h[2]) + 1   # COLMAP pixel centres
            1 <= i <= g[1] && 1 <= j <= g[2] && (counts[i, j] += 1)
        end
    end
    seen = counts[counts .> 0]
    return Dict{String, Any}("nodes" => prod(g), "nodes_with_samples" => length(seen), "per_node_median" => isempty(seen) ? 0 : median(seen),
                             "per_node_p10" => isempty(seen) ? 0 : quantile(seen, 0.1), "per_node_min" => isempty(seen) ? 0 : minimum(seen),
                             "per_node_max" => isempty(seen) ? 0 : maximum(seen), "spacing_px" => collect(h))
end

"The chrome numbers of `label` in an evaluate product: the held-out head-to-head (common dots) and its paired rows against the reference."
function chrome_metrics(edir, label)
    f = joinpath(edir, "eval_$label.json"); isfile(f) || return nothing
    e = VgtkIO.read_record(f; allownan = true)
    out = Dict{String, Any}()
    for v in ("motion_heldout", "motion")
        haskey(e, Symbol(v)) || continue
        cm = getproperty(e, Symbol(v)).common
        out[v] = Dict{String, Any}(String(k) => x for (k, x) in pairs(cm))
    end
    pf = joinpath(edir, PAIRED_FILE)
    if isfile(pf)
        out["paired"] = [Dict{String, Any}(r) for r in read_tsv(pf) if r["model"] == label]
    end
    return out
end
paired_row(m, reference, variant, region, stat) = (rs = get(m, "paired", Any[]); i = findfirst(r -> r["reference"] == reference && r["variant"] == variant && r["region"] == region && r["stat"] == stat, rs);
                                                    i === nothing ? nothing : (; d = parse(Float64, rs[i]["difference"]), lo = parse(Float64, rs[i]["lo"]), hi = parse(Float64, rs[i]["hi"]), photos = parse(Int, rs[i]["photos"]), dots = parse(Int, rs[i]["dots"])))

"The texture numbers of variant `v` in a compare product: its chosen scores (both blur models), learned − pinhole, and − the reference."
function texture_metrics(cdir, v)
    f = joinpath(cdir, "chosen.json"); isfile(f) || return nothing
    ch = VgtkIO.read_record(f; allownan = true)
    i = findfirst(r -> String(r.name) == v, collect(ch.runs)); i === nothing && return nothing
    r = ch.runs[i]
    out = Dict{String, Any}("chosen" => plain(r.chosen), "learned_minus_pinhole" => r.paired === nothing ? nothing : plain(r.paired))
    vr = [plain(x) for x in ch.paired_vs_reference if String(x.run) == v]
    out["minus_reference"] = Dict{String, Any}(x["variant"] => x for x in vr)
    return out
end

"Every arm's row: its keys, lattice, run time, focus surface, coverage, Thorlabs and texture numbers."
function record_rows()
    rows = Dict{String, Any}[]
    for c in CASES
        rc = loaded_refinement(c)
        ev = Dict(k => (cf = chrome_config(c, k); cf === nothing ? nothing : (p = eval_product((c, k)); produced(p) ? product_dir(p) : nothing)) for k in EVAL_KINDS)
        cdir = c in compare_items() ? (p = compare_product(c); produced(p) ? product_dir(p) : nothing) : nothing
        for a in arms(c)
            n = arm_case(c, a); sp = start_product(rc, n); rp = refine_product(rc, n)
            row = Dict{String, Any}("case" => c, "arm" => a, "kind" => a in SS["variants"] ? "variant" : "context", "case_name" => n,
                                    "start_key" => sp.key, "refine_key" => rp.key, "label" => LABEL, "overrides" => a in SS["variants"] ? variant_overrides(c, a) : nothing)
            if produced(sp)
                L = start_lattice(product_dir(sp)); row["lattice"] = L === nothing ? nothing : Dict{String, Any}(String(k) => x for (k, x) in pairs(L))
                sj = VgtkIO.read_record(joinpath(product_dir(sp), RadiometricCamera.START_RECORD_FILE))
                row["samples_drawn"] = sum(Int.(collect(sj.samples.drawn)))
                row["coverage"] = try coverage_metrics(product_dir(sp)) catch err; "not computed: " * sprint(showerror, err) end
            end
            if produced(rp)
                P = VgtkIO.read_record(joinpath(product_dir(rp), "parameters.json"); allownan = true)
                it = Int(P.iterations); sec = Float64(P.seconds)
                row["refine"] = Dict{String, Any}("iterations" => it, "seconds" => sec, "s_per_iteration" => it > 0 ? sec / it : NaN, "retcode" => String(P.retcode),
                                                  "stopped" => String(P.stopped), "folds" => plain(get(P, :calibrated_folds, nothing)))
                ref_s = get(CFG["runtime"]["reference_s_per_iteration"], c, nothing)
                row["refine"]["regression"] = ref_s !== nothing && it > 0 && sec / it > CFG["runtime"]["regression_factor"] * ref_s
                row["focus"] = focus_metrics(product_dir(rp))
            end
            for k in EVAL_KINDS
                ev[k] === nothing || (row["chrome_$k"] = chrome_metrics(ev[k], a))
            end
            cdir !== nothing && a in SS["variants"] && (row["texture"] = texture_metrics(cdir, a))
            push!(rows, row)
        end
        # the context's inputs (the set's Schöps baseline, its parametric start): their numbers in the context evaluation
        ev["context"] === nothing && continue
        for m in context_models(c)
            haskey(m, "case") && continue
            push!(rows, Dict{String, Any}("case" => c, "arm" => m["label"], "kind" => "context", "label" => LABEL, "input" => m["input"],
                                          "chrome_context" => chrome_metrics(ev["context"], m["label"])))
        end
    end
    return rows
end
record_product() = driver_product("record", Dict{String, Any}("experiment" => CFG["experiment"], "sampling_strategy" => SS, "runtime" => CFG["runtime"], "cases" => CASES,
                                                              "upstream" => Dict("arms" => Dict("$c/$a" => arm_product((c, a)).key for (c, a) in arm_items()),
                                                                                 "evaluate" => Dict("$c/$k" => eval_product((c, k)).key for (c, k) in eval_items()),
                                                                                 "compare" => Dict(c => compare_product(c).key for c in compare_items())));
                                  output = "sampling_strategy.json", threads = nothing)
_num(x) = x === nothing ? "" : x isa Real ? (isfinite(x) ? @sprintf("%.4f", x) : "") : string(x)
function run_record(_)
    produce(record_product(); product_records(DRIVER, CFG_PATH)...) do dir
        rows = record_rows()
        VgtkIO.write_record(joinpath(dir, "sampling_strategy.json"), Dict{String, Any}("schema" => EXPERIMENT_RECORD_SCHEMAS.sampling_strategy, "label" => LABEL, "reference" => SS["reference"],
                                                                                         "headline" => SS["headline"], "rows" => rows); allownan = true)
        cols = ("case", "arm", "label", "nodes", "m", "s_per_it", "iterations", "retcode", "heldout_median_px", "d_median_px", "d_median_lo", "d_median_hi",
                "d_mean_px", "d_p95_px", "d_rms_px", "centre_median_px", "edge_median_px", "d_centre_median", "d_edge_median", "dome", "focus_min_m", "focus_max_m",
                "focus_max_dev", "bending_energy", "runaway", "tex_test_rel_rms", "tex_rms_1mm", "tex_d_1mm", "tex_d_lo", "tex_d_hi", "tex_lp_d", "tex_lp_lo", "tex_lp_hi",
                "per_node_median", "refine_key")
        line(r) = begin
            cv = get(r, "chrome_variants", nothing); cv === nothing && (cv = get(r, "chrome_context", nothing))   # a context row: its own evaluation
            h = cv === nothing ? Dict() : get(cv, "motion_heldout", Dict())
            pr(reg, st) = cv === nothing ? nothing : paired_row(cv, SS["reference"], "motion_heldout", reg, st)
            dm = pr("all", "median_px"); fo = get(r, "focus", nothing); fo = fo === nothing ? Dict() : fo
            tx = get(r, "texture", nothing); tl = tx === nothing ? Dict() : get(tx["chosen"], "learned", Dict())
            td = tx === nothing ? nothing : get(tx["minus_reference"], "learned", nothing); lp = tx === nothing ? nothing : tx["learned_minus_pinhole"]
            L = get(r, "lattice", nothing); R = get(r, "refine", Dict()); cov = get(r, "coverage", nothing)
            (r["case"], r["arm"], r["label"], L === nothing ? "" : L["nodes"], L === nothing ? "" : join(L["subdivision"], "x"), _num(get(R, "s_per_iteration", nothing)),
             get(R, "iterations", ""), get(R, "retcode", ""), _num(get(h, "median_px", nothing)),
             _num(dm === nothing ? nothing : dm.d), _num(dm === nothing ? nothing : dm.lo), _num(dm === nothing ? nothing : dm.hi),
             _num((x = pr("all", "mean_px"); x === nothing ? nothing : x.d)), _num((x = pr("all", "p95_px"); x === nothing ? nothing : x.d)), _num((x = pr("all", "rms_px"); x === nothing ? nothing : x.d)),
             _num(get(h, "centre_median_px", nothing)), _num(get(h, "edge_median_px", nothing)),
             _num((x = pr("centre", "median_px"); x === nothing ? nothing : x.d)), _num((x = pr("edge", "median_px"); x === nothing ? nothing : x.d)),
             get(fo, "dome", ""), _num(get(fo, "min_m", nothing)), _num(get(fo, "max_m", nothing)), _num(get(fo, "max_dev", nothing)), (x = get(fo, "bending_energy", nothing); x === nothing ? "" : @sprintf("%.4g", x)), get(fo, "runaway", ""),
             _num(get(tl, "test_rel_rms", nothing)), _num(get(tl, "rms_block", nothing)),
             _num(td === nothing ? nothing : td["rms_block_mean_diff"]), _num(td === nothing ? nothing : td["rms_block_diff_ci95"][1]), _num(td === nothing ? nothing : td["rms_block_diff_ci95"][2]),
             _num(lp === nothing ? nothing : lp["rms_block_mean_diff"]), _num(lp === nothing ? nothing : lp["rms_block_diff_ci95"][1]), _num(lp === nothing ? nothing : lp["rms_block_diff_ci95"][2]),
             cov isa AbstractDict ? _num(cov["per_node_median"]) : "", get(r, "refine_key", ""))
        end
        write_tsv(joinpath(dir, "sampling_strategy.tsv"), cols, [line(r) for r in rows])
        # the index of the experiment's keyed runs in the one results root (one row per product)
        idx = Tuple[]
        for r in rows
            haskey(r, "case_name") || continue                       # a context input: no run of this experiment
            rc = loaded_refinement(r["case"])
            for (st, p) in (("start", start_product(rc, r["case_name"])), ("refine", refine_product(rc, r["case_name"])))
                push!(idx, ("sampling_strategy", r["case"], r["arm"], st, p.key, product_dir(p), produced(p) ? "complete" : "missing", LABEL))
            end
        end
        for (c, k) in eval_items(); p = eval_product((c, k)); push!(idx, ("sampling_strategy", c, k, "chrome_evaluate", p.key, product_dir(p), produced(p) ? "complete" : "missing", LABEL)); end
        for c in compare_items(); p = compare_product(c); push!(idx, ("sampling_strategy", c, "variants", "texture_compare", p.key, product_dir(p), produced(p) ? "complete" : "missing", LABEL)); end
        write_tsv(joinpath(dir, "index.tsv"), ("experiment", "case", "arm", "stage", "key", "path", "status", "label"), idx)
        cp(joinpath(dir, "index.tsv"), joinpath(results_root(), "sampling_strategy", "index.tsv"); force = true)
        Produced(; inputs = String[])
    end
end

# ---- the figure and the table ----------------------------------------------------------------------------------------------
figure_product() = driver_product("figure", Dict{String, Any}("experiment" => CFG["experiment"], "figure" => CFG["figure"], "style" => [collect(style_row("ours"))],
                                                              "upstream" => Dict("record" => record_product().key)); output = basename(CFG["figure"]["asset"]), threads = nothing)
"""The figure: per case a panel of the held-out Thorlabs median Δ (px) of every variant − the reference with its 95 % paired interval, and
for the texture case a panel of the 1 mm texture RMS Δ; pgfplots, the data inline, the paper's style (src/figure_style.jl)."""
function write_figure(path, rows)
    o = style_row("ours")
    vs = [v for v in SS["variants"] if v != SS["reference"]]
    sym(v) = replace(v, "_" => "")                                   # a pgfplots symbolic coordinate (no underscore)
    xt = join(sym.(vs), ","); xl = join((replace(v, "_" => "\\_") for v in vs), ",")
    io = IOBuffer()
    println(io, "% fig_sampling_strategy.tex — scripts/sampling_strategy.jl (generated, do not edit). ", LABEL, ".")
    println(io, "\\begin{tikzpicture}\\begin{groupplot}[group style={group size=$(length(CASES) + length(tex_cases())) by 1, horizontal sep=1.6cm}, rc axis]")
    panel(title, ylabel, pts) = begin
        println(io, "\\nextgroupplot[$(house_axis("title={$title}", "ylabel={$ylabel}", "symbolic x coords={$xt}", "xtick={$xt}", "xticklabels={$xl}", "x tick label style={rotate=45, anchor=east}"))]")
        println(io, "\\addplot[$(o.name), only marks, error bars/.cd, y dir=both, y explicit] coordinates {")
        for (v, d, lo, hi) in pts; println(io, @sprintf("  (%s, %.5f) += (0, %.5f) -= (0, %.5f)", sym(v), d, hi - d, d - lo)); end
        println(io, "};")
        println(io, "\\draw[gray, dashed] (rel axis cs:0,0) -- (rel axis cs:1,0);")
    end
    for c in CASES
        pts = Any[]
        for v in vs
            r = findfirst(x -> x["case"] == c && x["arm"] == v, rows); r === nothing && continue
            cv = get(rows[r], "chrome_variants", nothing); cv === nothing && continue
            p = paired_row(cv, SS["reference"], "motion_heldout", "all", "median_px"); p === nothing || push!(pts, (v, p.d, p.lo, p.hi))
        end
        panel(replace(c, "_" => " ") * ": Thorlabs", "\$\\Delta\$ held-out median (px)", pts)
    end
    for c in tex_cases()
        pts = Any[]
        for v in vs
            r = findfirst(x -> x["case"] == c && x["arm"] == v, rows); r === nothing && continue
            tx = get(rows[r], "texture", nothing); tx === nothing && continue
            d = get(tx["minus_reference"], "learned", nothing); d === nothing || push!(pts, (v, d["rms_block_mean_diff"], d["rms_block_diff_ci95"]...))
        end
        panel(replace(c, "_" => " ") * ": texture", "\$\\Delta\$ RMS at 1 mm", pts)
    end
    println(io, "\\end{groupplot}\\end{tikzpicture}")
    mkpath(dirname(path)); write(path, String(take!(io)))
    return path
end
"The results table: per case and variant the held-out Δ [CI], the texture Δ at 1 mm [CI], the dome and the run time per iteration."
function write_table(path, rows)
    io = IOBuffer(); f(x) = x === nothing ? "--" : @sprintf("%+.3f", x)
    println(io, "% tab_sampling_strategy.tex — scripts/sampling_strategy.jl (generated, do not edit). ", LABEL, ". Δ: variant − ", SS["reference"], ", paired 95 % interval.")
    println(io, "\\begin{tabular}{llrrrr}\\toprule")
    println(io, "case & variant & Thorlabs \$\\Delta\$ median (px) & texture \$\\Delta\$ 1 mm & dome (max dev) & s / it \\\\\\midrule")
    for r in rows
        r["kind"] == "variant" || continue
        cv = get(r, "chrome_variants", nothing); p = cv === nothing ? nothing : paired_row(cv, SS["reference"], "motion_heldout", "all", "median_px")
        tx = get(r, "texture", nothing); d = tx === nothing ? nothing : get(tx["minus_reference"], "learned", nothing)
        fo = get(r, "focus", nothing); R = get(r, "refine", nothing)
        th = p === nothing ? "--" : "$(f(p.d)) [$(f(p.lo)), $(f(p.hi))]"
        te = d === nothing ? "--" : "$(f(d["rms_block_mean_diff"])) [$(f(d["rms_block_diff_ci95"][1])), $(f(d["rms_block_diff_ci95"][2]))]"
        dome = fo === nothing ? "--" : @sprintf("%s (%.3f)", fo["dome"] ? "yes" : "no", fo["max_dev"])
        sp = R === nothing ? "--" : @sprintf("%.0f", R["s_per_iteration"])
        println(io, replace(r["case"], "_" => " "), " & ", replace(r["arm"], "_" => "\\_"), " & ", th, " & ", te, " & ", dome, " & ", sp, " \\\\")
    end
    println(io, "\\bottomrule\\end{tabular}")
    mkpath(dirname(path)); write(path, String(take!(io)))
    return path
end
function run_figure(_)
    assets = [joinpath("latex", CFG["figure"]["asset"]), joinpath("latex", CFG["figure"]["table"]), joinpath("latex", "figure_styles.tex")]
    produce_figures(figure_product(), assets; driver = DRIVER, config_path = CFG_PATH) do dir
        rows = VgtkIO.read_record(joinpath(produce_upstream(record_product(), "the sampling-strategy record"; driver_run(DRIVER, CFG_PATH, "record")...), "sampling_strategy.json"); allownan = true).rows
        rows = [plain(r) for r in rows]
        write_figure(plotsdir(assets[1]), rows); write_table(plotsdir(assets[2]), rows); write_figure_styles(plotsdir(assets[3]))
        for a in assets; cp(plotsdir(a), joinpath(dir, basename(a)); force = true); end
        write_figure_preview(dir, splitext(CFG["figure"]["asset"])[1]; size = ("5cm", "5cm"))
    end
end

const GPU_STAGES = [:prepare, :refine, :evaluate, :texture]
const STAGES = [
    Stage(:configs, "per case the refinement config of the variants (expand_overrides) and the context cases"; items = () -> [nothing],
          product = _ -> configs_product(), run = _ -> run_configs(configs_product())),
    Stage(:prepare, "per case the prepare and detect of its photographs, its front-end runs and its image region (the owners' drivers), once before the arms";
          items = () -> CASES, product = prepared_product, run = run_prepared),
    Stage(:refine, "per case and arm: the start, the lattice guard (m and the node count), the refinement (scripts/radiometric_refinement.jl)";
          items = arm_items, product = arm_product, run = run_arm),
    Stage(:evaluate, "per case the chrome evaluation of record on the produced variants, and with the context (scripts/chrome_evaluation.jl)";
          items = eval_items, product = eval_product, run = run_eval),
    Stage(:texture, "per texture case and produced variant: the texture experiment on its camera (scripts/texture_recovery.jl)";
          items = texture_items, product = texture_product, run = run_texture),
    Stage(:compare, "per texture case: the variants' page runs by one scorer, each − the reference paired over the cells (scripts/texture_recovery.jl compare)";
          items = compare_items, product = compare_product, run = run_compare),
    Stage(:record, "the one record: sampling_strategy.json, sampling_strategy.tsv, index.tsv"; items = () -> [nothing], product = _ -> record_product(), run = run_record),
    Stage(:figure, "the figure and the table (pgfplots: plots/latex/fig_sampling_strategy.tex, tab_sampling_strategy.tex)";
          items = () -> [nothing], product = _ -> figure_product(), run = run_figure),
]

println(">> sampling_strategy: ", LABEL)
run_stages("sampling_strategy", STAGES)
