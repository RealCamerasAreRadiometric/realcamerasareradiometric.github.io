# =============================================================================
# src/warping_ablation.jl — the board warping ablation (owner 2026-11-16; configs/studies/warping_ablation.toml,
# scripts/warping_ablation.jl): which board parameterisation is right for each camera model, judged only by the held-out Thorlabs
# (chrome) error. The owner's hypothesis: the global (parametric) models lack degrees of freedom, so the board's warping parameters
# absorb lens error during the radiometric refinement — the photometric fit improves while the geometry worsens.
#
# Nothing here refines or scores: the cases are the refinement's cases of record ([upstream] radiometric_refinement), each refined
# once per [modes.<name>] as an OVERRIDE of its [pipeline] (src/drivers.jl `override_variant`, the experiments' one variant rule) by
# the refinement driver's own stage (`run_refine`, its keyed products); every start and every mode's refinement of a chrome set is
# scored in ONE scoring by the chrome evaluation's scorer (src/chrome_evaluation.jl `score_chrome_models`) under the set's protocol of
# record (its [evaluate] table: the rotation-centre motion fit with the camera held, holdout on, its paired photograph bootstrap).
# The cases are enumerated from the paper's chrome sets ([upstream] chrome_evaluation_paper; `warping_cases`), never listed.
# The record (warping_ablation.json, EXPERIMENT_RECORD_SCHEMAS.warping_ablation, and its table warping_ablation.tsv) is the figure stage's
# one input (`write_warping_figure`: a pgfplots figure and a booktabs table through src/figure_style.jl).
# Included after src/drivers.jl, src/radiometric_refinement.jl, src/chrome_evaluation.jl and src/figure_style.jl.
# =============================================================================
using Printf: Printf, @printf, @sprintf

"The tables of configs/studies/warping_ablation.toml."
const WARPING_ABLATION_TABLES = ("experiment", "inputs", "upstream", "modes", "ablation", "figure")
"The keys of `[upstream]`, every one required: the refinements' config of record (the cases) and the chrome paper config (the chrome sets)."
const WARPING_UPSTREAM_KEYS = ("radiometric_refinement", "chrome_evaluation_paper")
"The keys of a `[modes.<name>]` table, every one required: its legend label, the [pipeline] keys it sets and those it deletes."
const WARPING_MODE_KEYS = ("label", "set", "unset")
"The keys of `[ablation]`, every one required: the reference mode, the headline variant, the in-sample variant and the statistics."
const WARPING_ABLATION_KEYS = ("reference", "variant", "in_sample_variant", "stats")
"The keys of `[figure]`, every one required (non-empty strings)."
const WARPING_FIGURE_KEYS = ("asset_dir", "ylabel", "label", "caption")
"The record's file (the record product's output) and its table beside it."
const WARPING_RECORD_FILE = "warping_ablation.json"
const WARPING_TABLE_FILE = "warping_ablation.tsv"
"The columns of warping_ablation.tsv: one row per refinement (capture set, model, mode), its headline scores, the start's, the differences against the reference and the start (the median's paired 95 % interval), and the run."
const WARPING_TABLE_COLUMNS = ("chrome", "prefix", "family", "kind", "mode", "label", "variant_case", "photos", "dots",
                               "median_px", "mean_px", "p95_px", "rms_px", "all_median_px", "coverage_fraction",
                               "start_median_px", "start_mean_px", "start_p95_px", "start_rms_px", "beats_start",
                               "ref_d_median_px", "ref_lo_px", "ref_hi_px", "start_d_median_px", "start_lo_px", "start_hi_px",
                               "rel_rms_start", "rel_rms_final", "iterations", "retcode", "stop_rule", "heights_rms_mm", "heights_max_mm",
                               "inplane_aspect", "inplane_skew", "fit_retcode")

"""
    load_warping_ablation_config(path) -> Dict

The config of record of the warping ablation, every table and key checked: `[upstream]` the refinements' config of record and the
chrome paper config; every `[modes.<name>]` (a lower-case name: it ends the scored model's label) a `label`, the [pipeline] keys it
`set`s (a table of `"<table>.<key>" = value`) and those it `unset`s (a list of such keys), never both on one key; `[ablation]
reference` a mode that sets and unsets nothing (the config of record itself), `variant` and `in_sample_variant` two chrome
evaluation variants, `stats` the paired bootstrap's statistics (`PAIRED_STATS`); `[figure]` non-empty strings.
"""
function load_warping_ablation_config(path::AbstractString)
    c = load_driver_config(path, "warping_ablation"; tables = WARPING_ABLATION_TABLES)
    for t in ("upstream", "modes", "ablation", "figure")
        haskey(c, t) && c[t] isa AbstractDict && !isempty(c[t]) || throw(ArgumentError("$path: no [$t] table"))
    end
    check_required_keys(c["upstream"], WARPING_UPSTREAM_KEYS, "$path [upstream]")
    for (k, v) in c["upstream"]
        v isa AbstractString && isfile(projectdir(v)) || throw(ArgumentError("$path [upstream] $k: $(repr(v)) is not a config of this project"))
    end
    key = r"^[a-z_]+(\.[a-z_]+)?$"
    for (m, t) in c["modes"]
        where = "$path [modes.$m]"
        occursin(r"^[a-z]+$", m) || throw(ArgumentError("$where: a mode's name is lower-case letters (it ends its models' labels)"))
        check_required_keys(t, WARPING_MODE_KEYS, where)
        t["label"] isa AbstractString && !isempty(t["label"]) || throw(ArgumentError("$where label: a non-empty string"))
        t["set"] isa AbstractDict && all(k -> occursin(key, k), keys(t["set"])) ||
            throw(ArgumentError("$where set: a table of \"<table>.<key>\" = value ([pipeline] keys)"))
        t["unset"] isa AbstractVector && all(k -> k isa AbstractString && occursin(key, k), t["unset"]) ||
            throw(ArgumentError("$where unset: a list of \"<table>.<key>\" ([pipeline] keys)"))
        isempty(intersect(keys(t["set"]), t["unset"])) || throw(ArgumentError("$where: a key both set and unset"))
    end
    A = check_required_keys(c["ablation"], WARPING_ABLATION_KEYS, "$path [ablation]")
    r = get(c["modes"], A["reference"], nothing)
    r === nothing && throw(ArgumentError("$path [ablation] reference: $(repr(A["reference"])) is not a [modes.<name>]"))
    isempty(r["set"]) && isempty(r["unset"]) ||
        throw(ArgumentError("$path [ablation] reference: the mode $(A["reference"]) sets or unsets keys — the reference is the config of record itself"))
    A["variant"] isa AbstractString && A["in_sample_variant"] isa AbstractString && A["variant"] != A["in_sample_variant"] ||
        throw(ArgumentError("$path [ablation]: variant and in_sample_variant are two chrome evaluation variants"))
    A["stats"] isa AbstractVector && !isempty(A["stats"]) && all(s -> s in PAIRED_STATS, A["stats"]) ||
        throw(ArgumentError("$path [ablation] stats: a list of $(PAIRED_STATS)"))
    check_required_keys(c["figure"], WARPING_FIGURE_KEYS, "$path [figure]")
    all(k -> c["figure"][k] isa AbstractString && !isempty(c["figure"][k]), WARPING_FIGURE_KEYS) ||
        throw(ArgumentError("$path [figure]: every key is a non-empty string"))
    return c
end

"The refinements' config of record `[upstream] radiometric_refinement` names (src/radiometric_refinement.jl)."
warping_refinements(cfg::AbstractDict) = load_radiometric_refinement_config(projectdir(cfg["upstream"]["radiometric_refinement"]))

"""
    warping_chrome_sets(cfg) -> Vector{Tuple{String, String}}

The chrome sets of `[upstream] chrome_evaluation_paper` (its `[chrome.<key>] config`), `(key, config)` sorted by key; a set whose
config is an excluded set's (configs/excluded.toml) is left out, said so.
"""
function warping_chrome_sets(cfg::AbstractDict)
    paper = TOML.parsefile(projectdir(cfg["upstream"]["chrome_evaluation_paper"]))
    excluded = Dict(normpath(c) => s for (s, e) in excluded_sets() for c in get(e, "configs", String[]))
    out = Tuple{String, String}[]
    for k in sort(collect(keys(paper["chrome"])))
        p = String(paper["chrome"][k]["config"])
        haskey(excluded, normpath(p)) && (println("skipped chrome set $k: the excluded set $(excluded[normpath(p)]) (configs/excluded.toml)"); continue)
        push!(out, (k, p))
    end
    return out
end

"""
    check_case_of_record(t, case)

The case of record `case` (its table `t`) is what the modes override: its board has the heights (`warping.degree` > 0) and the in-plane
print error (`warping.inplane = "aspect_skew"`); and a Schöps-started grid case never subdivides his lattice (no `grid.node_count`:
m = 1 — the slow lattice is refused). An `ArgumentError` names the case otherwise.
"""
function check_case_of_record(t::AbstractDict, case::AbstractString)
    o = pipeline_options(t["pipeline"])
    o.warping.degree > 0 && o.warping.inplane == "aspect_skew" ||
        throw(ArgumentError("case $case: warping.degree = $(o.warping.degree), inplane = $(repr(o.warping.inplane)) — the modes are overrides of a case " *
                            "of record with the heights and the in-plane print error"))
    t["kind"] == "grid" && o.grid.node_count !== nothing &&
        throw(ArgumentError("case $case: grid.node_count = $(o.grid.node_count) subdivides Schöps's lattice (m > 1, the slow lattice): refused"))
    return nothing
end

"""
    warping_cases(cfg, rc = warping_refinements(cfg)) -> Vector{NamedTuple}

The cases of the ablation, ENUMERATED: in every chrome set of `warping_chrome_sets`, every capture set (a case's `images`) among whose
[evaluate] case models one is of kind "lens" (a global model); of it every case model of kind "lens" (the global models) and "grid"
(the central spline: the control), each `(; chrome, chrome_config, capture, prefix, family, case, kind, start)` — `prefix` and
`family` the label's (`<prefix>_<family>_refined`), `start` the model the set scores beside it (`<prefix>_<family>_start`, an input:
`(; label, kind, input)`). Each case is checked (`check_case_of_record`). In set, then label order.
"""
function warping_cases(cfg::AbstractDict, rc::AbstractDict = warping_refinements(cfg))
    rows = NamedTuple[]
    for (k, p) in warping_chrome_sets(cfg)
        cc = load_chrome_evaluation_config(projectdir(p))
        ms = cc["evaluate"]["models"]
        labels = Dict(m["label"] => m for m in ms)
        bycapture = Dict{String, Vector{Any}}()
        for m in ms
            haskey(m, "case") || continue
            haskey(rc["cases"], m["case"]) || throw(ArgumentError("$p: model $(m["label"]) names case $(m["case"]), not a case of $(cfg["upstream"]["radiometric_refinement"])"))
            push!(get!(bycapture, rc["cases"][m["case"]]["images"], Any[]), m)
        end
        for (capture, cms) in sort(collect(bycapture); by = first)
            any(m -> m["kind"] == "lens", cms) || continue
            for m in sort([m for m in cms if m["kind"] in ("lens", "grid")]; by = m -> m["label"])
                label = String(m["label"])
                endswith(label, "_refined") || throw(ArgumentError("$p: model $label scores a case but is not labelled <prefix>_<model>_refined"))
                parts = split(label, '_')
                length(parts) >= 3 || throw(ArgumentError("$p: model $label is not labelled <prefix>_<model>_refined"))
                start = replace(label, r"_refined$" => "_start")
                s = get(labels, start, nothing)
                s !== nothing && haskey(s, "input") || throw(ArgumentError("$p: model $label has no start $start (an input model) beside it"))
                check_case_of_record(rc["cases"][m["case"]], m["case"])
                push!(rows, (; chrome = String(k), chrome_config = p, capture = String(capture), prefix = String(parts[1]),
                             family = String(join(parts[2:end-1], '_')), case = String(m["case"]), kind = String(m["kind"]),
                             start = (; label = start, kind = String(s["kind"]), input = String(s["input"]))))
            end
        end
    end
    return rows
end

"""
    warping_plan(cfg; rc, cases) -> (; rc, rows, sets, modes)

The ablation's refinements: `rc`, the refinements' config of record with every case of `cases` (`warping_cases`) expanded into one
case per mode (`override_variant` of the mode's `set` and `unset`; the reference mode IS the case, every other mode a variant of it,
each variant's [pipeline] read through `pipeline_options`); `rows` per case and mode `(; case's fields…, mode, variant, label)` —
`variant` the refinement case's name in `rc`, `label` the scored model's (`<prefix>_<family>_<mode>`); `sets` the chrome sets' configs
by key; `modes` the mode names, sorted.
"""
function warping_plan(cfg::AbstractDict; rc::AbstractDict = warping_refinements(cfg), cases = warping_cases(cfg, rc))
    modes = sort(collect(String, keys(cfg["modes"])))
    ref = cfg["ablation"]["reference"]
    out = deepcopy(rc); out["cases"] = Dict{String, Any}()
    rows = NamedTuple[]
    for c in cases, m in modes
        mt = cfg["modes"][m]
        name, t = override_variant(c.case, rc["cases"][c.case], Dict{String, Any}(mt["set"]); unset = mt["unset"])
        (m == ref) == (name == c.case) ||
            throw(ArgumentError("case $(c.case), mode $m: $(m == ref ? "the reference mode is not the case of record" : "the mode leaves the case as it is")"))
        try
            pipeline_options(t["pipeline"])
        catch err
            err isa ArgumentError ? throw(ArgumentError("case $(c.case), mode $m [pipeline]: $(err.msg)")) : rethrow()
        end
        haskey(out["cases"], name) && throw(ArgumentError("case $(c.case), mode $m: the variant $name twice"))
        out["cases"][name] = t
        name == c.case || (get!(Dict{String, Any}, out, "_locations")[name] = "$(product_experiment(cfg["experiment"]["name"]))/$(rc["experiment"]["name"])")   # a mode's variant lies in the ablation's directory (`variant_location`)
        push!(rows, (; c..., mode = m, variant = name, label = "$(c.prefix)_$(c.family)_$m"))
    end
    sets = Dict(k => load_chrome_evaluation_config(projectdir(p)) for (k, p) in unique([(r.chrome, r.chrome_config) for r in rows]))
    return (; rc = out, rows, sets, modes)
end

"""The chrome set `cc`'s scoring protocol as it keys the ablation's scoring: its `[evaluate]` table without its models and comparisons
(the ablation's own); its holdout must be on (the headline variant is the held-out fit)."""
function warping_protocol(cc::AbstractDict, where::AbstractString)
    RadiometricCamera.read_switch(cc["evaluate"]["holdout"], "$where [evaluate] holdout") ||
        throw(ArgumentError("$where [evaluate] holdout is off: the ablation's headline is the held-out fit"))
    return Dict{String, Any}(k => v for (k, v) in cc["evaluate"] if !(k in ("models", "comparisons")))
end

"""
    warping_models(plan, k) -> (; rows, starts, comparisons)

The scoring of chrome set `k`: its rows of `plan`, the starts beside them (unique, by label) and the comparisons — every mode against
the reference mode of its case (`reference`) and every mode against its start — sorted.
"""
function warping_models(plan, k::AbstractString; reference::AbstractString)
    rows = sort([r for r in plan.rows if r.chrome == k]; by = r -> r.label)
    starts = sort(unique([r.start for r in rows]); by = s -> s.label)
    cmp = NamedTuple[]
    for r in rows
        r.mode == reference || push!(cmp, (; model = r.label, reference = "$(r.prefix)_$(r.family)_$reference"))
        push!(cmp, (; model = r.label, reference = r.start.label))
    end
    return (; rows, starts, comparisons = sort(cmp; by = c -> (c.model, c.reference)))
end

"""
    warping_evaluate_product(cfg, plan, k) -> Product

The scoring of chrome set `k` (its output summary.md, the scorer's, written last): keyed by the set's protocol (`warping_protocol`),
the models (each start's input, each refinement's variant) and the comparisons, the inputs read (the target, the starts) and the keys
of the products it reads (`upstream`: the set's index, its image region, every refinement).
"""
function warping_evaluate_product(cfg::AbstractDict, plan, k::AbstractString)
    cc = plan.sets[k]
    M = warping_models(plan, k; reference = cfg["ablation"]["reference"])
    up = Dict{String, Any}("index" => index_product(cc).key,
                           "radiometric_refinement" => Dict{String, Any}(r.variant => refinement_product(plan.rc, r.variant).key for r in M.rows))
    n = lens_coverage_name(cc["evaluate"]["lens_coverage"])
    n === nothing || (up["image_region"] = image_region_product(cc, n).key)
    ids = unique([cc["evaluate"]["target"]; [s.input for s in M.starts]])
    models = [[Dict{String, Any}("label" => s.label, "kind" => s.kind, "input" => s.input) for s in M.starts];
              [Dict{String, Any}("label" => r.label, "kind" => r.kind, "case" => r.variant) for r in M.rows]]
    rec = Dict{String, Any}("experiment" => cfg["experiment"], "chrome" => Dict{String, Any}(k => warping_protocol(cc, first(M.rows).chrome_config)),
                            "models" => models, "comparisons" => [Dict{String, Any}("model" => c.model, "reference" => c.reference) for c in M.comparisons],
                            "inputs" => inputs_record(cc, ids), "upstream" => up)
    return driver_product("evaluate", rec; output = "summary.md")
end

"""
    warping_record_product(cfg, plan) -> Product

The record (warping_ablation.json, written last; its table warping_ablation.tsv beside it): keyed by `[ablation]`, the modes and the keys
of every scoring and refinement it reads.
"""
warping_record_product(cfg::AbstractDict, plan) =
    driver_product("record", Dict{String, Any}("experiment" => cfg["experiment"], "ablation" => cfg["ablation"], "modes" => cfg["modes"],
                                               "record" => EXPERIMENT_RECORD_SCHEMAS.warping_ablation,
                                               "upstream" => Dict{String, Any}("evaluate" => Dict{String, Any}(k => warping_evaluate_product(cfg, plan, k).key for k in keys(plan.sets)),
                                                                               "refine" => Dict{String, Any}(r.variant => refinement_product(plan.rc, r.variant).key for r in plan.rows)));
                   output = WARPING_RECORD_FILE, threads = nothing)

"""
    warping_figure_product(cfg, plan) -> Product

The figure and the table: keyed by `[figure]` and the record's key; its output the figure fragment's copy `figure.tex`, written last.
"""
warping_figure_product(cfg::AbstractDict, plan) =
    driver_product("figure", Dict{String, Any}("experiment" => cfg["experiment"], "figure" => cfg["figure"],
                                               "upstream" => Dict{String, Any}("record" => warping_record_product(cfg, plan).key));
                   output = "figure.tex", threads = nothing)

# ---- the stages' work ------------------------------------------------------------------------------------------------------------

"""
    warping_refine(cfg, plan, variant; driver, config_path) -> dir

The refinement of `variant` (a case of `plan.rc`): the refinement driver's own stage (`run_refine`: its start and refinement products,
produced or loaded by their keys, every upstream produced on demand), refused while the refinements' config has `[pending]`.
"""
function warping_refine(cfg::AbstractDict, plan, variant::AbstractString; driver::AbstractString, config_path::AbstractString)
    produced(refinement_product(plan.rc, variant)) || refuse_pending(plan.rc, projectdir(cfg["upstream"]["radiometric_refinement"]))
    return run_refine(plan.rc, variant; driver, config_path)
end

"""
    warping_upstreams(cfg, plan, k; driver, config_path) -> nothing

Every upstream the refinements and the scoring of chrome set `k` read, produced on demand (`produce_upstream`; run once on a GPU,
before the refinements, so no two refinements make one at the same time): each case's decode (`require_prepared`), the front ends'
runs its [pipeline] names (`config_source`: Schöps's runs, the fits), its lens coverage (`require_lens_coverage`), the chrome set's
detect and index (`chrome_upstream`), its image region and each start model (`config_source`).
"""
function warping_upstreams(cfg::AbstractDict, plan, k::AbstractString; driver::AbstractString, config_path::AbstractString)
    cc = plan.sets[k]; rows = [r for r in plan.rows if r.chrome == k]
    cpath = projectdir(first(rows).chrome_config)
    for r in rows
        t = plan.rc["cases"][r.variant]
        require_prepared(plan.rc, t, "case $(r.variant)"; driver, config_path)
        foreach(id -> config_source(plan.rc, id), table_runs(t["pipeline"]))
        require_lens_coverage(plan.rc, t, "case $(r.variant)")
    end
    chrome_upstream(cc, cpath, "index", "the chrome index of $(first(rows).chrome_config)")
    W, H = index_camera_size(cc, index_settings(cc, cpath))
    lens_coverage_region(cc, cc["evaluate"]["lens_coverage"], W, H; what = "$cpath [evaluate]")
    foreach(s -> config_source(cc, s.input), unique([r.start for r in rows]))
    return nothing
end

"""
    warping_evaluate(cfg, plan, k; driver, config_path) -> dir

The scoring of chrome set `k` (`warping_evaluate_product`): every start (`config_source`) and every mode's refinement (its refine product,
produced on demand: a GPU's, refused in a CPU job naming the command) scored by the chrome scorer (`score_chrome_models`) under the
set's protocol with the ablation's comparisons, into the product: eval_<label>.json, summary.tsv, summary.md, paired.tsv (the scorer's).
"""
function warping_evaluate(cfg::AbstractDict, plan, k::AbstractString; driver::AbstractString, config_path::AbstractString)
    cc = plan.sets[k]
    M = warping_models(plan, k; reference = cfg["ablation"]["reference"])
    cpath = projectdir(first(M.rows).chrome_config)
    produce(warping_evaluate_product(cfg, plan, k); product_records(driver, config_path)...) do dir
        specs = [(s.label, "$(s.kind):" * config_source(cc, s.input)) for s in M.starts]
        for r in M.rows
            rdir = produce_upstream(refinement_product(plan.rc, r.variant), "the refinement $(r.variant)";
                                    make = () -> warping_refine(cfg, plan, r.variant; driver, config_path),
                                    command = driver_run(driver, config_path, "refine"; case = r.variant).command,
                                    gpu = runs_on_gpu(plan.rc["cases"][r.variant]))
            f = last_model(rdir, r.kind)
            isempty(f) && throw(ArgumentError("the refinement $(r.variant) ($rdir) holds no $(r.kind) model"))
            push!(specs, (r.label, "$(r.kind):$f"))
        end
        E = merge(evaluate_settings(cc, cpath), (; comparisons = M.comparisons))   # the set's protocol, the ablation's comparisons
        sc = score_chrome_models(cc, cpath, nothing, dir; settings = E, models = specs)
        Produced(; inputs = sc.inputs)
    end
end

"""
    warping_run_summary(refine_dir, start_dir, kind) -> NamedTuple

A refinement's run, read through VisualGeometryToolkitIO: the LM's rel-RMS at the start (`cost.init`) and at its end, its iterations,
`ReturnCode` and stop rule (parameters.json), the board's fitted TPS nodal heights (radiometric_target.json `warping`, mm: their RMS
and largest magnitude; `nothing` with no free heights) and its in-plane print error (Core's aspect `a` and skew `u`; `nothing` without
one). A grid start that subdivides Schöps's lattice (its start record's `resample.subdivision` > 1) is refused: the slow lattice.
"""
function warping_run_summary(refine_dir::AbstractString, start_dir::AbstractString, kind::AbstractString)
    p = VgtkIO.read_record(joinpath(refine_dir, "parameters.json"))
    t = VgtkIO.read_record(joinpath(refine_dir, "radiometric_target.json"))
    if kind == "grid"
        rs = RadiometricCamera.load_refinement_start(start_dir).resample
        sub = rs isa AbstractDict ? get(rs, "subdivision", nothing) : nothing
        sub !== nothing && any(>(1), sub) && throw(ArgumentError("$start_dir: the start subdivides Schöps's lattice ($(sub)): the slow lattice is refused"))
    end
    free = get(get(p, "board_warping", Dict()), "free", false) === true
    h = free ? Float64.(t["warping"]) : Float64[]
    a = u = nothing
    ip = get(t, "inplane_print_error", nothing)
    if get(get(p, "inplane_print_error", Dict()), "free", false) === true && ip isa AbstractDict
        for c in get(ip, "compose", Any[])
            c["type"] == "aspect" && (a = Float64(only(c["parameters"])))
            c["type"] == "skew" && (u = Float64(only(c["parameters"])))
        end
    end
    cost = p["cost"]
    return (; rel_rms_start = Float64(cost["init"]["rel_rms"]), rel_rms_final = Float64(cost["after_refinement"]["rel_rms"]),
            iterations = Int(p["iterations"]), retcode = String(p["retcode"]), stop_rule = String(p["stop_rule"]), seconds = Float64(p["seconds"]),
            heights_rms_mm = isempty(h) ? nothing : sqrt(sum(abs2, h) / length(h)), heights_max_mm = isempty(h) ? nothing : maximum(abs, h),
            inplane_aspect = a, inplane_skew = u)
end

# a model's scores of one variant from a scoring's summary.tsv rows: the head-to-head statistics, its all-dots ones and its coverage
function _variant_scores(rows, label, variant)
    i = findfirst(r -> r["model"] == label && r["variant"] == variant, rows)
    i === nothing && throw(ArgumentError("the scoring has no $variant row of model $label"))
    r = rows[i]
    return (; photos = Int(r["photos"]), dots = Int(r["dots"]), median_px = Float64(r["median_px"]), mean_px = Float64(r["mean_px"]),
            p95_px = Float64(r["p95_px"]), rms_px = Float64(r["rms_px"]), all_median_px = Float64(r["all_median_px"]), all_mean_px = Float64(r["all_mean_px"]),
            all_p95_px = Float64(r["all_p95_px"]), all_rms_px = Float64(r["all_rms_px"]), coverage_dots = Int(r["coverage_dots"]),
            coverage_fraction = Float64(r["coverage_fraction"]), retcode = String(r["retcode"]))
end
# the paired differences model − reference of one variant, every common dot: per statistic `(; difference, lo, hi, excludes_zero)`
function _paired(rows, model, reference, variant, stats)
    pick(st) = (i = findfirst(r -> r["model"] == model && r["reference"] == reference && r["variant"] == variant && r["region"] == "all" && r["stat"] == st, rows);
                i === nothing && throw(ArgumentError("the scoring has no paired $st of $model − $reference ($variant)"));
                r = rows[i]; (; difference = Float64(r["difference"]), lo = Float64(r["lo"]), hi = Float64(r["hi"]), excludes_zero = r["excludes_zero"] in ("true", true)))
    return NamedTuple{Tuple(Symbol.(stats))}(Tuple(pick(st) for st in stats))
end

"""
    warping_record(cfg, plan, dirs) -> NamedTuple

The record of the ablation (EXPERIMENT_RECORD_SCHEMAS.warping_ablation) from the scorings' and refinements' directories `dirs` — `(; evaluate
= k => dir, refine = variant => dir, start = variant => dir)` — per refinement (`runs`): its scores of the headline and in-sample
variants, its start's, whether it beats its start (the headline median), its paired differences against the reference mode
(`vs_reference`; `nothing` for the reference) and its start (`vs_start`), and its run (`run_summary`); per capture set and model
(`winners`): the mode of the lowest headline median and whether its difference to the reference excludes 0.
"""
function warping_record(cfg::AbstractDict, plan, dirs)
    A = cfg["ablation"]; v = A["variant"]; vi = A["in_sample_variant"]; ref = A["reference"]; stats = String.(A["stats"])
    summaries = Dict(k => read_summary(joinpath(d, "summary.tsv")) for (k, d) in dirs.evaluate)
    paired = Dict(k => read_summary(joinpath(d, PAIRED_FILE)) for (k, d) in dirs.evaluate)
    runs = map(sort(plan.rows; by = r -> (r.chrome, r.prefix, r.family, r.mode))) do r
        S = summaries[r.chrome]; P = paired[r.chrome]
        scores = Dict(x => _variant_scores(S, r.label, x) for x in (v, vi))
        start = Dict(x => _variant_scores(S, r.start.label, x) for x in (v, vi))
        (; chrome = r.chrome, capture = r.capture, prefix = r.prefix, family = r.family, kind = r.kind, case = r.case, mode = r.mode,
         variant_case = r.variant, label = r.label, start_label = r.start.label, scores, start_scores = start,
         beats_start = scores[v].median_px < start[v].median_px,
         vs_reference = r.mode == ref ? nothing : _paired(P, r.label, "$(r.prefix)_$(r.family)_$ref", v, stats),
         vs_start = _paired(P, r.label, r.start.label, v, stats),
         run = warping_run_summary(dirs.refine[r.variant], dirs.start[r.variant], r.kind))
    end
    winners = map(unique([(r.chrome, r.prefix, r.family) for r in runs])) do (k, pre, fam)
        rs = [r for r in runs if (r.chrome, r.prefix, r.family) == (k, pre, fam)]
        b = rs[argmin([r.scores[v].median_px for r in rs])]
        (; chrome = k, prefix = pre, family = fam, best = b.mode, best_median_px = b.scores[v].median_px,
         start_median_px = b.start_scores[v].median_px, beats_start = [r.mode for r in rs if r.beats_start],
         best_vs_reference = b.vs_reference === nothing ? nothing : b.vs_reference.median_px)
    end
    return (; schema = EXPERIMENT_RECORD_SCHEMAS.warping_ablation, headline = v, in_sample = vi, reference = ref, stats,
            modes = Dict{String, Any}(m => cfg["modes"][m]["label"] for m in plan.modes), runs, winners)
end

"""
    load_warping_record(path) -> Dict

The record at `path` (`warping_record`'s fields), read through VisualGeometryToolkitIO, its schema checked.
"""
function load_warping_record(path::AbstractString)
    r = VgtkIO.read_record(path)
    r["schema"] == EXPERIMENT_RECORD_SCHEMAS.warping_ablation || throw(ArgumentError("$path: schema $(r["schema"]), not $(EXPERIMENT_RECORD_SCHEMAS.warping_ablation)"))
    return r
end

# a record value as a table cell: `nothing` empty, a Bool as true/false
_cell(x) = x === nothing ? "" : x
_get(d, ks...) = (for k in ks; d === nothing && return nothing; d = get(d, k, nothing); end; d)

"""
    warping_table_rows(r) -> Vector{Tuple}

The rows of warping_ablation.tsv (`WARPING_TABLE_COLUMNS`) of the record `r` (`load_warping_record`'s Dict): one per refinement.
"""
function warping_table_rows(r::AbstractDict)
    v = r["headline"]
    return [(x["chrome"], x["prefix"], x["family"], x["kind"], x["mode"], x["label"], x["variant_case"],
             x["scores"][v]["photos"], x["scores"][v]["dots"], x["scores"][v]["median_px"], x["scores"][v]["mean_px"], x["scores"][v]["p95_px"],
             x["scores"][v]["rms_px"], x["scores"][v]["all_median_px"], x["scores"][v]["coverage_fraction"],
             x["start_scores"][v]["median_px"], x["start_scores"][v]["mean_px"], x["start_scores"][v]["p95_px"], x["start_scores"][v]["rms_px"],
             x["beats_start"], _cell(_get(x["vs_reference"], "median_px", "difference")), _cell(_get(x["vs_reference"], "median_px", "lo")),
             _cell(_get(x["vs_reference"], "median_px", "hi")), x["vs_start"]["median_px"]["difference"], x["vs_start"]["median_px"]["lo"],
             x["vs_start"]["median_px"]["hi"], x["run"]["rel_rms_start"], x["run"]["rel_rms_final"], x["run"]["iterations"], x["run"]["retcode"],
             x["run"]["stop_rule"], _cell(x["run"]["heights_rms_mm"]), _cell(x["run"]["heights_max_mm"]), _cell(x["run"]["inplane_aspect"]),
             _cell(x["run"]["inplane_skew"]), x["scores"][v]["retcode"]) for x in r["runs"]]
end

# ---- the figure and the table (src/figure_style.jl: the rows `warping start` and `warping <mode>`, the style sheet, the preview) ----

"The style row of a mode's series (src/figure_style.jl STYLE_ROWS `warping <mode>`) and of the starts' (`warping start`)."
warping_style(mode::AbstractString) = style_row("warping $mode")

"""
    warping_fragment(r, t) -> String

The figure of the record `r` under `[figure]` `t`: per capture set and model (the x axis, `<prefix>-<family>`) the headline median of
the start and of every mode, each a series of marks in its style row's style, the legend above the axes (no text inside them), a
comment header naming the record. No width or height.
"""
function warping_fragment(r::AbstractDict, t::AbstractDict)
    v = r["headline"]
    xs = unique(["$(x["prefix"])-$(x["family"])" for x in r["runs"]])
    modes = sort(collect(keys(r["modes"])))
    point(x, y) = @sprintf("(%s,%.5f)", x, y)
    starts = unique([("$(x["prefix"])-$(x["family"])", x["start_scores"][v]["median_px"]) for x in r["runs"]])
    io = IOBuffer()
    println(io, "% warping_ablation.tex — the board warping ablation: the $v Thorlabs median per capture set and model of the start and of every mode.")
    println(io, "% Generated by experiments src/warping_ablation.jl from warping_ablation.json. \\input figure_styles.tex first (src/figure_style.jl).")
    println(io, "\\begin{tikzpicture}")
    println(io, "\\begin{axis}[rc axis, symbolic x coords={$(join(xs, ','))}, xtick=data, x tick label style={rotate=60, anchor=east}, ",
            "enlarge x limits=0.04, ymin=0, ylabel={$(t["ylabel"])}, legend columns=4, legend style={at={(0.5,1.03)}, anchor=south}]")
    println(io, "  \\addplot[$(warping_style("start").name), only marks] coordinates {", join((point(x, y) for (x, y) in starts), ' '),
            "};\\addlegendentry{$(warping_style("start").label)}")
    for m in modes
        pts = [point("$(x["prefix"])-$(x["family"])", x["scores"][v]["median_px"]) for x in r["runs"] if x["mode"] == m]
        println(io, "  \\addplot[$(warping_style(m).name), only marks] coordinates {", join(pts, ' '), "};\\addlegendentry{$(warping_style(m).label)}")
    end
    println(io, "\\end{axis}")
    println(io, "\\end{tikzpicture}")
    return String(take!(io))
end

"""
    warping_table(r, t) -> String

The booktabs table of the record `r` under `[figure]` `t` (its `label`, `caption`): per capture set and model the headline median [px]
of the start and of every mode, each non-reference mode's difference to the reference with its paired 95 % interval (* where it
excludes 0), and the mode of the lowest median.
"""
function warping_table(r::AbstractDict, t::AbstractDict)
    v = r["headline"]; ref = r["reference"]
    modes = sort(collect(keys(r["modes"]))); others = [m for m in modes if m != ref]
    io = IOBuffer()
    println(io, "% table_warping_ablation.tex — generated by experiments src/warping_ablation.jl from warping_ablation.json ($v).")
    println(io, "\\begin{table*}[t]\\centering\\small")
    println(io, "\\caption{$(t["caption"])}\\label{$(t["label"])}")
    println(io, "\\begin{tabular}{ll", "r"^(2 + length(others)), "l"^length(others), "l}")
    println(io, "\\toprule")
    println(io, "set & model & start & ", join(["$m" for m in others], " & "), " & $ref & ", join(["$m \$-\$ $ref [95\\,\\%]" for m in others], " & "), " & best \\\\")
    println(io, "\\midrule")
    for w in r["winners"]
        rs = Dict(x["mode"] => x for x in r["runs"] if (x["chrome"], x["prefix"], x["family"]) == (w["chrome"], w["prefix"], w["family"]))
        med(m) = @sprintf("%.3f", rs[m]["scores"][v]["median_px"])
        ci(m) = (d = rs[m]["vs_reference"]["median_px"];
                 @sprintf("%+.3f [%+.3f, %+.3f]%s", d["difference"], d["lo"], d["hi"], d["excludes_zero"] ? "*" : ""))
        println(io, "$(w["prefix"]) & $(w["family"]) & ", @sprintf("%.3f", w["start_median_px"]), " & ", join(med.(others), " & "), " & ", med(ref), " & ",
                join(ci.(others), " & "), " & $(w["best"]) \\\\")
    end
    println(io, "\\bottomrule")
    println(io, "\\end{tabular}")
    println(io, "\\end{table*}")
    return String(take!(io))
end

"""
    write_warping_figure(dir, r, t) -> (; fragment, table, styles, preview)

The figure and the table of the record `r` under `[figure]` `t` into `dir`: the fragment warping_ablation.tex, the table
table_warping_ablation.tex, the project's style sheet figure_styles.tex (`write_figure_styles`) and a standalone preview.tex of the
figure (`write_figure_preview`; preview.pdf where latexmk is installed) — vector output only.
"""
function write_warping_figure(dir::AbstractString, r::AbstractDict, t::AbstractDict)
    mkpath(dir)
    frag = joinpath(dir, "warping_ablation.tex"); tab = joinpath(dir, "table_warping_ablation.tex")
    write(frag, warping_fragment(r, t)); write(tab, warping_table(r, t))
    sty = write_figure_styles(joinpath(dir, "figure_styles.tex"))
    write_figure_preview(dir, "warping_ablation"; size = ("16cm", "6cm"))
    return (; fragment = frag, table = tab, styles = sty, preview = joinpath(dir, "preview.tex"))
end
