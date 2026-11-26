# =============================================================================
# src/chrome_evaluation_paper.jl — the chrome evaluation's paper stages: the paper's chrome outputs straight from the chrome sets'
# evaluate products, driven by the paper config of record (configs/experiments/chrome_evaluation_paper.toml: [chrome.<key>] one
# chrome set's config each, [figures.<name>] the residual maps, [tables.<name>] the tables, [render] their digits):
#   tables   every [tables.<name>] (`write_chrome_tables`): table_<name>.tsv / .md / .tex (src/figures.jl write_summary_tables) — the
#            paper's tab:refinement-chrome (table_refinement_chrome.tex), tab:defocus (table_defocus.tex), chrome_all — and the
#            number macros of the tables with a `numbers_prefix` and the convergence counts (`chrome_table_macros`): numbers.tex
#   figures  every [figures.<name>] (`chrome_residual_figures`): the start and refined models' chrome residuals under one rotation
#            centre (fig:color-wheels, src/figures.jl figure_wheel_map)
# The products a stage reads are named by the products NamedTuple (`chrome_products`: each set's evaluate run of record; a test hands
# fixtures). scripts/chrome_evaluation_paper.jl runs these stages. Included after src/drivers.jl,
# src/radiometric_refinement.jl, src/chrome_evaluation.jl and src/figures.jl (module Figures).
# =============================================================================

using VisualGeometryToolkitEval.Experiments: read_provenance
using .Figures: figure_wheel_map, check_wheel_render, SUMMARY_STATS, summary_rows, summary_table, write_summary_tables, table_numbers!

"The tables of a chrome evaluation paper config."
const CHROME_PAPER_TABLES = ("experiment", "inputs", "chrome", "figures", "tables", "render")
"The keys every row of a `[tables.<name>]` names (and `macro`, a numbered table's; `camera`, the rows `mark_best` compares)."
const CHROME_TABLE_ROW_KEYS = ("set", "model", "chrome", "start", "refined")

"""
    load_chrome_paper_config(path) -> Dict

The chrome evaluation paper config at `path` checked (an `ArgumentError` names the defect): `[chrome.<key>] config` each a chrome
set's config of this project; `[render]` `significant_digits` and `change_digits` (integers ≥ 1);
`[figures.<name>]` the residual maps (style `wheels` | `arrows`, each panel's view `sensor` | `overflow` | `zoom`); `[tables.<name>]`
`rows` (each: `set`, `model`, `chrome` — a `[chrome.<key>]` —, `start` — `""` for a model without an evaluated start —, `refined`;
`macro` per row of a table with a `numbers_prefix`; optionally `camera`, the rows one best value is chosen among — default its
`chrome`, the rows scored on the same chrome photographs), `variant` (default "motion"; the paper's tables: "motion_heldout", the
headline), `in_sample_variant` (optional, another variant: the refinement's in-sample score and the gap beside the headline, in the
long form, the md and the numbers — never a column of the tex, `summary_table`), `confidence` (optional switch, `"on"` | `"off"`,
default off: each row's refined − start interval from its set's paired bootstrap, paired.tsv; the long form and the md only), `stats`
(of `SUMMARY_STATS`, default all four), `layout`, `label`, `caption`, `worse_color`, `mark_best` and `inline_change` (switches, `"on"` | `"off"`, default off;
`inline_change` with layout "blocks", Ties's format: src/figures.jl `write_summary_tables`).
"""
function load_chrome_paper_config(path::AbstractString)
    c = load_driver_config(path, "chrome_evaluation_paper"; tables = CHROME_PAPER_TABLES)
    check_tables(c, path, "chrome", ("config",); required = ("config",))
    for (k, t) in c["chrome"]
        isfile(projectdir(t["config"])) || throw(ArgumentError("$path [chrome.$k]: no config $(t["config"])"))
    end
    haskey(c, "render") || throw(ArgumentError("$path: no [render] table"))
    R = check_required_keys(c["render"], ("significant_digits", "change_digits"), "$path [render]")
    for k in ("significant_digits", "change_digits")
        R[k] isa Integer && R[k] >= 1 || throw(ArgumentError("$path [render] $k: an integer ≥ 1"))
    end
    # the residual maps (fig:color-wheels: src/figures.jl figure_wheel_map; style "arrows" with cells_x, len_cells_x and a panel's view)
    check_tables(c, path, "figures", ("asset", "variant", "size", "cols", "dpi", "title", "panels", "style", "cells_x", "len_cells_x");
                 required = ("asset", "variant", "panels"))
    for (f, t) in get(c, "figures", Dict{String,Any}())
        check_wheel_render(get(t, "style", "wheels"), (get(p, "view", "sensor") for p in t["panels"]), "$path [figures.$f]")
        for p in t["panels"]
            haskey(c["chrome"], get(p, "chrome", "")) || throw(ArgumentError("$path [figures.$f]: a panel's chrome $(repr(get(p, "chrome", ""))) is not a [chrome.<key>]"))
        end
    end
    check_tables(c, path, "tables", ("variant", "in_sample_variant", "confidence", "stats", "rows", "layout", "label", "caption", "worse_color",
                                     "mark_best", "inline_change", "numbers_prefix");
                 required = ("rows",))
    for (n, t) in get(c, "tables", Dict{String,Any}())
        haskey(t, "in_sample_variant") && !(t["in_sample_variant"] isa AbstractString && t["in_sample_variant"] != get(t, "variant", "motion")) &&
            throw(ArgumentError("$path [tables.$n] in_sample_variant: a variant (a string) other than the table's `variant`"))
        table_switch(t, "confidence", "$path [tables.$n]")
        t["rows"] isa AbstractVector || throw(ArgumentError("$path [tables.$n]: `rows` must be an array of tables"))
        table_switch(t, "mark_best", "$path [tables.$n]")
        table_switch(t, "inline_change", "$path [tables.$n]") && get(t, "layout", "per_set") != "blocks" &&
            throw(ArgumentError("$path [tables.$n]: inline_change is the layout \"blocks\"'s"))
        for (i, r) in enumerate(t["rows"])
            where = "$path [tables.$n] row $i"
            r isa AbstractDict || throw(ArgumentError("$where must be a table"))
            VgtkCore.check_keys(r, (CHROME_TABLE_ROW_KEYS..., "macro", "camera"), where)
            haskey(r, "camera") && !(r["camera"] isa AbstractString && !isempty(r["camera"])) && throw(ArgumentError("$where: `camera` is a name"))
            haskey(t, "numbers_prefix") && !(get(r, "macro", "") isa AbstractString && !isempty(r["macro"])) &&
                throw(ArgumentError("$where: a table with `numbers_prefix` names each row's `macro` (letters)"))
            for k in CHROME_TABLE_ROW_KEYS
                haskey(r, k) && r[k] isa AbstractString || throw(ArgumentError("$where lacks `$k` (a string)"))
                k == "start" || !isempty(r[k]) || throw(ArgumentError("$where: `$k` is empty"))
            end
            haskey(c["chrome"], r["chrome"]) || throw(ArgumentError("$where: chrome \"$(r["chrome"])\" is not a [chrome.<key>] of this config"))
        end
        for st in get(t, "stats", collect(SUMMARY_STATS))
            st in SUMMARY_STATS || throw(ArgumentError("$path [tables.$n]: stat \"$st\" is not one of $(SUMMARY_STATS)"))
        end
    end
    return c
end

"A `[tables.<name>]` `t`'s switch `k` (`confidence`, `mark_best`, `inline_change`): RadiometricCamera's `read_switch`, `\"off\"` when absent."
table_switch(t::AbstractDict, k::AbstractString, where::AbstractString) = RadiometricCamera.read_switch(get(t, k, "off"), "$where $k")

"The chrome sets of paper config `P`, sorted."
chrome_paper_sets(P::AbstractDict) = sort(collect(keys(P["chrome"])))
"The config file of chrome set `k` of paper config `P`."
chrome_set_path(P::AbstractDict, k::AbstractString) = projectdir(P["chrome"][k]["config"])

"""
    chrome_products(P) -> (products, cfgs, runs)

The evaluate products of record of paper config `P`'s chrome sets: each set's config (`load_chrome_evaluation_config`), its evaluate
product (`evaluate_product`) and the products the paper stages read, by chrome set `k` — `(; summary, vectors, retcode,
provenance, paired)`: `summary(k)` the path of its evaluate product's summary.tsv, `paired(k)` its paired.tsv (the paired bootstrap), `vectors(k, label, variant)` model `label`'s residual vectors
(vectors_<label>_<variant>.json), `retcode(k, label)` the name of the LM `ReturnCode` of the radiometric refinement model `label`
scores (`chrome_refinement_retcode`, each set's refinement config read once), `provenance(k)` `(; run_key, commit)` of its evaluate
product (Eval's `read_provenance`). A test hands its own fixtures in the same shape.
"""
function chrome_products(P::AbstractDict)
    cfgs = Dict(k => load_chrome_evaluation_config(chrome_set_path(P, k)) for k in chrome_paper_sets(P))
    runs = Dict(k => evaluate_product(cfgs[k]) for k in chrome_paper_sets(P))
    refinements = Dict{String, Any}()                                                # each set's refinement config, read once, when a row needs it
    function retcode(k, label)
        where = "chrome $k ($(P["chrome"][k]["config"]))"
        haskey(cfgs[k], "upstream") || throw(ArgumentError("$where: model $label is a table's refined row, but the set has no [upstream] refinements"))
        return chrome_refinement_retcode(cfgs[k], get!(() -> upstream_refinements(cfgs[k]), refinements, k), label; where)
    end
    products = (; summary = k -> joinpath(product_dir(runs[k]), "summary.tsv"), paired = k -> joinpath(product_dir(runs[k]), PAIRED_FILE),
                vectors = (k, label, variant) -> joinpath(product_dir(runs[k]), "vectors_$(label)_$(variant).json"),
                retcode,
                provenance = k -> (; run_key = String(runs[k].key), commit = String(read_provenance(product_dir(runs[k]))["gitcommit"])))
    return (products, cfgs, runs)
end

"""
    chrome_refinement_retcode(cfg, refinements, label; where) -> String

The LM `ReturnCode` (its name) of the radiometric refinement that chrome set config `cfg`'s model `label` scores: its `[evaluate]
models` row's `case`, that case's refine product in the refinement config `refinements` (`upstream_refinements(cfg)`, `refinement_product`), its
parameters.json `retcode` (RadiometricCamera's `rc/run_parameters` record, `load_run_parameters`); `"unfinished"` when the run holds
no parameters.json. A model with no `case` is an `ArgumentError` (`where` names the set): a table's refined row reads a refinement.
"""
function chrome_refinement_retcode(cfg::AbstractDict, refinements::AbstractDict, label::AbstractString; where::AbstractString)
    rows = filter(x -> x["label"] == label, cfg["evaluate"]["models"])
    length(rows) == 1 && haskey(only(rows), "case") || throw(ArgumentError("$where: model $label is not one refinement case"))
    f = joinpath(product_dir(refinement_product(refinements, only(rows)["case"])), "parameters.json")
    isfile(f) || return "unfinished"
    return String(RadiometricCamera.load_run_parameters(f).retcode)          # the run's rc/run_parameters record
end

"""
    chrome_table_rows(P, name, products; cache = Dict()) -> Vector{NamedTuple}

The rows of `[tables.<name>]` with their summary rows (`Figures.summary_rows` of each chrome set's evaluation, the table's variant):
`(; set, model, variant, name_part, chrome, camera, start, refined, in_sample, ci, refinement_retcode)`, `start` `nothing` for a row
whose `start` is `""`, `camera` the row's `camera` or else its `chrome` (the rows `mark_best` compares: one camera's, scored on the
same chrome photographs), `in_sample` the refined model's row in the table's `in_sample_variant` (`nothing` without one), `ci` with
`confidence` "on" the row's refined − start differences of the set's paired bootstrap (`products.paired`, region "all", the table's
variant) by statistic, `(; difference, lo, hi, excludes_zero)` (`nothing` without `confidence` or without a start).
A label absent from its summary is an `ArgumentError` (a missing evaluation must fail the stage). `cache` holds each chrome set's
parsed summary (by set): `chrome_tables_rows` shares one across the tables.
"""
function chrome_table_rows(P::AbstractDict, name::AbstractString, products; cache::AbstractDict = Dict{String, Any}())
    t = P["tables"][name]; v = String(get(t, "variant", "motion")); iv = haskey(t, "in_sample_variant") ? String(t["in_sample_variant"]) : nothing
    function pick(k, label, v = v)
        rows = get!(() -> summary_rows(products.summary(k)), cache, k)
        haskey(rows, (label, v)) || throw(ArgumentError("[tables.$name]: no row $label / $v in $(products.summary(k))"))
        return rows[(label, v)]
    end
    conf = table_switch(t, "confidence", "[tables.$name]")
    function ci(k, model, ref)
        ps = get!(() -> read_summary(products.paired(k)), cache, "paired:" * k)
        out = Dict{String, Any}()
        for p in ps
            (p["model"], p["reference"], p["variant"], p["region"]) == (model, ref, v, "all") || continue
            out[String(p["stat"])] = (; difference = Float64(p["difference"]), lo = Float64(p["lo"]), hi = Float64(p["hi"]), excludes_zero = p["excludes_zero"] in (true, "true"))
        end
        isempty(out) && throw(ArgumentError("[tables.$name]: no paired difference $model − $ref / $v in $(products.paired(k))"))
        return out
    end
    return [(; set = String(r["set"]), model = String(r["model"]), variant = v, name_part = String(get(r, "macro", "")), chrome = String(r["chrome"]),
              camera = String(get(r, "camera", r["chrome"])),
              start = isempty(r["start"]) ? nothing : pick(r["chrome"], r["start"]), refined = pick(r["chrome"], r["refined"]),
              in_sample = iv === nothing ? nothing : pick(r["chrome"], r["refined"], iv),
              ci = conf && !isempty(r["start"]) ? ci(r["chrome"], r["refined"], r["start"]) : nothing,
              refinement_retcode = products.retcode(r["chrome"], r["refined"])) for r in t["rows"]]
end

"""
    chrome_figures_product(P, runs) -> Product

The figures product of paper config `P` (its output numbers.tex, written last; its tables and figures kept in `assets/` and published
to plotsdir(), `produce_figures`): keyed by `[figures]`, `[tables]` and `[render]` and the keys of the sets' evaluate products `runs`
(`chrome_products`) — a changed set re-renders, a render-only change re-scores nothing.
"""
chrome_figures_product(P::AbstractDict, runs::AbstractDict) =
    driver_product("figures", Dict{String,Any}("experiment" => P["experiment"], "figures" => get(P, "figures", Dict{String,Any}()),
                                               "tables" => get(P, "tables", Dict{String,Any}()), "render_files" => P["render"],
                                               "upstream" => Dict{String,Any}(k => runs[k].key for k in chrome_paper_sets(P)));
                   output = "numbers.tex", threads = nothing)

"The table names of paper config `P`, sorted."
chrome_tables(P::AbstractDict) = sort(collect(keys(get(P, "tables", Dict{String,Any}()))))

"""
    chrome_tables_rows(P, products) -> Dict{String, Vector}

Every `[tables.<name>]` of paper config `P` by name, its rows (`chrome_table_rows`), built once for the tables and their macros.
"""
chrome_tables_rows(P::AbstractDict, products) =
    (cache = Dict{String, Any}(); Dict(n => chrome_table_rows(P, n, products; cache) for n in chrome_tables(P)))   # each set's summary parsed once

"""
    write_chrome_tables(P, dirs, rows) -> Vector{String}

Every `[tables.<name>]` of paper config `P` (its rows `rows[name]`, `chrome_tables_rows`) into each directory of `dirs`:
table_<name>.tsv (the long form, every number raw), .md and .tex (src/figures.jl `write_summary_tables`: [render]
significant_digits of a pixel value, change_digits of a change in percent; the table's layout, label, caption, worse_color, mark_best
and inline_change).
Returns the files written.
"""
function write_chrome_tables(P::AbstractDict, dirs, rows::AbstractDict)
    R = P["render"]
    number = x -> Reproduce.tex_number(x; sigdigits = R["significant_digits"]); change = x -> Reproduce.tex_number(x; sigdigits = R["change_digits"])
    files = String[]
    for n in chrome_tables(P)
        t = P["tables"][n]
        stats = String.(get(t, "stats", collect(SUMMARY_STATS)))
        for s in summary_table(rows[n]; stats)
            println(">> table $n · $(s.set) · $(s.model) [$(s.variant)] $(s.stat): ", isnan(s.start) ? "" : "$(round(s.start; digits = 4)) → ",
                    round(s.refined; digits = 4), isnan(s.change_pct) ? "" : " ($(round(s.change_pct; digits = 1)) %)", "; photos $(s.photos_start) / $(s.photos_refined)")
        end
        for dir in dirs
            append!(files, write_summary_tables(dir, n, rows[n]; stats, number, change, layout = String(get(t, "layout", "per_set")),
                                                label = String(get(t, "label", "")), caption = String(get(t, "caption", "")),
                                                worse_color = String(get(t, "worse_color", "")), mark_best = table_switch(t, "mark_best", "[tables.$n]"),
                                                inline_change = table_switch(t, "inline_change", "[tables.$n]")))
        end
    end
    return files
end

"""
    chrome_table_macros(P, path, rows, provenance; driver) -> PaperNumbers

The number macros of the tables of paper config `P` (read from `path`; their rows `rows`, `chrome_tables_rows`): of a table whose
config states a `numbers_prefix`, `table_numbers!` (per row its `macro`; the residuals in [render] significant_digits, the changes in
change_digits), each number's provenance the chrome evaluation run its row reads (`provenance(chrome)`: its run key and recorded
gitcommit). No convergence count: the paper reports none (owner 2026-11-16); the rows' termination records stay in their tsv/json.
"""
function chrome_table_macros(P::AbstractDict, path::AbstractString, rows::AbstractDict, provenance; driver::AbstractString)
    R = P["render"]
    nums = Reproduce.PaperNumbers(; driver, config = relpath(path, projectdir()), run_key = "per-number", commit = "per-number")   # each number states its run's
    for n in chrome_tables(P)
        t = P["tables"][n]
        if haskey(t, "numbers_prefix")
            prov = Dict(c => provenance(c) for c in unique(r.chrome for r in rows[n]))
            table_numbers!(nums, String(t["numbers_prefix"]), rows[n]; stats = String.(get(t, "stats", collect(SUMMARY_STATS))),
                           sigdigits = R["significant_digits"], change_sigdigits = R["change_digits"],
                           label = String(get(t, "label", "")), provenance = r -> prov[r.chrome])
        end
    end
    return nums
end

"""
    chrome_residual_figures(P, dir, products) -> Vector{String}

Every `[figures.<name>]` of paper config `P` into `dir`: the residual maps of its panels' models (`products.vectors` of the panel's
chrome set, model and the figure's variant; src/figures.jl `figure_wheel_map`). Returns the files written.
"""
chrome_residual_figures(P::AbstractDict, dir::AbstractString, products) =
    [figure_wheel_map([(String(p["label"]), "scatter", products.vectors(p["chrome"], p["model"], t["variant"]), Float64(p["vmax"])) for p in t["panels"]];
                      render = t, dir) for (_, t) in sort(collect(get(P, "figures", Dict{String,Any}())); by = first)]
