# =============================================================================
# src/radiometric_refinement.jl — the config of record of the refinements of the paper's chrome-board table (tab:refinement-chrome;
# configs/experiments/radiometric_refinement.toml), shared by scripts/radiometric_refinement.jl and the driver that scores the refined models
# on the chrome board (scripts/chrome_evaluation.jl, its [upstream] radiometric_refinement). Included after src/drivers.jl.
#
# The cases are the shared case machinery of src/drivers.jl (prepare sets, [views], [cases.<name>] with a [pipeline] table run by
# bin/refine.jl into refine-<key>/), as in night_stars; the evaluation is not here: each lens's chrome photographs are a config of
# scripts/chrome_evaluation.jl whose [evaluate] models name these cases.
#
# `[material_grid]`: the grid the material start of every case is searched on (stage material_grid, src/material_grid.jl), required:
# per quantity its range and its number of cells, or `source`, the config that states them.
#
# `[pending]`: what the config of record does not hold yet — a value or an input of the run of record that is not in this repository
# (each entry `<what> = "<why, and where it is>"`). Description, not a setting: it keys no run. While it is not empty the config
# loads and plans, and the refine stage refuses to run (`refuse_pending`): a run with a stand-in value would be a run of another
# config, under the name of this one.
# =============================================================================

"The tables of configs/experiments/radiometric_refinement.toml."
const REFINEMENT_CHROME_TABLES = ("experiment", "inputs", "image_regions", "runs", "prepare", "views", "material_grid", "cases", "pending")
"The keys of a `[cases.<name>]` table: the model kind it writes (lens | grid), its photographs (a [prepare] set), its views (a [views] name) and its [pipeline]."
const REFINEMENT_CHROME_CASE_KEYS = ("kind", "images", "views", "pipeline")

"""
    load_radiometric_refinement_config(path) -> Dict

The config of record of the chrome-table refinements, every table and key checked (`load_driver_config`; every case key required,
its photographs a [prepare] set, its views a [views] list, its kind lens | grid; `[material_grid]` required and resolved,
`material_grid_table`; `[pending]` a table of strings).
"""
function load_radiometric_refinement_config(path::AbstractString)
    c = load_driver_config(path, "radiometric_refinement"; tables = REFINEMENT_CHROME_TABLES, case_keys = REFINEMENT_CHROME_CASE_KEYS)
    for t in ("prepare", "views", "cases")
        haskey(c, t) && c[t] isa AbstractDict && !isempty(c[t]) || throw(ArgumentError("$path: no [$t] table"))
    end
    check_tables(c, path, "cases", REFINEMENT_CHROME_CASE_KEYS; required = REFINEMENT_CHROME_CASE_KEYS)
    check_prepare_tables(c, path)
    for (s, t) in c["prepare"]
        r = get(c["inputs"], t["raw"], nothing)
        r !== nothing && r["stage"] == "prepare" || throw(ArgumentError("$path [prepare.$s]: raw = $(repr(t["raw"])) is not an input of stage prepare"))
    end
    for (n, v) in c["views"]
        v isa AbstractVector && all(x -> x isa AbstractString, v) || throw(ArgumentError("$path [views] $n must be a list of image names"))
    end
    for (n, t) in c["cases"]
        where = "$path [cases.$n]"
        t["kind"] in REFINEMENT_KINDS || throw(ArgumentError("$where: kind $(repr(t["kind"])) is not one of $(REFINEMENT_KINDS)"))
        haskey(c["prepare"], t["images"]) || throw(ArgumentError("$where: no [prepare.$(t["images"])]"))
        t["views"] isa AbstractString && haskey(c["views"], t["views"]) || throw(ArgumentError("$where: views must name a [views] list"))
    end
    c["material_grid"] = material_grid_table(c, path)               # the resolved grid: its own keys, whichever config states them
    p = get(c, "pending", Dict{String,Any}())
    p isa AbstractDict && all(v -> v isa AbstractString && !isempty(v), values(p)) ||
        throw(ArgumentError("$path [pending]: each entry is `<what> = \"<why, and where it is>\"`"))
    return c
end

"""
    refuse_pending(cfg, path)

An `ArgumentError` listing `[pending]` when the config of record still has pending values or inputs; nothing when it has none.
"""
function refuse_pending(cfg::AbstractDict, path::AbstractString)
    p = get(cfg, "pending", Dict{String,Any}())
    isempty(p) && return nothing
    throw(ArgumentError("$path is not complete — its [pending] values are not in this repository, so no run of it is a run of record:\n" *
                        join(("  $k: $v" for (k, v) in sort(collect(p); by = first)), "\n")))
end
