# =============================================================================
# src/panorama_figure.jl — the supplement's panoramas figure (scripts/panorama_figure.jl, its config of record
# configs/supplement/panorama_figure.toml): per lens the panorama of its pure-rotation sweep composed from the frames as they are
# and from the frames with the learned vignetting divided out, side by side. Nothing is computed here: every panorama is a file of
# a lens's `panoramas` product (scripts/panorama_stitching.jl on the lens's config), read by its key; the drawing code
# (src/paper_figures/panoramas.jl) puts each through one display curve and reduces it to the panels' width.
#
# Included after src/drivers.jl, src/panorama_example.jl, src/radiometric_refinement.jl and src/panorama_stitching.jl.
# =============================================================================

"The tables of configs/supplement/panorama_figure.toml."
const PANORAMA_FIGURE_TABLES = ("experiment", "inputs", "lenses", "figure")
"""The keys of a `[lenses.<lens>]` table, every one required: the lens's panorama_stitching config of record whose panoramas are
drawn (`config`, relative to the project), the lens's name in two lines (`label`: LaTeX, the two parted by `\\\\`; turned at the
left of its row, and in one line — a space for the break — over a row in one column and in the caption), its aperture in the
caption (`aperture`, LaTeX) and the name its panels' files begin with (`panel`: <asset_dir>/<panel>-<column>.tex, .png)."""
const PANORAMA_FIGURE_LENS_KEYS = ("config", "label", "aperture", "panel")
"""The keys of `[figure]`, every one required: the figure's name under plotsdir() — its file <asset_dir>.tex, the folder of its
panels and its caption's file (`asset_dir`), the lenses from the top (`order`, every lens of `[lenses]` once), the columns from
the left (`columns`, each a table of `PANORAMA_FIGURE_COLUMN_KEYS`), the display curve every panorama goes through (`gain`,
`gamma`) and the panels' width in pixels (`width_px`)."""
const PANORAMA_FIGURE_KEYS = ("asset_dir", "order", "columns", "gain", "gamma", "width_px")
"The keys of one entry of `[figure] columns`: the end of its panels' file names (`panel`), the candidate of the panorama stage it shows (`candidate`: panorama_<candidate>.png) and its title (`title`, LaTeX)."
const PANORAMA_FIGURE_COLUMN_KEYS = ("panel", "candidate", "title")

"""
    load_panorama_figure_config(path) -> Dict

The config of record of the panoramas figure, every table and key checked: each `[lenses.<lens>]` names a panorama_stitching config
of this project, a label of two lines, an aperture and a panel no other lens names; `[figure] asset_dir` a non-empty string,
`order` every lens once, `columns` two tables of `PANORAMA_FIGURE_COLUMN_KEYS` (the caption is written for two: the frames as they
are, then the frames with the learned vignetting divided out), each with a panel of its own, `gain` and `gamma` positive and
`width_px` a positive integer.
"""
function load_panorama_figure_config(path::AbstractString)
    c = load_driver_config(path, "panorama_figure"; tables = PANORAMA_FIGURE_TABLES)
    for t in ("lenses", "figure")
        haskey(c, t) && c[t] isa AbstractDict && !isempty(c[t]) || throw(ArgumentError("$path: no [$t] table"))
    end
    for (k, t) in c["lenses"]
        where = "$path [lenses.$k]"
        check_required_keys(t, PANORAMA_FIGURE_LENS_KEYS, where)
        all(x -> t[x] isa AbstractString && !isempty(t[x]), PANORAMA_FIGURE_LENS_KEYS) || throw(ArgumentError("$where: every key is a non-empty string"))
        isfile(projectdir(t["config"])) || throw(ArgumentError("$where: no config $(t["config"])"))
        length(split(t["label"], "\\\\")) == 2 || throw(ArgumentError("$where: label is the lens's name in two lines, \"<first>\\\\\\\\<second>\", got $(repr(t["label"]))"))
    end
    allunique(t["panel"] for t in values(c["lenses"])) || throw(ArgumentError("$path [lenses]: every lens names a panel of its own (its panels' files)"))
    f = check_required_keys(c["figure"], PANORAMA_FIGURE_KEYS, "$path [figure]")
    f["asset_dir"] isa AbstractString && !isempty(f["asset_dir"]) || throw(ArgumentError("$path [figure] asset_dir: the figure's name, a non-empty string"))
    f["order"] isa AbstractVector && sort(String.(f["order"])) == sort(collect(keys(c["lenses"]))) ||
        throw(ArgumentError("$path [figure] order: every lens of [lenses] once, from the top ($(join(sort(collect(keys(c["lenses"]))), ", ")))"))
    cols = f["columns"]
    cols isa AbstractVector && length(cols) == 2 || throw(ArgumentError("$path [figure] columns: two columns — the frames as they are, then the frames with the learned vignetting divided out"))
    for (i, col) in enumerate(cols)
        check_required_keys(col, PANORAMA_FIGURE_COLUMN_KEYS, "$path [figure] columns[$i]")
        all(x -> col[x] isa AbstractString && !isempty(col[x]), PANORAMA_FIGURE_COLUMN_KEYS) || throw(ArgumentError("$path [figure] columns[$i]: every key is a non-empty string"))
    end
    allunique(col["panel"] for col in cols) || throw(ArgumentError("$path [figure] columns: every column names a panel of its own"))
    for k in ("gain", "gamma")
        f[k] isa Real && f[k] > 0 || throw(ArgumentError("$path [figure] $k: the display curve's $k, a positive number"))
    end
    f["width_px"] isa Integer && f["width_px"] > 0 || throw(ArgumentError("$path [figure] width_px: the panels' width in pixels, a positive integer"))
    return c
end

"""
    panorama_figure_sources(cfg; path = "the config") -> Dict

Per lens of `cfg`, `lens => (; config, product)`: its panorama_stitching config (the path as the config states it) and that config's
panoramas product (`panoramas_product`). A candidate of `[figure] columns` that the lens's config does not render (neither
"original", the frames as they are, nor one of its `[maps]`) is an `ArgumentError` naming `path`, the file `cfg` was read from.
"""
function panorama_figure_sources(cfg::AbstractDict; path::AbstractString = "the config")
    return Dict{String, Any}(k => begin
            pc = load_panorama_stitching_config(projectdir(t["config"]))
            for col in cfg["figure"]["columns"]
                col["candidate"] == "original" || haskey(pc["maps"], col["candidate"]) ||
                    throw(ArgumentError("$path [figure] columns: $(t["config"]) renders no candidate $(repr(col["candidate"])) (original, $(join(sort(collect(keys(pc["maps"]))), ", ")))"))
            end
            (; config = String(t["config"]), product = panoramas_product(pc, panorama_stitching_upstream(pc; path = t["config"])))
        end for (k, t) in cfg["lenses"])
end

"""
    panorama_figure_product(cfg, sources = panorama_figure_sources(cfg)) -> Product

The figure's product (its output numbers.tex, written last; the figure's files in `assets/`): keyed by the config's `[lenses]` and
`[figure]` and the keys of the panoramas products it reads. One result at any thread count.
"""
panorama_figure_product(cfg::AbstractDict, sources = panorama_figure_sources(cfg)) =
    driver_product("figures", Dict{String, Any}("experiment" => cfg["experiment"], "lenses" => cfg["lenses"], "figure" => cfg["figure"],
                                                "upstream" => Dict{String, Any}(k => s.product.key for (k, s) in sources));
                   output = "numbers.tex", threads = nothing)

"""
    panorama_figure_settings(cfg) -> NamedTuple

The settings of `[figure]` and `[lenses]` of `cfg` as the drawing code takes them (src/paper_figures/panoramas.jl
`figure_panoramas`): `asset_dir`, `order`, per lens its `panels`, `labels` (the two lines of its name) and `apertures`, the
`columns`, the display curve (`gain`, `gamma`) and the panels' `width`.
"""
function panorama_figure_settings(cfg::AbstractDict)
    F = cfg["figure"]; L = cfg["lenses"]; order = String.(F["order"])
    return (; asset_dir = String(F["asset_dir"]), order,
            panels = Dict{String, String}(k => L[k]["panel"] for k in order),
            labels = Dict{String, Tuple{String, String}}(k => Tuple(String.(split(L[k]["label"], "\\\\"))) for k in order),
            apertures = Dict{String, String}(k => L[k]["aperture"] for k in order),
            columns = [(; panel = String(c["panel"]), candidate = String(c["candidate"]), title = String(c["title"])) for c in F["columns"]],
            gain = F["gain"], gamma = F["gamma"], width = F["width_px"])
end
