# =============================================================================
# src/board_warping.jl — the board-warping experiment (configs/supplement/board_warping.toml; scripts/board_warping.jl): the height
# of the printed calibration boards above their plane as the radiometric refinements estimated it, for the supplement's figure
# fig:board-warping. Included after src/drivers.jl, src/radiometric_refinement.jl and src/paper_figures/PaperFigures.jl (the drawing
# code: T. Harder's figure generators, ported; board_warping.py is src/paper_figures/board_warping.jl).
#
# A second config of record of the driver, configs/supplement/board_warping_lenses.toml, draws the sets of ONE session, in which
# each board stayed in place while it was photographed with the three lenses: the same heights, lattice and panels, and tables that
# also state how far the lenses' surfaces of a board differ (its drawing code: board_warping_lenses.py, ported as
# src/paper_figures/board_warping_lenses.jl). A config names its drawing code (`[figure] drawing`).
#
# Nothing is estimated here. A refinement writes its calibrated target as radiometric_target.json (RadiometricCamera's
# `load_radiometric_target`): the geometric target with its printed page and the warping's nodal heights. The surface over the page
# is VisualRadiometryToolkitCore's `board_height` — VisualGeometryToolkitCore's height field (a thin-plate spline through the
# warping's nodes) over the page, the height along the board's normal, positive towards the camera. The piston and the two tilts
# of a board belong to its poses, so the field holds the bending alone.
#
# The figure shows one refinement per capture set: the config names each set with its case of the refinements' config, and the
# stage reads those cases alone.
#
# The stage's product holds what the drawing code reads (src/paper_figures/board_warping.jl states the formats):
#   cases.tsv                one line per case: case, set, model, refine_product (the directory of the refinement's product)
#   dump/<case>.f64          the height field on a lattice over the page
#   dump/<case>_nodes.tsv    the warping's nodes: where they lie, their stored heights and the surface there
#   meta.toml                per case the page's size and the surface's range and RMS (the sets of one session: also the in-plane
#                            aspect and skew, and per board how far the lenses' surfaces differ)
# and the drawing itself: assets/<asset_dir>.tex with its panels in assets/<asset_dir>/ (published to plotsdir()), the generated
# caption and source.toml beside them in the product (the paper's caption is written by hand).
# =============================================================================
import VisualRadiometryToolkitCore as VrtkCore
import Printf
using StaticArrays: SVector

"The tables of configs/supplement/board_warping.toml."
const BOARD_WARPING_TABLES = ("experiment", "inputs", "upstream", "heights", "sets", "figure")
"The keys of `[heights]`, every one required: the number of samples per page axis."
const BOARD_WARPING_HEIGHTS_KEYS = ("samples",)
"""The keys of `[figure]`, every one required: the drawing code (`drawing`), the name of the figure's files under plotsdir()
(`asset_dir`), the layout — the columns' labels and, per row, its label and its sets — and what the panels draw: the candidates of
a colour band's height and the most bands on each side of zero, the mesh's stride, the view and the heights' exaggeration."""
const BOARD_WARPING_FIGURE_KEYS = ("drawing", "asset_dir", "columns", "rows", "band_heights_mm", "max_bands", "mesh_stride", "view_azimuth_elevation_deg",
                                   "height_exaggeration")
"""The drawing codes a config may name (`[figure] drawing`, each a generator of src/paper_figures: `PaperFigures.FIGURES`): the
boards' surfaces from capture sets of any sessions, and the surfaces of ONE session's sets, whose tables also compare the sets of a
row (the lenses that photographed the board in place)."""
const BOARD_WARPING_DRAWINGS = ("board-warping", "board-warping-lenses")
"The keys of one entry of `[figure] rows`: the row's label and its sets, one per column."
const BOARD_WARPING_ROW_KEYS = ("label", "sets")

"""
    board_warping_model(sets; where = "[sets]") -> String

The camera model of the refinements `sets` names (`set => case`, the config's `[sets]`), read off the cases' names: the drawing
code finds the files of set `s` under the name `<s>_<model>_warped` and draws the refinements of ONE camera model, which its
caption names. An `ArgumentError` (naming `where`) for a case that is not named so, or for sets of different models.
"""
function board_warping_model(sets::AbstractDict; where::AbstractString = "[sets]")
    models = String[]
    for (s, case) in sets
        startswith(case, "$(s)_") && endswith(case, "_warped") && length(case) > length("$(s)__warped") ||
            throw(ArgumentError("$where $s = $(repr(case)): the case of a set is named <set>_<camera model>_warped (the drawing code finds its files by that name)"))
        push!(models, String(chopsuffix(chopprefix(case, "$(s)_"), "_warped")))
    end
    length(unique(models)) == 1 ||
        throw(ArgumentError("$where: the cases are of the camera models $(sort(unique(models))); the figure draws the refinements of one camera model"))
    return first(models)
end

"The refinement config of record `[upstream] radiometric_refinement` names (src/radiometric_refinement.jl)."
board_warping_refinements(cfg::AbstractDict) = load_radiometric_refinement_config(projectdir(cfg["upstream"]["radiometric_refinement"]))

"""
    load_board_warping_config(path) -> Dict

The config of record of the board-warping figure, every table and key checked: `[upstream] radiometric_refinement` a refinement
config of record; `[heights] samples` an integer ≥ 2; `[sets]` per capture set its case of the upstream config, the cases distinct
and of one camera model (`board_warping_model`); `[figure]` a drawing code of `BOARD_WARPING_DRAWINGS`, a name for the files, a
layout whose rows name one set per column and every set of `[sets]` once, and what the panels draw: ascending positive band
heights, a positive band count, mesh stride and height exaggeration, a view within 0 … 90° of azimuth and of elevation.
"""
function load_board_warping_config(path::AbstractString)
    c = load_driver_config(path, "board_warping"; tables = BOARD_WARPING_TABLES)
    for t in ("upstream", "heights", "sets", "figure")
        haskey(c, t) && c[t] isa AbstractDict && !isempty(c[t]) || throw(ArgumentError("$path: no [$t] table"))
    end
    check_required_keys(c["upstream"], ("radiometric_refinement",), "$path [upstream]")
    isfile(projectdir(c["upstream"]["radiometric_refinement"])) || throw(ArgumentError("$path [upstream]: no config $(c["upstream"]["radiometric_refinement"])"))
    check_required_keys(c["heights"], BOARD_WARPING_HEIGHTS_KEYS, "$path [heights]")
    n = c["heights"]["samples"]
    n isa Integer && n >= 2 || throw(ArgumentError("$path [heights] samples must be an integer ≥ 2 (the page's two edges at least), got $(repr(n))"))
    rc = board_warping_refinements(c)
    for (s, case) in c["sets"]
        case isa AbstractString && !isempty(case) || throw(ArgumentError("$path [sets] $s must name a case of [upstream] radiometric_refinement, got $(repr(case))"))
        haskey(rc["cases"], case) || throw(ArgumentError("$path [sets] $s: no case $case in $(c["upstream"]["radiometric_refinement"])"))
    end
    allunique(values(c["sets"])) || throw(ArgumentError("$path [sets]: two sets name the same case"))
    board_warping_model(c["sets"]; where = "$path [sets]")
    f = c["figure"]; where = "$path [figure]"
    check_required_keys(f, BOARD_WARPING_FIGURE_KEYS, where)
    f["drawing"] in BOARD_WARPING_DRAWINGS && haskey(PaperFigures.FIGURES, f["drawing"]) ||
        throw(ArgumentError("$where drawing must name the figure's drawing code, one of $(join(BOARD_WARPING_DRAWINGS, ", ")), got $(repr(f["drawing"]))"))
    f["asset_dir"] isa AbstractString && !isempty(f["asset_dir"]) || throw(ArgumentError("$where asset_dir must be a non-empty string, got $(repr(f["asset_dir"]))"))
    f["columns"] isa AbstractVector && !isempty(f["columns"]) && all(x -> x isa AbstractString && !isempty(x), f["columns"]) ||
        throw(ArgumentError("$where columns must be a non-empty array of labels"))
    f["rows"] isa AbstractVector && !isempty(f["rows"]) || throw(ArgumentError("$where rows must be a non-empty array of {label, sets}"))
    for r in f["rows"]
        check_required_keys(r, BOARD_WARPING_ROW_KEYS, "$where rows")
        r["label"] isa AbstractString && !isempty(r["label"]) || throw(ArgumentError("$where rows: a row's label is a non-empty string"))
        r["sets"] isa AbstractVector && length(r["sets"]) == length(f["columns"]) && all(s -> s isa AbstractString, r["sets"]) ||
            throw(ArgumentError("$where rows: the row $(repr(r["label"])) names one set per column ($(length(f["columns"])))"))
    end
    drawn = String[s for r in f["rows"] for s in r["sets"]]
    sort(drawn) == sort(collect(String, keys(c["sets"]))) ||
        throw(ArgumentError("$where rows must name every set of [sets] once: the rows name $(sort(drawn)), [sets] has $(sort(collect(String, keys(c["sets"]))))"))
    b = f["band_heights_mm"]
    b isa AbstractVector && !isempty(b) && all(x -> x isa Real && x > 0, b) && issorted(b) && allunique(b) ||
        throw(ArgumentError("$where band_heights_mm must be a non-empty array of positive heights in mm, ascending, got $(repr(b))"))
    for k in ("max_bands", "mesh_stride")
        f[k] isa Integer && f[k] >= 1 || throw(ArgumentError("$where $k must be a positive integer, got $(repr(f[k]))"))
    end
    v = f["view_azimuth_elevation_deg"]                                       # the drawing code computes a panel's height for a view from this quadrant
    v isa AbstractVector && length(v) == 2 && all(x -> x isa Real && 0 <= x <= 90, v) ||
        throw(ArgumentError("$where view_azimuth_elevation_deg must be [azimuth, elevation] in degrees, each within 0 … 90, got $(repr(v))"))
    f["height_exaggeration"] isa Real && f["height_exaggeration"] > 0 ||
        throw(ArgumentError("$where height_exaggeration must be a positive number, got $(repr(f["height_exaggeration"]))"))
    return c
end

"""
    board_warping_cases(cfg) -> Vector{NamedTuple}

The cases the experiment reads, `(; case, set, model)`, in the order of cases.tsv: the figure's sets row by row, each with its case
(`[sets]`); `model` is the camera model of them all (`board_warping_model`).
"""
function board_warping_cases(cfg::AbstractDict)
    model = board_warping_model(cfg["sets"])
    return [(; case = String(cfg["sets"][s]), set = String(s), model) for r in cfg["figure"]["rows"] for s in r["sets"]]
end

"The layout of the figure as the drawing code takes it: per row `(label, sets)`, and the columns' labels."
board_warping_layout(figure::AbstractDict) =
    (; rows = [(String(r["label"]), Tuple(String.(r["sets"]))) for r in figure["rows"]], columns = Tuple(String.(figure["columns"])))

"""
    board_warping_figures_product(cfg) -> Product

The figures product (its output numbers.tex, written last): keyed by `[heights]`, `[sets]`, `[figure]` and the key of every case's
refine product. Sampling a surface and writing text sum over no thread chunks, so the thread count does not key it.
"""
function board_warping_figures_product(cfg::AbstractDict)
    rc = board_warping_refinements(cfg)
    up = Dict{String, Any}(c.case => refine_product(rc, c.case).key for c in board_warping_cases(cfg))
    return driver_product("figures", Dict{String, Any}("experiment" => cfg["experiment"], "heights" => cfg["heights"], "sets" => cfg["sets"],
                                                       "figure" => cfg["figure"], "upstream" => up); output = "numbers.tex", threads = nothing)
end

"""
    target_page(target) -> (W, H)

The printed page of a calibration target, its width and height in metres: the rectangle of the image printed on the target's canvas,
the canvas's own when it carries no image — the page a refinement builds its board on (VisualRadiometryToolkitCore's
`TexturedTarget`), over which the warping is defined.
"""
function target_page(target)
    cv = VgtkCore.canvas(target)
    cv === nothing && throw(ArgumentError("the target carries no canvas (the printed page)"))
    r = VgtkCore.Rect(cv.texture === nothing ? cv : cv.texture)
    return Float64(r.widths[1]), Float64(r.widths[2])
end

"""
    board_warping_heights(record; samples) -> (; W, H, Z, nodes)

The board warping a refinement estimated, from its calibrated target `record` (a refine product's radiometric_target.json): the
page's width `W` and height `H` in metres; the heights `Z` above the board's plane in metres on a `samples` × `samples` lattice
over the page, `Z[i, j]` at the page fraction u = (i − 1) / (samples − 1) along the columns and v = (j − 1) / (samples − 1) from
the TOP along the rows; and the warping's `nodes`, each `(; u, v, height, surface)` — its page fraction, its stored nodal height
and the surface there (metres).

The surface is VisualRadiometryToolkitCore's `board_height`. It reads the page's size and the facing sign of its board; the nodal
heights are stored along the normal that faces the camera, so the sign is +1 and positive heights lie towards the camera.
"""
function board_warping_heights(record::AbstractString; samples::Integer)
    t = RadiometricCamera.load_radiometric_target(record)
    t.target === nothing && throw(ArgumentError("$record holds no geometric target: its page is not known"))
    W, H = target_page(t.target)
    t.aspect === nothing || isapprox(t.aspect, H / W; rtol = 1e-9) ||
        throw(ArgumentError("$record: the warping was learned on a page of aspect (height / width) $(t.aspect), the target's page has $(H / W)"))
    page = (; size = (W, H), facing = 1.0)
    h = collect(Float64, t.warping)
    us = range(0.0, 1.0; length = samples)
    Z = reshape(VrtkCore.board_height(page, h, [SVector(u, v) for v in us for u in us]), samples, samples)   # u runs fastest: Z[i, j] = z(u_i, v_j)
    zn = VrtkCore.board_height(page, h, VrtkCore.HEIGHT_NODES)
    nodes = [(; u = Float64(n[1]), v = Float64(n[2]), height = hi, surface = zi) for (n, hi, zi) in zip(VrtkCore.HEIGHT_NODES, h, zn)]
    return (; W, H, Z, nodes)
end

"""
    write_board_warping_dump(dump, case, x) -> Vector{String}

The height field `x` (`board_warping_heights`) of `case` as the drawing code reads it, into the folder `dump`:
`<case>.f64` — Int64 n, Float64 page width and height [m], then the n × n Float64 heights [m] column-major (the first index along
u), little-endian — and `<case>_nodes.tsv`, one header line and per node its page fraction u, v, its stored height and the surface
there [mm, six decimals].
"""
function write_board_warping_dump(dump::AbstractString, case::AbstractString, x)
    mkpath(dump)
    field = joinpath(dump, "$case.f64"); nodes = joinpath(dump, "$(case)_nodes.tsv")
    open(field, "w") do io
        write(io, htol(Int64(size(x.Z, 1)))); write(io, htol(x.W)); write(io, htol(x.H)); write(io, htol.(x.Z))
    end
    open(nodes, "w") do io
        println(io, "u\tv\tnodal_height_mm\tsurface_mm")
        for q in x.nodes
            Printf.@printf(io, "%.6f\t%.6f\t%.6f\t%.6f\n", q.u, q.v, 1e3 * q.height, 1e3 * q.surface)
        end
    end
    return [field, nodes]
end

"""
    write_board_warping_data(dir, cases; samples) -> dir

The data folder of the drawing code in `dir`, from the refinements of `cases` (each `(; case, set, model, product)`, `product` the
directory of the case's refine product): per case its height field and nodes (`board_warping_heights` of the product's
radiometric_target.json, `write_board_warping_dump` into `dir`/dump) and `dir`/cases.tsv, one header line and the cases in their
order. Prints each surface's range.
"""
function write_board_warping_data(dir::AbstractString, cases; samples::Integer)
    mm(x) = Printf.@sprintf("%+.3f", 1e3 * x)
    for c in cases
        record = joinpath(c.product, "radiometric_target.json")
        isfile(record) || throw(ArgumentError("case $(c.case): $(c.product) has no radiometric_target.json (the refinement is not complete)"))
        x = board_warping_heights(record; samples)
        write_board_warping_dump(joinpath(dir, "dump"), c.case, x)
        println("   ", rpad(c.case, 28), " page $(round(1e3 * x.W; digits = 1)) x $(round(1e3 * x.H; digits = 1)) mm, surface $(mm(minimum(x.Z))) … $(mm(maximum(x.Z))) mm")
    end
    open(joinpath(dir, "cases.tsv"), "w") do io
        println(io, "case\tset\tmodel\trefine_product")
        foreach(c -> println(io, join((c.case, c.set, c.model, c.product), '\t')), cases)
    end
    return dir
end

"""
    board_warping_tables(data, out, figure; writer = nothing) -> paths

The tables of the drawing code of `[figure]` `figure` (its `drawing`) from the data folder `data` into `out`: meta.toml, per case
the page's size and the surface's range and RMS — and, for the sets of one session ("board-warping-lenses", whose tables take the
layout's rows: the boards and their sets), each case's in-plane aspect and skew and per board how far the lenses' surfaces differ.
`writer` is what the file names as its writer (a driver's stage names itself; `nothing`: the drawing code's own file).
"""
function board_warping_tables(data::AbstractString, out::AbstractString, figure::AbstractDict; writer::Union{Nothing, AbstractString} = nothing)
    compared = figure["drawing"] == "board-warping-lenses" ? (; rows = board_warping_layout(figure).rows) : (;)   # the sets its tables compare, row by row
    return PaperFigures.generate(String(figure["drawing"]); data, out, part = "tables", compared..., named_writer(writer)...)
end

"The drawing code's option that names the files' generator, when a caller names one (`writer`); none for `nothing`: the code's default."
named_writer(writer::Union{Nothing, AbstractString}) = writer === nothing ? (;) : (; generator = String(writer))

"""
    board_warping_drawing(data, out, figure; model, writer = nothing) -> paths

The figure of the drawing code of `[figure]` `figure` (its `drawing`: src/paper_figures/board_warping.jl, or
board_warping_lenses.jl with the same panels and its own caption), from the data folder `data` into `out`: `<asset_dir>.tex`, its panels (each the height field as a surface over the page) and colour key in `<asset_dir>/`,
`<asset_dir>-caption.tex` and source.toml — the refinements of the camera model `model` (`board_warping_model`), laid out in the
table's rows and columns (`board_warping_layout`), with its bands, mesh, view and exaggeration. `writer` is what the files' first
lines name as their generator (a driver's stage names itself; `nothing`: the drawing code's own file).
"""
function board_warping_drawing(data::AbstractString, out::AbstractString, figure::AbstractDict; model::AbstractString,
                               writer::Union{Nothing, AbstractString} = nothing)
    return PaperFigures.generate(String(figure["drawing"]); data, out, part = "figure3d", asset_dir = String(figure["asset_dir"]), model, board_warping_layout(figure)...,
                                 band_heights_mm = figure["band_heights_mm"], max_bands = figure["max_bands"], mesh_stride = figure["mesh_stride"],
                                 view_azimuth_elevation_deg = figure["view_azimuth_elevation_deg"], height_exaggeration = figure["height_exaggeration"],
                                 named_writer(writer)...)
end

"""
    board_warping_figure(dir, cases, cfg; writer = nothing) -> dir

The figures stage's work in the product directory `dir`, from the refine products of `cases` (the config's, `board_warping_cases`,
each with its `product`: `(; case, set, model, product)`): the data folder (`write_board_warping_data`: cases.tsv, dump/), the
tables (meta.toml) and the drawing of `[figure]`; `writer` is what the tables and the drawing's files name as their generator (the
stage names itself).
The files the paper reads — `<asset_dir>.tex` and the folder `<asset_dir>/` — are written into `dir`/assets, which the stage
publishes to plotsdir(); the generated caption and source.toml are kept in `dir`, beside the data (the paper's caption is written
by hand, and source.toml is a name every generator uses).
"""
function board_warping_figure(dir::AbstractString, cases, cfg::AbstractDict; writer::Union{Nothing, AbstractString} = nothing)
    f = cfg["figure"]
    write_board_warping_data(dir, cases; samples = Int(cfg["heights"]["samples"]))
    board_warping_tables(dir, dir, f; writer)
    assets = joinpath(dir, "assets")
    board_warping_drawing(dir, assets, f; model = board_warping_model(cfg["sets"]), writer)
    for name in ("$(f["asset_dir"])-caption.tex", "source.toml")
        mv(joinpath(assets, name), joinpath(dir, name); force = true)
    end
    return dir
end
