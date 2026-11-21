# =============================================================================
# src/paper_figures/board_warping_lenses.jl — the board warping across lenses (board-warping-lenses.tex, its panels
# board-warping-lenses/<set>.tex and its colour key board-warping-lenses/key.tex): the height of each printed calibration board
# above its plane as the refinement of each of three lenses estimated it, from the photographs of ONE session in which a board
# stayed in place while it was photographed with every lens. The surfaces of a row are then the same board's shape, and how far
# they differ is the refinements' disagreement. The port of T. Harder's board_warping_lenses.py (the figure folder's tools/), step
# for step; every number is printed as Python printed it. What the paper shows is ported: the drawing with 3-D surface panels (the
# Python's figure3d) and its `tables`. Its drawing with image panels and its figure of the lenses' differences are not here.
#
# The drawing is the board-warping figure's (board_warping.jl: the panels, the key and the frame, which the two Python generators
# write line for line alike), under this figure's name and with this figure's caption. The tables are this figure's own: per case
# the page, the surface and the in-plane aspect and skew its refinement estimated, and per target how far the surfaces of two
# lenses differ.
#
# What the drawing shows is the caller's to state, nothing of it has a value here (`bwl_figure`'s and `bwl_tables`' keywords;
# scripts/board_warping.jl hands over `[figure]` of configs/supplement/board_warping_lenses.toml).
#
# The data folder is the board-warping figure's (cases.tsv, dump/<case>.f64, dump/<case>_nodes.tsv: board_warping.jl states the
# formats). The tables read one more file per case, outside the data folder: radiometric_target.json of the case's refine product
# (the folder cases.tsv names), for the in-plane print error.
# =============================================================================
using ..BoardWarping: bw_cases, bw_load, bw_deplane, bw_rms, bw_g, bw_signed, bw_figure

"""
    bwl_inplane(product) -> Dict

The in-plane print error a refinement estimated, from `product`/radiometric_target.json: per part of the composition (aspect,
skew) its first parameter; empty when the record has none.
"""
function bwl_inplane(product::AbstractString)
    e = get(JSON.parsefile(joinpath(product, "radiometric_target.json")), "inplane_print_error", nothing)
    e === nothing && return Dict{String, Float64}()
    return Dict{String, Float64}(String(c["type"]) => Float64(c["parameters"][1]) for c in get(e, "compose", []))
end

"""
    bwl_tables(data, out; rows, generator) -> [path]

Writes `out`/meta.toml: per case of the data folder its set, camera model and refine product, the page's size, the surface's range
and RMS (as it is, and without its least-squares plane) and the refinement's in-plane aspect and skew; per target — a row of `rows`,
its label and its sets, one per lens — how far the surfaces of two lenses differ: the RMS and the largest absolute value of the
difference of the two height fields after its least-squares plane is removed (a plane belongs to the poses, not to the warping).
board_warping_lenses.py wrote this file into the data folder; the figure does not read it. `generator` is what the file names as
its writer.
"""
function bwl_tables(data::AbstractString, out::AbstractString; rows, generator::AbstractString)
    C = bw_cases(data); dump = joinpath(data, "dump"); case = Dict(q.set => q.case for q in C)
    L = ["# the estimated board warping per case and the lenses' agreement per board; written by $generator", ""]
    Z = Dict{String, Matrix{Float64}}()
    for q in C
        W, H, z = bw_load(dump, q.case); Z[q.set] = z
        ip = bwl_inplane(q.product)
        push!(L, "[cases.$(q.case)]", "set = \"$(q.set)\"", "model = \"$(q.model)\"", "refine_product = \"$(basename(q.product))\"",
              "page_mm = [$(fixed(W, 3)), $(fixed(H, 3))]", "surface_min_mm = $(fixed(minimum(z), 4))", "surface_max_mm = $(fixed(maximum(z), 4))",
              "surface_rms_mm = $(fixed(bw_rms(z), 4))", "surface_rms_without_plane_mm = $(fixed(bw_rms(bw_deplane(z)), 4))",
              (isempty(ip) ? String[] : ["inplane_aspect = $(bw_g(ip["aspect"]))", "inplane_skew = $(bw_g(ip["skew"]))"])..., "")
    end
    for (r, ss) in rows
        all(s -> haskey(case, s), ss) || error("rows: the data folder has no case of a set of the row \"$r\" ($(join(ss, ", ")))")
        size(unique(size(Z[s]) for s in ss), 1) == 1 || error("rows: the height fields of the row \"$r\" have different lattices")
        push!(L, "[boards.$(replace(lowercase(r), " " => "_"))]",
              "# the difference between two lenses' surfaces of the same board, its best-fit plane removed [mm]: RMS over the page, largest absolute value")
        d = [(a, b, bw_rms(D), maximum(abs, D)) for (i, a) in enumerate(ss) for b in ss[i+1:end] for D in (bw_deplane(Z[a] .- Z[b]),)]   # every pair of lenses
        for (a, b, rms, most) in d
            push!(L, "\"$a - $b\" = { rms_mm = $(fixed(rms, 4)), max_abs_mm = $(fixed(most, 4)) }")
        end
        push!(L, "")
        println(rpad(r, 14), " ", join(("$s $(bw_signed(minimum(Z[s]), 2))…$(bw_signed(maximum(Z[s]), 2)) mm" for s in ss), "; "))
        for (a, b, rms, most) in d
            println(" "^14, " $a − $b: RMS $(fixed(rms, 3)) mm, largest $(fixed(most, 3)) mm (plane removed)")
        end
    end
    return [write_lines(joinpath(out, "meta.toml"), L)]
end

"""
    bwl_figure(data, out; asset_dir, model, rows, height_exaggeration, generator, settings...) -> paths

Draws the figure `asset_dir` from the refinements of the camera model `model` (the case of set `s` is `<s>_<model>_warped`) into
`out`: the board-warping figure's drawing (`bw_figure` of board_warping.jl, which takes every keyword and states them: one panel
per set, each the height field as a PGFPlots surface over the page, ONE colour key, the frame of rows and columns, source.toml) and
this figure's caption, `<asset_dir>-caption.tex`: the photographs are of one session, in which each target stayed in place while it
was photographed with the lenses. The caption names the band's height as the drawing states it (source.toml) and each target's
page size.
"""
function bwl_figure(data::AbstractString, out::AbstractString; asset_dir::AbstractString, model::AbstractString, rows, height_exaggeration::Real,
                    generator::AbstractString, settings...)
    written = bw_figure(data, out; asset_dir, model, rows, height_exaggeration, generator, settings...)
    band = TOML.parsefile(joinpath(out, "source.toml"))["band_mm"]
    dump = joinpath(data, "dump")
    page(s) = (p = bw_load(dump, "$(s)_$(model)_warped"); (round(Int, p[1]), round(Int, p[2])))
    # each target's page size (the sizes of its sets, should they differ)
    size_of(ss) = join(("\$$w\\times$h\$\\,mm" for (w, h) in sort(unique(page(s) for s in ss))), " and ")
    write_lines(joinpath(out, "$asset_dir-caption.tex"), [                    # in the place of the drawing's own caption, which names no session
        "The board warping estimated jointly with the radiometric camera ($model camera model) from the photographs of one session, in which " *
        (length(rows) > 1 ? "each of the two calibration targets (rows) stayed in place while it was photographed with the three lenses (columns): " :
                            "the $(rows[1][1]) target stayed in place while it was photographed with the three lenses (columns): ") *
        "the height of the printed page above its plane, a thin-plate spline through \$5\\times5\$ nodes (dots), " *
        "drawn as a surface over the page with the heights $height_exaggeration times as large as lengths on the page (ticks in mm), coloured " *
        "in bands of $(bw_g(band))\\,mm; positive heights lie towards the camera. " *
        join(("The $r page is $(size_of(ss))." for (r, ss) in rows), " ")])
    return written
end

"""
    figure_board_warping_lenses(; data, out, part, generator, settings...) -> paths

The board warping across lenses from the data folder `data` into `out`. `part` is board_warping_lenses.py's subcommand:

  - `figure3d` (the paper's drawing): every panel a 3-D surface over the page; no image files. `settings` are what is drawn,
    every one required (`bwl_figure`'s keywords, which are the board-warping figure's: `asset_dir`, `model`, `rows`, `columns`,
    `band_heights_mm`, `max_bands`, `mesh_stride`, `view_azimuth_elevation_deg`, `height_exaggeration`).
  - `tables`: `out`/meta.toml, per case the page's size, the surface's range and RMS and the in-plane aspect and skew, and per
    target how far the lenses' surfaces differ (no drawing: of the settings it takes `rows` alone, the targets and their sets).

`generator`: what the written files name as their generator (by default this file; the stage `figures` of scripts/board_warping.jl
names itself).
"""
function figure_board_warping_lenses(; data::AbstractString, out::AbstractString, part::AbstractString,
                                     generator::AbstractString = "experiments/src/paper_figures/board_warping_lenses.jl", settings...)
    part in ("figure3d", "tables") || error("part takes \"figure3d\" or \"tables\"")
    part == "tables" && return bwl_tables(data, out; generator, settings...)
    return bwl_figure(data, out; generator, settings...)
end

FIGURES["board-warping-lenses"] = figure_board_warping_lenses
