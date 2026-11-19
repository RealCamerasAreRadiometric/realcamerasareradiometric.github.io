# =============================================================================
# src/figures.jl — module Figures: the paper's figures, rendered in Julia
# (CairoMakie), from the drivers' evaluation records. Ported from the
# experiments' Python figure code:
#
#   flow_rgb / colour wheel   experiments/lib/flowcolor.py (flow_colors, add_wheel)
#   wheel_map_figure          experiments/lib/wheel_maps.py, `--interp points`
#                             (the mode every paper figure used): every
#                             measurement a node, the field linear on the
#                             Delaunay triangulation of the nodes
#
# Every figure function is `figure_<name>(inputs…; render, dir = plotsdir()) -> path`:
# it reads the run directories it is given and writes the asset `render["asset"]`
# into `dir` (plotsdir(); the tests give a scratch directory). `render` is the figure's `[figures.<name>]` table of the driver's
# config of record (asset name, title, panel labels, sizes); a figure holds no
# data constant of its own.
#
# House style (VisualGeometryToolkitEval Render, the house axis-style source):
# no grid lines of any kind (`HOUSE_AXIS_COMMON` "grid=none"), plain-number
# tick labels, the per-panel statistics as the axis subtitle (no free text
# inside the axes); the colour wheel is the panel's key.
# =============================================================================
module Figures

using CairoMakie
using CairoMakie: Makie
import VisualGeometryToolkitIO as VgtkIO
import VisualGeometryToolkitCore as VgtkCore
using RadiometricCamera: read_switch      # a switch of a figure config, "on" | "off" (RC 0.11.4)
using StaticArrays: SVector
import PNGFiles
using DelaunayTriangulation: triangulate, each_solid_triangle, triangle_vertices, get_point
using Statistics: mean, median, quantile, var
using Printf: @sprintf
using DrWatson: plotsdir
using VisualGeometryToolkitEval.Render: HOUSE_AXIS_COMMON
import VisualGeometryToolkitEval.Reproduce as Reproduce

export flow_rgb, flow_colors, wheel_rings, dense_points, grid_counts, load_vectors, wheel_map_figure, thin_vectors, arrow_wheel_figure,
       figure_wheel_map, focus_surface_figure,
       SUMMARY_STATS, SUMMARY_STAT_TITLES, TABLE_COLUMNS, summary_rows, summary_table, write_summary_tables, table_numbers!,
       row_retcodes, row_converged

"Grid lines are drawn only when the house axis style asks for them (it says `grid=none`)."
const HOUSE_GRID = !occursin("grid=none", HOUSE_AXIS_COMMON)

# ---- the colour coding (flowcolor.py) ------------------------------------------------------------------------------------------

# matplotlib.colors.hsv_to_rgb, element for element (the same float operations, so the colours are the Python figures' colours).
function _hsv_to_rgb(h::Float64, s::Float64, v::Float64)
    i = floor(Int, h * 6.0); f = h * 6.0 - i
    p = v * (1.0 - s); q = v * (1.0 - s * f); t = v * (1.0 - s * (1.0 - f))
    k = mod(i, 6)
    k == 0 && return (v, t, p)
    k == 1 && return (q, v, p)
    k == 2 && return (p, v, t)
    k == 3 && return (p, q, v)
    k == 4 && return (t, p, v)
    return (v, p, q)
end

"""
    flow_rgb(dx, dy, vmax) -> (r, g, b)

The direction + length colour coding of a residual vector (optical-flow style, flowcolor.py's `flow_colors`): hue = the direction
of the vector as drawn in the image (x right, y down), saturation = length / `vmax` (clipped at 1: short = white, ≥ `vmax` = fully
saturated), value = 1. Float64 components in [0, 1].
"""
function flow_rgb(dx::Real, dy::Real, vmax::Real)
    h = mod(atan(Float64(dy), Float64(dx)) / (2π), 1.0)
    s = clamp(hypot(Float64(dx), Float64(dy)) / Float64(vmax), 0.0, 1.0)
    return _hsv_to_rgb(h, s, 1.0)
end

"`flow_rgb` of every vector, as `RGBf` colours of the arrays' shape."
flow_colors(dx::AbstractArray, dy::AbstractArray, vmax::Real) = map((a, b) -> RGBf(flow_rgb(a, b, vmax)...), dx, dy)

"The rings of a colour wheel of radius `vmax`: the round lengths between 0.15 vmax and vmax (wheel_maps.py)."
wheel_rings(vmax) = Tuple(q for q in (0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.5, 1, 2, 5) if 0.15 * vmax <= q <= vmax)

# The colour wheel (the key of a panel) into the layout cell `cell` of `fig`: an inset axis at the bottom right, oriented like the
# image (y down), rings at `rings`, the outer radius written below it ("r = <vmax> <unit>").
function _add_wheel!(cell, vmax, rings, unit, fontsize)
    ins = Axis(cell; width = Relative(0.22), height = Relative(0.33), halign = 0.99, valign = 0.03, aspect = DataAspect(),
               yreversed = true, backgroundcolor = :transparent)
    hidedecorations!(ins); hidespines!(ins)
    n = 241
    g = range(-vmax, vmax; length = n)
    img = [hypot(x, y) <= vmax ? RGBAf(flow_rgb(x, y, vmax)..., 1.0) : RGBAf(1, 1, 1, 0) for x in g, y in g]
    image!(ins, (-vmax, vmax), (-vmax, vmax), img; interpolate = true)
    θ = range(0, 2π; length = 181)
    lines!(ins, vmax .* cos.(θ), vmax .* sin.(θ); color = "#111111", linewidth = 0.7)
    for q in rings
        lines!(ins, q .* cos.(θ), q .* sin.(θ); color = "#111111", linewidth = 0.7)
        text!(ins, q * 0.707 + 0.04 * vmax, q * 0.707 + 0.04 * vmax; text = @sprintf("%g", q), fontsize = 0.765 * fontsize,
              color = "#111111", align = (:left, :top))
    end
    lines!(ins, [0, vmax], [0, 0]; color = "#111111", linewidth = 0.6); lines!(ins, [0, 0], [0, vmax]; color = "#111111", linewidth = 0.6)
    text!(ins, 0, -vmax * 1.12; text = @sprintf("r = %.3g%s", vmax, isempty(unit) ? "" : " " * unit), fontsize = 0.765 * fontsize,
          color = "#111111", align = (:center, :bottom))
    limits!(ins, -vmax * 1.35, vmax * 1.35, vmax * 1.35, -vmax * 1.35)
    return ins
end

# ---- the vector fields ---------------------------------------------------------------------------------------------------------

"""
    load_vectors(kind, path) -> (x, y, dx, dy)

The residual vectors of a wheel-map panel (colmap px, vector = model or detection − truth): `kind` is `"bias"` (a directory of the
detector-bias records `V*.json`: every corner's `x_true, y_true, dx, dy`), `"scatter"` (a vectors JSON `{x, y, dx, dy}` of the
evaluation) or `"eval"` (a night-star evaluation `eval_<label>.json`: its `stars`).
"""
function load_vectors(kind::AbstractString, path::AbstractString)
    if kind == "bias"
        X = Float64[]; Y = Float64[]; DX = Float64[]; DY = Float64[]
        for f in sort(filter(n -> occursin(r"^V.*\.json$", n), readdir(path)))
            for c in VgtkIO.read_record(joinpath(path, f)).corners
                (haskey(c, "dx") && c.dx !== nothing) || continue
                push!(X, c.x_true); push!(Y, c.y_true); push!(DX, c.dx); push!(DY, c.dy)
            end
        end
        return X, Y, DX, DY
    elseif kind == "scatter"
        J = VgtkIO.read_record(path; allownan = true)                 # (a vector may be NaN / ±Infinity: the Python tools' and earlier writers')
        return Float64.(J.x), Float64.(J.y), Float64.(J.dx), Float64.(J.dy)
    elseif kind == "eval"
        s = VgtkIO.read_record(path; allownan = true).stars
        s isa AbstractDict && return Float64.(s.x), Float64.(s.y), Float64.(s.dx), Float64.(s.dy)
        return Tuple([Float64(t[k]) for t in s] for k in ("x", "y", "dx", "dy"))
    end
    throw(ArgumentError("unknown panel kind $(repr(kind)) (bias | scatter | eval)"))
end

"""
    dense_points(x, y, dx, dy, W, H, cell) -> (ux, uy, ok)

The field of the scattered vectors on the cell centres `((i + ½)·cell, (j + ½)·cell)` of a `W × H` image: every measurement a node
(measurements at the same position, rounded to 1e-6 px, are one node with their mean vector), the field linear on the Delaunay
triangulation of the nodes (barycentric weights of the three surrounding measurements), `ok = false` outside the nodes' convex hull.
Arrays of size `(ny, nx)` (rows = y), as wheel_maps.py's `dense_points`.
"""
function dense_points(x, y, dx, dy, W, H, cell)
    m = isfinite.(x) .& isfinite.(y) .& isfinite.(dx) .& isfinite.(dy)
    x, y, dx, dy = x[m], y[m], dx[m], dy[m]
    acc = Dict{Tuple{Float64,Float64}, NTuple{3,Float64}}()
    for i in eachindex(x)
        k = (round(x[i]; digits = 6), round(y[i]; digits = 6))
        a = get(acc, k, (0.0, 0.0, 0.0)); acc[k] = (a[1] + dx[i], a[2] + dy[i], a[3] + 1)
    end
    ks = sort(collect(keys(acc)))
    nodes = [(k[1], k[2]) for k in ks]; vx = [acc[k][1] / acc[k][3] for k in ks]; vy = [acc[k][2] / acc[k][3] for k in ks]
    nx, ny = ceil(Int, W / cell), ceil(Int, H / cell)
    ux = zeros(ny, nx); uy = zeros(ny, nx); ok = falses(ny, nx)
    length(nodes) < 3 && return ux, uy, ok
    tri = triangulate(nodes)
    for T in each_solid_triangle(tri)
        i, j, k = triangle_vertices(T)
        (x1, y1), (x2, y2), (x3, y3) = get_point(tri, i), get_point(tri, j), get_point(tri, k)
        den = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
        den == 0 && continue
        c0 = max(1, floor(Int, min(x1, x2, x3) / cell - 0.5)); c1 = min(nx, ceil(Int, max(x1, x2, x3) / cell + 0.5))
        r0 = max(1, floor(Int, min(y1, y2, y3) / cell - 0.5)); r1 = min(ny, ceil(Int, max(y1, y2, y3) / cell + 0.5))
        for c in c0:c1, r in r0:r1
            ok[r, c] && continue
            px = (c - 0.5) * cell; py = (r - 0.5) * cell
            l1 = ((y2 - y3) * (px - x3) + (x3 - x2) * (py - y3)) / den
            l2 = ((y3 - y1) * (px - x3) + (x1 - x3) * (py - y3)) / den
            l3 = 1 - l1 - l2
            (l1 >= -1e-12 && l2 >= -1e-12 && l3 >= -1e-12) || continue
            ux[r, c] = l1 * vx[i] + l2 * vx[j] + l3 * vx[k]
            uy[r, c] = l1 * vy[i] + l2 * vy[j] + l3 * vy[k]
            ok[r, c] = true
        end
    end
    return ux, uy, ok
end

"""
    grid_counts(x, y, W, H, nx, ny) -> Matrix{Int}

The number of points `(x, y)` (colmap px) in each cell of an `nx × ny` grid over a `W × H` image (cell `(i, j)` = rows `y ∈
[(i − 1)·H/ny, i·H/ny)`, columns `x ∈ [(j − 1)·W/nx, j·W/nx)`), size `(ny, nx)` (rows = y). Points outside the image or not finite
are not counted.
"""
function grid_counts(x, y, W, H, nx::Integer, ny::Integer)
    C = zeros(Int, ny, nx)
    for (u, v) in zip(x, y)
        (isfinite(u) && isfinite(v) && 0 <= u < W && 0 <= v < H) || continue
        C[min(ny, floor(Int, v / (H / ny)) + 1), min(nx, floor(Int, u / (W / nx)) + 1)] += 1
    end
    return C
end

# The count grid over a wheel-map panel: the `nx × ny` cells' borders as thin lines, and in each cell's top-left corner the number of
# the points `(x, y)` in it (`grid_counts`) in small black type.
function _count_grid!(ax, x, y, W, H, nx, ny, pt)
    C = grid_counts(x, y, W, H, nx, ny)
    cw, ch = W / nx, H / ny
    seg = Point2f[]
    for j in 1:(nx - 1); push!(seg, Point2f(j * cw, 0), Point2f(j * cw, H)); end
    for i in 1:(ny - 1); push!(seg, Point2f(0, i * ch), Point2f(W, i * ch)); end
    isempty(seg) || linesegments!(ax, seg; color = RGBAf(0, 0, 0, 0.45), linewidth = 0.35 * pt)
    pos = [Point2f((j - 1) * cw + 0.03 * cw, (i - 1) * ch + 0.03 * ch) for i in 1:ny for j in 1:nx]
    txt = [string(C[i, j]) for i in 1:ny for j in 1:nx]
    text!(ax, pos; text = txt, fontsize = 4.5 * pt, color = :black, align = (:left, :top))
    return C
end

# ---- the wheel maps ----------------------------------------------------------------------------------------------------------

"""
    wheel_map_figure(panels; size, cols, title = "", circle = nothing, point_cell = 8, dpi = 180, unit = "px") -> (Figure, stats)

Dense colour-wheel maps of residual / bias vector fields, one panel per `(label, kind, path, vmax)` of `panels` (`load_vectors`), on
a `size = (W, H)` image: the field of `dense_points` drawn pixelwise in the `flow_rgb` coding (white outside the measurements' hull
and outside `circle = (cx, cy, r)`, which is drawn with its centre), the panel's colour wheel as its key, and the statistics (the mean
vector length of the raw vectors and the mean and 95th percentile of the dense field) as its subtitle. `count_grid = (nx, ny)`
overlays an `nx × ny` grid with the number of points in each cell at its top-left corner (`grid_counts`): the points of
`count_points[k]` (an `(x, y)` pair of vectors) for panel `k`, or the panel's own vectors' positions when `count_points` is
`nothing` or its entry is. The figure is `5.2 cols × 3.75 rows` inches at `dpi`, as wheel_maps.py's. Returns the figure and the
per-panel statistics.
"""
function wheel_map_figure(panels; size, cols::Integer, title::AbstractString = "", circle = nothing, point_cell::Integer = 8,
                          dpi::Integer = 180, unit::AbstractString = "px", count_grid = nothing, count_points = nothing)
    W, H = size
    n = length(panels); cols = min(cols, n); rows = cld(n, cols)
    pt = dpi / 72                                                  # one typographic point in px at the figure's dpi
    fig = Figure(; size = (round(Int, 5.2 * cols * dpi), round(Int, 3.75 * rows * dpi)), fontsize = 7 * pt, backgroundcolor = :white)
    stats = NamedTuple[]
    for (k, (label, kind, path, vmax)) in enumerate(panels)
        r, c = fldmod1(k, cols)
        x, y, dx, dy = load_vectors(kind, path)
        ux, uy, ok = dense_points(x, y, dx, dy, W, H, point_cell)
        ny, nx = Base.size(ux)
        if circle !== nothing
            for j in 1:nx, i in 1:ny
                hypot((j - 0.5) * point_cell - circle[1], (i - 0.5) * point_cell - circle[2]) <= circle[3] || (ok[i, j] = false)
            end
        end
        mag = hypot.(dx, dy); m_ok = hypot.(ux, uy)[ok]
        st = (; label, n = length(mag), mean_px = mean(mag), field_mean_px = isempty(m_ok) ? NaN : mean(m_ok),
                field_p95_px = isempty(m_ok) ? NaN : quantile(m_ok, 0.95))
        push!(stats, st)
        img = [ok[i, j] ? RGBf(flow_rgb(ux[i, j], uy[i, j], vmax)...) : RGBf(1, 1, 1) for j in 1:nx, i in 1:ny]   # (x, y) indexing
        ax = Axis(fig[r, c]; title = label, titlesize = 8.5 * pt, titlefont = :regular,
                  subtitle = @sprintf("|v| mean %.3g %s, n %d; field mean %.3g, p95 %.3g %s", st.mean_px, unit, st.n, st.field_mean_px,
                                      st.field_p95_px, unit),
                  subtitlesize = 6.3 * pt, subtitlecolor = RGBf(0.25, 0.25, 0.25), yreversed = true, aspect = DataAspect(),
                  xticks = [0, W ÷ 2, W], yticks = [0, H ÷ 2, H], xticklabelsize = 7 * pt, yticklabelsize = 7 * pt,
                  xgridvisible = HOUSE_GRID, ygridvisible = HOUSE_GRID)
        image!(ax, (0, nx * point_cell), (0, ny * point_cell), img; interpolate = true)
        if circle !== nothing
            θ = range(0, 2π; length = 361)
            lines!(ax, circle[1] .+ circle[3] .* cos.(θ), circle[2] .+ circle[3] .* sin.(θ); color = RGBf(0.3, 0.3, 0.3), linewidth = 0.6 * pt)
            scatter!(ax, [circle[1]], [circle[2]]; marker = '+', markersize = 6 * pt, color = RGBf(0.3, 0.3, 0.3))
        end
        if count_grid !== nothing
            cp = count_points === nothing ? nothing : count_points[k]
            gx, gy = cp === nothing ? (x, y) : cp
            _count_grid!(ax, gx, gy, W, H, Int(count_grid[1]), Int(count_grid[2]), pt)
        end
        limits!(ax, 0, W, H, 0)
        _add_wheel!(fig[r, c], vmax, wheel_rings(vmax), unit, 8.5 * pt)
    end
    isempty(title) || Label(fig[0, 1:cols], title; fontsize = 10 * pt, tellwidth = false)
    return fig, stats
end

# The panel tuple of a `[[…panels]]` entry of a render table resolved against `dir`: (label, kind, path, vmax).
_panel(p, path) = (String(p["label"]), String(p["kind"]), path, Float64(p["vmax"]))

"The styles of a residual-map figure (`figure_wheel_map`: dense colour wheels or black arrows) and the views of an arrows row."
const WHEEL_STYLES = ("wheels", "arrows")
const WHEEL_VIEWS = ("sensor", "overflow", "zoom")

"""
    check_wheel_render(style, views, where)

A residual-map figure's `style` (of `WHEEL_STYLES`) and its rows' `views` (of `WHEEL_VIEWS`) checked: an `ArgumentError` naming `where`.
"""
function check_wheel_render(style, views, where::AbstractString)
    style in WHEEL_STYLES || throw(ArgumentError("$where: style $(repr(style)) is not $(join(WHEEL_STYLES, " | "))"))
    for v in views
        v in WHEEL_VIEWS || throw(ArgumentError("$where: view $(repr(v)) is not $(join(WHEEL_VIEWS, " | "))"))
    end
end

"""
    figure_wheel_map(panels; render, dir = plotsdir()) -> path

A wheel-map figure (`wheel_map_figure`) of resolved `panels` `(label, kind, path, vmax)` written to `dir/render["asset"]`:
`render` gives `title`, `size` ([W, H] px of the photographs), `cols`, `dpi` and optionally `circle` ([cx, cy, r]) and `count_grid`
([nx, ny]: the grid of point counts, wheels only; the points per panel from `count_points`, see `wheel_map_figure`).
"""
function figure_wheel_map(panels; render::AbstractDict, dir::AbstractString = plotsdir(), count_points = nothing)
    if get(render, "style", "wheels") == "arrows"                     # fig:color-wheels: black arrows (arrow_wheel_figure)
        haskey(render, "count_grid") && throw(ArgumentError("count_grid is drawn on the colour wheels only (style = \"wheels\")"))
        cols = Int(render["cols"]); views = [String(get(p, "view", "sensor")) for p in render["panels"]]
        rows = [(panels[i][4], views[i], [(panels[k][1], panels[k][2], panels[k][3]) for k in i:min(i + cols - 1, length(panels))])
                for i in 1:cols:length(panels)]
        fig, _ = arrow_wheel_figure(rows; size = Tuple(Int.(render["size"])), cells_x = Int(render["cells_x"]),
                                    len_cells_x = Int(render["len_cells_x"]), dpi = render["dpi"], title = render["title"])
        return _save(fig, render["asset"], dir)
    end
    fig, _ = wheel_map_figure(panels; size = Tuple(Int.(render["size"])), cols = render["cols"], title = render["title"],
                              circle = haskey(render, "circle") ? Tuple(Float64.(render["circle"])) : nothing, dpi = render["dpi"],
                              count_grid = haskey(render, "count_grid") ? Tuple(Int.(render["count_grid"])) : nothing, count_points)
    return _save(fig, render["asset"], dir)
end

# ---- the arrow maps (fig:color-wheels) ------------------------------------------------------------------------------------

"""
    thin_vectors(x, y, dx, dy, x0, y0, c, nx, ny) -> (x, y, dx, dy)

Of every cell of the `nx × ny` grid of `c`-px cells from `(x0, y0)` the ONE measurement nearest the cell's centre, as itself (no
averaging). The choice depends on the positions alone, so the start and the refinement of a chrome set (the same dots) show the
same measurements.
"""
function thin_vectors(x, y, dx, dy, x0, y0, c, nx, ny)
    best = Dict{Int, Tuple{Float64, Int}}()
    for k in eachindex(x)
        i = floor(Int, (x[k] - x0) / c); j = floor(Int, (y[k] - y0) / c)
        (0 <= i < nx && 0 <= j < ny) || continue
        d = hypot(x[k] - (x0 + (i + 0.5) * c), y[k] - (y0 + (j + 0.5) * c)); key = i * ny + j
        (haskey(best, key) && best[key][1] <= d) || (best[key] = (d, k))
    end
    ks = sort([v[2] for v in values(best)])
    return x[ks], y[ks], dx[ks], dy[ks]
end

"""
    arrow_wheel_figure(rows; size, cells_x, len_cells_x, dpi = 150, title = "", unit = "px") -> (Figure, stats)

The residual vectors as thin black arrows, one row per `(unit_px, view, panels)` of `rows`, `panels` the row's `(label, kind, path)`
(`load_vectors`): of every cell of a grid of `cells_x` square cells across the sensor the one measurement nearest the cell's centre
(`thin_vectors`), its arrow's tail at the measurement. An arrow's LENGTH is linear in the residual, `unit_px` residual = 0.95 of a
cell of `len_cells_x` across the sensor on the page, the same for the row's panels; the panel's length key (a white box, lower
right) draws it. `view`: `"sensor"` (the sensor, arrows clipped at its edge), `"overflow"` (the sensor, its frame the border, the
arrows NOT clipped: they run past it over the page) or `"zoom"` (the measured area enlarged to the panel, widened to every arrow;
its cells shrink with the view, so a cell — and the length unit — has the same size on the page as in the other rows). The
subtitle: how many arrows of how many measurements, and the median, 95th percentile and maximum of every measurement's |v|.
`5.2 · panels × 3.75 · rows` inches at `dpi`.
"""
function arrow_wheel_figure(rows; size, cells_x::Integer, len_cells_x::Integer, dpi::Integer = 150, title::AbstractString = "",
                            unit::AbstractString = "px")
    W, H = size
    ncol = maximum(length(r[3]) for r in rows); nrow = length(rows)
    pt = dpi / 72; inch = dpi
    ink, ink2 = RGBf(0.067, 0.067, 0.067), RGBf(0.25, 0.25, 0.25)
    fig = Figure(; size = (round(Int, (5.2 * ncol + 0.15) * inch), round(Int, 3.75 * nrow * inch)), fontsize = 7 * pt,
                 backgroundcolor = :white, figure_padding = (round(Int, 0.18 * inch), round(Int, 0.05 * inch), round(Int, 0.1 * inch), round(Int, 0.13 * inch)))
    arrow_kw = (; color = ink, shaftwidth = 0.012 * inch, tipwidth = 3.2 * 0.012 * inch, tiplength = 4.0 * 0.012 * inch, minshaftlength = 0,
                align = :tail, strokemask = 0)
    C0 = W / cells_x
    stats = NamedTuple[]; overflow = []
    for (r, (upx, view, panels)) in enumerate(rows)
        raw = [load_vectors(kind, path) for (_, kind, path) in panels]
        raw = [(m = isfinite.(x) .& isfinite.(y) .& isfinite.(dx) .& isfinite.(dy); (x[m], y[m], dx[m], dy[m])) for (x, y, dx, dy) in raw]
        if view == "zoom"                                              # the measured area, padded, widened to the sensor's aspect
            xs = reduce(vcat, first.(raw)); ys = reduce(vcat, getindex.(raw, 2))
            bx0, bx1, by0, by1 = minimum(xs), maximum(xs), minimum(ys), maximum(ys); p = 0.04 * max(bx1 - bx0, by1 - by0)
            bx0, bx1, by0, by1 = bx0 - p, bx1 + p, by0 - p, by1 + p
            if (bx1 - bx0) / (by1 - by0) < W / H
                d = (by1 - by0) * W / H - (bx1 - bx0); bx0, bx1 = bx0 - d / 2, bx1 + d / 2
            else
                d = (bx1 - bx0) * H / W - (by1 - by0); by0, by1 = by0 - d / 2, by1 + d / 2
            end
            zf = (bx1 - bx0) / W; c = C0 * zf; nx, ny = ceil(Int, (bx1 - bx0) / c), ceil(Int, (by1 - by0) / c); x0, y0 = bx0, by0
        elseif view in ("sensor", "overflow")
            zf = 1.0; c = C0; nx, ny = cells_x, ceil(Int, H / C0); x0, y0 = 0.0, 0.0
        else
            throw(ArgumentError("arrow_wheel_figure: view $(repr(view)) is not sensor | overflow | zoom"))
        end
        data = [thin_vectors(v..., x0, y0, c, nx, ny) for v in raw]
        s = upx / (0.95 * (W / len_cells_x) * zf)                     # residual px per image px of arrow
        ex = reduce(vcat, [vcat(X, X .+ U ./ s) for (X, _, U, _) in data]); ey = reduce(vcat, [vcat(Y, Y .+ V ./ s) for (_, Y, _, V) in data])
        lim = view == "zoom" ? (min(bx0, minimum(ex) - 20), max(bx1, maximum(ex) + 20), min(by0, minimum(ey) - 20), max(by1, maximum(ey) + 20)) :
                               (0.0, Float64(W), 0.0, Float64(H))
        for (k, ((X, Y, U, V), (x, y, dx, dy), (label, _, _))) in enumerate(zip(data, raw, panels))
            La = hypot.(dx, dy)
            sub = @sprintf("%d of %d shown (one per %.0f-px cell) · all: |v| median %.3g, p95 %.3g, max %.3g %s%s", length(X), length(dx), c,
                           median(La), quantile(La, 0.95), maximum(La), unit, view == "zoom" ? @sprintf(" · zoomed ×%.2g", 1 / zf) : "")
            push!(stats, (; label, shown = length(X), residual_summary(La)...))
            tk = view == "zoom" ? _round_ticks(lim[1], lim[2]) : [0, W ÷ 2, W]
            tky = view == "zoom" ? _round_ticks(lim[3], lim[4]) : [0, H ÷ 2, H]
            ax = Axis(fig[r, k]; title = label, titlesize = 8.5 * pt, titlefont = :regular, titlegap = 12 * pt / 2,
                      subtitle = sub, subtitlesize = 6.3 * pt, subtitlecolor = ink2, yreversed = true, aspect = DataAspect(),
                      xticks = tk, yticks = tky, xticklabelsize = 6.5 * pt, yticklabelsize = 6.5 * pt, xgridvisible = false, ygridvisible = false,
                      spinewidth = 0.6 * pt, xtickwidth = 0.6 * pt, ytickwidth = 0.6 * pt, xticksize = 2.5 * pt, yticksize = 2.5 * pt,
                      backgroundcolor = view == "overflow" ? :transparent : :white)
            limits!(ax, lim[1], lim[2], lim[4], lim[3])
            o = sortperm(hypot.(U, V))                                 # the longest drawn last (on top)
            # the length key in the panel's lower right corner, on a white box above the arrows (data coordinates)
            bw, bh = 0.125 * (lim[2] - lim[1]), 0.105 * (lim[4] - lim[3])
            kx0 = lim[1] + 0.855 * (lim[2] - lim[1]); ky1 = lim[4] - 0.025 * (lim[4] - lim[3])
            key = (; x0 = kx0, y1 = ky1, bw, bh, len = upx / s, label = @sprintf("%g %s", upx, unit))
            if view == "overflow"                                     # drawn over the whole page once the layout is known
                push!(overflow, (ax, X[o], Y[o], U[o] ./ s, V[o] ./ s, key))
            else
                arrows2d!(ax, X[o], Y[o], U[o] ./ s, V[o] ./ s; arrow_kw...)
                _arrow_key!(ax, key, (x, y) -> (x, y), 1.0, 1.0, arrow_kw, pt, ink)
            end
        end
    end
    isempty(title) || Label(fig[0, 1:ncol], title; fontsize = 10 * pt, tellwidth = false)
    colgap!(fig.layout, 0.32 * inch); rowgap!(fig.layout, 0.42 * inch)
    if !isempty(overflow)                                              # the overflowing rows' arrows over the whole page, after the layout
        Makie.update_state_before_display!(fig)
        top = Scene(fig.scene; viewport = fig.scene.viewport, clear = false, camera = campixel!)
        for (ax, X, Y, U, V, key) in overflow
            vp = ax.scene.viewport[]; fl = ax.finallimits[]
            sx = vp.widths[1] / fl.widths[1]; sy = vp.widths[2] / fl.widths[2]
            topx(x, y) = (vp.origin[1] + (x - fl.origin[1]) * sx, vp.origin[2] + vp.widths[2] - (y - fl.origin[2]) * sy)   # y reversed
            P = topx.(X, Y)
            arrows2d!(top, first.(P), last.(P), U .* sx, -V .* sy; arrow_kw...)
            _arrow_key!(top, key, topx, sx, -sy, arrow_kw, pt, ink)
        end
    end
    return fig, stats
end

# The length key (a white box with the reference arrow and its value) of `key` (data coordinates of its box: lower-left x0, the
# bottom y1 of the image-down axis, size bw × bh, arrow length len) drawn into `scene` through `tr(x, y)`, the scales `sx`, `sy`
# of a data unit (sy < 0 for a y-up pixel scene).
function _arrow_key!(scene, key, tr, sx, sy, arrow_kw, pt, ink)
    a = tr(key.x0, key.y1); b = tr(key.x0 + key.bw, key.y1 - key.bh)
    poly!(scene, Rect2f(min(a[1], b[1]), min(a[2], b[2]), abs(b[1] - a[1]), abs(b[2] - a[2])); color = :white,
          strokecolor = RGBf(0.74, 0.74, 0.74), strokewidth = 0.5 * pt, overdraw = true)
    kc = key.x0 + key.bw / 2; t = tr(kc - key.len / 2, key.y1 - 0.62 * key.bh); m = tr(kc, key.y1 - 0.42 * key.bh)
    arrows2d!(scene, [t[1]], [t[2]], [key.len * sx], [0.0]; arrow_kw..., overdraw = true)
    text!(scene, m[1], m[2]; text = key.label, fontsize = 7 * pt, color = ink, align = (:center, :top), overdraw = true)   # screen-space alignment: below the anchor in both scenes
    return scene
end

# Three round ticks inside [lo, hi].
function _round_ticks(lo, hi)
    spacings = [100, 200, 250, 500, 1000]; st = spacings[argmin(abs.(spacings .- (hi - lo) / 3))]
    return collect(ceil(lo / st) * st:st:hi)
end

function _save(fig, asset, dir)
    out = joinpath(dir, asset)
    mkpath(dirname(out))
    save(out, fig; px_per_unit = 1)
    return out
end

# ---- the focus surfaces (fig:focus-surfaces) ------------------------------------------------------------------------------

"""
    focus_surface_figure(panels; size_in = (3.3, 1.75), dpi = 300, title = "", grid_every = 14, elevations_deg = fill(36, length(panels))) -> Figure

The in-focus surfaces of `panels` `(label, X, Y, Z, D)` — the in-focus points of a refined camera in its own coordinates (metres;
x right, y down, z forward; NaN where the lens images nothing) on a lattice of the sensor and their focus distance `D` — as 3D
surfaces side by side, equal data aspect: a fisheye's sphere about the camera, a narrow lens's plane. Each surface is one light
tone, slightly transparent, with a dark line along every `grid_every`-th row and column of its lattice, which shows how it bends;
the focus distance is not coloured (owner 2026-11-15). The camera (the origin) is a black dot, drawn first so that a surface in
front of it covers it; panel `k` is seen from `elevations_deg[k]` degrees above the x–y plane (a fisheye's dome from low down,
its camera under it). `size_in` inches at `dpi`.
"""
function focus_surface_figure(panels; size_in = (3.3, 1.75), dpi::Integer = 300, title::AbstractString = "", grid_every::Integer = 14,
                              elevations_deg = fill(36.0, length(panels)))
    isempty(panels) && throw(ArgumentError("focus_surface_figure: no panels"))
    grid_every >= 2 || throw(ArgumentError("focus_surface_figure: grid_every = $grid_every (a line every ≥ 2 lattice samples)"))
    length(elevations_deg) == length(panels) || throw(ArgumentError("focus_surface_figure: $(length(elevations_deg)) elevations for $(length(panels)) panels"))
    pt = dpi / 72
    fig = Figure(; size = (round(Int, size_in[1] * dpi), round(Int, size_in[2] * dpi)), fontsize = 6 * pt, backgroundcolor = :white,
                 figure_padding = round(Int, 3 * pt))
    line, face = RGBAf(0.12, 0.12, 0.12, 0.95), RGBAf(0.90, 0.92, 0.95, 0.78)
    for (c, (label, X, Y, Z, D)) in enumerate(panels)
        any(isfinite, D) || throw(ArgumentError("focus_surface_figure: panel $(repr(label)) has no finite focus point"))
        ax = Axis3(fig[1, c]; title = String(label), titlesize = 6 * pt, titlefont = :regular, aspect = :data,
                   xlabel = "x [m]", ylabel = "y [m]", zlabel = "z [m]", xlabelsize = 5 * pt, ylabelsize = 5 * pt, zlabelsize = 5 * pt,
                   xticklabelsize = 4.5 * pt, yticklabelsize = 4.5 * pt, zticklabelsize = 4.5 * pt, xticks = LinearTicks(3), yticks = LinearTicks(3),
                   zticks = LinearTicks(3), azimuth = -0.35π, elevation = deg2rad(Float64(elevations_deg[c])), protrusions = round(Int, 12 * pt),
                   xlabeloffset = 11 * pt, ylabeloffset = 14 * pt, zlabeloffset = 14 * pt)   # the axis names clear of their tick labels
        scatter!(ax, [0.0], [0.0], [0.0]; color = :black, markersize = 4 * pt)
        # the grid: the lattice's every grid_every-th row and column in the line colour, the rest the face
        on(i) = mod(i - 1, grid_every) == grid_every ÷ 2
        surface!(ax, X, Y, Z; color = [on(i) || on(j) ? line : face for i in axes(X, 1), j in axes(X, 2)], shading = NoShading)
    end
    isempty(title) || Label(fig[0, :], title; fontsize = 7 * pt, tellwidth = false)
    colgap!(fig.layout, round(Int, 4 * pt))
    return fig
end

# ---- the chrome tables (the chrome evaluation paper config's [tables.<name>], src/chrome_evaluation_paper.jl: every model on every chrome set) ------------------------------

"The statistics of a chrome evaluation's summary.tsv a table prints, in this order, and their column titles."
const SUMMARY_STATS = ("median_px", "mean_px", "p95_px", "rms_px")
const SUMMARY_STAT_TITLES = Dict("median_px" => "median", "mean_px" => "mean", "p95_px" => "p95", "rms_px" => "RMS")
"The columns of a table's long-form CSV (`write_summary_tables`): one row per table row and statistic."
const TABLE_COLUMNS = ("set", "model", "variant", "stat", "start", "refined", "change_pct", "photos_start", "photos_refined", "dots_start", "dots_refined",
                       "retcode_start", "retcode_refined", "retcode_refinement", "converged", "coverage_dots_start", "coverage_dots_refined",
                       "coverage_fraction_start", "coverage_fraction_refined", "refined_all", "photos_refined_all")
"""The long form's columns after `TABLE_COLUMNS` of a table with an in-sample variant (its rows' `in_sample`, `summary_table`): the
variant, the refinement's in-sample score, the gap (the headline's refined score − the in-sample one, px) and the in-sample fit's retcode."""
const IN_SAMPLE_COLUMNS = ("variant_in_sample", "in_sample", "gap", "retcode_in_sample")
"""The long form's columns of a table with intervals (its rows' `ci`, `summary_table`): the refined − start difference of the paired
photograph bootstrap, its interval and whether it excludes 0 (`NaN`, `NaN`, false for a row without a start)."""
const CI_COLUMNS = ("difference", "ci_lo", "ci_hi", "ci_excludes_zero")
"A table row's paired-bootstrap intervals by statistic (`nothing` without them)."
_ci(r) = get(r, :ci, nothing)
"Whether the table of `rows` prints an interval beside each change (a row of it carries one)."
has_ci(rows) = any(r -> _ci(r) !== nothing, rows)
"A table row's in-sample summary row (`nothing` for a table without an in-sample variant, or a row built without one)."
_in_sample(r) = get(r, :in_sample, nothing)
"Whether the table of `rows` prints the in-sample score and the gap beside its headline (its rows carry an in-sample row)."
has_in_sample(rows) = !isempty(rows) && all(r -> _in_sample(r) !== nothing, rows)

"""
    row_retcodes(r) -> (start, refined, refinement)

The termination records of a table row `r` (`summary_table`'s input), each the name of a VisualGeometryToolkitCore `ReturnCode`: its
start's and its refinement's chrome evaluations (the rotation-centre refine, the summary's `retcode` column; `""` for a row without a
start) and the radiometric refinement's own LM (`r.refinement_retcode`, its run's parameters.json `retcode`, or `"unfinished"` /
`"unrecorded"`: no record).
"""
row_retcodes(r) = (r.start === nothing ? "" : String(r.start["retcode"]), String(r.refined["retcode"]), String(r.refinement_retcode))
"The retcode of a table row's in-sample evaluation (`\"\"` without one: `_in_sample`)."
in_sample_retcode(r) = _in_sample(r) === nothing ? "" : String(_in_sample(r)["retcode"])

# recorded_retcode: the ReturnCode a run record names (src/return_codes.jl)
include("return_codes.jl")
# residual_summary: the one summary of a residual vector (src/residual_summary.jl, shared with the prelude)
include("residual_summary.jl")
# a chrome evaluation's summary table read back (src/summary_table.jl, shared with the prelude)
include("summary_table.jl")

"""
    row_converged(r) -> Bool

Whether every fit of table row `r` converged ([`row_retcodes`](@ref)): Core's `successful_retcode` of each recorded `ReturnCode`
(Core 0.11.0: an LM's tolerance stop, `ftol` or `xtol`, is `Success` and never marked; the iteration and time caps, a rejection-cap
stall and a failure are)
([`recorded_retcode`](@ref); a fit without a record has not), a row without a start reading only its refined evaluation and its
refinement, and its in-sample evaluation where the table has one ([`in_sample_retcode`](@ref)). A record's metadata (the rows'
`converged` and retcodes in table_<name>.tsv/json), never printed in a table: the paper reports no convergence (owner 2026-11-16).
"""
row_converged(r) = all(c -> c == "" || (rc = recorded_retcode(c); rc !== nothing && VgtkCore.successful_retcode(rc)), (row_retcodes(r)..., in_sample_retcode(r)))


"""
    summary_rows(path) -> Dict{Tuple{String,String}, Dict{String,Any}}

The rows of a chrome evaluation's `summary.tsv` (scripts/chrome_evaluation.jl, its `EVALUATE_SUMMARY_COLUMNS`; VisualGeometryToolkitEval's
table, `read_summary`) by `(model, variant)`: column name => value as read (numbers; an empty cell `""`).
"""
summary_rows(path::AbstractString) = Dict((String(r["model"]), String(r["variant"])) => r for r in read_summary(path))

"""
    summary_table(rows; stats = SUMMARY_STATS) -> Vector{NamedTuple}

The long form of a chrome table (`TABLE_COLUMNS`): one record per row of `rows` and statistic of `stats` — `set`, `model`, `variant`,
`stat`, `start` (`NaN` for a row without a start), `refined`, `change_pct` (`100 (refined − start) / start`, `NaN` without a start),
`photos_start`, `photos_refined`, `dots_start`, `dots_refined` (0 without a start), the row's retcodes (`retcode_start`,
`retcode_refined`, `retcode_refinement`: [`row_retcodes`](@ref)), `converged` ([`row_converged`](@ref)), the coverages
(`coverage_dots_start`, `coverage_dots_refined`, `coverage_fraction_start`, `coverage_fraction_refined`: the dots each model scores and
their fraction of the registered dots; 0 and `NaN` without a start) and the refinement on all of its own dots (`refined_all`, the
summary's `all_<stat>`, over `photos_refined_all` photographs). `start`, `refined` and `change_pct` are the head-to-head, on the dots
every model of the set's evaluation scores (owner 2026-11-15). A row is
`(; set, model, variant, start, refined, refinement_retcode)`: `start` a summary row (`summary_rows`) or `nothing`, `refined` a summary
row, `refinement_retcode` the name of the radiometric refinement's LM `ReturnCode`; with `in_sample`, the refined model's summary row
in the table's in-sample variant (the paper's tables: `variant` "motion_heldout", the headline, and in-sample "motion"; ruling
2026-11-15), each record also has `IN_SAMPLE_COLUMNS`: `variant_in_sample`, `in_sample` (its `stat`), `gap` = `refined − in_sample`
(px: what holding the dots out costs the refinement) and `retcode_in_sample`.
"""
function summary_table(rows; stats = SUMMARY_STATS)
    out = NamedTuple[]
    for r in rows, st in stats
        s = r.start === nothing ? NaN : Float64(r.start[st]); e = Float64(r.refined[st])
        push!(out, (; set = String(r.set), model = String(r.model), variant = String(r.variant), stat = String(st), start = s, refined = e,
                      change_pct = isnan(s) ? NaN : 100 * (e - s) / s,
                      photos_start = r.start === nothing ? 0 : Int(r.start["photos"]), photos_refined = Int(r.refined["photos"]),
                      dots_start = r.start === nothing ? 0 : Int(r.start["dots"]), dots_refined = Int(r.refined["dots"]),
                      retcode_start = row_retcodes(r)[1], retcode_refined = row_retcodes(r)[2], retcode_refinement = row_retcodes(r)[3],
                      converged = row_converged(r),
                      coverage_dots_start = r.start === nothing ? 0 : Int(r.start["coverage_dots"]), coverage_dots_refined = Int(r.refined["coverage_dots"]),
                      coverage_fraction_start = r.start === nothing ? NaN : Float64(r.start["coverage_fraction"]),
                      coverage_fraction_refined = Float64(r.refined["coverage_fraction"]),
                      refined_all = Float64(r.refined["all_" * st]), photos_refined_all = Int(r.refined["all_photos"]),
                      (_in_sample(r) === nothing ? (;) :
                       (; variant_in_sample = String(_in_sample(r)["variant"]), in_sample = Float64(_in_sample(r)[st]),
                          gap = e - Float64(_in_sample(r)[st]), retcode_in_sample = in_sample_retcode(r)))...,
                      (_ci(r) === nothing ? (;) : (c = _ci(r)[st]; (; difference = c.difference, ci_lo = c.lo, ci_hi = c.hi, ci_excludes_zero = c.excludes_zero)))...))
    end
    return out
end

"""
    table_numbers!(nums, prefix, rows; stats = SUMMARY_STATS, sigdigits, change_sigdigits, label = "") -> nums

The number macros of a chrome table, through VisualGeometryToolkitEval's number writer (`add_number!` on the driver's
`PaperNumbers` `nums`), so the prose cites the values the table prints. Each row of `rows` (`summary_table`'s input, a start and a
refinement) carries `name_part` (its config row's `macro`), its name part; per row and statistic (`median_px` → `Median`, `mean_px` → `Mean`, `p95_px` →
`PNineFive`, `rms_px` → `Rms`, by Eval's `number_macro_name`):

- `\\<prefix><Row><Stat>Start` and `…Refined`: the residuals [px], `sigdigits` significant digits (`better = :lower`);
- `…Change`: the change in percent, `100 (refined − start) / start`, `change_sigdigits` (`better = :lower`: negative improves);
- of a table with an in-sample variant, `…InSample`: the refinement's in-sample residual [px] and `…Gap`: the headline's refined
  residual minus it [px], `sigdigits` (`better = :lower`);

and, per consecutive pair of rows (a lens's aperture row and its pinhole row, in either order: the aperture row is the one whose name
part ends in `aperture`), per statistic `\\<prefix><Lens><Stat>ImprovementGain`: the aperture row's improvement (−change) minus the
pinhole row's, in percentage points (`better = :higher`), `<Lens>` the aperture row's name part without its trailing `aperture` (none
for `aperture` itself). `label` (the
table's `tab:…`) goes into each macro's provenance comment; `provenance(row)` gives a row's own fields of it (the chrome
evaluation run it reads), `(;)` by default.
"""
function table_numbers!(nums, prefix::AbstractString, rows; stats = SUMMARY_STATS, sigdigits::Integer, change_sigdigits::Integer,
                        label::AbstractString = "", provenance = r -> (;))
    stat_part(st) = replace(String(st), r"_px$" => "")
    changes = Dict{Tuple{Int, String}, Float64}()
    for (i, r) in enumerate(rows), t in summary_table([r]; stats)
        name(what) = Reproduce.number_macro_name(prefix, String(r.name_part), stat_part(t.stat), what)
        isnan(t.start) && throw(ArgumentError("table_numbers!: row $(r.name_part) has no start: a change needs one"))
        p = provenance(r)
        Reproduce.add_number!(nums, name("start"), t.start; sigdigits, label, better = :lower, provenance = p)
        Reproduce.add_number!(nums, name("refined"), t.refined; sigdigits, label, better = :lower, provenance = p)
        Reproduce.add_number!(nums, name("change"), t.change_pct; sigdigits = change_sigdigits, label, better = :lower, provenance = p)
        if haskey(t, :in_sample)
            Reproduce.add_number!(nums, name("in_sample"), t.in_sample; sigdigits, label, better = :lower, provenance = p)
            Reproduce.add_number!(nums, name("gap"), t.gap; sigdigits, label, better = :lower, provenance = p)
        end
        changes[(i, t.stat)] = t.change_pct
    end
    iseven(length(rows)) || throw(ArgumentError("table_numbers!: rows come in pairs (a lens's aperture row and its pinhole row), got $(length(rows))"))
    is_aperture(r) = occursin(r"[Aa]perture$", String(r.name_part))
    for i in 1:2:length(rows)
        is_aperture(rows[i]) != is_aperture(rows[i + 1]) ||
            throw(ArgumentError("table_numbers!: rows $(rows[i].name_part) and $(rows[i + 1].name_part) are not a pair: one name part of the two ends in `aperture`"))
        a, p = is_aperture(rows[i]) ? (i, i + 1) : (i + 1, i)             # the aperture row and the pinhole row of the pair
        lens = replace(String(rows[a].name_part), r"[Aa]perture$" => "")
        for st in stats
            gain = -changes[(a, st)] - (-changes[(p, st)])
            Reproduce.add_number!(nums, Reproduce.number_macro_name(prefix, lens, stat_part(st), "improvement", "gain"), gain;
                                  sigdigits = change_sigdigits, label, better = :higher, provenance = provenance(rows[a]))
        end
    end
    return nums
end

"A set or model label as LaTeX text: it is authored LaTeX (math such as `\$f/32\$` stays), with the characters a plain label holds that LaTeX
reads otherwise escaped (`& % _ #`, a backslash, the middle dot)."
_tex_text(s) = replace(string(s), "\\" => "\\textbackslash{}", "&" => "\\&", "%" => "\\%", "_" => "\\_", "#" => "\\#", "·" => "\\textperiodcentered{}")
_count_cell(r, key) = r.start === nothing ? string(Int(r.refined[key])) : string(Int(r.start[key]), " / ", Int(r.refined[key]))
"A row's coverage cell: the fraction of the registered dots each model scores, in percent (one decimal), `start / refined`."
function _coverage_cell(r)
    pc(x) = string(round(100 * Float64(x["coverage_fraction"]); digits = 1))
    return r.start === nothing ? pc(r.refined) : string(pc(r.start), " / ", pc(r.refined))
end

"""
    write_summary_tables(dir, name, rows; stats = SUMMARY_STATS, number = string, change = string, layout, label, caption, worse_color, mark_best, inline_change) -> (csv, md, tex)

The table `name` of `rows` (`summary_table`'s rows) in `dir`: `table_<name>.tsv` (the long form, `TABLE_COLUMNS`, the numbers raw: VisualGeometryToolkitEval's table),
`table_<name>.md` (one markdown table per set — a model per row; per statistic the start, the refinement and the change in percent
on the dots both models score (the head-to-head, `summary_table`), and the refinement on all of its own dots; the photographs and dots
as `start / refined`, and each model's coverage, the percent of the registered dots it scores, `start / refined`) and `table_<name>.tex`, a booktabs table (the paper loads booktabs): `layout = "per_set"`
one `table` float per set, its caption the set, with the photographs, dots and coverage; `layout = "blocks"` ONE `table*` across both
columns, a header row per set and a model per row, the coverage and no photographs and dots, its `caption` and `label` (the paper's tables). `worse_color` (e.g.
"red", xcolor): in the tex the refined value and the change of a statistic that did not improve (change ≥ 0) in that colour and bold.
`mark_best` (owner 2026-11-15): in the tex the best (lowest) head-to-head value of each statistic among the starts and the refinements
of the rows of one camera (a row's `camera`; a row without one: its set) in bold — every value that prints as the best does, a tie at
the printed digits included — and a non-improvement then in `worse_color` alone, so bold means the best only. `inline_change` (layout
"blocks" only, owner 2026-11-15): per statistic `start → refined (change %)` — an arrow between the two values and the signed change in
brackets directly behind the refinement (one cell), no column of its own — and the tabular scaled down to the text width where it is
wider (graphicx's `\\resizebox`). `number(x)`
prints a pixel value and `change(pct)` a change in percent (VisualGeometryToolkitEval's `tex_number`); a row without a start prints blank start and
change cells. A row's convergence ([`row_converged`](@ref)) is in the records only, never in the md or the tex (owner 2026-11-16), and
the table states the count under its rows. The sets keep the order of their first row.
"""
function write_summary_tables(dir::AbstractString, name::AbstractString, rows; stats = SUMMARY_STATS, number = string, change = string,
                              layout::AbstractString = "per_set", label::AbstractString = "", caption::AbstractString = "",
                              worse_color::AbstractString = "", mark_best::Bool = false, inline_change::Bool = false)
    layout in ("per_set", "blocks") || throw(ArgumentError("write_summary_tables: layout $(repr(layout)) is not per_set | blocks"))
    inline_change && layout != "blocks" && throw(ArgumentError("write_summary_tables: inline_change is the layout \"blocks\"'s, got layout $(repr(layout))"))
    mkpath(dir)
    long = summary_table(rows; stats)
    insample = has_in_sample(rows)                                          # the in-sample score and the gap beside the headline
    withci = has_ci(rows)                                                   # the paired bootstrap's interval beside each change
    csv = joinpath(dir, "table_$name.tsv")                                 # VisualGeometryToolkitEval's table
    write_tsv(csv, (TABLE_COLUMNS..., (insample ? IN_SAMPLE_COLUMNS : ())..., (withci ? CI_COLUMNS : ())...),
              ((t.set, t.model, t.variant, t.stat, t.start, t.refined, t.change_pct, t.photos_start, t.photos_refined, t.dots_start,
                t.dots_refined, t.retcode_start, t.retcode_refined, t.retcode_refinement, t.converged, t.coverage_dots_start,
                t.coverage_dots_refined, t.coverage_fraction_start, t.coverage_fraction_refined, t.refined_all, t.photos_refined_all,
                (insample ? (t.variant_in_sample, t.in_sample, t.gap, t.retcode_in_sample) : ())...,
                (withci ? (haskey(t, :ci_lo) ? (t.difference, t.ci_lo, t.ci_hi, t.ci_excludes_zero) : (NaN, NaN, NaN, false)) : ())...) for t in long))
    kmd = 4 + (insample ? 2 : 0) + (withci ? 1 : 0)                         # the md's columns per statistic (the tex's: 4, or 3 with inline_change)
    sets = unique(String[String(r.set) for r in rows]); titles = [SUMMARY_STAT_TITLES[String(st)] for st in stats]
    # the best value of each statistic per camera (`mark_best`): the lowest of its rows' starts and refinements, the head-to-head
    camera(r) = String(hasproperty(r, :camera) ? r.camera : r.set)
    lowest = Dict{Tuple{String, String}, Float64}()
    if mark_best
        for r in rows, st in stats
            v = r.start === nothing ? Float64(r.refined[st]) : min(Float64(r.start[st]), Float64(r.refined[st]))
            k = (camera(r), String(st)); lowest[k] = min(get(lowest, k, Inf), v)
        end
    end
    # a row's cells per statistic: the start (blank without one), the refinement, the change (blank without a start) — the head-to-head —
    # then, in the md only, the paired bootstrap's interval and the in-sample score and gap (the tex never prints them: owner 2026-11-15,
    # Ties's table format; they stay in the long form, the numbers and the evaluation's records), and the refinement on all of its own
    # dots (not in the `inline_change` tex); `tex`: a non-improvement's refined value and change in `worse_color` (and bold, without
    # `mark_best`), the camera's best value bold (`mark_best`); `inline_change`: the start, the arrow, the refinement with its signed change
    # in brackets (the sign a math sign)
    function cells(r; tex::Bool = false)
        c = String[]
        for st in stats
            s = r.start === nothing ? NaN : Float64(r.start[st]); e = Float64(r.refined[st])
            worse = tex && !isempty(worse_color) && !isnan(s) && e >= s
            bold(x) = "\\textbf{" * x * "}"
            mark(x) = worse ? "\\textcolor{" * worse_color * "}{" * (mark_best ? x : bold(x)) * "}" : x
            best(x, v) = tex && mark_best && number(v) == number(lowest[(camera(r), String(st))]) ? bold(x) : x   # as printed: a tie at the printed digits is bold too
            own = number(Float64(r.refined["all_" * st]))                                     # the refinement on all of its own dots
            if tex && inline_change
                pct = 100 * (e - s) / s
                push!(c, isnan(s) ? "" : best(number(s), s), isnan(s) ? "" : "\$\\rightarrow\$",
                      mark(best(number(e), e) * (isnan(s) ? "" : "\\,(\$" * (pct > 0 ? "+" : "") * change(pct) * "\$\\%)")))
            else
                push!(c, isnan(s) ? "" : best(number(s), s), mark(best(number(e), e)), isnan(s) ? "" : mark(change(100 * (e - s) / s)))
                tex || withci && push!(c, _ci(r) === nothing ? "" : (q = _ci(r)[st]; "[" * number(q.lo) * ", " * number(q.hi) * "]" * (q.excludes_zero ? "*" : "")))
                tex || insample && (i = Float64(_in_sample(r)[st]); push!(c, number(i), number(e - i)))
                push!(c, own)
            end
        end
        return c
    end
    stat_cols(t) = join(["$t start", "$t refined", "$t Δ %", (withci ? ["$t 95 % CI"] : String[])..., (insample ? ["$t in-sample", "$t gap"] : String[])..., "$t all"], " | ")
    ci_note = withci ? "; CI: the refined − start difference's interval (px) of the paired photograph bootstrap, * where it excludes 0" : ""
    insample_note = insample ? "; in-sample: the refinement scored on the dots its fit saw (variant $(String(first(rows).in_sample["variant"]))), gap: refined − in-sample [px]" : ""
    md = joinpath(dir, "table_$name.md")
    open(md, "w") do io
        println(io, "# ", name, "\n\nChrome-board residuals [px] of the geometric start and the radiometric refinement (variant ",
                isempty(rows) ? "" : String(first(rows).variant), "), the change in percent (negative: an improvement), on the dots both models score; ",
                "\"all\": the refinement on every dot it scores", ci_note, insample_note, "; photographs and dots as start / refined; coverage: the percent of the registered dots each scores, start / refined.\n")
        for s in sets
            println(io, "## ", s, "\n")
            println(io, "| model | ", join([stat_cols(t) for t in titles], " | "), " | photos | dots | coverage % |")
            println(io, "|---|", repeat("---:|", kmd * length(titles)), "---:|---:|---:|")
            for r in rows
                String(r.set) == s || continue
                println(io, "| ", r.model, " | ", join(cells(r), " | "), " | ", _count_cell(r, "photos"), " | ", _count_cell(r, "dots"), " | ", _coverage_cell(r), " |")
            end
            println(io)
        end
    end
    tex = joinpath(dir, "table_$name.tex"); n = length(titles)
    # the tex's header: `inline_change` (Ties's format, owner 2026-11-15) ONE row, each statistic over its start → refined (%) columns;
    # otherwise two, the statistics over start / refined / % / all, then the column names
    head(io, extras) = if inline_change
        println(io, "model & ", join(["\\multicolumn{3}{c}{$t [px]}" for t in titles], " & "), " \\\\")
        println(io, join(["\\cmidrule(lr){$(3j - 1)-$(3j + 1)}" for j in 1:n], " "))
    else
        println(io, " & ", join(["\\multicolumn{4}{c}{$t [px]}" for t in titles], " & "), join(" & " .* extras), " \\\\")
        println(io, join(["\\cmidrule(lr){$(4j - 2)-$(4j + 1)}" for j in 1:n], " "))
        println(io, "model & ", join(["start & refined & \\% & all" for _ in 1:n], " & "), repeat(" &", length(extras)), " \\\\\n\\midrule")
    end
    open(tex, "w") do io
        println(io, "% generated by scripts/chrome_evaluation_paper.jl ([tables.", name, "] of its paper config); do not edit")
        if layout == "blocks"
            # inline_change: per statistic the start (right-aligned), the arrow (a narrow column of its own) and the refinement with its
            # change (left-aligned), no `all` and no coverage column; the tabular keeps its natural width up to the text width and is
            # scaled down beyond it (graphicx's \resizebox)
            ncols = inline_change ? 1 + 3n : 2 + 4n
            println(io, "\\begin{table*}[t]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{", inline_change ? "3.2pt" : "2.4pt", "}")
            inline_change && println(io, "\\resizebox{\\ifdim\\width>\\textwidth\\textwidth\\else\\width\\fi}{!}{%")
            println(io, "\\begin{tabular}{", inline_change ? "@{}l" * repeat("r@{\\hspace{2pt}}c@{\\hspace{2pt}}l", n) * "@{}" : "l" * repeat("rrrr", n) * "r", "}\n\\toprule")
            head(io, ["coverage [\\%]"])
            for (i, s) in enumerate(sets)
                i == 1 || println(io, "\\addlinespace")
                println(io, "\\multicolumn{", ncols, "}{l}{\\textit{", _tex_text(s), "}} \\\\")
                for r in rows
                    String(r.set) == s || continue
                    println(io, _tex_text(r.model), " & ", join(cells(r; tex = true), " & "), inline_change ? "" : " & " * _coverage_cell(r), " \\\\")
                end
            end
            println(io, "\\bottomrule"); println(io, "\\end{tabular}", inline_change ? "}" : "")
            isempty(caption) || println(io, "\\caption{", caption, "}")
            isempty(label) || println(io, "\\label{", label, "}")
            println(io, "\\end{table*}")
        else
            for s in sets
                println(io, "\\begin{table}[t]\n\\centering\n\\small\n\\setlength{\\tabcolsep}{3pt}")
                println(io, "\\begin{tabular}{l", repeat("r", 4n), "rrr}\n\\toprule")
                head(io, ["photos", "dots", "coverage [\\%]"])
                for r in rows
                    String(r.set) == s || continue
                    println(io, _tex_text(r.model), " & ", join(cells(r; tex = true), " & "), " & ", _count_cell(r, "photos"), " & ", _count_cell(r, "dots"),
                            " & ", _coverage_cell(r), " \\\\")
                end
                println(io, "\\bottomrule"); println(io, "\\end{tabular}")
                println(io, "\\caption{", _tex_text(s), ": chrome-board residuals [px] of the geometric start and the radiometric refinement (",
                        isempty(rows) ? "" : _tex_text(String(first(rows).variant)), "), the change in percent (negative: an improvement), on the dots both models score; all: the refinement on every dot it scores; photographs and dots as start / refined; coverage: the percent of the registered dots each scores, start / refined.}")
                println(io, "\\end{table}\n")
            end
        end
    end
    return (csv, md, tex)
end

end # module Figures
