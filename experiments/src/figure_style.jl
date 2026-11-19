# =============================================================================
# src/figure_style.jl — THE figure style of the experiments' pgfplots figures: one VisualGeometryToolkitEval Render `PaperStyle`
# (`PAPER_STYLE`) and the few helpers every pgfplots figure stage of this project calls. THIN by contract, as LocalOptimizationDoneRight's
# accv2026/src/figure_style.jl: the styling machinery is Render's (the house axis conventions `HOUSE_AXIS_COMMON`, the colour emitters
# `color_defs` / `paper_color_defs`, the preview `write_article_preview`); this file supplies the DATA only — the method rows (key,
# legend label, colour name, mark, dashed) and the colours they name. ONE file for every pgfplots figure of the paper (the sample
# efficiency experiment's, the bearing-divergence plot's, …): a figure stage EXTENDS `STYLE_ROWS` / `STYLE_COLORS` here, never defines
# its own palette.
#
# The paper owns the styles: `write_figure_styles` writes `figure_styles.tex`, a `\pgfplotsset` block of one named style per row
# (`rc <key>`: colour, mark, line; `rc <key> band`: the row's fill between) and `rc axis` (the house axis options), with the
# colours as `\providecolor` (the paper's own definition wins). No style holds a width or a height: the paper sizes every figure
# (`every axis/.append style`, a subfigure); a fragment's standalone `preview.tex` (`write_figure_preview`) supplies a default size.
# Included by the figure drivers after src/experiments.jl (which loads VisualGeometryToolkitEval).
# =============================================================================
using VisualGeometryToolkitEval.Render: PaperStyle, HOUSE_AXIS_COMMON, HOUSE_LINE_WIDTH, HOUSE_MARK_SIZE, color_defs, paper_color_defs,
                                        write_article_preview, default_palette, pgf_coords

"""The colours of the rows, `name => "RRGGBB"`: the house base palette's (`default_palette`) — ours the house "ours" red (Jeffreys
e6194b), the baselines hues chosen by the house so none competes with it (MAGSAC++ orange f58231, GaU cyan 42d4f4)."""
const STYLE_COLORS = let p = Dict(default_palette())
    ["ColorOurs" => p["Jeffreys"], "ColorSchoeps" => p["MAGSAC"], "ColorOpenCV" => p["GaU"],
     "ColorSweepFit" => p["IG-MLE"], "ColorSweepPairs" => p["IG-Robust"], "ColorFlat" => p["Oracle"]]
end

"""The method rows of the paper's figures, in legend order: `(key, legend label, colour name, mark, dashed)` (Render's
`PaperStyle` rows). The key is what a figure's data names (the sample efficiency record's `method`)."""
const STYLE_ROWS = [
    ("schoeps", "Sch\\\"{o}ps et al.", "ColorSchoeps", "square*", false),
    ("opencv", "OpenCV model (Kannala--Brandt)", "ColorOpenCV", "triangle*", true),
    ("ours", "Ours", "ColorOurs", "*", false),
    # the convergence figure (src/convergence.jl): the bearing divergence of Schöps's detections in his hue (they are his), the
    # Thorlabs error of our camera in ours; the medians solid, the p95 and the in-sample fit dashed
    ("divergence median", "bearing divergence, median", "ColorSchoeps", "*", false),
    ("divergence p95", "bearing divergence, p95", "ColorSchoeps", "o", true),
    ("thorlabs heldout", "Thorlabs error, held-out median", "ColorOurs", "square*", false),
    ("thorlabs in-sample", "Thorlabs error, in-sample median", "ColorOurs", "square", true),
    # the warping ablation (src/warping_ablation.jl): the held-out median of the geometric start and of each board mode, marks only;
    # the rigid board and the heights in the house's other "ours" hues (IG-MLE, IG-Robust), the start in the neutral grey (Oracle)
    ("warping start", "geometric start", "ColorFlat", "triangle*", false),
    ("warping rigid", "A: rigid board", "ColorSweepFit", "square*", false),
    ("warping heights", "B: heights only", "ColorSweepPairs", "diamond*", false),
    ("warping both", "C: heights + in-plane (of record)", "ColorOurs", "*", false),
]

"THE paper style: the one `PaperStyle` every pgfplots figure of experiments is handed (house rule: no grid)."
const PAPER_STYLE = PaperStyle(; colors = STYLE_COLORS, rows = STYLE_ROWS, grid = "", zeta_x = "", zeta_med_y = (_...; kw...) -> "",
                               zeta_maa_y = "", med_ylabel = "", maa_ylabel = "")

"""
    style_row(key) -> NamedTuple

The style row of `key` in `PAPER_STYLE` as `(; key, label, color, mark, dashed, name)` — `name` its pgfplots style (`rc <key>`); an
`ArgumentError` for a key no row styles (a figure never invents one).
"""
function style_row(key::AbstractString)
    i = findfirst(r -> r[1] == key, PAPER_STYLE.rows)
    i === nothing && throw(ArgumentError("figure_style: no style row $(repr(key)) (rows: $(join(first.(PAPER_STYLE.rows), ", "))); add it to STYLE_ROWS"))
    k, label, color, mark, dashed = PAPER_STYLE.rows[i]
    return (; key = k, label, color, mark, dashed, name = "rc $k")
end

"""
    house_axis(opts...) -> String

The axis options of a figure: Render's `HOUSE_AXIS_COMMON` (no grid, plain-number log ticks, the house fonts and legend), then the
panel's own `opts` (labels, modes, limits) — never a width or a height.
"""
house_axis(opts::AbstractString...) = join([HOUSE_AXIS_COMMON; collect(opts)], ", ")

"""
    write_figure_styles(path) -> path

The paper-owned style sheet: the rows' colours (`\\providecolor`, Render's `paper_color_defs`), one `\\pgfplotsset` style per row —
`rc <key>` (its colour, mark and line at the house width and mark size, dashed where the row says) and `rc <key> band` (its fill
between, 18 % opacity, no outline) — and `rc axis` (`HOUSE_AXIS_COMMON`). No width, no height.
"""
function write_figure_styles(path::AbstractString)
    lines = ["% figure_styles.tex — the experiments' pgfplots styles (src/figure_style.jl write_figure_styles; generated, do not edit).",
             "% \\input once in the preamble (or before a figure fragment). No width or height: the paper sizes every figure.",
             paper_color_defs(PAPER_STYLE),
             "\\pgfplotsset{rc axis/.style={$(HOUSE_AXIS_COMMON)}}"]
    for (k, _, color, mark, dashed) in PAPER_STYLE.rows
        push!(lines, "\\pgfplotsset{rc $k/.style={color=$color, mark=$mark, mark size=$HOUSE_MARK_SIZE, $HOUSE_LINE_WIDTH, $(dashed ? "dashed" : "solid"), mark options={solid}}}",
                     "\\pgfplotsset{rc $k band/.style={fill=$color, fill opacity=0.18, draw=none}}")
    end
    mkpath(dirname(path))
    write(path, join(lines, '\n') * '\n')
    return path
end

"""
    write_figure_preview(dir, fragment; styles = "figure_styles", size = (\"8cm\", \"6cm\")) -> nothing

A standalone `preview.tex` in `dir` for the figure fragment `fragment` (its file name without .tex, in `dir`): Render's
`write_article_preview` with the style sheet `styles` (`write_figure_styles`, in `dir`) and the preview's own default axis size —
the one place a size is stated; compiled when latexmk is installed.
"""
function write_figure_preview(dir::AbstractString, fragment::AbstractString; styles::AbstractString = "figure_styles",
                              size::Tuple{AbstractString, AbstractString} = ("8cm", "6cm"))
    pre = "\\input{$styles}\n\\pgfplotsset{every axis/.append style={width=$(size[1]), height=$(size[2])}}"
    write_article_preview(dir, fragment, pre)
    return nothing
end

"""
    dual_axis(left, right; x, xlabel) -> String

A figure of two quantities on one x axis with their own y axes — pgfplots' two overlaid `axis` environments in one `tikzpicture`:
the first `axis y line*=left`, the second `axis y line*=right, axis x line=none`, both on the x range of `x` (the curves' common
abscissae, labelled `xlabel` below the first) and the house options (`house_axis`). `left` and `right` are `(; opts, curves)`: `opts` the axis's own options (its
`ylabel`, limits; the axis's y label and ticks are drawn in its first curve's colour), `curves` the `(key, ys)` pairs it draws, each
in its style row's style (`rc <key>`, `style_row`) with Render's `pgf_coords` of `(x, ys)` (a non-finite y dropped). The legend
(above the axes, two columns; no text inside them) lists every curve of both axes by its row's label: the left axis's through
`\addlegendimage`. No width or height.
"""
function dual_axis(left, right; x::AbstractVector{<:Real}, xlabel::AbstractString)
    coords(ys) = pgf_coords(x, ys; fmt = eltype(x) <: Integer ? "(%d,%.5f)" : "(%.6f,%.5f)")
    xr = "xmin=$(minimum(x)), xmax=$(maximum(x))"
    # each y axis in its first curve's colour, its tick labels plain fixed-point numbers (no ×10ⁿ factor: the house's plain numbers)
    tint(side) = (c = style_row(first(first(side.curves))).color;
                  "ylabel style={color=$c}, y tick label style={color=$c, /pgf/number format/fixed}, scaled y ticks=false")
    io = IOBuffer()
    println(io, "\\begin{tikzpicture}")
    println(io, "\\begin{axis}[rc axis, $xr, axis y line*=left, xlabel={$xlabel}, $(tint(left)), $(left.opts)]")
    foreach(((k, ys),) -> println(io, "  \\addplot[$(style_row(k).name)] coordinates {", coords(ys), "};"), left.curves)
    println(io, "\\end{axis}")
    println(io, "\\begin{axis}[rc axis, $xr, axis y line*=right, axis x line=none, $(tint(right)), legend columns=2, legend style={at={(0.5,1.03)}, anchor=south}, $(right.opts)]")
    foreach(((k, _),) -> println(io, "  \\addlegendimage{$(style_row(k).name)}\\addlegendentry{$(style_row(k).label)}"), left.curves)
    foreach(((k, ys),) -> println(io, "  \\addplot[$(style_row(k).name)] coordinates {", coords(ys), "};\\addlegendentry{$(style_row(k).label)}"), right.curves)
    println(io, "\\end{axis}")
    println(io, "\\end{tikzpicture}")
    return String(take!(io))
end
