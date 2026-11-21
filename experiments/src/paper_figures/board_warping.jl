# =============================================================================
# src/paper_figures/board_warping.jl — the paper's board-warping figure (fig:board-warping, src/fig/board-warping.tex, its panels
# board-warping/<set>.tex and its colour key board-warping/key.tex): the height of the printed calibration boards above their plane
# as the radiometric refinement of one camera model estimated it, the two targets in rows and the three lenses in columns. The
# port of T. Harder's board_warping.py (the figure folder's tools/), step for step; every number is printed as Python printed it.
# What the paper's figure needs is ported: the drawing with 3-D surface panels (the Python's figure3d) and the per-case table of
# its `tables`. Its drawing with image panels and its comparison of several camera models' surfaces on one set are not here.
#
# What the drawing shows is the caller's to state, nothing of it has a value here: the figure's name, the sets in rows and columns
# with their labels, the camera model, the candidates of a band's height, the mesh's stride, the view and the heights' exaggeration
# (`bw_figure`'s keywords; scripts/board_warping.jl hands over `[figure]` of its config of record).
#
# The data folder holds what the stack dumped from the refinements (the Julia step tools/warping_dump.jl of the figure folder):
#   cases.tsv                    one line per case: case, set, model, refine_product (tab-separated, one header line)
#   dump/<case>.f64              the height field over the page: Int64 n, Float64 page width and height [m], n × n Float64
#                                heights [m], column-major with the first index along u (little-endian)
#   dump/<case>_nodes.tsv        the warping's nodes: u, v (page fractions, v from the top), nodal_height_mm, surface_mm
# A case is named <set>_<model>_warped.
# =============================================================================
import LinearAlgebra

"`f\"{v:g}\"`: a number at six significant digits, trailing zeros dropped."
bw_g(v::Real) = general(v, 6)
"`f\"{v:+.<digits>f}\"`: fixed notation with the sign always written."
bw_signed(v::Real, digits::Integer) = Printf.format(Printf.Format("%+.$(digits)f"), Float64(v))

"""
    bw_load(dump, case) -> (W, H, z)

The height field of `case`: the page's width and height in mm and the heights in mm, rows = v (from the page's top), columns = u.
"""
function bw_load(dump::AbstractString, case::AbstractString)
    open(joinpath(dump, "$case.f64")) do io
        n = Int(ltoh(read(io, Int64))); W = ltoh(read(io, Float64)); H = ltoh(read(io, Float64))
        Z = Matrix{Float64}(undef, n, n); read!(io, Z)                    # Z[i, j]: u index i, v index j
        return 1e3 * W, 1e3 * H, 1e3 .* permutedims(ltoh.(Z))
    end
end

"The cases of `data`/cases.tsv in the file's order: (case, set, model, product)."
function bw_cases(data::AbstractString)
    rows = [split(l, '\t') for l in readlines(joinpath(data, "cases.tsv"))[2:end] if !isempty(l)]
    return [(case = String(r[1]), set = String(r[2]), model = String(r[3]), product = String(r[4])) for r in rows]
end

"The nodes of a case (the rows of dump/<case>_nodes.tsv): u, v, nodal height, surface height."
bw_nodes(dump::AbstractString, case::AbstractString) =
    [parse.(Float64, split(l)) for l in readlines(joinpath(dump, "$(case)_nodes.tsv"))[2:end] if !isempty(strip(l))]

"`D` without its least-squares plane (the plane belongs to the poses, not to the warping)."
function bw_deplane(D::AbstractMatrix)
    n, m = size(D)
    A = [k == 1 ? 1.0 : k == 2 ? Float64(i - 1) : Float64(j - 1) for j in 1:n, i in 1:m, k in 1:3]   # 1, the column, the row of every sample
    A = reshape(A, n * m, 3)
    return D .- reshape(A * (A \ vec(D)), n, m)
end

"The root mean square of the values of `D`."
bw_rms(D) = sqrt(Statistics.mean(abs2, D))

"""
    bw_tables(data, out; generator) -> [path]

Writes `out`/meta.toml: per case of the data folder its set, camera model and refine product, the page's size and the surface's
range and RMS (as it is, and without its least-squares plane). board_warping.py wrote this file into the data folder; the figure
does not read it. `generator` is what the file names as its writer.
"""
function bw_tables(data::AbstractString, out::AbstractString; generator::AbstractString)
    dump = joinpath(data, "dump")
    L = ["# the estimated board warping per case; written by $generator", ""]
    for q in bw_cases(data)
        W, H, z = bw_load(dump, q.case)
        push!(L, "[cases.$(q.case)]", "set = \"$(q.set)\"", "model = \"$(q.model)\"", "refine_product = \"$(basename(q.product))\"",
              "page_mm = [$(fixed(W, 3)), $(fixed(H, 3))]", "surface_min_mm = $(fixed(minimum(z), 4))", "surface_max_mm = $(fixed(maximum(z), 4))",
              "surface_rms_mm = $(fixed(bw_rms(z), 4))", "surface_rms_without_plane_mm = $(fixed(bw_rms(bw_deplane(z)), 4))", "")
    end
    return [write_lines(joinpath(out, "meta.toml"), L)]
end

"The scale's colour (r, g, b in 0 … 255) at `t` in [-1, 1]: linear between the classes of the diverging scale."
function bw_colour(t::Real)
    # a diverging scale (ColorBrewer RdBu, 11 classes): blue = behind the plane (away from the camera), white = on it, red = towards the camera
    rdbu = ((5.0, 48.0, 97.0), (33.0, 102.0, 172.0), (67.0, 147.0, 195.0), (146.0, 197.0, 222.0), (209.0, 229.0, 240.0), (247.0, 247.0, 247.0),
            (253.0, 219.0, 199.0), (244.0, 165.0, 130.0), (214.0, 96.0, 77.0), (178.0, 24.0, 43.0), (103.0, 0.0, 31.0))
    x = (clamp(t, -1.0, 1.0) + 1) / 2 * (length(rdbu) - 1); i = clamp(floor(Int, x), 0, length(rdbu) - 2); f = x - i
    return ntuple(k -> rdbu[i+1][k] * (1 - f) + rdbu[i+2][k] * f, 3)
end
"An xcolor colour of an (r, g, b) in 0 … 255."
bw_xcol(c) = "{rgb,255:red,$(round(Int, c[1]));green,$(round(Int, c[2]));blue,$(round(Int, c[3]))}"

"""
    bw_panel3d(path, s, model, product, W, H, z, nodes, nb, band, lim; asset_dir, mesh_stride, view_azimuth_elevation_deg, height_exaggeration, generator)

One panel: the height field of set `s` as a PGFPlots surface over the page (mm), coloured in `nb` bands of `band` mm on each side
of zero, the box -lim … +lim for every panel. Every `mesh_stride`-th sample is a mesh point; the view is PGFPlots' (its azimuth
and elevation in degrees), the heights `height_exaggeration` times as large as lengths on the page. `asset_dir` is the figure's (the
file is its `<asset_dir>/<s>.tex`) and `generator` what the first line names as the file's generator.
"""
function bw_panel3d(path::AbstractString, s::AbstractString, model::AbstractString, product::AbstractString, W::Real, H::Real, z::AbstractMatrix, nodes, nb::Integer, band::Real, lim::Real;
                    asset_dir::AbstractString, mesh_stride::Integer, view_azimuth_elevation_deg, height_exaggeration::Real, generator::AbstractString)
    azimuth, elevation = view_azimuth_elevation_deg
    zlbl = 10                                                # pt at a panel's left for the labels of the height's ticks
    n = size(z, 1); idx = collect(0:mesh_stride:n-1); idx[end] == n - 1 || push!(idx, n - 1)   # 0-based samples; the last one closes the mesh
    # rows of constant y, from the page's bottom edge
    rows = [join(("($(fixed(i / (n - 1) * W, 1)),$(fixed((1 - j / (n - 1)) * H, 1)),$(fixed(z[j+1, i+1], 3)))" for i in idx), " ") for j in reverse(idx)]
    a = deg2rad(azimuth); e = deg2rad(elevation)
    wide = W * cos(a) + H * sin(a); high = (W * sin(a) + H * cos(a)) * sin(e) + height_exaggeration * 2 * lim * cos(e)
    cm = join(("rgb255=($(round(Int, c[1])),$(round(Int, c[2])),$(round(Int, c[3])))" for c in (bw_colour((k + 0.5) / nb) for k in -nb:nb-1)), " ")
    # three ticks: 0 and about two thirds of the limit
    zt = lim >= 1.5 ? [t for t in (-2 * floor(lim / 3 + 0.5), 0.0, 2 * floor(lim / 3 + 0.5)) if abs(t) < lim] : [-lim / 2, 0.0, lim / 2]
    return write_lines(path, [
        "% $asset_dir/$s.tex — generated by $generator; do not edit.",
        "% One 3-D panel: the estimated height field of set $s as a surface over the page ($(fixed(W, 0)) x $(fixed(H, 0)) mm; x to the right, y up the page,",
        "% z the height in mm, towards the camera), $model camera, refine product $(basename(product)). $(length(idx)) x $(length(idx)) mesh points; every patch has",
        "% the colour of its band ($(bw_g(band)) mm). Orthographic view (azimuth $azimuth, elevation $elevation degrees) with the heights $height_exaggeration times as large as",
        "% lengths on the page: the axis' height is chosen so that PGFPlots' filling of width x height is exactly that projection. The box is",
        "% -$(bw_g(lim)) ... +$(bw_g(lim)) mm in every panel. Its width is \\rcpanelwidth, of which $zlbl pt at the left hold the tick labels.",
        "\\providecommand{\\rcfigdir}{src/fig}%", "\\providecommand{\\rcpanelwidth}{\\linewidth}%",
        "\\begin{tikzpicture}[baseline=(current axis.center)]",
        "\\pgfmathsetlengthmacro{\\rcwarpw}{\\rcpanelwidth-$(zlbl)pt}%",
        "\\begin{axis}[",
        "  scale only axis, width=\\rcwarpw, height=$(fixed(high / wide, 5))*\\rcwarpw, view={$azimuth}{$elevation},",
        "  xmin=0, xmax=$(fixed(W, 1)), ymin=0, ymax=$(fixed(H, 1)), zmin=$(bw_g(-lim)), zmax=$(bw_g(lim)), clip=false,",
        "  xtick=\\empty, ytick=\\empty, ztick={" * join(bw_g.(zt), ",") * "}, zticklabel style={font=\\tiny, inner sep=1pt},",
        "  tick style={line width=0.2pt, black!60}, major tick length=1.5pt, zmajorgrids, grid style={line width=0.15pt, black!22},",
        "  axis line style={line width=0.2pt, black!60},",
        "  colormap={warp}{$cm}, colormap access=piecewise const, point meta min=$(bw_g(-lim)), point meta max=$(bw_g(lim))]",
        "\\addplot3[surf, shader=faceted, faceted color=black!45, line width=0.08pt, mesh/rows=$(length(idx)), z buffer=sort, point meta=z] coordinates {",
        rows..., "};",
        "\\addplot3[only marks, mark=*, mark size=0.4pt, black] coordinates {" * join(("($(fixed(q[1] * W, 1)),$(fixed((1 - q[2]) * H, 1)),$(fixed(q[4], 3)))" for q in nodes), " ") * "};",
        "\\end{axis}",
        "\\pgfresetboundingbox\\path ([xshift=-$(zlbl)pt]current axis.south west) rectangle (current axis.north east);   % the box is the axis and the strip of the tick labels",
        "\\end{tikzpicture}%"])
end

"""
    bw_figure(data, out; asset_dir, model, rows, columns, band_heights_mm, max_bands, mesh_stride, view_azimuth_elevation_deg,
              height_exaggeration, generator) -> paths

Draws the figure `asset_dir` from the refinements of the camera model `model` (the case of set `s` is `<s>_<model>_warped`) into `out`:
`<asset_dir>.tex` (\\input it inside a figure; it follows \\linewidth and needs pgfplots, compat 1.18), one panel per set in `<asset_dir>/`,
the colour key `<asset_dir>/key.tex`, `<asset_dir>-caption.tex` and source.toml. ONE colour scale for all panels, symmetric about zero, in
bands (so the bands' borders are the contour lines). Every panel is a PGFPlots surface over the page: a mesh of every
`mesh_stride`-th sample, each patch in the colour of its band, an orthographic view from the page's bottom edge, the heights
`height_exaggeration` times as large as lengths on the page, the same box for all panels; no image files.

Every keyword is required, the caller states what is drawn:
  - `rows` and `columns`: the layout — per row its label and its sets, one per column, and the columns' labels (LaTeX text);
  - `band_heights_mm` and `max_bands`: the candidates of a band's height in mm, ascending — the first that needs at most
    `max_bands` bands on each side of zero for the largest |height| of all panels is taken, the last when none does;
  - `view_azimuth_elevation_deg`: the panels' view, PGFPlots' azimuth and elevation in degrees, each within 0 … 90 (the axis'
    height is computed for a view from that quadrant);
  - `generator`: what the files' first lines name as their generator.
"""
function bw_figure(data::AbstractString, out::AbstractString; asset_dir::AbstractString, model::AbstractString, rows, columns, band_heights_mm, max_bands::Integer,
                   mesh_stride::Integer, view_azimuth_elevation_deg, height_exaggeration::Real, generator::AbstractString)
    GEN = "generated by $generator; do not edit"
    gap, rowlabel = 3, 9                                     # pt between panels; pt of the row labels' column
    keyh, keyshare = 5, 0.62                                 # the key: its height in pt and its share of \linewidth
    all(length(ss) == length(columns) for (_, ss) in rows) || error("rows: every row names one set per column ($(length(columns)))")
    dump = joinpath(data, "dump"); dir = joinpath(out, asset_dir); written = String[]
    product = Dict(q.case => q.product for q in bw_cases(data))
    sets = [s for (_, ss) in rows for s in ss]; case(s) = "$(s)_$(model)_warped"
    P = Dict(s => bw_load(dump, case(s)) for s in sets)
    top = maximum(maximum(abs, P[s][3]) for s in sets)
    k = findfirst(b -> ceil(top / b) <= max_bands, band_heights_mm); band = band_heights_mm[something(k, length(band_heights_mm))]
    nb = Int(ceil(top / band - 1e-9)); lim = nb * band
    for s in sets
        W, H, z = P[s]
        push!(written, bw_panel3d(joinpath(dir, "$s.tex"), s, model, product[case(s)], W, H, z, bw_nodes(dump, case(s)), nb, band, lim;
                                  asset_dir, mesh_stride, view_azimuth_elevation_deg, height_exaggeration, generator))
    end
    # the key: the bands between -lim and +lim, a tick at every second border
    bw = "\\dimexpr$(pystr(keyshare))\\linewidth/$(2 * nb)\\relax"
    tick(v) = v == 0 ? "0" : replace(rstrip(rstrip(bw_signed(v, 2), '0'), '.'), "-" => "{-}", "+" => "{+}")
    push!(written, write_lines(joinpath(dir, "key.tex"), [
        "% $asset_dir/key.tex — $GEN.",
        "% The colour key of the panels: $(2 * nb) bands of $(bw_g(band)) mm from $(bw_g(-lim)) to $(bw_g(lim)) mm; its width is $(pystr(keyshare)) \\linewidth.",
        "\\begin{tikzpicture}[baseline=(current bounding box.north)]",
        ("\\fill[fill=$(bw_xcol(bw_colour((k + 0.5) / nb)))] ($(k + nb)*$bw,0) rectangle ($(k + nb + 1)*$bw+0.05pt,$(keyh)pt);" for k in -nb:nb-1)...,
        "\\draw[line width=0.2pt] (0,0) rectangle ($(2 * nb)*$bw,$(keyh)pt);",
        ("\\draw[line width=0.2pt] ($(k + nb)*$bw,0) -- ++(0,-1.5pt) node[below, inner sep=1pt, font=\\scriptsize] {\$$(tick(k * band))\$};" for k in -nb:(nb > 3 ? 2 : 1):nb)...,
        "\\node[right, inner sep=2pt, font=\\scriptsize] at ($(2 * nb)*$bw,$(pystr(keyh / 2))pt) {mm};",
        "\\end{tikzpicture}%"]))
    cell(s) = "\\input{\\rcfigdir/$asset_dir/$s}"; nc = length(columns)
    push!(written, write_lines(joinpath(out, "$asset_dir.tex"), [
        "% $asset_dir.tex — $GEN.",
        "% The estimated board warping: $(length(rows)) rows (the targets) x $nc columns (the lenses) of panels ($asset_dir/<set>.tex, one PGFPlots",
        "% picture each: the height field as a 3-D surface over the page), ONE colour key under them. The lens names, the target names and the",
        "% key's numbers are LaTeX text in the host document's fonts. \\input it inside a figure: it follows \\linewidth (one column or the",
        "% text width); it needs pgfplots (compat 1.18). \\rcfigdir is the folder this file lies in, as the host document sees it (default src/fig).",
        "\\providecommand{\\rcfigdir}{src/fig}%",
        "\\begingroup",
        "\\ifdefined\\rcpanelwd\\else\\newlength\\rcpanelwd\\fi",
        "\\setlength{\\rcpanelwd}{\\dimexpr(\\linewidth-$(rowlabel + gap * nc)pt)/$nc\\relax}\\def\\rcpanelwidth{\\rcpanelwd}%",
        "\\footnotesize\\setlength{\\tabcolsep}{0pt}\\renewcommand{\\arraystretch}{1}%",
        "\\begin{tabular}{@{}p{$(rowlabel)pt}" * "@{\\hspace{$(gap)pt}}c"^nc * "@{}}",
        "  & " * join(("\\makebox[0pt]{\\scriptsize $c}" for c in columns), " & ") * " \\\\\\noalign{\\vskip 2pt}",
        ("  \\makebox[$(rowlabel)pt][c]{\\rotatebox[origin=c]{90}{\\scriptsize $r}} & " * join((cell(s) for s in ss), " & ") * " \\\\\\noalign{\\vskip $(gap)pt}" for (r, ss) in rows)...,
        "  \\multicolumn{$(nc + 1)}{@{}c@{}}{\\input{\\rcfigdir/$asset_dir/key}} \\\\\\noalign{\\vskip 1pt}",
        "  \\multicolumn{$(nc + 1)}{@{}c@{}}{\\scriptsize height above the board's plane (positive: towards the camera)} \\\\",
        "\\end{tabular}%", "\\endgroup"]))
    # the caption names each target's page size (the sizes of its sets, should they differ)
    size_of(ss) = join(("\$$w\\times$h\$\\,mm" for (w, h) in sort(unique((round(Int, P[s][1]), round(Int, P[s][2])) for s in ss))), " and ")
    push!(written, write_lines(joinpath(out, "$asset_dir-caption.tex"), [
        "The board warping estimated jointly with the radiometric camera ($model camera model), for the two calibration targets (rows) and the " *
        "three lenses (columns): the height of the printed page above its plane, a thin-plate spline through \$5\\times5\$ nodes (dots), " *
        "drawn as a surface over the page with the heights $height_exaggeration times as large as lengths on the page (ticks in mm), coloured " *
        "in bands of $(bw_g(band))\\,mm; positive heights lie towards the camera. " *
        join(("The $r page is $(size_of(ss))." for (r, ss) in rows), " ")]))
    push!(written, write_lines(joinpath(out, "source.toml"), [
        "# what $asset_dir was drawn from; written by $generator", "model = \"$model\"", "band_mm = $(bw_g(band))", "limit_mm = $(bw_g(lim))",
        "panels = \"3-D surfaces\"", "view_azimuth_elevation_deg = [$(view_azimuth_elevation_deg[1]), $(view_azimuth_elevation_deg[2])]", "height_exaggeration = $height_exaggeration",
        "mesh_stride = $mesh_stride",
        ("$s = \"$(product[case(s)])\"" for s in sets)...]))
    println("-> $out/$asset_dir.tex + $asset_dir/ ($model: largest |height| $(fixed(top, 3)) mm; $(2 * nb) bands of $(bw_g(band)) mm, scale ±$(bw_g(lim)) mm)")
    for (r, ss) in rows, s in ss
        W, H, z = P[s]
        println("   ", rpad(r, 14), " ", rpad(s, 7), " page $(fixed(W, 0)) x $(fixed(H, 0)) mm, surface $(bw_signed(minimum(z), 3)) … $(bw_signed(maximum(z), 3)) mm")
    end
    return written
end

"""
    figure_board_warping(; data, out, part, generator, settings...) -> paths

The board-warping figure from the data folder `data` into `out`. `part` is board_warping.py's subcommand:

  - `figure3d` (the paper's drawing): every panel a 3-D surface over the page; no image files. `settings` are what is drawn,
    every one required (`bw_figure`'s keywords: `asset_dir`, `model`, `rows`, `columns`, `band_heights_mm`, `max_bands`, `mesh_stride`,
    `view_azimuth_elevation_deg`, `height_exaggeration`).
  - `tables`: `out`/meta.toml, per case the page's size and the surface's range and RMS (no drawing: it takes no setting).

`generator`: what the written files name as their generator (by default this file; the stage `figures` of scripts/board_warping.jl
names itself).
"""
function figure_board_warping(; data::AbstractString, out::AbstractString, part::AbstractString,
                              generator::AbstractString = "experiments/src/paper_figures/board_warping.jl", settings...)
    part in ("figure3d", "tables") || error("part takes \"figure3d\" or \"tables\"")
    part == "tables" && return bw_tables(data, out; generator, settings...)
    return bw_figure(data, out; generator, settings...)
end

FIGURES["board-warping"] = figure_board_warping
