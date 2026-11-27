# =============================================================================
# src/paper_figures/board_warping_offsets.jl — the paper's board-warping figure (fig:board-warping; <asset_dir>.tex with its panels
# and keys in <asset_dir>/): both printed calibration boards as the calibration with each of three lenses reconstructs them, from the
# photographs of ONE session in which a board stayed in place for its three sets. The port of T. Harder's board_warping_offsets.py
# (the figure folder's tools/), its variant b6 in the layout with large surfaces, the surfaces first and small maps; every number is
# printed as Python printed it.
#
# What is drawn is the board's OFFSET learned with a lens: the vector by which the calibration moves a point of the planar board.
#   in the plane      the in-plane print error Aspect(a) ∘ Skew(u) in the placed chart c = (b1, −α b2) of the page fraction
#                     b = (b1 across, b2 down), α = height / width (the page in units of its width, origin at its top-left corner,
#                     y up): p = (eᵃ (c1 + u c2), e⁻ᵃ c2), the offset width · (p − c) in mm; a and u from the refinement's
#                     radiometric_target.json
#   along the normal  the height of the refinement's height field at that point, in mm
# Nothing is fitted and nothing is removed (no best-fit plane, no rigid motion).
#
#   surfaces  per set the planar board PLUS the board's offsets learned with its lens, the whole vector drawn enlarged: a PGFPlots
#             surface whose patches have the colour of their height's band, in the orthographic view the caller states
#   maps      per board and pair of lenses the LENGTH of the difference between the offsets learned with the two lenses, at every
#             sample of the page, as an image in bands of a sequential scale; its RMS and its largest value under the map
#
# What the drawing shows is the caller's to state, nothing of it has a value here (`bwo_figure`'s keywords; scripts/board_warping.jl
# hands over `[figure]` of configs/supplement/board_warping_lenses.toml). The data folder is the board-warping figure's (cases.tsv,
# dump/<case>.f64, dump/<case>_nodes.tsv: board_warping.jl states the formats), with radiometric_target.json of each case's refine
# product for the in-plane print error, as the three-lens figure's tables read it (board_warping_lenses.jl).
# =============================================================================
using ..BoardWarping: bw_cases, bw_load, bw_nodes, bw_rms, bw_g, bw_xcol
using ..BoardWarpingLenses: bwl_inplane, bwl_table_lines

"""
    bwo_print_error(b1, b2, W, H, a, u) -> (dx, dy)

The in-plane offset in mm (x along the columns, y up) of the design point at the page fraction (`b1` across, `b2` down) of a page
of `W` × `H` mm: Aspect(`a`) ∘ Skew(`u`) in the placed chart, minus the point itself.
"""
function bwo_print_error(b1::Real, b2::Real, W::Real, H::Real, a::Real, u::Real)
    c1 = b1; c2 = -(H / W) * b2
    return W * (exp(a) * (c1 + u * c2) - c1), W * (exp(-a) * c2 - c2)
end

"""
    bwo_offsets(data) -> Dict

Per set of the data folder `data` the board's offsets learned with its lens, on the lattice of its height field, in mm:
`(; W, H, dx, dy, z, a, u, case, product)` — the page, the offsets in the plane and along the normal (rows = v from the page's top,
columns = u) and the in-plane aspect and skew they come from. A refinement without an in-plane print error is an error.
"""
function bwo_offsets(data::AbstractString)
    O = Dict{String, NamedTuple}()
    for q in bw_cases(data)
        W, H, z = bw_load(joinpath(data, "dump"), q.case); n = size(z, 1)
        ip = bwl_inplane(q.product)
        haskey(ip, "aspect") && haskey(ip, "skew") || error("$(q.product): the refinement of case $(q.case) has no in-plane print error (aspect and skew)")
        a, u = ip["aspect"], ip["skew"]
        d = [bwo_print_error((i - 1) / (n - 1), (j - 1) / (n - 1), W, H, a, u) for j in 1:n, i in 1:n]
        O[q.set] = (; W, H, dx = first.(d), dy = last.(d), z, a, u, q.case, q.product)
    end
    return O
end

"""
    bwo_differences(O, rows, pairs) -> Vector{NamedTuple}

Per board (a row of `rows`: its label and its sets, one per lens) and pair of lenses (`pairs`: the two columns, the first lens
minus the second) the difference between the offsets learned with the two lenses (`O`, `bwo_offsets`), in the order of the maps:
`(; row, a, b, W, H, d3, rms, most, plane, normal, normal_most)` — the two sets, the page, the length of the difference at every
sample, its RMS and largest value, and the RMS of its part in the plane and of its part along the normal (the heights' difference,
with its largest absolute value), all in mm.
"""
function bwo_differences(O::AbstractDict, rows, pairs)
    D = NamedTuple[]
    for (r, ss) in rows, ((f, s), _) in pairs
        haskey(O, ss[f]) && haskey(O, ss[s]) || error("rows: the data folder has no case of a set of the row \"$r\" ($(ss[f]), $(ss[s]))")
        A, B = O[ss[f]], O[ss[s]]
        (A.W, A.H) == (B.W, B.H) && size(A.z) == size(B.z) || error("two sets of one board with different pages or lattices: $(ss[f]), $(ss[s])")
        dx = A.dx .- B.dx; dy = A.dy .- B.dy; dz = A.z .- B.z
        d3 = sqrt.(dx .^ 2 .+ dy .^ 2 .+ dz .^ 2)
        push!(D, (; row = r, a = ss[f], b = ss[s], A.W, A.H, d3, rms = bw_rms(d3), most = maximum(d3), plane = sqrt(Statistics.mean(dx .^ 2 .+ dy .^ 2)),
                    normal = bw_rms(dz), normal_most = maximum(abs, dz)))
    end
    return D
end

"""
    bwo_tables(data, out; rows, pairs, generator) -> [path]

Writes `out`/meta.toml: the three-lens figure's tables (`bwl_table_lines`: per case the page, the surface and the in-plane aspect
and skew; per board the lenses' agreement in the heights alone, a best-fit plane removed) and, per board and pair of lenses, the
measure the figure draws — the difference between the board's offsets learned with the two lenses, nothing removed: its RMS, the
RMS of its part in the plane and of its part along the normal, and its largest value (mm).
"""
function bwo_tables(data::AbstractString, out::AbstractString; rows, pairs, generator::AbstractString)
    L = bwl_table_lines(data; rows, generator)
    D = bwo_differences(bwo_offsets(data), rows, pairs)
    for (r, _) in rows
        push!(L, "[offsets.$(replace(lowercase(r), " " => "_"))]",
              "# the difference between the board's offsets learned with two lenses (in the plane and along the normal), nothing removed [mm]: its RMS over the page,",
              "# the RMS of its part in the plane and of its part along the normal, its largest value")
        for q in D
            q.row == r || continue
            push!(L, "\"$(q.a) - $(q.b)\" = { rms_mm = $(fixed(q.rms, 4)), inplane_rms_mm = $(fixed(q.plane, 4)), normal_rms_mm = $(fixed(q.normal, 4)), max_mm = $(fixed(q.most, 4)) }")
            println(" "^14, " $(q.a) − $(q.b): whole offset RMS $(fixed(q.rms, 3)) mm (in the plane $(fixed(q.plane, 3)), along the normal $(fixed(q.normal, 3))), largest $(fixed(q.most, 3)) mm")
        end
        push!(L, "")
    end
    return [write_lines(joinpath(out, "meta.toml"), L)]
end

"The colour (r, g, b in 0 … 255) at `t` in [0, 1] of the scale `classes`: linear between its classes."
function bwo_colour(t::Real, classes)
    x = clamp(t, 0, 1) * (length(classes) - 1); i = clamp(floor(Int, x), 0, length(classes) - 2); f = x - i
    return ntuple(k -> classes[i+1][k] * (1 - f) + classes[i+2][k] * f, 3)
end

"The colours of the heights' `n` bands, lowest first: a diverging scale (ColorBrewer RdBu, 11 classes; blue = behind the plane, white = on it, red = towards the camera)."
function bwo_height_colours(n::Integer)
    rdbu = ((5.0, 48.0, 97.0), (33.0, 102.0, 172.0), (67.0, 147.0, 195.0), (146.0, 197.0, 222.0), (209.0, 229.0, 240.0), (247.0, 247.0, 247.0),
            (253.0, 219.0, 199.0), (244.0, 165.0, 130.0), (214.0, 96.0, 77.0), (178.0, 24.0, 43.0), (103.0, 0.0, 31.0))
    return [bwo_colour((k + 0.5) / n, rdbu) for k in 0:n-1]
end

"The colours of the maps' `n` bands from zero: ONE hue, light to dark (ColorBrewer Greens, the first 8 of 9 classes: the dots stay visible on the darkest band)."
function bwo_map_colours(n::Integer)
    greens = ((247.0, 252.0, 245.0), (229.0, 245.0, 224.0), (199.0, 233.0, 192.0), (161.0, 217.0, 155.0), (116.0, 196.0, 118.0), (65.0, 171.0, 93.0),
              (35.0, 139.0, 69.0), (0.0, 109.0, 44.0))
    return [bwo_colour((k + 0.5) / n, greens) for k in 0:n-1]
end

"""
    bwo_key(path, name, colours, lo, band, signed; generator) -> path

A colour key as its own TikZ picture, the file `name` of the figure: the bands of `colours`, each `band` mm, the first band's
lower border at `lo` · `band`; a number at both ends and at every k-th border between them (at most seven), with its sign when
`signed`; "mm" at its right.
"""
function bwo_key(path::AbstractString, name::AbstractString, colours, lo::Integer, band::Real, signed::Bool; generator::AbstractString)
    nb = length(colours); hi = lo + nb
    bw = "\\dimexpr0.62\\linewidth/$nb\\relax"                              # a band's width: the key is 0.62 \linewidth wide
    k = first(k for k in 1:nb if nb % k == 0 && mod(lo, k) == 0 && nb ÷ k + 1 <= 7)
    number(v) = v == 0 ? "0" : replace(rstrip(rstrip(Printf.format(Printf.Format(signed ? "%+.2f" : "%.2f"), Float64(v)), '0'), '.'), "-" => "{-}", "+" => "{+}")
    return write_lines(path, [
        "% $name — generated by $generator; do not edit.",
        "% The colour key of the panels: $nb bands of $(bw_g(band)) mm from $(bw_g(lo * band)) to $(bw_g(hi * band)) mm; its width is 0.62 \\linewidth.",
        "\\begin{tikzpicture}[baseline=(current bounding box.north)]",
        ("\\fill[fill=$(bw_xcol(c))] ($(i - 1)*$bw,0) rectangle ($i*$bw+0.05pt,5pt);" for (i, c) in enumerate(colours))...,
        "\\draw[line width=0.2pt] (0,0) rectangle ($nb*$bw,5pt);",
        ("\\draw[line width=0.2pt] ($(t - lo)*$bw,0) -- ++(0,-1.5pt) node[below, inner sep=1pt, font=\\scriptsize] {\$$(number(t * band))\$};" for t in lo:k:hi)...,
        "\\node[right, inner sep=2pt, font=\\scriptsize] at ($nb*$bw,2.5pt) {mm};",
        "\\end{tikzpicture}%"])
end

"""
    bwo_projection(S, lim; view_azimuth_elevation_deg, offset_enlargement) -> (elevation, ratio)

How PGFPlots is made to draw the orthographic view from `view_azimuth_elevation_deg` of a box of `S` × `S` mm of page and
−`lim` … +`lim` mm of height, the height `offset_enlargement` times as large as lengths on the page. PGFPlots projects the UNIT
cube of the three axis ranges with the angles of `view` and then stretches the picture to width × height; that is the wanted view
exactly when the x and y ranges are equal, the elevation handed over is atan(S tan(elevation) / (enlargement · 2 lim)) and
height / width is the projection's. Returns that elevation in degrees and height / width.
"""
function bwo_projection(S::Real, lim::Real; view_azimuth_elevation_deg, offset_enlargement::Real)
    a = deg2rad(view_azimuth_elevation_deg[1]); e = deg2rad(view_azimuth_elevation_deg[2]); zr = offset_enlargement * 2 * lim
    wide = S * (cos(a) + sin(a)); high = S * (sin(a) + cos(a)) * sin(e) + zr * cos(e)
    return rad2deg(atan(S * tan(e), zr)), high / wide
end

"""
    bwo_surface(path, s, model, o, nodes, nb, band, lim, margin; asset_dir, mesh_stride, view_azimuth_elevation_deg, offset_enlargement, generator)

One surface: the planar board of set `s` plus the board's offsets learned with its lens (`o`, `bwo_offsets`), as a PGFPlots surface
in mm of page — the point of the page at (x, y) drawn at (x + E dx, y + E dy) with its height as z, E = `offset_enlargement`.
Every `mesh_stride`-th sample is a mesh point; the patches are coloured in `nb` bands of `band` mm on each side of zero; the dots
are the warping's `nodes`, displaced the same way. The box is the page widened by `margin` mm at all four sides and
−`lim` … +`lim` mm, the same in every panel; 12 pt at the panel's left hold the tick labels and their unit.
"""
function bwo_surface(path::AbstractString, s::AbstractString, model::AbstractString, o, nodes, nb::Integer, band::Real, lim::Real, margin::Real;
                     asset_dir::AbstractString, mesh_stride::Integer, view_azimuth_elevation_deg, offset_enlargement::Real, generator::AbstractString)
    W, H, z = o.W, o.H, o.z; E = offset_enlargement; azimuth, elevation = view_azimuth_elevation_deg
    n = size(z, 1); idx = collect(0:mesh_stride:n-1); idx[end] == n - 1 || push!(idx, n - 1)   # 0-based samples; the last one closes the mesh
    abs(W - H) < 1e-9 || error("the page of set $s is not square: the box must be (`bwo_projection`)")
    X = [i / (n - 1) * W + E * o.dx[j+1, i+1] for j in 0:n-1, i in 0:n-1]; Y = [(1 - j / (n - 1)) * H + E * o.dy[j+1, i+1] for j in 0:n-1, i in 0:n-1]
    minimum(X) >= -margin && maximum(X) <= W + margin && minimum(Y) >= -margin && maximum(Y) <= H + margin && maximum(abs, z) <= lim || error("set $s: a mesh point outside the box")
    # rows of constant b2, from the page's bottom edge
    rows = [join(("($(fixed(X[j+1, i+1], 2)),$(fixed(Y[j+1, i+1], 2)),$(fixed(z[j+1, i+1], 3)))" for i in idx), " ") for j in reverse(idx)]
    dots = join((begin
                     x, y = bwo_print_error(q[1], q[2], W, H, o.a, o.u)
                     "($(fixed(q[1] * W + E * x, 2)),$(fixed((1 - q[2]) * H + E * y, 2)),$(fixed(q[4], 3)))"
                 end for q in nodes), " ")
    elev, ratio = bwo_projection(W + 2 * margin, lim; view_azimuth_elevation_deg, offset_enlargement)
    cm = join(("rgb255=($(round(Int, c[1])),$(round(Int, c[2])),$(round(Int, c[3])))" for c in bwo_height_colours(2 * nb)), " ")
    # three ticks: 0 and about two thirds of the limit
    zt = lim >= 1.5 ? [t for t in (-2 * floor(lim / 3 + 0.5), 0.0, 2 * floor(lim / 3 + 0.5)) if abs(t) < lim] : [-lim / 2, 0.0, lim / 2]
    g = bw_g
    return write_lines(path, [
        "% $asset_dir/$s.tex — generated by $generator; do not edit.",
        "% One 3-D panel: the planar board of set $s ($(fixed(W, 0)) x $(fixed(H, 0)) mm; x to the right, y up the page, z towards the camera) PLUS the offset",
        "% vector that the lens's calibration gives every point ($model camera, refine product $(basename(o.product))): in the plane the print",
        "% error Aspect(a = $(g(o.a))) o Skew(u = $(g(o.u))), along the normal the height. The whole vector is drawn $(g(E)) times as large as lengths on",
        "% the page: the point of the page at (x, y) mm is at (x + $(g(E)) dx, y + $(g(E)) dy), its z is the height in mm. $(length(idx)) x $(length(idx)) mesh points; every",
        "% patch has the colour of its height's band ($(g(band)) mm); the dots are the warping's 5 x 5 nodes, displaced the same way.",
        "% The picture is the orthographic view from azimuth $(g(azimuth)), elevation $(g(elevation)) degrees with the heights $(g(E)) times as large as lengths on the",
        "% page: PGFPlots stretches the projected unit cube of the axis ranges to width x height, which is that view when the x and y",
        "% ranges are equal, `view`'s elevation is atan(range tan $(g(elevation)) / ($(g(E)) * height range)) = $(fixed(elev, 4)) degrees and height / width is the",
        "% projection's. The box: the page widened by $(g(margin)) mm ($(g(margin / E)) mm of offset) at all four sides, -$(g(lim)) ... +$(g(lim)) mm; the same margin and",
        "% heights in every panel. Its width is \\rcpanelwidth, of which 12 pt at the left hold the tick labels and their unit.",
        "\\providecommand{\\rcfigdir}{src/fig}%", "\\providecommand{\\rcpanelwidth}{\\linewidth}%",
        "\\begin{tikzpicture}[baseline=(current axis.center)]",
        "\\pgfmathsetlengthmacro{\\rcwarpw}{\\rcpanelwidth-12pt}%",
        "\\begin{axis}[",
        "  scale only axis, width=\\rcwarpw, height=$(fixed(ratio, 5))*\\rcwarpw, view={$(g(azimuth))}{$(fixed(elev, 4))},",
        "  xmin=$(g(-margin)), xmax=$(fixed(W + margin, 1)), ymin=$(g(-margin)), ymax=$(fixed(H + margin, 1)), zmin=$(g(-lim)), zmax=$(g(lim)), clip=false,",
        "  xtick=\\empty, ytick=\\empty, ztick={" * join(g.(zt), ",") * "}, zticklabel style={font=\\tiny, inner sep=1pt},",
        "  tick style={line width=0.2pt, black!60}, major tick length=1.5pt, zmajorgrids, grid style={line width=0.15pt, black!22},",
        "  axis line style={line width=0.2pt, black!60},",
        "  colormap={warp}{$cm}, colormap access=piecewise const, point meta min=$(g(-lim)), point meta max=$(g(lim))]",
        "\\addplot3[surf, shader=faceted, faceted color=black!45, line width=0.08pt, mesh/rows=$(length(idx)), z buffer=sort, point meta=z] coordinates {",
        rows..., "};",
        "\\addplot3[only marks, mark=*, mark size=0.4pt, black] coordinates {" * dots * "};",
        "\\node[anchor=south east, font=\\tiny, inner sep=1pt] at (axis cs:$(g(-margin)),$(g(-margin)),$(g(lim))) {mm};   % the unit of the height's ticks, over their labels",
        "\\end{axis}",
        "\\pgfresetboundingbox\\path ([xshift=-12pt]current axis.south west) rectangle (current axis.north east);   % the box is the axis and the strip of the tick labels",
        "\\end{tikzpicture}%"])
end

"""
    bwo_map(dir, q, model, products, nodes, colours, band; asset_dir, map_pixels_per_sample, generator) -> paths

One map: the page with the length of the difference `q` (`bwo_differences`) at every sample as an image, `<a>-minus-<b>.png` —
every sample `map_pixels_per_sample` × `map_pixels_per_sample` pixels in the colour of its band of `band` mm (`colours`, from
zero; no interpolation) — and its PGFPlots picture over the page in mm with the warping's `nodes` of the first set as dots.
`products` are the refine products of the two sets.
"""
function bwo_map(dir::AbstractString, q, model::AbstractString, products, nodes, colours, band::Real; asset_dir::AbstractString,
                 map_pixels_per_sample::Integer, generator::AbstractString)
    name = "$(q.a)-minus-$(q.b)"; W, H = q.W, q.H; up = map_pixels_per_sample; n, m = size(q.d3)
    img = Array{UInt8}(undef, up * n, up * m, 3)
    for j in 1:n, i in 1:m
        c = colours[Int(clamp(floor(q.d3[j, i] / band), 0, length(colours) - 1))+1]
        for k in 1:3
            img[up*(j-1)+1:up*j, up*(i-1)+1:up*i, k] .= round(UInt8, c[k])
        end
    end
    return [save_png(joinpath(dir, "$name.png"), img),
            write_lines(joinpath(dir, "$name.tex"), [
                "% $asset_dir/$name.tex — generated by $generator; do not edit.",
                "% One panel: the page ($(fixed(W, 0)) x $(fixed(H, 0)) mm, the axis in mm, y up) with the length of the offset vector of set $(q.a) minus that of set $(q.b) (in the plane and along the normal; nothing removed),",
                "% as an image in bands of $(bw_g(band)) mm ($model camera; refine products $(basename(products[1])) and $(basename(products[2])));",
                "% RMS $(fixed(q.rms, 4)) mm, largest value $(fixed(q.most, 4)) mm. The dots are the warping's 5 x 5 nodes on the planar board.",
                "% Its width is \\rcpanelwidth (\\linewidth when the host does not define it); \\rcfigdir is the figure's folder as the host document sees it.",
                "\\providecommand{\\rcfigdir}{src/fig}%", "\\providecommand{\\rcpanelwidth}{\\linewidth}%",
                "\\begin{tikzpicture}[baseline=(current axis.center)]", "\\begin{axis}[",
                "  scale only axis, width=\\rcpanelwidth, height=$(fixed(H / W, 5))*\\rcpanelwidth,", "  " * HOUSE_AXIS_COMMON * ",",
                "  xmin=0, xmax=$(fixed(W, 3)), ymin=0, ymax=$(fixed(H, 3)), enlargelimits=false, clip=false,",
                "  xtick=\\empty, ytick=\\empty, axis on top, axis line style={line width=0.2pt}]",
                "\\addplot graphics[xmin=0, xmax=$(fixed(W, 3)), ymin=0, ymax=$(fixed(H, 3))] {\\rcfigdir/$asset_dir/$name.png};",
                "\\addplot[only marks, mark=*, mark size=0.45pt, black] coordinates {" * join(("($(fixed(p[1] * W, 2)),$(fixed((1 - p[2]) * H, 2)))" for p in nodes), " ") * "};",
                "\\end{axis}",
                "\\pgfresetboundingbox\\path (current axis.south west) rectangle (current axis.north east);   % the box is the page",
                "\\end{tikzpicture}%"])]
end

"""
    bwo_figure(data, out; asset_dir, model, rows, columns, lenses, pairs, band_heights_mm, max_bands, mesh_stride, view_azimuth_elevation_deg,
               offset_enlargement, box_margin_rounding_mm, map_band_heights_mm, map_max_bands, map_width_share, map_pixels_per_sample, generator) -> paths

Draws the figure `asset_dir` from the refinements of the camera model `model` into `out`: `<asset_dir>.tex` (\\input it inside a
figure*; it follows \\linewidth and needs pgfplots, compat 1.18), in `<asset_dir>/` one surface per set (`<set>.tex`), one map per
board and pair of lenses (`<a>-minus-<b>.tex` with its image), the surfaces' key `key.tex` and the maps' key `maps-key.tex`,
`<asset_dir>-caption.tex` and source.toml. In each row the surfaces come first and the maps follow; a map carries its header (first
row) and its RMS and largest value itself.

Every keyword is required, the caller states what is drawn:
  - `rows`, `columns`: per row (a board) its label and its sets, one per column; the columns' headers (LaTeX text);
  - `lenses`: the columns' lenses in running text (the caption, the files' comments);
  - `pairs`: the maps, left to right — per map the two columns (the first lens minus the second) and the two lines of its header;
  - `band_heights_mm`, `max_bands`: the candidates of a height band, ascending — the first that needs at most `max_bands` bands on
    each side of zero for the largest |height| of all sets is taken, the last when none does;
  - `map_band_heights_mm`, `map_max_bands`: the same for the maps' bands from zero, for the largest value of all maps;
  - `view_azimuth_elevation_deg`, `offset_enlargement`: the orthographic view of the surfaces, and how many times as large as
    lengths on the page the offsets are drawn;
  - `box_margin_rounding_mm`: every surface's box is its page widened at all four sides by the largest in-plane offset (along x
    or y) of all sets, rounded up to a multiple of this, as it is drawn;
  - `map_width_share`: a map's width in units of a surface's;
  - `mesh_stride`, `map_pixels_per_sample`: every `mesh_stride`-th sample is a mesh point of a surface; a sample of a map is that
    many pixels wide and high;
  - `generator`: what the files' first lines name as their generator.
"""
function bwo_figure(data::AbstractString, out::AbstractString; asset_dir::AbstractString, model::AbstractString, rows, columns, lenses, pairs, band_heights_mm,
                    max_bands::Integer, mesh_stride::Integer, view_azimuth_elevation_deg, offset_enlargement::Real, box_margin_rounding_mm::Real, map_band_heights_mm,
                    map_max_bands::Integer, map_width_share::Real, map_pixels_per_sample::Integer, generator::AbstractString)
    all(length(ss) == length(columns) for (_, ss) in rows) && length(lenses) == length(columns) || error("rows, lenses: one set and one lens per column ($(length(columns)))")
    g = bw_g; E = offset_enlargement; dir = joinpath(out, asset_dir); dump = joinpath(data, "dump"); written = String[]
    O = bwo_offsets(data); sets = [s for (_, ss) in rows for s in ss]
    all(s -> haskey(O, s) && O[s].case == "$(s)_$(model)_warped", sets) || error("rows: the data folder has no case <set>_$(model)_warped of every set ($(join(sets, ", ")))")
    D = bwo_differences(O, rows, pairs)

    # the surfaces and their key: ONE scale of the heights and ONE margin for all panels
    top = maximum(maximum(abs, O[s].z) for s in sets)
    k = findfirst(b -> ceil(top / b) <= max_bands, band_heights_mm); band = band_heights_mm[something(k, length(band_heights_mm))]
    nb = Int(ceil(top / band - 1e-9)); lim = nb * band
    reach = maximum(max(maximum(abs, O[s].dx), maximum(abs, O[s].dy)) for s in sets)        # mm: the largest in-plane offset along x or y
    margin = E * (ceil(reach / box_margin_rounding_mm - 1e-9) * box_margin_rounding_mm)       # mm of page: that offset, rounded up, as it is drawn
    for s in sets
        push!(written, bwo_surface(joinpath(dir, "$s.tex"), s, model, O[s], bw_nodes(dump, O[s].case), nb, band, lim, margin;
                                   asset_dir, mesh_stride, view_azimuth_elevation_deg, offset_enlargement, generator))
    end
    push!(written, bwo_key(joinpath(dir, "key.tex"), "$asset_dir/key.tex", bwo_height_colours(2 * nb), -nb, band, true; generator))

    # the maps and their key: ONE scale from zero for all maps
    dtop = maximum(q.most for q in D)
    k = findfirst(b -> ceil(dtop / b - 1e-9) <= map_max_bands, map_band_heights_mm); dband = map_band_heights_mm[something(k, length(map_band_heights_mm))]
    dnb = Int(ceil(dtop / dband - 1e-9)); colours = bwo_map_colours(dnb)
    for q in D
        append!(written, bwo_map(dir, q, model, (O[q.a].product, O[q.b].product), bw_nodes(dump, O[q.a].case), colours, dband; asset_dir, map_pixels_per_sample, generator))
    end
    push!(written, bwo_key(joinpath(dir, "maps-key.tex"), "$asset_dir/maps-key.tex", colours, 0, dband, false; generator))

    # the frame: per board its surfaces, then its maps; the two keys under them
    nc = length(columns); np = length(pairs)
    gaps = vcat(fill(4, nc - 1), 10, fill(8, np - 1))      # pt between two surfaces, between the two blocks, between two maps
    free = 3                                               # pt kept free after the last map: a line of its text is wider than the map
    plain(t) = replace(t, "\\," => " ", "\$-\$" => "-")    # LaTeX text as a comment line names it
    stat(q) = "\\rcmapstat{RMS $(fixed(q.rms, 2))\\,mm\\\\max $(fixed(q.most, 2))\\,mm}"
    L = [
        "% $asset_dir.tex — generated by $generator; do not edit.",
        "% The three-lens session's board warping in one figure: per board (row) the planar board plus the offset vector that each lens's",
        "% calibration gives its points (in the plane the print error, along the normal the height; drawn $(g(E)) times enlarged) as a 3-D surface",
        "% ($(join(plain.(lenses), ", "))) and, after the three surfaces, per pair of lenses a map over the page of the length of the difference of",
        "% the two lenses' offset vectors (whole offset, 3-D), as it is: nothing fitted or removed. \\input it inside a figure* (it follows \\linewidth; pgfplots, compat 1.18).",
        "% The maps, left to right: $(join((plain(h[1]) * " " * plain(h[2]) for (_, h) in pairs), ", ")): all three pairs are shown.",
        "% \\rcfigdir is the folder this file lies in, as the host document sees it (default src/fig).",
        "\\providecommand{\\rcfigdir}{src/fig}%",
        "\\begingroup",
        "\\ifdefined\\rcpanelwd\\else\\newlength\\rcpanelwd\\fi",
        "\\ifdefined\\rcdiffwd\\else\\newlength\\rcdiffwd\\fi",
        "% three surfaces of width w, then three maps of width $(g(map_width_share)) w; gaps of 4 pt between surfaces, 10 pt between the two blocks, 8 pt between maps,",
        "% $free pt free after the last map (a line of its text is wider than the map); the row labels 12 pt",
        "\\setlength{\\rcpanelwd}{\\dimexpr(\\linewidth-$(12 + sum(gaps) + free)pt)*100/$(round(Int, 100 * (nc + map_width_share * np)))\\relax}\\setlength{\\rcdiffwd}{$(g(map_width_share))\\rcpanelwd}\\def\\rcpanelwidth{\\rcpanelwd}%",
        "\\footnotesize\\setlength{\\tabcolsep}{0pt}\\renewcommand{\\arraystretch}{1}%",
        "\\newcommand{\\rcdiffpanel}[1]{{\\def\\rcpanelwidth{\\rcdiffwd}\\input{\\rcfigdir/$asset_dir/#1}}}%",
        "% a map's header: directly over the map (its last line's baseline 3 pt above it), \\tiny; it does not add to the row's height (the header row is empty over the maps)",
        "\\newcommand{\\rcmaphead}[1]{\\llap{\\smash{\\raisebox{\\dimexpr0.5\\rcdiffwd+3pt\\relax}{\\makebox[\\rcdiffwd]{\\tiny\\begin{tabular}[b]{@{}c@{}}#1\\end{tabular}}}}}}%",
        "% a map's two lines of numbers: directly under the map, \\tiny, centred on it",
        "\\newcommand{\\rcmapstat}[1]{\\llap{\\raisebox{\\dimexpr-0.5\\rcdiffwd-1pt-\\height\\relax}{\\makebox[\\rcdiffwd]{\\tiny\\begin{tabular}[t]{@{}c@{}}#1\\end{tabular}}}}}%",
        "\\begin{tabular}{@{}p{9pt}@{\\hspace{3pt}}c" * join(("@{\\hspace{$(x)pt}}c" for x in gaps)) * "@{\\hspace{$(free)pt}}}",
        "  & " * join(vcat(["\\makebox[0pt]{\\scriptsize $c}" for c in columns], fill("", np)), " & ") * " \\\\\\noalign{\\vskip 2pt}"]
    for (i, (r, ss)) in enumerate(rows)
        maps = ["\\rcdiffpanel{$(q.a)-minus-$(q.b)}" * (i == 1 ? "\\rcmaphead{$(h[1])\\\\$(h[2])}" : "") * stat(q) for (q, (_, h)) in zip(D[(i-1)*np+1:i*np], pairs)]
        push!(L, "  \\makebox[9pt][c]{\\rotatebox[origin=c]{90}{\\scriptsize $r}} & " * join(vcat(["\\input{\\rcfigdir/$asset_dir/$s}" for s in ss], maps), " & ") *
                 " \\\\\\noalign{\\vskip $(i < length(rows) ? 5 : 6)pt}")
    end
    push!(L, "  \\multicolumn{$(nc + np + 1)}{@{}c@{}}{%",
          "    \\begin{minipage}[t]{0.49\\linewidth}\\centering\\input{\\rcfigdir/$asset_dir/key}\\\\[1pt]{\\scriptsize surfaces: the planar board plus the board offsets learned with this lens " *
          "(in the plane and along the normal), drawn $(g(E)) times enlarged; colour: height above the board's plane (positive: towards the camera)}\\end{minipage}\\hfill",
          "    \\begin{minipage}[t]{0.49\\linewidth}\\centering\\input{\\rcfigdir/$asset_dir/maps-key}\\\\[1pt]{\\scriptsize maps: length of the difference between the board offsets learned with two lenses " *
          "(in the plane and along the normal); below: RMS and largest value}\\end{minipage}} \\\\",
          "\\end{tabular}%", "\\endgroup")
    push!(written, write_lines(joinpath(out, "$asset_dir.tex"), L))

    # the caption: the paper's
    push!(written, write_lines(joinpath(out, "$asset_dir-caption.tex"), [
        "The two calibration boards (rows) as the calibration of each of three lenses reconstructs them, from the photographs of one session. " *
        "\\emph{Surfaces:} every point of the planar board moved by the offset learned for it in the calibration with that lens, in the plane by the print error " *
        "(aspect and skew) and along the normal by the height; the offsets are drawn $(g(E)) times enlarged, the colour is the height in bands of $(g(band))\\,mm, " *
        "and the dots are the \$5\\times5\$ nodes of the height field. " *
        "\\emph{Maps:} the length of the difference between the offsets learned with the two lenses (in the plane and along the normal together) at every point of the " *
        "page. Under each map: the RMS and the largest value of that length in mm. " *
        "In each row the three surfaces come first and the three maps follow, one for each pair of lenses (" *
        join(("$(lenses[f]) minus $(lenses[s])" for ((f, s), _) in pairs), ", ") * ")."]))

    # what was drawn, and how large the drawn offsets are
    R = ["# what $asset_dir was drawn from; written by $generator", "model = \"$model\"", "offset_enlargement = $(g(E))",
         "view_azimuth_elevation_deg = [$(g(view_azimuth_elevation_deg[1])), $(g(view_azimuth_elevation_deg[2]))]", "mesh_stride = $mesh_stride",
         "height_band_mm = $(g(band))", "height_limit_mm = $(g(lim))", "box_margin_mm = $(g(margin))", "map_width_share = $(g(map_width_share))",
         "map_band_mm = $(g(dband))", "map_range_mm = [0, $(g(dnb * dband))]", "removed = \"nothing: no best-fit plane, no rigid motion\"", ""]
    for s in sets
        o = O[s]; elev, _ = bwo_projection(o.W + 2 * margin, lim; view_azimuth_elevation_deg, offset_enlargement)
        push!(R, "[$s]", "refine_product = \"$(o.product)\"", "page_mm = [$(fixed(o.W, 3)), $(fixed(o.H, 3))]", "inplane_aspect = $(g(o.a))", "inplane_skew = $(g(o.u))",
              "inplane_offset_x_mm = [$(fixed(minimum(o.dx), 4)), $(fixed(maximum(o.dx), 4))]", "inplane_offset_y_mm = [$(fixed(minimum(o.dy), 4)), $(fixed(maximum(o.dy), 4))]",
              "inplane_offset_largest_mm = $(fixed(maximum(hypot.(o.dx, o.dy)), 4))", "height_mm = [$(fixed(minimum(o.z), 4)), $(fixed(maximum(o.z), 4))]",
              "pgfplots_view_elevation_deg = $(fixed(elev, 4))", "")
    end
    for q in D
        push!(R, "\"$(q.a) - $(q.b)\" = { height_rms_mm = $(fixed(q.normal, 4)), height_max_abs_mm = $(fixed(q.normal_most, 4)), offset_rms_mm = $(fixed(q.rms, 4)), " *
                 "offset_max_mm = $(fixed(q.most, 4)), inplane_rms_mm = $(fixed(q.plane, 4)) }")
    end
    push!(written, write_lines(joinpath(out, "source.toml"), R))
    println("-> $out/$asset_dir.tex + $asset_dir/ ($model: heights ±$(g(lim)) mm in bands of $(g(band)) mm; box = page + $(g(margin)) mm at every side; " *
            "maps 0 … $(g(dnb * dband)) mm in $dnb bands of $(g(dband)) mm, largest value $(fixed(dtop, 3)) mm)")
    for q in D
        println("   ", rpad(q.row, 14), " $(q.a) − $(q.b): whole offset RMS $(fixed(q.rms, 3)), largest $(fixed(q.most, 3)) mm; heights RMS $(fixed(q.normal, 3)), in the plane RMS $(fixed(q.plane, 3)) mm")
    end
    return written
end

"""
    figure_board_warping_offsets(; data, out, part, generator, settings...) -> paths

The paper's board-warping figure from the data folder `data` into `out`. `part`:

  - `figure3d`: the drawing — per board its surfaces and its maps, the two keys, the caption and source.toml. `settings` are what
    is drawn, every one required (`bwo_figure`'s keywords).
  - `tables`: `out`/meta.toml — the three-lens figure's tables and, per board and pair of lenses, the difference between the
    board's offsets learned with the two lenses (no drawing: of the settings it takes `rows` and `pairs`).

`generator`: what the written files name as their generator (by default this file; the stage `figures` of scripts/board_warping.jl
names itself).
"""
function figure_board_warping_offsets(; data::AbstractString, out::AbstractString, part::AbstractString,
                                      generator::AbstractString = "experiments/src/paper_figures/board_warping_offsets.jl", settings...)
    part in ("figure3d", "tables") || error("part takes \"figure3d\" or \"tables\"")
    part == "tables" && return bwo_tables(data, out; generator, settings...)
    return bwo_figure(data, out; generator, settings...)
end

FIGURES["board-warping-offsets"] = figure_board_warping_offsets
