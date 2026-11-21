# focus_surfaces.jl — the paper's focus-surface figure (fig:focus-surfaces: src/fig/focus-surfaces.tex, its panels
# focus-surfaces/<lens>.tex and focus-surfaces-caption.tex) as PGFPlots / TikZ; the port of the figure folder's
# tools/focus_surfaces.py (its two parts, `tables` and `figure`). Every panel is its own PGFPlots picture (a 3-D axis); a LaTeX
# tabular puts them side by side, and its titles are text of the host document. What the figure shows is the caller's to say (the
# figure's config of record, configs/experiments/focus_surfaces.toml [panels]): the figure's name, the lattice's spacing and, per
# panel, its file name, case, title, view and the spacing of its drawn lattice lines.
#
#   PaperFigures.generate("focus-surfaces"; data = <tables>, out = <dir>, asset_dir = <the figure's name>)
#   PaperFigures.generate("focus-surfaces"; part = "tables", data = <dumps>, out = <tables>, spacing = <px>, panels = <the panels>)
#
#   panel     the in-focus surface a refinement learned, in the camera's coordinates (m): every pixel's in-focus point on its ray,
#             the camera centre (the black dot) at the origin, z along the optical axis. The three axes have ONE scale, so the
#             surface's shape is true.
#   surface   the cells between lines of the pixel lattice (per lens a line every `every` lattice samples, and the sensor's
#             border), each a polygon of the surface's own points, filled in the colour of its focus distance with its outline: the
#             outlines are the lattice lines. A cell's focus distance is the mean over its polygon's vertices; the colour runs from
#             the panel's nearest cell (a light blue) to its farthest (a mid-dark blue), one hue, so the black lines show on all. A
#             cell the image circle cuts ends at the circle. Along a line a polygon has as many vertices as it needs to stay within
#             0.05 pt (for a 100 pt box) of the surface. The cells are drawn the farthest first for the panel's view (and the camera
#             in its place between them), so nearer cells cover farther ones.
#   view      per lens (meta.toml; PGFPlots' azimuth and elevation), one from the front left above. A panel states the view as the
#             axis' three unit vectors (x=, y=, z=), computed here such that the axis box is exactly \rcpanelwidth wide; the cells'
#             order belongs to them.
#   axes      the box of the axis from z = 0 (the camera's plane), symmetric in x and y about the camera. Numbers on three edges only
#             (x: the floor's front edge, y: its right edge, z: the left edge), with tick marks drawn here, square to their edge, on
#             these edges alone (PGFPlots would repeat them on the opposite edges). The x and y axes are named beside their numbers,
#             the z axis over its upper end, where the box leaves the page free.
#   figure    the panels side by side, their axis boxes at ONE height (so their widths differ), a title over each box and under
#             each its key: the colour strip between the focus distances of the panel's nearest and farthest cell. Under the keys,
#             once for all, the name of what the colour shows.
#
# The part `tables` reads from <dumps>, per lens, <case>.f64: a refined camera's in-focus points on the pixel lattice (two
# Int64 nx, ny; then X, Y, Z and the focus distance as Float64, nx × ny each, column-major, little-endian, NaN where the lens images
# nothing). It writes the tables the part `figure` reads:
#   meta.toml     spacing_px (px between lattice samples), order (the lenses, left to right) and per lens [lenses.<lens>]: label,
#                 case, nx, ny (the lattice), every (a drawn line every … samples), view (azimuth, elevation in degrees), imaged (the
#                 samples the lens images), focus_min, focus_max, focus_median (the focus distance over all of them, m)
#   <lens>.tsv    i j x y z: the lattice sample (i, j), 0-based, at the pixel ((i + ½) spacing_px, (j + ½) spacing_px), and its
#                 in-focus point (m) — the samples on the drawn lattice lines only, every one of them along a line, so that the
#                 place where the image circle cuts a line is known to one sample
#
# The numbers are computed in the order NumPy computed them, so that the files come out the same.

"The drawn lattice lines of `n` samples, 0-based: every `every` samples, and the sensor's border."
focus_lines(n::Integer, every::Integer) = sort!(unique!([collect(every÷2:every:n-1); 0; n - 1]))
"The xcolor colour at the share `t` of the way from the nearest cell's fill to the farthest's."
function focus_shade(t::Real)
    near, far = (234, 242, 251), (43, 108, 176)     # a cell's fill: the panel's nearest cell (a light blue, EAF2FB) … its farthest (a mid-dark blue, 2B6CB0); black lines stay visible on both
    return "{rgb,255:" * join(("$c,$(round(Int, a + (b - a) * clamp(t, 0.0, 1.0)))" for (c, a, b) in zip(("red", "green", "blue"), near, far)), ";") * "}"
end
"A lens's name as a title: \"Tamron 20 mm lens, f/8\" gives \"Tamron 20\\,mm lens, \$f/8\$\"."
focus_title(s::AbstractString) = ((a, b) = split(s, ", f/"); replace(a, " mm" => "\\,mm") * ", \$f/$b\$")
"A title inside a sentence: \"fisheye lens\"."
focus_insent(t::AbstractString) = startswith(t, "Fisheye") ? lowercasefirst(t) : t
"PGFPlots' view={az}{el} as the page's right and up directions and the direction towards the viewer (camera coordinates)."
function focus_basis(az::Real, el::Real)
    a = deg2rad(az); e = deg2rad(el)
    return (cos(a), sin(a), 0.0), (-sin(a) * sin(e), cos(a) * sin(e), cos(e)), (sin(a) * cos(e), -cos(a) * cos(e), sin(e))
end
focus_dot(p, v) = p[1] * v[1] + p[2] * v[2] + p[3] * v[3]
"The mean of `a`, its sum taken in NumPy's order (common.jl npsum)."
focus_mean(a) = npsum(a) / length(a)
"A table of numbers, a matrix with one row per line; what stands behind a # is a comment."
function focus_table(path::AbstractString)
    rows = [parse.(Float64, split(l)) for l in (first(split(l, '#')) for l in eachline(path)) if !isempty(strip(l))]
    return permutedims(reduce(hcat, rows))
end
focus_g(v) = general(v, 6)
focus_f4(v) = isnan(v) ? "nan" : fixed(v, 4)        # Python prints a NaN in lower case

"""
    focus_tables(dump, tdir; spacing, panels, generator) -> Vector{String}

The part `tables`: writes the figure's tables into `tdir` from the surfaces' dumps `dump`/<case>.f64 — per lens <lens>.tsv, the
samples on the drawn lattice lines, and meta.toml, which names `generator` as its writer. `spacing` is the px between the dumps'
lattice samples. `panels` are the figure's panels, left to right, each `(; lens, case, label, azimuth_deg, elevation_deg,
grid_every)`: the panel's file name, the refinement's case, the panel's title, its view (PGFPlots' azimuth and elevation) and the
spacing of its drawn lattice lines, one every `grid_every` samples. Returns the files written.
"""
function focus_tables(dump::AbstractString, tdir::AbstractString; spacing::Integer, panels, generator::AbstractString)
    M = ["# The numbers of fig:focus-surfaces ($generator): the in-focus surfaces of two refined cameras on the pixel lattice — per lens",
         "# <lens>.tsv (i j x y z: lattice sample (i, j), 0-based, at the pixel ((i + 1/2) spacing_px, (j + 1/2) spacing_px); its in-focus point in the",
         "# camera's coordinates, m), the samples on the drawn lattice lines only (per lens a line every `every` samples). focus_* : the focus distance over ALL lattice samples the lens images (m).",
         "spacing_px = $spacing", "order = [" * join(("\"$(l.lens)\"" for l in panels), ", ") * "]"]
    written = String[]
    for l in panels
        path = joinpath(dump, l.case * ".f64")
        X, Y, Z, D = open(path) do io
            nx = Int(ltoh(read(io, Int64))); ny = Int(ltoh(read(io, Int64)))
            A = [ltoh.(read!(io, Matrix{Float64}(undef, nx, ny))) for _ in 1:4]
            eof(io) || error("$path holds more than its four $nx × $ny arrays")
            A
        end
        nx, ny = size(D); ok = isfinite.(D)
        # a central camera: the point's distance from the origin is the focus distance
        off = filter(!isnan, [abs(sqrt(X[k]^2 + Y[k]^2 + Z[k]^2) - D[k]) for k in eachindex(D)])
        (any(ok) && !isempty(off) && maximum(off) < 1e-9) || error("$(l.case): the in-focus points do not lie at their focus distance from the camera centre")
        on = falses(nx, ny); on[focus_lines(nx, l.grid_every) .+ 1, :] .= true; on[:, focus_lines(ny, l.grid_every) .+ 1] .= true
        rows = ["$(i - 1)\t$(j - 1)\t$(focus_f4(X[i, j]))\t$(focus_f4(Y[i, j]))\t$(focus_f4(Z[i, j]))" for i in 1:nx for j in 1:ny if on[i, j] && ok[i, j]]
        push!(written, write_lines(joinpath(tdir, l.lens * ".tsv"), ["# i\tj\tx\ty\tz\t($(l.case): the in-focus points on the drawn lattice lines, m)"; rows]))
        d = D[ok]; lo, hi = extrema(d); med = Statistics.median(d)
        append!(M, ["", "[lenses.$(l.lens)]", "label = \"$(l.label)\"", "case = \"$(l.case)\"", "nx = $nx", "ny = $ny", "every = $(l.grid_every)",
                    "view = [$(focus_g(l.azimuth_deg)), $(focus_g(l.elevation_deg))]", "imaged = $(count(ok))",
                    "focus_min = $(fixed(lo, 4))", "focus_max = $(fixed(hi, 4))", "focus_median = $(fixed(med, 4))"])
        println("   $(l.label) ($(l.case)): $nx x $ny lattice samples of $spacing px, $(count(ok)) imaged, $(length(rows)) on the drawn lines (every $(l.grid_every * spacing) px); " *
                "focus distance $(fixed(lo, 3)) to $(fixed(hi, 3)) m (median $(fixed(med, 3)))")
    end
    push!(written, write_lines(joinpath(tdir, "meta.toml"), M)); println("-> $tdir/")
    return written
end

"""
    focus_deviation(X, Y, Z, right, up, sub, every) -> Float64

How far (m, on the page) a lattice line drawn with a vertex every `sub` samples leaves the line through all its samples: the largest
distance of a skipped sample from its chord, over the stretches between the drawn lines that the lens images whole.
"""
function focus_deviation(X, Y, Z, right, up, sub::Integer, every::Integer)
    nx, ny = size(X); ii = focus_lines(nx, every) .+ 1; jj = focus_lines(ny, every) .+ 1; worst = 0.0
    page(i, j) = (p = (X[i, j], Y[i, j], Z[i, j]); (focus_dot(p, right), focus_dot(p, up)))
    segs = [[[page(i, j) for i in ii[a]:ii[a + 1]] for a in 1:length(ii)-1 for j in jj];
            [[page(i, j) for j in jj[b]:jj[b + 1]] for b in 1:length(jj)-1 for i in ii]]
    for g in segs
        all(p -> isfinite(p[1]) && isfinite(p[2]), g) || continue
        n = length(g); knots = unique!([collect(1:sub:n); n])
        for (q0, q1) in zip(knots[1:end-1], knots[2:end])
            q1 > q0 + 1 || continue
            d = g[q1] .- g[q0]
            v = maximum(abs(d[1] * (g[m][2] - g[q0][2]) - d[2] * (g[m][1] - g[q0][1])) for m in q0+1:q1-1) / hypot(d[1], d[2])
            v > worst && (worst = v)
        end
    end
    return worst
end

"""
    focus_cells(X, Y, Z, sub, every) -> Vector{Vector{NTuple{3, Float64}}}

The surface's cells between the drawn lattice lines as polygons with a vertex every `sub` samples. A cell the image circle cuts is
the longest run of its border that the lens images, its two ends joined across the cell.
"""
function focus_cells(X, Y, Z, sub::Integer, every::Integer)
    nx, ny = size(X); ii = focus_lines(nx, every); jj = focus_lines(ny, every); polys = Vector{NTuple{3, Float64}}[]
    for a in 1:length(ii)-1, b in 1:length(jj)-1
        i0, i1, j0, j1 = ii[a], ii[a + 1], jj[b], jj[b + 1]
        # the cell's border, once around from (i0, j0), as 0-based lattice samples
        border = [[(i, j0) for i in i0:i1-1]; [(i1, j) for j in j0:j1-1]; [(i, j1) for i in i1:-1:i0+1]; [(i0, j) for j in j1:-1:j0+1]]
        v = [isfinite(X[i + 1, j + 1]) for (i, j) in border]; n = length(border)
        any(v) || continue
        run = collect(1:n)
        if !all(v)                                  # the runs of imaged samples around the border, from its first sample that is not imaged
            runs = Vector{Int}[]; s0 = findfirst(!, v); k = s0
            while k < s0 + n
                if v[mod1(k, n)]
                    r = Int[]
                    while k < s0 + n && v[mod1(k, n)]
                        push!(r, mod1(k, n)); k += 1
                    end
                    push!(runs, r)
                else
                    k += 1
                end
            end
            run = argmax(length, runs)              # the longest, the first of them among equals
        end
        corners = ((i0, j0), (i1, j0), (i1, j1), (i0, j1))
        keep = [q for (m, q) in enumerate(run) if m == 1 || m == length(run) || border[q] in corners ||
                                                  (mod(border[q][1] - i0, sub) == 0 && mod(border[q][2] - j0, sub) == 0)]
        length(keep) >= 3 && push!(polys, [(X[i + 1, j + 1], Y[i + 1, j + 1], Z[i + 1, j + 1]) for (i, j) in border[keep]])
    end
    return polys
end

"Round numbers about 0 within ±`lim`: the multiples of the smallest of `spacings` that gives at most `most` of them, and not the box's very ends."
function focus_ticks(lim::Real, most::Integer, spacings = (0.05, 0.1, 0.2, 0.25, 0.3, 0.5, 1.0))
    for s in spacings
        k = floor(Int, 0.93 * lim / s + 1e-9)
        (k >= 1 && 2 * k + 1 <= most) && return [q * s for q in -k:k]
    end
    return [0.0]
end

"""
    focus_figure(tdir, out; asset_dir, generator) -> Vector{String}

The part `figure`: draws the figure `asset_dir` from the tables in `tdir` into `out` — `asset_dir`/<lens>.tex per panel, `asset_dir`.tex and
`asset_dir`-caption.tex. `generator` is what the files' first lines name as their generator. Returns the files written.
"""
function focus_figure(tdir::AbstractString, out::AbstractString; asset_dir::AbstractString, generator::AbstractString)
    f2(v) = fixed(v, 2); f3(v) = fixed(v, 3); f4(v) = fixed(v, 4); g = focus_g; GEN = "generated by $generator; do not edit"
    V = TOML.parsefile(joinpath(tdir, "meta.toml")); spacing = V["spacing_px"]
    tick = 1.5                                      # pt: the length of a tick mark
    pad = 1.06                                      # the axis box: the surface's largest |x|, |y| and z times this
    P = []; written = String[]
    for slug in V["order"]
        L = V["lenses"][slug]; nx, ny, every = L["nx"], L["ny"], L["every"]; az, el = L["view"]
        (0 < az < 90 && 0 < el < 90) || error("$slug: the view ($az, $el) is not one from the front left above")
        right, up, viewer = focus_basis(az, el)
        T = focus_table(joinpath(tdir, slug * ".tsv")); X = fill(NaN, nx, ny); Y = fill(NaN, nx, ny); Z = fill(NaN, nx, ny)
        for r in axes(T, 1)
            i = trunc(Int, T[r, 1]) + 1; j = trunc(Int, T[r, 2]) + 1; X[i, j], Y[i, j], Z[i, j] = T[r, 3], T[r, 4], T[r, 5]
        end
        most(A) = maximum(filter(!isnan, A))
        xl, yl, zl = pad * most(abs.(X)), pad * most(abs.(Y)), pad * most(Z)
        box = [(sx * xl, sy * yl, z) for sx in (-1, 1) for sy in (-1, 1) for z in (0.0, zl)]     # the axis box's corners
        span(v) = (e = extrema(focus_dot(c, v) for c in box); e[2] - e[1])
        bw = span(right); bh = span(up)             # the box's width and height as the view projects it (m)
        # a polygon vertex every `sub` samples along a line: the largest of these that keeps the drawn line within 5e-4 × the box's
        # width of the surface (0.05 pt for a 100 pt box)
        sub = first(q for q in (14, 7, 2, 1) if focus_deviation(X, Y, Z, right, up, q, every) <= 5e-4 * bw)
        polys = focus_cells(X, Y, Z, sub, every)
        # a cell's focus distance: the mean over its vertices (a central camera: a point's distance from the origin)
        dist = [focus_mean([sqrt(q[1] * q[1] + q[2] * q[2] + q[3] * q[3]) for q in p]) for p in polys]; lo, hi = extrema(dist)
        hi > lo || error("$slug: all cells have one focus distance")
        # the farthest first; the camera (depth 0, number 0) in its place. A cell's depth: its vertices' mean, towards the viewer.
        centre(p) = ntuple(c -> foldl(+, (q[c] for q in p)) / length(p), 3)
        items = sort!([[(focus_dot(centre(p), viewer), k) for (k, p) in enumerate(polys)]; (0.0, 0)]; lt = (s, t) -> s[1] < t[1] || (s[1] == t[1] && s[2] < t[2]))
        body = [k == 0 ? "\\fill (axis cs:0,0,0) circle[radius=1.3pt];" :
                "\\path[rcfs, fill=$(focus_shade((dist[k] - lo) / (hi - lo)))] " * join(("(axis cs:$(f4(q[1])),$(f4(q[2])),$(f4(q[3])))" for q in polys[k]), " -- ") * " -- cycle;" for (_, k) in items]
        # The numbers stand on three edges of the box (where PGFPlots puts them for a view from the front left above: 0 < az, el < 90): x along the
        # front edge of the floor, y along its right edge, z up the left edge. The tick marks are drawn here, on these edges only, in the floor's
        # plane (x, y) and to the left (z).
        square(c) = (d = (right[c], up[c]); h = hypot(d...); (tick * d[2] / h, tick * -d[1] / h))   # square to an edge along the axis c on the page, to its right hand: outwards for these two edges
        xt = focus_ticks(xl, 3); yt = [q for q in focus_ticks(yl, 3) if q != 0]; zt = [q for q in focus_ticks(zl, 5, (0.05, 0.1, 0.2, 0.3, 0.5)) if q >= 0]
        tk(v) = join((g(q) for q in v), ",")
        mark(at, d) = "\\draw[rctk] (axis cs:$(f4(at[1])),$(f4(at[2])),$(f4(at[3]))) -- ++($(f3(d[1]))pt,$(f3(d[2]))pt);"
        marks = [[mark((q, -yl, 0.0), square(1)) for q in xt]; [mark((xl, q, 0.0), square(2)) for q in yt]; [mark((-xl, -yl, q), (-tick, 0.0)) for q in zt];
                 # the z axis' name: over its upper end, where the box leaves the page free
                 "\\node[font=\\footnotesize, anchor=south east, inner sep=0pt, yshift=2pt] at (axis cs:$(f4(-xl)),$(f4(-yl)),$(f4(zl))) {\$z\$ [m]};"]
        unit(c) = "($(fixed(right[c] / bw, 5))*\\rcpanelwidth,$(fixed(up[c] / bw, 5))*\\rcpanelwidth)"
        push!(written, write_lines(joinpath(out, asset_dir, slug * ".tex"), [
            "% $asset_dir/$slug.tex — $GEN.";
            "% One panel ($(L["label"]), case $(L["case"])): the in-focus points of the refined camera in its own coordinates (m; the camera, the black dot,";
            "% at the origin, z along the optical axis), as the $(length(polys)) cells between the pixel lattice's lines every $(every * spacing) px, each a polygon of the surface's own";
            "% points (a vertex every $(sub * spacing) px; a cell the image circle cuts ends at the circle), the farthest first for this view.";
            "% A cell's fill is its focus distance (the mean over its vertices), from the nearest cell's $(f3(lo)) m (light) to the farthest's $(f3(hi)) m (dark).";
            "% Focus distance over all pixels $(f3(L["focus_min"])) to $(f3(L["focus_max"])) m. The axis' unit vectors x, y, z are PGFPlots' view={$(g(az))}{$(g(el))} with ONE scale on the three axes, such that";
            "% the axis box is \\rcpanelwidth wide (\\linewidth when the host does not define it); do not change them by hand (the cells' order belongs to them).";
            "% The baseline is the box's centre, and the picture is trimmed to the box at the left and the right.";
            "\\providecommand{\\rcpanelwidth}{\\linewidth}%";
            "\\begin{tikzpicture}[baseline=(current axis.center), trim axis left, trim axis right,";
            "  rcfs/.style={draw=black, line width=0.2pt, line join=round}, rctk/.style={black, line width=0.3pt}]";
            "\\begin{axis}[";
            "  x={$(unit(1))}, y={$(unit(2))}, z={$(unit(3))},";
            "  " * HOUSE_AXIS_COMMON * ",";
            "  label style={font=\\footnotesize}, axis line style={line width=0.3pt}, tick align=outside, major tick length=$(g(tick))pt, tick style={draw=none}, scaled ticks=false, clip=false,";
            "  xmin=$(f4(-xl)), xmax=$(f4(xl)), ymin=$(f4(-yl)), ymax=$(f4(yl)), zmin=0, zmax=$(f4(zl)),";
            "  xtick={$(tk(xt))}, ytick={$(tk(yt))}, ztick={$(tk(zt))},";
            "  xlabel={\$x\$ [m]}, ylabel={\$y\$ [m]}, xlabel shift=-7pt, ylabel shift=-7pt]";   # the x and y axis names: closer to their numbers than PGFPlots puts them
            body; marks; "\\end{axis}"; "\\end{tikzpicture}%"]))
        push!(P, (; slug, L, ratio = bh / bw, cells = length(polys), lo, hi))
    end

    # the composition: the panels side by side at ONE height (their boxes' widths follow), the titles text of the host document
    wsum = foldl(+, (1 / q.ratio for q in P); init = 0); share = [round(Int, 1000 / q.ratio / wsum) for q in P]
    # the gaps (pt): left of the first box (the z axis' numbers and name), between the boxes (the y numbers of the first, the z
    # numbers and name of the second), right of the last (its y numbers)
    gap_left, gap_centre, gap_right = 22, 34, 20; gaps = gap_left + gap_centre + gap_right
    function key(lo, hi)                            # the key of a panel: the colour strip between the focus distances of its nearest and its farthest cell
        patches, width, height = 24, 2.0, 4.0       # the strip: its patches of one colour each, a patch's width and the strip's height (pt)
        strip = join("\\fill[fill=$(focus_shade((k + 0.5) / patches))] ($(g(k * width))pt,0) rectangle ($(g((k + 1) * width + 0.05))pt,$(g(height))pt);" for k in 0:patches-1)
        return "{\\scriptsize $(f2(lo))\\,m\\,\\tikz[baseline=0.5pt]{$strip\\draw[line width=0.2pt] (0,0) rectangle ($(g(patches * width))pt,$(g(height))pt);}\\,$(f2(hi))\\,m}"
    end
    push!(written, write_lines(joinpath(out, "$asset_dir.tex"), [
        "% $asset_dir.tex — $GEN.",
        "% The figure: its $(length(P)) panels (one PGFPlots picture each, a 3-D axis, $asset_dir/*.tex) side by side in a tabular, their axis boxes at one",
        "% height, under each its key (the cells' colour: focus distance, light = near, dark = far) and under the keys their name. The titles, the keys' numbers and their name are LaTeX text in the host document's fonts, and so are the axes' names and numbers inside the panels. \\input it inside a",
        "% figure (one column); it needs \\usepackage{pgfplots} and \\pgfplotsset{compat=1.18}. \\rcfigdir is the folder this file lies in, as the host",
        "% document sees it (default src/fig).",
        "\\providecommand{\\rcfigdir}{src/fig}%",
        "\\begingroup",
        "\\footnotesize\\setlength{\\tabcolsep}{0pt}%",
        "\\begin{tabular}{@{\\hspace{$(gap_left)pt}}" * join(("c" for _ in P), "@{\\hspace{$(gap_centre)pt}}") * "@{\\hspace{$(gap_right)pt}}}",
        "  " * join(("\\makebox[0pt]{$(focus_title(q.L["label"]))}" for q in P), " & ") * " \\\\\\noalign{\\vskip 3pt}",
        "  " * join(("\\def\\rcpanelwidth{\\dimexpr(\\linewidth-$(gaps)pt)*$s/1000\\relax}\\input{\\rcfigdir/$asset_dir/$(q.slug)}" for (q, s) in zip(P, share)), " &\n  ") * " \\\\\\noalign{\\vskip 3pt}",
        "  " * join(("\\makebox[0pt]{$(key(q.lo, q.hi))}" for q in P), " & ") * " \\\\\\noalign{\\vskip 1pt}",   # centred under its box without widening its column: a key is wider than the narrower box
        "  \\multicolumn{$(length(P))}{@{\\hspace{$(gap_left)pt}}c@{\\hspace{$(gap_right)pt}}}{\\scriptsize focus distance from the camera centre} \\\\",   # the keys' name, what the colour shows: once, centred under the boxes
        "\\end{tabular}%", "\\endgroup"]))
    titles = [focus_title(q.L["label"]) for q in P]; sp = [q.L["every"] * spacing for q in P]
    push!(written, write_lines(joinpath(out, "$asset_dir-caption.tex"), [
        "The in-focus surfaces two refinements learned, in camera coordinates (the camera centre is the black dot, \$z\$ runs along the optical axis; one scale on the three axes): " *
        "every pixel's in-focus point on its ray. The lines follow the pixel lattice, " *
        (allequal(sp) ? "every $(sp[1])\\,px" : join(("every $q\\,px ($(first(split(focus_insent(n), ", \$f/"))))" for (q, n) in zip(sp, titles)), " and ")) *
        "; a cell's colour is its mean focus distance, from the panel's nearest cell (light) to its farthest (dark), as the key under the panel states. " *
        "Over all pixels the focus distance is " * join(("$(f2(q.L["focus_min"])) to $(f2(q.L["focus_max"]))\\,m ($(focus_insent(n)))" for (q, n) in zip(P, titles)), " and ") * "."]))
    println("-> $out/$asset_dir.tex + $asset_dir/ ($(length(P)) panels)")
    for q in P
        println("   $(q.L["label"]): $(q.cells) cells (lines every $(q.L["every"] * spacing) px; cell means $(f3(q.lo)) to $(f3(q.hi)) m), view $(g(q.L["view"][1])) / $(g(q.L["view"][2])) deg, " *
                "axis box height / width $(f3(q.ratio)), focus distance $(f3(q.L["focus_min"])) to $(f3(q.L["focus_max"])) m")
    end
    return written
end

"""
    figure_focus_surfaces(; data, out, part = "figure", generator, settings...) -> Vector{String}

The focus-surface figure. `part = "figure"` draws it from the tables in `data` into `out`; `part = "tables"` writes these tables into
`out` from the surfaces' dumps in `data`. Prints the numbers of the part. Returns the files written.

`settings`: what the part needs to be told, every one required — `asset_dir` (the figure's name) for the part `figure`
(`focus_figure`), `spacing` and `panels` for the part `tables` (`focus_tables`).

`generator`: what the written files name as their generator (by default this file; the stage `panels`
of scripts/focus_surfaces.jl names itself).
"""
function figure_focus_surfaces(; data::AbstractString, out::AbstractString, part::AbstractString = "figure",
                               generator::AbstractString = "experiments/src/paper_figures/focus_surfaces.jl", settings...)
    part == "figure" && return focus_figure(data, out; generator, settings...)
    part == "tables" && return focus_tables(data, out; generator, settings...)
    error("part takes \"figure\" or \"tables\"; got \"$part\"")
end

FIGURES["focus-surfaces"] = figure_focus_surfaces
