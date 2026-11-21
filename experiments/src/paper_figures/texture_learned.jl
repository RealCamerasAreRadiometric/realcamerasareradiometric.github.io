# =============================================================================
# src/paper_figures/texture_learned.jl — the paper's learned-texture figure (fig:texture-learned: src/fig/texture-learned.tex, its
# bands texture-learned/<band>.{png,tex} and texture-learned-caption.tex): the first lines of the printed page's text, once from
# each source, one band each, stacked without a gap — in the paper three lines from four sources: the printed design (the ground
# truth), a training photograph, the texture optimised with a pinhole camera, the texture optimised with our aperture and focus
# surface. The port of T. Harder's texture_learned.py (the figure folder's tools/), step for step; every number is printed as
# Python printed it.
#
# What the figure shows is the caller's — a driver's config of record (scripts/texture_recovery.jl, `[bands]`): the figure's name,
# the bands (what each shows, its name in the margin, the caption's words for it), the page's text block, the text lines per band,
# the level a blank row is above and the pixels a texel is drawn with. This file states none of them.
#
# A band's `source` says what it shows: "design" — the printed design, the run's design.png (the ground truth; it is also the
# reference of the cuts and of the textures' errors, whether a band shows it or not); "photograph" — the training photograph's cut;
# any other word names a blur model of the run, and the band shows that model's texture (the run's texture_<model>.png).
#
# Two parts, as there:
#   tables   from a run of the stack's texture_recovery driver to the figure's tables (the data folder): meta.toml and one 8-bit
#            grey image per band, the textures and the design one pixel per texel over the region of the bands' text lines.
#   figure   from the tables to the figure: every band its own PGFPlots picture over its image (each texel drawn as a square of
#            pixels, not interpolated), composed by a LaTeX tabular with the bands' names in a margin at the left.
#
# One piece of `tables` is NOT ported and stays with texture_learned.py: the training photograph's band. It decodes the RAW file
# (rawpy), finds the page in it (OpenCV: SIFT matches, a RANSAC homography) and cuts the band out, turned level (OpenCV: bicubic).
# What it writes is an input here — photograph.png and, in the meta.toml it wrote, the photograph's file name and the numbers of
# that registration and cut ([bands.photograph] pixels, photo_box; [photograph] view, in_plane_deg, turned_deg,
# photo_px_per_texel, sift_matches, homography_inliers). Everything else of meta.toml is computed here.
#
# The 8-bit grey PNGs are read and written by common.jl's load_grey_png and save_grey_png.
# =============================================================================

# ---- tables ---------------------------------------------------------------------------------------------------------------------

"The page's text block `block_mm` (`[x0, y0, width, height]`, mm) in texels of `tex` mm: x0, y0, x1, y1 (0-based, first and past-last)."
function texlearned_block(block_mm, tex::Real)
    x0, y0, w, h = (round(Int, v / tex) for v in block_mm)
    return (x0, y0, x0 + w, y0 + h)
end

"""
    texlearned_bands(design, n, read_on; block, lines_per_band, white) -> (rows, top)

The rows of `n` bands in the text block `block` (texels: x0, y0, x1, y1) of the page `design`, each 0-based as first, past-last,
and the row the first band begins at. A row of the block lies in a gap between two text lines when the design's darkest texel in
it is lighter than `white` (design units 0 … 1); a band holds `lines_per_band` text lines — with `read_on` the lines after the
band above, without it the first ones in every band.
"""
function texlearned_bands(design::AbstractMatrix{UInt8}, n::Integer, read_on::Bool; block, lines_per_band::Integer, white::Real)
    x0, y0, x1, y1 = block; PER = lines_per_band
    # a row is blank when its darkest texel is lighter than `white` (single precision, as the Python compared)
    blank = [Float32(minimum(@view design[y0+r+1, x0+1:x1])) / 255.0f0 > Float32(white) for r in 0:y1-y0-1]
    edges = [e for e in 0:length(blank)-2 if blank[e+1] != blank[e+2]]                 # rows after which blank changes
    gaps = [(a + 1, b + 1) for (a, b) in zip(edges[1:end-1], edges[2:end]) if blank[a+2]]   # first, past-last row of each gap between two lines
    length(gaps) >= PER * (read_on ? n : 1) || error("the text block has only $(length(gaps)) gaps between text lines")
    mid(i) = (gaps[i][1] + gaps[i][2]) ÷ 2                                           # the middle of the gap after text line i (1-based)
    top = max(0, (findfirst(!, blank) - 1) - (gaps[1][2] - gaps[1][1]) ÷ 2)           # half a line gap above the first line
    if read_on                                                                       # band k: the lines PER (k − 1) + 1 … PER k
        cuts = [mid(PER * k) for k in 1:n]
        return collect(zip([top; cuts[1:end-1]], cuts)), top
    end
    return fill((top, mid(PER)), n), top                                             # every band the first PER lines
end

"A photograph's pose: the sheet's tilt against the optical axis [deg], the sheet's angle off the axis [deg], its distance [m]."
function texlearned_pose(view)
    R = Float64.(collect(Iterators.flatten(view["R_world_to_camera"]))); t = Float64.(collect(Iterators.flatten(view["t_world_to_camera"])))
    return rad2deg(acos(min(1.0, abs(R[9])))), rad2deg(atan(hypot(t[1], t[2]), t[3])), Float64(view["camera_distance_m"])
end

"""
    texlearned_tables(evaluate, poses, roles, photograph, tdir; bands, read_on, block_mm, lines_per_band, white, generator) -> the files written

The part `tables`: writes the figure's tables into `tdir` from a run of the stack's texture_recovery driver.

- `evaluate`: its evaluate product (design.png — the reference —, the blur models' texture_<model>.png, report.json);
- `poses`: the init stage's poses.json (the photographs' poses);
- `roles`: the refine stage's page.json (photographs, roles): the photograph must be one the texture is optimised on;
- `photograph`: the folder with what texture_learned.py's `tables` wrote for the photograph (photograph.png and meta.toml; see
  the head of this file). The photograph is chosen here again — of the eight training photographs with the least tilt the one
  with the least tilt + off-axis angle — and must be the one of that folder.
- `bands`: the bands from the top, each with its `name` (its image is `<name>.png`) and its `source` (the head of this file);
  `read_on`: the text reads on from band to band (lines 1–3, 4–6, …) instead of every band showing the first lines;
- `block_mm`: the page's text block, `[x0, y0, width, height]` in mm; `lines_per_band`: the text lines a band holds; `white`:
  the level a blank row of the design is lighter than (`texlearned_bands`);
- `generator`: what meta.toml names as its writer.

The cuts lie in the middle of the white gaps between two text lines; a band begins half a line gap above its first line.
"""
function texlearned_tables(edir::AbstractString, poses::AbstractString, roles::AbstractString, photo::AbstractString, tdir::AbstractString;
                           bands, read_on::Bool, block_mm, lines_per_band::Integer, white::Real, generator::AbstractString)
    PER = lines_per_band
    reference = "design.png"        # the run's file of the printed design (the ground truth): the reference, and a band of the source "design"
    training = "fit"                # the role (page.json) of a photograph the texture is optimised on
    order = [String(b.name) for b in bands]
    (!isempty(order) && allunique(order) && allunique(b.source for b in bands)) || error("bands: at least one, each name and each source once")
    # the run's file a band cuts its rows from (the photograph's band has none: its cut is an input)
    page(b) = b.source == "design" ? reference : "texture_$(b.source).png"
    for b in bands
        b.source == "photograph" || isfile(joinpath(edir, page(b))) || error("bands: $(b.name) shows $(b.source), and the run has no $(page(b)) ($edir)")
    end
    rep = JSON.parsefile(joinpath(edir, "report.json")); tex = Float64(rep["tex_mm"]); x0, y0, x1, y1 = block = texlearned_block(block_mm, tex)
    design = load_grey_png(joinpath(edir, reference)); TH, TW = size(design)
    T = Dict(String(b.name) => (b.source == "design" ? design : load_grey_png(joinpath(edir, page(b)))) for b in bands if b.source != "photograph")
    (all(t -> size(t) == (TH, TW), values(T)) && (TW, TH) == (rep["TW"], rep["TH"]) && x1 <= TW && y1 <= TH) || error("the textures' size $TW x $TH does not fit report.json or the text block")
    rows, top = texlearned_bands(design, length(order), read_on; block, lines_per_band, white)
    lines = [read_on ? (PER * k + 1, PER * (k + 1)) : (1, PER) for k in 0:length(order)-1]
    region = (x0, y0 + top, x1, y0 + maximum(b for (_, b) in rows))

    # the photograph: a training photograph; the sheet's normal (world z) against the optical axis, and the sheet near the axis
    views = JSON.parsefile(poses)["per_view"]; pg = JSON.parsefile(roles)
    ([v["view"] for v in views] == collect(pg["photographs"]) && length(pg["roles"]) == length(views)) || error("the poses and the run's photographs (page.json) differ")
    fit = [v for (v, r) in zip(views, pg["roles"]) if r == training]; isempty(fit) && error("no photograph of role $training")
    tilts = [texlearned_pose(v)[1] for v in fit]
    rank = sortperm(tilts; alg = MergeSort)[1:min(8, end)]                             # stable, as Python's sorted
    number = rank[argmin([sum(texlearned_pose(fit[i])[1:2]) for i in rank])]          # its number among the training photographs, in the run's (file) order
    best = fit[number]; stem = splitext(best["view"])[1]; tilt, off, dist = texlearned_pose(best)

    # the piece that is not ported: the photograph's file, its registration and its cut, as texture_learned.py wrote them
    rec = TOML.parsefile(joinpath(photo, "meta.toml")); ph = rec["photograph"]; raw_name = ph["view"]
    splitext(raw_name)[1] == stem || error("the photograph of $photo is $raw_name; the training photograph chosen from the poses is $stem")
    written = String[]; mkpath(tdir)
    k = findfirst(b -> b.source == "photograph", bands)
    if k !== nothing
        pb = rec["bands"]["photograph"]
        (pb["rows"] == [rows[k][1] - top, rows[k][2] - top] && pb["lines"] == collect(lines[k])) || error("the photograph of $photo was cut for other text lines than this layout's")
        cut = load_grey_png(joinpath(photo, "photograph.png"))
        size(cut) == (pb["pixels"][2], pb["pixels"][1]) || error("photograph.png of $photo is not of the size its meta.toml states")
        dst = joinpath(tdir, order[k] * ".png")
        abspath(dst) == abspath(joinpath(photo, "photograph.png")) || cp(joinpath(photo, "photograph.png"), dst; force = true, follow_symlinks = true)
    end
    for b in bands
        b.source == "photograph" ? push!(written, joinpath(tdir, b.name * ".png")) :
            push!(written, save_grey_png(joinpath(tdir, b.name * ".png"), T[b.name][region[2]+1:region[4], x0+1:x1]))
    end

    shown = findfirst(b -> b.source == "design", bands)                                # the band that shows the reference, if one does
    M = ["# The numbers of fig:texture-learned ($generator): <band>.png — the $(length(order)) sources (8-bit grey, design units 0 … 1 as 0 … 255).",
         "# The design and the textures: one pixel per texel of the page over the region of the bands' text lines. Texel coordinates are",
         "# 0-based, x right, y down; a range is first, past-last. [bands.*] rows: the band's rows of the region; lines: its text lines",
         "# (1-based) of the text block. The photograph: a cut of the decoded photograph at its own scale, not rectified — photo_box: the",
         "# bounding box of its band's rectangle in the photograph (its pixels); pixels: the cut's width and height; [photograph] turned_deg:",
         "# the cut is turned in the image plane by this angle about the band's centre so that the text lines are level (0: the untouched pixels).",
         "# [errors.*]: |texture − design| over the whole text block, design units. design: the run's file of the printed design (the",
         "# ground truth), the reference of the photograph's place, of the cuts and of the errors" * (shown === nothing ? "." : "; as a band it is \"$(order[shown])\"."),
         "tex_mm = $(general(tex, 6))", "page = [$TW, $TH]", "text_block = [$x0, $y0, $x1, $y1]", "region = [$(join(region, ", "))]", "design = \"$reference\"",
         "lines_per_band = $PER", "read_on = $(read_on ? "true" : "false")", "order = [" * join(("\"$s\"" for s in order), ", ") * "]"]
    for ((a, b), (la, lb), band) in zip(rows, lines, bands)
        push!(M, "", "[bands.$(band.name)]", "source = \"$(band.source == "photograph" ? raw_name : page(band))\"", "rows = [$(a - top), $(b - top)]", "lines = [$la, $lb]")
        band.source == "photograph" && push!(M, "pixels = [$(join(rec["bands"]["photograph"]["pixels"], ", "))]", "photo_box = [$(join(rec["bands"]["photograph"]["photo_box"], ", "))]")
    end
    push!(M, "", "[photograph]", "view = \"$raw_name\"", "role = \"$training\"", "number = $number", "of = $(length(fit))", "of_all_photographs = $(length(views))",
          "tilt_deg = $(fixed(tilt, 2))", "off_axis_deg = $(fixed(off, 2))", "distance_m = $(fixed(dist, 3))", "in_plane_deg = $(fixed(ph["in_plane_deg"], 2))",
          "turned_deg = $(fixed(ph["turned_deg"], 2))", "rectified = false",
          "photo_px_per_texel = $(fixed(ph["photo_px_per_texel"], 3))", "sift_matches = $(ph["sift_matches"])", "homography_inliers = $(ph["homography_inliers"])")
    # the textures' errors against the design over the whole text block: differences of single-precision greys, as the Python took them
    grey(t) = Float32.(@view t[y0+1:y1, x0+1:x1]) ./ 255.0f0
    for band in bands
        band.source in ("design", "photograph") && continue                            # an error is a blur model's texture's
        e = vec(Float64.(abs.(grey(T[band.name]) .- grey(design))))
        push!(M, "", "[errors.$(band.name)]", "median = $(fixed(Statistics.median(e), 4))", "mean = $(fixed(Statistics.mean(e), 4))", "rms = $(fixed(sqrt(Statistics.mean(abs2, e)), 4))")
    end
    push!(written, write_lines(joinpath(tdir, "meta.toml"), M))
    return written
end

# ---- figure ---------------------------------------------------------------------------------------------------------------------

"""
    texlearned_figure(tdir, out; asset_dir, bands, pixels_per_texel, generator) -> the files written

The part `figure`: draws the figure from the tables `tdir` into `out`: `<asset_dir>.tex`, `<asset_dir>/<band>.tex` and `.png`,
`<asset_dir>-caption.tex`.

- `asset_dir`: the figure's name among the paper's files;
- `bands`: the bands of the tables, each with its `name`, its `label` — its name in the margin at the left (LaTeX; the
  photograph's is followed by a line with its number among the training photographs) — and its `caption` — what the band is, in
  the caption and in the head of the band's file (the photograph's is followed by its number and by how its cut was made);
- `pixels_per_texel`: a band's image shows each texel (the photograph's band: each of its pixels) as a square of this many pixels,
  not interpolated;
- `generator`: what the files' first lines name as their generator.
"""
function texlearned_figure(tdir::AbstractString, out::AbstractString; asset_dir::AbstractString, bands, pixels_per_texel::Integer, generator::AbstractString)
    NAME = asset_dir; GEN = "generated by $generator; do not edit"; UP = pixels_per_texel
    NAMEW, GAP = 44, 4              # pt: the width of the names' margin; between it and the bands
    labels = Dict(String(b.name) => b.label for b in bands); captions = Dict(String(b.name) => b.caption for b in bands)
    V = TOML.parsefile(joinpath(tdir, "meta.toml")); x0, y0, x1, y1 = V["region"]; W = x1 - x0; tex = general(V["tex_mm"], 6); ph = V["photograph"]
    all(s -> haskey(labels, s), V["order"]) || error("bands: the tables of $tdir hold $(join(V["order"], ", ")); given are $(join((b.name for b in bands), ", "))")
    turned = Float64(get(ph, "turned_deg", 0))
    written = String[]; P = []      # per band: its slug and its height / width
    for slug in V["order"]
        B = V["bands"][slug]; a, b = B["rows"]; im = load_grey_png(joinpath(tdir, slug * ".png"))
        if haskey(B, "pixels")      # the photograph: its own pixels, the whole cut
            w, h = B["pixels"]; size(im) == (h, w) || error("$slug.png is $(size(im, 2)) x $(size(im, 1)) px, not $w x $h"); px0, py0, px1, py1 = B["photo_box"]
            what = "training photograph $(ph["number"]) of $(ph["of"]) ($(ph["view"])), not rectified: $w x $h px at its own scale from its pixels x $px0 … $px1, y $py0 … $py1" *
                   (turned != 0 ? ", turned by $(fixed(turned, 2)) deg in the image plane about the band's centre so that the lines are level (bicubic) —" : ", not warped, not resampled —") *
                   " the text lines $(B["lines"][1]) to $(B["lines"][2]) of the page's text block"
            unit = "pixel of the photograph"
        else
            size(im) == (y1 - y0, W) || error("$slug.png is $(size(im, 2)) x $(size(im, 1)) px, not the region's $W x $(y1 - y0)")
            im = im[a+1:b, :]; h, w = size(im)
            what = "$(captions[slug]) — the text lines $(B["lines"][1]) to $(B["lines"][2]) of the page's text block, $w x $h texels of $tex mm (the page's texels x $x0 … $x1, y $(y0 + a) … $(y0 + b))"
            unit = "texel"
        end
        push!(written, save_grey_png(joinpath(out, NAME, slug * ".png"), repeat(im; inner = (UP, UP))))   # a texel stays a texel at any magnification
        push!(written, write_lines(joinpath(out, NAME, slug * ".tex"), [
            "% $NAME/$slug.tex — $GEN.",
            "% One band: $what,",
            "% as one PGFPlots axis with the band's image ($UP x $UP pixels per $unit, not interpolated).",
            "% Its width is \\rcpanelwidth (\\linewidth when the host does not define it), the baseline the axis' centre;",
            "% \\rcfigdir is the figure's folder as the host document sees it. No frame, no ticks: the picture is the band.",
            "\\providecommand{\\rcfigdir}{src/fig}%", "\\providecommand{\\rcpanelwidth}{\\linewidth}%",
            "\\begin{tikzpicture}[baseline=(current axis.center)]",
            "\\begin{axis}[",
            "  scale only axis, width=\\rcpanelwidth, height=$(fixed(h / w, 5))*\\rcpanelwidth,",
            "  " * HOUSE_AXIS_COMMON * ",",
            "  xmin=0, xmax=$w, ymin=0, ymax=$h, enlargelimits=false,",
            "  xtick=\\empty, ytick=\\empty, axis on top, axis line style={draw=none}]",
            "\\addplot graphics[xmin=0, xmax=$w, ymin=0, ymax=$h] {\\rcfigdir/$NAME/$slug.png};",
            "\\end{axis}", "\\end{tikzpicture}%"]))
        push!(P, (slug, h / w))
    end

    # the composition: the bands one under the other without a gap, their names in a margin at the left
    cut(slug) = haskey(V["bands"][slug], "pixels")                                    # the photograph's band: the tables hold its cut's size
    label(slug) = cut(slug) ? "$(labels[slug])\\\\$(ph["number"]) of $(ph["of"])" : labels[slug]
    # a box of the band's height with its name in the middle (its baseline the box's centre, as the band's); the rule lies on its
    # lower edge and takes no room. The names' font: three lines of 7.5 pt fit a band of about 25 pt with room to the rules.
    name(slug, h, rule) = "\\tikz[baseline=(n.center)]{\\node[inner sep=0pt, outer sep=0pt, minimum width=$(NAMEW)pt, minimum height=$(fixed(h, 5))\\rcpanelwd, align=center, font=\\fontsize{6.5}{7.5}\\selectfont] (n) {$(label(slug))};" *
                          (rule ? "\\draw[overlay, black!60, line width=0.4pt] (n.south west) -- (n.south east);" : "") * "}"   # the short rule between two names
    same = !get(V, "read_on", true)
    push!(written, write_lines(joinpath(out, NAME * ".tex"), [
        "% $NAME.tex — $GEN.",
        "% The figure: its $(length(P)) bands (one PGFPlots picture each, $NAME/*.tex) one under the other in a tabular, without a gap: " *
        (same ? "the same text lines of the page," : "together they are one"),
        (same ? "% once from each source." : "% block of text that reads on from band to band, each band from another source.") *
        " The bands' names are LaTeX text in the host document's",
        "% fonts, in a margin at the left; the short rules between them separate the names. \\input it inside a figure (one column); it needs",
        "% \\usepackage{pgfplots} and \\pgfplotsset{compat=1.18}. \\rcfigdir is the folder this file lies in, as the host document sees it (default",
        "% src/fig). The bands are as wide as the line less the names' margin ($NAMEW pt) and the gap ($GAP pt).",
        "\\providecommand{\\rcfigdir}{src/fig}%",
        "\\begingroup",
        "\\ifdefined\\rcpanelwd\\else\\newlength\\rcpanelwd\\fi",
        "\\setlength{\\rcpanelwd}{\\dimexpr\\linewidth-$(NAMEW + GAP)pt\\relax}\\def\\rcpanelwidth{\\rcpanelwd}%",
        "\\footnotesize\\setlength{\\tabcolsep}{0pt}%",
        "\\begin{tabular}{@{}c@{\\hspace{$(GAP)pt}}c@{}}",
        ("  $(name(slug, h, k < length(P))) &\n  \\input{\\rcfigdir/$NAME/$slug} \\\\" for (k, (slug, h)) in enumerate(P))...,
        "\\end{tabular}%", "\\endgroup"]))

    # what the caption says of a band: its caption's words; of the photograph also its number and how its cut was made
    told(slug) = !cut(slug) ? captions[slug] :
                 captions[slug] * " (number $(ph["number"]) of the $(ph["of"]) the texture is optimised on), " *
                 (get(ph, "rectified", true) ? "registered to the page" :
                  turned != 0 ? "rotated in the image plane so that its lines are level, not rectified" : "its own pixels, not rectified")
    words = Dict(3 => "three", 4 => "four", 5 => "five")                              # the counts of lines the caption spells out; any other is a numeral
    what = [told(slug) for (slug, _) in P]; per = get(words, V["lines_per_band"], V["lines_per_band"])
    push!(written, write_lines(joinpath(out, NAME * "-caption.tex"), [
        "The texture of a printed page of text, recovered by inverse rendering from out-of-focus photographs" *
        (same ? ": the first $per lines of the page's text, once from each source. " : ", shown as one block of text that reads on from band to band, $per lines per band. ") *
        "From the top: " * join(what[1:end-1], "; ") * "; and " * what[end] * ". A texel is $tex\\,mm of the page."]))
    return written
end

"""
    figure_texture_learned(; data, out, part = "figure", asset_dir, bands, read_on, block_mm, lines_per_band, white, pixels_per_texel, evaluate, poses, roles, photograph = data, generator)

The learned-texture figure. `part` is texture_learned.py's subcommand:

  - `figure` (the default): draws the figure from the tables in `data` into `out`.
  - `tables`: writes the tables into `data` from the run's products `evaluate` (the evaluate product's folder), `poses` (the init
    product's poses.json) and `roles` (the refine product's page.json), and from the photograph's band as texture_learned.py
    made it, in the folder `photograph` (photograph.png and meta.toml; by default `data` itself).
  - `all`: tables, then figure.

What the figure shows is the caller's, every one of these required:

  - `asset_dir`: the figure's name among the paper's files (`<asset_dir>.tex`, the folder `<asset_dir>/` of its bands,
    `<asset_dir>-caption.tex`);
  - `bands`: the bands from the top, each `(; name, source, label, caption)` — its files' name, what it shows (the head of this
    file), its name in the margin and the caption's words for it (`texlearned_figure`);
  - `read_on`: whether the text reads on from band to band;
  - `block_mm`, `lines_per_band`, `white`: the page's text block, the text lines a band holds and the level a blank row of the
    design is lighter than (`texlearned_tables`);
  - `pixels_per_texel`: the pixels a texel is drawn with, along each side.

`generator`: what the written files name as their generator (by default this file; the stage `bands`
of scripts/texture_recovery.jl names itself).
"""
function figure_texture_learned(; data::AbstractString, out = nothing, part::AbstractString = "figure", asset_dir::AbstractString, bands, read_on::Bool,
                                block_mm, lines_per_band::Integer, white::Real, pixels_per_texel::Integer,
                                evaluate = nothing, poses = nothing, roles = nothing, photograph::AbstractString = data,
                                generator::AbstractString = "experiments/src/paper_figures/texture_learned.jl")
    part in ("figure", "tables", "all") || error("part: one of figure, tables, all")
    written = String[]
    if part != "figure"
        (evaluate === nothing || poses === nothing || roles === nothing) && error("the part tables needs evaluate, poses and roles")
        append!(written, texlearned_tables(evaluate, poses, roles, photograph, data; bands, read_on, block_mm, lines_per_band, white, generator))
    end
    if part != "tables"
        out === nothing && error("the part figure needs out")
        append!(written, texlearned_figure(data, out; asset_dir, bands, pixels_per_texel, generator))
    end
    return written
end

FIGURES["texture-learned"] = figure_texture_learned
