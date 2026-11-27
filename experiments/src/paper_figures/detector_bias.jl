# detector_bias.jl — the paper's detector-bias figure on rendered images (fig:detector-bias: src/fig/<asset_dir>.tex, its panels
# <asset_dir>/<row>-<panel>.{tex,png,jpg}, the two keys and <asset_dir>-caption.tex) as PGFPlots / TikZ; the port of the figure
# folder's tools/detector_bias.py (its two parts, `tables` and `figure`). How a feature detector's bias enters a geometric
# calibration and what the radiometric refinement leaves of it, on renders with known camera and poses: every panel is its own
# PGFPlots picture, stacked by a LaTeX tabular whose titles, row names and key numbers are text of the host document. What the
# figure shows is the caller's to say (the figure's config of record, configs/experiments/detector_bias.toml [figure]): the
# figure's name, its rows — each a detector, a target, a camera model, the camera of its renders, the name of its files — and the
# settings of the bins, the images and the keys.
#
#   PaperFigures.generate("detector-bias-mitsuba"; part = "tables", data = <tables>, rows = <the rows, each with its records>, settings...)
#   PaperFigures.generate("detector-bias-mitsuba"; data = <tables>, out = <dir>, rows = <the rows>, settings...)
#
#   row names (vertical) | Render | True Projected vs. Detected | Calibrated Camera & Pose: Geometric, Radiometric | Calibrated Camera: Geometric, Radiometric
#
#   row names    detector + target, written once over the rows that share it
#   Render       one render of the row's set, headed by the camera it was rendered with
#   True Projected vs. Detected   the detector's bias: the mean (detected − true) position per sensor bin, ONE plain sensor map
#                over all detections of the set, the same rule for every row (an AprilGrid row's features are the tag corners,
#                all four of every tag together; a Siemens-star row's the star centres). Every detection enters, also a gross one
#   Calibrated Camera & Pose      every true board point projected with the calibrated camera under the calibrated poses minus its
#                true pixel: Geometric = the detector-based calibration, Radiometric = after the radiometric refinement
#   Calibrated Camera             the same under the TRUE poses, one global rotation removed (the camera frame's gauge)
#   heads        a calibration pair is headed, as the render is, on plain white over its two panels: the row's camera model, and
#                under the true poses what was done about the gauge
# The fields are bin means, interpolated bilinearly between the bin centres (common.jl field_rgb), one PNG per panel. The keys lie
# on the panels on plain white: the bias panel has its own, Geometric and Radiometric share one between them, its radius the
# quantile `key_quantile` of the two panels' bin vectors, the larger of the pair, rounded up to two digits. With the quantile 1 the
# radius is the largest bin vector. Below 1 the longer vectors are drawn at full saturation, and the key says so — a band of the
# full colours around the disc and "≥" before its number (a few rim bins with very large vectors would otherwise set the radius
# and leave the rest pale). No lines around the panels: a small gap between the two panels of a pair and between the rows, a
# larger one between Render | bias | pair | pair.
#
# The part `tables` reads, per row, the evaluation of its case (`evaluate`: scores.json — last_iteration — and the records
# learned_it<k>.json, truepose_it<k>.json for k = 0 and the last iteration: x, y the true pixels of the board points, dx, dy
# projected − true), the render pipeline's detections of the whole set (`detections`: its data.npy, a pickled dictionary; of it
# detected and projected, per image an n × 2 array of positions) and one render (`render`, a PNG). It writes the tables the part
# `figure` reads:
#   rows.json                    sensor = [W, H] px; rows = per row: slug (its files' name), group = [detector, target], model,
#                                camera, render (a file of the folder), panels (the names of its tables), bias_kind (corners |
#                                stars), facts (the statistics of every panel), and camera_gauge = false where the Calibrated
#                                Camera tables keep the camera's own frame
#   <slug>-<panel>.tsv           the bin means of a panel: ix, iy, dx, dy (px; nan = too few vectors in the bin)
#   <slug>-render.jpg            the row's render, reduced
#
# The numbers are computed in the order NumPy computed them, so that the files come out the same.
import ColorTypes, JpegTurbo

"`f\"{v:g}\"`: a number at six significant digits, trailing zeros dropped."
db_g(v::Real) = general(v, 6)
"The largest value of `a` that is not NaN (numpy.nanmax)."
db_nanmax(a) = maximum(v for v in a if !isnan(v))
"""
The quantile `q` of the values of `a` that are not NaN, as numpy.nanquantile takes it (its default, linear interpolation between
the two neighbouring order statistics, in NumPy's order of operations).
"""
function db_nanquantile(a, q::Real)
    v = sort!([x for x in a if !isnan(x)]); n = length(v)
    n == 0 && return NaN
    vi = (n - 1) * q                                          # the quantile's place among the n sorted values, from 0
    i = clamp(floor(Int, vi), 0, n - 1); g = vi - floor(vi)
    lo = v[i+1]; hi = v[min(i + 1, n - 1)+1]; d = hi - lo
    return g >= 0.5 ? hi - d * (1 - g) : lo + d * g
end

"""The calibration panels of a row, left to right: the panel's name, the evaluation's record it is binned from (`learned`: under
the calibrated poses; `truepose`: under the true poses) and whether that record is the refinement's last iteration (`false`:
iteration 0, the detector-based calibration)."""
db_calibration_panels() = (("pose-geometric", "learned", false), ("pose-radiometric", "learned", true),
                           ("camera-geometric", "truepose", false), ("camera-radiometric", "truepose", true))

"""
A row's settings checked: `files`, `key_quantile` and `notes`, which both parts read — the name of the row's files, a quantile
in (0, 1] and a note per panel named (a bias or calibration panel).
"""
function db_check_rows(rows)
    isempty(rows) && error("rows: at least one")
    allunique(String(r.files) for r in rows) || error("rows: every row names files of its own, got $(join((String(r.files) for r in rows), ", "))")
    panels = ["bias"; [p for (p, _, _) in db_calibration_panels()]]
    for r in rows
        0 < r.key_quantile <= 1 || error("rows: $(r.files): key_quantile is in (0, 1], got $(r.key_quantile)")
        all(in(panels), keys(r.notes)) || error("rows: $(r.files): a note lies on one of the panels $(join(panels, ", ")), got $(join(sort!(collect(keys(r.notes))), ", "))")
    end
    return rows
end

# ---- the part tables: the numbers as NumPy took them ------------------------------------------------------------------------------

"The mean of `a`, its sum taken in NumPy's order (common.jl npsum)."
db_mean(a::Vector{Float64}) = npsum(a) / length(a)
"n, median, mean, RMS and maximum of the lengths `l`, as rows.json records them."
db_stats(l::Vector{Float64}) = Pair{String, Any}["n" => length(l), "median" => Statistics.median(l), "mean" => db_mean(l), "rms" => sqrt(db_mean(l .^ 2)), "max" => maximum(l)]

"""
    db_pyjson(io, x, level = 0)

Writes `x` as Python's `json.dump(x, indent = 1)` does: a vector of pairs is an object (its keys in their order), any other vector
an array, a float is printed with the shortest digits that read back, and text is escaped to ASCII.
"""
function db_pyjson(io::IO, x, level::Int = 0)
    pad = "\n" * " "^(level + 1)
    if x isa AbstractVector
        object = x isa AbstractVector{<:Pair}; brackets = object ? ("{", "}") : ("[", "]")
        isempty(x) && return print(io, brackets...)
        print(io, brackets[1])
        for (i, v) in enumerate(x)
            print(io, i > 1 ? "," : "", pad)
            object && (db_pyjson(io, String(first(v))); print(io, ": "))
            db_pyjson(io, object ? last(v) : v, level + 1)
        end
        print(io, "\n", " "^level, brackets[2])
    elseif x === nothing
        print(io, "null")
    elseif x isa Bool
        print(io, x ? "true" : "false")
    elseif x isa Integer
        print(io, x)
    elseif x isa AbstractFloat
        print(io, isnan(x) ? "NaN" : isinf(x) ? (x > 0 ? "Infinity" : "-Infinity") : pystr(x))
    elseif x isa AbstractString
        print(io, '"')
        for c in x
            u = UInt32(c)
            if c == '"' || c == '\\'
                print(io, '\\', c)
            elseif c in ('\n', '\r', '\t', '\b', '\f')
                print(io, '\\', Dict('\n' => 'n', '\r' => 'r', '\t' => 't', '\b' => 'b', '\f' => 'f')[c])
            elseif 0x20 <= u < 0x7f
                print(io, c)
            elseif u < 0x10000
                print(io, "\\u", string(u; base = 16, pad = 4))
            else                                                   # beyond the basic plane: a surrogate pair
                v = u - 0x10000
                print(io, "\\u", string(0xd800 + (v >> 10); base = 16, pad = 4), "\\u", string(0xdc00 + (v & 0x3ff); base = 16, pad = 4))
            end
        end
        print(io, '"')
    else
        error("db_pyjson: no JSON form of a $(typeof(x))")
    end
    return nothing
end

# ---- the part tables: the render pipeline's data.npy ------------------------------------------------------------------------------
# data.npy is not a table of numbers: NumPy stored a Python dictionary (detected, projected, object_points: per image an array of
# positions) in Python's pickle format, which NPZ does not read. What follows reads the few pickle instructions NumPy uses for
# dictionaries, lists and arrays of numbers, and stops at anything else; nothing in the file is executed.

"A NumPy array as a pickle holds it: shape, element type (f4, f8, …; O8 = Python objects), memory order, and the data (raw bytes, or the elements of an object array)."
mutable struct DBArray
    shape::Vector{Int}; code::String; fortran::Bool; data::Any
end
"A NumPy element type as a pickle holds it: its code and byte order."
mutable struct DBDtype
    code::String; order::String
end

"Reads one pickled Python object from `io` (protocols 2 to 4; dictionaries, lists, tuples, numbers, text, bytes and NumPy arrays)."
function db_unpickle(io::IO)
    stack = Any[]; marks = Int[]; memo = Dict{Int, Any}()
    le(T) = ltoh(read(io, T))
    since_mark() = (m = pop!(marks); items = stack[m+1:end]; resize!(stack, m); items)
    while true
        op = read(io, UInt8)
        if op == 0x80; read(io, UInt8)                                              # the protocol's number
        elseif op == 0x95; read(io, UInt64)                                         # a frame's length
        elseif op == UInt8('.'); return pop!(stack)                                 # the end
        elseif op == UInt8('('); push!(marks, length(stack))
        elseif op == UInt8('}'); push!(stack, Dict{Any, Any}())
        elseif op == UInt8(']'); push!(stack, Any[])
        elseif op == UInt8(')'); push!(stack, ())
        elseif op == UInt8('N'); push!(stack, nothing)
        elseif op == 0x88; push!(stack, true)
        elseif op == 0x89; push!(stack, false)
        elseif op == UInt8('K'); push!(stack, Int(read(io, UInt8)))
        elseif op == UInt8('M'); push!(stack, Int(le(UInt16)))
        elseif op == UInt8('J'); push!(stack, Int(le(Int32)))
        elseif op == 0x8a                                                           # a long integer: little-endian bytes, two's complement
            b = read(io, Int(read(io, UInt8))); v = isempty(b) ? big(0) : sum(big(b[k]) << (8 * (k - 1)) for k in eachindex(b))
            (!isempty(b) && b[end] >= 0x80) && (v -= big(1) << (8 * length(b)))
            push!(stack, Int(v))
        elseif op == UInt8('G'); push!(stack, ntoh(read(io, Float64)))
        elseif op == 0x8c; push!(stack, String(read(io, Int(read(io, UInt8)))))
        elseif op == UInt8('X'); push!(stack, String(read(io, Int(le(UInt32)))))
        elseif op == 0x8d; push!(stack, String(read(io, Int(le(UInt64)))))
        elseif op == UInt8('C'); push!(stack, read(io, Int(read(io, UInt8))))
        elseif op == UInt8('B'); push!(stack, read(io, Int(le(UInt32))))
        elseif op == 0x8e; push!(stack, read(io, Int(le(UInt64))))
        elseif op == 0x94; memo[length(memo)] = stack[end]                          # remembered: referred to again by its number
        elseif op == UInt8('q'); memo[Int(read(io, UInt8))] = stack[end]
        elseif op == UInt8('r'); memo[Int(le(UInt32))] = stack[end]
        elseif op == UInt8('h'); push!(stack, memo[Int(read(io, UInt8))])
        elseif op == UInt8('j'); push!(stack, memo[Int(le(UInt32))])
        elseif op == 0x85; push!(stack, (pop!(stack),))
        elseif op == 0x86; b = pop!(stack); a = pop!(stack); push!(stack, (a, b))
        elseif op == 0x87; c = pop!(stack); b = pop!(stack); a = pop!(stack); push!(stack, (a, b, c))
        elseif op == UInt8('t'); push!(stack, Tuple(since_mark()))
        elseif op == UInt8('c'); push!(stack, readline(io) * "." * readline(io))    # a Python name: module.name
        elseif op == 0x93; name = pop!(stack); push!(stack, pop!(stack) * "." * name)
        elseif op == UInt8('R')                                                     # a call: only NumPy's array and element type are made
            args = pop!(stack); f = pop!(stack)
            push!(stack, f isa String && endswith(f, "multiarray._reconstruct") ? DBArray(Int[], "", false, nothing) :
                         f == "numpy.dtype" ? DBDtype(String(args[1]), "=") : nothing)
        elseif op == UInt8('b')                                                     # the state of the object made before
            state = pop!(stack); x = stack[end]
            if x isa DBArray                                                        # (version, shape, element type, Fortran order, data)
                state[3].order == ">" && error("data.npy: a big-endian array")
                x.shape = collect(Int, state[2]); x.code = state[3].code; x.fortran = state[4]; x.data = state[5]
            elseif x isa DBDtype                                                    # (version, byte order, …)
                x.order = String(state[2])
            end
        elseif op == UInt8('a'); v = pop!(stack); push!(stack[end], v)
        elseif op == UInt8('e'); items = since_mark(); append!(stack[end], items)
        elseif op == UInt8('s'); v = pop!(stack); k = pop!(stack); stack[end][k] = v
        elseif op == UInt8('u'); items = since_mark(); for i in 1:2:length(items); stack[end][items[i]] = items[i+1]; end
        else error("data.npy: the pickle instruction 0x$(string(op; base = 16, pad = 2)) is not read here")
        end
    end
end

"The dictionary NumPy stored in the file `path` (np.save of a Python dictionary; np.load(path, allow_pickle = True).item())."
function db_load_dict(path::AbstractString)
    open(path) do io
        read(io, 6) == UInt8[0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59] || error("$path: not a .npy file")
        major = read(io, UInt8); read(io, UInt8)
        header = String(read(io, major == 1 ? Int(ltoh(read(io, UInt16))) : Int(ltoh(read(io, UInt32)))))
        occursin(r"'descr':\s*'\|O'", header) || error("$path: not a pickled Python object (its header: $(strip(header)))")
        top = db_unpickle(io)
        (top isa DBArray && startswith(top.code, "O") && length(top.data) == 1 && top.data[1] isa AbstractDict) || error("$path: not a pickled dictionary")
        return top.data[1]
    end
end

"The entries of a per-image list of `data.npy` (a Python list or a NumPy array of objects)."
db_entries(v) = v isa DBArray ? (startswith(v.code, "O") ? v.data : error("data.npy: an array of numbers where a list of arrays is expected")) : v
"The positions of one image (an n × 2 array of numbers) as Float64."
function db_points(a)::Matrix{Float64}
    a isa DBArray || error("data.npy: an entry that is not a NumPy array")
    T = get(Dict("f4" => Float32, "f8" => Float64), a.code, nothing)
    T === nothing && error("data.npy: positions of the element type $(a.code)")
    v = reinterpret(T, a.data::Vector{UInt8})
    isempty(v) && return zeros(0, 2)
    (length(a.shape) == 2 && a.shape[2] == 2) || error("data.npy: positions of the shape $(a.shape)")
    return a.fortran ? Float64.(reshape(v, a.shape[1], 2)) : Float64.(permutedims(reshape(v, 2, a.shape[1])))
end

"""
    db_tables(out; rows, bins, render_px, render_jpeg_quality, bias_min_features, calibration_min_points, camera_gauge) -> Vector{String}

The part `tables`: writes the tables the figure is drawn from into `out` — rows.json, <files>-<panel>.tsv (ix iy dx dy per bin, px)
and <files>-render.jpg. `rows` are the figure's rows from the top, each `(; files, detector, target, model, camera, features,
evaluate, detections, render, …)`:

  - `files`: the name of the row's files; `detector`, `target`: the row's name at the left; `model`: the camera model of its
    calibration, the head of its calibration pairs; `camera`: the camera its renders were made with, the head of its render (all
    LaTeX text); `features`: what its detector finds, `corners` (the tag corners of an AprilGrid) or `stars` (the star centres);
  - `evaluate`: the folder of the evaluation of the row's case (the driver's stage evaluate): scores.json and the records
    learned_it<k>.json, truepose_it<k>.json for k = 0 and the last iteration;
  - `detections`: the set's data.npy of the render pipeline;
  - `render`: one render of the set (a PNG), shown in the Render column as a JPEG of `render_px` = [width, height] pixels
    (Pillow's Lanczos resize) at the quality `render_jpeg_quality`.

`bins` = [columns, rows] are the sensor bins of every field; a bias bin is kept with at least `bias_min_features` features, a
calibration bin with at least `calibration_min_points` board points. `camera_gauge`: whether the evaluation under the true poses
removed one global rotation (the records' own statement; the figure's heads and caption say it). Prints each row's statistics.
Returns the files written.
"""
function db_tables(out::AbstractString; rows, bins, render_px, render_jpeg_quality::Integer, bias_min_features::Integer, calibration_min_points::Integer, camera_gauge::Bool)
    db_check_rows(rows); nx, ny = bins
    mkpath(out); R = Any[]; sensor = nothing; written = String[]
    for r in rows
        slug = String(r.files); kind = String(r.features); facts = Pair{String, Any}[]
        kind in ("corners", "stars") || error("rows: $slug: features is corners or stars, got $kind")
        img = load_png(r.render); H, W = size(img, 1), size(img, 2)
        (sensor === nothing || sensor == (W, H)) || error("the renders differ in size: $sensor and $((W, H))")
        sensor = (W, H)
        small = pillow_resize_lanczos(img, render_px[1], render_px[2])   # as Pillow's Image.resize(…, LANCZOS) (common.jl)
        JpegTurbo.jpeg_encode(joinpath(out, "$slug-render.jpg"), collect(reinterpret(reshape, ColorTypes.RGB{ColorTypes.N0f8}, permutedims(small, (3, 1, 2)))); quality = render_jpeg_quality)
        push!(written, joinpath(out, "$slug-render.jpg"))
        # the bias: detected − true of every feature (the tag corners or the star centres), one map; every detection enters
        o = db_load_dict(r.detections); D = db_entries(o["detected"])
        det = reduce(vcat, db_points.(D)); pro = reduce(vcat, db_points.(db_entries(o["projected"])))
        ex = det[:, 1] .- pro[:, 1]; ey = det[:, 2] .- pro[:, 2]; L = hypot.(ex, ey)
        push!(written, save_bins(joinpath(out, "$slug-bias.tsv"), bin_means(pro[:, 1], pro[:, 2], ex, ey, W, H, nx, ny, bias_min_features)...,
                                 "bins $nx x $ny of the $W x $H sensor, px; detected - true $(kind == "corners" ? "tag corner" : "star centre")"))
        push!(facts, "bias" => Pair{String, Any}["images" => length(D), "features" => length(L), "cut_px" => nothing,   # cut_px null: no detection is left out
                                                 "gross_percent_all_$kind" => count(>=(1.0), L) / length(L) * 100, "error" => db_stats(L),
                                                 "mean_vector" => [db_mean(ex), db_mean(ey)]])
        # the calibration panels: the evaluation's vectors at the true pixels
        n = JSON.parsefile(joinpath(r.evaluate, "scores.json"))["last_iteration"]; push!(facts, "last_iteration" => n); summary = String[]
        for (panel, record, refined) in db_calibration_panels()
            name = "$(record)_it$(refined ? n : 0)"; T = JSON.parsefile(joinpath(r.evaluate, "$name.json"))
            x, y, dx, dy = (Float64.(T[q]) for q in ("x", "y", "dx", "dy"))
            push!(written, save_bins(joinpath(out, "$slug-$panel.tsv"), bin_means(x, y, dx, dy, W, H, nx, ny, calibration_min_points)..., "bins $nx x $ny of the $W x $H sensor, px; $name.json"))
            s = db_stats(hypot.(dx, dy)); push!(facts, panel => s); v = Dict(s)
            push!(summary, "$panel median $(fixed(v["median"], 4)) mean $(fixed(v["mean"], 4)) RMS $(fixed(v["rms"], 4)) px")
        end
        row = Pair{String, Any}["slug" => slug, "group" => [String(r.detector), String(r.target)], "model" => String(r.model), "camera" => String(r.camera),
                                "render" => "$slug-render.jpg", "panels" => ["bias"; [p for (p, _, _) in db_calibration_panels()]], "bias_kind" => kind, "facts" => facts]
        camera_gauge || push!(row, "camera_gauge" => false)
        push!(R, row)
        println("$slug: ", join(summary, "; "))
    end
    open(io -> db_pyjson(io, Pair{String, Any}["sensor" => collect(sensor), "rows" => R]), joinpath(out, "rows.json"), "w")
    push!(written, joinpath(out, "rows.json")); println("-> ", out)
    return written
end

# ---- the part figure --------------------------------------------------------------------------------------------------------------

"""
    db_figure(data, out; asset_dir, field_px, rows, generator) -> Vector{String}

The part `figure`: draws the figure `asset_dir` from the tables in `data` into `out` — `asset_dir`.tex (\\input it in a figure*;
it needs pgfplots, compat 1.18; \\rcfigdir = the folder it lies in, default src/fig), `asset_dir`/<files>-<panel>.tex with its
image beside it (one panel each, width \\rcpanelwidth; a field image is `field_px` = [width, height] pixels), the keys
flow-key.tex and flow-key-beyond.tex, and `asset_dir`-caption.tex. `rows` are the tables' rows in their order, each
`(; files, key_quantile, notes, …)`: the row's files, the quantile of the bin vectors that is the radius of its calibration
pairs' keys (1: the largest vector) and its notes, panel => a text set in light grey in the middle of that panel. `generator` is
what the files' first lines name as their generator. Prints the keys. Returns the files written.
"""
function db_figure(data::AbstractString, out::AbstractString; asset_dir::AbstractString, field_px, rows, generator::AbstractString)
    db_check_rows(rows)
    J = JSON.parsefile(joinpath(data, "rows.json")); W, H = J["sensor"]; T = J["rows"]
    [String(t["slug"]) for t in T] == [String(r.files) for r in rows] ||
        error("rows: the tables of $data hold $(join((t["slug"] for t in T), ", ")), the rows asked for are $(join((r.files for r in rows), ", "))")
    head(rel) = "% $rel — generated by $generator; do not edit."
    panels = [p for (p, _, _) in db_calibration_panels()]; cols = ["render"; "bias"; panels]
    keygroups = (["bias"], panels[1:2], panels[3:4])                 # the panels that share a key
    note_colour = "black!35"                                         # a note: light grey on the white panel
    gs, gl = 1.5, 10.0                                               # pt: the gap within a pair and between the rows; between Render | bias | pair | pair
    namew = 22                                                       # pt: the row names' column (detector + target, on two lines beside one row) and its gap to the panels
    gaps = (gl, gl, gs, gl, gs)                                      # after each panel column but the last
    groups = [String[g for g in t["group"]] for t in T]
    dir = joinpath(out, asset_dir); written = String[]
    push!(written, flow_key(joinpath(dir, "flow-key.tex"), head("$asset_dir/flow-key.tex")))
    # the cut key has a smaller disc: with its band it is about as large as the plain key
    push!(written, flow_key(joinpath(dir, "flow-key-beyond.tex"), head("$asset_dir/flow-key-beyond.tex"); beyond = true, radius_pt = 3.6))
    KEYS = Dict{Tuple{String, String}, Float64}(); CUT = Dict{Tuple{String, String}, Tuple{Int, Float64}}()
    for (row, r) in zip(T, rows)
        slug = String(row["slug"]); P = Dict(String(col) => load_bins(joinpath(data, "$slug-$col.tsv")) for col in row["panels"])
        all(c -> haskey(P, c), cols[2:end]) || error("$slug: the tables hold the panels $(join(row["panels"], ", ")), the figure draws $(join(cols[2:end], ", "))")
        for grp in keygroups   # the key's radius: what the panels show
            m = [hypot.(P[c]...) for c in grp]
            q = grp[1] == "bias" ? 1.0 : Float64(r.key_quantile)
            top = maximum(db_nanmax, m)
            if q == 1
                KEYS[(slug, grp[1])] = nice(top)
            else                  # the quantile of the panel with the longer vectors; the bins beyond it saturate
                v = nice(maximum(a -> db_nanquantile(a, q), m)); KEYS[(slug, grp[1])] = v
                CUT[(slug, grp[1])] = (sum(count(>(v), a) for a in m), top)
            end
        end
        for col in cols
            if col == "render"
                img = String(row["render"]); mkpath(dir)
                push!(written, cp(joinpath(data, img), joinpath(dir, img); force = true))
            else
                img = "$slug-$col.png"; key = KEYS[(slug, first(g[1] for g in keygroups if col in g))]
                push!(written, save_png(joinpath(dir, img), field_rgb(P[col]..., key, field_px[1], field_px[2])))
            end
            push!(written, image_panel(joinpath(dir, "$slug-$col.tex"), head("$asset_dir/$slug-$col.tex"), asset_dir, img, W, H))
        end
    end
    # the composition: titles, row names and the keys' numbers are text of the host document
    C = [head("$asset_dir.tex"),
         "% The figure: its panels (one PGFPlots picture each, $asset_dir/*.tex) stacked in a tabular. The column titles, the row names, the",
         "% renders' heads and the keys' numbers are LaTeX text in the host document's fonts; the keys lie on the panels (one per bias",
         "% panel; Geometric and Radiometric share the one between them). \\input it inside a figure*; it needs \\usepackage{pgfplots} and",
         "% \\pgfplotsset{compat=1.18}. \\rcfigdir is the folder this file lies in, as the host document sees it (default src/fig).",
         "% The six panels share the line width left by the name column ($(db_g(namew)) pt) and the gaps ($(db_g(sum(gaps))) pt).",
         "\\providecommand{\\rcfigdir}{src/fig}%",
         "\\begingroup",
         "\\ifdefined\\rcpanelwd\\else\\newlength\\rcpanelwd\\fi\\ifdefined\\rcpanelht\\else\\newlength\\rcpanelht\\fi",
         "\\setlength{\\rcpanelwd}{\\dimexpr(\\linewidth-$(db_g(namew + sum(gaps)))pt)/$(length(cols))\\relax}\\setlength{\\rcpanelht}{$(fixed(H / W, 5))\\rcpanelwd}%",
         "\\def\\rcpanelwidth{\\rcpanelwd}%",
         "\\ifdefined\\rcflowkeybox\\else\\newsavebox\\rcflowkeybox\\fi",
         "\\sbox\\rcflowkeybox{\\scriptsize\\input{\\rcfigdir/$asset_dir/flow-key}}%",
         "\\ifdefined\\rcflowkeybeyondbox\\else\\newsavebox\\rcflowkeybeyondbox\\fi",
         "\\sbox\\rcflowkeybeyondbox{\\scriptsize\\input{\\rcfigdir/$asset_dir/flow-key-beyond}}%",
         "\\footnotesize\\setlength{\\tabcolsep}{0pt}%",
         "\\begin{tabular}{@{}c@{\\hspace{2pt}}" * join("c@{\\hspace{$(db_g(x))pt}}" for x in gaps) * "c@{}}",
         " & Render & \\makebox[\\rcpanelwd][c]{True Projected vs.\\ Detected} & \\multicolumn{2}{c@{\\hspace{$(db_g(gl))pt}}}{Calibrated Camera \\& Pose} & \\multicolumn{2}{c@{}}{Calibrated Camera} \\\\",
         " & & & " * join(("{\\scriptsize $t}" for t in ("Geometric", "Radiometric", "Geometric", "Radiometric")), " & ") * " \\\\\\noalign{\\vskip 2pt}"]
    pairbox(s) = "\\makebox[0pt][r]{\\makebox[\\dimexpr2\\rcpanelwd+$(db_g(gs))pt\\relax][c]{$s}}"   # centred over the two panels of a pair, taking no room
    for (k, (row, r)) in enumerate(zip(T, rows))
        slug = String(row["slug"]); first_of_group = k == 1 || groups[k-1] != groups[k]; n = 1
        while first_of_group && k + n <= length(T) && groups[k+n] == groups[k]
            n += 1
        end
        # the row name: one line over several rows, two beside one
        gtxt = n > 1 ? join(groups[k], " + ") : "\\begin{tabular}{@{}c@{}}$(groups[k][1]) +\\\\$(groups[k][2])\\end{tabular}"
        outer = first_of_group ? "\\raisebox{\\dimexpr-0.5\\height-$(db_g((n - 1) / 2))\\rcpanelht\\relax}[0pt][0pt]{\\rotatebox{90}{\\footnotesize $gtxt}}" : ""
        cells = String[]          # (the camera model is not named at the left: the calibration pairs carry it in their heads)
        for col in cols
            s = "\\input{\\rcfigdir/$asset_dir/$slug-$col}"
            if haskey(r.notes, col)   # centred on the panel: over it, taking no room
                s *= "\\makebox[0pt][r]{\\makebox[\\rcpanelwd][c]{\\raisebox{-0.5ex}[0pt][0pt]{\\scriptsize\\textcolor{$note_colour}{$(r.notes[col])}}}}"
            end
            col == "render" && (s *= "\\makebox[0pt][r]{\\makebox[\\rcpanelwd][c]{$(headbox(row["camera"]))}}")
            for grp in keygroups
                # after the group's last panel: the bias panel's key at its lower right, a pair's between the two
                col == grp[end] || continue
                kb = haskey(CUT, (slug, grp[1])) ? keybox("\$\\geq$(num(KEYS[(slug, grp[1])]))\$\\,px", "\\rcflowkeybeyondbox") : keybox("\$$(num(KEYS[(slug, grp[1])]))\$\\,px")
                s *= length(grp) == 1 ? "\\makebox[0pt][r]{$kb\\kern2pt}" : pairbox(kb)
                if length(grp) == 2
                    # a pair's head, centred over its two panels as the render's head is over its panel: the camera model of the pair,
                    # and — under the true poses — the gauge the evaluation removes (rows.json: camera_gauge = false where a row's
                    # Calibrated Camera tables keep the camera's own frame)
                    head_txt = row["model"] * (startswith(grp[1], "camera") ? (get(row, "camera_gauge", true) ? ", global rotation removed" : ", rotation not removed") : "")
                    s *= pairbox(headbox(head_txt))
                end
            end
            push!(cells, s)
        end
        push!(C, outer * " & " * join(cells, " & ") * (k < length(T) ? " \\\\\\noalign{\\vskip $(db_g(gs))pt}" : " \\\\"))
    end
    push!(C, "\\end{tabular}", "\\endgroup")
    push!(written, write_lines(joinpath(out, "$asset_dir.tex"), C))

    # the caption
    kinds = Pair{Vector{String}, String}[]                         # per detector + target (once each): what its features are
    for (k, row) in enumerate(T)
        what = Dict("corners" => "tag corners", "stars" => "star centres")[row["bias_kind"]]
        i = findfirst(p -> first(p) == groups[k], kinds)
        i === nothing ? push!(kinds, groups[k] => what) : (kinds[i] = groups[k] => what)
    end
    cut_groups = (panels[1], panels[3])                            # a pair's key is named by its first panel
    cut_text = join(begin
            c = [CUT[(String(row["slug"]), g)] for g in cut_groups if haskey(CUT, (String(row["slug"]), g))]
            " A key with a band around its disc and \$\\geq\$ ($(join(groups[k], " + "))) is cut: its radius is the $(fixed(100 * r.key_quantile, 0))\\,\\% quantile of the bin vectors, " *
            "and the longer ones ($(join((string(n) for (n, _) in c), " and ")) bins, up to " *
            "\$$(num(maximum(top for (_, top) in c)))\$\\,px) are drawn in the full colour of their direction."
        end for (k, (row, r)) in enumerate(zip(T, rows)) if any(haskey(CUT, (String(row["slug"]), g)) for g in cut_groups))
    push!(written, write_lines(joinpath(out, "$asset_dir-caption.tex"), [
        "Detector bias and what a calibration inherits from it, on rendered images with known camera and poses. " *
        "Hue encodes the direction of a vector in the image and saturation its length; the disc is the key, the number the length at full saturation " *
        "(\\emph{Geometric} and \\emph{Radiometric} share the key between them). The fields are means per sensor bin, interpolated bilinearly. " *
        "\\emph{True Projected vs.\\ Detected}: detected minus true position, the mean over all detections of a bin (" *
        join(("$(join(g, " + ")): the $what" for (g, what) in kinds), "; ") * "). " *
        "\\emph{Calibrated Camera \\& Pose}: board points projected with the calibrated camera and poses, minus their true pixels. " *
        "\\emph{Calibrated Camera}: the same under the true poses, one global rotation removed" *
        join(" (not for $(join(groups[k], " + ")): its camera is taken in its own frame, as calibrated)" for (k, row) in enumerate(T) if !get(row, "camera_gauge", true)) * ". " *
        "\\emph{Geometric}: the detector-based calibration; \\emph{Radiometric}: after our radiometric refinement." *
        cut_text]))

    println("-> $out/$asset_dir.tex + $asset_dir/ ($(length(T)) rows x $(length(cols)) panels)")
    for (row, r) in zip(T, rows), grp in keygroups
        haskey(CUT, (String(row["slug"]), grp[1])) || continue
        n, top = CUT[(String(row["slug"]), grp[1])]
        println("   $(row["slug"]) $(grp[1]): key $(num(KEYS[(String(row["slug"]), grp[1])])) px = the $(fixed(100 * r.key_quantile, 0)) % quantile; $n bins beyond it, the largest $(fixed(top, 4)) px")
    end
    for (k, row) in enumerate(T)
        println("   $(join([groups[k]; String(row["model"])], " + ")): keys (px) " *
                join(("$(join(g, "/")) $(num(KEYS[(String(row["slug"]), g[1])]))" for g in keygroups), "  "))
    end
    return written
end

"""
    figure_detector_bias(; data, out, part = "figure", generator, asset_dir, rows, bins, field_px, render_px, render_jpeg_quality,
                         bias_min_features, calibration_min_points, camera_gauge) -> Vector{String}

The detector-bias figure. `part` is detector_bias.py's subcommand:

  - `figure` (the default): draws the figure from the tables in `data` into `out`;
  - `tables`: writes the tables into `data` from the rows' records (each row's `evaluate`, `detections`, `render`);
  - `all`: tables, then figure.

What the figure shows is the caller's, every one of these required:

  - `asset_dir`: the figure's name among the paper's files (`<asset_dir>.tex`, the folder `<asset_dir>/` of its panels and keys,
    `<asset_dir>-caption.tex`);
  - `rows`: the rows from the top (`db_tables`, `db_figure`);
  - `bins`, `render_px`, `render_jpeg_quality`, `bias_min_features`, `calibration_min_points`, `camera_gauge`: the sensor bins,
    the shown render and what the evaluation did about the gauge (`db_tables`);
  - `field_px`: a field image's pixels (`db_figure`).

`generator`: what the written files name as their generator (by default this file; the stage `figures` of
scripts/detector_bias.jl names itself). Returns the files written.
"""
function figure_detector_bias(; data::AbstractString, out = nothing, part::AbstractString = "figure", asset_dir::AbstractString, rows, bins, field_px, render_px,
                              render_jpeg_quality::Integer, bias_min_features::Integer, calibration_min_points::Integer, camera_gauge::Bool,
                              generator::AbstractString = "experiments/src/paper_figures/detector_bias.jl")
    part in ("figure", "tables", "all") || error("part: one of figure, tables, all")
    written = String[]
    part != "figure" && append!(written, db_tables(data; rows, bins, render_px, render_jpeg_quality, bias_min_features, calibration_min_points, camera_gauge))
    if part != "tables"
        out === nothing && error("the part figure needs out")
        append!(written, db_figure(data, out; asset_dir, field_px, rows, generator))
    end
    return written
end

FIGURES["detector-bias-mitsuba"] = figure_detector_bias
