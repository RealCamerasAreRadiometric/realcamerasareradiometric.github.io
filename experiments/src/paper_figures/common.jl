# common.jl — what the PGFPlots / TikZ generators of the paper's figures share (from the figure folders' tools/common.py; included
# by PaperFigures.jl). Every number is formatted as Python formatted it there, so that the files come out the same. Beside that it
# holds what the generators need of NumPy and Pillow to get the same numbers and images: NumPy's order of summation (npsum),
# Pillow's Lanczos resize (pillow_resize_lanczos) and PNG files as Pillow writes them (8-bit RGB or grey, without colour chunks).
#
# The pieces of a colour-wheel panel are here too: the mean vector per sensor bin, the bins as an image and the key. Their colour
# coding is the project's one (src/flow_colour.jl flow_rgb): hue = direction of a vector in the image (x right, y down),
# saturation = length / the key's radius, value = 1.

"Writes `lines` to `path` (its directory is made), one per line, with a final newline."
function write_lines(path::AbstractString, lines)
    mkpath(dirname(abspath(path)))
    open(io -> print(io, join(lines, "\n"), "\n"), path, "w")
    return path
end

# ---- numbers as Python prints them ----------------------------------------------------------------------------------------------

"`f\"{v:.<digits>f}\"`. A value that is not finite is nan, inf or -inf, as Python prints it (Printf writes NaN and Inf)."
function fixed(v::Real, digits::Integer)
    x = Float64(v)
    isnan(x) && return "nan"
    isinf(x) && return x > 0 ? "inf" : "-inf"
    return Printf.format(Printf.Format("%.$(digits)f"), x)
end
"""`f\"{v:.<digits>g}\"`, by Python's rule (Printf's %g differs where the rounding carries into the next decade: 99.6 at two
digits is "1e+02" in Python): the exponent X of the value rounded to `digits` digits decides — fixed notation for −4 ≤ X < digits,
else exponential — and trailing zeros go."""
function general(v::Real, digits::Integer)
    x = Float64(v)
    isnan(x) && return "nan"
    isinf(x) && return x > 0 ? "inf" : "-inf"
    p = max(digits, 1)
    e = Printf.format(Printf.Format("%.$(p - 1)e"), x)
    mant, ex = split(e, 'e'); X = parse(Int, ex)
    strip0(t) = occursin('.', t) ? rstrip(rstrip(t, '0'), '.') : t
    -4 <= X < p && return String(strip0(Printf.format(Printf.Format("%.$(p - 1 - X)f"), x)))
    return String(strip0(mant)) * "e" * (X < 0 ? "-" : "+") * lpad(abs(X), 2, '0')
end
"`f\"{v:.2g}\"`: a value at two significant digits."
num(v::Real) = general(v, 2)
"`str(v)` of a Python float or int (5.0 → \"5.0\", 5 → \"5\")."
pystr(v::Integer) = string(v)
function pystr(v::AbstractFloat)
    x = Float64(v)
    isnan(x) && return "nan"
    isinf(x) && return x > 0 ? "inf" : "-inf"
    x == 0 && return signbit(x) ? "-0.0" : "0.0"
    b, e10 = Base.Ryu.reduce_shortest(abs(x))                     # the shortest digits that read back: |x| = b · 10^e10
    d = string(b); n = length(d); E = n - 1 + e10                 # E: the decimal exponent of the first digit
    sgn = x < 0 ? "-" : ""
    if -4 <= E < 16                                               # Python's repr: plain notation in this range, at least one decimal
        e10 >= 0 && return sgn * d * "0"^e10 * ".0"
        E >= 0 && return sgn * d[1:E+1] * "." * d[E+2:end]
        return sgn * "0." * "0"^(-E - 1) * d
    end
    return sgn * d[1:1] * (n > 1 ? "." * d[2:end] : "") * "e" * (E < 0 ? "-" : "+") * lpad(abs(E), 2, '0')
end
"Rounded up to two digits (the key radius of a panel)."
nice(m::Real) = (e = exp10(floor(log10(m)) - 1); ceil(m / e - 1e-9) * e)   # exp10: the exact power of ten, as numpy's 10.0 ** k

# ---- sums as NumPy takes them ---------------------------------------------------------------------------------------------------

"The sum of `a[i0 : i0 + n - 1]` as NumPy's pairwise sum takes it: halves down to blocks of at most 128 values, each summed with eight running sums."
function _np_pairwise(a::AbstractVector{Float64}, i0::Int, n::Int)
    if n < 8
        s = -0.0
        for i in i0:i0+n-1
            s += a[i]
        end
        return s
    elseif n <= 128
        r = ntuple(k -> a[i0+k-1], 8); m = n - n % 8
        for i in i0+8:8:i0+m-1
            r = ntuple(k -> r[k] + a[i+k-1], 8)
        end
        s = ((r[1] + r[2]) + (r[3] + r[4])) + ((r[5] + r[6]) + (r[7] + r[8]))
        for i in i0+m:i0+n-1
            s += a[i]
        end
        return s
    end
    h = n ÷ 2; h -= h % 8
    return _np_pairwise(a, i0, h) + _np_pairwise(a, i0 + h, n - h)
end
"""
    npsum(a) -> Float64

The sum of the vector `a` in NumPy's order (numpy.sum, and with it numpy.mean = npsum(a) / length(a)): the vector goes through in
pieces of 8192 values (NumPy's buffer), each summed pairwise, and the pieces' sums are added up in order, starting from 0. Up to
8192 values this is the pairwise sum of the whole vector; beyond, the two differ in the last bits.
"""
function npsum(a::AbstractVector{Float64})
    Base.require_one_based_indexing(a)
    s = 0.0
    for i0 in 1:8192:length(a)
        s += _np_pairwise(a, i0, min(8192, length(a) - i0 + 1))
    end
    return s
end

# ---- colours --------------------------------------------------------------------------------------------------------------------

# flow_rgb: the one colour coding of a residual vector (src/flow_colour.jl, shared with src/figures.jl's module)
include(joinpath(@__DIR__, "..", "flow_colour.jl"))

"An xcolor colour of an (r, g, b) in [0, 1]."
xcol(c) = ((r, g, b) = (round(Int, 255 * x) for x in c); "{rgb,255:red,$r;green,$g;blue,$b}")

# ---- sensor bins ----------------------------------------------------------------------------------------------------------------

"The mean vector per sensor bin (ny × nx matrices, row = iy, NaN where a bin has fewer than `nmin` vectors)."
function bin_means(x, y, dx, dy, W::Real, H::Real, nx::Integer, ny::Integer, nmin::Real)
    n = zeros(ny, nx); sx = zeros(ny, nx); sy = zeros(ny, nx)
    for k in eachindex(x)
        ix = clamp(trunc(Int, x[k] / W * nx), 0, nx - 1) + 1; iy = clamp(trunc(Int, y[k] / H * ny), 0, ny - 1) + 1
        n[iy, ix] += 1; sx[iy, ix] += dx[k]; sy[iy, ix] += dy[k]
    end
    ok = n .>= nmin
    return ifelse.(ok, sx ./ n, NaN), ifelse.(ok, sy ./ n, NaN)
end

_f6(v::Real) = isnan(v) ? "nan" : fixed(v, 6)
"Writes the bin means as a table (ix, iy from 0; rows iy-major), as numpy.savetxt wrote it."
function save_bins(path::AbstractString, mx::AbstractMatrix, my::AbstractMatrix, note::AbstractString)
    ny, nx = size(mx)
    L = ["# ix\tiy\tdx\tdy\t($note)"]
    for iy in 1:ny, ix in 1:nx
        push!(L, string(ix - 1, "\t", iy - 1, "\t", _f6(mx[iy, ix]), "\t", _f6(my[iy, ix])))
    end
    return write_lines(path, L)
end
"Reads a table of `save_bins`: (mx, my), ny × nx."
function load_bins(path::AbstractString)
    rows = [split(l) for l in eachline(path) if !isempty(strip(l)) && !startswith(lstrip(l), "#")]
    ix = [parse(Int, r[1]) for r in rows]; iy = [parse(Int, r[2]) for r in rows]
    nx = maximum(ix) + 1; ny = maximum(iy) + 1
    length(rows) == nx * ny || error("load_bins: $path has $(length(rows)) rows, not $nx × $ny")
    mx = fill(NaN, ny, nx); my = fill(NaN, ny, nx)
    for (k, r) in enumerate(rows)                                 # row k is (iy, ix) in iy-major order, as numpy's reshape read it
        j = (k - 1) ÷ nx + 1; i = (k - 1) % nx + 1
        mx[j, i] = parse(Float64, r[3]); my[j, i] = parse(Float64, r[4])
    end
    return mx, my
end

"""The bin means as an image (h × w × 3, UInt8; row 1 = top): the VECTORS interpolated bilinearly between the bin centres, then
coloured (`flow_rgb` under the key radius `vmax`). A pixel is drawn where at least half of its interpolation weight lies on bins
with data (the data's outline, interpolated the same way), from those of the four surrounding centres that have data (weights
renormalised); beyond the outermost centres the field is constant. Elsewhere it is white."""
function field_rgb(mx::AbstractMatrix, my::AbstractMatrix, vmax::Real, w::Integer, h::Integer)
    ny, nx = size(mx)
    gx = [(i - 0.5) / w * nx - 0.5 for i in 1:w]; gy = [(j - 0.5) / h * ny - 0.5 for j in 1:h]   # pixel centres in bin-centre coordinates
    fx0 = floor.(Int, gx); fx = gx .- fx0; x1 = clamp.(fx0 .+ 1, 0, nx - 1) .+ 1; x0 = clamp.(fx0, 0, nx - 1) .+ 1
    fy0 = floor.(Int, gy); fy = gy .- fy0; y1 = clamp.(fy0 .+ 1, 0, ny - 1) .+ 1; y0 = clamp.(fy0, 0, ny - 1) .+ 1
    ok = isfinite.(mx) .& isfinite.(my)
    okf = Float64.(ok); ax = ifelse.(ok, mx, 0.0); ay = ifelse.(ok, my, 0.0)
    bil(a, j, i) = a[y0[j], x0[i]] * ((1 - fy[j]) * (1 - fx[i])) + a[y0[j], x1[i]] * ((1 - fy[j]) * fx[i]) +
                   a[y1[j], x0[i]] * (fy[j] * (1 - fx[i])) + a[y1[j], x1[i]] * (fy[j] * fx[i])
    img = fill(0xff, h, w, 3)
    for j in 1:h, i in 1:w
        sw = bil(okf, j, i)
        sw >= 0.5 || continue
        c = flow_rgb(bil(ax, j, i) / sw, bil(ay, j, i) / sw, vmax)
        for k in 1:3
            img[j, i, k] = round(UInt8, c[k] * 255)
        end
    end
    return img
end

# ---- TikZ pieces ----------------------------------------------------------------------------------------------------------------

"""The key of the colour-wheel panels as its own TikZ picture: a disc whose saturation grows with its radius, as a vector's length
does. beyond: the key of panels whose longest vectors are CUT at the key's radius — a band of the full colours around the disc,
set off from it by a white gap: every vector longer than the radius is drawn in the colour of its direction on that band."""
function flow_key(path::AbstractString, head::AbstractString; sectors::Integer = 48, rings::Integer = 6, radius_pt::Real = 5.0, beyond::Bool = false)
    K = [head, "% The panels' key: hue = direction of a vector as drawn in the image (x right, y down), saturation = length / radius.",
         "\\begin{tikzpicture}[baseline=-0.55ex]"]
    f1(v) = fixed(v, 1); f3(v) = fixed(v, 3)
    for kr in 0:rings-1
        r0 = radius_pt * kr / rings; r1 = radius_pt * (kr + 1) / rings
        for ks in 0:sectors-1
            a0 = 360.0 * ks / sectors; a1 = 360.0 * (ks + 1) / sectors; phi = deg2rad(0.5 * (a0 + a1))
            c = xcol(flow_rgb(cos(phi), -sin(phi), rings / (kr + 0.5)))          # page angle φ (y up) = image direction (cos φ, −sin φ)
            if kr == 0
                push!(K, "\\path[fill=$c,draw=$c,line width=0.1pt] (0,0) -- ($(f1(a0)):$(f3(r1))pt) arc ($(f1(a0)):$(f1(a1)):$(f3(r1))pt) -- cycle;")
            else
                push!(K, "\\path[fill=$c,draw=$c,line width=0.1pt] ($(f1(a0)):$(f3(r0))pt) arc ($(f1(a0)):$(f1(a1)):$(f3(r0))pt) -- ($(f1(a1)):$(f3(r1))pt) arc ($(f1(a1)):$(f1(a0)):$(f3(r1))pt) -- cycle;")
            end
        end
    end
    push!(K, "\\draw[black!70,line width=0.3pt] (0,0) circle[radius=$(pystr(radius_pt))pt];")
    if beyond   # the band of the vectors beyond the radius: full saturation, from 1.3 to 1.7 radii
        rb0 = 1.3 * radius_pt; rb1 = 1.7 * radius_pt
        push!(K, "% beyond the radius: vectors longer than the key's radius keep the full colour of their direction")
        for ks in 0:sectors-1
            a0 = 360.0 * ks / sectors; a1 = 360.0 * (ks + 1) / sectors; phi = deg2rad(0.5 * (a0 + a1)); c = xcol(flow_rgb(cos(phi), -sin(phi), 1.0))
            push!(K, "\\path[fill=$c,draw=$c,line width=0.1pt] ($(f1(a0)):$(f3(rb0))pt) arc ($(f1(a0)):$(f1(a1)):$(f3(rb0))pt) -- ($(f1(a1)):$(f3(rb1))pt) arc ($(f1(a1)):$(f1(a0)):$(f3(rb1))pt) -- cycle;")
        end
        push!(K, "\\draw[black!70,line width=0.2pt] (0,0) circle[radius=$(f3(rb0))pt];", "\\draw[black!70,line width=0.3pt] (0,0) circle[radius=$(f3(rb1))pt];")
    end
    push!(K, "\\end{tikzpicture}%")
    return write_lines(path, K)
end

"""One panel of a colour-wheel figure: one PGFPlots axis over the sensor (W × H px) holding the panel's image `image` (a file of
the figure's folder `figure`). No frame: the gaps of the composition separate the panels."""
function image_panel(path::AbstractString, head::AbstractString, figure::AbstractString, image::AbstractString, W::Real, H::Real)
    return write_lines(path, [head,
        "% One panel: one PGFPlots axis over the sensor with the panel's image. Its width is \\rcpanelwidth (\\linewidth when the",
        "% host does not define it), the baseline the axis' centre; \\rcfigdir is the figure's folder as the host document sees it.",
        "\\providecommand{\\rcfigdir}{src/fig}%", "\\providecommand{\\rcpanelwidth}{\\linewidth}%",
        "\\begin{tikzpicture}[baseline=(current axis.center)]",
        "\\begin{axis}[",
        "  scale only axis, width=\\rcpanelwidth, height=$(fixed(H / W, 5))*\\rcpanelwidth,",
        "  " * HOUSE_AXIS_COMMON * ",",
        "  xmin=0, xmax=$(pystr(W)), ymin=0, ymax=$(pystr(H)), enlargelimits=false,",
        "  xtick=\\empty, ytick=\\empty, axis on top, axis line style={draw=none}]",
        "\\addplot graphics[xmin=0, xmax=$(pystr(W)), ymin=0, ymax=$(pystr(H))] {\\rcfigdir/$figure/$image};",
        "\\end{axis}", "\\end{tikzpicture}%"])
end

"A key (the disc of the composition's saved box `box`, \\rcflowkeybox, and `text`) on plain white, without a frame, lying 1.5 pt above a panel's lower edge."
keybox(text::AbstractString, box::AbstractString = "\\rcflowkeybox") =
    "\\raisebox{\\dimexpr-0.5\\rcpanelht+1.5pt\\relax}[0pt][0pt]{\\tikz[baseline=(k.south)]\\node[fill=white,inner sep=1.2pt](k){\\usebox$box\\,{\\scriptsize $text}};}"

"A label on plain white, without a frame, lying 1.5 pt below a panel's upper edge."
headbox(text::AbstractString) =
    "\\raisebox{\\dimexpr0.5\\rcpanelht-1.5pt\\relax}[0pt][0pt]{\\tikz[baseline=(k.north)]\\node[fill=white,inner sep=1.2pt,font=\\scriptsize](k){$text};}"

# ---- images ---------------------------------------------------------------------------------------------------------------------

"The chunks that describe a PNG's colour space. PNGFiles writes them into every file; the Python's files (Pillow) have none."
const PNG_COLOUR_CHUNKS = ("sRGB", "gAMA", "cHRM")

"""
    png_without_chunks(bytes, drop = PNG_COLOUR_CHUNKS) -> Vector{UInt8}

The PNG file `bytes` without its chunks of the types `drop`. A PNG file is a signature of 8 bytes and then chunks, each a length
of 4 bytes (big-endian), a type of 4 letters, the data and a checksum of 4 bytes. A chunk is left out whole: nothing is compressed
again, and the image data stay as they are.
"""
function png_without_chunks(bytes::AbstractVector{UInt8}, drop = PNG_COLOUR_CHUNKS)
    (length(bytes) >= 8 && bytes[1:8] == UInt8[0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]) || error("png_without_chunks: not a PNG file")
    out = bytes[1:8]; i = 9
    while i <= length(bytes)
        i + 11 <= length(bytes) || error("png_without_chunks: the last chunk is cut off")
        n = (Int(bytes[i]) << 24) | (Int(bytes[i+1]) << 16) | (Int(bytes[i+2]) << 8) | Int(bytes[i+3]); stop = i + 11 + n   # length, type, data, checksum
        stop <= length(bytes) || error("png_without_chunks: the last chunk is cut off")
        String(bytes[i+4:i+7]) in drop || append!(out, @view bytes[i:stop])
        i = stop + 1
    end
    return out
end

"""
    save_grey_png(path, img) -> path

Writes an h × w UInt8 matrix (row 1 = top) as an 8-bit grey PNG without the colour chunks, as Pillow writes it (the folder is
made). PNGFiles has no option to leave the chunks out, so the file is encoded in memory, the chunks are taken out and the rest
is written.
"""
function save_grey_png(path::AbstractString, img::AbstractMatrix{UInt8})
    mkpath(dirname(abspath(path)))
    io = IOBuffer(); PNGFiles.save(io, collect(reinterpret(ColorTypes.Gray{ColorTypes.N0f8}, img)); compression_level = 9, compression_strategy = 0)
    write(path, png_without_chunks(take!(io)))
    return path
end
"Reads an 8-bit grey PNG as an h × w UInt8 matrix (row 1 = top). Any other kind of image is an error."
function load_grey_png(path::AbstractString)
    a = PNGFiles.load(path)
    eltype(a) == ColorTypes.Gray{ColorTypes.N0f8} || error("$path is not an 8-bit grey image (it holds $(eltype(a)))")
    return collect(reinterpret(UInt8, a))
end

"""
    save_png(path, img) -> path

Writes an h × w × 3 UInt8 array (row 1 = top) as an 8-bit RGB PNG without the colour chunks, as Pillow writes it (the folder is
made): encoded in memory, the chunks taken out, the rest written (as `save_grey_png`).
"""
function save_png(path::AbstractString, img::AbstractArray{UInt8, 3})
    mkpath(dirname(abspath(path)))
    io = IOBuffer(); PNGFiles.save(io, permutedims(reinterpret(reshape, ColorTypes.RGB{ColorTypes.N0f8}, permutedims(img, (3, 1, 2))), (1, 2)))
    write(path, png_without_chunks(take!(io)))
    return path
end
"Reads a PNG as an h × w × 3 UInt8 array (row 1 = top); grey images get three equal channels, alpha is dropped."
function load_png(path::AbstractString)
    a = PNGFiles.load(path)
    c = ColorTypes.RGB{ColorTypes.N0f8}.(a)
    out = Array{UInt8}(undef, size(c, 1), size(c, 2), 3)
    for j in axes(c, 1), i in axes(c, 2)
        out[j, i, 1] = reinterpret(ColorTypes.red(c[j, i])); out[j, i, 2] = reinterpret(ColorTypes.green(c[j, i])); out[j, i, 3] = reinterpret(ColorTypes.blue(c[j, i]))
    end
    return out
end

# ---- images as Pillow makes them ------------------------------------------------------------------------------------------------
# The Python generators resize their images with Pillow. What they use of it is done here with Pillow's own arithmetic, so that
# the images come out the same, pixel for pixel.

# Pillow's filter (Resample.c lanczos_filter): the sinc windowed by a sinc three times as wide, zero from three samples on.
# sin(πx) is taken as C takes it, sin of the product in double precision, not sinpi, which would round differently.
_pillow_sinc(x::Float64) = x == 0.0 ? 1.0 : (y = x * Float64(π); sin(y) / y)
_pillow_lanczos(x::Float64) = -3.0 <= x < 3.0 ? _pillow_sinc(x) * _pillow_sinc(x / 3) : 0.0

"""
    pillow_lanczos_coefficients(insize, outsize) -> (first, weights)

The weights Pillow resamples one axis of `insize` pixels to `outsize` pixels with (Resample.c precompute_coeffs and
normalize_coeffs_8bpc, filter LANCZOS): output pixel k is the sum of `weights[k][t]` times the input pixels `first[k] + t - 1`.
The weights of a pixel sum to one and are stored as integers in units of 2^-22, each rounded half away from zero, as Pillow
stores them for 8-bit images.
"""
function pillow_lanczos_coefficients(insize::Integer, outsize::Integer)
    scale = insize / outsize; filterscale = max(scale, 1.0)        # reducing: the filter is stretched by the scale (it then also smooths)
    support = 3.0 * filterscale; ss = 1.0 / filterscale
    first = Vector{Int}(undef, outsize); weights = Vector{Vector{Int32}}(undef, outsize)
    for xx in 0:outsize-1
        center = (xx + 0.5) * scale                                # the output pixel's centre in input coordinates
        xmin = max(trunc(Int, center - support + 0.5), 0); n = min(trunc(Int, center + support + 0.5), insize) - xmin
        w = [_pillow_lanczos((x + xmin - center + 0.5) * ss) for x in 0:n-1]
        ww = 0.0
        for v in w                                                 # summed in order, as C sums them (sum() adds in pairs)
            ww += v
        end
        ww != 0.0 && (w ./= ww)
        first[xx+1] = xmin + 1
        weights[xx+1] = [trunc(Int32, (v < 0 ? -0.5 : 0.5) + v * 4194304.0) for v in w]   # 4194304 = 2^22
    end
    return first, weights
end

"""
    pillow_resize_lanczos(img, w, h) -> Array{UInt8, 3}

`Image.resize((w, h), Image.LANCZOS)` of Pillow for an 8-bit image `img` (rows × columns × channels; row 1 = top): first the
columns are resampled to `w`, the result is rounded to 8 bit, then the rows are resampled to `h` (Resample.c
ImagingResampleInner). The sums are integer sums in units of 2^-22 and are rounded half up, exactly as Pillow's; the channels are
independent. An axis that keeps its size is not resampled.
"""
function pillow_resize_lanczos(img::AbstractArray{UInt8, 3}, w::Integer, h::Integer)
    H, W, C = size(img)
    half = 1 << 21                                                 # one half in units of 2^-22: the sums are rounded, not cut
    a = img
    if w != W
        first, weights = pillow_lanczos_coefficients(W, w)
        b = Array{UInt8}(undef, H, w, C)
        for c in 1:C, i in 1:w
            x0 = first[i] - 1; k = weights[i]
            for j in 1:H
                s = half
                for t in eachindex(k)
                    s += Int(a[j, x0+t, c]) * k[t]
                end
                b[j, i, c] = clamp(s >> 22, 0, 255) % UInt8
            end
        end
        a = b
    end
    if h != H
        first, weights = pillow_lanczos_coefficients(H, h)
        b = Array{UInt8}(undef, h, size(a, 2), C)
        for c in 1:C, i in axes(a, 2), j in 1:h
            y0 = first[j] - 1; k = weights[j]
            s = half
            for t in eachindex(k)
                s += Int(a[y0+t, i, c]) * k[t]
            end
            b[j, i, c] = clamp(s >> 22, 0, 255) % UInt8
        end
        a = b
    end
    return a === img ? copy(img) : a
end
