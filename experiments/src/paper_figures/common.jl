# common.jl — what the PGFPlots / TikZ generators of the paper's figures share (from the figure folders' tools/common.py; included
# by PaperFigures.jl). Every number is formatted as Python formatted it there, so that the files come out the same. Beside that it
# holds what the generators need of NumPy and Pillow to get the same numbers and images: NumPy's order of summation (npsum) and
# 8-bit grey PNG files as Pillow writes them (without colour chunks).

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
