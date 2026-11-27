# What the generators of the paper's figures share (src/paper_figures/common.jl; the generators are the ports of T. Harder's Python
# generators), on small cases whose answers Python gave (fixtures/paper_figures/common/): numbers as Python prints them, sums in
# NumPy's order, images as Pillow resizes them, PNG files without colour chunks, as Pillow writes them, and the sensor bins and the
# pieces of a colour-wheel panel as the figure folders' common.py wrote them. The generators themselves are tested beside their
# drivers (focus_surfaces_tests.jl, texture_learned_figure_tests.jl, vignetting_figure_tests.jl, board_warping_tests.jl,
# detector_bias_tests.jl).

@testitem "paper figures: numbers as Python prints them" default_imports = false begin
    using Test
    using DrWatson
    include(srcdir("paper_figures", "PaperFigures.jl"))                 # the module PaperFigures: the generators and what they share
    const P = PaperFigures
    F = projectdir("test", "fixtures", "paper_figures", "common")

    @testset "a sample of values, formatted by Python (python_format.tsv)" begin
        rows = [split(l, '\t') for l in eachline(joinpath(F, "python_format.tsv")) if !startswith(l, "#")]
        @test length(rows) == 95
        bad = String[]
        for c in rows
            x = parse(Float64, c[1])
            got = (P.general(x, 2), P.general(x, 3), P.general(x, 4), P.general(x, 1), P.fixed(x, 3), P.fixed(x, 1), P.pystr(x), P.general(-x, 2))
            for k in eachindex(got)
                got[k] == c[k+1] || push!(bad, "$(c[1]), column $(k + 1): $(got[k]), Python wrote $(c[k+1])")
            end
        end
        foreach(b -> @info(b), bad)
        @test isempty(bad)
    end

    @testset "fixed: f\"{v:.<digits>f}\"" begin
        @test [P.fixed(v, d) for (v, d) in ((NaN, 3), (Inf, 1), (-Inf, 1), (2.5, 0), (0.125, 2), (-0.0, 1), (1, 2), (1234.5678, 1), (0.5, 0), (1.5, 0))] ==
              ["nan", "inf", "-inf", "2", "0.12", "-0.0", "1.00", "1234.6", "0", "2"]                 # a tie goes to the even digit
    end

    @testset "general: f\"{v:.<digits>g}\"" begin
        cases = ((99.6, 2), (0.0001, 2), (0.00001234, 3), (123456.0, 6), (1234567.0, 6), (0.0, 2), (-0.0, 2), (NaN, 2), (Inf, 3), (-Inf, 3),
                 (100.0, 6), (2.5, 6), (1e16, 6), (0.00009996, 3), (999999.5, 6), (0.105, 2), (5.0, 0), (1.2, 6), (4.0, 6), (2.2, 6))
        @test [P.general(v, d) for (v, d) in cases] ==
              ["1e+02", "0.0001", "1.23e-05", "123456", "1.23457e+06", "0", "-0", "nan", "inf", "-inf",
               "100", "2.5", "1e+16", "0.0001", "1e+06", "0.1", "5", "1.2", "4", "2.2"]              # 99.6: the rounding carries into the next decade
    end

    @testset "pystr: str(v) of a float or an int" begin
        floats = (5.0, 1e16, 1e15, 0.0001, 0.00001, 1.5e-7, -0.0, NaN, Inf, -Inf, 0.1 + 0.2, 123456789.125, 1e22, -1234.5, 0.62, 2.5,
                  9999999999999998.0, 12345678901234567890.0)
        @test [P.pystr(v) for v in floats] ==
              ["5.0", "1e+16", "1000000000000000.0", "0.0001", "1e-05", "1.5e-07", "-0.0", "nan", "inf", "-inf", "0.30000000000000004", "123456789.125",
               "1e+22", "-1234.5", "0.62", "2.5", "9999999999999998.0", "1.2345678901234567e+19"]
        @test P.pystr(5) == "5" && P.pystr(-12) == "-12"
    end

    @testset "num is two digits; nice: rounded up to two digits" begin
        @test P.num(99.6) == "1e+02" && P.num(0.0456) == "0.046" && P.num(3) == "3"
        @test [P.pystr(P.nice(v)) for v in (0.123, 0.12, 99.1, 0.0531, 1.0, 250.0, 0.00012345)] ==
              ["0.13", "0.12", "100.0", "0.054", "1.0", "250.0", "0.00013000000000000002"]            # the last: Python's own product, digit for digit
    end

    @testset "the lines common.py wrote for nine values, a colour and the two boxes (python_misc.txt)" begin
        vals = [0.00012345, 0.0456, 1.5, 12.6, 123.4, 99.6, 1e-5, 0.105, 3.0]
        @test vcat(P.num.(vals), P.pystr.(P.nice.(vals)), [P.xcol(P.flow_rgb(0.3, -0.2, 0.5)), P.keybox("t"), P.headbox("h")]) == readlines(joinpath(F, "python_misc.txt"))
    end
end

@testitem "paper figures: NumPy's sums, grey PNG files without colour chunks" default_imports = false begin
    using Test
    using DrWatson
    import ColorTypes, PNGFiles
    include(srcdir("paper_figures", "PaperFigures.jl"))                 # the module PaperFigures: the generators and what they share
    const P = PaperFigures
    F = projectdir("test", "fixtures", "paper_figures", "common")
    "The chunk types of a PNG file's bytes, in the file's order."
    function png_chunk_types(bytes::AbstractVector{UInt8})
        types = String[]; i = 9                                         # after the signature: length (4 bytes), type (4), data, checksum (4)
        while i + 11 <= length(bytes)
            n = (Int(bytes[i]) << 24) | (Int(bytes[i+1]) << 16) | (Int(bytes[i+2]) << 8) | Int(bytes[i+3])
            push!(types, String(bytes[i+4:i+7])); i += 12 + n
        end
        return types
    end

    @testset "npsum: numpy.sum of the same vector (numpy_sum.tsv)" begin
        @test P.npsum(Float64[]) === 0.0
        rows = [split(l, '\t') for l in eachline(joinpath(F, "numpy_sum.tsv")) if !startswith(l, "#")]
        @test [parse(Int, r[1]) for r in rows] == [1, 7, 8, 9, 100, 128, 129, 1000, 8192, 8193, 20000]   # around the block sizes 8, 128 and 8192
        for r in rows
            n = parse(Int, r[1])
            a = [(isodd(i) ? -1.0 : 1.0) * 1.0e6 / i + 1 / 3 for i in 1:n]
            @test foldl(+, a) === parse(Float64, r[3])                  # the vector is NumPy's, bit for bit: summed from left to right it gives the same
            @test P.npsum(a) === parse(Float64, r[2])
            n >= 8 && @test P.npsum(a) != foldl(+, a)                   # and NumPy's order gives another sum
        end
    end

    @testset "PNG files: 8-bit grey, no colour chunks" begin
        grey = UInt8[(37 * j + 11 * i * i) % 256 for j in 1:16, i in 1:24]   # 16 rows of 24 pixels, every row and column another
        dir = mktempdir()
        grey_file = P.save_grey_png(joinpath(dir, "sub", "grey.png"), grey)  # the folder is made
        @test P.load_grey_png(grey_file) == grey
        @test eltype(PNGFiles.load(grey_file)) == ColorTypes.Gray{ColorTypes.N0f8}
        types = png_chunk_types(read(grey_file))
        @test first(types) == "IHDR" && last(types) == "IEND" && "IDAT" in types
        @test isdisjoint(types, P.PNG_COLOUR_CHUNKS)
        rgb_file = joinpath(dir, "rgb.png"); PNGFiles.save(rgb_file, fill(ColorTypes.RGB{ColorTypes.N0f8}(1, 0, 0), 3, 4))
        @test_throws ErrorException P.load_grey_png(rgb_file)           # a colour image is no grey image
        # png_without_chunks: a gAMA chunk put after the header (25 bytes after the signature's 8) is taken out again, nothing else changes
        ours = read(grey_file)
        gamma = vcat(UInt8[0, 0, 0, 4], codeunits("gAMA"), UInt8[0, 0, 0xb1, 0x8f], UInt8[0x0b, 0xfc, 0x61, 0x05])
        with = vcat(ours[1:33], gamma, ours[34:end])
        @test png_chunk_types(with)[1:2] == ["IHDR", "gAMA"]
        @test P.png_without_chunks(with) == ours
        @test P.png_without_chunks(with, ("IDAT",)) != ours && !("IDAT" in png_chunk_types(P.png_without_chunks(with, ("IDAT",))))
        @test_throws ErrorException P.png_without_chunks(UInt8[1, 2, 3])
        @test_throws ErrorException P.png_without_chunks(ours[1:end-5])  # the last chunk is cut off
    end
end

@testitem "paper figures: Pillow's resize, RGB PNG files, sensor bins and the pieces of a colour-wheel panel" default_imports = false begin
    using Test
    using DrWatson
    include(srcdir("paper_figures", "PaperFigures.jl"))                 # the module PaperFigures: the generators and what they share
    const P = PaperFigures
    F = projectdir("test", "fixtures", "paper_figures", "common")
    "The chunk types of a PNG file's bytes, in the file's order."
    function png_chunk_types(bytes::AbstractVector{UInt8})
        types = String[]; i = 9                                         # after the signature: length (4 bytes), type (4), data, checksum (4)
        while i + 11 <= length(bytes)
            n = (Int(bytes[i]) << 24) | (Int(bytes[i+1]) << 16) | (Int(bytes[i+2]) << 8) | Int(bytes[i+3])
            push!(types, String(bytes[i+4:i+7])); i += 12 + n
        end
        return types
    end

    @testset "pillow_resize_lanczos: Pillow's resize of a 24 × 16 image" begin
        img = P.load_png(joinpath(F, "pillow_in.png"))
        @test size(img) == (16, 24, 3)
        for (w, h) in ((9, 7), (40, 30), (24, 5), (11, 16))             # smaller, larger, the rows alone, the columns alone
            @test P.pillow_resize_lanczos(img, w, h) == P.load_png(joinpath(F, "pillow_lanczos_$(w)x$(h).png"))
        end
        same = P.pillow_resize_lanczos(img, 24, 16)                     # the same size: a copy, not resampled
        @test same == img && same !== img && same == P.load_png(joinpath(F, "pillow_lanczos_24x16.png"))
        start, weights = P.pillow_lanczos_coefficients(24, 9)
        @test length(start) == length(weights) == 9
        @test all(k -> start[k] >= 1 && start[k] + length(weights[k]) - 1 <= 24, 1:9)        # every weight lies on an input pixel
        @test all(w -> abs(sum(w) - 2^22) <= length(w), weights)                             # a pixel's weights sum to one (2^22), up to their rounding
    end

    @testset "PNG files: 8-bit RGB, no colour chunks" begin
        img = P.load_png(joinpath(F, "pillow_in.png"))
        rgb_file = P.save_png(joinpath(mktempdir(), "sub", "rgb.png"), img)   # the folder is made
        @test P.load_png(rgb_file) == img
        @test_throws ErrorException P.load_grey_png(rgb_file)
        types = png_chunk_types(read(rgb_file))
        @test first(types) == "IHDR" && last(types) == "IEND" && "IDAT" in types && isdisjoint(types, P.PNG_COLOUR_CHUNKS)
        grey = UInt8[(37 * j + 11 * i * i) % 256 for j in 1:16, i in 1:24]
        @test P.load_png(P.save_grey_png(joinpath(mktempdir(), "grey.png"), grey)) == cat(grey, grey, grey; dims = 3)   # a grey file read as colour: three equal channels
    end

    @testset "bin_means by hand: two bins of a 2 × 2 grid hold vectors, one of them too few" begin
        mx, my = P.bin_means([10.0, 20.0, 3990.0], [10.0, 20.0, 2990.0], [1.0, 3.0, 5.0], [0.0, 2.0, 7.0], 4000, 3000, 2, 2, 2)
        @test mx[1, 1] == 2.0 && my[1, 1] == 1.0
        @test count(isnan, mx) == 3 && count(isnan, my) == 3
    end

    pts = [parse.(Float64, split(l, '\t')) for l in eachline(joinpath(F, "bin_points.tsv")) if !startswith(l, "#")]
    x, y, dx, dy = (getindex.(pts, k) for k in 1:4)
    @test length(pts) == 300
    mx, my = P.bin_means(x, y, dx, dy, 4000, 3000, 4, 3, 20)            # 4 × 3 bins of at least 20 vectors; six points lie on or beyond the sensor's edges

    @testset "bin_means, save_bins, load_bins: the table common.py wrote from the same 300 vectors (python_bins.tsv)" begin
        @test size(mx) == size(my) == (3, 4)
        @test count(isnan, mx) == 3 && isnan(mx[2, 4]) && isnan(mx[3, 3]) && isnan(mx[3, 4]) && isequal(isnan.(mx), isnan.(my))
        t = P.save_bins(joinpath(mktempdir(), "bins.tsv"), mx, my, "note")
        @test read(t, String) == read(joinpath(F, "python_bins.tsv"), String)
        lx, ly = P.load_bins(joinpath(F, "python_bins.tsv"))
        @test isequal(isnan.(lx), isnan.(mx)) && isequal(isnan.(ly), isnan.(my))
        @test all(i -> isnan(mx[i]) || (abs(lx[i] - mx[i]) <= 5.01e-7 && abs(ly[i] - my[i]) <= 5.01e-7), eachindex(mx))   # the table has six decimals: half a unit of the last, and a tie's rounding
        short = joinpath(mktempdir(), "short.tsv")
        write(short, join(readlines(t)[1:end-1], "\n") * "\n")          # a table with a row missing is refused
        @test_throws ErrorException P.load_bins(short)
    end

    @testset "field_rgb: the bins as an image (python_field_0.png); image_panel: the file common.py wrote" begin
        @test P.field_rgb(mx, my, 0.1, 48, 36) == P.load_png(joinpath(F, "python_field_0.png"))
        @test read(P.image_panel(joinpath(mktempdir(), "panel.tex"), "% head", "fig", "a.png", 4000, 3000), String) == read(joinpath(F, "python_panel.tex"), String)
    end
end
