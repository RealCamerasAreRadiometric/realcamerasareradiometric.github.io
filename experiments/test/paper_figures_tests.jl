# What the generators of the paper's figures share (src/paper_figures/common.jl; the generators are the ports of T. Harder's Python
# generators), on small cases whose answers Python gave (fixtures/paper_figures/common/): numbers as Python prints them, sums in
# NumPy's order, and 8-bit grey PNG files without colour chunks, as Pillow writes them. The generators themselves are tested beside
# their drivers (focus_surfaces_tests.jl, texture_learned_figure_tests.jl, vignetting_figure_tests.jl, board_warping_tests.jl).

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
