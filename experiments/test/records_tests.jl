# the experiments' own result records through VisualGeometryToolkitIO 0.10 (src/record_schemas.jl): every kind of the one schema table is
# written by IO's `write_record` with its schema and read back by `read_record` — the schema first, IO's envelope (generator, units,
# conventions) the writer's, the body as written — and every `write_record` call of the project names its schema from the table.
@testitem "records: every experiments kind written and read back through IO 0.10's envelope" default_imports = false begin
    using Test
    using DrWatson
    import VisualGeometryToolkitIO as VgtkIO
    include(srcdir("record_schemas.jl"))

    @test allunique(values(EXPERIMENT_RECORD_SCHEMAS))
    for (k, s) in pairs(EXPERIMENT_RECORD_SCHEMAS)
        @test s == "experiments/$(k)/v1.0"                                         # "<owner>/<kind>/v<major>.<minor>"
        f = joinpath(mktempdir(), "$(k).json")
        VgtkIO.write_record(f, (; schema = s, label = String(k), x = [1.0, 2.5], n = 3, table = Dict("a" => [1, 2])))
        r = VgtkIO.read_record(f)
        @test r.schema == s && first(keys(r)) in ("schema", :schema)            # the schema first
        @test r.generator.name == "VisualGeometryToolkitIO" && haskey(r, :units) && haskey(r, :conventions)   # IO's envelope
        @test r.label == String(k) && r.x == [1.0, 2.5] && r.n == 3 && r.table.a == [1, 2]                     # the body as written
        # the envelope is the writer's: a record stating it is refused, and one without a schema
        @test_throws ArgumentError VgtkIO.write_record(f, (; schema = s, units = "mm"))
        @test_throws ArgumentError VgtkIO.write_record(f, (; label = "no schema"))
    end
    # every write_record call of the project's code takes its schema from the table (no literal, no record without one)
    files = [f for d in ("src", "scripts") for f in readdir(projectdir(d); join = true) if endswith(f, ".jl")]
    for f in files
        lines = readlines(f)
        for (i, l) in enumerate(lines)
            occursin("write_record(", l) && !startswith(lstrip(l), "#") || continue
            call = join(lines[max(1, i - 4):min(end, i + 2)], "\n")   # the call, its record's first lines and the record built just before it
            ok = occursin("EXPERIMENT_RECORD_SCHEMAS.", call) || occursin("pairs(rec)", call)
            ok || @info "write_record without a schema of the table" file = basename(f) line = i
            @test ok
        end
    end
    used = join((read(f, String) for f in files), "\n")
    @test all(k -> occursin("EXPERIMENT_RECORD_SCHEMAS.$(k)", used) || k in (:evaluation_fixture, :night_stars),
              keys(EXPERIMENT_RECORD_SCHEMAS))                                       # no kind of the table unwritten (the fixtures' own aside)
end

# The tool's input folder (src/frontends.jl `write_tool_images`) holds its record tool_images.json beside the photographs: the
# detect stage reads the photographs from the record (`tool_photos`), never from a listing of the folder (review finding 5: the first
# file and the count were the folder's, the record and any other file among them).
@testitem "front ends: the tool's photographs are the ones its record names, not the folder's files" default_imports = false begin
    using Test
    using DrWatson
    include(srcdir("experiments.jl"))
    include(srcdir("drivers.jl"))
    include(srcdir("boards.jl"))
    include(srcdir("parametric_starts.jl"))
    include(srcdir("evaluation.jl"))
    include(srcdir("frontends.jl"))
    import ColorTypes
    lin = mktempdir(); out = joinpath(mktempdir(), "set_tool")
    for n in ("b.png", "a.png")
        PNGFiles.save(joinpath(lin, n), fill(ColorTypes.Gray{N0f16}(0.25), 6, 8))
    end
    @test write_tool_images(lin, out) == 2
    write(joinpath(out, "0_notes.txt"), "a stray file, sorted first")
    @test tool_photos(out) == ["a.png", "b.png"]                                  # the record's photographs, sorted
    @test sort(readdir(out)) != tool_photos(out)                                 # the folder holds more than the photographs
    VgtkIO.write_record(joinpath(out, "tool_images.json"), (; schema = EXPERIMENT_RECORD_SCHEMAS.conversion, photos = ["a.png"]))
    @test_throws ArgumentError tool_photos(out)                                  # another record is refused
end

@testitem "front ends: a Schöps view with no features counts 0 (view_feature_counts)" default_imports = false begin
    using Test
    using DrWatson
    include(srcdir("experiments.jl"))
    include(srcdir("drivers.jl"))
    include(srcdir("boards.jl"))
    include(srcdir("parametric_starts.jl"))
    include(srcdir("evaluation.jl"))
    include(srcdir("frontends.jl"))
    using StaticArrays
    c = VgtkCore.correspondences([1, 4], [SVector(10.5, 20.25), SVector(30.0, 40.75)])
    frames = Dict(1 => Dict(:board => c), 2 => Dict{Symbol, typeof(c)}())   # view 2: dataset.bin's image with no pattern found
    tp = VgtkCore.TargetObservations{VgtkCore.PixelFrame{:colmap}}(frames; images = Dict(1 => "a.png", 2 => "b.png"), sensors = "cam",
                                                                   targets = [repeat("ab", 32)])
    @test view_feature_counts(tp) == [("a.png", 2), ("b.png", 0)]        # it was a MethodError (`sum` over no values)
end

@testitem "front ends: each product keyed by the code its stage runs, not the drivers' (code review 0.11.0, finding 2)" default_imports = false begin
    using Test
    using DrWatson
    include(srcdir("experiments.jl"))
    include(srcdir("drivers.jl"))
    # the per-package rule: a front-end product's code is the packages its stage's computation uses — never RadiometricCamera's or
    # the project's own source, so a commit of the drivers leaves every Schöps run and fit produced ([runs.<id>] references hold)
    for (stage, pkgs) in FRONTEND_CODE
        c = code_of(pkgs)
        @test issubset(keys(c), pkgs) && !haskey(c, "RadiometricCamera") && !any(k -> k in ("src", "ext", "Project.toml"), keys(c))
    end
    @test haskey(code_of(FRONTEND_CODE["decode"]), "VisualGeometryToolkitFeatures")
    cfg = TOML.parsefile(projectdir("configs", "experiments", "frontends.toml"))
    set = first(sort(collect(keys(cfg["cases"]))))
    p = frontend_decode_product(cfg, set)
    @test p.key == Product("frontends", "decode", p.record; output = p.output, code = code_of(FRONTEND_CODE["decode"])).key
    @test p.key != Product("frontends", "decode", p.record; output = p.output, code = CODE_VERSION).key   # not the whole tree's
end
