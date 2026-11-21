# The supplement's board-warping figure (scripts/board_warping.jl, configs/supplement/board_warping.toml, src/board_warping.jl; the
# drawing code src/paper_figures/board_warping.jl): the config of record loads, names its six cases and keys its product by their
# refinements; and the figures stage's work on the calibrated targets of the refinements the paper's figure was drawn from
# (fixtures/paper_figures/board-warping/: six records of 3 kB) writes the figure folder's table of these cases and the paper's
# files. No item starts the driver: its --plan is the drivers item's (drivers_tests.jl, DRIVERS). No real run: the refinements need
# the GPU and the published data.
@testitem "board warping: the config of record, its cases and its product" default_imports = false begin
    using Test
    using DrWatson
    using TOML: TOML
    include(srcdir("experiments.jl"))
    include(srcdir("drivers.jl"))
    include(srcdir("radiometric_refinement.jl"))
    include(srcdir("paper_figures", "PaperFigures.jl"))
    include(srcdir("board_warping.jl"))

    path = projectdir("configs", "supplement", "board_warping.toml")
    cfg = load_board_warping_config(path)
    rc = board_warping_refinements(cfg)
    cases = board_warping_cases(cfg)
    # the config of record with an edit, as a file of its own
    edited(f) = (c = TOML.parsefile(path); f(c); p = joinpath(mktempdir(), "board_warping.toml"); open(io -> TOML.print(io, c), p, "w"); p)

    @testset "the cases: one per set, in the figure's order, each a case of the refinements' config" begin
        @test [c.set for c in cases] == ["ag30", "ag10fe", "ag08ta", "si30", "si30fe", "si17ta"]            # the figure's rows, one after the other
        @test [c.case for c in cases] == ["$(c.set)_spline_warped" for c in cases] && all(c -> c.model == "spline", cases)
        @test first(cases) == (; case = "ag30_spline_warped", set = "ag30", model = "spline")
        @test all(c -> haskey(rc["cases"], c.case), cases)
        @test Dict(c.set => c.case for c in cases) == cfg["sets"]                                             # nothing but [sets] is read
    end

    @testset "the camera model of the cases, by their names" begin
        @test board_warping_model(cfg["sets"]) == "spline"
        @test board_warping_model(Dict("a" => "a_kb_warped", "b_2" => "b_2_kb_warped")) == "kb"
        @test_throws ArgumentError board_warping_model(Dict("a" => "a_kb_warped", "b" => "b_spline_warped"))   # two camera models
        @test_throws ArgumentError board_warping_model(Dict("a" => "b_kb_warped"))                            # not the set's name
        @test_throws ArgumentError board_warping_model(Dict("a" => "a_kb"))                                   # not a ..._warped case
        @test_throws ArgumentError board_warping_model(Dict("a" => "a_warped"))                               # no camera model
    end

    @testset "the figure of record: the paper's name, layout, bands, mesh and view" begin
        f = cfg["figure"]; L = board_warping_layout(f)
        @test L.rows == [("AprilGrid", ("ag30", "ag10fe", "ag08ta")), ("Siemens stars", ("si30", "si30fe", "si17ta"))]
        @test L.columns == ("Sigma 14\\,mm lens", "Fisheye lens", "Tamron 20\\,mm lens")
        @test f["asset_dir"] == "board-warping" && haskey(PaperFigures.FIGURES, "board-warping")              # the paper's file names; the drawing code the stage runs
        @test f["band_heights_mm"] == [0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 2.0] && f["max_bands"] == 8
        @test f["mesh_stride"] == 10 && f["view_azimuth_elevation_deg"] == [20, 38] && f["height_exaggeration"] == 50
        @test sort(collect(keys(f))) == ["asset_dir", "band_heights_mm", "columns", "height_exaggeration", "max_bands", "mesh_stride", "rows",
                                         "view_azimuth_elevation_deg"] && cfg["heights"]["samples"] == 201
    end

    @testset "a key that is not the config's or is not there, a case that does not exist, a layout or a setting that does not fit: refused" begin
        @test load_board_warping_config(edited(identity))["figure"] == cfg["figure"]                               # the copy itself loads
        @test_throws ArgumentError load_board_warping_config(edited(c -> c["heights"]["sample"] = 201))
        @test_throws ArgumentError load_board_warping_config(edited(c -> c["heights"]["samples"] = 1))
        @test_throws ArgumentError load_board_warping_config(edited(c -> c["figure"]["colour"] = "red"))
        @test_throws ArgumentError load_board_warping_config(edited(c -> c["figure"]["model"] = "spline"))      # the cases are named, no model is chosen
        @test_throws ArgumentError load_board_warping_config(edited(c -> delete!(c["figure"], "columns")))
        @test_throws ArgumentError load_board_warping_config(edited(c -> c["colours"] = Dict("scale" => "RdBu")))
        @test_throws ArgumentError load_board_warping_config(edited(c -> delete!(c["figure"], "mesh_stride")))     # a setting of the drawing: the code has no value for it
        @test_throws ArgumentError load_board_warping_config(edited(c -> c["figure"]["asset_dir"] = ""))
        @test_throws ArgumentError load_board_warping_config(edited(c -> c["figure"]["band_heights_mm"] = [0.5, 0.25]))   # not ascending
        @test_throws ArgumentError load_board_warping_config(edited(c -> c["figure"]["band_heights_mm"] = Float64[]))
        @test_throws ArgumentError load_board_warping_config(edited(c -> c["figure"]["max_bands"] = 0))
        @test_throws ArgumentError load_board_warping_config(edited(c -> c["figure"]["mesh_stride"] = 2.5))
        @test_throws ArgumentError load_board_warping_config(edited(c -> c["figure"]["view_azimuth_elevation_deg"] = [20, 120]))   # the panels' height holds for 0 … 90° alone
        @test_throws ArgumentError load_board_warping_config(edited(c -> c["figure"]["view_azimuth_elevation_deg"] = [20]))
        @test_throws ArgumentError load_board_warping_config(edited(c -> c["figure"]["height_exaggeration"] = 0))
        @test_throws ArgumentError load_board_warping_config(edited(c -> c["sets"]["si30fe"] = "si30fe_bc5_warped"))   # no such case in the refinements' config
        @test_throws ArgumentError load_board_warping_config(edited(c -> c["sets"]["si30"] = "si30_kb_warped"))       # a case, of another camera model than the rest
        @test_throws ArgumentError load_board_warping_config(edited(c -> c["sets"]["si30"] = "ag30_spline_warped"))   # another set's case
        @test_throws ArgumentError load_board_warping_config(edited(c -> c["sets"]["si30"] = ["si30_spline_warped", "si30_kb_warped"]))   # one case per set
        @test_throws ArgumentError load_board_warping_config(edited(c -> pop!(c["figure"]["rows"][1]["sets"])))    # a row with two sets, three columns
        @test_throws ArgumentError load_board_warping_config(edited(c -> c["sets"]["sitla"] = "sitla_spline_warped"))   # a case, but no row draws the set
        @test_throws ArgumentError load_board_warping_config(edited(c -> c["upstream"]["radiometric_refinement"] = "configs/experiments/none.toml"))
    end

    @testset "the product: the supplement's, keyed by its tables and every case's refinement, not by the thread count" begin
        p = board_warping_figures_product(cfg)
        @test p.experiment == "supplement/board_warping" && p.name == "figures" && p.output == "numbers.tex"
        @test p.record["upstream"] == Dict(c.case => refine_product(rc, c.case).key for c in cases) && length(p.record["upstream"]) == 6
        @test !haskey(p.record, "execution")
        @test board_warping_figures_product(load_board_warping_config(edited(identity))).key == p.key          # the config's content, wherever its file lies
        other(f) = (c = deepcopy(cfg); f(c); board_warping_figures_product(c).key)
        @test other(c -> c["figure"]["columns"][1] = "Sigma 14\\,mm") != p.key && other(c -> c["heights"]["samples"] = 101) != p.key
        @test other(c -> c["figure"]["height_exaggeration"] = 25) != p.key && other(c -> c["figure"]["asset_dir"] = "board-warping-check") != p.key
    end
end

@testitem "board warping: the stored refinements' heights, their table and the paper's figure" default_imports = false begin
    using Test
    using DrWatson
    using JSON: JSON
    include(srcdir("experiments.jl"))
    include(srcdir("drivers.jl"))
    include(srcdir("radiometric_refinement.jl"))
    include(srcdir("paper_figures", "PaperFigures.jl"))
    include(srcdir("board_warping.jl"))

    F = projectdir("test", "fixtures", "paper_figures", "board-warping"); E = joinpath(F, "expected")
    cfg = load_board_warping_config(projectdir("configs", "supplement", "board_warping.toml"))
    rows = [split(l, '\t') for l in readlines(joinpath(F, "cases.tsv"))[2:end] if !isempty(l)]
    cases = [(; case = String(r[1]), set = String(r[2]), model = String(r[3]), product = joinpath(F, r[4])) for r in rows]
    quietly(f) = redirect_stdout(f, devnull)
    # two text files with the same lines but the first, which names the generator
    same(a, b) = (la = readlines(a); lb = readlines(b); length(la) == length(lb) && la[2:end] == lb[2:end])
    files_under(dir) = sort([relpath(joinpath(r, f), dir) for (r, _, fs) in walkdir(dir) for f in fs])
    record(key) = joinpath(F, "products", key, "radiometric_target.json")

    # the stored cases are the config's, in its order: nothing else is stored, nothing else is read
    @test [(; c.case, c.set, c.model) for c in cases] == board_warping_cases(cfg)
    @test sort(readdir(joinpath(F, "products"))) == sort([basename(c.product) for c in cases])

    @testset "one height field: the page, the lattice, the nodes" begin
        x = board_warping_heights(record("cb974231c49f581c"); samples = 201)                                  # ag30_spline_warped
        @test (x.W, x.H) == (0.979, 0.979) && size(x.Z) == (201, 201) && length(x.nodes) == 25
        @test [(q.u, q.v) for q in x.nodes[1:6]] == [(0.0, 0.0), (0.25, 0.0), (0.5, 0.0), (0.75, 0.0), (1.0, 0.0), (0.0, 0.25)]   # u runs first, v from the top
        @test all(q -> isapprox(q.surface, q.height; atol = 1e-12), x.nodes)                                  # the spline takes the stored heights at its nodes
        @test all(q -> isapprox(x.Z[round(Int, 200 * q.u) + 1, round(Int, 200 * q.v) + 1], q.surface; atol = 1e-12), x.nodes)   # Z[i, j] at (u_i, v_j)
        @test round(1e3 * x.Z[1, 1]; digits = 6) == -0.655228 && round(1e3 * x.Z[201, 1]; digits = 6) == -2.214855   # the top-left and the top-right corner [mm]
        @test isapprox(extrema(x.Z) .* 1e3 |> collect, [-2.2149, 0.3968]; atol = 5e-5)                        # the figure folder's meta.toml
        y = board_warping_heights(record("cb974231c49f581c"); samples = 41)                                   # a coarser lattice: the same surface
        @test size(y.Z) == (41, 41) && isapprox(y.Z, x.Z[1:5:end, 1:5:end]; atol = 1e-12)
        @test (z = board_warping_heights(record("0c9917f6d6087128"); samples = 5); (z.W, z.H) == (1.0, 1.0))  # si30_spline_warped: the Siemens page
    end

    @testset "a record that does not fit is refused" begin
        r = JSON.parsefile(record("cb974231c49f581c")); dir = mktempdir()
        edited(f) = (c = deepcopy(r); f(c); p = joinpath(dir, "radiometric_target_$(length(readdir(dir))).json"); open(io -> JSON.print(io, c), p, "w"); p)
        @test board_warping_heights(edited(identity); samples = 5).W == 0.979                                 # the copy itself reads
        @test_throws ArgumentError board_warping_heights(edited(c -> c["warping_aspect"] = 0.5); samples = 5)   # not the page's aspect
        @test_throws ArgumentError board_warping_heights(edited(c -> delete!(c, "target")); samples = 5)        # no page
        @test_throws ArgumentError write_board_warping_data(mktempdir(), [(; case = "c", set = "s", model = "m", product = mktempdir())]; samples = 5)   # no record
    end

    dir = mktempdir()
    quietly(() -> board_warping_figure(dir, cases, cfg))

    @testset "the data folder, as the drawing code reads it" begin
        @test readlines(joinpath(dir, "cases.tsv")) == ["case\tset\tmodel\trefine_product"; [join((c.case, c.set, c.model, c.product), '\t') for c in cases]]
        @test sort(readdir(joinpath(dir, "dump"))) == sort([["$(c.case).f64" for c in cases]; ["$(c.case)_nodes.tsv" for c in cases]])
        x = board_warping_heights(record("cb974231c49f581c"); samples = 201)
        W, H, z = PaperFigures.BoardWarping.bw_load(joinpath(dir, "dump"), "ag30_spline_warped")             # mm; rows = v, columns = u
        @test W ≈ 979 && H ≈ 979 && z == 1e3 .* permutedims(x.Z)
        @test filesize(joinpath(dir, "dump", "ag30_spline_warped.f64")) == 8 * (3 + 201^2)
        @test readlines(joinpath(dir, "dump", "ag30_spline_warped_nodes.tsv"))[1:3] ==
              ["u\tv\tnodal_height_mm\tsurface_mm", "0.000000\t0.000000\t-0.655228\t-0.655228", "0.250000\t0.000000\t-0.461394\t-0.461394"]
    end

    @testset "the table: the figure folder's numbers of these cases; one block per case and nothing else" begin
        @test same(joinpath(dir, "meta.toml"), joinpath(E, "meta.toml"))
        meta = readlines(joinpath(dir, "meta.toml"))
        @test [l for l in meta if startswith(l, "[")] == ["[cases.$(c.case)]" for c in cases]
        @test count(startswith("page_mm = "), meta) == 6 && count(startswith("surface_min_mm = "), meta) == 6 && count(startswith("surface_max_mm = "), meta) == 6
    end

    @testset "the figure: the paper's files; the caption and source.toml stay in the product" begin
        paper = filter(!=("meta.toml"), files_under(E))
        @test length(paper) == 8 && files_under(joinpath(dir, "assets")) == paper                            # the frame, six panels, the key: nothing else is published
        for f in paper
            @test same(joinpath(dir, "assets", f), joinpath(E, f))
        end
        @test isfile(joinpath(dir, "board-warping-caption.tex")) && isfile(joinpath(dir, "source.toml"))
        @test occursin("(spline camera model)", read(joinpath(dir, "board-warping-caption.tex"), String))
        @test occursin("in bands of 0.5\\,mm", read(joinpath(dir, "board-warping-caption.tex"), String))
    end

    @testset "another layout, mesh and name; a drawing the code does not have, a setting left out; a stage that names itself" begin
        fig = cfg["figure"]
        one = mktempdir()                                                                                     # one target's row, a coarser mesh
        quietly(() -> board_warping_drawing(dir, one, merge(fig, Dict("rows" => fig["rows"][2:2], "mesh_stride" => 20)); model = "spline"))
        @test files_under(joinpath(one, "board-warping")) == ["key.tex", "si17ta.tex", "si30.tex", "si30fe.tex"]
        frame = read(joinpath(one, "board-warping.tex"), String)
        @test occursin("1 rows (the targets) x 3 columns", frame) && occursin("Siemens stars", frame) && !occursin("AprilGrid", frame)
        panel = read(joinpath(one, "board-warping", "si30.tex"), String)
        @test occursin("refine product 0c9917f6d6087128", panel)                                              # si30_spline_warped
        @test occursin("11 x 11 mesh points", panel) && occursin("mesh/rows=11,", panel)                      # every 20th of 201 samples
        # another name: the files and the paths in them carry it, every line of a panel but the first is the paper's
        other = mktempdir()
        quietly(() -> board_warping_drawing(dir, other, merge(fig, Dict("asset_dir" => "warping-check")); model = "spline"))
        @test files_under(other) == sort(vcat(["source.toml", "warping-check-caption.tex", "warping-check.tex", "warping-check/key.tex"],
                                              ["warping-check/$(c.set).tex" for c in cases]))
        @test occursin("\\input{\\rcfigdir/warping-check/ag30}", read(joinpath(other, "warping-check.tex"), String))
        @test startswith(readline(joinpath(other, "warping-check", "ag30.tex")), "% warping-check/ag30.tex — generated by ")
        @test same(joinpath(other, "warping-check", "ag30.tex"), joinpath(E, "board-warping", "ag30.tex"))
        # the drawing code has no value of its own for what it draws: a setting left out is an error, and the tables take none
        generate(; options...) = quietly(() -> PaperFigures.generate("board-warping"; data = dir, out = mktempdir(), options...))
        @test_throws UndefKeywordError generate(; part = "figure3d", asset_dir = "board-warping", model = "spline", board_warping_layout(fig)...)
        @test_throws MethodError generate(; part = "tables", mesh_stride = 10)
        @test_throws ErrorException generate(; part = "figure")                                            # the drawing with image panels is not here
        # a stage that names itself: the first lines of the table and of the figure's files name it, every other line is the same
        named = mktempdir()
        quietly(() -> board_warping_tables(dir, named; writer = "a driver (its stage)"))
        quietly(() -> board_warping_drawing(dir, named, fig; model = "spline", writer = "a driver (its stage)"))
        @test endswith(readline(joinpath(named, "meta.toml")), "written by a driver (its stage)") && same(joinpath(named, "meta.toml"), joinpath(E, "meta.toml"))
        @test endswith(readline(joinpath(named, "source.toml")), "written by a driver (its stage)")
        for f in ("board-warping.tex", "board-warping/key.tex", "board-warping/ag30.tex")
            @test readline(joinpath(named, f)) == "% $f — generated by a driver (its stage); do not edit." && same(joinpath(named, f), joinpath(E, f))
        end
        # the stage figures of the driver hands over its own name
        @test occursin("writer = \"experiments/\$DRIVER (stage figures; T. Harder's board_warping.py)\")", read(scriptsdir("board_warping.jl"), String))
    end
end
