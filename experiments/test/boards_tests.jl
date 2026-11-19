# =============================================================================
# The printed boards' design records (src/boards.jl, module Boards; ported with the package's test/unit/board_records_tests.jl
# from experiments/lib/board_records.jl): VisualGeometryToolkitIO target records written by ONE writer; the Schöps boards as Core's
# PlanarTarget(SchoepsPattern), the AprilGrid as Core's PlanarTarget(AprilGrid); the records of record (tier rc_calibration) when
# data_root() holds them.
# =============================================================================

# Boards (src/boards.jl) and the page and the AprilGrid layout checks of both items below.
@testsetup module BoardRecords
    using DrWatson
    include(srcdir("experiments.jl"))
    include(srcdir("boards.jl"))
    using StaticArrays: SVector
    using SHA: sha256
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: mm, ustrip
    using RadiometricCamera: dataset_features
    const B = Boards
    file_sha256(f) = bytes2hex(open(sha256, f))
    # the page frame [mm] of a record's point: the design chart less the canvas's origin (the page's lower-left corner)
    page_mm(b, p) = ustrip.(mm, p) .- ustrip.(mm, collect(VgtkCore.Rect(VgtkCore.canvas(b.target)).origin) .* VgtkCore.m)

    # aprilgrid_board.py's corner_mm (texture frame: x right, y down; columns right, rows upward), transcribed
    corner_mm(row, col, k, margin, rows, tag, gap) = (margin + gap + (tag + gap) * col + (k in (1, 2) ? tag : 0.0),
                                                     margin + gap + (tag + gap) * (rows - 1 - row) + (k in (0, 1) ? tag : 0.0))
    function check(b, d)
        pts = VgtkCore.design(b.target); grid = B.aprilgrid_of(d)
        rows, cols, P = Int(d.rows), Int(d.cols), Float64(d.page_mm)
        r = VgtkCore.Rect(VgtkCore.canvas(b.target))                         # the page in the design chart, metres
        o = ustrip.(mm, collect(r.origin) .* VgtkCore.m)
        e = maximum(begin
            x, y = corner_mm(row, col, k, Float64(d.margin_mm), rows, Float64(d.tag_mm), Float64(d.gap_mm))
            p = ustrip.(mm, pts[VgtkCore.point_index(grid, cols * row + col, k)])
            max(abs(p[1] - o[1] - x), abs(P - (p[2] - o[2]) - y))
        end for row in 0:(rows - 1), col in 0:(cols - 1), k in 0:3)
        return length(pts), e, r
    end
end

@testitem "board records (src/boards.jl)" default_imports = false setup = [BoardRecords] begin
    using Test
    using DrWatson
    using TOML: TOML
    using StaticArrays: SVector
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: mm, ustrip
    using .BoardRecords: B, page_mm, check

    @testset "the writer and the reader: nominal points in the design chart, the page and its texture, the design kept" begin
        design = (; generator = "test", page_mm = [70.0, 60.0], squares = [6, 5], square_mm = 10.0, pattern_start_mm = [5.0, 4.0],
                  segments = 16, apriltags = [(; x = 2, y = 1, width = 2, height = 2, index = 3)], texture_px = [700, 600])
        sha = repeat("ab", 32)
        mktempdir() do dir
            B.save_schoeps_board(dir, design; texture_sha256 = sha, report = (; check = 1))
            b = B.load_board(dir)
            pts = VgtkCore.design(b.target)
            @test length(pts) == 11
            core = VgtkCore.design(VgtkCore.PlanarTarget(B.schoeps_pattern_of(b.design)))                          # Core's PlanarTarget(SchoepsPattern)
            @test length(core) == length(pts) && maximum(maximum(abs, ustrip.(mm, p) - ustrip.(mm, q)) for (p, q) in zip(pts, core)) < 1e-9
            # feature id 0 = (gx, gy) = (0, 0): the pattern chart (10, (5 − 1)·10) mm; on the page, texture (5 + 10, 4 + 10) mm, y down
            # → (15, 60 − 14) mm, y up
            @test ustrip.(mm, pts[1]) ≈ SVector(10.0, 40.0) atol = 1e-12
            @test page_mm(b, pts[1]) ≈ SVector(15.0, 46.0) atol = 1e-12
            @test page_mm(b, pts[2]) ≈ SVector(55.0, 46.0) atol = 1e-12                         # id 1 = (4, 0)
            cv = VgtkCore.canvas(b.target)
            r = VgtkCore.Rect(cv)                                                                # the page, metres
            @test collect(r.origin) ≈ [-0.005, -0.006] && collect(r.widths) ≈ [0.07, 0.06]
            @test cv.texture.path == "texture.png" && cv.texture.sha256 == sha && VgtkCore.Rect(cv.texture) == r
            @test Float64(b.design.square_mm) == 10.0 && Int(b.design.apriltags[1].index) == 3 && Int(b.report.check) == 1
            @test B.schoeps_tag_mm(b.design, b.design.apriltags[1]) == (25.0, 14.0, 20.0)
            B.save_schoeps_board(dir, design)                                                    # no texture: a page without one
            @test VgtkCore.canvas(B.load_board(dir).target).texture === nothing
        end
    end

    @testset "the AprilGrid: Core's PlanarTarget(AprilGrid) on its page, the front-end's corner layout" begin
        design = (; page_mm = 979.0, texture_px = 3916, px_per_mm = 4.0, margin_mm = 8.0, tag_mm = 90.0, gap_mm = 27.0, cols = 8, rows = 8,
                  ids = [85, 148])
        mktempdir() do dir
            B.save_aprilgrid_board(dir, design; texture_sha256 = repeat("cd", 32), report = (; check = 2))
            b = B.load_board(dir)
            n, e, r = check(b, b.design)
            @test n == 256 && e < 1e-9
            @test collect(r.widths) ≈ [0.979, 0.979] && collect(r.origin) ≈ [-0.035, -0.035]
            @test VgtkCore.canvas(b.target).texture.sha256 == repeat("cd", 32) && Int(b.report.check) == 2
            @test_throws ErrorException B.save_aprilgrid_board(dir, merge(design, (; page_mm = 1000.0)))   # the layout does not fill the page
        end
        # the manufactured dot grid (configs/targets/thorlabs_r3l3s7n1.toml): 17 x 17 dots at 5 mm, decimal-exact, Rigid
        design = TOML.parsefile(projectdir("configs", "targets", "thorlabs_r3l3s7n1.toml"))
        mktempdir() do dir
            path = B.save_dot_grid_board(dir, design)
            b = B.load_board(dir)
            P = VgtkCore.points(b.target)
            @test length(P) == 289 && VgtkCore.calibration(b.target) isa VgtkCore.Rigid
            @test ustrip.(mm, P[1]) == [0.0, 0.0] && ustrip.(mm, P[2]) == [5.0, 0.0] && ustrip.(mm, P[18]) == [0.0, 5.0]
            @test ustrip.(mm, P[289]) == [80.0, 80.0]                                     # the 80.0 mm pattern, exactly
            r = VgtkCore.Rect(VgtkCore.canvas(b.target))
            @test collect(r.widths) == [0.09, 0.09] && collect(r.origin) == [-0.005, -0.005]   # the 90 mm glass, centred
            @test occursin("\n                80.0,", read(path, String))                 # decimal-exact in the file
            @test String(b.design.part) == "R3L3S7N1"
            bytes = read(path); B.save_dot_grid_board(dir, design); @test read(path) == bytes   # rewritten byte for byte
        end
    end
end

# The records of record (tiers rc_calibration, rc_synthetic): a :data item, skipped (a recorded skip) where no data root is bound.
@testitem "board records: the records of record under data_root()" default_imports = false tags = [:data] setup = [BoardRecords] skip = !(
        haskey(ENV, "VGTK_DATA_ROOT") ||
        (p = joinpath(dirname(Base.active_project()), "data", "exp_raw"); islink(p) || isdir(p))) begin
    using Test
    using DrWatson
    using StaticArrays: SVector
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: mm, ustrip
    using RadiometricCamera: dataset_features
    using .BoardRecords: B, file_sha256, page_mm, check, data_root
    const DATA = data_root()
    const HAS_BOARDS = isfile(joinpath(DATA, "geometric_accuracy", "data", "target", "siemens_1m", "board", "target.json"))

    HAS_BOARDS || @info "board records: the boards of record are not under data_root() (tiers rc_calibration, rc_synthetic); their checks skipped"
    HAS_BOARDS && @testset "the record of record: the Siemens board (tier rc_calibration)" begin
        for (rel, n, page, start) in (("geometric_accuracy/data/target/siemens_1m/board", 816, (1000.0, 1000.0), 4.0),)
            dir = joinpath(DATA, rel)
            b = B.load_board(dir)
            pts = VgtkCore.design(b.target)
            @test length(pts) == n
            cv = VgtkCore.canvas(b.target)
            r = VgtkCore.Rect(cv)
            @test collect(r.widths) ≈ 1e-3 .* collect(page)
            core = VgtkCore.design(VgtkCore.PlanarTarget(B.schoeps_pattern_of(b.design)))                          # Core's PlanarTarget(SchoepsPattern)
            @test length(core) == length(pts) && maximum(maximum(abs, ustrip.(mm, p) - ustrip.(mm, q)) for (p, q) in zip(pts, core)) < 1e-9
            # the texture's digest in the record is the file's
            tex = joinpath(dir, "texture.png")
            @test cv.texture.sha256 == file_sha256(tex)
            sq = Float64(b.design.square_mm)
            @test page_mm(b, pts[1]) ≈ SVector(start + sq, page[2] - (Float64(b.design.pattern_start_mm[2]) + sq)) atol = 1e-9
        end
        # the Siemens record against the front-end's feature file (texture px at 4 px/mm, y down; id k ↔ point k + 1), when it is on disk
        dir = joinpath(DATA, "geometric_accuracy", "data", "target", "siemens_1m", "board")
        tf = joinpath(dir, "texture_features.bin")
        b = B.load_board(dir)
        pts = VgtkCore.design(b.target)
        feats = only(dataset_features(tf))[2]
        @test length(feats) == length(pts)
        @test maximum((p = page_mm(b, pts[id + 1]); max(abs(p[1] - x / 4), abs((1000 - p[2]) - y / 4))) for (x, y, id) in feats) < 1e-4
    end
    HAS_BOARDS && @testset "the AprilGrid: the records of record" begin
        # the two AprilGrid records the configs of record read (tier rc_calibration: the design, and the 8 px/mm SDF render of the geometric
        # pipelines' front ends)
        for rel in ("geometric_accuracy/data/target/aprilgrid/board", "geometric_accuracy/data/target/aprilgrid/board_sdf_8ppm")
            dir = joinpath(DATA, rel)
            b = B.load_board(dir)
            n, e, _ = check(b, b.design)
            @test n == 256 && e < 1e-9
            tex = joinpath(dir, "texture.png")
            @test VgtkCore.canvas(b.target).texture.sha256 == file_sha256(tex)
        end
    end
end
