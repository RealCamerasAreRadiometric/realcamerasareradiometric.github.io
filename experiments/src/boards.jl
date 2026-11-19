# =============================================================================
# src/boards.jl — module Boards: the design records of the printed boards (the Schöps Siemens-star boards and the AprilGrid), ported
# from experiments/lib/board_records.jl. A board record is a VisualGeometryToolkitIO target record (vgio/target),
# `<board dir>/target.json`, holding a `PlanarTarget` — its nominal points (the star features, the tag corners) and its canvas (the
# printed page, with the texture when the board directory has one) — and, as the record's provenance, the generator's design
# (`design`) and the front-end's run report (`report`). The records of the boards of record are inputs (tier rc_calibration); this
# module is their ONE writer (`save_schoeps_board`, `save_aprilgrid_board`; the Python front-ends that called it — siemens_board.py,
# aprilgrid_board.py, macro_board.py — are archived with the tier rc_baselines' provenance) and their reader (`load_board`).
#
# THE DESIGN (Schöps's tagged Siemens-star pattern, create_calibration_pattern.py): `squares` = [nx, ny] squares of `square_mm`, the
# pattern's top-left corner `pattern_start_mm` on a `page_mm` page (texture frame: x right, y DOWN), `apriltags` = [{x, y, width, height,
# index}] in squares. The target is Core's `PlanarTarget(SchoepsPattern)`: the star features in Schöps's order (feature id k is the
# target's nominal point k + 1 — VisualGeometryToolkitIO's Schöps reader numbers a detection's point the same way, index = id + 1), in
# Core's design chart: x right and y UP as the print is seen from the front, the PATTERN's lower-left corner at the origin; the page is
# the canvas, its lower-left corner at (−pattern_start_x, pattern_start_y + ny·square − page_height) (texture feature (gx, gy) at
# pattern_start + (g + 1)·square is the chart point ((gx + 1)·square, (ny − 1 − gy)·square)).
# =============================================================================
module Boards

import JSON                                                                    # a design's one form (no file: JSON.parse(JSON.json(x)))
import VisualGeometryToolkitCore as VgtkCore
import VisualGeometryToolkitIO as VgtkIO
using VisualGeometryToolkitCore: mm, ustrip, Size2, Canvas, CanvasTexture, PlanarTarget, AbstractTarget, AprilGrid, point_index, SchoepsPattern
using StaticArrays: SVector

"""
    schoeps_pattern_of(design) -> VisualGeometryToolkitCore.SchoepsPattern

The layout of a Schöps `design` (`squares`, `square_mm`, `apriltags` in squares) as Core's `SchoepsPattern`, in millimetres.
"""
schoeps_pattern_of(d) = SchoepsPattern(; squares_x = Int(d.squares[1]), squares_y = Int(d.squares[2]), square_length = Float64(d.square_mm) * mm,
    apriltags = [(; x = Int(t.x), y = Int(t.y), width = Int(t.width), height = Int(t.height)) for t in d.apriltags])

"""
    save_schoeps_board(dir, design; texture_sha256 = nothing, report = (;)) -> path

Write `dir/target.json`: the `PlanarTarget` of a Schöps `design` (a NamedTuple or JSON object with `page_mm`, `squares`, `square_mm`,
`pattern_start_mm`, `apriltags`, …; every key is kept in the record's provenance) — Core's `PlanarTarget(SchoepsPattern)`, the star
features in Schöps's order in the design chart, the pattern's lower-left corner at the origin — on its page, the canvas (its lower-left
corner at (−pattern_start_x, pattern_start_y + ny·square − page_height)), with the texture `texture.png` (covering the page) when
`texture_sha256` is given. `report`: the front-end's run report (checks, the dataset it wrote), kept beside the design.
"""
function save_schoeps_board(dir::AbstractString, design; texture_sha256 = nothing, report = (;))
    d = JSON.parse(JSON.json(design))                                         # one form for a NamedTuple and a JSON object
    W, H = Float64(d.page_mm[1]), Float64(d.page_mm[2])
    x0, y0, sq, ny = Float64(d.pattern_start_mm[1]), Float64(d.pattern_start_mm[2]), Float64(d.square_mm), Int(d.squares[2])
    page = Size2(width = W * mm, height = H * mm)
    origin = SVector(-x0 * mm, (y0 + ny * sq - H) * mm)                        # the page's lower-left corner in the pattern's chart
    texture = texture_sha256 === nothing ? nothing : CanvasTexture("texture.png", String(texture_sha256), page, origin)
    target = PlanarTarget(schoeps_pattern_of(d); canvas = Canvas(page, origin; texture))
    path = joinpath(dir, "target.json")
    VgtkIO.save(path, target; provenance = (; design = d, report = JSON.parse(JSON.json(report)),
        nominal_points = "Core's PlanarTarget(SchoepsPattern): Schöps's feature id k is nominal point k + 1; mm in the design chart (x right, " *
                         "y up), the pattern's lower-left corner at the origin, the page's at the canvas origin"))
    return path
end

"""
    load_board(dir) -> (; target, design, report)

The board record `dir/target.json` (a Schöps board's or the AprilGrid's): the `PlanarTarget` (VisualGeometryToolkitIO's reader) and the
generator's design and the front-end's report from its provenance (JSON objects).
"""
function load_board(dir::AbstractString)
    path = joinpath(dir, "target.json")
    target = VgtkIO.load(AbstractTarget, path)
    prov = VgtkIO.read_record(path).provenance
    return (; target, design = prov.design, report = prov.report)
end

"""
    aprilgrid_of(design) -> VisualGeometryToolkitCore.AprilGrid

The layout of an AprilGrid `design` (`rows`, `cols`, `tag_mm`, `gap_mm`: Kalibr's tag rows and columns, the tag side and the gap
between two tags) as Core's `AprilGrid`, in millimetres.
"""
aprilgrid_of(d) = AprilGrid(; tag_rows = Int(d.rows), tag_cols = Int(d.cols), tag_size = Float64(d.tag_mm) * mm, tag_spacing = Float64(d.gap_mm) * mm)

"""
    save_aprilgrid_board(dir, design; texture_sha256 = nothing, report = (;)) -> path

Write `dir/target.json`: the `PlanarTarget` of an AprilGrid `design` (detector_bias/aprilgrid_board.py's layout: `rows`, `cols`,
`tag_mm`, `gap_mm`, `margin_mm`, `page_mm`, `ids`, …; every key is kept in the record's provenance) — Core's `PlanarTarget(AprilGrid)`:
the tag corners in Kalibr's point order, in the design chart (x along the columns, y along the rows: right and up as the print is seen
from the front), tag 0's first corner at the origin — on its page, the canvas: `page_mm` square, its lower-left corner at
−(`margin_mm` + `gap_mm`) on both axes (the margin and the corner square around the outer tags), with the texture `texture.png` when
`texture_sha256` is given. The front-end's feature id `4 (tag − ids[1]) + k` is the record's point `point_index(grid, tag − ids[1], k)`.
"""
function save_aprilgrid_board(dir::AbstractString, design; texture_sha256 = nothing, report = (;))
    d = JSON.parse(JSON.json(design))
    grid = aprilgrid_of(d)
    m, gap, tag = Float64(d.margin_mm), Float64(d.gap_mm), Float64(d.tag_mm)
    side(n) = 2m + 2gap + n * tag + (n - 1) * gap                                # the page along n tags: margin, corner squares, tags, gaps
    P = Float64(d.page_mm)
    (side(Int(d.cols)) ≈ P && side(Int(d.rows)) ≈ P) || error("save_aprilgrid_board: the layout spans $(side(Int(d.cols))) × $(side(Int(d.rows))) mm, the page is $P mm")
    page = Size2(width = P * mm, height = P * mm)
    origin = SVector(-(m + gap) * mm, -(m + gap) * mm)
    texture = texture_sha256 === nothing ? nothing : CanvasTexture("texture.png", String(texture_sha256), page, origin)
    target = PlanarTarget(grid; canvas = Canvas(page, origin; texture))
    path = joinpath(dir, "target.json")
    VgtkIO.save(path, target; provenance = (; design = d, report = JSON.parse(JSON.json(report)),
        nominal_points = "Kalibr's AprilGrid point order (2 rows × 2 cols corners, row by row); the front-end's feature id 4 (tag − $(Int(d.ids[1]))) + k " *
                         "is point point_index(AprilGrid, tag − $(Int(d.ids[1])), k) (1-based); mm in the design chart (x along the columns, y along the rows), " *
                         "tag 0's corner 0 at the origin"))
    return path
end

"""
    save_dot_grid_board(dir, design; report = (;)) -> path

Write `dir/target.json`: the `PlanarTarget` of a manufactured dot-grid reticle `design` (configs/targets/: `part`, `rows`, `cols`,
`spacing_mm`, `dot_diameter_mm`, `glass_mm`, `tolerance_mm`, …; every key is kept in the record's provenance). Its dimensions are the
maker's, fixed to `tolerance_mm`, so the calibration is `Rigid()`: the nominal points ARE the target. The dot centres, row by row from the
bottom row, left to right — point `(r − 1)·cols + c` at `((c − 1)·spacing, (r − 1)·spacing)` — in the design chart: x right and y up as the
pattern is seen from the front (the maker's label reading correctly), the lower-left dot at the origin; the glass, `glass_mm` square and
centred on the pattern, is the canvas. Lengths are read decimal-exact (`read_length`), never multiplied by a scale.
"""
function save_dot_grid_board(dir::AbstractString, design; report = (;))
    d = JSON.parse(JSON.json(design))
    rows, cols = Int(d.rows), Int(d.cols)
    s, G = Float64(d.spacing_mm), Float64(d.glass_mm)
    len(x) = VgtkIO.read_length(x, mm)
    nominal = [SVector(len((c - 1) * s), len((r - 1) * s)) for r in 1:rows for c in 1:cols]
    margin = (G - (cols - 1) * s) / 2
    margin == (G - (rows - 1) * s) / 2 || error("save_dot_grid_board: a square glass needs a square pattern; got $rows × $cols dots")
    canvas = Canvas(Size2(width = len(G), height = len(G)), SVector(len(-margin), len(-margin)))
    path = joinpath(dir, "target.json")
    VgtkIO.save(path, PlanarTarget(nominal; canvas); provenance = (; design = d, report = JSON.parse(JSON.json(report)),
        nominal_points = "the dot centres row by row from the bottom row, left to right (point (r − 1)·$cols + c); mm in the design chart " *
                         "(x right, y up, seen from the front), the lower-left dot at the origin; Rigid: the maker's dimensions to ±$(d.tolerance_mm) mm"))
    return path
end

"""The tag `t` of a design as its texture-frame square [mm]: (x, y, side) of its top-left corner."""
schoeps_tag_mm(d, t) = (Float64(d.pattern_start_mm[1]) + t.x * Float64(d.square_mm), Float64(d.pattern_start_mm[2]) + t.y * Float64(d.square_mm),
                        t.width * Float64(d.square_mm))

export save_schoeps_board, save_aprilgrid_board, save_dot_grid_board, load_board, schoeps_pattern_of, aprilgrid_of, schoeps_tag_mm

end # module Boards
