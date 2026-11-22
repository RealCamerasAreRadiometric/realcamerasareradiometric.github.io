# =============================================================================
# src/texture_recovery.jl — the texture test of the blur model (the defocus experiment, fig:texture-closeup; scripts/texture_recovery.jl,
# configs/experiments/texture_recovery_<set>.toml). Included after src/drivers.jl and src/radiometric_refinement.jl.
#
# A finished refinement's camera is held fixed — its lens, focus surface, aperture radius and vignetting V, nothing of it fitted here —
# on photographs of a printed sheet it never saw: an A4 BlobBoards board with text, a flat page of its own. Each photograph's pose comes
# from its blobs alone: BlobBoards' find_boards on its pose arm through the refinement's BoundedCamera (its RANSAC and local optimization
# in raw pixels), the board's frame put on the page by the sheet's blob table (Estimation's rigid alignment, Core's transforms). The pixel model is the renderer's own quadrature — the sensor samples, the aperture disc, `world_ray`, the
# sheet's intersection, bilinear texels — and it is linear in the sheet's texture x: I = V(𝐩) · Σ w · x(𝐛). The sheet is lit by itself:
# its radiance IS the unknown, so no material or lighting of the calibration board enters. The whole page is one unknown (every pixel's
# full footprint a row; no tiles, no seams), solved on the GPU by CGLS to convergence under two blur models that differ in the View's
# blur alone: `learned` (the refinement's aperture radius and focus surface) and `pinhole` (r → 0: the chief ray). The photographs are
# split three ways by index (`split_roles`: fit, validation, test); every texture carries the Tikhonov gradient penalty λ ‖∇x‖² (λ = β ·
# Core's `penalty_scale`; Core's `difference_rows`, `PenalizedOperator` and `cgls`)
# whose weight each blur model chooses BLIND on the validation photographs; the texture of record is scored on the test photographs
# (rel-RMS over all their pixels, the centre and the edge of the image) and against the known design (one affine map a · design + b
# per blur model, the RMS at the texel and at `[report] block_mm`), paired learned − pinhole over the page's `cell_mm` cells with a
# bootstrap. Only the cells enough fit, validation and test photographs cover are scored.
# =============================================================================
using RadiometricCamera
using RadiometricCamera: chart, compute_mask, draws, focus, inside, photo_file, residuals,
                         world_ray
const RC = RadiometricCamera
import VisualGeometryToolkitCore as VgtkCore
import VisualRadiometryToolkitCore as VrtkCore
import VisualGeometryToolkitEstimation as VgtkEstimation
import VisualGeometryToolkitFeatures as VgtkFeatures
import VisualGeometryToolkitIO as VgtkIO
import BlobBoards
import PNGFiles
import SHA
using VisualGeometryToolkitCore: EuclideanMap, angle_between, Point2, Ellipse
using VisualGeometryToolkitEval.Stats: bootstrap_ci          # the paired cells' interval: the stack's one bootstrap (Eval)
using StaticArrays, LinearAlgebra, Statistics, Printf
using SparseArrays: sparse, nnz, nonzeros, SparseMatrixCSC
import Adapt
using StableRNGs: StableRNG
using CairoMakie
using CairoMakie: Makie

# ---- the config of record ----------------------------------------------------------------------------------------------------------

"The blur models compared: the calibration's camera as it is, and the chief ray alone."
const TEXTURE_VARIANTS = ("learned", "pinhole")
"The tables of a texture config of record (configs/experiments/texture_recovery_<set>.toml)."
const TEXTURE_TABLES = ("experiment", "inputs", "prepare", "upstream", "poses", "texture", "report", "figures", "render", "folds", "compare")
"The optional tables of a texture config: the rotation of the split (`[folds]`) and the comparison of page runs (`[compare]`)."
const TEXTURE_OPTIONAL_TABLES = ("folds", "compare")
"The keys of `[folds]`: the rotations of the split (`split_roles`' `fold` 0 … rotations − 1; 0 the split of record)."
const TEXTURE_FOLDS_KEYS = ("rotations",)
"""The keys of `[compare]`: the page runs compared (`runs`: name → a refine key in this results root), the run whose texture the others'
differences are taken from (`reference`), this config's own page run's name among them (`this`, optional) and the page rectangle cropped (`crop_mm`)."""
const TEXTURE_COMPARE_KEYS = ("runs", "reference", "crop_mm")
"The keys of `[upstream]`: the refinement config of record, its case whose camera is held fixed, the state file, the case's photographs."
const TEXTURE_UPSTREAM_KEYS = ("refinement", "case", "state", "images")
"""The keys of `[poses]`: the photographs BlobBoards' `find_boards` reads and the board's design JSON (inputs of stage detect), the
sheet (an input of stage init), and `find_boards`, BlobBoards' own `[find_boards]` table (its pose arm: `[poses.find_boards.geometry]
type = "pose"`)."""
const TEXTURE_POSES_KEYS = ("images", "board_design", "board", "find_boards")
"The keys of `[texture]`: the split, the cells, the sampling, the CGLS, the coverage a cell needs, the start, the pinhole, the regions, the penalty weights."
const TEXTURE_KEYS = ("split_every", "cell_mm", "pixel_fraction", "cgls_max_iter", "cgls_rtol", "min_fit_views", "min_val_views", "min_test_views",
                      "min_pixels", "start", "pinhole_r_m", "seed", "centre_rho", "edge_rho", "regularization_weights")
"The keys of `[report]`: the scoring block, the bootstrap draws and their seed."
const TEXTURE_REPORT_KEYS = ("block_mm", "bootstrap", "seed")
"The keys of a `[figures.<name>]` table by its `kind`."
const TEXTURE_FIGURE_KEYS = Dict("closeup" => ("kind", "asset", "size_in", "dpi", "crop_mm", "titles"),
                                 "stacked" => ("kind", "asset", "size_in", "dpi", "block_mm", "lines_per_band", "white", "label_fraction", "variants", "labels",
                                               "photograph"),
                                 "page" => ("kind", "asset", "size_in", "dpi", "downsample", "error_range", "title"))
const TEXTURE_RENDER_KEYS = ("numbers_prefix", "significant_digits")

_check(ok::Bool, msg::AbstractString) = ok || throw(ArgumentError(msg))
_is_count(x) = x isa Integer && !(x isa Bool) && x >= 0
_is_number(x) = x isa Real && !(x isa Bool) && isfinite(x)

"""
    load_texture_recovery_config(path) -> Dict

The texture config of record, every table and key checked (`load_driver_config` for the tables, the inputs and the experiment's name):
one `[prepare.<set>]` (the texture photographs, its `raw` an input of stage prepare); `[upstream]` a case of the refinement config it names
(`load_radiometric_refinement_config`), on that case's photographs; `[poses]` the photographs and the board's design (stage detect), the sheet (stage init) and the
pose arm (BlobBoards' `[find_boards]` table, `geometry.type = "pose"`; the retired keys refused by name); `[texture]` (`TEXTURE_KEYS`) with its numbers in range — `split_every` ≥ 3, `cell_mm` > 0, 0 < `pixel_fraction` ≤ 1,
`cgls_max_iter` ≥ 1, 0 < `cgls_rtol` < 1, the `min_*` counts, a finite `start`, `pinhole_r_m` > 0, 0 ≤ `centre_rho` < `edge_rho` ≤ 1,
`regularization_weights` positive and strictly increasing; `[report]` (`TEXTURE_REPORT_KEYS`); every `[figures.<name>]` of a known
`kind` with that kind's keys; `[render]`.
"""
function load_texture_recovery_config(path::AbstractString)
    c = load_driver_config(path, "texture_recovery"; tables = TEXTURE_TABLES)
    for t in TEXTURE_TABLES
        t in TEXTURE_OPTIONAL_TABLES && !haskey(c, t) && continue
        haskey(c, t) && c[t] isa AbstractDict && !isempty(c[t]) || throw(ArgumentError("$path: no [$t] table"))
    end
    check_prepare_tables(c, path)
    length(c["prepare"]) == 1 || throw(ArgumentError("$path: one [prepare.<set>] table (the texture photographs), got $(length(c["prepare"]))"))
    for (s, t) in c["prepare"]
        r = get(c["inputs"], t["raw"], nothing)
        r !== nothing && r["stage"] == "prepare" || throw(ArgumentError("$path [prepare.$s]: raw = $(repr(t["raw"])) is not an input of stage prepare"))
    end
    u = c["upstream"]; check_required_keys(u, TEXTURE_UPSTREAM_KEYS, "$path [upstream]")
    all(k -> u[k] isa AbstractString && !isempty(u[k]), TEXTURE_UPSTREAM_KEYS) || throw(ArgumentError("$path [upstream]: every key a non-empty string"))
    rp = projectdir(u["refinement"])
    isfile(rp) || throw(ArgumentError("$path [upstream] refinement = $(repr(u["refinement"])): no such config of record"))
    ref = load_radiometric_refinement_config(rp)
    haskey(ref["cases"], u["case"]) || throw(ArgumentError("$path [upstream] case = $(repr(u["case"])) is not a case of $(u["refinement"])"))
    haskey(ref["prepare"], u["images"]) || throw(ArgumentError("$path [upstream] images = $(repr(u["images"])) is not a [prepare] set of $(u["refinement"])"))
    ref["cases"][u["case"]]["images"] == u["images"] ||
        throw(ArgumentError("$path [upstream] images = $(repr(u["images"])) is not the photographs of case $(u["case"]) ($(ref["cases"][u["case"]]["images"]))"))
    p = c["poses"]; w = "$path [poses]"
    p isa AbstractDict || throw(ArgumentError("$w must be a table"))
    check_required_keys(p, TEXTURE_POSES_KEYS, w)
    for (k, st) in (("images", "detect"), ("board_design", "detect"), ("board", "init"))
        i = get(c["inputs"], p[k], nothing)
        i !== nothing && i["stage"] == st || throw(ArgumentError("$w $k = $(repr(p[k])) is not an input of stage $st"))
    end
    fb = p["find_boards"]
    _check(fb isa AbstractDict && get(fb, "geometry", nothing) isa AbstractDict && get(fb["geometry"], "type", nothing) == "pose",
           "$w find_boards: BlobBoards' [find_boards] table on its pose arm ([poses.find_boards.geometry] type = \"pose\": the refinement's camera)")
    s = c["texture"]; w = "$path [texture]"; check_required_keys(s, TEXTURE_KEYS, w)
    _check(_is_count(s["split_every"]) && s["split_every"] >= 3, "$w split_every = $(repr(s["split_every"])): an integer ≥ 3 (a test, a validation and at least one fit photograph per cycle)")
    _check(_is_number(s["cell_mm"]) && s["cell_mm"] > 0, "$w cell_mm = $(repr(s["cell_mm"])): a positive length (mm)")
    _check(_is_number(s["pixel_fraction"]) && 0 < s["pixel_fraction"] <= 1, "$w pixel_fraction = $(repr(s["pixel_fraction"])): in (0, 1]")
    _check(_is_count(s["cgls_max_iter"]) && s["cgls_max_iter"] >= 1, "$w cgls_max_iter = $(repr(s["cgls_max_iter"])): an integer ≥ 1")
    _check(_is_number(s["cgls_rtol"]) && 0 < s["cgls_rtol"] < 1, "$w cgls_rtol = $(repr(s["cgls_rtol"])): in (0, 1)")
    for k in ("min_fit_views", "min_val_views", "min_test_views", "min_pixels", "seed")
        _check(_is_count(s[k]), "$w $k = $(repr(s[k])): an integer ≥ 0")
    end
    _check(_is_number(s["start"]), "$w start = $(repr(s["start"])): a finite number (the texture every solve starts from)")
    _check(_is_number(s["pinhole_r_m"]) && s["pinhole_r_m"] > 0, "$w pinhole_r_m = $(repr(s["pinhole_r_m"])): a positive radius (m)")
    _check(_is_number(s["centre_rho"]) && _is_number(s["edge_rho"]) && 0 <= s["centre_rho"] < s["edge_rho"] <= 1,
           "$w centre_rho = $(repr(s["centre_rho"])), edge_rho = $(repr(s["edge_rho"])): 0 ≤ centre_rho < edge_rho ≤ 1 (the image radius over the half-diagonal)")
    rw = s["regularization_weights"]
    _check(rw isa AbstractVector && !isempty(rw) && all(x -> _is_number(x) && x > 0, rw) && issorted(rw; lt = <=),
           "$w regularization_weights = $(repr(rw)): positive finite weights, strictly increasing")
    if haskey(c, "folds")
        f = c["folds"]; w = "$path [folds]"; check_required_keys(f, TEXTURE_FOLDS_KEYS, w)
        _check(_is_count(f["rotations"]) && 2 <= f["rotations"] <= c["texture"]["split_every"],
               "$w rotations = $(repr(f["rotations"])): an integer in 2 … [texture] split_every (each rotation shifts the split by one photograph)")
    end
    if haskey(c, "compare")
        m = c["compare"]; w = "$path [compare]"; m isa AbstractDict || throw(ArgumentError("$w must be a table"))
        VgtkCore.check_keys(m, (TEXTURE_COMPARE_KEYS..., "this"), w); foreach(k -> VgtkCore.required_key(m, k, w), TEXTURE_COMPARE_KEYS)
        _check(m["runs"] isa AbstractDict && all(v -> v isa AbstractString && occursin(r"^[0-9a-f]{16}$", v), values(m["runs"])),
               "$w runs: a table name → the 16-hex-digit key of a refine run of this results root")
        names = [collect(keys(m["runs"])); haskey(m, "this") ? [m["this"]] : String[]]
        _check(allunique(names) && length(names) >= 2, "$w: at least two distinct runs (runs, and this)")
        _check(m["reference"] isa AbstractString && m["reference"] in names, "$w reference = $(repr(m["reference"])): one of the runs ($(join(names, ", ")))")
        _check(m["crop_mm"] isa AbstractVector && length(m["crop_mm"]) == 4 && all(_is_number, m["crop_mm"]) && m["crop_mm"][3] > 0 && m["crop_mm"][4] > 0,
               "$w crop_mm: [x0, y0, width, height] in mm")
    end
    r = c["report"]; w = "$path [report]"; check_required_keys(r, TEXTURE_REPORT_KEYS, w)
    _check(_is_number(r["block_mm"]) && r["block_mm"] > 0, "$w block_mm = $(repr(r["block_mm"])): a positive length (mm)")
    _check(_is_count(r["bootstrap"]) && r["bootstrap"] >= 1, "$w bootstrap = $(repr(r["bootstrap"])): an integer ≥ 1")
    _check(_is_count(r["seed"]), "$w seed = $(repr(r["seed"])): an integer ≥ 0")
    for (f, t) in c["figures"]
        w = "$path [figures.$f]"
        t isa AbstractDict && haskey(t, "kind") || throw(ArgumentError("$w: a table with a kind ($(join(sort(collect(keys(TEXTURE_FIGURE_KEYS))), " | ")))"))
        haskey(TEXTURE_FIGURE_KEYS, t["kind"]) || throw(ArgumentError("$w kind = $(repr(t["kind"])): $(join(sort(collect(keys(TEXTURE_FIGURE_KEYS))), " | "))"))
        check_required_keys(t, TEXTURE_FIGURE_KEYS[t["kind"]], w)
        _check(t["asset"] isa AbstractString && endswith(t["asset"], ".png"), "$w asset = $(repr(t["asset"])): a .png name")
        _check(t["size_in"] isa AbstractVector && length(t["size_in"]) == 2 && all(x -> _is_number(x) && x > 0, t["size_in"]), "$w size_in: [width, height] in inches")
        _check(_is_count(t["dpi"]) && t["dpi"] >= 1, "$w dpi = $(repr(t["dpi"])): an integer ≥ 1")
        if t["kind"] == "closeup"
            _check(t["crop_mm"] isa AbstractVector && length(t["crop_mm"]) == 3 && all(_is_number, t["crop_mm"]) && t["crop_mm"][3] > 0, "$w crop_mm: [x0, y0, size] in mm")
            _check(t["titles"] isa AbstractVector && length(t["titles"]) == 1 + length(TEXTURE_VARIANTS) && all(x -> x isa AbstractString, t["titles"]),
                   "$w titles: one per panel (the design, then $(join(TEXTURE_VARIANTS, ", ")))")
        elseif t["kind"] == "stacked"
            _check(t["block_mm"] isa AbstractVector && length(t["block_mm"]) == 4 && all(_is_number, t["block_mm"]) && t["block_mm"][3] > 0 && t["block_mm"][4] > 0,
                   "$w block_mm: [x0, y0, width, height] in mm")
            _check(_is_count(t["lines_per_band"]) && t["lines_per_band"] >= 1, "$w lines_per_band = $(repr(t["lines_per_band"])): an integer ≥ 1")
            _check(_is_number(t["white"]) && 0 < t["white"] < 1, "$w white = $(repr(t["white"])): a design level in (0, 1)")
            _check(_is_number(t["label_fraction"]) && 0 < t["label_fraction"] < 1, "$w label_fraction = $(repr(t["label_fraction"])): the labels' share of the width, in (0, 1)")
            _check(t["variants"] isa AbstractVector && !isempty(t["variants"]) && all(v -> v isa AbstractString && v in TEXTURE_VARIANTS, t["variants"]) && allunique(t["variants"]),
                   "$w variants: distinct names among $(join(TEXTURE_VARIANTS, ", ")), a band each")
            _check(t["labels"] isa AbstractVector && length(t["labels"]) == 2 + length(t["variants"]) && all(x -> x isa AbstractString, t["labels"]),
                   "$w labels: one per band (the design, the photograph, then the variants)")
            _check(t["photograph"] isa AbstractString && !isempty(t["photograph"]), "$w photograph: the name of a fit photograph of the [poses] images")
        else
            _check(_is_count(t["downsample"]) && t["downsample"] >= 1, "$w downsample = $(repr(t["downsample"])): an integer ≥ 1")
            _check(_is_number(t["error_range"]) && t["error_range"] > 0, "$w error_range = $(repr(t["error_range"])): a positive number")
            _check(t["title"] isa AbstractString, "$w title: a string")
        end
    end
    check_required_keys(c["render"], TEXTURE_RENDER_KEYS, "$path [render]")
    _check(c["render"]["numbers_prefix"] isa AbstractString, "$path [render] numbers_prefix: a string (\"\" for no macros)")
    _check(_is_count(c["render"]["significant_digits"]) && c["render"]["significant_digits"] >= 1, "$path [render] significant_digits: an integer ≥ 1")
    return c
end

# ---- the sheet and the poses of its photographs --------------------------------------------------------------------------------------

"""
    page_frame(page_w, page_h) -> (𝐨, 𝐮, 𝐯, 𝐧f)

The sheet's frame in metres: origin at the design's top-left corner, x right along `𝐮`, y down along `𝐯`, the printed side's normal
`𝐧f = −(𝐮 × 𝐯)` (towards the camera).
"""
function page_frame(page_w::Real, page_h::Real)
    𝐨 = SVector(0.0, 0.0, 0.0); 𝐮 = SVector(Float64(page_w), 0.0, 0.0); 𝐯 = SVector(0.0, Float64(page_h), 0.0)
    return 𝐨, 𝐮, 𝐯, -normalize(cross(𝐮, 𝐯))
end

"""
    save_sheet_target(dir; page_mm, blobs_mm, texture = "texture.png") -> path

The sheet (the `[poses] board` input) as VisualGeometryToolkitIO's target record `dir/target.json` (`VgtkIO.save` of Core's
`PlanarTarget`): its nominal points the blobs (point `j` = BlobBoards blob index `j`, `blobs_mm[j]` its centre on the page, mm from
the page's top-left corner, x right, y down — the design chart), its `Canvas` the page (`page_mm` = width, height, at the origin)
with the page's image `dir/texture` as its `CanvasTexture` (the whole page, its SHA-256). Lengths decimal-exact (`read_length`).
"""
function save_sheet_target(dir::AbstractString; page_mm, blobs_mm::AbstractDict, texture::AbstractString = "texture.png")
    n = length(blobs_mm)
    sort(collect(keys(blobs_mm))) == collect(1:n) || throw(ArgumentError("save_sheet_target: the blobs are indexed 1…$n (the target's points)"))
    len(x) = VgtkIO.read_length(Float64(x), VgtkCore.mm)
    page = VgtkCore.Size2(width = len(page_mm[1]), height = len(page_mm[2])); o = SVector(len(0.0), len(0.0))
    sha = bytes2hex(open(SHA.sha256, joinpath(dir, texture)))
    t = VgtkCore.PlanarTarget([SVector(len(blobs_mm[j][1]), len(blobs_mm[j][2])) for j in 1:n];
                              canvas = VgtkCore.Canvas(page, o; texture = VgtkCore.CanvasTexture(texture, sha, page, o)))
    path = joinpath(dir, "target.json")
    VgtkIO.save(path, t; provenance = (; nominal_points = "point j: BlobBoards blob j's centre on the page, mm from its top-left corner, x right, y down"))
    return path
end

"""
    read_sheet(dir) -> NamedTuple

The sheet (the `[poses] board` input): its target record `dir/target.json` ([`save_sheet_target`](@ref), IO's
`load(AbstractTarget, …)`): the page (its canvas), the texel (the canvas texture's width over its image's columns) and per BlobBoards
blob index its centre on the page (mm from the page's top-left corner, x right, y down). Returns `(; page_w, page_h, texel_mm, blobs)`,
the page in metres, `blobs` a Dict index → position (mm).
"""
function read_sheet(dir::AbstractString)
    t = VgtkCore.ustrip(VgtkCore.m, VgtkIO.load(VgtkCore.AbstractTarget, joinpath(dir, "target.json")))   # plain metres
    cv = VgtkCore.canvas(t)
    (cv === nothing || cv.texture === nothing) && throw(ArgumentError("$dir/target.json: the sheet's target needs its page (a canvas with its texture)"))
    r = VgtkCore.Rect(cv); rt = VgtkCore.Rect(cv.texture); o = r.origin
    cols = size(PNGFiles.load(joinpath(dir, cv.texture.path)), 2)
    blobs = Dict(j => 1e3 .* (SVector{2, Float64}(p) .- SVector(o[1], o[2])) for (j, p) in enumerate(t.nominal))
    return (; page_w = Float64(r.widths[1]), page_h = Float64(r.widths[2]), texel_mm = 1e3 * Float64(rt.widths[1]) / cols, blobs)
end

"""
    board_finder(board_json, camera, table; gallery, device = RC.default_backend()) -> NamedTuple

BlobBoards' `find_boards` set up for the board of `board_json` (its design JSON, `load_board`) on its pose arm through `camera`, the
refinement's `BoundedCamera` held fixed: the configuration `FindBoardsConfig(table; camera)` (`table` BlobBoards' own `[find_boards]`
table, `[poses.find_boards]`, read by BlobBoards with the stack's strategy readers; its geometry the pose arm), the descriptor network of
BlobBoards' configuration (`descriptor_network_path()`; on the CPU its CPU provider) and the board's gallery features (`BoardGallery`,
cached in the file `gallery`). Returns `(; board, features, network, bundles, config, device)`.
"""
function board_finder(board_json::AbstractString, camera, table::AbstractDict; gallery::AbstractString, device = RC.default_backend())
    config = BlobBoards.FindBoardsConfig(table; camera)
    config.geometry isa BlobBoards.PoseGeometry || throw(ArgumentError("board_finder: the [find_boards] geometry is $(typeof(config.geometry)), not the pose arm (type = \"pose\")"))
    network = device isa BlobBoards.KernelAbstractions.CPU ?
        BlobBoards.DescriptorNetwork(BlobBoards.descriptor_network_path(); execution_provider = :cpu) : BlobBoards.DescriptorNetwork(BlobBoards.descriptor_network_path())
    board = BlobBoards.load_board(board_json)
    features = Dict(insert!(BlobBoards.BoardGallery(gallery, network), [board_json], network; device))
    return (; board, features, network, bundles = Dict(board.uid => board.bundle), config, device)
end

"""
    board_page_map(bundle, sheet) -> (; map, residuals_mm)

The board's frame on the sheet's page: the rigid map (VisualGeometryToolkitCore's `EuclideanMap`, mm) from BlobBoards' `:world` frame
of the board (`blobs(bundle, :world)`: centred, y up, z = 0, in the bundle's blob order) to the page's (mm from the design's top-left
corner, x right, y down, z = 0 — `page_frame`'s, in mm): Estimation's least-squares `fit_rigid_alignment` of the board's blobs onto the
sheet's blob table (`read_sheet`, by blob index), and its per-blob residuals (mm). The y flip is the rotation by π about x.
"""
function board_page_map(bundle, sheet)
    W = BlobBoards.blobs(bundle, :world)
    idx = sort([j for j in keys(sheet.blobs) if 1 <= j <= length(W)])
    length(idx) >= 3 || throw(ArgumentError("board_page_map: the sheet holds $(length(idx)) of the board's $(length(W)) blobs (≥ 3, not collinear)"))
    pairs = [SVector(Float64(VgtkCore.centroid(W[j])[1]), Float64(VgtkCore.centroid(W[j])[2]), 0.0) => SVector(sheet.blobs[j][1], sheet.blobs[j][2], 0.0)
             for j in idx]
    fit = VgtkEstimation.fit_rigid_alignment(pairs)
    return (; map = VgtkCore.estimate(fit), residuals_mm = fit.metadata.residuals)
end

"""
    sheet_poses(finder, photos; page, log) -> (views, residuals, positions, not_projected)

Per photograph of `photos` (`(name, path)`, a grey image of the camera's sensor) its pose (page → camera, `R`, `t` in metres, the page
frame `page_frame`'s) from its blobs alone: BlobBoards' `find_boards` (`finder`, [`board_finder`](@ref)) on its pose arm through the
refinement's camera — its board → camera pose (mm) taken onto the page by `page` ([`board_page_map`](@ref)) through Core's transforms,
`T = E ∘ page⁻¹` in metres (Core's units: `ustrip(m, with_unit(·, mm))`; the page frame `page_frame` is the page's in metres). A photograph on which BlobBoards accepts no pose of the board is left out
(logged). Returns per view a NamedTuple (`view`, `R`, `t`, `blobs` — BlobBoards' inliers that reproject —, `median_px`, `rms_px`,
`bearing_median_mrad`, `bearing_p90_mrad`, `camera_distance_m`), every inlier's reprojection residual (px) with its pixel, and the
count of inliers no projection converged for.
"""
function sheet_poses(finder, photos; page, log = println)
    camera = finder.config.geometry.camera
    views = NamedTuple[]; allr = Float64[]; XY = SVector{2, Float64}[]; nan = 0
    for (name, path) in photos
        img = VgtkFeatures.load_linear_image(path)[:, :, 1]             # Features' one linear reader: the grey channel
        VgtkCore.Size2(size(img, 2), size(img, 1)) == VgtkCore.extent(camera) ||
            throw(ArgumentError("sheet_poses: $path is $(size(img, 2)) × $(size(img, 1)) pixels, not the camera's sensor $(VgtkCore.extent(camera))"))
        acc = BlobBoards.find_boards(finder.features, finder.network, img; board_bundles = finder.bundles, config = finder.config, device = finder.device)
        i = findfirst(a -> a[1] == finder.board.uid, acc)
        if i === nothing
            log("  $name: BlobBoards accepts no pose of board $(finder.board.uid) — left out"); continue
        end
        _, E, _, cs = acc[i]
        T = VgtkCore.ustrip(VgtkCore.m, VgtkCore.with_unit(E ∘ inv(page), VgtkCore.mm))   # page (m) → camera (m)
        R = SMatrix{3, 3, Float64}(VgtkCore.rotation(T)); t = SVector{3, Float64}(VgtkCore.translation(T))
        r = Float64[]; ang = Float64[]; nn = 0
        for (src, dst, w) in zip(VgtkCore.sources(cs), VgtkCore.targets(cs), cs.weights)
            VgtkCore.is_map_inlier(w) || continue
            b = VgtkCore.centroid(src); p = SVector{2, Float64}(VgtkCore.centroid(dst))
            Xc = R * VgtkCore.ustrip.(VgtkCore.m, page(SVector(Float64(b[1]), Float64(b[2]), 0.0)) .* VgtkCore.mm) + t
            f = VgtkCore.backproject(camera, p); all(isfinite, f) && push!(ang, angle_between(normalize(Xc), normalize(SVector{3, Float64}(f))))
            q = RC._project_from(camera, [SVector{3, Float64}(Xc)], [p])[1]
            all(isfinite, q) ? (push!(r, norm(q - p)); push!(XY, p)) : (nn += 1)
        end
        nan += nn; append!(allr, r)
        push!(views, (; view = String(name), R, t, blobs = length(r), median_px = median(r), rms_px = sqrt(mean(abs2, r)),
                        bearing_median_mrad = 1e3 * median(ang), bearing_p90_mrad = 1e3 * quantile(ang, 0.9), camera_distance_m = norm(R' * t)))
    end
    return views, allr, XY, nan
end

"""
    write_poses(path, views, residuals, positions, not_projected; W, H, centre_rho, edge_rho, label) -> path

The poses record: per view `R_world_to_camera` (9 values, column-major) and `t_world_to_camera` (m) with its blob count and errors, and
the reprojection statistics over every blob, the centre (ρ < `centre_rho`) and the edge (ρ > `edge_rho`; ρ the image radius over the
half-diagonal).
"""
function write_poses(path::AbstractString, views, allr, XY, nan; W::Integer, H::Integer, centre_rho::Real, edge_rho::Real, label::AbstractString)
    ρ = [norm(p - SVector(W / 2, H / 2)) / hypot(W / 2, H / 2) for p in XY]
    per = [Dict{String, Any}("view" => v.view, "R_world_to_camera" => collect(vec(Matrix(v.R))), "t_world_to_camera" => collect(v.t),
                             "blobs" => v.blobs, "median_px" => v.median_px, "rms_px" => v.rms_px, "bearing_median_mrad" => v.bearing_median_mrad,
                             "bearing_p90_mrad" => v.bearing_p90_mrad, "camera_distance_m" => v.camera_distance_m) for v in views]
    VgtkIO.write_record(path, Dict{String, Any}("schema" => EXPERIMENT_RECORD_SCHEMAS.texture_poses, "label" => label, "views" => length(views), "all" => residual_summary(allr), "centre" => residual_summary(allr[ρ .< centre_rho]),
                                       "edge" => residual_summary(allr[ρ .> edge_rho]), "centre_rho" => centre_rho, "edge_rho" => edge_rho,
                                       "not_projected" => nan, "per_view" => per); allownan = true)
    return path
end

"The poses record read back: `[(name, R, t)]` (world → camera)."
function read_poses(path::AbstractString)
    P = VgtkIO.read_record(path; allownan = true)
    return [(String(pv.view), SMatrix{3, 3, Float64}(Float64.(collect(pv.R_world_to_camera))), SVector{3, Float64}(Float64.(collect(pv.t_world_to_camera))))
            for pv in P.per_view]
end

# ---- the camera held fixed and the scene -------------------------------------------------------------------------------------------

"""
    image_circle_of(camera) -> Union{Nothing, Ellipse}

The image circle of the refinement's radiometric camera (its state's `camera`: the lens coverage of its `BoundedCamera`, which the
state record holds in IO's bounded camera body), or `nothing` when the lens lights the whole sensor (the coverage is the sensor).
"""
function image_circle_of(camera::RC.RadiometricCameraModel)
    c = VgtkCore.lens_coverage(camera.geometric)
    c isa VgtkCore.Ellipse && return c
    c isa VgtkCore.Circle && return Ellipse(Point2(Float64(c.center[1]), Float64(c.center[2])), Float64(c.r), Float64(c.r), 0.0)
    return nothing
end

"""
    sheet_scene(camera::CalibratedCamera, target, sheet, design_png, images_dir, poses, W, H; characterisation = nothing) -> NamedTuple

The photographs in `images_dir` (`photo_file` of each pose's name) are read through RadiometricCamera's ONE entry point under the
camera's own observation model (`photo_set(camera, files; characterisation)`, `sample_pixels`; owner 2026-11-15: a camera predicts
only photographs read as it was calibrated): a camera calibrated on RAW photosites under a sensor record reads the RAW photographs at
their green photosites on the raw grid under that record — its black (the dark of the exposure), its defects masked —, the
`characterisation` the record's location (its sha256 checked against the camera's); a camera of decoded images their pixels, the
green channel. Another record, another kind of photograph, a frame the record does not describe: refused. No flag chooses.

What the page solve shares: the photographs (the `camera` — a `RadiometricCameraModel`, central — at the `poses`, camera → world)
and their decoded files (`images_dir/<name>`), the sheet as the scene's board (`page_frame`, a flat surface with the `target`'s
printed-edge parameters: the calibration board's warping is never evaluated on the sheet), its ray intersector, the vignetting, the
render settings (the calibration's own, `camera.settings`: the quadrature), the design (`design_png`, the green channel, as a `TW × TH` matrix [x, y]) and its texel size.
"""
function sheet_scene(cc::RC.CalibratedCamera, target, sheet, design_png::AbstractString, images_dir::AbstractString, poses, W::Integer, H::Integer;
                     characterisation::Union{Nothing, AbstractString} = nothing)
    isempty(poses) && throw(ArgumentError("sheet_scene: no poses"))
    cc.observation.channel in (:G, :RGB) ||
        throw(ArgumentError("sheet_scene: the page solve observes the green channel; the camera was calibrated on $(cc.observation.channel) alone"))
    camera = cc.model; settings = cc.settings
    𝐨, 𝐮, 𝐯, _ = page_frame(sheet.page_w, sheet.page_h)
    names = [String(p[1]) for p in poses]
    photos = RC.photo_set(cc, [RC.photo_file(images_dir, n) for n in names]; characterisation)   # THE entry point, the camera's own model
    cam_poses = [EuclideanMap(SMatrix{3, 3, Float64}(R'), SVector{3, Float64}(-R' * t); check = false) for (_, R, t) in poses]   # camera → world
    bd = RC.TexturedTarget(𝐨, 𝐮, 𝐯, fill(0.5, 4, 4, 3); facing = cam_poses[1].t)
    surf = RC.TargetState(; h = zero(target.warping), edge = RC.edge_parameters(target.radiometry))
    board = RC.with_state(bd, surf); rays = RC.BoardRays(board)
    img = VgtkFeatures.load_linear_image(design_png); TH, TW = size(img, 1), size(img, 2)
    design = Matrix{Float64}(permutedims(Float64.(img[:, :, 2])))                           # [x, y]
    page_mm = (1e3 * sheet.page_w, 1e3 * sheet.page_h); tex_mm = page_mm[1] / TW
    abs(page_mm[2] / TH - tex_mm) < 1e-6 * tex_mm || throw(ArgumentError("sheet_scene: the design's texels are not square ($(page_mm[1] / TW) × $(page_mm[2] / TH) mm)"))
    abs(tex_mm - sheet.texel_mm) < 1e-6 * tex_mm || throw(ArgumentError("sheet_scene: the design's texel $(tex_mm) mm is not the sheet target's $(sheet.texel_mm) mm"))
    return (; names, images_dir = String(images_dir), photos, camera, poses = cam_poses, bd, surf, board, rays, sett = settings, design, TW, TH,
            tex_mm, page_mm, W = Int(W), H = Int(H), learned_r = camera.aperture_radius)
end

"""
    variant_camera(ctx, name; pinhole_r) -> RadiometricCameraModel

The camera under the blur model `name`: `learned` the calibration's as it is; `pinhole` the aperture radius `pinhole_r` (the chief ray
alone), everything else the same.
"""
function variant_camera(ctx, name::AbstractString; pinhole_r::Real)
    c = ctx.camera
    name == "learned" && return c
    name == "pinhole" && return VrtkCore.with_aperture_radius(c, Float64(pinhole_r))
    throw(ArgumentError("variant_camera: unknown blur model $(repr(name)) ($(join(TEXTURE_VARIANTS, " | ")))"))
end

# ---- the pixels: the mask, the sample, the roles, the cells ------------------------------------------------------------------------

"""
    split_roles(K, every; fold = 0) -> Vector{Symbol}

The role of photographs 1 … K by index: `(k − fold) % every == 0` `:test`, `== 1` `:val`, the rest `:fit` (`every = 5`: 1/5 test,
1/5 validation, 3/5 fit). `fold = 0` is the split of record (Ties's held-out split, `k % every`); `fold = f` rotates every role by
`f` photographs, so the `every` folds `0 … every − 1` put each photograph once in the test and once in the validation set.
"""
function split_roles(K::Integer, every::Integer; fold::Integer = 0)
    every >= 3 || throw(ArgumentError("split_roles: every = $every (≥ 3: a test, a validation and at least one fit photograph per cycle)"))
    0 <= fold < every || throw(ArgumentError("split_roles: fold = $fold (0 … $(every - 1): a rotation of the split by `fold` photographs)"))
    return [(r = mod(k - fold, every); r == 0 ? :test : r == 1 ? :val : :fit) for k in 1:K]
end

"""
    sample_pixels(ctx, k, external; fraction, seed, backend) -> (; px, masked, values, weights)

Photograph `k`'s sampled pixels: the package's pixel mask (`compute_mask`: the chief ray and every footprint-corner × aperture-border ray
on the sheet, inside the image circle `external`), of which a `fraction` is kept (`StableRNG(seed + k)`); per pixel `(u, v, x_mm, y_mm)`
with its chief ray's page point. `masked` the mask's pixel count; the mask is one kernel on `backend`, required (the refine's device).
The photograph is read here, once, through the entry point under the camera's observation model (`observe(ctx.photos, k, mask)`): a
RAW photograph's usable green photosites (its record's defects masked) restrict the mask and `values` are their linear values under the
record (Core's `linearview` under the frame's sensor model; an uncertain value's value); a decoded photograph's `values` are its linear green. `values` are in `px` order: `run_page` reads them, never the photograph again.
`weights` are their rows' weights by the plane's element TYPE — RadiometricCamera's `masked_observations`, the calibration's one
weighting: an uncertain plane (a RAW photograph under a sensor model with a variance law) 1/σ per value, σ² Core's variance of the
value; a plain plane `RC.Unweighted()` (rows as measured).
"""
function sample_pixels(ctx, k::Integer, external; fraction::Real, seed::Integer, backend)
    mask = RC.compute_mask(ctx.camera, ctx.poses[k], ctx.board, ctx.sett; δ = 0.0, external, backend)
    plane, mask = RC.observe(ctx.photos, k, mask; name = ctx.names[k])   # the camera's observation model: its record's photosites, or the image's pixels
    size(plane, 1) == ctx.H && size(plane, 2) == ctx.W || error("sample_pixels: $(ctx.names[k]) is $(size(plane, 2)) × $(size(plane, 1)), the board $(ctx.W) × $(ctx.H)")
    rng = StableRNG(seed + k); px = NTuple{4, Float64}[]
    for (u, v) in RC.masked_pixels(mask)
        rand(rng) < fraction || continue
        o, d = RC.world_ray(ctx.camera, ctx.poses[k], RC.𝐩_of_pixel(u, v), SVector(0.0, 0.0, 0.0)); sr, 𝐱 = RC.intersect_board(ctx.rays, o, d)
        (isfinite(sr) && sr > 0) || continue
        𝐛 = RC.chart(ctx.bd, 𝐱); push!(px, (u, v, 𝐛[1] * ctx.page_mm[1], 𝐛[2] * ctx.page_mm[2]))
    end
    # a photosite's one value (and, uncertain, its 1/σ); a decoded photograph's linear green — RC's masked_observations, channel G
    values, weights, _ = RC.masked_observations(plane, [(Int(p[1]), Int(p[2])) for p in px]; channel = 2)
    return (; px, masked = count(mask), values, weights)
end

"""
    row_weights(weights, groups, n) -> Union{RC.Unweighted, Vector{Float64}}

The weights of the `n` sampled rows from each photograph's (`weights[k]`, `sample_pixels`' — one observation model, so one kind):
`RC.Unweighted()` when every photograph's rows are as measured, else the 1/σ of every row in `rows_px` order (`groups`,
`rows_by_photograph`). Dispatch on the weights' type, never a flag.
"""
row_weights(::AbstractVector{RC.Unweighted}, groups, n::Integer) = RC.Unweighted()
function row_weights(weights::AbstractVector{<:AbstractVector{<:Real}}, groups, n::Integer)
    w = fill(NaN, n)
    for (k, qs) in groups
        length(weights[k]) == length(qs) || throw(ArgumentError("row_weights: photograph $k has $(length(weights[k])) weights for $(length(qs)) sampled pixels"))
        w[qs] .= weights[k]
    end
    all(x -> isfinite(x) && x > 0, w) || throw(ArgumentError("row_weights: a row has no finite positive weight"))
    return w
end
row_weights(weights::AbstractVector, groups, n::Integer) =
    throw(ArgumentError("row_weights: the photographs' rows mix weighted and unweighted observations ($(unique(typeof.(weights)))): one observation model weights them all"))

"The record of a page solve's row weighting (`row_weights`): as measured, or whitened by the sensor record's variance law."
weighting_record(::RC.Unweighted) = "as measured (a plain observation: unweighted rows)"
weighting_record(::AbstractVector) = "whitened: each row and its value times 1/σ, σ² the sensor record's variance law at the photosite (RC's masked_observations, _whiten)"

"The sensor record a photo set's observation model states: its sha256, or nothing (no record)."
sensor_record_sha256(s::RC.SensorRecord) = s.sha256
sensor_record_sha256(::RC.NoSensorRecord) = nothing

"The cell `(i, j)` of a page point (mm) on the `cell_mm` grid from the design's top-left corner."
cell_of(x_mm::Real, y_mm::Real, cell_mm::Real) = (floor(Int, x_mm / cell_mm) + 1, floor(Int, y_mm / cell_mm) + 1)

"""
    cell_coverage(samples, roles, ncx, ncy, cell_mm) -> Array{Int, 3}

`cnt[i, j, k]`: the sampled pixels of photograph `k` whose chief ray falls in cell `(i, j)` (`samples[k]` the `px` of `sample_pixels`).
"""
function cell_coverage(samples, ncx::Integer, ncy::Integer, cell_mm::Real)
    cnt = zeros(Int, ncx, ncy, length(samples))
    for (k, px) in enumerate(samples), (_, _, x, y) in px
        i, j = cell_of(x, y, cell_mm)
        (1 <= i <= ncx && 1 <= j <= ncy) && (cnt[i, j, k] += 1)
    end
    return cnt
end

"""
    select_cells(cnt, roles; min_fit_views, min_val_views, min_test_views, min_pixels) -> [(i, j)]

The cells scored: at least `min_fit_views` fit, `min_val_views` validation and `min_test_views` test photographs each put at least
`min_pixels` sampled pixels into the cell.
"""
function select_cells(cnt::AbstractArray{<:Integer, 3}, roles::AbstractVector{Symbol}; min_fit_views::Integer, min_val_views::Integer,
                      min_test_views::Integer, min_pixels::Integer)
    nv(i, j, ρ) = count(k -> roles[k] == ρ && cnt[i, j, k] >= min_pixels, eachindex(roles))
    return [(i, j) for j in axes(cnt, 2) for i in axes(cnt, 1) if nv(i, j, :fit) >= min_fit_views && nv(i, j, :val) >= min_val_views && nv(i, j, :test) >= min_test_views]
end

# ---- the rows: every pixel's footprint over the page's texels ------------------------------------------------------------------------

"""
    texel_weights(𝐛, TW, TH) -> ((texel, weight) × 4) or nothing

The bilinear weights of the page position `𝐛` (∈ [0, 1]², the design's chart) over the `TW × TH` texels (texel `(i, j)` is column
`(j − 1) TW + i`, x fastest): Core's `bilinear_weights` (the half-texel chart of `bilinear_sample`), the position strictly inside the
texel centres' grid, `nothing` otherwise (Core's clamp would extend the edge; a footprint off the grid is dropped).
"""
@inline function texel_weights(𝐛, TW::Integer, TH::Integer)
    uv = SVector(𝐛[1], 𝐛[2])
    VgtkCore.is_inside_texel_centres(uv, (TW, TH)) || return nothing
    i, j, fx, fy = VgtkCore.bilinear_weights(uv, (TW, TH))
    tid(a, b) = (b - 1) * TW + a
    return ((tid(i, j), (1 - fx) * (1 - fy)), (tid(i + 1, j), fx * (1 - fy)), (tid(i, j + 1), (1 - fx) * fy), (tid(i + 1, j + 1), fx * fy))
end

"""
    _FootprintAt(view, board, st, vig, TW, TH, cols, vals)

The blur row of a sampled pixel `(q, u, v)` (`q` its index in the launch) under `view`, the renderer's quadrature: for every sensor
sample and every aperture-disc sample the ray `world_ray`, its intersection with the sheet and the bilinear texels there
(`texel_weights`), weighted by the sensor and aperture weights and accumulated per texel in sample order, normalised to a sum of 1 and
scaled by the chief ray's vignetting `V(𝐩)` (a pixel's model is I = V · Σ w x). Its `n` distinct texels go into the first `n` of the
`_footprint_slots(st)` slots from `(q − 1) · slots + 1` of `cols` (the texel, ascending) and `vals` (its weight); returns
`n`, or 0 when a sample misses the sheet or the grid (the pixel is dropped). A functor of Core's `pointwise!`: one KernelAbstractions
kernel on every backend, the CPU's included.
"""
struct _FootprintAt{V, S, T, G, C, W}
    view::V
    board::S
    st::T
    vig::G
    TW::Int32
    TH::Int32
    cols::C
    vals::W
end
Adapt.@adapt_structure _FootprintAt
"The slots of a pixel's blur row: 4 bilinear texels per sensor sample × aperture-disc sample of the render settings `st` (its most distinct texels)."
_footprint_slots(st) = 4 * length(st.sensor) * RC.n_disc(st.aperture)
@inline function (f::_FootprintAt)(q::Int32, u::Int32, v::Int32)
    st, view = f.st, f.view
    bd = f.board; rays = RC.BoardRays(f.board)
    TW, TH = Int(f.TW), Int(f.TH)
    base = (Int(q) - 1) * _footprint_slots(st); n = 0; tot = 0.0
    @inbounds for ks in 1:length(st.sensor)
        𝐩 = RC.𝐩_sample(st.sensor, ks, Int(u), Int(v)); rs = view.r; sw = st.sensor.weights[ks]
        for la in 1:RC.n_disc(st.aperture)
            𝐚 = RC.𝐚_sample(st.aperture, la, rs)
            o, d = RC.world_ray(view, 𝐩, 𝐚); sr, 𝐱 = RC.intersect_board(rays, o, d)
            (isfinite(sr) && sr > 0) || return Int32(0)
            tw = texel_weights(RC.chart(bd, 𝐱), TW, TH); tw === nothing && return Int32(0)
            w = sw * RC.𝐚_weight(st.aperture, la, rs)
            for (t, b) in tw                                           # the texel's slot (a linear search of the ≤ n so far), or a new one
                c = Int32(t); i = base + 1
                while i <= base + n && f.cols[i] != c
                    i += 1
                end
                if i > base + n
                    n += 1; f.cols[i] = c; f.vals[i] = 0.0
                end
                f.vals[i] += w * b
            end
            tot += w
        end
    end
    tot > 0 || return Int32(0)
    V = f.vig(RC.𝐩_of_pixel(Int(u), Int(v)))
    @inbounds for i in (base + 1):(base + n)
        f.vals[i] = V * (f.vals[i] / tot)
    end
    @inbounds for i in (base + 2):(base + n)                              # the row's texels ascending: the order of a sparse operator's row
        c = f.cols[i]; x = f.vals[i]; j = i - 1
        while j > base && f.cols[j] > c
            f.cols[j + 1] = f.cols[j]; f.vals[j + 1] = f.vals[j]; j -= 1
        end
        f.cols[j + 1] = c; f.vals[j + 1] = x
    end
    return Int32(n)
end

"The indices of `rows_px` (`(k, u, v)` each) by photograph `k`, each in order."
function rows_by_photograph(rows_px)
    g = Dict{Int, Vector{Int}}()
    for (q, r) in enumerate(rows_px)
        push!(get!(Vector{Int}, g, Int(r[1])), q)
    end
    return g
end

# The page's operators are built where the rows are computed (owner 2026-11-20: as the calibration's normal equations, nothing of
# the system's construction on the host). The footprint kernel leaves each sampled pixel's texels in its slots, ascending; the rows
# are compacted, selected, weighted and transposed by row-parallel kernels (Core's `_launch_rows!`: a work item per row, no atomics)
# and one device sort of the entries' keys, into Core's `CSRMatrix`es — the arrays Core's `PenalizedOperator` holds. The entries and
# their order are the host assembly's (`sparse(I, J, V)` of the same slots, then Core's upload), so the solves are the same numbers.

# Row `i` of a gather: its `cnt[i]` entries from `src[i] + 1 …` of (`colsrc`, `valsrc`) to `dst[i] + 1 …` of (`colout`, `valout`).
Base.@propagate_inbounds function _gather_row!(valout, i, colout, dst, src, cnt, colsrc, valsrc)
    d = dst[i]; s = src[i]
    for j in 1:cnt[i]
        colout[d + j] = colsrc[s + j]; valout[d + j] = valsrc[s + j]
    end
    return nothing
end
# Row `i`'s entries times its weight `w[i]` (RC's one weighting, `_whiten`: the same product as its sparse row scaling).
Base.@propagate_inbounds function _weight_row!(nzval, i, rowptr, w)
    for e in rowptr[i]:(rowptr[i + 1] - 1)
        nzval[e] = RC._whiten(w, i, nzval[e])
    end
    return nothing
end
# Row `i`'s entries' keys `(column − 1) · nz + entry`: ascending by column, then by entry (by row: the entries are stored row by row).
Base.@propagate_inbounds function _transpose_keys!(keys, i, rowptr, colval, nz)
    for e in rowptr[i]:(rowptr[i + 1] - 1)
        keys[e] = (colval[e] - 1) * nz + e
    end
    return nothing
end
# Row `i`'s entries' row index (the transpose's column values, before the sort).
Base.@propagate_inbounds function _entry_rows!(rows, i, rowptr)
    for e in rowptr[i]:(rowptr[i + 1] - 1)
        rows[e] = i
    end
    return nothing
end
# Entry `e` of the transpose: the entry `(keys[e] − 1) % nz + 1` of the operator (its row, its value).
Base.@propagate_inbounds function _transposed_entry!(nzout, e, rowout, keys, nz, rows, nzval)
    s = (keys[e] - 1) % nz + 1
    rowout[e] = rows[s]; nzout[e] = nzval[s]
    return nothing
end
# Column `j`'s first entry among the sorted `keys`: 1 + the keys below `(j − 1) · nz + 1` (a binary search).
Base.@propagate_inbounds function _column_start!(ptr, j, keys, nz)
    t = (j - 1) * nz + 1; lo = 1; hi = length(keys) + 1
    while lo < hi
        mid = (lo + hi) >>> 1
        keys[mid] < t ? (lo = mid + 1) : (hi = mid)
    end
    ptr[j] = lo
    return nothing
end

"The row pointers of rows with `cnt[i]` entries each (`cnt` on a device): `1`, then one past each row's last entry."
_row_pointers(cnt::AbstractVector{Int}, device) = vcat(Adapt.adapt(device, [1]), cumsum(cnt) .+ 1)

"""
    csr_rows(C, sel; device) -> CSRMatrix

The rows `sel` (indices, any order) of the device operator `C` (Core's `CSRMatrix`) as an operator of its own on `device`: row `r`
is `C`'s row `sel[r]`, its entries in their stored order (one gather kernel, a work item per row).
"""
function csr_rows(C::VgtkCore.CSRMatrix, sel::AbstractVector{<:Integer}; device)
    n = length(sel)
    seld = Adapt.adapt(device, Vector{Int}(sel))
    src = C.rowptr[seld] .- 1
    cnt = C.rowptr[seld .+ 1] .- 1 .- src
    ptr = _row_pointers(cnt, device); nz = n == 0 ? 0 : Int(sum(cnt))
    colval = VgtkCore.KA.zeros(device, Int, nz); nzval = VgtkCore.KA.zeros(device, Float64, nz)
    VgtkCore._launch_rows!(_gather_row!, nzval, n, colval, ptr .- 1, src, cnt, C.colval, C.nzval)
    return VgtkCore.CSRMatrix(n, C.n, ptr, colval, nzval)
end

"""
    csr_transpose(C; device) -> CSRMatrix

The transpose of the device operator `C` (Core's `CSRMatrix`) as an operator on `device`: its rows `C`'s columns, each row's entries
by `C`'s row, ascending — the compressed columns of the matrix, as Core's `PenalizedOperator` holds a transpose. One sort of the
entries' keys on the device, the rest row-parallel kernels (no atomics, so the result is the same on every backend).
"""
function csr_transpose(C::VgtkCore.CSRMatrix; device)
    nz = length(C.nzval)
    keys = VgtkCore.KA.zeros(device, Int, nz); rows = VgtkCore.KA.zeros(device, Int, nz)
    VgtkCore._launch_rows!(_transpose_keys!, keys, C.m, C.rowptr, C.colval, nz)
    VgtkCore._launch_rows!(_entry_rows!, rows, C.m, C.rowptr)
    sort!(keys)
    colval = VgtkCore.KA.zeros(device, Int, nz); nzval = VgtkCore.KA.zeros(device, Float64, nz)
    VgtkCore._launch_rows!(_transposed_entry!, nzval, nz, colval, keys, nz, rows, C.nzval)
    ptr = VgtkCore.KA.zeros(device, Int, C.n + 1)
    VgtkCore._launch_rows!(_column_start!, ptr, C.n + 1, keys, nz)
    return VgtkCore.CSRMatrix(C.n, C.m, ptr, colval, nzval)
end

"The device operator `C` (Core's `CSRMatrix`) as a host sparse matrix (the tests and small pages read rows of it)."
host_sparse(C::VgtkCore.CSRMatrix) =
    sparse(transpose(SparseMatrixCSC(C.n, C.m, Vector{Int}(Array(C.rowptr)), Vector{Int}(Array(C.colval)), Vector{Float64}(Array(C.nzval)))))

"""
    page_footprints(ctx, rows_px, camera, groups; device) -> (; A, kept)

The blur rows (`_FootprintAt`) of every sampled pixel `rows_px[q] = (k, u, v)` of `camera` (a `RadiometricCameraModel`: the blur model,
`variant_camera`) at photograph `k`'s pose `ctx.poses[k]`, on `device` and kept there: one footprint kernel per photograph (`groups`,
`rows_by_photograph`; Core's `pointwise!` on `OnBackend`), its slots compacted by a gather kernel. `A` is Core's `CSRMatrix` of
`length(rows_px)` rows in `rows_px` order over the page's `TW · TH` texels, each row's texels ascending (a texel its samples share
summed); a pixel with a sample off the sheet is dropped: an empty row, `kept[q]` false.
"""
function page_footprints(ctx, rows_px, camera::RC.RadiometricCameraModel, groups::AbstractDict{<:Integer, <:AbstractVector{<:Integer}}; device)
    N = _footprint_slots(ctx.sett); nT = ctx.TW * ctx.TH; KA = VgtkCore.KA
    order = Int[]; cnts = Any[]; cols = Any[]; vals = Any[]
    for k in sort!(collect(keys(groups)))
        qs = groups[k]; m = length(qs)
        m == 0 && continue
        slot_cols = KA.zeros(device, Int32, N * m); slot_vals = KA.zeros(device, Float64, N * m); n = KA.zeros(device, Int32, m)
        f = _FootprintAt(RC.View(ctx.poses[k], camera), ctx.board, ctx.sett, camera.vignetting, Int32(ctx.TW), Int32(ctx.TH), slot_cols, slot_vals)   # the kernel's view: the camera at pose k
        VgtkCore.pointwise!(f, n, Adapt.adapt(device, Int32.(1:m)), Adapt.adapt(device, Int32[rows_px[q][2] for q in qs]),
                            Adapt.adapt(device, Int32[rows_px[q][3] for q in qs]); exec = VgtkCore.OnBackend(device))
        KA.synchronize(device)
        cnt = Int.(n); dst = cumsum(cnt) .- cnt; nz = Int(sum(cnt))
        colk = KA.zeros(device, Int, nz); valk = KA.zeros(device, Float64, nz)
        VgtkCore._launch_rows!(_gather_row!, valk, m, colk, dst, Adapt.adapt(device, collect(0:N:(N * (m - 1)))), cnt, slot_cols, slot_vals)
        append!(order, qs); push!(cnts, cnt); push!(cols, colk); push!(vals, valk)
    end
    length(order) == length(rows_px) || throw(ArgumentError("page_footprints: the groups hold $(length(order)) of the $(length(rows_px)) sampled pixels"))
    if isempty(cnts)
        A = VgtkCore.CSRMatrix(0, nT, Adapt.adapt(device, [1]), KA.zeros(device, Int, 0), KA.zeros(device, Float64, 0))
        return (; A, kept = falses(0))
    end
    cnt = reduce(vcat, cnts)
    A = VgtkCore.CSRMatrix(length(order), nT, _row_pointers(cnt, device), reduce(vcat, cols), reduce(vcat, vals))
    order == eachindex(rows_px) || (A = csr_rows(A, invperm(order); device))       # rows in rows_px order (already, when rows_px runs by photograph)
    ptr = Array(A.rowptr)
    return (; A, kept = BitVector(ptr[q + 1] > ptr[q] for q in eachindex(rows_px)))
end

"""
    page_rows(ctx, rows_px, camera, blocks, groups; device) -> (; blocks, kept)

The blur rows of every sampled pixel ([`page_footprints`](@ref), on `device`) as host sparse matrices: `blocks` names the row sets
wanted, each a Bool per sampled pixel; per block `(; A, rows)`: its sparse matrix (`TW · TH` columns; rows in `rows_px` order) and
`rows`, the indices of `rows_px` it holds (the block's kept pixels). The refine stage keeps the rows on the device (`run_page`); this
form is the tests' and a small page's.
"""
function page_rows(ctx, rows_px, camera::RC.RadiometricCameraModel, blocks::NamedTuple, groups::AbstractDict{<:Integer, <:AbstractVector{<:Integer}}; device)
    P = page_footprints(ctx, rows_px, camera, groups; device)
    out = map(blocks) do b
        rows = findall(b .& P.kept)
        (; A = host_sparse(csr_rows(P.A, rows; device)), rows)
    end
    return (; blocks = out, kept = P.kept)
end

"""
    fit_operator(A, rows, w, D; device) -> PenalizedOperator

Core's `PenalizedOperator` of the fit rows `rows` of the page's device rows `A` (`page_footprints`) under their weights `w` (RC's one
weighting: `Unweighted()`, or 1/σ per fit row) and the penalty operator `D` (a host sparse matrix, uploaded): the rows selected,
weighted and transposed on `device` (`csr_rows`, `csr_transpose`) — the arrays Core's constructor makes from the host matrix.
"""
function fit_operator(A::VgtkCore.CSRMatrix, rows::AbstractVector{<:Integer}, w, D::SparseMatrixCSC; device)
    Af = csr_rows(A, rows; device)
    w isa RC.Unweighted || VgtkCore._launch_rows!(_weight_row!, Af.nzval, Af.m, Af.rowptr, Adapt.adapt(device, Vector{Float64}(w)))
    pen = VgtkCore.PenalizedOperator(SparseMatrixCSC{Float64, Int}(undef, 0, size(D, 2)), D; device)       # the penalty's two operators, Core's upload
    return VgtkCore.PenalizedOperator(Af, csr_transpose(Af; device), pen.D, pen.Dt, pen.device)
end

# ---- the solve: Core's CGLS on the page, on the host or the device -----------------------------------------------------------------

"""
    page_solves(op, b, x0; scale, weights, rtol, maxit) -> (; main, reg)

The page's textures by Core's `cgls` on the penalized operator `op` (Core's `PenalizedOperator` of the fit rows and the first
differences, `difference_rows((TW, TH), 1)`, on its device): `main` the unregularised one (penalty weight 0) and `reg[k]` the one of
`min ‖A x − b‖² + λ_k ‖∇x‖²` for every β of `weights`, `λ_k = β_k · scale` (`scale` Core's `penalty_scale` of the fit rows' energy, the
texels and the operator's mean positive eigenvalue, so β is the stack's normalized weight), each `(; x, iterations, converged, relres)`
with `x` on the host. `x0` the start; Core's stopping rule (relative to the normal equations' right-hand side). A regularised solve
is Jacobi-preconditioned by the normal matrix's diagonal `diag(AᵀA) + λ diag(DᵀD)` (`column_sumsq`; Core's `diagonal`), the
unregularised one never: without the penalty the problem is ill-posed in the texels the rows barely see, and the scaling moves those
by 10²–10³ (where the unpreconditioned iteration leaves them at the start). The 1 + |weights|
solves are independent and run at once, each on a worker of its own (RC's `_each_chunk`: on a device, its own stream), `op`'s arrays
shared read-only by them (RC's `_shared_read_only!`).
"""
function page_solves(op, b, x0; scale::Real, weights, rtol::Real, maxit::Integer)
    ws = [0.0; [β * scale for β in weights]]                           # the unregularised solve, then one per weight
    be = op.device
    dA, dD = column_sumsq(op.At; device = be), column_sumsq(op.Dt; device = be)   # diag(AᵀA), diag(DᵀD): the Jacobi diagonals' parts
    RC._shared_read_only!(be, (op.A, op.At, op.D, op.Dt, dA, dD))              # read by every solve at once (RC's CUDA extension: no per-buffer sync)
    sols = RC._each_chunk(be, ws) do w
        s = VgtkCore.cgls(op, b; w, x0, rtol, max_iter = maxit, diagonal = w > 0 ? dA .+ w .* dD : nothing)
        merge(s, (; x = Array(s.x)))
    end
    return (; main = sols[1], reg = sols[2:end])
end

"""
    column_sumsq(Ct; device) -> vector

The squared column norms of the operator whose transpose is the device operator `Ct` (Core's `CSRMatrix`): `Ct`'s rows' sums of
squares, one row kernel on `device`, so the diagonal of `CᵀC` (`page_solves`' Jacobi diagonal, `diag(AᵀA + λ DᵀD)`).
"""
function column_sumsq(Ct::VgtkCore.CSRMatrix; device)
    out = VgtkCore.KA.allocate(device, Float64, Ct.m)                  # every entry written
    VgtkCore._launch_rows!(_row_sumsq!, out, Ct.m, Ct.rowptr, Ct.nzval)
    return out
end
# Row `i`'s sum of squares.
Base.@propagate_inbounds function _row_sumsq!(out, i, rowptr, nzval)
    acc = zero(eltype(out))
    for e in rowptr[i]:(rowptr[i + 1] - 1)
        acc += nzval[e]^2
    end
    out[i] = acc
    return nothing
end

# ---- the refine stage: the page under both blur models --------------------------------------------------------------------------------

"The files of blur model `v` in a page run: the unregularised texture, the regularised ones (one column per weight), the texels its fit rows read, its record."
page_files(dir::AbstractString, v::AbstractString) = (; texture = joinpath(dir, "page_$v.f32"), reg = joinpath(dir, "page_$(v)_reg.f32"),
                                                      seen = joinpath(dir, "page_$(v)_seen.u8"), record = joinpath(dir, "page_$v.json"))

"""
    run_page(ctx, samples, roles, cells, circle, out; settings, variants, device, values, log)

The refine stage: every sampled pixel's row (`samples`, `page_footprints`: on `device`, where they stay) under each blur model of `variants`
(`variant_camera`), the fit rows' operator built there (`fit_operator`: Core's `PenalizedOperator` on the GPU's `CUDABackend()`, or the CPU) and `page_solves` (the unregularised texture and
every weight of `settings["regularization_weights"]`); the squared errors of every solution on the fit, validation and test rows that
fall in the scored `cells` (the test also at the centre ρ < `centre_rho` and the edge ρ > `edge_rho` of the image) with the photographs'
sums of squares. The observed values are `values[k]`: `sample_pixels`' values of photograph `k` at its sampled pixels, read when its mask
was made (each photograph read once, by its type). Writes `page.json` (the run's record: the grid, the cells,
the photographs and their roles, the weights, the image circle), `design.f32`, and per blur model its `page_files`.
"""
function run_page(ctx, samples, roles::AbstractVector{Symbol}, cells, circle, out::AbstractString; settings, variants = TEXTURE_VARIANTS, device,
                  values, weights, log = println)
    s = settings; mkpath(out); TW, TH = ctx.TW, ctx.TH
    W8 = Float64.(collect(s["regularization_weights"])); rtol = Float64(s["cgls_rtol"]); maxit = Int(s["cgls_max_iter"])
    cell_mm = Float64(s["cell_mm"]); cr, ce = Float64(s["centre_rho"]), Float64(s["edge_rho"]); PINR = Float64(s["pinhole_r_m"])
    cellset = Set(cells)
    rows_px = Tuple{Int, Int, Int}[]; scored = Bool[]; ρ = Float64[]
    for (k, px) in enumerate(samples), (u, v, x, y) in px
        push!(rows_px, (k, Int(u), Int(v))); push!(scored, cell_of(x, y, cell_mm) in cellset)
        push!(ρ, hypot(u - 0.5 - ctx.W / 2, v - 0.5 - ctx.H / 2) / (0.5 * hypot(ctx.W, ctx.H)))
    end
    groups = rows_by_photograph(rows_px)                               # the rows of each photograph, once
    yv = zeros(length(rows_px))
    for k in sort!(collect(keys(groups)))                              # each photograph's values at its sampled pixels (sample_pixels)
        qs = groups[k]
        length(values[k]) == length(qs) || throw(ArgumentError("run_page: photograph $k has $(length(values[k])) values for $(length(qs)) sampled pixels"))
        yv[qs] .= values[k]
    end
    rw = row_weights(map(identity, weights), groups, length(rows_px))   # the rows' 1/σ, or Unweighted (the observation model's type)
    role = [roles[r[1]] for r in rows_px]
    isfit, isval, istest = role .== :fit, role .== :val, role .== :test
    blocks = (; fit = isfit, fit_scored = isfit .& scored, val_scored = isval .& scored, test_scored = istest .& scored)
    write(joinpath(out, "design.f32"), Float32.(vec(ctx.design)))
    meta = Dict{String, Any}("TW" => TW, "TH" => TH, "tex_mm" => ctx.tex_mm, "page_mm" => collect(ctx.page_mm), "image_size" => [ctx.W, ctx.H],
                             "cell_mm" => cell_mm, "cells" => [[i, j] for (i, j) in cells], "photographs" => ctx.names, "roles" => String.(roles),
                             "variants" => collect(variants), "regularization_weights" => W8, "cgls_max_iter" => maxit, "cgls_rtol" => rtol,
                             "pinhole_r_m" => PINR, "learned_r_m" => Float64(ctx.learned_r), "centre_rho" => cr, "edge_rho" => ce, "rows_sampled" => length(rows_px),
                             "rows_scored" => count(scored), "weighting" => weighting_record(rw),
                             "sensor_record_sha256" => sensor_record_sha256(ctx.photos.observation.sensor),
                             "sensor_models" => [Dict{String, Any}(String(k) => v for (k, v) in pairs(r)) for r in RC.sensor_record(ctx.names, ctx.photos.models)],
                             "image_circle" => circle === nothing ? nothing :
                                 Dict("center" => [circle.center[1], circle.center[2]], "semi_axes" => [circle.a, circle.b]))
    VgtkIO.write_record(joinpath(out, "page.json"), merge(Dict{String, Any}("schema" => EXPERIMENT_RECORD_SCHEMAS.texture_page), meta); allownan = true)
    log(@sprintf("page: %d × %d texels (%.3f mm), %d of %d cells of %.1f mm scored, %d photographs (fit %d, validation %d, test %d), %d sampled pixels (%d in scored cells)",
                 TW, TH, ctx.tex_mm, length(cells), ceil(Int, ctx.page_mm[1] / cell_mm) * ceil(Int, ctx.page_mm[2] / cell_mm), cell_mm, length(roles),
                 count(==(:fit), roles), count(==(:val), roles), count(==(:test), roles), length(rows_px), count(scored)))
    D = VgtkCore.difference_rows((TW, TH), 1); x0 = fill(Float64(s["start"]), TW * TH)
    m̄ = VgtkCore.mean_positive_eigenvalue(D)                          # the operator's normalization of the weight (Core's)
    for name in variants
        t1 = time()
        R = page_footprints(ctx, rows_px, variant_camera(ctx, name; pinhole_r = PINR), groups; device)   # every row, on the device
        F, Fs, Vs, Ts = (findall(b .& R.kept) for b in blocks)         # indices of rows_px: the fit rows, the scored fit, validation and test rows
        nkept = count(R.kept); nval = count(R.kept .& isval); ntest = count(R.kept .& istest)
        wF = RC._select_rows(rw, F)                                    # the fit rows' weights (RC's one weighting: _whiten)
        bw = RC._whiten(wF, yv[F])                                     # the values whitened as the rows, or as measured
        t2 = time(); op = fit_operator(R.A, F, wF, D; device)          # the fit rows selected, weighted and transposed on the device
        scale = VgtkCore.penalty_scale(sum(abs2, Array(op.At.nzval)), TW * TH, m̄)   # λ = β · scale, the fit rows' energy in the cost's units
        log(@sprintf("page %s: %d rows kept (fit %d, validation %d, test %d), %d fit nonzeros, m̄ %.4g, λ/β %.4g, rows %s  [%.0f s]", name, nkept, length(F), nval, ntest,
                     length(op.A.nzval), m̄, scale, weighting_record(rw), time() - t1))
        sw = page_solves(op, bw, x0; scale, weights = W8, rtol, maxit)
        colptr = Array(op.At.rowptr); op = nothing
        log(@sprintf("page %s: unregularised %d its%s; Tikhonov %s  [%.0f s]", name, sw.main.iterations, sw.main.converged ? "" : " (NOT converged)",
                     join([@sprintf("%g:%d", λ, r.iterations) for (λ, r) in zip(W8, sw.reg)], " "), time() - t2))
        mc = ρ[Ts] .< cr; me = ρ[Ts] .> ce
        yd = Adapt.adapt(device, yv); pred = similar(yd)
        function record(x, iters, converged, weight)
            mul!(pred, R.A, Adapt.adapt(device, x)); e = Array(pred .- yd)    # every row's error A x − y, the rows as measured (a dropped pixel's unread)
            ef = e[Fs]; ev = e[Vs]; et = e[Ts]
            Dict{String, Any}("weight" => weight, "iters" => iters, "converged" => converged, "fit_sse" => sum(abs2, ef), "val_sse" => sum(abs2, ev),
                              "test_sse" => sum(abs2, et), "test_sse_centre" => sum(abs2, et[mc]), "test_sse_edge" => sum(abs2, et[me]))
        end
        recs = [record(sw.main.x, sw.main.iterations, sw.main.converged, 0.0); [record(r.x, r.iterations, r.converged, λ) for (λ, r) in zip(W8, sw.reg)]]
        seen = [colptr[j + 1] > colptr[j] for j in 1:(TW * TH)]                # the texels the fit rows read
        files = page_files(out, name)
        write(files.texture, Float32.(sw.main.x)); write(files.reg, Float32.(reduce(hcat, [r.x for r in sw.reg]; init = zeros(TW * TH, 0)))); write(files.seen, UInt8.(seen))
        VgtkIO.write_record(files.record, Dict{String, Any}("schema" => EXPERIMENT_RECORD_SCHEMAS.texture_page_variant, "variant" => name, "rows" => nkept, "fit_rows" => length(F), "val_rows" => nval, "test_rows" => ntest,
                                                    "scored_rows" => Dict("fit" => length(Fs), "validation" => length(Vs), "test" => length(Ts), "test_centre" => count(mc), "test_edge" => count(me)),
                                                    "fit_sumsq" => sum(abs2, yv[Fs]), "val_sumsq" => sum(abs2, yv[Vs]), "test_sumsq" => sum(abs2, yv[Ts]),
                                                    "test_sumsq_centre" => sum(abs2, yv[Ts][mc]), "test_sumsq_edge" => sum(abs2, yv[Ts][me]), "relres" => sw.main.relres,
                                                    "solutions" => recs); allownan = true)
        log(@sprintf("page %s: test rel-RMS unregularised %.4f; by weight %s  [%.0f s]", name, sqrt(recs[1]["test_sse"] / max(sum(abs2, yv[Ts]), 1e-300)),
                     join([@sprintf("%g:%.4f", r["weight"], sqrt(r["test_sse"] / max(sum(abs2, yv[Ts]), 1e-300))) for r in recs[2:end]], " "), time() - t1))
    end
    return nothing
end

"Whether a page run `dir` holds every blur model's files."
page_complete(dir::AbstractString, variants = TEXTURE_VARIANTS) = isfile(joinpath(dir, "page.json")) && isfile(joinpath(dir, "design.f32")) &&
    all(v -> all(isfile, values(page_files(dir, v))), variants)

# ---- the evaluate stage: the report ----------------------------------------------------------------------------------------------------

"The affine map `a · d + b` of the design `d` onto the recovered `x` (least squares) and the RMS of `x` about it in design units (÷ a)."
function affine_fit(x::AbstractVector, d::AbstractVector)
    A = hcat(Float64.(d), ones(length(d))); ab = A \ Float64.(x); a, b = ab[1], ab[2]
    return (; a, b, rms = sqrt(mean(abs2, Float64.(x) .- (a .* d .+ b))) / a)
end
"The means over whole `bs × bs` blocks of the matrix `A` (the trailing remainder dropped)."
function block_means(A::AbstractMatrix, bs::Integer)
    n1, n2 = size(A, 1) ÷ bs, size(A, 2) ÷ bs
    return [mean(@view A[(i - 1) * bs + 1:i * bs, (j - 1) * bs + 1:j * bs]) for i in 1:n1, j in 1:n2]
end
"The NaN-aware means of `A` over `ds × ds` cells (a cell with no finite value stays NaN; the trailing partial cells kept)."
function nanbox(A::AbstractMatrix, ds::Integer)
    ds == 1 && return Matrix{Float64}(A)
    return [(b = @view A[(i - 1) * ds + 1:min(i * ds, size(A, 1)), (j - 1) * ds + 1:min(j * ds, size(A, 2))]; v = filter(isfinite, vec(b)); isempty(v) ? NaN : mean(v))
            for i in 1:cld(size(A, 1), ds), j in 1:cld(size(A, 2), ds)]
end

"The texels of the scored `cells` (a `TW × TH` Bool matrix)."
function cells_mask(cells, TW::Integer, TH::Integer, tex_mm::Real, cell_mm::Real)
    q = falses(TW, TH)
    for (i, j) in cells
        x0 = max(1, round(Int, (i - 1) * cell_mm / tex_mm) + 1); x1 = min(TW, round(Int, i * cell_mm / tex_mm))
        y0 = max(1, round(Int, (j - 1) * cell_mm / tex_mm) + 1); y1 = min(TH, round(Int, j * cell_mm / tex_mm))
        q[x0:x1, y0:y1] .= true
    end
    return q
end

"""
    texture_scores(X, sel, design, bs) -> (; a, b, rms_texel, rms_block, n_texels, n_blocks, blocks)

The texture `X` (`TW × TH`) against the `design` over the texels `sel`: ONE affine map a · design + b (`affine_fit`), its RMS in design
units at the texel size and over the `bs × bs` blocks fully inside `sel` (`rms_block`); `blocks` the per-block squared errors (NaN
outside) for the per-cell scores.
"""
function texture_scores(X::AbstractMatrix, sel::AbstractMatrix{Bool}, design::AbstractMatrix, bs::Integer)
    q = findall(sel); af = affine_fit(X[q], design[q])
    sb = block_means(Float64.(sel), bs) .> 0.999
    xb = block_means(X, bs); db = block_means(design, bs); e2 = (xb .- (af.a .* db .+ af.b)) .^ 2 ./ af.a^2
    blocks = [sb[i, j] ? e2[i, j] : NaN for i in axes(e2, 1), j in axes(e2, 2)]
    return (; af.a, af.b, rms_texel = af.rms, rms_block = sqrt(mean(e2[sb])), n_texels = length(q), n_blocks = count(sb), blocks)
end

"The texel rectangle of cell `(i, j)` in block units (the blocks whose first texel lies in the cell)."
function cell_blocks(i::Integer, j::Integer, bs::Integer, tex_mm::Real, cell_mm::Real, nb1::Integer, nb2::Integer)
    t(x) = round(Int, x * cell_mm / tex_mm)
    b0 = cld(t(i - 1) + 1, bs); b1 = min(nb1, fld(t(i) + bs - 1, bs)); c0 = cld(t(j - 1) + 1, bs); c1 = min(nb2, fld(t(j) + bs - 1, bs))
    return b0:b1, c0:c1
end

"""
    paired_cells_ci(d; settings) -> (lo, hi)

The 95 % interval of the mean of the paired per-cell differences `d` (one blur model or run minus another, each cell's block RMS):
VisualGeometryToolkitEval's `bootstrap_ci` (the stack's one interval) over the cells, `settings["bootstrap"]` draws of
`StableRNG(settings["seed"])` (the `[report]` table). The one paired interval of the texture scores (`report_page`, `compare_page_runs`).
"""
paired_cells_ci(d::AbstractVector{<:Real}; settings) =
    bootstrap_ci(mean, StableRNG(Int(settings["seed"])), d; n_resamples = Int(settings["bootstrap"]), alpha = 0.05)

"""
    report_page(dir; settings, log) -> (report, textures)

The scores of a page run: per blur model the curve over the weights λ ∈ {0 (unregularised), the `regularization_weights`} — the validation
and test rel-RMS (√(SSE / Σ y²) over the test rows in the scored cells; the test also at the centre and the edge), the texture against
the design over the scored cells' texels the fit rows read (`texture_scores`: one affine map, the RMS at the texel and at `block_mm`),
the CGLS iterations — and the weight of record λ*, the least validation rel-RMS (the `photos` rule: blind to the design and the test
photographs); every score reported at λ*. Paired over the scored cells (each cell's block RMS at λ*): learned − pinhole, the mean with a
BCa interval over the cells (VisualGeometryToolkitEval's `bootstrap_ci`, `bootstrap` draws, `StableRNG(seed)`) and the count of cells where the learned model is lower. `settings` is
the `[report]` table. `choose` (a Dict blur model → index into the weights, 1 the unregularised) puts the weight of record where it
says instead of at the least validation rel-RMS (`fold_report`: the weight of the folds' mean validation curve). Returns the report
(written by the caller) and the textures of record in design units (NaN where not seen).
"""
function report_page(dir::AbstractString; settings, choose = nothing, log = println)
    T = VgtkIO.read_record(joinpath(dir, "page.json"); allownan = true); TW, TH = Int(T.TW), Int(T.TH); tex_mm = Float64(T.tex_mm); cell_mm = Float64(T.cell_mm)
    variants = String.(collect(T.variants)); W8 = Float64.(collect(T.regularization_weights)); L = [0.0; W8]
    cells = [(Int(c[1]), Int(c[2])) for c in T.cells]
    bs = round(Int, Float64(settings["block_mm"]) / tex_mm); bs >= 1 || error("report_page: block_mm $(settings["block_mm"]) is below the texel size $tex_mm mm")
    design = page_design(dir, TW, TH)
    cm = cells_mask(cells, TW, TH, tex_mm, cell_mm)
    nb1, nb2 = TW ÷ bs, TH ÷ bs
    out = Dict{String, Any}("TW" => TW, "TH" => TH, "tex_mm" => tex_mm, "cell_mm" => cell_mm, "block_mm" => Float64(settings["block_mm"]), "cells" => length(cells),
                            "photographs" => length(T.photographs), "roles" => Dict("fit" => count(==("fit"), T.roles), "validation" => count(==("val"), T.roles), "test" => count(==("test"), T.roles)),
                            "variants" => variants, "weights" => L, "rule" => "photos: the least validation rel-RMS over the weights, each blur model its own",
                            "cgls_max_iter" => Int(T.cgls_max_iter), "cgls_rtol" => Float64(T.cgls_rtol), "image_circle" => T.image_circle === nothing ? nothing : plain(T.image_circle),
                            "blur_parameters" => Dict("learned" => Dict("r_m" => Float64(T.learned_r_m), "focus" => "the calibration's learned surface"), "pinhole" => Dict("r_m" => Float64(T.pinhole_r_m))))
    textures = Dict{String, Matrix{Float64}}(); per_cell = Dict{String, Vector{Float64}}(); chosen = Dict{String, Any}(); curves = Dict{String, Any}(); cgls = Dict{String, Any}()
    for v in variants
        f = page_files(dir, v); rec = VgtkIO.read_record(f.record; allownan = true); sols = collect(rec.solutions)
        X0 = reshape(Float64.(reinterpret(Float32, read(f.texture))), TW, TH)
        XR = reshape(Float64.(reinterpret(Float32, read(f.reg))), TW, TH, length(W8))
        seen = reshape(reinterpret(UInt8, read(f.seen)) .!= 0, TW, TH); sel = BitMatrix(seen) .& cm
        Xk(k) = k == 1 ? X0 : XR[:, :, k - 1]
        rel(k, key, ss) = (d = Float64(getproperty(rec, ss)); d > 0 ? sqrt(Float64(getproperty(sols[k], key)) / d) : NaN)
        curve = Dict{String, Vector{Float64}}(k => Float64[] for k in ("val_rel_rms", "test_rel_rms", "test_rel_rms_centre", "test_rel_rms_edge", "rms_texel", "rms_block", "iters"))
        scores = map(1:length(L)) do k
            ts = texture_scores(Xk(k), sel, design, bs)
            push!(curve["val_rel_rms"], rel(k, :val_sse, :val_sumsq)); push!(curve["test_rel_rms"], rel(k, :test_sse, :test_sumsq))
            push!(curve["test_rel_rms_centre"], rel(k, :test_sse_centre, :test_sumsq_centre)); push!(curve["test_rel_rms_edge"], rel(k, :test_sse_edge, :test_sumsq_edge))
            push!(curve["rms_texel"], ts.rms_texel); push!(curve["rms_block"], ts.rms_block); push!(curve["iters"], Float64(sols[k].iters))
            ts
        end
        kbest = choose === nothing ? argmin(curve["val_rel_rms"]) : Int(choose[v]); ts = scores[kbest]
        chosen[v] = Dict{String, Any}("weight" => L[kbest], "val_rel_rms" => curve["val_rel_rms"][kbest], "test_rel_rms" => curve["test_rel_rms"][kbest],
                                      "test_rel_rms_centre" => curve["test_rel_rms_centre"][kbest], "test_rel_rms_edge" => curve["test_rel_rms_edge"][kbest],
                                      "a" => ts.a, "b" => ts.b, "rms_texel" => ts.rms_texel, "rms_block" => ts.rms_block, "n_texels" => ts.n_texels, "n_blocks" => ts.n_blocks,
                                      "iters" => Int(sols[kbest].iters), "converged" => Bool(sols[kbest].converged))
        curves[v] = curve; cgls[v] = Dict{String, Any}("unregularised_iters" => Int(sols[1].iters), "unregularised_converged" => Bool(sols[1].converged), "relres" => Float64(rec.relres))
        X = (Xk(kbest) .- ts.b) ./ ts.a; X[.!sel] .= NaN; textures[v] = X
        per_cell[v] = [(r = cell_blocks(i, j, bs, tex_mm, cell_mm, nb1, nb2); e = filter(isfinite, vec(ts.blocks[r[1], r[2]])); isempty(e) ? NaN : sqrt(mean(e))) for (i, j) in cells]
        log(@sprintf("%-8s λ* %g (least validation rel-RMS %.4f): test rel-RMS %.4f (centre %.4f, edge %.4f); texture RMS vs design %.4f (%.3g mm) / %.4f (%.3g mm) over %d texels, %d blocks; CGLS %d its%s; unregularised: test %.4f, texture %.4f / %.4f",
                     v, L[kbest], curve["val_rel_rms"][kbest], curve["test_rel_rms"][kbest], curve["test_rel_rms_centre"][kbest], curve["test_rel_rms_edge"][kbest], ts.rms_texel, tex_mm, ts.rms_block,
                     Float64(settings["block_mm"]), ts.n_texels, ts.n_blocks, Int(sols[kbest].iters), Bool(sols[kbest].converged) ? "" : " (NOT converged)", curve["test_rel_rms"][1],
                     curve["rms_texel"][1], curve["rms_block"][1]))
    end
    out["chosen"] = chosen; out["curves"] = curves; out["cgls"] = cgls
    if all(in(variants), TEXTURE_VARIANTS)
        a, b = per_cell["learned"], per_cell["pinhole"]; ok = isfinite.(a) .& isfinite.(b); d = a[ok] .- b[ok]
        B = Int(settings["bootstrap"]); lo, hi = paired_cells_ci(d; settings)
        out["paired"] = Dict{String, Any}("rms_block_mean_diff" => mean(d), "rms_block_diff_ci95" => [lo, hi],
                                          "learned_lower_cells" => count(<(0), d), "cells" => length(d), "bootstrap" => B,
                                          "test_rel_rms_diff" => chosen["learned"]["test_rel_rms"] - chosen["pinhole"]["test_rel_rms"])
        log(@sprintf("paired over the cells (learned − pinhole) at the chosen weights: texture RMS at %.3g mm mean %+.4f (95 %% [%+.4f, %+.4f]), learned lower in %d of %d; test rel-RMS %+.4f",
                     Float64(settings["block_mm"]), mean(d), lo, hi, count(<(0), d), length(d), out["paired"]["test_rel_rms_diff"]))
    end
    out["per_cell"] = [Dict{String, Any}("cell_i" => cells[k][1], "cell_j" => cells[k][2], [("rms_block_$v" => per_cell[v][k]) for v in variants]...) for k in eachindex(cells)]
    return out, textures
end

"""
    page_design(dir, TW, TH) -> Matrix{Float64}

The design a page run holds (`run_page`'s `design.f32`, the sheet's green channel `[x, y]`): read by the page run's own reader, the
one place it is read.
"""
page_design(dir::AbstractString, TW::Integer, TH::Integer) = reshape(Float64.(reinterpret(Float32, read(joinpath(dir, "design.f32")))), Int(TW), Int(TH))

"A record read back (`VgtkIO.read_record`: NamedTuples, vectors) as plain Dicts and Vectors, to be written again."
plain(x) = x
plain(x::NamedTuple) = Dict{String, Any}(String(k) => plain(v) for (k, v) in pairs(x))
plain(x::AbstractDict) = Dict{String, Any}(string(k) => plain(v) for (k, v) in x)
plain(x::AbstractVector) = Any[plain(v) for v in x]

"""
    write_textures(dir, textures, design; tex_mm) -> Dict

The textures of record (`report_page`'s, design units, NaN where not seen) and the design as `texture_<v>.f32` / `design.f32` (Float32,
x fastest) and as 8-bit PNGs at one pixel per texel (`texture_<v>.png`; NaN black), with `textures.json` naming them and the texel size.
"""
function write_textures(dir::AbstractString, textures, design::AbstractMatrix; tex_mm::Real)
    files = Dict{String, Any}("design" => "design.f32")
    write(joinpath(dir, "design.f32"), Float32.(vec(design)))
    png(name, X) = PNGFiles.save(joinpath(dir, name), reinterpret.(N0f8, round.(UInt8, 255 .* permutedims(clamp.(replace(X, NaN => 0.0), 0, 1)))))   # rows = y
    png("design.png", design)
    for (v, X) in textures
        write(joinpath(dir, "texture_$v.f32"), Float32.(vec(X))); png("texture_$v.png", X); files[v] = "texture_$v.f32"
    end
    meta = Dict{String, Any}("size" => collect(size(design)), "tex_mm" => tex_mm, "files" => files,
                             "note" => "Float32 x fastest (reshape to size[1] × size[2], [x, y]); NaN where not seen or not scored; design units")
    VgtkIO.write_record(joinpath(dir, "textures.json"), merge(Dict{String, Any}("schema" => EXPERIMENT_RECORD_SCHEMAS.textures), meta); allownan = true)
    return meta
end

"A texture of `textures.json` read back as [x, y]."
function read_texture(dir::AbstractString, name::AbstractString)
    meta = VgtkIO.read_record(joinpath(dir, "textures.json"); allownan = true); nx, ny = Int(meta.size[1]), Int(meta.size[2])
    return Matrix{Float64}(reshape(Float64.(reinterpret(Float32, read(joinpath(dir, String(getproperty(meta.files, Symbol(name))))))), nx, ny)), meta
end

# ---- the comparison of page runs and the fold rotation: VisualGeometryToolkitIO records and rasters ---------------------------------------

"""
    crop_texels(crop_mm, tex_mm, TW, TH) -> (xs, ys)

The texels of the page rectangle `crop_mm = [x0, y0, width, height]` (mm from the design's top-left corner, y down): the columns `xs`
and rows `ys` of a `TW × TH` texture `[x, y]`, the rectangle's first texel at `round(x0 / tex_mm) + 1`; the rectangle must lie on the page.
"""
function crop_texels(crop_mm, tex_mm::Real, TW::Integer, TH::Integer)
    x0, y0, w, h = Float64.(crop_mm)
    xs = round(Int, x0 / tex_mm) .+ (1:round(Int, w / tex_mm)); ys = round(Int, y0 / tex_mm) .+ (1:round(Int, h / tex_mm))
    (first(xs) >= 1 && last(xs) <= TW && first(ys) >= 1 && last(ys) <= TH) || throw(ArgumentError("crop_texels: crop_mm $(collect(crop_mm)) is not on the $TW × $TH page of $tex_mm mm texels"))
    return xs, ys
end

"""
    save_texture_crop(path, X, xs, ys; tex_mm, provenance) -> path

The texels `X[xs, ys]` of a page texture (`[x, y]`, design units; NaN unseen) as VisualGeometryToolkitIO's raster record
(`save_raster`: the JSON header at `path`, its Float32 TIFF beside it): row 1 the crop's top edge (the page's y down), in the design
chart's frame (`chart = :page`, x right and y up, its area the crop's rectangle in metres: lower-left corner and widths), NaN meaning
unobserved. Read back bit for bit (`load_raster`) before it is returned.
"""
function save_texture_crop(path::AbstractString, X::AbstractMatrix, xs::AbstractUnitRange, ys::AbstractUnitRange; tex_mm::Real, provenance::NamedTuple = (;))
    values = Matrix{Float32}(permutedims(X[xs, ys]))                    # rows = page y (down), row 1 on top
    page_h = size(X, 2) * tex_mm                                          # mm
    w, h = length(xs) * tex_mm, length(ys) * tex_mm
    area = VgtkCore.Rect2((first(xs) - 1) * tex_mm / 1000, (page_h - (first(ys) - 1) * tex_mm - h) / 1000, w / 1000, h / 1000)   # the design chart (y up), metres
    VgtkIO.save_raster(path, values; frame = (chart = :page, area), nan = any(isnan, values) ? :unobserved : :none, provenance)
    back = VgtkIO.load_raster(path)
    isequal(back.values, values) || error("$path: the raster does not read back bit for bit")
    return path
end

"""
    compare_page_runs(runs, out; settings, crop_mm, reference, log) -> Dict

The page runs `runs` (`name => refine directory`, each a complete `run_page` run) scored by ONE scorer (`report_page` under `settings`,
the `[report]` table) and written as VisualGeometryToolkitIO records into `out`:
- `curves.json`: per run, its weighting and sensor record, and per blur model the curve over the weights β (validation and test
  rel-RMS — all, centre, edge —, the texture RMS against the design at the texel and at `block_mm`, the CGLS iterations);
- `chosen.json`: per run and blur model the weight of record (the least validation rel-RMS) and its scores, and the paired learned −
  pinhole over the cells (the mean, its bootstrap 95 % interval, the cells, the cells where the learned model is lower);
- the rectangle `crop_mm` of every run's textures of record (`crop_<run>_<model>.json`), of the design (`crop_design.json`: the runs'
  designs are refused unless equal) and of every other run's texture minus the `reference` run's (`diff_<run>_minus_<reference>_<model>.json`),
  each a raster record (`save_texture_crop`).
A weight β is comparable only within a run (λ = β · penalty_scale of that run's fit rows). Returns the `chosen.json` body.
"""
function compare_page_runs(runs::AbstractVector{<:Pair}, out::AbstractString; settings, crop_mm, reference::AbstractString, log = println)
    names = String[String(first(r)) for r in runs]
    allunique(names) && reference in names || throw(ArgumentError("compare_page_runs: distinct run names, the reference $(repr(reference)) among them ($(join(names, ", ")))"))
    mkpath(out)
    scored = Dict{String, Any}(); design = nothing; xs = ys = 1:0; tex_mm = NaN
    for (name, dir) in runs
        page_complete(dir) || throw(ArgumentError("compare_page_runs: $name: $dir is not a complete page run"))
        T = VgtkIO.read_record(joinpath(dir, "page.json"); allownan = true)
        rep, tex = report_page(dir; settings, log = s -> log("$name: $s"))
        d = page_design(dir, rep["TW"], rep["TH"])
        if design === nothing
            design = d; tex_mm = rep["tex_mm"]; xs, ys = crop_texels(crop_mm, tex_mm, rep["TW"], rep["TH"])
        else
            (size(d) == size(design) && d == design && rep["tex_mm"] == tex_mm) || throw(ArgumentError("compare_page_runs: $name's design is not the other runs' (another sheet)"))
        end
        scored[name] = (; rep, tex, dir = String(dir), weighting = haskey(T, :weighting) ? String(T.weighting) : "as measured (recorded before the weighting was stated)",
                          sensor = haskey(T, :sensor_record_sha256) ? T.sensor_record_sha256 : nothing)
    end
    variants = collect(TEXTURE_VARIANTS)
    prov(what) = (; what, crop_mm = collect(Float64.(crop_mm)), runs = Dict(n => scored[n].dir for n in names))
    rasters = Dict{String, Any}("design" => basename(save_texture_crop(joinpath(out, "crop_design.json"), design, xs, ys; tex_mm, provenance = prov("the design (the sheet's green channel)"))))
    for n in names, v in variants
        X = scored[n].tex[v]
        rasters["$(n)_$(v)"] = basename(save_texture_crop(joinpath(out, "crop_$(n)_$(v).json"), X, xs, ys; tex_mm,
                                                          provenance = prov("run $n, $v: the texture of record at its chosen weight, design units")))
        n == reference && continue
        rasters["$(n)_minus_$(reference)_$(v)"] = basename(save_texture_crop(joinpath(out, "diff_$(n)_minus_$(reference)_$(v).json"), X .- scored[reference].tex[v], xs, ys; tex_mm,
                                                                             provenance = prov("run $n minus run $reference, $v: the textures of record, design units")))
    end
    # each run minus the reference, per blur model, paired over the cells both score (each cell's block RMS at its run's chosen weight)
    cellmap(n, v) = Dict((Int(r["cell_i"]), Int(r["cell_j"])) => Float64(r["rms_block_$v"]) for r in scored[n].rep["per_cell"])
    vs_reference = Dict{String, Any}[]
    for n in names, v in variants
        n == reference && continue
        A, R = cellmap(n, v), cellmap(reference, v)
        cs = sort!([c for c in keys(A) if haskey(R, c) && isfinite(A[c]) && isfinite(R[c])])
        d = [A[c] - R[c] for c in cs]
        lo, hi = isempty(d) ? (NaN, NaN) : paired_cells_ci(d; settings)
        push!(vs_reference, Dict{String, Any}("run" => n, "reference" => reference, "variant" => v, "block_mm" => Float64(settings["block_mm"]),
                                              "rms_block_mean_diff" => isempty(d) ? NaN : mean(d), "rms_block_diff_ci95" => [lo, hi],
                                              "lower_cells" => count(<(0), d), "cells" => length(d), "bootstrap" => Int(settings["bootstrap"]),
                                              "test_rel_rms_diff" => scored[n].rep["chosen"][v]["test_rel_rms"] - scored[reference].rep["chosen"][v]["test_rel_rms"]))
        log(@sprintf("%s − %s (%s), paired over %d cells: texture RMS at %.3g mm %+.4f (95 %% [%+.4f, %+.4f])", n, reference, v, length(d),
                     Float64(settings["block_mm"]), isempty(d) ? NaN : mean(d), lo, hi))
    end
    run_head(n) = Dict{String, Any}("name" => n, "refine" => scored[n].dir, "weighting" => scored[n].weighting, "sensor_record_sha256" => scored[n].sensor)
    curves = Dict{String, Any}("weights_comparable" => "within a run only (λ = β · that run's penalty_scale)",
                               "runs" => [merge(run_head(n), Dict{String, Any}("weights" => scored[n].rep["weights"], "curves" => scored[n].rep["curves"])) for n in names])
    VgtkIO.write_record(joinpath(out, "curves.json"), merge(Dict{String, Any}("schema" => EXPERIMENT_RECORD_SCHEMAS.texture_recovery_curves), curves); allownan = true)
    chosen = Dict{String, Any}("reference" => reference, "crop_mm" => collect(Float64.(crop_mm)),
                               "tex_mm" => tex_mm, "block_mm" => Float64(settings["block_mm"]), "rasters" => rasters,
                               "weights_comparable" => "within a run only (λ = β · that run's penalty_scale)", "rule" => scored[reference].rep["rule"],
                               "runs" => [merge(run_head(n), Dict{String, Any}("cells" => scored[n].rep["cells"], "chosen" => scored[n].rep["chosen"], "paired" => get(scored[n].rep, "paired", nothing)))
                                          for n in names],
                               "paired_vs_reference" => vs_reference)
    VgtkIO.write_record(joinpath(out, "chosen.json"), merge(Dict{String, Any}("schema" => EXPERIMENT_RECORD_SCHEMAS.texture_recovery_chosen), chosen); allownan = true)
    return chosen
end

"""
    fold_report(dirs, out; settings, log) -> Dict

The rotation of the split (`split_roles`' `fold`): `dirs[f + 1]` the page run of fold `f` (fold 0 the split of record), each scored by
`report_page` under `settings`. Per blur model: every fold's validation and test curve, the mean validation curve over the folds and
its weight β̄ (the least mean validation rel-RMS: the one choice of the rotation), every fold's own chosen β, and per fold the paired
learned − pinhole over the cells both at its own β and at β̄ (`report_page`'s `choose`); the spread of the chosen β and of the
paired margin over the folds (mean, standard deviation, min, max). Written as `folds.json` (a VisualGeometryToolkitIO record); returns it.
"""
function fold_report(dirs::AbstractVector{<:AbstractString}, out::AbstractString; settings, log = println)
    length(dirs) >= 2 || throw(ArgumentError("fold_report: at least two folds"))
    reps = [report_page(d; settings, log = s -> log("fold $(f - 1): $s"))[1] for (f, d) in enumerate(dirs)]
    L = reps[1]["weights"]; all(r -> r["weights"] == L, reps) || throw(ArgumentError("fold_report: the folds' weights differ"))
    variants = collect(TEXTURE_VARIANTS)
    mean_val = Dict(v => [mean(r["curves"][v]["val_rel_rms"][k] for r in reps) for k in eachindex(L)] for v in variants)
    kbar = Dict(v => argmin(mean_val[v]) for v in variants)
    at_bar = [report_page(d; settings, choose = kbar, log = _ -> nothing)[1] for d in dirs]
    spread(x) = Dict{String, Any}("values" => x, "mean" => mean(x), "sd" => length(x) > 1 ? std(x) : 0.0, "min" => minimum(x), "max" => maximum(x))
    paired_own = [r["paired"]["rms_block_mean_diff"] for r in reps]; paired_bar = [r["paired"]["rms_block_mean_diff"] for r in at_bar]
    body = Dict{String, Any}("folds" => length(dirs), "refine" => collect(String.(dirs)), "weights" => L,
                             "rule" => "the least mean validation rel-RMS over the folds (β̄), each blur model its own; per fold also its own least validation rel-RMS",
                             "models" => Dict(v => Dict{String, Any}("mean_val_rel_rms" => mean_val[v], "chosen_weight_mean_curve" => L[kbar[v]],
                                                                      "chosen_weight_per_fold" => spread([r["chosen"][v]["weight"] for r in reps]),
                                                                      "val_rel_rms_per_fold" => [r["curves"][v]["val_rel_rms"] for r in reps],
                                                                      "test_rel_rms_per_fold" => [r["curves"][v]["test_rel_rms"] for r in reps],
                                                                      "test_rel_rms_at_mean_weight" => spread([r["chosen"][v]["test_rel_rms"] for r in at_bar]),
                                                                      "rms_block_at_mean_weight" => spread([r["chosen"][v]["rms_block"] for r in at_bar])) for v in variants),
                             "paired_at_own_weight" => merge(spread(paired_own), Dict{String, Any}("per_fold" => [r["paired"] for r in reps])),
                             "paired_at_mean_weight" => merge(spread(paired_bar), Dict{String, Any}("per_fold" => [r["paired"] for r in at_bar])))
    mkpath(out); VgtkIO.write_record(joinpath(out, "folds.json"), merge(Dict{String, Any}("schema" => EXPERIMENT_RECORD_SCHEMAS.texture_recovery_folds), body); allownan = true)
    log(@sprintf("folds: %d; learned β̄ %g, pinhole β̄ %g; paired (learned − pinhole) at the folds' own β %+.4f ± %.4f [%+.4f, %+.4f], at β̄ %+.4f ± %.4f [%+.4f, %+.4f]",
                 length(dirs), L[kbar["learned"]], L[kbar["pinhole"]], mean(paired_own), body["paired_at_own_weight"]["sd"], minimum(paired_own), maximum(paired_own),
                 mean(paired_bar), body["paired_at_mean_weight"]["sd"], minimum(paired_bar), maximum(paired_bar)))
    return body
end

"""
    write_texture_summary(path; experiment, texture, report, chosen = nothing, folds = nothing, products, keys, rc_version) -> path

The experiment's own summary (`summary.md`), every number read from the records: `report` the evaluate stage's `report.json`, `chosen`
the comparison's `chosen.json` (`compare_page_runs`; its runs the table's rows, else this run's report alone), `folds` the rotation's
`folds.json` (`fold_report`). The metrics' definitions are written from the scorer's own settings — `texture` the config's
`[texture]` table (the split, the regions, the cells' coverage), the report's rule, texel, block and bootstrap. `products` / `run_keys` name the
products read (their directories and run keys), `rc_version` RadiometricCamera's version.
"""
function write_texture_summary(path::AbstractString; experiment::AbstractString, texture::AbstractDict, report, chosen = nothing, folds = nothing,
                               products::AbstractDict, run_keys::AbstractDict, rc_version::AbstractString)
    rep = plain(report); ch = chosen === nothing ? nothing : plain(chosen); fo = folds === nothing ? nothing : plain(folds)
    t = texture; every = Int(t["split_every"]); f4(x) = x === nothing ? "–" : @sprintf("%.4f", x); f3(x) = @sprintf("%+.4f", x)
    cnt(x) = string(round(Int, x))                                     # a count (a record's numbers read back as floats)
    runs = ch === nothing ? [Dict{String, Any}("name" => "this", "refine" => products["evaluate"], "chosen" => rep["chosen"], "paired" => get(rep, "paired", nothing),
                                               "cells" => rep["cells"], "weighting" => "", "sensor_record_sha256" => nothing)] : ch["runs"]
    io = IOBuffer()
    println(io, "# Texture recovery (", experiment, "): summary\n")
    println(io, "Written by the pipeline's `report` stage from the records of the products below; no number is typed by hand.\n")
    println(io, "## Metrics\n")
    println(io, "- **rel-RMS** = √(Σ (y − ŷ)² / Σ y²) over the sampled pixels (a fraction ", t["pixel_fraction"], " of each photograph's masked pixels) of the held-out ",
            "photographs — the validation or the test set — whose chief ray falls in a scored cell; y the observed value, ŷ = V · (blur · texture) the prediction. ",
            "A cell (", rep["cell_mm"], " mm) is scored when ≥ ", t["min_fit_views"], " fit, ≥ ", t["min_val_views"], " validation and ≥ ", t["min_test_views"],
            " test photographs each put ≥ ", t["min_pixels"], " sampled pixels in it. Centre: ρ < ", t["centre_rho"], ", edge: ρ > ", t["edge_rho"],
            " (ρ the image radius over the half-diagonal).")
    println(io, "- **The split**: photograph k (1 … ", cnt(rep["photographs"]), ") is test when k % ", every, " == 0, validation when k % ", every, " == 1, fit otherwise (",
            cnt(rep["roles"]["fit"]), " / ", cnt(rep["roles"]["validation"]), " / ", cnt(rep["roles"]["test"]), ")",
            fo === nothing ? "." : "; the rotation's fold f uses (k − f) % $every, f = 0 … $(cnt(fo["folds"] - 1)) (fold 0 the split of record).")
    println(io, "- **The weight β** of λ ‖∇x‖² (λ = β · Core's penalty_scale of the run's fit rows, so β is comparable only within a run): ", rep["rule"],
            " — blind to the test photographs and to the design. β ∈ {", join(rep["weights"], ", "), "} (0 the unregularised solve).")
    println(io, "- **Design RMS**: the texture of record against the printed design after ONE affine map a · design + b per blur model (least squares over the ",
            "scored texels the fit rows read), in design units, per texel (", rep["tex_mm"], " mm) and over ", rep["block_mm"], " mm blocks.")
    println(io, "- **Paired learned − pinhole**: per scored cell the ", rep["block_mm"], " mm design RMS of each model at its chosen β; the mean difference over the cells, ",
            "its 95 % bootstrap interval over the cells (", cnt(rep["paired"]["bootstrap"]), " draws) and the cells where the learned model is lower.\n")
    println(io, "## Results at the chosen β\n")
    println(io, "| run | blur model | β | validation rel-RMS | test rel-RMS all / centre / edge | design RMS ", rep["tex_mm"], " mm / ", rep["block_mm"], " mm | CGLS iterations |")
    println(io, "|---|---|---|---|---|---|---|")
    for r in runs, v in TEXTURE_VARIANTS
        c = r["chosen"][v]
        println(io, "| ", r["name"], " | ", v, " | ", c["weight"], " | ", f4(c["val_rel_rms"]), " | ", f4(c["test_rel_rms"]), " / ", f4(c["test_rel_rms_centre"]), " / ",
                f4(c["test_rel_rms_edge"]), " | ", f4(c["rms_texel"]), " / ", f4(c["rms_block"]), " | ", cnt(c["iters"]), c["converged"] ? "" : " (not converged)", " |")
    end
    println(io, "\n| run | paired learned − pinhole at ", rep["block_mm"], " mm | 95 % interval | cells | learned lower in |")
    println(io, "|---|---|---|---|---|")
    for r in runs
        q = r["paired"]; q === nothing && continue
        println(io, "| ", r["name"], " | ", f3(q["rms_block_mean_diff"]), " | [", f3(q["rms_block_diff_ci95"][1]), ", ", f3(q["rms_block_diff_ci95"][2]), "] | ", cnt(q["cells"]), " | ",
                cnt(q["learned_lower_cells"]), " |")
    end
    if fo !== nothing
        println(io, "\n## The rotation of the split (", cnt(fo["folds"]), " folds)\n")
        println(io, "Rule: ", fo["rule"], ".\n")
        println(io, "| blur model | β of the mean validation curve | chosen β per fold | test rel-RMS at that β (mean ± sd) | design RMS ", rep["block_mm"], " mm at that β (mean ± sd) |")
        println(io, "|---|---|---|---|---|")
        for v in TEXTURE_VARIANTS
            m = fo["models"][v]
            println(io, "| ", v, " | ", m["chosen_weight_mean_curve"], " | ", join(m["chosen_weight_per_fold"]["values"], ", "), " | ",
                    @sprintf("%.4f ± %.4f", m["test_rel_rms_at_mean_weight"]["mean"], m["test_rel_rms_at_mean_weight"]["sd"]), " | ",
                    @sprintf("%.4f ± %.4f", m["rms_block_at_mean_weight"]["mean"], m["rms_block_at_mean_weight"]["sd"]), " |")
        end
        for (lab, k) in (("each fold's own β", "paired_at_own_weight"), ("the mean curve's β", "paired_at_mean_weight"))
            q = fo[k]
            println(io, "\nPaired learned − pinhole at ", lab, ": ", join(f3.(q["values"]), ", "), " — mean ", f3(q["mean"]), " ± ", @sprintf("%.4f", q["sd"]), " (sd), range [",
                    f3(q["min"]), ", ", f3(q["max"]), "].")
        end
    end
    println(io, "\n## Provenance\n")
    println(io, "- RadiometricCamera ", rc_version)
    for n in sort(collect(keys(products)))
        println(io, "- ", n, ": key `", run_keys[n], "`, `", products[n], "`")
    end
    for r in runs
        println(io, "- run ", r["name"], ": page run `", r["refine"], "`; sensor record sha256 ", r["sensor_record_sha256"] === nothing ? "none stated" : "`$(r["sensor_record_sha256"])`",
                isempty(r["weighting"]) ? "" : "; rows $(r["weighting"])")
    end
    if ch !== nothing
        println(io, "- raster records (vgio/raster, `", products["compare"], "`): ", join(("`$(v)`" for (_, v) in sort(collect(ch["rasters"]); by = first)), ", "))
    end
    mkpath(dirname(path)); write(path, String(take!(io)))
    return path
end

# ---- the figures -----------------------------------------------------------------------------------------------------------------------

"""
    figure_texture_closeup(report_dir, report; render, dir) -> path

The blur at 1:1 texels: the square `render["crop_mm"] = [x0, y0, size]` of the page — the design and the texture of record of each blur
model (design units 0 … 1), side by side, titled `render["titles"]`. `render`: `asset`, `size_in`, `dpi`, `crop_mm`, `titles`.
"""
function figure_texture_closeup(report_dir::AbstractString, rep; render::AbstractDict, dir::AbstractString)
    variants = String.(collect(rep["variants"])); tex = Float64(rep["tex_mm"])
    design, meta = read_texture(report_dir, "design"); TW, TH = size(design)
    cx, cy, S = Float64.(render["crop_mm"])
    ix = round(Int, cx / tex) .+ (1:round(Int, S / tex)); iy = round(Int, cy / tex) .+ (1:round(Int, S / tex))
    (first(ix) >= 1 && last(ix) <= TW && first(iy) >= 1 && last(iy) <= TH) || throw(ArgumentError("figure_texture_closeup: crop_mm $(render["crop_mm"]) is not inside the page"))
    crops = [design[ix, iy], (read_texture(report_dir, v)[1][ix, iy] for v in variants)...]
    dpi = render["dpi"]; pt = dpi / 72; w, h = render["size_in"]
    fig = Figure(; size = (round(Int, w * dpi), round(Int, h * dpi)), fontsize = 7 * pt, backgroundcolor = :white, figure_padding = round(Int, 2 * pt))
    for (c, (A, t)) in enumerate(zip(crops, render["titles"]))
        ax = Axis(fig[1, c]; title = String(t), titlesize = 7 * pt, titlefont = :regular, yreversed = true, aspect = DataAspect())
        hidedecorations!(ax); hidespines!(ax)
        image!(ax, (cx, cx + S), (cy, cy + S), clamp.(replace(A, NaN => 0.0), 0, 1); colormap = :grays, colorrange = (0, 1), interpolate = false)
        limits!(ax, cx, cx + S, cy + S, cy)
    end
    colgap!(fig.layout, round(Int, 4 * pt))
    out = joinpath(dir, render["asset"]); mkpath(dir); save(out, fig; px_per_unit = 1)
    return out
end

"""
    text_bands(block; lines_per_band, bands, white) -> Vector{Tuple{Int, Int}}

A block of printed text `block` (the design, `[x, y]`, 0 … 1) cut into `bands` bands of `lines_per_band` text lines each, top to
bottom, as `(first, last)` rows (y): a row is blank when every texel of it is above `white`, the text lines are the runs of rows
between blank ones; a band ends in the middle of the gap after its last line, the next begins on the row below, and the first begins
half the first gap above the first line — so the bands cut between lines and the text reads on. A block with too few lines (the
last band's gap needs one more line) is an `ArgumentError`.
"""
function text_bands(block::AbstractMatrix; lines_per_band::Integer, bands::Integer, white::Real)
    blank = [all(>(white), @view block[:, j]) for j in axes(block, 2)]
    lines = Tuple{Int, Int}[]; j = 1; n = length(blank)
    while j <= n
        blank[j] && (j += 1; continue)
        a = j
        while j <= n && !blank[j]; j += 1; end
        push!(lines, (a, j - 1))
    end
    need = lines_per_band * bands
    length(lines) > need || throw(ArgumentError("text_bands: the block holds $(length(lines)) text lines; $bands bands of $lines_per_band need $(need + 1) (a band ends in the gap after its last line)"))
    cut(i) = (lines[i][2] + lines[i + 1][1]) ÷ 2                      # the middle row of the gap after line i
    top = max(1, lines[1][1] - (lines[2][1] - lines[1][2] - 1) ÷ 2)
    ends = [cut(lines_per_band * k) for k in 1:bands]
    return [(k == 1 ? top : ends[k - 1] + 1, ends[k]) for k in 1:bands]
end

"""
    page_photograph(img, lens, R, t; TW, TH, tex_mm, coarse = 16) -> Matrix{Float64}

A photograph of the sheet on the page's texels: `img` (H × W, linear grey, the camera's sensor) under its pose (page → camera, `R`,
`t` in metres: `read_poses`) through the central camera `lens`. Every pixel centre's chief ray (Core's `backproject`, where the lens
images: `in_domain`) meets the page's plane at a page point (`page_frame`: x right, y down from the design's top-left corner), so the
pixel lattice is a mesh over the page; a texel's centre takes the value linearly interpolated in the mesh triangle it lies in, NaN
where none does. The pixels are those of the page's bounding box in the photograph, found on every `coarse`-th pixel. `TW × TH`
texels of `tex_mm` mm; the result is `[x, y]`, as the textures.
"""
function page_photograph(img::AbstractMatrix, lens, R, t; TW::Integer, TH::Integer, tex_mm::Real, coarse::Integer = 16)
    H, W = size(img)
    n = SVector{3, Float64}(R[1, 3], R[2, 3], R[3, 3]); tt = SVector{3, Float64}(t); nt = dot(n, tt)   # the page's plane in the camera frame: n · X = n · t
    Rm = SMatrix{3, 3, Float64}(R)
    function on_page(u, v)                                           # pixel (u, v)'s chief ray on the page, in texels from its top-left corner
        𝐩 = RC.𝐩_of_pixel(u, v)
        RC.in_domain(lens, 𝐩) || return SVector(NaN, NaN)
        d = SVector{3, Float64}(VgtkCore.backproject(lens, 𝐩)); all(isfinite, d) || return SVector(NaN, NaN)
        s = nt / dot(n, d); (isfinite(s) && s > 0) || return SVector(NaN, NaN)
        X = Rm' * (s * d - tt)
        return SVector(1e3 * X[1] / tex_mm, 1e3 * X[2] / tex_mm)
    end
    inside(q) = isfinite(q[1]) && -1 <= q[1] <= TW + 1 && -1 <= q[2] <= TH + 1
    us = Int[]; vs = Int[]
    for v in 1:coarse:H, u in 1:coarse:W
        inside(on_page(u, v)) && (push!(us, u); push!(vs, v))
    end
    isempty(us) && throw(ArgumentError("page_photograph: no pixel's chief ray meets the page (the photograph does not show it under this pose)"))
    u0, u1 = max(1, minimum(us) - coarse), min(W, maximum(us) + coarse); v0, v1 = max(1, minimum(vs) - coarse), min(H, maximum(vs) + coarse)
    Q = [on_page(u, v) for u in u0:u1, v in v0:v1]
    out = fill(NaN, TW, TH)
    function fill_triangle!(p1, p2, p3, f1, f2, f3)                  # the texel centres inside the triangle: its three values, barycentric
        (isfinite(p1[1]) && isfinite(p2[1]) && isfinite(p3[1])) || return
        det = (p2[2] - p3[2]) * (p1[1] - p3[1]) + (p3[1] - p2[1]) * (p1[2] - p3[2])
        abs(det) > 1e-12 || return
        for i in max(1, ceil(Int, min(p1[1], p2[1], p3[1]) + 0.5)):min(TW, floor(Int, max(p1[1], p2[1], p3[1]) + 0.5)),
            j in max(1, ceil(Int, min(p1[2], p2[2], p3[2]) + 0.5)):min(TH, floor(Int, max(p1[2], p2[2], p3[2]) + 0.5))
            x, y = i - 0.5, j - 0.5
            l1 = ((p2[2] - p3[2]) * (x - p3[1]) + (p3[1] - p2[1]) * (y - p3[2])) / det
            l2 = ((p3[2] - p1[2]) * (x - p3[1]) + (p1[1] - p3[1]) * (y - p3[2])) / det
            l3 = 1 - l1 - l2
            (l1 >= -1e-9 && l2 >= -1e-9 && l3 >= -1e-9) && (out[i, j] = l1 * f1 + l2 * f2 + l3 * f3)
        end
    end
    for b in 1:(size(Q, 2) - 1), a in 1:(size(Q, 1) - 1)
        u, v = u0 + a - 1, v0 + b - 1
        f00, f10, f01, f11 = Float64(img[v, u]), Float64(img[v, u + 1]), Float64(img[v + 1, u]), Float64(img[v + 1, u + 1])
        fill_triangle!(Q[a, b], Q[a + 1, b], Q[a, b + 1], f00, f10, f01)
        fill_triangle!(Q[a + 1, b], Q[a + 1, b + 1], Q[a, b + 1], f10, f11, f01)
    end
    return out
end

"""
    figure_texture_stacked(report_dir, report; render, dir, photograph) -> path

The texture test on the page's text (fig:texture-learned; owner 2026-11-15): ONE block of the printed text, `render["block_mm"] =
[x0, y0, width, height]` of the page, stacked from four sources, `render["lines_per_band"]` text lines each, cut between the lines
(`text_bands`, the design's rows above `render["white"]` blank) so that the text reads on from top to bottom — the design (the
ground truth), a photograph of the sheet on the page's texels (`photograph`, `[x, y]` as the textures: `page_photograph` of a fit
photograph; its grey scaled so that the paper's white, the 90th percentile over the block, is the design's), then the texture of
record of each blur model of `render["variants"]` — at 1:1 texels, grey, design units 0 … 1. The bands are named in a margin on the
left, `render["label_fraction"]` of the width, by `render["labels"]` (one per band), with a tick at each cut. `render`: `asset`,
`size_in`, `dpi`, `block_mm`, `lines_per_band`, `white`, `label_fraction`, `variants`, `labels` (and `photograph`, the caller's).
"""
function figure_texture_stacked(report_dir::AbstractString, rep; render::AbstractDict, dir::AbstractString, photograph::AbstractMatrix)
    variants = String.(render["variants"]); tex = Float64(rep["tex_mm"]); labels = String.(render["labels"])
    design, _ = read_texture(report_dir, "design"); TW, TH = size(design)
    size(photograph) == (TW, TH) || throw(ArgumentError("figure_texture_stacked: the photograph is $(size(photograph, 1)) × $(size(photograph, 2)) texels, the page $TW × $TH"))
    x0, y0, bw, bh = Float64.(render["block_mm"])
    ix = round(Int, x0 / tex) .+ (1:round(Int, bw / tex)); iy = round(Int, y0 / tex) .+ (1:round(Int, bh / tex))
    (first(ix) >= 1 && last(ix) <= TW && first(iy) >= 1 && last(iy) <= TH) || throw(ArgumentError("figure_texture_stacked: block_mm $(render["block_mm"]) is not inside the page"))
    D = design[ix, iy]; P = photograph[ix, iy]
    seen = filter(isfinite, P)
    isempty(seen) && throw(ArgumentError("figure_texture_stacked: the photograph does not show the block"))
    P = P .* (quantile(vec(D), 0.9) / quantile(seen, 0.9))           # the paper's white as the design's
    sources = [D, P, (read_texture(report_dir, v)[1][ix, iy] for v in variants)...]
    bands = text_bands(D; lines_per_band = Int(render["lines_per_band"]), bands = length(sources), white = Float64(render["white"]))
    stack = reduce(hcat, [clamp.(replace(S[:, a:b], NaN => 0.0), 0, 1) for (S, (a, b)) in zip(sources, bands)])   # [x, y]: the bands one under the other
    SW, SH = size(stack); margin = SW * Float64(render["label_fraction"]) / (1 - Float64(render["label_fraction"]))   # the labels' margin, in texels
    dpi = render["dpi"]; pt = dpi / 72; w, h = render["size_in"]
    fig = Figure(; size = (round(Int, w * dpi), round(Int, h * dpi)), fontsize = 6 * pt, backgroundcolor = :white, figure_padding = round(Int, 1 * pt))
    ax = Axis(fig[1, 1]; yreversed = true, aspect = DataAspect())
    hidedecorations!(ax); hidespines!(ax)
    image!(ax, (0, SW), (0, SH), stack; colormap = :grays, colorrange = (0, 1), interpolate = false)
    edge = 0.0
    for (k, (a, b)) in enumerate(bands)
        text!(ax, -margin / 2, edge + (b - a + 1) / 2; text = labels[k], align = (:center, :center), fontsize = 6 * pt, color = :black)
        edge += b - a + 1
        k < length(bands) && lines!(ax, [-0.95 * margin, -0.05 * margin], [edge, edge]; color = RGBf(0.4, 0.4, 0.4), linewidth = 0.5 * pt)
    end
    limits!(ax, -margin, SW, SH, 0)
    out = joinpath(dir, render["asset"]); mkpath(dir); save(out, fig; px_per_unit = 1)
    return out
end

"""
    figure_texture_page(report_dir, report; render, dir) -> path

The page: top, the design and the texture of record under each blur model (grey, design units 0 … 1, `render["downsample"]` texels per
cell); bottom, each blur model's error against the design at the report's block size (diverging, ± `render["error_range"]`) and the
scores. `render`: `asset`, `size_in`, `dpi`, `downsample`, `error_range`, `title`.
"""
function figure_texture_page(report_dir::AbstractString, rep; render::AbstractDict, dir::AbstractString)
    dpi = render["dpi"]; pt = dpi / 72; w, h = render["size_in"]; ds = Int(render["downsample"]); er = Float64(render["error_range"])
    design, meta = read_texture(report_dir, "design"); tex = Float64(meta.tex_mm); bs = round(Int, Float64(rep["block_mm"]) / tex)
    ext = (0.0, size(design, 1) * tex, 0.0, size(design, 2) * tex)
    variants = String.(rep["variants"]); labels = Dict("pinhole" => "pinhole (no defocus)", "learned" => "learned aperture + focus surface")
    fig = Makie.Figure(; size = (round(Int, w * dpi), round(Int, h * dpi)), fontsize = 8 * pt, backgroundcolor = :white, figure_padding = round(Int, 4 * pt))
    xs(A) = range(ext[1], ext[2]; length = size(A, 1)); ys(A) = range(ext[3], ext[4]; length = size(A, 2))
    panel(r, c, title) = Makie.Axis(fig[r, c]; title, titlefont = :regular, titlesize = 8.5 * pt, xlabel = "x [mm]", ylabel = c == 1 ? "y [mm]" : "",
                                    aspect = Makie.DataAspect(), yreversed = true, xticklabelsize = 7 * pt, yticklabelsize = 7 * pt, xgridvisible = false, ygridvisible = false)
    D = nanbox(design, ds)
    ax = panel(1, 1, "design"); Makie.heatmap!(ax, xs(D), ys(D), D; colormap = :grays, colorrange = (0, 1))
    for (c, v) in enumerate(variants)
        X, _ = read_texture(report_dir, v); A = nanbox(X, ds)
        ax = panel(1, c + 1, "recovered: $(labels[v])"); Makie.heatmap!(ax, xs(A), ys(A), A; colormap = :grays, colorrange = (0, 1))
        E = nanbox(X .- design, bs)
        ax = panel(2, c + 1, @sprintf("%s − design at %.3g mm: RMS %.3f", v, rep["block_mm"], rep["chosen"][v]["rms_block"]))
        Makie.heatmap!(ax, xs(E), ys(E), E; colormap = :RdBu, colorrange = (-er, er))
    end
    Makie.Colorbar(fig[2, length(variants) + 2]; colormap = :RdBu, limits = (-er, er), label = "recovered − design", labelsize = 7.5 * pt, ticklabelsize = 7 * pt)
    lines = String[]
    for v in variants
        c = rep["chosen"][v]
        push!(lines, @sprintf("%s, λ = %g\n  test rel-RMS %.3f (centre %.3f, edge %.3f)\n  texture RMS %.3f at %.3g mm, %.3f at %.3g mm", labels[v], c["weight"], c["test_rel_rms"],
                              c["test_rel_rms_centre"], c["test_rel_rms_edge"], c["rms_texel"], rep["tex_mm"], c["rms_block"], rep["block_mm"]))
    end
    if haskey(rep, "paired")
        q = rep["paired"]
        push!(lines, @sprintf("learned − pinhole, paired over %d cells (95 %%)\n  texture %+.3f [%+.3f, %+.3f], lower in %d\n  test rel-RMS %+.3f", q["cells"], q["rms_block_mean_diff"],
                              q["rms_block_diff_ci95"]..., q["learned_lower_cells"], q["test_rel_rms_diff"]))
    end
    Makie.Label(fig[2, 1], join(lines, "\n\n"); fontsize = 6.5 * pt, justification = :left, halign = :left, valign = :center, tellwidth = false, tellheight = false)
    Makie.Label(fig[0, :], render["title"]; fontsize = 9 * pt, halign = :left, tellwidth = false)
    out = joinpath(dir, render["asset"]); mkpath(dirname(out)); Makie.save(out, fig; px_per_unit = 1)
    return out
end
