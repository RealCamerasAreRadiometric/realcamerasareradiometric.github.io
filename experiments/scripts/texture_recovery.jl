#!/usr/bin/env julia
# reproduce: writes ../paper/fig/texture-learned.tex
# reproduce: writes plots/texture-learned.png (plotsdir(); an earlier rendering of the same figure, not in the paper)
# reproduce: reads configs/experiments/texture_recovery_14mm_small.toml
# provenance: every product writes _config.toml and _provenance.toml (VisualGeometryToolkitEval produce) beside its results.
# =============================================================================
# scripts/texture_recovery.jl — the texture test of the blur model (the defocus experiment, fig:texture-learned): a finished refinement's
# camera held fixed (the case [upstream] names, of scripts/radiometric_refinement.jl: its lens, focus surface, aperture radius and vignetting)
# on photographs of a printed sheet it never saw; each pose from the sheet's blobs alone (BlobBoards' find_boards on its pose arm); the page's texture recovered from mid-grey by
# CGLS on the GPU under two blur models — the learned aperture + focus surface, and the pinhole (the chief ray alone) — with a Tikhonov
# weight each model chooses blind on the validation photographs, and scored on the test photographs and against the design
# (src/texture_recovery.jl). Configs of record: configs/experiments/texture_recovery_<set>.toml.
#
# STAGES (each run keyed by its config record and the keys of the runs it reads; <results>/<experiment>/<stage>/<key>/)
#   prepare   the texture photographs decoded as the refinement's (Features' load_raw: [prepare.<set>])
#   detect    the photographs BlobBoards' find_boards reads and the board's design JSON (tier rc_baselines), present
#   init      per photograph its pose from its blobs alone: BlobBoards' find_boards on its pose arm ([poses.find_boards]) through the
#             upstream refinement's camera held fixed, the board's frame on the page by the sheet's blob table (poses.json; the two
#             decodes' gains checked equal)
#   refine    the page: the package's pixel mask inside the refinement's image circle, a fraction of every photograph's pixels, every pixel's
#             blur row under both models, CGLS to convergence for the unregularised texture and every [texture] regularization_weights λ
#             (page.json, page_<model>.f32, _reg.f32, _seen.u8, page_<model>.json); on one GPU
#             The rows are weighted by the observation model's type (RC's masked_observations, the calibration's one weighting): a
#             RAW photograph under a sensor record with a variance law contributes whitened rows (1/σ), a plain one its rows as measured
#   folds     [folds]: the page runs of the split's other rotations (fold f: (k − f) % split_every; fold 0 the refine run), one GPU
#   evaluate  the report: each model's weight by the least validation error, its test and texture scores, the paired test over the cells,
#             the textures of record (report.json, report.md, curves.tsv, per_cell.tsv, texture_<model>.f32 / .png)
#   fold_report [folds]: every fold scored, the β of the folds' mean validation curve, the spread of each fold's β and paired margin (folds.json)
#   compare   [compare]: page runs by their refine keys (and `this`, the config's own) scored by one scorer — curves.json, chosen.json
#             and the crop [compare] crop_mm of every texture of record, of the design and of each run minus the reference, as
#             VisualGeometryToolkitIO raster records (crop_*.json + .tiff, diff_*.json + .tiff)
#   report    summary.md: the metrics' definitions from the scorer's settings, the results per run and blur model at the chosen β, the
#             paired margin with its interval and cells, the folds' spread, the provenance (run keys, RC's version, the sensor record,
#             the raster records) — every number from the records
#   figures   [figures.<name>] → plotsdir() (fig:texture-learned: one block of the page's text stacked from the design, a training
#             photograph on the page's texels, the pinhole's texture and ours, a few lines each; the page overview with the errors; a
#             close-up of the design and both models), numbers.tex
#   bands     [bands]: the paper's figure (fig:texture-learned as the paper prints it) → plotsdir(): the first three lines of the
#             page's text, once from the design, from a training photograph, from the pinhole's texture and from ours, one band
#             each — the PGFPlots / TikZ files texture-learned.tex, texture-learned/<band>.{tex,png} and texture-learned-caption.tex
#             (src/paper_figures/texture_learned.jl, its parts tables and figure) from the evaluate stage's textures, the init
#             stage's poses and the refine stage's roles. The photograph's band is an input (an author-side step's output:
#             configs/inputs/texture_learned_photograph_20261121.toml)
#
#   julia --project=experiments experiments/scripts/texture_recovery.jl [--config=<toml>] [--plan] [--stage=NAME[,NAME]] [--force]
#
# Data: VGTK_DATA_ROOT binds data_root() (REPRODUCE.md). The thread count keys the runs; the refine stage needs a GPU (CUDA_VISIBLE_DEVICES=0;
# GPU_STAGES). The upstream refinement must exist (scripts/radiometric_refinement.jl --config=<[upstream] refinement> --stage=prepare,refine
# --case=<[upstream] case>).
# =============================================================================
using DrWatson
@quickactivate "experiments"
include(srcdir("experiments.jl"))
include(srcdir("drivers.jl"))
include(srcdir("radiometric_refinement.jl"))
include(srcdir("texture_recovery.jl"))
include(srcdir("texture_gpu.jl"))
include(srcdir("paper_figures", "PaperFigures.jl"))                           # the paper figure's PGFPlots text (its generator: texture_learned.jl there)
using Printf: @sprintf, @printf
using StaticArrays: SVector

const DRIVER = "scripts/texture_recovery.jl"
const CFG_PATH = abspath(cli_option(ARGS, "config", projectdir("configs", "experiments", "texture_recovery_14mm_small.toml")))   # default: a config of record

# ---- [bands]: the paper's figure, its table ------------------------------------------------------------------------------------
"""The keys of `[bands]`: the figure's name under plotsdir() (`asset_dir`); the bands from the top (`order`, each with its table
`[bands.<band>]`); whether the text reads on from band to band (`read_on`, a switch); the page's text block (`block_mm`), the text
lines a band holds (`lines_per_band`) and the level a blank row of the design is above (`white`); the pixels a texel is drawn
with (`pixels_per_texel`); and the photograph band's two inputs (`BANDS_INPUT_KEYS`)."""
const BANDS_KEYS = ("asset_dir", "order", "read_on", "block_mm", "lines_per_band", "white", "pixels_per_texel", "photograph_cut", "photograph_record")
"The keys of a `[bands.<band>]` table: what the band shows (`source`), its name in the margin (`label`) and the caption's words for it (`caption`)."
const BAND_KEYS = ("source", "label", "caption")
"The keys of `[bands]` that name the photograph band's inputs, each of one file: its cut (`photograph_cut`) and the record of that cut (`photograph_record`)."
const BANDS_INPUT_KEYS = ("photograph_cut", "photograph_record")

"""
    check_bands_table(c, where) -> c

`[bands]` of the texture config `c` checked (a config without the table has no stage bands): exactly `BANDS_KEYS` and a table
per band of `order`; `asset_dir` a non-empty string; `order` a non-empty array of distinct band names; every `[bands.<band>]`
exactly `BAND_KEYS`, its `source` the design, the photograph or a blur model of the run (`TEXTURE_VARIANTS`), each source once, its
`label` and `caption` non-empty strings; `read_on` a switch (`"on"` | `"off"`); `block_mm` four numbers `[x0, y0, width, height]`,
`lines_per_band` and `pixels_per_texel` integers ≥ 1, `white` a design level in (0, 1); `photograph_cut` and `photograph_record`
inputs of stage figures, each one file pinned by its `sha256` (an author-side step's outputs are inputs of record: the figure is
keyed by their content). An `ArgumentError` names `where` and the key.
"""
function check_bands_table(c::AbstractDict, where::AbstractString)
    haskey(c, "bands") || return c
    t = c["bands"]; w = "$where [bands]"
    t isa AbstractDict || throw(ArgumentError("$w must be a table"))
    order = get(t, "order", nothing)
    order isa AbstractVector && !isempty(order) && all(b -> b isa AbstractString && !isempty(b) && !(b in BANDS_KEYS), order) && allunique(order) ||
        throw(ArgumentError("$w order = $(repr(order)): a non-empty array of distinct band names (none a key of [bands]), from the top"))
    check_required_keys(t, (BANDS_KEYS..., order...), w)
    t["asset_dir"] isa AbstractString && !isempty(t["asset_dir"]) ||
        throw(ArgumentError("$w asset_dir = $(repr(t["asset_dir"])): a non-empty string (the figure's files are <asset_dir>.tex, <asset_dir>/<band>.{tex,png}, <asset_dir>-caption.tex)"))
    sources = ("design", "photograph", TEXTURE_VARIANTS...)
    for b in order
        check_required_keys(t[b], BAND_KEYS, "$w.$b")
        t[b]["source"] in sources ||
            throw(ArgumentError("$w.$b source = $(repr(t[b]["source"])): design (the printed design), photograph (the training photograph's cut) or a blur model ($(join(TEXTURE_VARIANTS, ", ")))"))
        all(k -> t[b][k] isa AbstractString && !isempty(t[b][k]), ("label", "caption")) || throw(ArgumentError("$w.$b label, caption: non-empty strings"))
    end
    allunique(t[b]["source"] for b in order) || throw(ArgumentError("$w: every band another source, got $(join((t[b]["source"] for b in order), ", "))"))
    RadiometricCamera.read_switch(t["read_on"], "$w read_on")
    t["block_mm"] isa AbstractVector && length(t["block_mm"]) == 4 && all(_is_number, t["block_mm"]) && t["block_mm"][3] > 0 && t["block_mm"][4] > 0 ||
        throw(ArgumentError("$w block_mm = $(repr(t["block_mm"])): [x0, y0, width, height] in mm"))
    for k in ("lines_per_band", "pixels_per_texel")
        _is_count(t[k]) && t[k] >= 1 || throw(ArgumentError("$w $k = $(repr(t[k])): an integer ≥ 1"))
    end
    _is_number(t["white"]) && 0 < t["white"] < 1 || throw(ArgumentError("$w white = $(repr(t["white"])): a design level in (0, 1)"))
    for k in BANDS_INPUT_KEYS
        i = t[k] isa AbstractString ? get(c["inputs"], t[k], nothing) : nothing
        i !== nothing && i["stage"] == "figures" && haskey(i, "sha256") ||
            throw(ArgumentError("$w $k = $(repr(t[k])) is not an input of stage figures pinned by its sha256"))
    end
    return c
end

const CFG = check_bands_table(load_texture_recovery_config(CFG_PATH), CFG_PATH)
const SET = first(keys(CFG["prepare"]))
const REFINEMENT_PATH = projectdir(CFG["upstream"]["refinement"])
const REFINEMENT = load_radiometric_refinement_config(REFINEMENT_PATH)      # the refinement whose camera is held fixed
const CASE = CFG["upstream"]["case"]

function logger(dir)
    mkpath(dir)
    return msg -> (println(msg); flush(stdout); open(io -> println(io, msg), joinpath(dir, "run.log"), "a"))
end
"""The common decode gain of `[prepare.<set>]`'s photographs, by what the prepared folder holds: decoded photographs' conversion record
(`conversion.json`; an error when they differ), else RAW files only, read by their type (Features' linearisation, no gain): 1."""
function decode_gain(cfg, set)
    linear_dir = prepared_images(cfg, set)
    if !isfile(joinpath(linear_dir, "conversion.json"))
        files = filter(f -> !startswith(f, "."), readdir(linear_dir))
        all(RC.is_raw_file, files) && !isempty(files) || error("$linear_dir: neither decoded photographs' conversion.json nor RAW files only")
        return 1.0
    end
    g = unique(Float64(p.gain) for p in values(read_conversion(linear_dir).photos))
    length(g) == 1 || error("$linear_dir: the photographs were decoded at different gains $g")
    return g[1]
end
# ---- sheet: the sheet's target record, a run of its own ------------------------------------------------------------------------
# The input `[poses] board` holds the sheet as the author made it: sheet.json (page_mm), sheet_blobs.tsv (per BlobBoards blob its
# centre on the page, mm from the top-left corner, y down) and texture.png. This stage writes it as VisualGeometryToolkitIO's target
# record (`save_sheet_target`: target.json, the blobs its nominal points, the page its canvas, texture.png its canvas texture) into
# its keyed run, beside a copy of texture.png; every later stage reads the sheet from this run (`sheet_dir`), never from the input.
sheet_product() = driver_product("sheet", Dict{String, Any}("experiment" => CFG["experiment"], "inputs" => inputs_record(CFG, [CFG["poses"]["board"]])); output = "target.json")
sheet_dir() = product_dir(sheet_product())
sheet(p) = produce(p; product_records(DRIVER, CFG_PATH)...) do dir
    key = p.key
    src = config_input(CFG, CFG["poses"]["board"])
    cp(joinpath(src, "texture.png"), joinpath(dir, "texture.png"); force = true)
    pm = Float64.(collect(VgtkIO.read_record(joinpath(src, "sheet.json"); allownan = true).page_mm))
    rows = read_tsv(joinpath(src, "sheet_blobs.tsv"))              # VisualGeometryToolkitEval's table reader: blob, x_mm, y_mm
    blobs = Dict(parse(Int, r["blob"]) => SVector(parse(Float64, r["x_mm"]), parse(Float64, r["y_mm"])) for r in rows)
    path = save_sheet_target(dir; page_mm = (pm[1], pm[2]), blobs_mm = blobs)
    back = read_sheet(dir)
    maximum(j -> maximum(abs.(back.blobs[j] .- blobs[j])), keys(blobs)) < 1e-9 || error("$path: the blobs do not read back")
    println(">> sheet → ", path, ": ", length(blobs), " blobs, page ", pm[1], " × ", pm[2], " mm, texel ", back.texel_mm, " mm")
    Produced(; inputs = [src])
end

# ---- prepare, detect ---------------------------------------------------------------------------------------------------------
detect() = (require_stage_inputs(CFG, "detect"); nothing)

# ---- init: the poses of the sheet's photographs ------------------------------------------------------------------------------
upstream_product() = refine_product(REFINEMENT, CASE)
state_path() = joinpath(product_dir(upstream_product()), CFG["upstream"]["state"])
function require_state()
    isfile(state_path()) && return state_path()
    error("the upstream refinement's state $(state_path()) is not there: julia --project=experiments experiments/scripts/radiometric_refinement.jl " *
          "--config=$(relpath(REFINEMENT_PATH, projectdir())) --stage=prepare,refine --case=$CASE")
end
"The image regions of the poses' statistics and the test scores: [texture] centre_rho, edge_rho."
regions() = Dict{String, Any}("centre_rho" => CFG["texture"]["centre_rho"], "edge_rho" => CFG["texture"]["edge_rho"])
poses_product() = driver_product("init", Dict{String, Any}("experiment" => CFG["experiment"], "poses" => CFG["poses"], "regions" => regions(),
                                                  "inputs" => inputs_record(CFG, [CFG["poses"]["images"], CFG["poses"]["board_design"], CFG["poses"]["board"]]),
                                                  "upstream" => Dict("refine" => upstream_product().key, "prepare" => prepare_product(CFG, SET).key, "sheet" => sheet_product().key)); output = "poses.json")
poses(p) = produce(p; product_records(DRIVER, CFG_PATH)...) do dir
    key = p.key
    inputs = require_inputs_through(CFG, "init")
    state = require_state()
    say = logger(dir)
    # the sheet's photographs decoded as the refinement's: the same gain, so the vignetting and the radiance scale carry over
    mine = prepared_images(CFG, SET); theirs = prepared_images(REFINEMENT, CFG["upstream"]["images"])
    g1, g2 = decode_gain(CFG, SET), decode_gain(REFINEMENT, CFG["upstream"]["images"])
    g1 == g2 || error("the sheet's photographs were decoded at gain $g1, the refinement's at $g2 ($theirs): the two decodes must agree ([prepare.$SET] gain)")
    st = RC.load_state(state); p = CFG["poses"]
    W, H = Tuple(VgtkCore.extent(st.camera.geometric))   # the photographs' size: the refinement's camera (its pixels are theirs)
    sheet = read_sheet(sheet_dir()); photos = config_input(CFG, p["images"]); design = config_input(CFG, p["board_design"])
    finder = board_finder(design, st.camera.geometric, p["find_boards"]; gallery = joinpath(dir, "gallery.jld2"))
    page = board_page_map(finder.board.bundle, sheet)
    say(@sprintf("poses: state %s (aperture r %.4f mm), page %.1f × %.1f mm with %d blobs; board %s (%s) on the page: %d blobs, rigid fit RMS %.4f mm, max %.4f mm; BlobBoards' pose arm on %s (%s); decode gain %g", state, 1e3 * st.camera.aperture_radius, 1e3 * sheet.page_w, 1e3 * sheet.page_h,
                 length(sheet.blobs), finder.board.uid, design, length(page.residuals_mm), sqrt(mean(abs2, page.residuals_mm)), maximum(page.residuals_mm),
                 photos, finder.device, g1))
    names = filter(RC.is_png_file, RC.photo_files(photos))
    views, allr, XY, nan = sheet_poses(finder, [(f, joinpath(photos, f)) for f in names]; page = page.map, log = say)
    isempty(views) && error("poses: no photograph got a pose")
    write_poses(joinpath(dir, "poses.json"), views, allr, XY, nan; W, H, centre_rho = Float64(CFG["texture"]["centre_rho"]), edge_rho = Float64(CFG["texture"]["edge_rho"]),
                label = "$(CFG["experiment"]["name"]): case $CASE held fixed")
    S = residual_summary(allr)
    say(@sprintf("RESULT: %d photographs with a pose, %d blobs (%d not projected): reprojection median %.4f / mean %.4f / RMS %.4f / p95 %.4f / max %.3f px",
                 length(views), S.n, nan, S.median_px, S.mean_px, S.rms_px, S.p95_px, S.max_px))
    Produced(; inputs = [inputs; state; mine; theirs])
end

# ---- refine: the page under both blur models ------------------------------------------------------------------------------------
"""The page run of fold `f` of the split (`split_roles`' rotation; 0 the split of record, keyed as before the rotation existed), one
product each in the refine stage's directory."""
function texture_product(fold::Integer = 0)
    rec = Dict{String, Any}("experiment" => CFG["experiment"], "texture" => CFG["texture"], "inputs" => inputs_record(CFG, [CFG["poses"]["board"]]),
                            "upstream" => Dict("init" => poses_product().key, "prepare" => prepare_product(CFG, SET).key, "refine" => upstream_product().key))
    fold == 0 || (rec["fold"] = Int(fold))
    return driver_product("refine", rec; output = "page.json")
end
"The rotations of `[folds]` (none without the table); `--fold=1,3` selects some."
fold_rotations() = haskey(CFG, "folds") ? Int(CFG["folds"]["rotations"]) : 1
fold_items() = parse.(Int, cli_subset(ARGS, "fold", string.(1:(fold_rotations() - 1))))
refine(fold::Integer) = produce(texture_product(fold); product_records(DRIVER, CFG_PATH)...) do dir
    inputs = require_inputs_through(CFG, "refine")
    state = require_state()
    say = logger(dir)
    device = RC.rc_backend("gpu"; inline = false, fallback = false)   # refused without a functional GPU: no CPU fallback
    say("refine: the page solve on the GPU ($(TextureGPU.gpu_name()))")
    s = CFG["texture"]
    st = RC.load_state(state); sheet = read_sheet(sheet_dir()); images = prepared_images(CFG, SET); W, H = Tuple(VgtkCore.extent(st.camera.geometric))
    poses = read_poses(joinpath(product_dir(poses_product()), "poses.json"))
    # the camera held fixed is the calibration's COMPLETE camera (its observation model and render settings): the sheet's photographs
    # are read through RadiometricCamera's entry point under the camera's own sensor record, located by the case's characterisation input
    ctx = sheet_scene(RC.calibrated_camera(st), st.target, sheet, joinpath(sheet_dir(), "texture.png"), images, poses, W, H;
                      characterisation = case_characterisation(REFINEMENT, CASE))
    circle = image_circle_of(st.camera)
    say(circle === nothing ? "image circle: none (the lens lights the whole sensor)" :
        @sprintf("image circle: the refinement's camera's lens coverage — centre (%.1f, %.1f), semi-axes %.1f / %.1f px", circle.center[1], circle.center[2], circle.a, circle.b))
    external = circle === nothing ? nothing : VgtkCore.pixel_mask(circle, W, H)   # the pixels whose centres are inside the circle (compute_mask's `external`)
    K = length(poses); roles = split_roles(K, Int(s["split_every"]); fold)
    say("split: fold $fold of the rotation (k − $fold) % $(s["split_every"]): test 0, validation 1, the rest fit")
    t0 = time(); samples = Vector{Vector{NTuple{4, Float64}}}(undef, K); masked = zeros(Int, K); values = Vector{Any}(nothing, K); weights = Vector{Any}(nothing, K)
    Threads.@threads for k in 1:K
        r = sample_pixels(ctx, k, external; fraction = Float64(s["pixel_fraction"]), seed = Int(s["seed"]), backend = device)
        samples[k] = r.px; masked[k] = r.masked; values[k] = r.values; weights[k] = r.weights   # each photograph read once, by its type
    end
    say(@sprintf("pixels: %d photographs masked in %.0f s — median %d pixels on the sheet per photograph, %d sampled in all (fraction %.3f)", K, time() - t0,
                 round(Int, median(masked)), sum(length, samples), Float64(s["pixel_fraction"])))
    cell_mm = Float64(s["cell_mm"]); ncx = ceil(Int, ctx.page_mm[1] / cell_mm); ncy = ceil(Int, ctx.page_mm[2] / cell_mm)
    cnt = cell_coverage(samples, ncx, ncy, cell_mm)
    cells = select_cells(cnt, roles; min_fit_views = Int(s["min_fit_views"]), min_val_views = Int(s["min_val_views"]), min_test_views = Int(s["min_test_views"]),
                         min_pixels = Int(s["min_pixels"]))
    isempty(cells) && error("refine: no cell is covered by ≥ $(s["min_fit_views"]) fit, ≥ $(s["min_val_views"]) validation and ≥ $(s["min_test_views"]) test photographs with ≥ $(s["min_pixels"]) pixels")
    run_page(ctx, samples, roles, cells, circle, dir; settings = s, device, values, weights, log = say)
    Produced(; inputs = [inputs; state; images; product_dir(poses_product())])
end

# ---- evaluate: the report ------------------------------------------------------------------------------------------------------
report_product() = config_product("evaluate", CFG, ("report",); upstream = ("refine" => texture_product(),), output = "report.md")
# every stage below produces (or loads, by its key) the products it reads before its own: `with_upstream` (src/drivers.jl)
report(p) = produce(p; product_records(DRIVER, CFG_PATH)...) do dir
    key = p.key
    say = logger(dir)
    tdir = produce_upstream(texture_product(), "the page solve the report reads";
                            driver_run(DRIVER, CFG_PATH, "prepare,detect,sheet,init,refine")..., gpu = true)   # its refine stage: a GPU's
    rep, textures = report_page(tdir; settings = CFG["report"], log = say)
    design = page_design(tdir, rep["TW"], rep["TH"])
    write_textures(dir, textures, design; tex_mm = rep["tex_mm"])
    # VisualGeometryToolkitEval's tables: the curves per blur model and weight, the per-cell block RMS
    write_tsv(joinpath(dir, "curves.tsv"), ("variant", "weight", "val_rel_rms", "test_rel_rms", "test_rel_rms_centre", "test_rel_rms_edge", "rms_texel", "rms_block", "iters"),
              ((v, λ, rep["curves"][v]["val_rel_rms"][k], rep["curves"][v]["test_rel_rms"][k], rep["curves"][v]["test_rel_rms_centre"][k],
                rep["curves"][v]["test_rel_rms_edge"][k], rep["curves"][v]["rms_texel"][k], rep["curves"][v]["rms_block"][k], Int(rep["curves"][v]["iters"][k]))
               for v in rep["variants"] for (k, λ) in enumerate(rep["weights"])))
    ks = ["cell_i", "cell_j", ["rms_block_$v" for v in rep["variants"]]...]
    write_tsv(joinpath(dir, "per_cell.tsv"), ks, ([r[k] for k in ks] for r in rep["per_cell"]))
    rep["upstream"] = Dict("refine" => texture_product().key, "init" => poses_product().key, "refinement" => upstream_product().key, "case" => CASE)
    VgtkIO.write_record(joinpath(dir, "report.json"), merge(Dict{String, Any}("schema" => EXPERIMENT_RECORD_SCHEMAS.texture_recovery_report), rep); allownan = true)
    open(joinpath(dir, "report.md"), "w") do io
        println(io, "# Texture recovery (", CFG["experiment"]["name"], "): ", rep["photographs"], " photographs of the sheet — fit ", rep["roles"]["fit"], " / validation ",
                rep["roles"]["validation"], " / test ", rep["roles"]["test"], "; ", rep["cells"], " cells of ", rep["cell_mm"], " mm scored")
        println(io, "\nCamera = case ", CASE, " of ", relpath(REFINEMENT_PATH, projectdir()), " held fixed (image circle ", rep["image_circle"] === nothing ? "none" : "the refinement's",
                "); poses from the sheet's blobs alone; the pixel model V · (blur · texture); the whole page in one CGLS solve to convergence (rtol ", rep["cgls_rtol"],
                ", at most ", rep["cgls_max_iter"], " iterations) per Tikhonov weight β of λ ‖∇x‖², λ = β · Core's penalty_scale (β ∈ {", join(rep["weights"], ", "), "}); each blur model's weight: ", rep["rule"], ".\n")
        println(io, "| blur model | aperture r | λ | validation rel-RMS | test rel-RMS all / centre / edge | texture RMS vs design ", rep["tex_mm"], " mm / ", rep["block_mm"], " mm | CGLS iterations |")
        println(io, "|---|---|---|---|---|---|---|")
        for v in rep["variants"]
            c = rep["chosen"][v]
            @printf(io, "| %s | %.4f mm | %g | %.4f | %.4f / %.4f / %.4f | %.4f / %.4f | %d%s |\n", v, 1e3 * rep["blur_parameters"][v]["r_m"], c["weight"], c["val_rel_rms"], c["test_rel_rms"],
                    c["test_rel_rms_centre"], c["test_rel_rms_edge"], c["rms_texel"], c["rms_block"], c["iters"], c["converged"] ? "" : " (not converged)")
        end
        if haskey(rep, "paired")
            q = rep["paired"]
            @printf(io, "\nPaired over the %d cells (learned − pinhole, each at its weight): texture RMS at %g mm mean %+.4f (95 %% bootstrap [%+.4f, %+.4f]), learned lower in %d cells; test rel-RMS %+.4f.\n",
                    q["cells"], rep["block_mm"], q["rms_block_mean_diff"], q["rms_block_diff_ci95"]..., q["learned_lower_cells"], q["test_rel_rms_diff"])
        end
        println(io, "\nPer weight: curves.tsv (λ = 0 the unregularised texture); per cell: per_cell.tsv; the textures of record: texture_<model>.f32 / .png (design units).")
    end
    Produced(; inputs = [tdir])
end

# ---- folds: the rotation of the split, its report ---------------------------------------------------------------------------------
fold_report_product() = config_product("fold_report", CFG, ("report", "folds"); upstream = Tuple("fold$f" => texture_product(f) for f in 0:(fold_rotations() - 1)), output = "folds.json")
fold_report(p) = (with_upstream(() -> foreach(refine, 0:(fold_rotations() - 1)), p); produce(p; product_records(DRIVER, CFG_PATH)...) do dir
    say = logger(dir)
    dirs = [product_dir(texture_product(f)) for f in 0:(fold_rotations() - 1)]
    fold_report(dirs, dir; settings = CFG["report"], log = say)
    Produced(; inputs = dirs)
end)

# ---- compare: page runs scored by one scorer, as IO records and rasters -------------------------------------------------------------
"The page runs `[compare]` names (name → refine directory): its `runs` by key in this results root, and `this` the config's own page run."
function compare_runs()
    m = CFG["compare"]; base = dirname(product_dir(texture_product()))
    runs = Pair{String, String}[n => joinpath(base, k) for (n, k) in sort(collect(m["runs"]); by = first)]
    haskey(m, "this") && push!(runs, m["this"] => product_dir(texture_product()))
    return runs
end
compare_product() = config_product("compare", CFG, ("report", "compare"); upstream = haskey(CFG["compare"], "this") ? ("this" => texture_product(),) : (), output = "chosen.json")
compare(p) = (with_upstream(() -> haskey(CFG["compare"], "this") && refine(0), p); produce(p; product_records(DRIVER, CFG_PATH)...) do dir
    say = logger(dir)
    runs = compare_runs()
    compare_page_runs(runs, dir; settings = CFG["report"], crop_mm = CFG["compare"]["crop_mm"], reference = CFG["compare"]["reference"], log = say)
    Produced(; inputs = last.(runs))
end)

# ---- report: summary.md from the records ---------------------------------------------------------------------------------------------
"The products the summary reads: the evaluate stage's report, and the comparison and the folds' report when the config has them."
function summary_upstream()
    u = Pair{String, Any}["evaluate" => report_product()]
    haskey(CFG, "compare") && push!(u, "compare" => compare_product())
    haskey(CFG, "folds") && push!(u, "fold_report" => fold_report_product())
    return Tuple(u)
end
summary_product() = config_product("report", CFG, ("texture", "report"); upstream = summary_upstream(), output = "summary.md")
"The producers of the summary's upstream products, by name (`summary_upstream`)."
const SUMMARY_PRODUCERS = Dict("evaluate" => () -> report(report_product()), "compare" => () -> compare(compare_product()),
                               "fold_report" => () -> fold_report(fold_report_product()))
summary(p) = (with_upstream(() -> foreach(((n, _),) -> SUMMARY_PRODUCERS[n](), summary_upstream()), p); produce(p; product_records(DRIVER, CFG_PATH)...) do dir
    ups = summary_upstream()
    rec(n, f) = (i = findfirst(u -> first(u) == n, ups); i === nothing ? nothing : VgtkIO.read_record(joinpath(product_dir(last(ups[i])), f); allownan = true))
    path = write_texture_summary(joinpath(dir, "summary.md"); experiment = CFG["experiment"]["name"], texture = CFG["texture"], report = rec("evaluate", "report.json"),
                                 chosen = rec("compare", "chosen.json"), folds = rec("fold_report", "folds.json"),
                                 products = Dict(n => product_dir(q) for (n, q) in ups), run_keys = Dict(n => q.key for (n, q) in ups),
                                 rc_version = string(pkgversion(RadiometricCamera)))
    println(">> summary → ", path)
    Produced(; inputs = [product_dir(last(u)) for u in ups])
end)

# ---- figures ---------------------------------------------------------------------------------------------------------------------
# A stacked figure also reads its photograph (the [poses] images copy), its pose (the init stage) and the camera held fixed (the
# upstream refinement's state): the init product is an upstream of the figures', as the report's is.
figures_product() = config_product("figures", CFG, ("figures", "render"); upstream = ("evaluate" => report_product(), "init" => poses_product()), output = "numbers.tex")
"""
    stacked_photograph(t, rep) -> Matrix{Float64}

The photograph of the stacked figure `t` on the page's texels (`page_photograph`): `t["photograph"]`, the decoded copy BlobBoards read
(the `[poses] images` input, its grey channel through Features' linear reader), under its pose of the init stage through the
refinement's camera held fixed. The figure shows it as a training photograph, so it must be one the page was fitted to
(`split_roles`: a fit photograph): a validation or test photograph, or one without a pose, is refused.
"""
function stacked_photograph(t, rep)
    poses = read_poses(joinpath(product_dir(poses_product()), "poses.json"))
    k = findfirst(p -> p[1] == t["photograph"], poses)
    k === nothing && error("[figures] photograph $(t["photograph"]): no pose of that name in the init stage's $(length(poses))")
    role = split_roles(length(poses), Int(CFG["texture"]["split_every"]))[k]
    role == :fit || error("[figures] photograph $(t["photograph"]) is a $role photograph (photograph $k of $(length(poses)), [texture] split_every = " *
                          "$(CFG["texture"]["split_every"])): the figure shows a training photograph — name a fit one")
    img = VgtkFeatures.load_linear_image(joinpath(config_input(CFG, CFG["poses"]["images"]), t["photograph"]))[:, :, 1]
    _, R, tr = poses[k]
    return page_photograph(img, RC.load_state(require_state()).camera.geometric, R, tr; TW = Int(rep["TW"]), TH = Int(rep["TH"]), tex_mm = Float64(rep["tex_mm"]))
end
figures(p) = produce_figures(p, figure_assets(CFG); driver = DRIVER, config_path = CFG_PATH,
                             inputs = [product_dir(report_product()), product_dir(poses_product()), config_input(CFG, CFG["poses"]["images"])]) do dir
    key = p.key
    rdir = product_dir(report_product()); rep = plain(VgtkIO.read_record(joinpath(rdir, "report.json"); allownan = true))
    for (f, t) in sort(collect(CFG["figures"]); by = first)
        path = t["kind"] == "closeup" ? figure_texture_closeup(rdir, rep; render = t, dir = plotsdir()) :
               t["kind"] == "stacked" ? figure_texture_stacked(rdir, rep; render = t, dir = plotsdir(), photograph = stacked_photograph(t, rep)) :
               figure_texture_page(rdir, rep; render = t, dir = plotsdir())
        println(">> $f → ", path)
    end
    prefix = CFG["render"]["numbers_prefix"]; digits = CFG["render"]["significant_digits"]
    nums = driver_numbers(CFG_PATH, DRIVER, key)
    if !isempty(prefix)                                                # "": no macros (the paper cites none)
        for v in rep["variants"]
            c = rep["chosen"][v]
            Reproduce.add_number!(nums, Reproduce.number_macro_name(prefix, v, "weight"), c["weight"]; sigdigits = digits)
            Reproduce.add_number!(nums, Reproduce.number_macro_name(prefix, v, "test rel rms"), c["test_rel_rms"]; sigdigits = digits)
            Reproduce.add_number!(nums, Reproduce.number_macro_name(prefix, v, "test centre rel rms"), c["test_rel_rms_centre"]; sigdigits = digits)
            Reproduce.add_number!(nums, Reproduce.number_macro_name(prefix, v, "test edge rel rms"), c["test_rel_rms_edge"]; sigdigits = digits)
            Reproduce.add_number!(nums, Reproduce.number_macro_name(prefix, v, "texture rms texel"), c["rms_texel"]; sigdigits = digits)
            Reproduce.add_number!(nums, Reproduce.number_macro_name(prefix, v, "texture rms block"), c["rms_block"]; sigdigits = digits)
        end
        Reproduce.add_number!(nums, Reproduce.number_macro_name(prefix, "cells"), Int(rep["cells"]))
        Reproduce.add_number!(nums, Reproduce.number_macro_name(prefix, "photographs"), Int(rep["photographs"]))
        Reproduce.add_number!(nums, Reproduce.number_macro_name(prefix, "fit photographs"), Int(rep["roles"]["fit"]))
        Reproduce.add_number!(nums, Reproduce.number_macro_name(prefix, "test photographs"), Int(rep["roles"]["test"]))
        Reproduce.add_number!(nums, Reproduce.number_macro_name(prefix, "block mm"), rep["block_mm"]; sigdigits = digits)
        if haskey(rep, "paired")
            q = rep["paired"]
            Reproduce.add_number!(nums, Reproduce.number_macro_name(prefix, "paired texture diff"), q["rms_block_mean_diff"]; sigdigits = digits)
            Reproduce.add_number!(nums, Reproduce.number_macro_name(prefix, "paired texture diff ci low"), q["rms_block_diff_ci95"][1]; sigdigits = digits)
            Reproduce.add_number!(nums, Reproduce.number_macro_name(prefix, "paired texture diff ci high"), q["rms_block_diff_ci95"][2]; sigdigits = digits)
            Reproduce.add_number!(nums, Reproduce.number_macro_name(prefix, "paired learned lower"), Int(q["learned_lower_cells"]))
            Reproduce.add_number!(nums, Reproduce.number_macro_name(prefix, "paired test diff"), q["test_rel_rms_diff"]; sigdigits = digits)
        end
    end
    Reproduce.write_numbers(joinpath(dir, "numbers.tex"), nums)
    nothing
end

# ---- bands: the paper's figure ------------------------------------------------------------------------------------------------------
# fig:texture-learned as the paper prints it: the generator's two parts (src/paper_figures/texture_learned.jl) on this run's own
# records. Its part `tables` cuts the bands' text lines out of the evaluate stage's textures and design, chooses the training
# photograph from the init stage's poses among the refine stage's fit photographs, and takes that photograph's band from an input:
# cutting it out of the RAW photograph needs a demosaic and a feature registration (rawpy, OpenCV), an author-side step whose two
# outputs are inputs of record (`[bands] photograph_cut`, `photograph_record`). The part refuses a record of another photograph
# than the one it chooses, or of other text lines than this layout's: a run that moves the choice needs the step run again.
# A stage of its own, after the driver's figures (as the convergence driver's panels beside its figure): it alone reads that input
# — a data root without it must not stop the figures and their numbers.tex — and the page solve's roles beside the report, and its
# product is keyed apart from the renders of [figures].
"""The bands product (its output figure.tex, the figure's main file beside its tables; the tables in `tables/`, the figure kept in
`assets/`): keyed by `[bands]`, the photograph band's two inputs (tier, path, sha256) and the keys of the three products it reads.
One result at any thread count."""
bands_product() = driver_product("bands", Dict{String, Any}("experiment" => CFG["experiment"], "bands" => CFG["bands"],
                                                            "inputs" => inputs_record(CFG, [CFG["bands"][k] for k in BANDS_INPUT_KEYS]),
                                                            "upstream" => Dict("evaluate" => report_product().key, "init" => poses_product().key,
                                                                               "refine" => texture_product().key));
                                 output = "figure.tex", threads = nothing)
"""
    write_bands(t, dir; evaluate, poses, roles) -> Vector{String}

The paper's figure of the `[bands]` table `t` from a run's records, into `dir`; the files written. `evaluate` is an evaluate
product's directory (report.json, design.png, texture_pinhole.png, texture_learned.png), `poses` an init product's poses.json and
`roles` a refine product's page.json (photographs, roles). The photograph band's two inputs are written where the generator's part
`tables` reads them — `dir/tables`, as T. Harder's texture_learned.py left them there: the cut stays, the record is rewritten with
the whole figure's numbers (meta.toml) — then the parts `tables` (→ `dir/tables`: meta.toml and one 8-bit image per band) and
`figure` (→ `dir/assets`: `<asset_dir>.tex`, the folder of its bands, its caption) run, with every setting of `t` handed to the
generator; the files name this stage as their generator.
"""
function write_bands(t::AbstractDict, dir::AbstractString; evaluate::AbstractString, poses::AbstractString, roles::AbstractString)
    tables = joinpath(dir, "tables"); mkpath(tables)
    write(joinpath(tables, "photograph.png"), read(config_input(CFG, t["photograph_cut"])))      # written, not copied: an input's file may be read-only
    write(joinpath(tables, "meta.toml"), read(config_input(CFG, t["photograph_record"])))
    bands = [(; name = b, source = t[b]["source"], label = t[b]["label"], caption = t[b]["caption"]) for b in t["order"]]
    return PaperFigures.generate("texture-learned"; part = "all", data = tables, out = joinpath(dir, "assets"), evaluate, poses, roles,
                                 asset_dir = t["asset_dir"], bands, read_on = RadiometricCamera.read_switch(t["read_on"], "[bands] read_on"),
                                 block_mm = t["block_mm"], lines_per_band = t["lines_per_band"], white = t["white"], pixels_per_texel = t["pixels_per_texel"],
                                 generator = "experiments/$DRIVER (stage bands; T. Harder's texture_learned.py)")
end
"""Stage bands: the paper's figure from the report (its textures and design), the poses and the page solve (its roles), each
produced on demand; kept in the product (`assets/`) and published to plotsdir() (`produce_figures`)."""
bands(p) = produce_figures(p, String[]; driver = DRIVER, config_path = CFG_PATH,
                           inputs = [product_dir(report_product()), product_dir(poses_product()), product_dir(texture_product()),
                                     (config_input(CFG, CFG["bands"][k]) for k in BANDS_INPUT_KEYS)...]) do dir
    edir = produce_upstream(report_product(), "the report whose textures the bands show"; make = () -> report(report_product()),
                            command = driver_run(DRIVER, CFG_PATH, "evaluate").command)
    idir = produce_upstream(poses_product(), "the poses the training photograph is chosen by"; make = () -> poses(poses_product()),
                            command = driver_run(DRIVER, CFG_PATH, "prepare,detect,sheet,init").command)
    tdir = produce_upstream(texture_product(), "the page solve whose roles name the training photographs";
                            driver_run(DRIVER, CFG_PATH, "prepare,detect,sheet,init,refine")..., gpu = true)
    files = write_bands(CFG["bands"], dir; evaluate = edir, poses = joinpath(idir, "poses.json"), roles = joinpath(tdir, "page.json"))
    figure = joinpath(dir, "assets", CFG["bands"]["asset_dir"] * ".tex")
    println(">> bands → ", figure, " (", length(files), " files: the tables, the figure, its bands, its caption)")
    cp(figure, joinpath(dir, "figure.tex"); force = true)                      # the output, written last
end

"The refine stage solves on the GPU (src/texture_gpu.jl); every other stage on the CPU. No stage hands a [pipeline] to bin/refine.jl."
const GPU_STAGES = [:refine, :folds]

const STAGES = [
    Stage(:prepare, "the sheet's photographs decoded as the refinement's (Features' load_raw)";
          items = () -> [prepare_product(CFG, SET)], product = identity, run = _ -> run_prepare(CFG, SET; driver = DRIVER, config_path = CFG_PATH)),
    Stage(:detect, "the photographs find_boards reads and the board's design JSON (tier rc_baselines)"; check = detect),
    Stage(:sheet, "the sheet's target record (IO's target: target.json beside texture.png) from the sheet input's files"; items = () -> [sheet_product()], product = identity, run = sheet),
    Stage(:init, "per photograph its pose from the sheet's blobs alone (BlobBoards' find_boards, its pose arm through the refinement's camera held fixed; poses.json)";
          items = () -> [poses_product()], product = identity, run = poses),
    Stage(:refine, "the page's texture under both blur models, unregularised and per Tikhonov weight (GPU)"; items = () -> [0], product = texture_product, run = refine),
    Stage(:folds, "[folds]: the page runs of the split's other rotations (GPU; --fold=f selects)"; items = fold_items, product = texture_product, run = refine),
    Stage(:evaluate, "the report: each model's weight by the validation photographs, the test and texture scores, the paired test, the textures";
          items = () -> [report_product()], product = identity, run = report),
    Stage(:fold_report, "[folds]: every fold scored, the mean validation curve's weight, the spread of the chosen weight and of the paired margin (folds.json)";
          items = () -> haskey(CFG, "folds") ? [fold_report_product()] : [], product = identity, run = fold_report),
    Stage(:compare, "[compare]: the page runs scored by one scorer — curves.json, chosen.json and the crops as raster records";
          items = () -> haskey(CFG, "compare") ? [compare_product()] : [], product = identity, run = compare),
    Stage(:report, "summary.md: the metrics' definitions, the results per run and blur model, the folds' spread, the provenance (from the records)";
          items = () -> [summary_product()], product = identity, run = summary),
    Stage(:figures, "fig:texture-closeup → plotsdir(), numbers.tex"; items = () -> [figures_product()], product = identity, run = figures),
    Stage(:bands, "[bands]: the paper's figure fig:texture-learned (plotsdir(<asset_dir>.tex, <asset_dir>/<band>.{tex,png}, <asset_dir>-caption.tex))";
          items = () -> haskey(CFG, "bands") ? [bands_product()] : Product[], product = identity, run = bands),
]

run_stages("texture_recovery", STAGES)
