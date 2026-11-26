#!/usr/bin/env julia
# reproduce: reads configs/experiments/chrome_evaluation_fx3_14mm_central30cm.toml
# reproduce: reads configs/experiments/chrome_evaluation_fx3_14mm_central30cm_ag_nc.toml
# reproduce: reads configs/experiments/chrome_evaluation_fx3_14mm_central30cm_nc.toml
# reproduce: reads configs/experiments/chrome_evaluation_fx3_fisheye_10cm_ag.toml
# reproduce: reads configs/experiments/chrome_evaluation_fx3_fisheye_10cm_ag_nc.toml
# reproduce: reads configs/experiments/chrome_evaluation_fx3_tamron_largeaperture.toml
# provenance: every product writes _config.toml and _provenance.toml (VisualGeometryToolkitEval produce) beside its results.
# =============================================================================
# scripts/chrome_evaluation.jl — the chrome-board evaluation of the geometric-accuracy experiment: the Thorlabs R3L3S7N1 dot-grid
# target (17 × 17 dots at 5.000 mm), backlit and photographed by the Sony FX3 under a rotation about one point, per lens and session
# (configs/experiments/chrome_evaluation_fx3_<set>.toml); the models are the refinement cases of configs/experiments/radiometric_refinement.toml
# and their geometric starts.
#
# STAGES (each a product keyed by its config record, the inputs it reads and the keys of the products it reads; datadir("exp_pro", <experiment>, <stage>, <key>))
#   prepare   (a config with [prepare]) each [prepare.<set>]'s raw input decoded (Features' load_raw, every mode) in its
#             `mode` (src/drivers.jl RAW_MODES; "raw_green": the bilinear green on the raw grid, "bin2": the cell grid): <set>_linear/ (conversion.json states
#             the images' frame and its map onto the raw), <set>_mask/, exif
#   detect    detect_sfop then the seeded detect_foerstner (VisualGeometryToolkitFeatures) on each photograph of [detect] photographs
#             (a raw_green set: its RAW mosaic under the frame's sensor model, Features' bilinear green on the raw grid),
#             each seed in the window of its selected scale, on [detect] device (the GPU: CUDA_VISIBLE_DEVICES=0): detections/<photo>.json
#             (src/chrome_evaluation.jl: position, 2 × 2 covariance px², label, window, F; the photograph's COLMAP pixels, as detected:
#             a RAW mosaic's raw pixels) and summary.tsv (per photograph: its size, the counts per label)
#   index     each photograph's circular points registered against the grid of the target record through the camera of record:
#             VisualGeometryToolkitEstimation's RANSAC on src/chrome_evaluation.jl's GridPoseProblem (register_grid): each hypothesis
#             scored in board space (the dots' rays met with the board, matched one-to-one to the board window's nodes by the
#             Hungarian with a dustbin at half a pitch, MSAC at that half pitch, mm), its local optimization
#             [index.local_optimization_strategy] (plain least squares on the scorer's matches, frozen), then the Hungarian under the
#             final pose and the pixel refine [index.refine_strategy] on its matches until they repeat; one connected component of
#             matched nodes or refused): points/<photo>.json (vgio/target_points2d), poses/<photo>.json
#             (vgio/target_poses), registration/<photo>.tsv, targets.json (vgio/rig: the record's target, which the point indices
#             resolve in), summary.tsv (per photograph: circular, inliers, matched, unassigned, trials, rms_px, max_px) and
#             photos.json (per photograph its registration, seconds and every local-optimization call: Estimation's CountedLO, the
#             index grid's rows); src/chrome_evaluation.jl register_photos
#   evaluate  every model of [evaluate].models on the registered dots of the photographs with every dot of the grid assigned
#             (src/chrome_evaluation.jl evaluate_grid_models): the rotation-centre motion fit from the index stage's poses,
#             residuals in px through the model's local scale: eval_<label>.json, summary.tsv, summary.md (the head-to-head on
#             the dots every model scores, each model's coverage beside it, and each model on all of its own dots; the Schöps
#             baseline, kind "schoeps", is T. Schöps's own camera, refused otherwise), and paired.tsv: the paired photograph
#             bootstrap of [evaluate] comparisons (src/chrome_evaluation.jl paired_differences). The start is
#             fit_rotation_center (c, d from the dots' bearings at the index rotations); the refine is Estimation's
#             LevenbergMarquardt on CameraPosesProblem of RotationCenterViews, the model's camera held, at [evaluate] optimization_strategy (required;
#             an Estimation strategy table, type "triggs": the Triggs-corrected LM at efficiency_constant × the index stage's pooled
#             noise scale), capped at [evaluate] iterations; eval_<label>.json's `fit` records its retcode, iterations, scale and width. The optional weights / holdout switches whiten the dots by the detect run's covariances and add
#             the held-out variant (EVALUATE_FIT_SWITCHES). The registrations are the
#             index stage's ([evaluate] index = "stage") or an index run of record given as an input (a config without [detect] and
#             [index]); a model is an input or a case of [upstream] radiometric_refinement
#             (scripts/radiometric_refinement.jl runs it)
#
# The paper's chrome tables and figures from every set: scripts/chrome_evaluation_paper.jl (configs/experiments/chrome_evaluation_paper.toml).
#
#   julia --project=experiments experiments/scripts/chrome_evaluation.jl [--config=<toml>] [--plan] [--stage=NAME[,NAME]] [--force]
#
# Data: VGTK_DATA_ROOT binds data_root() (REPRODUCE.md): the decoded chrome photographs, the target record and the camera of record
# (configs/data_artifacts.toml; scripts/fetch_data.jl). The stages run on the CPU.
# =============================================================================
using DrWatson
@quickactivate "experiments"
include(srcdir("experiments.jl"))
include(srcdir("drivers.jl"))
include(srcdir("radiometric_refinement.jl"))
include(srcdir("chrome_evaluation.jl"))

const DRIVER = "scripts/chrome_evaluation.jl"
const CFG_PATH = abspath(cli_option(ARGS, "config", projectdir("configs", "experiments", "chrome_evaluation_fx3_14mm_central30cm.toml")))   # default: the config of record
const CFG = load_chrome_evaluation_config(CFG_PATH)
const PREPARE = sort!(collect(String, keys(get(CFG, "prepare", Dict{String, Any}()))))   # the sets, checked by load_chrome_evaluation_config
const DETECT = haskey(CFG, "detect") ? detect_settings(CFG, CFG_PATH) : nothing
# the detect device, resolved at top level: `rc_backend` loads the GPU vendor packages for "gpu", and the stage functions below run after this
# statement, in the world that has CUDA's KernelAbstractions methods (resolved inside `detect()` they would not: a MethodError for
# KA's `allocate` on the CUDABackend, 2026-11-11)
const DETECT_DEVICE = DETECT === nothing ? nothing : RadiometricCamera.rc_backend(DETECT.device; inline = false)
# the device the product records: the configured one, or the CPU it fell back to
const DETECT_DEVICE_NAME = DETECT === nothing ? nothing :
    DETECT.device == "gpu" && DETECT_DEVICE isa VgtkCore.KA.CPU ? "cpu (fallback: no functional GPU)" : DETECT.device
const INDEX = haskey(CFG, "index") ? index_settings(CFG, CFG_PATH) : nothing
const EVAL = haskey(CFG, "evaluate") ? evaluate_settings(CFG, CFG_PATH) : nothing

# ---- detect -------------------------------------------------------------------------------------------------------------------
detect(p) = produce(p; product_records(DRIVER, CFG_PATH)..., fields = Dict{String,Any}("device" => DETECT_DEVICE_NAME)) do dir
    Produced(; inputs = [_detect(dir)])
end
function _detect(dir)
    if DETECT.prepared
        produce_upstream(prepare_product(CFG, DETECT.photos), "the prepare set $(DETECT.photos) detect reads";
                         make = () -> run_prepare(CFG, DETECT.photos; driver = DRIVER, config_path = CFG_PATH),
                         command = driver_run(DRIVER, CFG_PATH, "prepare").command)
        photos = prepared_images(CFG, DETECT.photos)
    else
        photos = only(require_stage_inputs(CFG, "detect"))
    end
    names = check_photo_list(DETECT, photos)
    println(">> detect_sfop → detect_foerstner on ", DETECT_DEVICE_NAME)
    out = joinpath(dir, "detections")
    mkpath(out)
    rows = Tuple[]
    # a raw_green set's photographs are read as their RAW mosaics under each frame's sensor model (`raw_photographs`), detected on
    # Features' bilinear green image on the raw grid; a luma set's are its decoded photographs (no sensor model:
    # `sensor_characterisation` empty)
    raw = DETECT.image == "raw_green" ? raw_photographs(CFG, DETECT.photos, names, DETECT.sensor_characterisation) : nothing
    raw === nothing || VgtkIO.write_record(joinpath(dir, "sensor_models.json"),
        Dict{String, Any}("schema" => EXPERIMENT_RECORD_SCHEMAS.sensor_models, "sensor_characterisation" => DETECT.sensor_characterisation,
                          "photos" => RadiometricCamera.sensor_record(names, raw.models)))
    for (i, name) in enumerate(names)
        img = raw === nothing ? detector_image(photos, name) : (VgtkFeatures.load_raw_mosaic(raw.files[i]), raw.models[i])
        points = detect_photo(img, DETECT, DETECT_DEVICE)
        write_detections(joinpath(out, splitext(name)[1] * ".json"), name, points)   # the photograph's COLMAP pixels
        c, k, u = label_counts(points)
        w, h = photograph_size(img)
        push!(rows, (name, w, h, c, k, u, length(points)))
        println(">> ", name, " (", w, " x ", h, "): circular ", c, ", corner ", k, ", unknown ", u)
        flush(stdout)
    end
    write_tsv(joinpath(dir, "summary.tsv"), DETECT_SUMMARY_COLUMNS, rows)              # VisualGeometryToolkitEval's table: the output, written last
    println(">> detect → ", dir)
    return photos
end

"The detect stage runs on the GPU where [detect] device is \"gpu\" (the configs of record's)."
const GPU_STAGES = DETECT !== nothing && DETECT.device == "gpu" ? [:detect] : Symbol[]
const STAGES = Stage[]
isempty(PREPARE) || push!(STAGES, Stage(:prepare, "the raw photographs decoded per [prepare.<set>] (Features' load_raw)";
                                        items = () -> PREPARE, product = s -> prepare_product(CFG, s),
                                        run = s -> run_prepare(CFG, s; driver = DRIVER, config_path = CFG_PATH)))
DETECT === nothing || push!(STAGES, Stage(:detect, "detect_sfop → seeded detect_foerstner on each photograph (detections/<photo>.json, summary.tsv)";
                                          items = () -> [detect_product(CFG)], product = identity, run = detect))
# src/chrome_evaluation.jl's register_photos (the one loop, the index grid's too) and evaluate_models: each produces its product
INDEX === nothing || push!(STAGES, Stage(:index, "the circular points registered against the target record's grid through the camera of record (points/, poses/, registration/, targets.json, summary.tsv, photos.json)";
                                         items = () -> [index_product(CFG)], product = identity, run = _ -> register_photos(CFG, CFG_PATH; driver = DRIVER)))
EVAL === nothing || push!(STAGES, Stage(:evaluate, "every model of [evaluate] on the registered dots: the rotation-centre motion fit from the index poses (eval_<label>.json, summary.tsv, summary.md)";
                                        items = () -> [evaluate_product(CFG)], product = identity, run = _ -> evaluate_models(CFG, CFG_PATH; driver = DRIVER)))

run_stages("chrome_evaluation", STAGES)
