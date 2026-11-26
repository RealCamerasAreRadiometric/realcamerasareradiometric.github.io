# =============================================================================
# src/frontends.jl — the geometric front ends of the paper's sets as a stage of this project (scripts/frontends.jl, its config of
# record configs/experiments/frontends.toml). The external calibrators are the one allowed exception, run from their pinned
# containers: T. Schöps's camera_calibration (detection + central-generic). Everything around them is the stack's:
#
#   prepare   the board photographs decoded by Features' raw module (`decode_raw`, src/drivers.jl: the one RAW path) into the
#             geometry image the case states (`image = "raw_green"`: the bilinear green on the raw grid at full resolution, each
#             frame under its sensor model — the chrome detect's image, owner 2026-11-16; its frame map onto the raw the identity),
#             and the tool's input written from that decode: an 8-bit
#             sRGB PNG per photograph (Features' `srgb_image`, the one display encoding) — the format the frozen chain fed the tool
#   detect    the tool on those images (its arguments verbatim from the frozen driver, each with its citation), then its run
#             IMPORTED through VisualGeometryToolkitIO (`load(BoundedCamera, intrinsics0.yaml)`, `load(TargetObservations, dataset.bin)`)
#             and moved onto the raw grid through the decode's frame map (the identity for raw_green; Core's `reframed_calibration` for the
#             camera, the map itself for the detections): `camera.json` (RadiometricCamera's `save_grid`) and `detections.json`
#             (IO's vgio/target_points2d), the files RadiometricCamera's `schoeps_run_file` reads before the tool's own
#   init      the index cameras: Estimation's `Ransac` on `AbsolutePoseProblem` per view, then `LevenbergMarquardt` on `CameraPosesProblem` (src/parametric_starts.jl
#             `_fit_start`, the one fit of a parametric start) on the imported detections; RadiometricCamera's `save_lens`
#
# Every run is a RESULT, keyed under results_root() (src/frontends_runs.jl: <results>/frontends/<stage>/<key>, by its table, its
# inputs and the keys of the runs it reads; never by the thread count: no step sums over threads); a config that reads one names it
# in its `[runs]` table (src/drivers.jl). Included after src/drivers.jl and src/parametric_starts.jl.
# =============================================================================
using Printf: @printf
using StableRNGs: StableRNG
using LinearAlgebra: normalize

"The tables of configs/experiments/frontends.toml."
const FRONTENDS_TABLES = ("experiment", "inputs", "image_regions", "cases", "schoeps", "fits", "compare")
"""The keys of a `[cases.<set>]`, a set of photographs (the runner's array: one task per case), every one required: its RAW input
(stage prepare), the geometry image the tool reads (`image`: \"raw_green\", the bilinear green on the raw grid — the chrome detect's
image, owner 2026-11-16; no default), the sensor characterisation each frame is read under (an input of stage prepare, the chrome
detect's), the decode's gain (a number, or \"auto\": the tool images' exposure) and the set's lens coverage (`lens_coverage`:
\"sensor\" or \"image_region:<name>\", src/image_region.jl), which bounds the set's cameras (the Schöps import's and the fits')."""
const FRONTEND_CASE_KEYS = ("raw", "image", "sensor_characterisation", "gain", "lens_coverage")
"""The keys of a `[schoeps.<run>]`: the decoded set (`images`, a case), the pattern and board inputs (stage detect), the container
image (an input of stage detect, pinned by its sha256), the tool's arguments."""
const FRONTEND_SCHOEPS_KEYS = ("images", "pattern", "board", "image", "args")
"The keys of a `[fits.<name>]`: the Schöps run whose import it fits, the lens family, the board input (stage init), the nominal
focal length and sensor width, the least detections a view needs, the pose starts' scorer and the LM's iteration cap."
const FRONTEND_FIT_KEYS = ("detections", "model", "board", "f_mm", "sensor_mm", "min_features", "scoring", "max_iter")
"""The keys of a `[compare.<name>]`: what is compared (`kind`: a Schöps run or an index fit, by its table name `run`) with the start of
record (`record`, an input of stage evaluate: a Schöps run's directory or a camera record), on every `spacing_px`-th pixel."""
const FRONTEND_COMPARE_KEYS = ("kind", "run", "record", "spacing_px")
"The kinds of a comparison: a Schöps run (`[schoeps.<run>]`) or an index fit (`[fits.<name>]`)."
const FRONTEND_COMPARE_KINDS = ("schoeps", "fit")
"Where the camera_calibration binary is in its image (experiments/containers/schoeps.def)."
const SCHOEPS_BINARY = "/opt/camera_calibration/build/applications/camera_calibration/camera_calibration"
"The arguments the stage owns, never a config's: the tool's input and output paths."
const SCHOEPS_OWN_ARGS = ("--image_directories", "--pattern_files", "--dataset_output_path", "--output_directory", "--dataset_files")

"""
    load_frontends_config(path) -> Dict

configs/experiments/frontends.toml checked (`load_driver_config`): `[cases.<set>]` with `FRONTEND_CASE_KEYS` (its raw input of
stage prepare, a positive gain), `[schoeps.<run>]` with `FRONTEND_SCHOEPS_KEYS` (a case, inputs of stage detect: the pattern, the
board and the container image `*.sqsh` with its sha256), the tool's arguments as strings without the stage's own `SCHOEPS_OWN_ARGS`), `[fits.<name>]` with `FRONTEND_FIT_KEYS`
(a Schöps run, a family of `START_MODELS`, an input of stage init, the scorer read by Estimation's `scoring_from_toml`).
"""
function load_frontends_config(path::AbstractString)
    c = load_driver_config(path, "frontends"; tables = FRONTENDS_TABLES, case_keys = FRONTEND_CASE_KEYS)
    for t in ("cases", "schoeps")
        haskey(c, t) && c[t] isa AbstractDict && !isempty(c[t]) || throw(ArgumentError("$path: no [$t] table"))
    end
    check_tables(c, path, "cases", FRONTEND_CASE_KEYS; required = FRONTEND_CASE_KEYS)
    check_tables(c, path, "schoeps", FRONTEND_SCHOEPS_KEYS; required = FRONTEND_SCHOEPS_KEYS)
    check_tables(c, path, "fits", FRONTEND_FIT_KEYS; required = FRONTEND_FIT_KEYS)
    input_of(id, stage, where) = (t = get(c["inputs"], id, nothing); t !== nothing && t["stage"] == stage ||
                                  throw(ArgumentError("$where: $(repr(id)) is not an input of stage $stage")))
    for (s, t) in c["cases"]
        input_of(t["raw"], "prepare", "$path [cases.$s] raw")
        RadiometricCamera.check_geometry_image(t, "$path [cases.$s]") == "raw_green" ||
            throw(ArgumentError("$path [cases.$s] image: the front ends decode RAW photographs, image = \"raw_green\" (\"luma\" is a decoded image's)"))
        input_of(t["sensor_characterisation"], "prepare", "$path [cases.$s] sensor_characterisation")
        t["gain"] == "auto" || t["gain"] isa Real && t["gain"] > 0 ||
            throw(ArgumentError("$path [cases.$s] gain: a positive number, or \"auto\" (decode_raw: the set's 99.99th percentile → 0.9)"))
        check_lens_coverage(c, t["lens_coverage"], "$path [cases.$s]")
    end
    for (n, t) in c["schoeps"]
        where = "$path [schoeps.$n]"
        haskey(c["cases"], t["images"]) || throw(ArgumentError("$where images: no [cases.$(t["images"])]"))
        input_of(t["pattern"], "detect", "$where pattern"); input_of(t["board"], "detect", "$where board"); input_of(t["image"], "detect", "$where image")
        endswith(c["inputs"][t["image"]]["path"], ".sqsh") && haskey(c["inputs"][t["image"]], "sha256") ||
            throw(ArgumentError("$where image: a container image input (<name>.sqsh) pinned by its sha256"))
        t["args"] isa AbstractString && !isempty(schoeps_args(t)) || throw(ArgumentError("$where args: the tool's command line, a non-empty string"))
        bad = intersect(schoeps_args(t), SCHOEPS_OWN_ARGS)
        isempty(bad) || throw(ArgumentError("$where args: $(join(bad, ", ")) are the stage's (the decoded images, the pattern input, the run's directory)"))
    end
    for (n, t) in get(c, "fits", Dict{String, Any}())
        where = "$path [fits.$n]"
        haskey(c["schoeps"], t["detections"]) || throw(ArgumentError("$where detections: no [schoeps.$(t["detections"])]"))
        t["model"] in START_MODELS || throw(ArgumentError("$where model: one of $(join(START_MODELS, ", "))"))
        input_of(t["board"], "init", "$where board")
        t["f_mm"] > 0 && t["sensor_mm"] > 0 || throw(ArgumentError("$where f_mm, sensor_mm: positive millimetres"))
        t["min_features"] isa Integer && t["min_features"] >= 3 || throw(ArgumentError("$where min_features: an integer ≥ 3 (P3P)"))
        t["max_iter"] isa Integer && t["max_iter"] >= 1 || throw(ArgumentError("$where max_iter: an integer ≥ 1"))
        t["scoring"] isa AbstractDict || throw(ArgumentError("$where scoring: a scorer table ({ type = \"msac\", threshold = … })"))
        VgtkEstimation.scoring_from_toml(t["scoring"]; problem = VgtkEstimation.AbsolutePoseProblem, section = "$where scoring")
    end
    check_tables(c, path, "compare", FRONTEND_COMPARE_KEYS; required = FRONTEND_COMPARE_KEYS)
    for (n, t) in get(c, "compare", Dict{String, Any}())
        where = "$path [compare.$n]"
        t["kind"] in FRONTEND_COMPARE_KINDS || throw(ArgumentError("$where kind: one of $(join(FRONTEND_COMPARE_KINDS, ", "))"))
        haskey(get(c, t["kind"] == "schoeps" ? "schoeps" : "fits", Dict()), t["run"]) || throw(ArgumentError("$where run: no [$(t["kind"] == "schoeps" ? "schoeps" : "fits").$(t["run"])]"))
        input_of(t["record"], "evaluate", "$where record")
        t["spacing_px"] isa Integer && t["spacing_px"] >= 1 || throw(ArgumentError("$where spacing_px: an integer ≥ 1"))
    end
    return c
end

# ---- prepare: the decode and the tool's images ------------------------------------------------------------------------------

"""
    write_tool_images(linear_dir, out) -> Int

The external tool's input from a raw_green decode (`decode_raw`'s `<set>_linear`, the 16-bit linear bilinear green on the raw grid): per photograph an
8-bit sRGB grey PNG of the same values (Features' `srgb_image`, the one display encoding; the decode's gain is its exposure), and
`out/tool_images.json` naming the encoding and the photographs. Returns the number written.
"""
function write_tool_images(linear_dir::AbstractString, out::AbstractString)
    assert_not_input(out)
    mkpath(out)
    names = filter(RC.is_png_file, RC.photo_files(linear_dir))
    isempty(names) && throw(ArgumentError("write_tool_images: no decoded photograph in $linear_dir"))
    Threads.@threads for n in names
        g = VgtkFeatures.load_linear_image(joinpath(linear_dir, n))          # the geometry image, linear, as decoded (a grey PNG: its plane 1)
        PNGFiles.save(joinpath(out, n), VgtkFeatures.srgb_image(N0f8, Float32.(view(g, :, :, 1:1))))
    end
    VgtkIO.write_record(joinpath(out, "tool_images.json"),
                        Dict("schema" => EXPERIMENT_RECORD_SCHEMAS.tool_images, "encoding" => "8-bit sRGB grey (VisualGeometryToolkitFeatures.srgb_image) of the geometry image as decoded (gain applied); its frame, conversion.json beside the decode",
                             "photos" => names))
    return length(names)
end

"""
    tool_photos(dir) -> Vector{String}

The photographs of the tool's input `dir` ([`write_tool_images`](@ref)), as its record `tool_images.json` names them (sorted): the
folder also holds that record (and whatever else), so it is never listed for them.
"""
function tool_photos(dir::AbstractString)
    r = VgtkIO.read_record(joinpath(dir, "tool_images.json"))
    r.schema == EXPERIMENT_RECORD_SCHEMAS.tool_images || throw(ArgumentError("$dir/tool_images.json: not a $(EXPERIMENT_RECORD_SCHEMAS.tool_images) record ($(repr(r.schema)))"))
    return String[String(n) for n in r.photos]
end

"""
    run_frontend_decode(cfg, set; driver, config_path) -> dir

Stage prepare of `[cases.<set>]`: `decode_raw` (Features' raw module, the case's `image`, its frames' sensor models, the set's gain) into the run, then the tool's
images (`write_tool_images`), and the run's records.
"""
function run_frontend_decode(cfg::AbstractDict, set::AbstractString; driver::AbstractString, config_path::AbstractString)
    p = frontend_decode_product(cfg, set)
    produce(p; product_records(driver, config_path)...) do dir
        key = p.key
        t = cfg["cases"][set]
        raw = config_input(cfg, t["raw"])
        src = selected_raw_dir(cfg, t["raw"], dir)
        t0 = time()
        linear = decode_raw(src, dir, set; gain = t["gain"] == "auto" ? "auto" : Float64(t["gain"]), mode = t["image"], srgb = false,
                            characterisation = config_input(cfg, t["sensor_characterisation"]))
        t1 = time()
        n = write_tool_images(linear, joinpath(dir, "$(set)_tool"))
        @printf("decode %s: %d photographs, load_raw %.1f s, tool images %.1f s (%d threads)\n", set, n, t1 - t0, time() - t1, Threads.nthreads())
        Produced(; inputs = [raw])
    end
end

# ---- detect: the tool in its container, then the import ------------------------------------------------------------------

"""
    schoeps_command(image_dir, pattern, out, args; container) -> Cmd

camera_calibration in its enroot container `container` (started with offscreen Qt; the job's GPU through enroot's NVIDIA hook)
on the tool images `image_dir` with the pattern `pattern`, its arguments `args` (the config's, verbatim), the dataset and the
calibration written into `out`. The directories are mounted at their real paths.
"""
function schoeps_command(image_dir, pattern, out, args; container::AbstractString)
    dirs = unique(realpath.([image_dir, dirname(realpath(pattern)), out]))
    mounts = reduce(vcat, [["-m", "$d:$d"] for d in dirs])
    return `enroot start -e QT_QPA_PLATFORM=offscreen $mounts $container $SCHOEPS_BINARY --image_directories $(realpath(image_dir)) --pattern_files $(realpath(pattern)) $args --dataset_output_path $(joinpath(realpath(out), "dataset.bin")) --output_directory $(realpath(out))`
end

"""
    view_feature_counts(tp) -> Vector{Tuple{String, Int}}

Per view of the observations `tp` (a `TargetObservations`), in frame order: its image and its count of features over every
target. A view with none counts 0 (Schöps's dataset.bin keeps the images it found no pattern in).
"""
view_feature_counts(tp) = [(tp.images[t], sum(length, values(tp[t]); init = 0)) for t in sort(collect(keys(tp)))]

"""
    import_schoeps_run(dir, S, raw_size, target; lens_coverage) -> (; camera, detections, views)

The Schöps run in `dir` (its own `intrinsics0.yaml`, `dataset.bin`, in the pixels of the images it read) through
VisualGeometryToolkitIO, moved onto the raw grid by `S` (the decode's frame map onto the raw's COLMAP pixels: the identity for raw_green): the camera
by Core's `reframed_calibration` (K is the spline's gauge: the lens is unchanged) bounded by `raw_size` with the set's lens coverage
`lens_coverage` (raw pixels: the sensor's rectangle or the set's measured image region, `lens_coverage_region`; never the tool's
own bound), written by RadiometricCamera's `save_grid` as `dir/camera.json`; the detections through `S` (a covariance through its
linear part, Core's `propagate`) as `dir/detections.json` (vgio/target_points2d, `target` the board's content id). Returns the
camera, the detections and the per-view counts (also `dir/views.txt`, "<image> <features>").
"""
function import_schoeps_run(dir::AbstractString, S, raw_size::VgtkCore.Size2, target::AbstractString; lens_coverage)
    cam = VgtkIO.load(BoundedCamera, joinpath(dir, "intrinsics0.yaml"))
    m = VgtkCore.model(cam)
    m′ = VgtkCore.CameraModel(VgtkCore.LogicalIntrinsics(VgtkCore.Cameras.reframed_calibration(S, m.intrinsics.K)), m.lens)
    cam′ = BoundedCamera(m′, raw_size; lens_coverage)
    RC.save_grid(joinpath(dir, "camera.json"), cam′; provenance = (; source = "T. Schöps's camera_calibration (central-generic) in $dir, read by VisualGeometryToolkitIO and moved onto the raw grid by the decode's frame map (Core reframed_calibration)"))
    tp = VgtkIO.load(VgtkCore.TargetObservations, joinpath(dir, "dataset.bin"); target = :board)
    moved(xy::VgtkCore.Uncertain) = VgtkCore.propagate(S, xy)
    moved(xy) = SVector{2, Float64}(S(xy))
    tp′ = VgtkCore.TargetObservations{VgtkCore.PixelFrame{:colmap}}(
        Dict(t => Dict(id => VgtkCore.correspondences(VgtkCore.sources(cs), map(moved, VgtkCore.targets(cs))) for (id, cs) in sets)
             for (t, sets) in tp.frames);
        images = tp.images, sensors = isempty(tp.sensors) ? "camera0" : tp.sensors, times = tp.times, time_origin = tp.time_origin, targets = [target])
    VgtkIO.save(joinpath(dir, "detections.json"), tp′; provenance = (; source = "dataset.bin of $dir through the decode's frame map onto the raw grid"))
    views = view_feature_counts(tp)
    open(joinpath(dir, "views.txt"), "w") do io
        foreach(((n, k),) -> println(io, n, " ", k), views)
    end
    return (; camera = cam′, detections = tp′, views)
end

"""
    run_frontend_schoeps(cfg, name; driver, config_path) -> dir

Stage detect of `[schoeps.<name>]`: the container (its image input, made an enroot container of this job, removed after) on
the decode's tool images, its log `run.log`, then `import_schoeps_run` onto the raw grid (the size of the decode's photographs
through the frame map), and the run's records. Needs a GPU (the tool's).
"""
function run_frontend_schoeps(cfg::AbstractDict, name::AbstractString; driver::AbstractString, config_path::AbstractString)
    p = frontend_schoeps_product(cfg, name)
    produce(p; product_records(driver, config_path)...) do dir
        key = p.key
        t = cfg["schoeps"][name]
        produce_upstream(frontend_decode_product(cfg, t["images"]), "[schoeps.$name]: the decode of $(t["images"])";
                         make = () -> run_frontend_decode(cfg, t["images"]; driver, config_path),
                         command = driver_run(driver, config_path, "prepare"; case = t["images"]).command)
        pattern = config_input(cfg, t["pattern"]); board = config_input(cfg, t["board"])
        sqsh = config_input(cfg, t["image"])                   # the pinned image: Eval checks its sha256
        images = frontend_tool_images(cfg, t["images"])
        container = "rcfe-$(key)-$(gethostname())-$(getpid())"      # this process's own container (the image unpacked for it, removed after)
        t0 = time()
        run(`enroot create -f -n $container $sqsh`)
        t1 = time()
        try
            ok = open(joinpath(dir, "run.log"), "w") do io
                success(pipeline(schoeps_command(images, pattern, dir, schoeps_args(t); container); stdout = io, stderr = io))
            end
            ok || error("[schoeps.$name]: camera_calibration failed: $(joinpath(dir, "run.log"))\n" * join(last(readlines(joinpath(dir, "run.log")), 15), "\n"))
        finally
            run(`enroot remove -f $container`)
        end
        t2 = time()
        dec = product_dir(frontend_decode_product(cfg, t["images"]))
        S = prepared_frame_map(joinpath(dec, "$(t["images"])_linear"))
        photos = tool_photos(images)                                    # the photographs, by the record (the folder holds it too)
        first_photo = VgtkFeatures.load_linear_image(joinpath(images, first(photos)))
        raw_size = VgtkCore.Size2(round.(Int, S(SVector{2, Float64}(size(first_photo, 2), size(first_photo, 1))))...)
        lc = lens_coverage_region(cfg, cfg["cases"][t["images"]]["lens_coverage"], raw_size...; what = "[schoeps.$name]")   # the set's coverage, raw px
        imp = import_schoeps_run(dir, S, raw_size, VgtkIO.content_id(joinpath(board, "target.json")); lens_coverage = lc)
        n = length(photos)
        localized = count(v -> last(v) > 0, imp.views)
        @printf("schoeps %s: container %.0f s, camera_calibration %.0f s, import %.1f s; %d of %d photographs with features, %d features (median %d per view); %s\n",
                name, t1 - t0, t2 - t1, time() - t2, localized, n, sum(last, imp.views; init = 0),
                isempty(imp.views) ? 0 : round(Int, median(last.(imp.views))), schoeps_report(dir))
        Produced(; inputs = [pattern, board, images])
    end
end

"The reprojection summary camera_calibration wrote (`report_camera0_info.txt`'s reprojection_error lines), or \"no report\"."
function schoeps_report(dir::AbstractString)
    f = joinpath(dir, "report_camera0_info.txt")
    isfile(f) || return "no report"
    return join(strip.(filter(l -> occursin("reprojection_error", l), readlines(f))), "; ")
end

# ---- init: the index cameras ------------------------------------------------------------------------------------------------

"""The Schöps run `[schoeps.<name>]` a consumer (`what`) reads, produced on demand (`produce_upstream`: its stage, a GPU's —
Schöps's camera_calibration needs CUDA)."""
frontend_upstream_schoeps(cfg::AbstractDict, name::AbstractString, what::AbstractString; driver::AbstractString, config_path::AbstractString) =
    produce_upstream(frontend_schoeps_product(cfg, name), "$what: the Schöps run $name"; make = () -> run_frontend_schoeps(cfg, name; driver, config_path),
                     command = driver_run(driver, config_path, "prepare,detect"; case = cfg["schoeps"][name]["images"]).command, gpu = true)

"""
    run_frontend_fit(cfg, name; driver, config_path) -> dir

Stage init of `[fits.<name>]`: `_fit_start` (src/parametric_starts.jl: Estimation's `Ransac` on `AbsolutePoseProblem` per view, then
Core's `LevenbergMarquardt` on `CameraPosesProblem`) of the family `model` on the imported detections of its Schöps run (`schoeps_views` of the run: the
views with ≥ `min_features`), the board's design points (`board_points`), the camera of the raw grid's size; `camera.json`
(RadiometricCamera's `save_lens`, `write_start`) and the fit's report `fit.toml`.
"""
function run_frontend_fit(cfg::AbstractDict, name::AbstractString; driver::AbstractString, config_path::AbstractString,
                          rng = StableRNG(1))
    p = frontend_fit_product(cfg, name)
    produce(p; product_records(driver, config_path)...) do dir
        key = p.key
        t = cfg["fits"][name]
        frontend_upstream_schoeps(cfg, t["detections"], "[fits.$name]"; driver, config_path)
        board = config_input(cfg, t["board"])
        sch = product_dir(frontend_schoeps_product(cfg, t["detections"]))
        sc = VgtkIO.load(BoundedCamera, RC.schoeps_run_file(sch, :camera))
        W, H = VgtkCore.extent(sc)
        views = schoeps_views(sch, board_points(board); min_features = Int(t["min_features"]))
        t0 = time()
        r = _fit_start(t["model"], views, W, H, t["f_mm"], t["sensor_mm"]; scoring = t["scoring"], max_iter = Int(t["max_iter"]), rng)
        params = start_parameters(r.camera)
        @printf("fit %s: %s on %d views (%d correspondences, %d not projected), %s (%s) after %d iterations, rms %.4f px [%.1f s]\n", name, t["model"],
                length(r.names), r.n_correspondences, r.n_dropped, r.retcode, r.stop_rule, r.iterations, r.rms, time() - t0)
        write_start(joinpath(dir, "camera.json"), r.camera, W, H; lens_coverage = VgtkCore.lens_coverage(sc),   # the set's, as its import holds it
                    provenance = (; model = name, source = "scripts/frontends.jl (configs/experiments/frontends.toml [fits.$name])",
                                  method = "VisualGeometryToolkitEstimation Ransac on AbsolutePoseProblem per view, then LevenbergMarquardt on CameraPosesProblem",
                                  detections = sch, views = length(r.names), correspondences = r.n_correspondences, rms_px = r.rms,
                                  retcode = string(r.retcode), stop_rule = string(r.stop_rule)))
        report = Dict{String, Any}("fit" => name, "model" => t["model"], "retcode" => string(r.retcode), "stop_rule" => string(r.stop_rule),
                                   "iterations" => r.iterations, "rms_px" => r.rms, "views" => length(r.names), "correspondences" => r.n_correspondences,
                                   "not_projected" => r.n_dropped, "start_xi_alpha" => collect(r.start), "parameters" => params,
                                   "seconds" => time() - t0, "view_rms_px" => Dict(n => v for (n, v) in zip(r.names, r.view_rms)))
        open(io -> TOML.print(io, report), joinpath(dir, "fit.toml"), "w")
        Produced(; inputs = [board, sch])
    end
end

# ---- evaluate: the comparison with the starts of record ------------------------------------------------------------------------

"""
    run_frontend_compare(cfg, name; driver, config_path) -> dir

Stage evaluate of `[compare.<name>]`: the camera a front-end run made (a Schöps run's import `camera.json`, or an index fit's) against
the start of record (`record`: Schöps's own run directory, read as IO reads it, or a camera record), both on the raw grid: their
directions at every `spacing_px`-th pixel (Evaluation's `direction_scores`: the angle × the record's f_x [px], raw and with the camera
frame's gauge rotation removed; a pixel either does not image is dropped); for a fit, its parameters and rms beside the record's
parameters; for a Schöps run, its registered views and features, both runs' reprojection reports and the directions again at the
record's own detections (where it was measured: a grid spline is extrapolated beyond them). Writes `compare.toml`.
"""
function run_frontend_compare(cfg::AbstractDict, name::AbstractString; driver::AbstractString, config_path::AbstractString)
    p = frontend_compare_product(cfg, name)
    produce(p; product_records(driver, config_path)...) do dir
        key = p.key
        t = cfg["compare"][name]
        schoeps = t["kind"] == "schoeps"
        src = schoeps ? frontend_upstream_schoeps(cfg, t["run"], "[compare.$name]"; driver, config_path) :
                        produce_upstream(frontend_fit_product(cfg, t["run"]), "[compare.$name]: the fit $(t["run"])";
                                         make = () -> run_frontend_fit(cfg, t["run"]; driver, config_path),
                                         command = driver_run(driver, config_path, "prepare,detect,init"; case = cfg["schoeps"][cfg["fits"][t["run"]]["detections"]]["images"]).command,
                                         gpu = !produced(frontend_schoeps_product(cfg, cfg["fits"][t["run"]]["detections"])))   # its Schöps run, when missing
        record = config_input(cfg, t["record"])
        new = RC.load_camera((schoeps ? "grid:" : "lens:") * joinpath(src, "camera.json"))
        old = RC.load_camera((schoeps ? "schoeps:" : "lens:") * record)
        W, H = VgtkCore.extent(old)
        VgtkCore.extent(new) == VgtkCore.extent(old) ||
            throw(ArgumentError("[compare.$name]: the run's camera is $(VgtkCore.extent(new)), the record's $(VgtkCore.extent(old)) px"))
        s = Float64(t["spacing_px"])
        P = [SVector{2, Float64}(x, y) for y in (s / 2):s:H for x in (s / 2):s:W]
        dirs(c) = [normalize(SVector{3, Float64}(VgtkCore.backproject(c, p))) for p in P]
        F = Float64(VgtkCore.calibration_matrix(VgtkCore.model(old))[1, 1])
        sc = Evaluation.direction_scores(dirs(old), dirs(new), F)
        out = Dict{String, Any}("compare" => name, "kind" => t["kind"], "run" => src, "record" => record, "image_size" => [W, H], "spacing_px" => t["spacing_px"],
                                "directions_px" => Dict(String(k) => v for (k, v) in pairs(sc)))
        if schoeps
            views = [(String(a), parse(Int, b)) for (a, b) in split.(readlines(joinpath(src, "views.txt")))]
            out["run_views"] = length(views); out["run_features"] = sum(last, views; init = 0)
            out["run_report"] = schoeps_report(src); out["record_report"] = schoeps_report(record)
            rec_det = RC.dataset_features(RC.schoeps_run_file(record, :detections))
            out["record_views"] = length(rec_det)
            # where the record was measured: its own detections (a grid spline is extrapolated beyond them; a fisheye's grid mostly is)
            Q = [SVector{2, Float64}(x, y) for (_, fs) in rec_det for (x, y, _) in fs]
            qd(c) = [normalize(SVector{3, Float64}(VgtkCore.backproject(c, q))) for q in Q]
            out["directions_at_record_detections_px"] = Dict(String(k) => v for (k, v) in pairs(Evaluation.direction_scores(qd(old), qd(new), F)))
        else
            fit = TOML.parsefile(joinpath(src, "fit.toml"))
            out["run_parameters"] = fit["parameters"]; out["run_rms_px"] = fit["rms_px"]; out["run_views"] = fit["views"]
            out["record_parameters"] = start_parameters(VgtkCore.model(old))
            prov = get(VgtkIO.read_record(record), "provenance", Dict())
            out["record_provenance"] = Dict(String(k) => string(v) for (k, v) in pairs(prov))
        end
        open(io -> TOML.print(io, out; sorted = true), joinpath(dir, "compare.toml"), "w")
        @printf("compare %s: %d pixels (%d dropped), directions raw median %.3f / p95 %.3f px, gauge median %.3f / p95 %.3f px, rotation %.3f mrad\n",
                name, sc.n, sc.dropped, sc.raw_median, sc.raw_p95, sc.gauge_median, sc.gauge_p95, sc.rotation_mrad)
        Produced(; inputs = [record, src])
    end
end
