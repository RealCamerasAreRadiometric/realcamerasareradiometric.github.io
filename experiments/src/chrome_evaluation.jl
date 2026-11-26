# =============================================================================
# src/chrome_evaluation.jl — the chrome evaluation config of record and its detect stage, shared by
# scripts/chrome_evaluation.jl and its tests. Included after src/drivers.jl.
#
# The detector (`detect_photo`): Features' SFOP scale selection for circular points (`detect_sfop`), each seed localised at full
# resolution by the seeded Förstner window estimation (`detect_foerstner(img, seeds)`) in the window of its own selected scale — no
# per-photograph integration scale. Its F-test (corner vs circular at `significance_level`) runs at the residual sums' effective
# degrees of freedom, so windows between dots, on line artefacts and on letter strokes are not labelled circular.
#
# The detections of a photograph are written as VisualGeometryToolkitIO's detection file (`vgio/detection`, unassociated
# detections: the dots are not assigned to the grid yet), one record per detect_foerstner row: its point (COLMAP pixel coordinates
# of the photograph, top-left pixel centre at (0.5, 0.5)) with its 2 × 2 covariance in px² (absent for an :unknown row, the window
# centre with no estimate, its covariance_status saying why), its label (circular | corner | unknown), window centre (COLMAP px),
# F (the classification statistic Ω_corner / Ω_circular) and covariance_status (Features' producer policy).
#
# The detector's grey image (`[detect] image`, stated in the config so that its intention is explicit):
#   luma         a photograph of an input (a JPEG as the camera wrote it, or a decoded PNG) read as grey, Gray.(·); the detections
#                are in its own pixels
#   raw_green    the RAW photographs of a `[prepare.<set>]` of mode "photosites" (prepare links them and records their EXIF,
#                nothing decoded), each read by detect as its mosaic (Features' `load_raw_mosaic`, Core's `RawMosaic`) under its frame's sensor model, and detected on Features' BILINEAR green
#                image on the raw grid (`detect_sfop(m::RawMosaic; sensor_model)`, `detect_foerstner(m, seeds; sensor_model)`,
#                Features 0.9.6: a green photosite its own value, a red or blue one the mean of its usable green 4-neighbours;
#                saturated and defect photosites masked; SFOP's 2× scale space for the seeds; the covariances propagated from the
#                photosites). The detections are the RAW's COLMAP pixels as detected (no map); the selected scales are raw pixels.
# =============================================================================
using VisualGeometryToolkitFeatures: detect_sfop, detect_foerstner, load_linear_image, gray_f32
using Statistics: cov
using StaticArrays: SVector
using JpegTurbo: jpeg_decode
using ColorTypes: Gray, RGB

"The tables of a Thorlabs config of record (configs/experiments/chrome_evaluation_<variant>.toml)."
const CHROME_TABLES = ("experiment", "inputs", "image_regions", "runs", "upstream", "prepare", "detect", "index", "evaluate")
"The keys of `[upstream]`: the configs of record whose runs `[evaluate]` models read (`radiometric_refinement`: scripts/radiometric_refinement.jl's cases)."
const CHROME_UPSTREAM_KEYS = ("radiometric_refinement",)
"""The detector keywords of `[detect]`, every one required (no code default): `significance_level` and `precision_threshold`
detect_sfop's (its λ₂ test and its precision test T_w = 1/precision_threshold²; significance_level also the seeded
detect_foerstner's F-test), `derivative` and `derivative_sigma` the seeded detect_foerstner's, `polarity` SFOP's gate. A luma set
states `noise_sigma` too (`DETECT_LUMA_KEYS`: its λ₂ test's grey-level noise); a raw_green set refuses it — SFOP's λ₂ test reads
the sensor model's noise law (the frame's gain), so the key would be inert and only key the product."""
const DETECT_KEYWORDS = ("derivative", "derivative_sigma", "precision_threshold", "significance_level", "polarity")
"The `[detect]` keys of a luma set alone: the grey-level noise of its decoded photographs (no sensor model)."
const DETECT_LUMA_KEYS = ("noise_sigma",)
"The blob signs `[detect] polarity` names (Features' polarity gate of detect_sfop and the seeded detect_foerstner, ASIFT's vocabulary)."
const DETECT_POLARITIES = ("light", "dark", "none")
"The keys of `[detect]`: the photographs (an input or a prepare set), how they become the detector's grey image, the precision and
device (a tag of RadiometricCamera's `BACKEND_TAGS`), the keywords, the photographs detected (`photographs`), the excluded ones and
the camera's sensor characterisation (`sensor_characterisation`: the input id of its vgio/sensor_characterisation record, or empty for decoded photographs;
RadiometricCamera's entry point `photo_set`), whose per-frame model a RAW photograph's mosaic is linearised and detected under."
const CHROME_DETECT_KEYS = ("photos", "image", "precision", "device", DETECT_KEYWORDS..., "photographs", "excluded",
                            "sensor_characterisation")
"The detector's grey images `[detect] image` names (this file's header)."
const DETECT_IMAGES = RadiometricCamera.GEOMETRY_IMAGES   # the one vocabulary of the geometry images ("raw_green", "luma")
"The element types `[detect] precision` names."
const DETECT_PRECISIONS = Dict("Float32" => Float32, "Float64" => Float64)
"The summary table's header: per photograph, its size and the counts per label."
const DETECT_SUMMARY_COLUMNS = ("image", "width", "height", "circular", "corner", "unknown", "total")

"""
    load_chrome_evaluation_config(path) -> Dict

The chrome evaluation config of record, every table and key checked (`load_driver_config`, `check_prepare_tables`, `detect_settings`,
`index_settings`, `evaluate_settings`). A config runs the stages it has, in order — `[prepare]`, `[detect]`, `[index]`, `[evaluate]`,
each optional — but `[index]` needs `[detect]` (the index stage reads the detect run), and `[evaluate]` reads either the `[index]`
stage or an index run of record that is an input (`[evaluate] index`): the photographs' detection and registration of that run are
then not codified here, only their result.
"""
function load_chrome_evaluation_config(path::AbstractString)
    c = load_driver_config(path, "chrome_evaluation"; tables = CHROME_TABLES)
    haskey(c, "index") && !haskey(c, "detect") && throw(ArgumentError("$path: [index] needs [detect] (the index stage reads the detect run)"))
    any(k -> haskey(c, k), ("prepare", "detect", "evaluate")) || throw(ArgumentError("$path: no stage — none of [prepare], [detect], [evaluate]"))
    if haskey(c, "upstream")
        VgtkCore.check_keys(c["upstream"], CHROME_UPSTREAM_KEYS, "$path [upstream]")
        for (k, v) in c["upstream"]
            v isa AbstractString && isfile(projectdir(v)) || throw(ArgumentError("$path [upstream] $k: $(repr(v)) is not a config of this project"))
        end
    end
    check_prepare_tables(c, path)
    for (set, t) in get(c, "prepare", Dict{String, Any}())     # the chrome stage reads its RAW photographs as an input of stage prepare
        haskey(c["inputs"], t["raw"]) && c["inputs"][t["raw"]]["stage"] == "prepare" ||
            throw(ArgumentError("$path [prepare.$set] raw: \"$(t["raw"])\" is not an input of stage prepare"))
    end
    haskey(c, "detect") && detect_settings(c, path)
    haskey(c, "index") && index_settings(c, path)
    haskey(c, "evaluate") && evaluate_settings(c, path)
    return c
end

"""
    detect_settings(cfg, path = "the chrome evaluation config") -> NamedTuple

`[detect]` checked: every key of `CHROME_DETECT_KEYS` required (an `ArgumentError` names a missing or unknown one), a luma set's
`noise_sigma` required and a raw_green set's refused (`DETECT_LUMA_KEYS`), `photos` an
input of stage detect or a `[prepare.<set>]`, `image` one of `DETECT_IMAGES` (`raw_green` exactly for a prepare set of mode
"photosites"), a known precision, a device tag of RadiometricCamera's `BACKEND_TAGS` (`gpu`, the configs of record's, or `cpu`, the test oracle; resolved when the stage runs, RadiometricCamera's `rc_backend`), `photographs` a
non-empty array of distinct file names, `excluded` a table of reasons disjoint from it. Returns
`(; photos, prepared, image, T, device, sfop, window, sensor_characterisation, photographs, excluded)`: `prepared` whether `photos` is
a prepare set, `sfop` detect_sfop's keywords and `window` the seeded detect_foerstner's (`DETECT_KEYWORDS`; its `f_tests` the
configuration's one FoerstnerTests table, shared by every photograph's call), `photographs` sorted.
"""
function detect_settings(cfg::AbstractDict, path::AbstractString = "the chrome evaluation config")
    t = get(cfg, "detect", nothing)
    where = "$path [detect]"
    t isa AbstractDict || throw(ArgumentError("$path: no [detect] table"))
    VgtkCore.check_keys(t, (CHROME_DETECT_KEYS..., DETECT_LUMA_KEYS...), where)
    check_required_keys(Dict(k => v for (k, v) in t if !(k in DETECT_LUMA_KEYS)), CHROME_DETECT_KEYS, where)
    photos = t["photos"]
    prepared = haskey(get(cfg, "prepare", Dict()), photos)
    prepared || (haskey(cfg["inputs"], photos) && cfg["inputs"][photos]["stage"] == "detect") ||
        throw(ArgumentError("$where photos: \"$photos\" is neither an input of stage detect nor a [prepare.<set>]"))
    image = t["image"]
    image == "green_cells" && throw(ArgumentError("$where image: \"green_cells\" is retired (RC 0.10.2): a RAW set is detected on its mosaic's bilinear green on the raw grid, image = \"raw_green\""))
    image in DETECT_IMAGES || throw(ArgumentError("$where image: $(repr(image)) is not one of $(join(DETECT_IMAGES, ", "))"))
    # one RAW path (owner 2026-11-14): a prepare set (RAW photographs, read, not decoded) is detected on its RAW mosaics, a luma set is
    # an input's decoded photographs; never a decoded image of a RAW
    raw = prepared && photosites_set(cfg, photos)
    (image == "raw_green") == raw == prepared ||
        throw(ArgumentError("$where image: \"raw_green\" reads exactly a prepare set of mode \"photosites\", \"luma\" an input of stage detect (photos \"$photos\" is $(prepared ? "a prepare set of mode \"" * cfg["prepare"][photos]["mode"] * "\"" : "an input"))"))
    if image == "raw_green"
        haskey(t, "noise_sigma") && throw(ArgumentError("$where noise_sigma: a raw_green set is detected under its frames' sensor model, whose noise law (the gain) SFOP's λ₂ test reads — the key is inert: delete it"))
    else
        foreach(k -> VgtkCore.required_key(t, k, where), DETECT_LUMA_KEYS)
    end
    haskey(DETECT_PRECISIONS, t["precision"]) ||
        throw(ArgumentError("$where precision: \"$(t["precision"])\" is not one of $(sort(collect(keys(DETECT_PRECISIONS))))"))
    try                                               # RC's backend registry: its tags, and its refusal of a retired one ("cuda")
        RadiometricCamera.BACKEND_TAGS[t["device"]]
    catch e
        e isa Exception && throw(ArgumentError("$where device: $(repr(t["device"])): $(sprint(showerror, e))"))
        rethrow()
    end
    t["derivative"] isa AbstractString || throw(ArgumentError("$where derivative must be a string"))
    t["polarity"] in DETECT_POLARITIES || throw(ArgumentError("$where polarity: $(repr(t["polarity"])) is not one of $(join(DETECT_POLARITIES, ", "))"))
    for k in ("derivative_sigma", "precision_threshold", "significance_level", (image == "luma" ? DETECT_LUMA_KEYS : ())...)
        t[k] isa Real || throw(ArgumentError("$where $k must be a number"))
    end
    ph = t["photographs"]
    ph isa AbstractVector && !isempty(ph) && all(v -> v isa AbstractString, ph) && allunique(ph) ||
        throw(ArgumentError("$where photographs must be a non-empty array of distinct file names"))
    ex = excluded_table(t["excluded"], where)
    both = intersect(ph, keys(ex))
    isempty(both) || throw(ArgumentError("$where: $(sort(collect(both))) both in photographs and excluded"))
    level = Float64(t["significance_level"])
    # the polarity gate is SFOP's, on its candidates before the NMS: the seeds are of that sign, so the seeded windows are not gated again
    sfop = (; significance_level = level, precision_threshold = Float64(t["precision_threshold"]), polarity = Symbol(t["polarity"]),
            (image == "luma" ? (; noise_sigma = Float64(t["noise_sigma"])) : (;))...)
    # the F-test's critical values, computed once for this configuration and read by every photograph's call (Features'
    # FoerstnerTests: on the host, per window half-width; recomputed per photograph they were most of the GPU detect's time)
    window = (; derivative = Symbol(t["derivative"]), derivative_sigma = Float64(t["derivative_sigma"]), significance_level = level,
              f_tests = VgtkFeatures.FoerstnerTests())
    sc = t["sensor_characterisation"]
    sc == RadiometricCamera.NO_SENSOR_CHARACTERISATION &&
        throw(ArgumentError("$where sensor_characterisation: \"none\" is retired (RadiometricCamera 0.11.2): leave it empty — the photographs' kind decides (\"luma\": decoded, no sensor model)"))
    sc isa AbstractString && (isempty(sc) || haskey(cfg["inputs"], sc) && cfg["inputs"][sc]["stage"] == "detect") ||
        throw(ArgumentError("$where sensor_characterisation: $(repr(sc)) is neither empty nor an input of stage detect (the record)"))
    image == "luma" && !isempty(sc) &&
        throw(ArgumentError("$where sensor_characterisation: \"luma\" reads decoded photographs, which no sensor model describes: leave it empty"))
    image == "raw_green" && isempty(sc) &&
        throw(ArgumentError("$where sensor_characterisation: \"raw_green\" reads RAW mosaics, detected under their frames' sensor model: name the characterisation"))
    return (; photos, prepared, image, T = DETECT_PRECISIONS[t["precision"]], device = String(t["device"]), sfop, window,
            sensor_characterisation = String(sc), photographs = sort(String.(ph)), excluded = ex)
end

"""
    detect_product(cfg) -> Product

The detect product (its output summary.tsv, written last): keyed by `[detect]` without its `excluded` reasons (description: which
photographs run is `photographs`) and the photographs' input (tier, path) — or, for a prepare set (detected on its RAW mosaics, in
raw pixels), that set's product key (`upstream`).
"""
function detect_product(cfg::AbstractDict)
    d = Dict{String,Any}(k => v for (k, v) in cfg["detect"] if k != "excluded")
    photos = cfg["detect"]["photos"]
    rec = Dict{String,Any}("experiment" => cfg["experiment"], "detect" => d)
    if haskey(get(cfg, "prepare", Dict()), photos)
        rec["upstream"] = Dict{String,Any}("prepare" => prepare_product(cfg, photos).key)
    else
        rec["inputs"] = inputs_record(cfg, [photos])
    end
    sc = cfg["detect"]["sensor_characterisation"]   # the record the RAW mosaics are linearised under: its content keys the run
    isempty(sc) || (rec["sensor_characterisation"] = inputs_record(cfg, [sc]))
    return driver_product("detect", rec; output = "summary.tsv", threads = nothing)   # one detection per photograph
end

"""
    check_photo_list(settings, dir) -> Vector{String}

The photographs of the run, sorted: `photographs`. With `excluded` they must be exactly the files of `dir` (an
`ArgumentError` names a photograph the config does not account for, or one it names that is absent).
"""
function check_photo_list(s, dir::AbstractString)
    files = Set(filter(f -> !startswith(f, ".") && f != "conversion.json", readdir(dir)))
    named = union(Set(s.photographs), Set(keys(s.excluded)))
    missing_ = setdiff(named, files)
    isempty(missing_) || throw(ArgumentError("[detect] names photographs absent from $dir: $(sort(collect(missing_)))"))
    extra = setdiff(files, named)
    isempty(extra) || throw(ArgumentError("[detect]: the photographs $(sort(collect(extra))) of $dir are in neither photographs nor excluded"))
    return s.photographs
end

"""
    photograph_size(img) -> (width, height)

The size of the photograph the detector's image `img` is: its own (a RAW mosaic's: raw pixels).
"""
photograph_size(img::Tuple{VgtkCore.RawMosaic, VgtkCore.SensorModel}) = photograph_size(img[1])
photograph_size(img::AbstractMatrix) = (size(img, 2), size(img, 1))

"""
    write_detections(path, image, points) -> path

The detections `points` of detect_foerstner (Features' rows, `Attributed{Uncertain{PointFrame}}` with `label`, `window`, `F` and
`covariance_status`) of the photograph `image` as VisualGeometryToolkitIO's detection file (`vgio/detection`, IO's `save` of
`image => detections`) in the photograph's COLMAP pixels, as detected (the detector's image is the photograph: a RAW mosaic's raw
pixels, a luma photograph's own). A row without a covariance (an `:unknown` row, NaN) is written without one, its
`covariance_status` the detector's reason.
"""
function write_detections(path::AbstractString, image::AbstractString, points)
    rows = map(points) do d
        xy = SVector{2, Float64}(Tuple(VgtkCore.centroid(d.value))); w = SVector{2, Float64}(Tuple(d.attributes.window)); a = d.attributes
        VgtkCore.Attributed(Uncertain(VgtkCore.PointFrame(xy[1], xy[2]), CovMat(SMatrix{2, 2, Float64, 4}(cov(d.value)))),
                            (; label = a.label, window = VgtkCore.Point2(w[1], w[2]), F = Float64(a.F),
                             covariance_status = a.covariance_status))
    end
    VgtkIO.save(path, Dict(String(image) => collect(rows)))
    return path
end

"""
    detect_photo(img, s[, backend]) -> StructArray

The detections of the grey image `img` with the settings `s` (`detect_settings`), on `backend`: Features' `detect_sfop` (SFOP's
circular points and their selected scales: `significance_level`, `precision_threshold`, a luma image's `noise_sigma`), each seed localised by the
seeded `detect_foerstner` in the window of its scale (`derivative`, `derivative_sigma`, `significance_level`), precision `s.T`;
SFOP's polarity gate (`polarity`) keeps the seeds of the blobs of that sign.
A RAW photograph (`raw_photographs`: its mosaic with its frame's sensor model) is detected on Features' bilinear green image on the raw
grid under that model (Features 0.9.6's `detect_sfop(m::RawMosaic)` and seeded `detect_foerstner(m, seeds)`): SFOP's λ₂ test reads the
photosites' noise, and their Poisson–Gaussian noise (a gain) is propagated into each detection's covariance (`covariance_status`
:propagated_<source>); without a gain the documented a-posteriori covariance. Its rows are raw COLMAP pixels.
"""
detect_photo(img::AbstractMatrix, s, backend) =
    detect_foerstner(s.T, img, detect_sfop(img; device = backend, s.sfop...); device = backend, s.window...)
detect_photo(img::Tuple{VgtkCore.RawMosaic, VgtkCore.SensorModel}, s, backend) =
    detect_foerstner(s.T, img[1], detect_sfop(img[1]; sensor_model = img[2], device = backend, s.sfop...); sensor_model = img[2], device = backend, s.window...)
# The device resolved here: `rc_backend` may load CUDA, whose KernelAbstractions methods (`allocate` on a CUDABackend) exist only
# in the world after that load — the call is made in the latest world (a caller compiled before the load would hit KA's
# "unimplemented" fallback: the MethodError of the GPU detect on dgx01 and on 8ee1f0f4, 2026-11-11). A driver resolves the device
# once at top level (`scripts/chrome_evaluation.jl`), so its own calls are in the new world.
detect_photo(img, s) = Base.invokelatest(detect_photo, img, s, RadiometricCamera.rc_backend(s.device; inline = false))

"""A photograph as a grey image: a JPEG (the HX400V's, as the camera wrote it) or a linear PNG (prepare's decode, the front ends'
photographs), read by Features' one linear reader (`load_linear_image`: the samples as stored, no gAMA re-encoding) and made grey by
Features' `gray_f32` (Rec. 601 luma of the RGB samples, Float32)."""
function load_grey(path::AbstractString)
    endswith(lowercase(path), ".png") || return Gray.(jpeg_decode(path))
    img = load_linear_image(path)
    return gray_f32(RGB.(view(img, :, :, 1), view(img, :, :, 2), view(img, :, :, 3)))
end

"""
    detector_image(dir, name) -> AbstractMatrix

The detector's grey image of photograph `name` in `dir` for `[detect] image = "luma"` (this file's header): the photograph read as
grey. A `raw_green` set's photographs are read as their RAW mosaics (`raw_photographs`).
"""
detector_image(dir::AbstractString, name::AbstractString) = load_grey(joinpath(dir, name))

"""
    raw_photographs(cfg, set, names, characterisation) -> (; files, models)

The RAW photographs `names` of the photosites prepare set `set` (its folder of links, `photosite_images`; their EXIF in its
photosites.json, `run_prepare`), each frame's sensor model resolved by its EXIF from the characterisation (`[detect]`'s: an input
id, read through `config_input`, its sha256 checked) through RadiometricCamera's ONE entry point (`photo_set` under the observation
model of RAW photosites under that record, `observation_model`: the same resolution, and the same refusals, as the calibration's and
every prediction's). `detect` reads each as its mosaic
(Features' `load_raw_mosaic`) and detects it under its model ([`detect_photo`](@ref)).
"""
function raw_photographs(cfg::AbstractDict, set::AbstractString, names::AbstractVector{<:AbstractString}, characterisation::AbstractString)
    spec = config_input(cfg, characterisation)
    root = photosite_images(cfg, set)
    files = [joinpath(root, n) for n in names]
    models = raw_sensor_models(files, spec; what = "raw_green ($set)")   # the one resolution (src/drivers.jl), the front ends' decode's too
    return (; files, models)
end

"The label counts of `points`: `(circular, corner, unknown)`."
label_counts(points) = (count(==(:circular), points.attributes.label), count(==(:corner), points.attributes.label),
                        count(==(:unknown), points.attributes.label))

# =============================================================================
# Stage index: each photograph's circular points registered against the grid of the target record, through the camera of record.
#
# The grid is the record's: its nominal points (VisualGeometryToolkitIO load(AbstractTarget, …)), their rows, columns and spacing read
# from the record's provenance `design`; point (r − 1)·cols + c is node (c − 1, r − 1), at ((c − 1), (r − 1))·spacing. The camera is
# a camera of record (vgio/cameras) by its name. Units: the record's points are lengths; the stage strips the target once, in
# metres (`ustrip(m, target)`), so the pose estimators take plain metres and return the pose in metres; the pose is written as a
# length again (`with_unit(·, m)`).
#
# The registration (`register_grid`, owner 2026-11-14) is VisualGeometryToolkitEstimation's RANSAC on a problem with no
# correspondences given (`GridPoseProblem`), every step the framework's:
#   1. the minimal sample is a random dot and two of its wound neighbours a quarter turn apart (`GridSeedSampler`), the lattice's
#      (0, 0), (1, 0), (0, 1) when they are edge neighbours; with the camera of record the plane's pose [r₁ r₂ t] is P3P's;
#   2. a hypothesis is scored in BOARD space (the problem's `mahalanobis_distances!`): every dot's ray met with the board plane
#      z = 0 (`board_point`, no rounding) and the dots matched ONE-TO-ONE to the nodes of a cols × rows window of the lattice by the
#      Hungarian algorithm (Hungarian.jl; `board_assignment`), the window's integer offset the hypothesis's (the window holding most
#      of the dots' nearest nodes, `densest_window`, so a stray off the board finds no node to take), the cost the board-plane
#      distance (mm), and a DUSTBIN in the assignment itself — one dummy node per dot at cost pitch / 2 — so no dot is matched
#      farther than half a pitch: a dot in the dustbin is unmatched (an infinite residual, the scorer's uniform part);
#   3. the residuals are the matched dots' board-plane distances, mm, scored by RANSAC's scorer ([index] scoring: MSAC) at the
#      threshold pitch / 2 — the dustbin's cost, the decision known exactly — derived from the record's spacing (`index_scoring`), a
#      dustbin's infinite residual costing the threshold's square: the scorer picks each hypothesis's inliers and score and so the
#      best model;
#   4. RANSAC's local optimization ([index.local_optimization_strategy], Estimation's `Irls` at the Gaussian shape: plain least
#      squares, NOT robust — the matches are within the threshold by construction) refits the pose on the correspondences the
#      scorer produced for the hypothesis — FROZEN: the problem's `refine!` is a board-mm least squares (Core's chart LM) on them, and the residuals of every pose it returns are measured on them
#      (`BoardCorrespondences`), the Hungarian never re-run inside the local optimization.
# Under the final pose the Hungarian runs once more (same window, same dustbin): its matches are the photograph's correspondences —
# the set the one-connected-component rule checks (owner, 2026-11-14: more than one 8-connected component of matched nodes is
# refused, never resolved) and on which the pose is refined in PIXELS through the camera ([index.refine_strategy], Core's
# `LevenbergMarquardt` on Estimation's `AbsolutePoseProblem`, plain least squares: no robust loss anywhere in the index). The MAD scale of the refined pose's pixel
# residuals (Core's `MedianNormScale`) is the photograph's recorded noise scale (its `noise_scale` and `dof`, pooled per set for the
# evaluation's rotation-centre refine, whose Huber width it sets, px; MSAC estimates none). The record's cols × rows grid is the final window: the board
# is matched against an INFINITE lattice under a free pose, so a whole-pitch shift of the window is another pose with the same
# residuals, never a refusal; where the photographs are linked — the evaluate stage's one rotation centre — each one's offset is
# chosen with its symmetry by that centre (`canonical_registrations`). A camera of record whose error at the rim exceeds half a
# pitch in pixels (the fisheye 10 cm's DSC06962, whose nearest-node labelling took its top rows one column over) is still within
# half a pitch on the board, where the residuals are measured.
#
# Outputs per photograph: points/<photo>.json (vgio/observations: each matched dot's record point index and its pixel with its
# Förstner covariance, COLMAP pixels, naming targets.json by its content_id), poses/<photo>.json (vgio/target_poses: the target's
# pose, TargetFrame → CameraFrame, metres), registration/<photo>.tsv (per circular point of the detections: its pixel, the record point
# it was assigned to (0: unassigned) and its reprojection residual, px); targets.json (vgio/rig: the record's target, gauged on its
# own frame — the definition the point indices resolve in) and summary.tsv.
# =============================================================================
using VisualGeometryToolkitCore: points, ustrip, m, mm, model, project, backproject, P3P, rotation,
    translation, correspondences, found_model, successful_retcode, unlabel, with_unit,
    solve, EstimationProblem, RefineProblem, MinimalProblem, LevenbergMarquardt, MedianNormScale, NoiseScale,
    EuclideanMap, TargetFrame, CameraFrame, PixelFrame, PosedTarget, Rig, TargetObservations, TargetPoses, Uncertain, CovMat,
    AbstractTarget, Cameras
import VisualGeometryToolkitCore: data_size, codimension, model_type, test_model, mahalanobis_distances!, refine!, residual_rows!
using VisualGeometryToolkitCore: Triggs, Irls             # the core strategies: Core owns them (a2-reexport)
using VisualGeometryToolkitEstimation: Ransac, RansacConfig, AbstractRansacProblem, AbstractSampler,
    AbsolutePoseProblem, optimization_strategy_from_toml, CountedLO
import VisualGeometryToolkitEstimation: draw!, reset!, sampler, sample_models
using VisualGeometryToolkitEstimation: CameraPosesProblem, RotationCenterViews   # the rotation-centre refine (stage evaluate): bearings, its dots without one counted (its metadata)
using VisualGeometryToolkitCore: FitResult
using Random: AbstractRNG
using VisualGeometryToolkitEval.Stats: bootstrap_ci      # the stack's one bootstrap interval (the paired photograph bootstrap of stage evaluate)
import VisualGeometryToolkitIO as VgtkIO
using StaticArrays: SVector, SMatrix, StaticVector
using LinearAlgebra: norm, dot, I, normalize, det, svdvals
using Hungarian: hungarian             # the scorer's one-to-one assignment of dots to board nodes (the stack has no assignment solver)
using ImageMorphology: centered, label_components   # the one-connected-component rule on the window's matched nodes
using Printf: @printf, @sprintf
import VisualGeometryToolkitEstimation as VgtkEst
using LinearAlgebra: lu, issuccess

"The keys of `[index]`, every one required (no code default)."
const CHROME_INDEX_KEYS = ("target", "camera", "camera_name", "target_id", "label", "scoring", "seed_tolerance", "rng_seed",
                             "min_assigned", "ransac", "local_optimization_strategy", "refine_strategy", "excluded")
# `[index] scoring`, RANSAC's scorer (`index_scoring`): a scorer table of the stack's one robust schema, its `type` required (the
# configs of record state `{ type = "msac" }`; no type is the code's), read by Estimation's `scoring_from_toml`. Its width is NOT the table's: the threshold is half the record's pitch, in mm (the
# board-space residual's unit; the assignment's dustbin), derived from the target record's spacing, so a table stating
# `threshold` (or `outlier_halfwidth`) is refused.
# `[index.refine_strategy]`: the photograph's pixel refine after RANSAC (Core's `LevenbergMarquardt` on Estimation's `AbsolutePoseProblem`), plain least squares:
# `{ type = "none" }` (a refine's least squares, `optimization_strategy = nothing`).
"The keys of `[index.ransac]` (VisualGeometryToolkitEstimation's RansacConfig)."
const RANSAC_KEYS = ("success_prob", "max_trials", "min_trials")
"The index scorers the dustbin rule admits (`[index] scoring` type): MSAC, the threshold a truncation at half the pitch."
const INDEX_SCORING_TYPES = ("msac",)
"The keys of the target record's provenance `design` the grid is read from."
const GRID_DESIGN_KEYS = ("rows", "cols", "spacing_mm")
"The registration table's header: per circular point, its pixel (COLMAP), its record point (0: unassigned) and its residual (px; NaN unassigned)."
const REGISTRATION_COLUMNS = ("x", "y", "point", "residual_px")
"The index summary's header."
const INDEX_SUMMARY_COLUMNS = ("photo", "circular", "inliers", "matched", "unassigned", "trials", "rms_px", "max_px")
"The keys `[index]` no longer reads, refused by name with the reason (the Hungarian index, owner 2026-11-14)."
const RETIRED_INDEX_KEYS = Dict("assignment_level" => "the lattice growth's prediction gate is retired: the Hungarian with its dustbin at half a pitch assigns the dots")

_number(t, k, where; integer = false, positive = true) =
    (v = t[k]; (integer ? v isa Integer : v isa Real) && (!positive || v > 0) ||
        throw(ArgumentError("$where $k must be a positive $(integer ? "integer" : "number")")); v)

"""
    index_settings(cfg, path = "the chrome evaluation config") -> NamedTuple

`[index]` checked: every key of `CHROME_INDEX_KEYS` (and of its `ransac`, `local_optimization_strategy` and `refine_strategy`
tables) required, nothing else; `target` and `camera` inputs of stage index; `excluded` a table of reasons keyed by photographs of
the detect stage. Returns the settings with `scoring` its scorer table (`index_scoring`, the width added per grid), `ransac` its
RansacConfig, `local_optimization_strategy` an Estimation `Irls` strategy at the Gaussian shape (plain least squares) and `refine_strategy` `nothing`, a refine's least squares (both read by
`optimization_strategy_from_toml`, which refuses an unknown type, loss or key) and `photos`, the photographs registered (the detect
stage's less the excluded), sorted.
"""
function index_settings(cfg::AbstractDict, path::AbstractString = "the chrome evaluation config")
    t = get(cfg, "index", nothing)
    where = "$path [index]"
    t isa AbstractDict || throw(ArgumentError("$path: no [index] table"))
    for (k, why) in RETIRED_INDEX_KEYS
        haskey(t, k) && throw(ArgumentError("$where $k is retired: $why"))
    end
    check_required_keys(t, CHROME_INDEX_KEYS, where)
    for k in ("target", "camera")                             # an input, or a [runs] file (the camera: a front end's index fit)
        src = get(cfg["inputs"], t[k], get(get(cfg, "runs", Dict()), t[k], nothing))
        src !== nothing && src["stage"] == "index" ||
            throw(ArgumentError("$where $k: \"$(t[k])\" is not an input or a [runs] entry of stage index"))
    end
    for k in ("camera_name", "target_id", "label")
        t[k] isa AbstractString && !isempty(t[k]) || throw(ArgumentError("$where $k must be a non-empty string"))
    end
    t["label"] in ("circular", "corner") || throw(ArgumentError("$where label: \"$(t["label"])\" is not circular or corner"))
    t["rng_seed"] isa Integer && t["rng_seed"] ≥ 0 || throw(ArgumentError("$where rng_seed must be a non-negative integer"))
    rc = t["ransac"]
    rc isa AbstractDict || throw(ArgumentError("$where ransac must be a table"))
    check_required_keys(rc, RANSAC_KEYS, "$where.ransac")
    ransac = RansacConfig(Float64(_number(rc, "success_prob", "$where.ransac")), Int(_number(rc, "max_trials", "$where.ransac"; integer = true)),
                          Int(_number(rc, "min_trials", "$where.ransac"; integer = true)))
    sc = t["scoring"]
    sc isa AbstractDict || throw(ArgumentError("$where scoring must be a scorer table { type = … } (the stack's SCORING_TAGS)"))
    VgtkCore.required_key(sc, "type", "$where.scoring")      # the scorer is the config's (no code default): Estimation's scoring_from_toml refuses an unknown type
    sc["type"] in INDEX_SCORING_TYPES || throw(ArgumentError("$where scoring type = $(repr(sc["type"])): one of $(join(INDEX_SCORING_TYPES, ", ")) — the threshold is the assignment's dustbin at half the pitch, an MSAC truncation"))
    for k in ("threshold", "outlier_halfwidth")
        haskey(sc, k) && throw(ArgumentError("$where scoring: `$k` is not the table's — the threshold is half the target record's pitch (mm), the assignment's dustbin"))
    end
    scoring = Dict{String, Any}(sc)
    VgtkEst.scoring_from_toml(merge(scoring, Dict{String, Any}("threshold" => 1.0)); problem = GridPoseProblem, section = "$where.scoring")   # checked: the table's keys (the threshold, a grid's, a placeholder here)
    lt = t["local_optimization_strategy"]
    lt isa AbstractDict || throw(ArgumentError("$where local_optimization_strategy must be a strategy table { type = \"irls\", loss = \"gaussian\", max_iter = … }"))
    local_optimization_strategy = optimization_strategy_from_toml(lt; class = :bearing_pose, section = "$where.local_optimization_strategy")
    local_optimization_strategy isa Irls{VgtkCore.GaussianShape} ||
        throw(ArgumentError("$where.local_optimization_strategy: $(get(lt, "type", "")) $(get(lt, "loss", "")) — the grid problem's local optimization is plain least squares on the scorer's matches (all within its threshold), type \"irls\", loss \"gaussian\""))
    rt = t["refine_strategy"]
    rt isa AbstractDict || throw(ArgumentError("$where refine_strategy must be a strategy table { type = \"none\" }"))
    refine_strategy = optimization_strategy_from_toml(rt; class = :absolute_pose, section = "$where.refine_strategy")
    refine_strategy === nothing ||
        throw(ArgumentError("$where.refine_strategy: $(get(rt, "type", "")) — the pixel refine is plain least squares, type \"none\""))
    ex = excluded_table(t["excluded"], where)
    detected = cfg["detect"]["photographs"]
    stray = setdiff(keys(ex), detected)
    isempty(stray) || throw(ArgumentError("$where excluded: $(sort(collect(stray))) are not photographs of the detect stage"))
    return (; target = t["target"], camera = t["camera"], camera_name = String(t["camera_name"]), target_id = Symbol(t["target_id"]),
            label = String(t["label"]), scoring,
            seed_tolerance = Float64(_number(t, "seed_tolerance", where)), rng_seed = Int(t["rng_seed"]),
            min_assigned = Int(_number(t, "min_assigned", where; integer = true)),
            ransac, local_optimization_strategy, refine_strategy,
            excluded = ex, photos = sort(collect(setdiff(detected, keys(ex)))))
end
"""
    index_product(cfg) -> Product

The index product (its output photos.json, written last): keyed by `[index]` with its `excluded` table as the names it excludes
(the reasons are description), its inputs (tier, path) and the detect product's key (`upstream`).
"""
index_product(cfg::AbstractDict) =
    driver_product("index", Dict{String,Any}("experiment" => cfg["experiment"],
                                             "index" => merge(Dict{String,Any}(cfg["index"]), Dict{String,Any}("excluded" => sort(collect(keys(cfg["index"]["excluded"]))))),
                                             "inputs" => inputs_record(cfg, [cfg["index"]["target"], cfg["index"]["camera"]]),
                                             "upstream" => Dict{String,Any}("detect" => detect_product(cfg).key)); output = "photos.json",
                   threads = nothing)                                       # the registrations are the serial ones, one per thread

"""
    ThorlabsGrid

The grid of a target record: its plain target (metres), the node (column, row), 0-based, of every record point, the inverse map
node → point index, its columns, rows and spacing (m).
"""
struct ThorlabsGrid
    target::AbstractTarget                       # plain, metres
    nodes::Vector{SVector{2,Int}}
    index::Dict{SVector{2,Int},Int}
    cols::Int
    rows::Int
    spacing::Float64                             # m
end

"""
    thorlabs_grid(target, design) -> ThorlabsGrid

The grid of the target record `target` (lengths) whose provenance `design` (a dictionary with `GRID_DESIGN_KEYS`) states its rows,
columns and spacing: every record point must sit on a node (within 1e-9 of a spacing) and the nodes must be exactly the
cols × rows lattice from the origin, point (r − 1)·cols + c at node (c − 1, r − 1) (an `ArgumentError` otherwise).
"""
function thorlabs_grid(target::AbstractTarget, design::AbstractDict)
    foreach(k -> VgtkCore.required_key(design, k, "the target record's provenance design"), GRID_DESIGN_KEYS)
    cols, rows = Int(design["cols"]), Int(design["rows"])
    plain = ustrip(m, target)
    spacing = Float64(ustrip(m, Float64(design["spacing_mm"]) * mm))
    P = points(plain)
    length(P) == rows * cols || throw(ArgumentError("the target record has $(length(P)) points, its design $rows × $cols"))
    nodes = map(P) do p
        l = SVector(p[1], p[2]) / spacing
        n = round.(Int, l)
        norm(l - n) < 1e-9 || throw(ArgumentError("record point $p is not on the grid of spacing $(design["spacing_mm"]) mm"))
        n
    end
    nodes == [SVector(c - 1, r - 1) for r in 1:rows for c in 1:cols] ||
        throw(ArgumentError("the record's points are not the $cols × $rows lattice row by row from the origin"))
    return ThorlabsGrid(plain, nodes, Dict(n => i for (i, n) in enumerate(nodes)), cols, rows, spacing)
end

"The grid of the target record at `tpath` (`thorlabs_grid` of IO's `load(AbstractTarget, tpath)` and the record's provenance design)."
grid_of_record(tpath::AbstractString) = thorlabs_grid(VgtkIO.load(AbstractTarget, tpath), VgtkIO.read_record(tpath)["provenance"]["design"])

"""
    wound_neighbours(xy, i) -> NTuple{8,Int}

The 8 points of `xy` nearest to point `i`, ordered by their direction around it (angle, COLMAP px): a dot inside the grid has its
3 × 3 neighbourhood there, the four edge neighbours and the four diagonal ones alternating in that order.
"""
function wound_neighbours(xy::AbstractVector{<:SVector{2}}, i::Integer)
    length(xy) > 8 || throw(ArgumentError("wound_neighbours: $(length(xy)) points, a seed needs 8 neighbours"))
    x0 = xy[i]
    nb = sortperm([norm(p - x0) for p in xy])[2:9]                                    # [1] is i itself
    return Tuple(nb[sortperm([atan(xy[j][2] - x0[2], xy[j][1] - x0[1]) for j in nb])])
end

"""
    GridSeedSampler(xy)

The grid's minimal sample: a random point, the seed, and two of its wound neighbours a quarter turn apart, `nb[k]` and `nb[k + 2]`
with `k` ∈ {1, 2} (`wound_neighbours`, tabulated once). Inside the grid one parity is the edge neighbours, so the sample is the
lattice's (0, 0), (1, 0), (0, 1); the other is diagonals and fails. Which edge neighbour is (1, 0) and which way the quarter turn
goes are not drawn: on an infinite lattice of identical dots every rotation and mirror registers the same dots.
"""
struct GridSeedSampler <: AbstractSampler
    nb::Vector{NTuple{8,Int}}
end
GridSeedSampler(xy::AbstractVector{<:SVector{2}}) = GridSeedSampler([wound_neighbours(xy, i) for i in eachindex(xy)])
function draw!(rng::AbstractRNG, idx::AbstractVector{Int}, s::GridSeedSampler)
    i, k = rand(rng, eachindex(s.nb)), rand(rng, 1:2)
    idx[1], idx[2], idx[3] = i, s.nb[i][k], s.nb[i][k + 2]
    return nothing
end
reset!(::GridSeedSampler) = nothing

"""
    BoardCorrespondences{M}

The correspondences a `GridPoseProblem`'s scorer produced (`board_assignment`, the Hungarian with its dustbin) for the last
hypothesis it scored, `anchor`, and the poses the local optimization refitted from them (`frozen`): `pairs` dot index => its
node, the matched dots only; `fitted`, every pose of the run the local optimization started from or refitted, with the correspondences it was fitted on (`register_grid`
compares its final matches with them). RANSAC's local optimization refits on these, and the residuals of every pose in `frozen` are measured
on them, so the Hungarian runs only in the scorer of a fresh hypothesis (owner, 2026-11-14). One per problem: a problem is scored by
one RANSAC at a time.
"""
mutable struct BoardCorrespondences{M}
    anchor::Union{Nothing, M}
    pairs::Vector{Pair{Int, SVector{2, Int}}}
    frozen::Vector{M}
    fitted::Vector{Pair{M, Vector{Pair{Int, SVector{2, Int}}}}}
end
BoardCorrespondences{M}() where {M} =
    BoardCorrespondences{M}(nothing, Pair{Int, SVector{2, Int}}[], M[], Pair{M, Vector{Pair{Int, SVector{2, Int}}}}[])

"""
    GridPoseProblem(camera, xy, grid; seed_tolerance)

The pose [r₁ r₂ t] of the planar lattice of dots of `grid` (a `ThorlabsGrid`: pitch, columns, rows) seen by `camera` in the
points `xy` (COLMAP px), as a RANSAC problem with no correspondences given: the sample (`GridSeedSampler`) is the lattice's (0, 0),
(1, 0), (0, 1), solved by P3P on the camera's bearings; a pose is kept when it predicts the seed's opposite neighbour, node (−1, 0),
within `seed_tolerance` px. A pose's residuals (`mahalanobis_distances!`) are board-plane distances, mm: each point's ray met with
the board (`board_point`) and matched one-to-one to a node of the board window by the Hungarian algorithm with a dustbin at half a
pitch (`board_assignment`), a point in the dustbin at an infinite residual. Its weighted refit (`refine!`, the step of the stack's
local optimization) refines the pose on the scorer's correspondences, frozen (`BoardCorrespondences`), by a weighted least squares
of their board-plane residuals (Core's chart LM, `board_refit`).
"""
struct GridPoseProblem{C} <: AbstractRansacProblem{3}
    camera::C
    xy::Vector{SVector{2,Float64}}
    bearings::Vector{SVector{3,Float64}}
    spacing::Float64                     # m
    cols::Int                            # the board window
    rows::Int
    seed_tolerance::Float64
    sampler::GridSeedSampler
    correspondences::BoardCorrespondences{model_type(P3P())}
end
GridPoseProblem(camera, xy::AbstractVector{<:SVector{2}}, g::ThorlabsGrid; seed_tolerance::Real) =
    GridPoseProblem(camera, collect(xy), [SVector{3,Float64}(normalize(backproject(camera, u))) for u in xy], g.spacing, g.cols, g.rows,
                    Float64(seed_tolerance), GridSeedSampler(xy), BoardCorrespondences{model_type(P3P())}())
data_size(p::GridPoseProblem) = length(p.xy)
codimension(::Type{<:GridPoseProblem}) = 2
model_type(::Type{<:GridPoseProblem}) = model_type(P3P())   # P3P's pose type (Core's `CameraExtrinsics`)
sampler(p::GridPoseProblem) = p.sampler
"The record point of node `n` as a plain 3-vector on the target plane (z = 0), metres, of a lattice of pitch `x.spacing` (a
`ThorlabsGrid`, a `GridPoseProblem`)."
_node_point(x::Union{ThorlabsGrid, GridPoseProblem}, n) = SVector(n[1] * x.spacing, n[2] * x.spacing, 0.0)
"Millimetres per metre: the board-space residual's unit (mm) from the problem's (m)."
const _MM_PER_M = Float64(ustrip(mm, 1.0 * m))
"The dustbin's cost and the scorer's threshold of the problem `p`: half its pitch, mm."
dustbin_mm(p::GridPoseProblem) = p.spacing * _MM_PER_M / 2

const _SAMPLE_NODES = (SVector(0, 0), SVector(1, 0), SVector(0, 1))
sample_models(p::GridPoseProblem, idx::AbstractVector{Int}) =
    solve(MinimalProblem(model_type(P3P()), correspondences([_node_point(p, n) for n in _SAMPLE_NODES], p.bearings[idx]), 1:3), P3P())

function test_model(p::GridPoseProblem, pose, idx::AbstractVector{Int})
    nb = p.sampler.nb[idx[1]]
    k = findfirst(==(idx[2]), nb)
    Xc = pose(_node_point(p, SVector(-1, 0)))
    return Xc[3] > 0 && norm(project(p.camera, Xc) - p.xy[nb[mod1(k + 4, 8)]]) < p.seed_tolerance
end

"The target plane z = 0 (target frame), the dots' lattice plane."
const _TARGET_PLANE = VgtkCore.Plane3D(0.0, 0.0, 1.0, 0.0)

"""
    board_point(R, t, f) -> SVector{2} or nothing
    board_point(p, pose, i) -> SVector{2,Float64} or nothing

The bearing `f` (camera frame) of a camera at pose (R, t) (target → camera) met with the target plane z = 0 (Core's
`ray_plane_intersection`, the plane a constant: generic in the ray's number type, a chart LM's dual numbers go through): the hit's target
coordinates (x, y), metres, NOT rounded to a node; `nothing` when the ray does not meet the plane in front of the camera. With a
problem: point `i`'s bearing under `pose`.
"""
function board_point(R, t, f)
    C, d = -(R' * t), R' * f                                                           # the camera centre and the ray, target frame
    x, y, _, λ = VgtkCore.ray_plane_intersection(_TARGET_PLANE, C, d[1], d[2], d[3])
    isfinite(λ) && λ > 0 || return nothing                                            # parallel, or the plane behind the camera
    return SVector(x, y)
end
board_point(p::GridPoseProblem, pose, i::Integer) = board_point(rotation(pose), translation(pose), p.bearings[i])

"""
    board_residual(p, R, t, i, node) -> SVector{2} or nothing

Point `i`'s board-plane residual from lattice node `node` under the pose (R, t): its board point (`board_point`) less the node's
(`_node_point`), mm — the scorer's cost, the frozen distances and the local optimization's rows; `nothing` where the ray misses.
"""
function board_residual(p::GridPoseProblem, R, t, i::Integer, node)
    q = board_point(R, t, p.bearings[i])
    return q === nothing ? nothing : board_residual(p, q, node)
end
board_residual(p::GridPoseProblem, q::SVector{2}, node) = (q - _node_point(p, node)[SVector(1, 2)]) * _MM_PER_M

"""
    board_assignment(p, pose) -> (; window, pairs, distance)

The scorer's correspondences under `pose`: every point's board point (`board_point`, pitches) matched ONE-TO-ONE to the nodes of
the cols × rows board window by the Hungarian algorithm (Hungarian.jl's `hungarian`), the cost the board-plane distance (mm). The
window's origin `window` is the hypothesis's: the window holding most of the points' nearest nodes (`densest_window`), so a
detection off the board has no node of its own to take. The DUSTBIN is part of the assignment: one dummy node per point at cost
half a pitch (`dustbin_mm`), so a match costing more loses to it and no point is matched farther than half a pitch. `pairs`: point
index => its node, the matched points in index order; `distance`: per point its board-plane distance from its node, mm, `Inf` for
a point in the dustbin or whose ray misses the board (the scorer's uniform part). `window` is `nothing` when no ray meets the board.
"""
function board_assignment(p::GridPoseProblem, pose)
    n = length(p.xy)
    b = [board_point(p, pose, i) for i in 1:n]
    # a ray meeting the plane further than an Int can round (a grazing hypothesis: λ ~ 1e17) is a miss, as one parallel to it
    hit = [i for i in 1:n if b[i] !== nothing && all(x -> abs(x) < maxintfloat(Float64), b[i] / p.spacing)]
    pairs = Pair{Int, SVector{2, Int}}[]; distance = fill(Inf, n)
    isempty(hit) && return (; window = nothing, pairs, distance)
    near = [round.(Int, b[i] / p.spacing) for i in hit]                               # each point's nearest node
    o = densest_window(unique(near), p.cols, p.rows)
    h = dustbin_mm(p)
    # the assignment's candidates: a point whose nearest window node is nearer than the dustbin, and those nodes (a match costing
    # more than h always loses to the dustbin, so the rest are the dustbin's without entering the matrix: the same assignment)
    inwin(q) = all(o .<= q .<= o .+ SVector(p.cols - 1, p.rows - 1))
    d(k, q) = norm(board_residual(p, b[hit[k]], q))
    rows = [k for k in eachindex(hit) if inwin(near[k]) && d(k, near[k]) < h]
    isempty(rows) && return (; window = o, pairs, distance)
    nodes = unique(near[rows])
    cost = [d(k, q) for k in rows, q in nodes]
    a, _ = hungarian(hcat(cost, fill(h, length(rows), length(rows))))                 # the candidate nodes, then one dustbin per point
    for (j, k) in enumerate(rows)
        a[j] in eachindex(nodes) || continue                                          # the dustbin
        push!(pairs, hit[k] => nodes[a[j]]); distance[hit[k]] = cost[j, a[j]]
    end
    sort!(pairs; by = first)
    return (; window = o, pairs, distance)
end

"""
    frozen_distances!(r, p, pose, pairs) -> r

The board-plane distance (mm) of each point of `pairs` (point => node) from its node under `pose`; `Inf` for every other point and
for a ray that misses the board.
"""
function frozen_distances!(r::AbstractVector, p::GridPoseProblem, pose, pairs)
    fill!(r, Inf)
    R, t = rotation(pose), translation(pose)
    for (i, nd) in pairs
        e = board_residual(p, R, t, i, nd)
        e === nothing || (r[i] = norm(e))
    end
    return r
end

"""
    _same_pose(a, b) -> Bool

Whether two poses are the same map, whatever their types and frame labels (the local optimization hands the problem its model as
another `EuclideanMap` type than the one the scorer saw): the rotations' matrices and the translations equal to `_SAME_POSE_RTOL`,
so a re-representation of one pose (a quaternion or another matrix type, a re-orthonormalisation: a few ulps) is the same pose. Two
hypotheses of RANSAC differ far beyond it (a minimal sample's pose, not a rounding of another's).
"""
_same_pose(a, b) = isapprox(SMatrix{3, 3, Float64}(rotation(a)), SMatrix{3, 3, Float64}(rotation(b)); rtol = _SAME_POSE_RTOL, atol = _SAME_POSE_RTOL) &&
                   isapprox(SVector{3, Float64}(translation(a)), SVector{3, Float64}(translation(b)); rtol = _SAME_POSE_RTOL)
"The relative tolerance of `_same_pose`: 1e-12, about 4500 ulps of Float64 (2.2e-16) — any re-representation's rounding, far below
any two hypotheses' difference."
const _SAME_POSE_RTOL = 1e-12

# The RANSAC protocol's per-point DISTANCES (≥ 0, mm; Core's `mahalanobis_distances!`): a fresh hypothesis is scored by the
# Hungarian with its dustbin (`board_assignment`), its correspondences kept as the anchor's; a pose the local optimization refitted
# from them (`frozen`) is measured on those correspondences, never re-assigned.
function mahalanobis_distances!(r::AbstractVector, p::GridPoseProblem, pose)
    c = p.correspondences
    any(q -> _same_pose(q, pose), c.frozen) && return frozen_distances!(r, p, pose, c.pairs)
    a = board_assignment(p, pose)
    c.anchor, c.pairs = pose, a.pairs; empty!(c.frozen)
    return copyto!(r, a.distance)
end

"""
    refine!(ws, p::GridPoseProblem, pose, w) -> pose or nothing

The grid problem's weighted refit, the model step of the stack's local optimization (Estimation's `posterior_irls!` under an `Irls`
strategy, `WeightedRefine`): the pose refined from `pose` on the correspondences the scorer produced for the hypothesis the local
optimization started from (`BoardCorrespondences`: `pose` is that hypothesis or a pose refitted from it; the Hungarian is not run
again), the points of positive weight `w`, by the weighted least squares of their board-plane residuals (`board_refit`). The
refitted pose joins the frozen ones, so its residuals are measured on the same correspondences. `nothing` when fewer than four
matched points carry weight or the LM goes non-finite. A pose the scorer did not score (neither its last hypothesis nor a refit of it)
is scored on demand — assigned by the Hungarian, made the anchor — never an error: the protocol's state is the problem's own.
"""
function refine!(_ws, p::GridPoseProblem, pose, w::AbstractVector)
    c = p.correspondences
    at_anchor = c.anchor !== nothing && _same_pose(pose, c.anchor)
    frozen = any(q -> _same_pose(q, pose), c.frozen)
    if !(at_anchor || frozen)     # a pose not scored: assigned now (the Hungarian under it), its correspondences the anchor's
        c.anchor, c.pairs = pose, board_assignment(p, pose).pairs; empty!(c.frozen)
    end
    frozen || (push!(c.frozen, c.anchor); push!(c.fitted, c.anchor => c.pairs))
    pairs = [i => nd for (i, nd) in c.pairs if w[i] > 0]
    length(pairs) >= 4 || return nothing
    q = board_refit(p, pose, pairs, [Float64(w[i]) for (i, _) in pairs])
    q === nothing || (push!(c.frozen, q); push!(c.fitted, q => c.pairs))
    return q
end

"""
    BoardRows(ch, problem, pairs, sqrtw)

The linearization state of `board_refit`'s chart LM (Core's `lmsolve_chart!` protocol): the chart `ch` at the pose, the points and
nodes `pairs`, each point's √weight. Its rows (`residual_rows!`, Core's ForwardDiff Jacobian) are each point's board-plane residual
(board point − node, mm) under the pose `retract(ch, δ)`, × √w; a ray that misses the board at Core's `OUT_OF_FIELD_RESIDUAL`.
"""
struct BoardRows{C, P}
    ch::C
    problem::P
    pairs::Vector{Pair{Int, SVector{2, Int}}}
    sqrtw::Vector{Float64}
end
function residual_rows!(r, lm::BoardRows, δ)
    p = lm.problem
    pose = VgtkCore.retract(lm.ch, δ)
    R, t = rotation(pose), translation(pose)
    for (k, (i, nd)) in enumerate(lm.pairs)
        e = something(board_residual(p, R, t, i, nd), SVector(VgtkCore.OUT_OF_FIELD_RESIDUAL, VgtkCore.OUT_OF_FIELD_RESIDUAL))
        r[2k - 1] = lm.sqrtw[k] * e[1]; r[2k] = lm.sqrtw[k] * e[2]
    end
    return r
end

"""
    board_refit(p, pose, pairs, w) -> pose or nothing

The pose refined from `pose` on the correspondences `pairs` (point => node) at the weights `w`: Core's chart LM (`lmsolve_chart!`,
its stop settings `LmStops()`) on the pose's chart (`minimal_manifold`) over the weighted board-plane residuals (`BoardRows`, mm).
`nothing` when the LM goes non-finite.
"""
function board_refit(p::GridPoseProblem, pose, pairs, w::AbstractVector)
    ch = VgtkCore.minimal_manifold(pose)
    ews = VgtkCore.ManifoldLmWorkspace(2 * length(pairs), VgtkCore.tangent_size(ch))
    x = first(VgtkCore.lmsolve_chart!(ews, BoardRows(ch, p, collect(pairs), sqrt.(w))))
    any(!isfinite, x) && return nothing
    return convert(typeof(pose), VgtkCore.retract(ch, x))
end

"""
    index_scoring(p, table) -> AbstractScoring

RANSAC's scorer of the grid problem `p` from `[index] scoring` (`index_settings`): Estimation's `scoring_from_toml` of the table
for the `GridPoseProblem` (codimension 2, mm) at the threshold half the pitch (`dustbin_mm`, the dustbin's cost): the width is the
record's, never the table's.
"""
index_scoring(p::GridPoseProblem, table::AbstractDict) =
    VgtkEst.scoring_from_toml(merge(Dict{String, Any}(table), Dict{String, Any}("threshold" => Float64(dustbin_mm(p))));
                              problem = GridPoseProblem, section = "index.scoring")

"""
    densest_window(nodes, cols, rows) -> SVector{2,Int}

The origin `o` of the `cols × rows` window of lattice nodes (`o .≤ n .≤ o .+ (cols - 1, rows - 1)`) holding most of `nodes`, among
the windows that hold at least one: the candidates are every `n - (c, r)` in the order of `nodes`, then `c`, then `r`, and the first
of the most held wins. A window's count is read off the 2-D prefix sums of the nodes over their distinct columns and rows
(coordinate-compressed: at most N × N cells for N nodes however far apart they lie — a hypothesis's stray rays meet the plane far
off the board), two binary searches per candidate; it runs on every RANSAC hypothesis.
"""
function densest_window(nodes, cols::Integer, rows::Integer)
    ns = collect(SVector{2,Int}, nodes)
    isempty(ns) && throw(ArgumentError("densest_window: no node"))
    xs = sort!(unique(n[1] for n in ns)); ys = sort!(unique(n[2] for n in ns))
    S = zeros(Int, length(xs) + 1, length(ys) + 1)                                      # S[i + 1, j + 1]: the nodes at columns ≤ xs[i], rows ≤ ys[j]
    for n in ns
        S[searchsortedfirst(xs, n[1]) + 1, searchsortedfirst(ys, n[2]) + 1] += 1
    end
    cumsum!(S, S; dims = 1); cumsum!(S, S; dims = 2)
    function held(o)
        i0, i1 = searchsortedfirst(xs, o[1]), searchsortedlast(xs, o[1] + cols - 1)
        j0, j1 = searchsortedfirst(ys, o[2]), searchsortedlast(ys, o[2] + rows - 1)
        (i1 < i0 || j1 < j0) && return 0
        return S[i1 + 1, j1 + 1] - S[i0, j1 + 1] - S[i1 + 1, j0] + S[i0, j0]
    end
    best, o = -1, ns[1]
    for n in ns, c in 0:(cols - 1), r in 0:(rows - 1)
        k = held(n - SVector(c, r))
        k > best && ((best, o) = (k, n - SVector(c, r)))
    end
    return o
end

"""
    LatticeComponents(msg, sizes)

`register_grid`'s refusal of a photograph whose matched nodes form more than one 8-connected component (`sizes`: each one's dots,
largest first) — a detection or assignment defect to investigate (owner, 2026-11-14), never resolved.
"""
struct LatticeComponents <: Exception
    msg::String
    sizes::Vector{Int}
end
Base.showerror(io::IO, e::LatticeComponents) = print(io, e.msg)

"""
    component_sizes(nodes, o, cols, rows) -> Vector{Int}

The sizes, largest first, of the 8-connected components of the nodes `nodes` of the cols × rows window at origin `o`
(ImageMorphology's `label_components` on the window's occupancy).
"""
function component_sizes(nodes, o::SVector{2, Int}, cols::Integer, rows::Integer)
    A = falses(cols, rows)
    for n in nodes
        A[n[1] - o[1] + 1, n[2] - o[2] + 1] = true
    end
    L = label_components(A, centered(trues(3, 3)))   # 8-connectivity: a centered 3 × 3 mask, as ImageMorphology documents
    return sort([count(==(k), L) for k in 1:maximum(L; init = 0)]; rev = true)
end

"""
    MislabelledDots(msg, dots)

`register_grid`'s refusal of a photograph with a dot labelled with a node whose image is not the nearest node image to it
(`dots`: the indices into `xy`; `mislabelled_dots`): a labelling the photograph's own final pose contradicts, refused, never
resolved (owner, 2026-11-14: precision over recall).
"""
struct MislabelledDots <: Exception
    msg::String
    dots::Vector{Int}
end
Base.showerror(io::IO, e::MislabelledDots) = print(io, e.msg)

"The node offsets a dot's own node is compared against (`mislabelled_dots`): its 5 × 5 neighbourhood, itself excluded."
const _MISLABEL_NEIGHBOURS = [SVector(a, b) for a in -2:2 for b in -2:2 if (a, b) != (0, 0)]

"""
    mislabelled_dots(camera, pose, p, final) -> Vector{Int}

The dots of the labelling `final` (index into `p.xy` => lattice node) whose own node's image under `pose` through `camera` is not
the nearest to them (px): some node of its 5 × 5 neighbourhood images nearer the dot than its own (Core's `reprojection_error`; a
node that does not image — behind the camera, off the domain — is never nearer). Sorted; empty for a labelling the pose bears out.
"""
function mislabelled_dots(camera, pose, p::GridPoseProblem, final::AbstractDict)
    dist(n, x) = (e = VgtkCore.reprojection_error(camera, pose, _node_point(p, n), x); all(isfinite, e) ? norm(e) : Inf)
    bad = Int[]
    for (i, n) in final
        own = dist(n, p.xy[i])
        any(d -> dist(n + d, p.xy[i]) < own, _MISLABEL_NEIGHBOURS) && push!(bad, i)
    end
    return sort!(bad)
end

"""
    register_grid(camera, g, xy, s; name = "the photograph") -> NamedTuple

Register the points `xy` (COLMAP px) against the grid `g` (`thorlabs_grid`) through `camera` (a `CameraModel`) with the settings
`s` (`index_settings`): VisualGeometryToolkitEstimation's `Ransac` on the `GridPoseProblem` (scored in board space by `index_scoring`
of `s.scoring`, MSAC at half a pitch, the Hungarian with its dustbin inside the scorer; the local optimization of
`s.local_optimization_strategy` on the scorer's correspondences, frozen; `s.ransac`'s trial schedule, a StableRNG of `s.rng_seed`).
Under RANSAC's pose the Hungarian runs once more (`board_assignment`) and the pose is refined on its matches in pixels through the
camera (`pixel_refit`: Core's `LevenbergMarquardt` on Estimation's `AbsolutePoseProblem`, `s.refine_strategy`, plain least squares), the Hungarian re-run under the
refined pose and the pose refined again until its matches repeat (a cycle refused): the final matches are the photograph's
correspondences, which must form one 8-connected component of nodes (`component_sizes`). The record's grid is the final window
(any whole-pitch offset is a pose of its own, chosen across the photographs by the evaluate stage's rotation centre), the pose
moved to its origin. Returns `(; assignment, pose, residuals, trials, inliers, noise_scale, dof, reassigned, rounds)`: `assignment` record point index → index into `xy`, `pose` the target's pose (target plane, metres →
camera), `residuals` per record point assigned its reprojection error (px) through the pose, `trials` RANSAC's, `inliers` its
scorer's inlier count, `noise_scale` and `dof` the MAD scale (px, Core's `MedianNormScale`) of the refined pose's pixel residuals and their
degrees of freedom (2k − 6, k dots), `reassigned` the dots whose final match differs
from the correspondences RANSAC's local optimization fitted (added, dropped or moved; `nothing` when no local optimization fitted
RANSAC's pose), `rounds` the pixel refines run. Refuses (`ArgumentError`
naming `name`) a photograph where RANSAC finds no pose, whose rays miss the board under it, that matches fewer than `min_assigned`
points, (`LatticeComponents`) one whose matched nodes form more than one connected component, and (`MislabelledDots`) one with a
dot nearer another node's image than its own under the final pose (`mislabelled_dots`).
"""
function register_grid(camera, g::ThorlabsGrid, xy::AbstractVector{<:SVector{2}}, s; name::AbstractString = "the photograph")
    p = GridPoseProblem(camera, xy, g; s.seed_tolerance)
    r = solve(EstimationProblem(p), Ransac(; scoring = index_scoring(p, s.scoring), local_optimization_strategy = s.local_optimization_strategy,
                                           config = s.ransac); rng = StableRNG(s.rng_seed))
    found_model(r) || throw(ArgumentError("$name: RANSAC found no pose of the grid"))
    pose = r.u
    k = findlast(f -> _same_pose(f.first, pose), p.correspondences.fitted)                      # the LO's correspondences, when it fitted the pose
    fitted = k === nothing ? nothing : Dict(p.correspondences.fitted[k].second)
    # the final correspondences: the Hungarian once more under RANSAC's (locally optimized) pose — same window, same dustbin — and
    # the pose refined on them in PIXELS through the camera (plain least squares); while the Hungarian under the refined pose
    # matches differently, the pose is refined again on its matches (DSC06962: under RANSAC's board-space pose five rim dots of the
    # top row are in the dustbin and a highlight takes a node; under the pixel refine every dot is within 1.7 mm of its node)
    a, rounds, seen = _final_assignment(p, pose, name), 1, Set{Dict{Int, SVector{2, Int}}}()
    while true
        final = Dict(a.pairs)
        length(final) ≥ s.min_assigned ||
            throw(ArgumentError("$name: $(length(final)) points assigned, fewer than min_assigned = $(s.min_assigned)"))
        push!(seen, final)
        pose = pixel_refit(p, pose, final, s.refine_strategy; name)
        a = _final_assignment(p, pose, name)
        Dict(a.pairs) == final && break                                                 # a fixed assignment
        Dict(a.pairs) in seen && throw(ArgumentError("$name: the final assignment cycles between the pixel refine's poses ($rounds rounds)"))
        rounds += 1
    end
    final = Dict(a.pairs)
    reassigned = fitted === nothing ? nothing : count(i -> get(final, i, nothing) != get(fitted, i, nothing), union(keys(final), keys(fitted)))
    # ONE connected component (owner, 2026-11-14): more than one means a detection or assignment defect, refused, never resolved
    sizes = component_sizes(values(final), a.window, g.cols, g.rows)
    length(sizes) > 1 && throw(LatticeComponents("$name: the labelled dots form $(length(sizes)) connected components ($(join(sizes, ", ")) dots), not one", sizes))
    # no mislabel (owner, 2026-11-14): every labelled dot nearest its own node's image under the final pose, or the photograph is refused
    bad = mislabelled_dots(camera, pose, p, final)
    isempty(bad) || throw(MislabelledDots("$name: $(length(bad)) labelled dots are nearer another node's image than their own (dots $(join(bad, ", ")))", bad))
    ids = sort(collect(keys(final)))
    e = [SVector{2, Float64}(VgtkCore.reprojection_error(camera, pose, _node_point(p, final[i]), p.xy[i])) for i in ids]   # the pixel residuals
    σ = Float64(solve(EstimationProblem(NoiseScale, e), MedianNormScale()).u)   # the photograph's noise scale (px), pooled per set by the evaluate stage
    o = a.window
    pose = pose ∘ EuclideanMap(SMatrix{3,3,Float64}(I), _node_point(p, o))              # the record's origin is node o
    assignment = Dict(g.index[final[i] - o] => i for i in ids)
    residuals = Dict(g.index[final[i] - o] => norm(ek) for (i, ek) in zip(ids, e))
    return (; assignment, pose, residuals, trials = r.stats.nsteps, inliers = count(VgtkCore.inlier_mask(r)),
            noise_scale = σ, dof = 2 * length(ids) - 6, reassigned, rounds)
end

"The Hungarian's correspondences under the pose (`board_assignment`), refused (naming `name`) when no ray meets the board."
function _final_assignment(p::GridPoseProblem, pose, name)
    a = board_assignment(p, pose)
    a.window === nothing && throw(ArgumentError("$name: no ray meets the board under the pose"))
    return a
end

"""
    pixel_refit(p, pose, pairs, strategy; name = "the photograph") -> pose

The pose refined from `pose` on the correspondences `pairs` (point => node) in PIXELS through the problem's camera: Core's
`LevenbergMarquardt` on Estimation's `AbsolutePoseProblem` (the raw-pixel reprojection error) at `strategy` (`[index] refine_strategy`: plain least squares); an
`ArgumentError` naming `name` (the photograph refused) when no pose comes back.
"""
function pixel_refit(p::GridPoseProblem, pose, pairs::AbstractDict, strategy; name::AbstractString = "the photograph")
    obs = [_node_point(p, nd) => p.xy[i] for (i, nd) in sort(collect(pairs); by = first)]
    est = solve(RefineProblem(AbsolutePoseProblem(p.camera, obs), pose), LevenbergMarquardt(; optimization_strategy = strategy)).u
    est === nothing && throw(ArgumentError("$name: the pose does not refine in pixels on its $(length(obs)) matches"))
    return convert(typeof(pose), est)
end

"""
    read_detections(path) -> Vector{Attributed}
    read_detections(path, label) -> Vector{Uncertain}

The detections of the file `path` ([`write_detections`](@ref), IO's `vgio/detection`: one photograph) in the file's order: each an
`Attributed` uncertain `PointFrame` (COLMAP pixels, its 2 × 2 covariance in px², NaN where the detector stated none) with its
`label`, `window`, `F` and `covariance_status`. With `label`: those labelled `label`, each Core's `Uncertain` of its pixel
(`SVector`) and covariance.
"""
read_detections(path::AbstractString) =
    only(values(VgtkIO.load(Dict{String, Vector{VgtkCore.AbstractDetection}}, path)))
function read_detections(path::AbstractString, label::AbstractString)
    out = Uncertain[]
    for d in read_detections(path)
        String(d.attributes.label) == label || continue
        u = d.value
        push!(out, Uncertain(SVector{2, Float64}(Tuple(VgtkCore.centroid(VgtkCore.geom(u)))), cov(u)))
    end
    return [x for x in out]
end

"""
    register_photo(camera, grid, detect_dir, s, photo) -> (; u, reg, err, seconds)

The index stage's one photograph: the detections of `photo` labelled `s.label` in the detect run `detect_dir` (`read_detections`,
Core's `Uncertain` pixels `u`) registered on `grid` through `camera` (`register_grid` with the settings `s`). A photograph that does
not register is recorded, not fatal: `reg` is `nothing` and `err` the exception — the refusals (`ArgumentError`, `LatticeComponents`,
`MislabelledDots`) and any other failure of its registration alike, so one photograph never ends the index product (`nothing` otherwise). `seconds`: the
registration's wall time (the detections' read excluded).
"""
function register_photo(camera, grid::ThorlabsGrid, detect_dir::AbstractString, s, photo::AbstractString)
    u = read_detections(joinpath(detect_dir, "detections", splitext(photo)[1] * ".json"), s.label)
    t = time()
    reg, err = try
        register_grid(camera, grid, SVector{2, Float64}[VgtkCore.geom(x) for x in u], s; name = photo), nothing   # typed: none detected is a refusal, not a MethodError
    catch e
        e isa InterruptException && rethrow()
        e isa Union{ArgumentError, LatticeComponents, MislabelledDots} || @warn "$photo: the registration failed; the photograph is refused" exception = (e, catch_backtrace())
        nothing, e
    end
    return (; u, reg, err, seconds = time() - t)
end

"The target rig the point indices resolve in: the record's target, gauged on its own frame (identity pose, metres)."
target_rig(target::AbstractTarget, id::Symbol) =
    Rig{TargetFrame{id}}(Dict(id => PosedTarget(target, with_unit(EuclideanMap(SMatrix{3,3,Float64}(I), zero(SVector{3,Float64})), m))))

"""
    write_registration(dir, stem, photo, reg, u, s, rig_id) -> nothing

The outputs of one photograph's registration `reg` (`register_grid`) of the detections `u` (`read_detections`: `Uncertain` pixels):
points/<stem>.json (the assigned detections, in the detections' order, each with its record point), poses/<stem>.json (IO's
`TargetPoses`, the evaluate stage's start), registration/<stem>_scale.json (the MAD scale of its final pixel residuals and their degrees of freedom)
and registration/<stem>.tsv (VisualGeometryToolkitEval's table, `REGISTRATION_COLUMNS`).
"""
function write_registration(dir, stem, photo, reg, u::AbstractVector{<:Uncertain}, s, rig_id::AbstractString)
    ks = sort(collect(keys(reg.assignment)); by = k -> reg.assignment[k])     # the detections' order: the evaluate stage's sums
    obs = correspondences(ks, [u[reg.assignment[k]] for k in ks])               # record point index => its detection
    tp = TargetObservations{PixelFrame{:colmap}}(Dict(1 => Dict(s.target_id => obs)); images = Dict(1 => photo), sensors = s.camera_name,
                                                 targets = [rig_id])
    foreach(d -> mkpath(joinpath(dir, d)), ("points", "poses", "registration"))
    VgtkIO.save(joinpath(dir, "points", stem * ".json"), tp)
    poses = TargetPoses{CameraFrame}(Dict(1 => Dict(s.target_id => with_unit(unlabel(reg.pose), m)));
                                     images = Dict(1 => photo), sensors = s.camera_name)
    VgtkIO.save(joinpath(dir, "poses", stem * ".json"), poses)
    VgtkIO.write_record(joinpath(dir, "registration", stem * "_scale.json"),                # the photograph's MAD scale of its index pixel residuals (px) and dof
        (; schema = EXPERIMENT_RECORD_SCHEMAS.registration_scale, photo, noise_scale = (σ = get(reg, :noise_scale, NaN); isfinite(σ) ? σ : nothing), dof = get(reg, :dof, 0)))   # null: none (an index run before the MAD scale)
    point_of = Dict(i => k for (k, i) in reg.assignment)
    write_tsv(joinpath(dir, "registration", stem * ".tsv"), REGISTRATION_COLUMNS,          # VisualGeometryToolkitEval's table
              ((Float64(p[1]), Float64(p[2]), k, k == 0 ? NaN : Float64(reg.residuals[k])) for (i, p) in enumerate(VgtkCore.geom.(u)) for k in (get(point_of, i, 0),)))
    return nothing
end

"The driver of every chrome set's config (configs/experiments/chrome_evaluation_fx3_<set>.toml)."
const CHROME_EVALUATION_DRIVER = "scripts/chrome_evaluation.jl"
"""
    chrome_upstream(cfg, path, stage, what) -> dir

The detect or index product (`stage`) of the chrome set `cfg` (the config at `path`) a consumer (`what`) reads, produced on demand
(`produce_upstream`): its driver through that stage (`driver_run`: prepare, detect[, index], each loading or producing its own
upstream), a GPU's when its detect is missing and runs on one (`[detect] device = "gpu"`).
"""
function chrome_upstream(cfg::AbstractDict, path::AbstractString, stage::AbstractString, what::AbstractString)
    p = stage == "detect" ? detect_product(cfg) : stage == "index" ? index_product(cfg) :
        throw(ArgumentError("chrome_upstream: stage detect or index, got $(repr(stage))"))
    gpu = detect_settings(cfg, path).device == "gpu" && !produced(detect_product(cfg))
    return produce_upstream(p, what; driver_run(CHROME_EVALUATION_DRIVER, path, stage == "detect" ? "prepare,detect" : "prepare,detect,index")..., gpu)
end

"""
    register_photos(cfg, path; driver) -> (; dir, rows, wall)

The index stage of the chrome evaluation config `cfg` (read from `path`): every photograph of `[index]` (`index_settings`)
registered by `register_photo` on the grid of the target record through the camera of record, into the index product
(`index_product(cfg)`, produced or loaded: `produce`): `write_registration`'s files per registered photograph, targets.json, summary.tsv (`INDEX_SUMMARY_COLUMNS`, VisualGeometryToolkitEval's table),
photos.json and the run records (`driver`, `path`). Each photograph's local optimization is counted (Estimation's `CountedLO`, a
fresh one per photograph: it returns the model its strategy returns): photos.json holds per photograph its registration (inliers,
matched, trials, rms, the detections assigned), its seconds and every LO call's iterations and retcode — the rows the index grid
reads, so the photographs are registered once. The photographs are registered concurrently, one per thread (each its own RNG of
`rng_seed`): the registrations are the serial ones; `wall`: the sum of their seconds (each measured on its thread; no paper asset
reads it, the index grid's comparison only).
"""
function register_photos(cfg::AbstractDict, path::AbstractString; driver::AbstractString)
    dir = produce(index_product(cfg); product_records(driver, path)...) do dir
        Produced(; inputs = _register_photos(cfg, path, dir))
    end
    r = VgtkIO.read_record(joinpath(dir, "photos.json"); allownan = true)
    return (; dir, rows = r["photos"], wall = r["wall_s"])
end
function _register_photos(cfg::AbstractDict, path::AbstractString, dir::AbstractString)
    I = index_settings(cfg, path)
    detect_dir = chrome_upstream(cfg, path, "detect", "the detect product the index reads")
    paths = Dict(zip(stage_inputs(cfg, "index"), require_stage_inputs(cfg, "index")))
    tpath, cpath = paths[I.target], paths[I.camera]
    grid = grid_of_record(tpath)
    # the camera the dots are registered through (IO: the store's camera by name), read as the BoundedCamera it is stored as: the
    # registration maps given points (the detections, the grid's nodes) through its model alone, so a fisheye index camera bounded
    # by its measured image region (the front end's fit, e.g. fits.fisheye30cm_kb) is read too — never a pixel-domain operation
    camera = VgtkCore.model(VgtkIO.load(Cameras.BoundedCamera, cpath; name = I.camera_name))
    rig = target_rig(VgtkIO.load(AbstractTarget, tpath), I.target_id)          # the record's target as it is (the grid's is plain metres)
    VgtkIO.save(joinpath(dir, "targets.json"), rig)
    rig_id = VgtkIO.content_id(joinpath(dir, "targets.json"))
    # every photograph on its own thread (each its own StableRNG of rng_seed, its own LO counter and files); the rows in photo order
    done = Vector{Any}(undef, length(I.photos))
    Threads.@threads for i in eachindex(I.photos)
        lo = CountedLO(I.local_optimization_strategy)            # a fresh counter per photograph: its call_records are this photograph's
        done[i] = (; lo, r = register_photo(camera, grid, detect_dir, merge(I, (; local_optimization_strategy = lo)), I.photos[i]))   # u: Core's Uncertain pixels
    end
    summary = Tuple[]
    rows = Dict{String, Any}[]; wall = 0.0
    for (photo, d) in zip(I.photos, done)
        stem = splitext(photo)[1]
        lo = d.lo; (; u, reg, err, seconds) = d.r
        wall += seconds
        calls = [[c.iterations, string(c.retcode)] for c in lo.call_records]      # each LO call: its iterations, its ReturnCode by name
        n_u = length(u)
        if reg === nothing                                        # a photograph that does not register is recorded, not fatal (its row NaN)
            push!(summary, (photo, n_u, 0, 0, n_u, 0, NaN, NaN))   # INDEX_SUMMARY_COLUMNS: no inliers, nothing matched
            push!(rows, Dict{String, Any}("photo" => photo, "n" => n_u, "ok" => false, "reason" => sprint(showerror, err), "seconds" => seconds, "lo" => calls))
            println(">> ", photo, ": NOT REGISTERED — ", sprint(showerror, err)); flush(stdout)
            continue
        end
        write_registration(dir, stem, photo, reg, u, I, rig_id)
        rs = residual_summary(collect(values(reg.residuals)))
        n = length(reg.assignment)
        push!(summary, (photo, n_u, reg.inliers, n, n_u - n, reg.trials, Float64(rs.rms_px), Float64(rs.max_px)))
        push!(rows, Dict{String, Any}("photo" => photo, "n" => n_u, "ok" => true, "inliers" => reg.inliers, "matched" => n, "trials" => reg.trials,
                                      "rms" => rs.rms_px, "scale_px" => reg.noise_scale, "reassigned" => reg.reassigned, "rounds" => reg.rounds, "seconds" => seconds, "assigned" => sort(collect(values(reg.assignment))), "lo" => calls))
        println(">> ", photo, ": ", n, " of ", n_u, " ", I.label, " points matched (", reg.inliers, " inliers, ", reg.trials, " trials), rms ",
                round(rs.rms_px; digits = 3), " px, max ", round(rs.max_px; digits = 3), " px, ", something(reg.reassigned, "none"), " re-assigned from the LO's matches")
        flush(stdout)
    end
    # the photographs refused for more than one connected component: a defect to investigate, said loudly
    split_ = [photo for (photo, d) in zip(I.photos, done) if d.r.err isa LatticeComponents]
    isempty(split_) || println(">> WARNING: ", length(split_), " of ", length(rows), " photographs refused for more than one connected component: ", join(split_, ", "))
    mislabel = [photo for (photo, d) in zip(I.photos, done) if d.r.err isa MislabelledDots]
    isempty(mislabel) || println(">> WARNING: ", length(mislabel), " of ", length(rows), " photographs refused for a mislabelled dot: ", join(mislabel, ", "))
    write_tsv(joinpath(dir, "summary.tsv"), INDEX_SUMMARY_COLUMNS, summary)
    VgtkIO.write_record(joinpath(dir, "photos.json"), Dict{String, Any}("schema" => EXPERIMENT_RECORD_SCHEMAS.chrome_index_photos, "wall_s" => wall, "photos" => rows); allownan = true)
    println(">> index → ", dir)
    return [tpath, cpath, detect_dir]
end

"""
    index_camera_size(cfg, I) -> (W, H)

The photographs' size: the extent of the index stage's camera of record (`I = index_settings(cfg, path)`; its `camera`, an input or a
`[runs]` entry, read by `config_source`, the camera `camera_name` of that file).
"""
index_camera_size(cfg::AbstractDict, I) =
    Tuple(VgtkCore.extent(VgtkIO.load(Cameras.BoundedCamera, config_source(cfg, I.camera); name = I.camera_name)))

"""
    evaluate_models(cfg, path; driver, index_dir = nothing, out = nothing) -> String

The evaluate stage of the chrome evaluation config `cfg` (read from `path`): every model of `[evaluate]` (`evaluate_model_specs`) on the
registered dots (`evaluate_grid_models`) of the index stage's run (`[evaluate] index = "stage"`) or the index run of record, into the
evaluate run with its records (`driver`, `path`). Given `index_dir` and `out`, the index run `index_dir` (a subset of a stage's
registrations: the index grid's common photographs) is scored into `out`, no run records. Returns the directory written.
"""
function evaluate_models(cfg::AbstractDict, path::AbstractString; driver::AbstractString, index_dir = nothing, out = nothing)
    out === nothing || return (_evaluate_models(cfg, path, index_dir, mkpath(out)); out)   # a subset of a stage's registrations: no records
    return produce(evaluate_product(cfg); product_records(driver, path)...) do dir
        Produced(; inputs = _evaluate_models(cfg, path, nothing, dir))
    end
end
_evaluate_models(cfg::AbstractDict, path::AbstractString, index_dir, dir::AbstractString) = score_chrome_models(cfg, path, index_dir, dir).inputs

"""
    score_chrome_models(cfg, path, index_dir, dir; settings = evaluate_settings(cfg, path), models = evaluate_model_specs(cfg))
        -> (; inputs, summaries)

The evaluate stage's scoring of the chrome set `cfg` (the config at `path`) into `dir`: its index run (the index product, or
`index_dir`), its target's grid and its lens coverage, and `evaluate_grid_models` on `models` (`(label, "<kind>:<file>")` pairs,
the `[evaluate]` models by default) under `settings` (`evaluate_settings`'s, the config's by default). Returns the files read
(`inputs`) and `evaluate_grid_models`' summaries. THE chrome scorer: the evaluate stage's, and the convergence experiment's
(src/convergence.jl: a refinement's every iteration scored as one model each, `holdout` on).
"""
function score_chrome_models(cfg::AbstractDict, path::AbstractString, index_dir, dir::AbstractString;
                             settings = evaluate_settings(cfg, path), models = evaluate_model_specs(cfg))
    E = settings
    stage = E.index == "stage"
    if stage
        idir = index_dir === nothing ? chrome_upstream(cfg, path, "index", "the index product the evaluation reads") : index_dir
        I = index_settings(cfg, path)
        W, H = index_camera_size(cfg, I)
    else
        idir = something(index_dir, config_input(cfg, E.index))                                    # the index run of record
        W, H = E.image_size
    end
    E = merge(E, profile_frame(lens_coverage_region(cfg, E.lens_coverage, W, H; what = "$path [evaluate]"), W, H))   # the set's coverage: the profile's frame
    tpath = config_input(cfg, E.target)
    grid = grid_of_record(tpath)
    foreach(((label, spec),) -> require_data(replace(spec, r"^\w+:" => ""); what = "model $label", provenance = "[evaluate] model $label"), models)
    detections = stage ? joinpath(product_dir(detect_product(cfg)), "detections") : nothing      # the dots' Förstner covariances, which the start reads
    summaries = evaluate_grid_models(idir, grid, E, dir; W, H, models, detections)
    println(">> evaluate → ", dir)
    return (; inputs = [tpath; idir; detections === nothing ? String[] : [dirname(detections)]; [replace(spec, r"^\w+:" => "") for (_, spec) in models]],
            summaries)
end


# ---- stage evaluate -----------------------------------------------------------------------------------------------------------------
# The registered dots (stage index: points/<photo>.json, the record point of each circular point) scored through every model of
# [evaluate].models. The camera turned on a tripod head about ONE fixed point while the target stood still, so a set's poses are ONE
# rigid motion model, x_cam,k = R_k (X − c) + d — the rotation centre c (target frame), the projection centre's offset from it d (camera
# frame) and a rotation per photograph: 6 + 3N unknowns in place of 6N free poses, which would absorb most of a model's error over the
# patch one photograph covers. Per model (round 7, owner rulings of 2026-11-11):
#   1. the start, `_fit_rotation_center`: c, d from every dot's bearing (the model's rays, `load_ray_model`, each ray's foot point
#      nearest the model's best single centre c★: the origins along a ray line are a gauge) at the index stage's rotations, the
#      Sampson-whitened cross-product rows in one least squares (below);
#   2. the refine: Core's `LevenbergMarquardt` on VisualGeometryToolkitEstimation's `CameraPosesProblem` of its rotation-centre view parametrization
#      (`RotationCenterViews`: one rotation per photograph, c and d shared), the model's camera held (every parameter `fixed`), on the
#      BEARING residual (owner ruling 2026-11-12 (B)(ii)): each dot back-projected once through the model, the tangent-plane angle of its
#      bearing to the predicted bearing of R_k(X − c) + d, whitened into pixels by the back-projection's Jacobian — no forward
#      projection inside the solve; [evaluate] optimization_strategy (the Triggs-corrected Huber LM, one robustified solve) at
#      efficiency_constant × the index stage's pooled noise scale (Estimation's `PooledScale` of the photographs' MAD scales of
#      their index pixel residuals, handed down: `ScaleUpstream`; a refine estimates no scale). No scorer, no re-selection: the refine is the stack's.
# The gauge: none when the
# photographs turned about two axes; a set turned about one axis is refused by Estimation. The fit is capped at [evaluate] iterations,
# and eval_<label>.json `fit` records its retcode, iterations, scale and its source, loss width and down-weighted dots; `start` the
# start's c, d, retcode and conditioning (the whitened system's singular values, normalized and not). The translation residual of the
# start against the index poses says how far the capture was from one rotation centre: it is recorded per photograph
# (`start_translation_residual_mm`) and summarised in summary.md.
# Scored after the refine (not the refine's residual, which is the bearing's above): each dot's reprojection error in px through the
# model's camera at the refined motion (prediction − detection), and in mrad through the
# model's local scale at the dot (the ray direction's analytic Jacobian, Core's `backproject_with_jacobian` of RadiometricCamera's
# `load_camera`), overall and in two regions about `centre`: ρ = |𝐩 − centre| / rmax below `centre_rho` and above `edge_rho`.
# Each model is fitted and scored on the dots IT can evaluate (owner 2026-11-15): those it unprojects, the start projects and the
# estimate projects — its coverage, reported beside its score (the dots and their fraction of every registered dot). The head-to-head
# between the models of the run is on the dots EVERY model scores (`common_dots`, by photograph and registered dot), so every median
# and RMS of summary.tsv is on the same observations; each model is also scored on all of its own dots (the `all_` columns). The
# Schöps baseline is his own camera (kind "schoeps": `check_schoeps_baseline`, `schoeps_own_camera`), never one re-estimated from his.
# Only the photographs whose registration assigns every dot of the grid enter (`_all_dots`): a grid cut off by the frame, or with dots
# undetected, may be placed in the record's grid with a shift of whole spacings, which the motion model would read as a translation of
# the camera.
# Outputs: eval_<label>.json (the motion variant's summary — and motion_heldout's with `holdout` — per-photograph table, radial profile
# about `centre` out to `rmax`, the 8 × 6 map, c and d, the rays' offsets from c*), vectors_<label>_<variant>.json (every dot's
# reprojection error, prediction − detection in px: figures.jl's "scatter" vectors {x, y, dx, dy} for the colour-wheel maps),
# summary.tsv and summary.md.
using Statistics: median, mean, quantile

"The keys of `[evaluate]`, every one required (no code default); `image_size` exactly when `index` names an input."
const CHROME_EVALUATE_KEYS = ("index", "target", "models", "optimization_strategy", "iterations", "profile_bins", "lens_coverage",
                                "centre_rho", "edge_rho", "holdout", "bootstrap", "comparisons")
"The model kinds of an `[evaluate]` model (`load_ray_model`)."
const CHROME_MODEL_KINDS = ("lens", "grid", "schoeps")
"""The evaluation summary's header (owner 2026-11-15): `photos`, `dots` and the statistics through `edge_mean_px` the head-to-head,
on the dots every model of the run scores (`common_dots`); `coverage_dots` the dots the model scores and `coverage_fraction` their
fraction of every registered dot (`dot_coverage`); `all_photos` and the `all_` statistics the model on every dot it scores."""
const EVALUATE_SUMMARY_COLUMNS = ("model", "variant", "photos", "dots", "coverage_dots", "coverage_fraction", "median_px", "mean_px", "p95_px", "rms_px",
                                  "median_mrad", "centre_median_px", "centre_mean_px", "edge_median_px", "edge_mean_px",
                                  "all_photos", "all_median_px", "all_mean_px", "all_p95_px", "all_rms_px", "c_mm", "d_mm", "retcode")
"""The `ReturnCode` of a summary row's rotation-centre refine (`EVALUATE_SUMMARY_COLUMNS` `retcode`, written by its name): Core's
`retcode` of the fit; of the held-out variant its two fits' records combined by Core's `worst_retcode` (the more severe)."""
variant_retcode(fits) = first(VgtkCore.worst_retcode(((f.r.retcode, f.r.stop_rule) for f in fits)...))
"""
    upstream_refinements(cfg) -> Dict

The config of record `[upstream] radiometric_refinement` names (`load_radiometric_refinement_config`, src/radiometric_refinement.jl, which the
driver includes before this file).
"""
upstream_refinements(cfg::AbstractDict) = load_radiometric_refinement_config(projectdir(cfg["upstream"]["radiometric_refinement"]))

"""
    evaluate_settings(cfg, path = "the chrome evaluation config") -> NamedTuple

`[evaluate]` checked: every key of `CHROME_EVALUATE_KEYS` required, nothing else but `image_size`. `index` is `"stage"` (this
config's index run) or an input of stage evaluate — an index run of record (its `registration/`), read in place of the index stage,
with `image_size` [W, H] (px) the photographs' size, which that stage would read off the detect run. `target` is an input: the target
record whose grid the registrations index. `models` is a non-empty array of `{label, kind, input}` or `{label, kind, case}` tables —
`kind` one of `CHROME_MODEL_KINDS` (`load_ray_model`), `input` an input of stage evaluate (a model record under data_root()),
`case` a case of the config `[upstream] radiometric_refinement` names (its refined model: `last_model`); labels unique.
`lens_coverage` the chrome set's coverage (src/image_region.jl: "sensor" or "image_region:<name>"), the radial profile's frame
(`profile_frame`: its centre and its farthest imaged radius, COLMAP px — no centre or radius typed in); the summary's regions are
ρ = |𝐩 − centre| / rmax below `centre_rho` (the centre) and above `edge_rho` (the edge), 0 ≤ `centre_rho` < `edge_rho` ≤ 1. `max_unassigned` (optional, default 0): a photograph enters the evaluation when at
most this many dots of the grid are unassigned (`_all_dots`); it must stay below the grid's shorter side (`check_max_unassigned`),
where a labelling shifted by a whole spacing would leave a whole row or column unassigned, so the labelling stays pinned (2026-11-06:
the fisheye at ≈ 10 cm, whose central camera of record leaves a few rim dots > 8 px off in most photographs). `optimization_strategy`
(required): the rotation-centre refine's strategy, an Estimation strategy table read by `optimization_strategy_from_toml` (the
`camera_poses_problem` class) of type `"triggs"` — the Triggs-corrected LM at its loss and efficiency_constant × the index stage's
pooled noise scale (`PooledScale`; a refine estimates none); `iterations`: its LM iteration cap. `holdout`, a switch
(`"on"` | `"off"`, RadiometricCamera's `read_switch`; `"on"` in the configs of record, the paper's headline): also the held-out
variant `motion_heldout`, the motion fit on the dots of one parity of the grid nodes, (i + j) even or odd, scoring the other, pooled.
`bootstrap` `{resamples, seed, alpha}` and `comparisons` (an array of `{model, reference}`, two labels of `models`): the paired
photograph bootstrap of every comparison's differences (`paired_differences`; `resamples` ≥ 1 draws, `StableRNG(seed)`, the
`1 − alpha` interval, 0 < `alpha` < 1).
"""
function evaluate_settings(cfg::AbstractDict, path::AbstractString = "the chrome evaluation config")
    t = get(cfg, "evaluate", nothing)
    where = "$path [evaluate]"
    t isa AbstractDict || throw(ArgumentError("$path: no [evaluate] table"))
    VgtkCore.check_keys(t, (CHROME_EVALUATE_KEYS..., "image_size", "max_unassigned"), where)
    foreach(k -> VgtkCore.required_key(t, k, where), CHROME_EVALUATE_KEYS)
    max_unassigned = haskey(t, "max_unassigned") ? Int(_number(t, "max_unassigned", where; integer = true)) : 0
    max_unassigned >= 0 || throw(ArgumentError("$where max_unassigned must be ≥ 0"))
    stage = t["index"] == "stage"
    if stage
        haskey(cfg, "index") || throw(ArgumentError("$where index = \"stage\" needs the [detect] and [index] stages"))
        haskey(t, "image_size") && throw(ArgumentError("$where image_size: the index stage's photographs give it (the detect run's summary)"))
    else
        i = get(cfg["inputs"], t["index"], nothing)
        i !== nothing && i["stage"] == "evaluate" ||
            throw(ArgumentError("$where index: $(repr(t["index"])) is neither \"stage\" nor an input of stage evaluate"))
        haskey(cfg, "index") && throw(ArgumentError("$where index: an index run of record and an [index] stage — one of them"))
        sz = get(t, "image_size", nothing)
        sz isa AbstractVector && length(sz) == 2 && all(x -> x isa Integer && x > 0, sz) ||
            throw(ArgumentError("$where image_size must be [W, H] (px) when index is an input"))
    end
    haskey(cfg["inputs"], t["target"]) || throw(ArgumentError("$where target: $(repr(t["target"])) is not an input"))
    stage && t["target"] != cfg["index"]["target"] && throw(ArgumentError("$where target: the index stage registers against $(repr(cfg["index"]["target"]))"))
    ms = t["models"]
    ms isa AbstractVector && !isempty(ms) && all(x -> x isa AbstractDict, ms) ||
        throw(ArgumentError("$where models must be a non-empty array of {label, kind, input | case} tables"))
    rc = nothing
    for m in ms
        VgtkCore.check_keys(m, ("label", "kind", "input", "case"), "$where model")
        label = get(m, "label", "")
        label isa AbstractString && !isempty(label) || throw(ArgumentError("$where: a model without a label"))
        get(m, "kind", "") in CHROME_MODEL_KINDS || throw(ArgumentError("$where model $label: kind must be one of $(CHROME_MODEL_KINDS)"))
        haskey(m, "input") ⊻ haskey(m, "case") || throw(ArgumentError("$where model $label: one of input, case"))
        check_schoeps_baseline(cfg, m, "$where model $label")
        if haskey(m, "input")
            i = get(cfg["inputs"], m["input"], get(get(cfg, "runs", Dict()), m["input"], nothing))   # an input, or a [runs] file
            i !== nothing && i["stage"] == "evaluate" || throw(ArgumentError("$where model $label: $(repr(m["input"])) is not an input or a [runs] entry of stage evaluate"))
        else
            haskey(get(cfg, "upstream", Dict()), "radiometric_refinement") ||
                throw(ArgumentError("$where model $label: a case model needs [upstream] radiometric_refinement"))
            rc === nothing && (rc = upstream_refinements(cfg))
            haskey(rc["cases"], m["case"]) || throw(ArgumentError("$where model $label: no case $(repr(m["case"])) in $(cfg["upstream"]["radiometric_refinement"])"))
            rc["cases"][m["case"]]["kind"] == m["kind"] ||
                throw(ArgumentError("$where model $label: case $(m["case"]) writes a $(rc["cases"][m["case"]]["kind"]) model, not $(m["kind"])"))
        end
    end
    labels = [m["label"] for m in ms]
    allunique(labels) || throw(ArgumentError("$where: model labels must be unique"))
    lens_coverage = check_lens_coverage(cfg, t["lens_coverage"], where)
    centre_rho = Float64(_number(t, "centre_rho", where)); edge_rho = Float64(_number(t, "edge_rho", where))
    0 <= centre_rho < edge_rho <= 1 || throw(ArgumentError("$where centre_rho = $centre_rho, edge_rho = $edge_rho: 0 ≤ centre_rho < edge_rho ≤ 1"))
    holdout = RadiometricCamera.read_switch(t["holdout"], "$where holdout")       # "on" | "off" (RC 0.11.4), parsed once here
    t["optimization_strategy"] isa AbstractDict ||
        throw(ArgumentError("$where optimization_strategy must be a strategy table ({ type = \"triggs\", loss = …, efficiency_constant = … })"))
    lo = optimization_strategy_from_toml(t["optimization_strategy"]; class = :camera_poses_problem, section = "$where.optimization_strategy")
    lo isa Triggs || throw(ArgumentError("$where optimization_strategy: $(t["optimization_strategy"]["type"]) — the rotation-centre refine is the robustified LM, type \"triggs\""))
    b = check_required_keys(t["bootstrap"], ("resamples", "seed", "alpha"), "$where bootstrap")
    bootstrap = (; resamples = Int(_number(b, "resamples", "$where bootstrap"; integer = true)), seed = Int(_number(b, "seed", "$where bootstrap"; integer = true)),
                 alpha = Float64(_number(b, "alpha", "$where bootstrap")))
    bootstrap.resamples >= 1 && 0 < bootstrap.alpha < 1 || throw(ArgumentError("$where bootstrap: resamples ≥ 1 and 0 < alpha < 1"))
    cs = t["comparisons"]
    cs isa AbstractVector && all(x -> x isa AbstractDict, cs) || throw(ArgumentError("$where comparisons must be an array of {model, reference} tables"))
    comparisons = map(cs) do c
        check_required_keys(c, ("model", "reference"), "$where comparison")
        c["model"] in labels && c["reference"] in labels && c["model"] != c["reference"] ||
            throw(ArgumentError("$where comparison $(repr(c["model"])) − $(repr(c["reference"])): two different labels of `models`"))
        (; model = String(c["model"]), reference = String(c["reference"]))
    end
    allunique(comparisons) || throw(ArgumentError("$where: a comparison is stated twice"))
    return (; index = String(t["index"]), target = String(t["target"]), image_size = stage ? nothing : (Int(t["image_size"][1]), Int(t["image_size"][2])),
            models = [Dict{String,Any}(m) for m in ms],
            optimization_strategy = lo, optimization_strategy_table = Dict{String, Any}(t["optimization_strategy"]),
            iterations = Int(_number(t, "iterations", where; integer = true)),
            profile_bins = Int(_number(t, "profile_bins", where; integer = true)),
            lens_coverage, centre_rho, edge_rho, max_unassigned,
            holdout, bootstrap, comparisons)
end

"""
    check_schoeps_baseline(cfg, m, where)

The Schöps baseline is T. Schöps's OWN camera (owner 2026-11-15), never a camera re-estimated from his (a refinement's grid, an index
fit): a model `m` of kind `"schoeps"` names an `input` — a `[runs]` entry of a front end's Schöps run (`run = "schoeps.<name>"`,
`path = "camera.json"`: the front end's camera record, his camera imported onto the raw grid) or an `[inputs]` directory, a Schöps
run (`schoeps_own_camera` refuses one that does not hold his `intrinsics0.yaml`) — never a `case`; and a front end's Schöps run is
read as the Schöps baseline only (kind `"schoeps"`), never as another kind's model. An `ArgumentError` names what is refused.
"""
function check_schoeps_baseline(cfg::AbstractDict, m::AbstractDict, where::AbstractString)
    run = haskey(m, "input") ? get(get(cfg, "runs", Dict()), m["input"], nothing) : nothing    # a [runs] entry, or nothing (an input, a case)
    schoeps_run = run !== nothing && startswith(String(run["run"]), "schoeps.")
    if m["kind"] == "schoeps"
        haskey(m, "case") && throw(ArgumentError("$where: kind schoeps is T. Schöps's own camera, never a refinement case's model (case $(m["case"]))"))
        if run !== nothing
            schoeps_run && run["path"] == "camera.json" ||
                throw(ArgumentError("$where: kind schoeps is T. Schöps's own camera, the camera record of a front end's Schöps run " *
                                    "([runs] run = \"schoeps.<name>\", path = \"camera.json\"), not $(run["run"])'s $(run["path"])"))
        else
            p = String(get(get(cfg["inputs"], m["input"], Dict()), "path", ""))
            isempty(splitext(p)[2]) || throw(ArgumentError("$where: kind schoeps is a Schöps run's directory (his intrinsics0.yaml), not the file $p"))
        end
    elseif schoeps_run
        throw(ArgumentError("$where: $(m["input"]) is the camera of the Schöps run $(run["run"]): the Schöps baseline, kind \"schoeps\", not $(m["kind"])"))
    end
    return nothing
end

"""
    schoeps_own_camera(path) -> String

T. Schöps's own camera of the Schöps run `path` names (its directory, or its camera file): the run's camera file, RadiometricCamera's
`schoeps_run_file` (the front end's import `camera.json`, else his `intrinsics0.yaml`). An `ArgumentError` for anything else: a
directory or a file that no Schöps run (his `intrinsics0.yaml`) holds as its camera — a camera re-estimated from his is never the
Schöps baseline (`check_schoeps_baseline`).
"""
function schoeps_own_camera(path::AbstractString)
    dir = isdir(path) ? path : dirname(path)
    f = RadiometricCamera.schoeps_run_file(dir, :camera)
    isfile(joinpath(dir, "intrinsics0.yaml")) && isfile(f) && (isdir(path) || abspath(path) == abspath(f)) ||
        throw(ArgumentError("the Schöps baseline $path is not T. Schöps's own camera: no Schöps run (his intrinsics0.yaml) holds it as its camera " *
                            "— a camera re-estimated from his is never the baseline"))
    return f
end

"""
    evaluate_model_specs(cfg) -> Vector{Tuple{String,String}}

The models of `[evaluate]` as `(label, spec)` (`load_ray_model`): an input's path under data_root(), or a case's refined model
(`last_model` of its run: the final model, else the newest checkpoint; an `ArgumentError` naming the driver when the case has none).
"""
function evaluate_model_specs(cfg::AbstractDict)
    rc = any(m -> haskey(m, "case"), cfg["evaluate"]["models"]) ? upstream_refinements(cfg) : nothing
    out = Tuple{String,String}[]
    for m in cfg["evaluate"]["models"]
        if haskey(m, "input")
            push!(out, (m["label"], "$(m["kind"]):" * config_source(cfg, m["input"])))
        else
            f = last_model(product_dir(refinement_product(rc, m["case"])), m["kind"])
            isempty(f) && throw(ArgumentError("model $(m["label"]): case $(m["case"]) has no $(m["kind"]) model yet " *
                                              "(julia --project=experiments experiments/scripts/radiometric_refinement.jl --stage=refine --case=$(m["case"]))"))
            push!(out, (m["label"], "$(m["kind"]):$f"))
        end
    end
    return out
end

"""
    evaluate_product(cfg) -> Product

The evaluate product (its output summary.md, written last): keyed by `[evaluate]`, the inputs it reads (tier, path: the target,
the models given as inputs, an index run of record) and the keys of the products it reads (`upstream`: the index stage's, the
refinements of its case models) — an evaluate-only change re-keys it alone and loads the index and the refinements.
"""
function evaluate_product(cfg::AbstractDict)
    t = cfg["evaluate"]
    ids = [t["target"]; [m["input"] for m in t["models"] if haskey(m, "input")]; t["index"] == "stage" ? String[] : [t["index"]]]
    up = Dict{String,Any}()
    t["index"] == "stage" && (up["index"] = index_product(cfg).key)
    n = lens_coverage_name(t["lens_coverage"])
    n === nothing || (up["image_region"] = image_region_product(cfg, n).key)   # the set's coverage: the profile's frame
    cases = [m["case"] for m in t["models"] if haskey(m, "case")]
    if !isempty(cases)
        rc = upstream_refinements(cfg)
        up["radiometric_refinement"] = Dict{String,Any}(c => refinement_product(rc, c).key for c in cases)
    end
    return driver_product("evaluate", Dict{String,Any}("experiment" => cfg["experiment"], "evaluate" => Dict{String,Any}(t),
                                                       "inputs" => inputs_record(cfg, unique(ids)), "upstream" => up); output = "summary.md")
end

"""
    read_registrations(index_dir, g) -> Vector

The registered dots of an index run: per photograph `(name, xy, X, R, t, Σ)` — its points record (points/<stem>.json, IO's
`TargetObservations`: the detections assigned to a record point, in the detections' order) as their pixels (COLMAP) and Förstner
covariances (px²), the points' positions on the target (metres, target frame), the registration's pose (poses/<stem>.json, IO's
`TargetPoses`, in metres: `R`, `t`, target → camera) and its index pixel residuals' MAD scale (registration/<stem>_scale.json).
"""
function read_registrations(index_dir::AbstractString, g::ThorlabsGrid)
    out = []
    for f in sort(filter(endswith(".json"), readdir(joinpath(index_dir, "points"); join = true)))
        stem = splitext(basename(f))[1]
        pj = joinpath(index_dir, "poses", stem * ".json"); sj = joinpath(index_dir, "registration", stem * "_scale.json")
        isfile(pj) && isfile(sj) || continue
        sets = VgtkIO.load(TargetObservations, f)[1]
        isempty(sets) && continue
        obs = only(values(sets))                                                       # record point index => detection
        pose = VgtkCore.ustrip(m, only(values(VgtkIO.load(TargetPoses, pj)[1])))         # target → camera, metres
        sc = VgtkIO.read_record(sj)
        push!(out, (; name = stem, xy = [SVector{2, Float64}(VgtkCore.geom(x)) for x in VgtkCore.targets(obs)],
                    X = [_node_point(g, g.nodes[k]) for k in VgtkCore.sources(obs)],                     # the record point on the target plane (z = 0), metres
                    R = SMatrix{3, 3, Float64}(pose.R), t = SVector{3, Float64}(pose.t),
                    Σ = [SMatrix{2, 2, Float64}(cov(x)) for x in VgtkCore.targets(obs)],
                    noise_scale = Float64(something(get(sc, "noise_scale", nothing), NaN)), dof = Int(something(get(sc, "dof", nothing), 0))))
    end
    return out
end

"""
    _all_dots(r, g, max_unassigned = 0) -> Bool

Whether the registration `r` (`read_registrations`) assigns every dot of the grid `g` but at most `max_unassigned`: only then is its
labelling pinned to the record. A labelling shifted by a whole spacing leaves a whole row or column unassigned, so `max_unassigned`
must stay below the grid's shorter side (`check_max_unassigned`); 0 (the default) admits the complete registrations alone.
"""
_all_dots(r, g::ThorlabsGrid, max_unassigned::Integer = 0) = length(g.nodes) - length(r.X) <= max_unassigned

"`[evaluate] max_unassigned` against the grid `g`: below its shorter side, else an ArgumentError (`_all_dots`)."
function check_max_unassigned(mu::Integer, g::ThorlabsGrid)
    mu < min(g.cols, g.rows) || throw(ArgumentError("evaluate: max_unassigned = $mu — a labelling shifted by a whole spacing leaves $(min(g.cols, g.rows)) dots unassigned, so it must stay below that"))
    return mu
end

# ---- one target frame for every photograph ---------------------------------------------------------------------------------------
# A photograph's registration labels the grid from an arbitrary corner (the dots are identical: the 4 rotations of the square and the
# mirror of viewing through the glass, absorbed by the pose), so the index stage's poses live in one of 8 target frames each. The motion
# model needs ONE: every photograph's labelling is mapped by the symmetry that puts its camera on the +z side of the target and its pose
# nearest the set's one rotation centre — the motion model's own constraint on the poses, t′ = d − R′ c, fitted to the index poses (a
# rotation nearest the first photograph's alone took the wrong quarter turn where the camera rolled far: the fisheye 27.10's DSC02938
# and DSC02939, 140 mm off the centre and ≈ 20 px on every dot). A symmetry X′ = A X + b (A a rotation about the
# grid's normal through its centre, with or without the row mirror) relabels the nodes and transforms the pose exactly: R′ = R A⁻¹ (times
# D = diag(1, 1, −1) when A is improper — the points lie in z = 0, so the z flip costs nothing and makes R′ proper), t′ = t − R′ b.
"The 8 symmetries of the square grid as `(A, b)`, metres: rotations by k · 90° about the centre, each without and with the row mirror."
function grid_symmetries(g::ThorlabsGrid)
    g.cols == g.rows || throw(ArgumentError("grid_symmetries: a $(g.cols) × $(g.rows) grid has no 90° symmetry"))
    cg = SVector((g.cols - 1) / 2 * g.spacing, (g.rows - 1) / 2 * g.spacing, 0.0)
    Mx = SMatrix{3, 3, Float64}(1, 0, 0, 0, -1, 0, 0, 0, 1); bm = SVector(0.0, (g.rows - 1) * g.spacing, 0.0)
    out = Tuple{SMatrix{3, 3, Float64}, SVector{3, Float64}}[]
    for k in 0:3, mirror in (false, true)
        θ = k * π / 2; Rz = SMatrix{3, 3, Float64}(cos(θ), sin(θ), 0, -sin(θ), cos(θ), 0, 0, 0, 1)
        push!(out, mirror ? (Rz * Mx, Rz * (bm - cg) + cg) : (Rz, (I - Rz) * cg))
    end
    return out
end

"""
    _linear_least_squares(A, y) -> FitResult

The plain least squares of `y ≈ A θ`: Estimation's `ReweightedLeastSquares` with no optimization strategy (its L2 solve) on
`EstimationProblem(LinearModel, (A, y, prior))`, the prior all ones (the rows already carry their weights).
"""
_linear_least_squares(A, y) =
    solve(EstimationProblem(VgtkEst.LinearModel, (A, y, ones(length(y)))), VgtkEst.ReweightedLeastSquares(; optimization_strategy = nothing))

"""
    canonical_registrations(regs, g) -> Vector

The registrations (`read_registrations`) in one target frame: per photograph the symmetry of the grid whose transformed pose has the
camera on the +z side and lies nearest the set's one rotation centre, its record points relabelled by it. The candidates' poses
(R′, t′) are scored by the rotation-centre constraint of the motion model, `|t′ − (d − R′ c)|` (Estimation's `rotation_center_pose`),
with the centre c and the offset d the least squares of the chosen poses (`_linear_least_squares`: Estimation's `ReweightedLeastSquares` on a `LinearModel`). The start is a minimal
sample tried exhaustively: the first photograph's first candidate (the frame's gauge) with each candidate of each other photograph
gives a (c, d) — the minimum-norm solution of the pair, whose c is free along their relative rotation axis (two poses fix five of
the six coordinates) — and the one that puts every photograph's nearest candidate nearest wins; then each photograph takes its
nearest labelling and (c, d) is refitted on all, until no choice changes. A labelling is a symmetry AND a whole-pitch offset of the
infinite lattice (X′ + Δ·spacing, t′ − R′Δ·spacing: the index stage places each photograph's grid on its densest window, any offset
being a pose of its own), the offsets after the symmetries settle (the
set's sum of squared constraint residuals at its refitted centre lowered by changing one photograph's symmetry and moving its offset up to two pitches, until no
move lowers it): one photograph labelled a pitch off is moved back onto the set's centre. The refit on all is determined when the photographs
turned about two axes (the evaluation's motion fit refuses a set turned about one, Estimation's gauge check). The transform is checked exact (`R X + t = R′ X′ + t′` for every dot).
"""
function canonical_registrations(regs, g::ThorlabsGrid)
    syms = grid_symmetries(g); D = SMatrix{3, 3, Float64}(1, 0, 0, 0, 1, 0, 0, 0, -1)
    cands = map(regs) do r
        out = Tuple{SMatrix{3, 3, Float64, 9}, SVector{3, Float64}, SMatrix{3, 3, Float64, 9}, SVector{3, Float64}}[]
        for (A, b) in syms
            Rp = SMatrix{3, 3, Float64}(r.R * inv(A) * (det(A) < 0 ? D : I)); tp = r.t - Rp * b
            (-Rp' * tp)[3] > 0 && push!(out, (A, b, Rp, tp))
        end
        isempty(out) && throw(ArgumentError("canonical_registrations: $(r.name) has no labelling with the camera on the +z side"))
        out
    end
    # a candidate moved by Δ whole pitches of the infinite lattice (X′ + Δ·spacing): the same dots under another pose, t′ − R′Δ·spacing
    function shifted((A, b, Rp, tp), Δ::SVector{2, Int})
        δ = SVector(Δ[1] * g.spacing, Δ[2] * g.spacing, 0.0)
        return (A, b + δ, Rp, tp - Rp * δ)
    end
    # a labelling (j, Δ) of photograph k: its candidate j moved by Δ pitches
    labelled(k, (j, Δ)) = shifted(cands[k][j], Δ)
    # (c, d) of the chosen labellings `sel` (photograph => labelling): the least squares of t′ = d − R′ c, `nothing` when degenerate
    function centre_of(sel)
        rows = reduce(vcat, [hcat(-labelled(k, l)[3], SMatrix{3, 3, Float64}(I)) for (k, l) in sel])
        fit = _linear_least_squares(rows, reduce(vcat, [labelled(k, l)[4] for (k, l) in sel]))
        successful_retcode(fit) || return nothing
        θ = fit.u
        return SVector{3, Float64}(θ[1:3]), SVector{3, Float64}(θ[4:6])
    end
    off(cand, (center, offset)) = norm(cand[4] - translation(VgtkEst.rotation_center_pose(cand[3], center, offset)))
    nearest(cd) = [argmin(l -> off(labelled(k, l), cd), [(j, SVector(0, 0)) for j in eachindex(cands[k])]) for k in eachindex(cands)]
    total(cd) = (ch = nearest(cd); sum(k -> off(labelled(k, ch[k]), cd), eachindex(cands)))
    # each photograph takes its nearest symmetry (at the index stage's offset) and (c, d) is refitted on all, until no choice changes
    function settle(choice)
        seen = Set{typeof(choice)}()
        while length(regs) > 1 && !(choice in seen)                                 # one photograph: its own frame; a cycle ends it
            push!(seen, choice)
            cd = centre_of(collect(enumerate(choice)))
            cd === nothing && break                                                 # the least squares' refusal
            choice = nearest(cd)
        end
        return choice
    end
    # the set's consistency with one centre: the residual sum of squares of t′ = d − R′ c at its least squares, from the sums of
    # the normal equations (rows [−R′ I], right side t′): S = ΣR′, g = ΣR′ᵀt′, h = Σt′, e = Σ|t′|², and RSS = e − bᵀM⁻¹b with
    # M = [nI −Sᵀ; −S nI], b = [−g; h]. A change of one photograph's labelling is a change of its terms (`terms`), O(1)
    terms(l) = (l[3], l[3]' * l[4], l[4], sum(abs2, l[4]))
    function rss(S, gs, hs, es, n)
        M = [n * SMatrix{3, 3, Float64}(I) -S'; -S n * SMatrix{3, 3, Float64}(I)]
        b = [-gs; hs]
        F = lu(SMatrix{6, 6, Float64}(M); check = false)
        issuccess(F) || return Inf
        return es - dot(b, F \ b)
    end
    # then the offsets: one photograph's labelling at a time — its symmetry, and its offset moved by up to two pitches along each
    # axis — the change that lowers the RSS most, until none lowers it. A photograph a pitch off pulls the centre of all toward itself, so its offset is judged by the
    # refitted set, not by the centre it biased; one whose offset the centre cannot see (it alone fixes c along an axis) is left
    # as the index stage placed it (no move lowers the sum)
    # the board is finite: a labelling must put every registered dot on one of its nodes (a whole-pitch shift that moves a dot
    # off the board is no labelling). `boxes[k][j]`: the node range of photograph k's dots under symmetry j at its index offset
    nodebox(k, j) = (c = labelled(k, (j, SVector(0, 0))); ns = [round.(Int, SVector((c[1] * x + c[2])[1], (c[1] * x + c[2])[2]) ./ g.spacing) for x in regs[k].X];
                     (lo = reduce((a, b) -> min.(a, b), ns), hi = reduce((a, b) -> max.(a, b), ns)))
    boxes = [[nodebox(k, j) for j in eachindex(cands[k])] for k in eachindex(cands)]
    on_board(k, (j, Δ)) = all(boxes[k][j].lo .+ Δ .≥ 0) && all(boxes[k][j].hi .+ Δ .≤ SVector(g.cols - 1, g.rows - 1))
    function settle_offsets(choice)
        n = length(choice)
        T = [terms(labelled(k, choice[k])) for k in 1:n]
        S, gs, hs, es = sum(x -> x[1], T), sum(x -> x[2], T), sum(x -> x[3], T), sum(x -> x[4], T)
        cur = rss(S, gs, hs, es, n)
        isfinite(cur) || return choice                                              # a degenerate set: no centre to judge by
        while n > 1
            best_move, v = nothing, cur                                             # the single change that lowers the RSS most
            for k in 1:n, j in eachindex(cands[k]), a in -2:2, b in -2:2
                (j, a, b) == (choice[k][1], 0, 0) && continue
                l = (j, choice[k][2] + SVector(a, b))
                on_board(k, l) || continue
                x = terms(labelled(k, l))
                w = rss(S - T[k][1] + x[1], gs - T[k][2] + x[2], hs - T[k][3] + x[3], es - T[k][4] + x[4], n)
                w < v && ((best_move, v) = ((k, l, x), w))
            end
            (best_move === nothing || v ≥ cur - 1e-12 * es) && break                 # a lower RSS beyond its rounding (es: its scale)
            k, l, x = best_move
            S, gs, hs, es = S - T[k][1] + x[1], gs - T[k][2] + x[2], hs - T[k][3] + x[3], es - T[k][4] + x[4]
            choice = copy(choice); choice[k] = l; T[k] = x; cur = v
        end
        return choice
    end
    # the start: the first photograph's first labelling (the gauge) and each other photograph's each symmetry, at the index stage's
    # offsets, their (c, d) exact; the pair whose centre puts every photograph nearest wins (a minimal sample over the set, every
    # symmetry tried); the symmetries settle, then the offsets
    zero2 = SVector(0, 0)
    starts = filter(!isnothing, [centre_of((1 => (1, zero2), k => (j, zero2))) for k in 2:length(cands) for j in eachindex(cands[k])])
    choice = settle_offsets(settle(isempty(starts) ? [(1, zero2) for _ in cands] : nearest(argmin(total, starts))))
    out = []
    for (k, r) in enumerate(regs)
        A, b, Rp, tp = labelled(k, choice[k])
        X = [A * x + b for x in r.X]
        err = maximum(norm(r.R * r.X[i] + r.t - (Rp * X[i] + tp)) for i in eachindex(X))
        err < 1e-9 || throw(ArgumentError("canonical_registrations: $(r.name): the relabelled pose is off by $err m"))
        push!(out, (; r.name, r.xy, X, R = Rp, t = tp, Σ = get(r, :Σ, nothing),     # the pixels' covariances do not change
                    noise_scale = get(r, :noise_scale, NaN), dof = get(r, :dof, 0), offset = Tuple(choice[k][2])))   # the whole pitches it moved
    end
    return out
end

"""The unit direction's analytic Jacobian (3 × 2, rad / px) of the camera at the COLMAP pixel `p` (Core's `backproject_with_jacobian`), or
`nothing` where the camera does not back-project it. A bounded camera (every `load_camera` model) is its
model's: Core forwards `backproject_with_jacobian` through `BoundedCamera` to the model (no bound applied), and the dots outside a
model's area are dropped before (`load_ray_model`)."""
function _direction_jacobian(camera, p::SVector{2, Float64})
    M = VgtkCore.backproject_with_jacobian(camera, p)[2]
    return all(isfinite, M) ? M : nothing
end

"The unprojection's local scale at `p` (rad / px): the mean over x and y of the unit direction's derivative (the column norms of
`_direction_jacobian`), NaN where the camera does not back-project `p`."
function _local_scale(camera, p::SVector{2, Float64})
    M = _direction_jacobian(camera, p)
    return M === nothing ? NaN : (norm(M[:, 1]) + norm(M[:, 2])) / 2
end

"""
    load_ray_model(spec, W, H) -> (ray, label, camera)

`ray(𝐩)` = `(o, d)`, the ray of the COLMAP pixel 𝐩 — a point on it (metres, camera frame) and its unit direction — or `nothing` where
the model is undefined, through the model spec's bounded camera, RadiometricCamera's `load_camera` (the one loader): the rays
(`grid:`, `lens:`, `schoeps:`) go through the origin, where the camera's domain holds 𝐩 (`in_domain`) and its back-projection is finite. A model whose
image size is not `W × H` is warned about. `camera` is the bounded camera the rays are of (the evaluation's projections and Jacobians).
"""
function load_ray_model(spec::AbstractString, W::Integer, H::Integer)
    kind, rest = split(spec, ":"; limit = 2)
    camera = RadiometricCamera.load_camera(spec)
    Wm, Hm = VgtkCore.extent(camera)
    (Wm, Hm) == (W, H) || @warn "$kind model image size $(Wm)×$(Hm) ≠ $(W)×$(H)"
    label = "$kind:$(basename(kind == "schoeps" && !isdir(rest) ? dirname(rest) : rest))"
    z = zero(SVector{3, Float64})
    # a central grid's back-projection is NaN where it images nothing (Core 0.7; RC 0.4: the grid holds only the nodes its region
    # reaches): such a dot has no ray, as a `nothing` (a refined fisheye grid's NaN rays once reached the pose fit)
    central(𝐩) = (q = SVector{2, Float64}(𝐩); RadiometricCamera.in_domain(camera, q) || return nothing;
                  d = VgtkCore.backproject(camera, q); all(isfinite, d) ? (z, SVector{3, Float64}(normalize(d))) : nothing)
    return central, label, camera
end

# ---- the rotation-centre start ------------------------------------------------------------------------------------------------------
# The refine's start (owner rulings of 2026-11-11, round 7): the rotation centre c and the offset d recovered from the target-to-image
# CORRESPONDENCES of every registered photograph at once, its rotation R_k the index stage's. A dot's constraint is the collinearity of
# its bearing f with the ray from its foot point o to its target point, the homography DLT's cross product, linear in θ = [c; d]:
# g = Bᵀ [f]× (R_k (X − c) + d − o), its two independent rows on f's tangent basis B. It enters Core's constraint protocol as the model
# `RotationCenterBearing` (base/sampson.jl), whose measurement is the dot's bearing with its pixel back-projection Jacobian M (the bearing
# forms' convention, `EssentialMat`'s M₁, M₂), so G_x = −Bᵀ [P]× M is in pixels and Core's whitening (`whitened_residuals!`,
# `whitened_jacobian_rows!`) reads the dot's own pixel covariance — the target side is exact. P, the ray at which the weights are taken,
# is a column held at the index pose, so the whitened rows are linear in θ: ONE least squares (`_linear_least_squares`: Estimation's `ReweightedLeastSquares` on a `LinearModel` with
# `optimization_strategy = nothing`, its L2 solve; the rows are whitened, so its prior is all ones). Normalized on both sides with
# Core's strategies (`AffineMap(strategy, …)`): the target side by `HartleyNormalization` of the board points' (x, y) — X′ = s (X − X̄),
# o′ = s o, P′ = s P, θ′ = [s (c − X̄); s d], unnormalized exactly on the way out — and the image side by `DiagonalNormalization` of the
# image (the half-diagonal: M′ = M / σ, Σ′ = σ² Σ, the covariance's shape kept). The rows' unknowns part Bᵀ [f]× [−R I] does not read
# X, and the image scale cancels in the whitening (G_x′ = σ G_x against Σ′ = σ² Σ), so the system's conditioning is the same either way;
# the target side removes the right-hand side's cancellation at a target origin far from the board.
"""
    RotationCenterBearing(R, c, d)

One photograph of the rotation-centre motion as a model of Core's constraint protocol: its rotation `R` (target → camera, the index
stage's) fixed and the coordinates θ = [c; d], the rotation centre (target frame) and the projection centre's offset from it (camera
frame). The measurement is a dot's unit bearing `f` with the columns `X` (its target point), `o` (its ray's foot point), `M` (its
pixel back-projection Jacobian, 3 × 2) and `P` (the ray the weights are taken at). `constraint_with_jacobians` gives
g = Bᵀ [f]× (R (X − c) + d − o), G_x = −Bᵀ [P]× M (pixels) and G_θ = Bᵀ [f]× [−R I]; P is data, so ∂G_x/∂θ = 0.
"""
struct RotationCenterBearing
    R::SMatrix{3, 3, Float64, 9}
    c::SVector{3, Float64}
    d::SVector{3, Float64}
end
@inline function VgtkCore.constraint_with_jacobians(m::RotationCenterBearing, f::StaticVector{3, Float64}; X, o, M, P)
    b1, b2 = VgtkCore.s2_tangent_basis(f)
    Bt = SMatrix{2, 3, Float64, 6}(b1[1], b2[1], b1[2], b2[2], b1[3], b2[3])
    F = VgtkCore.skew(f)
    return (Bt * (F * (m.R * (X - m.c) + m.d - o)), -(Bt * VgtkCore.skew(P) * M), Bt * F * hcat(-m.R, SMatrix{3, 3, Float64}(I)))
end
VgtkCore.observation_gradient_jacobian(::RotationCenterBearing, f::StaticVector{3, Float64}; X, o, M, P) = zero(SMatrix{4, 6, Float64, 24})
VgtkCore.constraint_keywords(::RotationCenterBearing, xs) = (; X = xs.X, o = xs.o, M = xs.M, P = xs.P)

"""
    RotationCenterStartMetadata

The metadata of `_fit_rotation_center`'s result: the singular values of its whitened 6-unknown system as solved (both sides
normalized) and unnormalized (the conditioning report), and the robust contract's record — no scale (`scale` NaN, `ScaleNone`).
"""
struct RotationCenterStartMetadata
    singular_values::Vector{Float64}
    singular_values_unnormalized::Vector{Float64}
    scale::Float64
    scale_source::VgtkCore.ScaleSource
end

# The whitened rows of every photograph's dots, normalized by the board's `(s, X̄)` and the image's `σ`: `A θ′ = y`.
function _rotation_center_rows(R, dots, ids, s, X̄, σ)
    rows, rhs = Matrix{Float64}[], Vector{Float64}[]
    for k in ids
        xs = dots[k]; n = length(xs)
        ys = VgtkCore.PointSet(collect(xs); Σ = [σ^2 * S for S in xs.Σ], X = [s * (x - X̄) for x in xs.X], o = s .* xs.o,
                               M = [J / σ for J in xs.M], P = s .* xs.P)
        m0 = RotationCenterBearing(SMatrix{3, 3, Float64, 9}(R[k]), zero(SVector{3, Float64}), zero(SVector{3, Float64}))   # g(θ′) = g(0) + G_θ θ′
        push!(rhs, -VgtkCore.whitened_residuals!(zeros(2n), m0, ys))
        push!(rows, VgtkCore.whitened_jacobian_rows!(zeros(2n, 6), m0, SMatrix{6, 6, Float64}(I), ys))
    end
    return reduce(vcat, rows), reduce(vcat, rhs)
end

"""
    _fit_rotation_center(R, dots; image_size) -> FitResult{(; center, offset), RotationCenterStartMetadata}

The rotation-centre start: the centre c (target frame) and the offset d (camera frame) of photographs turned about one fixed point,
from the dots of every photograph `k` (`RotationCenterBearing`) — `R[k]` its rotation (the index stage's) and `dots[k]` a
`PointSet` of its dots' unit bearings with the columns `Σ` (pixel covariance, px²), `X` (target point), `o` (foot point), `M`
(back-projection Jacobian) and `P` (the ray the weights are taken at; the index pose's). The whitened rows of every dot stacked
into one 6-unknown linear system, normalized on both sides (`HartleyNormalization` of the board, `DiagonalNormalization` of the
`image_size` (W, H) image), solved by `_linear_least_squares` (Estimation's `ReweightedLeastSquares`, `optimization_strategy = nothing`: its L2 solve) and unnormalized.
Its termination record, iterations, rms (whitened) and support are that solve's (`ReturnCode.InitialFailure`, not `successful_retcode`, for
a degenerate system: c, d NaN); the metadata records the
system's singular values, normalized and not, and no scale (`ScaleNone`).
"""
function _fit_rotation_center(R::AbstractDict{K}, dots::AbstractDict{K}; image_size) where {K}
    ids = sort(collect(keys(dots)))
    Set(keys(R)) == Set(ids) || throw(ArgumentError("fit_rotation_center: the rotations and the dots name different photographs"))
    B = VgtkCore.AffineMap(VgtkCore.HartleyNormalization(), [SVector(x[1], x[2]) for k in ids for x in dots[k].X])
    s = B.linear[1, 1]; X̄ = SVector(-B.translation[1] / s, -B.translation[2] / s, 0.0)
    σ = VgtkCore.AffineMap(VgtkCore.DiagonalNormalization(image_size)).linear[1, 1]
    A, y = _rotation_center_rows(R, dots, ids, s, X̄, σ)
    A0, _ = _rotation_center_rows(R, dots, ids, 1.0, zero(SVector{3, Float64}), 1.0)
    fit = _linear_least_squares(A, y)
    θ = fit.u
    est = (; center = SVector{3, Float64}(θ[1:3]) / s + X̄, offset = SVector{3, Float64}(θ[4:6]) / s)
    return FitResult(est, fit.retcode, fit.stop_rule, fit.stats, fit.rms, fit.n,
                     RotationCenterStartMetadata(svdvals(A), svdvals(A0), NaN, VgtkCore.ScaleNone))
end

"""
    pooled_index_scale(data) -> FitResult

The set's one noise scale for the rotation-centre refine: Estimation's `PooledScale` of the photographs' index scales
(the MAD of each one's final pixel residuals), as their registration records hold them (`noise_scale`, `dof`: `register_grid`, `write_registration`; a photograph
whose record holds no scale — null, NaN — is left out of the pool, as `PooledScale` leaves out a `ScaleNone` result). An
`ArgumentError` when no photograph holds one: an index run before round 7 (re-run the index stage).
"""
function pooled_index_scale(data)
    # the photographs' registrations as their records hold them: Estimation's pool of the recorded scales and degrees of freedom
    pooled = solve(EstimationProblem(NoiseScale, [d.noise_scale for d in data] => [d.dof for d in data]), VgtkEst.PooledScale())
    successful_retcode(pooled) || throw(ArgumentError(
        "evaluate: no photograph's registration record holds a noise scale — an index run before round 7 (re-run the index stage), " *
        "or an index scorer that originated none: the rotation-centre refine needs the index stage's scale"))
    return pooled
end


"""
    foot_points(rays) -> (c★, feet)

The best single centre of a model's rays `rays` (`(o, d)` pairs): the point closest to all their lines (least squares, Core's `NearestPoint`), and each
ray's point nearest it (the origins along the ray lines are a gauge). Central rays (every o = 0) give c★ = 0 and feet 0.
"""
function foot_points(rays)
    all(r -> iszero(r[1]), rays) && return zero(SVector{3, Float64}), fill(zero(SVector{3, Float64}), length(rays))
    c = solve(EstimationProblem(VgtkCore.Point, [o => d for (o, d) in rays]), VgtkCore.NearestPoint()).u   # Core's least-squares point nearest to the ray lines
    return c, [o + dot(c - o, d) * d for (o, d) in rays]
end

"""The fewest dots a photograph's view of a refine holds: a pose's minimal set (three correspondences, P3P; Estimation's
`BearingPoseProblem` refuses fewer). A photograph whose start projects fewer — in either parity, for the held-out fits — is refused
by name (`start_refusals!`), never handed to the refine."""
const VIEW_MIN_DOTS = 3

"""
    start_refusals!(live, names, even = nothing) -> Vector{NamedTuple}

The photographs a model's rotation-centre start refuses: `live[k]` marks the dots of photograph `k` (named `names[k]`) the start
projects; one with fewer than `VIEW_MIN_DOTS` (a pose's minimal set) — or, given `even` (each dot's node parity: the held-out fits),
fewer of one parity — would have no view, or a view no pose fits, in the refine (`evaluate_grid_models`), so its `live` is cleared and it is returned as `(; photo, reason)`, in index order. Its dots
are never looked up after: it has no estimate pose to score them at (a kept photograph's dots the start missed are scored at its estimate).
"""
function start_refusals!(live::AbstractVector{<:AbstractVector{Bool}}, names::AbstractVector, even = nothing)
    out = NamedTuple{(:photo, :reason), Tuple{String, String}}[]
    for k in eachindex(live)
        n = count(live[k])
        reason = n == 0 ? "the start projects none of its $(length(live[k])) dots" :
                 n < VIEW_MIN_DOTS ? "the start projects $n of its dots, fewer than a pose's $VIEW_MIN_DOTS" :
                 even !== nothing && min(count(live[k] .& even[k]), count(live[k] .& .!even[k])) < VIEW_MIN_DOTS ?
                     "the start projects fewer than $VIEW_MIN_DOTS of its dots of one parity (the held-out fits)" : nothing
        reason === nothing && continue
        live[k] .= false
        push!(out, (; photo = String(names[k]), reason))
    end
    return out
end

"""
    evaluate_grid_models(index_dir, g, s, out; W, H, models, detections = nothing) -> Vector

Every model of `models` (`(label, spec)` pairs: `evaluate_model_specs`) on the registered dots of `index_dir` (`read_registrations`):
the rotation-centre refine from the index stage's rotations (see the section comment) — the start `_fit_rotation_center`, then
Core's `LevenbergMarquardt` on Estimation's `CameraPosesProblem` of `RotationCenterViews` with the model's camera held, `s.optimization_strategy` at the index
stage's pooled noise scale (`pooled_index_scale`, `ScaleUpstream`) — reported in px and mrad, overall and in the centre and edge
regions, with a per-photograph table, the radial profile and the 8 × 6 map. Each model is fitted and scored on its own dots (those
it unprojects): fitted on those its start projects, scored on every one of a photograph it refines that its estimate projects
(`DotScores`) — a dot the start does not project is left out of the fit and re-tested at the estimate (owner 2026-11-15), counted
under `start.unprojected_at_start`; one the estimate does not project either is counted (`unprojected_at_estimate`). summary.tsv's
head-to-head is on the dots every model scores (`common_dots`, `dot_summary`), beside each model's coverage (`dot_coverage`) and
its score on all of its own dots (the `all_` columns; the eval record's `summary`, its `common` and `coverage`). A `schoeps:` model
must be T. Schöps's own camera (`schoeps_own_camera`), refused otherwise. Writes `out/eval_<label>.json`,
`out/vectors_<label>_<variant>.json`, `summary.tsv`, `summary.md`; returns the summaries. `s` is `evaluate_settings`'s: the refine's
strategy and iteration cap, and the held-out variant `motion_heldout` (the fits on one parity of the grid nodes scoring the other,
pooled). The start reads the dots' Förstner covariances (the index run's points records) when `detections` (the detect run's
`detections/` directory) is given; unit pixel noise without it (the rotation-centre refine whitens no dot). Every eval record holds
the start (`start`: c, d, its retcode and the conditioning) and under `fit` the refine's retcode, iterations, scale, its source and
the loss width. A photograph the start projects none of the dots of (held out: none of one parity) has no view in the refine: it is
refused for that model — `refused` in its eval record (photograph, reason) and a note in summary.md — and never looked up.
"""
function evaluate_grid_models(index_dir::AbstractString, g::ThorlabsGrid, s, out::AbstractString; W::Integer, H::Integer, models,
                              detections::Union{Nothing, AbstractString} = nothing)
    mkpath(out)
    holdout = s.holdout
    regs = read_registrations(index_dir, g)
    nreg = length(regs)
    mu = check_max_unassigned(get(s, :max_unassigned, 0), g)
    regs = filter(r -> _all_dots(r, g, mu), regs)                             # the labelling is pinned when every dot (or all but < a row) is assigned
    isempty(regs) && throw(ArgumentError("evaluate: no registered photograph of $index_dir has every dot of the grid assigned$(mu > 0 ? " but at most $mu" : "")"))
    @printf("%d of %d registered photographs have all %d dots assigned%s; the others are discarded\n", length(regs), nreg, length(g.nodes), mu > 0 ? " but at most $mu" : "")
    regs = canonical_registrations(regs, g)                                   # one target frame (the grid's symmetries and offsets undone)
    moved = [(r.name, r.offset) for r in regs if r.offset != (0, 0)]
    @printf("%d photographs re-offset by whole pitches onto the set's rotation centre%s\n", length(moved), isempty(moved) ? "" : ": " * join(["$n $o" for (n, o) in moved], ", "))
    loaded = []                                                               # a model whose record is not there yet (a refinement still running) is skipped, named
    for (label, spec) in models
        kind, rest = split(spec, ":"; limit = 2)
        kind == "schoeps" && schoeps_own_camera(rest)                         # his own camera, or refused (never skipped)
        try
            push!(loaded, (label, spec, load_ray_model(spec, W, H)...))
        catch err
            err isa Union{ArgumentError, SystemError, Base.IOError} || rethrow()
            @warn "evaluate: model $label skipped — $(sprint(showerror, err))"
        end
    end
    isempty(loaded) && throw(ArgumentError("evaluate: none of the $(length(models)) models could be loaded"))
    ndots = sum(length(r.xy) for r in regs)                                   # every registered dot: the coverages' denominator
    # each model's own dots, those it unprojects (its domain): a model is fitted and scored on its own, never on another model's domain
    own = [[BitVector([ray(p) !== nothing for p in r.xy]) for r in regs] for (_, _, ray, _, _) in loaded]
    @printf("%d photographs, %d registered dots, %d models; unprojected: %s\n", length(regs), ndots, length(loaded),
            join(["$(l[1]) $(sum(sum, o))" for (l, o) in zip(loaded, own)], ", "))
    ρ = reduce(vcat, [[norm(p - s.centre) / s.rmax for p in r.xy] for r in regs])
    centre_ = ρ .< s.centre_rho; edge_ = ρ .> s.edge_rho
    @printf("regions about (%.1f, %.1f) px, rmax %.1f px: centre (ρ < %.1f) %d dots, edge (ρ > %.1f) %d dots\n", s.centre..., s.rmax, s.centre_rho,
            count(centre_), s.edge_rho, count(edge_))
    # the set's one noise scale: the photographs' index scales (pixel MAD) pooled (Estimation's PooledScale), handed to every refine
    pooled = pooled_index_scale(regs)
    no_scale = [r.name for r in regs if !(isfinite(r.noise_scale) && r.dof > 0)]      # left out of the pool: recorded, never silent
    @printf("the index stage's pooled noise scale: %.4f px over %d photographs (%d dof); %d without a scale left out of the pool%s\n",
            VgtkCore.noise_scale(pooled), pooled.metadata.pooled, VgtkCore.dof(pooled), length(no_scale),
            isempty(no_scale) ? "" : ": " * join(no_scale, ", "))
    variant_names = holdout ? ("motion", "motion_heldout") : ("motion",)
    notes = String[]
    Rd = Dict(r.name => SMatrix{3, 3, Float64, 9}(r.R) for r in regs)
    scores = Dict(v => Dict{String, DotScores}() for v in variant_names)      # per variant and model: every dot it scores
    evaluated = []                                                            # per model: its record and its variants' summaries
    for (m, (label, spec, ray, mlabel, camera)) in enumerate(loaded)
        # the model's photographs: those with a dot it unprojects; `ids` each dot's index among its photograph's registered dots
        data = [(; r.name, xy = r.xy[own[m][k]], X = r.X[own[m][k]], ids = findall(own[m][k]), r.R, r.t,
                   Σ = detections === nothing ? nothing : r.Σ[own[m][k]], r.noise_scale, r.dof)   # the covariances whiten the start
                for (k, r) in enumerate(regs) if any(own[m][k])]
        isempty(data) && throw(ArgumentError("evaluate: $label unprojects the dots of no photograph"))
        N = length(data)
        names = [d.name for d in data]
        nvalid = sum(d -> length(d.xy), data)
        # the held-out variant's parity of each dot's node, (i + j) even or odd
        even = [[iseven(sum(round.(Int, x[1:2] ./ g.spacing))) for x in d.X] for d in data]
        rays = [[ray(p) for p in d.xy] for d in data]
        cstar, feet = foot_points(reduce(vcat, rays))
        O = let k0 = 0; [(n = length(d.xy); o = feet[k0 + 1:k0 + n]; k0 += n; o) for d in data] end
        offset_mm = _MM_PER_M .* norm.(reduce(vcat, O) .- Ref(cstar))
        u = [[r[2] for r in rk] for rk in rays]
        cm = VgtkCore.model(camera)
        # the model's local scale at each dot, rad / px (the mrad report): the ray direction's analytic Jacobian (Core's
        # backproject_with_jacobian), the photograph's median where the camera does not back-project the dot
        scale = [[_local_scale(camera, d.xy[i]) for i in eachindex(d.xy)] for (k, d) in enumerate(data)]
        for sc in scale
            ok = filter(isfinite, sc); isempty(ok) && throw(ArgumentError("evaluate: $label unprojects no neighbourhood of a photograph's dots"))
            sc[.!isfinite.(sc)] .= median(ok)
        end
        # each dot's back-projection Jacobian (the start's pixel whitening): the direction Jacobian, tangent to its bearing; where the
        # camera does not back-project the dot, the photograph's median local scale, isotropic
        jac = [[(J = _direction_jacobian(camera, d.xy[i]);
                 J === nothing ? (B = VgtkCore.s2_tangent_basis(u[k][i]); scale[k][i] * SMatrix{3, 2, Float64, 6}(B[1]..., B[2]...)) :
                                 SMatrix{3, 2, Float64, 6}(J)) for i in eachindex(d.xy)] for (k, d) in enumerate(data)]
        unit = SMatrix{2, 2, Float64, 4}(1.0, 0.0, 0.0, 1.0)
        dots = Dict(d.name => VgtkCore.PointSet(u[k]; Σ = d.Σ === nothing ? fill(unit, length(d.xy)) : d.Σ, X = d.X, o = O[k], M = jac[k],
                                                P = [d.R * d.X[i] + d.t - O[k][i] for i in eachindex(d.xy)]) for (k, d) in enumerate(data))
        # the rotations of the model's own photographs (`data`: those with a dot it unprojects) — a model whose camera covers fewer
        # photographs (a fisheye start with a narrower valid domain) is fitted on those, its coverage reporting the rest; the index's
        # rotations of the others are not this model's
        start = _fit_rotation_center(Dict(n => Rd[n] for n in names), dots; image_size = (W, H))
        successful_retcode(start) || throw(ArgumentError("evaluate: $label: the rotation-centre start's system is degenerate (singular values $(start.metadata.singular_values))"))
        c0, d0 = start.u
        # each photograph's pose at the start: Estimation's rotation-centre view pose, (R, d − R c)
        P0 = [VgtkEst.rotation_center_pose(VgtkCore.Rotations.RotMatrix{3, Float64}(Rd[d.name]), c0, d0) for d in data]
        # the dots the camera projects at the start: the refine's dots (it refuses one that does not); a dot the start does not project
        # is left out of the fit only — its residual has no start — and re-tested at the estimate (owner 2026-11-15)
        live = [[all(isfinite, VgtkCore.reprojection_error(cm, P0[k], d.X[i], d.xy[i])) for i in eachindex(d.xy)] for (k, d) in enumerate(data)]
        unprojected_at_start = sum(k -> count(!, k), live)                      # the dots the start does not project: not in the fit
        refusals = start_refusals!(live, names, holdout ? even : nothing)          # no view in the refine: refused, never looked up
        kept = [k for k in 1:N if any(live[k])]
        # the dots scored, at the estimate: held out, every one of a photograph the refine has a view of (a refused one has no estimate
        # pose; the start's misses included, out of every fit); in-sample, the dots its fit saw (the start's misses are out of sample)
        scored = [k in kept ? collect(eachindex(d.xy)) : Int[] for (k, d) in enumerate(data)]
        fitted = [k in kept ? findall(live[k]) : Int[] for k in 1:N]
        foreach(r -> @printf("%-28s %s refused: %s\n", label, r.photo, r.reason), refusals)
        isempty(kept) && throw(ArgumentError("evaluate: $label: the start projects the dots of no photograph"))
        tres = [norm(d.t - (d0 - d.R * c0)) for d in data]          # how far each index pose is from the one rotation centre
        startpx = reduce(vcat, [[norm(VgtkCore.reprojection_error(cm, P0[k], d.X[i], d.xy[i])) for i in findall(live[k])] for (k, d) in enumerate(data)])
        sm = start.metadata
        startrec = (; method = "fit_rotation_center", c_m = collect(c0), d_m = collect(d0), retcode = start.retcode, stop_rule = start.stop_rule,
                      singular_values = sm.singular_values, singular_values_unnormalized = sm.singular_values_unnormalized,
                      condition = sm.singular_values[1] / sm.singular_values[end], whitened_rms = start.rms,
                      residual_px = (; median = median(startpx), p95 = quantile(startpx, 0.95)), unprojected_at_start)
        @printf("%-28s start: c %s mm, d %s mm (%s, cond %.1f), dots median %.3f p95 %.3f px\n", label, @sprintf("%.1f/%.1f/%.1f", (_MM_PER_M .* c0)...),
                @sprintf("%.1f/%.1f/%.1f", (_MM_PER_M .* d0)...), startrec.retcode, startrec.condition, startrec.residual_px.median, startrec.residual_px.p95)
        held = VgtkEst.held_parameters(camera)                 # every camera parameter held (Estimation)
        # one rotation-centre refine on the dots of `mask` (per photograph): Core's LevenbergMarquardt on Estimation's CameraPosesProblem of RotationCenterViews
        # from the start, the camera held, `s.optimization_strategy` at the pooled index scale
        function _refine_on(mask)
            sel = [findall(mask[k] .& live[k]) for k in 1:N]
            views = Dict(names[k] => [data[k].X[i] => data[k].xy[i] for i in sel[k]] for k in 1:N if !isempty(sel[k]))
            t0 = time()
            f = CameraPosesProblem(views; fixed = held)
            st = VgtkCore.default_stops(f)                       # the problem's stops of record, the cap the evaluation's
            r = solve(RefineProblem(f, (; camera, views = RotationCenterViews(Dict(n => Rd[n] for n in keys(views)), c0, d0))),
                      LevenbergMarquardt(; optimization_strategy = s.optimization_strategy); scale = pooled,
                      stops = VgtkCore.LmStops(; max_iter = s.iterations, ftol = st.ftol, xtol = st.xtol))
            v = r.u.views
            pose(k) = VgtkEst.rotation_center_pose(VgtkCore.Rotations.RotMatrix{3, Float64}(SMatrix{3, 3, Float64, 9}(v.rotations[names[k]])), v.center, v.offset)
            return (; r, pose, c = v.center, d = v.offset, seconds = time() - t0)
        end
        fitof(f) = (; retcode = f.r.retcode, stop_rule = f.r.stop_rule, converged = successful_retcode(f.r), iterations = f.r.stats.nsteps,
                      scale_px = VgtkCore.noise_scale(f.r), scale_source = VgtkCore.scale_source(f.r), loss_width_px = f.r.metadata.loss_width,
                      downweighted = count(<(0.5), VgtkCore.weights(f.r)), dots = f.r.n, rms_px = f.r.rms, ν = VgtkCore.dof(f.r), seconds = f.seconds,
                      # the dots the refine made no row of (on bearings, every camera: no bearing, or a Jacobian that cannot whiten it),
                      # counted and named: (photograph, index into its refined dots)
                      n_without_bearing = f.r.metadata.n_without_bearing, without_bearing = [collect(p) for p in f.r.metadata.without_bearing])
        extra(f) = (; c_m = collect(f.c), d_m = collect(f.d), iterations = f.r.stats.nsteps, start_translation_residual_mm = _MM_PER_M .* tres[kept],   # as distance_m: the photographs refined
                      distance_m = [norm(f.pose(k)(mean(data[k].X))) for k in kept])      # the photographs refined (not the refused)
        variants = Dict{String, Any}()
        fitrec = Dict{String, Any}()
        fi = _refine_on([trues(length(d.xy)) for d in data])
        variants["motion"] = (; pose_at = (k, i) -> fi.pose(k), dots = fitted, extra = extra(fi), retcode = variant_retcode((fi,)))
        fitrec["in_sample"] = fitof(fi)
        if holdout
            # held out: each fit on one parity of the grid nodes (`even`), its poses scoring the dots of the other parity
            fe = _refine_on(even); fo = _refine_on([.!e for e in even])
            variants["motion_heldout"] = (; pose_at = (k, i) -> (even[k][i] ? fo : fe).pose(k), dots = scored,
                                            extra = (; extra(fe)..., iterations = fe.r.stats.nsteps + fo.r.stats.nsteps, c_m_odd = collect(fo.c), d_m_odd = collect(fo.d)),
                                            retcode = variant_retcode((fe, fo)))
            fitrec["held_out"] = (; even = (; fitof(fe)..., dots = sum(sum, even)), odd = (; fitof(fo)..., dots = sum(e -> count(!, e), even)))
        end
        rec = Dict{String, Any}("label" => label, "spec" => spec, "model" => mlabel, "photos" => length(kept), "photos_registered" => nreg, "refused" => refusals,
                                "reoffset" => [(; photo = n, offset = collect(o)) for (n, o) in moved],
                                "dots" => nvalid, "dots_registered" => ndots, "image_size" => [W, H],
                                "centre" => collect(s.centre), "rmax" => s.rmax, "c_star_m" => collect(cstar),
                                "ray_offset_mm" => (; median = median(offset_mm), max = maximum(offset_mm)), "start" => startrec,
                                "pooled_scale" => (; scale_px = VgtkCore.noise_scale(pooled), photos = pooled.metadata.pooled, dof = VgtkCore.dof(pooled),
                                                   left_out = length(no_scale), without_scale = no_scale))
        rec["fit"] = merge(Dict{String, Any}("optimization_strategy" => s.optimization_strategy_table, "iterations" => s.iterations,
                                             "holdout" => holdout, "held_camera" => collect(String.(held))), fitrec)
        sums = Dict{String, Any}()
        for variant in variant_names
            v = variants[variant]
            # every scored dot's reprojection error under its photograph's estimate (held out: the fit that did not see it, the dots the
            # start did not project too; in-sample: the dots its fit saw), prediction − detection; a dot the estimate does not project (its error not finite: off the lens's
            # domain, e.g. a SplineLens camera's forward chart) is counted (`unprojected_at_estimate`), not scored
            Eall = [[VgtkCore.reprojection_error(cm, v.pose_at(k, i), data[k].X[i], data[k].xy[i]) for i in v.dots[k]] for k in 1:N]
            fin = [[all(isfinite, e) for e in ek] for ek in Eall]
            ix = [v.dots[k][fin[k]] for k in 1:N]
            unprojected_at_estimate = sum(k -> count(!, fin[k]), 1:N)
            E = [Eall[k][fin[k]] for k in 1:N]
            px = [norm.(e) for e in E]; allpx = reduce(vcat, px)
            xs = reduce(vcat, [[d.xy[i][1] for i in ix[k]] for (k, d) in enumerate(data)])
            ys = reduce(vcat, [[d.xy[i][2] for i in ix[k]] for (k, d) in enumerate(data)])
            # the residual vectors of every dot the model scores (its coverage: the figures' maps show where it reaches)
            VgtkIO.write_record(joinpath(out, "vectors_$(label)_$variant.json"),
                (; schema = EXPERIMENT_RECORD_SCHEMAS.residual_vectors, x = xs, y = ys, dx = reduce(vcat, [[e[1] for e in ek] for ek in E]), dy = reduce(vcat, [[e[2] for e in ek] for ek in E])))
            ρl = reduce(vcat, [[norm(d.xy[i] - s.centre) / s.rmax for i in ix[k]] for (k, d) in enumerate(data)])
            # every dot the model scores, by photograph and registered dot: its error (px), in mrad, and its ρ
            scores[variant][label] = DotScores((data[k].name, data[k].ids[i]) => (px[k][j], 1e3 * px[k][j] * scale[k][i], norm(data[k].xy[i] - s.centre) / s.rmax)
                                               for k in 1:N for (j, i) in enumerate(ix[k]))
            edges = range(0, 1; length = s.profile_bins + 1)
            profile = [(b = (ρl .>= edges[j]) .& (ρl .< edges[j + 1]); (; r_lo = edges[j] * s.rmax, r_hi = edges[j + 1] * s.rmax, n = count(b),
                        median_px = count(b) > 0 ? median(allpx[b]) : nothing, mean_px = count(b) > 0 ? mean(allpx[b]) : nothing)) for j in 1:s.profile_bins]   # empty bin: null
            map_ = [(b = (xs .>= (i - 1) * W / 8) .& (xs .< i * W / 8) .& (ys .>= (j - 1) * H / 6) .& (ys .< j * H / 6);
                     (; i, j, n = count(b), median_px = count(b) > 0 ? median(allpx[b]) : nothing)) for i in 1:8 for j in 1:6]
            per_photo = [(; photo = data[k].name, residual_summary(px[k])..., distance_m = v.extra.distance_m[j]) for (j, k) in enumerate(kept)]
            # the model on all of its own dots (the head-to-head on the common dots follows, once every model is scored)
            summary = (; variant, dot_summary(scores[variant][label], keys(scores[variant][label]), s)..., photos = length(kept), unprojected_at_estimate, v.extra...)
            rec[variant] = (; summary, per_photo, profile, map = map_)
            sums[variant] = (; summary, retcode = v.retcode)
            @printf("%-28s %-14s %3d photos %6d dots (all its own): median %.4f mean %.4f p95 %.4f px (%.4f mrad)%s\n", label, variant, length(kept), length(allpx),
                    summary.median_px, summary.mean_px, summary.p95_px, summary.median_mrad,
                    @sprintf("; %d its, %s%s", v.extra.iterations, fitrec["in_sample"].retcode,
                             unprojected_at_estimate > 0 ? "; $unprojected_at_estimate dots not projected at the estimate" : "")); flush(stdout)
        end
        push!(notes, @sprintf("%s: the start's translation residual against the index poses (one rotation centre for the set) median %.1f mm, p90 %.1f mm, max %.1f mm.",
                              label, _MM_PER_M * median(tres[kept]), _MM_PER_M * quantile(tres[kept], 0.9), _MM_PER_M * maximum(tres[kept])))
        isempty(refusals) || push!(notes, "$label: $(length(refusals)) photographs refused — " * join(["$(r.photo) ($(r.reason))" for r in refusals], ", ") * ".")
        maximum(offset_mm) > 0 && @printf("%-28s rays: foot points from c★ median %.3f mm, max %.3f mm\n", label, median(offset_mm), maximum(offset_mm))
        push!(evaluated, (; label, rec, sums))
    end
    # the head-to-head: every model on the dots every model scores (owner 2026-11-15), its coverage beside it
    common = Dict(v => common_dots(values(scores[v])) for v in variant_names)
    foreach(v -> isempty(common[v]) && throw(ArgumentError("evaluate: the $(length(loaded)) models score no dot in common ($v)")), variant_names)
    rows = Tuple[]
    md = String["| model | variant | photos | dots | coverage | median px | mean px | p95 px | rms px | median mrad | centre median | centre mean | edge median | edge mean | all dots: photos | median px | mean px | p95 px | rms px | c mm | d mm |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    summaries = []
    mdcell(x) = x === nothing ? "—" : @sprintf("%.4f", x)
    for (; label, rec, sums) in evaluated, variant in variant_names
        sc = scores[variant][label]
        head = dot_summary(sc, common[variant], s); all_ = sums[variant].summary; cov = dot_coverage(sc, ndots)
        rec[variant] = (; rec[variant]..., common = head, coverage = cov)
        cmm = @sprintf("%.1f/%.1f/%.1f", (_MM_PER_M .* all_.c_m)...); dmm = @sprintf("%.1f/%.1f/%.1f", (_MM_PER_M .* all_.d_m)...)
        push!(rows, (label, variant, head.photos, head.n, cov.coverage_dots, cov.coverage_fraction, head.median_px, head.mean_px, head.p95_px, head.rms_px,
                     head.median_mrad, tsv_cell(head.centre_median_px), tsv_cell(head.centre_mean_px), tsv_cell(head.edge_median_px), tsv_cell(head.edge_mean_px),
                     all_.photos, all_.median_px, all_.mean_px, all_.p95_px, all_.rms_px, cmm, dmm, sums[variant].retcode))
        push!(md, @sprintf("| %s | %s | %d | %d | %d (%.1f %%) | %.4f | %.4f | %.4f | %.4f | %.4f | %s | %s | %s | %s | %d | %.4f | %.4f | %.4f | %.4f | %s | %s |", label, variant,
                           head.photos, head.n, cov.coverage_dots, 100 * cov.coverage_fraction, head.median_px, head.mean_px, head.p95_px, head.rms_px, head.median_mrad,
                           mdcell(head.centre_median_px), mdcell(head.centre_mean_px), mdcell(head.edge_median_px), mdcell(head.edge_mean_px),
                           all_.photos, all_.median_px, all_.mean_px, all_.p95_px, all_.rms_px, cmm, dmm))
        @printf("%-28s %-14s head-to-head %6d common dots: median %.4f (centre %s, edge %s) mean %.4f p95 %.4f rms %.4f px; coverage %d (%.1f %%)\n", label, variant,
                head.n, head.median_px, mdcell(head.centre_median_px), mdcell(head.edge_median_px), head.mean_px, head.p95_px, head.rms_px,
                cov.coverage_dots, 100 * cov.coverage_fraction)
        push!(summaries, (; label, all_..., common = head, cov...))
    end
    # the paired photograph bootstrap of every comparison of [evaluate] whose two models were scored, each variant (no new fit: the
    # per-dot errors above, on the head-to-head's common dots)
    have = Set(e.label for e in evaluated)
    cmp = [c for c in get(s, :comparisons, NamedTuple[]) if c.model in have && c.reference in have]
    for c in get(s, :comparisons, NamedTuple[])
        (c.model in have && c.reference in have) || push!(notes, "comparison $(c.model) − $(c.reference): not bootstrapped (a model was not loaded).")
    end
    paired = isempty(cmp) ? NamedTuple[] : paired_differences(scores, common, cmp, variant_names, s; s.bootstrap...)
    for (; label, rec) in evaluated
        rec["paired"] = [p for p in paired if p.model == label]
    end
    isempty(cmp) || write_tsv(joinpath(out, PAIRED_FILE), PAIRED_COLUMNS, (Tuple(p[Symbol(k)] for k in PAIRED_COLUMNS) for p in paired))
    for p in paired
        p.region == "all" && p.stat == "median_px" &&
            @printf("%-28s − %-28s %-14s median %+.4f px [%+.4f, %+.4f]%s\n", p.model, p.reference, p.variant, p.difference, p.lo, p.hi, p.excludes_zero ? " *" : "")
    end
    for (; label, rec) in evaluated
        VgtkIO.write_record(joinpath(out, "eval_$label.json"), merge(Dict{String, Any}("schema" => EXPERIMENT_RECORD_SCHEMAS.chrome_evaluation), rec))
    end
    push!(md, "", @sprintf("%d of %d registered photographs have all %d dots assigned%s; the others are discarded (their labelling may be shifted by whole spacings).", length(regs), nreg, length(g.nodes), mu > 0 ? " but at most $mu" : ""))
    push!(md, "", "The head-to-head (photos, dots and the statistics through edge mean) is on the dots every model above scores: " *
                  join(["$v $(length(common[v]))" for v in variant_names], ", ") * " of the $ndots registered dots. Each model's coverage is the dots it scores " *
                  "(it unprojects them and its estimate projects them; in-sample, its fit saw them too: its start projects them) and their fraction of the registered dots; \"all dots\" scores it on every one of them.")
    push!(md, "", @sprintf("Regions about (%.1f, %.1f) px with rmax %.1f px: centre ρ < %.1f (%d dots), edge ρ > %.1f (%d dots).", s.centre..., s.rmax, s.centre_rho, count(centre_), s.edge_rho, count(edge_)))
    push!(md, "", @sprintf("The rotation-centre refine ran at the index stage's pooled noise scale %.4f px (ScaleUpstream), pooled over %d photographs; %d whose index scorer recorded no scale were left out of the pool%s.",
                           VgtkCore.noise_scale(pooled), pooled.metadata.pooled, length(no_scale), isempty(no_scale) ? "" : " (" * join(no_scale, ", ") * ")"))
    append!(md, ["", notes...])
    write_tsv(joinpath(out, "summary.tsv"), EVALUATE_SUMMARY_COLUMNS, rows)          # VisualGeometryToolkitEval's table
    write(joinpath(out, "summary.md"), join(md, '\n') * '\n')
    return summaries
end

"The dots a model scores, `(photograph, registered dot) => (error px, error mrad, ρ)` (`evaluate_grid_models`)."
const DotScores = Dict{Tuple{String, Int}, NTuple{3, Float64}}

# ---- the paired photograph bootstrap of the differences between two models (owner 2026-11-15) ---------------------------------------
"The file of an evaluate run's paired differences (`paired_differences`, VisualGeometryToolkitEval's table), beside summary.tsv."
const PAIRED_FILE = "paired.tsv"
"""The columns of paired.tsv: per comparison (`model` − `reference`), variant, region and statistic, the point difference on the common
dots, its interval `[lo, hi]` (`level` = 1 − alpha), whether it excludes 0, the photographs and dots resampled, the draws and the seed."""
const PAIRED_COLUMNS = ("model", "reference", "variant", "region", "stat", "difference", "lo", "hi", "level", "excludes_zero", "photos", "dots",
                        "resamples", "seed")
"The regions of a paired difference: every common dot, the centre (ρ < centre_rho) and the edge out to the rim (ρ > edge_rho)."
const PAIRED_REGIONS = ("all", "centre", "edge")
"The statistics of a paired difference (Core's residual summary's: `residual_summary`)."
const PAIRED_STATS = ("median_px", "mean_px", "p95_px", "rms_px")

"""
    paired_photos(model, reference, dots, inregion) -> Vector{NamedTuple}

The common dots `dots` of two models' `DotScores` grouped by photograph — the bootstrap's unit (the dots of a photograph share its
pose, blur and lighting) — each `(; photo, model, reference)`: the two models' errors (px) of its dots with `inregion(ρ)`, in dot order.
"""
function paired_photos(model::DotScores, reference::DotScores, dots, inregion)
    ks = sort!(collect(dots))
    out = NamedTuple{(:photo, :model, :reference), Tuple{String, Vector{Float64}, Vector{Float64}}}[]
    for k in ks
        inregion(model[k][3]) || continue
        (isempty(out) || out[end].photo != k[1]) && push!(out, (; photo = k[1], model = Float64[], reference = Float64[]))
        push!(out[end].model, model[k][1]); push!(out[end].reference, reference[k][1])
    end
    return out
end

"""
    paired_difference(photos, stats; resamples, rng, alpha) -> Vector{(; difference, lo, hi)}

The differences `stat(model) − stat(reference)` of every `stat` of `stats` (fields of `residual_summary`) of `photos`
(`paired_photos`) pooled, and their `1 − alpha` intervals by VisualGeometryToolkitEval's `bootstrap_ci` (Stats: the BCa interval,
`resamples` draws) over the PHOTOGRAPHS: each draw takes photographs with replacement and computes the statistics of both models on
the same drawn photographs' same dots — a paired difference, never a dot-level resample. One replicate set for every statistic (RC
0.11.5, the 0.11.4 review's item 7c): each statistic's `bootstrap_ci` is seeded by the same `rng` state (a `copy`), so the
statistics share their draws (and the BCa jackknife's), and a replicate's two residual summaries are computed once for all of them
(`summaries`, by the replicate's photographs in draw order); before, each statistic drew its own replicates and summarized them again.
"""
function paired_difference(photos, stats; resamples::Integer, rng::AbstractRNG, alpha::Real)
    cat_(v) = isempty(v) ? Float64[] : reduce(vcat, v)                     # reduce(vcat, ::Vector) without init: the one-copy fast path
    memo = Dict{Vector{String}, Vector{Float64}}()                        # a replicate's differences, every statistic (its photographs, in draw order)
    function summaries(ps)
        get!(memo, [p.photo for p in ps]) do
            a = cat_([p.model for p in ps]); b = cat_([p.reference for p in ps])
            isempty(a) && return fill(NaN, length(stats))
            sa, sb = residual_summary(a), residual_summary(b)
            return [Float64(sa[st]) - Float64(sb[st]) for st in stats]
        end
    end
    return map(eachindex(stats)) do i
        lo, hi = bootstrap_ci(ps -> summaries(ps)[i], copy(rng), photos; n_resamples = Int(resamples), alpha = Float64(alpha))
        (; difference = summaries(photos)[i], lo, hi)
    end
end

"""
    paired_differences(scores, common, comparisons, variants, s; resamples, seed, alpha) -> Vector{NamedTuple}

The paired photograph bootstrap of every comparison `(; model, reference)` (`[evaluate] comparisons`) in every variant of `variants`
(the held-out headline and the in-sample fit), on the variant's common dots `common[v]` (the head-to-head's), per region
(`PAIRED_REGIONS`, `s.centre_rho` / `s.edge_rho`) and statistic (`PAIRED_STATS`): `paired_difference`, the statistics of a
comparison, variant and region on ONE replicate set, its stream `StableRNG(seed + g)`, `g` its index in the comparisons × variants ×
regions (reproducible, the run's key stable; on one Julia version: Eval's `bootstrap_ci`). (RC ≤ 0.11.4: each interval its own
`StableRNG(seed + j)`, `j` over the statistics too — every interval the same BCa interval of the same draws' count, its Monte Carlo
draws now shared by the four statistics.) A row per difference, `PAIRED_COLUMNS`. No fit: the per-dot errors of `scores` (`DotScores`).
"""
function paired_differences(scores, common, comparisons, variants, s; resamples::Integer, seed::Integer, alpha::Real)
    region(r) = r == "all" ? (ρ -> true) : r == "centre" ? (ρ -> ρ < s.centre_rho) : (ρ -> ρ > s.edge_rho)
    groups = [(c, v, r) for c in comparisons for v in variants for r in PAIRED_REGIONS]
    rows = Vector{Any}(undef, length(groups))
    Threads.@threads for g in eachindex(groups)                               # each its own seeded stream: the rows do not depend on the threads
        c, v, r = groups[g]
        ps = paired_photos(scores[v][c.model], scores[v][c.reference], common[v], region(r))
        ds = paired_difference(ps, Symbol.(PAIRED_STATS); resamples, rng = StableRNG(Int(seed) + g), alpha)
        rows[g] = [(; model = c.model, reference = c.reference, variant = v, region = r, stat = st, d.difference, d.lo, d.hi, level = 1 - alpha,
                    excludes_zero = isfinite(d.lo) && isfinite(d.hi) && (d.lo > 0 || d.hi < 0), photos = length(ps), dots = sum(p -> length(p.model), ps; init = 0),
                    resamples = Int(resamples), seed = Int(seed)) for (st, d) in zip(PAIRED_STATS, ds)]
    end
    return [row for g in eachindex(groups) for row in rows[g]]
end

"""
    common_dots(scores) -> Vector{Tuple{String, Int}}

The dots every model scores (`scores`: each model's `DotScores`), sorted: the head-to-head's observations, the intersection of the
dots each model can evaluate, so every model's statistics are on the same observations (owner 2026-11-15).
"""
common_dots(scores) = sort!(collect(reduce(intersect!, (Set(keys(sc)) for sc in scores))))

"""
    dot_summary(score, dots, s) -> NamedTuple

The statistics of a model's `score` (`DotScores`) on the dots `dots` (each one it scores): Core's residual summary (`residual_summary`:
`n`, `median_px`, `mean_px`, `rms_px`, `p95_px`, `max_px`), `photos` (the photographs holding them), `median_mrad`, and the median and
mean in the centre (ρ < `s.centre_rho`) and edge (ρ > `s.edge_rho`) regions, `nothing` for a region without a dot. The dots are
read in sorted order (the statistics' sums are reproducible).
"""
function dot_summary(score::DotScores, dots, s)
    ks = sort!(collect(dots))
    px = [score[k][1] for k in ks]; mrad = [score[k][2] for k in ks]; ρ = [score[k][3] for k in ks]
    reg(f, b) = any(b) ? f(px[b]) : nothing
    cen, edg = ρ .< s.centre_rho, ρ .> s.edge_rho
    return (; residual_summary(px)..., photos = length(unique(first.(ks))), median_mrad = isempty(mrad) ? nothing : median(mrad),
            centre_median_px = reg(median, cen), centre_mean_px = reg(mean, cen), edge_median_px = reg(median, edg), edge_mean_px = reg(mean, edg))
end

"""
    dot_coverage(score, ndots) -> (; coverage_dots, coverage_fraction)

A model's coverage: the dots it scores (`score`, `DotScores`) and their fraction of every registered dot `ndots`.
"""
dot_coverage(score::DotScores, ndots::Integer) = (; coverage_dots = length(score), coverage_fraction = length(score) / ndots)

