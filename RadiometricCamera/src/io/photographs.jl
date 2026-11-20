# ---------------------------------------------------------------------------
# Data loading. Every file is read by the stack: the images by VisualGeometryToolkitFeatures (`load_linear_image`: the samples as
# stored, an sRGB decode for photographs, a box downsample in linear space), a calibration session — the geometric central model and
# the camera poses a radiometric calibration starts from — by VisualGeometryToolkitIO from its records (CaliBlobs'
# `save_calibration`: `camera.json`, `targets_<c>.json`, `trajectory_<c>.json`, `target_poses.json`, `textures/`).
# ---------------------------------------------------------------------------


"""
    thin_lens_focus_distance(lens, sensor, r₀; d_s = nothing) -> d  (metres; Inf when the rays are parallel)

The thin-lens construction, in the meridional plane (pinhole 𝔭 at the origin, aperture in z = 0, sensor at
`z = −d_s`): from the sensor point under the aperture border (height r₀), intersect the chief ray through 𝔭
with the ray that runs parallel to the axis to the aperture border and from there through the focal point
(`0, f_m = f_px · pitch`, the camera's focal length in pixels times the pixel pitch of VisualGeometryToolkitCore's `sensor`); with the
intersection x, `d = ‖x − 𝔭‖ cos α` (α the chief ray's incidence) — the
thin-lens conjugate `d = f_m·d_s/(d_s − f_m)`. The image distance `d_s` (metres) is an input: the default (`nothing`), the pinhole
model's sensor distance (`= f_m`), makes the two rays exactly parallel → d = Inf (focus at infinity, t̂ ≡ 0);
any image distance `d_s > f_m` makes them intersect at a finite d.
"""
function thin_lens_focus_distance(lens, sensor::VgtkCore.Sensor, r₀::Real; d_s::Union{Nothing, Real} = nothing)
    f_m = VgtkCore.calibration_matrix(VgtkCore.model(lens))[1, 1] * _pixel_pitch(sensor); h = Float64(r₀)
    d_s = d_s === nothing ? f_m : Float64(d_s)
    # ray 1: (h, −d_s) → (0, 0): direction (−h, d_s); ray 2: (h, 0) → (0, f_m): direction (−h, f_m)
    # intersection along z: z* = f_m d_s / (d_s − f_m)
    den = d_s - f_m
    abs(den) <= 1e-9 * f_m && return Inf
    z = f_m * d_s / den
    x = SVector(-h * z / d_s, z)                        # the point on ray 1 at height −h·z/d_s (thin-lens conjugate)
    cosα = d_s / hypot(h, d_s)
    d = hypot(x[1], x[2]) * cosα
    return z > 0 ? d : Inf                              # a virtual (behind-lens) intersection also means no finite focus
end

"""The pixel pitch of a Core `Sensor` (the length of one pixel along the rows), in metres."""
_pixel_pitch(s::VgtkCore.Sensor) = VgtkCore.ustrip(VgtkCore.m, VgtkCore.px * s.pitch.width)

"""
    load_session(records, image_dir; component = 1, lens = :camera, grid_size = (65, 50), focus_init = :thin_lens,
                 image_distance = nothing, downsample = 8, texture_downsample = 8, sensor_width = 7mm, r₀ = 1.7e-3, linearize = true,
                 focus_grid = 5, load_targets = true, raw_dir = nothing, lens_identity = nothing)

Scene, camera parts and initial poses of a calibration session stored as VisualGeometryToolkitIO records (`records`: CaliBlobs'
`save_calibration`, or its one-time migration of a BlobBoards session export, `scripts/migrate_rc_session.jl`):

* the camera: `load(BoundedCamera, camera.json)` — K in COLMAP pixels, the image extent, e.g. BlobBoards' division camera — bounded
  by its `Sensor` (`sensor_width` over the record's extent) and moved to the scene's pixels when the photographs are box-averaged by
  `downsample` (K re-expressed under the box downsample: Core's `Cameras.reframed_calibration` by the similarity x ↦ x / f): `lens = :camera` that pixel camera, `:grid` the generic grid model of
  `grid_size` sampled from it (`grid_model`);
* the board: the target of component `component`, `load(Rig, targets_<component>.json)` — Core's planar target with its Canvas and
  the CanvasTexture of the printed image, whose file (`textures/<id>.png`, beside the records) is read by Features' `load_linear_image`
  and box-averaged by `texture_downsample` — as the `PrintedTarget` of the scene, whose world frame is the target's (metres);
* the poses: the frames of `trajectory_<component>.json` (`load(Trajectory, …)`: target → camera per frame, the frames' image names),
  each photograph's camera → world pose (`poses`) its inverse (lengths in metres through FlexUnits);
* the focus surface: `focus_init = :board_center` (the board centre is in focus: its mean axial distance `d_c`, the lens equation's image
  distance `d_s = f·d_c/(d_c − f)`, the start plane `t̂ᵢ = cos αᵢ/d_c`), `:thin_lens` (`thin_lens_focus_distance`; t̂ ≡ 0 when it is
  Inf), `:board` (the mean principal-ray distance to the board plane) or a number (metres); `image_distance` overrides `d_s`
  (`nothing`: from `focus_init`, `:board` the conjugate of the board distance, or a number);
* the targets: the photographs `image_dir/<image>` in frame order (sRGB-decoded when `linearize`, box-averaged by `downsample`); with
  `load_targets = false` only the first is read, for the scene size;
* the capture's camera settings (`settings`): with `raw_dir`, each frame's RAW file (`raw_file`: the image's stem, a RAW extension)
  read by Features' `load_raw_metadata` (the EXIF alone) into Features' `extract_capture_settings` — Core's `CaptureSettings`, frame
  index → exposure and lens settings, with the body and the lens (`lens_identity` where the caller states it: an adapted lens
  EXIF misnames); `nothing` without `raw_dir`. Reading the RAW files is Features' decoder (its LibRaw extension, loaded with this package).
"""
function load_session(records::AbstractString, image_dir::AbstractString; component::Integer = 1, lens::Symbol = :camera,
                      grid_size::Tuple{Int, Int} = (65, 50), focus_init = :thin_lens, image_distance::Union{Nothing, Symbol, Real} = nothing,
                      downsample::Integer = 8, texture_downsample::Integer = 8, sensor_width = 7VgtkCore.mm, r₀::Real = 1.7e-3, linearize::Bool = true,
                      focus_grid::Integer = 5, load_targets::Bool = true, raw_dir::Union{Nothing, AbstractString} = nothing,
                      lens_identity::Union{Nothing, VgtkCore.DeviceIdentity} = nothing)
    bc = VgtkIO.load(VgtkCore.BoundedCamera, joinpath(records, "camera.json"))
    W₀, H₀ = VgtkCore.extent(bc).width, VgtkCore.extent(bc).height
    sensor = VgtkCore.Sensor(VgtkCore.Size2(width = W₀, height = H₀), sensor_width / W₀)   # the camera's sensor: its width over the record's extent
    lens === :camera || lens === :grid || throw(ArgumentError("lens must be :camera or :grid"))
    # the board: the component's target, its canvas texture's image beside the records
    rig = VgtkIO.load(VgtkCore.Rig, joinpath(records, "targets_$(component).json"))
    length(rig.members) == 1 || @warn "component $component holds $(length(rig.members)) targets; the scene holds one, the first by id" ids = sort!(collect(keys(rig.members)))
    id = first(sort!(collect(keys(rig.members))))
    pt = rig.members[id]
    tex_rec = VgtkCore.canvas(VgtkCore.model(pt)).texture
    tex_rec === nothing && throw(ArgumentError("target $id of $records carries no canvas texture (the printed image)"))
    tex = VgtkFeatures.load_linear_image(joinpath(records, tex_rec.path); downsample = texture_downsample)
    target = VgtkCore.ustrip(VgtkCore.m, pt)                              # the posed target in plain metres (the world = the rig's frame)
    # the poses: target → camera per frame; the camera → world pose is its inverse
    traj = VgtkIO.load(VgtkCore.Trajectory, joinpath(records, "trajectory_$(component).json"))
    frames = sort!(collect(keys(traj.poses)))
    names = [traj.images[t] for t in frames]
    cam_poses = [_pose_eltype(Float64, inv(VgtkCore.ustrip(VgtkCore.m, traj.poses[t]))) for t in frames]
    board = PrintedTarget(target, tex; facing = cam_poses[1].t)
    photo(n) = VgtkFeatures.load_linear_image(joinpath(image_dir, n); srgb = linearize, downsample)
    targets = load_targets ? [photo(n) for n in names] : Array{Float64, 3}[]
    t1 = load_targets ? targets[1] : photo(names[1])
    H, W = size(t1, 1), size(t1, 2); t1 = nothing
    scene = Scene(board, W, H)
    # the record's camera on the scene's pixels: Features' box downsample by f averages the f × f block of raw pixels j f + 1 … (j + 1) f
    # (the remainder rows / columns dropped), so a COLMAP coordinate (pixel corners on the integers) is x / f, corners to corners —
    # the similarity of scale 1/f, no translation — and K is re-expressed in that frame by Core's `reframed_calibration`
    (W, H) == (W₀ ÷ downsample, H₀ ÷ downsample) || error("load_session: the photographs are $W × $H, the record's $W₀ × $H₀ downsampled by $downsample is $(W₀ ÷ downsample) × $(H₀ ÷ downsample)")
    m0 = VgtkCore.model(bc)
    Kd = VgtkCore.Cameras.reframed_calibration(VgtkCore.SimilarityMap(LinearMap(UniformScaling(1 / downsample))), m0.intrinsics.K)
    lc = VgtkCore.BoundedCamera(CameraModel(LogicalIntrinsics(Kd), m0.lens), VgtkCore.Sensor(VgtkCore.Size2(Int(W), Int(H)), VgtkCore.px * sensor.pitch.width * downsample))
    lens₀ = lens === :camera ? lc : grid_model(lc, grid_size[1], grid_size[2])
    f_m = VgtkCore.calibration_matrix(VgtkCore.model(lc))[1, 1] * _pixel_pitch(VgtkCore.sensor(lc))
    nodes = grid_nodes(focus_grid, VgtkCore.Rect(0, 0, W, H))
    cosα_i = chief_cos_at_nodes(lens₀, nodes)                 # NaN at a node the lens does not image
    K = VgtkCore.model(lc).intrinsics.K
    board_distances = map(cam_poses) do p
        dir = p.R * normalize(backproject(lens₀, SVector(K[1, 3], K[2, 3])))                 # the principal ray
        dot(world_point(board, SVector(0.0, 0.0)) - p.t, board_normal(board)) / dot(dir, board_normal(board))
    end
    g = mean(board_distances)
    𝐛_c = 𝔟(board)                                         # board-centre point (world)
    center_distances = [dot(p.R[:, 3], 𝐛_c - p.t) for p in cam_poses]   # axial distance to the board centre
    d_c = mean(center_distances)
    d_s = image_distance === nothing ? (focus_init === :board_center ? f_m * d_c / (d_c - f_m) : f_m) :
          image_distance === :board ? f_m * g / (g - f_m) : Float64(image_distance)
    𝔱₀ = focus_init === :board_center ? d_c :
         focus_init === :thin_lens ? thin_lens_focus_distance(lens₀, sensor, r₀; d_s) :
         focus_init === :board ? g : Float64(focus_init)
    focus₀ = Focus(FocusSpline(nodes), isfinite(𝔱₀) ? focus_start(lens₀, nodes, 𝔱₀) : zeros(length(nodes)))
    settings = raw_dir === nothing ? nothing :
        VgtkFeatures.extract_capture_settings(Dict(t => VgtkFeatures.load_raw_metadata(raw_file(raw_dir, traj.images[t])) for t in frames);
                                              lens = lens_identity)
    return (; scene, poses = cam_poses, targets, names, frames, lens₀, focus₀, r₀ = Float64(r₀), sensor = VgtkCore.sensor(lc), cosα = cosα_i, nodes, 𝔱₀, d_s, board_distances,
            center_distances, target_id = id, extent = VgtkCore.extent(bc), downsample, settings)
end

"""
    settings_line(cs::CaptureSettings; lens_stated = false) -> String

One line of a capture's camera settings for a run's log: the frame count, the body, the lens with its source (the config's `lens_stated`,
else EXIF) and the exposure — "constant 1/60 s f/4 ISO 100" when every frame is exposed alike (Core's `is_constant`), the ranges
otherwise.
"""
function settings_line(cs::VgtkCore.CaptureSettings; lens_stated::Bool = false)
    name(d) = (s = join(filter(!isnothing, (d.make, d.model)), " "); d.serial === nothing ? s : "$s #$(d.serial)")
    sec = VgtkCore.parse_unit("s")
    seconds(t) = (x = VgtkCore.ustrip(sec, t); x >= 1 ? @sprintf("%.3g s", x) : @sprintf("1/%d s", round(Int, 1 / x)))
    num(x) = @sprintf("%.3g", x)
    ex = [f.exposure for f in Base.values(cs.frames)]     # (`values` is the package's own name too)
    function span(f, show)
        lo, hi = extrema(f.(ex))
        return lo == hi ? show(lo) : "$(show(lo))–$(show(hi))"
    end
    exposure = VgtkCore.is_constant(cs) ? "constant $(seconds(ex[1].exposure_time)) f/$(num(ex[1].f_number)) ISO $(num(ex[1].iso))" :
        "$(span(e -> e.exposure_time, seconds)), f/$(span(e -> e.f_number, num)), ISO $(span(e -> e.iso, num))"
    n = length(cs.frames)
    return "$n frame$(n == 1 ? "" : "s"); body $(name(cs.body)); lens $(name(cs.lens)) ($(lens_stated ? "config" : "EXIF")); exposure $exposure"
end

"""
    green_photosite_mask(m::RawMosaic) -> BitMatrix

The usable green photosites of the mosaic `m` on the raw grid (a RAW photograph's observations, `observe`; owner ruling 2026-11-12): both greens of every
Bayer cell (Core's `photosite_colors`, 1 or 3), the censored photosites dropped at both ends — the ADC floor (raw = 0) and the
white level (Core's `saturationview`, raw ≥ white); a value below the black level is noise, not censored, and stays.
"""
function green_photosite_mask(m::VgtkCore.RawMosaic)
    c = VgtkCore.photosite_colors(m)
    return ((c .== 1) .| (c .== 3)) .& (m .> 0) .& .!VgtkCore.saturationview(m)
end

"""The geometry images, the one vocabulary of every geometry stage's `image` key (the chrome detect, the front ends' decode, the
image-region measurement; owner 2026-11-16: ONE geometric image for every detector): `"raw_green"` a RAW photograph's
VisualGeometryToolkitFeatures bilinear green on the raw grid ([`raw_green_image`](@ref): full resolution, the raw's COLMAP pixels,
no map), `"luma"` a decoded image read as grey (its own pixels). The half-resolution green cells are no geometry image."""
const GEOMETRY_IMAGES = ("raw_green", "luma")

"""
    check_geometry_image(t, where) -> String

The `image` key of the table `t` (a geometry stage's), REQUIRED (no default: every config of record states its image) and one of
[`GEOMETRY_IMAGES`](@ref); an `ArgumentError` naming `where` otherwise (`"green_cells"` by name: retired for geometry).
"""
function check_geometry_image(t::AbstractDict, where::AbstractString)
    haskey(t, "image") || throw(ArgumentError("$where image: required — the geometry image, one of $(join(repr.(GEOMETRY_IMAGES), ", ")) " *
                                              "(\"raw_green\": a RAW photograph's bilinear green on the raw grid, owner 2026-11-16)"))
    v = t["image"]
    v == "green_cells" && throw(ArgumentError("$where image: \"green_cells\" is retired for geometry (owner 2026-11-16): a RAW set is " *
                                               "read as its bilinear green on the raw grid, image = \"raw_green\""))
    v isa AbstractString && v in GEOMETRY_IMAGES || throw(ArgumentError("$where image = $(repr(v)): one of $(join(repr.(GEOMETRY_IMAGES), ", "))"))
    return String(v)
end

"""
    raw_green_image([T = Float32], m::RawMosaic; sensor_model::SensorModel, device) -> Matrix{T}

The geometry image `"raw_green"` of the mosaic `m` (on the host): VisualGeometryToolkitFeatures' bilinear green on the raw grid under
the frame's `sensor_model` — the image Features' `detect_sfop(m::RawMosaic)` and the seeded `detect_foerstner(m, seeds)` detect on
(the chrome detect), computed on `device` by Features' one kernel (`_bilinear_green`, raw_green.jl: a green photosite its own value,
a red or blue one the mean of its usable green 4-neighbours; defects missing, saturation censored at 1). Linear values (Core's
`linearview` under the model), in the raw's COLMAP pixels (the mosaic's `raw_frame_map`: the identity as decoded); NaN where no usable
green is in the 3×3 (a defect cluster). No second implementation: Features exports no name for the plane alone (a Features API is the
follow-up), so this is the one caller outside its detectors.
"""
raw_green_image(m::VgtkCore.RawMosaic; kw...) = raw_green_image(Float32, m; kw...)
raw_green_image(::Type{T}, m::VgtkCore.RawMosaic; sensor_model::VgtkCore.SensorModel, device) where {T <: AbstractFloat} =
    Array(VgtkFeatures._bilinear_green(T, device, m, sensor_model).value)

"""
    raw_image_size(path) -> (W, H)

The image size of the RAW photograph `path` on the raw grid (its decoded mosaic's columns and rows, Features' `load_raw_mosaic`): the
size of a photosites photograph, the camera's sensor in pixels.
"""
raw_image_size(path::AbstractString) = (m = VgtkFeatures.load_raw_mosaic(path); (size(m, 2), size(m, 1)))

"Whether `path` is a RAW file (its extension one of `RAW_EXTENSIONS`, compared case-insensitively; never a macOS sidecar,
`_listable`): THE test of every RAW lookup."
is_raw_file(path::AbstractString) = _listable(path) && lowercase(splitext(path)[2]) in RAW_EXTENSIONS
"Whether `path` is a PNG image (its extension `.png`, case-insensitively; never a macOS sidecar, `_listable`): THE test of every PNG listing."
is_png_file(path::AbstractString) = _listable(path) && lowercase(splitext(path)[2]) == ".png"
"""Whether a directory entry may be a photograph: not a macOS AppleDouble sidecar (`._<name>`) nor Finder's `.DS_Store` — a copy
from a Mac leaves both beside the files, and they were taken for photographs before (owner, 2026-11-15)."""
_listable(path::AbstractString) = (b = basename(path); !startswith(b, "._") && b != ".DS_Store")
"""
    photo_files(dir; join = false) -> Vector{String}

The photographs of the directory `dir`, sorted: its RAW (`is_raw_file`) and PNG (`is_png_file`) files, never a macOS sidecar; their
paths with `join = true` (as `readdir`). THE listing of every photograph set RC and its paper project read.
"""
photo_files(dir::AbstractString; join::Bool = false) = sort!(filter(f -> is_raw_file(f) || is_png_file(f), readdir(dir; join)))
"""The RAW extensions a session's frames are looked up by (`raw_file`), compared case-insensitively."""
const RAW_EXTENSIONS = (".cr2", ".arw", ".nef", ".dng")

"The RAW files in `dir` with the stem of the photograph `image` (`is_raw_file`): `raw_file`'s and `photo_file`'s one lookup."
_raw_files_of(dir::AbstractString, image::AbstractString) =
    (stem = first(splitext(basename(image))); [f for f in readdir(dir) if first(splitext(f)) == stem && is_raw_file(f)])

"""
    raw_file(raw_dir, image) -> path

The RAW file of the photograph `image` in `raw_dir`: the one file with `image`'s stem and a RAW extension (`RAW_EXTENSIONS`, any
case); an `ArgumentError` naming the image when there is none or more than one.
"""
function raw_file(raw_dir::AbstractString, image::AbstractString)
    stem = first(splitext(basename(image)))
    hits = _raw_files_of(raw_dir, image)
    length(hits) == 1 || throw(ArgumentError(isempty(hits) ?
        "no RAW file of $(repr(image)) in $raw_dir (a file $(stem) with an extension of $(join(RAW_EXTENSIONS, ", ")))" :
        "$(length(hits)) RAW files of $(repr(image)) in $raw_dir ($(join(hits, ", "))): one is expected"))
    return joinpath(raw_dir, only(hits))
end

"""
    photo_file(dir, name) -> path

The file of the photograph `name` in `dir`, decided by the file, never by a flag (owner ruling 2026-11-13): its RAW file — `name`'s
stem with a RAW extension (`is_raw_file`) — when `dir` holds one, else `name` itself, a decoded image. Both, neither, or more than one
RAW file is an `ArgumentError` naming them: the choice is never guessed.
"""
function photo_file(dir::AbstractString, name::AbstractString)
    raws = _raw_files_of(dir, name)
    decoded = !is_raw_file(name) && isfile(joinpath(dir, name))
    length(raws) == 1 && !decoded && return joinpath(dir, only(raws))
    isempty(raws) && decoded && return joinpath(dir, name)
    throw(ArgumentError(isempty(raws) ? "no photograph $(repr(name)) in $dir: neither the file nor a RAW file of its stem" :
                        decoded ? "the photograph $(repr(name)) in $dir is both a decoded image and a RAW file ($(join(raws, ", "))): one is expected" :
                        "$(length(raws)) RAW files of $(repr(name)) in $dir ($(join(raws, ", "))): one is expected"))
end

"""
    load_photo(path; linearize = false) -> RawMosaic | AbstractArray

THE loader of a photograph, decided by the file (`is_raw_file`), never by a flag: a RAW file is Core's `RawMosaic` through Features'
`load_raw_mosaic` (the one RAW path), a decoded image its linear array (Features' `load_linear_image`; `linearize` decodes sRGB). A
RAW photograph is linear: `linearize` with one is an `ArgumentError`. What a photograph contributes is then decided by its TYPE
(the observation model, [`observe`](@ref)).
"""
function load_photo(path::AbstractString; linearize::Bool = false)
    is_raw_file(path) || return VgtkFeatures.load_linear_image(path; srgb = linearize)
    linearize && throw(ArgumentError("$path: a RAW photograph is linear (Features' linearisation): `io.linearize` does not apply"))
    return VgtkFeatures.load_raw_mosaic(path)
end

"""
    sensor_record(names, models) -> Vector{NamedTuple}

The run record's sensor models, one per photograph (`names[k]`, `models[k]`): Core's `component_sources` (each slot's source,
`nothing` when absent; the resolver's reason; whether a specified component overrode a measured one), the black component the
photograph's values were linearised with (`black_component`: `Dark` — per photosite, the photograph's EXIF exposure inside the
dark's band —, `ChannelBlack` — per colour, e.g. reason `DarkOutOfBand` —, or `FileBlack` — the RAW's own, e.g. reason
`NoCharacterisation`) and the model's identity (Core's `content_digest`), as plain strings.
"""
sensor_record(names, models) = [begin
                                     c = VgtkCore.component_sources(m)
                                     str(x) = x === nothing ? nothing : string(x)
                                     (; view = String(n), reason = string(c.reason), black = str(c.black),
                                        black_component = string(nameof(typeof(m.black))), noise = str(c.noise),
                                        defects = str(c.defects), specified_override = c.specified_override, digest = VgtkCore.content_digest(m))
                                 end for (n, m) in zip(names, models)]

# The photosites a sensor model's defects mask, on the mosaic's raw grid: Core's public `defect_mask` (the mask LinearRaw's kernel
# applies; `nothing` for NoDefects). The mask covers the visible raw; the mosaic's photosites are read at its `window_origin`.
function _defect_photosites(model::VgtkCore.SensorModel, m::VgtkCore.RawMosaic)
    A = VgtkCore.defect_mask(model, m)
    A === nothing && return nothing
    (r0, c0) = m.window_origin
    return A[r0 .+ (1:size(m, 1)), c0 .+ (1:size(m, 2))]
end
_usable(::Nothing) = true
_usable(defects::AbstractMatrix{Bool}) = .!Array(defects)

# ---- THE entry point of every photograph a camera is calibrated on or predicts (RC 0.11.1) ------------------------------------
"""
    PhotoSet

The photographs of one calibration or prediction, read under ONE [`ObservationModel`](@ref) ([`photo_set`](@ref)): the model, the
files and each photograph's Core `SensorModel` (`models[k]`, resolved by its EXIF from the sensor record; the absent model without
one). [`observe`](@ref) reads photograph `k` into its observations.
"""
struct PhotoSet{O <: ObservationModel}
    observation::O
    files::Vector{String}
    models::Vector{VgtkCore.SensorModel}
end

"""
    photo_set(camera::CalibratedCamera, files; characterisation = nothing, lens = nothing, sensor_model = AutoSensorModel()) -> PhotoSet
    photo_set(observation::ObservationModel, files; characterisation = nothing, lens = nothing) -> PhotoSet

THE entry point of photographs (owner 2026-11-15: "if a sensor was estimated with a model, then it must be used to predict the
model"): the photographs `files` read exactly as the camera's — or the calibration's — observation model states, by dispatch on its
types, never by a flag. `characterisation` is a LOCATION only (a path of the `vgio/sensor_characterisation` record); what it must be is
the camera's statement. `lens`: the lens the config states (an adapted lens EXIF misnames; Features' `extract_capture_settings`).
A camera's prediction reads them under `sensor_model` (`prediction_observation`: "off" reads RAW photosites without the
record, with the loud warning).

- [`SensorRecord`](@ref), [`Photosites`](@ref): RAW photographs only; `characterisation` required and its file's sha256 equal to the
  record's — another file is refused naming both. Each photograph's model is Core's resolver `sensor_model(characterisation,
  capture, k)` by its EXIF; a frame the record does not describe (another body, `OtherBody`; an ISO it holds no entry for,
  `UnmatchedISO`) is refused by name.
- [`NoSensorRecord`](@ref), [`Photosites`](@ref): RAW photographs only, each under the file's black (Core's resolver without a
  characterisation, reason `NoCharacterisation`); a `characterisation` is refused: the camera was calibrated without one.
- [`NoSensorRecord`](@ref), [`Decoded`](@ref): decoded images only, the absent model each; a RAW photograph is refused, so is a
  `characterisation`.
"""
photo_set(cam::CalibratedCamera, files::AbstractVector{<:AbstractString}; sensor_model::SensorModelMode = AutoSensorModel(),
          characterisation::Union{Nothing, AbstractString} = nothing, kwargs...) =
    photo_set(prediction_observation(sensor_model, cam.observation, characterisation)..., files; kwargs...)
"""
    prediction_observation(sensor_model, observation, characterisation) -> (observation, (; characterisation))

The observation model, and the record location, a prediction of a camera calibrated under `observation` reads its photographs with,
by `sensor_model` ([`SensorModelMode`](@ref); `camera.sensor_model`): "auto" and "on" read them as the camera states (`photo_set`'s
refusals: another record, another kind, a missing location) — "on" refuses a camera calibrated without a record; "off" reads RAW
photosites WITHOUT a record present (the camera's, or the `characterisation` given, which must exist and, for a camera's record, be
that record), with [`warn_sensor_model_off`](@ref).
"""
prediction_observation(::AutoSensorModel, om::ObservationModel, c) = (om, (; characterisation = c))
prediction_observation(::OnSensorModel, om::ObservationModel, c) = (om, (; characterisation = c))
prediction_observation(::OnSensorModel, om::ObservationModel{NoSensorRecord}, c) =
    throw(ArgumentError("`camera.sensor_model = \"on\"`: the camera was calibrated without a sensor record ($(observation_line(om))): no sensor model to predict with"))
function prediction_observation(::OffSensorModel, om::ObservationModel{SensorRecord, Photosites}, c)
    if c !== nothing                                              # a location given: the camera's own record, or refused
        got = sensor_statement(c)
        got == om.sensor || throw(ArgumentError("the camera was calibrated under the sensor characterisation sha256 $(om.sensor.sha256), given sha256 $(got.sha256) ($c)"))
    end
    warn_sensor_model_off(c === nothing ? _record_line(om.sensor) : _record_line(c, om.sensor))
    return (ObservationModel(NoSensorRecord(), Photosites(), om.channel), (; characterisation = nothing))
end
function prediction_observation(::OffSensorModel, om::ObservationModel{NoSensorRecord, Photosites}, c)
    c === nothing || warn_sensor_model_off(_record_line(c, sensor_statement(c)))   # a record present on disk, not used
    return (om, (; characterisation = nothing))
end
prediction_observation(::OffSensorModel, om::ObservationModel, c) = (om, (; characterisation = c))   # decoded images: photo_set's refusals
photo_set(om::ObservationModel, ch::NamedTuple, files::AbstractVector{<:AbstractString}; kwargs...) = photo_set(om, files; ch..., kwargs...)
function photo_set(om::ObservationModel{SensorRecord, Photosites}, files::AbstractVector{<:AbstractString};
                   characterisation::Union{Nothing, AbstractString} = nothing, lens::Union{Nothing, VgtkCore.DeviceIdentity} = nothing)
    _require_kind(om, files)
    want = om.sensor.sha256
    characterisation === nothing &&
        throw(ArgumentError("the camera was calibrated under the sensor characterisation sha256 $want: its photographs are read under that record (give its location: `characterisation`, the path)"))
    got = open(VgtkCore.sha256_hex, characterisation)
    got == want || throw(ArgumentError("the camera was calibrated under the sensor characterisation sha256 $want, given sha256 $got ($characterisation): " *
                                       "a prediction reads its photographs under the camera's own record"))
    c = VgtkIO.load(VgtkCore.SensorCharacterisation, characterisation)
    models = _resolved_models(c, files, lens)
    _require_described(want, files, models)
    return PhotoSet(om, String.(files), models)
end
# A frame the sensor record does not describe — another body (`OtherBody`), an ISO it holds no entry for (`UnmatchedISO`) — is refused by name.
function _require_described(sha256::AbstractString, files::AbstractVector{<:AbstractString}, models::AbstractVector{<:VgtkCore.SensorModel})
    bad = [(basename(f), VgtkCore.component_sources(m).reason) for (f, m) in zip(files, models)
           if VgtkCore.component_sources(m).reason in (VgtkCore.UnmatchedISO, VgtkCore.OtherBody)]
    isempty(bad) || throw(ArgumentError("the sensor characterisation $sha256 does not describe $(length(bad)) photographs " *
                                        "($(join(["$n ($r)" for (n, r) in bad], ", "))): the camera's record predicts the frames it describes"))
    return nothing
end
function photo_set(om::ObservationModel{NoSensorRecord, Photosites}, files::AbstractVector{<:AbstractString};
                   characterisation::Union{Nothing, AbstractString} = nothing, lens::Union{Nothing, VgtkCore.DeviceIdentity} = nothing)
    _require_kind(om, files)
    _refuse_characterisation(characterisation)
    return PhotoSet(om, String.(files), _resolved_models(nothing, files, lens))
end
function photo_set(om::ObservationModel{NoSensorRecord, Decoded}, files::AbstractVector{<:AbstractString};
                   characterisation::Union{Nothing, AbstractString} = nothing, lens::Union{Nothing, VgtkCore.DeviceIdentity} = nothing)
    _require_kind(om, files)
    _refuse_characterisation(characterisation)
    return PhotoSet(om, String.(files), VgtkCore.SensorModel[VgtkCore.SensorModel() for _ in files])
end
_refuse_characterisation(::Nothing) = nothing
_refuse_characterisation(path::AbstractString) =
    throw(ArgumentError("the camera was calibrated without a sensor characterisation: its photographs are read without one (given $path)"))
_kind_name(::Photosites) = "RAW photographs (their photosites)"
_kind_name(::Decoded) = "decoded images"
function _require_kind(om::ObservationModel, files)
    wrong = [basename(f) for f in files if is_raw_file(f) != (om.kind isa Photosites)]
    isempty(wrong) || throw(ArgumentError("the camera was calibrated on $(_kind_name(om.kind)): it predicts those only, not $(join(wrong, ", "))"))
    return nothing
end
# Core's resolver per frame, by the photographs' EXIF (Features' `load_raw_metadata`, the headers alone, into `extract_capture_settings`)
function _resolved_models(c, files, lens)
    capture = VgtkFeatures.extract_capture_settings(Dict(k => VgtkFeatures.load_raw_metadata(f) for (k, f) in enumerate(files)); lens)
    return VgtkCore.SensorModel[VgtkCore.sensor_model(c, capture, k) for k in eachindex(files)]
end

"""
    observe(set::PhotoSet, k, mask; name = basename(set.files[k])) -> (values, mask)

Photograph `k` of `set` ([`photo_set`](@ref)) under the compute `mask`, read once (`load_photo`) under its model:
- RAW photosites: its green photosites on the raw grid — the mask restricted to the usable greens (`green_photosite_mask`: both
  greens, uncensored at the ADC floor and the white level) that its sensor model does not list as defects (Core's defect mask:
  masked, never interpolated); the values its linear view under the model (Core's `linearview`: the model's black subtracted — the
  file's, a measured channel black or a dark frame's per photosite). Under a noise with a variance law (a gain: PoissonGaussian, or a
  RelativeNoise) Core's uncertain image, each value with its variance, whose rows are then whitened (`masked_observations`); plain
  values otherwise. A mosaic whose size is not the mask's (the camera's sensor) is refused;
- decoded images: the image itself (sRGB-decoded when the model's `Decoded` states it) under the mask; an image whose size is not
  the mask's is refused.
"""
observe(set::PhotoSet, k::Integer, mask::AbstractMatrix{Bool}; name::AbstractString = basename(set.files[k])) =
    _observe(set.observation.kind, set.files[k], mask, name, set.models[k])
# the photograph read once (`load_photo`, by its file); `photo_set` checked that every file is of the model's kind
_observe(k::Photosites, path::AbstractString, mask, name, model) = _observe(k, load_photo(path)::VgtkCore.RawMosaic, mask, name, model)
_observe(k::Decoded, path::AbstractString, mask, name, model) = _observe(k, load_photo(path; linearize = k.srgb)::AbstractArray, mask, name, model)
_observe(::Decoded, ::VgtkCore.RawMosaic, ::AbstractMatrix{Bool}, name::AbstractString, ::VgtkCore.SensorModel) =
    throw(ArgumentError("$name: a RAW mosaic under an observation model of decoded images: the camera predicts decoded images only"))
function _observe(::Photosites, m::VgtkCore.RawMosaic, mask::AbstractMatrix{Bool}, name::AbstractString, model::VgtkCore.SensorModel)
    size(m) == size(mask) || error("$name: the RAW mosaic is $(size(m, 2)) × $(size(m, 1)), the camera's sensor $(size(mask, 2)) × $(size(mask, 1))")
    return VgtkCore.linearview(m; sensor_model = model), mask .& green_photosite_mask(m) .& _usable(_defect_photosites(model, m))
end
function _observe(::Decoded, img::AbstractArray, mask::AbstractMatrix{Bool}, name::AbstractString, ::VgtkCore.SensorModel)
    size(img)[1:2] == size(mask) ||
        error("$name: the photograph is $(size(img, 2)) × $(size(img, 1)) px, the start camera $(size(mask, 2)) × $(size(mask, 1)) px: refusing — the start is defined on the photographs' pixels")
    return img, mask
end

"""The state's `channel` of `io.channel`: 0 for `"RGB"` (three residuals per pixel), 1 / 2 / 3 for `"R"` / `"G"` / `"B"` (that
channel alone: ONE residual per measured pixel, `fit_channels`)."""
function _io_channel(channel::AbstractString)
    channel == "RGB" && return 0
    k = findfirst(==(channel), CHANNEL_NAMES)
    k === nothing && throw(ArgumentError("io.channel: $(repr(channel)) is not one of RGB, $(join(CHANNEL_NAMES, ", "))"))
    return k
end
