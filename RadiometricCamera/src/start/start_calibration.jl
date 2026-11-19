# ---------------------------------------------------------------------------
# The start of a radiometric calibration: the geometric calibration it starts from (`StartGeometry`: a geometric camera, its poses,
# the observations of the target, the target) and the step that lifts it into a radiometric state (`start_calibration`). The
# pipeline's front ends build the geometry from their inputs through VisualGeometryToolkitIO (`start_geometry`, start/frontends/);
# a start is written as its record (`save_calibration_start`) and a refinement's state is rebuilt from one (`load_calibration_start`).
# ---------------------------------------------------------------------------

"""
    StartGeometry(camera, poses, observations, target; names, image_size = extent(camera), init_region = nothing, resample = nothing,
                  features_for = _ -> observations, world = "the target's frame", page_flip = 1.0, board_y_sign = 1.0)

The geometric calibration a radiometric calibration starts from: the geometric `camera` (a `BoundedCamera`, the start camera), one
`poses[k]` per photograph (camera → world, `EuclideanMap`), the `observations[k]` of the target in photograph `k` (`(𝐩, X)` pairs: the
image point, COLMAP px, and the target point in the world, plain metres) and the printed `target` (a [`PrintedTarget`](@ref)). The
front end's facts ride along: the photographs' `names`, the image size, the start camera's initialization region (where it is
defined; `nothing`: its own domain), the record of its exact resample (`resample`, [`exact_start`](@ref)'s; `nothing` for a camera
that was not resampled), `features_for(surface)` (the observations under a warped or misprinted surface), the world's
name and the front end's frame choices (`page_flip`, `board_y_sign`) its records keep.
"""
struct StartGeometry{C, P, T, R, F}
    camera::C
    poses::Vector{P}
    observations::Vector{Vector{Tuple{SVector{2, Float64}, SVector{3, Float64}}}}
    target::T
    names::Vector{String}
    image_size::Tuple{Int, Int}
    init_region::R
    resample::Union{Nothing, NamedTuple}
    features_for::F
    world::String
    page_flip::Float64
    board_y_sign::Float64
end
function StartGeometry(camera, poses::AbstractVector, observations::AbstractVector, target; names::AbstractVector{<:AbstractString},
                       image_size = Tuple(VgtkCore.extent(camera)), init_region = nothing, resample = nothing, features_for = _ -> observations,
                       world::AbstractString = "the target's frame", page_flip::Real = 1.0, board_y_sign::Real = 1.0)
    K = length(names)
    length(poses) == K && length(observations) == K ||
        throw(DimensionMismatch("StartGeometry: $(K) photographs, $(length(poses)) poses and $(length(observations)) observation sets"))
    obs = [Tuple{SVector{2, Float64}, SVector{3, Float64}}[(SVector{2, Float64}(p), SVector{3, Float64}(X)) for (p, X) in o] for o in observations]
    return StartGeometry(camera, collect(poses), obs, target, String.(names), (Int(image_size[1]), Int(image_size[2])), init_region,
                         resample, features_for, String(world), Float64(page_flip), Float64(board_y_sign))
end

"""
    start_geometry(options; record = nothing) -> StartGeometry

The start geometry of the run `options` from its front end (`frontend.type`: a calibration session's records, or a Schöps run's), every
input read through VisualGeometryToolkitIO: the start camera (the front end's onto the run's lattice, `start_grid`; or the stored start
`io.init`), its poses, the observations of the target and the printed target. With a start `record` (a start record's,
`load_calibration_start`) its camera, initialization region and poses are the record's, nothing fitted again.
"""
function start_geometry(options::PipelineOptions; record = nothing, load_state = RadiometricCamera.load_state)
    R0 = VgtkCore.ustrip(VgtkCore.m, _focal_length(options)) / (2 * options.camera.fnumber)
    fe = _frontend(options.frontend.type, options,
                   (; R0, height0 = _height_start(options), inplane0 = _inplane_start(options), stored = record,
                      page_flip = record === nothing ? _state_page_flip(options, load_state) : record.page_flip))   # a rendered / resumed state's ±y choice holds; a start record's frame
    return StartGeometry(fe.lens0, fe.poses0, fe.features, fe.board; names = fe.names, image_size = (fe.W₀, fe.H₀), fe.init_region, fe.resample,
                         fe.features_for, fe.world, fe.page_flip, fe.board_y_sign)
end

"""
    start_calibration(geometry::StartGeometry, photographs; options, device = default_device()) -> CalibrationState
    start_calibration(camera, poses, observations, photographs; target, options, device = default_device(), names) -> CalibrationState

The start of a radiometric calibration: it lifts the geometric camera of `geometry` (or of `camera`, `poses`, `observations` and
`target`, the geometric calibration as data) into a radiometric state — the focus surface through the mean target centre, the
aperture from the f-number, the image circle measured where the run names its frames, the masks, the photographs' observations
streamed (`photographs[k]` the file of view `k`: a RAW photograph by its file, its green photosites, or a decoded image), the
vignetting and the lighting by the closed-form start, the grid smoothness. `options` are the start's ([`start_options`](@ref)). The
state's `start` holds what the refinement and its records read (the start views, the observations, the costs of the start).

Every input is a value: the pipeline's front ends read theirs through VisualGeometryToolkitIO ([`start_geometry`](@ref)); a caller
with a Schöps run, a Kalibr or an OpenCV calibration of its own loads its camera and observations with IO's importers itself
(`VisualGeometryToolkitIO.load(BoundedCamera, …)`, `load(TargetObservations, …)`) — RC reads no foreign format.
"""
start_calibration(geometry::StartGeometry, photographs::AbstractVector{<:AbstractString}; options::PipelineOptions,
                  device = VgtkCore.default_device()) = _start_state(geometry, photographs, options, nothing; backend = device)
start_calibration(camera::VgtkCore.BoundedCamera, poses::AbstractVector, observations::AbstractVector, photographs::AbstractVector{<:AbstractString};
                  target::PrintedTarget, options::PipelineOptions, device = VgtkCore.default_device(), names = basename.(photographs)) =
    start_calibration(StartGeometry(camera, poses, observations, target; names), photographs; options, device)

"""
    save_calibration_start(dir, state; options) -> dir

The start `state` ([`start_calibration`](@ref)'s, of the start options `options`) as the start record `dir`: every value a
refinement takes from its start (start/start_record.jl), so [`load_calibration_start`](@ref) rebuilds the start's state from the
photographs and this record without computing it again.
"""
save_calibration_start(dir::AbstractString, state::CalibrationState; options::PipelineOptions) =
    (save_start(dir, state, state.start, start_options(options)); dir)   # the record keys by the start options alone (no output: io.out, lm.log)

"""
    load_calibration_start(dir; options, device = default_device(), observations = nothing, photographs) -> CalibrationState

The refinement's state from the start record `dir` ([`save_calibration_start`](@ref); refused when its start options are not
`start_options(options)`): the front end's target and observations, the record's camera, region and poses, the photographs streamed
at the record's masks (or the start's own streamed `observations`, handed over in the process that computed it: no photograph read
again), and the closed-form start's state, costs and GCV weights loaded — nothing of the start computed again. `photographs`: the
files of the record's views, by default the run's (`io.images`, [`start_photographs`](@ref)).
"""
function load_calibration_start(dir::AbstractString; options::PipelineOptions, device = VgtkCore.default_device(), observations = nothing,
                                photographs = nothing, load_state = RadiometricCamera.load_state)
    record = load_refinement_start(dir; masks = observations === nothing ? nothing : observations.masks)
    check_start_options(record, options)
    g = start_geometry(options; record, load_state)
    files = photographs === nothing ? start_photographs(options, g.names) : photographs
    return _start_state(g, files, options, record; backend = device, observations)
end

"""The photographs of the views `names` of the run `options` (`io.images`; a RAW file by its stem, `photo_file`)."""
function start_photographs(options::PipelineOptions, names::AbstractVector{<:AbstractString})
    isempty(options.io.images) && error("io.images required (only a self-check, selfcheck.only, runs without photographs)")
    return [photo_file(options.io.images, n) for n in names]
end

"""
    selfcheck(geometry; options)

The diagnostic `selfcheck.only`: the bearing error of the start geometry's observations through its camera (per view, under its poses
or those of the `selfcheck.poses` state) → `selfcheck.json`, `selfcheck_vectors.json` in `io.out`. No photograph is read.
"""
function selfcheck(g::StartGeometry; options::PipelineOptions, load_state = RadiometricCamera.load_state)
    mkpath(options.io.out)
    fpx = fpx_of(g.camera, g.image_size...)
    be0 = _start_bearings(g, fpx)
    return _write_selfcheck(g, options, fpx, be0, load_state)
end
"""The bearing errors of the observations through the start camera under the start poses, on its initialization region (px)."""
_start_bearings(g::StartGeometry, fpx) =
    detection_bearings(g.camera, g.poses, [in_initialization(g.observations[k], g.init_region) for k in eachindex(g.names)], fpx)
