# ---------------------------------------------------------------------------
# The start of a radiometric calibration: the geometric calibration it starts from (`StartGeometry`: a geometric camera, its poses,
# the observations of the target, the target) and the step that lifts it into a radiometric state (`_start_state`, the `init` of a `RadiometricCalibration`). The
# pipeline's front ends build the geometry from their inputs through VisualGeometryToolkitIO (`start_geometry`, start/frontends/);
# a start is written as its record (`save_calibration_start`) and a refinement's state is rebuilt from one (`load_calibration_start`).
# ---------------------------------------------------------------------------

"""
    StartGeometry(camera, poses, observations, target; names, image_size = extent(camera), init_region = nothing, resample = nothing,
                  features_for = _ -> observations, world = "the target's frame", page_flip = 1.0, board_y_sign = 1.0)

The geometric calibration a radiometric calibration starts from: the geometric `camera` (a `BoundedCamera`, the start camera), one
`poses[k]` per photograph (camera → world, `EuclideanMap`), the `observations[k]` of the target in photograph `k` (`(𝐩, X)` pairs: the
image point, COLMAP px, and the target point in the world, plain metres) and the printed `target` (a `TexturedTarget`). The
front end's facts ride along: the photographs' `names`, the image size, the start camera's initialization region (where it is
defined; `nothing`: its own domain), the record of its exact resample (`resample`, [`GridStart`](@ref)'s; `nothing` for a camera
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
    CalibrationPhotographs(geometry::StartGeometry, photographs)

The data of a radiometric calibration from scratch ([`RadiometricCalibration`](@ref)'s `EstimationProblem(CalibrationEstimate,
CalibrationPhotographs(geometry, photographs), p)`): the geometric calibration it starts from (`geometry`, a [`StartGeometry`](@ref))
and the photographs' files, `photographs[k]` the file of view `k` (a RAW photograph by its file, its green photosites, or a decoded
image; [`start_photographs`](@ref) resolves a run's), one per photograph of the geometry (a `DimensionMismatch` otherwise).
"""
struct CalibrationPhotographs{G <: StartGeometry}
    geometry::G
    photographs::Vector{String}
end
function CalibrationPhotographs(geometry::StartGeometry, photographs::AbstractVector{<:AbstractString})
    length(photographs) == length(geometry.names) ||
        throw(DimensionMismatch("CalibrationPhotographs: $(length(geometry.names)) photographs in the geometry, $(length(photographs)) files"))
    return CalibrationPhotographs(geometry, collect(String, photographs))
end

"""
    start_geometry(run, settings; record = nothing) -> StartGeometry

The start geometry of the run `run` (its [`RunOptions`](@ref); `settings` its calibration's, the grid's lattice) from its front end (`frontend.type`: a calibration session's records, or a Schöps run's), every
input read through VisualGeometryToolkitIO: the start camera (the front end's onto the run's lattice, `GridStart`; or the stored start
`io.init`), its poses, the observations of the target and the printed target. With a start `record` (a start record's,
`load_calibration_start`) its camera, initialization region and poses are the record's, nothing fitted again.
"""
function start_geometry(run::RunOptions, settings::CalibrationSettings; record = nothing, load_state = RadiometricCamera.load_state)
    R0 = VgtkCore.ustrip(VgtkCore.m, _focal_length(run)) / (2 * run.camera.fnumber)
    fe = _frontend(run.frontend.type, run, settings,
                   (; R0, height0 = _height_start(settings), inplane0 = _inplane_start(settings), stored = record,
                      page_flip = record === nothing ? _state_page_flip(run, load_state) : record.page_flip))   # a rendered / resumed state's ±y choice holds; a start record's frame
    return StartGeometry(fe.lens0, fe.poses0, fe.features, fe.board; names = fe.names, image_size = (fe.W₀, fe.H₀), fe.init_region, fe.resample,
                         fe.features_for, fe.world, fe.page_flip, fe.board_y_sign)
end

"""
    save_calibration_start(dir, f, θ; run, settings) -> dir

The start — data `f` and parameters `θ` (the `init` of a [`RadiometricCalibration`](@ref) of the run `run` and the settings `settings`:
its cache's `f` and `u0`) — as the start record `dir`: every value a refinement takes from its start (start/start_record.jl), so
[`load_calibration_start`](@ref) rebuilds the start from the photographs and this record without computing it again.
"""
save_calibration_start(dir::AbstractString, f::CalibrationData, θ::CalibrationParameters; run::RunOptions, settings::CalibrationSettings) =
    (save_start(dir, f, θ, f.start, start_options(run, settings)...); dir)   # the record keys by the start options alone (no output: io.out, optimizer.log)

"""
    load_calibration_start(dir; run, settings, device = default_backend(), observations = nothing, photographs) -> RefineProblem

The refinement's problem from the start record `dir`, `RefineProblem(f, u0, run)` of the data `f` ([`CalibrationData`](@ref)), the
start's parameters `u0` ([`CalibrationParameters`](@ref)) and the run's options `run` (its context `p`), which [`RadiometricRefinement`](@ref) solves ([`save_calibration_start`](@ref); refused when its start options are not
`start_options(run, settings)`): the front end's target and observations, the record's camera, region and poses, the photographs streamed
at the record's masks (or the start's own streamed `observations`, handed over in the process that computed it: no photograph read
again), and the closed-form start's parameters, costs and GCV weights loaded — nothing of the start computed again. `photographs`: the
files of the record's views, by default the run's (`io.images`, [`start_photographs`](@ref)).
"""
function load_calibration_start(dir::AbstractString; run::RunOptions, settings::CalibrationSettings, device = default_backend(), observations = nothing,
                                photographs = nothing, load_state = RadiometricCamera.load_state)
    record = load_refinement_start(dir; masks = observations === nothing ? nothing : observations.masks)
    check_start_options(record, run, settings)
    g = start_geometry(run, settings; record, load_state)
    files = photographs === nothing ? start_photographs(run, g.names) : photographs
    f, θ = _start_state(g, files, run, settings, record; backend = device, observations, radiometry = nothing)   # a record's start: no closed-form solve, no LM
    return VgtkCore.RefineProblem(f, θ, run)
end

"""The photographs of the views `names` of the run `run` (`io.images`; a RAW file by its stem, `photo_file`)."""
function start_photographs(run::RunOptions, names::AbstractVector{<:AbstractString})
    isempty(run.io.images) && error("io.images required (only a self-check, selfcheck.only, runs without photographs)")
    return [photo_file(run.io.images, n) for n in names]
end

"""
    selfcheck(geometry; run)

The diagnostic `selfcheck.only`: the bearing error of the start geometry's observations through its camera (per view, under its poses
or those of the `selfcheck.poses` state) → `selfcheck.json`, `selfcheck_vectors.json` in `io.out`. No photograph is read.
"""
function selfcheck(g::StartGeometry; run::RunOptions, load_state = RadiometricCamera.load_state)
    mkpath(run.io.out)
    fpx = fpx_of(g.camera, g.image_size...)
    be0 = _start_bearings(g, fpx)
    return _write_selfcheck(g, run, fpx, be0, load_state)
end
"""The bearing errors of the observations through the start camera under the start poses, on its initialization region (px)."""
_start_bearings(g::StartGeometry, fpx) =
    detection_bearings(g.camera, g.poses, [in_initialization(g.observations[k], g.init_region) for k in eachindex(g.names)], fpx)
