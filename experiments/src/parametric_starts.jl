# =============================================================================
# src/parametric_starts.jl — the geometric starts of the parametric models on a Siemens-star set, fitted to Schöps's star detections
# through VisualGeometryToolkitEstimation: one pose per view by `Ransac` on `AbsolutePoseProblem` (P3P + RANSAC through the nominal
# camera), then Core's `LevenbergMarquardt` on `CameraPosesProblem` (the camera's intrinsics and lens coefficients jointly with every
# pose, the reprojection error in pixels;
# Core's parameter protocol, every parameter but the skew free). The camera is Core's pixel camera, `CameraModel(K; lens)` with the
# Kannala–Brandt lens `KannalaBrandt(k₁, …, k₄)` or the Double Sphere `Lens(DoubleSphere(ξ, α))`, saved as the camera record
# (RadiometricCamera's `save_lens`, vgio/cameras) a refinement case starts from (`io.init = "lens:<record>"`).
# The front ends' stage fits them (scripts/frontends.jl, configs/experiments/frontends.toml [fits.<name>]: src/frontends.jl run_frontend_fit),
# the one route of a parametric start (the AUTHOR-SIDE make_parametric_starts.jl and its config are retired, 2026-11-13).
# Reads: src/boards.jl (`Boards.load_board`), included by the caller.
# =============================================================================
using StaticArrays: SVector, SMatrix
using LinearAlgebra: norm
using Statistics: median
using Random: AbstractRNG, default_rng
using TOML: TOML
import RadiometricCamera as RC
import VisualGeometryToolkitCore as VgtkCore
import VisualGeometryToolkitEstimation as VgtkEstimation
using VisualGeometryToolkitCore: CameraModel, KannalaBrandt, DoubleSphere, Lens, EuclideanMap, BoundedCamera, Size2, found_model,
                                 solve, EstimationProblem, RefineProblem, LevenbergMarquardt, LmStops, default_stops

"The lens families a start is fitted in: `kannala_brandt` (four coefficients), `double_sphere` (ξ, α)."
const START_MODELS = ("kannala_brandt", "double_sphere")
"The Double Sphere starts tried (ξ, α), the best final rms kept: the pinhole, and the shapes of a wide and a fisheye lens."
const DOUBLE_SPHERE_STARTS = ((0.0, 0.0), (-0.5, 0.3), (0.0, 0.5), (0.5, 0.6), (-0.2, 0.6))

"""
    board_points(board_dir) -> Vector{SVector{3, Float64}}

The star positions of a Schöps board record (`Boards.load_board`): Core's design points of its `PlanarTarget` in metres on the board
plane (z = 0), feature id k ↔ point k + 1 (the board's `texture_features.bin` ids).
"""
function board_points(board_dir::AbstractString)
    pts = VgtkCore.design(Boards.load_board(board_dir).target)
    return [SVector{3, Float64}(VgtkCore.ustrip(VgtkCore.m, p[1]), VgtkCore.ustrip(VgtkCore.m, p[2]), 0.0) for p in pts]
end

"""
    schoeps_views(dir, X; min_features) -> Vector{Pair{String, Vector{Pair{SVector{3, Float64}, SVector{2, Float64}}}}}

Per photograph of a Schöps run `dir` (its detections, `RC.schoeps_run_file`: the run's import or its `dataset.bin`; `RC.dataset_features`: COLMAP pixels, 0-based ids) the correspondences design
point `X[id + 1]` => pixel, for the ids the board holds; the views with at least `min_features` of them, in the file's order.
"""
function schoeps_views(dir::AbstractString, X::AbstractVector; min_features::Integer)
    views = Pair{String, Vector{Pair{SVector{3, Float64}, SVector{2, Float64}}}}[]
    for (name, feats) in RC.dataset_features(RC.schoeps_run_file(dir, :detections))
        cs = [X[id + 1] => SVector{2, Float64}(x, y) for (x, y, id) in feats if 0 <= id < length(X)]
        length(cs) >= min_features && push!(views, String(name) => cs)
    end
    isempty(views) && throw(ArgumentError("schoeps_views: no view of $dir has ≥ $min_features detections of the board's $(length(X)) stars"))
    return views
end

"""
    nominal_camera(model, W, H, f_mm, sensor_mm; xi = 0.0, alpha = 0.0) -> CameraModel

The start camera: K of the focal length `f_mm / sensor_mm · W` pixels at the image centre (COLMAP pixels), the lens of `model` at its
nominal — Kannala–Brandt with zero coefficients (the equidistant projection), the Double Sphere at (`xi`, `alpha`).
"""
function nominal_camera(model::AbstractString, W::Integer, H::Integer, f_mm::Real, sensor_mm::Real; xi::Real = 0.0, alpha::Real = 0.0)
    f = Float64(f_mm) / Float64(sensor_mm) * W
    K = SMatrix{3, 3, Float64}(f, 0.0, 0.0, 0.0, f, 0.0, W / 2, H / 2, 1.0)
    lens = model == "kannala_brandt" ? KannalaBrandt(0.0, 0.0, 0.0, 0.0) :
           model == "double_sphere" ? Lens(DoubleSphere(Float64(xi), Float64(alpha))) :
           throw(ArgumentError("nominal_camera: model $(repr(model)) is not one of $(join(START_MODELS, ", "))"))
    return CameraModel(K; lens)
end

"The reprojection of `X` through `camera` at `pose` (world → camera) is finite: the point is in front and inside the lens's domain."
_projects(camera, pose, X, u) = all(isfinite, VgtkCore.reprojection_error(camera, pose, X, u))

"""
    pose_starts(camera, views; scoring, rng) -> (poses, kept)

One pose per view from Estimation's `Ransac` on `AbsolutePoseProblem(camera, cs)` (P3P + RANSAC through `camera`, scored by the table `scoring`, read by
Estimation's `scoring_from_toml`): those that found a pose (Core's `found_model`), world → camera `EuclideanMap`s, and the indices of
their views.
"""
function pose_starts(camera::CameraModel, views; scoring::AbstractDict, rng::AbstractRNG = default_rng())
    sc = VgtkEstimation.scoring_from_toml(scoring; problem = VgtkEstimation.AbsolutePoseProblem, section = "fit.scoring")
    results = [solve(EstimationProblem(VgtkEstimation.AbsolutePoseProblem(camera, cs)), VgtkEstimation.Ransac(; scoring = sc); rng) for (_, cs) in views]
    kept = [k for (k, r) in enumerate(results) if found_model(r)]
    return [results[k].u for k in kept], kept
end

"""
    _fit_start(model, views, W, H, f_mm, sensor_mm; scoring, max_iter, rng) -> NamedTuple

The start of `model` on `views` (`schoeps_views`): from the nominal camera (`nominal_camera`; the Double Sphere from each of
`DOUBLE_SPHERE_STARTS`, the least final rms kept) the pose starts (`pose_starts`), then Core's `LevenbergMarquardt` on Estimation's `CameraPosesProblem` over the
correspondences the start projects (a detection the nominal camera cannot project — past the lens's domain — is left out of the fit
and counted). Returns `camera` (the refined `CameraModel`), `poses` and `names` (the fitted views), `rms` [px], the refine's
termination record `retcode` (Core's `ReturnCode`) and `stop_rule`,
`iterations`, `n_correspondences`, `n_dropped`, the `start` (ξ, α) and the per-view rms `view_rms`.
"""
function _fit_start(model::AbstractString, views, W::Integer, H::Integer, f_mm::Real, sensor_mm::Real;
                   scoring::AbstractDict, max_iter::Integer, rng::AbstractRNG = default_rng())
    best = nothing
    for (xi, alpha) in (model == "double_sphere" ? DOUBLE_SPHERE_STARTS : ((0.0, 0.0),))
        cam0 = nominal_camera(model, W, H, f_mm, sensor_mm; xi, alpha)
        poses, kept = pose_starts(cam0, views; scoring, rng)
        length(kept) >= 2 || continue
        vs = [[c for c in views[k][2] if _projects(cam0, p, c[1], c[2])] for (k, p) in zip(kept, poses)]
        dropped = sum(length(views[k][2]) for k in kept) - sum(length, vs)
        ok = [length(v) >= 3 for v in vs]
        f = VgtkEstimation.CameraPosesProblem(vs[ok])
        d = default_stops(f)                                            # the problem's stops of record, the cap the caller's
        r = solve(RefineProblem(f, (; camera = cam0, poses = poses[ok])), LevenbergMarquardt(; optimization_strategy = nothing);
                  stops = LmStops(; max_iter = Int(max_iter), ftol = d.ftol, xtol = d.xtol))
        est = r.u
        names = [views[k][1] for k in kept[ok]]
        view_rms = [sqrt(sum(sum(abs2, VgtkCore.reprojection_error(est.camera, p, c[1], c[2])) for c in v) / length(v)) for (p, v) in zip(est.poses, vs[ok])]
        res = (; camera = est.camera, poses = est.poses, names, rms = r.rms, retcode = r.retcode, stop_rule = r.stop_rule, iterations = r.stats.nsteps,
               n_correspondences = sum(length, vs[ok]), n_dropped = dropped, start = (xi, alpha), view_rms)
        (best === nothing || res.rms < best.rms) && (best = res)
    end
    best === nothing && throw(ArgumentError("_fit_start: no start of the $model camera gives two views with a pose"))
    return best
end

"""
    write_start(path, camera, W, H; lens_coverage, provenance) -> path

The camera as the record a refinement starts from (RadiometricCamera's `save_lens`: vgio/cameras, K in COLMAP pixels, the lens
with its coefficients, the `W × H` extent bounded by the set's `lens_coverage` — the sensor's rectangle or its measured image region,
src/image_region.jl's `lens_coverage_region`, stated by the caller: no default), `provenance` a NamedTuple of labels.
"""
function write_start(path::AbstractString, camera::CameraModel, W::Integer, H::Integer; lens_coverage, provenance::NamedTuple)
    mkpath(dirname(path))
    RC.save_lens(path, BoundedCamera(camera, Size2(Int(W), Int(H)); lens_coverage); provenance)
    return path
end

"The parameters of a start camera by name (Core's parameter protocol), for reports and tests."
start_parameters(camera::CameraModel) = Dict(string(n) => Float64(v) for (n, v) in zip(VgtkCore.parameter_names(camera), VgtkCore.parameter_vector(camera)))
