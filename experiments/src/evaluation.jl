# =============================================================================
# src/evaluation.jl — module Evaluation: the evaluation steps of the drivers,
# ported from the experiments' scripts into functions (one call, files out):
#
#   poses_json_of_state     experiments/geometric_accuracy/detector_bias/state_poses_json.jl
#   truepose_scores         …/detector_bias/truepose_score.jl
#   reprojection_vectors    …/detector_bias/reproj_vectors.jl
#   truepose_vectors        …/detector_bias/reproj_vectors_truepose.jl
#
# The numbers, file names and JSON fields are the scripts' own, so the figures
# read the same records. Lens specs are RadiometricCamera's (`load_lens`:
# "grid:<record>" | "lens:<record>", VisualGeometryToolkitIO records; `load_model`: also "schoeps:"). Pixel
# convention: COLMAP px (Julia's 1-based pixel i spans [i − 1, i]), the cameras' own frame: a lens's π⁻¹ and projection take and give COLMAP pixels.
# The synthetic ground truth is a report (`gt/poses.json`: the board's true points, per view its true projections) beside its
# records (`gt/camera.json` the true lens, `gt/trajectory.json` the true poses); the poses of a run are trajectory records.
# Every setting the scripts carried as a constant is a keyword without a default: the drivers give their configs' values (marked
# there as inherited from the scripts).
# =============================================================================
module Evaluation

using StaticArrays: SVector, SMatrix
using LinearAlgebra: normalize, norm, dot, tr, I
using Statistics: median, mean, quantile
using Printf: @printf, @sprintf
import RadiometricCamera as RC
import VisualGeometryToolkitCore as VgtkCore
import VisualGeometryToolkitIO as VgtkIO
using VisualGeometryToolkitCore: angle_between
# residual_statistics: the one summary of a residual vector (src/residual_statistics.jl, shared with the prelude)
include("residual_statistics.jl")
include("record_schemas.jl")                  # the result records' schemas (one table, shared with the prelude)
using RadiometricCamera: load_lens, load_model, load_state

export poses_json_of_state, truepose_scores, direction_scores, reprojection_vectors, truepose_vectors

_readjson(path) = VgtkIO.read_record(path; allownan = true)   # a report may hold NaN / ±Infinity (every number then a Float64: the ids are Int(…) at use)
# The records beside a ground-truth report: `…/poses<s>.json` → `…/trajectory<s>.json`, and the true lens `camera.json`.
_gt_trajectory_path(gt) = joinpath(dirname(gt), replace(basename(gt), "poses" => "trajectory"))
_gt_camera_path(gt) = joinpath(dirname(gt), "camera.json")
# A trajectory record (world → camera per image, lengths in metres) as `image => (R, t)` with X_cam = R X_world + t.
function _trajectory(path)
    tr = VgtkIO.load(VgtkCore.Trajectory, path)
    return Dict(tr.images[k] => (p = VgtkCore.ustrip(VgtkCore.m, tr.poses[k]); (SMatrix{3, 3, Float64}(p.R), SVector{3, Float64}(p.t)))
                for k in keys(tr.poses))
end
# The rotation R with `b ≈ R a` over the vector pairs (Kabsch/Wahba, VisualGeometryToolkitCore's `Kabsch`; not centred).
_rotation(a, b) = SMatrix{3, 3, Float64}(VgtkCore.solve(VgtkCore.EstimationProblem(VgtkCore.Rotations.RotMatrix{3}, [SVector{3, Float64}(a[i]) => SVector{3, Float64}(b[i]) for i in eachindex(a)]),
                                                                VgtkCore.Kabsch()).u)
"""
    _forward(lens, X, 𝐩₀) -> SVector{2}

The COLMAP pixel whose chief ray through `lens` goes through the camera-frame point `X`: Core's projection of the camera's type
(RadiometricCamera's `_project_from`: a parametric lens's `project`; a SplineLens camera's batched `project` warm-started at `𝐩₀`),
NaN where it does not converge or lands outside the lens's domain.
"""
function _forward(lens, X, 𝐩₀)
    𝐩 = RC._project_from(lens, [SVector{3, Float64}(X)], [SVector{2, Float64}(𝐩₀)])[1]
    return all(isfinite, 𝐩) && RC.in_domain(lens, 𝐩) ? 𝐩 : SVector(NaN, NaN)
end

"""
    poses_json_of_state(state, out) -> out

The poses of a refinement state (`load_state`: the start state of `io.save_init`, a `state_it<N>.json`: RadiometricCamera's state record) as the
pipeline's poses record (RadiometricCamera's `save_poses`: VisualGeometryToolkitIO's trajectory record, world → camera, mm), for
[`reprojection_vectors`](@ref).
"""
function poses_json_of_state(state::AbstractString, out::AbstractString)
    st = load_state(state)
    mkpath(dirname(out))
    RC.save_poses(out, st.poses, String.(st.names); world = "the refinement's (from $(basename(state)))")
    return out
end

"""
    truepose_scores(gt, truth, lens; f_px = nothing) -> NamedTuple

GROUND-TRUTH scoring of an estimated lens on the synthetic photographs (`gt` = gt/poses.json): at every TRUE feature projection the
direction of the estimated `lens` (spec) is compared with the TRUE direction (the `truth` lens's direction at that pixel) — no pose
refit, nothing absorbed. RAW (the true poses as they are) and GAUGE (one global rotation of all rays removed by Kabsch): angle × f_px
in px, `(raw_median, raw_mean, raw_p95, gauge_median, gauge_mean, gauge_p95, rotation_mrad, dropped, n, f_px)`. `f_px` defaults to
the render's lens (the f_x of the ground truth's camera record, gt/camera.json).
"""
function truepose_scores(gt::AbstractString, truth::AbstractString, lens::AbstractString; f_px = nothing)
    J = _readjson(gt)
    tl = load_lens(truth)
    F = f_px === nothing ? (isfile(_gt_camera_path(gt)) ? Float64(VgtkCore.model(RC.load_parametric_lens(_gt_camera_path(gt))).intrinsics.K[1, 1]) :
                            error("$gt: no camera record beside it; give f_px")) : Float64(f_px)
    P = [SVector{2, Float64}(c.px_colmap[1], c.px_colmap[2]) for v in J.views for c in v.corners]
    dt = [normalize(VgtkCore.backproject(tl, p)) for p in P]
    l = load_lens(lens)
    return direction_scores(dt, [normalize(VgtkCore.backproject(l, p)) for p in P], F)
end

"""
    direction_scores(da, db, F) -> NamedTuple

Two models' unit directions at the same pixels (`da` the reference, `db` the other; a non-finite direction — a pixel the model does not
image — is dropped): their angle × `F` [px], RAW (as they are) and GAUGE (one global rotation of `db` removed by Kabsch: the camera
frame's gauge), `(raw_median, raw_mean, raw_p95, gauge_median, gauge_mean, gauge_p95, rotation_mrad, dropped, n, f_px)`. The one
comparison of two models' directions ([`truepose_scores`](@ref), the front ends' comparison with their starts of record).
"""
function direction_scores(da::AbstractVector, db::AbstractVector, F::Real)
    ok = [all(isfinite, a) && all(isfinite, b) for (a, b) in zip(da, db)]
    e0 = [F * angle_between(da[i], db[i]) for i in eachindex(da) if ok[i]]
    Rk = _rotation(db[ok], da[ok])                                                       # d_a ≈ Rk d_b
    e1 = [F * angle_between(da[i], Rk * db[i]) for i in eachindex(da) if ok[i]]
    return (; raw_median = median(e0), raw_mean = mean(e0), raw_p95 = quantile(e0, 0.95),
              gauge_median = median(e1), gauge_mean = mean(e1), gauge_p95 = quantile(e1, 0.95),
              rotation_mrad = 1e3 * VgtkCore.Rotations.rotation_angle(VgtkCore.Rotations.RotMatrix{3}(Rk)), dropped = count(!, ok), n = length(da), f_px = Float64(F))
end

# The board points of gt/poses.json by id.
_board_points(J) = Dict(Int(c.id) => SVector{3, Float64}(c.X_m...) for c in J.board.corners)

"""
    reprojection_vectors(gt, lens, poses, out; label = lens) -> NamedTuple

THE REPROJECTION ERROR WITH THE GIVEN POSES, no refit: every TRUE board point of `gt` projected through `lens` (spec) under the pose
of `poses` (a trajectory record: world → camera per image) by `_forward`; vector = predicted − true pixel at the true pixel. Pooled
over the views present in the poses record → `out` = `{label, lens, poses, n_views, x, y, dx, dy, view, id}` and the
`residual_statistics` of the errors (a wheel-map `scatter` panel). Returns that summary with `n_views`.
"""
function reprojection_vectors(gt::AbstractString, lens::AbstractString, poses::AbstractString, out::AbstractString; label::AbstractString = lens)
    J = _readjson(gt)
    l = load_lens(lens); X = _board_points(J)
    pose = _trajectory(poses)
    xs = Float64[]; ys = Float64[]; dxs = Float64[]; dys = Float64[]; vid = String[]; ids = Int[]; nv = 0
    for v in J.views
        nm = String(v.name); haskey(pose, nm) || continue; nv += 1; R_wc, t = pose[nm]
        for c in v.corners
            id = Int(c.id); haskey(X, id) || continue; xc = R_wc * X[id] + t; xc[3] > 0 || continue
            u0, v0 = Float64(c.px_colmap[1]), Float64(c.px_colmap[2])
            p = _forward(l, xc, SVector(u0, v0)); all(isfinite, p) || continue
            u, w = p; push!(xs, u0); push!(ys, v0); push!(dxs, u - u0); push!(dys, w - v0); push!(vid, nm); push!(ids, id)
        end
    end
    e = hypot.(dxs, dys)
    stats = merge(residual_statistics(e), (; n_views = nv))
    mkpath(dirname(out))
    VgtkIO.write_record(out, (; schema = EXPERIMENT_RECORD_SCHEMAS.residual_vectors, label, lens, poses, n_views = nv, x = xs, y = ys, dx = dxs, dy = dys, view = vid, id = ids, residual_statistics(e)...))
    return stats
end

"""
    truepose_vectors(gt, lens, out; label = lens, gauge = true) -> NamedTuple

THE REPROJECTION ERROR WITH THE TRUE POSES: every true board point of `gt` projected through `lens` under the TRUE pose of its view
(the ground truth's trajectory record), after removing ONE global rotation shared by all views (the frame gauge of a generic lens;
Kabsch between the lens's bearings of the true pixels and the true bearings; `gauge = false` keeps the raw frame) — the lens error
alone, as pixel vectors at the true pixels → `out` (a wheel-map `scatter` panel, with `gauge_mrad`). Returns its
`residual_statistics` with `n_views` and `gauge_mrad`.
"""
function truepose_vectors(gt::AbstractString, lens::AbstractString, out::AbstractString; label::AbstractString = lens, gauge::Bool = true)
    J = _readjson(gt)
    l = load_lens(lens); X = _board_points(J); truth = _trajectory(_gt_trajectory_path(gt))
    xtrue = SVector{3, Float64}[]; dlens = SVector{3, Float64}[]; pxs = SVector{2, Float64}[]; vid = String[]; ids = Int[]; nv = 0
    for v in J.views
        R_wc, t = truth[String(v.name)]
        nv += 1
        for cr in v.corners
            id = Int(cr.id); haskey(X, id) || continue; xc = R_wc * X[id] + t; xc[3] > 0 || continue
            p0 = SVector{2, Float64}(cr.px_colmap[1], cr.px_colmap[2]); dl = normalize(VgtkCore.backproject(l, p0)); all(isfinite, dl) || continue
            push!(xtrue, xc); push!(dlens, dl); push!(pxs, p0)
            push!(vid, String(v.name)); push!(ids, id)
        end
    end
    Rg = gauge ? _rotation(dlens, normalize.(xtrue)) : SMatrix{3, 3, Float64}(I)      # d_true ≈ Rg d_lens
    rot_mrad = 1e3 * VgtkCore.Rotations.rotation_angle(VgtkCore.Rotations.RotMatrix{3}(Rg))
    xs = Float64[]; ys = Float64[]; dxs = Float64[]; dys = Float64[]; vids = String[]; idd = Int[]
    for i in eachindex(xtrue)
        p = _forward(l, Rg' * xtrue[i], pxs[i]); all(isfinite, p) || continue
        u, w = p; push!(xs, pxs[i][1]); push!(ys, pxs[i][2]); push!(dxs, u - pxs[i][1]); push!(dys, w - pxs[i][2]); push!(vids, vid[i]); push!(idd, ids[i])
    end
    e = hypot.(dxs, dys)
    stats = merge(residual_statistics(e), (; n_views = nv, gauge_mrad = rot_mrad))
    mkpath(dirname(out))
    VgtkIO.write_record(out, (; schema = EXPERIMENT_RECORD_SCHEMAS.residual_vectors, label, lens, poses = "true poses" * (gauge ? " (one global rotation removed)" : ""), n_views = nv,
                              x = xs, y = ys, dx = dxs, dy = dys, view = vids, id = idd, residual_statistics(e)..., gauge_mrad = rot_mrad))
    return stats
end

end # module Evaluation
