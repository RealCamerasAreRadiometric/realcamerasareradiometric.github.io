# ---- front-ends ----------------------------------------------------------------------------------------------------------------
"""Focal length in px for the bearing → px conversion and the image-distance start: the `f_x` of the camera's pinhole (K, or a grid
model's nominal) unless the model's angular pixel pitch at the image centre disagrees with it by > 1 % — the (ξ, f) pair of a fitted
DoubleSphere is degenerate on the axis (f_eff ≈ f/(1 + ξ)), so there the pitch is the focal length."""
function fpx_of(lens, W, H)
    fn = VgtkCore.calibration_matrix(VgtkCore.model(lens))[1, 1]
    c = SVector(W / 2, H / 2)
    d0 = normalize(backproject(lens, c)); d1 = normalize(backproject(lens, c + SVector(1.0, 0.0)))
    fe = 1 / angle_between(d0, d1)
    abs(fe / fn - 1) > 0.01 && @printf("  focal length: f_x·W = %.1f px, angular pixel pitch at the centre %.1f px → using the pitch\n", fn, fe)
    return abs(fe / fn - 1) > 0.01 ? fe : fn
end
"""Bearing error [px] of detections `feats = [(𝐩, X)]` through `lens` under `pose`: the angle between the model's ray of 𝐩 and the direction
to the board point X, times the focal length in px."""
bearing_px(lens, pose, feats, fpx) = [fpx * angle_between(normalize(backproject(lens, 𝐩)), normalize(pose.R' * (X - pose.t))) for (𝐩, X) in _imaged(lens, feats)]
"""The detections `feats = [(𝐩, X)]` the lens images: those with a ray (Core's `backproject` finite). A parametric lens has none
outside its domain (past the image circle or the fold); a grid lens none outside its present windows (an absent node's window, off
its lattice) — so a bearing statistic counts the detections ITS lens images: the start's (`be0`) and a refined lens's may differ."""
_imaged(lens, feats) = [f for f in feats if all(isfinite, backproject(lens, f[1]))]
"""
    detection_bearings(lens, poses, features, fpx = 1.0) -> Vector{Float64}

The bearing errors of every view's detections, `features[k] = [(𝐩, X)]` under `poses[k]`, through `lens` ([`bearing_px`](@ref) per
view, concatenated in view order): in radians with `fpx = 1`, in px with the focal length `fpx` — the refinement monitor's
`bearing(detections, own poses)` (its px are `fpx ×` the radians, the same product per detection).
"""
detection_bearings(lens, poses, features, fpx::Real = 1.0) =
    reduce(vcat, [bearing_px(lens, poses[k], features[k], fpx) for k in eachindex(features)]; init = Float64[])
"""
    state_bearings(geometry::StartGeometry, state) -> Vector{Float64}

The bearing errors [rad] of the start geometry's detections through a refinement state's camera under that state's own poses, the
target points placed on the state's own surface (its warping and in-plane print error, the geometry's `features_for`): the quantity the
refinement's monitor prints as `bearing(detections, own poses)` at the iteration of `state` (`fpx ×` these, in px). `state` is a
state of [`load_state`](@ref) — a checkpoint `state_it<N>.json` (`io.checkpoint_each`) or a start record's `state.json` (iteration
0); its views are matched to the geometry's by name (an `ArgumentError` naming a view it lacks).
"""
function state_bearings(g::StartGeometry, st)
    idx = [findfirst(==(n), st.names) for n in g.names]
    any(isnothing, idx) && throw(ArgumentError("state_bearings: the state has no view $(join(g.names[isnothing.(idx)], ", ")) of the start geometry"))
    feats = _board_features(g, board_surface(g.target, st.target), g.observations)
    return detection_bearings(st.camera.geometric, st.poses[idx], feats)
end

# ---- the self-check and its outputs --------------------------------------------------------------------------------------------
"""
    project_points(camera, Xcs, 𝐩₀s) -> Vector{SVector{2, Float64}}

The sensor positions (COLMAP pixels) of the camera-frame points `Xcs` through the camera, by Core's projection of its type: a
SplineLens camera's batched `project` of its model, warm-started at the detections `𝐩₀s` (NaN where it does not converge onto a
present window); a parametric pixel camera's closed-form `project` (NaN where it does not image the point; no seed read).
"""
project_points(m::CameraModel{<:VgtkCore.AbstractIntrinsics, <:SplineLens}, Xcs, 𝐩₀s) =
    Vector{SVector{2, Float64}}(VgtkCore.project(m, Xcs, 𝐩₀s; device = KernelAbstractions.CPU()).px)
project_points(g::BoundedCamera{<:CameraModel{<:VgtkCore.AbstractIntrinsics, <:SplineLens}}, Xcs, 𝐩₀s) = project_points(VgtkCore.model(g), Xcs, 𝐩₀s)
project_points(g::_PixelCamera, Xcs, _) = [SVector{2, Float64}(project(g, X)) for X in Xcs]

"""Residual VECTORS in image px: from each detection to the model's prediction — the projection of the board point under the pose
through the lens (`project_points`, Core's projections), started at the detection."""
function residual_vectors(lens, pose, feats)
    q = project_points(lens, [SVector{3, Float64}(inv(pose)(X)) for (_, X) in feats], [SVector{2, Float64}(𝐩) for (𝐩, _) in feats])
    return NTuple{4, Float64}[all(isfinite, qk) ? (𝐩[1], 𝐩[2], qk[1] - 𝐩[1], qk[2] - 𝐩[2]) : (𝐩[1], 𝐩[2], NaN, NaN) for ((𝐩, _), qk) in zip(feats, q)]
end
"""The same residual on the BOARD: the detection's ray (lens + pose) intersected with the board plane, in chart units, next to the chart
of the board point it was matched to."""
function board_hits(lens, pose, feats, board)
    out = NTuple{4, Float64}[]
    for (𝐩, X) in feats
        d_w = pose.R * normalize(backproject(lens, 𝐩)); o_w = pose.t
        s, 𝐱 = board_plane_hit(board, o_w, d_w); b_hit = chart(board, 𝐱); b_pt = chart(board, X)
        push!(out, (b_pt[1], b_pt[2], isfinite(s) && s > 0 ? b_hit[1] - b_pt[1] : NaN, isfinite(s) && s > 0 ? b_hit[2] - b_pt[2] : NaN))
    end
    return out
end
"""`selfcheck.only`: the bearing error of the detections through the start model (per view, under the refit poses or those of
the `selfcheck.poses` state) → selfcheck.json, selfcheck_vectors.json."""
function _write_selfcheck(g::StartGeometry, run::RunOptions, fpx, be0, load_state)
    names = g.names; K = length(names); poses0 = copy(g.poses); lens0 = g.camera
    features = [in_initialization(f, g.init_region) for f in g.observations]   # the detections in the start's initialization region
    if !isempty(run.selfcheck.poses)
        sst = load_state(run.selfcheck.poses); sidx = [findfirst(==(n), sst.names) for n in names]
        any(isnothing, sidx) && error("selfcheck.poses: views not in the state: $(names[isnothing.(sidx)])")
        for k in 1:K
            p = sst.poses[sidx[k]]; e0 = median(bearing_px(lens0, poses0[k], features[k], fpx))
            poses0[k] = _pose_eltype(Float64, p)
            @printf("  %s: pose of the state %s instead of the refit, bearing median %.3f → %.3f px\n", names[k], basename(dirname(run.selfcheck.poses)), e0, median(bearing_px(lens0, poses0[k], features[k], fpx)))
        end
        flush(stdout)
    end
    pv = [bearing_px(lens0, poses0[k], features[k], fpx) for k in 1:K]
    for k in 1:K; @printf("  %-14s n %5d  median %.3f  p90 %.3f  mean %.3f px\n", names[k], length(pv[k]), median(pv[k]), quantile(pv[k], 0.9), mean(pv[k])); end
    rv = [residual_vectors(lens0, poses0[k], features[k]) for k in 1:K]
    lv = reduce(vcat, [[hypot(r[3], r[4]) for r in v] for v in rv]); nbad = count(!isfinite, lv)
    @printf("residual vectors: median length %.3f px (bearing median %.3f px); %d of %d detections not converged (left out)\n",
            median(filter(isfinite, lv)), median(be0), nbad, length(lv))
    bh = [board_hits(lens0, poses0[k], features[k], g.target) for k in 1:K]
    fin(v) = isfinite(v) ? v : nothing
    VgtkIO.write_record(joinpath(run.io.out, "selfcheck_vectors.json"),
        (; schema = RECORD_SCHEMAS.selfcheck_vectors, init = run.io.init, views = [(; name = names[k], x = [r[1] for r in rv[k]], y = [r[2] for r in rv[k]], dx = [fin(r[3]) for r in rv[k]],
                                           dy = [fin(r[4]) for r in rv[k]], e = pv[k],
                                           bu = [r[1] for r in bh[k]], bv = [r[2] for r in bh[k]],
                                           du = [fin(r[3]) for r in bh[k]], dv = [fin(r[4]) for r in bh[k]]) for k in 1:K]))
    VgtkIO.write_record(joinpath(run.io.out, "selfcheck.json"),
        (; schema = RECORD_SCHEMAS.selfcheck, init = run.io.init, schoeps = run.frontend.schoeps, views = names, n = length.(pv), median = median.(pv), p90 = [quantile(v, 0.9) for v in pv], mean = mean.(pv),
           all = (; n = length(be0), median = median(be0), p90 = quantile(be0, 0.9), mean = mean(be0))))
    println("self-check only: written to ", joinpath(run.io.out, "selfcheck.json")); flush(stdout)
    return nothing
end
