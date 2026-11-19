"""The world points `Xs` => the camera-frame unit rays `vs` observing them: the
bearing correspondences of VisualGeometryToolkitEstimation's `refine_absolute_pose(pose, cs)` — the tangent-plane angle residual on
Core's `LeftRigidManifold6`, least squares over every ray (`optimization_strategy = nothing`), the pose world → camera."""
function _ray_correspondences(Xs, vs)
    src = [SVector{3, Float64}(X) for X in Xs]; dst = [normalize(SVector{3, Float64}(v)) for v in vs]
    return VgtkCore.Correspondences(src, dst)
end

"""
    _fit_page_pose(Xs, vs, 𝐨, 𝐧, bc, distances) -> (e, R, t)

The world → camera pose `(R, t)` of a view of a planar page through `𝐨` whose printed side's normal is `𝐧`, from the page points
`Xs` and their camera-frame rays `vs` (Estimation's `refine_absolute_pose` on [`_ray_correspondences`](@ref)): a multi-start over `distances` (along the page's normal, towards
the printed side, from the page centre `bc`) and 5 tilts, the camera always on the printed side. A refined pose whose camera centre
`−Rᵀt` is not on that side (Core's `signed_distance` to the page plane `≤ 0`: cheirality) is refused. `e` is the median bearing
angle of the best pose (rad); `e = Inf` when no start gives a pose on the printed side.
"""
function _fit_page_pose(Xs, vs, 𝐨, 𝐧, bc, distances)
    page = VgtkCore.Plane3D(VgtkCore.Point3(𝐨...), VgtkCore.Vec3(𝐧...))
    # the start rotation: the camera's +z (its optical axis) along −𝐧, towards the page (𝐧 = −z: the identity; 𝐧 = +z: 180° about x)
    Rface = 𝐧[3] < 0 ? SMatrix{3, 3, Float64}(I) : SMatrix{3, 3, Float64}(Diagonal(SVector(1.0, -1.0, -1.0)))
    best = (Inf, SMatrix{3, 3, Float64}(I), SVector(0.0, 0.0, 1.0))
    for d in distances, (rx, ry) in ((0.0, 0.0), (0.5, 0.0), (-0.5, 0.0), (0.0, 0.5), (0.0, -0.5))
        R0 = exp_so3(SVector(rx, ry, 0.0)) * Rface; t0 = SVector(0.0, 0.0, d) - R0 * bc
        est = VgtkCore.estimate(VgtkEstimation.refine_absolute_pose(EuclideanMap(R0, t0; check = false), _ray_correspondences(Xs, vs); optimization_strategy = nothing))
        R, t = SMatrix{3, 3, Float64}(est.R), SVector{3, Float64}(est.t)
        VgtkCore.signed_distance(page, VgtkCore.Point3((-R' * t)...)) > 0 || continue   # cheirality: the camera sees the printed side
        e = median(angle_between(normalize(R * X + t), v) for (X, v) in zip(Xs, vs))
        isfinite(e) && e < best[1] && (best = (e, R, t))
    end
    return best
end
