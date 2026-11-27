# The calibration fixture's two-view scene (the fixture board, a 24 × 24 double-sphere camera with a focus surface and a tensor
# vignetting, two poses, a lighting and the renders under the material `mat`), shared by the scenes that build on it (PhotositeScene,
# SingleChannelScene). Included into a @testsetup that has the names below in scope (RadiometricCamera's, VisualRadiometryToolkitCore's
# vignetting and light, `exp_so3`, StaticArrays, `cam` and `fixture_board`).

"""`two_view_scene(mat; W = 24, H = 24)`: the fixture's board, render settings, camera (and its parts), two poses, lighting, the
compute masks and the renders under the material `mat`."""
function two_view_scene(mat_t; W = 24, H = 24)
    tex = fixture_board()
    board = RadiometricCamera.with_facing(TexturedTarget([-1.0, -1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], tex), [0.0, 0.0, 1.5])
    lens_t = cam(VgtkCore.Lens(VgtkCore.DoubleSphere(0.25, 0.4)), 1.288 * W, (0.51 * W, 0.48 * H), W, H)
    tps_t = FocusSpline(VgtkCore.grid_nodes(3, VgtkCore.Rect(0, 0, W, H)))
    focus_t = Focus(tps_t, [1 / (1.5 + 0.1 * sin(3i)) for i in 1:9])
    poses = [VgtkCore.EuclideanMap(SMatrix{3, 3}(exp_so3(SVector(π, 0.0, 0.0)) * exp_so3(SVector(0.15 * cos(k), 0.12 * sin(k), 0.1 * k))), SVector{3}(SVector(0.3 * cos(k), 0.3 * sin(k), 1.4 + 0.1 * k)); check = false) for k in 1:2]
    vig_t = RadiometricCamera.gauge_center_value(let v = TensorProductVignetting((4, 4), VgtkCore.Rect(0, 0, W, H); degree = 2); g = range(-0.5, 0.5, length = 4); with_control_points(v, [1.0 - 0.4 * (g[i]^2 + g[j]^2) for i in 1:4, j in 1:4]) end)
    camera_t = RadiometricCameraModel(lens_t, focus_t, 0.0386, vig_t)
    b = VgtkCore.centroid(board); r_SH = sphere_radius(b, [p.t for p in poses])
    a_t = zeros(3, 9); a_t[:, 1] .= (2.0, 1.9, 1.8); a_t[:, 3] .= (0.4, 0.3, 0.2); a_t[:, 4] .= (0.1, 0.15, 0.2)
    light_t = EnvironmentLight(SphericalHarmonicMap(a_t, 2), b, r_SH)
    st = RenderSettings(; n_𝐩 = 2, n_ρ = 2, n_φ = 6, N = 4)
    masks = [compute_mask(camera_t, p, board, st; δ = 0.02, backend = RadiometricCamera.KernelAbstractions.CPU()) for p in poses]
    renders = [render_view(k, camera_t, poses[k], mat_t, light_t, board, st, masks[k]) for k in 1:2]
    return (; board, st, lens_t, tps_t, focus_t, poses, camera_t, vig_t, mat_t, light_t, a_t, b, r_SH, masks, renders, W, H)
end
