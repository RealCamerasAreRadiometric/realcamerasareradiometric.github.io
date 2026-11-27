@testitem "calibration" setup = [RCFixtures] begin
    using RadiometricCamera: all_parts, aperture_radius, build_problem, compute_mask, cost, focus, Focus, FocusSpline, Material,
                             PerViewResidual, TexturedTarget, render_view, RenderSettings, _lm_views!, _lm_radiometry!,
                             _update_lighting!, _update_vignetting!, _update_diffuse_weight!, sphere_radius, with_nonlinear, with_vignetting, with_μ, 𝔟
    using StaticArrays, LinearAlgebra, Statistics, ForwardDiff
    using VisualGeometryToolkitCore: exp_so3
    using VisualGeometryToolkitCore: center
    using VisualRadiometryToolkitCore: TensorProductVignetting, RadialVignetting, with_control_points
    import VisualGeometryToolkitCore as VgtkCore
    import VisualRadiometryToolkitCore as VrtkCore
    using Rotations: rotation_angle, RotMatrix

    tex = fixture_board()
    board = TexturedTarget([-1.0, -1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], tex; facing = [0.0, 0.0, 1.5])
    W = H = 24
    lens_t = cam(VgtkCore.Lens(VgtkCore.DoubleSphere(0.25, 0.4)), 30.912, (12.24, 11.52), W, H)
    tps_t = FocusSpline(VgtkCore.grid_nodes(3, VgtkCore.Rect(0, 0, W, H)))
    focus_t = Focus(tps_t, [1 / (1.5 + 0.1 * sin(3i)) for i in 1:9])
    r_t = 0.0386
    poses = [VgtkCore.EuclideanMap(SMatrix{3, 3}(exp_so3(SVector(π, 0.0, 0.0)) * exp_so3(SVector(0.15 * cos(k), 0.12 * sin(k), 0.1 * k))), SVector{3}(SVector(0.3 * cos(k), 0.3 * sin(k), 1.4 + 0.1 * k)); check = false) for k in 1:2]
    # the truth field at V(𝐜) = 1, the pin every state holds (2026-11-11: no `pin_center = false`): a solve recovers it exactly
    vig_t = RC.gauge_center_value(let v = TensorProductVignetting((4, 4), VgtkCore.Rect(0, 0, W, H); degree = 2); gr = range(-0.5, 0.5, length = 4); with_control_points(v, [1.0 - 0.4 * (gr[i]^2 + gr[j]^2) for i in 1:4, j in 1:4]) end)
    camera_t = RadiometricCameraModel(lens_t, focus_t, r_t, vig_t)          # the true camera; a start's camera is it with another vignetting
    b = 𝔟(board); r_SH = sphere_radius(b, [p.t for p in poses])
    a_t = zeros(3, 9); a_t[:, 1] .= (2.0, 1.9, 1.8); a_t[:, 3] .= (0.4, 0.3, 0.2); a_t[:, 4] .= (0.1, 0.15, 0.2)
    light_t = EnvironmentLight(SphericalHarmonicMap(a_t, 2), b, r_SH)
    mat_t = Material(; roughness_ink = 0.12, roughness_paper = 0.35, refractive_index_ink = 1.65, refractive_index_paper = 1.33, diffuse_weight = 0.4)
    st = RenderSettings(; n_𝐩 = 2, n_ρ = 2, n_φ = 6, N = 4)
    masks = [compute_mask(camera_t, p, board, st; δ = 0.02, backend = RadiometricCamera.KernelAbstractions.CPU()) for p in poses]
    targets = [render_view(k, camera_t, poses[k], mat_t, light_t, board, st, masks[k]) for k in 1:2]
    # single linear-block steps at the truth of everything else are exact
    s = calibration_problem(board, st, camera_t, poses, mat_t, EnvironmentLight(SphericalHarmonicMap(0.5, 2), b, r_SH), targets; masks)
    _update_lighting!(s.f, s.u0); @test maximum(abs, Matrix(coefficients(s.u0.light)) - a_t) < 1e-8
    s = calibration_problem(board, st, with_vignetting(camera_t, TensorProductVignetting((4, 4), VgtkCore.Rect(0, 0, W, H); degree = 2)), poses, mat_t, light_t, targets; masks)
    _update_vignetting!(s.f, s.u0); @test maximum(abs, Matrix(s.u0.camera.vignetting.P) - Matrix(vig_t.P)) < 1e-6
    # under an imaged region (a circle of radius 7 px about the sensor centre) the tensor surface's nodes outside it are
    # dependent (Core's BoundaryRule through the surface's support trait): the solve's unknowns are the interior nodes, one column each
    reg = VgtkCore.Rect(0, 0, W, H) ∩ VgtkCore.Ellipse(VgtkCore.Point2(12.0, 12.0), 7.0, 7.0, 0.0)
    s = calibration_problem(board, st, with_vignetting(camera_t, TensorProductVignetting((6, 6), VgtkCore.Rect(0, 0, W, H); degree = 2)), poses, mat_t, light_t, targets; masks, region = reg)
    rule = s.f.vig_rule
    @test rule.interior == VgtkCore.node_mask(s.u0.camera.vignetting, reg) && !rule.interior[1, 1] && rule.interior[3, 3] && !isempty(rule.dependent)
    @test size(RC.vignetting_basis(s.f, s.u0)) == (36, count(rule.interior) - 1) && Int(RC.NLLSsolver.nvars(VgtkCore.restrict(VgtkCore.ParameterManifold(s.u0.camera.vignetting), RC.vignetting_basis(s.f, s.u0)))) == count(rule.interior) - 1   # the pin V(𝐜) = 1 takes one direction, always
    P0 = Matrix(s.u0.camera.vignetting.P); _update_vignetting!(s.f, s.u0); P1 = Matrix(s.u0.camera.vignetting.P)
    @test P1 ≈ VgtkCore.extend_controls(rule, P1, P0) && P1[rule.interior] != P0[rule.interior]
    s = calibration_problem(board, st, camera_t, poses, with_μ(mat_t, 0.8), light_t, targets; masks)
    _update_diffuse_weight!(s.f, s.u0); @test s.u0.mat.diffuse_weight ≈ 0.4 atol = 1e-8
    # the JOINT closed-form radiometry (2026-10-13): from a flat V, a wrong grey light and a wrong μ the joint Levenberg–Marquardt on the
    # parts recovers V (up to the V × lighting gauge of the centre pin), the lighting and μ exactly — the block solves creep along V × light
    s = calibration_problem(board, st, with_vignetting(camera_t, TensorProductVignetting((4, 4), VgtkCore.Rect(0, 0, W, H); degree = 2)), poses, with_μ(mat_t, 0.7), EnvironmentLight(SphericalHarmonicMap(0.5, 2), b, r_SH), targets; masks)
    parts_j = all_parts(s.f, s.u0); _lm_radiometry!(s.f, s.u0, parts_j; damping = VgtkCore.LevenbergDamping(), stops = VgtkCore.LmStops(; max_iter = 80)); cj = RC.parts_cost(s.f, s.u0, parts_j)
    @test cj < 1e-16 * sum(sum(abs2, v) for v in s.f.ϕ)
    ratio_j = [s.u0.camera.vignetting(SVector(x, y)) / vig_t(SVector(x, y)) for x in 1.2:3.6:22.8 for y in 1.2:3.6:22.8]
    @test maximum(ratio_j) / minimum(ratio_j) - 1 < 1e-6
    @test s.u0.mat.diffuse_weight ≈ 0.4 atol = 1e-6
    # Jacobian of the per-view residual vs finite differences (the pin V(𝐜) = 1 held exactly: the centre's direction is no column)
    # (the state's V is the truth's normalized to V(𝐜) = 1, the gauge: the truth's lighting carries the V(𝐜), so the residual vanishes)
    s = calibration_problem(board, st, camera_t, poses, mat_t, with_coefficients(light_t, coefficients(light_t) .* vig_t(center(vig_t))), targets; masks)
    prob, idx = build_problem(s.f, s.u0)
    res = filter(c -> c isa PerViewResidual{9}, prob.blocks)[1]
    vars = RC.NLLSsolver.getvars(res, prob.variables)
    r0 = RC.NLLSsolver.computeresidual(res, vars...)
    @test maximum(abs, r0) < 1e-12                                     # at the truth the residual vanishes
    rj, J = RC.NLLSsolver.computeresjac(RC.static(127), res, vars...)
    @test size(J) == (length(r0), 6 + 6 + 9 + 1 + 9 + 15 + 27)       # pose | dist | focus | r | material 8 + μ | vig 15 (the pin's direction eliminated) | light 3×9
    @test isapprox(rj, r0; atol = 1e-10)
    pv, dv, fv, av, mv, vv, lv = vars
    u6(i, h) = (z = zeros(6); z[i] = h; z)
    for (col, setter) in ((2, h -> (RC.NLLSsolver.update(pv, u6(2, h)), dv, fv, av, mv, vv, lv)),
                          (8, h -> (pv, RC.NLLSsolver.update(dv, u6(2, h)), fv, av, mv, vv, lv)),
                          (14, h -> (pv, dv, RC.NLLSsolver.update(fv, unit_increment(9, 2, h)), av, mv, vv, lv)),
                          (24, h -> (pv, dv, fv, av, RC.NLLSsolver.update(mv, unit_increment(9, 2, h)), vv, lv)),
                          (31, h -> (pv, dv, fv, av, RC.NLLSsolver.update(mv, unit_increment(9, 9, h)), vv, lv)),        # μ
                          (33, h -> (pv, dv, fv, av, mv, RC.NLLSsolver.update(vv, unit_increment(15, 2, h)), lv)),      # vignetting (free direction 2)
                          (50, h -> (pv, dv, fv, av, mv, vv, RC.NLLSsolver.update(lv, unit_increment(27, 4, h)))))      # lighting a[1, 2]
        fd = (RC.NLLSsolver.computeresidual(res, setter(1e-6)...) - RC.NLLSsolver.computeresidual(res, setter(-1e-6)...)) / 2e-6
        @test norm(J[:, col] - fd) / norm(fd) < 2e-3        # bilinear texture: FD may straddle texel-cell boundaries
    end
    # the HARD pin (2026-10-30, the default): the centre's free direction is eliminated (`gauge_basis`) — one V column fewer, no pin
    # residual, the field normalised to V(𝐜) = 1 at construction and kept there by every step
    sh = calibration_problem(board, st, camera_t, poses, mat_t, light_t, targets; masks)
    @test !hasfield(typeof(sh.f), :pin_center) && !hasfield(typeof(sh.u0), :pin_center)   # always held: no switch
    @test sh.u0.camera.vignetting(center(sh.u0.camera.vignetting)) ≈ 1 atol = 1e-12
    prob_h, idx_h = build_problem(sh.f, sh.u0); vv_h = prob_h.variables[idx_h.vignetting]
    @test Int(RC.NLLSsolver.nvars(vv_h)) == 15
    @test VgtkCore.model(RC.NLLSsolver.update(vv_h, randn(15)))(center(sh.u0.camera.vignetting)) ≈ 1 atol = 1e-12
    res_h = filter(c -> c isa PerViewResidual{9}, prob_h.blocks)[1]; vars_h = RC.NLLSsolver.getvars(res_h, prob_h.variables)
    _, Jh = RC.NLLSsolver.computeresjac(RC.static(127), res_h, vars_h...); @test size(Jh, 2) == 6 + 6 + 9 + 1 + 9 + 15 + 27
    fdh = (RC.NLLSsolver.computeresidual(res_h, vars_h[1:5]..., RC.NLLSsolver.update(vars_h[6], unit_increment(15, 2, 1e-6)), vars_h[7]) -
           RC.NLLSsolver.computeresidual(res_h, vars_h[1:5]..., RC.NLLSsolver.update(vars_h[6], unit_increment(15, 2, -1e-6)), vars_h[7])) / 2e-6
    @test norm(Jh[:, 33] - fdh) / norm(fdh) < 2e-3
    # joint LM recovery from a perturbed start of EVERY block (vignetting, lighting and μ included)
    vig_p = let v = TensorProductVignetting((4, 4), VgtkCore.Rect(0, 0, W, H); degree = 2); gr = range(-0.5, 0.5, length = 4); with_control_points(v, [1.0 - 0.2 * (gr[i]^2 + gr[j]^2) for i in 1:4, j in 1:4]) end
    a_p = 0.8 .* a_t; a_p[:, 4] .= 0.0
    s2 = calibration_problem(board, st, RadiometricCameraModel(cam(VgtkCore.Lens(VgtkCore.DoubleSphere(0.15, 0.5)), 30.0, (12.0, 12.0), W, H), Focus(tps_t, fill(1 / 1.7, 9)), 0.03, vig_p),
                          [VgtkCore.retract(VgtkCore.LeftRigidManifold6(p), vcat(SVector(0.01, -0.008, 0.012), SVector(0.01, -0.01, 0.015))) for p in poses],
                          with_μ(with_nonlinear(mat_t, SVector(0.2, 0.3, 1.5, 1.5, 1.5, 1.4, 1.4, 1.4)), 0.55), EnvironmentLight(SphericalHarmonicMap(a_p, 2), b, r_SH), targets; masks)
    c0 = cost(s2.f, s2.u0)[1]
    # The joint LM recovers EVERY block of this two-view fixture to machine precision (≈ 90 iterations with the relative stopping test at
    # 1e-12: cost/c0 2e-26, r and 𝛂 to 1e-14, μ and the lighting to 1e-12). The (aperture radius, μ, lighting, focus) direction is flat but not
    # degenerate: with the absolute cost-decrease test at 1e-12 (the former default) the solver stopped at a cost of ≈ 2e-12 with r −8 %,
    # μ +0.09 and the light 3.5 % off, which looked like an identifiability limit and was the stopping test.
    result = _lm_views!(s2.f, s2.u0; stops = RC._calibration_stops(; max_iter = 200, ftol = 1e-12), damping = VgtkCore.LevenbergDamping())
    @test result.metadata.cost / 2 < 1e-20 * c0          # (Core's record: ‖r‖²; `cost` ½‖r‖²)
    @test aperture_radius(s2.u0) ≈ r_t rtol = 1e-9
    @test pinhole_θ(RC.geometric(s2.u0))[3] ≈ pinhole_θ(lens_t)[3] atol = 1e-9 * W      # (`lens` is shadowed by a test variable)
    @test pinhole_θ(RC.geometric(s2.u0))[4] ≈ pinhole_θ(lens_t)[4] atol = 1e-9 * H
    @test all(rad2deg(rotation_angle(RotMatrix{3}(s2.u0.poses[k].R' * poses[k].R))) < 1e-8 for k in 1:2)
    @test maximum(abs, focus(s2.u0).t̂_i - focus_t.t̂_i) < 1e-9
    @test s2.u0.mat.roughness_ink ≈ 0.12 atol = 1e-9
    @test s2.u0.mat.roughness_paper ≈ 0.35 atol = 1e-9
    @test s2.u0.mat.diffuse_weight ≈ 0.4 atol = 1e-9
    # V·L is recovered up to the gauge: the hard pin holds V(𝐜) = 1 exactly (the truth is not pinned)
    g_t = vig_t(center(vig_t)); g_2 = s2.u0.camera.vignetting(center(s2.u0.camera.vignetting))
    @test g_2 ≈ 1 atol = 1e-6
    @test maximum(abs, Matrix(s2.u0.camera.vignetting.P) ./ g_2 .- Matrix(vig_t.P) ./ g_t) < 1e-9
    @test maximum(abs, Matrix(coefficients(s2.u0.light)) .* g_2 .- a_t .* g_t) / maximum(abs, a_t) < 1e-9
    # rounds of the joint LM (`_lm_views!`, the per-view route) from wrong everything: must decrease the cost monotonically (the retired
    # `calibrate!` loop's test, ported: its rounds of n_inner iterations until a relative cost change below 1e-4)
    s3 = calibration_problem(board, st, RadiometricCameraModel(cam(VgtkCore.Lens(VgtkCore.DoubleSphere(0.2, 0.45)), 30.24, (12.0, 12.0), W, H), Focus(tps_t, fill(1 / 1.6, 9)), 0.035,
                                                            TensorProductVignetting((4, 4), VgtkCore.Rect(0, 0, W, H); degree = 2)),
                          [VgtkCore.retract(VgtkCore.LeftRigidManifold6(p), vcat(SVector(0.005, -0.004, 0.006), zeros(SVector{3}))) for p in poses],
                          Material(), EnvironmentLight(SphericalHarmonicMap(1.0, 2), b, r_SH), targets; masks)
    hist = [cost(s3.f, s3.u0)[1]]
    for _ in 1:3
        _lm_views!(s3.f, s3.u0; stops = RC._calibration_stops(; max_iter = 4), damping = VgtkCore.LevenbergDamping())
        push!(hist, cost(s3.f, s3.u0)[1])
        abs(hist[end - 1] - hist[end]) <= 1e-4 * hist[end - 1] && break
    end
    @test length(hist) >= 2
    @test hist[end] < 0.05 * hist[1]
    @test all(hist[i] <= hist[i - 1] * (1 + 1e-9) for i in 2:length(hist))
    # the lens block by Core's parameter names, and the focal tie (the aspect fixed): Core's ParameterManifold of the pixel camera, f on
    # its log chart (f ← f·e^δ) — the double sphere's ξ on its log chart around −1, α (closed ends, Core 6401735) on the plain chart
    dv = RC.lens_manifold(lens_t; fixed = (:cx, :cy, :aspect))
    @test free_names(dv) == (:f, :xi, :alpha) && Int(RC.NLLSsolver.nvars(dv)) == 3
    l2 = block_camera(RC.NLLSsolver.update(dv, [0.01, 0.02, -0.03]), lens_t)
    θ0, θ2 = pinhole_θ(lens_t), pinhole_θ(l2)
    @test θ2[1] ≈ θ0[1] * exp(0.01) && θ2[2] ≈ θ0[2] * exp(0.01)                                  # tied: f_y/f_x kept
    @test θ2[3] == θ0[3] && θ2[4] == θ0[4]                                                        # fixed
    ξ0, α0 = VgtkCore.parameter_vector(VgtkCore.lens(lens_t)); ξ2, α2 = VgtkCore.parameter_vector(VgtkCore.lens(l2))
    @test ξ2 ≈ -1 + (ξ0 + 1) * exp(0.02) && α2 ≈ α0 - 0.03                                    # α's ends are closed: the plain chart (a mirror at 0 and 1)
    probm, idxm = build_problem(s.f, s.u0; fixed = (:cx, :cy, :aspect))
    resm = filter(c -> c isa PerViewResidual{9}, probm.blocks)[1]
    varsm = RC.NLLSsolver.getvars(resm, probm.variables)
    _, Jm = RC.NLLSsolver.computeresjac(RC.static(31), resm, varsm...)
    @test size(Jm, 2) == 6 + 3 + 9 + 1 + 9 && !any(iszero, [norm(Jm[:, 6 + j]) for j in 1:3])    # three lens columns, all moving the flux
    @test norm(Jm[:, 7]) > 0 && norm(Jm[:, 11]) > 0
    # trace recording
    s5 = calibration_problem(board, st, camera_t, poses, mat_t, EnvironmentLight(SphericalHarmonicMap(0.5, 2), b, r_SH), targets; masks)
    trc = NamedTuple[]
    _lm_views!(s5.f, s5.u0; stops = RC._calibration_stops(; max_iter = 2), damping = VgtkCore.LevenbergDamping(), trace = trc, trace_meta = (; stage = 1, factor = 1, ϕ² = sum(sum(abs2, p) for p in s5.f.ϕ)))
    @test !isempty(trc) && all(t.phase == :lm for t in trc)
    @test haskey(trc[end].snap, :camera) && length(trc[end].snap.poses) == 2
    @test all(isfinite(sqrt(2t.cost / t.ϕ²)) for t in trc)
    # radial vignetting profile (2026-10-21): the closed-form V solve and the joint radiometry recover a radial truth exactly (the
    # quadratic has V′(0) = 0, so its coefficients satisfy the pole tie P_0 = P_1), the LM problem builds with the (N − 1)-dimensional
    # tied block, and the tensor interface (control_points as N × 1) is enough everywhere
    𝐜r = SVector(12.72, 11.52); rmax_r = 0.55 * W
    vig_r = RC.gauge_center_value(with_control_points(RadialVignetting(5, 𝐜r, rmax_r), let kn = VgtkCore.clamped_knots(5, 3, 0.0, 1.0)
        # Greville abscissae → the quadratic 1 − 0.4 ρ² is reproduced exactly by a cubic B-spline with these coefficients? Not in general:
        # fit the coefficients by least squares on a dense ρ grid instead (exact for a quadratic, see vignetting_tests.jl)
        ρs = range(0, 1, length = 80); A = reduce(vcat, [collect(VgtkCore.bspline_basis(kn, Val(3), ρ))' for ρ in ρs]); A \ [1 - 0.4ρ^2 for ρ in ρs] end))
    @test vig_r.P[1] ≈ vig_r.P[2] atol = 1e-10
    targets_r = [render_view(k, with_vignetting(camera_t, vig_r), poses[k], mat_t, light_t, board, st, masks[k]) for k in 1:2]
    s = calibration_problem(board, st, with_vignetting(camera_t, RadialVignetting(5, 𝐜r, rmax_r)), poses, mat_t, light_t, targets_r; masks)
    _update_vignetting!(s.f, s.u0); @test maximum(abs, Matrix(VrtkCore.control_points(s.u0.camera.vignetting)) - Matrix(VrtkCore.control_points(vig_r))) < 1e-6
    s = calibration_problem(board, st, with_vignetting(camera_t, RadialVignetting(5, 𝐜r, rmax_r)), poses, with_μ(mat_t, 0.7), EnvironmentLight(SphericalHarmonicMap(0.5, 2), b, r_SH), targets_r; masks)
    parts_r = all_parts(s.f, s.u0); _lm_radiometry!(s.f, s.u0, parts_r; damping = VgtkCore.LevenbergDamping(), stops = VgtkCore.LmStops(; max_iter = 80)); cj_r = RC.parts_cost(s.f, s.u0, parts_r)
    @test cj_r < 1e-14 * sum(sum(abs2, v) for v in s.f.ϕ)
    ratio_r = [s.u0.camera.vignetting(SVector(x, y)) / vig_r(SVector(x, y)) for x in 1.2:3.6:22.8 for y in 1.2:3.6:22.8]
    @test maximum(ratio_r) / minimum(ratio_r) - 1 < 1e-5
    @test s.u0.mat.diffuse_weight ≈ 0.4 atol = 1e-5
    prob_r, idx_r = build_problem(s.f, s.u0)
    vv_r = prob_r.variables[idx_r.vignetting]; @test Int(RC.NLLSsolver.nvars(vv_r)) == 3                # 5 coefficients, the first two tied and, under the hard pin, not a column (V(𝐜) = P_0 = 1)
    @test s.u0.camera.vignetting(𝐜r) ≈ 1 atol = 1e-12
    res_r = filter(c -> c isa PerViewResidual{9}, prob_r.blocks)[1]; vars_r = RC.NLLSsolver.getvars(res_r, prob_r.variables)
    @test maximum(abs, RC.NLLSsolver.computeresidual(res_r, vars_r...)) < 1e-5
end

@testitem "calibration parametric warping" setup = [RCFixtures] begin
    using RadiometricCamera: board_surface, TargetState, build_problem, compute_mask, cost, Focus, FocusSpline, HEIGHT_NC,
                             HEIGHT_NFREE, height_on, Material, PerViewResidual, TexturedTarget, render_view, RenderSettings,
                             sphere_radius, with_state, 𝔟
    # 2026-10-30: the parametric (per-view) route with a FREE board warping — `build_problem(; height_degree = 1)` / `_lm_views!(; height_degree)`:
    # the nodal heights ride in the material block as in the grid route (MaterialVar{Float64, HEIGHT_NC}: 9 + HEIGHT_NFREE columns,
    # h = h₀ + B y, B the board's height_basis), seeded as duals from a FLAT start h₀ = 0 (ForwardDiff's `iszero` of a seeded dual is false, so the
    # height field is on for the Jacobian evaluation although the start is the plane)
    using StaticArrays, LinearAlgebra, Statistics
    using VisualGeometryToolkitCore: exp_so3
    using VisualRadiometryToolkitCore: TensorProductVignetting, with_control_points
    import VisualGeometryToolkitCore as VgtkCore
    import VisualRadiometryToolkitCore as VrtkCore

    tex = fixture_board()
    board = TexturedTarget([-1.0, -1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], tex; facing = [0.0, 0.0, 1.5])
    W = H = 24
    h_true = SVector{HEIGHT_NC}([3e-3 * (1 - 4 * ((b[1] - 0.5)^2 + (b[2] - 0.5)^2)) + 1e-3 * 4 * (b[1] - 0.5) * (b[2] - 0.5) for b in RC.HEIGHT_NODES])   # bowl 3 mm + saddle 1 mm on the 2 m board
    board_t = RadiometricCamera.with_state(board, TargetState(h = h_true))
    board_0 = board                                                      # the start: the flat page
    lens_t = cam(VgtkCore.Lens(VgtkCore.DoubleSphere(0.25, 0.4)), 30.912, (12.24, 11.52), W, H)
    tps_t = FocusSpline(VgtkCore.grid_nodes(3, VgtkCore.Rect(0, 0, W, H)))
    focus_t = Focus(tps_t, [1 / (1.5 + 0.1 * sin(3i)) for i in 1:9])
    r_t = 0.0386
    poses = [VgtkCore.EuclideanMap(SMatrix{3, 3}(exp_so3(SVector(π, 0.0, 0.0)) * exp_so3(SVector(0.15 * cos(k), 0.12 * sin(k), 0.1 * k))), SVector{3}(SVector(0.3 * cos(k), 0.3 * sin(k), 1.4 + 0.1 * k)); check = false) for k in 1:2]
    vig_t = let v = TensorProductVignetting((4, 4), VgtkCore.Rect(0, 0, W, H); degree = 2); gr = range(-0.5, 0.5, length = 4); with_control_points(v, [1.0 - 0.4 * (gr[i]^2 + gr[j]^2) for i in 1:4, j in 1:4]) end
    camera_t = RadiometricCameraModel(lens_t, focus_t, r_t, vig_t)
    b = 𝔟(board); r_SH = sphere_radius(b, [p.t for p in poses])
    a_t = zeros(3, 9); a_t[:, 1] .= (2.0, 1.9, 1.8); a_t[:, 3] .= (0.4, 0.3, 0.2); a_t[:, 4] .= (0.1, 0.15, 0.2)
    light_t = EnvironmentLight(SphericalHarmonicMap(a_t, 2), b, r_SH)
    mat_t = Material(; roughness_ink = 0.12, roughness_paper = 0.35, refractive_index_ink = 1.65, refractive_index_paper = 1.33, diffuse_weight = 0.4)
    st = RenderSettings(; n_𝐩 = 2, n_ρ = 2, n_φ = 6, N = 4)
    masks = [compute_mask(camera_t, p, board_t, st; δ = 0.02, backend = RadiometricCamera.KernelAbstractions.CPU()) for p in poses]
    targets = [render_view(k, camera_t, poses[k], mat_t, light_t, board_t, st, masks[k]) for k in 1:2]

    s = calibration_problem(board_0, st, camera_t, poses, mat_t, light_t, targets; masks)
    @test !height_on(board_surface(s.u0))
    prob, idx = build_problem(s.f, s.u0; height_degree = 1)
    mv0 = prob.variables[idx.material]
    @test Int(RC.NLLSsolver.nvars(mv0)) == 9 + HEIGHT_NFREE && RC.n_height(mv0) == HEIGHT_NC
    res = filter(c -> c isa PerViewResidual{9}, prob.blocks)[1]
    vars = RC.NLLSsolver.getvars(res, prob.variables)
    r0 = RC.NLLSsolver.computeresidual(res, vars...)
    @test maximum(abs, r0) > 1e-6                                                     # the flat start misses the warped truth
    rj, J = RC.NLLSsolver.computeresjac(RC.static(127), res, vars...)
    c0 = 6 + 6 + 9 + 1 + 9                                                            # pose | dist | focus | r | material 8 + μ; then the HEIGHT_NFREE height columns
    @test size(J) == (length(r0), c0 + HEIGHT_NFREE + 15 + 27)                       # | vig 15 (the pin's direction eliminated) | light 3×9
    @test isapprox(rj, r0; atol = 1e-10)
    pv, dv, fv, av, mv, vv, lv = vars
    for q in (1, 7, 13, HEIGHT_NFREE)                                                 # height columns (duals from h₀ = 0) vs central differences through `update`
        fd = central_difference(res, vars, 5, unit_increment(9 + HEIGHT_NFREE, 9 + q, 1e-5))
        @test norm(fd) > 0
        @test norm(J[:, c0 + q] - fd) / norm(fd) < 5e-3
    end
    for (col, setter) in ((2, h -> (RC.NLLSsolver.update(pv, unit_increment(6, 2, h)), dv, fv, av, mv, vv, lv)),                              # the other blocks' columns, shifted by the height block
                          (24, h -> (pv, dv, fv, av, RC.NLLSsolver.update(mv, unit_increment(9 + HEIGHT_NFREE, 2, h)), vv, lv)),
                          (c0, h -> (pv, dv, fv, av, RC.NLLSsolver.update(mv, unit_increment(9 + HEIGHT_NFREE, 9, h)), vv, lv)),               # μ
                          (c0 + HEIGHT_NFREE + 2, h -> (pv, dv, fv, av, mv, RC.NLLSsolver.update(vv, unit_increment(15, 2, h)), lv)),          # vignetting node 2
                          (c0 + HEIGHT_NFREE + 15 + 4, h -> (pv, dv, fv, av, mv, vv, RC.NLLSsolver.update(lv, unit_increment(27, 4, h)))))     # lighting a[1, 2]
        fd = (RC.NLLSsolver.computeresidual(res, setter(1e-6)...) - RC.NLLSsolver.computeresidual(res, setter(-1e-6)...)) / 2e-6
        @test norm(J[:, col] - fd) / norm(fd) < 2e-3
    end
    # the free height parameters move the nodal heights along the board's warping's basis (pure bending: no affine term in their
    # interpolant, Core 0.5's rule); the material is untouched
    Aff = VgtkCore.ThinPlateSplineSystem(RC.HEIGHT_NODES).coeff[(HEIGHT_NC + 1):(HEIGHT_NC + 3), :]    # the (square page's) interpolant's affine term a(v) = Aff v
    v_up = RC.NLLSsolver.update(mv, unit_increment(9 + HEIGHT_NFREE, 9 + 1, 1e-3))
    @test mv.basis == RC.height_basis(board)
    @test v_up.surface.h ≈ mv.surface.h + 1e-3 * mv.basis[:, 1] && v_up.mat == mv.mat && maximum(abs, Aff * (v_up.surface.h - mv.surface.h)) < 1e-12
    # without the flag nothing changes: the material block has 9 columns and the Jacobian its former width
    prob0, idx0 = build_problem(s.f, s.u0)
    @test Int(RC.NLLSsolver.nvars(prob0.variables[idx0.material])) == 9
    res0 = filter(c -> c isa PerViewResidual{9}, prob0.blocks)[1]
    @test size(RC.NLLSsolver.computeresjac(RC.static(127), res0, RC.NLLSsolver.getvars(res0, prob0.variables)...)[2], 2) == c0 + 15 + 27
    # a few LM iterations from the flat start (poses + material block free, the rest held) lower the cost and bend the board towards the truth
    s2 = calibration_problem(board_0, st, camera_t, poses, mat_t, light_t, targets; masks)
    c_before = cost(s2.f, s2.u0)[1]
    RC._lm_views!(s2.f, s2.u0; stops = RC._calibration_stops(; max_iter = 8), damping = VgtkCore.LevenbergDamping(), height_degree = 1, fixed = (:distortion, :focus, :aperture, :vignetting, :lighting))
    @test cost(s2.f, s2.u0)[1] < c_before
    @test height_on(board_surface(s2.u0))
    # the bending part of the heights (the oblique projection v ↦ v − M a(v) along the plane fields: what board_height sees; the
    # affine part is the poses')
    M = hcat(ones(HEIGHT_NC), [b[1] for b in RC.HEIGHT_NODES], [b[2] for b in RC.HEIGHT_NODES])
    Pf = I - M * Aff
    @test norm(Pf * (board_surface(s2.u0).h - h_true)) < norm(Pf * h_true)
end

@testitem "material charts: multiplicative α and η − 1, logistic μ; columns scaled" setup = [RCFixtures] begin
    using RadiometricCamera: HEIGHT_NC, Material
    using StaticArrays, LinearAlgebra
    # 2026-10-31: the material block steps on charts — α = eᵘ, η = 1 + eᵘ, μ = lo + (hi − lo) σ(w) on the state's mu_bounds — instead of additive
    # steps clamped to a box; the Jacobian's nine material columns are scaled by the charts' derivatives (`_chart_material_columns!`)
    m0 = Material(; roughness_ink = 0.2, roughness_paper = 0.5, refractive_index_ink = 1.5, refractive_index_paper = 1.2, diffuse_weight = 0.3)
    v = RC.MaterialVar(m0, RC.TargetState(; h = zero(SVector{HEIGHT_NC, Float64})))
    n = Int(RC.NLLSsolver.nvars(v)); @test n == 9
    v2 = RC.NLLSsolver.update(v, unit_increment(n, 1, log(2.0)))
    @test v2.mat.roughness_ink ≈ 0.4 && v2.mat.roughness_paper ≈ 0.5 && v2.mat.refractive_index_ink == m0.refractive_index_ink && v2.mat.diffuse_weight ≈ 0.3
    v3 = RC.NLLSsolver.update(v, unit_increment(n, 3, log(3.0)))
    @test v3.mat.refractive_index_ink[1] ≈ 1 + 3 * 0.5 && v3.mat.refractive_index_ink[2] ≈ 1.5 && v3.mat.refractive_index_paper == m0.refractive_index_paper
    @test RC.NLLSsolver.update(v, unit_increment(n, 1, -10.0)).mat.roughness_ink > 0 && RC.NLLSsolver.update(v, unit_increment(n, 6, -10.0)).mat.refractive_index_paper[1] > 1   # positive, > 1 by the charts
    @test RC.NLLSsolver.update(v, unit_increment(n, 9, -log(0.3 / 0.7))).mat.diffuse_weight ≈ 0.5                                                   # logit(0.3) + u = 0 → σ = 1/2
    μ_big = RC.NLLSsolver.update(v, unit_increment(n, 9, 10.0)).mat.diffuse_weight; @test 0.99 < μ_big < 1.0                                        # stays inside (0, 1)
    # the column scaling: columns μ_col − 8 … μ_col − 1 by α, α, (η − 1) × 6, the μ column by μ(1 − μ); the others untouched
    J = ones(2, 12); lay = (; μ_col = 10)
    RC._chart_material_columns!(J, lay, v)
    @test J[1, 2:9] ≈ [0.2, 0.5, 0.5, 0.5, 0.5, 0.2, 0.2, 0.2] && J[1, 10] ≈ 0.3 * 0.7 && J[1, 1] == 1 && J[1, 11] == 1 && J[1, 12] == 1
    @test RC._chart_material_columns!(ones(2, 12), (; μ_col = 0), v) == ones(2, 12)                                      # frozen material: no columns
end
