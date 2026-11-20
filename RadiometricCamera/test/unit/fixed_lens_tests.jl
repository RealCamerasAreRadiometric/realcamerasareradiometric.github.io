# Any VisualGeometryToolkitCore lens model is a parametric chief-ray model as Core's pixel camera (a BoundedCamera of a CameraModel, K
# in pixels) — fixed in the per-view LM when no parameter is freed: its sampling into the generic grid model (the route that makes it
# refinable), Float32 cleanliness, duals through Core's lens inverses, and the per-view problem with the lens block fixed.
@testitem "Core lens as a fixed pixel camera" setup = [RCFixtures] begin
    using RadiometricCamera: build_problem, compute_mask, cost, Focus, FocusSpline, grid_angular_error, grid_model, in_domain,
                             Material, PerViewResidual, PrintedTarget, render_view, RenderSettings, run_lm!, Scene,
                             sphere_radius, _chief_ray, 𝔟
    using StaticArrays, LinearAlgebra, Statistics, ForwardDiff
    using VisualGeometryToolkitCore: exp_so3
    using VisualGeometryToolkitCore: TensorProductVignetting

    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: Lens, Perspective, Equidistant, Additive, Radial, Tangential
    using Random
    W₀, H₀ = 1200, 800
    f_px = 650.0; c_px = (W₀ / 2 + 3.5, H₀ / 2 - 2.0)
    kb = VgtkCore.KannalaBrandt(-0.02, 0.004, -0.0007, 0.00008)
    bc = Lens(Perspective(), Additive(Radial((-0.12, 0.03)), Tangential(0.0005, -0.0003)))   # Brown–Conrady k₁, k₂, p₁, p₂
    rng = MersenneTwister(7)
    px = [SVector(0.08W₀ + 0.84W₀ * rand(rng), 0.08H₀ + 0.84H₀ * rand(rng)) for _ in 1:25]
    for (lens, name) in ((kb, "Kannala–Brandt"), (bc, "Brown–Conrady"), (Lens(Equidistant()), "equidistant"), (VgtkCore.Pinhole(), "pinhole"))
        l = cam(lens, f_px, c_px, W₀, H₀)
        @test pinhole_θ(l) == SVector(f_px, f_px, c_px[1], c_px[2])
        @test last(RC.lens_free(l; fixed = VgtkCore.parameter_names(VgtkCore.model(l)))) && !last(RC.lens_free(l))
        for 𝐩 in px
            m = SVector((𝐩[1] - c_px[1]) / f_px, (𝐩[2] - c_px[2]) / f_px)              # Core's normalised (distorted) point
            d = _chief_ray(l, 𝐩)
            @test norm(d) ≈ 1 && in_domain(l, 𝐩)
            x_u = SVector(d[1] / d[3], d[2] / d[3])                                     # the undistorted perspective point
            @test VgtkCore.lens(l)(x_u) ≈ m atol = 1e-9                                 # Core's forward distortion closes the loop
            @test VgtkCore.project(l, d) ≈ 𝐩 atol = 1e-8                                # the camera's own projection (pixels)
        end
        # Float32 narrowing (the device path)
        l32 = VgtkCore.convert_eltype(Float32, l)
        @test has_lens(l32, VgtkCore.AbstractLensModel) && eltype(pinhole_θ(l32)) === Float32 && isbitstype(typeof(l32))
        @test maximum(abs, _chief_ray(l32, SVector{2, Float32}(px[1])) .- _chief_ray(l, px[1])) < 1e-4    # Float32 Newton / undistortion
    end
    # the Kannala–Brandt route is spherical: rays beyond 90° are admitted (a perspective inverse is not)
    kbw = cam(VgtkCore.KannalaBrandt(0.05, 0.0, 0.0, 0.0), 180.0, (W₀ / 2, H₀ / 2), W₀, H₀)
    d = _chief_ray(kbw, SVector(1140.0, 400.0))
    @test norm(d) ≈ 1 && d[3] < 0.3                                                     # ≈ 80° off axis
    @test in_domain(kbw, SVector(1140.0, 400.0))
    # duals go through the model (the dual sensor positions of the grid sampling / AD seeds)
    l = cam(kb, f_px, c_px, W₀, H₀)
    @test _chief_ray(l, SVector(ForwardDiff.Dual(720.0, 1.0), ForwardDiff.Dual(440.0, 0.0))) isa SVector{3, <:ForwardDiff.Dual}
    @test has_lens(l, VgtkCore.KannalaBrandt{4, Float64}) && VgtkCore.lens(l) == kb
    # the refinable route: sample into the SplineLens camera, whose K is the camera's own (the gauge)
    g = grid_model(l, 35, 25)
    @test grid_angular_error(g, l) < 2e-3                                               # 35 × 25 quasi-interpolation of a KB lens
    @test grid_angular_error(grid_model(l, 105, 75), l) < 2e-4
    @test VgtkCore.calibration_matrix(VgtkCore.model(g)) == VgtkCore.calibration_matrix(VgtkCore.model(l))
    # a fixed lens block: held fixed by the LM (NLLSsolver's variables are never empty, so it keeps the default coordinates)
    names_f, fixed_f = RC.lens_free(l; fixed = VgtkCore.parameter_names(VgtkCore.model(l)))
    @test fixed_f && names_f == filter(!=(:skew), VgtkCore.parameter_names(VgtkCore.model(l))) && free_names(RC.lens_manifold(l; fixed = VgtkCore.parameter_names(VgtkCore.model(l)))) == names_f
    # per-view problem with a KB lens on the synthetic board: Jacobian layout, zero distortion columns, LM smoke
    tex = fixture_board()
    board = PrintedTarget([-1.0, -1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], tex; facing = [0.0, 0.0, 1.5])
    scene = Scene(board, 24, 24)
    lk = cam(VgtkCore.KannalaBrandt(-0.01, 0.002, 0.0, 0.0), 31.2, (12.24, 11.52), 24, 24)
    tps_k = FocusSpline(VgtkCore.grid_nodes(3, VgtkCore.Rect(0, 0, 24, 24))); focus_k = Focus(tps_k, fill(1 / 1.5, 9))
    poses = [VgtkCore.EuclideanMap(SMatrix{3, 3}(exp_so3(SVector(π, 0.0, 0.0)) * exp_so3(SVector(0.15 * cos(k), 0.12 * sin(k), 0.1 * k))), SVector{3}(SVector(0.3 * cos(k), 0.3 * sin(k), 1.4 + 0.1 * k)); check = false) for k in 1:2]
    vig_k = TensorProductVignetting((4, 4), VgtkCore.Rect(0, 0, 24, 24); degree = 2); b = 𝔟(board); r_SH = sphere_radius(b, [p.t for p in poses])
    camera_k = RadiometricCameraModel(lk, focus_k, 0.0386, vig_k)
    light_k = EnvironmentLight(SphericalHarmonicMap(2.0, 2), b, r_SH); mat_k = Material()
    st = RenderSettings(; n_𝐩 = 2, n_ρ = 2, n_φ = 6, N = 4)
    masks = [compute_mask(camera_k, p, scene, st; δ = 0.02, backend = RadiometricCamera.KernelAbstractions.CPU()) for p in poses]
    targets = [render_view(k, camera_k, poses[k], mat_k, light_k, scene, st, masks[k]) for k in 1:2]
    s = CalibrationState(scene, st, camera_k, copy(poses), mat_k, light_k, targets; masks)
    fixed = VgtkCore.parameter_names(VgtkCore.model(lk))   # every parameter held
    prob, idx = build_problem(s; fixed)
    res = get(prob.problem.costs, PerViewResidual{9})[1]
    vars = RC.NLLSsolver.getvars(res, prob.variables)
    r0, J = RC.NLLSsolver.computeresjac(RC.static(125), res, vars...)                 # the LM's flags: every block but the fixed lens
    @test maximum(abs, r0) < 1e-12 && size(J, 2) == 6 + 0 + 9 + 1 + 9 + 15 + 27        # the fixed lens has no columns; 15 vignetting columns (the hard pin)
    @test norm(J[:, 1]) > 0 && norm(J[:, 6 + 9 + 1]) > 0                               # pose and aperture columns are live
    s2 = CalibrationState(scene, st, camera_k, [VgtkCore.retract(VgtkCore.LeftRigidManifold6(p), vcat(SVector(0.004, -0.003, 0.004), SVector(0.003, -0.003, 0.004))) for p in poses],
                          mat_k, light_k, targets; masks)
    c0 = cost(s2)[1]
    run_lm!(s2; maxiters = 8, fixed)
    @test cost(s2)[1] < 1e-3 * c0
    @test pinhole_θ(RC.geometric(s2)) ≈ pinhole_θ(lk) && VgtkCore.lens(RC.geometric(s2)) == VgtkCore.lens(lk)   # untouched by the LM
end
