# A single-channel run (`io.channel` "R"/"G"/"B", the green photosites) takes ONE residual per measured pixel (owner ruling
# 2026-11-12). Until then the channel was copied into the three channels of a colour state, so every measurement was three residuals:
# that state ("today's") is the reference here. At the same parameters the single-channel cost is exactly a third of it, its residual
# and Jacobian are the reference's channel rows and the measured channel's columns, and a fit from the same start reaches the same
# parameters; the other channels' lighting and η are no parameters (they follow the measured channel).

@testsetup module SingleChannelScene
    using RadiometricCamera: compute_mask, Focus, FocusSpline, Material, TexturedTarget, render_view, RenderSettings,
                             sphere_radius, 𝔟
    using RadiometricCamera, StaticArrays
    using VisualGeometryToolkitCore: exp_so3
    using VisualRadiometryToolkitCore: TensorProductVignetting, with_control_points, EnvironmentLight, SphericalHarmonicMap
    import VisualGeometryToolkitCore as VgtkCore
    const RC = RadiometricCamera
    include(joinpath(@__DIR__, "fixture_board.jl"))
    include(joinpath(@__DIR__, "fixture_camera.jl"))
    include(joinpath(@__DIR__, "fixture_scene.jl"))        # two_view_scene
    export single_channel_board

    "The calibration fixture's two views; the colour reference (`io.channel` copied into three) and the single-channel state of
    channel `c`, both from the start camera and poses (or the truth) with the lighting and η tied to channel `c`."
    function single_channel_board(; c = 2, W = 24, H = 24, perturbed = false, backend = nothing)
        f = two_view_scene(Material(0.12, 0.35, SVector(1.6, 1.65, 1.7), SVector(1.3, 1.33, 1.36), 0.4); W, H)   # η differs by channel
        (; board, st, poses, camera_t, focus_t, vig_t, mat_t, light_t, masks, renders) = f
        copied = [repeat(r[:, :, c:c], 1, 1, 3) for r in renders]                     # today's io.channel: channel c in all three
        camera0 = perturbed ? RadiometricCameraModel(cam(VgtkCore.Lens(VgtkCore.DoubleSphere(0.2, 0.45)), 1.27 * W, (0.5 * W, 0.49 * H), W, H), focus_t, 0.0386, vig_t) :
                              camera_t
        poses0 = perturbed ? [VgtkCore.retract(VgtkCore.LeftRigidManifold6(p), vcat(SVector(0.004, -0.003, 0.005), SVector(0.004, -0.004, 0.006))) for p in poses] :
                             poses
        light0, mat0 = RC.tie_channels(light_t, c), RC.tie_channels(mat_t, c)          # the reference starts tied, as the single-channel state
        kw = backend === nothing ? (;) : (; backend)
        ref = calibration_problem(board, st, camera0, copy(poses0), mat0, light0, copied; masks, kw...)
        one = calibration_problem(board, st, camera0, copy(poses0), mat_t, light_t, renders; masks, channel = c, kw...)
        return (; ref, one, renders, masks, c)
    end
end

@testitem "single channel: one residual per measured pixel, a third of today's cost, its rows and columns" setup = [RCFixtures, SingleChannelScene] begin
    using RadiometricCamera: build_problem, cost, PerViewResidual
    using LinearAlgebra
    f = single_channel_board()
    ref, one, c = f.ref, f.one, f.c
    @test RC.residual_channels(one.f) == (c,) && RC.n_channels(one.f) == 1 && RC.residual_channels(ref.f) == (1, 2, 3)
    @test all(k -> length(one.f.ϕ[k]) == length(one.f.px[k]) == length(ref.f.ϕ[k]) ÷ 3, 1:2)
    @test all(k -> one.f.ϕ[k] == ref.f.ϕ[k][c:3:end], 1:2)                                 # the measured entries, once
    # the other channels' lighting and η follow the measured channel
    a = coefficients(one.u0.light)
    @test a[1, :] == a[c, :] == a[3, :] && one.u0.mat.refractive_index_ink == fill(one.u0.mat.refractive_index_ink[c], 3) && one.u0.mat.refractive_index_paper == fill(one.u0.mat.refractive_index_paper[c], 3)
    @test coefficients(ref.u0.light) == a && ref.u0.mat == one.u0.mat
    # the cost: exactly a third (the same squared residuals, once instead of three times)
    @test cost(one.f, one.u0)[1] ≈ cost(ref.f, ref.u0)[1] / 3 rtol = 1e-12
    @test cost(one.f, one.u0)[2] ≈ cost(ref.f, ref.u0)[2] rtol = 1e-12                                       # the relative RMS is the same
    # the residual and Jacobian of the per-view block: the reference's channel rows; the columns of the single-channel blocks
    p1, i1 = build_problem(one.f, one.u0; damping = VgtkCore.LevenbergDamping()); p0, i0 = build_problem(ref.f, ref.u0; damping = VgtkCore.LevenbergDamping())
    @test Int(RC.NLLSsolver.nvars(p1.variables[i1.lighting])) == 9 && Int(RC.NLLSsolver.nvars(p0.variables[i0.lighting])) == 27
    @test Int(RC.NLLSsolver.nvars(p1.variables[i1.material])) == 5 && Int(RC.NLLSsolver.nvars(p0.variables[i0.material])) == 9
    r1 = get(p1.problem.costs, PerViewResidual{9})[1]; r0 = get(p0.problem.costs, PerViewResidual{9})[1]
    v1 = RC.NLLSsolver.getvars(r1, p1.variables); v0 = RC.NLLSsolver.getvars(r0, p0.variables)
    @test RC.NLLSsolver.nres(r1) == length(one.f.px[1]) && RC.NLLSsolver.nres(r0) == 3 * length(one.f.px[1])
    e1, J1 = RC.NLLSsolver.computeresjac(RC.static(127), r1, v1...)
    e0, J0 = RC.NLLSsolver.computeresjac(RC.static(127), r0, v0...)
    @test e1 == e0[c:3:end] && e1 == RC.NLLSsolver.computeresidual(r1, v1...)
    @test size(J1, 2) == size(J0, 2) - 18 - 4                                            # 2 × 9 lighting and 4 η columns fewer
    lay = RC._jacobian_layout(127, 9, length(one.f.settings.sensor), VgtkCore.tangent_size(v0[6]), RC.n_control_points(VgtkCore.model(v0[6])), 9,
                              VgtkCore.tangent_size(v0[2]))
    @test J1 == RC._channel_jacobian(J0, lay, c)
    # the 22 columns dropped are the ones the measured channel does not see: no parameter of the fit is left with a zero column
    zc(M) = count(j -> iszero(view(M, :, j)), axes(M, 2))
    @test zc(J1) == zc(J0[c:3:end, :]) - 22
end

@testitem "single channel: a fit reaches today's parameters" setup = [RCFixtures, SingleChannelScene] begin
    using RadiometricCamera: cost, _lm_views!
    using Rotations: rotation_angle, RotMatrix
    f = single_channel_board(; perturbed = true)
    ref, one, c = f.ref, f.one, f.c
    _lm_views!(ref.f, ref.u0; stops = RC._calibration_stops(; max_iter = 200, ftol = 1e-12), damping = VgtkCore.LevenbergDamping(), fixed = (:focus, :aperture))
    _lm_views!(one.f, one.u0; stops = RC._calibration_stops(; max_iter = 200, ftol = 1e-12), damping = VgtkCore.LevenbergDamping(), fixed = (:focus, :aperture))
    @test cost(one.f, one.u0)[1] ≈ cost(ref.f, ref.u0)[1] / 3 rtol = 1e-6 atol = 1e-20
    @test all(rad2deg(rotation_angle(RotMatrix{3}(one.u0.poses[k].R' * ref.u0.poses[k].R))) < 1e-6 for k in 1:2)
    @test maximum(abs, VgtkCore.parameter_vector(VgtkCore.model(RC.geometric(one.u0))) .- VgtkCore.parameter_vector(VgtkCore.model(RC.geometric(ref.u0)))) < 1e-6
    @test maximum(abs, coefficients(one.u0.light) .- coefficients(ref.u0.light)) < 1e-6 * maximum(abs, coefficients(ref.u0.light))
    @test abs(one.u0.mat.diffuse_weight - ref.u0.mat.diffuse_weight) < 1e-6 && abs(one.u0.mat.refractive_index_ink[c] - ref.u0.mat.refractive_index_ink[c]) < 1e-6
    # the other channels still follow the measured one
    a = coefficients(one.u0.light); @test a[1, :] == a[c, :] == a[3, :]
end

@testitem "single channel: the grid route's tiles, one residual per measured pixel" setup = [RCFixtures, GridTilesScene] begin
    using RadiometricCamera: build_problem_grid
    c = 2
    copied = [repeat(t[:, :, c:c], 1, 1, 3) for t in targets]
    ref = calibration_problem(board, st, camera_t, copy(poses), RC.tie_channels(mat_t, c), RC.tie_channels(light_t, c), copied; masks)
    one = calibration_problem(board, st, camera_t, copy(poses), mat_t, light_t, targets; masks, channel = c)
    p0, _ = build_problem_grid(ref.f, ref.u0; damping = VgtkCore.LevenbergDamping(), patch = (5, 5)); p1, _ = build_problem_grid(one.f, one.u0; damping = VgtkCore.LevenbergDamping(), patch = (5, 5))
    t0 = RC.tile_costs(p0); t1 = RC.tile_costs(p1)
    @test length(t0) == length(t1)
    for (a, b) in zip(t0, t1)
        va = RC.NLLSsolver.getvars(a, p0.variables); vb = RC.NLLSsolver.getvars(b, p1.variables)
        @test RC.NLLSsolver.nres(b) == length(b.pix) && b.ϕt == a.ϕt[c:3:end]
        ea, Ja = RC.NLLSsolver.computeresjac(RC.static(1023), a, va...)
        eb, Jb = RC.NLLSsolver.computeresjac(RC.static(1023), b, vb...)
        @test eb == ea[c:3:end] && size(Jb) == (length(b.pix), size(Ja, 2) - 22)
        nz(M) = M[:, [j for j in axes(M, 2) if any(!iszero, view(M, :, j))]]            # the columns that see the measured channel
        @test nz(Jb) == nz(Ja[c:3:end, :])
    end
end

@testitem "single channel: the device evaluates the same residual and Jacobian" tags = [:gpu] setup = [RCFixtures, RCGPU, SingleChannelScene] begin
    using RadiometricCamera: build_problem, PerViewResidual
    import CUDA
    using CUDA: CUDABackend
    require_gpu()
    if RC_GPU_TESTS
        cpu = single_channel_board().one
        gpu = single_channel_board(; backend = CUDABackend(; always_inline = true)).one
        pc, ic = build_problem(cpu.f, cpu.u0; damping = VgtkCore.LevenbergDamping()); pg, ig = build_problem(gpu.f, gpu.u0; damping = VgtkCore.LevenbergDamping())
        rc = get(pc.problem.costs, PerViewResidual{9})[1]; rg = get(pg.problem.costs, PerViewResidual{9})[1]
        vc = RC.NLLSsolver.getvars(rc, pc.variables); vg = RC.NLLSsolver.getvars(rg, pg.variables)
        ec, Jc = RC.NLLSsolver.computeresjac(RC.static(127), rc, vc...)
        eg, Jg = RC.NLLSsolver.computeresjac(RC.static(127), rg, vg...)
        @test length(eg) == length(ec) == length(cpu.f.px[1]) && size(Jg) == size(Jc)
        @test maximum(abs, eg .- ec) < 2e-3 * max(1.0, maximum(abs, ec))                      # Float32 device, Float64 CPU
        @test eg == RC.NLLSsolver.computeresidual(rg, vg...)                                   # the device residual, bitwise its own
        for j in axes(Jc, 2)
            n = sqrt(sum(abs2, view(Jc, :, j)))
            n < 1e-3 * maximum(sqrt(sum(abs2, view(Jc, :, q))) for q in axes(Jc, 2)) && continue
            @test sqrt(sum(abs2, view(Jc, :, j) .- view(Jg, :, j))) / n < 2e-2
        end
    end
end
