# The GPU backend: the CPU-side Float32 cleanliness and the out-of-domain samples (always run), then the device path (the :gpu item).

# The two Kannala–Brandt lenses and the 24² scene of the out-of-domain checks both items use.
@testsetup module GPUDomainScene
    using RadiometricCamera: Material, TexturedTarget, RenderSettings
    using RadiometricCamera, StaticArrays, LinearAlgebra, ForwardDiff
    using VisualGeometryToolkitCore: exp_so3
    using VisualRadiometryToolkitCore: TensorProductVignetting, EnvironmentLight, SphericalHarmonicMap
    import VisualGeometryToolkitCore as VgtkCore
    const RC = RadiometricCamera
    include(joinpath(@__DIR__, "fixture_board.jl"))
    include(joinpath(@__DIR__, "fixture_camera.jl"))
    export kb_in, kb_out, dom_board, dom_args

    kb_in = cam(VgtkCore.KannalaBrandt(0.0, 0.0, 0.0, 0.0), 9.6, (12.0, 12.0), 24, 24)             # equidistant, θ ≤ 1.77 at the sensor corners: all imaged
    kb_out = cam(VgtkCore.KannalaBrandt(0.0, 0.0, 0.0, 0.0), 4.8, (12.0, 12.0), 24, 24)            # θ = 3.54 > π at the corners: no ray there
    dom_board, dom_args = let tex_d = fixture_board()
        bd = RadiometricCamera.with_facing(TexturedTarget([-1.0, -1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], tex_d), [0.0, 0.0, 1.5])
        sc = bd
        pose = VgtkCore.EuclideanMap(SMatrix{3, 3}(exp_so3(SVector(π, 0.0, 0.0))), SVector{3}(SVector(0.0, 0.0, 1.5)); check = false)
        sc, (; pose, vig = TensorProductVignetting((4, 4), VgtkCore.Rect(0, 0, 24, 24); degree = 2), mat = Material(), light = EnvironmentLight(SphericalHarmonicMap(1.0, 2), VgtkCore.centroid(bd), 2.0),
              st = RenderSettings(; n_𝐩 = 2, n_ρ = 2, n_φ = 6, N = 4), sensor = 0.066)
    end
end

@testitem "GPU backend (CPU side)" setup = [RCFixtures, GPUDomainScene] begin
    using RadiometricCamera: cost, Focus, in_domain, Material, TexturedTarget, render_view, RenderSettings, View, Φ_pixel
    using StaticArrays, LinearAlgebra, Statistics, ForwardDiff
    using VisualGeometryToolkitCore: exp_so3
    using VisualRadiometryToolkitCore: TensorProductVignetting
    cpu = RadiometricCamera.KernelAbstractions.CPU()   # chosen: the CPU side runs on the CPU whatever the default device

    import VisualGeometryToolkitCore as VgtkCore

    # CPU-side Float32 cleanliness (always runs)
    p0 = VgtkCore.EuclideanMap(SMatrix{3, 3}(1.0I), zeros(SVector{3}); check = false)
    @test isbitstype(typeof(View(p0, VgtkCore.convert_eltype(Float32, cam(VgtkCore.Lens(VgtkCore.DoubleSphere(0.25, 0.4)), 30.912, (12.0, 12.0), 24, 24)), VgtkCore.convert_eltype(Float32, Focus(3, 1.5, 24, 24)), 0.03f0))) &&
          isbitstype(typeof(View(p0, cam(VgtkCore.Division(0.0, 0.0), 24.0, (12.0, 12.0), 24, 24), Focus(5, 1.5, 24, 24), 0.03)))
    @test isbitstype(typeof(View(VgtkCore.EuclideanMap(SMatrix{3, 3}(SMatrix{3, 3}(1.0I)), SVector{3}(zeros(SVector{3})); check = false),
                                 VgtkCore.convert_eltype(Float32, cam(VgtkCore.Lens(VgtkCore.Equidistant()), 31.2, (12.0, 12.0), 24, 24)),
                                 VgtkCore.convert_eltype(Float32, Focus(3, 1.5, 24, 24)), 0.03f0)))
    @test isbitstype(typeof(VgtkCore.convert_eltype(Float32, RenderSettings())))
    let tex32
        tex32 = fixture_board()
        bd = RadiometricCamera.with_facing(TexturedTarget([-1.0, -1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], tex32), [0.0, 0.0, 1.5])
        sc = bd
        pose = VgtkCore.EuclideanMap(SMatrix{3, 3}(exp_so3(SVector(π, 0.0, 0.0))), SVector{3}(SVector(0.0, 0.0, 1.5)); check = false)
        vg = TensorProductVignetting((4, 4), VgtkCore.Rect(0, 0, 32, 32); degree = 2); mt = Material(); lt = EnvironmentLight(SphericalHarmonicMap(1.0, 2), VgtkCore.centroid(bd), 2.0)
        camera = RadiometricCameraModel(cam(VgtkCore.Lens(VgtkCore.DoubleSphere(0.25, 0.4)), 41.216, (16.0, 16.0), 32, 32), Focus(3, 1.5, 32, 32), 0.0386, vg)
        stt = RenderSettings(; N = 4)
        args32 = VgtkCore.convert_eltype.(Float32, (View(pose, camera), vg, mt, lt, sc, stt))   # the kernels' Float32 record (the device's element type)
        Φ32 = RC._Φ_pixel(16, 16, 1, RC._image_height(camera), args32...)   # the image's height is the camera's
        @test Φ32 isa SVector{3, Float32}
        Φ64 = Φ_pixel(16, 16, 1, camera, pose, mt, lt, sc, stt)
        @test maximum(abs, (Φ32 .- Φ64) ./ Φ64) < 1e-4
    end

    # A sample outside the lens's domain (Core's π⁻¹: no ray) renders ZERO flux — the board test skips its NaN ray — so the residual stays
    # finite and a step that pushes samples out of the domain only raises the cost, which the LM rejects (raising the damping). (A NaN
    # cost would be worse: NLLSsolver's LevMar takes `!(NaN > bestcost)` as success and its optimize loop stops.) Tolerances fixed before
    # the run: every Φ finite; Φ exactly zero at a pixel whose samples all lie outside the domain; the moved lens's cost finite and higher.
    let a = dom_args
        @test in_domain(kb_in, SVector(24.0, 24.0)) && !in_domain(kb_out, SVector(24.0, 24.0)) && in_domain(kb_out, SVector(12.0, 12.0))
        cam_in = RadiometricCameraModel(kb_in, Focus(3, 1.5, 24, 24), 0.0386, a.vig)
        cam_out = RadiometricCameraModel(kb_out, Focus(3, 1.5, 24, 24), 0.0386, a.vig)
        Φs = [Φ_pixel(u, v, 1, cam_out, a.pose, a.mat, a.light, dom_board, a.st) for u in 1:24, v in 1:24]
        @test all(all(isfinite, Φ) for Φ in Φs)
        @test iszero(Φs[1, 1]) && iszero(Φs[24, 24])                               # the corner pixels: every sample outside the domain
        full = [trues(24, 24)]
        tgt = [render_view(1, cam_in, a.pose, a.mat, a.light, dom_board, a.st, full[1]; backend = cpu)]
        s_in = calibration_problem(dom_board, a.st, cam_in, [a.pose], a.mat, a.light, tgt; masks = full, backend = cpu)
        s_out = calibration_problem(dom_board, a.st, cam_out, [a.pose], a.mat, a.light, tgt; masks = full, backend = cpu)
        c_in = cost(s_in.f, s_in.u0)[1]; c_out = cost(s_out.f, s_out.u0)[1]
        @test isfinite(c_out) && c_out > c_in
    end
end

@testitem "GPU backend" tags = [:gpu] setup = [RCFixtures, RCGPU, GPUDomainScene] begin
    using RadiometricCamera: build_problem, compute_mask, cost, Focus, FocusSpline, Material, PerViewResidual, TexturedTarget,
                             render_all, render_view, RenderSettings, _lm_views!, _update_vignetting!, sphere_radius, View, with_nonlinear,
                             chief_ray, sensor_position
    using StaticArrays, LinearAlgebra, Statistics, ForwardDiff
    using VisualGeometryToolkitCore: exp_so3
    using VisualRadiometryToolkitCore: TensorProductVignetting, RadialVignetting, with_control_points
    import VisualGeometryToolkitCore as VgtkCore
    import CUDA

    using CUDA: CUDABackend
    require_gpu()                                  # errors with the reason when the GPU the runner chose is not usable here
    if RC_GPU_TESTS
        @info "GPU tests on $(CUDA.name(CUDA.device()))"
        tex = fixture_board()
        board = RadiometricCamera.with_facing(TexturedTarget([-1.0, -1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], tex), [0.0, 0.0, 1.5])
        W = H = 32
        lens_t = cam(VgtkCore.Lens(VgtkCore.DoubleSphere(0.25, 0.4)), 41.216, (16.32, 15.36), W, H)
        tps_g = FocusSpline(VgtkCore.grid_nodes(3, VgtkCore.Rect(0, 0, W, H)))
        focus_t = Focus(tps_g, [1 / (1.5 + 0.1 * sin(3i)) for i in 1:9])
        poses = [VgtkCore.EuclideanMap(SMatrix{3, 3}(exp_so3(SVector(π, 0.0, 0.0)) * exp_so3(SVector(0.15cos(k), 0.12sin(k), 0.1k))), SVector{3}(SVector(0.3cos(k), 0.3sin(k), 1.4 + 0.1k)); check = false) for k in 1:2]
        # the truth in the hard pin's gauge V(𝐜) = 1 (`gauge_center_value`; the default pin, 2026-10-30): a state built from it is the truth itself
        vig_t = RC.gauge_center_value(let v = TensorProductVignetting((4, 4), VgtkCore.Rect(0, 0, W, H); degree = 2); g = range(-0.5, 0.5, length = 4); with_control_points(v, [1.0 - 0.4 * (g[i]^2 + g[j]^2) for i in 1:4, j in 1:4]) end)
        camera_t = RadiometricCameraModel(lens_t, focus_t, 0.0386, vig_t)
        b = VgtkCore.centroid(board); r_SH = sphere_radius(b, [p.t for p in poses])
        a_t = zeros(3, 9); a_t[:, 1] .= (2.0, 1.9, 1.8); a_t[:, 3] .= (0.4, 0.3, 0.2)
        light_t = EnvironmentLight(SphericalHarmonicMap(a_t, 2), b, r_SH)
        mat_t = Material(; roughness_ink = 0.12, roughness_paper = 0.35, refractive_index_ink = 1.65, refractive_index_paper = 1.33, diffuse_weight = 0.4)
        st = RenderSettings(; n_𝐩 = 2, n_ρ = 2, n_φ = 6, N = 4)
        cpu = RadiometricCamera.KernelAbstractions.CPU()                 # the CPU reference, chosen: the default device is the GPU here
        masks = [compute_mask(camera_t, p, board, st; δ = 0.02, backend = cpu) for p in poses]
        targets = [render_view(k, camera_t, poses[k], mat_t, light_t, board, st, masks[k]; backend = cpu) for k in 1:2]
        s_cpu = calibration_problem(board, st, camera_t, poses, mat_t, light_t, targets; masks, backend = cpu)
        s_gpu = calibration_problem(board, st, camera_t, poses, mat_t, light_t, targets; masks, backend = CUDABackend())
        # render parity (Float64 CPU vs Float32 GPU; different draw precision → tolerance)
        img_cpu = render_all(s_cpu.f, s_cpu.u0)[1]; img_gpu = render_all(s_gpu.f, s_gpu.u0)[1]
        @test maximum(abs, img_gpu .- img_cpu) / maximum(img_cpu) < 2e-3
        # render_view on the device (its `backend`): the flux kernel in Float32 there, the CPU's image to Float32 rounding
        img_dev = render_view(1, camera_t, poses[1], mat_t, light_t, board, st, masks[1]; backend = CUDABackend())
        @test size(img_dev) == size(targets[1]) && maximum(abs, img_dev .- targets[1]) / maximum(targets[1]) < 2e-3
        @test render_view(1, camera_t, poses[1], mat_t, light_t, board, st, masks[1]; backend = cpu) == targets[1]
        # residual and Jacobian parity
        prob_c, _ = build_problem(s_cpu.f, s_cpu.u0); prob_g, _ = build_problem(s_gpu.f, s_gpu.u0)
        rc = filter(c -> c isa PerViewResidual{9}, prob_c.blocks)[1]; rg = filter(c -> c isa PerViewResidual{9}, prob_g.blocks)[1]
        vc = RC.NLLSsolver.getvars(rc, prob_c.variables); vgg = RC.NLLSsolver.getvars(rg, prob_g.variables)
        r_c = RC.NLLSsolver.computeresidual(rc, vc...); r_g = RC.NLLSsolver.computeresidual(rg, vgg...)
        @test maximum(abs, r_c .- r_g) < 2e-3 * max(1.0, maximum(abs, r_c))
        rc0, Jc = RC.NLLSsolver.computeresjac(RC.static(127), rc, vc...)
        rg0, Jg = RC.NLLSsolver.computeresjac(RC.static(127), rg, vgg...)
        @test size(Jc) == size(Jg)
        @test rg0 == r_g                     # bitwise: NLLSsolver compares the two costs with isapprox after every step
        @test all(isfinite, Jg) && all(isfinite, rg0)                # scrubbed derivative singularities
        # held blocks (focus and vignetting fixed, varflags 91): the device row kernel compiles their absent inputs out (2026-11-13: a
        # held focus failed to compile on the device), and its Jacobian and Gauss–Newton blocks are the CPU's to Float32 rounding
        _, Jc91 = RC.NLLSsolver.computeresjac(RC.static(91), rc, vc...)
        _, Jg91 = RC.NLLSsolver.computeresjac(RC.static(91), rg, vgg...)
        @test size(Jg91) == size(Jc91) && all(isfinite, Jg91)
        @test maximum(abs, Jg91 .- Jc91) <= 2e-2 * maximum(abs, Jc91)
        _, g91, H91 = RC.NLLSsolver.computecostgradhess(RC.static(91), rg, vgg...)
        @test length(g91) == size(Jg91, 2) && all(isfinite, g91) && all(isfinite, H91)
        # a non-finite trial state must give a NaN cost on the device path (NLLSsolver's damping loop needs it)
        nanpose = VgtkCore.LeftRigidManifold6(VgtkCore.EuclideanMap(SMatrix{3, 3}(VgtkCore.model(vgg[1]).R), SVector{3}(SVector(NaN, NaN, NaN)); check = false))
        @test all(isnan, RC.NLLSsolver.computeresidual(rg, nanpose, vgg[2], vgg[3], vgg[4], vgg[5], vgg[6], vgg[7]))
        colnorms = [norm(Jc[:, col]) for col in axes(Jc, 2)]
        for col in axes(Jc, 2)
            nc = colnorms[col]
            nc < 1e-3 * maximum(colnorms) && continue          # tiny columns are Float32-noise-dominated
            # Float32 vs Float64: geometry columns ~1e-2 at this tiny low-sample scene; the material
            # columns differentiate *through* the VNDF sampler and carry similar attached-derivative noise
            @test norm(Jc[:, col] .- Jg[:, col]) / nc < 2e-2
        end
        # LM smoke on the GPU backend
        camera_2 = RadiometricCameraModel(cam(VgtkCore.Lens(VgtkCore.DoubleSphere(0.15, 0.5)), 40.0, (16.0, 16.0), W, H), Focus(tps_g, fill(1 / 1.7, 9)), 0.03, vig_t)
        poses_2 = [VgtkCore.retract(VgtkCore.LeftRigidManifold6(p), vcat(SVector(0.008, -0.006, 0.01), SVector(0.008, -0.008, 0.012))) for p in poses]
        s2 = calibration_problem(board, st, camera_2, poses_2, with_nonlinear(mat_t, SVector(0.2, 0.3, 1.5, 1.5, 1.5, 1.4, 1.4, 1.4)), light_t, targets;
                              masks, backend = CUDABackend())
        c0 = cost(s2.f, s2.u0)[1]
        result = _lm_views!(s2.f, s2.u0; stops = RC._calibration_stops(; max_iter = 20), damping = VgtkCore.LevenbergDamping())
        @test result.metadata.cost / 2 < 1e-2 * c0                     # Float32 J (80 dual slots, all blocks free) stalls above the Float64 floor
        # the same parity for the 18-coordinate lens block of the 14-coefficient lens (3 dual chunks of 15 on the device path)
        cv_o = [-0.05, 0.01, 1e-3, -2e-3, 0.0, 0.02, 0.0, 0.0, 1e-3, 0.0, -1e-3, 0.0, 0.01, -0.005]            # cv2's coefficient order
        lens_o = cam(VgtkIO.opencv_lens(cv_o), 41.216, (16.32, 15.36), W, H)
        camera_o = RadiometricCameraModel(lens_o, focus_t, 0.0386, vig_t)
        masks_o = [compute_mask(camera_o, p, board, st; δ = 0.02, backend = RadiometricCamera.KernelAbstractions.CPU()) for p in poses]
        targets_o = [render_view(k, camera_o, poses[k], mat_t, light_t, board, st, masks_o[k]; backend = cpu) for k in 1:2]
        so_cpu = calibration_problem(board, st, camera_o, poses, mat_t, light_t, targets_o; masks = masks_o, backend = cpu)
        so_gpu = calibration_problem(board, st, camera_o, poses, mat_t, light_t, targets_o; masks = masks_o, backend = CUDABackend())
        po_c, _ = build_problem(so_cpu.f, so_cpu.u0); po_g, _ = build_problem(so_gpu.f, so_gpu.u0)
        ro_c = filter(c -> c isa PerViewResidual{9}, po_c.blocks)[1]; ro_g = filter(c -> c isa PerViewResidual{9}, po_g.blocks)[1]
        vo_c = RC.NLLSsolver.getvars(ro_c, po_c.variables); vo_g = RC.NLLSsolver.getvars(ro_g, po_g.variables)
        @test vo_g[2] isa VgtkCore.SubsetManifold && Int(RC.NLLSsolver.nvars(vo_g[2])) == 18
        _, Jo_c = RC.NLLSsolver.computeresjac(RC.static(127), ro_c, vo_c...)
        ro0, Jo_g = RC.NLLSsolver.computeresjac(RC.static(127), ro_g, vo_g...)
        @test size(Jo_c) == size(Jo_g) == (length(ro0), 6 + 18 + 9 + 1 + 9 + 15 + 27)   # 15 vignetting columns (the hard pin)
        @test ro0 == RC.NLLSsolver.computeresidual(ro_g, vo_g...) && all(isfinite, Jo_g)
        cn_o = [norm(Jo_c[:, col]) for col in axes(Jo_c, 2)]
        for col in 7:24                                      # the lens columns (f, aspect, c, the 14 coefficients)
            cn_o[col] < 1e-3 * maximum(cn_o) && continue
            @test norm(Jo_c[:, col] .- Jo_g[:, col]) / cn_o[col] < 2e-2
        end
        # the start: f and aspect perturbed by 0.64 and k1 by 0.03, k1 found by NAME — Core 0.5's composed lens (Tilt ∘ Additive)
        # lists the tilt (:tx, :ty) first, so the fifth coordinate is no longer k1 (a tilt start the block (:f, :aspect, :k1) cannot undo)
        ik1_o = 4 + findfirst(==(:k1), VgtkCore.parameter_names(VgtkCore.lens(lens_o)))
        lens_o2 = camera_of(VgtkIO.opencv_lens(zeros(14)), vcat(pinhole_θ(lens_o), VgtkCore.parameter_vector(VgtkCore.lens(lens_o))) + SVector{18}(ntuple(i -> i <= 2 ? 0.64 : i == ik1_o ? 0.03 : 0.0, 18)), W, H)
        so2 = calibration_problem(board, st, RadiometricCameraModel(lens_o2, focus_t, 0.0386, vig_t), poses, mat_t, light_t, targets_o;
                               masks = masks_o, backend = CUDABackend())
        co0 = cost(so2.f, so2.u0)[1]
        ro = _lm_views!(so2.f, so2.u0; stops = RC._calibration_stops(; max_iter = 4), damping = VgtkCore.LevenbergDamping(), fixed = (:poses, :focus, :aperture, :material, :vignetting, :lighting,
                             :cx, :cy, :tx, :ty, :k2, :k3, :d, :p, :sx, :sy))
        @test ro.metadata.cost / 2 < 0.05 * co0 && has_lens(RC.geometric(so2.u0), typeof(VgtkIO.opencv_lens(zeros(14))))
        # radial vignetting profile on the device (2026-10-21): render parity, and the closed-form V solve from the DEVICE parts
        # (the parts kernel evaluates coefficient_jacobian / V of the radial type) recovers the profile
        vig_r = with_control_points(RadialVignetting(5, SVector(16.64, 15.04), 0.6 * W), [1.0, 1.0, 0.9, 0.75, 0.6])   # P_0 = P_1: the pole tie
        camera_r = RC.with_vignetting(camera_t, vig_r)
        targets_r = [render_view(k, camera_r, poses[k], mat_t, light_t, board, st, masks[k]; backend = cpu) for k in 1:2]
        sr_cpu = calibration_problem(board, st, camera_r, poses, mat_t, light_t, targets_r; masks, backend = cpu)
        sr_gpu = calibration_problem(board, st, camera_r, poses, mat_t, light_t, targets_r; masks, backend = CUDABackend())
        imr_cpu = render_all(sr_cpu.f, sr_cpu.u0)[1]; imr_gpu = render_all(sr_gpu.f, sr_gpu.u0)[1]
        @test maximum(abs, imr_gpu .- imr_cpu) / maximum(imr_cpu) < 2e-3
        sr_fit = calibration_problem(board, st, RC.with_vignetting(camera_t, RadialVignetting(5, SVector(16.64, 15.04), 0.6 * W)), poses, mat_t, light_t, targets_r;
                                  masks, backend = CUDABackend())
        _update_vignetting!(sr_fit.f, sr_fit.u0)
        pts_r = [sensor_position(u, v) for (u, v) in sr_fit.f.px[1]]
        @test maximum(abs(sr_fit.u0.camera.vignetting(𝐩) - vig_r(𝐩)) for 𝐩 in pts_r) < 5e-3
        # the LM problem with the radial block on the device: residual parity with the CPU and finite columns
        probr_c, _ = build_problem(sr_cpu.f, sr_cpu.u0); probr_g, _ = build_problem(sr_gpu.f, sr_gpu.u0)
        rr_c = filter(c -> c isa PerViewResidual{9}, probr_c.blocks)[1]; rr_g = filter(c -> c isa PerViewResidual{9}, probr_g.blocks)[1]
        vr_c = RC.NLLSsolver.getvars(rr_c, probr_c.variables); vr_g = RC.NLLSsolver.getvars(rr_g, probr_g.variables)
        @test maximum(abs, RC.NLLSsolver.computeresidual(rr_c, vr_c...) .- RC.NLLSsolver.computeresidual(rr_g, vr_g...)) < 2e-3
        _, Jr_g = RC.NLLSsolver.computeresjac(RC.static(127), rr_g, vr_g...)
        @test size(Jr_g, 2) == 6 + 6 + 9 + 1 + 9 + 3 + 27 && all(isfinite, Jr_g)               # vignetting block: 5 coefficients, the first two tied and, under the hard pin, not a column (V(𝐜) = P_0 = 1)
        # Core's Kannala–Brandt, division and OpenCV-chain lenses on the device (Float32 and duals): the mask and values agree with the host's Float32
        # result, and so do the dual partials. Tolerances fixed before the run: masks equal; values within 1e-5; partials within 1e-3 of
        # the largest host partial (the blob_kb lens, f = 0.179, has partials of order 1e2 near its rim).
        for l in (cam(VgtkCore.KannalaBrandt(-0.03, 0.004, 0.0, 0.0), 41.216, (16.32, 15.36), W, H),
                  cam(VgtkCore.KannalaBrandt(-0.0416, 0.000407, 0.000185, -7.7e-5), 932.2, (2604.0, 1738.0), 5208, 3476; f_y = 931.6), cam(VgtkCore.Division(0.144, -0.293), 43.2, (16.32, 15.36), W, H; f_y = 24.32),
                  lens_o, cam(VgtkIO.opencv_lens(vcat(-3.0, zeros(13))), 25.6, (16.0, 16.0), W, H; f_y = 32.0))  # the OpenCV chain (Task 2b); k₁ = −3 folds inside the lattice
            Wl, Hl = VgtkCore.extent(l)
            Pq = [SVector{2, Float32}(x, y) for x in range(0.0f0, Float32(Wl), length = 64), y in range(0.0f0, Float32(Hl), length = 64)]
            Pq_d = CUDA.CuArray(Pq)
            l32 = VgtkCore.convert_eltype(Float32, l)
            dev = Array(map(p -> chief_ray(l32, p), Pq_d)); host = [chief_ray(l32, p) for p in Pq]
            fin = [all(isfinite, d) for d in host]
            @test fin == [all(isfinite, d) for d in dev] && any(fin)
            @test maximum(i -> fin[i] ? norm(dev[i] - host[i]) : 0.0f0, eachindex(Pq)) < 1f-5
            ld = RC._seeded_lens(RC.lens_manifold(l32), l32, x -> ForwardDiff.Dual{Nothing}(Float32(x), 0.0f0), j -> ForwardDiff.Dual{Nothing}(0.0f0, j == 5 ? 1.0f0 : 0.0f0))   # the first lens coefficient (after f, aspect, cx, cy)
            devd = Array(map(p -> ForwardDiff.partials.(chief_ray(ld, p), 1), Pq_d)); hostd = [ForwardDiff.partials.(chief_ray(ld, p), 1) for p in Pq]
            scale = maximum(i -> fin[i] ? maximum(abs, hostd[i]) : 0.0f0, eachindex(Pq))
            @test maximum(i -> fin[i] ? norm(devd[i] - hostd[i]) : 0.0f0, eachindex(Pq)) < 1f-3 * max(1.0f0, scale)
        end
        # the out-of-domain samples on the device: finite residuals that match the CPU's, and the same higher cost
        let a = dom_args, full = [trues(24, 24)]
            tgt = [render_view(1, RadiometricCameraModel(kb_in, Focus(3, 1.5, 24, 24), 0.0386, a.vig), a.pose, a.mat, a.light, dom_board, a.st, full[1]; backend = cpu)]
            co = RadiometricCameraModel(kb_out, Focus(3, 1.5, 24, 24), 0.0386, a.vig)
            sc_o = calibration_problem(dom_board, a.st, co, [a.pose], a.mat, a.light, tgt; masks = full, backend = cpu)
            sg_o = calibration_problem(dom_board, a.st, co, [a.pose], a.mat, a.light, tgt; masks = full, backend = CUDABackend())
            im_c = render_all(sc_o.f, sc_o.u0)[1]; im_g = render_all(sg_o.f, sg_o.u0)[1]
            @test all(isfinite, im_g) && maximum(abs, im_g .- im_c) / maximum(im_c) < 2e-3
            @test isfinite(cost(sg_o.f, sg_o.u0)[1]) && abs(cost(sg_o.f, sg_o.u0)[1] - cost(sc_o.f, sc_o.u0)[1]) < 2e-3 * cost(sc_o.f, sc_o.u0)[1]
        end
        # timing report (informational)
        t_gpu = @elapsed RC.NLLSsolver.computeresjac(RC.static(127), rg, vgg...)
        t_cpu = @elapsed RC.NLLSsolver.computeresjac(RC.static(127), rc, vc...)
        @info "Jacobian timing (32², 2 views board)" t_cpu t_gpu speedup = t_cpu / t_gpu
    else
        @info "GPU testset skipped (no functional GPU, RC_TEST_GPU=0, or < 20 GiB free)"
    end
end
