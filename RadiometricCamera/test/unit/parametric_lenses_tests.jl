# The Kannala–Brandt and the 14-coefficient (cv2) lenses: against cv2 (test/fixtures/opencv_reference.json: cv2.projectPoints / undistortPoints of a random
# 14-coefficient model with tilt and of a random fisheye model), Core's pixel cameras over them, AD through the inverses, and the per-view
# LM with an 18-coordinate lens block (Jacobian vs finite differences, the GPU path in gpu_tests.jl).
@testitem "Kannala–Brandt / 14-coefficient lenses (cv2, parametric LM)" setup = [RCFixtures] begin
    using RadiometricCamera: build_problem, compute_mask, cost, Focus, FocusSpline, in_domain, lens, lens_type_name, load_lens,
                             load_model, load_parametric_lens, Material, PerViewResidual, PrintedTarget, render_view,
                             RenderSettings, run_lm!, save_lens, Scene, sphere_radius, View, _chief_ray, 𝔟
    using StaticArrays, LinearAlgebra, Statistics, ForwardDiff
    using VisualGeometryToolkitCore: exp_so3
    using VisualGeometryToolkitCore: TensorProductVignetting, with_control_points

    import VisualGeometryToolkitCore as VgtkCore
    import VisualGeometryToolkitIO as VgtkIO
    import VisualGeometryToolkitEstimation as VgtkEstimation
    # the coefficients of a parametric lens: Core's parameter_vector; an OpenCV lens's Θ_d in cv2's order (VisualGeometryToolkitIO's
    # opencv_coefficients), the layout the reference bodies below index
    _kc(l) = VgtkCore.parameter_vector(VgtkCore.lens(l))
    _θd(l) = vcat(pinhole_θ(l), _kc(l))                                    # (f_x, f_y, c_x, c_y in pixels, coefficients…): `camera_of`'s vector
    o = VgtkIO.read_record(joinpath(@__DIR__, "..", "fixtures", "opencv_reference.json"))
    W, H = Int(o.image_size[1]), Int(o.image_size[2])
    _CV_ORDER = Int.(VgtkIO.opencv_coefficients(VgtkCore.lens(camera_of(VgtkIO.opencv_lens(zeros(14)), vcat(ones(4), 1.0:14.0), W, H))))   # Core's index of each cv2 entry
    _θcv(l) = (θ = _θd(l); vcat(θ[1:4], θ[4 .+ _CV_ORDER]))
    # the literal lenses below were drawn with their pinhole relative to the image (f in image widths and heights, c from the image
    # centre in its sizes, then cv2's coefficient order): the camera of a `Wi × Hi` image
    _cv14(θ, Wi = W, Hi = H) = cam(VgtkIO.opencv_lens(θ[5:18]), θ[1] * Wi, ((θ[3] + 0.5) * Wi, (θ[4] + 0.5) * Hi), Wi, Hi; f_y = θ[2] * Hi)
    _rel(lens, f, fy, c1, c2, Wi = W, Hi = H) = cam(lens, f * Wi, ((c1 + 0.5) * Wi, (c2 + 0.5) * Hi), Wi, Hi; f_y = fy * Hi)
    # the lattice of a camera's image and beyond it (one image size on every side), n × n
    _P(l, n) = ((Wl, Hl) = VgtkCore.extent(l); [SVector(x, y) for x in range(-Wl, 2Wl, length = n), y in range(-Hl, 2Hl, length = n)])
    K = [Float64(o.K[i][j]) for i in 1:3, j in 1:3]
    l14 = VgtkCore.BoundedCamera(VgtkIO.opencv_camera(K, Float64.(collect(o.dist))), VgtkCore.Size2(W, H))
    @test length(VgtkCore.parameter_vector(VgtkCore.model(l14))) == 19 && length(_θd(l14)) == 18 && isbitstype(typeof(l14))
    cv2col = VgtkCore.EuclideanMap(VgtkCore.PixelFrame{:opencv}, VgtkCore.PixelFrame{:colmap})   # OpenCV px (pixel centre i) → COLMAP
    @test pinhole_θ(l14)[1] ≈ K[1, 1] && pinhole_θ(l14)[3] ≈ cv2col(SVector(K[1, 3], K[2, 3]))[1]
    cvpx(p) = inv(cv2col)(SVector{2, Float64}(p))                            # COLMAP → OpenCV px
    colpx(px) = cv2col(SVector{2, Float64}(px))
    # forward map = cv2.projectPoints, inverse = cv2.undistortPoints (+ Newton), round trip
    @test maximum(norm(cvpx(VgtkCore.project(l14, SVector(Float64(x[1]), Float64(x[2]), 1.0))) - SVector(Float64(p[1]), Float64(p[2]))) for (x, p) in zip(o.undist_norm, o.proj_px)) < 1e-9
    @test maximum(begin d = RC._chief_ray(l14, colpx(SVector(Float64(p[1]), Float64(p[2])))); norm(SVector(d[1] / d[3], d[2] / d[3]) - SVector(Float64(u[1]), Float64(u[2]))) end
                  for (p, u) in zip(o.px, o.undist_of_px)) < 1e-12
    @test maximum(norm(VgtkCore.project(l14, RC._chief_ray(l14, colpx(SVector(Float64(p[1]), Float64(p[2]))))) - colpx(SVector(Float64(p[1]), Float64(p[2])))) for p in o.px) < 1e-10
    @test all(in_domain(l14, SVector(x, y)) for x in 0.0:300.0:1200.0, y in 0.0:225.0:900.0)
    @test all(norm(RC._chief_ray(l14, SVector(x, y))) ≈ 1 for x in 0.0:300.0:1200.0, y in 0.0:225.0:900.0)
    @test has_lens(camera_of(VgtkIO.opencv_lens(zeros(14)), _θd(l14), W, H), typeof(VgtkIO.opencv_lens(zeros(14)))) && _θd(camera_of(VgtkIO.opencv_lens(zeros(14)), 2 .* _θd(l14), W, H)) == 2 .* _θd(l14)
    @test length(VgtkCore.parameter_names(VgtkCore.lens(l14))) == 14
    pp = VgtkIO.opencv_parameters(VgtkCore.model(l14)); @test pp.K ≈ K && pp.dist ≈ Float64.(collect(o.dist))
    # AD through the Newton inverse (parameters and position) against finite differences; Float32 agreement
    θ = _θd(l14); 𝐩 = SVector(972.0, 252.0)
    Jθ = ForwardDiff.jacobian(t -> RC._chief_ray(camera_of(VgtkIO.opencv_lens(zeros(14)), t, W, H), 𝐩), θ)
    hθ = [i <= 4 ? 1e-3 : 1e-6 for i in 1:18]                                     # the pinhole in pixels, the coefficients unit-free
    Jfd = hcat([(RC._chief_ray(camera_of(VgtkIO.opencv_lens(zeros(14)), θ + hθ[i] * (1:18 .== i), W, H), 𝐩) - RC._chief_ray(camera_of(VgtkIO.opencv_lens(zeros(14)), θ - hθ[i] * (1:18 .== i), W, H), 𝐩)) / 2hθ[i] for i in 1:18]...)
    @test maximum(abs, (Jθ - Jfd) .* hθ') < 1e-7 * maximum(abs, Jfd .* hθ')
    J𝐩 = ForwardDiff.jacobian(q -> RC._chief_ray(l14, q), 𝐩)
    @test norm(J𝐩[:, 1] - (RC._chief_ray(l14, 𝐩 + SVector(1e-3, 0.0)) - RC._chief_ray(l14, 𝐩 - SVector(1e-3, 0.0))) / 2e-3) < 1e-6 * norm(J𝐩[:, 1])
    l32 = VgtkCore.convert_eltype(Float32, l14)
    @test has_lens(l32, typeof(VgtkCore.convert_eltype(Float32, VgtkIO.opencv_lens(zeros(14))))) && norm(RC._chief_ray(l32, SVector{2, Float32}(𝐩)) - RC._chief_ray(l14, 𝐩)) < 1e-6
    @test isbitstype(typeof(View(VgtkCore.EuclideanMap(SMatrix{3, 3}(1.0I), zeros(SVector{3}); check = false), l32, VgtkCore.convert_eltype(Float32, Focus(3, 1.5, W, H)), 0.03f0)))
    # a folded model (k₁ = −3: r(1 − 3r²) turns over at r = 1/√9) leaves in_domain false beyond the fold, true near the axis
    lf = cam(VgtkIO.opencv_lens(vcat(-3.0, zeros(13))), pinhole_θ(l14)[1], (pinhole_θ(l14)[3], pinhole_θ(l14)[4]), W, H; f_y = pinhole_θ(l14)[2])
    @test in_domain(lf, SVector(600.0, 450.0)) && in_domain(lf, SVector(720.0, 495.0)) && !in_domain(lf, SVector(1140.0, 810.0))
    # the lens record (VisualGeometryToolkitIO's camera record: the 14-coefficient chain in COLMAP pixels) and the model specs
    tmp = mktempdir()
    save_lens(joinpath(tmp, "l.json"), l14; provenance = (; model = "test", note = 1))
    lj = load_parametric_lens(joinpath(tmp, "l.json"))
    @test has_lens(lj, typeof(VgtkIO.opencv_lens(zeros(14)))) && maximum(abs, _θd(lj) - _θd(l14)) < 1e-12 && VgtkCore.extent(lj) == VgtkCore.Size2(W, H)
    @test has_lens(load_lens("lens:" * joinpath(tmp, "l.json")), typeof(VgtkIO.opencv_lens(zeros(14))))
    unp, lab = load_model("lens:" * joinpath(tmp, "l.json"), W, H)
    @test lab == "lens:l.json" && unp(𝐩) ≈ RC._chief_ray(l14, 𝐩)
    @test lens_type_name(cam(VgtkCore.Lens(VgtkCore.DoubleSphere(0.2, 0.5)), 1560.0, (600.0, 450.0), W, H)) == "double_sphere"
    save_lens(joinpath(tmp, "ds.json"), cam(VgtkCore.Lens(VgtkCore.DoubleSphere(0.2, 0.5)), 1560.0, (612.0, 450.0), W, H))
    @test has_lens(load_parametric_lens(joinpath(tmp, "ds.json")), VgtkCore.Lens{VgtkCore.DoubleSphere{Float64}, Nothing})

    # Kannala–Brandt: cv2.fisheye, the Core-lens adapter, AD, in_domain
    fo = o.fisheye; Kf = [Float64(fo.K[i][j]) for i in 1:3, j in 1:3]
    cf = VgtkCore.EuclideanMap(VgtkCore.PixelFrame{:opencv}, VgtkCore.PixelFrame{:colmap})(SVector(Kf[1, 3], Kf[2, 3]))   # cv2's centre → COLMAP
    kb = VgtkCore.BoundedCamera(VgtkCore.CameraModel(VgtkCore.LogicalIntrinsics(VgtkCore.CameraCalibrationMat(SMatrix{3, 3, Float64}(Kf[1, 1], 0, 0, 0, Kf[2, 2], 0, cf[1], cf[2], 1))),
                                                     VgtkCore.KannalaBrandt(Float64.(collect(fo.k))...)), VgtkCore.Size2(W, H))
    @test length(_θd(kb)) == 8 && isbitstype(typeof(kb)) && VgtkCore.parameter_names(VgtkCore.lens(kb)) == (:k1, :k2, :k3, :k4)
    @test maximum(norm(cvpx(VgtkCore.project(kb, SVector(Float64(x[1]), Float64(x[2]), 1.0))) - SVector(Float64(p[1]), Float64(p[2]))) for (x, p) in zip(fo.undist_norm, fo.proj_px)) < 1e-9
    @test maximum(begin d = RC._chief_ray(kb, colpx(SVector(Float64(p[1]), Float64(p[2])))); norm(SVector(d[1] / d[3], d[2] / d[3]) - SVector(Float64(u[1]), Float64(u[2]))) end
                  for (p, u) in zip(fo.px, fo.undist_of_px)) < 1e-12
    @test all(in_domain(kb, SVector(x, y)) for x in 0.0:300.0:1200.0, y in 0.0:225.0:900.0)
    θk = _θd(kb)
    hk = [i <= 4 ? 1e-3 : 1e-6 for i in 1:8]
    Jk = ForwardDiff.jacobian(t -> RC._chief_ray(camera_of(VgtkCore.KannalaBrandt(0.0, 0.0, 0.0, 0.0), t, W, H), 𝐩), θk)
    Jkfd = hcat([(RC._chief_ray(camera_of(VgtkCore.KannalaBrandt(0.0, 0.0, 0.0, 0.0), θk + hk[i] * (1:8 .== i), W, H), 𝐩) - RC._chief_ray(camera_of(VgtkCore.KannalaBrandt(0.0, 0.0, 0.0, 0.0), θk - hk[i] * (1:8 .== i), W, H), 𝐩)) / 2hk[i] for i in 1:8]...)
    @test maximum(abs, (Jk - Jkfd) .* hk') < 1e-7 * maximum(abs, Jkfd .* hk')
    @test RC._chief_ray(kb, pinhole_θ(kb)[SVector(3, 4)]) ≈ SVector(0.0, 0.0, 1.0)
    @test has_lens(VgtkCore.convert_eltype(Float32, kb), VgtkCore.KannalaBrandt{4, Float32})
    @test maximum(norm(VgtkCore.project(kb, RC._chief_ray(kb, SVector(x, y))) - SVector(x, y)) for x in 60.0:180.0:1140.0, y in 45.0:135.0:855.0) < 1e-9
    @test pinhole_θ(kb)[1] ≈ Kf[1, 1] && pinhole_θ(kb)[3] ≈ Kf[1, 3] + 0.5
    # the per-view parametric route with an 18-coordinate lens block (Core's ParameterManifold of the pixel camera, the skew fixed): layout,
    # Jacobian vs finite differences of the block's own update, a few LM steps
    tex = fixture_board()
    board = PrintedTarget([-1.0, -1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], tex; facing = [0.0, 0.0, 1.5])
    scene = Scene(board, 24, 24)
    lens_t = _cv14(SVector(1.288, 1.288, 0.01, -0.02, -0.05, 0.01, 1e-3, -2e-3, 0.0, 0.02, 0.0, 0.0, 1e-3, 0.0, -1e-3, 0.0, 0.01, -0.005), 24, 24)
    tps_t = FocusSpline(VgtkCore.grid_nodes(3, VgtkCore.Rect(0, 0, 24, 24))); focus_t = Focus(tps_t, [1 / (1.5 + 0.1 * sin(3i)) for i in 1:9])
    poses = [VgtkCore.EuclideanMap(SMatrix{3, 3}(exp_so3(SVector(π, 0.0, 0.0)) * exp_so3(SVector(0.15 * cos(k), 0.12 * sin(k), 0.1 * k))), SVector{3}(SVector(0.3 * cos(k), 0.3 * sin(k), 1.4 + 0.1 * k)); check = false) for k in 1:2]
    # the truth in the hard pin's gauge V(𝐜) = 1 (`gauge_center_value`; the default pin, 2026-10-30): a state built from it is the truth itself
    vig_t = RC.gauge_center_value(let v = TensorProductVignetting((4, 4), VgtkCore.Rect(0, 0, 24, 24); degree = 2); gr = range(-0.5, 0.5, length = 4); with_control_points(v, [1.0 - 0.4 * (gr[i]^2 + gr[j]^2) for i in 1:4, j in 1:4]) end)
    camera_t = RadiometricCameraModel(lens_t, focus_t, 0.0386, vig_t)
    @test has_lens(View(poses[1], camera_t).lens, typeof(VgtkIO.opencv_lens(zeros(14))))     # the kernels' view keeps the 14-coefficient lens
    b = 𝔟(board); r_SH = sphere_radius(b, [p.t for p in poses])
    a_t = zeros(3, 9); a_t[:, 1] .= (2.0, 1.9, 1.8); a_t[:, 3] .= (0.4, 0.3, 0.2)
    light_t = EnvironmentLight(SphericalHarmonicMap(a_t, 2), b, r_SH)
    mat_t = Material(; 𝛂_ink = 0.12, 𝛂_noink = 0.35, 𝛈_ink = 1.65, 𝛈_noink = 1.33, μ = 0.4)
    st = RenderSettings(; n_𝐩 = 2, n_ρ = 2, n_φ = 6, N = 4)
    masks = [compute_mask(camera_t, p, scene, st; δ = 0.02, backend = RadiometricCamera.KernelAbstractions.CPU()) for p in poses]
    @test all(count(m) > 100 for m in masks)
    targets = [render_view(k, camera_t, poses[k], mat_t, light_t, scene, st, masks[k]) for k in 1:2]
    s = CalibrationState(scene, st, camera_t, poses, mat_t, light_t, targets; masks)
    prob, idx = build_problem(s)
    dv = prob.variables[idx.distortion]
    @test dv isa VgtkCore.SubsetManifold && Int(RC.NLLSsolver.nvars(dv)) == 18 && free_names(dv) == filter(!=(:skew), VgtkCore.parameter_names(VgtkCore.model(lens_t)))
    @test_throws ArgumentError build_problem(s; fixed = (:k7,))
    lay = RC._jacobian_layout(RC.static(127), 9, 4, 15, 16, 9, 18)                 # 15 free vignetting directions of 16 nodes (the hard pin)
    @test lay.off_d == 6 && lay.off_f == 24 && lay.off_m == 28 && lay.off_r == 36 && lay.P == 37 && lay.ncol == 6 + 18 + 9 + 1 + 9 + 15 + 27
    res = get(prob.problem.costs, PerViewResidual{9})[1]
    vars = RC.NLLSsolver.getvars(res, prob.variables)
    r0 = RC.NLLSsolver.computeresidual(res, vars...)
    @test maximum(abs, r0) < 1e-12
    rj, J = RC.NLLSsolver.computeresjac(RC.static(127), res, vars...)
    @test size(J) == (length(r0), lay.ncol) && isapprox(rj, r0; atol = 1e-10)
    pv, dvv, fv, av, mv, vv, lv = vars
    u18(i, h) = (z = zeros(18); z[i] = h; z)
    for (i, h) in ((1, 1e-6), (2, 1e-6), (3, 1e-6), (5, 1e-5), (7, 1e-5), (10, 1e-5), (13, 1e-5), (17, 1e-6))   # f, aspect, cx, tx, k1, d1, p1, sy1 (Core 0.5's order: the tilt first)
        fd = (RC.NLLSsolver.computeresidual(res, pv, RC.NLLSsolver.update(dvv, u18(i, h)), fv, av, mv, vv, lv) -
              RC.NLLSsolver.computeresidual(res, pv, RC.NLLSsolver.update(dvv, u18(i, -h)), fv, av, mv, vv, lv)) / 2h
        # a central difference across a texel kink of the bilinear texture is not the derivative there: the few rows whose ray (or a
        # radiance sample's) lies within h of one disagree, wherever the Monte-Carlo samples happen to fall (measured: ≤ 2 % of the rows
        # carry > 99 % of the disagreement; elsewhere it is ≤ 4e-5) — the kink-filtered Richardson check is jacobian_richardson_tests
        bad = abs.(J[:, 6 + i] - fd) .> 1e-3 .* (abs.(fd) .+ 1e-3 * maximum(abs, fd))
        @test count(bad) <= 0.03 * length(fd)
        @test norm((J[:, 6 + i] - fd)[.!bad]) / norm(fd[.!bad]) < 2e-4
    end
    # the non-lens columns keep their meaning after the lens block (focus column 1, μ, a lighting coefficient)
    u9(i, h) = (z = zeros(9); z[i] = h; z); u27(i, h) = (z = zeros(27); z[i] = h; z)
    for (col, setter) in ((6 + 18 + 1, h -> (pv, dvv, RC.NLLSsolver.update(fv, u9(1, h)), av, mv, vv, lv)),
                          (6 + 18 + 9 + 1 + 9, h -> (pv, dvv, fv, av, RC.NLLSsolver.update(mv, u9(9, h)), vv, lv)),
                          (6 + 18 + 9 + 1 + 9 + 15 + 4, h -> (pv, dvv, fv, av, mv, vv, RC.NLLSsolver.update(lv, u27(4, h)))))
        fd = (RC.NLLSsolver.computeresidual(res, setter(1e-6)...) - RC.NLLSsolver.computeresidual(res, setter(-1e-6)...)) / 2e-6
        @test norm(J[:, col] - fd) / norm(fd) < 2e-3
    end
    # a restricted block and the aspect held (`:aspect` in `fixed`: f_x and f_y move together at their ratio); a few LM iterations from a perturbed lens
    # lower the cost
    probm, _ = build_problem(s; fixed = (:cx, :cy, :tx, :ty, :k3, :d, :p, :sx, :sy, :aspect))
    dm = probm.variables[idx.distortion]; @test free_names(dm) == (:f, :k1, :k2) && Int(RC.NLLSsolver.nvars(dm)) == 3
    l0, l1 = pinhole_θ(block_camera(dm, lens_t)), pinhole_θ(block_camera(RC.NLLSsolver.update(dm, [1e-3, 0.0, 0.0]), lens_t))
    @test l1[1] ≈ l0[1] * exp(1e-3) && l1[2] / l1[1] ≈ l0[2] / l0[1] && l1[3] == l0[3]
    # the start: f perturbed by 0.48 px and k1 by 0.03, k1 found by NAME — Core 0.5's composed lens (Tilt ∘ Additive) lists the tilt
    # (:tx, :ty) first, so the fifth coordinate is no longer k1 (a tilt start the restricted block (:f, :aspect, :k1) cannot undo)
    ik1 = 4 + findfirst(==(:k1), VgtkCore.parameter_names(VgtkCore.lens(lens_t)))
    lens_start = camera_of(VgtkIO.opencv_lens(zeros(14)), _θd(lens_t) + SVector{18}(ntuple(i -> i <= 2 ? 0.48 : i == ik1 ? 0.03 : 0.0, 18)), 24, 24)
    s2 = CalibrationState(scene, st, RadiometricCameraModel(lens_start, focus_t, 0.0386, vig_t), poses, mat_t, light_t, targets; masks)
    c0 = cost(s2)[1]
    its = Int[]
    result = run_lm!(s2; maxiters = 4, fixed = (:poses, :focus, :aperture, :material, :vignetting, :lighting,
                             :cx, :cy, :tx, :ty, :k2, :k3, :d, :p, :sx, :sy), iter_callback = (it, c, camera, poses, mat, surface, light) -> push!(its, it))
    @test result.cost / 2 < 0.05 * c0 && length(its) >= 1 && has_lens(RC.lens(s2), typeof(VgtkIO.opencv_lens(zeros(14))))
    @test abs(pinhole_θ(RC.lens(s2))[1] - pinhole_θ(lens_t)[1]) < 0.24

    # Kannala–Brandt on Core's `KannalaBrandt(k…)` (Task 2 of the VisualGeometryToolkitCore port, fix round 1): equivalence against the
    # deleted RC body (fixed 10-step Newton from θ = θ_d), copied here verbatim as the reference. Tolerances fixed before the run, from
    # the old body's own convergence (it stops at |θ_d(θ) − ‖m‖| < 1e-9 only in its mask test; on these lenses it reaches roundoff):
    # π⁻¹ within 1e-11 and `project` within 1e-13 where both are defined, the parameter Jacobian through duals within 1e-10 relative.
    # The domain is Core's (a finite back-projection): on the lenses of the goldens, the tests and a folded lens it equals the old mask
    # on a dense lattice reaching past the fold; on the lenses where the old fixed-count Newton failed or landed past the fold it differs
    # (the recorded bug fix), pinned below.
    @testset "the Kannala–Brandt lens on Core's KannalaBrandt: equivalence with the deleted body" begin
        function kb_poly_old(l, θ)
            θ2 = θ * θ; θ4 = θ2 * θ2; θ6 = θ2 * θ4; θ8 = θ4 * θ4
            return θ * (1 + _kc(l)[1] * θ2 + _kc(l)[2] * θ4 + _kc(l)[3] * θ6 + _kc(l)[4] * θ8), 1 + 3_kc(l)[1] * θ2 + 5_kc(l)[2] * θ4 + 7_kc(l)[3] * θ6 + 9_kc(l)[4] * θ8
        end
        function kb_theta_old(l, θd)
            θ = θd; fp = one(θd)
            for _ in 1:10
                f, fp = kb_poly_old(l, θ)
                θ -= (f - θd) / (abs(fp) > eps(typeof(fp)) ? fp : (fp >= 0 ? eps(typeof(fp)) : -eps(typeof(fp))))
            end
            return θ, fp
        end
        function π⁻¹_old(l, 𝐩)
            q = pinhole_θ(l); m_x = (𝐩[1] - q[3]) / q[1]; m_y = (𝐩[2] - q[4]) / q[2]; θd = sqrt(m_x * m_x + m_y * m_y)
            θ, _ = kb_theta_old(l, θd); s = θd > 0 ? sin(θ) / θd : one(θd)
            return SVector(s * m_x, s * m_y, cos(θ))
        end
        function in_domain_old(l, 𝐩)
            q = pinhole_θ(l); m_x = (𝐩[1] - q[3]) / q[1]; m_y = (𝐩[2] - q[4]) / q[2]; θd = sqrt(m_x * m_x + m_y * m_y)
            θ, fp = kb_theta_old(l, θd); f, _ = kb_poly_old(l, θ)
            return isfinite(θ) && fp > 0 && abs(f - θd) < 1e-9 && 0 <= θ < π
        end
        function project_old(l, X)
            ρ = sqrt(X[1] * X[1] + X[2] * X[2]); θ = atan(ρ, X[3]); θd, _ = kb_poly_old(l, θ); s = ρ > 0 ? θd / ρ : one(ρ)
            q = pinhole_θ(l)
            return SVector(q[1] * s * X[1] + q[3], q[2] * s * X[2] + q[4])
        end
        P = _P(kb, 151)
        for l in (_rel(VgtkCore.KannalaBrandt(-0.03, 0.004, 0.0, 0.0), 1.288, 1.288, 0.01, -0.02),      # the goldens' lens
                  _rel(VgtkCore.KannalaBrandt(-0.05, 0.01, 0.0, 0.0), 0.9, 0.9, 0.0, 0.0), kb,             # test_unit_chief's, the cv2 fixture's
                  _rel(VgtkCore.KannalaBrandt(-0.3, 0.0, 0.0, 0.0), 0.2, 0.2, 0.0, 0.0))                   # folds at θ = 1.05, inside the lattice
            @test all(in_domain(l, 𝐩) == in_domain_old(l, 𝐩) for 𝐩 in P)
            @test maximum(𝐩 -> in_domain(l, 𝐩) ? norm(_chief_ray(l, 𝐩) - π⁻¹_old(l, 𝐩)) : 0.0, P) < 1e-11
            @test maximum(𝐩 -> in_domain(l, 𝐩) ? norm(VgtkCore.project(l, _chief_ray(l, 𝐩)) - project_old(l, _chief_ray(l, 𝐩))) : 0.0, P) < 1e-10
            𝐩 = SVector(660.0, 423.0)                                                           # inside every lens's domain
            Jo = ForwardDiff.jacobian(t -> π⁻¹_old(camera_of(VgtkCore.KannalaBrandt(0.0, 0.0, 0.0, 0.0), t, W, H), 𝐩), _θd(l))
            @test maximum(abs, ForwardDiff.jacobian(t -> _chief_ray(camera_of(VgtkCore.KannalaBrandt(0.0, 0.0, 0.0, 0.0), t, W, H), 𝐩), _θd(l)) - Jo) < 1e-10 * max(1.0, maximum(abs, Jo))
            Jq = ForwardDiff.jacobian(q -> π⁻¹_old(l, q), 𝐩)
            @test maximum(abs, ForwardDiff.jacobian(q -> _chief_ray(l, q), 𝐩) - Jq) < 1e-10 * max(1.0, maximum(abs, Jq))
        end
        # the bug fixes: the old 10-step Newton did not converge (the old mask under-reported) …
        lu = _rel(VgtkCore.KannalaBrandt(0.5, -0.4, 0.1, 0.0), 0.3, 0.3, 0.0, 0.0)
        @test count(in_domain(lu, 𝐩) for 𝐩 in P) > count(in_domain_old(lu, 𝐩) for 𝐩 in P)
        @test all(!in_domain_old(lu, 𝐩) || in_domain(lu, 𝐩) for 𝐩 in P)
        # Core's root is a root: within 1e-9 (re-stated after a first run at 1e-12 measured 1.8e-10 — Core's solve stops at a residual of
        # 1e-10 on the radial ratio, which this steep polynomial (θ_d′ ≈ 100 at θ = 2.5) amplifies in the image)
        @test maximum(𝐩 -> in_domain(lu, 𝐩) ? norm(VgtkCore.project(lu, _chief_ray(lu, 𝐩)) - 𝐩) : 0.0, P) < 1e-9 * W
        # … or landed on the far branch past the fold (the old mask accepted a second preimage)
        lf = _rel(VgtkCore.KannalaBrandt(-0.49173344963610416, 0.037315197007153224, 0.03316026423284343, -0.005033188743362174), 0.5985644145049734, 0.6284672124839231, -0.01982355235865314, -0.015258732442595378)
        S = [SVector(x, y) for x in range(0.0, Float64(W), length = 101), y in range(0.0, Float64(H), length = 101)]
        far = [𝐩 for 𝐩 in S if in_domain_old(lf, 𝐩) && !in_domain(lf, 𝐩)]
        @test !isempty(far) && all(acos(clamp(π⁻¹_old(lf, 𝐩)[3], -1, 1)) > 0.92 for 𝐩 in far)          # every one past the fold θ = 0.92
        @test all(!in_domain(lf, 𝐩) || in_domain_old(lf, 𝐩) for 𝐩 in S)
        # Float32 (the GPU precision)
        l32 = VgtkCore.convert_eltype(Float32, kb)
        @test _chief_ray(l32, SVector{2, Float32}(972.0, 252.0)) isa SVector{3, Float32} &&
              norm(_chief_ray(l32, SVector{2, Float32}(972.0, 252.0)) - _chief_ray(kb, SVector(972.0, 252.0))) < 1e-5
    end

    # The 14-coefficient lens on Core's OpenCV chain (Task 2b of the VisualGeometryToolkitCore port): equivalence against the deleted RC body (tilt
    # inverted exactly, then a fixed 12-step Newton on the distortion with its analytic Jacobian; the mask: Newton residual < 1e-9, g > 0,
    # d(rg)/dr > 0, det J > 0, de-tilted w > 0), copied here verbatim as the reference. Tolerances fixed before the run: π⁻¹ within 1e-12
    # and `project` within 1e-13 where both are defined (measured 5.9e-15 and 1.1e-15), the dual Jacobians within 1e-10 relative, the
    # masks equal on 121² points over [−1.5, 1.5]² (past the folds) for the fixture's, the tests' and the GPU test's lenses and a folded
    # lens. Where the masks differ on random in-bounds lenses, every difference is an RC defect (classified against a flood fill of the
    # axis component, Task 2b report): the old Newton missed an imaged root, or its local test accepted a root past the rational factor's
    # first fold or pole — pinned below on one lens of each kind.
    include(pkgdir(VgtkCore, "test", "fixtures", "branch_component.jl"))   # Core's axis-component fixture, the one copy
    module LensToolsTest                                      # the experiments' lens helpers (experiments/lib/lens_tools.jl, retired
    using RadiometricCamera                                   # 2026-10-29: these two are what this test pins), in a module of their own
    using StaticArrays: SVector
    import VisualGeometryToolkitCore as VgtkCore
    const RC = RadiometricCamera
    # The detections `feats[k] = [(𝐩, X)]` that `lens` images both ways under the view's pose `poses[k] = (R, t)` (X_cam = R X + t): a ray
    # at 𝐩 (π⁻¹ finite) and a finite projection of the board point; (feats′, the number dropped).
    function imaged_features(lens, poses, feats)
        keep(𝐩, X, R, t) = all(isfinite, RC._chief_ray(lens, 𝐩)) && all(isfinite, VgtkCore.project(lens, R * X + t))
        out = [[(𝐩, X) for (𝐩, X) in fk if keep(𝐩, X, poses[k][1], poses[k][2])] for (k, fk) in enumerate(feats)]
        return out, sum(length, feats; init = 0) - sum(length, out; init = 0)
    end
    # The detections as one vector of correspondences `X => 𝐩` per view: the `views` of VisualGeometryToolkitEstimation's refine_camera_and_poses.
    views_in_pixels(feats) = [[SVector{3, Float64}(X) => SVector{2, Float64}(𝐩) for (𝐩, X) in fk] for fk in feats]
    end
    using Random: MersenneTwister
    @testset "the 14-coefficient lens on Core's OpenCV chain: equivalence with the deleted body" begin
        function tilt_matrices_old(τx, τy)
            cx = cos(τx); sx = sin(τx); cy = cos(τy); sy = sin(τy)
            R = SMatrix{3, 3}(cy, zero(cx), sy, sy * sx, cx, -cy * sx, -sy * cx, sx, cy * cx)
            o = zero(cx); i = one(cx)
            Pz = SMatrix{3, 3}(R[3, 3], o, o,  o, R[3, 3], o,  -R[1, 3], -R[2, 3], i)
            Pi = SMatrix{3, 3}(i / R[3, 3], o, o,  o, i / R[3, 3], o,  R[1, 3] / R[3, 3], R[2, 3] / R[3, 3], i)
            return Pz * R, R' * Pi
        end
        function distort_and_jacobian_old(l, x, y)
            θ = _θcv(l)
            k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4 = θ[5], θ[6], θ[7], θ[8], θ[9], θ[10], θ[11], θ[12], θ[13], θ[14], θ[15], θ[16]
            s = x * x + y * y
            N = 1 + s * (k1 + s * (k2 + s * k3)); D = 1 + s * (k4 + s * (k5 + s * k6))
            dN = k1 + s * (2k2 + 3s * k3); dD = k4 + s * (2k5 + 3s * k6)
            g = N / D; dg = (dN * D - N * dD) / (D * D)
            xd = x * g + 2p1 * x * y + p2 * (s + 2x * x) + s1 * s + s2 * s * s
            yd = y * g + p1 * (s + 2y * y) + 2p2 * x * y + s3 * s + s4 * s * s
            J11 = g + 2x * x * dg + 2p1 * y + 6p2 * x + 2s1 * x + 4s2 * s * x
            J12 = 2x * y * dg + 2p1 * x + 2p2 * y + 2s1 * y + 4s2 * s * y
            J21 = 2x * y * dg + 2p1 * x + 2p2 * y + 2s3 * x + 4s4 * s * x
            J22 = g + 2y * y * dg + 6p1 * y + 2p2 * x + 2s3 * y + 4s4 * s * y
            return xd, yd, J11, J12, J21, J22
        end
        function undistort_old(l, 𝐩)
            m = SVector((𝐩[1] - _θcv(l)[3]) / _θcv(l)[1], (𝐩[2] - _θcv(l)[4]) / _θcv(l)[2])                  # (pixels: the pinhole of K)
            _, Mi = tilt_matrices_old(_θcv(l)[17], _θcv(l)[18])
            v = Mi * SVector(m[1], m[2], one(m[1]))
            w = v[3] != 0 ? 1 / v[3] : one(v[3])
            xt = v[1] * w; yt = v[2] * w
            x = xt; y = yt
            rx = zero(x); ry = zero(y)
            for _ in 1:12
                xd, yd, J11, J12, J21, J22 = distort_and_jacobian_old(l, x, y)
                rx = xd - xt; ry = yd - yt
                det = J11 * J22 - J12 * J21
                idet = 1 / (abs(det) > eps(typeof(det)) ? det : (det >= 0 ? eps(typeof(det)) : -eps(typeof(det))))
                x -= (J22 * rx - J12 * ry) * idet
                y -= (-J21 * rx + J11 * ry) * idet
            end
            return SVector(x, y), sqrt(rx * rx + ry * ry), v[3]
        end
        function π⁻¹_old(l, 𝐩)
            x, _, _ = undistort_old(l, 𝐩)
            n = sqrt(x[1] * x[1] + x[2] * x[2] + 1)
            return SVector(x[1] / n, x[2] / n, 1 / n)
        end
        function radial_factor_old(l, x, y)
            θ = _θcv(l); s = x * x + y * y
            N = 1 + s * (θ[5] + s * (θ[6] + s * θ[9])); D = 1 + s * (θ[10] + s * (θ[11] + s * θ[12]))
            dN = θ[5] + s * (2θ[6] + 3s * θ[9]); dD = θ[10] + s * (2θ[11] + 3s * θ[12])
            g = N / D
            return g, g + 2s * (dN * D - N * dD) / (D * D)
        end
        function in_domain_old(l, 𝐩)
            x, res, w = undistort_old(l, 𝐩)
            (isfinite(x[1]) && isfinite(x[2]) && isfinite(res)) || return false
            _, _, J11, J12, J21, J22 = distort_and_jacobian_old(l, x[1], x[2])
            g, dR = radial_factor_old(l, x[1], x[2])
            return res < 1e-9 && g > 0 && dR > 0 && (J11 * J22 - J12 * J21) > 0 && w > 0
        end
        function project_old(l, X)
            xd, yd, _, _, _, _ = distort_and_jacobian_old(l, X[1] / X[3], X[2] / X[3])
            M, _ = tilt_matrices_old(_θcv(l)[17], _θcv(l)[18])
            v = M * SVector(xd, yd, one(xd))
            w = v[3] != 0 ? 1 / v[3] : one(v[3])
            return SVector(_θcv(l)[1] * v[1] * w + _θcv(l)[3], _θcv(l)[2] * v[2] * w + _θcv(l)[4])
        end
        for l in (l14, lens_t, lens_start, lf)  # fixture, tests/GPU, its LM start, k₁ = −3
            P = _P(l, 121); Wl, Hl = VgtkCore.extent(l)
            @test all(in_domain(l, 𝐩) == in_domain_old(l, 𝐩) for 𝐩 in P)
            @test maximum(𝐩 -> in_domain(l, 𝐩) ? norm(_chief_ray(l, 𝐩) - π⁻¹_old(l, 𝐩)) : 0.0, P) < 1e-12
            @test maximum(𝐩 -> in_domain(l, 𝐩) ? norm(VgtkCore.project(l, _chief_ray(l, 𝐩)) - project_old(l, _chief_ray(l, 𝐩))) : 0.0, P) < 1e-13 * Wl
            # the dual Jacobians near the axis, halfway out and at the in-domain lattice point farthest out (next to the fold for the
            # k₁ = −3 lens): measured ≤ 6.4e-14 relative
            ctr = SVector(Wl / 2, Hl / 2)
            ins = [𝐩 for 𝐩 in P if in_domain(l, 𝐩)]; rmax = maximum(q -> norm(q - ctr), ins)
            for 𝐩 in (SVector(0.55Wl, 0.47Hl), ins[argmin([abs(norm(q - ctr) - rmax / 2) for q in ins])], ins[argmax([norm(q - ctr) for q in ins])])
                Jo = ForwardDiff.jacobian(t -> π⁻¹_old(camera_of(VgtkIO.opencv_lens(zeros(14)), t, Wl, Hl), 𝐩), _θd(l))
                @test maximum(abs, ForwardDiff.jacobian(t -> _chief_ray(camera_of(VgtkIO.opencv_lens(zeros(14)), t, Wl, Hl), 𝐩), _θd(l)) - Jo) < 1e-10 * max(1.0, maximum(abs, Jo))
                Jq = ForwardDiff.jacobian(q -> π⁻¹_old(l, q), 𝐩)
                @test maximum(abs, ForwardDiff.jacobian(q -> _chief_ray(l, q), 𝐩) - Jq) < 1e-10 * max(1.0, maximum(abs, Jq))
            end
            Jx = ForwardDiff.jacobian(X -> VgtkCore.project(l, X), SVector(0.1, -0.2, 1.0))
            @test maximum(abs, Jx - ForwardDiff.jacobian(X -> project_old(l, X), SVector(0.1, -0.2, 1.0))) < 1e-12 * max(1.0, maximum(abs, Jx))
        end
        P = _P(l14, 121)                                      # the lattice of the fixture's image (the lenses below share it)
        # the bug fixes: the old fixed-count Newton missed imaged roots (the old mask under-reported; Core's roots round-trip and satisfy the
        # old conditions) …
        lu = _cv14(SVector(1.2932193114954187, 1.1804152206469811, 0.031077443711510826, -0.034789335521032444, 1.4046820946686671,
                                -0.00298835153650821, 0.10369492602072385, -0.08036761807002042, 0.06640605662710539, -2.0331771798674327,
                                0.38773693444882806, 0.09184484706698552, -0.02767889985752991, -0.004341974832778656, 0.005671379339665546,
                                -0.0018545536464766863, -0.03269226430994721, -0.025050333188344983))
        gained = [𝐩 for 𝐩 in P if in_domain(lu, 𝐩) && !in_domain_old(lu, 𝐩)]
        @test length(gained) > 1000 && all(!in_domain_old(lu, 𝐩) || in_domain(lu, 𝐩) for 𝐩 in P)
        @test maximum(norm(VgtkCore.project(lu, _chief_ray(lu, 𝐩)) - 𝐩) for 𝐩 in gained) < 1e-9 * W
        function old_conditions(l, 𝐩, x)                     # the deleted mask's conditions (but the Newton residual) at a given root x = (x', y')
            m = SVector((𝐩[1] - _θcv(l)[3]) / _θcv(l)[1], (𝐩[2] - _θcv(l)[4]) / _θcv(l)[2])
            _, Mi = tilt_matrices_old(_θcv(l)[17], _θcv(l)[18]); w = (Mi * SVector(m[1], m[2], 1.0))[3]
            _, _, J11, J12, J21, J22 = distort_and_jacobian_old(l, x[1], x[2]); g, dR = radial_factor_old(l, x[1], x[2])
            return g > 0 && dR > 0 && J11 * J22 - J12 * J21 > 0 && w > 0
        end
        @test all(old_conditions(lu, 𝐩, VgtkCore.to_euclidean(_chief_ray(lu, 𝐩))) for 𝐩 in gained)   # the root x = (x', y'): the ray dehomogenised
        # … or its local test accepted a root past the rational factor's first fold or pole (a second preimage; the old mask over-reported)
        lo = _cv14(SVector(0.5135197409036605, 0.5377846170745176, 0.015410813940947269, 0.010569859546944045, -0.1509936742262308,
                                -0.05018233099799057, 0.002444691139998182, 0.005751991767387469, 0.023916020849413303, 0.18709720259547577,
                                0.02550856299058555, -0.0012866065737144083, -0.0015778062511211863, -0.0009697989382005757,
                                -0.0007628785407464717, 0.0021057123411919506, -0.009925873758215435, -0.0062453983699797))
        θo = _θcv(lo); Nf(s) = 1 + s * (θo[5] + s * (θo[6] + s * θo[9])); Df(s) = 1 + s * (θo[10] + s * (θo[11] + s * θo[12]))
        ρb = let ρ = 0.0, pr = 0.0; while ρ < 50; ρn = ρ + 1e-4; (Df(ρn^2) > 0 && ρn * Nf(ρn^2) / Df(ρn^2) > pr) || break; ρ, pr = ρn, ρn * Nf(ρn^2) / Df(ρn^2); end; ρ; end
        far = [𝐩 for 𝐩 in P if in_domain_old(lo, 𝐩) && !in_domain(lo, 𝐩)]
        @test length(far) > 1000 && all(norm(undistort_old(lo, 𝐩)[1]) > ρb for 𝐩 in far)            # every one past the fold (ρ ≈ 1.11)
        @test all(!in_domain(lo, 𝐩) || in_domain_old(lo, 𝐩) for 𝐩 in P)
        # … or both accept with different roots, Core's inside the rational factor's branch radius and the old one beyond it (random lens #28)
        l28 = _cv14(SVector(1.1005180978458666, 1.0847187054947218, -0.0192640030461103, 0.0043891982852572045, 0.8921405529397147,
                                 -0.5084821007886162, -0.008534187308990948, -0.005506168844860524, -0.06347362156292047, 0.4029969369998405,
                                 -0.8623919371120291, 0.13759282039735848, -0.019471951657020366, -6.579762770258748e-5, -0.0059362784294636995,
                                 0.008116914651566842, 0.1343776475729164, -0.14137932872911954))
        θ28 = _θcv(l28); ρ28 = radial_branch_radius((θ28[5], θ28[6], θ28[9]), (θ28[10], θ28[11], θ28[12]), 50.0; h = 1e-4)    # 1.265
        two = [𝐩 for 𝐩 in P if in_domain(l28, 𝐩) && in_domain_old(l28, 𝐩) && norm(_chief_ray(l28, 𝐩) - π⁻¹_old(l28, 𝐩)) > 1e-9]
        @test length(two) == 4151 && all(norm(VgtkCore.to_euclidean(_chief_ray(l28, 𝐩))) < ρ28 < norm(undistort_old(l28, 𝐩)[1]) for 𝐩 in two)

        # The classifier of the Task 2b report, on Task 2's 32 random in-bounds lenses (MersenneTwister(7)): every point where the two masks
        # or roots disagree, judged against the axis component of {det J > 0, rational factor before its first fold or pole} (flood fill,
        # Core's test fixture branch_component.jl, independent of both inverses). Counts measured on julia 1.12 (default and --check-bounds=yes) and pinned:
        # every disagreement is an RC defect, none is Core's.
        function randocv(rng, s)
            f = 0.5 + rand(rng)
            k = s .* randn(rng, 12) .* SVector(0.3, 0.1, 0.01, 0.01, 0.03, 0.2, 0.1, 0.03, 0.005, 0.002, 0.005, 0.002)
            _cv14(SVector(f, f * (0.9 + 0.2rand(rng)), 0.02randn(rng), 0.02randn(rng), k..., s * 0.03randn(rng), s * 0.03randn(rng)))
        end
        rs = (rng = MersenneTwister(7); [randocv(rng, s) for s in (0.5, 1.0, 2.0, 4.0) for _ in 1:8])
        function distort_tilt_old(l, x)                       # the deleted forward map in the normalised plane (before f, c)
            xd, yd, _, _, _, _ = distort_and_jacobian_old(l, x[1], x[2])
            M, _ = tilt_matrices_old(_θcv(l)[17], _θcv(l)[18]); v = M * SVector(xd, yd, one(xd))
            return SVector(v[1] / v[3], v[2] / v[3])
        end
        function classify_inverse(l)
            c = Dict{Symbol, Int}(); bump(k) = (c[k] = get(c, k, 0) + 1)
            recs = [(𝐩, in_domain_old(l, 𝐩), in_domain(l, 𝐩)) for 𝐩 in P]
            rmax = maximum(r -> max(r[2] ? norm(undistort_old(l, r[1])[1]) : 0.0, r[3] ? norm(VgtkCore.to_euclidean(_chief_ray(l, r[1]))) : 0.0), recs; init = 0.5)
            θ = _θcv(l); ρ = radial_branch_radius((θ[5], θ[6], θ[9]), (θ[10], θ[11], θ[12]), 1.05rmax)
            ac = axis_component(x -> distort_tilt_old(l, x), ρ, 1.05rmax, 1201)
            for (𝐩, a, b) in recs
                if a && b
                    norm(_chief_ray(l, 𝐩) - π⁻¹_old(l, 𝐩)) <= 1e-9 && (bump(:same); continue)
                    ic, io = in_component(ac, VgtkCore.to_euclidean(_chief_ray(l, 𝐩))), in_component(ac, undistort_old(l, 𝐩)[1])
                    bump(ic && !io ? :differ_rc_off_component : :differ_core_bug_or_policy)
                elseif a
                    bump(in_component(ac, undistort_old(l, 𝐩)[1]) ? :rc_only_core_bug : :rc_only_rc_off_component)
                elseif b
                    x = VgtkCore.to_euclidean(_chief_ray(l, 𝐩))
                    bump(in_component(ac, x) && old_conditions(l, 𝐩, x) ? :core_only_rc_missed : :core_only_core_bug_or_policy)
                else
                    bump(:neither)
                end
            end
            return c
        end
        inv_counts = mergewith(+, (classify_inverse(l) for l in rs)...)
        @test inv_counts == Dict(:same => 263314, :neither => 141285, :rc_only_rc_off_component => 28213, :core_only_rc_missed => 31439,
                                 :differ_rc_off_component => 4261)

        # The forward map: Core's `project` is NaN behind the camera and past the rational factor's first fold or pole, where the deleted body
        # returned a finite (wrong) point; elsewhere the two agree (relative 1e-12; measured 3.8e-9 absolute at |𝐩| ~ 1e8 near 90°). Rays on
        # 121² points of the equidistant chart over [−π, π]² (θ ≤ π): counts per category pinned, and no ray Core images that the old body did not.
        function classify_forward(l)
            θ = _θcv(l); ρ = radial_branch_radius((θ[5], θ[6], θ[9]), (θ[10], θ[11], θ[12]), 50.0; h = 1e-4)
            c = Dict{Symbol, Int}(); bump(k) = (c[k] = get(c, k, 0) + 1); dev = 0.0
            for a in range(-π, π, length = 121), b in range(-π, π, length = 121)
                t = hypot(a, b); t > π && continue
                X = t == 0 ? SVector(0.0, 0.0, 1.0) : SVector(sin(t) * a / t, sin(t) * b / t, cos(t))
                po = project_old(l, X); pn = VgtkCore.project(l, X); fo, fn = all(isfinite, po), all(isfinite, pn)
                if fo && fn
                    bump(:both_finite); dev = max(dev, norm(pn - po) / max(1.0, norm(po)))
                elseif fo
                    bump(X[3] <= 0 ? :nan_behind : hypot(X[1] / X[3], X[2] / X[3]) >= ρ ? :nan_past_fold : :nan_other)
                else
                    bump(fn ? :finite_where_old_nan : :both_nan)
                end
            end
            return c, dev
        end
        fwd = [classify_forward(l) for l in (l14, lens_t, lf)]
        @test first.(fwd) == [Dict(:both_finite => 1281, :nan_behind => 8468, :nan_past_fold => 1540), Dict(:both_finite => 2821, :nan_behind => 8468),
                              Dict(:both_finite => 121, :nan_behind => 8468, :nan_past_fold => 2700)]
        fwd_r = [classify_forward(l) for l in rs]
        @test mergewith(+, first.(fwd_r)...) == Dict(:both_finite => 37540, :nan_behind => 270976, :nan_past_fold => 52732)
        @test maximum(last, vcat(fwd, fwd_r)) < 1e-12
        # The experiments' joint LM (VisualGeometryToolkitEstimation's refine_camera_and_poses, on the views of lens_tools.jl's
        # views_in_pixels) is given only the detections with a ray and a finite projection under the view's pose (imaged_features), so its
        # reprojection residual stays finite; it refuses a point the guess does not image: one imaged detection, one past the fold, one behind
        X_in = SVector(0.05, -0.04, 1.0); X_fold = SVector(0.6, 0.5, 1.0); X_back = SVector(0.1, 0.1, -1.0)
        feats_t = [[(VgtkCore.project(lf, X_in), X_in), (SVector(1140.0, 810.0), X_fold), (SVector(600.0, 450.0), X_back)]]
        poses_t = [(SMatrix{3, 3, Float64}(I), zero(SVector{3, Float64}))]
        kept, nskip = LensToolsTest.imaged_features(lf, poses_t, feats_t)
        @test nskip == 2 && kept == [[feats_t[1][1]]]
        cam_t = VgtkCore.model(lf); P_t = VgtkCore.EuclideanMap(poses_t[1]...; check = false)
        rows(fs) = [VgtkCore.reprojection_error(cam_t, P_t, first(c), last(c)) for c in only(LensToolsTest.views_in_pixels(fs))]
        @test any(r -> any(isnan, r), rows(feats_t)) && all(r -> all(isfinite, r), rows(kept))
        @test_throws ArgumentError VgtkEstimation.refine_camera_and_poses(cam_t, [P_t], LensToolsTest.views_in_pixels(feats_t); free = ())
        # Float32 (the GPU precision): through the same Core chain
        l32 = VgtkCore.convert_eltype(Float32, l14)
        @test _chief_ray(l32, SVector{2, Float32}(972.0, 252.0)) isa SVector{3, Float32} &&
              norm(_chief_ray(l32, SVector{2, Float32}(972.0, 252.0)) - π⁻¹_old(l14, SVector(972.0, 252.0))) < 1e-6
        @test eltype(VgtkCore.project(l32, SVector{3, Float32}(0.1, -0.2, 1))) === Float32
    end

    # Owner rule 2026-10-26: Kannala–Brandt is omnidirectional — it images rays past 90° (z < 0) up to its first fold / θ < π, and they
    # round-trip. Tolerance fixed before the run: 1e-12 (measured 1.7e-16 by the review).
    let l = cam(VgtkCore.KannalaBrandt(-0.0416, 0.000407, 0.000185, -7.7e-5), 932.2, (2604.0, 1738.0), 5208, 3476; f_y = 931.6),   # the blob_kb lens
        S = [SVector(x, y) for x in range(0.0, 5208.0, length = 101), y in range(0.0, 3476.0, length = 101)]
        back = [𝐩 for 𝐩 in S if in_domain(l, 𝐩) && _chief_ray(l, 𝐩)[3] < 0]
        @test length(back) > 100
        @test maximum(norm(VgtkCore.project(l, _chief_ray(l, 𝐩)) - 𝐩) for 𝐩 in back) < 1e-9
    end
end
