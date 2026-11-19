# The refinement LM's termination record (start/prepare.jl `lm_termination`, VisualGeometryToolkitCore's `lmsolve_chart!` on the LM's
# `LmProblem`): Core 0.11.0 (owner ruling 2026-11-15) — a tolerance stop (the step size `xtol`, the cost decrease `ftol`) is `Success`,
# converged by definition, with no stationarity postcondition; the caps are not converged (one uniform convergence record, Core's).
@testitem "refinement termination: a step-size stop is Success (Core 0.11.0: a tolerance stop is converged)" setup = [RCFixtures] begin
    using RadiometricCamera: compute_mask, cost, Focus, FocusSpline, Material, PrintedTarget, render_view, RenderSettings,
                             run_lm!, Scene, sphere_radius, View, 𝔟
    using StaticArrays, LinearAlgebra
    using VisualGeometryToolkitCore: exp_so3
    import VisualGeometryToolkitCore as VgtkCore

    tex = fixture_board()
    board = PrintedTarget([-1.0, -1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], tex; facing = [0.0, 0.0, 1.5])
    W = H = 24
    scene = Scene(board, W, H)
    lens_t = cam(VgtkCore.Lens(VgtkCore.DoubleSphere(0.25, 0.4)), 30.912, (12.24, 11.52), W, H)
    tps_t = FocusSpline(VgtkCore.grid_nodes(3, VgtkCore.Rect(0, 0, W, H)))
    focus_t = Focus(tps_t, [1 / (1.5 + 0.1 * sin(3i)) for i in 1:9])
    poses = [VgtkCore.EuclideanMap(SMatrix{3, 3}(exp_so3(SVector(π, 0.0, 0.0)) * exp_so3(SVector(0.15 * cos(k), 0.12 * sin(k), 0.1 * k))),
                                   SVector{3}(SVector(0.3 * cos(k), 0.3 * sin(k), 1.4 + 0.1 * k)); check = false) for k in 1:2]
    views_t = [View(p, lens_t, focus_t, 0.0386) for p in poses]
    vig_t = RC.gauge_center_value(VgtkCore.TensorProductVignetting((4, 4), VgtkCore.Rect(0, 0, W, H); degree = 2))
    b = 𝔟(board); r_SH = sphere_radius(b, [p.t for p in poses])
    a_t = zeros(3, 9); a_t[:, 1] .= (2.0, 1.9, 1.8); a_t[:, 3] .= (0.4, 0.3, 0.2)
    light_t = EnvironmentLight(SphericalHarmonicMap(a_t, 2), b, r_SH)
    mat_t = Material(; 𝛂_ink = 0.12, 𝛂_noink = 0.35, 𝛈_ink = 1.65, 𝛈_noink = 1.33, μ = 0.4)
    st = RenderSettings(; n_𝐩 = 2, n_ρ = 2, n_φ = 6, N = 4)
    masks = [compute_mask(v, scene, st; δ = 0.02, backend = RadiometricCamera.KernelAbstractions.CPU()) for v in views_t]
    targets = [render_view(k, views_t[k], vig_t, mat_t, light_t, scene, st, masks[k]) for k in 1:2]
    # the poses alone free, from a start a few mrad and mm off the truth
    only_poses = (:distortion, :focus, :aperture, :material, :vignetting, :lighting)       # held: every block but the poses
    perturbed() = CalibrationState(scene, st, [View(VgtkCore.retract(VgtkCore.LeftRigidManifold6(p), vcat(SVector(0.004, -0.003, 0.005), SVector(0.004, -0.004, 0.006))),
                                                    lens_t, focus_t, 0.0386) for p in poses], vig_t, mat_t, light_t, targets; masks)

    # a step-size stop AWAY from a stationary point: the step tolerance (xtol) so large that the first iteration's step ends the run,
    # a few mrad from the optimum — Success with IncrementSize (Core 0.11.0: the tolerance is the run's; whether it is too loose is the
    # config's question, never the record's), and only one iteration ran
    sa = perturbed()
    ra = run_lm!(sa; maxiters = 50, fixed = only_poses, xtol = 1.0e3)
    @test RC.lm_termination(ra) == (VgtkCore.ReturnCode.Success, VgtkCore.IncrementSize) && ra.iterations == 1

    # the control: the LM run to its minimum is Success; the same step-size stop RESTARTED there is the step-size rule
    # (IncrementSize), Success, and leaves the cost where the minimum put it (never higher)
    sb = perturbed()
    r0 = run_lm!(sb; maxiters = 200, fixed = only_poses, reldcost = 1.0e-14)
    @test first(RC.lm_termination(r0)) == VgtkCore.ReturnCode.Success
    c_min = cost(sb)[1]
    rb = run_lm!(sb; maxiters = 50, fixed = only_poses, xtol = 1.0e3)
    code_b, rule_b = RC.lm_termination(rb)
    @test (code_b, rule_b) == (VgtkCore.ReturnCode.Success, VgtkCore.IncrementSize)
    @test cost(sb)[1] <= c_min

    # the problem's kept NLLSsolver cache is built for the damping the solve runs (LM_DAMPING), so it is reused across solves, never
    # rebuilt per call (RC 0.9.0 review, finding 1): the cache of `LmProblem(…; damping = LM_DAMPING)`, not of Core's default
    pb, _ = RC.build_problem(perturbed(); fixed = only_poses)
    same(d) = typeof(VgtkCore.LmProblem(pb.problem.variables, pb.blocks; fixed = findall(!, pb.free), damping = d).cache) == typeof(pb.cache)
    @test same(RC.LM_DAMPING) && !same(VgtkCore.MarquardtDamping())
    # no LM iteration: no record (Default, NoRule); the iteration cap is MaxIters
    @test RC.lm_termination(nothing) == (VgtkCore.ReturnCode.Default, VgtkCore.NoRule)
    rc = run_lm!(perturbed(); maxiters = 1, fixed = only_poses)
    @test_throws MethodError run_lm!(perturbed(); maxiters = 1, unfixed = (; poses = true))            # the retired keyword: no such keyword
    @test RC.lm_termination(rc) == (VgtkCore.ReturnCode.MaxIters, VgtkCore.IterationCap) && rc.iterations == 1
end

# The RAW path's LM (Core 0.10.4's downstream run: no RC test ran an LM whose stops carry a residual_resolution > 0): photographs read
# as photosites under a Poisson–Gaussian sensor model give whitened rows with their RAW code steps (`quanta`), so `lm_abstol` > 0 reaches
# Core's exact-fit test. At the noise level the fit is no exact fit (its cost ≫ the quantisation's square); a step-size stop is Success
# wherever it fires (Core 0.11.0: a tolerance stop is converged), at the minimum and away from it.
@testitem "refinement termination: the RAW path's LM, its stops' residual resolution > 0" setup = [RCFixtures, PhotositeScene] begin
    using RadiometricCamera: cost, run_lm!, View
    using Random, StaticArrays
    import VisualGeometryToolkitCore as VgtkCore
    f = photosite_scene(; W = 32, H = 32)
    K = 4.0; σr = 3.0                                                             # e⁻/code; read noise, codes
    model = VgtkCore.sensor_fixture(; gain = K, read_noise = (σr, σr, σr, σr), black = Tuple(PhotositeScene.BLACK)).model
    κ = 0.8 / maximum(maximum, f.renders)
    rng = Xoshiro(11)
    obs = [RC._observe(RC.Photosites(), noisy_mosaic(rng, r, κ; gain = K, read_noise = σr), f.masks[k], "v$k", model) for (k, r) in enumerate(f.renders)]
    ϕw = [RC.masked_observations(o[1], RC.masked_pixels(o[2]); channel = 2) for o in obs]
    light_κ = EnvironmentLight(SphericalHarmonicMap(κ .* f.a_t, 2), f.b, f.r_SH)
    only_poses = (:distortion, :focus, :aperture, :material, :vignetting, :lighting)
    start() = CalibrationState(f.scene, f.st, [View(VgtkCore.retract(VgtkCore.LeftRigidManifold6(v.pose), vcat(SVector(0.004, -0.003, 0.005), SVector(0.004, -0.004, 0.006))),
                                                   v.lens, v.focus, v.r) for v in f.views_t], f.vig_t, f.mat_t, light_κ,
                               [Array{Float64, 3}(undef, 0, 0, 0) for _ in 1:2]; masks = [o[2] for o in obs], ϕ = first.(ϕw),
                               weights = getindex.(ϕw, 2), quanta = last.(ϕw), channel = 2)
    s = start()
    @test RC.lm_abstol(s) > 0 && RC._lm_stops(s, 10, 60.0, 1e-10, 1e-12).residual_resolution == RC.lm_abstol(s)
    # away from the minimum: a step-size stop (xtol so large the first step ends the run) is Success (Core 0.11.0)
    @test RC.lm_termination(run_lm!(start(); maxiters = 50, fixed = only_poses, xtol = 1.0e3)) == (VgtkCore.ReturnCode.Success, VgtkCore.IncrementSize)
    # to the minimum: Success; its cost far above the resolution's square (no exact fit), and the step-size stop restarted there
    # is Success too
    r = run_lm!(s; maxiters = 200, fixed = only_poses, reldcost = 1.0e-14)
    @test first(RC.lm_termination(r)) == VgtkCore.ReturnCode.Success
    @test 2cost(s)[1] > 100 * RC.lm_abstol(s)^2
    @test RC.lm_termination(run_lm!(s; maxiters = 50, fixed = only_poses, xtol = 1.0e3)) == (VgtkCore.ReturnCode.Success, VgtkCore.IncrementSize)
end
