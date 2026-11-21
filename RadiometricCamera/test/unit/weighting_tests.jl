# The residuals' weighting (owner ruling D14; weighting.jl): a photograph read as Core's uncertain image under a sensor model with a
# gain gives whitened rows — each residual and its Jacobian row over the measured value's σ — in BOTH LM routes (the per-view blocks'
# host residual and device row kernel, the per-tile blocks') and the closed-form blocks (their reductions and their QR systems), by
# the rows' weights' TYPE (`Unweighted()` is today's arithmetic, bit for bit). The Poisson–Gaussian scene: at the truth the whitened
# residuals have unit variance (χ²/m ≈ 1), which only a variance law and a whitening that agree give.

@testitem "weighting: whitened rows in the per-view route and the closed form, one weighting" setup = [NEScene] begin
    using LinearAlgebra
    import VisualGeometryToolkitCore as VgtkCore
    import VisualRadiometryToolkitCore as VrtkCore
    NL = RC.NLLSsolver
    S = NEScene.SCENE
    state(; kw...) = RC.CalibrationState(S.sc, S.st, RC.with_vignetting(S.camera, VrtkCore.TensorProductVignetting((4, 4), VgtkCore.Rect(0, 0, NEScene.W, NEScene.W); degree = 2)),
                                         copy(S.poses), RC.with_μ(S.mat, 0.7), VrtkCore.EnvironmentLight(VrtkCore.SphericalHarmonicMap(0.5, 2), S.b, S.r_SH), S.targets; S.masks, kw...)
    su = state()
    wts = [1.0 .+ 0.5 .* sin.(1:length(p)) for p in su.ϕ]
    sw = state(; weights = wts)
    @test su.weights isa RC.Unweighted && sw.weights == wts
    # the cost's residuals: the photometric residuals times the weights; the relative RMS in the whitened norm
    @test all(k -> RC.residuals(sw)[k] == wts[k] .* RC.photometric_residuals(sw)[k], 1:2)
    @test RC.residuals(su) == RC.photometric_residuals(su)                           # Unweighted: the rows as measured, bit for bit
    @test RC.cost(sw)[1] ≈ sum(sum(abs2, wts[k] .* RC.photometric_residuals(su)[k]) for k in 1:2) / 2 rtol = 1e-14
    @test RC.data_norm2(sw) ≈ sum(sum(abs2, wts[k] .* su.ϕ[k]) for k in 1:2) rtol = 1e-14
    # the per-view route: the host residual and the row kernel's Jacobian, each row times its weight
    pu, _ = RC.build_problem(su); pw, _ = RC.build_problem(sw)
    cu = get(pu.problem.costs, RC.PerViewResidual{9}); cw = get(pw.problem.costs, RC.PerViewResidual{9})
    for k in 1:2
        vu = NL.getvars(cu[k], pu.variables); vw = NL.getvars(cw[k], pw.variables)
        ru = NL.computeresidual(cu[k], vu...); rw = NL.computeresidual(cw[k], vw...)
        @test rw == wts[k] .* ru
        rju, Ju = NL.computeresjac(RC.static(1023), cu[k], vu...); rjw, Jw = NL.computeresjac(RC.static(1023), cw[k], vw...)
        @test rjw == rw                                                              # the residual the LM sees, bitwise one
        @test Jw ≈ wts[k] .* Ju rtol = 1e-12
    end
    # the closed-form blocks: the QR systems and the reductions' cost, Jᵀr and JᵀJ, every row under its weight
    ppu = RC.all_parts(su); ppw = RC.all_parts(sw)
    W2 = reduce(vcat, wts)
    A, b = RC._weighted_system(sw, ppw, Val(:vig)); A0, b0 = RC._weighted_system(su, ppu, Val(:vig))
    @test A == W2 .* A0 && b == W2 .* b0
    cu0, gu, Hu = RC.radiometry_normal_equations(su, ppu; mu = false)
    s3 = state(; weights = [fill(3.0, length(w)) for w in wts])
    c3, g3, H3 = RC.radiometry_normal_equations(s3, RC.all_parts(s3); mu = false)
    @test isapprox(c3, 9cu0; rtol = 1e-12) && isapprox(Vector(g3), 9 .* Vector(gu); rtol = 1e-12) && isapprox(Matrix(H3), 9 .* Matrix(Hu); rtol = 1e-12)
    cw0 = RC.radiometry_normal_equations(sw, ppw; mu = false)[1]
    @test isapprox(cw0, RC.parts_cost(sw, ppw); rtol = 1e-14) && cw0 != cu0
    # the data's residual resolution (LmStops): the quantisation the data resolve, never the noise — each row's code step q over √12 in
    # the row's units (q·w whitened, q as measured); a decoded image has no code step: 0 (Core's own resolution alone)
    @test RC.lm_abstol(su) == 0 && RC.lm_abstol(sw) == 0
    q = [fill(1 / 15871, length(p)) for p in su.ϕ]
    sq = state(; quanta = q); swq = state(; weights = wts, quanta = q)
    @test RC.lm_abstol(sq)^2 ≈ sum(length, q) / 15871^2 / 12 rtol = 1e-13
    @test RC.lm_abstol(swq)^2 ≈ sum(sum(abs2, wts[k] .* q[k]) for k in 1:2) / 12 rtol = 1e-14
    @test RC._lm_stops(swq, 10, 60.0, 1e-10, 1e-12).residual_resolution == RC.lm_abstol(swq)
    @test_throws DimensionMismatch state(; quanta = [q[1][1:3], q[2]])
    @test_throws DimensionMismatch state(; weights = [wts[1]])
    # the views' device weights are formed once under the cache lock (the views' Jacobians run concurrently: `_windowed`)
    sc = state(; weights = wts)
    got = fetch.([Threads.@spawn RC._view_weights_device(sc, k) for _ in 1:16 for k in 1:2])
    ws = sc.device_cache.weights
    @test all(i -> got[i] === ws[mod1(i, 2)], eachindex(got)) && ws[1] == wts[1] && ws[2] == wts[2]
end

@testitem "weighting: whitened rows in the per-tile route" setup = [RCFixtures, GridTilesScene] begin
    using RadiometricCamera: build_problem_grid
    NL = RC.NLLSsolver
    su = RC.CalibrationState(board, st, camera_p, copy(poses_p), mat_t, light_t, targets; masks)
    wts = [1.0 .+ 0.5 .* sin.(1:length(p)) for p in su.ϕ]
    sw = RC.CalibrationState(board, st, camera_p, copy(poses_p), mat_t, light_t, targets; masks, weights = wts)
    pu, _ = build_problem_grid(su; patch = (5, 5)); pw, _ = build_problem_grid(sw; patch = (5, 5))
    for (tu, tw) in zip(RC.tile_costs(pu), RC.tile_costs(pw))
        w3 = [wts[tw.k][3 * (i - 1) + c] for i in tw.pix for c in 1:3]               # a colour state: three rows per pixel, the view's
        vu = NL.getvars(tu, pu.variables); vw = NL.getvars(tw, pw.variables)
        rju, Ju = NL.computeresjac(RC.static(1023), tu, vu...); rjw, Jw = NL.computeresjac(RC.static(1023), tw, vw...)
        @test isapprox(rjw, w3 .* rju; rtol = 1e-12) && isapprox(Jw, w3 .* Ju; rtol = 1e-12)
        @test NL.computeresidual(tw, vw...) == rjw
    end
end

@testitem "weighting: dispatch on the sensor model — Poisson–Gaussian whitened (χ²/m ≈ 1 at the truth), ReadNoise or none as measured" setup = [RCFixtures, PhotositeScene] begin
    using RadiometricCamera: cost
    using Random, Statistics
    import VisualGeometryToolkitCore as VgtkCore
    import VisualRadiometryToolkitCore as VrtkCore
    f = photosite_board(; W = 32, H = 32)
    K = 4.0; σr = 3.0                                                             # e⁻/code; read noise, codes
    model = VgtkCore.sensor_fixture(; gain = K, read_noise = (σr, σr, σr, σr), black = Tuple(PhotositeScene.BLACK)).model
    κ = 0.8 / maximum(maximum, f.renders)
    rng = Xoshiro(7)
    mosaics = [noisy_mosaic(rng, r, κ; gain = K, read_noise = σr) for r in f.renders]   # each photosite's black + Poisson + read noise, rounded
    obs = [RC._observe(RC.Photosites(), m, f.masks[k], "v$k", model, :G) for (k, m) in enumerate(mosaics)]
    @test all(o -> eltype(o[1]) <: VgtkCore.Attributed, obs)                     # the uncertain image: values with their variances
    pxs = [RC.masked_pixels(o[2]) for o in obs]
    ϕw = [RC.masked_observations(obs[k][1], pxs[k]; channel = 2) for k in 1:2]
    light_κ = EnvironmentLight(SphericalHarmonicMap(κ .* f.a_t, 2), f.b, f.r_SH)
    truth = CalibrationState(f.board, f.st, f.camera_t, copy(f.poses), f.mat_t, light_κ, [Array{Float64, 3}(undef, 0, 0, 0) for _ in 1:2];
                             masks = [o[2] for o in obs], ϕ = first.(ϕw), weights = getindex.(ϕw, 2), channel = 2)
    m = sum(length, truth.ϕ)
    χ2 = 2cost(truth)[1] / m
    # the render's own truth is the mean up to the 14-bit encoding of the mosaic (≪ σ): χ²/m is 1 within its spread √(2/m) (≈ 0.04 here)
    @test m > 500 && abs(χ2 - 1) < 0.15
    # each row's code step: one RAW code in the row's units, 1 / (white − black) of its photosite's channel (Core's linearisation)
    @test all(k -> ϕw[k][3] ≈ [1 / (mosaics[k].white - PhotositeScene.BLACK[VgtkCore.photosite_colors(mosaics[k])[v, u] + 1]) for (u, v) in pxs[k]], 1:2)
    tq = CalibrationState(f.board, f.st, f.camera_t, copy(f.poses), f.mat_t, light_κ, [Array{Float64, 3}(undef, 0, 0, 0) for _ in 1:2];
                          masks = [o[2] for o in obs], ϕ = first.(ϕw), weights = getindex.(ϕw, 2), quanta = last.(ϕw), channel = 2)
    # a fit at the noise level is no exact fit: its cost (χ²/2 ≈ m/2) is far above the resolution's square (the quantisation, Σ (q/σ)²/12),
    # so Core's stationarity postcondition decides it (Success only when stationary)
    @test 0 < RC.lm_abstol(tq)^2 < 1e-2 * cost(tq)[1]
    # the same photographs under the same black without a variance law (a ReadNoise alone: the gain dropped): plain values, the rows
    # as measured — the same values (each uncertain value's value) and mask
    rn = VgtkCore.sensor_fixture(; read_noise = (σr, σr, σr, σr), black = Tuple(PhotositeScene.BLACK)).model
    plain = [RC._observe(RC.Photosites(), mm, f.masks[k], "v$k", rn, :G) for (k, mm) in enumerate(mosaics)]
    @test all(k -> eltype(plain[k][1]) === Float64 && plain[k][1] == VgtkCore.value_of.(obs[k][1]) && plain[k][2] == obs[k][2], 1:2)
    @test RC.masked_observations(plain[1][1], pxs[1]; channel = 2)[2] isa RC.Unweighted &&
          RC.masked_observations(plain[1][1], pxs[1]; channel = 2)[3] ≈ ϕw[1][3]                 # the code steps the same, unweighted
    @test RC.masked_observations(rand(32, 32), pxs[1]; channel = 2)[3] === nothing          # a decoded image: no RAW code
    # no characterisation (the absent model): today's photosite values bit for bit (Core's linearview of the mosaic), unweighted
    v0, m0 = RC._observe(RC.Photosites(), mosaics[1], f.masks[1], "v1", VgtkCore.SensorModel(), :G)
    @test v0 == VgtkCore.linearview(mosaics[1]) && eltype(v0) === Float64 && RC.masked_observations(v0, pxs[1]; channel = 2)[2] isa RC.Unweighted
    # a decoded image: plain, whatever the model
    @test RC._observe(RC.Decoded(false), rand(32, 32, 3), f.masks[1], "d", model, :RGB)[1] isa Array{Float64, 3}
    # the retired option: refused by name, its migration the sensor model
    @test_throws "weighting follows the camera's sensor model" RC.pipeline_options(Dict("refine" => Dict("weighting" => "noise")))
end

@testitem "weighting: the device evaluates the same whitened rows" tags = [:gpu] setup = [RCGPU, NEScene] begin
    import CUDA
    using LinearAlgebra
    require_gpu()
    if RC_GPU_TESTS
        NL = RC.NLLSsolver
        su = ne_state()
        wts = [1.0 .+ 0.5 .* sin.(1:length(p)) for p in su.ϕ]
        S = NEScene.SCENE
        state(be) = RC.CalibrationState(S.sc, S.st, RC.with_vignetting(S.camera, su.camera.vignetting), copy(S.poses), su.mat, su.light, S.targets; S.masks, weights = wts, backend = be)
        sc = state(RC.KernelAbstractions.CPU()); sg = state(CUDA.CUDABackend())
        pc, _ = RC.build_problem(sc); pg, _ = RC.build_problem(sg)
        for k in 1:2
            rc = get(pc.problem.costs, RC.PerViewResidual{9})[k]; rg = get(pg.problem.costs, RC.PerViewResidual{9})[k]
            r1, J1 = NL.computeresjac(RC.static(1023), rc, NL.getvars(rc, pc.variables)...)
            r2, J2 = NL.computeresjac(RC.static(1023), rg, NL.getvars(rg, pg.variables)...)
            @test maximum(abs, r1 .- r2) < 2e-3 * max(1.0, maximum(abs, r1)) && norm(J1 - J2) < 2e-3 * norm(J1)
        end
        cc = RC.radiometry_normal_equations(sc, RC.all_parts(sc); mu = false); cg = RC.radiometry_normal_equations(sg, RC.all_parts(sg); mu = false)
        @test isapprox(cg[1], cc[1]; rtol = 1e-3) && isapprox(Vector(cg[2]), Vector(cc[2]); rtol = 1e-3)
    end
end

@testitem "weighting: a sparse design matrix's rows whitened sparse, the same products as the dense broadcast" setup = [RCFixtures] begin
    using SparseArrays: sprand, issparse, nnz
    using StableRNGs: StableRNG
    A = sprand(StableRNG(3), 2000, 50_000, 1e-4); w = rand(StableRNG(4), 2000) .+ 0.5
    Aw = RC._whiten(w, A)
    @test issparse(Aw) && nnz(Aw) == nnz(A) && Matrix(Aw[:, 1:200]) == w .* Matrix(A[:, 1:200])
    @test RC._whiten(RC.Unweighted(), A) === A
end
