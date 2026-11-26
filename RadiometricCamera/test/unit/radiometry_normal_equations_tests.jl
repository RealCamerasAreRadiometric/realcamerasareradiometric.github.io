# The closed-form blocks on the device (2026-11-13: the normal equations on the GPU, KernelAbstractions): the reduction kernel's
# normal equations against an independent host assembly from the design-row kernel, the gradient against central differences of the
# cost kernel, and the same kernels on KA.CPU() and on the GPU (the :gpu item). The reduction against the hand-written host normal
# equations it replaced agreed to ≤ 2.3e-15 relative on this scene (cost, Jᵀr, JᵀJ; colour and single-channel states, RC 0.8.0
# report); the joint solve's recovery of the truth is the "calibration" item's.

@testsetup module NEScene
    using RadiometricCamera: compute_mask, Focus, FocusSpline, Material, TexturedTarget, render_view, RenderSettings,
                             sphere_radius, with_μ, with_vignetting, 𝔟
    using RadiometricCamera, StaticArrays, LinearAlgebra
    import VisualGeometryToolkitCore as VgtkCore
    using VisualGeometryToolkitCore: exp_so3
    using VisualRadiometryToolkitCore: TensorProductVignetting, with_control_points, EnvironmentLight, SphericalHarmonicMap, coefficients
    const RC = RadiometricCamera
    include(joinpath(@__DIR__, "fixture_board.jl"))
    include(joinpath(@__DIR__, "fixture_camera.jl"))
    using VisualRadiometryToolkitCore: control_points
    export RC, ne_state, ne_host, relerr, coefficients, control_points
    const W = 24
    const SCENE = let tex = fixture_board()
        board = TexturedTarget([-1.0, -1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], tex; facing = [0.0, 0.0, 1.5])
        lens = cam(VgtkCore.Lens(VgtkCore.DoubleSphere(0.25, 0.4)), 30.912, (12.24, 11.52), W, W)
        focus = Focus(FocusSpline(VgtkCore.grid_nodes(3, VgtkCore.Rect(0, 0, W, W))), [1 / (1.5 + 0.1 * sin(3i)) for i in 1:9])
        poses = [VgtkCore.EuclideanMap(SMatrix{3, 3}(exp_so3(SVector(π, 0.0, 0.0)) * exp_so3(SVector(0.15 * cos(k), 0.12 * sin(k), 0.1 * k))),
                                       SVector{3}(SVector(0.3 * cos(k), 0.3 * sin(k), 1.4 + 0.1 * k)); check = false) for k in 1:2]
        vig = RC.gauge_center_value(let v = TensorProductVignetting((4, 4), VgtkCore.Rect(0, 0, W, W); degree = 2); gr = range(-0.5, 0.5, length = 4)
            with_control_points(v, [1.0 - 0.4 * (gr[i]^2 + gr[j]^2) for i in 1:4, j in 1:4]) end)
        camera = RadiometricCameraModel(lens, focus, 0.0386, vig)
        b = 𝔟(board); r_SH = sphere_radius(b, [p.t for p in poses])
        a = zeros(3, 9); a[:, 1] .= (2.0, 1.9, 1.8); a[:, 3] .= (0.4, 0.3, 0.2); a[:, 4] .= (0.1, 0.15, 0.2)
        mat = Material(; roughness_ink = 0.12, roughness_paper = 0.35, refractive_index_ink = 1.65, refractive_index_paper = 1.33, diffuse_weight = 0.4)
        st = RenderSettings(; n_𝐩 = 2, n_ρ = 2, n_φ = 6, N = 4)
        sc = board
        masks = [compute_mask(camera, p, sc, st; δ = 0.02, backend = RadiometricCamera.KernelAbstractions.CPU()) for p in poses]   # the scene's masks, made once on the CPU
        targets = [render_view(k, camera, poses[k], mat, EnvironmentLight(SphericalHarmonicMap(a, 2), b, r_SH), sc, st, masks[k]) for k in 1:2]
        (; sc, st, camera, poses, b, r_SH, mat, masks, targets)
    end
    "A start away from the truth (flat V, grey light, μ 0.7): every residual and every Jacobian column nonzero."
    ne_state(; backend = VgtkCore.KA.CPU(), channel = 0) = (S = SCENE;
        CalibrationState(S.sc, S.st, with_vignetting(S.camera, TensorProductVignetting((4, 4), VgtkCore.Rect(0, 0, W, W); degree = 2)), copy(S.poses),
                         with_μ(S.mat, 0.7), EnvironmentLight(SphericalHarmonicMap(0.5, 2), S.b, S.r_SH), S.targets; S.masks, backend, channel))
    """The normal equations of the lighting and vignetting blocks (μ held) assembled on the host from the design-row kernel's rows:
    J = [A_v Z | the lighting columns per measured channel], r = Φ − ϕ with Φ the lighting rows times the coefficients."""
    function ne_host(s, parts)
        chs = RC.residual_channels(s); nc = length(chs)
        Al = RC._design_matrix(s, parts, Val(:light)); Av = RC._design_matrix(s, parts, Val(:vig)); ϕ = Array(parts.ϕ)
        Z = RC.vignetting_basis(s); a = Matrix(coefficients(s.light)); J9 = size(Al, 2)
        Jl = zeros(size(Al, 1), nc * J9); Φ = zeros(size(Al, 1))
        for row in axes(Al, 1)
            jc = (row - 1) % nc + 1
            Φ[row] = dot(Al[row, :], a[chs[jc], :])
            for j in 1:J9; Jl[row, (j - 1) * nc + jc] = Al[row, j]; end
        end
        Jm = hcat(Av * Z, Jl); r = Φ - ϕ
        return 0.5 * dot(r, r), Jm' * r, Jm' * Jm
    end
    relerr(x, y) = norm(x - y) / norm(y)
end

@testitem "closed-form blocks: device normal equations" setup = [NEScene] begin
    using RadiometricCamera: all_parts, chart, _update_diffuse_weight!
    using StaticArrays, LinearAlgebra
    for channel in (0, 2)                                   # the colour state and a single-channel one (its J coefficients)
        s = ne_state(; channel); parts = all_parts(s)
        c, g, H = RC.radiometry_normal_equations(s, parts; mu = false)
        ch, gh, Hh = ne_host(s, parts)
        @test relerr(c, ch) < 1e-12 && relerr(Vector(g), gh) < 1e-12 && relerr(Matrix(H), Hh) < 1e-12
        @test c ≈ RC.parts_cost(s, parts) rtol = 1e-14       # the cost kernel's cost at the chart origin
        # the full layout (μ on its chart): the gradient is the cost kernel's central differences
        c, g, H = RC.radiometry_normal_equations(s, parts)
        lm = RC._radiometry_lm(s, parts, RC._radiometry_layout(size(RC.vignetting_basis(s), 2), 9, length(RC.residual_channels(s)); light = true, mu = true, chart = true, s.mu_bounds))
        N = length(g); h = 1e-6
        e(q) = SVector{N, Float64}(ntuple(i -> i == q ? h : 0.0, Val(N)))
        fd = [(RC.VgtkCore.residual_cost!(nothing, lm, e(q)) - RC.VgtkCore.residual_cost!(nothing, lm, -e(q))) / 2h for q in 1:N]
        @test norm(fd - g) / norm(g) < 1e-6
        @test issymmetric(Matrix(H)) && isposdef(Symmetric(Matrix(H)))
    end
    # μ alone (the additive chart): _update_diffuse_weight!'s one Gauss–Newton step is exact (Φ is affine in μ): the step's gradient vanishes
    s = ne_state(); parts = all_parts(s)
    _update_diffuse_weight!(s, parts)
    _, g, H = RC.radiometry_normal_equations(s, parts; light = false, vignetting = false, chart = false)
    @test abs(g[1]) < 1e-10 * sqrt(H[1, 1]) * sqrt(2 * RC.parts_cost(s, parts))
end

# The launch and the reduction through Core's one call (reduce_cost / reduce_normal_equations with the row functor, Core 0.9.7) are
# the launch of the row kernel and the reduction called apart, bit for bit.
@testitem "closed-form blocks: Core's launch-and-reduce is the launch then the reduction, bit for bit" setup = [NEScene] begin
    using RadiometricCamera: all_parts, chart
    using StaticArrays
    import VisualGeometryToolkitCore as VgtkCore
    s = ne_state(); parts = all_parts(s)
    lay = RC._radiometry_layout(size(RC.vignetting_basis(s), 2), 9, length(RC.residual_channels(s)); light = true, mu = true, chart = true, s.mu_bounds)
    lm = RC._radiometry_lm(s, parts, lay)
    N = lay.N
    x = SVector{N, Float64}(ntuple(i -> 1e-3 * sin(i), Val(N)))
    args = (lm.parts, lm.ϕ, lm.w, lm.chs, lm.vig, lm.a, lm.μ, lm.Z, lm.lay, x)
    RC._launch_row_job!(RC._radiometry_cost_row!, lm.partials, size(lm.partials, 2), args...)
    c0 = VgtkCore.reduce_cost(lm.partials)
    RC._launch_row_job!(RC._radiometry_normal_row!, lm.partials, size(lm.partials, 2), args...)
    n0 = VgtkCore.reduce_normal_equations(lm.partials, Val(N))
    @test VgtkCore.residual_cost!(nothing, lm, x) === c0
    n1 = VgtkCore.residual_normal_equations!(nothing, nothing, lm, x)
    @test all(map((a, b) -> a == b, n1, n0))
end

@testitem "closed-form blocks: device normal equations (GPU)" tags = [:gpu] setup = [RCGPU, NEScene] begin
    using RadiometricCamera: all_parts, RadiometryBlocks, RadiometryBlockDescent, _lm_radiometry!
    using StaticArrays, LinearAlgebra
    using CUDA: CUDABackend
    require_gpu()
    be = CUDABackend(; always_inline = true)
    for channel in (0, 2)
        sc = ne_state(; channel); sg = ne_state(; channel, backend = be)
        pc = all_parts(sc); pg = all_parts(sg)
        @test relerr(reinterpret(Float64, Array(pg.parts)), reinterpret(Float64, collect(pc.parts))) < 1e-12   # the parts: Float64 on every backend
        nc_ = RC.radiometry_normal_equations(sc, pc); ng = RC.radiometry_normal_equations(sg, pg)
        @test relerr(ng[1], nc_[1]) < 1e-10 && relerr(Vector(ng[2]), Vector(nc_[2])) < 1e-10 && relerr(Matrix(ng[3]), Matrix(nc_[3])) < 1e-10
        # the block sweep and the joint solve on the device land where the CPU's do
        for (s_, p_) in ((sc, pc), (sg, pg)); RC.VgtkCore.solve(RC.VgtkCore.EstimationProblem(RadiometryBlocks, (s_, p_)), RadiometryBlockDescent(; sweeps = 2)); end
        @test relerr(Matrix(coefficients(sg.light)), Matrix(coefficients(sc.light))) < 1e-10 && abs(sg.mat.diffuse_weight - sc.mat.diffuse_weight) < 1e-10
        rc = _lm_radiometry!(sc, pc; damping = VgtkCore.LevenbergDamping(), stops = VgtkCore.LmStops(; max_iter = 80)); rg = _lm_radiometry!(sg, pg; damping = VgtkCore.LevenbergDamping(), stops = VgtkCore.LmStops(; max_iter = 80))
        @test rc[1] == rg[1] == RC.VgtkCore.ReturnCode.Success
        @test relerr(vec(Matrix(control_points(sg.camera.vignetting))), vec(Matrix(control_points(sc.camera.vignetting)))) < 1e-8
    end
end
