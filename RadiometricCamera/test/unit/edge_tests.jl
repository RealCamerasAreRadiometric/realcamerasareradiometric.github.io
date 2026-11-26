# The printed edge (`Board.sdf`, `TargetState.edge`, printed_edge.jl): the normal CDF, the edge profile (half level at the ink spread δ,
# width = the print blur), the tile Jacobian through the distance map, CPU/GPU parity. The edge is never a parameter.

# The distance-map board, the two-view scene and its targets both items use.
@testsetup module EdgeScene
    using RadiometricCamera: TargetState, compute_mask, Focus, FocusSpline, _image_start, Material, TexturedTarget, render_all,
                             RenderSettings, sphere_radius, with_sdf, with_state, 𝔟
    using RadiometricCamera, StaticArrays, LinearAlgebra, ForwardDiff
    using VisualGeometryToolkitCore: exp_so3
    using VisualRadiometryToolkitCore: TensorProductVignetting, with_control_points, EnvironmentLight, SphericalHarmonicMap
    import VisualGeometryToolkitCore as VgtkCore
    const RC = RadiometricCamera
    include(joinpath(@__DIR__, "fixture_board.jl"))
    include(joinpath(@__DIR__, "fixture_camera.jl"))
    export page_e, rects_e, sdf_rects, sdf_e, board_e0, board_e, We, He, board_e, glens_e, focus_e, poses_e, camera_e, vig_e, b_e, a_e, light_e, st_e, masks_e, edge_true, mat_e, board_edge, edge_state, blank_e, targets_e, edge_s

    # ---------------------------------------------------------------------------------------------------------------
    # 2. a page of 200 mm with 16 ink squares (25 mm, pitch 50 mm) and its exact distance map
    page_e = (200.0, 200.0)
    rects_e = [(12.5 + 50i, 12.5 + 50j, 37.5 + 50i, 37.5 + 50j) for i in 0:3 for j in 0:3]      # (x0, y0, x1, y1) mm
    function sdf_rects(rects, H, W, page)
        out = zeros(H, W)
        for i in 1:H, j in 1:W
            x = (j - 0.5) / W * page[1]; y = (i - 0.5) / H * page[2]
            din = Inf; dout = Inf
            for (x0, y0, x1, y1) in rects
                if x0 <= x <= x1 && y0 <= y <= y1
                    din = min(din, x - x0, x1 - x, y - y0, y1 - y)
                else
                    dout = min(dout, hypot(max(x0 - x, 0.0, x - x1), max(y0 - y, 0.0, y - y1)))
                end
            end
            out[i, j] = isfinite(din) ? -din : dout
        end
        return out
    end
    sdf_e = sdf_rects(rects_e, 160, 160, page_e)
    board_e0 = TexturedTarget([-1.0, -1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], ones(16, 16, 3); facing = [0.0, 0.0, 1.5])
    board_e = with_sdf(board_e0, sdf_e, page_e)
    # ---------------------------------------------------------------------------------------------------------------
    # 3. a two-view scene on the distance-map board and its render
    We = He = 32
    board_e = board_e
    glens_e = _image_start(cam(VgtkCore.Lens(VgtkCore.DoubleSphere(0.25, 0.4)), 41.216, (16.32, 15.36), We, He), (10, 10)).camera
    focus_e = Focus(FocusSpline(VgtkCore.grid_nodes(3, VgtkCore.Rect(0, 0, We, He))), [1 / (1.5 + 0.1 * sin(3i)) for i in 1:9])
    poses_e = [VgtkCore.EuclideanMap(SMatrix{3, 3}(exp_so3(SVector(π, 0.0, 0.0)) * exp_so3(SVector(0.15 * cos(k), 0.12 * sin(k), 0.1 * k))), SVector{3}(SVector(0.2 * cos(k), 0.2 * sin(k), 1.2 + 0.1 * k)); check = false) for k in 1:2]
    vig_e = let v = TensorProductVignetting((4, 4), VgtkCore.Rect(0, 0, We, He); degree = 2); gr = range(-0.5, 0.5, length = 4); with_control_points(v, [1.0 - 0.3 * (gr[i]^2 + gr[j]^2) for i in 1:4, j in 1:4]) end
    camera_e = RadiometricCameraModel(glens_e, focus_e, 0.0386, vig_e)
    b_e = 𝔟(board_e); a_e = zeros(3, 9); a_e[:, 1] .= (2.0, 1.9, 1.8); a_e[:, 3] .= (0.4, 0.3, 0.2)
    light_e = EnvironmentLight(SphericalHarmonicMap(a_e, 2), b_e, sphere_radius(b_e, [p.t for p in poses_e]))
    st_e = RenderSettings(; n_𝐩 = 2, n_ρ = 2, n_φ = 6, N = 4)
    masks_e = [compute_mask(camera_e, p, board_e, st_e; δ = 0.02, backend = RadiometricCamera.KernelAbstractions.CPU()) for p in poses_e]
    edge_true = SVector(0.6, 3.0)                                         # δ 0.6 mm, σ_print 3 mm: about one pixel on the board, so the renders resolve the edge
    mat_e = Material(; roughness_ink = 0.12, roughness_paper = 0.35, refractive_index_ink = 1.65, refractive_index_paper = 1.33, diffuse_weight = 0.6)
    # the print's edge parameters (δ, σ_print) are the scene's surface parameters
    board_edge(e) = RadiometricCamera.with_state(board_e, TargetState(edge = e))
    edge_state(e, targets; kw...) = calibration_problem(board_edge(e), st_e, camera_e, poses_e, mat_e, light_e, targets; masks = masks_e, kw...)
    blank_e = [zeros(He, We, 3) for _ in 1:2]
    targets_e = let p = edge_state(edge_true, blank_e); render_all(p.f, p.u0) end
    edge_s = SVector(0.3, 3.0)
end

@testitem "printed edge (distance map)" setup = [RCFixtures, EdgeScene] begin
    using RadiometricCamera: board_surface, TargetState, build_problem_grid, cost, EDGE_NC, height_on, TexturedTarget, render_all,
                             _lm_tiles!, with_edge, with_h
    using VisualRadiometryToolkitCore: edge_albedo, has_sdf
    import VisualGeometryToolkitCore as VgtkCore
    using StaticArrays, LinearAlgebra, Statistics, ForwardDiff
    using VisualGeometryToolkitCore: exp_so3

    # (Φ is VisualGeometryToolkitCore's normcdf_as; its accuracy and its dual are Core's tests)
    @test has_sdf(board_e) && !has_sdf(board_e0) && board_e.page_mm == SVector(200.0, 200.0)
    @test board_e.sdf == sdf_e                                                                      # the distance map as given
    @test VgtkCore.convert_eltype(Float32, board_e).sdf isa Matrix{Float32} && VgtkCore.convert_eltype(Float32, board_e).page_mm == SVector{2, Float32}(200, 200)
    @test_throws ArgumentError TexturedTarget([-1.0, -1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], ones(4, 4, 3); sdf = sdf_e)
    @test VrtkCore.printed_albedo(VrtkCore.with_state(board_e0, TargetState()), SVector(0.3, 0.4)) == VrtkCore.ρ_c(board_e0, SVector(0.3, 0.4))

    # the profile across the left edge of the square (12.5…37.5 mm)²: paper at x < 12.5 (d = 12.5 − x), at mid-height y = 25 mm
    m_e = TargetState(edge = SVector(0.4, 0.3))
    @test with_h(m_e, m_e.h).edge == m_e.edge && !height_on(m_e) && VgtkCore.convert_eltype(Float32, m_e).edge == SVector{EDGE_NC, Float32}(m_e.edge)
    @test with_edge(TargetState(), m_e.edge) == m_e && TargetState{Float32}().edge == zero(SVector{EDGE_NC, Float32})
    σp = m_e.edge[2]                                                      # the edge's width: the print's blur
    bpt(x, y) = SVector(x / page_e[1], y / page_e[2])
    ρ_half = edge_albedo(VrtkCore.with_state(board_e, m_e), bpt(12.5 - 0.4, 25.0))              # d = δ → half ink
    @test all(abs.(ρ_half .- 0.5) .< 1e-6) && allequal(ρ_half)             # the same in every channel
    ρ_left = edge_albedo(VrtkCore.with_state(board_e, m_e), bpt(12.5 - 0.4 - σp, 25.0))[1]      # one σ further out: 1 − Φ(−1)
    @test abs(ρ_left - (1 - 0.15865525393145707)) < 1e-5
    ρ_top = edge_albedo(VrtkCore.with_state(board_e, m_e), bpt(25.0, 12.5 - 0.4 - σp))[2]       # the top edge the same
    @test abs(ρ_top - (1 - 0.15865525393145707)) < 1e-5
    @test all(edge_albedo(VrtkCore.with_state(board_e, m_e), bpt(25.0, 25.0)) .< 1e-4) && all(edge_albedo(VrtkCore.with_state(board_e, m_e), bpt(45.0, 45.0)) .> 1 - 1e-3)   # far inside / outside
    # the chart derivative (the pose / lens chain): ∂ρ/∂b₁ = −φ(z)/σ_print · 200 on the paper side of the left edge
    xg = 12.5 - 0.4 - 0.2
    ρd = edge_albedo(VrtkCore.with_state(board_e, m_e), SVector(ForwardDiff.Dual(xg / 200, 1.0), ForwardDiff.Dual(25.0 / 200, 0.0)))
    z = (0.4 - (12.5 - xg)) / σp
    @test ForwardDiff.partials(ρd[1], 1) ≈ -exp(-z^2 / 2) / sqrt(2π) / σp * 200 rtol = 1e-5   # (the approximation's derivative, ≈ φ to 1e-6)

    r_thin = let p = edge_state(SVector(1.6, 3.0), blank_e); render_all(p.f, p.u0) end
    @test all(isfinite, targets_e[1]) && sum(r_thin[1]) < sum(targets_e[1])        # more ink spread → darker
    @test maximum(abs, r_thin[1] .- targets_e[1]) > 0.02 * maximum(targets_e[1])

    # ---------------------------------------------------------------------------------------------------------------
    # 4. the tile problem on the distance-map board: the printed edge is no parameter of it (the material block: Θ_m and μ)
    s_e = edge_state(edge_s, targets_e)
    prob_e, idx_e = build_problem_grid(s_e.f, s_e.u0; damping = VgtkCore.LevenbergDamping(), patch = (5, 5))
    tiles_e = RC.tile_costs(prob_e)
    r_e = tiles_e[argmax([length(t.pix) for t in tiles_e])]
    vars_e = collect(RC.NLLSsolver.getvars(r_e, prob_e.variables))
    @test RC.NLLSsolver.nvars(vars_e[8]) == 9 && !any(n -> isdefined(RC, n), (:EDGE_NFREE, :n_fd, :n_edge, :EDGE_FD_H_MM))
    @test_throws MethodError build_problem_grid(s_e.f, s_e.u0; damping = VgtkCore.LevenbergDamping(), patch = (5, 5), edge_block = true)   # no such keyword
    @test_throws ArgumentError build_problem_grid(s_e.f, s_e.u0; damping = VgtkCore.LevenbergDamping(), patch = (5, 5), fixed = (:edge,))  # nothing of that name to hold
    res_e, J_e = RC.NLLSsolver.computeresjac(RC.static(1023), r_e, vars_e...)
    @test size(J_e) == (3 * length(r_e.pix), 6 + 200 + 9 + 1 + 9 + 15 + 27)   # 15 vignetting columns (the hard pin)
    @test res_e ≈ RC.NLLSsolver.computeresidual(r_e, vars_e...) atol = 1e-10
    fde(j, u) = central_difference(r_e, vars_e, j, u)
    # a NaN step (a singular solve's, here of μ) marks the block invalid: every residual of the block is NaN, and so is the problem's
    # cost, which NLLSsolver's LM rejects (a trial is accepted only when its cost is ≤ the best) instead of the refinement aborting
    @test RC.is_valid(vars_e[8]) && !RC.is_valid(RC.NLLSsolver.update(vars_e[8], unit_increment(9, 9, NaN)))
    @test all(isnan, RC.NLLSsolver.computeresidual(r_e, updated_vars(vars_e, 8, unit_increment(9, 9, NaN))...))
    let vars = copy(prob_e.variables)
        vars[idx_e.material] = RC.NLLSsolver.update(vars[idx_e.material], unit_increment(9, 9, NaN))
        @test isnan(RC.NLLSsolver.cost(vars, prob_e.problem.costs))
    end
    for j in 1:6                                                           # pose columns (AD through the distance map) vs central differences
        fd = fde(1, unit_increment(6, j, 1e-7))
        @test norm(J_e[:, j] .- fd) / norm(fd) < 5e-3
    end
    # the strongest lens-grid patch tangent columns
    pcols_e = sortperm([norm(J_e[:, 6 + i]) for i in 1:200]; rev = true)[1:4]
    for pc in pcols_e
        pq = 1 + (pc - 1) ÷ 50; pslot = pc - (pq - 1) * 50
        fd = fde(1 + pq, unit_increment(50, pslot, 1e-7))
        @test norm(J_e[:, 6 + pc] .- fd) / norm(fd) < 5e-3
    end

    # ---------------------------------------------------------------------------------------------------------------
    # 5. a refinement leaves the printed edge where it is: the material free, two iterations, δ and σ_print as given
    s_r = edge_state(edge_s, targets_e)
    c0e = cost(s_r.f, s_r.u0)[1]
    _lm_tiles!(s_r.f, s_r.u0; stops = RC._calibration_stops(; max_iter = 2), damping = VgtkCore.LevenbergDamping(), patch = (5, 5), fixed = (:poses, :grid, :focus, :aperture, :vignetting, :lighting))
    @test cost(s_r.f, s_r.u0)[1] <= c0e && board_surface(s_r.u0).edge == edge_s
end

@testitem "printed edge (distance map) (GPU)" tags = [:gpu] setup = [RCFixtures, RCGPU, EdgeScene] begin
    using RadiometricCamera: build_problem_grid
    using StaticArrays, LinearAlgebra, Statistics, ForwardDiff
    using VisualGeometryToolkitCore: exp_so3
    import CUDA

    # ---------------------------------------------------------------------------------------------------------------
    # 6. device path: CPU/GPU parity of the flux and of the pose columns through the distance map, batched == unbatched
    require_gpu()                                  # errors with the reason when the GPU the runner chose is not usable here
    if RC_GPU_TESTS
        s_g = edge_state(edge_s, targets_e; backend = CUDA.CUDABackend())
        prob_g, idx_g = build_problem_grid(s_g.f, s_g.u0; damping = VgtkCore.LevenbergDamping(), patch = (5, 5))
        tiles_g = RC.tile_costs(prob_g)
        rg = tiles_g[argmax([length(t.pix) for t in tiles_g])]
        s_c = edge_state(edge_s, targets_e)
        prob_c, _ = build_problem_grid(s_c.f, s_c.u0; damping = VgtkCore.LevenbergDamping(), patch = (5, 5))
        tiles_c = RC.tile_costs(prob_c)
        rc = tiles_c[findfirst(t -> t.k == rg.k && t.P0 == rg.P0 && length(t.pix) == length(rg.pix), tiles_c)]
        varsg = RC.NLLSsolver.getvars(rg, prob_g.variables)
        rjg, Jg = RC.NLLSsolver.computeresjac(RC.static(1023), rg, varsg...)
        rjc, Jc = RC.NLLSsolver.computeresjac(RC.static(1023), rc, RC.NLLSsolver.getvars(rc, prob_c.variables)...)
        @test maximum(abs, rjc .- rjg) < 2e-3 * max(1.0, maximum(abs, rjc))       # Float32 flux (gpu_tests.jl's tolerance)
        # a NaN step (of μ) marks the block invalid, which the device path checks too: NaN residuals, the step rejected
        vnan = ntuple(i -> i == 8 ? RC.NLLSsolver.update(varsg[8], [zeros(8); NaN]) : varsg[i], length(varsg))
        @test all(isnan, RC.NLLSsolver.computeresidual(rg, vnan...))
        for j in 1:6
            @test norm(Jc[:, j] .- Jg[:, j]) / norm(Jc[:, j]) < 2e-3
        end
        RC._register_grid_problem!(s_g.f, s_g.u0, prob_g, idx_g)
        rjg_b, Jg_b = RC.NLLSsolver.computeresjac(RC.static(1023), rg, varsg...)
        @test device_parity(rjg_b, rjg) && device_parity(Jg_b, Jg)
        RC._unregister_grid_problem!(s_g.f)
    end
end
