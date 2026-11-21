# Central grid lens (Core's SplineLens camera) + per-tile cost blocks: quasi-interpolation accuracy,
# per-view/per-tile residual consistency, finite-difference Jacobian columns of every block
# kind (incl. the patch tangent columns), and an LM smoke recovery of perturbed directions.

# The scene, the CPU problem and the perturbed camera and poses both items use (the GPU item compares its device tiles with `tiles`).
@testsetup module GridTilesScene
    using RadiometricCamera: build_problem_grid, compute_mask, Focus, FocusSpline, grid_model, Material, TexturedTarget,
                             render_view, RenderSettings, sphere_radius, 𝔟
    using RadiometricCamera, StaticArrays, LinearAlgebra, ForwardDiff
    using VisualGeometryToolkitCore: exp_so3
    using VisualRadiometryToolkitCore: TensorProductVignetting, with_control_points, EnvironmentLight, SphericalHarmonicMap
    import VisualGeometryToolkitCore as VgtkCore
    import VisualRadiometryToolkitCore as VrtkCore
    const RC = RadiometricCamera
    include(joinpath(@__DIR__, "fixture_board.jl"))
    include(joinpath(@__DIR__, "fixture_camera.jl"))
    export tex, board, W, H, board, src_t, glens_t, tps_t, focus_t, r_t, poses, camera_t, vig_t, b, r_SH, a_t, light_t, mat_t, st, masks, targets, s, prob, idx, tiles, G_p, glens_p, camera_p, poses_p,
           per_tile_blocks

    """`per_tile_blocks(vf, t, vars, s)`: the tile `t`'s `(cost, Jᵀr, JᵀJ)` at the varflags `vf` by the per-tile path — its own
    Jacobian from its view's batch (`_tile_jacobian_at`, `_gauss_newton_blocks`): the reference of the tiles' batch (`_tile_blocks!`)."""
    function per_tile_blocks(vf, t, vars, s)
        tj = RC._tile_jacobian_at(vf, t, vars...)
        g, H = RC._gauss_newton_blocks(tj.J, tj.res, tj.T, tj.lay, s.run)
        return 0.5 * Float64(RC.NLLSsolver.sqnorm(RC._tile_residual(tj.flux, t, tj.ivs))), g, H
    end

    tex = fixture_board()
    board = TexturedTarget([-1.0, -1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], tex; facing = [0.0, 0.0, 1.5])
    W = H = 24
    src_t = cam(VgtkCore.Lens(VgtkCore.DoubleSphere(0.25, 0.4)), 30.912, (12.24, 11.52), W, H)
    glens_t = grid_model(src_t, 10, 10)               # 7 × 7 cells; patches 5 × 5 → 2 × 2 patch blocks
    tps_t = FocusSpline(VgtkCore.grid_nodes(3, VgtkCore.Rect(0, 0, W, H)))
    focus_t = Focus(tps_t, [1 / (1.5 + 0.1 * sin(3i)) for i in 1:9])
    r_t = 0.0386
    poses = [VgtkCore.EuclideanMap(SMatrix{3, 3}(exp_so3(SVector(π, 0.0, 0.0)) * exp_so3(SVector(0.15 * cos(k), 0.12 * sin(k), 0.1 * k))), SVector{3}(SVector(0.3 * cos(k), 0.3 * sin(k), 1.4 + 0.1 * k)); check = false) for k in 1:2]
    # the truth in the hard pin's gauge V(𝐜) = 1 (`gauge_center_value`; the default pin, 2026-10-30): a state built from it is the truth itself
    vig_t = RC.gauge_center_value(let v = TensorProductVignetting((4, 4), VgtkCore.Rect(0, 0, W, H); degree = 2); gr = range(-0.5, 0.5, length = 4); with_control_points(v, [1.0 - 0.4 * (gr[i]^2 + gr[j]^2) for i in 1:4, j in 1:4]) end)
    camera_t = RadiometricCameraModel(glens_t, focus_t, r_t, vig_t)
    b = 𝔟(board); r_SH = sphere_radius(b, [p.t for p in poses])
    a_t = zeros(3, 9); a_t[:, 1] .= (2.0, 1.9, 1.8); a_t[:, 3] .= (0.4, 0.3, 0.2); a_t[:, 4] .= (0.1, 0.15, 0.2)
    light_t = EnvironmentLight(SphericalHarmonicMap(a_t, 2), b, r_SH)
    mat_t = Material(; roughness_ink = 0.12, roughness_paper = 0.35, refractive_index_ink = 1.65, refractive_index_paper = 1.33, diffuse_weight = 0.4)
    st = RenderSettings(; n_𝐩 = 2, n_ρ = 2, n_φ = 6, N = 4)
    masks = [compute_mask(camera_t, p, board, st; δ = 0.02, backend = RadiometricCamera.KernelAbstractions.CPU()) for p in poses]
    targets = [render_view(k, camera_t, poses[k], mat_t, light_t, board, st, masks[k]) for k in 1:2]
    s = CalibrationState(board, st, camera_t, copy(poses), mat_t, light_t, targets; masks)
    prob, idx = build_problem_grid(s; patch = (5, 5))
    tiles = RC.tile_costs(prob)
    # LM smoke: perturb the grid (small direction rotations) + a pose nudge; joint LM must recover sharply
    G_p = copy(VgtkCore.directions(VgtkCore.model(glens_t)))   # Core's layout [gy, gx]
    for g2 in 3:7, g1 in 3:7
        b1, b2 = RC._control_tangents(G_p[g2, g1])
        G_p[g2, g1] = G_p[g2, g1] + 2e-3 * sin(g1 + 2g2) * b1 + 2e-3 * cos(2g1 - g2) * b2
    end
    glens_p = RC.VgtkCore.with_directions(glens_t, G_p)
    camera_p = RadiometricCameraModel(glens_p, focus_t, r_t, vig_t)
    poses_p = [VgtkCore.retract(VgtkCore.LeftRigidManifold6(poses[k]), vcat(SVector(0.003, -0.002, 0.002), SVector(0.002, -0.003, 0.004))) for k in 1:2]
end

@testitem "grid lens / per-tile blocks" setup = [RCFixtures, GridTilesScene] begin
    using RadiometricCamera: cost, grid_angular_error, grid_model, run_lm_grid!
    using StaticArrays, LinearAlgebra, Statistics, ForwardDiff
    using VisualGeometryToolkitCore: exp_so3
    import VisualGeometryToolkitCore as VgtkCore
    import VisualRadiometryToolkitCore as VrtkCore

    # quasi-interpolation converges with grid resolution (the wide-FOV test lens is strongly curved;
    # the fine 64 × 48 production grid on the phone's division camera reaches ~3e-7 rad)
    @test grid_angular_error(glens_t, src_t) < 2e-2
    @test grid_angular_error(grid_model(src_t, 32, 32), src_t) < 5e-4
    @test idx.n_tiles >= 2 && size(idx.patches) == (2, 2)
    @test sum(r -> length(r.pix), tiles) == sum(length, s.px)          # every masked pixel in exactly one tile
    # residual at the truth vanishes, and the stacked tile residuals match the per-view residuals
    for r in tiles
        vars = RC.NLLSsolver.getvars(r, prob.variables)
        r0 = RC.NLLSsolver.computeresidual(r, vars...)
        @test maximum(abs, r0) < 1e-12
        rj, J = RC.NLLSsolver.computeresjac(RC.static(1023), r, vars...)
        @test isapprox(rj, r0; atol = 1e-10)
        @test size(J) == (3 * length(r.pix), 6 + 4 * 50 + 9 + 1 + 9 + 15 + 27)        # 15 vignetting columns: the hard pin eliminates the centre's direction
    end
    # finite-difference check of every Jacobian column kind on the largest tile
    r = tiles[argmax([length(t.pix) for t in tiles])]
    vars = collect(RC.NLLSsolver.getvars(r, prob.variables))
    _, J = RC.NLLSsolver.computeresjac(RC.static(1023), r, vars...)
    # an active patch node: pick the strongest patch tangent column
    pcol = 6 + argmax([norm(J[:, 6 + i]) for i in 1:200])              # patch blocks: 2 · 25 = 50 columns each
    pq = 1 + (pcol - 7) ÷ 50                                           # which of the 4 patch blocks
    pslot = pcol - 6 - (pq - 1) * 50
    for (col, j, u) in ((2, 1, unit_increment(6, 2, 1e-7)),                        # pose
                        (pcol, 1 + pq, unit_increment(50, pslot, 1e-7)),           # grid patch tangent
                        (6 + 200 + 2, 6, unit_increment(9, 2, 1e-7)),              # focus node 2
                        (6 + 200 + 9 + 1 + 2, 8, unit_increment(9, 2, 1e-7)),     # material 2
                        (6 + 200 + 9 + 1 + 9, 8, unit_increment(9, 9, 1e-7)),   # μ
                        (6 + 200 + 9 + 1 + 9 + 2, 9, unit_increment(15, 2, 1e-7)),  # vignetting (15 free directions: the hard pin's) (null-space coord 2)
                        (6 + 200 + 9 + 1 + 9 + 15 + 4, 10, unit_increment(27, 4, 1e-7)))  # lighting a[1, 2]
        fd = (RC.NLLSsolver.computeresidual(r, updated_vars(vars, j, u)...) - RC.NLLSsolver.computeresidual(r, updated_vars(vars, j, -u)...)) / (2 * norm(u))
        nfd = norm(fd)
        nfd > 0 || continue
        @test norm(J[:, col] - fd) / nfd < 5e-3
    end
    # the aperture column: r is a dual seed through the fixed unit-disc rule r² w̃_l L_c(·, r â_l) — the exact derivative of
    # the discrete aperture quadrature like every other column
    let u = unit_increment(1, 1, 1e-7)
        fd = (RC.NLLSsolver.computeresidual(r, updated_vars(vars, 7, u)...) - RC.NLLSsolver.computeresidual(r, updated_vars(vars, 7, -u)...)) / 2e-7
        col = J[:, 6 + 200 + 9 + 1]
        @test norm(col - fd) / norm(fd) < 5e-3
    end
    s2 = CalibrationState(board, st, camera_p, copy(poses_p), mat_t, light_t, targets; masks)   # a copy: the LM writes its poses back into
                                                         # the state's vector, and the GPU recovery below must start from the perturbed poses_p
    c0 = cost(s2)[1]
    res = run_lm_grid!(s2; patch = (5, 5), maxiters = 25)
    @test cost(s2)[1] < 3e-3 * c0
    # recovery vs the grid truth (not the DS source), modulo the common rotation the gauge holds at the start's (`rotation_gauge_coordinates`):
    # the rays compared after Core's Kabsch rotation between the two lenses' rays
    let probe = [SVector(x, y) for x in 2.0:4.0:22.0 for y in 2.0:4.0:22.0]
        a = [normalize(RC.chief_ray(glens_t, p)) for p in probe]; b = [normalize(RC.chief_ray(s2.camera.geometric, p)) for p in probe]
        R = SMatrix{3, 3, Float64}(VgtkCore.kabsch([a[i] => b[i] for i in eachindex(a)]))
        @test maximum(VgtkCore.angle_between(R' * b[i], a[i]) for i in eachindex(a)) < 5e-4
    end
    # trace entries carry the standard fields
    trc = NamedTuple[]
    run_lm_grid!(s2; patch = (5, 5), maxiters = 2, trace = trc, trace_meta = (; stage = 1, factor = 1, ϕ² = sum(sum(abs2, p) for p in s2.ϕ)))
    @test !isempty(trc) && all(t.phase == :lm for t in trc)
end

@testitem "grid lens / present nodes, absent nodes and smoothness" setup = [RCFixtures, GridTilesScene] begin
    using RadiometricCamera: build_problem_grid, render_view
    using StaticArrays, LinearAlgebra
    import VisualGeometryToolkitCore as VgtkCore
    import VisualRadiometryToolkitCore as VrtkCore
    # the nodes under the model's imaged region: Core's footprint rule (the double sphere source: the whole plane, every node present)
    held_t = RC.grid_present(glens_t)
    @test held_t == VgtkCore.node_mask(VgtkCore.model(glens_t), VgtkCore.imaged_region(glens_t)) && all(held_t)
    @test RC.interior_nodes(held_t) == trues(10, 10)
    all10 = trues(10, 10)
    @test RC.grid_smoothness(RC.NoSmoothness(), glens_t, 1.0, all10) === nothing && RC.grid_smoothness(RC.PSplineSmoothness(3, 0.0), glens_t, 1.0, all10) === nothing
    @test_throws ArgumentError RC.grid_smoothness(RC.PSplineSmoothness(3, 1.0), glens_t, 1.0, all10; weight = -1.0)   # a negative weight
    # the operators: the analytic bending energy of the lattice (Core's bending_stencils, BSplineKit's Galerkin form in pixels; m̄ the
    # mean positive eigenvalue of DᵀD, Core's)
    pr2 = RC.grid_smoothness(RC.BendingSmoothness(0.5), glens_t, 7.0, all10)
    h = VgtkCore.lattice_spacing(VgtkCore.model(glens_t))
    Gb = VgtkCore.bending_gram(VgtkCore.cubic_lattice_basis(10; spacing = h[1]), VgtkCore.cubic_lattice_basis(10; spacing = h[2]))
    X = [sin(0.3i) * cos(0.2j) + 0.01 * i * j for i in 1:10, j in 1:10]
    @test sum(row -> sum(c * X[n] for (n, c) in row)^2, pr2.rows) ≈ dot(vec(X), Gb * vec(X)) rtol = 1e-10   # every cell interior
    λ = eigvals(Symmetric(Gb)); λp = λ[λ .> 1e-10 * maximum(λ)]
    @test pr2.m̄ ≈ sum(λp) / length(λp) rtol = 1e-8
    @test pr2.w ≈ sqrt(0.5 * 7.0 / 100 / pr2.m̄)
    # present nodes that are not the whole lattice ([g_x, g_y]): the rows lie on them, N their count
    free = trues(10, 10); free[1:5, 1:5] .= false; free[7, 8] = false
    held = permutedims(free)                                          # the mask laid [gy, gx] like the model's controls
    @test RC.interior_nodes(held) == free
    pr2 = RC.grid_smoothness(RC.BendingSmoothness(0.5), glens_t, 7.0, free)
    prp = RC.grid_smoothness(RC.PSplineSmoothness(3, 0.5), glens_t, 7.0, free)
    @test all(row -> all(t -> free[first(t)], row), pr2.rows) && all(row -> all(t -> free[first(t)], row), prp.rows)
    # the lens holding them: NaN in the absent slots. No residual reads an absent node: a pixel whose window holds one is refused,
    # so the state's pixels are those whose every sensor sample reads present nodes only
    lens_m = RC.VgtkCore.with_present(glens_t, held)
    D(l) = VgtkCore.directions(VgtkCore.model(l))
    @test VgtkCore.present_nodes(VgtkCore.model(lens_m)) == held && all(v -> all(isnan, v), D(lens_m)[.!held])
    camera_m = RadiometricCameraModel(lens_m, focus_t, r_t, vig_t)
    s_all = CalibrationState(board, st, camera_m, copy(poses), mat_t, light_t, targets; masks)
    @test_throws ArgumentError build_problem_grid(s_all; patch = (5, 5))
    reads_held(u, v) = all(p -> all(isfinite, VgtkCore.backproject(lens_m, first(p))), RC.sensor_samples(st.sensor, u, v))
    masks_m = [BitMatrix([mk[v, u] && reads_held(u, v) for v in axes(mk, 1), u in axes(mk, 2)]) for mk in masks]
    @test all(k -> 0 < count(masks_m[k]) < count(masks[k]), 1:2)
    targets_m = [render_view(k, camera_t, poses[k], mat_t, light_t, board, st, masks_m[k]) for k in 1:2]
    s = CalibrationState(board, st, camera_m, copy(poses), mat_t, light_t, targets_m; masks = masks_m)
    # in a problem: zero cost at the start, the rows' Jacobian against central differences through the patch blocks' retraction,
    # no column for an absent node; a patch block with no present node is not a variable of the solve
    for pr in (pr2, prp)
        prob_p, idx_p = build_problem_grid(s; patch = (5, 5), smoothness = pr)
        @test idx_p.dependent_patches == [idx_p.patches[1, 1]]
        costs = [c for T in keys(prob_p.problem.costs) if T <: RC.GridSmoothnessResidual for c in get(prob_p.problem.costs, T)]
        @test sum(c -> length(c.terms), costs) == length(pr.rows)
        @test RC.grid_smoothness_cost(pr, glens_t) == 0.0
        for c in costs
            vars = collect(RC.NLLSsolver.getvars(c, prob_p.variables))
            @test maximum(abs, RC.NLLSsolver.computeresidual(c, vars...)) < 1e-14
            # a moved start: the residual measures d̂ − d̂₀
            moved = [RC.NLLSsolver.update(v, 1e-3 .* sin.(1:50)) for v in vars]
            flags = RC.static(2^length(vars) - 1)
            res, J = RC.NLLSsolver.computeresjac(flags, c, moved...)
            @test res ≈ RC.NLLSsolver.computeresidual(c, moved...)
            @test size(J) == (3 * length(c.terms), 50 * length(vars))
            for q in eachindex(moved), k in 1:50
                u = zeros(50); u[k] = 1e-6
                at(δ) = [i == q ? RC.NLLSsolver.update(moved[i], δ) : moved[i] for i in eachindex(moved)]
                fd = (RC.NLLSsolver.computeresidual(c, at(u)...) - RC.NLLSsolver.computeresidual(c, at(-u)...)) / 2e-6
                n = (k + 1) ÷ 2                                    # the patch's local node of coordinate k
                P = Tuple(findfirst(==(c.idx[q]), idx_p.patches)) .- 1
                gx, gy = 5 * P[1] + mod1(n, 5), 5 * P[2] + fld1(n, 5)
                col = J[:, 50 * (q - 1) + k]
                if free[gx, gy] && !RC._rotation_held(first(RC.tile_costs(prob_p)).held_rotation_coordinates, gx, gy, isodd(k) ? 1 : 2)   # a held coordinate: no column
                    @test norm(col - fd) <= 1e-6 * max(norm(fd), 1e-3)
                else
                    @test iszero(col)
                end
            end
        end
    end
    # the absent nodes have no tile column either, and every residual is finite (none reads a NaN slot)
    prob_f, idx_f = build_problem_grid(s; patch = (5, 5))
    for r in RC.tile_costs(prob_f)
        res, J = RC.NLLSsolver.computeresjac(RC.static(1023), r, RC.NLLSsolver.getvars(r, prob_f.variables)...)
        @test all(isfinite, res) && all(isfinite, J)
        r.P0 == (0, 0) && @test iszero(J[:, 7:56])                    # patch (0, 0): every node absent
    end
    # the solved lens: its present nodes moved, its absent ones still absent (no value: NaN)
    moved = [v isa RC._Patch ? RC.NLLSsolver.update(v, 1e-3 .* cos.(1:50)) : v for v in prob_f.variables]
    lens = RC.extract_variables_grid(s, moved, idx_f)[1].geometric     # (camera, poses, …): the camera's lens
    l0 = s.camera.geometric
    @test VgtkCore.present_nodes(VgtkCore.model(lens)) == held
    @test D(lens)[held] != D(l0)[held] && all(v -> all(isnan, v), D(lens)[.!held])
end

@testitem "grid lens / per-tile blocks (GPU)" tags = [:gpu] setup = [RCFixtures, RCGPU, GridTilesScene] begin
    using RadiometricCamera: build_problem_grid, cost, grid_angular_error, run_lm_grid!
    using StaticArrays, LinearAlgebra, Statistics, ForwardDiff
    using VisualGeometryToolkitCore: exp_so3
    import VisualGeometryToolkitCore as VgtkCore
    import VisualRadiometryToolkitCore as VrtkCore
    import CUDA

    # --- device (GPU) per-tile path: bitwise residual, column parity, LM smoke ---
    require_gpu()                                  # errors with the reason when the GPU the runner chose is not usable here
    if RC_GPU_TESTS
        s_g = CalibrationState(board, st, camera_t, copy(poses), mat_t, light_t, targets; masks, backend = CUDA.CUDABackend())
        prob_g, idx_g = build_problem_grid(s_g; patch = (5, 5))
        tiles_g = RC.tile_costs(prob_g)
        rg = tiles_g[argmax([length(t.pix) for t in tiles_g])]
        rc = tiles[findfirst(t -> t.k == rg.k && t.P0 == rg.P0 && length(t.pix) == length(rg.pix), tiles)]
        @test rc.pix == rg.pix
        varsg = RC.NLLSsolver.getvars(rg, prob_g.variables)
        r0g = RC.NLLSsolver.computeresidual(rg, varsg...)
        rjg, Jg = RC.NLLSsolver.computeresjac(RC.static(1023), rg, varsg...)
        @test rjg == r0g                                       # bitwise: both from the flux kernel
        @test maximum(abs, r0g) < 2e-3                          # Float32 render vs the Float64-rendered targets
        _, Jc2 = RC.NLLSsolver.computeresjac(RC.static(1023), rc, RC.NLLSsolver.getvars(rc, prob.variables)...)
        @test size(Jg) == size(Jc2)
        colnorms = [norm(Jc2[:, col]) for col in axes(Jc2, 2)]
        for col in axes(Jc2, 2)
            nc = colnorms[col]
            nc < 1e-3 * maximum(colnorms) && continue
            @test norm(Jc2[:, col] .- Jg[:, col]) / nc < 2e-2
        end
        # --- batched per-view memoization: parity with the unbatched path (bitwise; `device_parity`), and cache hits ---
        # (r0g / rjg / Jg above were computed UNBATCHED; registering the problem enables batching)
        RC._register_grid_problem!(s_g, prob_g, idx_g)
        r0g_b = RC.NLLSsolver.computeresidual(rg, varsg...)
        rjg_b, Jg_b = RC.NLLSsolver.computeresjac(RC.static(1023), rg, varsg...)
        @test r0g_b == r0g                                     # bitwise: same kernels, per-pixel independent
        @test rjg_b == rjg
        @test device_parity(Jg_b, Jg)
        # --- partial varflags (a FIXED block; here the aperture, bit 7) are served by the batched path too (2026-10-24: a run with a fixed
        #     aperture formerly dropped every tile onto the unbatched per-tile path): exact parity with the unbatched partial Jacobian, and
        #     the kept columns are the full Jacobian's without the r column
        vf_p = RC.static(1023 - 64)
        rj_p, Jg_p = RC.NLLSsolver.computeresjac(vf_p, rg, varsg...)             # batched (registered)
        RC._unregister_grid_problem!(s_g)
        rj_pu, Jg_pu = RC.NLLSsolver.computeresjac(vf_p, rg, varsg...)           # unbatched
        RC._register_grid_problem!(s_g, prob_g, idx_g)
        @test rj_p == rjg
        @test rj_p == rj_pu
        @test device_parity(Jg_p, Jg_pu)
        lay_full = RC._tile_layout(RC.static(1023), length(VgtkCore.model(varsg[6]).nodes), length(s_g.settings.sensor), 25, VgtkCore.tangent_size(varsg[9]),
                                   Int(VrtkCore.n_control_points(VgtkCore.model(varsg[9]))), Int(sh_count(max_degree(varsg[10].light))))
        @test size(Jg_p, 2) == lay_full.ncol - 1
        @test device_parity(Jg_p, Jg[:, setdiff(1:lay_full.ncol, lay_full.col_r)])
        # (the 24 × 24 fixture has one tile per view, so the cache-hit check re-evaluates the same tile)
        RC.NLLSsolver.computeresidual(rg, varsg...)            # primes the view cache for this variable state
        calls0 = s_g.run.device_calls[:flux]
        @test RC.NLLSsolver.computeresidual(rg, varsg...) == r0g
        @test s_g.run.device_calls[:flux] == calls0                 # same trial state: served from the cache
        r2 = tiles_g[findfirst(t -> t !== rg, tiles_g)]        # cross-tile parity (another view)
        vars2 = RC.NLLSsolver.getvars(r2, prob_g.variables)
        r0u2 = let; RC._unregister_grid_problem!(s_g); RC.NLLSsolver.computeresidual(r2, vars2...); end
        RC._register_grid_problem!(s_g, prob_g, idx_g)
        r0b2 = RC.NLLSsolver.computeresidual(r2, vars2...)
        @test r0b2 == r0u2
        # a *different* variable state must invalidate the cache (fresh objects → new key)
        vars2n = ntuple(i -> i == 1 ? RC.NLLSsolver.update(vars2[1], zeros(6)) : vars2[i], 10)
        RC.NLLSsolver.computeresidual(r2, vars2n...)
        @test s_g.run.device_calls[:flux] > calls0
        RC._unregister_grid_problem!(s_g)
        # a view's tiles in one batch (`_tile_blocks!`): the residuals' cost bit for bit, the blocks the per-tile path's to the products'
        # summation order (two strided batched gemms over the zero-padded rows); no tile forms its own Jacobian
        let ts = RC.tile_costs(prob_g), vars_of(t) = RC.NLLSsolver.getvars(t, prob_g.variables)
            per_tile, batched, jacs = RC._with_registered(s_g, prob_g, idx_g; concurrent = false) do
                ref = [per_tile_blocks(RC.static(1023), t, vars_of(t), s_g) for t in ts]
                jac0 = s_g.run.device_calls[:jac]
                ref, [RC.NLLSsolver.computecostgradhess(RC.static(1023), t, vars_of(t)...) for t in ts], s_g.run.device_calls[:jac] - jac0
            end
            @test jacs == 0
            for (x, y) in zip(per_tile, batched)
                @test x[1] == y[1]
                @test norm(y[2] - x[2]) <= 1e-12 * norm(x[2]) && norm(y[3] - x[3]) <= 1e-12 * norm(x[3])
            end
        end
        s2g = CalibrationState(board, st, camera_p, copy(poses_p), mat_t, light_t, targets; masks, backend = CUDA.CUDABackend())
        c0g = cost(s2g)[1]
        run_lm_grid!(s2g; patch = (5, 5), maxiters = 15)
        @test cost(s2g)[1] < 0.05 * c0g
        @test grid_angular_error(s2g.camera.geometric, glens_t) < 2e-3
    end
end

# A view's tiles' Gauss–Newton blocks in one batch (`_tile_blocks!`, decision 6): every tile's (cost, Jᵀr, JᵀJ) at the varflags the
# problem gives it, the batch's and the per-tile path's bit for bit on the CPU backend — with a dependent patch (its neighbours' tiles a
# batch of other varflags) and with a held block (the aperture) — and the batch formed once per view (no per-tile Jacobian).
@testitem "grid lens / a view's tiles in one batch" setup = [RCFixtures, GridTilesScene] begin
    using RadiometricCamera: build_problem_grid, render_view
    import VisualGeometryToolkitCore as VgtkCore
    # a 15 × 15 lattice (3 × 3 patch blocks, 2 × 2 tile windows) whose patch (0, 0) holds no present node: a dependent patch, so the
    # tiles of the window at (0, 0) have other varflags than their view's other tiles
    glens15 = RC.grid_model(src_t, 15, 15)
    free = trues(15, 15); free[1:5, 1:5] .= false
    lens_m = RC.VgtkCore.with_present(glens15, permutedims(free))
    reads_held(u, v) = all(p -> all(isfinite, VgtkCore.backproject(lens_m, first(p))), RC.sensor_samples(st.sensor, u, v))
    masks_m = [BitMatrix([mk[v, u] && reads_held(u, v) for v in axes(mk, 1), u in axes(mk, 2)]) for mk in masks]
    targets_m = [render_view(k, camera_t, poses[k], mat_t, light_t, board, st, masks_m[k]) for k in 1:2]
    camera_m = RadiometricCameraModel(lens_m, focus_t, r_t, vig_t)
    flags(free_vars, t) = sum(j -> free_vars[t.idx[j]] ? 1 << (j - 1) : 0, 1:10)     # NLLSsolver's `computevarflags` of the tile
    for (cam_, msk, tgt, fixed) in ((camera_t, masks, targets, ()), (camera_m, masks_m, targets_m, ()), (camera_t, masks, targets, (:aperture,)))
        s_b = CalibrationState(board, st, cam_, copy(poses), mat_t, light_t, tgt; masks = msk)
        prob_b, idx_b = build_problem_grid(s_b; patch = (5, 5), fixed)
        ts = RC.tile_costs(prob_b)
        vfs = [flags(idx_b.free, t) for t in ts]
        vars_of(t) = RC.NLLSsolver.getvars(t, prob_b.variables)
        blocks() = [RC.NLLSsolver.computecostgradhess(RC.static(vf), t, vars_of(t)...) for (t, vf) in zip(ts, vfs)]
        per_tile, batched, again, jacs = RC._with_registered(s_b, prob_b, idx_b; concurrent = false) do
            ref = [per_tile_blocks(RC.static(vf), t, vars_of(t), s_b) for (t, vf) in zip(ts, vfs)]
            jac0 = s_b.run.device_calls[:jac]
            ref, blocks(), blocks(), s_b.run.device_calls[:jac] - jac0   # the tiles asked again at the trial: their view's batch again
        end
        @test jacs == 0                                              # the batches: no tile formed its own Jacobian
        @test all(((x, y),) -> x[1] == y[1] && x[2] == y[2] && x[3] == y[3], zip(per_tile, batched))
        @test all(((x, y),) -> x[1] == y[1] && x[2] == y[2] && x[3] == y[3], zip(per_tile, again))
        cam_ === camera_m && @test all(k -> length(unique(vfs[[t.k == k for t in ts]])) > 1, 1:2)   # tiles of several varflags in a view
        :aperture in fixed && @test all(vf -> vf & 64 == 0, vfs)
    end
end
