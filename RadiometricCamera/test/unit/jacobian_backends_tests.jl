# The ONE Jacobian implementation (the pixel functors of device/jacobian_kernels.jl on the state's backend: Float64 on the CPU, Float32 on a device)
# against checks that share none of its chain rules: the board warping's height columns against ForwardDiff of the flux through
# Core's posed height field (the nodal heights seeded as duals into `Φ_pixel`, the ray met by Core's MappedSurface root, its implicit
# derivative and the projected thin-plate derivative of `with_parameters`), and against central differences of the tile residual
# along the material block's free height directions; the focus columns of a frozen node and of a direction-chart focus (the device
# path ignored both before 2026-11-11); and the two backends against each other, to the device's Float32 rounding.
#
# The scene is the printed-edge board of edge_tests.jl (a distance map: the paper's Siemens boards' albedo, whose sharp edges make
# the height columns sensitive) warped by a saddle of 2 % of the board's width, the paper cases' warping degree 1.

@testitem "one Jacobian implementation: height and focus columns against independent checks" setup = [RCFixtures, EdgeScene] begin
    using RadiometricCamera: build_problem_grid, Focus, HEIGHT_NC, View, with_h, with_surface, Φ_pixel
    import VisualGeometryToolkitCore as VgtkCore
    using StaticArrays, LinearAlgebra, ForwardDiff
    h_w = SVector{HEIGHT_NC}([0.5 * saddle(𝐛) for 𝐛 in RC.HEIGHT_NODES])
    warped(s) = (RC.set_surface!(s, with_h(RC.board_surface(s), h_w)); s)
    s_c = warped(edge_state(edge_s, targets_e))
    prob, _ = build_problem_grid(s_c; patch = (5, 5), height_degree = 1)
    tiles = RC.tile_costs(prob)
    r = tiles[argmax([length(t.pix) for t in tiles])]
    vars = collect(RC.NLLSsolver.getvars(r, prob.variables))
    m = vars[8]
    res, J = RC.NLLSsolver.computeresjac(RC.static(1023), r, vars...)
    @test res == RC.NLLSsolver.computeresidual(r, vars...)                  # bitwise: both from the flux kernel
    hc = (6 + 200 + 9 + 1 + 9) .+ (1:RC.HEIGHT_NFREE)                       # the free height columns (no edge block)
    @test RC.n_height(m) == HEIGHT_NC && size(J, 2) == 6 + 200 + 9 + 1 + 9 + RC.HEIGHT_NFREE + 15 + 27

    # 1. ForwardDiff of the flux through Core's posed height field, pixel by pixel, folded by the block's basis
    D = ForwardDiff.Dual{Nothing, Float64, HEIGHT_NC}
    hd = SVector{HEIGHT_NC, D}(ntuple(i -> D(m.surface.h[i], ForwardDiff.Partials(ntuple(j -> Float64(i == j), Val(HEIGHT_NC)))), Val(HEIGHT_NC)))
    surf_d = RC.BoardSurface{D}(hd, SVector{RC.EDGE_NC, D}(m.surface.edge), SVector{RC.INPLANE_NC, D}(m.surface.inplane))
    scene_d = with_surface(s_c.scene, surf_d)
    view = View(VgtkCore.model(vars[1]), RC._window_lens(s_c.camera.lens, r.P0, vars[2:5]...), VgtkCore.model(vars[6]), vars[7].r)
    Hd = zeros(3 * length(r.pix), HEIGHT_NC)
    for (i, iv) in enumerate(r.pix)
        u, v = s_c.px[r.k][iv]
        Φ = Φ_pixel(u, v, r.k, view, VgtkCore.model(vars[9]), m.mat, vars[10].light, scene_d, s_c.settings)
        for c in 1:3, j in 1:HEIGHT_NC
            Hd[3 * (i - 1) + c, j] = ForwardDiff.partials(Φ[c], j)
        end
    end
    Jfd_ad = Hd * Matrix(m.basis)
    @test norm(Jfd_ad) > 0
    for (q, col) in enumerate(hc)
        @test norm(J[:, col] .- Jfd_ad[:, q]) <= 1e-9 * norm(Jfd_ad[:, q])         # the CPU backend: Float64, the chain rule exact
    end
    # 2. central differences of the tile residual along the free height directions (through `update`: h = h₀ + B y)
    setv(u) = ntuple(i -> i == 8 ? RC.NLLSsolver.update(m, u) : vars[i], 10)
    nm = Int(RC.NLLSsolver.nvars(m))
    for q in (1, 6, 11, 17, RC.HEIGHT_NFREE)
        e = 1e-6
        fd = (RC.NLLSsolver.computeresidual(r, setv(unit_increment(nm, 9 + q, e))...) .- RC.NLLSsolver.computeresidual(r, setv(unit_increment(nm, 9 + q, -e))...)) ./ 2e
        @test norm(J[:, hc[q]] .- fd) <= 1e-4 * norm(fd)
    end

    # 3. a direction-chart focus (focus-ab's kind 1): its columns against central differences of the residual along each node
    ξ = vec([SVector(x, y) for x in (-0.5, 0.0, 0.5), y in (-0.5, 0.0, 0.5)])
    fk = Focus(RC.FocusSpline(ξ, RC.FOCUS_DIRECTION_CHART), [1 / (1.5 + 0.1 * sin(3i)) for i in 1:9])
    s_k = warped(CalibrationState(scene_edge(edge_s), st_e, [View(p, glens_e, fk, 0.0386) for p in poses_e],
                                  vig_e, mat_e, light_e, targets_e; masks = masks_e))
    pk, _ = build_problem_grid(s_k; patch = (5, 5), height_degree = 1)
    rk = RC.tile_costs(pk)[argmax([length(t.pix) for t in RC.tile_costs(pk)])]
    vk = collect(RC.NLLSsolver.getvars(rk, pk.variables))
    _, Jk = RC.NLLSsolver.computeresjac(RC.static(1023), rk, vk...)
    for i in (1, 5, 9)
        upk(x) = ntuple(j -> j == 6 ? RC.NLLSsolver.update(vk[6], [k == i ? x : 0.0 for k in 1:9]) : vk[j], 10)
        # the step 1e-4: every node's column agrees to 2e-6 or better (measured 2026-11-14); a step of 1e-3 crosses kinks of the residual on this
        # scene (up to 1.9 % at nodes 1 and 4), whose printed edge is σ_print wide everywhere since the camera's blur is removed
        fd = (RC.NLLSsolver.computeresidual(rk, upk(1e-4)...) .- RC.NLLSsolver.computeresidual(rk, upk(-1e-4)...)) ./ 2e-4
        @test norm(Jk[:, 6 + 200 + i] .- fd) <= 1e-3 * norm(fd)
    end

    # 4. a held focus node (the state's `focus_fixed`, the focus block's `fixed`): the block's columns are the free nodes' alone
    tile_jacobian(st) = (pf = first(build_problem_grid(st; patch = (5, 5), height_degree = 1));
                         rf = RC.tile_costs(pf)[findfirst(t -> t.k == r.k && t.P0 == r.P0, RC.tile_costs(pf))];
                         last(RC.NLLSsolver.computeresjac(RC.static(1023), rf, RC.NLLSsolver.getvars(rf, pf.variables)...)))
    J0 = tile_jacobian(s_c)
    try
        s_c.focus_fixed = [4]
        Jf = tile_jacobian(s_c)
        fcols = 6 + 200 .+ (1:9)
        @test size(Jf, 2) == size(J0, 2) - 1 && Jf[:, fcols[1:3]] == J0[:, fcols[1:3]] && Jf[:, fcols[4:8]] == J0[:, fcols[5:9]]
        @test Jf[:, fcols[end]:end] == J0[:, (fcols[end] + 1):end]          # the columns after the block, one to the left
    finally
        s_c.focus_fixed = Int[]
    end
end

@testitem "one Jacobian implementation: the CPU and the device backend (GPU)" tags = [:gpu] setup = [RCFixtures, RCGPU, EdgeScene] begin
    using RadiometricCamera: build_problem_grid, Focus, HEIGHT_NC, View, with_h
    import VisualGeometryToolkitCore as VgtkCore
    using StaticArrays, LinearAlgebra
    import CUDA

    require_gpu()
    if RC_GPU_TESTS
        h_w = SVector{HEIGHT_NC}([0.5 * saddle(𝐛) for 𝐛 in RC.HEIGHT_NODES])
        warped(s) = (RC.set_surface!(s, with_h(RC.board_surface(s), h_w)); s)
        pair(; kw...) = (warped(edge_state(edge_s, targets_e; kw...)), warped(edge_state(edge_s, targets_e; backend = CUDA.CUDABackend(), kw...)))
        tile_pair(sc, sg) = begin
            pc, _ = build_problem_grid(sc; patch = (5, 5), height_degree = 1); pg, _ = build_problem_grid(sg; patch = (5, 5), height_degree = 1)
            tg = RC.tile_costs(pg); rg = tg[argmax([length(t.pix) for t in tg])]
            tc = RC.tile_costs(pc); rc = tc[findfirst(t -> t.k == rg.k && t.P0 == rg.P0, tc)]
            vc = RC.NLLSsolver.getvars(rc, pc.variables); vg = RC.NLLSsolver.getvars(rg, pg.variables)
            (rc, vc, rg, vg)
        end
        colrel(Jc, Jg, cols) = maximum(c -> norm(Jg[:, c] .- Jc[:, c]) / norm(Jc[:, c]), cols)

        # 1. the one implementation on the two backends: the chunks' columns to the device's Float32 rounding, the nodal heights' to
        # Float64's (their path is Float64 on the device too, `HIGH_FLOAT`: 2.7e-14 here, 2026-11-11 — in Float32 they were 1e-3, and
        # 0.2–1.3 % on the paper-like Siemens case, experiments/test/pipeline/jacobian_parity.jl)
        s_c, s_g = pair()
        rc, vc, rg, vg = tile_pair(s_c, s_g)
        rjc, Jc = RC.NLLSsolver.computeresjac(RC.static(1023), rc, vc...)
        rjg, Jg = RC.NLLSsolver.computeresjac(RC.static(1023), rg, vg...)
        @test rjg == RC.NLLSsolver.computeresidual(rg, vg...)                     # bitwise: both from the flux kernel
        @test maximum(abs, rjg .- rjc) < 2e-3 * max(1.0, maximum(abs, rjc))
        hc = (6 + 200 + 9 + 1 + 9) .+ (1:RC.HEIGHT_NFREE)
        @test colrel(Jc, Jg, hc) < 1e-9
        @test colrel(Jc, Jg, 1:6) < 1e-9                                           # the pose: its chunks in Float64 too (`HIGH_FLOAT`)
        @test colrel(Jc, Jg, 6 + 200 .+ (1:9)) < 1e-2                              # the focus

        # 2. a held focus node: the device's block drops its column too
        try
            s_c.focus_fixed = [4]; s_g.focus_fixed = [4]
            rc2, vc2, rg2, vg2 = tile_pair(s_c, s_g)
            _, Jg2 = RC.NLLSsolver.computeresjac(RC.static(1023), rg2, vg2...)
            @test size(Jg2, 2) == size(Jg, 2) - 1 && colrel(Jg, Jg2, 6 + 200 .+ (1:3)) < 1e-12 && colrel(Jg[:, 6 + 200 .+ (5:9)], Jg2[:, 6 + 200 .+ (4:8)], 1:5) < 1e-12
        finally
            s_c.focus_fixed = Int[]; s_g.focus_fixed = Int[]
        end

        # 3. a direction-chart focus (focus-ab's kind 1): the focus columns are the chart's weights through the lens of the chart's own
        # surface (before 2026-11-11 the device evaluated the dual focus as a sensor TPS, kind 0, and folded it with the sensor TPS's
        # weights: its columns were 1.5–4× off). The device against the CPU, and against central differences of the device residual
        # along each node (the Float32 differences: 0.7–2.2 % on this scene, 2026-11-11)
        ξ = vec([SVector(x, y) for x in (-0.5, 0.0, 0.5), y in (-0.5, 0.0, 0.5)])
        fk = Focus(RC.FocusSpline(ξ, RC.FOCUS_DIRECTION_CHART), [1 / (1.5 + 0.1 * sin(3i)) for i in 1:9])
        chart_state(be) = warped(CalibrationState(scene_edge(edge_s), st_e, [View(p, glens_e, fk, 0.0386) for p in poses_e],
                                                  vig_e, mat_e, light_e, targets_e; masks = masks_e, backend = be))
        rck, vck, rgk, vgk = tile_pair(chart_state(RC.KernelAbstractions.CPU()), chart_state(CUDA.CUDABackend()))
        _, Jck = RC.NLLSsolver.computeresjac(RC.static(1023), rck, vck...)
        _, Jgk = RC.NLLSsolver.computeresjac(RC.static(1023), rgk, vgk...)
        fcols = 6 + 200 .+ (1:9)
        @test colrel(Jck, Jgk, fcols) < 2e-2
        vg = collect(vgk)
        for i in (1, 5, 9)
            e = 1e-3
            up(x) = ntuple(j -> j == 6 ? RC.NLLSsolver.update(vg[6], [k == i ? x : 0.0 for k in 1:9]) : vg[j], 10)
            fd = (RC.NLLSsolver.computeresidual(rgk, up(e)...) .- RC.NLLSsolver.computeresidual(rgk, up(-e)...)) ./ 2e
            @test norm(Jgk[:, fcols[i]] .- fd) <= 5e-2 * norm(fd)
        end
    end
end
