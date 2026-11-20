# The board warping h (`BoardSurface.h`, `board_height`, `intersect_board` with the field — the out-of-plane shape of the print,
# paper §Board Warping): the model and the ray–surface intersection, its dual-number Jacobian columns in the grid-tile path, LM recovery
# of a known saddle, CPU/GPU parity.

# The saddle and the two-view scene with its targets both items use.
@testsetup module HeightScene
    using RadiometricCamera: BoardSurface, compute_mask, Focus, FocusSpline, grid_model, HEIGHT_NC, Material, PrintedTarget,
                             render_all, RenderSettings, Scene, sphere_radius, with_surface, 𝔟
    using RadiometricCamera, StaticArrays, LinearAlgebra, ForwardDiff
    using VisualGeometryToolkitCore: exp_so3
    using VisualRadiometryToolkitCore: TensorProductVignetting, with_control_points, EnvironmentLight, SphericalHarmonicMap
    import VisualGeometryToolkitCore as VgtkCore
    const RC = RadiometricCamera
    include(joinpath(@__DIR__, "fixture_board.jl"))
    include(joinpath(@__DIR__, "fixture_camera.jl"))
    export page_board, sad, hh, tex_h, board_hs, Wh, Hh, scene_h, scene_hh, glens_h, focus_h, poses_h, camera_h, vig_h, b_h, a_h, light_h, mat_h, st_h, masks_h, blank_h, height_true, targets_h

    """A board of `aspect` = height / width (2 m wide), seen from z > 0."""
    page_board(aspect) = PrintedTarget([-1.0, -1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0 * aspect, 0.0], rand(4, 4, 3); facing = [0.0, 0.0, 1.5])
    hh = SVector{HEIGHT_NC}([saddle(𝐛) for 𝐛 in RC.HEIGHT_NODES])
    # ---------------------------------------------------------------------------------------------------------------
    # 2. a two-view scene of the blob fixture; the truth = a saddle with a 2 % sag of the board width at the edge (≈ 0.6 px at 40°)
    tex_h = fixture_board()
    board_hs = PrintedTarget([-1.0, -1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], tex_h; facing = [0.0, 0.0, 1.5])
    Wh = Hh = 32
    scene_h = Scene(board_hs, Wh, Hh)
    scene_hh(h) = with_surface(scene_h, BoardSurface(h = h))               # the warping is the scene's surface parameter
    glens_h = grid_model(cam(VgtkCore.Lens(VgtkCore.DoubleSphere(0.25, 0.4)), 41.216, (16.32, 15.36), Wh, Hh), 10, 10)
    focus_h = Focus(FocusSpline(VgtkCore.grid_nodes(3, VgtkCore.Rect(0, 0, Wh, Hh))), [1 / (1.5 + 0.1 * sin(3i)) for i in 1:9])
    poses_h = [VgtkCore.EuclideanMap(SMatrix{3, 3}(exp_so3(SVector(π, 0.0, 0.0)) * exp_so3(SVector(0.15 * cos(k), 0.12 * sin(k), 0.1 * k))), SVector{3}(SVector(0.2 * cos(k), 0.2 * sin(k), 1.2 + 0.1 * k)); check = false) for k in 1:2]
    vig_h = let v = TensorProductVignetting((4, 4), VgtkCore.Rect(0, 0, Wh, Hh); degree = 2); gr = range(-0.5, 0.5, length = 4); with_control_points(v, [1.0 - 0.3 * (gr[i]^2 + gr[j]^2) for i in 1:4, j in 1:4]) end
    b_h = 𝔟(board_hs); a_h = zeros(3, 9); a_h[:, 1] .= (2.0, 1.9, 1.8); a_h[:, 3] .= (0.4, 0.3, 0.2)
    light_h = EnvironmentLight(SphericalHarmonicMap(a_h, 2), b_h, sphere_radius(b_h, [p.t for p in poses_h]))
    mat_h = Material(; roughness_ink = 0.12, roughness_paper = 0.35, refractive_index_ink = 1.65, refractive_index_paper = 1.33, diffuse_weight = 0.6)
    st_h = RenderSettings(; n_𝐩 = 2, n_ρ = 2, n_φ = 6, N = 4)
    camera_h = RadiometricCameraModel(glens_h, focus_h, 0.0386, vig_h)
    masks_h = [compute_mask(camera_h, p, scene_h, st_h; δ = 0.02, backend = RadiometricCamera.KernelAbstractions.CPU()) for p in poses_h]             # the mask is the plane's (the surface stays inside the padding)
    blank_h = [zeros(Hh, Wh, 3) for _ in 1:2]
    height_true = hh                                                                      # the saddle, as nodal heights
    targets_h = render_all(CalibrationState(scene_hh(height_true), st_h, camera_h, poses_h, mat_h, light_h, blank_h; masks = masks_h))
end

@testitem "board height field" setup = [RCFixtures, HeightScene] begin
    using RadiometricCamera: board_height, board_normal, board_surface, BoardSurface, build_problem_grid, chart, cost, EDGE_NC,
                             HEIGHT_NC, height_nparams, height_on, intersect_board, posed_target, PrintedTarget, render_all,
                             run_lm_grid!, Scene, with_edge, with_h, with_surface, world_point
    import VisualGeometryToolkitCore as VgtkCore
    using StaticArrays, LinearAlgebra, Statistics, ForwardDiff
    using VisualGeometryToolkitCore: exp_so3

    # ---------------------------------------------------------------------------------------------------------------
    # 1. the model: the plane when off, the thin plate spline, the surface intersection (on the ray, on the surface, the sign), duals, bookkeeping
    sf0 = BoardSurface()
    @test !height_on(sf0) && iszero(sf0.h) && length(sf0.h) == HEIGHT_NC
    @test height_nparams(0) == 0 && height_nparams(1) == HEIGHT_NC && height_nparams(2) == HEIGHT_NC && height_nparams(3) == HEIGHT_NC
    # nodal heights of a saddle (thin plate spline through them, affine part dropped): an affine set of nodal heights is the plane
    W1 = page_board(1.0)                                                                    # a square page
    haff = SVector{HEIGHT_NC}([0.01 + 0.02 * 𝐛[1] - 0.03 * 𝐛[2] for 𝐛 in RC.HEIGHT_NODES])
    for 𝐛 in (SVector(0.5, 0.5), SVector(0.1, 0.9), SVector(0.97, 0.03))
        @test abs(board_height(W1, haff, 𝐛)) < 1e-12                                        # piston + tilts are the poses': removed by construction
        @test board_height(W1, hh + haff, 𝐛) ≈ board_height(W1, hh, 𝐛) atol = 1e-12
    end
    # the bending part of the saddle: at the nodes the TPS interpolates, so the bending part = nodal value − the TPS's own affine part
    let (w, a) = RC.VgtkCore.tps_coefficients(VgtkCore.ThinPlateSplineSystem(RC.HEIGHT_NODES), collect(hh))
        for (i, 𝐛) in enumerate(RC.HEIGHT_NODES)
            @test board_height(W1, hh, 𝐛) ≈ hh[i] - (a[1] + a[2] * 𝐛[1] + a[3] * 𝐛[2]) atol = 1e-10
        end
    end
    board_h1 = PrintedTarget([-1.0, -1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], rand(8, 8, 3); facing = [0.0, 0.0, 1.5])
    @test board_h1.warping isa VgtkCore.MappedSurface                                     # the flat warping's parts, Core's surface
    oh = SVector(0.3, -0.2, 1.4); dh = normalize(SVector(-0.4, 0.3, -1.0))
    sh0, 𝐱h0 = intersect_board(board_h1, oh, dh)
    @test intersect_board(board_h1, oh, dh, zero(SVector{HEIGHT_NC, Float64})) == (sh0, 𝐱h0)
    @test intersect_board(board_h1, oh, dh, sf0.h) == (sh0, 𝐱h0)
    sh1, 𝐱h1 = intersect_board(board_h1, oh, dh, hh)
    @test 𝐱h1 ≈ oh + sh1 * dh
    @test abs(dot(𝐱h1 - world_point(board_h1, SVector(0.0, 0.0)), board_normal(board_h1)) - board_height(board_h1, hh, chart(board_h1, 𝐱h1))) < 1e-10     # the hit lies on the surface (Core's root)
    # the warped board as Core's posed target with a HeightField calibration: Core's surface of it is the board's surface
    let pt = posed_target(board_h1, with_h(BoardSurface(), hh)), sf = RC.VgtkCore.surface(pt)
        @test sf isa RC.VgtkCore.MappedSurface
        sc, 𝐱c = RC.VgtkCore.intersection(sf, oh, dh)
        @test abs(sc - sh1) < 1e-12 && norm(𝐱c - 𝐱h1) < 1e-12
    end
    @test 1e-4 < norm(𝐱h1 - 𝐱h0) < 0.1                                                 # the bending part of the saddle moves the hit by ≈ 0.6 mm here
    # the sign: z > 0 = towards the camera (the `facing` side of the normal)
    hz = SVector{HEIGHT_NC}([0.1 * ((𝐛[1] - 0.5)^2 + (𝐛[2] - 0.5)^2) for 𝐛 in RC.HEIGHT_NODES])    # a bowl rising towards the camera at the edges (its bending part: positive at the corners)
    𝐱he = intersect_board(board_h1, oh, normalize(SVector(-0.8, 0.8, -1.0)), hz)[2]   # near the corner (−0.82, 0.92), on the page
    @test dot(𝐱he - world_point(board_h1, SVector(0.0, 0.0)), board_normal(board_h1)) > 0 && dot(board_normal(board_h1), SVector(0.0, 0.0, 1.5) - world_point(board_h1, SVector(0.0, 0.0))) > 0
    # duals flow through the surface intersection: ∂𝐱/∂d₁ against a central difference
    ddh = SVector(ForwardDiff.Dual(dh[1], 1.0), ForwardDiff.Dual(dh[2], 0.0), ForwardDiff.Dual(dh[3], 0.0))
    _, 𝐱hd = intersect_board(board_h1, SVector(ForwardDiff.Dual(oh[1], 0.0), ForwardDiff.Dual(oh[2], 0.0), ForwardDiff.Dual(oh[3], 0.0)), ddh, hh)
    let ε = 1e-6
        𝐱p = intersect_board(board_h1, oh, dh + SVector(ε, 0.0, 0.0), hh)[2]; 𝐱m = intersect_board(board_h1, oh, dh - SVector(ε, 0.0, 0.0), hh)[2]
        cd = (𝐱p - 𝐱m) / 2ε
        @test norm(ForwardDiff.partials.(𝐱hd, 1) - cd) < 1e-6 * norm(cd)
    end
    # bookkeeping: the surface carries the nodal heights; the modifiers and the precision conversion keep them
    shh = with_h(sf0, hh)
    @test height_on(shh) && shh.edge == sf0.edge
    @test with_edge(shh, shh.edge).h == hh && VgtkCore.convert_eltype(Float32, shh).h == SVector{HEIGHT_NC, Float32}(hh)
    @test BoardSurface(h = hh).h == hh && BoardSurface(h = hh).edge == zero(SVector{EDGE_NC, Float64})
    @test Scene(board_h1, 8, 8).surface == BoardSurface() && with_surface(Scene(board_h1, 8, 8), shh).surface.h == hh

    r_nom_h = render_all(CalibrationState(scene_h, st_h, camera_h, poses_h, mat_h, light_h, blank_h; masks = masks_h))
    @test maximum(maximum(abs, targets_h[k] .- r_nom_h[k]) for k in 1:2) > 0.02 * maximum(maximum, r_nom_h)

    # ---------------------------------------------------------------------------------------------------------------
    # 3. Jacobian columns of the free height parameters (material block: HEIGHT_NFREE AD columns after the 9 BRDF parameters; h = h₀ + B y) vs differences of the tile
    #    residual through `update`; the pose columns still match (duals flow through the surface intersection)
    s_h = CalibrationState(scene_hh(0.5 * height_true), st_h, camera_h, poses_h, mat_h, light_h, targets_h; masks = masks_h)
    prob_h, idx_h = build_problem_grid(s_h; patch = (5, 5), height_degree = 2)
    tiles_h = RC.tile_costs(prob_h)
    r_h = tiles_h[argmax([length(t.pix) for t in tiles_h])]
    vars_h = collect(RC.NLLSsolver.getvars(r_h, prob_h.variables))
    @test RC.NLLSsolver.nvars(vars_h[8]) == 9 + RC.HEIGHT_NFREE && RC.n_height(vars_h[8]) == HEIGHT_NC
    res_h, J_h = RC.NLLSsolver.computeresjac(RC.static(1023), r_h, vars_h...)
    ncol_h = 6 + 200 + 9 + 1 + (9 + RC.HEIGHT_NFREE) + 15 + 27          # (15: the hard pin eliminates the centre's vignetting direction)
    @test size(J_h) == (3 * length(r_h.pix), ncol_h)
    @test res_h ≈ RC.NLLSsolver.computeresidual(r_h, vars_h...) atol = 1e-10
    fdh(j, u) = (RC.NLLSsolver.computeresidual(r_h, updated_vars(vars_h, j, u)...) - RC.NLLSsolver.computeresidual(r_h, updated_vars(vars_h, j, -u)...)) / (2 * norm(u))
    for q in (1, 7, 13, RC.HEIGHT_NFREE)                               # AD columns of the free height parameters vs central differences through `update`
        fd = fdh(8, unit_increment(9 + RC.HEIGHT_NFREE, 9 + q, 1e-5))
        @test norm(fd) > 0
        @test norm(J_h[:, 6 + 200 + 9 + 1 + 9 + q] .- fd) / norm(fd) < 5e-3
    end
    fdμh = fdh(8, unit_increment(9 + RC.HEIGHT_NFREE, 9, 1e-6))                                 # μ column unchanged by the extra block
    @test norm(J_h[:, 6 + 200 + 9 + 1 + 9] .- fdμh) / norm(fdμh) < 1e-5
    for j in 1:6                                                       # pose columns (AD through the surface intersection) vs central differences
        fd = fdh(1, unit_increment(6, j, 1e-7))
        @test norm(J_h[:, j] .- fd) / norm(fd) < 5e-3
    end
    # the free height parameters move the nodal heights along the columns of the board's warping's basis (no piston, no tilt); the
    # material is untouched
    @test vars_h[8].basis == RC.height_basis(board_hs)
    v_up = RC.NLLSsolver.update(vars_h[8], unit_increment(9 + RC.HEIGHT_NFREE, 9 + RC.HEIGHT_NFREE, 1e-3))
    @test v_up.surface.h ≈ vars_h[8].surface.h + 1e-3 * vars_h[8].basis[:, RC.HEIGHT_NFREE] && v_up.mat == vars_h[8].mat
    v_up2 = RC.NLLSsolver.update(vars_h[8], unit_increment(9 + RC.HEIGHT_NFREE, 9 + 1, 1e-3))
    @test v_up2.surface.h ≈ vars_h[8].surface.h + 1e-3 * vars_h[8].basis[:, 1]
    Aff = VgtkCore.ThinPlateSplineSystem(RC.HEIGHT_NODES).coeff[(RC.HEIGHT_NC + 1):(RC.HEIGHT_NC + 3), :]    # the interpolant's affine term a(v) = Aff v
    @test maximum(abs, Aff * (v_up2.surface.h - vars_h[8].surface.h)) < 1e-12                       # pure bending: no affine term in the free directions

    # ---------------------------------------------------------------------------------------------------------------
    # 4. LM recovery of the saddle from the flat start, everything else at the truth
    s_hr = CalibrationState(scene_h, st_h, camera_h, poses_h, mat_h, light_h, targets_h; masks = masks_h)
    c0h = cost(s_hr)[1]
    run_lm_grid!(s_hr; patch = (5, 5), maxiters = 25, height_degree = 2,
                 fixed = (:poses, :grid, :focus, :aperture, :vignetting, :lighting))
    @test cost(s_hr)[1] < 1e-3 * c0h
    let pts = [SVector(x, y) for x in 0.05:0.3:0.95 for y in 0.05:0.3:0.95], zt = [board_height(W1, height_true, 𝐛) for 𝐛 in pts], zr = [board_height(W1, board_surface(s_hr).h, 𝐛) for 𝐛 in pts]
        @test maximum(abs.(zr .- zt)) < 0.1 * maximum(abs.(zt))                           # the SURFACE is recovered (the affine null directions of the nodal heights are not observable)
    end
end

@testitem "board height field (GPU)" tags = [:gpu] setup = [RCFixtures, RCGPU, HeightScene] begin
    using RadiometricCamera: build_problem_grid
    using StaticArrays, LinearAlgebra, Statistics, ForwardDiff
    using VisualGeometryToolkitCore: exp_so3
    import CUDA

    # ---------------------------------------------------------------------------------------------------------------
    # 5. device path: CPU/GPU parity of the height columns, batched == unbatched
    require_gpu()                                  # errors with the reason when the GPU the runner chose is not usable here
    if RC_GPU_TESTS
        scene_hg = scene_hh(0.5 * height_true)
        s_g = CalibrationState(scene_hg, st_h, camera_h, poses_h, mat_h, light_h, targets_h; masks = masks_h, backend = CUDA.CUDABackend())
        prob_g, idx_g = build_problem_grid(s_g; patch = (5, 5), height_degree = 2)
        tiles_g = RC.tile_costs(prob_g)
        rg = tiles_g[argmax([length(t.pix) for t in tiles_g])]
        s_c = CalibrationState(scene_hg, st_h, camera_h, poses_h, mat_h, light_h, targets_h; masks = masks_h)
        prob_c, _ = build_problem_grid(s_c; patch = (5, 5), height_degree = 2)
        tiles_c = RC.tile_costs(prob_c)
        rc = tiles_c[findfirst(t -> t.k == rg.k && t.P0 == rg.P0 && length(t.pix) == length(rg.pix), tiles_c)]
        varsg = RC.NLLSsolver.getvars(rg, prob_g.variables)
        rjg, Jg = RC.NLLSsolver.computeresjac(RC.static(1023), rg, varsg...)
        rjc, Jc = RC.NLLSsolver.computeresjac(RC.static(1023), rc, RC.NLLSsolver.getvars(rc, prob_c.variables)...)
        @test maximum(abs, rjg - rjc) < 2e-3
        for q in 1:3
            col = 6 + 200 + 9 + 1 + 9 + q
            @test norm(Jc[:, col] .- Jg[:, col]) / norm(Jc[:, col]) < 5e-3                          # a forward difference of two Float32 renders
        end
        for j in 1:6
            @test norm(Jc[:, j] .- Jg[:, j]) / norm(Jc[:, j]) < 3e-2                        # Float32 device twin of the pose columns through the surface
        end
        RC._register_grid_problem!(s_g, prob_g, idx_g)
        rjg_b, Jg_b = RC.NLLSsolver.computeresjac(RC.static(1023), rg, varsg...)
        @test device_parity(rjg_b, rjg) && device_parity(Jg_b, Jg)
        RC._unregister_grid_problem!(s_g)
    end
end

@testitem "board warping at a non-square aspect: Float32 and the device (CPU side)" setup = [RCFixtures, HeightScene] begin
    using RadiometricCamera: HEIGHT_NC, intersect_board, PrintedTarget, warping_aspect
    using StaticArrays, LinearAlgebra
    import VisualGeometryToolkitCore as VgtkCore
    const Adapt = RC.Adapt
    # a board of a non-square page carries its page's warping (Core's flat surface over its nodes), and its Float32 board the
    # converted one: the same hits to Float32 rounding; adapt (an Array adaptor) keeps it bit for bit
    tall = PrintedTarget([-1.0, -0.7, 0.0], [2.0, 0.0, 0.0], [0.0, 1.4, 0.0], tex_h; facing = [0.0, 0.0, 1.5])
    @test warping_aspect(tall) ≈ 0.7 && tall.warping isa VgtkCore.MappedSurface
    t32 = VgtkCore.convert_eltype(Float32, tall)
    ta = Adapt.adapt(Array, t32)
    h32 = SVector{HEIGHT_NC, Float32}(hh)
    o = SVector(0.1, 0.05, 1.4)
    for d in (normalize(SVector(-0.3, 0.2, -1.0)), normalize(SVector(0.4, -0.1, -1.0)), normalize(SVector(0.0, 0.3, -1.0)))
        s64, x64 = intersect_board(tall, o, d, hh)
        s32, x32 = intersect_board(t32, SVector{3, Float32}(o), SVector{3, Float32}(d), h32)
        @test isfinite(s64) && abs(s32 - s64) < 1e-5 && norm(x32 - x64) < 1e-5
        @test intersect_board(ta, SVector{3, Float32}(o), SVector{3, Float32}(d), h32) == (s32, x32)
    end
    @test RC.height_basis(tall) != RC.height_basis(page_board(1.0))                    # the tall page's basis, not the square's
end

@testitem "board warping at a non-square aspect (GPU)" tags = [:gpu] setup = [RCFixtures, RCGPU, HeightScene] begin
    using RadiometricCamera: BoardSurface, build_problem_grid, compute_mask, PrintedTarget, render_all, Scene, warping_aspect,
                             with_surface
    using StaticArrays, LinearAlgebra
    import CUDA
    import VisualGeometryToolkitCore as VgtkCore
    const Adapt = RC.Adapt
    # the device path of a page that is not square: the adapted flat surface (its field's system on the device) at
    # α = 0.7 give the CPU's residuals and height columns
    require_gpu()
    if RC_GPU_TESTS
        tall = PrintedTarget([-1.0, -0.7, 0.0], [2.0, 0.0, 0.0], [0.0, 1.4, 0.0], tex_h; facing = [0.0, 0.0, 1.5])
        @test warping_aspect(tall) ≈ 0.7
        wd = Adapt.adapt(CUDA.CuArray, VgtkCore.convert_eltype(Float32, tall.warping))
        @test wd isa VgtkCore.MappedSurface                                     # the flat field's system on the device
        scene_t = with_surface(Scene(tall, Wh, Hh), BoardSurface(h = 0.5 * height_true))
        masks_t = [compute_mask(camera_h, p, Scene(tall, Wh, Hh), st_h; δ = 0.02, backend = RadiometricCamera.KernelAbstractions.CPU()) for p in poses_h]
        targets_t = render_all(CalibrationState(with_surface(Scene(tall, Wh, Hh), BoardSurface(h = height_true)), st_h, camera_h, poses_h, mat_h, light_h, blank_h; masks = masks_t))
        s_g = CalibrationState(scene_t, st_h, camera_h, poses_h, mat_h, light_h, targets_t; masks = masks_t, backend = CUDA.CUDABackend())
        s_c = CalibrationState(scene_t, st_h, camera_h, poses_h, mat_h, light_h, targets_t; masks = masks_t)
        @test maximum(maximum(abs, a .- b) for (a, b) in zip(render_all(s_g), render_all(s_c))) < 2e-3 * maximum(maximum, render_all(s_c))
        prob_g, _ = build_problem_grid(s_g; patch = (5, 5), height_degree = 2)
        prob_c, _ = build_problem_grid(s_c; patch = (5, 5), height_degree = 2)
        tiles_g = RC.tile_costs(prob_g); tiles_c = RC.tile_costs(prob_c)
        rg = tiles_g[argmax([length(t.pix) for t in tiles_g])]
        rc = tiles_c[findfirst(t -> t.k == rg.k && t.P0 == rg.P0 && length(t.pix) == length(rg.pix), tiles_c)]
        varsg = RC.NLLSsolver.getvars(rg, prob_g.variables)
        @test varsg[8].basis == RC.height_basis(tall)                  # the tall page's basis, not the square's
        rjg, Jg = RC.NLLSsolver.computeresjac(RC.static(1023), rg, varsg...)
        rjc, Jc = RC.NLLSsolver.computeresjac(RC.static(1023), rc, RC.NLLSsolver.getvars(rc, prob_c.variables)...)
        @test maximum(abs, rjg - rjc) < 2e-3
        for q in (1, 7, 13, RC.HEIGHT_NFREE)
            col = 6 + 200 + 9 + 1 + 9 + q
            @test norm(Jc[:, col] .- Jg[:, col]) / norm(Jc[:, col]) < 5e-3
        end
    end
end

@testitem "board warping: Core's height manifold moves the pure-bending heights (no gauge)" begin
    using RadiometricCamera: board_height, BoardSurface, HEIGHT_NC, HEIGHT_NFREE, Material, PrintedTarget, warping_nodes
    using RadiometricCamera, StaticArrays, LinearAlgebra
    import VisualGeometryToolkitCore as VgtkCore
    const RC = RadiometricCamera
    page_board(aspect) = PrintedTarget([-1.0, -1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0 * aspect, 0.0], rand(4, 4, 3); facing = [0.0, 0.0, 1.5])
    W1 = page_board(1.0)
    B = Matrix(RC.height_basis(W1))
    @test B == VgtkCore.retract_jacobian(RC.height_manifold(W1))
    # the square page's manifold is Core's height field over its placed nodes (u, −v): the same span as over the page fractions
    # themselves (a reflection keeps the affine rows' null space)
    B0 = VgtkCore.retract_jacobian(VgtkCore.TargetManifold(VgtkCore.PlanarTarget(collect(RC.HEIGHT_NODES); calibration = VgtkCore.HeightField(collect(RC.HEIGHT_NODES)))))
    @test rank(hcat(B, B0); rtol = 1e-10) == HEIGHT_NC - 3
    @test size(B) == (HEIGHT_NC, HEIGHT_NC - 3) && HEIGHT_NFREE == HEIGHT_NC - 3 && rank(B) == HEIGHT_NC - 3
    # Core 0.5's rule (the thin-plate kinds are pure bending, no gauge): the free nodal heights are those whose interpolant has NO
    # affine term — the null space of the interpolant's affine rows a(v), not the orthogonal complement of the plane fields {1, b₁, b₂}
    # over the nodes (the warping's own basis until 2026-10-28, and Core's gauge until 0.5); the projection onto them is oblique,
    # v ↦ v − M a(v) along the plane fields M
    M = hcat(ones(HEIGHT_NC), [b[1] for b in RC.HEIGHT_NODES], [b[2] for b in RC.HEIGHT_NODES])
    Aff = VgtkCore.ThinPlateSplineSystem(RC.HEIGHT_NODES).coeff[(HEIGHT_NC + 1):(HEIGHT_NC + 3), :]    # a(v) = Aff v
    @test maximum(abs, Aff * B) < 1e-12                                  # no affine term in the free directions
    @test maximum(abs, Aff * M - I) < 1e-12                              # the plane fields are their own affine term
    v = [sin(1.7i) for i in 1:HEIGHT_NC]
    pv = v - M * (Aff * v)                                               # the oblique projection: in the free span, the same bending
    @test norm(pv - B * (B \ pv)) < 1e-12 * norm(v)
    @test maximum(abs(board_height(W1, SVector{HEIGHT_NC}(pv), SVector(x, y)) - board_height(W1, SVector{HEIGHT_NC}(v), SVector(x, y)))
                  for x in 0:0.1:1, y in 0:0.1:1) < 1e-12
    # what the warping drops is invisible: an affine set of nodal heights is the plane, on a page of any aspect
    h = SVector{HEIGHT_NC}(M * [0.3, -0.2, 0.5])
    for wp in (W1, page_board(0.5), page_board(1.5))
        @test maximum(abs(board_height(wp, h, SVector(x, y))) for x in 0:0.1:1, y in 0:0.1:1) < 1e-12
    end
    # every page's warping has its own basis: the null space of THAT warping's affine rows (the interpolant's affine part depends on
    # the aspect). The square page's basis carries an affine part on another page, which board_height would drop silently
    probes = [SVector(x, y) for x in 0.05:0.1:0.95 for y in 0.05:0.1:0.95]
    for α in (0.7, 0.4)
        wp = page_board(α)
        Bα = Matrix(RC.height_basis(wp))
        Affα = VgtkCore.ThinPlateSplineSystem(warping_nodes(α)).coeff[(HEIGHT_NC + 1):(HEIGHT_NC + 3), :]    # that page's affine rows
        y = [cos(0.9k) for k in 1:HEIGHT_NFREE]
        @test maximum(abs, Affα * (Bα * y)) < 1e-12 * norm(y)
        @test maximum(abs, Affα * Bα) < 1e-12 && rank(Bα) == HEIGHT_NFREE
        @test maximum(abs, Affα * B) > 1e-3                              # the square page's basis is not pure bending here
        # and it reaches every bending: the warping of B_α y over all y spans 22 independent surfaces
        Z = hcat([[board_height(wp, SVector{HEIGHT_NC}(Bα[:, k]), p) for p in probes] for k in 1:HEIGHT_NFREE]...)
        @test rank(Z; rtol = 1e-10) == HEIGHT_NFREE
        # the material block takes its board's basis: a step of the free height parameters is pure bending of that page's spline
        mv = RC.MaterialVar(Material(; roughness_ink = 0.12, roughness_paper = 0.35, refractive_index_ink = 1.65, refractive_index_paper = 1.33, diffuse_weight = 0.6), BoardSurface(), wp; height = HEIGHT_NC)
        @test mv.basis == RC.height_basis(wp)
        u = zeros(9 + HEIGHT_NFREE); u[10:end] .= 1e-3 .* y
        @test maximum(abs, Affα * RC.NLLSsolver.update(mv, u).surface.h) < 1e-15
    end
    @test_throws ArgumentError RC.MaterialVar(Material(; roughness_ink = 0.12, roughness_paper = 0.35, refractive_index_ink = 1.65, refractive_index_paper = 1.33, diffuse_weight = 0.6), BoardSurface(); height = HEIGHT_NC)
end

@testitem "board warping: the spline is the page's physical bending, whatever its aspect" begin
    using RadiometricCamera: board_height, board_normal, BoardSurface, chart, HEIGHT_NC, intersect_board, posed_target,
                             PrintedTarget, warping_aspect, with_h, world_point
    using RadiometricCamera, StaticArrays, LinearAlgebra
    import VisualGeometryToolkitCore as VgtkCore
    const RC = RadiometricCamera
    # The warping's thin-plate spline is over the page in units of its width: a uniform scaling of the page's metric coordinates, so
    # it is the thin-plate spline through the same nodal heights at the nodes' METRIC positions (Core's, built in metres), whatever
    # the aspect — and not the one over the page fractions scaled per axis (the warping until 2026-11-06), which bends differently
    # on a page that is not square.
    h = SVector{HEIGHT_NC}([1e-3 * sin(2.1i) for i in 1:HEIGHT_NC])
    for (Wm, Hm) in ((2.0, 2.0), (0.8, 0.6), (0.297, 0.210), (1.0, 2.5))
        b = PrintedTarget([0.0, 0.0, 0.0], [Wm, 0.0, 0.0], [0.0, Hm, 0.0], rand(4, 4, 3); facing = [0.4, 0.3, 1.5])
        metric = VgtkCore.ThinPlateSplineSystem([SVector(n[1] * Wm, n[2] * Hm) for n in RC.HEIGHT_NODES])
        wm, am = VgtkCore.tps_coefficients(metric, collect(h))
        probes = [SVector(x, y) for x in 0.05:0.1:0.95 for y in 0.05:0.1:0.95]
        # board_height drops the affine part of its own decomposition: it differs from the metric spline (affine part included) by
        # an affine function of 𝐛, and by nothing else
        Δ = [board_height(b, h, 𝐛) - VgtkCore.tps_evaluate(metric, wm, am, SVector(𝐛[1] * Wm, 𝐛[2] * Hm)) for 𝐛 in probes]
        A = hcat(ones(length(probes)), first.(probes), last.(probes))
        @test maximum(abs, Δ - A * (A \ Δ)) < 1e-14
        # the board of that page: its warping, the relief over its quad (metres) and Core's posed target agree on the surface
        @test warping_aspect(b) == Hm / Wm
        o = SVector(0.31 * Wm, 0.62 * Hm, 1.2); d = normalize(SVector(0.02, -0.03, -1.0))
        s1, x1 = intersect_board(b, o, d, h)
        @test abs(dot(x1 - world_point(b, SVector(0.0, 0.0)), board_normal(b)) - board_height(b, h, chart(b, x1))) < 1e-12
        sc, xc = VgtkCore.intersection(VgtkCore.surface(posed_target(b, with_h(BoardSurface(), h))), o, d)
        @test abs(sc - s1) < 1e-12 && norm(xc - x1) < 1e-12
    end
    # the per-axis chart is NOT the same bending on a non-square page (the defect this replaces): a measurable difference
    chart_tps = VgtkCore.ThinPlateSplineSystem(RC.HEIGHT_NODES)
    page_board(aspect) = PrintedTarget([-1.0, -1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0 * aspect, 0.0], rand(4, 4, 3); facing = [0.0, 0.0, 1.5])
    wc, _ = VgtkCore.tps_coefficients(chart_tps, collect(h))
    old(𝐛) = VgtkCore.tps_evaluate(chart_tps.nodes, wc, zero(SVector{3, Float64}), 𝐛)
    wp = page_board(0.5)
    probes = [SVector(x, y) for x in 0.05:0.1:0.95 for y in 0.05:0.1:0.95]
    Δ = [board_height(wp, h, p) - old(p) for p in probes]
    Δ .-= hcat(ones(length(probes)), first.(probes), last.(probes)) * (hcat(ones(length(probes)), first.(probes), last.(probes)) \ Δ)
    @test maximum(abs, Δ) > 1e-6
    # and on a square page it is the chart's (Core's field over the reflected nodes: the same bending, to the sums' rounding)
    @test maximum(abs(board_height(page_board(1.0), h, p) - old(p)) for p in probes) < 1e-15
end

@testitem "board warping: the warped intersection under ForwardDiff (Core's implicit derivative) through the board's rays" setup = [RCFixtures, HeightScene] begin
    using RadiometricCamera: HEIGHT_NC, intersect_board
    using StaticArrays, LinearAlgebra, ForwardDiff
    # Core's MappedSurface root is differentiated by the implicit function theorem, its one path (the former per-backend choice of
    # the derivative is retired): the board's rays as every render forms them give the per-ray form's values, and the partials against
    # central differences. The values to rounding, not bit for bit: plain numbers take Core's 1-D Newton of a height field over an
    # affine chart, duals the generic 2 × 2 root (Core measured the two within 8.9e-16 of each other)
    o = SVector(0.1, -0.2, 1.4); d = normalize(SVector(0.05, 0.1, -1.0))
    hd = SVector{HEIGHT_NC}(ntuple(i -> ForwardDiff.Dual(hh[i], ntuple(j -> Float64(i == j), 3)...), HEIGHT_NC))
    od = SVector{3}(ntuple(i -> ForwardDiff.Dual(o[i], 0.0, 0.0, 0.0), 3)); dd = SVector{3}(ntuple(i -> ForwardDiff.Dual(d[i], 0.0, 0.0, 0.0), 3))
    r = intersect_board(RC.BoardRays(board_hs, hd), od, dd)
    @test norm(ForwardDiff.value.(r[2]) - intersect_board(board_hs, o, d, hh)[2]) < 1e-14
    for j in 1:3
        e = SVector{HEIGHT_NC}(ntuple(i -> Float64(i == j) * 1e-7, HEIGHT_NC))
        fd = (intersect_board(board_hs, o, d, hh + e)[2] - intersect_board(board_hs, o, d, hh - e)[2]) / 2e-7
        @test norm(map(x -> ForwardDiff.partials(x)[j], r[2]) - fd) <= 1e-6 * max(norm(fd), 1e-9)
    end
end

@testitem "board warping: a flat board whose heights are constants of a dual type takes the plane hit" setup = [RCFixtures, HeightScene] begin
    using RadiometricCamera: HEIGHT_NC, intersect_board
    using StaticArrays, LinearAlgebra, ForwardDiff
    # is_warped: a nonzero height, or a seeded one (a nonzero partial); zero heights with zero partials are the plane in any element
    # type (a Jacobian chunk's constant heights)
    D = ForwardDiff.Dual{Nothing, Float64, 3}
    z = zero(SVector{HEIGHT_NC, Float64}); zd = SVector{HEIGHT_NC, D}(z)
    seeded = SVector{HEIGHT_NC, D}(ntuple(i -> D(0.0, ForwardDiff.Partials(ntuple(j -> Float64(i == j), 3))), HEIGHT_NC))
    @test !RC.is_warped(z) && !RC.is_warped(zd) && RC.is_warped(seeded) && RC.is_warped(hh) && RC.is_warped(SVector{HEIGHT_NC, D}(hh))
    # the ray seeded (a chunk's pose slots), the flat heights constant: the plane hit in the height field's numbers, its value the
    # plain plane hit bit for bit, its partials the height field's implicit ones (Core's Newton and implicit step at h = 0) to rounding
    o = SVector(0.1, -0.2, 1.4); d = normalize(SVector(0.05, 0.1, -1.0))
    od = SVector{3, D}(ntuple(i -> D(o[i], ForwardDiff.Partials(ntuple(j -> Float64(i == j), 3))), 3))
    dd = SVector{3, D}(ntuple(i -> D(d[i], ForwardDiff.Partials(ntuple(j -> 0.3 * Float64(i == j), 3))), 3))
    s, x = intersect_board(RC.BoardRays(board_hs, zd), od, dd)
    s0, x0 = intersect_board(board_hs, o, d)
    @test s isa D && eltype(x) === D && ForwardDiff.value(s) == s0 && ForwardDiff.value.(x) == x0
    @test intersect_board(board_hs, od, dd, zd) == (s, x)                       # the per-ray form: the same hit
    sn, xn = RC.VgtkCore.intersection(RC.board_relief(board_hs, zd), od, dd)   # the height field's path at the same flat board
    @test abs(ForwardDiff.value(sn) - s0) < 1e-14 && norm(ForwardDiff.value.(xn) - x0) < 1e-14
    @test all(j -> norm(map(q -> ForwardDiff.partials(q)[j], x) - map(q -> ForwardDiff.partials(q)[j], xn)) < 1e-12, 1:3)
    @test norm(collect(ForwardDiff.partials(s)) - collect(ForwardDiff.partials(sn))) < 1e-12
    # plain rays on constant-dual flat heights: the same types as the height field's hit (one type on both branches)
    sp, xp = intersect_board(RC.BoardRays(board_hs, zd), o, d)
    @test sp isa D && eltype(xp) === D && ForwardDiff.value(sp) == s0
end

@testitem "radiance record: a ray's hit and normal; a chunk's hit on the recorded hit's tangent plane" setup = [RCFixtures, HeightScene] begin
    using RadiometricCamera: HEIGHT_NC, intersect_board
    using StaticArrays, LinearAlgebra, ForwardDiff
    # The record finds a ray's hit once (the height field's root) and keeps the board point and the surface's normal there; a Jacobian
    # chunk meets the recorded hit's tangent plane with its dual ray: the same value, Core's implicit-function partials — on the
    # warped board and on the plane.
    for h in (hh, zero(hh))
        rays = RC.BoardRays(board_hs, h)
        o = SVector(0.1, -0.2, 1.4); d = normalize(SVector(0.05, 0.1, -1.0))
        rec = [zero(RC.RadianceJac{Float64})]
        s, x, n = RC._ray_hit(nothing, RC.RadianceRecord(rec, 0), 1, rays, o, d)
        s0, x0 = intersect_board(rays, o, d)
        @test abs(s - s0) < 1e-14 && norm(x - x0) < 1e-14 && dot(n, d) ≈ 1
        @test norm(cross(normalize(n), VgtkCore.normal(rays.relief, x0))) < 1e-9          # Core's normal of the surface there (either sense)
        # the record's entry as a chunk reads it, [L | ∂L/∂ℓ | 𝐱 | 𝐧], and the dual ray: origin and direction seeded
        rec[1] = RC.RadianceJac{Float64}(ntuple(q -> (j = (q - 1) ÷ 3; c = (q - 1) % 3 + 1; j == RC.RAD_NL + 1 ? x[c] : j == RC.RAD_NL + 2 ? n[c] : 0.0), 3 * (RC.RAD_NL + 3)))
        D = ForwardDiff.Dual{Nothing, Float64, 6}
        seed(v, k0) = SVector{3, D}(ntuple(i -> D(v[i], ForwardDiff.Partials(ntuple(j -> Float64(j == k0 + i), 6))), 3))
        od, dd = seed(o, 0), seed(d, 3)
        st, xt, _ = RC._ray_hit(nothing, RC.RadianceRays(rec, 0), 1, rays, od, dd)
        sc, xc = intersect_board(rays, od, dd)                    # Core's hit of the dual ray (the plane's, or the Newton's with its implicit step)
        @test abs(ForwardDiff.value(st) - s0) < 1e-13 && norm(ForwardDiff.value.(xt) - x0) < 1e-13
        @test all(j -> norm(map(q -> ForwardDiff.partials(q)[j], xt) - map(q -> ForwardDiff.partials(q)[j], xc)) < 1e-10, 1:6)
        @test norm(collect(ForwardDiff.partials(st)) - collect(ForwardDiff.partials(sc))) < 1e-10
        # a ray the record did not reach (zeros): not finite, so the flux skips it; the parts' hit likewise
        blank = RC.RadianceRays([zero(RC.RadianceJac{Float64})], 0)
        @test !isfinite(ForwardDiff.value(first(RC._ray_hit(nothing, blank, 1, rays, od, dd)))) && !isfinite(first(RC._parts_hit(blank, 1, rays, o, d)))
        sp, xp = RC._parts_hit(RC.RadianceRays(rec, 0), 1, rays, o, d)
        @test abs(sp - s0) < 1e-13 && xp == x && RC._parts_hit(nothing, 1, rays, o, d) == (s0, x0)
    end
end
