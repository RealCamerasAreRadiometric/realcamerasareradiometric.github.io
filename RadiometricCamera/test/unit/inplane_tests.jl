# The in-plane print error of the board (`BoardSurface.inplane`, `inplane_calibration`: Core's gauge-free factors `Aspect ∘ Skew` of an
# affine print error under a jointly estimated pose — the design stretched along one page axis against the other and sheared): the map
# and its chart against Core's posed target, the render (the texture sampled at the DESIGN point a physical point shows), its dual-number
# Jacobian columns in the grid-tile route and the per-view route, LM recovery of a known (a, u) from the as-designed start, CPU/GPU parity.

# The truth and the two-view scene of the blob fixture the items share: a 3 % stretch of x against y and a 2 % shear — 3 cm at the edge
# of the 2 m board, ≈ 0.5 px at the 32-px camera (the height tests' scene, with the in-plane error in place of the saddle).
@testsetup module InplaneScene
    using RadiometricCamera: BoardSurface, compute_mask, Focus, FocusSpline, grid_model, Material, PrintedTarget, render_all,
                             RenderSettings, Scene, sphere_radius, with_surface, 𝔟
    using RadiometricCamera, StaticArrays, LinearAlgebra, ForwardDiff
    using VisualGeometryToolkitCore: exp_so3
    using VisualRadiometryToolkitCore: TensorProductVignetting, with_control_points, EnvironmentLight, SphericalHarmonicMap
    import VisualGeometryToolkitCore as VgtkCore
    const RC = RadiometricCamera
    include(joinpath(@__DIR__, "fixture_board.jl"))
    include(joinpath(@__DIR__, "fixture_camera.jl"))
    export θ_true, tex_i, board_i, Wi, Hi, scene_i, scene_iθ, glens_i, lens_param_i, focus_i, poses_i, camera_i, camera_param_i, vig_i, light_i, mat_i, st_i, masks_i, blank_i, targets_i, targets_param_i

    θ_true = SVector(0.03, 0.02)                                                           # (a, u): x by e^0.03 against y, shear 2 %
    tex_i = fixture_board()
    board_i = PrintedTarget([-1.0, -1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], tex_i; facing = [0.0, 0.0, 1.5])
    Wi = Hi = 32
    scene_i = Scene(board_i, Wi, Hi)
    scene_iθ(θ) = with_surface(scene_i, BoardSurface(inplane = θ))                        # the print error is the scene's surface parameter
    lens_param_i = cam(VgtkCore.Lens(VgtkCore.DoubleSphere(0.25, 0.4)), 41.216, (16.32, 15.36), Wi, Hi)
    glens_i = grid_model(lens_param_i, 10, 10)
    focus_i = Focus(FocusSpline(VgtkCore.grid_nodes(3, VgtkCore.Rect(0, 0, Wi, Hi))), [1 / (1.5 + 0.1 * sin(3i)) for i in 1:9])
    poses_i = [VgtkCore.EuclideanMap(SMatrix{3, 3}(exp_so3(SVector(π, 0.0, 0.0)) * exp_so3(SVector(0.15 * cos(k), 0.12 * sin(k), 0.1 * k))), SVector{3}(SVector(0.2 * cos(k), 0.2 * sin(k), 1.2 + 0.1 * k)); check = false) for k in 1:2]
    vig_i = let v = TensorProductVignetting((4, 4), VgtkCore.Rect(0, 0, Wi, Hi); degree = 2); gr = range(-0.5, 0.5, length = 4); with_control_points(v, [1.0 - 0.3 * (gr[i]^2 + gr[j]^2) for i in 1:4, j in 1:4]) end
    camera_i = RadiometricCameraModel(glens_i, focus_i, 0.0386, vig_i)                    # the grid camera
    camera_param_i = RadiometricCameraModel(lens_param_i, focus_i, 0.0386, vig_i)         # the parametric camera it is sampled from
    b_i = 𝔟(board_i); a_i = zeros(3, 9); a_i[:, 1] .= (2.0, 1.9, 1.8); a_i[:, 3] .= (0.4, 0.3, 0.2)
    light_i = EnvironmentLight(SphericalHarmonicMap(a_i, 2), b_i, sphere_radius(b_i, [p.t for p in poses_i]))
    mat_i = Material(; roughness_ink = 0.12, roughness_paper = 0.35, refractive_index_ink = 1.65, refractive_index_paper = 1.33, diffuse_weight = 0.6)
    st_i = RenderSettings(; n_𝐩 = 2, n_ρ = 2, n_φ = 6, N = 4)
    masks_i = [compute_mask(camera_i, p, scene_i, st_i; δ = 0.03, backend = RadiometricCamera.KernelAbstractions.CPU()) for p in poses_i]                 # the mask is the plane's, padded past the 3 % stretch
    blank_i = [zeros(Hi, Wi, 3) for _ in 1:2]
    targets_i = render_all(CalibrationState(scene_iθ(θ_true), st_i, camera_i, poses_i, mat_i, light_i, blank_i; masks = masks_i))
    targets_param_i = render_all(CalibrationState(scene_iθ(θ_true), st_i, camera_param_i, poses_i, mat_i, light_i, blank_i; masks = masks_i))
end

@testitem "in-plane print error: the map, Core's posed target, the render" setup = [RCFixtures, InplaneScene] begin
    using RadiometricCamera: BoardSurface, design_chart, HEIGHT_NC, height_on, inplane_calibration, INPLANE_NC, inplane_on,
                             intersect_board, physical_chart, posed_target, PrintedTarget, render_all, with_edge, with_h,
                             with_inplane
    import VisualGeometryToolkitCore as VgtkCore
    using StaticArrays, LinearAlgebra, Statistics, ForwardDiff

    # ---------------------------------------------------------------------------------------------------------------
    # 1. the map: Core's Aspect ∘ Skew in the placed chart (the page in widths, origin top-left, y up), the identity at zero, its inverse
    @test INPLANE_NC == 2
    θ = SVector(0.01, -0.004)
    wp = PrintedTarget([-1.0, -0.7, 0.0], [2.0, 0.0, 0.0], [0.0, 1.4, 0.0], rand(4, 4, 3); facing = [0.0, 0.0, 1.5])   # a 10 : 7 page
    cal = inplane_calibration(θ)
    @test VgtkCore.members(cal) == (VgtkCore.Aspect(0.01), VgtkCore.Skew(-0.004))
    for 𝐛 in (SVector(0.3, 0.6), SVector(0.0, 0.0), SVector(1.0, 0.7), SVector(0.97, 0.03))
        c = RC.placed_chart(wp, 𝐛)
        @test c == SVector(𝐛[1], -0.7 * 𝐛[2])
        @test RC.page_fraction(wp, c) ≈ 𝐛 atol = 1e-15
        # the physical print of the design point: Core's factors by hand — the shear first (the inner factor), then the stretch
        c_phys = SVector(exp(0.01) * (c[1] - 0.004 * c[2]), exp(-0.01) * c[2])
        @test cal(c) ≈ c_phys atol = 1e-15
        @test physical_chart(wp, θ, 𝐛) ≈ RC.page_fraction(wp, c_phys) atol = 1e-15
        @test design_chart(wp, θ, physical_chart(wp, θ, 𝐛)) ≈ 𝐛 atol = 1e-14            # the inverse
        @test physical_chart(wp, θ, design_chart(wp, θ, 𝐛)) ≈ 𝐛 atol = 1e-14
        @test design_chart(wp, zero(θ), 𝐛) === 𝐛 && physical_chart(wp, zero(θ), 𝐛) === 𝐛   # off: the point itself, untouched
    end
    @test !inplane_on(zero(SVector{2, Float64})) && inplane_on(θ) && inplane_on(SVector(0.0, 1e-9))
    # a Float32 point stays Float32 (the device path), a dual point a dual of the same kind
    @test design_chart(wp, SVector{2, Float32}(θ), SVector(0.3f0, 0.6f0)) isa SVector{2, Float32}
    # duals seeded in the parameters flow through the map: ∂/∂a and ∂/∂u of the design chart against central differences
    let 𝐛 = SVector(0.8, 0.25), ε = 1e-6
        θd = SVector(ForwardDiff.Dual(θ[1], 1.0, 0.0), ForwardDiff.Dual(θ[2], 0.0, 1.0))
        @test inplane_on(SVector(ForwardDiff.Dual(0.0, 1.0, 0.0), ForwardDiff.Dual(0.0, 0.0, 1.0)))     # a seeded identity is on (its partials)
        d = design_chart(wp, θd, 𝐛)
        ga = (design_chart(wp, θ + SVector(ε, 0.0), 𝐛) - design_chart(wp, θ - SVector(ε, 0.0), 𝐛)) / 2ε
        gu = (design_chart(wp, θ + SVector(0.0, ε), 𝐛) - design_chart(wp, θ - SVector(0.0, ε), 𝐛)) / 2ε
        @test norm(ForwardDiff.partials.(d, 1) - ga) < 1e-8 && norm(ForwardDiff.partials.(d, 2) - gu) < 1e-8
    end

    # ---------------------------------------------------------------------------------------------------------------
    # 2. bookkeeping: the surface carries the two parameters; the modifiers, the keyword constructor and the precision conversion keep them
    sf0 = BoardSurface()
    @test !inplane_on(sf0) && iszero(sf0.inplane) && length(sf0.inplane) == INPLANE_NC
    sfθ = with_inplane(sf0, θ)
    @test inplane_on(sfθ) && sfθ.inplane == θ && iszero(sfθ.h) && iszero(sfθ.edge) && !height_on(sfθ)
    @test with_h(sfθ, sfθ.h).inplane == θ && with_edge(sfθ, [0.02, 0.03]).inplane == θ && with_edge(sfθ, [0.02, 0.03]).edge == SVector(0.02, 0.03)
    @test VgtkCore.convert_eltype(Float32, sfθ).inplane == SVector{2, Float32}(θ) && VgtkCore.convert_eltype(Float32, sfθ) isa BoardSurface{Float32}
    @test BoardSurface(inplane = θ) == sfθ && BoardSurface(inplane = [0.01, -0.004]).inplane == θ
    @test BoardSurface(h = zeros(HEIGHT_NC), edge = [0.02, 0.03], inplane = θ).inplane == θ
    @test BoardSurface{Float32}().inplane == zero(SVector{2, Float32})
    @test isbitstype(typeof(sfθ)) && isbitstype(typeof(VgtkCore.convert_eltype(Float32, sfθ)))

    # ---------------------------------------------------------------------------------------------------------------
    # 3. Core's posed target: a flat board with the error is the page's target with the calibration Aspect ∘ Skew, placed; with the
    #    warping too the record is HeightField ∘ Aspect ∘ Skew (Core's canonical order); its surface is the board's (the in-plane
    #    error relabels the print, it moves no surface point)
    @test posed_target(board_i, sf0) === board_i.target
    ptθ = posed_target(board_i, sfθ)
    @test VgtkCore.model(ptθ) !== VgtkCore.model(board_i.target)
    calθ = VgtkCore.calibration(VgtkCore.model(ptθ))
    @test VgtkCore.members(calθ) == (VgtkCore.Aspect(θ[1]), VgtkCore.Skew(θ[2]))
    @test VgtkCore.pose(ptθ) == VgtkCore.pose(board_i.target)
    # the placed chart of the board (a 2 × 2 m square page, origin (−1, −1), y up): a design point's physical print through Core's
    # calibration equals ours through `physical_chart`, in the page's own chart
    wp1 = board_i
    for 𝐛 in (SVector(0.3, 0.6), SVector(0.9, 0.1))
        @test calθ(RC.placed_chart(wp1, 𝐛)) ≈ RC.placed_chart(wp1, physical_chart(wp1, θ, 𝐛)) atol = 1e-15
    end
    o = SVector(0.3, -0.2, 1.4); d = normalize(SVector(-0.4, 0.3, -1.0))
    s0, 𝐱0 = intersect_board(board_i, o, d)
    let sf = VgtkCore.surface(ptθ)
        sc, 𝐱c = VgtkCore.intersection(sf, o, d)
        @test abs(sc - s0) < 1e-12 && norm(𝐱c - 𝐱0) < 1e-12                               # the flat board's hit, whatever the print error
    end
    hh = SVector{HEIGHT_NC}([(q = 𝐛 .- 0.5; 0.08 * q[1]^2 - 0.04 * q[1] * q[2] + 0.06 * q[2]^2) for 𝐛 in RC.HEIGHT_NODES])
    sfhθ = with_inplane(with_h(sf0, hh), θ)
    pthθ = posed_target(board_i, sfhθ)
    ms = VgtkCore.members(VgtkCore.calibration(VgtkCore.model(pthθ)))
    @test length(ms) == 3 && ms[1] isa VgtkCore.HeightField && ms[2] == VgtkCore.Aspect(θ[1]) && ms[3] == VgtkCore.Skew(θ[2])
    @test VgtkCore.parameter_vector(ms[1]) == VgtkCore.parameter_vector(VgtkCore.members(VgtkCore.calibration(VgtkCore.model(posed_target(board_i, with_h(sf0, hh)))))[1])   # the same field as without the error
    let sf = VgtkCore.surface(pthθ), (sh, 𝐱h) = intersect_board(board_i, o, d, hh)
        @test sf isa VgtkCore.MappedSurface
        sc, 𝐱c = VgtkCore.intersection(sf, o, d)
        @test abs(sc - sh) < 1e-12 && norm(𝐱c - 𝐱h) < 1e-12                                 # the warped board's hit, whatever the print error
    end

    # ---------------------------------------------------------------------------------------------------------------
    # 4. the render: the texture is sampled at the DESIGN point the physical point shows, so the error moves the rendered pattern; the
    #    as-designed render is the flat board's bit for bit
    r_flat = render_all(CalibrationState(scene_i, st_i, camera_i, poses_i, mat_i, light_i, blank_i; masks = masks_i))
    r_zero = render_all(CalibrationState(scene_iθ(zero(θ_true)), st_i, camera_i, poses_i, mat_i, light_i, blank_i; masks = masks_i))
    @test all(r_zero[k] == r_flat[k] for k in 1:2)
    @test maximum(maximum(abs, targets_i[k] .- r_flat[k]) for k in 1:2) > 0.02 * maximum(maximum, r_flat)
    # a pixel's radiance through Φ_pixel: with the error, the same as the flat board's at the design point's physical place — checked
    # through the texture lookup itself: ρ at the physical chart 𝐛 under θ is the design's ρ at design_chart(𝐛)
    let bd = board_i
        for 𝐛 in (SVector(0.3, 0.6), SVector(0.72, 0.41))
            ρθ = RC._albedo(bd, sfθ, design_chart(bd, θ, 𝐛))
            @test ρθ == RC._albedo(bd, sf0, design_chart(bd, θ, 𝐛))                   # the lookup itself is the design's
            @test ρθ != RC._albedo(bd, sf0, 𝐛) || norm(design_chart(bd, θ, 𝐛) - 𝐛) < 1e-3
        end
    end
end

@testitem "in-plane print error: Jacobian columns (grid tiles, per view) and LM recovery" setup = [RCFixtures, InplaneScene] begin
    using RadiometricCamera: board_surface, build_problem, build_problem_grid, cost, extract_variables, extract_variables_grid,
                             HEIGHT_NC, HEIGHT_NFREE, INPLANE_NC, PerViewResidual, radiometric_target, read_back!, run_lm!,
                             run_lm_grid!
    import VisualGeometryToolkitCore as VgtkCore
    using StaticArrays, LinearAlgebra, Statistics, ForwardDiff

    # ---------------------------------------------------------------------------------------------------------------
    # 1. the grid-tile route: the material block carries (a, u) as two AD columns after the 9 BRDF parameters (and after the free height
    #    parameters when the warping is free); the columns match central differences of the tile residual through `update`; the pose
    #    columns still match (the duals flow through the surface and the texture lookup)
    s_i = CalibrationState(scene_iθ(0.5 * θ_true), st_i, camera_i, poses_i, mat_i, light_i, targets_i; masks = masks_i)
    prob_i, idx_i = build_problem_grid(s_i; patch = (5, 5), inplane = true)
    tiles_i = RC.tile_costs(prob_i)
    r_i = tiles_i[argmax([length(t.pix) for t in tiles_i])]
    vars_i = collect(RC.NLLSsolver.getvars(r_i, prob_i.variables))
    mv = vars_i[8]
    @test RC.NLLSsolver.nvars(mv) == 9 + INPLANE_NC && RC.n_inplane(mv) == INPLANE_NC && RC.n_height(mv) == 0
    res_i, J_i = RC.NLLSsolver.computeresjac(RC.static(1023), r_i, vars_i...)
    c0 = 6 + 200 + 9 + 1 + 9                                                           # pose | grid | focus | r | material 8 + μ; then the 2 in-plane columns
    @test size(J_i) == (3 * length(r_i.pix), c0 + INPLANE_NC + 15 + 27)                # (15: the hard pin eliminates the centre's vignetting direction)
    @test res_i ≈ RC.NLLSsolver.computeresidual(r_i, vars_i...) atol = 1e-10
    fdi(j, u) = (RC.NLLSsolver.computeresidual(r_i, updated_vars(vars_i, j, u)...) - RC.NLLSsolver.computeresidual(r_i, updated_vars(vars_i, j, -u)...)) / (2 * norm(u))
    for q in 1:INPLANE_NC                                                              # the two in-plane columns vs central differences through `update`
        fd = fdi(8, unit_increment(9 + INPLANE_NC, 9 + q, 1e-6))
        @test norm(fd) > 0
        @test norm(J_i[:, c0 + q] .- fd) / norm(fd) < 5e-3
    end
    fdμ = fdi(8, unit_increment(9 + INPLANE_NC, 9, 1e-6))                                             # μ column unchanged by the extra block
    @test norm(J_i[:, c0] .- fdμ) / norm(fdμ) < 1e-5
    for j in 1:6                                                                       # pose columns vs central differences
        fd = fdi(1, unit_increment(6, j, 1e-7))
        @test norm(J_i[:, j] .- fd) / norm(fd) < 5e-3
    end
    # `update` adds to the parameters (additive, no gauge); the material and the heights are untouched
    v_up = RC.NLLSsolver.update(mv, unit_increment(9 + INPLANE_NC, 9 + 1, 1e-3))
    @test v_up.surface.inplane == mv.surface.inplane + SVector(1e-3, 0.0) && v_up.mat == mv.mat && v_up.surface.h == mv.surface.h && v_up.valid
    v_up2 = RC.NLLSsolver.update(mv, unit_increment(9 + INPLANE_NC, 9 + 2, -2e-3))
    @test v_up2.surface.inplane == mv.surface.inplane + SVector(0.0, -2e-3)
    @test !RC.NLLSsolver.update(mv, unit_increment(9 + INPLANE_NC, 9 + 1, NaN)).valid              # a non-finite step is an invalid block (the LM rejects it)
    # with the warping free too, the in-plane columns follow the height columns
    prob_hi, _ = build_problem_grid(s_i; patch = (5, 5), height_degree = 2, inplane = true)
    tiles_hi = RC.tile_costs(prob_hi)
    r_hi = tiles_hi[findfirst(t -> t.k == r_i.k && t.P0 == r_i.P0 && length(t.pix) == length(r_i.pix), tiles_hi)]
    vars_hi = collect(RC.NLLSsolver.getvars(r_hi, prob_hi.variables))
    @test RC.NLLSsolver.nvars(vars_hi[8]) == 9 + HEIGHT_NFREE + INPLANE_NC && RC.n_height(vars_hi[8]) == HEIGHT_NC && RC.n_inplane(vars_hi[8]) == INPLANE_NC
    _, J_hi = RC.NLLSsolver.computeresjac(RC.static(1023), r_hi, vars_hi...)
    @test size(J_hi, 2) == c0 + HEIGHT_NFREE + INPLANE_NC + 15 + 27
    for q in 1:INPLANE_NC
        @test norm(J_hi[:, c0 + HEIGHT_NFREE + q] .- J_i[:, c0 + q]) / norm(J_i[:, c0 + q]) < 1e-10   # the same columns, shifted past the heights
    end
    fdh(j, u) = (RC.NLLSsolver.computeresidual(r_hi, updated_vars(vars_hi, j, u)...) - RC.NLLSsolver.computeresidual(r_hi, updated_vars(vars_hi, j, -u)...)) / (2 * norm(u))
    for q in (1, 7, HEIGHT_NFREE)                                                      # the height columns beside the in-plane block, against central differences
        fd = fdh(8, unit_increment(9 + HEIGHT_NFREE + INPLANE_NC, 9 + q, 1e-5))
        @test norm(fd) > 0
        @test norm(J_hi[:, c0 + q] .- fd) / norm(fd) < 5e-3
    end
    for q in 1:INPLANE_NC                                                              # and the in-plane columns there, against central differences
        fd = fdh(8, unit_increment(9 + HEIGHT_NFREE + INPLANE_NC, 9 + HEIGHT_NFREE + q, 1e-6))
        @test norm(J_hi[:, c0 + HEIGHT_NFREE + q] .- fd) / norm(fd) < 5e-3
    end
    v_hup = RC.NLLSsolver.update(vars_hi[8], unit_increment(9 + HEIGHT_NFREE + INPLANE_NC, 9 + HEIGHT_NFREE + 2, 1e-3))
    @test v_hup.surface.inplane == vars_hi[8].surface.inplane + SVector(0.0, 1e-3) && v_hup.surface.h == vars_hi[8].surface.h
    # without the flag nothing changes: 9 material columns, the former width
    prob_0, _ = build_problem_grid(s_i; patch = (5, 5))
    @test RC.NLLSsolver.nvars(collect(RC.NLLSsolver.getvars(RC.tile_costs(prob_0)[1], prob_0.variables))[8]) == 9

    # ---------------------------------------------------------------------------------------------------------------
    # 2. the per-view route (a parametric lens): `build_problem(; inplane)` — the same two columns after the material block
    s_p = CalibrationState(scene_iθ(0.5 * θ_true), st_i, camera_param_i, poses_i, mat_i, light_i, targets_param_i; masks = masks_i)
    prob_p, idx_p = build_problem(s_p; inplane = true)
    mvp = prob_p.variables[idx_p.material]
    @test Int(RC.NLLSsolver.nvars(mvp)) == 9 + INPLANE_NC && RC.n_inplane(mvp) == INPLANE_NC && RC.n_height(mvp) == 0
    # a hold is `fixed`, never a boolean (review finding 8): `:inplane` / `:height` in `fixed` leave the material block without their
    # columns, on both routes, as the model without them; the boolean hold keywords are gone
    nmat(prob, i) = Int(RC.NLLSsolver.nvars(prob.variables[i]))
    @test nmat(first(build_problem(s_p; inplane = true, fixed = (:inplane,))), idx_p.material) == 9
    @test nmat(first(build_problem(s_p; height_degree = 1, fixed = (:height,))), idx_p.material) == 9
    @test nmat(first(build_problem(s_p; height_degree = 1, inplane = true)), idx_p.material) == 9 + HEIGHT_NFREE + INPLANE_NC
    # the printed edge is never a parameter: no hold of the LM ("edge" in the options' `fixed` refused, RC 0.11.0), so the cases build
    # their problem without it (review round 2, finding 1: every parametric run must build), and `:edge` in a builder's `fixed` names nothing
    @test_throws ArgumentError build_problem(s_p; fixed = (:edge,))
    # (the package's own options, the shapes of the paper's parametric cases: a printed edge, the warping free or not, the print error
    # free or held — the configs of record are the paper project's, experiments/test reads them: owner D15)
    for d in (Dict("edge" => Dict("sdf" => "distance_map.json"), "warping" => Dict("degree" => 1, "inplane" => "aspect_skew")),
              Dict("edge" => Dict("sdf" => "distance_map.json"), "warping" => Dict("degree" => 1, "inplane" => "aspect_skew"), "fixed" => ["inplane"]),
              Dict("edge" => Dict("sdf" => "distance_map.json"), "warping" => Dict("inplane" => "aspect_skew")),
              Dict("edge" => Dict("sdf" => "distance_map.json")))
        o = RC.pipeline_options(d)
        @test !isempty(o.edge.sdf) && !("edge" in o.fixed)
        kw = RC._refinement_lm(o, RC.PerViewRoute()).kw
        @test !(:edge in kw.fixed)
        pk, ik = build_problem(s_p; kw...)
        @test nmat(pk, ik.material) == 9 + (kw.height_degree > 0 ? HEIGHT_NFREE : 0) + (kw.inplane && !(:inplane in kw.fixed) ? INPLANE_NC : 0)
    end
    for kw in ((; inplane_free = true), (; height_free = true), (; tie_f = true), (; callback = (a...) -> (a[1], 0)))
        @test_throws MethodError run_lm!(s_p; maxiters = 1, kw...)
    end
    pg_held, ig_held = build_problem_grid(s_i; patch = (5, 5), inplane = true, height_degree = 2, fixed = (:inplane, :height))
    @test RC.NLLSsolver.nvars(collect(RC.NLLSsolver.getvars(RC.tile_costs(pg_held)[1], pg_held.variables))[8]) == 9
    # the holds by name: a config's strings are the Symbols (review round 2, finding 8: the grid route compared them raw)
    pg_str, _ = build_problem_grid(s_i; patch = (5, 5), inplane = true, height_degree = 2, fixed = ("inplane", "height"))
    @test RC.NLLSsolver.nvars(collect(RC.NLLSsolver.getvars(RC.tile_costs(pg_str)[1], pg_str.variables))[8]) == 9
    pg_pose, ip_pose = build_problem_grid(s_i; patch = (5, 5), fixed = ("poses",))
    @test !any(ip_pose.free[ip_pose.poses])                                       # "poses" holds the poses (and so no origin gauge)
    for kw in ((; callback = (a...) -> (a[1], 0)), (; tie_f = true))
        @test_throws MethodError run_lm_grid!(s_i; patch = (5, 5), maxiters = 1, kw...)
    end
    # the two routes' one per-iteration callback and one read-back (review finding 7: the grid route duplicated both)
    srcs = [read(joinpath(pkgdir(RadiometricCamera), "src", "calibration", "lm", f), String) for f in ("per_view.jl", "per_tile.jl")]
    @test all(t -> occursin("_lm_callback(s, idx, extract_variables", t), srcs) && occursin("read_back!(s, problem, idx, extract_variables_grid)", srcs[2])
    @test !any(t -> occursin("problem_.varnext", t), [srcs[2]]) && count("problem_.varnext", srcs[1]) == 2   # the varnext read: once, in _lm_callback
    res_p = get(prob_p.problem.costs, PerViewResidual{9})[1]
    vars_p = RC.NLLSsolver.getvars(res_p, prob_p.variables)
    r0 = RC.NLLSsolver.computeresidual(res_p, vars_p...)
    @test maximum(abs, r0) > 1e-6                                                      # half the error misses the truth
    rj, J_p = RC.NLLSsolver.computeresjac(RC.static(127), res_p, vars_p...)
    ND = VgtkCore.tangent_size(prob_p.variables[idx_p.distortion])
    cp = 6 + ND + 9 + 1 + 9                                                            # pose | lens | focus | r | material 8 + μ; then the 2 in-plane columns
    @test size(J_p) == (length(r0), cp + INPLANE_NC + 15 + 27)                         # | vig 15 (the pin's direction eliminated) | light 3 × 9
    @test isapprox(rj, r0; atol = 1e-10)
    pv, dv, fv, av, mvv, vv, lv = vars_p
    for q in 1:INPLANE_NC
        setter(h) = (pv, dv, fv, av, RC.NLLSsolver.update(mvv, unit_increment(9 + INPLANE_NC, 9 + q, h)), vv, lv)
        fd = (RC.NLLSsolver.computeresidual(res_p, setter(1e-6)...) - RC.NLLSsolver.computeresidual(res_p, setter(-1e-6)...)) / 2e-6
        @test norm(fd) > 0
        @test norm(J_p[:, cp + q] - fd) / norm(fd) < 5e-3
    end
    for (col, setter) in ((2, h -> (RC.NLLSsolver.update(pv, unit_increment(6, 2, h)), dv, fv, av, mvv, vv, lv)),                     # the other blocks' columns, shifted by the in-plane block
                          (cp, h -> (pv, dv, fv, av, RC.NLLSsolver.update(mvv, unit_increment(9 + INPLANE_NC, 9, h)), vv, lv)),         # μ
                          (cp + INPLANE_NC + 2, h -> (pv, dv, fv, av, mvv, RC.NLLSsolver.update(vv, unit_increment(15, 2, h)), lv)),   # vignetting free direction 2
                          (cp + INPLANE_NC + 15 + 4, h -> (pv, dv, fv, av, mvv, vv, RC.NLLSsolver.update(lv, unit_increment(27, 4, h)))))   # lighting a[1, 2]
        fd = (RC.NLLSsolver.computeresidual(res_p, setter(1e-6)...) - RC.NLLSsolver.computeresidual(res_p, setter(-1e-6)...)) / 2e-6
        @test norm(J_p[:, col] - fd) / norm(fd) < 2e-3
    end
    prob_p0, idx_p0 = build_problem(s_p)
    @test Int(RC.NLLSsolver.nvars(prob_p0.variables[idx_p0.material])) == 9

    # ---------------------------------------------------------------------------------------------------------------
    # 3. LM recovery of (a, u) from the as-designed start, everything else at the truth: the grid route, then the per-view route
    s_r = CalibrationState(scene_i, st_i, camera_i, poses_i, mat_i, light_i, targets_i; masks = masks_i)
    c0r = cost(s_r)[1]
    run_lm_grid!(s_r; patch = (5, 5), maxiters = 25, inplane = true,
                 fixed = (:poses, :grid, :focus, :aperture, :vignetting, :lighting))
    @test cost(s_r)[1] < 1e-3 * c0r
    @test norm(board_surface(s_r).inplane - θ_true) < 0.05 * norm(θ_true)                      # a 3 % stretch and a 2 % shear, recovered to 5 % of themselves
    @test iszero(board_surface(s_r).h)
    s_rp = CalibrationState(scene_i, st_i, camera_param_i, poses_i, mat_i, light_i, targets_param_i; masks = masks_i)
    c0p = cost(s_rp)[1]
    run_lm!(s_rp; maxiters = 25, inplane = true,
            fixed = (:poses, :distortion, :focus, :aperture, :vignetting, :lighting))
    @test cost(s_rp)[1] < 1e-3 * c0p
    @test norm(board_surface(s_rp).inplane - θ_true) < 0.05 * norm(θ_true)
    # the learned error is the target's record: `radiometric_target` carries it, `board_surface` gives it back on the board
    t_r = radiometric_target(s_r)
    @test t_r.inplane == board_surface(s_r).inplane && board_surface(board_i, t_r).inplane == board_surface(s_r).inplane
end

@testitem "in-plane print error (GPU)" tags = [:gpu] setup = [RCFixtures, RCGPU, InplaneScene] begin
    using RadiometricCamera: build_problem, build_problem_grid, PerViewResidual, INPLANE_NC
    using StaticArrays, LinearAlgebra, Statistics, ForwardDiff
    import CUDA

    # ---------------------------------------------------------------------------------------------------------------
    # 4. device path: CPU/GPU parity of the in-plane columns in the grid-tile route, batched == unbatched; the per-view device Jacobian too
    require_gpu()                                  # errors with the reason when the GPU the runner chose is not usable here
    if RC_GPU_TESTS
        scene_g = scene_iθ(0.5 * θ_true)
        s_g = CalibrationState(scene_g, st_i, camera_i, poses_i, mat_i, light_i, targets_i; masks = masks_i, backend = CUDA.CUDABackend())
        prob_g, idx_g = build_problem_grid(s_g; patch = (5, 5), inplane = true)
        tiles_g = RC.tile_costs(prob_g)
        rg = tiles_g[argmax([length(t.pix) for t in tiles_g])]
        s_c = CalibrationState(scene_g, st_i, camera_i, poses_i, mat_i, light_i, targets_i; masks = masks_i)
        prob_c, _ = build_problem_grid(s_c; patch = (5, 5), inplane = true)
        tiles_c = RC.tile_costs(prob_c)
        rc = tiles_c[findfirst(t -> t.k == rg.k && t.P0 == rg.P0 && length(t.pix) == length(rg.pix), tiles_c)]
        varsg = RC.NLLSsolver.getvars(rg, prob_g.variables)
        rjg, Jg = RC.NLLSsolver.computeresjac(RC.static(1023), rg, varsg...)
        rjc, Jc = RC.NLLSsolver.computeresjac(RC.static(1023), rc, RC.NLLSsolver.getvars(rc, prob_c.variables)...)
        @test maximum(abs, rjg - rjc) < 2e-3
        c0 = 6 + 200 + 9 + 1 + 9
        for q in 1:INPLANE_NC
            @test norm(Jc[:, c0 + q] .- Jg[:, c0 + q]) / norm(Jc[:, c0 + q]) < 5e-3                 # Float32 device twin of the in-plane columns
        end
        for j in 1:6
            @test norm(Jc[:, j] .- Jg[:, j]) / norm(Jc[:, j]) < 3e-2                                # the pose columns through the texture lookup
        end
        RC._register_grid_problem!(s_g, prob_g, idx_g)
        rjg_b, Jg_b = RC.NLLSsolver.computeresjac(RC.static(1023), rg, varsg...)
        @test device_parity(rjg_b, rjg) && device_parity(Jg_b, Jg)
        RC._unregister_grid_problem!(s_g)
        # the per-view route on the device (a parametric lens)
        s_pg = CalibrationState(scene_g, st_i, camera_param_i, poses_i, mat_i, light_i, targets_param_i; masks = masks_i, backend = CUDA.CUDABackend())
        s_pc = CalibrationState(scene_g, st_i, camera_param_i, poses_i, mat_i, light_i, targets_param_i; masks = masks_i)
        prob_pg, _ = build_problem(s_pg; inplane = true); prob_pc, idx_pc = build_problem(s_pc; inplane = true)
        res_pg = get(prob_pg.problem.costs, PerViewResidual{9})[1]; res_pc = get(prob_pc.problem.costs, PerViewResidual{9})[1]
        rpg, Jpg = RC.NLLSsolver.computeresjac(RC.static(127), res_pg, RC.NLLSsolver.getvars(res_pg, prob_pg.variables)...)
        rpc, Jpc = RC.NLLSsolver.computeresjac(RC.static(127), res_pc, RC.NLLSsolver.getvars(res_pc, prob_pc.variables)...)
        @test maximum(abs, rpg - rpc) < 2e-3
        cp = 6 + Int(VgtkCore.tangent_size(prob_pc.variables[idx_pc.distortion])) + 9 + 1 + 9
        for q in 1:INPLANE_NC
            @test norm(Jpc[:, cp + q] .- Jpg[:, cp + q]) / norm(Jpc[:, cp + q]) < 5e-3
        end
    end
end
