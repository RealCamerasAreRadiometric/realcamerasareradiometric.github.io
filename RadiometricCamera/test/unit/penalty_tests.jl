# The smoothness penalties of the spline regressions and their weights (2026-11-04; the operator, m̄, the GCV curve and its search are
# Core's and Estimation's since 2026-11-11, tested there): the vignetting solve's Δ³ penalty (fixed and GCV weight), the grid smoothness `pspline`
# and its GCV weight on the linearization at the start, the boundary order option (order 3: Core's BoundaryRule(m, region; order))
# and the options' new keys. Each item is named "penalties: …".

@testitem "penalties: the vignetting solve's Δ³ penalty, fixed and by GCV" setup = [RCFixtures] begin
    using RadiometricCamera: all_parts, compute_mask, Focus, FocusSpline, Material, TexturedTarget, render_view, RenderSettings,
                             update_radiometry!, update_vignetting!, sphere_radius, with_vignetting, 𝔟
    using StaticArrays, LinearAlgebra, Random
    using VisualGeometryToolkitCore: exp_so3
    using VisualGeometryToolkitCore: center
    using VisualRadiometryToolkitCore: TensorProductVignetting, with_control_points
    tex = fixture_board()
    board = TexturedTarget([-1.0, -1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], tex; facing = [0.0, 0.0, 1.5])
    W = H = 24
    lens_t = cam(VgtkCore.Lens(VgtkCore.DoubleSphere(0.25, 0.4)), 30.912, (12.24, 11.52), W, H)
    focus_t = Focus(FocusSpline(VgtkCore.grid_nodes(3, VgtkCore.Rect(0, 0, W, H))), [1 / (1.5 + 0.1 * sin(3i)) for i in 1:9])
    poses = [VgtkCore.EuclideanMap(SMatrix{3, 3}(exp_so3(SVector(π, 0.0, 0.0)) * exp_so3(SVector(0.15 * cos(k), 0.12 * sin(k), 0.1 * k))), SVector{3}(SVector(0.3 * cos(k), 0.3 * sin(k), 1.4 + 0.1 * k)); check = false) for k in 1:2]
    vig_t = let v = TensorProductVignetting((6, 6), VgtkCore.Rect(0, 0, W, H); degree = 2); gr = range(-0.5, 0.5, length = 6); with_control_points(v, [1.0 - 0.4 * (gr[i]^2 + gr[j]^2) for i in 1:6, j in 1:6]) end
    camera_t = RadiometricCameraModel(lens_t, focus_t, 0.0386, vig_t)
    camera_0 = with_vignetting(camera_t, TensorProductVignetting((6, 6), VgtkCore.Rect(0, 0, W, H); degree = 2))   # the flat start
    b = 𝔟(board); r_SH = sphere_radius(b, [p.t for p in poses])
    a_t = zeros(3, 9); a_t[:, 1] .= (2.0, 1.9, 1.8); a_t[:, 3] .= (0.4, 0.3, 0.2); a_t[:, 4] .= (0.1, 0.15, 0.2)
    light_t = EnvironmentLight(SphericalHarmonicMap(a_t, 2), b, r_SH)
    mat_t = Material(; roughness_ink = 0.12, roughness_paper = 0.35, refractive_index_ink = 1.65, refractive_index_paper = 1.33, diffuse_weight = 0.4)
    st = RenderSettings(; n_𝐩 = 2, n_ρ = 2, n_φ = 6, N = 4)
    masks = [compute_mask(camera_t, p, board, st; δ = 0.02, backend = RadiometricCamera.KernelAbstractions.CPU()) for p in poses]
    rng = Xoshiro(11)
    targets = [render_view(k, camera_t, poses[k], mat_t, light_t, board, st, masks[k]) .* (1 .+ 0.02 .* randn(rng, H, W, 3)) for k in 1:2]   # noisy photographs
    state() = CalibrationState(board, st, camera_0, poses, mat_t, light_t, targets; masks)
    s0 = state(); parts = all_parts(s0)
    update_vignetting!(s0, parts)
    s1 = state(); update_vignetting!(s1, parts; smoothness_weight = 0.0)
    @test s1.camera.vignetting.P == s0.camera.vignetting.P                                           # weight 0: the unpenalized solve, bit for bit
    # a dominating weight: the field's nodes become Δ³-free (quadratic along each axis); the data fit worsens
    D, m̄ = RC.vignetting_smoothness(s1)
    @test size(D) == (2 * 6 * 3, 36) && m̄ > 0
    s2 = state(); update_vignetting!(s2, parts; smoothness_weight = 1e6)
    @test norm(D * vec(Matrix(s2.camera.vignetting.P))) < 1e-4 * norm(D * vec(Matrix(s0.camera.vignetting.P)))
    @test RC.parts_cost(s2, parts) > RC.parts_cost(s0, parts)
    @test s2.camera.vignetting(center(s2.camera.vignetting)) ≈ 1 atol = 1e-12           # the hard pin holds under the penalty
    # by GCV: β inside the bracket, recorded with the curve's minimum; the solve is the fixed-weight solve at that β
    rec = NamedTuple[]
    s3 = state(); update_vignetting!(s3, parts; smoothness_weight = :gcv, gcv_record = rec)
    @test length(rec) == 1 && isfinite(rec[1].gcv) && 0 < rec[1].dof <= rec[1].n_free && rec[1].rows == 36
    @test rec[1].log10_β_bracket == [-8.0, 4.0] && 1e-8 <= rec[1].β <= 1e4 && !rec[1].at_bracket_end      # Estimation's gcv_select
    s4 = state(); update_vignetting!(s4, parts; smoothness_weight = rec[1].β)
    @test s4.camera.vignetting.P ≈ s3.camera.vignetting.P rtol = 1e-10
    # through the block sweep; the joint solve has no penalty
    rec2 = NamedTuple[]
    s5 = state(); update_radiometry!(s5; parts, light = false, mu = false, vignetting_smoothness = :gcv, gcv_record = rec2)
    @test length(rec2) == 1 && s5.camera.vignetting.P ≈ s3.camera.vignetting.P
    @test_throws ArgumentError update_radiometry!(state(); parts, joint = true, vignetting_smoothness = 0.1)
    @test_throws ArgumentError update_vignetting!(state(), parts; smoothness_weight = -1.0)
    @test_throws ArgumentError update_vignetting!(state(), parts; smoothness_weight = :aic)
    # the boundary order: 2 is the natural rule (Δ²c = 0), 3 the curvature-preserving Δ³c = 0 — Core's BoundaryRule(m, region; order)
    reg = VgtkCore.Rect(0, 0, W, H) ∩ VgtkCore.Ellipse(VgtkCore.Point2(12.0, 12.0), 7.0, 7.0, 0.0)
    sr = CalibrationState(board, st, camera_0, poses, mat_t, light_t, targets; masks, region = reg, boundary_order = 2)
    @test sr.vig_rule.interior == VgtkCore.BoundaryRule(sr.camera.vignetting,reg).interior
    @test RC.smoothness_nodes(sr.camera.vignetting,sr.vig_rule) == sr.vig_rule.interior
    s3r = CalibrationState(board, st, camera_0, poses, mat_t, light_t, targets; masks, region = reg, boundary_order = 3)
    r3 = VgtkCore.BoundaryRule(s3r.camera.vignetting,reg; order = 3)
    @test s3r.vig_rule.order == 3 && s3r.vig_rule.interior == r3.interior == sr.vig_rule.interior
    @test s3r.vig_rule.dependent == r3.dependent && s3r.vig_rule.terms == r3.terms && s3r.vig_rule.terms != sr.vig_rule.terms
    # order 3 extends a quadratic correction exactly where its rule has an axis or a diagonal to read (its coefficients sum to one)
    @test all(tm -> sum(last, tm) ≈ 1, s3r.vig_rule.terms)
    @test_throws ArgumentError RC.vignetting_rule(sr.camera.vignetting,reg; order = 4)    # Core's BoundaryRule refuses the order
end

@testitem "penalties: the grid smoothness pspline and its weight by GCV on the linearization" setup = [RCFixtures, GridTilesScene] begin
    using RadiometricCamera: build_problem_grid
    using LinearAlgebra
    all10 = trues(10, 10)
    free = trues(10, 10); free[1:5, 1:5] .= false; free[7, 8] = false
    pr = RC.grid_smoothness(RC.PSplineSmoothness(3, 0.5), glens_t, 7.0, free)
    @test pr.type === :pspline && pr.rows == [collect(zip(n, c)) for (n, c) in RC.VgtkCore.row_stencils(RC.VgtkCore.difference_rows(free, 3))]
    DtD = zeros(100, 100)
    for row in pr.rows, (a, ca) in row, (b2, cb) in row; DtD[a, b2] += ca * cb; end
    λ = eigvals(Symmetric(DtD)); λp = λ[λ .> 1e-10 * maximum(λ)]
    @test pr.m̄ ≈ sum(λp) / length(λp) rtol = 1e-10
    @test pr.w ≈ sqrt(0.5 * 7.0 / count(free) / pr.m̄)
    @test RC.grid_smoothness_cost(pr, glens_t) == 0.0
    @test length(RC.grid_smoothness(RC.PSplineSmoothness(3, 0.5), glens_t, 7.0, all10).rows) == 2 * 10 * 7
    # in a problem: rows on the present nodes only, zero at the start
    prob_p, _ = build_problem_grid(s; patch = (5, 5), smoothness = pr)
    @test sum(c -> length(c.terms), [c for T in keys(prob_p.problem.costs) if T <: RC.GridSmoothnessResidual for c in get(prob_p.problem.costs, T)]) == length(pr.rows)
    # the weight by GCV on the linearization at a perturbed start: the smoothness of that weight, the selection recorded
    sp = CalibrationState(board, st, camera_p, poses_p, mat_t, light_t, targets; masks)
    held_p = RC.interior_nodes(RC.grid_present(glens_p))
    smooth, sel = RC.select_grid_smoothness_weight(sp, RC.PSplineSmoothness(3, :gcv), held_p)
    @test smooth.type === :pspline && smooth.β == sel.β && sel.kind === :pspline && isfinite(sel.gcv) && sel.n_columns == 4 * 50
    @test sel.n_observations == 3 * sum(length, sp.px) && 0 < sel.dof <= sel.n_columns
    @test 1e-8 <= sel.β <= 1e4
    @test RC.select_grid_smoothness_weight(sp, RC.NoSmoothness(), held_p) == (nothing, nothing)
    # the nodes held: Core's footprint rule; :all every node the lens holds
    @test RC.grid_present(glens_t) == VgtkCore.node_mask(VgtkCore.model(glens_t), VgtkCore.imaged_region(glens_t))
    @test RC.grid_present(glens_t; nodes = :all) == trues(10, 10)
    @test_throws ArgumentError RC.grid_present(glens_t; nodes = :dependent)
end

@testitem "penalties: the options' new keys" setup = [RCFixtures] begin
    using TOML
    ps(w) = Dict("smoothness" => Dict("type" => "pspline", "order" => 3, "weight" => w))
    o = RC.pipeline_options(Dict("grid" => ps("gcv"), "vignetting" => merge(ps("gcv"), Dict("boundary_order" => 3))))
    @test o.grid.smoothness == RC.PSplineSmoothness(3, :gcv) && o.vignetting.smoothness == RC.PSplineSmoothness(3, :gcv) && o.vignetting.boundary_order == 3
    @test RC.pipeline_options(TOML.parse(sprint(TOML.print, RC.config_dict(o)))) == o      # the writer's "gcv" reads back
    @test RC.pipeline_options(Dict("vignetting" => ps(2))).vignetting.smoothness == RC.PSplineSmoothness(3, 2.0)
    @test RC.pipeline_options(Dict{String, Any}(); set = ["grid.smoothness={type = \"pspline\", order = 3, weight = \"gcv\"}"]).grid.smoothness == RC.PSplineSmoothness(3, :gcv)
    # today's values are the defaults: no penalty on V, order 2, the grid smoothness's weight a number
    d = RC.PipelineOptions()
    @test d.vignetting.smoothness == RC.PSplineSmoothness(3, 0.0) && d.vignetting.boundary_order == 2 && d.grid.smoothness == RC.PSplineSmoothness(3, 0.1)
    for bad in (Dict("grid" => ps("aic")), Dict("grid" => ps(-1.0)),
                Dict("vignetting" => ps("x")), Dict("grid" => Dict("boundary_order" => 2)),   # retired: no dependent nodes
                Dict("vignetting" => Dict("boundary_order" => 1)),
                Dict("vignetting" => ps(0.1), "sweep" => Dict("joint" => "on")))
        @test_throws ArgumentError RC.pipeline_options(bad)
    end
end
