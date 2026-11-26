# The pathwise Jacobian against Richardson-extrapolated finite differences (paper §Second Order Optimization: the Jacobian of the
# discretised residual 𝐫(Θ̄) = ϕ − Φ(Θ̄) is forward-mode AD through the same quadrature that evaluates Φ, so it must agree with finite
# differences of that residual to the roundoff level). Design: ONE fixed quadrature rule and the fixed common-random-number keys (the
# discretised Φ is then a smooth function of Θ̄ away from the sample-rejection kinks), central differences along `NLLSsolver.update` (the
# same retractions as the Jacobian) at the steps h, h/2, h/4, and Richardson extrapolation in h (O(h²) → O(h⁶)):
#   D₁ = (4 D(h/2) − D(h))/3,  D₂ = (4 D(h/4) − D(h/2))/3,  D_R = (16 D₂ − D₁)/15.
# A row whose two finest estimates D(h/2), D(h/4) disagree by more than 1e-7 of the column's scale (or than the roundoff of the
# differences, ≈ 400 ε max|𝐫| / h) sits on a kink of the sampled integrand within ±h — a ray missing the board, cos α′ ≤ 0 or the pdf
# guard of a VNDF sample, a texel boundary of the piecewise-bilinear albedo — and is excluded; at least 80 % of the rows must be in
# range. The ray-moving parameters (Θ_e, Θ_d, the grid tangents) cross a texel boundary in a few per cent of the pixels; a kink closer
# than h/4 to the point escapes the filter, so these families use the smaller step h = 1e-7 (their roundoff stays near 1e-9). Tolerance ‖J_col − D_R‖/‖D_R‖ < 1e-6 for every dual-number family (pose Θ_e, the parametric Θ_d, the grid
# tangents, t̂, r, 𝛂 / 𝛈, the nodal heights h), ≤ 1e-8 for the analytic μ / Θ_v / Θ_l columns (the same samples: exact up to roundoff);
# the printed edge is held (never a parameter): its board's pose columns are checked like the others.
# (The quadrature-level Richardson check of the earlier experiments estimated the derivative of the CONTINUOUS integral, which differs from
# the pathwise derivative of the discretised objective by the quadrature error; that is not what this test measures.)

# The scene and the per-tile (grid) problem at a point away from the truth, with its Jacobian, both items use (the GPU
# item compares the device Jacobian with `J_g`).
@testsetup module RichardsonScene
    using RadiometricCamera: TargetState, build_problem_grid, compute_mask, Focus, FocusSpline, _image_start, HEIGHT_NC, Material,
                             TexturedTarget, render_all, RenderSettings, sphere_radius, with_state, with_μ, 𝔟
    using RadiometricCamera, StaticArrays, LinearAlgebra, ForwardDiff
    using VisualGeometryToolkitCore: exp_so3
    using VisualRadiometryToolkitCore: TensorProductVignetting, with_control_points, n_control_points, EnvironmentLight, SphericalHarmonicMap,
                                       sh_count, max_degree
    import VisualGeometryToolkitCore as VgtkCore
    const RC = RadiometricCamera
    include(joinpath(@__DIR__, "fixture_board.jl"))
    include(joinpath(@__DIR__, "fixture_camera.jl"))
    export tex_r, board_r, Wr, Hr, h_r, board_r, focus_r, poses_r, vig_r, b_r, a_r, light_r, mat_r, st_r, kb_r, glens_r, blank_r, camera_g, masks_g, targets_g, s_g, prob_g, idx_g, tiles_g, res_g, vars_g, r0_g, J_g, lay_g, patch_cols_g,
           richardson_differences
    """Central differences of `f` at the steps h, h/2, h/4 and their Richardson extrapolation: `(D_R, D(h), D(h/2), D(h/4))`."""
    function richardson_differences(f, h)
        D(hh) = (f(hh) - f(-hh)) / (2hh)
        Dh, Dh2, Dh4 = D(h), D(h / 2), D(h / 4)
        D1 = (4Dh2 - Dh) / 3; D2 = (4Dh4 - Dh2) / 3
        return (16D2 - D1) / 15, Dh, Dh2, Dh4
    end

    # ---------------------------------------------------------------------------------------------------------------
    # the scene: the blob fixture with a saddle warping, two views, 32 × 32 pixels, the small fixed rule
    tex_r = fixture_board()
    board_r = TexturedTarget([-1.0, -1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], tex_r; facing = [0.0, 0.0, 1.5])
    Wr = Hr = 32
    h_r = SVector{HEIGHT_NC}([VgtkCore.Synthetic.saddle_height(𝐛) for 𝐛 in RC.HEIGHT_NODES])
    board_r = RadiometricCamera.with_state(board_r, TargetState(h = h_r))
    focus_r = Focus(FocusSpline(VgtkCore.grid_nodes(3, VgtkCore.Rect(0, 0, Wr, Hr))), [1 / (1.5 + 0.1 * sin(3i)) for i in 1:9])
    poses_r = [VgtkCore.EuclideanMap(SMatrix{3, 3}(exp_so3(SVector(π, 0.0, 0.0)) * exp_so3(SVector(0.15 * cos(k), 0.12 * sin(k), 0.1 * k))), SVector{3}(SVector(0.2 * cos(k), 0.2 * sin(k), 1.2 + 0.1 * k)); check = false) for k in 1:2]
    vig_r = let v = TensorProductVignetting((4, 4), VgtkCore.Rect(0, 0, Wr, Hr); degree = 2); gr = range(-0.5, 0.5, length = 4); with_control_points(v, [1.0 - 0.3 * (gr[i]^2 + gr[j]^2) for i in 1:4, j in 1:4]) end
    b_r = 𝔟(board_r); a_r = zeros(3, 9); a_r[:, 1] .= (2.0, 1.9, 1.8); a_r[:, 3] .= (0.4, 0.3, 0.2)
    light_r = EnvironmentLight(SphericalHarmonicMap(a_r, 2), b_r, sphere_radius(b_r, [p.t for p in poses_r]))
    mat_r = Material(; roughness_ink = 0.12, roughness_paper = 0.35, refractive_index_ink = 1.65, refractive_index_paper = 1.33, diffuse_weight = 0.6)      # strictly inside every box
    st_r = RenderSettings(; n_𝐩 = 2, n_ρ = 2, n_φ = 6, N = 4, seed = 42)
    kb_r = cam(VgtkCore.KannalaBrandt(-0.03, 0.004, 0.0, 0.0), 41.216, (16.32, 15.36), Wr, Hr)
    glens_r = _image_start(cam(VgtkCore.Lens(VgtkCore.DoubleSphere(0.25, 0.4)), 41.216, (16.32, 15.36), Wr, Hr), (10, 10)).camera
    blank_r = [zeros(Hr, Wr, 3) for _ in 1:2]
    # ---------------------------------------------------------------------------------------------------------------
    # 2. the per-tile route (the grid model: the node tangents through the bicubic chain rule, the nodal heights h as duals, Θ_e, Θ_f, r, Θ_m)
    camera_g = RadiometricCameraModel(glens_r, focus_r, 0.0386, vig_r)
    masks_g = [compute_mask(camera_g, p, board_r, st_r; δ = 0.03, backend = RadiometricCamera.KernelAbstractions.CPU()) for p in poses_r]
    targets_g = let p = calibration_problem(board_r, st_r, camera_g, poses_r, mat_r, light_r, blank_r; masks = masks_g); render_all(p.f, p.u0) end
    s_g = calibration_problem(RadiometricCamera.with_state(board_r, TargetState(h = 0.5 .* h_r)), st_r, camera_g, poses_r, with_μ(mat_r, 0.55), light_r, targets_g; masks = masks_g)
    prob_g, idx_g = build_problem_grid(s_g.f, s_g.u0; damping = VgtkCore.LevenbergDamping(), patch = (5, 5), height_degree = 1)
    tiles_g = RC.tile_costs(prob_g)
    res_g = tiles_g[argmax([length(t.pix) for t in tiles_g])]
    vars_g = collect(RC.NLLSsolver.getvars(res_g, prob_g.variables))
    r0_g, J_g = RC.NLLSsolver.computeresjac(RC.static(1023), res_g, vars_g...)
    lay_g = RC._tile_layout(RC.static(1023), 9, length(st_r.sensor), 25, VgtkCore.tangent_size(vars_g[9]), n_control_points(s_g.u0.camera.vignetting), sh_count(max_degree(s_g.u0.light)),
                            RC.n_height(vars_g[8]))
    # the strongest grid tangent columns of each free patch (the others are near zero: nodes outside the tile's support)
    patch_cols_g = Tuple{Int, Int, Int, Int, Float64, Float64}[]
    for q in 1:4
        pc = lay_g.patch_cols[q]; pc > 0 || continue
        strengths = [norm(J_g[:, pc - 1 + n]) for n in 1:50]
        for n in sortperm(strengths; rev = true)[1:3]
            push!(patch_cols_g, (1 + q, 50, n, pc - 1 + n, 1e-7, 1e-6))
        end
    end
end

@testitem "pathwise Jacobian vs Richardson finite differences" setup = [RCFixtures, RichardsonScene] begin
    using RadiometricCamera: TargetState, build_problem, build_problem_grid, compute_mask, focus, material, PerViewResidual,
                             TexturedTarget, render_all, t̂, with_sdf, with_state, with_μ, with_vignetting, 𝛂, 𝛈
    using StaticArrays, LinearAlgebra, Statistics, ForwardDiff
    using VisualGeometryToolkitCore: exp_so3
    using VisualRadiometryToolkitCore: with_control_points, control_points, n_control_points

    """Central differences of `f` (a residual vector) along `update` of one block at the steps h, h/2, h/4 and the Richardson extrapolation;
    returns `(D_R, inrange)` with `inrange[i]` the in-range flag of row i. `fscale` = the magnitude of the summed flux (the observations ϕ of
    the rows: the residual is a difference of two such numbers, so its roundoff follows the flux, not the residual)."""
    function richardson_column(f, h, fscale)
        DR, Dh, Dh2, Dh4 = richardson_differences(f, h)
        scale = max.(abs.(DR), 1e-3 * maximum(abs, DR))              # rows with a negligible derivative are judged against the column's scale
        noise = 400 * eps() * fscale / h                            # the roundoff of the differences (each 𝐫 sums ≈ 200 rays of size ≈ fscale)
        d12 = abs.(Dh - Dh2); d24 = abs.(Dh2 - Dh4)
        # in range: the two finest estimates agree to roundoff (a smooth row at a small step), or the differences shrink by ≥ 1/0.3 ≈ 4
        # (clean O(h²) truncation, which the extrapolation removes); a kink within ±h does neither
        inrange = (d24 .<= max.(1e-7 .* scale, noise)) .| (d24 .<= 0.3 .* d12 .+ noise)
        return DR, inrange
    end
    """Relative error of a Jacobian column on the in-range rows, the in-range fraction and the error over ALL rows (for the report)."""
    function column_error(Jcol, DR, inrange)
        frac = count(inrange) / length(inrange)
        d = norm(Jcol[inrange] .- DR[inrange]); n = norm(DR[inrange])
        dall = norm(Jcol .- DR); nall = norm(DR)
        return (n > 0 ? d / n : d), frac, (nall > 0 ? dall / nall : dall)
    end
    # ---------------------------------------------------------------------------------------------------------------
    # 1. the per-view route (a parametric model: Θ_e, Θ_d of the Kannala–Brandt model, Θ_f, r, Θ_m; analytic Θ_v, Θ_l, μ)
    camera_r = RadiometricCameraModel(kb_r, focus_r, 0.0386, vig_r)
    masks_r = [compute_mask(camera_r, p, board_r, st_r; δ = 0.03, backend = RadiometricCamera.KernelAbstractions.CPU()) for p in poses_r]
    targets_r = let p = calibration_problem(board_r, st_r, camera_r, poses_r, mat_r, light_r, blank_r; masks = masks_r); render_all(p.f, p.u0) end
    s_v = calibration_problem(board_r, st_r, with_vignetting(camera_r, with_control_points(vig_r, 1.05 .* Matrix(control_points(vig_r)))), poses_r, with_μ(mat_r, 0.55),
                           with_coefficients(light_r, 0.9 .* a_r), targets_r; masks = masks_r)          # away from the truth: residuals ≠ 0
    prob_v, idx_v = build_problem(s_v.f, s_v.u0; damping = VgtkCore.LevenbergDamping())
    res_v = get(prob_v.problem.costs, PerViewResidual{9})[1]
    vars_v = collect(RC.NLLSsolver.getvars(res_v, prob_v.variables))
    r0_v, J_v = RC.NLLSsolver.computeresjac(RC.static(127), res_v, vars_v...)
    @test r0_v ≈ RC.NLLSsolver.computeresidual(res_v, vars_v...) atol = 1e-12
    lay_v = RC._jacobian_layout(RC.static(127), 9, length(st_r.sensor), VgtkCore.tangent_size(vars_v[6]), n_control_points(s_v.u0.camera.vignetting), sh_count(max_degree(s_v.u0.light)), VgtkCore.tangent_size(vars_v[2]))
    fres_v(j, nvar, i) = hh -> RC.NLLSsolver.computeresidual(res_v, updated_vars(vars_v, j, unit_increment(nvar, i, hh))...)
    fscale_v = maximum(abs, s_v.f.ϕ[res_v.k])
    # (block, variable index, column, step, tolerance)
    pose_cols_v = [(1, 6, i, i, 1e-7, 1e-6) for i in 1:6]                                         # Θ_e: ω [rad], ρ [m]
    kb_cols_v = [(2, 8, i, 6 + i, (3e-8, 3e-8, 3e-8, 3e-8, 1e-7, 1e-7, 1e-6, 1e-5)[i], 1e-6) for i in 1:8]   # the Kannala–Brandt lens block (Core's ParameterManifold: ln f, ln aspect, cx, cy, k₁…k₄): f, c move every ray (the smallest step against texel kinks); k₃, k₄ ∝ θ⁷, θ⁹ barely move one (larger steps against roundoff)
    focus_cols_v = [(3, 9, i, lay_v.focus_cols[i], 1e-6, 1e-6) for i in (1, 5, 9)]                 # t̂ nodes (the TPS chain rule)
    r_cols_v = [(4, 1, 1, lay_v.col_r, 3e-6, 1e-6)]                                                # ln r (log-parametrised radius; ≈ the relative step of 1e-7 at r = 0.0386)
    mat_cols_v = [(5, 9, i, lay_v.μ_col - 9 + i, 1e-6, 1e-6) for i in 1:8]                          # 𝛂, 𝛈 (dual through the VNDF sampling)
    μ_cols_v = [(5, 9, 9, lay_v.μ_col, 1e-4, 1e-8)]                                                # μ: analytic (affine)
    vig_cols_v = [(6, length(lay_v.vig_cols), i, lay_v.vig_cols[i], 1e-3, 1e-8) for i in (1, 6, 11, length(lay_v.vig_cols))]   # Θ_v: analytic (linear); the last free direction (15 under the hard pin)
    light_cols_v = [(7, length(lay_v.light_cols), i, lay_v.light_cols[i], 1e-3, 1e-8) for i in (1, 2, 7, 14, 27)]   # Θ_l: analytic (linear)
    for (name, cols) in (("pose", pose_cols_v), ("Kannala–Brandt lens block", kb_cols_v), ("focus t̂", focus_cols_v), ("aperture r", r_cols_v),
                         ("material 𝛂/𝛈", mat_cols_v), ("μ (analytic)", μ_cols_v), ("vignetting (analytic)", vig_cols_v), ("lighting (analytic)", light_cols_v))
        worst = 0.0; minfrac = 1.0; worst_all = 0.0
        for (j, nvar, i, col, h, tol) in cols
            DR, inrange = richardson_column(fres_v(j, nvar, i), h, fscale_v)
            err, frac, err_all = column_error(J_v[:, col], DR, inrange)
            worst = max(worst, err); minfrac = min(minfrac, frac); worst_all = max(worst_all, err_all)
            @test frac >= 0.8
            @test err < tol
        end
        @info "pathwise Jacobian vs Richardson FD (per-view route): $name" worst_relative_error = worst min_in_range_fraction = minfrac worst_error_all_rows = worst_all
    end

    @test r0_g ≈ RC.NLLSsolver.computeresidual(res_g, vars_g...) atol = 1e-12
    fres_g(j, nvar, i) = hh -> RC.NLLSsolver.computeresidual(res_g, updated_vars(vars_g, j, unit_increment(nvar, i, hh))...)
    fscale_g = maximum(abs, res_g.ϕt)
    pose_cols_g = [(1, 6, i, i, 1e-7, 1e-6) for i in 1:6]
    focus_cols_g = [(6, 9, i, lay_g.focus_cols[i], 1e-6, 1e-6) for i in (1, 5, 9)]
    r_cols_g = [(7, 1, 1, lay_g.col_r, 3e-6, 1e-6)]                                                # ln r
    mat_cols_g = [(8, 9 + RC.HEIGHT_NFREE, i, lay_g.μ_col - 9 + i, 1e-6, 1e-6) for i in 1:8]
    μ_cols_g = [(8, 9 + RC.HEIGHT_NFREE, 9, lay_g.μ_col, 1e-4, 1e-8)]
    height_cols_g = [(8, 9 + RC.HEIGHT_NFREE, 9 + i, lay_g.height_cols[i], 1e-6, 1e-6) for i in (1, 7, 13, RC.HEIGHT_NFREE)]   # h [m]: duals through the Newton intersection
    vig_cols_g = [(9, length(lay_g.vig_cols), i, lay_g.vig_cols[i], 1e-3, 1e-8) for i in (1, 6, 11, length(lay_g.vig_cols))]
    light_cols_g = [(10, length(lay_g.light_cols), i, lay_g.light_cols[i], 1e-3, 1e-8) for i in (1, 2, 7, 14, 27)]
    for (name, cols) in (("pose", pose_cols_g), ("grid tangents", patch_cols_g), ("focus t̂", focus_cols_g), ("aperture r", r_cols_g),
                         ("material 𝛂/𝛈", mat_cols_g), ("μ (analytic)", μ_cols_g), ("board warping h", height_cols_g),
                         ("vignetting (analytic)", vig_cols_g), ("lighting (analytic)", light_cols_g))
        worst = 0.0; minfrac = 1.0; worst_all = 0.0
        for (j, nvar, i, col, h, tol) in cols
            DR, inrange = richardson_column(fres_g(j, nvar, i), h, fscale_g)
            err, frac, err_all = column_error(J_g[:, col], DR, inrange)
            worst = max(worst, err); minfrac = min(minfrac, frac); worst_all = max(worst_all, err_all)
            @test frac >= 0.8
            @test err < tol
        end
        @info "pathwise Jacobian vs Richardson FD (per-tile route): $name" worst_relative_error = worst min_in_range_fraction = minfrac worst_error_all_rows = worst_all
    end

    # ---------------------------------------------------------------------------------------------------------------
    # 3. a distance-map board (the printed edge, held): the pose columns through the distance map against the Richardson FD
    page_r = (200.0, 200.0)
    rects_r = [(12.5 + 50i, 12.5 + 50j, 37.5 + 50i, 37.5 + 50j) for i in 0:3 for j in 0:3]
    sdf_r = let H = 160, W = 160, out = zeros(H, W)
        for i in 1:H, j in 1:W
            x = (j - 0.5) / W * page_r[1]; y = (i - 0.5) / H * page_r[2]
            din = Inf; dout = Inf
            for (x0, y0, x1, y1) in rects_r
                if x0 <= x <= x1 && y0 <= y <= y1; din = min(din, x - x0, x1 - x, y - y0, y1 - y)
                else dout = min(dout, hypot(max(x0 - x, 0.0, x - x1), max(y0 - y, 0.0, y - y1))) end
            end
            out[i, j] = isfinite(din) ? -din : dout
        end
        out
    end
    board_e_r = with_sdf(TexturedTarget([-1.0, -1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], ones(16, 16, 3); facing = [0.0, 0.0, 1.5]), sdf_r, page_r)
    # the print's (δ, σ_print) on the surface: σ_print 3 mm, about one pixel on the board, so the render resolves the edge
    board_e_r = RadiometricCamera.with_state(board_e_r, TargetState(edge = SVector(0.3, 3.0)))
    camera_e_r = RadiometricCameraModel(glens_r, focus_r, 0.0386, vig_r)
    masks_e_r = [compute_mask(camera_e_r, p, board_e_r, st_r; δ = 0.03, backend = RadiometricCamera.KernelAbstractions.CPU()) for p in poses_r]
    targets_e_r = let p = calibration_problem(RadiometricCamera.with_state(board_e_r, TargetState(edge = SVector(0.6, 3.0))), st_r,
                                              camera_e_r, poses_r, mat_r, light_r, blank_r; masks = masks_e_r); render_all(p.f, p.u0) end
    s_e = calibration_problem(board_e_r, st_r, camera_e_r, poses_r, mat_r, light_r, targets_e_r; masks = masks_e_r)
    prob_e, _ = build_problem_grid(s_e.f, s_e.u0; damping = VgtkCore.LevenbergDamping(), patch = (5, 5))
    tiles_e = RC.tile_costs(prob_e)
    res_e = tiles_e[argmax([length(t.pix) for t in tiles_e])]
    vars_e = collect(RC.NLLSsolver.getvars(res_e, prob_e.variables))
    _, J_e = RC.NLLSsolver.computeresjac(RC.static(1023), res_e, vars_e...)
    fscale_e = maximum(abs, res_e.ϕt)
    # the pose columns through the distance map: dual numbers again
    for i in 1:6
        DR, inrange = richardson_column(hh -> RC.NLLSsolver.computeresidual(res_e, updated_vars(vars_e, 1, unit_increment(6, i, hh))...), 1e-7, fscale_e)
        err, frac, _ = column_error(J_e[:, i], DR, inrange)
        @test frac >= 0.8
        @test err < 1e-6
    end
end

@testitem "pathwise Jacobian vs Richardson finite differences (GPU)" tags = [:gpu] setup = [RCFixtures, RCGPU, RichardsonScene] begin
    using RadiometricCamera: TargetState, build_problem_grid, focus, material, t̂, with_state, with_μ
    using StaticArrays, LinearAlgebra, Statistics, ForwardDiff
    using VisualGeometryToolkitCore: exp_so3
    import CUDA

    # ---------------------------------------------------------------------------------------------------------------
    # 4. the Float32 device path (under the GPU gate): CPU / GPU column parity per family and the residual bitwise from the flux kernel
    require_gpu()                                  # errors with the reason when the GPU the runner chose is not usable here
    if RC_GPU_TESTS
        s_gd = calibration_problem(RadiometricCamera.with_state(board_r, TargetState(h = 0.5 .* h_r)), st_r, camera_g, poses_r, with_μ(mat_r, 0.55), light_r, targets_g; masks = masks_g,
                                backend = CUDA.CUDABackend())
        prob_gd, idx_gd = build_problem_grid(s_gd.f, s_gd.u0; damping = VgtkCore.LevenbergDamping(), patch = (5, 5), height_degree = 1)
        tiles_gd = RC.tile_costs(prob_gd)
        res_gd = tiles_gd[findfirst(t -> t.k == res_g.k && t.P0 == res_g.P0 && length(t.pix) == length(res_g.pix), tiles_gd)]
        vars_gd = RC.NLLSsolver.getvars(res_gd, prob_gd.variables)
        r0_gd, J_gd = RC.NLLSsolver.computeresjac(RC.static(1023), res_gd, vars_gd...)
        @test r0_gd == RC.NLLSsolver.computeresidual(res_gd, vars_gd...)                           # bitwise: both from the flux kernel
        @test maximum(abs, r0_gd .- r0_g) < 2e-3 * max(1.0, maximum(abs, r0_g))
        for (name, cols) in (("pose", 1:6), ("grid tangents", [c[4] for c in patch_cols_g]), ("focus t̂", lay_g.focus_cols), ("aperture r", lay_g.col_r:lay_g.col_r),
                             ("material + μ", (lay_g.μ_col - 8):lay_g.μ_col), ("board warping h", lay_g.height_cols), ("vignetting", lay_g.vig_cols), ("lighting", lay_g.light_cols))
            rel = norm(J_gd[:, cols] .- J_g[:, cols]) / norm(J_g[:, cols])
            @test rel < 1e-2                                                                       # Float32 duals, 48 rays per pixel
            @info "device Jacobian parity (Float32): $name" relative_error = rel
        end
    end
end
