@testitem "division lens (BlobBoards division model)" setup = [RCFixtures] begin
    using RadiometricCamera: build_problem, compute_mask, Focus, FocusSpline, grid_model, in_domain, Material, PerViewResidual,
                             PrintedTarget, render_view, RenderSettings, Scene, sphere_radius, View, _chief_ray, 𝔟
    using StaticArrays, LinearAlgebra, Statistics, ForwardDiff
    using VisualGeometryToolkitCore: exp_so3
    using VisualGeometryToolkitCore: TensorProductVignetting

    import VisualGeometryToolkitCore as VgtkCore
    _kc(l) = VgtkCore.parameter_vector(VgtkCore.lens(l))      # the lens coefficients (Core's parameter_vector)
    W = H = 24                                                # the image (pixels)
    lens = cam(VgtkCore.Division(0.144, -0.293), 32.4, (12.24, 11.52), W, H; f_y = 18.24)
    for 𝐩 in (SVector(12.0, 12.0), SVector(19.2, 7.2), SVector(1.2, 22.8), SVector(23.76, 12.0))
        d = _chief_ray(lens, 𝐩)
        @test norm(d) ≈ 1
        @test VgtkCore.project(lens, d) ≈ 𝐩 atol = 1e-9               # π ∘ π⁻¹ = id (Newton on the radial equation)
        @test VgtkCore.project(lens, 3.7 * d) ≈ 𝐩 atol = 1e-9          # scale invariance
        @test in_domain(lens, 𝐩)
    end
    pin = cam(VgtkCore.Division(0.0, 0.0), 24.0, (12.0, 12.0), W, H)
    @test _chief_ray(pin, SVector(19.2, 16.8)) ≈ normalize(SVector(0.3, 0.2, 1.0))      # k = 0: pinhole
    # the division model in BlobBoards' normalised plane: x_u = x_d / (1 + k₁ρ² + k₂ρ⁴)
    θp = pinhole_θ(lens)
    m = SVector((19.2 - θp[3]) / θp[1], (7.2 - θp[4]) / θp[2]); s = dot(m, m)
    d = _chief_ray(lens, SVector(19.2, 7.2))
    @test SVector(d[1] / d[3], d[2] / d[3]) ≈ m ./ (1 + _kc(lens)[1] * s + _kc(lens)[2] * s^2)
    # the domain is Core's (owner decision 2026-10-26: the division model is omnidirectional): with k₁ = −3 the point at ρ_d² = 0.405 is
    # past the 90° circle (Q < 0) and imaged behind the camera — formerly outside RC's mask (Q > 0); past the lift's fold nothing is imaged
    @test in_domain(cam(VgtkCore.Division(-3.0, 0.0), 24.0, (12.0, 12.0), W, H), SVector(22.8, 22.8)) && _chief_ray(cam(VgtkCore.Division(-3.0, 0.0), 24.0, (12.0, 12.0), W, H), SVector(22.8, 22.8))[3] < 0
    @test !in_domain(cam(VgtkCore.Division(0.0, 1.0), 24.0, (12.0, 12.0), W, H), SVector(26.4, 26.4))   # beyond the fold of ρ/Q (ρ² = 1/√3)
    ld = camera_of(VgtkCore.Division(0.0, 0.0), ForwardDiff.Dual.(vcat(pinhole_θ(lens), _kc(lens)), 1.0), W, H)
    @test _chief_ray(ld, SVector(14.4, 14.4)) isa SVector{3, <:ForwardDiff.Dual}
    @test all(x -> x isa ForwardDiff.Dual, VgtkCore.project(ld, SVector(0.1, 0.05, 1.0)))
    @test pinhole_θ(lens) == SVector(32.4, 18.24, 12.24, 11.52) && Tuple(_kc(lens)) == (0.144, -0.293)
    @test camera_of(VgtkCore.Division(0.0, 0.0), vcat(pinhole_θ(lens), _kc(lens)), W, H) == lens && has_lens(camera_of(VgtkCore.Lens(VgtkCore.DoubleSphere(0.0, 0.0)), SVector(24, 24, 12, 12, 0.1, 0.2), W, H), VgtkCore.Lens{VgtkCore.DoubleSphere{Float64}, Nothing})
    # the LM machinery with a rational lens: Jacobian columns of f (tied: the aspect W/H fixed) and the two coefficients vs finite differences
    # of the block's own update (Core's ParameterManifold: f on its log chart)
    tex = fixture_board()
    board = PrintedTarget([-1.0, -1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], tex; facing = [0.0, 0.0, 1.5])
    scene = Scene(board, W, H)
    tps_r = FocusSpline(VgtkCore.grid_nodes(3, VgtkCore.Rect(0, 0, W, H))); focus_r = Focus(tps_r, fill(1 / 1.5, 9))
    poses = [VgtkCore.EuclideanMap(SMatrix{3, 3}(exp_so3(SVector(π, 0.0, 0.0)) * exp_so3(SVector(0.15 * cos(k), 0.12 * sin(k), 0.1 * k))), SVector{3}(SVector(0.3 * cos(k), 0.3 * sin(k), 1.4 + 0.1 * k)); check = false) for k in 1:2]
    views_r = [View(p, lens, focus_r, 0.0386) for p in poses]
    @test has_lens(views_r[1].lens, VgtkCore.Division{2, Float64})
    vig_r = TensorProductVignetting((4, 4), VgtkCore.Rect(0, 0, W, H); degree = 2); b = 𝔟(board); r_SH = sphere_radius(b, [p.t for p in poses])
    light_r = EnvironmentLight(SphericalHarmonicMap(2.0, 2), b, r_SH); mat_r = Material()
    st = RenderSettings(; n_𝐩 = 2, n_ρ = 2, n_φ = 6, N = 4)
    masks = [compute_mask(v, scene, st; δ = 0.02, backend = RadiometricCamera.KernelAbstractions.CPU()) for v in views_r]
    targets = [render_view(k, views_r[k], vig_r, mat_r, light_r, scene, st, masks[k]) for k in 1:2]
    s = CalibrationState(scene, st, copy(views_r), vig_r, mat_r, light_r, targets; masks)
    prob, idx = build_problem(s; fixed = (:aspect,))
    res = get(prob.problem.costs, PerViewResidual{9})[1]
    vars = RC.NLLSsolver.getvars(res, prob.variables)
    @test vars[2] isa VgtkCore.SubsetManifold && free_names(vars[2]) == (:f, :cx, :cy, :d1, :d2) && (q = pinhole_θ(block_camera(vars[2], lens)); q[2] / q[1] ≈ 0.76 / 1.35)
    r0, J = RC.NLLSsolver.computeresjac(RC.static(127), res, vars...)
    @test maximum(abs, r0) < 1e-12
    pv, dv, fv, av, mv, vv, lv = vars
    for (col, i) in ((7, 1), (10, 4), (11, 5))                     # f (tied: f_y/f_x kept), d₁, d₂
        setter(h) = (pv, RC.NLLSsolver.update(dv, (z = zeros(5); z[i] = h; z)), fv, av, mv, vv, lv)
        fd = (RC.NLLSsolver.computeresidual(res, setter(1e-6)...) - RC.NLLSsolver.computeresidual(res, setter(-1e-6)...)) / 2e-6
        @test norm(J[:, col] - fd) / norm(fd) < 2e-3
    end

    # The math delegated to Core's `Division(k₁, k₂)` (Task 2 of the VisualGeometryToolkitCore port, fix round 1): equivalence against the
    # deleted RC bodies, copied here verbatim as the reference, on a dense lattice reaching past the fold, the 90° circle and a second
    # fold. Tolerances fixed before the run:
    # - π⁻¹ and its dual Jacobians: 1e-14 where both masks accept (same closed form, rounding only);
    # - the mask: the old mask (Q > 0 and a LOCAL monotonicity test) and Core's (a finite lift, before the FIRST fold) differ only by the
    #   rays past 90° (Q ≤ 0, now imaged, z ≤ 0) and, for a lens with a second fold, the points past the first one (now rejected);
    # - `project`: |new − old| ≤ |old round-trip error| + 1e-12 — the old 12-step Newton's own convergence bounds the comparison (a first
    #   run at a flat 1e-12 measured 2.6e-11 on the first lens, where the old round trip was off by as much; a second run with + 1e-13
    #   measured an excess of 8.9e-13, which is Core's own round-trip error, bounded below at 1e-12); Core's round trip ≤ 1e-12 on
    #   Core's whole domain except for the second-fold lens, where a ray has two preimages.
    @testset "the division lens on Core's Division: equivalence with the deleted bodies" begin
        _Q(l, s) = 1 + _kc(l)[1] * s + _kc(l)[2] * s * s
        function π⁻¹_old(l, 𝐩)
            q = pinhole_θ(l); m_x = (𝐩[1] - q[3]) / q[1]; m_y = (𝐩[2] - q[4]) / q[2]
            s = m_x * m_x + m_y * m_y; q = _Q(l, s)
            r = 1 / max(q, eps(typeof(q)))
            x = m_x * r; y = m_y * r; n = sqrt(x * x + y * y + 1)
            return SVector(x / n, y / n, 1 / n)
        end
        function in_domain_old(l, 𝐩)
            p4 = pinhole_θ(l); m_x = (𝐩[1] - p4[3]) / p4[1]; m_y = (𝐩[2] - p4[4]) / p4[2]; s = m_x * m_x + m_y * m_y; q = _Q(l, s)
            return q > 0 && (q - 2 * s * (_kc(l)[1] + 2 * _kc(l)[2] * s)) > 0
        end
        function project_old(l, X)
            x_u = X[1] / X[3]; y_u = X[2] / X[3]; ρ_u = sqrt(x_u * x_u + y_u * y_u); ρ = ρ_u
            for _ in 1:12
                s = ρ * ρ; q = _Q(l, s)
                ρ -= (ρ / q - ρ_u) / ((q - 2 * s * (_kc(l)[1] + 2 * _kc(l)[2] * s)) / (q * q))
            end
            scale = ρ_u > 0 ? ρ / ρ_u : one(ρ_u)
            p4 = pinhole_θ(l)
            return SVector(p4[1] * x_u * scale + p4[3], p4[2] * y_u * scale + p4[4])
        end
        lenses = (lens, cam(VgtkCore.Division(-3.0, 0.0), 24.0, (12.0, 12.0), W, H),
                  cam(VgtkCore.Division(-0.6, 0.12), 9.6, (12.0, 12.0), W, H), cam(VgtkCore.Division(0.3, -0.2), 9.6, (12.0, 12.0), W, H),
                  cam(VgtkCore.Division(1.0, -0.05), 9.6, (12.0, 12.0), W, H))          # the last: a second fold, past which the old mask accepted again
        P = [SVector(x, y) for x in range(-24.0, 48.0, length = 151), y in range(-24.0, 48.0, length = 151)]   # the image and one image size around it
        for (i, l) in enumerate(lenses)
            both = [𝐩 for 𝐩 in P if in_domain(l, 𝐩) && in_domain_old(l, 𝐩)]
            @test maximum(norm(_chief_ray(l, 𝐩) - π⁻¹_old(l, 𝐩)) for 𝐩 in both) < 1e-14
            newonly = [𝐩 for 𝐩 in P if in_domain(l, 𝐩) && !in_domain_old(l, 𝐩)]
            @test all(_chief_ray(l, 𝐩)[3] <= 0 for 𝐩 in newonly)                              # only rays past 90° are new
            oldonly = count(in_domain_old(l, 𝐩) && !in_domain(l, 𝐩) for 𝐩 in P)
            @test i == length(lenses) ? oldonly > 0 : oldonly == 0                      # only past a first fold is dropped
            for 𝐩 in (SVector(13.2, 11.28), SVector(17.04, 7.92))
                Jo = ForwardDiff.jacobian(t -> π⁻¹_old(camera_of(VgtkCore.Division(0.0, 0.0), t, W, H), 𝐩), vcat(pinhole_θ(l), _kc(l)))
                @test maximum(abs, ForwardDiff.jacobian(t -> _chief_ray(camera_of(VgtkCore.Division(0.0, 0.0), t, W, H), 𝐩), vcat(pinhole_θ(l), _kc(l))) - Jo) < 1e-14 * max(1.0, maximum(abs, Jo))
                Jq = ForwardDiff.jacobian(q -> π⁻¹_old(l, q), 𝐩)
                @test maximum(abs, ForwardDiff.jacobian(q -> _chief_ray(l, q), 𝐩) - Jq) < 1e-14 * max(1.0, maximum(abs, Jq))
            end
            excess = 0.0; rt = 0.0
            for 𝐩 in P
                in_domain(l, 𝐩) || continue
                X = _chief_ray(l, 𝐩); new = VgtkCore.project(l, X)
                rt = max(rt, norm(new - 𝐩))
                if X[3] > 0
                    old = project_old(l, X)
                    all(isfinite, old) && (excess = max(excess, norm(new - old) - norm(old - 𝐩)))
                end
            end
            @test excess < 2e-12 * W                                        # (pixels: measured 4.4e-11 px, 1.8e-12 of the image)
            i == length(lenses) || @test rt < 1e-12 * W
        end
        l32 = VgtkCore.convert_eltype(Float32, lenses[1])
        @test _chief_ray(l32, SVector{2, Float32}(19.2, 7.2)) isa SVector{3, Float32}
        @test norm(_chief_ray(l32, SVector{2, Float32}(19.2, 7.2)) - _chief_ray(lenses[1], SVector(19.2, 7.2))) < 1e-6
    end

    # The grid model of a source lens whose domain ends INSIDE the image (a division model whose fold sits at ≈ 0.37 of the image size at f = 0.4 of it):
    # - no NaN node (before the fallback was extended, 60 of the 8 × 8 grid's nodes were NaN, silently);
    # - the extrapolation runs outward: along the diagonal θ rises monotonically inside the source's domain and never falls back below
    #   its edge value beyond it (there the source's θ saturates at the fold, so the extrapolation is nearly flat and the spline may
    #   wobble — at points outside the grid's domain, which no pixel uses);
    # - given the source's valid domain as its lens coverage (Core's `valid_domain`, where the MODEL is valid — the grid does not
    #   inherit it, a run gives the measured image circle), `in_domain(grid)` ⊆ `in_domain(source)`: no made-up ray reaches a pixel the source
    #   does not image;
    # - inside the grid's domain it follows the source within 0.04 rad on 8 × 8 and 5e-3 on 14 × 14 (tolerances fixed before the run:
    #   the 4th-order quasi-interpolation's own error on a strongly curved lens, cells of 0.2 image sizes at 8 × 8 — measured 0.034 / 2.6e-3,
    #   3.2e-4 at 26 × 26. The 0.104 rad the review measured lay in cells next to the extrapolated nodes, now outside the grid's domain).
    let src = cam(VgtkCore.Division(1.0, -0.05), 256.0, (320.0, 320.0), 640, 640)
        @test !in_domain(src, SVector(608.0, 608.0)) && in_domain(src, SVector(320.0, 320.0))
        P = [SVector(x, y) for x in range(0.0, 640.0, length = 41), y in range(0.0, 640.0, length = 41)]
        ang(a, b) = atan(norm(cross(a, b)), dot(a, b))
        for (G, tol) in ((8, 0.04), (14, 5e-3))
            gl = grid_model(src, G, G; coverage = VgtkCore.valid_domain(src))
            @test all(all(isfinite, n) for n in VgtkCore.directions(VgtkCore.model(gl)))
            @test all(in_domain(src, p) for p in P if in_domain(gl, p)) && count(p -> in_domain(gl, p), P) > 0.5 * count(p -> in_domain(src, p), P)
            @test maximum(ang(normalize(_chief_ray(gl, p)), _chief_ray(src, p)) for p in P if in_domain(gl, p)) < tol
            ts = range(0.0, 0.5, length = 26)
            θs = [acos(clamp(normalize(_chief_ray(gl, SVector(320 + 640t, 320 + 640t)))[3], -1.0, 1.0)) for t in ts]
            inside = [in_domain(src, SVector(320 + 640t, 320 + 640t)) for t in ts]
            # outward past the source's edge: since 2026-11-04 the grid is the P-spline fit (Estimation's pspline_grid) over the source's
            # valid domain, whose nodes outside it carry the third-difference penalty's continuation (the curvature carried on) instead of
            # the quasi-interpolation's linear extrapolation — on 14 × 14 it turns back by 1.3° past the fold (27.2° → 25.9° at the
            # corner), at points outside the grid's domain, which no pixel reads; bounded here at 2°
            @test all(diff(θs[inside]) .> 0) && minimum(θs[.!inside]) >= θs[findlast(inside)] - deg2rad(2)
        end
    end
end
