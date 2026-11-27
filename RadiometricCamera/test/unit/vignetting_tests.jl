@testitem "B-spline vignetting" setup = [RCFixtures] begin
    using RadiometricCamera: center_constraint, sensor_position
    using StaticArrays, LinearAlgebra, Statistics, ForwardDiff
    using VisualGeometryToolkitCore: exp_so3
    using VisualGeometryToolkitCore: center
    using VisualRadiometryToolkitCore: TensorProductVignetting, RadialVignetting, with_control_points, control_points, n_control_points, coefficient_jacobian
    import VisualGeometryToolkitCore as VgtkCore
    import VisualRadiometryToolkitCore as VrtkCore

    # (the knots and the basis against BSplineKit are VisualGeometryToolkitCore's tests: clamped_knots, bspline_basis)
    vig = TensorProductVignetting((5, 5), VgtkCore.Rect(0, 0, 400, 300); degree = 2)                              # over the sensor [0, 400] × [0, 300] (pixels)
    @test vig(SVector(280.0, 30.0)) ≈ 1
    P = rand(5, 5); v2 = with_control_points(vig, P)
    𝐩 = SVector(252.0, 69.0)
    @test v2(𝐩) ≈ dot(coefficient_jacobian(v2, 𝐩), vec(P))
    @test v2(SVector(560.0, 150.0)) ≈ v2(SVector(400.0, 150.0))       # clamped extrapolation
    @test center(v2) == SVector(200.0, 150.0)
    C, d = center_constraint(v2)
    @test (C * vec(P))[1] ≈ v2(SVector(200.0, 150.0))
    # the surface over the pixels is the surface over the normalized image [−½, ½]², through the affine map between them
    knx = VgtkCore.clamped_knots(5, 2, -0.5, 0.5)
    @test v2(𝐩) ≈ dot(VgtkCore.bspline_basis(knx, Val(2), 𝐩[1] / 400 - 0.5), P * VgtkCore.bspline_basis(knx, Val(2), 𝐩[2] / 300 - 0.5))
    # linearity of V in the nodes
    P1 = rand(5, 5); P2 = rand(5, 5); σ = 0.37
    @test with_control_points(vig, σ * P1 + P2)(𝐩) ≈ σ * with_control_points(vig, P1)(𝐩) + with_control_points(vig, P2)(𝐩)
    # radial profile (2026-10-21): a B-spline in the pixel radius about a fixed centre, radially symmetric by construction
    W, H = 400, 300; 𝐜 = sensor_position(210.0, 140.0); rv = RadialVignetting(7, 𝐜, 120.0)
    @test n_control_points(rv) == 7 && size(control_points(rv)) == (7, 1) && center(rv) == 𝐜
    for 𝐩 in (𝐜, SVector(320.0, 90.0), SVector(0.0, 300.0))
        @test rv(𝐩) ≈ 1 && sum(coefficient_jacobian(rv, 𝐩)) ≈ 1                                   # partition of unity, flat start
    end
    c = rand(7); rv2 = with_control_points(rv, c)
    @test with_control_points(rv, reshape(c, 7, 1)).P == rv2.P
    for r in (0.0, 35.0, 80.0, 119.9), φ in (0.0, 1.1, 2.9, 4.4)
        𝐩 = 𝐜 + SVector(r * cos(φ), r * sin(φ))
        @test rv2(𝐩) ≈ dot(coefficient_jacobian(rv2, 𝐩), c)
        @test rv2(𝐩) ≈ rv2(𝐜 + SVector(r, 0.0)) atol = 1e-12                          # radial symmetry in the pixel radius
    end
    @test rv2(𝐜 + SVector(200.0, 0.0)) ≈ rv2(𝐜 + SVector(120.0, 0.0))                   # constant beyond r_max
    C, d = center_constraint(rv2); @test (C * c)[1] ≈ rv2(𝐜) && d == [1.0]
    # the pole tie (2026-10-29): the free directions tie P_0 = P_1, so V′(0) = 0 and the field has no cone at the centre
    Zr = RC.vignetting_basis(rv, nothing); @test size(Zr) == (7, 6) && Zr[:, 1] == [1, 1, 0, 0, 0, 0, 0] && Zr[3:7, 2:6] == I
    rt = with_control_points(rv, Zr * [1.0, 0.9, 0.7, 0.5, 0.4, 0.35]); δ = SVector(1e-4 * VrtkCore.field_radius(rt), 0.0)
    d1 = rt(𝐜 + δ) - rt(𝐜); d2 = rt(𝐜 + 2δ) - rt(𝐜)
    @test rt.P[1] == rt.P[2] && abs(d2 / d1 - 4) < 1e-2                                       # V − V(𝐜) = O(r²), not O(r)
    @test RC.tie_pole(rv2).P == setindex(rv2.P, rv2.P[1], 2) && RC.tie_pole(vig) === vig
    # the support rule (2026-10-29): nodes whose knot span holds a masked pixel are parameters, the others DEPENDENT — since
    # 2026-11-07 under Core's BoundaryRule of the N × 1 lattice, the tensor surface's rule: a zero `order`-th difference inward (order 2:
    # P[k] = 2 P[k−1] − P[k−2], the last fitted slope carried past the data; until then they replicated the last supported one).
    # Pixels within ρ < 0.45 of a profile with knot intervals of 0.25: the spans of P[1] … P[5] reach the first two intervals, P[6] and P[7] do not
    pxr = [[(u, v) for u in 150:270 for v in 100:180 if hypot(u - 210, v - 140) < 0.45 * 120]]
    rule = RC.vignetting_rule(rv, nothing, pxr)
    @test rule isa RC.RadialSupportRule && rule.supported == [true, true, true, true, true, false, false] && rule.rule.order == 2
    @test rule.rule isa VgtkCore.BoundaryRule && vec(rule.rule.interior) == rule.supported && rule.rule.dependent == [CartesianIndex(6, 1), CartesianIndex(7, 1)]
    Zs = RC.vignetting_basis(rv, rule); @test size(Zs) == (7, 4) && Zs[1:2, 1] == [1, 1] && Zs[3, 2] == 1 && Zs[4, 3] == 1 && Zs[5, 4] == 1
    @test Zs[6, :] ≈ [0, 0, -1, 2] && Zs[7, :] ≈ [0, 0, -2, 3] && all(sum(Zs; dims = 2) .≈ 1)                    # P[6] = 2P[5] − P[4], P[7] = 2P[6] − P[5] = 3P[5] − 2P[4] (Core's natural weights, numeric)
    rs = RC.tie_pole(rv2, rule); @test rs.P[1] == rs.P[2] == rv2.P[1] && rs.P[3:5] == rv2.P[3:5] && rs.P[6] ≈ 2rs.P[5] - rs.P[4] && rs.P[7] ≈ 3rs.P[5] - 2rs.P[4]
    r3 = RC.vignetting_rule(rv, nothing, pxr; order = 3); Z3 = RC.vignetting_basis(rv, r3)                        # Δ³c = 0: P[6] = 3P[5] − 3P[4] + P[3], P[7] = 3P[6] − 3P[5] + P[4]
    @test r3.supported == rule.supported && r3.rule.order == 3 && Z3[6, :] == [0, 1, -3, 3] && Z3[7, :] == [0, 3, -8, 6] && Z3[1:5, :] == Zs[1:5, :]
    @test_throws ArgumentError RC.vignetting_rule(rv, nothing, pxr; order = 4)                                   # Core's BoundaryRule refuses the order
    full = RC.vignetting_rule(rv, nothing, [[(u, v) for u in 90:330 for v in 20:260]])          # pixels out to ρ > 1: every point supported
    @test all(full.supported) && RC.vignetting_basis(rv, full) == Zr && RC.vignetting_rule(vig, nothing, pxr) === nothing
    # the data-weight criterion (2026-10-30): the relative weights Σ N_i² sum to n; with `support_min` the lightly weighted nodes
    # (here the outermost: the pixels end at ρ = 1 where only P[7]'s basis is large) become dependent and continue the supported ones
    @test length(full.weight) == 7 && sum(full.weight) ≈ 7 && all(full.weight .> 0)
    thin = RC.vignetting_rule(rv, nothing, pxr; support_min = 0.3)
    @test thin.supported[1:2] == [true, true] && !thin.supported[5] && count(thin.supported) < count(rule.supported) && all(thin.weight[.!thin.supported] .< 0.3)
    @test RC.vignetting_rule(rv, nothing, pxr; support_min = 0).supported == rule.supported
    # the hard pin (2026-10-30): the free directions restricted to the null space of the centre row, so every step keeps V(𝐜) = 1 exactly;
    # for the profile (V(0) = P_0 on clamped knots) the pole column drops out, for the surface the centre-most node absorbs the others
    Zh = RC.gauge_basis(rv, Zr); @test Zh == Zr[:, 2:end] && RC.gauge_basis(rv, Zs) == Zs[:, 2:end]
    Zt = RC.gauge_basis(vig, RC.vignetting_basis(vig, nothing)); cr = collect(coefficient_jacobian(vig, center(vig)))
    @test size(Zt) == (n_control_points(vig), n_control_points(vig) - 1) && maximum(abs, cr' * Zt) < 1e-12 && rank(Zt) == size(Zt, 2)
    rp = RC.gauge_center_value(rv2); @test rp(𝐜) ≈ 1 && rp.P ≈ rv2.P ./ rv2(𝐜)
    vp = RC.gauge_center_value(with_control_points(vig, 0.5 .+ rand(n_control_points(vig)))); @test vp(center(vig)) ≈ 1
    @test RC.gauge_basis(rv, Zr[:, 2:end]) == Zr[:, 2:end]                                 # nothing to eliminate: the centre depends on no column
    c1 = rand(7); c2 = rand(7); σ = 0.37; 𝐪 = 𝐜 + SVector(44.0, 21.0)
    @test with_control_points(rv, σ * c1 + c2)(𝐪) ≈ σ * with_control_points(rv, c1)(𝐪) + with_control_points(rv, c2)(𝐪)
    r32 = VgtkCore.convert_eltype(Float32, rv2); @test r32 isa RadialVignetting{Float32}
    @test r32(𝐪) ≈ rv2(𝐪) rtol = 1e-6
    # a cubic profile represents any quadratic in ρ exactly: fit 1 − 0.4 ρ² by least squares on the basis rows
    ρs = range(0, 1, length = 60); A = reduce(vcat, [collect(coefficient_jacobian(rv, 𝐜 + SVector(ρ * 120.0, 0.0)))' for ρ in ρs])
    cq = A \ [1 - 0.4ρ^2 for ρ in ρs]
    @test maximum(abs, A * cq - [1 - 0.4ρ^2 for ρ in ρs]) < 1e-12
end
