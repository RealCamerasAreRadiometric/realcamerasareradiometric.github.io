# Numerics the package owned and now takes from their upstream owner (RC port, Task 3). Each section keeps the package's former body
# as a frozen REFERENCE (nothing in src/ calls it) and checks the upstream function against it on the inputs where the two could
# differ, so the replacement is pinned to "the same numbers" rather than to "a plausible formula".
@testitem "RC numerics replaced by the upstream owner's (equivalence)" setup = [RCFixtures] begin
    using RadiometricCamera: board_normal, board_plane_hit, HEIGHT_NC, intersect_board, TexturedTarget, world_point
    using StaticArrays, LinearAlgebra, Statistics, ForwardDiff
    using VisualGeometryToolkitCore: exp_so3

    using VisualGeometryToolkitCore: SO3, exp_so3, left_jacobian, skew
    using Random: Xoshiro

    # ---- B1: hat / expSO3 / leftJacobianSO3 → Core skew / exp_so3 / left_jacobian(SO3, ω) ---------------------------------------------
    # The former bodies (src/camera/pose.jl). Small-angle switch at θ² < eps^(3/4) with the truncated series I + K + K²/2 and
    # I + K/2 + K²/6. Core's exp_so3 switches at θ² < eps with a two-term series; Core's left_jacobian(SO3, ω) (fixed upstream,
    # hand-eye-nlls 6d5602e) takes a Taylor series below θ = 1/4 and the closed form above it, ≤ 4 eps from a 300-bit reference.
    ref_hat(ω) = @SMatrix [zero(ω[1]) -ω[3] ω[2]; ω[3] zero(ω[1]) -ω[1]; -ω[2] ω[1] zero(ω[1])]
    function ref_expSO3(ω)
        θ² = ω[1]^2 + ω[2]^2 + ω[3]^2
        K = ref_hat(ω)
        ForwardDiff.value(θ²) < eps(typeof(ForwardDiff.value(θ²)))^(3 // 4) && return one(SMatrix{3, 3, typeof(θ²)}) + K + K * K / 2
        θ = sqrt(θ²)
        return one(SMatrix{3, 3, typeof(θ²)}) + (sin(θ) / θ) * K + ((1 - cos(θ)) / θ²) * (K * K)
    end
    function ref_leftJacobianSO3(ω)
        θ² = ω[1]^2 + ω[2]^2 + ω[3]^2
        K = ref_hat(ω)
        ForwardDiff.value(θ²) < eps(typeof(ForwardDiff.value(θ²)))^(3 // 4) && return one(SMatrix{3, 3, typeof(θ²)}) + K / 2 + K * K / 6
        θ = sqrt(θ²)
        return one(SMatrix{3, 3, typeof(θ²)}) + ((1 - cos(θ)) / θ²) * K + ((θ - sin(θ)) / (θ² * θ)) * (K * K)
    end

    @testset "B1: SO(3) maps" begin
        rng = Xoshiro(20261026)
        rand_ω(scale) = normalize(randn(rng, SVector{3, Float64})) * scale * (0.5 + rand(rng))
        @test all(skew(v) == ref_hat(v) for v in (randn(rng, SVector{3, Float64}) for _ in 1:1000))
        # exp_so3: outside the window between the two small-angle switches (θ ∈ [√eps, eps^(3/8)] ≈ [1.5e-8, 1.4e-6]) the same arithmetic:
        # bit-identical in the default compilation, within 1e-15 under Pkg.test's --check-bounds=yes (which changes how the 3 × 3 products
        # are compiled: measured 0 at θ ≤ 1e-3, ≤ 1.1e-16 at θ ~ 0.1, ≤ 4.4e-16 (2 ulp) at θ ~ 1-3); inside the window within one ulp of the identity's scale
        for scale in (0.0, 1e-9, 1e-3, 0.1, 1.0, 3.0)
            ωs = [scale == 0 ? zero(SVector{3, Float64}) : rand_ω(scale) for _ in 1:500]
            @test maximum(maximum(abs, exp_so3(ω) - ref_expSO3(ω)) for ω in ωs) <= 1e-15
        end
        for scale in (1e-7, 1e-6)
            @test maximum(maximum(abs, exp_so3(ω) - ref_expSO3(ω)) for ω in (rand_ω(scale) for _ in 1:500)) <= 2.3e-16
        end
        # left_jacobian: the same closed form for θ ≥ 1/4 and the same value at θ ≤ 1e-9 (both series): bit-identical there in the default
        # compilation, within 1e-15 under --check-bounds=yes (measured: 0 at θ ≤ 1e-9; ≤ 1.1e-16 at θ ∈ [0.26, 1]; ≤ 2.2e-16 at θ = 3). Between, Core's
        # Taylor branch is the accurate one: within 1e-15 of a 300-bit reference (measured 1.4e-16), while the former closed form cancels in
        # (1 − cos θ)/θ² and (θ − sin θ)/θ³ (error ≈ eps/θ): the two differ by RC's own error, ≤ 1e-10 (measured 2.1e-11 at θ ≈ 2.5e-6).
        # Tolerances stated before the run.
        for θ in (0.0, 1e-12, 1e-9, 0.26, 0.3, 0.5, 1.0, 2.0, 3.0)
            ωs = [θ * normalize(randn(rng, SVector{3, Float64})) for _ in 1:200]
            @test maximum(maximum(abs, left_jacobian(SO3, ω) - ref_leftJacobianSO3(ω)) for ω in ωs) <= 1e-15
        end
        function lj_big(ω)
            ωb = BigFloat.(ω); θ = sqrt(sum(ωb .^ 2)); K = BigFloat[0 -ωb[3] ωb[2]; ωb[3] 0 -ωb[1]; -ωb[2] ωb[1] 0]
            return Matrix{BigFloat}(I, 3, 3) + ((1 - cos(θ)) / θ^2) * K + ((θ - sin(θ)) / θ^3) * (K * K)
        end
        setprecision(BigFloat, 300) do
            ωs = [θ * normalize(randn(rng, SVector{3, Float64})) for θ in exp10.(range(-8, log10(0.249), length = 300))]
            @test maximum(Float64(maximum(abs.(left_jacobian(SO3, ω) - lj_big(ω)))) for ω in ωs) <= 1e-15
            @test maximum(maximum(abs, left_jacobian(SO3, ω) - ref_leftJacobianSO3(ω)) for ω in ωs) <= 1e-10
        end
        # the LM anchor ω = 0: value and first derivative identical (the per-view Jacobian's pose columns)
        ω0 = zero(SVector{3, Float64})
        @test ForwardDiff.jacobian(w -> vec(exp_so3(SVector{3}(w))), ω0) == ForwardDiff.jacobian(w -> vec(ref_expSO3(SVector{3}(w))), ω0)
        @test ForwardDiff.jacobian(w -> vec(left_jacobian(SO3, SVector{3}(w))), ω0) ==
              ForwardDiff.jacobian(w -> vec(ref_leftJacobianSO3(SVector{3}(w))), ω0)
        @test exp_so3(SVector(1e-9, 0.0, 0.0)) ≈ I atol = 1e-8                # (moved from geometry_tests.jl)
        @test left_jacobian(SO3, zeros(SVector{3})) ≈ I
    end

    # ---- D1: the board's plane hit → Core ray_plane_intersection ------------------------------------------------------------------------
    # The former body (src/board/printed_target.jl): s = ((𝐨 − 𝐨_ray)⋅𝐧)/(𝐝⋅𝐧). Core evaluates −(d + 𝐧⋅𝐨_ray)/(𝐝⋅𝐧) with the plane's d = −𝐧⋅𝐨
    # precomputed (and the plane's normal renormalized at construction): the same quantity, rounded differently. Since Task 6a the
    # board is Core's posed target; its plane hit (on or off the page) is `board_plane_hit` (`intersect_board` is the page: off it, NaN).
    function ref_intersect_board(b, o, d)
        𝐨 = world_point(b, SVector(0.0, 0.0)); 𝐧 = board_normal(b)
        s = dot(𝐨 - o, 𝐧) / dot(d, 𝐧)
        return s, o + s * d
    end

    @testset "D1: flat-board ray intersection" begin
        # 20 000 random boards and rays, grazing ones included: |Δs|/max(|s|, 1) and |Δ𝐱|/max(|𝐱|, 1) within 1e-13 where |𝐝⋅𝐧| ≥ 0.05
        # (measured 7.3e-15 on 18 975 rays), within 1e-12 on the grazing subset (1/(𝐝⋅𝐧) amplifies the different rounding; measured
        # 8.9e-14 on 1025 rays); every hit behind the ray origin (s < 0, 10 060 rays) keeps its sign
        rng = Xoshiro(20261027)
        worst = [0.0, 0.0]; n = [0, 0]; behind = 0
        for _ in 1:20_000
            o0 = randn(rng, SVector{3, Float64}); u = randn(rng, SVector{3, Float64}); v = randn(rng, SVector{3, Float64})
            v = v - dot(v, u) / dot(u, u) * u                                      # a page's edges are orthogonal
            bd = RadiometricCamera.with_facing(TexturedTarget(o0, u, v, ones(4, 4, 3)), o0 + 3 * randn(rng, SVector{3, Float64}))
            o = 2 * randn(rng, SVector{3, Float64}); d = normalize(randn(rng, SVector{3, Float64}))
            s, 𝐱 = board_plane_hit(bd, o, d); s_ref, 𝐱_ref = ref_intersect_board(bd, o, d)
            k = abs(dot(d, board_normal(bd))) < 0.05 ? 2 : 1                                  # 2: grazing
            worst[k] = max(worst[k], abs(s - s_ref) / max(abs(s_ref), 1), maximum(abs, 𝐱 - 𝐱_ref) / max(norm(𝐱_ref), 1)); n[k] += 1
            s_ref < 0 && (behind += 1; @test s < 0)
        end
        @test n[1] + n[2] == 20_000 && n[2] > 500 && behind > 5000
        @test worst[1] <= 1e-13 && worst[2] <= 1e-12
        # the golden scenes' board, a ray of a camera looking down at it, with dual ray partials: values and partials agree to roundoff
        bd = RadiometricCamera.with_facing(TexturedTarget([-1.0, -1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], ones(4, 4, 3)), [0.0, 0.0, 1.5])
        o = SVector(0.05, -0.05, 1.5); d = normalize(SVector(0.2, -0.1, -1.0))
        f(t) = (r = board_plane_hit(bd, o + t[1:3], d + t[4:6]); SVector(r[1], r[2]...))
        f_ref(t) = (r = ref_intersect_board(bd, o + t[1:3], d + t[4:6]); SVector(r[1], r[2]...))
        t0 = zeros(SVector{6, Float64})
        @test f(t0) ≈ f_ref(t0) rtol = 1e-14
        @test ForwardDiff.jacobian(f, t0) ≈ ForwardDiff.jacobian(f_ref, t0) rtol = 1e-14
        # a ray parallel to the plane: off the plane both are infinite, on the plane both are NaN (0/0); a NaN ray gives NaN in both
        @test isinf(board_plane_hit(bd, o, SVector(1.0, 0.0, 0.0))[1]) && isinf(ref_intersect_board(bd, o, SVector(1.0, 0.0, 0.0))[1])
        o_on = SVector(0.3, -0.2, 0.0)
        @test isnan(board_plane_hit(bd, o_on, SVector(1.0, 0.0, 0.0))[1]) && isnan(ref_intersect_board(bd, o_on, SVector(1.0, 0.0, 0.0))[1])
        @test isnan(board_plane_hit(bd, o, SVector(NaN, 0.0, -1.0))[1]) && isnan(ref_intersect_board(bd, o, SVector(NaN, 0.0, -1.0))[1])
        @test isnan(board_plane_hit(bd, SVector(NaN, 0.0, 1.0), d)[1]) && isnan(ref_intersect_board(bd, SVector(NaN, 0.0, 1.0), d)[1])
        # the warped-board intersection starts from (and, with all heights zero, returns) the flat hit
        @test intersect_board(bd, o, d, zero(SVector{HEIGHT_NC, Float64})) == intersect_board(bd, o, d)
    end

    # ---- E1: gauss_legendre → QuadGK gauss(n) --------------------------------------------------------------------------------------------
    # The former body (src/sensor/quadrature.jl): Newton on Pₙ from Chebyshev-like starts, nodes stored ascending. QuadGK computes the rule
    # by Golub–Welsch (an eigenproblem) — the same rule, rounded differently (n = 3: QuadGK's weights within 3.3e-16 of 5/9, 8/9,
    # the Newton weights 2.8e-15 off).
    function ref_gauss_legendre(n::Integer)
        x = zeros(n); w = zeros(n)
        for i in 1:n
            z = cos(π * (i - 0.25) / (n + 0.5))
            pp = 0.0
            for _ in 1:100
                p1, p2 = 1.0, 0.0
                for j in 1:n
                    p3 = p2; p2 = p1
                    p1 = ((2j - 1) * z * p2 - (j - 1) * p3) / j
                end
                pp = n * (z * p1 - p2) / (z^2 - 1)
                dz = p1 / pp
                z -= dz
                abs(dz) < 1e-15 && break
            end
            x[i] = -z
            w[i] = 2 / ((1 - z^2) * pp^2)
        end
        return x, w
    end

    using QuadGK: gauss
    @testset "E1: Gauss–Legendre rule" begin
        for n in 1:12
            x, w = gauss(n); x_ref, w_ref = ref_gauss_legendre(n)
            @test issorted(x)                                                  # ascending, as the package's rules index them
            @test maximum(abs, x - x_ref) <= 4e-16                             # measured ≤ 1.1e-16 (n = 2, 8, 9, 10, 12); headroom for QuadGK/LAPACK
            @test maximum(abs, w - w_ref) <= 1e-14                             # measured ≤ 4.4e-15 (n = 10); headroom for QuadGK/LAPACK
            @test sum(w) ≈ 2 atol = 1e-14
        end
        x, w = gauss(5)                                                        # (moved from quadrature_tests.jl)
        @test sum(w .* x .^ 4) ≈ 2 / 5 atol = 1e-14
        @test maximum(abs, gauss(3)[2] - [5 / 9, 8 / 9, 5 / 9]) <= 4e-16
    end
end
