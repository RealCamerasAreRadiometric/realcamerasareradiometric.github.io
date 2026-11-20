@testitem "radiance" setup = [RCFixtures] begin
    using RadiometricCamera: TargetState, chart, draws, cook_torrance, L_c, L_from_parts, Material, TexturedTarget,
                             radiance_parts, RadianceSampler, SampleKey, texture_weight, with_μ, 𝔟, 𝛂, 𝛈
    using VisualRadiometryToolkitCore: ρ_c
    using StaticArrays, LinearAlgebra, Statistics, ForwardDiff
    using VisualGeometryToolkitCore: exp_so3

    # (the Cook–Torrance BSDF and its GGX pieces are VisualRadiometryToolkitCore's, tested there; here: the Monte-Carlo radiance)
    # Monte-Carlo radiance vs brute-force hemisphere integral of f^CT L cos
    tex = ones(16, 16, 3) .* reshape([0.8, 0.5, 0.3], 1, 1, 3)
    bd = TexturedTarget([-1.0, -1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], tex; facing = [0.0, 0.0, 1.0])
    a = zeros(3, 9); a[:, 1] .= 2.0; a[:, 3] .= 0.5; a[:, 4] .= -0.3; a[:, 8] .= 0.2
    lt = EnvironmentLight(SphericalHarmonicMap(a, 2), 𝔟(bd), 2.5)
    m = Material(; roughness_ink = 0.3, roughness_paper = 0.3, refractive_index_ink = 1.5, refractive_index_paper = 1.5, diffuse_weight = 0.6)
    𝐱 = SVector(0.2, -0.1, 0.0); 𝐛 = chart(bd, 𝐱); ωv = normalize(SVector(-0.4, 0.3, 1.0))
    t, b, n = RC.board_frame(bd); ω_l = RC.to_local(t, b, n, ωv)
    ref = zero(SVector{3, Float64}); nθ, nφ = 300, 600
    ρ = ρ_c(bd, 𝐛); wv = texture_weight(ρ); αb = 𝛂(m, wv); ηb = 𝛈(m, wv)
    for i in 1:nθ, j in 1:nφ
        θ = (i - 0.5) * (π / 2) / nθ; φ = (j - 0.5) * 2π / nφ
        ω′ = SVector(sin(θ) * cos(φ), sin(θ) * sin(φ), cos(θ))
        h = normalize(ω_l + ω′)
        L = VrtkCore.radiance(lt, 𝐱,RC.local_to_world(t, b, n, ω′[1], ω′[2], ω′[3]))
        global ref += cook_torrance(m, ρ, αb, ηb, ω_l, ω′, h) .* L * cos(θ) * sin(θ) * (π / 2 / nθ) * (2π / nφ)
    end
    mc = L_c(bd, m, lt, RadianceSampler(60000; hammersley = true), SampleKey(1, 2, 3, 4), 𝐛, 𝐱, -ωv)
    @test norm(mc - ref) / norm(ref) < 0.01
    # basis form and μ-split reproduce the estimator, linearity in a, affinity in μ
    s8 = RadianceSampler(8); key = SampleKey(1, 2, 3, 4)
    parts = radiance_parts(bd, m, lt, s8, key, 𝐛, 𝐱, -ωv)
    @test L_from_parts(parts, coefficients(lt), m.diffuse_weight) ≈ L_c(bd, m, lt, s8, key, 𝐛, 𝐱, -ωv) atol = 1e-12
    a2 = rand(3, 9); σ = 0.4
    @test L_c(bd, m, with_coefficients(lt, σ * a + a2), s8, key, 𝐛, 𝐱, -ωv) ≈ σ * L_c(bd, m, lt, s8, key, 𝐛, 𝐱, -ωv) + L_c(bd, m, with_coefficients(lt, a2), s8, key, 𝐛, 𝐱, -ωv) atol = 1e-12
    L0 = L_c(bd, with_μ(m, 0.0), lt, s8, key, 𝐛, 𝐱, -ωv); L1 = L_c(bd, with_μ(m, 1.0), lt, s8, key, 𝐛, 𝐱, -ωv)
    @test L_c(bd, with_μ(m, 0.3), lt, s8, key, 𝐛, 𝐱, -ωv) ≈ 0.3 * L1 + 0.7 * L0 atol = 1e-12
    # determinism (common random numbers)
    @test L_c(bd, m, lt, s8, key, 𝐛, 𝐱, -ωv) == L_c(bd, m, lt, s8, key, 𝐛, 𝐱, -ωv)
    @test draws(s8, key, 1) != draws(s8, SampleKey(1, 2, 3, 5), 1)
    # Dual numbers through the estimator (attached derivative w.r.t. roughness)
    md = Material(ForwardDiff.Dual(0.3, 1.0), ForwardDiff.Dual(0.3, 1.0), ForwardDiff.Dual.(SVector(1.5, 1.5, 1.5), 0.0), ForwardDiff.Dual.(SVector(1.5, 1.5, 1.5), 0.0), ForwardDiff.Dual(0.6, 0.0))
    Ld = L_c(bd, md, lt, s8, key, 𝐛, 𝐱, -ωv)
    fd = (L_c(bd, Material(; roughness_ink = 0.3 + 1e-6, roughness_paper = 0.3 + 1e-6, refractive_index_ink = 1.5, refractive_index_paper = 1.5, diffuse_weight = 0.6), lt, s8, key, 𝐛, 𝐱, -ωv) -
          L_c(bd, Material(; roughness_ink = 0.3 - 1e-6, roughness_paper = 0.3 - 1e-6, refractive_index_ink = 1.5, refractive_index_paper = 1.5, diffuse_weight = 0.6), lt, s8, key, 𝐛, 𝐱, -ωv)) / 2e-6
    @test ForwardDiff.partials.(Ld, 1) ≈ fd rtol = 1e-4
end
