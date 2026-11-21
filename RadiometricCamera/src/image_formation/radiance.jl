# ---------------------------------------------------------------------------
# Per-channel scene radiance by Monte-Carlo estimation over VNDF samples:
#
#   L_c(𝐱, ω; Θ_r) ≈ (1/N) Σ_{i=1}^{N} f_c^CT(𝐱, ω_o, ω′_i; Θ_m) L_c^SVSH(𝐱, ω′_i; Θ_l) |cos α′(ω′_i)| / PDF(𝐧_i, ω_o; 𝛂)
#
# where ω is the direction from the aperture point to the board point 𝐱 (the paper's second argument
# π̂⁻¹(𝐩, 𝐚) − 𝐚) and ω_o = −ω the outgoing direction of the BRDF, with 𝐧_i ~ D^v(⋅, ω_o; 𝛂(𝐛)),
# ω′_i = 2(ω_o⋅𝐧_i)𝐧_i − ω_o. The only surface is the planar
# board, so 𝐱′(𝐱, ω′) lies on the light sphere (single bounce). The uniform draws
# come from a counter-based hash of a `SampleKey` (view, pixel, sub-pixel, aperture
# sample, i): common random numbers — deterministic in Θ̄, shared by every basis /
# split render, thread-safe.
# ---------------------------------------------------------------------------

"""
    RadianceSampler

`N` VNDF samples per (𝐱, ω); `seed` of the counter-based random stream; `hammersley`
uses the deterministic Hammersley set (scrambled per key) instead of pseudo-random draws.
"""
struct RadianceSampler
    N::Int
    seed::UInt64
    hammersley::Bool
end
RadianceSampler(N::Integer = 8; seed::Integer = 42, hammersley::Bool = false) = RadianceSampler(N, UInt64(seed), hammersley)

"""Identifies the sample stream of one radiance evaluation (view, pixel, sub-pixel, aperture index)."""
struct SampleKey
    view::UInt64
    pixel::UInt64
    sub::UInt64
    aperture::UInt64
end

@inline function _mix64(z::UInt64)
    z += 0x9e3779b97f4a7c15
    z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9
    z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
    return z ⊻ (z >> 31)
end

"""Key hash of one radiance evaluation (hoist out of the per-sample loop)."""
@inline key_hash(s::RadianceSampler, k::SampleKey) =
    _mix64(s.seed ⊻ _mix64(k.view ⊻ _mix64(k.pixel ⊻ _mix64(k.sub ⊻ _mix64(k.aperture)))))

@inline _bits_to_unit(::Type{Float64}, h::UInt64) = (h >> 11) * 2.0^-53
@inline _bits_to_unit(::Type{Float32}, h::UInt64) = Float32(h >> 40) * Float32(2)^-24
@inline _unit_eps(::Type{T}) where {T} = T(1//1000000000)

"""
    draws(sampler, [::Type{T} = Float64,] h_or_key, i) -> (u₁, u₂)

Two `T`-typed uniforms in (0, 1) for sample `i` of a key hash `h` (counter-based, stateless). The type follows the sampler (Julia's
order, as `rand(rng, T, dims)`: the sampler is the random stream).
"""
@inline function draws(s::RadianceSampler, ::Type{T}, h::UInt64, i::Integer) where {T}
    lo = _unit_eps(T); hi = one(T) - lo
    if s.hammersley
        u1, u2 = hammersley(T, i, s.N)
        r = _mix64(h)
        u1 += _bits_to_unit(T, r); u1 -= (u1 >= 1) * one(T)
        u2 += _bits_to_unit(T, _mix64(r)); u2 -= (u2 >= 1) * one(T)
        return clamp(u1, lo, hi), clamp(u2, lo, hi)
    end
    h1 = _mix64(h ⊻ (UInt64(i) * 0xd1342543de82ef95))
    h2 = _mix64(h1)
    return clamp(_bits_to_unit(T, h1), lo, hi), clamp(_bits_to_unit(T, h2), lo, hi)
end
@inline draws(s::RadianceSampler, ::Type{T}, k::SampleKey, i::Integer) where {T} = draws(s, T, key_hash(s, k), i)
@inline draws(s::RadianceSampler, h_or_key::Union{UInt64, SampleKey}, i::Integer) = draws(s, Float64, h_or_key, i)

"""Direction with local coordinates `(x, y, z)` in the frame `(t, b, n)`."""
@inline local_to_world(t, b, n, x, y, z) = x * t + y * b + z * n

"""Local board frame `(t, b, n)` of a board."""
@inline function board_frame(bd::TexturedTarget)
    n = board_normal(bd)
    t, b = VgtkCore.direction_tangents(n)            # the tangent basis of the normal: Core's, in n's element type
    return t, b, n
end
@inline to_local(t, b, n, w::StaticVector{3}) = SVector(dot(w, t), dot(w, b), dot(w, n))

"""The common start of a ray's radiance (`L_c`, the radiance-Jacobian modes of `_ray_radiance`): the ray's direction ω_l in the board's local frame, the
element type `T` of its radiance, whether the ray reaches the board from the front (`front`; from behind its radiance is zero) and
the material `(ρ, α, η)` at the chart point 𝐛 (`_radiance_material`), or of the albedo `ρ` the caller looked up (a chunk's anchored
lookup, `_ray_radiance`). Type-stable in the kernels: the material is evaluated for every ray (a ray from behind discards it)."""
@inline _ray_material(bd, m, lt, 𝐛::StaticVector{2}, 𝐱, ω) = _ray_material(bd, m, lt, physical_albedo(bd, 𝐛), 𝐱, ω)
@inline function _ray_material(bd, m, lt, ρ::StaticVector{3}, 𝐱, ω)
    t, b, n = board_frame(bd)
    ω_l = to_local(t, b, n, -ω)
    T = promote_type(eltype(ω_l), eltype(𝐱), eltype(m.refractive_index_ink), eltype(coefficients(lt)))
    return ω_l, T, ω_l[3] > 0, _radiance_material(m, ρ)
end
"""
    L_c(board, mat, light, sampler, key, 𝐛, 𝐱, ω) -> SVector{3}

Monte-Carlo estimate of the per-channel radiance of the point 𝐱 (chart 𝐛) of the board (a `TexturedTarget`, under its state:
its printed edge and in-plane print error) seen along ω, the
unit direction from the aperture point to the board point (world frame; the paper's `π̂⁻¹(𝐩, 𝐚) − 𝐚`).
The radiance leaves the board towards −ω.
"""
@inline function L_c(bd::TexturedTarget, m::Material, lt::EnvironmentLight, s::RadianceSampler, key::SampleKey,
                     𝐛::StaticVector{2}, 𝐱::StaticVector{3}, ω::StaticVector{3})
    ω_l, Tz, front, (ρ, α, η) = _ray_material(bd, m, lt, 𝐛, 𝐱, ω)   # ω_l: the outgoing direction ω_o = −ω; 𝐱 may be a dual board point (HeightRays)
    front || return zero(SVector{3, Tz})
    return _radiance_mc(bd, m, lt, s, key, ρ, α, η, ω_l, 𝐱, Tz)
end
"""The albedo ρ, roughness α and refractive indices η of a board point of albedo `ρ` (`L_c`'s material inputs; ρ the target's
`physical_albedo`: the texture, or the printed edge, at the DESIGN point the physical point shows): α and η blend ink ↔ no ink by
the albedo's value."""
@inline function _radiance_material(m::Material, ρ::StaticVector{3})
    wv = texture_weight(ρ)
    return ρ, 𝛂(m, wv), 𝛈(m, wv)
end
"""`L_c`'s Monte-Carlo estimator at its local inputs: the albedo ρ, roughness α, refractive indices η, the outgoing direction ω_l
(board frame, facing: ω_l[3] > 0) and the board point 𝐱; `Tz` its element type. μ and the lighting enter as they are."""
@inline function _radiance_mc(bd::TexturedTarget, m::Material, lt::EnvironmentLight, s::RadianceSampler, key::SampleKey,
                              ρ, α, η, ω_l::StaticVector{3}, 𝐱::StaticVector{3}, ::Type{Tz}) where {Tz}
    t, b, n = board_frame(bd)
    acc = zero(SVector{3, Tz})
    h = key_hash(s, key)
    Tu = typeof(ForwardDiff.value(zero(Tz)))
    pdf_guard = eps(Tu) * Tu(10000)
    for i in 1:s.N
        u₁, u₂ = draws(s, Tu, h, i)
        𝐧_h = sample_visible_normal(α, ω_l, u₁, u₂)
        ω′ = reflect(ω_l, 𝐧_h)
        cosα′ = ω′[3]
        cosα′ <= 0 && continue
        pdf = reflection_pdf(α, 𝐧_h, ω_l)
        ForwardDiff.value(pdf) <= pdf_guard && continue
        Lin = VrtkCore.radiance(lt, 𝐱, local_to_world(t, b, n, ω′[1], ω′[2], ω′[3]))
        acc += cook_torrance(m, ρ, α, η, ω_l, ω′, 𝐧_h) .* Lin * (cosα′ / pdf)
    end
    return acc / s.N
end

"""
    RadianceParts{J, T}

Bilinear decomposition of the estimator: `L_c = Σ_j a[c,j] (μ Bd[c,j] + (1−μ) Bs[c,j])`,
i.e. `Bd`/`Bs` are the diffuse/specular response coefficients of the SH basis (both linear
in Θ_l, the combination affine in μ).
"""
struct RadianceParts{J, T, J3}
    Bd::SMatrix{3, J, T, J3}
    Bs::SMatrix{3, J, T, J3}
end

"""
    radiance_parts(board, mat, light, sampler, key, 𝐛, 𝐱, ω) -> RadianceParts

Same samples and direction convention as [`L_c`](@ref) (ω from the aperture point to the board point);
used to build the design matrices of the linear solves.
"""
function radiance_parts(bd::TexturedTarget, m::Material, lt::EnvironmentLight{<:SphericalHarmonicMap{L, C, J}},
                        s::RadianceSampler, key::SampleKey, 𝐛::StaticVector{2}, 𝐱::StaticVector{3}, ω::StaticVector{3}) where {L, C, J}
    t, b, n = board_frame(bd)
    ω_l = to_local(t, b, n, -ω)                      # outgoing direction ω_o = −ω (ω: aperture point → board point)
    T = promote_type(eltype(ω_l), eltype(m.refractive_index_ink))
    Bd = zero(SMatrix{3, J, T, 3J}); Bs = zero(SMatrix{3, J, T, 3J})
    ω_l[3] <= 0 && return RadianceParts{J, T, 3J}(Bd, Bs)
    ρ = physical_albedo(bd, 𝐛)   # the texture (or the printed edge) at the DESIGN point the physical 𝐛 shows
    wv = texture_weight(ρ)                                                # the texture value blends ink ↔ no ink
    α = 𝛂(m, wv); η = 𝛈(m, wv)
    h = key_hash(s, key)
    Tu = typeof(ForwardDiff.value(zero(T)))
    pdf_guard = eps(Tu) * Tu(10000)
    for i in 1:s.N
        u₁, u₂ = draws(s, Tu, h, i)
        𝐧_h = sample_visible_normal(α, ω_l, u₁, u₂)
        ω′ = reflect(ω_l, 𝐧_h)
        cosα′ = ω′[3]
        cosα′ <= 0 && continue
        pdf = reflection_pdf(α, 𝐧_h, ω_l)
        ForwardDiff.value(pdf) <= pdf_guard && continue
        y = VrtkCore.coefficient_jacobian(lt, 𝐱, local_to_world(t, b, n, ω′[1], ω′[2], ω′[3]))
        d, sp = cook_torrance_terms(ρ, α, η, ω_l, ω′, 𝐧_h)
        w = cosα′ / pdf / s.N
        Bd += (w * d) * y'
        Bs += (w * sp) * y'
    end
    return RadianceParts{J, T, 3J}(Bd, Bs)
end

"""Evaluate `L_c` from parts for coefficients `a` and weight `μ`."""
@inline function L_from_parts(p::RadianceParts{J}, a::SMatrix{3, J}, μ) where {J}
    return SVector(ntuple(c -> sum(a[c, j] * (μ * p.Bd[c, j] + (1 - μ) * p.Bs[c, j]) for j in 1:J), Val(3)))
end
