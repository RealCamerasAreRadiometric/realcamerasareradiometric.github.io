# ---------------------------------------------------------------------------
# Spatially varying Cook–Torrance BSDF (paper §SVBSDF and Monte Carlo Estimation):
#
#   f_c^CT(𝐱, ω, ω′; Θ_m) = μ ρ_c(𝐱)/π
#        + (1−μ) F_c(ω, ω_h; η_c) D(ω_h; α) G_2(ω, ω′) / (4 cos θ(ω) cos θ(ω′))
#
#   D    GGX normal distribution,  G_2 height-correlated Smith masking–shadowing,
#   F_c  exact unpolarised dielectric Fresnel term with refractive index η_c,
#   w(𝐛) = texture value = luma(ρ(𝐛)) ∈ [0, 1] (1 = no ink),  α(𝐛) = w α_¬ink + (1−w) α_ink,  η(𝐛) = w η_¬ink + (1−w) η_ink
#   (paper: "using the texture value to blend between α_ink and α_¬ink, η_ink and η_¬ink" — the coverage of a
#   printed texel is linear in its albedo, so the microfacet mix is linear in the texture value)
#
#   Θ_m = {α_¬ink, α_ink, η_¬ink, η_ink, μ}: Θ_m ∖ {μ} = 8 nonlinear parameters (η per channel), μ affine in the flux.
#
# Microfacet normals are sampled from the GGX distribution of visible normals
# D^v(𝐧, ω; α) with the spherical-caps method (Dupuy & Benyoub 2023) and turned
# into directions by the reflection ω′ = 2(ω⋅𝐧)𝐧 − ω, PDF(𝐧, ω; α) = D^v/(4 ω⋅𝐧).
# All directions in this file are in the LOCAL board frame (z = board normal).
# ---------------------------------------------------------------------------

"""
    Material{T}

`Θ_m`: the roughness endpoints `𝛂_ink`, `𝛂_noink` (= `α_ink`, `α_¬ink`), the refractive indices `𝛈_ink`, `𝛈_noink` (= `η_ink`, `η_¬ink`, per
channel) and the diffuse/specular weight `μ`; the ink ↔ no-ink blend weight of a board point is its texture value (`texture_weight`).
"""
struct Material{T}
    𝛂_ink::T
    𝛂_noink::T
    𝛈_ink::SVector{3, T}
    𝛈_noink::SVector{3, T}
    μ::T
end
function Material(𝛂_ink, 𝛂_noink, 𝛈_ink::AbstractVector, 𝛈_noink::AbstractVector, μ)
    T = promote_type(typeof(𝛂_ink), typeof(𝛂_noink), eltype(𝛈_ink), eltype(𝛈_noink), typeof(μ))
    return Material{T}(T(𝛂_ink), T(𝛂_noink), SVector{3, T}(𝛈_ink), SVector{3, T}(𝛈_noink), T(μ))
end
Material(; 𝛂_ink = 0.15, 𝛂_noink = 0.4, 𝛈_ink = 1.6, 𝛈_noink = 1.33, μ = 0.5) =
    Material(𝛂_ink, 𝛂_noink, SVector(𝛈_ink, 𝛈_ink, 𝛈_ink) .* 1.0, SVector(𝛈_noink, 𝛈_noink, 𝛈_noink) .* 1.0, μ)

"""`Θ_m ∖ {μ}` as an `SVector{8}` (order `𝛂_ink, 𝛂_noink, 𝛈_ink[1:3], 𝛈_noink[1:3]`)."""
Θ_m_nonlinear(m::Material{T}) where {T} = SVector{8, T}(m.𝛂_ink, m.𝛂_noink, m.𝛈_ink..., m.𝛈_noink...)
"""
    with_nonlinear(m::Material, θ::StaticVector{8}) -> Material

The material `m` with its nonlinear parameters `Θ_m ∖ {μ}` replaced by `θ`, in the order of [`Θ_m_nonlinear`](@ref)
(`𝛂_ink, 𝛂_noink, 𝛈_ink[1:3], 𝛈_noink[1:3]`); `μ` is kept, so `with_nonlinear(m, Θ_m_nonlinear(m)) == m`.
"""
with_nonlinear(m::Material, θ::StaticVector{8}) = Material(θ[1], θ[2], SVector(θ[3], θ[4], θ[5]), SVector(θ[6], θ[7], θ[8]), m.μ)
"""
    with_μ(m::Material, μ) -> Material

The material `m` with the diffuse weight `μ` (dimensionless, `f_c^CT = μ ρ_c/π + (1 − μ)` × the specular term); the roughnesses and
the indices of refraction are kept.
"""
with_μ(m::Material, μ) = Material(m.𝛂_ink, m.𝛂_noink, m.𝛈_ink, m.𝛈_noink, μ)
VgtkCore.convert_eltype(::Type{T}, m::Material) where {T} = Material(T(m.𝛂_ink), T(m.𝛂_noink), SVector{3, T}(m.𝛈_ink), SVector{3, T}(m.𝛈_noink), T(m.μ))

"""Ink ↔ no-ink blend weight of a board point: the texture value `w = luma(ρ(𝐛))` clamped to [0, 1] (1 = no ink, 0 = ink)."""
@inline texture_weight(ρ::StaticVector{3}) = clamp(luma(ρ), 0, 1)
"""Spatially varying roughness `α(𝐛) = w α_¬ink + (1−w) α_ink` at the blend weight `w = texture_weight(ρ)`."""
@inline 𝛂(m::Material, w) = w * m.𝛂_noink + (1 - w) * m.𝛂_ink
"""Spatially varying refractive indices `η_c(𝐛) = w η_¬ink + (1−w) η_ink` at the blend weight `w`."""
@inline 𝛈(m::Material, w) = w * m.𝛈_noink + (1 - w) * m.𝛈_ink

"""GGX normal distribution `D(ω_h; α)` (`cosθ_h = ω_h ⋅ 𝐧`)."""
@inline function D_ggx(α, cosθ_h)
    cosθ_h <= 0 && return zero(cosθ_h * α)
    t = cosθ_h^2 * (α^2 - 1) + 1
    return α^2 / (π * t^2)
end

"""Smith Λ for GGX at `cosθ = ω ⋅ 𝐧`."""
@inline function Λ_ggx(α, cosθ)
    T = typeof(α * cosθ)
    μ² = max(cosθ^2, sqrt(eps(T)) * T(1//1000))
    return (sqrt(1 + α^2 * (1 - μ²) / μ²) - 1) / 2
end
"""`G_1(ω) = 1/(1 + Λ(ω))`."""
@inline G1_ggx(α, cosθ) = 1 / (1 + Λ_ggx(α, cosθ))
"""Height-correlated `G_2(ω, ω′) = 1/(1 + Λ(ω) + Λ(ω′))` (Heitz 2014)."""
@inline G2_ggx(α, cosθ, cosθ′) = 1 / (1 + Λ_ggx(α, cosθ) + Λ_ggx(α, cosθ′))

"""
    F_dielectric(η, c) 

Exact unpolarised dielectric Fresnel reflectance for `c = |ω ⋅ ω_h|` and refractive index `η`.
"""
@inline function F_dielectric(η, c)
    g² = η^2 - 1 + c^2
    g² <= 0 && return one(g²)                    # total internal reflection
    g = sqrt(g²)
    a = (g - c) / (g + c)
    b = (c * (g + c) - 1) / (c * (g - c) + 1)
    return a^2 / 2 * (1 + b^2)
end

"""
    sample_vndf_caps(α, ω, u₁, u₂) -> 𝐧

Microfacet normal `𝐧 ~ D^v(⋅, ω; α)` by the spherical-caps method (local frame, `ω_z > 0`).
"""
@inline function sample_vndf_caps(α, ω::StaticVector{3}, u₁, u₂)
    i = SVector(α * ω[1], α * ω[2], ω[3])
    i = i / sqrt(dot(i, i))
    φ = 2 * typeof(u₁)(π) * u₁
    z = (1 - u₂) * (1 + i[3]) - i[3]
    s = sqrt(max(1 - z^2, eps(typeof(ForwardDiff.value(z)))))       # ε floor: keeps the derivative finite at the cap pole
    o = SVector(s * cos(φ), s * sin(φ), z)
    m = i + o
    n = SVector(α * m[1], α * m[2], m[3])
    return n / sqrt(dot(n, n))
end

"""Visible-normal density `D^v(𝐧, ω; α) = G_1(ω) max(0, ω⋅𝐧) D(𝐧) / cosθ(ω)`."""
@inline function D_v(α, 𝐧::StaticVector{3}, ω::StaticVector{3})
    ωn = dot(ω, 𝐧)
    return G1_ggx(α, ω[3]) * max(ωn, zero(ωn)) * D_ggx(α, 𝐧[3]) / ω[3]
end

"""PDF of the reflected direction ω′ = 2(ω⋅𝐧)𝐧 − ω: `D^v(𝐧, ω; α)/(4 ω⋅𝐧)`."""
@inline PDF_reflect(α, 𝐧::StaticVector{3}, ω::StaticVector{3}) = D_v(α, 𝐧, ω) / (4 * dot(ω, 𝐧))

"""Reflection of ω about the microfacet normal 𝐧."""
@inline reflect(ω::StaticVector{3}, 𝐧::StaticVector{3}) = 2 * dot(ω, 𝐧) * 𝐧 - ω

"""
    f_CT_parts(ρ, α, η, ω, ω′, 𝐧_h) -> (diffuse::SVector{3}, specular::SVector{3})

The two terms of `f_c^CT` **without** the μ weights: `ρ_c/π` and
`F_c D G_2 / (4 cosθ cosθ′)`, so `f_c^CT = μ ⋅ diffuse + (1−μ) ⋅ specular`.
"""
@inline function f_CT_parts(ρ::StaticVector{3}, α, η::StaticVector{3}, ω::StaticVector{3}, ω′::StaticVector{3}, 𝐧_h::StaticVector{3})
    cosθ = ω[3]; cosθ′ = ω′[3]
    c = abs(dot(ω, 𝐧_h))
    ϵ = sqrt(eps(typeof(ForwardDiff.value(cosθ))))
    k = D_ggx(α, 𝐧_h[3]) * G2_ggx(α, cosθ, cosθ′) / (4 * max(cosθ, ϵ) * max(cosθ′, ϵ))
    spec = SVector(F_dielectric(η[1], c), F_dielectric(η[2], c), F_dielectric(η[3], c)) * k
    return ρ / π, spec
end

"""`f_c^CT` for all channels."""
@inline function f_CT(m::Material, ρ, α, η, ω, ω′, 𝐧_h)
    d, s = f_CT_parts(ρ, α, η, ω, ω′, 𝐧_h)
    return m.μ * d + (1 - m.μ) * s
end
