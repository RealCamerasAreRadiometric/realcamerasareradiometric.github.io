# ---------------------------------------------------------------------------
# The image-formation integrals' quadrature: the sensor footprint ℐ_𝐮 (the pixel box), the aperture disc 𝒜 and its
# border ∂𝒜, plus low-discrepancy points for the Monte-Carlo radiance estimator. The rules are VisualGeometryToolkitCore's
# (numerics/quadrature.jl): the tensor rule over a square (`square_rule`) of QuadGK's Gauss–Legendre `gauss(n)`, the polar rule
# over the unit disc (`unit_disc_rule`), the Hammersley points (`hammersley([T,] i, N)`).
# What is RadiometricCamera's: the footprint in pixel fractions, the aperture's border samples, and the
# statically sized (isbits) containers the KernelAbstractions kernels take unchanged.
# ---------------------------------------------------------------------------
using VisualGeometryToolkitCore: hammersley, square_rule, unit_disc_rule

"""
    SensorQuadrature{T, NS}

Tensor Gauss–Legendre rule over one pixel footprint ℐ_𝐮 (`NS` nodes). `offsets[k]` are
positions inside the footprint in pixels from its centre (each component in
`(-0.5, 0.5)`), `weights[k]` sum to **one**: the rule computes the mean of the
integrand over ℐ_𝐮, i.e. the paper's `∫_{ℐ_𝐮} … d𝐩` over the footprint's area of one
square pixel.
"""
struct SensorQuadrature{T, NS}
    offsets::SVector{NS, SVector{2, T}}
    weights::SVector{NS, T}
end

"""
    SensorQuadrature{T}(n = 2)

The tensor `n`-point Gauss–Legendre rule over the pixel box.
"""
function SensorQuadrature{T}(n::Integer = 2) where {T}
    x, w = gauss(n); x = x ./ 2; w = w ./ 2
    nodes, weights = square_rule(x, w)                      # Core's tensor rule over the square: (x[i], x[j]), w[i]·w[j]
    NS = n * n
    return SensorQuadrature{T, NS}(SVector{NS}(SVector{2, T}.(nodes)), SVector{NS}(T.(weights)))
end
SensorQuadrature(n::Integer = 2) = SensorQuadrature{Float64}(n)
Base.length(::SensorQuadrature{T, NS}) where {T, NS} = NS

"""Sensor position (COLMAP pixels) of sub-sample `k` of pixel `(u, v)`."""
@inline 𝐩_sample(q::SensorQuadrature, k::Integer, u::Integer, v::Integer) = 𝐩_of_pixel(u + q.offsets[k][1], v + q.offsets[k][2])

"""
    ApertureQuadrature{T, NA, NB}

Polar rule over the unit disc `{‖â‖ ≤ 1}`: Gauss–Legendre in the radial
coordinate ρ̃ ∈ [0, 1] (`n_ρ` nodes) × uniform angles φ (`n_φ` nodes), `NA = n_ρ·n_φ`
disc nodes. For an aperture of radius `r` the nodes are `𝐚 = r â` and the weights
`r² w̃` (`Σ w̃ = π`, so the rule integrates over the disc area exactly like the paper's
`∫_𝒜 … d𝐚`). `φ_border` / `w_border` (`NB = n_φ`) sample the border ∂𝒜 (the mask's circle-of-confusion test).
"""
struct ApertureQuadrature{T, NA, NB}
    â::SVector{NA, SVector{2, T}}
    w̃::SVector{NA, T}
    φ_border::SVector{NB, T}
    w_border::SVector{NB, T}
end

function ApertureQuadrature{T}(n_ρ::Integer = 3, n_φ::Integer = 8) where {T}
    nodes, weights = unit_disc_rule(n_ρ, n_φ)              # Core's polar rule over the unit disc (Gauss–Legendre radius × uniform angles)
    â = SVector{2, T}.(nodes); w̃ = T.(weights)
    φ = [2π * (j - 0.5) / n_φ for j in 1:n_φ]              # the border's samples: the rule's angles
    NA = n_ρ * n_φ
    return ApertureQuadrature{T, NA, n_φ}(SVector{NA}(â), SVector{NA}(w̃),
                                          SVector{n_φ}(T.(φ)), SVector{n_φ}(fill(T(2π / n_φ), n_φ)))
end
ApertureQuadrature(n_ρ::Integer = 3, n_φ::Integer = 8) = ApertureQuadrature{Float64}(n_ρ, n_φ)
"""
    n_disc(q::ApertureQuadrature) -> Int

The number of nodes of the rule on the unit disc 𝒟 (`n_ρ × n_φ`).
"""
n_disc(::ApertureQuadrature{T, NA}) where {T, NA} = NA
"""
    n_border(q::ApertureQuadrature) -> Int

The number of samples of the aperture's border ∂𝒜 (`n_φ`).
"""
n_border(::ApertureQuadrature{T, NA, NB}) where {T, NA, NB} = NB

"""Aperture point `𝐚_l = r â_l` (aperture plane z = 0 of the camera frame) of disc node `l`."""
@inline 𝐚_sample(q::ApertureQuadrature, l::Integer, r) = SVector(r * q.â[l][1], r * q.â[l][2], zero(r))
"""Disc weight `r² w̃_l` of node `l`."""
@inline 𝐚_weight(q::ApertureQuadrature, l::Integer, r) = r^2 * q.w̃[l]

"""
    𝐚_border(q, j, r) -> (𝐚(φ_j), 𝐯(φ_j), 𝐧(φ_j), ds_j)

Polar parameterisation of ∂𝒜: `𝐚(φ) = r (cos φ, sin φ, 0)`, velocity
`𝐯 = ∂𝐚/∂r = (cos φ, sin φ, 0)`, outward normal `𝐧 = 𝐚/‖𝐚‖` and line element
`ds = r dφ` (the weight already contains dφ).
"""
@inline function 𝐚_border(q::ApertureQuadrature, j::Integer, r)
    s, c = sincos(q.φ_border[j])
    return SVector(r * c, r * s, zero(r)), SVector(c, s, zero(r)), SVector(c, s, zero(r)), r * q.w_border[j]
end

# host-side convenience iterators (tests, mask computation)
"""
    sensor_samples(q::SensorQuadrature, u, v)

An iterator of `(𝐩_k, w_k)` over the sensor rule's nodes on the pixel 𝓘_𝐮 of Julia's 1-based pixel `(u, v)`: the sensor positions
in COLMAP pixels ([`𝐩_sample`](@ref)) and their weights. A host-side convenience; the renderers index the rule directly.
"""
sensor_samples(q::SensorQuadrature, u::Integer, v::Integer) = ((𝐩_sample(q, k, u, v), q.weights[k]) for k in 1:length(q))
"""
    aperture_samples(q::ApertureQuadrature, r)

An iterator of `(𝐚_l, r² w̃_l)` over the disc rule's nodes on the aperture 𝒜 of radius `r` ([`𝐚_sample`](@ref),
[`𝐚_weight`](@ref)); `r` in the camera frame's units (metres).
"""
aperture_samples(q::ApertureQuadrature, r) = ((𝐚_sample(q, l, r), 𝐚_weight(q, l, r)) for l in 1:n_disc(q))
"""
    aperture_border(q::ApertureQuadrature, r)

An iterator of [`𝐚_border`](@ref)`(q, j, r)` over the samples of the aperture's border ∂𝒜 of radius `r`.
"""
aperture_border(q::ApertureQuadrature, r) = (𝐚_border(q, j, r) for j in 1:n_border(q))

