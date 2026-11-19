# ---------------------------------------------------------------------------
# The photosite sample's edge strength (owner design, 2026-11-15): `[masks] edge_detector` gives each usable observation of a view an
# edge strength g, `edge_rank` its ECDF rank u (ties at their mean rank) over the view's usable observations (rank-based: invariant to the photograph's contrast
# and exposure), and the draw (`draw_samples`, masks.jl) weights each by u^α (`[masks] gradient_bias = α`). The detector, a table
# dispatched by its `type` (one type in RC 0.11.4: the L0 band and the importance weights of the sweep were left out, no gain):
#   {type = "gaussian_derivative", sigma}   g = |∇(G_σ ∗ photograph)|: Features' `gaussian_gradient` (Förstner's Gaussian-derivative
#                                           filters), NORMALIZED over the usable observations (its mask: a RAW photograph's green
#                                           photosites alone, the red and blue sites and the defects never read)
# It reads the PHOTOGRAPH (the observed values the residuals compare against; the rendered target needs the model the sample feeds).
# ---------------------------------------------------------------------------

"""An edge detector of the photosite sample (`[masks] edge_detector`): [`GaussianDerivativeEdges`](@ref)."""
abstract type AbstractEdgeDetector end

"""
    GaussianDerivativeEdges(sigma)

g = |∇(G_σ ∗ image)| at scale `sigma` (pixels): Features' `gaussian_gradient` on the observations, normalized over them.
"""
struct GaussianDerivativeEdges <: AbstractEdgeDetector
    sigma::Float64
end

"The keys of each `[masks] edge_detector` type, every one required (no default inside a stated table)."
const EDGE_DETECTOR_KEYS = (gaussian_derivative = ("type", "sigma"),)
"The retired `[masks] edge_detector` types, each refused naming why."
const RETIRED_EDGE_DETECTORS = (l0 = "retired (owner / controller ruling, 2026-11-15): the L0 band detector gave no gain on the fisheye sweep and costs ≈ 1.8 s per view; use type = \"gaussian_derivative\"",)

"""
    edge_detector_from_toml(v, key) -> AbstractEdgeDetector

The `[masks] edge_detector` table `v`: dispatched by its `type` (`EDGE_DETECTOR_KEYS`), each of the type's keys required, an unknown
one refused, every scale and weight positive.
"""
function edge_detector_from_toml(v, key)
    typed_table_type(v, key, EDGE_DETECTOR_KEYS; retired = RETIRED_EDGE_DETECTORS, shape = "{type = \"gaussian_derivative\", sigma}")   # toml.jl's one reader
    pos(k) = (x = VgtkCore.convert_field(Float64, v[k], "$key.$k"); isfinite(x) && x > 0 || throw(ArgumentError("`$key.$k` = $x: a positive number")); x)
    return GaussianDerivativeEdges(pos("sigma"))
end
"The `[masks] edge_detector` table of a detector (the reader's inverse)."
edge_detector_table(d::GaussianDerivativeEdges) = Dict{String, Any}("type" => "gaussian_derivative", "sigma" => d.sigma)

"""
    edge_plane(img, channels) -> Matrix{Float32}

The plane the edge detector reads, the one host copy of the photograph the detector makes (RC 0.11.5, the 0.11.4 review's item 7b:
before, 4–5 full-frame copies per view), in the detector's `Float32`: an `H × W` image itself (a RAW photograph's linear mosaic, whose
usable green photosites the detector's support keeps), an `H × W × C` image's measured `channels` summed (in `Float64`, in channel
order); an uncertain image's values (Core's `value_of`). Each value `Float32(Float64(x))`: RC ≤ 0.11.4's `Float64` plane converted
on upload, bit for bit.
"""
edge_plane(img::AbstractMatrix, channels) = map(x -> Float32(Float64(VgtkCore.value_of(x))), img)
function edge_plane(img::AbstractArray{<:Any, 3}, channels)
    cs = collect(channels)
    return [Float32(_channel_sum(img, i, j, cs)) for i in axes(img, 1), j in axes(img, 2)]
end
_channel_sum(img, i, j, cs) = (s = 0.0; for c in cs; s += Float64(VgtkCore.value_of(img[i, j, c])); end; s)

"""
    edge_strength(detector, plane, support; device) -> Matrix{Float32}

Each pixel's edge strength g (`H × W`, on the host) of the image `plane` (`edge_plane`) read over `support` (its usable observations),
on `device`: the plane crosses to the device once (none on the CPU: the plane itself), the support as its `Bool`s (one byte per pixel, Core's
narrow upload, converted there), g is formed there and comes back once (none on the CPU).
"""
function edge_strength(d::GaussianDerivativeEdges, plane::AbstractMatrix{<:Real}, support::AbstractMatrix{Bool}; device = VgtkCore.default_device())
    a = VgtkCore.on_device(Float32, device, plane)
    m = VgtkCore.on_device(Float32, device, Matrix{Bool}(support))   # a BitMatrix has no device backend: its Bools (1 B/px)
    gx, gy = VgtkFeatures.gaussian_gradient(a, d.sigma, m)      # the normalized convolution over the support (gaussian_blur's)
    return VgtkCore.on_device(KernelAbstractions.CPU(), hypot.(gx, gy))
end

"""
    edge_rank(g, idx) -> Vector{Float64}

The ECDF rank u ∈ (0, 1] of the edge strengths `g[idx]` among themselves, in `idx`'s order: the mid-distribution ECDF
(F(g⁻) + F(g)) / 2 + 1/(2n) = StatsBase's `tiedrank` / n — u = #{g_j ≤ g_i} / n (StatsBase's `ecdf`) for distinct strengths, a tied
group its mean rank (flat zeros rank at their middle, not at the top of the group). Invariant to any increasing map of g — the
photograph's contrast and exposure.
"""
edge_rank(g::AbstractMatrix{<:Real}, idx::AbstractVector) = (x = Float64[g[i] for i in idx]; StatsBase.tiedrank(x) ./ length(x))
