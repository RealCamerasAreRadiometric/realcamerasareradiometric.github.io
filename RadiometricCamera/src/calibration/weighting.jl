# ---------------------------------------------------------------------------
# The residuals' weighting: whitened least squares under the photographs' sensor noise (D14; owner ruling 2026-11-14: no flag).
#
# A RAW photograph whose sensor model has a noise with a variance law (a gain: PoissonGaussian, or a RelativeNoise) is read as Core's
# uncertain image (`linearview` under the model, each value with its scalar variance σ², D11/D14) and contributes its rows divided by
# σ: the residual `(Φ − ϕ)/σ` and its Jacobian row `J/σ` (whitened residuals; the cost Σ((Φ − ϕ)/σ)² the negative log-likelihood up
# to a constant). A plain image (a decoded photograph, or a RAW one whose model has no variance law: no characterisation, or a
# ReadNoise alone) contributes its rows as measured, today's behaviour bit for bit. The choice is what the camera's sensor model
# provides, by TYPE, never a flag: set once where the photographs are read (`masked_observations`), the rows' weights are
# `Unweighted()` or the per-row 1/σ vector, and every consumer — both LM routes' row kernel and host residual, the closed-form blocks'
# reductions and design rows, the cost, the penalties' data norm and the LM's residual resolution — reads them through `_whiten`,
# THE one weighting.
# ---------------------------------------------------------------------------

"""
    Unweighted()

The weights of rows taken as measured (a plain image): every row's weight is one, and [`_whiten`](@ref) is the identity, so the
residuals, Jacobians and costs are today's bit for bit. The other weights are a vector of 1/σ per row (a whitened view).
"""
struct Unweighted end

"""
    _whiten(w, e, x)
    _whiten(w, x::AbstractArray)

THE weighting of the residual rows: row `e`'s value (a residual, a Jacobian row, a design row, a measured value) `x` times its weight
`w[e]` (1/σ of its measured value) when `w` is a weight vector, `x` itself when `w` is `Unweighted()`. The array form weights every
row of `x`, a vector or a matrix of rows (host or device broadcast). In a kernel the same expression is the row kernel's.
"""
@inline _whiten(::Unweighted, e, x) = x
@inline _whiten(w::AbstractVector, e, x) = (@inbounds w[e]) * x
_whiten(::Unweighted, x::AbstractArray) = x
_whiten(w::AbstractVector, x::AbstractArray) = w .* x
# a sparse matrix of rows (the texture page's design rows): its row scaling, sparse (the broadcast of a vector against a sparse matrix
# allocates for the full shape: a page's 7 M rows × 1 M texels ran out of host memory, dgx01 2026-11-15); the same products, bit for bit
_whiten(w::AbstractVector, x::SparseArrays.AbstractSparseMatrixCSC) = Diagonal(w) * x

"""
    masked_observations(img, px; channel = 0) -> (ϕ, w, q)

The measured values `ϕ` of the masked pixels `px` of `img` (`masked_vector`'s order) and their rows' weights `w`, by the image's
element TYPE: a plain image's (`Real` elements) weights are `Unweighted()`; an uncertain image's (Core's `Uncertain` elements: a RAW
photograph read under a sensor model with a gain, `observe`) values are their `value_of` and their weights 1/σ, σ² the
element's variance (`Statistics.var`, Core's law, D11) — the variance must be finite and positive at every masked photosite. `q`:
each row's code step (`_quanta`), what the data resolve (`lm_abstol`).
"""
masked_observations(img::AbstractArray{<:Real}, px::AbstractVector; channel::Integer = 0) =
    (masked_vector(img, px; channel), Unweighted(), _quanta(img, px))
function masked_observations(plane::AbstractMatrix{<:VgtkCore.Attributed}, px::AbstractVector; channel::Integer = 0)
    channel > 0 || throw(ArgumentError("masked_observations: a single plane is one measurement per pixel: a single-channel state (channel 1–3)"))
    x = [plane[v, u] for (u, v) in px]
    σ² = Float64[Statistics.var(p) for p in x]
    all(v -> isfinite(v) && v > 0, σ²) ||
        throw(ArgumentError("masked_observations: an uncertain photosite's variance is not finite and positive (min $(minimum(σ²; init = Inf)))"))
    return Float64[VgtkCore.value_of(p) for p in x], 1 ./ sqrt.(σ²), _quanta(plane, px)
end

"""
    _quanta(img, px) -> Union{Nothing, Vector{Float64}}

Each masked pixel's code step: one RAW code in the value's units, `1 / (white − black[c])` of its photosite's channel (Core's one
linearisation, `linearview`), for a RAW photograph's linear view (plain or uncertain): Core's `linear_quantum`; `nothing` for a
decoded image, which has no code (it resolves to its float precision: Core's own resolution).
"""
_quanta(::AbstractArray, px::AbstractVector) = nothing
_quanta(v::VgtkCore.LinearRawView, px::AbstractVector) = Float64[VgtkCore.linear_quantum(v, r, c) for (c, r) in px]

"""
    _view_weights(weights, k)

View `k`'s row weights from a state's `weights` (`Unweighted()` for every view, or one 1/σ vector per view).
"""
_view_weights(::Unweighted, k::Integer) = Unweighted()
_view_weights(w::AbstractVector{<:AbstractVector}, k::Integer) = w[k]
"""The weights of the rows `rows` of one view's (a block's subset, its observations' order; `Unweighted()` as it is)."""
_select_rows(w::Unweighted, rows) = w
_select_rows(w::AbstractVector, rows) = w[rows]
"""Weights on the backend `be`, shared read-only like a block's other device data (`Unweighted()` as it is)."""
_device_weights(be, w::Unweighted) = w
_device_weights(be, w::AbstractVector) = _shared_read_only!(be, adapt(be, w))
"""The weights of the views' rows stacked in view order (the closed-form blocks' rows, `all_parts`)."""
_stacked_weights(::Unweighted) = Unweighted()
_stacked_weights(w::AbstractVector{<:AbstractVector}) = reduce(vcat, w; init = Float64[])

"""
    data_norm2(s) -> Float64

The squared norm of the measured values in the cost's units, `Σ ‖w ∘ ϕ‖²` over every view (`Σ ϕ²` unweighted): the data scale the
penalties' weights are relative to (Core's `penalty_scale`: the grid, focus and vignetting smoothness) and the
relative RMS's denominator (`cost`).
"""
data_norm2(s) = sum(k -> sum(abs2, _whiten(_view_weights(s.weights, k), s.ϕ[k])), eachindex(s.ϕ); init = 0.0)

"""
    lm_abstol(s) -> Float64

The absolute residual-norm tolerance (SciML's `abstol`) in the cost's units, which Core's `LmStops(; residual_resolution)` reads (the
meaning of `StaticLmStops`'): the RAW quantisation the data resolve, never the noise level (owner ruling 2026-11-14). A row's resolution is its code step `q` (`_quanta`, one
RAW code in the row's units) over √12, so the resolution's square is `Σᵢ (qᵢ wᵢ)² / 12` — `wᵢ = 1/σᵢ` for whitened rows, `1` for rows
as measured — through the one weighting (`_whiten`); `0` for views without a code (decoded images: Core's own resolution alone). A fit
at the noise level (χ² ≈ m) is then far above it: no exact fit (Core's stop rules decide its termination).
"""
lm_abstol(s) = sqrt(sum(k -> _resolution2(_view_weights(s.weights, k), s.quanta[k]), eachindex(s.ϕ); init = 0.0))
# a view's Σ (q·w)² / 12 (`_whiten`: whitened rows or as measured); a view without code steps (a decoded image) resolves exactly
_resolution2(w, ::Nothing) = 0.0
_resolution2(w, q::AbstractVector) = sum(abs2, _whiten(w, q)) / 12
