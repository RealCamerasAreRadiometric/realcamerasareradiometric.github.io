# ---------------------------------------------------------------------------
# The smoothness of a spline component — the grid's nodal directions, the vignetting profile's nodes, the focus surface's
# nodal values — one vocabulary for the three (the owner, 2026-11-14), in the options and in the code: a `[<component>.smoothness]`
# table dispatched by its `type` (the stack's `{type, …}` convention), no flag.
#
#   [grid.smoothness]       type = "pspline", order = 3, weight = 0.1    (weight may be "gcv")
#   [vignetting.smoothness] type = "pspline", order = 3, weight = 0.0    (weight may be "gcv")
#   [focus.smoothness]      type = "bending", weight = 0.0
#
# `pspline`: the `order`-th differences over the component's lattice (Eilers & Marx's P-spline penalty); `bending`: the analytic
# bending energy; `none`. The weight β is relative to the data energy (Core's `penalty_scale`), so it means the same for every
# operator, lattice and region; "gcv" chooses it by generalized cross-validation where the component supports it.
# ---------------------------------------------------------------------------

"""The smoothness of a spline component (`[<component>.smoothness]`): `PSplineSmoothness`, `BendingSmoothness`, `NoSmoothness`."""
abstract type AbstractSmoothness end
"""`type = "pspline"`: the `order`-th differences of the component over its lattice, weight `weight` (β ≥ 0, or `:gcv`)."""
struct PSplineSmoothness <: AbstractSmoothness
    order::Int
    weight::Union{Float64, Symbol}
end
"""`type = "bending"`: the analytic bending energy ∫∫ f_xx² + 2 f_xy² + f_yy² of the component, weight `weight` (β ≥ 0, or `:gcv`)."""
struct BendingSmoothness <: AbstractSmoothness
    weight::Union{Float64, Symbol}
end
"""`type = "none"`: no smoothness."""
struct NoSmoothness <: AbstractSmoothness end

"""The smoothness types by their `type` tag."""
const SMOOTHNESS_TYPES = (pspline = PSplineSmoothness, bending = BendingSmoothness, none = NoSmoothness)
"""The `type` tag of a smoothness (`:pspline`, `:bending`, `:none`): what its records and log lines name."""
smoothness_type(::Type{PSplineSmoothness}) = :pspline
smoothness_type(::Type{BendingSmoothness}) = :bending
smoothness_type(::Type{NoSmoothness}) = :none
smoothness_type(s::AbstractSmoothness) = smoothness_type(typeof(s))
"""The weight β of a smoothness (0 for none; `:gcv` when chosen by GCV)."""
smoothness_weight(s::Union{PSplineSmoothness, BendingSmoothness}) = s.weight
smoothness_weight(::NoSmoothness) = 0.0
"""Whether a smoothness penalizes anything: a type other than none, of a weight other than 0."""
smoothing(s::AbstractSmoothness) = !(s isa NoSmoothness) && smoothness_weight(s) != 0
"""The smoothness's log line: `pspline order 3, weight 0.1`, `bending, weight 0`, `none`."""
smoothness_line(s::PSplineSmoothness) = "pspline order $(s.order), weight $(s.weight === :gcv ? "gcv" : s.weight)"
smoothness_line(s::BendingSmoothness) = "bending, weight $(s.weight === :gcv ? "gcv" : s.weight)"
smoothness_line(::NoSmoothness) = "none"
"""A smoothness as a record's fields (`type`, `order` — a pspline's, else `nothing` — and `weight`, a number or `"gcv"`): what the
run records write, the `[<component>.smoothness]` table's keys."""
smoothness_record(s::AbstractSmoothness) = (; type = String(smoothness_type(s)), order = s isa PSplineSmoothness ? s.order : nothing,
                                             weight = (w = smoothness_weight(s); w === :gcv ? "gcv" : w))
"""The same smoothness of another weight (the one GCV chose)."""
with_weight(s::PSplineSmoothness, w) = PSplineSmoothness(s.order, w)
with_weight(s::BendingSmoothness, w) = BendingSmoothness(w)
with_weight(s::NoSmoothness, w) = s

"""
    focus_smoothness!(s, f::Focus, ϕ², sm::AbstractSmoothness) -> row_weight

The bending smoothness rows of the run from `[focus.smoothness]` (its weight β), written into the state `s` (its `focus_smoothness`,
every focus block's smoothness residual reads it; none for β = 0): `√(β Σϕ²/N) L/√m̄`, `L` Core's `bending_rows` of the focus
spline's system (one row per positive eigenvalue of the bending energy Q) and m̄ Core's `mean_positive_eigenvalue` of it:
`‖L t̂‖²/m̄ = t̂ᵀQt̂/m̄`, the bending energy relative to its mean positive eigenvalue. Returns the row weight β Σϕ²/N (0 for none).
"""
function focus_smoothness!(s, f::Focus, ϕ²::Real, sm::AbstractSmoothness)
    β = Float64(smoothness_weight(sm))
    β > 0 || return 0.0
    N = length(f.t̂_i)
    Lc = VgtkCore.bending_rows(FocusSpline(f).sys)
    L = sqrt(β * ϕ² / N / VgtkCore.mean_positive_eigenvalue(Lc)) .* Lc
    s.focus_smoothness = L
    @printf("focus smoothness %s: row weight β Σϕ²/N %.4g, start smoothness cost %.4g\n", smoothness_line(sm), β * ϕ² / N, 0.5 * sum(abs2, L * collect(f.t̂_i))); flush(stdout)
    return β * ϕ² / N
end
