# ---------------------------------------------------------------------------
# The data support of a spline's nodes: which of them the run's pixels determine. ONE rule for every spline of the
# camera that has one — the radial vignetting's nodes (`vignetting_rule`, 2026-10-30) and the focus surface's TPS nodes
# (`focus_node_support`, 2026-11-15): a node carrying less than `support_min` of the mean DATA WEIGHT is DEPENDENT, continued by the
# spline from the supported ones, never a free parameter.
# ---------------------------------------------------------------------------

"""
    data_weight(basis, px, n) -> Vector{Float64}

The data weight of the `n` nodes of a spline under the masked pixels `px` (per view, Julia pixel positions `(u, v)`): `Σ_pixels b_i(𝐩)²`,
`b = basis(𝐩)` the spline's values per node at the pixel's position (its coefficient Jacobian) — the diagonal of the spline block's
normal matrix, up to the pixels' weights.
"""
function data_weight(basis, px, n::Integer)
    w = zeros(n)
    for list in px, (u, q) in list
        b = basis(𝐩_of_pixel(u, q))
        for i in 1:n; w[i] += b[i] * b[i]; end
    end
    return w
end

"""
    node_support(weight, support_min) -> (; weight, supported)

The support rule of a spline's nodes from their data weights `weight` (`data_weight`): the weights relative to their mean
(`weight`), and `supported[i]` when node `i` carries at least `support_min` of the mean (and any weight at all; `support_min = 0`: one
pixel suffices). The unsupported nodes are DEPENDENT — what continues them is the spline's (a lattice's boundary rule, a scattered-node
spline's interpolant of the supported nodes).
"""
function node_support(weight::AbstractVector{<:Real}, support_min::Real)
    n = length(weight)
    rel = sum(weight) > 0 ? weight ./ (sum(weight) / n) : float.(weight)
    return (; weight = Vector{Float64}(rel), supported = BitVector(rel .>= max(Float64(support_min), 0.0) .&& weight .> 0))
end
