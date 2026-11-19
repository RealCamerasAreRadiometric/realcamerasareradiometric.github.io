# ---------------------------------------------------------------------------
# Focus surface Θ_f = {t̂_i}: the INVERSE focus distance t̂ = 1/𝔱 is modelled by
# a thin plate spline over the nodes of an equally spaced grid {𝐩_i} that can see light: the 5 × 5 grid over the bounding box of the
# run's imaged region (the sensor ℐ = [0, W] × [0, H] in COLMAP pixels, or the sensor ∩ the image circle's box), kept where the start
# lens images them inside the region (`focus_start_nodes`: the CVT of the imaged region since 2026-11-15; `_focus_support`, start/prepare.jl; 2026-10-31 — before, the sensor grid with
# every node a parameter). `Focus(n, 𝔱, W, H)` below builds the plain sensor grid (tests, sessions). Paper:
# "we model the depth 𝔱 not directly, instead we compute the inverse depth
# t̂: ℐ → ℝ, 𝐩 ↦ 1/t(𝐩)"), so that
#
#   𝔱(𝐩) = 1 / t̂(𝐩),      𝔣(𝐩) = 𝔭 + 𝔱(𝐩) π⁻¹(𝐩)   (see backprojection.jl)
# ---------------------------------------------------------------------------

"""
    Focus{T, N}

Inverse-depth focus surface: nodal values `t̂_i` (= `Θ_f`) on the TPS grid, the node
positions and the cached TPS coefficients `(w, a)`, `a = (a_1, a_x, a_y)`, for these values. Fully isbits —
the *solver* object (`FocusSpline`: VisualGeometryToolkitCore's spline system) stays host-side
(rebuilt by the focus block, `focus_block`, at a parameter update).
"""
struct Focus{T, N}
    nodes::SVector{N, SVector{2, T}}
    t̂_i::SVector{N, T}
    w::SVector{N, T}
    a::SVector{3, T}
    kind::Int          # EXPERIMENT (focus-ab): 0 sensor TPS (pixels), 1 direction-chart TPS, 2 spherical spline (focus_charts.jl)
end
Focus{T, N}(nodes, t̂_i, w, a) where {T, N} = Focus{T, N}(nodes, t̂_i, w, a, 0)

"""
    FocusSpline(nodes[, kind = 0])

The host-side solver of a focus surface on `N` nodes: VisualGeometryToolkitCore's `ThinPlateSplineSystem` of the nodes (kind 0: the
sensor (pixels); kind 1, EXPERIMENT focus-ab: the azimuthal-equidistant direction chart) or its `SphericalSplineSystem` of their directions
(kind 2). The host evaluates through the system (Core's `tps_coefficients` / `spherical_spline_coefficients`, `nodal_weights`);
`coeff` is the map of nodal values to `Focus`'s `(w; a₁, a_x, a_y)` in one layout for every kind (the sphere's constant-only map padded
with two zero rows), which the device evaluations convert to Float32 — until Core has a device-safe thin-plate value (remaining: the
dedup audit's b4).
"""
struct FocusSpline{N, S}
    sys::S
    nodes::SVector{N, SVector{2, Float64}}
    coeff::Matrix{Float64}                   # (N+3) × N
    kind::Int
end
function FocusSpline(nodes::AbstractVector{<:StaticVector{2}}, kind::Integer = 0)
    N = length(nodes)
    ns = SVector{N}(SVector{2, Float64}.(nodes))
    if kind == 2
        sys = VgtkCore.SphericalSplineSystem([ae_dir(n) for n in ns])
        return FocusSpline{N, typeof(sys)}(sys, ns, vcat(sys.coeff, zeros(2, N)), 2)
    end
    sys = VgtkCore.ThinPlateSplineSystem(collect(ns))
    return FocusSpline{N, typeof(sys)}(sys, ns, sys.coeff, Int(kind))
end
"""The host-side solver rebuilt from a focus's nodes and kind."""
FocusSpline(f::Focus) = FocusSpline(SVector{2, Float64}.(f.nodes), f.kind)

function Focus(tps::FocusSpline{N}, t̂_i::AbstractVector{T}) where {N, T}
    w, a = _focus_coefficients(tps.sys, t̂_i)
    return Focus{T, N}(SVector{N}(SVector{2, T}.(tps.nodes)), SVector{N, T}(t̂_i), SVector{N, T}(w), SVector{3, T}(a), tps.kind)
end
# (w, a) through the nodal values: Core's coefficient maps; the sphere's constant-only spline has a_x = a_y = 0
_focus_coefficients(sys::VgtkCore.ThinPlateSplineSystem, t) = VgtkCore.tps_coefficients(sys, t)
_focus_coefficients(sys::VgtkCore.SphericalSplineSystem, t) =
    ((w, a₁) = VgtkCore.spherical_spline_coefficients(sys, t); (w, SVector(a₁, zero(a₁), zero(a₁))))

"""
    Focus(n, focus_distance, W, H; T = Float64)

Constant focus surface `𝔱 ≡ focus_distance` on the `n × n` grid of the sensor `[0, W] × [0, H]` (`t̂_i ≡ 1/focus_distance`).
"""
Focus(n::Integer, focus_distance::Real, W::Real, H::Real; T = Float64) =
    Focus(FocusSpline(grid_nodes(n, VgtkCore.Rect(0, 0, W, H))), fill(T(1 / focus_distance), n * n))

"""
    n_nodes(f::Focus) -> Int

The number of nodes of the focus surface's thin plate spline (the length of Θ_f, the nodal values of t̂).
"""
n_nodes(::Focus{T, N}) where {T, N} = N
Base.values(f::Focus) = f.t̂_i                                 # its nodal values t̂ᵢ (Base's `values`, extended: RC shadows no Base name)

"""`t̂(𝐩)` — inverse focus distance at sensor position 𝐩."""
@inline t̂(f::Focus, 𝐩::StaticVector{2}) = VgtkCore.tps_evaluate(f.nodes, f.w, f.a, 𝐩)

"""`𝔱(𝐩) = 1/t̂(𝐩)` — focus distance along the chief ray of 𝐩."""
@inline 𝔱(f::Focus, 𝐩::StaticVector{2}) = 1 / t̂(f, 𝐩)


"""
    chief_cos_at_nodes(lens, nodes) -> Vector{Float64}

cos α of the chief ray at each focus-surface node 𝐩ᵢ, `normalize(π⁻¹(lens, 𝐩ᵢ))[3]`, and NaN at a node the lens does not image (no
ray: π⁻¹ not finite, outside the lens's domain).
"""
function chief_cos_at_nodes(lens, nodes::AbstractVector{<:StaticVector{2}})
    return [begin d = backproject(lens, 𝐩); all(isfinite, d) ? Float64(normalize(d)[3]) : NaN end for 𝐩 in nodes]
end

"""
    focus_start(lens, nodes, 𝔱₀; support = nothing) -> Vector{Float64}

Start values t̂ᵢ = max(cos αᵢ / 𝔱₀, 0) of the focus surface's TPS nodes for a focus plane at distance 𝔱₀ (the LM refines them; they
are a start, not a fit).

- The plane start has no meaning for a ray past 90° (cos α < 0: the plane lies behind it), and the LM's own constraint is t̂ ≥ 0
  (the focus block, `focus_block`, sets t̂ᵢ = max(t̂ᵢ + uᵢ, 0): its box); so every start value is floored at 0, back rays included. (A better start for back rays
  is a modelling question, left open.)
- A node is USABLE when the lens images it (π⁻¹ finite) and, if `support` (a `Bool` per node: inside the observed support, e.g. the
  image circle) is given, it lies in the support. Any other node has no data — no pixel sees its light, and only the TPS coupling to
  its neighbours sees its value — and is filled (`fill_unimaged_nodes`) from the usable ones: the TPS through them clamped to their
  (non-negative) range, their mean when fewer than 3 are non-collinear, an `ArgumentError` naming the lens when there is none.

When every node is usable with cos α ≥ 0 the result is exactly `cos αᵢ / 𝔱₀`. (The parametric lenses' π⁻¹ used to return finite
non-rays outside the image circle — an unconverged Newton's θ — and the focus start took them as they came.)
"""
function focus_start(lens, nodes::AbstractVector{<:StaticVector{2}}, 𝔱₀::Real; support = nothing)
    t = [max(c / 𝔱₀, 0.0) for c in chief_cos_at_nodes(lens, nodes)]            # NaN (no ray) stays NaN: max(NaN, 0) is NaN
    support === nothing || (t[.!support] .= NaN)
    return fill_unimaged_nodes(t, nodes; lens)
end

"""
    fill_unimaged_nodes(values, nodes; lens) -> Vector{Float64}

The nodal t̂ `values` with their non-finite entries (unimaged nodes) filled as `focus_start` describes: the TPS through the finite ones clamped to
their range, their mean when fewer than 3 of them are non-collinear; an `ArgumentError` naming `lens` when none is finite.
"""
function fill_unimaged_nodes(values::AbstractVector{<:Real}, nodes::AbstractVector{<:StaticVector{2}}; lens = nothing)
    v = Float64.(values)
    ok = isfinite.(v)                                          # the usable nodes
    all(ok) && return v
    any(ok) || throw(ArgumentError("focus start: the lens $(lens === nothing ? "" : string(typeof(lens)))" *
                                   " images none of the $(length(nodes)) focus nodes"))
    lo, hi = extrema(v[ok])
    A = [ones(count(ok)) getindex.(nodes[ok], 1) getindex.(nodes[ok], 2)]
    if rank(A) < 3                                             # fewer than 3 non-collinear imaged nodes: no TPS through them
        v[.!ok] .= sum(v[ok]) / count(ok)
    else
        f = Focus(FocusSpline(nodes[ok]), v[ok])
        for i in findall(.!ok)
            v[i] = clamp(t̂(f, SVector{2, Float64}(nodes[i])), lo, hi)
        end
    end
    return v
end
