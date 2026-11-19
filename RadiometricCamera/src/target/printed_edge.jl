# ---------------------------------------------------------------------------
# The printed edge (boards with a distance map `PrintedTarget.sdf`): the ink coverage of a board point is a Gaussian edge
#
#   ink(𝐛) = Φ( (δ − d(𝐛)) / σ_print ),   ρ = 1 − ink,
#
# Φ = VisualGeometryToolkitCore's `normcdf_as` (Abramowitz & Stegun 7.1.26; a dual's derivative is the approximation's, the pathwise
# Jacobian of the evaluated objective), d(𝐛) = signed distance [mm] to the nearest ink edge of the DESIGN (+ on paper), δ = ink spread [mm] (the printed line
# width grows by 2δ), σ_print = the blur of the print [mm]. The defocus disc is the aperture integral's. δ and σ_print are the
# print's (`BoardSurface.edge`); δ is the finite-difference column of the material block (`MaterialVar(…; edge = true)`), σ_print
# stays at the print's value.
# ---------------------------------------------------------------------------

const EDGE_EPS_MM = 1e-4                       # floor of σ_c [mm] (a step edge without blur would be a δ-function derivative)

"""
    edge_albedo(board, surface, 𝐛) -> SVector{3}

Albedo `1 − Φ((δ − d)/σ_print)` of the printed edge at chart 𝐛 (see the file header), the same in every channel. The derivative
w.r.t. 𝐛 flows through the distance map (bilinear), w.r.t. the print's parameters through `BoardSurface.edge`.
"""
@inline function edge_albedo(bd::PrintedTarget, surf::BoardSurface, 𝐛::StaticVector{2})
    d = VgtkCore.bilinear_sample(bd.sdf, 𝐛)                                  # the signed distance [mm] (Core's bilinear read)
    e = surf.edge
    δ = e[1]; σp = e[2]
    e2 = oftype(ForwardDiff.value(d), EDGE_EPS_MM)^2
    ρ = 1 - VgtkCore.normcdf_as((δ - d) / sqrt(σp * σp + e2))
    return SVector(ρ, ρ, ρ)
end

# (dispatch on the FIELD type: a method on `PrintedTarget{T, Q, A3, Nothing} where {T, Q, A3}` is not preferred over one on `PrintedTarget` —
# Julia's specificity rules with the struct's bounded parameters — so every no-map board took the edge path)
"""Albedo ρ_c(𝐛) of a board point: the texture, or the printed edge on a board with a distance map."""
@inline _albedo(bd::PrintedTarget, surf::BoardSurface, 𝐛::StaticVector{2}) = _albedo(bd.sdf, bd, surf, 𝐛)
@inline _albedo(::Nothing, bd::PrintedTarget, surf::BoardSurface, 𝐛::StaticVector{2}) = ρ_c(bd, 𝐛)
@inline _albedo(::AbstractMatrix, bd::PrintedTarget, surf::BoardSurface, 𝐛::StaticVector{2}) = edge_albedo(bd, surf, 𝐛)

"""Whether a board carries the printed-edge model (a distance map)."""
@inline has_sdf(bd::PrintedTarget) = !(bd.sdf isa Nothing)

"""
    load_distance_map(path) -> (values::Matrix{Float32}, page_mm::NTuple{2, Float64})

The printed design's edge distance map (`edge.sdf`): a VisualGeometryToolkitIO raster record (vgio/raster/v2.0, `load_raster`: float32,
the page chart's rectangle, row 1 its top edge) — its values (rows × columns, mm, + on paper, − in ink) and the page's size in mm, the
chart's widths. The one reader (owner D6, 2026-11-15: no hand parser; the generator's raw sdf.f32 is converted by the experiments'
scripts/distance_maps.jl, a keyed step). A raster in another frame, or one starting off the page's corner, is refused by name.
"""
function load_distance_map(path::AbstractString)
    isfile(path) || throw(ArgumentError("edge.sdf = $(repr(path)): no such file — the distance map is a VisualGeometryToolkitIO raster record " *
                                        "(…/distance_map.json, experiments scripts/distance_maps.jl writes it from a generator's sdf.f32 + sdf.json)"))
    r = VgtkIO.load_raster(path)
    r.frame isa NamedTuple && r.frame.chart === :page ||
        throw(ArgumentError("edge.sdf = $(repr(path)): a distance map is a raster on the page chart, this one is on $(repr(r.frame))"))
    a = r.frame.area
    iszero(a.origin[1]) && iszero(a.origin[2]) ||
        throw(ArgumentError("edge.sdf = $(repr(path)): the distance map covers the page from its lower-left corner, this one starts at $(a.origin) m"))
    return r.values, (1000 * a.widths[1], 1000 * a.widths[2])
end
