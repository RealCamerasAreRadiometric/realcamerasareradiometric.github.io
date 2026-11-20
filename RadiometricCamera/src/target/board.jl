# ---------------------------------------------------------------------------
# The calibration board 𝓑 (the paper's §Image Formation, §Board Warping): VisualRadiometryToolkitCore's `TexturedTarget` —
# VisualGeometryToolkitCore's posed planar target with its texture, its printed edge and its own state (`TargetState`: the warping
# h, the print's edge parameters (δ, σ_print), the in-plane print error). The type, its chart, its albedo, the printed-edge model,
# the warped page and the in-plane print error are that package's; a render takes the camera, the pose and the board, and the
# image size is the camera's.
#
# What is the calibration's stays here: the light sphere's radius about the board, the free parameters of the warping (the LM's
# basis of the pure-bending nodal heights), and the reader of a design's edge distance map (a VisualGeometryToolkitIO record).
# The Jacobian seeds the nodal heights h with dual numbers and takes the edge columns by finite differences; the LM carries the
# board's state in the material block (`MaterialVar`, per_view.jl — NLLSsolver caps a cost block at ten variable dependencies,
# which the per-tile blocks already use); δ is the finite-difference column of the material block (`MaterialVar(…; edge = true)`),
# σ_print stays at the print's value.
# ---------------------------------------------------------------------------

"""
    sphere_radius(b::TexturedTarget, centres) -> r_SH

The pipeline's light-sphere radius about the board centre 𝔟: the farthest of the camera centres AND of the page's corners
(`sphere_radius(𝔟(b), centres)` is the cameras alone). Every board point then lies inside the sphere, so every light ray from the
page leaves it through the sphere (Core's `exit_direction` with Δ ≥ 0) — with cameras closer than the page's half-diagonal (the
27.10 set: 0.61 m against 0.71 m) the corners lay outside, where the exit falls back to the ray's closest point (2026-11-04).
"""
sphere_radius(b::TexturedTarget, centres) = max(sphere_radius(𝔟(b), centres), maximum(norm(c - 𝔟(b)) for c in corners(b)))
"""
    sphere_radius(𝔟, centres) -> r_SH

The light-sphere radius `max_k ‖𝐜_k − 𝔟‖` about the point 𝔟 from the camera centres `centres` (the radius of Core's
`EnvironmentLight` about 𝔟).
"""
sphere_radius(𝔟::AbstractVector, centres) = maximum(norm(SVector{3}(c) - SVector{3}(𝔟)) for c in centres)

"""
    HEIGHT_NFREE = HEIGHT_NC - 3

The number of free parameters of the board warping h: the pure-bending nodal heights (Core's `tangent_size` of the warping's
`TargetManifold`, [`height_basis`](@ref), the same on a page of any aspect). The piston and the two tilts are the poses', so they are
not parameters of h.
"""
const HEIGHT_NFREE = HEIGHT_NC - 3
"""Number of free height-field parameters of `degree` 0 (none: the plane) or ≥ 1 (all `HEIGHT_NC` nodal heights; the three affine
combinations — piston and the two tilts, the poses' — are null directions of the representation, so they cannot be tilted into)."""
height_nparams(degree::Integer) = degree <= 0 ? 0 : HEIGHT_NC

"""
    height_manifold(b) -> VgtkCore.TargetManifold

The manifold of the board `b`'s nodal heights: Core's `TargetManifold` of a `HeightField` over its page's `warping_nodes`.
Core's thin-plate kinds are pure bending and have no gauge: the nodal heights move only within the heights whose interpolant on
these nodes has no affine term (the piston and the two tilts are the poses'), so they are never parameters. The affine rows depend
on the aspect, so each page has its own manifold.
"""
function height_manifold(b)
    nodes = warping_nodes(warping_aspect(b))
    return VgtkCore.TargetManifold(VgtkCore.PlanarTarget(nodes; calibration = VgtkCore.HeightField(nodes)))
end
"""
    height_basis(b) -> SMatrix{HEIGHT_NC, HEIGHT_NFREE}

Core's `retract_jacobian` of [`height_manifold`](@ref)`(b)`: a basis of the page's pure-bending nodal heights, `h = h₀ + B y` (the
free height parameters y of the refinement's material block, `MaterialVar`), static for the kernels' dual arithmetic.
"""
function height_basis(b)
    B = VgtkCore.retract_jacobian(height_manifold(b))
    size(B) == (HEIGHT_NC, HEIGHT_NFREE) ||
        throw(ArgumentError("height_basis: the warping's manifold has $(size(B, 2)) free directions, expected HEIGHT_NFREE = $HEIGHT_NFREE"))
    return SMatrix{HEIGHT_NC, HEIGHT_NFREE, Float64, HEIGHT_NC * HEIGHT_NFREE}(B)
end

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
