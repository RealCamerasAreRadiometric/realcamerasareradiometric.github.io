# ---------------------------------------------------------------------------
# Board warping (paper §Board Warping) and the per-scene surface parameters.
#
#   h    the out-of-plane shape of the board 𝓑: Core's HeightField over the page in units of its width (any aspect), placed and posed
#        through HEIGHT_N × HEIGHT_N nodal heights at the page's node grid (world units along the board normal 𝐧, z > 0 towards
#        the camera); all nodal heights zero = the plane. The
#        ray–surface intersection is Core's `MappedSurface` root of the relief's chart map, seeded at the plane hit.
#   edge the print's edge parameters (δ, σ_print) of a board whose design is a distance map (printed_edge.jl)
#
# Both live in `BoardSurface`, the parameter container of the scene that every render receives BY VALUE (the board itself
# is cached on the device): the Jacobian seeds the nodal heights h with dual numbers and takes the edge columns by finite
# differences, and the LM carries the surface in the material block (`MaterialVar`, per_view.jl — NLLSsolver caps a cost block
# at ten variable dependencies, which the per-tile blocks already use).
# ---------------------------------------------------------------------------

"""
    EDGE_NC = 2

The number of printed-edge parameters of a [`BoardSurface`](@ref) (`edge`: the ink spread δ [mm] and the print's blur `σ_print`
[mm], read by [`edge_albedo`](@ref)).
"""
const EDGE_NC = 2

# --- the height field h: Core's HeightField over the page, in units of the page's width ---
#
# The warping is VisualGeometryToolkitCore's `HeightField` on the page's PLACED CHART — the page in units of its width, origin at its
# top-left corner, x along the columns, y UP: 𝐜 = (𝐛₁, −α 𝐛₂) of the page fraction 𝐛, α = H / W (a thin-plate spline is the same
# surface under a similarity of its domain, so the warping is the page's physical bending, whatever its aspect) — placed on the
# design by a `SimilarityMap{3}` (scale W, the identity rotation, the page's top-left corner), and posed by the board's pose: the
# render's surface is Core's `MappedSurface(pose, field, placement, page)` (the composed chart map `pose ∘ S₃ ∘ field ∘ S₂⁻¹` and its
# ray root), the same parts `posed_target` gives Core's posed target. Its nodes are the HEIGHT_N × HEIGHT_N grid over the page (page
# fractions `HEIGHT_NODES`); the nodal heights h of `BoardSurface.h` are world lengths along the board's `facing` normal (z > 0 towards
# the camera), the field's heights `−facing h / W` (along the design normal, in chart units: the placement scales them back). Core's
# field keeps only the pure bending of h (the piston and the two tilts are the poses'), so an affine set of nodal heights is the plane.
const HEIGHT_N = 5                                                                  # nodes per page axis
"""
    HEIGHT_NC = 25

The number of nodal heights of the board warping h (`BoardSurface.h`): Core's `HeightField` through a 5 × 5 node grid over the page,
heights in world units along the board normal. Of these, [`HEIGHT_NFREE`](@ref) are free parameters.
"""
const HEIGHT_NC = HEIGHT_N * HEIGHT_N
"""The warping's node grid as PAGE FRACTIONS 𝐛 ∈ [0, 1]² (columns, rows from the top; the first index fastest): where nodal height `i`
sits on any page."""
const HEIGHT_NODES = grid_nodes(HEIGHT_N; lo = 0.0, hi = 1.0)
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
    warping_nodes(aspect) -> Vector{SVector{2, Float64}}

The warping's nodes on a page of `aspect` = height / width, in its placed chart (the page in units of its width, origin at the
top-left corner, y up): `(uᵢ, −aspect · vᵢ)` of the page fractions `(uᵢ, vᵢ) = HEIGHT_NODES[i]`.
"""
function warping_nodes(aspect::Real)
    α = Float64(aspect)
    (isfinite(α) && α > 0) || throw(ArgumentError("warping_nodes: the page's aspect (height / width) must be positive and finite, got $aspect"))
    return [SVector(n[1], -α * n[2]) for n in HEIGHT_NODES]
end
"""
    page_chart(b) -> (; aspect, height_scale)

THE page chart of the board `b`'s warping, in `b`'s element type (device-capable): the page's `aspect` = height / width, and the
`height_scale` = −facing / W taking a nodal height (world units, towards the cameras) to the chart's height (along the design
normal, in units of the page's width). Its points are `chart_point(aspect, 𝐛)`; its placement on the design `page_placement` of the
page's rectangle.
"""
page_chart(b) = (; aspect = b.size[2] / b.size[1], height_scale = -b.facing / b.size[1])
"""The placed chart point `(𝐛₁, −aspect 𝐛₂)` of the page fraction `𝐛` (the page in units of its width, origin at its top-left
corner, y up): where [`warping_nodes`](@ref) lie."""
chart_point(aspect, 𝐛) = SVector(𝐛[1], -aspect * 𝐛[2])
"""The page's aspect (height / width) of the board `b`: its `page_chart`'s."""
warping_aspect(b) = Float64(page_chart(b).aspect)
"""
    warping_field(b, h) -> VgtkCore.HeightField

Core's height field of the nodal heights `h` on the board `b`'s page (host): over [`warping_nodes`](@ref) of its aspect, the heights
`−facing h / W` (along the design normal, in units of the page's width). All nodal heights zero, or any affine set of them, = flat.
"""
warping_field(b, h::AbstractVector) =
    VgtkCore.HeightField(warping_nodes(warping_aspect(b)), Float64(page_chart(b).height_scale) .* Float64.(collect(h)))
"""The placement of the warping's chart on the design (Core's `SimilarityMap{3}`): scale the page's width, the identity rotation,
translation to the page's top-left corner `origin` (design chart, metres)."""
warping_placement(origin::StaticVector{2}, side::Real) =
    VgtkCore.SimilarityMap(Float64(side), VgtkCore.RotMatrix{3}(SMatrix{3, 3, Float64}(I)), SVector(Float64(origin[1]), Float64(origin[2]), 0.0))
"""The placement of the page chart on the design for the page's rectangle `r` (design chart, metres, y up): `warping_placement` at
its top-left corner, scaled by its width."""
page_placement(r) = warping_placement(SVector(Float64(r.origin[1]), Float64(r.origin[2]) + Float64(r.widths[2])), Float64(r.widths[1]))
"""The page's rectangle in the design chart (metres, y up): the canvas texture's, the canvas's without one."""
_page_rect(t) = (cv = VgtkCore.canvas(t); VgtkCore.Rect(cv.texture === nothing ? cv : cv.texture))
"""
    flat_surface(target) -> VgtkCore.MappedSurface

The board's surface of a flat warping, built ONCE on the host by Core's builder `MappedSurface(pose, field, placement, page)` of the
posed target `target` (its pose, the flat field over its page's nodes, the placement at the page's top-left corner, the page's
rectangle): the parts every render rebuilds with its nodal heights (`board_relief`).
"""
function flat_surface(target::VgtkCore.PosedTarget)
    r = _page_rect(VgtkCore.model(target))
    W, H = Float64(r.widths[1]), Float64(r.widths[2])
    return VgtkCore.MappedSurface(VgtkCore.pose(target), VgtkCore.HeightField(warping_nodes(H / W)), page_placement(r),
                                  VgtkCore.Rect2{Float64}(Float64(r.origin[1]), Float64(r.origin[2]), W, H))
end
"""
    height_manifold(b) -> VgtkCore.TargetManifold

The manifold of the board `b`'s nodal heights: Core's `TargetManifold` of a `HeightField` over its page's [`warping_nodes`](@ref).
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
    board_height(b, h, 𝐛) -> z
    board_height(b, h, 𝐛s) -> Vector

The board's height above its plane at the page fraction `𝐛` [world units, along the board's `facing` normal: z > 0 = towards the
camera] under the nodal heights `h` (host, plain numbers): Core's `HeightField` of [`warping_field`](@ref) at the placed chart point,
scaled back to world units; at every page fraction of `𝐛s`, the field built once. The warping's report and its metrics read it.
"""
board_height(b, h::AbstractVector, 𝐛::StaticVector{2}) = only(board_height(b, h, (𝐛,)))
function board_height(b, h::AbstractVector, 𝐛s)
    W, α, f = Float64(b.size[1]), warping_aspect(b), warping_field(b, h)
    return [-Float64(b.facing) * W * f(chart_point(α, SVector{2, Float64}(𝐛)))[3] for 𝐛 in 𝐛s]
end

"""
    board_relief(b, h) -> VgtkCore.MappedSurface

The warped page: the board's `flat_surface` (`b.warping`, Core's composed chart map, adapted to the device with the board)
with its field rebuilt from the nodal heights `h` by Core's `with_parameters` of the surface (the field's heights `−facing h / W`,
every other member of its chart map kept) — Core's
`MappedSurface` of the posed target `posed_target(b, with_h(BoardSurface(), h))` gives, member for member. Any element type
(Float32, duals).
"""
@inline function board_relief(b::PrintedTarget, h::StaticVector{HEIGHT_NC, T}) where {T}
    return VgtkCore.with_parameters(b.warping, h .* page_chart(b).height_scale)
end

"""
    intersect_board(b, 𝐨_ray, 𝐝, h) -> (s, 𝐱)

Ray–surface intersection with the board's height field `h` (`board_relief`): VisualGeometryToolkitCore's `intersection` with the
warped page's `MappedSurface` — its height field's 1-D Newton; with ForwardDiff Duals in the heights (the nodal-height mode) that
same hit and its partials by ONE implicit-function step at it (Core's, hoists-09-intersect; never the generic 2 × 2 root). `chart(b, 𝐱)` of the result is the surface point's chart position (the normal component is
ignored by the chart). All nodal heights zero = the plane hit.
"""
@inline function intersect_board(b::PrintedTarget, o::StaticVector{3}, d::StaticVector{3}, h::StaticVector{HEIGHT_NC, T}) where {T}
    is_warped(h) || return intersect_board(b, o, d)
    return VgtkCore.intersection(board_relief(b, h), o, d)
end
"""Whether the nodal heights `h` warp the board: nonzero, or seeded (a seeded flat board still needs its partials)."""
@inline is_warped(h::StaticVector{HEIGHT_NC, T}) where {T} = T <: ForwardDiff.Dual || !iszero(h)

"""
    BoardRays(b, h)
    BoardRays(scene)

The board of a render prepared for its rays: the `board_relief` of the heights `h` formed ONCE (its field's weights), for every ray of
a pixel's quadrature — `intersect_board(rays, 𝐨, 𝐝)` is `intersect_board(b, 𝐨, 𝐝, h)` bit for bit, without rebuilding the relief per
ray.
"""
struct BoardRays{B, R}
    board::B
    relief::R
    warped::Bool
end
@inline BoardRays(b::PrintedTarget, h::StaticVector{HEIGHT_NC}) = BoardRays(b, board_relief(b, h), is_warped(h))
@inline intersect_board(r::BoardRays, o::StaticVector{3}, d::StaticVector{3}) =
    r.warped ? VgtkCore.intersection(r.relief, o, d) : intersect_board(r.board, o, d)


# --- the in-plane print error: Core's Aspect ∘ Skew in the warping's placed chart ---
#
# The print's in-plane error — the design stretched along one axis against the other and sheared — as Core's two gauge-free factors
# of an affine print error under a jointly estimated pose, `Aspect(a) ∘ Skew(u)` (Core's target_calibration.jl: `D = diag(eᵃ, e⁻ᵃ)`,
# det 1, and `U = [1 u; 0 1]`; a rigid member and a uniform scale are the pose's, so they are never parameters). It acts in the CHART
# Core's `posed_target` places the height field in — the page in units of its width, origin at the page's top-left corner, x along
# the columns, y UP: 𝐜 = (𝐛₁, −α 𝐛₂) of the page fraction 𝐛, α = H / W — so the posed target's record `HeightField ∘ Aspect ∘ Skew`
# is the map the render uses. Core's calibration takes the DESIGN to the PHYSICAL print: a physical surface point at the page fraction
# 𝐛 shows the design at `inv(Aspect ∘ Skew)` of its chart point (`design_chart`), where the renderer samples the texture and the edge
# map; the height field stays a function of the physical point (Core's canonical order). Every parameter zero = the design as printed.

"""
    INPLANE_NC = 2

The number of parameters of the in-plane print error of the board (`BoardSurface.inplane`): the log-stretch `a` of Core's `Aspect`
(x by eᵃ, y by e⁻ᵃ; `a = 0.001` stretches x by 0.1 % against y) and the shear `u` of Core's `Skew` (`x ↦ x + u y`), both in the
warping's placed chart (the page in units of its width, y up).
"""
const INPLANE_NC = 2

"""
    inplane_calibration(θ) -> Core's `Aspect(θ[1]) ∘ Skew(θ[2])`

Core's in-plane print error of the parameters `θ = (a, u)` (`INPLANE_NC`): the map from the design to the physical print in the
warping's placed chart, the identity at zero. Its element type follows `θ` (duals seed the Jacobian columns).
"""
@inline inplane_calibration(θ::StaticVector{INPLANE_NC}) = VgtkCore.Aspect(θ[1]) ∘ VgtkCore.Skew(θ[2])
"""Whether the in-plane print error is on: a nonzero parameter (ForwardDiff's `iszero` of a dual checks its partials too, so parameters
seeded for a Jacobian column count as on — a seeded identity still needs its partials)."""
@inline inplane_on(θ::StaticVector{INPLANE_NC}) = !iszero(θ)
"""The placed chart point of the page fraction 𝐛 on the board `b`'s page: `(𝐛₁, −α 𝐛₂)` — the page in units of its width,
origin at its top-left corner, y up (the chart Core's `posed_target` places the calibration in; the aspect enters as Core's
`primal_float` of 𝐛's element type, so it does not promote the point)."""
@inline placed_chart(b::PrintedTarget, 𝐛::StaticVector{2}) = SVector(𝐛[1], -𝐛[2] * _aspect_in(eltype(𝐛), b))
"""The page fraction of a placed chart point (the inverse of `placed_chart`)."""
@inline page_fraction(b::PrintedTarget, 𝐜::StaticVector{2}) = SVector(𝐜[1], -𝐜[2] / _aspect_in(eltype(𝐜), b))
# the page's aspect in the primal float of a point's element type (a Dual's value type), so it does not promote the point
@inline _aspect_in(::Type{E}, b::PrintedTarget) where {E} = (F = VgtkCore.primal_float(E); F(b.size[2]) / F(b.size[1]))
"""
    design_chart(b, θ, 𝐛) -> 𝐛_design

The page fraction of the DESIGN a physical surface point at the page fraction `𝐛` shows under the in-plane print error `θ`
(`inplane_calibration`): `inv(Aspect ∘ Skew)` of its placed chart point, back as a page fraction — where the renderer samples the
texture and the edge map. `𝐛` itself while the error is off.
"""
@inline function design_chart(b::PrintedTarget, θ::StaticVector{INPLANE_NC}, 𝐛::StaticVector{2})
    inplane_on(θ) || return 𝐛
    return page_fraction(b, inv(inplane_calibration(θ))(placed_chart(b, 𝐛)))
end
"""
    physical_chart(b, θ, 𝐛_design) -> 𝐛

The page fraction where the design point `𝐛_design` is printed under the in-plane print error `θ`: Core's calibration forward
(`Aspect ∘ Skew`) in the placed chart, the inverse of `design_chart` — the board features of a refinement are placed with it.
"""
@inline function physical_chart(b::PrintedTarget, θ::StaticVector{INPLANE_NC}, 𝐛::StaticVector{2})
    inplane_on(θ) || return 𝐛
    return page_fraction(b, inplane_calibration(θ)(placed_chart(b, 𝐛)))
end

"""
    BoardSurface{T}

The scene's surface parameters: the board warping `h` (the `HEIGHT_NC` nodal heights of the thin plate spline, `board_height`;
all zero = the plane), the print's edge parameters `edge = (δ [mm], σ_print [mm])` of a board with a distance map
(`PrintedTarget.sdf`, `edge_albedo`; unused otherwise) and the in-plane print error `inplane = (a, u)` (`inplane_calibration`,
Core's `Aspect ∘ Skew`; zero = the design as printed). `BoardSurface()` is the plane with no edge parameters, printed as designed.
"""
struct BoardSurface{T}
    h::SVector{HEIGHT_NC, T}
    edge::SVector{EDGE_NC, T}
    inplane::SVector{INPLANE_NC, T}
end
BoardSurface{T}() where {T} = BoardSurface{T}(zero(SVector{HEIGHT_NC, T}), zero(SVector{EDGE_NC, T}), zero(SVector{INPLANE_NC, T}))
function BoardSurface(; h = nothing, edge = nothing, inplane = nothing, T = nothing)
    Tv = T !== nothing ? T : h !== nothing ? float(eltype(h)) : edge !== nothing ? float(eltype(edge)) : inplane !== nothing ? float(eltype(inplane)) : Float64
    hv = h === nothing ? zero(SVector{HEIGHT_NC, Tv}) : SVector{HEIGHT_NC, Tv}(h)
    ev = edge === nothing ? zero(SVector{EDGE_NC, Tv}) : SVector{EDGE_NC, Tv}(edge)
    iv = inplane === nothing ? zero(SVector{INPLANE_NC, Tv}) : SVector{INPLANE_NC, Tv}(inplane)
    return BoardSurface{Tv}(hv, ev, iv)
end
"""The surface with the nodal heights replaced."""
with_h(s::BoardSurface{T}, h) where {T} = BoardSurface{T}(SVector{HEIGHT_NC, T}(h), s.edge, s.inplane)
"""The surface with the print's edge parameters `(δ mm, σ_print mm)` replaced (see `edge_albedo`)."""
with_edge(s::BoardSurface{T}, edge) where {T} = BoardSurface{T}(s.h, SVector{EDGE_NC, T}(edge), s.inplane)
"""The surface with the in-plane print error's parameters `(a, u)` replaced (`inplane_calibration`)."""
with_inplane(s::BoardSurface{T}, θ) where {T} = BoardSurface{T}(s.h, s.edge, SVector{INPLANE_NC, T}(θ))
"""Whether the board is warped; all nodal heights zero (the default) = the plane."""
@inline height_on(s::BoardSurface) = !iszero(s.h)
"""Whether the surface carries an in-plane print error (`inplane_on` of its parameters)."""
@inline inplane_on(s::BoardSurface) = inplane_on(s.inplane)
VgtkCore.convert_eltype(::Type{T}, s::BoardSurface) where {T} =
    BoardSurface{T}(SVector{HEIGHT_NC, T}(s.h), SVector{EDGE_NC, T}(s.edge), SVector{INPLANE_NC, T}(s.inplane))

"""
    page_points(𝐨, 𝐮, 𝐯, 𝐧, h, 𝐛s; inplane = (0, 0)) -> Vector{SVector{3, Float64}}

The world points at the page fractions `𝐛s` of a page with its top-left corner `𝐨`, its edges `𝐮` (along the columns) and `𝐯` (along
the rows, orthogonal to `𝐮`) and the warping's normal `𝐧` (the side the nodal heights `h` rise to), under the warping `h` and the
in-plane print error `inplane` (`inplane_calibration`): Core's `points` of the page's planar target — the design points `(W 𝐛₁, −H 𝐛₂)`
in the chart of its top-left corner, its calibration `HeightField ∘ Aspect ∘ Skew` (the warping over [`warping_nodes`](@ref), heights
in units of the page's width) placed at that corner by `warping_placement` — posed by the page's frame. The surface the
render's `board_relief` meets, so a board feature placed with it lies on the rendered board.
"""
function page_points(𝐨::AbstractVector, 𝐮::AbstractVector, 𝐯::AbstractVector, 𝐧::AbstractVector, h::AbstractVector, 𝐛s::AbstractVector;
                     inplane = zero(SVector{INPLANE_NC, Float64}))
    W, H = norm(𝐮), norm(𝐯)
    eu, ev = SVector{3, Float64}(𝐮) / W, SVector{3, Float64}(𝐯) / H
    abs(dot(eu, ev)) <= 1e-9 || throw(ArgumentError("page_points: the page edges 𝐮, 𝐯 are not orthogonal"))
    ez = cross(eu, -ev)                                                    # the design normal of the frame (x, y) = (𝐮, −𝐯)
    σ = sign(dot(SVector{3, Float64}(𝐧), ez))
    σ == 0 && throw(ArgumentError("page_points: the warping's normal 𝐧 lies in the page"))
    θ = SVector{INPLANE_NC, Float64}(inplane)
    warped = any(!iszero, h)
    ip = inplane_calibration(θ)
    c = page_chart((; size = SVector(W, H), facing = -σ))                 # the page's chart: σ / W, the heights along the design normal
    cal = warped ? (hf = VgtkCore.HeightField(warping_nodes(c.aspect), c.height_scale .* Float64.(collect(h))); iszero(θ) ? hf : hf ∘ ip) :
          iszero(θ) ? VgtkCore.Rigid() : ip
    design = [SVector(W * Float64(𝐛[1]), -H * Float64(𝐛[2])) for 𝐛 in 𝐛s]
    t = VgtkCore.PlanarTarget(design; calibration = cal, placement = warping_placement(SVector(0.0, 0.0), W))
    R = hcat(eu, -ev, ez)
    o = SVector{3, Float64}(𝐨)
    lift(p) = length(p) == 3 ? SVector{3, Float64}(p) : SVector(Float64(p[1]), Float64(p[2]), 0.0)
    return [o + R * lift(p) for p in VgtkCore.points(t)]
end

"""
    posed_target(b, surface) -> VgtkCore.PosedTarget

The warped board as VisualGeometryToolkitCore's posed planar target: the page's target with the warping's `HeightField`
([`warping_field`](@ref): the page in units of its width, origin at its top-left corner, y up) placed on the design by
`warping_placement` — the parts of the render's surface (`flat_surface`, `board_relief`), so Core's `surface` of the result
is the board's surface. An in-plane print error composes after it in Core's canonical order, `HeightField ∘ Aspect ∘ Skew`
(`inplane_calibration`, in the same placed chart; `Aspect ∘ Skew` alone on a flat board). The placement is in plain metres (Core's
placement form, for a target with lengths too). The board's own target when the surface is flat and printed as designed.
"""
function posed_target(b::PrintedTarget, s::BoardSurface)
    height_on(s) || inplane_on(s) || return b.target
    t = VgtkCore.model(b.target)
    r = _page_rect(t)
    ip = inplane_calibration(SVector{INPLANE_NC, Float64}(s.inplane))
    cal = if height_on(s)
        hf = warping_field(b, s.h)
        inplane_on(s) ? hf ∘ ip : hf
    else
        ip
    end
    P = page_placement(r)
    return VgtkCore.PosedTarget(VgtkCore.PlanarTarget(VgtkCore.reference(t), nothing, cal, VgtkCore.board(t), VgtkCore.canvas(t), P), VgtkCore.pose(b.target))
end
