# ---------------------------------------------------------------------------
# The calibration board 𝓑 (the paper's §Image Formation): its GEOMETRY and its DESIGN are VisualGeometryToolkitCore's — a
# `PosedTarget` of a `PlanarTarget` whose `Canvas` is the printed page, and the canvas's `CanvasTexture` the printed image (its file,
# hash and the rectangle of the design chart it covers) — and this package attaches only the board's APPEARANCE (`PrintedTarget`): the
# image's pixels (albedo ρ_c), the printed layout's edge map (the printed-edge model, printed_edge.jl) and the design size in mm that
# scales the map. Its out-of-plane warping h is in warping.jl.
#
#   target     VgtkCore.PosedTarget: the planar target (Core's design chart: x right, y up as the printed side is seen from the front,
#              metres) posed into the world (metres, plain numbers); its `canvas` holds the page and the printed image
#   quad       the page — the canvas texture's rectangle (the canvas's when it has no texture) — as a Core `Quad` with the texture's
#              chart: corner at the image's top-left, e_u along +x, e_v along −y (the image rows), n = e_u × e_v; a ray off the page
#              misses (NaN)
#   chart      𝐛 = chart(quad, 𝐱) ./ size ∈ [0, 1]² — the image fraction: 𝐛₁ → columns, 𝐛₂ → rows from the top
#   size       the page's (width, height) [m] (the rectangle of the canvas texture; the kernels' isbits copy)
#   facing     +1 or −1: the side of the quad's normal the cameras look at (`facing` at construction); the board's normal for the
#              shading and the warping is facing · n
#   texture    albedo ρ_c(𝐛), (H_t, W_t, 3) array: the printed image's pixels (linear)
#   sdf        optional distance map of the printed layout (H_s × W_s): the signed distance [mm] from every texel centre to the
#              nearest ink edge of the design (+ on paper, − in ink); when set, the albedo is the printed-edge model
#              (printed_edge.jl)
#   page_mm    the design size [mm] the distance map spans along e_u and e_v (the scale of `sdf` and of the edge parameters; 0 without one)
#   𝔟          board centre (light-sphere centre)
# ---------------------------------------------------------------------------

"""
    PrintedTarget{T}

The calibration board as the radiometry sees it: VisualGeometryToolkitCore's posed planar target (its geometry and design: `target`,
whose canvas is the page; the `quad` its rays meet) with the printed appearance attached — the image's pixels, the optional distance map
of the printed layout and its size in mm. See the file header for the chart convention. On a device the target itself stays behind
(`nothing`): the kernels read only the quad and the appearance.
"""
struct PrintedTarget{T, Q <: VgtkCore.Quad{T}, A3 <: AbstractArray{T, 3}, S <: Union{Nothing, AbstractMatrix{T}}, P, Wp}
    target::P                    # VgtkCore.PosedTarget{<:VgtkCore.PlanarTarget} with a Canvas (host), `nothing` on a device
    quad::Q                      # the page: the canvas texture's rectangle, rows down
    size::SVector{2, T}          # the page along e_u, e_v [m]
    facing::T                    # ±1: the side of the quad's normal the cameras see
    texture::A3
    sdf::S
    page_mm::SVector{2, T}
    warping::Wp                  # VgtkCore.MappedSurface: the flat warping's surface, the parts a render rebuilds with h (warping.jl)
end
Adapt.adapt_structure(to, b::PrintedTarget) =
    PrintedTarget(nothing, b.quad, b.size, b.facing, adapt(to, b.texture), adapt(to, b.sdf), b.page_mm, adapt(to, b.warping))

"""
    PrintedTarget(target::PosedTarget, texture; facing = nothing, sdf = nothing, page_mm = nothing)

The board of Core's posed planar `target` whose `canvas` is the printed page — e.g. a calibration session's target record
(VisualGeometryToolkitIO's rig/v6: its canvas and the canvas texture) — with the image's pixels `texture` laid over the canvas
texture's rectangle (the canvas's when it carries no texture). `facing` (a world point, e.g. a camera centre) chooses the side of the
page's normal the cameras see; `sdf` + `page_mm`: the distance map of the printed layout and the design size it spans in mm
(`with_sdf`; printed_edge.jl).
"""
function PrintedTarget(target::VgtkCore.PosedTarget, texture::AbstractArray{<:Real, 3}; facing = nothing, sdf = nothing, page_mm = nothing,
                       T::Type = Float64)
    cv = VgtkCore.canvas(VgtkCore.model(target))
    cv === nothing && throw(ArgumentError("PrintedTarget: the target carries no canvas (the printed page)"))
    r = VgtkCore.Rect(cv.texture === nothing ? cv : cv.texture)            # the page in the design chart (metres, y up)
    W, H = Float64(r.widths[1]), Float64(r.widths[2])
    pose = VgtkCore.pose(target)
    R = SMatrix{3, 3, Float64}(VgtkCore.rotation(pose))
    o = SVector{3, Float64}(pose(SVector(Float64(r.origin[1]), Float64(r.origin[2]) + H, 0.0)))    # the image's top-left corner
    frame = hcat(R[:, 1], -R[:, 2], -R[:, 3])                                                       # e_u = +x, e_v = −y, n = e_u × e_v
    quad = VgtkCore.Quad{T}(VgtkCore.Quad(o, frame, VgtkCore.Rect2{Float64}(0.0, 0.0, W, H)))
    n = frame[:, 3]
    side = facing !== nothing && dot(SVector{3, Float64}(facing) - o, n) < 0 ? -one(T) : one(T)
    (sdf === nothing) == (page_mm === nothing) || throw(ArgumentError("PrintedTarget: `sdf` needs `page_mm` (and vice versa)"))
    sd = sdf === nothing ? nothing : Matrix{T}(sdf)
    pm = page_mm === nothing ? zero(SVector{2, T}) : SVector{2, T}(page_mm)
    return PrintedTarget(target, quad, SVector{2, T}(W, H), T(side), Array{T, 3}(texture), sd, pm, VgtkCore.convert_eltype(T, flat_surface(target)))
end

"""
    PrintedTarget(𝐨, 𝐮_B, 𝐯_B, texture; sdf = nothing, page_mm = nothing, facing = nothing, design = :printed_page)

The page with its image's top-left corner `𝐨` and edges `𝐮_B` (along the image's columns), `𝐯_B` (along its rows) (world, metres;
orthogonal — `𝐯_B` is orthogonalised to rounding): Core's planar target `design` whose board and canvas are the page (design chart
x along `𝐮_B`, y against `𝐯_B`, the origin at the page's lower-left corner `𝐨 + 𝐯_B`), posed into the world, with the `texture` laid
over it. `facing` (a world point, e.g. a camera centre) chooses the side of the normal the cameras see. `sdf` + `page_mm`: the distance
map of the printed layout and the page size in mm (`with_sdf`; printed_edge.jl).
"""
function PrintedTarget(𝐨::AbstractVector, 𝐮_B::AbstractVector, 𝐯_B::AbstractVector, texture::AbstractArray{<:Real, 3};
                       sdf = nothing, page_mm = nothing, facing = nothing, design::Symbol = :printed_page)
    T = float(promote_type(eltype(𝐨), eltype(𝐮_B), eltype(𝐯_B)))
    o = SVector{3, Float64}(𝐨); u = SVector{3, Float64}(𝐮_B); v = SVector{3, Float64}(𝐯_B)
    W = norm(u); e_u = u / W
    v⊥ = v - dot(v, e_u) * e_u
    H = norm(v⊥); e_v = v⊥ / H
    n = cross(e_u, e_v)
    abs(dot(v, e_u)) <= 1e-9 * W * norm(v) || throw(ArgumentError("PrintedTarget: the page edges 𝐮_B, 𝐯_B are not orthogonal"))
    page = VgtkCore.Size2(W, H)
    pose = VgtkCore.EuclideanMap(VgtkCore.Rotations.RotMatrix{3, Float64}(hcat(e_u, -e_v, -n)), o + H * e_v)   # design → world
    target = VgtkCore.PosedTarget(VgtkCore.PlanarTarget(design; board = VgtkCore.Board(page), canvas = VgtkCore.Canvas(page)), pose)
    return PrintedTarget(target, texture; facing, sdf, page_mm, T)
end

"""The board with the distance map `sdf` (`H_s × W_s`, signed distance [mm] to the nearest ink edge of the design, + on paper)
of a page of `page_mm` = (width, height) mm: its albedo becomes the printed-edge model (`edge_albedo`)."""
with_sdf(b::PrintedTarget{T}, sdf::AbstractMatrix, page_mm) where {T} =
    PrintedTarget(b.target, b.quad, b.size, b.facing, b.texture, Matrix{T}(sdf), SVector{2, T}(page_mm), b.warping)

"""Core's posed planar target of the board (its geometry)."""
posed_target(b::PrintedTarget) = b.target
"""The board's normal on the cameras' side, `facing · (e_u × e_v)`: the shading frame's normal and the direction of the warping h."""
@inline board_normal(b::PrintedTarget) = b.facing * VgtkCore.normal(b.quad)

"""World point of chart coordinates 𝐛 (Core's `surface_point` of the quad at the design point 𝐛 ⊙ size)."""
@inline world_point(b::PrintedTarget, 𝐛::StaticVector{2}) = VgtkCore.surface_point(b.quad, SVector(𝐛[1] * b.size[1], 𝐛[2] * b.size[2]))
"""Board centre 𝔟 (the page's centre)."""
𝔟(b::PrintedTarget) = world_point(b, SVector(0.5, 0.5))
"""The page's corners: `𝐨`, `𝐨 + 𝐮_B`, `𝐨 + 𝐮_B + 𝐯_B`, `𝐨 + 𝐯_B`."""
corners(b::PrintedTarget) = (world_point(b, SVector(0.0, 0.0)), world_point(b, SVector(1.0, 0.0)), world_point(b, SVector(1.0, 1.0)),
                             world_point(b, SVector(0.0, 1.0)))
"""
    sphere_radius(b::PrintedTarget, centres) -> r_SH

The pipeline's light-sphere radius about the board centre 𝔟: the farthest of the camera centres AND of the page's corners
(`sphere_radius(𝔟(b), centres)` is the cameras alone). Every board point then lies inside the sphere, so every light ray from the
page leaves it through the sphere (Core's `exit_direction` with Δ ≥ 0) — with cameras closer than the page's half-diagonal (the
27.10 set: 0.61 m against 0.71 m) the corners lay outside, where the exit falls back to the ray's closest point (2026-11-04).
"""
sphere_radius(b::PrintedTarget, centres) = max(sphere_radius(𝔟(b), centres), maximum(norm(c - 𝔟(b)) for c in corners(b)))
"""
    sphere_radius(𝔟, centres) -> r_SH

The light-sphere radius `max_k ‖𝐜_k − 𝔟‖` about the point 𝔟 from the camera centres `centres` (the radius of Core's
`EnvironmentLight` about 𝔟).
"""
sphere_radius(𝔟::AbstractVector, centres) = maximum(norm(SVector{3}(c) - SVector{3}(𝔟)) for c in centres)
"""
    texture_size(b::PrintedTarget) -> (H_t, W_t)

The size of the board's texture in texels, rows first (`size(texture)[1:2]`).
"""
texture_size(b::PrintedTarget) = (size(b.texture, 1), size(b.texture, 2))

"""
    intersect_board(b, 𝐨_ray, 𝐝) -> (s, 𝐱)

Where the ray `𝐱 = 𝐨_ray + s 𝐝` (s ≥ 0) meets the page: Core's `intersection` with the board's quad (`ray_plane_intersection`'s
arithmetic). A ray that misses — parallel to the page, hitting behind its origin or off the page — gives `s` and `𝐱` NaN.
"""
@inline intersect_board(b::PrintedTarget, o::StaticVector{3}, d::StaticVector{3}) = VgtkCore.intersection(b.quad, o, d)

"""
    board_plane_hit(b, 𝐨_ray, 𝐝) -> (s, 𝐱)

Where the ray meets the PLANE of the page, on or off the page, in front or behind (Core's `ray_plane_intersection` on the quad's
plane): the self-check's board residuals, which measure the detections' rays wherever they land.
"""
@inline function board_plane_hit(b::PrintedTarget, o::StaticVector{3}, d::StaticVector{3})
    x, y, z, s = VgtkCore.ray_plane_intersection(b.quad.plane, o, d[1], d[2], d[3])
    return s, SVector(x, y, z)
end

"""Board chart coordinates 𝐛 ∈ ℝ² (the page fraction) of a world point on the board: Core's `chart` of the quad over the page size."""
@inline function chart(b::PrintedTarget, 𝐱::StaticVector{3})
    c = VgtkCore.chart(b.quad, 𝐱)
    return SVector(c[1] / b.size[1], c[2] / b.size[2])
end

"""`true` if 𝐛 lies inside the board with padding `δ` (chart units)."""
@inline inside(𝐛::StaticVector{2}, δ = zero(eltype(𝐛))) = (δ <= 𝐛[1] <= 1 - δ) && (δ <= 𝐛[2] <= 1 - δ)

"""Albedo `ρ_c(𝐛)` (RGB) of the board texture: VisualGeometryToolkitCore's `bilinear_sample` at the page chart 𝐛 (half-texel convention,
clamped to the edge, a non-finite 𝐛 reads the corner texel; dual-safe: the texel is chosen on the value part)."""
@inline ρ_c(b::PrintedTarget, 𝐛::StaticVector{2}) = VgtkCore.bilinear_sample(b.texture, Val(3), 𝐛)
