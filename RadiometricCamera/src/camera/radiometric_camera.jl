# ---------------------------------------------------------------------------
# Radiometric back-projection.
#
#   π⁻¹(𝐩)          central (chief-ray) back-projection of the lens model
#   d̂(𝐩)            = π⁻¹(𝐩)/‖π⁻¹(𝐩)‖   UNIT chief direction (`chief_dir`): the grid lenses' bicubic blend of unit node directions is
#                                        not unit (≤ 2e-4 off inside the image circle), the parametric models are. Decision 2026-10-23:
#                                        the focus point uses the unit direction, so 1/t̂(𝐩) IS the distance from 𝔭 to 𝔣(𝐩).
#   𝔣(𝐩)            = 𝔭 + 𝔱(𝐩) d̂(𝐩)    focus point on the Petzval surface 𝔉
#   π̂⁻¹(𝐩, 𝐚)      = (𝔣(𝐩) − 𝐚)/‖𝔣(𝐩) − 𝐚‖   direction of the ray from aperture point 𝐚
#                    = (d̂(𝐩) − t̂(𝐩) 𝐚)/‖…‖   (same direction, regular at t̂ = 0)
#
# Camera frame: CV convention (+x right, +y down, +z forward); the aperture 𝒜 is
# the disc of radius r in the plane z = 0 centred at 𝔭 = 0.
# ---------------------------------------------------------------------------

"""
    RadiometricCameraModel(lens, focus, aperture_radius, vignetting)

The radiometric camera, what every photograph of a calibration shares, as ONE object: the geometric camera π⁻¹ (`lens`: Core's
pixel camera or a grid model; Θ_d), the focus surface (`focus`, Θ_f), the aperture radius r (`aperture_radius`, metres) and the
vignetting V (`vignetting`, Core's `AbstractVignetting`, Θ_v). Every function of the camera takes it whole: a photograph is the
camera at a pose (camera → world, Core's `EuclideanMap`) — [`render_view`](@ref)`(k, camera, pose, …)`,
[`compute_mask`](@ref)`(camera, pose, …)`, [`Φ_pixel`](@ref)`(u, v, k, camera, pose, …)`,
[`world_ray`](@ref)`(camera, pose, 𝐩, 𝐚)` — and a calibration holds one camera and its poses
([`CalibrationState`](@ref)`(scene, settings, camera, poses, …)`).

The image distance is not a parameter: the defocus is the focus surface t̂(𝐩) along each chief ray, and the cos⁴/s′² obliquity is in V.
"""
struct RadiometricCameraModel{L, F <: Focus, T <: Real, V <: AbstractVignetting}
    lens::L
    focus::F
    aperture_radius::T
    vignetting::V
end
"""
    RadiometricCameraModel(lens; focus, aperture_radius, vignetting = V ≡ 1)

The radiometric camera composed from its geometric camera `lens` (a VisualGeometryToolkitCore camera bounded by its image: a
`BoundedCamera` of a `CameraModel` over any Core lens, or a grid model) by adding the three parts a geometric camera lacks:

- `focus`: the focus surface, a [`Focus`](@ref), or a distance in metres — the surface focused at that distance everywhere
  (`Focus(lens, distance)`, a 3 × 3 thin plate spline on the camera's image);
- `aperture_radius`: the radius r of the aperture disc, metres;
- `vignetting`: Core's `AbstractVignetting` on the camera's image; by default V ≡ 1 as a flat 4 × 4 quadratic B-spline surface over
  the image (`TensorProductVignetting((4, 4), Rect(0, 0, W, H); degree = 2)`), the surface a calibration then estimates.

The image size is the geometric camera's (`extent`): it is stated once. The camera's parts are replaced one at a time by
[`with_lens`](@ref), [`with_focus`](@ref), [`with_aperture_radius`](@ref) and [`with_vignetting`](@ref).
"""
function RadiometricCameraModel(lens::VgtkCore.BoundedCamera; focus, aperture_radius::Real,
                                vignetting::AbstractVignetting = _flat_vignetting(lens))
    return RadiometricCameraModel(lens, _focus_surface(lens, focus), aperture_radius, vignetting)
end
_focus_surface(lens, focus::Focus) = focus
_focus_surface(lens, distance::Real) = Focus(lens, distance)
"""The image rectangle `[0, W] × [0, H]` of a bounded geometric camera (COLMAP pixels): the domain of its focus and vignetting surfaces."""
_image_rect(lens::VgtkCore.BoundedCamera) = ((W, H) = VgtkCore.extent(lens); VgtkCore.Rect(0, 0, W, H))
"""V ≡ 1 on the camera's image: the flat 4 × 4 quadratic B-spline surface (Core's `TensorProductVignetting`)."""
_flat_vignetting(lens::VgtkCore.BoundedCamera) = VgtkCore.TensorProductVignetting((4, 4), _image_rect(lens); degree = 2)

"""
    Focus(lens, focus_distance; nodes = 3, T = Float64)

The constant focus surface `𝔱 ≡ focus_distance` (metres) on the `nodes × nodes` grid of the image of the geometric camera `lens` (its
`extent`): `Focus(nodes, focus_distance, W, H)`.
"""
Focus(lens::VgtkCore.BoundedCamera, focus_distance::Real; nodes::Integer = 3, T = Float64) =
    ((W, H) = VgtkCore.extent(lens); Focus(nodes, focus_distance, W, H; T))

"""The camera with its geometric camera (the lens, Θ_d) replaced."""
with_lens(c::RadiometricCameraModel, lens) = RadiometricCameraModel(lens, c.focus, c.aperture_radius, c.vignetting)
"""The camera with its focus surface (Θ_f) replaced."""
with_focus(c::RadiometricCameraModel, f::Focus) = RadiometricCameraModel(c.lens, f, c.aperture_radius, c.vignetting)
"""The camera with its aperture radius r (metres) replaced."""
with_aperture_radius(c::RadiometricCameraModel, r::Real) = RadiometricCameraModel(c.lens, c.focus, r, c.vignetting)
"""The camera with its vignetting replaced."""
with_vignetting(c::RadiometricCameraModel, v::AbstractVignetting) = RadiometricCameraModel(c.lens, c.focus, c.aperture_radius, v)
"""
    focus(camera::RadiometricCameraModel) -> Focus

The camera's focus surface (Θ_f).
"""
focus(c::RadiometricCameraModel) = c.focus
"""
    aperture_radius(camera::RadiometricCameraModel)

The camera's aperture radius r, in metres.
"""
aperture_radius(c::RadiometricCameraModel) = c.aperture_radius
"""
    lens(camera::RadiometricCameraModel)

The camera's geometric model π⁻¹ (Θ_d): Core's pixel camera or a grid model.
"""
lens(c::RadiometricCameraModel) = c.lens
# the geometric camera's maps (Core's contract generics), forwarded to the lens
VgtkCore.project(c::RadiometricCameraModel, X) = VgtkCore.project(c.lens, X)
VgtkCore.backproject(c::RadiometricCameraModel, 𝐩) = VgtkCore.backproject(c.lens, 𝐩)

"""
    View{T, N}

Internal: one photograph of a [`RadiometricCameraModel`](@ref) as the renderer's kernels evaluate it — the camera at a pose
(`View(pose, camera)`), flattened into one element type `T` (Float64, Float32 on the device, a dual in the Jacobian evaluations): the
pose (camera → world, Θ_e), the lens Θ_d, the focus surface Θ_f and the aperture radius r. The vignetting is not in it: the kernels
take V apart, as they differentiate it analytically. Callers give the camera and the pose; the public functions form the view.
"""
struct View{T, N, L, P}
    pose::P                                   # camera → world: Core's EuclideanMap in T (`_pose_eltype`; t = the camera centre)
    lens::L                                   # Core's pixel camera (a BoundedCamera of a CameraModel) or a grid lens (Core's spline model / window)
    focus::Focus{T, N}
    r::T
    # inner: no default outer constructor (it would be ambiguous with `View(pose, lens, focus, r)` below)
    View{T, N, L, P}(pose::P, lens::L, focus::Focus{T, N}, r::T) where {T, N, L, P} = new{T, N, L, P}(pose, lens, focus, r)
end

function View(pose::EuclideanMap, lens, focus::Focus{<:Any, N}, r) where {N}
    T = promote_type(eltype(pose.t), _model_number_type(lens), eltype(focus.t̂_i), typeof(r))
    foc = Focus{T, N}(SVector{N}(SVector{2, T}.(focus.nodes)), SVector{N, T}(focus.t̂_i),
                      SVector{N, T}(focus.w), SVector{3, T}(focus.a), focus.kind)     # element-type conversion only, no TPS solve
    lensT = _model_eltype(T, lens)
    poseT = _pose_eltype(T, pose)
    return View{T, N, typeof(lensT)}(poseT, lensT, foc, T(r))
end
"""The photograph of `camera` at `pose` (camera → world)."""
View(pose::EuclideanMap, c::RadiometricCameraModel) = View(pose, c.lens, c.focus, c.aperture_radius)
"""A view whose blocks are already in the element type `T` (the device and Jacobian evaluations): the pose's type from the pose."""
View{T, N, L}(pose::P, lens::L, focus::Focus{T, N}, r) where {T, N, L, P <: EuclideanMap} = View{T, N, L, P}(pose, lens, focus, r)

Adapt.adapt_structure(to, v::View{T, N}) where {T, N} =
    (l = adapt(to, v.lens); View{T, N, typeof(l)}(v.pose, l, v.focus, v.r))   # a grid lens carries an array

"""Centre of projection 𝔭 (origin of the camera frame)."""
const 𝔭 = SVector(0.0, 0.0, 0.0)

"""The view's chief ray of 𝐩 (Θ_d; camera frame, `_chief_ray`)."""
@inline _chief_ray(v::View, 𝐩::StaticVector{2}) = _chief_ray(v.lens, 𝐩)


"""
    chief_dir(lens, 𝐩) -> SVector{3}

Unit chief direction d̂(𝐩) = π⁻¹(𝐩)/‖π⁻¹(𝐩)‖ of a sensor position: the direction along which the focus point 𝔣(𝐩) lies at the
distance 1/t̂(𝐩) from the centre of projection. `π⁻¹` of the grid lenses is a bicubic blend of unit node directions and is not unit
(≤ 2e-4 off inside the image circle on the real grids); the parametric models return unit vectors. Dual-generic: when `π⁻¹` carries
partials (per-view lens AD, seeded grid-node directions) the normalisation projects them onto the tangent plane of d̂.
"""
@inline chief_dir(lens, 𝐩::StaticVector{2}) = normalize(_chief_ray(lens, 𝐩))
@inline chief_dir(v::View, 𝐩::StaticVector{2}) = chief_dir(v.lens, 𝐩)
@inline chief_dir(c::RadiometricCameraModel, 𝐩::StaticVector{2}) = chief_dir(c.lens, 𝐩)

"""The camera's own functions (camera frame, no pose) take the camera or the kernels' view of it: the same fields, one code path."""
const _CameraOrView = Union{RadiometricCameraModel, View}

"""
    𝔣(camera, 𝐩) -> SVector{3}

Focus point 𝔣(𝐩) = 𝔱(𝐩) d̂(𝐩) on the Petzval surface, d̂ the UNIT chief direction (camera frame, the centre of projection
𝔭 = 0): ‖𝔣(𝐩)‖ = 𝔱(𝐩) = 1/t̂(𝐩) exactly, for every lens class.
"""
@inline 𝔣(v::_CameraOrView, 𝐩::StaticVector{2}) = inv(t̂(v, 𝐩)) * chief_dir(v.lens, 𝐩)

"""
    t̂(camera, 𝐩)

The camera's inverse focus distance at the sensor position 𝐩. EXPERIMENT (focus-ab): the sensor TPS at 𝐩 (kind 0, as before), or a
direction chart's surface at the chief direction of 𝐩 through the camera's lens (kinds 1, 2; focus_charts.jl).
"""
@inline t̂(v::_CameraOrView, 𝐩::StaticVector{2}) = v.focus.kind == 0 ? t̂(v.focus, 𝐩) : t̂_dir(v.focus, chief_dir(v.lens, 𝐩))

"""
    π̂⁻¹(camera, 𝐩, 𝐚) -> SVector{3}

Radiometric extension: unit direction from the aperture point 𝐚 towards 𝔣(𝐩) (camera frame),
computed as `normalize(d̂(𝐩) − t̂(𝐩) 𝐚)` with the unit chief direction d̂ = `chief_dir(𝐩)` (regular for focus at infinity).
`π̂⁻¹(𝐩, 𝔭) = π⁻¹(𝐩)`.
"""
@inline function π̂⁻¹(v::_CameraOrView, 𝐩::StaticVector{2}, 𝐚::StaticVector{3})
    d = chief_dir(v.lens, 𝐩) - t̂(v, 𝐩) * 𝐚
    return d / sqrt(d[1]^2 + d[2]^2 + d[3]^2)
end
"""As above with a hoisted `t̂(𝐩)` (evaluated once per sensor sample by `Φ_pixel`)."""
@inline function π̂⁻¹(v::_CameraOrView, 𝐩::StaticVector{2}, 𝐚::StaticVector{3}, t̂val)
    d = chief_dir(v.lens, 𝐩) - t̂val * 𝐚
    return d / sqrt(d[1]^2 + d[2]^2 + d[3]^2)
end

"""
    world_ray(camera, pose, 𝐩, 𝐚) -> (origin, direction)

Ray of the sensor position 𝐩 through the aperture point 𝐚 of `camera` at `pose` (camera → world), in world coordinates.
"""
@inline world_ray(c::RadiometricCameraModel, pose::EuclideanMap, 𝐩::StaticVector{2}, 𝐚::StaticVector{3}) =
    world_ray(View(pose, c), 𝐩, 𝐚)
@inline function world_ray(v::View, 𝐩::StaticVector{2}, 𝐚::StaticVector{3})
    return v.pose(𝐚), v.pose.R * π̂⁻¹(v, 𝐩, 𝐚)
end
@inline function world_ray(v::View, 𝐩::StaticVector{2}, 𝐚::StaticVector{3}, t̂val)
    return v.pose(𝐚), v.pose.R * π̂⁻¹(v, 𝐩, 𝐚, t̂val)
end
"""As above with the UNIT chief direction `chief_dir(𝐩)` hoisted by the caller (once per sensor sample)."""
@inline function world_ray_dir(v::View, 𝐚::StaticVector{3}, dch::StaticVector{3}, t̂val)
    d = dch - t̂val * 𝐚
    return v.pose(𝐚), v.pose.R * (d / sqrt(d[1]^2 + d[2]^2 + d[3]^2))
end

