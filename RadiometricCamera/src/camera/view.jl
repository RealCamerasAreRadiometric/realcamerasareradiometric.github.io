# ---------------------------------------------------------------------------
# The kernels' record of one photograph: a radiometric camera at a pose.
#
# The radiometric camera (`RadiometricCameraModel`: geometric camera, focus surface, aperture radius, vignetting), its camera-frame
# maps (`chief_direction`, `𝔣`, `t̂`, `π̂⁻¹`, `world_ray`: the paper's symbols are its public aliases of `focus_point`,
# `inverse_focus_distance` and VisualGeometryToolkitCore's `backproject`) and the centre of projection 𝔭 are VisualRadiometryToolkitCore's. A `View` holds
# the camera's parts flattened into one element type with the pose; each map of a view is a one-line forward to
# VisualRadiometryToolkitCore's implementation on the parts it reads — never a second body.
#
# The calibration's START of the vignetting is here too: the flat estimable surface (`flat_vignetting`). A camera composed from its
# geometric camera has NO vignetting (VisualRadiometryToolkitCore's `NoVignetting`: V ≡ 1, no parameters, no Θ_v); a calibration
# estimates V, so its camera is given a field to estimate.
# ---------------------------------------------------------------------------

"""The image rectangle `[0, W] × [0, H]` of a bounded geometric camera (COLMAP pixels): the domain of its focus and vignetting surfaces."""
function _image_rect(geometric::VgtkCore.BoundedCamera)
    W, H = VgtkCore.extent(geometric)
    return VgtkCore.Rect(0, 0, W, H)
end
"""
    _image_size(camera) -> (W, H)

The image size of a camera, in pixels: its geometric camera's `extent` (VisualGeometryToolkitCore's `BoundedCamera`). THE image
size of a render, of its masks and of the pixel ids' stride (`(u − 1) H + v`): it is stated once, by the camera. A kernel that
holds only a bare model of a tile is given the height by its caller, read here from the state's camera.
"""
_image_size(geometric::VgtkCore.BoundedCamera) = (e = Tuple(VgtkCore.extent(geometric)); (Int(e[1]), Int(e[2])))
_image_size(c::RadiometricCameraModel) = _image_size(c.geometric)
_image_width(c) = _image_size(c)[1]
_image_height(c) = _image_size(c)[2]
"""
    flat_vignetting(geometric) -> TensorProductVignetting

The calibration's start of the vignetting on the image of the geometric camera `geometric`: V ≡ 1 as a flat 4 × 4 quadratic B-spline
surface over the image (VisualRadiometryToolkitCore's `TensorProductVignetting((4, 4), Rect(0, 0, W, H); degree = 2)`), the surface a
calibration then estimates. A camera composed from its geometric camera alone has `NoVignetting()` (no Θ_v); a calibration's camera
states the field: `RadiometricCameraModel(geometric; focus, aperture_radius, vignetting = flat_vignetting(geometric))`.
"""
flat_vignetting(geometric::VgtkCore.BoundedCamera) = TensorProductVignetting((4, 4), _image_rect(geometric); degree = 2)

"""
    View{T, N}

Internal: one photograph of a `RadiometricCameraModel` as the renderer's kernels evaluate it — the camera at a pose
(`View(pose, camera)`), flattened into one element type `T` (Float64, Float32 on the device, a dual in the Jacobian evaluations): the
pose (camera → world, Θ_e), the geometric camera Θ_d, the focus surface Θ_f and the aperture radius r. The vignetting is not in it: the kernels
take V apart, as they differentiate it analytically. Callers give the camera and the pose; the public functions form the view.
"""
struct View{T, N, L, P}
    pose::P                                   # camera → world: Core's EuclideanMap in T (`_pose_eltype`; t = the camera centre)
    geometric::L                              # Core's pixel camera (a BoundedCamera of a CameraModel) or a grid lens (Core's spline model / window)
    focus::Focus{T, N}
    r::T
    # inner: no default outer constructor (it would be ambiguous with `View(pose, geometric, focus, r)` below)
    View{T, N, L, P}(pose::P, geometric::L, focus::Focus{T, N}, r::T) where {T, N, L, P} = new{T, N, L, P}(pose, geometric, focus, r)
end

function View(pose::EuclideanMap, geometric, focus::Focus{<:Any, N}, r) where {N}
    T = promote_type(eltype(pose.t), _model_number_type(geometric), eltype(focus.t̂_i), typeof(r))
    foc = VgtkCore.convert_eltype(T, focus)                                          # element-type conversion only, no TPS solve
    geometricT = _model_eltype(T, geometric)
    poseT = _pose_eltype(T, pose)
    return View{T, N, typeof(geometricT)}(poseT, geometricT, foc, T(r))
end
"""The photograph of `camera` at `pose` (camera → world)."""
View(pose::EuclideanMap, c::RadiometricCameraModel) = View(pose, c.geometric, c.focus, c.aperture_radius)
"""A view whose blocks are already in the element type `T` (the device and Jacobian evaluations): the pose's type from the pose."""
View{T, N, L}(pose::P, geometric::L, focus::Focus{T, N}, r) where {T, N, L, P <: EuclideanMap} = View{T, N, L, P}(pose, geometric, focus, r)

Adapt.adapt_structure(to, v::View{T, N}) where {T, N} =
    (l = adapt(to, v.geometric); View{T, N, typeof(l)}(v.pose, l, v.focus, v.r))   # a grid lens carries an array

_image_size(v::View) = _image_size(v.geometric)

# ---- the camera-frame maps of a view: VisualRadiometryToolkitCore's, on the view's parts (one-line forwards)
@inline chief_direction(v::View, 𝐩::StaticVector{2}) = chief_direction(v.geometric, 𝐩)
@inline 𝔣(v::View, 𝐩::StaticVector{2}) = 𝔣(v.geometric, v.focus, 𝐩)
@inline t̂(v::View, 𝐩::StaticVector{2}) = t̂(v.geometric, v.focus, 𝐩)
# (π̂⁻¹ is VisualGeometryToolkitCore's `backproject`: a view's method takes the aperture point as its third argument; the part-wise
# implementation is VisualRadiometryToolkitCore's `aperture_direction`)
@inline π̂⁻¹(v::View, 𝐩::StaticVector{2}, 𝐚::StaticVector{3}) = aperture_direction(v.geometric, v.focus, 𝐩, 𝐚)
"""As above with a hoisted `t̂(𝐩)` (evaluated once per sensor sample by `Φ_pixel`)."""
@inline π̂⁻¹(v::View, 𝐩::StaticVector{2}, 𝐚::StaticVector{3}, t̂val) = aperture_direction(v.geometric, 𝐩, 𝐚, t̂val)
@inline world_ray(v::View, 𝐩::StaticVector{2}, 𝐚::StaticVector{3}) = world_ray(v.pose, 𝐚, π̂⁻¹(v, 𝐩, 𝐚))
@inline world_ray(v::View, 𝐩::StaticVector{2}, 𝐚::StaticVector{3}, t̂val) = world_ray(v.pose, 𝐚, π̂⁻¹(v, 𝐩, 𝐚, t̂val))
"""The view's ray of the aperture point 𝐚 with the UNIT chief direction `chief_direction(𝐩)` hoisted by the caller (once per sensor sample)."""
@inline world_ray_dir(v::View, 𝐚::StaticVector{3}, dch::StaticVector{3}, t̂val) = world_ray(v.pose, 𝐚, aperture_direction(dch, 𝐚, t̂val))
