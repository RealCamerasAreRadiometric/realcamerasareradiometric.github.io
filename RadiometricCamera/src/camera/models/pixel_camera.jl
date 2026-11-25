# ---------------------------------------------------------------------------
# The parametric chief-ray models ARE VisualGeometryToolkitCore's pixel cameras: a `BoundedCamera` of a `CameraModel(LogicalIntrinsics(K),
# lens)` — K in COLMAP pixels of the run's image, any Core lens (`Lens(DoubleSphere(ξ, α))`, `KannalaBrandt(k1, k2, k3, k4)`,
# BlobBoards' `Division(d1, d2)`, the 14-coefficient chain of VisualGeometryToolkitIO's `opencv_camera`, `Lens(Perspective())`, …) — bounded by
# the image's extent or its `Sensor`; isbits, so it travels into a device kernel as it is. Every fact of the model is Core's: π⁻¹ is its
# `backproject` (NaN where the lens does not image the point), the forward projection its `project`, the parameters its
# `parameter_vector` / `parameter_names` / `with_parameters` (`f, aspect, skew, cx, cy` in pixels, then the lens's), the domain its
# `parameter_bounds`, the element type its `convert_eltype`, where the model is valid its `valid_domain` (where light falls is
# measured, VisualGeometryToolkitFeatures' `detect_image_circle`). Its LM block is Core's
# `ParameterManifold` of the model with the LM's search box and the image-size coordinate scale (`lens_manifold`, per_view.jl). What is
# RadiometricCamera's: the box's numbers (`lm_box`), never a fact of a lens. A fixed model (an OpenCV fit to be sampled into a grid) is
# the same camera with no parameter freed.
# ---------------------------------------------------------------------------

"""A parametric chief-ray model: Core's bounded pixel camera of a `CameraModel`."""
const _PixelCamera = VgtkCore.BoundedCamera{<:CameraModel}

# (the image formation's chief ray of a pixel camera is VisualRadiometryToolkitCore's `chief_ray`: Core's `backproject(camera, 𝐩)`)

"""Whether the camera images the sensor position: its back-projection is finite there (a grid's: its 4 × 4 window holds present
nodes) and, when its lens coverage is an image circle (Core's `lens_coverage`: a `Circle` or an `Ellipse`, the run's measured
one), inside it. A `Rect` coverage (the whole sensor, the default) bounds no position: a position off the sensor that the lens
images stays imaged, as the model's own domain says."""
@inline function in_domain(c::VgtkCore.BoundedCamera, 𝐩::StaticVector{2})
    _coverage_bounds(VgtkCore.lens_coverage(c), 𝐩) || return false
    d = _direction(backproject(c, 𝐩))
    return isfinite(d[1]) && isfinite(d[2]) && isfinite(d[3])
end
_coverage_bounds(::VgtkCore.Rect2, 𝐩) = true
_coverage_bounds(region, 𝐩) = 𝐩 ∈ region
_direction(d::StaticVector{3}) = d
_direction(r::VgtkCore.Ray3) = r.direction

"""The camera in the element type `T` (Float32, or a dual type for a Jacobian evaluation): Core's `convert_eltype` (any `T <: Real`; K and
the lens's coefficients converted, the bound kept)."""
_model_eltype(::Type{T}, c::_PixelCamera) where {T} = eltype(VgtkCore.model(c).intrinsics.K) === T ? c : VgtkCore.convert_eltype(T, c)
"""The number type of a camera's parameters."""
_model_number_type(c::_PixelCamera) = eltype(VgtkCore.model(c).intrinsics.K)

# ---- the LM's search box and coordinate scale ----------------------------------------------------------------------------------
"""
    lm_box(camera, W, H) -> NamedTuple

The Levenberg–Marquardt search box of a pixel camera's parameters of a `W × H` image, by Core's parameter names (`f, aspect, skew, cx,
cy`, then the lens's `parameter_names`): a solver option, not a fact of any lens — the `box` of Core's `ParameterManifold`, which
intersects it with the model's domain (`parameter_bounds`: f > 0, aspect > 0, the double sphere's ξ > −1 and α ∈ [0, 1]) and clamps
every LM step into it. f between 0.1 and 50 image widths (a 180° fisheye to a 400 mm telephoto on APS-C; a cap of 3 widths once
clamped a 65 mm lens and stalled the LM); the aspect `f_y/f_x` 0.1 … 10 (1 for square pixels); the principal point within 0.3 of the
image's width and height of its centre; the double sphere's ξ ∈ [−0.95, 1.5] (the Siemens 17 mm lens fits ξ ≈ −0.6); OpenCV's sensor
tilt within ±0.5 rad; any other parameter within ±10³.
"""
function lm_box(camera::CameraModel, W::Real, H::Real)
    box = (; f = (0.1W, 50.0W), aspect = (0.1, 10.0), cx = (0.2W, 0.8W), cy = (0.2H, 0.8H), xi = (-0.95, 1.5), tx = (-0.5, 0.5), ty = (-0.5, 0.5))
    names = VgtkCore.parameter_names(camera)
    return NamedTuple{names}(map(n -> Float64.(get(box, n, (-1.0e3, 1.0e3))), names))
end
"""
    lm_scale(W, H) -> NamedTuple

The coordinate scale of a pixel camera's LM block (Core's `ParameterManifold(camera; scale)`): one coordinate of the principal point
(and of the skew) per image width and height, so the LM's uniform damping λ·I weighs it like the log-charted focal length and the
unit-free lens coefficients (the hoist gate measured the Jacobian of the scaled pixel camera equal to the normalized image's to
2.2e-16).
"""
lm_scale(W::Real, H::Real) = (; skew = Float64(W), cx = Float64(W), cy = Float64(H))

