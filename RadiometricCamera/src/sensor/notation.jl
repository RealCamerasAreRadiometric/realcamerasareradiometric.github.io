# ---------------------------------------------------------------------------
# Notation and the sensor domain
#
#   ℐ = [0, W] × [0, H]          the sensor in COLMAP pixels of the run's image (pixel i spans [i, i + 1), its centre i + ½):
#                                Core's `Rect(sensor)`, the frame of the cameras' calibration matrices K
#   𝐩 ∈ ℐ                        a sensor position, in those pixels
#   ℐ_𝐮 ⊂ ℐ                     footprint of pixel 𝐮 = (u, v), u ∈ 1:W, v ∈ 1:H (Julia's 1-based pixel indices): the unit square about
#                                its centre `𝐩_of_pixel(u, v)`
#   𝐮, W, H                     pixel index (column u, row v), image size
#
# Julia's pixel convention (the centre of pixel (u, v) at (u, v)) reaches COLMAP's through VisualGeometryToolkitCore's origin map
# `EuclideanMap(PixelFrame{:julia}, PixelFrame{:colmap})`.
# ---------------------------------------------------------------------------

# Julia's pixel convention → COLMAP's, VisualGeometryToolkitCore's origin map (an isbits constant: the kernels read its offset)
const _JULIA_TO_COLMAP = EuclideanMap(VgtkCore.PixelFrame{:julia}, VgtkCore.PixelFrame{:colmap})
const _COLMAP_TO_JULIA = inv(_JULIA_TO_COLMAP)

"""
    𝐩_of_pixel(u, v) -> SVector{2}

Sensor position 𝐩 ∈ ℐ (COLMAP pixels) of the continuous Julia pixel position `(u, v)` (pixel centres at integer `(u, v)`): Core's
origin map.
"""
@inline function 𝐩_of_pixel(u, v)
    T = float(promote_type(typeof(u), typeof(v)))
    t = _JULIA_TO_COLMAP.t
    return SVector(u + T(t[1]), v + T(t[2]))
end
@inline 𝐩_of_pixel(𝐮::StaticVector{2}) = 𝐩_of_pixel(𝐮[1], 𝐮[2])

"""
    pixel_of_𝐩(𝐩) -> SVector{2}

Inverse of [`𝐩_of_pixel`](@ref): continuous Julia pixel position of a sensor position.
"""
@inline function pixel_of_𝐩(𝐩::StaticVector{2})
    T = eltype(𝐩)
    t = _COLMAP_TO_JULIA.t
    return SVector(𝐩[1] + T(t[1]), 𝐩[2] + T(t[2]))
end

"""
    ℐ_𝐮(u, v) -> (lo::SVector{2}, hi::SVector{2})

Bounds of the footprint ℐ_𝐮 ⊂ ℐ of pixel `(u, v)`: its corners `(u ∓ ½, v ∓ ½)` in COLMAP pixels.
"""
@inline ℐ_𝐮(u::Integer, v::Integer) = (𝐩_of_pixel(u - 0.5, v - 0.5), 𝐩_of_pixel(u + 0.5, v + 0.5))

"""The colour channels by name, in the photographs' and every per-channel quantity's order (the lighting rows, the η's,
`io.channel`)."""
const CHANNEL_NAMES = ("R", "G", "B")

"""
    luma(rgb) -> scalar

Rec. 709 luma of an RGB triple (used for the ink indicator fallback).
"""
@inline function luma(ρ::StaticVector{3})
    T = eltype(ρ)
    return T(0.2126) * ρ[1] + T(0.7152) * ρ[2] + T(0.0722) * ρ[3]
end
"""
    luma_image(img) -> Matrix

The Rec. 709 luma plane (`luma`) of an `H × W × 3` image, pixel by pixel; an `H × W × 1` image's one plane as it is.
"""
luma_image(img::AbstractArray{<:Real, 3}) = size(img, 3) == 1 ? img[:, :, 1] :
    [luma(SVector(img[v, u, 1], img[v, u, 2], img[v, u, 3])) for v in axes(img, 1), u in axes(img, 2)]
