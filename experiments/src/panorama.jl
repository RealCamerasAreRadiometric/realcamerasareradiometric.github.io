# =============================================================================
# src/panorama.jl — module Panorama: what the panorama_stitching experiment
# adds to PanoramaStitching's direct sampling (scripts/panorama_stitching.jl
# runs the stages; src/panorama_example.jl binds PanoramaStitching's example of
# record, whose decode and registration the driver reads).
#
# A pure-rotation sequence of a distant scene sees one world direction d at a
# different sensor point in every frame k: I_k(p) = L(R_k·f(p)) · V(u(p)). A
# panorama composed from the frames as they are therefore shows the lens's
# vignetting as a brightness difference wherever two frames meet; composed from
# the frames divided by the right V it shows none. The stage panoramas renders
# both and scores the seams:
#
#   the candidates   no correction ([`no_vignetting`](@ref)) and a learned
#                    vignetting sampled on a regular grid of the sensor
#                    ([`vignetting_map`](@ref)) as a vignetting at a sensor point
#                    ([`MapVignetting`](@ref)), divided by its mean over the
#                    central ring of the field ([`ring_mean_map`](@ref)), so that
#                    it is 1 on the axis as the uncorrected frames' V ≡ 1
#   the panoramas    [`sweep_panoramas`](@ref): per candidate the frames' gains,
#                    PanoramaStitching's `score_seams` and `render_panorama`;
#                    [`panorama_exposure`](@ref): the one exposure the
#                    candidates' images are written at
#
# Every map is sampled as the learned maps are: one value per `spacing` camera
# pixels, at the centre ((j − ½)·spacing, (i − ½)·spacing). Rings are of the
# equisolid field angle 2 asin(ρ / 2f) about the centre of the sweep's
# vignetting. This module depends on the stack and PanoramaStitching only (no
# RadiometricCamera).
# =============================================================================
module Panorama

using Statistics: quantile
using StaticArrays: SVector
import VisualGeometryToolkitCore as VgtkCore
using VisualGeometryToolkitCore: bilinear_sample
import PanoramaStitching

export MapVignetting, no_vignetting, vignetting_map, ring_mean_map, equisolid_theta, sweep_panoramas, panorama_exposure

# ---- the candidates as vignettings at a sensor point ------------------------------------------------------------------------------

"No vignetting correction: V ≡ 1 at every sensor point `u` and ray `f`."
no_vignetting(u, f) = 1.0

"""
    MapVignetting(M, image_size, scale)

A vignetting map on a regular grid (`M`: rows y, columns x, one value per `spacing = H / rows` camera pixels at the centre
`((j − ½)·spacing, (i − ½)·spacing)`; `image_size = (W, H)` camera pixels) as PanoramaStitching's vignetting at a sensor point,
`v(u, f) = bilinear(M, u / (W, H)) / scale` (Core's `bilinear_sample`, edge-clamped; the ray `f` is not read). `scale` normalizes it
(its central ring's mean). A host array: the score and the render that read it run on the host.
"""
struct MapVignetting{A <: AbstractMatrix}
    M::A
    W::Float64
    H::Float64
    scale::Float64
end
MapVignetting(M::AbstractMatrix, image_size, scale::Real) = MapVignetting(M, Float64(image_size[1]), Float64(image_size[2]), Float64(scale))
(v::MapVignetting)(u, f) = bilinear_sample(v.M, SVector(u[1] / v.W, u[2] / v.H)) / v.scale

"""
    vignetting_map(v, spacing, image_size) -> Matrix

The vignetting `v` (any `v(u, f)`: a VisualRadiometryToolkitCore vignetting model, a [`MapVignetting`](@ref)) on the learned maps'
grid: `H ÷ spacing` rows by `W ÷ spacing` columns of `image_size = (W, H)` camera pixels, each at its centre
`((j − ½)·spacing, (i − ½)·spacing)`.
"""
vignetting_map(v, spacing::Integer, image_size) =
    [Float64(v(SVector((j - 0.5) * spacing, (i - 0.5) * spacing), nothing)) for i in 1:(image_size[2] ÷ spacing), j in 1:(image_size[1] ÷ spacing)]

# ---- rings -----------------------------------------------------------------------------------------------------------------------

"The equisolid field angle (degrees) of a radius `r` px about the centre of the rings: `2 asin(r / 2f)`."
equisolid_theta(r, f_px) = rad2deg(2 * asin(clamp(r / (2 * f_px), 0, 1)))

# the grid points of a map: (x, y) camera pixels, about the circle's centre
function _grid_points(M::AbstractMatrix, circle, image_size)
    cx, cy, _ = circle; spacing = image_size[2] / size(M, 1)
    return ((i, j, (j - 0.5) * spacing, (i - 0.5) * spacing) for j in axes(M, 2), i in axes(M, 1)), cx, cy
end

"""
    ring_mean_map(M, circle, f_px, image_size, lo, hi, tmax) -> Vector

Ring means of a map `M` (rows y, columns x, one value per `spacing = H / rows` px at the centre `(j − ½)·spacing`) over the disc
`circle = (cx, cy, r)` (px) by the equisolid field angle `2 asin(ρ / 2f)` about its centre: ring q holds the grid points of field
angle `lo[q] ≤ θ < hi[q]` (degrees), each ring cut at `tmax`. NaN for a ring without a grid point.
"""
function ring_mean_map(M::AbstractMatrix, circle, f_px, image_size, lo, hi, tmax)
    pts, cx, cy = _grid_points(M, circle, image_size); rc = circle[3]
    acc = zeros(length(lo)); n = zeros(Int, length(lo))
    for (i, j, X, Y) in pts
        r = hypot(X - cx, Y - cy)
        r < rc || continue
        t = equisolid_theta(r, f_px)
        for q in eachindex(lo)
            if lo[q] <= t < min(hi[q], tmax)
                acc[q] += M[i, j]; n[q] += 1
            end
        end
    end
    return [n[q] > 0 ? acc[q] / n[q] : NaN for q in eachindex(lo)]
end

# ---- the panoramas ---------------------------------------------------------------------------------------------------------------

"""
    sweep_panoramas(sweep, candidates; gains, seam_deg_per_pixel, deg_per_pixel, chunk, plane, rim_margin_px, fov_deg,
                    saturation_margin_px, say = println) -> (; images, candidates, extent)

The panoramas of one pure-rotation sweep under each candidate vignetting, and their seam scores. `sweep` is the registered sweep:
`lens` (a Core `BoundedCamera`), `frames` (H × W × 3 linear images on the host), `saturated` (their masks), `rotations` (frame k's
camera → the common frame), `frame_map` and `plane_maps` (the frames' and the colour planes' maps onto the camera's pixels).
`candidates` are `(slug, label, v)`, `v(u, f)` a vignetting at a sensor point.

Per candidate: the frames' gains `gains(v)` (K × 3, per frame and colour plane: ones for one exposure, or a fit with `v` held
fixed); PanoramaStitching's `score_seams` of colour plane `plane` on the equirectangular raster of `seam_deg_per_pixel` (sampled in
chunks of `chunk` pixels; the saturation masks dilated by `saturation_margin_px`); and its `render_panorama` on the raster of
`deg_per_pixel`. Both rasters span the sweep's coverage (`coverage_extent`) in its level frame (`up_frame(sweep_up(rotations).up)`);
`rim_margin_px` and `fov_deg` bound every sample as they bound the sweep's own fit.

Returns `images` (`slug => ` the rendered image, rows × columns × 3, NaN where no frame sees the pixel), `candidates` (`slug => `
its record: `label`, `seam_rms`, `seam_mad`, `seam_pixels`, `gains`, one `[g_R, g_G, g_B]` per frame) and the rasters' `extent`
(lon₀, lon₁, lat₀, lat₁, degrees). `say` receives one line per candidate.
"""
function sweep_panoramas(sweep, candidates; gains, seam_deg_per_pixel::Real, deg_per_pixel::Real, chunk::Integer, plane::Integer,
                         rim_margin_px::Real, fov_deg::Real, saturation_margin_px::Real, say = println)
    up = PanoramaStitching.up_frame(PanoramaStitching.sweep_up(sweep.rotations).up)   # the sweep's level frame
    extent = PanoramaStitching.coverage_extent(sweep.lens, sweep.frames, sweep.rotations, up, sweep.frame_map)
    sview = PanoramaStitching.panorama_view(VgtkCore.Equirectangular(), extent, seam_deg_per_pixel; up)
    pview = PanoramaStitching.panorama_view(VgtkCore.Equirectangular(), extent, deg_per_pixel; up)
    # the camera is the sweep's BoundedCamera: its forward map and its imaged region (the sensor ∩ its lens coverage) bound every
    # sample; the frames' and the planes' maps onto its pixels are the decode's
    mkw = (; saturated = sweep.saturated, frame_map = sweep.frame_map, rim_margin_px, fov_deg, plane_maps = sweep.plane_maps)
    images = Dict{String, Any}(); records = Dict{String, Any}()
    for (slug, label, v) in candidates
        g = gains(v)
        s = PanoramaStitching.score_seams(sweep.lens, sweep.frames, sweep.rotations, sview; vignetting = v, gains = g, plane, chunk,
                                          saturation_margin_px, mkw...)
        images[slug] = Array(PanoramaStitching.render_panorama(sweep.lens, sweep.frames, sweep.rotations, pview; vignetting = v, gains = g,
                                                               frame_map = mkw.frame_map, rim_margin_px = mkw.rim_margin_px, fov_deg = mkw.fov_deg,
                                                               plane_maps = mkw.plane_maps).image)
        records[slug] = Dict{String, Any}("label" => label, "seam_rms" => s.rms, "seam_mad" => s.mad, "seam_pixels" => s.n,
                                          "gains" => [collect(r) for r in eachrow(g)])
        say("  $(rpad(slug, 15)) $label: gains $(round.(g; digits = 3)); seams rms $(round(s.rms; digits = 4)), 1.4826·MAD $(round(s.mad; digits = 4)) ($(s.n) px)")
    end
    return (; images, candidates = records, extent)
end

"""
    panorama_exposure(image, percentile) -> Float64

The exposure the candidates' panoramas are written at: 1 over the `percentile` (%) of the green plane of `image` (rows × columns × 3
linear; the pixels no frame sees, NaN, left out).
"""
panorama_exposure(image::AbstractArray{<:Real, 3}, percentile::Real) = 1 / quantile(filter(!isnan, vec(view(image, :, :, 2))), percentile / 100)

end # module Panorama
