# # The imaged region of a camera, measured from its photographs
#
# Where light falls on the sensor is measured, never predicted from a lens model: RadiometricCamera's `measure_image_region` (the
# capture set's image-region step: VisualGeometryToolkitFeatures' `detect_image_circle`, its fit through
# VisualGeometryToolkitEstimation, loaded here) finds the image region (an `Ellipse`) in a session's photographs, VisualGeometryToolkitCore's `imaged_region` intersects it with the sensor (a disc cut
# by the frame is closed by the frame's edges), and `pixel_mask` rasterizes it — the pixels whose centres lie in the region. The
# figure overlays the mask on the photographs' per-pixel maximum.
#
# Run: `julia --project=experiments examples/image_circle_mask.jl <frames> [out.png]` — `<frames>` a directory of `.png`
# photographs of one camera, lens and focus (a few frames, each lighting a different part of the rim, are enough), read as linear
# images; `out.png` defaults to `examples/output/image_circle_mask.png`. (The experiments environment carries CairoMakie.)

using RadiometricCamera, Printf
using RadiometricCamera: lens, luma_image
using CairoMakie: Figure, Axis, image!, lines!, Label, save, RGBAf, DataAspect, Point2f
import VisualGeometryToolkitCore as VgtkCore
import VisualGeometryToolkitFeatures as VgtkFeatures
using VisualGeometryToolkitEstimation                 # detect_image_circle's fit: Features' extension on Estimation

frames_dir = get(ARGS, 1, "")
isdir(frames_dir) || error("usage: julia --project=experiments examples/image_circle_mask.jl <directory of .png photographs> [out.png]")
out = get(ARGS, 2, joinpath(@__DIR__, "output", "image_circle_mask.png"))
files = sort!(filter(f -> endswith(lowercase(f), ".png"), readdir(frames_dir; join = true)))
isempty(files) && error("no .png photographs in $frames_dir")

# the photographs as linear gray images (Rec. 709 luma of the linear RGB), streamed: one camera, so one size
frames() = (luma_image(VgtkFeatures.load_linear_image(f)) for f in files)

# 1. the image region, measured (an Ellipse; the seed gives the same region every run): the step a capture set's config names
#    (`[image_regions]` of the paper's configs), on the device
m = RadiometricCamera.measure_image_region(files; outlier_halfwidth_px = 500.0, seed = 1)   # green-cell px (the configs of record's): 1000 full-resolution px on .png frames
e = m.lens_coverage
@printf("image region: centre (%.1f, %.1f), semi-axes %.1f / %.1f px at %.1f°; %d of %d rim candidates inliers%s; largest rim gap %.0f°\n",
        e.center..., e.a, e.b, rad2deg(e.θ), m.inliers, m.candidates, m.cut ? ", cut by the frame" : "", rad2deg(m.rim_gap))

# 2. the imaged region: the sensor ∩ the image circle, and its pixel mask
H, W = size(first(frames()))
region = VgtkCore.Rect(0, 0, W, H) ∩ e          # imaged_region(camera, e) for a camera bounded by this W × H extent
mask = VgtkCore.pixel_mask(region, W, H)         # H × W: mask[v, u] is pixel (u, v), centre (u − ½, v − ½)
@printf("imaged region: %.1f %% of the %d × %d sensor\n", 100 * count(mask) / length(mask), W, H)

# 3. the figure: the photographs' per-pixel maximum, the masked part tinted red, the image circle in orange
upper = reduce((a, b) -> max.(a, b), frames())
t = range(0, 2π; length = 721)
rim = Point2f.(e.center[1] .+ e.a .* cos.(t) .* cos(e.θ) .- e.b .* sin.(t) .* sin(e.θ),
               e.center[2] .+ e.a .* cos.(t) .* sin(e.θ) .+ e.b .* sin.(t) .* cos(e.θ))
fig = Figure(size = (1400, round(Int, 1400 * H / W) + 80))
ax = Axis(fig[1, 1]; aspect = DataAspect(), yreversed = true, title = basename(frames_dir))
image!(ax, (0, W), (0, H), permutedims(upper); colormap = :grays, colorrange = (0, maximum(upper) / 4), interpolate = false)
image!(ax, (0, W), (0, H), permutedims(Float32.(.!mask)); colormap = [RGBAf(0, 0, 0, 0), RGBAf(1.0, 0.2, 0.1, 0.45)], interpolate = false)
lines!(ax, rim; color = :orange, linewidth = 2)
Label(fig[2, 1], "red: outside the imaged region (masked); orange: the measured image circle", fontsize = 13)
mkpath(dirname(out)); save(out, fig)
println("saved $out")
