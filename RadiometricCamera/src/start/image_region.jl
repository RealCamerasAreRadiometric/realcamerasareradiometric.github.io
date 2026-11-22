# ---- the imaged region: the one support of a run ---------------------------------------------------------------------------
# The lens coverage of a capture set (where light falls on the sensor) is ONE keyed step of the set, never a number carried by
# hand (owner ruling 2026-11-15): `measure_image_region` measures it on the set's own photographs (Features' `detect_image_circle`,
# on the device), `save_image_region` writes the `rc/image_region/v1.0` record (records.jl, IO's region body) and every consumer
# reads that record — a run through `camera.lens_coverage` (`run_lens_coverage`). A set whose lens lights the whole sensor states
# `camera.lens_coverage = "sensor"`: no record, the detector never runs (it fits spurious ellipses on fully lit board sets).
"""The sensor of a run: Core's `Sensor` of the `W × H` image whose pitch is the sensor width `camera.sensor_mm` over `W` pixels."""
run_sensor(opts::PipelineOptions, W::Integer, H::Integer) = VgtkCore.Sensor(VgtkCore.Size2(Int(W), Int(H)), _sensor_width(opts) / W)

"""
    measure_image_region(files; image, outlier_halfwidth_px, seed, device = default_backend(), detect = detect_image_circle) -> NamedTuple

The image region of a capture set — where light falls on the sensor (the lens's lit region, an `Ellipse` in COLMAP pixels of the
raw grid) — MEASURED on the set's own photographs `files` (RAW files, or full-resolution decoded `.png`), never predicted from a lens
(a model's fold says where the model is valid, not where the field stop cuts the light): VisualGeometryToolkitFeatures'
`detect_image_circle` (`detect`), streamed over the frames — the lens coverage's OUTER edge, the RANSAC ellipse (seeded by
`StableRNG(seed)`: the same region every run) through each angle's log-brightness drops on the frames' per-pixel maximum, resampled
log-polar about the image centre on `device`. `image` is the geometry image measured on (one of [`GEOMETRY_IMAGES`](@ref), the
config's statement; no default): `"raw_green"` a RAW frame's bilinear green on the raw grid ([`raw_green_image`](@ref), full
resolution — the image every geometry detector reads, owner 2026-11-16), `"luma"` a decoded `.png` (a synthetic render) as stored, as
Rec. 709 luma, on the raw grid. Both are on the raw grid (their frame map onto the raw is the identity), so the ellipse and
`outlier_halfwidth_px` are raw pixels as given, nothing scaled. A `"raw_green"` set of decoded frames, a `"luma"` set of RAW frames or
a mixed set is refused; no ellipse found is an error (the set's config must then state `camera.lens_coverage = "sensor"` or name other
frames). Returns `(; lens_coverage, extent, frames, center_scale, axes_scale, inliers, candidates, rim_gap, cut, seconds)`, the fields
of the record ([`save_image_region`](@ref)).
The measurement is GEOMETRIC (a rim threshold on log-brightness drops): its frames are read under the file's black, no sensor model —
exempt from the observation model (every photograph a camera is calibrated on or predicts goes through `photo_set`).
"""
function measure_image_region(files::AbstractVector{<:AbstractString}; image::AbstractString, outlier_halfwidth_px::Real, seed::Integer,
                              device = default_backend(), detect = VgtkFeatures.detect_image_circle)
    isempty(files) && throw(ArgumentError("measure_image_region: no photographs"))
    check_geometry_image(Dict("image" => image), "measure_image_region")
    isfinite(outlier_halfwidth_px) && outlier_halfwidth_px > 0 ||
        throw(ArgumentError("measure_image_region: outlier_halfwidth_px = $outlier_halfwidth_px: a half-width > 0 (raw px)"))
    raw = is_raw_file.(files)
    all(raw) || !any(raw) || throw(ArgumentError("measure_image_region: RAW and decoded frames mixed in $(join(basename.(files), ", "))"))
    (image == "raw_green") == all(raw) ||
        throw(ArgumentError("measure_image_region: image = $(repr(image)) reads $(image == "raw_green" ? "RAW frames" : "decoded frames"), " *
                            "$(basename(first(files))) … are $(all(raw) ? "RAW" : "decoded")"))
    hw = Float64(outlier_halfwidth_px)                                # raw px: both images are on the raw grid
    if image == "raw_green"                                           # the bilinear green on the raw grid, the file's black (no model)
        frames = (raw_green_image(VgtkFeatures.load_raw_mosaic(f); sensor_model = VgtkCore.SensorModel(), device) for f in files)
        extent = Tuple(raw_image_size(first(files)))
    else                                                              # full-resolution decodes on the raw grid
        frames = (luma_image(VgtkFeatures.load_linear_image(f)) for f in files)   # its Rec. 709 luma
        img = VgtkFeatures.load_linear_image(first(files))
        extent = (size(img, 2), size(img, 1))
    end
    t = time()
    res = detect(StableRNG(seed), frames; outlier_halfwidth = hw, device)
    res === nothing && throw(ArgumentError("measure_image_region: detect_image_circle found no image region in $(length(files)) photographs " *
                                           "($(basename(first(files))) …): fewer than 5 rim candidates, no consensus ellipse through them, or the " *
                                           "coverage contains the whole frame — a lens that lights the whole sensor is stated `lens_coverage = \"sensor\"`"))
    e = res.circle                                                    # an Ellipse (detect_image_circle's default model), raw pixels
    mask = VgtkCore.inlier_mask(res.fit)
    return (; lens_coverage = e, extent = (Int(extent[1]), Int(extent[2])), frames = String.(basename.(files)),
            center_scale = Float64.(collect(res.center_scale)), axes_scale = Float64.(collect(res.axes_scale)),
            inliers = count(mask), candidates = length(mask), rim_gap = Float64(res.rim_gap), cut = Bool(res.cut), seconds = time() - t)
end

"""
    run_lens_coverage(opts, W, H) -> Union{Nothing, Ellipse{Float64}}

The lens coverage of a run on `W × H` photographs, as its options state it (`camera.lens_coverage`): `nothing` for `"sensor"` (the
sensor's rectangle, Core's default: the lens lights the whole sensor; nothing measured), else the measured image region of the
`rc/image_region/v1.0` record the option names ([`load_image_region`](@ref): the capture set's keyed step) — refused when the record
is missing or of other photographs (its extent not `W × H`). No other source: a grid start's stored coverage is not read.
"""
function run_lens_coverage(opts::PipelineOptions, W::Integer, H::Integer)
    spec = opts.camera.lens_coverage
    if spec == SENSOR_LENS_COVERAGE
        println("lens coverage: the sensor (camera.lens_coverage = \"sensor\"): the whole sensor, nothing measured")
        return nothing
    end
    isfile(spec) || throw(ArgumentError("camera.lens_coverage = $(repr(spec)): no image-region record there — the capture set's image " *
                                        "region is a step of its own (measure it first), or the lens lights the whole sensor: \"sensor\""))
    r = load_image_region(spec)
    r.extent == (Int(W), Int(H)) ||
        throw(ArgumentError("camera.lens_coverage = $spec: an image region of $(r.extent[1]) × $(r.extent[2]) photographs, the run's are $W × $H"))
    e = r.lens_coverage
    @printf("lens coverage: the measured image region %s (%d frames: %s) — centre (%.2f, %.2f), semi-axes %.2f / %.2f px at %.2f°\n",
            spec, length(r.frames), join(r.frames, ", "), e.center..., e.a, e.b, rad2deg(e.θ)); flush(stdout)
    return e
end

"""Whether the COLMAP pixel position `(x, y)` lies in the region."""
_in_region(region, x, y) = VgtkCore.Point2{Float64}(x, y) ∈ region
