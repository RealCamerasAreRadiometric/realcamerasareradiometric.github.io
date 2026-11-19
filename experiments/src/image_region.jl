# =============================================================================
# src/image_region.jl — the lens coverage of a capture set as ONE keyed step (owner ruling 2026-11-15): no coverage number is
# carried by hand (no circle literal in a config, no ellipse copied from a log, no shared image-circle PNG). Every config of record
# states the coverage of each set it reads explicitly, as `lens_coverage`:
#
#   lens_coverage = "sensor"                     # the lens lights the whole sensor: its rectangle; nothing is measured (the detector
#                                                #   is never run: it fits spurious ellipses on fully lit board sets)
#   lens_coverage = "image_region:<name>"        # the set's measured image region, the step of the config's [image_regions.<name>]
#
#   [image_regions.<name>]
#   input = "<inputs id>"                        # the folder of the set's own photographs (RAW, or a synthetic set's renders)
#   frames = { "DSC04161.ARW" = "<sha256>" }     # the frames measured on, each pinned by its content (checked when measured)
#   image = "raw_green"                          # the geometry image measured on (no default): "raw_green" a RAW set's bilinear green
#                                                #   on the raw grid (owner 2026-11-16), "luma" a decoded set's (a synthetic render)
#   outlier_halfwidth_px = 1000.0                # detect_image_circle's outlier half-width [raw px], a literal with its derivation
#   seed = 1                                     # the RANSAC ellipse fit's StableRNG seed
#
# The step (`image_region_product`) is VisualGeometryToolkitEval's Product <results>/image_region/detect/<key>, keyed by ONLY what it
# reads — the frames (the folder's tier and path, each frame's sha256), the detector's settings and the code of the packages it runs
# (`IMAGE_REGION_CODE`) — never by the config or the region's name: two configs naming the same frames read the one step.
# scripts/image_regions.jl measures every region its config's configs name (RadiometricCamera's `measure_image_region`: Features'
# detect_image_circle on the device), writing the rc/image_region/v1.0 record (RadiometricCamera's `save_image_region`, through
# IO); every consumer only reads it (`lens_coverage_record`, `lens_coverage_region`), and one whose record is missing fails, naming
# the stage to run. Included at the end of src/drivers.jl.
# =============================================================================
using Printf: @printf
import StaticArrays

"The keys of an `[image_regions.<name>]` table, every one required."
const IMAGE_REGION_KEYS = ("input", "frames", "image", "outlier_halfwidth_px", "seed")
"A `lens_coverage` naming a measured image region of the config: `image_region:<name>`."
const IMAGE_REGION_REF = r"^image_region:([A-Za-z0-9_]+)$"
"The record a measured image region's step writes (rc/image_region/v1.0), the file every consumer reads."
const IMAGE_REGION_FILE = "image_region.json"
"The driver that measures every image region, and its config of record (what an on-demand measurement records as its producer)."
const IMAGE_REGIONS_DRIVER = "scripts/image_regions.jl"
const IMAGE_REGIONS_CONFIG = joinpath("configs", "experiments", "image_regions.toml")
"""The packages the image-region step runs (the per-package rule of the code a product is keyed by): RadiometricCamera's
`measure_image_region` and its record, Features' detect_image_circle (its ellipse fit Estimation's), on Core and IO."""
const IMAGE_REGION_CODE = ("VisualGeometryToolkitCore", "VisualGeometryToolkitIO", "VisualGeometryToolkitFeatures",
                           "VisualGeometryToolkitEstimation", "RadiometricCamera")
"""The shared image-circle photographs of the configs before 2026-11-15 (a tier folder of PNG links, read by several sets): retired,
refused by name as an input — each set's region is measured on its own frames."""
const RETIRED_IMAGE_CIRCLE_INPUTS = "vignetting/image_circle"

"""
    check_image_regions(cfg, path)

Every `[image_regions.<name>]` of `cfg` takes exactly `IMAGE_REGION_KEYS`: `input` an `[inputs]` id (the folder of the set's
photographs), `frames` a non-empty table of file name = its sha256 (64 hex digits), `image` a geometry image (RadiometricCamera's
`check_geometry_image`: required, no default), `outlier_halfwidth_px` > 0 (raw px) and `seed` an
integer; an `ArgumentError` naming the table otherwise.
"""
function check_image_regions(cfg::AbstractDict, path::AbstractString)
    for (n, t) in get(cfg, "image_regions", Dict{String, Any}())
        where = "$path [image_regions.$n]"
        occursin(r"^[A-Za-z0-9_]+$", n) || throw(ArgumentError("$where: a name of letters, digits and _"))
        check_required_keys(t, IMAGE_REGION_KEYS, where)
        haskey(get(cfg, "inputs", Dict()), t["input"]) || throw(ArgumentError("$where input: $(repr(t["input"])) is not an [inputs] id"))
        f = t["frames"]
        f isa AbstractDict && !isempty(f) && all(((k, v),) -> !occursin('/', k) && v isa AbstractString && occursin(r"^[0-9a-f]{64}$", v), f) ||
            throw(ArgumentError("$where frames: a table of file name = its sha256 (64 lower-case hex digits), not empty"))
        RadiometricCamera.check_geometry_image(t, where)
        h = t["outlier_halfwidth_px"]
        h isa Real && isfinite(h) && h > 0 || throw(ArgumentError("$where outlier_halfwidth_px: a half-width > 0 (raw px)"))
        t["seed"] isa Integer || throw(ArgumentError("$where seed: an integer"))
    end
    return nothing
end

"""
    check_lens_coverage(cfg, v, where) -> String

A `lens_coverage` statement `v` of `cfg`: `"sensor"`, or `"image_region:<name>"` with an `[image_regions.<name>]` table; an
`ArgumentError` naming `where` otherwise (every config of record states it: there is no default).
"""
function check_lens_coverage(cfg::AbstractDict, v, where::AbstractString)
    v == RadiometricCamera.SENSOR_LENS_COVERAGE && return String(v)
    m = v isa AbstractString ? match(IMAGE_REGION_REF, v) : nothing
    m === nothing && throw(ArgumentError("$where lens_coverage = $(repr(v)): \"sensor\" (the lens lights the whole sensor) or " *
                                         "\"image_region:<name>\" (the set's measured image region, [image_regions.<name>])"))
    haskey(get(cfg, "image_regions", Dict()), m.captures[1]) ||
        throw(ArgumentError("$where lens_coverage = $(repr(v)): no [image_regions.$(m.captures[1])]"))
    return String(v)
end

"The name of the measured image region a `lens_coverage` statement names, or `nothing` for the sensor."
lens_coverage_name(v::AbstractString) = (m = match(IMAGE_REGION_REF, v); m === nothing ? nothing : String(m.captures[1]))

"""
    image_region_product(cfg, name) -> Product

The step of `[image_regions.<name>]` of `cfg`: keyed by its frames (the input's tier and path, each frame's sha256), the detector's
settings (its geometry image among them) and the code of `IMAGE_REGION_CODE` — not by the config, the region's name or the thread count (one measurement): two configs
naming the same frames share it. Its output: the record `IMAGE_REGION_FILE`.
"""
function image_region_product(cfg::AbstractDict, name::AbstractString)
    t = cfg["image_regions"][name]
    i = cfg["inputs"][t["input"]]
    rec = Dict{String, Any}("image_region" => Dict{String, Any}("folder" => Dict{String, Any}("tier" => i["tier"], "path" => i["path"]),
                                                               "frames" => Dict{String, Any}(String(k) => String(v) for (k, v) in t["frames"]),
                                                               "image" => String(t["image"]),
                                                               "outlier_halfwidth_px" => Float64(t["outlier_halfwidth_px"]), "seed" => Int(t["seed"])))
    return Product("image_region", "detect", rec; output = IMAGE_REGION_FILE, code = code_of(IMAGE_REGION_CODE))
end

"""The keying record of a `lens_coverage` statement (what a consumer's product is keyed by): `"sensor"`, or the key of the image
region's step."""
lens_coverage_key(cfg::AbstractDict, v::AbstractString) =
    (n = lens_coverage_name(v); n === nothing ? String(v) : "image_region:" * image_region_product(cfg, n).key)

"""
    image_region_frames(cfg, name) -> Vector{String}

The frames of `[image_regions.<name>]`, absolute paths in its input's folder (sorted), each checked to be there with its pinned
sha256 (VisualGeometryToolkitEval's `file_sha256`): another content is an `ArgumentError` naming the frame.
"""
function image_region_frames(cfg::AbstractDict, name::AbstractString)
    t = cfg["image_regions"][name]
    dir = config_input(cfg, t["input"])
    files = String[]
    for (f, want) in sort(collect(t["frames"]); by = first)
        p = joinpath(dir, f)
        isfile(p) || throw(ArgumentError("[image_regions.$name]: the frame $f is not in $dir"))
        got = VisualGeometryToolkitEval.file_sha256(p)
        got == want || throw(ArgumentError("[image_regions.$name]: the frame $p has sha256 $got, the config pins $want"))
        push!(files, p)
    end
    return files
end

"""
    run_image_region(cfg, name; driver, config_path, detect = VgtkFeatures.detect_image_circle) -> dir

The step of `[image_regions.<name>]`, produced or loaded: RadiometricCamera's `measure_image_region` on its frames
([`image_region_frames`](@ref), the sha256 checked) on the device (`default_device()`: GPU first), its record written by
RadiometricCamera's `save_image_region` (IO). `detect` is Features' detector (a test hands a stub).
"""
function run_image_region(cfg::AbstractDict, name::AbstractString; driver::AbstractString, config_path::AbstractString,
                          detect = VgtkFeatures.detect_image_circle)
    p = image_region_product(cfg, name)
    produce(p; product_records(driver, config_path)...) do dir
        t = cfg["image_regions"][name]
        files = image_region_frames(cfg, name)
        m = RadiometricCamera.measure_image_region(files; image = String(t["image"]), outlier_halfwidth_px = Float64(t["outlier_halfwidth_px"]), seed = Int(t["seed"]),
                                                    device = VgtkCore.default_device(), detect)
        RadiometricCamera.save_image_region(joinpath(dir, IMAGE_REGION_FILE), m;
                                            provenance = (; source = "VisualGeometryToolkitFeatures detect_image_circle on $(length(files)) frames of $(dirname(first(files)))",
                                                          config = relpath(config_path, projectdir()), region = name))
        e = m.lens_coverage
        @printf("image region %s: %d frames, %.0f s — centre (%.2f, %.2f) ± (%.2f, %.2f), semi-axes %.2f / %.2f ± %.2f / %.2f px at %.2f°; %d of %d rim candidates inliers, largest rim gap %.0f°%s\n",
                name, length(files), m.seconds, e.center..., m.center_scale..., e.a, e.b, m.axes_scale..., rad2deg(e.θ), m.inliers, m.candidates,
                rad2deg(m.rim_gap), m.cut ? "; cut by the frame" : "")
        Produced(; inputs = files)
    end
end

"""
    lens_coverage_record(cfg, v; what, detect = VgtkFeatures.detect_image_circle, has = nothing) -> String

What a run reads for the `lens_coverage` statement `v` (RadiometricCamera's `camera.lens_coverage`): `"sensor"`, or the path of the
measured image region's record, produced on demand (`produce_upstream`: its own step, [`run_image_region`](@ref), the producer of
scripts/image_regions.jl, under its own key; a GPU stage, so a CPU job is refused naming the command). `what` names the consumer;
`detect` is Features' detector and `has` overrides the device list (a test's).
"""
function lens_coverage_record(cfg::AbstractDict, v::AbstractString; what::AbstractString, detect = VgtkFeatures.detect_image_circle, has = nothing)
    n = lens_coverage_name(v)
    n === nothing && return String(v)
    dir = produce_upstream(image_region_product(cfg, n), "$what: the image region $n";
                           make = () -> run_image_region(cfg, n; driver = IMAGE_REGIONS_DRIVER, config_path = projectdir(IMAGE_REGIONS_CONFIG), detect),
                           command = driver_run(IMAGE_REGIONS_DRIVER, IMAGE_REGIONS_CONFIG, "detect").command, gpu = true, has)   # its stage: a GPU's
    return joinpath(dir, IMAGE_REGION_FILE)
end

"""The path a `lens_coverage` statement resolves to when a run is keyed (no record read): `"sensor"`, or where the image region's
record is (its step's directory; checked by [`lens_coverage_record`](@ref) when the run runs)."""
lens_coverage_path(cfg::AbstractDict, v::AbstractString) =
    (n = lens_coverage_name(v); n === nothing ? String(v) : joinpath(product_dir(image_region_product(cfg, n)), IMAGE_REGION_FILE))

"""
    lens_coverage_region(cfg, v, W, H; what) -> Union{Rect2, Ellipse, Circle}

The lens coverage the statement `v` gives `W × H` photographs: the sensor's rectangle for `"sensor"`, else the measured image region
of its record ([`lens_coverage_record`](@ref); RadiometricCamera's `load_image_region`), refused when the record is of photographs of
another size.
"""
function lens_coverage_region(cfg::AbstractDict, v::AbstractString, W::Integer, H::Integer; what::AbstractString)
    lens_coverage_name(v) === nothing && return VgtkCore.Rect2{Float64}(0, 0, W, H)
    path = lens_coverage_record(cfg, v; what)
    r = RadiometricCamera.load_image_region(path)
    r.extent == (Int(W), Int(H)) || throw(ArgumentError("$what: the image region $path is of $(r.extent[1]) × $(r.extent[2]) photographs, not $W × $H"))
    return r.lens_coverage
end

"""
    require_lens_coverage(cfg, t, what)

The lens coverage the `[pipeline]` of the case or start table `t` states, before bin/refine.jl reads it: nothing to read for
`"sensor"`; a measured image region's record, produced on demand ([`lens_coverage_record`](@ref)).
"""
function require_lens_coverage(cfg::AbstractDict, t::AbstractDict, what::AbstractString)
    v = get(get(get(t, "pipeline", Dict{String, Any}()), "camera", Dict{String, Any}()), "lens_coverage", nothing)
    v === nothing || lens_coverage_record(cfg, v; what)
    return nothing
end

"""
    referenced_image_regions(cfg) -> Vector{String}

The names of the measured image regions `cfg` reads: every `lens_coverage = "image_region:<name>"` anywhere in it (a case's
[pipeline.camera], a front-end case, a chrome evaluation's [evaluate], a panorama config's top level), sorted, unique.
"""
function referenced_image_regions(cfg)
    out = String[]
    walk(x) = nothing
    walk(d::AbstractDict) = for (k, v) in d
        k == "lens_coverage" && v isa AbstractString && (n = lens_coverage_name(v); n === nothing || push!(out, n))
        walk(v)
    end
    walk(v::AbstractVector) = foreach(walk, v)
    walk(cfg)
    return sort(unique(out))
end

"""
    check_image_regions_read(cfg, path)

Every `[image_regions.<name>]` of `cfg` is read by one of its `lens_coverage` statements ([`referenced_image_regions`](@ref)): a
region no statement names is an `ArgumentError` (a step nothing reads is not kept in a config of record).
"""
function check_image_regions_read(cfg::AbstractDict, path::AbstractString)
    dead = setdiff(collect(keys(get(cfg, "image_regions", Dict{String, Any}()))), referenced_image_regions(cfg))
    isempty(dead) || throw(ArgumentError("$path: [image_regions] $(join(sort(dead), ", ")) read by no lens_coverage of the config"))
    return nothing
end

"The tables of configs/experiments/image_regions.toml."
const IMAGE_REGIONS_TABLES = ("experiment", "inputs", "measure")

"""
    load_image_regions_config(path) -> Dict

The config of record of scripts/image_regions.jl: `[experiment] name = "image_regions"`, `[inputs]` (none of its own: each region's
frames are an input of the config that names it) and `[measure] configs`, the configs of record whose image regions it measures
(project-relative paths, each a config of this project with an `[image_regions]` table, each checked: `check_image_regions`).
"""
function load_image_regions_config(path::AbstractString)
    c = load_driver_config(path, "image_regions"; tables = IMAGE_REGIONS_TABLES)
    m = get(c, "measure", nothing)
    m isa AbstractDict || throw(ArgumentError("$path: no [measure] table"))
    VgtkCore.check_keys(m, ("configs",), "$path [measure]")
    cs = get(m, "configs", nothing)
    cs isa AbstractVector && !isempty(cs) && all(x -> x isa AbstractString, cs) && allunique(cs) ||
        throw(ArgumentError("$path [measure] configs: a non-empty list of distinct project-relative config paths"))
    for f in cs
        isfile(projectdir(f)) || throw(ArgumentError("$path [measure] configs: $(repr(f)) is not a config of this project"))
        haskey(TOML.parsefile(projectdir(f)), "image_regions") || throw(ArgumentError("$path [measure] configs: $f has no [image_regions]"))
    end
    return c
end

"""
    image_region_products(cfg) -> Dict{String, NamedTuple}

The image regions the image-regions config `cfg` measures: per key of a region's product (`image_region_product`), the first config of `[measure]
configs` naming it, `(; cfg, name, path)` — the config (its `[inputs]` and `[image_regions]`, each checked), the region's name there
and the config's path. Two configs naming the same frames and settings give one step.
"""
function image_region_products(cfg::AbstractDict)
    out = Dict{String, NamedTuple}()
    for f in cfg["measure"]["configs"]
        path = projectdir(f)
        c = TOML.parsefile(path)
        foreach(((id, t),) -> check_input_table(t, "$path [inputs.$id]"), get(c, "inputs", Dict{String, Any}()))
        check_image_regions(c, path)
        for n in sort(collect(keys(c["image_regions"])))
            k = image_region_product(c, n).key
            haskey(out, k) || (out[k] = (; cfg = c, name = n, path))
        end
    end
    return out
end

"""
    profile_frame(coverage, W, H) -> (; centre, rmax)

The radial profile's frame of a chrome set whose lens coverage is `coverage` on `W × H` photographs (`lens_coverage_region`): the
coverage's centre (the sensor's centre for the sensor's rectangle, the image region's for a measured one) and the farthest point of
its imaged region from it (Core's `farthest_radius` of the sensor ∩ the coverage: the half-diagonal of the sensor, the rim of an
image circle inside it), COLMAP px.
"""
function profile_frame(coverage, W::Integer, H::Integer)
    centre = coverage isa VgtkCore.Rect2 ? StaticArrays.SVector{2, Float64}(coverage.origin .+ coverage.widths ./ 2) : StaticArrays.SVector{2, Float64}(coverage.center...)
    sensor = VgtkCore.Rect2{Float64}(0, 0, W, H)
    rmax = Float64(VgtkCore.farthest_radius(coverage isa VgtkCore.Rect2 ? coverage : sensor ∩ coverage, centre))
    return (; centre, rmax)
end

