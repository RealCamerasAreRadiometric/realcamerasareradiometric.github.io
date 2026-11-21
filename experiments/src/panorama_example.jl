# =============================================================================
# src/panorama_example.jl — PanoramaStitching's example of record,
# examples/panorama_vignetting/run.jl, as the drivers bind it: its config from a
# driver's `[example]` table and `[inputs]` (scripts/panorama_vignetting.jl's
# configs of record), the driver's keyed run that holds that config, the
# example's own keyed products that hold its outputs (under Eval's results root,
# keyed by the example's config), and the example's functions (included once,
# in a module of their own; called, never copied).
#
# Two drivers read it, after src/experiments.jl and src/drivers.jl:
#   scripts/panorama_vignetting.jl   runs the example's chain (stage evaluate) on the config its keyed run holds
#   scripts/panorama_stitching.jl    reads that run of its [upstream] panorama_vignetting config (the sweep's decoded frames,
#                                    rotations and lens) through the example's stage functions
# =============================================================================
import PanoramaStitching
include(joinpath(@__DIR__, "return_codes.jl"))                    # recorded_retcode: the ReturnCode a record names

"The example of record: PanoramaStitching's examples/panorama_vignetting/run.jl (its `run_example` and stage functions)."
const PANORAMA_EXAMPLE = joinpath(pkgdir(PanoramaStitching), "examples", "panorama_vignetting", "run.jl")
"The example config's tables (a driver's [example]); its inputs' paths are the driver's."
const PANORAMA_EXAMPLE_TABLES = ("name", "inputs", "compute", "decode", "registration", "domain", "vignetting", "seams", "render")
"""The keys of the example's config the driver supplies from `[inputs]`, as `(table, key) => input`: the sweep's frames, its camera
of record and the frames' sensor record (the example reads it for the decode and, with `detection_image = "tone"`, for the
detector: one record, stated once)."""
const PANORAMA_EXAMPLE_INPUTS = (("inputs", "raw_dir") => "raw", ("inputs", "lens") => "lens",
                                 ("decode", "sensor_characterisation") => "sensor_characterisation",
                                 ("registration", "sensor_characterisation") => "sensor_characterisation")

"""
    load_panorama_vignetting_config(path) -> Dict

A config of record of scripts/panorama_vignetting.jl: `[experiment]` (name "panorama_vignetting"), `[inputs]` `raw`, `lens` (the
camera store of record, its camera bounded by the sensor) and `sensor_characterisation` (the frames' sensor record), `lens_coverage`
(the sweep's lens coverage, src/image_region.jl: "sensor" or "image_region:<name>" of its `[image_regions]`, which the driver puts on
the camera: `write_panorama_example_config`), `[example]`, the example's config less what the driver supplies
(`PANORAMA_EXAMPLE_INPUTS`: a config that states one of those keys is refused), and `[expected]`, the registration's rotation-bundle
termination every config states (no default): `bundle`, the name of the VisualGeometryToolkitCore `ReturnCode` the bundle records
(its registration's `bundle_return_code`; `"MaxIters"` when `[example.registration] bundle_max_iter = 0`, PanoramaStitching's record
of a bundle that runs no solve), and its `reason`. A retired key of `[example.registration]` is refused by name — with its successor
— at load, by PanoramaStitching's one public list (`PanoramaStitching.refuse_retired_registration`; a config without that table: an
`ArgumentError`), so a config with a retired key neither plans nor keys a run; the example itself is not loaded (the plans and the
run keys need none). `[expected]` does not key the run: it is a statement checked against it ([`check_bundle_retcode`](@ref)).
"""
function load_panorama_vignetting_config(path::AbstractString)
    c = load_driver_config(path, "panorama_vignetting"; tables = ("experiment", "lens_coverage", "inputs", "image_regions", "example", "expected"))
    VgtkCore.check_keys(c["inputs"], ("raw", "lens", "sensor_characterisation", (k for (_, t) in get(c, "image_regions", Dict()) for k in (t["input"],))...), "$path [inputs]")
    foreach(k -> VgtkCore.required_key(c["inputs"], k, "$path [inputs]"), ("raw", "lens", "sensor_characterisation"))
    haskey(c, "lens_coverage") || throw(ArgumentError("$path: no lens_coverage — the sweep's coverage is stated: \"sensor\" or \"image_region:<name>\" (src/image_region.jl)"))
    check_lens_coverage(c, c["lens_coverage"], path)
    VgtkCore.check_keys(c["example"], PANORAMA_EXAMPLE_TABLES, "$path [example]")
    get(c["example"], "registration", nothing) isa AbstractDict ||
        throw(ArgumentError("$path: no [example.registration] table (the example's registration settings)"))
    PanoramaStitching.refuse_retired_registration(c["example"]["registration"], "$path [example.registration]")
    for ((table, k), id) in PANORAMA_EXAMPLE_INPUTS
        haskey(get(c["example"], table, Dict{String, Any}()), k) && throw(ArgumentError("$path [example.$table]: `$k` is the driver's ([inputs.$id])"))
    end
    haskey(c, "expected") || throw(ArgumentError("$path: no [expected] table (bundle = \"<a Core ReturnCode name>\", reason = \"…\")"))
    VgtkCore.check_keys(c["expected"], ("bundle", "reason"), "$path [expected]")
    for k in ("bundle", "reason")
        haskey(c["expected"], k) && c["expected"][k] isa AbstractString && !isempty(c["expected"][k]) ||
            throw(ArgumentError("$path [expected] lacks `$k` (a non-empty string)"))
    end
    expected = expected_bundle_retcode(c["expected"]["bundle"]; where = "$path [expected]")
    c["example"]["registration"]["bundle_max_iter"] == 0 && expected != VgtkCore.ReturnCode.MaxIters &&
        throw(ArgumentError("$path [expected] bundle = \"$(c["expected"]["bundle"])\" but [example.registration] bundle_max_iter = 0 " *
                            "(a bundle that runs no solve records \"MaxIters\")"))
    return c
end

"""
    expected_bundle_retcode(name; where) -> ReturnCode.T

The `ReturnCode` a config's `[expected] bundle` names ([`recorded_retcode`](@ref)): an `ArgumentError` naming `where` for a name that
is not one (an outcome Symbol of an earlier release: `"converged"`, `"not_run"`) or that states no record (`"unfinished"`,
`"unrecorded"`).
"""
function expected_bundle_retcode(name::AbstractString; where::AbstractString)
    rc = try
        recorded_retcode(name)
    catch err
        err isa ArgumentError || rethrow()
        throw(ArgumentError("$where bundle = \"$name\": $(err.msg)"))
    end
    rc === nothing && throw(ArgumentError("$where bundle = \"$name\" states no termination record (a Core ReturnCode name: \"Success\", …)"))
    return rc
end

"""
    check_bundle_retcode(retcode, expected; reason, where = "") -> Nothing

A run's rotation-bundle termination (`retcode`: its registration's `bundle_return_code`, the name of PanoramaStitching's
bundle `ReturnCode`) against the config's `[expected] bundle` (a `ReturnCode` name, [`expected_bundle_retcode`](@ref)) and its
`reason`: an error naming both when they differ — a bundle expected to succeed that does not, and an expected failure that succeeds
or ends otherwise, are both errors.
"""
function check_bundle_retcode(retcode::AbstractString, expected::AbstractString; reason::AbstractString, where::AbstractString = "")
    expected_bundle_retcode(expected; where = "$where [expected]") == recorded_retcode(retcode) && return nothing
    error("$where: the rotation bundle's termination is \"$retcode\" but [expected] bundle = \"$expected\" ($reason) — " *
          "the registration is not the one the config records; fix the input or state the new termination and its reason")
end

"""
    check_registration(cfg, reg; where)

The registration `reg` of a run (PanoramaStitching's example `registration(C, inp, lens)`: its result, the keyed record of the
example's product, never a side file) against `cfg`'s `[expected]` ([`check_bundle_retcode`](@ref)): the bundle's termination record
(`bundle_return_code`, Core's `ReturnCode` by name) must be the one the config states, and LoMa must have run on the provider the
config states (`[example.registration] execution_provider`): the record's `loma` provenance, descriptor and matcher, names the
provider actually used (Features' `loma_provenance`). A mismatch — a run on the CPU when the config states "cuda" — is an error.
"""
function check_registration(cfg::AbstractDict, reg; where::AbstractString)
    check_bundle_retcode(String(reg.bundle_return_code), cfg["expected"]["bundle"]; reason = cfg["expected"]["reason"], where)
    want = String(cfg["example"]["registration"]["execution_provider"])
    for part in ("descriptor", "matcher")
        got = String(reg.loma[part]["execution_provider"])
        got == want || error("$where: LoMa's $part ran on \"$got\", the config states execution_provider = \"$want\" " *
                             "(the registration's loma provenance): refusing the run")
    end
    return nothing
end

"""
    example_registration(C) -> (; inp, lens, reg, W, H)

The registration of the example's context `C` (its `load_config`) through the example's own functions: the decoded frames and their
saturation masks (its decode cache), the camera size (the RAW reader's raw grid of the first frame: RadiometricCamera's
`raw_image_size`, Features' `load_raw_mosaic`), the lens and the registration (its keyed product: computed once, then read back by
its settings' key; the example resolves LoMa's provider before it reads the record, so a config that states "cuda" is read in a
process that holds a GPU).
"""
function example_registration(C)
    inp = panorama_example_call(:decoded_frames, C; masks = true)
    W, H = RadiometricCamera.raw_image_size(joinpath(panorama_example_call(:input_path, C.cfg["inputs"]["raw_dir"]), first(inp.names)))
    lens = panorama_example_call(:load_lens, C; raw_width = W, raw_height = H)
    reg = panorama_example_call(:registration, C, inp, lens; log = devnull)
    return (; inp, lens, reg, W, H)
end

"""
    require_frames_decoded_as(inp, camera)

The sweep's frames (`inp`, the PanoramaStitching example's `decoded_frames`) are corrected by the learned V only when the example
decoded them under the learned camera's own sensor record: the sha256 it states for its frames' decode equals the camera's
(`nothing` for a camera calibrated without a record).
"""
function require_frames_decoded_as(inp, camera::RadiometricCamera.CalibratedCamera)
    want = camera.observation.sensor isa RadiometricCamera.SensorRecord ? camera.observation.sensor.sha256 : nothing
    inp.sensor_characterisation_sha256 == want ||
        throw(ArgumentError("the sweep's frames were decoded under the sensor record $(something(inp.sensor_characterisation_sha256, "none")), " *
                            "the learned camera was calibrated under $(something(want, "none")): one record for both"))
    return nothing
end

"""The driver's product of the example's chain on the config `cfg` (`<results_root()>/<experiment>/evaluate/<key>`; it holds the
example's config, example_config.toml, and the camera it reads, `panorama_lens_file`): keyed by `[experiment]`, `[example]`, the
inputs the example reads and the sweep's lens coverage (`lens_coverage_key`: "sensor", or its image region's key)."""
panorama_vignetting_product(cfg::AbstractDict) =
    driver_product("evaluate", Dict{String,Any}("experiment" => cfg["experiment"], "example" => cfg["example"],
                                                "inputs" => inputs_record(cfg, ["raw", "lens", "sensor_characterisation"]),
                                                "lens_coverage" => lens_coverage_key(cfg, cfg["lens_coverage"]));
                   output = "example_config.toml")

"""The camera the example's chain reads, in the driver's run `dir`: the camera of record (`[inputs] lens`) bounded by the sweep's
lens coverage (`write_panorama_example_config`), a vgio/cameras store."""
panorama_lens_file(dir::AbstractString) = joinpath(dir, "lens.json")

"""
    panorama_lens(cfg) -> BoundedCamera

The camera of the sweep of `cfg`: its camera of record (`[inputs] lens`, the store's camera `[example.inputs] lens_camera`, IO) with
the sweep's lens coverage (`lens_coverage_region` of `lens_coverage`: the sensor's rectangle, or the measured image region's record,
which must be produced — no fallback) — the one place a panorama camera gets its coverage.
"""
function panorama_lens(cfg::AbstractDict)
    lens_coverage_record(cfg, cfg["lens_coverage"]; what = "the panorama camera")   # its coverage first: no camera without its record
    cam = VgtkIO.load(VgtkCore.BoundedCamera, config_input(cfg, "lens"); name = cfg["example"]["inputs"]["lens_camera"])
    W, H = VgtkCore.extent(cam)
    return VgtkCore.with_lens_coverage(cam, lens_coverage_region(cfg, cfg["lens_coverage"], W, H; what = "the panorama camera"))
end

"""
    write_panorama_example_config(cfg, dir) -> path

The example's config of the run `dir`: `[example]` of `cfg` with its inputs' paths resolved under data_root() (absolute: the example
joins them to its data root, and an absolute path stays itself), written as `dir/example_config.toml`; the camera it reads is the
sweep's ([`panorama_lens`](@ref): the camera of record with the sweep's lens coverage), written as `panorama_lens_file(dir)` (IO); the
sensor record the frames are decoded under is the input `sensor_characterisation`, stated for the decode and for the detector. A
`raw` input with a selection (`frames` | `exclude`) is handed over as its folder of links in `dir` (`selected_raw_dir`). The example
writes its outputs into its own keyed products, never into `dir`.
"""
function write_panorama_example_config(cfg::AbstractDict, dir::AbstractString)
    ex = deepcopy(cfg["example"])
    mkpath(dir)
    ex["inputs"]["raw_dir"] = abspath(selected_raw_dir(cfg, "raw", dir))
    lens = panorama_lens_file(dir)
    VgtkIO.save(lens, Dict(Symbol(cfg["example"]["inputs"]["lens_camera"]) => panorama_lens(cfg));
                provenance = (; camera = config_input(cfg, "lens"), lens_coverage = lens_coverage_path(cfg, cfg["lens_coverage"])))
    ex["inputs"]["lens"] = abspath(lens)
    record = abspath(config_input(cfg, "sensor_characterisation"))
    ex["decode"]["sensor_characterisation"] = record
    ex["registration"]["sensor_characterisation"] = record
    path = joinpath(dir, "example_config.toml")
    open(io -> TOML.print(io, ex; sorted = true), path, "w")
    return path
end

"The example's module: run.jl included once, in a module of its own."
const PANORAMA_EXAMPLE_MODULE = Ref{Module}()
function panorama_example()
    isassigned(PANORAMA_EXAMPLE_MODULE) && return PANORAMA_EXAMPLE_MODULE[]
    m = Module(:PanoramaVignettingExample)
    Base.include(m, PANORAMA_EXAMPLE)
    return PANORAMA_EXAMPLE_MODULE[] = m
end

"""
    panorama_example_call(f, args...; kw...)

The example's function `f` (a Symbol) called on `args`, with `VGTK_DATA_ROOT` bound to this project's data root (the example
reads its inputs under it; the paths a driver writes are absolute anyway).
"""
panorama_example_call(f::Symbol, args...; kw...) =
    withenv("VGTK_DATA_ROOT" => data_root()) do
        Base.invokelatest(panorama_example_name(f), args...; kw...)
    end
"The example's binding `name` (a function or a constant), read in the latest world (the module is included at run time)."
panorama_example_name(name::Symbol) = Base.invokelatest(getglobal, panorama_example(), name)
