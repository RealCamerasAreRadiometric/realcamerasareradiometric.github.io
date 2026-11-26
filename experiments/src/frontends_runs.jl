# =============================================================================
# src/frontends_runs.jl — the products of the front ends (scripts/frontends.jl, configs/experiments/frontends.toml) by their keys:
# each is a keyed product (VisualGeometryToolkitEval's Product, DrWatson's datadir("exp_pro", "frontends", <stage>, <key>)). The
# driver produces them; a config that reads one (a refinement's start, a chrome evaluation's index camera) finds it here through its
# `[runs]` table (src/drivers.jl, `RUN_RESOLVERS["frontends"]`). A product is keyed by its table, its inputs, the keys of the
# products it reads and the code of the packages its stage runs (`FRONTEND_CODE`) — never by the thread count (no stage sums over
# threads), so every reader finds the one product. Included at the
# end of src/drivers.jl.
# =============================================================================

using VisualGeometryToolkitEval.Experiments: run_isolated   # a missing run made by its driver in a module of its own (the resolver's maker)

"""The packages each front-end stage's computation runs (the per-package rule of the code a product is keyed by: the packages its
step uses, never the drivers' own source): the decode Features' raw module on Core and IO; the Schöps run its container — pinned by
its image's sha256 among the run's inputs — whose outputs IO and Core read; the fits Estimation; the comparisons Core and IO."""
const FRONTEND_CODE = Dict("decode" => ("VisualGeometryToolkitCore", "VisualGeometryToolkitIO", "VisualGeometryToolkitFeatures"),
                           "schoeps" => ("VisualGeometryToolkitCore", "VisualGeometryToolkitIO"),
                           "fit" => ("VisualGeometryToolkitCore", "VisualGeometryToolkitIO", "VisualGeometryToolkitEstimation"),
                           "compare" => ("VisualGeometryToolkitCore", "VisualGeometryToolkitIO"))
"The code version of the packages `names` alone (`CODE_VERSION` restricted to them)."
code_of(names) = Dict{String, String}(k => v for (k, v) in CODE_VERSION if k in names)
"""A front-end product of `stage` keyed by `record` and the code of the packages the stage runs (`FRONTEND_CODE`): a commit of the
drivers or of RadiometricCamera re-keys none of them, so a `[runs.<id>]` reference stays produced. Its `output` the file it writes last."""
frontend_product(stage::AbstractString, record::AbstractDict; output::AbstractString) =
    Product("frontends", stage, record; output, code = code_of(FRONTEND_CODE[stage]))

"""The decode of `[cases.<set>]`: its table less the set's lens coverage (the decode reads none: the Schöps run's import is keyed by
it) — its `image`, the geometry image it decodes (the config's statement, no default) — its RAW input, its sensor characterisation
input (each frame's model) and the frame of that image (`decoded_frame`)."""
function frontend_decode_product(cfg::AbstractDict, set::AbstractString)
    t = cfg["cases"][set]
    frontend_product("decode", Dict{String, Any}("experiment" => cfg["experiment"],
                                                 "cases" => Dict(set => filter(((k, _),) -> k != "lens_coverage", t)),
                                                 "input" => input_record(cfg["inputs"][t["raw"]]),
                                                 "sensor_characterization" => input_record(cfg["inputs"][t["sensor_characterization"]]),
                                                 "frame_map" => decoded_frame(t["image"]).record);
                     output = joinpath("$(set)_tool", "tool_images.json"))
end

"The tool images of a decode run: `<run>/<set>_tool`, an 8-bit sRGB PNG per photograph."
frontend_tool_images(cfg::AbstractDict, set::AbstractString) = joinpath(product_dir(frontend_decode_product(cfg, set)), "$(set)_tool")

"""The arguments of `[schoeps.<run>]`: its `args`, the tool's command line as one string (so the TOML quotes it as the frozen drivers
gave it, `--model` and its value together), split as a shell splits it; a run is keyed by this list."""
schoeps_args(t::AbstractDict) = String.(Base.shell_split(t["args"]))
"""The Schöps run of `[schoeps.<name>]`: its table, its inputs (the pattern, the board, the container image), its decode's key and
its set's lens coverage (`lens_coverage_key`: \"sensor\", or the key of the set's image-region step), which bounds the import."""
function frontend_schoeps_product(cfg::AbstractDict, name::AbstractString)
    t = merge(cfg["schoeps"][name], Dict{String, Any}("args" => schoeps_args(cfg["schoeps"][name])))   # keyed by the argument list
    return frontend_product("schoeps", Dict{String, Any}("experiment" => cfg["experiment"], "schoeps" => Dict(name => t),
                                                         "inputs" => inputs_record(cfg, [t["pattern"], t["board"], t["image"]]),
                                                         "upstream" => Dict("decode" => frontend_decode_product(cfg, t["images"]).key),
                                                         "lens_coverage" => lens_coverage_key(cfg, cfg["cases"][t["images"]]["lens_coverage"]));
                            output = "detections.json")
end

"The fit of `[fits.<name>]`: its table, its board input and its Schöps run's key."
function frontend_fit_product(cfg::AbstractDict, name::AbstractString)
    t = cfg["fits"][name]
    return frontend_product("fit", Dict{String, Any}("experiment" => cfg["experiment"], "fits" => Dict(name => t),
                                                     "inputs" => inputs_record(cfg, [t["board"]]),
                                                     "upstream" => Dict("schoeps" => frontend_schoeps_product(cfg, t["detections"]).key));
                            output = "fit.toml")
end

"The comparison of `[compare.<name>]`: its table, its record input and the key of the run it compares."
function frontend_compare_product(cfg::AbstractDict, name::AbstractString)
    t = cfg["compare"][name]
    up = t["kind"] == "schoeps" ? frontend_schoeps_product(cfg, t["run"]).key : frontend_fit_product(cfg, t["run"]).key
    return frontend_product("compare", Dict{String, Any}("experiment" => cfg["experiment"], "compare" => Dict(name => t),
                                                         "inputs" => inputs_record(cfg, [t["record"]]), "upstream" => Dict(t["kind"] => up));
                            output = "compare.toml")
end

# A run of the front ends' config by "<table>.<name>" (a `[runs]` entry of another config): the runs other configs read, each with
# its maker (owner 2026-11-15: a consumer produces its missing upstream): scripts/frontends.jl in a module of its own (Eval's
# run_isolated, `driver_run`) on the run's set, through the stage that makes it — the decode, Schöps's run (a GPU: `gpu`) and the fits —
# each stage producing or loading its own upstream product by its key.
RUN_RESOLVERS["frontends"] = function (cfg::AbstractDict, run::AbstractString, config_path::AbstractString)
    table, name = split(run, '.'; limit = 2)
    haskey(get(cfg, table, Dict()), name) || throw(ArgumentError("frontends: no [$table.$name]"))
    set, stages, product = table == "cases" ? (name, "prepare", frontend_decode_product(cfg, name)) :
                           table == "schoeps" ? (cfg["schoeps"][name]["images"], "prepare,detect", frontend_schoeps_product(cfg, name)) :
                           table == "fits" ? (cfg["schoeps"][cfg["fits"][name]["detections"]]["images"], "prepare,detect,init", frontend_fit_product(cfg, name)) :
                           throw(ArgumentError("frontends: a run is cases.<set>, schoeps.<run> or fits.<name>, got $run"))
    d = driver_run("scripts/frontends.jl", config_path, stages; case = set)
    gpu = occursin("detect", stages) && !produced(table == "schoeps" ? product : frontend_schoeps_product(cfg, cfg["fits"][name]["detections"]))
    return (; product, d.make, d.command, gpu)   # a missing Schöps run is a GPU's (camera_calibration, the detect stage)
end
