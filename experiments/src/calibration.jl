# =============================================================================
# src/calibration.jl — the calibration cases' configs of record
# (configs/calibration/<case>.toml), shared by scripts/calibrate.jl and the
# suite (test/calibration_tests.jl). Included after src/experiments.jl.
#
# A config of record has three tables, each key checked, and an optional fourth:
#   [experiment]  case (the file's name) and dataset (its input tier under data_root())
#   [pipeline]    the full table of the run's RadiometricCamera options (its RunOptions and CalibrationSettings);
#                 its input paths are relative to data_root() or `run:<id>` (a
#                 [runs.<id>]: a derived file, produced on demand), and io.out is the
#                 driver's (the run's keyed directory), never the config's
#   [render]      how the run's summary is printed (not part of the run key)
#   [runs.<id>]   the derived files [pipeline] names as `run:<id>` (src/drivers.jl
#                 check_runs_table: e.g. the board's distance map, scripts/distance_maps.jl)
# =============================================================================
using RadiometricCamera: RadiometricCamera, pipeline_options
import VisualGeometryToolkitCore as VgtkCore

"The keys of the tables of a calibration config other than `[pipeline]` (which `pipeline_options` checks)."
const CALIBRATION_KEYS = Dict(
    "experiment" => ("case", "dataset"),
    "render" => ("title", "significant_digits"),
)

"""
The `[pipeline]` keys that name an input file or directory, as `table.key` (`io.init` is a model spec, `lens:<path>` or
`grid:<path>`): the package's declaration, `RadiometricCamera.PIPELINE_PATH_KEYS` (its suite checks that every string option is an
input path, an output path or declared no path).
"""
const PIPELINE_INPUT_KEYS = RadiometricCamera.PIPELINE_PATH_KEYS

"The `[pipeline]` keys that name an output (`RadiometricCamera.PIPELINE_OUTPUT_KEYS`; current keys, a table names them as its version does, `config_key`): `io.out` is set by the driver; `io.save_init` (a diagnostic dump) and the optimizer's log are refused."
const PIPELINE_OUTPUT_KEYS = RadiometricCamera.PIPELINE_OUTPUT_KEYS

# The value of `table.key` in a [pipeline] table, "" when absent.
function _pipeline_value(table::AbstractDict, key::AbstractString)
    t, k = split(key, '.')
    return get(get(table, t, Dict{String,Any}()), k, "")
end

"""
    load_calibration_config(path) -> (cfg, run, settings)

Read a calibration config of record: the parsed TOML `cfg`, the run's `RunOptions` `run` and its `CalibrationSettings` `settings`. Every table and key is checked
(an unknown one is an `ArgumentError` naming it and the allowed ones), `[experiment] case` must be the file's name, and a
`[pipeline]` output key (`PIPELINE_OUTPUT_KEYS`) is refused: the run's output directory is the driver's.
"""
function load_calibration_config(path::AbstractString)
    isfile(path) || throw(ArgumentError("calibration config not found: $path"))
    cfg = TOML.parsefile(path)
    VgtkCore.check_keys(cfg, ("experiment", "pipeline", "render", "runs"), path)
    for t in ("experiment", "pipeline", "render")
        haskey(cfg, t) && cfg[t] isa AbstractDict || throw(ArgumentError("$path: no [$t] table"))
    end
    for (t, allowed) in CALIBRATION_KEYS
        VgtkCore.check_keys(cfg[t], allowed, "$path [$t]")
        for k in allowed
            haskey(cfg[t], k) || throw(ArgumentError("$path: [$t] lacks `$k`"))
        end
    end
    check_runs_table(cfg, path)                                  # [runs.<id>]: the derived files [pipeline] names as `run:<id>`
    dead = setdiff(collect(keys(get(cfg, "runs", Dict{String, Any}()))), table_runs(cfg["pipeline"]))
    isempty(dead) || throw(ArgumentError("$path: [runs] $(join(sort(dead), ", ")) named by no `run:<id>` of [pipeline]"))
    case = cfg["experiment"]["case"]
    case == splitext(basename(path))[1] || throw(ArgumentError("$path: [experiment] case = \"$case\" is not the file's name"))
    # the case's lens coverage, stated (src/image_region.jl): a calibration case has no image-region step, its lens lights the sensor
    _pipeline_value(cfg["pipeline"], "camera.lens_coverage") == RadiometricCamera.SENSOR_LENS_COVERAGE ||
        throw(ArgumentError("$path [pipeline.camera] lens_coverage: a calibration case states \"sensor\" (its lens lights the whole sensor; " *
                            "a measured image region is a step of the paper's configs, src/image_region.jl)"))
    for k in (RadiometricCamera.config_key(cfg["pipeline"], k) for k in PIPELINE_OUTPUT_KEYS)
        isempty(_pipeline_value(cfg["pipeline"], k)) ||
            throw(ArgumentError("$path: [pipeline] $k is an output; the driver writes the run into its keyed directory"))
    end
    run, settings = try
        pipeline_options(cfg["pipeline"])
    catch err
        err isa ArgumentError ? throw(ArgumentError("$path [pipeline]: $(err.msg)")) : rethrow()
    end
    return cfg, run, settings
end

"""
    pipeline_inputs(cfg) -> Vector{Pair{String,String}}

The inputs a calibration config's `[pipeline]` names, `table.key => path` relative to `data_root()`, in `PIPELINE_INPUT_KEYS`
order (an empty key is not an input: RadiometricCamera's `names_input_path`; `io.init`'s `lens:`/`grid:` prefix is dropped; a
`run:<id>` is a `[runs]` file under results_root(), not an input under data_root(): [`config_source`](@ref) reads it). A `selfcheck.only` run reads no
photograph, so its `io.images` (required non-empty by bin/refine.jl, never opened) is not an input either.
"""
function pipeline_inputs(cfg::AbstractDict)
    out = Pair{String,String}[]
    selfcheck_only = _pipeline_value(cfg["pipeline"], "selfcheck.only") == true
    for k in PIPELINE_INPUT_KEYS
        v = String(_pipeline_value(cfg["pipeline"], k))
        (!RadiometricCamera.names_input_path(k, v) || (selfcheck_only && k == "io.images")) && continue   # "none" names no path
        k == "io.init" && (v = replace(v, r"^(lens|grid):" => ""))
        match(RUN_REF, v) === nothing || continue                                                          # a [runs] file
        push!(out, k => v)
    end
    return out
end

"""
    calibration_products(cfg; threads = execution_threads()) -> (; selfcheck, output, start, refine, evaluate, figures)

The products of a calibration case `cfg` (scripts/calibrate.jl), each keyed by only what it reads, under the driver's name: its
start (stage start: `start_settings` of its [pipeline], the start record's schema), its refinement (`refinement_settings` and the
start's key; a self-check, `selfcheck.only`, is one product keyed by its whole [pipeline], no start), the metrics (evaluate) and the
summary table (figures, keyed by `[render]`); the thread count `threads` keys the start and the refinement (bin/refine.jl).
"""
function calibration_products(cfg::AbstractDict; threads::AbstractString = execution_threads())
    selfcheck = _pipeline_value(cfg["pipeline"], "selfcheck.only") == true
    output = selfcheck ? "selfcheck.json" : "parameters.json"        # the pipeline's last file of the run
    e = cfg["experiment"]
    ids = table_runs(cfg["pipeline"])                                # the [runs] files [pipeline] names: keyed by their runs' keys
    runs = isempty(ids) ? Dict{String, Any}() : Dict{String, Any}("runs" => runs_record(cfg, ids))
    start = driver_product("start", Dict{String, Any}("experiment" => e, "start" => start_settings(cfg),
                                                     "start_record" => RadiometricCamera.RECORD_SCHEMAS.refinement_start, runs...);
                           experiment = "calibrate", output = RadiometricCamera.START_RECORD_FILE, threads)
    refine = selfcheck ? driver_product("refine", Dict{String, Any}("experiment" => e, "pipeline" => cfg["pipeline"], runs...); experiment = "calibrate", output, threads) :
             driver_product("refine", Dict{String, Any}("experiment" => e, "refine" => refinement_settings(cfg), "upstream" => Dict("start" => start.key));
                            experiment = "calibrate", output, threads)
    evaluate = driver_product("evaluate", Dict{String, Any}("experiment" => e, "upstream" => Dict("refine" => refine.key));
                              experiment = "calibrate", output = "metrics.toml", threads = nothing)
    figures = driver_product("figures", Dict{String, Any}("experiment" => e, "summary" => cfg["render"], "upstream" => Dict("evaluate" => evaluate.key));
                             experiment = "calibrate", output = "summary.tex", threads = nothing)
    return (; selfcheck, output, start, refine, evaluate, figures)
end
