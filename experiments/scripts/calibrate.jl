#!/usr/bin/env julia
# provenance: every product writes _config.toml and _provenance.toml (VisualGeometryToolkitEval produce) beside its results.
# =============================================================================
# scripts/calibrate.jl — the calibration pipeline on one case: a config of
# record under configs/calibration/ in, a run directory keyed by that config's
# run key out. The first staged driver of the project, and the skeleton every
# later driver copies: config → stages → keyed run directory.
#
# The pipeline is RadiometricCamera's own entry point, bin/refine.jl, run in a
# child process (src/drivers.jl `run_pipeline`: the package's environment,
# this process's Julia and thread count — the closed-form start sums over
# thread chunks, so the thread count is part of the arithmetic and keys the
# run) with data_root() as its working directory, which is what the config's
# relative input paths are relative to. The run directory therefore holds
# exactly what `bin/refine.jl --config` writes, plus the run's records and the
# child's log. The package's golden gate (experiments/test/pipeline/run_pipeline_checks.sh)
# runs `--stage=refine --force` of each case and compares that directory.
#
# STAGES (each product keyed by only what it reads: <results>/calibrate/<stage>/<key>/, DrWatson's datadir("exp_pro", …))
#   prepare   the inputs [pipeline] names, each resolved under data_root() (require_input), and its [runs] files (produced on demand)
#   start     bin/refine.jl --stage=start on the [pipeline] table → start/<key>/, the start record (RadiometricCamera's
#             start.json, state.json, grid.json, masks/): keyed by every setting but the refinement-only ones
#             (RadiometricCamera.REFINE_ONLY_KEYS) and the [execution] threads, so a refine-only change loads it
#   refine    bin/refine.jl --start=<the start> → refine/<key>/ (keyed by the refinement-only settings and the start's key), with
#             _config.toml, _provenance.toml, pipeline_input.toml (the table as bin/refine.jl read it, io.out set) and refine.log
#             (the child's output); a self-check (selfcheck.only) runs here alone, keyed by its whole [pipeline]
#   evaluate  the pipeline's own metrics (RadiometricCamera.load_pipeline_metrics) → evaluate/<key>/metrics.toml
#   figures   no figure yet: the run's summary table → figures/<key>/summary.tex
#
#   julia --project=experiments experiments/scripts/calibrate.jl --config=<toml> [--plan] [--stage=NAME] [--force]
#
# Data: the cases' inputs are the tier rc_calibration of configs/data_artifacts.toml
# (scripts/fetch_data.jl --only=<it>, or VGTK_DATA_ROOT bound to a data root that
# holds it: REPRODUCE.md, data/README.md).
# =============================================================================
using DrWatson
@quickactivate "experiments"
include(srcdir("experiments.jl"))
include(srcdir("drivers.jl"))
include(srcdir("calibration.jl"))

const CFG_PATH = abspath(cli_option(ARGS, "config"))
const CFG, _, _ = load_calibration_config(CFG_PATH)
# the products, each keyed by only what it reads, under the driver's name (the config names a case): src/calibration.jl
const PRODUCTS = calibration_products(CFG)
const SELFCHECK, OUTPUT = PRODUCTS.selfcheck, PRODUCTS.output
const START, REFINE, EVALUATE, FIGURES = PRODUCTS.start, PRODUCTS.refine, PRODUCTS.evaluate, PRODUCTS.figures
const RUN_DIR = product_dir(REFINE)
const KEY = REFINE.key
const METRICS = product_output(EVALUATE)
const CASE = CFG["experiment"]["case"]
const R = product_records("scripts/calibrate.jl", CFG_PATH)

# The declared inputs, absolute under data_root() (resolved at use: --plan needs no data): the case's tier, then every input
# [pipeline] names; and its [runs] files (`run:<id>`: a derived file under results_root(), produced on demand, config_source).
const INPUT_PROVENANCE = "the case data of RadiometricCamera's pipeline checks: the tier rc_calibration of configs/data_artifacts.toml " *
                         "(scripts/fetch_data.jl, or VGTK_DATA_ROOT bound to a data root that holds it)"
input_paths() = ["[experiment] dataset" => joinpath(data_root(), CFG["experiment"]["dataset"]);
                 ["[pipeline] $k" => joinpath(data_root(), p) for (k, p) in pipeline_inputs(CFG)]]

prepare() = String[[require_data(path; what = "$k of case $CASE", provenance = INPUT_PROVENANCE) for (k, path) in input_paths()];
             [config_source(CFG, id) for id in table_runs(CFG["pipeline"])]]
"The [pipeline] table `table` as bin/refine.jl reads it: each `run:<id>` the [runs] file's absolute path, produced on demand (config_source)."
resolved(table) = resolve_pipeline(CFG, Dict{String, Any}("pipeline" => table))

"Whether this run's refine stage produces the refinement (it runs, and the refinement is not produced or --force): then the start is its child's too."
refine_runs() = stage_runs("refine") && ("--force" in ARGS || !produced(REFINE))
start(p) = refine_runs() || produce(p; R...) do dir
    run_pipeline(resolved(start_settings(CFG)["pipeline"]), dir; source = relpath(CFG_PATH, projectdir()), label = "case $CASE (start $(p.key))", args = ["--stage=start"])
    Produced(; inputs = prepare())
end

# the refinement from its produced start (its own child), or — the start not produced — start and refinement in ONE bin/refine.jl
# child, the start written into its own product (`--start-out`)
refine(p) = produce(p; R...) do dir
    inputs = prepare()                                       # the case's inputs, checked once
    run = a -> run_pipeline(resolved(CFG["pipeline"]), dir; source = relpath(CFG_PATH, projectdir()), label = "case $CASE (run $KEY)", args = a)
    if SELFCHECK
        run(String[])
    elseif start_reused(START)
        run(["--start=$(product_dir(START))"])
    else                                                     # --force with the start stage: produced again with the refinement
        produce(START; R..., force = true) do sd
            run(["--start-out=$sd"])
            Produced(; inputs)
        end
    end
    Produced(; inputs)
end

evaluate(p) = produce(p; R..., inputs = [RUN_DIR]) do dir
    metrics = RadiometricCamera.load_pipeline_metrics(RUN_DIR)
    open(joinpath(dir, "metrics.toml"), "w") do io
        println(io, "# The pipeline's own metrics of run $KEY (case $CASE), read from its $OUTPUT by scripts/calibrate.jl.")
        TOML.print(io, metrics; sorted = true)
    end
    nothing
end

# The metrics as (dotted name, value) rows, in key order.
function metric_rows(d::AbstractDict, prefix = "")
    rows = Pair{String,Any}[]
    for k in sort(collect(keys(d)))
        v = d[k]; name = isempty(prefix) ? k : prefix * "." * k
        v isa AbstractDict ? append!(rows, metric_rows(v, name)) : push!(rows, name => v)
    end
    return rows
end
latex_value(x::AbstractFloat, digits) = string(round(x; sigdigits = digits))
latex_value(x::AbstractVector, digits) = join((latex_value(v, digits) for v in x), ", ")
latex_value(x, digits) = latex_escape(string(x))
# The LaTeX special characters of a text escaped (a title or a name with _, %, & or # breaks the build of what includes it).
latex_escape(s::AbstractString) = replace(s, r"([_%&#$])" => s"\\\1")

figures(p) = produce(p; R..., inputs = [METRICS]) do dir
    digits = CFG["render"]["significant_digits"]
    open(joinpath(dir, "summary.tex"), "w") do io
        println(io, "% generated by scripts/calibrate.jl from $(relpath(CFG_PATH, projectdir())) (run $KEY); do not edit")
        println(io, "\\begin{table}\\centering")
        println(io, "\\caption{", latex_escape(CFG["render"]["title"]), " (case \\texttt{", replace(CASE, "_" => "\\_"), "}, run \\texttt{", KEY, "}).}")
        println(io, "\\begin{tabular}{ll}\\toprule")
        println(io, "metric & value \\\\ \\midrule")
        for (name, v) in metric_rows(TOML.parsefile(METRICS))
            println(io, "\\texttt{", replace(name, "_" => "\\_"), "} & ", latex_value(v, digits), " \\\\")
        end
        println(io, "\\bottomrule\\end{tabular}")
        println(io, "\\end{table}")
    end
    nothing
end

# The stages that run the GPU: start and refine hand the [pipeline] to bin/refine.jl (its compute.backend).
const GPU_STAGES = SELFCHECK ? gpu_stages(:refine => [CFG]) : gpu_stages(:start => [CFG], :refine => [CFG])

const STAGES = [
    Stage(:prepare, "the inputs of [pipeline], resolved under data_root()"; check = prepare),
    Stage(:start, "bin/refine.jl --stage=start: the case's start record"; items = () -> SELFCHECK ? Product[] : [START], product = identity, run = start),
    Stage(:refine, "bin/refine.jl --start=<the start> into the refinement's product"; items = () -> [REFINE], product = identity, run = refine),
    Stage(:evaluate, "the pipeline's own metrics → metrics.toml"; items = () -> [EVALUATE], product = identity, run = evaluate),
    Stage(:figures, "no figure yet: the summary table → summary.tex"; items = () -> [FIGURES], product = identity, run = figures),
]

run_stages("calibrate", STAGES)
