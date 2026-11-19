#!/usr/bin/env julia
# reproduce: reads configs/experiments/frontends.toml
# provenance: every product writes _config.toml and _provenance.toml (VisualGeometryToolkitEval produce) beside its results.
# =============================================================================
# scripts/frontends.jl — the geometric front ends of the Siemens-star sets (src/frontends.jl; config of record
# configs/experiments/frontends.toml): the start cameras and index cameras the refinements and the chrome evaluations read, made from
# the published photographs. Every run is a result, keyed: <results>/frontends/<stage>/<key> (src/frontends_runs.jl); a config
# that reads one names it in its [runs] table.
#
# STAGES
#   prepare   per [cases.<set>]: Features' raw module (the case's image: raw_green) and the tool's 8-bit sRGB images (CPU)
#   detect    per [schoeps.<run>]: camera_calibration in its container, then its import onto the raw grid      (one GPU: the tool's)
#   init      per [fits.<name>]: Estimation's fit of an index camera to the imported detections                 (CPU)
#   evaluate  per [compare.<name>]: a run against the start of record (Evaluation's direction_scores)          (CPU)
#
#   julia --project=experiments experiments/scripts/frontends.jl [--config=<toml>] [--plan] [--stage=NAME[,NAME]] [--case=a,b] [--force]
#
# --case restricts every stage to the named [cases.<set>] and the runs and fits made from them (one array task per case: the
# runner's array over [cases], scripts/slurm/frontends.sbatch). The container image is an input pinned by its sha256 (tier
# unpublished); the tool needs the job's GPU.
# =============================================================================
using DrWatson
@quickactivate "experiments"
include(srcdir("experiments.jl"))
include(srcdir("drivers.jl"))
include(srcdir("boards.jl"))
include(srcdir("parametric_starts.jl"))
include(srcdir("evaluation.jl"))
include(srcdir("frontends.jl"))
using .Boards

const DRIVER = "scripts/frontends.jl"
const CFG_PATH = abspath(cli_option(ARGS, "config", projectdir("configs", "experiments", "frontends.toml")))   # default: the config of record
const CFG = load_frontends_config(CFG_PATH)
const SETS = case_names(CFG)
const RUNS = sort([n for (n, t) in CFG["schoeps"] if t["images"] in SETS])
const FITS = sort([n for (n, t) in get(CFG, "fits", Dict{String, Any}()) if CFG["schoeps"][t["detections"]]["images"] in SETS])
"The decode set a comparison's run reads."
compared_set(t) = t["kind"] == "schoeps" ? CFG["schoeps"][t["run"]]["images"] : CFG["schoeps"][CFG["fits"][t["run"]]["detections"]]["images"]
const COMPARES = sort([n for (n, t) in get(CFG, "compare", Dict{String, Any}()) if compared_set(t) in SETS])

const R = (; driver = DRIVER, config_path = CFG_PATH)
const STAGES = [
    Stage(:prepare, "the photographs decoded (Features' raw module, the case's geometry image) and the tool's images";
          items = () -> SETS, product = s -> frontend_decode_product(CFG, s), run = s -> run_frontend_decode(CFG, s; R...)),
    Stage(:detect, "Schöps's camera_calibration in its container, imported onto the raw grid (GPU)";
          items = () -> RUNS, product = r -> frontend_schoeps_product(CFG, r), run = r -> run_frontend_schoeps(CFG, r; R...)),
    Stage(:init, "the index cameras: Estimation's fit to the imported detections";
          items = () -> FITS, product = f -> frontend_fit_product(CFG, f), run = f -> run_frontend_fit(CFG, f; R...)),
    Stage(:evaluate, "the runs compared with the starts of record (directions; a fit's parameters and rms)";
          items = () -> COMPARES, product = c -> frontend_compare_product(CFG, c), run = c -> run_frontend_compare(CFG, c; R...)),
]

run_stages("frontends", STAGES)
