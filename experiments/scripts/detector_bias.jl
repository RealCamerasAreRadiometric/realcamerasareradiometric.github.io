#!/usr/bin/env julia
# reproduce: writes ../paper/fig/detector-bias-mitsuba.tex
# reproduce: reads configs/experiments/detector_bias.toml
# provenance: every product writes _config.toml and _provenance.toml (VisualGeometryToolkitEval produce) beside its results.
# =============================================================================
# scripts/detector_bias.jl — the detector-bias experiment (fig:detector-bias): on renders with a known camera at known poses, the
# detectors' bias, its inheritance by the geometric calibrations and what the radiometric refinement leaves of it. Per case a
# detector-based geometric calibration (its start) refined radiometrically, both evaluated against the truth (src/detector_bias.jl,
# src/evaluation.jl), and the paper's figure from the evaluations (src/paper_figures/detector_bias.jl). Config of record:
# configs/experiments/detector_bias.toml.
#
# STAGES (each product keyed by only what it reads and the keys of the products it reads; <results>/<experiment>/<stage>/<key>/)
#   detect    the renders, their truth, the detectors' outputs, the geometric calibrations and the boards of record, present
#   start     per case, its start (bin/refine.jl --stage=start): keyed by every setting but the refinement-only ones
#   refine    per case, the radiometric refinement from its start (bin/refine.jl --start, a checkpoint per iteration)
#   evaluate  per case, at the start and at the last checkpoint: the reprojection error under the run's own poses
#             (learned_<it>.json) and the camera alone under the true poses (truepose_<it>.json), scores.json
#   figures   the paper's figure (its tables in the product's tables/, its files → plotsdir(<asset_dir>.tex, <asset_dir>/,
#             <asset_dir>-caption.tex)), numbers.tex (empty: the figure cites no number)
#
#   julia --project=experiments experiments/scripts/detector_bias.jl [--config=<toml>] [--plan] [--stage=NAME[,NAME]] [--case=a,b] [--force]
#
# --case restricts start, refine and evaluate to the named cases (a GPU budget); the figure reads the cases its rows name, and a
# product it lacks is produced first (a refinement on a GPU). Data: VGTK_DATA_ROOT binds data_root() (REPRODUCE.md); the renders
# and their truth are the tier rc_synthetic, the detectors' outputs and the starts rc_baselines, the boards rc_calibration.
# =============================================================================
using DrWatson
@quickactivate "experiments"
include(srcdir("experiments.jl"))
include(srcdir("drivers.jl"))
include(srcdir("evaluation.jl"))
include(srcdir("paper_figures", "PaperFigures.jl"))                           # the figure's PGFPlots text (its generator: detector_bias.jl there)
include(srcdir("detector_bias.jl"))

const DRIVER = "scripts/detector_bias.jl"
const CFG_PATH = abspath(cli_option(ARGS, "config", projectdir("configs", "experiments", "detector_bias.toml")))   # default: the config of record
const CFG = load_detector_bias_config(CFG_PATH)
const CASES = case_names(CFG)
const R = (; driver = DRIVER, config_path = CFG_PATH)            # the drivers' runners' keywords (run_start, run_refine, produce_figures)
const P = product_records(DRIVER, CFG_PATH)                     # produce's

detect() = (require_stage_inputs(CFG, "detect"); nothing)

"The directory of the start product of case `c`, produced on demand (its stage start, on a GPU where its [pipeline] runs on one)."
start_dir(c) = produce_upstream(start_product(CFG, c), "the evaluation of case $c: its start"; make = () -> run_start(CFG, c; R...),
                                command = driver_run(DRIVER, CFG_PATH, "start"; case = c).command, gpu = runs_on_gpu(CFG["cases"][c]))
"The directory of the refine product of case `c`, produced on demand (its stage refine, on a GPU where its [pipeline] runs on one)."
refine_dir(c) = produce_upstream(refine_product(CFG, c), "the evaluation of case $c: its refinement"; make = () -> run_refine(CFG, c; R...),
                                 command = driver_run(DRIVER, CFG_PATH, "start,refine"; case = c).command, gpu = runs_on_gpu(CFG["cases"][c]))

"Stage evaluate of case `c` (`evaluate_detector_bias`): from its refine and start products, each produced first when it is missing."
evaluate_case(c) = produce(detector_bias_evaluate_product(CFG, c); P...) do dir
    refine = refine_dir(c); start = start_dir(c)
    evaluate_detector_bias(CFG, c, dir; start, refine, key = detector_bias_evaluate_product(CFG, c).key)
    Produced(; inputs = [config_input(CFG, CFG["cases"][c]["truth"]), refine, start])
end

"The figure's product: keyed by [figure], its detections inputs and the keys of the evaluations it reads (`detector_bias_figures_product`)."
figures_product() = detector_bias_figures_product(CFG)

"The directory of the evaluate product of case `c`, produced on demand (its stage evaluate)."
evaluate_dir(c) = produce_upstream(detector_bias_evaluate_product(CFG, c), "the detector-bias figure: the evaluation of case $c"; make = () -> evaluate_case(c),
                                   command = driver_run(DRIVER, CFG_PATH, "evaluate"; case = c).command)

"""Stage figures: the figure's tables and files by its drawing code (`write_detector_bias_figure`; the files name this stage as
their generator), kept in the product (`tables/`, `assets/`) and published to plotsdir() (`produce_figures`); numbers.tex, the
output, written last. What it reads is declared to the product's provenance: the rows' evaluations, the inputs of stage figures
(the sets' detections) and the rows' renders."""
function figures(p)
    cases = detector_bias_figure_cases(CFG)
    inputs = [[product_dir(detector_bias_evaluate_product(CFG, c)) for c in cases]; require_stage_inputs(CFG, "figures");
              [detector_bias_render(CFG, r) for r in CFG["figure"]["order"]]]
    return produce_figures(p, String[]; R..., inputs) do dir
        written = write_detector_bias_figure(CFG, dir, Dict{String, String}(c => evaluate_dir(c) for c in cases);
                                             generator = "experiments/$DRIVER (stage figures; T. Harder's detector_bias.py)")
        foreach(f -> println(">> figures → ", f), written)
        Reproduce.write_numbers(joinpath(dir, "numbers.tex"), driver_numbers(CFG_PATH, DRIVER, p.key))   # no number: an empty fragment
    end
end

# The stages that run the GPU: start and refine hand each case's [pipeline] to bin/refine.jl (its compute.backend).
const GPU_STAGES = gpu_stages(:start => [CFG["cases"][c] for c in CASES], :refine => [CFG["cases"][c] for c in CASES])

const STAGES = [
    Stage(:detect, "the renders, their truth, the detectors' outputs, the geometric calibrations and the boards of record (tiers rc_synthetic, rc_baselines, rc_calibration)"; check = detect),
    Stage(:start, "the start of every case (bin/refine.jl --stage=start: its start record, keyed without the refinement-only options)";
          items = () -> CASES, product = c -> start_product(CFG, c),
          run = c -> start_with_refine(CFG, c) || run_start(CFG, c; R...)),   # with its refinement to run: one child, the refine stage's
    Stage(:refine, "the radiometric refinement of every case from its start (bin/refine.jl --start)";
          items = () -> CASES, product = c -> refine_product(CFG, c), run = c -> run_refine(CFG, c; R...)),
    Stage(:evaluate, "per case, at its start and its last checkpoint: the errors against the truth under the run's own and under the true poses";
          items = () -> CASES, product = c -> detector_bias_evaluate_product(CFG, c), run = evaluate_case),
    Stage(:figures, "the paper's figure fig:detector-bias (plotsdir(<asset_dir>.tex, <asset_dir>/, <asset_dir>-caption.tex)), its tables, numbers.tex";
          items = () -> [figures_product()], product = identity, run = figures),
]

run_stages("detector_bias", STAGES)
