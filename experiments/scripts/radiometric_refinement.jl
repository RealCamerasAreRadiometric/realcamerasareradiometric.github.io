#!/usr/bin/env julia
# reproduce: reads configs/experiments/radiometric_refinement.toml
# reproduce: reads configs/experiments/radiometric_refinement_vignetting.toml
# reproduce: reads configs/experiments/radiometric_refinement_board_warping.toml
# provenance: every product writes _config.toml and _provenance.toml (VisualGeometryToolkitEval produce) beside its results.
# =============================================================================
# scripts/radiometric_refinement.jl — the radiometric refinements of the paper's real photographs: per case a geometric start model
# refined on the board photographs by bin/refine.jl. Configs of record: configs/experiments/radiometric_refinement.toml (the geometric-
# accuracy experiment: tab:refinement-chrome, tab:defocus) and configs/experiments/radiometric_refinement_vignetting.toml
# (the large-aperture refinements; fig:focus-surfaces reads the fisheye's) and configs/experiments/radiometric_refinement_board_warping.toml
# (the board-warping session's six sets: the supplement's board warping across lenses). Every case refines the
# printed board's in-plane print error (aspect and skew) beside its heights (owner ruling 2026-11-13).
# The refined models of the first are scored on the Thorlabs R3L3S7N1 chrome board by scripts/chrome_evaluation.jl (one config
# per chrome set whose [evaluate] models name these cases: configs/experiments/chrome_evaluation_fx3_<set>.toml).
#
# STAGES (each product keyed by only what it reads and the keys of the products it reads; <results>/<experiment>/<stage>/<key>/)
#   prepare   the board photographs decoded: Features' load_raw (LibRaw_jll) per [prepare.<set>]
#   detect    the detections, geometric calibrations and start fits of record (tier rc_baselines), present
#   start     per case, its start (bin/refine.jl --stage=start): keyed by every setting but the refinement-only ones
#             (RadiometricCamera.REFINE_ONLY_KEYS), so a refine-only change loads it
#   refine    per case, the radiometric refinement from its start (bin/refine.jl --start, a checkpoint per iteration); keyed by the
#             refinement-only settings and the start's key; refused while the config has [pending]
#
#   julia --project=experiments experiments/scripts/radiometric_refinement.jl [--config=<toml>] [--plan] [--stage=NAME[,NAME]] [--case=a,b] [--force]
#
# --case restricts refine to the named cases. Data: VGTK_DATA_ROOT binds data_root() (REPRODUCE.md); the raw photographs are the
# tier rc_raw_photos, the start fits and Schöps's runs rc_baselines, the boards rc_calibration (configs/data_artifacts.toml).
# =============================================================================
using DrWatson
@quickactivate "experiments"
include(srcdir("experiments.jl"))
include(srcdir("drivers.jl"))
include(srcdir("radiometric_refinement.jl"))

const DRIVER = "scripts/radiometric_refinement.jl"
const CFG_PATH = abspath(cli_option(ARGS, "config", projectdir("configs", "experiments", "radiometric_refinement.toml")))   # default: the config of record
const CFG = load_radiometric_refinement_config(CFG_PATH)
const CASES = case_names(CFG)
const SETS = sort(collect(keys(CFG["prepare"])))

detect() = (require_stage_inputs(CFG, "detect"); nothing)

const R = (; driver = DRIVER, config_path = CFG_PATH)

# The stages that run the GPU: start (the starts) and refine hand the cases' [pipeline] tables to bin/refine.jl (their compute.backend).
const GPU_STAGES = gpu_stages(:start => [CFG["cases"][c] for c in CASES], :refine => [CFG["cases"][c] for c in CASES])

const STAGES = [
    Stage(:prepare, "the board photographs decoded (Features' load_raw)";
          items = () -> SETS, product = s -> prepare_product(CFG, s), run = s -> run_prepare(CFG, s; R...)),
    Stage(:detect, "the detections, calibrations and start fits of record (tier rc_baselines)"; check = detect),
    Stage(:start, "the start of every case (bin/refine.jl --stage=start: its start record, keyed without the refinement-only options)";
          items = () -> CASES, product = c -> start_product(CFG, c),
          run = c -> start_with_refine(CFG, c) || run_start(CFG, c; R...)),   # with its refinement to run: one child, the refine stage's
    Stage(:refine, "the radiometric refinement of every case from its start (bin/refine.jl --start)";
          items = () -> CASES, product = c -> refine_product(CFG, c),
          run = c -> (produced(refine_product(CFG, c)) || refuse_pending(CFG, CFG_PATH); run_refine(CFG, c; R...))),   # [pending] blocks new work only
]

run_stages("radiometric_refinement", STAGES)
