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
#   material_grid  per case, its material start (scripts/material_grid.jl, src/material_grid.jl): the start's closed-form solves at
#             every candidate (α_ink, α_¬ink, η_ink, η_¬ink) of the config's [material_grid], the candidate of the lowest cost kept
#             as a start record; keyed by [material_grid] and the start's key
#   material_grid_part  only with --grid-parts=n: per case and part, part k of n of the case's candidates (every n-th from the k-th),
#             their rows; keyed as material_grid and by the part. Stage material_grid then reads the n parts' rows and solves at the
#             chosen candidate alone: the same product under the same key (declared before it, so one run may do both)
#   refine    per case, the radiometric refinement from its material start (bin/refine.jl --start, a checkpoint per iteration);
#             keyed by the refinement-only settings and the material grid's key; refused while the config has [pending]
#
#   julia --project=experiments experiments/scripts/radiometric_refinement.jl [--config=<toml>] [--plan] [--stage=NAME[,NAME]] [--case=a,b] [--force]
#         [--grid-parts=n [--shard=k/m]]
#
# --case restricts start, material_grid_part, material_grid and refine to the named cases. --grid-parts=n splits every case's
# candidates of the material grid over n GPUs, which shortens the wait for a case and changes no result: after the case's start
# (`--stage=start`: jobs that run at once would each produce a missing one), n jobs `--stage=material_grid_part --grid-parts=n
# --shard=k/n` (k = 0 … n-1: --shard splits the stage's items, a case's part each, between the jobs), then
# `--stage=material_grid,refine --grid-parts=n` (a part not produced by then is produced there). Data: VGTK_DATA_ROOT binds data_root() (REPRODUCE.md); the raw photographs are the
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

# The stages that run the GPU: start (the starts) and refine hand the cases' [pipeline] tables to bin/refine.jl, material_grid_part
# and material_grid to scripts/material_grid.jl (their compute.backend).
const GPU_STAGES = gpu_stages(:start => [CFG["cases"][c] for c in CASES], :material_grid_part => [CFG["cases"][c] for c in CASES],
                              :material_grid => [CFG["cases"][c] for c in CASES], :refine => [CFG["cases"][c] for c in CASES])
# --grid-parts=n: the parts every case's candidates of the material grid are split into (src/material_grid.jl); none without it
const GRID_PARTS = material_grid_parts()

const STAGES = [
    Stage(:prepare, "the board photographs decoded (Features' load_raw)";
          items = () -> SETS, product = s -> prepare_product(CFG, s), run = s -> run_prepare(CFG, s; R...)),
    Stage(:detect, "the detections, calibrations and start fits of record (tier rc_baselines)"; check = detect),
    Stage(:start, "the start of every case (bin/refine.jl --stage=start: its start record, keyed without the refinement-only options)";
          items = () -> CASES, product = c -> start_product(CFG, c), run = c -> run_start(CFG, c; R...)),
    Stage(:material_grid_part, "with --grid-parts=n, the candidates of every case's material grid in n parts (their rows; --shard=k/m: a job's share of them)";
          items = () -> cli_shard(ARGS, [(c, k) for c in CASES for k in 1:something(GRID_PARTS, 0)]),
          product = x -> material_grid_part_product(CFG, x..., GRID_PARTS), run = x -> run_material_grid_part(CFG, x..., GRID_PARTS; R...)),
    Stage(:material_grid, "the material start of every case (scripts/material_grid.jl: the start's closed-form solves at every candidate of [material_grid], the lowest cost kept)";
          items = () -> CASES, product = c -> material_grid_product(CFG, c), run = c -> run_material_grid(CFG, c; R...)),
    Stage(:refine, "the radiometric refinement of every case from its material start (bin/refine.jl --start)";
          items = () -> CASES, product = c -> refine_product(CFG, c),
          run = c -> (produced(refine_product(CFG, c)) || refuse_pending(CFG, CFG_PATH); run_refine(CFG, c; R...))),   # [pending] blocks new work only
]

run_stages("radiometric_refinement", STAGES)
