#!/usr/bin/env julia
# =============================================================================
# scripts/material_grid.jl — the child process of stage material_grid of scripts/radiometric_refinement.jl (src/material_grid.jl,
# `run_material_grid_search`): the material start of ONE refinement by the grid search. It rebuilds the case's start from its start
# record (the start's own options, masks and photographs: nothing of the start is computed again), runs the start's closed-form
# solves at every candidate (α_ink, α_¬ink, η_ink, η_¬ink) of the config's [material_grid] — the lighting with V ≡ 1, then one
# sweep of the lighting, the vignetting and μ — keeps the candidate of the lowest cost, and writes into the [pipeline]'s io.out
#   material_grid.json   the grid, the start's own material and costs, every candidate's costs, the chosen candidate
#   a start record       (start.json, state.json, grid.json | lens.json, frontend/, masks/, pipeline_config.toml) of the chosen
#                        material: what bin/refine.jl --stage=start writes for it, and what bin/refine.jl --start refines from
# It replaces the study this file held before (per candidate a scan of μ and an alternation of μ and the lighting with V flat): the
# start's solves themselves are what a candidate is judged by.
#
#   julia --project=experiments experiments/scripts/material_grid.jl --config=<input.toml> --start=<start record>
#     --config  the stage's input, written by the driver: [pipeline] (the case's start settings, io.out the product's directory)
#               and [material_grid] (the config of record's)
#     --start   the case's start record (bin/refine.jl --stage=start)
# Run in data_root(), the directory the [pipeline]'s relative input paths are relative to (the driver does). The driver's stage
# keys the product and writes its records (_config.toml, _provenance.toml: VisualGeometryToolkitEval's produce) beside these files.
# =============================================================================
using DrWatson
@quickactivate "experiments"
include(srcdir("experiments.jl"))
include(srcdir("drivers.jl"))

const USAGE = "usage: experiments/scripts/material_grid.jl --config=<input.toml> --start=<start record>"
const INPUT = cli_option(ARGS, "config", "")
const START = cli_option(ARGS, "start", "")
(isempty(INPUT) || isempty(START)) && throw(ArgumentError("--config and --start are required; $USAGE"))
const TABLES = TOML.parsefile(INPUT)
const GRID = check_material_grid(get(TABLES, "material_grid", nothing), "$INPUT [material_grid]")
const OPTIONS = pipeline_options(get(TABLES, "pipeline", Dict{String, Any}()))
isempty(OPTIONS.io.out) && throw(ArgumentError("$INPUT [pipeline]: io.out is required (the stage's product)"))
abspath(START) == abspath(OPTIONS.io.out) &&
    throw(ArgumentError("--start=$START is the [pipeline]'s io.out: the material start is written beside the start it is searched from, never into it"))
# the device, resolved at top level as bin/refine.jl does (RadiometricCamera's one resolver): the calls below run in the world the
# vendor package's load made
const DEVICE = RadiometricCamera.rc_backend(OPTIONS.compute.backend)

t0 = time()
state = RadiometricCamera.load_calibration_start(START; options = OPTIONS, device = DEVICE)
println("start loaded in ", round(time() - t0; digits = 1), " s"); flush(stdout)
t1 = time()
result = material_grid_search!(state, OPTIONS, GRID)
println("material grid searched in ", round(time() - t1; digits = 1), " s (", length(result.candidates), " candidates)"); flush(stdout)
write_material_grid(OPTIONS.io.out, GRID, result)
RadiometricCamera.save_calibration_start(OPTIONS.io.out, state; options = material_start_options(OPTIONS, result.candidates[result.chosen]))
println("MATERIAL START written to ", OPTIONS.io.out, " in ", round(time() - t0; digits = 1), " s"); flush(stdout)
